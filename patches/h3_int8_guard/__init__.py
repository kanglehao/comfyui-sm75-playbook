# -*- coding: utf-8 -*-
"""H3 int8 rowwise 量化加固钩子 —— 温启动 + 有界重试 / update-proof, lives OUTSIDE site-packages.

WHY THIS FILE EXISTS
--------------------
症状（2026-10-08，MiniMax-H3 `fastvideo_fasth3_8step_v2_pruned_int8_convrot` 图生视频）：
在**采样第一步**就整条工作流报错：

    RuntimeError: CUDA INT8 rowwise quantization failed: invalid argument
    Node Type: SamplerCustomAdvanced

调用链（实测 traceback）：

    comfy/ldm/minimax/model.py:521 preprocess_text_embeds
      -> token_refiner(...)                       <- 条件预处理，采样最开始
      -> comfy/ops.py:1489 forward -> F.linear
      -> comfy_kitchen tensor/int8.py:287 _handle_int8_linear_tensorwise
      -> comfy_kitchen/backends/cuda/__init__.py:2452 int8_linear
           x_qdata, x_scale = quantize_and_rotate_rowwise(x_2d, h, group_size)
      -> ...:1953 quantize_int8_rowwise(x_rot)
      -> ...:1044 _C.quantize_int8_rowwise(...)   <- 原生 CUDA 内核
      => cudaErrorInvalidValue

为什么偏偏是这个模型（同机 + 同包的其它 H3 int8 模型跑得动）
-----------------------------------------------------------
1) H3 的 `token_refiner` 权重是 `qkv_proj [21504, 5376] int8`，即 k=5376。
2) Turing(sm_75) 的共享内存上限是 `shared_memory_per_block = 49152`（48KB，
   `_max_dynamic_shared_memory_per_block()` 在 Turing 上取的是非 opt-in 值）。
   k=5376 走 fusion 判定时 `(5376 + 16*2*256)*4 = 54272 > 49152` -> 装不下 ->
   `_fused_convrot_ok = False`；同时 `_should_use_convrot_fused_kernel` 要求
   `k <= 5120 or k >= 8192`，5376 正好落在被**故意跳过**的 5120<k<8192 带里
   （源码注释：这条带子 fused 打不过 rotate-matmul 回退路径）。
   => 必然走 `__init__.py:2452` 那条**回退路径**，即调用裸的 `quantize_int8_rowwise`。
3) 本机所有 H3 int8 模型里，**只有这个 8step v2 模型的 token_refiner 被量化**
   （其余模型的 token_refiner 都是 BF16，blocks 才是 I8）——所以只有它会
   在「采样的第一次 int8 线性层」就撞上这个原生算子。
4) 这个原生算子（以及同族的 `dequantize_int8_convrot_weight`）在本机有**已知的偶发
   `invalid argument`**：所有静态参数（shape/dtype/连续性/16B 对齐/device/stream/capture）
   都被 dump 证明合法，而**同一份入参原样重发必定成功**（历史上观测到 call_no=409/1101）。
   `_H3NativeProxy` 的记录器当天也打了这一行：
       H3CRECORD native op quantize_int8_rowwise failed 'invalid argument' on a cold
       path; if this kills the workflow it needs its own warm-up
   —— 也就是说：**这个算子的加固还没做**。姊妹算子（convrot dequant）早在
   2026-09-21 就由 `custom_nodes/h3_convrot_warmup/` 做了「温启动 + 有界重试」。

所以本文件就是给 `quantize_int8_rowwise` 补上同一套加固。

WHY IT IS A CUSTOM NODE AND NOT A PATCH INSIDE THE PACKAGE
---------------------------------------------------------
`custom_nodes/` 不属于 pip、也不会被 ComfyUI 仓库的 git pull 覆盖，所以这里放的加固
在更新（含启动器「安装/修复依赖」触发 pip 重装包）之后依然存活。
`site-packages` 里的诊断补丁随时可能被重装抹掉，不能当长期方案。

CONTRACT
--------
* 永不因为本文件导致 ComfyUI 启动失败；
* **语义透明**：`quantize_int8_rowwise(x, *a, **k)` 的返回值原样透传，不改数值；
* 只在命中「可重试的偶发原生失败」字样时重试，其它异常一次都不重试、原样抛出；
* 重试次数有上限（RETRY_ATTEMPTS），用尽则抛最后一次异常，绝不无限吞错；
* 正常无故障时行为与不打补丁完全一致（无日志、attempt==1）；
* 上游改名/内部结构变化时打印 SKIP 行并保持惰性。
"""

import json
import os
import sys
import time
import traceback

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(NODE_DIR, "h3_int8_guard.log")
STATUS_PATH = os.path.join(NODE_DIR, "h3_int8_guard_status.json")
VERSION_TAG = "v1-rowwise"

BACKEND_MODULE = "comfy_kitchen.backends.cuda"
FUNC_NAME = "quantize_int8_rowwise"
RAW_OP_NAME = "quantize_int8_rowwise"

# 输入是激活值，模板维度是**输入 dtype**（fp32/fp16/bf16 各一份实例）。
WARM_DTYPES = ("float32", "float16", "bfloat16")
# 5376 = MiniMax-H3 的隐藏维度（k % 256 == 0），和真实崩溃时同一量级。
WARM_SHAPE = (256, 5376)

# 中途偶发失败的有界重试。依据：同族算子 D1-retry-same 必定成功。
RETRY_ATTEMPTS = 3
# 只有这些字样才认定是「可重试的偶发原生失败」，其余异常一律不碰。
TRANSIENT_MARKERS = ("rowwise quantization failed", "invalid argument")


_state = {
    "tag": VERSION_TAG,
    "wrapped": False,
    "func_found": False,
    "real_calls": 0,
    "warm_runs": 0,
    "warm_done_devices": [],
    "retry_events": 0,       # 触发过重试的调用次数
    "retry_recovered": 0,    # 重试后成功的次数
    "retry_gave_up": 0,      # 重试用尽仍失败、最终抛出的次数
    "last_retry_at": None,
    "last_retry_shape": None,
    "last_retry_dtype": None,
}


def _log(msg):
    line = "%s [h3_int8_guard] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _status(extra=None):
    data = dict(_state)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["python"] = sys.version.split()[0]
    data["pid"] = os.getpid()
    data["retry_attempts_cfg"] = RETRY_ATTEMPTS
    if extra:
        data.update(extra)
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _warm(mod, device):
    """Burn the kernel's one-shot cold path on throwaway tensors. Never raises."""
    try:
        import torch
    except Exception as exc:
        _log("warm SKIP: torch import failed %r" % (exc,))
        return

    raw = getattr(mod, "_C", None)
    wrap = getattr(mod, "_wrap_for_dlpack", None)
    if raw is None or wrap is None:
        _log("warm SKIP %s: package internals changed (_C=%s wrap=%s)"
             % (device, raw is not None, wrap is not None))
        return

    try:
        previous_device = torch.cuda.current_device()
    except Exception:
        previous_device = None

    # `_wrap_for_dlpack` refuses cross-device export while the process-global
    # current device points elsewhere (BufferError). Multi-GPU nodes park the
    # current device on another index, so pin it for the warm-up and restore.
    try:
        torch.cuda.set_device(device)
    except Exception:
        pass

    ok_dtypes = []
    try:
        try:
            stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
        except Exception as exc:
            _log("warm SKIP %s: stream lookup failed %r" % (device, exc))
            return

        rows, k = WARM_SHAPE
        for name in WARM_DTYPES:
            dtype = getattr(torch, name, None)
            if dtype is None:
                continue
            try:
                x_dummy = torch.zeros((rows, k), dtype=dtype, device=device)
                q_dummy = torch.empty((rows, k), dtype=torch.int8, device=device)
                s_dummy = torch.empty((rows, 1), dtype=torch.float32, device=device)
            except Exception as exc:
                _log("warm %s in=%s alloc FAIL %r" % (device, name, exc))
                continue
            for attempt in (1, 2):
                try:
                    raw.quantize_int8_rowwise(
                        wrap(x_dummy), wrap(q_dummy), wrap(s_dummy),
                        False, 0, stream_ptr,
                    )
                    torch.cuda.synchronize(device)
                    _log("warm %s in=%s attempt=%d OK (cold path consumed)"
                         % (device, name, attempt))
                    ok_dtypes.append(name)
                    break
                except Exception as exc:
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        pass
                    _log("warm %s in=%s attempt=%d FAIL %r"
                         % (device, name, attempt, str(exc)[:180]))
            try:
                del x_dummy, q_dummy, s_dummy
            except Exception:
                pass
    finally:
        if previous_device is not None:
            try:
                torch.cuda.set_device(previous_device)
            except Exception:
                pass

    _state["warm_runs"] += 1
    _status({"last_warm_device": str(device), "last_warm_ok_dtypes": ok_dtypes})


def _is_transient(exc):
    """True when the exception looks like the known transient native failure."""
    try:
        msg = str(exc)
    except Exception:
        return False
    return any(m in msg for m in TRANSIENT_MARKERS)


def _drain_device(device):
    """Best-effort sync to clear the driver-side stale error slot before a retry.

    Never raises: a sync that fails is itself a stale-error readout, so we try
    once more and swallow whatever is left.
    """
    try:
        import torch
    except Exception:
        return
    for _ in (1, 2):
        try:
            torch.cuda.synchronize(device)
            return
        except Exception as exc:
            _log("sync before retry raised (ignored): %r" % (exc,))


def _call_with_retry(real, args, kwargs=None, attempts=RETRY_ATTEMPTS, device=None, on_note=None):
    """Invoke ``real(*args, **kwargs)`` with a bounded retry on the known transient error.

    ``on_note(kind, msg)`` is called with kind ``"fail"`` on every failed attempt
    and ``"recover"`` when a later attempt succeeds. Returns ``(result, attempt)``;
    raises the original exception untouched for non-transient errors and the last
    exception once attempts are used up. Kept as a plain function so it can be
    unit-tested without CUDA.
    """
    kwargs = kwargs or {}

    def _note(kind, msg):
        if on_note is not None:
            try:
                on_note(kind, msg)
            except Exception:
                pass

    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            out = real(*args, **kwargs)
        except Exception as exc:
            if not _is_transient(exc):
                raise
            _note("fail", "attempt %d/%d failed: %s" % (attempt, attempts, str(exc)[:180]))
            if attempt < attempts:
                _drain_device(device)
                continue
            raise
        if attempt > 1:
            _note("recover", "recovered on attempt %d/%d (transient native failure)"
                  % (attempt, attempts))
        return out, attempt


def _install():
    import torch

    if not torch.cuda.is_available():
        _log("SKIP: torch.cuda.is_available() is False")
        return

    try:
        from comfy_kitchen.backends import cuda as backend
    except Exception as exc:
        _log("SKIP: cannot import %s -- %r" % (BACKEND_MODULE, exc))
        return

    real = getattr(backend, FUNC_NAME, None)
    if real is None:
        _log("SKIP: %s has no attribute %s (upstream renamed it?)"
             % (BACKEND_MODULE, FUNC_NAME))
        return
    if getattr(real, "_h3_rw_wrapper", False):
        _state["wrapped"] = True
        _state["func_found"] = True
        _status()
        _log("already installed (wrapper present)")
        return

    _state["func_found"] = True

    def wrapper(x, *args, **kwargs):
        _real = wrapper._h3_rw_real
        _state["real_calls"] += 1
        try:
            key = str(x.device)
            if key not in _state["warm_done_devices"]:
                # 只有真正走温启动才把设备记成 done（先记后判会让设备永远不被预热）。
                _state["warm_done_devices"].append(key)
                _warm(backend, x.device)
        except Exception as exc:
            _log("warm hook error (ignored): %r" % (exc,))

        def _note(kind, msg):
            if kind == "fail":
                _state["retry_events"] += 1
                _state["last_retry_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    _state["last_retry_shape"] = list(x.shape)
                    _state["last_retry_dtype"] = str(x.dtype)
                except Exception:
                    _state["last_retry_shape"] = None
                    _state["last_retry_dtype"] = None
            _log("rowwise %s | shape=%s dtype=%s dev=%s | %s"
                 % ("RETRY" if kind == "fail" else "RETRY-OK",
                    getattr(x, "shape", "?"), getattr(x, "dtype", "?"),
                    getattr(x, "device", "?"), msg))
            _status()

        try:
            out, attempt = _call_with_retry(
                _real, (x,) + args, kwargs,
                device=getattr(x, "device", None), on_note=_note,
            )
        except Exception as exc:
            # 只有「命中可重试字样、且重试用尽」才算 gave_up；
            # 其它异常原样穿过，不能污染计数（否则会把普通报错误读成「重试也治不好」）。
            if _is_transient(exc):
                _state["retry_gave_up"] += 1
                _status()
            raise
        if attempt > 1:
            _state["retry_recovered"] += 1
            _status()
        return out

    wrapper._h3_rw_wrapper = True
    wrapper._h3_rw_version = VERSION_TAG
    wrapper._h3_rw_real = real
    wrapper.__wrapped__ = real
    wrapper.__name__ = getattr(real, "__name__", FUNC_NAME)
    wrapper.__doc__ = getattr(real, "__doc__", None)

    setattr(backend, FUNC_NAME, wrapper)
    _state["wrapped"] = True
    _status()
    _log("installed (%s): %s.%s is now warm-up + bounded-retry wrapper (attempts=%d)"
         % (VERSION_TAG, BACKEND_MODULE, FUNC_NAME, RETRY_ATTEMPTS))


try:
    _install()
except Exception:
    _log("install FAILED (ComfyUI startup unaffected): "
         + traceback.format_exc()[-900:])


# This node registers no UI nodes; it exists purely for the import-time hook.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
