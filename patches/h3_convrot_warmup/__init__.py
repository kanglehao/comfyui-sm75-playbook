# -*- coding: utf-8 -*-
"""H3 convrot dequant 加固钩子 —— 温启动 + 有界重试 / update-proof, lives OUTSIDE site-packages.

WHY THIS FILE EXISTS
--------------------
症状（MiniMax-H3 int8_convrot 工作流跑到中途炸）：

    RuntimeError: CUDA INT8 convrot dequantization failed: invalid argument

raised from `SamplerCustomAdvanced` while dequantizing MiniMax-H3
`*_int8_convrot.safetensors` weights.

第一版结论（2026-09-19，v0.36.0）说是「进程内第 1 次调用失败一次，之后都好」，
所以只做了温启动（warm-up）。**2026-09-21 在 v0.37.0 上复测，这个结论被推翻：**

    13:17:10  H3WARM cuda:0 out=float32/float16/bfloat16 attempt=1 OK   <- 温启动成功
    13:17:10  H3TRACE #1 .. #56                                         <- 之后一路正常
    13:18:26  H3DBG dump_no=1  call_no=409  -> invalid argument         <- 挂在第 409 次调用
              bisect: D1-retry-same: OK   D2-fresh-clone: OK
                      D3-cuda0: OK        D4-default-stream: OK
    13:18:42  H3DBG dump_no=2  call_no=410  -> 同一个错（工作流自己重试那次）

    另一轮：call_no=1101 / 1102 同样报错，D1..D4 依然全 OK。

=> 失败**不是冷路径**（冷路径已被前面的温启动吃掉），而是**跑到中途的偶发失败**；
   而且「原样重试同一次调用」必定成功。所有静态参数（形状/步长/dtype/连续性/16 字节对齐
   /device/stream/capture 状态）在 dump 里都证明合法，`peek_last=0 capture=0 drv_ctx_dev=0`。

   最省事且被证据直接支持的修法：**在真实调用路径上加有界重试**（重试前先
   同步一次，把驱动侧残留在错误槽里的上一次异步错误清掉），而不是只在开头温启动。
   失败时把每一次尝试都写进日志，绝不静默吞掉。

WHY IT IS A CUSTOM NODE AND NOT A PATCH INSIDE THE PACKAGE
---------------------------------------------------------
`custom_nodes/` 不属于 pip、也不会被 ComfyUI 仓库的 git pull 覆盖，所以这里放的加固
在更新（含启动器「安装/修复依赖」触发 pip 重装包）之后依然存活。
`site-packages` 里的诊断补丁随时可能被重装抹掉，不能当长期方案。

CONTRACT
--------
* 永不因为本文件导致 ComfyUI 启动失败；
* 只在「原生 convrot 反量化内核」这条路径上做事，其它异常一律原样抛出；
* 重试次数有上限（RETRY_ATTEMPTS），用尽则抛最后一次异常，绝不无限吞错；
* 正常无故障时行为与不打补丁完全一致；
* 上游改名/内部结构变化时打印 SKIP 行并保持惰性。
"""

import os
import sys
import json
import time
import traceback

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(NODE_DIR, "h3_warmup.log")
STATUS_PATH = os.path.join(NODE_DIR, "h3_warmup_status.json")
VERSION_TAG = "v2-retry"  # v1 = 仅温启动；v2 = 温启动 + 有界重试

BACKEND_MODULE = "comfy_kitchen.backends.cuda"
FUNC_NAME = "dequantize_int8_convrot_weight_dtype"
RAW_OP_NAME = "dequantize_int8_convrot_weight"

# The kernel is templated on (group_size, output dtype). group_size is pinned to
# 256 by the call-site gate, so only the output dtype varies across a real run.
WARM_DTYPES = ("float32", "float16", "bfloat16")
WARM_SHAPE = (256, 256)   # k=256 -> passes `k % 256 == 0`, and is tiny

# 中途偶发失败的有界重试。依据 2026-09-21 的 bisect：D1-retry-same 必定成功。
RETRY_ATTEMPTS = 3
# 只有这些字样才认定是「可重试的偶发原生失败」，其余异常一律不碰。
TRANSIENT_MARKERS = ("convrot dequantization failed", "invalid argument")


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
}


def _log(msg):
    line = "%s [h3_warmup] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
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


def _needs_native_kernel(mod, q, group_size):
    """Only warm up when the call will actually reach the native kernel."""
    probe = getattr(mod, "_should_use_convrot_dequant_kernel", None)
    try:
        if callable(probe):
            return bool(probe(q, q.shape[-1], group_size))
    except Exception:
        pass
    try:  # fallback: mirror the documented gate (gs==256, k%256==0, k<=16384)
        k = int(q.shape[-1])
        return int(group_size) == 256 and k % 256 == 0 and k <= 16384
    except Exception:
        return False


def _warm(mod, device):
    """Burn the kernel's one-shot cold path on throwaway tensors. Never raises."""
    try:
        import torch
    except Exception as exc:
        _log("warm SKIP: torch import failed %r" % (exc,))
        return

    raw = getattr(mod, "_C", None)
    wrap = getattr(mod, "_wrap_for_dlpack", None)
    d2c = getattr(mod, "DTYPE_TO_CODE", None)
    if raw is None or wrap is None or not isinstance(d2c, dict):
        _log("warm SKIP %s: package internals changed "
             "(_C=%s wrap=%s DTYPE_TO_CODE=%s)"
             % (device, raw is not None, wrap is not None, isinstance(d2c, dict)))
        return

    try:
        previous_device = torch.cuda.current_device()
    except Exception:
        previous_device = None

    # `_wrap_for_dlpack` refuses cross-device export while the process-global
    # current device points elsewhere (BufferError). H3-MultiGPU intentionally
    # parks the current device on another index, so pin it for the duration of
    # the warm-up and put it back exactly as we found it.
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

        for name in WARM_DTYPES:
            dtype = getattr(torch, name, None)
            if dtype is None or dtype not in d2c:
                continue
            try:
                q_dummy = torch.empty(WARM_SHAPE, dtype=torch.int8, device=device)
                s_dummy = torch.ones((1,), dtype=torch.float32, device=device)
                o_dummy = torch.empty(WARM_SHAPE, dtype=dtype, device=device)
            except Exception as exc:
                _log("warm %s out=%s alloc FAIL %r" % (device, name, exc))
                continue
            for attempt in (1, 2):
                try:
                    raw.dequantize_int8_convrot_weight(
                        wrap(q_dummy), wrap(s_dummy), wrap(o_dummy),
                        WARM_SHAPE[1], stream_ptr,
                    )
                    torch.cuda.synchronize(device)
                    _log("warm %s out=%s attempt=%d OK (cold path consumed)"
                         % (device, name, attempt))
                    ok_dtypes.append(name)
                    break
                except Exception as exc:
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        pass
                    _log("warm %s out=%s attempt=%d FAIL %r"
                         % (device, name, attempt, str(exc)[:180]))
            try:
                del q_dummy, s_dummy, o_dummy
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


def _call_with_retry(real, args, attempts=RETRY_ATTEMPTS, device=None, on_note=None):
    """Invoke ``real(*args)`` with a bounded retry on the known transient error.

    ``on_note(kind, msg)`` is called with kind ``"fail"`` on every failed
    attempt and ``"recover"`` when a later attempt succeeds. The function
    returns ``(result, attempt)``; it raises the original exception untouched
    for non-transient errors and the last exception once attempts are used up.

    Kept as a plain function so it can be unit-tested without CUDA.
    """
    def _note(kind, msg):
        if on_note is not None:
            try:
                on_note(kind, msg)
            except Exception:
                pass

    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            out = real(*args)
        except Exception as exc:
            if not _is_transient(exc):
                raise
            _note("fail", "attempt %d/%d failed: %s" % (attempt, attempts, str(exc)[:180]))
            if attempt < attempts:
                _drain_device(device)
                continue
            raise
        if attempt > 1:
            _note("recover", "recovered on attempt %d/%d (transient native failure)" % (attempt, attempts))
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
    if getattr(real, "_h3_warm_wrapper", False):
        _state["wrapped"] = True
        _state["func_found"] = True
        _status()
        _log("already installed (wrapper present)")
        return

    _state["func_found"] = True

    def wrapper(q, scale, group_size, output_dtype_code, _real=real):
        try:
            _state["real_calls"] += 1
            key = str(q.device)
            if key not in _state["warm_done_devices"]:
                # NOTE: 只有真正走了温启动才把设备记成 done。第一版是先记后判，
                # 若某设备的首次调用并不走原生内核，该设备就永远不会被温启动。
                if _needs_native_kernel(backend, q, group_size):
                    _state["warm_done_devices"].append(key)
                    _warm(backend, q.device)
                else:
                    _log("skip warm %s: call will not use the native kernel" % key)
        except Exception as exc:
            _log("warm hook error (ignored): %r" % (exc,))

        def _note(kind, msg):
            if kind == "fail":
                _state["retry_events"] += 1
                _state["last_retry_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    _state["last_retry_shape"] = list(q.shape)
                except Exception:
                    _state["last_retry_shape"] = None
            _log("convrot %s | shape=%s gs=%s out_code=%s | %s"
                 % ("RETRY" if kind == "fail" else "RETRY-OK",
                    getattr(q, "shape", "?"), group_size, output_dtype_code, msg))
            _status()

        try:
            out, attempt = _call_with_retry(
                _real, (q, scale, group_size, output_dtype_code),
                device=getattr(q, "device", None), on_note=_note,
            )
        except Exception:
            # 重试用尽仍失败：计数后照常抛出，绝不静默吞错。
            _state["retry_gave_up"] += 1
            _status()
            raise
        if attempt > 1:
            _state["retry_recovered"] += 1
            _status()
        return out

    wrapper._h3_warm_wrapper = True
    wrapper._h3_warm_version = VERSION_TAG
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
