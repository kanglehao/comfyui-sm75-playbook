# 03b · SageAttention 2 在 sm75 上到底跑什么

> 一句话：**Triton 在 cc < 8 上一条张量核指令都不生成**，所以 SageAttention 2 的官方实现
> 在 Turing 上是**结构性不存在**，不是"没配好"。有人能让它跑起来，靠的是绕开 Triton 的补丁版；
> 但即便如此，2080 Ti 上也不值得用它。

---

## 结论先行

| 问题 | 结论 |
|---|---|
| 官方 PyPI 的 sageattention 2.x 能在 sm75 上直接用吗 | **不能**。Triton 路径在 cc<8 上退化到 CUDA core |
| 那为什么有人能跑 | 用的是**绕开 Triton 的定制包**（本机的 2.1.2 就是），走预编译 CUDA 扩展 |
| 跑起来快吗 | **能跑，但不如 sm75 专用 CUTLASS 核**，且维护成本高 |
| 和 CUTLASS 节点能叠加吗 | **不能，前者会被短路**。见"优先级"一节 |
| 那到底选哪个 | 2080 Ti 选 CUTLASS 节点；sage2 留给 30/40/50 系 |

---

## 一、为什么官方版在 sm75 上不行

SageAttention 2 的核心是：QK 做 INT8 量化 → INT8 矩阵乘 → PV 用 FP16。
它的 kernel 用 Triton 写，靠 Triton 把代码编到目标架构的张量核指令上。

问题在于 **Triton 的后端在 `cc < 8` 上根本不发张量核指令**：

- 实测 Triton 3.7 编出来的 kernel 里 `mma.sync` 计数 = **0**
- 所有 dot 被降级成 `fma.rn.f16`，也就是走 CUDA core 的标量/向量单元
- 结果不是"慢一点"，而是**比 PyTorch 原生 SDPA 还慢**——多了一层量化开销却没吃到张量核

这跟 Ampere 的差异不只是指令集：sm80 每 SM 有 164 KB shared memory，sm75 只有 **64 KB**。
Triton 的 tile 切分策略是按大共享内存设计的，在 sm75 上连 tile 都放不下。

> 所以别再花时间"调 Triton 的 config"。它不是参数问题，是后端没有这条代码路径。

---

## 二、定制版做了什么（本机实测版本：2.1.2）

打开 `site-packages/sageattention/core.py`，`sageattn()` 开头就是架构分派：

```python
arch = get_cuda_arch_versions()[q.device.index]
if arch == "sm75":
    # sm75 (Turing, e.g. RTX 20-series) has no working Triton JIT toolchain on this
    # Windows + CUDA 13 setup, so force the native-CUDA per-warp quantization path
    # (quant_per_warp_int8_cuda in the _fused extension) which avoids Triton entirely.
    return sageattn_qk_int8_pv_fp16_cuda(
        q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale,
        return_lse=return_lse, pv_accum_dtype="fp32", qk_quant_gran="per_warp", **kwargs)
```

三件事同时发生了：

### 1. 绕开 Triton

量化部分从 `per_thread_int8_triton()` 换成 `per_warp_int8_cuda()`（在 `_fused` 原生扩展里）。
**这一换才有张量核指令可用。**

### 2. 借用 sm80 的预编译 kernel

最终调的是 `_qattn_sm80` 扩展：

```python
lse = _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(
    q_int8, k_int8, v, o, q_scale, k_scale, ...)
```

`sm80` 的 kernel 在 sm75 上**能跑**（Turing 的 INT8 `mma` 指令是它的子集），
但它是按 164 KB shared memory 和 Ampere 的寄存器布局编的，在 64 KB 的 sm75 上只能降级使用。

### 3. tile 配置被强制缩小

```python
major, minor = torch.cuda.get_device_capability(q.device)
is_sm75 = (major == 7 and minor == 5)
if is_sm75:
    BLKQ = 64;  BLKK = 32;  WARPQ = 16;  WARPK = 32
else:
    BLKQ = 128; BLKK = 64;  WARPQ = 32;  WARPK = 64
```

**Q 方向的块从 128 砍到 64，K 方向从 64 砍到 32。** 这是 shared memory 不够的直接后果，
也是"能跑但不快"的根源——每个 block 干的活少了一半，调度开销占比上升。

同时 `pv_accum_dtype` 被写死成 `"fp32"`，`qk_quant_gran` 被写死成 `"per_warp"`，
都是为了在降级路径上保住数值稳定性。

> 一句话：**这本质是一份"能跑通的移植"，不是"针对 sm75 优化过的实现"。**
> 后者需要重写 kernel —— 那是 `PatchTuringCutlassAttention` 干的事，
> 它是用 CUTLASS 直接按 sm75 的 mma 指令写的，不借道 sm80。

### 怎么判断你装的是不是定制版

```bash
python -c "
import inspect, sageattention
print(sageattention.__version__)
print('sm75 branch:', 'sm75' in inspect.getsource(sageattention.core.sageattn))
"
```

`sm75 branch: True` 才说明你拿到的是补丁版。官方版这一行是 `False`，
跑起来会静默 fallback 或者直接抛 `Unsupported CUDA architecture`。

---

## 三、启动参数：每个是干什么的

ComfyUI 的注意力后端由 `comfy/cli_args.py` 里的一组互斥参数决定。**只需要选一个**。

| 参数 | 目的 | sm75 上的实际效果 |
|---|---|---|
| `--use-sage-attention` | 把全局默认注意力换成 `attention_sage` | 需定制包；否则 fallback 到 pytorch |
| `--use-flash-attention` | 换成 FlashAttention | ❌ sm75 不支持，flash_attn 根本不会被调用 |
| `--use-pytorch-cross-attention` | 用 PyTorch 原生 SDPA | ✅ 永远可用的保底项 |
| `--use-split-cross-attention` | 老的分块注意力，省显存 | ⚠️ 慢，只在显存不够时用 |
| `--use-quad-cross-attention` | 四分块版本 | ⚠️ 同上，更老 |
| `--disable-xformers` | 禁用 xformers 后端 | ✅ **必加**，见下 |

### 为什么必须 `--disable-xformers`

sm75 上 xformers 对 H3 这类 DiT 没有收益，而且它会抢占 attention 的接管权，
和 `PatchTuringCutlassAttention` 的 `optimized_attention_override` 打架——
表现是补丁"装上了但没生效"，日志里看不到任何 CUTLASS 核的字样。

### 验证参数是否真的生效

不要看启动日志有没有报错，要看运行时实际绑定的函数：

```python
import comfy.model_management as mm
from comfy.ldm.modules import attention

print(mm.sage_attention_enabled())               # True 才说明 --use-sage-attention 生效
print(attention.optimized_attention)             # 应该是 attention_sage
print(attention.SAGE_ATTENTION_IS_AVAILABLE)     # 包能不能 import
print(attention.SAGE_ATTENTION_SUPPORTS_MASK)    # 这个很重要，见下
```

本机实测输出（打开 `--use-sage-attention` 时）：

```
args.use_sage_attention      = True
mm.sage_attention_enabled()  = True
SAGE_ATTENTION_IS_AVAILABLE  = True
SAGE_ATTENTION_SUPPORTS_MASK = False        ← 关键
optimized_attention          = attention_sage
已注册后端: ['comfy_kitchen_int8', 'flash', 'pytorch', 'sage', 'split', 'sub_quad']
```

### `SAGE_ATTENTION_SUPPORTS_MASK = False` 是个陷阱

ComfyUI 的 `attention_sage()` 开头有这么一段：

```python
if kwargs.get("low_precision_attention", True) is False or \
   (mask is not None and not SAGE_ATTENTION_SUPPORTS_MASK):
    return attention_pytorch(...)
```

**只要这次调用带 mask，就静默回退到 PyTorch SDPA。**

有些工作流（带 padding mask、参考图区域约束的）会在部分层带 mask，
于是你以为开了 sage，实际上**只有一部分层在用，另一部分悄悄走了 SDPA**。
表现就是"开了 sage 只快了一点点"，或者"快了但部分帧质量不一样"。

---

## 四、和 CUTLASS 节点能叠加吗

**技术上不冲突，实际上不生效。**

ComfyUI 的 `wrap_attn` 装饰器在**最外层**先查 `optimized_attention_override`：

```python
def wrap_attn(func):
    def wrapper(*args, **kwargs):
        ...
        if "optimized_attention_override" in transformer_options:
            return optimized_attention_override(func, *args, **kwargs)   # 直接 return
        return func(*args, **kwargs)
```

而 `PatchTuringCutlassAttention` 正是往 `transformer_options` 里塞这个 key。

```
调用 attention
   └─ wrap_attn
        ├─ 有 optimized_attention_override？ → 走 CUTLASS，return   ← 命中这里
        └─ 没有 → 走 optimized_attention（可能是 attention_sage）
```

**接了 CUTLASS 节点之后，`--use-sage-attention` 就白开了**——它连执行的机会都没有。

> 这个结论来自读源码，没有单独做 A/B。
> 也正因为如此，本机 sage 开/关的对照实验一直测不出差异：
> 只要工作流里挂了 CUTLASS 节点，sage 这个变量就被短路掉了。
>
> 排查 convrot 崩溃时"全局 sage 已排除"是同一个道理——**它压根没参与计算**。

### 那到底选哪个

| 你的卡 | 选择 |
|---|---|
| sm75（2080 Ti / 2080 / 2070） | **CUTLASS 节点**，`--use-sage-attention` 不要加 |
| sm86 / sm89 / sm90 | sage2 或 FlashAttention，别用 CUTLASS 节点（它按 sm75 特化） |
| 想两条都试 | 只能二选一，跑两轮分别测端到端，别指望叠加 |

---

## 五、如果你还是要装 sage2

### sm75 用户

先想清楚目的。如果是为了 H3 出片速度，直接上 CUTLASS 节点，别在这条路上花时间。

如果你确实需要独立的 sage2（比如跑别的模型），要么找到绕开 Triton 的构建，
要么自己写后端。装完必须验证：

```bash
python -c "
import inspect, sageattention
src = inspect.getsource(sageattention.core.sageattn)
print('bypasses triton:', 'sm75' in src)
"
```

### sm86+ 用户

```bash
pip install sageattention
```

官方 wheel 直接可用，不需要任何补丁。这时候它确实是好东西。

### 验证真伪的最快方式

不管哪条路，最终判据只有一个：**看日志里 attention 的实际耗时**。
如果和 `--use-pytorch-cross-attention` 的基线差不多，说明你根本没走上快路径，
不管 `pip list` 显示什么。

---

## 六、启动参数完整建议

### sm75 + H3（本仓库路线）

```bash
python main.py --disable-xformers --disable-pinned-memory --cuda-device all
```

不加 `--use-sage-attention`（加了也不生效，还多一层不确定性）。

### 通用工作流（非 H3，需要 sage 的场合，sm86+）

```bash
python main.py --use-sage-attention --disable-xformers
```

### 排查时

```bash
python main.py --use-pytorch-cross-attention --disable-xformers --verbose DEBUG
```

用 SDPA 做基线，才能判断"加速到底有没有生效"。
如果你连基线都没跑过，任何"变快了"的说法都不可信。
