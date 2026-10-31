# 02 · 启动参数

## 单卡与双卡

单卡：

```
--disable-xformers --disable-pinned-memory
```

双卡再加一个（H3VM 引擎要靠它看到第二张卡）：

```
--cuda-device all
```

本机长期还挂着 `--disable-dynamic-vram`，见下面「要不要开」那一节——
**它不在最小集合里**，不加也能跑。

前两个不是拍脑袋凑的，它们决定**注意力走哪条路**和**锁页内存怎么用**：

| 参数 | 管什么 | 去掉会怎样 |
|---|---|---|
| `--disable-xformers` | 注意力后端 | CUTLASS 补丁被抢接管权，静默失效 |
| `--disable-pinned-memory` | 锁页内存 | 长时间跑之后加载阶段卡死或崩 |
| `--cuda-device all` | 设备可见性 | 双卡引擎找不到第二张卡 |

下面逐条说清目的和副作用。

---

## 逐条解释

### `--disable-xformers`

**做什么**：禁用 xformers 的 attention 后端。

**为什么**：sm75 上 xformers 对 H3 这类 DiT 没有收益，而且它会抢占 attention 的接管权，
和 `PatchTuringCutlassAttention` 的 `optimized_attention_override` 打架——
表现是补丁"装上了但没生效"，日志里看不到 CUTLASS 核的任何字样。

**副作用**：无。其他工作流（SDXL/Flux）会退到 PyTorch 原生 SDPA，速度差别很小。

---

### `--disable-pinned-memory`

**做什么**：关闭锁页内存（pinned memory）分配。

**为什么**：在 22G 改卡 + 大模型加载的场景下，锁页内存会占用大量不可回收的物理内存，
长时间多段生成后表现为"卡住不动"或者加载阶段突然崩。关掉之后加载稍慢 2~3 秒，但稳定性明显更好。

**副作用**：首次加载模型慢一点点，可忽略。

---

### `--cuda-device all`

**做什么**：让 ComfyUI 在启动时看到**全部** GPU。

**为什么**：H3VM 双卡引擎的前提条件。如果启动时只暴露一张卡，
节点能加载但运行时找不到第二张设备，报错信息很模糊（不是"没有第二张卡"，而是各种 device 不匹配）。

**副作用**：单卡工作流会默认用 cuda:0，不受影响。

---

## 注意力后端：六个互斥参数选一个

这一组参数决定 ComfyUI 默认用哪个 attention 实现。**只需要选一个**，同时给多个时后写的覆盖前面的。

| 参数 | 目的 | sm75 上建议 |
|---|---|---|
| `--use-pytorch-cross-attention` | 用 PyTorch 原生 SDPA。最稳，永远可用 | ✅ 做基线时用 |
| `--use-sage-attention` | 换成 SageAttention | ⚠️ 需定制 wheel，且会被 CUTLASS 节点短路 |
| `--use-flash-attention` | 换成 FlashAttention | ❌ sm75 不支持 |
| `--use-split-cross-attention` | 分块注意力，省显存但慢 | ⚠️ 显存告急时用 |
| `--use-quad-cross-attention` | 四分块，更老 | ⚠️ 同上 |
| `--disable-xformers` | 关掉 xformers 后端 | ✅ **必加** |

### 为什么必须 `--disable-xformers`

sm75 上 xformers 对 H3 这类 DiT 没有收益，而且它会抢占 attention 的接管权，
和 `PatchTuringCutlassAttention` 的 `optimized_attention_override` 打架——
表现是补丁"装上了但没生效"，日志里看不到任何 CUTLASS 核的字样。

### `--use-sage-attention` 在 sm75 上的真相

简短版：**能用，但要定制 wheel；而且只要工作流里挂了 CUTLASS 节点，它就完全不参与计算。**

详细版（含源码路径、为什么需要 sm75 补丁、与 CUTLASS 的优先级关系）见
[03b-sageattention.md](03b-sageattention.md)。

验证参数是否真的生效，别看日志，看运行时绑定：

```python
import comfy.model_management as mm
print(mm.sage_attention_enabled())        # True 才算开了
```

---

### `--disable-dynamic-vram`（可选，本机默认开着）

**做什么**：关闭 ComfyUI 0.37 引入的 DynamicVRAM（`comfy_aimdo`），让权重按传统方式常驻/换出。

**为什么加**：22 GB 的改卡上，DynamicVRAM 的按页 pin/unpin 比较激进——
它会在段与段之间回收权重页，而 H3 的 int8 原生核直接拿张量地址去 launch，
不经过 comfy 的 `vbar_fault` 补页路径。配套的 `--disable-pinned-memory` 是同一个目的：
别让 aimdo 的缓冲策略和原生核抢内存。

**它不解决 convrot 崩溃。** 这一点必须说清楚：参数确实进了进程命令行，
工作流照样崩在原生算子上。真正止血的是预热 + 有界重试补丁
（见 [05-pitfalls.md](05-pitfalls.md) 坑 #1）。

**要不要开**：本机是两个都开着跑稳的，但**没有做过"只开一个"的对照实验**，
所以不能说是哪个在起作用。取舍很直接——关了动态显存调度，超大分辨率下更容易 OOM；
开着，则在长序列上有更多换页动作。22G 卡上如果你不跑 1344×768 以上，开着更稳。

> 别把两件事混在一起：`--disable-dynamic-vram` 管的是**显存策略**，
> 补丁管的是**原生算子的失败**。它们不是互为替代的关系，也没有证据表明缺一必崩。

---

## ⛔ 不要用的参数

### `--fast fp16_accumulation`

sm75 上实测**全黑画面**，音频轨报 EINVAL。两轮独立测试结果一致。
不要因为"它有 fp16 字样"就以为对 Turing 友好。

### `--vram-headroom N`（Linux / 大分辨率场景）

**这是本篇最反直觉的一条**：加余量参数不是"预留空间"，而是"抬高 aimdo 的保留下限"。

结果是峰值显存不变、可用空间更少，必然 `illegal memory access`
（崩在 `reset_cast_buffers` → `synchronize`）。7 次测试全崩。

**要余量只能降分辨率或缩段时长，不能靠参数。**

### `--disable-comfy-compiler`

曾经怀疑 ComfyUI 的模型编译器（CUDA Graph 捕获）是 convrot 崩溃的元凶。
实测关掉之后，日志里 "graph breaks" 那行确实消失了，**但崩溃一模一样**。
已证伪，不要加。

---

## 启动器面板里的开关

如果你用的是带 GUI 的启动器（TE 模式之类），面板上有一堆开关
（`fp16_accumulation` / `cuda_malloc` / `sage_attention` / `reserve_vram`……）。
它们**不直接等价于 CLI 参数**，不同启动器版本映射方式还会变。

判断实际生效值的唯一可靠方法：看进程的命令行。

```powershell
# Windows：看 ComfyUI 进程到底带了哪些参数
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object -ExpandProperty CommandLine
```

```bash
# Linux
tr '\0' ' ' < /proc/$(pgrep -f main.py | head -1)/cmdline
```

本文档里写的都是 CLI 层的结论，因为它们可验证、可复现、可 diff。

---

## 一份完整可用的启动命令

单卡（最小集合）：

```bash
python main.py \
  --port 8188 \
  --disable-xformers \
  --disable-pinned-memory
```

双卡（H3VM 引擎）：

```bash
python main.py \
  --port 8188 \
  --disable-xformers \
  --disable-pinned-memory \
  --cuda-device all
```

本机实际还带一个（可选，理由见上）：

```
--disable-dynamic-vram
```

排障时再加 `--verbose DEBUG`，但正常跑不要开——它会显著拖慢出图。
