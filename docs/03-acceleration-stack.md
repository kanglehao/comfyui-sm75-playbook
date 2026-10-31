# 03 · 加速栈：装什么、接什么、参数怎么填

## 全景

```
                    ┌─────────────────────────────────────────┐
   UNETLoader ──────►  LoraLoaderModelOnly (8步蒸馏 LoRA)      │  层1 步数
   (int8_convrot)    │            │                            │  层2 权重格式
                     │            ▼                            │
                     │  PatchTuringCutlassAttention (sm75)      │  层3 注意力
                     │            │                            │
                     │            ▼                            │
                     │      BasicGuider / BasicScheduler        │
                     └─────────────────────────────────────────┘
                                  │
                                  ▼
                     H3VMCoreEngine（可选，双卡）                层4 多卡
                                  │
                                  ▼
                     H3VMModeAwareVideoVAEDecode                层5 VAE
```

**顺序是硬性的**：注意力补丁必须在所有 LoRA 之后、guider 之前。
放错位置不会报错，只会静默失效。

---

## 层 1：步数蒸馏 LoRA

| 项 | 值 |
|---|---|
| 节点 | `LoraLoaderModelOnly`（ComfyUI 内置） |
| 文件 | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |
| 强度 | `1.0` |
| 采样步数 | **8**（用 4 步 LoRA 时填 4） |

**这是收益最大的一层**：20 步 → 8 步，直接砍掉 60% 的采样时间，而且不影响显存。

**注意**：
- 步数必须和 LoRA 匹配。8 步 LoRA 跑 20 步不会更好，只会更慢
- sampler 建议 `euler` + `simple`，或 `res_multistep` + `simple`，两者都实测可用
- CFG 用 1.0（BasicGuider 无条件分支），turbo LoRA 不需要 CFG

### PDD 蒸馏（另一条路线）

MiniMax 官方的 PDD（Parallel Decoding Distillation）用不同机制：不是普通 LoRA，
而是带一个 **32 份按区间复制的 head bank**，每个采样步融合成一个 mean-block-velocity head。
普通 LoRA 加载器读不了它，必须用专用节点。

| 节点 | 用途 |
|---|---|
| `MiniMaxH3PDDAccApply` | 应用 PDD 文件，输出 `MODEL + SIGMAS` |
| `MiniMaxH3PDDAccScheduler` | 生成训练网格上的 sigma（中途接管时用） |

参数 `nfe`：`8`（训练块大小，默认）/ `4`（两步合一，官方认可）/ `6`（非均匀 `8,8,4,4,4,4`）。
**超过 8 步会掉出训练包络，画面直接变噪点。**

> PDD 与 turbo LoRA 二选一，不要叠加。

---

## 层 2：权重格式

| 权重 | 显存 | 备注 |
|---|---|---|
| `bf16` | ~14 GB | 2080 Ti 单卡放不下 |
| `pruned_fp8_scaled` | ~7 GB | 可用，但 fp8 在 sm75 上是软件模拟，不快 |
| **`pruned_int8_convrot`** | **~7 GB** | ✅ 推荐。Turing 的 INT8 张量核是真的 |
| `nvfp4` | ~4 GB | ❌ **sm75 不支持 FP4**，别下 |

"convrot" 指权重里带了一个旋转（Hadamard）变换，用来提升 INT8 量化的数值稳定性。
反量化由 `comfy-kitchen` 的原生扩展完成，这就是坑 #1 里那个 kernel。

**文本编码器同理**：必须用 `int4_convrot` 版本。
官方的 `nvfp4_awq` 版本在 sm75 上加载即失败——这是最容易在第一步就卡住的地方。

---

## 层 3：Turing CUTLASS 注意力（核心）

### 节点

`Patch Turing CUTLASS Attention (sm75)`，类型名 `PatchTuringCutlassAttention`，
来自 `custom_nodes/comfyui-minimax-h3-turing`。

它是 **drop-in 替代** `Patch Sage Attention KJ` 的：同样的位置、同样的 MODEL→MODEL。

### 参数

| 参数 | 可选值 | 建议 | 说明 |
|---|---|---|---|
| `streaming_chunk` | `low` / `high` / `false` / `off` | **`low`** | MLP 流式分块。`low`=2048 分块（速度+省显存），`off`=完全跳过 MLP 补丁 |
| `attention_mode` | `hadamard_dual_int8` / `t5_cutlass_sm75` / `disabled` / `pytorch_sdpa` | **`hadamard_dual_int8`** | 前者 ~75 ms 且精度 0.99987，后者 ~108 ms 但 FP16-PV 保真 |
| `smooth_k` | 布尔 | `true` | t5 模式下对 K 减去 per-head 均值，数值更稳 |
| `pad_tail` | 布尔 | `true` | 序列长度自动 pad 到 128 的倍数 |

### 它顺带做了两件事

这个节点不只是换注意力：

1. **MiniMax H3 的 FP16 精确数学修复** —— 所以**不需要** `--fp16-unet`，也不需要额外的 dtype 补丁节点
2. **MLP GEMM 补丁（W4A8）** —— 由 `streaming_chunk != off` 控制

所以接完这个节点，**不要再接 `TaiYang2080ti_minimaxh3`**，两者都在改 compute dtype，
叠加的结果是未验证的。二选一，默认选前者。

### 什么时候改用 `t5_cutlass_sm75`

画面出现异常色块、或者 prompt 里有精确文字要求时。它慢 40%，但数值路径更保守。

---

## 层 4：双卡 H3VM 引擎（可选）

| 节点 | 作用 |
|---|---|
| `H3VMCoreEngine` | 接管 UNET 的执行与显存调度，双卡流水 |
| `H3VMModeAwareVideoVAEDecode` | 双卡 VAE 解码 |

**前提**：启动参数必须带 `--cuda-device all`。

**收益**：同形状从单卡 49.8 s/it 压到 **6.5 s/it**（768×512 下 8.43 s/it）。

**代价**：
- 峰值顶到 21.40~21.45 GiB，余量 0.55 GiB
- 进度条的 s/it **不可信**（异步流水，进度条跑在工作前面）
- 二采开不了（OOM）

⚠️ **重要**：接了这个引擎之后，`PatchTuringCutlassAttention` 必须放在引擎**之前**。
放在之后会报 `'H3SnapshotPrefixProxy' object has no attribute 'attn'`——
引擎把 blocks 换成了代理对象，注意力补丁读不到 `attn` 属性。

> 这条顺序限制对所有"运行时钩子式"的注意力加速器都成立（包括 Veda）。
> 因为引擎的 `_copy_weight_patch_state()` 只重放 `patches`（权重补丁），
> 不重放 `callbacks`（运行时钩子）。

---

## 层 5：VAE（目前无解）

| 方案 | 状态 |
|---|---|
| `minimax_h3_video_vae_fp16` | ✅ 当前唯一可用 |
| `minimax_h3_video_vae_int8_convrot` | ⚠️ 可用，速度差不多 |
| `minimax_h3_sol_vae_fp16`（24 层蒸馏） | ❌ ComfyUI 0.37 不支持，选中即 `VAE is invalid: None` |

Sol-VAE 是冲着"VAE 解码占 63%"这个事实来的，但需要 ComfyUI ≥ 0.44 的
`decoder.blocks.0.scale1`。升级前先确认你的其他自定义节点能跟上。

---

## 可选：Veda 稀疏注意力

> ⚠️ **上游 Veda 不支持 sm75。** 它的 `docs/hardware.md` 明确写着「SM75 及以下：无 kernel，节点报错」。
> 本机能跑，是因为补了一套**自研的 sm75 后端**（见下），这部分**不在本仓库范围内**。
> 直接 `git clone` 上游 Veda 放到 2080 Ti 上，节点会报硬件不支持。

**仅在单卡裸工作流下有效**：

| 配置 | s/it | 端到端 |
|---|---|---|
| 无 Veda | 49.84 | 432~558 s |
| Veda（32 tile） | 15.77 | 163.5 s |

但接双卡引擎之后收益归零（±5% 噪声），原因见 [04-benchmarks.md](04-benchmarks.md)。

### 自研 sm75 后端做了什么

上游的注意力路径是 `CUDA sm80+ → triton-int8`。本机把它改成：

```
CUDA sm75 → cuda-int8-mma-sm75   （张量核 m8n8k16.s8 / m16n8k8.f16，首选）
          → cuda-int8-sm75       （DP4A 兜底，只在首选编不过时用）
```

**为什么必须自己写**：Triton 3.7 在 `cc < 8` 上一条张量核指令都不发
（实测 `mma.sync` 计数 = 0，所有 dot 降级成 `fma.rn.f16`），
所以 Triton 路径在 Turing 上是结构性不存在，不是被闸门挡住。

涉及文件（升级/重装 Veda 会全部失效）：

| 文件 | 作用 |
|---|---|
| `veda_comfy/backends/__init__.py` | 注册表加 `[VEDA-SM75]` 两个后端 |
| `veda_comfy/backends/cuda_int8.py` | DP4A 后端 |
| `veda_comfy/backends/cuda_int8_mma.py` | 张量核后端（首选） |
| `veda_comfy/kernels/sage_cuda/` | `sparse_int8_cuda.py` + `sparse_int8_mma.py` + `src/` |

**怎么判断补丁还在**：节点文本会打印实际 kernel 名，出现
`Turing CUDA INT8 MMA (sm75)` 或 `CUDA INT8 DP4A (sm75)` 即为生效；
若报「硬件不支持 / SM80 起」，说明补丁被冲掉了。
`veda_comfy/backends/` 下存在 `.sm75orig` 备份文件 = 魔改痕迹仍在。

**精度边界**：INT8 Q/K + 每块一个 scale 是格式本身的下限（稠密 ~1.3%，逐点 ~2.7%），
后端 tolerance 设 0.05。整数路径的精度跳变比 fp16 更硬，商业交付前要跟全注意力比一遍。

### 模型前提

**额外前提**：模型必须是 **56 头**（如 `fl2va_pruned_int8_convrot`）。
28 头的 fused 融合模型用不了，会报：

```
This model has 50 x 28 x 128, but the predictor was trained for 50 x 56 x 128
```

---

## 节点清单速查

| 节点类型名 | 显示名 | 来源 | 层 |
|---|---|---|---|
| `PatchTuringCutlassAttention` | Patch Turing CUTLASS Attention (sm75) | comfyui-minimax-h3-turing | 3 |
| `MiniMaxH3MLPGemmPatch` | MiniMax H3 MLP GEMM Patch | 同上（已被上面那个内置） | 3 |
| `TaiYang2080ti_minimaxh3` | TaiYang2080ti MiniMaxH3 | 独立节点 | 3（二选一） |
| `MiniMaxH3PDDAccApply` | PDD Acc LoRA (Apply) | ComfyUI-MiniMax-H3-PDD-Acc | 1 |
| `H3VMCoreEngine` | H3VM Core Engine | ComfyUI-H3-MultiGPU | 4 |
| `H3VMModeAwareVideoVAEDecode` | H3VM VAE Decode | 同上 | 5 |
| `LoraLoaderModelOnly` | LoRA Loader (Model Only) | ComfyUI 内置 | 1 |

> 本仓库不重新分发这些节点的代码。它们的授权、更新、issue 都在各自仓库。
