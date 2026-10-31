# 06 · 模型清单

## 目录结构

ComfyUI 会按 `folder_paths` 扫描这些子目录。带子文件夹的写法（`MiniMax-H3/xxx`）是为了让
多个版本的 H3 模型不打架，ComfyUI 支持多层目录。

```
ComfyUI/models/
├── diffusion_models/
│   └── MiniMax-H3/
│       ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors      ← 主用
│       └── minimax_h3_ref2va_pruned_int8_convrot.safetensors     ← 多参考图
├── text_encoders/
│   └── MiniMax-H3/
│       └── qwen3vl_32b_minimax_h3_int4_convrot.safetensors       ← 必须 int4
├── vae/
│   └── MiniMax-H3/
│       ├── minimax_h3_video_vae_fp16.safetensors                 ← 视频
│       └── minimax_h3_audio_vae_fp32.safetensors                 ← 音频
├── loras/
│   └── MiniMax-H3/
│       ├── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
│       └── minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
└── pdd_acc/                                                       ← PDD 蒸馏（可选）
    └── minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors
```

---

## 必备四件

| 用途 | 文件 | 磁盘 | 说明 |
|---|---|---|---|
| UNET | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | ~20 GB | **56 头**，Turing 可用 INT8 张量核 |
| 文本编码器 | `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` | ~14 GB | **不能换 nvfp4 版**，见坑 #2 |
| 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | ~1 GB | Sol-VAE 暂不可用 |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | 小 | H3 是音视频同时生成，别漏 |

**合计约 35 GB。**

> 如果多套 ComfyUI 共用模型，把 `models` 做成 junction / 符号链接即可，
> 不用下两份。判断是链接还是真目录：`fsutil reparsepoint query <path>`。

### 一个要留意的模型

`fastvideo_fasth3_8step_v2_pruned_int8_convrot` 能跑，但它把 `token_refiner` 也量化了，
于是会走 `quantize_int8_rowwise` 的回退路径——**这个算子需要 `h3_int8_guard` 加固**，
否则在采样第一步就报 `CUDA INT8 rowwise quantization failed`。

其余 H3 int8 模型的 `token_refiner` 还是 BF16，不会碰这条路径。
细节见 [05-pitfalls.md](05-pitfalls.md) 坑 #1。

### 权重格式怎么选

| 格式 | UNET 大小 | sm75 可用 | 备注 |
|---|---|---|---|
| `bf16` | ~28 GB | ✅ | 单卡放不下 |
| `pruned_fp8_scaled` | ~14 GB | ⚠️ | fp8 在 sm75 上是软件模拟 |
| **`pruned_int8_convrot`** | **~20 GB** | ✅ **推荐** | 走真实 INT8 张量核 |
| `nvfp4` | ~8 GB | ❌ | **无 FP4 硬件支持** |

> int8 的文件反而比 fp8 大，因为 convrot 版本保留了旋转矩阵和 per-group scale。
> 这是正常的，别以为是下错了。

---

## 加速 LoRA（按需）

| 文件 | 步数 | 用途 |
|---|---|---|
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 8 | **默认选择**。速度与质量平衡最好 |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 4 | 更快，细节更弱，适合草稿 |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | 8 | 参考图生视频（ref2v）专用 |

**8 步与 4 步的 LoRA 不能混用步数**。用 4 步 LoRA 就必须把采样步数设成 4。

---

## PDD 蒸馏（可选，另一条路线）

来自 MiniMax 官方 `alibaba-pai/MiniMax-H3-Acc-LoRAs`。它不是普通 LoRA，
必须用 `MiniMaxH3PDDAccApply` 节点加载。

| 文件 | 配对 |
|---|---|
| `minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors` | fl2va UNET |
| `minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors` | ref2va UNET |

注意配对关系：**FL2VA 的 PDD 只能配 fl2va 的 UNET**，配错了会出噪点而不是报错。

**与 turbo LoRA 二选一。**

---

## Veda 打分器（仅单卡方案需要）

| 文件 | 放哪里 | 用于 |
|---|---|---|
| `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors` | `models/veda/` | T2VA |

只有 T2VA 有公开的打分器。R2VA 另有专用版本，不一定公开。

**前提**：底模必须是 56 头。用 Veda 就别用 `fused_refdelta_*` 那类 28 头融合模型。

---

## 可选：latent 放大（二采）

| 文件 | 说明 |
|---|---|
| `minimax_h3_latent_upscaler_3d_bf16.safetensors` | 二采用 |

⚠️ 双卡引擎下开二采会 OOM（详见 [00-hardware.md](00-hardware.md) 的显存预算）。
单卡或低分辨率才考虑。

---

## 下载来源

| 内容 | 仓库 |
|---|---|
| int8_convrot / int4 量化权重 | `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` |
| 官方权重、VAE | `Comfy-Org/MiniMax-H3` |
| 实验性 VAE / 量化 | `Kijai/MiniMax-H3-experimental` |
| PDD 蒸馏 LoRA（原始） | `alibaba-pai/MiniMax-H3-Acc-LoRAs` |
| PDD 蒸馏 LoRA（ComfyUI 键名转换版） | `aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI` |

下载大文件时建议：

```bash
# 断点续传 + 校验，别用浏览器
huggingface-cli download <repo> <file> --local-dir ./tmp --resume
```

下完先核对文件大小，再移到 `models/` 对应目录。**移到一半的 safetensors 会让 ComfyUI 报
各种莫名其妙的 tensor 缺失错误**，比文件不存在更难查。

---

## 一个容易忽略的点

`models/` 目录可以用符号链接指向别的盘。这在多台机器共用一份模型时很有用：

```bash
# Linux
ln -s /data/models ComfyUI/models
```

```powershell
# Windows（需管理员或开发者模式）
New-Item -ItemType SymbolicLink -Path "ComfyUI\models" -Target "D:\models"
```

但要注意：**ComfyUI 更新时不会动符号链接本身**，所以升级后模型不用重下。
反过来说，如果你把 `models` 换成了真目录，之前的东西就"消失"了——
不是丢了，是路径变了。
