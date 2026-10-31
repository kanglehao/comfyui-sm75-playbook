# 01 · 环境矩阵

所有版本号都是实测通过的组合。**不要"顺手升一下"**——这套东西的每一环都卡在具体的版本边界上，
升级任意一项都可能让某个 int8 核忽然找不到匹配的 ABI。

## 实测通过

| 组件 | 版本 | 备注 |
|---|---|---|
| 系统 | Windows 11 专业版 | Linux 同样适用，差异见文末 |
| ComfyUI | **0.41.3**（commit `8713616`） | 0.37 起有 DynamicVRAM，见 [02](02-launch-flags.md) |
| Python | 3.13.6 | 嵌入式发行版 |
| PyTorch | **2.12.1+cu132** | `sm_75` 必须由官方 cu13x 轮子提供；cu132 是 Turing CUTLASS 核的前提 |
| CUDA Runtime | 13.2 | 由 torch 自带，不需要单独装 toolkit |
| NVIDIA 驱动 | 617.42 | 需满足 CUDA 13 |
| torchvision | 0.27.1+cu132 | |
| torchaudio | 2.11.0a0+d20260331 | |
| comfy-kitchen | **0.2.37** | int8_convrot 反量化 / rowwise 量化算子在这里 |
| comfy-aimdo | 0.5.6 | DynamicVRAM 依赖 |
| comfyui_frontend_package | 1.55.17 | 工作流 JSON 格式跟它相关 |
| comfyui-manager | 4.0.2 | |

## 装了但主链不用

这些装了不代表在用。列出来是为了说明"别的教程让你装的，其实用不上"：

| 组件 | 版本 | 状态 |
|---|---|---|
| sageattention | 2.1.2 | ⚠️ 见 [03b](03b-sageattention.md)，sm75 上收益有限且会被 CUTLASS 短路 |
| triton-windows | 3.7.0 | ⚠️ cc < 8 上一发张量核指令都不生成，结构性用不上 |
| flash_attn | 2.8.4 | ❌ sm75 不支持，实际不会被调用 |
| xformers | 未启用 | 用 `--disable-xformers` 主动关掉 |

> 关键在于：**"装上了"和"跑在快路径上"是两件事。**
> 判断依据不是 pip list，而是运行时实际绑定的函数，或日志里出现的 kernel 名。

---

## 两个实例，别搞混

本机同时存在两套 ComfyUI，版本差了一整个大版本：

| | 主力 | 旧实例 |
|---|---|---|
| 路径 | `D:\ComfyUI_Sage` | `E:\AI_Project\AIGc\ComfyUI_Sage` |
| ComfyUI | **0.41.3** | 0.37.0 |
| comfy-kitchen | 0.2.37 | 0.2.35 |
| 用途 | 日常生产、H3 工作流 | 历史排障、旧工作流兼容 |

**本仓库所有数字都来自主力实例（0.41.3）。** 如果你在两台机器间切换，
先跑 `scripts/check_env.py` 确认自己在哪一套上——版本不同，kernel 行为和可用节点都会不同。

`models` 目录在两个实例间是 junction（同一个物理目录），所以模型不用下两份。

---

## 版本边界（动这几项前先备份）

| 组件 | 边界条件 | 越界后果 |
|---|---|---|
| comfy-kitchen | 0.2.33 / 0.2.34 / 0.2.35 的 dequant 输出**逐位相同** | 版本不是崩溃原因，别在这里找 |
| **comfy-kitchen** | `requirements.txt` 里 pin 的版本可能比你装的低 | 点"安装/修复依赖"会把你降回去，然后重新踩坑 |
| ComfyUI | < 0.33 缺 H3 carried-audio 机制 | PDD 节点会 fail-closed 报版本过低 |
| ComfyUI | ≥ 0.38 | Veda 节点的最低要求 |
| ComfyUI | 0.37 起引入 DynamicVRAM | 与 int8 原生核的交互见 [02](02-launch-flags.md) |
| Sol-VAE | 需要 ComfyUI ≥ 0.44 的 `decoder.blocks.0.scale1` | 0.41 上选中即 `VAE is invalid: None` |

### 升级后必做的三件事

1. 跑一遍 `python scripts/check_env.py`，对比上面的表
2. 手动确认 `comfy-kitchen` 没被降版本
3. 跑一条 73 帧的小样，看日志有没有 `invalid argument`（见 [05-pitfalls.md](05-pitfalls.md) 坑 #1）
4. 确认 `custom_nodes/h3_convrot_warmup`、`h3_int8_guard` 还在（补丁被冲掉的判断见 [patches/README.md](../patches/README.md)）

---

## Windows 与 Linux 的差异

| 项 | Windows | Linux |
|---|---|---|
| 崩溃后恢复 | 关掉 ComfyUI 进程重开 | `systemctl restart comfyui-*` |
| 进程守护 | 启动器 | systemd unit |
| 大文件传输 | — | 本地 HTTP + `wget -c`，走 2.5G 直连约 290 MB/s |
| 显存回收 | 相对保守 | 更激进，长时间跑建议每轮重启 |

**共同点：一次 CUDA 崩溃会污染整个上下文，不重启进程就一定会接着崩。**
不要指望"再点一次运行"能自己好。
