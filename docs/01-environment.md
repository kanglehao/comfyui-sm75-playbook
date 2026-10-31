# 01 · 环境矩阵

所有版本号都是实测通过的组合。**不要"顺手升一下"**——这套东西的每一环都卡在具体的版本边界上，
升级任意一项都可能让某个 int8 核忽然找不到匹配的 ABI。

## 实测通过

| 组件 | 版本 | 备注 |
|---|---|---|
| 系统 | Windows 11 专业版 | Linux 同样适用，差异见文末 |
| ComfyUI | **0.37.0**（commit `73c9bad`） | 0.37 引入了 DynamicVRAM，见坑 #1 |
| Python | 3.13.6 | 嵌入式发行版 |
| PyTorch | **2.12.1+cu132** | `sm_75` 必须由官方 cu13x 轮子提供 |
| CUDA Runtime | 13.2 | 由 torch 自带，不需要单独装 toolkit |
| torchvision | 0.27.1+cu132 | |
| torchaudio | 2.11.0a0+d20260331 | |
| comfy-kitchen | **0.2.35** | int8_convrot 反量化算子在这里 |
| comfy-aimdo | 0.5.5 | DynamicVRAM 依赖，0.37 一起带进来的 |
| comfyui_frontend_package | 1.52.7 | 前端；工作流 JSON 格式跟它相关 |
| comfyui-manager | 4.0.2 | |

## 装了但主链不用

这些装了不代表在用。列出来是为了说明"别的教程让你装的，其实用不上"：

| 组件 | 版本 | 状态 |
|---|---|---|
| sageattention | 2.1.2 | ⚠️ 能 import，但 sm75 上无收益，且与 CUTLASS 补丁二选一 |
| triton-windows | 3.7.0 | ⚠️ Turing 无张量核路径，装了也是退化执行 |
| flash_attn | 2.8.4 | ❌ sm75 不支持，实际不会被调用 |
| xformers | 未启用 | 用 `--disable-xformers` 主动关掉 |

> 关键在于：**"装上了"和"跑在快路径上"是两件事。**
> 判断依据不是 pip list，而是启动日志里有没有对应的 kernel 名。

---

## 版本边界（动这几项前先备份）

| 组件 | 边界条件 | 越界后果 |
|---|---|---|
| comfy-kitchen | 0.2.33 / 0.2.34 / 0.2.35 的 dequant 输出**逐位相同** | 版本不是崩溃原因，别在这里找 |
| **comfy-kitchen** | `requirements.txt` 里 pin 的是 `0.2.34` | 点"安装 requirements"会把你降回去，然后重新踩坑 |
| ComfyUI | < 0.33 缺 H3 carried-audio 机制 | PDD 节点会 fail-closed 报版本过低 |
| ComfyUI | 0.37 引入 DynamicVRAM | 与 int8 原生核有冲突，见坑 #1 |
| Sol-VAE | 需要 ComfyUI ≥ 0.44 的 `decoder.blocks.0.scale1` | 0.37 上选中即 `VAE is invalid: None` |

### 升级后必做的三件事

1. 跑一遍 `python scripts/check_env.py`，对比 `docs/01-environment.md` 的表
2. 手动确认 `comfy-kitchen` 没被降版本
3. 跑一条 73 帧的小样，看日志有没有 `invalid argument`（见坑 #1）

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
