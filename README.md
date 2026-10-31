# 让 MiniMax-H3 在 2080 Ti（sm75）上跑到可用速度

> 一套在 **RTX 2080 Ti / Turing（sm_75）** 上把 ComfyUI + MiniMax-H3 视频生成从"能跑"推到"能用"的实测配方：
> 每一步都有数字，每个被否掉的方案也都写了为什么否。

---

## 如果你只记住一件事

> **在 sm75 上不要再去编译 sageattention / triton 那一套。**
> 用 `PatchTuringCutlassAttention` 一个节点替代，它内部是手写的 CUTLASS INT8 注意力核，
> 不依赖 triton，装好就能用。
> 真正的大头不在注意力，而在**步数蒸馏**和**权重格式**——这两项决定了你是 20 步还是 8 步、是 12 GB 还是 7 GB 的权重读取量。

注意力只占端到端时间的四分之一左右。很多"9 倍加速"的教程把注意力当成了全部，所以在 8 步模型上收效甚微。

---

## 这是什么 / 能到什么速度

本仓库不是又一个整合包，而是一份**可核对的调优记录**：硬件前提、版本矩阵、启动参数、节点接线、
实测耗时、以及踩过的坑。所有数字都来自同一台机器的重复测量，条件写在每一张表头上。

参考量级（单张 2080 Ti 22G，768×432，124 帧 ≈ 5.2 s，8 步）：

| 配置 | 每步耗时 | 采样段 | 端到端 |
|---|---|---|---|
| 全注意力基线 | 49.8 s/it | 398 s | 558 s（含冷加载）/ 432 s（热） |
| + Turing CUTLASS 注意力 | 约 1/3 | — | — |
| + 8 步蒸馏 LoRA | 步数 20 → 8 | 按比例 | — |
| 双卡 H3VM 引擎（736×416 / 10 步） | 6.5 s/it | 65 s | **253 s** |

> 不是"2080 Ti 能打 5090"，而是"2080 Ti 能从 10 分钟一条压到 4 分钟一条，且画面可用"。

---

## 适用范围

**适用**

- 显卡：RTX 2080 Ti / 2080 Super / 2070 等 **Turing（sm_75）** 卡；显存 11G 可跑 576×320 级别
- 系统：Windows 10/11 + 便携版 ComfyUI，或 Linux
- 模型：**MiniMax-H3**（本仓库的所有实测都在 H3 上做的）

**不适用 / 请绕行**

| 场景 | 原因 |
|---|---|
| 50 系 / 40 系（sm_89+） | 用不着这套。直接上 sageattention 2 / FlashAttention，或官方 TensorRT 路径 |
| 30 系（sm_86） | CUTLASS 核是按 sm75 特化的，30 系请走 sageattention |
| 想靠注意力拿到 3 倍以上 | 在双卡/多卡引擎下注意力已经不是瓶颈，见 [docs/04-benchmarks.md](docs/04-benchmarks.md) |
| 显存 < 11G | H3 的 int8 权重 + 32B 文本编码器放不下，先解决显存 |

---

## 加速栈：五层，按收益排序

把"加速"当成一个整体是误判的根源。拆成五层之后，每一层该不该做、能拿多少，都很清楚。

| 层 | 手段 | 典型收益 | 代价 |
|---|---|---|---|
| 1 | **步数蒸馏 LoRA**（20 步 → 8 步 / 4 步） | 2.5× | 画面细节略降，需配对应步数 |
| 2 | **int8_convrot 权重**（代替 bf16） | 显存 −40%，读盘 −50% | 需要 sm75 兼容的反量化核 |
| 3 | **Turing CUTLASS 注意力**（sm75 专用） | 注意力段 3× | 只能配 H3 类 DiT |
| 3' | SageAttention 2（备选） | sm75 上收益有限 | 需定制 wheel，且与 3 互斥 |
| 4 | **双卡 H3VM 引擎**（两张卡流水） | 端到端 1.6~2× | 显存顶到 21.4/22 GiB，没有余量 |
| 5 | **VAE 解码**（当前无解） | 理论 60% | Sol-VAE 尚未被 ComfyUI 支持 |

第 3 和 3' 是**二选一**：CUTLASS 节点通过 `optimized_attention_override` 接管注意力，
会直接短路全局的 `--use-sage-attention`。也就是说两个都开时，sage 根本不参与计算。
细节见 [docs/03b-sageattention.md](docs/03b-sageattention.md)。

第 5 层是重点：**端到端 253 秒里有 160 秒在 VAE 解码和封装**。
所以继续压注意力最多再省 20% 的 26%，不如去等 VAE 侧的支持。这一点在
[docs/04-benchmarks.md](docs/04-benchmarks.md) 里有逐阶段拆解。

---

## 已证伪的方案（别浪费时间）

下面每一条都在这台机器上试过，或经过交叉验证后排除。写在最前面，是为了让别人少走一遍。

| 方案 | 结论 | 依据 |
|---|---|---|
| `--fast fp16_accumulation` | ❌ **全黑画面 + 音频 EINVAL** | sm75 上两轮实测 |
| `--vram-headroom N` | ❌ **必然 `illegal memory access`** | 7 次测试全崩，它只抬高保留下限，不降峰值 |
| `--disable-comfy-compiler` | ❌ 与 convrot 崩溃无关 | 关掉后仍崩在同一次调用 |
| 换 comfy-kitchen 版本（0.2.33/34/35） | ❌ 逐位相同 | 三版 dequant 算子输出一致 |
| Spark-H3 / Sol-H3 / Spectrum / Open-VC | ❌ 仅 SM89+/Blackwell | 官方与社区均确认不支持 sm75 |
| FastH3 V2/V3 的"9 倍" | ⚠️ 8 步下质量崩塌 | 社区实测：高步与官方流差 <10% |
| Veda 稀疏注意力（单卡裸流） | ✅ 3.16× | 仅限单卡；**接双卡引擎后收益归零**，见下 |
| Veda + H3VM 双卡引擎 | ❌ 无效（±5% 噪声） | 引擎已把同形状压到 6.5 s/it，无可压缩空间 |
| 二采（SelfLift）+ 双卡 | ❌ OOM | 需 946 MiB，GPU1 仅余 631 MiB |
| `--use-sage-attention` 叠加 CUTLASS 节点 | ❌ 静默失效 | `optimized_attention_override` 在最外层直接 return，sage 不参与计算 |
| 官方 PyPI 版 sageattention 2.x 直接用 | ❌ sm75 无 Triton 张量核路径 | 需打 sm75 补丁的定制 wheel，见 [03b](docs/03b-sageattention.md) |

> Veda 那条值得单独说：它在单人单卡裸工作流上确实把 49.8 → 15.8 s/it，
> 但那个"49.8"是**单卡稠密注意力**的基线，不是生产链的基线。
> 拿它去乘生产链的 6.5 s/it，就变成了一个不存在的收益。

---

## 六个坑

完整版见 [docs/05-pitfalls.md](docs/05-pitfalls.md)，这里只列标题：

1. **`CUDA INT8 convrot dequantization failed: invalid argument`** —— 进程内第一次原生 kernel launch 的冷启动假失败，预热一次即可，不是显存/对齐/流的问题。
2. **NVFP4 文本编码器在 sm75 上不可用** —— 必须换 `int4_convrot` 版本，否则加载即失败。
3. **26 头 / 28 头 / 56 头模型不能混用注意力加速器** —— 头数不匹配会直接报 predictor 维度错误。
4. **同 seed 重跑会命中 ComfyUI 执行缓存** —— 2 秒返回旧产物，看起来像"加速成功"。**验收必须换 seed，并比对产物 MD5。**
5. **多卡引擎下进度条 s/it 不可信** —— 异步流水，进度条跑在工作前面，只能看端到端。
6. **崩溃一次后 CUDA 上下文中毒** —— Linux 上必须重启服务进程，Windows 上必须重启 ComfyUI 进程。

---

## 复现步骤

### 1. 确认硬件确实是 sm75

```bash
python -c "import torch; print(torch.cuda.get_device_capability(0))"
# 期望输出 (7, 5)
```

一步到位自检：`python scripts/check_env.py`

### 2. 装加速节点

```bash
cd ComfyUI/custom_nodes

# sm75 专用注意力 + MLP GEMM 补丁（核心）
git clone <comfyui-minimax-h3-turing 仓库地址> comfyui-minimax-h3-turing

# 可选：8 步 PDD 蒸馏
git clone <ComfyUI-MiniMax-H3-PDD-Acc 仓库地址>
```

> 这些是**第三方节点**，本仓库不重新分发它们的代码，只记录版本与接线。
> 具体地址见 [docs/03-acceleration-stack.md](docs/03-acceleration-stack.md)。

### 3. 放模型

见 [docs/06-models.md](docs/06-models.md)。关键三条：
UNET 用 `int8_convrot`，文本编码器用 `int4_convrot`，视频 VAE 用 `fp16`。

### 4. 用加速版启动参数

见 [docs/02-launch-flags.md](docs/02-launch-flags.md)。最小集合：

```
--disable-xformers --disable-pinned-memory
```

### 5. 加载工作流

打开 [`workflows/h3-i2v-2080ti-cutlass.json`](workflows/h3-i2v-2080ti-cutlass.json)。
它基于 ComfyUI 官方 H3 I2V 模板，只在 MODEL 链末端多插了一个
`Patch Turing CUTLASS Attention (sm75)`。

想改造自己的工作流？用脚本注入，别手改 JSON：

```bash
python scripts/inject_accel.py --src 你的工作流.json --out 加速版.json --dry-run
python scripts/inject_accel.py --src 你的工作流.json --out 加速版.json
```

它会找到 MODEL 链进入 sampler/guider 的那个点，插节点、修两端的 link id，并做结构自检
（手改 JSON 最常见的错误就是把 link 指丢了，ComfyUI 不报错，只是补丁静默失效）。

### 6. 验收

```bash
# 1) 换一个 seed，别用上一次的
# 2) 看日志里是否出现 CUTLASS / turing 相关字样
# 3) 比对产物：时长、帧数、音频轨、MD5
```

---

## 仓库结构

```
README.md                      你正在看的
docs/
  00-hardware.md               2080 Ti 的硬件前提（含 TU102 解锁）
  01-environment.md            实测版本矩阵：每一项都写死版本号
  02-launch-flags.md           启动参数逐条解释 + 副作用
  03-acceleration-stack.md     五层加速：装什么、接什么、参数怎么填
  03b-sageattention.md         SageAttention 2 在 sm75 上的真相
  04-benchmarks.md             实测数据与逐阶段拆解
  05-pitfalls.md               六个坑的完整版
  06-models.md                 模型清单、来源、放哪个目录
  07-workflow-wiring.md        节点接线规则与常见错接
workflows/
  h3-i2v-2080ti-cutlass.json   基于官方模板的加速版工作流
scripts/
  check_env.py                 环境自检，输出可对照的报告
  inject_accel.py              把加速节点注入任意 H3 工作流
```

---

## 关于 TU102 解锁

22G 改卡（TU102）出厂时把 Tensor Core 的 FP16→FP32 累加砍了一半，在 UEFI 阶段可以改回来，
实测 `mma_peak` 从 65.1 / 67.3 TFLOPS 提到 126.7 / 129.4 TFLOPS。

但它对视频生成的收益**远小于硬件数字**：这是带宽和显存受限的负载，不是 GEMM 受限。
是否值得做，见 [docs/00-hardware.md](docs/00-hardware.md) 里的收益边界分析。
完整方案在另一个仓库：`tu102-22g-windows-unlock`。

---

## 致谢

本仓库的所有结论都建立在下列项目之上，加速来自它们，不来自我：

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) —— 执行引擎与 H3 支持
- MiniMax-H3 官方与 comfy-kitchen —— 模型与 int8_convrot 量化算子
- `comfyui-minimax-h3-turing` —— sm75 的 CUTLASS 注意力核与 MLP GEMM 补丁
- `ComfyUI-H3-MultiGPU`（H3VM）—— 双卡显存与执行引擎
- `ComfyUI-MiniMax-H3-PDD-Acc` —— 8 步 / 4 步 PDD 蒸馏 LoRA 支持
- [Kijai](https://github.com/kijai) 与 `alibaba-pai` —— 实验模型与加速 LoRA

## License

文档与脚本：[MIT](LICENSE)。第三方节点与模型的授权请遵循各自仓库。
