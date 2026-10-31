# 2080 Ti（sm75）上的 ComfyUI：能跑、能修、能快

> 一台双 RTX 2080 Ti 的机器，跑 ComfyUI 半年，踩完的坑和量出来的数。
> 不是整合包，不是教程搬运——**每一节都能对应到一次真实的故障或一组真实的计时**。

---

## 如果你只记住四件事

> **1. sm75 是分水岭，别照抄 4090 的方案。**
> 加速器先看它有没有 Turing 的原生核，再看它自称几倍。
>
> **2. 真正的速度不在注意力上。**
> 步数蒸馏和权重格式拿走了大部分收益；端到端时间里 **63% 是 VAE 解码**。
>
> **3. 升级前端之后，先怀疑"接管绘制函数"的扩展。**
> 连线错乱、加载中止、`structuredClone` 失败，九成是它们。
>
> **4. "文件存在"≠"能加载"≠"能执行"。**
> 唯一可信的判据是运行中实例的 `/object_info` 加浏览器 `graphToPrompt()` 成功。

---

## 这个仓库里有四条线

做了半年，实际动过的东西分四类。每条线都有独立文档，可以只看你需要的那条。

| 线 | 解决什么 | 文档 |
|---|---|---|
| **加速** | 2080 Ti 上把 H3 视频生成从 10 分钟压到 4 分钟 | [03](docs/03-acceleration-stack.md) · [04](docs/04-benchmarks.md) |
| **崩溃加固** | `CUDA INT8 convrot/rowwise ... invalid argument` | [05](docs/05-pitfalls.md) · [patches/](patches/) |
| **前端兼容** | 连线错乱、加载中止、子图值不生效 | [10](docs/10-frontend-compat.md) |
| **工作流与依赖** | 下拉值红框、成片 ImportError、大文件下不完 | [11](docs/11-workflow-repair.md) · [12](docs/12-deps-and-download.md) |

外加一篇 [SageAttention 2 在 sm75 上的真相](docs/03b-sageattention.md)——
因为几乎每个人都在这上面浪费过时间。

---

## 环境坐标

| 项 | 值 |
|---|---|
| GPU | RTX 2080 Ti ×2（22 GB 改卡，可用 21.66 GiB），`sm_75` |
| ComfyUI | **0.41.3**（commit `8713616`） |
| Python / torch | 3.13.6 / **2.12.1+cu132** |
| 关键包 | comfy-kitchen **0.2.37**、comfy-aimdo 0.5.6 |
| 前端 | comfyui_frontend_package 1.55.17 |
| 节点包 | 141 个 |

跑 `python scripts/check_env.py` 会输出一份可对照的报告。

---

## 加速：四层，按收益排序

| 层 | 手段 | 收益 | 代价 |
|---|---|---|---|
| 1 | **步数蒸馏 LoRA**（20 → 8 / 4 步） | 2.5× | 细节略降，步数必须配对 |
| 2 | **int8_convrot 权重** | 显存 −40%，读盘 −50% | 需要 sm75 的反量化核 |
| 3 | **Turing CUTLASS 注意力** | 注意力段约 6× | 只能配 H3 类 DiT，`head_dim` 必须 128 |
| 4 | **双卡 H3VM 引擎** | 端到端 1.6~2× | 峰值顶到 21.4/22 GiB，二采开不了 |

### 实测数字

单卡 768×432 / 124 帧 / 8 步：

| 配置 | s/it | 采样段 | 端到端 |
|---|---|---|---|
| 全注意力基线 | 49.84 | 398 s | 432 s（热态） |
| + Veda 稀疏（仅单卡） | 15.77 | 126 s | 163.5 s |

双卡 H3VM 引擎 736×416 / 124 帧 / 10 步：**253 s**。

### 但瓶颈已经不在注意力

以那 253 秒为例：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 编码 / 条件准备 | ~28 s | 11% |
| 10 步采样 | ~65 s | 26% |
| **VAE 解码 + 音频 + 封装** | **~160 s** | **63%** |

**继续优化注意力最多再省 20% 的 26%。** 真想再快，方向是 VAE
（Sol-VAE 要 ComfyUI ≥ 0.44，当前版本选中即报 `VAE is invalid: None`）。

完整数据、测量方法和反缓存处理见 [04-benchmarks.md](docs/04-benchmarks.md)。

---

## 已证伪的方案（别浪费时间）

每一条都试过，或经交叉验证后排除。

| 方案 | 结论 | 依据 |
|---|---|---|
| `--fast fp16_accumulation` | ❌ 全黑画面 + 音频 EINVAL | sm75 两轮实测 |
| `--vram-headroom N` | ❌ 必然 `illegal memory access` | 7 次全崩；它只抬保留下限，不降峰值 |
| `--disable-comfy-compiler` | ❌ 与 convrot 崩溃无关 | 关掉后仍崩在同一次调用 |
| 换 comfy-kitchen 版本 | ❌ dequant 输出逐位相同 | 0.2.33/34/35 对比 |
| 官方 sageattention 2 直接用 | ❌ Triton 在 cc<8 不发张量核指令 | `mma.sync` 计数 = 0 |
| `--use-sage-attention` 叠加 CUTLASS 节点 | ❌ 静默失效 | `optimized_attention_override` 在最外层直接 return |
| Spark-H3 / Sol-H3 / Spectrum / Open-VC | ❌ 仅 SM89+/Blackwell | 官方与社区均确认 |
| Veda + 双卡引擎 | ❌ 收益归零（±5% 噪声） | 引擎已把同形状压到 6.5 s/it |
| 二采（SelfLift）+ 双卡 | ❌ OOM | 需 946 MiB，GPU1 仅余 631 MiB |

> Veda 那条值得单独说：它的 3.16× 是**单卡稠密注意力**基线下的 3.16×。
> 拿它去乘生产链的数字，就变成了一个不存在的收益。
> 它自己的稀疏核在双卡链路上要跑 52.5 s，而整段采样才 65 s。

---

## 崩溃加固：为什么需要两个补丁

`CUDA INT8 convrot dequantization failed: invalid argument` 这个错，
网上能查到的解释大多指向显存、对齐、动态 VRAM——**我们都排除过**。

能确定的事实只有三条：

1. 失败现场的所有静态参数都合法（形状/步长/dtype/连续性/16B 对齐/device/stream/capture）
2. **同一份入参原样重发必定成功**
3. 失败点不固定（观测到第 409 次、第 1101 次）

所以结论只到"原生算子在特定时序下的偶发失败"，修法是**预热 + 有界重试**，
而不是去调什么参数。

同一台机器上还有第二个算子（`quantize_int8_rowwise`）会报同样的错，
但只在把 `token_refiner` 也量化的模型上触发——因为 k=5376 正好落在
fused kernel 被跳过的区间，加上 Turing 的 48 KB 共享内存上限，必然走未加固的回退路径。

**两个补丁都要装**，见 [patches/](patches/)。它们放在 `custom_nodes/` 而不是
`site-packages/`，因为后者会被 pip 重装冲掉。

---

## 前端兼容：四种炸法

| 症状 | 根因 | 处置 |
|---|---|---|
| 连线变直角折线、端点不在插槽上 | `quick-connections` 的 Circuit Board 接管 `drawConnections`，与新前端槽位几何失配 | 改一个设置项，F5 即可 |
| 连线整体放大偏移（仅在缩放≠100% 时） | mixlab 的 `td_background.js` 把 dpr 乘了两次 | 让覆盖函数透传给原生实现 |
| 加载弹窗 `t.findInputSlot is not a function` | Impact-Pack 用 `stackTrace.includes()` 判断加载态，Vite 压缩后失效 | 补判据 + try/catch |
| `structuredClone ... could not be cloned` | 某节点把 Proxy/类实例塞进 widget | 装一个序列化守卫 |

共同规律：**升级前端后，优先怀疑所有接管 `drawConnections` / `drawNode` / `drawBackCanvas`
的显示类扩展。** 判据是"所有工作流一起坏"，而不是某个文件坏。

细节见 [10-frontend-compat.md](docs/10-frontend-compat.md)。

---

## 工作流修复：三条血泪规则

1. **模型下拉值必须是反斜杠**
   `/object_info` 给的是 `MiniMax-H3\xxx.safetensors`，写成 `/` 或裸名 → 红框报错，
   哪怕磁盘上文件确实在。**校验只能用 `/object_info` 候选池精确比对，绝不能用 `os.path.exists`。**

2. **子图的 widget 值存两份**
   `definitions.subgraphs[].nodes[].widgets_values` 和实例节点的 `widgets_values`。
   前端读的是**实例那份**，而实例的 `type` 是子图 UUID。
   我们第一批改 552 处，抽检才发现漏了 109 处。

3. **验证分五步，别停在第一步**
   节点存在 → 下拉值匹配 → 浏览器能加载 → `graphToPrompt()` 成功 → 必填输入都接了。

详见 [11-workflow-repair.md](docs/11-workflow-repair.md)。

---

## 依赖：升级后成片挂掉的两个真凶

| 症状 | 根因 | 修法 |
|---|---|---|
| 几十个包同时 `ImportError: cannot import name 'HybridCache'` | transformers 5.x 移除了它 | `pip install --no-deps --upgrade peft==0.21.2` |
| `cannot import name 'median_blur' from 'albucore'` | albucore 0.0.24 移除 | 需整套降级，别单独降 |

**给嵌入式 Python 装包一律加 `--no-deps`。** 不加会把 `filelock`、`packaging`
一路抬上去，然后 torch 生态开始随机报错。

还有：别照节点包的 `requirements.txt` 装。典型反例是某个包写死 `numpy<=1.26.4`，
照装会把 numpy 2.2.6 降下去，拖垮整个 torch 2.12。

下载链路、HF 镜像、大文件分块重试见 [12-deps-and-download.md](docs/12-deps-and-download.md)。

---

## 复现步骤

```bash
# 1. 确认硬件是 sm75
python -c "import torch; print(torch.cuda.get_device_capability(0))"   # 期望 (7, 5)

# 2. 全套自检
python scripts/check_env.py --comfyui /path/to/ComfyUI

# 3. 装加固补丁（两个都要）
cp -r patches/h3_convrot_warmup patches/h3_int8_guard  ComfyUI/custom_nodes/

# 4. 装加速节点（第三方，见 docs/03）
#    comfyui-minimax-h3-turing / ComfyUI-H3-MultiGPU / ComfyUI-MiniMax-H3-PDD-Acc

# 5. 启动
python main.py --disable-xformers --disable-pinned-memory --disable-dynamic-vram --cuda-device all

# 6. 加载加速版工作流
#    workflows/h3-i2v-2080ti-cutlass.json
```

想改造自己的工作流，别手改 JSON——用脚本注入并做结构自检：

```bash
python scripts/inject_accel.py --src 我的工作流.json --dry-run
python scripts/inject_accel.py --src 我的工作流.json --out 加速版.json
```

手改 JSON 最常见的错是只改了一端 link，ComfyUI 不报错，补丁静默失效。

---

## 仓库结构

```
README.md
docs/
  00-hardware.md          sm75 的边界，TU102 解锁与它的收益天花板
  01-environment.md       版本矩阵（钉死版本号）+ 两个实例别搞混
  02-launch-flags.md      启动参数逐条：目的、副作用、别用的
  03-acceleration-stack.md 四层加速：装什么、接什么、参数怎么填
  03b-sageattention.md    SageAttention 2 在 sm75 上跑的是什么
  04-benchmarks.md        实测数据、阶段拆解、测量方法
  05-pitfalls.md          六个坑：崩溃、NVFP4、头数、缓存、s/it、上下文中毒
  06-models.md            模型清单、格式选择、来源
  07-workflow-wiring.md   节点接线规则与两种错接
  10-frontend-compat.md   连线错乱 / 加载中止 / structuredClone
  11-workflow-repair.md   下拉值、子图双份、五步验证
  12-deps-and-download.md 依赖陷阱、HF 镜像、大文件下载
patches/
  h3_convrot_warmup/      原生算子加固（dequant）
  h3_int8_guard/          原生算子加固（rowwise）
workflows/
  h3-i2v-2080ti-cutlass.json
scripts/
  check_env.py            环境自检
  inject_accel.py         加速节点注入 + 结构自检
```

---

## 关于 TU102 解锁

22G 改卡出厂把 Tensor Core 的 FP16→FP32 累加砍了一半，UEFI 阶段可以改回来，
`mma_peak` 从 65.1 / 67.3 提到 126.7 / 129.4 TFLOPS。

**但对视频生成几乎没有帮助**——这是带宽和显存受限的负载，不是 GEMM 受限。
实测在 MoE prefill 上只有 +8~13%，decode 完全没变。
要不要做，见 [docs/00-hardware.md](docs/00-hardware.md) 的收益边界。
完整方案在另一个仓库：`tu102-22g-windows-unlock`。

---

## 致谢

加速和兼容性来自这些项目，不来自我：

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — 执行引擎
- comfy-kitchen / comfy-aimdo — int8 量化算子与显存管理
- `comfyui-minimax-h3-turing` — sm75 的 CUTLASS 注意力核
- `ComfyUI-H3-MultiGPU`（H3VM）— 双卡执行引擎
- `ComfyUI-MiniMax-H3-PDD-Acc` — PDD 蒸馏支持
- [Kijai](https://github.com/kijai)、`alibaba-pai` — 实验模型与加速 LoRA
- quick-connections / mixlab / Impact-Pack — 它们的 bug 也教会了我们很多

## License

文档、脚本与 `patches/`：[MIT](LICENSE)。
第三方节点与模型的授权请遵循各自仓库。
