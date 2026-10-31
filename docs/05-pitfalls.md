# 05 · 六个坑

按"卡住你的概率"排序。前两个会让你在第一天就放弃。

---

## 坑 #1 · `CUDA INT8 convrot dequantization failed: invalid argument`

### 症状

```
CUDA INT8 convrot dequantization failed: invalid argument
```

两种触发时机，看起来像两个问题，其实是同一类：

| 形态 | 现象 |
|---|---|
| A：进程内首次 | 第一个工作流必崩一次，**再点一次运行就好了** |
| B：多段生成中途 | 第 1 段完整出片，第 2 段起必崩；有时是第 409 次调用，有时是第 1101 次 |

形态 B 更像"真 bug"，所以很多人（包括我）会往显存、对齐、流、动态 VRAM 上去猜。

### 已经排除的方向（别再走一遍）

这些全部实测排除，逐条列出来是为了省别人的时间：

- ❌ 全局 sage attention 开关
- ❌ `comfy-kitchen` 版本（0.2.33 / 0.2.34 / 0.2.35 的 dequant 输出**逐位相同**）
- ❌ ComfyUI 模型编译器（CUDA Graph）—— 关掉后 `graph breaks` 日志消失，崩溃一模一样
- ❌ `--cuda-malloc` / aimdo 缓冲区
- ❌ 指针对齐、张量形状、dtype
- ❌ 共享内存、跨设备 sticky、非默认流
- ❌ `allocation_context`、遗留 CUDA 错误、图捕获态、driver context 错位
- ❌ `--disable-dynamic-vram`

### 真正的成因

**进程内第一次原生 kernel launch 的冷启动假失败。**

证据是形态 A 的规律性：每轮恰好只失败 **1 次**，之后全部正常。失败本身把惰性初始化推进到位了，
所以第二次调用就好了。这解释不了形态 B——但形态 B 的 dump 显示失败也是"偶发一次"，
重试即过，和形态 A 是同一个冷路径。

### 解法：预热

在真实 launch 之前，用哑元张量把该 kernel 的冷路径烧掉：

```python
def _warm_convrot_dequant():
    """Run the int8 convrot dequant kernel once per dtype on dummy tensors."""
    import torch
    from comfy_kitchen.backends.cuda import dequantize_int8_convrot_weight

    q = torch.zeros((256, 256), dtype=torch.int8, device="cuda")
    scale = torch.ones((), dtype=torch.float32, device="cuda")

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for attempt in range(2):          # 第 1 次本来就应该失败
            try:
                dequantize_int8_convrot_weight(q, scale, dtype)
                break
            except Exception:
                if attempt == 1:
                    raise
```

几个必须遵守的约束：

| 约束 | 原因 |
|---|---|
| 形状必须满足 `gs == 256 and k % 256 == 0` | 否则根本不会走这个 kernel，预热等于没做 |
| 覆盖 fp32 / fp16 / bf16 三种 dtype | kernel 按 `(group_size, 输出 dtype)` 模板化，gs 恒为 256，只有 dtype 变 |
| 每设备只做一次 | 代价是 3 次 256×256 调用，可忽略 |
| **绝不抛异常、绝不碰调用方数据** | 预热失败不应该影响主流程 |

### 验证预热不会改变结果

这是最容易漏的一步。做完预热后，用同一个输入跑"开预热"和"关预热"两次，
比较输出的 **sha256**——必须完全相同。

如果不相同，说明你的预热动到了共享状态，比不做更糟。

### 怎么部署才不会被升级冲掉

如果你直接改 `site-packages/comfy_kitchen/backends/cuda/__init__.py`，
下一次 `pip install` 就没了，问题原样复发。

关键机制：`comfy_kitchen/registry.py` 的 `get_implementation()` 是**每次调用都 `getattr`**。
所以在包外替换 backend 模块的属性即可生效，且与版本无关。

做成一个独立 custom_node（比如 `h3_convrot_warmup`），在 `__init__.py` 里导入时就完成替换：

```python
# custom_nodes/h3_convrot_warmup/__init__.py
import comfy_kitchen.backends.cuda as ck_cuda

_orig = ck_cuda.dequantize_int8_convrot_weight
_warmed = False

def _patched(*args, **kwargs):
    global _warmed
    if not _warmed:
        _warmed = True
        try:
            _warm_convrot_dequant()
        except Exception as e:
            print(f"[warmup] skipped: {e}")
    return _orig(*args, **kwargs)

ck_cuda.dequantize_int8_convrot_weight = _patched

NODE_CLASS_MAPPINGS = {}      # 不需要节点，纯副作用
```

**要点：`_warm` 里注入的失败必须被吃掉，真实调用里的失败必须原样抛出。**
如果为了"稳妥"给真实调用也加重试，你会把真 bug 藏起来，调试成本翻十倍。

---

## 坑 #2 · NVFP4 文本编码器在 sm75 上直接失败

H3 的官方模板默认用 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`。

**Turing 没有 FP4 张量核**，这个文件加载即失败。必须换成：

```
qwen3vl_32b_minimax_h3_int4_convrot.safetensors
```

这个坑的恶劣之处在于它是一个 **32 GB 的下载**——很多人下完了才发现跑不了。

同理，UNET 也不要选任何 `nvfp4` 版本。

> 本仓库的 `workflows/h3-i2v-2080ti-cutlass.json` 已经改成 int4 版本，
> 并且删掉了原模板里指向 nvfp4 文件的下载 URL，避免 ComfyUI 自动去拉错文件。

---

## 坑 #3 · 头数不匹配

H3 有 **28 头**和 **56 头**两种结构的权重：

| 模型 | 头数 | 能用 Veda？ |
|---|---|---|
| `fl2va_pruned_int8_convrot` | 56 | ✅ |
| `fused_refdelta_r1024_turbo8_*` | 28 | ❌ |

把 56 头训练的打分器套到 28 头模型上，报错信息不会告诉你"头数不对"，而是：

```
This model has 50 x 28 x 128, but the predictor was trained for 50 x 56 x 128
```

**这个报错曾被误判成 kernel 不兼容**，白查了一轮。
只要看到 `predictor was trained for` 这类字样，先去核对头数。

---

## 坑 #4 · 同 seed 重跑会命中执行缓存

ComfyUI 会缓存执行结果。同样的 prompt 图 + seed + 参数，第二次提交**2 秒返回**，
产物 MD5 与第一次**逐字节相同**。

这会造成两种误判：

- 以为加速方案生效了（快了 100 倍）
- 以为质量问题消失了（其实是同一份产物）

**验收规范**：

```
✅ 每轮换 seed
✅ 记录端到端耗时，判断是否落在预期量级
✅ 比对产物 MD5，必须不同
✅ ffprobe 检查时长/帧数/音频轨
```

只看 `Prompt executed` 一行是不够的。

---

## 坑 #5 · 多卡引擎的 s/it 会骗人

H3VM 是异步流水。进度条显示的是"已提交的步"，不是"已完成的工作"，
所以它天然跑在真实进度前面，显示一个偏小的 s/it。

实测反例：

| 配置 | 进度条 s/it | 端到端 |
|---|---|---|
| 双卡基线 | 6.5 | 253.4 s |
| 双卡 + Veda | 7.9 | 240.8 s |

**s/it 说 Veda 慢 21%，端到端说 Veda 快 5%。** 两个方向相反。

**只信端到端，且必须交替顺序跑两轮。**

---

## 坑 #6 · 一次 CUDA 崩溃会污染整个上下文

崩溃之后，"再点一次运行"通常会接着崩，甚至崩在更早的位置。

| 平台 | 恢复方式 |
|---|---|
| Linux | `systemctl restart comfyui-*` |
| Windows | 关掉 ComfyUI 进程重开（不是刷新页面） |

**不要试图在同一个进程里恢复。** CUDA 上下文中毒是不可逆的，
继续尝试只会让错误信息越来越偏离真相。

---

## 附：还有几个地方会浪费你半天

| 现象 | 原因 | 处理 |
|---|---|---|
| 日志出现 `Skip .../veda_sm75 module ... due to the lack of NODE_CLASS_MAPPINGS` | 0.37 加载器先打 ERROR 再走 `comfy_entrypoint` 分支 | **误导性报错**。查 `/object_info` 是否有该节点，有就是正常的 |
| 注意力补丁"装上了但没生效" | 节点放错位置（在 LoRA 之前 / guider 之后） | 必须在 MODEL 链最末端 |
| 补丁节点接完就报 `'H3SnapshotPrefixProxy' object has no attribute 'attn'` | 接在 H3VM 引擎之后了 | 挪到引擎之前 |
| `VAE is invalid: None` | 选了 Sol-VAE，ComfyUI < 0.44 不支持 | 换回 `minimax_h3_video_vae_fp16` |
| 二采直接 OOM | 双卡峰值已 21.4/22 GiB | 降分辨率或关二采，参数腾不出空间 |
