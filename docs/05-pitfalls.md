# 05 · 六个坑

按"卡住你的概率"排序。前两个会让你在第一天就放弃。

---

## 坑 #1 · `CUDA INT8 convrot dequantization failed: invalid argument`

### 症状

```
RuntimeError: CUDA INT8 convrot dequantization failed: invalid argument
Node Type: SamplerCustomAdvanced
```

两种触发形态，看起来像两个问题：

| 形态 | 现象 |
|---|---|
| A：进程内首次 | 第一个工作流必崩一次，**再点一次运行就好了** |
| B：跑到中途 | 前面全正常，突然在第 409 次 / 第 1101 次调用上崩 |

形态 B 更像"真 bug"，所以很容易往显存、对齐、流、动态 VRAM 上去猜。
**我们也是这么猜错的。**

### 已经排除的方向（别再走一遍）

- ❌ 全局 sage attention 开关
- ❌ `comfy-kitchen` 版本（0.2.33 / 0.2.34 / 0.2.35 的 dequant 输出**逐位相同**）
- ❌ ComfyUI 模型编译器（CUDA Graph）—— 关掉后 `graph breaks` 日志消失，崩溃一模一样
- ❌ `--cuda-malloc` / aimdo 缓冲区
- ❌ 指针对齐、张量形状、dtype、连续性
- ❌ 共享内存、跨设备 sticky、非默认流
- ❌ `allocation_context`、遗留 CUDA 错误、图捕获态、driver context 错位
- ❌ `--disable-dynamic-vram`

以上每一条都有对照实验，不是"觉得不像"。

### 现在能确定的事实

不说"根因是 X"这种过度断言，只列证据能支撑的部分：

1. **所有静态参数在失败现场都合法**——形状、步长、dtype、连续性、16 字节对齐、device、
   stream、capture 状态，dump 里逐项验过，`peek_last=0 capture=0 drv_ctx_dev=0`
2. **同一份入参原样重发必定成功**——这是最有价值的一条。它意味着状态是"可恢复"的，
   而不是输入本身有毒
3. **失败点不固定**——观测到第 409 次、第 1101 次；不是某个固定形状触发
4. **与预热无关**——早期版本在开头做了温启动，中途照样炸

所以结论只到这一步：**这是一个原生算子在特定时序下返回的偶发错误，入参和版本都没问题。**

早期版本（2026-09-19）曾把它归因为"进程内第一次 kernel launch 的冷启动假失败"。
2026-09-21 复测推翻了那个说法：温启动成功之后，仍在第 409 次调用崩。
**别再按"冷启动"去修了。**

### 解法：预热 + 有界重试

组合拳，缺一不可：

| 手段 | 作用 |
|---|---|
| 温启动（warm-up） | 覆盖形态 A。用哑元张量先把 kernel 跑一遍 |
| 有界重试（bounded retry） | 覆盖形态 B。**重试前先 `torch.cuda.synchronize()`**，把驱动侧错误槽里的残留清掉 |
| 只对已知错误重试 | 用错误信息匹配，其它异常原样抛出 |
| 重试用尽则抛 | 绝不静默吞错，否则真 bug 会被藏起来 |

代码骨架：

```python
TRANSIENT_MARKERS = ("convrot dequantization failed",)

def _drain_device(device):
    """Retry the sync once: a failing sync is itself the stale-error readout."""
    import torch
    for _ in (1, 2):
        try:
            torch.cuda.synchronize(device)
            return
        except Exception:
            pass

def _call_with_retry(real, args, attempts=3, device=None):
    for attempt in range(1, attempts + 1):
        try:
            return real(*args)
        except Exception as exc:
            if not any(m in str(exc) for m in TRANSIENT_MARKERS):
                raise                      # 不是已知错误，原样抛出
            if attempt == attempts:
                raise                      # 次数用尽，抛最后一次
            _drain_device(device)
    raise AssertionError("unreachable")
```

三个必须遵守的约束：

| 约束 | 原因 |
|---|---|
| **只对白名单错误重试** | 否则会把 OOM、shape 错、权重损坏也重试掉，排查成本翻十倍 |
| **重试前必须 sync** | 不清错误槽的话，重试会读到同一个残留错误，白试 |
| **重试要计数并写日志** | 出现 `recovered on attempt N` 就说明这个坑还在，恢复率是健康指标 |

### 姊妹算子：`quantize_int8_rowwise` 也会崩

同一台机器上还有另一个算子会报同样的 `invalid argument`：

```
RuntimeError: CUDA INT8 rowwise quantization failed: invalid argument
Node Type: SamplerCustomAdvanced
```

触发条件是**特定模型的特定结构**：

- 只有 `fastvideo_fasth3_8step_v2_pruned_int8_convrot` 这个模型把 `token_refiner` 也量化了
  （其余 H3 int8 模型的 `token_refiner` 还是 BF16）
- 它的 `qkv_proj` 是 `[21504, 5376] int8`，k = 5376
- Turing 的 `shared_memory_per_block` = 49152（48 KB），fusion 判定算下来要 54272 字节 → 装不下
- 同时 `_should_use_convrot_fused_kernel` 要求 `k <= 5120 or k >= 8192`，
  **5376 正好落在被故意跳过的 5120 < k < 8192 区间**
- 于是必然走回退路径，调用裸的 `quantize_int8_rowwise` —— 而它还没做加固

**结论：同一个加固要打两次，一个给 dequant，一个给 rowwise。**
只用其中一个，另一个模型照样崩。

### 怎么部署才不会被升级冲掉

直接改 `site-packages` 里的文件，下一次 pip 重装就没了，问题原样复发。

关键机制：`comfy_kitchen/registry.py` 的 `get_implementation()` **每次调用都 `getattr`**。
所以在包外替换 backend 模块的属性即可生效，且与版本无关。

做成独立 custom_node，在 import 时完成替换（完整实现见
[`patches/`](../patches/README.md)，本仓库直接提供两份可用的）：

```python
import comfy_kitchen.backends.cuda as backend

_real = getattr(backend, "dequantize_int8_convrot_weight_dtype")
setattr(backend, "dequantize_int8_convrot_weight_dtype",
        lambda *a, **kw: _call_with_retry(_real, a, device=...))

NODE_CLASS_MAPPINGS = {}      # 不需要 UI 节点，纯副作用
```

`custom_nodes/` 不受 pip 管理、也不会被 ComfyUI 的 git pull 覆盖，
所以这里的加固在"安装/修复依赖"之后依然存活。

---

## 坑 #2 · NVFP4 文本编码器在 sm75 上直接失败

H3 的官方模板默认用 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`。

**Turing 没有 FP4 张量核**，这个文件加载即失败。必须换成：

```
qwen3vl_32b_minimax_h3_int4_convrot.safetensors
```

这个坑的恶劣之处在于它是一个 **14 GB 的下载**——很多人下完了才发现跑不了。

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

另外 `PatchTuringCutlassAttention` 有硬约束：**`head_dim` 必须 = 128**，
否则 `RuntimeError: head_dim must be 128`。H3 正好是 128，换模型前要确认。

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

> 注意区分：坑 #1 的"原样重发必定成功"讲的是**那一次原生调用**，不是整个工作流。
> 工作流级崩溃之后，进程已经不可信了，该重启还是要重启。

---

## 附：还有几个地方会浪费你半天

| 现象 | 原因 | 处理 |
|---|---|---|
| 日志出现 `Skip .../veda_sm75 module ... due to the lack of NODE_CLASS_MAPPINGS` | 加载器先打 ERROR 再走 `comfy_entrypoint` 分支 | **误导性报错**。查 `/object_info` 是否有该节点，有就是正常的 |
| 裸脚本 import Turing 节点报 `DLL load failed` | `.pyd` 依赖 torch 先加载 | **必须先 `import torch` 再 import 节点**。ComfyUI 里正常，裸诊断时会误判成"内核坏了" |
| 注意力补丁"装上了但没生效" | 节点放错位置（在 LoRA 之前 / guider 之后） | 必须在 MODEL 链最末端 |
| 补丁节点接完就报 `'H3SnapshotPrefixProxy' object has no attribute 'attn'` | 接在 H3VM 引擎之后了 | 挪到引擎之前 |
| `VAE is invalid: None` | 选了 Sol-VAE，ComfyUI < 0.44 不支持 | 换回 `minimax_h3_video_vae_fp16` |
| 二采直接 OOM | 双卡峰值已 21.4/22 GiB | 降分辨率或关二采，参数腾不出空间 |
