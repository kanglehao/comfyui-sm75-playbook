# patches/ —— 两个抗覆盖的加固钩子

这里的两段代码解决的是同一个类问题：**comfy-kitchen 的原生 CUDA 算子在本机偶发
`invalid argument`，而错误的入参本身是合法的、原样重发必定成功。**

它们被放在 `custom_nodes/` 而不是 `site-packages/` 里，原因只有一个：
**`site-packages` 会被 pip 重装冲掉，`custom_nodes` 不会。**

---

## 装法

```bash
cp -r patches/h3_convrot_warmup  ComfyUI/custom_nodes/
cp -r patches/h3_int8_guard      ComfyUI/custom_nodes/
```

重启 ComfyUI。两个包都**不注册 UI 节点**（`NODE_CLASS_MAPPINGS = {}`），
装完在节点列表里看不到任何东西——这是正常的，它们是 import 时的副作用钩子。

---

## 两个补丁分别管什么

| 补丁 | 加固的算子 | 触发它的模型 |
|---|---|---|
| `h3_convrot_warmup` | `dequantize_int8_convrot_weight_dtype` | 所有 `*_int8_convrot.safetensors` |
| `h3_int8_guard` | `quantize_int8_rowwise` | 只有把 `token_refiner` 也量化的模型（如 `fastvideo_fasth3_8step_v2_pruned_int8_convrot`） |

**两个都要装。** 只装一个，另一类模型照样崩。

### 为什么会有两个不同的算子崩

第二个的成因值得单独说，因为它解释了"为什么只有某个模型会崩"：

- 那个模型的 `token_refiner` `qkv_proj` 是 `[21504, 5376] int8`，k = 5376
- Turing 的 `shared_memory_per_block` = 49152（48 KB），fusion 路径算下来要 54272 字节 → 装不下
- 同时 fused kernel 的启用条件是 `k <= 5120 or k >= 8192`，
  **5376 正好落在被故意跳过的 5120 < k < 8192 区间**
- 于是必然走回退路径，调用裸的 `quantize_int8_rowwise`

也就是说：**模型结构 + 硬件共享内存上限 + 源码里那条"这条带子不划算"的注释，
三者一起把它逼到了未加固的路径上。**

---

## 怎么验证装上了

看 ComfyUI 启动日志，或两个包目录下自动生成的日志/状态文件：

```
installed (v2-retry): comfy_kitchen.backends.cuda.dequantize_int8_convrot_weight_dtype
is now warm-up + bounded-retry wrapper (attempts=3)
```

状态文件里会有这几个计数：

| 字段 | 含义 |
|---|---|
| `wrapped` | 钩子挂上了 |
| `retry_recovered` | 重试救回来的次数（>0 说明这个坑还在，但被挡住了） |
| `retry_gave_up` | 重试用尽仍失败（>0 说明遇到了非偶发的问题，要查） |

**`retry_recovered` 长期为 0 是好事**——说明这段时间没触发过。
但也不要因为它为 0 就删掉补丁，这个错误的出现是概率性的。

---

## 设计契约（改之前请先读）

这两段代码的可靠性来自"克制"，不是来自"努力":

1. **永不因为本文件导致 ComfyUI 启动失败** —— 顶层 `try/except` 包住整个安装过程
2. **只在已知算子路径上做事**，其它异常一律原样抛出
3. **重试有上限**（3 次），用尽则抛最后一次，绝不无限吞错
4. **只对白名单错误重试**（匹配错误信息片段），不对所有异常重试
5. **无故障时行为与不打补丁完全一致**
6. **上游改名/结构变化时打印 SKIP 并保持惰性**，不报错

第 4 条最重要。如果为了"稳妥"对所有异常都重试，OOM、形状错、权重损坏都会被重试掉，
真正的 bug 被藏起来，排查成本翻十倍。

---

## 被升级冲掉怎么判断

这两个补丁在 `custom_nodes/` 里，**ComfyUI 本体升级、pip 重装 comfy-kitchen 都不会冲掉它们**。

会冲掉的情况只有一种：手动删了目录，或者恢复了整个 `custom_nodes` 的备份。

判断方法：

```
1. 看日志里有没有 "installed (v2-retry)" 这一行
2. 看包目录下有没有 h3_warmup_status.json / h3_int8_guard_status.json
3. 最直接：跑一次多段生成，看是否复现 invalid argument
```

### 如果 comfy-kitchen 改了函数名

补丁会打印 SKIP 并保持惰性，不会崩。这时需要更新 `FUNC_NAME` 常量：

```python
BACKEND_MODULE = "comfy_kitchen.backends.cuda"
FUNC_NAME = "dequantize_int8_convrot_weight_dtype"   # ← 上游改名时改这里
```

上游版本变动时，先确认这个函数还在：

```bash
python -c "import comfy_kitchen.backends.cuda as b; print(hasattr(b,'dequantize_int8_convrot_weight_dtype'))"
```

---

## 为什么不用"改 site-packages"的方式

能改，也确实能生效——我们最初就是那么做的（直接打补丁进
`comfy_kitchen/backends/cuda/__init__.py`）。

问题是：**下一次"安装/修复依赖"，一切归零，问题原样复发，而且你不会立刻发现**
（因为症状是概率性的）。

包外替换之所以够用，是因为 `comfy_kitchen/registry.py` 的 `get_implementation()`
**每次调用都 `getattr`**。所以替换模块属性即可生效，且与版本无关。
这是这套方案能成立的技术前提。

---

## 授权

这两段代码是本仓库的一部分，MIT（见根目录 LICENSE）。
它们依赖 `comfy_kitchen` 的内部函数名，**不是稳定 API**，上游重构时可能需要跟着改。
