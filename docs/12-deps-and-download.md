# 12 · 依赖与下载：升级后成片挂掉的真实原因

> 这一篇解决两类最常见的"环境突然坏了"：
> 节点包成片 `ImportError`，和大模型下到一半永远下不完。

---

## 一、两条环境级根因（升级后先查这两个）

ComfyUI / transformers 一升级，几十个节点包同时加载失败时，
**别一个个包去查**。九成是这两条。

### 根因 1：transformers 5.x 移除了 `HybridCache` → `import peft` 必炸

```
ImportError: cannot import name 'HybridCache' from 'transformers'
```

修法：

```bash
pip install --no-deps --upgrade peft==0.21.2
```

为什么是这个版本：0.21.2 把那句 import 挪进了 `if transformers_lt_4_56:` 分支，
5.x 下根本不执行。

**已知会连带挂掉的包**：FluxTrainer / SDPose-OOD / SeedVR2 / SkyReels-A2 / SecNodes，
以及任何依赖 `peft` 的节点。

### 根因 2：albucore 0.0.24 移除了 `median_blur` → `import insightface` 必炸

```
ImportError: cannot import name 'median_blur' from 'albucore'
```

修法是降 albucore，但它会牵扯 `albumentations`（ReActor 依赖），
动一个连带动一串。**要么整套一起降，要么暂时不修**——不要单独降一个。

**已知会连带挂掉的包**：InstantID / face-similarity。

### 为什么这两条值得单独记

因为它们的症状是"**节点包成片加载失败**"，看起来像整个环境崩了。
实际只是两个第三方库的 API 变了。不知道这一点，会从 ComfyUI 版本一路查到显卡驱动。

---

## 二、给嵌入式 Python 装包的铁律

### 一律加 `--no-deps`

```bash
python_embeded\python.exe -m pip install --no-deps <package>
```

不加的后果很具体：装一个节点的两个依赖，pip 会顺手把 `filelock` 抬到 4.x、
`packaging` 抬到 26.3，然后你的 torch 生态开始出现各种莫名其妙的报错。

**这个环境里所有包都是手工对齐过的，pip 的依赖解析只会破坏它。**

### 别照节点包的 requirements 装

典型反例：`ComfyUI-FluxTrainer` 的 requirements 里写死 `numpy<=1.26.4`。
照它装会把 numpy 从 2.2.6 降下去，**拖垮整个 torch 2.12 生态**。

而且这类包往往 HEAD 停在很久以前，跟当前的 transformers 5.x 根本不兼容
（还有 `CLIPFeatureExtractor` 这类早被移除的名字）。

> 判断标准：**这个 requirements 是不是跟着 ComfyUI 主版本一起更新的？**
> 不是的话，就只挑它真正需要的包手动装。

### `transformers 5.x` 是 `_LazyModule`，别想硬塞别名

```python
# 这样不行：hasattr 为真，但 from transformers import X 仍然失败
transformers.HybridCache = SomeClass
```

5.x 用 `_LazyModule` 做延迟加载，内存里塞属性打不进去。
想做兼容层得走 `sys.modules` 层面的包装，成本远高于直接升 peft。

---

## 三、HF 下载：直连不通时怎么办

### 实测结果

| 域名 | 状态 |
|---|---|
| `huggingface.co` | ❌ 直连超时 |
| `hf-mirror.com` | ⚠️ 能通，但 CDN 段慢 |
| `aifasthub.com` | ✅ 最稳，会跟跳到 `us.aws.cdn.hf.co` |

单路 0.7~2.3 MB/s，5 路并发合计约 8~14 MB/s。

### URL 形态

```
https://aifasthub.com/<user>/<repo>/resolve/main/<file>
```

⚠️ **没有 `/datasets/` 段**，写错会返回 401。和 HF 官方的路径结构不一样，容易搞错。

### 校验

大文件的正确 sha256 从 HF API 的 `lfs.oid` 拿，别用页面上的短 hash。
下完先校验再改名：

```bash
sha256sum model.safetensors.part
mv model.safetensors.part model.safetensors
```

---

## 四、大文件下载的坑（比想象的多）

### 1. 同一条链路前后能差 100 倍

实测一条链路上：开头 48 MB/s、中段 2~3 MB/s、尾段 0.43 MB/s 还疯狂重连。

**别用前几秒的速率估剩余时间。** 会严重乐观。

### 2. 用分块并发，别用单流

单流一旦卡住就得整体重来。正确做法是切成 16 块各自重试，
某块失败只重下那一块。

### 3. "杀进程重来"在慢链路下是负收益

重连要重新握手，反而更慢。链路已经通了就先让它跑。

### 4. 杀下载进程会留文件句柄

症状：分片能写，但不能 `os.rename`（`WinError 32`），同一文件追加也不长。

处理：等句柄释放后改名续传（`.pN` → `.partN` 这样改一次就成功了）。

```powershell
# taskkill 有时报"没有此实例"但进程还在
Stop-Process -Id <pid> -Force
```

更极端：父进程被 Force 杀掉后 curl 变孤儿，`taskkill /F` 和 WMI `Terminate` 都失败，
只能等它自己退干净（实测约 20 分钟）。

> **杀之前想清楚。杀不干净就别急着重启下载，两个进程会抢同一个文件。**

---

## 五、模型目录可以是 junction / 符号链接

多套 ComfyUI 共用一份模型时：

```powershell
# Windows，需管理员或开发者模式
New-Item -ItemType SymbolicLink -Path "D:\ComfyUI_B\models" -Target "E:\models"
```

```bash
# Linux
ln -s /data/models /opt/ComfyUI_B/models
```

**要点**：

- ComfyUI 更新不会动链接本身，所以升级后模型不用重下
- 但如果你把 `models` 换成了真目录，之前的模型就"消失"了——不是丢了，是路径变了
- 判断是链接还是真目录：`fsutil reparsepoint query <path>`（Windows）

> 我们的两个实例就是共用一份物理目录。所以"模型下到哪个盘"这件事，
> 查清楚链接指向再决定，别凭印象。

---

## 六、一个容易被忽略的稳定性问题

如果机器会**意外关机**，先看硬件日志再折腾软件。

排查顺序：

```
1. Windows 事件日志 41/6008（非正常关机）
2. 电源功率是否够（双卡 + 高功耗 CPU，瞬时峰值容易顶到保护阈值）
3. 显卡供电线是否独立、是否有转接
4. 磁盘健康（SMART / chkdsk）
```

我们遇到过连续三次掉电，最后定位到供电而非软件。
**这种情况下任何 "CUDA error" 都是结果，不是原因。**
