# 10 · 前端兼容：升级后最容易炸的一片

ComfyUI 的前端（`comfyui_frontend_package`）在 1.5x 版本做了大量重构：
从手写 JS 变成 Vite 压缩产物、插槽几何改成动态测量、widget 也计入输入槽。
**任何"接管核心绘制函数"的扩展都会在这一轮里失配。**

这一篇记录三种实际炸过的形态，都有明确的判据和处置。

---

## 通用原则

> **升级前端后，优先怀疑所有接管 `drawConnections` / `drawNode` / `drawBackCanvas` 的显示类扩展。**

它们靠假设 litegraph 的内部结构工作，而前端每次重构都会改这些结构。
判据通常是"**所有**工作流一起坏"，而不是某个文件坏——这是原型级接管被全局污染的标志。

好消息是：这类问题**大多不用改代码、不用重启后端**，改个设置或刷个页面就回来了。

---

## 形态一：连线变成直角折线，端点不落在插槽上

### 症状

- 打开**任何**工作流，连线都是直角折线（本该是贝塞尔曲线）
- 端点飘在节点上方或旁边的空白处
- 节点本身位置、大小都正常
- 换个工作流、清浏览器缓存、缩放画布，全都无效

### 根因

`quick-connections` 的 **Circuit Board Lines（电路板走线）** 接管了
`LGraphCanvas.prototype.drawConnections`。它拿到正确的 `node.getOutputPos()` 之后，
**又把坐标改写成自己维护的包围盒边界**：

```js
outputXY[0] = outputNodeInfo.linesArea[2];   // 包围盒右边界
inputXY[0]  = nodeInfo.linesArea[0] - 1;     // 包围盒左边界
```

`linesArea` 来自 `getBounding()` + 节点排序 + `lineSpace` 累加，是按老版 litegraph 写的。

前端升到 **1.55.17** 后，节点包围盒把 widget 也计为输入槽
（`_measureSlots` / `_arrangeWidgetInputSlots` / `_setConcreteSlots`），
这套假设对不上 ⇒ 端点被推出节点外。

**因为它是全局原型级接管，所以所有工作流一起错。**

### 处置：改一个设置，不改代码

```bash
curl -X POST -H "Content-Type: application/json" \
     -d "0" http://127.0.0.1:8188/api/settings/circuit-board-lines.enable
```

或者设置里搜 `Circuit Board`，把 `Circuit Board lines` 改成 `Off`。

**刷新页面（F5）即生效，不需要重启后端。**

### 怎么确认是这个原因（而不是文件坏了）

做 A/B：在浏览器侧拦截 `/extensions/*.js`，不动服务器。

```
A 组（原样）：直角折线 + 端点偏离   ❌
B 组（全拦）：贝塞尔曲线 + 端点准确  ✅
```

再二分：只拦 `quick-connections` → 恢复正常；只拦别的 → 仍错位。

同时可以验证数据本身没坏：检查 `links` 与 `nodes.inputs/outputs` 的双向一致性，
如果没有悬空 link、没有重复 id，就说明**文件是好的，问题在显示层**。

> 这条区分很重要。方向搞错了会去改 JSON，越改越乱。

### 如果确实想保留电路板风格

把 `CircuitBoardLines.js` 里那两行赋值改成直接用 `node.getSlotPosition(slot, isInput)`，
`linesArea` 只保留做"避让判断"。风险是该扩展还会覆盖 `Comfy.LinkRenderMode`，要一并放开。

---

## 形态二：连线整体放大并偏移（只在缩放 ≠ 100% 时出现）

### 症状

和形态一很像，但有个关键区别：**只在系统缩放 125%/150% 或页面缩放 >100% 时出现**，
dpr=1 时完全正常。

### 根因

`comfyui-mixlab-nodes` 的 `web/javascript/td_background.js` 覆盖了
`LGraphCanvas.prototype.drawBackCanvas`——**连线层就画在这个背景 canvas 上**。

它里面有一段旧式 dpr 判断：

```js
let scale = LGraphCanvas.prototype.drawBackCanvas.toString()
              .match('window.devicePixelRatio') ? window.devicePixelRatio : 1
ctx.setTransform(scale, 0, 0, scale, 0, 0)
```

新版前端自己已经把 bgcanvas 按物理像素放大并处理 dpr，mixlab 又乘了一次
⇒ **连线层被二次放大 dpr 倍**，而节点画在前景 canvas 上不受影响 ⇒ 两层比例不一致。

### 处置：让覆盖函数直接透传

在覆盖函数开头接管原生实现，不再自行绘制连线层：

```js
const _nativeDrawBackCanvas = LGraphCanvas.prototype.drawBackCanvas

LGraphCanvas.prototype.drawBackCanvas = function () {
  if (_nativeDrawBackCanvas) {
    return _nativeDrawBackCanvas.apply(this, arguments)
  }
  // ...原有兜底逻辑
}
```

**副作用**：mixlab 的"TouchDesigner 背景图"功能随之停用（就是把输出图当画布背景那个）。
要恢复就还原备份文件。

⚠️ 备份文件后缀**不能是 `.js`**，否则会被当成插件加载。用 `.js.bak_日期`。

### 验证

用同一工作流、同一代码，只改 `devicePixelRatio` 渲染：

| dpr | 结果 |
|---|---|
| 1.0 | 连线精确落在插槽上 |
| 1.5 | 连线整体放大 1.5 倍并偏移（= 故障现象） |

修复后 dpr=1.5 应该与 dpr=1 的基线**一致**。

---

## 形态三：加载工作流弹窗报错，但节点已经画出来了

### 症状

```
由于重新加载工作流数据出错，加载被中止
TypeError: t.findInputSlot is not a function
这可能由于以下脚本：/extensions/ComfyUI-Impact-Pack/Impact-pack.js
```

节点其实已经渲染出来了，但加载流程以报错收场。

### 根因

`Impact-Pack` 用 `stackTrace.includes('loadGraphData')` 判断"是否正在加载"。
新版前端是 **Vite 压缩产物，函数名被压成单字母**，这个判据永远返回 false
⇒ hotfix 被跳过 ⇒ 走到旧签名的 `node.connect(slot, node.id, 'input1')` ⇒ 抛错中止整图。

### 处置

给它打补丁：判据补上 `app.configuringGraph` / `configuringGraphLevel`，并加 try/catch 兜底。

```js
const isLoading = stackTrace.includes('loadGraphData')
                || app?.configuringGraph
                || app?.configuringGraphLevel > 0
```

**只影响含 `ImpactSwitch` 的工作流。** 改前备份 `impact-pack.js.bak_日期`。

⚠️ Manager 更新该包会覆盖补丁。JS 是 `Cache-Control: no-store`，
**刷新浏览器即生效，不用重启 ComfyUI**。

---

## 形态四：`structuredClone` 加载失败

### 症状

```
Failed to execute 'structuredClone' on 'Window':
#<Object> could not be cloned.
```

同样：节点画出来了，但加载报错。

### 根因链

```
loadGraphData
  → afterLoadNewGraph(workflow, rootGraph.serialize(), …)
  → activateLoadedWorkflow
  → createNewTemporary
  → createNewWorkflow
  → ensureWorkflowId(structuredClone(t))     ← 这里炸
```

`t` 就是 `serialize()` 的产物。**只要某个节点把 Proxy / 类实例 / 函数塞进 widget 值，就必炸。**

### 处置：装一个纯前端守卫

写一个极小的扩展，给 `LGraph.prototype.serialize` 套一层：
序列化结果不可克隆时，降级成 `JSON.parse(JSON.stringify())`。

```js
const _orig = LGraph.prototype.serialize
LGraph.prototype.serialize = function () {
  const out = _orig.apply(this, arguments)
  try {
    structuredClone(out)
    return out
  } catch {
    return JSON.parse(JSON.stringify(out))
  }
}
```

⚠️ **新扩展必须重启一次后端才会被 `/extensions` 登记。**

### 怎么定位是哪个节点

1. 页面控制台里跑 `structuredClone(app.rootGraph.serialize())`，确认能不能复现
2. 按**节点类型**单独加载，二分
3. 还不行就按**节点 id** 二分

已知会触发这个问题的节点类型包括带自定义编辑器状态的导演类节点
（如 `LTXDirector`、`PointsEditor` 这类）。它们把编辑器实例挂进了 widget。

---

## 附：一个纯前端的"坑中之坑"

**`SetNode` / `GetNode` 不是缺插件。**

新版 KJNodes 把它们重写成**纯前端虚拟节点**（在 `web/js/setgetnodes.js` 里注册），
Python 侧根本不注册。所以节点扫描列表里必然没有它们，
但浏览器里能正常显示、连线，执行前会被剥掉。

**别为了这两个节点去装包或改工作流。**
