# GNN + HRNet 热预测模型 (gnnhrnet.py)

把「GNN 的 chiplet 级物理/交互信息」与「HRNet 的多尺度温度场解码」结合起来,预测
64×64 温度场 + 峰值温度。相比纯 U-Net 场头,HRNet 场头**全程保留 64×64 高分辨率分支**、
用多尺度交换单元做密集跨尺度融合,既能有全局感受野又不抹平热点细节。

## 一、整体结构

```
                      ┌─────────────────────────────────────────────┐
 chiplet 图 ──────────► GNN 编码器 (GATv2×3) ──► 全局图向量 global_cond [B,256]
 (节点/边/batch)       │      │                                     │
                       │      └─► PeakHead ──► peak [B,1]            │ FiLM 调制
                       │             └─► node_peak [N,1]             │
 场栅格 [B,3,64,64] ──► HRNet 场头 (64/32/16 多尺度分支) ◄────────────┘
                                └─► heatmap [B,1,64,64]
```

- **GNN 编码器**:对 chiplet 图做消息传递,产出每个 chiplet 的嵌入;再 mean+max 池化成
  每个布局的全局图向量 `global_cond`。
- **PeakHead**:每 chiplet 回归峰值温度,再用 hard max 池化得到布局全局峰值(无偏)。
- **HRNet 场头**:多尺度 CNN,把场栅格解码成温度场,每一层都用 `global_cond` 做 FiLM
  调制 —— 把 chiplet 级信息(功率/尺寸/位置/hubump 交互)注入空间重建。

## 二、输入格式与内容

一个 batch 有 4 个张量 + 1 个场栅格:

### 1. 节点特征 `x` — `[N, 8]` (float32)

`N` = 该 batch 内所有布局的 chiplet 总数(每布局 3~20 个)。每行 8 维,均已归一化:

| 维度 | 内容 | 归一化 |
|---|---|---|
| 0 | chiplet 功耗 | `power / 200.0` (W) |
| 1 | 宽度 | `w / side_mm` |
| 2 | 高度 | `h / side_mm` |
| 3 | 面积 | `area / side_mm²` |
| 4 | 中心 x 坐标 | `xc / side_mm` |
| 5 | 中心 y 坐标 | `yc / side_mm` |
| 6 | hubump 环宽 | `hb / 2.0` (mm) |
| 7 | hubump 环面积 | `2·(w+h)·hb / side_mm²` |

`side_mm` = interposer 方形边长(mm)。`hubump` 是每个 chiplet 外围一圈 Ubump 条构成的
垂直热导环,宽度从 `L4_ChipLayer.flp` 解析。

### 2. 边 `edge_index` / `edge_attr`

- `edge_index`: `[2, E]` (long),全连接有向边(i→j,i≠j)。
- `edge_attr`: `[E, 1]` (float32),两 chiplet 中心距(按 `side_mm` 归一化)。

### 3. `batch` — `[N]` (long)

每个节点所属布局的 id(0..B-1)。

### 4. 场栅格 `field_raster` — `[B, 3, 64, 64]` (float32)

3 通道,由 chiplet 矩形栅格化到 64×64(row=y,col=x,覆盖 `[0, side_mm]²`):

| 通道 | 内容 | 取值 |
|---|---|---|
| 0 | 功率密度 | `power / area / 10.0` (W/mm² 归一化),背景 0 |
| 1 | chiplet 掩码 | 二值 0/1 |
| 2 | hubump 环掩码 | 二值 0/1(Ubump 条并集) |

> 模型内部还会追加 2 张坐标图 x/y ∈ [-1,1],所以 HRNet 场头实际输入是 5 通道。

## 三、输出格式与内容

| 输出 | 形状 | 内容 |
|---|---|---|
| `heatmap` | `[B, 1, 64, 64]` | 归一化温度场,`T = x·(276.12-45.22) + 45.22` (°C) |
| `peak` | `[B, 1]` | 布局全局峰值温度(归一化,同上反归一化) |
| `node_peak` | `[N, 1]` | 每个 chiplet 身体区域的最大温度(归一化) |

## 四、模块细节

### GNNEncoder (`hidden=128, heads=4, num_layers=3, edge_dim=1`)

`Linear(8→128)` → 3 × (`GATv2Conv(concat=False)` + 残差 + `LayerNorm`)。`concat=False`
即多头取均值,输出维度保持 128。

### 全局池化 `_global_pool`

每个图对节点嵌入做 **mean + max** 拼接 → `global_cond [B, 256]`。

### PeakHead

`Linear(128→128→64→1)` 得每节点峰值 → 全图用 `index_reduce(amax)` 取 hard max。
hard max 无偏(LSE 偏高、softmax-mean 偏低),且每节点有独立监督,梯度不稀疏。

### HRNetFieldHead (`base=64, stages=4, blocks_per_stage=2, expand_ratio=2`)

| 阶段 | 说明 |
|---|---|
| stem | `ConvGNAct(5→64, s=1)` + `LiteInvertedResidual`,输入已在 64×64 |
| 分支 | 64×64(64ch)、32×32(128ch)、16×16(256ch),经 stride-2 conv 生成 |
| FiLM | 每分支用 `global_cond` 做 `feat·(1+γ)+β` 调制 |
| 4 个 stage | 每 stage:3 个 `HRBranch`(各 2 个 LiteInvertedResidual) + `ExchangeUnit` 多尺度融合 |
| 头部 | 32/16 上采样到 64,concat `[x64, u32, u16, raster]` → `ConvGNAct` → 1×1 conv → 输出 |

`LiteInvertedResidual` 是 MobileNetV3 风格倒残差块(SiLU + GroupNorm + 3×3 depthwise)。

## 五、损失函数

```
loss = 1.0·MSE(heatmap) + 0.1·Sobel梯度差 + 1.0·MSE(peak) + 1.0·MSE(node_peak)
```

## 六、训练配置

- 优化器:AdamW,`lr=5e-4`,`weight_decay=1e-4`,梯度裁剪 1.0
- 学习率:CosineAnnealingLR 衰减到 0
- batch=64,28 workers,数据划分 8:1:1(按布局 i 划分,seed=0)
- 参数量:**7.35M**;推理 ~1.76 ms/样本(batch 64,RTX 5090)
- 每 epoch 保存 `checkpoints/gnnhrnet/best.pth`(val 热图 RMSE 最优)
