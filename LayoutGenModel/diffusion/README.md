# 芯粒布局 Flow-Matching 扩散模型（2.5D Chiplet Placement）

本目录实现并训练一个 **flow-matching** 生成模型，学习条件分布

$$p(\text{layout} \mid \text{graph}, \text{power})$$

即在给定芯粒连接图（netlist graph）与功耗（power）的条件下，生成每个 chiplet 的 **2D 坐标**。这是 2.5D 芯粒（chiplet）布局生成任务的核心：条件（图 + 功耗）是固定的，模型只学布局几何。

---

## 一、模型结构

### 1. Flow-Matching 形式化

模型不预测坐标本身，而是学习一个**速度场** $v_\theta(x_t, t)$。对每个样本：

- 噪声 $z \sim \mathcal N(0, I)$，时间 $t \sim \mathcal U(10^{-4}, 1)$
- 插值：$x_t = (1-t)\,x + t\,z$（$x$ 是真实布局坐标）
- 目标速度：$\mathrm{target} = z - x$
- 训练目标：$\min_\theta \| v_\theta(x_t, t) - (z - x) \|^2$

采样（推理）时从 $x_0 \sim \mathcal N(0,I)$ 出发，用欧拉步沿 $v_\theta$ 反演
$x_{t} \leftarrow x_t - \Delta t \cdot v_\theta(x_t, t)$，逐步回到 $x$。

条件（图结构 + 功耗）在**整个扩散过程中固定不变**，通过节点/边特征注入网络；
`is_ports` 标记的端口节点被 mask，不参与加噪也不参与流动损失。

### 2. 主干网络：`GeometryAttGNN`

`backbone = geometry_att_gnn`，即 [networks/gnn.py](networks/gnn.py) 中的 `GeometryAttGNN`：

| 部件 | 配置 |
|---|---|
| 图卷积 | **GAT**（`GATv2Conv`），4 heads，concat |
| 隐藏维度 | `hidden_size = 256` |
| 分块 | 3 个 block，`hidden_node_features = [256, 256, 256]`，每 block `layers_per_block = 2` |
| 时间编码 | sinusoid，`t_encoding_dim = 32` |
| 几何注意力 | 开启，4 heads，**flash attention**，`sigma = 0.35`，pair-MLP 2 层 |
| 热消息传递 | `thermal_mp_enabled = True`，`sigma = 0.35`，power 按 `graph_max` 归一化 |
| 边特征 | `edge_features = 8` |
| 额外节点特征 | `extra_node_feature_keys = [node_power]`，`graph_max` 归一化 |
| 条件节点特征 | `cond_node_features = 2`（chiplet 尺寸） |
| 输入 / 输出 | 坐标 `(V, 2)` |

主干由三个主要模块交替组成：

1. **`ResGNNBlock`** — GAT 图卷积 + LayerNorm + `FiLM` 时间条件注入，带残差。
2. **`GeometryAttentionBlock`** — 稠密几何注意力：`q·k` 注意力 logits 加上由成对
   布局特征（距离、重叠面积、功耗、热核）经过 MLP 得到的 `pair_bias`，用于捕捉
   空间几何关系。
3. **`ThermalMessagePassing`** — 热扩散核加权消息传递：
   $w_{ij} = P_j \cdot \exp(-d_{ij}^2 / \sigma^2)$，按行归一化后聚合邻居消息。

此外带一个 `node_power` 编码层，以及 legality 辅助头（见下）。

### 3. 归一化（多布局批处理的关键）

模型支持把**多个不同布局拼成一个 mega-graph** 一次前向，因此所有跨节点的归一化
都必须是**逐图（per-graph）**的：

- `_graph_max_scale`：节点特征按**本图**内节点的最大值归一化（用 `scatter_reduce`
  按 `cond.batch` 分组求 max），避免跨布局泄漏尺度。
- `_GraphGroupNorm`：`GroupNorm(1, C)` 的逐图版本 —— 对每个图单独计算节点轴上的
  均值/方差，而不是把批内所有布局的节点混在一起。
- `_same_graph_mask`：块对角掩码，把注意力/热消息传递中的**跨图**成对项置 0（或
  `-inf`），保证每个布局只与自己的节点交互。

> 这三处是「批处理 == 逐布局单独前向」的等价性保证。实测：纯 fp32 下 mega 前向与
> 逐图前向逐位一致到 $1.4\times10^{-6}$；模型本身用 fp16 autocast 训练时，差为
> $2^{-8}$（一个 fp16 ulp），属于 autocast 量化噪声而非语义差异。

### 4. 冻结代理与辅助头

训练中调用两个**冻结**的代理模型作为辅助目标（不更新其权重）：

- **热代理** `ThermalGNNHRNet`（`gnnhrnet_pwin/best.pth`，grid 64）：预测温度场，
  峰值经 `logsumexp` 平滑 max 提取。
- **线长代理** `WirelengthSurrogate`（`best_wlmodel_total_60k.pt`，objective
  `log_total`）：预测 CPLEX 求解的 2.5D microbump 布线总长。

以及 legality 辅助表示头：overlap / boundary 两个 head，加上直接的重叠/越界惩罚。

---

## 二、训练过程

### 1. 损失函数

总损失 = 主 flow 损失 + 加权辅助项：

$$\mathcal L = \mathcal L_{\text{flow}} + w_{\text{th}}\,\mathcal L_{\text{thermal}} + w_{\text{wl}}\,\mathcal L_{\text{wirelength}} + w_{\text{bbox}}\,\mathcal L_{\text{bbox}} + \mathcal L_{\text{legality}}$$

- 辅助项作用在 $x_0$ 估计 $\hat x = x_t - t \cdot v_\theta(x_t, t)$ 上。
- **aux t-reweight**：辅助梯度对模型的作用是 $-t \cdot \partial\mathcal L_{\text{aux}}/\partial\hat x$，
  在 $t$ 大（$\hat x$ 还是噪声）时几乎无梯度。开启 `flow.aux_t_reweight=true`，把
  权重除以 `1/clamp(t)`，使每个 $t$ 贡献相等。
- 权重 **warmup**：辅助权重从 0 线性升到全值，前 2000 步爬升，避免初始化时压垮
  主 flow 目标。

本轮权重（`train_flow_pareto_20k_wsl.sh`）：

| 项 | 权重 | warmup |
|---|---|---|
| thermal | 0.1 | 2000 |
| wirelength | 0.05 | 2000 |
| bbox | 0.0（关） | — |
| legality overlap/boundary head | 0.02 / 0.02 | 10000 |
| legality overlap/boundary direct | 0.05 / 0.05 | 10000 |

> ⚠️ 重要：辅助损失是对**生成数据集时用的同一批冻结代理**算的，所以新数据集在
> aux loss 上必然更低 —— **aux loss 不能作为数据集/模型优劣的证据**，只能看真实
> 物理评测（`eval_layout.py` 走真 HotSpot + CPLEX）。

### 2. Mega-batching（充分利用 GPU）

普通做法一个 step 只画一个布局、沿 batch 维复制。现改为**真·多布局批处理**：

- 一个 step 采样 `layouts_per_step = 16` 个**不同**布局，用 `collate_flow_graphs`
  拼成一个 mega-graph（`sumV` 个节点，边索引按图偏移）。
- 把 mega-graph 沿 batch 维扩展 `noise_per_layout = 8` 份，每份独立采样 $(t, z)$。
- **一次前向**处理 $16 \times 8 = 128$ 个样本；辅助代理逐图循环（每个仍批量处理
  8 份噪声），再按图数平均。

关键代码：`utils.collate_flow_graphs` / `split_mega_graph` / `get_batch_mega`，
以及 `train_graph_thermal.py` 的 `ThermalFlowMatchingModel.loss_mega`。

**热代理的显存靠梯度 checkpointing，不靠 chunk。** 主干 flow 模型 128 样本一次前向
只有 2.3GB；但冻结热代理 HRNet 的 field head 为反传保存的激活随 `layouts × noise ×
grid²` 增长，128 个温度场带梯度一次前向要 ~65GB，超过 32GB 显存触发主机换页（首步
恶化到 80s+）。因此开启 `thermal.checkpoint_field_head=true`：反传时**重算** field head
而不是存激活，峰值 65GB → ~11GB。实测稳态 **1.27s/step @ 11.4GB（~101 samples/s，chunk=2）**。

> ⚠️ chunk 不省显存：不加 checkpoint 时，8 个 chunk 的激活在 backward 之前全部同时
> 存活，峰值仍是 65GB（≈ 8×8GB），跟不 chunk 一样。历史 ~51s/step 的元凶是布尔掩码
> 索引 `x[active]` 的 `nonzero`+`cudaStreamSynchronize` 串行化（已改整数索引），加上
> 热代理带梯度 128 场的激活换页；checkpointing 修完后者。

### 3. 数据与超参数

- **数据集** `placement_dataset_opt_pareto`：**182,594** 条帕累托最优布局，
  来自 **40,000** 个 system（源 `placement_dataset_tw` chunks 1–8，id 1–40000）。
- **划分**（按 case 分组，无泄漏）：**146,080 train / 18,262 val / 18,252 test**。
- 可变 chiplet 数 3–20；画布（chip_size/side）44–439 mm。
- 优化器 **Adam**，`lr = 3e-4`；**AMP**（`GradScaler` + fp16 autocast）。
- `train_steps = 200000`（≈ 12.5 epoch over 146,080 train 布局），
  `monitor.every = 100`，`print_every = 500`，`seed = 61`。

### 4. 启动

```bash
bash train_flow_pareto_20k_wsl.sh            # 默认 200k 步
bash train_flow_pareto_20k_wsl.sh --fresh    # 全新 run（不续跑旧 checkpoint）
```

checkpoint 落在 `LayoutGenModel/checkpoints/model_pareto/placement_pareto_20k/seed_61`，
训练过程中 `training_monitor.png` 每 100 步原子刷新。

### 5. 关键文件

| 文件 | 作用 |
|---|---|
| `train_graph_thermal.py` | `ThermalFlowMatchingModel` + `loss` / `loss_mega` + 训练主循环 |
| `networks/gnn.py` | `GeometryAttGNN` 主干 + 逐图归一化/掩码 |
| `utils.py` | `collate_flow_graphs` / `split_mega_graph` / `get_batch_mega`（mega-batching） |
| `models.py` | `FlowMatchingModel` 基类（插值、速度目标、采样反演） |
| `configs/config_graph_fm.yaml` | 模型/训练超参数 |
| `guidance.py` | bbox / 合法性 / 热排斥等引导势 |
| `wirelength_surrogate.py` | 冻结线长代理 |
