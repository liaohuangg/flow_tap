# ChipletFM — 2.5D 芯粒布局 Flow-Matching 扩散模型

本仓库实现并训练一个 **flow-matching** 生成模型，学习条件分布

$$p(\text{layout} \mid \text{graph}, \text{power})$$

给定芯粒连接图（netlist graph）与功耗（power），生成每个 chiplet 的 **2D 坐标**。
条件（图 + 功耗）在整个扩散过程中**固定不变**；模型只学布局几何。

训练入口 `diffusion/train_graph_thermal.py`；评估入口 `diffusion/eval_thermal_guided.py`。

---

## 1. 模型结构

### 1.1 Flow-Matching 形式化

模型不直接预测坐标，而是学习一个**速度场** $v_\theta(x_t, t)$，把标准高斯噪声
**输运**到真实布局。对每个样本：

- 噪声 $z \sim \mathcal N(0, I)$，时间 $t \sim \mathcal U(10^{-4},\, 1)$
- 插值：$x_t = (1-t)\,x + t\,z$（$x$ 为真实布局坐标）
- 目标速度：$\mathrm{target} = z - x$
- 训练目标：$\min_\theta \|\, v_\theta(x_t, t) - (z - x)\,\|^2$

`is_ports` 标记的端口节点被 mask：不参与加噪、不参与流动损失，采样时保持固定。

### 1.2 主干网络 `GeometryAttGNN`

`backbone = geometry_att_gnn`（[diffusion/networks/gnn.py](diffusion/networks/gnn.py) 的 `GeometryAttGNN`）。
输入是节点坐标 `(V, 2)`，输出同形状的速度场。

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

主干由三类模块交替堆叠：

1. **`ResGNNBlock`** — GAT 图卷积 + `LayerNorm` + `FiLM` 时间条件注入，带残差。
2. **`GeometryAttentionBlock`** — 稠密几何注意力：`q·k` 注意力 logits 加上由成对布局
   特征（相对位移、曼哈顿距离、重叠面积、功耗、热核 $P_j e^{-d^2/\sigma^2}$）经 MLP
   得到的 `pair_bias`，捕捉空间几何关系。
3. **`ThermalMessagePassing`** — 热扩散核加权消息传递：
   $w_{ij} = P_j \exp(-d_{ij}^2/\sigma^2)$，去自环、按行归一化后聚合邻居消息。

另带一个 `node_power` 编码层，以及 legality 辅助头（overlap / boundary，见 1.4）。

### 1.3 逐图归一化与掩码（多布局批处理的关键）

模型支持把**多个不同布局拼成一个 mega-graph** 一次前向，因此所有跨节点的归一化都
必须**逐图（per-graph）**：

- `_graph_max_scale`：节点特征按**本图**内节点最大值归一化（`scatter_reduce` 按
  `cond.batch` 分组求 max）。
- `_GraphGroupNorm`：`GroupNorm(1, C)` 的逐图版本 —— 对每个图单独计算节点轴均值/方差。
- `_same_graph_mask`：块对角掩码，把注意力 / 热消息传递中的**跨图**成对项置 0
  （或 `-inf`）。

实测等价性：纯 fp32 下 mega 前向与逐图前向逐位一致到 $1.4\times10^{-6}$；模型用
fp16 autocast 训练时差为 $2^{-8}$（一个 fp16 ulp），属量化噪声而非语义差异。

### 1.4 冻结代理与辅助头

训练中调用两个**冻结**代理（不更新其权重）作为辅助目标：

- **热代理** `ThermalGNNHRNet`（`gnnhrnet_pwin/best.pth`，grid 64）：渲染布局到
  `64×64` 温度网格 → HRNet（base 96，4 stages）→ 温度场；峰值经 `logsumexp` 平滑 max
  提取。
- **线长代理** `WirelengthSurrogate`（`best_wlmodel_total_60k.pt`，objective
  `log_total`）：预测 CPLEX 求解的 2.5D microbump 布线总长。

以及 legality 辅助表示头：overlap / boundary 两个 head，加上直接的重叠 / 越界惩罚。

---

## 2. 训练

### 2.1 损失函数

$$\mathcal L = \mathcal L_{\text{flow}} + w_{\text{th}}\mathcal L_{\text{thermal}} + w_{\text{wl}}\mathcal L_{\text{wirelength}} + w_{\text{bbox}}\mathcal L_{\text{bbox}} + \mathcal L_{\text{legality}}$$

- 辅助项作用在 $x_0$ 估计 $\hat x = x_t - t \cdot v_\theta(x_t, t)$ 上。
- **aux t-reweight**：辅助梯度对模型的作用是 $-t \cdot \partial\mathcal L_{\text{aux}}/\partial\hat x$，
  在 $t$ 大（$\hat x$ 仍是噪声）时几乎无梯度。开启 `flow.aux_t_reweight=true` 把权重
  除以 `1/clamp(t)`，使每个 $t$ 贡献相等。
- 权重 **warmup**：辅助权重从 0 线性升到全值，避免初始化时压垮主 flow 目标。

本轮权重（`train_flow_pareto_20k_wsl.sh`）：

| 项 | 权重 | warmup |
|---|---|---|
| thermal | 0.1 | 2000 |
| wirelength | 0.05 | 2000 |
| bbox | 0.0（关） | — |
| legality overlap / boundary head | 0.02 / 0.02 | 10000 |
| legality overlap / boundary direct | 0.05 / 0.05 | 10000 |

> ⚠️ 辅助损失是对**生成数据集时用的同一批冻结代理**算的，新数据集在 aux loss 上
> 必然更低 —— **aux loss 不能作为数据集/模型优劣的证据**，只能看真实物理评测
> （`eval_layout.py` 走真 HotSpot + CPLEX）。

### 2.2 Mega-batching（多布局批处理）

普通做法一个 step 画一个布局、沿 batch 维复制。现改为**真·多布局批处理**：

- 一个 step 采样 `layouts_per_step = 16` 个**不同**布局，用 `collate_flow_graphs`
  拼成一个 mega-graph（`sumV` 个节点，边索引按图偏移）。
- 沿 batch 维扩展 `noise_per_layout = 8` 份，每份独立采样 $(t, z)$。
- **一次前向**处理 $16 \times 8 = 128$ 个样本；辅助代理逐图循环（每个仍批量 8 份
  噪声），再按图数平均。

关键代码：`utils.collate_flow_graphs` / `split_mega_graph` / `get_batch_mega`，
`ThermalFlowMatchingModel.loss_mega`。

**热代理的显存靠梯度 checkpointing，不靠 chunk。** 主干 flow 模型对 128 个样本一次
前向只有 2.3GB，很健康；但冻结热代理 HRNet 的 field head 为反传保存的激活随
`layouts × noise × grid²` 增长，把全部 128 个温度场**带梯度**一次前向要 ~65GB，超过
5090 的 32GB，触发 WSL2 主机内存换页（首步恶化到 80s+）。因此开启
`thermal.checkpoint_field_head=true`：反传时**重算** field head 而不是存激活，峰值
65GB → ~11GB。实测稳态 **1.27s/step @ 11.4GB（chunk=2，~101 samples/s）**。

> ⚠️ chunk 不省显存：不加 checkpoint 时，N 个 chunk 的激活在 backward 之前全部同时
> 存活，峰值仍是 65GB，跟不 chunk 一样。加 checkpoint 后 chunk 才控制单次前向的
> 瞬时显存：chunk=2 = 1.27s/11.4GB（甜点）、chunk=1 = 1.47s/7.2GB、chunk=4 =
> 1.27s/19.5GB、chunk=16 换页。历史 ~51s/step 的元凶不是算力，而是 (1) 布尔掩码索引
> `x[active]` 在每个 forward/backward 里触发 `aten::nonzero` + `cudaStreamSynchronize`，
> 把 GPU 队列串行化（已改整数 `index_select`）；(2) 热代理带梯度 128 场的激活换页。
> checkpointing 修完后者。

**性能基准（16 布局 × 8 噪声 = 128 样本，单张 RTX 5090）**：

| 部件 | 显存 | 耗时 |
|---|---|---|
| flow 主干（128 样本一次前向） | 2.3GB | 0.2s |
| + 线长代理 ×16 | 2.4GB | ~1.6s |
| + 热代理（checkpoint ON，chunk=2） | 11.4GB | ~1.3s |
| **完整一步（AMP fwd+bwd+opt）** | **11.4GB** | **~1.27s** |

### 2.3 数据与超参数

- **数据集** `placement_dataset_opt_pareto`：**182,594** 条帕累托最优布局，来自
  **40,000** 个 system（源 `placement_dataset_tw` chunks 1–8）。
- **划分**（按 case 分组，无泄漏）：**146,080 train / 18,262 val / 18,252 test**。
- 可变 chiplet 数 3–20；画布（chip_size/side）44–439 mm。
- 优化器 **Adam**，`lr = 3e-4`；**AMP**（`GradScaler` + fp16 autocast）。
- `train_steps = 200000`，`monitor.every = 100`，`print_every = 500`，`seed = 61`。

### 2.4 启动

```bash
bash train_flow_pareto_20k_wsl.sh            # 默认 200k 步
bash train_flow_pareto_20k_wsl.sh --fresh    # 全新 run（不续跑旧 checkpoint）
```

checkpoint 落在 `LayoutGenModel/checkpoints/model_pareto/placement_pareto_20k/seed_61`，
训练中 `training_monitor.png` 每 100 步原子刷新。

---

## 3. 采样

### 3.1 反向 ODE 积分（`reverse_samples`）

采样即沿速度场做**欧拉积分**，把高斯噪声输运回布局：

1. 采样 $x \sim \mathcal N(0, I)$，形状 `(B, V, F)`；
2. 端口节点用 `is_ports` mask 固定到参考输入，不参与输运；
3. 步数 `num_timesteps` 默认取 `max_diffusion_steps = 200`，步长 $dt = 1 / \text{num\_timesteps}$；
4. 从 $t = 1$ 递减到 $1/\text{num\_timesteps}$：
   $$v = v_\theta(x, \text{cond}, t),\qquad x \leftarrow x - dt \cdot v \ (+ \ \text{guidance force})$$
   每步后把端口 mask 回参考值，并把坐标 `clamp` 到 `[-2, 2]`；
5. 返回最终布局 $x$。

默认评估策略 `eval_policy = open_loop`（[diffusion/policies.py](diffusion/policies.py)）
就是纯 `reverse_samples`，不加引导。

### 3.2 引导采样（默认关闭）

`guidance_mode = "none"` 时不加引导。设置 `guidance_mode` 为 `"sgd"` 或 `"opt"` 并在
`reverse_samples` 里叠加引导力 `guidance_force`（见 `FlowMatchingModel`）：

- **`"sgd"`（无状态）**：把当前 $x$ 克隆为可微变量，对引导势做
  `grad_descent_steps` 步 SGD，返回位移 `x_guided - x`。
- **`"opt"`（有状态/自适应）**：SGD/Adam 变体，legality 权重 $\alpha$ 可学习
  （$\alpha_{\text{crit}}$ 门控）。

引导势（[diffusion/guidance.py](diffusion/guidance.py)）：

| 势 | 作用 |
|---|---|
| `legality_guidance_potential` | 重叠 softmax 惩罚，拉近合法性 |
| `hpwl_guidance_potential` | 半周长线长（HPWL） |
| `bbox_area_guidance_potential` | 包围盒面积（softmax 近似） |
| `heat_repulsion_guidance_potential` | 热排斥（$P_iP_j e^{-d^2/\sigma^2}$） |

权重可按 `t` 通过 `guidance_schedule` 调度。

### 3.3 评估

```bash
python diffusion/eval_thermal_guided.py --config-name config_eval_fm \
  task=placement_pareto_20k \
  from_checkpoint=checkpoints/model_pareto/placement_pareto_20k/seed_61/best.ckpt
```

真实物理评测（`eval_layout.py`）走 **真 HotSpot**（温度）与 **CPLEX**（线长），
是唯一有效证据。

---

## 4. 环境与依赖

- 用 `chipdiffusion` conda 环境。
- 外部依赖（非 pip）：IBM CPLEX、TAP-2.5D、HotSpot。
- `wandb` 默认关闭。
- 大文件（数据集/checkpoint/日志）不进 git。

关键文件一览：

| 文件 | 作用 |
|---|---|
| `diffusion/train_graph_thermal.py` | `ThermalFlowMatchingModel` + `loss`/`loss_mega` + 训练主循环 |
| `diffusion/networks/gnn.py` | `GeometryAttGNN` 主干 + 逐图归一化/掩码 |
| `diffusion/models.py` | `FlowMatchingModel` 基类（插值、速度目标、`reverse_samples`） |
| `diffusion/utils.py` | `collate_flow_graphs` / `split_mega_graph` / `get_batch_mega` |
| `diffusion/guidance.py` | legality / hpwl / bbox / heat-repulsion 引导势 |
| `diffusion/configs/config_graph_fm.yaml` | 模型/训练超参数 |
| `diffusion/wirelength_surrogate.py` | 冻结线长代理 |
