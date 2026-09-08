# GNN + HRNet 热预测模型 (gnnhrnet.py)

把「GNN 的 chiplet 级物理/交互信息」与「HRNet 的多尺度温度场解码」结合起来,预测
64×64 温度场。相比纯 U-Net 场头,HRNet 场头**全程保留 64×64 高分辨率分支**、
用多尺度交换单元做密集跨尺度融合,既能有全局感受野又不抹平热点细节。

> 本文档说明两部分:**(一) 数据从哪里来、张量长什么样(输入)**;**(二) 模型吐出什么(输出)**。
> 数据侧的原始来源在 `dataLoader.py` + `gnnhrnet.py` 的 `GNNThermalDataset`。

---

## 一、整体结构

```
                      ┌───────────────────────────────────────────────┐
 chiplet 图 ──────────► GNN 编码器 (GATv2×3) ──► 全局图向量 global_cond [B,256]
 (节点/边/batch)       │                                            │ FiLM 调制
                       └────────────────────────────────────────────┤
 场栅格 [B,3,64,64] ──► HRNet 场头 (64/32/16 多尺度分支) ◄────────────┘
                                └─► heatmap [B,1,64,64]   (唯一输出)
```

- **GNN 编码器**:对 chiplet 图做消息传递,产出每个 chiplet 的嵌入;再 mean+max 池化成
  每个布局的全局图向量 `global_cond`。
- **HRNet 场头**:多尺度 CNN,把场栅格解码成温度场,每一分支都用 `global_cond` 做 FiLM
  调制 —— 把 chiplet 级信息(功率/尺寸/位置/hubump 交互)注入空间重建。
- 模型**只输出**归一化的 64×64 温度场 `heatmap`;不再有单独的峰值/节点峰值回归头。

---

## 二、输入:数据来源与张量

### 2.1 `dataLoader.py` 的角色(已清理)

`dataLoader.py` 现在只保留被 `gnnhrnet.py` 复用的底层工具函数(旧 128×128 的
`ThermalDataset` 数据类及其配套 `compute_minmax` / `MinMaxStats` / `flp_to_mask` /
`vec_to_grid` 等,已随纯 HRNet 管线一并删除):

| 函数 | 作用 |
|---|---|
| `list_cases(dir)` | 扫描 `power_map` 目录,列出所有 case 索引 `(i, j)` |
| `split_cases_by_i(cases, seed)` | 按布局 `i` 划分 80/10/10(train/val/test),同 `i` 的所有 `(i,j)` 归入同一划分 |
| `parse_flp_rects(flp)` | 解析 `.flp` 里的矩形 `(x,y,w,h,name)`,排除 TIM 块 |
| `interposer_side_m(flp)` | 求 interposer 方形边长(米) |
| `read_index_value_csv(path)` | 读 `idx,value` 两列 CSV → `np.ndarray` |
| `read_scalar_csv(path)` | 读单值文件 → `float` |
| `minmax_scale(x, vmin, vmax)` | min-max 归一化 |

> GNN 模型的数据集是 `gnnhrnet.py` 里的 `GNNThermalDataset`(下节),它复用上表这些
> 工具函数,并额外在 `load_case()` 里读取 `.ptrace` / `L4_ChipLayer.flp` 构建图与场栅格。

### 2.2 GNN 模型真正读取的原始文件(`thermal_dataset_64`)

数据根目录 `Dataset/dataset/thermal_dataset_64`。每个 case 由 `(i, j)` 标识:
`i` = 布局(layout),`j` = 该布局下的功率/温度样本。

`load_case(i, j)` 读取以下文件:

| 文件(相对 `thermal_dataset_64`) | 内容 |
|---|---|
| `config/system_{i}_config/system.flp` | chiplet 矩形几何 |
| `config/system_{i}_config/system_{i}L4_ChipLayer.flp` | chiplet / Ubump 几何 → interposer 边长、hubump 环宽 |
| `config/system_{i}_config/system_{i}_{j}.ptrace` | 每个 chiplet 的功率(W),两行(表头 + 值) |
| `thermal_map/system_temp_{i}_{j}.csv` | 64×64 仿真温度场(标签) |
| `max_temp/system_maxtemp_{i}_{j}.csv` | 该 case 的全局最高温标量(标签) |

> 归一化常数(写死在 gnnhrnet.py):温度 `TEMP_MIN=45.22`, `TEMP_MAX=276.12`;
> 特征 `POWER_SCALE=200.0`(W)、`HUBUMP_SCALE=2.0`(mm)、`POWER_GRID_SCALE=10.0`(W/mm²)。

### 2.3 一个 batch 的输入张量(collate 之后)

一个 batch 有 5 个图/场张量 + 3 个标签张量:

#### (1) 节点特征 `x` — `[N, 8]` (float32)

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

#### (2) 边 `edge_index` / `edge_attr`

- `edge_index`: `[2, E]` (long),全连接有向边(i→j,i≠j)。
- `edge_attr`: `[E, 1]` (float32),两 chiplet 中心距(按 `side_mm` 归一化)。

#### (3) `batch` — `[N]` (long)

每个节点所属布局的 id(0..B-1)。

#### (4) 场栅格 `field` — `[B, 3, 64, 64]` (float32)

3 通道,由 chiplet 矩形栅格化到 64×64(row=y,col=x,覆盖 `[0, side_mm]²`):

| 通道 | 内容 | 取值 |
|---|---|---|
| 0 | 功率密度 | `power / area / 10.0` (W/mm² 归一化),背景 0 |
| 1 | chiplet 掩码 | 二值 0/1 |
| 2 | hubump 环掩码 | 二值 0/1(Ubump 条并集) |

> 模型内部还会追加 2 张坐标图 x/y ∈ [-1,1],所以 HRNet 场头实际输入是 5 通道。

#### (5) 标签(loss 用,推理时不需要)

| 键 | 形状 | 内容 |
|---|---|---|
| `temp` | `[B, 1, 64, 64]` | 归一化仿真温度场(监督目标) |
| `peak` | `[B, 1]` | 全局最高温(归一化标量) |
| `peak_loc` | `[B, 2]` | 温度场 argmax 位置 `[row, col]`(峰值窗口损失用) |

---

## 三、输出:模型吐出什么

模型 `forward(x, edge_index, batch, edge_attr, field_raster)` 的返回值只有一个:

| 输出 | 形状 | 内容 |
|---|---|---|
| `heatmap` | `[B, 1, 64, 64]` | 归一化温度场,`T = heatmap·(276.12 − 45.22) + 45.22` (°C) |

即模型预测的是**整张 64×64 温度场**(单一通道,值域约 [0,1])。要得到物理温度,用上面的
min-max 反归一化公式。峰值温度由 `heatmap` 上的最大值间接得到(评估时 `pred_max = heatmap.amax`)。

---

## 四、模块细节

### GNNEncoder (`hidden=128, heads=4, num_layers=3, edge_dim=1`)

`Linear(8→128)` → 3 × (`GATv2Conv(concat=False)` + 残差 + `LayerNorm`)。`concat=False`
即多头取均值,输出维度保持 128。

### 全局池化 `_global_pool`

每个图对节点嵌入做 **mean + max** 拼接 → `global_cond [B, 256]`(= 2·hidden)。

### HRNetFieldHead (`in_channels=3, base=64, stages=4, blocks_per_stage=2, expand_ratio=2`)

| 阶段 | 说明 |
|---|---|
| stem | `ConvGNAct(5→64, s=1)` + `LiteInvertedResidual`,输入已在 64×64 |
| 分支 | 64×64(64ch)、32×32(128ch)、16×16(256ch),经 stride-2 conv 生成 |
| FiLM | 每分支用 `global_cond` 做 `feat·(1+γ)+β` 调制 |
| 4 个 stage | 每 stage:3 个 `HRBranch`(各 2 个 LiteInvertedResidual) + `ExchangeUnit` 多尺度融合 |
| 头部 | 32/16 上采样到 64,concat `[x64, u32, u16, raster]` → `ConvGNAct` → 1×1 conv → `[B,1,64,64]` |

`LiteInvertedResidual` 是 MobileNetV3 风格倒残差块(SiLU + GroupNorm + 3×3 depthwise)。

---

## 五、损失函数

```
loss = 1.0·MSE(heatmap)              # 全局热图均方误差
     + 0.15·L1(Sobel梯度差)           # grad_w: 一阶空间梯度匹配
     + 0.20·L1(拉普拉斯 ∇²T 差)       # laplace_w: 二阶曲率匹配(峰值处曲率大且为负 → 锐化峰值)
     + 0.40·MSE(峰值 3×3 邻域)        # peak_window_w: 真实峰值位置周围 3×3 窗口监督
```

- **Sobel 梯度损失**:`_spatial_gradient_loss`,匹配 x/y 方向一阶梯度,惩罚空间结构走样。
- **拉普拉斯损失**:`_laplacian_loss`,匹配 ∇²T。峰值处曲率大且为负 → 逼模型锐化热点;
  平滑区曲率小 → 约束平滑。
- **峰值窗口损失**:`_peak_window_loss`,对真实 argmax 峰值位置周围 (2·1+1)²=3×3 窗口做
  MSE(而非单点)。真实仿真热点常占 2~3 格,单点监督易震荡,小窗口监督更稳。这是
  「热点峰值低估」问题的主修项。

> 这一组权重就是 **q_l1_pwin** 配置:全量 5-epoch val 上 `hm_rmse 0.658°C`、
> `peak_bias -0.169°C`(峰值几乎无偏)。

---

## 六、训练配置

训练入口 `./auto_train.sh`(默认 200 epochs),全部超参如下:

| 类别 | 参数 | 值 |
|---|---|---|
| 训练量 | epochs | 200 |
| 批量 | batch_size | 64 |
| 优化器 | optimizer / lr / weight_decay | AdamW / 2e-4 / 1e-4 |
| 梯度 | grad_clip | 1.0 (max_norm) |
| 学习率 | scheduler | CosineAnnealingLR(T_max=epochs),衰减到 0 |
| 数据 | num_train / num_val | 64000 / 8000(总 80000,8:1:1 按布局 `i` 划分,seed=0) |
| 并行 | num_workers | 28 |
| GNN | hidden / heads / num_layers | 128 / 4 / 3(GATv2) |
| 场头 | grid / base / stages / blocks_per_stage / expand_ratio | 64 / 96 / 4 / 2 / 2 |
| 损失 | grad_w / laplace_w / peak_window_w | 0.15 / 0.2 / 0.4(q_l1_pwin) |
| 评估 | hotspot_thr | 0.05 |
| 保存 | save_every | 5(每 5 epoch 存 `checkpoint_epXXX.pth`) |

- 参数量:**15.71M**(场头 base=96 比 base=64 更宽)
- 保存:每 5 个 epoch 存 `checkpoints/gnnhrnet_pwin/checkpoint_ep{ep:03d}.pth`;
  val 热图 RMSE 更优时另存 `best.pth`

训练 / 评估入口:
```bash
./auto_train.sh                                     # 训练 200 epochs
./auto_val.sh checkpoints/gnnhrnet_pwin/best.pth    # 在 val/test 上评估
```
