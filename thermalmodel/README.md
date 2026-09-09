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
| 批量 | batch_size | 32 |
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

---

## 七、训练结果 (base=96, lr=2e-4, 200 epochs)

最终在 `best.pth`(epoch 69)上的评估指标(运行 `./auto_val.sh`):

| 指标 | val (8000) | test (8000) |
|---|---|---|
| hm_rmse | 0.486 °C | 0.479 °C |
| peak_mae | 0.613 °C | 0.627 °C |
| peak_bias | -0.030 °C | -0.034 °C |
| hotspot_rmse | 1.232 °C | 1.196 °C |

- 测试集略优于验证集 → 无过拟合,泛化良好。
- `peak_bias ≈ 0`(< 0.04°C)→ 热点峰值低估问题基本解决。
- 训练在 epoch 76 手动停止,`best.pth` 保存于 epoch 69(val_hm_rmse 最低)。
- `auto_val.sh` 已显式传 `--base 96` 等结构参数,与 `auto_train.sh` 一致。

---

## 八、评估指标说明 (`eval_hrnet_ckpt.py` 输出)

所有温度误差都在**反归一化后的摄氏度 (°C)** 空间计算。一个 case 的温度场为 64×64=4096 个
像素,设预测 `pred`、真实 `gt`,误差 `e = pred − gt`。逐 case 计算后对全部 case 聚合(mean=均值、
min=最小、max=最大)。

### 8.1 整场误差 (°C)

| 指标 | 含义 |
|---|---|
| `hm_rmse` | 整场**池化** RMSE `sqrt(mean(e²))`(所有 case 所有像素合在一起求)。与训练日志的 `val_hm_rmse`/`hm_rmse` 同口径。注意:池化 RMSE ≠ 逐 case RMSE 的均值(因 RMSE 非线性,前者通常略大) |
| `mean_rmse` / `min_rmse` / `max_rmse` | 每个 case 整场 RMSE `sqrt(mean(e²))` 的均值/最小/最大。**主指标**,反映整场平均精度(热点像素占比小,对热点低估不敏感) |
| `mean_mae` | 每个 case 整场平均绝对误差 `mean(|e|)` 的均值 |
| `max_mae` | 最坏 case 的整场平均绝对误差 |
| `mean_mse` | 每个 case 整场均方误差 `mean(e²)` 的均值(单位 °C²) |
| `max_ae` | 所有 case、所有像素中**最大的单个绝对误差** `max(|e|)`(最差单点) |

### 8.2 相对误差(无量纲 / %)

| 指标 | 含义 |
|---|---|
| `mean_mape_pct` | 逐像素相对误差 `|e|/|gt|` 平均 ×100(%)。标准 MAPE |
| `mean_abs_rel` | 每个 case 的 `mean(|e|) / mean(gt)` 的均值(用该 case 平均温度归一化的绝对相对误差) |
| `mean_rel` | 每个 case 的 `mean(e) / mean(gt)` 的均值(带符号:正=整体高估,负=整体低估) |

### 8.3 峰值 / 热点 (°C)

| 指标 | 含义 |
|---|---|
| `mean_peak_ae` | 每个 case 的 `|pred 全局峰值 − gt 全局峰值|` 的均值。反映"峰值温度"能否预测准 |
| `peak_bias` | `(pred 全局峰值 − gt 全局峰值)` 的均值(带符号:正=峰值高估,负=峰值低估)。**"热点峰值低估"问题的主指标,越接近 0 越好** |
| `mean_peak_abs_rel` | 每个 case 的 `峰值绝对误差 / gt 峰值` 的均值 |
| `hotspot_rmse` | 只在功率密度 > 阈值(默认 0.05,即原始约 0.5 W/mm²)的"热点像素"上求 RMSE。反映高热区域的空间精度 |
| `mean_peak_loc_err` | 每个 case 的「预测峰值位置」与「真实峰值位置」欧氏距离的均值(单位:格点)。峰值位置 = 温度场 argmax 的 `[row,col]`。越接近 0 说明热点**定位**越准 |
| `peak_loc_hit_1` | 预测峰值落在真实峰值 **1 格邻域内**(即峰值窗口损失用的 3×3 窗口,含对角)的 case 占比(%)。直接回答"热点位置预测对不对" |
| `peak_loc_hit_2` | 预测峰值落在真实峰值 **2 格邻域内**的 case 占比(%)。 |

> 峰值约定:`pred 全局峰值` = 模型输出热图的最大值 `heatmap.amax`(模型自己预测的最大值);
> `gt 全局峰值` 来自 `max_temp` 目录的仿真最高温标量。
> 峰值位置:`pred 峰值位置` = 预测热图 argmax 的 `[row,col]`;`gt 峰值位置` = 温度场 argmax 的
> `[row,col]`(即 `peak_loc`)。定位误差单位「格点」= 64×64 网格的一格,约等于 interposer 边长 / 64 mm。

### 8.4 空间结构

| 指标 | 含义 |
|---|---|
| `mean_grad` | 预测与真实温度场的一阶空间梯度(Sobel)差的平均绝对值(单位 °C/格)。越小说明空间结构(梯度走向)越吻合 |

评估入口(逐 case 指标 + 最坏/最好 topk 图 + 日志):
```bash
python eval_hrnet_ckpt.py --ckpt checkpoints/gnnhrnet_pwin/best.pth \
    --split test --out_log logs/test.log --out_fig_dir figs/test --topk 20
```
