# thermalmodel — ThermalGuidanceHRNet 热预测

2.5D chiplet 布局热场预测（HRNet 多尺度结构）。输入功率图 + 布局掩码 + 总功耗标量，输出温度场 + 平均温度。

## 文件

| 文件 | 作用 |
|---|---|
| `HRNet.py` | **主入口**：模型定义 + loss + `train`/`test` 两个子命令（自包含 CLI） |
| `dataLoader.py` | 数据集读取（`ThermalDataset` / `MinMaxStats` / `split_cases_by_i` / `compute_minmax`） |
| `auto_train.sh` | 训练 + 按 val `mean_rmse` 选最优 ckpt + test 评估 的一键脚本 |
| `eval_hrnet_ckpt.py` | 单个 ckpt 独立评估（val/test 全指标 + top-k 图） |
| `auto_val.sh` | 对某目录所有 ckpt 逐个 val 评估 → 选最优 → test 评估 + top-50 图 |
| `hrnet_test.py` | 按 val 选最优 → test 评估（flow_GCN 迁移，路径需改） |
| `hrnet_hotspot_time.py` | 推理耗时 benchmark（flow_GCN 迁移，路径需改） |
| `eval_placement_json.py` | 从 placement JSON 直接推理温度热图 |
| `draw_thermal_fig.py` | 热图可视化（chiplet 边框叠加） |
| `train.py` | 旧 128×128 训练入口（已被 `HRNet.py` 取代） |

## 数据集

路径：`Dataset/dataset/thermal_dataset_64`（相对项目根 `/root/placement/flow_tap`），由
`gen_dataset/gen_thermal_dataset.py --grid 64` 生成。命名中 `i` = layout 索引，`j` = 该 layout 下的 case。

| 子目录 | 文件 | 形状 |
|---|---|---|
| `power_map/` | `system_power_{i}_{j}.csv` | 64×64（4096 行 `idx,value`） |
| `thermal_map/` | `system_temp_{i}_{j}.csv` | 64×64（温度真值 ℃） |
| `total_power/` | `system_totalpower_{i}_{j}.csv` | 标量（总功耗 W） |
| `avg_temp/` | `system_avgtemp_{i}_{j}.csv` | 标量（平均温度 ℃） |
| `layout_mask/` | `system_mask_{i}.csv` | 64×64（chiplet 掩码） |
| `config/` | `system_{i}_config/` | HotSpot FLP/config |

归一化统计量（power / total_power / temp 的 min–max）只在 train split 上计算（`compute_minmax`）。

## 模型

当前 `HRNet.py` 的输入/输出约定：

- `power_grid` `(B,1,128,128)`、`layout_mask` `(B,1,128,128)`、`total_power` `(B,1)`
- 内部拼接 x/y 坐标图 → stem `(B,4,128,128)`，stem stride-2 → 64×64
- 三分支（分辨率 / 通道）：64×64/`base`、32×32/`base*2`、16×16/`base*4`，每 stage 多尺度 exchange
- 输出 `temp_grid` `(B,1,64,64)`（归一化 [0,1]）与 `avg_temp` `(B,1)`

loss = `guidance_loss`：热点加权 MSE（默认 `linear`，weight = 1 + 3·norm_t）+ `grad_w`·梯度 loss +
`avg_w`·avg loss + `mean_consistency_w`·均值一致性；`topk_w` / `peak_w` / `maxpool_w` 默认关闭。

## ⚠️ 已知不一致（当前代码 vs 数据集）

`HRNet.py`（`main_train`/`main_test`）与 `eval_hrnet_ckpt.py` 里的路径与网格是硬编码常量，当前值来自
flow_GCN 迁移，与 `thermal_dataset_64` 不一致：

| 项 | 代码当前值 | 数据集实际 |
|---|---|---|
| 数据集路径 `thermal_map_rel` | `Dataset/dataset/output/thermal/thermal_map` | `Dataset/dataset/thermal_dataset_64` |
| 配置路径 `hotspot_cfg_rel` | `Dataset/dataset/output/thermal/hotspot_config` | `Dataset/dataset/thermal_dataset_64/config` |
| `power_grid_size` | 128 | 64 |
| `temp_grid_size` | 64 | 64 |

即：代码按「功率 128×128 → 温度 64×64」设计，而 `thermal_dataset_64` 是「功率 / 温度均 64×64」。
要用该数据集，需把上面路径 / 网格改一致，且模型 stem 需 64→64（stride 1，见历史 `old_hrnet_64.py`），
或重新生成 128×128 的 power_map。

## 使用顺序

环境：`conda run -n chipdiffusion python ...`（torch + GPU）。

**1）训练**（一键脚本，等价于调 `HRNet.py train`）：

```bash
cd /root/placement/flow_tap
bash thermalmodel/auto_train.sh --full
```

手动等价命令：

```bash
conda run -n chipdiffusion python thermalmodel/HRNet.py train \
  --epochs 200 --batch_size 32 --lr 2e-4 --base 96 \
  --stages 4 --blocks_per_stage 2 --expand_ratio 2 \
  --grad_w 0.1 --avg_w 0.1 --mean_consistency_w 0.1 \
  --topk_w 0.0 --topk_k 0 --peak_w 0.0 \
  --seed 0 --ckpt_every 5
```

小数据快速试跑：`bash thermalmodel/auto_train.sh --small`。

**2）评估单个 checkpoint**：

```bash
conda run -n chipdiffusion python thermalmodel/eval_hrnet_ckpt.py \
  --ckpt <ckpt.pth> --split test --seed 0 --eval_bs 32 \
  --topk 50 --out_fig_dir <fig_dir>
```

或直接用主入口的 test 子命令：

```bash
conda run -n chipdiffusion python thermalmodel/HRNet.py test \
  --ckpt <ckpt.pth> --split test --batch_size 32
```

**3）对训练目录所有 ckpt 选最优 → test**（注意 `auto_val.sh` 内路径仍是 flow_GCN，需先改）：

```bash
bash thermalmodel/auto_val.sh <ckpt_dir>
```

**4）从 placement JSON 推理温度图**：

```bash
conda run -n chipdiffusion python thermalmodel/eval_placement_json.py \
  --placement_json <xxx.json> --ckpt <ckpt.pth> --out_dir <dir>
```

## 已确认最优超参数

来源：`fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0`

| 参数 | 值 |
|---|---|
| base | 96 |
| lr | 2e-4 |
| batch_size | 32 |
| stages | 4 |
| blocks_per_stage | 2 |
| expand_ratio | 2 |
| grad_w | 0.1 |
| avg_w | 0.1 |
| mean_consistency_w | 0.1 |
| topk_w / topk_k | 0.0 / 0 |
| peak_w | 0.0 |
| under_w | 1.0（默认） |
| hotspot_mode | linear（默认） |
| epochs | 200（最优 ckpt 在 ep175） |
| seed | 0 |
| limit_train / limit_val | 0（全量） |
| 精度 | fp32（无 amp） |
