# thermalmodel (ThermalGuidanceHRNet)

128×128 温度场预测模型 + 训练/评估脚本。

## 文件

| 文件 | 作用 |
|---|---|
| `HRNet.py` | 模型定义 + loss（自包含，仅依赖 torch） |
| `dataLoader.py` | 数据集（读 128×128 power/layout/temp CSV） |
| `train.py` | 训练入口 |
| `eval.py` | 评估入口（val/test 指标 + 可选 top-k 图） |
| `eval_placement_json.py` | 从 placement JSON 推理温度热图 |
| `draw_thermal_fig.py` | 可视化 |

## 模型

输入（batch first）：
- `power_grid` `(B,1,128,128)` — 归一化功率图
- `layout_mask` `(B,1,128,128)` — chiplet 占用掩码
- `total_power` `(B,1)` — 归一化总功率标量

内部拼接坐标图后 stem 输入为 `(B,4,128,128)`。

三分支（分辨率 / 通道）：**128×128 / `base`**、**64×64 / `base*2`**、**32×32 / `base*4`**，
每个 stage 做多尺度 exchange。输出：

- `temp_grid` `(B,1,128,128)` — 归一化温度场 [0,1]
- `avg_temp` `(B,1)` — 平均温度

loss 由 `guidance_loss` 组合：加权 MSE（热点加权 + 低估惩罚）+ 梯度 loss + avg loss + 均值一致性。

## 用法

训练（`chipdiffusion` 环境）：

```bash
conda run -n chipdiffusion python thermalmodel/train.py \
  --epochs 200 --batch_size 32 --lr 2e-4 --base 32 --ckpt_every 5
```

关键参数：`--base`（顶层通道数）、`--stages`、`--blocks_per_stage`、`--expand_ratio`、`--limit_train/--limit_val`（小数据试跑）、`--resume`。

评估：

```bash
conda run -n chipdiffusion python thermalmodel/eval.py \
  --ckpt checkpoints/hrnet_base32_seed0_ep0200.pth --split test --topk 50
```

## 数据划分

训练/验证/测试按 layout index `i` 切分（同一 `i` 的所有 `j` 同属一个 split），比例 0.8/0.1/0.1，
`seed` 控制随机划分；归一化统计量只从训练集计算。
