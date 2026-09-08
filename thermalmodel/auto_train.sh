#!/bin/bash
# auto_train.sh — GNN+HRNet 训练脚本 (q_l1_pwin 配置, 全部超参显式)
# 模型: GNN GATv2 + HRNet 64/32/16 场头 (gnnhrnet.py)
# 损失: MSE + grad_w=0.15 + laplace_w=0.2 + peak_window_w=0.4
#       (全量 5-epoch val: rmse 0.658 / peak_mae 1.022 / peak_bias -0.169 / hotspot 1.486)
#
# 用法:
#   ./auto_train.sh            # 完整训练 50 epochs
#   ./auto_train.sh 5          # 快速试跑 5 epochs
#
# 说明: 调度器为 CosineAnnealingLR(T_max=epochs), 在 gnnhrnet.py 内写死。
set -u
cd "$(dirname "$0")"
PY=/root/anaconda3/envs/chipdiffusion/bin/python

EPOCHS="${1:-50}"

$PY -u gnnhrnet.py \
  --epochs "$EPOCHS" \
  --batch_size 64 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --num_train 64000 \
  --num_val 8000 \
  --num_workers 28 \
  --seed 0 \
  --hidden 128 \
  --heads 4 \
  --num_layers 3 \
  --grid 64 \
  --base 64 \
  --stages 4 \
  --blocks_per_stage 2 \
  --expand_ratio 2 \
  --grad_w 0.15 \
  --laplace_w 0.2 \
  --peak_window_w 0.4 \
  --hotspot_thr 0.05 \
  --out_dir checkpoints/gnnhrnet_pwin
