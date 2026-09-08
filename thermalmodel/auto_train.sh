#!/bin/bash
# auto_train.sh — HRNet 统一训练脚本 (128x128 power/mask -> 64x64 temp)
# 参数对应参考配置:
#   fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0
# 用法:
#   ./auto_train.sh            # 完整训练 200 epochs
#   ./auto_train.sh 5          # 快速试跑 5 epochs
set -u
cd "$(dirname "$0")"
PY=/root/anaconda3/envs/chipdiffusion/bin/python

EPOCHS="${1:-200}"

$PY -u HRNet.py train \
  --epochs "$EPOCHS" \
  --batch_size 32 \
  --lr 2e-4 \
  --base 96 \
  --stages 4 \
  --blocks_per_stage 2 \
  --expand_ratio 2 \
  --grad_w 0.1 \
  --avg_w 0.1 \
  --mean_consistency_w 0.1 \
  --under_w 1.0 \
  --hotspot_mode linear \
  --topk_w 0.0 --topk_k 0 \
  --peak_w 0.0 \
  --seed 0 \
  --limit_train 0 --limit_val 0 \
  --ckpt_every 5 \
  --print_every 100 \
  --out_dir checkpoints/auto
