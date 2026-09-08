#!/usr/bin/env bash
# auto_val.sh — GNN+HRNet 评估脚本 (加载 checkpoint, 在 val / test 上报告指标)
# 指标: hm_rmse(°C) / peak_mae(°C) / peak_bias(°C) / hotspot_rmse(°C)
#
# 用法:
#   ./auto_val.sh                                     # 用默认 checkpoint (checkpoints/gnnhrnet_pwin/best.pth)
#   ./auto_val.sh checkpoints/xxx/best.pth            # 指定 checkpoint
set -euo pipefail

cd /root/placement/flow_tap/thermalmodel
PY=/root/anaconda3/envs/chipdiffusion/bin/python

CKPT="${1:-checkpoints/gnnhrnet_pwin/best.pth}"

echo "===== eval on VAL (ckpt=$CKPT) ====="
$PY -u gnnhrnet.py \
  --eval_ckpt "$CKPT" \
  --eval_split val \
  --num_val 8000 \
  --batch_size 64 \
  --num_workers 28 \
  --seed 0 \
  --hotspot_thr 0.05

echo ""
echo "===== eval on TEST (ckpt=$CKPT) ====="
$PY -u gnnhrnet.py \
  --eval_ckpt "$CKPT" \
  --eval_split test \
  --batch_size 64 \
  --num_workers 28 \
  --seed 0 \
  --hotspot_thr 0.05
