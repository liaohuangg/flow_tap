#!/usr/bin/env bash
# auto_val.sh — GNN+HRNet 评估脚本 (加载 checkpoint, 在 val / test 上报告指标)
# 指标: hm_rmse(°C) / peak_mae(°C) / peak_bias(°C) / hotspot_rmse(°C)
#
# 用法:
#   ./auto_val.sh                                     # 默认 checkpoint: checkpoints/gnnhrnet_pwin/best.pth
#   ./auto_val.sh checkpoints/xxx/best.pth            # 指定 checkpoint
#
# 注意: 模型结构参数 (--base 96 等) 必须与训练时 auto_train.sh 完全一致,
#       否则加载权重时形状不匹配 (gnnhrnet.py 的 --base 默认是 64)。
set -euo pipefail

cd /root/placement/flow_tap/thermalmodel
PY=/root/anaconda3/envs/chipdiffusion/bin/python

CKPT="${1:-checkpoints/gnnhrnet_pwin/best.pth}"

# 与 auto_train.sh 保持一致的模型结构参数 (GNN + HRNet 场头)
MODEL_ARGS=(
  --hidden 128 --heads 4 --num_layers 3 --grid 64
  --base 96 --stages 4 --blocks_per_stage 2 --expand_ratio 2
)

echo "===== eval on VAL (ckpt=$CKPT) ====="
$PY -u gnnhrnet.py \
  --eval_ckpt "$CKPT" \
  --eval_split val \
  --num_val 8000 \
  --batch_size 64 \
  --num_workers 28 \
  --seed 0 \
  --hotspot_thr 0.05 \
  "${MODEL_ARGS[@]}"

echo ""
echo "===== eval on TEST (ckpt=$CKPT) ====="
$PY -u gnnhrnet.py \
  --eval_ckpt "$CKPT" \
  --eval_split test \
  --batch_size 64 \
  --num_workers 28 \
  --seed 0 \
  --hotspot_thr 0.05 \
  "${MODEL_ARGS[@]}"
