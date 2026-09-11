#!/usr/bin/env bash
# auto_val.sh — GNN+HRNet 测试集评估 (加载 checkpoint, 全量指标写入 logs/test.log)
# 指标: hm_rmse / mean_rmse / mean_mae / mape / peak_ae / peak_bias / peak_loc / hotspot_rmse / 最坏·最好 case 等
#
# 用法:
#   ./auto_val.sh                                     # 默认 checkpoint: checkpoints/gnnhrnet_pwin/best.pth
#   ./auto_val.sh checkpoints/xxx/best.pth            # 指定 checkpoint
#
# 注意: 模型结构参数 (--base 96 等) 必须与训练时 auto_train.sh 完全一致。
set -euo pipefail

cd /root/placement/flow_tap/thermalmodel
PY=/root/anaconda3/envs/chipdiffusion/bin/python

CKPT="${1:-checkpoints/gnnhrnet_pwin/best.pth}"

# 与 auto_train.sh 保持一致的模型结构参数 (GNN + HRNet 场头)
MODEL_ARGS=(
  --hidden 128 --heads 4 --num_layers 3 --grid 64
  --base 96 --stages 4 --blocks_per_stage 2 --expand_ratio 2
)

echo "===== eval on TEST (ckpt=$CKPT, batch_size=8) ====="
$PY -u eval_hrnet_ckpt.py \
  --ckpt "$CKPT" \
  --split test \
  --eval_bs 8 \
  --num_workers 8 \
  --seed 0 \
  --hotspot_thr 0.05 \
  --topk 20 \
  --out_log logs/test.log \
  --out_fig_dir figs/test \
  "${MODEL_ARGS[@]}"
