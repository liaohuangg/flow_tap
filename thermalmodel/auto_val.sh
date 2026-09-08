#!/usr/bin/env bash
set -euo pipefail

cd /root/placement/flow_tap/thermalmodel

# Usage:
#   bash auto_val.sh [CKPT_DIR]
#
# 1) Evaluate every checkpoint on VAL and log all metrics
# 2) Pick the checkpoint with the best (lowest) VAL mean_rmse
# 3) Evaluate that checkpoint on TEST, logging all metrics
# 4) Save top-50 best and top-50 worst figures (by per-sample RMSE)

PY=/root/anaconda3/envs/chipdiffusion/bin/python

CKPT_DIR_DEFAULT="/root/placement/flow_tap/thermalmodel/checkpoints/auto"
CKPT_DIR="${1:-$CKPT_DIR_DEFAULT}"

OUT_LOG_VAL="/root/placement/flow_tap/thermalmodel/logs/val_eval.log"
OUT_LOG_TEST="/root/placement/flow_tap/thermalmodel/logs/test_eval_best_from_val.log"

SEED=0
EVAL_BS=32
DEVICE="cuda"

TEST_FIG_DIR="/root/placement/flow_tap/thermalmodel/test_result/test_best_from_val_top50"

mkdir -p "$(dirname "$OUT_LOG_VAL")" "$TEST_FIG_DIR"

best_ckpt=""
best_rmse=""

shopt -s nullglob
ckpts=("$CKPT_DIR"/*.pth)
if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "No checkpoints found under: $CKPT_DIR"
  exit 1
fi

for ckpt in "${ckpts[@]}"; do
  echo "[val] ckpt=$ckpt"

  out="$(
    $PY -u eval_hrnet_ckpt.py \
      --ckpt "$ckpt" \
      --split val \
      --seed "$SEED" \
      --eval_bs "$EVAL_BS" \
      --device "$DEVICE" \
      --out_log "$OUT_LOG_VAL" \
      --append \
      --limit_cases 0 \
      --topk 0 \
      2>&1
  )"

  rmse="$({ printf '%s\n' "$out" | awk '/^metrics /{for(i=1;i<=NF;i++) if($i ~ /^mean_rmse=/){sub(/^mean_rmse=/, "", $i); print $i; exit}}'; } || true)"

  if [[ -z "$rmse" ]]; then
    echo "[val] WARN: could not parse mean_rmse for ckpt=$ckpt"
    continue
  fi

  echo "[val] mean_rmse=$rmse ckpt=$ckpt"

  if [[ -z "$best_rmse" ]]; then
    best_rmse="$rmse"
    best_ckpt="$ckpt"
  else
    is_better="$({ awk -v a="$rmse" -v b="$best_rmse" 'BEGIN{print (a<b)?1:0}'; } || true)"
    if [[ "$is_better" == "1" ]]; then
      best_rmse="$rmse"
      best_ckpt="$ckpt"
    fi
  fi

done

if [[ -z "$best_ckpt" ]]; then
  echo "No best checkpoint selected (parsing failed?)."
  exit 2
fi

echo "[best] val_mean_rmse=$best_rmse ckpt=$best_ckpt"

echo "[test] ckpt=$best_ckpt"
$PY -u eval_hrnet_ckpt.py \
  --ckpt "$best_ckpt" \
  --split test \
  --seed "$SEED" \
  --eval_bs "$EVAL_BS" \
  --device "$DEVICE" \
  --out_fig_dir "$TEST_FIG_DIR" \
  --topk 50 \
  --limit_cases 0 \
  --out_log "$OUT_LOG_TEST" \
  --append
