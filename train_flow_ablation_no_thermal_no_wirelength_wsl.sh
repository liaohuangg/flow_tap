#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
method="${METHOD:-placement-300k-ablation-no-thermal-no-wirelength}"
train_steps="${TRAIN_STEPS:-3000000}"
batch_size="${BATCH_SIZE:-8}"
val_batch_size="${VAL_BATCH_SIZE:-8}"
monitor_every="${MONITOR_EVERY:-100}"
print_every="${PRINT_EVERY:-500}"

full_model_method="placement-300k-thermal-wirelength-legality"
balanced_model_method="placement-300k-balanced-legality"
if [[ "$method" == "$full_model_method" || "$method" == "$balanced_model_method" ]]; then
  echo "Refusing to use a full-model checkpoint directory for the ablation: $method" >&2
  exit 2
fi

layout_root="$repo_root/LayoutGenModel"
train_entry="$layout_root/diffusion/train_graph_thermal.py"
required_paths=(
  "$python_executable"
  "$train_entry"
  "$repo_root/Dataset/splits/placement_first_300k/manifest.yaml"
)
for required_path in "${required_paths[@]}"; do
  [[ -e "$required_path" ]] || { echo "Required training input is missing: $required_path" >&2; exit 1; }
done

log_dir="$layout_root/logs/model/placement_first_300k/$method/seed_61"
echo "Ablation: thermal proxy loss OFF, wirelength proxy loss OFF"
echo "Training from scratch; no full-model checkpoint will be loaded."
echo "Checkpoints: $log_dir"
echo "Live monitor: $log_dir/training_monitor.png"

cd "$layout_root"
export PYTHONPATH=".:diffusion:.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

exec "$python_executable" "$train_entry" \
  --config-name config_graph_fm \
  "method=$method" \
  seed=61 \
  from_checkpoint=none \
  "train_steps=$train_steps" \
  "batch_size=$batch_size" \
  "val_batch_size=$val_batch_size" \
  lr=3e-4 \
  thermal.train_weight=0.0 \
  wirelength.train_weight=0.0 \
  legality_aux.enabled=True \
  legality_aux.overlap_head_weight=0.02 \
  legality_aux.boundary_head_weight=0.02 \
  legality_aux.overlap_direct_weight=0.05 \
  legality_aux.boundary_direct_weight=0.05 \
  bbox.train_weight=0.0 \
  monitor.enabled=True \
  "monitor.every=$monitor_every" \
  "print_every=$print_every"
