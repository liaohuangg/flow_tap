#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
source_checkpoint="${SOURCE_CHECKPOINT:-$repo_root/LayoutGenModel/logs/model/placement_first_300k/placement-300k-thermal-wirelength-legality/seed_61/latest.ckpt}"
method="${METHOD:-placement-300k-balanced-legality}"
train_steps="${TRAIN_STEPS:-50000}"
batch_size="${BATCH_SIZE:-8}"
val_batch_size="${VAL_BATCH_SIZE:-8}"

layout_root="$repo_root/LayoutGenModel"
train_entry="$layout_root/diffusion/train_graph_thermal.py"
required_paths=(
  "$python_executable"
  "$train_entry"
  "$source_checkpoint"
  "$repo_root/thermalmodel/checkpoints/gnnhrnet_pwin/best.pth"
  "$repo_root/wirelengthmodel/checkpoint/best_wlmodel_total_60k.pt"
  "$repo_root/Dataset/splits/placement_first_300k/manifest.yaml"
)
for required_path in "${required_paths[@]}"; do
  [[ -e "$required_path" ]] || { echo "Required training input is missing: $required_path" >&2; exit 1; }
done

log_dir="$layout_root/logs/model/placement_first_300k/$method/seed_61"
if [[ "$(dirname "$source_checkpoint")" == "$log_dir" ]]; then
  echo "Refusing to fine-tune in the source checkpoint directory: $log_dir" >&2
  exit 2
fi
echo "Source checkpoint: $source_checkpoint"
echo "Balanced fine-tuning steps: $train_steps"
echo "Checkpoints: $log_dir"
echo "Live monitor: $log_dir/training_monitor.png"

cd "$layout_root"
export PYTHONPATH=".:diffusion:.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

exec "$python_executable" "$train_entry" \
  --config-name config_graph_fm_finetune \
  task=placement_first_300k \
  "method=$method" \
  seed=61 \
  "from_checkpoint=$source_checkpoint" \
  "train_steps=$train_steps" \
  "batch_size=$batch_size" \
  "val_batch_size=$val_batch_size" \
  lr=1e-4 \
  thermal.train_weight=0.02 \
  thermal.start_step=0 \
  thermal.warmup_steps=10000 \
  wirelength.train_weight=0.01 \
  wirelength.start_step=0 \
  wirelength.warmup_steps=10000 \
  legality_aux.enabled=True \
  legality_aux.start_step=0 \
  legality_aux.warmup_steps=10000 \
  legality_aux.overlap_direct_weight=0.15 \
  legality_aux.overlap_head_weight=0.05 \
  legality_aux.boundary_direct_weight=0.02 \
  legality_aux.boundary_head_weight=0.01 \
  monitor.enabled=True \
  monitor.every=100 \
  print_every=500
