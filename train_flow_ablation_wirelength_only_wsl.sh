#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
method="${METHOD:-placement-300k-wirelength-only}"
train_steps="${TRAIN_STEPS:-100000}"
batch_size="${BATCH_SIZE:-8}"
val_batch_size="${VAL_BATCH_SIZE:-8}"
monitor_every="${MONITOR_EVERY:-100}"
print_every="${PRINT_EVERY:-500}"

layout_root="$repo_root/LayoutGenModel"
train_entry="$layout_root/diffusion/train_graph_thermal.py"
wirelength_ckpt="$repo_root/wirelengthmodel/checkpoint/best_wlmodel_total_60k.pt"
normalizer="$repo_root/wirelengthmodel/checkpoint/normalizer_compat_69_76.json"
manifest="$repo_root/Dataset/splits/placement_first_300k/manifest.yaml"
log_dir="$layout_root/logs/model/placement_first_300k/$method/seed_61"

case "$method" in
  placement-300k-thermal-wirelength-legality|placement-300k-balanced-legality|placement-300k-thermal-only|placement-300k-ablation-no-thermal-no-wirelength)
    echo "Refusing to write the wirelength-only ablation into an existing experiment: $method" >&2
    exit 2
    ;;
esac

required_paths=("$python_executable" "$train_entry" "$wirelength_ckpt" "$normalizer" "$manifest")
for required_path in "${required_paths[@]}"; do
  [[ -e "$required_path" ]] || { echo "Required training input is missing: $required_path" >&2; exit 1; }
done

echo "Ablation: thermal proxy loss OFF, neural wirelength proxy loss ON (weight=0.02)"
echo "Training from scratch for $train_steps steps."
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
  wirelength.train_weight=0.02 \
  wirelength.start_step=0 \
  wirelength.warmup_steps=10000 \
  "wirelength.ckpt=$wirelength_ckpt" \
  "wirelength.normalizer=$normalizer" \
  legality_aux.enabled=True \
  legality_aux.start_step=0 \
  legality_aux.warmup_steps=10000 \
  legality_aux.overlap_head_weight=0.02 \
  legality_aux.boundary_head_weight=0.02 \
  legality_aux.overlap_direct_weight=0.05 \
  legality_aux.boundary_direct_weight=0.05 \
  bbox.train_weight=0.0 \
  monitor.enabled=True \
  "monitor.every=$monitor_every" \
  "print_every=$print_every"
