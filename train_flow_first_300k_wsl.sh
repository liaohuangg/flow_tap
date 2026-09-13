#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
train_steps=3000000
batch_size=8
val_batch_size=8
monitor_every=100
print_every=500
method="placement-300k-thermal-wirelength-legality"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-steps) train_steps="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --val-batch-size) val_batch_size="$2"; shift 2 ;;
    --monitor-every) monitor_every="$2"; shift 2 ;;
    --print-every) print_every="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

layout_root="$repo_root/LayoutGenModel"
train_entry="$layout_root/diffusion/train_graph_thermal.py"
required_paths=(
  "$python_executable"
  "$train_entry"
  "$repo_root/thermalmodel/checkpoints/gnnhrnet_pwin/best.pth"
  "$repo_root/wirelengthmodel/checkpoint/best_wlmodel_total_60k.pt"
  "$repo_root/Dataset/splits/placement_first_300k/manifest.yaml"
)
for required_path in "${required_paths[@]}"; do
  [[ -e "$required_path" ]] || { echo "Required training input is missing: $required_path" >&2; exit 1; }
done

monitor_path="$layout_root/logs/model/placement_first_300k/$method/seed_61/training_monitor.png"
echo "WSL Python: $python_executable"
echo "Live monitor: $monitor_path"
echo "The PNG is replaced every $monitor_every training steps."

cd "$layout_root"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
exec "$python_executable" "$train_entry" \
  --config-name config_graph_fm \
  "method=$method" \
  "train_steps=$train_steps" \
  "batch_size=$batch_size" \
  "val_batch_size=$val_batch_size" \
  "monitor.every=$monitor_every" \
  "print_every=$print_every"
