#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
method="${METHOD:-placement-pareto-clean-neural-pairdist-200k}"
train_steps="${TRAIN_STEPS:-200000}"
batch_size="${BATCH_SIZE:-8}"

layout_root="$repo_root/LayoutGenModel"
output_dir="$layout_root/logs/model/placement_pareto_clean/$method/seed_61"

required_paths=(
  "$python_executable"
  "$layout_root/diffusion/train_graph_thermal.py"
  "$repo_root/thermalmodel/checkpoints/gnnhrnet_pwin/best.pth"
  "$repo_root/wirelengthmodel/checkpoint/best_wlmodel_total_60k.pt"
  "$repo_root/wirelengthmodel/checkpoint/normalizer_compat_69_76.json"
  "$repo_root/Dataset/splits/placement_pareto_clean/manifest.yaml"
  "$repo_root/Dataset/dataset/placement_dataset/placement_dataset_opt_pareto_clean"
)
for required_path in "${required_paths[@]}"; do
  [[ -e "$required_path" ]] || {
    echo "Required input is missing: $required_path" >&2
    exit 1
  }
done

if [[ -e "$output_dir/latest.ckpt" ]]; then
  echo "Refusing to overwrite existing checkpoint: $output_dir/latest.ckpt" >&2
  echo "Set METHOD to a new name if you intend to start another run." >&2
  exit 2
fi

echo "Dataset: placement_pareto_clean"
echo "Steps: $train_steps"
echo "Neural pair-distance feature: enabled"
echo "Checkpoints: $output_dir"
echo "Live monitor: $output_dir/training_monitor.png"

cd "$layout_root"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

exec "$python_executable" diffusion/train_graph_thermal.py \
  --config-name config_graph_fm \
  task=placement_pareto_clean \
  "method=$method" \
  seed=61 \
  from_checkpoint=none \
  "train_steps=$train_steps" \
  "batch_size=$batch_size" \
  "val_batch_size=$batch_size" \
  eval_every=0 \
  print_every=500 \
  wirelength.pair_feature_enabled=True \
  monitor.enabled=True \
  monitor.every=100
