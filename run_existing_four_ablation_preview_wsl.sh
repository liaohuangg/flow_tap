#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
layout_root="$repo_root/LayoutGenModel"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
runner="$repo_root/run_cases_hubump_full_no_proxy_guidance_wsl.sh"
output_root="$layout_root/logs/output/cases_hubump"
report_dir="$output_root/cases-hubump-util50-ablation-existing-preview"
mkdir -p "$report_dir"

thermal_method="cases-hubump-util50-trained-thermal-only-preview"
wire_method="cases-hubump-util50-trained-wirelength-only-preview"
thermal_ckpt="$layout_root/logs/model/placement_first_300k/placement-300k-thermal-only/seed_61/latest.ckpt"
wire_ckpt="$layout_root/logs/model/placement_first_300k/placement-300k-wirelength-only/seed_61/latest.ckpt"

for path in "$python_executable" "$runner" "$thermal_ckpt" "$wire_ckpt"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

if [[ "${WAIT_FOR_TRAINING:-1}" == "1" ]]; then
  while pgrep -f '[t]rain_graph_thermal.py' >/dev/null; do
    echo "$(date '+%F %T') waiting for active Flow Matching training to finish"
    sleep 30
  done
fi

echo "Running thermal-only trained model, seed 61, inference proxy guidance OFF"
CHECKPOINT="$thermal_ckpt" METHOD="$thermal_method" bash "$runner"

echo "Running wirelength-only trained model, seed 61, inference proxy guidance OFF"
CHECKPOINT="$wire_ckpt" METHOD="$wire_method" bash "$runner"

cd "$layout_root"
export PYTHONPATH=".:diffusion:.."

declare -A methods=(
  [no_proxy]="cases-hubump-util50-ablation-no-thermal-no-wirelength"
  [thermal_only]="$thermal_method"
  [wirelength_only]="$wire_method"
  [full]="cases-hubump-util50-full-no-proxy-guidance"
)
models=(no_proxy thermal_only wirelength_only full)
summary_args=()
for model in "${models[@]}"; do
  method_dir="$output_root/${methods[$model]}"
  "$python_executable" -m evaluation.aggregate_hubump_multiseed \
    "$method_dir" --output "$method_dir/all_candidates.csv"
  summary_args+=(--input "$model=$method_dir/all_candidates.csv")
done

"$python_executable" -m evaluation.summarize_training_ablation \
  "${summary_args[@]}" \
  --baseline no_proxy \
  --output-dir "$report_dir"

echo "Preview comparison completed: $report_dir/ablation_report.md"
