#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
layout_root="$repo_root/LayoutGenModel"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
seed_start="${SEED_START:-61}"
num_seeds="${NUM_SEEDS:-5}"
chiplet_area_ratio="${CHIPLET_AREA_RATIO:-0.50}"
experiment="${EXPERIMENT:-ablation-100k-5seeds-v1}"

declare -A checkpoints=(
  [full]="$layout_root/logs/model/placement_first_300k/placement-300k-full-100k-fair/seed_61/latest.ckpt"
  [thermal_only]="$layout_root/logs/model/placement_first_300k/placement-300k-thermal-only/seed_61/latest.ckpt"
  [wirelength_only]="$layout_root/logs/model/placement_first_300k/placement-300k-wirelength-only/seed_61/latest.ckpt"
  [no_proxy]="$layout_root/logs/model/placement_first_300k/placement-300k-no-proxy-100k-fair/seed_61/latest.ckpt"
)
models=(no_proxy thermal_only wirelength_only full)

benchmark_dir="$repo_root/benchmark/cases_hubump"
shopt -s nullglob
case_files=("$benchmark_dir"/*.json)
shopt -u nullglob
num_cases="${#case_files[@]}"
(( num_cases > 0 )) || { echo "No benchmark JSON files found in $benchmark_dir" >&2; exit 1; }
for model in "${models[@]}"; do
  [[ -e "${checkpoints[$model]}" ]] || {
    echo "Missing fair checkpoint for $model: ${checkpoints[$model]}" >&2
    echo "Run the two preparation commands printed in the accompanying instructions first." >&2
    exit 1
  }
done

if [[ -z "${CPLEX_STUDIO_DIR:-}" && -d /opt/ibm/ILOG/CPLEX_Studio221 ]]; then
  export CPLEX_STUDIO_DIR=/opt/ibm/ILOG/CPLEX_Studio221
fi
cd "$layout_root"
export PYTHONPATH=".:diffusion:.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export FLOW_TAP_CHIPLET_AREA_RATIO="$chiplet_area_ratio"

report_dir="$layout_root/logs/output/cases_hubump/$experiment"
mkdir -p "$report_dir"

for model in "${models[@]}"; do
  method="${experiment}-${model}"
  method_dir="$layout_root/logs/output/cases_hubump/$method"
  mkdir -p "$method_dir/batch_logs"
  for ((offset=0; offset<num_seeds; offset++)); do
    seed=$((seed_start + offset))
    seed_dir="$method_dir/seed_$seed"
    log="$method_dir/batch_logs/seed_${seed}.log"
    if [[ -s "$seed_dir/metrics.csv" ]] && [[ $(wc -l < "$seed_dir/metrics.csv") -eq $((num_cases + 1)) ]]; then
      echo "$model seed=$seed already complete; skipping"
      continue
    fi
    echo "Running model=$model seed=$seed ($num_cases cases)"
    "$python_executable" diffusion/eval_thermal_guided.py \
      --config-name config_eval_fm \
      task=cases_hubump \
      "method=$method" \
      seed="$seed" \
      logger.wandb=False \
      "from_checkpoint=${checkpoints[$model]}" \
      eval_samples=0 \
      "num_output_samples=$num_cases" \
      val_batch_size=8 \
      model.max_diffusion_steps=100 \
      model.legality_guidance_weight=12.0 \
      model.guidance_schedule.legality_initial_weight=12.0 \
      model.guidance_schedule.legality_mid_weight=12.0 \
      model.guidance_schedule.legality_final_weight=10.0 \
      model.backbone_params.auxiliary_legality_heads_enabled=True \
      wirelength.enabled=False \
      wirelength.guidance_weight=0.0 \
      wirelength.initial_weight=0.0 \
      model.hpwl_guidance_weight=0.0 \
      model.guidance_schedule.hpwl_initial_weight=0.0 \
      model.guidance_schedule.hpwl_final_weight=0.0 \
      thermal.guidance_weight=0.0 \
      thermal.guidance_steps=0 \
      thermal.report_guidance_enabled=False \
      thermal.schedule.enabled=False \
      thermal.schedule.initial_weight=0.0 \
      thermal.schedule.final_weight=0.0 \
      thermal.legality_weight=0.0 \
      model.heat_repulsion_guidance_weight=0.0 \
      model.guidance_schedule.heat_repulsion_initial_weight=0.0 \
      model.guidance_schedule.heat_repulsion_final_weight=0.0 \
      legalization.mode=standard \
      legalization.grad_descent_steps=1000 \
      2>&1 | tee "$log"
  done
  "$python_executable" -m evaluation.aggregate_hubump_multiseed \
    "$method_dir" --output "$method_dir/all_candidates.csv"
done

summary_args=()
for model in "${models[@]}"; do
  summary_args+=(--input "$model=$layout_root/logs/output/cases_hubump/${experiment}-${model}/all_candidates.csv")
done
"$python_executable" -m evaluation.summarize_training_ablation \
  "${summary_args[@]}" --baseline no_proxy --output-dir "$report_dir"
echo "Done: $report_dir/ablation_report.md"
