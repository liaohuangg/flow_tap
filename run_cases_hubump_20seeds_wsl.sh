#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
layout_root="$repo_root/LayoutGenModel"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
checkpoint="${CHECKPOINT:-$layout_root/logs/model/placement_first_300k/placement-300k-balanced-legality/seed_61/latest.ckpt}"
chiplet_area_ratio="${CHIPLET_AREA_RATIO:-0.50}"
seed_start="${SEED_START:-61}"
num_seeds="${NUM_SEEDS:-20}"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
method="${METHOD:-cases-hubump-util50-candidates-${num_seeds}seeds-${run_tag}}"

# Candidate-generation guidance. All values may be overridden through env vars.
wirelength_weight="${WIRELENGTH_WEIGHT:-0.02}"
thermal_weight="${THERMAL_WEIGHT:-0.002}"
thermal_lr="${THERMAL_LR:-0.001}"
pair_feature_enabled="${PAIR_FEATURE_ENABLED:-False}"

[[ -e "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
[[ "$seed_start" =~ ^[0-9]+$ ]] || { echo "SEED_START must be an integer" >&2; exit 2; }
[[ "$num_seeds" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_SEEDS must be a positive integer" >&2; exit 2; }

benchmark_dir="$repo_root/benchmark/cases_hubump"
shopt -s nullglob
case_files=("$benchmark_dir"/*.json)
shopt -u nullglob
num_cases="${#case_files[@]}"
(( num_cases > 0 )) || { echo "No benchmark JSON files found in: $benchmark_dir" >&2; exit 1; }

method_dir="$layout_root/logs/output/cases_hubump/$method"
batch_log_dir="$method_dir/batch_logs"
mkdir -p "$batch_log_dir"
master_log="$batch_log_dir/master.log"
summary_csv="$method_dir/all_candidates.csv"

if [[ -z "${CPLEX_STUDIO_DIR:-}" && -d /opt/ibm/ILOG/CPLEX_Studio221 ]]; then
  export CPLEX_STUDIO_DIR=/opt/ibm/ILOG/CPLEX_Studio221
fi

cd "$layout_root"
export PYTHONPATH=".:diffusion:.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export FLOW_TAP_CHIPLET_AREA_RATIO="$chiplet_area_ratio"

{
  echo "Method: $method"
  echo "Checkpoint: $checkpoint"
  echo "Cases: $num_cases"
  echo "Seeds: $seed_start..$((seed_start + num_seeds - 1))"
  echo "Chiplet body-area ratio: $chiplet_area_ratio"
  echo "Wirelength guidance weight: $wirelength_weight"
  echo "Neural pair-distance feature: $pair_feature_enabled"
  echo "Thermal guidance: weak (weight=$thermal_weight, lr=$thermal_lr, one late step)"
  echo "Legality filtering: disabled (every generated candidate is retained)"
  echo "Output: $method_dir"
} | tee -a "$master_log"

for ((offset=0; offset<num_seeds; offset++)); do
  seed=$((seed_start + offset))
  seed_dir="$method_dir/seed_$seed"
  seed_log="$batch_log_dir/seed_${seed}.log"

  if [[ -s "$seed_dir/metrics.csv" ]] && [[ $(wc -l < "$seed_dir/metrics.csv") -eq $((num_cases + 1)) ]]; then
    echo "[$((offset + 1))/$num_seeds] seed=$seed already complete; skipping" | tee -a "$master_log"
  else
    echo "[$((offset + 1))/$num_seeds] starting seed=$seed" | tee -a "$master_log"
    "$python_executable" diffusion/eval_thermal_guided.py \
      --config-name config_eval_fm \
      task=cases_hubump \
      "method=$method" \
      seed="$seed" \
      logger.wandb=False \
      "from_checkpoint=$checkpoint" \
      eval_samples=0 \
      "num_output_samples=$num_cases" \
      val_batch_size=8 \
      model.max_diffusion_steps=100 \
      model.legality_guidance_weight=12.0 \
      model.guidance_schedule.legality_initial_weight=12.0 \
      model.guidance_schedule.legality_mid_weight=12.0 \
      model.guidance_schedule.legality_final_weight=10.0 \
      model.backbone_params.auxiliary_legality_heads_enabled=True \
      wirelength.enabled=True \
      wirelength.pair_feature_enabled="$pair_feature_enabled" \
      wirelength.guidance_weight="$wirelength_weight" \
      wirelength.initial_weight=0.0 \
      wirelength.start=0.5 \
      wirelength.full=0.8 \
      model.hpwl_guidance_weight=0.0 \
      model.guidance_schedule.hpwl_initial_weight=0.0 \
      model.guidance_schedule.hpwl_final_weight=0.0 \
      thermal.guidance_weight="$thermal_weight" \
      thermal.guidance_lr="$thermal_lr" \
      thermal.guidance_steps=1 \
      thermal.grad_clip=0.02 \
      thermal.report_guidance_enabled=True \
      thermal.schedule.enabled=True \
      thermal.schedule.start=0.75 \
      thermal.schedule.full=0.90 \
      thermal.schedule.initial_weight=0.0 \
      thermal.schedule.final_weight="$thermal_weight" \
      thermal.legality_weight=0.0 \
      model.heat_repulsion_guidance_weight=0.0 \
      model.guidance_schedule.heat_repulsion_initial_weight=0.0 \
      model.guidance_schedule.heat_repulsion_final_weight=0.0 \
      legalization.mode=standard \
      legalization.grad_descent_steps=1000 \
      2>&1 | tee "$seed_log"
    echo "[$((offset + 1))/$num_seeds] finished seed=$seed" | tee -a "$master_log"
  fi

  "$python_executable" -m evaluation.aggregate_hubump_multiseed \
    "$method_dir" --output "$summary_csv" | tee -a "$master_log"
done

echo "Completed $num_seeds seeds x $num_cases cases = $((num_seeds * num_cases)) candidates." | tee -a "$master_log"
echo "Combined metrics: $summary_csv" | tee -a "$master_log"
echo "Per-seed logs: $batch_log_dir" | tee -a "$master_log"
