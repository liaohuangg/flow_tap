#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
checkpoint="${CHECKPOINT:-$repo_root/LayoutGenModel/logs/model/placement_first_300k/placement-300k-balanced-legality/seed_61/latest.ckpt}"
num_cases="${NUM_CASES:-2}"
output_indices="${OUTPUT_INDICES:-}"
seed="${SEED:-61}"

[[ -e "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
if [[ -z "${CPLEX_STUDIO_DIR:-}" && -d /opt/ibm/ILOG/CPLEX_Studio221 ]]; then
  export CPLEX_STUDIO_DIR=/opt/ibm/ILOG/CPLEX_Studio221
fi

cd "$repo_root/LayoutGenModel"
export PYTHONPATH=".:diffusion:.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export FLOW_TAP_CHIPLET_AREA_RATIO="${CHIPLET_AREA_RATIO:-0.50}"

method="${METHOD:-cases-hubump-tune${num_cases}-expanded-legal-balanced}"
sample_args=("num_output_samples=$num_cases")
if [[ -n "$output_indices" ]]; then
  sample_args+=("+output_indices=$output_indices")
fi

exec "$python_executable" diffusion/eval_thermal_guided.py \
  --config-name config_eval_fm \
  task=cases_hubump \
  "method=$method" \
  seed="$seed" \
  +per_case_seed=True \
  logger.wandb=False \
  "from_checkpoint=$checkpoint" \
  eval_samples=0 \
  "${sample_args[@]}" \
  val_batch_size=8 \
  model.max_diffusion_steps=100 \
  model.legality_guidance_weight=12.0 \
  model.guidance_schedule.legality_initial_weight=12.0 \
  model.guidance_schedule.legality_mid_weight=12.0 \
  model.guidance_schedule.legality_final_weight=10.0 \
  model.backbone_params.auxiliary_legality_heads_enabled=True \
  wirelength.enabled=True \
  wirelength.guidance_weight=0.02 \
  wirelength.initial_weight=0.0 \
  wirelength.start=0.5 \
  wirelength.full=0.8 \
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
  legalization.grad_descent_steps="${LEGALIZATION_STEPS:-1000}" \
  legalization.step_size="${LEGALIZATION_STEP_SIZE:-0.20}" \
  legalization.softmax_max="${LEGALIZATION_SOFTMAX_MAX:-50.0}" \
  legalization.legality_weight="${LEGALITY_WEIGHT:-1.0}" \
  legalization.legality_increase_factor="${LEGALITY_INCREASE_FACTOR:-1.0}" \
  +legalization.legality_clearance="${LEGALITY_CLEARANCE:-0.0002}" \
  +legalization.enforce_legality="${ENFORCE_LEGALITY:-True}" \
  +legalization.legality_extra_steps="${LEGALITY_EXTRA_STEPS:-5000}" \
  +legalization.legality_check_every="${LEGALITY_CHECK_EVERY:-100}" \
  +legalization.legality_extra_step_size="${LEGALITY_EXTRA_STEP_SIZE:-0.10}" \
  +legalization.thermal_weight="${THERMAL_WEIGHT:-0.05}" \
  +legalization.thermal_start_factor="${THERMAL_START_FACTOR:-0.30}" \
  +legalization.thermal_end_factor="${THERMAL_END_FACTOR:-0.60}" \
  +legalization.thermal_zero_factor="${THERMAL_ZERO_FACTOR:-0.80}" \
  +legalization.thermal_increase_factor=1.0
