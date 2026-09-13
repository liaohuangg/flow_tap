#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
checkpoint="${CHECKPOINT:-$repo_root/LayoutGenModel/logs/model/placement_first_300k/placement-300k-balanced-legality/seed_61/latest.ckpt}"
method="${METHOD:-cases-hubump-tune3-joint-thermal-wirelength}"

cd "$repo_root/LayoutGenModel"
export PYTHONPATH=".:diffusion:.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export FLOW_TAP_CHIPLET_AREA_RATIO="${CHIPLET_AREA_RATIO:-0.50}"
if [[ -z "${CPLEX_STUDIO_DIR:-}" && -d /opt/ibm/ILOG/CPLEX_Studio221 ]]; then
  export CPLEX_STUDIO_DIR=/opt/ibm/ILOG/CPLEX_Studio221
fi

exec "$python_executable" diffusion/eval_thermal_guided.py \
  --config-name config_eval_fm \
  task=cases_hubump \
  "method=$method" \
  seed=61 \
  +per_case_seed=True \
  logger.wandb=False \
  "from_checkpoint=$checkpoint" \
  eval_samples=0 \
  num_output_samples=3 \
  val_batch_size=8 \
  model.max_diffusion_steps=100 \
  model.legality_guidance_weight=12.0 \
  model.guidance_schedule.legality_initial_weight=12.0 \
  model.guidance_schedule.legality_mid_weight=12.0 \
  model.guidance_schedule.legality_final_weight=10.0 \
  model.backbone_params.auxiliary_legality_heads_enabled=True \
  wirelength.enabled=True \
  wirelength.guidance_weight="${WIRELENGTH_WEIGHT:-0.02}" \
  wirelength.initial_weight=0.0 \
  wirelength.start=0.5 \
  wirelength.full=0.8 \
  model.hpwl_guidance_weight=0.0 \
  model.guidance_schedule.hpwl_initial_weight=0.0 \
  model.guidance_schedule.hpwl_final_weight=0.0 \
  thermal.report_guidance_enabled=True \
  thermal.guidance_weight="${THERMAL_GUIDANCE_WEIGHT:-0.04}" \
  thermal.guidance_lr="${THERMAL_GUIDANCE_LR:-0.0125}" \
  thermal.guidance_steps="${THERMAL_GUIDANCE_STEPS:-1}" \
  +thermal.smooth_max_beta="${THERMAL_SMOOTH_MAX_BETA:-20.0}" \
  +thermal.mean_weight="${THERMAL_MEAN_WEIGHT:-0.05}" \
  thermal.grad_clip=0.0 \
  +thermal.gradient_normalize=True \
  thermal.legality_weight=0.0 \
  +thermal.hpwl_weight=0.0 \
  thermal.schedule.enabled=True \
  thermal.schedule.start="${THERMAL_SCHEDULE_START:-0.35}" \
  thermal.schedule.full="${THERMAL_SCHEDULE_FULL:-0.70}" \
  thermal.schedule.initial_weight=0.0 \
  thermal.schedule.final_weight="${THERMAL_GUIDANCE_WEIGHT:-0.04}" \
  model.heat_repulsion_guidance_weight=0.0 \
  model.guidance_schedule.heat_repulsion_initial_weight=0.0 \
  model.guidance_schedule.heat_repulsion_final_weight=0.0 \
  legalization.mode=standard \
  legalization.grad_descent_steps=1000 \
  legalization.step_size=0.2 \
  legalization.softmax_max=50.0 \
  legalization.legality_weight=1.0 \
  legalization.legality_increase_factor=1.0 \
  +legalization.legality_clearance=0.0002 \
  +legalization.enforce_legality=False \
  +legalization.legality_extra_steps=0 \
  +legalization.thermal_weight="${LEGALIZATION_THERMAL_WEIGHT:-0.05}" \
  +legalization.thermal_smooth_max_beta="${THERMAL_SMOOTH_MAX_BETA:-20.0}" \
  +legalization.thermal_mean_weight="${THERMAL_MEAN_WEIGHT:-0.05}" \
  +legalization.thermal_start_factor=0.3 \
  +legalization.thermal_end_factor=0.6 \
  +legalization.thermal_zero_factor=0.8 \
  +legalization.thermal_increase_factor=1.0
