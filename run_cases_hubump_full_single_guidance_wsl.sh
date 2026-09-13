#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
checkpoint="${CHECKPOINT:-$repo_root/LayoutGenModel/logs/model/placement_first_300k/placement-300k-balanced-legality/seed_61/latest.ckpt}"
chiplet_area_ratio="${CHIPLET_AREA_RATIO:-0.50}"
guidance=""
method=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --guidance) guidance="$2"; shift 2 ;;
    --chiplet-area-ratio) chiplet_area_ratio="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$guidance" in
  wirelength)
    [[ -n "$method" ]] || method="cases-hubump-util50-full-wirelength-only"
    proxy_args=(
      wirelength.enabled=True
      wirelength.guidance_weight=0.02
      wirelength.initial_weight=0.0
      wirelength.start=0.5
      wirelength.full=0.8
      model.hpwl_guidance_weight=0.0
      model.guidance_schedule.hpwl_initial_weight=0.0
      model.guidance_schedule.hpwl_final_weight=0.0
      thermal.guidance_weight=0.0
      thermal.guidance_steps=0
      thermal.report_guidance_enabled=False
      thermal.schedule.enabled=False
      thermal.schedule.initial_weight=0.0
      thermal.schedule.final_weight=0.0
      thermal.legality_weight=0.0
      model.heat_repulsion_guidance_weight=0.0
      model.guidance_schedule.heat_repulsion_initial_weight=0.0
      model.guidance_schedule.heat_repulsion_final_weight=0.0
    )
    ;;
  thermal)
    [[ -n "$method" ]] || method="cases-hubump-util50-full-thermal-only"
    proxy_args=(
      wirelength.enabled=False
      wirelength.guidance_weight=0.0
      wirelength.initial_weight=0.0
      model.hpwl_guidance_weight=0.0
      model.guidance_schedule.hpwl_initial_weight=0.0
      model.guidance_schedule.hpwl_final_weight=0.0
      thermal.guidance_weight=0.01
      thermal.guidance_lr=0.002
      thermal.guidance_steps=1
      thermal.report_guidance_enabled=True
      thermal.schedule.enabled=True
      thermal.schedule.start=0.7
      thermal.schedule.full=0.9
      thermal.schedule.initial_weight=0.0
      thermal.schedule.final_weight=0.01
      thermal.legality_weight=0.0
      model.heat_repulsion_guidance_weight=0.0
      model.guidance_schedule.heat_repulsion_initial_weight=0.0
      model.guidance_schedule.heat_repulsion_final_weight=0.0
    )
    ;;
  thermal-strong)
    [[ -n "$method" ]] || method="cases-hubump-util50-full-thermal-strong-only"
    proxy_args=(
      wirelength.enabled=False
      wirelength.guidance_weight=0.0
      wirelength.initial_weight=0.0
      model.hpwl_guidance_weight=0.0
      model.guidance_schedule.hpwl_initial_weight=0.0
      model.guidance_schedule.hpwl_final_weight=0.0
      thermal.guidance_weight=0.05
      thermal.guidance_lr=0.01
      thermal.guidance_steps=2
      thermal.grad_clip=0.05
      thermal.report_guidance_enabled=True
      thermal.schedule.enabled=True
      thermal.schedule.start=0.5
      thermal.schedule.full=0.8
      thermal.schedule.initial_weight=0.0
      thermal.schedule.final_weight=0.05
      thermal.legality_weight=0.0
      model.heat_repulsion_guidance_weight=0.0
      model.guidance_schedule.heat_repulsion_initial_weight=0.0
      model.guidance_schedule.heat_repulsion_final_weight=0.0
    )
    ;;
  *)
    echo "Usage: $0 --guidance wirelength|thermal|thermal-strong [--checkpoint PATH] [--chiplet-area-ratio RATIO] [--method NAME]" >&2
    exit 2
    ;;
esac

[[ -e "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
benchmark_dir="$repo_root/benchmark/cases_hubump"
shopt -s nullglob
case_files=("$benchmark_dir"/*.json)
shopt -u nullglob
num_output_samples="${#case_files[@]}"
(( num_output_samples > 0 )) || { echo "No benchmark JSON files found in: $benchmark_dir" >&2; exit 1; }

if [[ -z "${CPLEX_STUDIO_DIR:-}" && -d /opt/ibm/ILOG/CPLEX_Studio221 ]]; then
  export CPLEX_STUDIO_DIR=/opt/ibm/ILOG/CPLEX_Studio221
fi

cd "$repo_root/LayoutGenModel"
export PYTHONPATH=".:diffusion:.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export FLOW_TAP_CHIPLET_AREA_RATIO="$chiplet_area_ratio"

echo "Full checkpoint: $checkpoint"
echo "Enabled proxy guidance: $guidance only"
echo "Chiplet body-area ratio: $chiplet_area_ratio"
echo "Benchmark cases: $num_output_samples"
echo "Output method: $method"

common_args=(
  --config-name config_eval_fm
  task=cases_hubump
  "method=$method"
  seed=61
  logger.wandb=False
  "from_checkpoint=$checkpoint"
  eval_samples=0
  "num_output_samples=$num_output_samples"
  val_batch_size=8
  model.max_diffusion_steps=100
  model.legality_guidance_weight=12.0
  model.guidance_schedule.legality_initial_weight=12.0
  model.guidance_schedule.legality_mid_weight=12.0
  model.guidance_schedule.legality_final_weight=10.0
  model.backbone_params.auxiliary_legality_heads_enabled=True
  legalization.mode=standard
  legalization.grad_descent_steps=1000
)

exec "$python_executable" diffusion/eval_thermal_guided.py "${common_args[@]}" "${proxy_args[@]}"
