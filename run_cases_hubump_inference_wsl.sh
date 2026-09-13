#!/usr/bin/env bash
set -euo pipefail

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
checkpoint="${CHECKPOINT:-$repo_root/LayoutGenModel/logs/model/placement_first_300k/placement-300k-balanced-legality/seed_61/latest.ckpt}"
chiplet_area_ratio="${CHIPLET_AREA_RATIO:-0.50}"
method="${METHOD:-cases-hubump-13-util50-latest}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chiplet-area-ratio) chiplet_area_ratio="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

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

echo "Checkpoint: $checkpoint"
echo "Chiplet body-area ratio: $chiplet_area_ratio"
echo "Benchmark cases: $num_output_samples"
echo "Output method: $method"

exec "$python_executable" diffusion/eval_thermal_guided.py \
  --config-name config_eval_fm \
  task=cases_hubump \
  "method=$method" \
  seed=61 \
  logger.wandb=False \
  "from_checkpoint=$checkpoint" \
  eval_samples=0 \
  "num_output_samples=$num_output_samples" \
  val_batch_size=8 \
  model.max_diffusion_steps=100 \
  model.legality_guidance_weight=12.0 \
  model.guidance_schedule.legality_initial_weight=12.0 \
  model.guidance_schedule.legality_mid_weight=12.0 \
  model.guidance_schedule.legality_final_weight=10.0 \
  model.backbone_params.auxiliary_legality_heads_enabled=True \
  legalization.mode=standard \
  legalization.grad_descent_steps=1000
