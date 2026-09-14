#!/usr/bin/env bash
set -euo pipefail

# Sample the 20k-opt flow-matching model on the cases_hubump benchmark, 5 seeds.
#
# Same sampling pipeline as run_cases_hubump_20seeds_wsl.sh; the differences are
# all about pointing it at the model trained on placement_dataset_opt:
#   * checkpoint defaults to the placement_opt_20k run's best.ckpt rather than a
#     300k latest.ckpt.  best.ckpt is the right one to sample from here: val/loss
#     bottomed out around step 4500 (0.7187) and then climbed to 0.87 by step
#     16000 as the thermal/wirelength aux weights finished their 10k warmup, so
#     latest.ckpt is the worse model.
#   * repo_root defaults to the directory holding this script.
#   * PYTHONPATH keeps _env_compat on the path.  The 20seeds script sets
#     PYTHONPATH outright, which drops the NumPy<2 shim and makes `import wandb`
#     fail inside utils.py.
#
# Usage:
#   ./run_cases_hubump_5seeds_opt_wsl.sh
#   NUM_SEEDS=5 CHECKPOINT=/path/to/best.ckpt ./run_cases_hubump_5seeds_opt_wsl.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${FLOW_TAP_ROOT:-$script_dir}"

if [[ -n "${FLOW_TAP_PYTHON:-}" ]]; then
  python_executable="$FLOW_TAP_PYTHON"
elif [[ -x /root/anaconda3/envs/chipdiffusion/bin/python ]]; then
  python_executable=/root/anaconda3/envs/chipdiffusion/bin/python
elif [[ -x /home/user/miniconda3/envs/chipdiffusion/bin/python ]]; then
  python_executable=/home/user/miniconda3/envs/chipdiffusion/bin/python
else
  echo "No chipdiffusion interpreter found; set FLOW_TAP_PYTHON." >&2
  exit 1
fi

layout_root="$repo_root/LayoutGenModel"
checkpoint="${CHECKPOINT:-$layout_root/checkpoints/model_opt/placement_opt_20k/seed_61/best.ckpt}"
chiplet_area_ratio="${CHIPLET_AREA_RATIO:-0.50}"
seed_start="${SEED_START:-61}"
num_seeds="${NUM_SEEDS:-5}"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
method="${METHOD:-cases-hubump-opt20k-${num_seeds}seeds-${run_tag}}"

wirelength_weight="${WIRELENGTH_WEIGHT:-0.02}"

# Thermal guidance.  These are NOT the config defaults (0.01 / 0.002 / clip 0.01),
# and they are not the old values of this script either (0.002 / 0.001).  The old
# values made thermal guidance inert: measured on seed_5's 12 output layouts with
# gen_dataset/probe_guidance.py, one guided step displaced a chiplet by 2.9e-8 in
# normalized coordinates, so the whole run moved things ~4e-4 mm -- while the
# dataset coordinate grid is 1e-3 mm, i.e. it could not push a chiplet by even
# 1/50th of one grid unit.  Wirelength guidance moved a median 2.7 mm over the
# same run, so the two differ by ~150,000x.
#
# Two independent causes, both fixed here:
#   * nominal scale: old lr 0.001 x weight 0.002 x 1 step = 2e-6, versus
#     wirelength's 0.1 x 0.02 x 8 = 1.6e-2 per guided step.  1000x apart.
#   * grad_clip=0.02 never bound: the weighted thermal gradient peaks at ~3e-5,
#     so the clip sat unused and the weight acted linearly.  Raising only the
#     weight would therefore have bought ~0.2 um, not a real change.
#
# Now the clip DOES bind (weight 3.0 x the weakest per-case gradient p90 of
# 0.00917 = 0.027 > 0.02), which is the intent: the step becomes lr x clip =
# 3e-3 normalized, and the gradient's raw scale no longer decides how far the
# layout moves, only which way.  Over the 50 force-application steps that the
# widened schedule gives, total displacement is ~0.15 normalized, i.e. 2.7-4.8 mm
# depending on canvas -- deliberately the same order as wirelength's 2.7 mm so
# neither objective silently dominates the other.
#
# Re-measure before changing these: gen_dataset/probe_guidance.py prints the
# per-step displacement, whether the clip binds, and a reverse table of the
# (lr, weight) needed to hit a target displacement.
thermal_weight="${THERMAL_WEIGHT:-3.0}"
thermal_lr="${THERMAL_LR:-0.15}"
thermal_steps="${THERMAL_STEPS:-1}"
thermal_clip="${THERMAL_CLIP:-0.02}"

# Force-application window, as a fraction of sampling progress.  Was 0.75 -> 0.90,
# which for max_diffusion_steps=100 is a ~25 step window; 0.5 -> 0.8 doubles it to
# ~50 and lines it up with the wirelength schedule above.
thermal_start="${THERMAL_START:-0.5}"
thermal_full="${THERMAL_FULL:-0.8}"

# eval_thermal_guided.py:1018 adds this to the *thermal* objective.  It was 0.0,
# which was harmless while the thermal step could not move anything; now that the
# step is ~3e-3 normalized, the thermal gradient can push chiplets into overlap
# and would otherwise rely entirely on the outer legality_guidance_weight=12.0 to
# undo it.  Small positive value so the thermal step polices itself.
thermal_legality_weight="${THERMAL_LEGALITY_WEIGHT:-1.0}"

# 采样后的热精修 (LayoutGenModel/diffusion/thermal_refine.py)。这一段和上面那些
# 采样时引导是**两回事**: 上面那些在采样循环里和模型流场对抗, 实测 12 个 case 只
# 兑现 0.53K; 这一段在采样**结束、合法化之后**直接在最终布局上做合法性保持的局部
# 搜索, 同样这 12 个布局拿到 -3 ~ -9K。
#
# 顺序是"先合法化再精修", 不能反过来: 1000 步合法化会把 400 步精修的热收益抹掉。
# 精修自己用候选过滤保证每一步都合法 (放下去不重叠且不出画布才接受), 所以不需要
# 合法化给它兜底 —— 它跑在最后, 输出即最终布局。
#
# wirelength_weight 默认 1e-4 是实测选的: 用真线长/热代理在 12 个 case 上,
# 1e-4 温度平均 -4.81K、真线长平均 -2.06%; 3e-4 线长稳定改善 12-13% 但温度收益减半;
# 0 (纯热) 会把小 case 的线长搞坏到 +74%, 所以线长项不能省。
thermal_refine_enabled="${THERMAL_REFINE_ENABLED:-True}"
thermal_refine_steps="${THERMAL_REFINE_STEPS:-400}"
thermal_refine_candidates="${THERMAL_REFINE_CANDIDATES:-32}"
thermal_refine_radius="${THERMAL_REFINE_RADIUS:-0.30}"
thermal_refine_wirelength_weight="${THERMAL_REFINE_WIRELENGTH_WEIGHT:-1.0e-4}"
thermal_refine_seed="${THERMAL_REFINE_SEED:-0}"
thermal_refine_verbose="${THERMAL_REFINE_VERBOSE:-False}"

compat_dir="$layout_root/diffusion/_env_compat"
required_paths=(
  "$python_executable"
  "$checkpoint"
  "$compat_dir/sitecustomize.py"
  "$layout_root/diffusion/eval_thermal_guided.py"
  "$layout_root/diffusion/thermal_refine.py"
  "$layout_root/evaluation/aggregate_hubump_multiseed.py"
  "$layout_root/datasets/graph/cases_hubump/config.yaml"
)
for required_path in "${required_paths[@]}"; do
  [[ -e "$required_path" ]] || { echo "Required sampling input is missing: $required_path" >&2; exit 1; }
done
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
export PYTHONPATH="$compat_dir:.:diffusion:.."
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
  echo "Thermal guidance: weight=$thermal_weight lr=$thermal_lr steps=$thermal_steps clip=$thermal_clip"
  echo "  Window: progress > $thermal_start, ramping to $thermal_full"
  echo "  Thermal-step legality weight: $thermal_legality_weight"
  echo "  Probe before changing these: python gen_dataset/probe_guidance.py --placement-dir <seed>/placement"
  echo "Thermal refine (post-sampling): enabled=$thermal_refine_enabled steps=$thermal_refine_steps candidates=$thermal_refine_candidates radius=$thermal_refine_radius"
  echo "  Objective: T_max + $thermal_refine_wirelength_weight * wHPWL (search) ; seed=$thermal_refine_seed verbose=$thermal_refine_verbose"
  echo "  Runs AFTER legalization; candidate filtering keeps every step legal."
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
      wirelength.guidance_weight="$wirelength_weight" \
      wirelength.initial_weight=0.0 \
      wirelength.start=0.5 \
      wirelength.full=0.8 \
      model.hpwl_guidance_weight=0.0 \
      model.guidance_schedule.hpwl_initial_weight=0.0 \
      model.guidance_schedule.hpwl_final_weight=0.0 \
      thermal.guidance_weight="$thermal_weight" \
      thermal.guidance_lr="$thermal_lr" \
      thermal.guidance_steps="$thermal_steps" \
      thermal.grad_clip="$thermal_clip" \
      thermal.report_guidance_enabled=True \
      thermal.schedule.enabled=True \
      thermal.schedule.start="$thermal_start" \
      thermal.schedule.full="$thermal_full" \
      thermal.schedule.initial_weight=0.0 \
      thermal.schedule.final_weight="$thermal_weight" \
      thermal.legality_weight="$thermal_legality_weight" \
      model.heat_repulsion_guidance_weight=0.0 \
      model.guidance_schedule.heat_repulsion_initial_weight=0.0 \
      model.guidance_schedule.heat_repulsion_final_weight=0.0 \
      legalization.mode=standard \
      legalization.grad_descent_steps=1000 \
      thermal_refine.enabled="$thermal_refine_enabled" \
      thermal_refine.steps="$thermal_refine_steps" \
      thermal_refine.candidates="$thermal_refine_candidates" \
      thermal_refine.radius="$thermal_refine_radius" \
      thermal_refine.wirelength_weight="$thermal_refine_wirelength_weight" \
      thermal_refine.seed="$thermal_refine_seed" \
      thermal_refine.verbose="$thermal_refine_verbose" \
      2>&1 | tee "$seed_log"
    echo "[$((offset + 1))/$num_seeds] finished seed=$seed" | tee -a "$master_log"
  fi

  "$python_executable" -m evaluation.aggregate_hubump_multiseed \
    "$method_dir" --output "$summary_csv" | tee -a "$master_log"
done

echo "Completed $num_seeds seeds x $num_cases cases = $((num_seeds * num_cases)) candidates."
echo "Combined metrics: $summary_csv"
echo "Per-seed logs: $batch_log_dir"
