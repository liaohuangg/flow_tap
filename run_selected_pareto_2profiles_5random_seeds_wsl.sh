#!/usr/bin/env bash
set -euo pipefail

# Timing run of the selected per-case thermal/wirelength parameter profiles.
#
# Three things differ from the original sampled sweep this file used to be:
#
#   * One profile, one seed by default (alpha_0p9 x seed 61).  This is a
#     timing probe, not a comparison table, so the second profile column and
#     the four extra seeds only multiplied the wall clock without adding
#     information.  The sweep is still reachable -- PROFILES and NUM_SEEDS
#     drive the same loops they always did:
#       PROFILES="alpha_0p9 alpha_0p1" NUM_SEEDS=5 SEEDS="61,62,63,64,65" \
#         ./run_selected_pareto_2profiles_5random_seeds_wsl.sh
#
#   * Everything runs on the CPU.  CUDA_VISIBLE_DEVICES is emptied before any
#     interpreter starts, so torch.cuda.is_available() is False inside
#     eval_thermal_guided.py (which picks its device that way) and inside the
#     legalizer (which additionally gets an explicit --device cpu).  Nothing
#     in the pipeline can fall back to the GPU even if one is present.
#
#   * Inference and legalization are timed separately.  The old script only
#     kept a total, which cannot tell you which of the two stages is the CPU
#     bottleneck.  Both land in timing.csv alongside the total.
#
# Output: "thermal_weight wirelength_weight".

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
checkpoint="${CHECKPOINT:-$layout_root/logs/model/placement_pareto_clean/placement-pareto-clean-neural-pairdist-200k/seed_61/latest.ckpt}"
legalizer="$layout_root/legalizeLayout/legalize_layout.py"
compat_dir="$layout_root/diffusion/_env_compat"

profiles_input="${PROFILES:-alpha_0p9}"
num_seeds="${NUM_SEEDS:-1}"
jobs="${JOBS:-1}"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# CPU-only switch.  Set CPU_ONLY=0 to get the old GPU behaviour back.
cpu_only="${CPU_ONLY:-1}"
# torch's own default is the *physical* core count -- 14 on this 14-core /
# 28-thread box -- which leaves half the machine idle and makes the timing a
# half-box number.  Default to every logical CPU instead; set CPU_THREADS to
# pin it (e.g. 14 for a physical-core-only measurement).
cpu_threads="${CPU_THREADS:-$(nproc 2>/dev/null || echo "")}"

# Objective-weight scales, applied on top of the per-case table below.  The
# table was tuned against the GPU sweep; these let a run push thermal harder,
# ease off wirelength, or tighten legality without editing 24 table rows.
# 1 leaves the table's own values alone.
#
# Current defaults are the "thermal up, wirelength down, legality up" setting:
# thermal x10, wirelength /10, legality x4 (12.0 -> 48.0 initial/mid, 10.0 ->
# 40.0 final).  Set all three to 1 to reproduce the published table.
#
# Caveat worth knowing before reading the results: guidance_lr and grad_clip
# are deliberately left at 0.10, so the thermal step saturates at lr * clip
# once the weighted gradient exceeds the clip.  For the cases whose table
# weight is already large (Case10 and xerox8_m are 100) raising it further
# therefore changes nothing -- the displacement is pinned by the clip, not the
# weight.  The low-weight cases (hp11_m 0.03, cpu-dram 0.1, Case9 1, acend910
# 3) are the ones x10 will actually move.
thermal_scale="${THERMAL_SCALE:-10}"
wirelength_scale="${WIRELENGTH_SCALE:-0.1}"
legality_scale="${LEGALITY_SCALE:-4}"

# Legality guidance before scaling.  Deliberately stronger than the config
# defaults, and held flat across the schedule's start and middle -- only the
# tail relaxes, so the final steps are not the ones undoing overlaps.
legality_initial_base="${LEGALITY_INITIAL_BASE:-12.0}"
legality_final_base="${LEGALITY_FINAL_BASE:-10.0}"

# Canvas size, per case.  The loader does not take a canvas as input -- it
# derives one so the chiplet bodies fill target_utilization of it
# (json_benchmark_dataset.py:102):
#
#     side = sqrt(body_area / utilization) = sqrt(body_area * canvas_multiple)
#
# and then rejects the case if a single hubump-expanded footprint is longer than
# that side.  A multiple of 2 (utilization 0.50) is the repo-wide convention and
# is comfortable for 14 of the 16 cases -- the next tightest after the two below
# is hp11_m at 1.25, i.e. everything else would still fit at 0.80.
#
# Two hp cases are the exception.  Both carry the same 33.04mm die (F/G) while
# their total body area is small, so the derived canvas ends up shorter than the
# die it has to hold: hp8_m misses by 0.12mm (needs >= 2.015) and hp6_m needs
# >= 3.468.  They get their own multiple rather than the whole benchmark being
# loosened, since loosening the canvas lowers the packing density and would make
# every other row an easier problem than the one it was measured on.
#
# These rows are therefore NOT density-comparable with the rest: a smaller
# utilization means a looser packing problem, which flatters both the wall clock
# and the quality metrics.  Compare them against each other, not against the 14.
canvas_multiple="${CANVAS_MULTIPLE:-2}"
hp6_m_canvas_multiple="${HP6_M_CANVAS_MULTIPLE:-3.5}"
hp8_m_canvas_multiple="${HP8_M_CANVAS_MULTIPLE:-3}"

# CHIPLET_AREA_RATIO is the older spelling of the same knob, and every other
# script in the repo still uses it.  Honour it if set so a caller who exports it
# keeps getting what they asked for, rather than silently getting the default.
if [[ -n "${CHIPLET_AREA_RATIO:-}" ]]; then
  canvas_multiple="$(awk -v utilization="$CHIPLET_AREA_RATIO" 'BEGIN { printf "%.6g", 1.0 / utilization }')"
fi

canvas_multiple_for() {
  case "$1" in
    hp6_m) printf '%s' "$hp6_m_canvas_multiple" ;;
    hp8_m) printf '%s' "$hp8_m_canvas_multiple" ;;
    *) printf '%s' "$canvas_multiple" ;;
  esac
}

# Invert the multiple rather than hardcoding 0.3333/0.2857, so there is one
# number to change and the rounded constant can never drift from it.
canvas_utilization_for() {
  awk -v multiple="$(canvas_multiple_for "$1")" 'BEGIN { printf "%.6g", 1.0 / multiple }'
}

[[ -x "$python_executable" ]] || { echo "Python not found: $python_executable" >&2; exit 1; }
[[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
[[ -f "$legalizer" ]] || { echo "Legalizer not found: $legalizer" >&2; exit 1; }
[[ -f "$compat_dir/sitecustomize.py" ]] || { echo "Env compat shim not found: $compat_dir/sitecustomize.py" >&2; exit 1; }
[[ "$num_seeds" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_SEEDS must be positive" >&2; exit 2; }
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be positive" >&2; exit 2; }
[[ "$cpu_only" =~ ^[01]$ ]] || { echo "CPU_ONLY must be 0 or 1" >&2; exit 2; }

read -r -a profiles <<< "$profiles_input"
for profile in "${profiles[@]}"; do
  case "$profile" in
    alpha_0p9|alpha_0p1) ;;
    *) echo "Unknown profile: $profile (expected alpha_0p9 / alpha_0p1)" >&2; exit 2 ;;
  esac
done

all_cases=(Case10 Case6 Case7 Case8 Case9 acend910 cpu-dram hp11_m hp6_m hp8_m multigpu syn1 syn4 xerox6_m xerox7_m xerox8_m)

# A case's dataset index is a property of the directory, not of the case.  The
# loader globs benchmark/cases_hubump, sorts by filename, and uses the list
# position as the output index (json_benchmark_dataset.py:79), so dropping a
# new JSON into that directory renumbers every case that sorts after it.  The
# hardcoded index table this replaced did not know that: adding hp6_m/hp8_m
# silently shifted multigpu 8->10, syn1 9->11, syn4 10->12 and xerox8_m 11->15,
# and the run would have quietly sampled the wrong four circuits.
#
# So derive the position from the same sorted glob the loader uses rather than
# restating it.  glob() is driven by Python here, not by shell globbing, because
# the shell's collation depends on locale and the loader's does not -- a
# mismatch would renumber the table in a way that is invisible at run time.
benchmark_dir="$repo_root/benchmark/cases_hubump"
[[ -d "$benchmark_dir" ]] || { echo "Benchmark dir not found: $benchmark_dir" >&2; exit 1; }
mapfile -t benchmark_cases < <("$python_executable" - "$benchmark_dir" <<'PY'
from pathlib import Path
import sys

for path in sorted(Path(sys.argv[1]).glob("*.json")):
    print(path.stem)
PY
)
[[ "${#benchmark_cases[@]}" -gt 0 ]] || { echo "No JSON cases in $benchmark_dir" >&2; exit 1; }

case_index_of() {
  local wanted="$1" i
  for i in "${!benchmark_cases[@]}"; do
    if [[ "${benchmark_cases[$i]}" == "$wanted" ]]; then
      printf '%s' "$i"
      return 0
    fi
  done
  return 1
}

# CASES narrows the run to a subset; the default is the full table.  Handy for
# a smoke test before committing to the whole probe.
if [[ -n "${CASES:-}" ]]; then
  read -r -a cases <<< "${CASES//,/ }"
else
  cases=("${all_cases[@]}")
fi

case_indices=()
for case_name in "${cases[@]}"; do
  case_index="$(case_index_of "$case_name")" || {
    echo "Unknown case: $case_name (not in $benchmark_dir; have: ${benchmark_cases[*]})" >&2
    exit 2
  }
  case_indices+=("$case_index")
done

weights_for() {
  local profile="$1" case_name="$2"
  case "$profile:$case_name" in
    alpha_0p9:Case10)   echo "100 2" ;;
    alpha_0p9:Case6)    echo "10 2" ;;
    alpha_0p9:Case7)    echo "30 2" ;;
    alpha_0p9:Case8)    echo "30 2" ;;
    alpha_0p9:Case9)    echo "1 2" ;;
    alpha_0p9:acend910) echo "3 2" ;;
    alpha_0p9:cpu-dram) echo "0.1 50" ;;
    alpha_0p9:hp11_m)   echo "0.03 2" ;;
    # hp6_m/hp8_m and xerox6_m/xerox7_m are smaller siblings of hp11_m and
    # xerox8_m and arrived without a tuned row of their own.  Inherit the
    # sibling's weights: this run measures wall clock, and an untuned weight
    # would still produce a valid timing while making the row look like it
    # belonged to a different experiment.
    alpha_0p9:hp6_m)    echo "0.03 2" ;;
    alpha_0p9:hp8_m)    echo "0.03 2" ;;
    alpha_0p9:multigpu) echo "30 0.02" ;;
    alpha_0p9:syn1)     echo "30 0.2" ;;
    alpha_0p9:syn4)     echo "30 0.2" ;;
    alpha_0p9:xerox6_m) echo "100 0.02" ;;
    alpha_0p9:xerox7_m) echo "100 0.02" ;;
    alpha_0p9:xerox8_m) echo "100 0.02" ;;

    alpha_0p1:Case10)   echo "100 0" ;;
    alpha_0p1:Case6)    echo "10 2" ;;
    alpha_0p1:Case7)    echo "10 50" ;;
    alpha_0p1:Case8)    echo "30 2" ;;
    alpha_0p1:Case9)    echo "1 2" ;;
    alpha_0p1:acend910) echo "3 2" ;;
    alpha_0p1:cpu-dram) echo "0.1 50" ;;
    alpha_0p1:hp11_m)   echo "1 50" ;;
    alpha_0p1:hp6_m)    echo "1 50" ;;
    alpha_0p1:hp8_m)    echo "1 50" ;;
    alpha_0p1:multigpu) echo "0.03 50" ;;
    alpha_0p1:syn1)     echo "1 50" ;;
    alpha_0p1:syn4)     echo "10 0" ;;
    alpha_0p1:xerox6_m) echo "100 0.02" ;;
    alpha_0p1:xerox7_m) echo "100 0.02" ;;
    alpha_0p1:xerox8_m) echo "100 0.02" ;;
    *) echo "Missing parameter mapping: $profile/$case_name" >&2; return 2 ;;
  esac
}

weight_label() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

# Wall-clock helper.  date +%s alone rounds to the second, which is too coarse
# once a stage takes tens of seconds -- the difference between two stages would
# be buried in the rounding.
now() { date +%s.%N; }
elapsed() { awk -v start="$1" -v end="$2" 'BEGIN { printf "%.3f", end - start }'; }

# Multiply a per-case weight by a scale, and print it plainly -- %g rather than
# %f so the value that reaches hydra is what a human would have typed (0.2, not
# 0.200000) and shows up legibly in the method name and the logs.
scaled() { awk -v value="$1" -v factor="$2" 'BEGIN { printf "%.6g", value * factor }'; }

# Pull the stage breakdown out of the two JSON reports a run leaves behind.
#
# Both processes already time themselves internally, but neither record is a
# complete account of its own wall clock, so this emits the internal segments
# AND the two residuals that close the gap:
#
#   inference  = preprocess | sample loop | post-sample | metrics  + launch
#   legalizer  = stage A | stage B (+ hotspot locate)             + launch
#
# "launch" is interpreter start, imports, checkpoint/surrogate load and
# process teardown -- everything the internal timers never see.  Without it the
# segments silently fail to add up to the wall clock in columns 10/11, and the
# missing time looks like measurement noise instead of unmeasured work.
#
# What is deliberately NOT here: a split of the sampling loop by objective
# (forward vs legality vs thermal vs wirelength guidance).  model.reverse_samples
# has no timers inside it, so that breakdown does not exist in the code today.
# It is one segment, reported as sample_loop_seconds.
timing_details() {
  "$python_executable" - "$1" "$2" "${3:-}" "${4:-}" <<'PY'
import json
import sys


def load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def fmt(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def to_float(value):
    """The wall-clock bounds arrive as shell strings, not JSON numbers."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def residual(wall, *segments):
    """Wall clock minus the segments that claim to explain it.

    Negative values are real -- the internal stopwatch starts after some setup
    and can overrun the shell's coarse reading -- so they are reported as-is
    rather than clamped to zero.
    """
    total = to_float(wall)
    if total is None:
        return ""
    known = sum(part for part in segments if part is not None)
    return total - known


metrics = load(sys.argv[1])
report = load(sys.argv[2])
wall_inference = sys.argv[3]
wall_legalization = sys.argv[4]

preprocess = number(metrics.get("preprocess_time"))
loop = number(metrics.get("model_time"))
postsample = number(metrics.get("postsample_time"))
eval_time = number(metrics.get("eval_time"))
generation = number(metrics.get("generation_time"))

timing = report.get("timing_s") or {}
stage_a = number(timing.get("stage_a_repair"))
stage_b = number(timing.get("stage_b_refine"))
legal_total = number(timing.get("total"))
hotspot = report.get("hotspot") or {}
locate = number(hotspot.get("locate_time_s"))

print(
    ",".join(
        [
            # inference, in execution order
            fmt(preprocess),
            fmt(loop),
            fmt(postsample),
            fmt(eval_time),
            fmt(generation),
            fmt(residual(wall_inference, generation, eval_time)),
            # legalizer, in execution order
            fmt(stage_a),
            fmt(stage_b),
            fmt(locate),
            # Inside the legalizer's own total but outside every stage it labels:
            # report assembly, figure rendering, JSON writes.
            fmt(residual(legal_total, stage_a, stage_b, locate)),
            fmt(legal_total),
            fmt(residual(wall_legalization, legal_total)),
            # outcome, so a row can be read without opening the reports
            fmt(report.get("num_chiplets")),
            fmt(metrics.get("thermal_max_c")),
            fmt(metrics.get("legality_2")),
            fmt(metrics.get("expanded_legality_2")),
        ]
    )
)
PY
}

batch_root="$layout_root/logs/output/cases_hubump/selected-pareto-${run_tag}"
log_dir="$batch_root/logs"
seed_file="$batch_root/seeds.txt"
manifest="$batch_root/manifest.csv"
timing_csv="$batch_root/timing.csv"
mkdir -p "$log_dir"

if [[ -n "${SEEDS:-}" ]]; then
  read -r -a seeds <<< "${SEEDS//,/ }"
  printf '%s\n' "${seeds[@]}" > "$seed_file"
elif [[ "$num_seeds" -eq 1 ]]; then
  # Single-seed timing probe: fixed 61, matching the seed_61 checkpoint every
  # other script in this repo samples from, so runs are comparable.
  seeds=("${SEED:-61}")
  printf '%s\n' "${seeds[@]}" > "$seed_file"
elif [[ -s "$seed_file" ]]; then
  mapfile -t seeds < "$seed_file"
else
  mapfile -t seeds < <("$python_executable" - "$num_seeds" <<'PY'
import secrets
import sys

count = int(sys.argv[1])
for seed in secrets.SystemRandom().sample(range(1, 2_147_483_647), count):
    print(seed)
PY
  )
  printf '%s\n' "${seeds[@]}" > "$seed_file"
fi

[[ "${#seeds[@]}" -eq "$num_seeds" ]] || {
  echo "Expected $num_seeds seeds, found ${#seeds[@]} in $seed_file" >&2
  exit 2
}
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done

if [[ ! -f "$manifest" ]]; then
  echo "profile,case,case_index,seed,thermal_weight,wirelength_weight,method,raw_json,legalized_json" > "$manifest"
fi
if [[ ! -f "$timing_csv" ]]; then
  {
    printf '%s\n' "profile,case,seed,thermal_weight,wirelength_weight,legality_weight,device,"\
"inference_status,legalization_status,"\
"wall_inference_seconds,wall_legalization_seconds,total_seconds,"\
"sample_preprocess_seconds,sample_loop_seconds,sample_postsample_seconds,sample_metrics_seconds,"\
"sample_generation_seconds,inference_launch_seconds,"\
"legal_stage_a_seconds,legal_stage_b_seconds,legal_locate_seconds,legal_unattributed_seconds,"\
"legal_internal_total_seconds,legal_launch_seconds,"\
"num_chiplets,thermal_max_c,legality_2,expanded_legality_2,"\
"canvas_multiple,inference_log,legalization_log"
  } > "$timing_csv"
fi

cd "$layout_root"
# _env_compat carries the NumPy<2 aliases wandb still imports at module load;
# without it `import wandb` dies inside utils.py before any layout is sampled.
export PYTHONPATH=".:diffusion:..:$compat_dir"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
# FLOW_TAP_CHIPLET_AREA_RATIO is set per case inside run_one from
# canvas_utilization_for -- it is not a single value any more, because two cases
# need a looser canvas than the rest.  Setting it here as well would just be a
# second source of truth for the same number.

if [[ "$cpu_only" == "1" ]]; then
  # Emptied rather than unset: an empty value is honoured by every CUDA
  # runtime and makes torch.cuda.is_available() return False, which is exactly
  # the branch eval_thermal_guided.py and legalize_layout.py test.
  export CUDA_VISIBLE_DEVICES=""
  legalizer_device="cpu"
  device_label="cpu/${cpu_threads:-auto}threads"
else
  legalizer_device="cuda"
  device_label="cuda"
fi

# FLOW_TAP_TORCH_THREADS is the one that actually lands: this env's OpenMP
# runtime ignores OMP_NUM_THREADS (set it to 28 on this box and
# torch.get_num_threads() still reports the 14 physical cores), so the pool can
# only be widened from inside the process.  _env_compat/sitecustomize.py does
# that at interpreter startup, before the first parallel region -- for the
# legalizer too, since it gets the same PYTHONPATH.  The OMP/MKL variables are
# kept as belt-and-braces for any library that does read them.
if [[ -n "$cpu_threads" ]]; then
  export FLOW_TAP_TORCH_THREADS="$cpu_threads"
  export OMP_NUM_THREADS="$cpu_threads"
  export MKL_NUM_THREADS="$cpu_threads"
  export NUMEXPR_NUM_THREADS="$cpu_threads"
fi

# Prove the CPU-only claim instead of asserting it: this is the same call the
# pipeline itself makes to choose its device.
device_report="$("$python_executable" - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | threads={torch.get_num_threads()}")
PY
)"
echo "Device: $device_report"

run_one() {
  local profile="$1" case_name="$2" case_index="$3" seed="$4"
  local thermal_weight wirelength_weight thermal_label wirelength_label method
  local output_dir raw_json legalized_dir legal_json inference_log legalization_log
  local metrics_json legal_report_json details
  local legality_initial legality_final
  local start_seconds inference_start inference_end legalization_start legalization_end
  local total_seconds inference_seconds legalization_seconds inference_rc legalization_rc
  local canvas_multiple_used
  local inference_status="not_run" legalization_status="not_run"

  read -r thermal_weight wirelength_weight < <(weights_for "$profile" "$case_name")
  thermal_weight="$(scaled "$thermal_weight" "$thermal_scale")"
  wirelength_weight="$(scaled "$wirelength_weight" "$wirelength_scale")"
  legality_initial="$(scaled "$legality_initial_base" "$legality_scale")"
  legality_final="$(scaled "$legality_final_base" "$legality_scale")"
  thermal_label="$(weight_label "$thermal_weight")"
  wirelength_label="$(weight_label "$wirelength_weight")"
  method="selected-${profile}-${case_name}-t${thermal_label}-w${wirelength_label}-${run_tag}"
  output_dir="$layout_root/logs/output/cases_hubump/$method/seed_$seed"
  raw_json="$output_dir/placement/$(printf '%02d' "$case_index")_${case_name}_placement.json"
  legalized_dir="$output_dir/legalized"
  legal_json="$legalized_dir/$(basename "${raw_json%.json}")_legal.json"
  legal_report_json="$legalized_dir/$(basename "${raw_json%.json}")_legal_report.json"
  metrics_json="$output_dir/metrics_summary.json"
  inference_log="$log_dir/${profile}_${case_name}_seed${seed}_inference.log"
  legalization_log="$log_dir/${profile}_${case_name}_seed${seed}_legalization.log"
  start_seconds="$(now)"

  # The canvas has to be fixed before either process starts, and both read the
  # same variable -- the legalizer rebuilds the condition set from the placement
  # JSON, so it has to agree with the sampler about how big the canvas is.
  canvas_multiple_used="$(canvas_multiple_for "$case_name")"
  export FLOW_TAP_CHIPLET_AREA_RATIO="$(canvas_utilization_for "$case_name")"

  inference_seconds="0.000"
  legalization_seconds="0.000"

  if [[ -s "$raw_json" ]]; then
    inference_status="skipped_existing"
  else
    echo "[$profile][$case_name][seed=$seed] inference thermal=$thermal_weight wl=$wirelength_weight canvas=${canvas_multiple_used}x"
    set +e
    inference_start="$(now)"
    "$python_executable" diffusion/eval_thermal_guided.py \
      --config-name config_eval_fm \
      task=cases_hubump \
      "method=$method" \
      "seed=$seed" \
      logger.wandb=False \
      "from_checkpoint=$checkpoint" \
      eval_samples=0 \
      num_output_samples=1 \
      "+output_indices=[$case_index]" \
      val_batch_size=1 \
      model.max_diffusion_steps=100 \
      model.grad_descent_rate=0.1 \
      model.grad_descent_steps=8 \
      "model.legality_guidance_weight=$legality_initial" \
      "model.guidance_schedule.legality_initial_weight=$legality_initial" \
      "model.guidance_schedule.legality_mid_weight=$legality_initial" \
      "model.guidance_schedule.legality_final_weight=$legality_final" \
      model.backbone_params.auxiliary_legality_heads_enabled=True \
      wirelength.enabled=True \
      wirelength.pair_feature_enabled=False \
      "wirelength.guidance_weight=$wirelength_weight" \
      "wirelength.initial_weight=$wirelength_weight" \
      wirelength.start=0.0 \
      wirelength.full=0.0 \
      model.hpwl_guidance_weight=0.0 \
      model.guidance_schedule.hpwl_initial_weight=0.0 \
      model.guidance_schedule.hpwl_final_weight=0.0 \
      "thermal.guidance_weight=$thermal_weight" \
      thermal.guidance_lr=0.10 \
      thermal.guidance_steps=2 \
      thermal.grad_clip=0.10 \
      thermal.report_guidance_enabled=True \
      thermal.schedule.enabled=True \
      thermal.schedule.start=0.0 \
      thermal.schedule.full=0.0 \
      "thermal.schedule.initial_weight=$thermal_weight" \
      "thermal.schedule.final_weight=$thermal_weight" \
      thermal.legality_weight=1.0 \
      model.heat_repulsion_guidance_weight=0.0 \
      model.guidance_schedule.heat_repulsion_initial_weight=0.0 \
      model.guidance_schedule.heat_repulsion_final_weight=0.0 \
      legalization.mode=none \
      >"$inference_log" 2>&1
    inference_rc=$?
    inference_end="$(now)"
    set -e
    inference_seconds="$(elapsed "$inference_start" "$inference_end")"
    if [[ "$inference_rc" -eq 0 && -s "$raw_json" ]]; then
      inference_status="ok"
    else
      inference_status="failed_rc_${inference_rc}"
    fi
  fi

  if [[ -s "$legal_json" ]]; then
    legalization_status="skipped_existing"
  elif [[ -s "$raw_json" ]]; then
    mkdir -p "$legalized_dir"
    set +e
    legalization_start="$(now)"
    "$python_executable" "$legalizer" \
      --input "$raw_json" \
      --output-dir "$legalized_dir" \
      --device "$legalizer_device" \
      >"$legalization_log" 2>&1
    legalization_rc=$?
    legalization_end="$(now)"
    set -e
    legalization_seconds="$(elapsed "$legalization_start" "$legalization_end")"
    if [[ -s "$legal_json" ]]; then
      legalization_status="ok"
    else
      legalization_status="failed_rc_${legalization_rc}"
    fi
  fi

  total_seconds="$(elapsed "$start_seconds" "$(now)")"
  # Empty fields rather than a skipped row when a stage never produced its
  # report -- a failed run still has to show up in the timing table.
  details="$(timing_details "$metrics_json" "$legal_report_json" "$inference_seconds" "$legalization_seconds")"
  # 16 empty fields, matching the number timing_details emits (15 commas).
  [[ -n "$details" ]] || details=",,,,,,,,,,,,,,,"
  {
    flock 9
    if ! grep -Fq ",$method,$raw_json," "$manifest"; then
      echo "$profile,$case_name,$case_index,$seed,$thermal_weight,$wirelength_weight,$method,$raw_json,$legal_json" >> "$manifest"
    fi
    echo "$profile,$case_name,$seed,$thermal_weight,$wirelength_weight,$legality_initial,$device_label,$inference_status,$legalization_status,$inference_seconds,$legalization_seconds,$total_seconds,$details,$canvas_multiple_used,$inference_log,$legalization_log" >> "$timing_csv"
  } 9>>"$batch_root/state.lock"
  echo "[$profile][$case_name][seed=$seed] $inference_status / $legalization_status (wall: inference ${inference_seconds}s, legalization ${legalization_seconds}s, total ${total_seconds}s | stages: $details)"
}

echo "Checkpoint: $checkpoint"
echo "Profiles: ${profiles[*]}"
echo "Cases: ${cases[*]}"
echo "Seeds: ${seeds[*]}"
echo "CPU only: $cpu_only (legalizer device: $legalizer_device)"
echo "Weight scales: thermal x$thermal_scale, wirelength x$wirelength_scale, legality x$legality_scale"
echo "Legality weights: initial/mid $(scaled "$legality_initial_base" "$legality_scale"), final $(scaled "$legality_final_base" "$legality_scale")"
echo "Total runs: $((${#profiles[@]} * ${#cases[@]} * ${#seeds[@]}))"
echo "Output: $batch_root"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for profile in "${profiles[@]}"; do
    for i in "${!cases[@]}"; do
      read -r thermal_weight wirelength_weight < <(weights_for "$profile" "${cases[$i]}")
      echo "$profile,${cases[$i]},${case_indices[$i]},seed=${seeds[0]},thermal=$(scaled "$thermal_weight" "$thermal_scale"),wl=$(scaled "$wirelength_weight" "$wirelength_scale"),canvas=$(canvas_multiple_for "${cases[$i]}")x,util=$(canvas_utilization_for "${cases[$i]}")"
    done
  done
  exit 0
fi

batch_start="$(now)"

active_jobs=0
for profile in "${profiles[@]}"; do
  for i in "${!cases[@]}"; do
    for seed in "${seeds[@]}"; do
      if [[ "$jobs" -eq 1 ]]; then
        run_one "$profile" "${cases[$i]}" "${case_indices[$i]}" "$seed"
      else
        run_one "$profile" "${cases[$i]}" "${case_indices[$i]}" "$seed" &
        active_jobs=$((active_jobs + 1))
        if [[ "$active_jobs" -ge "$jobs" ]]; then
          wait -n
          active_jobs=$((active_jobs - 1))
        fi
      fi
    done
  done
done
if [[ "$active_jobs" -gt 0 ]]; then
  wait
fi

batch_seconds="$(elapsed "$batch_start" "$(now)")"

# Per-stage totals, so the split between sampling and legalization -- and
# inside each of those, forward pass vs guidance and Stage A vs Stage B -- is
# readable without opening the CSV by hand.
awk -F, -v grand="$batch_seconds" '
  NR > 1 {
    wall_inf += $10+0; wall_leg += $11+0;
    preprocess += $13+0; loop += $14+0; postsample += $15+0; sample_metrics += $16+0;
    generation += $17+0; launch_inf += $18+0;
    stage_a += $19+0; stage_b += $20+0; locate += $21+0; unattributed += $22+0;
    legal_total += $23+0; launch_leg += $24+0;
    printf "  %-9s infer %7.1fs (pre %6.1f loop %7.1f post %6.1f launch %5.1f) | legal %7.1fs (A %6.1f B %6.1f other %5.1f launch %5.1f)\n",
           $2, $10+0, $13+0, $14+0, $15+0, $18+0, $11+0, $19+0, $20+0, $22+0, $24+0
  }
  END {
    n = NR - 1;
    printf "\n--- segment totals over %d run(s) ---\n", n;
    printf "sampling    : preprocess %.1fs | loop %.1fs | post-sample %.1fs | metrics %.1fs | launch %.1fs  => wall %.1fs\n",
           preprocess, loop, postsample, sample_metrics, launch_inf, wall_inf;
    printf "legalization: Stage A %.1fs | Stage B %.1fs | hotspot locate %.1fs | other %.1fs | launch %.1fs  => wall %.1fs\n",
           stage_a, stage_b, locate, unattributed, launch_leg, wall_leg;
    printf "batch total : %.1fs\n", grand;
  }
' "$timing_csv"

echo "Completed. Manifest: $manifest"
echo "Timing: $timing_csv"
echo "Seeds: $seed_file"
