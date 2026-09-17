#!/usr/bin/env bash
set -euo pipefail

# Run the two per-case thermal/wirelength parameter profiles shown in the
# comparison table.  Five random, paired seeds are generated once and recorded
# in seeds.txt; every profile/case uses the same seeds for fair comparison.
# Re-running with the same RUN_TAG reuses seeds.txt and skips completed outputs.

repo_root="${FLOW_TAP_ROOT:-/mnt/d/WORK/96/flow_tap}"
layout_root="$repo_root/LayoutGenModel"
python_executable="${FLOW_TAP_PYTHON:-/home/user/miniconda3/envs/chipdiffusion/bin/python}"
checkpoint="${CHECKPOINT:-$layout_root/logs/model/placement_pareto_clean/placement-pareto-clean-neural-pairdist-200k/seed_61/latest.ckpt}"
legalizer="$layout_root/legalizeLayout/legalize_layout.py"
num_seeds="${NUM_SEEDS:-5}"
jobs="${JOBS:-1}"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

[[ -x "$python_executable" ]] || { echo "Python not found: $python_executable" >&2; exit 1; }
[[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
[[ -f "$legalizer" ]] || { echo "Legalizer not found: $legalizer" >&2; exit 1; }
[[ "$num_seeds" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_SEEDS must be positive" >&2; exit 2; }
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be positive" >&2; exit 2; }

profiles=(alpha_0p9 alpha_0p1)
cases=(Case10 Case6 Case7 Case8 Case9 acend910 cpu-dram hp11_m multigpu syn1 syn4 xerox8_m)
case_indices=(0 1 2 3 4 5 6 7 8 9 10 11)

# Output: "thermal_weight wirelength_weight".
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
    alpha_0p9:multigpu) echo "30 0.02" ;;
    alpha_0p9:syn1)     echo "30 0.2" ;;
    alpha_0p9:syn4)     echo "30 0.2" ;;
    alpha_0p9:xerox8_m) echo "100 0.02" ;;

    alpha_0p1:Case10)   echo "100 0" ;;
    alpha_0p1:Case6)    echo "10 2" ;;
    alpha_0p1:Case7)    echo "10 50" ;;
    alpha_0p1:Case8)    echo "30 2" ;;
    alpha_0p1:Case9)    echo "1 2" ;;
    alpha_0p1:acend910) echo "3 2" ;;
    alpha_0p1:cpu-dram) echo "0.1 50" ;;
    alpha_0p1:hp11_m)   echo "1 50" ;;
    alpha_0p1:multigpu) echo "0.03 50" ;;
    alpha_0p1:syn1)     echo "1 50" ;;
    alpha_0p1:syn4)     echo "10 0" ;;
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

batch_root="$layout_root/logs/output/cases_hubump/selected-pareto-${run_tag}"
log_dir="$batch_root/logs"
seed_file="$batch_root/seeds.txt"
manifest="$batch_root/manifest.csv"
timing_csv="$batch_root/timing.csv"
mkdir -p "$log_dir"

if [[ -n "${SEEDS:-}" ]]; then
  read -r -a seeds <<< "${SEEDS//,/ }"
  if [[ "${#seeds[@]}" -ne "$num_seeds" ]]; then
    echo "SEEDS must contain exactly NUM_SEEDS=$num_seeds integers" >&2
    exit 2
  fi
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
  echo "Expected $num_seeds seeds in $seed_file, found ${#seeds[@]}" >&2
  exit 2
}
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done

if [[ ! -f "$manifest" ]]; then
  echo "profile,case,case_index,seed,thermal_weight,wirelength_weight,method,raw_json,legalized_json" > "$manifest"
fi
if [[ ! -f "$timing_csv" ]]; then
  echo "profile,case,seed,thermal_weight,wirelength_weight,inference_status,legalization_status,total_seconds,inference_log,legalization_log" > "$timing_csv"
fi

cd "$layout_root"
export PYTHONPATH=".:diffusion:.."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export FLOW_TAP_CHIPLET_AREA_RATIO="${CHIPLET_AREA_RATIO:-0.50}"

run_one() {
  local profile="$1" case_name="$2" case_index="$3" seed="$4"
  local thermal_weight wirelength_weight thermal_label wirelength_label method
  local output_dir raw_json legalized_dir legal_json inference_log legalization_log
  local start_seconds total_seconds inference_rc legalization_rc
  local inference_status="not_run" legalization_status="not_run"

  read -r thermal_weight wirelength_weight < <(weights_for "$profile" "$case_name")
  thermal_label="$(weight_label "$thermal_weight")"
  wirelength_label="$(weight_label "$wirelength_weight")"
  method="selected-${profile}-${case_name}-t${thermal_label}-w${wirelength_label}-${run_tag}"
  output_dir="$layout_root/logs/output/cases_hubump/$method/seed_$seed"
  raw_json="$output_dir/placement/$(printf '%02d' "$case_index")_${case_name}_placement.json"
  legalized_dir="$output_dir/legalized"
  legal_json="$legalized_dir/$(basename "${raw_json%.json}")_legal.json"
  inference_log="$log_dir/${profile}_${case_name}_seed${seed}_inference.log"
  legalization_log="$log_dir/${profile}_${case_name}_seed${seed}_legalization.log"
  start_seconds="$(date +%s)"

  if [[ -s "$raw_json" ]]; then
    inference_status="skipped_existing"
  else
    echo "[$profile][$case_name][seed=$seed] inference thermal=$thermal_weight wl=$wirelength_weight"
    set +e
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
      model.legality_guidance_weight=12.0 \
      model.guidance_schedule.legality_initial_weight=12.0 \
      model.guidance_schedule.legality_mid_weight=12.0 \
      model.guidance_schedule.legality_final_weight=10.0 \
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
    set -e
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
    "$python_executable" "$legalizer" \
      --input "$raw_json" \
      --output-dir "$legalized_dir" \
      --device cuda \
      >"$legalization_log" 2>&1
    legalization_rc=$?
    set -e
    if [[ -s "$legal_json" ]]; then
      legalization_status="ok"
    else
      legalization_status="failed_rc_${legalization_rc}"
    fi
  fi

  total_seconds=$(( $(date +%s) - start_seconds ))
  {
    flock 9
    if ! grep -Fq ",$method,$raw_json," "$manifest"; then
      echo "$profile,$case_name,$case_index,$seed,$thermal_weight,$wirelength_weight,$method,$raw_json,$legal_json" >> "$manifest"
    fi
    echo "$profile,$case_name,$seed,$thermal_weight,$wirelength_weight,$inference_status,$legalization_status,$total_seconds,$inference_log,$legalization_log" >> "$timing_csv"
  } 9>>"$batch_root/state.lock"
  echo "[$profile][$case_name][seed=$seed] $inference_status / $legalization_status (${total_seconds}s)"
}

echo "Checkpoint: $checkpoint"
echo "Profiles: ${profiles[*]}"
echo "Cases: ${cases[*]}"
echo "Random paired seeds: ${seeds[*]}"
echo "Total runs: $((${#profiles[@]} * ${#cases[@]} * ${#seeds[@]}))"
echo "Output: $batch_root"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for profile in "${profiles[@]}"; do
    for i in "${!cases[@]}"; do
      read -r thermal_weight wirelength_weight < <(weights_for "$profile" "${cases[$i]}")
      echo "$profile,${cases[$i]},${case_indices[$i]},thermal=$thermal_weight,wl=$wirelength_weight"
    done
  done
  exit 0
fi

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

echo "Completed. Manifest: $manifest"
echo "Timing: $timing_csv"
echo "Seeds: $seed_file"
