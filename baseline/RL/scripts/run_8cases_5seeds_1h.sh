#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${RL_DIR}"

DEFAULT_PYTHON="/home/user/miniconda3/envs/chipdiffusion/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

# Seed is the outer loop so seed 0 produces one complete 8-case round first.
SEED_LIST="${SEED_LIST:-0 1 2 3 4}"
CASE_LIST="${CASE_LIST:-acend910 cpu-dram hp11_m multigpu syn1 syn4 syn6 xerox8_m}"
read -r -a CASES <<< "${CASE_LIST//,/ }"

TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-3600}"
MAX_EPOCHS="${MAX_EPOCHS:-600}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-480}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS:-4}"
GRID_RESOLUTION="${GRID_RESOLUTION:-auto}"
MAX_AUTO_GRID_RESOLUTION="${MAX_AUTO_GRID_RESOLUTION:-100}"
EXACT_ACTION_SLOTS="${EXACT_ACTION_SLOTS:-50000}"
THERMAL_INTP_SIZE="${THERMAL_INTP_SIZE:-50}"
PLACEMENT_AREA_WEIGHT="${PLACEMENT_AREA_WEIGHT:-1.0}"
PLACEMENT_PIN_WEIGHT="${PLACEMENT_PIN_WEIGHT:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
HARD_TIMEOUT_GRACE_SECONDS="${HARD_TIMEOUT_GRACE_SECONDS:-600}"
RUN_PREFIX="${RUN_PREFIX:-rl_8cases_5seeds_1h_$(date +%Y%m%d_%H%M%S)}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export MPLCONFIGDIR

BATCH_DIR="${RL_DIR}/runs/${RUN_PREFIX}"
MASTER_LOG="${BATCH_DIR}/experiment.log"
MASTER_SUMMARY="${BATCH_DIR}/summary.tsv"
mkdir -p "${BATCH_DIR}" "${MPLCONFIGDIR}"

{
    echo "experiment=${RUN_PREFIX}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "python=${PYTHON_BIN}"
    echo "cases=${CASES[*]}"
    echo "seeds=${SEED_LIST}"
    echo "time_limit_seconds=${TIME_LIMIT_SECONDS} (includes forced thermal tables, RL, and checkpoint)"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
    echo "num_minibatches=${NUM_MINIBATCHES}"
    echo "ppo_update_epochs=${PPO_UPDATE_EPOCHS}"
    echo "grid_resolution=${GRID_RESOLUTION}"
    echo "max_auto_grid_resolution=${MAX_AUTO_GRID_RESOLUTION}"
    echo "exact_action_slots=${EXACT_ACTION_SLOTS}"
    echo "thermal_intp_size=${THERMAL_INTP_SIZE}"
    echo "placement_area_weight=${PLACEMENT_AREA_WEIGHT}"
    echo "placement_pin_weight=${PLACEMENT_PIN_WEIGHT}"
    echo "thermal_reuse_across_seeds=disabled (--force_thermal_tables)"
    echo "checkpoint=best policy only (no optimizer state)"
    echo "batch_dir=${BATCH_DIR}"
    echo
} | tee "${MASTER_LOG}"

echo "Checking PyTorch and CPLEX..." | tee -a "${MASTER_LOG}"
if ! "${PYTHON_BIN}" -c \
    'import torch; from fastTM.routing import _load_cplex; cplex = _load_cplex(); print(f"torch={torch.__version__}, cplex={cplex.Cplex().get_version()}")' \
    2>&1 | tee -a "${MASTER_LOG}"
then
    echo "Preflight failed: PyTorch or CPLEX is unavailable." | tee -a "${MASTER_LOG}"
    exit 2
fi

HEADER="seed\tcase\tstatus\texit_code\twall_seconds\ttermination\tcompleted_epochs\ttotal_transitions\tthermal_seconds\trl_seconds\ttotal_seconds\tbest_reward\ttemperature\tavg_wirelength\tbbox_utilization\tcheckpoint\tlayout_json\trun_dir"
printf "%b\n" "${HEADER}" > "${MASTER_SUMMARY}"

hard_timeout_seconds=$((TIME_LIMIT_SECONDS + HARD_TIMEOUT_GRACE_SECONDS))
total_runs=0
passed=0
failed=0

for seed in ${SEED_LIST}; do
    ROUND_SUMMARY="${BATCH_DIR}/seed_${seed}_summary.tsv"
    printf "%b\n" "${HEADER}" > "${ROUND_SUMMARY}"
    echo "======================================================================" | tee -a "${MASTER_LOG}"
    echo "Starting complete round for seed=${seed}" | tee -a "${MASTER_LOG}"

    for case_name in "${CASES[@]}"; do
        total_runs=$((total_runs + 1))
        case_json="examples/${case_name}.json"
        run_name="${RUN_PREFIX}_seed${seed}_${case_name}"
        run_dir="${RL_DIR}/runs/${run_name}"
        checkpoint="${run_dir}/checkpoints/ppo_best.pt"
        best_summary="${run_dir}/best_summary.json"
        timing_json="${run_dir}/timing.json"

        cmd=(
            "${PYTHON_BIN}" train.py
            --name "${run_name}"
            --json "${case_json}"
            --num_epochs "${MAX_EPOCHS}"
            --rollout_batch_size "${ROLLOUT_BATCH_SIZE}"
            --num_minibatches "${NUM_MINIBATCHES}"
            --ppo_update_epochs "${PPO_UPDATE_EPOCHS}"
            --log_interval "${LOG_INTERVAL}"
            --grid_resolution "${GRID_RESOLUTION}"
            --max_auto_grid_resolution "${MAX_AUTO_GRID_RESOLUTION}"
            --exact_action_slots "${EXACT_ACTION_SLOTS}"
            --thermal_intp_size "${THERMAL_INTP_SIZE}"
            --placement_area_weight "${PLACEMENT_AREA_WEIGHT}"
            --placement_pin_weight "${PLACEMENT_PIN_WEIGHT}"
            --seed "${seed}"
            --time_limit_seconds "${TIME_LIMIT_SECONDS}"
            --force_thermal_tables
            --save_checkpoint
        )

        echo "----------------------------------------------------------------------" | tee -a "${MASTER_LOG}"
        echo "run=${total_runs}, seed=${seed}, case=${case_name}" | tee -a "${MASTER_LOG}"
        echo "command: ${cmd[*]}" | tee -a "${MASTER_LOG}"
        start_ts="$(date +%s)"

        timeout --preserve-status --signal=INT --kill-after=60s \
            "${hard_timeout_seconds}s" "${cmd[@]}" 2>&1 | tee -a "${MASTER_LOG}"
        exit_code="${PIPESTATUS[0]}"
        wall_seconds=$(( $(date +%s) - start_ts ))

        if [[ "${exit_code}" -eq 0 && -f "${checkpoint}" && -f "${best_summary}" && -f "${timing_json}" ]]; then
            status="ok"
            passed=$((passed + 1))
        elif [[ "${exit_code}" -eq 124 || "${exit_code}" -eq 130 || "${exit_code}" -eq 143 ]]; then
            status="hard_timeout"
            failed=$((failed + 1))
        else
            status="failed"
            failed=$((failed + 1))
        fi

        details=$("${PYTHON_BIN}" - "${timing_json}" "${best_summary}" "${checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

timing_path, best_path, checkpoint_path = map(Path, sys.argv[1:])
timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
metrics = best.get("metrics") or {}

values = [
    timing.get("termination_reason", ""),
    timing.get("completed_epochs", ""),
    timing.get("total_transitions", ""),
    (timing.get("thermal_tables") or {}).get("seconds", ""),
    (timing.get("rl_solve") or {}).get("seconds", ""),
    timing.get("total_seconds", ""),
    best.get("reward", ""),
    metrics.get("rlplanner_temperature", ""),
    metrics.get("rlplanner_avg_wirelength", ""),
    metrics.get("bbox_utilization", ""),
    str(checkpoint_path) if checkpoint_path.exists() else "",
    best.get("json", ""),
]
print("\t".join(str(value) for value in values))
PY
        )

        row="${seed}\t${case_name}\t${status}\t${exit_code}\t${wall_seconds}\t${details}\t${run_dir}"
        printf "%b\n" "${row}" >> "${MASTER_SUMMARY}"
        printf "%b\n" "${row}" >> "${ROUND_SUMMARY}"
        echo "status=${status}, wall_seconds=${wall_seconds}, checkpoint=${checkpoint}" | tee -a "${MASTER_LOG}"
    done

    echo "Completed round seed=${seed}; summary=${ROUND_SUMMARY}" | tee -a "${MASTER_LOG}"
done

{
    echo "======================================================================"
    echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "total_runs=${total_runs}"
    echo "passed=${passed}"
    echo "failed=${failed}"
    echo "summary=${MASTER_SUMMARY}"
    echo "log=${MASTER_LOG}"
} | tee -a "${MASTER_LOG}"

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi
