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

NUM_EPOCHS="${NUM_EPOCHS:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS:-4}"
GRID_RESOLUTION="${GRID_RESOLUTION:-auto}"
EXACT_ACTION_SLOTS="${EXACT_ACTION_SLOTS:-50000}"
MAX_AUTO_GRID_RESOLUTION="${MAX_AUTO_GRID_RESOLUTION:-100}"
PLACEMENT_AREA_WEIGHT="${PLACEMENT_AREA_WEIGHT:-1.0}"
PLACEMENT_PIN_WEIGHT="${PLACEMENT_PIN_WEIGHT:-1.0}"
THERMAL_INTP_SIZE="${THERMAL_INTP_SIZE:-}"
FORCE_THERMAL_TABLES="${FORCE_THERMAL_TABLES:-0}"
RUN_PREFIX="${RUN_PREFIX:-all_examples_2epochs_$(date +%Y%m%d_%H%M%S)}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export MPLCONFIGDIR

mkdir -p runs "${MPLCONFIGDIR}"
BATCH_DIR="${RL_DIR}/runs/${RUN_PREFIX}"
mkdir -p "${BATCH_DIR}"
BATCH_LOG="${BATCH_DIR}/batch.log"
SUMMARY="${BATCH_DIR}/summary.tsv"

{
    echo "batch_name=${RUN_PREFIX}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "python=${PYTHON_BIN}"
    echo "num_epochs=${NUM_EPOCHS}"
    echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
    echo "num_minibatches=${NUM_MINIBATCHES}"
    echo "ppo_update_epochs=${PPO_UPDATE_EPOCHS}"
    echo "grid_resolution=${GRID_RESOLUTION}"
    echo "max_auto_grid_resolution=${MAX_AUTO_GRID_RESOLUTION}"
    echo "placement_area_weight=${PLACEMENT_AREA_WEIGHT}"
    echo "placement_pin_weight=${PLACEMENT_PIN_WEIGHT}"
    echo "thermal_intp_size=${THERMAL_INTP_SIZE:-default}"
    echo "force_thermal_tables=${FORCE_THERMAL_TABLES}"
    echo "case_names=${CASE_NAMES:-ALL}"
    echo "batch_dir=${BATCH_DIR}"
    echo
} | tee "${BATCH_LOG}"

echo "Checking PyTorch and CPLEX..." | tee -a "${BATCH_LOG}"
if ! "${PYTHON_BIN}" -c \
    'import torch; from fastTM.routing import _load_cplex; cplex = _load_cplex(); print(f"torch={torch.__version__}, cplex={cplex.Cplex().get_version()}")' \
    2>&1 | tee -a "${BATCH_LOG}"
then
    echo "Preflight failed: PyTorch or the CPLEX Python API is unavailable." | tee -a "${BATCH_LOG}"
    exit 2
fi

printf "case\trun_name\tstatus\texit_code\tseconds\trun_dir\ttrain_log\n" > "${SUMMARY}"

if [[ -n "${CASE_NAMES:-}" ]]; then
    CASE_FILES=()
    for case_name in ${CASE_NAMES//,/ }; do
        case_name="${case_name%.json}"
        CASE_FILES+=("examples/${case_name}.json")
    done
else
    mapfile -t CASE_FILES < <(find examples -maxdepth 1 -type f -name "*.json" | sort)
fi

if [[ "${#CASE_FILES[@]}" -eq 0 ]]; then
    echo "No JSON files found under ${RL_DIR}/examples." | tee -a "${BATCH_LOG}"
    exit 1
fi

total="${#CASE_FILES[@]}"
index=0
passed=0
failed=0
skipped=0

for case_json in "${CASE_FILES[@]}"; do
    index=$((index + 1))
    case_name="$(basename "${case_json}" .json)"
    run_name="${RUN_PREFIX}_${case_name}"
    run_dir="${RL_DIR}/runs/${run_name}"
    train_log="${run_dir}/train.log"

    if [[ ! -f "${case_json}" ]]; then
        status="skipped_missing"
        skipped=$((skipped + 1))
        echo "[${index}/${total}] ${case_name}: ${status}" | tee -a "${BATCH_LOG}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${run_name}" "${status}" "0" "0" "${run_dir}" "${train_log}" \
            >> "${SUMMARY}"
        continue
    fi

    if ! "${PYTHON_BIN}" - "${case_json}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    data = json.load(stream)
raise SystemExit(0 if (data.get("chiplets") or data.get("dies")) else 1)
PY
    then
        status="skipped_non_layout"
        skipped=$((skipped + 1))
        echo "[${index}/${total}] ${case_name}: ${status} (no chiplets/dies)" | tee -a "${BATCH_LOG}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${run_name}" "${status}" "0" "0" "${run_dir}" "${train_log}" \
            >> "${SUMMARY}"
        continue
    fi

    cmd=(
        "${PYTHON_BIN}" train.py
        --name "${run_name}"
        --json "${case_json}"
        --num_epochs "${NUM_EPOCHS}"
        --rollout_batch_size "${ROLLOUT_BATCH_SIZE}"
        --num_minibatches "${NUM_MINIBATCHES}"
        --ppo_update_epochs "${PPO_UPDATE_EPOCHS}"
        --save_interval "${NUM_EPOCHS}"
        --no_checkpoint
        --log_interval 1
        --grid_resolution "${GRID_RESOLUTION}"
        --exact_action_slots "${EXACT_ACTION_SLOTS}"
        --max_auto_grid_resolution "${MAX_AUTO_GRID_RESOLUTION}"
        --placement_area_weight "${PLACEMENT_AREA_WEIGHT}"
        --placement_pin_weight "${PLACEMENT_PIN_WEIGHT}"
    )

    if [[ -n "${THERMAL_INTP_SIZE}" ]]; then
        cmd+=(--thermal_intp_size "${THERMAL_INTP_SIZE}")
    fi
    if [[ "${FORCE_THERMAL_TABLES}" == "1" ]]; then
        cmd+=(--force_thermal_tables)
    fi

    echo "======================================================================" | tee -a "${BATCH_LOG}"
    echo "[${index}/${total}] case=${case_name}, run=${run_name}" | tee -a "${BATCH_LOG}"
    echo "command: ${cmd[*]}" | tee -a "${BATCH_LOG}"
    start_ts="$(date +%s)"

    "${cmd[@]}" 2>&1 | tee -a "${BATCH_LOG}"
    exit_code="${PIPESTATUS[0]}"
    seconds=$(( $(date +%s) - start_ts ))

    if [[ "${exit_code}" -eq 0 ]]; then
        status="ok"
        passed=$((passed + 1))
    else
        status="failed"
        failed=$((failed + 1))
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${case_name}" "${run_name}" "${status}" "${exit_code}" "${seconds}" "${run_dir}" "${train_log}" \
        >> "${SUMMARY}"
    echo "[${index}/${total}] status=${status}, exit_code=${exit_code}, seconds=${seconds}" | tee -a "${BATCH_LOG}"
done

{
    echo "======================================================================"
    echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "passed=${passed}"
    echo "failed=${failed}"
    echo "skipped=${skipped}"
    echo "summary=${SUMMARY}"
    echo "batch_log=${BATCH_LOG}"
} | tee -a "${BATCH_LOG}"

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi
