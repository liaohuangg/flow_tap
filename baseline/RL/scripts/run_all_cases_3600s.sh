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

TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-3600}"
EPISODES="${EPISODES:-1000000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-480}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
GRID_RESOLUTION="${GRID_RESOLUTION:-auto}"
EXACT_ACTION_SLOTS="${EXACT_ACTION_SLOTS:-50000}"
THERMAL_INTP_SIZE="${THERMAL_INTP_SIZE:-100}"
RUN_PREFIX="${RUN_PREFIX:-allcases_3600s_$(date +%Y%m%d_%H%M%S)}"
FORCE_THERMAL_TABLES="${FORCE_THERMAL_TABLES:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export MPLCONFIGDIR

mkdir -p runs
mkdir -p "${MPLCONFIGDIR}"

BATCH_DIR="${RL_DIR}/runs/${RUN_PREFIX}"
BATCH_LOG="${BATCH_DIR}/batch.log"
SUMMARY="${BATCH_DIR}/summary.tsv"


{
    echo "batch_name=${RUN_PREFIX}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "python=${PYTHON_BIN}"
    echo "time_limit_seconds=${TIME_LIMIT_SECONDS}"
    echo "episodes=${EPISODES}"
    echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
    echo "log_interval=${LOG_INTERVAL}"
    echo "save_interval=${SAVE_INTERVAL}"
    echo "grid_resolution=${GRID_RESOLUTION}"
    echo "exact_action_slots=${EXACT_ACTION_SLOTS}"
    echo "thermal_intp_size=${THERMAL_INTP_SIZE}"
    echo "force_thermal_tables=${FORCE_THERMAL_TABLES}"
    echo "case_names=${CASE_NAMES:-ALL}"
    echo "mplconfigdir=${MPLCONFIGDIR}"
    echo "batch_dir=${BATCH_DIR}"
    echo
} | tee "${BATCH_LOG}"

printf "case\trun_name\tstatus\texit_code\tseconds\trun_dir\ttrain_log\n" > "${SUMMARY}"

if [[ -n "${CASE_NAMES:-}" ]]; then
    CASE_FILES=()
    for case_name in ${CASE_NAMES//,/ }; do
        case_name="${case_name%.json}"
        CASE_FILES+=("examples/${case_name}.json")
    done
else
    # Default order: run cases from xerox8_m down to acend910 in reverse
    # lexicographic order, skipping schema-only files such as EMIB.json later.
    mapfile -t CASE_FILES < <(
        find examples -maxdepth 1 -type f -name "*.json" \
            | sort -r \
            | awk 'BEGIN{keep=0} /examples\/xerox8_m\.json$/{keep=1} keep{print} /examples\/acend910\.json$/{exit}'
    )
fi

if [[ "${#CASE_FILES[@]}" -eq 0 ]]; then
    echo "No cases found under ${RL_DIR}/examples" | tee -a "${BATCH_LOG}"
    exit 1
fi

total="${#CASE_FILES[@]}"
index=0
failed=0

for case_json in "${CASE_FILES[@]}"; do
    index=$((index + 1))
    case_name="$(basename "${case_json}" .json)"
    run_name="${RUN_PREFIX}_${case_name}"
    run_dir="${RL_DIR}/runs/${run_name}"
    train_log="${run_dir}/train.log"

    if [[ ! -f "${case_json}" ]]; then
        echo "[${index}/${total}] case=${case_name} skipped: file not found: ${case_json}" | tee -a "${BATCH_LOG}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${run_name}" "skipped_missing" "0" "0" "${run_dir}" "${train_log}" \
            >> "${SUMMARY}"
        continue
    fi

    if ! "${PYTHON_BIN}" - "${case_json}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
sys.exit(0 if (data.get("chiplets") or data.get("dies")) else 1)
PY
    then
        echo "[${index}/${total}] case=${case_name} skipped: JSON has no chiplets/dies schema" | tee -a "${BATCH_LOG}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${run_name}" "skipped_schema" "0" "0" "${run_dir}" "${train_log}" \
            >> "${SUMMARY}"
        continue
    fi

    cmd=(
        "${PYTHON_BIN}" train.py
        --name "${run_name}"
        --json "${case_json}"
        --num_episodes "${EPISODES}"
        --rollout_batch_size "${ROLLOUT_BATCH_SIZE}"
        --save_interval "${SAVE_INTERVAL}"
        --no_checkpoint
        --log_interval "${LOG_INTERVAL}"
        --grid_resolution "${GRID_RESOLUTION}"
        --exact_action_slots "${EXACT_ACTION_SLOTS}"
        --thermal_intp_size "${THERMAL_INTP_SIZE}"
    )

    if [[ "${FORCE_THERMAL_TABLES}" == "1" ]]; then
        cmd+=(--force_thermal_tables)
    fi

    echo "======================================================================" | tee -a "${BATCH_LOG}"
    echo "[${index}/${total}] case=${case_name} run=${run_name}" | tee -a "${BATCH_LOG}"
    echo "timeout=${TIME_LIMIT_SECONDS}s" | tee -a "${BATCH_LOG}"
    echo "command: ${cmd[*]}" | tee -a "${BATCH_LOG}"
    start_ts="$(date +%s)"

    timeout --preserve-status --signal=INT --kill-after=30s "${TIME_LIMIT_SECONDS}s" "${cmd[@]}" 2>&1 | tee -a "${BATCH_LOG}"
    exit_code="${PIPESTATUS[0]}"
    end_ts="$(date +%s)"
    seconds=$((end_ts - start_ts))

    if [[ "${exit_code}" -eq 0 ]]; then
        status="ok"
    elif [[ "${exit_code}" -eq 124 || "${exit_code}" -eq 130 || "${exit_code}" -eq 143 ]]; then
        status="timeout"
    else
        status="failed"
        failed=$((failed + 1))
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${case_name}" "${run_name}" "${status}" "${exit_code}" "${seconds}" "${run_dir}" "${train_log}" \
        >> "${SUMMARY}"

    echo "[${index}/${total}] status=${status} exit_code=${exit_code} seconds=${seconds}" | tee -a "${BATCH_LOG}"
done

{
    echo
    echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "summary=${SUMMARY}"
    echo "batch_log=${BATCH_LOG}"
    echo "failed=${failed}"
} | tee -a "${BATCH_LOG}"

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi
