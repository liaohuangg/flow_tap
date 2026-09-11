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

EPISODES="${EPISODES:-10}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
GRID_RESOLUTION="${GRID_RESOLUTION:-auto}"
EXACT_ACTION_SLOTS="${EXACT_ACTION_SLOTS:-50000}"
RUN_PREFIX="${RUN_PREFIX:-all_cases_thermal_${EPISODES}_$(date +%Y%m%d_%H%M%S)}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export MPLCONFIGDIR

mkdir -p runs "${MPLCONFIGDIR}"
BATCH_LOG="runs/${RUN_PREFIX}_batch.log"
SUMMARY="runs/${RUN_PREFIX}_summary.tsv"

{
    echo "batch_name=${RUN_PREFIX}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "python=${PYTHON_BIN}"
    echo "episodes=${EPISODES}"
    echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
    echo "log_interval=${LOG_INTERVAL}"
    echo "save_interval=${SAVE_INTERVAL}"
    echo "grid_resolution=${GRID_RESOLUTION}"
    echo "exact_action_slots=${EXACT_ACTION_SLOTS}"
    echo "case_names=${CASE_NAMES:-ALL}"
    echo
} | tee "${BATCH_LOG}"

printf "case\trun_name\tstatus\texit_code\tseconds\trun_dir\texport_kind\texport_json\n" > "${SUMMARY}"

if [[ -n "${CASE_NAMES:-}" ]]; then
    CASE_FILES=()
    for case_name in ${CASE_NAMES//,/ }; do
        case_name="${case_name%.json}"
        CASE_FILES+=("examples/${case_name}.json")
    done
else
    mapfile -t CASE_FILES < <(find examples -maxdepth 1 -type f -name "*.json" | sort)
fi

failed=0
total="${#CASE_FILES[@]}"
index=0

for case_json in "${CASE_FILES[@]}"; do
    index=$((index + 1))
    case_name="$(basename "${case_json}" .json)"
    run_name="${RUN_PREFIX}_${case_name}"
    run_dir="${RL_DIR}/runs/${run_name}"

    if [[ ! -f "${case_json}" ]]; then
        echo "[${index}/${total}] case=${case_name} skipped: missing ${case_json}" | tee -a "${BATCH_LOG}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${run_name}" "skipped_missing" "0" "0" "${run_dir}" "" "" >> "${SUMMARY}"
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
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${run_name}" "skipped_schema" "0" "0" "${run_dir}" "" "" >> "${SUMMARY}"
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
    )

    echo "======================================================================" | tee -a "${BATCH_LOG}"
    echo "[${index}/${total}] case=${case_name} run=${run_name}" | tee -a "${BATCH_LOG}"
    echo "command: ${cmd[*]}" | tee -a "${BATCH_LOG}"
    start_ts="$(date +%s)"

    "${cmd[@]}" 2>&1 | tee -a "${BATCH_LOG}"
    exit_code="${PIPESTATUS[0]}"
    end_ts="$(date +%s)"
    seconds=$((end_ts - start_ts))

    read -r export_kind export_json < <(
        "${PYTHON_BIN}" - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
best = run_dir / "best_summary.json"
illegal = run_dir / "illegal" / "illegal_summary.json"
if best.exists():
    data = json.load(open(best, encoding="utf-8"))
    print("legal", data.get("json", ""))
elif illegal.exists():
    data = json.load(open(illegal, encoding="utf-8"))
    print("illegal", data.get("json", ""))
else:
    print("none", "")
PY
    )

    if [[ "${exit_code}" -eq 0 ]]; then
        status="ok"
    else
        status="failed"
        failed=$((failed + 1))
    fi
    if [[ "${export_kind}" == "none" ]]; then
        status="failed_no_export"
        failed=$((failed + 1))
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${case_name}" "${run_name}" "${status}" "${exit_code}" "${seconds}" "${run_dir}" "${export_kind}" "${export_json}" \
        >> "${SUMMARY}"

    echo "[${index}/${total}] status=${status} export=${export_kind} seconds=${seconds}" | tee -a "${BATCH_LOG}"
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
