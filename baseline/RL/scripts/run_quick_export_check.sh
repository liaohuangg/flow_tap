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
RUN_PREFIX="${RUN_PREFIX:-quick_export_${EPISODES}_$(date +%Y%m%d_%H%M%S)}"
CASE_NAMES="${CASE_NAMES:-xerox8_m syn6 sys_micro150}"
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
    echo "case_names=${CASE_NAMES}"
    echo
} | tee "${BATCH_LOG}"

printf "case\trun_name\tstatus\texit_code\trun_dir\texport_kind\texport_json\tthermal_tables\ttemperature\tthermal_error\n" > "${SUMMARY}"

failed=0
for case_name in ${CASE_NAMES//,/ }; do
    case_name="${case_name%.json}"
    case_json="examples/${case_name}.json"
    run_name="${RUN_PREFIX}_${case_name}"
    run_dir="${RL_DIR}/runs/${run_name}"

    if [[ ! -f "${case_json}" ]]; then
        echo "case=${case_name} missing: ${case_json}" | tee -a "${BATCH_LOG}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${run_name}" "missing" "0" "${run_dir}" "" "" "" "" "" >> "${SUMMARY}"
        failed=$((failed + 1))
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
    echo "case=${case_name} run=${run_name}" | tee -a "${BATCH_LOG}"
    echo "command: ${cmd[*]}" | tee -a "${BATCH_LOG}"
    "${cmd[@]}" 2>&1 | tee -a "${BATCH_LOG}"
    exit_code="${PIPESTATUS[0]}"

    read -r export_kind export_json thermal_tables temperature thermal_error < <(
        "${PYTHON_BIN}" - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
best = run_dir / "best_summary.json"
illegal = run_dir / "illegal" / "illegal_summary.json"
timing = run_dir / "timing.json"

export_kind = "none"
export_json = ""
temperature = ""
thermal_error = ""
if best.exists():
    data = json.load(open(best, encoding="utf-8"))
    export_kind = "legal"
    export_json = data.get("json", "")
    metrics = data.get("metrics", {})
    temperature = metrics.get("rlplanner_temperature", "")
    thermal_error = metrics.get("rlplanner_reward_error", "")
elif illegal.exists():
    data = json.load(open(illegal, encoding="utf-8"))
    export_kind = "illegal"
    export_json = data.get("json", "")
    metrics = data.get("metrics", {})
    temperature = metrics.get("rlplanner_temperature", "")
    thermal_error = metrics.get("rlplanner_reward_error", "")

thermal_tables = ""
if timing.exists():
    timing_data = json.load(open(timing, encoding="utf-8"))
    thermal_tables = (timing_data.get("thermal_tables") or {}).get("status", "")

print(export_kind, export_json, thermal_tables, temperature, thermal_error)
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
    if [[ ! -f "${run_dir}/train.log" ]]; then
        status="failed_no_train_log"
        failed=$((failed + 1))
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${case_name}" "${run_name}" "${status}" "${exit_code}" "${run_dir}" \
        "${export_kind}" "${export_json}" "${thermal_tables}" "${temperature}" "${thermal_error}" \
        >> "${SUMMARY}"
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
