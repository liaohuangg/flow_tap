#!/usr/bin/env bash
# run_3cases_par5.sh — hp6_m / xerox6_m / xerox7_m 的 RL 训练, 最多 JOBS 个并发。
#
# 求解参数与 scripts/run_8cases_5seeds_1h.sh 逐项一致 (600 epochs / 3600s 预算 /
# --force_thermal_tables / --save_checkpoint), 只是把 (case, seed) 展平成任务池,
# 并发跑。run 目录命名沿用今天 par3 那批的家族:
#
#   runs/<TAG>_<case>_seed<k>_<case>/
#
# 之所以不直接并发跑 run_8cases_5seeds_1h.sh 的多个实例: 那个脚本开头会把
# runs/${RUN_PREFIX}/summary.tsv 截断重建, 两个实例共用 RUN_PREFIX 就会互相盖掉。
#
# 用法:
#   JOBS=5 bash run_3cases_par5.sh
#   SEEDS_OVERRIDE="2" CASES_OVERRIDE="hp6_m" DRY_RUN=1 bash run_3cases_par5.sh
set -uo pipefail

RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${RL_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/chipdiffusion/bin/python}"
CASE_LIST="${CASE_LIST:-hp6_m xerox6_m xerox7_m}"
SEED_LIST="${SEED_LIST:-2 3 4}"
JOBS="${JOBS:-5}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a CASES <<< "${CASE_LIST//,/ }"
read -r -a SEEDS <<< "${SEED_LIST//,/ }"
if [ -n "${CASES_OVERRIDE:-}" ]; then read -r -a CASES <<< "${CASES_OVERRIDE//,/ }"; fi
if [ -n "${SEEDS_OVERRIDE:-}" ]; then read -r -a SEEDS <<< "${SEEDS_OVERRIDE//,/ }"; fi

TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-3600}"
HARD_TIMEOUT_GRACE_SECONDS="${HARD_TIMEOUT_GRACE_SECONDS:-600}"
BATCH_TAG="${BATCH_TAG:-rl_3cases_5seeds_1h_par${JOBS}_$(date +%Y%m%d_%H%M%S)}"
BATCH_DIR="${RL_DIR}/runs/${BATCH_TAG}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${BATCH_DIR}" "${MPLCONFIGDIR}"

LOG="${BATCH_DIR}/experiment.log"
SUMMARY="${BATCH_DIR}/summary.tsv"
HEADER="case\tseed\tstatus\texit_code\twall_seconds\trun_dir"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

{
    echo "batch=${BATCH_TAG}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "python=${PYTHON_BIN}"
    echo "cases=${CASES[*]}"
    echo "seeds=${SEEDS[*]}"
    echo "jobs=${JOBS}"
    echo "time_limit_seconds=${TIME_LIMIT_SECONDS}"
    echo "thermal_reuse_across_seeds=disabled (--force_thermal_tables)"
    echo "batch_dir=${BATCH_DIR}"
    echo
} | tee "${LOG}"

printf "%b\n" "${HEADER}" > "${SUMMARY}"

if [ "${DRY_RUN}" = "1" ]; then
    for case_name in "${CASES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "would run: ${case_name} seed=${seed} -> ${BATCH_DIR}_${case_name}_seed${seed}_${case_name}"
        done
    done
    exit 0
fi

log "Checking PyTorch and CPLEX..."
if ! "${PYTHON_BIN}" -c \
    'import torch; from fastTM.routing import _load_cplex; cplex = _load_cplex(); print(f"torch={torch.__version__}, cplex={cplex.Cplex().get_version()}")' \
    2>&1 | tee -a "${LOG}"
then
    log "Preflight failed: PyTorch or CPLEX is unavailable."
    exit 2
fi

run_one() {
    local case_name="$1" seed="$2"
    local run_name="${BATCH_TAG}_${case_name}_seed${seed}_${case_name}"
    local run_dir="${RL_DIR}/runs/${run_name}"
    local checkpoint="${run_dir}/checkpoints/ppo_best.pt"
    local console="${BATCH_DIR}/${case_name}_seed${seed}.log"

    local cmd=(
        "${PYTHON_BIN}" train.py
        --name "${run_name}"
        --json "examples/${case_name}.json"
        --num_epochs "${MAX_EPOCHS:-600}"
        --rollout_batch_size "${ROLLOUT_BATCH_SIZE:-480}"
        --num_minibatches "${NUM_MINIBATCHES:-4}"
        --ppo_update_epochs "${PPO_UPDATE_EPOCHS:-4}"
        --log_interval "${LOG_INTERVAL:-1}"
        --grid_resolution "${GRID_RESOLUTION:-auto}"
        --max_auto_grid_resolution "${MAX_AUTO_GRID_RESOLUTION:-100}"
        --exact_action_slots "${EXACT_ACTION_SLOTS:-50000}"
        --thermal_intp_size "${THERMAL_INTP_SIZE:-50}"
        --placement_area_weight "${PLACEMENT_AREA_WEIGHT:-1.0}"
        --placement_pin_weight "${PLACEMENT_PIN_WEIGHT:-1.0}"
        --seed "${seed}"
        --time_limit_seconds "${TIME_LIMIT_SECONDS}"
        --force_thermal_tables
        --save_checkpoint
    )

    local t0; t0=$(date +%s)
    log "START case=${case_name} seed=${seed}"
    timeout --preserve-status --signal=INT --kill-after=60s \
        "$((TIME_LIMIT_SECONDS + HARD_TIMEOUT_GRACE_SECONDS))s" \
        "${cmd[@]}" > "${console}" 2>&1
    local rc=$?
    local wall=$(( $(date +%s) - t0 ))

    local status="failed"
    if [ "${rc}" -eq 0 ] && [ -f "${checkpoint}" ]; then
        status="ok"
    elif [ "${rc}" -eq 124 ] || [ "${rc}" -eq 130 ] || [ "${rc}" -eq 143 ]; then
        status="hard_timeout"
    fi
    log "DONE  case=${case_name} seed=${seed} status=${status} rc=${rc} wall=${wall}s"
    printf "%b\n" "${case_name}\t${seed}\t${status}\t${rc}\t${wall}\t${run_dir}" >> "${SUMMARY}"
}

for case_name in "${CASES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do sleep 10; done
        run_one "${case_name}" "${seed}" &
    done
done
wait

log "=== 全部结束 ==="
cat "${SUMMARY}" | tee -a "${LOG}"
