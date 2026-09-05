#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_cases.sh <json_root_dir> [out_root] [model_type]
#
# Example:
#   ./run_cases.sh /root/placement/flow_GCN/benchmark/placement/base/seed_13012/placement \
#     /root/placement/flow_GCN/tap2.5d_hoteval/config grid

JSON_ROOT_DIR="${1:-}"
OUT_ROOT="${2:-}"
MODEL_TYPE="${3:-grid}"

if [[ -z "${JSON_ROOT_DIR}" ]]; then
  echo "Usage: $0 <json_root_dir> [out_root] [model_type]" >&2
  exit 2
fi

if [[ ! -d "${JSON_ROOT_DIR}" ]]; then
  echo "[error] json_root_dir is not a directory: ${JSON_ROOT_DIR}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_HOTSPOT_PY="${SCRIPT_DIR}/run_hotspot.py"

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="${SCRIPT_DIR}/config"
fi

if [[ ! -f "${RUN_HOTSPOT_PY}" ]]; then
  echo "[error] run_hotspot.py not found: ${RUN_HOTSPOT_PY}" >&2
  exit 2
fi

shopt -s nullglob

# Find json files under the root directory (recursively) and run each case.
mapfile -t JSON_FILES < <(find "${JSON_ROOT_DIR}" -type f -name "*.json" | sort)

if [[ ${#JSON_FILES[@]} -eq 0 ]]; then
  echo "[warn] no .json files found under: ${JSON_ROOT_DIR}" >&2
  exit 0
fi

echo "[info] json_root_dir: ${JSON_ROOT_DIR}"
echo "[info] out_root:     ${OUT_ROOT}"
echo "[info] model_type:   ${MODEL_TYPE}"

for json_path in "${JSON_FILES[@]}"; do
  case_name="$(basename "${json_path}" .json)"
  echo "\n[case] ${case_name}"
  python "${RUN_HOTSPOT_PY}" \
    --json "${json_path}" \
    --out_root "${OUT_ROOT}" \
    --case "${case_name}" \
    --model_type "${MODEL_TYPE}"
done

 #./run_cases.sh /root/placement/flow_GCN/benchmark/placement/final-thermal/placement /root/placement/flow_GCN/resultEval/config_thermal_529 grid