#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHIPLETFM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLOW_GCN_ROOT="$(cd "${CHIPLETFM_ROOT}/.." && pwd)"

cd "${CHIPLETFM_ROOT}"
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" && -n "${CONDA_ENV:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH=".:diffusion:.."
export MPLCONFIGDIR=/tmp/mplconfig

SCRIPT_PATH="${SCRIPT_DIR}/run_case1_10.sh"
CONFIG="${CONFIG:-${SCRIPT_DIR}/run_case1_10_canvas_config.json}"
METHOD="${METHOD:-63-case1-10}"
SEED="${SEED:-13012}"
TASK="${TASK:-atplace_case1_10}"
DST_DATASET="${CHIPLETFM_ROOT}/benckmark/${TASK}"
MODEL_CKPT="${MODEL_CKPT:-${CHIPLETFM_ROOT}/checkpoints/model/best.ckpt}"
THERMAL_CKPT="${THERMAL_CKPT:-${FLOW_GCN_ROOT}/thermalmodel/checkpoints/fp32/fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0_20260507_171818/hrnet_fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0_ep0175_seed0_bs32_lr2e-04_base96_gw0p1_s4_b2_er2_tr0_va0.pth}"

DEFAULT_CASE_SIZES_JSON='{"Case1":[55,55],"Case2":[48,48],"Case3":[45,45],"Case4":[70,70],"Case5":[75,75],"Case6":[50,50],"Case7":[80,80],"Case8":[48,50],"Case9":[105,75],"Case10":[105,80]}'
CASE_SIZES_JSON="${DEFAULT_CASE_SIZES_JSON}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_HOTSPOT="${SKIP_HOTSPOT:-0}"
read -r -a EVAL_EXTRA_ARGS <<< "${EVAL_EXTRA_ARGS:-}"

if [[ "${RUN_CASE1_10_SINGLE:-0}" != "1" && -f "${CONFIG}" ]]; then
  echo "[batch] config=${CONFIG}"
  python - "${CONFIG}" <<'PY' | while IFS=$'\t' read -r method seed; do
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
default_seed = cfg.get("seed", 13012)

experiments = list(cfg.get("experiments", []))
seed_sweep = cfg.get("seed_sweep")
if isinstance(seed_sweep, dict):
    method_prefix = seed_sweep.get("method_prefix", "63-case1-10-seed")
    seeds = seed_sweep.get("seeds", [])
    if not isinstance(seeds, list) or not seeds:
        raise SystemExit("seed_sweep.seeds must be a non-empty list")
    for seed in seeds:
        experiments.append({
            "method": f"{method_prefix}-{int(seed)}",
            "seed": int(seed),
        })

for i, exp in enumerate(experiments, start=1):
    method = exp.get("method") or f"63-case1-10-{i:03d}"
    seed = exp.get("seed", default_seed)
    print(f"{method}\t{seed}")
PY
    echo
    echo "[batch] running method=${method} seed=${seed}"
    RUN_CASE1_10_SINGLE=1 \
      METHOD="${method}" \
      SEED="${seed}" \
      TASK="${TASK}" \
      CASE_SIZES_JSON="${CASE_SIZES_JSON}" \
      SKIP_EVAL="${SKIP_EVAL}" \
      SKIP_HOTSPOT="${SKIP_HOTSPOT}" \
      EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS[*]}" \
      bash "${SCRIPT_PATH}"
  done
  exit 0
fi

CASE_SIZES_JSON="${CASE_SIZES_JSON}" python - "${DST_DATASET}" <<'PY'
import json
import os
import pickle
import sys
from pathlib import Path

import torch

dst = Path(sys.argv[1])
if not dst.is_dir():
    raise SystemExit(f"[dataset][ERROR] dataset directory not found: {dst}")
for required in ("config.yaml", "index_map.json"):
    if not (dst / required).is_file():
        raise SystemExit(f"[dataset][ERROR] missing required file: {dst / required}")

sizes_text = os.environ.get("CASE_SIZES_JSON", "{}").strip()
sizes, end = json.JSONDecoder().raw_decode(sizes_text)
if sizes_text[end:].strip():
    print(f"[canvas][WARN] ignored trailing CASE_SIZES_JSON text: {sizes_text[end:].strip()!r}")
if not isinstance(sizes, dict) or not sizes:
    raise SystemExit("[canvas][ERROR] CASE_SIZES_JSON must be a non-empty object")
index_map = json.loads((dst / "index_map.json").read_text(encoding="utf-8"))
record = {}
missing_cases = []
missing_files = []

for item in index_map:
    idx = int(item["dataset_index"])
    case = str(item["benchmark_name"])
    graph_path = dst / f"graph{idx}.pickle"
    output_path = dst / f"output{idx}.pickle"
    if not graph_path.is_file():
        missing_files.append(str(graph_path))
        continue
    if not output_path.is_file():
        missing_files.append(str(output_path))
        continue
    with graph_path.open("rb") as f:
        graph = pickle.load(f)

    old = graph.chip_size.detach().cpu().view(-1).tolist() if hasattr(graph.chip_size, "detach") else list(graph.chip_size)
    old_width = float(old[2] - old[0]) if len(old) == 4 else float(old[0])
    old_height = float(old[3] - old[1]) if len(old) == 4 else float(old[1])

    if case not in sizes:
        missing_cases.append(case)
        continue
    width, height = sizes[case]
    width = float(width)
    height = float(height)

    new = [0.0, 0.0, width, height]
    changed = abs(old_width - width) > 1e-6 or abs(old_height - height) > 1e-6
    if changed:
        graph.chip_size = torch.tensor(new, dtype=torch.float32)
        with graph_path.open("wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    record[case] = {
        "old_chip_size": old,
        "new_chip_size": new,
        "width": width,
        "height": height,
        "changed": changed,
    }

if missing_files:
    raise SystemExit("[dataset][ERROR] missing required dataset files:\n" + "\n".join(missing_files))
if missing_cases:
    raise SystemExit(f"[canvas][ERROR] missing canvas sizes for cases: {', '.join(missing_cases)}")

for item in index_map:
    idx = int(item["dataset_index"])
    case = str(item["benchmark_name"])
    graph_path = dst / f"graph{idx}.pickle"
    with graph_path.open("rb") as f:
        graph = pickle.load(f)
    actual = graph.chip_size.detach().cpu().view(-1).tolist() if hasattr(graph.chip_size, "detach") else list(graph.chip_size)
    actual_width = float(actual[2] - actual[0]) if len(actual) == 4 else float(actual[0])
    actual_height = float(actual[3] - actual[1]) if len(actual) == 4 else float(actual[1])
    expected_width = float(sizes[case][0])
    expected_height = float(sizes[case][1])
    if abs(actual_width - expected_width) > 1e-6 or abs(actual_height - expected_height) > 1e-6:
        raise SystemExit(
            f"[canvas][ERROR] verification failed for {case}: "
            f"actual {actual_width} x {actual_height}, expected {expected_width} x {expected_height}"
        )
PY

if [[ "${SKIP_EVAL}" != "1" ]]; then
  python diffusion/eval_thermal_guided.py \
    --config-name config_eval_fm \
    "model=geometry-att-gnn" \
    "task=${TASK}" \
    "method=${METHOD}" \
    "seed=${SEED}" \
    logger.wandb=False \
    "from_checkpoint=${MODEL_CKPT}" \
    eval_samples=0 \
    num_output_samples=10 \
    val_batch_size=16 \
    model.max_diffusion_steps=100 \
    model.guidance_mode=sgd \
    model.legality_guidance_weight=15.0 \
    model.hpwl_guidance_weight=0.1 \
    model.bbox_guidance_weight=0.0 \
    model.heat_repulsion_guidance_weight=0.5 \
    model.heat_repulsion_sigma=0.45 \
    model.heat_repulsion_power_key=node_power \
    model.heat_repulsion_normalize_power=graph_max \
    model.grad_descent_steps=16 \
    model.grad_descent_rate=0.035 \
    model.legality_softmax_factor_min=10.0 \
    model.legality_softmax_factor_max=10.0 \
    model.legality_softmax_critical_factor=0.0 \
    model.guidance_schedule.enabled=True \
    model.guidance_schedule.legality_only_until=0.5 \
    model.guidance_schedule.joint_until=0.9 \
    model.guidance_schedule.legality_initial_weight=15.0 \
    model.guidance_schedule.legality_mid_weight=15.0 \
    model.guidance_schedule.legality_final_weight=15.0 \
    model.guidance_schedule.hpwl_initial_weight=0.0 \
    model.guidance_schedule.hpwl_final_weight=5.0e-05 \
    model.guidance_schedule.bbox_initial_weight=0.0 \
    model.guidance_schedule.bbox_final_weight=0.0 \
    model.guidance_schedule.heat_repulsion_initial_weight=0.0 \
    model.guidance_schedule.heat_repulsion_final_weight=0.5 \
    model.backbone_params.edge_features=8 \
    model.backbone_params.hidden_size=256 \
    "model.backbone_params.hidden_node_features=[256,256,256]" \
    "model.backbone_params.attention_node_features=[256,256,256]" \
    "model.backbone_params.extra_node_feature_keys=[node_power]" \
    model.backbone_params.extra_node_feature_normalize=graph_max \
    model.backbone_params.thermal_mp_enabled=True \
    model.backbone_params.thermal_mp_power_key=node_power \
    model.backbone_params.thermal_mp_sigma=0.35 \
    model.backbone_params.thermal_mp_topk=0 \
    model.backbone_params.thermal_mp_normalize_power=graph_max \
    model.backbone_params.geometry_attention_enabled=True \
    model.backbone_params.geometry_attention_layers=0 \
    model.backbone_params.geometry_attention_heads=4 \
    model.backbone_params.geometry_attention_power_key=node_power \
    model.backbone_params.geometry_attention_sigma=0.35 \
    model.backbone_params.geometry_attention_normalize_power=graph_max \
    model.backbone_params.auxiliary_legality_heads_enabled=True \
    model.backbone_params.auxiliary_legality_head_layers=2 \
    legalization.mode=standard \
    legalization.softmax_min=5.0 \
    legalization.softmax_max=180.0 \
    legalization.step_size=0.035 \
    legalization.grad_descent_steps=600 \
    legalization.legality_weight=35.0 \
    legalization.hpwl_weight=1.0e-07 \
    legalization.bbox_weight=1.0e-05 \
    legalization.bbox_start_factor=0.2 \
    legalization.bbox_end_factor=0.5 \
    legalization.bbox_increase_factor=1.0 \
    legalization.bbox_softmax_beta=30.0 \
    legalization.softmax_critical_factor=1.0 \
    legalization.guidance_critical_factor=1.0 \
    legalization.zero_hpwl_factor=1.0 \
    legalization.legality_increase_factor=6.0 \
    +legalization.thermal_weight=0.05 \
    "thermal.ckpt=${THERMAL_CKPT}" \
    thermal.report_guidance_enabled=True \
    thermal.guidance_weight=0.1 \
    thermal.guidance_lr=0.01 \
    thermal.guidance_steps=4 \
    thermal.legality_weight=5.0 \
    thermal.grad_clip=0.03 \
    thermal.schedule.enabled=True \
    thermal.schedule.start=0.3 \
    thermal.schedule.full=0.8 \
    thermal.schedule.initial_weight=0.0 \
    thermal.schedule.final_weight=0.5 \
    "${EVAL_EXTRA_ARGS[@]}"
else
  echo "[eval] skipped because SKIP_EVAL=${SKIP_EVAL}"
fi

LOG_DIR="${CHIPLETFM_ROOT}/logs/output/${TASK}/${METHOD}/seed_${SEED}"
PLACEMENT_DIR="${LOG_DIR}/placement"
HOTSPOT_RUN_ROOT="${HOTSPOT_RUN_ROOT:-${FLOW_GCN_ROOT}/resultEval}"
HOTSPOT_ROOT="${HOTSPOT_ROOT:-${LOG_DIR}/hotspot}"
HOTSPOT_DIR="${HOTSPOT_DIR:-${HOTSPOT_ROOT}}"
HOTSPOT_RUN_CASES="${HOTSPOT_RUN_CASES:-${HOTSPOT_RUN_ROOT}/run_cases.sh}"
if [[ ! -f "${HOTSPOT_RUN_CASES}" && -f "${FLOW_GCN_ROOT}/tap2.5d_hoteval/run_cases.sh" ]]; then
  HOTSPOT_RUN_CASES="${FLOW_GCN_ROOT}/tap2.5d_hoteval/run_cases.sh"
fi
HOTSPOT_LOG="${LOG_DIR}/hotspot_run_cases.log"

mkdir -p "${LOG_DIR}"

if [[ "${SKIP_HOTSPOT}" != "1" ]]; then
  echo "[hotspot] placement_dir=${PLACEMENT_DIR}" | tee "${HOTSPOT_LOG}"
  echo "[hotspot] hotspot_dir=${HOTSPOT_DIR}" | tee -a "${HOTSPOT_LOG}"
  bash "${HOTSPOT_RUN_CASES}" "${PLACEMENT_DIR}" "${HOTSPOT_DIR}" grid 2>&1 | tee -a "${HOTSPOT_LOG}"
else
  echo "[hotspot] skipped because SKIP_HOTSPOT=${SKIP_HOTSPOT}"
fi

python - "${METHOD}" "${HOTSPOT_DIR}" "${LOG_DIR}" "${PLACEMENT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

method = sys.argv[1]
hotspot_dir = Path(sys.argv[2])
log_dir = Path(sys.argv[3])
placement_dir = Path(sys.argv[4])


def parse_temperature_file(path):
    values = []
    if not path.exists():
        return None, None, 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values.append(float(parts[1]))
        except ValueError:
            pass
    if not values:
        return None, None, 0
    return max(values), sum(values) / len(values), len(values)


rows = []
case_names = [p.stem for p in sorted(placement_dir.glob("*.json"))]
for case in case_names:
    case_dir = hotspot_dir / case
    grid_path = case_dir / f"{case}.grid.steady"
    steady_path = case_dir / f"{case}.steady"
    grid_max_k, grid_mean_k, grid_count = parse_temperature_file(grid_path)
    block_max_k, block_mean_k, block_count = parse_temperature_file(steady_path)
    rows.append({
        "method": method,
        "case": case,
        "grid_max_k": grid_max_k,
        "grid_max_c": None if grid_max_k is None else grid_max_k - 273.15,
        "grid_mean_k": grid_mean_k,
        "grid_mean_c": None if grid_mean_k is None else grid_mean_k - 273.15,
        "grid_points": grid_count,
        "block_max_k": block_max_k,
        "block_max_c": None if block_max_k is None else block_max_k - 273.15,
        "block_mean_k": block_mean_k,
        "block_mean_c": None if block_mean_k is None else block_mean_k - 273.15,
        "block_count": block_count,
        "grid_steady_path": str(grid_path),
        "steady_path": str(steady_path),
    })

csv_path = log_dir / "hotspot_temperatures.csv"
json_path = log_dir / "hotspot_temperatures.json"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
else:
    csv_path.write_text("", encoding="utf-8")
json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[hotspot] wrote {csv_path}")
print(f"[hotspot] wrote {json_path}")
PY
