#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash draw_thermal_fig.sh [layer] [unit]
# Example:
#   bash draw_thermal_fig.sh -1 C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/config_thermal_tap25"
DRAW="${SCRIPT_DIR}/../Dataset/dataset/hotspot/draw_thermal_map.py"
# /root/placement/flow_GCN/resultEval/config_thermal_tap25
LAYER="${1:-0}"
UNIT="${2:-C}"

shopt -s nullglob

for d in "${ROOT}"/*_Case*_placement; do
  [ -d "$d" ] || continue

  for steady in "$d"/*_Case*_placement.grid.steady; do
    [ -f "$steady" ] || continue

    base="$(basename "$steady" .grid.steady)"
    out="$d/${base}_layer${LAYER}.png"

    # Prefer per-case FLP in the same directory, e.g. 00_Case1_placementL4_ChipLayer.flp
    flp="$d/${base}L4_ChipLayer.flp"
    if [ ! -f "$flp" ]; then
      flp="$d/L4_ChipLayer.flp"
    fi

    python "$DRAW" \
      --steady "$steady" \
      --out "$out" \
      --layer "$LAYER" \
      --unit "$UNIT" \
      --show-names 0 \
      --flp "$flp"

    echo "[draw] saved: $out"
  done
done

# -1 C 0
