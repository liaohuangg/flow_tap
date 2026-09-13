#!/usr/bin/env bash
set -euo pipefail

# 画 AT / RL 全部布局的热图。
#
# 数据来源: eval_layout.py 的热仿真输出 <method>_result/eval_out/<stem>/<stem>.grid.steady
# 绘制脚本: gen_dataset/draw_thermal_map.py (自动把 HotSpot 的 K 翻转并转成 ℃,
#           并从 L4_ChipLayer.flp 读 chiplet 矩形叠加边框)
# 输出:     resultEval/thermal_figs/<AT|RL>/<stem>.png
#
# 用法:
#   bash draw_thermal_fig.sh                 # 画 AT + RL, 单位 ℃
#   bash draw_thermal_fig.sh K               # 单位 K
#   bash draw_thermal_fig.sh C AT            # 只画 AT
#   bash draw_thermal_fig.sh C "AT RL"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAW="${SCRIPT_DIR}/../gen_dataset/draw_thermal_map.py"
PY="/root/anaconda3/envs/chipdiffusion/bin/python"

UNIT="${1:-C}"
METHODS="${2:-AT RL}"

if [[ ! -f "${DRAW}" ]]; then
  echo "[error] 找不到绘制脚本: ${DRAW}" >&2
  exit 2
fi

OUT_ROOT="${SCRIPT_DIR}/thermal_figs"
n=0

for m in ${METHODS}; do
  ROOT="${SCRIPT_DIR}/${m}_result/eval_out"
  OUTDIR="${OUT_ROOT}/${m}"
  if [[ ! -d "${ROOT}" ]]; then
    echo "[skip] 无热仿真输出目录: ${ROOT}" >&2
    continue
  fi
  mkdir -p "${OUTDIR}"

  shopt -s nullglob
  for steady in "${ROOT}"/*/*.grid.steady; do
    stem="$(basename "${steady}" .grid.steady)"
    flp="${ROOT}/${stem}/${stem}L4_ChipLayer.flp"
    out="${OUTDIR}/${stem}.png"

    if [[ ! -f "${flp}" ]]; then
      echo "[warn] ${stem}: 缺 ${flp}, 只画温度场不叠布局边框" >&2
      flp=""
    fi

    if [[ -n "${flp}" ]]; then
      "${PY}" "${DRAW}" --steady "${steady}" --flp "${flp}" --out "${out}" \
        --unit "${UNIT}" --show-names 0 --title "${m}  ${stem}"
    else
      "${PY}" "${DRAW}" --steady "${steady}" --out "${out}" \
        --unit "${UNIT}" --show-names 0 --title "${m}  ${stem}"
    fi

    echo "[draw] ${out}"
    n=$((n + 1))
  done
  shopt -u nullglob
done

echo "[draw] 完成: ${n} 张热图 -> ${OUT_ROOT}"
