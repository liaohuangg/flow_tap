#!/usr/bin/env bash
# wlsweep_status.sh — 一行看清 wl_weight 扫描跑到哪了。
# 用法:  ./wlsweep_status.sh          看一次
#        ./wlsweep_status.sh -f       持续刷新 (每 30s, Ctrl-C 退出)
#
# 默认看 result_wlsweep (acend910/hp11_m, 50 点)。看别的批次要指目录 + case:
#   WLSWEEP_DIR=$PWD/result_wlsweep_c6c8 WLSWEEP_CASES="Case6_bump Case8_bump" ./wlsweep_status.sh
set -uo pipefail
DEFAULT_R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/result_wlsweep"
R="${WLSWEEP_DIR:-$DEFAULT_R}"
N_WL="${N_WL:-50}"
read -ra CASES <<< "${WLSWEEP_CASES:-acend910_bump hp11_m_bump}"
TOTAL=$(( N_WL * ${#CASES[@]} ))
show() {
  local d f s run
  d=$(grep -c '^\[done \]' "$R/progress.log" 2>/dev/null || true)
  f=$(grep -c '^\[fail \]' "$R/progress.log" 2>/dev/null || true)
  s=$(grep -c '^\[skip\]'  "$R/progress.log" 2>/dev/null || true)
  run=$(( $(pgrep -c -f '[r]eproduce.py' 2>/dev/null || true) / 2 ))

  # 进度条
  local w=50
  local filled=$(( d * w / TOTAL ))
  local bar; bar=$(printf '#%.0s' $(seq 1 $filled 2>/dev/null))$(printf -- '-%.0s' $(seq 1 $((w-filled)) 2>/dev/null))

  echo "==== wl_weight 扫描  $(date '+%F %T')   ($R) ===="
  printf '  [%s] %3d/%d\n' "$bar" "$d" "$TOTAL"
  echo "  已完成 $d   失败 $f   跳过 $s   正在跑 $run 路"
  echo
  for c in "${CASES[@]}"; do
    local n; n=$(ls -d "$R/$c"/wl*/seed1/layout.json 2>/dev/null | wc -l)
    local last; last=$(ls -d "$R/$c"/wl* 2>/dev/null | sed 's|.*/wl||' | sort -n | tail -1)
    printf '  %-16s %2d/%d' "$c" "$n" "$N_WL"
    [ -n "${last:-}" ] && printf '   (已建到 wl%s, 含在跑的)' "$last"
    echo
  done
  echo
  echo "  最近 3 条:"
  tail -3 "$R/progress.log" 2>/dev/null | sed 's/^/    /'
  if [ "$f" -gt 0 ]; then echo; echo "  ⚠ 失败明细:"; sed 's/^/    /' "$R/wlsweep_report.txt"; fi
  [ "$d" -ge "$TOTAL" ] && { echo; echo "  ✅ $TOTAL 个槽位全部完成"; }
true
}
if [ "${1:-}" = "-f" ]; then while true; do clear; show; sleep 30; done; else show; fi
