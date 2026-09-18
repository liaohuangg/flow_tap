#!/usr/bin/env bash
# run_wlsweep_50set.sh — 对 cases_50set 里的 case 做 wl_weight 扫描。
#
# 目的: 固定 random_seed=1, 只改 Thermal-aware.json 里的 wl_weight,
#       在 [0, 0.01] 上均匀取 50 个点, 每个点跑出 1 个解。
#       => 每个 case 50 个解。
#       cases_50set 共 11 个 case; 5 个已扫满 (acend910 / cpu-dram / hp11_m / xerox8_m /
#       hp6_m), hp8_m_bump 求解不出来 (placeflow NaN) 已弃, CASES 默认只列剩下 5 个。
#
# 与 run_seed1-5_0wl.sh 的差异:
#   - 输入用例目录: cases_50set
#   - 结果目录:     result_wlsweep   (单独目录, 不碰 result / result_0wl)
#   - 扫描维度的不是 seed 而是 wl_weight: 每个 case 一个 wl 网格 (50 点)
#   - 可以并行 (JOBS 环境变量, 默认 12); 参考脚本是纯串行
#
# 并行安全前提 (2026-09-16 修):
#   HotSpot 的中间文件 (new_hotspot.config / *.flp / *.ptrace / *.steady) 全部由
#   Thermal_solver.path 拼接, 默认是共享的 $REPO_ROOT/thermal/。并行时同名文件互相
#   覆盖、再读回别人算的温度, 会让布局被污染 —— 实测同一组参数两次跑出的 hpwl 差
#   0.6~2.2%、坐标全变。修法: 每次求解用 ATPLACE_THERMAL_DIR 指到 out_dir 下的
#   私有 thermal 目录 (复制 hotspot.config + 软链 hotspot 二进制)。JOBS=1 时共享
#   目录本来也不会冲突, 但本脚本一律用私有目录, 保证 JOBS 可以随便调。
#   另固定 PYTHONHASHSEED=0, 排除 set/dict 遍历顺序带来的不确定性。
#
# 规则 (与参考脚本一致):
#   - 每次求解只跑 1 个 100-iter 尝试 (ATP_TIMEOUT=30 让内部 deadline 立刻过期),
#     seed 固定为 1, 结果写到 result_wlsweep/<case>/wl<NN>/seed1/。
#   - 单次求解 wall-clock 上限 OUTER_TIMEOUT (默认 1 小时); 没跑出来就记录耗时+原因。
#   - 可重入: 已有 layout.json 的槽位默认跳过, 中断后直接重跑本脚本即可续上。
#
# 用法:
#   ./run_wlsweep_50set.sh                    # 默认 7 个 case x 50 = 350 次
#   DRY_RUN=1 ./run_wlsweep_50set.sh          # 只生成 param 文件, 不跑
#   JOBS=8 ./run_wlsweep_50set.sh             # 8 路并行
#   CASES_OVERRIDE=acend910_bump WL_OVERRIDE="0 25 49" ./run_wlsweep_50set.sh
#                                             # 只跑 acend910 的 wl00 / wl25 / wl49
#   FORCE=1 ./run_wlsweep_50set.sh            # 忽略已有结果, 全部重跑
#
#   # 另一批 case / 另一个 wl 范围 -> 换结果目录, 不碰上一轮:
#   CASES_OVERRIDE="Case6_bump Case8_bump" WL_MAX=0.02 \
#     RESULT_DIR=$PWD/result_wlsweep_c6c8 ./run_wlsweep_50set.sh
set -uo pipefail

PYTHON="${PYTHON:-/root/anaconda3/envs/ATPlan39/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="$REPO_ROOT/cases_50set"
# 结果目录可用环境变量覆盖 —— 换 case / 换 wl 范围时指到新目录, 不要盖掉上一轮的记录
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/result_wlsweep}"

MODE="${MODE:-thermal}"
PARAM_NAME="Thermal-aware.json"
ATP_TIMEOUT="${ATP_TIMEOUT:-30}"     # 内部 deadline=30s → 只跑第一个(唯一) seed 一次(100 iter)
OUTER_TIMEOUT="${OUTER_TIMEOUT:-3600}"
SEED=1                               # 固定 seed, 不扫
N_WL="${N_WL:-50}"                   # wl_weight 网格点数
WL_MAX="${WL_MAX:-0.01}"
JOBS="${JOBS:-12}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# 只列还没扫满的 case。已在 result_wlsweep 里扫满的 (acend910 / cpu-dram / hp11_m /
# xerox8_m 各 50 个, hp6_m 50 个) 不重跑 —— 想一起提交也行, run_one 见到
# seed1/layout.json 会直接跳过。
# hp8_m_bump 除外: 已确认**求解不出来** —— 每个槽位跑 ~11 分钟后 placeflow 崩在
# "Error Constant is Nan happen during placeflow!", 50 个槽位只出来 1 个 (wl00),
# 用户 2026-09-18 决定不再扫。要复查的话别把它加回来, 单跑一个槽位看 run_wl*.log 即可。
CASES=(multigpu_bump syn1_bump syn4_bump xerox6_m_bump xerox7_m_bump)

# 可用环境变量临时收窄范围, 空格分隔
if [ -n "${CASES_OVERRIDE:-}" ]; then read -ra CASES <<< "$CASES_OVERRIDE"; fi

log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$RESULT_DIR"
PARAM_DIR="$RESULT_DIR/params"
REPORT="$RESULT_DIR/wlsweep_report.txt"
PROGRESS="$RESULT_DIR/progress.log"
[ "$FORCE" = "1" ] && : > "$REPORT" || touch "$REPORT"
# progress.log 每轮只记本轮 (下面的计数按 '^\[done \]' 统计, 混轮会串味),
# 但上一轮的耗时记录还有用, 覆盖前先留一份带时间戳的副本。
if [ -s "$PROGRESS" ]; then
  cp "$PROGRESS" "$RESULT_DIR/progress_$(date '+%Y%m%d_%H%M%S').log"
fi
: > "$PROGRESS"

# ---- 生成 wl_weight 网格 + 每个 (case, wl) 的 param 文件 --------------------
# 网格: wl[k] = WL_MAX * k / (N_WL-1), k = 0..N_WL-1, 端点严格取到 0 和 WL_MAX。
mapfile -t WL_VALUES < <("$PYTHON" - "$N_WL" "$WL_MAX" <<'PY'
import sys
n, mx = int(sys.argv[1]), float(sys.argv[2])
for k in range(n):
    # 端点写死, 中间用 k/(n-1) 均匀分布, 避免浮点误差导致最后一个点不是 WL_MAX
    v = 0.0 if k == 0 else (mx if k == n - 1 else mx * k / (n - 1))
    print(f"{v:.12g}")
PY
)

if [ -n "${WL_OVERRIDE:-}" ]; then
  read -ra IDX <<< "$WL_OVERRIDE"
else
  IDX=($(seq 0 $((N_WL-1))))
fi

echo "wl_weight 网格 ($N_WL 点, ${WL_VALUES[0]} .. ${WL_VALUES[$((N_WL-1))]}, 步长 $(printf '%.8g' "$("$PYTHON" -c "print($WL_MAX/($N_WL-1))")"))"
echo "将跑的槽位: ${#IDX[@]} 个 wl × ${#CASES[@]} 个 case = $(( ${#IDX[@]} * ${#CASES[@]} )) 次求解 (JOBS=$JOBS)"

# 每个 (case, wl) 的 param.json 预先全部生成 —— 即使 DRY_RUN 也生成, 便于先检查
for case in "${CASES[@]}"; do
  case_dir="$CASES_DIR/$case"
  if [ ! -f "$case_dir/$PARAM_NAME" ]; then
    echo "[err ] 缺少 $case_dir/$PARAM_NAME" >&2
    exit 1
  fi
  mkdir -p "$PARAM_DIR/$case"
  for k in "${IDX[@]}"; do
    wl="${WL_VALUES[$k]}"
    wl_tag=$(printf "wl%02d" "$k")
    "$PYTHON" - "$case_dir/$PARAM_NAME" "$SEED" "$wl" "$PARAM_DIR/$case/${wl_tag}_${PARAM_NAME}" <<'PY'
import json, os, sys
src, seed, wl, dst = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
d = json.load(open(src))
d["random_seed"] = seed
d["wl_weight"] = wl
it = os.environ.get("ITER_OVERRIDE", "")
if it:
    d.setdefault("floorplan_stages", [{}])[0]["iteration"] = int(it)
json.dump(d, open(dst, "w"), indent=2, ensure_ascii=False)
PY
  done
done
log "param 文件已生成 -> $PARAM_DIR/<case>/wl<NN>_${PARAM_NAME}"

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN=1, 只生成 param, 不跑求解。退出。"
  exit 0
fi

# ---- 单个槽位的求解 ---------------------------------------------------------
run_one() {
  local case="$1" k="$2" wl="$3"
  local wl_tag; wl_tag=$(printf "wl%02d" "$k")
  local out_dir="$RESULT_DIR/$case/$wl_tag"
  local seed_dir="$out_dir/seed${SEED}"
  mkdir -p "$out_dir"

  # stat 锁: 防止同一槽位被两个 worker 同时跑 (重入时也可能撞上)
  local lock="$out_dir/.lock"
  if ! mkdir "$lock" 2>/dev/null; then
    echo "[skip] $case/$wl_tag: 另一进程正在跑" >> "$PROGRESS"
    return 0
  fi
  trap 'rmdir "$lock" 2>/dev/null' RETURN

  if [ -f "$seed_dir/layout.json" ] && [ "$FORCE" != "1" ]; then
    echo "[skip] $case/$wl_tag wl=$wl: 已有 layout.json" >> "$PROGRESS"
    return 0
  fi

  # 交付给 reproduce.py 的 param 文件: 直接用它, 同时复制一份到 out_dir 便于对照
  cp "$PARAM_DIR/$case/${wl_tag}_${PARAM_NAME}" "$out_dir/param.json"

  # !!! 关键: HotSpot 的中间文件 (new_hotspot.config / *.flp / *.ptrace / *.steady)
  # 全部用 Thermal_solver.path 拼接, 默认指向共享的 $REPO_ROOT/thermal/。
  # 并行时多个进程会用同名文件互相覆盖、再读回别人算的温度 —— 布局会被污染。
  # 所以每次求解给一个私有 thermal 目录: 复制 config + 软链 hotspot 二进制。
  local th_dir="$out_dir/thermal"
  rm -rf "$th_dir"; mkdir -p "$th_dir"
  cp "$REPO_ROOT/thermal/hotspot.config" "$th_dir/hotspot.config"
  ln -sf "$REPO_ROOT/thermal/hotspot" "$th_dir/hotspot"

  local t0; t0=$(date +%s)
  echo "[start] $case/$wl_tag wl=$wl" >> "$PROGRESS"
  ATPLACE_TIMEOUT="$ATP_TIMEOUT" ATPLACE_THERMAL_DIR="$th_dir" \
    PYTHONHASHSEED="${PYTHONHASHSEED:-0}" \
    timeout -k 30 "$OUTER_TIMEOUT" \
    "$PYTHON" "$REPO_ROOT/reproduce.py" --case "$case" --mode "$MODE" \
    --case-dir "$CASES_DIR/$case" --param-file "$out_dir/param.json" --out-dir "$out_dir" \
    > "$out_dir/run_wl${k}.log" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  local elap_fmt; elap_fmt=$(printf '%02d:%02d:%02d' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60)))

  if [ -f "$seed_dir/layout.json" ]; then
    local hpwl; hpwl=$(grep -o '"hpwl": [0-9.e+-]*' "$seed_dir/layout.json" | head -1)
    echo "[done ] $case/$wl_tag wl=$wl ${elap_fmt} ($hpwl)" >> "$PROGRESS"
  else
    local reason
    reason=$(sed -n 's/.*\[placeflow\] error: //p' "$out_dir/run_wl${k}.log" | tail -1)
    if [ -z "$reason" ]; then
      case "$rc" in
        124|137) reason="超时(${OUTER_TIMEOUT}s 上限, rc=$rc)" ;;
        *)       reason="无 layout.json (rc=$rc)" ;;
      esac
    fi
    echo "[fail ] $case/$wl_tag wl=$wl ${elap_fmt} rc=$rc :: $reason" >> "$PROGRESS"
    echo "FAIL  case=$case  wl_index=$k  wl=$wl  耗时=${elap_fmt}  rc=$rc  原因=$reason" >> "$REPORT"
  fi
}

# ---- 主循环: 简单的 JOBS 路并行 --------------------------------------------
running=0
total=0
for case in "${CASES[@]}"; do
  for k in "${IDX[@]}"; do
    run_one "$case" "$k" "${WL_VALUES[$k]}" &
    running=$((running+1)); total=$((total+1))
    if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
  done
done
wait

# ---- 汇总 -------------------------------------------------------------------
ok=$(grep -c '^\[done \]' "$PROGRESS" || true)
sk=$(grep -c '^\[skip\]' "$PROGRESS" || true)
fa=$(grep -c '^\[fail \]' "$PROGRESS" || true)
log "=== 结束: 提交 $total 个槽位 (成功 $ok, 跳过 $sk, 失败 $fa) ==="
echo
echo "===== progress: $PROGRESS ====="
echo "===== 失败明细: $REPORT ====="
if [ -s "$REPORT" ]; then cat "$REPORT"; else echo "(无失败)"; fi
