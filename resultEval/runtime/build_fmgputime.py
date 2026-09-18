#!/usr/bin/env python3
"""build_fmgputime.py — FM(本方法)**本机 GPU** 按 (case, profile) 的阶段耗时表。

列口径与 ``FMCPUTime.log`` 逐列对齐 (前 13 列 + seed 明细 + canvas_multiple + source),
便于两张表并排读:

    total = sampling + stage_a + stage_b          (other 单列备查, **不计入** total)

数据来源 (只用一个文件, 不跨批次混)
--------------------------------------------------------------------------
``resultEval/FM_result/pareto_table_profiles_5seed/runtime.csv``
  120 例那批 (12 case x 2 profile x 5 seed) 的**真机 GPU** 耗时, 由
  resultEval/FM_result/pareto_table_profiles_5seed/build_runtime.py 生成, 字段:
      sampling_s           <- metrics_summary.json 的 model_time (纯 FM 采样)
      legalize_stage_a_s   <- legal report timing_s.stage_a_repair
      legalize_stage_b_s   <- legal report timing_s.stage_b_refine
      legalize_other_s     <- timing_s.total - A - B
      legalize_total_s     <- timing_s.total
  该批是 **JOBS=1 独占**跑的 (无并发抢卡), 所以行间可直接横比。

**不含** preprocess / 出图 / 进程启动 / 加载 ckpt —— 与 FMCPUTime 同口径。
那些加在 runtime.csv 的 wall_total_s / setup_overhead_s 里。

覆盖范围与缺口 (读表前必须知道)
--------------------------------------------------------------------------
1. 本表覆盖 runtime.csv 里的 **12 个 case** (Case6-10 + acend910/cpu-dram/hp11_m/
   multigpu/syn1/syn4/xerox8_m), 每个 2 个 profile x 5 seed = 24 行。
2. **4 个新 case (hp6_m/hp8_m/xerox6_m/xerox7_m) 不在本表里** —— 它们不在
   runtime.csv 的批次中。它们在 CPU 表 (FMCPUTime.log)、以及 newsweep 那批
   (selected-pareto-newsweep_20260918, 3 seed x 50 组权重) 里; 后者是
   **JOBS=3 并发 + 另一训练任务同时在跑**, 计时不可与独占行横比, 故未纳入。
   要补新 case 的干净 GPU 时间, 需要以 JOBS=1 单独再跑一轮。

canvas_multiple = 画布**面积倍数** = side^2 / 本体面积 (与 FMCPUTime 同义: 2 = util 0.50)。
   side 优先取 benchmark json 的 ``canvas_side_mm`` (hp6_m/hp8_m 由 ATPlace interposer
   给出 -> 3.5x / 3.0x), 无该字段时按仓库统一的 util 0.50 (2x)。
   Case6-10 已从 benchmark 移除取不到 json, 回退 2 —— 与 CPU 那轮一致 (它们也是 2x)。

用法:
    python build_fmgputime.py           # 写同目录 FMGpuTime.log
    python build_fmgputime.py --check   # 只打印, 不写盘
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # .../resultEval/runtime
PROJECT = HERE.parent.parent
BENCH = PROJECT / "benchmark" / "cases_hubump"
RUNTIME_CSV = (PROJECT / "resultEval" / "FM_result" / "pareto_table_profiles_5seed"
               / "runtime.csv")

PROFILE_ORDER = ["alpha_0p1", "alpha_0p9"]
SOURCE_LABEL = "pareto_table_profiles_5seed/runtime.csv (JOBS=1 exclusive)"


def natkey(case: str):
    """Case6 < Case9 < Case10 —— 纯字典序会把 Case10 排到 Case6 前面。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", case)]


def r3(value):
    return "" if value is None else round(value, 3)


def num(row, key):
    """空字段 = 没测到, 按 None 往下传, 不要当成 0。"""
    raw = (row.get(key) or "").strip()
    return None if not raw else float(raw)


def canvas_multiple(case: str) -> float:
    """画布面积倍数 = side^2 / 本体面积; side 取 benchmark 的 canvas_side_mm, 否则 util 0.50。"""
    path = BENCH / f"{case}.json"
    if not path.is_file():
        return 2.0                       # Case6-10 已从 benchmark 移除; CPU 那轮也是 2x
    rec = json.loads(path.read_text(encoding="utf-8"))
    area = sum(float(c["width"]) * float(c["height"]) for c in rec["chiplets"])
    side = rec.get("canvas_side_mm")
    if side is None:
        side = math.sqrt(area / 0.50)
    return round((float(side) ** 2) / area, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只打印, 不写 FMGpuTime.log")
    args = ap.parse_args()

    if not RUNTIME_CSV.is_file():
        raise SystemExit(f"缺输入: {RUNTIME_CSV}")
    with RUNTIME_CSV.open(encoding="utf-8") as fh:
        src = list(csv.DictReader(fh))
    problems = []

    keys = sorted({(r["case"], r["profile"]) for r in src},
                  key=lambda k: (natkey(k[0]),
                                 PROFILE_ORDER.index(k[1]) if k[1] in PROFILE_ORDER else 99, k[1]))

    out_rows = []
    for case, profile in keys:
        group = sorted((r for r in src if r["case"] == case and r["profile"] == profile),
                       key=lambda r: int(r["seed"]))
        n = len(group)
        sampling = [num(r, "sampling_s") for r in group]
        stage_a = [num(r, "legalize_stage_a_s") for r in group]
        stage_b = [num(r, "legalize_stage_b_s") for r in group]
        legal_tot = [num(r, "legalize_total_s") for r in group]
        if any(v is None for v in sampling + stage_a + stage_b + legal_tot):
            problems.append(f"{case}/{profile}: 有阶段字段为空")
            continue
        other = [t - a - b for t, a, b in zip(legal_tot, stage_a, stage_b)]
        total = [s + a + b for s, a, b in zip(sampling, stage_a, stage_b)]
        for label, vals in (("stage_a", stage_a), ("stage_b", stage_b), ("other", other)):
            if min(vals) < -1e-6:
                problems.append(f"{case}/{profile}: {label} 出现负值 {min(vals):.3f}s")

        row = {
            "case": case, "profile": profile, "n_seeds": n,
            "seeds": ";".join(r["seed"] for r in group),
            "sampling_mean_s": r3(st.mean(sampling)),
            "stage_a_mean_s": r3(st.mean(stage_a)),
            "stage_b_mean_s": r3(st.mean(stage_b)),
            "other_mean_s": r3(st.mean(other)),
            "legalize_total_mean_s": r3(st.mean(legal_tot)),
            "total_mean_s": r3(st.mean(total)),
            "total_std_s": r3(st.stdev(total)) if n > 1 else "",
            "total_min_s": r3(min(total)),
            "total_max_s": r3(max(total)),
        }
        for i, r in enumerate(group, start=1):
            row[f"seed{i}_sampling_s"] = r3(num(r, "sampling_s"))
            row[f"seed{i}_stage_a_s"] = r3(num(r, "legalize_stage_a_s"))
            row[f"seed{i}_stage_b_s"] = r3(num(r, "legalize_stage_b_s"))
            row[f"seed{i}_total_s"] = r3(num(r, "sampling_s")
                                         + num(r, "legalize_stage_a_s")
                                         + num(r, "legalize_stage_b_s"))
        row["canvas_multiple"] = canvas_multiple(case)
        row["source"] = SOURCE_LABEL
        out_rows.append(row)

    columns = list(out_rows[0].keys())
    print(f"{'case':12s}{'profile':10s}{'canvas':>7}{'samp':>9}{'A':>8}{'B':>8}"
          f"{'other':>8}{'total':>9}{'std':>7}{'n':>3}")
    for row in out_rows:
        print(f"{row['case']:12s}{row['profile']:10s}{str(row['canvas_multiple']) + 'x':>7}"
              f"{row['sampling_mean_s']:>9.3f}{row['stage_a_mean_s']:>8.3f}"
              f"{row['stage_b_mean_s']:>8.3f}{row['other_mean_s']:>8.3f}"
              f"{row['total_mean_s']:>9.3f}{str(row['total_std_s']):>7}{row['n_seeds']:>3}")
    print(f"\n行数 = {len(out_rows)}  ({len({r['case'] for r in out_rows})} 个 case, "
          f"源 {len(src)} 次运行)")
    missing = sorted({r["case"] for r in src} ^ {r["case"] for r in out_rows})
    if missing:
        print(f"未纳入的 case: {missing}")
    if problems:
        print(f"\n{len(problems)} 条问题:")
        for item in problems:
            print(f"  - {item}")

    if args.check:
        print("\n--check: 未写盘")
        return 1 if problems else 0

    out_log = HERE / "FMGpuTime.log"
    with out_log.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=columns)
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"\n-> {out_log}  ({len(out_rows)} 行)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
