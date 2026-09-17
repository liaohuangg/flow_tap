#!/usr/bin/env python3
"""build_fmtime.py — FM(本方法) 按 (case, profile) 汇总的**阶段耗时表**。

一行 = 一个 (case, profile) = 5 个 seed 的均值, 12 case x 2 profile = 24 行。

总时间的口径
--------------------------------------------------------------------------
    total = sampling + stage_a + stage_b

采样侧
  sampling_s        = ``metrics_summary.json`` 的 ``model_time``
                      纯 FM 采样 (100 步 ODE + 热/线长 guidance)
合法化侧 (``*_legal_report.json`` 的 ``timing_s``)
  stage_a_s         = ``stage_a_repair``   去重叠 (repair_overlap)
  stage_b_s         = ``stage_b_refine``   双守卫下的线长优化 (refine_wl)
  other_s           = ``total - stage_a - stage_b``
                      热点定位 + 两次神经前向 + 画对比图 + 写 report。
                      **不**计入本表 total (它不是布局阶段的耗时), 单列一列备查。
  legalize_total_s  = ``timing_s.total``   = stage_a + stage_b + other

出处: 同批 run 的 ``../FM_result/pareto_table_profiles_5seed/runtime.csv``
(那一层是 120 行 = 一个解一行, 由 build_runtime.py 生成)。
本表只做按 (case, profile) 的 5-seed 聚合, 不再碰源 json。

用法:
    python build_fmtime.py            # 写同目录 FMtime.csv
    python build_fmtime.py --check    # 只打印, 不写盘
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../resultEval/runtime
SRC_CSV = HERE.parent / "FM_result" / "pareto_table_profiles_5seed" / "runtime.csv"
OUT_CSV = HERE / "FMtime.csv"

N_SEEDS = 5
PROFILE_ORDER = ["alpha_0p1", "alpha_0p9"]


def natkey(case: str):
    """Case6 < Case9 < Case10 —— 纯字典序会把 Case10 排到 Case6 前面。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", case)]


def mean(vals):
    return st.mean(vals) if vals else None


def std(vals):
    return st.stdev(vals) if len(vals) > 1 else 0.0


def r3(value):
    return "" if value is None else round(value, 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只打印, 不写 FMtime.csv")
    args = ap.parse_args()

    if not SRC_CSV.is_file():
        raise SystemExit(f"缺输入: {SRC_CSV}")

    with SRC_CSV.open(encoding="utf-8") as handle:
        src = list(csv.DictReader(handle))

    groups = defaultdict(list)
    for row in src:
        groups[(row["case"], row["profile"])].append(row)

    problems = []
    keys = sorted(groups, key=lambda k: (natkey(k[0]),
                                         PROFILE_ORDER.index(k[1])
                                         if k[1] in PROFILE_ORDER else 99,
                                         k[1]))

    rows = []
    for case, profile in keys:
        seeds = sorted(groups[(case, profile)], key=lambda r: int(r["seed"]))
        n = len(seeds)
        if n != N_SEEDS:
            problems.append(f"{case}/{profile}: seed 数 {n} != {N_SEEDS}")

        sampling = [float(r["sampling_s"]) for r in seeds]
        stage_a = [float(r["legalize_stage_a_s"]) for r in seeds]
        stage_b = [float(r["legalize_stage_b_s"]) for r in seeds]
        other = [float(r["legalize_other_s"]) for r in seeds]
        legal_tot = [float(r["legalize_total_s"]) for r in seeds]
        # 总时间按 seed 逐个相加, 再统计 —— 不等于"均值的和"以外的任何口径
        total = [s + a + b for s, a, b in zip(sampling, stage_a, stage_b)]

        for label, vals in (("stage_a", stage_a), ("stage_b", stage_b),
                            ("other", other)):
            if min(vals) < -1e-6:
                problems.append(f"{case}/{profile}: {label} 出现负值 {min(vals):.3f}s")

        row = {
            "case": case,
            "profile": profile,
            "n_seeds": n,
            "seeds": ";".join(r["seed"] for r in seeds),
            "sampling_mean_s": r3(mean(sampling)),
            "stage_a_mean_s": r3(mean(stage_a)),
            "stage_b_mean_s": r3(mean(stage_b)),
            "other_mean_s": r3(mean(other)),
            "legalize_total_mean_s": r3(mean(legal_tot)),
            "total_mean_s": r3(mean(total)),
            "total_std_s": r3(std(total)),
            "total_min_s": r3(min(total)),
            "total_max_s": r3(max(total)),
        }
        for i, r in enumerate(seeds, start=1):
            row[f"seed{i}_sampling_s"] = r3(float(r["sampling_s"]))
            row[f"seed{i}_stage_a_s"] = r3(float(r["legalize_stage_a_s"]))
            row[f"seed{i}_stage_b_s"] = r3(float(r["legalize_stage_b_s"]))
            row[f"seed{i}_total_s"] = r3(float(r["sampling_s"])
                                         + float(r["legalize_stage_a_s"])
                                         + float(r["legalize_stage_b_s"]))
        rows.append(row)

    columns = list(rows[0].keys())

    print(f"{'case':10s}{'profile':10s}{'samp':>9}{'A':>8}{'B':>8}{'other':>8}"
          f"{'total':>9}{'std':>7}")
    for row in rows:
        print(f"{row['case']:10s}{row['profile']:10s}"
              f"{row['sampling_mean_s']:>9.3f}{row['stage_a_mean_s']:>8.3f}"
              f"{row['stage_b_mean_s']:>8.3f}{row['other_mean_s']:>8.3f}"
              f"{row['total_mean_s']:>9.3f}{row['total_std_s']:>7.3f}")

    print(f"\n行数 = {len(rows)}  (源 {len(src)} 行 / {len(groups)} 组)")
    print(f"total 均值 = {mean([float(r['total_mean_s']) for r in rows]):.3f}s")
    print(f"其中 other 被排除的部分, 均值 = "
          f"{mean([float(r['other_mean_s']) for r in rows]):.3f}s")

    if problems:
        print(f"\n{len(problems)} 条问题:")
        for item in problems:
            print(f"  - {item}")

    if args.check:
        print("\n--check: 未写盘")
        return 1 if problems else 0

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {OUT_CSV}  ({len(rows)} 行)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
