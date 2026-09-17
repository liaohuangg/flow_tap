#!/usr/bin/env python3
"""build_final_time.py — 最终耗时对比表 (精简版): 每 case 一行, AT 均值 vs FM alpha_0p1。

12 行 = 12 个 case。列只有 6 个:
    case                     case 名
    AT_mean_s                ATPlace 5 个 seed 的平均墙钟 (per case, 无 profile 概念)
    FM_alpha_0p1_total_s     = sampling + stage_a + stage_b
    FM_alpha_0p1_sampling_s  纯 FM 采样 (100 步 ODE + 热/线长 guidance)
    FM_alpha_0p1_stage_a_s   去重叠 (repair_overlap)
    FM_alpha_0p1_stage_b_s   双守卫下的线长优化 (refine_wl)

alpha_0p9 那一半**不进这张表** (用户口径: 只留 alpha_0p1)。源 FMtime.csv 两个
profile 都在, 要一起看直接看那份, 或把本脚本的 PROFILE 改回双 profile。

FM 的 total 不含合法化里的 other 开销 (热点定位 + 两次神经前向 + 画图 + 写 report,
本批 run 约 5.7s/case) —— 那是加分项开销不是布局阶段。需要含 other 的口径时,
用 sampling + legalize_total, FMtime.csv / 上级 runtime.csv 里都有原始字段。

用法:
    python build_final_time.py            # 写同目录 final_time.csv
    python build_final_time.py --check    # 只打印, 不写盘
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
AT_CSV = HERE / "ATtime.csv"
FM_CSV = HERE / "FMtime.csv"
OUT_CSV = HERE / "final_time.csv"

PROFILE = "alpha_0p1"
COLUMNS = ["case", "AT_mean_s", "FM_alpha_0p1_total_s", "FM_alpha_0p1_sampling_s",
           "FM_alpha_0p1_stage_a_s", "FM_alpha_0p1_stage_b_s"]


def natkey(case: str):
    """Case6 < Case9 < Case10 —— 纯字典序会把 Case10 排到 Case6 前面。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", case)]


def r3(value):
    return "" if value is None else round(value, 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只打印, 不写 final_time.csv")
    args = ap.parse_args()

    for path in (AT_CSV, FM_CSV):
        if not path.is_file():
            raise SystemExit(f"缺输入: {path}\n(先跑 build_fmtime.py 生成 FMtime.csv)")

    with AT_CSV.open(encoding="utf-8") as handle:
        at_rows = {r["case"]: r for r in csv.DictReader(handle)}
    with FM_CSV.open(encoding="utf-8") as handle:
        fm_rows = [r for r in csv.DictReader(handle) if r["profile"] == PROFILE]

    # 同一 case 只应有一个 profile 行, 多出来说明源表结构变了
    seen, problems = set(), []
    for row in fm_rows:
        if row["case"] in seen:
            problems.append(f"{row['case']}: FMtime.csv 里 {PROFILE} 不止一行")
        seen.add(row["case"])
    for case in sorted(set(fm_rows and seen) - set(at_rows), key=natkey):
        problems.append(f"{case}: FM 有, ATtime.csv 里没有")
    for case in sorted(set(at_rows) - seen, key=natkey):
        problems.append(f"{case}: ATtime.csv 有, FM 里没有")

    rows = []
    for entry in sorted(fm_rows, key=lambda r: natkey(r["case"])):
        case = entry["case"]
        at = at_rows.get(case, {})
        sampling = float(entry["sampling_mean_s"])
        stage_a = float(entry["stage_a_mean_s"])
        stage_b = float(entry["stage_b_mean_s"])
        rows.append({
            "case": case,
            "AT_mean_s": r3(float(at["total_time_mean_s"]) if at else None),
            "FM_alpha_0p1_total_s": r3(sampling + stage_a + stage_b),
            "FM_alpha_0p1_sampling_s": r3(sampling),
            "FM_alpha_0p1_stage_a_s": r3(stage_a),
            "FM_alpha_0p1_stage_b_s": r3(stage_b),
        })

    print(f"{'case':10s}{'AT(s)':>10}{'FM_tot':>9}{'=samp':>8}{'A':>8}{'B':>8}{'AT/FM':>8}")
    for row in rows:
        at_ms, fm_ms = row["AT_mean_s"], row["FM_alpha_0p1_total_s"]
        ratio = at_ms / fm_ms if at_ms and fm_ms else None
        print(f"{row['case']:10s}{at_ms:>10.1f}{fm_ms:>9.3f}"
              f"{row['FM_alpha_0p1_sampling_s']:>8.3f}"
              f"{row['FM_alpha_0p1_stage_a_s']:>8.3f}"
              f"{row['FM_alpha_0p1_stage_b_s']:>8.3f}"
              f"{ratio if ratio is None else round(ratio, 2):>8}")

    print(f"\n行数 = {len(rows)}")
    if problems:
        print(f"\n{len(problems)} 条问题:")
        for item in problems:
            print(f"  - {item}")

    if args.check:
        print("\n--check: 未写盘")
        return 1 if problems else 0

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {OUT_CSV}  ({len(rows)} 行 x {len(COLUMNS)} 列)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
