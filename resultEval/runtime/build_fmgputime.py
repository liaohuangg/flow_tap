#!/usr/bin/env python3
"""build_fmgputime.py — FM(本方法)**本机 GPU** 按 (case, profile) 的阶段耗时表。

列口径与 ``FMCPUTime.log`` 逐列对齐 (前 13 列 + seed 明细 + canvas_multiple + source),
便于两张表并排读:

    total = sampling + stage_a + stage_b          (other 单列备查, **不计入** total)

数据来源 (两个文件, 每个 case 只来自其中一个, 不混算)
--------------------------------------------------------------------------
1. ``resultEval/FM_result/pareto_table_profiles_5seed/runtime.csv``
   120 例那批 (12 case x 2 profile x 5 seed) —— 旧 benchmark 的 12 个 case。
   source 记 ``pareto_table_profiles_5seed/runtime.csv (JOBS=1 exclusive)``。
2. ``resultEval/FM_result/pareto_new4_gpu/runtime.csv``
   后来加进 benchmark 的 4 个 case (hp6_m/hp8_m/xerox6_m/xerox7_m),
   20 例 (4 case x 1 profile x 5 seed), 见该目录 build_runtime.py。

两批的字段口径相同, 都由各自的 build_runtime.py 从
``metrics_summary.json`` + legal report 的 ``timing_s`` 抽出来:

    sampling_s           <- metrics_summary.json 的 model_time (纯 FM 采样)
    legalize_stage_a_s   <- legal report timing_s.stage_a_repair
    legalize_stage_b_s   <- legal report timing_s.stage_b_refine
    legalize_other_s     <- timing_s.total - A - B
    legalize_total_s     <- timing_s.total

两批都是 **JOBS=1 独占**跑的 (无并发抢卡), 所以行间可直接横比。

**不含** preprocess / 出图 / 进程启动 / 加载 ckpt —— 与 FMCPUTime 同口径。
那些加在 runtime.csv 的 wall_total_s / setup_overhead_s 里。

覆盖范围与缺口 (读表前必须知道)
--------------------------------------------------------------------------
1. 旧 12 个 case: 每个 **2 个 profile** (alpha_0p1/alpha_0p9) x 5 seed = 24 行。
2. 新 4 个 case: 每个 **1 个 profile** (alpha_0p9, 与 FMCPUTime.log 里那 4 行同槽)
   x 5 seed = 4 行 —— 只跑了 alpha_0p9, 没有 alpha_0p1 那一列。它们与旧 12 例
   用的是**同一个脚本、同一张权重表、同一批 5 个随机 seed** (1343216877 /
   1619340970 / 1991307090 / 312945035 / 731936390), 只把 CASES 收窄到 4 个,
   所以 seed 之间可以一一对上。
3. 合计 28 行 / 16 个 case —— 与 FMCPUTime.log 的 16 个 case 对齐。
4. **两个来源的 ``other`` 不可横比** (``total`` 口径相同, 但新批次的 other 几乎为 0):
   旧 120 例那批的 legal report 里 ``total - A - B`` 约 5.5~6.2s, 而同一份报告里
   ``hotspot.locate_time_s`` 只有 ~0.05s —— 也就是说那 ~5.8s 是 legalizer 内部
   **没有单独打点**的活 (两次神经前向 / 画对比图 / 写 report), 只能落在 other 里。
   新 4 例这批 (以及 09-18 的 newsweep) 同一台机器同一个 legalizer, other 只有
   ~0.07~0.10s, 即 A + B 已经把 legalize total 占满 —— 两次实验之间 legalizer 的
   内部打点口径变了。
   结论: ``stage_a`` / ``stage_b`` 的切分和 ``other`` **不要跨来源比**;
   ``legalize_total_mean_s`` 两边同口径, 可以比。副作用是旧 24 行的
   ``total = sampling + A + B`` 比"采样 + 完整合法化"少算了那 ~5.8s, 新 4 行没这个问题。
   旧行按用户要求保持原样不动, 这里只记录差异。

canvas_multiple = 画布**面积倍数** = side^2 / 本体面积 (与 FMCPUTime 同义: 2 = util 0.50)。
   side 的取值优先级与 diffusion/json_benchmark_dataset.py:131-137 完全一致:
   ``canvas_side_mm`` (mm) > ``interposer_size`` (um, 可 [x,y]) >
   ``interposer_size_um`` (um) > 按仓库统一的 util 0.50 推导。
   hp6_m/hp8_m 带 ATPlace 的 interposer 尺寸 -> 3.5x / 3.0x; 其余 -> 2x。
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
FM_RESULT = PROJECT / "resultEval" / "FM_result"

# (runtime.csv, source 标签, 是否独占) —— 顺序只影响控制台打印顺序。
SOURCES = [
    (FM_RESULT / "pareto_table_profiles_5seed" / "runtime.csv",
     "pareto_table_profiles_5seed/runtime.csv (JOBS=1 exclusive)"),
    (FM_RESULT / "pareto_new4_gpu" / "runtime.csv",
     "pareto_new4_gpu/runtime.csv (JOBS=1 exclusive)"),
]

PROFILE_ORDER = ["alpha_0p1", "alpha_0p9"]
# 与 FMCPUTime.log 的 16 个 case 对齐 (Case6-10 虽已从 benchmark 移除, 但表里有)。
EXPECTED_CASES = ["Case6", "Case7", "Case8", "Case9", "Case10",
                  "acend910", "cpu-dram", "hp11_m", "hp6_m", "hp8_m",
                  "multigpu", "syn1", "syn4", "xerox6_m", "xerox7_m", "xerox8_m"]


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
    """画布面积倍数 = side^2 / 本体面积; side 口径同 json_benchmark_dataset.py:131-137。"""
    path = BENCH / f"{case}.json"
    if not path.is_file():
        return 2.0                       # Case6-10 已从 benchmark 移除; CPU 那轮也是 2x
    rec = json.loads(path.read_text(encoding="utf-8"))
    area = sum(float(c["width"]) * float(c["height"]) for c in rec["chiplets"])
    side = rec.get("canvas_side_mm")
    if side is None and rec.get("interposer_size") is not None:
        side = _square_side_mm(rec["interposer_size"])
    if side is None and rec.get("interposer_size_um") is not None:
        side = _square_side_mm(rec["interposer_size_um"])
    if side is None:
        side = math.sqrt(area / 0.50)
    return round((float(side) ** 2) / area, 2)


def _square_side_mm(value) -> float:
    """interposer_size 是**微米**, 可写标量或 [x, y] (非正方形直接报错, 别静默取 x)。"""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"interposer_size 应为标量或 [x, y], 实际 {value!r}")
        x, y = float(value[0]), float(value[1])
        if abs(x - y) > 1e-6:
            raise ValueError(f"interposer_size 非正方形 ({x} x {y})")
        um = x
    else:
        um = float(value)
    return um / 1000.0


def load_grouped():
    """-> {(case, profile): (rows, source)}; 空/缺文件只记问题, 不静默跳过整张表。"""
    grouped, problems = {}, []
    for path, label in SOURCES:
        if not path.is_file():
            problems.append(f"缺输入: {path}")
            continue
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row["case"], row["profile"])
                if key not in grouped:
                    grouped[key] = ([], label)
                rows, first_label = grouped[key]
                if first_label != label:
                    # 同一个 (case, profile) 只能有一个来源, 否则 total_std 会把
                    # 两批不同实验的 seed 混在一起算。
                    problems.append(f"{key[0]}/{key[1]}: 两个来源都有, 保留 "
                                    f"{first_label!r}, 丢弃 {label!r}")
                    continue
                rows.append(row)
    return grouped, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只打印, 不写 FMGpuTime.log")
    args = ap.parse_args()

    grouped, problems = load_grouped()
    if not grouped:
        raise SystemExit("没有任何输入行; 检查 " + ", ".join(str(p) for p, _ in SOURCES))

    keys = sorted(grouped,
                  key=lambda k: (natkey(k[0]),
                                 PROFILE_ORDER.index(k[1]) if k[1] in PROFILE_ORDER else 99, k[1]))

    out_rows = []
    for case, profile in keys:
        group, label = grouped[(case, profile)]
        group = sorted(group, key=lambda r: int(r["seed"]))
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
        for label_, vals in (("stage_a", stage_a), ("stage_b", stage_b), ("other", other)):
            if min(vals) < -1e-6:
                problems.append(f"{case}/{profile}: {label_} 出现负值 {min(vals):.3f}s")

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
        row["source"] = label
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
          f"源 {sum(len(g) for g, _ in grouped.values())} 次运行)")

    missing = [c for c in EXPECTED_CASES if c not in {r["case"] for r in out_rows}]
    extra = sorted({r["case"] for r in out_rows} - set(EXPECTED_CASES), key=natkey)
    if missing:
        print(f"与 FMCPUTime.log 的 16 个 case 相比还缺: {missing}")
    else:
        print("16 个 case 全齐 (与 FMCPUTime.log 对齐)")
    if extra:
        print(f"多出 (不在 CPU 表里): {extra}")
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
