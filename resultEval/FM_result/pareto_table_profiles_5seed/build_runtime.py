#!/usr/bin/env python3
"""build_runtime.py — 按例子汇总这批 FM 解的**采样时间**与**合法化时间**。

输出 ``runtime.csv``: 一行 = 一个解 (120 行), 列的定义与出处如下。

采样侧 (来源: ``<run>/seed_<N>/metrics_summary.json``, 由
LayoutGenModel/diffusion/eval_thermal_guided.py:1107-1354 打点)
--------------------------------------------------------------------------
  sampling_s        = ``model_time``   = t2 - t1
                      纯 FM 采样: policies.open_loop (100 步 ODE + 热/线长 guidance)
  generation_s      = ``generation_time`` = t3 - t0
                      预处理 + 采样 + 出图 (2048x2048), **不含**神经评测
                      (t0->t1 是 cond 预处理, t2->t3 是 visualize_placement)
  inference_eval_s  = ``eval_time``    = t4 - t3
                      生成后立即跑的神经评测 (热模型 + 线长模型前向), 非物理评测

合法化侧 (来源: ``<run>/seed_<N>/legalized/<NN>_<case>_placement_legal_report.json``
的 ``timing_s``, 由 LayoutGenModel/legalizeLayout/legalize_layout.py 打点)
--------------------------------------------------------------------------
  legalize_stage_a_s  = ``timing_s.stage_a_repair``  去重叠 (repair_overlap)
  legalize_stage_b_s  = ``timing_s.stage_b_refine``  双守卫下的线长优化 (refine_wl)
  legalize_total_s    = ``timing_s.total``
  legalize_other_s    = legalize_total - stage_a - stage_b
                        热点定位 + 两次神经前向 + 画对比图 + 写 report, 报告里没有
                        单独字段, 约占 legalize_total 的 1/4, **不要**把
                        stage_a + stage_b 当成合法化总时间

批次墙钟 (来源: ``<batch>/timing.csv`` 的 ``total_seconds``, 由
run_selected_pareto_2profiles_5random_seeds_wsl.sh 的 run_one 用 date +%s 前后相减)
--------------------------------------------------------------------------
  wall_total_s      = 该解从进程启动到合法化结束的墙钟 (整数秒, 有 +/-1s 取整)
  setup_overhead_s  = wall_total - generation_s - legalize_total_s
                      进程启动 / 加载 checkpoint / 写文件的固定开销
  恒等式: wall_total_s == generation_s + legalize_total_s + setup_overhead_s

单位全部是秒。这批 run 是 ``JOBS=1`` 顺序跑的 (批次目录 mtime 跨度 7835s 与
``total_seconds`` 之和 7914s 吻合), 所以各例时间之间没有资源争用。
另: 本 run 的 ``timing.csv`` 只有 ``total_seconds`` 一列, 没有老 sweep 里的
``inference_seconds`` / ``legalization_seconds`` —— 采样时间只能从
``metrics_summary.json`` 取, 合法化时间只能从 legal report 的 ``timing_s`` 取。

用法:
    python build_runtime.py             # 生成/刷新同目录下的 runtime.csv
    python build_runtime.py --check     # 只校验不写盘
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../FM_result/pareto_table_profiles_5seed
PROJECT = HERE.parents[2]                       # .../flow_tap
RUN_TAG = "table_profiles_5seed_20260917"
BATCH_ROOT = PROJECT / "LayoutGenModel" / "logs" / "output" / "cases_hubump"
BATCH_DIR = BATCH_ROOT / f"selected-pareto-{RUN_TAG}"
MANIFEST = HERE / "manifest.csv"
OUT_CSV = HERE / "runtime.csv"

COLUMNS = [
    "out_file", "profile", "case", "seed", "thermal_weight", "wirelength_weight",
    "sampling_s", "generation_s", "inference_eval_s",
    "legalize_stage_a_s", "legalize_stage_b_s", "legalize_other_s", "legalize_total_s",
    "wall_total_s", "setup_overhead_s",
]


def num(value):
    """取数值, 取不到返回 None (NaN/字符串/缺字段都算取不到)。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def r3(value):
    return "" if value is None else round(value, 3)


def load_batch_wall() -> dict:
    """(profile, case, seed) -> total_seconds。"""
    wall = {}
    with (BATCH_DIR / "timing.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wall[(row["profile"], row["case"], row["seed"])] = num(float(row["total_seconds"]))
    return wall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只校验已算出的数据, 不写 runtime.csv")
    args = ap.parse_args()

    for path in (MANIFEST, BATCH_DIR / "timing.csv"):
        if not path.exists():
            print(f"缺输入: {path}", file=sys.stderr)
            return 2

    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    wall = load_batch_wall()

    rows, problems = [], []
    for entry in manifest:
        stem = Path(entry["out_file"]).stem
        profile, case, seed = entry["profile"], entry["case"], entry["seed"]
        t_label, w_label = entry["thermal_weight"], entry["wirelength_weight"]
        run_dir = BATCH_ROOT / f"selected-{profile}-{case}-t{t_label}-w{w_label}-{RUN_TAG}"
        seed_dir = run_dir / f"seed_{seed}"
        if not seed_dir.is_dir():
            problems.append(f"{stem}: 源目录不存在 {seed_dir}")
            continue

        metrics_path = seed_dir / "metrics_summary.json"
        if not metrics_path.is_file():
            problems.append(f"{stem}: 缺 {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}

        reports = sorted((seed_dir / "legalized").glob("*_legal_report.json"))
        if not reports:
            problems.append(f"{stem}: 缺 legal report")
        report = json.loads(reports[0].read_text(encoding="utf-8")) if reports else {}
        timing = report.get("timing_s") or {}

        sampling = num(metrics.get("model_time"))
        generation = num(metrics.get("generation_time"))
        eval_time = num(metrics.get("eval_time"))
        stage_a = num(timing.get("stage_a_repair"))
        stage_b = num(timing.get("stage_b_refine"))
        legal_total = num(timing.get("total"))
        legal_other = (legal_total - (stage_a or 0.0) - (stage_b or 0.0)
                       if legal_total is not None else None)
        wall_total = wall.get((profile, case, seed))
        overhead = (wall_total - (generation or 0.0) - (legal_total or 0.0)
                    if wall_total is not None else None)

        for label, value in (("sampling_s", sampling), ("generation_s", generation),
                             ("legalize_total_s", legal_total), ("wall_total_s", wall_total)):
            if value is None:
                problems.append(f"{stem}: {label} 取不到")
        if legal_other is not None and legal_other < -1e-6:
            problems.append(f"{stem}: stage_a+stage_b > legalize_total (差 {legal_other:.3f}s)")
        if overhead is not None and overhead < -1e-6:
            problems.append(f"{stem}: generation+legalize > wall_total (差 {overhead:.3f}s)")

        rows.append({
            "out_file": entry["out_file"], "profile": profile, "case": case, "seed": seed,
            "thermal_weight": t_label, "wirelength_weight": w_label,
            "sampling_s": r3(sampling), "generation_s": r3(generation),
            "inference_eval_s": r3(eval_time),
            "legalize_stage_a_s": r3(stage_a), "legalize_stage_b_s": r3(stage_b),
            "legalize_other_s": r3(legal_other), "legalize_total_s": r3(legal_total),
            "wall_total_s": r3(wall_total), "setup_overhead_s": r3(overhead),
        })

    print(f"例子数 = {len(rows)}  (manifest {len(manifest)} 行)")
    if problems:
        print(f"\n{len(problems)} 条问题:", file=sys.stderr)
        for item in problems[:20]:
            print(f"  - {item}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... 其余 {len(problems) - 20} 条略", file=sys.stderr)

    def agg(key):
        vals = [float(r[key]) for r in rows if r[key] != ""]
        if not vals:
            return None
        return min(vals), sum(vals) / len(vals), max(vals), sum(vals)

    print(f"\n{'列':<20}{'min':>9}{'mean':>9}{'max':>9}{'sum':>11}")
    for key in ("sampling_s", "generation_s", "inference_eval_s", "legalize_stage_a_s",
                "legalize_stage_b_s", "legalize_other_s", "legalize_total_s",
                "wall_total_s", "setup_overhead_s"):
        stat = agg(key)
        if stat:
            lo, mean, hi, total = stat
            print(f"{key:<20}{lo:>9.1f}{mean:>9.1f}{hi:>9.1f}{total:>11.0f}")

    if args.check:
        print("\n--check: 未写盘")
        return 1 if problems else 0

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {OUT_CSV}  ({len(rows)} 行)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
