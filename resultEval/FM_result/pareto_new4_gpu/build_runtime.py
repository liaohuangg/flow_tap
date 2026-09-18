#!/usr/bin/env python3
"""build_runtime.py — 汇总**新增 4 个 case** 那批 FM 解的采样时间与合法化时间。

输出 ``runtime.csv``: 一行 = 一个解 (4 case x 1 profile x 5 seed = 20 行)。
列的定义、单位与 `<../pareto_table_profiles_5seed/runtime.csv>` **逐列一致**,
方便两张表拼起来读; 详细口径见那个目录里的 build_runtime.py, 简述:

    采样侧 (metrics_summary.json)
      sampling_s       = model_time       纯 FM 采样 (100 步 ODE + 热/线长 guidance)
      generation_s     = generation_time  预处理 + 采样 + 出图 (不含神经评测)
      inference_eval_s = eval_time        生成后的神经评测 (热/线长模型前向, 非物理评测)
    合法化侧 (legalized/<NN>_<case>_placement_legal_report.json 的 timing_s)
      legalize_stage_a_s = stage_a_repair  去重叠
      legalize_stage_b_s = stage_b_refine  双守卫下的线长优化
      legalize_total_s   = total
      legalize_other_s   = total - stage_a - stage_b   热点定位 + 神经前向 + 出图 + 写报告
    批次墙钟 (batch timing.csv 的 total_seconds, date 前后相减)
      wall_total_s       = 进程启动 -> 合法化结束
      setup_overhead_s   = wall_total - generation_s - legalize_total_s
      恒等式: wall_total_s == generation_s + legalize_total_s + setup_overhead_s

为什么单独一批
--------------------------------------------------------------------------
``pareto_table_profiles_5seed`` 那批 120 例只覆盖旧 benchmark 的 12 个 case
(Case6-10 + acend910/cpu-dram/hp11_m/multigpu/syn1/syn4/xerox8_m)。
hp6_m/hp8_m/xerox6_m/xerox7_m 是后来加进 benchmark 的, 当时**没有任何 GPU 计时**,
只有一份 CPU 计时 (resultEval/runtime/FMCPUTime.log)。本批就是给它们补的
**GPU 计时**: 用与旧 120 例**完全相同**的脚本/配置/权重表/随机 seed 跑
(CASES 收窄到 4 个 case, PROFILES=alpha_0p9, NUM_SEEDS=5,
THERMAL_SCALE=WIRELENGTH_SCALE=LEGALITY_SCALE=1 即权重表原值),
并且 **JOBS=1 独占显卡** 顺序执行 —— 跑之前确认过卡上没有别的 compute 进程,
所以行与行之间、以及与旧 120 例之间都可以直接横比。

canvas 倍数按各 case 自己的设定: hp6_m 3.5x / hp8_m 3.0x (ATPlace interposer 尺寸决定,
见 benchmark/cases_hubump/hp6_m.json 的 interposer_size), xerox6_m / xerox7_m 2.0x
(仓库统一的 util 0.50)。与 FMCPUTime.log 里同名的 4 行一致, 不是可比的密度。

用法:
    python build_runtime.py             # 生成/刷新同目录下的 runtime.csv
    python build_runtime.py --check     # 只校验不写盘
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../FM_result/pareto_new4_gpu
PROJECT = HERE.parents[2]                       # .../flow_tap
RUN_TAG = "gpu4_20260918"
BATCH_ROOT = PROJECT / "LayoutGenModel" / "logs" / "output" / "cases_hubump"
BATCH_DIR = BATCH_ROOT / f"selected-pareto-{RUN_TAG}"
MANIFEST = BATCH_DIR / "manifest.csv"
OUT_CSV = HERE / "runtime.csv"

COLUMNS = [
    "out_file", "profile", "case", "seed", "thermal_weight", "wirelength_weight",
    "sampling_s", "generation_s", "inference_eval_s",
    "legalize_stage_a_s", "legalize_stage_b_s", "legalize_other_s", "legalize_total_s",
    "wall_total_s", "setup_overhead_s",
]

EXPECTED_CASES = ["hp6_m", "hp8_m", "xerox6_m", "xerox7_m"]


def num(value):
    """取数值, 取不到返回 None (NaN/字符串/缺字段都算取不到)。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def r3(value):
    return "" if value is None else round(value, 3)


def to_local(raw: str) -> Path:
    """batch manifest 里记的是**采样时的 WSL 绝对路径** (``/mnt/d/...``)。

    本脚本可能在 Windows 侧用 python 跑 (只需要标准库), 那时 ``/mnt/d/...`` 打不开;
    在 Windows 上把 ``/mnt/<盘>/<rest>`` 翻成 ``<盘>:/<rest>``, 在 Linux 上原样返回。
    注意 run 目录是 batch 目录的**兄弟** (``selected-<profile>-<case>-t..-w..-<tag>``),
    不在 batch 目录里面, 所以只能从 manifest 的路径反推, 不能按 batch 目录拼。
    """
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw)
    if match and os.name == "nt":
        return Path(f"{match.group(1).upper()}:/{match.group(2)}")
    return Path(raw)


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
        profile, case, seed = entry["profile"], entry["case"], entry["seed"]
        # seed 目录直接从 manifest 记的 raw_json 反推, 不去拼 method 名 ——
        # 权重在 method 里被写成 0p02 这种标签, 手工拼字符串容易错。
        seed_dir = to_local(entry["raw_json"]).parent.parent
        if not seed_dir.is_dir():
            problems.append(f"{case}/seed{seed}: 源目录不存在 {seed_dir}")
            continue

        metrics_path = seed_dir / "metrics_summary.json"
        if not metrics_path.is_file():
            problems.append(f"{case}/seed{seed}: 缺 {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}

        reports = sorted((seed_dir / "legalized").glob("*_legal_report.json"))
        if not reports:
            problems.append(f"{case}/seed{seed}: 缺 legal report")
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
                problems.append(f"{case}/seed{seed}: {label} 取不到")
        if legal_other is not None and legal_other < -1e-6:
            problems.append(f"{case}/seed{seed}: stage_a+stage_b > legalize_total (差 {legal_other:.3f}s)")
        if overhead is not None and overhead < -1e-6:
            problems.append(f"{case}/seed{seed}: generation+legalize > wall_total (差 {overhead:.3f}s)")

        rows.append({
            "out_file": f"{case}_{profile}_t{entry['thermal_weight']}-w{entry['wirelength_weight']}"
                        f"_seed{seed}.json",
            "profile": profile, "case": case, "seed": seed,
            "thermal_weight": entry["thermal_weight"],
            "wirelength_weight": entry["wirelength_weight"],
            "sampling_s": r3(sampling), "generation_s": r3(generation),
            "inference_eval_s": r3(eval_time),
            "legalize_stage_a_s": r3(stage_a), "legalize_stage_b_s": r3(stage_b),
            "legalize_other_s": r3(legal_other), "legalize_total_s": r3(legal_total),
            "wall_total_s": r3(wall_total), "setup_overhead_s": r3(overhead),
        })

    rows.sort(key=lambda r: (EXPECTED_CASES.index(r["case"]) if r["case"] in EXPECTED_CASES else 99,
                             int(r["seed"])))
    seen = {r["case"] for r in rows}
    for case in EXPECTED_CASES:
        if case not in seen:
            problems.append(f"{case}: 一行都没有")

    print(f"例子数 = {len(rows)}  (manifest {len(manifest)} 行, 期望 "
          f"{len(EXPECTED_CASES)} case x 5 seed = {len(EXPECTED_CASES) * 5})")
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
