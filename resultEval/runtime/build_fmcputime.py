#!/usr/bin/env python3
"""build_fmcputime.py — FM(本方法)**纯 CPU** 按 (case, profile) 的阶段耗时表。

FMtime.csv 的 CPU 版本。列口径与它保持一致，便于两张表并排看:

    total = sampling + stage_a + stage_b

采样侧
  sampling_s        = ``metrics_summary.json`` 的 ``model_time`` —— 纯 FM 采样
                      (100 步 ODE + 热/线长 guidance)。就是 timing.csv 的
                      ``sample_loop_seconds``。
                      **不含** preprocess / postsample / 进程启动 —— 那些在
                      timing.csv 的 ``wall_inference_seconds`` 里, 不在本表口径中。
合法化侧 (``*_legal_report.json`` 的 ``timing_s``)
  stage_a_s         = ``stage_a_repair``
  stage_b_s         = ``stage_b_refine``
  other_s           = ``legalize_total - stage_a - stage_b``
                      热点定位 + 未归因的收尾。**不计入** total, 单列备查 —— 同 FMtime。
  legalize_total_s  = ``timing_s.total``
  total_s           = sampling + stage_a + stage_b

与 FMtime.csv 的差别, 读表前必须知道
--------------------------------------------------------------------------
1. **单 seed**。CPU 跑一轮的成本决定了一个 (case, profile) 只能负担一次采样,
   所以 n_seeds=1, 且 total_std_s 留空 —— 不是 0, 是"没测"。留 0 会把"未测量"
   说成"无波动"。
   实测同一输入重复跑的采样墙钟可以差 37~47% (产物逐位相同, 是机器层面波动),
   所以**本表内小于约 40% 的行间差异不要当结论**; 结构性的量 (Stage B 占合法化
   ~94%) 不受影响。
2. **画布不统一**。见下面的 canvas_multiple 列。hp6_m/hp8_m 用了 3.5x/3x 的松
   画布 (那颗 33mm 长条 die 在 2x 下摆不下), 其余 14 个是仓库统一的 2x。松画布 =
   更松的装填问题 = 计时偏快、质量偏好, 所以这两行的**计时与质量都不能与其余行
   横比**, 只能这两个互相比。

用法:
    python build_fmcputime.py           # 写同目录 FMCPUTime.csv
    python build_fmcputime.py --check   # 只打印, 不写盘
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../resultEval/runtime
FM_ROOT = HERE.parent.parent / "LayoutGenModel" / "logs" / "output" / "cases_hubump"

# 源批次, 显式列出 —— 不做"自动挑最新"的猜测。两个批次的画布口径就不同
# (cpu12 全程 2x; new4b 引入按 case 覆盖), 自动发现会在不声不响之间把它们混起来。
# 换批次时改这里。
SOURCES = ["selected-pareto-cpu12", "selected-pareto-new4b"]

PROFILE_ORDER = ["alpha_0p1", "alpha_0p9"]


def natkey(case: str):
    """Case6 < Case9 < Case10 —— 纯字典序会把 Case10 排到 Case6 前面。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", case)]


def num(row, key, default=None):
    """空字段 = 该 stage 没产出报告, 按 None 往下传, 不要当成 0。"""
    raw = (row.get(key) or "").strip()
    if not raw:
        return default
    return float(raw)


def r3(value):
    return "" if value is None else round(value, 3)


def load_rows():
    rows = []
    seen = {}
    for tag in SOURCES:
        path = FM_ROOT / tag / "timing.csv"
        if not path.is_file():
            raise SystemExit(f"缺输入: {path}")
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                # 失败行照样在 timing.csv 里 (设计如此, 失败也要留痕), 但进不了本表。
                if row["inference_status"] != "ok" or row["legalization_status"] != "ok":
                    continue
                key = (row["case"], row["profile"])
                if key in seen:
                    raise SystemExit(
                        f"{key[0]}/{key[1]} 同时出现在 {seen[key]} 和 {tag} —— "
                        f"两个批次的配置可能不同, 不能默默取一个。"
                    )
                seen[key] = tag
                # cpu12 那批跑在 canvas_multiple 列存在之前; 它当时用的是全局 0.50,
                # 即 2x。
                row["canvas_multiple"] = (row.get("canvas_multiple") or "").strip() or "2"
                row["source"] = tag
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只打印, 不写 FMCPUTime.csv")
    args = ap.parse_args()

    src = load_rows()
    problems = []

    keys = sorted({(r["case"], r["profile"]) for r in src},
                  key=lambda k: (natkey(k[0]),
                                 PROFILE_ORDER.index(k[1]) if k[1] in PROFILE_ORDER else 99,
                                 k[1]))

    out_rows = []
    for case, profile in keys:
        group = sorted((r for r in src if r["case"] == case and r["profile"] == profile),
                       key=lambda r: int(r["seed"]))
        n = len(group)

        sampling = [num(r, "sample_loop_seconds") for r in group]
        stage_a = [num(r, "legal_stage_a_seconds") for r in group]
        stage_b = [num(r, "legal_stage_b_seconds") for r in group]
        legal_tot = [num(r, "legal_internal_total_seconds") for r in group]
        if any(v is None for v in sampling + stage_a + stage_b + legal_tot):
            problems.append(f"{case}/{profile}: 有 stage 字段为空")
            continue
        # other = 合法化总时长里没被 A/B 认领的部分 (热点定位 + 收尾)。
        other = [t - a - b for t, a, b in zip(legal_tot, stage_a, stage_b)]
        # total 逐 seed 相加, 同 FMtime 的口径: other 不计入。
        total = [s + a + b for s, a, b in zip(sampling, stage_a, stage_b)]

        for label, vals in (("stage_a", stage_a), ("stage_b", stage_b), ("other", other)):
            if min(vals) < -1e-6:
                problems.append(f"{case}/{profile}: {label} 出现负值 {min(vals):.3f}s")

        canvases = sorted({r["canvas_multiple"] for r in group})
        if len(canvases) > 1:
            problems.append(f"{case}/{profile}: 组内画布不一致 {canvases}")

        row = {
            "case": case,
            "profile": profile,
            "n_seeds": n,
            "seeds": ";".join(r["seed"] for r in group),
            "sampling_mean_s": r3(st.mean(sampling)),
            "stage_a_mean_s": r3(st.mean(stage_a)),
            "stage_b_mean_s": r3(st.mean(stage_b)),
            "other_mean_s": r3(st.mean(other)),
            "legalize_total_mean_s": r3(st.mean(legal_tot)),
            "total_mean_s": r3(st.mean(total)),
            # n=1 时留空而不是 0.0 —— 0 会被读成"没有波动"。
            "total_std_s": r3(st.stdev(total)) if n > 1 else "",
            "total_min_s": r3(min(total)),
            "total_max_s": r3(max(total)),
        }
        for i, r in enumerate(group, start=1):
            row[f"seed{i}_sampling_s"] = r3(num(r, "sample_loop_seconds"))
            row[f"seed{i}_stage_a_s"] = r3(num(r, "legal_stage_a_seconds"))
            row[f"seed{i}_stage_b_s"] = r3(num(r, "legal_stage_b_seconds"))
            row[f"seed{i}_total_s"] = r3(
                num(r, "sample_loop_seconds")
                + num(r, "legal_stage_a_seconds")
                + num(r, "legal_stage_b_seconds"))
        # 追加在末尾而不是插在中间, 这样前 13 列与 FMtime.csv 逐列对齐。
        row["canvas_multiple"] = canvases[0] if len(canvases) == 1 else "|".join(canvases)
        row["source"] = ";".join(sorted({r["source"] for r in group}))
        out_rows.append(row)

    columns = list(out_rows[0].keys())

    print(f"{'case':12s}{'profile':10s}{'canvas':>7}{'samp':>9}{'A':>8}{'B':>8}"
          f"{'other':>8}{'total':>9}")
    for row in out_rows:
        print(f"{row['case']:12s}{row['profile']:10s}{row['canvas_multiple'] + 'x':>7}"
              f"{row['sampling_mean_s']:>9.3f}{row['stage_a_mean_s']:>8.3f}"
              f"{row['stage_b_mean_s']:>8.3f}{row['other_mean_s']:>8.3f}"
              f"{row['total_mean_s']:>9.3f}")

    print(f"\n行数 = {len(out_rows)}  (源 {len(src)} 行 / {len(keys)} 组)")
    loose = [r["case"] for r in out_rows if r["canvas_multiple"] != "2"]
    if loose:
        print(f"非 2x 画布, 计时与其余行不可横比: {', '.join(loose)}")

    if problems:
        print(f"\n{len(problems)} 条问题:")
        for item in problems:
            print(f"  - {item}")

    if args.check:
        print("\n--check: 未写盘")
        return 1 if problems else 0

    out_csv = HERE / "FMCPUTime.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\n-> {out_csv}  ({len(out_rows)} 行)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
