#!/usr/bin/env python3
"""report_gaps.py — 把每个 (case, 配置) 的**最优性间隙**汇总成 csv, 落到结果目录。

两个目标的 gap 口径**根本不同**, 所以列名不同、分两个文件, 不要合表:

  bbox  -> <out_dir>/gap_report.csv
      `mip_gap` 是**真 gap**: 目标里的 maxX/maxY/aspect 都是精确变量,
      W+H 和 |W-H| 就是字面量, 没有线性化误差。所以 gap=0 就是**证明最优**,
      `proven_optimal=True`。gap>0 时报的是真实可行解, 标成上界。

  wl    -> <out_dir>/gap_report.csv
      **没有**对真实线长的最优性证书, 一个都没有。真实线长是
      min over 可行流 of Σ f·d, 要给它下界得走 LP 对偶, 不是这条路。
      这里的 `surrogate_gap` 只是 MILP 对它自己那条线性化代理目标 Σλ·d
      的收敛间隙 —— 只能说明"固定这组 λ 时解到什么程度"。
      绝对量纲也对不上: hp6_m 的 obj_model 0.894 对应真实线长 2642.0 mm,
      代理值系统性偏大 (λ 是容量感知的贪心分配, 不是精确弧流量)。
      **不要把 surrogate_gap 读成线长的 gap。**

用法:
    python report_gaps.py                 # 两个目标都写
    python report_gaps.py --objective wl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ilp_core as C  # noqa: E402
from run_ilp_cases import OBJECTIVES  # noqa: E402

PROJECT = C.PROJECT

# 共有列: 两个目标的行都从 raw json 的同一份 schema 取
COMMON = [
    ("case", lambda d: d["case"]),
    ("seed", lambda d: d["seed"]),
    ("config", lambda d: d["config"]),
    ("n_chiplets", lambda d: d["n_chiplets"]),
    ("objective", lambda d: d["objective"]),
    ("profile", lambda d: d["profile"]),
    ("canvas_mm", lambda d: _r(d.get("canvas_mm"), 6)),
    ("canvas_overflow_mm", lambda d: _r(max(d["canvas_overflow_mm"]), 6)),
    ("bbox_wh_mm", lambda d: _r(d.get("bbox_wh_mm"), 4)),
    ("aspect_absdiff_mm", lambda d: _r(d.get("aspect_absdiff_mm"), 4)),
    ("bbox_area_mm2", lambda d: _r(d.get("bbox_area_mm2"), 3)),
    ("best_stage", lambda d: d.get("best_stage")),
    ("wall_s", lambda d: d.get("wall_s")),
    ("budget_s_total", lambda d: d.get("budget_s_total")),
    ("gap_target", lambda d: _gap_target(d)),
    ("gap_hit", lambda d: _gap_hit(d)),
    ("solved_at", lambda d: None),   # build() 里按 raw 文件 mtime 填
    ("proven_optimal", lambda d: d.get("proven_optimal")),
    ("mip_gap", lambda d: _r(d.get("mip_gap"), 8)),
    ("best_bound", lambda d: _r(d.get("best_bound"), 8)),
    ("obj_model", lambda d: _r(d.get("obj_model"), 8)),
]

COLUMNS = {
    # bbox: gap 是真 gap, 直接可用
    "bbox": COMMON,
    # wl: 代理 gap —— 列名点明它**不是**线长的最优性间隙
    "wl": [c for c in COMMON if c[0] != "mip_gap"] + [
        ("total_wirelength_mm", lambda d: _r(d.get("obj_eval_total_wl"), 4)),
        ("greedy_wirelength_mm", lambda d: _r(d.get("greedy_init_total_wl"), 4)),
        ("surrogate_gap", lambda d: _r(d.get("mip_gap"), 8)),
        ("surrogate_proven_optimal", lambda d: d.get("proven_optimal")),
    ],
}


def _r(v, nd):
    return None if v is None else round(float(v), nd)


def _gap_target(d: dict):
    """该目标要求的 gap 收敛阈值 (两个目标都是 0.05)。"""
    spec = OBJECTIVES.get(d.get("objective"))
    return None if spec is None else spec.get("gap")


def _gap_hit(d: dict):
    """达没达标。True/False, 缺 gap 或阈值时 None —— 不要拿 None 当 False 读。"""
    t, g = _gap_target(d), d.get("mip_gap")
    if t is None or g is None:
        return None
    return bool(g <= t)


def build(objective: str) -> list[dict]:
    spec = OBJECTIVES[objective]
    cols = COLUMNS[objective]
    rows = []
    for p in sorted(spec["raw_dir"].glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("objective") != objective:      # 目标之间不会混, 但防手滑
            continue
        row = {name: fn(d) for name, fn in cols}
        # 落盘时间戳: raw/ 里的文件是按 <case>_seed1.json 覆盖写的, 一个目录里
        # 可能同时躺着上一轮的旧结果, 光看数字分不出来。这列让它自己暴露。
        row["solved_at"] = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        rows.append(row)
    rows.sort(key=lambda r: (r["case"], r["seed"]))
    return rows


def summarize(objective: str, rows: list[dict]) -> None:
    if not rows:
        print(f"[{objective}] 没有 raw 记录")
        return
    gapcol = "mip_gap" if objective == "bbox" else "surrogate_gap"
    proven = sum(1 for r in rows if r.get("proven_optimal"))
    worst = max(rows, key=lambda r: r[gapcol] if r[gapcol] is not None else -1)
    label = "证明最优" if objective == "bbox" else "代理目标证明最优"
    hit = sum(1 for r in rows if r.get("gap_hit"))
    print(f"[{objective}] {len(rows)} 行, **达标 {hit}/{len(rows)}** (gap <= "
          f"{rows[0]['gap_target']}); {proven} 行 {label}; "
          f"最大 {gapcol} = {worst[gapcol]} @ {worst['case']}_seed{worst['seed']}")
    if objective == "wl":
        print("          注意: surrogate_gap 是 MILP 对线性化代理目标的收敛间隙, "
              "不是真实线长的最优性间隙 —— 线长没有最优性证书。")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", choices=["wl", "bbox", "both"], default="both")
    args = ap.parse_args()

    targets = ["wl", "bbox"] if args.objective == "both" else [args.objective]
    for obj in targets:
        spec = OBJECTIVES[obj]
        rows = build(obj)
        out = spec["out_dir"] / "gap_report.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[c[0] for c in COLUMNS[obj]])
            w.writeheader()
            w.writerows(rows)
        summarize(obj, rows)
        print(f"          -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
