#!/usr/bin/env python3
"""convert_illegal_pairs_to_format.py — 把 pareto 权重扫描里**本来非法、又跑过 legalizeLayout
之后合法了**的布局, 转成 eval_layout.py 能吃的格式。

背景
----
`pareto_weight_sweep_seed121_illegal_pairs/` 这一批的输入是 FM pareto 扫描里挑出来的
非法布局 (`legalized/*_legalized_illegal.json`)。这些布局随后被 `legalizeLayout` 处理,
产物是 `legalized/*_legalized_illegal_legal.json` —— 但它们**没有 hubump 字段**, 且
`connections` 里带着 EMIB* 冗余字段, 不能直接喂 `eval_layout.py`。

本脚本做的事就是 `convert_FM_layout_to_case.py` 的那五处映射 (坐标口径原样、rotation 校验、
hubump 从 benchmark 取、connections 取 benchmark 的 3 字段版本、chiplet 顺序对齐 benchmark),
**直接 import 它的 `convert_one`, 不重写一份** —— 那五个映射里有逐 chiplet 的交叉校验,
重新实现一遍等于把这些校验也重写一遍, 迟早分叉。

为什么不能直接跑 `convert_FM_layout_to_case.py`
--------------------------------------------
那个脚本是**写死的**: 目录固定在 `FM_result/fm-result/`, 且要求文件名形如
`<prefix>_best{T,WL}_seed<N>`、父目录名等于 prefix。这一批的文件名是
`<case>_candidate<NN>_thermal<W>_wirelength<WL>_seed121_legalized_illegal_legal.json`,
一个条件都不满足。所以这里只借用它的转换函数, 输入/输出的选取另起。

命名
----
输出 stem = `<case>_thermal<W>_wirelength<WL>_seed<seed>`, 与
`pareto_weight_sweep_seed121_legal_only_results/result.csv` 的 `case` 列**同一套口径** ——
两批要放在一起比 (见 pareto_front/plot_pareto3d.py), case 列对不上就没法比。

`candidate` 不进 stem: 这 107 个里 `<case>_thermal<W>_wirelength<WL>` 已经**两两不重复**
(脚本会断言), 加进去只是噪声; 它记在 manifest.csv 里。

注意 `eval_layout.py::_split_stem` 是 `rsplit("_seed", 1)`, 所以 stem 里只能有**一个**
`_seed`; 上面这个形式满足。

用法:
  python convert_illegal_pairs_to_format.py               # 全量
  python convert_illegal_pairs_to_format.py --dry-run     # 只报会写什么, 不落盘
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_FM_layout_to_case import convert_one, _resolve_benchmark  # noqa: E402

FM_ROOT = "/root/placement/flow_tap/resultEval/FM_result"
BATCH = "pareto_weight_sweep_seed121_illegal_pairs"
BATCH_DIR = os.path.join(FM_ROOT, BATCH)
OUT_DIR = os.path.join(FM_ROOT, f"{BATCH}_format")
MANIFEST_IN = os.path.join(BATCH_DIR, "manifest.json")
MANIFEST_OUT = os.path.join(BATCH_DIR, "manifest.csv")

# 与 legal_only 那批的 manifest.csv 同列, 外加 legalized_legal 三个字段说明来源
MANIFEST_COLUMNS = [
    "case", "candidate", "seed", "thermal_weight", "wirelength_weight",
    "legalization_status", "is_legal",
    "before_overlap_pairs", "after_overlap_pairs",
    "source_input", "legalized_json", "compare_png",
    "eval_input", "eval_stem",
]


def _fmt_weight(value) -> str:
    """0.02 -> '0p02', 0.2 -> '0p2', 0 -> '0'。与 legal_only 的 case 列口径一致。"""
    text = f"{float(value):g}"
    return text.replace(".", "p")


def make_stem(entry: dict) -> str:
    return (f"{entry['case']}_thermal{_fmt_weight(entry['thermal_weight'])}"
            f"_wirelength{_fmt_weight(entry['wirelength_weight'])}"
            f"_seed{entry['seed']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="只报会写什么, 不落盘")
    args = ap.parse_args()

    entries = json.load(open(MANIFEST_IN, encoding="utf-8"))
    print(f"manifest: {len(entries)} 条  ({MANIFEST_IN})")

    # stem 碰撞必须先查: eval_layout 按 stem 存缓存, 撞了就是静默复用别人的读数
    stems = [make_stem(e) for e in entries]
    dup = {s for s in stems if stems.count(s) > 1}
    if dup:
        raise SystemExit(f"stem 有重复, 会在 eval_cache 里互相覆盖: {sorted(dup)[:5]}")
    print(f"stem: {len(set(stems))} 个, 无重复")

    if not args.dry_run:
        os.makedirs(OUT_DIR, exist_ok=True)

    rows, n_ok, n_err, n_skip = [], 0, 0, 0
    for entry, stem in zip(entries, stems):
        legal = os.path.join(BATCH_DIR, entry["legalized_json"].replace(
            "_legalized_illegal.json", "_legalized_illegal_legal.json"))
        compare = legal.replace(".json", "_compare.png")
        out_path = os.path.join(OUT_DIR, f"{stem}.json")

        if not os.path.exists(legal):
            print(f"[skip] {stem}: 没有合法化产物 {os.path.basename(legal)}", file=sys.stderr)
            n_skip += 1
            continue

        status, is_legal = "ok", "true"
        try:
            if args.dry_run:
                json.load(open(legal, encoding="utf-8"))       # 至少确认能解析
            else:
                bench, _fb = _resolve_benchmark(entry["case"])
                convert_one(legal, bench, out_path)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[err ] {stem}: {type(e).__name__}: {e}", file=sys.stderr)
            status, is_legal = f"error: {type(e).__name__}", ""
            n_err += 1

        rows.append({
            "case": entry["case"],
            "candidate": entry["candidate"],
            "seed": entry["seed"],
            "thermal_weight": entry["thermal_weight"],
            "wirelength_weight": entry["wirelength_weight"],
            "legalization_status": status,
            "is_legal": is_legal,
            "before_overlap_pairs": entry.get("before_overlap_pairs", ""),
            "after_overlap_pairs": entry.get("after_overlap_pairs", ""),
            "source_input": entry.get("legalized_json", ""),
            "legalized_json": os.path.relpath(legal, BATCH_DIR),
            "compare_png": os.path.relpath(compare, BATCH_DIR) if os.path.exists(compare) else "",
            "eval_input": out_path,
            "eval_stem": stem,
        })

    if not args.dry_run:
        with open(MANIFEST_OUT, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            w.writeheader()
            w.writerows(rows)

    tag = "[dry-run] " if args.dry_run else ""
    print(f"\n{tag}转换 {n_ok} 个 -> {OUT_DIR}")
    if n_err or n_skip:
        print(f"{tag}失败 {n_err}, 跳过 {n_skip}")
    if not args.dry_run:
        print(f"{tag}manifest -> {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
