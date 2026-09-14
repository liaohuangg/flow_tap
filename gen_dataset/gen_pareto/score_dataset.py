#!/usr/bin/env python3
"""score_dataset.py — 用两个冻结代理给任意布局数据集打分, 并支持配对比较。

指标 (逐 system 一行): n, m, max_c, mean_c, p99_c, wl, bbox_area, canvas_side, total_power

用法
----
    # 单份打分 -> stats.json
    python gen_dataset/gen_pareto/score_dataset.py --dataset <dir> --out /tmp/stats.json

    # 配对比较 (必须 system_id 一一对应) —— 头条指标是 **配对 Δ 的中位数**
    python gen_dataset/gen_pareto/score_dataset.py --compare A B

    # 跨数据集比较时**绝不能**比绝对值: 热代理的偏差是"每个 system 一个常数偏移"
    # (见 thermal-surrogate-validation), 只有配对 Δ 有意义。

必须用 chipdiffusion 环境:
    /root/anaconda3/envs/chipdiffusion/bin/python
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opt_dataset_lib as L  # noqa: E402

METRICS = ("max_c", "mean_c", "p99_c", "wl", "bbox_area", "canvas_side", "total_power")


# --------------------------------------------------------------------------------------
def files_of(target: Path) -> List[Path]:
    target = Path(target)
    if target.is_file():
        return [target]
    fs = sorted(target.glob("chiplet_dataset_*.json"),
                key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not fs:
        raise FileNotFoundError(f"{target} 下没有 chiplet_dataset_*.json")
    return fs


def score_target(target: Path, scorer: L.SurrogateScorer, limit: Optional[int] = None,
                 verbose: bool = True) -> Dict[str, dict]:
    """返回 {system_id: {metrics..., n, m}}。"""
    rows: Dict[str, dict] = {}
    t0 = time.time()
    for f in files_of(target):
        recs = L.load_chunk(f)
        sids = L.sorted_sids(recs)
        if limit is not None:
            sids = sids[: max(0, limit - len(rows))]
        if not sids:
            break
        records = [recs[s] for s in sids]
        wl = scorer.wl(records)
        th = scorer.thermal_one(records)
        for i, sid in enumerate(sids):
            r = records[i]
            fp = [L.footprint_of(c) for c in r["chiplets"]]
            rows[sid] = dict(th[i], wl=wl[i], bbox_area=L.bbox_area(fp),
                             canvas_side=L.canvas_side(fp),
                             total_power=sum(float(c["power"]) for c in r["chiplets"]),
                             n=len(fp), m=len(r.get("connections", [])))
        if verbose:
            print(f"  {f.name}: 累计 {len(rows)} systems  ({time.time() - t0:.0f}s)", flush=True)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _pct(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def summarize(rows: Dict[str, dict]) -> dict:
    out = {"count": len(rows)}
    for k in METRICS:
        v = [r[k] for r in rows.values() if k in r]
        if not v:
            continue
        out[k] = {"p5": _pct(v, 0.05), "p25": _pct(v, 0.25), "median": st.median(v),
                  "mean": st.mean(v), "p75": _pct(v, 0.75), "p95": _pct(v, 0.95),
                  "min": min(v), "max": max(v)}
    return out


def print_summary(title: str, s: dict) -> None:
    print(f"\n=== {title}  (n={s['count']}) ===")
    print(f"{'指标':<12s} {'p5':>11s} {'中位':>11s} {'均值':>11s} {'p95':>11s} {'max':>11s}")
    for k in METRICS:
        if k not in s:
            continue
        v = s[k]
        fmt = "{:11.2f}" if k.endswith("_c") else "{:11.0f}"
        print(f"{k:<12s}" + "".join(fmt.format(v[x]) for x in ("p5", "median", "mean", "p95", "max")))


# --------------------------------------------------------------------------------------
def compare(ra: Dict[str, dict], rb: Dict[str, dict], label_a: str, label_b: str) -> dict:
    """配对比较: 只保留两边都有的 system_id。Δ = B - A。"""
    common = sorted(set(ra) & set(rb), key=lambda s: int(s.split("_")[1]))
    if not common:
        raise SystemExit("两份数据集没有共同的 system_id —— 无法做配对比较")
    res = {"paired_n": len(common), "only_a": len(set(ra) - set(rb)), "only_b": len(set(rb) - set(ra))}
    print(f"\n=== 配对比较: {label_b} - {label_a}   (配对 {len(common)} 个 system, "
          f"仅A {res['only_a']}, 仅B {res['only_b']}) ===")
    print(f"{'指标':<12s} {'Δ中位':>11s} {'Δ均值':>11s} {'改善数':>16s} {'Δ绝对中位':>12s} {'相对中位':>10s}")
    for k in METRICS:
        d = [rb[s][k] - ra[s][k] for s in common if k in ra[s] and k in rb[s]]
        if not d:
            continue
        rel = [(rb[s][k] - ra[s][k]) / ra[s][k] for s in common
               if k in ra[s] and k in rb[s] and abs(ra[s][k]) > 1e-12]
        better = sum(1 for x in d if x < 0)
        res[k] = {"delta_median": st.median(d), "delta_mean": st.mean(d),
                  "n_better": better, "n_worse": sum(1 for x in d if x > 0), "n": len(d),
                  "rel_median": st.median(rel) if rel else float("nan")}
        print(f"{k:<12s} {st.median(d):11.3f} {st.mean(d):11.3f} "
              f"{f'{better}/{len(d)}':>16s} "
              f"{st.median([abs(x) for x in d]):12.3f} "
              f"{100 * st.median(rel) if rel else float('nan'):9.2f}%")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, help="数据集目录或单个 chiplet_dataset_k.json")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"),
                    help="配对比较两份数据集 (B - A)")
    ap.add_argument("--limit", type=int, default=None, help="每份最多打分多少个 system")
    ap.add_argument("--out", type=Path, default=None, help="把逐 system 行 + 汇总写成 JSON")
    ap.add_argument("--csv", type=Path, default=None, help="把逐 system 行写成 CSV")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.dataset and not args.compare:
        ap.error("需要 --dataset 或 --compare")

    scorer = L.SurrogateScorer(device=args.device)

    def run(tag: str, target: Path) -> Dict[str, dict]:
        print(f"[score] {tag}: {target}", flush=True)
        rows = score_target(target, scorer, limit=args.limit)
        s = summarize(rows)
        print_summary(tag, s)
        return rows

    if args.compare:
        ra = run("A", args.compare[0])
        rb = run("B", args.compare[1])
        res = compare(ra, rb, str(args.compare[0]), str(args.compare[1]))
        if args.out:
            L.write_json_atomic(args.out, {"a": summarize(ra), "b": summarize(rb), "compare": res},
                                indent=2)
            print(f"\n[score] 写出 {args.out}")
        if args.csv:
            _write_csv(args.csv, [("A", ra), ("B", rb)], args.compare)
        return

    rows = run("dataset", args.dataset)
    if args.out:
        L.write_json_atomic(args.out, {"summary": summarize(rows),
                                       "per_system": rows}, indent=2)
        print(f"[score] 写出 {args.out}")
    if args.csv:
        _write_csv(args.csv, [(str(args.dataset), rows)], [])


def _write_csv(path: Path, groups, names) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["group", "system_id", "n", "m"] + list(METRICS)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for tag, rows in groups:
            for sid in sorted(rows, key=lambda s: int(s.split("_")[1])):
                r = rows[sid]
                w.writerow([tag, sid] + [r.get(k, "") for k in cols[2:]])
    print(f"[score] 写出 {path}")


if __name__ == "__main__":
    main()
