#!/usr/bin/env python3
"""把本方法 ("Ours") 每个 case 的解集整理成 50 个, 与 AT 的 50 个解对等。

背景: AT_result/50set.csv 每个 case 有 50 个解 (wlsweep), 而本方法的权重扫描
(pareto_weight_sweep_seed121_legal_only_results/result.csv) 在 acend910 / cpu-dram
上只有 38 / 48 个解, 图上图例就成了 "Ours (20 of 38 on the front)" 对比
"AT (21 of 50 on the front)", 采样数不对等, 看着不公平。

做法不是"只补不删", 而是**从候选里挑最好的 50 个**: 权重扫描的现有解和 seed
扫描的候选取并集, 按下面的规则排序, 取前 50 —— 排在 50 名之外的现有解会被换掉。

候选来源 (都是 seed 扫描, case 列没有参数尾缀):
  FM_result/result_old.csv   13 解/case
  FM_result/result.csv        5 解/case
acend910: 现有 38 + 候选 18 = 56 -> 取 50; cpu-dram: 48 + 18 = 66 -> 取 50。

选取规则 (与用户确认):
  0. 硬约束: 外接框面积超过中介层的解是不可行解, 一律先去掉 (见
     interposer_area_mm2)。AT 的 50 个解在每个 case 里都恰好压在限内, 所以
     这条只淘汰本方法自己的解。
  1. 剩下的里, 对 "AT + 现有 Ours" 非支配的排前面 —— 这些是真能改善前沿的;
  2. 其余按距全局前沿的距离排。距离在三目标各自 min-max 归一化后算欧氏
     距离, 取到前沿最近点的距离。不引入主观权重。
  3. 同距离时按 (来源, seed) 定序, 结果可复现。

  注: 支配性是对 "已有解集" (AT + 全部现有 Ours) 判的, 不含其它候选 —— 候选
  本身不属于该方法, 除非被选中。这样规则不依赖候选之间的相互支配关系。
  非支配的现有解一定排在前面, 所以换掉的只会是原本就被支配、且离前沿最远的
  那些, 前沿本身不会被这次替换削掉。

产物:
  FM_result/ours_50set.csv   本方法的解集 (被处理的 case 恰好 50 个, 其余 case 原样),
                             case 列形如 <case>_seed<NN> 或原参数尾缀,
                             多一列 source (sweep / result_old / result)
控制台打印每个 case 换掉了谁、补进了谁。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

RESULT_EVAL = Path("/root/placement/flow_tap/resultEval")
CASES_50SET = Path("/root/placement/flow_tap/baseline/ATPlace_pub/cases_50set")
AT_CSV = RESULT_EVAL / "AT_result" / "50set.csv"
SWEEP_CSV = (RESULT_EVAL / "FM_result" / "pareto_weight_sweep_seed121_legal_only_results"
             / "result.csv")
SUPPLEMENT_CSVS = [
    ("result_old", RESULT_EVAL / "FM_result" / "result_old.csv"),
    ("result", RESULT_EVAL / "FM_result" / "result.csv"),
]
OUT_CSV = RESULT_EVAL / "FM_result" / "ours_50set.csv"

TARGET = 50  # 每个 case 本方法要有的解数 (与 AT 对齐)
CASES = ["acend910", "cpu-dram", "hp11_m", "xerox8_m"]
OBJECTIVES = ["total_wirelength_mm", "bbox_area_mm2", "max_temp_C"]


def interposer_area_mm2(case: str) -> float | None:
    """case 里中介层能容纳的最大面积 (mm^2)。

    Thermal-aware.json 的 interposer_size 是边长 (um), 外接框面积是 mm^2,
    所以边长先 /1000 换成 mm 再平方。布局的外接框一旦超出中介层就是不可行解。
    单位对得上: 每个 case 里 AT 的 50 个解最大面积都恰好压在中介层面积之下
    (acend910 2009.1<2018.1, cpu-dram 1288.4<1295.6, hp11_m 1880.1<1888.8,
    xerox8_m 2642.3<2652.6)。
    """
    path = CASES_50SET / f"{case}_bump" / "Thermal-aware.json"
    if not path.exists():
        print(f"  警告: 找不到 {path}, 这个 case 不做面积过滤")
        return None
    size = json.loads(path.read_text(encoding="utf-8")).get("interposer_size")
    if size is None:
        return None
    w, h = (size, size) if isinstance(size, (int, float)) else size
    return (w / 1000.0) * (h / 1000.0)


def case_of(label: str) -> str:
    """两边 csv 的 case 列都带尾缀, 剥回 case 名。

    AT:   <case>_wl<NN>                    -> <case>
    FM:   <case>_thermal<X>_wirelength<Y>  -> <case>
    补充: <case>_seed<NN>                  -> <case>
    """
    return label.split("_thermal", 1)[0].split("_wl", 1)[0].split("_seed", 1)[0]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]


def vals_of(rows: list[dict]) -> np.ndarray:
    return np.array([[float(r[c]) for c in OBJECTIVES] for r in rows], dtype=float)


def pareto_mask(vals: np.ndarray) -> np.ndarray:
    """True = 非支配解 (三项都不劣于它、且至少一项更优的解不存在)。"""
    n = len(vals)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if np.any(np.all(vals <= vals[i], axis=1) & np.any(vals < vals[i], axis=1)):
            keep[i] = False
    return keep


def area_of(row: dict) -> float:
    return float(row["bbox_area_mm2"])


def select_case(case: str, at_rows: list[dict], existing: list[dict],
                candidates: list[tuple[str, dict]], target: int,
                area_cap: float | None):
    """从 现有解 + 候选 里挑 target 个最好的。返回 (选中的 (source, row) 列表, 统计)。

    先按硬约束筛: 外接框超过中介层面积的解一律去掉 (不可行), 再在剩下的里挑。
    """
    over_existing = [r for r in existing if area_cap and area_of(r) > area_cap]
    over_cand = [(s, r) for s, r in candidates if area_cap and area_of(r) > area_cap]
    existing = [r for r in existing if not (area_cap and area_of(r) > area_cap)]
    candidates = [(s, r) for s, r in candidates if not (area_cap and area_of(r) > area_cap)]

    base_v = vals_of(at_rows + existing)
    pool = [("sweep", r) for r in existing] + candidates
    pool_v = vals_of([r for _, r in pool])

    # 归一化基准和前沿都取 "AT + 现有 Ours", 三个量级差很大的目标才可比
    lo, hi = base_v.min(axis=0), base_v.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    front = ((base_v - lo) / span)[pareto_mask(base_v)]
    npool = (pool_v - lo) / span

    dist = np.array([np.linalg.norm(p - front, axis=1).min() for p in npool])
    dominated = np.array([
        bool(np.any(np.all(base_v <= p, axis=1) & np.any(base_v < p, axis=1)))
        for p in pool_v
    ])

    # 非支配优先, 其次按离前沿的距离; 同距离按 (来源, seed) 定序保证可复现
    order = sorted(range(len(pool)),
                   key=lambda i: (bool(dominated[i]), dist[i], pool[i][0],
                                  str(pool[i][1].get("seed", ""))))
    chosen = order[:target]
    dropped = [i for i in order[target:]]

    n_nd = int((~dominated[chosen]).sum())
    print(f"=== {case}: AT {len(at_rows)} 解, 现有 Ours {len(existing)} + 候选 "
          f"{len(candidates)} = {len(pool)}, 取 {len(chosen)} 个 "
          f"(其中对 AT+现有Ours 非支配 {n_nd} 个)")
    if area_cap:
        at_over = [r for r in at_rows if area_of(r) > area_cap]
        print(f"  面积上限 {area_cap:.1f}mm2: 淘汰现有解 {len(over_existing)} 个, "
              f"淘汰候选 {len(over_cand)} 个"
              + (f"; AT 也有 {len(at_over)} 个超限(!)" if at_over else "; AT 全部在限内"))
        for r in over_existing:
            print(f"    x 超中介层 {r['case']:34s} area={area_of(r):8.1f} "
                  f"wl={float(r['total_wirelength_mm']):9.0f} "
                  f"T={float(r['max_temp_C']):6.2f}")
    if len(chosen) < target:
        print(f"  ⚠ 可用解只有 {len(chosen)} 个, 凑不到 {target}")
    added = [i for i in chosen if pool[i][0] != "sweep"]
    removed = [i for i in dropped if pool[i][0] == "sweep"]
    keep_sweep = [i for i in chosen if pool[i][0] == "sweep"]
    print(f"  留下的原有解 {len(keep_sweep)}/{len(existing)}, "
          f"换进来 {len(added)} 个候选, 换掉 {len(removed)} 个原有解")
    for i in added:
        src, row = pool[i]
        print(f"    + {src:10s} seed={row['seed']:>4s} "
              f"wl={float(row['total_wirelength_mm']):9.0f} "
              f"area={float(row['bbox_area_mm2']):7.1f} "
              f"T={float(row['max_temp_C']):6.2f}  "
              f"[{'非支配' if not dominated[i] else '被支配'}, 距前沿 {dist[i]:.4f}]")
    for i in removed:
        _, row = pool[i]
        print(f"    - 换掉 {row['case']:34s} "
              f"wl={float(row['total_wirelength_mm']):9.0f} "
              f"area={float(row['bbox_area_mm2']):7.1f} "
              f"T={float(row['max_temp_C']):6.2f}  [距前沿 {dist[i]:.4f}]")

    final = [pool[i] for i in chosen]
    before = pareto_mask(vals_of(at_rows + existing)).sum()
    after = pareto_mask(vals_of(at_rows + [r for _, r in final])).sum()
    ours_b = pareto_mask(vals_of(existing)).sum()
    ours_a = pareto_mask(vals_of([r for _, r in final])).sum()
    print(f"  本方法前沿 {ours_b} -> {ours_a}, 全局前沿 {before} -> {after}")
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=",".join(CASES),
                    help=f"要整理的 case, 逗号分隔 (默认 {','.join(CASES)})")
    ap.add_argument("--target", type=int, default=TARGET,
                    help=f"每个 case 整理成多少个解 (默认 {TARGET})")
    ap.add_argument("--no-area-filter", action="store_true",
                    help="不按中介层面积淘汰不可行解 (默认会淘汰)")
    args = ap.parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    at_all = read_rows(AT_CSV)
    sweep_all = read_rows(SWEEP_CSV)
    supp_all = [(src, r) for src, path in SUPPLEMENT_CSVS for r in read_rows(path)]

    replaced: dict[str, list[tuple[str, dict]]] = {}
    for case in cases:
        at_rows = [r for r in at_all if case_of(r["case"]) == case]
        existing = [r for r in sweep_all if case_of(r["case"]) == case]
        if not at_rows:
            print(f"=== {case}: AT 里没有这个 case, 跳过")
            continue
        candidates = [(src, dict(r, case=f"{r['case']}_seed{r['seed']}"))
                      for src, r in supp_all if case_of(r["case"]) == case]
        cap = None if args.no_area_filter else interposer_area_mm2(case)
        if len(existing) <= args.target and len(existing) + len(candidates) <= args.target:
            print(f"=== {case}: 现有 {len(existing)} 解, 候选 {len(candidates)} 个, "
                  f"凑不到 {args.target}, 原样保留")
            continue
        replaced[case] = select_case(case, at_rows, existing, candidates, args.target, cap)
        print()

    # ---- 写出整理后的本方法解集 -------------------------------------------
    fields = list(read_rows(SWEEP_CSV)[0].keys()) + ["source"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sweep_all:
            case = case_of(r["case"])
            if case in replaced:
                continue  # 这个 case 由下面的整理结果统一写出
            w.writerow(dict(r, source="sweep"))
        for case, rows in replaced.items():
            for src, r in rows:
                w.writerow(dict(r, source=src))
    print(f"写出 {OUT_CSV}")
    counts = {}
    for r in read_rows(OUT_CSV):
        counts[case_of(r["case"])] = counts.get(case_of(r["case"]), 0) + 1
    print("  每个 case 的本方法解数:")
    for k in sorted(counts):
        mark = "  <- 已整理" if k in replaced else ""
        print(f"    {k:12s} {counts[k]}{mark}")


if __name__ == "__main__":
    main()
