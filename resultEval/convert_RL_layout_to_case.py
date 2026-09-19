#!/usr/bin/env python3
"""convert_RL_layout_to_case.py — 把 RL 的最优布局转成 placement_dataset 的 chiplet 格式。

目标格式 (Dataset/dataset/README.md, `placement_dataset/chiplet_dataset_*.json` 里的单条记录):

    {
      "system_id": "Case6_seed1",
      "chiplets": [
        { "name": "CPU2_0", "x-position": 0.045, "y-position": 29.045,
          "width": 25.91, "height": 24.91, "rotation": 0,
          "power": 65.0, "hubump": 0.045 },
        ...
      ],
      "connections": [ { "node1": "A", "node2": "B", "wireCount": 512 }, ... ]
    }

输入:  RL_result/runs/shard<N>/<prefix>_seed<M>/
         best_cost_summary.json  -> "json" 字段指向该 seed 的最优布局文件
         results/top_layouts/*.metrics.json -> 该 run 存下来的全部候选解
       每个 seed 出**一个**解, 口径 = "cost (rlplanner_cost) 最小、且外接框不超
       该 case 真实中介层面积"的那个, 见 pick_layout()。
输出:  RL_result/format_result/<prefix>_seed<M>.json

RL 的布局 json 本来就与 README 同口径 (mm, 本体左下角 + 本体尺寸), 所以字段基本是原样搬运,
只去掉 README 没有的 `occupied-*` (那 4 个是 RL 自己的 footprint 包络, = 本体 ± hubump):

    x-position / y-position  本体左下角 (mm) —— RL 的 occupied-x/y = 本体坐标 - hubump,
                             已经验证过, 所以两者同口径, 直接取本体那一组。
    width / height           本体尺寸 (mm), **已按 rotation 摆放过**, 不能再交换
                             (README 第五节第 2 条: 按 rotation 交换会造出 1315/12242 重叠)。
    rotation                 RL 与 README 一样只取 0/1, 直接搬。
    hubump                   DUMMY_* 填充块 hubump=0, RL 会省略该字段, 缺省按 0 处理。
    connections              RL 里 connections 是 **空数组**, 整段从前缀对应的
                             benchmark/cases_hubump/<prefix>.json 取, 只保留 README 的 3 个字段
                             (node1/node2/wireCount), benchmark 里的 EMIB* 按 README 约定丢掉。

所有字段都与 benchmark 逐 chiplet 交叉校验, 不一致直接抛错。chiplet 顺序按 benchmark 排,
便于与 AT 的输出直接对比。

用法:
  python convert_RL_layout_to_case.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

RL_ROOT = "/root/placement/flow_tap/resultEval/RL_result"
RUNS_DIR = os.path.join(RL_ROOT, "runs")
OUT_DIR = os.path.join(RL_ROOT, "format_result")
BENCH_DIR = "/root/placement/flow_tap/benchmark/cases_hubump"

# 中介层面积上限的来源, 与 pareto_front/build_case_solution.py 的 cap_of() 同口径:
# 数据本身是 bump 版布局, 所以优先 _bump 目录, 没有才回退到不带后缀的目录。
CASES_BASE = "/root/placement/flow_tap/baseline/ATPlace_pub/cases"
CAP_TOL = 1e-6          # 与 build_case_solution.over_cap 一致: 贴边可行解不算超限

# RL 的 json 字段指向 baseline/RL/... ; 若该路径不存在, 用同后缀在 RL_result 下回退查找
BASELINE_PREFIX = "/root/placement/flow_tap/baseline/RL/"
FALLBACK_PREFIX = os.path.join(RL_ROOT, "")
TOL = 1e-6


def _resolve_layout_path(raw_path: str) -> str:
    """best_cost_summary.json 的 json 字段 -> 实际可读的布局文件路径。"""
    if os.path.exists(raw_path):
        return raw_path
    if raw_path.startswith(BASELINE_PREFIX):
        alt = os.path.join(FALLBACK_PREFIX, raw_path[len(BASELINE_PREFIX):])
        if os.path.exists(alt):
            return alt
    return raw_path  # 交给调用方报错


def cap_area(prefix: str) -> float | None:
    """该 case 的中介层面积上限 (mm^2); 找不到 Thermal-aware.json 时返回 None。"""
    for d in (prefix + "_bump", prefix):
        p = os.path.join(CASES_BASE, d, "Thermal-aware.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                s = json.load(f)["interposer_size"]
            return (float(s[0]) / 1000.0) * (float(s[1]) / 1000.0)
    return None


def _candidates(seed_dir: str) -> list[tuple[float, float, str]]:
    """该 run 存下来的全部候选解 -> [(cost, 外接框面积, 布局路径), ...]。

    top_layouts/ 里每个解有 .json (布局) 和 .metrics.json (指标), 这里只认成对存在的,
    指标取 metrics.json —— 它的 bounding_rect_area 与 eval_layout.py 算的外接框逐位相等
    (含 hubump 的轴对齐外接矩形), 所以可以直接拿来当面积上限的判据。
    """
    out = []
    for mp in sorted(glob.glob(os.path.join(seed_dir, "results", "top_layouts", "*.metrics.json"))):
        jp = mp[: -len(".metrics.json")] + ".json"
        if not os.path.exists(jp):
            continue
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("rlplanner_cost") is None or m.get("bounding_rect_area") is None:
            continue
        out.append((float(m["rlplanner_cost"]), float(m["bounding_rect_area"]), jp))
    return out


def pick_layout(seed_dir: str, summary: dict, cap: float | None) -> tuple[str, str]:
    """该 seed 出一个解: cost 最小、且外接框不超中介层。返回 (布局路径, 说明)。

    第一顺位仍是 best_cost_summary.json 指的那个 —— 它就是该 run 内 cost 最小的解,
    旧 case 全部走这一条。只有当它超限时才退到 results/top_layouts/ 里限内 cost 最小的
    那个, 口径不变 (还是 cost 最小), 只是把搜索范围从"整个 run"收窄到"限内的解"，
    而不是换成别的目标 (reward / 线长 / 温度都不参与挑解)。

    会走到第二顺位, 是因为那批 RL 是用 50 mm 的固定画布跑的 (launcher 没按 case 传
    interposer 尺寸), 画布比真实中介层大时 RL 从没为超限付过代价, cost 最小的解往往
    放不进真实中介层 —— hp6_m (真实 33.28 mm) 5 个 seed 里有 4 个是这样。
    """
    raw = summary.get("json")
    best = _resolve_layout_path(raw) if raw else ""
    best_area = (summary.get("metrics") or {}).get("bbox_area")
    if not best or not os.path.exists(best):
        return best, "best_cost_summary 指的布局不存在"
    if cap is None or best_area is None or best_area <= cap * (1 + CAP_TOL):
        return best, f"best_cost_summary (area={best_area if best_area is None else round(best_area, 2)})"

    feas = [c for c in _candidates(seed_dir) if c[1] <= cap * (1 + CAP_TOL)]
    if not feas:
        return best, (f"best_cost_summary 超限 (area={best_area:.2f} > cap={cap:.2f}) "
                      f"且 {len(_candidates(seed_dir))} 个候选解无一限内, 保持原解 (下游会丢弃)")
    cost, area, path = min(feas, key=lambda c: (c[0], c[1]))
    return path, (f"best_cost_summary 超限 ({best_area:.2f} > cap={cap:.2f}) -> "
                  f"限内 cost 最小候选 ({len(feas)}/{len(_candidates(seed_dir))} 个限内, "
                  f"cost={cost:.4f}, area={area:.2f})")


def convert_one(layout_path: str, benchmark_path: str, out_path: str) -> dict:
    src = json.load(open(layout_path, encoding="utf-8"))
    bench = json.load(open(benchmark_path, encoding="utf-8"))
    bench_by_name = {c["name"]: c for c in bench["chiplets"]}

    chiplets = []
    for c in src["chiplets"]:
        name = c["name"]
        b = bench_by_name.get(name)
        if b is None:
            raise KeyError(f"{layout_path}: chiplet {name!r} 在 {benchmark_path} 中不存在")

        # DUMMY_* 填充块没有 bump 环 (hubump=0), RL 会省略 hubump / occupied-* 字段
        hubump = float(c.get("hubump", 0.0))
        width, height = float(c["width"]), float(c["height"])
        x_pos, y_pos = float(c["x-position"]), float(c["y-position"])
        rotation = int(c.get("rotation", 0))

        if rotation not in (0, 1):
            raise ValueError(f"{layout_path}: chiplet {name!r} 的 rotation={rotation!r} 不是 0/1")
        if width <= 0 or height <= 0:
            raise ValueError(f"{layout_path}: chiplet {name!r} 尺寸非正 ({width}, {height})")

        # --- 与 benchmark 交叉校验 ---
        fp_w, fp_h = width + 2.0 * hubump, height + 2.0 * hubump
        want = sorted([b["footprint_w"], b["footprint_h"]])
        if abs(sorted([fp_w, fp_h])[0] - want[0]) > TOL or abs(sorted([fp_w, fp_h])[1] - want[1]) > TOL:
            raise ValueError(f"{layout_path}: chiplet {name!r} 的 footprint {sorted([fp_w, fp_h])} "
                             f"与 benchmark {want} 不符")
        for label, got, want_v in (
            ("hubump", hubump, b["hubump"]),
            ("power", float(c["power"]), b["power"]),
        ):
            if abs(got - want_v) > TOL:
                raise ValueError(f"{layout_path}: chiplet {name!r} 的 {label} 与 benchmark 不符: "
                                 f"{got} != {want_v}")

        # RL 自己的 footprint 包络自洽性检查 (DUMMY 没有这组字段, 跳过)
        if "occupied-x-position" in c:
            for label, got, want_v in (
                ("occupied-x", float(c["occupied-x-position"]), x_pos - hubump),
                ("occupied-y", float(c["occupied-y-position"]), y_pos - hubump),
                ("occupied-width", float(c["occupied-width"]), fp_w),
                ("occupied-height", float(c["occupied-height"]), fp_h),
            ):
                if abs(got - want_v) > 1e-6:
                    raise ValueError(f"{layout_path}: chiplet {name!r} 的 {label} 不自洽: "
                                     f"{got} != {want_v}")

        chiplets.append({
            "name": name,
            "x-position": round(x_pos, 6),
            "y-position": round(y_pos, 6),
            "width": round(width, 6),
            "height": round(height, 6),
            "rotation": rotation,
            "power": b["power"],
            "hubump": hubump,
        })

    # chiplet 顺序对齐 benchmark (RL 的书写顺序不同, 便于与 AT 的输出/benchmark 直接对比)
    order = {c["name"]: k for k, c in enumerate(bench["chiplets"])}
    if set(order) != {c["name"] for c in chiplets}:
        raise ValueError(f"{layout_path}: chiplet 名字集合与 benchmark 不一致")
    chiplets.sort(key=lambda c: order[c["name"]])

    stem = os.path.splitext(os.path.basename(out_path))[0]
    out = {
        "system_id": stem,                     # 文件名去掉 .json, 与 README 的 "system_id" 同义
        "chiplets": chiplets,
        "connections": [                       # RL 里 connections 为空, 整段取自 benchmark; 只留 3 个字段
            {"node1": e["node1"], "node2": e["node2"], "wireCount": e["wireCount"]}
            for e in bench["connections"]
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok = n_err = 0
    seed_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "shard*", "*_seed*")))
    for seed_dir in seed_dirs:
        if not os.path.isdir(seed_dir):
            continue
        seed_tag = os.path.basename(seed_dir)          # 形如 Case6_seed1
        prefix = seed_tag.rsplit("_seed", 1)[0]
        tag = f"{prefix}/{seed_tag}"

        summary_path = os.path.join(seed_dir, "best_cost_summary.json")
        if not os.path.exists(summary_path):
            print(f"[skip] {tag}: 无 best_cost_summary.json", file=sys.stderr)
            n_err += 1
            continue
        summary = json.load(open(summary_path, encoding="utf-8"))
        if not summary.get("json"):
            print(f"[skip] {tag}: best_cost_summary.json 无 json 字段", file=sys.stderr)
            n_err += 1
            continue

        layout_path, why = pick_layout(seed_dir, summary, cap_area(prefix))
        if not os.path.exists(layout_path):
            print(f"[skip] {tag}: 最优布局不存在 {layout_path}", file=sys.stderr)
            n_err += 1
            continue

        benchmark_path = os.path.join(BENCH_DIR, f"{prefix}.json")
        if not os.path.exists(benchmark_path):
            print(f"[skip] {tag}: 无对应 benchmark {benchmark_path}", file=sys.stderr)
            n_err += 1
            continue

        out_path = os.path.join(OUT_DIR, f"{seed_tag}.json")
        try:
            out = convert_one(layout_path, benchmark_path, out_path)
        except Exception as e:  # noqa: BLE001
            print(f"[err ] {tag}: {type(e).__name__}: {e}", file=sys.stderr)
            n_err += 1
            continue
        print(f"[ok] {tag:<22} -> {os.path.basename(out_path):<22} "
              f"chiplets={len(out['chiplets'])} connections={len(out['connections'])} "
              f"选自 {os.path.basename(layout_path)}  [{why}]")
        n_ok += 1

    print(f"\n完成: 生成 {n_ok} 个文件 -> {OUT_DIR}  (跳过/失败 {n_err})")


if __name__ == "__main__":
    main()
