#!/usr/bin/env python3
"""给 8 个转换后的 case 包裹 hubump 环, 并用 CPLEX 简单测一下线长是否可解。

hubump = f(die): 用 TAP-2.5D routing.py get_input 的整数化 pmax 容量判定
(即 gen_wirelength_dataset.compute_hubump), 保证 bump 环容量 >= 连接数 s, ILP 可解。
footprint = die + 2*hubump (包裹一层后芯片四周预留的 microbump 环)。

测试: 对每个 case, 用网格随便摆个非重叠位置, 跑 solve_cplex_avg, 看是否有可行解
(无 CPLEX Error 1217)。线长数值只取决于位置的相对距离, 与可行性无关。
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_wirelength_dataset as wl  # noqa: E402  (compute_hubump / TapSystem / solve_cplex_avg)

CASES_DIR = Path("/root/placement/ATPlace_pub/examples_8")
OUT_DIR = Path("/root/placement/ATPlace_pub/cases_hubump")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = ["acend910", "cpu-dram", "hp11_m", "multigpu", "syn1", "syn4", "syn6", "xerox8_m"]


def wrap_case(case: str) -> dict:
    data = json.loads((CASES_DIR / f"{case}.json").read_text(encoding="utf-8"))
    chiplets = data["chiplets"]
    connections = data["connections"]

    M = wl._connection_matrix(chiplets, connections)
    n = len(chiplets)

    wrapped = []
    for i, c in enumerate(chiplets):
        w = float(c["width"])
        h = float(c["height"])
        s = sum(M[i][j] + M[j][i] for j in range(n))  # 2×入射 wireCount
        hubump = wl.compute_hubump(w, h, s)
        wrapped.append({
            "name": c["name"],
            "width": round(w, 6), "height": round(h, 6),
            "power": c.get("power", 0),
            "hubump": round(hubump, 6),
            "footprint_w": round(w + 2 * hubump, 6),
            "footprint_h": round(h + 2 * hubump, 6),
            "incident_wirecount": int(round(s / 2)),
        })
    return {"case": case, "chiplets": wrapped, "connections": connections, "wrapped": wrapped}


def grid_record(wrapped: list, connections: list) -> dict:
    """随便摆个网格(用 footprint 尺寸留间距, 非重叠), 仅供可行性测试。"""
    n = len(wrapped)
    cols = math.ceil(math.sqrt(n))
    max_fw = max(c["footprint_w"] for c in wrapped)
    max_fh = max(c["footprint_h"] for c in wrapped)
    gap = 1.0
    chiplets = []
    for i, c in enumerate(wrapped):
        row, col = divmod(i, cols)
        chiplets.append({
            "name": c["name"], "width": c["width"], "height": c["height"],
            "power": c["power"],
            "x-position": round(col * (max_fw + gap), 6),
            "y-position": round(row * (max_fh + gap), 6),
            "hubump": c["hubump"],
        })
    return {"chiplets": chiplets, "connections": connections}


def main() -> int:
    print(f"{'case':<10} {'chiplet: hubump(mm)->footprint(mm)':<70} {'feasible':<8} {'avg_wl(mm)':<12} {'total_wl(mm)'}")
    print("-" * 130)
    ok = 0
    for case in CASES:
        rec = wrap_case(case)
        wrapped = rec["wrapped"]

        # 写出包裹后的 case JSON (body + hubump + footprint)
        out = {
            "case": case,
            "chiplets": [{
                "name": c["name"], "width": c["width"], "height": c["height"],
                "power": c["power"], "hubump": c["hubump"],
                "footprint_w": c["footprint_w"], "footprint_h": c["footprint_h"],
            } for c in wrapped],
            "connections": rec["connections"],
        }
        (OUT_DIR / f"{case}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        # 简单 CPLEX 测试
        system = wl.TapSystem(grid_record(wrapped, rec["connections"]), hubump_mode="stored")
        avg_wl, total_wl, _d, _s = wl.solve_cplex_avg(system)
        feasible = total_wl is not None
        ok += feasible

        desc = " ".join(f"{c['name']}:{c['hubump']:.3f}->{c['footprint_w']:.2f}x{c['footprint_h']:.2f}" for c in wrapped)
        print(f"{case:<10} {desc:<70} {'YES' if feasible else 'NO':<8} "
              f"{avg_wl if feasible else float('nan'):<12.4f} {total_wl if feasible else float('nan'):.4f}")

    print("-" * 130)
    print(f"feasible: {ok}/{len(CASES)}   wrapped JSON -> {OUT_DIR}")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
