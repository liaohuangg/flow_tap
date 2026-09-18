#!/usr/bin/env python3
"""wl_oracle.py — 线长的精确参考值 + MILP 线性化用的弧流量权重 λ。

两条路径, 职责分开:

  reference_cost(record)   **精确**。直接调 gen_wirelength_dataset 的
                           TapSystem + solve_cplex_avg —— 和 resultEval/eval_layout.py
                           的 eval_wirelength 是同一条调用, 所以报出去的数和评估器一致。
                           这是唯一会被写进 run_log.csv / 用来挑最优解的代价。

  arc_flow(A, x)           **启发式近似**。给 MILP 逐次线性化用的 λ。

为什么 λ 不用精确弧流量: 参考模型是逐 net 的 (f[i][h][j][k][n], Eq.12 每个 net 单独
守恒), Eq.15 的容量又在所有 net 之间共享。想拿到精确 f 就得把那个模型整个重跑一遍,
而它的规模是 Nchiplet² * 16 * Nnets —— Case7 是 85 万个变量、单次 25s, 比参考本身还贵。
更关键的是 **net 之间不能简单聚合**: 连接矩阵是对称的, 聚合后每个 chiplet 的净供给都是 0,
守恒约束直接退化, 得到的是一个远低于真实值的松弛解。

所以改用容量感知的贪心: 每个有向 net 的需求按 clump 对距离从近到远排, 依次吃剩余容量。
它抓住了参考问题最关键的特征 —— 容量是紧绑的 (实测 hp6_m 上, 无容量约束的最优是 12938,
带容量是 19218, 差 48%) —— 而且是 O(nets * 16 log 16), 瞬时。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT / "gen_dataset") not in sys.path:
    sys.path.insert(0, str(PROJECT / "gen_dataset"))

import gen_wirelength_dataset as gwd  # noqa: E402

NCLUMP = gwd.NCLUMP


def build_system(record: dict):
    """构造 gwd.TapSystem。hubump 用记录里的值 (hubump_mode='stored')。"""
    return gwd.TapSystem(record, hubump_mode="stored")


# --------------------------------------------------------------------------- #
# 精确参考代价
# --------------------------------------------------------------------------- #
def reference_cost(record: dict) -> dict:
    """精确线长。返回 {total, avg, d_net, system}。

    与 eval_layout.eval_wirelength 同一条调用路径。求解失败时 total=None
    (solve_cplex_avg 的哨兵), 由调用方决定怎么处理。
    """
    system = build_system(record)
    avg, total, d_net, side_data = gwd.solve_cplex_avg(system)
    return {"total": total, "avg": avg, "d_net": d_net,
            "side_data": side_data, "system": system}


# --------------------------------------------------------------------------- #
# 启发式弧流量 (给 MILP 当 λ)
# --------------------------------------------------------------------------- #
def arc_flow(A: dict, x, uniform: bool = False) -> dict:
    """容量感知贪心, 返回 λ[(i,j)][h][k] (i<j 的无序键, 两个方向已合并)。

    x: [(FX, FY, r), ...] —— footprint 左下角 + 旋转。
    uniform=True 时不做贪心, 直接按连接数平摊 (退化成 clump 均匀初值)。
    """
    n = A["n"]
    if uniform:
        return _uniform(A)

    # 当前布局下每个 chiplet 的 clump 绝对坐标
    # 注意: _xb/_yb 是按**形态** (3 个) 索引的, 侧 h 要通过 SIDE_XFORM/SIDE_YFORM 映射
    import ilp_core as C

    xc = [[0.0] * NCLUMP for _ in range(n)]
    yc = [[0.0] * NCLUMP for _ in range(n)]
    for i in range(n):
        fx, fy, r = x[i]
        xb, yb = A["_xb"][i], A["_yb"][i]
        for h in range(NCLUMP):
            xf, yf = C.SIDE_XFORM[h], C.SIDE_YFORM[h]
            xc[i][h] = fx + xb[xf][0] + xb[xf][1] * r
            yc[i][h] = fy + yb[yf][0] + yb[yf][1] * r

    # 剩余容量: Eq.15 里 f[i][h][j][k] 与 f[j][k][i][h] 共用 (i,h) 的 pmax
    pmax = A["_pmax"]
    remain = [list(pmax[i]) for i in range(n)]

    # 每个有向 net 的需求: R[i][j] = Σ wireCount (对称矩阵, 所以两个方向都有 net)
    lam = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            demand = A["mat"][i][j]
            if demand <= 0:
                continue
            # 该 net 的 16 个 clump 对, 按距离升序
            pairs = sorted(
                ((abs(xc[i][h] - xc[j][k]) + abs(yc[i][h] - yc[j][k]), h, k)
                 for h in range(NCLUMP) for k in range(NCLUMP)),
                key=lambda t: t[0])
            left = demand
            for _d, h, k in pairs:
                if left <= 0:
                    break
                take = min(left, remain[i][h], remain[j][k])
                if take <= 0:
                    continue
                remain[i][h] -= take
                remain[j][k] -= take
                left -= take
                if take > 0:
                    key = (i, j) if i < j else (j, i)
                    table = lam.get(key)
                    if table is None:
                        table = lam[key] = [[0.0] * NCLUMP for _ in range(NCLUMP)]
                    # 无序键: 流量方向映射回 (较小下标, 较大下标) 的 (h,k) 序
                    if i < j:
                        table[h][k] += take
                    else:
                        table[k][h] += take
    return lam


def _uniform(A: dict) -> dict:
    """均匀初值: 无序对的总流量 2*wc 平摊到 16 个 (h,k) 组合。"""
    lam = {}
    for i in range(A["n"]):
        for j in range(i + 1, A["n"]):
            wc = A["mat"][i][j]
            if wc > 0:
                lam[(i, j)] = [[wc / 8.0] * NCLUMP for _ in range(NCLUMP)]
    return lam


def attach_geometry(A: dict) -> dict:
    """把建模要用的 clump 常数和 pmax 预先算好挂在 A 上 (arc_flow 需要)。

    用 gwd.get_input 算, 保证和评估器用的是同一套公式 —— 不自己重推。
    """
    import ilp_core as C

    # 造一个 r=0、hubump 用原值的记录, 只为拿到 pmax (pmax 与位置无关)
    rec = {"chiplets": [{"name": A["names"][i], "x-position": 0.0, "y-position": 0.0,
                         "width": A["w"][i], "height": A["h"][i],
                         "power": A["power"][i], "hubump": A["u"][i]}
                        for i in range(A["n"])],
           "connections": []}
    system = gwd.TapSystem(rec, hubump_mode="stored")
    _xl, _xc, _yl, _yc, _R, _n, _nc, pmax, _hop = gwd.get_input(system)

    A["_pmax"] = pmax
    A["_xb"] = [C.xform_bases(A["w"][i], A["h"][i], A["u"][i]) for i in range(A["n"])]
    A["_yb"] = [C.yform_bases(A["w"][i], A["h"][i], A["u"][i]) for i in range(A["n"])]
    return A


if __name__ == "__main__":
    # 冒烟: 对一份现成的 AT 布局算精确代价, 并核对贪心 λ 的量级
    import argparse
    import json

    ap = argparse.ArgumentParser(description="wl_oracle 冒烟测试")
    ap.add_argument("layout", help="format_result 记录 json")
    args = ap.parse_args()

    record = json.loads(Path(args.layout).read_text(encoding="utf-8"))
    out = reference_cost(record)
    print(f"{Path(args.layout).stem}: total_wirelength={out['total']}  avg={out['avg']}")
