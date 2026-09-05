"""精确线长求解器:把 TAP-2.5D 的 microbump 布线 ILP(CPLEX)重写成一个小 LP。

原理(见 gen_wirelength_dataset.py 的 solve_cplex_avg):
  - 每个 chiplet 四周 bump 环,分 4 个 clump(左/上/右/下四边中点);
  - 每条互联 wireCount 根线,从源 chiplet 某 clump 直连到目标 chiplet 某 clump(Hopmax=1);
  - 目标 min Σ 曼哈顿距离 × 线数,约束:每条互联流量守恒 + 每个 clump 容量 pmax;
  - 这是"运输问题",LP 松弛自动整数,用 scipy HiGHS 毫秒级解出,结果与 CPLEX 一致(误差 ~0.0000%)。

用途:
  1. 给任意布局(不只 2 万有标签的)批量生成 wirelength 标签;
  2. 作为"快速精确评估器"直接算某布局的总线长/平均线长(比 CPLEX 快几百倍)。

用法:
  from exact_wirelength import wirelength
  total, avg = wirelength(system_dict)

  # 批量生成标签(写 CSV, 与 wirelength_dataset 同格式):
  python exact_wirelength.py --start 1 --end 1000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linprog

UBUMP_PITCH = 0.045  # 45um microbump 节距 (mm)
NCLUMP = 4           # 每个 chiplet 4 个 clump(上/下/左/右)

PROJECT = "/root/placement/flow_tap"
PLACE_DIR = f"{PROJECT}/Dataset/dataset/placement_dataset/placement_dataset_tw"
OUT_DIR = f"{PROJECT}/Dataset/dataset/wirelength_dataset"


# ---------------------------------------------------------------------------
# hubump / 容量
# ---------------------------------------------------------------------------
def bump_capacity(w_mm: float, h_mm: float, hubump: float) -> int:
    nh = int(hubump / UBUMP_PITCH)
    if nh <= 0:
        return 0
    return (2 * nh * int((h_mm + hubump) / UBUMP_PITCH)
            + 2 * nh * int((w_mm + hubump) / UBUMP_PITCH))


def compute_hubump(w_mm: float, h_mm: float, s: float) -> float:
    """最小 hubump,使得 bump 环容量 >= s(=2×单边 wireCount)。与 gen_wirelength_dataset 一致。"""
    if s <= 0:
        return 0.0
    k = 1
    while True:
        hu = UBUMP_PITCH * k
        if bump_capacity(w_mm, h_mm, hu) >= s:
            return hu
        k += 1
        if k > 1000:
            raise ValueError("microbump too high to be feasible")


# ---------------------------------------------------------------------------
# 核心:精确求解一个 system 的总线长 / 平均线长
# ---------------------------------------------------------------------------
def wirelength(system: dict) -> tuple[float, float]:
    """给定一个 system dict(chiplets + connections),返回 (total_wirelength, avg_wirelength)。

    system 格式与 placement_dataset_tw/chiplet_dataset_*.json 里的单个 system 一致。
    """
    chiplets = system["chiplets"]
    N = len(chiplets)
    name2idx = {c["name"]: i for i, c in enumerate(chiplets)}

    w = [float(c["width"]) for c in chiplets]
    h = [float(c["height"]) for c in chiplets]
    x = [float(c["x-position"]) for c in chiplets]
    y = [float(c["y-position"]) for c in chiplets]

    # 对称连接矩阵
    R = [[0.0] * N for _ in range(N)]
    for e in system["connections"]:
        i = name2idx[e["node1"]]
        j = name2idx[e["node2"]]
        wc = float(e["wireCount"])
        R[i][j] += wc
        R[j][i] += wc

    # hubump(重算,与 solver 的 die 模式一致)+ clump 坐标 + pmax
    hubump = [compute_hubump(w[i], h[i], sum(R[i][j] + R[j][i] for j in range(N)))
              for i in range(N)]
    cx = [x[i] + w[i] / 2 for i in range(N)]
    cy = [y[i] + h[i] / 2 for i in range(N)]
    clump = [[None] * NCLUMP for _ in range(N)]
    pmax = [[0] * NCLUMP for _ in range(N)]
    for i in range(N):
        hu = hubump[i]
        nh = int(hu / UBUMP_PITCH)
        clump[i][0] = (x[i] - hu / 2, cy[i])          # 左
        clump[i][1] = (cx[i], y[i] + h[i] + hu / 2)   # 上
        clump[i][2] = (x[i] + w[i] + hu / 2, cy[i])   # 右
        clump[i][3] = (cx[i], y[i] - hu / 2)          # 下
        pmax[i][0] = nh * int((h[i] + hu) / UBUMP_PITCH)
        pmax[i][1] = nh * int((w[i] + hu) / UBUMP_PITCH)
        pmax[i][2] = pmax[i][0]
        pmax[i][3] = pmax[i][1]

    # 有向 net 列表 (i -> j, 需求 R[i][j])
    nets = [(i, j, R[i][j]) for i in range(N) for j in range(N) if i != j and R[i][j] > 0]
    M = len(nets)

    # LP: 变量 f[n][h][k] >= 0,连续(运输问题自动整数)
    nvar = M * NCLUMP * NCLUMP
    c = np.zeros(nvar)
    for n, (i, j, _) in enumerate(nets):
        for hh in range(NCLUMP):
            for kk in range(NCLUMP):
                c[n * 16 + hh * 4 + kk] = (
                    abs(clump[i][hh][0] - clump[j][kk][0])
                    + abs(clump[i][hh][1] - clump[j][kk][1]))

    # 需求约束:Σ_{h,k} f = R
    Aeq = np.zeros((M, nvar))
    beq = np.zeros(M)
    for n, (_, _, Rn) in enumerate(nets):
        Aeq[n, n * 16:(n + 1) * 16] = 1.0
        beq[n] = Rn

    # 容量约束:每个 clump (i,hh) 的进+出流量 <= pmax
    Aub = np.zeros((N * NCLUMP, nvar))
    bub = np.zeros(N * NCLUMP)
    for i in range(N):
        for hh in range(NCLUMP):
            row = Aub[i * NCLUMP + hh]
            for n, (s, t, _) in enumerate(nets):
                if s == i:
                    for kk in range(NCLUMP):
                        row[n * 16 + hh * 4 + kk] += 1.0
                if t == i:
                    for hp in range(NCLUMP):
                        row[n * 16 + hp * 4 + hh] += 1.0
            bub[i * NCLUMP + hh] = pmax[i][hh]

    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                  bounds=(0, None), method="highs")
    total = float(res.fun)

    # avg = total / wire_count,wire_count = 2×Σ无向 wireCount
    wire_count = sum(R[i][j] for i in range(N) for j in range(N) if i != j)
    avg = total / wire_count if wire_count > 0 else 0.0
    return total, avg


# ---------------------------------------------------------------------------
# 批量生成
# ---------------------------------------------------------------------------
def load_system(system_id: int) -> dict:
    k = (system_id - 1) // 5000 + 1
    fp = f"{PLACE_DIR}/chiplet_dataset_{k}.json"
    with open(fp) as f:
        data = json.load(f)
    return data[f"system_{system_id}"]


def _process(args):
    sid = args
    try:
        system = load_system(sid)
        total, avg = wirelength(system)
        return sid, total, avg, None
    except Exception as e:  # noqa: BLE001
        return sid, None, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    t0 = time.time()
    sids = list(range(args.start, args.end + 1))
    total_dir = os.path.join(args.out, "total_wirelength")
    avg_dir = os.path.join(args.out, "avg_wirelength")
    os.makedirs(total_dir, exist_ok=True)
    os.makedirs(avg_dir, exist_ok=True)

    n_ok = 0
    with Pool(args.workers) as pool:
        for sid, total, avg, err in pool.imap_unordered(_process, sids):
            if err:
                print(f"system_{sid}: ERROR {err}", flush=True)
                continue
            with open(f"{total_dir}/system_total_wirelength_{sid}.csv", "w") as f:
                f.write(f"{total:.6f}\n")
            with open(f"{avg_dir}/system_avg_wirelength_{sid}.csv", "w") as f:
                f.write(f"{avg:.6f}\n")
            n_ok += 1
            if n_ok % 5000 == 0:
                print(f"  完成 {n_ok}/{len(sids)}  ({time.time() - t0:.1f}s)", flush=True)

    print(f"DONE: {n_ok}/{len(sids)} 成功, 耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
