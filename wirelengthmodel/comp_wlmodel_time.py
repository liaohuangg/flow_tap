#!/usr/bin/env python3
"""对比三种线长计算方式的耗时与结果, 对 34 万之后前 1000 个 case
(system_340001..341000) 做基准:

  1. CPLEX ILP 精确求解 (min Σ d·f, 总线长 + 平均线长)
     自包含移植自 gen_dataset/gen_wirelength_dataset.py 的 solve_cplex_avg,
     不 import 外部 routing / gen_dataset 模块。
  2. 曼哈顿线长 (chiplet 中心点, 无拥塞启发式): Σ wireCount × (|Δx|+|Δy|)
  3. wlmodel GNN 预测 (checkpoint/best_wlmodel_total_60k.pt)

每种方式都统计: 总线长 total(mm) / 平均线长 avg(mm) / 单 system 计算耗时(s)。
输出逐 case CSV + 汇总统计(总耗时 / 均值 / min / max, 以及相对 CPLEX 的 MAPE)。

用法:
  python comp_wlmodel_time.py --start 340001 --count 1000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy

import cplex
import torch
from torch.utils.data import DataLoader

# 允许从本目录 import dataloader / wlmodel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader import (  # noqa: E402
    PLACEMENT_DIR,
    Normalizer,
    ChipletWirelengthDataset,
    collate_fn,
    load_labeled_systems,
    parse_system,
    split_records,
)
from wlmodel import WirelengthGNN  # noqa: E402

# --------------------------------------------------------------------------- #
# 常量 / 路径
# --------------------------------------------------------------------------- #
UBUMP_PITCH = 0.045     # 45um microbump 节距, mm
NCLUMP = 4              # 每个 chiplet 4 个 pin clump (上下左右)
TIMELIMIT_AVG = 500.0   # CPLEX 时间上限 (s), 与 routing.py 一致
CHUNK = 5000            # 每个 chiplet_dataset_{k}.json 含 5000 systems

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "checkpoint", "best_wlmodel_total_60k.pt")
NORM_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "checkpoint", "normalizer_total_60k.pt")
LABEL_DIR = "/root/placement/flow_tap/Dataset/dataset/wirelength_dataset/total_wirelength"


# --------------------------------------------------------------------------- #
# (1) CPLEX 精确求解 —— 自包含移植自 gen_dataset/gen_wirelength_dataset.py
# --------------------------------------------------------------------------- #
def _bump_capacity(w_mm: float, h_mm: float, hubump: float) -> int:
    nh = int(hubump / UBUMP_PITCH)
    if nh <= 0:
        return 0
    return 2 * nh * int((h_mm + hubump) / UBUMP_PITCH) + 2 * nh * int((w_mm + hubump) / UBUMP_PITCH)


def compute_hubump(w_mm: float, h_mm: float, s: float) -> float:
    """按连接数 s 计算芯片四周 bump 环宽度 (mm), 保证 pmax 容量 >= s (否则 ILP 无解)。"""
    if s <= 0:
        return 0.0
    h = 1
    w_stretch = UBUMP_PITCH * h
    while True:
        if _bump_capacity(w_mm, h_mm, w_stretch) >= s:
            return w_stretch
        h += 1
        w_stretch = UBUMP_PITCH * h
        if h > 1000:
            raise ValueError("microbump is too high to be a feasible case")


def _connection_matrix(chiplets: list[dict], connections: list[dict]) -> list[list[float]]:
    """对称连接矩阵 M[i][j] = chiplet i 与 j 之间的 wireCount 之和。"""
    names = [str(c.get("name", f"C{i}")) for i, c in enumerate(chiplets)]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    n = len(chiplets)
    M = [[0.0] * n for _ in range(n)]
    for conn in connections:
        n1 = str(conn.get("node1", ""))
        n2 = str(conn.get("node2", ""))
        if n1 not in name_to_idx or n2 not in name_to_idx:
            continue
        i = name_to_idx[n1]
        j = name_to_idx[n2]
        wc = float(conn.get("wireCount", 0.0))
        M[i][j] += wc
        M[j][i] += wc
    return M


class TapSystem:
    def __init__(self, record: dict):
        chiplets = record["chiplets"]
        connections = record.get("connections", [])
        self.chiplet_count = len(chiplets)

        self.width = [float(c["width"]) for c in chiplets]
        self.height = [float(c["height"]) for c in chiplets]
        # 中心 = body 左下角 + body/2
        self.x = [float(c["x-position"]) + self.width[i] / 2.0 for i, c in enumerate(chiplets)]
        self.y = [float(c["y-position"]) + self.height[i] / 2.0 for i, c in enumerate(chiplets)]

        M = _connection_matrix(chiplets, connections)
        self.connection_matrix = M
        self.hubump = [
            compute_hubump(self.width[i], self.height[i],
                           sum(M[i][j] + M[j][i] for j in range(self.chiplet_count)))
            for i in range(self.chiplet_count)
        ]

        self.intp_type = "passive"
        self.link_type = "nppl"


def get_input(system: TapSystem):
    Nchiplet = system.chiplet_count
    Hopmax = 1
    if system.intp_type == "passive":
        if system.link_type == "ppl":
            Hopmax = 2
    xl, yl = [None] * Nchiplet, [None] * Nchiplet
    xc = [[None for _ in range(NCLUMP)] for _ in range(Nchiplet)]
    yc = [[None for _ in range(NCLUMP)] for _ in range(Nchiplet)]
    pmax = [[None for _ in range(NCLUMP)] for _ in range(Nchiplet)]
    for i in range(Nchiplet):
        xl[i] = system.x[i] - system.width[i] / 2 - system.hubump[i]
        yl[i] = system.y[i] - system.height[i] / 2 - system.hubump[i]
        xc[i][0], yc[i][0], pmax[i][0] = (system.hubump[i] / 2, system.height[i] / 2 + system.hubump[i],
                                          int(system.hubump[i] / UBUMP_PITCH) * int((system.height[i] + system.hubump[i]) / UBUMP_PITCH))
        xc[i][1], yc[i][1], pmax[i][1] = (system.width[i] / 2 + system.hubump[i], system.hubump[i] * 1.5 + system.height[i],
                                          int(system.hubump[i] / UBUMP_PITCH) * int((system.width[i] + system.hubump[i]) / UBUMP_PITCH))
        xc[i][2], yc[i][2], pmax[i][2] = system.width[i] + system.hubump[i] * 1.5, yc[i][0], pmax[i][0]
        xc[i][3], yc[i][3], pmax[i][3] = xc[i][1], system.hubump[i] / 2, pmax[i][1]
    R = deepcopy(system.connection_matrix)
    return xl, xc, yl, yc, R, Nchiplet, NCLUMP, pmax, Hopmax


def get_index(i, h, j, k, n, Nchiplet, Nclump, Nmax):
    return (i * Nclump * Nchiplet * Nclump * Nmax + h * Nchiplet * Nclump * Nmax
            + j * Nclump * Nmax + k * Nmax + n)


def solve_cplex_avg(system: TapSystem):
    """求解平均线长 (routing.py 版): 最小化 Σ d·f (总线长)。

    返回 (avg_wirelength, total_wirelength, d_net, side_data)。
    """
    xl, xc, yl, yc, R, Nchiplet, Nclump, pmax, Hopmax = get_input(system)

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.parameters.threads.set(1)
    problem.parameters.timelimit.set(TIMELIMIT_AVG)
    problem.set_log_stream(None)
    problem.set_results_stream(None)

    d = [[[[0 for _ in range(Nclump)] for _ in range(Nchiplet)] for _ in range(Nclump)] for _ in range(Nchiplet)]
    for i in range(Nchiplet):
        for h in range(Nclump):
            for j in range(Nchiplet):
                for k in range(Nclump):
                    d[i][h][j][k] = (abs(xl[i] + xc[i][h] - xl[j] - xc[j][k])
                                     + abs(yl[i] + yc[i][h] - yl[j] - yc[j][k]))

    s, t = [], []
    n = 0
    wire_count = 0
    for i in range(Nchiplet):
        for j in range(Nchiplet):
            if (i != j) and (R[i][j] > 0):
                s.append(i)
                t.append(j)
                n += 1
                wire_count += R[i][j]
    Nmax = n

    # Eq.11: 决策变量 f[i][h][j][k][n]
    for i in range(Nchiplet):
        for h in range(Nclump):
            for j in range(Nchiplet):
                for k in range(Nclump):
                    for _n in range(Nmax):
                        if (i == j) and (h == k):
                            problem.variables.add(lb=[0.0], ub=[0.0], types=[problem.variables.type.integer])
                        else:
                            problem.variables.add(lb=[0.0], ub=[pmax[i][h]], types=[problem.variables.type.integer])

    # Eq.12
    for _n in range(Nmax):
        row_index, row_coeff = [], []
        for h in range(Nclump):
            for j in range(Nchiplet):
                if j != s[_n]:
                    for k in range(Nclump):
                        fij_index = get_index(s[_n], h, j, k, _n, Nchiplet, Nclump, Nmax)
                        row_index.append(fij_index)
                        row_coeff.append(1)
                        fji_index = get_index(j, k, s[_n], h, _n, Nchiplet, Nclump, Nmax)
                        row_index.append(fji_index)
                        row_coeff.append(-1)
        problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["E"], rhs=[R[s[_n]][t[_n]]])

        row_index, row_coeff = [], []
        for h in range(Nclump):
            for j in range(Nchiplet):
                if j != t[_n]:
                    for k in range(Nclump):
                        fij_index = get_index(t[_n], h, j, k, _n, Nchiplet, Nclump, Nmax)
                        row_index.append(fij_index)
                        row_coeff.append(1)
                        fji_index = get_index(j, k, t[_n], h, _n, Nchiplet, Nclump, Nmax)
                        row_index.append(fji_index)
                        row_coeff.append(-1)
        problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["E"], rhs=[-R[s[_n]][t[_n]]])

        for i in range(Nchiplet):
            if (i != s[_n]) and (i != t[_n]):
                row_index, row_coeff = [], []
                for h in range(Nclump):
                    for j in range(Nchiplet):
                        if j != i:
                            for k in range(Nclump):
                                fij_index = get_index(i, h, j, k, _n, Nchiplet, Nclump, Nmax)
                                row_index.append(fij_index)
                                row_coeff.append(1)
                                fji_index = get_index(j, k, i, h, _n, Nchiplet, Nclump, Nmax)
                                row_index.append(fji_index)
                                row_coeff.append(-1)
                problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["E"], rhs=[0])

    # Eq.13 & 14
    for _n in range(Nmax):
        srow_index, srow_coeff = [], []
        trow_index, trow_coeff = [], []
        for h in range(Nclump):
            for j in range(Nchiplet):
                for k in range(Nclump):
                    fs_index = get_index(j, k, s[_n], h, _n, Nchiplet, Nclump, Nmax)
                    srow_index.append(fs_index)
                    srow_coeff.append(1)
                    ft_index = get_index(t[_n], h, j, k, _n, Nchiplet, Nclump, Nmax)
                    trow_index.append(ft_index)
                    trow_coeff.append(1)
        problem.linear_constraints.add(lin_expr=[[srow_index, srow_coeff]], senses=["E"], rhs=[0])
        problem.linear_constraints.add(lin_expr=[[trow_index, trow_coeff]], senses=["E"], rhs=[0])

    # Eq.15
    for i in range(Nchiplet):
        for h in range(Nclump):
            row_index, row_coeff = [], []
            for j in range(Nchiplet):
                if i != j:
                    for k in range(Nclump):
                        for _n in range(Nmax):
                            fij_index = get_index(i, h, j, k, _n, Nchiplet, Nclump, Nmax)
                            row_index.append(fij_index)
                            row_coeff.append(1)
                            fji_index = get_index(j, k, i, h, _n, Nchiplet, Nclump, Nmax)
                            row_index.append(fji_index)
                            row_coeff.append(1)
            problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["L"], rhs=[pmax[i][h]])

    # Eq.18
    for _n in range(Nmax):
        row_index, row_coeff = [], []
        if Hopmax == 1:
            for i in range(Nchiplet):
                for h in range(Nclump):
                    for j in range(Nchiplet):
                        for k in range(Nclump):
                            f_index = get_index(i, h, j, k, _n, Nchiplet, Nclump, Nmax)
                            row_index.append(f_index)
                            row_coeff.append(1)
            problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["L"], rhs=[R[s[_n]][t[_n]]])
        elif Hopmax == 2:
            for h in range(Nclump):
                for k in range(Nclump):
                    f_index = get_index(s[_n], h, t[_n], k, _n, Nchiplet, Nclump, Nmax)
                    row_index.append(f_index)
                    row_coeff.append(2)
                    for i in range(Nchiplet):
                        for j in range(Nchiplet):
                            if i != s[_n] or j != t[_n]:
                                f_index = get_index(i, h, j, k, _n, Nchiplet, Nclump, Nmax)
                                row_index.append(f_index)
                                row_coeff.append(1)
            problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["L"], rhs=[2 * R[s[_n]][t[_n]]])

    # 目标: min Σ d·f
    for i in range(Nchiplet):
        for h in range(Nclump):
            for j in range(Nchiplet):
                for k in range(Nclump):
                    for _n in range(Nmax):
                        fij_index = get_index(i, h, j, k, _n, Nchiplet, Nclump, Nmax)
                        problem.objective.set_linear(fij_index, d[i][h][j][k])

    problem.solve()

    try:
        total_wirelength = problem.solution.get_objective_value()
        avg_wirelength = total_wirelength / wire_count
    except Exception:
        return 100.0, None, {}
    return avg_wirelength, total_wirelength, {}, {}


# --------------------------------------------------------------------------- #
# (2) 曼哈顿线长 (chiplet 中心点)
# --------------------------------------------------------------------------- #
def manhattan_center(record: dict):
    """Σ wireCount × 中心曼哈顿距离, 与 CPLEX total 同口径 (有向网求和 → ×2)。

    返回 (total_wirelength, avg_wirelength)。
    """
    chiplets = record["chiplets"]
    name2idx = {c["name"]: i for i, c in enumerate(chiplets)}
    cx = [float(c["x-position"]) + float(c["width"]) / 2.0 for c in chiplets]
    cy = [float(c["y-position"]) + float(c["height"]) / 2.0 for c in chiplets]

    undirected = 0.0
    wcount = 0.0
    for e in record.get("connections", []):
        i = name2idx[e["node1"]]
        j = name2idx[e["node2"]]
        w = float(e["wireCount"])
        d = abs(cx[i] - cx[j]) + abs(cy[i] - cy[j])
        undirected += w * d
        wcount += w
    total = 2.0 * undirected
    avg = undirected / wcount if wcount > 0 else 0.0
    return total, avg


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def load_systems(start: int, count: int) -> dict[int, dict]:
    end = start + count - 1
    k0 = (start - 1) // CHUNK + 1
    k1 = (end - 1) // CHUNK + 1
    systems: dict[int, dict] = {}
    for k in range(k0, k1 + 1):
        fp = os.path.join(PLACEMENT_DIR, f"chiplet_dataset_{k}.json")
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for sid, rec in data.items():
            i = int(sid.split("_")[1])
            if start <= i <= end:
                systems[i] = rec
    return systems


def _read_label(sid: int):
    fp = os.path.join(LABEL_DIR, f"system_total_wirelength_{sid}.csv")
    if not os.path.exists(fp):
        return None
    with open(fp, encoding="utf-8") as f:
        return float(f.read().strip())


# --------------------------------------------------------------------------- #
# (3) wlmodel: 归一化器重建 + 模型加载 + 推理
# --------------------------------------------------------------------------- #
def build_normalizer() -> Normalizer:
    """重建训练用的 Normalizer (fit 在 seed=42 的 train 切分上), 结果缓存。

    注意: 训练只在 weight 里存了 state_dict, 没存归一化统计; 这里用与 wlmodel.train()
    完全一致的顺序 (LABELED_FILES 69..80, dict 顺序) + split_records(seed=42) 复现。
    """
    if os.path.exists(NORM_CACHE):
        d = torch.load(NORM_CACHE, map_location="cpu", weights_only=False)
        return Normalizer(d["node_mean"], d["node_std"], d["edge_mean"], d["edge_std"])

    print("[wlmodel] 未找到归一化缓存, 加载 60000 个有标签 system 重建 Normalizer ...", flush=True)
    t0 = time.time()
    records = load_labeled_systems(use_congestion=True)
    train_recs, _, _ = split_records(records, seed=42)
    norm = Normalizer.fit([r[2] for r in train_recs])
    print(f"[wlmodel] Normalizer 重建完成 ({time.time() - t0:.1f}s), 缓存到 {NORM_CACHE}", flush=True)
    torch.save({"node_mean": norm.node_mean, "node_std": norm.node_std,
                "edge_mean": norm.edge_mean, "edge_std": norm.edge_std}, NORM_CACHE)
    return norm


def build_model(device: torch.device) -> WirelengthGNN:
    model = WirelengthGNN(node_dim=18, hidden=256, num_layers=6, heads=4, dropout=0.0,
                          use_residual=True, use_global=True).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _forward_batch(model, item, device):
    x = item["x"].to(device)
    ei = item["edge_index"].to(device)
    ea = item["edge_attr"].to(device)
    ew = item["edge_weight"].to(device)
    ng = item["node_geom"].to(device)
    ga = item["global_attr"].to(device)
    cong = item["cong"].to(device)
    batch = item["batch"].to(device)
    with torch.no_grad():
        total_pred, _, _ = model(x, ei, ea, ew, ng, batch, ga, cong)
    return total_pred


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=340001)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                            "comp_wlmodel_time_results.csv"))
    args = ap.parse_args()

    end = args.start + args.count - 1
    systems = load_systems(args.start, args.count)
    sids = sorted(systems)
    print(f"[load] 读取 system_{args.start}..{end}: {len(sids)} 个 case", flush=True)
    if not sids:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    # --- 预先加载 wlmodel 归一化器 + 模型 (与 CPLEX 循环无关) ---
    norm = build_normalizer()
    model = build_model(device)

    # 构造我的 1000 个 system 的图, 复用 dataloader 的归一化 / collate
    t0 = time.time()
    my_records = []
    for sid in sids:
        graph = parse_system(systems[sid], use_congestion=True)
        # y (log total) 仅用于训练损失, 推理不参与; 这里填 >0 即可, 优先用已存标签
        total_ref = _read_label(sid) or 1.0
        my_records.append((sid, systems[sid], graph, total_ref, None))
    dataset = ChipletWirelengthDataset(my_records, norm)
    print(f"[wlmodel] 构造 1000 个图 {time.time() - t0:.2f}s", flush=True)

    # --- wlmodel 批量推理 (吞吐) ---
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    # warmup
    for item in loader:
        _forward_batch(model, item, device)
        break
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_batch0 = time.time()
    pred_total = []
    with torch.no_grad():
        for item in loader:
            pred_total.append(_forward_batch(model, item, device).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_batch = time.time() - t_batch0
    pred_total = torch.cat(pred_total)
    wl_total = {sid: float(p) for sid, p in zip(sids, pred_total)}

    # --- wlmodel 单 system 延迟 (batch=1, 逐 system 计时, 含 sync) ---
    loader1 = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    single_times = []
    for idx, item in enumerate(loader1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = time.perf_counter()
        _forward_batch(model, item, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        single_times.append(time.perf_counter() - ts)
        if idx >= 20 and idx == 20:  # 前 21 个做 warmup 之外, 继续全部计时
            pass

    # ------------------------------------------------------------------ #
    # 逐 system: CPLEX / 曼哈顿 / wlmodel, 单线程串行
    # ------------------------------------------------------------------ #
    rows = []
    cplex_times = []
    manhattan_times = []
    for idx, sid in enumerate(sids):
        rec = systems[sid]

        # CPLEX
        t_cplex0 = time.perf_counter()
        try:
            system = TapSystem(rec)
            avg, total, _, _ = solve_cplex_avg(system)
            cplex_total = None if total is None else float(total)
            cplex_err = None
        except Exception as e:  # noqa: BLE001
            cplex_total = None
            cplex_err = f"{type(e).__name__}: {e}"
        t_cplex = time.perf_counter() - t_cplex0
        cplex_times.append(t_cplex)

        # 曼哈顿
        t_m0 = time.perf_counter()
        man_total, man_avg = manhattan_center(rec)
        t_manhattan = time.perf_counter() - t_m0
        manhattan_times.append(t_manhattan)

        label = _read_label(sid)

        rows.append({
            "system_id": f"system_{sid}",
            "cplex_total": cplex_total,
            "cplex_time": t_cplex,
            "cplex_error": cplex_err,
            "manhattan_total": man_total,
            "manhattan_time": t_manhattan,
            "wlmodel_total": wl_total[sid],
            "label_total": label,
        })

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"[progress] {idx + 1}/{len(sids)} system_{sid}: "
                  f"cplex={cplex_total if cplex_total is not None else 'ERR'}mm "
                  f"({t_cplex:.2f}s) man={man_total:.1f}mm "
                  f"wl={wl_total[sid]:.1f}mm label={label}", flush=True)

    # ------------------------------------------------------------------ #
    # 汇总统计
    # ------------------------------------------------------------------ #
    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"n": 0, "total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {"n": len(vals), "total": sum(vals), "mean": sum(vals) / len(vals),
                "min": min(vals), "max": max(vals)}

    cplex_ok = [r["cplex_total"] for r in rows if r["cplex_total"] is not None]
    label_ok = [r["label_total"] for r in rows if r["label_total"] is not None]
    # CPLEX vs 已存标签 校验 (确认移植正确)
    cplex_vs_label = []
    for r in rows:
        if r["cplex_total"] is not None and r["label_total"] is not None:
            cplex_vs_label.append(abs(r["cplex_total"] - r["label_total"]) / r["label_total"])

    def mape(preds, trues):
        n = 0
        s = 0.0
        for p, t in zip(preds, trues):
            if p is not None and t is not None and t > 0:
                s += abs(p - t) / t
                n += 1
        return s / n if n else float("nan"), n

    man_mape, _ = mape([r["manhattan_total"] for r in rows], [r["cplex_total"] for r in rows])
    wl_mape, _ = mape([r["wlmodel_total"] for r in rows], [r["cplex_total"] for r in rows])
    wl_mape_label, _ = mape([r["wlmodel_total"] for r in rows], [r["label_total"] for r in rows])

    sc, sm, sw = stats(cplex_times), stats(manhattan_times), stats(single_times)

    print("\n================ 汇总 ================")
    print(f"case 数: {len(rows)} (system_{args.start}..{end})")
    print(f"\n[CPLEX]  成功 {len(cplex_ok)}/{len(rows)}")
    print(f"  总耗时 {sc['total']:.1f}s  单case均值 {sc['mean']:.3f}s  "
          f"min {sc['min']:.3f}s  max {sc['max']:.3f}s")
    print(f"[曼哈顿] 总耗时 {sm['total']:.4f}s  单case均值 {sm['mean']*1e6:.1f}us")
    print(f"[wlmodel] 批量(128)总耗时 {t_batch:.3f}s  摊薄 {t_batch/len(sids)*1e3:.2f}ms/case")
    print(f"[wlmodel] 单system(batch=1, 含sync) 均值 {sw['mean']*1e3:.2f}ms  "
          f"min {sw['min']*1e3:.2f}ms  max {sw['max']*1e3:.2f}ms")

    print(f"\n[精度] 相对 CPLEX: 曼哈顿 MAPE {man_mape:.2%},  wlmodel MAPE {wl_mape:.2%}")
    if cplex_vs_label:
        print(f"[校验] CPLEX 与已存标签 max rel err {max(cplex_vs_label):.4%} "
              f"(mean {sum(cplex_vs_label)/len(cplex_vs_label):.4%})")
    if wl_mape_label == wl_mape_label:
        print(f"[精度] wlmodel 相对已存标签 MAPE {wl_mape_label:.2%}")

    # 写 CSV
    header = ["system_id", "cplex_total_mm", "cplex_time_s", "cplex_error",
              "manhattan_total_mm", "manhattan_time_s",
              "wlmodel_total_mm", "label_total_mm"]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join([
                r["system_id"],
                "" if r["cplex_total"] is None else f"{r['cplex_total']:.6f}",
                f"{r['cplex_time']:.6f}",
                r["cplex_error"] or "",
                f"{r['manhattan_total']:.6f}",
                f"{r['manhattan_time']:.6e}",
                f"{r['wlmodel_total']:.6f}",
                "" if r["label_total"] is None else f"{r['label_total']:.6f}",
            ]) + "\n")
    print(f"\n结果已写: {args.out}")


if __name__ == "__main__":
    main()
