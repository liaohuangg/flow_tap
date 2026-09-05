#!/usr/bin/env python3
"""
为 placement_dataset/placement_dataset_tw (body 坐标 + 每个 chiplet 的 hubump 字段) 生成线长数据集,
使用 CPLEX (IBM ILOG CPLEX 22.1.0) 求解 TAP-2.5D 的 microbump 布线整数规划 (ILP)。

自包含实现: 把 TAP-2.5D 的 routing.py (平均线长 avg wirelength) 与 routing_maxL.py
(最大线长 max wirelength) 两个 solve_Cplex 及其全部依赖函数 (get_input / get_index /
translate_index / read_input) 完整移植到本文件, 不 import 外部 routing 模块。

hubump 说明 (与 TAP-2.5D routing.py get_input 完全一致):
  - 布线模型拿 die(body) 尺寸, 向外计算 bump region: hubump = f(die), 即满足环形
    microbump 容量 >= 连接数 s 的最小环宽 (compute_hubump, 等价于 compute_ubump_overhead 口径)。
  - xl/yl = die 中心 - die/2 - hubump (向外扩), clump 位置与 pmax 按 routing.py get_input 公式计算。
  - preprocess_bump_region.py 存的 hubump 已是 f(die) 固定点, 与这里按 die 重算结果一致。

输出目录 (Dataset/dataset/wirelength_dataset/):
  total_wirelength/ system_total_wirelength_{i}.csv  标量(总线长, mm = min Σ d·f)
  avg_wirelength/   system_avg_wirelength_{i}.csv    标量(平均线长, mm = 总线长/wire_count)
  max_wirelength/   system_max_wirelength_{i}.csv    标量(最大线长, mm, 仅 --variant maxL/both)
  solve_time/       system_wirelength_time_{i}.csv   标量(求解耗时, s)
  summary.csv       汇总(每布局一行, 含 chiplet 数 / net 数 / 线长 / 耗时)

用法:
  python gen_wirelength_dataset.py --start 340001 --end 340010 --workers 4
  python gen_wirelength_dataset.py --start 340001 --end 340010 --variant avg
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import traceback
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

import cplex

PROJECT = Path("/root/placement/flow_tap")
DATASET = PROJECT / "Dataset" / "dataset"
PLACE_DATASET = DATASET / "placement_dataset" / "placement_dataset_tw"
OUT_DIR = DATASET / "wirelength_dataset"

AVG_DIR = OUT_DIR / "avg_wirelength"
TOTAL_DIR = OUT_DIR / "total_wirelength"
MAX_DIR = OUT_DIR / "max_wirelength"
TIME_DIR = OUT_DIR / "solve_time"

CHUNK = 5000            # 每个 chiplet_dataset_{k}.json 含 5000 systems
UBUMP_PITCH = 0.045     # 45um microbump 节距, mm
NCLUMP = 4              # 每个 chiplet 4 个 pin clump (上下左右)
TIMELIMIT_AVG = 500.0   # avg 变体的 CPLEX 时间上限 (s), 与 routing.py 一致
TIMELIMIT_MAX = 300.0   # maxL 变体的 CPLEX 时间上限 (s), 与 routing_maxL.py 一致

for _d in (AVG_DIR, TOTAL_DIR, MAX_DIR, TIME_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# hubump / 连接矩阵
# --------------------------------------------------------------------------- #
def _bump_capacity(w_mm: float, h_mm: float, hubump: float) -> int:
    """routing.get_input 里 pmax 的整数化 bump 容量 (上/下/左/右 4 个 clump 之和)。

    上/下 clump 用 height, 左/右 clump 用 width, 每 clump =
    int(hubump/0.045) * int((edge+hubump)/0.045)。
    """
    nh = int(hubump / UBUMP_PITCH)
    if nh <= 0:
        return 0
    return 2 * nh * int((h_mm + hubump) / UBUMP_PITCH) + 2 * nh * int((w_mm + hubump) / UBUMP_PITCH)


def compute_hubump(w_mm: float, h_mm: float, s: float) -> float:
    """按连接数 s 计算芯片四周 bump 环宽度 (mm)。

    与 TAP-2.5D compute_ubump_overhead 同一思路, 但直接用 routing.get_input 的整数化
    pmax 容量判定, 保证布线 ILP 的 bump 容量 >= s (可解)。若用连续公式 + int 截断,
    会出现 pmax 比 s 少 2~几十个 microbump 的缺口, 使 ILP 无解 (CPLEX Error 1217)。

    w_mm/h_mm 为芯片本体(body/die)尺寸。s = Σ(M[i][j] + M[j][i]) = 2×单边 wireCount。
    """
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


# --------------------------------------------------------------------------- #
# TapSystem: 从 body 记录构造求解所需的 system 对象 (与参考接口一致)
# --------------------------------------------------------------------------- #
class TapSystem:
    def __init__(self, record: dict, hubump_mode: str = "die"):
        chiplets = record["chiplets"]
        connections = record.get("connections", [])
        self.chiplet_count = len(chiplets)

        self.width = [float(c["width"]) for c in chiplets]    # body(die) 宽
        self.height = [float(c["height"]) for c in chiplets]  # body(die) 高
        # 中心 = body 左下角 + body/2 (die 中心 == footprint 中心)
        self.x = [float(c["x-position"]) + self.width[i] / 2.0 for i, c in enumerate(chiplets)]
        self.y = [float(c["y-position"]) + self.height[i] / 2.0 for i, c in enumerate(chiplets)]

        M = _connection_matrix(chiplets, connections)
        self.connection_matrix = M
        if hubump_mode == "stored":
            # 读 preprocess 存好的 hubump (固定点已保证 = f(die), 与下面重算一致)
            self.hubump = [float(c.get("hubump", 0.0)) for c in chiplets]
        else:
            # 用 body(die) 尺寸向外算 hubump (= TAP-2.5D compute_ubump_overhead 口径)
            self.hubump = [
                compute_hubump(self.width[i], self.height[i],
                               sum(M[i][j] + M[j][i] for j in range(self.chiplet_count)))
                for i in range(self.chiplet_count)
            ]

        self.intp_type = "passive"
        self.link_type = "nppl"


# --------------------------------------------------------------------------- #
# 以下为 TAP-2.5D routing.py 的完整移植 (平均线长 avg wirelength)
# --------------------------------------------------------------------------- #
def read_input():
    """仅用于复刻 Vaishnav 版本的 in/out (文件输入)。测试用途, 本脚本不调用。"""
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = ""

    with open(path + "OptPlaceRoute.cfg", "r") as Conf:
        Conf.readline()
        Conf.readline()
        Conf.readline()
        Nclump = int(Conf.readline().split()[1])
        Nchiplet = int(Conf.readline().split()[1])
        p = int(Conf.readline().split()[1])
        Hopmax = int(Conf.readline().split()[1])
    pmax = [[p for _ in range(Nclump)] for _ in range(Nchiplet)]

    xl, yl = [0] * Nchiplet, [0] * Nchiplet
    xc = [[0 for _ in range(Nclump)] for _ in range(Nchiplet)]
    yc = [[0 for _ in range(Nclump)] for _ in range(Nchiplet)]
    R = [[0 for _ in range(Nchiplet)] for _ in range(Nchiplet)]
    with open(path + "Xl.txt", "r") as Xchiplet:
        for i in range(Nchiplet):
            xl[i] = float(Xchiplet.readline())
    with open(path + "Xc.txt", "r") as Xclump:
        for h in range(Nclump):
            p = float(Xclump.readline())
            for i in range(Nchiplet):
                xc[i][h] = p
    with open(path + "Yl.txt", "r") as Ychiplet:
        for i in range(Nchiplet):
            yl[i] = float(Ychiplet.readline())
    with open(path + "Yc.txt", "r") as Yclump:
        for h in range(Nclump):
            p = float(Yclump.readline())
            for i in range(Nchiplet):
                yc[i][h] = p
    with open(path + "R.txt", "r") as Connection:
        for i in range(Nchiplet):
            R[i] = list(map(int, Connection.readline().split()))

    print(R)
    return xl, xc, yl, yc, R, Nchiplet, Nclump, pmax, Hopmax


def get_input(system):
    """把 system 对象转成求解所需的 xl/yl/xc/yc/pmax/R/Hopmax (与 routing.py 一致)。"""
    Nchiplet = system.chiplet_count
    Hopmax = 1
    if system.intp_type == "passive":
        if system.link_type == "ppl":
            Hopmax = 2
    # xl/yl 是 chiplet footprint 左下角 (body 左下角再向外扩 hubump)
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


def translate_index(f_index, Nchiplet, Nclump, Nmax):
    index = int(f_index)
    n = index % Nmax
    index = int((index - n) / Nmax)
    k = index % Nclump
    index = int((index - k) / Nclump)
    j = index % Nchiplet
    index = int((index - j) / Nchiplet)
    h = index % Nclump
    i = int((index - h) / Nclump)
    return i, h, j, k, n


def solve_cplex_avg(system) -> tuple[float, float | None]:
    """求解平均线长 (routing.py 版): 最小化 Σ d·f(总线长), 返回 (avg_wirelength, total_wirelength)。"""
    xl, xc, yl, yc, R, Nchiplet, Nclump, pmax, Hopmax = get_input(system)

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.parameters.threads.set(1)
    problem.parameters.timelimit.set(TIMELIMIT_AVG)
    problem.set_log_stream(None)
    problem.set_results_stream(None)

    # 距离矩阵 d[i][h][j][k]
    d = [[[[0 for _ in range(Nclump)] for _ in range(Nchiplet)] for _ in range(Nclump)] for _ in range(Nchiplet)]
    for i in range(Nchiplet):
        for h in range(Nclump):
            for j in range(Nchiplet):
                for k in range(Nclump):
                    d[i][h][j][k] = (abs(xl[i] + xc[i][h] - xl[j] - xc[j][k])
                                     + abs(yl[i] + yc[i][h] - yl[j] - yc[j][k]))

    # net (s,t) 与总 wire 数
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
        return avg_wirelength, total_wirelength
    except Exception:
        return 100.0, None


# --------------------------------------------------------------------------- #
# 以下为 TAP-2.5D routing_maxL.py 的完整移植 (最大线长 max wirelength)
# --------------------------------------------------------------------------- #
def get_index_maxL(i, h, j, k, n, Nchiplet, Nclump, Nmax):
    return (i * Nclump * Nchiplet * Nclump * Nmax + h * Nchiplet * Nclump * Nmax
            + j * Nclump * Nmax + k * Nmax + n) * 2


def translate_index_maxL(f_index, Nchiplet, Nclump, Nmax):
    index = int(f_index / 2)
    n = index % Nmax
    index = int((index - n) / Nmax)
    k = index % Nclump
    index = int((index - k) / Nclump)
    j = index % Nchiplet
    index = int((index - j) / Nchiplet)
    h = index % Nclump
    i = int((index - h) / Nclump)
    return i, h, j, k, n


def solve_cplex_maxL(system) -> float:
    """求解最大线长 (routing_maxL.py 版): 最小化 max 线长 λ, 返回 λ。"""
    xl, xc, yl, yc, R, Nchiplet, Nclump, pmax, Hopmax = get_input(system)

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.parameters.threads.set(1)
    problem.parameters.timelimit.set(TIMELIMIT_MAX)
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
    for i in range(Nchiplet):
        for j in range(Nchiplet):
            if (i != j) and (R[i][j] > 0):
                s.append(i)
                t.append(j)
                n += 1
    Nmax = n

    # Eq.11: 每个 (i,h,j,k,n) 两个变量 f(偶) 与 λ(奇)
    for i in range(Nchiplet):
        for h in range(Nclump):
            for j in range(Nchiplet):
                for k in range(Nclump):
                    for _n in range(Nmax):
                        if (i == j) and (h == k):
                            problem.variables.add(lb=[0.0, 0.0], ub=[0.0, 0.0],
                                                  types=[problem.variables.type.integer] * 2)
                        else:
                            problem.variables.add(lb=[0.0, 0.0], ub=[pmax[i][h], 1.0],
                                                  types=[problem.variables.type.integer] * 2)

    num_val = problem.variables.get_num()

    # Eq.12
    for _n in range(Nmax):
        row_index, row_coeff = [], []
        for h in range(Nclump):
            for j in range(Nchiplet):
                if j != s[_n]:
                    for k in range(Nclump):
                        row_index.append(get_index_maxL(s[_n], h, j, k, _n, Nchiplet, Nclump, Nmax))
                        row_coeff.append(1)
                        row_index.append(get_index_maxL(j, k, s[_n], h, _n, Nchiplet, Nclump, Nmax))
                        row_coeff.append(-1)
        problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["E"], rhs=[R[s[_n]][t[_n]]])

        row_index, row_coeff = [], []
        for h in range(Nclump):
            for j in range(Nchiplet):
                if j != t[_n]:
                    for k in range(Nclump):
                        row_index.append(get_index_maxL(t[_n], h, j, k, _n, Nchiplet, Nclump, Nmax))
                        row_coeff.append(1)
                        row_index.append(get_index_maxL(j, k, t[_n], h, _n, Nchiplet, Nclump, Nmax))
                        row_coeff.append(-1)
        problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["E"], rhs=[-R[s[_n]][t[_n]]])

        for i in range(Nchiplet):
            if (i != s[_n]) and (i != t[_n]):
                row_index, row_coeff = [], []
                for h in range(Nclump):
                    for j in range(Nchiplet):
                        if j != i:
                            for k in range(Nclump):
                                row_index.append(get_index_maxL(i, h, j, k, _n, Nchiplet, Nclump, Nmax))
                                row_coeff.append(1)
                                row_index.append(get_index_maxL(j, k, i, h, _n, Nchiplet, Nclump, Nmax))
                                row_coeff.append(-1)
                problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["E"], rhs=[0])

    # Eq.13 & 14
    for _n in range(Nmax):
        srow_index, srow_coeff = [], []
        trow_index, trow_coeff = [], []
        for h in range(Nclump):
            for j in range(Nchiplet):
                for k in range(Nclump):
                    srow_index.append(get_index_maxL(j, k, s[_n], h, _n, Nchiplet, Nclump, Nmax))
                    srow_coeff.append(1)
                    trow_index.append(get_index_maxL(t[_n], h, j, k, _n, Nchiplet, Nclump, Nmax))
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
                            row_index.append(get_index_maxL(i, h, j, k, _n, Nchiplet, Nclump, Nmax))
                            row_coeff.append(1)
                            row_index.append(get_index_maxL(j, k, i, h, _n, Nchiplet, Nclump, Nmax))
                            row_coeff.append(1)
            problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["L"], rhs=[pmax[i][h]])

    # Eq.16 (indicator): λ = 1 蕴含 f >= 1
    for i in range(Nchiplet):
        for h in range(Nclump):
            for j in range(Nchiplet):
                for k in range(Nclump):
                    for _n in range(Nmax):
                        f_index = get_index_maxL(i, h, j, k, _n, Nchiplet, Nclump, Nmax)
                        problem.indicator_constraints.add(indvar=f_index + 1, rhs=1.0, sense="G",
                                                          lin_expr=[[f_index], [1.0]], indtype=3)

    # Eq.17: λ_max >= d·λ
    # 原 routing_maxL.py 硬编码 ub=100, 对 span>100mm 的大布局会让 num_val 无解。
    # 这里把上界放宽到能覆盖任意两个 clump 的最大曼哈顿距离。
    max_d = max(d[i][h][j][k] for i in range(Nchiplet) for h in range(Nclump)
                for j in range(Nchiplet) for k in range(Nclump))
    ub_maxl = max(100.0, float(math.ceil(max_d) + 5.0))
    problem.variables.add(lb=[0.0], ub=[ub_maxl], types=[problem.variables.type.integer])
    for i in range(Nchiplet):
        for h in range(Nclump):
            for j in range(Nchiplet):
                for k in range(Nclump):
                    for _n in range(Nmax):
                        f_index = get_index_maxL(i, h, j, k, _n, Nchiplet, Nclump, Nmax) + 1
                        problem.linear_constraints.add(lin_expr=[[[f_index, num_val], [-d[i][h][j][k], 1]]],
                                                       senses=["G"], rhs=[0.0])

    # Eq.18
    for _n in range(Nmax):
        row_index, row_coeff = [], []
        if Hopmax == 1:
            for i in range(Nchiplet):
                for h in range(Nclump):
                    for j in range(Nchiplet):
                        for k in range(Nclump):
                            row_index.append(get_index_maxL(i, h, j, k, _n, Nchiplet, Nclump, Nmax))
                            row_coeff.append(1)
            problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["L"], rhs=[R[s[_n]][t[_n]]])
        elif Hopmax == 2:
            for h in range(Nclump):
                for k in range(Nclump):
                    row_index.append(get_index_maxL(s[_n], h, t[_n], k, _n, Nchiplet, Nclump, Nmax))
                    row_coeff.append(2)
                    for i in range(Nchiplet):
                        for j in range(Nchiplet):
                            if i != s[_n] or j != t[_n]:
                                row_index.append(get_index_maxL(i, h, j, k, _n, Nchiplet, Nclump, Nmax))
                                row_coeff.append(1)
            problem.linear_constraints.add(lin_expr=[[row_index, row_coeff]], senses=["L"], rhs=[2 * R[s[_n]][t[_n]]])

    problem.objective.set_linear(num_val, 1.0)

    problem.solve()

    try:
        return problem.solution.get_values()[-1]
    except Exception:
        return 100.0


# --------------------------------------------------------------------------- #
# 数据读取 / 单布局处理
# --------------------------------------------------------------------------- #
def load_range(start_sys: int, end_sys: int) -> dict:
    """从 placement_dataset/placement_dataset_tw/chiplet_dataset_*.json 读取 system_{start}..system_{end}。"""
    records: dict[int, dict] = {}
    k0 = (start_sys - 1) // CHUNK + 1
    k1 = (end_sys - 1) // CHUNK + 1
    for k in range(k0, k1 + 1):
        fp = PLACE_DATASET / f"chiplet_dataset_{k}.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for sid, rec in data.items():
            m = re.fullmatch(r"system_(\d+)", sid)
            if not m:
                continue
            i = int(m.group(1))
            if start_sys <= i <= end_sys:
                records[i] = rec
    return records


def _write_scalar_csv(path: Path, v: float) -> None:
    path.write_text(f"{float(v):.6f}\n", encoding="utf-8")


def process_layout(args) -> tuple[int, dict]:
    i, record, variant = args
    result = {
        "system_id": f"system_{i}",
        "chiplet_count": len(record["chiplets"]),
        "net_count": 0,
        "total_wirelength": None,
        "avg_wirelength": None,
        "max_wirelength": None,
        "avg_solve_time": None,
        "max_solve_time": None,
        "error": None,
    }
    try:
        system = TapSystem(record)
        # net 数 (有向边数)
        result["net_count"] = sum(
            1 for a in range(system.chiplet_count) for b in range(system.chiplet_count)
            if a != b and system.connection_matrix[a][b] > 0
        )

        # avg wirelength (routing.py): min Σ d·f = 总线长, avg = 总线长 / wire_count
        t0 = time.time()
        avg_wl, total_wl = solve_cplex_avg(system)
        result["avg_solve_time"] = time.time() - t0
        result["avg_wirelength"] = float(avg_wl)
        result["total_wirelength"] = None if total_wl is None else float(total_wl)

        # max wirelength (routing_maxL.py), 仅 maxL/both 变体需求时求解
        if variant in ("maxL", "both"):
            t0 = time.time()
            max_wl = solve_cplex_maxL(system)
            result["max_solve_time"] = time.time() - t0
            result["max_wirelength"] = float(max_wl)

        # 写标量 CSV
        _write_scalar_csv(AVG_DIR / f"system_avg_wirelength_{i}.csv", avg_wl)
        if result["total_wirelength"] is not None:
            _write_scalar_csv(TOTAL_DIR / f"system_total_wirelength_{i}.csv", result["total_wirelength"])
        if result["max_wirelength"] is not None:
            _write_scalar_csv(MAX_DIR / f"system_max_wirelength_{i}.csv", result["max_wirelength"])
        _write_scalar_csv(TIME_DIR / f"system_wirelength_time_{i}.csv",
                          result["avg_solve_time"] + (result["max_solve_time"] or 0.0))
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc(limit=2)
    return i, result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=340001)
    ap.add_argument("--end", type=int, default=340010)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variant", choices=["avg", "maxL", "both"], default="avg",
                    help="求解哪种线长: avg(平均+总线长, 默认) / maxL(最大) / both")
    ap.add_argument("--summary", default=str(OUT_DIR / "summary.csv"))
    args = ap.parse_args()

    records = load_range(args.start, args.end)
    items = sorted(records.items())
    print(f"[wirelength] 读取布局 {args.start}..{args.end}: {len(items)} 个 (variant={args.variant})", flush=True)
    if not items:
        print("[wirelength] 无数据, 退出")
        return

    t0 = time.time()
    results: dict[int, dict] = {}
    with Pool(args.workers) as pool:
        for i, res in pool.imap_unordered(process_layout, [(i, rec, args.variant) for i, rec in items]):
            results[i] = res
            err = res.get("error")
            if err:
                print(f"[wirelength] system_{i}: ERROR {err}", flush=True)
            else:
                tot = res["total_wirelength"]
                maxw = res["max_wirelength"]
                print(f"[wirelength] system_{i}: total={tot if tot is not None else 'NA'}mm "
                      f"avg={res['avg_wirelength']:.4f}mm "
                      f"max={maxw if maxw is not None else 'NA'}mm "
                      f"t_avg={res['avg_solve_time']:.2f}s "
                      f"t_max={(res['max_solve_time'] or 0.0):.2f}s", flush=True)

    # 汇总 CSV
    header = ["system_id", "chiplet_count", "net_count", "total_wirelength_mm",
              "avg_wirelength_mm", "max_wirelength_mm", "avg_solve_time_s",
              "max_solve_time_s", "error"]
    rows = []
    for i in sorted(results):
        r = results[i]
        rows.append([r["system_id"], r["chiplet_count"], r["net_count"],
                     r["total_wirelength"], r["avg_wirelength"], r["max_wirelength"],
                     r["avg_solve_time"], r["max_solve_time"], r.get("error", "")])
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")

    ok = sum(1 for r in results.values() if not r.get("error"))
    total_time = sum((r["avg_solve_time"] or 0) + (r["max_solve_time"] or 0) for r in results.values())
    print(f"[wirelength] DONE: {ok}/{len(results)} 成功, 总求解耗时 {total_time:.1f}s "
          f"(墙钟 {time.time() - t0:.1f}s), 汇总 -> {args.summary}", flush=True)


if __name__ == "__main__":
    main()
