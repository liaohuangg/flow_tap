#!/usr/bin/env python3
"""ilp_core.py — ILP 布局基线的核心: 几何常数、画布、模型构建、解提取、记录写盘。

目标函数**只有线长**。面积(画布)是**软约束**: 越界量以极小系数进目标, 塞得下时
自动归零退化成硬约束, 塞不下模型也不会不可行, 且罚项量级远小于线长项、不会主导目标。

决策变量是每个 chiplet 的 footprint 左下角 (FX, FY) 加一个旋转二值 r。
不重叠用 4 二值 big-M (照搬 MPDAP/src/ilp_method_EMIB_chiplet.py:1092-1137 的模式)。

线长用 clump 级精确线性化。关键事实: clump 坐标是 footprint 左下角的**仿射函数**,
偏移只由 (w, h, hubump) 这些数据常数决定 (见 `xform_bases` / `yform_bases`)。所以
clump 间曼哈顿距离 = |ΔFX + c| + |ΔFY + c'|, 常数 c/c' 只依赖 (i, 形态, j, 形态) ——
用辅助变量即可精确表达, **不需要 abs 二值** (9 个 x 形态系数和 9 个 y 形态系数
都严格为正, 最小化会自然把它们压到绝对值)。

线长口径对齐 gen_dataset/gen_wirelength_dataset.py 的 TAP-2.5D 布线 ILP:
那里的 cost 系数就是 clump 间曼哈顿距离, 权重是弧流量。我们做**逐次线性化**:
先用一组 λ 解 MILP, 再用 oracle 取回该解处真实的弧流量, 阻尼更新 λ 重解。

单位: 除画布(ATPlace 参数里是 µm, 加载时 /1000)外, 一律 mm。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

PROJECT = Path(__file__).resolve().parents[2]
CASES_DIR = PROJECT / "benchmark" / "cases_hubump"
ATPLACE_CASES = PROJECT / "baseline" / "ATPlace_pub" / "cases_50set"

# 侧顺序与 gwd.get_input 的 h=0..3 一致: 左/上/右/下
SIDES = ("left", "top", "right", "bottom")
NFORM = 3  # 每根轴上只有 3 个不同形态

# h -> 形态索引 (x 轴, y 轴)。推导见 xform_bases/yform_bases 的 docstring。
SIDE_XFORM = (0, 1, 2, 1)
SIDE_YFORM = (0, 1, 0, 2)


# --------------------------------------------------------------------------- #
# 几何: clump 位置的仿射常数
# --------------------------------------------------------------------------- #
def placed_dims(w: float, h: float, r: int) -> tuple[float, float]:
    """摆放后的本体宽高。r=1 时宽高交换 (与 convert_AT_layout_to_case.py 口径一致)。"""
    return (w + r * (h - w), h + r * (w - h))


def xform_bases(w: float, h: float, u: float):
    """3 个 x 形态的 (常数项, r 的系数): clump_x = FX + base + slope * r。

    由 gwd.get_input 的 xc 公式反推 (那里 xc 是相对 footprint 左下角 xl 的偏移):
        h=0 (左)  : u/2            -> 形态 0
        h=1 (上)  : w/2 + u        -> 形态 1
        h=2 (右)  : w + 1.5u       -> 形态 2
        h=3 (下)  : w/2 + u        -> 形态 1 (同上)
    r=1 时本体宽高交换, 于是 W = w + r(h-w), 形态 1 变成 W/2+u, 形态 2 变成 W+1.5u。
    """
    return ((u / 2.0, 0.0),
            (w / 2.0 + u, (h - w) / 2.0),
            (w + 1.5 * u, (h - w)))


def yform_bases(w: float, h: float, u: float):
    """3 个 y 形态的 (常数项, r 的系数): clump_y = FY + base + slope * r。

        h=0 (左)  : h/2 + u        -> 形态 0
        h=1 (上)  : h + 1.5u       -> 形态 1
        h=2 (右)  : h/2 + u        -> 形态 0 (同上)
        h=3 (下)  : u/2            -> 形态 2
    """
    return ((h / 2.0 + u, (w - h) / 2.0),
            (h + 1.5 * u, (w - h)),
            (u / 2.0, 0.0))


# --------------------------------------------------------------------------- #
# 画布 (软约束的上限)
# --------------------------------------------------------------------------- #
def canvas_mm(case: str, chiplets: list[dict]) -> tuple[float, str]:
    """返回 (画布边长 mm, 来源说明)。

    首选 ATPlace 参数里的 interposer_size (µm, 方形); 没有就用与 ATPlace 一致的
    退路公式 side = sqrt(2 * Σ footprint_w * footprint_h) (mm)。
    """
    path = ATPLACE_CASES / f"{case}_bump" / "Thermal-aware.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        size = raw.get("interposer_size")
        if size:
            # 理论上方形; 万一不是, 取小的那个 (偏保守, 越界由 slack 吸收)
            return min(float(size[0]), float(size[1])) / 1000.0, f"ATPlace:{path.name}"
    area = sum(float(c["footprint_w"]) * float(c["footprint_h"]) for c in chiplets)
    return math.sqrt(2.0 * area), "fallback:sqrt(2*sum(fp_w*fp_h))"


# --------------------------------------------------------------------------- #
# 用例加载
# --------------------------------------------------------------------------- #
def load_case(case: str) -> dict:
    path = CASES_DIR / f"{case}.json"
    if not path.is_file():
        raise SystemExit(f"缺用例: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def case_arrays(record: dict) -> dict:
    """benchmark 记录 -> 建模用的纯数组。"""
    chiplets = record["chiplets"]
    names = [str(c["name"]) for c in chiplets]
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(chiplets)

    # 对称连接矩阵, 复用参考实现的定义 (只累加 wireCount)
    mat = [[0.0] * n for _ in range(n)]
    conns = []
    for conn in record.get("connections", []):
        n1, n2 = str(conn.get("node1", "")), str(conn.get("node2", ""))
        if n1 not in idx or n2 not in idx:
            continue
        i, j = idx[n1], idx[n2]
        wc = float(conn.get("wireCount", 0.0))
        mat[i][j] += wc
        mat[j][i] += wc
        conns.append((i, j, wc))

    return {
        "names": names,
        "n": n,
        "w": [float(c["width"]) for c in chiplets],
        "h": [float(c["height"]) for c in chiplets],
        "u": [float(c["hubump"]) for c in chiplets],
        "power": [float(c["power"]) for c in chiplets],
        "mat": mat,
        "conns": conns,
        "degree": [sum(1 for j in range(n) if mat[i][j] > 0) for i in range(n)],
        "record": record,
    }


def check_case(A: dict) -> list[str]:
    """建模前的前置检查。返回问题列表 (空 = 通过)。"""
    problems = []
    for i in range(A["n"]):
        if A["degree"][i] > 0 and A["u"][i] <= 0:
            problems.append(f"{A['names'][i]}: 有连接但 hubump={A['u'][i]} -> 参考 ILP 会不可行")
        if A["u"][i] < 0:
            problems.append(f"{A['names'][i]}: hubump 为负 ({A['u'][i]})")
        if A["w"][i] <= 0 or A["h"][i] <= 0:
            problems.append(f"{A['names'][i]}: 尺寸非正 ({A['w'][i]}, {A['h'][i]})")
    return problems


# --------------------------------------------------------------------------- #
# λ 权重: 把弧流量折叠到 3x3 形态网格
# --------------------------------------------------------------------------- #
def lam_to_coeffs(A: dict, lam: dict) -> tuple[dict, float]:
    """λ[(i,h,j,k)] -> 目标的系数。

    λ 以**无序对** (i<j) 为键、已把两个方向合起来 (见 wl_oracle.arc_flow)。
    因为 clump 距离可加性地拆成 x 部分 + y 部分, 折叠到形态网格是精确的:
        λX[i,j][a][b] = Σ_{h: xf(h)=a} Σ_{k: xf(k)=b} λ[i,h,j,k]
        λY[i,j][c][d] = Σ_{h: yf(h)=c} Σ_{k: yf(k)=d} λ[i,h,j,k]
    返回 (系数, λ 总和)。系数里 9 个 x 和 9 个 y 都 > 0, 这正是免 abs 二值的前提。
    """
    coeffs = {}
    total = 0.0
    for (i, j), table in lam.items():
        lx = [[0.0] * NFORM for _ in range(NFORM)]
        ly = [[0.0] * NFORM for _ in range(NFORM)]
        for h in range(4):
            for k in range(4):
                val = table[h][k]
                if val <= 0.0:
                    continue
                total += val
                lx[SIDE_XFORM[h]][SIDE_XFORM[k]] += val
                ly[SIDE_YFORM[h]][SIDE_YFORM[k]] += val
        coeffs[(i, j)] = (lx, ly)
    return coeffs, total


def uniform_lam(A: dict) -> dict:
    """均匀初值: 每个无序对的总流量 2*wc 平摊到 16 个 (h,k) 组合。"""
    lam = {}
    for i in range(A["n"]):
        for j in range(i + 1, A["n"]):
            wc = A["mat"][i][j]
            if wc <= 0:
                continue
            lam[(i, j)] = [[wc / 8.0] * 4 for _ in range(4)]
    return lam


# --------------------------------------------------------------------------- #
# MILP
# --------------------------------------------------------------------------- #
class IlpModel:
    """一次 (λ 固定) 的 MILP。逐次线性化时每轮重建。"""

    def __init__(self, A: dict, S: float, lam: dict, allow_rot: bool = True,
                 big_m_pad: float = 1.0, objective: str = "wl",
                 w_wl: float | None = None, w_bbox: float = 0.0,
                 w_aspect: float = 0.0, wl_ref: float | None = None,
                 hard_canvas: bool = False):
        """objective: 'wl' 只优化线长 | 'bbox' 优化 W+H 与长宽比。

        hard_canvas=False (软): 画布写成 `FX+FW <= S + slack`, slack 以 1e-6 的系数
            进目标。能塞下时 slack 自动归零、等价于硬约束; 塞不下时模型仍可行。
            但 1e-6 小到等于没约束 —— 只压线长时求解器会毫不犹豫地破 13.5mm 画布。
        hard_canvas=True  (硬): slack 固定为 0, `FX+FW <= S` 直接进约束集。
            中介层面积在项目口径里本来就是**硬约束, 超限即不可行**, 用这个。

        目标各项都先归一化成无量纲, 权重之间才有可比性:
          线长       -> 除以 wl_ref (greedy 起点的精确参考代价)
          W+H / |W-H| / 画布越界量 -> 除以画布边长 S
        """
        self.A = A
        self.S = S
        self.n = A["n"]
        self.objective = objective
        self.coeffs, self.lam_sum = lam_to_coeffs(A, lam)

        if w_wl is None:
            w_wl = 1.0 if objective == "wl" else 0.0
        self.w_wl, self.w_bbox, self.w_aspect = w_wl, w_bbox, w_aspect
        self.wl_ref = float(wl_ref) if wl_ref else 1.0

        self.hard_canvas = hard_canvas
        # 越界罚 / 无连接芯粒的平局打破: 归一化后固定取极小值 —— 是"罚"不是"目标"
        self.beta_area = 1e-6
        self.eps_deg = 1e-9

        max_dim = max(max(A["w"][i], A["h"][i]) + 2 * A["u"][i] for i in range(self.n))
        self.max_dim = max_dim
        # big-M: 坐标上界 2S (见下面 FX 的 bound), 故 M >= 2S + max_dim
        self.M = 2.0 * S + max_dim + big_m_pad

        self.m = gp.Model("ilp_place")
        self.m.Params.OutputFlag = 0
        self.m.Params.Threads = 1
        self.m.Params.Seed = 0
        self.m.Params.NumericFocus = 1

        self._build(allow_rot)

    def _build(self, allow_rot: bool):
        A, m, n, S = self.A, self.m, self.n, self.S
        M = self.M

        # 坐标: 允许越界, 上界取 2S (slack 最大 S, 故 FX+FW <= S+S=2S 可行)
        self.FX = m.addVars(n, lb=0.0, ub=2.0 * S, name="FX")
        self.FY = m.addVars(n, lb=0.0, ub=2.0 * S, name="FY")

        # 旋转: 宽高相同的固定为 0, 省二值
        self.r = {}
        for i in range(n):
            if allow_rot and abs(A["w"][i] - A["h"][i]) > 1e-9:
                self.r[i] = m.addVar(vtype=GRB.BINARY, name=f"r{i}")
            else:
                self.r[i] = m.addVar(lb=0.0, ub=0.0, name=f"r{i}_fixed")

        # 画布: 软约束留 slack, 硬约束把 slack 钉死在 0 (约束式完全一样, 只是 ub 不同)
        slack_ub = 0.0 if self.hard_canvas else S
        self.slack_x = m.addVar(lb=0.0, ub=slack_ub, name="slack_x")
        self.slack_y = m.addVar(lb=0.0, ub=slack_ub, name="slack_y")

        # 摆放后的 footprint 尺寸表达式
        self.FW, self.FH = {}, {}
        for i in range(n):
            w, h, u = A["w"][i], A["h"][i], A["u"][i]
            self.FW[i] = (w + 2 * u) + self.r[i] * (h - w)
            self.FH[i] = (h + 2 * u) + self.r[i] * (w - h)
            m.addConstr(self.FX[i] + self.FW[i] <= S + self.slack_x, name=f"canvas_x{i}")
            m.addConstr(self.FY[i] + self.FH[i] <= S + self.slack_y, name=f"canvas_y{i}")

        # 对称破缺: 整体可平移, 故 w.l.o.g. min FX = min FY = 0
        self.tX = m.addVars(n, vtype=GRB.BINARY, name="tX")
        self.tY = m.addVars(n, vtype=GRB.BINARY, name="tY")
        for i in range(n):
            m.addConstr(self.FX[i] <= 2.0 * S * (1 - self.tX[i]), name=f"sb_x{i}")
            m.addConstr(self.FY[i] <= 2.0 * S * (1 - self.tY[i]), name=f"sb_y{i}")
        m.addConstr(self.tX.sum() >= 1, name="sb_x_atleast1")
        m.addConstr(self.tY.sum() >= 1, name="sb_y_atleast1")

        # 不重叠 (footprint): 每对无序 i<j 四个二值, pL+pR+pD+pU >= 1
        self.pair_bin = []
        for i in range(n):
            for j in range(i + 1, n):
                pL = m.addVar(vtype=GRB.BINARY, name=f"pL_{i}_{j}")
                pR = m.addVar(vtype=GRB.BINARY, name=f"pR_{i}_{j}")
                pD = m.addVar(vtype=GRB.BINARY, name=f"pD_{i}_{j}")
                pU = m.addVar(vtype=GRB.BINARY, name=f"pU_{i}_{j}")
                m.addConstr(self.FX[i] + self.FW[i] <= self.FX[j] + M * (1 - pL))
                m.addConstr(self.FX[j] + self.FW[j] <= self.FX[i] + M * (1 - pR))
                m.addConstr(self.FY[i] + self.FH[i] <= self.FY[j] + M * (1 - pD))
                m.addConstr(self.FY[j] + self.FH[j] <= self.FY[i] + M * (1 - pU))
                m.addConstr(pL + pR + pD + pU >= 1)
                self.pair_bin.append((i, j, pL, pR, pD, pU))

        # 线长: clump 形态级的 |Δ|, 每连通对 9 个 x + 9 个 y
        self.xb = [xform_bases(A["w"][i], A["h"][i], A["u"][i]) for i in range(n)]
        self.yb = [yform_bases(A["w"][i], A["h"][i], A["u"][i]) for i in range(n)]

        self.AXE, self.AYE = {}, {}
        obj = gp.LinExpr()

        # --- 线长项 (clump 形态级的 |Δ|) ------------------------------------- #
        if self.w_wl > 0.0 and self.coeffs:
            wl = gp.LinExpr()
            for (i, j), (lx, ly) in self.coeffs.items():
                axe = self.AXE[(i, j)] = {}
                aye = self.AYE[(i, j)] = {}
                for a in range(NFORM):
                    for b in range(NFORM):
                        v = m.addVar(lb=0.0, name=f"AXE_{i}_{j}_{a}_{b}")
                        axe[(a, b)] = v
                        # clump_x = FX + base + slope*r
                        lhs_i = self.FX[i] + self.xb[i][a][0] + self.xb[i][a][1] * self.r[i]
                        lhs_j = self.FX[j] + self.xb[j][b][0] + self.xb[j][b][1] * self.r[j]
                        m.addConstr(v >= lhs_i - lhs_j)
                        m.addConstr(v >= -lhs_i + lhs_j)
                        if lx[a][b] > 0:
                            wl += lx[a][b] * v
                for c in range(NFORM):
                    for d in range(NFORM):
                        v = m.addVar(lb=0.0, name=f"AYE_{i}_{j}_{c}_{d}")
                        aye[(c, d)] = v
                        lhs_i = self.FY[i] + self.yb[i][c][0] + self.yb[i][c][1] * self.r[i]
                        lhs_j = self.FY[j] + self.yb[j][d][0] + self.yb[j][d][1] * self.r[j]
                        m.addConstr(v >= lhs_i - lhs_j)
                        m.addConstr(v >= -lhs_i + lhs_j)
                        if ly[c][d] > 0:
                            wl += ly[c][d] * v
            obj += (self.w_wl / self.wl_ref) * wl

        # --- 外接框项 --------------------------------------------------------- #
        # 对称破缺已保证 min FX = min FY = 0, 所以 W = maxX, H = maxY, 不用再取差。
        self.maxX = m.addVar(lb=0.0, ub=2.0 * S, name="maxX")
        self.maxY = m.addVar(lb=0.0, ub=2.0 * S, name="maxY")
        for i in range(n):
            m.addConstr(self.maxX >= self.FX[i] + self.FW[i], name=f"bbox_maxx{i}")
            m.addConstr(self.maxY >= self.FY[i] + self.FH[i], name=f"bbox_maxy{i}")

        if self.w_bbox > 0.0:
            obj += (self.w_bbox / S) * (self.maxX + self.maxY)   # W + H

        if self.w_aspect > 0.0:
            # |W - H| 用同一套免二值的 abs 技巧: 两条 >= 约束, 系数为正
            self.aspect = m.addVar(lb=0.0, ub=2.0 * S, name="aspect")
            m.addConstr(self.aspect >= self.maxX - self.maxY, name="aspect_ab")
            m.addConstr(self.aspect >= self.maxY - self.maxX, name="aspect_ba")
            obj += (self.w_aspect / S) * self.aspect

        # 面积软约束罚 + 平局打破
        obj += self.beta_area * (self.slack_x + self.slack_y) / S
        for i in range(n):
            if A["degree"][i] == 0:
                obj += self.eps_deg * (self.FX[i] + self.FY[i]) / S

        m.setObjective(obj, GRB.MINIMIZE)
        self.obj_expr = obj
        # Gurobi 要先 update() 才能查到 NumVars/NumBinVars/NumConstrs,
        # 否则一律返回 0 (踩过)。
        m.update()

    # -- 求解 --------------------------------------------------------------- #
    def solve(self, stages, warm_start: list[tuple[float, float, int]] | None = None) -> dict:
        """stages: [(预算秒, MIPGap, MIPFocus), ...]。返回每段的解。"""
        if warm_start is not None:
            for i, (fx, fy, r) in enumerate(warm_start):
                self.FX[i].Start = fx
                self.FY[i].Start = fy
                self.r[i].Start = float(r)

        results = []
        for budget, gap, focus in stages:
            self.m.Params.TimeLimit = float(budget)
            self.m.Params.MIPGap = float(gap)
            self.m.Params.MIPFocus = int(focus)
            self.m.optimize()
            st = self.m.Status
            rec = {
                "status": st,
                "status_name": {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
                                GRB.INFEASIBLE: "INFEASIBLE", GRB.INF_OR_UNBD: "INF_OR_UNBD",
                                GRB.INTERRUPTED: "INTERRUPTED"}.get(st, str(st)),
                "budget_s": budget,
                "gap": gap,
                "focus": focus,
            }
            if self.m.SolCount > 0:
                rec["mip_gap"] = float(self.m.MIPGap)
                rec["best_bound"] = float(self.m.ObjBound)
                rec["obj_model"] = float(self.m.ObjVal)
                rec["x"] = self.extract()
            else:
                rec["x"] = None
            results.append(rec)
        return {"stages": results}

    def extract(self) -> list[tuple[float, float, int]]:
        out = []
        for i in range(self.n):
            out.append((float(self.FX[i].X), float(self.FY[i].X), int(round(self.r[i].X))))
        return out

    def model_obj_at(self, x) -> float:
        """把一组坐标代回目标表达式 (用于对齐 / 诊断)。"""
        vals = {}
        for i, (fx, fy, r) in enumerate(x):
            vals[self.FX[i]] = fx
            vals[self.FY[i]] = fy
            vals[self.r[i]] = float(r)
        return float(self.obj_expr.getValue())

    def stats(self) -> dict:
        return {
            "n_vars": self.m.NumVars,
            "n_bin": self.m.NumBinVars,
            "n_constrs": self.m.NumConstrs,
            "M": self.M,
            "S": self.S,
            "objective": self.objective,
            "w_wl": self.w_wl, "w_bbox": self.w_bbox, "w_aspect": self.w_aspect,
            "wl_ref": self.wl_ref,
            "beta_area": self.beta_area,
            "hard_canvas": self.hard_canvas,
            "lam_sum": self.lam_sum,
        }


# --------------------------------------------------------------------------- #
# 记录写盘 (契约格式, 模板 = resultEval/RL_result/format_result/acend910_seed2.json)
# --------------------------------------------------------------------------- #
def footprints_overlap(A: dict, x) -> list[tuple[int, int, float, float]]:
    """返回所有 footprint 重叠对 (i, j, 重叠宽, 重叠高)。"""
    bad = []
    for i in range(A["n"]):
        for j in range(i + 1, A["n"]):
            fxi, fyi, ri = x[i]
            fxj, fyj, rj = x[j]
            wi, hi = footprint_size(A, i, ri)
            wj, hj = footprint_size(A, j, rj)
            ox = min(fxi + wi, fxj + wj) - max(fxi, fxj)
            oy = min(fyi + hi, fyj + hj) - max(fyi, fyj)
            if ox > 1e-9 and oy > 1e-9:
                bad.append((i, j, ox, oy))
    return bad


def canvas_overflow(A: dict, x, S: float) -> tuple[float, float]:
    """(x 方向越界量, y 方向越界量)。软约束下越界量就是 slack 的取值。

    两个都是 0 说明画布没被触发 —— 此时"软约束"和"硬约束"给出完全相同的结果。
    """
    ox = oy = 0.0
    for i, (fx, fy, r) in enumerate(x):
        fw, fh = footprint_size(A, i, r)
        ox = max(ox, fx + fw - S)
        oy = max(oy, fy + fh - S)
    return max(ox, 0.0), max(oy, 0.0)


def footprint_size(A: dict, i: int, r: int) -> tuple[float, float]:
    """chiplet i 旋转 r 后的 footprint (含 hubump) 尺寸。"""
    w, h, u = A["w"][i], A["h"][i], A["u"][i]
    return (w + 2 * u + r * (h - w), h + 2 * u + r * (w - h))


def bbox_mm(A: dict, x) -> tuple[float, float, float]:
    """(宽, 高, 面积) mm —— 与 eval_layout 的 bbox_area_mm2 同口径 (含 hubump)。"""
    xs, ys = [], []
    for i, (fx, fy, r) in enumerate(x):
        fw, fh = footprint_size(A, i, r)
        xs.extend((fx, fx + fw))
        ys.extend((fy, fy + fh))
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    return w, h, w * h


SEPARATE_TOL = 1e-4   # 只处理取整级微重叠; 更大的重叠说明布局真有问题, 不该在这里被掩盖


def separate_rounding(A: dict, x, margin: float = 1e-5, max_iter: int = 200):
    """取整会让正好贴边的布局产生极小重叠, 沿重叠小的那根轴把一方推开。

    容差别取太小: round(...,6) 每个坐标最多偏 5e-7, 两个贴边芯粒的重叠因此可达 1e-6 ——
    早期用 5e-7 会把这种必然产物当成"真重叠"而放弃整份解, 白白丢掉更优的布局。

    只处理 <= SEPARATE_TOL 的微重叠; 更大的原样返回, 由调用方判定并丢弃。
    """
    x = [list(t) for t in x]
    for _ in range(max_iter):
        bad = footprints_overlap(A, x)
        if not bad:
            break
        moved = False
        for i, j, ox, oy in bad:
            if max(ox, oy) > SEPARATE_TOL:
                continue
            # 沿重叠较小的那根轴分开 (位移代价小); 谁在左边/下边就推谁
            axis = 0 if ox <= oy else 1
            lo, hi = (i, j) if x[i][axis] <= x[j][axis] else (j, i)
            x[lo][axis] -= margin
            moved = True
        if not moved:
            break
    return [tuple(t) for t in x]


def write_record(A: dict, x, case: str, seed: int, out_path: Path) -> dict:
    """把 (FX, FY, r) 写成契约格式并落盘。返回写出的记录。"""
    chiplets = []
    for i, (fx, fy, r) in enumerate(x):
        w, h, u = A["w"][i], A["h"][i], A["u"][i]
        pw, ph = placed_dims(w, h, r)
        # x-position/y-position = 本体左下角 = footprint 左下角 + hubump
        rec = {
            "name": A["names"][i],
            "x-position": round(fx + u, 6),
            "y-position": round(fy + u, 6),
            "width": round(pw, 6),
            "height": round(ph, 6),
            "rotation": int(r),
            "power": A["power"][i],
            "hubump": u,
        }
        # 旋转必须已体现在宽高里 —— 忘了交换会静默得到偏小的 bbox
        assert abs(rec["width"] - pw) < 1e-9 and abs(rec["height"] - ph) < 1e-9, \
            f"{A['names'][i]}: 宽高未按旋转交换"
        chiplets.append(rec)

    connections = [{"node1": A["names"][i], "node2": A["names"][j], "wireCount": wc}
                   for i, j, wc in A["conns"]]

    record = {"system_id": f"{case}_seed{seed}", "chiplets": chiplets,
              "connections": connections}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return record


def verify_record(A: dict, record: dict) -> None:
    """写盘后自查: TapSystem 重算的 hubump 必须与记录里的相等。

    这条断言用来抓将来 benchmark 被改动 (今天 16 个 case 都成立)。
    """
    import wl_oracle
    sys_ = wl_oracle.build_system(record)
    for i, c in enumerate(record["chiplets"]):
        if abs(sys_.hubump[i] - float(c["hubump"])) > 1e-9:
            raise AssertionError(
                f"hubump 不一致 {c['name']}: 记录 {c['hubump']} vs 重算 {sys_.hubump[i]}")


# --------------------------------------------------------------------------- #
# 第 1 步验证: 仿射 clump 常数 vs gwd.get_input
# --------------------------------------------------------------------------- #
def selftest(samples: int = 500, seed: int = 0) -> int:
    import random
    sys.path.insert(0, str(PROJECT / "gen_dataset"))
    import gen_wirelength_dataset as gwd

    rng = random.Random(seed)
    worst = 0.0
    checked = 0
    for _ in range(samples):
        w = rng.uniform(1.0, 30.0)
        h = rng.uniform(1.0, 30.0)
        u = rng.choice([0.0, 0.045, 0.09, 0.135, rng.uniform(0.0, 0.5)])
        r = rng.choice([0, 1])
        fx, fy = rng.uniform(0.0, 50.0), rng.uniform(0.0, 50.0)

        pw, ph = placed_dims(w, h, r)
        rec = {"chiplets": [{"name": "A", "x-position": fx + u, "y-position": fy + u,
                             "width": pw, "height": ph, "power": 1.0, "hubump": u}],
               "connections": []}
        system = gwd.TapSystem(rec, hubump_mode="stored")
        xl, xc, yl, yc, _R, _n, _nc, _pmax, _hop = gwd.get_input(system)

        # 注意: get_input 返回的 xc/yc 是**相对 footprint 左下角的偏移**, 绝对位置 =
        # xl + xc —— 见 solve_cplex_avg 里 d 的定义 (gen_wirelength_dataset.py:274)。
        xb = xform_bases(w, h, u)
        yb = yform_bases(w, h, u)
        for side in range(4):
            exp_x = fx + xb[SIDE_XFORM[side]][0] + xb[SIDE_XFORM[side]][1] * r
            exp_y = fy + yb[SIDE_YFORM[side]][0] + yb[SIDE_YFORM[side]][1] * r
            worst = max(worst, abs(xl[0] + xc[0][side] - exp_x),
                        abs(yl[0] + yc[0][side] - exp_y))
            checked += 1
        # footprint 左下角也要对上
        worst = max(worst, abs(xl[0] - fx), abs(yl[0] - fy))

        # 模型真正依赖的形态划分: 同一形态内的 clump 偏移必须完全一致
        for axis, forms, offs in (("x", SIDE_XFORM, xb), ("y", SIDE_YFORM, yb)):
            for f in range(NFORM):
                got = {(round(xc[0][s] - offs[forms[s]][0] - offs[forms[s]][1] * r, 12)
                        if axis == "x" else
                        round(yc[0][s] - offs[forms[s]][0] - offs[forms[s]][1] * r, 12))
                       for s in range(4) if forms[s] == f}
                if got != {0.0}:
                    raise AssertionError(f"{axis} 形态 {f} 内的 clump 偏移不一致: {got}")

    ok = worst < 1e-9
    print(f"selftest: {checked} 个 clump 位置 + {samples} 个 footprint 角, "
          f"最大误差 {worst:.3e}  -> {'通过' if ok else '失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ilp_core 自检")
    ap.add_argument("--selftest", action="store_true", help="验证仿射 clump 常数")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    ap.print_help()
