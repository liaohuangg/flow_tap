#!/usr/bin/env python3
"""opt_dataset_lib.py — 生成「热峰值 / 线长 / bbox 面积」帕累托布局数据集的公共库。

核心表示
--------
把每个 system 拆成 **抽象条件 (冻结)** + **一次实现 (自由)**:

  · 冻结: 槽位的 footprint 多重集 {(fw,fh)}、功耗多重集 {P}、抽象图 (边集 + 边权多重集)
  · 自由: 双射 σ (抽象图节点 -> 槽位)、功耗置换 ρ、bbox 等距变换 (镜像 x/y、转置)

于是 `after_dataset_process` 是本方案的一个特例: σ = QAP 解, ρ = 按外围度的 density 排布,
几何 = 恒等。本库只是把这三个自由度都打开, 再用两个冻结代理来*选择*。

坐标口径 (与 after_dataset_process 逐字一致)
--------------------------------------------
记录存 **body** 坐标 + hubump; footprint = body + 2*hubump, 而 **footprint 是冻结的**。
重算 hubump 只让 body 四边往内缩, footprint 与整个铺排都不动 —— 所以"非重叠"由构造保持,
不需要任何重放置。

两种"重叠"必须分清 (实测)
--------------------------
`gen_legal_pla_greedy._rects_overlap` 把 **边界贴合** 也算重叠。拿它当闸门会误杀真实数据:
tw chunk1 有 348/400 个 system 存在贴合, opt chunk1 有 347/400。而 **实质重叠 (双轴都
有正重叠) 全部为 0**。所以本库的合法性判据是 **实质重叠**: `min(ox, oy) > FP_EPS` 才算非法。
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT = Path("/root/placement/flow_tap")
GEN_DIR = Path(__file__).resolve().parent          # gen_pareto/ (本包)
GEN_PARENT = GEN_DIR.parent                          # gen_dataset/ (after_dataset_process, proxy_eval 依赖)
for _p in (str(PROJECT), str(PROJECT / "LayoutGenModel" / "diffusion"),
           str(GEN_DIR), str(GEN_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 该环境的 wandb 0.13 依赖 numpy<2 的别名, 而实际装的是 numpy 2.x。
# 与 proxy_eval.py 同一段补丁, 必须在 torch/wandb 相关 import 之前。
for _alias, _target in (("float_", np.float64), ("int_", np.int64), ("complex_", np.complex128),
                        ("unicode_", np.str_), ("str_", np.str_), ("bool8", np.bool_),
                        ("object_", object), ("long", np.int64)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

import after_dataset_process as adp  # noqa: E402  (以库的形式复用, 不修改)

CHUNK = 5000
UBUMP_PITCH = adp.UBUMP_PITCH            # 0.045 mm
GRID = 0.001
FP_EPS = 1e-6                            # footprint "实质重叠" 阈值 (mm)
SIG_EPS = 1e-9
CANVAS_PADDING_MM = 1.0                  # 与 proxy_eval._to_cond / PlacementManifestDataset 一致
WIRE_COUNTS = (128.0, 256.0, 512.0, 1024.0)

Layout = List[Tuple[float, float, float, float]]   # [(x, y, fw, fh)] 按槽位顺序


def snap001(v: float) -> float:
    """吸附到 0.001mm 栅格。与 gen_legal_pla_greedy.snap001 同实现 (此处避免 import tool)。"""
    return float(int(round(v * 1000.0))) / 1000.0


# ======================================================================================
# 几何
# ======================================================================================
def footprint_of(chiplet: dict) -> Tuple[float, float, float, float]:
    """记录里的一个 chiplet -> 它的 footprint (x, y, fw, fh)。"""
    hu = float(chiplet["hubump"])
    return (float(chiplet["x-position"]) - hu,
            float(chiplet["y-position"]) - hu,
            float(chiplet["width"]) + 2.0 * hu,
            float(chiplet["height"]) + 2.0 * hu)


def fp_pair_depth(a, b) -> float:
    """两个 footprint 的重叠"深度" = min(ox, oy)。<= 0 表示不重叠 (贴合算不重叠)。"""
    ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return min(ox, oy)


def fp_max_depth(fp_list: Sequence) -> float:
    """铺排里最深的一处实质重叠 (mm)。<=0 表示合法。"""
    worst = -float("inf")
    n = len(fp_list)
    for i in range(n):
        for j in range(i + 1, n):
            worst = max(worst, fp_pair_depth(fp_list[i], fp_list[j]))
    return worst


def bbox_span(fp_list: Sequence) -> Tuple[float, float]:
    x0 = min(f[0] for f in fp_list)
    y0 = min(f[1] for f in fp_list)
    x1 = max(f[0] + f[2] for f in fp_list)
    y1 = max(f[1] + f[3] for f in fp_list)
    return (x1 - x0, y1 - y0)


def bbox_area(fp_list: Sequence) -> float:
    w, h = bbox_span(fp_list)
    return w * h


def canvas_side(fp_list: Sequence) -> float:
    """与 proxy_eval._to_cond 的口径逐字一致: max(footprint bbox 两个跨度) + 1.0mm。

    注意它是 **footprint** 的函数, 与 hubump 无关 —— 这正是"同一个 system 的候选共享画布"的原因。
    """
    w, h = bbox_span(fp_list)
    return max(w, h) + CANVAS_PADDING_MM


# ======================================================================================
# 等距变换 (G3): 保持 footprint 多重集 / bbox 面积 / 画布边长不变
# ======================================================================================
def mirror_x(fp: Layout) -> Layout:
    x0 = min(f[0] for f in fp)
    x1 = max(f[0] + f[2] for f in fp)
    return [(x0 + x1 - f[0] - f[2], f[1], f[2], f[3]) for f in fp]


def mirror_y(fp: Layout) -> Layout:
    y0 = min(f[1] for f in fp)
    y1 = max(f[1] + f[3] for f in fp)
    return [(f[0], y0 + y1 - f[1] - f[3], f[2], f[3]) for f in fp]


def transpose(fp: Layout) -> Layout:
    """关于对角线 y=x 的镜像: (x,y,fw,fh) -> (y,x,fh,fw)。整幅铺排的等距变换。"""
    return [(f[1], f[0], f[3], f[2]) for f in fp]


# ======================================================================================
# 抽象条件 (SystemCore)
# ======================================================================================
@dataclass
class SystemCore:
    """一个 system 的冻结部分 + 其 L0 参考实现。"""

    sid: str
    names: List[str]
    fp: Layout                    # 槽位 -> footprint (冻结)
    rotation: List[int]
    edges: List[Tuple[int, int, float]]   # 抽象图: 节点索引 = 源槽位下标, 冻结
    powers: List[float]                   # 槽位 -> 参考功耗 (多重集冻结)
    ref: dict                             # L0 记录本身 (C0, 逐字节可达)

    @property
    def n(self) -> int:
        return len(self.names)

    def maxcap(self) -> List[float]:
        return [adp.max_capacity(f[2], f[3]) for f in self.fp]

    def dist(self) -> List[List[float]]:
        c = [(f[0] + f[2] / 2.0, f[1] + f[3] / 2.0) for f in self.fp]
        return [[abs(c[i][0] - c[j][0]) + abs(c[i][1] - c[j][1]) for j in range(self.n)]
                for i in range(self.n)]

    def demand(self) -> List[float]:
        return adp._demands(self.edges, self.n)


def load_core(sid: str, rec: dict) -> Optional[SystemCore]:
    """从一条记录构建 SystemCore。返回 None 表示该记录不被支持 (跳过)。"""
    chiplets = rec.get("chiplets") or []
    if not chiplets:
        return None
    names = [str(c["name"]) for c in chiplets]
    if len(set(names)) != len(names):
        return None
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    edges, seen = [], set()
    for e in rec.get("connections", []):
        a, b = name_to_idx.get(e["node1"]), name_to_idx.get(e["node2"])
        if a is None or b is None or a == b:
            return None
        key = (min(a, b), max(a, b))
        if key in seen:          # 重复的无向边不在语料里; 出现则视为不支持
            return None
        seen.add(key)
        edges.append((key[0], key[1], float(e["wireCount"])))

    return SystemCore(
        sid=sid,
        names=names,
        fp=[footprint_of(c) for c in chiplets],
        rotation=[int(c["rotation"]) for c in chiplets],
        edges=sorted(edges),
        powers=[float(c["power"]) for c in chiplets],
        ref=rec,
    )


# ======================================================================================
# 记录构造
# ======================================================================================
def make_record(core: SystemCore,
                sigma: Optional[Sequence[int]] = None,
                power_slots: Optional[Sequence[float]] = None,
                fp: Optional[Layout] = None,
                rotation: Optional[Sequence[int]] = None,
                sid: Optional[str] = None) -> Optional[dict]:
    """按 (σ, 功耗置换, 几何) 造一条合法记录。不可行 (hubump 无解 / body 边长<=0) 返回 None。

    · sigma: 长度 n 的双射, 候选边 = [(sigma[u], sigma[v], w) for (u,v,w) in core.edges]
    · power_slots: 长度 n 的功耗列表 (必须是 core.powers 的一个置换)
    · fp / rotation: 槽位的 footprint 与朝向 (默认取 core 的, 即几何不变)
    """
    n = core.n
    sigma = list(range(n)) if sigma is None else list(sigma)
    if sorted(sigma) != list(range(n)):
        raise ValueError("sigma 必须是 0..n-1 的双射")
    power_slots = list(core.powers) if power_slots is None else list(power_slots)
    fp = core.fp if fp is None else list(fp)
    rotation = core.rotation if rotation is None else list(rotation)

    edges_slots = [(sigma[u], sigma[v], w) for u, v, w in core.edges]

    s_sum = [0.0] * n
    for u, v, w in edges_slots:
        s_sum[u] += w
        s_sum[v] += w

    chiplets = []
    for i in range(n):
        hu = adp.solve_hubump(fp[i][2], fp[i][3], 2.0 * s_sum[i])
        if hu is None:
            return None
        x, y = snap001(fp[i][0] + hu), snap001(fp[i][1] + hu)
        w, h = snap001(fp[i][2] - 2.0 * hu), snap001(fp[i][3] - 2.0 * hu)
        if w <= 0.0 or h <= 0.0:
            return None
        chiplets.append({
            "name": core.names[i],
            "x-position": x,
            "y-position": y,
            "width": w,
            "height": h,
            "rotation": int(rotation[i]),
            "power": float(power_slots[i]),
            "hubump": round(hu, 6),
        })

    out = {
        "system_id": sid if sid is not None else core.sid,
        "chiplets": chiplets,
        "connections": [
            {"node1": core.names[u], "node2": core.names[v], "wireCount": int(w)}
            for u, v, w in edges_slots
        ],
    }
    return out


# ======================================================================================
# 不变量
# ======================================================================================
def signature(rec: dict) -> dict:
    """与 σ 无关的抽象条件签名, 用于跨记录比较 (I1-I4)。"""
    fp = [footprint_of(c) for c in rec["chiplets"]]
    edges = sorted((min(e["node1"], e["node2"]), max(e["node1"], e["node2"]), float(e["wireCount"]))
                   for e in rec.get("connections", []))
    deg: Dict[str, float] = {}
    for a, b, w in edges:
        deg[a] = deg.get(a, 0.0) + w
        deg[b] = deg.get(b, 0.0) + w
    # footprint 用 **无序对**: 转置 (x,y,fw,fh)->(y,x,fh,fw) 是整幅铺排的等距变换, 它把每颗
    # 芯粒的摆放尺寸 (fw,fh) 换成 (fh,fw)、同时翻转 rotation —— 芯片还是同一颗芯片, 只是朝向声明
    # 变了。用有序对会把它误判成"改了 footprint"。
    return {
        "n": len(fp),
        "fp_multiset": sorted(sorted((round(f[2], 6), round(f[3], 6))) for f in fp),
        "power_multiset": sorted(round(float(c["power"]), 6) for c in rec["chiplets"]),
        "edge_multiset": sorted(w for _, _, w in edges),
        "weighted_degree_multiset": sorted(round(v, 6) for v in deg.values()),
        "m": len(edges),
    }


def assert_invariants(rec: dict, core: SystemCore, sigma: Optional[Sequence[int]] = None,
                      expect_edges_slots: Optional[Sequence[Tuple[int, int, float]]] = None) -> List[str]:
    """逐条硬断言 I1-I9。返回违反列表 (空 = 通过)。"""
    bad: List[str] = []
    n = core.n
    chiplets = rec["chiplets"]
    if len(chiplets) != n:
        bad.append(f"I1 |V| {len(chiplets)} != {n}")
        return bad

    fp = [footprint_of(c) for c in chiplets]
    ref_sig = signature(core.ref)
    sig = signature(rec)

    # I1 footprint 多重集 (footprint 尺寸必须是整数, 4..30)
    if sig["fp_multiset"] != ref_sig["fp_multiset"]:
        bad.append("I1 footprint 多重集改变")
    for i, f in enumerate(fp):
        if abs(f[2] - round(f[2])) > 1e-6 or abs(f[3] - round(f[3])) > 1e-6:
            bad.append(f"I1 槽位{i} footprint 非整数 ({f[2]},{f[3]})")
        if not (4.0 - 1e-6 <= f[2] <= 30.0 + 1e-6 and 4.0 - 1e-6 <= f[3] <= 30.0 + 1e-6):
            bad.append(f"I1 槽位{i} footprint 越界 ({f[2]},{f[3]})")

    # I2 功耗多重集
    if sig["power_multiset"] != ref_sig["power_multiset"]:
        bad.append("I2 功耗多重集改变")

    # I3/I4 抽象图: 给出 σ 就逐边精确校验, 否则退回多重集签名
    if sigma is not None:
        # 逐对规范化 (min,max): 图是无向的, 而 σ 不保序 (sigma[u] 未必 < sigma[v])
        want = sorted((min(sigma[u], sigma[v]), max(sigma[u], sigma[v]), w)
                      for u, v, w in core.edges)
        got = []
        idx = {nm: i for i, nm in enumerate(core.names)}
        for e in rec.get("connections", []):
            a, b = idx[e["node1"]], idx[e["node2"]]
            got.append((min(a, b), max(a, b), float(e["wireCount"])))
        if sorted(got) != want:
            bad.append("I3 边集不是 σ 的像 (图被改动)")
    if sig["edge_multiset"] != ref_sig["edge_multiset"] or sig["m"] != ref_sig["m"]:
        bad.append("I3 边权多重集/边数改变")
    if sig["weighted_degree_multiset"] != ref_sig["weighted_degree_multiset"]:
        bad.append("I3 加权度序列改变 (图不同构)")
    for w in sig["edge_multiset"]:
        if float(w) not in WIRE_COUNTS:
            bad.append(f"I4 非法 wireCount {w}")

    # I5 hubump 由 2*Σw 重导, 是 0.045 的倍数, body 边长 > 0
    idx = {nm: i for i, nm in enumerate(core.names)}
    s_sum = [0.0] * n
    for e in rec.get("connections", []):
        a, b = idx[e["node1"]], idx[e["node2"]]
        w = float(e["wireCount"])
        s_sum[a] += w
        s_sum[b] += w
    for i, c in enumerate(chiplets):
        hu = float(c["hubump"])
        want = adp.solve_hubump(fp[i][2], fp[i][3], 2.0 * s_sum[i])
        if want is None or abs(want - hu) > 1e-6:
            bad.append(f"I5 槽位{i} hubump {hu} != solve_hubump {want}")
        if abs(hu / UBUMP_PITCH - round(hu / UBUMP_PITCH)) > 1e-6:
            bad.append(f"I5 槽位{i} hubump 非 0.045 倍数")
        if f"width" in c and (float(c["width"]) <= 0 or float(c["height"]) <= 0):
            bad.append(f"I5 槽位{i} body 边长非正")

    # I6 footprint 无实质重叠 (贴合合法!)
    depth = fp_max_depth(fp)
    if depth > FP_EPS:
        bad.append(f"I6 footprint 实质重叠 {depth:.6f}mm")

    # I7 坐标吸附到 0.001
    for i, c in enumerate(chiplets):
        for k in ("x-position", "y-position", "width", "height"):
            v = float(c[k])
            if abs(v - snap001(v)) > 1e-9:
                bad.append(f"I7 槽位{i} {k}={v} 不在 0.001 栅格上")

    # I8 rotation ∈ {0,1}; width/height 是摆放后尺寸 = footprint - 2*hubump
    for i, c in enumerate(chiplets):
        if int(c["rotation"]) not in (0, 1):
            bad.append(f"I8 槽位{i} rotation={c['rotation']}")
        hu = float(c["hubump"])
        if abs(float(c["width"]) - (fp[i][2] - 2.0 * hu)) > 1e-6 or \
           abs(float(c["height"]) - (fp[i][3] - 2.0 * hu)) > 1e-6:
            bad.append(f"I8 槽位{i} width/height 与 footprint-2*hubump 不符")

    # I9 schema
    want_chip = {"name", "x-position", "y-position", "width", "height", "rotation", "power", "hubump"}
    for i, c in enumerate(chiplets):
        if set(c) != want_chip:
            bad.append(f"I9 槽位{i} chiplet 字段 {sorted(set(c) ^ want_chip)}")
            break
    for e in rec.get("connections", []):
        if set(e) != {"node1", "node2", "wireCount"}:
            bad.append(f"I9 connection 字段 {sorted(set(e))}")
            break
    if [c["name"] for c in chiplets] != core.names:
        bad.append("I9 chiplet 名字/顺序改变")

    return bad


# ======================================================================================
# 候选库
# ======================================================================================
@dataclass
class Candidate:
    record: dict
    tag: str
    sigma: Optional[List[int]] = None
    objective: Optional[Tuple[float, float, float]] = None   # (max_c, wl, bbox_area), 越小越好


def gen_c0(core: SystemCore) -> Optional[Candidate]:
    """L0 参考记录本身 (σ=恒等, 功耗不动)。它是整个方案的单调性锚点。"""
    rec = make_record(core, sigma=list(range(core.n)), power_slots=list(core.powers),
                      sid=core.sid)
    if rec is None:
        return None
    return Candidate(record=rec, tag="C0", sigma=list(range(core.n)))


def gen_qap(core: SystemCore, rng: random.Random, restarts: int = 4,
            tries: int = 8) -> List[Candidate]:
    """G1: 重解 QAP 得 σ。多次独立重启 -> 多个互不相同的强局部最优 (结构多样性)。"""
    n = core.n
    if n < 2:
        return []
    demand, maxcap, dist = core.demand(), core.maxcap(), core.dist()
    seen, out = set(), []
    for _ in range(tries):
        a = adp.assign_graph(dist, core.edges, n, demand, maxcap, rng, restarts)
        if a is None:
            continue
        key = tuple(a)
        if key in seen or key == tuple(range(n)):
            continue
        seen.add(key)
        rec = make_record(core, sigma=a, sid=core.sid)
        if rec is None:
            continue
        out.append(Candidate(record=rec, tag=f"G1:qap{len(out)}", sigma=list(a)))
    return out


def _power_orders(core: SystemCore) -> Dict[str, List[int]]:
    """把若干"排布策略"表达成 槽位->功耗值 的列表。全部保持功耗多重集不变。"""
    n = core.n
    centers = [(f[0] + f[2] / 2.0, f[1] + f[3] / 2.0) for f in core.fp]
    areas = [f[2] * f[3] for f in core.fp]
    order = adp.peripheral_scores(centers)          # 越大越靠外
    values = sorted(core.powers, reverse=True)
    s_sum = [0.0] * n
    for u, v, w in core.edges:
        s_sum[u] += w
        s_sum[v] += w

    def by_key(key):
        rank = sorted(range(n), key=lambda i: -key[i])
        out = [0.0] * n
        for r, i in enumerate(rank):
            out[i] = values[r]
        return out

    return {
        "density": by_key([areas[i] * order[i] for i in range(n)]),   # adp 默认
        "power": by_key(order),
        "rev_density": by_key([-areas[i] * order[i] for i in range(n)]),
        "connectivity": by_key(s_sum),                 # 连得最多的吃最大功耗
        "connectivity_rev": by_key([-v for v in s_sum]),
        "area_asc": by_key([-areas[i] for i in range(n)]),
    }


def gen_power(core: SystemCore, rng: random.Random, n_random: int = 26) -> List[Candidate]:
    """G2: 功耗置换。多重集冻结, 只改"哪颗芯粒吃多少功耗"。"""
    n = core.n
    if n < 2:
        return []
    out, seen = [], set()
    for tag, pw in _power_orders(core).items():
        key = tuple(pw)
        if key in seen:
            continue
        seen.add(key)
        rec = make_record(core, power_slots=pw, sid=core.sid)
        if rec is not None:
            out.append(Candidate(record=rec, tag=f"G2:{tag}", sigma=list(range(n))))
    for _ in range(n_random):
        pw = list(core.powers)
        rng.shuffle(pw)
        key = tuple(pw)
        if key in seen:
            continue
        seen.add(key)
        rec = make_record(core, power_slots=pw, sid=core.sid)
        if rec is not None:
            out.append(Candidate(record=rec, tag="G2:rand", sigma=list(range(n))))
    return out


def gen_geom(core: SystemCore) -> List[Candidate]:
    """G3: bbox 等距变换。footprint 多重集、bbox 面积、画布边长都不变。"""
    out = []
    for tag, fn in (("mirror_x", mirror_x), ("mirror_y", mirror_y), ("transpose", transpose)):
        fp = fn(core.fp)
        rot = core.rotation if tag != "transpose" else [1 - r for r in core.rotation]
        rec = make_record(core, fp=fp, rotation=rot, sid=core.sid)
        if rec is not None:
            out.append(Candidate(record=rec, tag=f"G3:{tag}", sigma=list(range(core.n))))
    return out


def build_bank(core: SystemCore, rng: random.Random, qap_tries: int = 32,
               qap_restarts: int = 8, n_power_random: int = 26,
               with_geom: bool = True) -> List[Candidate]:
    """实测 (chunk5, 200 systems): tries=32/restarts=8 时每 system 约 7 个互不相同的 QAP 解,
    耗时 0.07 s/system。tries=8/restarts=4 只有 1.75 个。"""
    bank: List[Candidate] = []
    c0 = gen_c0(core)
    if c0 is not None:
        bank.append(c0)
    bank += gen_qap(core, rng, restarts=qap_restarts, tries=qap_tries)
    bank += gen_power(core, rng, n_random=n_power_random)
    if with_geom:
        bank += gen_geom(core)
    return bank


def with_power(rec: dict, powers: Sequence[float]) -> dict:
    """复制一条记录, 只替换功耗向量 (其余逐字节不动)。"""
    out = dict(rec)
    out["chiplets"] = [dict(c) for c in rec["chiplets"]]
    for c, p in zip(out["chiplets"], powers):
        c["power"] = float(p)
    return out


# ======================================================================================
# 帕累托
# ======================================================================================
# 目标值必须先量化再比支配关系, 否则浮点噪声会造出大量假"互相不支配"的点。实测:
# G3 的 mirror_x 与 C0 的 bbox 面积只差 4.5e-13 (镜像本就保面积), 用精确比较会认为
# "C0 的面积更大", 于是 C0 支配不了它, 前沿里就混进一个跟 C0 物理上等价的点。
# 量化分辨率按实测噪声底取: 热代理 ~1e-3°C, 线长代理 ~1e-6 相对, 面积是解析量。
MAXC_QUANTUM = 1e-2      # °C  (实测批/逐条与 run-to-run 噪声 ~1e-3, 取 10 倍)
WL_REL_QUANTUM = 1e-5    # 相对 (同上, 取 10 倍)
AREA_QUANTUM = 1e-6      # mm²


def obj_quantum(objectives: Sequence[Tuple[float, float, float]]):
    vals = [o for o in objectives if o is not None]
    scale = max((abs(o[1]) for o in vals), default=1.0)
    return (MAXC_QUANTUM, max(WL_REL_QUANTUM * scale, 1e-9), AREA_QUANTUM)


def quantize(obj, q):
    return tuple(round(obj[k] / q[k]) for k in range(3))


def dominates(oo, cc, q) -> bool:
    """量化后 o 是否严格支配 c (三目标全部最小化)。"""
    qo, qc = quantize(oo, q), quantize(cc, q)
    return all(qo[k] <= qc[k] for k in range(3)) and any(qo[k] < qc[k] for k in range(3))


def pareto_front(items: Sequence[Candidate], q=None) -> List[Candidate]:
    """三目标 (max_c, wl, bbox_area) 全部最小化的非支配集。先量化再比, 见上。"""
    for c in items:
        if c.objective is None:
            raise ValueError(f"{c.tag} 没有 objective")
    if q is None:
        q = obj_quantum([c.objective for c in items])
    front: List[Candidate] = []
    for c in items:
        if not any(dominates(o.objective, c.objective, q) for o in items if o is not c):
            front.append(c)
    # 量化后目标相同的只留一个 (浮点噪声造出的重复点)
    uniq, seen = [], set()
    for c in front:
        key = quantize(c.objective, q)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def normalize_objectives(items: Sequence[Candidate]) -> List[Tuple[float, float, float]]:
    dims = list(zip(*(c.objective for c in items)))
    lo = [min(d) for d in dims]
    hi = [max(d) for d in dims]
    return [tuple((c.objective[k] - lo[k]) / (hi[k] - lo[k]) if hi[k] > lo[k] else 0.0
                  for k in range(3)) for c in items]


def select_knee(front: Sequence[Candidate]) -> Candidate:
    """归一化后离理想点最近的那个前沿成员 (L2)。前沿只有 1 个时就是它。"""
    if len(front) == 1:
        return front[0]
    norm = normalize_objectives(front)
    best, best_d = front[0], float("inf")
    for c, v in zip(front, norm):
        d = sum(x * x for x in v)
        if d < best_d:
            best, best_d = c, d
    return best


# ======================================================================================
# 打分器 (需要 torch)
# ======================================================================================
class SurrogateScorer:
    """两个冻结代理。逐条口径与 proxy_eval.ProxyEvaluator 逐位一致, 另加"同 system 候选批量"。

    批量之所以成立: 同一 system 的所有候选 **共享 footprint 铺排**, 于是共享画布
    (canvas_side 只依赖 footprint)。逐图变化的只有位置、body 尺寸、功耗、hubump —— 后三者
    通过 `per_graph_*` 通道传入 (见 train_graph_thermal._thermal_gnn_hrnet_inputs)。
    """

    def __init__(self, device: str = "cuda", thermal_batch: int = 32):
        import torch  # noqa: F401  (延迟导入, 便于纯 CPU 的构造阶段不拉 torch)
        import proxy_eval as PE

        self._PE = PE
        self._torch = torch
        self.ev = PE.ProxyEvaluator(device=device)
        self.device = self.ev.device
        self.thermal_batch = int(thermal_batch)
        self._T = None

    @property
    def T(self):
        if self._T is None:
            import train_graph_thermal as T
            self._T = T
        return self._T

    # ---------------------------------------------------------------- 逐条 (基准口径)
    def wl(self, records: Sequence[dict]) -> List[float]:
        return self.ev.wirelength(list(records))

    def thermal_one(self, records: Sequence[dict]) -> List[dict]:
        return self.ev.thermal(list(records))

    # ---------------------------------------------------------------- 同 system 批量
    def thermal_batched(self, records: Sequence[dict]) -> List[dict]:
        """批量热前向。画布/节点数不一致时自动退回逐条 (并计入 self.fallback)。"""
        torch = self._torch
        PE, T = self._PE, self.T
        out: List[Optional[dict]] = [None] * len(records)
        self.fallback = getattr(self, "fallback", 0)

        groups: Dict[Tuple[int, int], List[int]] = {}
        conds: List = [None] * len(records)
        xs: List = [None] * len(records)
        for i, r in enumerate(records):
            x, c = PE.ProxyEvaluator._to_cond(r)
            xs[i], conds[i] = x, c
            groups.setdefault((int(x.shape[0]), round(float(c.chip_size[2]), 9)), []).append(i)

        for (_n, _side), idxs in groups.items():
            cond0 = conds[idxs[0]]
            for start in range(0, len(idxs), self.thermal_batch):
                chunk = idxs[start:start + self.thermal_batch]
                X = torch.stack([xs[i] for i in chunk]).to(self.device)
                pp = torch.stack([conds[i].node_power for i in chunk])
                ps = torch.stack([conds[i].tap_source_chiplet_sizes for i in chunk])
                ph = torch.stack([conds[i].tap_hubump for i in chunk])
                with torch.no_grad():
                    raw = T._thermal_forward(self.ev.th_model, X, cond0, grid_size=self.ev.thermal_grid,
                                             rect_sharpness=80.0, stats=self.ev.th_stats,
                                             differentiable=False, per_graph_power=pp,
                                             per_graph_sizes=ps, per_graph_hubump=ph)
                    grid, _avg = T._thermal_output_to_grid_and_avg(raw)
                    gc = T._denorm_temp_k(grid, self.ev.th_stats) - 273.15
                for k, i in enumerate(chunk):
                    flat = gc[k].flatten()
                    out[i] = {
                        "max_c": float(flat.max()),
                        "mean_c": float(flat.mean()),
                        "p99_c": float(flat.kthvalue(max(1, int(0.99 * flat.numel()))).values),
                    }
        return [dict(o) for o in out]      # type: ignore[arg-type]

    # ---------------------------------------------------------------- 组合
    def score_bank(self, records: Sequence[dict],
                   canonical_powers: Optional[Sequence[Sequence[float]]] = None) -> List[dict]:
        """返回 [{max_c, mean_c, p99_c, wl, wl_native, bbox_area, canvas_side, total_power}, ...]。

        `wl` 是 **排序口径**: 统一用 `canonical_powers[i]` (通常是 C0 的功耗向量) 求值。原因实测:
        `WirelengthGNN` 对功耗有约 0.15% 中位 / 1.08% 最大的依赖, 而真值 CPLEX 布线线长
        (min Σd·f) **对功耗完全不变** —— 这个依赖纯粹是训练副产物, 于是"置换功耗"会成为
        线长轴上一条无物理意义的作弊通道。固定功耗求值把它关掉。`wl_native` 保留原功耗下的
        读数, 仅供诊断。
        """
        wl_native = self.wl(records)
        if canonical_powers is None:
            wl_rank = wl_native
        else:
            wl_rank = self.wl([with_power(r, p) for r, p in zip(records, canonical_powers)])
        th = self.thermal_batched(records)
        out = []
        for i, r in enumerate(records):
            fp = [footprint_of(c) for c in r["chiplets"]]
            out.append(dict(th[i], wl=wl_rank[i], wl_native=wl_native[i],
                            bbox_area=bbox_area(fp), canvas_side=canvas_side(fp),
                            total_power=sum(float(c["power"]) for c in r["chiplets"])))
        return out


def objectives_of(scores: Sequence[dict]) -> List[Tuple[float, float, float]]:
    return [(s["max_c"], s["wl"], s["bbox_area"]) for s in scores]


# ======================================================================================
# 数据集 I/O
# ======================================================================================
def load_chunk(path: Path) -> Dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sorted_sids(records: Dict[str, dict]) -> List[str]:
    return sorted(records, key=lambda s: int(s.split("_")[1]))


def write_json_atomic(path: Path, payload, indent: Optional[int] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(path)
