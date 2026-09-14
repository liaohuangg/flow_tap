#!/usr/bin/env python3
"""对 placement_dataset_tw 做后处理: 按布局的几何分布重排互连线与功耗, 并同步重算 hubump。

动机
----
`placement_dataset_tw` 里每个 system 的 *条件* 是 gen_cfg.py 随机生成的 (随机的图 +
随机的功耗 + 随机的尺寸), *布局* 是 gen_placement.py 用 greedy legalizer 放出来的。
图与布局之间没有任何关系, 于是 p(layout | graph, power) ≈ p(layout) —— 模型从这种
数据里学不到"条件 -> 布局"的映射。

本脚本 **不改变布局几何**, 只修正条件这一侧, 让 互联 / 功耗 / hubump 与已有布局自洽:

  1. 互联: 把抽象图 H 的节点与物理 chiplet 做一次双射 (QAP), 最小化
     Σ wireCount · 曼哈顿中心距离。拓扑、边权多重集、连通性全部保持, 只是重新贴标签 ——
     所以条件侧的图分布不变, 改变的只有"哪张图配哪个布局"。
  2. 功耗: 保持功耗多重集不变, 按"越靠外围功耗密度越高"重排 (可切换为按绝对功耗)。
  3. hubump: 按新的互连总量重算, 公式与 preprocess_bump_region.compute_hubump 逐位一致:
         hubump = min{ k*0.045 : bump_capacity(body_w, body_h, k*0.045) >= s },
         s = Σ_j (M[i][j] + M[j][i]) = 2 × 该 chiplet 相连的所有 wireCount 之和

坐标约定 (关键)
--------------
`placement_dataset_tw` 存的是 **body** 坐标 + hubump。footprint = body + 2*hubump, 而
greedy 布局是对 **footprint** 做的无重叠放置 (见 gen_placement.py 的约定)。因此重算
hubump 后 **保持 footprint 不动**, 只把 body 往里缩:

    footprint = (x - hu, y - hu, w + 2*hu, h + 2*hu)
    body_new  = footprint 四边各让出 hu_new

footprint 不变 -> 布局合法性 (footprint 互不重叠) 自动保持, 不依赖任何重放置。旋转
不变 (bump_capacity 对 w/h 对称)。

若某 chiplet 的新 hubump 会让 body 边长 <= 0, 该 system 视为不可行整体丢弃并记录,
与 preprocess_bump_region.py 的处理一致。

用法
----
  python after_dataset_process.py --chunk 1                    # 处理第 1 个 chunk
  python after_dataset_process.py --start 1 --end 5000         # 处理 system 区间
  python after_dataset_process.py --chunk 1 --limit 200        # 试跑
  python after_dataset_process.py --chunk 1 --dry-run          # 只统计不写盘
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
SRC_DIR = PROJECT / "Dataset" / "dataset" / "placement_dataset" / "placement_dataset_tw"
OUT_DIR = PROJECT / "Dataset" / "dataset" / "placement_dataset" / "placement_dataset_opt"

CHUNK = 5000
UBUMP_PITCH = 0.045  # 45um microbump 节距, mm
EPS = 1e-9


# --------------------------------------------------------------------------------------
# hubump: 与 gen_dataset/preprocess_bump_region.py 完全一致
# --------------------------------------------------------------------------------------
def bump_capacity(w_mm: float, h_mm: float, hubump: float) -> int:
    """四周环形 microbump 能容纳的连接数 (TAP-2.5D routing 的 pmax 判定)。"""
    nh = int(hubump / UBUMP_PITCH)
    if nh <= 0:
        return 0
    return (
        2 * nh * int((h_mm + hubump) / UBUMP_PITCH)
        + 2 * nh * int((w_mm + hubump) / UBUMP_PITCH)
    )


def solve_hubump(footprint_w: float, footprint_h: float, s: float):
    """求最小 hubump 使 body = footprint - 2*hubump 的环容量装得下 s。无解返回 None。"""
    if s <= 0:
        return 0.0
    k = 1
    while k <= 1000:
        hubump = UBUMP_PITCH * k
        die_w = footprint_w - 2.0 * hubump
        die_h = footprint_h - 2.0 * hubump
        if die_w <= 0.0 or die_h <= 0.0:
            return None
        if bump_capacity(die_w, die_h, hubump) >= s:
            return hubump
        k += 1
    return None


def max_capacity(footprint_w: float, footprint_h: float) -> float:
    """该 footprint 能支撑的最大 s (body 缩到 >0 为止)。用于可行性判定。"""
    best = 0.0
    k = 1
    while k <= 1000:
        hubump = UBUMP_PITCH * k
        die_w = footprint_w - 2.0 * hubump
        die_h = footprint_h - 2.0 * hubump
        if die_w <= 0.0 or die_h <= 0.0:
            break
        best = max(best, float(bump_capacity(die_w, die_h, hubump)))
        k += 1
    return best


# --------------------------------------------------------------------------------------
# QAP: 抽象图节点 <-> 物理 chiplet 的双射
# --------------------------------------------------------------------------------------
def _demands(edges, n):
    """每个图节点的 s = 2 × 相连 wireCount 之和。"""
    s = [0.0] * n
    for u, v, w in edges:
        s[u] += 2.0 * w
        s[v] += 2.0 * w
    return s


def _violation(a, demand, maxcap, n):
    """可行性违反量: Σ max(0, s_u - cap(a[u]))。0 表示全部可行。"""
    tot = 0.0
    for u in range(n):
        over = demand[u] - maxcap[a[u]]
        if over > 0.0:
            tot += over
    return tot


def _spatial_cost(a, edges, dist):
    return sum(w * dist[a[u]][a[v]] for u, v, w in edges)


def _swap_delta(a, u, v, node_edges, dist, demand, maxcap):
    """交换 a[u] <-> a[v] 的 (violation_delta, spatial_delta)。

    只有与 u 或 v 相邻的边会变; 边 (u,v) 自身的距离不变 (dist 对称)。
    """
    pu, pv = a[u], a[v]
    d_spatial = 0.0
    for x, w in node_edges[u]:
        if x == v:
            continue
        d_spatial += w * (dist[pv][a[x]] - dist[pu][a[x]])
    for x, w in node_edges[v]:
        if x == u:
            continue
        d_spatial += w * (dist[pu][a[x]] - dist[pv][a[x]])

    def pen(node, phys):
        return max(0.0, demand[node] - maxcap[phys])

    d_viol = (pen(v, pu) + pen(u, pv)) - (pen(u, pu) + pen(v, pv))
    return d_viol, d_spatial


def _better(nv, nc, cv, cc):
    """字典序比较 (可行性, 空间代价)。先保证可行, 再压空间代价。"""
    nf, cf = nv <= EPS, cv <= EPS
    if nf != cf:
        return nf
    if nf:
        return nc < cc - EPS
    return (nv, nc) < (cv, cc)


def assign_graph(dist, edges, n, demand, maxcap, rng: random.Random, restarts: int):
    """返回 a: a[u] = 物理 chiplet 索引 (u 为图节点)。不可行则返回 None。"""
    node_edges = [[] for _ in range(n)]
    for u, v, w in edges:
        node_edges[u].append((v, w))
        node_edges[v].append((u, w))

    # 可行性最优初值: 大需求配大容量 (重排不等式)。
    by_demand = sorted(range(n), key=lambda u: -demand[u])
    by_cap = sorted(range(n), key=lambda p: -maxcap[p])
    a0 = [0] * n
    for u, p in zip(by_demand, by_cap):
        a0[u] = p
    if _violation(a0, demand, maxcap, n) > EPS:
        return None  # 任何双射都不可行

    # 备选初值: 0=可行性最优匹配, 1=原始配置(不动), 其余=随机扰动。
    identity = list(range(n))
    inits = [a0]
    if _violation(identity, demand, maxcap, n) <= EPS:
        inits.append(identity)
    while len(inits) < max(2, restarts):
        inits.append(_perturb(a0 if len(inits) % 2 else identity, n, rng))

    best_a, best_v, best_c = None, float("inf"), float("inf")
    for a in inits:
        cv = _violation(a, demand, maxcap, n)
        cc = _spatial_cost(a, edges, dist)
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    dv, dc = _swap_delta(a, u, v, node_edges, dist, demand, maxcap)
                    if _better(cv + dv, cc + dc, cv, cc):
                        a[u], a[v] = a[v], a[u]
                        cv += dv
                        cc += dc
                        improved = True
        if _better(cv, cc, best_v, best_c):
            best_v, best_c, best_a = cv, cc, list(a)

    if best_a is None or best_v > EPS:
        return None
    return best_a


def _perturb(a0, n, rng, strength=3):
    a = list(a0)
    for _ in range(rng.randint(1, strength)):
        i, j = rng.randrange(n), rng.randrange(n)
        a[i], a[j] = a[j], a[i]
    return a


# --------------------------------------------------------------------------------------
# 功耗重排
# --------------------------------------------------------------------------------------
def peripheral_scores(centers):
    """每个 chiplet 的"外围程度": 到布局质心的曼哈顿距离。越大越靠外。"""
    n = len(centers)
    if n == 0:
        return []
    cx = sum(c[0] for c in centers) / n
    cy = sum(c[1] for c in centers) / n
    return [abs(c[0] - cx) + abs(c[1] - cy) for c in centers]


def assign_power(areas, centers, powers, mode):
    """保持功耗多重集不变, 把大功耗排到外围。返回按 chiplet 顺序的新功耗列表。

    power    : 功耗与外围得分同序 -> 最大化 Σ P·s (热量重心被推到外围, 默认)
    density  : 功耗×面积 与外围得分同序 -> 避免小芯片被塞大功耗而产生密度尖峰
    none     : 不动
    """
    n = len(powers)
    if n == 0 or mode == "none":
        return list(powers)
    score = peripheral_scores(centers)
    key = [areas[i] * score[i] for i in range(n)] if mode == "density" else list(score)
    order = sorted(range(n), key=lambda i: -key[i])          # 最外围在前
    values = sorted(powers, reverse=True)                    # 最大功耗在前
    out = [0.0] * n
    for rank, i in enumerate(order):
        out[i] = values[rank]
    return out


# --------------------------------------------------------------------------------------
# 诊断指标
# --------------------------------------------------------------------------------------
def hpwl(edges, centers):
    """Σ wireCount × 曼哈顿中心距离 (与 gen_legal_pla_greedy._wirelength_from_layout 一致)。"""
    return sum(
        w * (abs(centers[u][0] - centers[v][0]) + abs(centers[u][1] - centers[v][1]))
        for u, v, w in edges
    )


def thermal_spread(centers, densities):
    """热点互斥度: Σ_{i<j} d_i·d_j / (dist + 1)。越小越好 (热点越分散)。"""
    n = len(centers)
    tot = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
            tot += densities[i] * densities[j] / (d + 1.0)
    return tot


# --------------------------------------------------------------------------------------
# 单个 system
# --------------------------------------------------------------------------------------
def process_system(rec, args, rng: random.Random):
    chiplets = rec["chiplets"]
    n = len(chiplets)
    if n == 0:
        return dict(rec), None
    name_to_phys = {c["name"]: i for i, c in enumerate(chiplets)}

    # --- 解析 body -> footprint (中心不变) ---
    fp_x, fp_y, fp_w, fp_h = [], [], [], []
    centers = []
    for c in chiplets:
        hu = float(c["hubump"])
        w, h = float(c["width"]), float(c["height"])
        fp_x.append(float(c["x-position"]) - hu)
        fp_y.append(float(c["y-position"]) - hu)
        fp_w.append(w + 2.0 * hu)
        fp_h.append(h + 2.0 * hu)
        centers.append((float(c["x-position"]) + w / 2.0, float(c["y-position"]) + h / 2.0))

    # --- 抽象图 H ---
    edges = []  # (u, v, wireCount) 以物理索引表示, "before" 的原始配置
    keep_edges = []
    for e in rec.get("connections", []):
        a, b = name_to_phys.get(e["node1"]), name_to_phys.get(e["node2"])
        if a is None or b is None or a == b:
            continue
        edges.append((a, b, float(e["wireCount"])))
        keep_edges.append(dict(e))
    if not edges:
        return dict(rec), None

    maxcap = [max_capacity(fp_w[i], fp_h[i]) for i in range(n)]
    dist = [
        [abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1]) for j in range(n)]
        for i in range(n)
    ]

    wl_before = hpwl(edges, centers)

    # --- QAP: 重贴标签 ---
    if args.graph_mode == "permute":
        demand = _demands(edges, n)
        a = assign_graph(dist, edges, n, demand, maxcap, rng, args.restarts)
        if a is None:
            return None, {"dropped": "infeasible_hubump"}
    else:  # none: 保持原图
        a = list(range(n))

    # a[u] = 物理索引, 即 图节点 u 落到物理 chiplet a[u]。
    # 新边: 图节点 (u,v,w) -> 物理 (a[u], a[v], w)
    new_edges = sorted(
        (tuple(sorted((a[u], a[v]))) + (w,)) for u, v, w in edges
    )

    # --- 重算 hubump, 回写 body (footprint 不动) ---
    s_sum = [0.0] * n
    for u, v, w in new_edges:
        s_sum[u] += w
        s_sum[v] += w
    new_hu = []
    for i in range(n):
        hu = solve_hubump(fp_w[i], fp_h[i], 2.0 * s_sum[i])
        if hu is None:
            return None, {"dropped": "infeasible_hubump"}
        new_hu.append(hu)

    new_chiplets = []
    new_centers = []
    areas = []
    for i, c in enumerate(chiplets):
        x = fp_x[i] + new_hu[i]
        y = fp_y[i] + new_hu[i]
        w = fp_w[i] - 2.0 * new_hu[i]
        h = fp_h[i] - 2.0 * new_hu[i]
        new_chiplets.append({
            "name": c["name"],
            "x-position": round(x, 6),
            "y-position": round(y, 6),
            "width": round(w, 6),
            "height": round(h, 6),
            "rotation": c["rotation"],
            "power": float(c["power"]),
            "hubump": round(new_hu[i], 6),
        })
        new_centers.append((x + w / 2.0, y + h / 2.0))
        areas.append(w * h)

    # --- 功耗重排 (放在 hubump 之后, 用新的 body 面积) ---
    old_dens = [
        float(chiplets[i]["power"]) / max(float(chiplets[i]["width"]) * float(chiplets[i]["height"]), EPS)
        for i in range(n)
    ]
    powers_before = [float(c["power"]) for c in chiplets]
    new_powers = assign_power(areas, new_centers, powers_before, args.power_mode)
    for i, c in enumerate(new_chiplets):
        c["power"] = float(new_powers[i])

    new_dens = [new_powers[i] / max(areas[i], EPS) for i in range(n)]

    out = dict(rec)
    out["chiplets"] = new_chiplets
    out["connections"] = [
        {"node1": chiplets[u]["name"], "node2": chiplets[v]["name"], "wireCount": int(w)}
        for u, v, w in new_edges
    ]
    if args.annotate:
        out["postprocess"] = {
            "source": "placement_dataset_tw",
            "graph_mode": args.graph_mode,
            "power_mode": args.power_mode,
        }

    # 热重心: 功耗密度加权的"外围程度"均值, 越大表示热量越靠外 (越利于散热)
    def heat_centroid(dens, cs):
        tot = sum(dens)
        if tot <= EPS:
            return 0.0
        sc = peripheral_scores(cs)
        return sum(dens[i] * sc[i] for i in range(n)) / tot

    stats = {
        "n": n,
        "m": len(new_edges),
        "wl_before": wl_before,
        "wl_after": hpwl(new_edges, new_centers),
        "spread_before": thermal_spread(centers, old_dens),
        "spread_after": thermal_spread(new_centers, new_dens),
        "heat_before": heat_centroid(old_dens, centers),
        "heat_after": heat_centroid(new_dens, new_centers),
        "hu_changed": sum(1 for i in range(n) if abs(new_hu[i] - float(chiplets[i]["hubump"])) > 1e-9),
    }
    return out, stats


# --------------------------------------------------------------------------------------
# 批量
# --------------------------------------------------------------------------------------
def _iter_systems(records, limit):
    items = sorted(records.items(), key=lambda kv: int(kv[0].split("_")[1]))
    for sid, rec in items[:limit] if limit else items:
        yield sid, rec


def run(args) -> dict:
    src = Path(args.src_dir) / f"chiplet_dataset_{args.chunk}.json"
    if not src.exists():
        raise FileNotFoundError(f"未找到输入: {src}")
    records = json.loads(src.read_text(encoding="utf-8"))

    out_records = {}
    stats = defaultdict(float)
    dropped = 0
    t0 = time.time()
    n_done = 0
    for sid, rec in _iter_systems(records, args.limit):
        rng = random.Random(args.seed + int(sid.split("_")[1]))
        new_rec, st = process_system(rec, args, rng)
        n_done += 1
        if new_rec is None:
            if dropped < 20:
                print(f"[post] {sid}: 丢弃 ({st['dropped']})", flush=True)
            dropped += 1
            continue
        if st is None:
            stats["trivial"] += 1
            out_records[sid] = new_rec
            continue
        out_records[sid] = new_rec
        for k, v in st.items():
            stats[k] += v
        if n_done % 2000 == 0:
            print(f"[post] 进度 {n_done} ({time.time() - t0:.0f}s)", flush=True)

    summary = _summarize(stats, len(out_records), dropped, time.time() - t0)
    print(summary, flush=True)

    if not args.dry_run:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"chiplet_dataset_{args.chunk}.json"
        dst.write_text(json.dumps(out_records, ensure_ascii=False), encoding="utf-8")
        print(f"[post] 写出 {dst} ({len(out_records)} systems)", flush=True)
    return stats


def _summarize(stats, kept, dropped, secs) -> str:
    n = stats.get("n", 0)
    if n == 0:
        return f"[post] DONE: 保留 {kept}, 丢弃 {dropped}, {secs:.1f}s (无统计)"
    wb, wa = stats["wl_before"], stats["wl_after"]
    sb, sa = stats["spread_before"], stats["spread_after"]
    return (
        f"[post] DONE: 保留 {kept}, 丢弃 {dropped}, {secs:.1f}s\n"
        f"       [曼哈顿 HPWL]  Σ{wa:,.0f} / 原 Σ{wb:,.0f}  = {wa / wb:.3f}x  (降低 {100 * (1 - wa / wb):.1f}%)\n"
        f"       [热点互斥度]  {sa / sb:.3f}x  (越低越好)\n"
        f"       [热重心外移]  {stats['heat_before'] / n:.2f} -> {stats['heat_after'] / n:.2f}  (越大越靠外)\n"
        f"       hubump 改变的 chiplet: {stats['hu_changed']:.0f} / {n:.0f}\n"
        f"       注: 上面两项是手搓代理, 真实评估请跑 proxy_eval.py"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", type=int, default=1, help="chiplet_dataset_{k}.json 的 k")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 system (0=全部)")
    ap.add_argument("--src-dir", type=Path, default=SRC_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--graph-mode", choices=["permute", "none"], default="permute",
                    help="permute: QAP 重贴标签 (默认); none: 保持原图")
    ap.add_argument("--power-mode", choices=["density", "power", "none"], default="density",
                    help="density: 外围功耗密度高 (默认); power: 外围绝对功耗高; none: 不变")
    ap.add_argument("--restarts", type=int, default=4, help="QAP 随机重启次数")
    ap.add_argument("--annotate", action="store_true",
                    help="在记录里追加 postprocess 元数据 (默认不加, 保持与输入完全同 schema)")
    ap.add_argument("--seed", type=int, default=20240913)
    ap.add_argument("--dry-run", action="store_true", help="只统计, 不写盘")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
