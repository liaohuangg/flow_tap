#!/usr/bin/env python3
"""greedy_init.py — 确定性的构造解, 给 MILP 当 warm start 和 λ 初值。

不追求好, 只追求"可行且连通性有序": 按连通性 BFS 序做货架式排布, 于是有连接的
chiplet 在货架上彼此相邻, 线长起点不会太离谱。MILP 从这里出发找 incumbent 明显更快。

不依赖任何随机数 —— 同一个 case 永远给同一个解。
"""
from __future__ import annotations

from collections import deque


def bfs_order(A: dict) -> list[int]:
    """从度数最高的 chiplet 出发做 BFS, 度数相同时取下标小的 (确定性)。"""
    n = A["n"]
    if n == 0:
        return []
    start = max(range(n), key=lambda i: (A["degree"][i], -i))
    seen = [False] * n
    order = []
    queue = deque([start])
    seen[start] = True
    while queue:
        i = queue.popleft()
        order.append(i)
        # 邻居按连接强度降序、下标升序 —— 保证确定性
        nbrs = sorted((j for j in range(n) if not seen[j] and A["mat"][i][j] > 0),
                      key=lambda j: (-A["mat"][i][j], j))
        for j in nbrs:
            seen[j] = True
            queue.append(j)
    # 无连接的孤立 chiplet 排在最后
    order.extend(i for i in range(n) if not seen[i])
    return order


def shelf_layout(A: dict, S: float, allow_rot: bool = False) -> list[tuple[float, float, int]]:
    """按 BFS 序从左下角开始逐行摆放 footprint。返回 [(FX, FY, r), ...]。"""
    n = A["n"]
    order = bfs_order(A)
    out: list[tuple[float, float, int]] = [(0.0, 0.0, 0) for _ in range(n)]

    r_of = [0] * n
    if allow_rot:
        # 不旋转就是最朴素的起点; 允许旋转时把"高的"竖起来, 让货架更矮
        for i in range(n):
            r_of[i] = 1 if A["h"][i] > A["w"][i] else 0

    cursor_x = 0.0
    cursor_y = 0.0
    row_h = 0.0
    for i in order:
        r = r_of[i]
        fw = A["w"][i] + 2 * A["u"][i] + r * (A["h"][i] - A["w"][i])
        fh = A["h"][i] + 2 * A["u"][i] + r * (A["w"][i] - A["h"][i])
        # 放不下就换行 (画布是软约束, 这里只是让起点好看一点)
        if cursor_x > 0.0 and cursor_x + fw > S:
            cursor_x = 0.0
            cursor_y += row_h
            row_h = 0.0
        out[i] = (cursor_x, cursor_y, r)
        cursor_x += fw
        row_h = max(row_h, fh)
    return out


if __name__ == "__main__":
    import argparse

    import ilp_core as C

    ap = argparse.ArgumentParser(description="greedy 构造解")
    ap.add_argument("case")
    args = ap.parse_args()

    record = C.load_case(args.case)
    A = C.case_arrays(record)
    S, src = C.canvas_mm(args.case, record["chiplets"])
    x = shelf_layout(A, S)
    print(f"{args.case}: n={A['n']} S={S:.3f}mm ({src})")
    print(f"  BFS 序: {[A['names'][i] for i in bfs_order(A)][:10]} ...")
    print(f"  重叠对: {len(C.footprints_overlap(A, x))}")
