"""独立的精确几何校验: 逐对判交 + 逐矩形判越界。

**刻意不共用优化器的代码**（不调 ``guidance`` / ``utils.check_legality_new``）。
如果校验和优化走同一条实现, 它只能证明"两者一致", 不能证明"结果正确";
这里用纯 numpy 重写一遍, 才构成真正的交叉验证。

容差 ``geom_tol_mm`` 必须非零: 合法化的作用正是把 chiplet 推到刚好贴边, 所以
浮点噪声导致的 -1e-7 级"负间距"在合法化之后是**常态而非异常**（thermal_refine.py
:52-63 记录了实测 3e-7 ~ 5e-7 的同类现象）。取 1e-6 mm 留了余量, 同时远小于任何
真实重叠。
"""
from __future__ import annotations

import numpy as np

__all__ = ["find_overlap_pairs", "find_out_of_canvas", "legal_ratio", "verify_layout"]

DEFAULT_GEOM_TOL_MM = 1.0e-6


def find_overlap_pairs(lower, sizes, geom_tol_mm: float = DEFAULT_GEOM_TOL_MM):
    """返回重叠对列表 [(i, j, overlap_x_mm, overlap_y_mm, overlap_area_mm2), ...]。

    两矩形相交当且仅当两个轴上的重叠量都 > 容差。
    """
    lower = np.asarray(lower, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    V = lower.shape[0]
    upper = lower + sizes
    pairs = []
    for i in range(V):
        for j in range(i + 1, V):
            dx = min(upper[i, 0], upper[j, 0]) - max(lower[i, 0], lower[j, 0])
            dy = min(upper[i, 1], upper[j, 1]) - max(lower[i, 1], lower[j, 1])
            if dx > geom_tol_mm and dy > geom_tol_mm:
                pairs.append((i, j, float(dx), float(dy), float(dx * dy)))
    return pairs


def find_out_of_canvas(lower, sizes, origin, side, geom_tol_mm: float = DEFAULT_GEOM_TOL_MM):
    """返回越界的 chiplet 索引列表 [(i, excess_mm), ...]。"""
    lower = np.asarray(lower, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    upper_limit = origin + float(side)
    out = []
    for i in range(lower.shape[0]):
        excess = max(
            origin[0] - lower[i, 0],
            origin[1] - lower[i, 1],
            (lower[i, 0] + sizes[i, 0]) - upper_limit[0],
            (lower[i, 1] + sizes[i, 1]) - upper_limit[1],
        )
        if excess > geom_tol_mm:
            out.append((i, float(excess)))
    return out


def legal_ratio(lower, sizes, origin, side, geom_tol_mm: float = DEFAULT_GEOM_TOL_MM) -> float:
    """合法率 = 落在画布内的矩形并集面积 / 矩形总面积。1.0 = 完全合法。

    与 ``utils.check_legality_new`` (utils.py:1638) 同口径 (union / sum, 裁剪到画布),
    但这里是独立实现: 用扫描线算并集面积, 不依赖 shapely。
    """
    lower = np.asarray(lower, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    total = float((sizes[:, 0] * sizes[:, 1]).sum())
    if total <= 0.0:
        return 1.0

    x_lo = np.maximum(lower[:, 0], origin[0])
    y_lo = np.maximum(lower[:, 1], origin[1])
    x_hi = np.minimum(lower[:, 0] + sizes[:, 0], origin[0] + float(side))
    y_hi = np.minimum(lower[:, 1] + sizes[:, 1], origin[1] + float(side))
    keep = (x_hi - x_lo > 0.0) & (y_hi - y_lo > 0.0)
    if not keep.any():
        return 0.0

    union = _union_area(x_lo[keep], y_lo[keep], x_hi[keep], y_hi[keep])
    return float(min(1.0, union / total))


def _union_area(x_lo, y_lo, x_hi, y_hi) -> float:
    """轴对齐矩形并集面积, 扫描线 + 区间合并, 纯 numpy。"""
    edges = np.unique(np.concatenate((x_lo, x_hi)))
    area = 0.0
    for k in range(len(edges) - 1):
        left, right = edges[k], edges[k + 1]
        if right <= left:
            continue
        width = right - left
        # 取覆盖这一段 x 区间的矩形, 合并它们的 y 区间
        active = (x_lo <= left) & (x_hi >= right)
        if not active.any():
            continue
        spans = np.stack((y_lo[active], y_hi[active]), axis=1)
        spans = spans[np.argsort(spans[:, 0])]
        covered = 0.0
        cur_lo, cur_hi = spans[0]
        for lo, hi in spans[1:]:
            if lo > cur_hi:
                covered += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
            else:
                cur_hi = max(cur_hi, hi)
        covered += cur_hi - cur_lo
        area += width * covered
    return float(area)


def verify_layout(lower, sizes, origin, side, geom_tol_mm: float = DEFAULT_GEOM_TOL_MM) -> dict:
    """一次拿到全部几何结论, 供报告和验收断言使用。"""
    overlaps = find_overlap_pairs(lower, sizes, geom_tol_mm)
    outside = find_out_of_canvas(lower, sizes, origin, side, geom_tol_mm)
    return {
        "overlap_pairs": len(overlaps),
        "overlap_area_mm2": float(sum(p[4] for p in overlaps)),
        "overlap_detail": [
            {"i": int(i), "j": int(j), "dx_mm": round(dx, 6), "dy_mm": round(dy, 6)}
            for i, j, dx, dy, _ in overlaps
        ],
        "out_of_canvas": len(outside),
        "out_of_canvas_detail": [
            {"i": int(i), "excess_mm": round(excess, 6)} for i, excess in outside
        ],
        "legal_ratio": legal_ratio(lower, sizes, origin, side, geom_tol_mm),
        "is_legal": not overlaps and not outside,
    }
