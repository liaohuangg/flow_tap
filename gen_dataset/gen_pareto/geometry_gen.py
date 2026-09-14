#!/usr/bin/env python3
"""geometry_gen.py — 固定条件 (图 + 功耗) 下生成多样化布局的几何生成器。

核心结论 (本机实测)
--------------------
热轴的真实提升来自「把布局铺开 → 散热片变大 → 对流热阻下降」的物理 (面积换温度), 而非
「热芯粒紧凑排边缘」的排序 (后者只值 ~-1°C, 计划 S4 已证)。

而**力导向放置 (FD)** 从随机初始出发有两宗罪:
  1. 没有全局压实 —— 随机撒开 + 成对引力只把互联组拉成团, 团之间没有力拉近, 于是即使
     α=0 (纯引力) 也塌缩成 ~200mm 的松散大铺 (比 C0 的 115mm 还大);
  2. 随机打乱 C0 的好线长排布 —— 实测同一温度下 FD 的线长是径向铺开的 2~4 倍, 全部被支配。

所以这里用**径向铺开 C0** 作为唯一几何杠杆:
  · 保留 C0 的好线长排布 (Rule 1: 高线数互联相邻, 原样不动)
  · 绕质心等比放大 footprint 位置、尺寸冻结, 单调扫出「紧凑 → 铺开」的面积-热前沿
  · 成本几乎为零 (纯缩放, 不 resolve), 且天然合法 (s>1 放大所有间距, 不产生重叠)

工作对象是 footprint [(x, y, fw, fh)]: fw/fh 冻结 (不变量 I1), 只动 x/y。
"""
from __future__ import annotations

import math
from typing import List, Tuple

Layout = List[List[float]]   # [[x, y, fw, fh], ...]

# 径向铺开的缩放因子扫描。s=1.0 就是 C0 本身; s>1 越大越铺开、越凉、线长越差。
# 实测 s=1.6 线长劣化 +88~100% (不可接受), 所以这里只扫到 1.3 (~+40% 线长, 换 ~-8°C):
# 线长与铺开近似线性, s 上限收在"线长劣化不过分"的区间内。
DEFAULT_SCALES = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3)


def radial_spread(core, s: float) -> Layout:
    """绕 footprint 质心把 footprint 位置等比放大 s 倍, 尺寸不变。

    s>1 是纯放大: 任意两 footprint 的间距都乘以 s, 故 C0 合法 ⇒ 放大后合法。
    """
    fp = [[f[0], f[1], f[2], f[3]] for f in core.fp]
    c = [(f[0] + f[2] / 2.0, f[1] + f[3] / 2.0) for f in fp]
    n = len(fp)
    cx = sum(p[0] for p in c) / n
    cy = sum(p[1] for p in c) / n
    for i, f in enumerate(fp):
        f[0] = cx + s * (c[i][0] - cx) - f[2] / 2.0
        f[1] = cy + s * (c[i][1] - cy) - f[3] / 2.0
    return fp


def bbox_side(fp: Layout) -> float:
    xs0 = min(f[0] for f in fp)
    xs1 = max(f[0] + f[2] for f in fp)
    ys0 = min(f[1] for f in fp)
    ys1 = max(f[1] + f[3] for f in fp)
    return max(xs1 - xs0, ys1 - ys0)


def spread_bank(core, scales=DEFAULT_SCALES, cap: float = 187.0) -> List[Tuple[float, Layout]]:
    """扫 s, 返回 (s, footprint) 列表, 只留 bbox 边长 ≤ cap 的 (压在外推区以内)。

    对 C0 画布已经 > cap 的 system, 只有 s=1.0 (C0 本身) 可能留下 —— 这类 system 的热分数
    本就低置信, 不再额外铺开。
    """
    out: List[Tuple[float, Layout]] = []
    for s in scales:
        fp = radial_spread(core, s)
        if bbox_side(fp) <= cap + 1e-6:
            out.append((s, fp))
    # 至少保证 C0 自己在 (即便它本身超 cap, 也不能把锚点丢掉)
    if not out:
        out.append((1.0, [[f[0], f[1], f[2], f[3]] for f in core.fp]))
    return out
