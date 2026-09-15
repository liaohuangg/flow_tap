"""热点定位: 热代理的温度图 -> 需要冻结哪几个 chiplet。

口径 (用户指定)
---------------
    热点 = 温度图里最热的 **top 5%** 格子 (64x64 = 4096 格 -> 约 205 格)
    冻结 = footprint 落在这些格子上的所有 chiplet

为什么用一块区域而不是最热的那一格
----------------------------------
单格 argmax 完全被 64x64 光栅化的格子边界效应支配: 实测同一布局把某个 chiplet
挪 0.05mm, 峰值读数就跳 +0.14K —— 那 0.14K 不是物理, 是"最热的格子换了一个"。
按单格定热点, 冻结对象会在两个 chiplet 之间反复横跳。取 top 5% 得到一块**区域**,
格子级噪声被平均掉, 压着这块区域的那几个 chiplet 才是真正决定峰值的。

栅格口径为什么直接调 `_thermal_rect_field`
-------------------------------------------
归属必须和**模型自己**的栅格口径逐位一致, 否则热点会整体平移一格。所以这里不
重写栅格化, 直接借 `train_graph_thermal._thermal_rect_field` —— 那正是
`_thermal_gnn_hrnet_inputs` 构造 layout/power 通道时用的同一个函数:
方格 (i,j) 覆盖归一化 x∈[j/G,(j+1)/G), y∈[i/G,(i+1)/G), 展平索引 = i*G + j
(行主序, 行=y), 与 `temperature_c` 里 `temp.reshape(C, -1)` 的展平方式一致。

自检
----
热点格子**未必**全被 chiplet 盖住 —— 芯片之间的空隙、hubump halo 也会出现在图上。
所以返回值里带 `coverage` (被任一 chiplet 覆盖的热点格子占比); coverage 明显低于
1 说明栅格口径对不上, 那是个必须当场看见的错, 不能静默接受。
"""
from __future__ import annotations

import numpy as np

from _bootstrap import setup as _setup

_setup()

__all__ = ["locate_hotspot"]


def _footprint_cells(lower, sizes, origin, side, grid_size):
    """(V, G*G) 的 0/1 footprint 栅格, 口径与模型输入完全一致。

    ``_thermal_rect_field`` 的入参必须是 **二维 (B, V)** —— 它按
    ``left.unsqueeze(-1).unsqueeze(-1)`` 广播, 传一维 (V,) 会多出一个前导 batch 维,
    得到 (1, V, G, G) 而不是 (B, V, G, G)。所以这里显式给一个长度为 1 的 batch 维,
    再把 ``body[0]`` 取出来。
    """
    import torch
    from train_graph_thermal import _thermal_rect_field

    lower = np.asarray(lower, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)

    left = torch.as_tensor((lower[:, 0] - origin[0]) / float(side), dtype=torch.float64).view(1, -1)
    bottom = torch.as_tensor((lower[:, 1] - origin[1]) / float(side), dtype=torch.float64).view(1, -1)
    right = left + torch.as_tensor(sizes[:, 0] / float(side), dtype=torch.float64).view(1, -1)
    top = bottom + torch.as_tensor(sizes[:, 1] / float(side), dtype=torch.float64).view(1, -1)
    # differentiable=False 是硬栅格 (判交式), sharpness 在该分支根本不参与计算,
    # 所以这里传 0.0 即可 —— 传别的值不会有任何区别。
    body = _thermal_rect_field(left, bottom, right, top, int(grid_size), 0.0, False)
    return body[0].numpy().reshape(len(lower), -1) > 0.5  # (V, G*G) bool


def locate_hotspot(
    temp_flat,
    lower,
    sizes,
    origin,
    side,
    *,
    names=None,
    top_frac=0.05,
    min_cells=1,
    min_movable=2,
):
    """温度图 (G*G,) -> 热点信息 dict, 其中 ``indices`` 是要冻结的 chiplet 下标。

    ``min_cells``  一个 chiplet 至少要盖住几个热点格子才算"压着热点"。默认 1
                   (用户口径: 落在热点格子上的就冻)。调大可以只冻最核心的那一两个。
    ``min_movable`` 兜底: 冻结后至少留这么多 chiplet 可动, 否则 Stage B 无事可做。
                   盖住热点格子最少的会被逐个解冻, 直到满足。
    """
    temp_flat = np.asarray(temp_flat, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    V = int(lower.shape[0])

    total = int(temp_flat.size)
    grid = int(round(float(np.sqrt(total))))
    if grid * grid != total:
        raise ValueError(f"温度图展平长度 {total} 不是完全平方数, 无法还原成 GxG 网格")

    n_hot = max(1, int(round(total * float(top_frac))))
    order = np.argsort(temp_flat)[::-1][:n_hot]
    hot = np.zeros(total, dtype=bool)
    hot[order] = True

    cells = _footprint_cells(lower, sizes, origin, side, grid)
    per_chiplet = cells[:, hot].sum(axis=1)          # (V,) 各 chiplet 盖住的热点格子数
    covered = int((cells[:, hot].any(axis=0)).sum())
    coverage = float(covered) / float(n_hot)

    picked = [i for i in np.argsort(per_chiplet)[::-1] if per_chiplet[i] >= int(min_cells)]
    # 兜底: 别把可动件冻光。按覆盖数从小到大解冻。
    floor = max(1, int(min_movable))
    while picked and V - len(picked) < floor:
        picked.pop()

    info = {
        "grid_size": grid,
        "hot_cells": n_hot,
        "hot_frac": float(top_frac),
        "threshold_C": float(temp_flat[order].min()),
        "coverage": coverage,
        "hot_cell_index": order.astype(np.int64),
        "indices": [int(i) for i in picked],
        "names": [str(names[i]) if names is not None else str(i) for i in picked],
        "per_chiplet_cells": {
            (str(names[i]) if names is not None else str(i)): int(per_chiplet[i]) for i in range(V)
        },
        "max_cells": int(per_chiplet.max()) if V else 0,
    }
    return info
