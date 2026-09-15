"""placement JSON -> PyG ``cond``, 即两个神经评估器的入参格式。

为什么要这一层
--------------
``WirelengthSurrogate.predict(placement, cond)`` 和
``_thermal_forward(model, x_hat, cond, ...)`` 都不吃 mm 左下角坐标, 它们要的是
**归一化中心坐标 (B,V,2) in [-1,1]** + ``cond`` (装着尺寸/网表/功耗/画布)。
本模块把 placement JSON 翻译成这套约定, 一个适配器同时喂两个模型。

字段口径照抄 ``json_benchmark_dataset.JsonBenchmarkDataset.__getitem__`` (:88) ——
那个类干的是同一件事, 但它会丢掉输入 JSON 里的 x/y 改做 shelf-packing, 所以不能
直接复用, 只能照它的字段构造方式自己写。

两个模型实际读到的 ``cond`` 字段 (已逐个核对实现):
  热  : chip_size (mm 画布) / tap_source_chiplet_sizes (mm) / node_power (W)
        / is_macros / x / tap_hubump (缺省则从连线需求推导)
  线长: edge_index / edge_weight (raw wireCount) / x / chip_size
        / node_power / tap_hubump (缺省则自行推导)
"""
from __future__ import annotations

import numpy as np

from _bootstrap import setup as _setup

_setup()

__all__ = ["build_cond"]

# 与 json_benchmark_dataset.py 保持一致的网表边特征布局:
#   [:, 0:2] 端点 u 的 pin 偏移, [:, 2:4] 端点 v 的 pin 偏移, [:, 4:] 连接元数据
_EDGE_META_DIM = 4


def build_cond(record: dict, layout, origin, side: float, hubump_source: str = "derived"):
    """构造 PyG ``Data``。所有张量留在 CPU, 由调用方按需 .to(device)。

    ``hubump_source``:
      * ``derived`` (默认) —— 不写 ``tap_hubump``, 让两个模型各自从连线需求推导。
        placement JSON 里没有 hubump 字段, 这是最省事且无需 join benchmark 的选择。
      * ``zero`` —— 显式置 0。
    两种选择在同一个 case 的前后对比中是自洽的, 因此不影响 ΔT / ΔWL 判据。
    """
    import torch
    from torch_geometric.data import Data

    sizes = np.asarray(layout.sizes, dtype=np.float64)
    V = int(sizes.shape[0])
    side = float(side)
    origin = np.asarray(origin, dtype=np.float64)

    size_norm = 2.0 * sizes / side
    cond = Data(
        x=torch.tensor(size_norm, dtype=torch.float32),
        is_ports=torch.zeros(V, dtype=torch.bool),
        is_macros=torch.ones(V, dtype=torch.bool),
        node_power=torch.tensor(np.asarray(layout.power, dtype=np.float64), dtype=torch.float32),
        chip_size=torch.tensor(
            [origin[0], origin[1], origin[0] + side, origin[1] + side], dtype=torch.float32
        ),
        tap_source_chiplet_sizes=torch.tensor(sizes, dtype=torch.float32),
    )

    index_by_name = {name: i for i, name in enumerate(layout.names)}
    pairs, attrs, weights = [], [], []
    for conn in record.get("connections", []) or []:
        node1 = str(conn.get("node1", ""))
        node2 = str(conn.get("node2", ""))
        if node1 not in index_by_name or node2 not in index_by_name:
            continue
        source = index_by_name[node1]
        target = index_by_name[node2]
        if source == target:
            continue
        wire_count = float(conn.get("wireCount", conn.get("edge_weight", 1.0)) or 1.0)
        bump_width = float(conn.get("EMIB_bump_width", 0.5) or 0.5)
        metadata = torch.tensor([wire_count / 1024.0, bump_width, 0.0, 1.0], dtype=torch.float32)
        attr = torch.cat((torch.zeros(_EDGE_META_DIM, dtype=torch.float32), metadata))
        pairs.extend(((source, target), (target, source)))
        attrs.extend((attr.clone(), attr.clone()))
        weights.extend((wire_count, wire_count))

    if pairs:
        cond.edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        cond.edge_attr = torch.stack(attrs)
        cond.edge_weight = torch.tensor(weights, dtype=torch.float32)
        # pin 偏移口径同 json_benchmark_dataset: 归一化半尺寸的负值
        cond.edge_attr[:, :2] = -cond.x[cond.edge_index[0]] / 2.0
        cond.edge_attr[:, 2:4] = -cond.x[cond.edge_index[1]] / 2.0
    else:
        cond.edge_index = torch.empty((2, 0), dtype=torch.long)
        cond.edge_attr = torch.empty((0, 2 * _EDGE_META_DIM), dtype=torch.float32)
        cond.edge_weight = torch.empty((0,), dtype=torch.float32)

    if str(hubump_source).lower() == "zero":
        cond.tap_hubump = torch.zeros(V, dtype=torch.float32)

    return cond
