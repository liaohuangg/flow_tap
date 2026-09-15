"""placement JSON 的读写、画布推导与度量计算。

字段与度量口径刻意对齐 LayoutGenModel/diffusion/utils.py, 使本工具的输出与
下游 (resultEval/eval_layout.py, evaluation/wirelength_bbox.py) 完全兼容:

  * ``_placement_bbox_stats``   utils.py:2079   area = bbox 宽*高, aspect_ratio = min/max
  * ``_placement_wirelength``   utils.py:2092   wireCount 加权的中心曼哈顿距离和
  * ``save_placement_json``     utils.py:2143   输出 schema
  * ``json_benchmark_dataset``  :88             画布/chip_size 的归一化口径

坐标系约定 (全库统一, 不可改):
  mm 左下角 (lower)  <--->  归一化中心坐标 ([-1,1] 画布)
      center_norm = 2 * (center_mm - origin) / side - 1
      size_norm   = 2 * size_mm / side
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Layout",
    "load_placement",
    "compute_canvas",
    "to_norm_centers",
    "from_norm_centers",
    "bbox_mm",
    "bbox_area_mm2",
    "hpwl_weighted",
    "write_placement",
    "make_report_record",
]


@dataclass
class Layout:
    """一个布局: 名称 + 左下角坐标 + 尺寸 + 功耗, 全部 mm / W。"""

    names: list
    lower: np.ndarray  # (V, 2) float64 mm 左下角
    sizes: np.ndarray  # (V, 2) float64 mm
    power: np.ndarray  # (V,)   float64 W

    @property
    def V(self) -> int:
        return int(self.lower.shape[0])

    def centers(self) -> np.ndarray:
        return self.lower + self.sizes / 2.0

    def copy(self) -> "Layout":
        return Layout(list(self.names), self.lower.copy(), self.sizes.copy(), self.power.copy())


def load_placement(path) -> tuple:
    """读 placement JSON, 返回 (原始 record, Layout)。

    保留原始 record 是为了输出时把 ``connections`` / ``rotation`` / ``EMIB*`` 等
    字段原样透传, 不去重建它们。
    """
    import json

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)

    chiplets = record["chiplets"]
    if not chiplets:
        raise ValueError(f"{path}: chiplets 为空")

    names = [str(ch.get("name", i)) for i, ch in enumerate(chiplets)]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: chiplet 名称有重复, 无法建立网表索引")

    lower = np.array(
        [[float(ch["x-position"]), float(ch["y-position"])] for ch in chiplets], dtype=np.float64
    )
    sizes = np.array(
        [[float(ch["width"]), float(ch["height"])] for ch in chiplets], dtype=np.float64
    )
    power = np.array([float(ch.get("power", 0.0) or 0.0) for ch in chiplets], dtype=np.float64)

    if np.any(sizes <= 0.0):
        bad = [names[i] for i in np.where((sizes <= 0.0).any(axis=1))[0]]
        raise ValueError(f"{path}: 以下 chiplet 的宽或高非正: {bad}")

    return record, Layout(names, lower, sizes, power)


def compute_canvas(layout: Layout, scale: float = 1.0, pad_mm: float = 0.0) -> tuple:
    """由输入布局的 bbox 推导正方形画布, 返回 (origin (2,), side)。

    取正方形而非原 bbox 的矩形: ``guidance.legality_potential_boundary`` 和
    ``utils.check_legality_new`` 的可行域都写死是 ``[-1,1]^2`` (正方形), 用非正方形
    画布会让合法性势能的 x/y 权重失真。
    """
    x0, y0, x1, y1 = bbox_mm(layout.lower, layout.sizes)
    width = max(x1 - x0, 0.0)
    height = max(y1 - y0, 0.0)
    side = max(width, height) * float(scale) + 2.0 * float(pad_mm)
    if not np.isfinite(side) or side <= 0.0:
        raise ValueError(f"推导出的画布边长非正: side={side} (bbox {width}x{height})")
    center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float64)
    origin = center - side / 2.0
    return origin, float(side)


def to_norm_centers(lower: np.ndarray, sizes: np.ndarray, origin, side: float) -> np.ndarray:
    """mm 左下角 -> 归一化中心坐标 (V, 2)。"""
    centers_mm = np.asarray(lower, dtype=np.float64) + np.asarray(sizes, dtype=np.float64) / 2.0
    return 2.0 * (centers_mm - np.asarray(origin, dtype=np.float64)) / float(side) - 1.0


def from_norm_centers(centers_norm: np.ndarray, sizes: np.ndarray, origin, side: float) -> np.ndarray:
    """归一化中心坐标 -> mm 左下角 (V, 2)。"""
    centers_mm = (np.asarray(centers_norm, dtype=np.float64) + 1.0) * float(side) / 2.0 + np.asarray(
        origin, dtype=np.float64
    )
    return centers_mm - np.asarray(sizes, dtype=np.float64) / 2.0


def bbox_mm(lower: np.ndarray, sizes: np.ndarray) -> tuple:
    """布局外接框 (x0, y0, x1, y1), mm。"""
    lower = np.asarray(lower, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    x0 = float(lower[:, 0].min())
    y0 = float(lower[:, 1].min())
    x1 = float((lower[:, 0] + sizes[:, 0]).max())
    y1 = float((lower[:, 1] + sizes[:, 1]).max())
    return x0, y0, x1, y1


def bbox_area_mm2(lower: np.ndarray, sizes: np.ndarray) -> float:
    """外接框面积 mm^2。对齐 utils._placement_bbox_stats。"""
    x0, y0, x1, y1 = bbox_mm(lower, sizes)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def aspect_ratio_mm(lower: np.ndarray, sizes: np.ndarray) -> float:
    x0, y0, x1, y1 = bbox_mm(lower, sizes)
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return min(width, height) / max(width, height, 1e-12)


def hpwl_weighted(layout: Layout, connections) -> float:
    """wireCount 加权的中心曼哈顿距离和, 对齐 utils._placement_wirelength (:2092)。

    这是"便宜口径", 用于和神经线长模型交叉验证 (后者 MAPE 1.2%, 但预测的是
    含 bump 绕行的路由线长, 绝对量级高于 HPWL, 只有变化方向可比)。
    """
    index = {name: i for i, name in enumerate(layout.names)}
    centers = layout.centers()
    total = 0.0
    for conn in connections or []:
        node1 = str(conn.get("node1", ""))
        node2 = str(conn.get("node2", ""))
        if node1 not in index or node2 not in index:
            continue
        i, j = index[node1], index[node2]
        weight = float(conn.get("wireCount", conn.get("edge_weight", 1.0)) or 1.0)
        total += weight * float(np.abs(centers[i] - centers[j]).sum())
    return total


def write_placement(record: dict, names, lower: np.ndarray, sizes: np.ndarray, out_path) -> Path:
    """按原 schema 写回 placement JSON。

    ``connections`` 与各 chiplet 的非几何字段 (rotation / power / EMIB*) 原样透传,
    只更新 x-position / y-position; ``wirelength`` / ``area`` / ``aspect_ratio``
    按 utils.py 的口径重算。
    """
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_by_name = {str(ch.get("name", i)): ch for i, ch in enumerate(record["chiplets"])}
    chiplets = []
    for i, name in enumerate(names):
        src = source_by_name.get(str(name), {})
        item = dict(src)
        item["name"] = str(name)
        item["x-position"] = float(lower[i, 0])
        item["y-position"] = float(lower[i, 1])
        item["width"] = float(sizes[i, 0])
        item["height"] = float(sizes[i, 1])
        item.setdefault("rotation", 0)
        chiplets.append(item)

    out = dict(record)
    out["chiplets"] = chiplets

    by_name = {str(ch["name"]): ch for ch in chiplets}
    connections = out.get("connections", [])

    x0, y0, x1, y1 = bbox_mm(lower, sizes)
    out["wirelength"] = float(_placement_wirelength_from_dict(by_name, connections))
    out["area"] = float(max(0.0, x1 - x0) * max(0.0, y1 - y0))
    out["aspect_ratio"] = float(aspect_ratio_mm(lower, sizes))

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    return out_path


def _placement_wirelength_from_dict(by_name: dict, connections) -> float:
    total = 0.0
    for conn in connections or []:
        node1 = str(conn.get("node1", ""))
        node2 = str(conn.get("node2", ""))
        if node1 not in by_name or node2 not in by_name:
            continue
        ch1, ch2 = by_name[node1], by_name[node2]
        c1x = float(ch1["x-position"]) + float(ch1["width"]) / 2.0
        c1y = float(ch1["y-position"]) + float(ch1["height"]) / 2.0
        c2x = float(ch2["x-position"]) + float(ch2["width"]) / 2.0
        c2y = float(ch2["y-position"]) + float(ch2["height"]) / 2.0
        weight = float(conn.get("wireCount", conn.get("edge_weight", 1.0)) or 1.0)
        total += weight * (abs(c1x - c2x) + abs(c1y - c2y))
    return total


def make_report_record(metrics: dict, extra: dict = None) -> dict:
    """把 float 度量整理成可 json.dump 的普通 dict (numpy 标量转 float)。"""
    out = {}
    for key, value in (metrics or {}).items():
        out[key] = _jsonable(value)
    if extra:
        out.update(extra)
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
