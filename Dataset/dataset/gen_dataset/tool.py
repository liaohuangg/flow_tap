"""
Utility functions for chiplet placement experiments.

这里包含：
- 从 JSON 输入中读取 chiplet 信息；
- 构建随机连接图；
- 绘制方框图（chiplet + phys 点 + 连接箭头）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

try:
    import pulp
except ImportError:
    pulp = None

try:
    import gurobipy as gp
except ImportError:
    gp = None

try:
    # 当以 placement.flow_GCN 作为顶层包安装/运行时
    from placement.flow_GCN.Dataset.dataset.input_preprocess import build_chiplet_table, load_chiplets_json
except ModuleNotFoundError:
    # 当直接在本仓库中运行脚本时（Dataset/dataset 作为当前目录或包的一部分）
    try:
        from .input_preprocess import build_chiplet_table, load_chiplets_json  # type: ignore
    except Exception:
        from input_preprocess import build_chiplet_table, load_chiplets_json

# 导入ILP相关的类型（如果可用）
try:
    from ilp_method import ILPModelContext, ILPPlacementResult
except ImportError:
    try:
        from ilp_method_compact import ILPModelContext, ILPPlacementResult
    except ImportError:
        # 如果ilp_method不可用，定义占位类型
        ILPModelContext = None
        ILPPlacementResult = None


def get_var_value(var):
    """
    统一获取变量值的辅助函数，支持 PuLP 和 Gurobi 变量。
    
    参数:
        var: PuLP 或 Gurobi 变量对象，或 None
    
    返回:
        变量的值，如果变量为 None 则返回 None
    """
    if var is None:
        return None
    
    # 检查是否是 Gurobi 变量
    if gp is not None and isinstance(var, gp.Var):
        try:
            return var.X
        except AttributeError:
            return None
    
    # 检查是否是 PuLP 变量
    if pulp is not None:
        try:
            # 检查是否有 value 方法（PuLP 变量）
            if hasattr(var, 'value'):
                return pulp.value(var)
            # 或者直接调用 value() 方法
            elif callable(getattr(var, 'value', None)):
                return var.value()
        except (AttributeError, TypeError):
            return None
    
    return None


# ---------------------------------------------------------------------------
# 数据结构与基础读入
# ---------------------------------------------------------------------------
@dataclass
class EMIBNode:
    """
    存储 EMIB 硅桥 chiplet 信息，由 connection 链接关系生成。
    width = max_Reach_length, height = EMIB_length
    bump_width = json 中的 "EMIB_bump_width"
    """
    node1: str
    node2: str
    wireCount: int
    EMIBType: str
    EMIB_length: float
    EMIB_bump_width: float
    EMIB_max_width: float
    width: float   # = 2*EMIB_bump_width
    height: float  # = EMIB_length

@dataclass
class ChipletNode:
    """A simple wrapper for a chiplet entry."""

    name: str
    dimensions: Dict
    phys: List[Dict]
    power: float

def build_bump_region_map(edges: List[dict], name_to_idx: Dict[str, int]) -> Dict[Tuple[int, int, int], dict]:
    """
    构造 bump_region 映射表
    key: (i, j, k) -> i,j 是互联对索引(i<j)，k 是所属 chiplet 索引
    value: {"length": EMIB_length, "width": bump_width}
    """
    bump_region_map = {}
    
    for edge in edges:
        # 仅处理包含 EMIB 硅桥的连接
        if edge.get("EMIBType") == "interfaceC":
            continue
            
        # 获取索引并排序，确保键值的 (i, j) 顺序一致
        idx1 = name_to_idx[edge["node1"]]
        idx2 = name_to_idx[edge["node2"]]
        i, j = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
        
        # 提取逻辑尺寸
        # length 通常对应笔记中的 EMIB_length (沿边方向)
        # width 通常对应预设的 bump 宽度 (垂直于边方向)
        emib_l = float(edge["EMIB_length"])
        emib_w = float(edge["EMIB_bump_width"])
        # 存入字典，一对连接对应两个标号
        bump_region_map[(i, j, i)] = {"length": emib_l, "width": emib_w}
        bump_region_map[(i, j, j)] = {"length": emib_l, "width": emib_w}
        
    return bump_region_map

def _parse_emib_connection(conn: dict, ctx: str = "") -> List:
    """
    将 JSON 中的单条连接解析为 [src, dst, wireCount, EMIBType, EMIB_length, EMIB_max_width, EMIB_bump_width]。
    仅支持对象格式：{node1, node2, wireCount, EMIBType, EMIB_length, EMIB_max_width, EMIB_bump_width}。
    """
    if not isinstance(conn, dict):
        raise ValueError(f"连接格式错误：必须是对象 {{node1, node2, wireCount, EMIBType, EMIB_length, EMIB_max_width, EMIB_bump_width}}。{ctx}当前连接: {conn}")
    src = conn.get("node1")
    dst = conn.get("node2")
    weight = conn.get("wireCount")
    emib_type = conn.get("EMIBType")
    emib_length = conn.get("EMIB_length")
    emib_max_width = conn.get("EMIB_max_width")
    emib_bump_width = conn.get("EMIB_bump_width")
    missing = []
    if src is None:
        missing.append("node1")
    if dst is None:
        missing.append("node2")
    if weight is None:
        missing.append("wireCount")
    if emib_type is None:
        missing.append("EMIBType")
    if emib_length is None:
        missing.append("EMIB_length")
    if emib_max_width is None:
        missing.append("EMIB_max_width")
    if emib_bump_width is None:
        missing.append("EMIB_bump_width")
    if missing:
        raise ValueError(f"连接格式错误：缺少必填字段 {missing}。{ctx}当前连接: {conn}")
    try:
        emib_length_f = float(emib_length)
        emib_max_width_f = float(emib_max_width)
        bump_width_f = float(emib_bump_width)
        weight_int = int(weight)
    except (TypeError, ValueError) as e:
        raise ValueError(f"连接格式错误：wireCount、EMIB_length、EMIB_max_width、EMIB_bump_width 必须为有效数值。{ctx}当前连接: {conn}") from e
    return [str(src), str(dst), weight_int, str(emib_type), emib_length_f, emib_max_width_f, bump_width_f]


def _emib_type_to_conn_type(emib_type: str) -> int:
    """EMIBType -> conn_type: interfaceA/interfaceB -> 1 (silicon_bridge), 其余 -> 0 (standard)"""
    return 1 if emib_type in ("interfaceA", "interfaceB") else 0


def load_emib_placement_json(
    json_path: str,
) -> Tuple[List["ChipletNode"], List[Tuple], Dict[Tuple[str, str], Dict], Dict[str, int]]:
    """
    从 JSON 文件加载 EMIB 布局输入。
    格式：{"chiplets": [...], "connections": [...]}
    connections 每条为对象：{node1, node2, wireCount, EMIBType, EMIB_length, EMIB_max_width, EMIB_bump_width}
    不满足格式条件或存在重复连接对时直接报错。

    返回:
        nodes: ChipletNode 列表
        edges: [{"node1", "node2", "wireCount", "EMIBType", "EMIB_length", "EMIB_max_width", "EMIB_bump_width"}, ...] 键值对列表，可原地修改
        edge_map: {(a,b): 同 edges 元素的字典引用}, ...}
        name_to_idx: {chiplet_name: index}
    """
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "chiplets" not in data or not isinstance(data["chiplets"], list):
        raise ValueError(f"JSON文件格式错误：必须包含 'chiplets' 列表字段。文件: {json_path}")

    nodes: List[ChipletNode] = []
    for i, chiplet_info in enumerate(data["chiplets"]):
        if not isinstance(chiplet_info, dict):
            raise ValueError(f"chiplet 格式错误：每个 chiplet 必须是对象。索引 {i}: {chiplet_info}")
        name = chiplet_info.get("name")
        width = chiplet_info.get("width")
        height = chiplet_info.get("height")
        power = chiplet_info.get("power")
        missing = []
        if name is None:
            missing.append("name")
        if width is None:
            missing.append("width")
        if height is None:
            missing.append("height")
        if power is None:
            missing.append("power")
        if missing:
            raise ValueError(f"chiplet 格式错误：缺少必填字段 {missing}。索引 {i}: {chiplet_info}")
        try:
            w_f, h_f = float(width), float(height)
            p_f = float(power)
        except (TypeError, ValueError) as e:
            raise ValueError(f"chiplet 格式错误：width、height、power 必须为有效数值。索引 {i}: {chiplet_info}") from e
        nodes.append(
            ChipletNode(
                name=str(name),
                dimensions={"x": w_f, "y": h_f},
                phys=[],
                power=p_f,
            )
        )

    if "connections" not in data or not isinstance(data["connections"], list):
        raise ValueError(f"JSON文件格式错误：必须包含 'connections' 列表字段。文件: {json_path}")
    if len(data["connections"]) == 0:
        raise ValueError(f"JSON文件格式错误：connections 不能为空。文件: {json_path}")

    chiplet_names = {n.name for n in nodes}
    connections: List[List] = []
    seen_pairs: set = set()
    for idx, conn in enumerate(data["connections"]):
        parsed = _parse_emib_connection(conn, ctx=f"connections[{idx}] ")
        a, b = (parsed[0], parsed[1]) if parsed[0] <= parsed[1] else (parsed[1], parsed[0])
        if a not in chiplet_names or b not in chiplet_names:
            raise ValueError(f"连接格式错误：节点 {a} 或 {b} 不在 chiplets 中。connections[{idx}]: {conn}")
        pair = (a, b)
        if pair in seen_pairs:
            raise ValueError(f"连接格式错误：存在重复连接对 ({a}, {b})。connections[{idx}]: {conn}")
        seen_pairs.add(pair)
        connections.append(parsed)

    name_to_idx = {node.name: k for k, node in enumerate(nodes)}
    chiplet_names_set = set(name_to_idx.keys())

    edge_map: Dict[Tuple[str, str], Dict] = {}
    for idx, row in enumerate(connections):
        if len(row) < 7:
            raise ValueError(f"连接格式错误：必须有 7 列。connections[{idx}]: {row}")
        s, t = row[0], row[1]
        emib_type = row[3]
        try:
            w = float(row[2])
            emib_len = float(row[4])
            max_width = float(row[5])
            bump_width = float(row[6])
        except (TypeError, ValueError) as e:
            raise ValueError(f"连接格式错误：wireCount、EMIB_length、EMIB_max_width、EMIB_bump_width 必须为有效数值。connections[{idx}]: {row}") from e
        ct = _emib_type_to_conn_type(emib_type)
        a, b = (s, t) if s <= t else (t, s)
        if a not in chiplet_names_set or b not in chiplet_names_set:
            raise ValueError(f"连接格式错误：节点 {a} 或 {b} 不在 chiplets 中。connections[{idx}]: {row}")
        if (a, b) in edge_map:
            raise ValueError(f"连接格式错误：存在重复连接对 ({a}, {b})。connections[{idx}]: {row}")
        edge_map[(a, b)] = {
            "node1": a,
            "node2": b,
            "wireCount": w,
            "conn_type": ct,
            "EMIBType": emib_type,
            "EMIB_length": emib_len,
            "EMIB_width": 2 * bump_width + max_width,
            "EMIB_max_width": max_width,
            "EMIB_bump_width": bump_width,
        }

    # edges: 列表，每个元素为字典（键值对），可原地修改
    edges = [v for (a, b), v in edge_map.items()]

    return nodes, edges, edge_map, name_to_idx


def print_emib_node_contents(
    emib_node_dict: Dict[Tuple, "EMIBNode"],
    key_formatter: Optional[Callable[[Tuple], str]] = None,
    prefix: str = "  ",
) -> None:
    """
    输出 EMIBNode 字典中每条连接的全部内容。
    emib_node_dict: key 为 (i,j) 或 (node1, node2)，value 为 EMIBNode
    key_formatter: 可选，将 key 格式化为字符串，例如 lambda k: f"({k[0]},{k[1]})"
    """
    for key, e in emib_node_dict.items():
        key_str = key_formatter(key) if key_formatter else str(key)
        print(f"{prefix}{key_str}: node1={e.node1}, node2={e.node2}, wireCount={e.wireCount}, "
              f"EMIBType={e.EMIBType}, EMIB_length={e.EMIB_length}, EMIB_max_width={e.EMIB_max_width}, "
              f"width={e.width}, height={e.height}, bump_width={e.EMIB_bump_width:.4f}")
# ---------------------------------------------------------------------------
# 硅桥精准定位与全链路布线距离计算（EMIB 后处理）
# ---------------------------------------------------------------------------


def _get_gurobi_var_val(var, default: float = 0.0) -> float:
    """从 Gurobi 变量获取求解值，兼容 None 或不可用情况。"""
    if var is None:
        return default
    try:
        return float(var.X)
    except (AttributeError, TypeError):
        return default


GRID_SIZE = 16  # 每个芯粒有效区域 16x16 均匀网格
EMIB_CENTER_STEP = 0.01  # 硅桥中心点搜索步长


def generate_chiplet_wire_grid_16x16(
    chiplet_layout: Dict[str, Tuple[float, float]],
    chiplet_dims: Dict[str, Tuple[float, float]],
    node_name: str,
    display_size: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """
    将芯粒长宽划分为 16x16 网格，返回网格点坐标。
    display_size 为 None 或 16 时返回 256 个点；display_size=4 时返回 4x4=16 个点（每 4x4 块的中心）。
    这些交叉点即为：线到硅桥中心点的起点 / 硅桥中心点到线的终点。
    
    返回
    ----
    List[Tuple[float, float]]
        网格点坐标 [(x,y), ...]，行优先顺序
    """
    x, y = chiplet_layout.get(node_name, (0, 0))
    w, h = chiplet_dims.get(node_name, (0, 0))
    size = display_size if display_size is not None and 1 <= display_size <= GRID_SIZE else GRID_SIZE
    if size == GRID_SIZE:
        pts = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                px = x + i * w / max(1, GRID_SIZE - 1)
                py = y + j * h / max(1, GRID_SIZE - 1)
                pts.append((px, py))
        return pts
    block = GRID_SIZE // size
    pts = []
    for bi in range(size):
        for bj in range(size):
            ci = bi * block + (block - 1) / 2
            cj = bj * block + (block - 1) / 2
            px = x + ci * w / max(1, GRID_SIZE - 1)
            py = y + cj * h / max(1, GRID_SIZE - 1)
            pts.append((px, py))
    return pts


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """欧氏距离"""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def compute_optimal_emib_center(
    src_grid_points: List[Tuple[float, float]],
    tgt_grid_points: List[Tuple[float, float]],
    shared_segment: Tuple[Tuple[float, float], Tuple[float, float]],
    step: float = EMIB_CENTER_STEP,
) -> Tuple[float, float]:
    """
    在共享边上遍历搜索，找到使总路径长度最短的硅桥中心点。
    总路径 = sum(源网格点到中心) + sum(目标网格点到中心)
    
    参数
    ----
    src_grid_points : List[Tuple[float, float]]
        源芯粒 16x16 网格点
    tgt_grid_points : List[Tuple[float, float]]
        目标芯粒 16x16 网格点
    shared_segment : Tuple[Tuple, Tuple]
        共享边线段 ((x1,y1), (x2,y2))
    
    返回
    ----
    Tuple[float, float]
        最优硅桥中心点坐标
    """
    (xa, ya), (xb, yb) = shared_segment
    seg_len = math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
    if seg_len <= 1e-9:
        return ((xa + xb) / 2, (ya + yb) / 2)
    best_center = None
    best_total = float("inf")
    n_steps = max(1, int(seg_len / step) + 1)
    for k in range(n_steps + 1):
        t = k / n_steps if n_steps > 0 else 0.5
        cx = xa + t * (xb - xa)
        cy = ya + t * (yb - ya)
        total = sum(_dist(p, (cx, cy)) for p in src_grid_points) + sum(
            _dist(p, (cx, cy)) for p in tgt_grid_points
        )
        if total < best_total:
            best_total = total
            best_center = (cx, cy)
    return best_center or ((xa + xb) / 2, (ya + yb) / 2)


def compute_emib_placement(
    chiplet_layout: Dict[str, Tuple[float, float]],
    chiplet_dims: Dict[str, Tuple[float, float]],
    emib_connections: List[dict],
    direction_vars: Dict[Tuple[int, int], Dict[str, float]],
    name_to_idx: Dict[str, int],
    idx_to_name: Dict[int, str],
) -> List[dict]:
    """
    从 ILP 求解结果提取硅桥精准放置位置，并计算最优硅桥中心点（总路径最短）。
    
    根据方向变量（z1L/z1R/z2D/z2U）判定芯粒相邻方式，匹配共享边长，
    校验 EMIB_length，在共享边上搜索使布线总路径最短的中心点。
    
    参数
    ----
    chiplet_layout : Dict[str, Tuple[float, float]]
        芯粒左下角坐标 name -> (x, y)
    chiplet_dims : Dict[str, Tuple[float, float]]
        芯粒尺寸（考虑旋转后）name -> (width, height)
    emib_connections : List[dict]
        仅含 EMIB 连接（EMIBType != interfaceC），每项 {node1, node2, wireCount, EMIB_length}
    direction_vars : Dict[Tuple[int,int], Dict[str, float]]
        (i,j) -> {z1L, z1R, z2D, z2U, z1, z2} 的求解值（0 或 1）
    name_to_idx : Dict[str, int]
        芯粒名称到索引
    idx_to_name : Dict[int, str]
        索引到芯粒名称
    
    返回
    ----
    List[dict]
        每项: {
            "emib_id": str,
            "node1": str, "node2": str,
            "direction": "horizontal" | "vertical",
            "x_start": float, "y_start": float, "x_end": float, "y_end": float,
            "emib_physical_dist": float,  # EMIB_max_width：芯粒间实际相隔距离（左右相邻=j左-i右；上下相邻=j下-i上）
            "shared_length": float,
            "emib_length_required": float,
            "emib_center": Tuple[float, float],  # 最优硅桥中心点
            "ok": bool,
            "warning": str | None,
        }
    """
    results = []
    for conn in emib_connections:
        n1, n2 = conn["node1"], conn["node2"]
        emib_len = float(conn["EMIB_length"])
        wire_count = int(conn.get("wireCount", 0))
        i, j = name_to_idx.get(n1), name_to_idx.get(n2)
        if i is None or j is None:
            results.append({
                "emib_id": f"{n1}-{n2}",
                "node1": n1, "node2": n2,
                "direction": None, "x_start": 0, "y_start": 0, "x_end": 0, "y_end": 0,
                "emib_physical_dist": 0, "shared_length": 0, "emib_length_required": emib_len,
                "ok": False, "warning": f"节点 {n1} 或 {n2} 不在布局中",
            })
            continue
        if i > j:
            i, j = j, i
            n1, n2 = n2, n1
        dv = direction_vars.get((i, j), {})
        z1L = dv.get("z1L", 0) > 0.5
        z1R = dv.get("z1R", 0) > 0.5
        z2D = dv.get("z2D", 0) > 0.5
        z2U = dv.get("z2U", 0) > 0.5
        z1 = dv.get("z1", 0) > 0.5
        z2 = dv.get("z2", 0) > 0.5

        xi, yi = chiplet_layout.get(n1, (0, 0))
        xj, yj = chiplet_layout.get(n2, (0, 0))
        wi, hi = chiplet_dims.get(n1, (0, 0))
        wj, hj = chiplet_dims.get(n2, (0, 0))

        emib_id = f"{n1}-{n2}"
        direction = None
        x_start, y_start, x_end, y_end = 0.0, 0.0, 0.0, 0.0
        emib_physical = 0.0
        shared_len = 0.0
        ok = True
        warning = None

        if z1:
            # 水平相邻：共享边在 y 方向，硅桥跨越 x 方向间隙
            direction = "horizontal"
            y_low = max(yi, yj)
            y_high = min(yi + hi, yj + hj)
            shared_len = max(0, y_high - y_low)
            if shared_len < emib_len - 1e-6:
                ok = False
                warning = f"共享边长 {shared_len:.4f} < EMIB_length {emib_len:.4f}"
            if z1L:
                # 左右相邻：i 在左、j 在右。EMIB_max_width = j左 - i右 = xj - (xi+wi)
                x_start = xi + wi
                x_end = xj
                emib_physical = xj - (xi + wi)
                y_start, y_end = y_low, y_high
            elif z1R:
                # 左右相邻：i 在右、j 在左。EMIB_max_width = i左 - j右 = xi - (xj+wj)
                x_start = xj + wj
                x_end = xi
                emib_physical = xi - (xj + wj)
                y_start, y_end = y_low, y_high
        elif z2:
            # 垂直相邻：共享边在 x 方向，硅桥跨越 y 方向间隙
            direction = "vertical"
            x_low = max(xi, xj)
            x_high = min(xi + wi, xj + wj)
            shared_len = max(0, x_high - x_low)
            if shared_len < emib_len - 1e-6:
                ok = False
                warning = f"共享边长 {shared_len:.4f} < EMIB_length {emib_len:.4f}"
            if z2D:
                # 上下相邻：i 在下、j 在上。EMIB_max_width = j下 - i上 = yj - (yi+hi)
                y_start = yi + hi
                y_end = yj
                emib_physical = yj - (yi + hi)
                x_start, x_end = x_low, x_high
            elif z2U:
                # 上下相邻：i 在上、j 在下。EMIB_max_width = i下 - j上 = yi - (yj+hj)
                y_start = yj + hj
                y_end = yi
                emib_physical = yi - (yj + hj)
                x_start, x_end = x_low, x_high
        else:
            ok = False
            warning = "无法确定相邻方向 (z1/z2 均为 0)"

        # 生成 16x16 网格点，计算最优硅桥中心（使总路径最短）
        emib_center = None
        if direction and shared_len > 1e-9:
            src_grid = generate_chiplet_wire_grid_16x16(chiplet_layout, chiplet_dims, n1)
            tgt_grid = generate_chiplet_wire_grid_16x16(chiplet_layout, chiplet_dims, n2)
            y_lo, y_hi = min(y_start, y_end), max(y_start, y_end)
            x_lo, x_hi = min(x_start, x_end), max(x_start, x_end)
            if direction == "horizontal":
                x_mid = (x_start + x_end) / 2
                shared_seg = ((x_mid, y_lo), (x_mid, y_hi))
            else:
                y_mid = (y_start + y_end) / 2
                shared_seg = ((x_lo, y_mid), (x_hi, y_mid))
            emib_center = compute_optimal_emib_center(src_grid, tgt_grid, shared_seg, step=EMIB_CENTER_STEP)
        else:
            if direction == "horizontal":
                emib_center = ((x_start + x_end) / 2, (y_start + y_end) / 2)
            elif direction == "vertical":
                emib_center = ((x_start + x_end) / 2, (y_start + y_end) / 2)
            else:
                emib_center = (0.0, 0.0)

        results.append({
            "emib_id": emib_id,
            "node1": n1, "node2": n2,
            "direction": direction,
            "x_start": x_start, "y_start": y_start, "x_end": x_end, "y_end": y_end,
            "emib_physical_dist": emib_physical,
            "shared_length": shared_len,
            "emib_length_required": emib_len,
            "emib_center": emib_center,
            "ok": ok,
            "warning": warning,
        })
    return results


def layout_wire_endpoints(
    emib_placement: dict,
    chiplet_layout: Dict[str, Tuple[float, float]],
    chiplet_dims: Dict[str, Tuple[float, float]],
    wire_count: int,
    node1_name: str,
    node2_name: str,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    将 wireCount 条布线的起止点分别在源/目标芯粒连接边上网格状均匀分布。
    
    连接边为两芯粒的共享边区域，起止点沿该区域均匀排布。
    若 wireCount 无法整除网格，采用最近可排布数量并尽量均匀。
    
    参数
    ----
    emib_placement : dict
        compute_emib_placement 返回的单条硅桥信息
    chiplet_layout : Dict[str, Tuple[float, float]]
        芯粒左下角坐标
    chiplet_dims : Dict[str, Tuple[float, float]]
        芯粒尺寸
    wire_count : int
        互联线数量
    node1_name : str
        源芯粒名称（约定为靠近 (x_start,y_start) 一侧）
    node2_name : str
        目标芯粒名称
    
    返回
    ----
    (start_points, end_points)
        start_points: 源芯粒上 wireCount 个起点 [(x,y), ...]
        end_points: 目标芯粒上 wireCount 个终点 [(x,y), ...]
    """
    direction = emib_placement.get("direction")
    x_s, y_s = emib_placement["x_start"], emib_placement["y_start"]
    x_e, y_e = emib_placement["x_end"], emib_placement["y_end"]
    shared_len = emib_placement["shared_length"]

    wire_count = max(1, int(wire_count))
    n = wire_count

    start_pts = []
    end_pts = []

    if direction == "horizontal":
        # 共享边沿 y 方向，起止点 y 坐标均匀分布
        if shared_len <= 1e-9:
            for _ in range(n):
                start_pts.append((x_s, y_s))
                end_pts.append((x_e, y_e))
        else:
            for k in range(n):
                t = (k + 1) / (n + 1) if n > 0 else 0.5
                y_pt = y_s + t * (y_e - y_s) if abs(y_e - y_s) > 1e-9 else y_s
                start_pts.append((x_s, y_pt))
                end_pts.append((x_e, y_pt))
    elif direction == "vertical":
        # 共享边沿 x 方向，起止点 x 坐标均匀分布
        if shared_len <= 1e-9:
            for _ in range(n):
                start_pts.append((x_s, y_s))
                end_pts.append((x_e, y_e))
        else:
            for k in range(n):
                t = (k + 1) / (n + 1) if n > 0 else 0.5
                x_pt = x_s + t * (x_e - x_s) if abs(x_e - x_s) > 1e-9 else x_s
                start_pts.append((x_pt, y_s))
                end_pts.append((x_pt, y_e))
    else:
        for _ in range(n):
            start_pts.append((x_s, y_s))
            end_pts.append((x_e, y_e))
    return start_pts, end_pts


def compute_wire_distances(
    start_points: List[Tuple[float, float]],
    end_points: List[Tuple[float, float]],
    emib_placement: dict,
) -> List[dict]:
    """
    计算每根互联线的分段距离与全链路总距离。
    
    （1）起始点到硅桥入口的直线距离
    （2）硅桥出口到结束点的直线距离
    （3）硅桥自身的物理距离（芯粒边缘间距）
    （4）全链路总距离 = (1) + (2) + (3)
    
    参数
    ----
    start_points : List[Tuple[float, float]]
        源芯粒上各布线起点
    end_points : List[Tuple[float, float]]
        目标芯粒上各布线终点
    emib_placement : dict
        compute_emib_placement 返回的单条硅桥信息
    
    返回
    ----
    List[dict]
        每根线: {
            "wire_id": int,
            "start": (x,y), "end": (x,y),
            "dist_start_to_emib": float,
            "dist_emib_to_end": float,
            "emib_physical_dist": float,
            "total_dist": float,
        }
    """
    emib_phys = emib_placement["emib_physical_dist"]

    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    results = []
    n = min(len(start_points), len(end_points))
    # 起止点在连接边上时：起点=硅桥入口，终点=硅桥出口，故 dist_start_to_emib=0, dist_emib_to_end=0
    # 全链路总距离 = 直线距离(start, end)，约等于 emib_physical_dist（当起止点在同 y 或同 x 时）
    for i in range(n):
        ps, pe = start_points[i], end_points[i]
        total = _dist(ps, pe)
        results.append({
            "wire_id": i,
            "start": ps, "end": pe,
            "dist_start_to_emib": 0.0,
            "dist_emib_to_end": 0.0,
            "emib_physical_dist": emib_phys,
            "total_dist": total,
        })
    return results


def run_emib_post_process(
    ctx,
    result,
    nodes: List,
    edge_map: Dict,
    name_to_idx: Dict[str, int],
) -> dict:
    """
    整合硅桥定位、布线点布局、距离计算的完整后处理流程。
    
    从 ILP 求解结果（ctx + result）中提取变量值，调用三大核心函数，
    输出硅桥定位结果与每根线的距离数据。
    
    参数
    ----
    ctx : ILPModelContext
        Gurobi ILP 模型上下文
    result : ILPPlacementResult
        求解结果
    nodes : List[ChipletNode]
        芯粒列表
    edge_map : Dict[Tuple[str,str], dict]
        (node1, node2) -> {node1, node2, wireCount, EMIB_length, EMIBType, ...}
    name_to_idx : Dict[str, int]
        芯粒名称到索引
    
    返回
    ----
    dict
        {
            "emib_placements": List[dict],
            "wire_distances": Dict[str, List[dict]],  # emib_id -> 每根线的距离
        }
    """
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    layout = result.layout if hasattr(result, "layout") else {}
    rotations = result.rotations if hasattr(result, "rotations") else {}

    chiplet_dims = {}
    for node in nodes:
        w0 = float(node.dimensions.get("x", 0) or 0)
        h0 = float(node.dimensions.get("y", 0) or 0)
        rot = rotations.get(node.name, False)
        chiplet_dims[node.name] = (h0, w0) if rot else (w0, h0)

    direction_vars = {}
    all_connected = getattr(ctx, "all_connected_pairs", {}) or {}
    for (i, j), edge in all_connected.items():
        if edge.get("EMIBType") == "interfaceC":
            continue
        z1 = getattr(ctx, "z1", None)
        z1L = getattr(ctx, "z1L", None)
        z1R = getattr(ctx, "z1R", None)
        z2 = getattr(ctx, "z2", None)
        z2D = getattr(ctx, "z2D", None)
        z2U = getattr(ctx, "z2U", None)
        dv = {}
        if z1 and (i, j) in z1:
            dv["z1"] = _get_gurobi_var_val(z1[(i, j)])
        if z2 and (i, j) in z2:
            dv["z2"] = _get_gurobi_var_val(z2[(i, j)])
        if z1L and (i, j) in z1L:
            dv["z1L"] = _get_gurobi_var_val(z1L[(i, j)])
        if z1R and (i, j) in z1R:
            dv["z1R"] = _get_gurobi_var_val(z1R[(i, j)])
        if z2D and (i, j) in z2D:
            dv["z2D"] = _get_gurobi_var_val(z2D[(i, j)])
        if z2U and (i, j) in z2U:
            dv["z2U"] = _get_gurobi_var_val(z2U[(i, j)])
        direction_vars[(i, j)] = dv

    emib_connections = []
    for (a, b), v in edge_map.items():
        if v.get("EMIBType") == "interfaceC":
            continue
        emib_connections.append({
            "node1": a, "node2": b,
            "wireCount": v.get("wireCount", 0),
            "EMIB_length": v.get("EMIB_length", 0),
        })

    emib_placements = compute_emib_placement(
        chiplet_layout=layout,
        chiplet_dims=chiplet_dims,
        emib_connections=emib_connections,
        direction_vars=direction_vars,
        name_to_idx=name_to_idx,
        idx_to_name=idx_to_name,
    )

    wire_distances = {}
    for emp, conn in zip(emib_placements, emib_connections):
        emib_id = emp["emib_id"]
        wc = int(conn.get("wireCount", 0))
        start_pts, end_pts = layout_wire_endpoints(
            emib_placement=emp,
            chiplet_layout=layout,
            chiplet_dims=chiplet_dims,
            wire_count=wc,
            node1_name=conn["node1"],
            node2_name=conn["node2"],
        )
        wire_distances[emib_id] = compute_wire_distances(
            start_points=start_pts,
            end_points=end_pts,
            emib_placement=emp,
        )
    return {
        "emib_placements": emib_placements,
        "wire_distances": wire_distances,
    }


def compute_emib_bottom_left(
    emib_center: Tuple[float, float],
    direction: str,
    emib_bump_width: float,
    emib_max_width: float,
    emib_length: float,
) -> Tuple[float, float]:
    """
    根据硅桥中心点、放置形态及固定参数，计算硅桥左下角 (x, y) 坐标。

    水平放置（芯粒左右相邻，不旋转）：
        x = center_x - EMIB_bump_width - EMIB_max_width/2
        y = center_y - EMIB_length/2

    垂直放置（芯粒上下相邻，旋转 90°）：
        x = center_x - EMIB_length/2
        y = center_y - EMIB_bump_width - EMIB_max_width/2

    参数
    ----
    emib_center : (cx, cy)
        硅桥中心点坐标
    direction : "horizontal" | "vertical"
        硅桥放置形态
    emib_bump_width, emib_length : float
        硅桥固定设计参数
    emib_max_width : float
        芯粒间实际相隔距离（动态计算，来自 emib_physical_dist）

    返回
    ----
    (x, y) 硅桥左下角坐标
    """
    cx, cy = emib_center
    if direction == "horizontal":
        # 硅桥水平放置：左下角 x = 中心x - bump_width - max_width/2
        x = cx - emib_bump_width - emib_max_width / 2
        # 左下角 y = 中心y - length/2
        y = cy - emib_length / 2
    else:
        # 硅桥垂直放置（旋转）：左下角 x = 中心x - length/2
        x = cx - emib_length / 2
        # 左下角 y = 中心y - bump_width - max_width/2
        y = cy - emib_bump_width - emib_max_width / 2
    return (x, y)

def generate_placement_json_with_EMIB(
    result,
    post: dict,
    nodes: list,
    edge_map: dict,
    output_path: str,
    emib_bump_width_override: Optional[float] = None,
    emib_length_override: Optional[float] = None,
    ctx=None,
) -> dict:
    """
    从 ILP 求解结果中，直接得到硅桥的坐标、长宽与旋转，生成 placement JSON。

    x-position、y-position、EMIB_length、EMIB_width、EMIB-rotation 必须从 ctx 的 ILP 变量
    (EMIB_x_grid_var, EMIB_y_grid_var, EMIB_w_var, EMIB_h_var, r_EMIB) 中读取；
    若 ctx 无 EMIB 变量，则回退到 result.placements（兼容旧字段 result.emib_placements）。

    参数
    ----
    result : ILPPlacementResult
        求解结果（含 layout, rotations, bounding_box）
    post : dict | None
        run_emib_post_process 返回值；可为 None
    nodes : list
        芯粒列表
    edge_map : dict
        (node1, node2) -> edge 信息（含 EMIB_bump_width, EMIB_length 等）
    output_path : str
        输出 JSON 文件路径
    emib_bump_width_override, emib_length_override : float | None
        可选，覆盖 edge 中的固定设计参数
    ctx : ILPModelContext | None
        若含 EMIB_x_grid_var 等变量，则从 ILP 解中直接读取硅桥位置与尺寸

    返回
    ----
    dict
        生成的 placement 数据结构
    """
    import json
    from pathlib import Path

    def _r3(v):
        return round(float(v or 0), 3)

    def _rects_overlap_xywh(
        ax: float,
        ay: float,
        aw: float,
        ah: float,
        bx: float,
        by: float,
        bw: float,
        bh: float,
    ) -> bool:
        # touching edges is allowed (no overlap)
        if ax + aw <= bx:
            return False
        if bx + bw <= ax:
            return False
        if ay + ah <= by:
            return False
        if by + bh <= ay:
            return False
        return True

    def _assert_no_overlap_chiplets(chiplets: list) -> None:
        rects = []
        for c in chiplets:
            rects.append(
                (
                    str(c.get("name")),
                    float(c.get("x-position", 0.0) or 0.0),
                    float(c.get("y-position", 0.0) or 0.0),
                    float(c.get("width", 0.0) or 0.0),
                    float(c.get("height", 0.0) or 0.0),
                )
            )
        rects.sort(key=lambda t: (t[0], t[1], t[2]))
        for i in range(len(rects)):
            ni, xi, yi, wi, hi = rects[i]
            for j in range(i + 1, len(rects)):
                nj, xj, yj, wj, hj = rects[j]
                if _rects_overlap_xywh(xi, yi, wi, hi, xj, yj, wj, hj):
                    raise RuntimeError(
                        f"Illegal placement: overlap detected between {ni} and {nj}: "
                        f"{ni}=(x={xi},y={yi},w={wi},h={hi}), {nj}=(x={xj},y={yj},w={wj},h={hj})"
                    )

    layout = result.layout if hasattr(result, "layout") else {}
    rotations = result.rotations if hasattr(result, "rotations") else {}
    bbox = getattr(result, "bounding_box", (0, 0))
    bbox_w, bbox_h = float(bbox[0]) if bbox else 0, float(bbox[1]) if bbox else 0

    chiplet_dims = {}
    min_w, min_h = float("inf"), float("inf")
    for node in nodes:
        w0 = float(node.dimensions.get("x", 0) or 0)
        h0 = float(node.dimensions.get("y", 0) or 0)
        min_w, min_h = min(min_w, w0), min(min_h, h0)
        rot = rotations.get(node.name, False)
        chiplet_dims[node.name] = (h0, w0) if rot else (w0, h0)

    chiplets_list = []
    for node in nodes:
        x, y = layout.get(node.name, (0, 0))
        w, h = chiplet_dims[node.name]
        rot = 1 if rotations.get(node.name, False) else 0
        power = float(getattr(node, "power", 0) or 0)
        chiplets_list.append({
            "name": node.name,
            "x-position": _r3(x),
            "y-position": _r3(y),
            "width": _r3(w),
            "height": _r3(h),
            "rotation": rot,
            "power": _r3(power),
        })

    _assert_no_overlap_chiplets(chiplets_list)

    connections_list = []
    emib_connected = getattr(ctx, "EMIB_connected_pairs", None) if ctx else None
    emib_has_vars = emib_connected and getattr(ctx, "EMIB_x_grid_var", None)
    if emib_has_vars:
        # 必须从 ILP 求解变量中直接读取：EMIB_x_grid_var, EMIB_y_grid_var, EMIB_w_var, EMIB_h_var, r_EMIB
        for (i, j), emib_node in emib_connected.items():
            ex = float(ctx.EMIB_x_grid_var[(i, j)].X) if (i, j) in ctx.EMIB_x_grid_var else 0.0
            ey = float(ctx.EMIB_y_grid_var[(i, j)].X) if (i, j) in ctx.EMIB_y_grid_var else 0.0
            ew = float(ctx.EMIB_w_var[(i, j)].X) if (i, j) in ctx.EMIB_w_var else 0.0
            eh = float(ctx.EMIB_h_var[(i, j)].X) if (i, j) in ctx.EMIB_h_var else 0.0
            er = bool(ctx.r_EMIB[(i, j)].X > 0.5) if (i, j) in ctx.r_EMIB else False
            na = nodes[i].name if i < len(nodes) else str(i)
            nb = nodes[j].name if j < len(nodes) else str(j)
            a, b = (na, nb) if na <= nb else (nb, na)
            edge = edge_map.get((a, b), {})
            emib_bump_width = emib_bump_width_override if emib_bump_width_override is not None else float(edge.get("EMIB_bump_width", 0) or getattr(emib_node, "EMIB_bump_width", 0) or 0)
            emib_max_width = float(edge.get("EMIB_max_width", getattr(emib_node, "EMIB_max_width", 0)) or 0)
            connections_list.append({
                "node1": na, "node2": nb,
                "EMIBType": edge.get("EMIBType", getattr(emib_node, "EMIBType", "interfaceB")),
                "EMIB_length": _r3(eh),
                "EMIB_max_width": _r3(emib_max_width),
                "EMIB_width": _r3(ew),
                "EMIB_bump_width": _r3(emib_bump_width),
                "EMIB-x-position": _r3(ex),
                "EMIB-y-position": _r3(ey),
                "EMIB-rotation": 1 if er else 0,
            })
    else:
        # 非 ILP(无 ctx EMIB 变量) 时：允许用 result.placements（优先）或旧字段 result.emib_placements（兼容）
        result_placements = getattr(result, "placements", None) if result else None
        if result_placements is None:
            result_placements = getattr(result, "emib_placements", None) if result else None

        if result_placements:
            for emp in result_placements:
                na, nb = emp.get("node1"), emp.get("node2")
                a, b = (na, nb) if na <= nb else (nb, na)
                edge = edge_map.get((a, b), {})
                emib_bump_width = emib_bump_width_override if emib_bump_width_override is not None else float(edge.get("EMIB_bump_width", 0) or 0)
                emib_max_width = float(edge.get("EMIB_max_width", 0) or 0)
                connections_list.append({
                    "node1": na, "node2": nb,
                    "EMIBType": edge.get("EMIBType", "interfaceB"),
                    "EMIB_length": _r3(emp.get("EMIB_length", 0)),
                    "EMIB_max_width": _r3(emib_max_width),
                    "EMIB_width": _r3(emp.get("EMIB_width", 0)),
                    "EMIB_bump_width": _r3(emib_bump_width),
                    "EMIB-x-position": _r3(emp.get("EMIB-x-position", 0)),
                    "EMIB-y-position": _r3(emp.get("EMIB-y-position", 0)),
                    "EMIB-rotation": emp.get("EMIB-rotation", 0),
                })

    wirelength = 0.0
    for (a, b), edge in edge_map.items():
        n1, n2 = edge.get("node1", a), edge.get("node2", b)
        x1, y1 = layout.get(n1, (0, 0))
        x2, y2 = layout.get(n2, (0, 0))
        w1, h1 = chiplet_dims.get(n1, (0, 0))
        w2, h2 = chiplet_dims.get(n2, (0, 0))
        cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
        cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
        wirelength += float(edge.get("wireCount", 1)) * (abs(cx1 - cx2) + abs(cy1 - cy2))

    area = bbox_w * bbox_h if bbox_w and bbox_h else 0
    aspect_ratio = min(bbox_w, bbox_h) / max(bbox_w, bbox_h, 1e-9)

    placement_data = {
        "chiplets": chiplets_list,
        "connections": connections_list,
        "wirelength": _r3(wirelength),
        "area": _r3(area),
        "aspect_ratio": _r3(aspect_ratio),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(placement_data, f, indent=2, ensure_ascii=False)
    return placement_data


def generate_placement_json(
    result,
    post: dict,
    nodes: list,
    edge_map: dict,
    output_path: str,
    emib_bump_width_override: Optional[float] = None,
    emib_length_override: Optional[float] = None,
) -> dict:
    """
    从 ILP 求解结果提取布局数据，动态计算 EMIB_max_width，再计算硅桥左下角坐标，生成 placement JSON。

    流程：EMIB_max_width 动态求解（芯粒间实际相隔距离）→ 判定硅桥形态 → 算左下角坐标。
    输出数值保留小数点后 3 位。

    参数
    ----
    result : ILPPlacementResult
        求解结果（含 layout, rotations, bounding_box）
    post : dict
        run_emib_post_process 返回值（含 emib_placements，其中 emib_physical_dist 即 EMIB_max_width）
    nodes : list
        芯粒列表
    edge_map : dict
        (node1, node2) -> edge 信息（含 EMIB_bump_width, EMIB_length 等）
    output_path : str
        输出 JSON 文件路径
    emib_bump_width_override, emib_length_override : float | None
        可选，覆盖 edge 中的固定设计参数

    返回
    ----
    dict
        生成的 placement 数据结构
    """
    import json
    from pathlib import Path

    def _r3(v):
        return round(float(v or 0), 3)

    def _rects_overlap_xywh(
        ax: float,
        ay: float,
        aw: float,
        ah: float,
        bx: float,
        by: float,
        bw: float,
        bh: float,
    ) -> bool:
        # touching edges is allowed (no overlap)
        if ax + aw <= bx:
            return False
        if bx + bw <= ax:
            return False
        if ay + ah <= by:
            return False
        if by + bh <= ay:
            return False
        return True

    def _assert_no_overlap_chiplets(chiplets: list) -> None:
        rects = []
        for c in chiplets:
            rects.append(
                (
                    str(c.get("name")),
                    float(c.get("x-position", 0.0) or 0.0),
                    float(c.get("y-position", 0.0) or 0.0),
                    float(c.get("width", 0.0) or 0.0),
                    float(c.get("height", 0.0) or 0.0),
                )
            )
        rects.sort(key=lambda t: (t[0], t[1], t[2]))
        for i in range(len(rects)):
            ni, xi, yi, wi, hi = rects[i]
            for j in range(i + 1, len(rects)):
                nj, xj, yj, wj, hj = rects[j]
                if _rects_overlap_xywh(xi, yi, wi, hi, xj, yj, wj, hj):
                    raise RuntimeError(
                        f"Illegal placement: overlap detected between {ni} and {nj}: "
                        f"{ni}=(x={xi},y={yi},w={wi},h={hi}), {nj}=(x={xj},y={yj},w={wj},h={hj})"
                    )

    layout = result.layout if hasattr(result, "layout") else {}
    rotations = result.rotations if hasattr(result, "rotations") else {}
    bbox = getattr(result, "bounding_box", (0, 0))
    bbox_w, bbox_h = float(bbox[0]) if bbox else 0, float(bbox[1]) if bbox else 0

    chiplet_dims = {}
    min_w, min_h = float("inf"), float("inf")
    for node in nodes:
        w0 = float(node.dimensions.get("x", 0) or 0)
        h0 = float(node.dimensions.get("y", 0) or 0)
        min_w, min_h = min(min_w, w0), min(min_h, h0)
        rot = rotations.get(node.name, False)
        chiplet_dims[node.name] = (h0, w0) if rot else (w0, h0)

    chiplets_list = []
    for node in nodes:
        x, y = layout.get(node.name, (0, 0))
        w, h = chiplet_dims[node.name]
        rot = 1 if rotations.get(node.name, False) else 0
        power = float(getattr(node, "power", 0) or 0)
        chiplets_list.append({
            "name": node.name,
            "x-position": _r3(x),
            "y-position": _r3(y),
            "width": _r3(w),
            "height": _r3(h),
            "rotation": rot,
            "power": _r3(power),
        })

    _assert_no_overlap_chiplets(chiplets_list)

    connections_list = []
    for emp in post.get("emib_placements", []):
        n1, n2 = emp["node1"], emp["node2"]
        a, b = (n1, n2) if n1 <= n2 else (n2, n1)
        edge = edge_map.get((a, b), {})

        # 固定设计参数（可配置 override）
        emib_bump_width = emib_bump_width_override if emib_bump_width_override is not None else float(edge.get("EMIB_bump_width", 0) or 0)
        emib_length = emib_length_override if emib_length_override is not None else float(edge.get("EMIB_length", 0) or 0)

        # 步骤 1：动态计算 EMIB_max_width（芯粒间实际相隔距离，非左下角坐标差）
        # 来源于 compute_emib_placement：左右相邻 = j左-i右 或 i左-j右；上下相邻 = j下-i上 或 i下-j上
        emib_max_width = float(emp.get("emib_physical_dist", 0) or 0)
        if emib_max_width < 0:
            emib_max_width = 0.0

        emib_center = emp.get("emib_center", (0, 0))
        direction = emp.get("direction", "horizontal")

        # 步骤 2：判定硅桥形态（direction 已由 compute_emib_placement 判定）→ 计算左下角坐标
        x_bl, y_bl = compute_emib_bottom_left(
            emib_center=emib_center,
            direction=direction,
            emib_bump_width=emib_bump_width,
            emib_max_width=emib_max_width,
            emib_length=emib_length,
        )

        emib_rotation = 0 if direction == "horizontal" else 1
        emib_width = emib_max_width + 2 * emib_bump_width
        connections_list.append({
            "node1": n1,
            "node2": n2,
            "EMIBType": edge.get("EMIBType", "interfaceB"),
            "EMIB_length": _r3(emib_length),
            "EMIB_max_width": _r3(emib_max_width),
            "EMIB_width": _r3(emib_width),
            "EMIB_bump_width": _r3(emib_bump_width),
            "EMIB-x-position": _r3(x_bl),
            "EMIB-y-position": _r3(y_bl),
            "EMIB-rotation": emib_rotation,
        })

    wirelength = 0.0
    for (a, b), edge in edge_map.items():
        n1, n2 = edge.get("node1", a), edge.get("node2", b)
        x1, y1 = layout.get(n1, (0, 0))
        x2, y2 = layout.get(n2, (0, 0))
        w1, h1 = chiplet_dims.get(n1, (0, 0))
        w2, h2 = chiplet_dims.get(n2, (0, 0))
        cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
        cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
        wirelength += float(edge.get("wireCount", 1)) * (abs(cx1 - cx2) + abs(cy1 - cy2))

    area = bbox_w * bbox_h if bbox_w and bbox_h else 0
    aspect_ratio = min(bbox_w, bbox_h) / max(bbox_w, bbox_h, 1e-9)

    placement_data = {
        "chiplets": chiplets_list,
        "connections": connections_list,
        "wirelength": _r3(wirelength),
        "area": _r3(area),
        "aspect_ratio": _r3(aspect_ratio),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(placement_data, f, indent=2, ensure_ascii=False)
    return placement_data


# ---------------------------------------------------------------------------
# EMIB 布局可视化（硅桥绿色、互联线蓝色、无固定芯粒）
# ---------------------------------------------------------------------------


def build_emib_placements_from_ctx(ctx, nodes: List, edge_map: Dict) -> List[dict]:
    """
    从 ILP ctx 中提取硅桥位置，构建与 post["emib_placements"] 相同结构，供可视化使用。
    中心点由左下角坐标直接计算：center = (x_bl + w/2, y_bl + h/2)
    """
    emib_placements = []
    emib_connected = getattr(ctx, "EMIB_connected_pairs", None)
    if not emib_connected or not getattr(ctx, "EMIB_x_grid_var", None):
        return emib_placements
    for (i, j), emib_node in emib_connected.items():
        ex = float(ctx.EMIB_x_grid_var[(i, j)].X) if (i, j) in ctx.EMIB_x_grid_var else 0.0
        ey = float(ctx.EMIB_y_grid_var[(i, j)].X) if (i, j) in ctx.EMIB_y_grid_var else 0.0
        ew = float(ctx.EMIB_w_var[(i, j)].X) if (i, j) in ctx.EMIB_w_var else (getattr(emib_node, "width", 0) or 0)
        eh = float(ctx.EMIB_h_var[(i, j)].X) if (i, j) in ctx.EMIB_h_var else (getattr(emib_node, "height", 0) or 0)
        na = nodes[i].name if i < len(nodes) else str(i)
        nb = nodes[j].name if j < len(nodes) else str(j)
        er = bool(ctx.r_EMIB[(i, j)].X > 0.5) if (i, j) in ctx.r_EMIB else False
        a, b = (na, nb) if na <= nb else (nb, na)
        edge = edge_map.get((a, b), {})
        bump = float(edge.get("EMIB_bump_width", 0) or getattr(emib_node, "EMIB_bump_width", 0) or 0)
        emib_max_width = max(0.0, (ew if not er else eh) - 2 * bump)
        emib_len = float(edge.get("EMIB_length", 0) or getattr(emib_node, "EMIB_length", 0) or eh)
        cx, cy = ex + ew / 2, ey + eh / 2
        direction = "vertical" if er else "horizontal"
        # 水平：共享边沿 y，shared_len=eh；垂直：共享边沿 x，shared_len=ew
        shared_len = eh if not er else ew
        emib_placements.append({
            "emib_id": f"{na}-{nb}",
            "node1": na, "node2": nb,
            "direction": direction,
            "x_start": ex, "y_start": ey, "x_end": ex + ew, "y_end": ey + eh,
            "emib_physical_dist": emib_max_width,
            "shared_length": shared_len,
            "emib_center": (cx, cy),
            "ok": True, "warning": None,
        })
    return emib_placements


def extract_layout_data_for_vis(
    result,
    post: Optional[dict],
    nodes: List,
    edge_map: Dict,
    ctx=None,
) -> dict:
    """
    从 ILP 求解结果中提取布局数据，供可视化使用。
    优先用 post["emib_placements"]；若 post 为 None 且 ctx 含 EMIB 变量，则从 ctx 提取。
    
    参数
    ----
    result : ILPPlacementResult
        求解结果（含 layout, rotations）
    post : dict | None
        run_emib_post_process 返回值；可为 None
    nodes : List[ChipletNode]
        芯粒列表
    edge_map : Dict[Tuple[str,str], dict]
        边映射
    ctx : ILPModelContext | None
        可选，ILP 上下文；post 为 None 时从此提取 emib_placements
    
    返回
    ----
    dict
        chiplet_layout, chiplet_dims, emib_placements, emib_connections, chiplet_power
    """
    layout = result.layout if hasattr(result, "layout") else {}
    rotations = result.rotations if hasattr(result, "rotations") else {}
    chiplet_dims = {}
    chiplet_power = {}
    for node in nodes:
        w0 = float(node.dimensions.get("x", 0) or 0)
        h0 = float(node.dimensions.get("y", 0) or 0)
        rot = rotations.get(node.name, False)
        chiplet_dims[node.name] = (h0, w0) if rot else (w0, h0)
        chiplet_power[node.name] = float(getattr(node, "power", 0) or 0)
    emib_connections = []
    for (a, b), v in edge_map.items():
        if v.get("EMIBType") == "interfaceC":
            continue
        emib_connections.append({
            "node1": a, "node2": b,
            "wireCount": v.get("wireCount", 0),
            "EMIB_length": v.get("EMIB_length", 0),
            "EMIB_max_width": v.get("EMIB_max_width", 0),
        })
    if post is not None and post.get("emib_placements"):
        emib_placements = post["emib_placements"]
    elif ctx is not None and getattr(ctx, "EMIB_x_grid_var", None):
        emib_placements = build_emib_placements_from_ctx(ctx, nodes, edge_map)
    else:
        emib_placements = []
    return {
        "chiplet_layout": dict(layout),
        "chiplet_dims": chiplet_dims,
        "chiplet_power": chiplet_power,
        "emib_placements": emib_placements,
        "emib_connections": emib_connections,
    }


def compute_emib_rect_coords(
    emib_placement: dict,
    emib_length: float,
    emib_width: float,
) -> dict:
    """
    根据硅桥中心点及尺寸规则，计算以中心为基准的硅桥矩形坐标。
    水平相邻：长度=共享边高，宽度=max(相隔距离,1)；垂直相邻：长度=共享边宽，宽度=max(相隔距离,1)。
    若相隔距离为0则宽度取1以保证可见。
    """
    direction = emib_placement.get("direction", "horizontal")
    shared_len = emib_placement.get("shared_length") or emib_length
    phys = emib_placement.get("emib_physical_dist") or emib_width
    phys = max(phys, 1.0) if phys <= 1e-9 else phys
    cx, cy = emib_placement.get("emib_center", (0, 0))
    if direction == "horizontal":
        w_rect = phys
        h_rect = shared_len
        x_min = cx - w_rect / 2
        y_min = cy - h_rect / 2
    else:
        w_rect = shared_len
        h_rect = phys
        x_min = cx - w_rect / 2
        y_min = cy - h_rect / 2
    return {
        "x_min": x_min, "y_min": y_min, "width": w_rect, "height": h_rect,
        "emib_center": (cx, cy),
        "direction": direction,
    }


def generate_wire_grid_and_paths(
    emib_placement: dict,
    emib_rect: dict,
    chiplet_layout: Dict[str, Tuple[float, float]],
    chiplet_dims: Dict[str, Tuple[float, float]],
    wire_count: int,
    node1_name: str,
    node2_name: str,
    display_grid_size: Optional[int] = None,
) -> List[dict]:
    """
    在源/目标芯粒网格上生成布线起止点，路径：网格点 → 硅桥中心点 → 网格点。
    display_grid_size=16 为 256 点，=4 为 4x4=16 点（每 4x4 块的中心），均匀分配 wireCount。
    """
    wire_count = max(1, int(wire_count))
    emib_center = emib_placement.get("emib_center") or emib_rect.get("emib_center", (0, 0))
    src_grid = generate_chiplet_wire_grid_16x16(chiplet_layout, chiplet_dims, node1_name, display_size=display_grid_size)
    tgt_grid = generate_chiplet_wire_grid_16x16(chiplet_layout, chiplet_dims, node2_name, display_size=display_grid_size)
    n_grid = len(src_grid)
    results = []
    for idx in range(wire_count):
        i = idx % n_grid
        start_pt = src_grid[i]
        end_pt = tgt_grid[i]
        path_points = [start_pt, emib_center, end_pt]
        results.append({
            "wire_id": idx,
            "start": start_pt, "end": end_pt,
            "path_points": path_points,
        })
    return results


def draw_emib_layout_diagram(
    chiplet_layout: Dict[str, Tuple[float, float]],
    chiplet_dims: Dict[str, Tuple[float, float]],
    emib_placements: List[dict],
    emib_connections: List[dict],
    chiplet_power: Optional[Dict[str, float]] = None,
    title: str = "EMIB Chiplet Layout",
    show_axes: bool = True,
    save_path: Optional[str] = None,
    save_format: str = "png",
    figsize: Tuple[float, float] = (10, 8),
    wire_paths: Optional[Dict[str, List[dict]]] = None,
    show: bool = False,
    display_grid_size: Optional[int] = 4,
) -> dict:
    """
    绘制 EMIB 芯粒布局图：芯粒矩形、硅桥中心红色圆点、蓝色细线互联（路径：网格点→硅桥中心→网格点）。
    
    参数
    ----
    chiplet_layout : Dict[str, Tuple[float, float]]
        芯粒左下角坐标
    chiplet_dims : Dict[str, Tuple[float, float]]
        芯粒尺寸
    emib_placements : List[dict]
        硅桥位置信息
    emib_connections : List[dict]
        互联关系（含 wireCount, EMIB_length, EMIB_max_width）
    title : str
        图形标题
    show_axes : bool
        是否显示坐标轴
    save_path : str | None
        保存路径， None 则不保存
    save_format : str
        保存格式，如 "png", "svg"
    figsize : Tuple[float, float]
        图形尺寸
    wire_paths : Dict[str, List[dict]] | None
        emib_id -> generate_wire_grid_and_paths 返回的路径列表，若 None 则内部生成
    display_grid_size : int | None
        显示的蓝线网格规模，16 为 16x16=256 条，4 为 4x4=16 条（每 4x4 块中心），默认 4
    
    返回
    ----
    dict
        结构化数据: emib_coords, wire_start_end, emib_edge_centers
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=figsize)
    emib_rects_data = []
    all_wire_paths = wire_paths or {}
    for emp, conn in zip(emib_placements, emib_connections):
        emib_id = emp["emib_id"]
        emib_len = float(conn.get("EMIB_length", 0) or emp.get("shared_length", 0))
        emib_w = float(conn.get("EMIB_max_width", 0) or emp.get("emib_physical_dist", 0))
        rect_info = compute_emib_rect_coords(emp, emib_length=emib_len, emib_width=emib_w)
        emib_rects_data.append(rect_info)
        if emib_id not in all_wire_paths:
            all_wire_paths[emib_id] = generate_wire_grid_and_paths(
                emib_placement=emp,
                emib_rect=rect_info,
                chiplet_layout=chiplet_layout,
                chiplet_dims=chiplet_dims,
                wire_count=int(conn.get("wireCount", 1)),
                node1_name=conn["node1"],
                node2_name=conn["node2"],
                display_grid_size=display_grid_size,
            )

    chiplet_power = chiplet_power or {}
    # 1. 绘制芯粒矩形，标注 name 与 power
    for name, (x, y) in chiplet_layout.items():
        w, h = chiplet_dims.get(name, (0, 0))
        rect = Rectangle((x, y), w, h, facecolor="lightgray", edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        power_val = chiplet_power.get(name, 0)
        label = f"{name}\npower: {power_val:.0f}"
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, fontweight="bold")

    # 2. 绘制蓝色细线互联（路径：网格点→硅桥中心→网格点）
    for emib_id, paths in all_wire_paths.items():
        for p in paths:
            pts = p["path_points"]
            for i in range(len(pts) - 1):
                ax.plot(
                    [pts[i][0], pts[i + 1][0]],
                    [pts[i][1], pts[i + 1][1]],
                    color="blue",
                    linewidth=0.6,
                    alpha=0.8,
                )

    # 3. 绘制硅桥中心点（红色大圆点，绘于蓝线上方以便可见）
    for rect_info in emib_rects_data:
        cx, cy = rect_info.get("emib_center", (0, 0))
        ax.plot(cx, cy, "o", color="red", markersize=20, markeredgecolor="darkred", markeredgewidth=2, zorder=10)

    ax.set_aspect("equal")
    ax.set_title(title)
    if not show_axes:
        ax.set_axis_off()
    x_lo = min(x for x, _ in chiplet_layout.values()) if chiplet_layout else 0
    y_lo = min(y for _, y in chiplet_layout.values()) if chiplet_layout else 0
    x_hi = max(x + chiplet_dims.get(nm, (0, 0))[0] for nm, (x, _) in chiplet_layout.items()) if chiplet_layout else 10
    y_hi = max(y + chiplet_dims.get(nm, (0, 0))[1] for nm, (_, y) in chiplet_layout.items()) if chiplet_layout else 10
    for rect_info in emib_rects_data:
        cx, cy = rect_info.get("emib_center", (0, 0))
        x_lo = min(x_lo, cx)
        y_lo = min(y_lo, cy)
        x_hi = max(x_hi, cx)
        y_hi = max(y_hi, cy)
    margin = (max(x_hi - x_lo, y_hi - y_lo) or 1) * 0.1
    ax.set_xlim(x_lo - margin, x_hi + margin)
    ax.set_ylim(y_lo - margin, y_hi + margin)
    plt.tight_layout()
    if save_path:
        path = save_path if save_path.endswith(f".{save_format}") else f"{save_path}.{save_format}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    wire_start_end = {}
    for eid, pl in all_wire_paths.items():
        wire_start_end[eid] = [{"start": p["start"], "end": p["end"]} for p in pl]
    out = {
        "emib_coords": [
            {
                "x_min": r["x_min"], "y_min": r["y_min"],
                "width": r["width"], "height": r["height"],
                "emib_center": r.get("emib_center", (0, 0)),
            }
            for r in emib_rects_data
        ],
        "wire_start_end": wire_start_end,
        "emib_centers": [r.get("emib_center", (0, 0)) for r in emib_rects_data],
    }
    return out


def load_chiplet_nodes(max_nodes: Optional[int] = None) -> List[ChipletNode]:
    """
    Load chiplets from JSON and convert them into :class:`ChipletNode` objects.
    
    Parameters
    ----------
    max_nodes:
        如果指定，只返回前 max_nodes 个 chiplet。默认返回前4个。
    """

    raw = load_chiplets_json()
    table = build_chiplet_table(raw)

    nodes: List[ChipletNode] = []
    limit = max_nodes if max_nodes is not None else 4  # 默认只取前4个
    for row in table[:limit]:
        nodes.append(
            ChipletNode(
                name=row["name"],
                dimensions=row["dimensions"],
                phys=row["phys"],
                power=row["power"]
            )
        )
    return nodes


def generate_random_links(
    node_names: List[str],
    edge_prob: float = 0.2,
    allow_self_loop: bool = False,
    undirected: bool = True,
    fixed_num_edges: int = 10,
) -> List[Tuple[str, str]]:
    """
    生成固定数量的链接信息（每次调用都返回相同的链接）。
    
    使用固定的随机种子，确保对于相同的节点列表，每次生成的链接都是相同的。
    """

    import random

    # 设置固定的随机种子，确保每次生成相同的链接
    random.seed(42)
    
    # 生成所有可能的边对（排除自环）
    all_possible_edges: List[Tuple[str, str]] = []
    n = len(node_names)
    for i in range(n):
        for j in range(n):
            # 明确排除自环（自己链接自己）
            if i == j:
                continue
            if undirected and j <= i:
                # 对于无向图，只保留 i < j 的边
                continue

            all_possible_edges.append((node_names[i], node_names[j]))

    # 如果可能的边数少于固定数量，返回所有边
    if len(all_possible_edges) <= fixed_num_edges:
        # 再次确保没有自环（双重保险）
        edges = [(a, b) for a, b in all_possible_edges if a != b]
        random.seed()
        return edges

    # 随机选择固定数量的边
    edges = random.sample(all_possible_edges, fixed_num_edges)
    
    # 最终过滤：确保没有任何自环（双重保险）
    edges = [(a, b) for a, b in edges if a != b]
    
    # 如果过滤后边数不足，重新选择
    while len(edges) < fixed_num_edges and len(all_possible_edges) > len(edges):
        remaining = [e for e in all_possible_edges if e not in edges]
        if not remaining:
            break
        needed = fixed_num_edges - len(edges)
        additional = random.sample(remaining, min(needed, len(remaining)))
        edges.extend(additional)
        edges = [(a, b) for a, b in edges if a != b]  # 再次过滤
    
    # 重置随机种子，避免影响其他使用随机数的代码
    random.seed()

    return edges


def generate_typed_edges(
    node_names: List[str],
    num_silicon_bridge_edges: int = 5,
    num_normal_edges: int = 5,
    seed: Optional[int] = 42,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    生成两种类型的链接边：硅桥互联边和普通链接边。
    
    Parameters
    ----------
    node_names:
        节点名称列表
    num_silicon_bridge_edges:
        硅桥互联边的数量
    num_normal_edges:
        普通链接边的数量
    seed:
        随机种子，用于可重复生成
    
    Returns
    -------
    Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        返回两个列表：
        - 第一个列表：硅桥互联边，格式为 (src, dst, "silicon_bridge")
        - 第二个列表：普通链接边，格式为 (src, dst, "normal")
    """
    if seed is not None:
        random.seed(seed)
    
    # 生成所有可能的边对（排除自环）
    all_possible_edges: List[Tuple[str, str]] = []
    n = len(node_names)
    for i in range(n):
        for j in range(i + 1, n):  # 无向图，只保留 i < j 的边
            all_possible_edges.append((node_names[i], node_names[j]))
    
    # 确保有足够的边可以生成
    total_needed = num_silicon_bridge_edges + num_normal_edges
    if len(all_possible_edges) < total_needed:
        print(f"警告: 可能的边数({len(all_possible_edges)})少于需要的边数({total_needed})")
        # 调整数量
        num_silicon_bridge_edges = min(num_silicon_bridge_edges, len(all_possible_edges) // 2)
        num_normal_edges = min(num_normal_edges, len(all_possible_edges) - num_silicon_bridge_edges)
        total_needed = num_silicon_bridge_edges + num_normal_edges
    
    # 随机选择边
    selected_edges = random.sample(all_possible_edges, total_needed)
    
    # 分配边类型
    silicon_bridge_edges = [
        (src, dst, "silicon_bridge") 
        for src, dst in selected_edges[:num_silicon_bridge_edges]
    ]
    
    normal_edges = [
        (src, dst, "normal") 
        for src, dst in selected_edges[num_silicon_bridge_edges:]
    ]
    
    # 重置随机种子
    if seed is not None:
        random.seed()
    
    return silicon_bridge_edges, normal_edges


def build_random_chiplet_graph(
    edge_prob: float = 0.2,
    max_nodes: Optional[int] = None,
    fixed_num_edges: int = 4,
    num_silicon_bridge_edges: Optional[int] = None,
    num_normal_edges: Optional[int] = None,
    seed: Optional[int] = 42,
) -> Tuple[List[ChipletNode], List[Tuple[str, str]]]:
    """
    Convenience helper: load chiplets and generate a random connectivity graph.
    
    Parameters
    ----------
    edge_prob:
        边的概率（已废弃，现在使用 fixed_num_edges 或 num_silicon_bridge_edges/num_normal_edges）
    max_nodes:
        如果指定，只加载前 max_nodes 个 chiplet。默认只取前4个。
    fixed_num_edges:
        生成的固定边数（当未指定 num_silicon_bridge_edges 和 num_normal_edges 时使用）。默认4条。
    num_silicon_bridge_edges:
        硅桥互联边的数量（如果指定，将使用类型化边生成）
    num_normal_edges:
        普通链接边的数量（如果指定，将使用类型化边生成）
    seed:
        随机种子，用于可重复生成（仅在指定 num_silicon_bridge_edges 或 num_normal_edges 时使用）
    
    Returns
    -------
    Tuple[List[ChipletNode], List[Tuple[str, str]]]:
        返回节点列表和边列表（旧格式，向后兼容）
    """

    nodes = load_chiplet_nodes(max_nodes=max_nodes)
    names = [n.name for n in nodes]
    
    # 如果指定了硅桥互联边或普通互联边的数量，使用类型化边生成
    if num_silicon_bridge_edges is not None or num_normal_edges is not None:
        # 设置默认值
        if num_silicon_bridge_edges is None:
            num_silicon_bridge_edges = fixed_num_edges // 2 if fixed_num_edges > 0 else 0
        if num_normal_edges is None:
            num_normal_edges = fixed_num_edges - num_silicon_bridge_edges if fixed_num_edges > 0 else 0
        
        # 生成类型化边
        silicon_bridge_edges, normal_edges = generate_typed_edges(
            node_names=names,
            num_silicon_bridge_edges=num_silicon_bridge_edges,
            num_normal_edges=num_normal_edges,
            seed=seed
        )
        
        # 合并为旧格式（不带类型标签）
        edges = [(src, dst) for src, dst, _ in silicon_bridge_edges + normal_edges]
    else:
        # 使用旧的生成方法（向后兼容）
        edges = generate_random_links(names, edge_prob=edge_prob, fixed_num_edges=fixed_num_edges)
    
    return nodes, edges


# ---------------------------------------------------------------------------
# 绘图相关
# ---------------------------------------------------------------------------


def default_grid_layout(nodes: List[ChipletNode]) -> Dict[str, Tuple[float, float]]:
    """
    为每个 chiplet 决定一个在大画布上的偏移 (origin_x, origin_y)。

    - 每个 chiplet 内部仍然以左下角为 (0, 0) 局部坐标；
    - 返回一个 dict: name -> (origin_x, origin_y)。
    """

    if not nodes:
        return {}

    max_w = max(n.dimensions.get("x", 0) for n in nodes)
    max_h = max(n.dimensions.get("y", 0) for n in nodes)
    margin = max(max_w, max_h) * 0.3

    n = len(nodes)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    layout: Dict[str, Tuple[float, float]] = {}
    for idx, node in enumerate(nodes):
        r = idx // cols
        c = idx % cols
        origin_x = c * (max_w + margin)
        origin_y = r * (max_h + margin)
        layout[node.name] = (origin_x, origin_y)

    return layout


def draw_chiplet_diagram(
    nodes: List[ChipletNode],
    edges: List[Tuple[str, str]] | List[Tuple[str, str, str]],
    save_path: Optional[str] = None,
    layout: Optional[Dict[str, Tuple[float, float]]] = None,
    edge_types: Optional[Dict[Tuple[str, str], str]] = None,
    labels: Optional[Dict[str, str]] = None,  # 可选：自定义块内显示文本（name -> label）
    fixed_chiplet_names: Optional[set] = None,  # 固定的chiplet名称集合，这些chiplet将用粉红色绘制
    grid_size: float = 1.0,  # 网格大小，用于将网格坐标转换为实际坐标
    rotations: Optional[Dict[str, bool]] = None,  # 旋转信息：name -> 是否旋转
):
    """
    画出 chiplet 方框图。

    参数
    ----
    nodes:
        Chiplet 列表。
    edges:
        连接边列表，可以是：
        - 旧格式：形如 (src_name, dst_name)
        - 新格式：形如 (src_name, dst_name, edge_type)，其中 edge_type 为 "silicon_bridge" 或 "normal"
    save_path:
        若给定，则保存到该路径；否则直接 ``plt.show()``。
    layout:
        可选，自定义布局 dict: name -> (x_grid, y_grid)，其中坐标是网格坐标（需要乘以 grid_size 得到实际坐标）。
        如果为 None，则使用 :func:`default_grid_layout`。
    edge_types:
        可选的边类型映射，格式为 {(src, dst): "silicon_bridge" | "normal"}。
        如果 edges 是新格式或提供了此参数，将根据类型使用不同颜色：
        - 硅桥互联边：绿色
        - 普通链接边：灰色
    labels:
        可选的显示文本映射，格式为 {chiplet_name: "显示文本"}。
        若提供，将在 chiplet 块中心显示该文本；否则显示 chiplet name。
    fixed_chiplet_names:
        固定的chiplet名称集合。如果提供，这些chiplet将用粉红色绘制，其他chiplet用淡蓝色。
    grid_size:
        网格大小，用于将 layout 中的网格坐标转换为实际坐标。默认值为 1.0。
    rotations:
        旋转信息字典：name -> 是否旋转。如果提供，会根据旋转状态交换 chiplet 的长宽。
    """

    if not nodes:
        raise ValueError("No chiplet nodes to draw.")

    if layout is None:
        layout = default_grid_layout(nodes)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 记录每个 chiplet 用于连边的锚点坐标（统一使用中心点，与接口无关）
    anchor: Dict[str, Tuple[float, float]] = {}

    # 调试：检查layout和nodes
    # print(f"[DEBUG 绘图] nodes数量: {len(nodes)}, layout中的chiplet数量: {len(layout)}")
    missing_in_layout = [n.name for n in nodes if n.name not in layout]
    if missing_in_layout:
        print(f"[警告] 以下chiplet不在layout中，将跳过绘制: {missing_in_layout}")

    # 1) 画 chiplet 方框和 phys 锚点
    drawn_count = 0
    for node in nodes:
        if node.name not in layout:
            print(f"[警告] chiplet {node.name} 不在layout中，跳过绘制")
            continue

        # 获取网格坐标并转换为实际坐标
        x_grid, y_grid = layout[node.name]
        origin_x = float(x_grid) * grid_size
        origin_y = float(y_grid) * grid_size
        
        drawn_count += 1
        # print(f"[DEBUG 绘图] 绘制 {node.name}: 网格坐标=({x_grid:.2f}, {y_grid:.2f}), 实际坐标=({origin_x:.2f}, {origin_y:.2f})")
        
        # 获取原始尺寸
        orig_w = float(node.dimensions.get("x", 0.0))
        orig_h = float(node.dimensions.get("y", 0.0))
        
        # 检查是否旋转
        is_rotated = False
        if rotations is not None and node.name in rotations:
            is_rotated = rotations[node.name]
        
        # 如果旋转，交换长宽
        if is_rotated:
            w = orig_h
            h = orig_w
        else:
            w = orig_w
            h = orig_h

        # 判断是否为固定chiplet，固定chiplet使用粉红色，其他使用淡蓝色
        if fixed_chiplet_names is not None and node.name in fixed_chiplet_names:
            facecolor = "pink"  # 粉红色
        else:
            facecolor = "#cce6ff"  # 淡蓝色
        
        rect = Rectangle(
            (origin_x, origin_y),
            w,
            h,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(rect)

        # 在chiplet块中心写名字
        center_x = origin_x + w / 2.0
        center_y = origin_y + h / 2.0
        display_text = labels.get(node.name, node.name) if labels is not None else node.name
        ax.text(
            center_x,
            center_y,
            display_text,
            fontsize=10,
            ha="center",
            va="center",
            weight="bold",
            color="black",
        )

        # phys 点：红色小方块（仅用于显示，不用于连接）
        if node.phys:
            for p in node.phys:
                px = origin_x + float(p.get("x", 0.0))
                py = origin_y + float(p.get("y", 0.0))

                anchor_size = min(w, h) * 0.05
                ax.add_patch(
                    Rectangle(
                        (px - anchor_size / 2.0, py - anchor_size / 2.0),
                        anchor_size,
                        anchor_size,
                        facecolor="red",
                        edgecolor="none",
                    )
                )

        # 所有连接都从chiplet的中心位置出发（与接口无关）
        anchor[node.name] = (origin_x + w / 2.0, origin_y + h / 2.0)

    # 2) 画有向边（箭头）
    # 构建边类型映射
    edge_type_map: Dict[Tuple[str, str], str] = {}
    if edge_types:
        edge_type_map.update(edge_types)
    
    # 从edges中提取类型信息（如果是新格式）
    # edges 可能是 (src, dst, conn_type) 格式，其中 conn_type 是整数：1=silicon_bridge, 0=standard
    for edge in edges:
        if len(edge) == 3:
            src, dst, conn_type = edge
            # 将整数 conn_type 转换为字符串类型
            if isinstance(conn_type, int):
                if conn_type == 1:
                    edge_type_map[(src, dst)] = "silicon_bridge"
                    edge_type_map[(dst, src)] = "silicon_bridge"  # 双向
                else:
                    edge_type_map[(src, dst)] = "normal"
                    edge_type_map[(dst, src)] = "normal"  # 双向
            elif isinstance(conn_type, str):
                # 如果已经是字符串，直接使用
                edge_type_map[(src, dst)] = conn_type
                edge_type_map[(dst, src)] = conn_type  # 双向
        elif len(edge) == 2:
            src, dst = edge
            # 如果没有提供类型信息，默认为普通链接边
            if (src, dst) not in edge_type_map:
                edge_type_map[(src, dst)] = "normal"
                edge_type_map[(dst, src)] = "normal"  # 双向
    
    # 调试输出：打印边类型映射
    # print(f"[DEBUG 绘图] 边类型映射:")
    # for (src, dst), etype in edge_type_map.items():
    #     print(f"  ({src}, {dst}): {etype}")
    
    for edge in edges:
        # 处理不同格式的边
        if len(edge) == 3:
            src, dst, _ = edge
        elif len(edge) == 2:
            src, dst = edge
        else:
            continue
            
        if src not in anchor or dst not in anchor:
            continue
        sx, sy = anchor[src]
        dx, dy = anchor[dst]
        
        # 根据 EMIBType 选择颜色：interfaceC 灰色，interfaceB 绿色，其他（含 interfaceA）红色
        edge_type = edge_type_map.get((src, dst), "normal")
        print(f"[DEBUG 绘图] 绘制边 ({src}, {dst}): 类型={edge_type}")
        if edge_type == "interfaceC":
            edge_color = "gray"
            linewidth = 1.0
        elif edge_type == "interfaceB":
            edge_color = "green"
            linewidth = 3.0
        else:
            edge_color = "red"  # interfaceA 及其他类型
            linewidth = 3.0

        arrow = FancyArrowPatch(
            (sx, sy),
            (dx, dy),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=linewidth,
            color=edge_color,
            alpha=0.8,
        )
        ax.add_patch(arrow)

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    
    # 调试输出
    print(f"[DEBUG 绘图] 实际绘制了 {drawn_count} 个chiplet（共 {len(nodes)} 个）")

    # 调整视图范围（考虑所有模块的完整范围，包括宽度和高度）
    all_x_min = []
    all_x_max = []
    all_y_min = []
    all_y_max = []
    
    for node in nodes:
        if node.name not in layout:
            continue
        x_grid, y_grid = layout[node.name]
        ox = float(x_grid) * grid_size
        oy = float(y_grid) * grid_size
        
        # 获取原始尺寸
        orig_w = float(node.dimensions.get("x", 0.0))
        orig_h = float(node.dimensions.get("y", 0.0))
        
        # 检查是否旋转
        is_rotated = False
        if rotations is not None and node.name in rotations:
            is_rotated = rotations[node.name]
        
        # 如果旋转，交换长宽
        if is_rotated:
            w = orig_h
            h = orig_w
        else:
            w = orig_w
            h = orig_h
        
        # 记录左下角和右上角坐标
        all_x_min.append(ox)
        all_x_max.append(ox + w)
        all_y_min.append(oy)
        all_y_max.append(oy + h)
    
    if all_x_min and all_y_min:
        # 计算所有模块的最小和最大坐标
        x_min = min(all_x_min)
        x_max = max(all_x_max)
        y_min = min(all_y_min)
        y_max = max(all_y_max)
        
        # 添加边距（10%的额外空间）
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin_x = max(x_range * 0.1, 1.0)  # 至少1.0的边距
        margin_y = max(y_range * 0.1, 1.0)
        
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
 


if __name__ == "__main__":
    # 简单测试：随机生成连接并画默认布局
    nodes, edges = build_random_chiplet_graph(edge_prob=0.3)
    # 使用相对路径，输出到项目根目录
    from pathlib import Path
    out_path = Path(__file__).parent.parent / "output" / "chiplet_diagram_from_tool.png"
    draw_chiplet_diagram(nodes, edges, save_path=str(out_path))
    print(f"Diagram saved to: {out_path}")
    
    # 测试生成两种类型的边
    print("\n=== 测试生成两种类型的链接边 ===")
    node_names = [node.name for node in nodes]
    silicon_bridge_edges, normal_edges = generate_typed_edges(
        node_names=node_names,
        num_silicon_bridge_edges=5,
        num_normal_edges=5,
        seed=42
    )
    
    print(f"\n硅桥互联边 ({len(silicon_bridge_edges)} 条):")
    for src, dst, edge_type in silicon_bridge_edges:
        print(f"  {src} <-> {dst} (类型: {edge_type})")
    
    print(f"\n普通链接边 ({len(normal_edges)} 条):")
    for src, dst, edge_type in normal_edges:
        print(f"  {src} <-> {dst} (类型: {edge_type})")


# ---------------------------------------------------------------------------
# 约束打印功能（用于调试ILP约束）
# ---------------------------------------------------------------------------

if pulp is not None:
    # 约束方向映射表
    SENSE_MAP = {
        pulp.LpConstraintLE: "<=",
        pulp.LpConstraintGE: ">=",
        pulp.LpConstraintEQ: "=",
    }

    def print_constraint_formal(constraint) -> None:
        """
        打印约束的形式化数学表达。
        
        参数:
            constraint: Pulp约束对象或Gurobi约束对象
        """
        # 检查是否是Gurobi约束
        if gp is not None and isinstance(constraint, gp.Constr):
            # Gurobi约束
            try:
                constraint_name = constraint.ConstrName
            except (AttributeError, Exception):
                # 如果约束还没有名称（例如刚添加但模型未更新），跳过打印
                return
            # Gurobi约束的字符串表示
            constraint_str = str(constraint)
            # 打印约束（可以修改为输出到日志文件）
            # print(f"[ADD CONSTRAINT] {constraint_name}: {constraint_str}")
            return
        
        # PuLP约束
        if pulp is not None and isinstance(constraint, pulp.LpConstraint):
            # 处理左侧表达式：移除冗余的 *1.0，美化输出
            lhs = str(constraint.expr).replace("*1.0", "").replace(" + ", " + ").strip()
            
            # 处理右侧常数：Pulp内部存储为 expr + constant <= 0，所以需要取负号
            rhs = round(-constraint.constant, 4)
            
            # 获取约束方向字符串
            sense_str = SENSE_MAP.get(constraint.sense, "?")
            
            # 构建形式化表达式
            formal_expr = f"[{constraint.name}] {lhs} {sense_str} {rhs}"
            
            # 打印约束（可以修改为输出到日志文件）
            # print(f"[ADD CONSTRAINT] {formal_expr}")
            return
        
        # 如果都不匹配，静默忽略
        pass
else:
    # 如果pulp未安装，提供占位函数
    def print_constraint_formal(*args, **kwargs):
        # 检查是否是Gurobi约束
        if gp is not None and len(args) > 0:
            constraint = args[0]
            if isinstance(constraint, gp.Constr):
                # Gurobi约束，静默处理
                return
        # 其他情况抛出错误
        raise ImportError("pulp库未安装，无法使用约束打印功能")


# ---------------------------------------------------------------------------
# ILP求解结果打印函数
# ---------------------------------------------------------------------------

def print_pair_distances_only(
    ctx,
    result,
    solution_idx: int,
    prev_pair_distances_list: Optional[List[Dict[Tuple[int, int], float]]] = None,
    min_pair_dist_diff: float = 1.0,
) -> None:
    """
    简化输出：只打印每对chiplet的相对距离，以及当前解与之前解的距离比较。
    
    参数:
        ctx: ILP模型上下文
        result: 求解结果
        solution_idx: 解的索引（从0开始）
        prev_pair_distances_list: 可选，之前所有解的chiplet对距离列表
        min_pair_dist_diff: 判断距离是否相同的最小差异阈值
    """
    if result.status != "Optimal":
        return
    
    nodes = ctx.nodes
    n = len(nodes)
    
    # 获取当前解的坐标
    x_curr = {}
    y_curr = {}
    for k in range(n):
        x_val = get_var_value(ctx.x_grid_var[k])
        y_val = get_var_value(ctx.y_grid_var[k])
        if x_val is not None and y_val is not None:
            x_curr[k] = float(x_val)
            y_curr[k] = float(y_val)
        else:
            return  # 如果无法获取坐标，直接返回
    
    # 计算当前解的每对chiplet的相对距离（x轴和y轴的绝对值差）
    curr_pair_distances = {}
    chiplet_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for i, j in chiplet_pairs:
        if i in x_curr and j in x_curr and i in y_curr and j in y_curr:
            # 计算x轴和y轴的相对距离（绝对值差）
            x_dist = abs(x_curr[i] - x_curr[j])
            y_dist = abs(y_curr[i] - y_curr[j])
            curr_pair_distances[(i, j)] = (x_dist, y_dist)
    
    # 输出当前解的相对距离
    # print(f"\n=== 解 {solution_idx + 1} ===")
    # print(f"\n每对chiplet的相对距离（|x[i]-x[j]|, |y[i]-y[j]|）:")
    # for i, j in sorted(chiplet_pairs):
    #     if (i, j) in curr_pair_distances:
    #         x_dist, y_dist = curr_pair_distances[(i, j)]
    #         name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
    #         name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
    #         # 计算曼哈顿距离（用于显示）
    #         manhattan_dist = x_dist + y_dist
    #         print(f"  ({i},{j}) [{name_i}, {name_j}]: x距离={x_dist:.3f}, y距离={y_dist:.3f}, 曼哈顿距离={manhattan_dist:.3f}")
    
    # 与之前解比较
    if prev_pair_distances_list and len(prev_pair_distances_list) > 0:
        # 获取grid_size，用于将grid坐标距离转换为实际坐标距离
        grid_size_val = ctx.grid_size if hasattr(ctx, 'grid_size') and ctx.grid_size is not None else 1.0
        
        print(f"\n与之前解的距离比较（阈值={min_pair_dist_diff:.3f}）:")
        for prev_idx, prev_distances in enumerate(prev_pair_distances_list):
            print(f"\n  与解 {prev_idx + 1} 比较:")
            same_pairs = []
            diff_pairs_with_info = []  # 存储不同对及其距离差信息
            
            for i, j in sorted(chiplet_pairs):
                if (i, j) in curr_pair_distances and (i, j) in prev_distances:
                    curr_x_dist, curr_y_dist = curr_pair_distances[(i, j)]
                    # prev_distances中存储的是grid坐标的曼哈顿距离，需要转换为实际坐标距离
                    prev_dist_grid = prev_distances[(i, j)]  # grid坐标的曼哈顿距离
                    prev_dist = prev_dist_grid * grid_size_val  # 转换为实际坐标距离
                    
                    # 计算当前解的曼哈顿距离（实际坐标）
                    curr_dist = curr_x_dist + curr_y_dist
                    
                    # 计算距离差（绝对值）
                    dist_diff = abs(curr_dist - prev_dist)
                    
                    if dist_diff < min_pair_dist_diff:
                        same_pairs.append((i, j))
                    else:
                        # 只有当距离差 >= min_pair_dist_diff 时，才认为不同
                        # 在else分支中，dist_diff >= min_pair_dist_diff 总是成立
                        diff_pairs_with_info.append((i, j, curr_dist, prev_dist, dist_diff))
            
            if same_pairs:
                print(f"    相同的chiplet对（距离差 < 阈值 {min_pair_dist_diff:.3f}）:")
                for i, j in same_pairs:
                    if (i, j) in curr_pair_distances and (i, j) in prev_distances:
                        curr_x_dist, curr_y_dist = curr_pair_distances[(i, j)]
                        curr_dist = curr_x_dist + curr_y_dist
                        prev_dist_grid = prev_distances[(i, j)]  # grid坐标距离
                        prev_dist = prev_dist_grid * grid_size_val  # 转换为实际坐标距离
                        dist_diff = abs(curr_dist - prev_dist)
                        name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
                        name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
                        print(f"      ({i},{j}) [{name_i}, {name_j}]: 当前距离={curr_dist:.3f}, 之前距离={prev_dist:.3f}, 距离差={dist_diff:.3f} (< {min_pair_dist_diff:.3f})")
            if diff_pairs_with_info:
                print(f"    不同的chiplet对（距离差 >= 阈值 {min_pair_dist_diff:.3f}）:")
                for i, j, curr_dist, prev_dist, dist_diff in diff_pairs_with_info:
                    name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
                    name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
                    print(f"      ({i},{j}) [{name_i}, {name_j}]: 当前距离={curr_dist:.3f}, 之前距离={prev_dist:.3f}, 距离差={dist_diff:.3f} (>= {min_pair_dist_diff:.3f}, 满足阈值)")
            if not same_pairs and not diff_pairs_with_info:
                print(f"    (无数据)")
    else:
        print(f"\n(第一个解，无历史解可比较)")


def print_all_variables(
    ctx: ILPModelContext, 
    result: ILPPlacementResult,
    prev_pair_distances_list: Optional[List[Dict[Tuple[int, int], float]]] = None
) -> None:
    """
    打印所有变量的值，包括排除约束相关的变量。
    
    参数:
        ctx: ILP模型上下文
        result: 求解结果
        prev_pair_distances_list: 可选，之前所有解的chiplet对距离列表，用于显示对比信息
    """
    if result.status != "Optimal":
        return
    
    nodes = ctx.nodes
    n = len(nodes)
    
    print("\n" + "=" * 80)
    print("变量值详情")
    print("=" * 80)
    
    # 1. 坐标变量 (x, y)
    print("\n【坐标变量】")
    for k in range(n):
        x_val = get_var_value(ctx.x[k])
        y_val = get_var_value(ctx.y[k])
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  x[{k}] ({node_name}): {x_val}")
        print(f"  y[{k}] ({node_name}): {y_val}")
    
    # 2. 网格坐标变量 (x_grid, y_grid)
    print("\n【网格坐标变量】")
    for k in range(n):
        # 兼容 PuLP 和 Gurobi 的变量获取方式
        if hasattr(ctx.prob, 'variablesDict'):
            # PuLP
            x_grid_var = ctx.prob.variablesDict().get(f"x_grid_{k}")
            y_grid_var = ctx.prob.variablesDict().get(f"y_grid_{k}")
        elif hasattr(ctx.prob, 'getVarByName'):
            # Gurobi
            x_grid_var = ctx.prob.getVarByName(f"x_grid_{k}")
            y_grid_var = ctx.prob.getVarByName(f"y_grid_{k}")
        else:
            x_grid_var = None
            y_grid_var = None
        x_grid_val = get_var_value(x_grid_var)
        y_grid_val = get_var_value(y_grid_var)
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  x_grid[{k}] ({node_name}): {x_grid_val}")
        print(f"  y_grid[{k}] ({node_name}): {y_grid_val}")
    
    # 3. 旋转变量 (r)
    print("\n【旋转变量】")
    for k in range(n):
        r_val = get_var_value(ctx.r[k])
        rotated_str = "是" if (r_val is not None and r_val > 0.5) else "否"
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  r[{k}] ({node_name}): {r_val} (旋转: {rotated_str})")
    
    # 4. 宽度和高度变量 (w, h)
    print("\n【尺寸变量】")
    for k in range(n):
        # 兼容 PuLP 和 Gurobi 的变量获取方式
        if hasattr(ctx.prob, 'variablesDict'):
            # PuLP
            w_var = ctx.prob.variablesDict().get(f"w_{k}")
            h_var = ctx.prob.variablesDict().get(f"h_{k}")
        elif hasattr(ctx.prob, 'getVarByName'):
            # Gurobi
            w_var = ctx.prob.getVarByName(f"w_{k}")
            h_var = ctx.prob.getVarByName(f"h_{k}")
        else:
            w_var = None
            h_var = None
        w_val = get_var_value(w_var)
        h_val = get_var_value(h_var)
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  w[{k}] ({node_name}): {w_val}")
        print(f"  h[{k}] ({node_name}): {h_val}")
    
    # 5. 中心坐标变量 (cx, cy)
    if hasattr(ctx, 'cx') and ctx.cx is not None:
        print("\n【中心坐标变量】")
        for k in range(n):
            cx_val = get_var_value(ctx.cx[k])
            cy_val = get_var_value(ctx.cy[k])
            node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
            print(f"  cx[{k}] ({node_name}): {cx_val}")
            print(f"  cy[{k}] ({node_name}): {cy_val}")
    
    # 6. 相邻方式变量 (z1, z2, z1L, z1R, z2D, z2U)
    connected_pairs = getattr(ctx, 'all_connected_pairs', []) or []

    if len(connected_pairs) > 0:
        print("\n【相邻方式变量】")
        for i, j in connected_pairs:
            name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
            name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
            z1_val = get_var_value(ctx.z1.get((i, j))) if (i, j) in ctx.z1 else None
            z2_val = get_var_value(ctx.z2.get((i, j))) if (i, j) in ctx.z2 else None
            z1L_val = get_var_value(ctx.z1L.get((i, j))) if (i, j) in ctx.z1L else None
            z1R_val = get_var_value(ctx.z1R.get((i, j))) if (i, j) in ctx.z1R else None
            z2D_val = get_var_value(ctx.z2D.get((i, j))) if (i, j) in ctx.z2D else None
            z2U_val = get_var_value(ctx.z2U.get((i, j))) if (i, j) in ctx.z2U else None
            print(f"  模块对 ({name_i}, {name_j}):")
            print(f"    z1[{i},{j}] (水平相邻): {z1_val}")
            print(f"    z2[{i},{j}] (垂直相邻): {z2_val}")
            if z1_val is not None and z1_val > 0.5:
                print(f"      z1R[{i},{j}] (i在右): {z1R_val}")
            if z2_val is not None and z2_val > 0.5:
                print(f"      z2D[{i},{j}] (i在下): {z2D_val}")
                print(f"      z2U[{i},{j}] (i在上): {z2U_val}")
    
    # 7. 非重叠约束变量 (p_left, p_right, p_down, p_up)
    print("\n【非重叠约束变量】")
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))
    
    for i, j in all_pairs:
        name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
        name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
        # 兼容 PuLP 和 Gurobi 的变量获取方式
        if hasattr(ctx.prob, 'variablesDict'):
            # PuLP
            p_left_var = ctx.prob.variablesDict().get(f"p_left_{i}_{j}")
            p_right_var = ctx.prob.variablesDict().get(f"p_right_{i}_{j}")
            p_down_var = ctx.prob.variablesDict().get(f"p_down_{i}_{j}")
            p_up_var = ctx.prob.variablesDict().get(f"p_up_{i}_{j}")
        elif hasattr(ctx.prob, 'getVarByName'):
            # Gurobi
            p_left_var = ctx.prob.getVarByName(f"p_left_{i}_{j}")
            p_right_var = ctx.prob.getVarByName(f"p_right_{i}_{j}")
            p_down_var = ctx.prob.getVarByName(f"p_down_{i}_{j}")
            p_up_var = ctx.prob.getVarByName(f"p_up_{i}_{j}")
        else:
            p_left_var = p_right_var = p_down_var = p_up_var = None
        
        p_left_val = get_var_value(p_left_var)
        p_right_val = get_var_value(p_right_var)
        p_down_val = get_var_value(p_down_var)
        p_up_val = get_var_value(p_up_var)
        
        print(f"  模块对 ({name_i}, {name_j}):")
        print(f"    p_left[{i},{j}]: {p_left_val}")
        print(f"    p_right[{i},{j}]: {p_right_val}")
        print(f"    p_down[{i},{j}]: {p_down_val}")
        print(f"    p_up[{i},{j}]: {p_up_val}")
    
    # 8. 边界框变量
    print("\n【边界框变量】")
    bbox_w_val = get_var_value(ctx.bbox_w)
    bbox_h_val = get_var_value(ctx.bbox_h)
    print(f"  bbox_w: {bbox_w_val}")
    print(f"  bbox_h: {bbox_h_val}")
    
    # 9. 其他辅助变量（shared_x, shared_y, dx_abs, dy_abs, bbox_min/max等）
    print("\n【其他辅助变量】")
    other_vars = []
    # 兼容 PuLP 和 Gurobi 的变量获取方式
    if hasattr(ctx.prob, 'variablesDict'):
        # PuLP
        var_dict = ctx.prob.variablesDict()
    elif hasattr(ctx.prob, 'getVars'):
        # Gurobi
        var_dict = {var.VarName: var for var in ctx.prob.getVars()}
    else:
        var_dict = {}
    
    for var_name, var in var_dict.items():
        if var_name.startswith("shared_") or var_name.startswith("dx_abs_") or \
           var_name.startswith("dy_abs_") or var_name.startswith("bbox_") or \
           var_name.startswith("bbox_area_proxy"):
            # 排除排除约束相关的变量（这些会在后面单独打印）
            if not (var_name.startswith("dx_abs_pair_") or var_name.startswith("dy_abs_pair_") or \
                    var_name.startswith("dx_grid_abs_pair_") or var_name.startswith("dy_grid_abs_pair_")):
                val = get_var_value(var)
                if val is not None:
                    other_vars.append((var_name, val))
    
    if other_vars:
        for var_name, val in sorted(other_vars):
            print(f"  {var_name}: {val}")
    else:
        print("  (无)")
    
    # 10. 排除解约束相关变量和约束（仅在第二次及以后的求解中打印）
    exclude_vars = []
    # 收集所有排除解约束相关的变量，包括所有可能的变量名模式
    # 兼容 PuLP 和 Gurobi 的变量获取方式
    if hasattr(ctx.prob, 'variablesDict'):
        # PuLP
        var_dict = ctx.prob.variablesDict()
    elif hasattr(ctx.prob, 'getVars'):
        # Gurobi
        var_dict = {var.VarName: var for var in ctx.prob.getVars()}
    else:
        var_dict = {}
    
    for var_name, var in var_dict.items():
        # 检查是否是排除解约束相关的变量
        is_exclude_var = (
            var_name.startswith("dx_abs_pair_") or 
            var_name.startswith("dy_abs_pair_") or 
            var_name.startswith("dx_grid_abs_pair_") or 
            var_name.startswith("dy_grid_abs_pair_") or 
            var_name.startswith("dist_curr_pair_") or 
            var_name.startswith("dist_diff_pair_") or 
            var_name.startswith("dist_diff_abs_pair_") or 
            var_name.startswith("diff_dist_pair_") or 
            var_name.startswith("same_dist_pair_")
        )
        if is_exclude_var:
            val = get_var_value(var)
            # 即使值为None也记录，以便调试
            exclude_vars.append((var_name, val))
    
    if exclude_vars:
        print("\n" + "=" * 80)
        print("排除解约束相关变量和约束")
        print("=" * 80)
        
        # 10.1 打印排除约束相关的变量
        print("\n【排除约束变量】")
        
        # 按变量类型分组
        dx_abs_pair_vars = []
        dy_abs_pair_vars = []
        dx_grid_abs_pair_vars = []
        dy_grid_abs_pair_vars = []
        dist_curr_pair_vars = []
        dist_diff_pair_vars = []
        dist_diff_abs_pair_vars = []
        diff_dist_pair_vars = []
        same_dist_pair_vars = []
        
        for var_name, val in exclude_vars:
            if var_name.startswith("dx_grid_abs_pair_"):
                dx_grid_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dy_grid_abs_pair_"):
                dy_grid_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dist_curr_pair_"):
                dist_curr_pair_vars.append((var_name, val))
            elif var_name.startswith("dx_abs_pair_"):
                dx_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dy_abs_pair_"):
                dy_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dist_diff_pair_") and not var_name.startswith("dist_diff_abs_pair_"):
                dist_diff_pair_vars.append((var_name, val))
            elif var_name.startswith("dist_diff_abs_pair_"):
                dist_diff_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("diff_dist_pair_"):
                diff_dist_pair_vars.append((var_name, val))
            elif var_name.startswith("same_dist_pair_"):
                same_dist_pair_vars.append((var_name, val))
        
        if dx_grid_abs_pair_vars:
            print("\n  dx_grid_abs_pair (chiplet对的x方向grid坐标距离绝对值):")
            for var_name, val in sorted(dx_grid_abs_pair_vars):
                print(f"    {var_name}: {val}")
        
        if dy_grid_abs_pair_vars:
            print("\n  dy_grid_abs_pair (chiplet对的y方向grid坐标距离绝对值):")
            for var_name, val in sorted(dy_grid_abs_pair_vars):
                print(f"    {var_name}: {val}")
        
        # 按chiplet对组织显示，使输出更清晰
        import re
        pair_info = {}  # key: (i, j), value: dict with all related vars
        
        # 解析所有变量，按chiplet对分组
        unmatched_vars = []  # 记录无法匹配的变量
        for var_name, val in exclude_vars:
            # 匹配模式：{prefix}_{suffix}_{i}_{j} 或 {prefix}_{suffix}_{i}_{j}_prev{prev_idx}
            # 注意：变量名可能是 dist_diff_abs_pair_{suffix}_{i}_{j}_prev{prev_idx}
            match = re.search(r'([^_]+(?:_[^_]+)*)_[^_]+_(\d+)_(\d+)(?:_prev(\d+))?', var_name)
            if match:
                prefix = match.group(1)
                i_val = int(match.group(2))
                j_val = int(match.group(3))
                prev_idx = match.group(4)
                pair_key = (i_val, j_val)
                
                if pair_key not in pair_info:
                    pair_info[pair_key] = {
                        'dx_grid_abs': None,
                        'dy_grid_abs': None,
                        'dist_curr': None,
                        'dist_diff': {},
                        'dist_diff_abs': {},
                        'diff_dist': None,
                        'same_dist': {}
                    }
                
                # 处理各种变量前缀
                if prefix == 'dx_grid_abs_pair':
                    pair_info[pair_key]['dx_grid_abs'] = val
                elif prefix == 'dy_grid_abs_pair':
                    pair_info[pair_key]['dy_grid_abs'] = val
                elif prefix == 'dist_curr_pair':
                    pair_info[pair_key]['dist_curr'] = val
                elif prefix == 'dist_diff_pair' and prev_idx:
                    pair_info[pair_key]['dist_diff'][int(prev_idx)] = val
                elif prefix == 'dist_diff_abs_pair' and prev_idx:
                    pair_info[pair_key]['dist_diff_abs'][int(prev_idx)] = val
                elif prefix == 'diff_dist_pair':
                    pair_info[pair_key]['diff_dist'] = val
                elif prefix == 'same_dist_pair' and prev_idx:
                    pair_info[pair_key]['same_dist'][int(prev_idx)] = val
                else:
                    # 无法匹配的变量，记录到unmatched_vars
                    unmatched_vars.append((var_name, val))
            else:
                # 无法解析的变量，记录到unmatched_vars
                unmatched_vars.append((var_name, val))
        
        # 按chiplet对显示详细信息
        if pair_info:
            print("\n  【按chiplet对分组显示】")
            for (i, j) in sorted(pair_info.keys()):
                info = pair_info[(i, j)]
                name_i = nodes[i].name if hasattr(nodes[i], 'name') and i < len(nodes) else f"Chiplet_{i}"
                name_j = nodes[j].name if hasattr(nodes[j], 'name') and j < len(nodes) else f"Chiplet_{j}"
                
                print(f"\n    模块对 ({name_i}, {name_j}) [索引: ({i}, {j})]:")
                
                if info['dx_grid_abs'] is not None:
                    print(f"      dx_grid_abs (x方向grid距离): {info['dx_grid_abs']:.2f}")
                if info['dy_grid_abs'] is not None:
                    print(f"      dy_grid_abs (y方向grid距离): {info['dy_grid_abs']:.2f}")
                if info['dist_curr'] is not None:
                    print(f"      dist_curr (当前距离，grid单位): {info['dist_curr']:.2f}")
                    print(f"        验证: dx_grid_abs + dy_grid_abs = {info['dx_grid_abs']:.2f} + {info['dy_grid_abs']:.2f} = {info['dx_grid_abs'] + info['dy_grid_abs']:.2f}")
                
                if info['dist_diff'] or info['dist_diff_abs']:
                    print(f"      与之前解的距离比较:")
                    for prev_idx in sorted(set(list(info['dist_diff'].keys()) + list(info['dist_diff_abs'].keys()))):
                        dist_diff = info['dist_diff'].get(prev_idx, None)
                        dist_diff_abs = info['dist_diff_abs'].get(prev_idx, None)
                        same_dist = info['same_dist'].get(prev_idx, None)
                        
                        # 显示之前解的距离（如果可用）
                        prev_dist = None
                        if prev_pair_distances_list and prev_idx < len(prev_pair_distances_list):
                            prev_dist = prev_pair_distances_list[prev_idx].get((i, j), None)
                        
                        print(f"        解 {prev_idx}:")
                        if prev_dist is not None:
                            print(f"          之前解的距离: {prev_dist:.2f} (grid单位)")
                        if info['dist_curr'] is not None:
                            print(f"          当前解的距离: {info['dist_curr']:.2f} (grid单位)")
                        if dist_diff is not None:
                            print(f"          距离差 (dist_diff): {dist_diff:.2f}")
                            if prev_dist is not None and info['dist_curr'] is not None:
                                print(f"            验证: {info['dist_curr']:.2f} - {prev_dist:.2f} = {dist_diff:.2f}")
                        if dist_diff_abs is not None:
                            print(f"          距离差绝对值 (dist_diff_abs): {dist_diff_abs:.2f}")
                        if same_dist is not None:
                            same_str = "是" if same_dist > 0.5 else "否"
                            print(f"          是否相同 (same_dist_pair): {same_dist} ({same_str})")
                            if dist_diff_abs is not None:
                                if same_dist > 0.5:
                                    print(f"            → 距离差 {dist_diff_abs:.2f} < 阈值，标记为相同")
                                else:
                                    print(f"            → 距离差 {dist_diff_abs:.2f} >= 阈值，标记为不同")
                
                if info['diff_dist'] is not None:
                    diff_str = "是" if info['diff_dist'] > 0.5 else "否"
                    print(f"      diff_dist_pair (与所有之前解都不同): {info['diff_dist']} ({diff_str})")
                    if info['diff_dist'] > 0.5:
                        print(f"        → 该chiplet对的距离与所有之前解都不同，满足排除约束")
        
        # 保留原有的详细变量列表输出（作为补充）
        if dist_curr_pair_vars:
            print("\n  【详细变量列表 - dist_curr_pair】")
            for var_name, val in sorted(dist_curr_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dx_abs_pair_vars:
            print("\n  【详细变量列表 - dx_abs_pair (旧版本)】")
            for var_name, val in sorted(dx_abs_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dy_abs_pair_vars:
            print("\n  【详细变量列表 - dy_abs_pair (旧版本)】")
            for var_name, val in sorted(dy_abs_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dist_diff_pair_vars:
            print("\n  【详细变量列表 - dist_diff_pair】")
            for var_name, val in sorted(dist_diff_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dist_diff_abs_pair_vars:
            print("\n  【详细变量列表 - dist_diff_abs_pair】")
            for var_name, val in sorted(dist_diff_abs_pair_vars):
                if val is not None:
                    print(f"    {var_name}: {val:.2f}")
                else:
                    print(f"    {var_name}: None (未求解)")
        
        if diff_dist_pair_vars:
            print("\n  【详细变量列表 - diff_dist_pair (二进制)】")
            for var_name, val in sorted(diff_dist_pair_vars):
                if val is not None:
                    diff_str = "是" if val > 0.5 else "否"
                    print(f"    {var_name}: {val} ({diff_str})")
                else:
                    print(f"    {var_name}: None (未求解)")
        
        # 打印所有其他排除解约束相关的变量（包括无法匹配的）
        if unmatched_vars:
            print("\n  【其他排除解约束相关变量（未在分组中显示）】")
            for var_name, val in sorted(unmatched_vars):
                if val is not None:
                    print(f"    {var_name}: {val}")
                else:
                    print(f"    {var_name}: None (未求解)")
        
        # 打印所有排除解约束相关变量的完整列表（用于调试）
        print("\n  【完整变量列表（所有排除解约束相关变量）】")
        for var_name, val in sorted(exclude_vars):
            if val is not None:
                # 根据变量类型格式化输出
                if var_name.startswith("diff_dist_pair_") or var_name.startswith("same_dist_pair_"):
                    # 二进制变量
                    binary_str = "是" if val > 0.5 else "否"
                    print(f"    {var_name}: {val} ({binary_str})")
                elif isinstance(val, (int, float)):
                    # 数值变量
                    print(f"    {var_name}: {val:.4f}")
                else:
                    print(f"    {var_name}: {val}")
            else:
                print(f"    {var_name}: None (未求解)")
        
        if same_dist_pair_vars:
            print("\n  same_dist_pair (chiplet对的距离是否与某个之前解相同，二进制变量):")
            # 按chiplet对和之前解索引分组显示
            same_dist_by_pair = {}
            import re
            for var_name, val in same_dist_pair_vars:
                # 解析变量名：same_dist_pair_{suffix}_{i}_{j}_prev{prev_idx}
                # 使用正则表达式匹配：same_dist_pair_*_数字_数字_prev数字
                match = re.search(r'same_dist_pair_[^_]+_(\d+)_(\d+)_prev(\d+)', var_name)
                if match:
                    i_val = int(match.group(1))
                    j_val = int(match.group(2))
                    prev_idx = int(match.group(3))
                    pair_key = (i_val, j_val, prev_idx)
                    if pair_key not in same_dist_by_pair:
                        same_dist_by_pair[pair_key] = []
                    same_dist_by_pair[pair_key].append((var_name, val))
                else:
                    # 如果正则匹配失败，直接显示变量名
                    if "unknown" not in same_dist_by_pair:
                        same_dist_by_pair["unknown"] = []
                    same_dist_by_pair["unknown"].append((var_name, val))
            
            # 按chiplet对和之前解索引排序显示
            for key, vars_list in sorted(same_dist_by_pair.items()):
                if key == "unknown":
                    print("    无法解析的变量:")
                    for var_name, val in sorted(vars_list):
                        print(f"      {var_name}: {val}")
                else:
                    i, j, prev_idx = key
                    name_i = nodes[i].name if hasattr(nodes[i], 'name') and i < len(nodes) else f"Chiplet_{i}"
                    name_j = nodes[j].name if hasattr(nodes[j], 'name') and j < len(nodes) else f"Chiplet_{j}"
                    print(f"    模块对 ({name_i}, {name_j}) 与解 {prev_idx}:")
                    for var_name, val in sorted(vars_list):
                        print(f"      {var_name}: {val}")
        
        # 10.2 打印排除约束相关的约束
        print("\n【排除约束】")
        exclude_constraints = []
        for constraint_name, constraint in ctx.prob.constraints.items():
            if constraint_name.startswith("dx_abs_pair_") or constraint_name.startswith("dy_abs_pair_") or \
               constraint_name.startswith("dx_grid_abs_pair_") or constraint_name.startswith("dy_grid_abs_pair_") or \
               constraint_name.startswith("dist_curr_pair_") or \
               constraint_name.startswith("dist_diff_pair_") or constraint_name.startswith("dist_diff_abs_pair_") or \
               constraint_name.startswith("exclude_solution_dist_pair_") or \
               constraint_name.startswith("same_dist_pair_") or constraint_name.startswith("diff_dist_pair_implies_") or \
               constraint_name.startswith("not_same_implies_") or constraint_name.startswith("all_not_same_implies_"):
                exclude_constraints.append(constraint_name)
        
        if exclude_constraints:
            print(f"  共找到 {len(exclude_constraints)} 个排除约束:")
            for constraint_name in sorted(exclude_constraints):
                constraint = ctx.prob.constraints[constraint_name]
                print(f"    {constraint_name}: {constraint}")
        else:
            print("  (未找到排除约束)")
    else:
        print("\n【排除解约束】")
        print("  (第一次求解，无排除约束)")
