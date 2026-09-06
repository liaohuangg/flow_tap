"""数据加载器:把芯粒布局 + 互联关系 + 线数量 构造成图,并加载线长标签。

每个 system 被表示成一张无向图:
  - 节点 = 芯粒(chiplet),特征 = [x, y, width, height, rotation, power, hubump]
  - 边   = 互联(net),边特征 = [log1p(wireCount), min_clump_manhattan]
    min_clump_manhattan = 两端芯粒 4×4 个 clump(四边中点)之间的最小曼哈顿距离,
    这是"无容量约束下"每根线的最短布线距离(总线长的主导项)。
  - 边权重 = 原始 wireCount(参与求和读出)
  - 图级标签 = log(total_wirelength);平均线长 avg = total / (2 * sum(wireCount)) 解析推出
"""
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------
# 路径 / 常量
# ----------------------------------------------------------------------------
PLACEMENT_DIR = "/root/placement/flow_tap/Dataset/dataset/placement_dataset/placement_dataset_tw"
WIRELENGTH_DIR = "/root/placement/flow_tap/Dataset/dataset/wirelength_dataset"

# 有 total 线长标签的布局文件:69..80,对应 system_340001 .. system_400000,共 60000 个
# (边距离/每侧流量标签只有 69..76 即 340001..380000; total-only 训练不需要这些,故可扩展)
LABELED_FILES = [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]

# 节点特征列顺序(18 维:前 7 个几何/物理,后 11 个容量/需求拥塞信号)
NODE_FEATURES = ["x", "y", "width", "height", "rotation", "power", "hubump",
                 "log_demand", "log_capacity", "load_ratio",
                 "log_cap_left", "log_cap_top", "log_cap_right", "log_cap_bottom",
                 "log_pref_left", "log_pref_top", "log_pref_right", "log_pref_bottom"]
# 边特征列顺序
EDGE_FEATURES = ["log_wireCount", "min_clump_manhattan"]

UBUMP_PITCH = 0.045  # 45um microbump 节距 (mm)


# ----------------------------------------------------------------------------
# 单个 system -> 图
# ----------------------------------------------------------------------------
def _clumps(x: float, y: float, w: float, h: float, hu: float):
    """返回 chiplet(body 左下角 (x,y), 尺寸 w×h, bump 环宽 hu) 的 4 个 clump 坐标。

    clump 位于 bump 环四条边的中点(左/上/右/下),与 TAP-2.5D get_input 一致。
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    return [
        (x - hu / 2.0, cy),          # 左
        (cx, y + h + hu / 2.0),      # 上
        (x + w + hu / 2.0, cy),      # 右
        (cx, y - hu / 2.0),          # 下
    ]


def _min_clump_manhattan(clumps_a, clumps_b) -> float:
    return min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in clumps_a for b in clumps_b)


def _preferred_side(clumps_i, clumps_j):
    """返回 (i 朝 j 的侧索引 h_i, j 朝 i 的侧索引 h_j)。

    h_i = 使两端 clump 距离最小的 i 侧(0..3 = 左/上/右/下),是"无拥塞时该 net 想走的侧"。
    """
    best = None
    h_i = h_j = 0
    for h, a in enumerate(clumps_i):
        for k, b in enumerate(clumps_j):
            d = abs(a[0] - b[0]) + abs(a[1] - b[1])
            if best is None or d < best:
                best = d
                h_i, h_j = h, k
    return h_i, h_j


def _bump_capacity(w_mm: float, h_mm: float, hubump: float) -> int:
    """bump 环总容量(与 LP/CPLEX 的 pmax 求和一致,离散计数)。"""
    nh = int(hubump / UBUMP_PITCH)
    if nh <= 0:
        return 0
    return (2 * nh * int((h_mm + hubump) / UBUMP_PITCH)
            + 2 * nh * int((w_mm + hubump) / UBUMP_PITCH))


def _per_side_capacity(w_mm: float, h_mm: float, hubump: float):
    """bump 环四条边各自的容量 [左, 上, 右, 下], 与 LP/CPLEX 的 pmax[i][h] 一致。

    左/右沿 height 方向、上/下沿 width 方向,每条边 nh 行 bump;四条边之和 = _bump_capacity。
    """
    nh = int(hubump / UBUMP_PITCH)
    if nh <= 0:
        return [0, 0, 0, 0]
    return [
        nh * int((h_mm + hubump) / UBUMP_PITCH),  # 左
        nh * int((w_mm + hubump) / UBUMP_PITCH),  # 上
        nh * int((h_mm + hubump) / UBUMP_PITCH),  # 右
        nh * int((w_mm + hubump) / UBUMP_PITCH),  # 下
    ]


def parse_system(system: dict, use_congestion: bool = True,
                 dist_by_pair: dict | None = None,
                 side_flow_by_name: dict | None = None) -> dict:
    """把一个 system 的原始 dict 解析成图的数据结构。

    use_congestion=True 时,节点额外含 11 个拥塞特征(log_demand/log_capacity/load_ratio
    + 每侧容量 log_cap_{L,T,R,B} + 每侧偏好需求 log_pref_{L,T,R,B})。
    dist_by_pair 提供每条互联边(无向)的路由距离标签 ({(node1,node2): distance}), 附着到边上。
    side_flow_by_name 提供每个 chiplet 每侧边的 bump 流量/容量 ({name: {flow_out,flow_in,capacity}}),
      附着到节点上 (node_side_flow = flow_out+flow_in, node_side_capacity = capacity)。
    """
    chiplets = system["chiplets"]
    name2idx = {c["name"]: i for i, c in enumerate(chiplets)}

    # clump 坐标(左/上/右/下),用于 dmin 与每侧偏好需求
    clumps = [
        _clumps(c["x-position"], c["y-position"], c["width"], c["height"], c["hubump"])
        for c in chiplets
    ]

    # 每 chiplet 总需求(入射带宽)+ 每侧偏好需求(哪侧朝向哪个邻居),几何确定、始终计算
    demand = [0.0] * len(chiplets)
    pref_demand = [[0.0] * 4 for _ in chiplets]
    for e in system["connections"]:
        u = name2idx[e["node1"]]
        v = name2idx[e["node2"]]
        wc = float(e["wireCount"])
        demand[u] += wc
        demand[v] += wc
        h_u, h_v = _preferred_side(clumps[u], clumps[v])
        pref_demand[u][h_u] += wc
        pref_demand[v][h_v] += wc

    nodes = []
    for i, c in enumerate(chiplets):
        feat = [
            float(c["x-position"]),
            float(c["y-position"]),
            float(c["width"]),
            float(c["height"]),
            float(c["rotation"]),
            float(c["power"]),
            float(c["hubump"]),
        ]
        if use_congestion:
            s = 2.0 * demand[i]  # 有向总带宽,与 hubump 选型一致
            cap = float(_bump_capacity(c["width"], c["height"], c["hubump"]))
            load = s / cap if cap > 0 else 0.0
            cl, ct, cr, cb = _per_side_capacity(c["width"], c["height"], c["hubump"])
            pl, pt, pr, pb = pref_demand[i]
            feat += [math.log1p(s), math.log1p(cap), load,
                     math.log1p(cl), math.log1p(ct), math.log1p(cr), math.log1p(cb),
                     math.log1p(pl), math.log1p(pt), math.log1p(pr), math.log1p(pb)]
        nodes.append(tuple(feat))

    # 每侧拥塞原始信号(供 correction 直接使用): [每侧容量(4), 每侧偏好需求(4)] log1p
    node_cong = []
    for i, c in enumerate(chiplets):
        cl, ct, cr, cb = _per_side_capacity(c["width"], c["height"], c["hubump"])
        pl, pt, pr, pb = pref_demand[i]
        node_cong.append([math.log1p(cl), math.log1p(ct), math.log1p(cr), math.log1p(cb),
                          math.log1p(pl), math.log1p(pt), math.log1p(pr), math.log1p(pb)])

    # 每 chiplet 每侧边的 bump 流量 (flow_out+flow_in, 原始计数) 与容量 (CPLEX side_flow)
    node_side_flow = None
    node_side_capacity = None
    if side_flow_by_name is not None:
        node_side_flow = []
        node_side_capacity = []
        for c in chiplets:
            sf = side_flow_by_name.get(c["name"])
            if sf is None:
                node_side_flow.append(None)
                node_side_capacity.append(None)
            else:
                node_side_flow.append([fo + fi for fo, fi in zip(sf["flow_out"], sf["flow_in"])])
                node_side_capacity.append(list(sf["capacity"]))

    edges = []
    wcount = 0.0
    for e in system["connections"]:
        u = name2idx[e["node1"]]
        v = name2idx[e["node2"]]
        wc = float(e["wireCount"])
        wcount += wc

        log_wc = math.log1p(wc)
        dmin = _min_clump_manhattan(clumps[u], clumps[v])
        dist = None
        if dist_by_pair is not None:
            dist = dist_by_pair.get((e["node1"], e["node2"]))
        # 无向边,存两条有向边(边特征/权重/标签相同)
        edges.append((u, v, log_wc, dmin, dist))
        edges.append((v, u, log_wc, dmin, dist))

    # die 包围盒面积
    min_x = min(c["x-position"] for c in chiplets)
    max_x = max(c["x-position"] + c["width"] for c in chiplets)
    min_y = min(c["y-position"] for c in chiplets)
    max_y = max(c["y-position"] + c["height"] for c in chiplets)
    die_area = (max_x - min_x) * (max_y - min_y)

    return {
        "nodes": nodes,
        "edges": edges,
        "wcount": wcount,
        "num_nets": len(system["connections"]),
        "die_area": die_area,
        "node_side_flow": node_side_flow,
        "node_side_capacity": node_side_capacity,
        "node_cong": node_cong,
    }


def _read_total_wirelength(system_id: int) -> float:
    p = os.path.join(WIRELENGTH_DIR, "total_wirelength",
                     f"system_total_wirelength_{system_id}.csv")
    with open(p) as f:
        return float(f.read().strip())


def _read_edge_labels(system_id: int):
    """读取每条互联边(无向)的路由距离标签(CPLEX 产出), 返回 [{node1,node2,wireCount,distance}...]。"""
    p = os.path.join(WIRELENGTH_DIR, "edge_wirelength",
                     f"system_edge_wirelength_{system_id}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _read_side_flow(system_id: int):
    """读取每个 chiplet 每侧边的 bump 流量/容量 (CPLEX 产出)。

    返回 {"sides":[...], "chiplets":[{name,flow_out,flow_in,capacity}...]}, 无文件则 None。
    """
    p = os.path.join(WIRELENGTH_DIR, "side_flow",
                     f"system_side_flow_{system_id}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_labeled_systems(use_congestion: bool = True) -> List[dict]:
    """加载全部有标签的 system,返回 [(sid, system, graph, total_wl, edge_labels)]。"""
    records = []
    for fnum in LABELED_FILES:
        path = os.path.join(PLACEMENT_DIR, f"chiplet_dataset_{fnum}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            systems = json.load(f)
        for sid, system in systems.items():
            sid_int = int(sid.split("_")[1])
            edge_labels = _read_edge_labels(sid_int)
            dist_by_pair = None
            if edge_labels is not None:
                dist_by_pair = {}
                for e in edge_labels:
                    dist_by_pair[(e["node1"], e["node2"])] = e["distance"]
                    dist_by_pair[(e["node2"], e["node1"])] = e["distance"]
            side_flow = _read_side_flow(sid_int)
            side_by_name = None
            if side_flow is not None:
                side_by_name = {e["name"]: e for e in side_flow["chiplets"]}
            graph = parse_system(system, use_congestion=use_congestion,
                                 dist_by_pair=dist_by_pair,
                                 side_flow_by_name=side_by_name)
            total_wl = _read_total_wirelength(sid_int)
            records.append((sid, system, graph, total_wl, edge_labels))
    return records


# ----------------------------------------------------------------------------
# 归一化统计(只 fit 在训练集上)
# ----------------------------------------------------------------------------
@dataclass
class Normalizer:
    node_mean: torch.Tensor  # [7]
    node_std: torch.Tensor   # [7]
    edge_mean: torch.Tensor  # [2]
    edge_std: torch.Tensor   # [2]

    @classmethod
    def fit(cls, graphs: List[dict]) -> "Normalizer":
        node_dim = len(graphs[0]["nodes"][0])
        edge_dim = 2
        node_sum = torch.zeros(node_dim, dtype=torch.float64)
        node_sq = torch.zeros(node_dim, dtype=torch.float64)
        node_cnt = 0
        edge_sum = torch.zeros(edge_dim, dtype=torch.float64)
        edge_sq = torch.zeros(edge_dim, dtype=torch.float64)
        edge_cnt = 0

        for g in graphs:
            for feat in g["nodes"]:
                f = torch.tensor(feat, dtype=torch.float64)
                node_sum += f
                node_sq += f * f
                node_cnt += 1
            for (u, v, log_wc, dmin, _dist) in g["edges"]:
                f = torch.tensor([log_wc, dmin], dtype=torch.float64)
                edge_sum += f
                edge_sq += f * f
                edge_cnt += 1

        node_mean = node_sum / node_cnt
        node_std = torch.sqrt(node_sq / node_cnt - node_mean ** 2).clamp_min(1e-6)
        edge_mean = edge_sum / edge_cnt
        edge_std = torch.sqrt(edge_sq / edge_cnt - edge_mean ** 2).clamp_min(1e-6)
        return cls(node_mean.float(), node_std.float(), edge_mean.float(), edge_std.float())


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class ChipletWirelengthDataset(Dataset):
    """图回归数据集。每个样本返回:
        x:           [N, 18] 标准化节点特征(use_congestion=True 时含 11 个拥塞特征)
        edge_index:  [2, 2E] 有向边索引
        edge_attr:   [2E, 2] 标准化边特征
        edge_weight: [2E]    原始 wireCount
        edge_dmin:   [2E]    原始 min_clump_manhattan(残差读出用)
        global_attr: [4]     log 化图级标量
        wcount:      [1]     总 wireCount(用于解析平均线长)
        y:           [1]     log(total_wirelength)

    所有张量在构造时一次性构建并缓存,避免每个 epoch 重复 Python 层解析(原瓶颈)。
    """

    def __init__(self, records: List[dict], normalizer: Normalizer):
        self.norm = normalizer
        self.samples = [self._build(r) for r in records]

    def __len__(self):
        return len(self.samples)

    def _build(self, record):
        _, _, graph, total_wl, _edge_labels = record

        x = torch.tensor(graph["nodes"], dtype=torch.float32)
        x = (x - self.norm.node_mean) / self.norm.node_std

        # 原始几何(未归一化): x, y, w, h, hubump —— 供前向里可微计算 min_clump_manhattan
        node_geom = torch.tensor(
            [[n[0], n[1], n[2], n[3], n[6]] for n in graph["nodes"]],
            dtype=torch.float32)

        # 每侧拥塞原始信号(供 correction 直接使用): [每侧容量(4), 每侧偏好需求(4)] log1p
        cong = torch.tensor(graph["node_cong"], dtype=torch.float32)

        # 每 chiplet 每侧边的 bump 流量标签 ([N,4] = flow_out+flow_in, 原始计数); 无标签则 None
        nsf = graph.get("node_side_flow")
        node_flow = (None if nsf is None or any(f is None for f in nsf)
                     else torch.tensor(nsf, dtype=torch.float32))

        edge_list = graph["edges"]
        edge_index = torch.tensor([(u, v) for (u, v, *_r) in edge_list],
                                  dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([[log_wc, dmin] for (_, _, log_wc, dmin, _d) in edge_list],
                                 dtype=torch.float32)
        edge_attr = (edge_attr - self.norm.edge_mean) / self.norm.edge_std

        # 原始 wireCount 作为边权重(求和读出用);dmin 由 log_wc 反推 expm1
        edge_weight = torch.tensor([math.expm1(log_wc) for (_, _, log_wc, dmin, _d) in edge_list],
                                   dtype=torch.float32)
        # 原始 min_clump_manhattan(参考值; 模型里从 node_geom 在线重算可微版本)
        edge_dmin = torch.tensor([dmin for (_, _, _, dmin, _d) in edge_list],
                                 dtype=torch.float32)

        # 每边路由距离标签(CPLEX): d_edge_label = d_net[i→j] + d_net[j→i], 若无标签则为 None
        dists = [d for (_, _, _, _, d) in edge_list]
        edge_label = (None if any(d is None for d in dists)
                      else torch.tensor(dists, dtype=torch.float32))

        n_chips = len(graph["nodes"])
        n_nets = graph["num_nets"]
        global_attr = torch.tensor([
            math.log1p(n_chips),
            math.log1p(n_nets),
            math.log1p(graph["wcount"]),
            math.log1p(graph["die_area"]),
        ], dtype=torch.float32)

        y = torch.tensor([math.log(total_wl)], dtype=torch.float32)

        return {
            "x": x,
            "node_geom": node_geom,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_weight": edge_weight,
            "edge_dmin": edge_dmin,
            "edge_label": edge_label,
            "node_flow": node_flow,
            "cong": cong,
            "global_attr": global_attr,
            "wcount": torch.tensor([graph["wcount"]], dtype=torch.float32),
            "y": y,
        }

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[dict]) -> dict:
    xs, geoms, edge_indices, edge_attrs, edge_weights, edge_dmins, edge_labels, node_flows, congs, global_attrs, wcounts, ys = \
        [], [], [], [], [], [], [], [], [], [], [], []
    node_offset = 0
    batch_list = []

    has_label = all(item["edge_label"] is not None for item in batch)
    has_flow = all(item["node_flow"] is not None for item in batch)

    for i, item in enumerate(batch):
        n = item["x"].size(0)
        xs.append(item["x"])
        geoms.append(item["node_geom"])
        edge_indices.append(item["edge_index"] + node_offset)
        edge_attrs.append(item["edge_attr"])
        edge_weights.append(item["edge_weight"])
        edge_dmins.append(item["edge_dmin"])
        if has_label:
            edge_labels.append(item["edge_label"])
        if has_flow:
            node_flows.append(item["node_flow"])
        congs.append(item["cong"])
        global_attrs.append(item["global_attr"])
        wcounts.append(item["wcount"])
        ys.append(item["y"])
        batch_list.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n

    out = {
        "x": torch.cat(xs, dim=0),
        "node_geom": torch.cat(geoms, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "edge_weight": torch.cat(edge_weights, dim=0),
        "edge_dmin": torch.cat(edge_dmins, dim=0),
        "cong": torch.cat(congs, dim=0),
        "global_attr": torch.stack(global_attrs, dim=0),
        "batch": torch.cat(batch_list, dim=0),
        "wcount": torch.cat(wcounts, dim=0),
        "y": torch.cat(ys, dim=0),
    }
    out["edge_label"] = torch.cat(edge_labels, dim=0) if has_label else None
    out["node_flow"] = torch.cat(node_flows, dim=0) if has_flow else None
    return out


def split_records(records: List[dict], seed: int = 42,
                  train_ratio: float = 0.8, val_ratio: float = 0.1):
    """按 system 打乱切分 train/val/test(默认 8:1:1)。"""
    import random
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return ([records[i] for i in train_idx],
            [records[i] for i in val_idx],
            [records[i] for i in test_idx])


def get_dataloaders(batch_size: int = 64, num_workers: int = 0, seed: int = 42,
                    use_congestion: bool = True):
    """加载数据、fit 归一化、切分并返回 (train_loader, val_loader, test_loader, normalizer)。"""
    records = load_labeled_systems(use_congestion=use_congestion)
    train_recs, val_recs, test_recs = split_records(records, seed=seed)

    normalizer = Normalizer.fit([r[2] for r in train_recs])

    train_ds = ChipletWirelengthDataset(train_recs, normalizer)
    val_ds = ChipletWirelengthDataset(val_recs, normalizer)
    test_ds = ChipletWirelengthDataset(test_recs, normalizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, normalizer


if __name__ == "__main__":
    records = load_labeled_systems()
    print(f"加载 {len(records)} 个有标签 system")
    g = records[0][2]
    print(f"样例: {len(g['nodes'])} 芯粒, {g['num_nets']} net, "
          f"wcount={g['wcount']:.0f}, die_area={g['die_area']:.1f}")

    loaders = get_dataloaders(batch_size=8)
    train_loader, val_loader, test_loader, norm = loaders
    batch = next(iter(train_loader))
    print("batch keys:", list(batch.keys()))
    print("x", tuple(batch["x"].shape), "edge_index", tuple(batch["edge_index"].shape),
          "edge_attr", tuple(batch["edge_attr"].shape), "edge_weight", tuple(batch["edge_weight"].shape))
    print("train/val/test:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
