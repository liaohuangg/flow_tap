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

# 有标签的布局文件:69..76,对应 system_340001 .. system_380000,共 40000 个
LABELED_FILES = [69, 70, 71, 72, 73, 74, 75, 76]

# 节点特征列顺序(10 维:前 7 个几何/物理,后 3 个容量/需求拥塞信号)
NODE_FEATURES = ["x", "y", "width", "height", "rotation", "power", "hubump",
                 "log_demand", "log_capacity", "load_ratio"]
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


def _bump_capacity(w_mm: float, h_mm: float, hubump: float) -> int:
    """bump 环总容量(与 LP/CPLEX 的 pmax 求和一致,离散计数)。"""
    nh = int(hubump / UBUMP_PITCH)
    if nh <= 0:
        return 0
    return (2 * nh * int((h_mm + hubump) / UBUMP_PITCH)
            + 2 * nh * int((w_mm + hubump) / UBUMP_PITCH))


def parse_system(system: dict, use_congestion: bool = True) -> dict:
    """把一个 system 的原始 dict 解析成图的数据结构。

    use_congestion=True 时,节点额外含 3 个拥塞特征(log_demand/log_capacity/load_ratio)。
    """
    chiplets = system["chiplets"]
    name2idx = {c["name"]: i for i, c in enumerate(chiplets)}

    # 每个 chiplet 的入射总带宽(需求),仅当需要拥塞特征时计算
    demand = [0.0] * len(chiplets)
    if use_congestion:
        for e in system["connections"]:
            demand[name2idx[e["node1"]]] += float(e["wireCount"])
            demand[name2idx[e["node2"]]] += float(e["wireCount"])

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
            feat += [math.log1p(s), math.log1p(cap), load]
        nodes.append(tuple(feat))

    clumps = [
        _clumps(c["x-position"], c["y-position"], c["width"], c["height"], c["hubump"])
        for c in chiplets
    ]

    edges = []
    wcount = 0.0
    for e in system["connections"]:
        u = name2idx[e["node1"]]
        v = name2idx[e["node2"]]
        wc = float(e["wireCount"])
        wcount += wc

        log_wc = math.log1p(wc)
        dmin = _min_clump_manhattan(clumps[u], clumps[v])
        # 无向边,存两条有向边(边特征相同,权重相同)
        edges.append((u, v, log_wc, dmin))
        edges.append((v, u, log_wc, dmin))

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
    }


def _read_total_wirelength(system_id: int) -> float:
    p = os.path.join(WIRELENGTH_DIR, "total_wirelength",
                     f"system_total_wirelength_{system_id}.csv")
    with open(p) as f:
        return float(f.read().strip())


def load_labeled_systems(use_congestion: bool = True) -> List[dict]:
    """加载全部 20000 个有标签的 system,返回 [(system_id, system_dict, graph_dict, total_wl)]。"""
    records = []
    for fnum in LABELED_FILES:
        path = os.path.join(PLACEMENT_DIR, f"chiplet_dataset_{fnum}.json")
        with open(path) as f:
            systems = json.load(f)
        for sid, system in systems.items():
            sid_int = int(sid.split("_")[1])
            graph = parse_system(system, use_congestion=use_congestion)
            total_wl = _read_total_wirelength(sid_int)
            records.append((sid, system, graph, total_wl))
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
            for (u, v, log_wc, dmin) in g["edges"]:
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
        x:           [N, 10] 标准化节点特征(use_congestion=True 时含 3 个拥塞特征)
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
        _, _, graph, total_wl = record

        x = torch.tensor(graph["nodes"], dtype=torch.float32)
        x = (x - self.norm.node_mean) / self.norm.node_std

        edge_list = graph["edges"]
        edge_index = torch.tensor([(u, v) for (u, v, *_) in edge_list],
                                  dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([[log_wc, dmin] for (_, _, log_wc, dmin) in edge_list],
                                 dtype=torch.float32)
        edge_attr = (edge_attr - self.norm.edge_mean) / self.norm.edge_std

        # 原始 wireCount 作为边权重(求和读出用);dmin 由 log_wc 反推 expm1
        edge_weight = torch.tensor([math.expm1(log_wc) for (_, _, log_wc, dmin) in edge_list],
                                   dtype=torch.float32)
        # 原始 min_clump_manhattan(残差读出: d_edge = dmin + 容量膨胀修正)
        edge_dmin = torch.tensor([dmin for (_, _, _, dmin) in edge_list],
                                 dtype=torch.float32)

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
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_weight": edge_weight,
            "edge_dmin": edge_dmin,
            "global_attr": global_attr,
            "wcount": torch.tensor([graph["wcount"]], dtype=torch.float32),
            "y": y,
        }

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[dict]) -> dict:
    xs, edge_indices, edge_attrs, edge_weights, edge_dmins, global_attrs, wcounts, ys = \
        [], [], [], [], [], [], [], []
    node_offset = 0
    batch_list = []

    for i, item in enumerate(batch):
        n = item["x"].size(0)
        xs.append(item["x"])
        edge_indices.append(item["edge_index"] + node_offset)
        edge_attrs.append(item["edge_attr"])
        edge_weights.append(item["edge_weight"])
        edge_dmins.append(item["edge_dmin"])
        global_attrs.append(item["global_attr"])
        wcounts.append(item["wcount"])
        ys.append(item["y"])
        batch_list.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n

    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "edge_weight": torch.cat(edge_weights, dim=0),
        "edge_dmin": torch.cat(edge_dmins, dim=0),
        "global_attr": torch.stack(global_attrs, dim=0),
        "batch": torch.cat(batch_list, dim=0),
        "wcount": torch.cat(wcounts, dim=0),
        "y": torch.cat(ys, dim=0),
    }


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
