#!/usr/bin/env python3
"""
数据集处理: 把 gen_cfg.py 生成的 cpu-dram .cfg 文件转换成
placement_dataset_tw/chiplet_dataset_{k}.json 格式 (与 chiplet_dataset_1.json 完全一致)。

- 5000 个 system 一个 json 文件 (CHUNK=5000, 与既有 placement_dataset_tw 对齐):
    system_{sid} -> chiplet_dataset_{(sid-1)//5000 + 1}.json
    例如 system_380001..400000 -> chiplet_dataset_77.json .. 80.json
- 每个 chiplet 输出字段:
    name        = A/B/C/...(按顺序)
    x-position  = cfg 的 x (初始位置, gen_cfg.py 默认全 0, 后续由 placer 填充)
    y-position  = cfg 的 y
    width/height= cfg 的 widths/heights (本体 body/die 尺寸)
    rotation    = 0 (cfg 未指定旋转)
    power       = cfg 的 powers
    hubump      = compute_hubump(body_w, body_h, s), s = Σ(M[i][j]+M[j][i])
                  即 2 × 该 chiplet 相连的所有 wireCount 之和
- connections: 对称矩阵上三角 (i<j 且 wireCount>0) -> [{node1, node2, wireCount}]

hubump 采用 gen_wirelength_dataset.py 的 compute_hubump 口径:
  以 body(die) 尺寸为输入, 用整数化 pmax 容量判定, 找最小满足容量的环宽,
  保证后续布线 ILP 的 bump 容量 >= s (可解), 无连续公式 int 截断缺口。

用法:
  python input_preprocess.py --start 380001 --end 400000
  python input_preprocess.py --start 380001 --end 400000 \
      --config-dir Dataset/config \
      --output-dir Dataset/dataset/placement_dataset/placement_dataset_tw
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
DEFAULT_CONFIG_DIR = PROJECT / "Dataset" / "config"
DEFAULT_OUT_DIR = PROJECT / "Dataset" / "dataset" / "placement_dataset" / "placement_dataset_tw"

CHUNK = 5000            # 每个 chiplet_dataset_{k}.json 含 5000 systems
UBUMP_PITCH = 0.045     # 45um microbump 节距, mm


# --------------------------------------------------------------------------- #
# hubump (与 gen_wirelength_dataset.py 的 compute_hubump / _bump_capacity 完全一致)
# --------------------------------------------------------------------------- #
def _bump_capacity(w_mm: float, h_mm: float, hubump: float) -> int:
    """routing.get_input 里 pmax 的整数化 bump 容量 (上/下/左/右 4 个 clump 之和)。

    上/下 clump 用 height, 左/右 clump 用 width, 每 clump =
    int(hubump/0.045) * int((edge+hubump)/0.045)。
    """
    nh = int(hubump / UBUMP_PITCH)
    if nh <= 0:
        return 0
    return (2 * nh * int((h_mm + hubump) / UBUMP_PITCH)
            + 2 * nh * int((w_mm + hubump) / UBUMP_PITCH))


def compute_hubump(w_mm: float, h_mm: float, s: float) -> float:
    """按连接数 s 计算芯片四周 bump 环宽度 (mm)。

    w_mm/h_mm 为芯片本体(body/die)尺寸。s = Σ(M[i][j] + M[j][i]) = 2×单边 wireCount。
    与 TAP-2.5D compute_ubump_overhead 同一思路, 但直接用整数化 pmax 容量判定,
    保证布线 ILP 的 bump 容量 >= s (可解)。
    """
    if s <= 0:
        return 0.0
    k = 1
    w_stretch = UBUMP_PITCH * k
    while True:
        if _bump_capacity(w_mm, h_mm, w_stretch) >= s:
            return w_stretch
        k += 1
        w_stretch = UBUMP_PITCH * k
        if k > 1000:
            raise ValueError("microbump is too high to be a feasible case")


# --------------------------------------------------------------------------- #
# cfg 解析
# --------------------------------------------------------------------------- #
def _parse_list(value: str):
    """'19,\t17,\t5' -> [19.0, 17.0, 5.0] (逗号分隔, 忽略空白/tab)。"""
    return [float(x.strip()) for x in value.split(",") if x.strip() != ""]


def _parse_matrix_row(value: str):
    """'0,\t0,\t256,...;' -> [0, 0, 256, ...] (去行尾分号, 逗号分隔)。"""
    value = value.strip().rstrip(";").strip()
    return [int(x.strip()) for x in value.split(",") if x.strip() != ""]


def parse_cfg(cfg_path) -> dict:
    """解析一个 cpu-dram .cfg 文件, 返回 {chiplet_count, widths, heights, powers,
    connections(对称矩阵), x, y}。逐行解析, 兼容 system_1.cfg 与 gen_cfg.py 输出。"""
    text = Path(cfg_path).read_text(encoding="utf-8")

    n = int(re.search(r"chiplet_count\s*=\s*(\d+)", text).group(1))
    widths = _parse_list(re.search(r"widths\s*=\s*(.+)", text).group(1))
    heights = _parse_list(re.search(r"heights\s*=\s*(.+)", text).group(1))
    powers = _parse_list(re.search(r"powers\s*=\s*(.+)", text).group(1))

    # connections 矩阵: 从 'connections = ' 到 'u = ' 行 (首行带前缀, 续行 3-tab 缩进)
    m = re.search(r"connections\s*=\s*(.*?)(?=^\s*u\s*=)", text, re.DOTALL | re.MULTILINE)
    if not m:
        raise ValueError(f"{cfg_path}: 未找到 connections 矩阵")
    matrix = []
    for line in m.group(1).strip().splitlines():
        row = _parse_matrix_row(line)
        if row:
            matrix.append(row)

    # x/y 初始位置 (gen_cfg.py 输出全 0; 参考 system_1.cfg 亦为 0, 实际布局由 placer 产生)
    x = _parse_list(re.search(r"^x\s*=\s*(.+)", text, re.MULTILINE).group(1))
    y = _parse_list(re.search(r"^y\s*=\s*(.+)", text, re.MULTILINE).group(1))

    # 校验
    assert len(widths) == len(heights) == len(powers) == n, f"{cfg_path}: 尺寸/功耗维度不一致"
    assert len(matrix) == n, f"{cfg_path}: connections 行数 {len(matrix)} != {n}"
    for i, row in enumerate(matrix):
        assert len(row) == n, f"{cfg_path}: connections 第 {i} 行列数 {len(row)} != {n}"

    return {
        "chiplet_count": n,
        "widths": widths,
        "heights": heights,
        "powers": powers,
        "connections": matrix,
        "x": x,
        "y": y,
    }


# --------------------------------------------------------------------------- #
# cfg -> record
# --------------------------------------------------------------------------- #
def _index_to_name(i: int) -> str:
    """0->A, 1->B, ... 与官方 cpu-dram.cfg 记录一致 (chiplet 名按字母顺序)。"""
    return chr(ord("A") + i)


def build_record(sid: int, data: dict) -> dict:
    """把一个 cfg 数据构造成 chiplet_dataset_{k}.json 里的单条 record。"""
    n = data["chiplet_count"]
    widths = data["widths"]
    heights = data["heights"]
    powers = data["powers"]
    M = data["connections"]
    x = data.get("x", [0.0] * n)
    y = data.get("y", [0.0] * n)

    chiplets = []
    for i in range(n):
        # s = Σ_j (M[i][j] + M[j][i]) = 2 × 入射 wireCount (对称矩阵, 对角为 0)
        s = sum(M[i][j] + M[j][i] for j in range(n))
        hubump = compute_hubump(widths[i], heights[i], s)
        chiplets.append({
            "name": _index_to_name(i),
            "x-position": float(x[i]) if i < len(x) else 0.0,
            "y-position": float(y[i]) if i < len(y) else 0.0,
            "width": float(widths[i]),
            "height": float(heights[i]),
            "rotation": 0,
            "power": float(powers[i]),
            "hubump": round(hubump, 6),
        })

    connections = []
    for i in range(n):
        for j in range(i + 1, n):
            wc = M[i][j]
            if wc > 0:
                connections.append({
                    "node1": _index_to_name(i),
                    "node2": _index_to_name(j),
                    "wireCount": wc,
                })

    return {
        "system_id": f"system_{sid}",
        "chiplets": chiplets,
        "connections": connections,
    }


# --------------------------------------------------------------------------- #
# 批量处理
# --------------------------------------------------------------------------- #
def process_range(start_sys: int, end_sys: int, config_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: dict[int, dict] = defaultdict(dict)
    missing: list[int] = []
    total_chiplets = 0
    total_connections = 0
    hubump_min, hubump_max = float("inf"), 0.0

    for sid in range(start_sys, end_sys + 1):
        cfg_path = config_dir / f"system_{sid}.cfg"
        if not cfg_path.exists():
            missing.append(sid)
            continue
        data = parse_cfg(cfg_path)
        rec = build_record(sid, data)

        total_chiplets += data["chiplet_count"]
        total_connections += len(rec["connections"])
        for c in rec["chiplets"]:
            hubump_min = min(hubump_min, c["hubump"])
            hubump_max = max(hubump_max, c["hubump"])

        k = (sid - 1) // CHUNK + 1
        chunks[k][f"system_{sid}"] = rec

    for k in sorted(chunks):
        out_fp = out_dir / f"chiplet_dataset_{k}.json"
        out_fp.write_text(json.dumps(chunks[k]), encoding="utf-8")
        print(f"[preprocess] 写出 {out_fp} ({len(chunks[k])} systems)", flush=True)

    n_sys = end_sys - start_sys + 1
    n_done = sum(len(v) for v in chunks.values())
    print(f"[preprocess] DONE: 转换 {n_done}/{n_sys} systems "
          f"({len(chunks)} 个 json 文件, 每批 {CHUNK})", flush=True)
    print(f"[preprocess] 统计: chiplet 总数 {total_chiplets}, 连接总数 {total_connections}, "
          f"hubump 范围 [{hubump_min:.4f}, {hubump_max:.4f}] mm", flush=True)
    if missing:
        print(f"[preprocess] 缺失 cfg ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}",
              flush=True)


# --------------------------------------------------------------------------- #
# 兼容保留: tool.py 的兜底导入 (flow_GCN 旧流水线)
# --------------------------------------------------------------------------- #
def load_chiplets_json(json_path: str | None = None):
    """[旧 flow_GCN 流水线] 加载 chiplets.json (顶层 key 为 chiplet 名)。仅供 tool.py 兜底。"""
    if json_path is None:
        raise FileNotFoundError(
            "load_chiplets_json 需要显式传入 json_path (旧 flow_GCN 默认路径已移除)"
        )
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_chiplet_table(chiplets: dict):
    """[旧 flow_GCN 流水线] 把 chiplets.json 转成表。仅供 tool.py 兜底。"""
    table = []
    for name, info in chiplets.items():
        table.append({
            "name": name,
            "dimensions": info.get("dimensions", {}),
            "phys": info.get("phys", []),
            "power": info.get("power", None),
        })
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=380001)
    ap.add_argument("--end", type=int, default=400000)
    ap.add_argument("--config-dir", type=str, default=str(DEFAULT_CONFIG_DIR),
                    help="cfg 文件目录 (默认 Dataset/config)")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR),
                    help="输出目录 (默认 Dataset/dataset/placement_dataset/placement_dataset_tw)")
    args = ap.parse_args()

    config_dir = Path(args.config_dir)
    out_dir = Path(args.output_dir)
    process_range(args.start, args.end, config_dir, out_dir)


if __name__ == "__main__":
    main()
