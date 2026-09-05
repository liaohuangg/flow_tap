#!/usr/bin/env python3
"""
预处理: 把 placement_dataset 里的 chiplet 矩形当作"footprint"(本体 + 四周 bump ring),
根据互连线数量 (wireCount) 计算每个 chiplet 的 bump region 宽度 (hubump),
然后向内缩一个 hubump, 得到真实的"本体(body)"坐标 (body ⊂ footprint, 无重叠)。

背景:
  - 原始 greedy 布局生成时未考虑 bump region 与连线, 产生的矩形互不重叠。
  - 现在把这些无重叠矩形重解释为 footprint(已被 bump 包围的芯片)。
  - 模型(resultEval 热模型 / TAP-2.5D 线长模型)内部是"拿 body + hubump 再向外扩回 footprint"。
  - 所以预处理只需: footprint -> body (中心不变, 四周各去 hubump), 并把 hubump 存下,
    模型拿 body + 存下的 hubump 即可精确还原原 footprint。

hubump 计算 (自洽固定点, 与 TAP-2.5D routing 的 pmax 容量判定一致, 只和 wireCount 有关):
  s = Σ(M[i][j] + M[j][i]) = 2 × 该 chiplet 相连的所有 wireCount 之和
  布线要求 hubump = f(die)(环形 microbump 容纳 s), 而 die = footprint - 2*hubump,
  联立解固定点 hubump = f(footprint - 2*hubump), 保证 footprint 不变 + 布线容量满足

不可行 case 处理:
  当某 chiplet 的 2*hubump 超过 footprint 尺寸 (即 body 边长 <= 0, 物理不可行:
  小 die 被塞了过高的互连带宽, 45um microbump 环宽超过芯片本体), 则**整体丢弃该 system**,
  并把编号记录到 OUT_DIR/dropped_systems.txt。

坐标约定:
  输入 x-position/y-position/width/height = footprint 左下角 + 尺寸 (mm)
  输出 x-position/y-position/width/height = body      左下角 + 尺寸 (mm)
        x' = x + hubump,  y' = y + hubump
        w' = w - 2*hubump, h' = h - 2*hubump
        chiplet["hubump"] = hubump  (供下游还原 footprint)

输出目录: dataset/placement_dataset/placement_dataset_tw/  (与原 placement_dataset 分离, 不改动原数据)
用法:
  python preprocess_bump_region.py --start 300001 --end 380000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
DATASET = PROJECT / "Dataset" / "dataset"
SRC_DIR = DATASET / "placement_dataset"
OUT_DIR = DATASET / "placement_dataset" / "placement_dataset_tw"

CHUNK = 5000  # 每个 chiplet_dataset_{k}.json 含 5000 systems
UBUMP_PITCH = 0.045  # 45um microbump 节距, mm
DROPPED_LIST = OUT_DIR / "dropped_systems.txt"


class InfeasibleBumpError(Exception):
    """bump region 不可行: hubump 过大, 芯片本体 body <= 0。"""

    def __init__(self, name: str, w: float, h: float, wire_count: float, hubump: float):
        self.name = name
        self.w = w
        self.h = h
        self.wire_count = wire_count  # 单边总 wireCount (实际互连线数量)
        self.hubump = hubump
        super().__init__(
            f"chiplet {name}: 2*hubump={2*hubump:.3f} 超过 footprint {w}x{h} (单边wireCount={wire_count:.0f})"
        )


def _bump_capacity(w_mm: float, h_mm: float, hubump: float) -> int:
    """四周环形 microbump 能容纳的连接数。与 TAP-2.5D routing.get_input 的 pmax 完全一致:
    int(hubump/0.045) * int((边+hubump)/0.045), 4 边求和。保证布线容量判定无截断误差。"""
    nh = int(hubump / UBUMP_PITCH)
    if nh <= 0:
        return 0
    return 2 * nh * int((h_mm + hubump) / UBUMP_PITCH) + 2 * nh * int((w_mm + hubump) / UBUMP_PITCH)


def compute_hubump(w_mm: float, h_mm: float, s: float) -> float:
    """自洽固定点: 求最小 hubump, 使 die = footprint - 2*hubump 且 hubump = f(die)。

    传入 w_mm/h_mm 是 footprint 尺寸(原始无重叠矩形)。布线模型要求 hubump = f(die)
    (环容量装得下 s 根连接), 而 die = footprint - 2*hubump, 二者联立:
        hubump = f(footprint - 2*hubump)
    这样 footprint 保持不变(不重叠), 同时布线容量满足, die 尺寸由 footprint 与 s 决定。

    若 footprint 太小(die 缩到 <=0 环容量仍不够), 抛 ValueError, 由上层丢弃该 system。
    """
    if s <= 0:
        return 0.0
    k = 1
    while True:
        hubump = UBUMP_PITCH * k
        die_w = w_mm - 2 * hubump
        die_h = h_mm - 2 * hubump
        if die_w <= 0 or die_h <= 0:
            raise ValueError("footprint too small to fit s connections")
        if _bump_capacity(die_w, die_h, hubump) >= s:
            return hubump
        k += 1
        if k > 1000:
            raise ValueError("microbump is too high to be a feasible case")


def _connection_matrix(chiplets: list[dict], connections: list[dict]) -> list[list[float]]:
    """对称连接矩阵 M[i][j] = chiplet i 与 j 之间的 wireCount 之和。"""
    names = [str(c.get("name", f"C{i}")) for i, c in enumerate(chiplets)]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    n = len(chiplets)
    M = [[0.0] * n for _ in range(n)]
    for conn in connections:
        n1 = str(conn.get("node1", ""))
        n2 = str(conn.get("node2", ""))
        if n1 not in name_to_idx or n2 not in name_to_idx:
            continue
        i = name_to_idx[n1]
        j = name_to_idx[n2]
        wc = float(conn.get("wireCount", 0.0))
        M[i][j] += wc
        M[j][i] += wc
    return M


def preprocess_record(record: dict) -> dict:
    """把单个 system 的 chiplet 从 footprint 缩成 body, 并写回 hubump。

    若任一 chiplet 不可行 (body <= 0), 抛 InfeasibleBumpError, 由上层整体丢弃该 system。
    """
    chiplets = record["chiplets"]
    connections = record.get("connections", [])
    M = _connection_matrix(chiplets, connections)
    n = len(chiplets)

    new_chiplets = []
    for i, c in enumerate(chiplets):
        w = float(c["width"])     # footprint 宽
        h = float(c["height"])    # footprint 高
        x = float(c["x-position"])  # footprint 左下角
        y = float(c["y-position"])
        s = sum(M[i][j] + M[j][i] for j in range(n))  # = 2×单边wireCount
        try:
            hubump = compute_hubump(w, h, s)  # 固定点: footprint 不变 + 容量满足
        except ValueError:
            raise InfeasibleBumpError(str(c.get("name", f"C{i}")), w, h, s / 2.0, 0.0)

        body_w = w - 2 * hubump
        body_h = h - 2 * hubump
        if body_w <= 0 or body_h <= 0:  # 防御, 固定点已保证 body > 0
            raise InfeasibleBumpError(str(c.get("name", f"C{i}")), w, h, s / 2.0, hubump)

        new_c = dict(c)
        new_c["x-position"] = round(x + hubump, 6)
        new_c["y-position"] = round(y + hubump, 6)
        new_c["width"] = round(body_w, 6)
        new_c["height"] = round(body_h, 6)
        new_c["hubump"] = round(hubump, 6)
        new_chiplets.append(new_c)

    new_record = dict(record)
    new_record["chiplets"] = new_chiplets
    return new_record


def _write_dropped_list(dropped: list[tuple[str, InfeasibleBumpError]]) -> None:
    """把丢弃的 system 编号 + 原因写入清单文件。"""
    DROPPED_LIST.parent.mkdir(parents=True, exist_ok=True)
    with DROPPED_LIST.open("w", encoding="utf-8") as f:
        f.write("# 丢弃的 system 编号 —— bump region 不可行 (hubump 过大, 芯片本体 body <= 0)\n")
        f.write("# 原因: 小 die 被塞了过高的互连带宽, 45um microbump 环宽超过芯片本体, 物理不可行。\n")
        f.write("# 字段: <system编号>\t<chiplet名>\t<footprint 尺寸 WxH mm>\t<单边 wireCount>\t<hubump mm>\n")
        if not dropped:
            f.write("# (无丢弃)\n")
        for sid, e in dropped:
            f.write(f"{sid}\t{e.name}\t{e.w:.3f}x{e.h:.3f}\t{e.wire_count:.0f}\t{e.hubump:.3f}\n")
    print(f"[bump] 丢弃清单已写入 {DROPPED_LIST} (共 {len(dropped)} 个)", flush=True)


def load_and_preprocess(start_sys: int, end_sys: int) -> None:
    k0 = (start_sys - 1) // CHUNK + 1
    k1 = (end_sys - 1) // CHUNK + 1
    total = 0
    dropped: list[tuple[str, InfeasibleBumpError]] = []
    for k in range(k0, k1 + 1):
        fp = SRC_DIR / f"chiplet_dataset_{k}.json"
        if not fp.exists():
            print(f"[bump] 跳过不存在的 {fp}", flush=True)
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        out_data = {}
        for sid, rec in data.items():
            m = re.fullmatch(r"system_(\d+)", sid)
            if not m:
                out_data[sid] = rec
                continue
            i = int(m.group(1))
            if start_sys <= i <= end_sys:
                try:
                    out_data[sid] = preprocess_record(rec)
                    total += 1
                except InfeasibleBumpError as e:
                    dropped.append((sid, e))
                    print(f"[bump] 丢弃 {sid}: {e}", flush=True)
            else:
                out_data[sid] = rec
        out_fp = OUT_DIR / f"chiplet_dataset_{k}.json"
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(json.dumps(out_data), encoding="utf-8")
        print(f"[bump] 写出 {out_fp} (共 {len(out_data)} systems)", flush=True)

    _write_dropped_list(dropped)
    print(f"[bump] 完成, 预处理 {total} 个布局 (丢弃 {len(dropped)} 个) -> {OUT_DIR}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=300001)
    ap.add_argument("--end", type=int, default=380000)
    args = ap.parse_args()
    load_and_preprocess(args.start, args.end)


if __name__ == "__main__":
    sys.exit(main())
