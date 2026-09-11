#!/usr/bin/env python3
"""复制 8 个 case 到 <case>_bump, 并把 .blocks 里 chiplet 尺寸包裹一层 hubump 环
(footprint = body + 2*hubump), 供 bump 环布局的 random_seed sweep 使用。

hubump 值来自 cases_hubump/<case>.json (wrap_hubump_cases.py 的输出, mm)。
只改 .blocks 的矩形顶点; .nets/.pl/.power 与两个 param json 原样拷贝(重命名 case 前缀)。
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path("/root/placement/ATPlace_pub")
CASES_DIR = ROOT / "cases"
HUB_DIR = ROOT / "cases_hubump"

CASES = ["acend910", "cpu-dram", "hp11_m", "multigpu", "syn1", "syn4", "syn6", "xerox8_m"]

BLOCK_RE = re.compile(r"^(\S+)\s+hardrectilinear\s+\d+\s+\(.*\)$")
PT_RE = re.compile(r"\(([^)]*)\)")


def rewrite_blocks(src: Path, dst: Path, hub_map: dict) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        m = BLOCK_RE.match(ln.strip())
        if not m:
            out.append(ln)
            continue
        name = m.group(1)
        if name not in hub_map:
            out.append(ln)
            continue
        verts = []
        for pt in PT_RE.findall(ln):
            x, y = pt.split(",")
            verts.append((float(x), float(y)))
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        hub_um = hub_map[name] * 1000.0
        nw = w + 2 * hub_um
        nh = h + 2 * hub_um
        out.append(f"{name} hardrectilinear 4 (0, 0) (0, {nh:.1f}) ({nw:.1f}, {nh:.1f}) ({nw:.1f}, 0)")
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    for case in CASES:
        src = CASES_DIR / case
        dst = CASES_DIR / f"{case}_bump"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        hub = json.loads((HUB_DIR / f"{case}.json").read_text(encoding="utf-8"))
        hub_map = {c["name"]: c["hubump"] for c in hub["chiplets"]}

        # 拷贝 + 重命名 case 文件与 param json
        for ext in (".blocks", ".nets", ".pl", ".power"):
            shutil.copy2(src / f"{case}{ext}", dst / f"{case}_bump{ext}")
        for j in ("Thermal-aware.json", "WL-driven.json"):
            shutil.copy2(src / j, dst / j)

        # 包裹 hubump: body -> footprint
        rewrite_blocks(src / f"{case}.blocks", dst / f"{case}_bump.blocks", hub_map)
        print(f"{case} -> {case}_bump  (chiplets wrapped with hubump)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
