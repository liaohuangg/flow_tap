#!/usr/bin/env python3
"""把 hubump 包裹后的 <case>.json 直接转成 cases/<case>_bump/ 目录 (Bookshelf 用例)。

make_bump_cases.py 只做后半程 (cases/<case>/ -> cases/<case>_bump/), 而 cases/<case>/
本身在仓库里没有任何生成脚本 —— 它们当初是手工/外部做出来的。本脚本补上前半程:
    <src>/<case>.json  ->  cases/<case>_bump/
其中 .blocks 仍交给 make_bump_cases.rewrite_blocks() 做 hubump 包裹, 与既有 8 个
_bump 用例完全同一条路径。

格式规则 (全部由现有 cases/ 反推并逐个校验过, 见 --check):

  <case>_bump.blocks
    NumSoftRectangularBlocks : 0 / NumHardRectilinearBlocks : N / NumTerminals : 0
    每个 chiplet 一行:
      <name> hardrectilinear 4 (0, 0) (0, H) (W, H) (W, 0)
    W/H = footprint_w/h * 1000 (µm, 即 body + 2*hubump)。

  <case>_bump.nets
    NumNets = Σ wireCount, NumPins = 2*NumNets。
    每个 connection 展开成 wireCount 条 NetDegree : 2 的 net, 两行分别是
    node1 / node2 的 pin, pin 名恒为 "B"。
    每个 chiplet 的 pin 沿自身周长均匀铺开: pitch = 周长 / 该 chiplet 的 pin 总数,
    第 j 个 pin 落在弧长 (j+0.5)*pitch 处; 走法是 底边(右->左) -> 左边(下->上)
    -> 顶边(左->右) -> 右边(上->下); 偏移按 坐标/尺寸*100 写成百分数, 5 位小数。
    各 chiplet 的 pin 按 connections 的先后顺序依次分配。
    校验: 8 个已有 case 里 6 个逐字节一致; hp11_m / syn1 / syn4 / syn6 只有个别行
    第 5 位小数差 1 (落在 .5 平局上, 折合约 1e-5 个百分点的偏移, 对布局无影响)。

  <case>_bump.pl     : <name>\\t0\\t0
  <case>_bump.power  : <name>\\t<power>
  Thermal-aware.json / WL-driven.json
    = 既有 <template>_bump 用例的模板 (逐字节相同), 只换 interposer_size:
            side = sqrt(2 * Σ(footprint_w * footprint_h))   (µm)
    即中介层面积 = chiplet (含 bump 环) 总面积的两倍。

用法:
    python make_bump_cases_from_json.py hp6_m hp8_m xerox6_m xerox7_m
    python make_bump_cases_from_json.py --src /path/to/cases_hubump --out cases --check
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_bump_cases as mbc  # noqa: E402  (rewrite_blocks: body -> footprint)

HERE = Path(__file__).resolve().parent
DEF_SRC = Path("/root/placement/flow_tap/benchmark/cases_hubump")
DEF_TEMPLATE = "acend910_bump"          # 模板 _bump 用例 (8 个 _bump 用例的参数模板一致)

def block_header(n: int) -> list:
    return ["NumSoftRectangularBlocks : 0",
            f"NumHardRectilinearBlocks : {n}",
            "NumTerminals : 0",
            ""]


# ---------------------------------------------------------------- .blocks
def gen_blocks(chiplets: list, footprint: bool) -> str:
    """footprint=True 用 footprint_w/h (即 bump 环包裹后), 否则用 body 尺寸。"""
    lines = block_header(len(chiplets))
    for c in chiplets:
        w = (c["footprint_w"] if footprint else c["width"]) * 1000.0
        h = (c["footprint_h"] if footprint else c["height"]) * 1000.0
        lines.append(f"{c['name']} hardrectilinear 4 (0, 0) (0, {h:.1f}) "
                     f"({w:.1f}, {h:.1f}) ({w:.1f}, 0)")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ .nets
def _pin_pct(W: float, H: float, n_pins: int, j: int) -> tuple:
    """第 j 个 pin 的 (x%, y%)。W/H 单位 µm。

    在 µm 精度上算坐标再换百分数 —— 与原始生成器一致, 8 个已有 case 里 6 个能
    逐字节复现。剩下的差异只在第 5 位小数的 .5 平局上 (见 --check / 文档)。
    """
    s = (j + 0.5) * (2 * (W + H) / n_pins)          # 沿周长的弧长
    if s <= W:
        x, y = W / 2 - s, -H / 2
    elif s <= W + H:
        x, y = -W / 2, -H / 2 + (s - W)
    elif s <= 2 * W + H:
        x, y = -W / 2 + (s - W - H), H / 2
    else:
        x, y = W / 2, H / 2 - (s - 2 * W - H)
    return x / W * 100, y / H * 100


def gen_nets(chiplets: list, connections: list) -> str:
    W = {c["name"]: float(c["width"]) * 1000.0 for c in chiplets}
    H = {c["name"]: float(c["height"]) * 1000.0 for c in chiplets}
    n_pins = {n: 0 for n in W}
    for c in connections:
        n_pins[c["node1"]] += c["wireCount"]
        n_pins[c["node2"]] += c["wireCount"]

    total = sum(c["wireCount"] for c in connections)
    out = [f"NumNets : {total}", f"NumPins : {2 * total}", ""]
    cur = {n: 0 for n in W}
    for c in connections:
        for _ in range(c["wireCount"]):
            out.append("NetDegree : 2")
            for n in (c["node1"], c["node2"]):
                x, y = _pin_pct(W[n], H[n], n_pins[n], cur[n])
                cur[n] += 1
                out.append(f"{n} B : %{x:.5f} %{y:.5f}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ params
def gen_params(case: str, chiplets: list, out_dir: Path, template: str) -> tuple:
    """拷贝模板 param json, 只替换 interposer_size。返回 (side_um, area_um2)。"""
    tpl = HERE / "cases" / template
    area = sum(float(c["footprint_w"]) * 1000.0 * float(c["footprint_h"]) * 1000.0
               for c in chiplets)
    side = math.sqrt(2.0 * area)
    for name in ("Thermal-aware.json", "WL-driven.json"):
        d = json.loads((tpl / name).read_text(encoding="utf-8"))
        d["interposer_size"] = [side, side]
        # 既有 param json 结尾不带换行, 这里保持一致
        (out_dir / name).write_text(
            json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return side, area


# ------------------------------------------------------------------- main
def convert(case: str, src: Path, out_root: Path, template: str, force: bool) -> dict:
    data = json.loads((src / f"{case}.json").read_text(encoding="utf-8"))
    chiplets, connections = data["chiplets"], data["connections"]
    dst = out_root / f"{case}_bump"
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} 已存在; 加 --force 才会覆盖")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # 1) 先写 body 版 .blocks, 再交给 make_bump_cases 包裹 hubump (与既有流程同一条路径)
    blocks = dst / f"{case}_bump.blocks"
    blocks.write_text(gen_blocks(chiplets, footprint=False), encoding="utf-8")
    hub_map = {c["name"]: c["hubump"] for c in chiplets}
    mbc.rewrite_blocks(blocks, blocks, hub_map)

    # 2) nets / pl / power
    (dst / f"{case}_bump.nets").write_text(gen_nets(chiplets, connections), encoding="utf-8")
    (dst / f"{case}_bump.pl").write_text(
        "".join(f"{c['name']}\t0\t0\n" for c in chiplets), encoding="utf-8")
    (dst / f"{case}_bump.power").write_text(
        "".join(f"{c['name']}\t{c.get('power', 0)}\n" for c in chiplets), encoding="utf-8")

    # 3) 两个 param json
    side, area = gen_params(case, chiplets, dst, template)
    return {"case": case, "dir": dst, "chiplets": len(chiplets),
            "nets": sum(c["wireCount"] for c in connections),
            "side_um": side, "area_mm2": area / 1e6}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", help="case 名 (不含 _bump)")
    ap.add_argument("--src", type=Path, default=DEF_SRC, help=f"hubump json 目录 (默认 {DEF_SRC})")
    ap.add_argument("--out", type=Path, default=HERE / "cases", help="cases 根目录")
    ap.add_argument("--template", default=DEF_TEMPLATE, help=f"param 模板 _bump 用例 (默认 {DEF_TEMPLATE})")
    ap.add_argument("--force", action="store_true", help="目标已存在时先删除")
    ap.add_argument("--check", action="store_true", help="转换后顺带做一致性自检")
    a = ap.parse_args()

    recs = []
    for case in a.cases:
        rec = convert(case, a.src, a.out, a.template, a.force)
        recs.append(rec)
        print(f"{case:10s} -> {rec['dir']}  chiplets={rec['chiplets']} nets={rec['nets']} "
              f"中介层={rec['side_um']:.4f}µm 边长 (面积={rec['area_mm2']:.2f}mm²)")

    if a.check:
        print("\n--- 自检 ---")
        ok = True
        for rec in recs:
            d, case = rec["dir"], rec["case"]
            blocks = (d / f"{case}_bump.blocks").read_text()
            nets = (d / f"{case}_bump.nets").read_text().splitlines()
            pl = (d / f"{case}_bump.pl").read_text().splitlines()
            pw = (d / f"{case}_bump.power").read_text().splitlines()
            para = json.loads((d / "Thermal-aware.json").read_text())
            wlb = json.loads((d / "WL-driven.json").read_text())
            # a) 行数与 NumNets/NumPins 自洽
            n_hdr = int(nets[0].split(":")[1])
            p_hdr = int(nets[1].split(":")[1])
            n_deg = sum(1 for l in nets if l.startswith("NetDegree"))
            n_pin = sum(1 for l in nets if re.match(r"^\S+ B : ", l))
            # b) blocks 尺寸 == footprint, 顶点闭合
            dims = []
            for ln in blocks.splitlines():
                m = re.match(r"^(\S+)\s+hardrectilinear\s+\d+\s+(\(.*\))$", ln.strip())
                if not m:
                    continue
                v = [tuple(map(float, p.split(","))) for p in mbc.PT_RE.findall(m.group(2))]
                xs = [x for x, _ in v]
                ys = [y for _, y in v]
                dims.append((m.group(1), max(xs) - min(xs), max(ys) - min(ys)))
            # c) 中介层面积 == 2×chiplet 总面积
            tot = sum(w * h for _, w, h in dims)
            ratio = para["interposer_size"][0] * para["interposer_size"][1] / tot
            # d) 所有 pin 偏移都落在 [-50, 50]
            offs = []
            for l in nets:
                offs.extend(float(t) for t in re.findall(r"%(-?[\d.]+)", l))
            nchip = int(blocks.splitlines()[1].split(":")[1])
            good = (n_hdr == n_deg == rec["nets"] and p_hdr == n_pin == 2 * rec["nets"]
                    and len(dims) == nchip == len(pl) == len(pw) == rec["chiplets"]
                    and abs(ratio - 2.0) < 1e-9 and abs(max(offs)) <= 50.0
                    and para["interposer_size"] == wlb["interposer_size"])
            ok &= good
            print(f"  {case}_bump: nets {n_hdr}/{n_deg}/{n_pin}  chiplet {nchip} "
                  f" 中介层/总面积={ratio:.9f}  偏移范围=[{min(offs):.5f},{max(offs):.5f}]  "
                  f"{'OK' if good else 'FAIL'}")
        print("自检:", "全部通过" if ok else "有问题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
