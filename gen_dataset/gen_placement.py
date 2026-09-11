#!/usr/bin/env python3
"""
用 gen_legal_pla_greedy.py 为 placement_dataset_tw 里的 system 生成布局坐标,
并把坐标写回 chiplet_dataset_{k}.json 的 x-position/y-position/width/height/rotation。

关键约定 (重要):
  - 布局用 **footprint 尺寸** (footprint = body + 2*hubump) 参与放置, 保证 footprint 互不重叠。
  - placer 输出的是 footprint 左下角 + footprint 最终长宽(可能被旋转) + rotation。
  - 回写时还原 body:
        body.x   = footprint.x + hubump
        body.y   = footprint.y + hubump
        body.w   = footprint.w - 2*hubump
        body.h   = footprint.h - 2*hubump
        rotation = placer 的 rotation (footprint 旋转, body 同步旋转)
  - hubump 不变 (compute_hubump 对 w/h 对称, 与旋转无关)。

用法:
  python gen_placement.py --start 380001 --end 400000 --workers 24
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
PLACEMENT_TW = PROJECT / "Dataset" / "dataset" / "placement_dataset" / "placement_dataset_tw"
CHUNK = 5000

# 让 gen_legal_pla_greedy / tool 可导入 (脚本位于 gen_dataset/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_legal_pla_greedy as g  # noqa: E402

# 与旧 pipeline 的 matrix_to_connections 一致: EMIB 字段只用于满足 placer 输入校验,
# 核心布局只依赖 wireCount + 尺寸, 不依赖这些 EMIB 值。
LINEAR_IO, MAX_REACH, AREA_IO = 40.0, 100.0, 80.0


def _to_footprint(rec: dict):
    """把一个 placement_dataset 记录转成 placer 输入 (footprint 尺寸 + EMIB 连接)。"""
    nodes = []
    for c in rec["chiplets"]:
        nodes.append(g.ChipletNode(
            name=c["name"],
            dimensions={"x": c["width"] + 2 * c["hubump"],
                        "y": c["height"] + 2 * c["hubump"]},
            phys=[],
            power=c["power"],
        ))
    edge_map = {}
    for e in rec["connections"]:
        wc = e["wireCount"]
        el = wc / LINEAR_IO
        emw = MAX_REACH - 2.0 * (wc / AREA_IO) / el
        ebw = (wc / AREA_IO) / el
        a, b = e["node1"], e["node2"]
        if a > b:
            a, b = b, a
        edge_map[(a, b)] = {
            "node1": a, "node2": b, "wireCount": float(wc), "conn_type": 0,
            "EMIBType": "interfaceC", "EMIB_length": el,
            "EMIB_width": 2 * ebw + emw, "EMIB_max_width": emw, "EMIB_bump_width": ebw,
        }
    return nodes, edge_map


def _place_worker(args):
    """单 system 布局: 返回 (sid, footprint_chiplets) 或 (sid, error_str)。"""
    sid, rec, tmp_dir = args
    try:
        nodes, edge_map = _to_footprint(rec)
        out_path = Path(tmp_dir) / f"system_{sid}.json"
        g._place_one_case(nodes, edge_map, out_path)
        out = json.loads(out_path.read_text(encoding="utf-8"))
        out_path.unlink(missing_ok=True)
        return sid, out["chiplets"], None
    except Exception as e:  # noqa: BLE001
        return sid, None, f"{type(e).__name__}: {e}"


def _apply_coords(rec: dict, footprint_chiplets: list) -> dict:
    """footprint 坐标 -> body 坐标, 写回记录。"""
    fp = {c["name"]: c for c in footprint_chiplets}
    new_chiplets = []
    for c in rec["chiplets"]:
        hu = c["hubump"]
        f = fp[c["name"]]
        new_chiplets.append({
            "name": c["name"],
            "x-position": round(f["x-position"] + hu, 6),
            "y-position": round(f["y-position"] + hu, 6),
            "width": round(f["width"] - 2 * hu, 6),
            "height": round(f["height"] - 2 * hu, 6),
            "rotation": f["rotation"],
            "power": c["power"],
            "hubump": hu,
        })
    new_rec = dict(rec)
    new_rec["chiplets"] = new_chiplets
    return new_rec


def load_records(start_sys: int, end_sys: int) -> dict:
    records: dict[int, dict] = {}
    k0 = (start_sys - 1) // CHUNK + 1
    k1 = (end_sys - 1) // CHUNK + 1
    for k in range(k0, k1 + 1):
        fp = PLACEMENT_TW / f"chiplet_dataset_{k}.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for sid, rec in data.items():
            i = int(sid.split("_")[1])
            if start_sys <= i <= end_sys:
                records[i] = rec
    return records


def write_records(records: dict[int, dict]) -> None:
    from collections import defaultdict
    chunks: dict[int, dict] = defaultdict(dict)
    for i, rec in sorted(records.items()):
        chunks[(i - 1) // CHUNK + 1][f"system_{i}"] = rec
    for k, data in sorted(chunks.items()):
        fp = PLACEMENT_TW / f"chiplet_dataset_{k}.json"
        fp.write_text(json.dumps(data), encoding="utf-8")
        print(f"[placement] 写回 {fp} ({len(data)} systems)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=380001)
    ap.add_argument("--end", type=int, default=400000)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    records = load_records(args.start, args.end)
    sids = sorted(records.keys())
    print(f"[placement] 读取 {len(sids)} systems ({args.start}..{args.end}), workers={args.workers}",
          flush=True)
    if not sids:
        print("[placement] 无数据, 退出")
        return

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="pla_out_")
    tasks = [(i, records[i], tmp_dir) for i in sids]

    t0 = time.time()
    done = 0
    failed: list[tuple[int, str]] = []
    results: dict[int, list] = {}
    with multiprocessing.Pool(args.workers) as pool:
        for sid, fp_chiplets, err in pool.imap_unordered(_place_worker, tasks, chunksize=8):
            if err is not None:
                failed.append((sid, err))
                print(f"[placement] system_{sid}: ERROR {err}", flush=True)
            else:
                results[sid] = fp_chiplets
            done += 1
            if done % 2000 == 0:
                print(f"[placement] 进度 {done}/{len(sids)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    # 回写 body 坐标
    updated = 0
    for sid in sids:
        if sid not in results:
            continue
        records[sid] = _apply_coords(records[sid], results[sid])
        updated += 1

    write_records(records)
    print(f"[placement] DONE: 成功 {updated}/{len(sids)}, 失败 {len(failed)}, "
          f"墙钟 {time.time() - t0:.0f}s", flush=True)
    if failed:
        print(f"[placement] 失败样例: {failed[:10]}", flush=True)


if __name__ == "__main__":
    main()
