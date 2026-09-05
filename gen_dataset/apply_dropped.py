#!/usr/bin/env python3
"""
把 gen_dropped_layouts.py 生成的可行布局补全到:
  1. dataset/placement_dataset/placement_dataset/     footprint 记录 -> 覆盖(旧布局替换为新可行布局)
  2. dataset/placement_dataset/placement_dataset_tw/  body 记录 -> 新增(原本缺失, 重新计算 bump region 后补上)

与 apply_replacement.py 的区别: body 是「新增」而非「覆盖」(这些 system 原本不在 body 里)。

替换前把被改动的文件备份到 /tmp/replacement_dropped/backup/。

用法: python apply_dropped.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
DATASET = PROJECT / "Dataset" / "dataset"
PLACE_DATASET = DATASET / "placement_dataset" / "placement_dataset"
BODY_DATASET = DATASET / "placement_dataset" / "placement_dataset_tw"
TMP = Path("/tmp/replacement_dropped")
BACKUP = TMP / "backup"
CHUNK = 5000

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_bump_region import preprocess_record  # noqa: E402

CHIPLET_FIELDS = ["name", "x-position", "y-position", "width", "height", "rotation", "power"]
CONN_FIELDS = ["node1", "node2", "wireCount"]


def build_new_record(target: int, seed: int) -> dict:
    spec = json.loads((TMP / "input_test" / f"system_{seed}.json").read_text(encoding="utf-8"))
    place = json.loads((TMP / "placement" / f"system_{seed}.json").read_text(encoding="utf-8"))
    chiplets = [{k: c[k] for k in CHIPLET_FIELDS} for c in place["chiplets"]]
    connections = [{k: c[k] for k in CONN_FIELDS} for c in spec["connections"]]
    return {"system_id": f"system_{target}", "chiplets": chiplets, "connections": connections}


def backup(path: Path) -> None:
    dst = BACKUP / path.parent.name / path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(path, dst)


def main() -> None:
    mapping = json.loads((TMP / "mapping.json").read_text(encoding="utf-8"))
    targets = sorted(int(k) for k in mapping)
    BACKUP.mkdir(parents=True, exist_ok=True)

    new_records: dict[int, dict] = {}
    for t in targets:
        seed = int(mapping[str(t)])
        new_records[t] = build_new_record(t, seed)

    # 1) footprint 覆盖 (placement_dataset/placement_dataset)
    groups: dict[int, list[int]] = {}
    for t in targets:
        groups.setdefault((t - 1) // CHUNK + 1, []).append(t)
    for k, tlist in sorted(groups.items()):
        fp = PLACE_DATASET / f"chiplet_dataset_{k}.json"
        backup(fp)
        data = json.loads(fp.read_text(encoding="utf-8"))
        for t in tlist:
            key = f"system_{t}"
            assert key in data, f"{key} 不在 {fp}(footprint 应存在)"
            data[key] = new_records[t]
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"[apply-dropped] footprint {fp.name}: 覆盖 {len(tlist)} 个", flush=True)

    # 2) body 新增 (placement_dataset/placement_dataset_tw)
    body_groups: dict[int, list[int]] = {}
    for t in targets:
        body_groups.setdefault((t - 1) // CHUNK + 1, []).append(t)
    for k, tlist in sorted(body_groups.items()):
        fp = BODY_DATASET / f"chiplet_dataset_{k}.json"
        backup(fp)
        data = json.loads(fp.read_text(encoding="utf-8"))
        for t in tlist:
            key = f"system_{t}"
            assert key not in data, f"{key} 已存在于 {fp}(应为缺失)"
            data[key] = preprocess_record(new_records[t])  # footprint -> body
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"[apply-dropped] body {fp.name}: 新增 {len(tlist)} 个", flush=True)

    print(f"[apply-dropped] 全部完成: 补全 {len(targets)} 个缺失布局", flush=True)


if __name__ == "__main__":
    main()
