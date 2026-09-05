#!/usr/bin/env python3
"""
把 gen_replacement_layouts.py 生成的 40 个合法布局, 原位替换到:
  1. dataset/placement_dataset/      40 个非法 system 的 footprint 记录 -> 新合法布局
  2. dataset/placement_dataset/placement_dataset_tw/ 后 8w 中 4 个非法 system -> 新 body 记录(补上被丢弃的)
  3. config/chiplet_dataset_*.zip    40 个非法 system 的 .cfg -> 新 .cfg

替换前把被改动的文件备份到 /tmp/replacement/backup/。

用法: python apply_replacement.py
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
DS = PROJECT / "Dataset"
DATASET = DS / "dataset"
CONFIG = DS / "config"
PLACE_DATASET = DATASET / "placement_dataset"
BODY_DATASET = DATASET / "placement_dataset" / "placement_dataset_tw"
TMP = Path("/tmp/replacement")
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
    dst = BACKUP / path.name
    if not dst.exists():
        shutil.copy2(path, dst)


def _rewrite_zip(zp: Path, replacements: dict[str, str]) -> None:
    """重写 zip, 替换若干 entry 的内容; 其余条目原样保留。"""
    tmp = zp.with_suffix(".tmp.zip")
    with zipfile.ZipFile(zp, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in replacements:
                data = replacements[item.filename].encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(zp)


def main() -> None:
    mapping = json.loads((TMP / "mapping.json").read_text(encoding="utf-8"))
    targets = sorted(int(k) for k in mapping)
    BACKUP.mkdir(parents=True, exist_ok=True)

    new_records: dict[int, dict] = {}
    new_cfgs: dict[int, str] = {}
    for t in targets:
        seed = int(mapping[str(t)])
        new_records[t] = build_new_record(t, seed)
        new_cfgs[t] = (TMP / "cfg" / f"system_{seed}.cfg").read_text(encoding="utf-8")

    # 1) 替换 placement_dataset (按文件分组)
    groups: dict[int, list[int]] = {}
    for t in targets:
        groups.setdefault((t - 1) // CHUNK + 1, []).append(t)
    for k, tlist in sorted(groups.items()):
        fp = PLACE_DATASET / f"chiplet_dataset_{k}.json"
        backup(fp)
        data = json.loads(fp.read_text(encoding="utf-8"))
        for t in tlist:
            key = f"system_{t}"
            assert key in data, f"{key} 不在 {fp}"
            data[key] = new_records[t]
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"[apply] placement_dataset/chiplet_dataset_{k}.json: 替换 {len(tlist)} 个", flush=True)

    # 2) 替换 placement_dataset/placement_dataset_tw (仅后 8w 的 4 个: body 数据只覆盖 300001..380000)
    body_groups: dict[int, list[int]] = {}
    for t in targets:
        if t >= 300001:
            body_groups.setdefault((t - 1) // CHUNK + 1, []).append(t)
    for k, tlist in sorted(body_groups.items()):
        fp = BODY_DATASET / f"chiplet_dataset_{k}.json"
        backup(fp)
        data = json.loads(fp.read_text(encoding="utf-8"))
        for t in tlist:
            key = f"system_{t}"
            assert key not in data, f"{key} 不应已存在于 {fp}(应为被丢弃状态)"
            data[key] = preprocess_record(new_records[t])  # footprint -> body
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"[apply] placement_dataset/placement_dataset_tw/chiplet_dataset_{k}.json: 补上 {len(tlist)} 个", flush=True)

    # 3) 替换 config zip
    zip_groups: dict[int, dict[str, str]] = {}
    for t in targets:
        k = (t - 1) // CHUNK + 1
        zip_groups.setdefault(k, {})[f"system_{t}.cfg"] = new_cfgs[t]
    for k, repl in sorted(zip_groups.items()):
        zp = CONFIG / f"chiplet_dataset_{k}.zip"
        backup(zp)
        _rewrite_zip(zp, repl)
        print(f"[apply] config/chiplet_dataset_{k}.zip: 替换 {len(repl)} 个 .cfg", flush=True)

    print(f"[apply] 全部完成: 替换 {len(targets)} 个非法布局", flush=True)


if __name__ == "__main__":
    main()
