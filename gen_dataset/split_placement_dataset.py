"""Create deterministic train/validation/test manifests for placement cases.

The source JSON files remain untouched.  Each output text file contains one
numeric system id per line, sorted for efficient chunked loading.  Membership
is assigned after a seeded shuffle, so the three subsets are random and fully
reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLACEMENT_DIR = (
    REPO_ROOT / "Dataset" / "dataset" / "placement_dataset" / "placement_dataset_tw"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Dataset" / "splits" / "placement_first_300k"
FILE_PATTERN = re.compile(r"chiplet_dataset_(\d+)\.json$")
SYSTEM_PATTERN = re.compile(r"system_(\d+)$")


def _source_files(placement_dir: Path, max_system_id: int) -> list[Path]:
    max_file_index = (max_system_id - 1) // 5000 + 1
    indexed = []
    for path in placement_dir.glob("chiplet_dataset_*.json"):
        match = FILE_PATTERN.fullmatch(path.name)
        if match and int(match.group(1)) <= max_file_index:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    expected = list(range(1, max_file_index + 1))
    actual = [index for index, _path in indexed]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise FileNotFoundError(f"missing source placement chunks: {missing}")
    return [path for _index, path in indexed]


def collect_system_ids(placement_dir: Path, limit: int) -> tuple[list[int], list[str]]:
    ids = []
    source_names = []
    for path in _source_files(placement_dir, limit):
        with path.open("r", encoding="utf-8") as handle:
            systems = json.load(handle)
        source_names.append(path.name)
        for key in systems:
            match = SYSTEM_PATTERN.fullmatch(key)
            if match:
                system_id = int(match.group(1))
                if 1 <= system_id <= limit:
                    ids.append(system_id)

    duplicates = len(ids) - len(set(ids))
    selected = sorted(set(ids))
    expected = list(range(1, limit + 1))
    if duplicates or selected != expected:
        missing = sorted(set(expected) - set(selected))
        extra = sorted(set(selected) - set(expected))
        raise ValueError(
            f"first {limit} cases are not a complete unique range: "
            f"duplicates={duplicates}, missing={missing[:20]}, extra={extra[:20]}"
        )
    return selected, source_names


def _load_groups(placement_dir: Path) -> dict[int, int] | None:
    """读 `groups.json` (行 -> case)。没有就返回 None, 走老的"一行一 case"路径。

    当主语料一个 case 有多行时, 必须按 case 分组切分: 否则同一个 case 的不同布局会同时
    落进 train 和 val, 验证集里出现与训练集几乎相同的布局, 指标虚高。
    """
    path = placement_dir / "groups.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row): int(case) for row, case in payload["row_to_case"].items()}


def _grouped_split(
    row_to_case: dict[int, int], seed: int, train_ratio: float, val_ratio: float
) -> dict[str, list[int]]:
    """按 case 洗牌分组, 再把行展开。行数比例与目标尽量接近, 但不切开任何 case。"""
    rows_of: dict[int, list[int]] = {}
    for row, case in row_to_case.items():
        rows_of.setdefault(case, []).append(row)
    cases = sorted(rows_of)
    random.Random(seed).shuffle(cases)

    total = len(row_to_case)
    train_target = total * train_ratio
    val_target = total * val_ratio
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for case in cases:
        rows = rows_of[case]
        if len(splits["train"]) < train_target:
            splits["train"] += rows
        elif len(splits["val"]) < val_target:
            splits["val"] += rows
        else:
            splits["test"] += rows
    return {name: sorted(ids) for name, ids in splits.items()}


def _write_ids(path: Path, ids: list[int]) -> str:
    content = "".join(f"{system_id}\n" for system_id in ids)
    path.write_text(content, encoding="utf-8", newline="\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def create_split(
    placement_dir: Path,
    output_dir: Path,
    limit: int = 300_000,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0):
        raise ValueError("invalid train/validation ratios")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    system_ids, source_names = collect_system_ids(placement_dir, limit)
    row_to_case = _load_groups(placement_dir)
    if row_to_case is not None:
        if sorted(row_to_case) != system_ids:
            raise ValueError(
                f"groups.json 的行集与放置目录里的 system_* 不一致: "
                f"n_rows={len(row_to_case)}, n_systems={len(system_ids)}"
            )
        splits = _grouped_split(row_to_case, seed, train_ratio, val_ratio)
    else:
        shuffled = list(system_ids)
        random.Random(seed).shuffle(shuffled)
        train_count = int(limit * train_ratio)
        val_count = int(limit * val_ratio)
        splits = {
            "train": sorted(shuffled[:train_count]),
            "val": sorted(shuffled[train_count : train_count + val_count]),
            "test": sorted(shuffled[train_count + val_count :]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = {
        name: _write_ids(output_dir / f"{name}.txt", ids)
        for name, ids in splits.items()
    }
    try:
        placement_label = placement_dir.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        placement_label = str(placement_dir.resolve())
    manifest = {
        "schema_version": 1,
        "dataset": "placement_dataset_tw",
        "placement_dir": placement_label,
        "selection": {
            "rule": "numeric system id in the inclusive range [1, limit]",
            "limit": limit,
            "first_system_id": 1,
            "last_system_id": limit,
            "source_files": source_names,
        },
        "split": {
            "method": ("seeded random permutation of cases, rows expanded afterwards"
                       if row_to_case is not None
                       else "seeded random permutation by system id"),
            "grouped_by_case": row_to_case is not None,
            "n_cases": (len(set(row_to_case.values())) if row_to_case is not None
                        else len(system_ids)),
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1.0 - train_ratio - val_ratio, 12),
            "counts": {name: len(ids) for name, ids in splits.items()},
        },
        "files": {
            name: {"path": f"{name}.txt", "sha256": checksums[name]}
            for name in splits
        },
    }
    # JSON is valid YAML and keeps this metadata readable without an added dependency.
    (output_dir / "manifest.yaml").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-dir", type=Path, default=DEFAULT_PLACEMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    manifest = create_split(
        placement_dir=args.placement_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
