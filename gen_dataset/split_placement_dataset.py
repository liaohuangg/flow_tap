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
            "method": "seeded random permutation by system id",
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
