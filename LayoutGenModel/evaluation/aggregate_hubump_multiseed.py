#!/usr/bin/env python3
"""Combine per-seed hubump inference metrics into one candidate table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    method_dir = args.method_dir.resolve()
    output = (args.output or method_dir / "all_candidates.csv").resolve()
    combined: list[dict[str, str]] = []

    def seed_number(path: Path) -> int:
        try:
            return int(path.name.removeprefix("seed_"))
        except ValueError:
            return 2**31 - 1

    for seed_dir in sorted(method_dir.glob("seed_*"), key=seed_number):
        metrics_path = seed_dir / "metrics.csv"
        names_path = seed_dir / "wirelength_bbox_results.csv"
        if not metrics_path.is_file():
            continue

        metrics = read_rows(metrics_path)
        names_by_idx: dict[str, str] = {}
        if names_path.is_file():
            for row in read_rows(names_path):
                if row.get("idx", "").isdigit():
                    names_by_idx[row["idx"]] = row.get("case_name", "")

        run_seed = seed_dir.name.removeprefix("seed_")
        for row in metrics:
            idx = row.get("idx", "")
            case_name = names_by_idx.get(idx, f"case_{idx}")
            prefix = f"{int(idx):02d}_{case_name}" if idx.isdigit() else case_name
            combined.append(
                {
                    "run_seed": run_seed,
                    "case_idx": idx,
                    "case_name": case_name,
                    **{key: value for key, value in row.items() if key != "idx"},
                    "placement_json": str(seed_dir / "placement" / f"{prefix}_placement.json"),
                    "placed_png": str(seed_dir / "samples" / f"{prefix}_placed.png"),
                    "seed_dir": str(seed_dir),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    if not combined:
        print(f"No completed seed metrics found under {method_dir}")
        return

    fieldnames = list(combined[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    print(f"Wrote {len(combined)} candidates to {output}")


if __name__ == "__main__":
    main()
