#!/usr/bin/env python3
"""Summarize paired multi-seed training ablations without cherry-picking."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("thermal_max_c", "thermal_mean_c", "tap_avg_wirelength")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def sample_std(values):
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def cluster_bootstrap_ci(case_values: dict[str, list[float]], draws=10000, seed=20260912):
    names = sorted(case_values)
    if not names:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        selected = [rng.choice(names) for _ in names]
        estimates.append(mean(mean(case_values[name]) for name in selected))
    estimates.sort()
    return estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=CSV")
    parser.add_argument("--baseline", default="no_proxy")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data: dict[str, list[dict]] = {}
    for item in args.input:
        label, sep, raw_path = item.partition("=")
        if not sep:
            parser.error(f"invalid --input {item!r}; expected LABEL=CSV")
        rows = read_csv(Path(raw_path))
        for row in rows:
            row["run_seed"] = int(row["run_seed"])
            row["case_idx"] = int(row["case_idx"])
            for metric in METRICS + ("legality_2", "expanded_legality_2"):
                row[metric] = float(row[metric])
            row["model"] = label
        data[label] = rows

    if args.baseline not in data:
        parser.error(f"baseline {args.baseline!r} is not one of {sorted(data)}")

    key = lambda row: (row["case_name"], row["run_seed"])
    baseline = {key(row): row for row in data[args.baseline]}
    baseline_keys = set(baseline)
    for label, rows in data.items():
        keys = {key(row) for row in rows}
        if keys != baseline_keys:
            missing = sorted(baseline_keys - keys)
            extra = sorted(keys - baseline_keys)
            raise SystemExit(f"{label}: paired sample mismatch; missing={missing[:5]}, extra={extra[:5]}")

    combined = []
    paired = []
    summaries = []
    best_of_k = []
    for label, rows in data.items():
        combined.extend(rows)
        delta_by_metric: dict[str, dict[str, list[float]]] = {
            metric: defaultdict(list) for metric in METRICS
        }
        wins = {metric: 0 for metric in METRICS}
        for row in rows:
            base = baseline[key(row)]
            out = {"model": label, "case_name": row["case_name"], "run_seed": row["run_seed"]}
            for metric in METRICS:
                delta = row[metric] - base[metric]
                pct = 100.0 * delta / base[metric] if base[metric] else float("nan")
                out[f"{metric}_delta"] = delta
                out[f"{metric}_pct"] = pct
                delta_by_metric[metric][row["case_name"]].append(pct)
                wins[metric] += int(delta < 0.0)
            paired.append(out)

        summary = {
            "model": label,
            "num_candidates": len(rows),
            "num_cases": len({row["case_name"] for row in rows}),
            "num_seeds": len({row["run_seed"] for row in rows}),
        }
        for metric in METRICS:
            values = [row[metric] for row in rows]
            pcts = [record[f"{metric}_pct"] for record in paired if record["model"] == label]
            low, high = cluster_bootstrap_ci(delta_by_metric[metric])
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = sample_std(values)
            summary[f"{metric}_paired_pct_mean"] = mean(pcts)
            summary[f"{metric}_paired_pct_ci95_low"] = low
            summary[f"{metric}_paired_pct_ci95_high"] = high
            summary[f"{metric}_win_rate"] = wins[metric] / len(rows)
        summary["legality_exact_rate"] = mean(row["legality_2"] >= 1.0 - 1e-9 for row in rows)
        summary["expanded_legality_exact_rate"] = mean(
            row["expanded_legality_2"] >= 1.0 - 1e-9 for row in rows
        )
        summaries.append(summary)

        by_case = defaultdict(list)
        for row in rows:
            by_case[row["case_name"]].append(row)
        best_of_k.append({
            "model": label,
            "num_seeds": summary["num_seeds"],
            "thermal_max_c_best_mean": mean(min(group, key=lambda r: r["thermal_max_c"])["thermal_max_c"] for group in by_case.values()),
            "tap_at_thermal_best_mean": mean(min(group, key=lambda r: r["thermal_max_c"])["tap_avg_wirelength"] for group in by_case.values()),
            "tap_avg_wirelength_best_mean": mean(min(group, key=lambda r: r["tap_avg_wirelength"])["tap_avg_wirelength"] for group in by_case.values()),
            "thermal_at_tap_best_mean": mean(min(group, key=lambda r: r["tap_avg_wirelength"])["thermal_max_c"] for group in by_case.values()),
        })

    out = args.output_dir.resolve()
    write_csv(out / "ablation_all_candidates.csv", combined)
    write_csv(out / "ablation_paired_deltas.csv", paired)
    write_csv(out / "ablation_summary.csv", summaries)
    write_csv(out / "ablation_best_of_k.csv", best_of_k)

    lines = [
        "# Training-proxy ablation",
        "",
        f"Baseline: `{args.baseline}`. All comparisons use identical case/seed pairs.",
        "Inference-time thermal and wirelength guidance are disabled.",
        "",
        "| model | peak C | peak change | TAP | TAP change | peak wins | TAP wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['thermal_max_c_mean']:.3f} | "
            f"{row['thermal_max_c_paired_pct_mean']:+.2f}% "
            f"[{row['thermal_max_c_paired_pct_ci95_low']:+.2f}, {row['thermal_max_c_paired_pct_ci95_high']:+.2f}] | "
            f"{row['tap_avg_wirelength_mean']:.3f} | "
            f"{row['tap_avg_wirelength_paired_pct_mean']:+.2f}% "
            f"[{row['tap_avg_wirelength_paired_pct_ci95_low']:+.2f}, {row['tap_avg_wirelength_paired_pct_ci95_high']:+.2f}] | "
            f"{100*row['thermal_max_c_win_rate']:.1f}% | {100*row['tap_avg_wirelength_win_rate']:.1f}% |"
        )
    (out / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ablation report written to {out / 'ablation_report.md'}")


if __name__ == "__main__":
    main()
