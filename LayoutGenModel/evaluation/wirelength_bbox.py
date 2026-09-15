"""
Calculate HPWL, bounding box area, and generation_time for a given atplace folder.
Usage:
    python -m evaluation.wirelength_bbox <folder_name>
Example:
    python -m evaluation.wirelength_bbox 525-thermal-3
"""
import csv
import json
import os
import sys
from pathlib import Path

CHIPLETFM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CHIPLETFM_ROOT.parent
INDEX_MAP_PATH = CHIPLETFM_ROOT / "benckmark" / "atplace_case1_10" / "index_map.json"
ATPLACE_DIR = Path(os.environ.get("ATPLACE_OUTPUT_DIR", CHIPLETFM_ROOT / "logs" / "output" / "atplace_case1_10"))


def load_index_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_wirecount_sum(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        case = json.load(f)
    return sum(conn["wireCount"] for conn in case.get("connections", []))


def resolve_input_file(entry):
    raw_path = Path(str(entry.get("input_file", "")))
    if raw_path.exists():
        return raw_path

    benchmark_name = entry.get("benchmark_name")
    if benchmark_name:
        benchmark_name = f"{benchmark_name}.json"
        candidates = []
        if os.environ.get("MTAP_ROOT"):
            candidates.append(Path(os.environ["MTAP_ROOT"]) / "benchmark" / "test_input" / benchmark_name)
        if os.environ.get("ATPLACE_BENCHMARK_ROOT"):
            candidates.append(Path(os.environ["ATPLACE_BENCHMARK_ROOT"]) / benchmark_name)
        candidates.extend([
            REPO_ROOT / "benchmark" / "ATPlace_json" / benchmark_name,
            REPO_ROOT / "MTAP" / "benchmark" / "test_input" / benchmark_name,
            Path.cwd() / "benchmark" / "ATPlace_json" / benchmark_name,
        ])
        for path in candidates:
            if path.exists():
                return path

    return None


def calc_bbox_area(placement_path):
    """Calculate the axis-aligned bounding box area of all chiplets in a placement file."""
    with open(placement_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for chiplet in data.get("chiplets", []):
        x = chiplet["x-position"]
        y = chiplet["y-position"]
        w = chiplet["width"]
        h = chiplet["height"]
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    return (max_x - min_x) * (max_y - min_y)


def export_wirelength_bbox_results(seed_dir, index_map_path=INDEX_MAP_PATH):
    seed_dir = Path(seed_dir)
    csv_path = seed_dir / "metrics.csv"
    placement_dir = seed_dir / "placement"

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return None
    if not placement_dir.is_dir():
        print(f"[ERROR] placement dir not found: {placement_dir}")
        return None

    # Derive the benchmark directory from the TASK name in the path
    # (<...>/logs/output/<task>/<method>/seed_N), not from a substring of the
    # method name.  The old check looked for the hyphenated "cases-hubump" in the
    # whole path, so it only worked when the *method* happened to be named e.g.
    # "cases-hubump-util50-..."; the task directory itself is "cases_hubump"
    # (underscore), so any other method name silently fell through to the
    # index_map fallback and raised IndexError.
    task_name = seed_dir.parent.parent.name
    benchmark_dir = REPO_ROOT / "benchmark" / task_name
    if benchmark_dir.is_dir():
        index_map = []
        for benchmark_path in sorted(benchmark_dir.glob("*.json")):
            with benchmark_path.open("r", encoding="utf-8") as f:
                benchmark = json.load(f)
            index_map.append({
                "benchmark_name": str(benchmark.get("case", benchmark_path.stem)),
                "input_file": str(benchmark_path),
            })
    else:
        index_map = load_index_map(index_map_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[WARN] No rows found in: {csv_path}")
        return None
    required_columns = {"idx", "generation_time"}
    if not required_columns.issubset(rows[0].keys()):
        missing = sorted(required_columns - set(rows[0].keys()))
        print(f"[WARN] Missing required columns in {csv_path}: {', '.join(missing)}")
        return None
    has_total_wirelength = "tap_total_wirelength" in rows[0]
    has_legacy_avg_wirelength = "tap_avg_wirelength" in rows[0]
    if not has_total_wirelength and not has_legacy_avg_wirelength:
        print(
            f"[WARN] Missing TAP wirelength column in {csv_path}: "
            "expected tap_total_wirelength (or legacy tap_avg_wirelength)"
        )
        return None

    # Read placement files sorted by name
    placement_files = sorted([f for f in os.listdir(placement_dir) if f.endswith(".json")])

    results = []
    total_wirelength_all = 0.0
    total_bbox_area = 0.0

    print(f"{'idx':>4}  {'case':<12}  {'tap_total_wl':>14}  {'wireCount':>10}  {'bbox_area':>12}  {'gen_time':>10}")
    print("-" * 82)

    for i, row in enumerate(rows):
        idx = int(row["idx"])
        gen_time = float(row["generation_time"])

        entry = index_map[idx]
        case_name = entry["benchmark_name"]
        input_file = resolve_input_file(entry)
        if input_file is None:
            print(f"[WARN] benchmark input not found for idx {idx} ({case_name}), wireCount set to zero")
            wirecount_sum = 0
        else:
            wirecount_sum = load_wirecount_sum(input_file)
        if has_total_wirelength:
            case_total = float(row["tap_total_wirelength"])
        else:
            # Backward compatibility for results produced before the evaluator
            # started exporting the CPLEX objective directly.
            case_total = float(row["tap_avg_wirelength"]) * wirecount_sum
        total_wirelength_all += case_total

        # Bbox area from corresponding placement file
        if i < len(placement_files):
            pl_path = placement_dir / placement_files[i]
            bbox_area = calc_bbox_area(pl_path)
            total_bbox_area += bbox_area
        else:
            bbox_area = 0.0
            print(f"[WARN] No placement file for idx {idx}")

        results.append((idx, case_name, case_total, wirecount_sum, bbox_area, gen_time))

        print(
            f"{idx:>4}  {case_name:<12}  {case_total:14.2f}  {wirecount_sum:10d}  {bbox_area:12.2f}  {gen_time:10.2f}"
        )

    print("-" * 82)
    avg_bbox = total_bbox_area / len(rows) if rows else 0
    print(f"      {'ALL':<12}  {total_wirelength_all:14.2f}              {total_bbox_area:12.2f}  {'---':>10}")
    print(f"      {'AVG':<12}  {'---':>14}              {avg_bbox:12.2f}  {'---':>10}")

    # Export results to file
    out_path = seed_dir / "wirelength_bbox_results.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "case_name", "tap_total_wirelength", "wireCount_sum",
                         "bbox_area", "generation_time"])
        writer.writerows(results)
        writer.writerow(["ALL", "", total_wirelength_all, "", total_bbox_area, ""])

    print(f"\nResults saved to: {out_path}")
    return str(out_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m evaluation.wirelength_bbox <folder_name>")
        sys.exit(1)

    folder_name = sys.argv[1]
    seed_dir = ATPLACE_DIR / folder_name / "seed_13012"
    export_wirelength_bbox_results(seed_dir)


if __name__ == "__main__":
    main()
