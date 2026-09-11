"""
Generate fastTM thermal resistance tables for benchmark JSON cases.

This creates:
  fastTM/configs/benchmark_<case>.cfg
  fastTM/outputs/benchmark_<case>/Chiplet_<i>.rself
  fastTM/outputs/benchmark_<case>/Chiplet_<i>.rmutu

The reward path in reward_cal.py is unchanged; it will consume these tables.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
FASTM_DIR = ROOT / "fastTM"
if str(FASTM_DIR) not in sys.path:
    sys.path.insert(0, str(FASTM_DIR))

# compute_temp.py imports pandas in this repository, but characterization does
# not need it. Keep this for optional post-checks and consistent imports.
pandas_stub = types.ModuleType("pandas")
pandas_stub.__spec__ = None
sys.modules.setdefault("pandas", pandas_stub)

import char_thermal_r  # noqa: E402


def _normalize_path(path_value: str | Path) -> Path:
    path_text = str(path_value).replace("\\", "/")
    if os.name == "nt" and path_text.startswith("/mnt/") and len(path_text) > 6:
        drive = path_text[5]
        rest = path_text[6:].lstrip("/")
        return Path(f"{drive.upper()}:/{rest}")
    return Path(path_text)


def _chiplet_name(chiplet_data: Dict, index: int) -> str:
    return str(chiplet_data.get("name") or chiplet_data.get("id") or f"Chiplet_{index}")


def _connection_matrix(chiplet_names: List[str], connections: List) -> List[List[int]]:
    name_to_idx = {name: i for i, name in enumerate(chiplet_names)}
    matrix = [[0 for _ in chiplet_names] for _ in chiplet_names]
    for conn in connections:
        if isinstance(conn, dict):
            node1 = conn.get("node1") or conn.get("source") or conn.get("from")
            node2 = conn.get("node2") or conn.get("target") or conn.get("to")
            wires = int(conn.get("wireCount", conn.get("weight", 1)))
        else:
            node1, node2 = conn[0], conn[1]
            wires = int(conn[2]) if len(conn) >= 3 else 1

        if node1 not in name_to_idx or node2 not in name_to_idx:
            raise ValueError(f"Unknown chiplet in connection: {node1}, {node2}")
        i, j = name_to_idx[node1], name_to_idx[node2]
        matrix[i][j] += wires
        matrix[j][i] += wires
    return matrix


def write_fasttm_config(
    json_path: Path,
    intp_size: float,
    granularity: float,
    overwrite: bool,
) -> Tuple[Path, Path]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chiplets = data.get("chiplets") or data.get("dies") or []
    if not chiplets:
        raise KeyError(f"{json_path} must contain chiplets or dies")

    names = [_chiplet_name(chiplet, i) for i, chiplet in enumerate(chiplets)]
    widths = [float(chiplet.get("width", 10.0)) for chiplet in chiplets]
    heights = [float(chiplet.get("height", 10.0)) for chiplet in chiplets]
    powers = [float(chiplet.get("power", 100.0)) for chiplet in chiplets]
    matrix = _connection_matrix(names, data.get("connections", []))

    cfg_path = FASTM_DIR / "configs" / f"benchmark_{json_path.stem}.cfg"
    output_dir = FASTM_DIR / "outputs" / f"benchmark_{json_path.stem}"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg_path.exists() and not overwrite:
        return cfg_path, output_dir

    cfg = configparser.ConfigParser()
    cfg["general"] = {
        "path": str(output_dir) + "/",
        "placer_granularity": str(granularity),
        "initial_placement": "given",
        "decay": "0.9",
    }
    cfg["interposer"] = {
        "intp_type": "passive",
        "intp_size": str(intp_size),
        "link_type": "nppl",
    }
    cfg["chiplets"] = {
        "chiplet_count": str(len(chiplets)),
        "widths": ",".join(map(str, widths)),
        "heights": ",".join(map(str, heights)),
        "powers": ",".join(map(str, powers)),
        "x": ",".join([str(intp_size / 2.0)] * len(chiplets)),
        "y": ",".join([str(intp_size / 2.0)] * len(chiplets)),
        "connections": ";\n\t\t\t".join(",".join(map(str, row)) for row in matrix),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        cfg.write(f)
    return cfg_path, output_dir


def generate_tables(cfg_path: Path, output_dir: Path, force: bool) -> None:
    # config.read_config() in fastTM reads sys.argv globally. Keep only argv[0]
    # while using char_thermal_r helpers directly.
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]]
    try:
        import config

        system = config.read_config(str(cfg_path))
        wh_to_group, unique_widths, unique_heights = char_thermal_r.unique_WH(
            system.width, system.height
        )
        sys_name = str(cfg_path.with_suffix(""))

        for group_id, (width, height) in enumerate(zip(unique_widths, unique_heights)):
            chiplet_name = f"Chiplet{group_id}"
            rself = output_dir / f"{chiplet_name}.rself"
            rmutu = output_dir / f"{chiplet_name}.rmutu"
            if force or not (rself.exists() and rmutu.exists()):
                char_thermal_r.char_self_r(
                    system.path,
                    sys_name,
                    chiplet_name,
                    system.intp_size,
                    width,
                    height,
                    100.0,
                )
                char_thermal_r.char_mutu_r(
                    system.path,
                    sys_name,
                    chiplet_name,
                    system.intp_size,
                    width,
                    height,
                    100.0,
                )

        for chiplet_id, size in enumerate(zip(system.width, system.height)):
            group_id = wh_to_group[size]
            shutil.copyfile(
                output_dir / f"Chiplet{group_id}.rself",
                output_dir / f"Chiplet_{chiplet_id}.rself",
            )
            shutil.copyfile(
                output_dir / f"Chiplet{group_id}.rmutu",
                output_dir / f"Chiplet_{chiplet_id}.rmutu",
            )
    finally:
        sys.argv = old_argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Benchmark JSON path")
    parser.add_argument("--intp-size", type=float, default=50.0)
    parser.add_argument("--granularity", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = _normalize_path(args.json).resolve()
    cfg_path, output_dir = write_fasttm_config(
        json_path,
        intp_size=args.intp_size,
        granularity=args.granularity,
        overwrite=args.overwrite_config,
    )
    print(f"Config: {cfg_path}")
    print(f"Output: {output_dir}")
    generate_tables(cfg_path, output_dir, force=args.force)
    print("Thermal tables generated.")


if __name__ == "__main__":
    main()
