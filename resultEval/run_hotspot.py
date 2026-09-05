"""run_hotspot.py

Generate TAP-2.5D HotSpot input files from a placement JSON and run HotSpot.

Inputs
- placement JSON: contains chiplets with fields:
  name, x-position, y-position, width, height, rotation, power
  where x-position/y-position are the chiplet bottom-left coordinates.
  width/height are already in the final orientation (do NOT apply rotation).
- placement JSON also contains connections with node1/node2/wireCount, used to
  derive the interconnect matrix and hubump width.

Outputs (written under tap2.5d_hoteval/config/<case_name>/)
- <case>L0_Substrate.flp
- <case>L1_C4Layer.flp
- <case>L2_Interposer.flp
- <case>sim.flp
- <case>L3.flp
- <case>L4.flp
- <case>L3_UbumpLayer.flp
- <case>L4_ChipLayer.flp
- <case>L5_TIM.flp
- <case>layers.lcf
- <case>.ptrace
- new_hotspot.config
- <case>.steady, <case>.grid.steady: HotSpot outputs

HotSpot binary/config/template paths are fixed to this directory tree:
- util/hotspot
- util/hotspot.config
- util/fill_space.py
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Import required by user constraint.
from util.fill_space import fill_space  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
UTIL_DIR = SCRIPT_DIR / "util"
HOTSPOT_BIN = UTIL_DIR / "hotspot"
HOTSPOT_TEMPLATE_CONFIG = UTIL_DIR / "hotspot.config"
OUTPUT_ROOT = SCRIPT_DIR / "config"
GRANULARITY_MM = 1.0
GRANULARITY_M = GRANULARITY_MM / 1000.0
EDGE_MARGIN_M = GRANULARITY_M / 2.0


@dataclass(frozen=True)
class Chiplet:
    json_name: str
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    rotation: float
    power_w: float


def _slug_case_name(json_path: Path) -> str:
    return json_path.stem


def _read_chiplets_and_connections(json_path: Path) -> tuple[list[Chiplet], list[dict]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "chiplets" not in data or not isinstance(data["chiplets"], list):
        raise ValueError(f"JSON missing 'chiplets' list: {json_path}")

    chiplets: list[Chiplet] = []
    for i, c in enumerate(data["chiplets"]):
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", f"Chiplet_{i}"))
        try:
            x = float(c["x-position"])
            y = float(c["y-position"])
            w = float(c["width"])
            h = float(c["height"])
        except Exception as e:
            raise ValueError(f"Invalid chiplet geometry at index {i}: {e}")
        if w <= 0 or h <= 0:
            raise ValueError(f"Chiplet {name}: width/height must be > 0 (w={w}, h={h})")
        rotation = float(c.get("rotation", 0.0))
        power = float(c.get("power", 0.0))
        chiplets.append(Chiplet(json_name=name, x_mm=x, y_mm=y, w_mm=w, h_mm=h, rotation=rotation, power_w=power))

    if not chiplets:
        raise ValueError(f"No chiplets found in JSON: {json_path}")

    connections = data.get("connections", [])
    if connections and not isinstance(connections, list):
        raise ValueError(f"JSON 'connections' must be a list if present: {json_path}")
    return chiplets, connections


def _mix_material(edge_diameter: float, core_diameter: float, core_spec_heat: float, other_spec_heat: float, core_resistivity: float, other_resistivity: float) -> str:
    aratio = (edge_diameter / core_diameter) * (edge_diameter / core_diameter) - 1
    resistivity = (1 + aratio) * core_resistivity * other_resistivity / (other_resistivity + aratio * core_resistivity)
    spec_heat = (core_spec_heat + aratio * other_spec_heat) / (1 + aratio)
    return f"\t{spec_heat}\t{resistivity}\n"


def _layer_materials() -> dict[str, str]:
    underfill = "\t2.32E+06\t0.625\n"
    silicon = "\t1.75E+06\t0.01\n"
    mat_c4 = _mix_material(0.000600, 0.000250, 3494400, 2320000, 0.0025, 0.625)
    mat_tsv = _mix_material(0.000050, 0.000010, 3494400, 1750000, 0.0025, 0.01)
    mat_ubump = _mix_material(0.000045, 0.000025, 3494400, 2320000, 0.0025, 0.625)
    return {
        "underfill": underfill,
        "silicon": silicon,
        "mat_c4": mat_c4,
        "mat_tsv": mat_tsv,
        "mat_ubump": mat_ubump,
    }


MATERIALS = _layer_materials()


def _build_connection_matrix(chiplets: list[Chiplet], connections: list[dict]) -> list[list[float]]:
    name_to_index = {chiplet.json_name: i for i, chiplet in enumerate(chiplets)}
    n = len(chiplets)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for entry in connections:
        if not isinstance(entry, dict):
            continue
        node1 = entry.get("node1")
        node2 = entry.get("node2")
        if node1 not in name_to_index or node2 not in name_to_index:
            print(f"[warn] skipping connection with unknown nodes: {node1!r}, {node2!r}", file=sys.stderr)
            continue
        i = name_to_index[node1]
        j = name_to_index[node2]
        if i == j:
            raise ValueError(f"a link from and to the same chiplet is not allowed: {node1!r}")
        wire_count = float(entry.get("wireCount", 0.0))
        matrix[i][j] += wire_count
        matrix[j][i] += wire_count

    return matrix


def _compute_hubumps(chiplets: list[Chiplet], connection_matrix: list[list[float]]) -> list[float]:
    hubumps: list[float] = []
    n = len(chiplets)
    for i, chiplet in enumerate(chiplets):
        s = 0.0
        for j in range(n):
            s += connection_matrix[i][j] + connection_matrix[j][i]
        h = 1
        w_stretch = 0.045 * h
        while ((chiplet.w_mm + chiplet.h_mm) * 2 * w_stretch + 4 * w_stretch * w_stretch) / 0.045 / 0.045 < s:
            h += 1
            w_stretch = 0.045 * h
            if h > 1000:
                raise ValueError("microbump is too high to be a feasible case")
        hubumps.append(w_stretch)
    return hubumps


def _layout_bbox_mm(chiplets: list[Chiplet], hubumps: list[float]) -> tuple[float, float, float, float]:
    lefts = [c.x_mm - h for c, h in zip(chiplets, hubumps)]
    bottoms = [c.y_mm - h for c, h in zip(chiplets, hubumps)]
    rights = [c.x_mm + c.w_mm + h for c, h in zip(chiplets, hubumps)]
    tops = [c.y_mm + c.h_mm + h for c, h in zip(chiplets, hubumps)]
    return min(lefts), min(bottoms), max(rights), max(tops)


def _prefixed(out_dir: Path, case: str, stem: str) -> Path:
    return out_dir / f"{case}{stem}"


def _write_flp_header(f, title: str) -> None:
    f.write(f"# {title}\n")
    f.write("# Line Format: <unit-name>\t<width>\t<height>\t<left-x>\t<bottom-y>\t[<specific-heat>]\t[<resistivity>]\n")
    f.write("# all dimensions are in meters\n")
    f.write("# comment lines begin with a '#' \n")
    f.write("# comments and empty lines are ignored\n\n")


def _write_simple_layer(path: Path, title: str, unit_name: str, side_m: float, material: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        _write_flp_header(f, title)
        if material:
            f.write(f"{unit_name}\t{side_m}\t{side_m}\t0.0\t0.0{material}")
        else:
            f.write(f"{unit_name}\t{side_m}\t{side_m}\t0.0\t0.0\n")


def _write_l3_l4_sim(
    out_dir: Path,
    case: str,
    chiplets: list[Chiplet],
    hubumps: list[float],
    shift_x_mm: float,
    shift_y_mm: float,
    intp_size_mm: float,
) -> tuple[Path, Path, Path]:
    l3_base = _prefixed(out_dir, case, "L3")
    l4_base = _prefixed(out_dir, case, "L4")
    sim_base = _prefixed(out_dir, case, "sim")

    side_m = intp_size_mm / 1000.0
    x_offset0_m = EDGE_MARGIN_M
    y_offset0_m = EDGE_MARGIN_M

    with l3_base.with_suffix(".flp").open("w", encoding="utf-8") as l3_f, \
        l4_base.with_suffix(".flp").open("w", encoding="utf-8") as l4_f, \
        sim_base.with_suffix(".flp").open("w", encoding="utf-8") as sim_f:
        _write_flp_header(l3_f, "Floorplan for Microbump Layer ")
        _write_flp_header(l4_f, "Floorplan for Chip Layer")

        l3_edge = MATERIALS["mat_ubump"]
        l4_edge = MATERIALS["mat_ubump"]
        silicon = MATERIALS["silicon"]
        mat_ubump = MATERIALS["mat_ubump"]

        l3_f.write(f"Edge_0\t{side_m - GRANULARITY_M}\t{EDGE_MARGIN_M}\t{EDGE_MARGIN_M}\t0{l3_edge}")
        l3_f.write(f"Edge_1\t{side_m - GRANULARITY_M}\t{EDGE_MARGIN_M}\t{EDGE_MARGIN_M}\t{side_m - EDGE_MARGIN_M}{l3_edge}")
        l3_f.write(f"Edge_2\t{EDGE_MARGIN_M}\t{side_m}\t0\t0{l3_edge}")
        l3_f.write(f"Edge_3\t{EDGE_MARGIN_M}\t{side_m}\t{side_m - EDGE_MARGIN_M}\t0{l3_edge}")

        l4_f.write(f"Edge_0\t{side_m - GRANULARITY_M}\t{EDGE_MARGIN_M}\t{EDGE_MARGIN_M}\t0{l4_edge}")
        l4_f.write(f"Edge_1\t{side_m - GRANULARITY_M}\t{EDGE_MARGIN_M}\t{EDGE_MARGIN_M}\t{side_m - EDGE_MARGIN_M}{l4_edge}")
        l4_f.write(f"Edge_2\t{EDGE_MARGIN_M}\t{side_m}\t0\t0{l4_edge}")
        l4_f.write(f"Edge_3\t{EDGE_MARGIN_M}\t{side_m}\t{side_m - EDGE_MARGIN_M}\t0{l4_edge}")

        for i, (chiplet, hubump_mm) in enumerate(zip(chiplets, hubumps)):
            unit_x_m = (chiplet.x_mm - hubump_mm + shift_x_mm) / 1000.0
            unit_y_m = (chiplet.y_mm - hubump_mm + shift_y_mm) / 1000.0
            chip_x_m = (chiplet.x_mm + shift_x_mm) / 1000.0
            chip_y_m = (chiplet.y_mm + shift_y_mm) / 1000.0
            if hubump_mm > 0:
                l3_f.write(f"Ubump_{4 * i}\t{(chiplet.w_mm + hubump_mm) / 1000.0}\t{hubump_mm / 1000.0}\t{unit_x_m}\t{unit_y_m}{mat_ubump}")
                l3_f.write(f"Ubump_{4 * i + 1}\t{hubump_mm / 1000.0}\t{(chiplet.h_mm + hubump_mm) / 1000.0}\t{unit_x_m}\t{unit_y_m + hubump_mm / 1000.0}{mat_ubump}")
                l3_f.write(f"Ubump_{4 * i + 2}\t{hubump_mm / 1000.0}\t{(chiplet.h_mm + hubump_mm) / 1000.0}\t{unit_x_m + (chiplet.w_mm + hubump_mm) / 1000.0}\t{unit_y_m}{mat_ubump}")
                l3_f.write(f"Ubump_{4 * i + 3}\t{(chiplet.w_mm + hubump_mm) / 1000.0}\t{hubump_mm / 1000.0}\t{unit_x_m + hubump_mm / 1000.0}\t{unit_y_m + (chiplet.h_mm + hubump_mm) / 1000.0}{mat_ubump}")
                l4_f.write(f"Ubump_{4 * i}\t{(chiplet.w_mm + hubump_mm) / 1000.0}\t{hubump_mm / 1000.0}\t{unit_x_m}\t{unit_y_m}{silicon}")
                l4_f.write(f"Ubump_{4 * i + 1}\t{hubump_mm / 1000.0}\t{(chiplet.h_mm + hubump_mm) / 1000.0}\t{unit_x_m}\t{unit_y_m + hubump_mm / 1000.0}{silicon}")
                l4_f.write(f"Ubump_{4 * i + 2}\t{hubump_mm / 1000.0}\t{(chiplet.h_mm + hubump_mm) / 1000.0}\t{unit_x_m + (chiplet.w_mm + hubump_mm) / 1000.0}\t{unit_y_m}{silicon}")
                l4_f.write(f"Ubump_{4 * i + 3}\t{(chiplet.w_mm + hubump_mm) / 1000.0}\t{hubump_mm / 1000.0}\t{unit_x_m + hubump_mm / 1000.0}\t{unit_y_m + (chiplet.h_mm + hubump_mm) / 1000.0}{silicon}")

            l3_f.write(f"Chiplet_{i}\t{chiplet.w_mm / 1000.0}\t{chiplet.h_mm / 1000.0}\t{unit_x_m + hubump_mm / 1000.0}\t{unit_y_m + hubump_mm / 1000.0}{mat_ubump}")
            l4_f.write(f"Chiplet_{i}\t{chiplet.w_mm / 1000.0}\t{chiplet.h_mm / 1000.0}\t{unit_x_m + hubump_mm / 1000.0}\t{unit_y_m + hubump_mm / 1000.0}{silicon}")
            sim_f.write(f"Unit_{i}\t{(chiplet.w_mm + 2 * hubump_mm) / 1000.0}\t{(chiplet.h_mm + 2 * hubump_mm) / 1000.0}\t{unit_x_m}\t{unit_y_m}\n")

    l3_filled = _prefixed(out_dir, case, "L3_UbumpLayer")
    l4_filled = _prefixed(out_dir, case, "L4_ChipLayer")
    fill_space(x_offset0_m, side_m - x_offset0_m, y_offset0_m, side_m - y_offset0_m, str(sim_base), str(l3_base), str(l3_filled))
    fill_space(x_offset0_m, side_m - x_offset0_m, y_offset0_m, side_m - y_offset0_m, str(sim_base), str(l4_base), str(l4_filled))
    return l3_filled.with_suffix(".flp"), l4_filled.with_suffix(".flp"), sim_base.with_suffix(".flp")


def _read_flp_unit_names(flp_path: Path) -> list[str]:
    names: list[str] = []
    for raw in flp_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"\s+", s)
        if parts:
            names.append(parts[0])
    return names


def _write_ptrace_from_flp(flp_path: Path, ptrace_path: Path, powers_by_name: dict[str, float]) -> None:
    names = _read_flp_unit_names(flp_path)
    if not names:
        raise ValueError(f"No modules parsed from flp: {flp_path}")

    powers: list[float] = []
    for name in names:
        if name in powers_by_name:
            powers.append(float(powers_by_name[name]))
        else:
            powers.append(0.0)

    ptrace_path.parent.mkdir(parents=True, exist_ok=True)
    with ptrace_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(names) + "\n")
        f.write("\t".join(str(p) for p in powers) + "\n")


def _replace_flag_value(config_text: str, flag: str, new_value: float) -> str:
    pattern = re.compile(rf"(^[ \t]*{re.escape(flag)}[ \t]+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.MULTILINE)

    def repl(m: re.Match) -> str:
        return f"{m.group(1)}{new_value:.6f}"

    out, n = pattern.subn(repl, config_text, count=1)
    if n != 1:
        raise ValueError(f"Flag not found or ambiguous in config: {flag} (matches={n})")
    return out


def _derive_hotspot_config(template_config: Path, out_config: Path, intp_size_mm: float) -> None:
    cfg = template_config.read_text(encoding="utf-8", errors="replace")
    size_spreader = 2.0 * intp_size_mm / 1000.0
    size_heatsink = 2.0 * size_spreader
    r_convec = 0.1 * 0.06 * 0.06 / (size_heatsink * size_heatsink)

    cfg = _replace_flag_value(cfg, "-s_spreader", size_spreader)
    cfg = _replace_flag_value(cfg, "-s_sink", size_heatsink)
    cfg = _replace_flag_value(cfg, "-r_convec", r_convec)

    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(cfg, encoding="utf-8")


def _run_hotspot(
    hotspot_bin: Path,
    config_file: Path,
    flp_file: Path,
    ptrace_file: Path,
    steady_file: Path,
    grid_steady_file: Path,
    layers_lcf: Path,
    model_type: str = "grid",
) -> tuple[int, str, str]:
    cmd = [
        str(hotspot_bin),
        "-c",
        str(config_file),
        "-f",
        str(flp_file),
        "-p",
        str(ptrace_file),
        "-steady_file",
        str(steady_file),
        "-grid_steady_file",
        str(grid_steady_file),
        "-model_type",
        model_type,
        "-detailed_3D",
        "on",
        "-grid_layer_file",
        str(layers_lcf),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


def _parse_max_temp_c(stdout: str) -> float | None:
    tokens = stdout.split()
    floats: list[float] = []
    for tok in tokens[3::2]:
        try:
            floats.append(float(tok))
        except ValueError:
            continue
    if not floats:
        for tok in re.split(r"\s+", stdout.strip()):
            try:
                floats.append(float(tok))
            except ValueError:
                continue
    if not floats:
        return None
    return max(floats) - 273.15


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate HotSpot inputs from placement JSON and run HotSpot")
    ap.add_argument("--json", required=True, help="Placement JSON path (chiplets with bottom-left coords; units are mm)")
    ap.add_argument("--out_root", default=str(OUTPUT_ROOT), help="Output root dir (default: tap2.5d_hoteval/config)")
    ap.add_argument("--case", default=None, help="Case name subdir (default: json stem)")
    ap.add_argument("--no_run", action="store_true", help="Only generate files; do not invoke HotSpot")
    ap.add_argument("--model_type", default="grid", choices=["grid", "block"], help="HotSpot model_type")

    args = ap.parse_args()

    json_path = Path(args.json).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    case = args.case or _slug_case_name(json_path)
    out_dir = Path(args.out_root).resolve() / case
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HOTSPOT_BIN.is_file():
        raise FileNotFoundError(f"HotSpot binary not found: {HOTSPOT_BIN}")
    if not HOTSPOT_TEMPLATE_CONFIG.is_file():
        raise FileNotFoundError(f"HotSpot template config not found: {HOTSPOT_TEMPLATE_CONFIG}")

    chiplets, connections = _read_chiplets_and_connections(json_path)
    connection_matrix = _build_connection_matrix(chiplets, connections)
    hubumps = _compute_hubumps(chiplets, connection_matrix)

    min_left_mm, min_bottom_mm, max_right_mm, max_top_mm = _layout_bbox_mm(chiplets, hubumps)
    span_w_mm = max_right_mm - min_left_mm
    span_h_mm = max_top_mm - min_bottom_mm
    intp_size_mm = max(span_w_mm, span_h_mm) + GRANULARITY_MM

    # Center the whole layout within the interposer square while preserving
    # the required edge margin (GRANULARITY/2) on each side.
    slack_x_mm = intp_size_mm - span_w_mm
    slack_y_mm = intp_size_mm - span_h_mm
    shift_x_mm = (GRANULARITY_MM / 2.0) - min_left_mm + (slack_x_mm - GRANULARITY_MM) / 2.0
    shift_y_mm = (GRANULARITY_MM / 2.0) - min_bottom_mm + (slack_y_mm - GRANULARITY_MM) / 2.0

    # Layer files with TAP-2.5D-style names prefixed by the case name.
    _write_simple_layer(_prefixed(out_dir, case, "L0_Substrate.flp"), "Floorplan for Substrate Layer with size " + str(intp_size_mm / 1000.0) + "x" + str(intp_size_mm / 1000.0) + " m", "Substrate", intp_size_mm / 1000.0)
    _write_simple_layer(_prefixed(out_dir, case, "L1_C4Layer.flp"), "Floorplan for C4 Layer ", "C4Layer", intp_size_mm / 1000.0, MATERIALS["mat_c4"])
    _write_simple_layer(_prefixed(out_dir, case, "L2_Interposer.flp"), "Floorplan for Silicon Interposer Layer", "Interposer", intp_size_mm / 1000.0, MATERIALS["mat_tsv"])

    l3_filled, l4_filled, _ = _write_l3_l4_sim(out_dir, case, chiplets, hubumps, shift_x_mm, shift_y_mm, intp_size_mm)
    _write_simple_layer(_prefixed(out_dir, case, "L5_TIM.flp"), "Floorplan for TIM Layer ", "TIM", intp_size_mm / 1000.0)

    layers_lcf = _prefixed(out_dir, case, "layers.lcf")
    with layers_lcf.open("w", encoding="utf-8") as lcf:
        lcf.write("# File Format:\n")
        lcf.write("#<Layer Number>\n")
        lcf.write("#<Lateral heat flow Y/N?>\n")
        lcf.write("#<Power Dissipation Y/N?>\n")
        lcf.write("#<Specific heat capacity in J/(m^3K)>\n")
        lcf.write("#<Resistivity in (m-K)/W>\n")
        lcf.write("#<Thickness in m>\n")
        lcf.write("#<floorplan file>\n")
        lcf.write("\n# Layer 0: substrate\n0\nY\nN\n1.06E+06\n3.33\n0.0002\n" + str(_prefixed(out_dir, case, "L0_Substrate.flp")) + "\n")
        lcf.write("\n# Layer 1: Epoxy SiO2 underfill with C4 copper pillar\n1\nY\nN\n2.32E+06\n0.625\n0.00007\n" + str(_prefixed(out_dir, case, "L1_C4Layer.flp")) + "\n")
        lcf.write("\n# Layer 2: silicon interposer\n2\nY\nN\n1.75E+06\n0.01\n0.00011\n" + str(_prefixed(out_dir, case, "L2_Interposer.flp")) + "\n")
        lcf.write("\n# Layer 3: Underfill with ubump\n3\nY\nN\n2.32E+06\n0.625\n1.00E-05\n" + str(l3_filled) + "\n")
        lcf.write("\n# Layer 4: Chip layer\n4\nY\nY\n1.75E+06\n0.01\n0.00015\n" + str(l4_filled) + "\n")
        lcf.write("\n# Layer 5: TIM\n5\nY\nN\n4.00E+06\n0.25\n2.00E-05\n" + str(_prefixed(out_dir, case, "L5_TIM.flp")) + "\n")

    derived_cfg = out_dir / "new_hotspot.config"
    _derive_hotspot_config(HOTSPOT_TEMPLATE_CONFIG, derived_cfg, intp_size_mm)

    ptrace = _prefixed(out_dir, case, ".ptrace")
    powers_by_name = {f"Chiplet_{i}": chiplet.power_w for i, chiplet in enumerate(chiplets)}
    _write_ptrace_from_flp(l4_filled, ptrace, powers_by_name)

    if args.no_run:
        print(f"[ok] Generated files under: {out_dir}")
        return

    steady = _prefixed(out_dir, case, ".steady")
    grid_steady = _prefixed(out_dir, case, ".grid.steady")

    rc, stdout, stderr = _run_hotspot(
        hotspot_bin=HOTSPOT_BIN,
        config_file=derived_cfg,
        flp_file=l4_filled,
        ptrace_file=ptrace,
        steady_file=steady,
        grid_steady_file=grid_steady,
        layers_lcf=layers_lcf,
        model_type=args.model_type,
    )

    if rc != 0:
        cmd = [
            str(HOTSPOT_BIN),
            "-c",
            str(derived_cfg),
            "-f",
            str(l4_filled),
            "-p",
            str(ptrace),
            "-steady_file",
            str(steady),
            "-grid_steady_file",
            str(grid_steady),
            "-model_type",
            args.model_type,
            "-detailed_3D",
            "on",
            "-grid_layer_file",
            str(layers_lcf),
        ]
        print("[hotspot] command:")
        print(" ".join(shlex.quote(x) for x in cmd))
        raise RuntimeError(f"HotSpot failed (rc={rc})\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    tmax_c = _parse_max_temp_c(stdout)
    if tmax_c is not None:
        print(f"[hotspot] max_temp = {tmax_c:.3f} C")
    else:
        print("[hotspot] completed (max temp not parsed)")


if __name__ == "__main__":
    main()


# python /root/placement/flow_GCN/tap2.5d_hoteval/run_hotspot.py \
#     --json /root/placement/flow_GCN/benchmark/placement/base/seed_13012/placement/00_Case1_placement.json \
#     --out_root /root/placement/flow_GCN/tap2.5d_hoteval/config \
#     --case 00_Case1_placement \          
#     --model_type grid  
