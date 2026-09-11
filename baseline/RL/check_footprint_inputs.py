#!/usr/bin/env python
"""Check that the RL placement env uses each benchmark's declared occupancy.

For every benchmark JSON under ``examples/`` this script resolves the environment
and reports, per chiplet, which occupied (bump-expanded) envelope placement
actually uses:

  * ``json_footprint`` - declared footprint_w/footprint_h (preferred)
  * ``json_hubump``    - declared hubump
  * ``recomputed``     - halo derived from input connectivity (last resort)

It then places every chiplet with random legal actions and asserts the resulting
layout is legal against that envelope: every occupied rectangle lies inside the
canvas and no two rectangles overlap.

Exit status is non-zero when any check fails. ``--require_full_placement`` also
treats an incomplete random packing as a failure, which is a canvas/search
property rather than a footprint property.

Usage:
    python check_footprint_inputs.py                       # all example cases
    python check_footprint_inputs.py --case acend910
    python check_footprint_inputs.py --episodes 20 --mode json_only
    python check_footprint_inputs.py --json path/to/case.json --verbose
    python check_footprint_inputs.py --mode recompute      # legacy A/B
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

from env import (  # noqa: E402
    FOOTPRINT_MODES,
    ChipletPlacementEnv,
    create_env_from_json,
)

TOL = 1e-6


def _fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _envelope_is_inside(env: ChipletPlacementEnv) -> list[str]:
    problems = []
    for chip_id, chip in env.state.layout.items():
        left, bottom, right, top = env._occupied_bounds(chip, chip_id)
        if (left < -TOL or bottom < -TOL
                or right > env.max_width + TOL or top > env.max_height + TOL):
            problems.append(
                f"{chip_id}: envelope ({_fmt(left)},{_fmt(bottom)})-"
                f"({_fmt(right)},{_fmt(top)}) escapes canvas "
                f"{_fmt(env.max_width)}x{_fmt(env.max_height)}"
            )
    return problems


def _envelopes_overlap(env: ChipletPlacementEnv) -> list[str]:
    bounds = {
        chip_id: env._occupied_bounds(chip, chip_id)
        for chip_id, chip in env.state.layout.items()
    }
    problems = []
    ids = list(bounds)
    for i, first in enumerate(ids):
        for second in ids[i + 1:]:
            a_left, a_bottom, a_right, a_top = bounds[first]
            b_left, b_bottom, b_right, b_top = bounds[second]
            if (a_left < b_right - 1e-9 and a_right > b_left + 1e-9
                    and a_bottom < b_top - 1e-9 and a_top > b_bottom + 1e-9):
                problems.append(f"{first} envelope overlaps {second}")
    return problems


def check_case(json_path: Path, args: argparse.Namespace) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    declared = {
        (c.get("name") or c.get("id")): c
        for c in json.loads(json_path.read_text(encoding="utf-8"))["chiplets"]
    }

    env = create_env_from_json(
        str(json_path),
        grid_resolution=args.grid_resolution,
        max_width=args.canvas,
        max_height=args.canvas,
        min_overlap=0.5,
        placement_reward=1.0,
        adjacency_reward=1.0,
        extra_adjacency_reward=100,
        compact=3,
        min_wirelength_reward_scale=0,
        terminal_util_reward_scale=100.0,
        max_auto_grid_resolution=args.max_auto_grid_resolution,
        lenbase_samples=0,
        footprint_mode=args.mode,
        verify_footprint=args.verify_footprint,
    )

    print(f"\n=== {json_path.stem} ===")
    print(f"  footprint_mode={env.footprint_mode}  canvas="
          f"{_fmt(env.max_width)}x{_fmt(env.max_height)}  chiplets={env.num_chiplets}")

    for entry in env.footprint_report():
        chip_id = entry["chip_id"]
        declared_chip = declared.get(chip_id, {})
        declared_fp = (declared_chip.get("footprint_w"), declared_chip.get("footprint_h"))
        flag = "" if entry["matches_connectivity"] else "  <-- differs from recomputed"
        line = (
            f"  {chip_id:<9} body {_fmt(entry['body_w'])}x{_fmt(entry['body_h'])}"
            f"  halo {_fmt(entry['halo_x'])}/{_fmt(entry['halo_y'])}"
            f"  envelope {_fmt(entry['footprint_w'])}x{_fmt(entry['footprint_h'])}"
            f"  [{entry['source']}]{flag}"
        )
        if args.verbose:
            line += (
                f"\n            recomputed halo {_fmt(entry['recomputed_halo'])}"
                f" -> {_fmt(entry['recomputed_footprint_w'])}x"
                f"{_fmt(entry['recomputed_footprint_h'])}"
                f"; json hubump={declared_chip.get('hubump')}"
                f"; json footprint={declared_fp}"
            )
        print(line)

        # A declared envelope must be reproduced verbatim.
        if args.mode != "recompute" and declared_fp[0] and declared_fp[1]:
            if (abs(entry["footprint_w"] - float(declared_fp[0])) > TOL
                    or abs(entry["footprint_h"] - float(declared_fp[1])) > TOL):
                ok = False
                messages.append(
                    f"{json_path.stem}/{chip_id}: env envelope "
                    f"{entry['footprint_w']}x{entry['footprint_h']} != declared "
                    f"{declared_fp[0]}x{declared_fp[1]}"
                )
        elif args.mode == "json_first" and declared_chip.get("hubump"):
            hubump = float(declared_chip["hubump"])
            if (abs(entry["halo_x"] - hubump) > TOL or abs(entry["halo_y"] - hubump) > TOL):
                ok = False
                messages.append(
                    f"{json_path.stem}/{chip_id}: halo {entry['halo_x']}/{entry['halo_y']} "
                    f"!= declared hubump {hubump}"
                )

    # Rotating an envelope must swap its axes, symmetric or not.
    for entry in env.footprint_report():
        chip_id = entry["chip_id"]
        straight = env.footprint_of(chip_id, 0)
        rotated = env.footprint_of(chip_id, 1)
        if (abs(rotated[0] - straight[1]) > TOL or abs(rotated[1] - straight[0]) > TOL):
            ok = False
            messages.append(
                f"{json_path.stem}/{chip_id}: rotation does not swap the envelope "
                f"({straight} -> {rotated})"
            )

    successes = 0
    placed_counts: list[int] = []
    total_footprint_area = sum(
        w * h for w, h in (env.footprint_of(cid, 0) for cid in env.chiplets)
    )
    canvas_area = env.max_width * env.max_height
    print(f"  declared envelope area {total_footprint_area:.2f} / canvas "
          f"{canvas_area:.2f} = {100.0 * total_footprint_area / canvas_area:.1f}%"
          " (random packing only; completeness is not a footprint property)")

    for episode in range(args.episodes):
        random.seed(args.seed + episode)
        env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            _, _, done, info = env.step(random.choice(valid))
            if "error" in info:
                messages.append(f"{json_path.stem}: step error {info['error']}")
                ok = False
                break
        placed = len(env.state.layout)
        placed_counts.append(placed)
        if placed == env.num_chiplets:
            successes += 1
        elif args.require_full_placement:
            ok = False
            messages.append(
                f"{json_path.stem}: episode {episode} placed {placed}/{env.num_chiplets}"
                " (canvas too tight for random legal packing)"
            )
        # Every placed chiplet must be legal against the declared envelope, even
        # when the episode could not complete.
        problems = _envelope_is_inside(env) + _envelopes_overlap(env)
        if problems:
            ok = False
            messages.extend(f"{json_path.stem}: {p}" for p in problems)
        if args.dump and episode == 0:
            out = Path(args.dump) / f"{json_path.stem}_layout.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            env.save_layout_json(str(out))
            print(f"  wrote {out}")

    print(f"  random-legal episodes fully placed: {successes}/{args.episodes}"
          f" (min placed {min(placed_counts) if placed_counts else 0}/{env.num_chiplets})")
    return ok, messages


def check_asymmetric_synthetic() -> tuple[bool, list[str]]:
    """Regression check for asymmetric footprints (absent from examples/)."""
    from chiplet_model import Chiplet, LayoutProblem

    problem = LayoutProblem()
    problem.chiplets["P"] = Chiplet("P", 10.0, 6.0, power=100.0,
                                    hubump=0.05, footprint_w=10.2, footprint_h=6.04)
    problem.chiplets["Q"] = Chiplet("Q", 6.0, 10.0, power=100.0,
                                    hubump=0.05, footprint_w=6.04, footprint_h=10.2)
    problem.chiplets["R"] = Chiplet("R", 4.0, 4.0, power=50.0,
                                    footprint_w=4.0, footprint_h=4.0)
    problem.chiplet_order = ["P", "Q", "R"]
    problem.connection_graph.add_edge("P", "Q", weight=128.0)
    problem.connection_graph.add_edge("Q", "R", weight=64.0)

    env = ChipletPlacementEnv(problem, max_width=50.0, max_height=50.0,
                              lenbase_samples=0, footprint_mode="json_first")
    messages: list[str] = []
    ok = True
    expected = {
        "P": ((0.1, 0.02), (10.2, 6.04), (6.04, 10.2)),
        "Q": ((0.02, 0.1), (6.04, 10.2), (10.2, 6.04)),
        "R": ((0.0, 0.0), (4.0, 4.0), (4.0, 4.0)),
    }
    print("\n=== synthetic asymmetric footprint probe ===")
    for chip_id, (halo, straight, rotated) in expected.items():
        got_halo = env._halo_axes(chip_id, 0)
        got_straight = env.footprint_of(chip_id, 0)
        got_rotated = env.footprint_of(chip_id, 1)
        print(f"  {chip_id}: body {_fmt(float(env.chiplets[chip_id].width))}x"
              f"{_fmt(float(env.chiplets[chip_id].height))}  halo {got_halo}"
              f"  envelope {got_straight} -> rotated {got_rotated}")
        for label, got, want in (
            ("halo", got_halo, halo),
            ("envelope", got_straight, straight),
            ("rotated envelope", got_rotated, rotated),
        ):
            if any(abs(g - w) > 1e-9 for g, w in zip(got, want)):
                ok = False
                messages.append(f"asymmetric probe {chip_id}: {label} {got} != {want}")
        if env.footprint_sources.get(chip_id) != "json_footprint":
            ok = False
            messages.append(
                f"asymmetric probe {chip_id}: source {env.footprint_sources.get(chip_id)} "
                "!= json_footprint"
            )
    return ok, messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="append", default=None,
                        help="explicit benchmark JSON (repeatable); default: examples/*.json")
    parser.add_argument("--case", action="append", default=None,
                        help="case stem to include, e.g. --case acend910 (repeatable)")
    parser.add_argument("--mode", choices=FOOTPRINT_MODES, default="json_first",
                        help="footprint_mode passed to the environment")
    parser.add_argument("--verify_footprint", action="store_true",
                        help="raise when the declared envelope differs from the recomputed halo")
    parser.add_argument("--episodes", type=int, default=5,
                        help="random-legal placement episodes per case (default 5)")
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument("--canvas", type=float, default=50.0,
                        help="canvas side in mm (default 50.0, the FastTM interposer)")
    parser.add_argument("--grid_resolution", default=None,
                        help="grid resolution or 'auto' (default: auto)")
    parser.add_argument("--max_auto_grid_resolution", type=int, default=100)
    parser.add_argument("--dump", default=None,
                        help="directory to write the first layout of each case")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--require_full_placement", action="store_true",
                        help="also fail when random legal packing cannot place every "
                             "chiplet (a canvas/search property, not a footprint one)")
    args = parser.parse_args()

    if args.json:
        paths = [Path(p).resolve() for p in args.json]
    else:
        paths = sorted((RL_DIR / "examples").glob("*.json"))
    if args.case:
        wanted = set(args.case)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        print("no benchmark JSON selected", file=sys.stderr)
        return 2

    all_ok = True
    messages: list[str] = []
    if args.mode != "recompute":
        probe_ok, probe_messages = check_asymmetric_synthetic()
        all_ok = all_ok and probe_ok
        messages.extend(probe_messages)
    for path in paths:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            all_ok = False
            continue
        case_ok, case_messages = check_case(path, args)
        all_ok = all_ok and case_ok
        messages.extend(case_messages)

    print("\n" + "=" * 70)
    if all_ok:
        if args.mode == "recompute":
            print(f"PASS: {len(paths)} case(s) checked with connectivity-derived "
                  "envelopes (footprint fields intentionally ignored)")
        else:
            print(f"PASS: {len(paths)} case(s) use the declared occupied envelope "
                  f"(footprint_mode={args.mode})")
    else:
        print(f"FAIL: {len(messages)} problem(s)")
        for message in messages[:40]:
            print(f"  - {message}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
