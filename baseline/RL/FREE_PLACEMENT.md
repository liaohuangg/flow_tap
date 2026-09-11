# Score-ordered free placement

The default order is descending `placement_area_weight * width * height +
placement_pin_weight * incident_pin_count`. Both weights default to 1; no
normalization is applied. Equal scores retain input order. Each connection
contributes its `wireCount` (falling back to `weight`, then 1) to both endpoints.
An explicit legacy `placement_order` must agree with this ordering.

Training and inference accept `--placement_area_weight` and
`--placement_pin_weight`; use the same values for both.

Training uses paper-style PPO defaults: 600 epochs, 480 rollout transitions,
four minibatches, four PPO passes, learning rate 2.5e-4, clip coefficient 0.1,
and entropy coefficient 0.01. `--num_episodes` remains an alias for
`--num_epochs`. Thermal-table generation/checking is mandatory. The original
reward path runs CPLEX microbump assignment followed by FastTM; CPLEX may be
discovered from a sibling Conda environment such as `tap25d`.

## Occupied envelope (bump-expanded) inputs

`footprint_mode` selects where each chiplet's occupied envelope comes from:

| Mode | Behaviour |
|---|---|
| `json_first` (default) | declared `footprint_w`/`footprint_h`, else `hubump`, else recomputed |
| `json_only` | same, but the JSON must declare a footprint or a `hubump` |
| `recompute` | always derive the halo from input connectivity (legacy) |

The modes are accepted by both `train.py` and `run.py` as
`--footprint_mode`; `--verify_footprint` turns the mismatch warning below into a
hard error.

The declared envelope is used **verbatim**. The microbump halo is derived per
axis as `halo_x = (footprint_w - width) / 2` and `halo_y = (footprint_h - height)
/ 2`, so an asymmetric footprint (one where `footprint_w - width` differs from
`footprint_h - height`) is reproduced exactly. Halos are declared against the
unrotated body and are swapped together with the body width/height on a
90-degree rotation, so a rotated chiplet occupies exactly the rotated footprint.

A declared footprint equal to the body is how a chiplet with no microbump ring (a
DUMMY die) is expressed. A `hubump` of exactly 0 counts as unset: to request a
zero halo, declare `footprint_w`/`footprint_h` equal to `width`/`height`.

The connectivity-derived halo (FastTM's `PassiveInterposer.compute_ubump_overhead()`,
pitch 45 um, smallest whole number of bump rows carrying the symmetric
bidirectional interface demand) is always computed as well. It is the fallback
and the cross-check: when a declared envelope differs from it, the declared value
wins and a warning listing the chiplet is printed and recorded in
`env.footprint_warnings`. `env.footprint_report()` returns the per-chiplet
provenance. Among the shipped examples only `Case10`'s `DUMMY_*` die differ
(`hubump: 0.0`, footprint 8x12 versus a recomputed 8.09x12.09).

The occupied rectangle is `(x-halo_x, y-halo_y, x+width+halo_x,
y+height+halo_y)`. Only this envelope — never the bare body — is tested against
the canvas bounds and against other chiplets, and it is what the observation and
the utilization reward use. The scalar `env.microbump_halos` is retained as the
symmetric FastTM ring width (`system.hubump`); it prefers the declared `hubump`,
then the footprint-derived halo, then the recomputed value.

## Layout export

`save_layout_json()` records `rotation`, `hubump`, `hubump_x`/`hubump_y`,
`footprint_w`/`footprint_h` for the placed rotation, `source_footprint_w`/
`source_footprint_h` as authored, `occupied_envelope_source` and the absolute
`occupied_bounds`. `train.py`'s geometry metrics derive the occupied area from
those bounds, so rotated and asymmetric chiplets are measured correctly.

## Checks

Run regression checks with `python -m unittest test_free_placement -v`; it covers
the ordering rules above, the halo geometry, and the occupied-envelope inputs.

Validate the envelope handling against every shipped benchmark with:

```bash
python check_footprint_inputs.py            # all examples/*.json
python check_footprint_inputs.py --case Case10 --verbose
python check_footprint_inputs.py --mode recompute   # legacy A/B comparison
```

All canvas-grid positions are considered for each rotation, including for the
first chiplet. Exact lower/upper boundary positions are added so the microbump
halo, rather than only the silicon body, may touch the canvas boundary. Only
canvas bounds and occupied-rectangle overlap restrict placement; bump halos may
touch but cannot overlap. The action space is still discrete. Automatic grids
are capped at 100 positions per axis; with two rotations and 50,000
exact-coordinate slots, the maximum default action dimension is 70,000. Adjust
`--grid_resolution` or `--max_auto_grid_resolution` only after checking GPU
memory.

Checkpoint saving is disabled by default because a flat action head can produce
very large files. Logs, metrics, and best-layout JSON/PNG artifacts are still
retained. Pass `--save_checkpoint` only when a model file is required.

EMIB shared-edge requirements and contact rewards are disabled. This does not
disable the physical microbump keep-out envelope. CPLEX still assigns the four
microbump clumps, and FastTM thermal centering now uses the complete occupied
envelope. Layout JSON files record `hubump` and expanded occupied geometry;
plots show the halo in orange. Training/inference plots do not draw bridges, and
separated chiplets do not receive fabricated bridge coordinates in exports.

Retrain policies for the changed order and candidate distribution. Existing
checkpoint tensor shapes alone do not establish behavioral compatibility.

