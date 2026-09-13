# Flow best-of-20 per case

13 cases, 20 seeds each (seeds 61-80, one candidate per seed). Candidate
selection is filtered to legal layouts (hubump-expanded footprint, no overlap), the
same rule the reselect report uses. Two picks per case, because the objectives
disagree and usually select different layouts:

- `bestT`  — minimum HotSpot peak temperature (unified 6-layer TAP-2.5D, 64x64 grid)
- `bestWL` — minimum TAP-2.5D average wirelength (bundled CPLEX, `hubump_mode="die"`)

Per pick: the raw candidate JSON (body-50% canvas, the scored artifact), the pipeline's
layout figure (`*.png`), and a self-describing `*.svg` rendered from the JSON.

| case | pick | seed | HotSpot peak (C) | TAP avg (mm) | chiplets | legal cands |
|---|---|---:|---:|---:|---:|---:|
| Case10 | bestT | 64 | 106.51 | 30.836 | 61 | 19/20 |
| Case10 | bestWL | 73 | 114.69 | 22.670 | 61 | 19/20 |
| Case6 | bestT | 67 | 97.48 | 27.500 | 20 | 17/20 |
| Case6 | bestWL | 79 | 106.12 | 21.917 | 20 | 17/20 |
| Case7 | bestT | 65 | 76.94 | 14.679 | 28 | 16/20 |
| Case7 | bestWL | 67 | 80.22 | 11.718 | 28 | 16/20 |
| Case8 | bestT | 65 | 78.43 | 13.956 | 36 | 20/20 |
| Case8 | bestWL | 66 | 79.00 | 10.547 | 36 | 20/20 |
| Case9 | bestT | 76 | 117.42 | 32.570 | 44 | 20/20 |
| Case9 | bestWL | 68 | 125.11 | 28.741 | 44 | 20/20 |
| acend910 | bestT | 76 | 73.72 | 19.614 | 6 | 15/20 |
| acend910 | bestWL | 69 | 75.09 | 8.233 | 6 | 15/20 |
| cpu-dram | bestT | 62 | 109.24 | 13.753 | 8 | 6/20 |
| cpu-dram | bestWL | 62 | 109.24 | 13.753 | 8 | 6/20 |
| hp11_m | bestT | 74 | 95.78 | 18.998 | 11 | 7/20 |
| hp11_m | bestWL | 77 | 101.98 | 13.898 | 11 | 7/20 |
| multigpu | bestT | 61 | 97.03 | 13.579 | 6 | 11/20 |
| multigpu | bestWL | 63 | 99.51 | 12.764 | 6 | 11/20 |
| syn1 | bestT | 80 | 85.36 | 19.812 | 10 | 13/20 |
| syn1 | bestWL | 76 | 93.25 | 12.831 | 10 | 13/20 |
| syn4 | bestT | 73 | 88.09 | 16.062 | 13 | 12/20 |
| syn4 | bestWL | 65 | 89.11 | 13.281 | 13 | 12/20 |
| syn6 | bestT | 80 | 85.62 | 21.063 | 15 | 16/20 |
| syn6 | bestWL | 79 | 90.44 | 14.917 | 15 | 16/20 |
| xerox8_m | bestT | 67 | 79.22 | 22.102 | 8 | 15/20 |
| xerox8_m | bestWL | 61 | 85.66 | 14.646 | 8 | 15/20 |

## Reproducing the reported numbers

The JSON in this folder is the candidate exactly as generated, on the body-50% canvas.
The evaluator centres it on the footprint-50% canvas first; the per-axis translation is
in `manifest.csv` (`translate_per_axis_mm`), so:

```python
import json
d = json.load(open('Case10/Case10_bestT_seed64.json'))
delta = 0.0  # translate_per_axis_mm from manifest.csv
for c in d['chiplets']:
    c['x-position'] += delta; c['y-position'] += delta
```

Temperature: `evaluate_baselines_unified.write_and_run_hotspot` (6-layer TAP-2.5D stack,
64x64 grid, same binary and template config for Flow/RL/ATPlace, `-ambient 318.15`).
Wirelength: `gen_dataset.gen_wirelength_dataset.solve_cplex_avg(TapSystem(layout,
hubump_mode="die"))` — the identical call the Flow metrics and the ATPlace recompute use.

## Notes

- `*.png` and `*.csv` are ignored by the repository root `.gitignore`; the nested
  `.gitignore` in this folder re-includes them. `git add LayoutGenModel/logs/output/flow_best_of_20`
  is enough — no `-f` needed. If your git is older and a file still refuses to stage, use
  `git add -f <path>`.
- `cpu-dram` has the same seed for both objectives, so its two picks are byte-identical.
- Pick counts differ per case because the legal filter varies (e.g. `cpu-dram` 6/20,
  `Case8` 20/20).

## Beyond this folder

The per-candidate neural-surrogate thermal heatmaps and `*.pkl` tensors stay in the pool
at `LayoutGenModel/logs/output/cases_hubump/cases-hubump-util50-candidates-20seeds-v1/`.
They are deliberately not bundled: the numbers quoted above come from HotSpot, not from
the surrogate (within-case Spearman is only ~0.75).
