"""Small file-based live monitor for long Flow Matching training runs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _read_history(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
    return rows


def _plot_series(ax, rows, keys, title, ylabel=None, log=False):
    for key, label in keys:
        points = [(row.get("step"), _number(row.get(key))) for row in rows]
        points = [(step, value) for step, value in points if step is not None and value is not None]
        if points:
            ax.plot([point[0] for point in points], [point[1] for point in points], label=label)
    ax.set_title(title)
    ax.set_xlabel("training step")
    if ylabel:
        ax.set_ylabel(ylabel)
    if log:
        positive = any(
            _number(row.get(key)) is not None and _number(row.get(key)) > 0
            for row in rows for key, _ in keys
        )
        if positive:
            ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=8)


def update_training_monitor(log_dir, step, metrics, filename="training_monitor.png", history_filename="training_monitor.jsonl"):
    """Append scalar metrics and atomically replace the live PNG."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / history_filename
    existing_rows = _read_history(history_path)
    if existing_rows and int(existing_rows[-1].get("step", -1)) >= int(step):
        # A checkpoint can lag behind the monitor if a run stops while saving.
        # Drop the stale tail before appending resumed steps.
        existing_rows = [row for row in existing_rows if int(row.get("step", -1)) < int(step)]
        temp_history = history_path.with_name(history_path.name + ".tmp")
        with temp_history.open("w", encoding="utf-8") as handle:
            for existing in existing_rows:
                handle.write(json.dumps(existing, ensure_ascii=False) + "\n")
        os.replace(temp_history, history_path)
    row = {"step": int(step)}
    for key, value in dict(metrics).items():
        value = _number(value)
        if value is not None:
            row[str(key)] = value
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = _read_history(history_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    _plot_series(
        axes[0, 0], rows,
        [("loss", "total"), ("flow_loss", "flow")],
        "Training losses", log=True,
    )
    _plot_series(
        axes[0, 1], rows,
        [("thermal_weighted_loss", "thermal weighted"),
         ("wirelength_weighted_loss", "wirelength weighted")],
        "Proxy contributions", log=True,
    )
    _plot_series(
        axes[1, 0], rows,
        [("legality_overlap_risk", "overlap risk"),
         ("legality_boundary_risk", "boundary risk"),
         ("legality_overlap_direct_weighted_loss", "overlap penalty"),
         ("legality_boundary_direct_weighted_loss", "boundary penalty")],
        "Legality", log=True,
    )
    _plot_series(
        axes[1, 1], rows,
        [("thermal_weight", "thermal"), ("wirelength_weight", "wirelength"),
         ("legality_overlap_direct_weight", "overlap"),
         ("legality_boundary_direct_weight", "boundary")],
        "Effective auxiliary weights",
    )
    fig.suptitle(f"Flow Matching training monitor — step {int(step):,}", fontsize=14)
    output_path = log_dir / filename
    temp_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    fig.savefig(temp_path, dpi=140, facecolor="white")
    plt.close(fig)
    os.replace(temp_path, output_path)
    return output_path
