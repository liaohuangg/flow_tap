#!/usr/bin/env python
"""Run every benchmark case with 5 seeds of time-boxed PPO training.

For each case in ``examples/`` and each seed the driver:

  1. sizes the square canvas so the total bump-expanded chiplet footprint is
     ``--ratio`` (default 0.50) of the canvas area, i.e.
     ``side = sqrt(sum(footprint_w * footprint_h) / ratio)``, and hands that same
     side to fastTM as ``thermal_intp_size`` so the layout always fits the
     interposer;
  2. generates the fastTM thermal tables **inside each run's own ``--seconds``
     budget**, and never reuses them across seeds: every run passes
     ``--force_thermal_tables``, so all five seeds of a case regenerate their own
     tables and each run is a self-contained ``--seconds`` window. ``train.py``
     charges the thermal stage against ``--time_limit_seconds``, so the RL loop
     receives ``--seconds`` minus that stage's duration. Reuse and generation
     outside the per-seed budget are rejected by the command-line parser;
  3. runs ``train.py`` for ``--seconds`` per seed, saving logs, metrics, the best
     layouts and a checkpoint;
  4. ranks the five seeds of the case by ``rlplanner_cost``
     (``avg_wirelength + max(T-80,0)**1.3 / (1 + exp(80 - T))``, the original
     RLPlanner cost) and copies the winning checkpoint and best layout to
     ``runs/<case>/best/``;
  5. writes ``runs/<case>/case_summary.json`` and the global
     ``runs/summary.json`` / ``runs/summary.csv``.

Run it *with the interpreter that has torch + cplex*, e.g.

    /home/user/miniconda3/envs/chipdiffusion/bin/python run_cases_seeds.py --smoke
    /home/user/miniconda3/envs/chipdiffusion/bin/python run_cases_seeds.py

The environment is checked before anything is launched: ``fastTM/routing.py``
imports ``cplex`` at module level and ``fastTM/util/hotspot`` is a Linux
executable invoked as ``./util/hotspot``, so the run must happen in Linux/WSL
with ``cwd=fastTM``.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent
FASTM_DIR = RL_DIR / "fastTM"
HOTSPOT_BIN = FASTM_DIR / "util" / "hotspot"
EXAMPLES_DIR = RL_DIR / "examples"

#: Draws the concrete seeds for 'random' entries in --seeds.
_SEED_RNG = random.SystemRandom()

#: Extra sinks for log(); the driver log is registered in main().
_LOG_STREAMS: list = []

#: reward_cal.py TEMP_LIMIT / env.TEMP_LIMIT
TEMP_LIMIT = 80.0

REQUIRED_MODULES = ("numpy", "torch", "scipy", "cplex")
#: compute_temp.py imports pandas but never calls it (0 `pd.` uses); env.py and
#: generate_thermal_tables.py already inject a stub, so pandas is only a warning.
OPTIONAL_MODULES = ("pandas",)

#: Directories under a CPLEX Studio install that hold the Python bindings.
CPLEX_ARCHES = ("x86-64_linux", "x86-64_win64", "x86-64_osx", "x86-64_linux_ppc64le")
CPLEX_STUDIO_GLOBS = ("/opt/ibm/ILOG/CPLEX_Studio*", "C:/Program Files/IBM/ILOG/CPLEX_Studio*",
                      "C:/Program Files/IBM/ILOG/CPLEX_Studio*/")

ENV_CHECK_SNIPPET = r"""
import importlib, json, sys
report = {"python": sys.executable, "version": sys.version.split()[0], "modules": {}, "errors": []}
for name in %(modules)r + %(optional)r:
    try:
        module = importlib.import_module(name)
        report["modules"][name] = getattr(module, "__version__", "ok")
    except Exception as exc:
        report["modules"][name] = None
        report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
try:
    import torch
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as exc:
    report["cuda_available"] = None
    report["cuda_device"] = None
    report["errors"].append(f"cuda probe: {type(exc).__name__}: {exc}")
print("ENVCHECK_JSON:" + json.dumps(report))
"""


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    for stream in _LOG_STREAMS:
        try:
            stream.write(line + "\n")
            stream.flush()
        except Exception:
            pass


def read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def temperature_penalty(temp, temp_limit: float = TEMP_LIMIT) -> float:
    """reward_cal.py penalty term; identical to FastTM/CPLEX reward path."""
    if temp is None:
        return float("nan")
    temp = float(temp)
    if not math.isfinite(temp):
        return float("nan")
    return math.pow(max(temp - temp_limit, 0.0), 1.3) / (1.0 + math.exp(temp_limit - temp))


def cost_from_metrics(metrics) -> float:
    """rlplanner_cost = avg_wirelength + temperature penalty.

    Prefers the recorded cost, otherwise recomputes it from the exported
    avg_wirelength/temperature so older run summaries can still be ranked.
    """
    if not isinstance(metrics, dict):
        return float("nan")
    recorded = metrics.get("rlplanner_cost")
    if isinstance(recorded, (int, float)) and math.isfinite(float(recorded)):
        return float(recorded)
    averaged = metrics.get("rlplanner_avg_wirelength", metrics.get("avg_wirelength"))
    temperature = metrics.get("rlplanner_temperature", metrics.get("temperature"))
    if averaged is None or temperature is None:
        return float("nan")
    penalty = temperature_penalty(temperature)
    if not math.isfinite(penalty):
        return float("nan")
    return float(averaged) + penalty


def summary_metrics(summary) -> dict:
    """Metrics dict of a train.py best-summary (full metrics, not the subset)."""
    if not isinstance(summary, dict):
        return {}
    metrics = summary.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def summary_cost(summary) -> float:
    if not isinstance(summary, dict):
        return float("nan")
    recorded = summary.get("rlplanner_cost")
    if isinstance(recorded, (int, float)) and math.isfinite(float(recorded)):
        return float(recorded)
    return cost_from_metrics(summary_metrics(summary))


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def discover_cases(selected: str) -> list[str]:
    available = sorted(path.stem for path in EXAMPLES_DIR.glob("*.json"))
    if selected.strip().lower() in ("", "all", "*"):
        return available
    wanted = [item.strip() for item in selected.split(",") if item.strip()]
    missing = [item for item in wanted if item not in available]
    if missing:
        raise SystemExit(f"unknown case(s): {', '.join(missing)}; available: {', '.join(available)}")
    return wanted


def footprint_area(payload: dict) -> tuple[float, float]:
    """Return (footprint area, body area) in mm^2 using declared footprints."""
    body = 0.0
    footprint = 0.0
    for chiplet in payload["chiplets"]:
        width = float(chiplet["width"])
        height = float(chiplet["height"])
        body += width * height
        fp_w = chiplet.get("footprint_w")
        fp_h = chiplet.get("footprint_h")
        if fp_w is None or fp_h is None:
            hubump = float(chiplet.get("hubump", 0.0) or 0.0)
            fp_w, fp_h = width + 2.0 * hubump, height + 2.0 * hubump
        footprint += float(fp_w) * float(fp_h)
    return footprint, body


def build_plan(cases: list[str], ratio: float, canvas_override: float | None) -> list[dict]:
    plan = []
    for case in cases:
        json_path = EXAMPLES_DIR / f"{case}.json"
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        footprint, body = footprint_area(payload)
        # Round once, and always hand the *same* decimal string to fastTM and to
        # train.py. train.py regenerates the thermal tables whenever the config's
        # intp_size differs from the requested value, and that regeneration is
        # charged against --time_limit_seconds, so a formatting difference of
        # 1e-9 would silently eat the whole training budget.
        side = round(float(canvas_override) if canvas_override else
                     math.sqrt(footprint / ratio), 6)
        plan.append({
            "case": case,
            "json": str(json_path),
            "chiplets": len(payload["chiplets"]),
            "footprint_area": footprint,
            "body_area": body,
            "side_mm": side,
            "side_arg": f"{side:.6f}",
            "ratio": ratio,
            "has_fasttm_config": (FASTM_DIR / "configs" / f"benchmark_{case}.cfg").exists(),
        })
    return plan


# ---------------------------------------------------------------------------
# environment / thermal tables
# ---------------------------------------------------------------------------

def python_version(python: str) -> str | None:
    result = subprocess.run(
        [python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True, text=True)
    version = (result.stdout or "").strip()
    return version or None


def discover_cplex_python_path(python: str) -> str | None:
    """Return a PYTHONPATH entry that makes `import cplex` work, if one exists.

    IBM ships the Python bindings inside the Studio install, so they can be used
    without installing anything into the conda environment.
    """
    probe = subprocess.run([python, "-c", "import cplex"], capture_output=True, text=True)
    if probe.returncode == 0:
        return None  # already importable; nothing to add

    version = python_version(python)
    if not version:
        return None
    studios = []
    if os.environ.get("CPLEX_STUDIO_DIR"):
        studios.append(Path(os.environ["CPLEX_STUDIO_DIR"]))
    for pattern in CPLEX_STUDIO_GLOBS:
        studios.extend(sorted(Path(p) for p in glob.glob(pattern)))
    for studio in studios:
        for arch in CPLEX_ARCHES:
            candidate = studio / "cplex" / "python" / version / arch
            if (candidate / "cplex" / "__init__.py").is_file():
                probe = subprocess.run([python, "-c", "import cplex"], capture_output=True,
                                       text=True, env=with_pythonpath(candidate))
                if probe.returncode == 0:
                    return str(candidate)
    return None


def with_pythonpath(*entries) -> dict:
    env = os.environ.copy()
    parts = [str(entry) for entry in entries if entry]
    if parts:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(parts + ([existing] if existing else []))
    return env


def env_check(python: str, strict: bool, child_env: dict, cplex_path: str | None) -> bool:
    if not Path(python).exists() and shutil.which(python) is None:
        log(f"FAIL python interpreter not found: {python}")
        return False
    if cplex_path:
        log(f"cplex       : resolved from {cplex_path} (via PYTHONPATH, no install)")
    snippet = ENV_CHECK_SNIPPET % {"modules": REQUIRED_MODULES, "optional": OPTIONAL_MODULES}
    result = subprocess.run([python, "-c", snippet], capture_output=True, text=True, env=child_env)
    payload = None
    for line in (result.stdout or "").splitlines():
        if line.startswith("ENVCHECK_JSON:"):
            payload = json.loads(line[len("ENVCHECK_JSON:"):])
    if payload is None:
        log("FAIL could not run the environment check")
        log(f"  stdout: {(result.stdout or '').strip()[:400]}")
        log(f"  stderr: {(result.stderr or '').strip()[:400]}")
        return False

    log(f"interpreter : {payload['python']} (python {payload['version']})")
    for name in REQUIRED_MODULES + OPTIONAL_MODULES:
        version = payload["modules"].get(name)
        required = name in REQUIRED_MODULES
        tag = "" if required else " (optional)"
        log(f"  {name:<8}: {version if version else 'MISSING'}{tag}")
    log(f"  cuda     : {payload.get('cuda_available')} {payload.get('cuda_device') or ''}".rstrip())

    missing_required = [name for name in REQUIRED_MODULES if not payload["modules"].get(name)]
    ok = True
    if missing_required:
        ok = False
        for error in payload["errors"]:
            if error.split(":", 1)[0] in missing_required:
                log(f"  module error -> {error}")
        log(f"  fastTM/routing.py does 'import cplex' at module level and the terminal")
        log(f"  reward (--terminal_rlplanner_cost_scale) needs it, so training cannot run.")
        if "cplex" in missing_required:
            log("  hint: point --cplex-pythonpath at "
                "<CPLEX_STUDIO_DIR>/cplex/python/<py-version>/x86-64_linux")
    if not payload["modules"].get("pandas"):
        log("  note: pandas is absent, which is fine - compute_temp.py never calls it and")
        log("        env.py/generate_thermal_tables.py already install a stub module.")
    if not HOTSPOT_BIN.exists():
        log(f"FAIL hotspot binary missing: {HOTSPOT_BIN}")
        ok = False
    elif os.name != "nt" and not os.access(HOTSPOT_BIN, os.X_OK):
        log(f"FAIL hotspot binary is not executable: chmod +x {HOTSPOT_BIN}")
        ok = False
    if os.name == "nt":
        log("WARNING running on Windows: fastTM/util/hotspot is a Linux ELF binary and")
        log("        char_thermal_r.py uses 'rm', so thermal-table generation will fail.")
        ok = False

    if not ok and strict:
        log("FAIL environment check failed; nothing was launched.")
    return ok


def _existing_intp_size(case: str) -> float | None:
    cfg_path = FASTM_DIR / "configs" / f"benchmark_{case}.cfg"
    if not cfg_path.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        if not parser.read(cfg_path):
            return None
        return parser.getfloat("interposer", "intp_size")
    except Exception:
        return None


def thermal_tables_ready(case: str, side: float, chiplet_count: int) -> bool:
    """True when the config and every Chiplet_i.rself/.rmutu exist at `side`."""
    existing = _existing_intp_size(case)
    if existing is None or abs(existing - float(side)) > 1e-9:
        return False
    table_dir = FASTM_DIR / "outputs" / f"benchmark_{case}"
    if not table_dir.is_dir():
        return False
    for index in range(chiplet_count):
        for suffix in (".rself", ".rmutu"):
            if not (table_dir / f"Chiplet_{index}{suffix}").exists():
                return False
    return True


def prepare_thermal_tables(case: str, side: float, chiplet_count: int, log_dir: Path,
                           python: str, child_env: dict, force: bool = False) -> dict:
    """Generate fastTM tables at `side` outside the training time budget."""
    if not force and thermal_tables_ready(case, side, chiplet_count):
        return {"status": "skipped_already_ready", "side_mm": side, "seconds": 0.0}

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "thermal_prep.log"
    json_path = (EXAMPLES_DIR / f"{case}.json").resolve()
    existing = _existing_intp_size(case)
    table_dir = FASTM_DIR / "outputs" / f"benchmark_{case}"
    has_tables = table_dir.is_dir() and any(table_dir.glob("Chiplet*.rself"))
    # Mirror train.py: a changed interposer size invalidates existing tables.
    need_force = bool(force or (existing is not None and abs(existing - float(side)) > 1e-9) or has_tables)

    cmd = [
        python,
        str(RL_DIR / "generate_thermal_tables.py"),
        str(json_path),
        "--intp-size", f"{side:.6f}",
        "--overwrite-config",
    ]
    if need_force:
        cmd.append("--force")

    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"command: {' '.join(cmd)}\n")
        log_file.write(f"cwd: {FASTM_DIR}\n")
        log_file.write(f"previous_intp_size: {existing}\n")
        log_file.write(f"has_existing_tables: {has_tables}\n\n")
        log_file.flush()
        try:
            result = subprocess.run(cmd, cwd=str(FASTM_DIR), stdout=log_file,
                                    stderr=subprocess.STDOUT, text=True, check=False,
                                    env=child_env)
            returncode = result.returncode
        except Exception as exc:
            log_file.write(f"\nEXCEPTION {type(exc).__name__}: {exc}\n")
            returncode = -1
    seconds = time.perf_counter() - started
    status = "ok" if returncode == 0 else f"failed:{returncode}"
    return {
        "status": status,
        "returncode": returncode,
        "side_mm": side,
        "previous_intp_size": existing,
        "force": need_force,
        "seconds": seconds,
        "log": str(log_path),
    }


# ---------------------------------------------------------------------------
# running one seed
# ---------------------------------------------------------------------------

def run_seed(python: str, run_name: str, json_path: Path, seed: int, budget: float,
             side: float, runs_root: Path, console_dir: Path, child_env: dict,
             force_thermal: bool = False, dry_run: bool = False) -> dict:
    cmd = [
        python, str(RL_DIR / "train.py"),
        "--json", str(json_path),
        "--name", run_name,
        "--runs-dir", str(runs_root),
        "--seed", str(seed),
        "--time_limit_seconds", f"{budget:.1f}",
        "--max_width", f"{side:.6f}",
        "--max_height", f"{side:.6f}",
        "--thermal_intp_size", f"{side:.6f}",
        "--footprint_mode", "json_first",
        "--save_checkpoint",
        "--log_interval", "1",
    ]
    if force_thermal:
        # Every run regenerates its own thermal tables, charged to its own budget.
        cmd.append("--force_thermal_tables")
    run_dir = runs_root / run_name
    console_dir.mkdir(parents=True, exist_ok=True)
    console_path = console_dir / f"{run_name}.console.log"

    if dry_run:
        log(f"  DRY-RUN would run: {' '.join(cmd)}")
        return {"run_name": run_name, "seed": seed, "dry_run": True, "command": cmd}

    started = time.perf_counter()
    log(f"  seed {seed}: {' '.join(cmd[:6])} ... budget {budget:.0f}s")
    with open(console_path, "w", encoding="utf-8") as console:
        console.write(f"command: {' '.join(cmd)}\n")
        console.write(f"started_at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        console.flush()
        result = subprocess.run(cmd, cwd=str(RL_DIR), stdout=console,
                                stderr=subprocess.STDOUT, text=True, check=False,
                                env=child_env)
    elapsed = time.perf_counter() - started

    if not run_dir.exists():
        log(f"    WARNING: train.py did not create {run_dir} (rc={result.returncode}); "
            f"its results may be under {RL_DIR / 'runs'} instead")
    if result.returncode != 0:
        hint = f"see {run_dir / 'train.log'}"
        if not (run_dir / "timing.json").exists():
            hint += (" -- the run aborted early. The thermal-table stage is charged "
                     "to this budget and is terminated when that budget is exhausted; "
                     "use a larger --seconds value")
        log(f"    WARNING: rc={result.returncode}; {hint}")

    timing = read_json(run_dir / "timing.json") or {}
    config = read_json(run_dir / "config.json") or {}
    info = {
        "run_name": run_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "command": cmd,
        "console_log": str(console_path),
        "returncode": result.returncode,
        "wall_seconds": elapsed,
        "termination_reason": timing.get("termination_reason"),
        "rl_solve_seconds": (timing.get("rl_solve") or {}).get("seconds"),
        "thermal_tables_seconds": (timing.get("thermal_tables") or {}).get("seconds"),
        "thermal_tables_status": (timing.get("thermal_tables") or {}).get("status"),
        "thermal_tables_forced": (timing.get("thermal_tables") or {}).get("force"),
        "num_epochs_requested": config.get("num_epochs"),
    }
    log(f"    -> rc={result.returncode} wall={elapsed:.1f}s "
        f"termination={info['termination_reason']} rl={info['rl_solve_seconds']}")
    return info


def read_run_results(run_info: dict) -> dict:
    """Collect per-seed results from a finished run directory."""
    run_dir = Path(run_info["run_dir"])
    cost_summary = read_json(run_dir / "best_cost_summary.json")
    reward_summary = read_json(run_dir / "best_summary.json")

    row = dict(run_info)
    # Derive the run metadata from disk so results can be re-collected later
    # without having to keep the in-memory record from run_seed().
    timing = read_json(run_dir / "timing.json") or {}
    config = read_json(run_dir / "config.json") or {}
    row.setdefault("termination_reason", timing.get("termination_reason"))
    row.setdefault("rl_solve_seconds", (timing.get("rl_solve") or {}).get("seconds"))
    row.setdefault("thermal_tables_seconds", (timing.get("thermal_tables") or {}).get("seconds"))
    row.setdefault("thermal_tables_status", (timing.get("thermal_tables") or {}).get("status"))
    row.setdefault("thermal_tables_forced", (timing.get("thermal_tables") or {}).get("force"))
    row.setdefault("num_epochs_requested", config.get("num_epochs"))
    row.setdefault("seed", config.get("seed"))
    row.setdefault("console_log", None)

    row["best_cost_summary"] = cost_summary
    row["best_reward_summary"] = reward_summary
    # Prefer the cost-ranked solution; fall back to the reward-ranked one.
    if math.isfinite(summary_cost(cost_summary)):
        chosen = cost_summary
        row["ranked_by"] = "best_cost_summary"
    else:
        chosen = reward_summary
        row["ranked_by"] = "best_summary"
    metrics = summary_metrics(chosen)
    row["cost"] = summary_cost(chosen)
    row["avg_wirelength"] = metrics.get("rlplanner_avg_wirelength")
    row["temperature"] = metrics.get("rlplanner_temperature")
    row["canvas_utilization"] = metrics.get("canvas_utilization")
    row["silicon_canvas_utilization"] = metrics.get("silicon_canvas_utilization")
    row["success_rate"] = _last_progress_value(run_dir, "total_success_rate")
    row["executed_epochs"] = _last_progress_value(run_dir, "epoch")
    row["total_rollouts"] = _last_progress_value(run_dir, "total_rollouts")
    return row


def _last_progress_value(run_dir: Path, key: str):
    path = run_dir / "progress.jsonl"
    if not path.exists():
        return None
    last = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except Exception:
                    continue
    except Exception:
        return None
    return last.get(key) if isinstance(last, dict) else None


# ---------------------------------------------------------------------------
# collecting the best solution of a case
# ---------------------------------------------------------------------------

def _candidate_checkpoint(row: dict) -> Path | None:
    run_dir = Path(row["run_dir"])
    for summary_key in ("best_cost_summary", "best_reward_summary"):
        summary = row.get(summary_key)
        if isinstance(summary, dict) and summary.get("checkpoint"):
            path = Path(summary["checkpoint"])
            if path.exists():
                return path
    for name in ("ppo_best_cost.pt", "ppo_best.pt"):
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    return None


def collect_case(case: str, plan: dict, seed_rows: list[dict], case_dir: Path,
                 prune_seed_checkpoints: bool = False,
                 prune_loser_artifacts: bool = False) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    best_dir = case_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    ranked = [row for row in seed_rows if math.isfinite(row.get("cost", float("nan")))]
    ranked.sort(key=lambda row: row["cost"])
    winner = ranked[0] if ranked else None

    if winner is None:
        log(f"  no seed produced a finite rlplanner_cost for {case}")
        return {
            "case": case,
            "plan": plan,
            "target_ratio": plan["ratio"],
            "canvas_side_mm": plan["side_mm"],
            "footprint_area_mm2": plan["footprint_area"],
            "chiplets": plan["chiplets"],
            "best_seed": None,
            "seeds": [
                {"seed": row["seed"], "run_name": row.get("run_name"),
                 "returncode": row.get("returncode"),
                 "rlplanner_cost": row.get("cost"),
                 "run_dir": row.get("run_dir")}
                for row in sorted(seed_rows, key=lambda item: item["seed"])
            ],
        }

    summary = winner.get("best_cost_summary") or winner.get("best_reward_summary") or {}
    artifacts = {}
    checkpoint = _candidate_checkpoint(winner)
    if checkpoint is not None:
        target = best_dir / f"best_{case}.pt"
        shutil.copyfile(checkpoint, target)
        artifacts["checkpoint"] = str(target)
    for key, suffix in (("json", ".json"), ("metrics_json", ".metrics.json"), ("img", ".png")):
        source = summary.get(key)
        if source and Path(source).exists():
            target = best_dir / f"best_{case}{suffix}"
            shutil.copyfile(source, target)
            artifacts["layout" if suffix == ".json" else suffix.lstrip(".")] = str(target)

    # Keep only the winning seed's checkpoint: delete the losers so a case costs
    # one model on disk instead of one per seed. The winner is copied above and
    # its original is left in place so train.py's own summaries stay valid.
    pruned = []
    if prune_seed_checkpoints:
        for row in seed_rows:
            if row["seed"] == winner["seed"]:
                continue
            run_dir = Path(row["run_dir"])
            for name in ("ppo_best_cost.pt", "ppo_best.pt"):
                path = run_dir / "checkpoints" / name
                if path.exists():
                    try:
                        path.unlink()
                        pruned.append(str(path))
                    except OSError as exc:
                        log(f"  could not prune {path}: {exc}")

    if prune_loser_artifacts:
        kept = 0
        removed = 0
        for row in seed_rows:
            if row["seed"] == winner["seed"]:
                continue
            top_dir = Path(row["run_dir"]) / "results" / "top_layouts"
            if not top_dir.is_dir():
                continue
            for path in top_dir.iterdir():
                if path.suffix in (".png", ".json") and not path.name.endswith(".metrics.json"):
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        kept += 1
        if removed:
            log(f"  pruned {removed} losing layout artifact(s)")
    metrics = summary_metrics(summary)
    result = {
        "case": case,
        "plan": plan,
        "target_ratio": plan["ratio"],
        "canvas_side_mm": plan["side_mm"],
        "footprint_area_mm2": plan["footprint_area"],
        "footprint_canvas_ratio": plan["footprint_area"] / (plan["side_mm"] ** 2),
        "chiplets": plan["chiplets"],
        "best_seed": winner["seed"],
        "best_run_name": winner["run_name"],
        "ranked_by": winner.get("ranked_by"),
        "rlplanner_cost": winner["cost"],
        "avg_wirelength": winner.get("avg_wirelength"),
        "temperature": winner.get("temperature"),
        "canvas_utilization": metrics.get("canvas_utilization"),
        "silicon_canvas_utilization": metrics.get("silicon_canvas_utilization"),
        "bbox_utilization": metrics.get("bbox_utilization"),
        "reward": summary.get("reward"),
        "episode": summary.get("episode"),
        "termination_reason": winner.get("termination_reason"),
        "total_success_rate": winner.get("success_rate"),
        "executed_epochs": winner.get("executed_epochs"),
        "rl_solve_seconds": winner.get("rl_solve_seconds"),
        "thermal_tables_seconds": winner.get("thermal_tables_seconds"),
        "pruned_seed_checkpoints": pruned,
        "artifacts": artifacts,
        "seeds": [
            {
                "seed": row["seed"],
                "run_name": row["run_name"],
                "returncode": row.get("returncode"),
                "wall_seconds": row.get("wall_seconds"),
                "rl_solve_seconds": row.get("rl_solve_seconds"),
                "thermal_tables_seconds": row.get("thermal_tables_seconds"),
                "thermal_tables_forced": row.get("thermal_tables_forced"),
                "thermal_tables_status": row.get("thermal_tables_status"),
                "termination_reason": row.get("termination_reason"),
                "rlplanner_cost": row.get("cost"),
                "avg_wirelength": row.get("avg_wirelength"),
                "temperature": row.get("temperature"),
                "canvas_utilization": row.get("canvas_utilization"),
                "total_success_rate": row.get("success_rate"),
                "ranked_by": row.get("ranked_by"),
                "run_dir": row.get("run_dir"),
            }
            for row in sorted(seed_rows, key=lambda item: item["seed"])
        ],
    }
    with open(case_dir / "case_summary.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=str)

    util = result["canvas_utilization"]
    log(f"  best seed {result['best_seed']}: cost={result['rlplanner_cost']:.4f} "
        f"T={result['temperature']} avg_wl={result['avg_wirelength']} "
        f"canvas_util={util if util is None else round(float(util), 4)}")
    return result


def write_global_summary(runs_root: Path, case_results: list[dict], plan: list[dict],
                         args, started_at: str, seconds_total: float) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "seconds_total": seconds_total,
        "python": args.python or sys.executable,
        "seconds_per_run": args.seconds,
        "budget_margin_seconds": args.budget_margin,
        "thermal_inside_budget": bool(args.thermal_inside_budget),
        "thermal_mode": args.thermal_mode,
        "reuse_thermal_tables": bool(args.reuse_thermal_tables),
        "thermal_prep": ("inside_each_run_forced" if args.force_thermal_per_run
                         else "inside_each_run" if args.thermal_mode == "per_run"
                         else "before_runs"),
        "seeds": args.seed_list,
        "seeds_spec": args.seed_spec,
        "target_footprint_canvas_ratio": args.ratio,
        "cases_requested": [entry["case"] for entry in plan],
        "cases_completed": [entry["case"] for entry in case_results],
        "plan": plan,
        "results": case_results,
    }
    with open(runs_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)

    columns = [
        "case", "chiplets", "canvas_side_mm", "target_ratio", "best_seed",
        "rlplanner_cost", "avg_wirelength", "temperature", "canvas_utilization",
        "silicon_canvas_utilization", "total_success_rate", "termination_reason",
        "thermal_tables_seconds", "rl_solve_seconds", "wall_seconds",
        "checkpoint", "layout_json",
    ]
    usable = [result for result in case_results if result.get("best_seed") is not None]
    with open(runs_root / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in usable:
            artifacts = result.get("artifacts") or {}
            seed_rows_out = result.get("seeds") or []
            winner_row = next(
                (row for row in seed_rows_out if row.get("seed") == result.get("best_seed")),
                seed_rows_out[0] if seed_rows_out else {})
            writer.writerow({
                "case": result["case"],
                "chiplets": result["chiplets"],
                "canvas_side_mm": round(result["canvas_side_mm"], 4),
                "target_ratio": result["target_ratio"],
                "best_seed": result.get("best_seed"),
                "rlplanner_cost": result.get("rlplanner_cost"),
                "avg_wirelength": result.get("avg_wirelength"),
                "temperature": result.get("temperature"),
                "canvas_utilization": result.get("canvas_utilization"),
                "silicon_canvas_utilization": result.get("silicon_canvas_utilization"),
                "total_success_rate": result.get("total_success_rate"),
                "termination_reason": result.get("termination_reason"),
                "thermal_tables_seconds": winner_row.get("thermal_tables_seconds"),
                "rl_solve_seconds": result.get("rl_solve_seconds"),
                "wall_seconds": winner_row.get("wall_seconds"),
                "checkpoint": artifacts.get("checkpoint"),
                "layout_json": artifacts.get("layout"),
            })

    log("")
    log("=" * 118)
    log(f"{'case':<11}{'side':>8}{'seed':>5}{'cost':>12}{'avg_wl':>11}{'T(C)':>9}"
        f"{'canvas_u':>10}{'success':>9}{'stop':>14}")
    log("-" * 118)
    for result in usable:
        def fmt(value, spec="{:.4f}"):
            if value is None:
                return "-"
            try:
                return spec.format(float(value))
            except Exception:
                return str(value)
        log(f"{result['case']:<11}{result['canvas_side_mm']:>8.2f}"
            f"{str(result.get('best_seed')):>5}"
            f"{fmt(result.get('rlplanner_cost')):>12}"
            f"{fmt(result.get('avg_wirelength'), '{:.2f}'):>11}"
            f"{fmt(result.get('temperature'), '{:.2f}'):>9}"
            f"{fmt(result.get('canvas_utilization'), '{:.4f}'):>10}"
            f"{fmt(result.get('total_success_rate'), '{:.3f}'):>9}"
            f"{str(result.get('termination_reason')):>14}")
    log("=" * 118)
    log(f"summary.json: {runs_root / 'summary.json'}")
    log(f"summary.csv : {runs_root / 'summary.csv'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="all",
                        help="'all' or a comma separated list, e.g. Case6,acend910")
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="comma separated seeds; the token 'random' draws a fresh "
                             "seed per case (recorded in the summaries)")
    parser.add_argument("--seconds", type=float, default=3600.0,
                        help="per-run budget in seconds (default 3600). In the default "
                             "--thermal-mode per_run this is the TOTAL budget: the "
                             "thermal-table stage is charged against it")
    parser.add_argument("--budget-margin", type=float, default=30.0,
                        help="only used with --thermal-mode before_runs: added to "
                             "--seconds to compensate for train.py's own thermal check "
                             "(default 30)")
    parser.add_argument("--ratio", type=float, default=0.50,
                        help="target footprint area / canvas area (default 0.50)")
    parser.add_argument("--canvas-side", type=float, default=None,
                        help="override the derived canvas side for every case (mm)")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter used for train.py/generate_thermal_tables.py")
    parser.add_argument("--cplex-pythonpath", default=None,
                        help="explicit CPLEX python binding directory, e.g. "
                             "$CPLEX_STUDIO_DIR/cplex/python/3.10/x86-64_linux; "
                             "auto-discovered from CPLEX_STUDIO_DIR and /opt/ibm/ILOG "
                             "when omitted")
    parser.add_argument("--runs-dir", default=str(RL_DIR / "runs"),
                        help="output root (default baseline/RL/runs)")
    parser.add_argument("--smoke", action="store_true",
                        help="short validation: one case, one seed, 60s, separate names")
    parser.add_argument("--thermal-mode", choices=("per_run", "before_runs"),
                        default="per_run",
                        help="must be per_run: every seed regenerates its own thermal "
                             "tables inside its own --seconds total budget")
    parser.add_argument("--no-thermal-prep", action="store_true",
                        help="deprecated alias for --thermal-mode per_run")
    parser.add_argument("--thermal-inside-budget", action="store_true",
                        help="deprecated alias for --thermal-mode per_run")
    parser.add_argument("--reuse-thermal-tables", action="store_true",
                        help="with --thermal-mode per_run: do not force regeneration, so "
                             "a seed may reuse tables left by an earlier seed")
    parser.add_argument("--force-thermal", action="store_true",
                        help="with --thermal-mode before_runs: regenerate the tables "
                             "during the prep step even when they are already valid")
    parser.add_argument("--keep-seed-checkpoints", action="store_true",
                        help="keep every seed's checkpoint; by default only the winning "
                             "seed of each case keeps its model (~73MB each)")
    parser.add_argument("--prune-loser-artifacts", action="store_true",
                        help="also delete each losing seed's layout_*.png/json under "
                             "results/top_layouts (metrics.json and logs are kept)")
    parser.add_argument("--skip-env-check", action="store_true",
                        help="do not abort when the environment check fails")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and commands only")
    args = parser.parse_args(argv)

    if args.ratio <= 0.0 or args.ratio > 1.0:
        parser.error("--ratio must be in (0, 1]")
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.no_thermal_prep or args.thermal_inside_budget:
        args.thermal_mode = "per_run"          # legacy aliases
    if args.thermal_mode != "per_run":
        parser.error("thermal tables must be generated per seed inside the --seconds budget")
    if args.reuse_thermal_tables:
        parser.error("thermal-table reuse across seeds is disabled by experiment protocol")
    # The value passed to train.py is the complete wall-clock budget.  Thermal
    # characterization is forced for every seed and deducted from this budget.
    args.budget_margin = 0.0
    args.thermal_inside_budget = True
    args.no_thermal_prep = True
    args.force_thermal_per_run = True
    args.seed_spec = []
    for token in (item.strip() for item in str(args.seeds).split(",")):
        if not token:
            continue
        if token.lower() in ("random", "rand"):
            args.seed_spec.append("random")
            continue
        try:
            args.seed_spec.append(int(token))
        except ValueError:
            parser.error(f"--seeds entries must be integers or 'random', got {token!r}")
    if not args.seed_spec:
        parser.error("--seeds must list at least one seed")
    # Fixed seeds only; 'random' entries are drawn per case in main().
    args.seed_list = [item for item in args.seed_spec if isinstance(item, int)]

    if args.smoke:
        args.cases = "acend910" if args.cases == "all" else args.cases
        args.seeds = "0"
        args.seed_spec = [0]
        args.seed_list = [0]
        args.seconds = 60.0
        args.name_prefix = "smoke_"
    else:
        args.name_prefix = ""
    return args


def materialize_seeds(spec, rng: random.Random) -> list[int]:
    """Turn a seed spec into concrete seeds, drawing fresh ones for 'random'."""
    return [rng.randrange(1, 10 ** 9) if item == "random" else int(item) for item in spec]


def main(argv=None) -> int:
    args = parse_args(argv)
    runs_root = Path(args.runs_dir).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    console_dir = runs_root / "_console"
    driver_log = runs_root / f"driver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    runs_root.mkdir(parents=True, exist_ok=True)
    try:
        _LOG_STREAMS.append(open(driver_log, "w", encoding="utf-8"))
    except OSError:
        pass

    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()

    log("=" * 118)
    log("RL chiplet placement - all cases x seeds, time-boxed PPO training")
    log(f"  cases     : {args.cases}   seeds: {args.seed_spec}")
    if args.thermal_mode == "per_run":
        reuse = "reused when valid" if args.reuse_thermal_tables else "regenerated per run"
        log(f"  per run   : {args.seconds:.0f}s TOTAL budget "
            f"(thermal tables inside the run, {reuse})")
    else:
        log(f"  per run   : {args.seconds:.0f}s RL budget (+{args.budget_margin:.0f}s margin; "
            f"thermal tables prepared once per case before the runs)")
    log(f"  canvas    : footprint area / canvas area = {args.ratio:.2%}  "
        f"side = sqrt(footprint_area / ratio)")
    log(f"  runs dir  : {runs_root}")
    log(f"  driver log: {driver_log}")
    log("=" * 118)

    cases = discover_cases(args.cases)
    plan = build_plan(cases, args.ratio, args.canvas_side)
    log("")
    log(f"{'case':<11}{'chiplets':>9}{'footprint':>12}{'body':>11}{'side(mm)':>10}"
        f"{'body/canvas':>12}{'fastTM cfg':>11}")
    for entry in plan:
        body_ratio = entry["body_area"] / (entry["side_mm"] ** 2)
        log(f"{entry['case']:<11}{entry['chiplets']:>9}{entry['footprint_area']:>12.1f}"
            f"{entry['body_area']:>11.1f}{entry['side_mm']:>10.2f}{body_ratio:>12.3f}"
            f"{str(entry['has_fasttm_config']):>11}")
    with open(runs_root / "_plan.json", "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)

    if args.force_thermal_per_run and args.seconds < 600:
        log("")
        log(f"WARNING: --thermal-mode per_run regenerates the thermal tables inside every "
            f"run, and {args.seconds:.0f}s may be less than that stage takes "
            f"(observed 60-130s, more for many-chiplet cases). The stage is terminated "
            f"when the total budget expires. Use --seconds >= 600 for a quick check.")

    if args.dry_run:
        log("")
        log("DRY-RUN: no training launched.")
        placeholder = materialize_seeds(args.seed_spec, _SEED_RNG)
        for entry in plan:
            log(f"  {entry['case']}: side={entry['side_mm']:.4f}mm "
                f"intp={entry['side_mm']:.4f}mm seeds={args.seed_spec} "
                f"names={args.name_prefix}{entry['case']}_seed<s>")
            run_seed(args.python, f"{args.name_prefix}{entry['case']}_seed0",
                     Path(entry["json"]), placeholder[0],
                     args.seconds + args.budget_margin, entry["side_mm"],
                     runs_root, console_dir, dict(os.environ),
                     force_thermal=args.force_thermal_per_run, dry_run=True)
        return 0

    # Make `import cplex` work for the child processes without touching the
    # conda environment: IBM ships the bindings inside the Studio install.
    cplex_path = args.cplex_pythonpath
    if cplex_path:
        probe = subprocess.run([args.python, "-c", "import cplex"], capture_output=True,
                               text=True, env=with_pythonpath(cplex_path))
        if probe.returncode != 0:
            log(f"FAIL --cplex-pythonpath does not make 'import cplex' work: {cplex_path}")
            return 2
    else:
        cplex_path = discover_cplex_python_path(args.python)
    child_env = with_pythonpath(cplex_path)

    if not env_check(args.python, strict=not args.skip_env_check, child_env=child_env,
                     cplex_path=cplex_path):
        return 2

    budget = args.seconds + args.budget_margin
    case_results: list[dict] = []
    for entry in plan:
        case = entry["case"]
        side = float(entry["side_mm"])
        case_dir = runs_root / f"{args.name_prefix}{case}"
        log("")
        log(f"### {case}: side={side:.4f}mm chiplets={entry['chiplets']} "
            f"footprint={entry['footprint_area']:.1f}mm2")

        if not args.no_thermal_prep:
            prep = prepare_thermal_tables(case, side, entry["chiplets"], case_dir,
                                          python=args.python, child_env=child_env,
                                          force=args.force_thermal)
            entry["thermal_prep"] = prep
            log(f"  thermal tables: {prep['status']} ({prep.get('seconds', 0.0):.1f}s)")
            if prep.get("status", "").startswith("failed"):
                log(f"  FAIL thermal-table preparation failed; see {prep.get('log')}")
                continue

        seed_rows = []
        case_seeds = materialize_seeds(args.seed_spec, _SEED_RNG)
        entry["seeds_used"] = case_seeds
        log(f"  seeds: {case_seeds}"
            + ("  (drawn randomly; recorded in case_summary.json)"
               if "random" in args.seed_spec else ""))
        for seed in case_seeds:
            run_name = f"{args.name_prefix}{case}_seed{seed}"
            info = run_seed(args.python, run_name, Path(entry["json"]), seed, budget,
                            side, runs_root, console_dir, child_env,
                            force_thermal=args.force_thermal_per_run)
            if info.get("dry_run"):
                continue
            seed_rows.append(read_run_results(info))
            if info.get("thermal_tables_seconds") is not None:
                log(f"    thermal stage {info['thermal_tables_seconds']:.1f}s "
                    f"(forced={info.get('thermal_tables_forced')}), "
                    f"RL {info.get('rl_solve_seconds')}")

        if not seed_rows:
            continue
        case_results.append(collect_case(
            case, entry, seed_rows, case_dir,
            prune_seed_checkpoints=not args.keep_seed_checkpoints,
            prune_loser_artifacts=args.prune_loser_artifacts))

    write_global_summary(runs_root, case_results, plan, args, started_at,
                         time.perf_counter() - started)
    usable = [result for result in case_results if result.get("best_seed") is not None]
    log("")
    log(f"done: {len(usable)}/{len(plan)} case(s) produced a cost-ranked best solution, "
        f"{time.perf_counter() - started:.0f}s total")
    if len(usable) != len(plan):
        log("NOTE: a case without a finite rlplanner_cost means no episode finished with a")
        log("      successful placement before the time limit, or the thermal reward failed.")
        log("      Check runs/<case>_seed<k>/train.log and thermal_tables.log.")
    return 0 if len(usable) == len(plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
