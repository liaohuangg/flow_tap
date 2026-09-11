"""
PPO训练脚本 - 芯片布局强化学习

使用Proximal Policy Optimization (PPO)算法训练芯片布局策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict
import json
from pathlib import Path
from datetime import datetime
import argparse
import subprocess
import sys
import time
import configparser
import shutil
import random

try:
    from .env import (
        DEFAULT_FOOTPRINT_MODE,
        FOOTPRINT_MODES,
        ChipletPlacementEnv,
        create_env_from_json,
    )
except ImportError:
    from env import (
        DEFAULT_FOOTPRINT_MODE,
        FOOTPRINT_MODES,
        ChipletPlacementEnv,
        create_env_from_json,
    )
from copy import deepcopy
try:
    import matplotlib
    # 使用无界面后端，避免在服务器/无显示环境下分配位图失败
    matplotlib.use('Agg')
except ImportError:
    matplotlib = None
try:
    from .unit import export_layout_result_json, visualize_layout_with_bridges
except ImportError:
    from unit import export_layout_result_json, visualize_layout_with_bridges
import contextlib
import os

RL_DIR = Path(__file__).resolve().parent
LOCAL_FASTTM_DIR = RL_DIR / "fastTM"

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def _set_global_seed(seed: int | None) -> int | None:
    """Seed every stochastic component used by the PPO training process."""
    if seed is None:
        return None
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass
    return seed


class Tee:
    """Write stdout/stderr to both console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                stream.write(data.encode(encoding, errors="replace").decode(encoding))
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _safe_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name).strip())
    return safe or datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_dir_for_name(name: str) -> Path:
    return RL_DIR / "runs" / _safe_run_name(name)


def _thermal_intp_size_for_generation(json_path: str, env_kwargs: Dict) -> float:
    if env_kwargs.get("thermal_intp_size") is not None:
        return float(env_kwargs["thermal_intp_size"])
    # Match local generate_thermal_tables.py default.
    return 50.0


def _align_default_canvas_to_thermal(json_path: str, env_kwargs: Dict) -> Dict:
    """Use the fastTM interposer size as the default RL canvas size."""
    intp_size = _thermal_intp_size_for_generation(json_path, env_kwargs)
    updates = {}
    if env_kwargs.get("max_width") is None:
        env_kwargs["max_width"] = intp_size
        updates["max_width"] = intp_size
    if env_kwargs.get("max_height") is None:
        env_kwargs["max_height"] = intp_size
        updates["max_height"] = intp_size
    return {
        "thermal_intp_size": intp_size,
        "updates": updates,
    }


def _existing_fasttm_intp_size(json_path: str) -> float | None:
    cfg_path = LOCAL_FASTTM_DIR / "configs" / f"benchmark_{Path(json_path).stem}.cfg"
    if not cfg_path.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        if not parser.read(cfg_path):
            return None
        return parser.getfloat("interposer", "intp_size")
    except Exception:
        return None


def _generate_thermal_tables(
    json_path: str,
    env_kwargs: Dict,
    run_dir: Path,
    force: bool = False,
) -> Dict:
    """Generate/check fastTM tables and return timing metadata."""
    thermal_log = run_dir / "thermal_tables.log"
    script = RL_DIR / "generate_thermal_tables.py"
    hotspot_bin = LOCAL_FASTTM_DIR / "util" / "hotspot"
    intp_size = _thermal_intp_size_for_generation(json_path, env_kwargs)
    existing_intp_size = _existing_fasttm_intp_size(json_path)
    force_for_size_change = (
        existing_intp_size is not None
        and abs(float(existing_intp_size) - float(intp_size)) > 1e-9
    )
    effective_force = bool(force or force_for_size_change)
    cmd = [
        sys.executable,
        str(script),
        str(Path(json_path).resolve()),
        "--intp-size",
        str(intp_size),
        "--overwrite-config",
    ]
    if effective_force:
        cmd.append("--force")

    start = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    status = "ok"
    returncode = None

    with open(thermal_log, "w", encoding="utf-8") as log_file:
        log_file.write(f"command: {' '.join(cmd)}\n")
        log_file.write(f"cwd: {LOCAL_FASTTM_DIR}\n")
        log_file.write(f"hotspot: {hotspot_bin}\n")
        log_file.write(f"started_at: {started_at}\n\n")
        log_file.flush()
        try:
            if not hotspot_bin.exists():
                status = f"error:missing_hotspot:{hotspot_bin}"
                log_file.write(f"{status}\n")
            elif not os.access(hotspot_bin, os.X_OK):
                status = f"error:hotspot_not_executable:{hotspot_bin}"
                log_file.write(f"{status}\n")
            else:
                result = subprocess.run(
                    cmd,
                    cwd=str(LOCAL_FASTTM_DIR),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                returncode = result.returncode
                if result.returncode != 0:
                    status = "failed"
        except Exception as exc:
            status = f"error:{type(exc).__name__}: {exc}"
            log_file.write(f"\n{status}\n")

    elapsed = time.perf_counter() - start
    return {
        "status": status,
        "returncode": returncode,
        "seconds": elapsed,
        "started_at": started_at,
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "cwd": str(LOCAL_FASTTM_DIR),
        "hotspot": str(hotspot_bin),
        "log": str(thermal_log),
        "intp_size": intp_size,
        "existing_intp_size": existing_intp_size,
        "force": effective_force,
        "force_requested": force,
        "force_reason": "intp_size_changed" if force_for_size_change and not force else None,
    }


def _layout_metrics(env: ChipletPlacementEnv, layout: Dict) -> Dict:
    """Compute export metrics, including rlplanner temperature when possible."""
    if not layout:
        return {}
    geometry_metrics = _layout_geometry_metrics(env, layout)
    try:
        rl_reward, metrics = env._calculate_rlplanner_terminal_reward(layout)
        metrics = dict(metrics)
        metrics["rlplanner_reward"] = rl_reward
        metrics.update(geometry_metrics)
        return metrics
    except Exception as exc:
        geometry_metrics["rlplanner_reward_error"] = f"{type(exc).__name__}: {exc}"
        return geometry_metrics


def _layout_geometry_metrics(env: ChipletPlacementEnv, layout: Dict) -> Dict:
    """Compute physical geometry metrics including the microbump envelopes."""
    if not layout:
        return {}

    chiplets = list(layout.values())
    occupied = [env._occupied_bounds(chip, chip_id) for chip_id, chip in layout.items()]
    x_min = min(bounds[0] for bounds in occupied)
    y_min = min(bounds[1] for bounds in occupied)
    x_max = max(bounds[2] for bounds in occupied)
    y_max = max(bounds[3] for bounds in occupied)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    bbox_area = bbox_width * bbox_height
    chiplet_area = sum(chip.width * chip.height for chip in chiplets)
    # Sum the placed occupied envelopes. Deriving the size from _occupied_bounds
    # keeps this correct for rotated chiplets and for asymmetric declared
    # footprints, where the halo differs per axis.
    occupied_area = sum(
        (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) for bounds in occupied
    )
    canvas_area = float(env.max_width) * float(env.max_height)

    return {
        "chiplet_area": float(chiplet_area),
        "microbump_occupied_area": float(occupied_area),
        "microbump_halos": dict(env.microbump_halos),
        "microbump_footprints": dict(env.footprint_sizes),
        "occupied_envelope_sources": dict(env.footprint_sources),
        "footprint_mode": env.footprint_mode,
        "bbox_min_x": float(x_min),
        "bbox_min_y": float(y_min),
        "bbox_max_x": float(x_max),
        "bbox_max_y": float(y_max),
        "bbox_width": float(bbox_width),
        "bbox_height": float(bbox_height),
        "bbox_area": float(bbox_area),
        "bbox_utilization": float(occupied_area / bbox_area) if bbox_area > 0.0 else None,
        "silicon_bbox_utilization": float(chiplet_area / bbox_area) if bbox_area > 0.0 else None,
        "canvas_area": float(canvas_area),
        "canvas_utilization": float(occupied_area / canvas_area) if canvas_area > 0.0 else None,
        "silicon_canvas_utilization": float(chiplet_area / canvas_area) if canvas_area > 0.0 else None,
    }


def _metric_text(value, precision: int = 4) -> str:
    if value is None:
        return "None"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "NaN"
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _best_layout_metrics_data(
    episode: int,
    reward: float,
    success: bool,
    export_metrics: Dict,
    timing: Dict,
) -> Dict:
    return {
        "episode": episode,
        "reward": float(reward),
        "success": bool(success),
        "wirelength": export_metrics.get("rlplanner_total_wirelength"),
        "avg_wirelength": export_metrics.get("rlplanner_avg_wirelength"),
        "emib_wirelength": export_metrics.get("rlplanner_emib_wirelength"),
        "normal_wirelength": export_metrics.get("rlplanner_normal_wirelength"),
        "total_wire_count": export_metrics.get("rlplanner_total_wire_count"),
        "wirelength_source": export_metrics.get("rlplanner_wirelength_source"),
        "temperature": export_metrics.get("rlplanner_temperature"),
        "rlplanner_cost": export_metrics.get("rlplanner_cost"),
        "thermal_error": export_metrics.get("rlplanner_reward_error"),
        "bounding_rect_area": export_metrics.get("bbox_area"),
        "runtime": timing,
    }


def _export_best_layout_artifacts(
    layout: Dict,
    problem,
    output_dir: Path,
    prefix: str,
    episode: int,
    reward: float,
    success: bool,
    export_metrics: Dict,
    timing: Dict,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}_ep{episode}_r{reward:.2f}.json"
    metrics_path = output_dir / f"{prefix}_ep{episode}_r{reward:.2f}.metrics.json"
    img_path = output_dir / f"{prefix}_ep{episode}_r{reward:.2f}.png"

    metrics_data = _best_layout_metrics_data(
        episode=episode,
        reward=reward,
        success=success,
        export_metrics=export_metrics,
        timing=timing,
    )
    export_layout_result_json(layout, problem, str(json_path), metrics=export_metrics)
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics_data, mf, indent=2, ensure_ascii=False, default=str)

    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                visualize_layout_with_bridges(
                    layout,
                    problem,
                    output_file=str(img_path),
                    show_bridges=False,
                    show_coordinates=True,
                )
    except Exception:
        pass

    return {
        "reward": reward,
        "episode": episode,
        "json": str(json_path),
        "metrics_json": str(metrics_path),
        "img": str(img_path),
        "metrics": export_metrics,
        "timing": timing,
        "success": success,
    }


class ActorCritic(nn.Module):
    """Shared encoder with policy and value heads."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        features = self.feature(x)
        return self.actor(features), self.critic(features)

    def get_action(self, obs, valid_actions):
        if not valid_actions:
            raise ValueError("valid_actions must not be empty")
        action_logits, value = self.forward(obs)
        mask = torch.full_like(action_logits, float("-inf"))
        mask[:, valid_actions] = 0.0
        dist = Categorical(logits=action_logits + mask)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(0), value.squeeze()


class PPOTrainer:
    """Masked PPO using rollout batches and shuffled minibatches."""

    def __init__(
        self,
        env: ChipletPlacementEnv,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
        num_minibatches: int = 4,
        update_epochs: int = 4,
        target_kl: float = 0.015,
        load_optimizer: bool = True,
    ):
        if num_minibatches <= 0 or update_epochs <= 0:
            raise ValueError("num_minibatches and update_epochs must be positive")
        self.env = env
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.num_minibatches = int(num_minibatches)
        self.update_epochs = int(update_epochs)
        self.target_kl = float(target_kl) if target_kl is not None else None
        self.learning_rate = float(lr)
        self.model = ActorCritic(env.observation_dim, env.action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.total_optimizer_steps = 0
        self.last_episode_info: Dict = {}
        self.reset_buffer()

    def reset_buffer(self):
        self.observations = []
        self.actions = []
        self.valid_action_sets = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    @property
    def buffer_size(self) -> int:
        return len(self.observations)

    def collect_episode(self) -> Tuple[float, bool, Dict]:
        obs = self.env.reset()
        done = False
        total_reward = 0.0
        info: Dict = {}
        max_steps = self.env.num_chiplets * 10

        for _ in range(max_steps):
            if done:
                break
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                self.last_episode_info = {"error": "no_valid_actions"}
                return total_reward, False, deepcopy(self.env.state.layout)

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_tensor, valid_actions)
            next_obs, reward, done, info = self.env.step(action)

            self.observations.append(np.asarray(obs, dtype=np.float32))
            self.actions.append(int(action))
            self.valid_action_sets.append(np.asarray(valid_actions, dtype=np.int32))
            self.log_probs.append(float(log_prob.item()))
            self.values.append(float(value.item()))
            self.rewards.append(float(reward))
            self.dones.append(bool(done))
            obs = next_obs
            total_reward += float(reward)

            if "error" in info:
                self.last_episode_info = dict(info)
                return total_reward, False, deepcopy(self.env.state.layout)

        self.last_episode_info = dict(info)
        return total_reward, bool(done), deepcopy(self.env.state.layout)

    def compute_returns(self) -> torch.Tensor:
        returns = []
        discounted_return = 0.0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_return = 0.0
            discounted_return = reward + self.gamma * discounted_return
            returns.append(discounted_return)
        returns.reverse()
        return torch.as_tensor(returns, dtype=torch.float32)

    def _masked_distribution(self, logits: torch.Tensor, sample_indices: np.ndarray) -> Categorical:
        mask = torch.full_like(logits, float("-inf"))
        for row, sample_idx in enumerate(sample_indices):
            valid = torch.as_tensor(
                self.valid_action_sets[int(sample_idx)], dtype=torch.long, device=logits.device
            )
            mask[row, valid] = 0.0
        return Categorical(logits=logits + mask)

    def update(self) -> Dict:
        sample_count = self.buffer_size
        if sample_count < 2:
            self.reset_buffer()
            return {}

        observations = torch.as_tensor(
            np.asarray(self.observations), dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(self.actions, dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=device)
        old_values = torch.as_tensor(self.values, dtype=torch.float32, device=device)
        returns = self.compute_returns().to(device)
        advantages = returns - old_values
        if advantages.numel() > 1 and advantages.std(unbiased=False) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        minibatch_size = max(1, int(np.ceil(sample_count / self.num_minibatches)))
        totals = {
            "loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "kl_div": 0.0,
            "clip_fraction": 0.0,
        }
        optimizer_steps = 0
        stopped_early = False

        for _ in range(self.update_epochs):
            permutation = np.random.permutation(sample_count)
            for start in range(0, sample_count, minibatch_size):
                mb_indices = permutation[start:start + minibatch_size]
                mb_tensor = torch.as_tensor(mb_indices, dtype=torch.long, device=device)
                logits, values = self.model(observations[mb_tensor])
                dist = self._masked_distribution(logits, mb_indices)
                new_log_probs = dist.log_prob(actions[mb_tensor])
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - old_log_probs[mb_tensor]
                ratio = log_ratio.exp()
                mb_advantages = advantages[mb_tensor]
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
                ) * mb_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                critic_loss = nn.functional.mse_loss(values.squeeze(-1), returns[mb_tensor])
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.total_optimizer_steps += 1
                optimizer_steps += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.epsilon).float().mean()
                totals["loss"] += float(loss.item())
                totals["actor_loss"] += float(actor_loss.item())
                totals["critic_loss"] += float(critic_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["kl_div"] += float(approx_kl.item())
                totals["clip_fraction"] += float(clip_fraction.item())

                if self.target_kl is not None and approx_kl > self.target_kl:
                    stopped_early = True
                    break
            if stopped_early:
                break

        self.reset_buffer()
        divisor = max(optimizer_steps, 1)
        metrics = {name: value / divisor for name, value in totals.items()}
        metrics.update({
            "optimizer_steps": optimizer_steps,
            "samples": sample_count,
            "minibatch_size": minibatch_size,
            "stopped_early": stopped_early,
        })
        return metrics

    def snapshot_model_state(self) -> Dict[str, torch.Tensor]:
        """Copy the current policy to CPU without retaining the live GPU graph."""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }

    def save(
        self,
        path: str,
        model_state_dict: Dict[str, torch.Tensor] | None = None,
        include_optimizer: bool = True,
        metadata: Dict | None = None,
    ):
        payload = {
            "format_version": 2,
            "model_state_dict": model_state_dict or self.model.state_dict(),
            "trainer_config": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
                "num_minibatches": self.num_minibatches,
                "update_epochs": self.update_epochs,
                "target_kl": self.target_kl,
                "total_optimizer_steps": self.total_optimizer_steps,
            },
            "env_config": {
                "grid_resolution": self.env.grid_resolution,
                "max_width": self.env.max_width,
                "max_height": self.env.max_height,
                "observation_dim": self.env.observation_dim,
                "action_dim": self.env.action_dim,
                "terminal_rlplanner_cost_scale": self.env.terminal_rlplanner_cost_scale,
                "thermal_intp_size": self.env.thermal_intp_size,
                "rlplanner_root": str(self.env.rlplanner_root),
                "rlplanner_table_dir": (
                    str(self.env.rlplanner_table_dir)
                    if self.env.rlplanner_table_dir is not None else None
                ),
                "exact_action_slots": self.env.exact_action_slots,
                "placement_order": self.env.placement_order,
                "placement_area_weight": self.env.placement_area_weight,
                "placement_pin_weight": self.env.placement_pin_weight,
                "microbump_pitch_mm": 0.045,
                "microbump_halos": dict(self.env.microbump_halos),
                "microbump_in_placement_constraints": True,
            },
        }
        if include_optimizer:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if metadata is not None:
            payload["metadata"] = metadata
        torch.save(payload, path)
        print(f"模型已保存到 {path}")

    def load(self, path: str, load_optimizer: bool = False):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"模型已从 {path} 加载")
    
 

def train(
    json_path: str,
    num_episodes: int = 600,
    save_interval: int = 10,
    log_interval: int = 1,
    trainer: PPOTrainer = None,
    name: str = None,
    generate_thermal_tables: bool = True,
    force_thermal_tables: bool = False,
    rollout_batch_size: int = 480,
    learning_rate: float = 2.5e-4,
    num_minibatches: int = 4,
    ppo_update_epochs: int = 4,
    target_kl: float = 0.015,
    save_checkpoint: bool = False,
    seed: int | None = None,
    time_limit_seconds: float | None = None,
    **env_kwargs
):
    """Run paper-style PPO epochs and export artifacts under runs/<name>."""
    if not generate_thermal_tables:
        raise ValueError("Thermal-table generation/checking is mandatory for training")
    if time_limit_seconds is not None and float(time_limit_seconds) <= 0:
        raise ValueError("time_limit_seconds must be positive")
    seed = _set_global_seed(seed)
    json_path = str(json_path)
    if name is None:
        case_name = Path(json_path).stem
        name = f"{case_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = _run_dir_for_name(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if save_checkpoint:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "results" / "top_layouts").mkdir(parents=True, exist_ok=True)
    canvas_defaults = _align_default_canvas_to_thermal(json_path, env_kwargs)

    config = {
        "name": name,
        "run_dir": str(run_dir),
        "json_path": str(Path(json_path).resolve()),
        "num_epochs": num_episodes,
        "rollout_batch_size": rollout_batch_size,
        "learning_rate": learning_rate,
        "num_minibatches": num_minibatches,
        "ppo_update_epochs": ppo_update_epochs,
        "target_kl": target_kl,
        "save_interval": save_interval,
        "save_checkpoint": save_checkpoint,
        "seed": seed,
        "time_limit_seconds": time_limit_seconds,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "log_interval": log_interval,
        "generate_thermal_tables": generate_thermal_tables,
        "force_thermal_tables": force_thermal_tables,
        "microbump_model": {
            "pitch_mm": 0.045,
            "capacity_source": "FastTM PassiveInterposer.compute_ubump_overhead",
            "included_in_bounds_and_overlap": True,
        },
        "env_kwargs": env_kwargs,
        "canvas_defaults": canvas_defaults,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    timing = {
        "name": name,
        "run_dir": str(run_dir),
        "started_at": config["started_at"],
        "thermal_tables": None,
        "rl_solve": None,
        "best_layout": None,
        "total_seconds": None,
    }

    total_start = time.perf_counter()
    train_log = run_dir / "train.log"
    with open(train_log, "w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"Run name: {name}")
            print(f"Run directory: {run_dir}")
            print(f"Training log: {train_log}")
            if canvas_defaults["updates"]:
                print(
                    "Default RL canvas aligned to fastTM interposer: "
                    f"{env_kwargs['max_width']} x {env_kwargs['max_height']}"
                )

            print("\n生成/检查 fastTM 热阻表...")
            thermal_info = _generate_thermal_tables(
                json_path=json_path,
                env_kwargs=env_kwargs,
                run_dir=run_dir,
                force=force_thermal_tables,
            )
            timing["thermal_tables"] = thermal_info
            print(f"  热阻表阶段状态: {thermal_info['status']}")
            print(f"  热阻表阶段耗时: {thermal_info['seconds']:.3f}s")
            print(f"  热阻表日志: {thermal_info['log']}")
            if thermal_info["status"] != "ok":
                raise RuntimeError(f"FastTM thermal-table stage failed: {thermal_info['status']}")

            remaining_seconds = None
            if time_limit_seconds is not None:
                remaining_seconds = float(time_limit_seconds) - (time.perf_counter() - total_start)
                if remaining_seconds <= 0:
                    raise TimeoutError(
                        "The total experiment budget was exhausted during thermal-table generation"
                    )

            rl_start = time.perf_counter()
            rl_started_at = datetime.now().isoformat(timespec="seconds")
            pre_rl_seconds = time.perf_counter() - total_start
            trainer_result = _train_impl(
                json_path=json_path,
                num_episodes=num_episodes,
                save_interval=save_interval,
                save_checkpoint=save_checkpoint,
                log_interval=log_interval,
                trainer=trainer,
                run_dir=run_dir,
                pre_rl_seconds=pre_rl_seconds,
                run_started_at=config["started_at"],
                rollout_batch_size=rollout_batch_size,
                learning_rate=learning_rate,
                num_minibatches=num_minibatches,
                ppo_update_epochs=ppo_update_epochs,
                target_kl=target_kl,
                seed=seed,
                time_limit_seconds=remaining_seconds,
                **env_kwargs,
            )
            rl_seconds = time.perf_counter() - rl_start
            timing["rl_solve"] = {
                "seconds": rl_seconds,
                "started_at": rl_started_at,
                "ended_at": datetime.now().isoformat(timespec="seconds"),
            }
            timing["best_layout"] = getattr(trainer_result, "best_layout_info", None)
            timing["best_cost_layout"] = getattr(trainer_result, "best_cost_layout_info", None)
            timing["seed"] = seed
            timing["time_limit_seconds"] = time_limit_seconds
            timing["termination_reason"] = getattr(trainer_result, "termination_reason", "epochs_complete")
            timing["completed_epochs"] = getattr(trainer_result, "completed_epochs", num_episodes)
            timing["total_transitions"] = getattr(trainer_result, "total_transitions", None)
            timing["total_seconds"] = time.perf_counter() - total_start
            timing["ended_at"] = datetime.now().isoformat(timespec="seconds")
            print("\n运行时间统计:")
            print(f"  热阻表: {timing['thermal_tables']['seconds']:.3f}s")
            print(f"  RL求解: {timing['rl_solve']['seconds']:.3f}s")
            print(f"  总时间: {timing['total_seconds']:.3f}s")
            if timing["best_layout"] is not None:
                best_timing = timing["best_layout"].get("timing", {})
                print(f"  best layout发现时间: {best_timing.get('found_at')}")
                print(f"  best layout累计RL时间: {_metric_text(best_timing.get('rl_elapsed_seconds'), 3)}s")
                print(f"  best layout累计总时间: {_metric_text(best_timing.get('total_elapsed_seconds'), 3)}s")

    with open(run_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False, default=str)

    trainer_result.run_dir = run_dir
    return trainer_result


def _train_impl(
    json_path: str,
    num_episodes: int = 600,
    save_interval: int = 10,
    log_interval: int = 1,
    trainer: PPOTrainer = None,
    run_dir: Path = Path("."),
    pre_rl_seconds: float = 0.0,
    run_started_at: str = None,
    rollout_batch_size: int = 480,
    learning_rate: float = 2.5e-4,
    num_minibatches: int = 4,
    ppo_update_epochs: int = 4,
    target_kl: float = 0.015,
    save_checkpoint: bool = False,
    seed: int | None = None,
    time_limit_seconds: float | None = None,
    **env_kwargs,
):
    """Train for PPO epochs; each epoch collects rollout_batch_size transitions."""
    num_epochs = int(num_episodes)
    if num_epochs <= 0 or rollout_batch_size <= 0:
        raise ValueError("num_epochs and rollout_batch_size must be positive")
    _set_global_seed(seed)

    print("=" * 70)
    print("PPO训练 - 芯片布局优化")
    print("=" * 70)
    if trainer is None:
        env = create_env_from_json(json_path, **env_kwargs)
        trainer = PPOTrainer(
            env,
            lr=learning_rate,
            gamma=0.99,
            epsilon=0.1,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            num_minibatches=num_minibatches,
            update_epochs=ppo_update_epochs,
            target_kl=target_kl,
        )
    else:
        env = trainer.env

    print("\n环境信息:")
    print(f"  芯片数量: {env.num_chiplets}")
    print(f"  放置顺序: {env.placement_order}")
    print(f"  观察维度: {env.observation_dim}")
    print(f"  动作维度: {env.action_dim}")
    print(f"  rollout batch: {rollout_batch_size} transitions")
    print(f"  minibatches: {trainer.num_minibatches}")
    print(f"  PPO update epochs: {trainer.update_epochs}")
    print(f"  随机种子: {seed if seed is not None else 'unfixed'}")
    if time_limit_seconds is not None:
        print(f"  剩余总时间预算: {time_limit_seconds:.1f}s")

    top_dir = run_dir / "results" / "top_layouts"
    top_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    if save_checkpoint:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    best_layout_info = None
    best_epoch_reward = float("-inf")
    best_model_path = None
    best_model_state = None
    # Best-by-rlplanner_cost solution (avg wirelength + temperature penalty).
    # The reward also contains utilization/placement terms, so its optimum is not
    # the cost optimum; this is tracked separately and exported alongside.
    best_cost = float("inf")
    best_cost_reward = None
    best_cost_layout_info = None
    best_cost_model_state = None
    best_cost_model_path = None
    epoch_mean_rewards = []
    total_rollouts = 0
    total_successes = 0
    total_transitions = 0
    train_loop_start = time.perf_counter()
    last_log_time = train_loop_start
    checkpoint_reserve = 0.0
    if save_checkpoint and time_limit_seconds is not None:
        checkpoint_reserve = min(30.0, max(5.0, float(time_limit_seconds) * 0.01))
    training_deadline = None
    if time_limit_seconds is not None:
        usable_seconds = max(0.0, float(time_limit_seconds) - checkpoint_reserve)
        training_deadline = train_loop_start + usable_seconds
    termination_reason = "epochs_complete"
    completed_epochs = 0
    executed_epochs = 0
    training_deadline = None
    if time_limit_seconds is not None:
        checkpoint_reserve = min(30.0, max(5.0, float(time_limit_seconds) * 0.01)) if save_checkpoint else 0.0
        training_deadline = train_loop_start + max(0.0, float(time_limit_seconds) - checkpoint_reserve)
    termination_reason = "epochs_complete"
    completed_epochs = 0
    executed_epochs = 0

    print("\n开始训练...")
    print("-" * 70)

    for epoch in range(1, num_epochs + 1):
        if training_deadline is not None and time.perf_counter() >= training_deadline:
            termination_reason = "time_limit"
            break

        epoch_rewards = []
        epoch_successes = 0
        epoch_transitions_start = trainer.buffer_size
        collection_attempts = 0
        time_limit_reached = False

        while trainer.buffer_size - epoch_transitions_start < rollout_batch_size:
            before = trainer.buffer_size
            reward, success, layout = trainer.collect_episode()
            collection_attempts += 1
            total_rollouts += 1
            epoch_rewards.append(float(reward))
            if success:
                epoch_successes += 1
                total_successes += 1

            terminal_metrics = {
                key: value for key, value in trainer.last_episode_info.items()
                if key.startswith("rlplanner_") or key.startswith("thermal_")
            }
            if layout:
                terminal_metrics.update(_layout_geometry_metrics(env, layout))

            episode_cost = terminal_metrics.get("rlplanner_cost")
            if (
                success
                and layout
                and episode_cost is not None
                and np.isfinite(episode_cost)
                and float(episode_cost) < best_cost
            ):
                best_cost = float(episode_cost)
                best_cost_reward = float(reward)
                if save_checkpoint:
                    best_cost_model_state = trainer.snapshot_model_state()
                cost_timing = {
                    "run_started_at": run_started_at,
                    "found_at": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch,
                    "rollout": total_rollouts,
                    "rl_elapsed_seconds": time.perf_counter() - train_loop_start,
                    "total_elapsed_seconds": pre_rl_seconds + (time.perf_counter() - train_loop_start),
                    "pre_rl_seconds": pre_rl_seconds,
                }
                best_cost_layout_info = _export_best_layout_artifacts(
                    layout=layout,
                    problem=env.problem,
                    output_dir=top_dir,
                    prefix="layout_best_cost",
                    episode=epoch,
                    reward=reward,
                    success=True,
                    export_metrics=terminal_metrics,
                    timing=cost_timing,
                )
                best_cost_layout_info["rlplanner_cost"] = best_cost
                with open(run_dir / "best_cost_summary.json", "w", encoding="utf-8") as f:
                    json.dump(best_cost_layout_info, f, indent=2, ensure_ascii=False, default=str)
                print("BEST_COST_LAYOUT")
                print(f"  epoch: {epoch}/{num_epochs}, rollout: {total_rollouts}")
                print(f"  rlplanner_cost: {best_cost:.4f}")
                print(f"  reward: {reward:.4f}")
                print(f"  温度: {_metric_text(terminal_metrics.get('rlplanner_temperature'))}")
                print(f"  CPLEX平均线长: {_metric_text(terminal_metrics.get('rlplanner_avg_wirelength'))}")

            if success and layout and reward > best_epoch_reward:
                best_epoch_reward = float(reward)
                if save_checkpoint:
                    best_model_state = trainer.snapshot_model_state()
                found_at = datetime.now().isoformat(timespec="seconds")
                rl_elapsed = time.perf_counter() - train_loop_start
                timing = {
                    "run_started_at": run_started_at,
                    "found_at": found_at,
                    "epoch": epoch,
                    "rollout": total_rollouts,
                    "rl_elapsed_seconds": rl_elapsed,
                    "total_elapsed_seconds": pre_rl_seconds + rl_elapsed,
                    "pre_rl_seconds": pre_rl_seconds,
                }
                best_layout_info = _export_best_layout_artifacts(
                    layout=layout,
                    problem=env.problem,
                    output_dir=top_dir,
                    prefix="layout_best",
                    episode=epoch,
                    reward=reward,
                    success=True,
                    export_metrics=terminal_metrics,
                    timing=timing,
                )
                with open(run_dir / "best_summary.json", "w", encoding="utf-8") as f:
                    json.dump(best_layout_info, f, indent=2, ensure_ascii=False, default=str)
                print("BEST_LAYOUT")
                print(f"  epoch: {epoch}/{num_epochs}, rollout: {total_rollouts}")
                print(f"  reward: {reward:.4f}")
                print(f"  温度: {_metric_text(terminal_metrics.get('rlplanner_temperature'))}")
                print(f"  CPLEX平均线长: {_metric_text(terminal_metrics.get('rlplanner_avg_wirelength'))}")

            if trainer.buffer_size == before and collection_attempts > rollout_batch_size:
                raise RuntimeError("Unable to collect rollout transitions from the environment")

            if training_deadline is not None and time.perf_counter() >= training_deadline:
                time_limit_reached = True
                termination_reason = "time_limit"
                break

        if not epoch_rewards:
            break

        collected_samples = trainer.buffer_size
        total_transitions += collected_samples
        update_metrics = trainer.update()
        executed_epochs = epoch
        partial_epoch = collected_samples < rollout_batch_size
        if not partial_epoch:
            completed_epochs += 1
        mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else float("nan")
        epoch_mean_rewards.append(mean_reward)
        success_rate = epoch_successes / max(len(epoch_rewards), 1)

        if epoch % log_interval == 0 or time_limit_reached:
            now = time.perf_counter()
            progress = {
                "epoch": epoch,
                "num_epochs": num_epochs,
                "seed": seed,
                "partial_epoch": partial_epoch,
                "termination_reason": termination_reason if time_limit_reached else None,
                "rollouts": len(epoch_rewards),
                "total_rollouts": total_rollouts,
                "samples": collected_samples,
                "mean_reward": mean_reward,
                "success_rate": success_rate,
                "total_success_rate": total_successes / max(total_rollouts, 1),
                "elapsed_seconds": now - train_loop_start,
                "interval_seconds": now - last_log_time,
                "metrics": update_metrics,
            }
            last_log_time = now
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  rollouts: {len(epoch_rewards)}, samples: {collected_samples}")
            print(f"  平均奖励: {mean_reward:.4f}, 成功率: {success_rate * 100:.1f}%")
            print(
                f"  optimizer steps: {update_metrics.get('optimizer_steps', 0)}, "
                f"KL: {update_metrics.get('kl_div', float('nan')):.6f}"
            )
            print(f"  累计RL时间: {progress['elapsed_seconds']:.2f}s")
            with open(progress_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, ensure_ascii=False, default=str) + "\n")

        if time_limit_reached:
            break

    if save_checkpoint:
        best_model_path = checkpoint_dir / "ppo_best.pt"
        if best_model_state is None:
            best_model_state = trainer.snapshot_model_state()
        checkpoint_metadata = {
            "seed": seed,
            "best_reward": best_epoch_reward if np.isfinite(best_epoch_reward) else None,
            "best_layout": best_layout_info,
            "termination_reason": termination_reason,
            "completed_epochs": completed_epochs,
            "executed_epochs": executed_epochs,
            "total_rollouts": total_rollouts,
            "total_transitions": total_transitions,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        trainer.save(
            best_model_path,
            model_state_dict=best_model_state,
            include_optimizer=False,
            metadata=checkpoint_metadata,
        )
        if best_layout_info is not None:
            best_layout_info["checkpoint"] = str(best_model_path)
            with open(run_dir / "best_summary.json", "w", encoding="utf-8") as f:
                json.dump(best_layout_info, f, indent=2, ensure_ascii=False, default=str)

        if best_cost_layout_info is not None:
            best_cost_model_path = checkpoint_dir / "ppo_best_cost.pt"
            if best_cost_model_state is None:
                best_cost_model_state = trainer.snapshot_model_state()
            cost_metadata = {
                "seed": seed,
                "best_rlplanner_cost": best_cost if np.isfinite(best_cost) else None,
                "best_cost_reward": best_cost_reward,
                "best_layout": best_cost_layout_info,
                "best_reward": best_epoch_reward if np.isfinite(best_epoch_reward) else None,
                "termination_reason": termination_reason,
                "completed_epochs": completed_epochs,
                "executed_epochs": executed_epochs,
                "total_rollouts": total_rollouts,
                "total_transitions": total_transitions,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            trainer.save(
                best_cost_model_path,
                model_state_dict=best_cost_model_state,
                include_optimizer=False,
                metadata=cost_metadata,
            )
            best_cost_layout_info["checkpoint"] = str(best_cost_model_path)
            with open(run_dir / "best_cost_summary.json", "w", encoding="utf-8") as f:
                json.dump(best_cost_layout_info, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"  请求epochs: {num_epochs}")
    print(f"  完整epochs: {completed_epochs}")
    print(f"  已执行epoch轮次: {executed_epochs}")
    print(f"  总rollouts: {total_rollouts}")
    print(f"  总transitions: {total_transitions}")
    print(f"  最终成功率: {total_successes / max(total_rollouts, 1) * 100:.1f}%")
    if epoch_mean_rewards:
        print(f"  最近平均奖励: {np.mean(epoch_mean_rewards[-100:]):.4f}")
    print(f"  optimizer steps: {trainer.total_optimizer_steps}")
    print(f"  终止原因: {termination_reason}")
    print("=" * 70)

    trainer.best_layout_info = best_layout_info
    trainer.best_cost_layout_info = best_cost_layout_info
    trainer.best_checkpoint_path = best_model_path
    trainer.best_cost_checkpoint_path = best_cost_model_path
    trainer.best_rlplanner_cost = best_cost if np.isfinite(best_cost) else None
    trainer.best_illegal_info = None
    trainer.termination_reason = termination_reason
    trainer.completed_epochs = completed_epochs
    trainer.executed_epochs = executed_epochs
    trainer.total_transitions = total_transitions
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO训练 - 芯片布局优化")
    parser.add_argument("--name", type=str, default=None, help="本次训练名称；输出到 runs/<name>")
    parser.add_argument("--json", dest="json_path", type=str, default=str(RL_DIR / "examples" / "multigpu.json"))
    parser.add_argument("--num_epochs", "--num_episodes", dest="num_epochs", type=int, default=600,
                        help="PPO epochs; --num_episodes is a compatibility alias")
    parser.add_argument("--rollout_batch_size", type=int, default=480,
                        help="transitions collected before each PPO update")
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--ppo_update_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--target_kl", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=None, help="可复现训练的随机种子")
    parser.add_argument(
        "--time_limit_seconds",
        type=float,
        default=None,
        help="单次实验总时间预算，包含热阻表、RL与最终checkpoint保存",
    )
    parser.add_argument("--save_interval", type=int, default=10)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--save_checkpoint",
        dest="save_checkpoint",
        action="store_true",
        help="保存 PPO checkpoint（默认关闭，模型可能很大）",
    )
    checkpoint_group.add_argument(
        "--no_checkpoint",
        dest="save_checkpoint",
        action="store_false",
        help="不保存 PPO checkpoint（默认）",
    )
    parser.set_defaults(save_checkpoint=False)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--grid_resolution", type=str, default="auto", help="数字或 auto")
    parser.add_argument("--max_width", type=float, default=None)
    parser.add_argument("--max_height", type=float, default=None)
    parser.add_argument("--placement_area_weight", type=float, default=1.0, help="w1: chiplet area weight")
    parser.add_argument("--placement_pin_weight", type=float, default=1.0, help="w2: total incident pin count weight")
    parser.add_argument(
        "--thermal_intp_size",
        type=float,
        default=None,
        help="fastTM interposer size in mm; layouts larger than this skip thermal evaluation",
    )
    parser.add_argument("--min_overlap", type=float, default=0.5)
    parser.add_argument("--exact_action_slots", type=int, default=50000, help="off-grid 精确坐标的动态动作槽数量")
    parser.add_argument(
        "--max_auto_grid_resolution",
        type=int,
        default=100,
        help="auto 网格每个轴的最大分辨率（默认 100，动作维度最大约 70000）",
    )
    parser.add_argument("--force_thermal_tables", action="store_true", help="强制重算 fastTM 热阻表")
    parser.add_argument(
        "--footprint_mode",
        choices=FOOTPRINT_MODES,
        default=DEFAULT_FOOTPRINT_MODE,
        help="占据包络来源：json_first=优先用 JSON footprint_w/h，再退回 hubump，最后按连线重算（默认）；"
             "json_only=必须由 JSON 声明，否则报错；recompute=始终按连线重算",
    )
    parser.add_argument(
        "--verify_footprint",
        action="store_true",
        help="声明包络与按连线重算值不一致时直接报错（默认只告警）",
    )
    args = parser.parse_args()

    grid_resolution = args.grid_resolution
    if grid_resolution is not None and str(grid_resolution).lower() != "auto":
        grid_resolution = int(grid_resolution)

    trained_model = train(
        json_path=args.json_path,
        name=args.name,
        num_episodes=args.num_epochs,
        rollout_batch_size=args.rollout_batch_size,
        num_minibatches=args.num_minibatches,
        ppo_update_epochs=args.ppo_update_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        seed=args.seed,
        time_limit_seconds=args.time_limit_seconds,
        save_interval=args.save_interval,
        save_checkpoint=args.save_checkpoint,
        log_interval=args.log_interval,
        generate_thermal_tables=True,
        force_thermal_tables=args.force_thermal_tables,
        grid_resolution=grid_resolution,
        max_width=args.max_width,
        max_height=args.max_height,
        placement_area_weight=args.placement_area_weight,
        placement_pin_weight=args.placement_pin_weight,
        thermal_intp_size=args.thermal_intp_size,
        min_overlap=args.min_overlap,
        exact_action_slots=args.exact_action_slots,
        max_auto_grid_resolution=args.max_auto_grid_resolution,
        footprint_mode=args.footprint_mode,
        verify_footprint=args.verify_footprint,
        placement_reward=1,  # 放置奖励
        adjacency_reward=20,   # 邻接奖励
        compact = 10,
        min_wirelength_reward_scale =0,
        extra_adjacency_reward=5,
        terminal_util_reward_scale=30 ,
        terminal_wirelength_reward_scale=0,
        terminal_rlplanner_cost_scale=1.0,
        lenbase_samples=0,
    )
    
    if args.save_checkpoint:
        print(f"\nModel saved to {trained_model.run_dir / 'checkpoints' / 'ppo_best.pt'}")
    else:
        print("\nCheckpoint saving disabled; layout metrics and logs were retained.")
