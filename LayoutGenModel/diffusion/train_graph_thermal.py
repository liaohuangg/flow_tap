import os
import sys
import time
from pathlib import Path

_DIFFUSION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DIFFUSION_DIR.parent
_FLOW_GCN_ROOT = _REPO_ROOT.parent
for _path in (_REPO_ROOT, _DIFFUSION_DIR, _FLOW_GCN_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import hydra
import torch
import utils
import models
import guidance
import common
from omegaconf import OmegaConf, open_dict

from train_graph import load_checkpoint


def _scheduled_train_weight(weight, start_step, warmup_steps, train_step=None, current_step=None):
    weight = float(weight or 0.0)
    if weight <= 0.0:
        return 0.0
    step = train_step if isinstance(train_step, int) else current_step
    if step is None:
        return weight
    if step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return weight
    progress = min(1.0, float(step - start_step + 1) / float(warmup_steps))
    return weight * progress


def _active_node_mask(cond, mask, batch_size, device):
    if mask is not None:
        active = (~mask.view(-1).bool()).to(device=device)
    elif "is_ports" in cond:
        active = (~cond.is_ports.view(-1).bool()).to(device=device)
    else:
        active = torch.ones(cond.x.shape[0], dtype=torch.bool, device=device)
    return active.view(1, -1).expand(batch_size, -1)


def _masked_mean(value, active):
    active_f = active.to(dtype=value.dtype)
    return (value * active_f).sum() / active_f.sum().clamp_min(1.0)


def _legality_node_risks(x_hat, cond, mask=None):
    B, V, _ = x_hat.shape
    dtype = x_hat.dtype
    device = x_hat.device
    sizes = cond.x[:, :2].to(device=device, dtype=dtype).view(1, V, 2).clamp_min(1e-8)
    active = _active_node_mask(cond, mask, B, device)

    pos_i = x_hat[..., :2].unsqueeze(2)
    pos_j = x_hat[..., :2].unsqueeze(1)
    size_i = sizes.unsqueeze(2)
    size_j = sizes.unsqueeze(1)
    overlap_xy = torch.relu(0.5 * (size_i + size_j) - torch.abs(pos_i - pos_j))
    overlap_area = overlap_xy[..., 0] * overlap_xy[..., 1]

    eye = torch.eye(V, dtype=torch.bool, device=device).view(1, V, V)
    active_pair = active.view(B, V, 1) & active.view(B, 1, V) & (~eye)
    overlap_area = overlap_area.masked_fill(~active_pair, 0.0)
    node_area = (sizes[..., 0] * sizes[..., 1]).clamp_min(1e-8)
    overlap_risk = overlap_area.sum(dim=-1) / node_area

    boundary_xy = torch.relu(torch.abs(x_hat[..., :2]) + sizes / 2.0 - 1.0)
    boundary_risk = boundary_xy.sum(dim=-1)
    overlap_risk = torch.where(active, overlap_risk, torch.zeros_like(overlap_risk))
    boundary_risk = torch.where(active, boundary_risk, torch.zeros_like(boundary_risk))
    return overlap_risk, boundary_risk, active


def _build_thermal_model_from_ckpt(ckpt, device):
    state = ckpt.get("model", {})
    is_hrnet = (
        "stages" in ckpt
        or "blocks_per_stage" in ckpt
        or "expand_ratio" in ckpt
        or any(key.startswith("film64.") or key.startswith("head_fuse.") for key in state)
    )
    if is_hrnet:
        from thermalmodel.HRNet import ThermalGuidanceHRNet

        mean_calib = bool(ckpt.get("mean_calib", ckpt.get("mean_calibration", False)))
        if "disable_mean_calib" in ckpt:
            mean_calib = not bool(ckpt.get("disable_mean_calib"))
        model = ThermalGuidanceHRNet(
            base=int(ckpt.get("base", 32)),
            stages=int(ckpt.get("stages", 4)),
            blocks_per_stage=int(ckpt.get("blocks_per_stage", 2)),
            expand_ratio=int(ckpt.get("expand_ratio", 2)),
            mean_calib=mean_calib,
        ).to(device)
    else:
        from thermalmodel.guidance_model import ThermalGuidanceNet

        model = ThermalGuidanceNet(base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _windows_long_path(path):
    path = str(path)
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    if len(path) > 240 and len(path) >= 3 and path[1:3] == ":\\":
        return "\\\\?\\" + path
    return path


def _load_thermal_checkpoint(path):
    load_path = _windows_long_path(path)
    try:
        if os.path.getsize(load_path) < 1024:
            with open(load_path, "rb") as f:
                head = f.read(128)
            if b"git-lfs.github.com/spec" in head:
                raise RuntimeError(
                    f"Thermal checkpoint is a Git LFS pointer, not the real weights: {path}. "
                    "Run git lfs pull for the thermalmodel repository before using this checkpoint."
                )
    except OSError:
        pass
    return torch.load(load_path, map_location="cpu")


def _thermal_output_to_grid_and_avg(output):
    if isinstance(output, (tuple, list)):
        temp_grid = output[0]
        avg_temp = output[1] if len(output) > 1 else None
    else:
        temp_grid = output
        avg_temp = None
    return temp_grid, avg_temp


class ThermalFlowMatchingModel(models.FlowMatchingModel):
    def __init__(self, *args, thermal_cfg=None, bbox_cfg=None, legality_aux_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.thermal_cfg = dict(thermal_cfg or {})
        self.thermal_train_weight = float(self.thermal_cfg.get("train_weight", 0.0) or 0.0)
        self.thermal_ckpt = self.thermal_cfg.get("ckpt", "none")
        self.thermal_grid_size = int(self.thermal_cfg.get("grid_size", 128))
        self.thermal_rect_sharpness = float(self.thermal_cfg.get("rect_sharpness", 80.0))
        self.thermal_smooth_max_beta = float(self.thermal_cfg.get("smooth_max_beta", 20.0))
        self.thermal_max_weight = float(self.thermal_cfg.get("max_weight", 1.0))
        self.thermal_mean_weight = float(self.thermal_cfg.get("mean_weight", 0.1))
        self.thermal_target_max_k = float(self.thermal_cfg.get("target_max_k", 0.0) or 0.0)
        self.thermal_start_step = int(self.thermal_cfg.get("start_step", 0) or 0)
        self.thermal_warmup_steps = int(self.thermal_cfg.get("warmup_steps", 0) or 0)
        self.bbox_cfg = dict(bbox_cfg or {})
        self.bbox_train_weight = float(self.bbox_cfg.get("train_weight", 0.0) or 0.0)
        self.bbox_softmax_beta = float(self.bbox_cfg.get("softmax_beta", 30.0))
        self.bbox_target_ratio = float(self.bbox_cfg.get("target_ratio", 0.95))
        self.bbox_start_step = int(self.bbox_cfg.get("start_step", 0) or 0)
        self.bbox_warmup_steps = int(self.bbox_cfg.get("warmup_steps", 0) or 0)
        self.legality_aux_cfg = dict(legality_aux_cfg or {})
        self.legality_aux_enabled = bool(self.legality_aux_cfg.get("enabled", False))
        self.legality_aux_start_step = int(self.legality_aux_cfg.get("start_step", 0) or 0)
        self.legality_aux_warmup_steps = int(self.legality_aux_cfg.get("warmup_steps", 0) or 0)
        self.overlap_head_weight = float(self.legality_aux_cfg.get("overlap_head_weight", 0.0) or 0.0)
        self.boundary_head_weight = float(self.legality_aux_cfg.get("boundary_head_weight", 0.0) or 0.0)
        self.overlap_direct_weight = float(self.legality_aux_cfg.get("overlap_direct_weight", 0.0) or 0.0)
        self.boundary_direct_weight = float(self.legality_aux_cfg.get("boundary_direct_weight", 0.0) or 0.0)
        # Keep the frozen thermal surrogate out of this module's state_dict.
        # It is an external loss model, not part of the diffusion checkpoint.
        self.__dict__["_thermal_model"] = None
        self.__dict__["_thermal_stats"] = None
        self.__dict__["_thermal_current_step"] = None

    def set_thermal_step(self, step):
        self.__dict__["_thermal_current_step"] = None if step is None else int(step)

    def _effective_thermal_weight(self, train_step=None):
        return _scheduled_train_weight(
            self.thermal_train_weight,
            self.thermal_start_step,
            self.thermal_warmup_steps,
            train_step=train_step,
            current_step=self.__dict__.get("_thermal_current_step"),
        )

    def _effective_bbox_weight(self, train_step=None):
        return _scheduled_train_weight(
            self.bbox_train_weight,
            self.bbox_start_step,
            self.bbox_warmup_steps,
            train_step=train_step,
            current_step=self.__dict__.get("_thermal_current_step"),
        )

    def _effective_legality_aux_weight(self, base_weight, train_step=None):
        if not self.legality_aux_enabled:
            return 0.0
        return _scheduled_train_weight(
            base_weight,
            self.legality_aux_start_step,
            self.legality_aux_warmup_steps,
            train_step=train_step,
            current_step=self.__dict__.get("_thermal_current_step"),
        )

    def loss(self, x, cond, train_step=None):
        B = x.shape[0]
        t = self._t_dist.sample((B,)).squeeze(dim=-1)
        assert t.shape == (B,), "t has to have shape (B,)"

        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)

        z = self._epsilon_dist.sample(x.shape).squeeze(dim=-1)
        t_view = t.view(B, 1, 1)
        x_t = (1 - t_view) * x + t_view * z
        x_t = torch.where(mask, x, x_t) if mask is not None else x_t

        velocity_target = z - x
        velocity_pred = self(x_t, cond, t)
        flow_loss = self._loss(velocity_pred, velocity_target, mask)
        loss = flow_loss

        thermal_loss = None
        thermal_weight = self._effective_thermal_weight(train_step)
        bbox_loss = None
        bbox_area_ratio = None
        bbox_weight = self._effective_bbox_weight(train_step)
        overlap_head_weight = self._effective_legality_aux_weight(self.overlap_head_weight, train_step)
        boundary_head_weight = self._effective_legality_aux_weight(self.boundary_head_weight, train_step)
        overlap_direct_weight = self._effective_legality_aux_weight(self.overlap_direct_weight, train_step)
        boundary_direct_weight = self._effective_legality_aux_weight(self.boundary_direct_weight, train_step)
        legality_aux_active = (
            overlap_head_weight > 0.0
            or boundary_head_weight > 0.0
            or overlap_direct_weight > 0.0
            or boundary_direct_weight > 0.0
        )
        if (
            (thermal_weight > 0.0 and self.thermal_ckpt not in (None, "", "none"))
            or bbox_weight > 0.0
            or legality_aux_active
        ):
            x_hat = x_t - t_view * velocity_pred
            x_hat = torch.where(mask, x, x_hat) if mask is not None else x_hat
            x_hat = torch.clamp(x_hat, -2.0, 2.0)
        else:
            x_hat = None

        if thermal_weight > 0.0 and self.thermal_ckpt not in (None, "", "none"):
            thermal_loss = self._thermal_potential(x_hat, cond).mean()
            loss = loss + thermal_weight * thermal_loss
        if bbox_weight > 0.0:
            pred_bbox_area = guidance.bbox_area_guidance_potential(
                x_hat,
                cond,
                mask=mask,
                softmax_beta=self.bbox_softmax_beta,
            )
            with torch.no_grad():
                ref_bbox_area = guidance.bbox_extents(x, cond, mask=mask)[2].detach().clamp_min(1e-12)
            bbox_area_ratio = pred_bbox_area / ref_bbox_area
            bbox_loss = torch.relu(bbox_area_ratio - self.bbox_target_ratio).mean()
            loss = loss + bbox_weight * bbox_loss

        overlap_risk = None
        boundary_risk = None
        overlap_direct_loss = None
        boundary_direct_loss = None
        overlap_head_loss = None
        boundary_head_loss = None
        if legality_aux_active:
            overlap_risk, boundary_risk, active_nodes = _legality_node_risks(x_hat, cond, mask=mask)
            if overlap_direct_weight > 0.0:
                overlap_direct_loss = _masked_mean(overlap_risk, active_nodes)
                loss = loss + overlap_direct_weight * overlap_direct_loss
            if boundary_direct_weight > 0.0:
                boundary_direct_loss = _masked_mean(boundary_risk, active_nodes)
                loss = loss + boundary_direct_weight * boundary_direct_loss

            aux_outputs = getattr(self._reverse_model, "last_aux_outputs", None)
            if aux_outputs is not None:
                if overlap_head_weight > 0.0 and "overlap" in aux_outputs:
                    pred_overlap = aux_outputs["overlap"]
                    overlap_head_loss = _masked_mean(
                        torch.nn.functional.smooth_l1_loss(
                            pred_overlap,
                            overlap_risk.detach(),
                            reduction="none",
                        ),
                        active_nodes,
                    )
                    loss = loss + overlap_head_weight * overlap_head_loss
                if boundary_head_weight > 0.0 and "boundary" in aux_outputs:
                    pred_boundary = aux_outputs["boundary"]
                    boundary_head_loss = _masked_mean(
                        torch.nn.functional.smooth_l1_loss(
                            pred_boundary,
                            boundary_risk.detach(),
                            reduction="none",
                        ),
                        active_nodes,
                    )
                    loss = loss + boundary_head_weight * boundary_head_loss

        pred_masked = velocity_pred.detach()[torch.logical_not(mask).expand(x.shape)] if mask is not None else velocity_pred.detach()
        metrics = {
            "flow_loss": flow_loss.detach().cpu().item(),
            "thermal_weight": thermal_weight,
            "bbox_weight": bbox_weight,
            "legality_overlap_head_weight": overlap_head_weight,
            "legality_boundary_head_weight": boundary_head_weight,
            "legality_overlap_direct_weight": overlap_direct_weight,
            "legality_boundary_direct_weight": boundary_direct_weight,
            "bbox_target_ratio": self.bbox_target_ratio,
            "velocity_theta_mean": pred_masked.mean().cpu().numpy(),
            "velocity_theta_std": pred_masked.std().cpu().numpy(),
        }
        if thermal_loss is not None:
            metrics["thermal_train_loss"] = thermal_loss.detach().cpu().item()
            metrics["thermal_weighted_loss"] = (thermal_weight * thermal_loss.detach()).cpu().item()
        if bbox_loss is not None:
            metrics["bbox_train_loss"] = bbox_loss.detach().cpu().item()
            metrics["bbox_hinge_loss"] = bbox_loss.detach().cpu().item()
            metrics["bbox_area_ratio"] = bbox_area_ratio.detach().mean().cpu().item()
            metrics["bbox_weighted_loss"] = (bbox_weight * bbox_loss.detach()).cpu().item()
        if overlap_risk is not None:
            metrics["legality_overlap_risk"] = _masked_mean(overlap_risk.detach(), active_nodes).cpu().item()
            metrics["legality_boundary_risk"] = _masked_mean(boundary_risk.detach(), active_nodes).cpu().item()
        if overlap_direct_loss is not None:
            metrics["legality_overlap_direct_loss"] = overlap_direct_loss.detach().cpu().item()
            metrics["legality_overlap_direct_weighted_loss"] = (
                overlap_direct_weight * overlap_direct_loss.detach()
            ).cpu().item()
        if boundary_direct_loss is not None:
            metrics["legality_boundary_direct_loss"] = boundary_direct_loss.detach().cpu().item()
            metrics["legality_boundary_direct_weighted_loss"] = (
                boundary_direct_weight * boundary_direct_loss.detach()
            ).cpu().item()
        if overlap_head_loss is not None:
            metrics["legality_overlap_head_loss"] = overlap_head_loss.detach().cpu().item()
            metrics["legality_overlap_head_weighted_loss"] = (
                overlap_head_weight * overlap_head_loss.detach()
            ).cpu().item()
        if boundary_head_loss is not None:
            metrics["legality_boundary_head_loss"] = boundary_head_loss.detach().cpu().item()
            metrics["legality_boundary_head_weighted_loss"] = (
                boundary_head_weight * boundary_head_loss.detach()
            ).cpu().item()
        return loss, metrics

    def _thermal_potential(self, x_hat, cond):
        model, stats = self._load_thermal_model(x_hat.device)
        power_grid, layout_grid, total_power = _thermal_rasterize(
            x_hat,
            cond,
            grid_size=self.thermal_grid_size,
            rect_sharpness=self.thermal_rect_sharpness,
            stats=stats,
        )
        temp, avg_temp = _thermal_output_to_grid_and_avg(model(power_grid, layout_grid, total_power))
        flat = temp.flatten(1)
        smooth_max = torch.logsumexp(flat * self.thermal_smooth_max_beta, dim=1) / self.thermal_smooth_max_beta
        mean_temp = avg_temp.view(-1) if avg_temp is not None else flat.mean(dim=1)
        if self.thermal_target_max_k > 0.0 and stats is not None:
            max_k = _denorm_temp_k(smooth_max, stats)
            return torch.relu(max_k - self.thermal_target_max_k).square()
        return self.thermal_max_weight * smooth_max + self.thermal_mean_weight * mean_temp

    def _load_thermal_model(self, device):
        if self.__dict__.get("_thermal_model") is not None:
            return self.__dict__["_thermal_model"], self.__dict__["_thermal_stats"]
        ckpt = _load_thermal_checkpoint(self.thermal_ckpt)
        model = _build_thermal_model_from_ckpt(ckpt, device)
        self.__dict__["_thermal_model"] = model
        self.__dict__["_thermal_stats"] = ckpt.get("stats") if isinstance(ckpt.get("stats"), dict) else None
        return self.__dict__["_thermal_model"], self.__dict__["_thermal_stats"]


def _thermal_chip_size(cond, device, dtype):
    if "chip_size" not in cond:
        return torch.ones((2,), dtype=dtype, device=device)
    chip_size = cond.chip_size
    chip_size = torch.as_tensor(chip_size, dtype=dtype, device=device).view(-1)
    if chip_size.numel() == 4:
        chip_size = chip_size[2:] - chip_size[:2]
    else:
        chip_size = chip_size[:2]
    return chip_size.clamp_min(1e-12)


def _thermal_rasterize(x_hat, cond, grid_size, rect_sharpness, stats=None):
    B, V, _ = x_hat.shape
    dtype = x_hat.dtype
    device = x_hat.device
    chip_size = _thermal_chip_size(cond, device=device, dtype=dtype).view(1, 1, 2)
    pos_phys = ((x_hat[..., :2] + 1.0) / 2.0) * chip_size
    size_phys = (cond.x[:, :2].to(device=device, dtype=dtype).view(1, V, 2) / 2.0) * chip_size

    if "is_macros" in cond:
        active = cond.is_macros.to(device=device).bool()
    elif "is_ports" in cond:
        active = ~cond.is_ports.to(device=device).bool()
    else:
        active = torch.ones((V,), dtype=torch.bool, device=device)
    if not bool(active.any()):
        active = torch.ones((V,), dtype=torch.bool, device=device)

    pos_phys = pos_phys[:, active, :]
    size_phys = size_phys[:, active, :].clamp_min(1e-12)
    if "node_power" in cond:
        raw_powers = cond.node_power.to(device=device, dtype=dtype)[active].abs()
    else:
        raw_powers = torch.ones((size_phys.shape[1],), dtype=dtype, device=device)
    if stats is not None and "power_min" in stats and "power_max" in stats:
        pmin = float(stats["power_min"])
        pmax = float(stats["power_max"])

    left_phys = pos_phys[..., 0] - size_phys[..., 0] / 2.0
    right_phys = pos_phys[..., 0] + size_phys[..., 0] / 2.0
    bottom_phys = pos_phys[..., 1] - size_phys[..., 1] / 2.0
    top_phys = pos_phys[..., 1] + size_phys[..., 1] / 2.0
    bbox_left = left_phys.min(dim=1).values
    bbox_right = right_phys.max(dim=1).values
    bbox_bottom = bottom_phys.min(dim=1).values
    bbox_top = top_phys.max(dim=1).values
    bbox_width = (bbox_right - bbox_left).clamp_min(1e-12)
    bbox_height = (bbox_top - bbox_bottom).clamp_min(1e-12)
    bbox_origin = torch.stack([bbox_left, bbox_bottom], dim=-1).view(B, 1, 2)
    bbox_size = torch.stack([bbox_width, bbox_height], dim=-1).view(B, 1, 2).clamp_min(1e-12)
    pos01 = (pos_phys - bbox_origin) / bbox_size
    size01 = (size_phys / bbox_size).clamp_min(1e-4)

    total_power_density = raw_powers.sum().view(1).expand(B) / (bbox_width * bbox_height).clamp_min(1e-12)
    if stats is not None and "total_power_min" in stats and "total_power_max" in stats:
        tp_min = float(stats["total_power_min"])
        tp_max = float(stats["total_power_max"])
        total_power = ((total_power_density - tp_min) / max(tp_max - tp_min, 1e-6)).clamp(0.0, 1.0)
    else:
        total_power = total_power_density / total_power_density.detach().abs().max().clamp_min(1e-6)
    total_power = total_power.view(B, 1)

    coords = torch.linspace(0.0, 1.0, grid_size, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    xx = xx.view(1, 1, grid_size, grid_size)
    yy = yy.view(1, 1, grid_size, grid_size)

    size = size01
    left = (pos01[..., 0] - size[..., 0] / 2.0).view(B, -1, 1, 1)
    right = (pos01[..., 0] + size[..., 0] / 2.0).view(B, -1, 1, 1)
    bottom = (pos01[..., 1] - size[..., 1] / 2.0).view(B, -1, 1, 1)
    top = (pos01[..., 1] + size[..., 1] / 2.0).view(B, -1, 1, 1)

    sx = torch.sigmoid(rect_sharpness * (xx - left)) * torch.sigmoid(rect_sharpness * (right - xx))
    sy = torch.sigmoid(rect_sharpness * (yy - bottom)) * torch.sigmoid(rect_sharpness * (top - yy))
    rect = sx * sy
    layout = rect.sum(dim=1, keepdim=True).clamp(0.0, 1.0)

    # Match thermalmodel.dataLoader/gen_powercsv.py: each chiplet's total power
    # is distributed over the grid cells it covers, not copied into every cell.
    rect_mass = rect.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    power_raw = (rect * (raw_powers.view(1, -1, 1, 1) / rect_mass)).sum(dim=1, keepdim=True)
    if stats is not None and "power_min" in stats and "power_max" in stats:
        power = ((power_raw - pmin) / max(pmax - pmin, 1e-6)).clamp(0.0, 1.0)
    else:
        power = power_raw / power_raw.detach().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return power, layout, total_power


def _denorm_temp_k(x01, stats):
    temp_min = float(stats["temp_min"])
    temp_max = float(stats["temp_max"])
    temp = x01 * (temp_max - temp_min) + temp_min

    unit = str(stats.get("temp_unit", stats.get("unit", ""))).lower()
    is_celsius = unit in {"c", "celsius", "degc", "degree_c", "degrees_c"} or (
        unit == "" and temp_max < 200.0
    )
    return temp + 273.15 if is_celsius else temp


@hydra.main(version_base=None, config_path="configs", config_name="config_graph_fm")
def main(cfg):
    OmegaConf.set_struct(cfg, True)
    if cfg.get("wandb") is not None:
        with open_dict(cfg):
            cfg.logger.wandb = cfg.wandb

    thermal_cfg = dict(cfg.get("thermal", {}) or {})
    bbox_cfg = dict(cfg.get("bbox", {}) or {})
    legality_aux_cfg = dict(cfg.get("legality_aux", {}) or {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = utils.model_log_dir(cfg)
    sample_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"saving checkpoints to: {log_dir}")
    torch.manual_seed(cfg.seed)

    train_set, val_set = utils.load_graph_data(
        cfg.task,
        augment=cfg.augment,
        train_data_limit=cfg.train_data_limit,
        val_data_limit=cfg.val_data_limit,
    )
    sample_shape = train_set[0][0].shape
    dataloader = utils.GraphDataLoader(train_set, val_set, cfg.batch_size, cfg.val_batch_size, device)
    with open_dict(cfg):
        cfg.model.update({
            "num_classes": cfg.num_classes,
            "input_shape": tuple(sample_shape),
            "device": device,
        })
        if legality_aux_cfg.get("enabled", False):
            cfg.model.backbone_params.auxiliary_legality_heads_enabled = True

    if cfg.family != "flow_matching":
        raise NotImplementedError("train_graph_thermal.py only supports flow_matching")

    model = ThermalFlowMatchingModel(
        **cfg.model,
        thermal_cfg=thermal_cfg,
        bbox_cfg=bbox_cfg,
        legality_aux_cfg=legality_aux_cfg,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    train_metrics = common.Metrics()

    num_params = sum(param.numel() for param in model.parameters())
    with open_dict(cfg):
        cfg.update({
            "num_params": num_params,
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
        common.logger.JSONLOutput(log_dir, pattern=cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    step = common.Counter()
    logger = common.Logger(step, outputs)
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))
    summary_metrics = {}
    utils.write_summary_metrics(
        summary_metrics,
        os.path.join(log_dir, "metrics_summary"),
        extra_metrics={"last/step": int(step)},
    )

    print(OmegaConf.to_yaml(cfg))
    print(f"model has {num_params} params")
    load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler)

    print(f"==== Start Thermal Training on Device: {device} ====")
    model.train()
    grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0) or 0.0)
    t_0 = time.time()
    t_1 = time.time()
    best_loss = 1e12
    while step < cfg.train_steps:
        x, cond = dataloader.get_batch("train")
        model.set_thermal_step(int(step))
        optim.zero_grad()
        loss, model_metrics = model.loss(x, cond, int(step))
        grad_scaler.scale(loss).backward()
        if grad_clip_norm > 0.0:
            grad_scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            model_metrics["grad_norm"] = float(grad_norm.detach().cpu())
        grad_scaler.step(optim)
        grad_scaler.update()

        train_metrics.add({"loss": loss.detach().cpu().item()})
        train_metrics.add(model_metrics)
        step.increment()

        if int(step) % cfg.print_every == 0:
            t_2 = time.time()
            model.set_thermal_step(int(step))
            validate_graphs = int(cfg.get("validate_graphs", 1) or 1)
            train_logs = utils.validate_graph_batches(dataloader, model, "train", validate_graphs)
            val_logs = utils.validate_graph_batches(dataloader, model, "val", validate_graphs)
            interval_logs = {
                "time_elapsed": t_2 - t_0,
                "ms_per_step": 1000 * (t_2 - t_1) / cfg.print_every,
            }
            train_metric_logs = train_metrics.result()
            logger.add(interval_logs)
            logger.add(train_metric_logs)
            logger.add(val_logs, prefix="val")
            logger.add(train_logs, prefix="train")
            utils.add_summary_metrics(summary_metrics, interval_logs)
            utils.add_summary_metrics(summary_metrics, train_metric_logs, prefix="train_window")
            utils.add_summary_metrics(summary_metrics, val_logs, prefix="val")
            utils.add_summary_metrics(summary_metrics, train_logs, prefix="train")
            utils.write_summary_metrics(
                summary_metrics,
                os.path.join(log_dir, "metrics_summary"),
                extra_metrics={"last/step": int(step), "last/best_loss": best_loss},
            )
            logger.write()
            t_1 = t_2

            checkpointer.save()
            if val_logs["loss"] < best_loss:
                best_loss = val_logs["loss"]
                checkpointer.save(os.path.join(log_dir, "best.ckpt"))
                print("saving best model")
            utils.write_summary_metrics(
                summary_metrics,
                os.path.join(log_dir, "metrics_summary"),
                extra_metrics={"last/step": int(step), "last/best_loss": best_loss},
            )

        if cfg.eval_every > 0 and int(step) % cfg.eval_every == 0:
            print(f"saving model at step {int(step)}")
            checkpointer.save(os.path.join(log_dir, f"step_{int(step)}.ckpt"))
            print("generating evaluation report")
            t3 = time.time()
            utils.generate_report(cfg.eval_samples, dataloader, model, logger, policy=cfg.eval_policy)
            logger.write()
            t4 = time.time()
            print(f"generated report in {t4 - t3:.3f} sec")

        cond.to(device="cpu")


if __name__ == "__main__":
    main()
