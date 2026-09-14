import os
import re
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
from wirelength_surrogate import WirelengthSurrogate
from training_monitor import update_training_monitor

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


def _legality_sizes(cond, V, dtype, device, use_footprint=False):
    """Body sizes by default; hubump-EXPANDED (footprint) sizes when enabled.

    Training and in-sampler guidance historically used body-only sizes while the
    reported `legal` flag and the legalizer use `expanded_legality_2`
    (footprint).  Measured on the 20-seed pool: 8.3% of body-legal candidates are
    footprint-illegal (cpu-dram: 8 of 14).  The expansion matches
    eval_thermal_guided._prepare_tap_expanded_cond: normalised sizes are
    2*size/side, so adding 4*hubump/side gives the footprint.
    """
    sizes = cond.x[:, :2].to(device=device, dtype=dtype).view(1, V, 2).clamp_min(1e-8)
    if not use_footprint or "tap_hubump" not in cond:
        return sizes
    hub = cond.tap_hubump.to(device=device, dtype=dtype).view(-1)
    if "chip_size" in cond:
        cs = torch.as_tensor(cond.chip_size, dtype=dtype, device=device).view(-1)
        side = (cs[2:] - cs[:2]) if cs.numel() == 4 else cs[:2]
        side = side.clamp_min(1e-8)
    else:
        side = torch.ones(2, dtype=dtype, device=device)
    return sizes + (4.0 * hub).view(1, -1, 1) / side.view(1, 1, -1)


def _legality_node_risks(x_hat, cond, mask=None, use_footprint=False):
    B, V, _ = x_hat.shape
    dtype = x_hat.dtype
    device = x_hat.device
    sizes = _legality_sizes(cond, V, dtype, device, use_footprint)
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


def _is_gnn_hrnet_checkpoint(ckpt):
    state = ckpt.get("model", {}) if isinstance(ckpt, dict) else {}
    return (
        any(key.startswith("encoder.") for key in state)
        and any(key.startswith("field_head.") for key in state)
    )


def _indexed_module_count(state, prefix):
    indices = set()
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    for key in state:
        match = pattern.match(key)
        if match:
            indices.add(int(match.group(1)))
    return max(indices) + 1 if indices else 0


def _gnn_hrnet_config_from_state(state, model_cfg=None):
    model_cfg = dict(model_cfg or {})
    in_weight = state["encoder.in_proj.weight"]
    stem_weight = state["field_head.stem.0.conv.weight"]
    attention = state.get("encoder.layers.0.gat.att")
    expand_weight = state.get("field_head.stem.1.net.0.conv.weight")
    hidden = int(in_weight.shape[0])
    base = int(stem_weight.shape[0])
    inferred = {
        "node_dim": int(in_weight.shape[1]),
        "hidden": hidden,
        "heads": int(attention.shape[1]) if attention is not None else 4,
        "num_layers": _indexed_module_count(state, "encoder.layers") or 3,
        "edge_dim": 1,
        "grid": int(model_cfg.get("grid_size", model_cfg.get("grid", 64))),
        "base": base,
        "stages": _indexed_module_count(state, "field_head.stages") or 4,
        "blocks_per_stage": _indexed_module_count(state, "field_head.stages.0.b64.blocks") or 2,
        "expand_ratio": (
            int(expand_weight.shape[0] // base)
            if expand_weight is not None and base > 0
            else 2
        ),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }
    for key in tuple(inferred):
        if key in model_cfg and key not in {"grid_size"}:
            inferred[key] = model_cfg[key]
    return inferred


def _build_thermal_model_from_ckpt(ckpt, device, model_cfg=None):
    state = ckpt.get("model", {})
    if _is_gnn_hrnet_checkpoint(ckpt):
        from thermalmodel.gnnhrnet import GNNHRNetModel

        config = _gnn_hrnet_config_from_state(state, model_cfg)
        model = GNNHRNetModel(**config).to(device)
        model.load_state_dict(state, strict=True)
        model._flow_tap_thermal_kind = "gnn_hrnet"
        model._flow_tap_thermal_config = config
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

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


def _thermal_stats_from_checkpoint(ckpt):
    stats = ckpt.get("stats") if isinstance(ckpt, dict) else None
    if isinstance(stats, dict):
        return dict(stats)
    if _is_gnn_hrnet_checkpoint(ckpt):
        return {
            "temp_min": 45.22,
            "temp_max": 276.12,
            "temp_unit": "celsius",
        }
    return None


def _windows_long_path(path):
    path = str(path)
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    if len(path) > 240 and len(path) >= 3 and path[1:3] == ":\\":
        return "\\\\?\\" + path
    return path


def _load_thermal_checkpoint(path):
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not path.is_absolute():
        candidates = (
            Path.cwd() / path,
            _DIFFUSION_DIR / path,
            _REPO_ROOT / path,
            _FLOW_GCN_ROOT / path,
        )
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    load_path = _windows_long_path(path.resolve())
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
    def __init__(self, *args, thermal_cfg=None, wirelength_cfg=None, bbox_cfg=None,
                 legality_aux_cfg=None, flow_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Stage-2 fine-tuning knob.  Default 1.0 reproduces the historical
        # behaviour exactly (loss = flow_loss), so existing runs/configs are
        # unaffected unless "+flow.train_weight=..." is passed explicitly.
        self.flow_cfg = dict(flow_cfg or {})
        self.flow_train_weight = float(self.flow_cfg.get("train_weight", 1.0))
        # Aux-loss t reweighting (opt-in; default reproduces historical behaviour)
        self.aux_t_reweight = bool(flow_cfg.get("aux_t_reweight", False)) if flow_cfg else False
        self.aux_t_eps = float(self.flow_cfg.get("aux_t_eps", 0.05) or 0.05)
        self.aux_t_max = float(self.flow_cfg.get("aux_t_max", 1.0) or 1.0)
        # 0 = disabled.  When >0, every N steps also measures per-term
        # parameter-gradient norms / shares / pairwise cosines.
        self.grad_diag_every = int(self.flow_cfg.get("grad_diag_every", 0) or 0)
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
        self.wirelength_cfg = dict(wirelength_cfg or {})
        self.wirelength_train_weight = float(self.wirelength_cfg.get("train_weight", 0.0) or 0.0)
        self.wirelength_start_step = int(self.wirelength_cfg.get("start_step", 0) or 0)
        self.wirelength_warmup_steps = int(self.wirelength_cfg.get("warmup_steps", 0) or 0)
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
        # Use hubump-expanded (footprint) sizes in the legality penalties, matching
        # the reported `expanded_legality_2` metric.  Default False = historical
        # body-only behaviour.
        self.legality_use_footprint = bool(self.legality_aux_cfg.get("use_footprint", False))
        # Keep the frozen thermal surrogate out of this module's state_dict.
        # It is an external loss model, not part of the diffusion checkpoint.
        self.__dict__["_thermal_model"] = None
        self.__dict__["_thermal_stats"] = None
        self.__dict__["_wirelength_surrogate"] = None
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

    def _effective_wirelength_weight(self, train_step=None):
        return _scheduled_train_weight(
            self.wirelength_train_weight,
            self.wirelength_start_step,
            self.wirelength_warmup_steps,
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
        loss = self.flow_train_weight * flow_loss

        # ---- auxiliary-loss t reweighting (default OFF: reproduces history) ----
        # d L_aux / d v = -t * d L_aux / d x_hat, so with t ~ U(1e-4,1) the gradient
        # reaching v is ~43x stronger at t~1 (where x_hat is noise) than at t~0
        # (where x_hat is the data).  Measured by overnight/t_buckets.py.
        # Weighting each sample by 1/clamp(t) and renormalising to mean 1 makes
        # w_i * t_i constant -> every t contributes equally to the gradient on v.
        t_w = None
        if self.aux_t_reweight:
            inv = 1.0 / t.clamp_min(self.aux_t_eps)
            if self.aux_t_max < 1.0:
                inv = torch.where(t <= self.aux_t_max, inv,
                                  torch.zeros_like(inv))
            t_w = inv / inv.mean().clamp_min(1e-12)

        def _reduce_aux(per_sample):
            return (per_sample * t_w).mean() if t_w is not None else per_sample.mean()

        thermal_loss = None
        thermal_weight = self._effective_thermal_weight(train_step)
        wirelength_loss = None
        wirelength_weight = self._effective_wirelength_weight(train_step)
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
            or wirelength_weight > 0.0
            or bbox_weight > 0.0
            or legality_aux_active
        ):
            x_hat = x_t - t_view * velocity_pred
            x_hat = torch.where(mask, x, x_hat) if mask is not None else x_hat
            x_hat = torch.clamp(x_hat, -2.0, 2.0)
        else:
            x_hat = None

        if thermal_weight > 0.0 and self.thermal_ckpt not in (None, "", "none"):
            thermal_loss = _reduce_aux(self._thermal_potential(x_hat, cond))
            loss = loss + thermal_weight * thermal_loss
        if wirelength_weight > 0.0:
            wirelength_loss = _reduce_aux(self._wirelength_potential(x_hat, cond))
            loss = loss + wirelength_weight * wirelength_loss
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
            bbox_loss = torch.relu(bbox_area_ratio - self.bbox_target_ratio)
            bbox_loss = _reduce_aux(bbox_loss)
            loss = loss + bbox_weight * bbox_loss

        overlap_risk = None
        boundary_risk = None
        overlap_direct_loss = None
        boundary_direct_loss = None
        overlap_head_loss = None
        boundary_head_loss = None
        if legality_aux_active:
            overlap_risk, boundary_risk, active_nodes = _legality_node_risks(
                x_hat, cond, mask=mask, use_footprint=self.legality_use_footprint)
            # The auxiliary HEADS regress onto the physical risk, so keep an
            # unweighted copy for their targets; only the direct penalties get the
            # t reweighting (which has mean 1, so the scale is unchanged).
            overlap_risk_raw, boundary_risk_raw = overlap_risk, boundary_risk
            if t_w is not None:
                overlap_risk = overlap_risk * t_w.view(-1, 1)
                boundary_risk = boundary_risk * t_w.view(-1, 1)
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
                            overlap_risk_raw.detach(),
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
                            boundary_risk_raw.detach(),
                            reduction="none",
                        ),
                        active_nodes,
                    )
                    loss = loss + boundary_head_weight * boundary_head_loss

        # ---- optional gradient-balance diagnostic (default OFF) --------------
        # Weights set from loss VALUES do not tell you the terms' actual influence:
        # what updates theta is ||grad_theta L_term||.  On diagnostic steps this
        # measures each term's parameter-gradient norm and the pairwise cosines
        # (a negative cosine means two terms fight each other on shared params).
        # Uses retain_graph so the outer loss.backward() still works.
        grad_diag = {}
        if self.grad_diag_every > 0 and isinstance(train_step, int) \
                and train_step % self.grad_diag_every == 0:
            terms = {"flow": self.flow_train_weight * flow_loss}
            if thermal_loss is not None and thermal_weight > 0.0:
                terms["thermal"] = thermal_weight * thermal_loss
            if wirelength_loss is not None and wirelength_weight > 0.0:
                terms["wirelength"] = wirelength_weight * wirelength_loss
            if overlap_direct_loss is not None and overlap_direct_weight > 0.0:
                terms["legality"] = overlap_direct_weight * overlap_direct_loss
            params = [p for p in self.parameters() if p.requires_grad]
            gs = {}
            for nm, term in terms.items():
                try:
                    grads = torch.autograd.grad(term, params, retain_graph=True,
                                                allow_unused=True)
                except RuntimeError:
                    continue
                flat = [g.reshape(-1) for g in grads if g is not None]
                if flat:
                    gs[nm] = torch.cat(flat).detach()
            tot = sum(g.square().sum() for g in gs.values()).sqrt() if gs else None
            for nm, g in gs.items():
                grad_diag[f"gradnorm/{nm}"] = float(g.norm())
                if tot is not None and tot > 0:
                    grad_diag[f"gradshare/{nm}"] = float(g.norm() / tot)
            nms = list(gs)
            for i in range(len(nms)):
                for j in range(i + 1, len(nms)):
                    a, b = gs[nms[i]], gs[nms[j]]
                    den = (a.norm() * b.norm()).clamp_min(1e-12)
                    grad_diag[f"gradcos/{nms[i]}_{nms[j]}"] = float((a * b).sum() / den)

        pred_masked = velocity_pred.detach()[torch.logical_not(mask).expand(x.shape)] if mask is not None else velocity_pred.detach()
        metrics = {
            "flow_loss": flow_loss.detach().cpu().item(),
            "flow_train_weight": self.flow_train_weight,
            "flow_weighted_loss": (self.flow_train_weight * flow_loss.detach()).cpu().item(),
            "thermal_weight": thermal_weight,
            "wirelength_weight": wirelength_weight,
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
        if wirelength_loss is not None:
            metrics["wirelength_train_loss"] = wirelength_loss.detach().cpu().item()
            metrics["wirelength_weighted_loss"] = (wirelength_weight * wirelength_loss.detach()).cpu().item()
        if bbox_loss is not None:
            metrics["bbox_train_loss"] = bbox_loss.detach().cpu().item()
            metrics["bbox_hinge_loss"] = bbox_loss.detach().cpu().item()
            metrics["bbox_area_ratio"] = bbox_area_ratio.detach().mean().cpu().item()
            metrics["bbox_weighted_loss"] = (bbox_weight * bbox_loss.detach()).cpu().item()
        if overlap_risk is not None:
            metrics["legality_overlap_risk"] = _masked_mean(overlap_risk_raw.detach(), active_nodes).cpu().item()
            metrics["legality_boundary_risk"] = _masked_mean(boundary_risk_raw.detach(), active_nodes).cpu().item()
            if t_w is not None:
                metrics["aux_t_weight_mean"] = float(t_w.mean())
                metrics["aux_t_weight_max"] = float(t_w.max())
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
        metrics.update(grad_diag)
        return loss, metrics

    def _thermal_potential(self, x_hat, cond):
        model, stats = self._load_thermal_model(x_hat.device)
        output = _thermal_forward(
            model,
            x_hat,
            cond,
            grid_size=self.thermal_grid_size,
            rect_sharpness=self.thermal_rect_sharpness,
            stats=stats,
        )
        temp, avg_temp = _thermal_output_to_grid_and_avg(output)
        flat = temp.flatten(1)
        smooth_max = torch.logsumexp(flat * self.thermal_smooth_max_beta, dim=1) / self.thermal_smooth_max_beta
        mean_temp = avg_temp.view(-1) if avg_temp is not None else flat.mean(dim=1)
        if self.thermal_target_max_k > 0.0 and stats is not None:
            max_k = _denorm_temp_k(smooth_max, stats)
            return torch.relu(max_k - self.thermal_target_max_k).square()
        return self.thermal_max_weight * smooth_max + self.thermal_mean_weight * mean_temp

    def _wirelength_potential(self, x_hat, cond):
        surrogate = self.__dict__.get("_wirelength_surrogate")
        if surrogate is None:
            surrogate = WirelengthSurrogate(self.wirelength_cfg, x_hat.device)
            self.__dict__["_wirelength_surrogate"] = surrogate
        return surrogate.potential(x_hat, cond)

    def _load_thermal_model(self, device):
        if self.__dict__.get("_thermal_model") is not None:
            return self.__dict__["_thermal_model"], self.__dict__["_thermal_stats"]
        ckpt = _load_thermal_checkpoint(self.thermal_ckpt)
        model = _build_thermal_model_from_ckpt(ckpt, device, self.thermal_cfg)
        self.__dict__["_thermal_model"] = model
        self.__dict__["_thermal_stats"] = _thermal_stats_from_checkpoint(ckpt)
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


def _thermal_canvas(cond, device, dtype):
    if "chip_size" not in cond:
        return (
            torch.ones((2,), dtype=dtype, device=device),
            torch.zeros((2,), dtype=dtype, device=device),
        )
    value = torch.as_tensor(cond.chip_size, dtype=dtype, device=device).view(-1)
    if value.numel() == 4:
        return (value[2:] - value[:2]).clamp_min(1e-12), value[:2]
    return value[:2].clamp_min(1e-12), torch.zeros((2,), dtype=dtype, device=device)


def _thermal_active_nodes(cond, device):
    if "is_macros" in cond:
        active = cond.is_macros.to(device=device).view(-1).bool()
    elif "is_ports" in cond:
        active = ~cond.is_ports.to(device=device).view(-1).bool()
    else:
        active = torch.ones(cond.x.shape[0], dtype=torch.bool, device=device)
    if not bool(active.any()):
        active = torch.ones_like(active)
    return active


def _thermal_hubump_widths(cond, sizes, active, device, dtype):
    if "tap_hubump" in cond:
        return cond.tap_hubump.to(device=device, dtype=dtype).view(-1)[active]

    node_count = cond.x.shape[0]
    demand = torch.zeros(node_count, dtype=dtype, device=device)
    if "edge_index" in cond and "edge_weight" in cond:
        edge_index = cond.edge_index.to(device=device, dtype=torch.long)
        weights = cond.edge_weight.to(device=device, dtype=dtype).view(-1)
        demand.index_add_(0, edge_index[0], weights)

    values = []
    # TAP's hubump sizing counts both communication directions for each incident link.
    directed_demand = 2.0 * demand[active]
    for (width, height), required in zip(sizes.detach().cpu(), directed_demand.detach().cpu()):
        rows = 1
        ring = 0.045
        while ((float(width) + float(height)) * 2.0 * ring + 4.0 * ring * ring) / (0.045 ** 2) < float(required):
            rows += 1
            ring = 0.045 * rows
            if rows > 1000:
                raise RuntimeError("infeasible microbump demand while building thermal features")
        values.append(ring)
    return torch.tensor(values, dtype=dtype, device=device)


def _thermal_rect_field(left, bottom, right, top, grid_size, sharpness, differentiable):
    dtype, device = left.dtype, left.device
    if differentiable:
        coords = (torch.arange(grid_size, dtype=dtype, device=device) + 0.5) / float(grid_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        xx = xx.view(1, 1, grid_size, grid_size)
        yy = yy.view(1, 1, grid_size, grid_size)
        left = left.unsqueeze(-1).unsqueeze(-1)
        right = right.unsqueeze(-1).unsqueeze(-1)
        bottom = bottom.unsqueeze(-1).unsqueeze(-1)
        top = top.unsqueeze(-1).unsqueeze(-1)
        sx = torch.sigmoid(sharpness * (xx - left)) * torch.sigmoid(sharpness * (right - xx))
        sy = torch.sigmoid(sharpness * (yy - bottom)) * torch.sigmoid(sharpness * (top - yy))
        return sx * sy

    cell_left = torch.arange(grid_size, dtype=dtype, device=device) / float(grid_size)
    cell_right = (torch.arange(grid_size, dtype=dtype, device=device) + 1.0) / float(grid_size)
    x0 = cell_left.view(1, 1, 1, grid_size)
    x1 = cell_right.view(1, 1, 1, grid_size)
    y0 = cell_left.view(1, 1, grid_size, 1)
    y1 = cell_right.view(1, 1, grid_size, 1)
    return (
        (x1 > left.unsqueeze(-1).unsqueeze(-1))
        & (x0 < right.unsqueeze(-1).unsqueeze(-1))
        & (y1 > bottom.unsqueeze(-1).unsqueeze(-1))
        & (y0 < top.unsqueeze(-1).unsqueeze(-1))
    ).to(dtype=dtype)


def _thermal_gnn_hrnet_inputs(x_hat, cond, grid_size, rect_sharpness, differentiable=True,
                              per_graph_power=None, per_graph_sizes=None, per_graph_hubump=None):
    """Build the exact GNN schema and a differentiable approximation of its raster schema.

    默认 (三个 per_graph_* 全为 None) 时, 尺寸/功耗/hubump 都取自 `cond` 并在 batch 上共享 ——
    与历史行为逐位一致。给其中一个传入 **按图** 的张量后, 该通道改为逐图取值, 使得
    "同一个 system 的多个候选 (共享画布, 但 body 尺寸 / 功耗 / hubump 各不相同)" 可以一次前向。

    per_graph_* 的索引口径与 `cond` 里的同名通道一致, 即按 **全部节点** (长度 node_count),
    而不是 active 之后的下标; 允许 (B, ...) 或 (..., ) 两种形状 (后者自动 broadcast)。
    """
    if x_hat.dim() == 2:
        x_hat = x_hat.unsqueeze(0)
    batch_size, node_count, _ = x_hat.shape
    dtype, device = x_hat.dtype, x_hat.device
    canvas, origin = _thermal_canvas(cond, device, dtype)
    side = canvas.max().clamp_min(1e-12)
    active = _thermal_active_nodes(cond, device)

    def _per_graph(value, trailing=0):
        """-> (batch_size, node_count, *trailing) 或 (batch_size, node_count)。"""
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        while tensor.dim() < 2 + trailing:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] not in (1, batch_size):
            raise ValueError(
                f"per_graph 通道的 batch 维应为 1 或 {batch_size}, 实际 {tensor.shape[0]}")
        if tensor.shape[1] != node_count:
            raise ValueError(
                f"per_graph 通道的节点维应为 {node_count}, 实际 {tensor.shape[1]}")
        return tensor.expand(batch_size, *tensor.shape[1:]) if tensor.shape[0] == 1 else tensor

    if "tap_source_chiplet_sizes" in cond and cond.tap_source_chiplet_sizes.shape[0] == node_count:
        all_sizes = cond.tap_source_chiplet_sizes.to(device=device, dtype=dtype)
    else:
        all_sizes = cond.x[:, :2].to(device=device, dtype=dtype) * canvas.view(1, 2) / 2.0
    sizes = all_sizes[active].clamp_min(1e-12)
    n_active = sizes.shape[0]
    sizes_bt = (_per_graph(per_graph_sizes, trailing=1)[:, active] if per_graph_sizes is not None
                else sizes.view(1, -1, 2).expand(batch_size, -1, -1)).clamp_min(1e-12)

    centers = ((x_hat[..., :2] + 1.0) * canvas.view(1, 1, 2) / 2.0 + origin.view(1, 1, 2))[:, active]
    lower = centers - sizes_bt / 2.0

    if per_graph_power is not None:
        powers_bt = _per_graph(per_graph_power)[:, active].abs()
    elif "node_power" in cond:
        powers_bt = cond.node_power.to(device=device, dtype=dtype).view(-1)[active].abs() \
            .view(1, -1).expand(batch_size, -1)
    else:
        powers_bt = torch.ones((batch_size, n_active), dtype=dtype, device=device)

    if per_graph_hubump is not None:
        hubump_bt = _per_graph(per_graph_hubump)[:, active].clamp_min(0.0)
    else:
        hubump_bt = _thermal_hubump_widths(cond, sizes, active, device, dtype) \
            .view(1, -1).expand(batch_size, -1)

    centers01 = (centers - origin.view(1, 1, 2)) / side
    sizes01 = sizes_bt / side
    lower01 = (lower - origin.view(1, 1, 2)) / side
    hubump01 = hubump_bt / side
    area = sizes_bt[..., 0] * sizes_bt[..., 1]
    hubump_area = 2.0 * (sizes_bt[..., 0] + sizes_bt[..., 1]) * hubump_bt
    nodes = torch.stack(
        (
            powers_bt / 200.0,
            sizes01[..., 0],
            sizes01[..., 1],
            area / side.square(),
            centers01[..., 0],
            centers01[..., 1],
            hubump_bt / 2.0,
            hubump_area / side.square(),
        ),
        dim=-1,
    )

    local = torch.arange(n_active, dtype=torch.long, device=device)
    src = local.repeat_interleave(n_active)
    dst = local.repeat(n_active)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    edge_copies = [torch.stack((src, dst), dim=0) + graph_id * n_active for graph_id in range(batch_size)]
    edge_index = torch.cat(edge_copies, dim=1) if edge_copies else torch.empty((2, 0), dtype=torch.long, device=device)
    if src.numel():
        edge_attr = torch.cat(
            [torch.linalg.vector_norm(centers01[graph_id, src] - centers01[graph_id, dst], dim=-1)
             for graph_id in range(batch_size)],
            dim=0,
        ).view(-1, 1)
    else:
        edge_attr = torch.empty((0, 1), dtype=dtype, device=device)

    left = lower01[..., 0]
    bottom = lower01[..., 1]
    right = left + sizes01[..., 0]
    top = bottom + sizes01[..., 1]
    body = _thermal_rect_field(left, bottom, right, top, grid_size, rect_sharpness, differentiable)
    layout = body.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    density = powers_bt / area / 10.0
    power = (body * density.view(batch_size, n_active, 1, 1)).sum(dim=1, keepdim=True)

    hb = hubump01
    strip_masks = (
        _thermal_rect_field(left - hb, bottom, left, top, grid_size, rect_sharpness, differentiable),
        _thermal_rect_field(left, top, right, top + hb, grid_size, rect_sharpness, differentiable),
        _thermal_rect_field(right, bottom, right + hb, top, grid_size, rect_sharpness, differentiable),
        _thermal_rect_field(left, bottom - hb, right, bottom, grid_size, rect_sharpness, differentiable),
    )
    valid_hubump = (hubump_bt > 0.0).to(dtype=dtype).view(batch_size, n_active, 1, 1)
    hubump_mask = torch.stack(strip_masks, dim=0).sum(dim=0).mul(valid_hubump)
    hubump_field = hubump_mask.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    field = torch.cat((power, layout, hubump_field), dim=1)

    return {
        "x": nodes.reshape(batch_size * n_active, 8),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": torch.arange(batch_size, device=device).repeat_interleave(n_active),
        "field": field,
    }


def _thermal_forward(model, x_hat, cond, grid_size, rect_sharpness, stats=None, differentiable=True,
                     per_graph_power=None, per_graph_sizes=None, per_graph_hubump=None):
    if getattr(model, "_flow_tap_thermal_kind", None) == "gnn_hrnet":
        inputs = _thermal_gnn_hrnet_inputs(
            x_hat,
            cond,
            grid_size=grid_size,
            rect_sharpness=rect_sharpness,
            differentiable=differentiable,
            per_graph_power=per_graph_power,
            per_graph_sizes=per_graph_sizes,
            per_graph_hubump=per_graph_hubump,
        )
        return model(
            inputs["x"],
            inputs["edge_index"],
            inputs["batch"],
            inputs["edge_attr"],
            inputs["field"],
        )

    power_grid, layout_grid, total_power = _thermal_rasterize(
        x_hat,
        cond,
        grid_size=grid_size,
        rect_sharpness=rect_sharpness,
        stats=stats,
    )
    return model(power_grid, layout_grid, total_power)


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
    flow_cfg = dict(cfg.get("flow", {}) or {})
    wirelength_cfg = dict(cfg.get("wirelength", {}) or {})
    bbox_cfg = dict(cfg.get("bbox", {}) or {})
    legality_aux_cfg = dict(cfg.get("legality_aux", {}) or {})
    monitor_cfg = dict(cfg.get("monitor", {}) or {})
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
        wirelength_cfg=wirelength_cfg,
        bbox_cfg=bbox_cfg,
        legality_aux_cfg=legality_aux_cfg,
        flow_cfg=flow_cfg,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    train_metrics = common.Metrics()
    monitor_metrics = common.Metrics()

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
    if monitor_cfg.get("enabled", False):
        print(f"live training monitor: {os.path.join(log_dir, monitor_cfg.get('filename', 'training_monitor.png'))}")
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
        monitor_metrics.add({"loss": loss.detach().cpu().item()})
        monitor_metrics.add(model_metrics)
        step.increment()

        monitor_every = max(1, int(monitor_cfg.get("every", 100) or 100))
        if monitor_cfg.get("enabled", False) and int(step) % monitor_every == 0:
            monitor_logs = monitor_metrics.result()
            try:
                update_training_monitor(
                    log_dir,
                    int(step),
                    monitor_logs,
                    filename=str(monitor_cfg.get("filename", "training_monitor.png")),
                    history_filename=str(monitor_cfg.get("history_filename", "training_monitor.jsonl")),
                )
            except Exception as error:
                print(f"warning: failed to update training monitor: {error}", flush=True)

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
