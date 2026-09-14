#!/usr/bin/env python3
"""用真实代理模型评估 placement 记录: 线长 GNN + GNNHRNet 热模型。

两个代理都是冻结的, 输入直接是 placement_dataset 的记录格式
(x-position/y-position/width/height/rotation/power/hubump + connections)。

  · 线长: wirelengthmodel/checkpoint/best_wlmodel_total_60k.pt
      走 wirelengthmodel.dataloader.parse_system -> ChipletWirelengthDataset -> WirelengthGNN。
      模型输出 total 即 **总线长本身** (训练 loss 是 MSE(log pred, log true)), 不是 log。
  · 热:   thermalmodel/checkpoints/gnnhrnet_pwin/best.pth (GNNHRNet, 原生 64x64 场)
      走 train_graph_thermal 的 _thermal_gnn_hrnet_inputs / _thermal_forward。

必须在 chipdiffusion 环境运行:
    /root/anaconda3/envs/chipdiffusion/bin/python
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# 该环境的 wandb 0.13 依赖 numpy<2 的别名, 而实际装的是 numpy 2.x。
# 这里就地补上别名, 让 train_graph_thermal -> utils -> wandb 能 import 成功。
for _alias, _target in (("float_", np.float64), ("int_", np.int64), ("complex_", np.complex128),
                        ("unicode_", np.str_), ("str_", np.str_), ("bool8", np.bool_),
                        ("object_", object), ("long", np.int64)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

PROJECT = Path("/root/placement/flow_tap")
DIFFUSION = PROJECT / "LayoutGenModel" / "diffusion"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(DIFFUSION) not in sys.path:
    sys.path.insert(0, str(DIFFUSION))

WL_CKPT = PROJECT / "wirelengthmodel" / "checkpoint" / "best_wlmodel_total_60k.pt"
WL_NORM = PROJECT / "wirelengthmodel" / "checkpoint" / "normalizer_compat_69_76.json"
TH_CKPT = PROJECT / "thermalmodel" / "checkpoints" / "gnnhrnet_pwin" / "best.pth"

WL_MODEL_CFG = dict(node_dim=18, edge_dim=2, global_dim=4, cong_dim=8, hidden=256,
                    num_layers=6, heads=4, dropout=0.0, use_residual=True, use_global=True)

# 与 PlacementManifestDataset._to_flow_graph 保持一致
CANVAS_PADDING_MM = 1.0
THERMAL_GRID = 64


class ProxyEvaluator:
    """两个冻结代理的统一入口。记录格式与 placement_dataset_*.json 相同。"""

    def __init__(self, device="cuda", wl_ckpt=WL_CKPT, wl_norm=WL_NORM,
                 thermal_ckpt=TH_CKPT, thermal_grid=THERMAL_GRID):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.thermal_grid = thermal_grid

        from wirelengthmodel.dataloader import Normalizer
        from wirelengthmodel.wlmodel import WirelengthGNN

        self.wl_model = WirelengthGNN(**WL_MODEL_CFG)
        self.wl_model.load_state_dict(torch.load(str(wl_ckpt), map_location="cpu"), strict=True)
        self.wl_model.to(self.device).eval()
        for p in self.wl_model.parameters():
            p.requires_grad_(False)
        self.wl_norm = Normalizer.from_dict(json.loads(Path(wl_norm).read_text(encoding="utf-8")))

        from train_graph_thermal import (
            _build_thermal_model_from_ckpt, _denorm_temp_k, _load_thermal_checkpoint,
            _thermal_forward, _thermal_output_to_grid_and_avg, _thermal_stats_from_checkpoint,
        )
        self._th = dict(
            forward=_thermal_forward, grid_and_avg=_thermal_output_to_grid_and_avg,
            denorm=_denorm_temp_k,
        )
        ckpt = _load_thermal_checkpoint(str(thermal_ckpt))
        self.th_model = _build_thermal_model_from_ckpt(ckpt, self.device, model_cfg={"grid_size": thermal_grid})
        self.th_model.eval()
        for p in self.th_model.parameters():
            p.requires_grad_(False)
        self.th_stats = _thermal_stats_from_checkpoint(ckpt)

    # ---------------------------------------------------------------- 线长
    def wirelength(self, records):
        """返回每条记录的总线长预测 (mm)。"""
        from wirelengthmodel.dataloader import ChipletWirelengthDataset, collate_fn, parse_system

        ds = ChipletWirelengthDataset(
            [(None, None, parse_system(r), 1.0, None) for r in records], self.wl_norm)
        out = []
        with torch.no_grad():
            for i in range(0, len(ds), 64):
                batch = collate_fn([ds[j] for j in range(i, min(i + 64, len(ds)))])
                total, _, _ = self.wl_model(
                    batch["x"].to(self.device), batch["edge_index"].to(self.device),
                    batch["edge_attr"].to(self.device), batch["edge_weight"].to(self.device),
                    batch["node_geom"].to(self.device), batch["batch"].to(self.device),
                    batch["global_attr"].to(self.device), batch["cong"].to(self.device))
                out += total.tolist()
        return out

    # ---------------------------------------------------------------- 热
    @staticmethod
    def _to_cond(record):
        """复刻 PlacementManifestDataset._to_flow_graph, 返回 (x_hat, cond)。"""
        from torch_geometric.data import Data

        chiplets = record["chiplets"]
        v = len(chiplets)
        sizes = torch.tensor([[c["width"], c["height"]] for c in chiplets], dtype=torch.float32)
        lower = torch.tensor([[c["x-position"], c["y-position"]] for c in chiplets], dtype=torch.float32)
        hubump = torch.tensor([float(c.get("hubump", 0.0)) for c in chiplets], dtype=torch.float32)
        powers = torch.tensor([float(c.get("power", 0.0)) for c in chiplets], dtype=torch.float32)

        outer_lower = lower - hubump.view(-1, 1)
        outer_upper = lower + sizes + hubump.view(-1, 1)
        spans = outer_upper.amax(0) - outer_lower.amin(0)
        side = float(spans.amax()) + CANVAS_PADDING_MM
        shift = (side - spans) / 2.0 - outer_lower.amin(0)
        lower = lower + shift.view(1, 2)
        scale = torch.tensor([side, side], dtype=torch.float32)

        cond = Data(
            x=2.0 * sizes / scale.view(1, 2),
            is_ports=torch.zeros(v, dtype=torch.bool),
            is_macros=torch.ones(v, dtype=torch.bool),
            node_power=powers,
            chip_size=torch.tensor([0.0, 0.0, side, side], dtype=torch.float32),
            tap_hubump=hubump,
            tap_source_chiplet_sizes=sizes.clone(),
        )
        x_hat = 2.0 * lower / scale.view(1, 2) - 1.0 + cond.x / 2.0
        return x_hat, cond

    def thermal(self, records):
        """返回每条记录的 {max_c, mean_c, p99_c}。逐条前向 (各 system 画布不同)。"""
        out = []
        for record in records:
            x_hat, cond = self._to_cond(record)
            with torch.no_grad():
                raw = self._th["forward"](self.th_model, x_hat.unsqueeze(0).to(self.device), cond,
                                          grid_size=self.thermal_grid, rect_sharpness=80.0,
                                          stats=self.th_stats, differentiable=False)
                grid, _avg = self._th["grid_and_avg"](raw)
                grid_c = self._th["denorm"](grid, self.th_stats) - 273.15
            flat = grid_c.flatten()
            out.append({
                "max_c": float(flat.max()),
                "mean_c": float(flat.mean()),
                "p99_c": float(flat.kthvalue(max(1, int(0.99 * flat.numel()))).values),
            })
        return out

    def evaluate(self, records, with_thermal=True):
        res = {"wl": self.wirelength(records)}
        if with_thermal:
            th = self.thermal(records)
            for k in ("max_c", "mean_c", "p99_c"):
                res[k] = [t[k] for t in th]
        return res


def _summ(vals):
    import statistics as st
    return f"mean {st.mean(vals):.3f}  median {st.median(vals):.3f}  min {min(vals):.3f}  max {max(vals):.3f}"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="对 --before / --after 两份数据集做代理对比")
    ap.add_argument("--before", required=True, help="原始 chiplet_dataset_k.json")
    ap.add_argument("--after", required=True, help="后处理后的同名文件")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-thermal", action="store_true")
    args = ap.parse_args()

    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    sids = sorted(set(before) & set(after), key=lambda s: int(s.split("_")[1]))[: args.limit]

    ev = ProxyEvaluator(device=args.device)
    rb = ev.evaluate([before[s] for s in sids], with_thermal=not args.no_thermal)
    ra = ev.evaluate([after[s] for s in sids], with_thermal=not args.no_thermal)

    print(f"对比 {len(sids)} 个 system\n")
    wb, wa = sum(rb["wl"]), sum(ra["wl"])
    print(f"  总线长      {wb:.4e} -> {wa:.4e}   {wa / wb:.4f}x  ({100 * (1 - wa / wb):+.1f}%)")
    rel = sorted((ra["wl"][i] - rb["wl"][i]) / rb["wl"][i] for i in range(len(sids)))
    print(f"    逐系统     中位 {100 * rel[len(rel) // 2]:+.2f}%   最差 {100 * max(rel):+.2f}%   "
          f"变差 {sum(1 for r in rel if r > 0)}/{len(rel)}")
    for key, label in (("max_c", "峰值温度"), ("mean_c", "平均温度"), ("p99_c", "P99温度")):
        if key not in rb:
            continue
        mb, ma = sum(rb[key]) / len(sids), sum(ra[key]) / len(sids)
        d = sorted(ra[key][i] - rb[key][i] for i in range(len(sids)))
        print(f"  {label}  {mb:.2f} -> {ma:.2f} °C  ({ma - mb:+.2f} K)   "
              f"逐系统中位 {d[len(d) // 2]:+.2f} K  变差 {sum(1 for x in d if x > 0)}/{len(d)}")


if __name__ == "__main__":
    main()
