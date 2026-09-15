"""三个评估器的统一封装: 热代理 / 神经线长 / 外接框面积。

全部在**归一化中心坐标** ``(C, V, 2) in [-1,1]`` 上批量工作 —— 这是
``_thermal_forward`` 和 ``WirelengthSurrogate.predict`` 的共同入参约定,
批量维就是候选维, 所以一次前向能同时评估 C 个候选布局。

实测吞吐 (Case10, V=61, 单张 GPU):
    热代理  32 个候选 152ms / 128 个 601ms / 256 个 1061ms  (~4.2-4.8 ms/布局)
    线长    32 个候选  57ms / 128 个 432ms               (~1.8-3.4 ms/布局)
注意每布局成本在 32~256 区间基本是平的: 批量只摊销调用开销, **不减少总工作量**。
真正的开销杠杆是候选总数 (steps x candidates), 不是 batch 大小。512 会显著劣化。
"""
from __future__ import annotations

import numpy as np

from _bootstrap import resolve_path, setup as _setup

_setup()

__all__ = ["Evaluators"]

DEFAULT_THERMAL_CKPT = "thermalmodel/checkpoints/gnnhrnet_pwin/best.pth"
DEFAULT_WL_CKPT = "wirelengthmodel/checkpoint/best_wlmodel_total_60k.pt"
DEFAULT_WL_NORMALIZER = "wirelengthmodel/checkpoint/normalizer_compat_69_76.json"


class Evaluators:
    """按需惰性加载三个评估器; 未启用的那一个保持为 None, 不占显存也不拖启动时间。"""

    def __init__(
        self,
        device,
        thermal_ckpt=None,
        wl_ckpt=None,
        wl_normalizer=None,
        wl_model="neural",
        grid_size=64,
        rect_sharpness=80.0,
        enable_thermal=True,
        enable_wirelength=True,
    ):
        import torch

        self.device = torch.device(device)
        self.grid_size = int(grid_size)
        self.rect_sharpness = float(rect_sharpness)
        self.wl_model = str(wl_model)
        self.thermal_ckpt = resolve_path(thermal_ckpt, DEFAULT_THERMAL_CKPT)
        self.wl_ckpt = resolve_path(wl_ckpt, DEFAULT_WL_CKPT)
        self.wl_normalizer = resolve_path(wl_normalizer, DEFAULT_WL_NORMALIZER)

        self._thermal_model = None
        self._thermal_stats = None
        self._wirelength = None
        self._enable_thermal = bool(enable_thermal)
        self._enable_wirelength = bool(enable_wirelength) and self.wl_model == "neural"

    # ---- 惰性加载 ---------------------------------------------------------

    def _load_thermal(self):
        if self._thermal_model is not None:
            return self._thermal_model, self._thermal_stats
        from train_graph_thermal import (
            _build_thermal_model_from_ckpt,
            _load_thermal_checkpoint,
            _thermal_stats_from_checkpoint,
        )

        if self.thermal_ckpt is None or not self.thermal_ckpt.exists():
            raise FileNotFoundError(f"热代理 checkpoint 不存在: {self.thermal_ckpt}")
        ckpt = _load_thermal_checkpoint(str(self.thermal_ckpt))
        model_cfg = {"grid_size": self.grid_size, "rect_sharpness": self.rect_sharpness}
        model = _build_thermal_model_from_ckpt(ckpt, self.device, model_cfg)
        model = model.to(self.device)
        self._thermal_stats = _thermal_stats_from_checkpoint(ckpt)
        self._thermal_model = model
        return self._thermal_model, self._thermal_stats

    def _load_wirelength(self):
        if self._wirelength is not None:
            return self._wirelength
        from wirelength_surrogate import WirelengthSurrogate

        if self.wl_ckpt is None or not self.wl_ckpt.exists():
            raise FileNotFoundError(f"线长 checkpoint 不存在: {self.wl_ckpt}")
        self._wirelength = WirelengthSurrogate(
            {
                "ckpt": str(self.wl_ckpt),
                "normalizer": str(self.wl_normalizer) if self.wl_normalizer else None,
                "objective": "log_total",
            },
            self.device,
        )
        return self._wirelength

    # ---- 温度 -------------------------------------------------------------

    def temperature_c(self, x_norm, cond):
        """(C, V, 2) -> (temp (C, G*G) **K**, avg (C,) K or None), 一次批量前向。

        注意单位是**开尔文**, 不是名字里的摄氏度 —— ``_denorm_temp_k`` 把模型的
        0/1 输出映回 stats 里的 ``[temp_min, temp_max]`` (那份 stats 的 unit 是
        celsius, 所以它加了 273.15)。要摄氏度请用 ``peak_celsius`` /
        ``mean_celsius`` / ``peak_and_region_celsius``, 那三个会减掉 273.15。
        本函数保留 K 是因为它同时是"拿整张温度图"的入口, 图本身用 K 更省一次减法;
        但直接拿它当 ℃ 用会差 273.15, 实测就是这么把热点阈值报成 346 的。
        """
        import torch
        from train_graph_thermal import (
            _denorm_temp_k,
            _thermal_forward,
            _thermal_output_to_grid_and_avg,
        )

        model, stats = self._load_thermal()
        x_batch, cond_dev = self._prep(x_norm, cond)
        with torch.no_grad():
            output = _thermal_forward(
                model,
                x_batch,
                cond_dev,
                grid_size=self.grid_size,
                rect_sharpness=self.rect_sharpness,
                stats=stats,
                differentiable=False,
            )
            temp, avg_temp = _thermal_output_to_grid_and_avg(output)
            if stats is not None and "temp_min" in stats and "temp_max" in stats:
                temp = _denorm_temp_k(temp, stats)
                if avg_temp is not None:
                    avg_temp = _denorm_temp_k(avg_temp, stats)
        temp = temp.reshape(temp.shape[0], -1)
        avg = None if avg_temp is None else avg_temp.reshape(-1)
        return temp, avg

    def peak_celsius(self, x_norm, cond):
        """(C, V, 2) -> (C,) 峰值温度 ℃。这是热守卫用的量。"""
        temp, _ = self.temperature_c(x_norm, cond)
        return temp.max(dim=1).values - 273.15

    def mean_celsius(self, x_norm, cond):
        temp, avg = self.temperature_c(x_norm, cond)
        if avg is not None:
            return avg - 273.15
        return temp.mean(dim=1) - 273.15

    def peak_and_region_celsius(self, x_norm, cond, cell_index=None):
        """一次前向同时给 ``(全局峰值 ℃, 热点区最高温 ℃)``。

        热点区守卫每步都要, 而它的代价必须是零 —— ``temperature_c`` 本来就把整张
        温度图算出来了, ``peak_celsius`` 却只留了 max。这里把同一张图再切一次:
        峰值是全局 max, 热点区是**指定格子上的 max** (``cell_index`` 来自
        ``hotspot.locate_hotspot``)。两次归约共用一次前向, 不额外调模型。
        """
        temp, _ = self.temperature_c(x_norm, cond)
        peaks = temp.max(dim=1).values - 273.15
        if cell_index is None:
            return peaks, None
        import torch

        index = torch.as_tensor(np.asarray(cell_index), dtype=torch.long, device=temp.device)
        region = temp.index_select(1, index).max(dim=1).values - 273.15
        return peaks, region

    # ---- 线长 -------------------------------------------------------------

    def wirelength(self, x_norm, cond, sizes_mm=None, origin=None, side=None):
        """(C, V, 2) -> (C,) 线长。

        ``neural`` 走冻结的 WirelengthGNN (mm, 含 bump 绕行, 官方 MAPE 1.2%);
        ``hpwl`` 走解析式 wireCount 加权中心曼哈顿距离和 —— 便宜且可向量化,
        但量级低于神经模型, 只适合同向比较。
        """
        import torch

        x_batch, cond_dev = self._prep(x_norm, cond)
        if self._enable_wirelength:
            surrogate = self._load_wirelength()
            with torch.no_grad():
                prediction = surrogate.predict(x_batch, cond_dev)
            return prediction["total"].reshape(-1)

        if sizes_mm is None or origin is None or side is None:
            raise ValueError("wl_model=hpwl 需要 sizes_mm / origin / side")
        return self._hpwl(x_batch, cond_dev, sizes_mm, origin, side)

    def _hpwl(self, x_batch, cond, sizes_mm, origin, side):
        import torch

        edge_index = cond.edge_index
        if edge_index.numel() == 0:
            return x_batch.new_zeros(x_batch.shape[0])
        weight = cond.edge_weight.to(device=x_batch.device, dtype=x_batch.dtype)
        origin_t = torch.as_tensor(origin, device=x_batch.device, dtype=x_batch.dtype).view(1, 2)
        center_mm = (x_batch[:, :, :2] + 1.0) * float(side) / 2.0 + origin_t
        delta = (center_mm[:, edge_index[0], :] - center_mm[:, edge_index[1], :]).abs().sum(dim=-1)
        return (delta * weight.view(1, -1)).sum(dim=-1)

    # ---- 外接框面积 -------------------------------------------------------

    @staticmethod
    def bbox_area_mm2(x_norm, sizes_mm, origin, side):
        """(C, V, 2) -> (C,) 外接框面积 mm^2。口径同 utils._placement_bbox_stats。"""
        import torch

        x_batch = Evaluators._as_batch(x_norm)
        origin_t = torch.as_tensor(origin, device=x_batch.device, dtype=x_batch.dtype).view(1, 2)
        size_t = torch.as_tensor(sizes_mm, device=x_batch.device, dtype=x_batch.dtype).view(1, -1, 2)
        center_mm = (x_batch[:, :, :2] + 1.0) * float(side) / 2.0 + origin_t
        lower = center_mm - size_t / 2.0
        upper = lower + size_t
        width = (upper[:, :, 0].max(dim=1).values - lower[:, :, 0].min(dim=1).values).clamp_min(0.0)
        height = (upper[:, :, 1].max(dim=1).values - lower[:, :, 1].min(dim=1).values).clamp_min(0.0)
        return width * height

    # ---- 工具 -------------------------------------------------------------

    def _prep(self, x_norm, cond):
        """把 x 和 cond 一并搬到 self.device。

        ``cond`` 必须在每次调用时都搬: 它可能刚由 ``build_cond`` 在 CPU 上构造出来,
        而模型在 GPU 上 —— 漏掉这一步会直接报 device 不匹配。张量都很小 (V<=61),
        重复拷贝的代价可以忽略。
        """
        x_batch = self._as_batch(x_norm)
        if x_batch.device != self.device:
            x_batch = x_batch.to(self.device)
        if cond is not None and getattr(cond, "x", None) is not None and cond.x.device != self.device:
            cond = cond.to(self.device)
        return x_batch, cond

    @staticmethod
    def _as_batch(x_norm):
        import torch

        if not isinstance(x_norm, torch.Tensor):
            x_norm = torch.as_tensor(np.asarray(x_norm), dtype=torch.float32)
        if x_norm.dim() == 2:
            x_norm = x_norm.unsqueeze(0)
        return x_norm
