"""采样后的热精修: 在固定画布内做合法性保持的局部搜索, 降低峰值温度。

为什么要单独做这一步
--------------------
flow-matching 采样时已经有热引导 (`eval_thermal_guided.py::_thermal_guided_step`),
但实测它结构性受限, 不是"力不够":

  * 采样循环里每个时间步是 `x_next = x - dt*v`(模型流场) 再叠加引导力。
    模型学到的是"布局铺满画布"(训练时画布定义为 GT 布局的外接框,
    `canvas_padding_mm: 0.0`), 加上 `bbox_target_ratio=0.95` 的训练损失继续压紧凑度。
    所以流场更新每一步都把引导推出去的位置往回拉, 引导只是叠加在它上面的扰动。
  * 线长引导 (`models.py::reverse_guidance_opt_force`, SGD lr=0.1 x 8 步, 无 clip)
    位移能力约 2.7mm, 方向是靠近, 和热引导反向。
  * 热引导只有 `guidance_steps=1` 个内层步。

实测结果(seed 5, 12 个 case): 把热引导从失效修到生效后, 布局确实移动了
(中位 0.55mm, 最大 2.6mm), 但峰值温度只降 0.53K。同样这 12 个布局、同一个热代理、
同一个画布, 换成采样后直接做合法性保持的重排搜索, 400 轮拿到 6-13K。

关键几何约束(决定了这里**不能**摊开, 只能重排)
------------------------------------------------
  * 画布是固定的: `json_benchmark_dataset.py:102` `side = sqrt(chiplet_body_area / 0.50)`
    (`FLOW_TAP_CHIPLET_AREA_RATIO=0.50` 覆盖了 config.yaml 里的 0.30)。
  * 模型产出的布局 `跨度/画布 ≈ 0.98`, 12 个 case 全一样 —— 它已经把画布填满了。
  * 所以唯一可用的杠杆是**在画布内重新排列**, 不是把布局摊开。

坐标约定
--------
全程是**归一化中心坐标 `[-1,1]`**, 没有离散网格。
`utils.preprocess_graph` (`utils.py:830,839`): `cond.x = 2*(cond.x/chip_size)`,
`x = 2*(x/chip_size) - 1 + cond.x/2`。边界就是 `[-1,1]`
(`utils.py:1483` 的 `shapely.box(-1,-1,1,1)`)。
FlowPlace 那种"吸附到可行网格单元"在这个代码库里没有对应物 —— 位置是连续的,
所以合法性用**候选过滤**实现, 效果等价(保证输出合法)而不引入量化误差。

合法性是怎么保证的
------------------
不做可微的合法性惩罚(那是采样时 `legality_guidance_potential` 的做法, 会和热目标抢梯度),
而是逐候选过滤: 一个候选位置只有在放下去之后**整体仍然不重叠、且不出画布**时才被接受。
被邻居夹住的 chiplet 可接受位置少、几乎动不了; 周围有空位的能自由移动。
这就是"有些位置可以移动有些不行", 逐候选判定, 不需要预先标记哪个 chiplet 可动。

调用顺序很重要: 本模块必须跑在 `legalization_fn` **之后**。
  1. 合法化先把采样输出修正成合法布局(实测 `legalization_before_extra ≈ 1.0`, 基本没动手);
  2. 精修以合法布局为起点, 且候选过滤保证每一步都合法, 所以输出天然合法;
  3. 反过来(先精修再合法化)不行: 那 1000 步合法化会把 400 步精修的热收益大半抹掉。
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from train_graph_thermal import (
    _denorm_temp_k,
    _thermal_forward,
    _thermal_output_to_grid_and_avg,
)

# 实测工作点: `wirelength_weight=1e-4` 时 12 个 case 峰值温度平均 -4.81K、
# 真线长模型平均 -2.06%(中位 -3.35%)。λ=3e-4 则线长稳定改善 12-13% 但温度收益减半。
# λ=0(纯热)会把小 case 的线长搞坏到 +74%, 所以线长项不能省。
DEFAULT_WIRELENGTH_WEIGHT = 1e-4

# 接受判据的容差。用相对容差而不是绝对值: 温度是 O(100) 的摄氏度, 线长是 O(1e5) 的
# mm·wire, 两者量纲差 3 个数量级, 固定绝对容差会让其中一个永远无法触发。
_ACCEPT_REL_TOL = 1e-6

# 几何违规判据的容差, 归一化坐标单位。
#
# 必须非零。实测: legalization 的产物里会出现"两个 chiplet 恰好贴边"的情形,
# 浮点误差把 x 轴间距算成 -2.98e-07 / -4.77e-07(hp11_m, 画布 42mm, 折合 6 纳米)。
# 零容差会把它判成重叠 —— 而合法化的作用正是把 chiplet 推到刚好贴边, 所以这个
# 情形在合法化之后是**常态**而不是异常。一旦误判成起点违规, 精修就退化到下面
# 那个弱得多的"逐步修复"分支: 实测 hp11_m 在 400 轮里浪费掉 399 轮。
#
# 取值依据: 观测到的浮点噪声是 3-5e-7, 1e-5 留了 20-30 倍余量;
# 同时在最小的画布上仍**小于一个数据集坐标网格**(数据网格 1e-3 mm,
# 42mm 画布下 1e-5 归一化 = 2.1e-4 mm ≈ 0.2 个网格), 所以藏不下一个真实的重叠步长。
# 这也和代码库其余判据一致: `utils.check_legality_new` 把面积 round 到 6 位小数,
# `guidance.legality_guidance_potential` 用 softmax 光滑势能, 两者对 3e-7 都无感。
_GEOM_TOL = 1e-5

# 半径内一个合法候选都没有时, 半径乘以这个系数。**实测必须是缩小, 不能放大。**
#
# 直觉上"找不到合法位置就该看得更远", 实测相反。12 个 case、400 轮、同一个 seed,
# 只改这一个系数:
#     *0.97(缩小): ΔK 均值 -6.34K, 接受 73 次/case, 卡住 70 次/case
#     *1.15(放大): ΔK 均值 -5.66K, 接受 53 次/case, 卡住 165 次/case
# 原因是候选过滤: 模型产出的布局已经把画布填到 0.98, 合法空隙是贴着当前位置的
# 一条窄缝。方盒开得越大, 32 个候选越是散落在已被占用的区域上, 全被过滤掉。
# 缩小半径 = 把采样集中到那条窄缝里。
_STUCK_SCALE = 0.97


def _as_1d_mask(mask, num_nodes: int, device) -> Optional[torch.Tensor]:
    """把各种形状的 mask 归一成 (V,) bool。约定 True = 固定不可动。

    这个约定和全代码库一致: `guidance.py:32-35` 用 `inv_mask = ~mask` 把 mask=True
    的节点排除出合法性势能; `eval_thermal_guided.py:1027` 用 `x_guided.grad *= (~mask)`
    把它们的梯度清零。
    """
    if mask is None:
        return None
    m = mask
    if m.dim() == 3:
        m = m.reshape(-1) if m.shape[0] == 1 else m.reshape(m.shape[1], -1)[:, 0]
    elif m.dim() == 2:
        m = m.reshape(-1)
    m = m.to(device=device, dtype=torch.bool)
    if m.numel() != num_nodes:
        raise ValueError(f"mask 有 {m.numel()} 个元素, 但节点数是 {num_nodes}")
    return m


def _violation_counts(x, footprint):
    """每个候选布局的违规数 = 重叠对数 + 出画布的 chiplet 数。

    x:         (C, V, 2) 归一化中心坐标
    footprint: (V, 2)    归一化**全尺寸**(不是半尺寸), 用 TAP 展开后的 footprint

    返回 (C,) int64。0 表示完全合法。
    """
    c, v, _ = x.shape
    half = footprint / 2.0
    # delta > 0 表示该轴上分开了; 两个轴都 < -tol 才算真重叠(盒判交)。
    # tol 不能省, 见 _GEOM_TOL 的注释: 合法化产出的贴边 chiplet 会被零容差误判。
    sep = (x.unsqueeze(2) - x.unsqueeze(1)).abs() - (half.unsqueeze(1) + half.unsqueeze(0))
    overlap = (sep < -_GEOM_TOL).all(dim=-1)  # (C, V, V)
    diag = torch.arange(v, device=x.device)
    overlap[:, diag, diag] = False  # 自配对不算
    overlaps = overlap.any(dim=-1).sum(dim=-1)  # (C,)
    outside = (x.abs() + half > 1.0 + _GEOM_TOL).any(dim=-1).sum(dim=-1)
    return overlaps + outside


class ThermalRefiner:
    """在固定画布内做合法性保持的坐标下降, 目标 = 峰值温度 + λ·线长。"""

    def __init__(
        self,
        evaluator,
        *,
        steps: int = 400,
        candidates: int = 32,
        radius: float = 0.30,
        radius_min: float = 0.005,
        radius_max: float = 0.30,
        stuck_scale: float = 0.97,
        thermal_weight: float = 1.0,
        wirelength_weight: float = DEFAULT_WIRELENGTH_WEIGHT,
        seed: int = 0,
        verbose: bool = False,
    ):
        # evaluator 是 eval_thermal_guided.ThermalEvaluator, 暴露冻结的热代理
        # (model / stats / grid_size / rect_sharpness)。这里直接复用它, 不重新加载,
        # 保证精修和采样时热引导、以及最终评估用的是**同一套权重**。
        self.evaluator = evaluator
        self.steps = int(steps)
        self.candidates = int(candidates)
        self.radius = float(radius)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        # 半径内找不到任何合法位置时的半径变化率。实测必须是**缩小**(见 _STUCK_SCALE 注释)。
        self.stuck_scale = float(stuck_scale)
        self.thermal_weight = float(thermal_weight)
        self.wirelength_weight = float(wirelength_weight)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    # ---- 目标函数 ----------------------------------------------------------

    @torch.no_grad()
    def _peak_celsius(self, x_batch, thermal_cond):
        """(C, V, 2) -> (C,) 峰值温度 ℃。和 `ThermalEvaluator.__call__` 同一条通路,
        只是这里保留了 batch 维(评估候选需要一次算几十个布局)。"""
        ev = self.evaluator
        output = _thermal_forward(
            ev.model,
            x_batch,
            thermal_cond,
            grid_size=ev.grid_size,
            rect_sharpness=ev.rect_sharpness,
            stats=ev.stats,
            differentiable=False,
        )
        temp, _avg = _thermal_output_to_grid_and_avg(output)
        if ev.stats is not None and "temp_min" in ev.stats and "temp_max" in ev.stats:
            temp = _denorm_temp_k(temp, ev.stats)
        # temp 可能是 (C, H, W) 或 (C, 1, H, W); 统一压平成 (C, -1) 再取 max
        return temp.reshape(temp.shape[0], -1).max(dim=1).values - 273.15

    def _wirelength(self, x_batch, edge_index, edge_weight, chip_side):
        """(C, V, 2) -> (C,) wireCount 加权的半周长和, 单位 mm·wire。

        这里用 HPWL 而不是 `wirelength_surrogate` 的 GNN: 每轮要评估 `candidates`
        个候选 x `steps` 轮 = 上万次评估, GNN 每次几十毫秒会让单个 case 从 8 秒变成
        4 分钟。HPWL 是向量化的, 且和真线长模型同向(实测 11/12 个 case 符号一致)。
        `metrics.csv` 里的 `neural_total_wirelength` / CPLEX `tap_avg_wirelength`
        才是最终判据, 用来复核。
        """
        if edge_index is None or edge_weight is None or edge_index.numel() == 0:
            return x_batch.new_zeros(x_batch.shape[0])
        phys = (x_batch + 1.0) / 2.0 * chip_side  # 归一化 -> mm
        delta = (phys[:, edge_index[0], :] - phys[:, edge_index[1], :]).abs().sum(dim=-1)
        return (delta * edge_weight.view(1, -1)).sum(dim=-1)

    def _objective(self, x_batch, thermal_cond, edge_index, edge_weight, chip_side):
        obj = self.thermal_weight * self._peak_celsius(x_batch, thermal_cond)
        if self.wirelength_weight > 0.0:
            obj = obj + self.wirelength_weight * self._wirelength(
                x_batch, edge_index, edge_weight, chip_side
            )
        return obj

    # ---- 主循环 ------------------------------------------------------------

    def refine(
        self,
        x,
        thermal_cond,
        footprint,
        mask=None,
        edge_index=None,
        edge_weight=None,
        chip_side=None,
    ):
        """x: (B, V, 2) 归一化中心坐标(合法化之后的布局)。

        返回 (x_refined, stats)。x_refined 保证合法(候选过滤不通过就不动),
        所以调用方不需要再跑一次合法化。
        """
        if self.steps <= 0 or self.candidates <= 0:
            return x, {"enabled": False}

        device = x.device
        b, v, _ = x.shape
        fp = footprint.to(device=device, dtype=x.dtype)
        fixed = _as_1d_mask(mask, v, device)
        movable = torch.arange(v, device=device) if fixed is None else torch.arange(v, device=device)[~fixed]
        if movable.numel() == 0:
            return x, {"enabled": False, "reason": "no movable chiplet"}

        if chip_side is None:
            chip_side = float(thermal_cond.chip_size[2])
        chip_side = torch.as_tensor(chip_side, device=device, dtype=x.dtype)
        if edge_index is not None:
            edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device=device, dtype=x.dtype)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        out = x.clone()
        stats = {
            "enabled": True,
            "accepted": 0,
            "rejected": 0,
            "no_legal_candidate": 0,
            "repaired": 0,
            "start_violations": 0,
            "end_violations": 0,
        }

        for bi in range(b):
            cur = x[bi : bi + 1].clone()
            cur_viol = int(_violation_counts(cur, fp)[0])
            stats["start_violations"] += cur_viol
            start_peak = float(self._peak_celsius(cur, thermal_cond)[0])

            cur_obj = float(self._objective(cur, thermal_cond, edge_index, edge_weight, chip_side)[0])
            radius = self.radius

            for _ in range(self.steps):
                pick = movable[torch.randint(movable.numel(), (1,), generator=gen)]
                node = int(pick)

                # 候选偏移取在半径 radius 的方盒内(圆盘采样会让四个角更稀疏,
                # 方盒更简单且在这个尺度下没有实际差别)。
                offsets = (torch.rand(self.candidates, 2, generator=gen, dtype=x.dtype) * 2.0 - 1.0)
                offsets = (offsets * radius).to(device)

                cand = cur.repeat(self.candidates, 1, 1)
                cand[:, node, :] = cur[0, node, :] + offsets

                viol = _violation_counts(cand, fp)
                if cur_viol == 0:
                    admissible = viol == 0
                else:
                    # 起点不合法时(理论上不该发生, 因为本模块跑在合法化之后),
                    # 退而求其次只接受"违规数变少"的候选, 逐步修复而不是原地不动。
                    admissible = viol < cur_viol
                if not bool(admissible.any()):
                    stats["no_legal_candidate"] += 1
                    radius = max(radius * self.stuck_scale, self.radius_min)
                    continue

                sub = cand[admissible]
                sub_obj = self._objective(sub, thermal_cond, edge_index, edge_weight, chip_side)
                best = int(torch.argmin(sub_obj))
                best_obj = float(sub_obj[best])
                # 相对容差, 见 _ACCEPT_REL_TOL 的注释
                if best_obj < cur_obj - _ACCEPT_REL_TOL * max(abs(cur_obj), 1.0):
                    if cur_viol > 0:
                        stats["repaired"] += 1
                    cur = sub[best : best + 1].clone()
                    cur_viol = int(_violation_counts(cur, fp)[0])
                    cur_obj = best_obj
                    stats["accepted"] += 1
                    radius = min(radius * 1.05, self.radius_max)
                else:
                    stats["rejected"] += 1
                    radius = max(radius * 0.98, self.radius_min)

            out[bi : bi + 1] = cur
            stats["end_violations"] += cur_viol
            if self.verbose:
                end_peak = float(self._peak_celsius(cur, thermal_cond)[0])
                print(
                    f"    [thermal_refine] case batch {bi}: 峰值 {start_peak:.2f} -> {end_peak:.2f} ℃ "
                    f"({end_peak - start_peak:+.2f}K), 接受 {stats['accepted']} 次",
                    flush=True,
                )

        return out, stats


def refine_positions(
    x,
    evaluator,
    thermal_cond,
    footprint,
    mask=None,
    edge_index=None,
    edge_weight=None,
    chip_side=None,
    *,
    steps: int = 400,
    candidates: int = 32,
    radius: float = 0.30,
    radius_min: float = 0.005,
    radius_max: float = 0.30,
    stuck_scale: float = _STUCK_SCALE,
    thermal_weight: float = 1.0,
    wirelength_weight: float = DEFAULT_WIRELENGTH_WEIGHT,
    seed: int = 0,
    verbose: bool = False,
):
    """便捷入口, 见 `ThermalRefiner`。"""
    refiner = ThermalRefiner(
        evaluator,
        steps=steps,
        candidates=candidates,
        radius=radius,
        radius_min=radius_min,
        radius_max=radius_max,
        stuck_scale=stuck_scale,
        thermal_weight=thermal_weight,
        wirelength_weight=wirelength_weight,
        seed=seed,
        verbose=verbose,
    )
    return refiner.refine(
        x,
        thermal_cond,
        footprint,
        mask=mask,
        edge_index=edge_index,
        edge_weight=edge_weight,
        chip_side=chip_side,
    )


def format_stats(stats) -> str:
    if not stats.get("enabled"):
        return "disabled"
    accepted = stats.get("accepted", 0)
    return (
        f"accepted={accepted} rejected={stats.get('rejected', 0)} "
        f"no_legal_candidate={stats.get('no_legal_candidate', 0)} "
        f"violations {stats.get('start_violations', 0)}->{stats.get('end_violations', 0)}"
    )
