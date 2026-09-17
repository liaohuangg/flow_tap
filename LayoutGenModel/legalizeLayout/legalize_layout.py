"""布局合法化 + 热点冻结 + 线长优化。CLI 入口。

    Stage A   最小重叠修复   —— 只推动重叠的 chiplet, 其余位移恒为 0
    热点定位                —— 合法化**之后**的温度图 top-5% 格子 -> 要冻结的 chiplet
    Stage B   拉近优化       —— 冻结/守卫热点件, 牵引其余件朝连线重心靠拢以降线长
    Stage C   独立校验 + 输出

阶段顺序不可换: 去重叠会让热点挪位, 按输入布局定热点会冻住一个已经不在热点的 chiplet。
三条守卫 (参照系都是**输入**布局): 全局峰值 <= +0.50K, 神经线长 <= +1.20%, 热点区 <= +0.50K。

用法
----
    PY=/root/anaconda3/envs/chipdiffusion/bin/python
    $PY legalize_layout.py --input <placement.json 或目录>
    $PY legalize_layout.py --input <目录> --output-dir /tmp/out
    $PY legalize_layout.py --input a.json --set refine.steps=400
    $PY legalize_layout.py --input <目录> --set refine.hotspot.freeze=false

    覆盖配置一律走 ``--set key.sub=value``, 没有 ``--refine.steps=400`` 这种写法。

时间开销 (单张 GPU, 实测吞吐见 evaluators.py 顶部注释)
    Stage A  仅重叠分量内的 chiplet 参与, 每步最多一次 batch-4 前向; 无重叠时 0 次
    Stage B  单步 = batch-top_k 热 + batch-top_k 线长; 默认 200 步 x (4+4) 次 ≈ 5 秒/case
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from _bootstrap import setup as _setup

_setup()  # 必须在 import 仓库模块之前: 挂 sys.path + 补 numpy 2.0 别名

from cond_builder import build_cond  # noqa: E402
from evaluators import Evaluators  # noqa: E402
from hotspot import locate_hotspot  # noqa: E402
from plot_layout import plot_comparison  # noqa: E402
from layout_io import (  # noqa: E402
    bbox_area_mm2,
    compute_canvas,
    hpwl_weighted,
    load_placement,
    make_report_record,
    to_norm_centers,
    write_placement,
)
from refine_wl import format_stats as format_refine_stats  # noqa: E402
from refine_wl import refine_wirelength  # noqa: E402
from repair_overlap import repair_overlap  # noqa: E402
from verify import verify_layout  # noqa: E402

_HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = _HERE / "configs" / "default.yaml"


# ---- 配置 -----------------------------------------------------------------


def _coerce(text: str):
    """把 CLI 覆盖值按 YAML 标量解析, 这样 --x=false / --x=1e-3 / --x=[1,2] 都能用。"""
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return text


def load_config(path, overrides):
    import yaml

    config = {}
    if path is not None and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set 需要 key=value 形式, 收到: {item!r}")
        key, value = item.split("=", 1)
        node = config
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise SystemExit(f"--set 的路径 {key!r} 与已有配置冲突 (中间项不是字典)")
        node[parts[-1]] = _coerce(value)
    return config


def _get(config, path, default=None):
    node = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# ---- 评估器装配 -----------------------------------------------------------


def build_evaluators(config, device):
    paths = config.get("paths", {}) or {}
    return Evaluators(
        device,
        thermal_ckpt=paths.get("thermal_ckpt"),
        wl_ckpt=paths.get("wirelength_ckpt"),
        wl_normalizer=paths.get("wirelength_normalizer"),
        enable_thermal=True,
        enable_wirelength=True,
    )


# ---- 单 case 处理 ---------------------------------------------------------


def process_one(path, config, evaluators, out_dir, device, verbose=False):
    started = time.perf_counter()
    record, layout = load_placement(path)
    names, sizes, power = layout.names, layout.sizes, layout.power
    V = layout.V

    repair_cfg = config.get("repair", {}) or {}
    refine_cfg = config.get("refine", {}) or {}
    repair_on = bool(repair_cfg.get("enabled", True))
    refine_on = bool(refine_cfg.get("enabled", True))
    base_scale = float(_get(config, "canvas.scale", 1.0))
    pad_mm = float(_get(config, "canvas.pad_mm", 0.0))

    def guard_for(section, origin, side, cond):
        guard = _get(config, f"{section}.guard", {}) or {}
        thermal_tol = float(guard.get("thermal_tol_k", 0.0))
        wl_tol = float(guard.get("wl_tol", 0.0))

        def evaluate(lower_batch):
            batch = np.asarray(lower_batch, dtype=np.float64)
            if batch.ndim == 2:
                batch = batch[None]
            x_norm = np.stack([to_norm_centers(b, sizes, origin, side) for b in batch], axis=0)
            peaks = evaluators.peak_celsius(x_norm, cond).detach().cpu().numpy().astype(np.float64)
            wls = (
                evaluators.wirelength(x_norm, cond, sizes, origin, side)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            return peaks, wls

        return evaluate, thermal_tol, wl_tol

    def prepare(scale):
        """给定画布放大系数, 重建 ``(origin, side, cond)`` 并跑一遍 Stage A。

        换画布**必须连 cond 一起重建**: 神经模型吃的是画布归一化坐标
        (``to_norm_centers`` / ``cond_builder``), 画布一变, 同一份物理布局在模型眼里
        就是另一组输入 —— 尺寸归一化 ``2*size_mm/side`` 也跟着变。复用旧 cond 会得到
        一组口径不一致的读数, 所以每次重试都整个重建。
        """
        origin, side = compute_canvas(layout, scale=scale, pad_mm=pad_mm)
        cond = build_cond(
            record, layout, origin, side, hubump_source=_get(config, "hubump.source", "derived")
        )
        cur = layout.lower.copy()
        stats = {"enabled": False}
        if repair_on:
            evaluate = None
            if evaluators is not None and bool(_get(config, "repair.guard.enabled", True)):
                evaluate, _, _ = guard_for("repair", origin, side, cond)
            cur, stats = repair_overlap(
                cur,
                sizes,
                origin,
                side,
                repair_cfg,
                evaluate=evaluate,
                cond=cond,
                device=device,
                verbose=verbose,
            )
        return origin, side, cond, cur, stats

    # ---- Stage A ----
    started_a = time.perf_counter()
    origin, side, cond, lower, repair_stats = prepare(base_scale)

    # ---- 画布升级 ----
    # 第一轮在自己的 bbox 里解不开时, 唯一的出路往往是**给画布多点地方**。实测
    # Case6_candidate35: 20 对重叠, 换遍 7 种策略、阻尼 0.4/0.6/0.8 全都停在 1 对
    # (那个局部区域把某一块推出去只能换来一个**更大**的重叠, 或者直接出画布); 画布
    # 放大 12% 之后 269 轮就收敛到 0 对, 而且 ΔWL 反而 -7.84%。所以卡点不是温度/线长
    # 守卫, 是画布这个硬约束。
    #
    # 触发条件限死在"**整份布局都要重排**" (所有 chiplet 都在重叠分量里): 这才是
    # "输入本身就是一份非法摆放"的特征。少数几块互相压住时仍严格待在输入 bbox 内,
    # 保住"最小干预"的承诺 —— 为一个小局部重叠把整张图撑大是不划算的。
    escalation = _get(config, "repair.canvas_escalation", {}) or {}
    if (
        repair_on
        and bool(escalation.get("enabled", True))
        and repair_stats.get("enabled")
        and repair_stats.get("overlap_pairs_end", 0) > 0
        and int(repair_stats.get("movable_nodes", 0)) == V
    ):
        lo_scale = base_scale                      # 已知不合法 (刚跑过)
        hi_scale = float(escalation.get("max_scale", 1.25))
        rel_tol = float(escalation.get("rel_tol", 0.03))
        max_evals = int(escalation.get("max_evals", 5))
        evals = 0
        origin_hi, side_hi, cond_hi, lower_hi, stats_hi = prepare(hi_scale)
        evals += 1
        best = None
        if stats_hi.get("overlap_pairs_end", 0) == 0:
            best = (hi_scale, origin_hi, side_hi, cond_hi, lower_hi, stats_hi)
            hi = hi_scale
            # 二分找**最小**够用的系数。维持不变式: lo 不合法 / hi 合法。
            # 同一档的两次运行结果会因 GPU 浮点不确定而略有出入 (见 README), 所以
            # 只把**实测跑出过 0 对**的那些档记下来, 最后返回的一定是亲眼看它合法的那档。
            while best is not None and evals < max_evals and (hi - lo_scale) / lo_scale > rel_tol:
                mid = 0.5 * (lo_scale + hi)
                o_m, s_m, c_m, l_m, st_m = prepare(mid)
                evals += 1
                if st_m.get("overlap_pairs_end", 0) == 0:
                    hi = mid
                    best = (mid, o_m, s_m, c_m, l_m, st_m)
                else:
                    lo_scale = mid
        if best is not None:
            used_scale, origin, side, cond, lower, repair_stats = best
            repair_stats["canvas_escalation"] = {
                "trigger": "整份布局都在重叠分量内 (movable_nodes == V)",
                "base_scale": base_scale,
                "used_scale": used_scale,
                "max_scale": hi_scale,
                "evals": evals,
            }
            if verbose:
                print(
                    f"        画布升级: scale {base_scale:.3f} -> {used_scale:.3f} "
                    f"({evals} 次重试, 重叠 {repair_stats['overlap_pairs_end']} 对)",
                    flush=True,
                )
        else:
            repair_stats["canvas_escalation"] = {
                "trigger": "整份布局都在重叠分量内 (movable_nodes == V)",
                "base_scale": base_scale,
                "used_scale": None,
                "max_scale": hi_scale,
                "evals": evals,
                "exhausted": True,     # 放到 max_scale 仍解不开
            }
            if verbose:
                print(f"        画布升级: 放到 {hi_scale:.2f} 倍仍解不开", flush=True)

    report = {
        "input": str(path),
        "num_chiplets": V,
        "num_connections": len(record.get("connections", []) or []),
        "canvas": {"origin": [float(origin[0]), float(origin[1])], "side_mm": float(side)},
        # before 一律按**最终画布**算, 不能用最初那档 —— 神经模型吃归一化坐标, 两联
        # 用不同画布算出来的 ΔT/ΔWL 是两个口径的差, 没有意义。
        "before": _geometric_metrics(layout.lower, sizes, origin, side, record, layout),
    }
    if evaluators is not None:
        _fill_neural_metrics(report["before"], layout.lower, sizes, origin, side, cond, evaluators)

    stage_a_time = time.perf_counter() - started_a
    report["repair"] = repair_stats

    # ---- 热点定位 (在**合法化之后**的布局上做, 见 hotspot.py) ----
    hotspot_info, frozen = None, None
    hotspot_cfg = _get(config, "refine.hotspot", {}) or {}
    if evaluators is not None and bool(hotspot_cfg.get("enabled", True)):
        started_h = time.perf_counter()
        hotspot_info, frozen = _locate(lower, sizes, origin, side, cond, evaluators, names, hotspot_cfg)
        report["hotspot"] = hotspot_info
        report["hotspot"]["locate_time_s"] = round(time.perf_counter() - started_h, 3)
        if verbose:
            print(
                f"        热点: {hotspot_info['names']} "
                f"(top {hotspot_info['hot_frac'] * 100:.0f}% = {hotspot_info['hot_cells']} 格, "
                f"阈值 {hotspot_info['threshold_C']:.1f}°C, "
                f"格子覆盖率 {hotspot_info['coverage'] * 100:.1f}%) -> 冻结 {len(hotspot_info['indices'])}/"
                f"{V} 个"
            )

    # ---- Stage B ----
    stage_b_time = 0.0
    if refine_on and evaluators is not None:
        started_b = time.perf_counter()
        # baseline 用**输入布局**的读数, 不是 Stage A 的终点 —— 否则 Stage A 已花掉的
        # 温度预算在 Stage B 里不算数, 两边各用满一条带宽, 净变化直接出带。
        baseline = (
            (report["before"]["peak_temp_C"], report["before"]["wirelength_neural_mm"])
            if "wirelength_neural_mm" in report["before"]
            else None
        )
        # 热点区守卫的参照值: **输入布局**在同一组热点格子上的最高温。格子是在 Stage A
        # 的输出上定的, 读数却取自输入 —— 这样"热点升了多少"才是相对原始布局量的。
        region_baseline = None
        if hotspot_info is not None:
            region_baseline = _region_celsius(
                layout.lower, sizes, origin, side, cond, evaluators, hotspot_info["hot_cell_index"]
            )
            report["hotspot"]["input_region_C"] = float(region_baseline)
        lower, refine_stats = refine_wirelength(
            lower,
            sizes,
            origin,
            side,
            refine_cfg,
            evaluators,
            cond,
            record=record,
            baseline=baseline,
            frozen=frozen,
            hotspot=hotspot_info,
            region_baseline=region_baseline,
            seed=int(refine_cfg.get("seed", 0)),
            verbose=verbose,
        )
        stage_b_time = time.perf_counter() - started_b
        report["refine"] = refine_stats
    else:
        report["refine"] = {"enabled": False, "reason": "disabled or no evaluators"}

    # ---- Stage C: 独立校验 + 输出 ----
    report["after"] = _geometric_metrics(lower, sizes, origin, side, record, layout)
    if evaluators is not None:
        _fill_neural_metrics(report["after"], lower, sizes, origin, side, cond, evaluators)
    displacement = np.linalg.norm(lower - layout.lower, axis=1)
    report["per_chiplet_displacement_mm"] = {
        names[i]: float(displacement[i]) for i in range(V)
    }
    report["max_displacement_mm"] = float(displacement.max()) if V else 0.0
    report["displacement_total_mm"] = float(displacement.sum())
    thermal_tol = float(_get(config, "repair.guard.thermal_tol_k", 0.0))
    wl_tol = float(_get(config, "repair.guard.wl_tol", 0.0))
    delta_t = _delta(report, "peak_temp_C")
    delta_wl = _delta(report, "wirelength_neural_mm", relative=True)
    report["guard_audit"] = {
        # 权威读数全来自神经模型; HPWL 只作同向交叉验证
        "peak_temp_delta_C": delta_t,
        "wirelength_neural_delta_rel": delta_wl,
        "wirelength_hpwl_delta_rel": _delta(report, "wirelength_hpwl_mm", relative=True),
        "bbox_area_delta_rel": _delta(report, "bbox_area_mm2", relative=True),
        # 需求: 温度与线长的上升都要落在**模型容错范围**内。带宽是两个模型各自的实测精度
        # (热代理组内 MAE 0.50K / 线长 GNN MAPE 1.20%), 见 configs/default.yaml。
        "thermal_tol_k": thermal_tol,
        "wl_tol_rel": wl_tol,
        "within_thermal_tolerance": bool(delta_t is not None and delta_t <= thermal_tol),
        "within_wl_tolerance": bool(delta_wl is not None and delta_wl <= wl_tol),
        # Stage A 内部让线长上升过 (去重叠的必然代价), 账交给 Stage B 还;
        # 还成没还成看上面前两项判定
        "stage_a_tolerated_wl_rise": bool(report["repair"].get("wl_rise_tolerated", False)),
    }
    # 热点: 冻结件是否真的一位没动 + 热点区自己的升降温。用户口径里"劣化热点"=
    # 热点升 0.5K 以上, 这一条是它的直接判定。
    #
    # 位移取的是 **Stage B 内部** 测的那个 (相对 Stage B 起点 = Stage A 输出), 不是
    # 这里的 `displacement` (相对**输入**)。两者不是一回事: 热点是在 Stage A **之后**
    # 定位的, 那个 chiplet 在 Stage A 里为解重叠合法地位移过 (实测 acend910 的 A 走了
    # 3.45mm), 那不算违约。要证的是"冻住之后它没再动"。
    if report.get("refine", {}).get("frozen_chiplets"):
        moved = report["refine"].get("frozen_displacement_mm")
        report["guard_audit"]["frozen_max_displacement_mm"] = moved
        report["guard_audit"]["frozen_stayed_put"] = bool(
            moved is not None and moved <= float(_get(config, "repair.geom_tol_mm", 1e-6))
        )
        report["guard_audit"]["frozen_displacement_since_input_mm"] = float(
            displacement[np.asarray(report["refine"]["frozen_chiplets"], dtype=np.int64)].max()
        )
    if isinstance(report.get("hotspot"), dict) and evaluators is not None and "input_region_C" in report["hotspot"]:
        after_region = _region_celsius(
            lower, sizes, origin, side, cond, evaluators, report["hotspot"]["hot_cell_index"]
        )
        report["hotspot"]["after_region_C"] = float(after_region)
        report["hotspot"]["region_delta_k"] = float(after_region - report["hotspot"]["input_region_C"])
        report["hotspot"]["degraded_beyond_tol"] = bool(
            report["hotspot"]["region_delta_k"] > thermal_tol
        )
        report["guard_audit"]["hotspot_region_delta_C"] = report["hotspot"]["region_delta_k"]
        report["guard_audit"]["hotspot_degraded_beyond_tol"] = report["hotspot"]["degraded_beyond_tol"]
    report["timing_s"] = {
        "stage_a_repair": round(stage_a_time, 3),
        "stage_b_refine": round(stage_b_time, 3),
        "total": round(time.perf_counter() - started, 3),
    }
    report["config"] = config

    out_dir = Path(out_dir)
    stem = Path(path).stem
    suffix = str(_get(config, "output.suffix", "_legal"))
    out_path = write_placement(record, names, lower, sizes, out_dir / f"{stem}{suffix}.json")
    report["output"] = str(out_path)

    if bool(_get(config, "output.save_fig", True)):
        edges = _edge_list(record, names)
        report["figure"] = str(
            plot_comparison(
                out_dir / f"{stem}{suffix}_compare.png",
                names,
                layout.lower,
                lower,
                sizes,
                edges,
                metrics_before=report["before"],
                metrics_after=report["after"],
                canvas=(origin, side),
                title=f"{stem}{suffix}   (input vs {suffix})",
                geom_tol=float(_get(config, "repair.geom_tol_mm", 1e-6)),
                hotspot=(report.get("hotspot") or {}).get("indices"),
            )
        )

    if bool(_get(config, "output.report", True)):
        report_path = out_dir / f"{stem}{suffix}_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(make_report_record(report), handle, indent=2, ensure_ascii=False)
        report["report"] = str(report_path)

    return report


def _edge_list(record, names):
    """connections -> [(src_idx, dst_idx, wireCount)], 只保留两端都在布局里的连线。"""
    index = {str(n): i for i, n in enumerate(names)}
    edges = []
    for conn in record.get("connections", []) or []:
        src = index.get(str(conn.get("node1", "")))
        dst = index.get(str(conn.get("node2", "")))
        if src is None or dst is None or src == dst:
            continue
        weight = conn.get("wireCount", conn.get("wire_count", 1)) or 1
        edges.append((src, dst, float(weight)))
    return edges


def _locate(lower, sizes, origin, side, cond, evaluators, names, cfg):
    """在**当前**布局上定位热点, 返回 ``(info, frozen_mask)``。

    时机很关键: 用户口径是"合法化**之后**固定热点位置的 chiplet", 所以这里跑在
    Stage A 输出上, 不是输入上。Stage A 已经把重叠推开了, 热点可能因此挪位; 按输入
    定位会冻住一个已经不在热点的 chiplet。
    """
    x_norm = to_norm_centers(lower, sizes, origin, side)[None]
    temp, _ = evaluators.temperature_c(x_norm, cond)
    # temperature_c 给的是**开尔文** (见它的 docstring), 这里转成摄氏度再往下传,
    # 免得热点阈值以 K 的形式混进一份全 ℃ 的报告里。
    temp_c = temp[0].cpu().numpy() - 273.15
    info = locate_hotspot(
        temp_c,
        lower,
        sizes,
        origin,
        side,
        names=names,
        top_frac=float(cfg.get("top_frac", 0.05)),
        min_cells=int(cfg.get("min_cells", 1)),
        min_movable=int(cfg.get("min_movable", 2)),
    )
    frozen = None
    if bool(cfg.get("freeze", True)):
        frozen = np.zeros(int(np.asarray(sizes).shape[0]), dtype=bool)
        frozen[np.asarray(info["indices"], dtype=np.int64)] = True
    return info, frozen


def _region_celsius(lower, sizes, origin, side, cond, evaluators, cell_index) -> float:
    """指定布局在**热点格子**上的最高温 ℃。一次前向, 与峰值共用。"""
    x_norm = to_norm_centers(lower, sizes, origin, side)[None]
    _, region = evaluators.peak_and_region_celsius(x_norm, cond, np.asarray(cell_index))
    return float(region[0])


def _fill_neural_metrics(node, lower, sizes, origin, side, cond, evaluators) -> None:
    """把神经模型的读数写进 ``node`` (前/后各调一次)。

    两个模型都只前向一次, 但**线长必须走神经模型**, 不能用 HPWL 代理代替:
    HPWL 只是个便宜的同向排序量, 拿它当"线长没劣化"的证据是自证 —— 两个模型的
    读数都留在报告里, 交叉验证时能看出方向是否一致。
    """
    x_norm = to_norm_centers(lower, sizes, origin, side)[None]
    node["peak_temp_C"] = float(evaluators.peak_celsius(x_norm, cond)[0])
    node["wirelength_neural_mm"] = float(
        evaluators.wirelength(x_norm, cond, sizes, origin, side)[0]
    )


def _geometric_metrics(lower, sizes, origin, side, record, layout) -> dict:
    verdict = verify_layout(lower, sizes, origin, side)
    return {
        "overlap_pairs": verdict["overlap_pairs"],
        "overlap_area_mm2": verdict["overlap_area_mm2"],
        "out_of_canvas": verdict["out_of_canvas"],
        "legal_ratio": verdict["legal_ratio"],
        "is_legal": verdict["is_legal"],
        "bbox_area_mm2": float(bbox_area_mm2(lower, sizes)),
        "wirelength_hpwl_mm": float(
            hpwl_weighted(
                type(layout)(list(layout.names), lower, sizes, layout.power),
                record.get("connections", []),
            )
        ),
        "overlap_detail": verdict["overlap_detail"],
        "out_of_canvas_detail": verdict["out_of_canvas_detail"],
    }


def _delta(report, key, relative=False):
    before, after = report["before"].get(key), report["after"].get(key)
    if before is None or after is None:
        return None
    if relative:
        return float(after / before - 1.0) if before else None
    return float(after - before)


# ---- 入口 -----------------------------------------------------------------


def collect_inputs(target):
    path = Path(target)
    if path.is_dir():
        found = sorted(path.rglob("*_placement.json"))
        if not found:
            found = sorted(path.rglob("*.json"))
        return found
    if not path.exists():
        raise SystemExit(f"输入不存在: {path}")
    return [path]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="布局合法化 (去重叠) + 双硬守卫下的线长优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="placement JSON 文件或目录 (目录则递归)")
    parser.add_argument("--output-dir", default=None,
                        help="默认与输入 JSON **同目录** (产物名后缀 _legal, 不会被下一轮当成输入)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 配置路径")
    parser.add_argument("--device", default=None, help="cuda / cpu, 默认自动")
    parser.add_argument("--set", action="append", default=[], help="覆盖配置, 形如 --set refine.steps=400")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config, args.set)

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    inputs = collect_inputs(args.input)
    if not inputs:
        raise SystemExit("没有找到输入 JSON")

    # 默认写到**每个输入 JSON 自己所在的目录** —— 合法布局和对比图跟原文件放一起,
    # 下游按目录取布局时不用改路径。每个 case 的 out_dir 在循环里单独算,
    # 因为一次批跑可能横跨多个目录。
    fixed_out_dir = Path(args.output_dir) if args.output_dir else None
    if fixed_out_dir is not None:
        fixed_out_dir.mkdir(parents=True, exist_ok=True)

    repair_on = bool(_get(config, "repair.enabled", True))
    refine_on = bool(_get(config, "refine.enabled", True))
    evaluators = build_evaluators(config, device) if (repair_on or refine_on) else None

    print(
        f"[legalizeLayout] {len(inputs)} 个 case, device={device}, "
        f"repair={'on' if repair_on else 'off'} refine={'on' if refine_on else 'off'}"
    )
    print(
        "[legalizeLayout] 输出目录: "
        + (str(fixed_out_dir) if fixed_out_dir is not None else "各输入 JSON 所在目录 (后缀 _legal)")
    )

    reports, failures = [], []
    for i, path in enumerate(inputs, 1):
        out_dir = fixed_out_dir if fixed_out_dir is not None else Path(path).parent
        try:
            report = process_one(path, config, evaluators, out_dir, device, verbose=args.verbose)
        except Exception as exc:  # 单个 case 失败不该中断整批
            failures.append((str(path), repr(exc)))
            print(f"  [{i}/{len(inputs)}] {path.name} 失败: {exc!r}")
            continue

        reports.append(report)
        before, after = report["before"], report["after"]
        audit = report["guard_audit"]
        # 标记按**最终**结果打, 不按 Stage A 的中间状态 —— Stage B 会把线长还回来。
        # "出容忍带"= 超出了模型自身的精度, 那才是真劣化; 带内的波动不算。
        flag = ""
        if not audit["within_thermal_tolerance"]:
            flag += " [温度出容错带]"
        if not audit["within_wl_tolerance"]:
            flag += " [线长出容错带]"
        if audit.get("hotspot_degraded_beyond_tol"):
            flag += " [热点劣化超0.5K]"
        print(
            f"  [{i}/{len(inputs)}] {Path(path).stem}: "
            f"重叠 {before['overlap_pairs']}->{after['overlap_pairs']}, "
            f"越界 {before['out_of_canvas']}->{after['out_of_canvas']}, "
            f"合法率 {before['legal_ratio']:.4f}->{after['legal_ratio']:.4f}, "
            f"ΔT {_fmt(audit['peak_temp_delta_C'], 'K')}, "
            f"ΔWL {_fmt(audit['wirelength_neural_delta_rel'], '', pct=True)}, "
            f"Δbbox {_fmt(audit['bbox_area_delta_rel'], '', pct=True)}, "
            f"ΔT_热点 {_fmt(audit.get('hotspot_region_delta_C'), 'K')}, "
            f"位移 {report['max_displacement_mm']:.3f}mm, "
            f"{report['timing_s']['total']:.1f}s{flag}"
        )
        hotspot = report.get("hotspot") or {}
        if hotspot.get("indices"):
            frozen = bool(_get(config, "refine.hotspot.freeze", True))
            print(
                f"        热点 chiplet ({'已冻结' if frozen else '只守卫不冻结'}): "
                f"{', '.join(hotspot['names'])}  "
                f"阈值 {hotspot['threshold_C']:.1f}°C, 区温 "
                f"{hotspot.get('input_region_C', float('nan')):.1f}->"
                f"{hotspot.get('after_region_C', float('nan')):.1f}°C "
                f"({_fmt(hotspot.get('region_delta_k'), 'K')})"
            )
        if args.verbose and report.get("refine", {}).get("enabled"):
            print(f"        refine: {format_refine_stats(report['refine'])}")

    if not reports:
        raise SystemExit("全部 case 都失败了")

    total_accepted = sum(r["refine"].get("accepted", 0) for r in reports)
    total_thermal = sum(r["refine"].get("thermal_rejected", 0) for r in reports)
    total_wl = sum(r["refine"].get("wl_rejected", 0) for r in reports)
    illegal = [Path(r["input"]).stem for r in reports if not r["after"]["is_legal"]]

    print("\n[汇总]")
    print(f"  成功 {len(reports)}/{len(inputs)}" + (f", 失败 {len(failures)}" if failures else ""))
    print(f"  合法化: {len(reports) - len(illegal)}/{len(reports)} 个 case 完全合法")
    if illegal:
        print(f"  !! 仍未合法: {', '.join(illegal)}")
    print(
        f"  Stage B 计数器: accepted={total_accepted}, "
        f"thermal_rejected={total_thermal}, wl_rejected={total_wl}"
    )

    # 三条指标的净变化。用神经模型的读数 (权威), 不用 HPWL 代理。
    def tally(key, rel=True, tol=0.0):
        better = worse = same = 0
        for r in reports:
            value = _delta(r, key, relative=rel)
            if value is None:
                continue
            if value < -tol:
                better += 1
            elif value > tol:
                worse += 1
            else:
                same += 1
        return better, same, worse

    thermal_tol = float(_get(config, "repair.guard.thermal_tol_k", 0.0))
    wl_tol = float(_get(config, "repair.guard.wl_tol", 0.0))
    print(f"  净变化 (神经模型读数, 优于 / 持平 / 劣于; 容错带 {thermal_tol:.2f}K / {wl_tol * 100:.2f}%):")
    for label, key, tol in (
        ("峰值温度", "peak_temp_C", thermal_tol),
        ("神经线长", "wirelength_neural_mm", wl_tol),
        ("外接框  ", "bbox_area_mm2", 0.0),
    ):
        better, same, worse = tally(key, rel=(key != "peak_temp_C"), tol=tol)
        print(f"    {label}: {better} / {same} / {worse}")

    hotspot_cases = [r for r in reports if (r.get("hotspot") or {}).get("indices")]
    if hotspot_cases:
        degraded = [
            Path(r["input"]).stem
            for r in hotspot_cases
            if r["hotspot"].get("degraded_beyond_tol")
        ]
        frozen_moved = [
            Path(r["input"]).stem
            for r in hotspot_cases
            if r["guard_audit"].get("frozen_stayed_put") is False
        ]
        frozen_on = bool(_get(config, "refine.hotspot.freeze", True))
        print(
            f"  热点定位: {len(hotspot_cases)}/{len(reports)} 个 case 定出热点件"
            f" ({'已冻结' if frozen_on else '只守卫不冻结'}); "
            f"热点区升温超 {thermal_tol:.2f}K 的: {len(degraded)} 个"
            + (f" ({', '.join(degraded)})" if degraded else "")
        )
        if frozen_moved:
            print(f"  !! 冻结件发生了位移 (不该发生): {', '.join(frozen_moved)}")

    out_of_band = [
        Path(r["input"]).stem
        for r in reports
        if not (r["guard_audit"]["within_thermal_tolerance"] and r["guard_audit"]["within_wl_tolerance"])
    ]
    if out_of_band:
        print(f"  !! 最后仍超出模型容错带的 case: {', '.join(out_of_band)}")
        print("     逐 case 的原因在报告的 guard_audit 与 repair.policies_tried 里")
    else:
        print("  两条指标全部落在模型容错带内")

    if total_thermal + total_wl > 20 * max(total_accepted, 1):
        print(
            "  提示: 守卫拒绝数远超接受数, 说明两条容错带把可行域压得很窄。\n"
            "        带宽已经是模型自身的精度了, 再放就等于接受模型分辨不出的变化。"
        )
    for path, error in failures:
        print(f"  !! {Path(path).name}: {error}")
    return 0 if not failures and not illegal else 1


def _fmt(value, unit, pct=False):
    if value is None:
        return "n/a"
    return f"{value * 100:+.2f}%" if pct else f"{value:+.2f}{unit}"


if __name__ == "__main__":
    sys.exit(main())
