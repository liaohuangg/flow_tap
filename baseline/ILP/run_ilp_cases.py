#!/usr/bin/env python3
"""run_ilp_cases.py — ILP 布局基线的 driver: cases x configs x 目标。

两种目标 (--objective):

  wl    只优化线长。面积(画布)是软约束。产物 -> resultEval/ILP_result/
  bbox  优化外接框 W+H (权重调大) + 长宽比 |W-H| (权重调小)。
        产物 -> resultEval/ILP_bbox_result/

流程 (每个 case x config):
    1. 建 A, 解析画布 S (软约束上限), 挂上 clump 常数/pmax
    2. λ 初值 (uniform 或 用 greedy 解的弧流量)   [bbox 目标不需要 λ]
    3. wl 目标: 外层逐次线性化, K 轮:
           MILP(λ) -> 分 3 段求解 (找可行解 / 平衡 / 压界)
           用**精确** reference_cost 评估, 按真实线长保留历史最优
           λ <- damp * 新弧流量 + (1-damp) * 旧 λ
       bbox 目标: 单次 MILP (K=1), 没有需要线性化的双线性项
    4. 把最优解写成契约格式 -> <out>/format_result/<case>_seed<k>.json
       同时写 raw/<case>_seed<k>.json 和 run_log.csv 一行

**MILP 是确定性的**, 所以这里的 seed=1..5 不是随机种子, 而是 5 个固定、写明的
确定性配置 (见 OBJECTIVES[...]['configs'])。下游 _split_stem / compare_*.csv 不用改。

用法:
    python run_ilp_cases.py --cases hp6_m --seeds 1 --profile quick
    python run_ilp_cases.py --all --objective bbox --profile quick --workers 12
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ilp_core as C  # noqa: E402
import wl_oracle as O  # noqa: E402
from greedy_init import shelf_layout  # noqa: E402

PROJECT = C.PROJECT

# 明确排除 Case6/Case8/Case9/Case10 (20/36/44/61 芯粒, 规模太大, 用户指定不跑)
ALL_CASES = ["Case7", "acend910", "cpu-dram", "hp11_m", "hp6_m", "hp8_m",
             "multigpu", "syn1", "syn4", "xerox6_m", "xerox7_m", "xerox8_m"]

# 时间预算 —— **是上限, 不是配额**。
#
# 收敛阈值是 gap=0.05 (见 OBJECTIVES[*]["gap"]), 求解器一旦达标就立刻返回,
# 所以小 case 根本花不到这些秒数。这里的数只决定"难 case 最多烧多久"。
#
# 早先那套 (60/180/900 和 300/900/5400) 是按"分段阈值到点就停"设计的,
# 换成 gap=0.05 之后明显不够: 刚化画布让可行域窄了很多, quick 预算下实测
# hp11_m gap 0.312 / syn1 0.194 / syn4 0.259, 离 0.05 差一个数量级。
#
# 总预算是**整次求解**的量, wl 目标还要按轮数 K 再平分 (见 solve_one)。
BUDGETS = {
    "quick": [(8, 600.0), (13, 1200.0), (28, 3600.0)],       # 冒烟用
    "full":  [(8, 1800.0), (13, 3600.0), (28, 10800.0)],     # 正式数字
}

# MILP 是确定性的, 报多个"随机种子"是假的。用户已明确: **每个 case 只跑一次**。
# 文件名仍是 <case>_seed1.json, 这样下游 _split_stem / compare_*.csv 零改动。
SEED = 1

OBJECTIVES = {
    # ---- 只优化线长 (面积不参与目标, 但画布是**硬约束**) ----------------------
    "wl": {
        "method": "ILP",
        "out_dir": PROJECT / "resultEval" / "ILP_result",
        "raw_dir": HERE / "raw",
        "run_log": HERE / "run_log.csv",
        "gap": 0.05,
        "configs": {
            1: {"name": "wl", "rot": True, "K": 3, "damp": 0.8,
                "uniform_init": True, "hard_canvas": True},
        },
    },
    # ---- 优化 W+H (主) + 长宽比 |W-H| (次) ----------------------------------
    # w_bbox 调大、w_aspect 调小 —— 就是用户指定的"在优化 W+H 的前提下优化长宽比"。
    # w_wl 是极小的平局打破项 (1%), 不参与 W+H / 长宽比 的量级。
    "bbox": {
        "method": "ILP_bbox",
        "out_dir": PROJECT / "resultEval" / "ILP_bbox_result",
        "raw_dir": HERE / "raw_bbox",
        "run_log": HERE / "run_log_bbox.csv",
        "gap": 0.05,
        "configs": {
            1: {"name": "bbox", "rot": True, "K": 1, "uniform_init": True,
                "w_bbox": 1.0, "w_aspect": 0.1, "w_wl": 0.01, "hard_canvas": True},
        },
    },
}


# 墙钟上限下剩得比这还少就别再开一段了 —— 一段还没热身就到点, 只会在落盘前
# 白等几十秒, 不如直接把手里最好的解导出去。
MIN_STAGE_S = 30.0


def budget_for(n: int, profile: str, cap: float | None = None) -> float:
    """按规模取预算; cap 给了就再取 min —— 上限语义, 只收不放。"""
    base = BUDGETS[profile][-1][1]
    for limit, secs in BUDGETS[profile]:
        if n <= limit:
            base = secs
            break
    return base if cap is None else min(base, cap)


def clip_stages(stages: list[tuple[float, float, int]], remain: float):
    """把各段时间上限按比例缩到总和 <= remain。

    给 `--budget-cap` 用。光调小总预算不够: 每轮之后还要重建模型 + 解一次参考
    布线 ILP 拿真实代价, 这段开销不在 Gurobi 的 TimeLimit 里, 轮数一多就会
    把总墙钟顶出用户要的一小时。所以每轮开跑前按**剩余墙钟**再收一次。
    """
    total = sum(s[0] for s in stages)
    if remain >= total:
        return stages
    f = max(remain, 1.0) / total
    return [(max(1.0, t * f), g, focus) for t, g, focus in stages]


def stage_plan(total_s: float, gap: float) -> list[tuple[float, float, int]]:
    """两段: 先宽松阈值快速抓一个好可行解, 再按用户指定的 gap 收敛。

    收敛阈值是用户指定的 **gap = 0.05**, 两个目标一致。

    为什么留第一段: 直接一段跑紧 gap 时, 求解器常常在"找可行解"上耗掉大半预算
    (刚化画布 + 4 二值不重叠的模型可行域很窄)。MIPFocus=1 让第一段优先找好解,
    第二段从它 warm start, 总预算不变。

    注意 gap 的含义随目标而变:
      wl   -> 间隙在**线性化代理目标** Σλ·d 上, 不是真实线长的最优性证明。
      bbox -> 目标里的 maxX/maxY/aspect 都是精确变量, gap 是真的 (但 0.05 意味着
              W+H 不再"证明最优", 报出去的数是上界)。
    """
    return [(0.35 * total_s, max(0.15, gap), 1),
            (0.65 * total_s, gap, 0)]


def clear_stale_cache(out_dir: Path, stem: str) -> None:
    """改写同一个 stem 的布局后, eval_layout 会静默复用旧温度和旧中间文件。"""
    ev = out_dir / "eval_out" / stem
    if ev.is_dir():
        shutil.rmtree(ev, ignore_errors=True)
    cache = out_dir / "eval_cache.json"
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if stem in data:
            data.pop(stem)
            cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 单个 (case, seed)
# --------------------------------------------------------------------------- #
def solve_one(case: str, seed: int, profile: str, objective: str,
              verbose: bool = False, budget_cap: float | None = None) -> dict:
    spec = OBJECTIVES[objective]
    cfg = spec["configs"][seed]
    t_start = time.time()

    record = C.load_case(case)
    A = C.case_arrays(record)
    S, canvas_src = C.canvas_mm(case, record["chiplets"])

    problems = C.check_case(A)
    if problems:
        return {"case": case, "seed": seed, "config": cfg["name"], "status": "SKIPPED",
                "error": "; ".join(problems), "n": A["n"]}

    O.attach_geometry(A)
    n = A["n"]

    # bbox 目标没有需要线性化的双线性项, 一轮就够; wl 目标要多轮更新 λ
    n_iter = 1 if objective == "bbox" else cfg["K"]

    # budget 是**整次求解**的总量, 按轮数平分。
    # (早期是每轮各拿一份, K=8 的 seed5 在 Case7/full 下就是 8x5400s = 12 小时一个任务,
    #  而且各 config 的总耗时差好几倍、互相没法比。)
    budget = budget_for(n, profile, budget_cap)
    budget_per_iter = budget / n_iter
    stages = stage_plan(budget_per_iter, spec["gap"])

    warm = shelf_layout(A, S, allow_rot=cfg["rot"])
    warm_rec = _record_of(A, warm, case, seed)
    ref = O.reference_cost(warm_rec)
    wl_ref = ref["total"] or 1.0

    # 起点先用 greedy 解垫底, 保证任何时候都有一个可行候选
    best_x = warm
    best_key = _rank_key(objective, {}, C.bbox_mm(A, warm)[:2], spec["gap"])
    best_obj = None
    best_stage = "greedy_init"
    best_gap = None      # 胜出那一段的 mip_gap; None = 落回 greedy 起点, 没有界可报
    best_bound = None

    # 只有 bbox 的配置带 w_wl 键; wl 目标必然是纯线长, 缺省按 1.0 (开着) 算。
    w_wl = cfg.get("w_wl", 1.0)
    lam = O.arc_flow(A, warm, uniform=cfg["uniform_init"]) if w_wl > 0 else {}
    history = []
    first_stats = None

    for it in range(n_iter):
        # 有墙钟上限时: 迟了就收工, 让 best_x (当前最好的解) 照常落盘。
        # 用户要的是"到点导出解", 不是"到点什么都不剩" —— 外面 kill 进程
        # 拿不到任何东西, 因为落盘在 solve_one 返回之后才发生。
        remain = None
        if budget_cap is not None:
            remain = budget_cap - (time.time() - t_start)
            if remain <= MIN_STAGE_S:
                if verbose:
                    print(f"    墙钟到点 (剩 {remain:.0f}s), 停在 iter{it}, 导出当前最好解")
                break
        model = C.IlpModel(A, S, lam, allow_rot=cfg["rot"], objective=objective,
                           w_bbox=cfg.get("w_bbox", 0.0),
                           w_aspect=cfg.get("w_aspect", 0.0),
                           w_wl=cfg.get("w_wl"), wl_ref=wl_ref,
                           hard_canvas=cfg.get("hard_canvas", False))
        if it == 0:
            first_stats = model.stats()
        res = model.solve(stages if remain is None else clip_stages(stages, remain),
                          warm_start=best_x)

        # 每段各出一个解, 逐段按 _rank_key 挑 (证书优先, 见 _rank_key 的说明)
        for rec in res["stages"]:
            if rec["x"] is None:
                continue
            x = C.separate_rounding(A, rec["x"])
            if C.footprints_overlap(A, x):
                history.append({"iter": it, "stage": rec["status_name"],
                                "note": "跳过: 取整后仍有重叠"})
                continue
            ev = O.reference_cost(_record_of(A, x, case, seed))
            if ev["total"] is None:
                history.append({"iter": it, "stage": rec["status_name"],
                                "note": "跳过: 参考求解不可行"})
                continue
            bw, bh, barea = C.bbox_mm(A, x)
            key = _rank_key(objective, rec, (bw, bh), spec["gap"])
            history.append({"iter": it, "stage": rec["status_name"],
                            "status": rec["status_name"], "mip_gap": rec.get("mip_gap"),
                            "obj_model": rec.get("obj_model"), "obj_eval": ev["total"],
                            "bbox_wh": round(bw + bh, 4), "aspect": round(abs(bw - bh), 4)})
            # 同一个解可能在多段里被重复找到: 末段 (MIPGap 更小) 会再报一次同样的
            # 目标值, 但那一份带着**更强的证书**。仅当 key 严格更小才换布局;
            # key 完全打平时 (两段报出同一个目标值) 只把更强的证书记下来,
            # 否则 proven_optimal 会漏报。
            gap = rec.get("mip_gap")
            better = key < best_key
            tie_with_better_proof = (key == best_key and gap is not None
                                     and (best_gap is None or gap < best_gap))
            if better or tie_with_better_proof:
                best_x, best_obj, best_key = x, rec.get("obj_model"), key
                best_stage = f"iter{it}/{rec['status_name']}"
                best_gap = gap
                best_bound = rec.get("best_bound")

        # λ 更新: 用当前最好解处的弧流量, 阻尼混合
        if w_wl > 0 and objective == "wl":
            new_lam = O.arc_flow(A, best_x)
            if new_lam:
                damp = cfg["damp"]
                merged = {}
                for k in set(lam) | set(new_lam):
                    old, new = lam.get(k), new_lam.get(k)
                    if old is None:
                        merged[k] = new
                    elif new is None:
                        merged[k] = old
                    else:
                        merged[k] = [[damp * new[h][kk] + (1 - damp) * old[h][kk]
                                      for kk in range(4)] for h in range(4)]
                lam = merged

        if verbose:
            print(f"    iter{it}: best={tuple(round(v, 4) for v in best_key)} ({best_stage})")

    # 落盘
    stem = f"{case}_seed{seed}"
    out_dir = spec["out_dir"]
    clear_stale_cache(out_dir, stem)
    format_dir = out_dir / "format_result"
    final_record = C.write_record(A, best_x, case, seed, format_dir / f"{stem}.json")
    C.verify_record(A, final_record)

    over_x, over_y = C.canvas_overflow(A, best_x, S)
    bw, bh, barea = C.bbox_mm(A, best_x)
    final_wl = O.reference_cost(final_record)["total"]
    raw = {
        "case": case, "seed": seed, "config": cfg["name"], "profile": profile,
        "objective": objective, "n_chiplets": n,
        "canvas_mm": S, "canvas_source": canvas_src,
        "budget_s_total": budget, "wall_s": round(time.time() - t_start, 2),
        "best_stage": best_stage, "obj_model": best_obj,
        # 胜出那一段的 MIP 界。bbox 目标末段 MIPGap=0 -> gap=0.0 就是**证明最优**;
        # 落回 greedy_init 时是 None (没有界可报, 别把它读成 0)。
        "mip_gap": best_gap, "best_bound": best_bound,
        "proven_optimal": best_gap == 0.0,
        "obj_eval_total_wl": final_wl, "greedy_init_total_wl": ref["total"],
        # 软约束的越界量: 两个都是 0 -> 画布没被触发, 软约束等价于硬约束
        "canvas_overflow_mm": [round(over_x, 6), round(over_y, 6)],
        "bbox_mm": [round(bw, 4), round(bh, 4)], "bbox_area_mm2": round(barea, 4),
        "bbox_wh_mm": round(bw + bh, 4), "aspect_absdiff_mm": round(abs(bw - bh), 4),
        "n_footprint_overlaps": len(C.footprints_overlap(A, best_x)),
        "model": first_stats, "config_detail": cfg,
        "history": history,
        "x_footprint_ll": [{"name": A["names"][i], "FX": round(best_x[i][0], 6),
                            "FY": round(best_x[i][1], 6), "r": best_x[i][2]}
                           for i in range(n)],
    }
    raw_dir = spec["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{stem}.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"case": case, "seed": seed, "config": cfg["name"], "status": "OK",
            "n": n, "obj_model": best_obj, "obj_eval": final_wl,
            "greedy": ref["total"], "wall_s": round(time.time() - t_start, 2),
            "canvas_mm": S, "canvas_source": canvas_src, "best_stage": best_stage,
            "mip_gap": best_gap, "proven_optimal": best_gap == 0.0,
            "bbox_wh": round(bw + bh, 4), "bbox_area": round(barea, 4),
            "aspect": round(abs(bw - bh), 4),
            "overflow": round(max(over_x, over_y), 4),
            "n_bin": first_stats["n_bin"], "n_constrs": first_stats["n_constrs"]}


def _rank_key(objective: str, rec: dict, bbox_wh: tuple[float, float],
              gap_target: float) -> tuple:
    """挑最优解的排序键 (越小越好)。**一律用模型自己的量, 不碰真实线长。**

    真实线长 (obj_eval) 要另解一个 TAP-2.5D 布线 ILP 才知道, 是**外部度量** ——
    没有任何一轮 MILP 在优化它。拿它当选择依据, 挑出来的解属于一个目标, 而它带着的
    gap 又描述另一个目标, 报出去自相矛盾。实测代价很大: xerox6_m 的 iter2 已经把
    代理目标**证到 gap 0**, 只因为真实线长比 iter0 那份多一点点就被丢掉, 一直报 0.05。

    两个目标都把**证书** (gap 是否 <= gap_target) 排在前面: gap 没达标的解, 模型
    自己都没收敛, 它报的目标值不可信 —— 先在达标的解里挑, 一个都没有再退回去。

      wl  : (证书, obj_model, gap)        几何不进 key —— 线长目标不看 bbox
      bbox: (W+H, |W-H|, 证书, obj_model) 几何是目标本身, 排在证书前面

    `rec` 是 `IlpModel.solve` 返回的单段记录; 传 {} 表示没有段记录 (greedy 兜底),
    此时一律算没达标, 排在所有 MILP 解后面。
    """
    gap = rec.get("mip_gap")
    miss = 0 if (gap is not None and gap <= gap_target) else 1
    obj = rec.get("obj_model")
    obj = float("inf") if obj is None else obj
    if objective == "wl":
        return (miss, obj, float("inf") if gap is None else gap)
    bw, bh = bbox_wh
    return (round(bw + bh, 6), round(abs(bw - bh), 6), miss, obj)


def _record_of(A: dict, x, case: str, seed: int) -> dict:
    """把 (FX,FY,r) 变成契约记录 (不落盘), 供参考求解器调用。"""
    chiplets = []
    for i, (fx, fy, r) in enumerate(x):
        w, h, u = A["w"][i], A["h"][i], A["u"][i]
        pw, ph = C.placed_dims(w, h, r)
        chiplets.append({"name": A["names"][i], "x-position": round(fx + u, 6),
                         "y-position": round(fy + u, 6), "width": round(pw, 6),
                         "height": round(ph, 6), "rotation": int(r),
                         "power": A["power"][i], "hubump": u})
    return {"system_id": f"{case}_seed{seed}", "chiplets": chiplets,
            "connections": [{"node1": A["names"][i], "node2": A["names"][j],
                             "wireCount": wc} for i, j, wc in A["conns"]]}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=None, help="case 名列表")
    ap.add_argument("--all", action="store_true", help=f"全部 {len(ALL_CASES)} 个 case")
    ap.add_argument("--seeds", nargs="*", type=int, default=[SEED],
                    help=f"只有 {SEED}。MILP 是确定性的, 每个 case 只跑一次; "
                         f"保留这个参数只为下游文件名 <case>_seed<N> 的兼容")
    ap.add_argument("--objective", default="wl", choices=sorted(OBJECTIVES),
                    help="wl=只优化线长(默认); bbox=优化 W+H + 长宽比")
    ap.add_argument("--profile", default="quick", choices=["quick", "full"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget-cap", type=float, default=None, metavar="SEC",
                    help="每个任务的墙钟上限 (秒)。到点就停止求解并导出当前最好解 "
                         "(不是 kill —— kill 拿不到任何东西, 落盘在 solve_one 返回之后)。"
                         "与 profile 预算取 min, 只收不放。")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    spec = OBJECTIVES[args.objective]
    cases = ALL_CASES if args.all else (args.cases or ["hp6_m"])
    for case in cases:
        if case not in ALL_CASES:
            raise SystemExit(f"未知 case {case!r}; 可选: {' '.join(ALL_CASES)}")
    for s in args.seeds:
        if s not in spec["configs"]:
            raise SystemExit(f"seed 必须是 {sorted(spec['configs'])} 之一")

    format_dir = spec["out_dir"] / "format_result"
    format_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s) for c in cases for s in args.seeds]
    print(f"method={spec['method']}  objective={args.objective}  "
          f"{len(jobs)} 个任务 (case x config), profile={args.profile}, workers={args.workers}")
    print(f"-> {format_dir}")

    t0 = time.time()
    results = []
    if args.workers > 1:
        from multiprocessing import Pool
        # imap_unordered 而不是 starmap: 大 case 要跑很久, 结果得一个一个吐出来,
        # 不然 log 要等到整批结束才有内容。
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(
                    _worker, [(c, s, args.profile, args.objective, args.verbose,
                               args.budget_cap)
                              for c, s in jobs]):
                results.append(r)
                _echo(r)
    else:
        for c, s in jobs:
            r = solve_one(c, s, args.profile, args.objective, args.verbose,
                          args.budget_cap)
            results.append(r)
            _echo(r)

    run_log = spec["run_log"]
    header = not run_log.is_file()
    with run_log.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if header:
            w.writerow(["case", "seed", "config", "objective", "profile", "status", "n",
                        "obj_model", "obj_eval_total_wl", "greedy_total_wl",
                        "bbox_wh_mm", "bbox_area_mm2", "aspect_absdiff_mm",
                        "canvas_overflow_mm", "wall_s", "canvas_mm", "best_stage",
                        "n_bin", "n_constrs", "error"])
        for r in results:
            w.writerow([r.get("case"), r.get("seed"), r.get("config"), args.objective,
                        args.profile, r.get("status"), r.get("n"), r.get("obj_model"),
                        r.get("obj_eval"), r.get("greedy"), r.get("bbox_wh"),
                        r.get("bbox_area"), r.get("aspect"), r.get("overflow"),
                        r.get("wall_s"), r.get("canvas_mm"), r.get("best_stage"),
                        r.get("n_bin"), r.get("n_constrs"), r.get("error", "")])

    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n=== 结束: {ok}/{len(results)} 成功, {time.time() - t0:.0f}s ===")
    print(f"run_log: {run_log}")
    return 0 if ok == len(results) else 1


def _worker(job):
    """imap_unordered 只把整个 job 元组当**一个**参数传进来 (不是 starmap 的拆包),
    所以这里必须收元组再自己解 —— 否则第一个结果回来时整批就 TypeError 崩掉,
    而且因为崩溃发生在 pool 的迭代里, 前面已经算好的布局全部拿不到。"""
    case, seed, profile, objective, verbose, budget_cap = job
    try:
        return solve_one(case, seed, profile, objective, verbose, budget_cap)
    except Exception as exc:  # noqa: BLE001 —— 一个任务炸了不该带走整批
        import traceback
        traceback.print_exc()
        return {"case": case, "seed": seed,
                "config": OBJECTIVES[objective]["configs"][seed]["name"],
                "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}


def _echo(r: dict) -> None:
    if r["status"] == "OK":
        tag = f"线长 {r['obj_eval']:.1f}" if r.get("obj_eval") else ""
        if r.get("bbox_wh"):
            tag += f"  W+H {r['bbox_wh']:.2f}  |W-H| {r['aspect']:.2f}"
        if r.get("overflow"):
            tag += f"  [画布超 {r['overflow']:.3f}mm]"
        print(f"[ok  ] {r['case']:10s} seed{r['seed']} {r['config']:9s} "
              f"n={r['n']:3d} {tag}  {r['wall_s']:.0f}s", flush=True)
    else:
        print(f"[{r['status'].lower():5s}] {r['case']:10s} seed{r['seed']}: {r.get('error')}",
              flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
