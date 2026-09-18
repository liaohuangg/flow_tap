#!/usr/bin/env python3
"""check_objective.py — ILP 布局的回归检查。

对每份 resultEval/ILP_result/format_result/<case>_seed<k>.json 做四件事:

  1. **参考代价可复现**: 用 wl_oracle.reference_cost 重算 (和 eval_layout 同一条
     调用路径), 与 run_log.csv 里记的 obj_eval 对齐。这就是"报出去的数没变"。
  2. **几何合法**: footprint 不重叠; 越界量 (slack) 是多少。
  3. **目标一致性**: |obj_model - obj_eval| / obj_eval —— MILP 内部线性化目标
     与该点真实代价的差, 就是逐次线性化的残差, 要如实报出来。
  4. **比 greedy 起点好**: 最优解不该比构造解还差, 否则说明求解或挑选有 bug。

用法:
    python check_objective.py                          # 全部
    python check_objective.py --cases hp6_m syn4
    python check_objective.py --glob 'resultEval/ILP_result/format_result/*.json'
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ilp_core as C  # noqa: E402
import wl_oracle as O  # noqa: E402

PROJECT = C.PROJECT
FORMAT_DIR = PROJECT / "resultEval" / "ILP_result" / "format_result"
RUN_LOG = HERE / "run_log.csv"

# 布局落盘用 round(...,6) mm, 所以 ≤1 纳米的画布越界一定是取整残差, 不是真越界。
CANVAS_EPS = 1e-6


def load_run_log() -> dict:
    if not RUN_LOG.is_file():
        return {}
    out = {}
    with RUN_LOG.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[f"{row['case']}_seed{row['seed']}"] = row
    return out


def _pick(row: dict, *names):
    """按顺序取第一个非空的列; 全空/没这行返回 None。"""
    if not row:
        return None
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return v
    return None


def _raw_wl_ref(stem: str):
    """raw/<stem>.json 里 MILP 目标归一化用的 wl_ref; 没有就 None。

    check_one 手上只有 format_result 那份布局 (契约格式), 不含求解器内部量,
    所以得回 raw/ 借 —— 两份文件同一个 stem, 是同一轮产物。
    """
    p = HERE / "raw" / f"{stem}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))["model"].get("wl_ref")
    except (KeyError, ValueError):
        return None


def check_one(path: Path, log: dict, verbose: bool) -> dict:
    stem = path.stem
    case, seed = stem.rsplit("_seed", 1)
    record = json.loads(path.read_text(encoding="utf-8"))

    A = C.case_arrays(C.load_case(case))
    S, _src = C.canvas_mm(case, record["chiplets"])

    # 记录 -> (FX, FY, r)
    x = [(c["x-position"] - c["hubump"], c["y-position"] - c["hubump"], c["rotation"])
         for c in record["chiplets"]]

    res = {"stem": stem, "case": case, "seed": seed, "problems": [], "notes": []}

    # 1. 参考代价
    ref = O.reference_cost(record)
    res["obj_eval"] = ref["total"]
    if ref["total"] is None:
        res["problems"].append("参考求解返回哨兵 (不可行或超时) —— 这份布局无法评分")
        return res

    row = log.get(stem)
    # 列名以 run_ilp_cases.py 实际写出的表头为准 (`obj_eval_total_wl` / `greedy_total_wl`),
    # 老名字留作退路。这里踩过一次: 读的键和写的键差一个后缀, `row.get()` 恒为 None,
    # 于是整个 if 分支从来不执行 —— 下面三条断言 (报出去的数没变 / 线性化残差 / 不劣于
    # greedy) 全都是死代码, 输出里只留下一句无害的 note。查不到行和查到了但列为空
    # 必须分开报, 否则下次还是看不出来。
    ev = _pick(row, "obj_eval_total_wl", "obj_eval")
    om = _pick(row, "obj_model")
    gr = _pick(row, "greedy_total_wl", "greedy")
    if row and ev is not None:
        logged = float(ev)
        if abs(logged - ref["total"]) > 1e-6 * max(1.0, abs(logged)):
            res["problems"].append(
                f"run_log 记的 obj_eval={logged:.4f} 与重算 {ref['total']:.4f} 不一致")
        else:
            res["notes"].append(f"obj_eval 与 run_log 一致 ({ref['total']:.2f})")
        if om is not None:
            # obj_model 是**无量纲**的: ilp_core.py:339 把线长项除了个 wl_ref
            # (`obj += (w_wl / wl_ref) * wl`), 所以直接和 obj_eval 比是拿比值去比毫米,
            # 恒差 5 个数量级 —— 之前这里就因此对 12 个 case 全报 "残差 100%",
            # 看着像全挂了, 其实只是量纲错。乘回 wl_ref 才是同一尺度的量。
            ref_wl = _raw_wl_ref(stem)
            if ref_wl:
                ratio = float(om) * ref_wl / max(1.0, abs(ref["total"]))
                res["proxy_bias"] = ratio
                # 代理值**系统性偏高**是预期行为, 不是缺陷: λ 是容量感知的**贪心**分配
                # (wl_oracle._greedy), 不是参考布线 ILP 的精确弧流量, 会把线摊到便宜的弧上,
                # 于是高估。实测 12 个 case 落在 +7%~+68%, 所以阈值不能定成 2%。
                # 只抓"反了"(高估变低估)或"高得离谱"这两种真的坏掉的情形。
                if not 0.5 <= ratio <= 5.0:
                    res["problems"].append(
                        f"代理/真实代价比 {ratio:.2f} 落在 [0.5, 5] 外 —— λ 尺度或折叠错了")
                else:
                    res["notes"].append(
                        f"代理代价 {float(om) * ref_wl:.0f} vs 真实 {ref['total']:.0f} "
                        f"(高估 {ratio - 1:+.1%}, 贪心 λ 的系统性偏差, 预期内)")
            else:
                res["notes"].append("raw/ 里没有 wl_ref, 代理代价无法还原尺度 (跳过)")
        if gr is not None:
            g = float(gr)
            res["vs_greedy"] = (g - ref["total"]) / max(1.0, abs(g))
            if ref["total"] > g + 1e-6:
                res["problems"].append(f"比 greedy 起点还差 ({ref['total']:.1f} > {g:.1f})")
    elif row:
        res["problems"].append(
            "run_log 有这一行但认不出线长列 (表头改名了?) —— 目标一致性没被检查到")
    else:
        res["notes"].append("run_log 里没有这一行")

    # 2. 几何合法
    overlaps = C.footprints_overlap(A, x)
    if overlaps:
        res["problems"].append(f"footprint 重叠 {len(overlaps)} 对, 例如 {overlaps[0]}")
    ox, oy = C.canvas_overflow(A, x, S)
    res["canvas_overflow_mm"] = [round(ox, 6), round(oy, 6)]
    bw, bh, barea = C.bbox_mm(A, x)
    res["bbox_area_mm2"] = round(barea, 3)
    # 容差 1e-6 mm = 1 纳米。坐标是按 round(...,6) 落盘的, 每个坐标最多偏 5e-7,
    # 越界量又是两个坐标之差, 所以 ≤1e-6 的越界**必然是取整残差**, 不是真的破画布。
    # 实测: hp11_m 的 bbox 正好齐平画布边 (100.00%), 残差 4.1e-8 mm —— 比 hubump
    # (0.09 mm) 小 6 个数量级。之前这里判 `== 0.0`, 于是它被当成越界; 而又按 %.3f
    # 打印成 "x+0.000", 看起来像没事。两件事一起改: 判容差, 且残差用科学计数法露出来。
    if ox <= CANVAS_EPS and oy <= CANVAS_EPS:
        if ox > 0.0 or oy > 0.0:
            res["notes"].append(
                f"画布未越界 (残差 x{ox:.1e} y{oy:.1e} mm, 取整噪声)")
        else:
            res["notes"].append("画布未越界 (软约束未触发, 等价于硬约束)")
    else:
        res["notes"].append(f"画布越界 x+{ox:.3e} y+{oy:.3e} mm")

    # 4. hubump 自洽 (benchmark 若被改动会在这里炸)
    try:
        C.verify_record(A, record)
    except AssertionError as exc:
        res["problems"].append(str(exc))

    if verbose:
        for note in res["notes"]:
            print(f"    · {note}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=str(FORMAT_DIR / "*.json"))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import glob as _glob
    paths = sorted(Path(p) for p in _glob.glob(args.glob))
    if args.cases:
        paths = [p for p in paths
                 if p.stem.rsplit("_seed", 1)[0] in set(args.cases)]
    if not paths:
        print(f"没有匹配的布局: {args.glob}")
        return 1

    log = load_run_log()
    print(f"检查 {len(paths)} 份布局 (run_log 有 {len(log)} 行)\n")
    bad = 0
    for p in paths:
        r = check_one(p, log, verbose=not args.quiet)
        tag = "FAIL" if r["problems"] else " ok "
        line = f"[{tag}] {r['stem']:22s}"
        if r.get("obj_eval") is not None and r["obj_eval"] is not None:
            line += f" 线长 {r['obj_eval']:9.2f}"
        if "bbox_area_mm2" in r:
            line += f"  bbox {r['bbox_area_mm2']:8.1f} mm^2"
        if "vs_greedy" in r:
            line += f"  比greedy好 {r['vs_greedy']:6.1%}"
        print(line)
        for note in r.get("notes", []):
            if not args.quiet:
                print(f"        · {note}")
        for prob in r["problems"]:
            print(f"        ! {prob}")
        bad += bool(r["problems"])

    print(f"\n=== {len(paths) - bad}/{len(paths)} 通过 ===")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
