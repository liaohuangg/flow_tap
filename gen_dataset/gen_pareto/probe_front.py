#!/usr/bin/env python3
"""probe_front.py — 看单个 system 的 (热峰值, 线长, 面积) 前沿: C0 + 径向铺开 + 功耗置换。

回答两个问题:
  1. 有没有「热更好且线长不劣化」的帕累托解? → 功耗置换 (perm): 坐标不动 ⇒ ΔWL=0, Δmax_c ~-3°C。
  2. 几何铺开的线长代价能否收住? → s 上限 1.3, 线长劣化 ~+43% 换 ~-8~11°C。

用法: python gen_pareto/probe_front.py [N] [chunk] [greedy_rounds]
"""
import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # gen_pareto/
import opt_dataset_lib as L  # noqa: E402
from geometry_gen import spread_bank  # noqa: E402
from gen_pareto_dataset import greedy_power_search  # noqa: E402

SCALES = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3)
CAP = 187.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", nargs="?", type=int, default=12)
    ap.add_argument("chunk", nargs="?", type=int, default=5)
    ap.add_argument("greedy_rounds", nargs="?", type=int, default=4)
    args = ap.parse_args()

    src = L.load_chunk(L.PROJECT / "Dataset/dataset/placement_dataset/placement_dataset_tw"
                       / f"chiplet_dataset_{args.chunk}.json")
    l0 = L.load_chunk(L.PROJECT / "Dataset/dataset/placement_dataset/_l0_raw"
                      / f"chiplet_dataset_{args.chunk}.json")
    sids = [s for s in L.sorted_sids(src) if s in l0][: args.N]

    scorer = L.SurrogateScorer(device="cuda", thermal_batch=32)
    rows = []
    for sid in sids:
        core = L.load_core(sid, l0[sid])
        if core is None:
            continue
        bank = []
        c0 = L.gen_c0(core)
        if c0 is not None:
            bank.append(c0)
        for s, fp in spread_bank(core, scales=SCALES, cap=CAP):
            if s == 1.0:
                continue
            rec = L.make_record(core, fp=[tuple(f) for f in fp], sid=core.sid)
            if rec is None:
                continue
            if L.assert_invariants(rec, core, sigma=list(range(core.n))):
                continue
            bank.append(L.Candidate(record=rec, tag=f"rad{s:.2f}",
                                    sigma=list(range(core.n))))
        if not bank:
            continue
        recs = [c.record for c in bank]
        scores = scorer.score_bank(recs, canonical_powers=[core.powers] * len(recs))
        for c, s in zip(bank, scores):
            c.objective = (s["max_c"], s["wl"], s["bbox_area"])
        wl_canon, area = scores[0]["wl"], scores[0]["bbox_area"]

        if args.greedy_rounds > 0 and core.n >= 2:
            best_pw, _ = greedy_power_search(
                core, scorer, [float(c["power"]) for c in bank[0].record["chiplets"]],
                args.greedy_rounds, min_gain=L.MAXC_QUANTUM)
            rec = L.make_record(core, power_slots=best_pw, sid=core.sid)
            if rec is not None:
                th = scorer.thermal_batched([rec])
                bank.append(L.Candidate(record=rec, tag="perm",
                                        sigma=list(range(core.n)),
                                        objective=(th[0]["max_c"], wl_canon, area)))

        q = L.obj_quantum([c.objective for c in bank])
        front = sorted(L.pareto_front(bank, q=q), key=lambda c: c.objective[0])
        c0_obj = next(c.objective for c in bank if c.tag == "C0")
        print(f"{sid} n={core.n:>2d}  C0(T={c0_obj[0]:6.1f}, W={c0_obj[1]:9.0f})")
        for c in front:
            dT = c.objective[0] - c0_obj[0]
            dW = (c.objective[1] / c0_obj[1] - 1.0) * 100
            print(f"    {c.tag:<10} T={c.objective[0]:6.1f} (ΔT={dT:+5.2f})  "
                  f"W={c.objective[1]:9.0f} (ΔW={dW:+5.1f}%)")
        rows.append((sid, c0_obj, front))

    print()
    # 统计: 每条前沿上, perm 相对 C0 的 ΔT, 以及最热铺开点的 ΔW
    perm_dT, rad_dW = [], []
    for sid, c0o, front in rows:
        perm = next((c for c in front if c.tag == "perm"), None)
        if perm:
            perm_dT.append(perm.objective[0] - c0o[0])
        rads = [c for c in front if c.tag.startswith("rad")]
        if rads:
            coolest = max(rads, key=lambda c: c.objective[0])  # 最热的 rad (最接近 C0)
            rad_dW.append((coolest.objective[1] / c0o[1] - 1.0) * 100)
    if perm_dT:
        print(f"perm  (线长中性) : Δmax_c 中位 {st.median(perm_dT):+.2f}°C  "
              f"(n={len(perm_dT)})  线长严格不变")
    if rad_dW:
        print(f"rad s=1.3 (最冷)  : 相对 C0 线长 {st.median(rad_dW):+.1f}%  (n={len(rad_dW)})")


if __name__ == "__main__":
    main()
