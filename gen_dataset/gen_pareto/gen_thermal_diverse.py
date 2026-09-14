#!/usr/bin/env python3
"""gen_thermal_diverse.py — 用两个冻结代理生成「热峰值 / 线长 / bbox 面积」帕累托布局数据集。

方向 (与 gen_pareto_dataset.py 的 σ/ρ 置换版**不同**)
----------------------------------------------------
flow matching 学的是 **p(layout | graph, power)** —— 条件 (抽象图 + 功耗) 冻结, 学**坐标**。
主杠杆 = **径向铺开 C0** (见 geometry_gen.py): 不换标签、不换功耗, 只在坐标上等比放大,
产出"同一条件对应多个不同布局"的语料, 单调扫出「紧凑=热=短线 ↔ 铺开=凉=长线」的热-线长前沿。

两条热杠杆 (可选 --greedy-rounds)
---------------------------------
  1. 径向铺开 (默认): 纯几何, 凉了但**线长按比例劣化** (s=1.15 → ~+21% 线长换 ~-3~7°C)。
  2. 功耗置换 (默认, --greedy-rounds=4): HRNet 贪心配对功耗交换, 坐标不动 ⇒ 线长/面积**严格不变**,
     是唯一的线长中性热杠杆 (中位 ~-3°C)。⚠ 它改的是功耗标签 (条件), 与 reframing 冲突,
     且会让前沿 WL 最优端的 perm 点支配 C0 —— 用 --greedy-rounds 0 关闭。

流程 (每 system 独立)
---------------------
  L1 (CPU, multiprocessing.Pool)  候选库 = C0 (L0 原样, 线长/紧凑锚点) + 径向铺开
                                  (s ∈ scales, 只留画布 ≤ cap 的: 紧凑 → 铺开的面积-热前沿)
  L2 (GPU)                       整库打分: 热峰值 (GNNHRNet) + 线长 (WirelengthGNN, 规范功耗) + 面积 (解析)
  L3 (GPU, 可选)                 贪心功耗置换 (线长中性热杠杆, --greedy-rounds >0)
  帕累托                          三目标 (max_c, wl, bbox_area) 全部最小化, 先量化再比
  落盘                            主语料 (每 case 一行 或 整条前沿) + 完整前沿 sidecar

为什么 C0 恒在库中
------------------
C0 = L0 记录本身。于是输出**按构造不可能被 C0 支配** (逐 system 帕累托不劣)。

输入 / 输出
-----------
  源   : placement_dataset_tw/chiplet_dataset_{5..8}.json      (id 20001..40000)
  L0   : _l0_raw/chiplet_dataset_{5..8}.json                    (C0 锚, adp 产物)
  输出 : <out-dir>/chiplet_dataset_{1..N}.json                  (重编号 1..N)
         <out-dir>/front/chiplet_dataset_{1..N}.json            (完整帕累托前沿 sidecar)
         <out-dir>/provenance.json

必须用 chipdiffusion 环境。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # gen_pareto/ (本地模块)
import opt_dataset_lib as L  # noqa: E402
from geometry_gen import spread_bank  # noqa: E402
from gen_pareto_dataset import greedy_power_search  # noqa: E402

# --------------------------------------------------------------------------------------
# L1: 候选库构造 (纯 CPU, 可并行; 不 import torch)
# --------------------------------------------------------------------------------------
_BANK_CFG: dict = {}


def build_bank(core: L.SystemCore, seed: int) -> List[L.Candidate]:
    """C0 锚 + 径向铺开候选 (固定条件, 只动坐标)。不可行/违法者丢弃。"""
    cfg = _BANK_CFG
    bank: List[L.Candidate] = []
    c0 = L.gen_c0(core)
    if c0 is not None:
        bank.append(c0)

    for s, fp in spread_bank(core, scales=cfg.get("scales"), cap=cfg.get("cap")):
        if s == 1.0:
            continue  # s=1.0 就是 C0, 已在库中
        rec = L.make_record(core, fp=[tuple(f) for f in fp], sid=core.sid)
        if rec is None:
            continue
        errs = L.assert_invariants(rec, core, sigma=list(range(core.n)))
        if errs:
            continue
        bank.append(L.Candidate(record=rec, tag=f"rad{s:.2f}", sigma=list(range(core.n))))
    return bank


_WORKER_SCORER: Optional[L.SurrogateScorer] = None
_WORKER_ARGS = None


def _get_scorer() -> L.SurrogateScorer:
    """每个 worker 进程惰性加载自己的代理 (torch 模型 + CUDA 上下文)。"""
    global _WORKER_SCORER
    if _WORKER_SCORER is None:
        _WORKER_SCORER = L.SurrogateScorer(device=_WORKER_ARGS.device,
                                           thermal_batch=_WORKER_ARGS.thermal_batch)
    return _WORKER_SCORER


def _full_job(payload):
    """一个 system 的完整流程 (L1 建库 + L2 打分 + L3 贪心 + 帕累托), 在 worker 进程内跑。

    打分 (占 ~90% 时间) 原来在主进程串行; 单张卡 GPU 利用率只有 ~1%, 瓶颈是每个 system
    的小批量 + CPU 侧开销, 所以用多进程共享同一张卡重叠 CPU/GPU, 能把空闲算力用起来。
    """
    sid, rec = payload
    core = L.load_core(sid, rec)
    if core is None:
        return sid, None, "load_core_failed"
    seed = _BANK_CFG.get("seed", 0) + int(sid.split("_")[1])
    bank = build_bank(core, seed)
    if not bank:
        return sid, None, "empty_bank"
    res = process_system(core, bank, _get_scorer(), _WORKER_ARGS)
    if res is None:
        return sid, None, "infeasible"
    return sid, res, None


# --------------------------------------------------------------------------------------
# 单 system 全流程
# --------------------------------------------------------------------------------------
def process_system(core: L.SystemCore, bank: List[L.Candidate], scorer: L.SurrogateScorer,
                   args) -> Optional[dict]:
    # L2: 打分 (线长用 C0 的功耗向量做排序口径, 抵消线长代理对功耗的虚假依赖)
    records = [c.record for c in bank]
    scores = scorer.score_bank(records, canonical_powers=[core.powers] * len(records))
    for c, s in zip(bank, scores):
        c.objective = (s["max_c"], s["wl"], s["bbox_area"])
    wl_canon = scores[0]["wl"]
    area = scores[0]["bbox_area"]

    # L3: 功耗置换 (HRNet 贪心) —— 唯一的线长中性热杠杆 (坐标不动 ⇒ 线长/面积严格不变)。
    #     ⚠ 它改的是功耗标签 (条件); 开启后前沿在 WL 最优端会被 perm 点支配 C0 (同样 WL, 更凉)。
    if args.greedy_rounds > 0 and core.n >= 2:
        best_pw, _ = greedy_power_search(
            core, scorer, [float(c["power"]) for c in bank[0].record["chiplets"]],
            args.greedy_rounds, min_gain=args.greedy_min_gain)
        rec = L.make_record(core, power_slots=best_pw, sid=core.sid)
        if rec is not None:
            th = scorer.thermal_batched([rec])
            bank.append(L.Candidate(record=rec, tag="perm",
                                    sigma=list(range(core.n)),
                                    objective=(th[0]["max_c"], wl_canon, area)))

    # 帕累托 (量化分辨率与 pareto_front 同一套)
    q = L.obj_quantum([c.objective for c in bank])
    front = L.pareto_front(bank, q=q)
    c0_obj = next((c.objective for c in bank if c.tag == "C0"), None)
    if c0_obj is not None:
        for c in front:
            if L.dominates(c0_obj, c.objective, q):
                raise AssertionError(
                    f"{core.sid}: 前沿成员 {c.tag} 被 C0 支配 (帕累托实现有 bug) "
                    f"C0={c0_obj} {c.tag}={c.objective}")

    # 选片
    rng = random.Random(args.seed + int(core.sid.split("_")[1]) * 7919)
    if args.corpus_mode == "paretor":
        selected = sorted(front, key=lambda c: c.objective[0])
    else:
        selected = [_select(front, rng, args)]
    pick = min(selected, key=lambda c: c.objective[0])

    return {
        "sid": core.sid,
        "selected": selected,
        "headline": pick,
        "front": front,
        "c0_obj": c0_obj,
        "ref": core.ref,
        "bank_size": len(bank),
        "front_size": len(front),
    }


def _select(front: Sequence[L.Candidate], rng: random.Random, args) -> L.Candidate:
    mode = args.front_select
    if mode == "knee":
        base = L.select_knee(front)
    elif mode == "thermal":
        base = min(front, key=lambda c: c.objective[0])
    elif mode == "wirelength":
        base = min(front, key=lambda c: c.objective[1])
    elif mode == "random":
        base = front[rng.randrange(len(front))]
    else:
        raise ValueError(mode)
    if len(front) > 1 and rng.random() < args.mix_fraction:
        return front[rng.randrange(len(front))]
    return base


# --------------------------------------------------------------------------------------
# 并行
# --------------------------------------------------------------------------------------
def iter_jobs(items, pool, workers: int, chunksize: int = 16):
    if pool is None or workers <= 1:
        for it in items:
            yield _full_job(it)
        return
    for out in pool.imap(_full_job, items, chunksize=chunksize):
        yield out


# --------------------------------------------------------------------------------------
# 中间产物 (可续跑)
# --------------------------------------------------------------------------------------
def _load_parts(parts_dir: Path):
    done, bufs = set(), {"main": {}, "front": {}, "meta": {}, "l0": {}}
    for f in sorted(parts_dir.glob("part_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        done.update(d["sids"])
        for key in bufs:
            bufs[key].update(d[key])
    return done, bufs


def _chunk_of(new_id: int) -> int:
    return (int(new_id) - 1) // L.CHUNK + 1


def assemble(args, parts_dir: Path, out_dir: Path, l0_out: Optional[Path], meta: dict) -> None:
    """合并 part 文件成主语料 chunk + 完整前沿 sidecar + L0 对照 + groups.json。

    行 id 按 case 升序统一分配 1..M (single 模式 M = n_systems; paretor 模式一行/前沿成员)。
    """
    parts = [json.loads(f.read_text(encoding="utf-8"))
             for f in sorted(parts_dir.glob("part_*.json"))]

    counts: Dict[int, int] = {}
    for d in parts:
        for nid_s, entries in d["main"].items():
            counts[int(nid_s)] = len(entries)
    start, rid = {}, 0
    for nid in sorted(counts):
        start[nid] = rid + 1
        rid += counts[nid]
    n_rows = rid
    n_chunks = max(1, (n_rows - 1) // 5000 + 1)

    for ci in range(1, n_chunks + 1):
        lo, hi = (ci - 1) * 5000 + 1, ci * 5000
        bucket: Dict[str, dict] = {}
        for d in parts:
            for nid_s, entries in d["main"].items():
                s = start[int(nid_s)]
                for k, e in enumerate(entries):
                    r = s + k
                    if lo <= r <= hi:
                        bucket[f"system_{r}"] = e["record"] | {"system_id": f"system_{r}"}
        L.write_json_atomic(out_dir / f"chiplet_dataset_{ci}.json", bucket)
        print(f"[fd-pareto] 写出 {out_dir / f'chiplet_dataset_{ci}.json'} ({len(bucket)} 行)",
              flush=True)

    row_to_case = {r: nid for nid, s in start.items() for r in range(s, s + counts[nid])}
    L.write_json_atomic(out_dir / "groups.json", {
        "schema_version": 1, "n_rows": n_rows, "n_cases": len(counts),
        "note": "行 -> case; 切分必须按 case 分组, 否则同 case 的不同布局会泄到 train/val 两侧。",
        "row_to_case": {str(r): c for r, c in sorted(row_to_case.items())},
    }, indent=0)
    print(f"[fd-pareto] 写出 {out_dir / 'groups.json'} ({n_rows} 行 / {len(counts)} case)",
          flush=True)

    if l0_out is not None:
        n_l0_chunks = _chunk_of(meta["n_systems"])
        for ci in range(1, n_l0_chunks + 1):
            bucket = {}
            for d in parts:
                for nid, rec in d["l0"].items():
                    if _chunk_of(nid) == ci:
                        bucket[f"system_{nid}"] = rec
            L.write_json_atomic(Path(l0_out) / f"chiplet_dataset_{ci}.json", bucket)
            print(f"[fd-pareto] 写出 {Path(l0_out) / f'chiplet_dataset_{ci}.json'} "
                  f"({len(bucket)} systems)", flush=True)

    front_dir = out_dir / "front"
    n_case_chunks = _chunk_of(meta["n_systems"])
    for ci in range(1, n_case_chunks + 1):
        bucket, fmeta = {}, {}
        for d in parts:
            for nid, entries in d["front"].items():
                if _chunk_of(nid) != ci:
                    continue
                for k, rec in enumerate(entries):
                    bucket[f"system_{nid}__c{k}"] = rec
            for nid, m in d["meta"].items():
                if _chunk_of(nid) == ci:
                    fmeta[f"system_{nid}"] = m
        L.write_json_atomic(front_dir / f"chiplet_dataset_{ci}.json", bucket)
        L.write_json_atomic(front_dir / f"meta_{ci}.json", fmeta, indent=1)
        print(f"[fd-pareto] 写出 {front_dir / f'chiplet_dataset_{ci}.json'} ({len(bucket)} 前沿成员)",
              flush=True)

    meta["n_rows"] = n_rows
    meta["n_chunks"] = n_chunks
    L.write_json_atomic(out_dir / "provenance.json", meta, indent=2)
    print(f"[fd-pareto] 写出 {out_dir / 'provenance.json'}", flush=True)


# --------------------------------------------------------------------------------------
def run(args) -> dict:
    src_dir, l0_dir = Path(args.src_dir), Path(args.l0_dir)
    out_dir = Path(args.out_dir)
    l0_out = Path(args.l0_out_dir) if args.l0_out_dir else None
    parts = out_dir / ".parts_fd"
    parts.mkdir(parents=True, exist_ok=True)

    all_items: List[Tuple[str, dict]] = []
    for k in args.chunk:
        src = L.load_chunk(src_dir / f"chiplet_dataset_{k}.json")
        l0 = L.load_chunk(l0_dir / f"chiplet_dataset_{k}.json")
        for sid in L.sorted_sids(src):
            if sid in l0:
                all_items.append((sid, l0[sid]))
    if not all_items:
        raise SystemExit("没有任何 (源, L0) 配对")
    src_ids = [int(s.split("_")[1]) for s, _ in all_items]
    base = min(src_ids) - 1
    print(f"[fd-pareto] {len(all_items)} systems, 源 id {min(src_ids)}..{max(src_ids)}, "
          f"重编号 -{base} -> 1..{len(all_items)}", flush=True)

    _BANK_CFG.update(seed=args.seed, scales=tuple(args.scales), cap=args.cap)

    done, bufs = _load_parts(parts) if args.resume else (set(), None)
    if done:
        print(f"[fd-pareto] 续跑: 已有 {len(done)} 个 system 完成", flush=True)
    bufs = bufs if bufs is not None else {"main": {}, "front": {}, "meta": {}, "l0": {}}
    n_parts = len(list(parts.glob("part_*.json")))

    todo = [(s, r) for s, r in all_items if s not in done]
    if args.start:
        todo = todo[args.start:]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[fd-pareto] 本次处理 {len(todo)} 个 system", flush=True)

    global _WORKER_ARGS
    _WORKER_ARGS = args
    pool = mp.get_context("fork").Pool(args.workers) if args.workers > 1 else None

    stats = defaultdict(float)
    deltas: Dict[str, List[float]] = {"max_c": [], "wl": [], "area": []}
    t0 = time.time()
    n_done = 0
    buf_sids: List[str] = []

    def flush_part():
        nonlocal n_parts, buf_sids
        if not buf_sids:
            return
        L.write_json_atomic(parts / f"part_{n_parts:05d}.json",
                            {"sids": buf_sids, **{k: v for k, v in bufs.items()}})
        n_parts += 1
        for v in bufs.values():
            v.clear()
        buf_sids = []

    try:
        for sid, res, err in iter_jobs(todo, pool, args.workers):
            if err is not None:
                stats[f"drop_{err}"] += 1
                continue
            if res is None:
                stats["drop_infeasible"] += 1
                continue
            nid = int(sid.split("_")[1]) - base
            bufs["main"][str(nid)] = [{"record": c.record, "tag": c.tag} for c in res["selected"]]
            bufs["front"][str(nid)] = [c.record for c in res["front"]]
            bufs["l0"][str(nid)] = res["ref"] | {"system_id": f"system_{nid}"}
            bufs["meta"][str(nid)] = {
                "src_id": sid, "bank": res["bank_size"], "front": res["front_size"],
                "rows": len(res["selected"]),
                "tags": sorted({c.tag for c in res["front"]}),
                "c0_obj": res["c0_obj"], "headline_obj": res["headline"].objective,
            }
            buf_sids.append(sid)
            n_done += 1
            stats["bank"] += res["bank_size"]
            stats["front"] += res["front_size"]
            if res["c0_obj"] and res["headline"].objective:
                c0o, selo = res["c0_obj"], res["headline"].objective
                q = L.obj_quantum([c0o, selo])
                q0, q1 = L.quantize(c0o, q), L.quantize(selo, q)
                n_better = 0
                for i, name in enumerate(("max_c", "wl", "area")):
                    d = selo[i] - c0o[i]
                    deltas[name].append(d)
                    stats[f"d_{name}"] += d
                    if q1[i] < q0[i]:
                        stats[f"better_{name}"] += 1
                        n_better += 1
                    elif q1[i] > q0[i]:
                        stats[f"worse_{name}"] += 1
                if n_better >= 2:
                    stats["better_2axis"] += 1
                if n_better >= 1:
                    stats["better_1axis"] += 1
            if n_done % args.subchunk == 0:
                flush_part()
            if n_done % args.print_every == 0 or n_done == len(todo):
                el = time.time() - t0
                rate = n_done / max(el, 1e-9)
                print(f"[fd-pareto] {n_done}/{len(todo)}  {el:.0f}s  {rate:.1f} sys/s  "
                      f"ETA {(len(todo) - n_done) / max(rate, 1e-9) / 60:.1f} min  "
                      f"库均 {stats['bank'] / n_done:.1f}  前沿均 {stats['front'] / n_done:.1f}",
                      flush=True)
    finally:
        flush_part()
        if pool is not None:
            pool.close()
            pool.join()

    n_total = len(all_items)
    if n_done == 0 and not done:
        raise SystemExit("一个 system 都没处理成功")

    print(f"\n[fd-pareto] 完成 {n_done} 个 (累计 {len(done) + n_done}/{n_total}), "
          f"用时 {(time.time() - t0) / 60:.1f} min")
    n = max(n_done, 1)
    if stats["bank"]:
        print(f"  候选库 {stats['bank'] / n:.1f}/system   前沿 {stats['front'] / n:.1f}/system")
        for name in ("max_c", "wl", "area"):
            d = deltas[name]
            med = st.median(d) if d else float("nan")
            print(f"  Δ{name}: 中位 {med:+.3f}  均值 {stats[f'd_{name}'] / n:+.3f}  "
                  f"改善 {stats[f'better_{name}']:.0f} 变差 {stats[f'worse_{name}']:.0f} "
                  f"({100 * stats[f'better_{name}'] / n:.1f}%)")
        print(f"  改善 ≥1 轴: {stats['better_1axis']:.0f}/{n} ({100 * stats['better_1axis'] / n:.1f}%)"
              f"   ≥2 轴: {stats['better_2axis']:.0f}/{n} ({100 * stats['better_2axis'] / n:.1f}%)")
        for k in sorted(k for k in stats if k.startswith("drop_")):
            print(f"  丢弃 {k[5:]}: {stats[k]:.0f}")

    if args.dry_run:
        print("[fd-pareto] --dry-run: 不落盘")
        return dict(stats)

    if len(done) + n_done < n_total and not args.force_assemble:
        print(f"[fd-pareto] ⚠ 只完成 {len(done) + n_done}/{n_total}, 跳过合并。"
              f"补齐后重跑即可续跑合并 (或用 --force-assemble 强行合并)。")
        return dict(stats)

    prov = {
        "schema_version": 1,
        "pipeline": "gen_thermal_diverse.py (固定条件 + 径向铺开 + 代理打分 + 帕累托)",
        "source": {"dir": str(src_dir), "chunks": list(args.chunk),
                   "src_ids": [min(src_ids), max(src_ids)],
                   "renumber_offset": base, "l0_dir": str(l0_dir)},
        "n_systems": n_total,
        "seed": args.seed,
        "params": {
            "scales": list(args.scales), "cap": args.cap,
            "front_select": args.front_select, "mix_fraction": args.mix_fraction,
            "corpus_mode": args.corpus_mode,
        },
        "objectives": ["therm_max_c (GNNHRNet)", "wirelength (WirelengthGNN, 原功耗)",
                       "bbox_area (解析)"],
        "note": "id 已重编号为 1..N; 老 id = 新 id + renumber_offset。front/ 下是每个 system 的"
                "完整帕累托前沿。条件 (图 + 功耗) 逐字冻结, 只动坐标 (径向铺开)。",
    }
    assemble(args, parts, out_dir, l0_out, prov)
    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--src-dir", type=Path,
                    default=L.PROJECT / "Dataset/dataset/placement_dataset/placement_dataset_tw")
    ap.add_argument("--l0-dir", type=Path,
                    default=L.PROJECT / "Dataset/dataset/placement_dataset/_l0_raw")
    ap.add_argument("--out-dir", type=Path,
                    default=L.PROJECT / "Dataset/dataset/placement_dataset/placement_dataset_opt_pareto")
    ap.add_argument("--l0-out-dir", type=Path,
                    default=L.PROJECT / "Dataset/dataset/placement_dataset/placement_dataset_l0")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 system (0=全部)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--subchunk", type=int, default=500)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--force-assemble", action="store_true")
    ap.add_argument("--seed", type=int, default=20240913)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行 worker 进程数。⚠ 实测单张卡上 1 vs 4 worker 都是 ~1.0 sys/s —— "
                         "瓶颈是 GPU (贪心热搜索打满单卡), 多进程只争抢不提速。"
                         "仅当有多张卡时才值得 >1。")
    ap.add_argument("--thermal-batch", type=int, default=32)
    ap.add_argument("--print-every", type=int, default=100)
    # 几何
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3],
                    help="径向铺开因子扫描 (1.0 = C0); 上限 1.3 把线长劣化收在 ~+43%% 以内")
    ap.add_argument("--cap", type=float, default=187.0,
                    help="画布边长上限 (mm) = 热代理训练上限, 压在外推区以内")
    # 功耗置换 (线长中性热杠杆; 默认开启)
    ap.add_argument("--greedy-rounds", type=int, default=4,
                    help="HRNet 贪心功耗置换轮数 (0=关闭; 默认 4, 打开线长中性的热杠杆)")
    ap.add_argument("--greedy-min-gain", type=float, default=L.MAXC_QUANTUM)
    # 选片
    ap.add_argument("--corpus-mode", choices=["single", "paretor"], default="paretor",
                    help="paretor (默认): 每 case 写整条非支配前沿 (多行, 多样性)。"
                         "single: 每 case 一行 = --front-select 选点 (老 20k 一行契约)。")
    ap.add_argument("--front-select", choices=["thermal", "wirelength", "knee", "random"],
                    default="thermal")
    ap.add_argument("--mix-fraction", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
