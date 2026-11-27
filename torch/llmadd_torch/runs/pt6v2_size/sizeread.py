#!/usr/bin/env python3
"""pt6v2-size read (post-hoc, exploratory; NO check gates it). The six pt6v2 priors at 190M / 505M / 1.07B, seed 16.
Definitions = runs/pt6v2/out_v8/v8read.py and runs/pt6v2/tests/posthoc_speed_20261120/v4final.py (v4final's prefix exec'd read
only: dim_strict, BKT, TOY_KEY):
  numbers = mean held_<b> over the 6 buckets; dim = strict (hits_dim_held over dim_strict(D)'s keep mask); cities / countries =
  <d>_held_2way; torus / box5 = TOY_KEY (held_nonid / held_no_negtwin_nonid); area = 100 * mean over every eval; final3 = 100 *
  mean of the last 3 evals; bpb = the last eval's val_bits_per_byte; LM share = median over prior-on evals of
  grad_lm / (grad_lm + grad_prior) (v4final).
Per size and manifold:
  1. all6 area and final3 vs none at N_m = the none lr with the highest area for that manifold ('open' when N_m is at the edge
     of the 4-lr grid); 2. all6full final3 vs the best none final3 (max over the 4 lrs); 3. only-dim vs all6 on dim (area,
  final3); 4. the fraction of training all6 needs to reach the best none final3 (first eval whose 3-eval running mean >= it,
  as F105 addendum 2; also the single-eval version); 5. bpb and LM share per run; device / check flags.
Reading rule (runs/pt6v2_size/AUDIT.md): exploratory, no gates; 'promising' = all six all6 - best-none differences positive at
a size (area and final3 reported side by side).
REP 505M (2026-11-24): --rep reads 505M seeds 16, 17, 18 with the none grid + 3.84e-2: one table per seed (the same
definitions; seed 16 also has none@2.4e-3), then the per-cell seed mean (area / final3 / bpb, n seeds per cell) with N_m and
the best none picked on the seed means over the lrs every seed has (4.8e-3 .. 3.84e-2), and each seed's difference at that
N_m. Without --rep the output is unchanged (seed 16, the registered 4-lr grids).
REP 190M (2026-11-24): --rep then also reads 190M seeds 16, 17, 18 with the none grid 1.2e-3 .. 3.84e-2 (6 lrs, every seed),
picking N_m and the best none on the seed means over all tested lrs; cells read with fewer seeds than the most-seeded cell are
marked '<n' (JSON out["rep_190m"], same layout as out["rep"] = 505M).
SAMELR (2026-11-25): --rep then also prints, after the three REP blocks, a same-lr bpb row per size: all6@6e-4 vs
none@6e-4 (the SAMELR runs), seeds 16-19, per seed and the mean over the seeds both arms have (JSON out["samelr_bpb"][size];
exploratory, no verdict). Everything printed or written before it is unchanged (samelr_checks/).
REP 1B (2026-11-24): --rep then also reads 1.07B seeds 16, 17, 18 (JSON out["rep_1b"], same layout). Seed 16: the registered
1.2e-3 .. 9.6e-3 + 1.92e-2, 3.84e-2 (6 lrs) and only-dim; seeds 17, 18: all6, all6full, none@4.8e-3 .. 3.84e-2 (no only-dim, no
1.2e-3 / 2.4e-3, by design). N_m and the best none are picked on the seed means over the lrs every seed has (4.8e-3 .. 3.84e-2, as
505M); cells with fewer seeds are marked '<n'; only-dim (seed 16 only by design) is left out of the unequal-n FLAG and compared
with all6 on the seeds both have.
BRACKET 7.68e-2 (2026-11-25): --rep adds none@7.68e-2 (seeds 16-18) to the 190M and 1.07B grids and to the lrs N_m / the best none
are picked from on the seed means (the lrs every seed has: 190M 1.2e-3 .. 7.68e-2, 1.07B 4.8e-3 .. 7.68e-2). 505M, SAMELR, the
default and --known53 outputs are unchanged.
usage: sizeread.py [--json out.json] [--known53] [--rep]   (--known53: KNOWN-ANSWER SMOKE ONLY: reads the registered 53M seed-16 runs
(all6 := all6@2e-3, all6full := V4 all6full@1e-3, only-dim@2e-3, none at the registered 6-lr grid) and compares final3 / bpb
with runs/pt6v2/out_v8/v8read.json and area with each run's own metrics block)"""
import json
import sys
from pathlib import Path

import numpy as np

V4F = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2/tests/posthoc_speed_20261120/v4final.py")
_src = V4F.read_text()
G = {"__name__": "v4final_prefix", "__file__": str(V4F)}
exec(compile(_src[:_src.index("# ---------------------------------------------------------------- load every run")], str(V4F), "exec"), G)
dim_strict, BKT, TOY_KEY = G["dim_strict"], G["BKT"], G["TOY_KEY"]
R = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2")
S = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2_size")
DOMS = ("numbers", "dim", "cities", "countries", "torus", "box5")
KNOWN = "--known53" in sys.argv
SEED = 16
NONE_A = ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2")
NONE_B = ("1.2e-3", "2.4e-3", "4.8e-3", "9.6e-3")
RULE = ("exploratory, no gates: per size and manifold report all6 - best-none on area and final3; 'promising' = all six positive at a "
        "size (-> 2 more seeds under the standing authorisation)")
if KNOWN:
    SIZES = {"53m(known)": {"all6": R / "out/pt6v2-all6-lr2e-3-s16", "all6full": R / "out_v4/pt6v2-all6full-lr1e-3-s16",
                            "only-dim": R / "out/pt6v2-only-dim-lr2e-3-s16",
                            "none": {lr: R / f"out/pt6v2-none-lr{lr}-s16" for lr in ("2e-3", "4e-3", "8e-3", "1.6e-2", "3.2e-2", "6.4e-2")}}}
else:
    SIZES = {z: {"all6": S / z / "all6-lr6e-4-s16", "all6full": S / z / "all6full-lr6e-4-s16", "only-dim": S / z / "only-dim-lr6e-4-s16",
                 "none": {lr: S / z / f"none-lr{lr}-s16" for lr in g}}
             for z, g in (("190m", NONE_A), ("505m", NONE_A), ("1b", NONE_B))}


def device(d):
    f = d / "gpu.csv"
    return ",".join(sorted({ln.split(",")[1].strip() for ln in f.read_text().splitlines()[1:] if ln.count(",") >= 1})) if f.exists() else None


def load(d):
    fs = sorted(d.glob("*/run_whole_*.json"))
    if len(fs) != 1:
        return None
    r = json.loads(fs[0].read_text())
    c = r["curve"]
    if not c or c[-1]["step"] != r["steps"] - 1:
        return None
    st = np.array([e["step"] for e in c])
    D = r["domains"]["dim"]
    keep, _, _ = dim_strict(D)
    H = np.array([[int(x) for x in e["hits_dim_held"]] for e in c], float) / D["eval_held"]["contexts"]
    Sx = {"numbers": np.array([np.mean([e[f"held_{b}"] for b in BKT]) for e in c]), "dim": H[:, keep].mean(1)}
    for dd in ("cities", "countries"):
        Sx[dd] = np.array([e[f"{dd}_held_2way"] for e in c])
    for dd in ("torus", "box5"):
        Sx[dd] = np.array([e[TOY_KEY[dd]] for e in c])
    rec = {"st": st, "S": Sx, "steps": r["steps"], "bpb": float(c[-1]["val_bits_per_byte"]), "dev": device(d), "path": str(fs[0]),
           "area": {m: 100 * float(np.mean(Sx[m])) for m in DOMS}, "final3": {m: 100 * float(np.mean(Sx[m][-3:])) for m in DOMS},
           "events": r.get("prior_events"), "n_params": r.get("n_params"), "lr": r["args"]["lr"], "d_model": r["args"].get("d_model")}
    on = [e for e in c if e.get("prior_on") and "grad_lm" in e and "grad_prior" in e]
    rec["share"] = float(np.median([e["grad_lm"] / (e["grad_lm"] + e["grad_prior"]) for e in on])) if on else None
    mt = r.get("metrics", {})
    rec["mx"] = {m: rec["area"][m] - mt[k]["area"] for m, k in (("numbers", "numbers"), ("cities", "cities_readout"),
                 ("countries", "countries_readout"), ("torus", "torus_readout"), ("box5", "box5_readout")) if k in mt}
    flags = []
    f = d / "check.json"
    if f.exists():
        k = json.loads(f.read_text())
        if not k.get("pass"):
            flags.append("check.json: " + "; ".join(x["check"][:50] for x in k["checks"] if not x["ok"])[:240])
    rec["flags"] = flags
    return rec


def first_frac(rec, m, thr, roll):
    v = 100 * rec["S"][m]
    st = rec["st"]
    if roll > 1:
        v = np.convolve(v, np.ones(roll) / roll, mode="valid")
        st = st[roll - 1:]
    w = np.where(v >= thr)[0]
    return (int(st[w[0]]), float((st[w[0]] + 1) / rec["steps"])) if len(w) else (None, None)


out = {"rule": RULE, "known53": KNOWN, "sizes": {}}
f1 = lambda v: "   n/a" if v is None else f"{v:6.1f}"   # noqa: E731
def read_size(z, cfg):
    runs = {a: load(cfg[a]) for a in ("all6", "all6full", "only-dim") if a in cfg}   # REP 1B: s17 / s18 have no only-dim
    nones = {lr: load(p) for lr, p in cfg["none"].items()}
    miss = [a for a, v in runs.items() if v is None] + [f"none@{lr}" for lr, v in nones.items() if v is None]
    print(f"\n######## {z}: missing {miss or 'none'}")
    zo = {"missing": miss, "manifolds": {}, "runs": {}}
    for a, v in list(runs.items()) + [(f"none@{lr}", v) for lr, v in nones.items()]:
        if v is not None:
            zo["runs"][a] = {"area": v["area"], "final3": v["final3"], "bpb": v["bpb"], "share": v["share"], "dev": v["dev"],
                             "flags": v["flags"], "events": v["events"], "n_params": v["n_params"], "lr": v["lr"], "path": v["path"],
                             "area_vs_metrics_max": max((abs(x) for x in v["mx"].values()), default=None)}
    A6, NN = runs["all6"], {lr: v for lr, v in nones.items() if v is not None}
    grid = list(cfg["none"])
    print(f"{'manifold':9s} | {'N_m':>7s} {'edge':>4s} | {'all6 area':>9s} {'none area':>9s} {'diff':>6s} | {'all6 f3':>7s} {'none f3':>7s} {'diff':>6s} "
          f"| {'best-f3 none':>12s} | {'full f3':>7s} {'diff':>6s} | {'reach f3(3)':>11s} {'reach(1)':>8s}")
    allpos = {"area": [], "final3": [], "full_final3": []}
    for m in DOMS:
        if not NN:
            break
        nm = max(NN, key=lambda lr: (NN[lr]["area"][m], float(lr)))   # area-best none lr (a tie goes to the higher lr)
        edge = "open" if (len(NN) == len(grid) and nm in (grid[0], grid[-1])) else ("?" if len(NN) < len(grid) else "")
        bf = max(NN, key=lambda lr: NN[lr]["final3"][m])
        row = {"N_m": nm, "edge": edge or None, "none_area": NN[nm]["area"][m], "none_final3": NN[nm]["final3"][m],
               "best_final3_lr": bf, "best_none_final3": NN[bf]["final3"][m]}
        if A6:
            row.update(all6_area=A6["area"][m], all6_final3=A6["final3"][m], d_area=A6["area"][m] - NN[nm]["area"][m],
                       d_final3=A6["final3"][m] - NN[nm]["final3"][m], d_final3_vs_bestf3=A6["final3"][m] - NN[bf]["final3"][m])
            st3, fr3 = first_frac(A6, m, NN[bf]["final3"][m], 3)
            st1, fr1 = first_frac(A6, m, NN[bf]["final3"][m], 1)
            row.update(reach_step_f3=st3, reach_frac_f3=fr3, reach_step_1=st1, reach_frac_1=fr1)
            allpos["area"].append(row["d_area"] > 0)
            allpos["final3"].append(row["d_final3"] > 0)
        if runs.get("all6full"):
            row.update(full_final3=runs["all6full"]["final3"][m], d_full_vs_bestf3=runs["all6full"]["final3"][m] - NN[bf]["final3"][m])
            allpos["full_final3"].append(row["d_full_vs_bestf3"] > 0)
        zo["manifolds"][m] = row
        g = row.get
        print(f"{m:9s} | {nm:>7s} {edge:>4s} | {f1(g('all6_area')):>9s} {f1(g('none_area')):>9s} {f1(g('d_area')):>6s} | {f1(g('all6_final3')):>7s} "
              f"{f1(g('none_final3')):>7s} {f1(g('d_final3')):>6s} | {f1(g('best_none_final3'))} @{bf:>6s} | {f1(g('full_final3')):>7s} "
              f"{f1(g('d_full_vs_bestf3')):>6s} | " + ("never" if g("reach_frac_f3") is None else f"{g('reach_frac_f3'):.2f}").rjust(11)
              + " " + ("never" if g("reach_frac_1") is None else f"{g('reach_frac_1'):.2f}").rjust(8))
    zo["all_six_positive"] = {k: (len(v) == 6 and all(v)) for k, v in allpos.items()}
    print(f"all six positive: all6 - none@N_m area {zo['all_six_positive']['area']}, final3 {zo['all_six_positive']['final3']}; "
          f"all6full - best none final3 {zo['all_six_positive']['full_final3']}")
    if A6 and runs.get("only-dim"):
        od = runs["only-dim"]
        zo["only_dim_vs_all6_dim"] = {"area": od["area"]["dim"] - A6["area"]["dim"], "final3": od["final3"]["dim"] - A6["final3"]["dim"],
                                      "only_dim_area": od["area"]["dim"], "only_dim_final3": od["final3"]["dim"]}
        print(f"only-dim vs all6 on dim: area {od['area']['dim']:.1f} vs {A6['area']['dim']:.1f} (all6 - only-dim {A6['area']['dim'] - od['area']['dim']:+.1f}); "
              f"final3 {od['final3']['dim']:.1f} vs {A6['final3']['dim']:.1f} (all6 - only-dim {A6['final3']['dim'] - od['final3']['dim']:+.1f})")
    print("runs: bpb | LM share | device | n_params | lr | events | flags")
    for a, v in zo["runs"].items():
        print(f"   {a:14s} {v['bpb']:.4f} | {('n/a' if v['share'] is None else format(v['share'], '.3f')):>5s} | {v['dev']} | "
              f"{v['n_params']} | {v['lr']} | {v['events']} | {'; '.join(v['flags']) or 'no flags'}"
              + (f" | AREA vs metrics {v['area_vs_metrics_max']:.1e}" if v["area_vs_metrics_max"] is not None and v["area_vs_metrics_max"] > 1e-6 else ""))
    devs = {v["dev"] for v in zo["runs"].values()}
    if len(devs) > 1:
        print(f"   FLAG mixed devices in {z}: {sorted(map(str, devs))}")
    return zo


REP_SEEDS, REP_GRID = (16, 17, 18), NONE_A + ("3.84e-2",)
REP190_GRID = ("1.2e-3",) + REP_GRID
# REP 190M (2026-11-24): per size, the none grid of each seed and the lrs N_m / best none are picked from on the seed means.
# 505M: as REP 505M (s16 also has 2.4e-3; the mean uses the lrs every seed has). 190M: every seed has 1.2e-3 .. 3.84e-2 (s16: the
# registered 4 + 1.2e-3, 3.84e-2), and N_m is picked over all tested lrs; a cell read with fewer seeds than the most-seeded cell
# is marked ('<n' after the edge column; JSON row["fewer_seeds"]).
REP_CFG = {"505m": {"grid": {16: REP_GRID, 17: REP_GRID[1:], 18: REP_GRID[1:]}, "mean_lrs": REP_GRID[1:],
                    "what": "none lrs common to all seeds"},
           "190m": {"grid": {s: REP190_GRID for s in REP_SEEDS}, "mean_lrs": REP190_GRID, "what": "all tested none lrs"}}
assert REP_GRID[0] == "2.4e-3"
# REP 1B (2026-11-24): s16 = the registered NONE_B + 1.92e-2, 3.84e-2 and only-dim; s17 / s18 = all6, all6full, none 4.8e-3 .. 3.84e-2.
REP1B_GRID = ("4.8e-3", "9.6e-3", "1.92e-2", "3.84e-2")
REP_CFG["1b"] = {"grid": {16: NONE_B + REP1B_GRID[2:], 17: REP1B_GRID, 18: REP1B_GRID}, "mean_lrs": REP1B_GRID,
                 "what": "none lrs common to all seeds", "arms": {16: ("all6", "all6full", "only-dim"), 17: ("all6", "all6full"),
                                                               18: ("all6", "all6full")}, "by_design": ("only-dim",)}
assert NONE_B[2:] == REP1B_GRID[:2]
# BRACKET 7.68e-2 (2026-11-25, exp-builder): none@7.68e-2 seeds 16, 17, 18 at 190M and 1.07B (the none optimum on dim / grid sat at
# 3.84e-2, the old top). Appended to every seed's grid and to mean_lrs at both sizes, so N_m / the best none on the seed means
# range over the lrs every seed has (190M 1.2e-3 .. 7.68e-2, 1.07B 4.8e-3 .. 7.68e-2) and 'open' now means 1.2e-3 / 4.8e-3 or 7.68e-2.
# 505M unchanged. Until the runs exist the new cell is 'missing' and the edge column reads '?'; every number is unchanged
# (bracket_checks/).
BR = "7.68e-2"
for _z in ("190m", "1b"):
    REP_CFG[_z]["grid"] = {s: tuple(g) + (BR,) for s, g in REP_CFG[_z]["grid"].items()}
    REP_CFG[_z]["mean_lrs"] = tuple(REP_CFG[_z]["mean_lrs"]) + (BR,)
assert REP_CFG["190m"]["mean_lrs"] == REP190_GRID + (BR,) and REP_CFG["1b"]["mean_lrs"] == REP1B_GRID + (BR,)
assert all(set(REP_CFG[z]["mean_lrs"]) <= set(g) for z in ("190m", "1b", "505m") for g in REP_CFG[z]["grid"].values())


def rep_read(z="505m"):
    """seeds 16-18 at size z: per-seed tables (read_size), then the per-cell seed mean and each seed's difference at the mean N_m."""
    rc = REP_CFG[z]
    ro = {"per_seed": {}, "mean": {}}
    for s in REP_SEEDS:
        cfg = {"all6": S / z / f"all6-lr6e-4-s{s}", "all6full": S / z / f"all6full-lr6e-4-s{s}", "only-dim": S / z / f"only-dim-lr6e-4-s{s}",
               "none": {lr: S / z / f"none-lr{lr}-s{s}" for lr in rc["grid"][s]}}
        for a in set(("all6", "all6full", "only-dim")) - set(rc.get("arms", {}).get(s, ("all6", "all6full", "only-dim"))):
            del cfg[a]   # REP 1B: an arm this seed does not have by design
        ro["per_seed"][s] = read_size(f"{z} seed {s}", cfg)
    common = list(rc["mean_lrs"])
    cells_ = ["all6", "all6full", "only-dim"] + [f"none@{lr}" for lr in common]
    per = {c: {s: ro["per_seed"][s]["runs"][c] for s in REP_SEEDS if c in ro["per_seed"][s]["runs"]} for c in cells_}
    mean = {c: {"n": len(v), "seeds": sorted(v), "area": {m: float(np.mean([x["area"][m] for x in v.values()])) for m in DOMS},
                "final3": {m: float(np.mean([x["final3"][m] for x in v.values()])) for m in DOMS},
                "bpb": float(np.mean([x["bpb"] for x in v.values()]))} for c, v in per.items() if v}
    ns = {c: v["n"] for c, v in mean.items()}
    nmax = max(ns.values(), default=0)
    print(f"\n######## {z} seed mean over {REP_SEEDS} ({rc['what']}: {common}); seeds per cell {ns}")
    bd = rc.get("by_design", ())   # REP 1B: cells with fewer seeds by design (only-dim: seed 16 only) stay out of the FLAG
    if len({n for c, n in ns.items() if c not in bd}) > 1 or len(ns) < len(cells_):
        print(f"   FLAG unequal seeds per cell (or missing cells): the means are not paired; read the per-seed rows")
    if bd:
        bds = ", ".join(f"{c} (seeds {mean[c]['seeds']})" for c in bd if c in mean)
        print(f"   by design fewer seeds: {bds}")
    NN = {lr: mean[f"none@{lr}"] for lr in common if f"none@{lr}" in mean}
    A6, FU, OD = mean.get("all6"), mean.get("all6full"), mean.get("only-dim")
    print(f"{'manifold':9s} | {'N_m':>7s} {'edge':>4s} | {'all6 area':>9s} {'none area':>9s} {'diff':>6s} {'per-seed diff @N_m':>22s} | "
          f"{'all6 f3':>7s} {'none f3':>7s} {'diff':>6s} {'per-seed diff @N_m':>22s} | {'best-f3 none':>14s} | {'full f3':>7s} {'diff':>6s} {'per-seed':>22s}")
    pos = {"area": [], "final3": [], "full_final3": []}
    for m in DOMS:
        if not NN:
            break
        nm = max(NN, key=lambda lr: (NN[lr]["area"][m], float(lr)))
        edge = "open" if (len(NN) == len(common) and nm in (common[0], common[-1])) else ("?" if len(NN) < len(common) else "")
        bf = max(NN, key=lambda lr: NN[lr]["final3"][m])
        row = {"N_m": nm, "edge": edge or None, "none_area": NN[nm]["area"][m], "none_final3": NN[nm]["final3"][m], "best_final3_lr": bf,
               "best_none_final3": NN[bf]["final3"][m]}
        sd = lambda a, k, lr: {s: per[a][s][k][m] - per[f"none@{lr}"][s][k][m] for s in REP_SEEDS   # noqa: E731
                               if s in per[a] and s in per[f"none@{lr}"]}
        if A6:
            row.update(all6_area=A6["area"][m], all6_final3=A6["final3"][m], d_area=A6["area"][m] - NN[nm]["area"][m],
                       d_final3=A6["final3"][m] - NN[nm]["final3"][m], seed_d_area=sd("all6", "area", nm), seed_d_final3=sd("all6", "final3", nm))
            pos["area"].append(row["d_area"] > 0)
            pos["final3"].append(row["d_final3"] > 0)
        if FU:
            row.update(full_final3=FU["final3"][m], d_full_vs_bestf3=FU["final3"][m] - NN[bf]["final3"][m], seed_d_full=sd("all6full", "final3", bf))
            pos["full_final3"].append(row["d_full_vs_bestf3"] > 0)
        few = sorted({c for c in ("all6", f"none@{nm}", "all6full", f"none@{bf}") if c in mean and mean[c]["n"] < nmax})
        row["fewer_seeds"] = few   # REP 190M: cells of this row read with fewer seeds than the most-seeded cell
        ro["mean"].setdefault("manifolds", {})[m] = row
        g = row.get
        ps = lambda d: " ".join(f"{v:+.1f}" for _, v in sorted((d or {}).items())) + f" ({sum(v > 0 for v in (d or {}).values())}/{len(d or {})}+)"   # noqa: E731
        fs = (" <n " + ",".join(f"{c}(n={mean[c]['n']})" for c in few)) if few else ""
        print(f"{m:9s} | {nm:>7s} {edge:>4s}{fs} | {f1(g('all6_area')):>9s} {f1(g('none_area')):>9s} {f1(g('d_area')):>6s} {ps(g('seed_d_area')):>22s} | "
              f"{f1(g('all6_final3')):>7s} {f1(g('none_final3')):>7s} {f1(g('d_final3')):>6s} {ps(g('seed_d_final3')):>22s} | "
              f"{f1(g('best_none_final3'))} @{bf:>7s} | {f1(g('full_final3')):>7s} {f1(g('d_full_vs_bestf3')):>6s} {ps(g('seed_d_full')):>22s}")
    ro["mean"]["cells"] = mean
    ro["mean"]["all_six_positive"] = {k: (len(v) == 6 and all(v)) for k, v in pos.items()}
    print(f"seed mean, all six positive: all6 - none@N_m area {ro['mean']['all_six_positive']['area']}, final3 "
          f"{ro['mean']['all_six_positive']['final3']}; all6full - best none final3 {ro['mean']['all_six_positive']['full_final3']}")
    if A6 and OD and "only-dim" in bd:   # REP 1B: only-dim vs all6 on the seeds both have
        ps_ = sorted(set(per["only-dim"]) & set(per["all6"]))
        oa = {k: float(np.mean([per["only-dim"][s][k]["dim"] for s in ps_])) for k in ("area", "final3")}
        aa = {k: float(np.mean([per["all6"][s][k]["dim"] for s in ps_])) for k in ("area", "final3")}
        ro["mean"]["only_dim_vs_all6_dim"] = {"area": oa["area"] - aa["area"], "final3": oa["final3"] - aa["final3"], "seeds": ps_}
        print(f"only-dim vs all6 on dim, seeds both have {ps_}: area {oa['area']:.1f} vs {aa['area']:.1f}; final3 {oa['final3']:.1f} vs {aa['final3']:.1f}")
    elif A6 and OD:
        ro["mean"]["only_dim_vs_all6_dim"] = {"area": OD["area"]["dim"] - A6["area"]["dim"], "final3": OD["final3"]["dim"] - A6["final3"]["dim"]}
        print(f"seed mean only-dim vs all6 on dim: area {OD['area']['dim']:.1f} vs {A6['area']['dim']:.1f}; final3 {OD['final3']['dim']:.1f} vs {A6['final3']['dim']:.1f}")
    print("cells: n seeds | mean bpb   ('<n' = fewer seeds than the most-seeded cell)")
    for c, v in mean.items():
        print(f"   {c:14s} {v['n']} {v['seeds']} | {v['bpb']:.4f}" + ("  <n" if v["n"] < nmax else ""))
    return ro


if "--rep" in sys.argv:   # 505M (out["rep"], the REP 505M layout) then 190M (out["rep_190m"]) then 1.07B (out["rep_1b"]), same layout
    out["rep"] = rep_read("505m")
    out["rep_190m"] = rep_read("190m")
    out["rep_1b"] = rep_read("1b")
if "--rep" in sys.argv:   # SAMELR (2026-11-25): same-lr bpb, all6@6e-4 - none@6e-4, seeds 16-19, after the REP blocks
    out["samelr_bpb"] = {}
    for z in ("190m", "505m", "1b"):
        ps = {s: {a: load(S / z / f"{a}-lr6e-4-s{s}") for a in ("all6", "none")} for s in (16, 17, 18, 19)}
        row = {s: {a: (None if v[a] is None else v[a]["bpb"]) for a in v} for s, v in ps.items()}
        both = [s for s, v in row.items() if None not in v.values()]
        dd = {s: row[s]["all6"] - row[s]["none"] for s in both}
        zo = {"per_seed": row, "diff": dd, "seeds_both": both,
              "mean": None if not both else {"all6": float(np.mean([row[s]["all6"] for s in both])),
                                             "none": float(np.mean([row[s]["none"] for s in both])),
                                             "diff": float(np.mean(list(dd.values())))}}
        out["samelr_bpb"][z] = zo
        print(f"\n######## {z} same-lr bpb (SAMELR): all6@6e-4 vs none@6e-4, seeds 16-19 (lower = better; exploratory)")
        for s, v in row.items():
            print(f"   seed {s}: all6 {'pending' if v['all6'] is None else format(v['all6'], '.4f')} | none "
                  f"{'pending' if v['none'] is None else format(v['none'], '.4f')}" + (f" | diff {dd[s]:+.4f}" if s in dd else ""))
        m = zo["mean"]
        print(f"   mean over seeds both arms have {both}: " + ("n/a" if m is None else
              f"all6 {m['all6']:.4f} none {m['none']:.4f} diff {m['diff']:+.4f} ({sum(x < 0 for x in dd.values())}/{len(dd)} seeds all6 lower)"))
for z, cfg in ({} if "--rep" in sys.argv else SIZES).items():
    out["sizes"][z] = read_size(z, cfg)

if KNOWN:   # known answers: v8read.json (final3 of six domains + bpb) for all6@2e-3 / only-dim@2e-3 / none@3.2e-2 seed 16
    kj = json.loads((R / "out_v8/v8read.json").read_text())["runs"]
    zo, bad = out["sizes"]["53m(known)"], []
    for a, key in (("all6", "all6-x1-s16"), ("only-dim", "only-dim-x1-s16"), ("none@3.2e-2", "none-x1-s16")):
        mine, ref = zo["runs"][a], kj[key]
        for m in DOMS:
            if abs(mine["final3"][m] - ref[f"{m}_f3"]) > 1e-9:
                bad.append(f"{a} {m} final3 {mine['final3'][m]} != {ref[f'{m}_f3']}")
        if abs(mine["bpb"] - ref["bpb"]) > 1e-12:
            bad.append(f"{a} bpb {mine['bpb']} != {ref['bpb']}")
    am = max(v["area_vs_metrics_max"] for v in zo["runs"].values() if v["area_vs_metrics_max"] is not None)
    print(f"\nKNOWN-ANSWER: final3 x 6 + bpb of 3 runs vs v8read.json: {'MATCH' if not bad else 'MISMATCH ' + '; '.join(bad)}; "
          f"max |own area - run metrics area| over runs {am:.2e}")
    out["known_answer"] = {"mismatch": bad, "area_vs_metrics_max": am}
print("\nreading rule: " + RULE)
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1) + "\n")
