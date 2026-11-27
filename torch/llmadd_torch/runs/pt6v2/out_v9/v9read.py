#!/usr/bin/env python3
"""pt6v2 Amendment V9 read (post-hoc, exploratory; NO check gates it). Same-count, dose-matched control for the joint lift on dim:
is h6 = all6 - only-dim (dim strict) due to the torus / box5 GEOMETRY or to the extra prior gradient?
Arms (lr 2e-3, seeds 16..23, paired by seed): all6 (out/), only-dim (out/), all4 = no torus / box5 priors (out_v4/),
all6-shufTB = all6 with torus / box5 given shuffled targets (out_v9/, --prior_shuf torus,box5).
Metrics (v4final.py / v8read.py definitions; v4final's prefix exec'd read only): dim strict = hits_dim_held over dim_strict(D)'s keep
mask; area = 100 * mean over every eval; final3 = 100 * mean of the last 3 evals; torus / box5 = TOY_KEY; cities / countries =
<d>_held_2way; numbers = mean held_<b>. Gradient dose over the prior-on evals: median grad_prior (total weighted prior gradient),
median of grad_lm / (grad_lm + grad_prior) (v4final's LM share), median gnorm_prior_<d> of torus / box5.
Contrasts (mean [95% t-CI] wins k/n): h6 = all6 - only, h4 = all4 - only, h_shuf = shufTB - only, all6 - shufTB (geometry beyond the
dose), shufTB - all4 (what the shuffled torus / box5 priors add over no torus / box5 prior).
Known answer (v4final_out.txt line 43, 8 seeds): dim AREA all4 65.6 all6 77.6 only 64.1, h4 +1.5, h6 +13.5; FINAL3 all4 55.3,
all6 77.3, only 52.3.
usage: v9read.py [--test-alias] [--json out.json]   (--test-alias: SMOKE ONLY, shufTB := all6, so h_shuf must equal h6)"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

V4F = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2/tests/posthoc_speed_20261120/v4final.py")
_src = V4F.read_text()
G = {"__name__": "v4final_prefix", "__file__": str(V4F)}
exec(compile(_src[:_src.index("# ---------------------------------------------------------------- load every run")], str(V4F), "exec"), G)
dim_strict, BKT, TOY_KEY = G["dim_strict"], G["BKT"], G["TOY_KEY"]
R = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2")
TEST = "--test-alias" in sys.argv
SEEDS = tuple(range(16, 24))
DOMS = ("numbers", "dim", "cities", "countries", "torus", "box5")
RULE = ("Exploratory, no gate. Lift from the torus / box5 geometry if h_shuf (all6-shufTB - only-dim, dim strict area) is close to "
        "h4 and all6 - all6-shufTB > 0 in most seeds; lift from the prior dose if h_shuf is close to h6.")
DIRS = {"all6": lambda s: R / "out" / f"pt6v2-all6-lr2e-3-s{s}",
        "only": lambda s: R / "out" / f"pt6v2-only-dim-lr2e-3-s{s}",
        "all4": lambda s: R / "out_v4" / f"pt6v2-all4-lr2e-3-s{s}",
        "shufTB": (lambda s: R / "out" / f"pt6v2-all6-lr2e-3-s{s}") if TEST else (lambda s: R / "out_v9" / f"pt6v2-all6shufTB-lr2e-3-s{s}")}


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
    D = r["domains"]["dim"]
    keep, _, _ = dim_strict(D)
    H = np.array([[int(x) for x in e["hits_dim_held"]] for e in c], float) / D["eval_held"]["contexts"]
    S = {"numbers": np.array([np.mean([e[f"held_{b}"] for b in BKT]) for e in c]), "dim": H[:, keep].mean(1)}
    for dd in ("cities", "countries"):
        S[dd] = np.array([e[f"{dd}_held_2way"] for e in c])
    for dd in ("torus", "box5"):
        S[dd] = np.array([e[TOY_KEY[dd]] for e in c])
    on = [e for e in c if e.get("prior_on") and e.get("grad_prior") is not None]
    med = lambda xs: float(np.median(xs)) if xs else None   # noqa: E731
    g = {"grad_prior": med([e["grad_prior"] for e in on]), "grad_prior_mean": float(np.mean([e["grad_prior"] for e in on])) if on else None,
         "share": med([e["grad_lm"] / (e["grad_lm"] + e["grad_prior"]) for e in on]),
         "grad_lm_share": med([e["grad_lm_share"] for e in on if e.get("grad_lm_share") is not None]), "n_on": len(on)}
    for dd in ("torus", "box5"):
        g[f"gp_{dd}"] = med([e[f"gnorm_prior_{dd}"] for e in on if e.get(f"gnorm_prior_{dd}") is not None])
    flags = []
    f = d / "check.json"
    if f.exists():
        k = json.loads(f.read_text())
        if not k.get("pass"):
            flags.append("check.json: " + "; ".join(x["check"][:50] for x in k["checks"] if not x["ok"])[:200])
    shuf = sorted(x for x in DOMS if r["domains"].get(x, {}).get("shuf"))
    return {"S": S, "g": g, "dev": device(d), "flags": flags, "shuf": shuf, "path": str(fs[0])}


area = lambda x: 100 * float(np.mean(x))   # noqa: E731
f3 = lambda x: 100 * float(np.mean(x[-3:]))   # noqa: E731


def summ(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) < 2:
        return f"n={len(x)} " + (f"{x[0]:+6.1f}" if len(x) else "")
    m, h = x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return f"{m:+6.1f} [{m - h:+6.1f}, {m + h:+6.1f}] wins {int((x > 0).sum())}/{len(x)}"


DB = {(a, s): load(DIRS[a](s)) for a in DIRS for s in SEEDS}
miss = sorted(f"{a}-s{s}" for (a, s), v in DB.items() if v is None)
print(f"V9 read{' (TEST ALIAS: shufTB := all6)' if TEST else ''} | runs loaded {sum(v is not None for v in DB.values())}/{len(DB)} "
      f"| missing / incomplete: {miss or 'none'}")
OK = [s for s in SEEDS if all(DB[(a, s)] is not None for a in DIRS)]
print(f"seeds with all four arms: {OK}")
out = {"seeds": OK, "rule": RULE, "per_seed": {}, "summary": {}}
for mname, mf in (("AREA", area), ("FINAL3", f3)):
    print(f"\n== dim strict {mname}: per seed (all6 / shufTB / all4 / only) and contrasts")
    print(f"{'seed':>4} {'all6':>6} {'shufTB':>6} {'all4':>6} {'only':>6} | {'h6':>6} {'h4':>6} {'h_shuf':>6} {'a6-shuf':>7} {'shuf-a4':>7}")
    rows = []
    for s in OK:
        v = {a: mf(DB[(a, s)]["S"]["dim"]) for a in DIRS}
        c = {"h6": v["all6"] - v["only"], "h4": v["all4"] - v["only"], "h_shuf": v["shufTB"] - v["only"],
             "all6-shufTB": v["all6"] - v["shufTB"], "shufTB-all4": v["shufTB"] - v["all4"]}
        rows.append((v, c))
        out["per_seed"].setdefault(mname, {})[s] = {**v, **c}
        print(f"{s:>4} {v['all6']:6.1f} {v['shufTB']:6.1f} {v['all4']:6.1f} {v['only']:6.1f} | {c['h6']:+6.1f} {c['h4']:+6.1f} "
              f"{c['h_shuf']:+6.1f} {c['all6-shufTB']:+7.1f} {c['shufTB-all4']:+7.1f}")
    if rows:
        print("means: " + "  ".join(f"{a} {np.mean([r[0][a] for r in rows]):.1f}" for a in DIRS))
        for k in ("h6", "h4", "h_shuf", "all6-shufTB", "shufTB-all4"):
            print(f"  {k:12s} {summ([r[1][k] for r in rows])}")
            out["summary"][f"{mname}:{k}"] = float(np.mean([r[1][k] for r in rows]))
        hs, h4, h6 = (np.mean([r[1][k] for r in rows]) for k in ("h_shuf", "h4", "h6"))
        print(f"  h_shuf {hs:+.1f}: distance to h4 {abs(hs - h4):.1f}, to h6 {abs(hs - h6):.1f}; position (h_shuf - h4) / (h6 - h4) = "
              f"{(hs - h4) / (h6 - h4) if h6 != h4 else float('nan'):.2f} (0 = h4, geometry; 1 = h6, dose)")
print("\n== final3 of every domain (mean over the seeds above): all6 / shufTB / all4 / only (torus / box5 under shufTB: the shuffled "
      "target replaces the true one, so their own scores are expected to move)")
for dd in DOMS:
    print(f"  {dd:10s} " + "  ".join(f"{a} {np.mean([f3(DB[(a, s)]['S'][dd]) for s in OK]):5.1f}" for a in DIRS))
print("\n== gradient dose over prior-on evals (per-seed medians, then mean over seeds): total prior gradient grad_prior, LM share "
      "grad_lm / (grad_lm + grad_prior), the torus / box5 prior gradient norms (audit_v4: 0.143 all6 vs 0.095 all4)")
for a in DIRS:
    gs = [DB[(a, s)]["g"] for s in OK]
    if not gs:
        continue
    fmt = lambda k: f"{np.mean([x[k] for x in gs if x[k] is not None]):.4f}" if any(x[k] is not None for x in gs) else "-"   # noqa: E731
    print(f"  {a:7s} grad_prior median {fmt('grad_prior')} (mean of means {fmt('grad_prior_mean')}) | LM share {fmt('share')} "
          f"(trainer grad_lm_share {fmt('grad_lm_share')}) | gnorm_prior torus {fmt('gp_torus')} box5 {fmt('gp_box5')} | n_on {fmt('n_on')}")
    out["summary"][f"dose:{a}"] = {k: float(np.mean([x[k] for x in gs if x[k] is not None])) for k in gs[0] if any(x[k] is not None for x in gs)}
print("\n== flags per run (device, check.json, shuffled domains)")
for s in OK:
    for a in DIRS:
        r = DB[(a, s)]
        exp = ["box5", "torus"] if a == "shufTB" and not TEST else []
        fl = ([] if r["shuf"] == exp else [f"shuf domains {r['shuf']} != {exp}"]) + r["flags"]
        if fl or a == "shufTB":
            print(f"  {a}-s{s}: device {r['dev']}; {'; '.join(fl) or 'no flag'}")
print(f"\nreading rule (AUDIT.md Amendment V9): {RULE}")
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1))
