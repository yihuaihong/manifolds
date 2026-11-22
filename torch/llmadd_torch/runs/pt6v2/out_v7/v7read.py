#!/usr/bin/env python3
"""pt6v2 Amendment V7 descriptive read (post-hoc, exploratory, NO verdict rule, no gate).
all6drop@2e-3 (out_v7/) vs the registered all6@2e-3 and only-dim@2e-3 (out/), paired by seed 16..23:
  1. dim strict final3 / area (the v4final.py definitions: hits_dim_held over dim_strict(D)'s keep mask; area = 100 * mean over
     the curve; final3 = 100 * mean of the last 3 evals)
  2. dim's drop from the window's close (the eval at step 7250, the last eval before the drop at 7324) to the end (final3)
  3. torus / box5 held-out accuracy after the drop (mean over the evals >= 7324) and final3 (all6drop vs all6)
  4. numbers / cities / countries final3 (a damage check)
  5. the device per run (gpu.csv) and check_v7.json / check.json status: flags, not gates.
Missing runs are listed and skipped (paired stats over the seeds present in every compared arm).
usage: v7read.py [--json <out.json>]"""
import json
import math
import sys
from pathlib import Path

import numpy as np

V4F = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2/tests/posthoc_speed_20261120/v4final.py")
src = V4F.read_text()
cut = src.index("# ---------------------------------------------------------------- load every run")
G = {"__name__": "v4final_prefix", "__file__": str(V4F)}
exec(compile(src[:cut], str(V4F), "exec"), G)   # constants, dim_strict, device (read only; no top-level loading)
dim_strict, BKT, TOY_KEY, SEEDS = G["dim_strict"], G["BKT"], G["TOY_KEY"], G["SEEDS"]


def device(d):   # = v4final.gpu_names (defined after the prefix): the GPU name column of gpu.csv
    f = d / "gpu.csv"
    return ",".join(sorted({ln.split(",")[1].strip() for ln in f.read_text().splitlines()[1:] if ln.count(",") >= 1})) if f.exists() else None
R = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2")
DROP_AT, WCLOSE = 7324, 7250
ARMS = {"all6drop": lambda s: R / "out_v7" / f"pt6v2-all6drop-lr2e-3-s{s}",
        "all6": lambda s: R / "out" / f"pt6v2-all6-lr2e-3-s{s}",
        "only-dim": lambda s: R / "out" / f"pt6v2-only-dim-lr2e-3-s{s}"}


def load(d):
    fs = sorted(d.glob("*/run_whole_*.json"))
    if len(fs) != 1:
        return None
    r = json.loads(fs[0].read_text())
    c = r["curve"]
    if not c or c[-1]["step"] != r["steps"] - 1:
        return None
    st = np.array([e["step"] for e in c])
    S = {"numbers": np.array([np.mean([e[f"held_{b}"] for b in BKT]) for e in c])}
    D = r["domains"]["dim"]
    keep, _, _ = dim_strict(D)
    H = np.array([[int(x) for x in e["hits_dim_held"]] for e in c], float) / D["eval_held"]["contexts"]
    S["dim"] = H[:, keep].mean(1)
    for dd in ("cities", "countries"):
        S[dd] = np.array([e[f"{dd}_held_2way"] for e in c])
    for dd in ("torus", "box5"):
        S[dd] = np.array([e[TOY_KEY[dd]] for e in c])
    flags = []
    for ck in ("check_v7.json", "check.json"):
        f = d / ck
        if f.exists():
            k = json.loads(f.read_text())
            if not k.get("pass"):
                flags.append(f"{ck}: " + "; ".join(x["check"][:50] for x in k["checks"] if not x["ok"])[:200])
    return {"st": st, "S": S, "dev": device(d), "flags": flags, "drop": r.get("drop_domains_after")}


def f3(x):
    return 100 * float(np.mean(x[-3:]))


def at(rec, d, step):
    i = int(np.where(rec["st"] == step)[0][0])
    return 100 * float(rec["S"][d][i])


RUN, missing = {}, []
for a, p in ARMS.items():
    for s in SEEDS:
        x = load(p(s))
        if x is None:
            missing.append(f"{a} s{s}")
        else:
            RUN[(a, s)] = x
print("missing / unfinished:", missing or "none")
row = {}
for (a, s), x in sorted(RUN.items()):
    post = x["st"] >= DROP_AT
    row[(a, s)] = {"dim_f3": f3(x["S"]["dim"]), "dim_area": 100 * float(np.mean(x["S"]["dim"])),
                   "dim_wclose": at(x, "dim", WCLOSE), "dim_drop": at(x, "dim", WCLOSE) - f3(x["S"]["dim"]),
                   **{f"{d}_post": 100 * float(np.mean(x["S"][d][post])) for d in ("torus", "box5")},
                   **{f"{d}_f3": f3(x["S"][d]) for d in ("torus", "box5", "numbers", "cities", "countries")},
                   "dev": x["dev"], "flags": x["flags"]}
cols = ["dim_f3", "dim_area", "dim_wclose", "dim_drop", "torus_post", "box5_post", "torus_f3", "box5_f3", "numbers_f3",
        "cities_f3", "countries_f3"]
print(f"{'arm':9s} {'seed':>4s} " + " ".join(f"{c:>12s}" for c in cols) + "  device / flags")
for (a, s), v in sorted(row.items()):
    print(f"{a:9s} {s:4d} " + " ".join(f"{v[c]:12.1f}" for c in cols) + f"  {v['dev']} {v['flags'] or ''}")


def paired(a, b, c):
    ss = [s for s in SEEDS if (a, s) in row and (b, s) in row]
    if not ss:
        return None
    d = np.array([row[(a, s)][c] - row[(b, s)][c] for s in ss])
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return {"n": len(ss), "mean": float(d.mean()), "sd": sd, "pos": int((d > 0).sum()), "neg": int((d < 0).sum()),
            "per_seed": dict(zip(ss, d.round(2).tolist()))}


print("\npaired by seed (A - B, points): mean, sd, sign counts")
OUTJ = {"rows": {f"{a}/s{s}": v for (a, s), v in row.items()}, "paired": {}}
for a, b in (("all6drop", "all6"), ("all6drop", "only-dim"), ("all6", "only-dim")):
    for c in ("dim_f3", "dim_area", "dim_drop", "torus_post", "box5_post", "numbers_f3", "cities_f3", "countries_f3"):
        p = paired(a, b, c)
        OUTJ["paired"][f"{a}-{b}/{c}"] = p
        if p:
            print(f"  {a:8s} - {b:8s} {c:13s} n={p['n']} {p['mean']:+7.2f} (sd {p['sd']:.2f}) +{p['pos']}/-{p['neg']}")
g = paired("all6", "only-dim", "dim_f3"); h = paired("all6drop", "only-dim", "dim_f3")
if g and h and not math.isnan(g["mean"]):
    print(f"\nshare of all6's dim final3 advantage over only-dim kept by all6drop: {h['mean'] / g['mean']:.2f} "
          f"(seeds: all6 {g['n']}, all6drop {h['n']}); descriptive, no verdict rule")
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(OUTJ, indent=1))
