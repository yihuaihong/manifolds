#!/usr/bin/env python3
"""pt6v2 Amendment V8 read (post-hoc, exploratory; NO check gates it). Longer training: does a model without the joint priors catch
up with all6@2e-3 1x (18,310 steps) at 2x / 4x the tokens?
Arms: all6@2e-3 1x (out/, seeds 16..23), only-dim@2e-3 x1 (out/) / x2 / x4 (out_v8/), none@3.2e-2 (N_m, the area-best none lr for
dim; audit_speed_20261120.md) x1 (out/) / x2 / x4 (out_v8/). The x2 / x4 runs have seeds 16, 17 (paired with the registered runs).
Metrics (the v4final.py / v7read.py definitions, its prefix exec'd read only):
  dim strict = hits_dim_held over dim_strict(D)'s keep mask; dim held = hits_dim_held over every held question;
  final3 = 100 * mean of the last 3 evals; peak = 100 * max over the evals (step in brackets);
  numbers = mean held_<b> over the 6 buckets; cities / countries = <d>_held_2way; torus / box5 = TOY_KEY; bpb = the last eval's
  val_bits_per_byte.
Tables:
  1. dim strict / held, final3 and peak: per seed and means (all6 1x: 8 seeds, range, and s16 / s17 alone);
  2. paired gap all6 1x - X (s16 / s17 and their mean), strict and held final3;
  3. inside the longer runs at the 1x step count: the eval at 18250 (the last eval <= 18309; the long runs evaluate every 250 steps,
     so 18309 itself is not an eval) and the mean of the 3 evals 17750 / 18000 / 18250 (the 1x final3's position), with the stretched
     schedule (LR at step 18309 is higher than in the 1x run);
  4. the first eval step at which each run reaches all6 1x's final3 (strict dim): single eval >= the 8-seed mean, and the 3-eval
     running mean >= the 8-seed mean / >= the same seed's all6 1x final3; 'never' if it does not;
  5. final3 of all six domains + bpb: none x1 / x2 / x4 and only-dim x1 / x2 / x4 vs all6 1x (s16 / s17 means; all6 also 8 seeds);
  6. device / check flags per run.
The reading rule (AUDIT.md Amendment V8, verbatim) is printed with the numbers; it is applied by the reader.
usage: v8read.py [--json <out.json>] [--test-alias]   (--test-alias: SMOKE ONLY, every x2 / x4 run := the registered x1 run of its
arm and seed, so the long-run tables run on known data; the x2 / x4 rows must then equal the x1 rows)"""
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
TEST = "--test-alias" in sys.argv
ALL6_SEEDS, V8_SEEDS, STEP1 = tuple(range(16, 24)), (16, 17), 18309
DOMS = ("numbers", "dim", "cities", "countries", "torus", "box5")
RULE = ("Report each arm's final3 strict dim at x2 / x4 against all6 1x; 'catches up' = mean >= all6 1x mean with both seeds within "
        "all6's 8-seed range; no gate.")
REG = {"all6": "pt6v2-all6-lr2e-3", "only-dim": "pt6v2-only-dim-lr2e-3", "none": "pt6v2-none-lr3.2e-2"}


def rdir(arm, k, s):
    if k == 1 or TEST:
        return R / "out" / f"{REG[arm]}-s{s}"
    return R / "out_v8" / f"{REG[arm]}-x{k}-s{s}"


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
    S = {"numbers": np.array([np.mean([e[f"held_{b}"] for b in BKT]) for e in c]),
         "dim": H[:, keep].mean(1), "dim_held": H.mean(1)}
    for dd in ("cities", "countries"):
        S[dd] = np.array([e[f"{dd}_held_2way"] for e in c])
    for dd in ("torus", "box5"):
        S[dd] = np.array([e[TOY_KEY[dd]] for e in c])
    flags = []
    f = d / "check.json"
    if f.exists():
        k = json.loads(f.read_text())
        if not k.get("pass"):
            flags.append("check.json: " + "; ".join(x["check"][:50] for x in k["checks"] if not x["ok"])[:200])
    xh = float(np.max(np.abs(S["dim_held"] - np.array([e["dim_held"] for e in c])))) if "dim_held" in c[0] else None
    return {"st": st, "S": S, "steps": r["steps"], "bpb": float(c[-1]["val_bits_per_byte"]), "dev": device(d), "flags": flags,
            "events": r.get("prior_events"), "held_xcheck": xh, "path": str(fs[0])}


f3 = lambda x: 100 * float(np.mean(x[-3:]))   # noqa: E731


def peak(rec, m):
    i = int(np.argmax(rec["S"][m]))
    return 100 * float(rec["S"][m][i]), int(rec["st"][i])


def at(rec, m, step):
    w = np.where(rec["st"] == step)[0]
    return 100 * float(rec["S"][m][w[0]]) if len(w) else None


def first_reach(rec, m, thr, roll):
    v = 100 * rec["S"][m]
    if roll > 1:
        v = np.convolve(v, np.ones(roll) / roll, mode="valid")
        st = rec["st"][roll - 1:]
    else:
        st = rec["st"]
    w = np.where(v >= thr)[0]
    return int(st[w[0]]) if len(w) else None


RUN, missing = {}, []
cells = [("all6", 1, ALL6_SEEDS)] + [(a, k, ALL6_SEEDS if k == 1 else V8_SEEDS) for a in ("only-dim", "none") for k in (1, 2, 4)]
for arm, k, seeds in cells:
    for s in seeds:
        rec = load(rdir(arm, k, s))
        if rec is None:
            missing.append(f"{arm} x{k} s{s} ({rdir(arm, k, s)})")
        else:
            RUN[(arm, k, s)] = rec
fmt = lambda v: "   n/a" if v is None else f"{v:6.1f}"   # noqa: E731
print(f"V8 read{' (TEST ALIAS: x2 / x4 := the registered x1 runs; smoke only)' if TEST else ''}; runs loaded {len(RUN)}; "
      f"missing {len(missing)}" + ("".join(f"\n  missing: {m}" for m in missing)))
xc = [r["held_xcheck"] for r in RUN.values() if r["held_xcheck"] is not None]
if xc:
    print(f"dim held = mean of hits_dim_held over every held question; max |it - the logged dim_held| over runs {max(xc):.2e}")
out = {"rule": RULE, "test_alias": TEST, "missing": missing, "runs": {}, "summary": {}}

# all6 1x reference
a6 = [RUN[("all6", 1, s)] for s in ALL6_SEEDS if ("all6", 1, s) in RUN]
REF = {}
for m in ("dim", "dim_held"):
    v = np.array([f3(r["S"][m]) for r in a6]) if a6 else np.array([])
    REF[m] = {"mean8": float(v.mean()) if len(v) else None, "min8": float(v.min()) if len(v) else None,
              "max8": float(v.max()) if len(v) else None, "n": len(v),
              **{f"s{s}": f3(RUN[("all6", 1, s)]["S"][m]) for s in V8_SEEDS if ("all6", 1, s) in RUN}}
out["summary"]["all6_1x"] = REF

print("\n== 1. dim, final3 and peak (strict = dim_strict keep mask; held = every held question)")
print(f"{'arm':10s} {'x':>2s} {'seed':>4s} {'steps':>6s} {'strict_f3':>9s} {'strict_pk':>9s} {'@step':>6s} {'held_f3':>8s} {'held_pk':>8s} {'@step':>6s}")
for arm, k, seeds in cells:
    for s in seeds:
        r = RUN.get((arm, k, s))
        if r is None:
            continue
        ps, pss = peak(r, "dim")
        ph, phs = peak(r, "dim_held")
        row = {"strict_f3": f3(r["S"]["dim"]), "strict_peak": ps, "strict_peak_step": pss, "held_f3": f3(r["S"]["dim_held"]),
               "held_peak": ph, "held_peak_step": phs, "steps": r["steps"], "bpb": r["bpb"], "dev": r["dev"], "flags": r["flags"],
               "events": r["events"], **{f"{d}_f3": f3(r["S"][d]) for d in DOMS}}
        out["runs"][f"{arm}-x{k}-s{s}"] = row
        if arm == "all6" or k > 1 or s in V8_SEEDS:
            print(f"{arm:10s} {k:2d} {s:4d} {r['steps']:6d} {row['strict_f3']:9.1f} {ps:9.1f} {pss:6d} {row['held_f3']:8.1f} {ph:8.1f} {phs:6d}")


def mean_of(arm, k, key, seeds):
    v = [out["runs"][f"{arm}-x{k}-s{s}"][key] for s in seeds if f"{arm}-x{k}-s{s}" in out["runs"]]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


print("\n   means (strict_f3 / strict_peak / held_f3 / held_peak; n seeds)")
for arm, k, seeds in cells:
    for lab, ss in ([("8 seeds", seeds)] if k == 1 else []) + [("s16/s17", V8_SEEDS)]:
        vals = [mean_of(arm, k, key, ss) for key in ("strict_f3", "strict_peak", "held_f3", "held_peak")]
        if vals[0][1]:
            print(f"   {arm:10s} x{k} {lab:8s}: " + " / ".join(fmt(v) for v, _ in vals) + f"  (n {vals[0][1]})")
            out["summary"][f"{arm}-x{k}-{lab}"] = {key: v for key, (v, _) in zip(("strict_f3", "strict_peak", "held_f3", "held_peak"), vals)}
if REF["dim"]["n"]:
    print(f"   all6 1x strict final3: 8-seed mean {REF['dim']['mean8']:.1f}, range [{REF['dim']['min8']:.1f}, {REF['dim']['max8']:.1f}]; "
          f"held final3 mean {REF['dim_held']['mean8']:.1f}, range [{REF['dim_held']['min8']:.1f}, {REF['dim_held']['max8']:.1f}]")

print("\n== 2. paired gap all6 1x - X (final3, pts; per seed and mean over s16 / s17)")
for arm in ("only-dim", "none"):
    for k in (1, 2, 4):
        g = {}
        for m, key in (("dim", "strict_f3"), ("dim_held", "held_f3")):
            g[m] = {s: out["runs"]["all6-x1-s%d" % s][key] - out["runs"][f"{arm}-x{k}-s{s}"][key]
                    for s in V8_SEEDS if f"all6-x1-s{s}" in out["runs"] and f"{arm}-x{k}-s{s}" in out["runs"]}
        if g["dim"]:
            print(f"   {arm:9s} x{k}: strict " + ", ".join(f"s{s} {v:+.1f}" for s, v in g["dim"].items()) + f"; mean {np.mean(list(g['dim'].values())):+.1f}"
                  + " | held " + ", ".join(f"s{s} {v:+.1f}" for s, v in g["dim_held"].items()) + f"; mean {np.mean(list(g['dim_held'].values())):+.1f}")
            out["summary"][f"gap_all6-{arm}-x{k}"] = {m: {f"s{s}": v for s, v in g[m].items()} for m in g}

print(f"\n== 3. inside the longer runs at the 1x step count (eval at 18250 = the last eval <= {STEP1}; f3@1x = mean of 17750 / 18000 / 18250)")
for arm in ("only-dim", "none"):
    for k in (1, 2, 4):
        for s in V8_SEEDS:
            r = RUN.get((arm, k, s))
            if r is None:
                continue
            v = {m: (at(r, m, 18250), None if any(at(r, m, x) is None for x in (17750, 18000, 18250))
                     else float(np.mean([at(r, m, x) for x in (17750, 18000, 18250)]))) for m in ("dim", "dim_held")}
            e18309 = at(r, "dim", STEP1)
            print(f"   {arm:9s} x{k} s{s}: strict @18250 {fmt(v['dim'][0])}, f3@1x {fmt(v['dim'][1])}"
                  + (f", @18309 {e18309:.1f}" if e18309 is not None else "") + f" | held @18250 {fmt(v['dim_held'][0])}, f3@1x {fmt(v['dim_held'][1])}"
                  f" | own final3 strict {f3(r['S']['dim']):.1f}")
            out["runs"][f"{arm}-x{k}-s{s}"].update(strict_at18250=v["dim"][0], strict_f3_at1x=v["dim"][1], held_at18250=v["dim_held"][0],
                                                     held_f3_at1x=v["dim_held"][1])

print("\n== 4. first eval step reaching all6 1x's strict final3 (single eval >= 8-seed mean | 3-eval mean >= 8-seed mean | "
      "3-eval mean >= same seed's all6)")
if REF["dim"]["n"]:
    for arm in ("only-dim", "none"):
        for k in (1, 2, 4):
            for s in V8_SEEDS:
                r = RUN.get((arm, k, s))
                if r is None:
                    continue
                own = REF["dim"].get(f"s{s}")
                fr = (first_reach(r, "dim", REF["dim"]["mean8"], 1), first_reach(r, "dim", REF["dim"]["mean8"], 3),
                      first_reach(r, "dim", own, 3) if own is not None else None)
                print(f"   {arm:9s} x{k} s{s} (of {r['steps']} steps): " + " | ".join("never" if x is None else str(x) for x in fr)
                      + f"   [targets {REF['dim']['mean8']:.1f} / {REF['dim']['mean8']:.1f} / {fmt(own).strip()}]")
                out["runs"][f"{arm}-x{k}-s{s}"].update(first_reach_mean8_single=fr[0], first_reach_mean8_f3=fr[1], first_reach_own_f3=fr[2])

print("\n== 5. final3 of all six domains + final bpb (means over s16 / s17; all6 also 8 seeds)")
print(f"   {'arm':14s} " + " ".join(f"{d:>9s}" for d in DOMS) + f" {'bpb':>7s}  n")
rows5 = [("all6", 1, ALL6_SEEDS, "all6 1x (8)"), ("all6", 1, V8_SEEDS, "all6 1x s16/17")] + \
        [(a, k, V8_SEEDS, f"{a} x{k}") for a in ("none", "only-dim") for k in (1, 2, 4)]
for arm, k, ss, lab in rows5:
    rr = [out["runs"][f"{arm}-x{k}-s{s}"] for s in ss if f"{arm}-x{k}-s{s}" in out["runs"]]
    if not rr:
        print(f"   {lab:14s} (missing)")
        continue
    mv = {d: float(np.mean([x[f"{d}_f3"] for x in rr])) for d in DOMS}
    bp = float(np.mean([x["bpb"] for x in rr]))
    print(f"   {lab:14s} " + " ".join(f"{mv[d]:9.1f}" for d in DOMS) + f" {bp:7.4f}  {len(rr)}")
    out["summary"][f"six_{lab}"] = {**mv, "bpb": bp, "n": len(rr)}
for arm in ("none", "only-dim"):
    for k in (1, 2, 4):
        dd = {d: [out["runs"][f"all6-x1-s{s}"][f"{d}_f3"] - out["runs"][f"{arm}-x{k}-s{s}"][f"{d}_f3"] for s in V8_SEEDS
                  if f"all6-x1-s{s}" in out["runs"] and f"{arm}-x{k}-s{s}" in out["runs"]] for d in DOMS}
        if dd["dim"]:
            print(f"   paired all6 1x - {arm} x{k} (s16, s17): " + ", ".join(f"{d} " + "/".join(f"{v:+.1f}" for v in dd[d]) for d in DOMS)
                  + f"; all six above in both seeds: {all(v > 0 for d in DOMS for v in dd[d])}")

print("\n== 6. device / flags")
for key, r in sorted(RUN.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    if key[1] > 1 or (key[2] in V8_SEEDS):
        print(f"   {key[0]:9s} x{key[1]} s{key[2]}: dev {r['dev']}; events {r['events']}; " + ("; ".join(r["flags"]) or "no flags"))

print("\n== reading rule (AUDIT.md Amendment V8): " + RULE)
if REF["dim"]["n"]:
    for arm in ("only-dim", "none"):
        for k in (2, 4):
            v = [out["runs"][f"{arm}-x{k}-s{s}"]["strict_f3"] for s in V8_SEEDS if f"{arm}-x{k}-s{s}" in out["runs"]]
            if len(v) == len(V8_SEEDS):
                cu = bool(np.mean(v) >= REF["dim"]["mean8"] and all(REF["dim"]["min8"] <= x <= REF["dim"]["max8"] for x in v))
                above = [round(x, 1) for x in v if x > REF["dim"]["max8"]]
                print(f"   {arm} x{k}: strict final3 mean {np.mean(v):.1f} (seeds {', '.join(f'{x:.1f}' for x in v)}) vs all6 1x mean "
                      f"{REF['dim']['mean8']:.1f} [{REF['dim']['min8']:.1f}, {REF['dim']['max8']:.1f}] -> "
                      f"{'catches up' if cu else 'does not catch up'} (the rule as written)"
                      + (f"; NOTE seeds above all6's 8-seed max: {above} (outside the range on the high side)" if above else ""))
                out["summary"][f"rule_{arm}-x{k}"] = bool(cu)
            else:
                print(f"   {arm} x{k}: n/a ({len(v)} of {len(V8_SEEDS)} seeds)")
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1) + "\n")
