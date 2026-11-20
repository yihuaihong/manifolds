"""Independent recomputation of the pt6v2 final read-out from the raw run JSONs (result audit, 2026-11-19).
Does NOT import readout_pt6v2 / readout_pt6comb / readout_ptdom* / rainbow6_summary / make_jobs_pt6v2.
The dim strict set and its FLOOR inputs are re-derived here from the definition (dim table rebuilt from scratch).
Only numpy / scipy. Writes nothing outside /scratch/yh6210/tmp_audit/pt6v2_final.
usage: python -I indep.py <readout JSON to compare> > indep_out.txt
"""
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

B = Path("/scratch/yh6210/llmadd_torch")
OUT = B / "runs" / "pt6v2" / "out"
MANI = ("numbers", "dim", "cities", "countries", "torus", "box5")
SEEDS = tuple(range(16, 24))
NONE_LRS = ("2e-3", "4e-3", "8e-3", "1.6e-2", "3.2e-2", "6.4e-2")
P, SAME = "1e-3", "2e-3"
BKT = ("b11", "b12", "b13", "b22", "b23", "b33")
W0, W1 = 916, 7323
NAME = re.compile(r"^pt6v2-(all6|none|only-[a-z0-9]+)-lr([0-9.e-]+)-s(\d+)(?:-r(\d+))?$")
TOY_KEY = {"torus": "torus_held_nonid", "box5": "box5_held_no_negtwin_nonid"}
TOY_MASK = {"torus": "nonid", "box5": "no_negtwin_nonid"}

# ---------------------------------------------------------------- dim table from scratch (E = {-1,0,1}^3; times: a+b, per: a-b)
E = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)
EK = {tuple(v): i for i, v in enumerate(E)}
NEG = np.array([EK[tuple(-v)] for v in E])
rows, ys = [], []
for op, sg in ((27, 1), (28, -1)):
    for i in range(27):
        for j in range(27):
            c = E[i] + sg * E[j]
            if np.abs(c).max() <= 1:
                rows.append((i, op, j))
                ys.append(EK[tuple(c)])
DX, DY = np.array(rows), np.array(ys)
assert len(DX) == 686
DROW = {tuple(x): i for i, x in enumerate(DX.tolist())}


def dim_strict(D):
    """-> (strict mask over eval_held, chance, heuristic) for one run's dim record."""
    held = [int(q) for q in D["eval_held"]["q"]]
    hs = set(held)
    train = set(q for q in range(len(DX)) if q not in hs)
    assert len(train) == D["n_train"]
    ro = [c == "1" for c in D["eval_held"]["masks"]["no_negtwin_nonid"]]
    keep = []
    for q, m in zip(held, ro):
        a, op, b = (int(v) for v in DX[q])
        c = int(DY[q])
        tw = False
        for (x1, x2), n1, n2, op2 in itertools.product(((a, b), (b, a)), (0, 1), (0, 1), (27, 28)):
            t = (int(NEG[x1]) if n1 else x1, op2, int(NEG[x2]) if n2 else x2)
            j = DROW.get(t)
            if j is None or j == q or DY[j] != c:
                continue
            if j in train:
                tw = True
        keep.append(m and not tw)
    keep = np.array(keep)
    maj = int(np.bincount(DY[sorted(train)], minlength=27).argmax())
    chance = max(1 / 27, float(np.mean([DY[q] == maj for q, k in zip(held, keep) if k])))
    acc = []
    for q, k in zip(held, keep):
        if not k:
            continue
        same = [j for j in range(len(DX)) if DX[j, 0] == DX[q, 0] and DX[j, 1] == DX[q, 1]]
        cand = {int(DY[j]) for j in same} - {int(DY[j]) for j in same if j in train}
        acc.append(1 / len(cand) if int(DY[q]) in cand else 0.0)
    return keep, 100 * chance, 100 * float(np.mean(acc))


def device(d):
    f = d / "gpu.csv"
    if f.exists():
        ln = f.read_text().splitlines()
        if len(ln) > 1:
            return ln[1].split(",")[0].strip()
    return None


# ---------------------------------------------------------------- load every run
DB, notes = {}, []
for dd in sorted(OUT.iterdir()):
    m = NAME.match(dd.name)
    if not m:
        notes.append(f"unmatched dir {dd.name}")
        continue
    arm, lr, s = m.group(1), m.group(2), int(m.group(3))
    assert not m.group(4), dd.name
    fs = sorted(dd.glob("*/run_whole_*.json"))
    assert len(fs) == 1, dd
    r = json.loads(fs[0].read_text())
    a = r["args"]
    pd = {"all6": "numbers,dim,cities,countries,torus,box5", "none": ""}.get(arm, arm[5:])
    assert a["seed"] == s and abs(a["lr"] - float(lr)) < 1e-15 and a["prior_domains"] == pd, (dd.name, a["seed"], a["lr"], a["prior_domains"])
    assert a["domains"] == "numbers,dim,cities,countries,torus,box5", dd.name
    c = r["curve"]
    st = [e["step"] for e in c]
    assert len(c) == 75 and st[-1] == 18309, dd.name
    ck = json.loads((dd / "check.json").read_text())
    fails = [k["check"][:70] for k in ck["checks"] if not k["ok"]]
    rec = {"pass": ck.get("pass"), "fails": fails, "dev": device(dd), "steps": st, "data_sha": c[-1].get("data_sha"), "S": {}, "fl": {}}
    nonfinite = [e["step"] for e in c if not all(v is None or math.isfinite(v) for v in (e.get("lm"), e.get("val_loss")))]
    rec["nonfinite"] = nonfinite
    # primary series
    rec["S"]["numbers"] = np.array([np.mean([e[f"held_{b}"] for b in BKT]) for e in c])
    D = r["domains"]["dim"]
    keep, ch, he = dim_strict(D)
    nc = D["eval_held"]["contexts"]
    H = np.array([[int(x) for x in e["hits_dim_held"]] for e in c], float) / nc
    rec["S"]["dim"] = H[:, keep].mean(1)
    rec["n_strict"] = int(keep.sum())
    rec["fl"]["dim"] = max(ch, he) + 5
    rec["dimro_dp"] = float(np.abs(H[:, np.array([x == "1" for x in D["eval_held"]["masks"]["no_negtwin_nonid"]])].mean(1)
                                   - np.array([e["dim_held_no_negtwin_nonid"] for e in c])).max())
    for d in ("cities", "countries"):
        rec["S"][d] = np.array([e[f"{d}_held_2way"] for e in c])
        Dd = r["domains"][d]
        rec["fl"][d] = max(50.0, 100 * Dd["heuristic_ansfreq"], 100 * Dd["heuristic_shorter"]) + 5   # lower bound (h3 not rebuilt)
    for d in ("torus", "box5"):
        v = np.array([e[TOY_KEY[d]] for e in c])
        Dd = r["domains"][d]
        mk = np.array([x == "1" for x in Dd["eval_held"]["masks"][TOY_MASK[d]]])
        assert int(mk.sum()) == Dd["n_readout"]
        Hd = np.array([[int(x) for x in e[f"hits_{d}_held"]] for e in c], float) / Dd["eval_held"]["contexts"]
        rec.setdefault("toy_dp", {})[d] = float(np.abs(Hd[:, mk].mean(1) - v).max())
        rec["S"][d] = v
        rec["fl"][d] = 100 * Dd["chance_readout"] + 5
    rec["fl"]["numbers"] = 100 * r["domains"]["numbers"].get("chance", 0.0) + 5
    rec["area"] = {d: 100 * float(np.mean(rec["S"][d])) for d in MANI}
    rec["final3"] = {d: 100 * float(np.mean(rec["S"][d][-3:])) for d in MANI}
    rec["final1"] = {d: 100 * float(rec["S"][d][-1]) for d in MANI}
    # cross-check against the run's own metrics where they exist
    mt = r["metrics"]
    rec["mx"] = {"numbers": rec["area"]["numbers"] - mt["numbers"]["area"],
                 "cities": rec["area"]["cities"] - mt["cities_readout"]["area"],
                 "countries": rec["area"]["countries"] - mt["countries_readout"]["area"],
                 "torus": rec["area"]["torus"] - mt["torus_readout"]["area"],
                 "box5": rec["area"]["box5"] - mt["box5_readout"]["area"]}
    # LM
    rec["val_loss"], rec["bpb"] = c[-1]["val_loss"], c[-1]["val_bits_per_byte"]
    on = [e for e in c if e.get("prior_on")]
    if arm != "none":
        sh = [e["grad_lm"] / (e["grad_lm"] + e["grad_prior"]) for e in on]
        rec["share"] = float(np.median(sh))
        rec["n_on"] = len(sh)
    else:
        rec["share"] = None
        rec["n_on"] = len(on)
    DB[(arm, lr, s)] = rec

print("runs loaded:", len(DB), "| notes:", notes or "none")
print("check.json pass:", sum(bool(v["pass"]) for v in DB.values()), "of", len(DB))
fl = {f"{k[0]}@{k[1]} s{k[2]}": v["fails"] for k, v in DB.items() if v["fails"]}
print("runs with failed checks:", fl or "none")
print("non-finite lm / val_loss:", {f"{k}": v["nonfinite"] for k, v in DB.items() if v["nonfinite"]} or "none")
print("max |own area - run metrics area|:", {d: max(abs(v["mx"][d]) for v in DB.values()) for d in DB[("all6", P, 16)]["mx"]})
print("max |hits on mask - curve value| toy3:", {d: max(v["toy_dp"][d] for v in DB.values()) for d in ("torus", "box5")},
      "dim no_negtwin_nonid:", max(v["dimro_dp"] for v in DB.values()))
print("dim strict n per seed:", {s: sorted({DB[k]["n_strict"] for k in DB if k[2] == s}) for s in SEEDS})
print("data_sha last eval, distinct per seed:", {s: len({DB[k]["data_sha"] for k in DB if k[2] == s}) for s in SEEDS})
print("device per seed:", {s: sorted({DB[k]["dev"] for k in DB if k[2] == s}) for s in SEEDS})
print("on-window evals per prior run:", sorted({v["n_on"] for k, v in DB.items() if k[0] != "none"}), "none:", sorted({v["n_on"] for k, v in DB.items() if k[0] == "none"}))

A = lambda arm, lr, d, f="area": np.array([DB[(arm, lr, s)][f][d] for s in SEEDS])   # noqa: E731

# ---------------------------------------------------------------- V4 final descriptive read (8 seeds; read only)
OUT4 = B / "runs" / "pt6v2" / "out_v4"
NM = {"numbers": "8e-3", "dim": "3.2e-2", "cities": "4e-3", "countries": "4e-3", "torus": "1.6e-2", "box5": "3.2e-2"}


def series4(r):
    c = r["curve"]
    S = {"numbers": np.array([np.mean([e[f"held_{b}"] for b in BKT]) for e in c])}
    D = r["domains"]["dim"]
    keep, _, _ = dim_strict(D)
    H = np.array([[int(x) for x in e["hits_dim_held"]] for e in c], float) / D["eval_held"]["contexts"]
    S["dim"] = H[:, keep].mean(1)
    for d in ("cities", "countries"):
        S[d] = np.array([e[f"{d}_held_2way"] for e in c])
    for d in ("torus", "box5"):
        S[d] = np.array([e[TOY_KEY[d]] for e in c])
    return S, c


def gpu_names(d):
    f = d / "gpu.csv"
    if not f.exists():
        return []
    return sorted({ln.split(",")[1].strip() for ln in f.read_text().splitlines()[1:] if ln.count(",") >= 1})


V = {}
for cell, lr, pd in (("all6full", "1e-3", "numbers,dim,cities,countries,torus,box5"), ("all4", "2e-3", "numbers,dim,cities,countries")):
    for s in SEEDS:
        fs = sorted((OUT4 / f"pt6v2-{cell}-lr{lr}-s{s}").glob("*/run_whole_*.json"))
        assert len(fs) == 1, (cell, s, fs)
        r = json.loads(fs[0].read_text())
        assert r["curve"][-1]["step"] == 18309 and len(r["curve"]) == 75, (cell, s)
        a = r["args"]
        assert a["seed"] == s and abs(a["lr"] - float(lr)) < 1e-15 and a["prior_domains"] == pd, (cell, s)
        if cell == "all6full":
            assert a["window"] == "0.05,1.0"
        chk = json.loads((fs[0].parent.parent / "check.json").read_text())
        S, c = series4(r)
        V[(cell, s)] = {"S": S, "bpb": c[-1]["val_bits_per_byte"], "sha": c[-1].get("data_sha") == DB[("all6", P, s)]["data_sha"],
                        "pass": chk.get("pass"), "gpus": gpu_names(fs[0].parent.parent)}
print("V4 runs:", len(V), " all CHECKS pass:", all(v["pass"] for v in V.values()), " data_sha == pt6v2 all:", all(v["sha"] for v in V.values()))
print("GPUs:", {k: v["gpus"] for k, v in sorted(V.items())})

f3 = lambda v: 100 * float(np.mean(v[-3:]))   # noqa: E731
ar = lambda v: 100 * float(np.mean(v))   # noqa: E731
TT = stats.t.ppf(0.975, len(SEEDS) - 1)


def summ(x):
    x = np.asarray(x, float)
    m, h = x.mean(), TT * x.std(ddof=1) / math.sqrt(len(x))
    return f"{m:+6.1f} [{m - h:+6.1f}, {m + h:+6.1f}] wins {int((x > 0).sum())}/8"


def summ4(x):
    x = np.asarray(x, float)
    m, h = x.mean(), TT * x.std(ddof=1) / math.sqrt(len(x))
    return f"{m:+.4f} [{m - h:+.4f}, {m + h:+.4f}] ({100 * m / 1.21:+.2f}%) lower in {int((x < 0).sum())}/8"


print("\n[A] all6full@1e-3 (priors on 0.05 -> end) vs all6 window@1e-3 (0.05 -> 0.4), same lr, same seed")
for d in MANI:
    full = np.array([f3(V[("all6full", s)]["S"][d]) for s in SEEDS])
    win = np.array([f3(DB[("all6", P, s)]["S"][d]) for s in SEEDS])
    afull = np.array([ar(V[("all6full", s)]["S"][d]) for s in SEEDS])
    awin = np.array([ar(DB[("all6", P, s)]["S"][d]) for s in SEEDS])
    print(f"  {d:9s} final3 full {full.mean():5.1f} win {win.mean():5.1f}  diff {summ(full - win)} | area full {afull.mean():5.1f} win {awin.mean():5.1f} diff {summ(afull - awin)}")

print("\n[B] all6full final3 vs no prior: (i) none@N_m (area-best lr), (ii) the lr with the best 8-seed mean final3, (iii) per-seed oracle max over 6 lrs")
for d in MANI:
    full = np.array([f3(V[("all6full", s)]["S"][d]) for s in SEEDS])
    nm = np.array([f3(DB[("none", NM[d], s)]["S"][d]) for s in SEEDS])
    mf = {lr: np.mean([f3(DB[("none", lr, s)]["S"][d]) for s in SEEDS]) for lr in NONE_LRS}
    blr = max(mf, key=mf.get)
    bl = np.array([f3(DB[("none", blr, s)]["S"][d]) for s in SEEDS])
    orc = np.array([max(f3(DB[("none", lr, s)]["S"][d]) for lr in NONE_LRS) for s in SEEDS])
    print(f"  {d:9s} full {full.mean():5.1f} | (i) none@{NM[d]} {nm.mean():5.1f} diff {summ(full - nm)} | (ii) none@{blr} {bl.mean():5.1f} diff {summ(full - bl)}"
          f" | (iii) oracle {orc.mean():5.1f} diff {summ(full - orc)}")

print("\n[C] bpb (final val bits per byte; paired per seed)")
bf = np.array([V[("all6full", s)]["bpb"] for s in SEEDS])
bw = np.array([DB[("all6", P, s)]["bpb"] for s in SEEDS])
print(f"  all6full {bf.mean():.4f} (median {np.median(bf):.4f}); all6win@1e-3 {bw.mean():.4f}; full - win {summ4(bf - bw)}")
nb = {lr: np.array([DB[("none", lr, s)]["bpb"] for s in SEEDS]) for lr in NONE_LRS}
for lr in NONE_LRS:
    print(f"  none@{lr:6s} {nb[lr].mean():.4f} (median {np.median(nb[lr]):.4f})  full - none {summ4(bf - nb[lr])}  median diff {np.median(bf - nb[lr]):+.4f}")
bo = np.array([min(DB[("none", lr, s)]["bpb"] for lr in NONE_LRS) for s in SEEDS])
print(f"  per-seed best none bpb (oracle) {bo.mean():.4f}; full - oracle {summ4(bf - bo)}")

print("\n[D] all4@2e-3 (priors numbers, dim, cities, countries; window 0.05-0.4) vs all6@2e-3 and only-m@2e-3 (same lr, same seed)")
for d in ("numbers", "dim", "cities", "countries"):
    a4 = np.array([ar(V[("all4", s)]["S"][d]) for s in SEEDS])
    a6 = np.array([ar(DB[("all6", SAME, s)]["S"][d]) for s in SEEDS])
    om = np.array([ar(DB[(f"only-{d}", SAME, s)]["S"][d]) for s in SEEDS])
    g4 = np.array([f3(V[("all4", s)]["S"][d]) for s in SEEDS])
    g6 = np.array([f3(DB[("all6", SAME, s)]["S"][d]) for s in SEEDS])
    go = np.array([f3(DB[(f"only-{d}", SAME, s)]["S"][d]) for s in SEEDS])
    print(f"  {d:9s} AREA all4 {a4.mean():5.1f} all6 {a6.mean():5.1f} only {om.mean():5.1f} | h4 = all4 - only {summ(a4 - om)} | h6 = all6 - only {summ(a6 - om)} | h6 - h4 = all6 - all4 {summ(a6 - a4)}")
    print(f"  {'':9s} FINAL3 all4 {g4.mean():5.1f} all6 {g6.mean():5.1f} only {go.mean():5.1f} | all4 - only {summ(g4 - go)} | all6 - only {summ(g6 - go)} | all6 - all4 {summ(g6 - g4)}")
print("  torus / box5 without their own prior (all4) vs none@2e-3, only-dim@2e-3 and all6@2e-3 (area; final3)")
for d in ("torus", "box5"):
    a4 = np.array([ar(V[("all4", s)]["S"][d]) for s in SEEDS])
    nn = np.array([ar(DB[("none", SAME, s)]["S"][d]) for s in SEEDS])
    od = np.array([ar(DB[("only-dim", SAME, s)]["S"][d]) for s in SEEDS])
    oo = np.array([ar(DB[(f"only-{d}", SAME, s)]["S"][d]) for s in SEEDS])
    a6 = np.array([ar(DB[("all6", SAME, s)]["S"][d]) for s in SEEDS])
    g4 = np.array([f3(V[("all4", s)]["S"][d]) for s in SEEDS])
    gn = np.array([f3(DB[("none", SAME, s)]["S"][d]) for s in SEEDS])
    print(f"  {d:6s} area all4 {a4.mean():5.1f} none {nn.mean():5.1f} only-dim {od.mean():5.1f} only-{d} {oo.mean():5.1f} all6 {a6.mean():5.1f} | all4 - none {summ(a4 - nn)} | only-dim - none {summ(od - nn)}"
          f" | final3 all4 {g4.mean():5.1f} none {gn.mean():5.1f} diff {summ(g4 - gn)}")
b4 = np.array([V[("all4", s)]["bpb"] for s in SEEDS])
b6 = np.array([DB[("all6", SAME, s)]["bpb"] for s in SEEDS])
print(f"  bpb all4 {b4.mean():.4f} (median {np.median(b4):.4f}) all6@2e-3 {b6.mean():.4f} none@2e-3 {nb['2e-3'].mean():.4f}; all4 - all6 {summ4(b4 - b6)}; all4 - none@2e-3 {summ4(b4 - nb['2e-3'])} median {np.median(b4 - nb['2e-3']):+.4f}")
print("  per-seed all4 bpb:", np.round(b4, 4).tolist())
