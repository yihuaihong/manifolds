"""runs/toyf84fu read-out: the PRE-REGISTERED readings of AUDIT.md for g1-g5 and the Latin split, from the run JSONs.
No training, no GPU. Uses the pinned toy_struct (code/tools, b74a78cb5472) for its exact Gehan / MWU p, auc() and
compare() conventions, latin.py for the S4 Latin-only implied cells, code/hits_wrap.unpack for per-question hits.

    python readout.py                 # every reading whose cells exist; missing cells are listed, never filled in
    python readout.py --selftest      # known answers on the existing data (audit_toy_optimiser_sweeps.md numbers,
                                      # compare() equivalence, A7 Latin levels); must pass before any reading is used
    python readout.py --new_root DIR  # read the new cells from DIR instead of runs/toyf84fu/out (end-to-end tests)
    python readout.py --toyshape_root DIR   # read toyshape's cells from DIR instead of runs/toyshape/out (tests)
Writes readout.md / readout.json next to this file (or to --new_root's parent with --new_root).

Definitions (one place; AUDIT.md section "Metrics"):
- t_grok = first eval with held-out accuracy >= 0.95 (censored = inf); AUC = mean held-out accuracy over every eval of the
  budget (toy_struct.auc); final = the last eval; last50 = mean of the last 50 evals (audit §6: the dim "final" is one
  noisy eval); final_train = last eval's training accuracy.
- @cut (common budget): only the evals at steps 0, 100, ..., cut - 100 (cut / 100 evals; the same set for a cut-step run
  and a 4x-longer run; a run's own last-step eval at cut - 1 is left out). t_grok@cut = the first of them >= 0.95.
- unstable run: share of evals from step 1000 with train < 0.9 above 0.25, or final train < 0.9; a config is an
  OPTIMISER FAILURE when >= 3/5 runs are unstable (audit §4: dim lr 3e-2 collapsed with finite CE). A failure config
  never counts as the best and closes a bracket only "by instability".
- late gain (toyshape rule 1c) = mean of the last 5 evals - mean of the first 5 evals at or after 0.75 x budget.
- rule-2 best = most groks, then the highest median AUC (same budget for every candidate).
- S4 Latin split: implied = held-out cells a pure Latin-square filler derives (naked + hidden singles from the training
  cells, latin.py; A7 / A8); non-implied = the rest. Per eval: exact from hits_te when logged, else bounds from the
  logged n_wrong: non-implied accuracy in [max(0, k - n_imp) / n_non, min(k, n_non) / n_non], k = correct answers.
"""
from __future__ import annotations

import glob
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code" / "tools"))
sys.path.insert(0, str(HERE / "code"))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import toy_struct as ts  # noqa: E402
import latin  # noqa: E402

assert ts.code_sha() == "b74a78cb5472", ts.code_sha()
T = Path("/scratch/yh6210/llmadd_torch/runs")
BR = Path("/scratch/yh6210/llmadd-burst-ro/results")
ARGS = sys.argv[1:]
NEW = Path(ARGS[ARGS.index("--new_root") + 1]).resolve() if "--new_root" in ARGS else HERE / "out"
TSH = Path(ARGS[ARGS.index("--toyshape_root") + 1]).resolve() if "--toyshape_root" in ARGS else T / "toyshape/out"
ALPHA, AUC_MIN, NEED = ts.ALPHA, ts.AUC_MIN, 4
SEEDS = {0.2: [0, 1, 2, 4, 5]}
L: list = []
J: dict = {"missing": []}


def P(*s):
    L.append(" ".join(str(x) for x in s) + "\n")


def unpack(hexstr: str, n: int) -> np.ndarray:   # = code/hits_wrap.unpack (not imported: importing it patches toy_struct)
    return np.unpackbits(np.frombuffer(bytes.fromhex(hexstr), dtype=np.uint8))[:n].astype(bool)


# ------------------------------------------------------------------------------------------------ loading
def load_cell(d: Path, lr: float, wd: float, steps: int, device: str, label: str, task: str, frac: float, arms=None):
    """{arm: [run dicts sorted by seed]} for a cell dir; None if absent / incomplete (recorded in J['missing'])."""
    fs = sorted(Path(d).glob("run_*.json"))
    if not fs:
        J["missing"].append(f"{label}: {d}")
        return None
    runs = {}
    for f in fs:
        r = json.loads(f.read_text())
        if arms and r["arm"] not in arms:
            continue
        assert r["task"] == task and abs(r["frac"] - frac) < 1e-12 and r["steps"] == steps, (f, r["task"], r["frac"], r["steps"])
        for k, v in (("lr", lr), ("wd", wd)):
            if r.get(k) is not None:
                assert abs(r[k] - v) < 1e-12, (f, k, r[k], v)
        assert len(r["curve"]) == ts.n_evals(steps, r.get("eval_every", 100)), (f, "incomplete curve")
        r.update(_lr=lr, _wd=wd, _dev=device, _src=label, _dir=str(d))
        runs.setdefault(r["arm"], []).append(r)
    want = SEEDS.get(frac, [0, 1, 2, 3, 4]) if task == "s4" else [0, 1, 2, 3, 4]
    for a in runs:
        runs[a].sort(key=lambda r: r["seed"])
        got = [r["seed"] for r in runs[a]]
        if got != want:
            J["missing"].append(f"{label}: {d} arm {a} seeds {got} (want {want})")
            return None
    return runs


def gpu_lines(d: Path) -> list:
    p = Path(d) / "gpu.txt"
    return p.read_text().splitlines() if p.exists() else []


# ------------------------------------------------------------------------------------------------ per-run metrics
def evals(r, cut=None):
    cv = r["curve"]
    if cut is None:
        return cv
    out = [c for c in cv if c["step"] < cut and c["step"] % 100 == 0]
    assert len(out) == cut // 100, (r["arm"], r["seed"], len(out), cut)
    return out


def series(r, sub="all", cut=None):
    """(lo, hi) arrays of held-out accuracy per eval on a subset: all / imp / non (S4 Latin split) / fresh / nonegtwin."""
    cv = evals(r, cut)
    if sub == "all":
        v = np.array([c["test"] for c in cv]); return v, v
    if sub in ("fresh", "nonegtwin"):
        if any(c.get("test_" + sub) is None for c in cv):
            return None, None
        v = np.array([c["test_" + sub] for c in cv], dtype=float); return v, v
    lv = latin.latin_split(r["frac"], r["seed"])
    assert lv["split_sha"] == r["split_sha"], ("Latin masks built on another split", r["_src"], r["seed"])
    imp = lv["imp_mask"]
    n, ni = len(imp), int(imp.sum())
    assert n == r["n_test"]
    m = imp if sub == "imp" else ~imp
    nm = int(m.sum())
    if nm == 0:
        return None, None
    if "hits_te" in cv[0]:
        h = np.array([unpack(c["hits_te"], n) for c in cv])
        v = h[:, m].mean(1); return v, v
    k = np.array([n - c["n_wrong"] for c in cv])
    other = n - nm
    return np.maximum(0, k - other) / nm, np.minimum(k, nm) / nm


def first_ge(steps, v, thr=0.95):
    i = np.flatnonzero(v >= thr)
    return math.inf if len(i) == 0 else float(steps[i[0]])


def run_metrics(r, sub="all", cut=None):
    lo, hi = series(r, sub, cut)
    if lo is None:
        return None
    st = np.array([c["step"] for c in evals(r, cut)])
    B = cut or r["steps"]
    late = [i for i, s in enumerate(st) if s >= 0.75 * B]
    g = lambda v: float(np.mean(v[-5:]) - np.mean(v[late[:5]])) if len(late) >= 5 else float("nan")
    tr = np.array([c["train"] for c in evals(r, cut)])
    after = tr[st >= 1000]
    bad = float(np.mean(after < 0.9)) if len(after) else 0.0
    return {"t_grok": (first_ge(st, hi), first_ge(st, lo)),   # (earliest possible, latest possible)
            "auc": (float(lo.mean()), float(hi.mean())), "final": (float(lo[-1]), float(hi[-1])),
            "last50": (float(lo[-50:].mean()), float(hi[-50:].mean())), "late_gain": (g(lo), g(hi)),
            "final_train": float(tr[-1]), "train_bad": bad, "unstable": bool(bad > 0.25 or tr[-1] < 0.9),
            "max_ce_after1k": float(max([c["ce"] for c in evals(r, cut) if c["step"] >= 1000] or [float("nan")])),
            "finite": all(math.isfinite(c["ce"]) and math.isfinite(c["test_ce"]) for c in r["curve"])}


def med(v):
    return ts.median_vals(list(v))


def ex(x):   # an exact value or an interval
    return x[0] if x[0] == x[1] or (isinstance(x[0], float) and abs(x[0] - x[1]) < 1e-12) else None


def fmt(x, p=3):
    if isinstance(x, tuple):
        e = ex(x)
        return fmt(e, p) if e is not None else f"[{fmt(x[0], p)}, {fmt(x[1], p)}]"
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "cens"
    if isinstance(x, float) and x == int(x) and abs(x) >= 100:
        return str(int(x))
    return f"{x:.{p}f}" if isinstance(x, float) else str(x)


def cell_stats(rs, sub="all", cut=None):
    M = [run_metrics(r, sub, cut) for r in rs]
    if any(m is None for m in M):
        return None
    side = lambda key, i: [m[key][i] for m in M]
    two = lambda f, key: (f(side(key, 0)), f(side(key, 1)))
    return {"n": len(M), "seeds": [r["seed"] for r in rs],
            "groks": (sum(math.isfinite(t) for t in side("t_grok", 1)), sum(math.isfinite(t) for t in side("t_grok", 0))),
            "t_grok": [m["t_grok"] for m in M], "t_grok_cmed": (med(side("t_grok", 0)), med(side("t_grok", 1))),
            "auc": two(med, "auc"), "auc_seed": [m["auc"] for m in M], "final": two(med, "final"),
            "last50": two(med, "last50"), "late_gain": two(med, "late_gain"),
            "final_train": med([m["final_train"] for m in M]), "n_unstable": sum(m["unstable"] for m in M),
            "failure": sum(m["unstable"] for m in M) >= 3, "max_ce": max(m["max_ce_after1k"] for m in M),
            "finite": all(m["finite"] for m in M)}


def compare_vals(ta, tb, ua, ub, chance):
    """toy_struct.compare's decision on given per-seed values (paired lists): t_grok (a earlier) and AUC (a higher)."""
    n = len(ta); need = math.ceil(0.8 * n)
    g = {"med_a": med(ta), "med_b": med(tb), "wins": sum(x < y for x, y in zip(ta, tb)), "p": ts.mwu_p(ta, tb)}
    g["pass"] = bool(g["med_a"] < g["med_b"] and g["wins"] >= need and g["p"] < ALPHA)
    u = {"med_a": med(ua), "med_b": med(ub), "wins": sum(x > y for x, y in zip(ua, ub)),
         "p": ts.mwu_p([-x for x in ua], [-x for x in ub])}
    u["pass"] = bool(u["med_a"] - u["med_b"] >= AUC_MIN and u["med_a"] > chance + AUC_MIN and u["wins"] >= need and u["p"] < ALPHA)
    floor = bool(not g["pass"] and not u["pass"] and sum(map(math.isinf, ta)) >= need and sum(map(math.isinf, tb)) >= need
                 and max(u["med_a"], u["med_b"]) <= chance + AUC_MIN)
    verdict = ("PASS (t_grok + AUC)" if g["pass"] and u["pass"] else "PASS (t_grok)" if g["pass"] else
               "PASS (AUC)" if u["pass"] else "floor: uninformative" if floor else f"not shown at n={n}")
    return {"n": n, "grok": g, "auc": u, "pass": bool(g["pass"] or u["pass"]), "floor": floor, "verdict": verdict}


def paired(ra, rb, sub="all", cut=None, chance=0.0):
    """compare_vals on exact per-seed values, paired by seed (same split_sha asserted). None if a side has only bounds."""
    sa, sb = {r["seed"]: r for r in ra}, {r["seed"]: r for r in rb}
    sd = sorted(set(sa) & set(sb))
    for s in sd:
        assert sa[s]["split_sha"] == sb[s]["split_sha"], ("pairing", s)
    ma = [run_metrics(sa[s], sub, cut) for s in sd]; mb = [run_metrics(sb[s], sub, cut) for s in sd]
    if any(m is None or ex(m["t_grok"]) is None or ex(m["auc"]) is None for m in ma + mb):
        return None
    t = compare_vals([m["t_grok"][0] for m in ma], [m["t_grok"][0] for m in mb], [m["auc"][0] for m in ma],
                     [m["auc"][0] for m in mb], chance)
    t["seeds"] = sd
    return t


def tline(t):
    if t is None:
        return "no test (bounds only or missing)"
    g, u = t["grok"], t["auc"]
    return (f"t_grok {fmt(g['med_a'])} vs {fmt(g['med_b'])}, wins {g['wins']}/{t['n']}, p {g['p']:.4f} | AUC {u['med_a']:.3f} vs "
            f"{u['med_b']:.3f}, wins {u['wins']}/{t['n']}, p {u['p']:.4f} -> {t['verdict']}")


def ratios(tn, tr, budget=None):
    """none / prior t_grok per seed; a censored none gives the lower bound budget / prior (marked True)."""
    out = []
    for a, b in zip(tn, tr):
        if not math.isfinite(b):
            out.append((float("nan"), False))
        elif math.isinf(a):
            out.append((budget / b if budget else math.inf, True))
        else:
            out.append((a / b, False))
    return out


def best_rule2(cfgs: dict, cut=None, sub="all"):
    """cfgs {key: runs}; rule-2 best (most groks, then the highest median AUC) among non-failure configs. Exact ties
    (pre-registered) go to the earlier censored-median t_grok, then the lower lr; a tie is recorded in J['ties']."""
    sc = {}
    for k, rs in cfgs.items():
        s = cell_stats(rs, sub, cut)
        if s is None:
            continue
        sc[k] = s
    ok = {k: s for k, s in sc.items() if not s["failure"]}
    if not ok:
        return None, sc
    key = lambda k: (ok[k]["groks"][0], ok[k]["auc"][0])
    tied = [k for k in ok if key(k) == max(map(key, ok))]
    top = min(tied, key=lambda k: (ok[k]["t_grok_cmed"][0], k))
    if len(tied) > 1:
        J.setdefault("ties", []).append([list(k) for k in tied])
    return top, sc


def bracket(best, grid_keys, failures, axis_vals):
    """best = (lr, wd); on each axis: 'closed', 'closed by instability' (a neighbour is a failure), 'OPEN (edge)'."""
    out = {}
    for ax, i in (("lr", 0), ("wd", 1)):
        line = sorted(k[i] for k in grid_keys if k[1 - i] == best[1 - i])
        lo = [v for v in line if v < best[i]]; hi = [v for v in line if v > best[i]]
        if not lo or not hi:
            out[ax] = f"OPEN ({'bottom' if not lo else 'top'} edge of {line})"
        else:
            nb = [(max(lo) if j == 0 else min(hi)) for j in (0, 1)]
            kk = [(v, best[1]) if i == 0 else (best[0], v) for v in nb]
            out[ax] = "closed by instability" if any(k in failures for k in kk) else "closed"
    return out


def stat_row(label, s):
    return (f"| {label} | {fmt(s['groks'])}/{s['n']} | {fmt(s['t_grok_cmed'])} | {', '.join(fmt(t) for t in s['t_grok'])} | "
            f"{fmt(s['auc'])} | {fmt(s['final'])} | {fmt(s['last50'])} | {s['final_train']:.3f} | {s['n_unstable']}/{s['n']}"
            f"{' FAILURE' if s['failure'] else ''} | {fmt(s['late_gain'])} |\n")


HDR = ("| config | grokked | t_grok cmed | t_grok per seed | AUC med | final med | last-50 med | final train med | unstable | late gain med |\n"
       "|---|---|---|---|---|---|---|---|---|---|\n")


# ------------------------------------------------------------------------------------------------ the cells
def s4_06():
    none, pri = {}, {}
    for wd in (0.1, 1.0, 3.0):
        w = f"{wd:g}"
        c = load_cell(T / f"toynone/out/s4-f0.6-lr1e-3-wd{w}", 1e-3, wd, 100000, "L40S", "toynone", "s4", 0.6)
        if c: none[(1e-3, wd, "toynone")] = c["none"]
        for lr, ls in ((3e-4, "3e-4"), (3e-3, "3e-3"), (1e-2, "1e-2")):
            c = load_cell(T / f"toynone2/out/s4-f0.6-lr{ls}-wd{w}", lr, wd, 100000, "L40S", "toynone2", "s4", 0.6)
            if c: none[(lr, wd, "toynone2")] = c["none"]
    g1 = load_cell(NEW / "g1-s4-f0.6-lr1e-2-wd1", 1e-2, 1.0, 100000, "L40S", "toyf84fu g1", "s4", 0.6)
    g2a = load_cell(NEW / "g2-s4-f0.6-lr3e-2-wd1", 3e-2, 1.0, 100000, "L40S", "toyf84fu g2", "s4", 0.6)
    g2b = load_cell(NEW / "g2-s4-f0.6-lr3e-2-wd3", 3e-2, 3.0, 100000, "L40S", "toyf84fu g2", "s4", 0.6)
    if g1: none[(1e-2, 1.0, "g1")] = g1["none"]
    if g2a: none[(3e-2, 1.0, "g2")] = g2a["none"]
    if g2b: none[(3e-2, 3.0, "g2")] = g2b["none"]
    c = load_cell(T / "toystruct/out/s4-f0.6-always", 1e-3, 1.0, 25000, "L40S", "toystruct F84", "s4", 0.6,
                  arms={"none", "R@always", "R_sign@always", "specrand@always"})
    if c:
        for a in ("R@always", "R_sign@always", "specrand@always"): pri[(a, 1e-3, "toystruct")] = c[a]
    c = load_cell(T / "toyprior_lr/out/s4-f0.6-lr3e-3-wd1", 3e-3, 1.0, 100000, "L40S", "toyprior_lr", "s4", 0.6)
    if c:
        for a in c: pri[(a, 3e-3, "toyprior_lr")] = c[a]
    c = load_cell(BR / "482-toyb-toyprior_lr-s4-f0.6-lr1e-2-wd1", 1e-2, 1.0, 100000, "A100", "Burst 482", "s4", 0.6)
    if c:
        for a in c: pri[(a, 1e-2, "burst482")] = c[a]
    if g1:
        for a in ("R@always", "R_sign@always"): pri[(a, 1e-2, "g1")] = g1[a]
    if g2a:
        pri[("R@always", 3e-2, "g2")] = g2a["R@always"]
    return none, pri, g1


def dim_cells():
    none, pri = {}, {}
    for wd in (0.1, 1.0, 3.0):
        w = f"{wd:g}"
        c = load_cell(T / f"toynone/out/dim-f0.3-lr1e-3-wd{w}", 1e-3, wd, 200000, "L40S", "toynone", "dim", 0.3)
        if c: none[(1e-3, wd)] = c["none"]
        for lr, ls in ((3e-4, "3e-4"), (3e-3, "3e-3"), (1e-2, "1e-2")):
            c = load_cell(T / f"toynone2/out/dim-f0.3-lr{ls}-wd{w}", lr, wd, 200000, "L40S", "toynone2", "dim", 0.3)
            if c: none[(lr, wd)] = c["none"]
    for d, wd, dev, lab in ((T / "toynone3/out/dim-f0.3-lr3e-2-wd0.1", 0.1, "L40S", "toynone3"),
                            (BR / "480-toyb-toynone3-dim-f0.3-lr3e-2-wd1", 1.0, "A100", "Burst 480"),
                            (BR / "481-toyb-toynone3-dim-f0.3-lr3e-2-wd3", 3.0, "A100", "Burst 481")):
        c = load_cell(d, 3e-2, wd, 200000, dev, lab, "dim", 0.3)
        if c: none[(3e-2, wd)] = c["none"]
    for name, lr, wd in (("g3-dim-f0.3-lr1e-2-wd0.03", 1e-2, 0.03), ("g3-dim-f0.3-lr2e-2-wd0.1", 2e-2, 0.1),
                         ("g3-dim-f0.3-lr2e-2-wd1", 2e-2, 1.0)):
        c = load_cell(NEW / name, lr, wd, 200000, "L40S", "toyf84fu g3", "dim", 0.3)
        if c: none[(lr, wd)] = c["none"]
    c = load_cell(T / "toystruct/out/dim-f0.3-oh", 1e-3, 1.0, 50000, "L40S", "toystruct F84", "dim", 0.3,
                  arms={"expo", "specrand", "none"})
    if c:
        for a in ("expo", "specrand"): pri[(a, 1e-3)] = c[a]
        none_f84 = c["none"]
    else:
        none_f84 = None
    for name, lr in (("g4-dim-f0.3-lr3e-3-wd1", 3e-3), ("g4-dim-f0.3-lr1e-2-wd1", 1e-2)):
        c = load_cell(NEW / name, lr, 1.0, 200000, "L40S", "toyf84fu g4", "dim", 0.3)
        if c:
            for a in ("expo", "specrand"): pri[(a, lr)] = c[a]
    return none, pri, none_f84


# AUDIT fix I1 (2026-10-21, before any toyf84fu result was read): the pre-registered toyshape S4 none grid
# (make_jobs_toyshape.py NONE_LR x NONE_WD, plus FLAT_EXT for its EXT_CELLS s4 0.2 / 0.3) and the R@always lrs. A config
# that is absent or incomplete makes the rescue reading at that fraction NO READING (INC), never a silent drop.
NONE_GRID = [(lr, wd) for lr in ("1e-3", "3e-3", "1e-2", "3e-2") for wd in ("0.1", "1", "3")]
NONE_GRID_EXT = [("1e-1", wd) for wd in ("0.1", "1", "3")] + [(lr, "10") for lr in ("1e-3", "3e-3", "1e-2", "3e-2")]
S4_LOW_NONE = {0.2: NONE_GRID + NONE_GRID_EXT, 0.3: NONE_GRID + NONE_GRID_EXT, 0.45: NONE_GRID}
S4_LOW_R = ("1e-3", "1e-2")
INC: dict = {}


def s4_low(frac):
    none, pri = {}, {}
    inc = INC.setdefault(frac, [])
    want = {f"s4-f{frac}-none-lr{lr}-wd{wd}" for lr, wd in S4_LOW_NONE[frac]}
    have = {Path(d).name for d in glob.glob(str(TSH / f"s4-f{frac}-none-lr*-wd*"))}
    for nm in sorted(want - have):
        if f"none {nm}: absent" not in inc: inc.append(f"none {nm}: absent")
    for d in sorted(glob.glob(str(TSH / f"s4-f{frac}-none-lr*-wd*"))):
        nm = Path(d).name
        lr, wd = float(nm.split("-lr")[1].split("-wd")[0]), float(nm.split("-wd")[1])
        c = load_cell(Path(d), lr, wd, 200000, "L40S", "toyshape", "s4", frac)
        if c: none[(lr, wd)] = c["none"]
        elif f"none {nm}: incomplete" not in inc: inc.append(f"none {nm}: incomplete")
    for ls in S4_LOW_R:
        d = TSH / f"s4-R-f{frac}-lr{ls}-wd1"
        c = load_cell(d, float(ls), 1.0, 50000, "L40S", "toyshape", "s4", frac)   # every arm (the Latin table shows them)
        if c:
            for a in c: pri[(a, float(ls), "toyshape")] = c[a]
        else:   # another arm incomplete must not drop R@always: only R's own seeds gate the rescue reading
            c = load_cell(d, float(ls), 1.0, 50000, "L40S", "toyshape", "s4", frac, arms={"R@always"})
            if c and "R@always" in c: pri[("R@always", float(ls), "toyshape")] = c["R@always"]
        if ("R@always", float(ls), "toyshape") not in pri and f"R@always {d.name}: absent or incomplete" not in inc:
            inc.append(f"R@always {d.name}: absent or incomplete")
    if not glob.glob(str(TSH / f"s4-f{frac}-none-lr*")):
        J["missing"].append(f"toyshape: S4 {frac} none / prior cells (not landed)")
    for ls, lr in (("3e-3", 3e-3), ("1e-2", 1e-2), ("3e-2", 3e-2)):
        c = load_cell(NEW / f"g5-s4-f{frac}-lr{ls}-wd1", lr, 1.0, 200000, "L40S", "toyf84fu g5", "s4", frac)
        if c:
            for a in c: pri[(a, lr, "g5")] = c[a]
    return none, pri


# ------------------------------------------------------------------------------------------------ readings
def read_g1(none06, pri06, g1):
    P("\n## g1: S4 0.6 same-device control (lr 1e-2, wd 1, one L40S job)\n")
    if not g1:
        P("NOT RUN YET (runs/toyf84fu/out/g1-s4-f0.6-lr1e-2-wd1)."); return
    gl = gpu_lines(NEW / "g1-s4-f0.6-lr1e-2-wd1")
    P(f"gpu.txt: {len(gl)} start(s); all L40S: {all('L40S' in x for x in gl)}; one physical GPU (uuid): "
      f"{len({x.split('GPU-')[-1] for x in gl}) == 1 if gl else 'n/a'}" + (" (requeued: same device TYPE, maybe not the same card)" if len(gl) > 1 else ""))
    N, chance = g1["none"], 1 / 24
    for arm in ("R@always", "R_sign@always"):
        Rr = g1[arm]
        tn = [ts.cens(r) for r in N]; tr = [ts.cens(r) for r in Rr]
        rt = ratios(tn, tr, N[0]["steps"])
        fin = [x for x, c in rt if math.isfinite(x)]
        rom = med(tn) / med(tr) if math.isfinite(med(tr)) else float("nan")
        mpr = med([x for x, _ in rt])
        t = paired(Rr, N, chance=max(chance, med([r["chance_majority"] for r in N])))
        P(f"\n**{arm} vs none** (seeds {[r['seed'] for r in N]}): none t_grok {[fmt(x) for x in tn]}; {arm} {[fmt(x) for x in tr]}")
        P(f"- ratio of censored medians {rom:.2f}x; per-seed ratios {[('>=' if c else '') + f'{x:.2f}' for x, c in rt]}; "
          f"median per-seed {mpr:.2f}x; min {min(fin) if fin else float('nan'):.2f}x; {arm} earlier on "
          f"{sum(b < a for a, b in zip(tn, tr))}/{len(tn)}; Gehan p {ts.mwu_p(tr, tn):.4f}")
        P(f"- paired T({arm}, none) over 100k: {tline(t)}")
        P(f"- AUC over the first 5k steps (median): none {med([run_metrics(r, cut=5000)['auc'][0] for r in N]):.3f}, "
          f"{arm} {med([run_metrics(r, cut=5000)['auc'][0] for r in Rr]):.3f}")
        earlier, p = sum(b < a for a, b in zip(tn, tr)), ts.mwu_p(tr, tn)
        if earlier == len(tn) and p < ALPHA:
            lo_, hi_ = sorted((rom, mpr))
            P(f"- PRE-REGISTERED wording: \"{arm} vs none on one L40S (lr 1e-2, wd 1): about {lo_:.1f}-{hi_:.1f}x "
              f"(ratio of censored medians {rom:.1f}x, median per-seed ratio {mpr:.1f}x), earlier on {earlier}/{len(tn)} seeds "
              f"(Gehan p {p:.3f})\"" + ("; '>4x' allowed (both estimators >= 4)" if lo_ >= 4 else "; NOT '>4x'"))
        else:
            P(f"- PRE-REGISTERED wording: \"speed-up not shown on one device at n = {len(tn)}\" ({arm} earlier {earlier}/{len(tn)}, p {p:.3f})")
    ref = none06.get((1e-2, 1.0, "toynone2"))
    if ref:
        strip = lambda cv: [{k: v for k, v in c.items() if not k.startswith("hits_")} for c in cv]   # g1 adds hits_* only
        same = [strip(a["curve"]) == strip(b["curve"]) for a, b in zip(ref, N)]
        P(f"\nDeterminism on the L40S (toynone2 none vs g1 none, same sha / seed / split / init): per-seed t_grok toynone2 "
          f"{[fmt(ts.cens(r)) for r in ref]} vs g1 {[fmt(ts.cens(r)) for r in N]}; curves bitwise identical: {same}")
        P("  (identical = a same-device re-run reproduces; different = run-to-run nondeterminism on one GPU type: the "
          "per-seed ratio is then itself noisy, and only g1's own none is paired with g1's R)")
    b = pri06.get(("R@always", 1e-2, "burst482"))
    if b:
        P(f"R@always A100 (Burst 482) vs L40S (g1): {[fmt(ts.cens(r)) for r in b]} vs {[fmt(ts.cens(r)) for r in g1['R@always']]}; "
          f"medians {fmt(med([ts.cens(r) for r in b]))} vs {fmt(med([ts.cens(r) for r in g1['R@always']]))} "
          "(device effect on R iff the medians differ by > 1.5x)")
    J["g1"] = True


def best_speed(sc: dict):
    """S4 0.6 speed rule (pre-registered, AUDIT.md): among non-failure configs with the most groks, the lowest censored
    median t_grok; ties -> the higher median AUC, then the lower lr. (Every S4 0.6 candidate groks 5/5; their 100k AUCs
    differ by <= 0.005 from post-grok wobble, so rule 2's AUC tie-break would select on noise.)"""
    ok = {k: s for k, s in sc.items() if not s["failure"]}
    if not ok:
        return None
    mg = max(s["groks"][0] for s in ok.values())
    c = [k for k, s in ok.items() if s["groks"][0] == mg]
    return min(c, key=lambda k: (ok[k]["t_grok_cmed"][0], -ok[k]["auc"][0], k))


def read_g2(none06, pri06):
    P("\n## g2: S4 0.6 upper lr edge (lr 3e-2)\n")
    grid = {}
    for (lr, wd, src), rs in none06.items():
        if (lr, wd) == (1e-2, 1.0) and src == "toynone2" and (1e-2, 1.0, "g1") in none06:
            continue   # pre-registered: g1's re-run is the grid value at (1e-2, 1); toynone2's is shown in g1
        grid[(lr, wd)] = rs
    r2, sc = best_rule2(grid)
    best = best_speed(sc)
    P("none, 100k steps (sources: toynone lr 1e-3, toynone2, g1 at lr 1e-2 wd 1, g2 at lr 3e-2):\n")
    P(HDR.rstrip("\n"))
    for k in sorted(sc):
        L.append(stat_row(f"lr {k[0]:g} wd {k[1]:g}", sc[k]))
    fails = {k for k, s in sc.items() if s["failure"]}
    if best:
        P(f"\nspeed-rule best none: lr {best[0]:g} wd {best[1]:g}; bracket {bracket(best, sc.keys(), fails, None)} "
          f"(rule 2, groks then AUC, would pick lr {r2[0]:g} wd {r2[1]:g})")
        if (3e-2, 1.0) in sc:
            P(f"none's optimum moves past 1e-2: {best[0] > 1e-2} (pre-registered: iff the speed-rule best is a lr 3e-2 config, "
              "which cannot be an OPTIMISER FAILURE)")
    rgrid = {lr: rs for (a, lr, src), rs in pri06.items() if a == "R@always" and src != "burst482" and lr > 1e-3}
    r2r, rsc = best_rule2({(lr, 1.0): rs for lr, rs in rgrid.items()})
    rb = best_speed(rsc)
    P("\nR@always, wd 1, 100k steps, L40S only (toyprior_lr 3e-3, g1 1e-2, g2 3e-2; lr 1e-3 = F84 at 25k, median 10.2k, "
      "not in the selection):\n")
    P(HDR.rstrip("\n"))
    for k in sorted(rsc):
        L.append(stat_row(f"lr {k[0]:g}", rsc[k]))
    if rb:
        rfails = {k for k, s in rsc.items() if s["failure"]}
        lrs = sorted(k[0] for k in rsc)
        br = bracket(rb, list(rsc.keys()) + ([(1e-3, 1.0)] if rb[0] == min(lrs) else []), rfails, None)["lr"]
        P(f"\nspeed-rule best R@always: lr {rb[0]:g}; lr bracket {br}; moves past 1e-2: {rb[0] > 1e-2}"
          + ("" if (3e-2, 1.0) in rsc else " (g2 not run yet)") + f" (rule 2 would pick lr {r2r[0]:g})")
    if best and rb:
        t = paired(rgrid[rb[0]], grid[best], chance=1 / 24)
        tn = [ts.cens(r) for r in grid[best]]; tr = [ts.cens(r) for r in rgrid[rb[0]]]
        rt = ratios(tn, tr, grid[best][0]["steps"])
        rom = med(tn) / med(tr) if math.isfinite(med(tr)) else float("nan")
        mpr = med([x for x, _ in rt])
        P(f"best-vs-best on the L40S (R@always lr {rb[0]:g} vs none lr {best[0]:g} wd {best[1]:g}): ratio of censored medians "
          f"{rom:.2f}x, median per-seed {mpr:.2f}x, per seed {[('>=' if c else '') + f'{x:.2f}' for x, c in rt]}; {tline(t)}")
        # AUDIT fix I2 (2026-10-21, before any result was read): g1's wording rule (AUDIT.md section 4), plus "best tested"
        nbr = bracket(best, sc.keys(), fails, None)
        open_on = [f"none {ax}: {v}" for ax, v in nbr.items() if "closed" not in v] + ([f"R@always lr: {br}"] if "closed" not in br else [])
        earlier, p = sum(b < a for a, b in zip(tn, tr)), ts.mwu_p(tr, tn)
        tag = f" (best tested: bracket open on {'; '.join(open_on)})" if open_on else ""
        if earlier == len(tn) and p < ALPHA:
            lo_, hi_ = sorted((rom, mpr))
            P(f"- PRE-REGISTERED wording: \"best R@always vs best none on one L40S: about {lo_:.1f}-{hi_:.1f}x\"{tag}, earlier on "
              f"{earlier}/{len(tn)} seeds (Gehan p {p:.3f})" + ("; '>4x' allowed (both estimators >= 4)" if lo_ >= 4 else "; NOT '>4x'"))
        else:
            P(f"- PRE-REGISTERED wording: \"best-vs-best speed-up not shown on one device at n = {len(tn)}\"{tag} "
              f"(R earlier {earlier}/{len(tn)}, p {p:.3f})")
    J["g2"] = {"best_none": best, "best_R": rb, "rule2_none": r2, "rule2_R": r2r}


def read_g3(dnone):
    P("\n## g3: dim 0.3 none, every swept optimiser (200k steps)\n")
    best, sc = best_rule2(dnone)
    P(HDR.rstrip("\n"))
    for k in sorted(sc):
        L.append(stat_row(f"lr {k[0]:g} wd {k[1]:g}{' (new)' if k in ((1e-2, 0.03), (2e-2, 0.1), (2e-2, 1.0)) else ''}", sc[k]))
    over = [k for k, s in sc.items() if s["groks"][0] >= 2]
    cap = [(k, s["final"][0], s["last50"][0]) for k, s in sc.items() if s["final"][0] > 0.20 or s["last50"][0] > 0.20]
    P(f"\n(a) toynone pre-registered criterion (<= 1/5 in EVERY config): {'HOLDS' if not over else 'FAILS at ' + str(over)}")
    P("(b) toyshape rule 1(b) cap 0.20 on the median final / median last-50 mean: configs above it: "
      + (", ".join(f"lr {k[0]:g} wd {k[1]:g} (final {f:.3f}, last-50 {l5:.3f})" for k, f, l5 in sorted(cap)) or "none"))
    P(f"(c) still rising (median late gain >= 0.02): {[f'lr {k[0]:g} wd {k[1]:g}' for k, s in sc.items() if s['late_gain'][0] >= 0.02] or 'none'}")
    fails = {k for k, s in sc.items() if s["failure"]}
    P(f"OPTIMISER FAILURE configs (>= 3/5 unstable): {[f'lr {k[0]:g} wd {k[1]:g}' for k in sorted(fails)] or 'none'}")
    if best:
        P(f"(d) rule-2 best none: lr {best[0]:g} wd {best[1]:g}; bracket {bracket(best, sc.keys(), fails, None)}")
    P(f"max over configs of the median last-50 mean: {max(s['last50'][0] for s in sc.values()):.3f}")
    P("(e) every grokked none run (sustained = post-grok mean >= 0.8):")
    for k in sorted(dnone):
        for r in dnone[k]:
            if r["t_grok"] is not None:
                post = [c["test"] for c in r["curve"] if c["step"] >= r["t_grok"]]
                P(f"  lr {k[0]:g} wd {k[1]:g} seed {r['seed']}: t_grok {r['t_grok']}, post-grok mean {np.mean(post):.3f}, "
                  f"share >= 0.9 {np.mean(np.array(post) >= 0.9):.2f}, final {r['curve'][-1]['test']:.3f}, last-50 "
                  f"{np.mean([c['test'] for c in r['curve'][-50:]]):.3f}")
    J["g3"] = {"best": best, "over_1of5": over, "above_cap": [list(k) for k, _, _ in cap]}
    return best, sc


def read_g4(dnone, dpri, none_f84):
    P("\n## g4: dim 0.3 prior lr scan (expo / specrand, window [0, 2000), wd 1)\n")
    C = 50000
    chance = 1 / 27
    P(f"Common budget: the first {C} steps (evals 0..{C - 100}; F84's lr 1e-3 cell ran {C} steps).\n")
    P(HDR.rstrip("\n").replace("| config |", "| arm / lr (@50k) |"))
    for arm in ("expo", "specrand"):
        for lr in (1e-3, 3e-3, 1e-2):
            if (arm, lr) in dpri:
                L.append(stat_row(f"{arm} lr {lr:g}", cell_stats(dpri[(arm, lr)], cut=C)))
    eb, esc = best_rule2({(lr, 1.0): rs for (a, lr), rs in dpri.items() if a == "expo"}, cut=C)
    nb, nsc = best_rule2(dnone, cut=C)
    if eb:
        efails = {k for k, s in esc.items() if s["failure"]}
        P(f"\nexpo rule-2 best lr (@50k): {eb[0]:g}; lr bracket {bracket(eb, esc.keys(), efails, None)['lr']} (wd not swept for expo: "
          "fixed at 1, as F84)")
    if nb:
        P(f"none rule-2 best (@50k, every none config truncated to 50k): lr {nb[0]:g} wd {nb[1]:g}; bracket "
          f"{bracket(nb, nsc.keys(), {k for k, s in nsc.items() if s['failure']}, None)}")
    # AUDIT fix I2 (2026-10-21, before any result was read): brackets first; a best-vs-best ratio only if all are closed
    open_on = []
    if eb:
        e_lr = bracket(eb, esc.keys(), efails, None)["lr"]
        if "closed" not in e_lr: open_on.append(f"expo lr: {e_lr}")
    if nb:
        nbr = bracket(nb, nsc.keys(), {k for k, s in nsc.items() if s["failure"]}, None)
        open_on += [f"none {ax}: {v}" for ax, v in nbr.items() if "closed" not in v]
    if eb and nb:
        E, N = dpri[("expo", eb[0])], dnone[nb]
        t = paired(E, N, cut=C, chance=chance)
        P(f"best expo vs best none @50k: {tline(t)}")
        tw = paired(E, N, sub="nonegtwin", cut=C, chance=chance)
        P(f"  no-twin subset (secondary, F84 amendment): {tline(tw)}")
        tn = [run_metrics(r, cut=C)["t_grok"][0] for r in N]; te = [run_metrics(r, cut=C)["t_grok"][0] for r in E]
        if open_on:
            P(f"  censored medians: none {fmt(med(tn))} vs expo {fmt(med(te))} -> NO RATIO: best tested (bracket open on "
              f"{'; '.join(open_on)})")
        else:
            P(f"  censored medians: none {fmt(med(tn))} vs expo {fmt(med(te))}" +
              (f" -> ratio > {C / med(te):.1f}x (none censored at 50k)" if math.isinf(med(tn)) and math.isfinite(med(te)) else
               f" -> ratio {med(tn) / med(te):.2f}x" if math.isfinite(med(te)) else ""))
    for lr in (3e-3, 1e-2):
        if ("expo", lr) in dpri:
            nb2, nsc2 = best_rule2({k: v for k, v in dnone.items()})
            E, S = dpri[("expo", lr)], dpri[("specrand", lr)]
            P(f"\nlr {lr:g}, 200k: expo vs specrand: {tline(paired(E, S, chance=chance))}")
            P(f"  @50k: {tline(paired(E, S, cut=C, chance=chance))}")
            if (lr, 1.0) in dnone:
                N1 = dnone[(lr, 1.0)]
                ts_, tn_ = cell_stats(S), cell_stats(N1)
                lower = sum(a[0] < b[0] for a, b in zip(ts_["auc_seed"], tn_["auc_seed"]))
                harm = lower >= NEED and ts_["auc"][0] - tn_["auc"][0] <= -AUC_MIN and tn_["auc"][0] > chance + AUC_MIN
                P(f"  specrand vs none at the same lr / wd 1 (200k AUC): {ts_['auc'][0]:.3f} vs {tn_['auc'][0]:.3f}, specrand lower on "
                  f"{lower}/5 -> {'specrand BLOCKS learning that none achieves' if harm else 'no blocking shown'}")
            if nb2:
                P(f"  expo lr {lr:g} vs the rule-2 best none at 200k (lr {nb2[0]:g} wd {nb2[1]:g}): {tline(paired(E, dnone[nb2], chance=chance))}")
            P(f"  expo per seed t_grok {[fmt(ts.cens(r)) for r in E]} (seed 0 failed at lr 1e-3)")
    if eb and nb:
        P(f"\nPRE-REGISTERED wording: a dim best-vs-best number may be written only if expo's lr bracket and none's lr AND wd "
          f"brackets are closed -> " + (f"\"best tested (bracket open on {'; '.join(open_on)})\"" if open_on else "all closed"))
    J["g4"] = {"best_expo": eb, "best_none50k": nb}


def latin_rows(label, rs, cut=None):
    s_all = cell_stats(rs, "all", cut)
    s_non = cell_stats(rs, "non", cut)
    s_imp = cell_stats(rs, "imp", cut)
    nimp = [int(latin.latin_split(r["frac"], r["seed"])["imp_mask"].sum()) for r in rs]
    hits = all("hits_te" in r["curve"][0] for r in rs)
    f = lambda s, k: "-" if s is None else fmt(s[k])
    return (f"| {label} | {'exact' if hits else 'bounds'} | {nimp} | {f(s_all, 'final')} | {f(s_imp, 'final')} | {f(s_non, 'final')} | "
            f"{f(s_non, 'last50')} | {'-' if s_non is None else fmt(s_non['groks'])} | {f(s_non, 't_grok_cmed')} | {f(s_non, 'auc')} |\n")


LHDR = ("| cell / arm | per-question | n implied per seed | final all | final implied | final NON-implied | last-50 non-implied | "
        "groks non-implied | t_grok non-implied cmed | AUC non-implied |\n|---|---|---|---|---|---|---|---|---|---|\n")


def read_latin(none06, pri06):
    P("\n## Latin split (coordinator (b)): held-out accuracy on Latin-implied vs NON-implied cells, every S4 arm\n")
    P("Implied = derived by naked + hidden singles from the seed's training cells (Latin-square property only; latin.py). "
      "S4 0.2 / 0.3 / 0.45: 0 implied cells for every seed, so non-implied = all held-out (exact without hits). S4 0.6: "
      "2.6-10.8% implied; exact where hits_te is logged (toyf84fu runs), bounds from n_wrong elsewhere.\n")
    P(LHDR.rstrip("\n"))
    for (lr, wd, src), rs in sorted(none06.items()):
        L.append(latin_rows(f"0.6 none lr {lr:g} wd {wd:g} ({src})", rs))
    for (a, lr, src), rs in sorted(pri06.items()):
        L.append(latin_rows(f"0.6 {a} lr {lr:g} ({src})", rs))
    for frac in (0.2, 0.3, 0.45):
        N, Pr = s4_low(frac)
        for (lr, wd), rs in sorted(N.items()):
            L.append(latin_rows(f"{frac} none lr {lr:g} wd {wd:g} (toyshape, 200k)", rs))
        for (a, lr, src), rs in sorted(Pr.items()):
            L.append(latin_rows(f"{frac} {a} lr {lr:g} ({src}, {rs[0]['steps'] // 1000}k)", rs))


def read_rescue():
    P("\n## Rescue reading per S4 fraction (pre-registered, AUDIT.md: a 'rescue' = the prior succeeds on NON-implied cells at a "
      "fraction where every tuned none config fails there)\n")
    B = 50000
    for frac in (0.2, 0.3, 0.45):
        N, Pr = s4_low(frac)
        if not N:
            P(f"- S4 {frac}: none sweep not landed (toyshape): NO READING"); continue
        if INC.get(frac):
            P(f"- S4 {frac}: NO READING (incomplete: {'; '.join(INC[frac])}); the 'every none config fails' condition needs "
              "the full pre-registered grid")
            J.setdefault("rescue", {})[str(frac)] = "NO READING (incomplete)"
            continue
        sc = {k: cell_stats(rs, "non") for k, rs in N.items()}
        fail_all = all(s["groks"][1] <= 1 and s["final"][1] <= 0.20 and s["last50"][1] <= 0.20 and s["late_gain"][1] < 0.02
                       for s in sc.values())
        bestN = max(N, key=lambda k: cell_stats(N[k], "non", B)["auc"][0])   # toyshape rule 3: highest median AUC_B
        P(f"- S4 {frac}: {len(N)} none configs; every none config fails on non-implied cells (<= 1/5 groks, median final and "
          f"last-50 <= 0.20, late gain < 0.02): {fail_all}; none config for the paired test (highest median AUC_B): lr {bestN[0]:g} wd {bestN[1]:g}")
        arms = sorted({(a, lr, src) for (a, lr, src) in Pr if a == "R@always"})
        if not arms:
            P("  R@always cells missing: NO READING"); continue
        cand = {(lr, src): cell_stats(Pr[("R@always", lr, src)], "non", B) for (_, lr, src) in arms}
        bl = max(cand, key=lambda k: cand[k]["auc"][0])
        s = cand[bl]
        Rr = Pr[("R@always", *bl)]
        t = paired(Rr, N[bestN], sub="non", cut=B, chance=1 / 24)
        tf = paired(Rr, N[bestN], sub="fresh", cut=B, chance=1 / 24)
        meets = s["groks"][0] >= NEED and t is not None and t["auc"]["pass"] and tf is not None and tf["auc"]["pass"]
        spec = Pr.get(("specrand@always", bl[0], bl[1]))
        ts_ = paired(Rr, spec, sub="non", cut=B, chance=1 / 24) if spec else None
        P(f"  R@always best lr by median AUC_B: {bl[0]:g} ({bl[1]}): groks within {B} on non-implied {fmt(s['groks'])}/{s['n']}; "
          f"vs none (non-implied): {tline(t)}; on test_fresh: {tline(tf)}")
        P(f"  vs specrand@always at that lr: {tline(ts_)}")
        verdict = ("RESCUE on non-implied cells" + ("" if ts_ and ts_["auc"]["pass"] else "; SPECIFICITY NOT SHOWN vs specrand")
                   if fail_all and meets else "not a rescue (" + ("some none config succeeds there" if not fail_all else
                                                                   "the prior does not meet the rule") + ")")
        P(f"  -> {verdict}")
        J.setdefault("rescue", {})[str(frac)] = verdict
    P("- S4 0.6: tuned none reaches final 1.000 on 5/5 seeds (lr 1e-2, wd 1), hence 1.000 on the non-implied cells: never a "
      "rescue fraction.")


# ------------------------------------------------------------------------------------------------ self-test
def selftest():
    P("# readout.py --selftest (known answers on existing data)\n")
    # 1. compare_vals == toy_struct.compare on real cells (full budget)
    for d, pairs_ in ((T / "toystruct/out/s4-f0.6-always", (("R@always", "none"), ("R@always", "specrand@always"), ("R", "specrand"))),
                      (T / "toystruct/out/dim-f0.3-oh", (("expo", "none"), ("expo", "specrand"), ("onehot", "none")))):
        runs = {}
        for f in sorted(d.glob("run_*.json")):
            r = json.loads(f.read_text()); runs.setdefault(r["arm"], []).append(r)
        rs_all = [r for v in runs.values() for r in v]
        ch = ts.chance_of(rs_all, rs_all[0]["task"])
        for a, b in pairs_:
            ref = ts.compare(runs, a, b, ch)
            mine = compare_vals([ts.cens(r) for r in sorted(runs[a], key=lambda r: r["seed"])],
                                [ts.cens(r) for r in sorted(runs[b], key=lambda r: r["seed"])],
                                [ts.auc(r) for r in sorted(runs[a], key=lambda r: r["seed"])],
                                [ts.auc(r) for r in sorted(runs[b], key=lambda r: r["seed"])], ch)
            for k in ("med_a", "med_b", "wins", "p", "pass"):
                assert ref["grok"][k] == mine["grok"][k] and ref["auc"][k] == mine["auc"][k], (d.name, a, b, k)
            assert ref["verdict"] == mine["verdict"] and ref["floor"] == mine["floor"], (d.name, a, b)
            P(f"- compare_vals == toy_struct.compare: {d.name} {a} vs {b}: {mine['verdict']}")
    # 2. the audit's S4 numbers (audit_toy_optimiser_sweeps.md §3)
    none06, pri06, _ = s4_06()
    N = none06[(1e-2, 1.0, "toynone2")]; R = pri06[("R@always", 1e-2, "burst482")]
    tn, tr = [ts.cens(r) for r in N], [ts.cens(r) for r in R]
    assert tn == [22700, 3100, 58100, 4600, 3800] and tr == [1000, 1000, 800, 1200, 1100], (tn, tr)
    assert abs(med(tn) / med(tr) - 4.6) < 1e-9 and abs(med([x for x, _ in ratios(tn, tr)]) - 3.8333) < 1e-3
    assert abs(ts.mwu_p(tr, tn) - 0.00397) < 1e-4
    a100 = med([run_metrics(r)["auc"][0] for r in N]); a5 = med([run_metrics(r, cut=5000)["auc"][0] for r in R])
    assert abs(a100 - 0.956) < 5e-4 and abs(a5 - 0.88) < 5e-3, (a100, a5)
    P(f"- S4: none lr1e-2 wd1 {tn}, R@always 1e-2 (A100) {tr}: ratio 4.60, median per-seed 3.83, Gehan p 0.0040, "
      f"none AUC 100k {a100:.3f}, R AUC <5k {a5:.3f} (audit: 4.6 / 3.8 / .004 / .956 / .88)")
    # 3. the audit's dim numbers (§2 table, §4)
    dnone, dpri, _ = dim_cells()
    s = cell_stats(dnone[(1e-2, 1.0)]); s01 = cell_stats(dnone[(1e-2, 0.1)])
    got = (s["groks"][0], round(s["auc"][0], 3), round(s["final"][0], 3), round(s["last50"][0], 3), s01["groks"][0], round(s01["last50"][0], 3))
    assert got == (0, 0.224, 0.214, 0.238, 1, 0.181), got
    f3 = cell_stats(dnone[(3e-2, 1.0)])
    assert f3["failure"] and not s["failure"], (f3["n_unstable"], s["n_unstable"])
    t = paired(dpri[("expo", 1e-3)], dnone[(1e-2, 0.1)], cut=50000, chance=1 / 27)
    assert round(t["grok"]["p"], 3) == 0.024 and t["grok"]["wins"] == 4, t["grok"]
    t1 = paired(dpri[("expo", 1e-3)], dnone[(1e-2, 1.0)], cut=50000, chance=1 / 27)
    P(f"- dim lr1e-2 wd1: {got[:4]} (audit 0/5 .224 .214 .238); wd0.1 1/5, last-50 {got[5]} (audit .181); lr 3e-2 wd 1 = "
      f"OPTIMISER FAILURE ({f3['n_unstable']}/5 unstable), lr 1e-2 wd 1 not ({s['n_unstable']}/5)")
    P(f"- expo@1e-3 vs none lr1e-2 wd0.1 @50k: t_grok wins {t['grok']['wins']}/5 p {t['grok']['p']:.4f}; AUC p {t['auc']['p']:.4f} "
      f"(vs wd 1: AUC p {t1['auc']['p']:.4f}); audit: p .024 / AUC p .075")
    assert round(t1["auc"]["p"], 3) == 0.075 or round(t["auc"]["p"], 3) == 0.075, (t["auc"]["p"], t1["auc"]["p"])
    # 4. Latin bounds bracket the logged accuracy; known answer at final 1.000
    for r in N:
        lo, hi = series(r, "non")
        k = np.array([r["n_test"] - c["n_wrong"] for c in r["curve"]])
        ni = int(latin.latin_split(0.6, r["seed"])["imp_mask"].sum())
        assert np.all(lo <= hi + 1e-12) and np.all(lo * (r["n_test"] - ni) <= k + 1e-9)
        assert r["curve"][-1]["test"] == 1.0 and lo[-1] == hi[-1] == 1.0
    P("- Latin bounds: lo <= hi everywhere; at final test 1.000 the non-implied accuracy is exactly 1.000 (toynone2 none lr1e-2 wd1)")
    # 5. A7 levels (latin.py main asserts them against the rb6 JSONs)
    latin.main()
    P("- latin.py: A7 Latin levels reproduced for rainbow seeds 0-7; recorded closure / closure_latin / n_readout equal")
    P("\nSELFTEST PASSED")


def main():
    if "--selftest" in ARGS:
        selftest()
        out = HERE / "readout_selftest.md"
        out.write_text("".join(L)); print("".join(L)); return
    P("# runs/toyf84fu read-out (pre-registered readings, AUDIT.md)\n")
    none06, pri06, g1 = s4_06()
    dnone, dpri, none_f84 = dim_cells()
    read_g1(none06, pri06, g1)
    read_g2(none06, pri06)
    read_g3(dnone)
    read_g4(dnone, dpri, none_f84)
    read_latin(none06, pri06)
    read_rescue()
    P("\n## Missing cells (not read)\n")
    for m in sorted(set(J["missing"])):
        P("-", m)
    root = NEW.parent if "--new_root" in ARGS else HERE
    (root / "readout.md").write_text("".join(L))
    (root / "readout.json").write_text(json.dumps(J, indent=1, default=str))
    print("".join(L))


if __name__ == "__main__":
    main()
