"""Pre-registered read-out of runs/pt300x (intent spec and rules: runs/pt300x/AUDIT.md "Pre-registered analysis", written
before any pt300x run). Question: on the fixed 3e8-token 7% ladder with six sizes (53M, 93M, 190M, 505M from L40S; 294M,
1.07B from B200), does the 8-seed paired area gain (prior - none, readout_rules.area) increase with ln N_nonemb?
Cells (each arm at its own pick, seeds 0-7): reused F87 ptlr 53M none@8e-3 / win-ctx@2e-3, 190M 4.8e-3 / 1.2e-3, 505M
9.6e-3 / 1.2e-3, ptlr_1b 1.07B 2.4e-3 / 6e-4 (make_jobs_pttpp.REUSED; one run per seed, no duplicates); new 93M and 294M at
the C decisions of runs/pt300x/decisions.json, recomputed here by select_pt300x.select_arm (F87 rule 1) and asserted equal.
Primary: OLS slope b of the six 8-seed mean gains on ln N_nonemb. 'SHOWN (direction)' iff ALL of
  (a) the 90% seed-bootstrap CI (seeds resampled within each size, arms paired; 10000 resamples, numpy default_rng(0), one
      draw of (10000, 8) seed indices per size in ascending N order) excludes 0 on the side of b;
  (b) the slopes on seeds 0-3 and on seeds 4-7 (point estimates) have the sign of b;
  (c) the size-level interval from the OLS residuals of the six points (t, df 4, 90%) excludes 0.
Otherwise 'NOT SHOWN' with the numbers. Any cell not final, failing an integrity check or with a run missing ->
'NOT DETERMINABLE'. A new cell outside its expected range (+-15) -> the verdict carries 'INVESTIGATE BEFORE READING'.
Descriptive (no verdict): five sizes without 1.07B; hardware (i) slope with a B200 indicator (294M, 1.07B), (ii) slope over
the L40S sizes only (53M, 93M, 190M, 505M); leave-one-size-out; none / prior separately; ln R_0.5 slope (sign only, F87
rule 8's metric); lr swaps within the tie band (seeds 0-3); the four reused points alone (= FINDINGS F97 FIXSLOPE).
    python readout_pt300x.py [--root RUNS] [--json OUT]
    python readout_pt300x.py --ka                  known answer: the four reused cells reproduce F97 FIXSLOPE
    python readout_pt300x.py --selftest [--work D] synthetic roots: positive / zero / negative slope, halves disagree,
                                                   residual interval includes 0, B200 offset, integrity, not final, range
Definitions: tools/readout_rules.py of the frozen pttpp read-out snapshot (e91ee8a8a846), unchanged."""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import select_pt300x as S   # noqa: E402  (the generator G, readout_rules R, run loading, integrity, F87 rule 1)

G, R, M = S.G, S.R, S.M
SIZES6 = ("53m", "93m", "190m", "294m", "505m", "1b")
LABEL = {"53m": "53M", "93m": "93M", "190m": "190M", "294m": "294M", "505m": "505M", "1b": "1.07B"}
HW = {"53m": "L40S", "93m": "L40S", "190m": "L40S", "294m": "B200", "505m": "L40S", "1b": "B200"}
assert all(HW[s] == G.HW[s].upper() for s in G.HW)
REUSED = {"53m": M.REUSED[("7", "53m", "21.2")], "190m": M.REUSED[("7", "190m", "2.65")],
          "505m": M.REUSED[("7", "505m", "0.745")], "1b": M.REUSED[("7", "1b", "0.318")]}
assert {s: {a: v[2] for a, v in REUSED[s].items()} for s in REUSED} == {
    "53m": {"none": "8e-3", "win-ctx": "2e-3"}, "190m": {"none": "4.8e-3", "win-ctx": "1.2e-3"},
    "505m": {"none": "9.6e-3", "win-ctx": "1.2e-3"}, "1b": {"none": "2.4e-3", "win-ctx": "6e-4"}}
NEW = tuple(G.GRID)
L40S4 = ("53m", "93m", "190m", "505m")
ALL8, H1, H2 = tuple(range(8)), (0, 1, 2, 3), (4, 5, 6, 7)
N_BOOT, TIE = R.N_BOOT, R.TIE
EXPECTED = {"93m": {"none": (60, 70), "win-ctx": (89, 89)}, "294m": {"none": (55, 67), "win-ctx": (89, 90)}}
MARGIN = 15.0
# FINDINGS F97 addendum FIXSLOPE (2026-11-02), the four reused cells (8 seeds; 1 decimal) and its numbers
F97 = {"gain": {"53m": 22.8, "190m": 23.5, "505m": 32.2, "1b": 36.1}, "none": {"53m": 66.1, "190m": 66.5, "505m": 57.5, "1b": 53.3},
       "prior": {"53m": 88.9, "190m": 90.0, "505m": 89.7, "1b": 89.4}}
# tolerances: half a unit of the published last digit + 0.001 (FINDINGS rounds some derived values, e.g. none -3.1249 -> -3.13)
F97_SLOPES = {"b": (3.25, 0.006), "ci": ((2.1, 4.5), 0.051), "resid_df2": ((0.1, 6.4), 0.051), "h1": (3.08, 0.006),
              "h2": (3.43, 0.006), "loso": ({"53m": 6.01, "190m": 3.07, "505m": 3.19, "1b": 2.57}, 0.006),
              "none": (-3.13, 0.006), "prior": (0.13, 0.006)}


# ---------------------------------------------------------------- cells
def reused_cell(root, size):
    c = dict(size=size, problems=[], pick={}, alts={"none": {}, "win-ctx": {}})
    for arm, (src, tag, lr) in REUSED[size].items():
        c["pick"][arm] = lr
        areas, ts, bud = [], [], None
        for s in ALL8:
            f = Path(root) / src / tag / f"run_whole_{G.target_of(arm)}_seed{s}_{tag}.json"
            r = S.load_json(f)
            if r is None or not R.finished(r):
                c["problems"].append(f"{LABEL[size]} {arm} seed {s}: {'missing' if r is None else 'unfinished'} ({f})")
                areas.append(np.nan); ts.append(np.nan)
                continue
            c["problems"] += [f"{LABEL[size]} {arm}@{lr} seed {s}: {w}" for w in S.integrity(r, size, arm, lr, s)]
            areas.append(float(R.area(r))); ts.append(R.t_theta(r, 0.5)); bud = R.budget_M(r)
        c["an" if arm == "none" else "ap"] = np.array(areas)
        c["tn" if arm == "none" else "tp"] = np.array(ts, float)
        c["bud"] = bud
        if size != "1b":   # lr alternatives on seeds 0-3 (F87 ptlr grid dirs); ptlr_1b's have seeds 0-1 only: not varied
            pre = tag.split("-lr")[0]
            for d in sorted((Path(root) / src).glob(f"{pre}-lr*-7pct-{arm}")):
                alt = d.name[len(pre) + 3:-len(f"-7pct-{arm}")]
                if alt == lr:
                    continue
                rs = [S.load_json(d / f"run_whole_{G.target_of(arm)}_seed{s}_{d.name}.json") for s in H1]
                if all(r is not None and R.finished(r) and not S.integrity(r, size, arm, alt, s) for r, s in zip(rs, H1)):
                    c["alts"][arm][alt] = np.array([float(R.area(r)) for r in rs])
    return c


def new_cell(root, size, state, own):
    c = dict(size=size, problems=[], pick={}, alts={"none": {}, "win-ctx": {}}, notfinal=[])
    for arm in G.ARMS:
        st = state[(size, arm)]
        if st["STOP"] or not st["C"]:
            c["notfinal"].append(f"{LABEL[size]} {arm}: {'STOP (the user decides)' if st['STOP'] else 'no C decision yet'}")
            continue
        pick = c["pick"][arm] = st["C"]
        sel = S.select_arm(root, size, arm, st["X"], own)
        if sel["status"] != "ok" or sel["best"] != pick:
            c["problems"].append(f"{LABEL[size]} {arm}: the recomputed selection is {sel['status']} / "
                                 f"{sel.get('best')} but the decided C is {pick}")
            continue
        recs = [sel["recs"][(pick, s)] for s in H1] + [S.new_run(root, size, arm, pick, s, own) for s in H2]
        for v in recs:
            if v["status"] != "ok":
                (c["notfinal"] if v["status"] in ("missing", "running", "nocheck") else c["problems"]).append(
                    f"{v['name']}: {v['status']}{(' ' + '; '.join(v['bad'])) if v['bad'] else ''}")
        if any(v["status"] != "ok" for v in recs):
            continue
        c["an" if arm == "none" else "ap"] = np.array([v["area"] for v in recs])
        c["tn" if arm == "none" else "tp"] = np.array([R.t_theta(v["r"], 0.5) if v["r"] else np.nan for v in recs], float)
        c["bud"] = 18310 * 32 * 512 / 1e6
        for lr in sel["grid"]:
            if lr != pick:
                c["alts"][arm][lr] = np.array([sel["recs"][(lr, s)]["area"] for s in H1])
        c["own"] = c.get("own", []) + [v["name"] for v in recs if v["own"]]
    return c


def finish(c):
    c["nne"] = M.SIZES[c["size"]][5]
    c["x"] = math.log(c["nne"])
    c["ok"] = "an" in c and "ap" in c and not c["problems"] and not c.get("notfinal")
    if c["ok"]:
        c["d"] = c["ap"] - c["an"]
    return c


# ---------------------------------------------------------------- statistics
def ols(x, Y):
    """slope of Y (..., n) on x (n)"""
    x = np.asarray(x, float)
    xc = x - x.mean()
    Y = np.asarray(Y, float)
    return (Y - Y.mean(-1, keepdims=True)) @ xc / (xc @ xc)


def ols_cov(X, y):
    """multiple OLS: coefficients, residual-based standard errors, df"""
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = len(y) - X.shape[1]
    res = y - X @ beta
    s2 = res @ res / df if df > 0 else np.nan
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))) if df > 0 else np.full(X.shape[1], np.nan)
    return beta, se, df


def boot_idx(n_cells, n_seeds):
    """one (10000, n_seeds) draw per size, in the cells' (ascending N) order, from one default_rng(0) stream"""
    rng = np.random.default_rng(0)
    return [rng.integers(0, n_seeds, size=(N_BOOT, n_seeds)) for _ in range(n_cells)]


def interval(b, se, df, level=0.90):
    if not (df > 0 and np.isfinite(se)):
        return (np.nan, np.nan, np.nan)
    t = stats.t.ppf(0.5 + level / 2, df)
    p = 2 * stats.t.sf(abs(b) / se, df) if se > 0 else 0.0
    return (b - t * se, b + t * se, p)


def slope_block(cells, metric="d", seeds=ALL8, design="plain"):
    """point slope, the within-size seed-bootstrap 90% CI, the residual t interval (90%), per-cell means;
    design 'b200': gain ~ a + b ln N + c [B200]"""
    cells = sorted(cells, key=lambda c: c["x"])
    pos = np.array(seeds)
    x = np.array([c["x"] for c in cells])
    y = np.array([c[metric][pos].mean() for c in cells])
    idx = boot_idx(len(cells), len(seeds))
    Yb = np.stack([c[metric][pos][i].mean(1) for c, i in zip(cells, idx)], -1)
    if design == "plain":
        b = float(ols(x, y))
        bb = ols(x, Yb)
        beta, se, df = ols_cov(np.c_[np.ones_like(x), x], y)
        extra = {}
    else:
        I = np.array([1.0 if HW[c["size"]] == "B200" else 0.0 for c in cells])
        X = np.c_[np.ones_like(x), x, I]
        beta, se, df = ols_cov(X, y)
        b = float(beta[1])
        bb = np.linalg.lstsq(X, Yb.T, rcond=None)[0][1]
        extra = {"c_b200": float(beta[2]), "c_se": float(se[2])}
    lo, hi = (float(np.percentile(bb, 5)), float(np.percentile(bb, 95)))
    rlo, rhi, p = interval(b, se[1], df)
    return dict(b=b, lo=lo, hi=hi, rlo=float(rlo), rhi=float(rhi), p=float(p), df=int(df), n=len(cells),
                sizes=[c["size"] for c in cells], means=[float(v) for v in y], **extra)


def sgn(v):
    return 1 if v > 0 else -1 if v < 0 else 0


def verdict(main, h1, h2):
    d = sgn(main["b"])
    a = d != 0 and ((d > 0 and main["lo"] > 0) or (d < 0 and main["hi"] < 0))
    b = d != 0 and sgn(h1["b"]) == d and sgn(h2["b"]) == d
    c = d != 0 and ((d > 0 and main["rlo"] > 0) or (d < 0 and main["rhi"] < 0))
    crit = {"a": a, "b": b, "c": c}
    nums = (f"b {main['b']:+.3f} per e-fold of N_nonemb; (a) seed bootstrap 90% CI [{main['lo']:+.3f}, {main['hi']:+.3f}] "
            f"{'excludes' if a else 'does not exclude'} 0; (b) seeds 0-3 {h1['b']:+.3f}, seeds 4-7 {h2['b']:+.3f}: "
            f"{'same sign' if b else 'NOT the same sign'}; (c) residual t interval (df {main['df']}, 90%) "
            f"[{main['rlo']:+.3f}, {main['rhi']:+.3f}] (p {main['p']:.3f}) {'excludes' if c else 'does not exclude'} 0")
    if all(crit.values()):
        return f"SHOWN ({'increase' if d > 0 else 'decrease'}): the gain {'increases' if d > 0 else 'decreases'} with ln N_nonemb; {nums}", crit
    return f"NOT SHOWN (failed: {', '.join('(' + k + ')' for k, v in crit.items() if not v)}); {nums}", crit


def lnR_block(cells):
    rows, pts = [], []
    for c in sorted(cells, key=lambda c: c["x"]):
        sp = R.speedup(c["tn"], c["tp"], c["bud"])
        rows.append((c["size"], sp))
        if sp["ratio"] is not None and sp["mark"] == "":
            pts.append((c["x"], math.log(sp["ratio"])))
    b = float(ols([p[0] for p in pts], [p[1] for p in pts])) if len(pts) >= 2 else None
    return rows, b, len(pts)


def swaps(cells, base):
    """seeds-0-3 slope with one arm of one cell at another lr within the tie band (|mean - pick mean| <= 2, seeds 0-3)"""
    out = []
    for c in cells:
        for arm, alts in c["alts"].items():
            mine = c["an" if arm == "none" else "ap"][:4]
            for lr, v in sorted(alts.items(), key=lambda kv: float(kv[0])):
                if abs(v.mean() - mine.mean()) > TIE:
                    continue
                cc = [dict(o) for o in cells]
                k = [o["size"] for o in cells].index(c["size"])
                key = "an" if arm == "none" else "ap"
                arr = cc[k][key].copy(); arr[:4] = v; cc[k][key] = arr
                cc[k]["d"] = cc[k]["ap"] - cc[k]["an"]
                b = slope_block(cc, seeds=H1)["b"]
                out.append((c["size"], arm, c["pick"][arm], lr, float(v.mean() - mine.mean()), b, sgn(b) != sgn(base)))
    return out


# ---------------------------------------------------------------- the read-out
def readout(root, out=print, check_findings=True):
    root = Path(root)
    log, state = G.decisions(root / "pt300x" / "decisions.json")
    own = S.parse_own(root)
    cells = {s: finish(reused_cell(root, s)) for s in REUSED}
    cells.update({s: finish(new_cell(root, s, state, own)) for s in NEW})
    out(f"# pt300x read-out; root {root}; generator {S.sha(S.GEN)}; readout_rules {S.sha(S.RULES_DIR / 'readout_rules.py')}; "
        f"decisions {len(log)} entries")
    out("# 4 of the 6 points (53M, 190M, 505M, 1.07B) were observed before this analysis was registered (F87, addenda, F97 "
        "FIXSLOPE): the test is only partly confirmatory; 93M, 294M and criterion (c) are the new evidence")
    res = {"cells": {}, "problems": [], "notfinal": []}
    if check_findings:   # the reused cells must still be the published ones
        for s in REUSED:
            c = cells[s]
            if c["ok"]:
                got = {"gain": c["d"].mean(), "none": c["an"].mean(), "prior": c["ap"].mean()}
                bad = {k: round(float(v), 2) for k, v in got.items() if abs(v - F97[k][s]) > 0.05 + 1e-9}
                if bad:
                    c["problems"].append(f"{LABEL[s]} differs from FINDINGS F97 FIXSLOPE: {bad} vs {({k: F97[k][s] for k in bad})}")
                    c["ok"] = False
    for s in SIZES6:
        res["problems"] += cells[s]["problems"]
        res["notfinal"] += cells[s].get("notfinal", [])
    out("\n## integrity and completeness")
    for p in res["problems"]:
        out(f"  PROBLEM {p}")
    for p in res["notfinal"]:
        out(f"  NOT FINAL {p}")
    if not res["problems"] and not res["notfinal"]:
        out("  every counted run present, finished, intended (integrity re-check; new runs: CHECK7 passed, double path)")
    for s in NEW:
        if cells[s].get("own"):
            out(f"  own failures counted (F87 rule 2): {cells[s]['own']}")
    ok = [cells[s] for s in SIZES6 if cells[s]["ok"]]
    flags = []
    out("\n## expected ranges of the new cells (AUDIT.md; outside +-15 -> investigate before reading)")
    for s in NEW:
        c = cells[s]
        if not c["ok"]:
            out(f"  {LABEL[s]}: not available")
            continue
        for arm, key in (("none", "an"), ("win-ctx", "ap")):
            lo, hi = EXPECTED[s][arm]
            v = float(c[key].mean())
            bad = not (lo - MARGIN <= v <= hi + MARGIN)
            out(f"  {LABEL[s]} {arm}: 8-seed mean {v:.1f}; expected {lo}-{hi} (+-{MARGIN:.0f}) {'OUTSIDE: INVESTIGATE' if bad else 'ok'}")
            if bad:
                flags.append(f"{LABEL[s]} {arm} {v:.1f}")
    out("\n## cells at the picks (8 seeds; area points; gain = prior - none, paired; R_0.5 = median t50(none) / median t50(prior), censored at the budget)")
    out(f"  {'size':6s} {'hw':5s} {'N_nonemb':>12s} {'ln N':>6s} {'none lr':>8s} {'none':>6s} {'prior lr':>8s} {'prior':>6s} "
        f"{'gain':>6s} {'[seed CI]':>15s} {'s0-3':>6s} {'s4-7':>6s} {'wins':>5s} {'R_0.5':>8s}")
    for c in sorted(ok, key=lambda c: c["x"]):
        d = c["d"]
        bi = boot_idx(1, 8)[0]
        bd = d[bi].mean(1)
        sp = R.speedup(c["tn"], c["tp"], c["bud"])
        rtxt = "none" if sp["ratio"] is None else f"{sp['mark']}{sp['ratio']:.2f}"
        out(f"  {LABEL[c['size']]:6s} {HW[c['size']]:5s} {c['nne']:12,d} {c['x']:6.3f} {c['pick']['none']:>8s} {c['an'].mean():6.1f} "
            f"{c['pick']['win-ctx']:>8s} {c['ap'].mean():6.1f} {d.mean():+6.1f} [{np.percentile(bd, 5):+6.1f}, {np.percentile(bd, 95):+6.1f}] "
            f"{d[:4].mean():+6.1f} {d[4:].mean():+6.1f} {int((d > 0).sum()):>3d}/8 {rtxt:>8s}")
        res["cells"][c["size"]] = dict(hw=HW[c["size"]], nne=c["nne"], pick=c["pick"], none=c["an"].tolist(), prior=c["ap"].tolist(),
                                       gain=float(d.mean()), r50=sp["ratio"], r50_mark=sp["mark"])
    if len(ok) < len(SIZES6):
        v = f"NOT DETERMINABLE ({len(SIZES6) - len(ok)} of 6 cells not available: " + \
            "; ".join((res["problems"] + res["notfinal"])[:3]) + ")"
        out(f"\n## PRIMARY\n  VERDICT: {v}")
        res["verdict"] = v
        return res
    main = slope_block(ok)
    h1, h2 = slope_block(ok, seeds=H1), slope_block(ok, seeds=H2)
    v, crit = verdict(main, h1, h2)
    if flags:
        v += f"  [INVESTIGATE BEFORE READING: outside the expected range: {', '.join(flags)}]"
    out("\n## PRIMARY (pre-registered): OLS slope of the six 8-seed mean gains on ln N_nonemb")
    out(f"  b = {main['b']:+.3f} per e-fold; seed bootstrap 90% CI [{main['lo']:+.3f}, {main['hi']:+.3f}]; residual t (df "
        f"{main['df']}) 90% [{main['rlo']:+.3f}, {main['rhi']:+.3f}], p {main['p']:.3f}")
    out(f"  halves: seeds 0-3 {h1['b']:+.3f}, seeds 4-7 {h2['b']:+.3f}")
    out(f"  VERDICT: {v}")
    res.update(primary=main, h1=h1, h2=h2, criteria=crit, verdict=v, investigate=flags)
    out("\n## descriptive (no verdict)")
    no1b = slope_block([c for c in ok if c["size"] != "1b"])
    out(f"  five sizes without 1.07B (294M is B200 in this batch, so not L40S-only): {no1b['b']:+.3f} [{no1b['lo']:+.3f}, "
        f"{no1b['hi']:+.3f}], residual t (df {no1b['df']}) [{no1b['rlo']:+.3f}, {no1b['rhi']:+.3f}]")
    hw1 = slope_block(ok, design="b200")
    out(f"  hardware (i) gain ~ a + b ln N + c [B200: 294M, 1.07B]: b {hw1['b']:+.3f} [{hw1['lo']:+.3f}, {hw1['hi']:+.3f}], "
        f"residual t (df {hw1['df']}) [{hw1['rlo']:+.3f}, {hw1['rhi']:+.3f}]; c {hw1['c_b200']:+.2f} (se {hw1['c_se']:.2f})")
    hw2 = slope_block([c for c in ok if c["size"] in L40S4])
    out(f"  hardware (ii) L40S sizes only (53M, 93M, 190M, 505M): {hw2['b']:+.3f} [{hw2['lo']:+.3f}, {hw2['hi']:+.3f}], "
        f"residual t (df {hw2['df']}) [{hw2['rlo']:+.3f}, {hw2['rhi']:+.3f}]")
    loso = {s: slope_block([c for c in ok if c["size"] != s])["b"] for s in SIZES6}
    out("  leave one size out: " + ", ".join(f"-{LABEL[s]} {b:+.3f}" for s, b in loso.items()))
    arms = {k: slope_block(ok, metric=m) for k, m in (("none", "an"), ("prior", "ap"))}
    out("  arms separately (mean area on ln N): " + "; ".join(f"{k} {v_['b']:+.3f} [{v_['lo']:+.3f}, {v_['hi']:+.3f}]"
                                                         for k, v_ in arms.items()))
    rows, bR, nR = lnR_block(ok)
    out("  ln R_0.5 (secondary, sign only; F87 rule 8's metric): R_0.5 " + ", ".join(
        f"{LABEL[s]} {'none' if sp['ratio'] is None else sp['mark'] + format(sp['ratio'], '.2f')}" for s, sp in rows)
        + (f"; slope over the {nR} point cells {bR:+.4f} ({'same sign as' if sgn(bR) == sgn(main['b']) else 'OTHER sign than'} the primary)"
           if bR is not None else "; slope not determinable (fewer than 2 point cells)"))
    sw = swaps(ok, h1["b"])
    out(f"  lr swaps within the tie band (seeds 0-3; slope at the picks {h1['b']:+.3f}; 1.07B's alternatives have seeds 0-1 only, not varied):"
        + ("" if sw else " none in the band"))
    for s, arm, p, lr, dm, b, flip in sw:
        out(f"    {LABEL[s]} {arm} {p} -> {lr} (mean {dm:+.2f}): {b:+.3f}{'  SIGN CHANGE' if flip else ''}")
    out(f"    {sum(x[-1] for x in sw)} of {len(sw)} swaps change the sign")
    four = [c for c in ok if c["size"] in REUSED]
    f4 = slope_block(four)
    out(f"  the four reused points alone (F97 FIXSLOPE): {f4['b']:+.3f} [{f4['lo']:+.3f}, {f4['hi']:+.3f}], residual t (df "
        f"{f4['df']}) [{f4['rlo']:+.3f}, {f4['rhi']:+.3f}]")
    res.update(no1b=no1b, hw_b200=hw1, hw_l40s=hw2, loso=loso, arms=arms, lnR50=dict(b=bR, n=nR), swaps=sw, four=f4)
    return res


# ---------------------------------------------------------------- known answer: F97 FIXSLOPE on the real reused cells
def ka(root):
    cells = [finish(reused_cell(root, s)) for s in REUSED]
    probs = [p for c in cells for p in c["problems"]]
    ok, n = 0, 0

    def T(name, got, want, tol):
        nonlocal ok, n
        n += 1
        g = np.atleast_1d(np.asarray(got, float)); w = np.atleast_1d(np.asarray(want, float))
        good = bool(np.all(np.abs(g - w) <= tol))
        ok += good
        print(f"{'PASS' if good else 'FAIL'} {name}: got {np.round(g, 3).tolist()} want {w.tolist()} (tol {tol})")

    n += 1
    ok += not probs
    print(f"{'PASS' if not probs else 'FAIL'} the 64 reused runs pass the integrity re-check" + (f": {probs[:3]}" if probs else ""))
    if probs:
        print(f"KA FAIL {ok}/{n}")
        return False
    for k, key in (("gain", "d"), ("none", "an"), ("prior", "ap")):
        T(f"8-seed {k} per size = FINDINGS (1 decimal)", [c[key].mean() for c in cells], [F97[k][c["size"]] for c in cells], 0.05 + 1e-9)
    m = slope_block(cells)
    T("slope", m["b"], F97_SLOPES["b"][0], F97_SLOPES["b"][1])
    T("seed bootstrap 90% CI (within size)", [m["lo"], m["hi"]], F97_SLOPES["ci"][0], F97_SLOPES["ci"][1])
    T("residual t interval df 2", [m["rlo"], m["rhi"]], F97_SLOPES["resid_df2"][0], F97_SLOPES["resid_df2"][1])
    T("seeds 0-3 slope", slope_block(cells, seeds=H1)["b"], F97_SLOPES["h1"][0], F97_SLOPES["h1"][1])
    T("seeds 4-7 slope", slope_block(cells, seeds=H2)["b"], F97_SLOPES["h2"][0], F97_SLOPES["h2"][1])
    lw, tol = F97_SLOPES["loso"]
    T("leave one size out (-53M, -190M, -505M, -1.07B)", [slope_block([c for c in cells if c["size"] != s])["b"] for s in lw],
      list(lw.values()), tol)
    T("none arm slope", slope_block(cells, metric="an")["b"], F97_SLOPES["none"][0], F97_SLOPES["none"][1])
    T("prior arm slope", slope_block(cells, metric="ap")["b"], F97_SLOPES["prior"][0], F97_SLOPES["prior"][1])
    print(f"KA {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    return ok == n


# ---------------------------------------------------------------- selftest (synthetic roots)
def fake_reused(root, size, arm, seed, area):
    src, tag, lr = REUSED[size][arm]
    F = G.facts("93m", arm, G.GRID["93m"][arm][0], 0, "A")
    a = dict(G.intent_x(F)["args"])
    d, nl, nh, dff, n_params, _ = M.SIZES[size]
    a.update(out=str(Path(root) / src / tag), tag=f"_{tag}", seed=seed, lr=float(lr), d_model=d, n_layers=nl, n_heads=nh,
             d_ff=dff, save_model=True, ckpt_every=500 if size == "1b" else 2000)
    r = {"args": a, "steps": 18310, "geo_layers": [nl // 3, 2 * nl // 3], "read_layers": [nl // 3, 2 * nl // 3],
         "n_params": n_params, "grad_ckpt_on": n_params > 150e6, "lm_chunk": 0, "eval_every_steps": 250,
         "code_sha": M.PIN7_SHA, "prior_events": G.WINDOW_EVENTS if arm != "none" else [],
         "curve": S.fake_curve(area, M.lm0_ref("7", size, seed), arm)}
    f = Path(root) / src / tag / f"run_whole_{G.target_of(arm)}_seed{seed}_{tag}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(r))
    return f


PICK = {"93m": {"none": "4.8e-3", "win-ctx": "1.2e-3"}, "294m": {"none": "4.8e-3", "win-ctx": "6e-4"}}


def build_root(base, name, gain, prior=92.0, q=None, decided=None, tweak=None, alt_tie=False):
    """gain(size, seed) -> designed gain; prior area = prior + q[seed]; none = prior area - gain. New cells: the pick has the
    designed areas, every other grid lr 10 points lower (alt_tie: 93M win-ctx 2.4e-3 one point lower, higher bpb)."""
    root = Path(base) / name
    (root / "pt300x").mkdir(parents=True)
    q = q or [0.5, -0.5, 1.0, -1.0, 0.5, -0.5, 1.0, -1.0]
    val = lambda s, arm, seed: (prior + q[seed]) - (gain(s, seed) if arm == "none" else 0.0)
    for s in REUSED:
        for arm in G.ARMS:
            for seed in ALL8:
                fake_reused(root, s, arm, seed, val(s, arm, seed))
    log = []
    for s in NEW:
        for arm in G.ARMS:
            pk = PICK[s][arm]
            for lr in G.GRID[s][arm]:
                for seed in H1:
                    v = val(s, arm, seed) - (0 if lr == pk else (1 if alt_tie and (s, arm, lr) == ("93m", "win-ctx", "2.4e-3") else 10))
                    S.fake_new_run(root, s, arm, lr, seed, v, bpb=1.19 if lr == pk else 1.2, check="synthetic")
            for seed in H2:
                S.fake_new_run(root, s, arm, pk, seed, val(s, arm, seed), check="synthetic")
            if decided is None or (s, arm) in decided:
                log.append(dict(seq=len(log), size=s, arm=arm, stage="C", lr=pk))
    (root / "pt300x" / "decisions.json").write_text(json.dumps({"log": log}))
    if tweak:
        tweak(root)
    return root


def selftest(work):
    base = Path(tempfile.mkdtemp(prefix="pt300x_readout_selftest_", dir=work))
    ok, n = 0, 0

    def T(name, cond, info=""):
        nonlocal ok, n
        n += 1
        ok += bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info and not cond else ""))

    xs = {s: math.log(M.SIZES[s][5]) for s in SIZES6}
    m = float(np.mean(list(xs.values())))
    lines = []
    quiet = lambda t="": lines.append(t)
    sym = [1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 2.0, -2.0]   # zero mean within each half
    try:
        r = readout(build_root(base, "pos", lambda s, k: 30 + 3 * (xs[s] - m) + sym[k], alt_tie=True), out=quiet, check_findings=False)
        T("positive slope: SHOWN (increase)", r["verdict"].startswith("SHOWN (increase)"), r["verdict"])
        T("  designed slope 3 read +3.000 (8 seeds, both halves); residual interval degenerate at 3",
          abs(r["primary"]["b"] - 3) < 1e-9 and abs(r["h1"]["b"] - 3) < 1e-9 and abs(r["h2"]["b"] - 3) < 1e-9
          and abs(r["primary"]["rlo"] - 3) < 1e-6, (r["primary"]["b"], r["h1"]["b"], r["h2"]["b"], r["primary"]["rlo"]))
        T("  arms: none -3.000, prior 0.000; B200-indicator b 3.000 c 0.00; L40S-only 3.000",
          abs(r["arms"]["none"]["b"] + 3) < 1e-9 and abs(r["arms"]["prior"]["b"]) < 1e-9 and abs(r["hw_b200"]["b"] - 3) < 1e-9
          and abs(r["hw_b200"]["c_b200"]) < 1e-9 and abs(r["hw_l40s"]["b"] - 3) < 1e-9)
        T("  ln R_0.5 slope positive (none slower at larger N), 6 point cells", r["lnR50"]["b"] > 0 and r["lnR50"]["n"] == 6, r["lnR50"])
        T("  the tied 93M win-ctx alternative (2.4e-3, -1 point) is swapped, no sign change; no other swap",
          len(r["swaps"]) == 1 and r["swaps"][0][:4] == ("93m", "win-ctx", "1.2e-3", "2.4e-3") and not r["swaps"][0][-1], r["swaps"])
        T("  no INVESTIGATE flag, no problem", not r["investigate"] and not r["problems"] and not r["notfinal"])
        r = readout(build_root(base, "zero", lambda s, k: 30 + [3.0, -1.0, 2.0, -4.0, 1.0, -3.0, 4.0, -2.0][k]), out=quiet, check_findings=False)
        T("zero slope: NOT SHOWN, (a) failed", r["verdict"].startswith("NOT SHOWN") and not r["criteria"]["a"] and abs(r["primary"]["b"]) < 1e-9,
          r["verdict"])
        r = readout(build_root(base, "neg", lambda s, k: 30 - 3 * (xs[s] - m) + sym[k]), out=quiet, check_findings=False)
        T("negative slope: SHOWN (decrease), -3.000", r["verdict"].startswith("SHOWN (decrease)") and abs(r["primary"]["b"] + 3) < 1e-9,
          r["verdict"])
        halves = lambda s, k: 30 + (6.0 if k < 4 else -1.5) * (xs[s] - m) + [1, -1, 1, -1][k % 4]
        r = readout(build_root(base, "halves", halves), out=quiet, check_findings=False)
        T("halves disagree (seeds 0-3 +6, seeds 4-7 -1.5): NOT SHOWN, (b) failed, 8-seed +2.25",
          r["verdict"].startswith("NOT SHOWN") and not r["criteria"]["b"] and abs(r["primary"]["b"] - 2.25) < 1e-9
          and abs(r["h1"]["b"] - 6) < 1e-9 and abs(r["h2"]["b"] + 1.5) < 1e-9, r["verdict"])
        X = np.c_[np.ones(6), [xs[s] for s in SIZES6]]
        e = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        e = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]
        e = 8 * e / np.abs(e).max()
        resid = dict(zip(SIZES6, e))
        tiny = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        r = readout(build_root(base, "resid", lambda s, k: 30 + 1.0 * (xs[s] - m) + resid[s] + tiny[k]), out=quiet, check_findings=False)
        T("residual interval includes 0 (b 1.0, size scatter +-8, seed noise +-0.1): NOT SHOWN, (c) failed, (a) and (b) passed",
          r["verdict"].startswith("NOT SHOWN") and r["criteria"]["a"] and r["criteria"]["b"] and not r["criteria"]["c"]
          and abs(r["primary"]["b"] - 1) < 1e-9 and r["primary"]["rlo"] < 0 < r["primary"]["rhi"], r["verdict"])
        r = readout(build_root(base, "b200", lambda s, k: 30 + 2 * (xs[s] - m) + (6.0 if HW[s] == "B200" else 0.0) + sym[k]),
                    out=quiet, check_findings=False)
        T("B200 offset +6 on 294M / 1.07B: hardware (i) b 2.000 and c 6.000; (ii) L40S-only 2.000; plain slope differs",
          abs(r["hw_b200"]["b"] - 2) < 1e-9 and abs(r["hw_b200"]["c_b200"] - 6) < 1e-9 and abs(r["hw_l40s"]["b"] - 2) < 1e-9
          and abs(r["primary"]["b"] - 2) > 0.1, (r["hw_b200"], r["hw_l40s"]["b"], r["primary"]["b"]))

        def bad_sha(root):
            f = next((root / "ptlr" / "out" / "190m-lr4.8e-3-7pct-none").glob("run_*seed5_*.json"))
            j = json.loads(f.read_text()); j["code_sha"] = "000000000000"; f.write_text(json.dumps(j))
        r = readout(build_root(base, "integ", lambda s, k: 30 + 3 * (xs[s] - m) + sym[k], tweak=bad_sha), out=quiet, check_findings=False)
        T("a reused run with another code_sha: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("code_sha" in p for p in r["problems"]), r["verdict"])

        def bad_check(root):
            tag, rf, cf = G.run_file("294m", "none", PICK["294m"]["none"], 6)
            (root / "pt300x" / "out" / tag / cf).write_text(json.dumps({"pass": False, "checks": [{"check": "x", "ok": False}],
                                                                         "info": {"area": 0}}))
        r = readout(build_root(base, "check", lambda s, k: 30 + 3 * (xs[s] - m) + sym[k], tweak=bad_check), out=quiet, check_findings=False)
        T("a C run whose CHECK7 failed: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("CHECK7 FAILED" in p for p in r["problems"]), r["verdict"])
        dec = {(s, a) for s in NEW for a in G.ARMS} - {("294m", "win-ctx")}
        r = readout(build_root(base, "notfinal", lambda s, k: 30 + 3 * (xs[s] - m) + sym[k], decided=dec), out=quiet, check_findings=False)
        T("294M win-ctx without its C decision: NOT DETERMINABLE (not final)", r["verdict"].startswith("NOT DETERMINABLE")
          and any("no C decision" in p for p in r["notfinal"]), r["verdict"])

        def miss_c(root):
            tag, rf, cf = G.run_file("93m", "none", PICK["93m"]["none"], 7)
            (root / "pt300x" / "out" / tag / rf).unlink()
        r = readout(build_root(base, "missing", lambda s, k: 30 + 3 * (xs[s] - m) + sym[k], tweak=miss_c), out=quiet, check_findings=False)
        T("a C run missing: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE") and any("missing" in p for p in r["notfinal"]),
          r["verdict"])

        def wrong_c(root):
            f = root / "pt300x" / "decisions.json"
            j = json.loads(f.read_text())
            for e_ in j["log"]:
                if (e_["size"], e_["arm"]) == ("93m", "none"):
                    e_["lr"] = "9.6e-3"
            f.write_text(json.dumps(j))
        r = readout(build_root(base, "wrongc", lambda s, k: 30 + 3 * (xs[s] - m) + sym[k], tweak=wrong_c), out=quiet, check_findings=False)
        T("a C decision that is not the recomputed pick: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("recomputed selection" in p for p in r["problems"]), r["verdict"])
        r = readout(build_root(base, "range", lambda s, k: 30 + 3 * (xs[s] - m) + sym[k], prior=73.0), out=quiet, check_findings=False)
        T("prior at ~73 (outside 89 +- 15): the verdict carries INVESTIGATE BEFORE READING", "INVESTIGATE BEFORE READING" in r["verdict"]
          and len(r["investigate"]) == 2, r["verdict"])
        r = readout(build_root(base, "findings", lambda s, k: 30 + 3 * (xs[s] - m) + sym[k]), out=quiet, check_findings=True)
        T("with the FINDINGS check on, designed reused cells != F97 -> NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("FINDINGS F97" in p for p in r["problems"]), r["verdict"])
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"READOUT SELFTEST {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    return ok == n


def main():
    if "--selftest" in sys.argv:
        w = sys.argv[sys.argv.index("--work") + 1] if "--work" in sys.argv else None
        sys.exit(0 if selftest(w) else 1)
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else S.ROOT_DEFAULT
    if "--ka" in sys.argv:
        sys.exit(0 if ka(root) else 1)
    res = readout(root)
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
