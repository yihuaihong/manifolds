"""Pre-registered read-out of runs/ptsz (intent spec and rules: runs/ptsz/AUDIT.md "Pre-registered analysis", written and
self-tested before any ptsz run). Question: on the fixed 3e8-token 7% ladder, all on L40S, does the 8-seed paired area gain
(prior - none, readout_rules.area) increase with ln N_nonemb?
Cells (each arm at its own pick, seeds 0-7, gain_s = area(win-ctx, s) - area(none, s), paired by seed):
  reused L40S: F87 ptlr 53M none@8e-3 / win-ctx@2e-3, 190M 4.8e-3 / 1.2e-3, 505M 9.6e-3 / 1.2e-3 (make_jobs_pttpp.REUSED);
               pt300x 93M 2.4e-3 / 1.2e-3 (runs/pt300x/decisions.json; CHECK7 files required);
  new L40S:    136m, 294m (the L40S re-run), 391m, 731m at the C decisions of runs/ptsz/decisions.json, recomputed here by
               select_ptsz.select_arm (F87 rule 1) and asserted equal; seeds 0-3 from stage A / X, 4-7 from C;
  B200 (descriptive only): pt300x 294M 9.6e-3 / 6e-4 and ptlr_1b 1.07B 2.4e-3 / 6e-4 (F99's two B200 points).
PRIMARY: OLS slope b of the eight L40S 8-seed mean gains on ln N_nonemb. 'SHOWN (direction)' iff ALL of
  (a) the 90% seed-bootstrap CI (seeds resampled within each size, arms paired; 10000 resamples, numpy default_rng(0), one
      draw of (10000, 8) seed indices per size in ascending N order) excludes 0 on the side of b;
  (b) the slopes on seeds 0-3 and on seeds 4-7 (point estimates) have the sign of b;
  (c) the size-level interval from the OLS residuals of the eight points (t, df 6, 90%) excludes 0.
SECONDARY (the confirmatory test for F99 I1; named separately): the same slope over the three never-seen sizes only
  (136m, 391m, 731m; 294m excluded: its B200 twin was seen), criteria (a) and (b) only; its df-1 residual interval is
  printed, not required.
Otherwise 'NOT SHOWN' with the numbers. A primary cell not final / failing an integrity check / with a run missing ->
'NOT DETERMINABLE' (the secondary likewise for its three cells). A new cell outside its expected range (+-15) -> the
verdict lines carry 'INVESTIGATE BEFORE READING'.
Robustness (amendment I1, after the fresh build audit, before any result; printed next to both verdicts, never a criterion):
the slope of the per-cell MEDIAN gain, of the trimmed-mean gain (min and max seed dropped per cell), the leave-one-seed-out
range (every cell x seed left out: min / max and which), and the count of none seeds with area < 45 (late learners); the
same three slopes for the 10-point descriptive slope; the 'few seeds' label (amendment I-0). Own failures (amendment I5):
an HONOURED own-failure entry (select_ptsz.own_failures) makes its size cell NOT DETERMINABLE (never scored). The
secondary's pre-registered false-SHOWN rates are printed with its verdict. NOT read-only: the first call that sees a final
new cell writes runs/ptsz/FIRST_READOUT.json (atomic, read-only; amendments I-A / 3), and a call refuses (RuntimeError) if
that file cannot be parsed, or is missing while runs/ptsz/readout.json / readout.txt already show a final new cell.
Descriptive (no verdict): the 10-point all-hardware slope (+ F99's B200 294M and 1.07B; with and without a B200 indicator);
the B200-vs-L40S 294M difference per arm and in gain (the hardware bridge at 294M); leave-one-size-out; none / prior arms
separately; ln R_0.5 slope (sign only, F87 rule 8's metric); the LM share of the gradient per prior cell (+ prior > LM evals,
clip coefficient); the 'without b11' gain per cell and its slope; lr swaps within the tie band (seeds 0-3); F99's six cells
and the four reused L40S cells alone (reproduced numbers, for reference).
    python readout_ptsz.py [--root RUNS] [--json OUT]
    python readout_ptsz.py --ka                  known answer: the reused cells reproduce F99 (FINDINGS / pt300x audit_results)
    python readout_ptsz.py --selftest [--work D] synthetic roots (AUDIT.md "Tests")
Definitions: tools/readout_rules.py of the frozen pttpp read-out snapshot (e91ee8a8a846), unchanged."""
import contextlib
import datetime
import io
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import select_ptsz as S   # noqa: E402  (the generator G, readout_rules R, run loading, integrity, F87 rule 1)

G, R, M = S.G, S.R, S.M
L40S8 = ("53m", "93m", "136m", "190m", "294m", "391m", "505m", "731m")
B200_2 = ("294m-b200", "1b")
NEW = G.REGISTERED                  # the new (ptsz) cells: 136m, 294m, 391m, 731m
NEW3 = G.NEW3                       # never seen: 136m, 391m, 731m
SIZE = {c: c.split("-")[0] for c in L40S8 + B200_2}
HW = {c: ("B200" if c in B200_2 else "L40S") for c in L40S8 + B200_2}
LABEL = {"53m": "53M", "93m": "93M", "136m": "136M", "190m": "190M", "294m": "294M", "391m": "391M", "505m": "505M",
         "731m": "731M", "294m-b200": "294M-B200", "1b": "1.07B"}
REUSED = {"53m": M.REUSED[("7", "53m", "21.2")], "93m": {"none": ("pt300x/out", "p3x-93m-lr2.4e-3-none", "2.4e-3"),
                                                          "win-ctx": ("pt300x/out", "p3x-93m-lr1.2e-3-win-ctx", "1.2e-3")},
          "190m": M.REUSED[("7", "190m", "2.65")], "505m": M.REUSED[("7", "505m", "0.745")],
          "294m-b200": {"none": ("pt300x/out", "p3x-294m-lr9.6e-3-none", "9.6e-3"),
                        "win-ctx": ("pt300x/out", "p3x-294m-lr6e-4-win-ctx", "6e-4")},
          "1b": M.REUSED[("7", "1b", "0.318")]}
assert {c: {a: v[2] for a, v in REUSED[c].items()} for c in REUSED} == {
    "53m": {"none": "8e-3", "win-ctx": "2e-3"}, "93m": {"none": "2.4e-3", "win-ctx": "1.2e-3"},
    "190m": {"none": "4.8e-3", "win-ctx": "1.2e-3"}, "505m": {"none": "9.6e-3", "win-ctx": "1.2e-3"},
    "294m-b200": {"none": "9.6e-3", "win-ctx": "6e-4"}, "1b": {"none": "2.4e-3", "win-ctx": "6e-4"}}
for (_s, _a), _lr in G._dec.items():   # the pt300x picks = its decisions.json (sha asserted by the generator)
    assert REUSED["93m" if _s == "93m" else "294m-b200"][_a][2] == _lr
CHECKED = {"93m", "294m-b200"}      # pt300x runs: their CHECK7 files must pass, area double path
ALL8, H1, H2 = tuple(range(8)), (0, 1, 2, 3), (4, 5, 6, 7)
N_BOOT, TIE = R.N_BOOT, R.TIE
EXPECTED = {"136m": {"none": (58, 68), "win-ctx": (88, 91)}, "294m": {"none": (55, 67), "win-ctx": (88, 91)},
            "391m": {"none": (53, 63), "win-ctx": (88, 91)}, "731m": {"none": (50, 60), "win-ctx": (88, 91)}}
MARGIN = 15.0
SHARE_FLAG = 0.6                    # LM share of the gradient below this -> flagged (manifold_aux_loss_principle)
LATE = 45.0                         # a none seed with area below this = a late learner (audit I1; 505M s5 34.2)
LATE_REL = 15.0                     # ... or below its cell's none median - 15 (size-relative; amendment audit m4)
RATIO = 0.5                         # amendment I-0: the 'few seeds' label when a robustness ratio to the mean slope is below this
SECONDARY_RATES = ("the secondary's pre-registered false-SHOWN rates under no growth (AUDIT.md; report the one of the verdict's "
                   "direction next to it in FINDINGS): increase 0.07 / 0.12 / 0.19 / 0.22, decrease 0.06 / 0.13 / 0.19 / 0.20 at "
                   "size scatter tau 0 / 1.85 / 3.12 / 3.41 (build simulation); increase 0.07-0.20 (fresh audit, the same model), "
                   "0.08-0.17 (with the lr selection simulated)")
READING = ("reading (AUDIT.md amendment, audit I3): primary SHOWN with secondary NOT SHOWN = only 'the new sizes did not fall "
           "below the old trend'; the primary alone still leans on the four points seen before registration")
# FINDINGS F99 (2026-11-04) table (1 decimal): the reused cells must still be the published ones
F99 = {"gain": {"53m": 22.8, "93m": 28.9, "190m": 23.5, "294m-b200": 29.1, "505m": 32.2, "1b": 36.1},
       "none": {"53m": 66.1, "93m": 60.9, "190m": 66.5, "294m-b200": 60.8, "505m": 57.5, "1b": 53.3},
       "prior": {"53m": 88.9, "93m": 89.8, "190m": 90.0, "294m-b200": 89.9, "505m": 89.7, "1b": 89.4}}
# known answers (pt300x tests/audit_results.md, independent recomputation, 3-4 decimals; FINDINGS F99 for the rest)
KA3 = {"gain": {"53m": 22.793, "93m": 28.874, "190m": 23.530, "294m-b200": 29.107, "505m": 32.188, "1b": 36.129},
       "none": {"53m": 66.092, "93m": 60.939, "190m": 66.508, "294m-b200": 60.754, "505m": 57.487, "1b": 53.260},
       "prior": {"53m": 88.885, "93m": 89.813, "190m": 90.038, "294m-b200": 89.861, "505m": 89.676, "1b": 89.389}}
KA_SLOPES = {"b6": (2.7871, 0.0006), "ci6": ((1.7098, 3.9214), 0.0006), "resid6": ((0.8455, 4.7287), 0.0006),
             "p6": (0.0377, 0.0006), "h1": (2.8064, 0.0006), "h2": (2.7677, 0.0006),
             "loso6": ({"53m": 2.9065, "93m": 3.2400, "190m": 2.7097, "294m-b200": 2.8337, "505m": 2.7453, "1b": 2.1544}, 0.0006),
             "none6": (-2.6947, 0.0006), "prior6": (0.0924, 0.0006), "l40s4": (2.1608, 0.0006),
             "l40s4_ci": ((0.5840, 4.0296), 0.0006), "l40s4_resid": ((-2.5355, 6.8571), 0.0006), "lnR6": (0.0301, 0.00006),
             "share6": ({"53m": 0.81, "93m": 0.86, "190m": 0.88, "294m-b200": 0.89, "505m": 0.91, "1b": 0.91}, 0.0051),
             "nob11_6": (2.52, 0.0051)}
F99_6 = ("53m", "93m", "190m", "294m-b200", "505m", "1b")
L40S4 = ("53m", "93m", "190m", "505m")


# ---------------------------------------------------------------- cells
def blank(cid):
    return dict(cid=cid, size=SIZE[cid], hw=HW[cid], problems=[], notfinal=[], pick={}, alts={"none": {}, "win-ctx": {}},
                runs={"none": [], "win-ctx": []})


def reused_cell(root, cid):
    c = blank(cid)
    size = SIZE[cid]
    for arm, (src, tag, lr) in REUSED[cid].items():
        c["pick"][arm] = lr
        areas, ts, rs, bud = [], [], [], None
        for s in ALL8:
            d = Path(root) / src / tag
            f = d / f"run_whole_{G.target_of(arm)}_seed{s}_{tag}.json"
            r = S.load_json(f)
            if r is None or not R.finished(r):
                c["problems"].append(f"{LABEL[cid]} {arm} seed {s}: {'missing' if r is None else 'unfinished'} ({f})")
                areas.append(np.nan); ts.append(np.nan); rs.append(None)
                continue
            c["problems"] += [f"{LABEL[cid]} {arm}@{lr} seed {s}: {w}" for w in S.integrity(r, size, arm, lr, s)]
            a = float(R.area(r))
            if cid in CHECKED:
                ck = S.load_json(d / f"check_s{s}.json")
                if ck is None or not ck.get("pass"):
                    c["problems"].append(f"{LABEL[cid]} {arm}@{lr} seed {s}: CHECK7 {'missing' if ck is None else 'FAILED'}")
                elif abs(a - ck.get("info", {}).get("area", math.inf)) > S.AREA_TOL:
                    c["problems"].append(f"{LABEL[cid]} {arm}@{lr} seed {s}: double path {a:.4f} vs {ck['info'].get('area')}")
            areas.append(a); ts.append(R.t_theta(r, 0.5)); rs.append(r); bud = R.budget_M(r)
        c["an" if arm == "none" else "ap"] = np.array(areas)
        c["tn" if arm == "none" else "tp"] = np.array(ts, float)
        c["runs"][arm] = rs
        c["bud"] = bud
        alt_dirs = []
        if src == "ptlr/out":   # lr alternatives on seeds 0-3 (F87 ptlr grid dirs; pt300x's rule); ptlr_1b: seeds 0-1 only
            pre = tag.split("-lr")[0]
            alt_dirs = [(d_, d_.name[len(pre) + 3:-len(f"-7pct-{arm}")]) for d_ in sorted((Path(root) / src).glob(f"{pre}-lr*-7pct-{arm}"))]
        elif src == "pt300x/out":
            pre = tag.split("-lr")[0]
            alt_dirs = [(d_, d_.name[len(pre) + 3:-len(f"-{arm}")]) for d_ in sorted((Path(root) / src).glob(f"{pre}-lr*-{arm}"))
                        if d_.name.endswith(f"-lr{d_.name[len(pre) + 3:-len(f'-{arm}')]}-{arm}")]
        for d_, alt in alt_dirs:
            if alt == lr:
                continue
            rr = [S.load_json(d_ / f"run_whole_{G.target_of(arm)}_seed{s}_{d_.name}.json") for s in H1]
            if all(r is not None and R.finished(r) and not S.integrity(r, size, arm, alt, s) for r, s in zip(rr, H1)):
                c["alts"][arm][alt] = np.array([float(R.area(r)) for r in rr])
    return c


def new_cell(root, size, state, own, out=print):
    c = blank(size)
    own_here = sorted(n for n in own if n.startswith(f"psz-{size}-lr"))
    if own_here:   # amendment I5 (honoured entries only: select_ptsz.own_failures, I-A): the whole size cell is NOT DETERMINABLE
        c["problems"].append(f"{LABEL[size]}: recurrent own failure(s) {own_here} (amendment I5): the cell is NOT DETERMINABLE")
        return c
    for arm in G.ARMS:
        st = state[(size, arm)]
        if st["STOP"] or not st["C"]:
            c["notfinal"].append(f"{LABEL[size]} {arm}: {'STOP (the user decides)' if st['STOP'] else 'no C decision yet'}")
            continue
        pick = c["pick"][arm] = st["C"]
        sel = S.select_arm(root, size, arm, st["X"], own, state[(size, "none")]["X"], out)
        if sel["status"] != "ok" or sel["best"] != pick:
            c["problems"].append(f"{LABEL[size]} {arm}: the recomputed selection is {sel['status']} / "
                                 f"{sel.get('best')} but the decided C is {pick}")
            continue
        recs = [sel["recs"][(pick, s)] for s in H1] + [S.new_run(root, size, arm, pick, s, own, out) for s in H2]
        for v in recs:
            if v["status"] != "ok":
                (c["notfinal"] if v["status"] in ("missing", "running", "dead", "nocheck") else c["problems"]).append(
                    f"{v['name']}: {v['status']}{(' ' + '; '.join(v['bad'])) if v['bad'] else ''}")
        if any(v["status"] != "ok" for v in recs):
            continue
        c["an" if arm == "none" else "ap"] = np.array([v["area"] for v in recs])
        c["tn" if arm == "none" else "tp"] = np.array([R.t_theta(v["r"], 0.5) if v["r"] else np.nan for v in recs], float)
        c["runs"][arm] = [v["r"] for v in recs]
        c["bud"] = 18310 * 32 * 512 / 1e6
        for lr in sel["grid"]:
            if lr != pick:
                c["alts"][arm][lr] = np.array([sel["recs"][(lr, s)]["area"] for s in H1])
        c["own"] = c.get("own", []) + [v["name"] for v in recs if v["own"]]
    return c


def finish(c):
    c["nne"] = G.SIZES[c["size"]][5]
    c["x"] = math.log(c["nne"])
    c["ok"] = "an" in c and "ap" in c and not c["problems"] and not c["notfinal"]
    if c["ok"]:
        c["d"] = c["ap"] - c["an"]
        own = any(r is None for a in G.ARMS for r in c["runs"][a])   # an own failure without a JSON: no b11 / grad
        if not own:
            c["d5"] = np.array([R.area(p, R.B5) - R.area(n, R.B5) for n, p in zip(c["runs"]["none"], c["runs"]["win-ctx"])])
    return c


# ---------------------------------------------------------------- statistics (pt300x's)
def ols(x, Y):
    """slope of Y (..., n) on x (n)"""
    x = np.asarray(x, float)
    xc = x - x.mean()
    Y = np.asarray(Y, float)
    return (Y - Y.mean(-1, keepdims=True)) @ xc / (xc @ xc)


def ols_cov(X, y):
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
        I = np.array([1.0 if c["hw"] == "B200" else 0.0 for c in cells])
        X = np.c_[np.ones_like(x), x, I]
        beta, se, df = ols_cov(X, y)
        b = float(beta[1])
        bb = np.linalg.lstsq(X, Yb.T, rcond=None)[0][1]
        extra = {"c_b200": float(beta[2]), "c_se": float(se[2])}
    lo, hi = (float(np.percentile(bb, 5)), float(np.percentile(bb, 95)))
    rlo, rhi, p = interval(b, se[1], df)
    return dict(b=b, lo=lo, hi=hi, rlo=float(rlo), rhi=float(rhi), p=float(p), df=int(df), n=len(cells),
                sizes=[c["cid"] for c in cells], means=[float(v) for v in y], **extra)


def sgn(v):
    return 1 if v > 0 else -1 if v < 0 else 0


def verdict(main, h1, h2, use_c=True):
    """the decision rule: (a)(b)(c) (primary) or (a)(b) (secondary, use_c False: the residual interval printed only)"""
    d = sgn(main["b"])
    a = d != 0 and ((d > 0 and main["lo"] > 0) or (d < 0 and main["hi"] < 0))
    b = d != 0 and sgn(h1["b"]) == d and sgn(h2["b"]) == d
    c = d != 0 and ((d > 0 and main["rlo"] > 0) or (d < 0 and main["rhi"] < 0))
    crit = {"a": a, "b": b, "c": c} if use_c else {"a": a, "b": b}
    nums = (f"b {main['b']:+.3f} per e-fold of N_nonemb; (a) seed bootstrap 90% CI [{main['lo']:+.3f}, {main['hi']:+.3f}] "
            f"{'excludes' if a else 'does not exclude'} 0; (b) seeds 0-3 {h1['b']:+.3f}, seeds 4-7 {h2['b']:+.3f}: "
            f"{'same sign' if b else 'NOT the same sign'}; " +
            (f"(c) residual t interval (df {main['df']}, 90%) [{main['rlo']:+.3f}, {main['rhi']:+.3f}] (p {main['p']:.3f}) "
             f"{'excludes' if c else 'does not exclude'} 0" if use_c else
             f"residual t interval (df {main['df']}, 90%; printed, not a criterion) [{main['rlo']:+.3f}, {main['rhi']:+.3f}] "
             f"(p {main['p']:.3f})"))
    if all(crit.values()):
        return f"SHOWN ({'increase' if d > 0 else 'decrease'}): {nums}", crit
    return f"NOT SHOWN (failed: {', '.join('(' + k + ')' for k, v in crit.items() if not v)}); {nums}", crit


def lnR_block(cells):
    rows, pts = [], []
    for c in sorted(cells, key=lambda c: c["x"]):
        sp = R.speedup(c["tn"], c["tp"], c["bud"])
        rows.append((c["cid"], sp))
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
                k = [o["cid"] for o in cells].index(c["cid"])
                key = "an" if arm == "none" else "ap"
                arr = cc[k][key].copy(); arr[:4] = v; cc[k][key] = arr
                cc[k]["d"] = cc[k]["ap"] - cc[k]["an"]
                b = slope_block(cc, seeds=H1)["b"]
                out.append((c["cid"], arm, c["pick"][arm], lr, float(v.mean() - mine.mean()), b, sgn(b) != sgn(base)))
    return out


def grad_stats(c):
    """the prior arm's on-window evals (pre-clip norms, the trainer's grad_info): per seed the median LM share a / (a + b);
    -> median over seeds, min over seeds, share of on-window evals with prior > LM, clip coefficient min(1, 1 / |g|)"""
    sh, gt, cl = [], [], []
    for r in c["runs"]["win-ctx"]:
        if r is None:
            continue
        on = [e for e in r["curve"] if e.get("prior_on") and e.get("grad_lm") is not None]
        if not on:
            continue
        a = np.array([e["grad_lm"] for e in on]); b = np.array([e["grad_prior"] for e in on]); cs = np.array([e["grad_cos"] for e in on])
        sh.append(float(np.median(a / (a + b)))); gt.append(float(np.mean(b > a)))
        cl.append(np.minimum(1.0, 1.0 / (np.sqrt(a ** 2 + b ** 2 + 2 * a * b * cs) + 1e-6)))
    if not sh:
        return None
    clc = np.concatenate(cl)
    return dict(med=float(np.median(sh)), min=float(min(sh)), gt=float(np.mean(gt)), clip_med=float(np.median(clc)),
                clip_min=float(clc.min()), clipped=float(np.mean(clc < 1)), n=len(sh))


def bridge(cl, cb):
    """L40S - B200 at 294M: 8-seed mean differences (none, prior, gain), seed bootstrap 90% CI (each cell resampled
    independently, default_rng(0))"""
    rng = np.random.default_rng(0)
    il, ib = rng.integers(0, 8, size=(N_BOOT, 8)), rng.integers(0, 8, size=(N_BOOT, 8))
    out = {}
    for k in ("an", "ap", "d"):
        dd = cl[k].mean() - cb[k].mean()
        bs = cl[k][il].mean(1) - cb[k][ib].mean(1)
        out[k] = (float(dd), float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))
    return out


def robust(cells):
    """amendment I1 (print only): slopes of the per-cell median gain and of the trimmed-mean gain (the min and the max seed
    dropped per cell), the leave-one-seed-out slopes (one seed of one cell left out: that cell's mean over the other seven),
    late none seeds per cell (area < 45, and size-relative: < the cell's none median - 15); amendment I-0: the ratios of the
    median / trimmed slopes and of the leave-one-seed-out extreme toward 0 to the mean slope, and the label"""
    cells = sorted(cells, key=lambda c: c["x"])
    x = [c["x"] for c in cells]
    mean = float(ols(x, [c["d"].mean() for c in cells]))
    med = float(ols(x, [float(np.median(c["d"])) for c in cells]))
    trim = float(ols(x, [float(np.sort(c["d"])[1:-1].mean()) for c in cells]))
    loso = {}
    for i, c in enumerate(cells):
        for s in range(len(c["d"])):
            y = [float(np.delete(o["d"], s).mean()) if j == i else float(o["d"].mean()) for j, o in enumerate(cells)]
            loso[(c["cid"], s)] = float(ols(x, y))
    kmin, kmax = min(loso, key=loso.get), max(loso, key=loso.get)
    kext = kmin if mean >= 0 else kmax   # the extreme toward 0 (amendment 3: its seed's arms are printed)
    byc = {c["cid"]: c for c in cells}
    seed_info = lambda k: dict(none=float(byc[k[0]]["an"][k[1]]), none_med=float(np.median(byc[k[0]]["an"])),
                               prior=float(byc[k[0]]["ap"][k[1]]), prior_med=float(np.median(byc[k[0]]["ap"])))
    # amendment 3 (R1): JSON-safe keys 'cid|sN' (json.dumps never applies default= to dict keys)
    rb = dict(mean=mean, median=med, trimmed=trim, loso={f"{k[0]}|s{k[1]}": v for k, v in loso.items()},
              loso_min=(loso[kmin], kmin), loso_max=(loso[kmax], kmax), loso_ext=(loso[kext], kext),
              loso_min_info=seed_info(kmin), loso_max_info=seed_info(kmax), loso_ext_info=seed_info(kext),
              late={c["cid"]: int((c["an"] < LATE).sum()) for c in cells},
              late_rel={c["cid"]: int((c["an"] < np.median(c["an"]) - LATE_REL).sum()) for c in cells}, n=len(cells))
    rb.update(few_seeds(rb))
    return rb


def few_seeds(rb):
    """amendment I-0 (print only, a label next to the verdicts): ratios to the mean slope of the median-gain slope, the
    trimmed-mean slope and the leave-one-seed-out extreme toward 0 (the minimum for a positive mean slope, the maximum for a
    negative one); 'the slope rests on a few seeds' when any ratio < 0.5; no label for a zero mean slope"""
    m = rb["mean"]
    if not (np.isfinite(m) and m != 0):
        return dict(ratios=None, label=False)
    ext = rb["loso_ext"][0]
    r = {"median": rb["median"] / m, "trimmed": rb["trimmed"] / m, "loso": ext / m}
    return dict(ratios=r, label=any(v < RATIO for v in r.values()))


def out_robust(out, rb, name):
    (lo, (c0, s0)), (hi, (c1, s1)), (ev, (ce, se_)) = rb["loso_min"], rb["loso_max"], rb["loso_ext"]
    ie = rb["loso_ext_info"]
    out(f"  robustness ({name}, {rb['n']} cells; amendments I1 / I-0, printed only, never a criterion): mean-gain slope {rb['mean']:+.3f}; "
        f"median-gain slope {rb['median']:+.3f}; trimmed-mean (min and max seed dropped) slope {rb['trimmed']:+.3f}; "
        f"leave-one-seed-out range [{lo:+.3f} (-{LABEL[c0]} s{s0}), {hi:+.3f} (-{LABEL[c1]} s{s1})]; the extreme toward 0: "
        f"{ev:+.3f} (-{LABEL[ce]} s{se_}: none {ie['none']:.1f} vs the cell's median {ie['none_med']:.1f}, prior {ie['prior']:.1f} "
        f"vs {ie['prior_med']:.1f})")
    out(f"    late none seeds per cell: absolute (area < {LATE:.0f}) " + ", ".join(f"{LABEL[k]} {v}" for k, v in rb["late"].items())
        + f"; size-relative (area < the cell's none median - {LATE_REL:.0f}) " + ", ".join(f"{LABEL[k]} {v}" for k, v in rb["late_rel"].items()))
    if rb["ratios"] is None:
        out("    ratios to the mean slope: not defined (mean slope 0); no label")
        return
    r = rb["ratios"]
    txt = (f"    ratios to the mean slope: median {r['median']:.2f}, trimmed {r['trimmed']:.2f}, leave-one-seed-out "
           f"{'minimum' if rb['mean'] > 0 else 'maximum'} {r['loso']:.2f} (label if any < {RATIO})")
    out(txt + (f" -> LABEL ({name}): the slope rests on a few seeds" if rb["label"] else " -> no label"))


# ---------------------------------------------------------------- the read-out
def earlier_results(root):
    """amendment 3: runs/ptsz/readout.json / readout.txt that already show a final new cell (a read-out of a ptsz result)"""
    seen = []
    j, txt = Path(root) / "ptsz" / "readout.json", Path(root) / "ptsz" / "readout.txt"
    if j.exists():
        try:
            if set(json.loads(j.read_text()).get("cells", {})) & set(NEW):
                seen.append(str(j))
        except Exception:   # noqa: BLE001  (unreadable: treat as a read-out that happened)
            seen.append(f"{j} (unparsable)")
    if txt.exists() and any(l.rstrip().endswith("  new") for l in txt.read_text(errors="replace").splitlines()):
        seen.append(str(txt))
    return seen


def load_cells(root, out=print):
    root = Path(root)
    log, state = G.decisions(root / "ptsz" / "decisions.json")
    S.load_states(root, out)
    own = S.own_failures(root, out)
    cells = {c: reused_cell(root, c) for c in REUSED}
    cells.update({s: new_cell(root, s, state, own, out) for s in NEW})
    return log, {k: finish(v) for k, v in cells.items()}


def readout(root, out=print, check_findings=True):
    root = Path(root)
    log, cells = load_cells(root, out)
    out(f"# ptsz read-out; root {root}; generator {S.sha(S.GEN)}; select {S.sha(HERE / 'select_ptsz.py')}; readout_rules "
        f"{S.sha(S.RULES_DIR / 'readout_rules.py')}; decisions {len(log)} entries")
    ownf = root / "ptsz" / "OWN_FAILURES.txt"
    frf = root / "ptsz" / "FIRST_READOUT.json"
    final = [s for s in NEW if cells[s]["ok"]]
    S.first_readout(root, strict=True)   # amendment 3: a FIRST_READOUT.json that cannot be parsed -> refuse (RuntimeError)
    if final and not frf.exists():   # amendment I-A: the first read-out that sees a final ptsz cell freezes OWN_FAILURES.txt
        seen = earlier_results(root)
        if seen:   # amendment 3: never silently re-freeze after a result was already read out
            raise RuntimeError(f"FIRST_READOUT.json is missing but {seen} already show a final ptsz cell: refusing (amendment 3; "
                               f"restore FIRST_READOUT.json from its recorded sha, or record the incident in AUDIT.md first)")
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        tmp = frf.with_name(frf.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"time": stamp, "final_cells": final, "own_entries": S.own_entries(root),
                                   "own_failures_sha": S.sha(ownf) if ownf.exists() else None,
                                   "readout_sha": S.sha(Path(__file__)), "by": "readout_ptsz.py"}, indent=1))
        os.replace(tmp, frf)   # atomic
        frf.chmod(0o444)
        out(f"# FIRST_READOUT.json written {stamp} (final cells {final}; sha256[:12] {S.sha(frf)}, read-only): OWN_FAILURES.txt "
            f"entries added after this are ignored")
    fr = S.first_readout(root)
    out(f"# OWN_FAILURES.txt {'sha256[:12] ' + S.sha(ownf) + ', ' + str(len(S.own_entries(root))) + ' entries' if ownf.exists() else 'absent'}; "
        f"honoured (amendments I5 / I-A / I-B): {sorted(S.own_failures(root, lambda *a: None)) or 'none'}; first read-out "
        f"{fr['time'] if fr else 'not yet'}")
    out("# primary: the eight L40S sizes (53M, 93M, 190M, 505M seen before registration; 136M, 294M-L40S, 391M, 731M new); "
        "secondary: the three never-seen sizes 136M, 391M, 731M only (the confirmatory test for F99 I1)")
    res = {"cells": {}, "problems": [], "notfinal": []}
    if check_findings:   # the reused cells must still be the published ones (FINDINGS F99, 1 decimal)
        for cid in REUSED:
            c = cells[cid]
            if c["ok"]:
                got = {"gain": c["d"].mean(), "none": c["an"].mean(), "prior": c["ap"].mean()}
                bad = {k: round(float(v), 2) for k, v in got.items() if abs(v - F99[k][cid]) > 0.05 + 1e-9}
                if bad:
                    c["problems"].append(f"{LABEL[cid]} differs from FINDINGS F99: {bad} vs {({k: F99[k][cid] for k in bad})}")
                    c["ok"] = False
    for cid in L40S8 + B200_2:
        res["problems"] += cells[cid]["problems"]
        res["notfinal"] += cells[cid]["notfinal"]
    out("\n## integrity and completeness")
    for p in res["problems"]:
        out(f"  PROBLEM {p}")
    for p in res["notfinal"]:
        out(f"  NOT FINAL {p}")
    if not res["problems"] and not res["notfinal"]:
        out("  every counted run present, finished, intended (integrity re-check; ptsz and pt300x runs: CHECK7 passed, double path)")
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
    ok = {cid: cells[cid] for cid in L40S8 + B200_2 if cells[cid]["ok"]}
    out("\n## cells at the picks (8 seeds; area points; gain = prior - none, paired; R_0.5 = median t50(none) / median t50(prior), censored at the budget)")
    out(f"  {'cell':9s} {'hw':5s} {'N_nonemb':>12s} {'ln N':>6s} {'none lr':>8s} {'none':>6s} {'prior lr':>8s} {'prior':>6s} "
        f"{'gain':>6s} {'[seed CI]':>15s} {'s0-3':>6s} {'s4-7':>6s} {'wins':>5s} {'R_0.5':>8s}  status")
    for c in sorted(ok.values(), key=lambda c: (c["x"], c["hw"])):
        d = c["d"]
        bd = d[boot_idx(1, 8)[0]].mean(1)
        sp = R.speedup(c["tn"], c["tp"], c["bud"])
        rtxt = "none" if sp["ratio"] is None else f"{sp['mark']}{sp['ratio']:.2f}"
        st = "new" if c["cid"] in NEW else "reused" + (" (descriptive only)" if c["hw"] == "B200" else "")
        out(f"  {LABEL[c['cid']]:9s} {c['hw']:5s} {c['nne']:12,d} {c['x']:6.3f} {c['pick']['none']:>8s} {c['an'].mean():6.1f} "
            f"{c['pick']['win-ctx']:>8s} {c['ap'].mean():6.1f} {d.mean():+6.1f} [{np.percentile(bd, 5):+6.1f}, {np.percentile(bd, 95):+6.1f}] "
            f"{d[:4].mean():+6.1f} {d[4:].mean():+6.1f} {int((d > 0).sum()):>3d}/8 {rtxt:>8s}  {st}")
        res["cells"][c["cid"]] = dict(hw=c["hw"], nne=c["nne"], pick=c["pick"], none=c["an"].tolist(), prior=c["ap"].tolist(),
                                      gain=float(d.mean()), r50=sp["ratio"], r50_mark=sp["mark"])
    inv = f"  [INVESTIGATE BEFORE READING: outside the expected range: {', '.join(flags)}]" if flags else ""
    res["investigate"] = flags
    # ---- primary
    p8 = [cells[c] for c in L40S8]
    missing = [LABEL[c["cid"]] for c in p8 if not c["ok"]]
    if missing:
        v = f"NOT DETERMINABLE ({len(missing)} of 8 L40S cells not available: {', '.join(missing)}; " + \
            "; ".join((res["problems"] + res["notfinal"])[:3]) + ")"
        out(f"\n## PRIMARY (pre-registered): eight L40S sizes\n  VERDICT: {v}")
        res["verdict"] = v
    else:
        main = slope_block(p8)
        h1, h2 = slope_block(p8, seeds=H1), slope_block(p8, seeds=H2)
        v, crit = verdict(main, h1, h2)
        v += inv
        out("\n## PRIMARY (pre-registered): OLS slope of the eight L40S 8-seed mean gains on ln N_nonemb")
        out(f"  b = {main['b']:+.3f} per e-fold; seed bootstrap 90% CI [{main['lo']:+.3f}, {main['hi']:+.3f}]; residual t (df "
            f"{main['df']}) 90% [{main['rlo']:+.3f}, {main['rhi']:+.3f}], p {main['p']:.3f}")
        out(f"  halves: seeds 0-3 {h1['b']:+.3f}, seeds 4-7 {h2['b']:+.3f}")
        out(f"  VERDICT: {v}")
        rb = robust(p8)
        out_robust(out, rb, "primary")
        res.update(primary=main, h1=h1, h2=h2, criteria=crit, verdict=v, robust=rb)
    # ---- secondary
    s3 = [cells[c] for c in NEW3]
    missing3 = [LABEL[c["cid"]] for c in s3 if not c["ok"]]
    if missing3:
        v2 = f"NOT DETERMINABLE ({len(missing3)} of the 3 new-size cells not available: {', '.join(missing3)})"
        out(f"\n## SECONDARY (pre-registered; the confirmatory test for F99 I1): the three never-seen sizes\n  VERDICT (secondary): {v2}")
        res["verdict2"] = v2
    else:
        m3 = slope_block(s3)
        g1, g2 = slope_block(s3, seeds=H1), slope_block(s3, seeds=H2)
        v2, crit2 = verdict(m3, g1, g2, use_c=False)
        v2 += inv
        out("\n## SECONDARY (pre-registered; the confirmatory test for F99 I1): slope over 136M, 391M, 731M only (never seen)")
        out(f"  b = {m3['b']:+.3f} per e-fold; seed bootstrap 90% CI [{m3['lo']:+.3f}, {m3['hi']:+.3f}]; residual t (df "
            f"{m3['df']}) 90% [{m3['rlo']:+.3f}, {m3['rhi']:+.3f}], p {m3['p']:.3f} (printed, not a criterion)")
        out(f"  halves: seeds 0-3 {g1['b']:+.3f}, seeds 4-7 {g2['b']:+.3f}")
        out(f"  VERDICT (secondary, new sizes only): {v2}")
        out(f"  {SECONDARY_RATES}")
        rb2 = robust(s3)
        out_robust(out, rb2, "secondary")
        res.update(secondary=m3, s_h1=g1, s_h2=g2, criteria2=crit2, verdict2=v2, robust2=rb2)
    if "verdict2" in res and "criteria" in res:
        out(f"  {READING}")
    if missing:
        return res
    # ---- descriptive
    out("\n## descriptive (no verdict)")
    ten = [cells[c] for c in L40S8 + B200_2]
    if all(c["ok"] for c in ten):
        t10, t10b = slope_block(ten), slope_block(ten, design="b200")
        out(f"  10-point all-hardware slope (+ F99's B200 294M and 1.07B): {t10['b']:+.3f} [{t10['lo']:+.3f}, {t10['hi']:+.3f}], "
            f"residual t (df {t10['df']}) [{t10['rlo']:+.3f}, {t10['rhi']:+.3f}]; with a B200 indicator: b {t10b['b']:+.3f} "
            f"[{t10b['lo']:+.3f}, {t10b['hi']:+.3f}], residual t (df {t10b['df']}) [{t10b['rlo']:+.3f}, {t10b['rhi']:+.3f}]; "
            f"c {t10b['c_b200']:+.2f} (se {t10b['c_se']:.2f})")
        br = bridge(cells["294m"], cells["294m-b200"])
        out(f"  hardware bridge at 294M (L40S - B200; picks L40S {cells['294m']['pick']['none']} / {cells['294m']['pick']['win-ctx']}, "
            f"B200 {cells['294m-b200']['pick']['none']} / {cells['294m-b200']['pick']['win-ctx']}): "
            + "; ".join(f"{k} {v_[0]:+.2f} [{v_[1]:+.2f}, {v_[2]:+.2f}]" for k, v_ in zip(("none", "prior", "gain"), br.values()))
            + " (pt300x's 505M bridge, B200 - L40S: none -4.0, gain +4.3)")
        rb10 = robust(ten)
        out_robust(out, rb10, "10-point")
        res.update(ten=t10, ten_b200=t10b, bridge294=br, robust10=rb10)
    else:
        out("  10-point all-hardware slope / hardware bridge at 294M: not available (a B200 cell is not available)")
    loso = {c["cid"]: slope_block([o for o in p8 if o is not c])["b"] for c in p8}
    out("  leave one size out (primary): " + ", ".join(f"-{LABEL[s]} {b:+.3f}" for s, b in loso.items())
        + f" ({sum(sgn(b) != sgn(res['primary']['b']) for b in loso.values())} sign changes)")
    arms = {k: slope_block(p8, metric=m) for k, m in (("none", "an"), ("prior", "ap"))}
    out("  arms separately (mean area on ln N, primary cells): " + "; ".join(
        f"{k} {v_['b']:+.3f} [{v_['lo']:+.3f}, {v_['hi']:+.3f}]" for k, v_ in arms.items()))
    rows, bR, nR = lnR_block(p8)
    out("  ln R_0.5 (sign only; F87 rule 8's metric): R_0.5 " + ", ".join(
        f"{LABEL[s]} {'none' if sp['ratio'] is None else sp['mark'] + format(sp['ratio'], '.2f')}" for s, sp in rows)
        + (f"; slope over the {nR} point cells {bR:+.4f} ({'same sign as' if sgn(bR) == sgn(res['primary']['b']) else 'OTHER sign than'} the primary)"
           if bR is not None else "; slope not determinable (fewer than 2 point cells)"))
    out(f"  LM share of the gradient (prior arm, on-window evals, pre-clip norms; flag < {SHARE_FLAG} or prior > LM in most evals):")
    gs = {}
    for c in sorted(ok.values(), key=lambda c: (c["x"], c["hw"])):
        g = gs[c["cid"]] = grad_stats(c)
        if g is None:
            out(f"    {LABEL[c['cid']]:9s} no grad_info")
            continue
        fl = g["med"] < SHARE_FLAG or g["gt"] > 0.5
        out(f"    {LABEL[c['cid']]:9s} LM share median {g['med']:.2f} [min seed {g['min']:.2f}]; prior > LM in {100 * g['gt']:.0f}% of "
            f"evals; clip median {g['clip_med']:.2f} (min {g['clip_min']:.2f}, clipped {100 * g['clipped']:.0f}%){'  FLAG' if fl else ''}")
    nb = [c for c in p8 if "d5" in c]
    if len(nb) == len(p8):
        bnb = float(ols([c["x"] for c in p8], [c["d5"].mean() for c in p8]))
        b3nb = float(ols([c["x"] for c in s3], [c["d5"].mean() for c in s3]))
        out("  without b11 (readout_rules.B5) gain: " + ", ".join(f"{LABEL[c['cid']]} {c['d5'].mean():+.1f}" for c in p8)
            + f"; slope primary {bnb:+.3f}, secondary {b3nb:+.3f}")
        res["nob11"] = dict(primary=bnb, secondary=b3nb)
    else:
        out("  without b11: not computed (an own failure without a run JSON)")
    sw = swaps(p8, res["h1"]["b"])
    out(f"  lr swaps within the tie band (seeds 0-3; primary slope at the picks {res['h1']['b']:+.3f}):" + ("" if sw else " none in the band"))
    for s, arm, p, lr, dm, b, flip in sw:
        out(f"    {LABEL[s]} {arm} {p} -> {lr} (mean {dm:+.2f}): {b:+.3f}{'  SIGN CHANGE' if flip else ''}")
    out(f"    {sum(x[-1] for x in sw)} of {len(sw)} swaps change the sign")
    if all(cells[c]["ok"] for c in F99_6):
        f6 = slope_block([cells[c] for c in F99_6])
        f4 = slope_block([cells[c] for c in L40S4])
        out(f"  reference: F99's six cells {f6['b']:+.3f} [{f6['lo']:+.3f}, {f6['hi']:+.3f}] (F99 +2.787); the four reused L40S "
            f"cells alone {f4['b']:+.3f} [{f4['lo']:+.3f}, {f4['hi']:+.3f}], residual t (df {f4['df']}) [{f4['rlo']:+.3f}, {f4['rhi']:+.3f}]")
        res.update(f99_6=f6, l40s4=f4)
    res.update(loso=loso, arms=arms, lnR50=dict(b=bR, n=nR), grad={k: v for k, v in gs.items()}, swaps=sw)
    return res


# ---------------------------------------------------------------- known answer: F99 on the real reused cells
def ka(root):
    cells = {c: finish(reused_cell(root, c)) for c in REUSED}
    probs = [p for c in cells.values() for p in c["problems"]]
    ok, n = 0, 0

    def T(name, got, want, tol):
        nonlocal ok, n
        n += 1
        g = np.atleast_1d(np.asarray(got, float)); w = np.atleast_1d(np.asarray(want, float))
        good = bool(np.all(np.abs(g - w) <= tol))
        ok += good
        print(f"{'PASS' if good else 'FAIL'} {name}: got {np.round(g, 4).tolist()} want {w.tolist()} (tol {tol})")

    n += 1
    ok += not probs
    print(f"{'PASS' if not probs else 'FAIL'} the 96 reused runs (6 cells x 2 arms x 8 seeds) pass the integrity re-check "
          f"(pt300x runs also CHECK7 + double path)" + (f": {probs[:3]}" if probs else ""))
    if probs:
        print(f"KA FAIL {ok}/{n}")
        return False
    six = [cells[c] for c in F99_6]
    for k, key in (("gain", "d"), ("none", "an"), ("prior", "ap")):
        T(f"8-seed {k} per cell (53M 93M 190M 294M-B200 505M 1.07B) = pt300x audit_results", [c[key].mean() for c in six],
          [KA3[k][c] for c in F99_6], 0.0006)
    m = slope_block(six)
    T("F99 primary slope (six cells)", m["b"], *KA_SLOPES["b6"])
    T("F99 seed bootstrap 90% CI", [m["lo"], m["hi"]], *KA_SLOPES["ci6"])
    T("F99 residual t interval df 4", [m["rlo"], m["rhi"]], *KA_SLOPES["resid6"])
    T("F99 residual p", m["p"], *KA_SLOPES["p6"])
    T("F99 seeds 0-3 slope", slope_block(six, seeds=H1)["b"], *KA_SLOPES["h1"])
    T("F99 seeds 4-7 slope", slope_block(six, seeds=H2)["b"], *KA_SLOPES["h2"])
    lw, tol = KA_SLOPES["loso6"]
    T("F99 leave one size out", [slope_block([c for c in six if c["cid"] != s])["b"] for s in lw], list(lw.values()), tol)
    T("F99 none arm slope", slope_block(six, metric="an")["b"], *KA_SLOPES["none6"])
    T("F99 prior arm slope", slope_block(six, metric="ap")["b"], *KA_SLOPES["prior6"])
    l4 = slope_block([cells[c] for c in L40S4])
    T("F99 L40S-only slope (53M, 93M, 190M, 505M: the four reused L40S cells of ptsz)", l4["b"], *KA_SLOPES["l40s4"])
    T("  its bootstrap CI (fresh draw)", [l4["lo"], l4["hi"]], *KA_SLOPES["l40s4_ci"])
    T("  its residual t interval df 2", [l4["rlo"], l4["rhi"]], *KA_SLOPES["l40s4_resid"])
    T("F99 ln R_0.5 slope", lnR_block(six)[1], *KA_SLOPES["lnR6"])
    sh, tol = KA_SLOPES["share6"]
    T("F99 LM share medians (FINDINGS, 2 decimals)", [grad_stats(cells[c])["med"] for c in sh], list(sh.values()), tol)
    T("F99 without-b11 slope (FINDINGS +2.52)", float(ols([c["x"] for c in six], [c["d5"].mean() for c in six])), *KA_SLOPES["nob11_6"])
    rb4 = robust([cells[c] for c in L40S4])
    T("I1 robustness, the four reused L40S cells: median-gain slope (coordinator +1.006)", rb4["median"], 1.006, 0.0006)
    T("  trimmed-mean slope (min and max seed dropped; coordinator +1.167)", rb4["trimmed"], 1.167, 0.0006)
    T("  leave out 505M seed 5 (coordinator +1.213)", rb4["loso"]["505m|s5"], 1.213, 0.0006)
    T("  mean slope +2.1608 = the L40S-only slope; 505M seed 5 is the leave-one-seed-out minimum", [rb4["mean"], float(rb4["loso_min"][1] == ("505m", 5))],
      [2.1608, 1.0], 0.0006)
    T("I1 robustness, F99's six cells: median-gain slope (fresh build audit +1.82)", robust(six)["median"], 1.82, 0.0051)
    r4 = rb4["ratios"]
    T("I-0: the four reused L40S cells: ratios median / mean 0.47, trimmed / mean 0.54, leave-one-seed-out minimum / mean 0.56 "
      "(the amendment audit's example) -> the 'few seeds' label (1 = yes)", [r4["median"], r4["trimmed"], r4["loso"], float(rb4["label"])],
      [0.4655, 0.5402, 0.5612, 1.0], 0.0006)
    r6 = robust(six)
    T("I-0: F99's six cells: ratios 0.65 / 0.88 / 0.88 -> no label (0 = no)", [r6["ratios"]["median"], r6["ratios"]["trimmed"],
      r6["ratios"]["loso"], float(r6["label"])], [1.8228 / 2.7871, 2.4413 / 2.7871, 2.4656 / 2.7871, 0.0], 0.0006)
    T("m4: late none seeds (absolute < 45 / size-relative < median - 15): 505M 1 / 1; F99's 1.07B 2 absolute (43.9, 44.0)",
      [rb4["late"]["505m"], rb4["late_rel"]["505m"], r6["late"]["1b"]], [1, 1, 2], 0)
    print(f"KA {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    return ok == n


# ---------------------------------------------------------------- selftest (synthetic roots)
REUSED_SRC = {cid: REUSED[cid] for cid in REUSED}


def fake_reused(root, cid, arm, seed, area, **kw):
    src, tag, lr = REUSED[cid][arm]
    size = SIZE[cid]
    F = G.facts("136m", arm, G.GRID["136m"][arm][0], 0, "A")
    a = dict(G.intent_x(F)["args"])
    d, nl, nh, dff, n_params, _ = G.SIZES[size]
    a.update(out=str(Path(root) / src / tag), tag=f"_{tag}", seed=seed, lr=float(lr), d_model=d, n_layers=nl, n_heads=nh,
             d_ff=dff, save_model=True, ckpt_every=500 if cid in B200_2 else 2000)
    r = {"args": a, "steps": 18310, "geo_layers": [nl // 3, 2 * nl // 3], "read_layers": [nl // 3, 2 * nl // 3],
         "n_params": n_params, "grad_ckpt_on": n_params > 150e6, "lm_chunk": 0, "eval_every_steps": 250,
         "code_sha": M.PIN7_SHA, "prior_events": G.WINDOW_EVENTS if arm != "none" else [],
         "curve": S.fake_curve(area, G.lm0_ref(size, seed), arm, **kw)}
    dd = Path(root) / src / tag
    f = dd / f"run_whole_{G.target_of(arm)}_seed{seed}_{tag}.json"
    dd.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(r))
    if cid in CHECKED:
        (dd / f"check_s{seed}.json").write_text(json.dumps({"pass": True, "checks": [], "info": {"area": round(area, 2)}}))
    return f


PICK = {"136m": {"none": "4.8e-3", "win-ctx": "1.2e-3"}, "294m": {"none": "9.6e-3", "win-ctx": "6e-4"},
        "391m": {"none": "4.8e-3", "win-ctx": "6e-4"}, "731m": {"none": "2.4e-3", "win-ctx": "6e-4"}}


def build_root(base, name, gain, prior=92.0, q=None, decided=None, tweak=None, alt_tie=False, skip=()):
    """gain(cell, seed) -> designed gain; prior area = prior + q[seed]; none = prior area - gain. New cells: the pick has
    the designed areas, every other grid lr 10 points lower (alt_tie: 136m win-ctx 2.4e-3 one point lower, higher bpb)."""
    root = Path(base) / name
    (root / "ptsz").mkdir(parents=True)
    q = q or [0.5, -0.5, 1.0, -1.0, 0.5, -0.5, 1.0, -1.0]
    pr = (lambda c: prior.get(c, 92.0)) if isinstance(prior, dict) else (lambda c: prior)
    val = lambda c, arm, seed: (pr(c) + q[seed]) - (gain(c, seed) if arm == "none" else 0.0)
    for c in REUSED:
        if c in skip:
            continue
        for arm in G.ARMS:
            for seed in ALL8:
                fake_reused(root, c, arm, seed, val(c, arm, seed))
    log = []
    for s in NEW:
        for arm in G.ARMS:
            pk = PICK[s][arm]
            for lr in G.GRID[s][arm]:
                for seed in H1:
                    v = val(s, arm, seed) - (0 if lr == pk else (1 if alt_tie and (s, arm, lr) == ("136m", "win-ctx", "2.4e-3") else 10))
                    S.fake_new_run(root, s, arm, lr, seed, v, bpb=1.19 if lr == pk else 1.2, check="synthetic")
            for seed in H2:
                S.fake_new_run(root, s, arm, pk, seed, val(s, arm, seed), check="synthetic")
            if decided is None or (s, arm) in decided:
                log.append(dict(seq=len(log), size=s, arm=arm, stage="C", lr=pk))
    (root / "ptsz" / "decisions.json").write_text(json.dumps({"log": log}))
    if tweak:
        tweak(root)
    return root


def selftest(work):
    base = Path(tempfile.mkdtemp(prefix="ptsz_readout_selftest_", dir=work))
    ok, n = 0, 0

    def T(name, cond, info=""):
        nonlocal ok, n
        n += 1
        ok += bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info and not cond else ""))

    xs = {c: math.log(G.SIZES[SIZE[c]][5]) for c in L40S8 + B200_2}
    m = float(np.mean([xs[c] for c in L40S8]))
    lines = []
    quiet = lambda t="": lines.append(t)
    real_sacct = S._sacct
    S._sacct = lambda: (0, "", "")   # no real sacct in the selftest (the rows a test needs are set by that test)
    sym = [1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 2.0, -2.0]   # zero mean within each half
    lin = lambda b: (lambda c, k: 30 + b * (xs[c] - m) + sym[k])
    try:
        r = readout(build_root(base, "pos", lin(3.0), alt_tie=True), out=quiet, check_findings=False)
        T("positive slope 3: primary SHOWN (increase)", r["verdict"].startswith("SHOWN (increase)"), r["verdict"])
        T("  primary +3.000 on 8 seeds and both halves; residual interval degenerate at 3",
          abs(r["primary"]["b"] - 3) < 1e-9 and abs(r["h1"]["b"] - 3) < 1e-9 and abs(r["h2"]["b"] - 3) < 1e-9
          and abs(r["primary"]["rlo"] - 3) < 1e-6 and r["primary"]["df"] == 6 and r["primary"]["n"] == 8, r.get("primary"))
        T("  secondary SHOWN (increase), +3.000, df 1, over exactly 136m / 391m / 731m",
          r["verdict2"].startswith("SHOWN (increase)") and abs(r["secondary"]["b"] - 3) < 1e-9 and r["secondary"]["df"] == 1
          and r["secondary"]["sizes"] == ["136m", "391m", "731m"] and set(r["criteria2"]) == {"a", "b"}, r.get("secondary"))
        T("  primary cells = the eight L40S cells in ascending N (no B200 cell)",
          r["primary"]["sizes"] == list(L40S8), r["primary"]["sizes"])
        T("  arms: none -3.000, prior 0.000; 10-point 3.000 (indicator c 0.00); bridge 0.00 in every arm",
          abs(r["arms"]["none"]["b"] + 3) < 1e-9 and abs(r["arms"]["prior"]["b"]) < 1e-9 and abs(r["ten"]["b"] - 3) < 1e-9
          and abs(r["ten_b200"]["c_b200"]) < 1e-9 and all(abs(v[0]) < 1e-9 for v in r["bridge294"].values()) and r["ten"]["n"] == 10)
        T("  leave-one-out all +3.000; without-b11 slope +3.000 (all buckets equal); ln R_0.5 slope positive, 8 point cells",
          all(abs(b - 3) < 1e-9 for b in r["loso"].values()) and len(r["loso"]) == 8 and abs(r["nob11"]["primary"] - 3) < 1e-9
          and r["lnR50"]["b"] > 0 and r["lnR50"]["n"] == 8, (r["loso"], r.get("nob11"), r["lnR50"]))
        g = r["grad"]["731m"]
        T("  LM share = 1 / 1.2 = 0.833 in every prior cell (designed grad_lm 1, grad_prior 0.2), prior > LM 0%, clip 0.962",
          all(abs(v["med"] - 1 / 1.2) < 1e-9 for v in r["grad"].values()) and len(r["grad"]) == 10 and g["gt"] == 0
          and abs(g["clip_med"] - 1 / (math.sqrt(1.08) + 1e-6)) < 1e-9, g)
        T("  the tied 136m win-ctx alternative (2.4e-3, -1 point) is swapped, no sign change; no other swap",
          len(r["swaps"]) == 1 and r["swaps"][0][:4] == ("136m", "win-ctx", "1.2e-3", "2.4e-3") and not r["swaps"][0][-1], r["swaps"])
        T("  no INVESTIGATE flag, no problem", not r["investigate"] and not r["problems"] and not r["notfinal"])
        T("  I1 robustness: median and trimmed-mean slopes +3.000 (primary, secondary, 10-point); no late none seed",
          all(abs(r[k]["median"] - 3) < 1e-9 and abs(r[k]["trimmed"] - 3) < 1e-9 for k in ("robust", "robust2", "robust10"))
          and len(r["robust"]["loso"]) == 64 and len(r["robust10"]["loso"]) == 80 and not any(r["robust"]["late"].values()),
          (r["robust"]["median"], r["robust"]["trimmed"], r["robust"]["late"]))
        T("  I-0: no late seed -> all ratios 1.00 (leave-one-seed-out minimum 0.97), no 'few seeds' label printed",
          not any(r[k]["label"] for k in ("robust", "robust2", "robust10")) and abs(r["robust"]["ratios"]["median"] - 1) < 1e-9
          and not any("LABEL" in l for l in lines) and any("-> no label" in l for l in lines), r["robust"]["ratios"])
        flat = lambda c, k: 30 + sym[k]
        plant = lambda c, k: 30 + sym[k] + (25.0 if (c, k) == ("731m", 6) else 0.0)   # 731m none seed 6 25 points lower
        r0 = readout(build_root(base, "flatref", flat), out=quiet, check_findings=False)
        lines.clear()
        r1 = readout(build_root(base, "late", plant), out=quiet, check_findings=False)
        T("I-0: the planted late seed -> median / mean 0.00 -> 'the slope rests on a few seeds' printed (primary, secondary, 10-point)",
          all(r1[k]["label"] for k in ("robust", "robust2", "robust10"))
          and sum("LABEL" in l and "the slope rests on a few seeds" in l for l in lines) == 3, [l for l in lines if "ratios" in l][:3])
        S.load_states(None, lambda *a: None)
        real4 = [finish(reused_cell(S.ROOT_DEFAULT, c)) for c in L40S4]   # the amendment audit's example (real runs, read-only)
        lines.clear()
        rb4 = robust(real4)
        out_robust(quiet, rb4, "the four reused L40S cells")
        T("I-0: the audit's example, the four reused L40S cells (real runs): median / mean 0.47, leave-505M-s5 0.56 -> the label is printed",
          rb4["label"] and any("median 0.47" in l and "minimum 0.56" in l and "the slope rests on a few seeds" in l for l in lines)
          and any("-505M s5: none 34.2" in l for l in lines), lines)
        T("I1: a planted late-learning none seed (731m seed 6, none area 25 points lower): the mean slope moves "
          "(primary 0 -> +0.44, secondary 0 -> +1.26), the median and trimmed-mean slopes do not (0.000)",
          abs(r0["primary"]["b"]) < 1e-9 and r1["primary"]["b"] > 0.3 and r1["secondary"]["b"] > 1.0
          and all(abs(r1[k][m]) < 1e-9 and abs(r0[k][m]) < 1e-9 for k in ("robust", "robust2", "robust10") for m in ("median", "trimmed")),
          (r1["primary"]["b"], r1["secondary"]["b"], r1["robust"]["median"], r1["robust"]["trimmed"]))
        T("  leave-one-seed-out: the lowest slope leaves out 731m seed 6 (primary, secondary, 10-point); late none seeds: 731m 1, others 0",
          all(r1[k]["loso_min"][1] == ("731m", 6) for k in ("robust", "robust2", "robust10"))
          and r1["robust"]["late"] == {c: int(c == "731m") for c in L40S8}, (r1["robust"]["loso_min"], r1["robust"]["late"]))
        r = readout(build_root(base, "zero", lambda c, k: 30 + [3.0, -1.0, 2.0, -4.0, 1.0, -3.0, 4.0, -2.0][k]), out=quiet, check_findings=False)
        T("zero slope: primary and secondary NOT SHOWN, (a) failed", r["verdict"].startswith("NOT SHOWN") and not r["criteria"]["a"]
          and r["verdict2"].startswith("NOT SHOWN") and not r["criteria2"]["a"] and abs(r["primary"]["b"]) < 1e-9, (r["verdict"], r["verdict2"]))
        r = readout(build_root(base, "neg", lin(-3.0)), out=quiet, check_findings=False)
        T("negative slope: SHOWN (decrease) in both, -3.000", r["verdict"].startswith("SHOWN (decrease)")
          and r["verdict2"].startswith("SHOWN (decrease)") and abs(r["primary"]["b"] + 3) < 1e-9, r["verdict"])
        halves = lambda c, k: 30 + (6.0 if k < 4 else -1.5) * (xs[c] - m) + [1, -1, 1, -1][k % 4]
        r = readout(build_root(base, "halves", halves), out=quiet, check_findings=False)
        T("halves disagree (seeds 0-3 +6, seeds 4-7 -1.5): NOT SHOWN (b) in both, primary 8-seed +2.25",
          r["verdict"].startswith("NOT SHOWN") and not r["criteria"]["b"] and abs(r["primary"]["b"] - 2.25) < 1e-9
          and r["verdict2"].startswith("NOT SHOWN") and not r["criteria2"]["b"], (r["verdict"], r["verdict2"]))
        X8 = np.c_[np.ones(8), [xs[c] for c in L40S8]]
        e = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        e = e - X8 @ np.linalg.lstsq(X8, e, rcond=None)[0]
        e = 8 * e / np.abs(e).max()
        resid = dict(zip(L40S8, e))
        tiny = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        r = readout(build_root(base, "resid", lambda c, k: 30 + 1.0 * (xs[c] - m) + resid.get(c, 0.0) + tiny[k]), out=quiet, check_findings=False)
        T("primary residual interval includes 0 (b 1.0, size scatter +-8, seed noise +-0.1): NOT SHOWN (c), (a) and (b) passed",
          r["verdict"].startswith("NOT SHOWN") and r["criteria"]["a"] and r["criteria"]["b"] and not r["criteria"]["c"]
          and abs(r["primary"]["b"] - 1) < 1e-9 and r["primary"]["rlo"] < 0 < r["primary"]["rhi"], r["verdict"])
        T("  the secondary has no criterion (c): its df-1 interval includes 0 and it is still decided by (a) and (b)",
          set(r["criteria2"]) == {"a", "b"} and r["secondary"]["rlo"] < 0 < r["secondary"]["rhi"]
          and r["verdict2"].startswith("SHOWN" if r["criteria2"]["a"] and r["criteria2"]["b"] else "NOT SHOWN"), (r["verdict2"], r["secondary"]))
        flat_new = lambda c, k: 30 + (0.0 if c in NEW3 else 3.0 * (xs[c] - m)) + sym[k]
        r = readout(build_root(base, "flatnew", flat_new), out=quiet, check_findings=False)
        T("old sizes slope 3, the three new sizes flat: secondary NOT SHOWN (a) with b 0.000 whatever the primary says",
          r["verdict2"].startswith("NOT SHOWN") and not r["criteria2"]["a"] and abs(r["secondary"]["b"]) < 1e-9, r["verdict2"])
        r = readout(build_root(base, "b200", lambda c, k: 30 + 2 * (xs[c] - m) + (6.0 if HW[c] == "B200" else 0.0) + sym[k]),
                    out=quiet, check_findings=False)
        T("B200 offset +6 on 294M-B200 / 1.07B: primary 2.000 (B200-free), 10-point indicator b 2.000 c 6.000, bridge gain -6.00",
          abs(r["primary"]["b"] - 2) < 1e-9 and abs(r["ten_b200"]["b"] - 2) < 1e-9 and abs(r["ten_b200"]["c_b200"] - 6) < 1e-9
          and abs(r["bridge294"]["d"][0] + 6) < 1e-9 and abs(r["ten"]["b"] - 2) > 0.05, (r["primary"]["b"], r.get("ten_b200"), r.get("bridge294")))
        r = readout(build_root(base, "nob200", lin(3.0), skip=("1b",)), out=quiet, check_findings=False)
        T("a B200 cell missing: the primary is still determined (SHOWN), the 10-point slope is 'not available'",
          r["verdict"].startswith("SHOWN") and "ten" not in r and any("10-point all-hardware slope / hardware bridge at 294M: not available" in l for l in lines), r["verdict"])

        def bad_sha(root):
            f = next((root / "ptlr" / "out" / "190m-lr4.8e-3-7pct-none").glob("run_*seed5_*.json"))
            j = json.loads(f.read_text()); j["code_sha"] = "000000000000"; f.write_text(json.dumps(j))
        r = readout(build_root(base, "integ", lin(3.0), tweak=bad_sha), out=quiet, check_findings=False)
        T("a reused run with another code_sha: primary NOT DETERMINABLE (the secondary still determined)",
          r["verdict"].startswith("NOT DETERMINABLE") and any("code_sha" in p for p in r["problems"])
          and r["verdict2"].startswith("SHOWN"), r["verdict"])

        def bad_check93(root):
            (root / "pt300x" / "out" / "p3x-93m-lr1.2e-3-win-ctx" / "check_s3.json").write_text(json.dumps({"pass": False, "checks": [], "info": {"area": 0}}))
        r = readout(build_root(base, "check93", lin(3.0), tweak=bad_check93), out=quiet, check_findings=False)
        T("a reused pt300x 93M run whose CHECK7 file failed: primary NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("CHECK7 FAILED" in p for p in r["problems"]), r["verdict"])

        def bad_check(root):
            tag, rf, cf = G.run_file("391m", "none", PICK["391m"]["none"], 6)
            (root / "ptsz" / "out" / tag / cf).write_text(json.dumps({"pass": False, "checks": [{"check": "x", "ok": False}],
                                                                       "info": {"area": 0}}))
        r = readout(build_root(base, "check", lin(3.0), tweak=bad_check), out=quiet, check_findings=False)
        T("a 391m C run whose CHECK7 failed: primary and secondary NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and r["verdict2"].startswith("NOT DETERMINABLE") and any("CHECK7 FAILED" in p for p in r["problems"]), r["verdict"])
        dec = {(s, a) for s in NEW for a in G.ARMS} - {("731m", "win-ctx")}
        r = readout(build_root(base, "notfinal", lin(3.0), decided=dec), out=quiet, check_findings=False)
        T("731m win-ctx without its C decision: both NOT DETERMINABLE (not final)", r["verdict"].startswith("NOT DETERMINABLE")
          and r["verdict2"].startswith("NOT DETERMINABLE") and any("no C decision" in p for p in r["notfinal"]), r["verdict"])
        dec2 = {(s, a) for s in NEW for a in G.ARMS} - {("294m", "none")}
        r = readout(build_root(base, "notfinal294", lin(3.0), decided=dec2), out=quiet, check_findings=False)
        T("294m none without its C: the primary NOT DETERMINABLE, the secondary (no 294m) still SHOWN",
          r["verdict"].startswith("NOT DETERMINABLE") and r["verdict2"].startswith("SHOWN"), (r["verdict"], r["verdict2"]))

        def miss_c(root):
            tag, rf, cf = G.run_file("136m", "none", PICK["136m"]["none"], 7)
            (root / "ptsz" / "out" / tag / rf).unlink()
        r = readout(build_root(base, "missing", lin(3.0), tweak=miss_c), out=quiet, check_findings=False)
        T("a C run missing: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE") and any("missing" in p for p in r["notfinal"]),
          r["verdict"])

        def wrong_c(root):
            f = root / "ptsz" / "decisions.json"
            j = json.loads(f.read_text())
            for e_ in j["log"]:
                if (e_["size"], e_["arm"]) == ("136m", "none"):
                    e_["lr"] = "9.6e-3"
            f.write_text(json.dumps(j))
        r = readout(build_root(base, "wrongc", lin(3.0), tweak=wrong_c), out=quiet, check_findings=False)
        T("a C decision that is not the recomputed pick: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("recomputed selection" in p for p in r["problems"]), r["verdict"])
        r = readout(build_root(base, "range", lin(3.0), prior=70.0), out=quiet, check_findings=False)
        T("prior at ~70 (outside 88-91 +- 15): both verdict lines carry INVESTIGATE BEFORE READING",
          "INVESTIGATE BEFORE READING" in r["verdict"] and "INVESTIGATE BEFORE READING" in r["verdict2"] and len(r["investigate"]) >= 4, r["verdict"])
        # ---- amendment 3: R1, the full CLI --json path on a COMPLETE root (reused cells = FINDINGS F99, so main()'s
        # FINDINGS check passes; primary, secondary and 10-point all final), reloaded
        f99gain = lambda c, k: (F99["gain"][c] if c in REUSED else 30 + 3.0 * (xs[c] - m)) + sym[k]
        rj = build_root(base, "cli", f99gain, prior={c: F99["prior"][c] for c in REUSED})
        jout = rj / "ptsz" / "readout.json"
        argv, buf = sys.argv, io.StringIO()
        try:
            sys.argv = ["readout_ptsz.py", "--root", str(rj), "--json", str(jout)]
            with contextlib.redirect_stdout(buf):
                main()
            cli_err = None
        except BaseException as e_:   # noqa: BLE001
            cli_err = repr(e_)
        finally:
            sys.argv = argv
        jr = json.loads(jout.read_text()) if jout.exists() else {}
        T("R1: the CLI --json path on a complete root runs and writes a JSON that reloads (robust / robust2 / robust10 with string keys)",
          cli_err is None and jr.get("verdict", "ND").split()[0] in ("SHOWN", "NOT") and not jr["verdict"].startswith("NOT DETERMINABLE") and not jr.get("verdict2", "NOT DETERMINABLE").startswith("NOT DETERMINABLE")
          and all(k in jr for k in ("robust", "robust2", "robust10", "ten", "bridge294", "grad", "loso"))
          and "731m|s6" in jr["robust"]["loso"] and len(jr["robust10"]["loso"]) == 80, (cli_err, sorted(jr)[:8]))
        bad_keys = []

        def walk(o, path):
            if isinstance(o, dict):
                for k, v in o.items():
                    if not isinstance(k, str):
                        bad_keys.append((path, k))
                    walk(v, f"{path}.{k}")
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]")
        sys.argv = ["readout_ptsz.py"]
        try:
            res_cli = readout(rj, out=quiet)
        finally:
            sys.argv = argv
        walk(res_cli, "res")
        T("R1: no non-string dict key anywhere in the read-out's result (recursive walk)", not bad_keys, bad_keys[:5])
        T("  the CLI run wrote FIRST_READOUT.json atomically, read-only (mode 444)",
          (rj / "ptsz" / "FIRST_READOUT.json").exists() and oct((rj / "ptsz" / "FIRST_READOUT.json").stat().st_mode & 0o777) == "0o444")
        # ---- amendment 3: the FIRST_READOUT refusals
        rr = build_root(base, "refuse", lin(3.0))
        (rr / "ptsz" / "readout.txt").write_text("  731M      L40S   615,911,040 20.239   2.4e-3   55.0   6e-4   89.0  +34.0 [ ... ]  new\n")
        try:
            readout(rr, out=quiet, check_findings=False)
            ref1 = None
        except RuntimeError as e_:
            ref1 = str(e_)
        T("amendment 3: FIRST_READOUT.json missing while readout.txt already shows a final new cell -> the read-out refuses",
          ref1 is not None and "refusing" in ref1 and not (rr / "ptsz" / "FIRST_READOUT.json").exists(), ref1)
        (rr / "ptsz" / "readout.txt").write_text("")   # a shell redirect of this very read-out creates an empty file first
        readout(rr, out=quiet, check_findings=False)
        T("  an empty readout.txt (the shell redirect of the first read-out itself) does not block it", (rr / "ptsz" / "FIRST_READOUT.json").exists())
        frc = rr / "ptsz" / "FIRST_READOUT.json"
        frc.chmod(0o644); frc.write_text('{"time": "2026-11-06T00:30:00-05:00", "own_e')
        try:
            readout(rr, out=quiet, check_findings=False)
            ref2 = None
        except RuntimeError as e_:
            ref2 = str(e_)
        T("amendment 3: a FIRST_READOUT.json that cannot be parsed -> the read-out refuses", ref2 is not None and "cannot be parsed" in ref2, ref2)
        # ---- amendment 3: the negative-slope extreme toward 0 (the leave-one-seed-out MAXIMUM) and its seed's arms
        negp = lambda c, k: 30 - 3.0 * (xs[c] - m) + sym[k] + (25.0 if (c, k) == ("53m", 6) else 0.0)
        lines.clear()
        r = readout(build_root(base, "negext", negp), out=quiet, check_findings=False)
        rb_ = r["robust"]
        T("amendment 3: negative mean slope -> the extreme toward 0 is the leave-one-seed-out maximum (-53M s6, the planted seed), "
          "and the printed arms are that seed's (none 25 points below the cell's median)",
          rb_["mean"] < 0 and rb_["loso_ext"][1] == rb_["loso_max"][1] == ("53m", 6) and rb_["loso_min"][1] != ("53m", 6)
          and any(f"the extreme toward 0: {rb_['loso_max'][0]:+.3f} (-53M s6: none" in l for l in lines)
          and abs(rb_["ratios"]["loso"] - rb_["loso_max"][0] / rb_["mean"]) < 1e-12, (rb_["loso_ext"], rb_["loso_max"], rb_["loso_min"]))
        own_run = "psz-731m-lr6e-4-win-ctx-s6"   # a C seed: the stage-A selection stays complete
        trained = f"=== {own_run} start\nmodel: 731.4M parameters; gradient checkpointing on\n{S.TRAIN_MARK} 250 steps\ntorch.OutOfMemoryError: CUDA out of memory\n"
        (base / "logs").mkdir(exist_ok=True)
        for jid in ("7001", "7002"):
            (base / "logs" / f"llmadd_{own_run}_{jid}.out").write_text(trained)
        entry = f"{own_run} 7001:OUT_OF_MEMORY 7002:OUT_OF_MEMORY 2026-11-05T22:00:00-05:00   # OOM at step 1000, again after a resubmission"

        def oom_731(root, listed=True):
            tag, rf, cf = G.run_file("731m", "win-ctx", "6e-4", 6)
            pth = root / "ptsz" / "out" / tag / rf
            j = json.loads(pth.read_text()); j["curve"] = j["curve"][:4]; pth.write_text(json.dumps(j))
            (root / "ptsz" / "out" / tag / cf).unlink()
            (root / "ptsz" / "out" / tag / (rf + ".claim")).write_text("7002\n")
            if listed:
                (root / "ptsz" / "OWN_FAILURES.txt").write_text(entry + "\n")
        S._sacct = lambda: (0, f"7001|llmadd_{own_run}|OUT_OF_MEMORY\n7002|llmadd_{own_run}|OUT_OF_MEMORY\n", "")
        r = readout(build_root(base, "own", lin(3.0), tweak=oom_731), out=quiet, check_findings=False)
        T("I5 / I-A: a valid own-failure entry (731m win-ctx s6 OOM'd in jobs 7001 and 7002, both trained) -> primary and secondary "
          "NOT DETERMINABLE, nothing scored", r["verdict"].startswith("NOT DETERMINABLE") and r["verdict2"].startswith("NOT DETERMINABLE")
          and any("amendment I5" in p_ for p_ in r["problems"]), r["verdict"])
        lines.clear()
        r = readout(build_root(base, "ownbare", lin(3.0), tweak=lambda rt_: (oom_731(rt_, False), (rt_ / "ptsz" / "OWN_FAILURES.txt").write_text(own_run + "\n"))),
                    out=quiet, check_findings=False)
        T("I-A: a bare-name entry -> WARNING, ignored: the cell is merely NOT FINAL (the unfinished 731m run, 'dead' -> NOT FINAL, m7), no I5",
          r["verdict"].startswith("NOT DETERMINABLE") and not any("amendment I5" in p_ for p_ in r["problems"])
          and any(own_run in nf for nf in r["notfinal"]) and any(l.startswith("WARNING: OWN_FAILURES.txt entry") for l in lines), (r["problems"], r["notfinal"][:2]))
        lines.clear()
        r = readout(build_root(base, "ownok", lin(3.0), tweak=lambda rt_: (rt_ / "ptsz" / "OWN_FAILURES.txt").write_text(entry + "\n")),
                    out=quiet, check_findings=False)
        T("I-A: the same entry for a run that finished with a passing check -> the restored WARNING, ignored, verdict SHOWN",
          r["verdict"].startswith("SHOWN") and any(l.startswith(f"WARNING: {own_run} is listed as an own failure but finished with a passing check") for l in lines),
          (r["verdict"], lines[:3]))
        S._sacct = lambda: (0, "", "")
        rt = build_root(base, "first", lin(3.0))
        r = readout(rt, out=quiet, check_findings=False)
        fr = json.loads((rt / "ptsz" / "FIRST_READOUT.json").read_text())
        T("I-A: the first read-out that sees a final cell writes FIRST_READOUT.json (time, final cells, OWN_FAILURES entries)",
          fr["final_cells"] == list(NEW) and fr["own_entries"] == [] and fr["time"], fr)
        e0 = build_root(base, "notyet", lin(3.0), decided=set())
        readout(e0, out=quiet, check_findings=False)
        T("  a read-out without any final cell writes nothing", not (e0 / "ptsz" / "FIRST_READOUT.json").exists())
        S._sacct = lambda: (0, f"7001|llmadd_{own_run}|OUT_OF_MEMORY\n7002|llmadd_{own_run}|OUT_OF_MEMORY\n", "")
        oom_731(rt)   # after the first read-out: the run 'fails' and a valid-looking entry is appended
        lines.clear()
        r = readout(rt, out=quiet, check_findings=False)
        T("I-A: an entry appended after the first read-out -> WARNING, ignored (the cell is NOT FINAL, not I5)",
          not any("amendment I5" in p_ for p_ in r["problems"]) and any("added after the first read-out" in l for l in lines), lines[:3])
        S._sacct = lambda: (0, "", "")
        r = readout(build_root(base, "findings", lin(3.0)), out=quiet, check_findings=True)
        T("with the FINDINGS check on, designed reused cells != F99 -> NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("FINDINGS F99" in p for p in r["problems"]), r["verdict"])
        rt = build_root(base, "lowshare", lin(3.0))
        for s in NEW:   # the 731m prior runs with grad_prior 1.5 > grad_lm 1: flagged (descriptive)
            for seed in ALL8:
                tag, rf, cf = G.run_file(s, "win-ctx", PICK[s]["win-ctx"], seed)
                p = rt / "ptsz" / "out" / tag / rf
                j = json.loads(p.read_text())
                for e_ in j["curve"]:
                    if "grad_prior" in e_ and s == "731m":
                        e_["grad_prior"] = 1.5
                p.write_text(json.dumps(j))
        lines.clear()
        r = readout(rt, out=quiet, check_findings=False)
        T("LM share 0.40 in 731m (prior > LM in every on-window eval): FLAG printed, verdict unchanged",
          abs(r["grad"]["731m"]["med"] - 0.4) < 1e-9 and r["grad"]["731m"]["gt"] == 1.0 and r["verdict"].startswith("SHOWN")
          and any("731M" in l and "FLAG" in l for l in lines), r["grad"]["731m"])
    finally:
        S._sacct = real_sacct
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
    try:
        res = readout(root)
    except RuntimeError as e:   # amendment 3: the FIRST_READOUT refusals, loudly
        print(f"REFUSED: {e}")
        sys.exit(2)
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
