"""Pre-registered read-out of runs/pt2sz (intent spec and rules: runs/pt2sz/AUDIT.md "Pre-registered analysis", written and
self-tested before any pt2sz run). Question: with five sizes of the 2% / 3e8-token setting on one cluster (Torch
A100-80GB, trainer 93e22bf4c4c0, pt2same's job body), does the 8-seed paired area gain (win-ctx - none, readout_rules.area)
increase with ln N_nonemb?
Cells (each arm at its own pick, seeds 0-7): reused = exactly F98's cells, read through readout_pt2same's own run_paths /
load_run (53M none@8e-3 / win-ctx@2e-3 and 190M 4.8e-3 / 6e-4 from pt2same; 505M 4.8e-3 / 6e-4 from pt2s505), asserted
equal to F98's numbers (pt2same readout.json, FINDINGS F98); new = 93M and 294M at the C decisions of runs/pt2sz/decisions.json,
recomputed here by select_pt2sz.select_arm (F87 rule 1) and asserted equal.
PRIMARY: OLS slope b of the five 8-seed mean gains on ln N_nonemb. 'SHOWN (direction)' iff ALL of
  (a) the 90% seed-bootstrap CI (pt300x's: seeds resampled within each size, arms paired; 10000 resamples, numpy
      default_rng(0), one draw of (10000, 8) seed indices per size in ascending N order; percentile) excludes 0 on b's side;
  (b) the slopes on seeds 0-3 and on seeds 4-7 (point estimates) have the sign of b;
  (c) the size-level interval from the OLS residuals of the five points (b +- t(0.95, df 3) x se(b)) excludes 0.
Otherwise 'NOT SHOWN' with the numbers and the failed criteria. Any cell not final, failing an integrity check, with a run
missing, or a reused cell that is not F98's -> 'NOT DETERMINABLE'. A new cell outside its expected range (+-15) -> the
verdict carries 'INVESTIGATE BEFORE READING'.
F98's own statistics on the five points (as F98 / pt2same defined them; no new verdict, printed next to the primary): the
slope with the JOINT paired seed bootstrap (one (10000, 8) draw of default_rng(0) shared by every size and both arms) and
pt2same's rule ('SHOWN (+/-)' iff that CI excludes 0 and both halves have its sign); the 8 per-seed slopes: positives x/8,
t interval (df 7, 90%) and p, exact sign-flip p; the slope of the per-size median gains.
Descriptive (no verdict): the two new points vs the line through the three old ones (residuals, joint bootstrap) and the
slope between the two new points; none and prior arms separately; leave one size out; slope weights; gain without b11; the
LM share of the gradient per prior cell (pttpp's grad_stats); the expected-range table.
    python readout_pt2sz.py [--root RUNS] [--json OUT]
    python readout_pt2sz.py --ka                  known answer: F98's numbers from the three reused cells (real runs)
    python readout_pt2sz.py --selftest [--work D] synthetic roots
Definitions: tools/readout_rules.py of the snapshot (e91ee8a8a846), unchanged."""
import functools
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
import select_pt2sz as S   # noqa: E402  (the generator G, readout_rules R, run loading, integrity, F87 rule 1)

G, R, PS, P2, TP, G2 = S.G, S.R, S.PS, S.P2, S.TP, S.G2
B = G.B
sys.path.insert(0, str(B / "runs" / "pt2same"))
assert S.sha(B / "runs" / "pt2same" / "readout_pt2same.py") == "bf7cced43238", "readout_pt2same.py changed"
import readout_pt2same as RO   # noqa: E402  (F98's frozen read-out: run_paths / load_run of the reused cells)

SIZES5 = ("53m", "93m", "190m", "294m", "505m")
OLD, NEW = ("53m", "190m", "505m"), tuple(G.SIZES)
LABEL = {"53m": "53M", "93m": "93M", "190m": "190M", "294m": "294M", "505m": "505M"}
ALL8, H1, H2 = tuple(range(8)), (0, 1, 2, 3), (4, 5, 6, 7)
N_BOOT = R.N_BOOT
assert N_BOOT == RO.N_BOOT == 10000 and RO.SIZES == OLD and RO.PICKS == {z: G.PICKS_3E8[z] for z in OLD}
assert all(RO.N_NONEMB[z] == TP.SIZES[z][5] for z in OLD)
# F98 (pt2same readout.json / readout.txt; FINDINGS F98, 1 decimal): 8-seed means of the reused cells and F98's statistics
F98_CELLS = {"none": {"53m": 20.59, "190m": 27.98, "505m": 24.45}, "prior": {"53m": 68.58, "190m": 77.93, "505m": 77.42},
             "gain": {"53m": 47.99, "190m": 49.95, "505m": 52.98}}
F98_JSON = B / "runs" / "pt2same" / "readout.json"
# expected ranges of the new cells (AUDIT.md "Expected ranges"): between the neighbours' F98 8-seed means; +-15 -> investigate
EXPECTED = {"93m": {"none": (20.6, 28.0), "prior": (68.6, 77.9), "gain": (48.0, 49.9)},
            "294m": {"none": (24.4, 28.0), "prior": (77.4, 77.9), "gain": (49.9, 53.0)}}
MARGIN = 15.0
LM_SHARE_FLAG = 0.6


# ---------------------------------------------------------------- cells
def ro_roots(root):
    return {"torch": Path(root) / "pt2same" / "out", "s505": Path(root) / "pt2s505" / "out", "burst": RO.ROOTS["burst"]}


def node_ok(d):
    """every GPU line of the run's node.txt (one block per start) names an A100-SXM4-80GB"""
    f = Path(d) / "node.txt"
    if not f.exists():
        return False, "no node.txt"
    g = [l.split(",")[0].strip() for l in f.read_text().splitlines() if l.startswith("NVIDIA") or ", " in l and "MiB" in l
         and not l.startswith("name")]
    g = [x for x in g if x]
    return bool(g) and all(x == "NVIDIA A100-SXM4-80GB" for x in g), (g if g else "no GPU line")


def reused_cell(root, size):
    c = dict(size=size, problems=[], pick=dict(RO.PICKS[size]), runs={"none": [], "win-ctx": []})
    roots = ro_roots(root)
    for arm in G.ARMS:
        lr = RO.PICKS[size][arm]
        areas = []
        for s in ALL8:
            run, chk = RO.run_paths("torch", size, arm, s, roots)
            st, a, info = RO.load_run(run, chk)
            if st != "ok":
                c["problems"].append(f"{LABEL[size]} {arm} seed {s}: {st} ({info})")
                areas.append(np.nan)
                c["runs"][arm].append(None)
                continue
            r = json.loads(run.read_text())
            c["problems"] += [f"{LABEL[size]} {arm}@{lr} seed {s}: {w}" for w in S.integrity(r, size, arm, lr, s)]
            good, g = node_ok(chk.parent)
            if not good:
                c["problems"].append(f"{LABEL[size]} {arm}@{lr} seed {s}: node.txt GPU {g}")
            if not math.isfinite(a):   # pt2same Amendment A3: the frozen load_run lets a NaN area through
                c["problems"].append(f"{LABEL[size]} {arm}@{lr} seed {s}: area {a} not finite")
            areas.append(a)
            c["runs"][arm].append(r)
        c["an" if arm == "none" else "ap"] = np.array(areas, float)
    return c


def new_cell(root, size, state, own):
    c = dict(size=size, problems=[], pick={}, notfinal=[], runs={"none": [], "win-ctx": []})
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
            elif not v["own"]:
                good, g = node_ok(G.run_paths(size, arm, pick, v["seed"], root)[1].parent)
                if not good:
                    c["problems"].append(f"{v['name']}: node.txt GPU {g}")
        if any(v["status"] != "ok" for v in recs):
            continue
        c["an" if arm == "none" else "ap"] = np.array([v["area"] for v in recs])
        c["runs"][arm] = [v["r"] for v in recs]
        c["own"] = c.get("own", []) + [v["name"] for v in recs if v["own"]]
    return c


def finish(c):
    c["nne"] = TP.SIZES[c["size"]][5]
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


def weights(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / ((x - x.mean()) ** 2).sum()


@functools.lru_cache(maxsize=None)
def boot_within(n_cells, n_seeds=8):
    """pt300x's: one (10000, n_seeds) draw per size, in the cells' (ascending N) order, from one default_rng(0) stream
    (deterministic, so cached; read-only arrays)"""
    rng = np.random.default_rng(0)
    out = [rng.integers(0, n_seeds, size=(N_BOOT, n_seeds)) for _ in range(n_cells)]
    for a in out:
        a.setflags(write=False)
    return out


@functools.lru_cache(maxsize=None)
def boot_joint(n_seeds=8):
    """pt2same's (F98): one (10000, 8) draw of default_rng(0), shared by every size and both arms (= readout_pt2same.boot_idx)"""
    a = RO.boot_idx(n_seeds)
    a.setflags(write=False)
    return a


def resid_interval(x, y, level=0.90):
    """b +- t(df) se(b), se from the OLS residuals of the size means; -> (b, lo, hi, p, df)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = len(y) - 2
    res = y - X @ beta
    se = math.sqrt(res @ res / df / ((x - x.mean()) ** 2).sum()) if df > 0 else float("nan")
    if not (df > 0 and math.isfinite(se)):
        return float(beta[1]), float("nan"), float("nan"), float("nan"), df
    t = stats.t.ppf(0.5 + level / 2, df)
    p = 2 * stats.t.sf(abs(beta[1]) / se, df) if se > 0 else 0.0
    return float(beta[1]), float(beta[1] - t * se), float(beta[1] + t * se), float(p), df


def slope_block(cells, metric="d", seeds=ALL8, boot="within"):
    """point slope, the 90% seed-bootstrap CI (within-size or joint), the residual t interval, per-cell means"""
    cells = sorted(cells, key=lambda c: c["x"])
    pos = np.array(seeds)
    x = np.array([c["x"] for c in cells])
    y = np.array([c[metric][pos].mean() for c in cells])
    if boot == "within":
        idx = boot_within(len(cells), len(seeds))
        Yb = np.stack([c[metric][pos][i].mean(1) for c, i in zip(cells, idx)], -1)
    else:
        assert boot == "joint" and len(seeds) == 8
        i = boot_joint()
        Yb = np.stack([c[metric][i].mean(1) for c in cells], -1)
    b = float(ols(x, y))
    bb = ols(x, Yb)
    _, rlo, rhi, p, df = resid_interval(x, y)
    return dict(b=b, lo=float(np.percentile(bb, 5)), hi=float(np.percentile(bb, 95)), rlo=rlo, rhi=rhi, p=p, df=int(df),
                n=len(cells), sizes=[c["size"] for c in cells], means=[float(v) for v in y])


def sgn(v):
    return 1 if v > 0 else -1 if v < 0 else 0


def verdict(main, h1, h2):
    """the pre-registered decision rule (a)(b)(c)"""
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


def f98_stats(cells):
    """F98's own statistics on these cells: joint-bootstrap slope + pt2same's rule; the 8 per-seed slopes (t df 7, exact
    sign-flip); the slope of the per-size medians"""
    cells = sorted(cells, key=lambda c: c["x"])
    x = np.array([c["x"] for c in cells])
    j = slope_block(cells, boot="joint")
    j1, j2 = slope_block(cells, seeds=H1, boot="within")["b"], slope_block(cells, seeds=H2, boot="within")["b"]
    sl = {"all": j["b"], "h1": j1, "h2": j2, "ci": (j["lo"], j["hi"])}
    rule = RO.verdict(sl)
    per = np.array([float(ols(x, [c["d"][s] for c in cells])) for s in ALL8])
    t = stats.t.ppf(0.95, 7)
    m, se = per.mean(), per.std(ddof=1) / math.sqrt(8)
    p = float(2 * stats.t.sf(abs(m) / se, 7)) if se > 0 else 0.0
    med = float(ols(x, [np.median(c["d"]) for c in cells]))
    return dict(b=j["b"], lo=j["lo"], hi=j["hi"], h1=j1, h2=j2, rule=rule, per_seed=per.tolist(), pos=int((per > 0).sum()),
                t_lo=float(m - t * se), t_hi=float(m + t * se), t_p=p, signflip_p=float(R.signflip_p(per)), median_slope=med)


def new_vs_old(cells):
    """the two new points vs the OLS line through the three old points (8-seed means), joint seed bootstrap (the same
    resample for all five cells, so the line and the new points move together); the slope between the two new points"""
    byz = {c["size"]: c for c in cells}
    xo = np.array([byz[z]["x"] for z in OLD])
    i = boot_joint()
    yo = np.array([byz[z]["d"].mean() for z in OLD])
    Yo = np.stack([byz[z]["d"][i].mean(1) for z in OLD], -1)
    bo = ols(xo, yo)
    Bo = ols(xo, Yo)
    ao, Ao = yo.mean() - bo * xo.mean(), Yo.mean(1) - Bo * xo.mean()
    out = {}
    for z in NEW:
        xz = byz[z]["x"]
        r = byz[z]["d"].mean() - (ao + bo * xz)
        rb = byz[z]["d"][i].mean(1) - (Ao + Bo * xz)
        out[z] = dict(pred=float(ao + bo * xz), got=float(byz[z]["d"].mean()), resid=float(r),
                      lo=float(np.percentile(rb, 5)), hi=float(np.percentile(rb, 95)))
    xn = np.array([byz[z]["x"] for z in NEW])
    yn = np.array([byz[z]["d"].mean() for z in NEW])
    Yn = np.stack([byz[z]["d"][i].mean(1) for z in NEW], -1)
    w = boot_within(2)
    Yw = np.stack([byz[z]["d"][ii].mean(1) for z, ii in zip(NEW, w)], -1)
    out["two_new"] = dict(b=float(ols(xn, yn)), lo_joint=float(np.percentile(ols(xn, Yn), 5)),
                          hi_joint=float(np.percentile(ols(xn, Yn), 95)), lo_within=float(np.percentile(ols(xn, Yw), 5)),
                          hi_within=float(np.percentile(ols(xn, Yw), 95)))
    out["old_line"] = dict(b=float(bo), a=float(ao))
    return out


def grad_stats(runs):
    """pttpp's grad_stats (runs/pttpp/readout/tools/pttpp_summary.py:514): LM vs prior gradient (pre-clip norms at the on-evals):
    share by norm a / (a + b) (median per seed, then median / min over seeds), the share of on-evals with prior > LM, the
    first on-eval's prior / LM, and the clip coefficient from the logged norms and cosine"""
    sh, gt, first, clip = [], [], [], []
    for r in runs:
        on = [e for e in r["curve"] if e.get("prior_on") and e.get("grad_lm") is not None]
        if not on:
            continue
        a = np.array([e["grad_lm"] for e in on]); b = np.array([e["grad_prior"] for e in on]); cs = np.array([e["grad_cos"] for e in on])
        sh.append(np.median(a / (a + b))); gt.append(float(np.mean(b > a))); first.append(b[0] / a[0])
        tot = np.sqrt(np.maximum(a ** 2 + b ** 2 + 2 * a * b * cs, 0))
        clip.append(np.minimum(1.0, 1.0 / (tot + 1e-6)))
    if not sh:
        return None
    cl = np.concatenate(clip)
    pooled = [e["grad_lm"] / (e["grad_lm"] + e["grad_prior"]) for r in runs for e in r["curve"]
              if e.get("prior_on") and e.get("grad_lm") is not None]
    return dict(share=float(np.median(sh)), share_min=float(np.min(sh)), share_pooled=float(np.median(pooled)),
                frac_gt=float(np.mean(gt)), first=(float(np.min(first)), float(np.max(first))), clip_med=float(np.median(cl)),
                clip_min=float(np.min(cl)), clip_frac=float(np.mean(cl < 1)), n=len(sh))


# ---------------------------------------------------------------- the read-out
def load_cells(root, check_f98=True):
    root = Path(root)
    log, state = G.decisions(root / "pt2sz" / "decisions.json")
    own = S.parse_own(root)
    cells = {s: finish(reused_cell(root, s)) for s in OLD}
    cells.update({s: finish(new_cell(root, s, state, own)) for s in NEW})
    if check_f98:   # the reused cells must still be F98's
        js = json.loads(F98_JSON.read_text())["sizes"] if F98_JSON.exists() else {}
        for s in OLD:
            c = cells[s]
            if not c["ok"]:
                continue
            got = {"gain": c["d"].mean(), "none": c["an"].mean(), "prior": c["ap"].mean()}
            bad = {k: round(float(v), 3) for k, v in got.items() if abs(v - F98_CELLS[k][s]) > 0.005 + 1e-9}
            if js:
                jj = {"gain": js[s]["all"], "none": js[s]["none"], "prior": js[s]["prior"]}
                bad.update({f"{k} (readout.json)": round(float(got[k]), 6) for k in jj if abs(got[k] - jj[k]) > 1e-9})
                if not np.allclose(c["d"], js[s]["delta_per_seed"], rtol=0, atol=1e-9):
                    bad["delta_per_seed (readout.json)"] = c["d"].round(3).tolist()
            if bad:
                c["problems"].append(f"{LABEL[s]} differs from F98 (pt2same readout.json / FINDINGS): {bad}")
                c["ok"] = False
    return log, cells


def readout(root, out=print, check_f98=True):
    log, cells = load_cells(root, check_f98)
    out(f"# pt2sz read-out; root {root}; generator {S.sha(S.GEN)}; readout_rules {S.sha(S.RULES_DIR / 'readout_rules.py')}; "
        f"decisions {len(log)} entries")
    out("# 3 of the 5 points (53M, 190M, 505M = F98's cells) were observed before this analysis was registered; 93M, 294M and "
        "criterion (c) are the new evidence (AUDIT.md: simulated false-SHOWN rate of the primary under no growth)")
    res = {"cells": {}, "problems": [], "notfinal": []}
    for s in SIZES5:
        res["problems"] += cells[s]["problems"]
        res["notfinal"] += cells[s].get("notfinal", [])
    out("\n## integrity and completeness")
    for p in res["problems"]:
        out(f"  PROBLEM {p}")
    for p in res["notfinal"]:
        out(f"  NOT FINAL {p}")
    if not res["problems"] and not res["notfinal"]:
        out("  every counted run present, finished, intended (integrity re-check, check.json passed, area double path, "
            "node.txt A100-SXM4-80GB); the three reused cells = F98's")
    for s in NEW:
        if cells[s].get("own"):
            out(f"  own failures counted (F87 rule 2): {cells[s]['own']}")
    ok = [cells[s] for s in SIZES5 if cells[s]["ok"]]
    flags = []
    out("\n## expected ranges of the new cells (AUDIT.md; outside +-15 -> investigate before reading)")
    for s in NEW:
        c = cells[s]
        if not c["ok"]:
            out(f"  {LABEL[s]}: not available")
            continue
        for k, v in (("none", c["an"].mean()), ("prior", c["ap"].mean()), ("gain", c["d"].mean())):
            lo, hi = EXPECTED[s][k]
            bad = not (lo - MARGIN <= v <= hi + MARGIN)
            out(f"  {LABEL[s]} {k}: 8-seed mean {v:.1f}; expected {lo}-{hi} (+-{MARGIN:.0f}) {'OUTSIDE: INVESTIGATE' if bad else 'ok'}")
            if bad:
                flags.append(f"{LABEL[s]} {k} {v:.1f}")
    out("\n## cells at the picks (8 seeds; area points; gain = win-ctx - none, paired by seed; Torch A100-80GB)")
    out(f"  {'size':5s} {'N_nonemb':>12s} {'ln N':>6s} {'none lr':>8s} {'none':>6s} {'prior lr':>8s} {'prior':>6s} {'gain':>6s} "
        f"{'[seed CI]':>15s} {'s0-3':>6s} {'s4-7':>6s} {'wins':>5s} {'src':>8s}")
    for c in sorted(ok, key=lambda c: c["x"]):
        d = c["d"]
        bd = d[boot_within(1)[0]].mean(1)
        out(f"  {LABEL[c['size']]:5s} {c['nne']:12,d} {c['x']:6.3f} {c['pick']['none']:>8s} {c['an'].mean():6.2f} "
            f"{c['pick']['win-ctx']:>8s} {c['ap'].mean():6.2f} {d.mean():+6.2f} [{np.percentile(bd, 5):+6.1f}, {np.percentile(bd, 95):+6.1f}] "
            f"{d[:4].mean():+6.2f} {d[4:].mean():+6.2f} {int((d > 0).sum()):>3d}/8 {'new' if c['size'] in NEW else 'F98'}")
        res["cells"][c["size"]] = dict(nne=c["nne"], pick=c["pick"], none=c["an"].tolist(), prior=c["ap"].tolist(),
                                       gain=float(d.mean()), gain_per_seed=d.tolist())
    if len(ok) < len(SIZES5):
        v = f"NOT DETERMINABLE ({len(SIZES5) - len(ok)} of 5 cells not available: " + \
            "; ".join((res["problems"] + res["notfinal"])[:3]) + ")"
        out(f"\n## PRIMARY\n  VERDICT: {v}")
        res["verdict"] = v
        return res
    main = slope_block(ok)
    h1, h2 = slope_block(ok, seeds=H1), slope_block(ok, seeds=H2)
    v, crit = verdict(main, h1, h2)
    if flags:
        v += f"  [INVESTIGATE BEFORE READING: outside the expected range: {', '.join(flags)}]"
    out("\n## PRIMARY (pre-registered): OLS slope of the five 8-seed mean gains on ln N_nonemb")
    out(f"  b = {main['b']:+.3f} per e-fold; seed bootstrap (within size) 90% CI [{main['lo']:+.3f}, {main['hi']:+.3f}]; "
        f"residual t (df {main['df']}) 90% [{main['rlo']:+.3f}, {main['rhi']:+.3f}], p {main['p']:.3f}")
    out(f"  halves: seeds 0-3 {h1['b']:+.3f}, seeds 4-7 {h2['b']:+.3f}")
    out(f"  VERDICT: {v}")
    res.update(primary=main, h1=h1, h2=h2, criteria=crit, verdict=v, investigate=flags)
    f = f98_stats(ok)
    out("\n## F98's own statistics on the five points (as F98 / pt2same defined them; not a second verdict)")
    out(f"  slope {f['b']:+.3f}, JOINT paired seed bootstrap 90% [{f['lo']:+.3f}, {f['hi']:+.3f}]; halves {f['h1']:+.3f} / "
        f"{f['h2']:+.3f}; pt2same's rule -> {f['rule']}")
    out(f"  8 per-seed slopes {', '.join(f'{v_:+.2f}' for v_ in f['per_seed'])}: {f['pos']}/8 positive; t (df 7) 90% "
        f"[{f['t_lo']:+.2f}, {f['t_hi']:+.2f}], p {f['t_p']:.3f}; exact sign-flip p {f['signflip_p']:.3f}; slope of the "
        f"per-size medians {f['median_slope']:+.2f}")
    res["f98_stats"] = f
    out("\n## descriptive (no verdict)")
    x5 = [c["x"] for c in sorted(ok, key=lambda c: c["x"])]
    w = weights(x5)
    out("  OLS slope weights: " + ", ".join(f"{LABEL[c['size']]} {wi:+.3f}" for c, wi in zip(sorted(ok, key=lambda c: c["x"]), w))
        + f"; the new points carry {100 * sum(abs(wi) for c, wi in zip(sorted(ok, key=lambda c: c['x']), w) if c['size'] in NEW) / sum(abs(w)):.0f}% of |w|")
    nv = new_vs_old(ok)
    out(f"  the line through the three old points: {nv['old_line']['b']:+.3f} per e-fold (= F98's slope)")
    for z in NEW:
        q = nv[z]
        out(f"  {LABEL[z]} vs that line: predicted {q['pred']:+.2f}, observed {q['got']:+.2f}, residual {q['resid']:+.2f} "
            f"(joint seed bootstrap 90% [{q['lo']:+.2f}, {q['hi']:+.2f}])")
    q = nv["two_new"]
    out(f"  slope between the two new points alone: {q['b']:+.3f} (90% joint [{q['lo_joint']:+.3f}, {q['hi_joint']:+.3f}], "
        f"within-size [{q['lo_within']:+.3f}, {q['hi_within']:+.3f}])")
    arms = {k: (slope_block(ok, metric=m), slope_block(ok, metric=m, boot="joint")) for k, m in (("none", "an"), ("prior", "ap"))}
    out("  arms separately (mean area on ln N): " + "; ".join(
        f"{k} {a_['b']:+.3f} [within {a_['lo']:+.3f}, {a_['hi']:+.3f}; joint {j_['lo']:+.3f}, {j_['hi']:+.3f}]" for k, (a_, j_) in arms.items()))
    loso = {s: slope_block([c for c in ok if c["size"] != s])["b"] for s in SIZES5}
    out("  leave one size out: " + ", ".join(f"-{LABEL[s]} {b:+.3f}" for s, b in loso.items()))
    for c in ok:
        c["d5"] = np.array([R.area(p_, R.B5) - R.area(n_, R.B5) for p_, n_ in zip(c["runs"]["win-ctx"], c["runs"]["none"])]) \
            if all(len(c["runs"][a]) == 8 and all(r is not None for r in c["runs"][a]) for a in G.ARMS) else None
    nb = None   # (an own failure has no run JSON: then the no-b11 slope is not printed)
    if all(c.get("d5") is not None for c in ok):
        nb = slope_block(ok, metric="d5")
        out("  gain without b11 (readout_rules.B5): " + ", ".join(f"{LABEL[c['size']]} {c['d5'].mean():+.2f}" for c in
                                                                 sorted(ok, key=lambda c: c["x"])) + f"; slope {nb['b']:+.3f}")
    gs = {}
    out("  LM share of the gradient in the window (prior cells; pre-clip norms a / (a + b) at on-evals, median per seed, "
        "then median [min] over seeds; pooled median; prior > LM evals; clip; first on-eval prior / LM)")
    for c in sorted(ok, key=lambda c: c["x"]):
        g = grad_stats([r for r in c["runs"]["win-ctx"] if r is not None])
        gs[c["size"]] = g
        if g:
            out(f"    {LABEL[c['size']]:5s} {g['share']:.3f} [{g['share_min']:.3f}] pooled {g['share_pooled']:.3f}; prior > LM "
                f"{100 * g['frac_gt']:.0f}%; clip median {g['clip_med']:.2f} (min {g['clip_min']:.2f}, < 1 on {100 * g['clip_frac']:.0f}%); "
                f"first {g['first'][0]:.2f}-{g['first'][1]:.2f}" + ("  [LM share < 0.6: the prior dominates the update]"
                                                                     if g["share"] < LM_SHARE_FLAG else ""))
    res.update(weights=dict(zip([c["size"] for c in sorted(ok, key=lambda c: c["x"])], w.tolist())), new_vs_old=nv,
               arms={k: {"within": a_, "joint": j_} for k, (a_, j_) in arms.items()}, loso=loso, no_b11=nb, grad=gs)
    return res


# ---------------------------------------------------------------- known answer: F98 from the three reused cells (real runs)
F98_KA = {"slope": (1.4354, 0.0006), "ci_joint": ((0.2201, 2.6442), 0.0006), "h1": (1.837, 0.0006), "h2": (1.034, 0.0006),
          "none_arm": ((1.388, 0.155, 2.668), 0.0006), "prior_arm": ((2.823, 2.557, 3.137), 0.0006),
          "per_seed": ((0.92, 0.24, 2.73, 3.45, 0.31, 1.10, -2.21, 4.93), 0.006), "t": ((-0.05, 2.92), 0.006),
          "t_p": (0.109, 0.0006), "signflip_p": (0.109, 0.0006), "median_slope": (1.08, 0.006),
          "weights": ((-0.316653, 0.047304, 0.269349), 1e-6), "lm_share": ((0.676, 0.740, 0.757), 0.0006)}


def ka(root):
    cells = [finish(reused_cell(root, s)) for s in OLD]
    probs = [p for c in cells for p in c["problems"]]
    ok, n = 0, 0

    def T(name, got, want, tol):
        nonlocal ok, n
        n += 1
        g = np.atleast_1d(np.asarray(got, float)); w = np.atleast_1d(np.asarray(want, float))
        good = g.shape == w.shape and bool(np.all(np.abs(g - w) <= tol))
        ok += good
        print(f"{'PASS' if good else 'FAIL'} {name}: got {np.round(g, 4).tolist()} want {w.tolist()} (tol {tol})")

    n += 1
    ok += not probs
    print(f"{'PASS' if not probs else 'FAIL'} the 48 reused runs (F98's cells, readout_pt2same's run_paths / load_run) pass the "
          "integrity re-check, area double path and node.txt A100-SXM4-80GB" + (f": {probs[:3]}" if probs else ""))
    if probs:
        print(f"KA FAIL {ok}/{n}")
        return False
    js = json.loads(F98_JSON.read_text())
    for k, key in (("gain", "d"), ("none", "an"), ("prior", "ap")):
        T(f"8-seed {k} per size (53M, 190M, 505M) = F98 readout (2 decimals)", [c[key].mean() for c in cells],
          [F98_CELLS[k][c["size"]] for c in cells], 0.005 + 1e-9)
    T("per-seed gains = pt2same readout.json delta_per_seed exactly", np.concatenate([c["d"] for c in cells]),
      np.concatenate([js["sizes"][c["size"]]["delta_per_seed"] for c in cells]), 1e-9)
    j = slope_block(cells, boot="joint")
    T("slope (F98 +1.44)", j["b"], *F98_KA["slope"])
    T("joint paired seed bootstrap 90% CI (F98 [+0.22, +2.64])", [j["lo"], j["hi"]], *F98_KA["ci_joint"])
    T("= readout.json slope / CI", [j["b"], j["lo"], j["hi"]], [js["slope"]["all"], *js["slope"]["ci"]], 1e-9)
    T("seeds 0-3 slope", slope_block(cells, seeds=H1)["b"], *F98_KA["h1"])
    T("seeds 4-7 slope", slope_block(cells, seeds=H2)["b"], *F98_KA["h2"])
    f = f98_stats(cells)
    T("pt2same's rule on these cells = F98's verdict 'SHOWN (+...'", float(f["rule"].startswith("SHOWN (+")), 1.0, 0)
    for arm, m in (("none", "an"), ("prior", "ap")):
        a_ = slope_block(cells, metric=m, boot="joint")
        T(f"{arm} arm slope + joint CI (F98 none +1.39 [+0.16, +2.67], prior +2.82 [+2.56, +3.14])", [a_["b"], a_["lo"], a_["hi"]],
          *F98_KA[f"{arm}_arm"])
    T("8 per-seed slopes (F98 audit I2)", f["per_seed"], *F98_KA["per_seed"])
    T("per-seed t interval df 7 (F98 [-0.05, +2.92])", [f["t_lo"], f["t_hi"]], *F98_KA["t"])
    T("per-seed t p (F98 .11)", f["t_p"], *F98_KA["t_p"])
    T("exact sign-flip p (F98 .11)", f["signflip_p"], *F98_KA["signflip_p"])
    T("slope of the per-size medians (F98 audit +1.08)", f["median_slope"], *F98_KA["median_slope"])
    T("OLS slope weights (pt2same Amendment A1: -0.317 / +0.047 / +0.269)", weights([c["x"] for c in cells]), *F98_KA["weights"])
    g = [grad_stats([r for r in c["runs"]["win-ctx"]]) for c in cells]
    T("LM share of the gradient, pooled median over the on-evals (F98 audit m4: 0.676 / 0.740 / 0.757)",
      [x["share_pooled"] for x in g], *F98_KA["lm_share"])
    print("  (pttpp's per-seed-median definition, printed by the read-out next to it: "
          + ", ".join(f"{LABEL[c['size']]} {x['share']:.3f} [{x['share_min']:.3f}]" for c, x in zip(cells, g)) + ")")
    print(f"KA {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    return ok == n


# ---------------------------------------------------------------- selftest (synthetic roots)
PICK = {"93m": {"none": "4.8e-3", "win-ctx": "6e-4"}, "294m": {"none": "9.6e-3", "win-ctx": "3e-4"}}


def fake_node(d, gpu="NVIDIA A100-SXM4-80GB"):
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / "node.txt").write_text(f"2026-11-06T00:00:00Z\nga001\nSLURM_JOB_ID 1 restart 0\nname, driver_version, memory.total [MiB]\n"
                                      f"{gpu}, 610.43.02, 81920 MiB\ncpus 8\n")


def build_root(base, name, gain, prior=77.0, q=None, decided=None, tweak=None, lm_share=0.8):
    """gain(size, seed) -> designed gain; prior area = prior + q[seed]; none = prior area - gain. New cells: the pick has the
    designed areas, every other grid lr 10 points lower. Old cells at readout_pt2same's paths (pt2same / pt2s505 layout)."""
    root = Path(base) / name
    (root / "pt2sz").mkdir(parents=True)
    q = q or [0.5, -0.5, 1.0, -1.0, 0.5, -0.5, 1.0, -1.0]
    val = lambda s, arm, seed: (prior + q[seed]) - (gain(s, seed) if arm == "none" else 0.0)
    roots = ro_roots(root)
    for s in OLD:
        for arm in G.ARMS:
            lr = RO.PICKS[s][arm]
            for seed in ALL8:
                run, chk = RO.run_paths("torch", s, arm, seed, roots)
                tag = run.parent.name
                F = G.facts(s, arm, lr, seed, "C" if seed >= 4 else "A", G2, tag=tag, lm0=S.ref_lm0(s, seed))
                I = G.intent_of(G2, F)
                S.fake_run(run, chk, I, val(s, arm, seed), check="synthetic", lm_share=lm_share)
                fake_node(chk.parent)
    log = []
    for s in NEW:
        for arm in G.ARMS:
            pk = PICK[s][arm]
            for lr in G.GRID[s][arm]:
                for seed in H1:
                    S.fake_new_run(root, s, arm, lr, seed, val(s, arm, seed) - (0 if lr == pk else 10),
                                   bpb=1.19 if lr == pk else 1.2, check="synthetic", lm_share=lm_share)
                    fake_node(G.run_paths(s, arm, lr, seed, root)[1].parent)
            for seed in H2:
                S.fake_new_run(root, s, arm, pk, seed, val(s, arm, seed), check="synthetic", lm_share=lm_share)
                fake_node(G.run_paths(s, arm, pk, seed, root)[1].parent)
            if decided is None or (s, arm) in decided:
                log.append(dict(seq=len(log), size=s, arm=arm, stage="C", lr=pk))
    (root / "pt2sz" / "decisions.json").write_text(json.dumps({"log": log}))
    if tweak:
        tweak(root)
    return root


def selftest(work):
    base = Path(tempfile.mkdtemp(prefix="pt2sz_readout_selftest_", dir=work))
    ok, n = 0, 0

    def T(name, cond, info=""):
        nonlocal ok, n
        n += 1
        ok += bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info and not cond else ""))

    xs = {s: math.log(TP.SIZES[s][5]) for s in SIZES5}
    m = float(np.mean(list(xs.values())))
    lines = []
    quiet = lambda t="": lines.append(t)
    sym = [1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 2.0, -2.0]   # zero mean within each half
    try:
        r = readout(build_root(base, "pos", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k], lm_share=0.7), out=quiet, check_f98=False)
        T("positive slope: SHOWN (increase)", r["verdict"].startswith("SHOWN (increase)"), r["verdict"])
        T("  designed slope 3 read +3.000 (8 seeds, both halves); residual interval degenerate at 3",
          abs(r["primary"]["b"] - 3) < 1e-9 and abs(r["h1"]["b"] - 3) < 1e-9 and abs(r["h2"]["b"] - 3) < 1e-9
          and abs(r["primary"]["rlo"] - 3) < 1e-6, (r["primary"]["b"], r["h1"]["b"], r["h2"]["b"], r["primary"]["rlo"]))
        T("  arms: none -3.000, prior 0.000; LOSO all 3.000; no-b11 slope 3.000",
          abs(r["arms"]["none"]["within"]["b"] + 3) < 1e-9 and abs(r["arms"]["prior"]["within"]["b"]) < 1e-9
          and all(abs(v - 3) < 1e-9 for v in r["loso"].values()) and abs(r["no_b11"]["b"] - 3) < 1e-9)
        T("  new points on the old line: residuals 0, two-new slope 3; F98 stats: per-seed slopes all 3, sign-flip p 1/128",
          all(abs(r["new_vs_old"][z]["resid"]) < 1e-9 for z in NEW) and abs(r["new_vs_old"]["two_new"]["b"] - 3) < 1e-9
          and all(abs(v - 3) < 1e-9 for v in r["f98_stats"]["per_seed"]) and abs(r["f98_stats"]["signflip_p"] - 2 / 256) < 1e-12
          and r["f98_stats"]["rule"].startswith("SHOWN (+"), r["f98_stats"])
        T("  LM share = the designed 0.700 in every prior cell; no INVESTIGATE flag, no problem",
          all(abs(g["share"] - 0.7) < 1e-9 for g in r["grad"].values()) and not r["investigate"] and not r["problems"]
          and not r["notfinal"], r["grad"])
        r = readout(build_root(base, "zero", lambda s, k: 50 + [3.0, -1.0, 2.0, -4.0, 1.0, -3.0, 4.0, -2.0][k]), out=quiet, check_f98=False)
        T("zero slope: NOT SHOWN, (a) failed", r["verdict"].startswith("NOT SHOWN") and not r["criteria"]["a"] and abs(r["primary"]["b"]) < 1e-9,
          r["verdict"])
        r = readout(build_root(base, "neg", lambda s, k: 50 - 3 * (xs[s] - m) + sym[k]), out=quiet, check_f98=False)
        T("negative slope: SHOWN (decrease), -3.000", r["verdict"].startswith("SHOWN (decrease)") and abs(r["primary"]["b"] + 3) < 1e-9,
          r["verdict"])
        halves = lambda s, k: 50 + (6.0 if k < 4 else -1.5) * (xs[s] - m) + [1, -1, 1, -1][k % 4]
        r = readout(build_root(base, "halves", halves), out=quiet, check_f98=False)
        T("halves disagree (seeds 0-3 +6, seeds 4-7 -1.5): NOT SHOWN, (b) failed, 8-seed +2.25",
          r["verdict"].startswith("NOT SHOWN") and not r["criteria"]["b"] and abs(r["primary"]["b"] - 2.25) < 1e-9
          and abs(r["h1"]["b"] - 6) < 1e-9 and abs(r["h2"]["b"] + 1.5) < 1e-9, r["verdict"])
        X = np.c_[np.ones(5), [xs[s] for s in SIZES5]]
        e = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        e = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]
        e = 6 * e / np.abs(e).max()
        resid = dict(zip(SIZES5, e))
        tiny = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        r = readout(build_root(base, "resid", lambda s, k: 50 + 1.0 * (xs[s] - m) + resid[s] + tiny[k]), out=quiet, check_f98=False)
        T("residual interval includes 0 (b 1.0, size scatter +-6, seed noise +-0.1): NOT SHOWN, (c) failed, (a) and (b) passed",
          r["verdict"].startswith("NOT SHOWN") and r["criteria"]["a"] and r["criteria"]["b"] and not r["criteria"]["c"]
          and abs(r["primary"]["b"] - 1) < 1e-9 and r["primary"]["rlo"] < 0 < r["primary"]["rhi"], r["verdict"])
        T("  ... and F98's rule (joint bootstrap + halves; no residual criterion) reads SHOWN (+) on the same data",
          r["f98_stats"]["rule"].startswith("SHOWN (+"), r["f98_stats"]["rule"])
        off = {"93m": 4.0, "294m": -2.0}
        r = readout(build_root(base, "offline", lambda s, k: 50 + 2 * (xs[s] - m) + off.get(s, 0.0) + sym[k]), out=quiet, check_f98=False)
        T("new points off the old line by +4 / -2: residuals read +4.000 / -2.000 (degenerate CIs)",
          abs(r["new_vs_old"]["93m"]["resid"] - 4) < 1e-9 and abs(r["new_vs_old"]["294m"]["resid"] + 2) < 1e-9
          and abs(r["new_vs_old"]["old_line"]["b"] - 2) < 1e-9, r["new_vs_old"])
        shared = [3.0, -3.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
        r = readout(build_root(base, "shared", lambda s, k: 50 + 1.5 * (xs[s] - m) + shared[k]), out=quiet, check_f98=False)
        T("a seed effect shared by every size: the joint (F98) CI collapses to the point, the within-size (primary) CI does not",
          abs(r["f98_stats"]["hi"] - r["f98_stats"]["lo"]) < 1e-9 and r["primary"]["hi"] - r["primary"]["lo"] > 0.1, r["f98_stats"])

        def bad_sha(root):
            run, _ = RO.run_paths("torch", "190m", "none", 5, ro_roots(root))
            j = json.loads(run.read_text()); j["code_sha"] = "000000000000"; run.write_text(json.dumps(j))
        r = readout(build_root(base, "integ", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k], tweak=bad_sha), out=quiet, check_f98=False)
        T("a reused run with another code_sha: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("code_sha" in p for p in r["problems"]), r["verdict"])

        def l40s(root):
            fake_node(G.run_paths("294m", "win-ctx", PICK["294m"]["win-ctx"], 6, root)[1].parent, "NVIDIA L40S")
        r = readout(build_root(base, "gpu", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k], tweak=l40s), out=quiet, check_f98=False)
        T("a counted new run whose node.txt names an L40S: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("node.txt" in p for p in r["problems"]), r["verdict"])

        def nan505(root):
            run, chk = RO.run_paths("torch", "505m", "win-ctx", 2, ro_roots(root))
            j = json.loads(run.read_text()); j["curve"][5]["held_b33"] = float("nan"); run.write_text(json.dumps(j))
            c = json.loads(chk.read_text()); c["info"]["area"] = float("nan"); chk.write_text(json.dumps(c))
        r = readout(build_root(base, "nan", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k], tweak=nan505), out=quiet, check_f98=False)
        T("a reused 505M run with a NaN area (the frozen load_run lets it through): NOT DETERMINABLE",
          r["verdict"].startswith("NOT DETERMINABLE") and any("not finite" in p or "double path" in p for p in r["problems"]), r["verdict"])

        def bad_check(root):
            _, chk, _ = G.run_paths("294m", "none", PICK["294m"]["none"], 6, root)
            chk.write_text(json.dumps({"pass": False, "checks": [{"check": "x", "ok": False}], "info": {"area": 0}}))
        r = readout(build_root(base, "check", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k], tweak=bad_check), out=quiet, check_f98=False)
        T("a C run whose check.json failed: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("CHECK_PY FAILED" in p for p in r["problems"]), r["verdict"])
        dec = {(s, a) for s in NEW for a in G.ARMS} - {("294m", "win-ctx")}
        r = readout(build_root(base, "notfinal", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k], decided=dec), out=quiet, check_f98=False)
        T("294M win-ctx without its C decision: NOT DETERMINABLE (not final)", r["verdict"].startswith("NOT DETERMINABLE")
          and any("no C decision" in p for p in r["notfinal"]), r["verdict"])

        def miss_c(root):
            G.run_paths("93m", "none", PICK["93m"]["none"], 7, root)[0].unlink()
        r = readout(build_root(base, "missing", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k], tweak=miss_c), out=quiet, check_f98=False)
        T("a C run missing: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE") and any("missing" in p for p in r["notfinal"]),
          r["verdict"])

        def miss_old(root):
            RO.run_paths("torch", "53m", "win-ctx", 3, ro_roots(root))[0].unlink()
        r = readout(build_root(base, "missold", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k], tweak=miss_old), out=quiet, check_f98=False)
        T("a reused run missing: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("incomplete" in p for p in r["problems"]), r["verdict"])

        def wrong_c(root):
            f = root / "pt2sz" / "decisions.json"
            j = json.loads(f.read_text())
            for e_ in j["log"]:
                if (e_["size"], e_["arm"]) == ("93m", "none"):
                    e_["lr"] = "9.6e-3"
            f.write_text(json.dumps(j))
        r = readout(build_root(base, "wrongc", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k], tweak=wrong_c), out=quiet, check_f98=False)
        T("a C decision that is not the recomputed pick: NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("recomputed selection" in p for p in r["problems"]), r["verdict"])
        r = readout(build_root(base, "range", lambda s, k: 20 + 3 * (xs[s] - m) + sym[k], prior=77.0), out=quiet, check_f98=False)
        T("gains ~20 (outside 48-53 +- 15): the verdict carries INVESTIGATE BEFORE READING (gain and none of both new cells)",
          "INVESTIGATE BEFORE READING" in r["verdict"] and sum("gain" in f_ for f_ in r["investigate"]) == 2, r["investigate"])
        r = readout(build_root(base, "f98", lambda s, k: 50 + 3 * (xs[s] - m) + sym[k]), out=quiet, check_f98=True)
        T("with the F98 check on, designed reused cells != F98's numbers -> NOT DETERMINABLE", r["verdict"].startswith("NOT DETERMINABLE")
          and any("differs from F98" in p for p in r["problems"]), r["verdict"])
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
