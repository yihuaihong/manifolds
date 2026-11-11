"""Pre-registered read-out of pt2same (runs/pt2same/AUDIT.md, "Pre-registered read-out"; written and self-tested BEFORE any
result of this batch existed). Implements exactly that section, nothing else:
  integrity   every counted run: finished (last eval at step 18309), its check.json present and passing, and the area
              recomputed here (readout_rules.area) = the check's own area within 0.01 (double path). Missing / unfinished
              -> 'incomplete'; a failed check or a double-path mismatch -> 'WITHHELD'. Either -> no verdict.
  range       each NEW (Torch) cell's 8-seed mean area vs F86's 8-seed mean of the same cell on Burst (R02 addendum):
              outside +-15 -> 'OUT OF RANGE': the read-out stops after the range table (no gain / slope / verdict) until
              the cell was investigated and the reason is given with --investigated "<note>" (printed with the result).
  per size    Delta_s = area(win-ctx, seed s) - area(none, seed s) (paired by seed: same init, data stream and held-out
              split); mean over seeds 0-3, 4-7, 0-7, wins, 90% CI of the 8-seed mean (paired seed bootstrap).
  slope       OLS slope of the per-size mean Delta on ln N_nonemb over 53M, 190M (this batch, Torch) and 505M (pt2s505,
              Torch); 90% CI = joint paired seed bootstrap (one resample of the 8 seed indices per replicate, used for every
              size and both arms), 10000 resamples, numpy default_rng(0), percentile; the slopes of seeds 0-3 and 4-7 alone.
  verdict     'SHOWN (+ ...)' / 'SHOWN (- ...)' iff the 8-seed CI excludes 0 AND the 0-3 and 4-7 slopes have the same sign
              as the 8-seed slope; otherwise 'no size trend shown' with the point estimate, the CI and the reason.
              Never 'constant' (no equivalence test is pre-registered).
  descriptive per-arm area slopes (same bootstrap); Burst vs Torch for the same cells and seeds (site effect: arm areas and
              Delta, paired by seed); the mixed-site slope (Burst 53M / 190M + Torch 505M) next to the one-cluster slope.
    python readout_pt2same.py [--json out.json] [--investigated "<note>"]
    python readout_pt2same.py --selftest          (synthetic data only; touches no result)"""
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent              # runs/pt2same
B = HERE.parent.parent                              # /scratch/yh6210/llmadd_torch
sys.path.insert(0, str(HERE / "code" / "tools"))
assert hashlib.sha256((HERE / "code" / "tools" / "readout_rules.py").read_bytes()).hexdigest()[:12] == "e91ee8a8a846"
import readout_rules as RR   # noqa: E402  (area, finished: the one definition set, F86's)

SIZES = ("53m", "190m", "505m")
NEW = ("53m", "190m")
ARMS = ("none", "win-ctx")
SEEDS = tuple(range(8))
H1, H2 = (0, 1, 2, 3), (4, 5, 6, 7)
N_NONEMB = {"53m": 14165376, "190m": 113283840, "505m": 402752512}   # n_params - 100278 x d_model (make_jobs_pttpp.SIZES)
PICKS = {"53m": {"none": "8e-3", "win-ctx": "2e-3"}, "190m": {"none": "4.8e-3", "win-ctx": "6e-4"},
         "505m": {"none": "4.8e-3", "win-ctx": "6e-4"}}
# F86 R02 addendum: the Burst 8-seed mean area of the same cells (readout_rules.area on the Burst run JSONs)
F86 = {("53m", "none"): 18.43, ("53m", "win-ctx"): 69.91, ("190m", "none"): 29.06, ("190m", "win-ctx"): 78.45}
RANGE = 15.0
N_BOOT, CI = 10000, 0.90
BURST_TASK = {}   # (size, arm, seed) -> Burst task name (make_pt2main_tasks.tasks(): pt2b 350-353 / 370-373, pt2s190 ...)
for _s in range(4):
    BURST_TASK.update({("53m", "none", _s): f"{350 + _s}-pt2b-lr8e-3-none-s{_s}", ("53m", "win-ctx", _s): f"{350 + _s}-pt2b-lr2e-3-win-ctx-s{_s}",
                       ("190m", "none", _s): f"{366 + _s}-pt2s190-lr4.8e-3-none-s{_s}",
                       ("190m", "win-ctx", _s): f"{358 + _s}-pt2s190-lr6e-4-win-ctx-s{_s}"})
for _s in range(4, 8):
    BURST_TASK.update({("53m", "none", _s): f"{366 + _s}-pt2b-lr8e-3-none-s{_s}", ("53m", "win-ctx", _s): f"{366 + _s}-pt2b-lr2e-3-win-ctx-s{_s}",
                       ("190m", "none", _s): f"{378 + _s}-pt2s190-lr4.8e-3-none-s{_s}",
                       ("190m", "win-ctx", _s): f"{378 + _s}-pt2s190-lr6e-4-win-ctx-s{_s}"})
ROOTS = {"torch": B / "runs" / "pt2same" / "out", "s505": B / "runs" / "pt2s505" / "out",
         "burst": Path("/scratch/yh6210/llmadd-burst-ro/results")}


def run_paths(site, size, arm, seed, roots):
    """(run JSON, check.json) of one run; site 'torch' = this batch (53M / 190M) or pt2s505 (505M), 'burst' = F86's runs"""
    lr = PICKS[size][arm]
    target = "none" if arm == "none" else "helix3"
    if site == "torch" and size == "505m":
        tag, d = f"pt2s505-lr{lr}-{arm}", roots["s505"] / f"pt2s505-lr{lr}-{arm}-s{seed}"
    elif site == "torch":
        tag = f"pt2same-{size}-lr{lr}-{arm}"
        d = roots["torch"] / f"{tag}-s{seed}"
    else:
        assert size in NEW
        name = BURST_TASK[(size, arm, seed)]
        tag, d = name[4:name.rindex("-s")], roots["burst"] / name
    return d / tag / f"run_whole_{target}_seed{seed}_{tag}.json", d / "check.json"


def load_run(run, check):
    """-> (status, area, info): status 'ok' / 'incomplete' / 'WITHHELD'"""
    if not run.exists():
        return "incomplete", None, "no run JSON"
    r = json.loads(run.read_text())
    if not RR.finished(r):
        return "incomplete", None, f"unfinished (last eval {r['curve'][-1]['step']} of {r['steps']})"
    if not check.exists():
        return "incomplete", None, "no check.json"
    c = json.loads(check.read_text())
    if not c.get("pass"):
        return "WITHHELD", None, "check.json failed: " + "; ".join(k["check"] for k in c.get("checks", []) if not k["ok"])
    a = float(RR.area(r))
    if abs(a - c["info"]["area"]) > 0.01:
        return "WITHHELD", None, f"double path: area {a:.3f} != check.json {c['info']['area']}"
    return "ok", a, {"lm0": r["curve"][0]["lm"], "n_evals": len(r["curve"])}


def load(roots):
    """{(site, size, arm): {"area": [8 values or None], "status": [...], "why": [...], "lm0": [...]}}"""
    out = {}
    for site, sizes in (("torch", SIZES), ("burst", NEW)):
        for size in sizes:
            for arm in ARMS:
                cell = {"area": [], "status": [], "why": [], "lm0": []}
                for s in SEEDS:
                    st, a, info = load_run(*run_paths(site, size, arm, s, roots))
                    cell["area"].append(a)
                    cell["status"].append(st)
                    cell["why"].append(info if st != "ok" else "")
                    cell["lm0"].append(info["lm0"] if st == "ok" else None)
                out[(site, size, arm)] = cell
    return out


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def boot_idx(n=8):
    """the joint seed resamples (one per replicate, shared by every size and both arms)"""
    return np.random.default_rng(0).integers(0, n, size=(N_BOOT, n))


def pct_ci(v):
    lo, hi = np.percentile(v, [100 * (1 - CI) / 2, 100 * (1 + CI) / 2])
    return float(lo), float(hi)


def slope_table(Y, sizes):
    """Y: {size: array of 8 per-seed values}. -> point slopes (0-7, 0-3, 4-7) and the 8-seed bootstrap CI"""
    x = [math.log(N_NONEMB[z]) for z in sizes]
    pt = {k: ols_slope(x, [np.mean([Y[z][s] for s in ss]) for z in sizes]) for k, ss in (("all", SEEDS), ("h1", H1), ("h2", H2))}
    idx = boot_idx()
    M = np.stack([np.asarray(Y[z], float)[idx].mean(axis=1) for z in sizes], axis=1)   # (N_BOOT, n sizes)
    xc = np.asarray(x) - np.mean(x)
    bs = (M - M.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
    pt["ci"] = pct_ci(bs)
    return pt


def verdict(sl):
    """the pre-registered decision rule on a slope_table() result"""
    lo, hi = sl["ci"]
    excl = lo > 0 or hi < 0
    sg = [np.sign(sl[k]) for k in ("all", "h1", "h2")]
    same = sg[0] != 0 and sg[0] == sg[1] == sg[2]
    if excl and same:
        return ("SHOWN (+: the prior's gain is larger at larger N)" if sl["all"] > 0 else
                "SHOWN (-: the prior's gain is smaller at larger N)")
    why = []
    if not excl:
        why.append("the 8-seed 90% CI includes 0")
    if not same and sg[0] == 0:
        why.append("the 8-seed slope is exactly 0")
    elif not same:
        why.append(f"the seed halves disagree in sign (0-3 {sl['h1']:+.2f}, 4-7 {sl['h2']:+.2f}, 0-7 {sl['all']:+.2f})")
    return "no size trend shown (" + "; ".join(why) + ")"


def gain_row(Y):
    d = np.asarray(Y, float)
    idx = boot_idx()
    return {"h1": float(d[list(H1)].mean()), "h2": float(d[list(H2)].mean()), "all": float(d.mean()),
            "wins": int((d > 0).sum()), "ci": pct_ci(d[idx].mean(axis=1))}


def read(roots, investigated=None, js=None, data=None):
    D = data if data is not None else load(roots)
    res = {"integrity": {}, "range": {}, "verdict": None}
    print("== integrity (finished, check.json passing, area double path)")
    ok = True
    for (site, size, arm), c in D.items():
        bad = [(s, st, w) for s, (st, w) in enumerate(zip(c["status"], c["why"])) if st != "ok"]
        res["integrity"][f"{site}|{size}|{arm}"] = "ok" if not bad else bad
        if bad:
            ok &= site == "burst"   # Burst runs are descriptive only: their gaps never block the verdict
            for s, st, w in bad:
                print(f"  {st:10s} {site} {size} {arm} seed {s}: {w}")
    if ok:
        print("  every counted Torch run (53M, 190M, 505M x 2 arms x 8 seeds) present, finished and passing")
    else:
        res["verdict"] = "no verdict: a counted Torch run is incomplete or WITHHELD"
        print("VERDICT:", res["verdict"])
        _dump(js, res)
        return res
    print("== expected range (new Torch cells vs F86's Burst 8-seed mean, +-%.0f)" % RANGE)
    out_of_range = []
    for size in NEW:
        for arm in ARMS:
            m = float(np.mean(D[("torch", size, arm)]["area"]))
            flag = abs(m - F86[(size, arm)]) > RANGE
            res["range"][f"{size}|{arm}"] = {"torch_mean": m, "f86": F86[(size, arm)], "out": flag}
            print(f"  {size:5s} {arm:8s} Torch 8-seed mean {m:6.2f}   F86 {F86[(size, arm)]:6.2f}   "
                  f"{'OUT OF RANGE' if flag else 'in range'}")
            if flag:
                out_of_range.append(f"{size} {arm}")
    if out_of_range and not investigated:
        res["verdict"] = "PENDING: out of the expected range (" + ", ".join(out_of_range) + "): investigate the cell first"
        print("VERDICT:", res["verdict"])
        _dump(js, res)
        return res
    if out_of_range:
        print(f"  investigated (--investigated): {investigated}")
        res["investigated"] = investigated
    A = {(z, a): np.asarray(D[("torch", z, a)]["area"], float) for z in SIZES for a in ARMS}
    G = {z: A[(z, "win-ctx")] - A[(z, "none")] for z in SIZES}
    print("== per size, Torch A100-80GB (area = 100 x mean over all evals of the 6 held-out buckets; Delta = win-ctx - none, "
          "paired by seed)")
    print(f"  {'size':5s} {'N_nonemb':>12s} {'none 0-7':>9s} {'prior 0-7':>9s} {'D 0-3':>7s} {'D 4-7':>7s} {'D 0-7':>7s} "
          f"{'90% CI':>16s} {'wins':>5s}")
    res["sizes"] = {}
    for z in SIZES:
        g = gain_row(G[z])
        res["sizes"][z] = {"none": float(A[(z, "none")].mean()), "prior": float(A[(z, "win-ctx")].mean()), **g,
                           "delta_per_seed": G[z].tolist()}
        print(f"  {z:5s} {N_NONEMB[z]:12,d} {A[(z, 'none')].mean():9.2f} {A[(z, 'win-ctx')].mean():9.2f} {g['h1']:+7.2f} "
              f"{g['h2']:+7.2f} {g['all']:+7.2f} [{g['ci'][0]:+6.2f}, {g['ci'][1]:+6.2f}] {g['wins']:3d}/8")
    sl = slope_table(G, SIZES)
    v = verdict(sl)
    res["slope"] = sl
    res["verdict"] = v
    print("== the 2% size comparison on one cluster: slope of the 8-seed Delta vs ln N_nonemb (53M, 190M, 505M; Torch)")
    print(f"  slope 0-7 {sl['all']:+.3f} area points per ln N, 90% CI [{sl['ci'][0]:+.3f}, {sl['ci'][1]:+.3f}] "
          f"(joint paired seed bootstrap, {N_BOOT}); seeds 0-3 {sl['h1']:+.3f}, seeds 4-7 {sl['h2']:+.3f}")
    print("VERDICT:", v)
    print("== descriptive")
    for a in ARMS:
        s_ = slope_table({z: A[(z, a)] for z in SIZES}, SIZES)
        res[f"slope_area_{a}"] = s_
        print(f"  area slope {a:8s}: {s_['all']:+.3f} [{s_['ci'][0]:+.3f}, {s_['ci'][1]:+.3f}] (0-3 {s_['h1']:+.3f}, 4-7 {s_['h2']:+.3f})")
    bok = all(st == "ok" for z in NEW for a in ARMS for st in D[("burst", z, a)]["status"])
    if bok:
        res["site"] = {}
        print("  site effect, same cells and seeds (Torch A100-80GB - Burst A100-40GB; mean, per-seed range, wins):")
        for z in NEW:
            for a in ARMS + ("Delta",):
                t = G[z] if a == "Delta" else A[(z, a)]
                bu = (np.asarray(D[("burst", z, "win-ctx")]["area"]) - np.asarray(D[("burst", z, "none")]["area"]) if a == "Delta"
                      else np.asarray(D[("burst", z, a)]["area"], float))
                d = t - bu
                res["site"][f"{z}|{a}"] = {"torch": float(t.mean()), "burst": float(bu.mean()), "diff": float(d.mean()),
                                           "ci": pct_ci(d[boot_idx()].mean(axis=1))}
                print(f"    {z:5s} {a:8s} Torch {t.mean():6.2f}  Burst {bu.mean():6.2f}  diff {d.mean():+6.2f} "
                      f"[{d.min():+6.2f} .. {d.max():+6.2f}] 90% CI [{res['site'][f'{z}|{a}']['ci'][0]:+.2f}, "
                      f"{res['site'][f'{z}|{a}']['ci'][1]:+.2f}]  Torch > Burst {int((d > 0).sum())}/8")
            l0 = [abs(x - y) for x, y in zip(D[("torch", z, "none")]["lm0"], D[("burst", z, "none")]["lm0"])]
            print(f"    {z:5s} step-0 lm |Torch - Burst| max {max(l0):.1e} (same init and first batch)")
        Gm = {"53m": np.asarray(D[("burst", "53m", "win-ctx")]["area"]) - np.asarray(D[("burst", "53m", "none")]["area"]),
              "190m": np.asarray(D[("burst", "190m", "win-ctx")]["area"]) - np.asarray(D[("burst", "190m", "none")]["area"]),
              "505m": G["505m"]}
        sm = slope_table(Gm, SIZES)
        res["slope_mixed_sites"] = sm
        print(f"  mixed-site slope (Burst 53M / 190M + Torch 505M; the comparison this batch replaces): {sm['all']:+.3f} "
              f"[{sm['ci'][0]:+.3f}, {sm['ci'][1]:+.3f}] (0-3 {sm['h1']:+.3f}, 4-7 {sm['h2']:+.3f}); not a verdict")
    else:
        print("  Burst runs not all readable: site effect not printed")
    _dump(js, res)
    return res


def _dump(js, res):
    if js:
        Path(js).write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------- self-test (synthetic data only)
def _fake_tree(root, areas, fail=(), missing=()):
    """write run JSONs + check.json files with the layout run_paths() reads; area of a run = 100 x a constant curve value.
    areas: {(site, size, arm): [8 areas]}"""
    roots = {"torch": root / "torch", "s505": root / "s505", "burst": root / "burst"}
    for (site, size, arm), vals in areas.items():
        for s, v in enumerate(vals):
            if (site, size, arm, s) in missing:
                continue
            run, chk = run_paths(site, size, arm, s, roots)
            run.parent.mkdir(parents=True, exist_ok=True)
            e = lambda st: {"step": st, "lm": 11.6, **{f"held_{b}": v / 100 for b in RR.B}}
            r = {"steps": 18310, "args": {"bs": 32, "seq": 512}, "curve": [e(st) for st in (0, 250, 18309)]}
            run.write_text(json.dumps(r))
            ok = (site, size, arm, s) not in fail
            chk.write_text(json.dumps({"pass": ok, "checks": [{"check": "fake", "ok": ok, "value": 0}], "info": {"area": round(v, 2)}}))
    return roots


def selftest():
    x = {z: math.log(N_NONEMB[z]) for z in SIZES}
    xm = np.mean(list(x.values()))
    rs = np.random.default_rng(123)
    n_ok = 0

    def case(name, G, want):
        nonlocal n_ok
        sl = slope_table(G, SIZES)
        v = verdict(sl)
        good = v.startswith(want)
        n_ok += good
        print(f"{'PASS' if good else 'FAIL'} {name}: slope {sl['all']:+.3f} CI [{sl['ci'][0]:+.3f}, {sl['ci'][1]:+.3f}] "
              f"0-3 {sl['h1']:+.3f} 4-7 {sl['h2']:+.3f} -> {v}")
        return sl
    # 1. a known positive slope: Delta = 50 + 6 (ln N - mean) + noise (sd 1) -> SHOWN (+)
    sl = case("known positive slope (+6 / ln N)", {z: 50 + 6 * (x[z] - xm) + rs.normal(0, 1, 8) for z in SIZES}, "SHOWN (+")
    assert abs(sl["all"] - 6) < 1, sl
    # 2. a known negative slope -> SHOWN (-)
    case("known negative slope (-4 / ln N)", {z: 50 - 4 * (x[z] - xm) + rs.normal(0, 1, 8) for z in SIZES}, "SHOWN (-")
    # 3. a known zero slope (every size mean exactly 50; the seeds scatter in opposite directions) -> no size trend shown
    d = np.array([5, -5, 3, -3, 4, -4, 2, -2], float)
    sl = case("known zero slope (exact 0, seeds scatter)", {"53m": 50 + d, "190m": 50 - d, "505m": 50 + 0.5 * d}, "no size trend shown")
    assert abs(sl["all"]) < 1e-12 and sl["ci"][0] < 0 < sl["ci"][1], sl
    # 4. a flat noisy case (sd 8, no trend): the realisation is checked to straddle 0 before asserting the rule
    G4 = {z: 50 + rs.normal(0, 8, 8) for z in SIZES}
    if slope_table(G4, SIZES)["ci"][0] < 0 < slope_table(G4, SIZES)["ci"][1]:
        case("flat noisy data (sd 8)", G4, "no size trend shown")
    else:
        n_ok += 1
        print("SKIP flat noisy data: this realisation's CI excludes 0 (a 10% event)")
    # 5. the halves disagree although the 8-seed CI excludes 0: seeds 0-3 +10 / ln N, seeds 4-7 -1 / ln N
    G5 = {z: np.array([50 + (10 if s < 4 else -1) * (x[z] - xm) + rs.normal(0, 0.3) for s in SEEDS]) for z in SIZES}
    sl = case("halves disagree (0-3: +10, 4-7: -1 per ln N)", G5, "no size trend shown")
    assert (sl["ci"][0] > 0 or sl["ci"][1] < 0) and "halves disagree" in verdict(sl), ("the halves rule did not decide", sl)
    # 6. the bootstrap is joint: a seed effect common to every size cancels in the slope (CI collapses to the point)
    G6 = {z: 40 + 3 * (x[z] - xm) + d for z in SIZES}
    sl = case("seed effect shared by every size (joint resampling)", G6, "SHOWN (+")
    assert abs(sl["ci"][1] - sl["ci"][0]) < 1e-9 and abs(sl["all"] - 3) < 1e-9, sl
    # 7. file layer on a fake tree: the full read-out, integrity, range rule, double path
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        base = {}
        for z in SIZES:
            for a in ARMS:
                base[("torch", z, a)] = [F86.get((z, a), 25.0 if a == "none" else 77.0) + 0.5 * k for k in range(8)]
                if z in NEW:
                    base[("burst", z, a)] = [F86[(z, a)] - 0.25 * k for k in range(8)]
        roots = _fake_tree(t / "a", base)
        r = read(roots)
        good = r["verdict"] is not None and not r["verdict"].startswith(("no verdict", "PENDING"))
        good &= abs(r["sizes"]["53m"]["all"] - (69.91 - 18.43)) < 1e-6 and abs(r["sizes"]["190m"]["all"] - (78.45 - 29.06)) < 1e-6
        good &= abs(r["site"]["53m|none"]["diff"] - np.mean([0.75 * k for k in range(8)])) < 1e-9
        print(f"{'PASS' if good else 'FAIL'} file layer: complete fake tree -> {r['verdict']}")
        n_ok += good
        roots = _fake_tree(t / "b", base, fail={("torch", "190m", "none", 3)})
        r = read(roots)
        good = r["verdict"].startswith("no verdict")
        print(f"{'PASS' if good else 'FAIL'} file layer: one failed check.json -> {r['verdict']}")
        n_ok += good
        roots = _fake_tree(t / "c", base, missing={("torch", "505m", "win-ctx", 6)})
        r = read(roots)
        good = r["verdict"].startswith("no verdict")
        print(f"{'PASS' if good else 'FAIL'} file layer: one missing run -> {r['verdict']}")
        n_ok += good
        oor = dict(base)
        oor[("torch", "53m", "none")] = [40.0] * 8          # F86 18.43 + 21.6: out of range
        roots = _fake_tree(t / "d", oor)
        r = read(roots)
        good = r["verdict"].startswith("PENDING") and "slope" not in r
        r2 = read(roots, investigated="synthetic test")
        good &= not r2["verdict"].startswith(("PENDING", "no verdict")) and r2.get("investigated") == "synthetic test"
        print(f"{'PASS' if good else 'FAIL'} file layer: a cell out of range -> {r['verdict']}; with --investigated -> {r2['verdict']}")
        n_ok += good
        roots = _fake_tree(t / "e", base)
        run, chk = run_paths("torch", "53m", "none", 2, roots)
        c = json.loads(chk.read_text()); c["info"]["area"] += 1.0; chk.write_text(json.dumps(c))
        r = read(roots)
        good = r["verdict"].startswith("no verdict")
        print(f"{'PASS' if good else 'FAIL'} file layer: area double-path mismatch -> {r['verdict']}")
        n_ok += good
    total = 11
    print(f"SELFTEST {'PASS' if n_ok == total else 'FAIL'} ({n_ok}/{total})")
    return n_ok == total


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    inv = sys.argv[sys.argv.index("--investigated") + 1] if "--investigated" in sys.argv else None
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    r = read(ROOTS, investigated=inv, js=js)
    sys.exit(0)
