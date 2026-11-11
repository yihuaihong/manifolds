"""Amendment A2 completion check (2026-11-04, after a session restart; a second path to the A2 tests, with its own
constructions): item by item against AUDIT.md A2.1-A2.3 and tests/audit_a1.md. Nothing is written into runs/ptdom (temp
roots, a temp --hold path); nothing is submitted.
  G  the gate statistic per (arm, lr) cell: a / (a + b) from hand-written gradient fields (incl. cos > 0, where A1's
     a / |g_T| passed), the boundary b = a, the median of run medians, each of the 5 fields missing at ONE eval, a NaN
     grad_prior, no on-window eval (prior never on; diverged before the window), the logged-share double path at 2e-3 /
     5e-4, off-window and post-divergence evals ignored
  S  select_ptdom.py: two cells of an arm fail (one admissible lr), every cell fails (AUX GATE FAILED, HOLD at the temp path,
     exit 2, the picks file still written with GATE units and the other units' decisions), the real HOLD never written
  P  the read-out at the pick over seeds 0-7: a confirm seed with a missing field / confirm seeds at 0.2 -> 'AUX GATE FAILED
     at the pick', the domain kept in the Holm family with its t p; the other domains' verdicts
  X  the A2 extension trigger through decide(): none's lower-edge argmax within 2 -> X 5e-4, then STOP only by the margin;
     a prior arm's / none's other edge argmax within 2 -> closed; the margin -> X -> closed
  W  aux_gate_check.py per cell (not pooled), a missing clip_coef, pre-window runs listed only
  M  minor items: pair_diff with divergences, shortcut_rule on final3, CHECK_PY's max, the mislabelled logs
  J  job files: the 92 stage-A files = the generator's in-memory rendering byte for byte; render(rr=1) names
"""
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REAL = Path("/scratch/yh6210/llmadd_torch/runs/ptdom")
# PTDOM_R: the dir holding the A2 code under test (default runs/ptdom). After Amendment A3 replaced runs/ptdom's code
# (2026-11-04 04:39-04:55 EST), the A2 versions live in runs/ptdom/superseded_A2; a2_verify_run.sh runs this script on a
# mirror of them (copies of readout / select / aux_gate_check + the 92 job files, code -> superseded_A2/code_revE,
# PTDOM_MK = the dir of the A2 make_jobs_ptdom.py).
R = Path(os.environ.get("PTDOM_R", str(REAL)))
B = Path(os.environ.get("PTDOM_MK", "/scratch/yh6210/llmadd_torch"))
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
sys.path.insert(0, str(R))
import readout_ptdom as RO   # noqa: E402

ok = []


def T(name, cond, info=None):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if info is None else f"  ({info})"))


def edit(root, arm, lr, seed, fn, rr=0):
    """Rewrite the curve of a fake run: fn(i, e) edits eval i in place."""
    d = Path(root) / (f"ptdom-53m-k2-{arm}-lr{lr}-s{seed}" + (f"-r{rr}" if rr else ""))
    f = RO.run_json(d, arm, lr, seed)
    r = json.loads(f.read_text())
    for i, e in enumerate(r["curve"]):
        fn(i, e)
    f.write_text(json.dumps(r))


def set_ab(a, b, cs, off=0.0):
    """On-window evals get a = grad_lm, b = grad_prior, cos; the logged share = a / |g_lm + g_prior| (+ off)."""
    def fn(i, e):
        if e["prior_on"]:
            e.update(grad_lm=a, grad_prior=b, grad_cos=cs, grad_lm_share=a / math.sqrt(a * a + b * b + 2 * a * b * cs) + off)
    return fn


def flat(arm, lr, s):
    return {d: 50.0 for d in RO.SIX}


def stageA(root, area=flat, seeds=RO.SEL, cells=None):
    for arm in RO.ARMS:
        for lr in RO.GRID[arm]:
            if cells is None or (arm, lr) in cells:
                for s in seeds:
                    RO.fake_run(root, arm, lr, s, area(arm, lr, s))


def gate_of(root, arm, lr, seeds=RO.SEL):
    R_ = RO.load(root)
    return RO.cell_gate([r for k, r in R_.items() if k[:2] == (arm, lr) and k[2] in seeds and r.get("finished")])


def run(cmd):
    o = subprocess.run(cmd, capture_output=True, text=True)
    return o.returncode, o.stdout + o.stderr


hold_real = (REAL / "HOLD").exists()
print(f"code under test: {R} (readout_ptdom {Path(RO.__file__).resolve()}, readout_rules {Path(RO.RR.__file__).resolve()})")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # ------------------------------------------------------------------ G: the statistic per cell
    g = tmp / "g"
    for s in RO.SEL:
        RO.fake_run(g, "only-stars", "1e-3", s, flat(0, 0, 0))
        edit(g, "only-stars", "1e-3", s, set_ab(2.0, 2.6, 0.5))   # b = 1.3 a, cos +0.5
    x = gate_of(g, "only-stars", "1e-3")
    a1 = 2.0 / math.sqrt(4 + 2.6 ** 2 + 2 * 2.0 * 2.6 * 0.5)
    T(f"G1 a = 2, b = 2.6, cos +0.5: cell median {x['median']:.4f} = 2 / 4.6 = {2 / 4.6:.4f} -> FAILS (A1's a / |g_T| {a1:.4f} >= 0.5 "
      f"would pass); b > a on {x['frac_b_gt_a']:.0%} of {x['n_on']} on-window evals",
      not x["ok"] and abs(x["median"] - 2 / 4.6) < 1e-12 and a1 >= 0.5 and x["frac_b_gt_a"] == 1.0 and x["n_on"] == 4 * 26)
    for s in RO.SEL:
        edit(g, "only-stars", "1e-3", s, set_ab(1.7, 1.7, 0.3))
    x = gate_of(g, "only-stars", "1e-3")
    T("G2 the boundary b = a (cos 0.3): median 0.5 exactly -> passes", x["ok"] and x["median"] == 0.5, x["median"])
    # run medians: s0-s2 13 evals at 0.3 + 13 at 0.95 (run median 0.625); s3 26 at 0.3 -> cell 0.625 (passes), while the
    # pooled median over the 104 evals (65 at 0.3) would be 0.3: the rule is the median of run medians
    def mix(lo_n, lo, hi):
        def fn(i, e):
            if e["prior_on"]:
                sh = lo if (i - 4) < lo_n else hi
                b = 1.0 / sh - 1.0
                e.update(grad_lm=1.0, grad_prior=b, grad_cos=0.0, grad_lm_share=1.0 / math.sqrt(1 + b * b))
        return fn
    for s in (0, 1, 2):
        edit(g, "only-stars", "1e-3", s, mix(13, 0.3, 0.95))
    edit(g, "only-stars", "1e-3", 3, mix(26, 0.3, 0.3))
    x = gate_of(g, "only-stars", "1e-3")
    want = float(np.median([(0.3 + 0.95) / 2] * 3 + [0.3]))
    pooled = float(np.median([0.3] * 65 + [0.95] * 39))
    T(f"G3 per run the median over its on-window evals, per cell the median of run medians: {x['median']:.4f} = {want:.4f} "
      f"-> passes (the pooled median over the 104 evals would be {pooled:.4f}); per-run {x['per_run']}",
      x["ok"] and abs(x["median"] - want) < 1e-9 and abs(x["per_run"][3] - 0.3) < 1e-9 and pooled < 0.5)
    for fld in RO.GRAD_FIELDS:
        root = tmp / f"g_{fld}"
        for s in RO.SEL:
            RO.fake_run(root, "only-stars", "1e-3", s, flat(0, 0, 0), lm_share=0.95)
        edit(root, "only-stars", "1e-3", 2, lambda i, e, fld=fld: e.pop(fld) if i == 17 else None)
        x = gate_of(root, "only-stars", "1e-3")
        T(f"G4 {fld} missing at ONE on-window eval of ONE run (share 0.95 elsewhere) -> the cell fails", not x["ok"]
          and any(fld in w and "1 of 26" in w for w in x["why"]), x["why"])
    root = tmp / "g_nan"
    for s in RO.SEL:
        RO.fake_run(root, "only-stars", "1e-3", s, flat(0, 0, 0), lm_share=0.95)
    edit(root, "only-stars", "1e-3", 0, lambda i, e: e.update(grad_prior=float("nan")) if i == 10 else None)
    x = gate_of(root, "only-stars", "1e-3")
    T("G5 a NaN grad_prior at one on-window eval (lm finite: not a divergence) -> not finite -> the cell fails", not x["ok"], x["why"])
    root = tmp / "g_noon"
    for s in RO.SEL:
        RO.fake_run(root, "only-stars", "1e-3", s, flat(0, 0, 0), lm_share=0.95)
    edit(root, "only-stars", "1e-3", 1, lambda i, e: e.update(prior_on=0))
    RO.fake_run(root, "only-stars", "2e-3", 0, flat(0, 0, 0), lm_share=0.95, diverge_at=2)
    for s in (1, 2, 3):
        RO.fake_run(root, "only-stars", "2e-3", s, flat(0, 0, 0), lm_share=0.95)
    x1, x2 = gate_of(root, "only-stars", "1e-3"), gate_of(root, "only-stars", "2e-3")
    T("G6 a finished run with no on-window eval (prior never on; or diverged at eval 2, before the window) -> unverifiable -> "
      "its cell fails", not x1["ok"] and not x2["ok"] and any("no on-window eval" in w for w in x1["why"])
      and any("no on-window eval" in w for w in x2["why"]), [x1["why"], x2["why"]])
    root = tmp / "g_dp"
    for s in RO.SEL:
        RO.fake_run(root, "only-stars", "1e-3", s, flat(0, 0, 0), lm_share=0.95, logged_off=2e-3 if s == 3 else 0.0)
        RO.fake_run(root, "only-stars", "2e-3", s, flat(0, 0, 0), lm_share=0.95, logged_off=5e-4 if s == 3 else 0.0)
    x1, x2 = gate_of(root, "only-stars", "1e-3"), gate_of(root, "only-stars", "2e-3")
    T("G7 logged grad_lm_share off its re-derivation by 2e-3 in one run -> fails; by 5e-4 -> passes (tolerance 1e-3)",
      not x1["ok"] and x2["ok"], [x1["dp_maxdiff"], x2["dp_maxdiff"]])
    root = tmp / "g_off"
    for s in RO.SEL:
        RO.fake_run(root, "only-stars", "1e-3", s, flat(0, 0, 0), lm_share=0.95, diverge_at=20)
        # off-window evals with b = 9 a (prior_on 0) and post-divergence on-window evals with b = 9 a: both ignored
        edit(root, "only-stars", "1e-3", s, lambda i, e: e.update(grad_lm=1.0, grad_prior=9.0, grad_cos=0.0,
                                                                   grad_lm_share=1 / math.sqrt(82), clip_coef=0.1)
             if (not e["prior_on"] and i > 0) or (e["prior_on"] and i >= 20) else None)
    x = gate_of(root, "only-stars", "1e-3")
    T(f"G8 off-window evals and on-window evals after a divergence (b = 9 a) are not counted: median {x['median']:.3f}, "
      f"{x['n_on'] // 4} on-window evals per run (4 .. 19)", x["ok"] and abs(x["median"] - 0.95) < 1e-9 and x["n_on"] == 4 * 16)

    # ------------------------------------------------------------------ S: select_ptdom.py with the gate
    def peak(arm, lr, s):   # prior arms peak inside at 1e-3 on their own domain, none inside at 4e-3: every unit closed
        if arm == "none":
            v = {"1e-3": 20, "2e-3": 25, "4e-3": 30, "8e-3": 25, "1.6e-2": 20}.get(lr, 10)
            return {d: v + 0.2 * s for d in RO.SIX}
        v = {"5e-4": 40, "1e-3": 50, "2e-3": 40}.get(lr, 30)
        return {d: (v + 0.3 * (s % 3) if d == arm[5:] else 35.0) for d in RO.SIX}
    s1 = tmp / "s1"
    stageA(s1, peak)
    for s in RO.SEL:
        for lr in ("1e-3", "2e-3"):
            edit(s1, "only-cities", lr, s, set_ab(1.0, 1.2, 0.0))   # 0.4545
    hold = tmp / "HOLD_s1"
    rc, out = run([PY, str(R / "select_ptdom.py"), "--stage", "A", "--root", str(s1), "--out", str(tmp / "pk_s1.json"), "--hold", str(hold)])
    pk = json.loads((tmp / "pk_s1.json").read_text()) if (tmp / "pk_s1.json").exists() else {}
    u = pk.get("units", {}).get("only-cities|cities", {})
    T("S1 only-cities@1e-3 (the area argmax, 50) and @2e-3 fail -> excluded; pick 5e-4, the only admissible lr (no bracket: "
      "closed) -> C; exit 0, no HOLD; failing cells listed in the picks file",
      rc == 0 and u.get("pick") == "5e-4" and u.get("decision") == "C" and u.get("excluded") == ["1e-3", "2e-3"]
      and pk.get("aux_gate_failed_cells") == ["only-cities@1e-3", "only-cities@2e-3"] and not hold.exists()
      and ["only-cities", "5e-4"] in pk.get("C", []), (rc, u.get("pick"), u.get("decision"), u.get("excluded")))
    s2 = tmp / "s2"
    stageA(s2, peak)
    for s in RO.SEL:
        for lr in RO.GRID["only-languages"]:
            edit(s2, "only-languages", lr, s, set_ab(1.0, 1.0001, 0.0))   # just below 0.5
    hold = tmp / "HOLD_s2"
    rc, out = run([PY, str(R / "select_ptdom.py"), "--stage", "A", "--root", str(s2), "--out", str(tmp / "pk_s2.json"), "--hold", str(hold)])
    pk = json.loads((tmp / "pk_s2.json").read_text()) if (tmp / "pk_s2.json").exists() else {}
    U = pk.get("units", {})
    T("S2 every only-languages cell at b = 1.0001 a -> 'AUX GATE FAILED', exit 2, the --hold file names only-languages, the picks "
      "file IS written: only-languages|languages and none|languages GATE (no pick, no X / C), the other 10 units C; none@4e-3 "
      "still confirmed (other domains)",
      rc == 2 and "AUX GATE FAILED" in out and hold.exists() and "only-languages" in hold.read_text()
      and "only-stars" not in hold.read_text() and pk.get("GATE") == ["none|languages", "only-languages|languages"]
      and U["only-languages|languages"]["pick"] is None and pk.get("hold_written") is True
      and all(U[k]["decision"] == "C" for k in U if "languages" not in k) and ["none", "4e-3"] in pk.get("C", [])
      and not any(c[0] == "only-languages" for c in pk.get("C", [])) and pk.get("X") == [], (rc, pk.get("GATE"), out[-200:]))

    # ------------------------------------------------------------------ P: the read-out at the pick, seeds 0-7
    def chainC(root, edit_fn=None):
        stageA(root, peak)
        R_ = RO.load(root)
        pa = RO.decide(R_)
        for arm, lr in pa["C"]:
            for s in RO.CONF:
                RO.fake_run(root, arm, lr, s, peak(arm, lr, s))
        if edit_fn:
            edit_fn(root)
        R_ = RO.load(root)
        return R_, pa
    R0, pa0 = chainC(tmp / "p0")
    o0 = RO.readout(R0, subs=RO.plant_subs(R0))
    T("P0 control: every unit C after A, every pick passes the gate over seeds 0-7, every domain PASS (gain 20 + noise)",
      pa0["X"] == [] and all(o0["domains"][d]["verdict"] in ("PASS", "NOT SEPARABLE") for d in RO.SIX)
      and all(o0["domains"][d]["pick_gate"]["seeds"] == list(range(8)) for d in RO.SIX),
      {d: o0["domains"][d]["verdict"] for d in RO.SIX})

    def miss_conf(root):   # stars' pick: seed 6 lacks clip_coef at one on-window eval
        edit(root, "only-stars", "1e-3", 6, lambda i, e: e.pop("clip_coef") if i == 12 else None)

    def low_conf(root):    # cities' pick: seeds 0-3 at 0.55 (pass at A), confirm seeds at 0.2 -> 8-seed median 0.375
        for s in RO.CONF:
            edit(root, "only-cities", "1e-3", s, set_ab(1.0, 4.0, 0.0))
    R1, pa1 = chainC(tmp / "p1", lambda root: (miss_conf(root), low_conf(root)))
    for s in RO.SEL:   # seeds 0-3 of the cities pick at 0.55 -> rebuild stage-A decision with them
        edit(tmp / "p1", "only-cities", "1e-3", s, set_ab(1.0, 1 / 0.55 - 1, 0.0))
    R1 = RO.load(tmp / "p1")
    o1 = RO.readout(R1, subs=RO.plant_subs(R1))
    D1 = o1["domains"]
    ps = []
    for d in RO.SIX:
        pp, pn = D1[d]["pick_prior"], D1[d]["pick_none"]
        gg = np.array([R1[(f"only-{d}", pp, s)]["area"][d] - R1[("none", pn, s)]["area"][d] for s in RO.SEL + RO.CONF])
        ps.append(RO.t_test(gg)[1])
    ph = RO.holm(ps)
    T("P1 stars' pick: confirm seed 6 lacks clip_coef at one eval -> 'AUX GATE FAILED at the pick' (no PASS); cities' pick: "
      "0.55 at seeds 0-3 (the selection passes it), 0.2 at seeds 4-7 -> median 0.375 -> 'AUX GATE FAILED at the pick'; both stay "
      "in the Holm family of six with their t p (Holm recomputed = the read-out's); the other four verdicts unchanged",
      RO.decide(R1)["units"]["only-cities|cities"]["decision"] == "C" and RO.aux_gate(R1)["ok"]
      and D1["stars"]["verdict"] == RO.GATE_PICK_FAIL and D1["cities"]["verdict"] == RO.GATE_PICK_FAIL
      and abs(D1["cities"]["pick_gate"]["median"] - 0.375) < 1e-9 and D1["stars"]["t_p"] < 0.05
      and all(abs(D1[d]["t_p_holm"] - ph[i]) < 1e-12 for i, d in enumerate(RO.SIX))
      and all(D1[d]["verdict"] == o0["domains"][d]["verdict"] for d in RO.SIX if d not in ("stars", "cities")),
      {d: (D1[d]["verdict"], round(D1[d]["t_p_holm"], 4)) for d in RO.SIX})
    txt = RO.report(o1)
    T("P1 the report prints the verdict text and the pick-gate FAIL line for stars and cities",
      sum(l.startswith(("stars ", "cities ")) and RO.GATE_PICK_FAIL in l for l in txt.splitlines()) == 2
      and sum("aux gate at the pick (A2 (ii)" in l and "FAIL" in l for l in txt.splitlines()) == 2)

    # ------------------------------------------------------------------ X: the extension trigger through decide()
    def ext_area(arm, lr, s):
        v = 30.0
        if arm == "none":
            st = {"5e-4": 0, "1e-3": 31, "2e-3": 30.5, "4e-3": 29, "8e-3": 28, "1.6e-2": 27}[lr]   # stars: argmax lower edge by 0.5
            ci = {"5e-4": 0, "1e-3": 27, "2e-3": 28, "4e-3": 29, "8e-3": 30.5, "1.6e-2": 31}[lr]   # cities: argmax upper edge by 0.5
            return dict({d: 25 + (lr == "4e-3") * 5 for d in RO.SIX}, stars=st, cities=ci)
        own = {"only-colours": {"5e-4": 41, "1e-3": 40, "2e-3": 35},   # argmax lower edge by 1: closed (A1: X)
               "only-cities": {"5e-4": 30, "1e-3": 35, "2e-3": 38}}     # pick at the upper edge by 3: open -> X 4e-3
        tab = own.get(arm, {"5e-4": 40, "1e-3": 50, "2e-3": 40})
        return {d: (tab.get(lr, v) if d == arm[5:] else 35.0) for d in RO.SIX}
    for case, v5, v4, want in (("stop", 34.0, 37.0, "STOP"), ("close", 32.0, 37.0, "C")):
        root = tmp / f"x_{case}"
        stageA(root, ext_area)
        pa = RO.decide(RO.load(root))
        U = pa["units"]
        a_ok = (U["none|stars"]["decision"], U["none|stars"]["x_lr"]) == ("X", "5e-4") and U["none|cities"]["decision"] == "C" \
            and U["only-colours|colours"]["decision"] == "C" and (U["only-cities|cities"]["decision"], U["only-cities|cities"]["x_lr"]) == ("X", "4e-3") \
            and sorted(map(tuple, pa["X"])) == [("none", "5e-4"), ("only-cities", "4e-3")]
        for s in RO.SEL:
            RO.fake_run(root, "none", "5e-4", s, dict({d: 20.0 for d in RO.SIX}, stars=v5))
            RO.fake_run(root, "only-cities", "4e-3", s, dict({d: 35.0 for d in RO.SIX}, cities=v4))
        pax = RO.decide(RO.load(root), pa)
        V = pax["units"]
        T(f"X {case}: A: none|stars argmax 1e-3 at the lower edge by 0.5 -> X 5e-4 (the none exception); none|cities argmax at "
          f"the UPPER edge by 0.5 -> C; only-colours argmax at its lower edge by 1 -> C (A1 would extend); only-cities pick 2e-3 "
          f"by 3 -> X 4e-3. AX: none@5e-4 = {v5} vs 31 at 1e-3 -> {want} (STOP only by the margin); only-cities 4e-3 = {v4} "
          f"within 2 of 38 -> C",
          a_ok and V["none|stars"]["decision"] == want and V["only-cities|cities"]["decision"] == "C"
          and V["none|cities"]["decision"] == "C" and V["none|cities"].get("stage_final") == "A",
          ({k: U[k]["decision"] for k in ("none|stars", "none|cities", "only-colours|colours", "only-cities|cities")},
           V["none|stars"]["decision"], V["none|stars"]["pick"]))

    # ------------------------------------------------------------------ W: the watcher per cell
    w = tmp / "w"
    for s in RO.SEL:
        for lr in RO.GRID["only-stars"]:
            RO.fake_run(w, "only-stars", lr, s, flat(0, 0, 0), evals=9, lm_share=0.3 if (lr == "2e-3" and s < 3) else 0.9)
        RO.fake_run(w, "only-months", "5e-4", s, flat(0, 0, 0), evals=3)
    rc, out = run([PY, str(R / "aux_gate_check.py"), "--root", str(w)])
    T("W1 partial runs: only-stars@2e-3 has 3 of 4 runs at 0.3 (pooled over the arm's 12 runs: 0.9) -> exit 1, the FAIL line "
      "names only-stars@2e-3 only; the pre-window months runs are listed, not judged",
      rc == 1 and sum(l.startswith("only-") and "FAIL:" in l for l in out.splitlines()) == 1
      and any(l.startswith("only-stars@2e-3") and "FAIL" in l for l in out.splitlines())
      and out.count("before the window (not judged yet): only-months@5e-4") == 4, out[-300:])
    w2 = tmp / "w2"
    for s in RO.SEL:
        RO.fake_run(w2, "only-stars", "1e-3", s, flat(0, 0, 0), evals=9, missing=("clip_coef",) if s == 0 else ())
    rc, out = run([PY, str(R / "aux_gate_check.py"), "--root", str(w2)])
    T("W2 one partial run without clip_coef -> exit 1 (unverifiable)", rc == 1 and "clip_coef" in out, out[-200:])

    # ------------------------------------------------------------------ M: minor items
    m = tmp / "m"
    for s in RO.SEL:
        RO.fake_run(m, "only-stars", "1e-3", s, flat(0, 0, 0), bpb=1.20, diverge_at=40 if s == 1 else None)
        RO.fake_run(m, "none", "1e-3", s, flat(0, 0, 0), bpb=1.21, diverge_at=40 if s == 2 else None)
    RO.fake_run(m, "only-stars", "1e-3", 3, flat(0, 0, 0), bpb=1.20, diverge_at=30)
    RO.fake_run(m, "none", "1e-3", 3, flat(0, 0, 0), bpb=1.21, diverge_at=30)
    Rm = RO.load(m)
    pdo = RO.pair_diff(Rm, ("only-stars", "1e-3"), ("none", "1e-3"), lambda r: 1e3 * r["bpb_off"] if r["bpb_off"] is not None else None)
    T("M1 same-lr LM row: s0 -10, s1 prior diverged +inf, s2 none diverged -inf, s3 both diverged 0 -> n 4 (none dropped), "
      "median -5, wins 1, diverged counts 2 / 2",
      pdo["n"] == 4 and pdo["n_div_a"] == 2 and pdo["n_div_b"] == 2 and abs(pdo["median"] - (-5.0)) < 1e-6 and pdo["wins"] == 1
      and all(Rm[k]["valid"] for k in Rm), pdo)
    T("M2 colours rule on the hard-subset final3: PASS needs final3 > baseline (70.0 vs 69.6 PASS; 69.6 vs 69.6 and None -> "
      "NOT SEPARABLE; a NOT SHOWN stays)",
      RO.shortcut_rule("PASS", 70.0, 69.6) == "PASS" and RO.shortcut_rule("PASS", 69.6, 69.6) == "NOT SEPARABLE"
      and RO.shortcut_rule("PASS", None, 69.6) == "NOT SEPARABLE" and RO.shortcut_rule("NOT SHOWN", 99.0, 1.0) == "NOT SHOWN")
    src = (R / "readout_ptdom.py").read_text()
    T("M2b the read-out feeds the colours rule the prior's hard3 (metrics <d>_readout_hard.final3), not the hard area",
      'hard3={dd: r["metrics"].get(f"{dd}_readout_hard", {}).get("final3")' in src
      and 'shortcut_rule(v, x["hard3_prior"], (x.get("sub") or {}).get("h3_hard"))' in src)
    jobs = sorted(R.glob("job_*.sbatch"))
    T("M3 CHECK_PY in all 92 job files takes the MAX of grad_T_reldiff (the A1 median gone) and logs lm_share_norm",
      len(jobs) == 92 and all("and max(rd) < 0.05," in j.read_text() and "statistics.median(rd) < 0.05" not in j.read_text()
                              and '"lm_share_norm"' in j.read_text() for j in jobs))
    t_ = R / "tests"
    T("M4 the mislabelled logs: a1_refs_full.out ends 'A1 REFS PARTIAL RE-RUN' (17 not re-run), reg_compare_a1_partial.out says "
      "REGRESSION PARTIAL, reg_trainer_a1.out has the pointer; the scripts print STORED CELLS / PARTIAL",
      (t_ / "a1_refs_full.out").read_text().strip().splitlines()[-1].startswith("A1 REFS PARTIAL RE-RUN")
      and "the other 17 were NOT re-run" in (t_ / "a1_refs_full.out").read_text()
      and "REGRESSION PARTIAL" in (t_ / "reg_compare_a1_partial.out").read_text()
      and "REGRESSION PARTIAL" in (t_ / "reg_trainer_a1.out").read_text()
      and "STORED CELLS" in (t_ / "a1_refs_compare.py").read_text() and "PARTIAL (identical where run" in (t_ / "reg_compare.py").read_text())

    # ------------------------------------------------------------------ J: job files = the generator's rendering
    sys.path.insert(0, str(B)); sys.path.insert(0, str(R / "code" / "tools"))
    import make_jobs_ptdom as M   # noqa: E402
    import make_rb6_tasks as P    # noqa: E402
    T(f"J0 the generator under test is the A2 one ({Path(M.__file__).resolve()}; GRID none {M.GRID['none']}; render(rr) exists)",
      M.GRID["none"] == ("1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2") and "rr" in M.render.__code__.co_varnames
      and Path(P.__file__).resolve().parent == (R / "code" / "tools").resolve())
    M.REFS = R / "ptdom_refs.json"
    M.R = R
    commit = M.snapshot()   # the snapshot under test (A2: 21cdc14, verified against git and WANT_SHA)
    M.R = REAL              # the job text embeds runs/ptdom paths (where the A2 files were rendered)
    RF = M.refs()
    same, names = 0, []
    for a, lr, s in M.cells("A"):
        name, tag, text, args, I = M.render(P, a, lr, s, RF[s], commit)
        names.append(f"job_{name}.sbatch")
        same += (R / f"job_{name}.sbatch").read_text() == text
    T(f"J1 the 92 stage-A job files on disk = the generator's in-memory rendering byte for byte ({same}/92); no other job file",
      same == 92 and sorted(names) == [j.name for j in jobs])
    n0, _, t0, a0, I0 = M.render(P, "only-stars", "1e-3", 0, RF[0], commit)
    n1, _, t1, a1_, I1 = M.render(P, "only-stars", "1e-3", 0, RF[0], commit, 1)
    import difflib
    dl = [l for l in difflib.unified_diff(t0.splitlines(), t1.splitlines(), lineterm="", n=0)
          if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    T(f"J2 render(rr=1): name ...-s0-r1, its own OUT, the same --tag and intent; NAME parses -r1; {len(dl)} diff lines, every "
      f"added one names -r1 or the re-run",
      n1 == n0 + "-r1" and I0 == I1 and f"/out/{n1}/" in a1_ and "--tag _ptdom-53m-k2-only-stars-lr1e-3 " in a1_
      and RO.NAME.match(n1).group(4) == "1" and all("-r1" in l or "re-run" in l for l in dl if l[0] == "+")
      and all("-s0" in l for l in dl if l[0] == "-"), dl)

T("the real runs/ptdom/HOLD was not created; no SUBMITTED_ flag, no picks file in runs/ptdom",
  (REAL / "HOLD").exists() == hold_real and not hold_real and not list(REAL.glob("SUBMITTED_*")) and not list(REAL.glob("picks_*"))
  and not (R / "HOLD").exists() and not list(R.glob("picks_*")))
print("A2 VERIFY", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)})")
sys.exit(0 if all(ok) else 1)
