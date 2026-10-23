"""Tests of tools/rb6scale_ext_summary.py (the rb6_190m_ext read-out) before any result of the batch exists, on the 16
real rb6_190m runs + 24 relabelled / edited copies standing in for the new cells (known answers):
  all6-lam1@1.2e-3    <- the real all6-lam1@6e-4 runs (lr / tag relabelled)        -> Q1 rows at 1.2e-3 = rows at 6e-4, PERSISTS
  only-numbers@1.2e-3 <- the real only-numbers@6e-4 runs
  only-clock@6e-4     <- the real all6-lam1@6e-4 runs with the prior table edited to clock only (target none)
  all6-minus-clock    <- the real all6-lam1@6e-4 runs with the clock prior edited off -> f = 1 -> '~ all6'
  none@9.6e-3         <- the 53M Burst none@8e-3 runs (466-469) with the 190M size facts (data are size-independent)
  none@1.92e-2        <- the 53M Burst none@1.6e-2 runs (503 / 505 / 507 / 509) the same way
Scenarios: base (strict, every known answer), missing, wronglr, clockon, failcheck, diverged, datasha; Amendment E1:
q2inconsistent (all6-minus-clock's clock grafted from none@6e-4 at s0, s1 and from only-numbers@6e-4 at s2, s3: the mean
f is past 2/3 but only 2/4 seeds are -> UNDETERMINED; per-seed OVERSHOOT flags), nanother (a NaN in a cell other than
none@1.92e-2 -> the read-out stops), the E1 notes / lm-curve summary printed; unit tests of the pre-registered reading
rules (attribution, q2_reading, range_flag, persistence, floor_label, floor_answer, best_tested); the rb6_190m read-out
re-run byte-identical to the stored output.
    python ro_ext_test.py
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SP = HERE.parent
WT = Path("/scratch/yh6210/llmadd-burst-scale")
RO = Path("/scratch/yh6210/llmadd-burst-ro")
REAL = Path("/scratch/yh6210/llmadd_torch/runs/rb6_190m/out")
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
OUT = HERE / "ro_work"
sys.path.insert(0, str(WT / "tools"))
import rb6scale_ext_summary as X   # noqa: E402
import rainbow6_summary as R6      # noqa: E402

ok_all = True


def say(cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


def runjson(d):
    return next(Path(d).glob("*/run_*.json"))


def real(arm, lr, s):
    return REAL / f"rb6tb-190m-2pct-k2-{arm}-lr{lr}-s{s}"


B53 = {"none@9.6e-3": [f"46{6 + s}-rb6mb-53m-2pct-k2-none-lr8e-3-s{s}" for s in range(4)],
       "none@1.92e-2": [f"{503 + 2 * s}-rb6mb-53m-2pct-k2-none-lr1.6e-2-s{s}" for s in range(4)]}


def write(root, src_dir, arm, lr, s, edit=None, size53=False, lm0=None):
    """copy one run (+ offline_lm, check) into root as the Torch cell (arm, lr, seed s) with an optional edit."""
    r = json.loads(runjson(src_dir).read_text())
    assert r["args"]["seed"] == s
    tag = f"rb6tb-190m-2pct-k2-{arm}-lr{lr}"
    tdir = root / f"{tag}-s{s}" / tag
    tdir.mkdir(parents=True)
    r["args"].update(tag=f"_{tag}", lr=float(lr))
    if size53:
        r["args"].update(d_model=768, n_layers=12, n_heads=12, d_ff=3072)
        r.update(n_params=190297344, grad_ckpt_on=True, geo_layers=[4, 8], read_layers=[4, 8], code_sha="93e22bf4c4c0")
        r["dep_sha"] = dict(r.get("dep_sha", {}), **{"toy_struct.py": "d182ce548bcc"})
        r["curve"][0]["lm"] = lm0
    if edit:
        edit(r)
    out = tdir / f"run_whole_{r['args']['target']}_seed{s}_{tag}.json"
    r["args"]["out"] = str(tdir)
    out.write_text(json.dumps(r))
    off = json.loads((Path(src_dir) / "offline_lm.json").read_text())
    assert len(off) == 1
    v = dict(next(iter(off.values())), tag=f"_{tag}", code_sha=r["code_sha"])
    (tdir.parent / "offline_lm.json").write_text(json.dumps({str(out): v}))
    (tdir.parent / "check.json").write_text(json.dumps({"pass": True, "checks": [], "info": {}}))
    return out


def prior_table(act, target):
    def ed(r):
        r["args"].update(prior_domains=",".join(act), target=target)
        for d in X.DOMS:
            r["domains"][d]["prior"] = d in act
            r["domains"][d]["lam"] = 1.0 if d in act else 0.0
        r["domains"]["numbers"]["target"] = target
        r["lam_domains"] = {d: (1.0 if d in act else 0.0) for d in X.DOMS}
    return ed


def make(root):
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    for s in range(4):
        write(root, real("all6-lam1", "6e-4", s), "all6-lam1", "1.2e-3", s)
        write(root, real("only-numbers", "6e-4", s), "only-numbers", "1.2e-3", s)
        write(root, real("all6-lam1", "6e-4", s), "only-clock", "6e-4", s, prior_table(("clock",), "none"))
        write(root, real("all6-lam1", "6e-4", s), "all6-minus-clock", "6e-4", s,
              prior_table(("numbers", "months", "weekdays", "s4", "dim"), "helix3"))
        lm0 = json.loads(runjson(real("none", "6e-4", s)).read_text())["curve"][0]["lm"]
        for c, dirs in B53.items():
            write(root, RO / "results" / dirs[s], "none", X.CELLS[c][1], s, size53=True, lm0=lm0)
    return root


def run(root, extra=(), name="x"):
    js = OUT / f"{name}.json"
    js.unlink(missing_ok=True)
    cmd = [PY, "tools/rb6scale_ext_summary.py", str(REAL), str(root), "--json", str(js), *extra]
    o = subprocess.run(cmd, cwd=WT, capture_output=True, text=True)
    (OUT / f"{name}.txt").write_text(o.stdout + o.stderr)
    return o.returncode, o.stdout + o.stderr, (json.loads(js.read_text()) if js.exists() and o.returncode == 0 else None)


def edit_run(root, cell, s, fn, offline=None, check=None):
    arm, lr = X.CELLS[cell]
    d = root / f"rb6tb-190m-2pct-k2-{arm}-lr{lr}-s{s}"
    p = runjson(d)
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))
    if offline is not None:
        o = json.loads((d / "offline_lm.json").read_text())
        for v in o.values():
            v["bpb_full"] = offline
        (d / "offline_lm.json").write_text(json.dumps(o))
    if check is not None:
        (d / "check.json").write_text(json.dumps(check))


def copy_base(name):
    dst = OUT / name
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(OUT / "base", dst)
    return dst


OUT.mkdir(exist_ok=True)
# ---------------- unit tests of the pre-registered rules
A = X.attribution
say(A(62.2, 62.2, 96.4)[0].startswith("~ all6") and abs(A(62.2, 62.2, 96.4)[1] - 1) < 1e-12, "rule Q2: mc = all6 -> f 1, '~ all6'")
say(A(96.4, 62.2, 96.4)[0].startswith("~ only-numbers") and A(96.4, 62.2, 96.4)[1] == 0, "rule Q2: mc = only-numbers -> f 0, '~ only-numbers'")
say(A(79.3, 62.2, 96.4)[0].startswith("UNDETERMINED (in between)"), "rule Q2: mc halfway -> UNDETERMINED")
g = 62.2 - 96.4
say(A(96.4 + g * 2 / 3, 62.2, 96.4)[0].startswith("~ all6") and A(96.4 + g / 3, 62.2, 96.4)[0].startswith("~ only-numbers"),
    "rule Q2: the boundaries f = 2/3 -> ~ all6, f = 1/3 -> ~ only-numbers")
say(A(50.0, 62.2, 96.4)[0].startswith("~ all6") and A(99.0, 62.2, 96.4)[0].startswith("~ only-numbers"),
    "rule Q2: beyond all6 (f > 1) -> ~ all6; above only-numbers (f < 0) -> ~ only-numbers")
say(A(80, 90, 95)[0].startswith("UNDETERMINED (|all6 - only-numbers| = 5.0 < 10"), "rule Q2: a gap < 10 points -> UNDETERMINED (no gap)")
Q = X.q2_reading
on_, a6_ = 96.4, 62.2
seedv = lambda fs: {s: (on_ + f_ * (a6_ - on_), a6_, on_) for s, f_ in enumerate(fs)}
mean_mc = lambda fs: on_ + sum(fs) / len(fs) * (a6_ - on_)
r_ = Q(mean_mc([1.97, 1.10, 0.0, 0.0]), a6_, on_, seedv([1.97, 1.10, 0.0, 0.0]))
say(r_[1] >= 2 / 3 and r_[0].startswith("UNDETERMINED (seed-inconsistent") and r_[3] == 2, f"rule Q2 (E1 M3): mean f {r_[1]:.2f} past 2/3, 2/4 seeds -> {r_[0][:40]}")
r_ = Q(mean_mc([0.9, 0.8, 0.8, 0.3]), a6_, on_, seedv([0.9, 0.8, 0.8, 0.3]))
say(r_[0].startswith("~ all6") and r_[3] == 3, f"rule Q2 (E1 M3): mean f {r_[1]:.2f}, 3/4 seeds >= 2/3 -> ~ all6")
r_ = Q(mean_mc([0.1, 0.0, 0.2, 0.9]), a6_, on_, seedv([0.1, 0.0, 0.2, 0.9]))
say(r_[0].startswith("~ only-numbers") and r_[3] == 3, f"rule Q2 (E1 M3): mean f {r_[1]:.2f}, 3/4 seeds <= 1/3 -> ~ only-numbers")
r_ = Q(mean_mc([0.1, 0.5, 0.6, 0.0]), a6_, on_, seedv([0.1, 0.5, 0.6, 0.0]))
say(r_[0].startswith("UNDETERMINED (seed-inconsistent") and r_[3] == 2, f"rule Q2 (E1 M3): mean f {r_[1]:.2f} <= 1/3 but 2/4 seeds -> UNDETERMINED")
r_ = Q(mean_mc([0.5] * 4), a6_, on_, seedv([0.5] * 4))
say(r_[0] == "UNDETERMINED (in between)" and r_[3] is None, "rule Q2 (E1 M3): in between stays UNDETERMINED (in between)")
RF = X.range_flag
say(RF(-0.2) == "REVERSED (< 0)" and RF(1.3) == "OVERSHOOT (> 1)" and RF(0.0) is None and RF(1.0) is None and RF(0.5) is None
    and RF(None) is None, "rule (L3): range flags REVERSED < 0 / OVERSHOOT > 1, none inside [0, 1]")
P_ = X.persistence
say(P_(-34.2, -34.2, [-40, -40, -25, -30])[0] == "PERSISTS at 1.2e-3", "rule Q1: same gap -> PERSISTS")
say(P_(-30.0, -34.2, [-60, 5, 6, -61])[0].startswith("UNDETERMINED (r >= 2/3 but only 2/4"), "rule Q1: r >= 2/3 but 2/4 seeds -> UNDETERMINED")
say(P_(-5.0, -34.2, [-5, -5, -5, -5])[0] == "LARGELY GONE at 1.2e-3" and P_(3.0, -34.2, [3] * 4)[0] == "LARGELY GONE at 1.2e-3",
    "rule Q1: r <= 1/3 (incl. a sign flip) -> LARGELY GONE")
say(P_(-17.0, -34.2, [-17] * 4)[0] == "PARTIAL / UNDETERMINED", "rule Q1: r = 1/2 -> PARTIAL / UNDETERMINED")
say(P_(-30.0, -5.0, [-30] * 4)[0].startswith("UNDETERMINED (no 6e-4 gap"), "rule Q1: no 6e-4 gap -> UNDETERMINED")
F = X.floor_label
say(F({"verdict": "FLOOR", "seed_ci": [-1, 3], "floor": 9.2}) == "AT THE FLOOR", "rule Q3: FLOOR -> AT THE FLOOR")
say(F({"verdict": "MEASURABLE", "seed_ci": [20, 60], "floor": 9.2}) == "LEFT THE FLOOR", "rule Q3: MEASURABLE -> LEFT")
say(F({"verdict": "BORDERLINE", "seed_ci": [80, 101], "floor": 11.5}) == "LEFT THE FLOOR", "rule Q3: BORDERLINE above the floor (CI > 90) -> LEFT")
say(F({"verdict": "BORDERLINE", "seed_ci": [-6, 35], "floor": 11.5}).startswith("UNDETERMINED"), "rule Q3: BORDERLINE straddling -> UNDETERMINED")
say(F({"verdict": "INVALID (seen 25 < 80)", "seed_ci": [20, 60], "floor": 9.2}).startswith("INVALID"), "rule Q3: INVALID -> not counted")
FA = X.floor_answer
say(FA({"a": "AT THE FLOOR", "b": "LEFT THE FLOOR"}).startswith("YES: b"), "rule Q3: any LEFT -> YES")
say(FA({"a": "AT THE FLOOR", "b": "AT THE FLOOR", "c": "INVALID (not counted)"}).startswith("NO:"), "rule Q3: all valid FLOOR -> NO")
say(FA({"a": "AT THE FLOOR", "b": "UNDETERMINED (x)"}).startswith("UNDETERMINED"), "rule Q3: FLOOR + straddling -> UNDETERMINED")
BT = X.best_tested
say(BT({"none@6e-4": 1.0, "none@4.8e-3": 3.0, "none@9.6e-3": 3.0, "none@1.92e-2": 2.0}) == "none@4.8e-3", "rule Q3: tie -> the lower lr")
say(BT({"none@6e-4": 1.0, "none@4.8e-3": 3.0, "none@9.6e-3": 9.0, "none@1.92e-2": None}, {"none@9.6e-3"}) == "none@4.8e-3",
    "rule Q3: INVALID and missing cells excluded")

# ---------------- base: known answers
root = make(OUT / "base")
rc, txt, J = run(root, name="base")
say(rc == 0 and J and J["status"] == "complete" and "40 runs in the design" in txt and not J["excluded"] and not J["ignored"],
    f"base: rc {rc}, status {J and J['status']}, 40 runs, none excluded / ignored")
M = json.loads((SP / "rb6scale_190m.json").read_text())   # the stored rb6_190m read-out (rb6scale_summary e807e43be36b)
ROLE = {"none@6e-4": "none_plr", "none@4.8e-3": "none_nlr", "all6-lam1@6e-4": "all6", "only-numbers@6e-4": "only_numbers"}
if J:
    nd = 0
    for c, ro in ROLE.items():
        for mt in X.METRICS:
            a, b = J["cells"][f"{c}|{mt}"], M["cells"][f"190m|{ro}|{mt}"]
            nd += not (a["values"] == b["values"] and a["mean"] == b["mean"]
                       and (a["gate"] or {}).get("verdict") == (b["gate"] or {}).get("verdict")
                       and (a["gate"] or {}).get("seed_ci") == (b["gate"] or {}).get("seed_ci"))
    say(nd == 0, f"base: the 4 main cells x 9 metrics = the stored rb6_190m read-out's values / means / gates ({nd} differ)")
    same = lambda c1, c2: all(J["cells"][f"{c1}|{mt}"]["values"] == J["cells"][f"{c2}|{mt}"]["values"] for mt in X.METRICS)
    say(same("all6-lam1@1.2e-3", "all6-lam1@6e-4") and same("only-numbers@1.2e-3", "only-numbers@6e-4")
        and same("only-clock@6e-4", "all6-lam1@6e-4") and same("all6-minus-clock@6e-4", "all6-lam1@6e-4"),
        "base: every relabelled cell = its source cell (9 metrics x 4 seeds)")
    q1 = {(r["a"], r["m"]): r for r in J["families"]["Q1"]}
    say(all(q1[(f"all6-lam1@1.2e-3", d)]["diffs"] == q1[(f"all6-lam1@6e-4", d)]["diffs"] for d in X.DOMS),
        "base Q1: the 1.2e-3 rows = the 6e-4 rows (known answer)")
    s3 = {r["m"]: r for r in M["families"]["190m|S3"]}
    say(all(abs(q1[("all6-lam1@6e-4", d)]["mean"] - s3[d]["mean"]) < 1e-12 and q1[("all6-lam1@6e-4", d)]["p_t"] == s3[d]["p_t"]
            for d in X.DOMS), "base Q1: the 6e-4 rows = the stored S3 family (means, t p)")
    q1r = J["readings"]["Q1"]
    say(q1r["label"] == "PERSISTS at 1.2e-3" and abs(q1r["r"] - 1) < 1e-12 and round(q1r["d6"], 1) == -34.2,
        f"base Q1 reading: {q1r['label']} (r {q1r['r']}, d6 {q1r['d6']:.1f})")
    q2 = {(r["a"], r["b"]): r for r in J["families"]["Q2"]}
    s1 = {r["m"]: r for r in M["families"]["190m|S1"]}
    say(q2[("only-clock@6e-4", "only-numbers@6e-4")]["diffs"] == s3["clock"]["diffs"]
        and q2[("only-clock@6e-4", "none@6e-4")]["diffs"] == s1["clock"]["diffs"]
        and q2[("all6-minus-clock@6e-4", "all6-lam1@6e-4")]["diffs"] == [0.0] * 4,
        "base Q2: only-clock (= all6 copy) rows = S3 / S1 clock rows; all6-minus-clock - all6 = 0 at every seed")
    q2r = J["readings"]["Q2"]
    say(q2r["label"].startswith("~ all6") and abs(q2r["f"] - 1) < 1e-12 and q2r["seeds_on_side"] == 4 and q2r["f_flag"] is None
        and not q2r["per_seed_flags"], f"base Q2 reading: {q2r['label'][:40]}... f {q2r['f']}, {q2r['seeds_on_side']}/4 seeds, no flag")
    say(X.Q1_SCOPE in txt and X.Q2_MEANING in txt and txt.count(X.Q3_MEANING) == 2 and "gradient split beside Q2" in txt
        and J["readings"]["Q1"]["r_flag"] is None and len(J["readings"]["Q1"]["per_seed_r"]) == 4,
        "base (E1): the Q1 scope, Q2 meaning, Q3 meaning (2 answer lines) and the Q2 gradient split are printed")
    say("none@1.92e-2 lm curve" in txt and len(J["lm_curve"]) == 4 and all(v["diverged_at"] is None for v in J["lm_curve"].values()),
        "base (E1): the none@1.92e-2 lm-curve summary, 4 seeds, finite throughout")
    # none@9.6e-3 = the 53M none@8e-3 cell of the stored read-out (gate incl.); none@1.92e-2 = an independent recomputation
    nd = 0
    for mt in X.METRICS:
        a, b = J["cells"][f"none@9.6e-3|{mt}"], M["cells"][f"53m|none_nlr|{mt}"]
        nd += not (a["values"] == b["values"] and (a["gate"] or {}).get("verdict") == (b["gate"] or {}).get("verdict"))
    say(nd == 0, f"base: none@9.6e-3 (53M none@8e-3 copies) = the stored 53M none@8e-3 cell, values + gates ({nd} differ)")
    src = [json.loads(runjson(RO / "results" / d).read_text()) for d in B53["none@1.92e-2"]]
    xs = [dict(r=r, path=runjson(RO / "results" / d)) for r, d in zip(src, B53["none@1.92e-2"])]
    ind = {mt: [R6.mval(x, mt) for x in xs] for mt in X.DOMS}
    ind["s4_hard"] = [R6.s4_hard_metrics(x["r"])["area"] for x in xs]
    say(all(J["cells"][f"none@1.92e-2|{mt}"]["values"] == v for mt, v in ind.items()),
        "base: none@1.92e-2 = the 53M none@1.6e-2 runs recomputed independently (rainbow6_summary.mval / s4_hard_metrics)")
    # Q3: labels from an independent gate computation, best tested = argmax
    for mt in ("s4", "dim"):
        lab = J["readings"][f"Q3_{mt}"]["labels"]
        exp = {}
        for c in X.NONE_GRID:
            g = J["cells"][f"{c}|{mt}"]["gate"]
            ci, fl = g["seed_ci"], g["floor"]
            exp[c] = ("AT THE FLOOR" if ci[1] < fl else "LEFT THE FLOOR" if ci[0] >= fl else "UNDETERMINED")
        say(all(lab[c].startswith(exp[c]) for c in X.NONE_GRID), f"base Q3 {mt}: labels {lab} = the CI rule")
    for mt in ("s4", "s4_hard", "dim"):
        means = {c: J["cells"][f"{c}|{mt}"]["mean"] for c in X.NONE_GRID}
        say(J["readings"]["Q3_best"][mt] == max(means, key=means.get), f"base Q3: best tested on {mt} = argmax {means}")
    q3 = J["families"]["Q3"]
    say(all(abs(r["mean"] - (J["cells"][f"{r['a']}|{r['m']}"]["mean"] - J["cells"][f"{r['b']}|{r['m']}"]["mean"])) < 1e-9 for r in q3)
        and len(q3) == 6, "base Q3: 6 rows, each mean = the cell means' difference")
    from scipy import stats
    r_ = q2[("only-clock@6e-4", "none@6e-4")]
    say(abs(stats.ttest_1samp(r_["diffs"], 0).pvalue - r_["p_t"]) < 1e-9, "base: a t p = scipy's")
    say("none@best" not in txt and "none@low" not in txt, "base: no 'none@best' / 'none@low' label")

# ---------------- scenarios
rt = copy_base("missing")
shutil.rmtree(rt / "rb6tb-190m-2pct-k2-all6-minus-clock-lr6e-4-s2")
rc, txt, _ = run(rt, name="missing_strict")
say(rc != 0 and "REFUSED" in txt and "all6-minus-clock@6e-4 [2]" in txt, "missing: strict REFUSES naming all6-minus-clock@6e-4 [2]")
rc, txt, J2 = run(rt, ("--allow_partial",), name="missing_partial")
say(rc == 0 and J2["status"] == "PROVISIONAL" and "PROVISIONAL (n=3 of 4)" in txt and J2["readings"]["Q2"]["provisional"],
    "missing: --allow_partial -> PROVISIONAL n=3 of 4, the Q2 reading PROVISIONAL")

rt = copy_base("wronglr")
edit_run(rt, "none@9.6e-3", 1, lambda r: r["args"].update(lr=4.8e-3))
rc, txt, J2 = run(rt, ("--allow_partial",), name="wronglr")
say(rc == 0 and any("none@9.6e-3 s1" in e[0] and "trainer args" in " ".join(e[1]) for e in J2["excluded"]),
    "wronglr: args.lr != the tag's lr -> EXCLUDED (trainer args)")

rt = copy_base("clockon")
def clock_on(r):
    r["domains"]["clock"]["prior"] = True
    r["lam_domains"]["clock"] = 1.0
edit_run(rt, "all6-minus-clock@6e-4", 0, clock_on)
rc, txt, J2 = run(rt, ("--allow_partial",), name="clockon")
say(rc == 0 and any("all6-minus-clock@6e-4 s0" in e[0] and "lam_domains" in " ".join(e[1]) for e in J2["excluded"]),
    "clockon: an all6-minus-clock run with the clock prior on -> EXCLUDED (lam_domains / active priors)")
rt = copy_base("clockoff_only")
def clock_off(r):
    r["domains"]["clock"]["prior"] = False
    r["lam_domains"]["clock"] = 0.0
edit_run(rt, "only-clock@6e-4", 3, clock_off)
rc, txt, J2 = run(rt, ("--allow_partial",), name="clockoff_only")
say(rc == 0 and any("only-clock@6e-4 s3" in e[0] for e in J2["excluded"]), "clockoff: an only-clock run without the clock prior -> EXCLUDED")

rt = copy_base("failcheck")
edit_run(rt, "only-clock@6e-4", 1, lambda r: None, check={"pass": False, "checks": [{"check": "x", "ok": False}], "info": {}})
rc, txt, J2 = run(rt, ("--allow_partial",), name="failcheck")
say(rc == 0 and any("only-clock@6e-4 s1" in e[0] and "FAILED" in " ".join(e[1]) for e in J2["excluded"]), "failcheck: -> EXCLUDED")

rt = copy_base("diverged")
def diverge(r):
    for e in r["curve"]:
        if e["step"] >= 9000:
            e["lm"] = float("nan")
            e["val_loss"] = float("nan")
edit_run(rt, "none@1.92e-2", 3, diverge, offline=float("nan"))
rc, txt, J2 = run(rt, name="diverged")
say(rc == 0 and J2["status"] == "complete" and any("DIVERGED" in w for w in J2["warnings"])
    and J2["lm_curve"]["3"]["diverged_at"] == 9000 and "DIVERGED at step 9000" in txt
    and J2["cells"]["none@1.92e-2|bpb"]["mean"] is None and "DIVERGED seed(s): s3" in txt
    and J2["cells"]["none@1.92e-2|s4"]["values"] == J["cells"]["none@1.92e-2|s4"]["values"],
    "diverged: kept as data (DIVERGED warning), its bpb mean NOT computed, its accuracies unchanged")

rt = copy_base("q2inconsistent")
def graft_clock(cell_src, s):
    """all6-minus-clock s: its clock read-outs (every clock_* / hits_clock_* curve key, the clock* trainer metrics) from
    the real cell_src run of seed s (the same eval sets per seed: the splits are arm-independent, asserted)"""
    src = json.loads(runjson(real(*cell_src, s)).read_text())
    def ed(r):
        strip = lambda d: {k: v for k, v in d.items() if k not in ("prior", "shuf", "lam")}
        assert strip(r["domains"]["clock"]) == strip(src["domains"]["clock"])
        assert [e["step"] for e in r["curve"]] == [e["step"] for e in src["curve"]]
        for e, es in zip(r["curve"], src["curve"]):
            for k in [k for k in e if k.startswith(("clock_", "hits_clock_"))]:
                del e[k]
            e.update({k: v for k, v in es.items() if k.startswith(("clock_", "hits_clock_"))})
        r["metrics"] = {k: v for k, v in r["metrics"].items() if not k.startswith("clock")}
        r["metrics"].update({k: v for k, v in src["metrics"].items() if k.startswith("clock")})
    return ed
for s, src_cell in ((0, ("none", "6e-4")), (1, ("none", "6e-4")), (2, ("only-numbers", "6e-4")), (3, ("only-numbers", "6e-4"))):
    edit_run(rt, "all6-minus-clock@6e-4", s, graft_clock(src_cell, s))
rc, txt, J2 = run(rt, name="q2inconsistent")
if J2:
    q = J2["readings"]["Q2"]
    want = [J["cells"]["none@6e-4|clock"]["values"][0], J["cells"]["none@6e-4|clock"]["values"][1],
            J["cells"]["only-numbers@6e-4|clock"]["values"][2], J["cells"]["only-numbers@6e-4|clock"]["values"][3]]
    say(rc == 0 and J2["cells"]["all6-minus-clock@6e-4|clock"]["values"] == want and q["f"] >= 2 / 3
        and q["label"].startswith("UNDETERMINED (seed-inconsistent") and q["seeds_on_side"] == 2
        and sum(v == "OVERSHOOT (> 1)" for v in q["per_seed_flags"].values()) == 2,
        f"q2inconsistent (E1 M3): mc clock = the grafted values, mean f {q['f']:.2f} >= 2/3 but 2/4 seeds -> {q['label'][:44]}...; "
        f"per-seed flags {q['per_seed_flags']}")
else:
    say(False, f"q2inconsistent: rc {rc} {txt[-400:]}")

rt = copy_base("nanother")
edit_run(rt, "all6-lam1@1.2e-3", 1, lambda r: r["curve"][40].update(lm=float("nan")))
rc, txt, _ = run(rt, ("--allow_partial",), name="nanother")
say(rc != 0 and "AssertionError" in txt, "nanother (E1 / L6): a NaN in all6-lam1@1.2e-3 (not none@1.92e-2) -> the read-out stops")

rt = copy_base("datasha")
edit_run(rt, "all6-lam1@1.2e-3", 2, lambda r: r["curve"][5].update(data_sha="0" * 16))
rc, txt, _ = run(rt, name="datasha")
say(rc != 0 and "data mixture differs" in txt, "datasha: a data_sha mismatch -> AssertionError")

# ---------------- the rb6_190m read-out is unchanged (byte-identical to the stored output)
cmd = (f"{PY} {WT}/tools/rb6scale_summary.py {REAL} results/401-rb6pb-* results/404-rb6pb-* results/430-rb6pb-* "
       "results/431-rb6pb-* results/419-rb6pb-* results/415-rb6pb-* results/432-rb6pb-* results/433-rb6pb-* "
       "results/438-rb6mb-* results/444-rb6mb-* results/450-rb6mb-* results/456-rb6mb-* results/46[6-9]-rb6mb-* "
       f"--json {OUT}/main_now.json")
o = subprocess.run(["bash", "-c", cmd], cwd=RO, capture_output=True, text=True)
say(o.returncode == 0 and o.stdout + o.stderr == (SP / "rb6_190m_readout.txt").read_text()
    and (OUT / "main_now.json").read_text() == (SP / "rb6scale_190m.json").read_text(),
    "the rb6_190m read-out (rb6scale_summary.py, the AUDIT command) = the stored output, text and JSON byte for byte")
print("RO EXT TEST", "PASS" if ok_all else "FAIL")
