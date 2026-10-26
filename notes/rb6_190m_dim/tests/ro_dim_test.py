"""Tests of tools/rb6scale_dim_summary.py (selection + read-out of rb6_190m_dim) before any run of the batch exists.
Fake trees (scale_work/rb190dim_scripts/ro_work/<scenario>/) of copied run JSONs (+ check.json, offline_lm.json):
  none@6e-4 / 4.8e-3 s0-3, all6-lam1@6e-4 s0-3      <- the real rb6_190m runs (unchanged)
  none@9.6e-3 / 1.92e-2 s0-3                         <- the real rb6_190m_ext runs (unchanged)
  none@3.84e-2 s0-3 (Stage D1 stand-in)              <- the real none@1.92e-2 runs, lr / tag relabelled (dim and bpb equal
                                                        -> an exact tie -> the lower lr 1.92e-2, closed)
  all6-lam1@6e-4 s4-7, none@1.92e-2 s4-7 (D2)        <- the Burst 53M confirm runs 471/473/475/477 (all6-lam1) and
                                                        470/472/474/476 (none) with the 190M size facts (the data are
                                                        size-independent; the method of ro_ext_test.py)
  + a real only-numbers@6e-4 s0 run (must be IGNORED)
Scenarios (known answers): base -> SURVIVES (seeds 0-3 row = the rb6_190m_ext Q3 dim row exactly; confirm row = the 53M
values), ns (p >= .05), unstable03 (2/4 wins), confirm_disagree, open (none@3.84e-2 = all6's dim: OPEN, D2 cell
none@3.84e-2), diverged (none@3.84e-2 s1 non-finite: data, worst bpb, 1.92e-2 picked), nanother (a NaN in all6 s5 ->
stops), incomplete_d1 / incomplete_d2 (strict refuses; --allow_partial INCOMPLETE), failcheck / wronglr (excluded ->
incomplete), datasha; flags: --picks agreeing / disagreeing, select never overwrites, --allow_partial on / off; unit tests
of choose() and verdict() on synthetic numbers (ties, exact 2-point boundary, OPEN, NaN bpb, REVERSED, every label).
    python ro_dim_test.py"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path("/scratch/yh6210/llmadd-burst-scale")
RO = Path("/scratch/yh6210/llmadd-burst-ro")
RT = Path("/scratch/yh6210/llmadd_torch/runs")
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
OUT = HERE / "ro_work"
sys.path.insert(0, str(WT / "tools"))
import rb6scale_dim_summary as D   # noqa: E402
import rainbow6_summary as R6      # noqa: E402
import readout_rules as RR         # noqa: E402

ok_all = True
EXT_JSON = Path("/scratch/yh6210/llmadd_torch/scale_work/rb190dim_reg/before_rb6_190m_ext.json")   # the stored ext read-out


def say(cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


def runjson(d):
    return next(Path(d).glob("*/run_*.json"))


def real(arm, lr, s):
    b = "rb6_190m_ext" if (arm, lr) in (("none", "9.6e-3"), ("none", "1.92e-2")) else "rb6_190m"
    return RT / b / "out" / f"rb6tb-190m-2pct-k2-{arm}-lr{lr}-s{s}"


B53 = {"all6-lam1": {s: f"{471 + 2 * (s - 4)}-rb6pb-53m-2pct-k2-all6-lam1-s{s}" for s in range(4, 8)},
       "none": {s: f"{470 + 2 * (s - 4)}-rb6pb-53m-2pct-k2-none-s{s}" for s in range(4, 8)}}


def write(root, src_dir, arm, lr, s, edit=None, size53=False):
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


def make(root, d2_none_lr="1.92e-2"):
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    for s in range(4):
        for lr in ("6e-4", "4.8e-3", "9.6e-3", "1.92e-2"):
            write(root, real("none", lr, s), "none", lr, s)
        write(root, real("none", "1.92e-2", s), "none", "3.84e-2", s)
        write(root, real("all6-lam1", "6e-4", s), "all6-lam1", "6e-4", s)
    for s in range(4, 8):
        write(root, RO / "results" / B53["all6-lam1"][s], "all6-lam1", "6e-4", s, size53=True)
        write(root, RO / "results" / B53["none"][s], "none", d2_none_lr, s, size53=True)
    write(root, real("only-numbers", "6e-4", 0), "only-numbers", "6e-4", 0)
    return root


def cdir(root, arm, lr, s):
    return root / f"rb6tb-190m-2pct-k2-{arm}-lr{lr}-s{s}"


def edit_run(root, arm, lr, s, fn, check=None):
    d = cdir(root, arm, lr, s)
    p = runjson(d)
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))
    if check is not None:
        (d / "check.json").write_text(json.dumps(check))


def graft_dim(src):
    """the run's dim read-outs (every dim_* / hits_dim_* curve key, the dim* trainer metrics) from the run JSON src of the
    same seed (the same eval sets per seed: asserted)"""
    S_ = json.loads(Path(src).read_text())

    def ed(r):
        strip = lambda d: {k: v for k, v in d.items() if k not in ("prior", "shuf", "lam")}
        assert strip(r["domains"]["dim"]) == strip(S_["domains"]["dim"]) and r["args"]["seed"] == S_["args"]["seed"]
        assert [e["step"] for e in r["curve"]] == [e["step"] for e in S_["curve"]]
        for e, es in zip(r["curve"], S_["curve"]):
            for k in [k for k in e if k.startswith(("dim_", "hits_dim_"))]:
                del e[k]
            e.update({k: v for k, v in es.items() if k.startswith(("dim_", "hits_dim_"))})
        r["metrics"] = {k: v for k, v in r["metrics"].items() if not k.startswith("dim")}
        r["metrics"].update({k: v for k, v in S_["metrics"].items() if k.startswith("dim")})
    return ed


def run(root, mode, extra=(), name="x", picks_out=None):
    js = OUT / f"{name}_{mode}.json"
    js.unlink(missing_ok=True)
    if mode == "select":
        js = picks_out or js
        js.unlink(missing_ok=True)
    cmd = [PY, "tools/rb6scale_dim_summary.py", mode, str(root), "--json", str(js), *extra]
    o = subprocess.run(cmd, cwd=WT, capture_output=True, text=True)
    (OUT / f"{name}_{mode}.txt").write_text(o.stdout + o.stderr)
    return o.returncode, o.stdout + o.stderr, (json.loads(js.read_text()) if js.exists() and o.returncode == 0 else None)


def copy_base(name, src="base"):
    dst = OUT / name
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(OUT / src, dst)
    return dst


def reading(J):
    return J["reading"]["label"] if J else None


OUT.mkdir(exist_ok=True)
# ---------------- unit tests on synthetic numbers -------------------------------------------------------------------
g = list(D.NONE_GRID)
dims = lambda m: {lr: [m.get(lr, 10.0)] * 4 for lr in g}     # {lr: mean} -> four equal seeds (other lrs 10.0)
bp = lambda m=None: {lr: [(m or {}).get(lr, 1.20)] * 4 for lr in g}
c = D.choose(dims({"1.92e-2": 40.0, "3.84e-2": 41.5}), bp({"1.92e-2": 1.22, "3.84e-2": 1.25}))
say(c["pick"] == "1.92e-2" and c["tied"] == ["1.92e-2", "3.84e-2"] and not c["open"],
    f"choose: 3.84e-2 1.5 above 1.92e-2 (tied) with a higher bpb -> 1.92e-2, closed ({c['bracket'][:60]})")
c = D.choose(dims({"1.92e-2": 40.0, "3.84e-2": 41.5}), bp({"1.92e-2": 1.22, "3.84e-2": 1.21}))
say(c["pick"] == "3.84e-2" and not c["open"], "choose: tied, 3.84e-2 has the lower bpb -> 3.84e-2, NOT open (<= 2 above)")
c = D.choose(dims({"1.92e-2": 40.0, "3.84e-2": 42.0}), bp({"1.92e-2": 1.22, "3.84e-2": 1.25}))
say(c["pick"] == "1.92e-2" and "3.84e-2" in c["tied"], "choose: exactly 2.0 above = tied (within 2) -> lower bpb 1.92e-2")
c = D.choose(dims({"1.92e-2": 40.0, "3.84e-2": 42.1}), bp({"1.92e-2": 1.22, "3.84e-2": 1.25}))
say(c["pick"] == "3.84e-2" and c["open"] and c["bracket"].startswith("bracket OPEN"), "choose: 2.1 above -> 3.84e-2, OPEN")
c = D.choose(dims({"1.92e-2": 40.0, "3.84e-2": 40.0}), bp({"1.92e-2": 1.22, "3.84e-2": 1.22}))
say(c["pick"] == "1.92e-2", "choose: exact tie in dim and bpb -> the lower lr")
c = D.choose(dims({"1.92e-2": 40.0, "3.84e-2": 41.0}), {**bp({"1.92e-2": 1.30}), "3.84e-2": [1.2, float("nan"), 1.2, 1.2]})
say(c["pick"] == "1.92e-2" and c["mean_bpb"]["3.84e-2"] == math.inf, "choose: a non-finite bpb is the worst (diverged)")
c = D.choose(dims({"9.6e-3": 50.0, "1.92e-2": 40.0, "3.84e-2": 30.0}), bp())
say(c["pick"] == "9.6e-3" and not c["open"] and c["bracket"] == "closed", "choose: a middle pick -> closed")


def sv_of(x03, y03, x47=None, y47=None):
    return RR.survives(x03, y03, *((x47, y47) if x47 is not None else ()))


CH = {"open": False}
cases = [
    ("SURVIVES", sv_of([90, 95, 85, 80], [40, 50, 10, 45], [80, 85, 90, 70], [30, 20, 40, 50]), True),
    ("NOT SIGNIFICANT (paired t p", sv_of([90, 95, 85, 80], [90, 50, 10, 45]), True),
    ("DIRECTION UNSTABLE (only 2/4", sv_of([90, 95, 85, 80], [90, 95, 10, 45]), True),
    ("NOT SIGNIFICANT (mean <= 0", sv_of([10, 40, 15, 20], [40, 50, 10, 45]), True),   # diffs -30 -10 +5 -25
    ("REVERSED", sv_of([10, 12, 9, 11], [40, 45, 38, 44]), True),
    ("DIRECTION UNSTABLE (the confirm", sv_of([90, 95, 85, 80], [40, 50, 10, 45], [80, 20, 90, 30], [30, 40, 40, 50]), True),
    ("INCOMPLETE (passes on seeds 0-3", sv_of([90, 95, 85, 80], [40, 50, 10, 45]), False),
]
for want, sv, cc in cases:
    lab = D.verdict(CH, sv, None, cc)
    say(lab.startswith(want), f"verdict: {want} -> {lab[:90]}")
say(D.verdict(None, None, None, False).startswith("INCOMPLETE (data missing"), "verdict: no selection -> INCOMPLETE")
lab = D.verdict({"open": True}, sv_of([90, 95, 85, 80], [40, 50, 10, 45], [80, 85, 90, 70], [30, 20, 40, 50]), None, True)
say(lab.startswith("OPEN") and "SURVIVES" in lab, f"verdict: OPEN takes precedence over SURVIVES -> {lab[:90]}")

# ---------------- base: SURVIVES; known answers -----------------------------------------------------------------------
base = make(OUT / "base")
rc, txt, PK = run(base, "select", name="base", picks_out=OUT / "base_picks.json")
say(rc == 0 and PK and PK["status"] == "picked" and PK["pick"]["none_dim"] == "1.92e-2" and PK["open"] is False
    and PK["tied"] == ["1.92e-2", "3.84e-2"], f"base select: picked none_dim 1.92e-2 (exact tie with the relabelled 3.84e-2 -> "
    f"the lower lr), closed: {PK and PK['pick']} {PK and PK['bracket']}")
say(PK and abs(PK["mean"]["1.92e-2"] - 40.1) < 0.05 and abs(PK["mean"]["6e-4"] - 14.3) < 0.05,
    "base select: the means = FINDINGS (none@1.92e-2 40.1, none@6e-4 14.3)")
o = subprocess.run([PY, "tools/rb6scale_dim_summary.py", "select", str(base), "--json", str(OUT / "base_picks.json")],
                   cwd=WT, capture_output=True, text=True)
say(o.returncode != 0 and "never overwritten" in o.stderr, "select refuses to overwrite an existing picks JSON")
rc, txt, J = run(base, "readout", ("--picks", str(OUT / "base_picks.json")), name="base")
EXT = json.loads(EXT_JSON.read_text())
q3 = next(r_ for r_ in EXT["families"]["Q3"] if r_["a"] == "all6-lam1@6e-4" and r_["m"] == "dim")
say(rc == 0 and reading(J).startswith("SURVIVES") and J["status"] == "complete",
    f"base readout: {reading(J) and reading(J)[:60]}; status {J and J['status']}")
r03 = J and J["rows"]["dim"]["0-3"]
say(r03 and q3["b"] == "none@1.92e-2" and all(abs(a - b) < 1e-9 for a, b in zip(r03["diffs"], q3["diffs"]))
    and abs(r03["p_t"] - q3["p_t"]) < 1e-9, f"base: the seeds 0-3 dim row = the stored rb6_190m_ext Q3 row exactly "
    f"({[round(v, 1) for v in r03['diffs']] if r03 else None}, p {r03 and round(r03['p_t'], 3)})")
r47 = J and J["rows"]["dim"]["4-7"]
want47 = [R6.mval({"r": json.loads(runjson(RO / "results" / B53["all6-lam1"][s]).read_text()), "path": None}, "dim")
          - R6.mval({"r": json.loads(runjson(RO / "results" / B53["none"][s]).read_text()), "path": None}, "dim")
          for s in range(4, 8)]
say(r47 and all(abs(a - b) < 1e-9 for a, b in zip(r47["diffs"], want47)),
    f"base: the confirm row = the 53M source runs' all6 - none dim at seeds 4-7 ({[round(v, 1) for v in want47]})")
say("IGNORED rb6tb-190m-2pct-k2-only-numbers-lr6e-4 s0" in txt and "(agrees)" in txt,
    "base: the only-numbers run is IGNORED; --picks printed as agreeing")
say("flags used: mode readout; allow_partial False; picks " + str(OUT / "base_picks.json") in txt,
    "base: the flags used are printed")
gl = [l for l in txt.splitlines() if l.strip().startswith(("s0 LM share", "s5 LM share"))]
say(len(gl) == 2 and "frac clipped" in gl[0], f"base: gradient split + clip printed per all6 seed ({gl[:1]})")
# --picks disagreeing -> refused
bad = json.loads((OUT / "base_picks.json").read_text())
bad["pick"]["none_dim"] = "9.6e-3"
(OUT / "bad_picks.json").write_text(json.dumps(bad))
rc, txt, _ = run(base, "readout", ("--picks", str(OUT / "bad_picks.json")), name="badpicks")
say(rc != 0 and "disagrees with the recomputed selection" in txt, "--picks disagreeing (9.6e-3) -> the read-out REFUSES")
rc, txt, J0 = run(base, "readout", (), name="nopicks")
say(rc == 0 and "picks - (selection recomputed only)" in txt and "(agrees)" not in txt and reading(J0) == reading(J),
    "without --picks: selection recomputed only, the same reading (the flag takes effect only as a check)")

# ---------------- ns / unstable03 / confirm_disagree ------------------------------------------------------------------
rt = copy_base("ns")
edit_run(rt, "none", "1.92e-2", 0, graft_dim(runjson(real("all6-lam1", "6e-4", 0))))
rc, txt, J = run(rt, "readout", (), name="ns")
say(rc == 0 and reading(J).startswith("NOT SIGNIFICANT (paired t p") and J["rows"]["dim"]["0-3"]["wins"] == 3
    and abs(J["rows"]["dim"]["0-3"]["diffs"][0]) < 1e-9, f"ns: none@1.92e-2 s0 = all6's dim -> {reading(J)}")
rt = copy_base("unstable03")
for s in (0, 1):
    edit_run(rt, "none", "1.92e-2", s, graft_dim(runjson(real("all6-lam1", "6e-4", s))))
rc, txt, J = run(rt, "readout", (), name="unstable03")
say(rc == 0 and reading(J).startswith("DIRECTION UNSTABLE (only 2/4"), f"unstable03: -> {reading(J)}")
rt = copy_base("confirm_disagree")
for s in (4, 5):
    edit_run(rt, "none", "1.92e-2", s, graft_dim(runjson(RO / "results" / B53["all6-lam1"][s])))
rc, txt, J = run(rt, "readout", (), name="confirm_disagree")
say(rc == 0 and reading(J).startswith("DIRECTION UNSTABLE (the confirm seeds 4-7 disagree") and
    J["rows"]["dim"]["4-7"]["wins"] == 2, f"confirm_disagree: none@1.92e-2 s4, s5 = all6's dim -> {reading(J)}")

# ---------------- open: none@3.84e-2 = all6's dim at s0-3; the D2 none cell is 3.84e-2 --------------------------------
rt = make(OUT / "open", d2_none_lr="3.84e-2")
for s in range(4):
    edit_run(rt, "none", "3.84e-2", s, graft_dim(runjson(real("all6-lam1", "6e-4", s))))
rc, txt, PK = run(rt, "select", name="open", picks_out=OUT / "open_picks.json")
say(rc == 0 and PK["pick"]["none_dim"] == "3.84e-2" and PK["open"] is True and PK["bracket"].startswith("bracket OPEN"),
    f"open select: none_dim {PK and PK['pick']}, open {PK and PK['open']}")
rc, txt, J = run(rt, "readout", ("--picks", str(OUT / "open_picks.json")), name="open")
say(rc == 0 and reading(J).startswith("OPEN (bracket open") and "not best-vs-best" in reading(J) and
    J["rows"]["dim"]["4-7"] and J["rows"]["dim"]["4-7"]["n"] == 4, f"open readout: {reading(J) and reading(J)[:100]}")

# ---------------- diverged: none@3.84e-2 s1 non-finite from eval 30 ----------------------------------------------------
rt = copy_base("diverged")
def blow(r):
    for e in r["curve"][30:]:
        e.update(lm=float("nan"), val_loss=float("nan"), val_bits_per_byte=float("nan"))
edit_run(rt, "none", "3.84e-2", 1, blow)
rc, txt, PK = run(rt, "select", name="diverged", picks_out=OUT / "diverged_picks.json")
step30 = json.loads(runjson(cdir(rt, "none", "3.84e-2", 1)).read_text())["curve"][30]["step"]
say(rc == 0 and PK["status"] == "picked" and PK["pick"]["none_dim"] == "1.92e-2" and PK["table"]["3.84e-2"]["diverged"] == [1]
    and "DIVERGED seeds [1] (data)" in txt, f"diverged select: data (its dim counts, bpb worst) -> 1.92e-2; DIVERGED s1 shown")
rc, txt, J = run(rt, "readout", (), name="diverged")
say(rc == 0 and f"DIVERGED at step {step30}" in txt and reading(J).startswith("SURVIVES"),
    f"diverged readout: the lm-curve line names DIVERGED at step {step30}; reading {reading(J) and reading(J)[:20]}")
rt = copy_base("nanother")
edit_run(rt, "all6-lam1", "6e-4", 5, lambda r: r["curve"][40].update(lm=float("nan")))
rc, txt, _ = run(rt, "readout", ("--allow_partial",), name="nanother")
say(rc != 0 and "AssertionError" in txt, "nanother: a NaN in all6-lam1@6e-4 s5 (not a high-lr none cell) -> the read-out stops")

# ---------------- incomplete ------------------------------------------------------------------------------------------
rt = copy_base("incomplete_d1")
shutil.rmtree(cdir(rt, "none", "3.84e-2", 2))
rc, txt, PK = run(rt, "select", name="incomplete_d1", picks_out=OUT / "inc1_picks.json")
say(rc == 0 and PK and PK["status"] == "incomplete" and PK["pick"] == {} and "none@3.84e-2 s2: missing" in txt,
    "incomplete_d1 select: status incomplete, no pick, the missing run named")
rc, txt, _ = run(rt, "readout", (), name="incomplete_d1_strict")
say(rc != 0 and "REFUSED (strict)" in txt, "incomplete_d1 readout strict: REFUSED")
rc, txt, J = run(rt, "readout", ("--allow_partial", "--picks", str(OUT / "inc1_picks.json")), name="incomplete_d1")
say(rc == 0 and reading(J).startswith("INCOMPLETE (data missing") and J["status"] == "PROVISIONAL",
    f"incomplete_d1 readout --allow_partial: {reading(J)}; status {J and J['status']} (the flag takes effect)")
rt = copy_base("incomplete_d2")
shutil.rmtree(cdir(rt, "all6-lam1", "6e-4", 7))
rc, txt, _ = run(rt, "readout", (), name="incomplete_d2_strict")
say(rc != 0 and "REFUSED (strict)" in txt and "all6-lam1@6e-4 s7" in txt, "incomplete_d2 readout strict: REFUSED (all6 s7)")
rc, txt, J = run(rt, "readout", ("--allow_partial",), name="incomplete_d2")
say(rc == 0 and reading(J).startswith("INCOMPLETE (passes on seeds 0-3") and "PROVISIONAL (n=3 of 4)" in txt,
    f"incomplete_d2 readout --allow_partial: {reading(J)}")
rt = copy_base("unfinished")
edit_run(rt, "none", "3.84e-2", 3, lambda r: r.update(curve=r["curve"][:-5]))
rc, txt, PK = run(rt, "select", name="unfinished", picks_out=OUT / "unf_picks.json")
say(rc == 0 and PK["status"] == "incomplete" and any("unfinished" in p for p in PK["problems"]),
    "unfinished: a 3.84e-2 run without its last evals -> select incomplete")
rt = copy_base("failcheck")
edit_run(rt, "none", "3.84e-2", 0, lambda r: None, check={"pass": False, "checks": [{"check": "x", "ok": False}]})
rc, txt, PK = run(rt, "select", name="failcheck", picks_out=OUT / "fc_picks.json")
say(rc == 0 and PK["status"] == "incomplete" and any("check FAILED" in p for p in PK["problems"]),
    "failcheck: a failed check.json -> select incomplete")
rc, txt, _ = run(rt, "readout", (), name="failcheck")
say(rc != 0 and "EXCLUDED none@3.84e-2 s0" in txt, "failcheck readout: the run is EXCLUDED, strict refuses")
rt = copy_base("wronglr")
edit_run(rt, "none", "3.84e-2", 3, lambda r: r["args"].update(lr=0.0192))
rc, txt, PK = run(rt, "select", name="wronglr", picks_out=OUT / "wl_picks.json")
say(rc == 0 and PK["status"] == "incomplete" and any("trainer args differ" in p and "lr" in p for p in PK["problems"]),
    "wronglr: a run whose args lr != its tag -> identity problem, select incomplete")
rt = copy_base("datasha")
edit_run(rt, "none", "1.92e-2", 6, lambda r: r["curve"][5].update(data_sha="0" * 16))
rc, txt, _ = run(rt, "readout", (), name="datasha")
say(rc != 0 and "data mixture differs" in txt, "datasha: a data_sha mismatch at seed 6 -> AssertionError")

print("RO DIM TEST " + ("PASS" if ok_all else "FAIL"))
