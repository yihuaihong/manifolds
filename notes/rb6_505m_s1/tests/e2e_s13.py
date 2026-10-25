"""Amendment S1.3 end-to-end test of the operator scripts runs/rb6_505m/x1_select.sh and readout_505m.sh (the files
at their real location) on FAKE batch dirs; no 505M value is read or printed. The read-out runs the WORKTREE summary
under test, snapshotted into each fake dir with its COMMIT. Each fake dir has out/: the 12 sweep, 2 X1, main-stage
and X1 b follow-up cells, each a 53M Burst run of the same seed relabelled with the 505M size facts, with planted seeds
0-1 numbers area / bpb at the selection cells. Every picks file is written by the tool: picks.json by the
pre-registered unflagged select, picks_x1.json by x1_select.sh.
Scenarios:
- B (X1 -> b): bad args; a broken environment refused with nothing recorded (S1.3 I-1); x1_select once only; b /
  b-pre run with the sha header; a / c refused for the right reason; a refused read-out is re-runnable (I-3); an
  --allow_partial with nothing missing is refused; a tampered picks_x1.json is refused.
- The real-dir branch, via copies whose REAL is the fake dir (I-2): the live squeue, a failing squeue, R with '//',
  an idle queue with the pinned sha.
- Bp: 2.4e-3 s2 / s3 missing: strict b, then b --allow_partial in natural order -> PROVISIONAL (I-3). Bq: one more
  gap -> refused.
- A (X1 -> a): only a runs.
- C: the integrity assert is a data outcome -> c.
- F: a non-data crash records nothing and is re-runnable (I-1).
- M: the X1 runs not final: refused, then --x1-runs-final -> incomplete -> c.
- N: NaN bpb, binding OK.
    python e2e_s13.py"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path("/scratch/yh6210/llmadd-burst-scale")
REALR = Path("/scratch/yh6210/llmadd_torch/runs/rb6_505m")
RO = Path("/scratch/yh6210/llmadd-burst-ro/results")
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
W = HERE / "e2e13"
sys.path.insert(0, str(WT / "tools"))
import rainbow6_summary as R6   # noqa: E402
ok_all = True
SRC = {"none": [f"{466 + s}-rb6mb-53m-2pct-k2-none-lr8e-3-s{s}" for s in range(4)],
       "none24": [f"{n}-rb6mb-53m-2pct-k2-none-lr3e-3-s{s}" for s, n in enumerate((502, 504, 506, 479))],
       "none6": ["401-rb6pb-53m-2pct-k2-none-s0", "404-rb6pb-53m-2pct-k2-none-s1", "430-rb6pb-53m-2pct-k2-none-s2",
                 "431-rb6pb-53m-2pct-k2-none-s3"],
       "all6": ["419-rb6pb-53m-2pct-k2-all6-lam1-s0", "415-rb6pb-53m-2pct-k2-all6-lam1-s1", "432-rb6pb-53m-2pct-k2-all6-lam1-s2",
                "433-rb6pb-53m-2pct-k2-all6-lam1-s3"],
       "onum": [f"{n}-rb6mb-53m-2pct-k2-only-numbers-s{s}" for s, n in enumerate((438, 444, 450, 456))]}
SWEEP = [("none", lr) for lr in ("4.8e-3", "9.6e-3", "1.92e-2")] + [("all6-lam1", lr) for lr in ("3e-4", "6e-4", "1.2e-3")]
AREA_B = {("none", "2.4e-3"): 45.0, ("none", "4.8e-3"): 40.0, ("none", "9.6e-3"): 39.5, ("none", "1.92e-2"): 20.0,
          ("all6-lam1", "3e-4"): 60.0, ("all6-lam1", "6e-4"): 70.0, ("all6-lam1", "1.2e-3"): 64.0}
BPB = {("none", "2.4e-3"): 1.10, ("none", "4.8e-3"): 1.12, ("none", "9.6e-3"): 1.15, ("none", "1.92e-2"): 1.20,
       ("all6-lam1", "3e-4"): 1.11, ("all6-lam1", "6e-4"): 1.10, ("all6-lam1", "1.2e-3"): 1.12}


def say(cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


def runjson(d):
    return next(Path(d).glob("*/run_*.json"))


def plant(r, area, bpb):
    cur = R6.metrics(r, "numbers")["area"]
    f = (lambda v, g=(100 - area) / (100 - cur): 1 - (1 - v) * g) if area >= cur else (lambda v, c=area / cur: v * c)
    for e in r["curve"]:
        for b in R6.B:
            e[f"held_{b}"] = f(e[f"held_{b}"])
    got = R6.metrics(r, "numbers")
    assert abs(got["area"] - area) < 1e-6
    r["metrics"]["numbers"] = dict(r["metrics"]["numbers"], **{k: got[k] for k in ("area", "final", "final3", "t50", "t90", "n_evals")})
    r["curve"][-1]["val_bits_per_byte"] = bpb


def write(out, src, arm, lr, s, area=None, bpb=None, edit=None):
    sd = RO / SRC[src][s]
    r = json.loads(runjson(sd).read_text())
    assert r["args"]["seed"] == s and r["args"]["d_model"] == 384
    tag = f"rb6tb-505m-2pct-k2-{arm}-lr{lr}"
    tdir = out / f"{tag}-s{s}" / tag
    tdir.mkdir(parents=True)
    r["args"].update(tag=f"_{tag}", lr=float(lr), d_model=1024, n_layers=24, n_heads=16, d_ff=4096, out=str(tdir))
    r.update(n_params=505437184, grad_ckpt_on=True, geo_layers=[8, 16], read_layers=[8, 16], code_sha="93e22bf4c4c0")
    r["dep_sha"] = dict(r.get("dep_sha", {}), **{"toy_struct.py": "d182ce548bcc"})
    if area is not None:
        plant(r, area, bpb)
    if edit:
        edit(r)
    p = tdir / f"run_whole_{r['args']['target']}_seed{s}_{tag}.json"
    p.write_text(json.dumps(r))
    off = json.loads((sd / "offline_lm.json").read_text())
    v = dict(next(iter(off.values())), tag=f"_{tag}", code_sha=r["code_sha"])
    (tdir.parent / "offline_lm.json").write_text(json.dumps({str(p): v}))
    (tdir.parent / "check.json").write_text(json.dumps({"pass": True, "checks": [], "info": {}}))


def build(name, area24, edit24=None, drop=()):
    R = W / name
    out = R / "out"
    src = {"none": "none", "all6-lam1": "all6"}
    for arm, lr in SWEEP:   # the sweep, seeds 0-1, planted
        for s in (0, 1):
            write(out, src[arm], arm, lr, s, AREA_B[(arm, lr)], BPB[(arm, lr)])
    for s in (0, 1):        # X1
        write(out, "none24", "none", "2.4e-3", s, area24, BPB[("none", "2.4e-3")], (edit24 or {}).get(s))
    main = [("none6", "none", "6e-4", s) for s in range(4)] + [("none", "none", "4.8e-3", s) for s in (2, 3)] + \
           [("all6", "all6-lam1", "6e-4", s) for s in (2, 3)] + [("onum", "only-numbers", "6e-4", s) for s in range(4)] + \
           [("none24", "none", "2.4e-3", s) for s in (2, 3)]
    for sr, arm, lr, s in main:
        if (arm, lr, s) not in drop:
            write(out, sr, arm, lr, s)
    # picks.json = the pre-registered (unflagged) selection on the 12 sweep cells, by the frozen x1_code tool
    roots = [out / f"rb6tb-505m-2pct-k2-{a}-lr{lr}-s{s}" for a, lr in SWEEP for s in (0, 1)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    o = subprocess.run([PY, str(REALR / "x1_code/tools/rb6scale_select.py"), *map(str, roots), "--json", str(R / "picks.json")],
                       capture_output=True, text=True, env=env)
    assert o.returncode == 0, o.stderr[-500:]
    (R / "x1_code").symlink_to(REALR / "x1_code")
    rc = R / "readout_code"   # the WORKTREE read-out under test, snapshotted with its COMMIT (as the frozen dir)
    shutil.copytree(WT / "tools", rc / "tools")
    fs = ("rb6scale_summary.py", "rainbow6_summary.py", "make_rb6m_tasks.py", "make_rb6_tasks.py", "readout_rules.py", "rb6scale_select.py")
    (rc / "COMMIT").write_text("worktree (test); sha256[:12]: " + ", ".join(
        f"{f} {hashlib.sha256((rc / 'tools' / f).read_bytes()).hexdigest()[:12]}" for f in fs) + "\n")
    for sc in ("x1_select.sh", "readout_505m.sh"):
        shutil.copy(REALR / sc, R / sc)
    return R


FAKEBIN = W / "fakebin"


def sh(R, script, *args, env_extra=None, realmode=False, sq=None, Rarg=None):
    """run a script on the fake batch dir R. realmode: a copy of the script whose REAL is R and whose pinned picks.json
    sha is R's (so the real-dir branch, incl. the squeue check, runs); sq: a fake squeue ('fail' / 'busy' / 'idle')."""
    path = REALR / script
    if realmode:
        t = path.read_text().replace("REAL=/scratch/yh6210/llmadd_torch/runs/rb6_505m", f"REAL={R}")
        t = t.replace("PICKS_SHA=8d285d925c66", "PICKS_SHA=" + hashlib.sha256((R / "picks.json").read_bytes()).hexdigest()[:12])
        assert f"REAL={R}" in t
        path = W / f"realmode_{R.name}_{script}"
        path.write_text(t)
    env = dict(os.environ, R=str(Rarg or R), PICKS_SHA=hashlib.sha256((R / "picks.json").read_bytes()).hexdigest()[:12])
    if sq:
        env.update(PATH=f"{FAKEBIN}:{os.environ['PATH']}", FAKE_SQ=sq)
    env.update(env_extra or {})
    o = subprocess.run(["bash", str(path), *args], capture_output=True, text=True, env=env)
    return o.returncode, o.stdout + o.stderr


def readout(R, o, *extra, **kw):
    rc, t = sh(R, "readout_505m.sh", o, *extra, **kw)
    stem = R / f"readout_505m_x1{o.replace('-', '_')}{'_partial' if '--allow_partial' in extra else ''}"
    if rc == 0:
        txt = stem.with_suffix(".txt").read_text()
        js = json.loads(stem.with_suffix(".json").read_text())
    else:
        fl = sorted(R.glob(stem.name + ".failed-*.txt"))
        txt = fl[-1].read_text() if fl else ""
        js = None
    return rc, t, txt, js


def why(txt):
    ls = [l for l in txt.splitlines() if "AssertionError" in l]
    return ls[-1][:150] if ls else txt.strip().splitlines()[-1][:150] if txt.strip() else ""


shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
FAKEBIN.mkdir()
(FAKEBIN / "squeue").write_text('#!/bin/bash\ncase "$FAKE_SQ" in fail) echo "squeue: error: timeout" >&2; exit 1;; '
                                'busy) echo llmadd_rb6_505m-none-lr6e-4-s0;; *) echo some_other_job;; esac\n')
(FAKEBIN / "squeue").chmod(0o755)
# ---- B: X1 -> b
RB = build("B", 45.0)
pre = json.loads((RB / "picks.json").read_text())
say(pre["pick"] == {"none": "4.8e-3", "all6-lam1": "6e-4"} and pre["bracket"]["none"].startswith("OPEN at the lower") and "grid" not in pre,
    "B: the pre-registered select on the fake sweep -> picks.json none 4.8e-3 OPEN lower, all6 6e-4 (as the real one)")
rc, t = sh(RB, "x1_select.sh", "a", "b")
say(rc == 2 and "usage" in t, "x1_select.sh with a wrong argument: usage, exit 2")
rc, t = sh(RB, "x1_select.sh", env_extra={"PYTHONHOME": "/nonexistent"})
say(rc == 3 and "does not import" in t and not any((RB / f).exists() for f in ("picks_x1.json", "picks_x1.exit", "picks_x1.sha", "x1_select.log")),
    "S1.3 I-1: a broken environment (PYTHONHOME=/nonexistent): refused before the selection, nothing recorded")
rc, t = sh(RB, "x1_select.sh")
xj = json.loads((RB / "picks_x1.json").read_text())
say(rc == 0 and (RB / "picks_x1.exit").read_text().strip() == "0" and xj["status"] == "picked" and xj["pick"]["none"] == "2.4e-3"
    and (RB / "picks_x1.sha").read_text().strip() == hashlib.sha256((RB / "picks_x1.json").read_bytes()).hexdigest()[:12]
    and not os.access(RB / "picks_x1.json", os.W_OK) and "s0 and s1 present and finished" in t,
    "B: then x1_select.sh -> picks_x1.json picked none 2.4e-3, exit 0, sha recorded, read-only")
rc, t = sh(RB, "x1_select.sh")
say(rc == 3 and "runs once" in t, "B: x1_select.sh refuses a second run")
rc, t = sh(RB, "readout_505m.sh", "b", "--allow_partial", "x")
say(rc == 2 and "usage" in t, "S1.3 m-4: readout_505m.sh with a third argument: usage, exit 2")
rc, t, txt, js = readout(RB, "b", "--allow_partial")
say(rc != 0 and "with nothing missing: run it strict" in txt, f"S1.3 I-3: readout b --allow_partial with nothing missing: refused ({why(txt)[:80]})")
for o, want, reason in (("b", 0, None), ("b-pre", 0, None), ("a", 1, "needs none's pick 4.8e-3"), ("c", 1, "gives outcome b")):
    rc, t, txt, js = readout(RB, o)
    good = rc == want and f"rc {want}" in txt
    if want == 0:
        good &= (js["status"] == "complete" and js["x1"]["state"] == o and js["sizes"] == ["53m", "190m", "505m"]
                 and js["design"]["505m"]["none_nlr"] == ("2.4e-3" if o == "b" else "4.8e-3") and "505m picks bound" in txt
                 and "== readout_505m.sh (sha256[:12] " in txt and "== readout_code/COMMIT: " in txt)
    else:
        good &= reason in txt
    say(good, f"B: readout_505m.sh {o}: exit {rc} (want {want})" + (f"; reason: {why(txt)[:110]}" if want else "; header with the shas"))
rc, t, txt, js = readout(RB, "b")
say(rc == 3 and "not overwritten" in t, "B: readout_505m.sh b again: refused (a successful read-out is never overwritten)")
rc, t, txt, js = readout(RB, "c")
say(rc == 1 and "gives outcome b" in txt and len(list(RB.glob("readout_505m_x1c.failed-*.txt"))) >= 1,
    "S1.3 I-3: a refused read-out (c) can be re-run: kept as .failed-<time>, never locked")
RBt = W / "B_tamper"
shutil.copytree(RB, RBt, symlinks=True)
(RBt / "picks_x1.json").chmod(0o644)
d = json.loads((RBt / "picks_x1.json").read_text())
d["rule"] += " "
(RBt / "picks_x1.json").write_text(json.dumps(d, indent=1))
for f in RBt.glob("readout_505m_x1b.*"):
    f.chmod(0o644)
    f.unlink()
rc, t, txt, js = readout(RBt, "b")
say(rc == 3 and "picks_x1.json != picks_x1.sha" in t, "B: a picks_x1.json edited after x1_select.sh: refused (sha)")
# ---- the real-dir branch (copies of the scripts whose REAL is the fake dir)
for f in RB.glob("readout_505m_x1b_pre.*"):   # B's successful b-pre read-out: cleared so the no-overwrite check passes
    f.chmod(0o644)
    f.unlink()
qn = subprocess.run(["squeue", "-h", "-u", subprocess.run(["id", "-un"], capture_output=True, text=True).stdout.strip(), "-o", "%j"],
                    capture_output=True, text=True).stdout
n_real = sum(l.startswith("llmadd_rb6_505m-") for l in qn.splitlines())
rc, t, txt, js = readout(RB, "b-pre", realmode=True)
say((rc == 3 and f"{n_real} llmadd_rb6_505m jobs queued / running" in t) if n_real else rc == 3,
    f"S1.3 I-2: real-dir branch with the live squeue ({n_real} llmadd_rb6_505m jobs now): readout refused")
rc, t, txt, js = readout(RB, "b-pre", realmode=True, sq="fail")
say(rc == 3 and "squeue failed" in t, "S1.3 I-2: squeue failing (fake): readout refused, not passed")
rc, t = sh(RB, "x1_select.sh", realmode=True, sq="fail")
say(rc == 3 and ("squeue failed" in t or "runs once" in t), "S1.3 I-2: squeue failing (fake): x1_select refused")
rc, t, txt, js = readout(RB, "b-pre", realmode=True, sq="busy", Rarg=str(RB) + "//")
say(rc == 3 and "1 llmadd_rb6_505m jobs queued / running" in t,
    "S1.3 I-2: R spelled with a trailing '//' is still the real dir (realpath): the squeue check runs and refuses")
rc, t, txt, js = readout(RB, "b-pre", realmode=True, sq="idle", env_extra={"PICKS_SHA": "000000000000"})
say(rc == 0 and js["x1"]["state"] == "b-pre", "S1.3 I-2: real-dir branch with an idle queue (fake): runs with the pinned picks sha "
    "(a PICKS_SHA from the environment is ignored)")
# ---- Bp: none@2.4e-3 s2 / s3 missing -> the natural order: strict b, then b --allow_partial
RBp = build("Bp", 45.0, drop={("none", "2.4e-3", 2), ("none", "2.4e-3", 3)})
sh(RBp, "x1_select.sh")
rc, t, txt, js = readout(RBp, "b")
say(rc != 0 and "REFUSED (strict)" in txt and "505m none@2.4e-3 [2, 3]" in txt, "Bp: strict readout b without 2.4e-3 s2 / s3: refused")
rc, t, txt, js = readout(RBp, "b", "--allow_partial")
say(rc == 0 and js["status"] == "PROVISIONAL" and js["missing"]["505m"]["none_nlr"] == [2, 3] and "PROVISIONAL (n=2 of 4)" in txt
    and (RBp / "readout_505m_x1b_partial.txt").exists(),
    "S1.3 I-3: then readout b --allow_partial (no cleaning in between): PROVISIONAL -> readout_505m_x1b_partial.*")
RBq = build("Bq", 45.0, drop={("none", "2.4e-3", 2), ("none", "2.4e-3", 3), ("only-numbers", "6e-4", 3)})
sh(RBq, "x1_select.sh")
rc, t, txt, js = readout(RBq, "b", "--allow_partial")
say(rc != 0 and "allowed only under outcome b with nothing missing" in txt, "Bq: readout b --allow_partial with only-numbers s3 missing too: refused")
rc, t, txt, js = readout(RBq, "b-pre", "--allow_partial")
say(rc != 0 and "allowed only under outcome b" in txt, "Bq: readout b-pre --allow_partial: refused")
# ---- A: X1 -> a
RA = build("A", 30.0)
rc, t = sh(RA, "x1_select.sh")
xa = json.loads((RA / "picks_x1.json").read_text())
say(rc == 0 and xa["pick"]["none"] == "4.8e-3" and xa["bracket"]["none"] == "closed", "A: x1_select.sh -> picks_x1.json none 4.8e-3 closed")
for o, want, reason in (("a", 0, None), ("b", 1, "needs none's pick 2.4e-3"), ("b-pre", 1, "gives outcome a"), ("c", 1, "gives outcome a")):
    rc, t, txt, js = readout(RA, o)
    good = rc == want and (js["status"] == "complete" and js["x1"]["state"] == "a" and "bracket closed by X1" in txt if want == 0
                           else reason in txt)
    say(good, f"A: readout_505m.sh {o}: exit {rc} (want {want})" + (f"; reason: {why(txt)[:110]}" if want else ""))
rc, t, txt, js = readout(RA, "a", "--allow_partial")
say(rc != 0 and "allowed only under outcome b" in txt, "A: readout a --allow_partial: refused")
# ---- C: integrity failure in the X1 selection (data) -> c
RC = build("C", 45.0, edit24={1: lambda r: r["curve"][0].update(lm=r["curve"][0]["lm"] + 0.01)})
rc, t = sh(RC, "x1_select.sh")
say(rc == 1 and not (RC / "picks_x1.json").exists() and (RC / "picks_x1.exit").read_text().strip() == "1"
    and (RC / "picks_x1.sha").read_text().strip() == "absent" and "step-0 lm differs" in (RC / "x1_select.log").read_text(),
    "C: the integrity assert in rainbow6_summary.check is a data outcome: exit 1 recorded, no JSON")
for o, want, reason in (("c", 0, None), ("b-pre", 1, "gives outcome c"), ("a", 1, "No such file")):
    rc, t, txt, js = readout(RC, o)
    good = rc == want and (js["x1"]["state"] == "c" and js["x1"]["x1_json_status"] == "absent (X1 selection exit 1)" and "X1 incomplete" in txt
                           if want == 0 else reason in txt)
    say(good, f"C: readout_505m.sh {o}: exit {rc} (want {want})" + (f"; reason: {why(txt)[:110]}" if want else ""))
# ---- F: a NON-data failure inside the selection (an unreadable main-stage run JSON) -> nothing recorded, re-runnable
RF = build("F", 45.0)
bad = next((RF / "out" / "rb6tb-505m-2pct-k2-only-numbers-lr6e-4-s3").glob("*/run_*.json"))
good_text = bad.read_text()
bad.write_text(good_text[:1000])
rc, t = sh(RF, "x1_select.sh")
say(rc == 3 and "NON-data reason" in t and not any((RF / f).exists() for f in ("picks_x1.json", "picks_x1.exit", "picks_x1.sha", "x1_select.log"))
    and len(list(RF.glob("x1_select.failed-*.log"))) == 1,
    "S1.3 I-1: a crash that is not the integrity assert (JSONDecodeError): nothing recorded, the log kept as x1_select.failed-*")
bad.write_text(good_text)
rc, t = sh(RF, "x1_select.sh")
say(rc == 0 and json.loads((RF / "picks_x1.json").read_text())["pick"]["none"] == "2.4e-3", "F: after the fix x1_select.sh runs: b")
# ---- M: the X1 runs not final (s1 missing)
RM = build("M", 45.0)
shutil.rmtree(RM / "out" / "rb6tb-505m-2pct-k2-none-lr2.4e-3-s1")
rc, t = sh(RM, "x1_select.sh")
say(rc == 3 and "s1: no run JSON" in t and "not final" in t and not (RM / "picks_x1.exit").exists(),
    "S1.3 I-2(4): X1 s1 missing: x1_select.sh refuses without --x1-runs-final, nothing recorded")
rc, t = sh(RM, "x1_select.sh", "--x1-runs-final")
xm = json.loads((RM / "picks_x1.json").read_text())
say(rc == 0 and xm["status"] == "incomplete" and any("missing run none@2.4e-3 s1" in p for p in xm["problems"]),
    "M: with --x1-runs-final (resubmissions used up): the selection records incomplete (outcome c)")
rc, t, txt, js = readout(RM, "c")
say(rc == 0 and js["x1"]["x1_json_status"] == "incomplete", "M: readout c runs (X1 incomplete)")
# ---- N: NaN bpb at 2.4e-3 s1 (data)
RN = build("N", 45.0, edit24={1: lambda r: r["curve"][-1].update(val_bits_per_byte=float("nan"))})
rc, t = sh(RN, "x1_select.sh")
xn = json.loads((RN / "picks_x1.json").read_text())
say(rc == 0 and xn["pick"]["none"] == "2.4e-3" and math.isnan(xn["table"]["none"]["bpb"]["2.4e-3"][1]),
    "N: x1_select.sh -> b with a NaN bpb at 2.4e-3 s1 in the table")
rc, t, txt, js = readout(RN, "b")
say(rc == 0 and js["status"] == "complete" and "505m picks bound" in txt, "N: readout b: the binding takes NaN = NaN")
print("E2E S1.3 TEST", "PASS" if ok_all else "FAIL")
