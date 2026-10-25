"""Amendment S1.2 end-to-end test of the two operator scripts on FAKE batch dirs (no 505M value is read or printed):
runs/rb6_505m/x1_select.sh (the frozen x1_code select, --grid_none) and runs/rb6_505m/readout_505m.sh (the 505M main
read-out, here with the WORKTREE summary under test, snapshotted into the fake dir with its COMMIT).
Each fake batch dir R has out/ = the 12 sweep + 2 X1 + main-stage + X1 b follow-up cells, each a 53M Burst run of the
same seed relabelled with the 505M size facts (none <- rb6mb none-lr8e-3, none@2.4e-3 <- rb6mb none-lr3e-3, none@6e-4 <-
rb6pb none, all6-lam1 <- rb6pb all6-lam1, only-numbers <- rb6mb only-numbers). The seeds 0-1 numbers area and last-eval
bpb of the selection cells are planted. picks.json = the pre-registered (unflagged) select on the 12 sweep cells, so
every picks file is written by the real tool. Scenarios:
  B  X1 -> b (2.4e-3 area 45 > 4.8e-3 40): readout b / b-pre run; a / c refused; no overwrite; x1_select runs once;
     a tampered picks_x1.json is refused (sha); partial: without 2.4e-3 s2 / s3, strict refused, --allow_partial
     PROVISIONAL; partial with another gap refused
  A  X1 -> a (2.4e-3 area 30): readout a runs; b / b-pre / c and a --allow_partial refused
  C  X1 integrity failure (2.4e-3 s1 step-0 lm + 0.01): the selection exits nonzero, no JSON; readout c runs
     (absent with the recorded exit); b-pre / a refused
  N  as B with 2.4e-3 s1's last-eval bpb NaN (data): X1 -> b; readout b runs (NaN = NaN in the binding)
    python e2e_s12.py"""
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
W = HERE / "e2e"
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


def sh(R, script, *args):
    env = dict(os.environ, R=str(R), PICKS_SHA=hashlib.sha256((R / "picks.json").read_bytes()).hexdigest()[:12])
    o = subprocess.run(["bash", str(REALR / script), *args], capture_output=True, text=True, env=env)
    return o.returncode, o.stdout + o.stderr


def readout(R, o, *extra):
    rc, t = sh(R, "readout_505m.sh", o, *extra)
    stem = R / f"readout_505m_x1{o.replace('-', '_')}"
    txt = (stem.with_suffix(".txt")).read_text() if stem.with_suffix(".txt").exists() else ""
    js = json.loads(stem.with_suffix(".json").read_text()) if stem.with_suffix(".json").exists() and rc == 0 else None
    return rc, t, txt, js


def clean(R, o):
    for suf in (".txt", ".json"):
        p = (R / f"readout_505m_x1{o.replace('-', '_')}").with_suffix(suf)
        if p.exists():
            p.chmod(0o644)
            p.unlink()


shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
pre = None
# ---- B
RB = build("B", 45.0)
pre = json.loads((RB / "picks.json").read_text())
say(pre["pick"] == {"none": "4.8e-3", "all6-lam1": "6e-4"} and pre["bracket"]["none"].startswith("OPEN at the lower") and "grid" not in pre,
    "B: the pre-registered select on the fake sweep -> picks.json none 4.8e-3 OPEN lower, all6 6e-4 (as the real one)")
rc, t = sh(RB, "x1_select.sh")
xj = json.loads((RB / "picks_x1.json").read_text())
say(rc == 0 and (RB / "picks_x1.exit").read_text().strip() == "0" and xj["status"] == "picked" and xj["pick"]["none"] == "2.4e-3"
    and (RB / "picks_x1.sha").read_text().strip() == hashlib.sha256((RB / "picks_x1.json").read_bytes()).hexdigest()[:12]
    and not os.access(RB / "picks_x1.json", os.W_OK), "B: x1_select.sh -> picks_x1.json picked none 2.4e-3, exit 0, sha recorded, read-only")
rc, t = sh(RB, "x1_select.sh")
say(rc == 3 and "the X1 selection runs once" in t, "B: x1_select.sh refuses a second run")
for o, want in (("b", 0), ("b-pre", 0), ("a", 1), ("c", 1)):
    rc, t, txt, js = readout(RB, o)
    good = rc == want and f"rc {want}" in txt
    if want == 0:
        good &= js["status"] == "complete" and js["x1"]["state"] == o and js["sizes"] == ["53m", "190m", "505m"] \
            and js["design"]["505m"]["none_nlr"] == ("2.4e-3" if o == "b" else "4.8e-3") and "505m picks bound" in txt
    else:
        good &= f"gives outcome b" in txt or "needs none's pick" in txt or "has no 'grid'" in txt or "this file has 'grid'" in txt
    say(good, f"B: readout_505m.sh {o}: exit {rc} (want {want})" + ("" if want == 0 else f"; reason: {[l for l in txt.splitlines() if 'AssertionError' in l][-1][:120] if 'AssertionError' in txt else txt[-120:]}"))
rc, t, txt, js = readout(RB, "b")
say(rc == 3 and "not overwritten" in t, "B: readout_505m.sh b again: refused (no overwrite)")
clean(RB, "b")
RBt = W / "B_tamper"
shutil.copytree(RB, RBt, symlinks=True)
(RBt / "picks_x1.json").chmod(0o644)
d = json.loads((RBt / "picks_x1.json").read_text())
d["rule"] += " "
(RBt / "picks_x1.json").write_text(json.dumps(d, indent=1))
for f in ("readout_505m_x1b_pre.txt", "readout_505m_x1b_pre.json"):
    if (RBt / f).exists():
        (RBt / f).chmod(0o644)
        (RBt / f).unlink()
rc, t, txt, js = readout(RBt, "b")
say(rc == 3 and "picks_x1.json != picks_x1.sha" in t, "B: a picks_x1.json edited after x1_select.sh: refused (sha)")
# ---- B partial: none@2.4e-3 s2 / s3 missing
RBp = build("Bp", 45.0, drop={("none", "2.4e-3", 2), ("none", "2.4e-3", 3)})
sh(RBp, "x1_select.sh")
rc, t, txt, js = readout(RBp, "b")
say(rc != 0 and "REFUSED (strict)" in txt and "505m none@2.4e-3 [2, 3]" in txt, "Bp: strict readout b without 2.4e-3 s2 / s3: refused")
clean(RBp, "b")
rc, t, txt, js = readout(RBp, "b", "--allow_partial")
say(rc == 0 and js["status"] == "PROVISIONAL" and js["missing"]["505m"]["none_nlr"] == [2, 3] and "PROVISIONAL (n=2 of 4)" in txt,
    "Bp: readout b --allow_partial: PROVISIONAL, only none@2.4e-3 [2, 3] missing")
RBq = build("Bq", 45.0, drop={("none", "2.4e-3", 2), ("none", "2.4e-3", 3), ("only-numbers", "6e-4", 3)})
sh(RBq, "x1_select.sh")
rc, t, txt, js = readout(RBq, "b", "--allow_partial")
say(rc != 0 and "allowed only under outcome b with nothing missing" in txt,
    "Bq: readout b --allow_partial with only-numbers s3 missing too: refused (S1.2)")
rc, t, txt, js = readout(RBq, "b-pre", "--allow_partial")
say(rc != 0 and "allowed only under outcome b" in txt, "Bq: readout b-pre --allow_partial: refused (S1.2)")
# ---- A
RA = build("A", 30.0)
rc, t = sh(RA, "x1_select.sh")
xa = json.loads((RA / "picks_x1.json").read_text())
say(rc == 0 and xa["pick"]["none"] == "4.8e-3" and xa["bracket"]["none"] == "closed", "A: x1_select.sh -> picks_x1.json none 4.8e-3 closed")
for o, want in (("a", 0), ("b", 1), ("b-pre", 1), ("c", 1)):
    rc, t, txt, js = readout(RA, o)
    good = rc == want and (want != 0 or (js["status"] == "complete" and js["x1"]["state"] == "a"
                                         and "bracket closed by X1" in txt))
    say(good, f"A: readout_505m.sh {o}: exit {rc} (want {want})")
clean(RA, "a")
rc, t, txt, js = readout(RA, "a", "--allow_partial")
say(rc != 0 and "allowed only under outcome b" in txt, "A: readout a --allow_partial: refused (S1.2)")
# ---- C: integrity failure in the X1 selection
RC = build("C", 45.0, edit24={1: lambda r: r["curve"][0].update(lm=r["curve"][0]["lm"] + 0.01)})
rc, t = sh(RC, "x1_select.sh")
say(rc == 0 and not (RC / "picks_x1.json").exists() and (RC / "picks_x1.exit").read_text().strip() not in ("", "0")
    and (RC / "picks_x1.sha").read_text().strip() == "absent" and "step-0 lm differs" in (RC / "x1_select.log").read_text(),
    f"C: x1_select.sh: the selection stops on the integrity assert (exit {(RC / 'picks_x1.exit').read_text().strip()}), no JSON")
for o, want in (("c", 0), ("b-pre", 1), ("a", 1)):
    rc, t, txt, js = readout(RC, o)
    good = rc == want and (want != 0 or (js["x1"]["state"] == "c" and js["x1"]["x1_json_status"].startswith("absent (X1 selection exit ")
                                         and "X1 incomplete" in txt))
    say(good, f"C: readout_505m.sh {o}: exit {rc} (want {want})")
# ---- N: NaN bpb of a diverged-like 2.4e-3 s1 (data)
RN = build("N", 45.0, edit24={1: lambda r: r["curve"][-1].update(val_bits_per_byte=float("nan"))})
rc, t = sh(RN, "x1_select.sh")
xn = json.loads((RN / "picks_x1.json").read_text())
say(rc == 0 and xn["pick"]["none"] == "2.4e-3" and math.isnan(xn["table"]["none"]["bpb"]["2.4e-3"][1]),
    "N: x1_select.sh -> b with a NaN bpb at 2.4e-3 s1 in the table")
rc, t, txt, js = readout(RN, "b")
say(rc == 0 and js["status"] == "complete" and "505m picks bound" in txt, "N: readout b: the binding takes NaN = NaN")
print("E2E S1.2 TEST", "PASS" if ok_all else "FAIL")
