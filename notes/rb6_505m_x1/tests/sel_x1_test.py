"""Amendment X1 tests of tools/rb6scale_select.py --grid_none (before any 2.4e-3 result exists).
Part 1 (regression, default = no flag): the old file (git HEAD f9c1435, = the batch snapshot's c2a7e222957c) and the new
one on the same inputs -> stdout and JSON byte-identical: (a) the 12 sweep dirs (also = the selection job's select.log
lines and runs/rb6_505m/picks.json byte for byte), (b) runs/rb6_505m/out as the selection job ran it (now incl. the
main-stage runs: NOTE lines).
Part 2 (the flag takes effect, fake data): the 12 real sweep runs + fake none@2.4e-3 s0 / s1 = the real none@4.8e-3 s0 / s1
relabelled (lr / tag / out) with planted numbers areas (held_b moved linearly, the trainer's metrics recomputed) and
planted in-training bpb; known answers per scenario; the grid printed / stored; invalid flags refused; the
pre-registered picks.json never overwritten.
    python sel_x1_test.py"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path("/scratch/yh6210/llmadd-burst-scale")
NEW = WT / "tools" / "rb6scale_select.py"
OLD = HERE / "old_tools" / "rb6scale_select.py"
RB = Path("/scratch/yh6210/llmadd_torch/runs/rb6_505m")
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
W = HERE / "sel_work"
X1 = "2.4e-3,4.8e-3,9.6e-3,1.92e-2"
SWEEP = [f"rb6tb-505m-2pct-k2-{a}-lr{lr}-s{s}" for a, lrs in (("none", ("4.8e-3", "9.6e-3", "1.92e-2")),
         ("all6-lam1", ("3e-4", "6e-4", "1.2e-3"))) for lr in lrs for s in (0, 1)]
sys.path.insert(0, str(WT / "tools"))
import rainbow6_summary as R6   # noqa: E402

ok_all = True


def say(cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


def run(script, roots, extra=(), name="x"):
    js = W / f"{name}.json"
    o = subprocess.run([PY, str(script), *map(str, roots), "--json", str(js), *extra], capture_output=True, text=True)
    (W / f"{name}.txt").write_text(o.stdout + o.stderr)
    return o.returncode, o.stdout, o.stderr, (js.read_bytes() if js.exists() else None)


shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
sweep_dirs = [RB / "out" / n for n in SWEEP]
assert all(d.is_dir() for d in sweep_dirs)

# ---- Part 1: regression without the flag
for lab, roots in (("12 sweep dirs", sweep_dirs), ("runs/rb6_505m/out", [RB / "out"])):
    k = lab.split()[0].replace("/", "_")
    rc_o, so, se_o, jo = run(OLD, roots, name=f"reg_old_{k}")
    rc_n, sn, se_n, jn = run(NEW, roots, name=f"reg_new_{k}")
    say(rc_o == 0 and rc_n == 0 and so == sn and jo == jn and jo is not None,
        f"default regression on {lab}: old / new stdout byte-identical ({len(sn.splitlines())} lines) and JSON byte-identical "
        f"({len(jn or b'')} bytes); rc {rc_o} / {rc_n}")
    if k == "12":
        log = (RB / "select.log").read_text().splitlines()
        i0 = next(i for i, l in enumerate(log) if l.startswith("505M rainbow sweep selection"))
        i1 = next(i for i, l in enumerate(log) if l.startswith("selection status"))
        say(sn.splitlines() == log[i0:i1], f"new stdout on the 12 sweep dirs = the selection job's select.log lines {i0 + 1}-{i1} "
            "(job 20599806, the batch snapshot's code)")
        say(jn == (RB / "picks.json").read_bytes(), "new JSON on the 12 sweep dirs = runs/rb6_505m/picks.json byte for byte")


# ---- Part 2: fake none@2.4e-3 runs
def runjson(d):
    return next(Path(d).glob("*/run_*.json"))


def plant(r, area, bpb):
    """numbers held_b moved linearly so that the numbers area = area; the last eval's val_bits_per_byte = bpb."""
    cur = R6.metrics(r, "numbers")["area"]
    if area >= cur:
        g = (100 - area) / (100 - cur)
        f = lambda v: 1 - (1 - v) * g
    else:
        f = lambda v, c=area / cur: v * c
    for e in r["curve"]:
        for b in R6.B:
            e[f"held_{b}"] = f(e[f"held_{b}"])
    got = R6.metrics(r, "numbers")
    assert abs(got["area"] - area) < 1e-6, (got["area"], area)
    r["metrics"]["numbers"] = dict(r["metrics"]["numbers"], **{k: got[k] for k in ("area", "final", "final3", "t50", "t90", "n_evals")})
    r["curve"][-1]["val_bits_per_byte"] = bpb


def fake(root, s, area, bpb, edit=None, check=None, lr="2.4e-3"):
    src = RB / "out" / f"rb6tb-505m-2pct-k2-none-lr4.8e-3-s{s}"
    r = json.loads(runjson(src).read_text())
    assert r["args"]["seed"] == s and r["args"]["lr"] == 4.8e-3
    tag = f"rb6tb-505m-2pct-k2-none-lr{lr}"
    tdir = root / f"{tag}-s{s}" / tag
    tdir.mkdir(parents=True)
    r["args"].update(lr=float(lr), tag=f"_{tag}", out=str(tdir))
    plant(r, area, bpb)
    if edit:
        edit(r)
    (tdir / f"run_whole_none_seed{s}_{tag}.json").write_text(json.dumps(r))
    shutil.copy(src / "offline_lm.json", tdir.parent / "offline_lm.json")
    (tdir.parent / "check.json").write_text(json.dumps(check) if check is not None else (src / "check.json").read_text())


def scen(name, per_seed, kw=None):
    """per_seed: {seed: (area, bpb)} of the fake none@2.4e-3 runs; -> its root."""
    root = W / f"fake_{name}"
    for s, (a, b) in per_seed.items():
        fake(root, s, a, b, **(kw or {}).get(s, {}))
    return root


def picks(jb):
    return json.loads(jb) if jb else None


# the real sweep values (from picks.json, the recorded selection)
P0 = json.loads((RB / "picks.json").read_text())["table"]["none"]
b48 = P0["bpb"]["4.8e-3"]   # per seed
CASES = [
    # name, per-seed (area, bpb), expected pick with the flag, expected bracket start, expected tied
    ("b_clear", {0: (40.0, 1.13), 1: (40.0, 1.13)}, "2.4e-3", "OPEN at the lower", ["2.4e-3"]),
    ("a_lower", {0: (30.0, 1.10), 1: (30.0, 1.10)}, "4.8e-3", "closed", ["4.8e-3", "9.6e-3"]),
    ("tie_lowbpb", {0: (37.0, 1.10), 1: (37.0, 1.10)}, "2.4e-3", "OPEN at the lower", ["2.4e-3", "4.8e-3", "9.6e-3"]),
    ("tie_highbpb", {0: (37.0, 1.20), 1: (37.0, 1.20)}, "4.8e-3", "closed", ["2.4e-3", "4.8e-3", "9.6e-3"]),
    ("tie_samebpb", {0: (36.4, b48[0]), 1: (36.4, b48[1])}, "2.4e-3", "OPEN at the lower", ["2.4e-3", "4.8e-3", "9.6e-3"]),
    ("diverged", {0: (37.0, 1.10), 1: (37.0, float("nan"))}, "4.8e-3", "closed", ["2.4e-3", "4.8e-3", "9.6e-3"]),
]
for name, ps, want, br, tied in CASES:
    kw = {}
    if name == "diverged":   # s1 diverges at eval 60: lm / val_loss / bpb non-finite from there (data: area counts, bpb worst)
        def dv(r):
            for e in r["curve"][60:]:
                e.update(lm=float("nan"), val_loss=float("nan"), val_bits_per_byte=float("nan"))
        kw = {1: dict(edit=dv)}
    root = scen(name, ps, kw)
    rc, so, se, jb = run(NEW, sweep_dirs + [root], ("--grid_none", X1), name=f"{name}_flag")
    J = picks(jb)
    good = (rc == 0 and J and J["status"] == "picked" and J["pick"]["none"] == want and J["bracket"]["none"].startswith(br)
            and J["table"]["none"]["tied"] == tied and J["pick"]["all6-lam1"] == "6e-4" and J["bracket"]["all6-lam1"] == "closed"
            and J["grid"]["none"] == X1.split(",") and so.startswith("GRID (--grid_none, Amendment X1): none 2.4e-3 4.8e-3 9.6e-3 "
                                                                       "1.92e-2;"))
    extra = ""
    if name == "diverged":
        good &= "DIVERGED seeds [1] (data)" in so and math.isinf(J["table"]["none"]["mean_bpb"]["2.4e-3"])
        extra = "; s1 DIVERGED (data), its bpb the worst (inf)"
    say(good, f"{name}: with --grid_none pick none {J and J['pick'].get('none')} (want {want}), bracket "
        f"'{J and J['bracket'].get('none')}', tied {J and J['table']['none'].get('tied')}; all6 6e-4 closed; grid printed + stored{extra}")
    rc2, so2, _, jb2 = run(NEW, sweep_dirs + [root], name=f"{name}_noflag")
    J2 = picks(jb2)
    say(rc2 == 0 and J2["pick"]["none"] == "4.8e-3" and J2["bracket"]["none"].startswith("OPEN at the lower")
        and "grid" not in J2 and "GRID" not in so2 and so2.count("NOTE not a sweep run (ignored): rb6tb-505m-2pct-k2-none-lr2.4e-3") == 2,
        f"{name}: same data WITHOUT the flag -> pick 4.8e-3 OPEN lower (the 2.4e-3 runs 'not a sweep run (ignored)'): "
        + ("the flag changes the selection" if want != "4.8e-3" else "the flag changes the bracket (OPEN -> closed)"))

# incomplete / invalid X1 data -> no picks (outcome c)
inc = [("missing_s1", {0: (40.0, 1.13)}, {}, "missing run none@2.4e-3 s1"),
       ("failcheck_s0", {0: (40.0, 1.13), 1: (40.0, 1.13)}, {0: dict(check={"pass": False, "checks": [{"check": "x", "ok": False}],
                                                                            "info": {}})}, "its check FAILED"),
       ("unfinished_s0", {0: (40.0, 1.13), 1: (40.0, 1.13)}, {0: dict(edit=lambda r: r["curve"].__delitem__(slice(70, None)))},
        "unfinished (last eval"),
       ("wronglr_s1", {0: (40.0, 1.13), 1: (40.0, 1.13)}, {1: dict(edit=lambda r: r["args"].update(lr=4.8e-3))},
        "trainer args differ from the intent")]
for name, ps, kw, why in inc:
    root = scen(name, ps, kw)
    rc, so, se, jb = run(NEW, sweep_dirs + [root], ("--grid_none", X1), name=name)
    J = picks(jb)
    say(rc == 0 and J["status"] == "incomplete" and J["pick"] == {} and "INCOMPLETE: no picks" in so
        and any(why in p for p in J["problems"]), f"{name}: incomplete, no picks; problem '{why}'")
rc, so, se, jb = run(NEW, sweep_dirs, ("--grid_none", X1), name="no_x1_runs")
J = picks(jb)
say(J["status"] == "incomplete" and sum("missing run none@2.4e-3" in p for p in J["problems"]) == 2,
    "flag on the 12 sweep runs only: incomplete (2 missing none@2.4e-3 runs)")

# invalid flag values are refused (exit != 0, no JSON)
root = W / "fake_b_clear"
for name, g in (("missing_preregistered", "2.4e-3,4.8e-3,9.6e-3"), ("not_ascending", "4.8e-3,2.4e-3,9.6e-3,1.92e-2"),
                ("duplicate", "2.4e-3,2.4e-3,4.8e-3,9.6e-3,1.92e-2"), ("respelled", "2.4e-3,0.0048,9.6e-3,1.92e-2"),
                ("nonpositive", "0,4.8e-3,9.6e-3,1.92e-2")):
    rc, so, se, jb = run(NEW, sweep_dirs + [root], ("--grid_none", g), name=f"bad_{name}")
    say(rc != 0 and jb is None and "AssertionError" in se, f"--grid_none {g} ({name}) refused (exit {rc}, no JSON)")

# the pre-registered picks.json is never overwritten with the flag; an X1 JSON may be
cp = W / "picks_copy.json"
shutil.copy(RB / "picks.json", cp)
o = subprocess.run([PY, str(NEW), *map(str, sweep_dirs + [root]), "--json", str(cp), "--grid_none", X1], capture_output=True, text=True)
say(o.returncode != 0 and "written without --grid_none" in o.stderr and cp.read_bytes() == (RB / "picks.json").read_bytes(),
    "a JSON written without the flag (copy of picks.json) is not overwritten with --grid_none")
x1j = W / "b_clear_flag.json"
before = x1j.read_bytes()
o = subprocess.run([PY, str(NEW), *map(str, sweep_dirs + [root]), "--json", str(x1j), "--grid_none", X1], capture_output=True, text=True)
say(o.returncode == 0 and x1j.read_bytes() == before, "an existing X1 JSON is rewritten (same inputs -> same bytes)")
print("SELECT X1 TEST", "PASS" if ok_all else "FAIL")
