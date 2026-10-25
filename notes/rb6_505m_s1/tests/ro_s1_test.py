"""Amendment S1 tests of tools/rb6scale_summary.py (505M mode), before any 505M main result was read. No real 505M metric is
read or printed:
- fake 505M runs = 53M F88 Burst runs of the same seed with the 505M size facts (as ro_x1_test): none@6e-4 <- rb6pb none,
  all6-lam1@6e-4 <- rb6pb all6-lam1, only-numbers@6e-4 <- rb6mb only-numbers, none@2.4e-3 and none@4.8e-3 <- rb6mb
  none-lr8e-3 (seeds 0-3);
- SYNTHETIC picks files (planted, obviously fake per-seed tables): P_pre = picks.json's form (no grid; none 4.8e-3 OPEN
  lower), P_a / P_b = picks_x1.json's form (X1 grid + amendment; a: none 4.8e-3 closed, b: none 2.4e-3 OPEN lower);
- the REAL picks.json and the real --grid_none outputs of the X1 tests (sel_work, rb6scale_select.py's own files) only
  through x1_check() in-process: format accepted / refused, nothing printed but PASS / FAIL.
Checks: outcomes a / b / b-pre / c (pick -> none_nlr, IGNORED cells, the S1 (i) role / lr-source lines, the S1 (ii) picks
block, cell notes, every none_nlr paired row's caveat, the side-by-side footnote, the JSON 'x1'); the values equal across
the outcomes (same source runs; only labels differ); refusals: no --x1, --x1 without --picks, an unknown outcome, (iv) a
picks file without grid for a / b and with grid for b-pre / c, (iii) wrong grid / amendment / a pick outside {2.4e-3,
4.8e-3} / all6 changed / bracket inconsistent, outcome vs pick mismatch.
    python ro_s1_test.py"""
import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path("/scratch/yh6210/llmadd-burst-scale")
RO = Path("/scratch/yh6210/llmadd-burst-ro/results")
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
X1W = HERE.parent / "x1" / "sel_work"
W = HERE / "ro_work"
SRC = {("none", "6e-4"): ["401-rb6pb-53m-2pct-k2-none-s0", "404-rb6pb-53m-2pct-k2-none-s1", "430-rb6pb-53m-2pct-k2-none-s2",
                          "431-rb6pb-53m-2pct-k2-none-s3"],
       ("all6-lam1", "6e-4"): ["419-rb6pb-53m-2pct-k2-all6-lam1-s0", "415-rb6pb-53m-2pct-k2-all6-lam1-s1",
                               "432-rb6pb-53m-2pct-k2-all6-lam1-s2", "433-rb6pb-53m-2pct-k2-all6-lam1-s3"],
       ("only-numbers", "6e-4"): [f"{n}-rb6mb-53m-2pct-k2-only-numbers-s{s}" for s, n in enumerate((438, 444, 450, 456))],
       ("none", "2.4e-3"): [f"{466 + s}-rb6mb-53m-2pct-k2-none-lr8e-3-s{s}" for s in range(4)],
       ("none", "4.8e-3"): [f"{466 + s}-rb6mb-53m-2pct-k2-none-lr8e-3-s{s}" for s in range(4)]}
sys.path.insert(0, str(WT / "tools"))
import rb6scale_summary as S   # noqa: E402
ok_all = True


def say(cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


def runjson(d):
    return next(Path(d).glob("*/run_*.json"))


def write(root, src_dir, arm, lr, s):
    r = json.loads(runjson(src_dir).read_text())
    assert r["args"]["seed"] == s and r["args"]["d_model"] == 384
    tag = f"rb6tb-505m-2pct-k2-{arm}-lr{lr}"
    tdir = root / f"{tag}-s{s}" / tag
    tdir.mkdir(parents=True)
    r["args"].update(tag=f"_{tag}", lr=float(lr), d_model=1024, n_layers=24, n_heads=16, d_ff=4096, out=str(tdir))
    r.update(n_params=505437184, grad_ckpt_on=True, geo_layers=[8, 16], read_layers=[8, 16], code_sha="93e22bf4c4c0")
    r["dep_sha"] = dict(r.get("dep_sha", {}), **{"toy_struct.py": "d182ce548bcc"})
    out = tdir / f"run_whole_{r['args']['target']}_seed{s}_{tag}.json"
    out.write_text(json.dumps(r))
    off = json.loads((Path(src_dir) / "offline_lm.json").read_text())
    v = dict(next(iter(off.values())), tag=f"_{tag}", code_sha=r["code_sha"])
    (tdir.parent / "offline_lm.json").write_text(json.dumps({str(out): v}))
    (tdir.parent / "check.json").write_text(json.dumps({"pass": True, "checks": [], "info": {}}))


OPEN = "OPEN at the lower edge (one more step outward only with the user's approval)"


def mk(pick_none, br_none, grid=None):
    """a synthetic picks file in rb6scale_select.py's format; per-seed tables planted (11.1 / 22.2 ..., bpb 1.01 ...)."""
    g = grid or S.PRE_GRID
    tab = {a: {"area": {lr: [11.1 + i, 22.2 + i] for i, lr in enumerate(g[a])},
               "bpb": {lr: [1.01 + i / 100, 1.02 + i / 100] for i, lr in enumerate(g[a])}, "diverged": {lr: [] for lr in g[a]}}
           for a in g}
    pk = {"size": "505m", "status": "picked", "rule": "fake", "problems": [], "pick": {"none": pick_none, "all6-lam1": "6e-4"},
          "bracket": {"none": br_none, "all6-lam1": "closed"}, "table": tab}
    if grid is not None:
        pk["grid"] = copy.deepcopy(grid)
        pk["amendment"] = "X1 (runs/rb6_505m/AUDIT.md): none's grid extended by --grid_none"
    return pk


def save(name, pk):
    p = W / f"{name}.json"
    p.write_text(json.dumps(pk))
    return p


def run(picks, x1=None, name="x", extra=()):
    js = W / f"out_{name}.json"
    js.unlink(missing_ok=True)
    cmd = [PY, "tools/rb6scale_summary.py", str(W / "fake505"), "--json", str(js), *extra]
    if picks:
        cmd += ["--picks", str(picks)]
    if x1:
        cmd += ["--x1", x1]
    o = subprocess.run(cmd, cwd=WT, capture_output=True, text=True, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    (W / f"out_{name}.txt").write_text(o.stdout + o.stderr)
    return o.returncode, o.stdout + o.stderr, (json.loads(js.read_text()) if js.exists() and o.returncode == 0 else None)


def sect(t, sz):
    """the per-size cell section '== <sz>: per cell ...' up to the next '=='."""
    m = re.search(rf"^== {sz}: per cell.*?(?=^==)", t, re.M | re.S)
    return m.group(0) if m else ""


shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
for (arm, lr), dirs in SRC.items():
    for s, d in enumerate(dirs):
        write(W / "fake505", RO / d, arm, lr, s)
X1G = S.X1_GRID
P_pre, P_a, P_b = save("P_pre", mk("4.8e-3", OPEN)), save("P_a", mk("4.8e-3", "closed", X1G)), save("P_b", mk("2.4e-3", OPEN, X1G))

# ---- the four outcomes
res = {}
for st, pf, nl, ign in (("a", P_a, "4.8e-3", "2.4e-3"), ("b", P_b, "2.4e-3", "4.8e-3"), ("b-pre", P_pre, "4.8e-3", "2.4e-3"),
                        ("c", P_pre, "4.8e-3", "2.4e-3")):
    rc, t, J = run(pf, st, name=st)
    res[st] = (t, J)
    lab = S.X1_STATES[st]["label"]
    sec = sect(t, "505m")
    gtxt = "{2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2}" if S.X1_STATES[st]["grid"] else "{4.8e-3, 9.6e-3, 1.92e-2}"
    nlr_rows = [r_ for f in ("S1b", "SE", "SH") for r_ in J["families"][f"505m|{f}"] if "none_nlr" in (r_["a"], r_["b"])] if J else []
    oth_rows = [r_ for k, v in (J["families"].items() if J else []) if k.startswith("505m|") for r_ in v
                if "none_nlr" not in (r_["a"], r_["b"])]
    other_sizes = [r_ for k, v in (J["families"].items() if J else []) if not k.startswith("505m|") for r_ in v]
    checks = {
        "rc 0, strict complete": rc == 0 and J and J["status"] == "complete",
        f"none_nlr = none@{nl}": J and J["design"]["505m"]["none_nlr"] == nl and J["labels"]["505m"]["none_nlr"] == f"none@{nl}",
        f"none@{ign} IGNORED (4)": sum(f"none-lr{ign}" in l and "IGNORED" in l for l in t.splitlines()) == 4,
        "(i) roles (505m) line, no pt2": any(l.startswith("  roles (505m): ") and "rb6_505m sweep (seeds 0-1)" in l and "pt2" not in l
                                             for l in t.splitlines()) and "  roles (53m, 190m): " in t,
        "(i) lr-source NOTEs: 190m pt2 / 505m this sweep + grid": ("NOTE (pre-registered): 190m: the lrs are reused from the pt2" in t
                                                                  and f"NOTE (Amendment S1): 505m: the lrs were selected on THIS batch's sweep" in t
                                                                  and f"none over {gtxt}" in t and "the Torch sizes' lrs" not in t),
        "(i) 505m cell header, no pt2": f"none@{nl}" in sec and "pt2" not in sec and "rb6_505m sweep (seeds 0-1)" in sec,
        "(ii) picks block (file, outcome, grid, amendment, bracket, per-seed)": (f"505m picks (Amendment S1): {pf}; X1 outcome {st}: {lab}" in t
                                                                               and f"    grid ({'Amendment X1 (--grid_none)' if S.X1_STATES[st]['grid'] else 'the pre-registered sweep grid'}): none {' '.join(json.loads(Path(pf).read_text()).get('grid', S.PRE_GRID)['none'])};" in t
                                                                               and "    bracket: none " in t and "    picks: none " in t),
        "(ii) none_nlr cell note": f"[{lab}; none's bracket: " in sec,
        "(ii) other 505m cells: all6's bracket": sec.count("[at all6-lam1's pick; its bracket: closed]") == 3,
        "(ii) every none_nlr row (S1b / SE / SH) carries the label; no other row": (len(nlr_rows) == 7 and all(lab in (r_["caveat"] or "") for r_ in nlr_rows)
                                                                                     and not any(lab in (r_.get("caveat") or "") for r_ in oth_rows + other_sizes)),
        "(ii) side-by-side footnote + final line": f"(505m: none@{nl} = " in t and f"== 505m X1 outcome {st} ({pf}): {lab}" in t,
        "JSON x1 / roles_505m / caveats": J and J["x1"]["state"] == st and J["x1"]["label"] == lab and "roles_505m" in J
                                          and J["caveats"]["lr_provenance"].startswith("190m: ") and "lr_provenance_505m" in J["caveats"]
                                          and J["cells"][f"505m|none_nlr|numbers"]["x1_note"].startswith(lab),
    }
    bad = [k for k, v in checks.items() if not v]
    say(not bad, f"outcome {st} ({pf.name}): " + ("; ".join(checks) if not bad else f"FAILED {bad}"))

strip = lambda J: ({k: {kk: vv for kk, vv in v.items() if kk not in ("label", "lr", "x1_note")} for k, v in J["cells"].items()},
                   {k: [{kk: vv for kk, vv in r_.items() if kk not in ("label", "caveat")} for r_ in v] for k, v in J["families"].items()})
say(all(res[s][1] for s in res) and strip(res["a"][1]) == strip(res["b"][1]) == strip(res["b-pre"][1]) == strip(res["c"][1]),
    "every cell value and paired statistic equal across a / b / b-pre / c (same source runs): S1 changes labels only")

# ---- refusals (exit != 0, the reason in the AssertionError)
bad = {"grid_no24": mk("4.8e-3", "closed", dict(X1G, none=["4.8e-3", "9.6e-3", "1.92e-2"])),
       "grid_extra": mk("4.8e-3", "closed", dict(X1G, none=["1.2e-3", "2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"])),
       "grid_all6": mk("4.8e-3", "closed", dict(X1G, **{"all6-lam1": ["3e-4", "6e-4"]})),
       "pick_96": mk("9.6e-3", "closed", X1G), "all6_pick": dict(mk("4.8e-3", "closed", X1G), pick={"none": "4.8e-3", "all6-lam1": "3e-4"}),
       "br_b_closed": mk("2.4e-3", "closed", X1G), "br_a_open": mk("4.8e-3", OPEN, X1G)}
noam = mk("4.8e-3", "closed", X1G)
del noam["amendment"]
bad["no_amendment"] = noam
cases = [("no --x1", P_pre, None, "needs --x1"), ("--x1 without --picks", None, "c", "it needs --picks"),
         ("unknown outcome", P_pre, "d", "needs --x1 one of"),
         ("(iv) a + picks.json (no grid)", P_pre, "a", "has no 'grid'"), ("(iv) b + picks.json (no grid)", P_pre, "b", "has no 'grid'"),
         ("b-pre + picks_x1 (grid)", P_a, "b-pre", "this file has 'grid'"), ("c + picks_x1 (grid)", P_b, "c", "this file has 'grid'"),
         ("a + the b file", P_b, "a", "needs none's pick 4.8e-3"), ("b + the a file", P_a, "b", "needs none's pick 2.4e-3"),
         ("(iii) grid without 2.4e-3", save("bad_grid_no24", bad["grid_no24"]), "a", "grid is not Amendment X1's"),
         ("(iii) grid with an extra lr", save("bad_grid_extra", bad["grid_extra"]), "a", "grid is not Amendment X1's"),
         ("(iii) all6 grid changed", save("bad_grid_all6", bad["grid_all6"]), "a", "grid is not Amendment X1's"),
         ("(iii) pick 9.6e-3 in an X1 file", save("bad_pick_96", bad["pick_96"]), "a", "X1 reading 3"),
         ("(iii) no amendment", save("bad_no_amendment", bad["no_amendment"]), "a", "not an Amendment X1 picks file"),
         ("all6 pick changed", save("bad_all6_pick", bad["all6_pick"]), "a", "all6-lam1 untouched"),
         ("b pick with a closed bracket", save("bad_br_b_closed", bad["br_b_closed"]), "b", "none's bracket"),
         ("a pick with an OPEN bracket", save("bad_br_a_open", bad["br_a_open"]), "a", "none's bracket")]
for lab_, pf, st, why in cases:
    rc, t, J = run(pf, st, name="ref_" + re.sub(r"\W+", "_", lab_))
    say(rc != 0 and J is None and "AssertionError" in t and why in t, f"refused: {lab_} (exit {rc}; '{why}')")

# ---- the real files' format, in-process (values never printed)
real = json.loads(Path("/scratch/yh6210/llmadd_torch/runs/rb6_505m/picks.json").read_text())
xb = json.loads((X1W / "b_clear_flag.json").read_text())
xa = json.loads((X1W / "a_lower_flag.json").read_text())


def acc(pk, st):
    try:
        S.x1_check(pk, st)
        return True
    except AssertionError:
        return False


say(acc(real, "b-pre") and acc(real, "c") and not acc(real, "a") and not acc(real, "b"),
    "the real runs/rb6_505m/picks.json: accepted for b-pre / c, refused for a / b (no grid)")
say(acc(xb, "b") and not acc(xb, "a") and not acc(xb, "c") and acc(xa, "a") and not acc(xa, "b") and not acc(xa, "b-pre"),
    "rb6scale_select.py --grid_none's own outputs (X1 tests' fake b / a): accepted for b / a only")
print("RO S1 TEST", "PASS" if ok_all else "FAIL")
