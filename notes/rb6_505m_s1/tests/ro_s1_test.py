"""Amendment S1 + S1.1 + S1.2 tests of tools/rb6scale_summary.py (505M mode), before any 505M main result was read. No
real 505M metric is read or printed:
- fake 505M runs = 53M Burst runs of the same seed with the 505M size facts: none@6e-4 <- rb6pb none, all6-lam1@6e-4 <-
  rb6pb all6-lam1, only-numbers@6e-4 <- rb6mb only-numbers, none@4.8e-3 <- rb6mb none-lr8e-3, none@2.4e-3 <- rb6mb
  none-lr3e-3 (a DIFFERENT source than 4.8e-3, so the b vs b-pre BH q values differ: build review I3);
- every read-out has the command's roots: the fake 505m dir + the real rb6_190m/out + the 16 53M F88 Burst dirs (absolute);
- SYNTHETIC picks files in fake batch dirs (the fixed names picks.json / picks_x1.json / picks_x1.exit, S1.2): the picks'
  seeds 0-1 tables come from the fake runs (binding, I4), every other lr is planted low so the replayed rule gives the
  pick (S1.2); end-to-end runs with tool-written picks files are in e2e_s12.py;
- the REAL picks.json and rb6scale_select.py's own X1 outputs (x1/sel_work) only in-process via x1_check / x1_derive;
  nothing printed but PASS / FAIL.
Checks: outcomes a / b / b-pre / c (absent X1 file + nonzero exit record) / c (incomplete X1 file): the cells, the (i) /
(ii) lines, the X1 selection line, the binding line, labels on the right lines (b-pre: every 505M line), the none@6e-4
note, JSON 'x1' / 'roles'; the 53m / 190m sections identical to the 190M read-out without --picks; values a = b-pre = c,
b's none cell from its own source. Refusals: S1's, S1.1's and S1.2's (fixed path, exit record, incomplete grid, replayed
rule, --allow_partial outside b).
    python ro_s1_test.py"""
import copy
import glob
import hashlib
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
R190 = "/scratch/yh6210/llmadd_torch/runs/rb6_190m/out"
PATS = ["401-rb6pb-*", "404-rb6pb-*", "430-rb6pb-*", "431-rb6pb-*", "419-rb6pb-*", "415-rb6pb-*", "432-rb6pb-*", "433-rb6pb-*",
        "438-rb6mb-*", "444-rb6mb-*", "450-rb6mb-*", "456-rb6mb-*", "46[6-9]-rb6mb-*"]
B53 = sorted(p for pt in PATS for p in glob.glob(f"{RO}/{pt}"))
assert len(B53) == 16, B53
SRC = {("none", "6e-4"): ["401-rb6pb-53m-2pct-k2-none-s0", "404-rb6pb-53m-2pct-k2-none-s1", "430-rb6pb-53m-2pct-k2-none-s2",
                          "431-rb6pb-53m-2pct-k2-none-s3"],
       ("all6-lam1", "6e-4"): ["419-rb6pb-53m-2pct-k2-all6-lam1-s0", "415-rb6pb-53m-2pct-k2-all6-lam1-s1",
                               "432-rb6pb-53m-2pct-k2-all6-lam1-s2", "433-rb6pb-53m-2pct-k2-all6-lam1-s3"],
       ("only-numbers", "6e-4"): [f"{n}-rb6mb-53m-2pct-k2-only-numbers-s{s}" for s, n in enumerate((438, 444, 450, 456))],
       ("none", "4.8e-3"): [f"{466 + s}-rb6mb-53m-2pct-k2-none-lr8e-3-s{s}" for s in range(4)],
       ("none", "2.4e-3"): [f"{n}-rb6mb-53m-2pct-k2-none-lr3e-3-s{s}" for s, n in enumerate((502, 504, 506, 479))]}
sys.path.insert(0, str(WT / "tools"))
import rb6scale_summary as S   # noqa: E402
import rainbow6_summary as R6  # noqa: E402
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
    return r


OPEN = "OPEN at the lower edge (one more step outward only with the user's approval)"
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
FAKE = {}
for (arm, lr), dirs in SRC.items():
    for s, d in enumerate(dirs):
        FAKE[(arm, lr, s)] = write(W / "fake505", RO / d, arm, lr, s)
sel = lambda arm, lr: {"area": [R6.metrics(FAKE[(arm, lr, s)], "numbers")["area"] for s in (0, 1)],
                       "bpb": [R6.final_bpb(FAKE[(arm, lr, s)]) for s in (0, 1)]}




def mk(pick_none, br_none, grid=None, status="picked"):
    """a synthetic picks file in rb6scale_select.py's format: the picks' seeds 0-1 = the fake runs (binding); every other
    lr planted at area 0 / bpb 2.0, so the pre-registered rule replayed on the table gives the pick (S1.2)."""
    g = grid or S.PRE_GRID
    tab = {a: {"area": {lr: [0.0, 0.0] for lr in g[a]}, "bpb": {lr: [2.0, 2.0] for lr in g[a]}, "diverged": {lr: [] for lr in g[a]}}
           for a in g}
    pk = {"size": "505m", "status": status, "rule": "fake", "problems": [], "pick": {}, "bracket": {}, "table": tab}
    if status == "picked":
        pk["pick"] = {"none": pick_none, "all6-lam1": "6e-4"}
        pk["bracket"] = {"none": br_none, "all6-lam1": "closed"}
        for a, lr in (("none", pick_none), ("all6-lam1", "6e-4")):
            if (a, lr, 0) not in FAKE or lr not in g[a]:   # a pick with no fake run (a refusal case): planted
                continue
            v = sel(a, lr)
            tab[a]["area"][lr], tab[a]["bpb"][lr] = v["area"], v["bpb"]
    if grid is not None:
        pk["grid"] = copy.deepcopy(grid)
        pk["amendment"] = "X1 (runs/rb6_505m/AUDIT.md): none's grid extended by --grid_none"
    return pk


def scen(name, picks=None, x1=None, exit_=None, other=None):
    """a fake batch dir: picks.json / picks_x1.json / picks_x1.exit (the fixed names, S1.2) / other.json."""
    d = W / f"sc_{name}"
    d.mkdir(parents=True, exist_ok=True)
    for fn, pk in (("picks.json", picks), ("picks_x1.json", x1), ("other.json", other)):
        if pk is not None:
            (d / fn).write_text(json.dumps(pk))
    if exit_ is not None:
        (d / "picks_x1.exit").write_text(exit_ + "\n")
    return d


def run(picks=None, x1=None, x1j=None, name="x", roots=None, extra=()):
    js = W / f"out_{name}.json"
    js.unlink(missing_ok=True)
    cmd = [PY, "tools/rb6scale_summary.py", *(roots or [str(W / "fake505"), R190, *B53]), "--json", str(js), *extra]
    for k, v in (("--picks", picks), ("--x1", x1), ("--x1_json", x1j)):
        if v is not None:
            cmd += [k, str(v)]
    o = subprocess.run(cmd, cwd=WT, capture_output=True, text=True, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    (W / f"out_{name}.txt").write_text(o.stdout + o.stderr)
    return o.returncode, o.stdout + o.stderr, (json.loads(js.read_text()) if js.exists() and o.returncode == 0 else None)


def blocks(txt, sz):
    return re.findall(rf"^== {sz}: .*?(?=^==)", txt, re.M | re.S)


X1G = S.X1_GRID
P_pre, P_a, P_b = mk("4.8e-3", OPEN), mk("4.8e-3", "closed", X1G), mk("2.4e-3", OPEN, X1G)
P_inc = mk(None, None, X1G, status="incomplete")
REF190 = (HERE / "reg" / "r190_new.txt").read_text()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
sA, sB = scen("a", x1=P_a), scen("b", x1=P_b)
sBP, sCA, sCI = scen("bpre", picks=P_pre, x1=P_b), scen("cabs", picks=P_pre, exit_="1"), scen("cinc", picks=P_pre, x1=P_inc)

# ---- the outcomes
res = {}
for nm, st, d, pfn, nl, ign, xstat in (("a", "a", sA, "picks_x1.json", "4.8e-3", "2.4e-3", "picked"),
                                       ("b", "b", sB, "picks_x1.json", "2.4e-3", "4.8e-3", "picked"),
                                       ("b-pre", "b-pre", sBP, "picks.json", "4.8e-3", "2.4e-3", "picked"),
                                       ("c_absent", "c", sCA, "picks.json", "4.8e-3", "2.4e-3", "absent (X1 selection exit 1)"),
                                       ("c_incomplete", "c", sCI, "picks.json", "4.8e-3", "2.4e-3", "incomplete")):
    pf, xj = d / pfn, d / "picks_x1.json"
    rc, t, J = run(pf, st, xj, name=nm)
    res[nm] = (t, J)
    lab = S.X1_STATES[st]["label"]
    sec = blocks(t, "505m")[0] if blocks(t, "505m") else ""
    gtxt = "{2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2}" if S.X1_STATES[st]["grid"] else "{4.8e-3, 9.6e-3, 1.92e-2}"
    rows505 = [r_ for k, v in (J["families"].items() if J else []) if k.startswith("505m|") for r_ in v]
    nlr_rows = [r_ for r_ in rows505 if "none_nlr" in (r_["a"], r_["b"])]
    oth_rows = [r_ for r_ in rows505 if "none_nlr" not in (r_["a"], r_["b"])]
    small = [r_ for k, v in (J["families"].items() if J else []) if not k.startswith("505m|") for r_ in v]
    checks = {
        "rc 0, strict complete, 3 sizes": rc == 0 and J and J["status"] == "complete" and J["sizes"] == ["53m", "190m", "505m"],
        f"none_nlr = none@{nl}": J and J["design"]["505m"]["none_nlr"] == nl and J["labels"]["505m"]["none_nlr"] == f"none@{nl}",
        f"none@{ign} IGNORED (4)": sum(f"none-lr{ign}" in l and "IGNORED" in l for l in t.splitlines()) == 4,
        "(i) roles / lr-source lines": ("  roles (53m, 190m): " in t and any(l.startswith("  roles (505m): ") and "pt2" not in l for l in t.splitlines())
                                        and "NOTE (pre-registered): 190m: the lrs are reused from the pt2" in t
                                        and "NOTE (Amendment S1): 505m: the lrs were selected on THIS batch's sweep" in t
                                        and f"none over {gtxt}" in t and "the Torch sizes' lrs" not in t),
        "(i) 505m section without pt2": bool(sec) and "pt2" not in sec and "rb6_505m sweep (seeds 0-1)" in sec,
        "(ii) picks block with sha": f"505m picks (Amendment S1): {pf} (sha256[:12] {sha(pf)}); X1 outcome {st}: {lab}" in t,
        "X1 selection line": (f"X1 selection output (--x1_json): {xj} (status {xstat}, sha256[:12] "
                              f"{sha(xj) if xj.exists() else None})" in t),
        "binding line": f"505m picks bound: the picks file's seeds 0-1 numbers area / in-training bpb = the loaded runs at none@{nl}" in t,
        "(ii) none_nlr cell note": f"[{lab}; none's bracket: " in sec,
        "m1 none@6e-4 note": "[none at all6-lam1's pick: not a grid point of none, does not close none's bracket; all6-lam1's bracket: closed" in sec,
        "(ii)/I3 labelled rows": (len(nlr_rows) == 7 and all(lab in (r_["caveat"] or "") for r_ in nlr_rows)
                                  and (all(lab in (r_["caveat"] or "") for r_ in oth_rows) if st == "b-pre"
                                       else not any(lab in (r_.get("caveat") or "") for r_ in oth_rows))
                                  and not any(lab in (r_.get("caveat") or "") for r_ in small)),
        "I3 b-pre cell headers all labelled": st != "b-pre" or sec.count(lab) >= 4,
        "footnote + final line": f"(505m: none@{nl} = " in t and f"== 505m X1 outcome {st} ({pf}): {lab}" in t,
        "JSON x1 / roles (m2)": (J and J["x1"]["state"] == st and J["x1"]["x1_json_status"] == xstat
                                 and set(J["roles"]) == {"53m,190m", "505m"} and "pt2" not in json.dumps(J["roles"]["505m"])
                                 and J["cells"]["505m|none_nlr|numbers"]["x1_note"].startswith(lab)),
        "53m / 190m sections = the 190M read-out": all(blocks(t, sz) == blocks(REF190, sz) and len(blocks(REF190, sz)) == 2
                                                       for sz in ("53m", "190m")),
    }
    bad = [k for k, v in checks.items() if not v]
    say(not bad, f"outcome {nm} ({d.name}: --picks {pfn} --x1 {st}): " + ("; ".join(checks) if not bad else f"FAILED {bad}"))

strip = lambda J: ({k: {kk: vv for kk, vv in v.items() if kk not in ("label", "lr", "x1_note")} for k, v in J["cells"].items()},
                   {k: [{kk: vv for kk, vv in r_.items() if kk not in ("label", "caveat")} for r_ in v] for k, v in J["families"].items()})
Js = {k: v[1] for k, v in res.items()}
say(all(Js.values()) and strip(Js["a"]) == strip(Js["b-pre"]) == strip(Js["c_absent"]) == strip(Js["c_incomplete"]),
    "values equal across a / b-pre / c (none_nlr = none@4.8e-3 in all): S1 changes labels only")
nb, na = Js["b"]["cells"]["505m|none_nlr|numbers"]["values"], Js["a"]["cells"]["505m|none_nlr|numbers"]["values"]
src24 = [R6.metrics(FAKE[("none", "2.4e-3", s)], "numbers")["area"] for s in range(4)]
say(nb == src24 and nb != na, "b reads the none@2.4e-3 runs (its own source, != a's none@4.8e-3 values)")
qb = {r_["m"]: r_["p_bh"] for r_ in Js["b"]["families"]["505m|SE"] if r_["b"] == "none_plr"}
qp = {r_["m"]: r_["p_bh"] for r_ in Js["b-pre"]["families"]["505m|SE"] if r_["b"] == "none_plr"}
say(qb != qp and all(S.X1_STATES["b-pre"]["label"] in (r_["caveat"] or "") for r_ in Js["b-pre"]["families"]["505m|SE"]),
    f"I3: b vs b-pre BH q of the SE none_plr rows differ ({qb} vs {qp}) and under b-pre those rows carry the SECONDARY label")

# ---- refusals (exit != 0, the reason in the AssertionError); each case: (label, picks file, --x1, --x1_json, reason)
B = {"grid_no24": mk("4.8e-3", "closed", dict(X1G, none=["4.8e-3", "9.6e-3", "1.92e-2"])),
     "grid_extra": mk("4.8e-3", "closed", dict(X1G, none=["1.2e-3", "2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"])),
     "grid_all6": mk("4.8e-3", "closed", dict(X1G, **{"all6-lam1": ["3e-4", "6e-4"]})),
     "pick_96": mk("9.6e-3", "closed", X1G), "all6_pick": dict(mk("4.8e-3", "closed", X1G), pick={"none": "4.8e-3", "all6-lam1": "3e-4"}),
     "br_b_closed": mk("2.4e-3", "closed", X1G), "br_a_open": mk("4.8e-3", OPEN, X1G)}
B["no_amendment"] = {k: v for k, v in mk("4.8e-3", "closed", X1G).items() if k != "amendment"}
rp = mk("2.4e-3", OPEN, X1G)   # replay: 4.8e-3's table clearly better than the stated pick 2.4e-3
rp["table"]["none"]["area"]["4.8e-3"] = [v + 5 for v in rp["table"]["none"]["area"]["2.4e-3"]]
inc_bad = dict(P_inc, grid=dict(X1G, none=["4.8e-3", "9.6e-3", "1.92e-2"]))
pl = mk("2.4e-3", OPEN, X1G)
pl["table"]["none"]["area"]["2.4e-3"] = [v + 0.5 for v in pl["table"]["none"]["area"]["2.4e-3"]]
x1p = lambda d: d / "picks_x1.json"
cases = [("no --x1", sBP / "picks.json", None, None, "needs --x1"),
         ("--x1 without --picks", None, "c", None, "they need --picks"),
         ("--x1_json without --picks", None, None, x1p(sB), "they need --picks"),
         ("unknown outcome", sBP / "picks.json", "d", None, "needs --x1 one of"),
         ("(iv) a + picks.json (no grid)", sBP / "picks.json", "a", x1p(sBP), "has no 'grid'"),
         ("(iv) b + picks.json (no grid)", sBP / "picks.json", "b", x1p(sBP), "has no 'grid'"),
         ("b-pre + picks_x1 (grid)", x1p(sA), "b-pre", x1p(sA), "this file has 'grid'"),
         ("c + picks_x1 (grid)", x1p(sB), "c", x1p(sB), "this file has 'grid'"),
         ("a + the b file", x1p(sB), "a", x1p(sB), "needs none's pick 4.8e-3"),
         ("b + the a file", x1p(sA), "b", x1p(sA), "needs none's pick 2.4e-3")]
for k, why, st in (("grid_no24", "grid is not Amendment X1's", "a"), ("grid_extra", "grid is not Amendment X1's", "a"),
                   ("grid_all6", "grid is not Amendment X1's", "a"), ("pick_96", "X1 reading 3", "a"),
                   ("no_amendment", "not an Amendment X1 picks file", "a"), ("all6_pick", "all6-lam1 untouched", "a"),
                   ("br_b_closed", "none's bracket", "b"), ("br_a_open", "none's bracket", "a")):
    d = scen(f"bad_{k}", x1=B[k])
    cases.append((f"(iii) {k}", x1p(d), st, x1p(d), why))
d = scen("replay", x1=rp)
cases.append(("S1.2 replay: the table's rule gives 4.8e-3, the file says 2.4e-3", x1p(d), "b", x1p(d), "selection rule on the picks file's own table"))
cases += [("S1.1 no --x1_json", sCA / "picks.json", "c", None, "needs --x1_json"),
          ("S1.1 c but the X1 output says b", sBP / "picks.json", "c", x1p(sBP), "gives outcome b"),
          ("S1.1 c but the X1 output says a", scen("c_says_a", picks=P_pre, x1=P_a) / "picks.json", "c", x1p(W / "sc_c_says_a"), "gives outcome a"),
          ("S1.1 b-pre but the X1 output says a", W / "sc_c_says_a" / "picks.json", "b-pre", x1p(W / "sc_c_says_a"), "gives outcome a"),
          ("S1.1 b-pre but the X1 output is absent", sCA / "picks.json", "b-pre", x1p(sCA), "gives outcome c"),
          ("S1.1 b but the X1 output is incomplete", scen("b_inc", x1=P_inc) / "picks_x1.json", "b", x1p(W / "sc_b_inc"), "status"),
          ("S1.1 a: --picks is another file than --x1_json", scen("a_other", x1=P_a, other=dict(P_a, rule="fake2")) / "other.json", "a",
           x1p(W / "sc_a_other"), "is not the X1 selection's output"),
          ("S1.1 --x1_json without grid (picks.json's form)", scen("x1_nogrid", picks=P_pre, x1=P_pre) / "picks.json", "c",
           x1p(W / "sc_x1_nogrid"), "has no 'grid': not written by"),
          ("S1.1 --x1_json picked with a wrong grid", scen("x1_badgrid", picks=P_pre, x1=B["grid_no24"]) / "picks.json", "c",
           x1p(W / "sc_x1_badgrid"), "grid is not Amendment X1's"),
          ("S1.2 --x1_json not picks_x1.json next to --picks", sBP / "picks.json", "b-pre", x1p(sB), "fixed output picks_x1.json next to --picks"),
          ("S1.2 X1 output absent, no picks_x1.exit", scen("noexit", picks=P_pre) / "picks.json", "c", x1p(W / "sc_noexit"),
           "does not record the X1 selection's exit"),
          ("S1.2 X1 output absent, picks_x1.exit 0", scen("exit0", picks=P_pre, exit_="0") / "picks.json", "c", x1p(W / "sc_exit0"),
           "not a nonzero exit"),
          ("S1.2 X1 output present, picks_x1.exit nonzero", scen("exit1json", picks=P_pre, x1=P_b, exit_="1") / "picks.json", "b-pre",
           x1p(W / "sc_exit1json"), "exists but"),
          ("S1.2 incomplete X1 output with a wrong grid", scen("incbad", picks=P_pre, x1=inc_bad) / "picks.json", "c", x1p(W / "sc_incbad"),
           "an incomplete X1 selection output without Amendment X1's grid")]
for lab_, pf, st, xj, why in cases:
    rc, t, J = run(pf, st, xj, name="ref_" + re.sub(r"\W+", "_", lab_))
    say(rc != 0 and J is None and "AssertionError" in t and why in t, f"refused: {lab_} (exit {rc}; '{why}')")
# refusals that need the runs loaded
d = scen("planted", x1=pl)
rc, t, J = run(x1p(d), "b", x1p(d), name="ref_binding")
say(rc != 0 and "AssertionError" in t and "the picks file's table != the loaded run at 505m none@2.4e-3 s0" in t,
    f"refused (I4): a picks table that is not the loaded runs' (none@2.4e-3 area + 0.5) (exit {rc})")
rc, t, J = run(x1p(sA), "a", x1p(sA), name="ref_sizes", roots=[str(W / "fake505"), R190])
say(rc != 0 and "AssertionError" in t and "sizes ['190m', '505m'] != 53m, 190m, 505m" in t, f"refused (I1): the 53M roots missing (exit {rc})")
rc, t, J = run(x1p(sA), "a", x1p(sA), name="ref_sizes_rel", roots=[str(W / "fake505"), R190, "results/401-rb6pb-*"])
say(rc != 0 and "sizes ['190m', '505m'] != 53m, 190m, 505m" in t, f"refused (I1): a 53M glob relative to another cwd (exit {rc})")
rc, t, J = run(x1p(sA), "a", x1p(sA), name="ref_partial_a", extra=("--allow_partial",))
say(rc != 0 and "allowed only under outcome b" in t, f"refused (S1.2): --allow_partial under a (exit {rc})")
rc, t, J = run(x1p(sB), "b", x1p(sB), name="partial_b_full", extra=("--allow_partial",))
say(rc != 0 and "with nothing missing: run it strict" in t, f"refused (S1.3): --allow_partial under b with nothing missing (exit {rc})")

# ---- the real files, in-process (values never printed)
real = json.loads(Path("/scratch/yh6210/llmadd_torch/runs/rb6_505m/picks.json").read_text())


def acc(fn, *a):
    try:
        return fn(*a)
    except AssertionError:
        return False


say(acc(S.x1_check, real, "b-pre") and acc(S.x1_check, real, "c") and not acc(S.x1_check, real, "a")
    and not acc(S.x1_check, real, "b"), "the real runs/rb6_505m/picks.json: x1_check (incl. the replayed rule) accepts b-pre / c, refuses a / b")
dv = {}
for k, f in (("b", "b_clear_flag.json"), ("a", "a_lower_flag.json"), ("inc", "missing_s1.json")):
    d = W / f"sc_real_{k}"
    d.mkdir()
    shutil.copy(X1W / f, d / "picks_x1.json")
    dv[k] = acc(S.x1_derive, d / "picks_x1.json", d / "picks.json")
dabs = W / "sc_real_abs"
dabs.mkdir()
(dabs / "picks_x1.exit").write_text("1\n")
say(dv["b"][0] == "b" and dv["a"][0] == "a" and dv["inc"][0] == "c" and dv["inc"][2] == "incomplete"
    and acc(S.x1_derive, dabs / "picks_x1.json", dabs / "picks.json") == ("c", None, "absent (X1 selection exit 1)"),
    "x1_derive on rb6scale_select.py's own X1 outputs: b / a / incomplete -> b / a / c; absent with exit 1 -> c")
print("RO S1 TEST", "PASS" if ok_all else "FAIL")
