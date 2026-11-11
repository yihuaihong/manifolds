"""Amendment A2 (supersedes the A1 version of this test, kept as tests/aux_gate_test_a1.py with its log tests/aux_gate_test.out):
the auxiliary-loss gate per (arm, lr) CELL with the LM share by norm a / (a + b) (a = grad_lm, b = grad_prior), on synthetic
run JSONs (nothing is written into runs/ptdom; the HOLD file of the tests is a temp path).
  select_ptdom.py --stage A on stage-A roots (92 runs, readout_ptdom.fake_run), incl. the A1 auditor's counter-examples
  (tests/audit_a1.md, important 1): CX1 the picked lr at 0.3 with the others at 0.9; CX1b 5 of 12 runs at 0.3; CX2 b = 1.5 a
  at cos 0 (A1's a / |g_T| = 0.555 passed); CX3 opposing gradients b = 1.2 a at cos -0.9 (A1's a / |g_T| = 1.89 passed) and the
  boundary b = a; plus: every cell of an arm failing (-> GATE units, HOLD, exit 2), a failing argmax cell that closes the open
  side, a missing grad_prior at an on-window eval.
  aux_gate_check.py on PARTIAL curves: CX6 (one cell at 0.3 among 0.9 cells: A1's pooled median 0.9 passed), CX7 (no
  grad_cos at any on-window eval: A1 exit 0), a missing grad_prior, opposing gradients, a logged share off its
  re-derivation, a truncated JSON, a run before the window, a -r1 re-run dir superseding its first dir.
  The quantity on the REAL smoke run JSONs (re-derived independently from the JSON fields). The real runs/ptdom/HOLD is never
  created."""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

R = Path("/scratch/yh6210/llmadd_torch/runs/ptdom")
sys.path.insert(0, str(R))
import readout_ptdom as RO   # noqa: E402

PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
ok = []
hold_before = (R / "HOLD").exists()


def T(name, cond, val=None):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val is not None else ""))


def stageA(root, kw_of, area_of=None):
    """92 stage-A runs; kw_of(arm, lr, s) -> fake_run gradient kwargs; areas peak at 1e-3 (prior arms) / 8e-3 (none: inside its
    grid, so no none unit is open) unless area_of is given."""
    for arm in RO.ARMS:
        for lr in RO.GRID[arm]:
            for s in RO.SEL:
                a = area_of(arm, lr, s) if area_of else {d: 30.0 + (5.0 if lr == PEAK[arm == "none"] else 0.0) + 0.01 * s for d in RO.SIX}
                RO.fake_run(root, arm, lr, s, a, **kw_of(arm, lr, s))


PEAK = {False: "1e-3", True: "8e-3"}


def select(tmp, tag, kw_of, area_of=None):
    """A fresh stage-A root under tmp/tag, then select_ptdom --stage A with temp --out / --hold."""
    root = tmp / tag / "out"
    stageA(root, kw_of, area_of)
    out, hold = tmp / tag / "picks_A.json", tmp / tag / "HOLD"
    o = subprocess.run([PY, str(R / "select_ptdom.py"), "--stage", "A", "--root", str(root), "--out", str(out), "--hold", str(hold)],
                       capture_output=True, text=True)
    return o, (json.loads(out.read_text()) if out.exists() else None), hold


def partial(root, arm, lr, seed, last_step, share, clip=1.0, cos=0.0, logged_off=0.0, missing=(), rr=0):
    """A partial run JSON as the trainer writes it at every eval: evals at 0, 250, .., last_step; on-window (prior_on) from
    step 916 on; a = grad_lm 1, b = grad_prior 1 / share - 1 (a / (a + b) = share), grad_cos cos, the logged grad_lm_share
    = a / sqrt(a^2 + b^2 + 2ab cos) + logged_off; the fields in `missing` absent at the on-window evals; rr: a -r<rr> dir."""
    tag = f"ptdom-53m-k2-{arm}-lr{lr}"
    d = Path(root) / (f"{tag}-s{seed}" + (f"-r{rr}" if rr else "")) / tag
    d.mkdir(parents=True, exist_ok=True)
    gp = 1 / share - 1
    lg = 1 / math.sqrt(1 + gp * gp + 2 * gp * cos) + logged_off
    c = []
    for st in range(0, last_step + 1, 250):
        e = {"step": st, "lm": 9.0, "val_loss": 5.0, "geo": 0.5, "prior_on": int(st >= 916), "grad_lm": 1.0,
             "grad_prior": gp if st >= 916 else None, "grad_cos": cos if st >= 916 else None,
             "grad_lm_share": lg if st >= 916 else 1.0, "clip_coef": clip if st >= 916 else 1.0}
        if st >= 916:
            for k in missing:
                e.pop(k)
        c.append(e)
    f = d / f"run_whole_none_seed{seed}_{tag}.json"
    f.write_text(json.dumps({"curve": c, "args": {}}))
    return f


def cell_line(stdout, key):
    return next((l for l in stdout.splitlines() if l.strip().startswith(key + " ")), "")


star = lambda f: (lambda arm, lr, s: f(lr, s) if arm == "only-stars" else {"lm_share": 0.9})
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # ---------------------------------------------------------------- select_ptdom (A2 (i))
    # CX1: only-stars@1e-3 = the area argmax (35 vs 30; inside the grid, closed) at 0.3 in all 4 seeds, the others at 0.9
    o, pk, hold = select(tmp, "cx1", star(lambda lr, s: {"lm_share": 0.3 if lr == "1e-3" else 0.9}))
    u = pk["units"]["only-stars|stars"] if pk else {}
    T("CX1 (A1 passed it: arm median 0.900): only-stars@1e-3 (the area argmax) at 0.3 -> its cell FAILS and is excluded; the "
      "pick moves to 5e-4 (5e-4 and 2e-3 tied, bpb tie-break), decision C; exit 0, no HOLD; the picks file lists the failing cell",
      o.returncode == 0 and "FAIL: median LM share a / (a + b) 0.300 < 0.5" in cell_line(o.stdout, "only-stars@1e-3")
      and "excluded from their arm's selection): only-stars@1e-3" in o.stdout and (u.get("pick"), u.get("decision"), u.get("excluded"))
      == ("5e-4", "C", ["1e-3"]) and pk["aux_gate_failed_cells"] == ["only-stars@1e-3"] and pk["amendment"] == "A2" and not hold.exists(),
      (o.returncode, u.get("pick"), u.get("decision"), o.stderr[-300:]))
    print("    " + cell_line(o.stdout, "only-stars@1e-3").strip())
    print("    " + next((l for l in o.stdout.splitlines() if l.startswith("only-stars|stars")), ""))
    # CX1b: 5 of the 12 only-stars runs at 0.3 (1e-3 seeds 0-2, 2e-3 seeds 0-1)
    o, pk, _ = select(tmp, "cx1b", star(lambda lr, s: {"lm_share": 0.3 if (lr == "1e-3" and s < 3) or (lr == "2e-3" and s < 2) else 0.9}))
    T("CX1b (A1 passed it): 5 of 12 runs at 0.3 -> only-stars@1e-3 (3 of 4 at 0.3: median 0.300) FAILS; only-stars@2e-3 (2 of 4: "
      "median (0.3 + 0.9) / 2 = 0.600) passes: the median of a cell's runs is the rule", o.returncode == 0
      and pk["aux_gate_failed_cells"] == ["only-stars@1e-3"] and "median LM share 0.600" in cell_line(o.stdout, "only-stars@2e-3")
      and "FAIL" not in cell_line(o.stdout, "only-stars@2e-3") and pk["units"]["only-stars|stars"]["pick"] != "1e-3")
    # CX2: b = 1.5 a at cos 0 -> a / (a + b) = 0.4 (A1's a / |g_T| = 1 / sqrt(1 + 2.25) = 0.555 passed)
    o, pk, _ = select(tmp, "cx2", star(lambda lr, s: {"lm_share": 0.4} if lr == "1e-3" else {"lm_share": 0.9}))
    T(f"CX2: prior gradient 1.5 x the LM's (cos 0; A1's share {1 / math.sqrt(3.25):.3f} passed) -> a / (a + b) 0.400 FAILS; b > a on "
      "100% of its on-window evals", o.returncode == 0 and pk["aux_gate_failed_cells"] == ["only-stars@1e-3"]
      and "median LM share 0.400" in cell_line(o.stdout, "only-stars@1e-3") and "b > a 100%" in cell_line(o.stdout, "only-stars@1e-3"))
    # CX3: opposing gradients b = 1.2 a at cos -0.9 (A1's a / |g_T| = 1.89 passed); boundary b = a at cos -0.9 (0.5: passes)
    o, pk, _ = select(tmp, "cx3", star(lambda lr, s: {"lm_share": 1 / 2.2, "cos": -0.9} if lr == "1e-3" else
                                       {"lm_share": 0.5, "cos": -0.9} if lr == "2e-3" else {"lm_share": 0.9}))
    a1_share = 1 / math.sqrt(1 + 1.44 - 2 * 1.2 * 0.9)
    T(f"CX3: opposing gradients b = 1.2 a, cos -0.9 (A1's share {a1_share:.2f} > 1 passed) -> a / (a + b) 0.455 FAILS; the boundary "
      "b = a (a / (a + b) = 0.5 exactly, cos -0.9) passes (not < 0.5; b > a on 0%)", o.returncode == 0
      and pk["aux_gate_failed_cells"] == ["only-stars@1e-3"] and "median LM share 0.455" in cell_line(o.stdout, "only-stars@1e-3")
      and "FAIL" not in cell_line(o.stdout, "only-stars@2e-3") and "median LM share 0.500" in cell_line(o.stdout, "only-stars@2e-3")
      and "b > a 0%" in cell_line(o.stdout, "only-stars@2e-3"))
    # every cell of only-stars fails -> AUX GATE FAILED, GATE units (prior and the domain's none unit), HOLD, exit 2
    o, pk, hold = select(tmp, "all", star(lambda lr, s: {"lm_share": 0.3}))
    T("every only-stars cell at 0.3 -> 'AUX GATE FAILED: only-stars', exit 2, the --hold file written naming only-stars only, the "
      "picks file records only-stars|stars and none|stars as GATE (no pick; no X / C for the domain); the other units decided",
      o.returncode == 2 and "AUX GATE FAILED: only-stars" in o.stdout and hold.exists() and "only-stars" in hold.read_text()
      and "only-months" not in hold.read_text().split("prior arm(s)")[1].split(":")[0] and pk is not None
      and pk["GATE"] == ["none|stars", "only-stars|stars"] and pk["units"]["only-stars|stars"]["pick"] is None
      and pk["aux_gate_arms_all_failed"] == ["only-stars"] and pk["hold_written"] and not any(c[0] == "only-stars" for c in pk["C"] + pk["X"])
      and ["only-months", "1e-3"] in pk["C"], (o.returncode, o.stdout[-600:], o.stderr[-300:]))
    # a failing area-argmax cell closes the open side: only-stars 5e-4 / 1e-3 / 2e-3 = 30 / 35 / 40 (stars), 2e-3 at 0.3 -> pick
    # 1e-3, open upper by > 2 toward 2e-3 = a failing cell -> closed: C at 1e-3, no X (without the gate: 2e-3 open -> X 4e-3)
    area_c = lambda arm, lr, s: {d: (({"5e-4": 30.0, "1e-3": 35.0, "2e-3": 40.0}[lr] if arm == "only-stars" and d == "stars" else
                                      30.0 + (5.0 if lr == PEAK[arm == "none"] else 0.0)) + 0.01 * s) for d in RO.SIX}
    o, pk, _ = select(tmp, "close", star(lambda lr, s: {"lm_share": 0.3 if lr == "2e-3" else 0.9}), area_c)
    u = pk["units"]["only-stars|stars"]
    T("a failing area-argmax cell (only-stars@2e-3, 40) -> pick 1e-3, open upper by > 2 but the next lr is that failing cell -> "
      "closed by the gate: C at 1e-3, no X", o.returncode == 0 and (u["pick"], u["decision"], u["grid"]) == ("1e-3", "C", ["5e-4", "1e-3"])
      and bool(u["closed_by_gate"]) and "closed by the gate" in o.stdout and pk["X"] == [], (u.get("pick"), u.get("decision"), pk["X"]))
    o0, pk0, _ = select(tmp, "close0", lambda arm, lr, s: {"lm_share": 0.9}, area_c)
    T("  the same areas without the gate failure -> only-stars picks 2e-3, open upper -> X 4e-3 (the gate changed the decision)",
      (pk0["units"]["only-stars|stars"]["pick"], pk0["units"]["only-stars|stars"]["decision"], pk0["X"]) == ("2e-3", "X", [["only-stars", "4e-3"]]))
    # a missing grad_prior at the on-window evals of one run -> its cell fails (unverifiable)
    o, pk, _ = select(tmp, "miss", star(lambda lr, s: {"lm_share": 0.9, "missing": ("grad_prior",) if (lr == "1e-3" and s == 0) else ()}))
    T("a run with no grad_prior at its on-window evals (prior_on 1) -> 'missing gradient fields (grad_prior)', its cell fails "
      "(A1 skipped such evals)", o.returncode == 0 and pk["aux_gate_failed_cells"] == ["only-stars@1e-3"]
      and "s0: missing gradient fields (grad_prior) at 26 of 26 on-window evals" in cell_line(o.stdout, "only-stars@1e-3"))
    # the all-pass control
    o, pk, hold = select(tmp, "pass", lambda arm, lr, s: {"lm_share": 0.9})
    T("control: 0.9 everywhere -> 'AUX GATE passed (every cell)', 18 prior cells printed with b > a / clip < 1, a decision "
      "(amendment A2, the per-cell gate summary in it), no HOLD", o.returncode == 0 and "AUX GATE passed (every cell)" in o.stdout
      and len(pk["aux_gate"]) == 18 and abs(pk["aux_gate"]["only-stars@1e-3"]["median"] - 0.9) < 1e-12 and pk["amendment"] == "A2"
      and not hold.exists() and sum("clip < 1" in l for l in o.stdout.splitlines()) == 18)
    # ---------------------------------------------------------------- aux_gate_check (partial runs)
    run = lambda r_: subprocess.run([PY, str(R / "aux_gate_check.py"), "--root", str(r_)], capture_output=True, text=True)
    r6 = tmp / "c6"
    partial(r6, "only-stars", "5e-4", 0, 1750, 0.9); partial(r6, "only-stars", "1e-3", 0, 1750, 0.9)
    partial(r6, "only-stars", "2e-3", 0, 1750, 0.3, clip=0.6)
    partial(r6, "none", "8e-3", 0, 1750, 0.2)   # none runs are not read (no prior)
    o = run(r6)
    T("CX6 (A1 pooled median 0.900 -> exit 0): only-stars@2e-3 s0 at 0.3 -> its cell FAILS -> exit 1; the 0.9 cells pass; per cell "
      "b > a and clip < 1 printed; none not read", o.returncode == 1 and "FAIL" in cell_line(o.stdout, "only-stars@2e-3")
      and "FAIL" not in cell_line(o.stdout, "only-stars@1e-3") and "b > a 100%, clip < 1 100%" in cell_line(o.stdout, "only-stars@2e-3")
      and "none@" not in o.stdout and "on-window evals   4" in cell_line(o.stdout, "only-stars@2e-3"), o.stdout[-700:])
    print("    " + o.stdout.strip().replace("\n", "\n    "))
    r7 = tmp / "c7"
    for s in (0, 1):
        partial(r7, "only-stars", "1e-3", s, 1500, 0.9, missing=("grad_cos",))
    o = run(r7)
    T("CX7 (A1 exit 0): no grad_cos at any on-window eval -> missing gradient fields -> exit 1", o.returncode == 1
      and "missing gradient fields (grad_cos)" in o.stdout, o.stdout[-400:])
    r8 = tmp / "c8"
    partial(r8, "only-stars", "1e-3", 0, 1500, 0.9); partial(r8, "only-stars", "1e-3", 1, 1500, 0.9, missing=("grad_prior",))
    o = run(r8)
    T("a missing grad_prior in one run's on-window evals -> exit 1", o.returncode == 1 and "missing gradient fields (grad_prior)" in o.stdout)
    r9 = tmp / "c9"
    partial(r9, "only-stars", "1e-3", 0, 1500, 1 / 2.2, cos=-0.9)
    o = run(r9)
    T("opposing gradients (b = 1.2 a, cos -0.9; the logged a / |g_T| = 1.89) -> a / (a + b) 0.455 -> exit 1", o.returncode == 1
      and "median LM share a / (a + b) 0.455" in o.stdout)
    r4 = tmp / "c4"
    for s in (0, 1):
        partial(r4, "only-stars", "1e-3", s, 1500, 0.9)
    bad = partial(r4, "only-cities", "2e-3", 0, 1500, 0.9)
    bad.write_text(bad.read_text()[:200])   # a JSON caught mid-write
    partial(r4, "only-months", "5e-4", 0, 750, 0.2)   # before the window: listed, not judged
    o = run(r4)
    T("LM share 0.9 -> exit 0; a truncated run JSON is reported unreadable and skipped; a run before the window is listed, not judged",
      o.returncode == 0 and "unreadable (being written?): ptdom-53m-k2-only-cities-lr2e-3-s0" in o.stdout and "passed so far" in o.stdout
      and "before the window (not judged yet): only-months@5e-4 s0" in o.stdout, o.stdout[-500:])
    r5 = tmp / "c5"
    partial(r5, "only-stars", "1e-3", 0, 750, 0.4)
    o = run(r5)
    T("runs before the window only (last eval step 750) -> exit 2 (no on-window eval yet)", o.returncode == 2 and "no on-window eval" in o.stdout, o.stdout[-300:])
    r10 = tmp / "c10"
    partial(r10, "only-stars", "1e-3", 0, 1500, 0.9, logged_off=0.05)
    o = run(r10)
    T("a logged grad_lm_share 0.05 off its re-derivation -> exit 1 (unverifiable)", o.returncode == 1 and "logged grad_lm_share" in o.stdout, o.stdout[-300:])
    r11 = tmp / "c11"
    partial(r11, "only-stars", "1e-3", 0, 1250, 0.3); partial(r11, "only-stars", "1e-3", 0, 1500, 0.9, rr=1)
    o = run(r11)
    T("A2 re-runs: an unfinished first dir at 0.3 and its -r1 dir at 0.9 -> the check reads the -r1 run (exit 0)", o.returncode == 0
      and "median LM share a / (a + b) 0.900" in cell_line(o.stdout, "only-stars@1e-3"), o.stdout[-300:])
# ---------------------------------------------------------------- the quantity on REAL Rev E run JSONs (200-step CPU smoke)
import glob
cs = json.loads(Path(glob.glob(str(R / "tests" / "smoke" / "stars" / "run_*.json"))[0]).read_text())["curve"]
cn = json.loads(Path(glob.glob(str(R / "tests" / "smoke" / "none" / "run_*.json"))[0]).read_text())["curve"]
gs, gn = RO.window_grad(cs), RO.window_grad(cn)
on = [e for e in cs if e.get("prior_on")]
indep = sorted(e["grad_lm"] / (e["grad_lm"] + e["grad_prior"]) for e in on)
indep_med = indep[len(indep) // 2] if len(indep) % 2 else (indep[len(indep) // 2 - 1] + indep[len(indep) // 2]) / 2
cg = RO.cell_gate([{"seed": 0, "grad": gs}])
T(f"real smoke JSONs (tiny d64 x 3): stars on-window median a / (a + b) {gs['share']:.4f} = the independent re-computation from the "
  f"JSON fields {indep_med:.4f} over {gs['n_on']} on-window evals (A1's a / |g_T| {gs['lm_share_T']:.4f}); no missing field; double "
  f"path {gs['dp_maxdiff']:.1e}; b > a on {gs['n_b_gt_a']}/{gs['n_ok']}, clip < 1 on {gs['n_clip_lt1']}/{gs['n_ok']}; the cell "
  f"FAILS (the tiny model's prior dominates, as the build noted); none: no on-window eval",
  gs["n_on"] == 3 and gs["n_missing"] == 0 and abs(gs["share"] - indep_med) < 1e-12 and gs["dp_maxdiff"] < 1e-3
  and gs["n_b_gt_a"] == 3 and not cg["ok"] and gn["n_on"] == 0)
T("the real runs/ptdom/HOLD was not created by the tests", (R / "HOLD").exists() == hold_before and not hold_before)
print("A2 AUX GATE", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)})")
sys.exit(0 if all(ok) else 1)
