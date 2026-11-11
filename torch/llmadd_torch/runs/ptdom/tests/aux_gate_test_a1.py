"""Amendment A1, the auxiliary-loss gate on synthetic run JSONs (nothing is written into runs/ptdom; the HOLD file of the
tests is a temp path):
  select_ptdom.py --stage A: a stage-A root (92 runs, readout_ptdom.fake_run) where only-stars has LM share 0.3 -> exit 2,
  'AUX GATE FAILED', the --hold file written with the reason, no picks file; the same root at share 0.9 -> a decision.
  aux_gate_check.py on PARTIAL curves (the trainer's JSON at every eval; steps 0, 250, ..; on-window from step 916):
  shares 0.4 -> exit 1; 0.9 -> exit 0 (its pooled median, clipped share and per-run lines printed); runs before step 1000
  only -> exit 2; a run JSON caught mid-write -> reported unreadable, the others read; a logged share that disagrees with
  the rb6 re-derivation -> exit 1. The quantity on the real smoke run JSONs (stars window-median LM share = the audit's
  smokechk value). The real runs/ptdom/HOLD is never created."""
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


def stageA(root, share_of):
    for arm in RO.ARMS:
        for lr in RO.GRID[arm]:
            for s in RO.SEL:
                RO.fake_run(root, arm, lr, s, {d: 30.0 + (5.0 if lr in ("1e-3", "8e-3") else 0.0) + 0.01 * s for d in RO.SIX},
                            lm_share=share_of(arm, lr, s))


def partial(root, arm, lr, seed, last_step, share, clip=1.0, cos=0.0, logged_off=0.0):
    """A partial run JSON as the trainer writes it at every eval: evals at 0, 250, .., last_step; on-window (prior_on) from
    step 916 (= the window's first step) on; grad_lm 1, grad_prior sqrt(1 / share^2 - 1) at cos 0 (rb6 re-derivation =
    share), the logged grad_lm_share = share + logged_off."""
    tag = f"ptdom-53m-k2-{arm}-lr{lr}"
    d = Path(root) / f"{tag}-s{seed}" / tag
    d.mkdir(parents=True, exist_ok=True)
    gp = math.sqrt(1 / share ** 2 - 1)
    c = [{"step": st, "lm": 9.0, "val_loss": 5.0, "geo": 0.5, "prior_on": int(st >= 916), "grad_lm": 1.0,
          "grad_prior": gp if st >= 916 else None, "grad_cos": cos if st >= 916 else None,
          "grad_lm_share": share + logged_off if st >= 916 else 1.0, "clip_coef": clip if st >= 916 else 1.0}
         for st in range(0, last_step + 1, 250)]
    (d / f"run_whole_none_seed{seed}_{tag}.json").write_text(json.dumps({"curve": c, "args": {}}))
    return d / f"run_whole_none_seed{seed}_{tag}.json"


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # 1. select_ptdom: gate failure and pass
    root = tmp / "a"
    stageA(root, lambda arm, lr, s: 0.3 if arm == "only-stars" else 0.9)
    hold, out = tmp / "HOLD", tmp / "picks_A.json"
    o = subprocess.run([PY, str(R / "select_ptdom.py"), "--stage", "A", "--root", str(root), "--out", str(out), "--hold", str(hold)],
                       capture_output=True, text=True)
    T("select_ptdom: only-stars median LM share 0.3 -> exit 2, 'AUX GATE FAILED', the HOLD file names only-stars, no picks file",
      o.returncode == 2 and "AUX GATE FAILED" in o.stdout and hold.exists() and "only-stars" in hold.read_text()
      and "only-months" not in hold.read_text().split("prior arm(s)")[1].split(":")[0] and not out.exists(), (o.returncode, o.stdout[-400:], o.stderr[-300:]))
    print("    " + o.stdout.strip().replace("\n", "\n    ")[-1500:])
    root2 = tmp / "b"
    stageA(root2, lambda arm, lr, s: 0.9)
    hold2 = tmp / "HOLD2"
    o = subprocess.run([PY, str(R / "select_ptdom.py"), "--stage", "A", "--root", str(root2), "--out", str(out), "--hold", str(hold2)],
                       capture_output=True, text=True)
    pk = json.loads(out.read_text()) if out.exists() else {}
    T("select_ptdom: LM share 0.9 everywhere -> 'AUX GATE passed', a decision written (amendment A1, the gate summary in it), no HOLD",
      o.returncode == 0 and "AUX GATE passed" in o.stdout and pk.get("amendment") == "A1" and abs(pk["aux_gate"]["only-stars"]["median_lm_share"] - 0.9) < 1e-12
      and not hold2.exists(), (o.returncode, o.stderr[-300:]))
    # 2. aux_gate_check on partial curves
    run = lambda r_: subprocess.run([PY, str(R / "aux_gate_check.py"), "--root", str(r_)], capture_output=True, text=True)
    r3 = tmp / "c"
    for s in (0, 1):
        partial(r3, "only-stars", "1e-3", s, 1750, 0.4, clip=0.6)
        partial(r3, "only-months", "1e-3", s, 1500, 0.95)
    partial(r3, "none", "8e-3", 0, 1750, 0.2)   # none runs are not read (no prior)
    o = run(r3)
    T("aux_gate_check: only-stars LM share 0.4 over its on-window evals so far (steps 1000-1750, 2 runs) -> exit 1; only-months 0.95 not flagged",
      o.returncode == 1 and "only-stars" in o.stdout and "FAIL" in [l for l in o.stdout.splitlines() if l.startswith("only-stars")][0]
      and "FAIL" not in [l for l in o.stdout.splitlines() if l.startswith("only-months")][0] and "none " not in o.stdout
      and "on-window evals   8" in [l for l in o.stdout.splitlines() if l.startswith("only-stars")][0] and "clipped 100%" in o.stdout, o.stdout[-700:])
    print("    " + o.stdout.strip().replace("\n", "\n    "))
    r4 = tmp / "d"
    for s in (0, 1):
        partial(r4, "only-stars", "1e-3", s, 1500, 0.9)
    bad = partial(r4, "only-cities", "2e-3", 0, 1500, 0.9)
    bad.write_text(bad.read_text()[:200])   # a JSON caught mid-write
    o = run(r4)
    T("aux_gate_check: LM share 0.9 -> exit 0; a truncated run JSON is reported unreadable and skipped",
      o.returncode == 0 and "unreadable (being written?): ptdom-53m-k2-only-cities-lr2e-3-s0" in o.stdout and "passed so far" in o.stdout, o.stdout[-500:])
    r5 = tmp / "e"
    partial(r5, "only-stars", "1e-3", 0, 750, 0.4)
    o = run(r5)
    T("aux_gate_check: runs before the window (last eval step 750) -> exit 2 (no on-window eval yet)", o.returncode == 2 and "no on-window eval" in o.stdout, o.stdout[-300:])
    r6 = tmp / "f"
    partial(r6, "only-stars", "1e-3", 0, 1500, 0.9, logged_off=0.05)
    o = run(r6)
    T("aux_gate_check: a logged share 0.05 off the rb6 re-derivation -> exit 1 (unverifiable)", o.returncode == 1, o.stdout[-300:])
# 3. the quantity on REAL Rev E run JSONs (the 200-step CPU smoke, tiny d64 x 3): the window-median LM share of the stars
# prior = the audit's smokechk.py value (0.083-0.099 at the 3 on-window evals), the rb6 re-derivation agrees, the clip
# acted at every on-window eval (clip_coef 0.10-0.12); none: no on-window eval
import glob
cs = json.loads(Path(glob.glob(str(R / "tests" / "smoke" / "stars" / "run_*.json"))[0]).read_text())["curve"]
cn = json.loads(Path(glob.glob(str(R / "tests" / "smoke" / "none" / "run_*.json"))[0]).read_text())["curve"]
gs, gn = RO.window_grad(cs), RO.window_grad(cn)
T(f"real smoke JSONs: stars window-median LM share {gs['lm_share']:.4f} in [0.083, 0.099] over {gs['n_on']} on-window evals, rb6 "
  f"re-derivation max diff {gs['dp_maxdiff']:.1e} < 1e-3, clipped {gs['n_clipped']}/{gs['n_clip_logged']}; none: no on-window eval",
  gs["n_on"] == 3 and 0.083 <= gs["lm_share"] <= 0.099 and gs["dp_maxdiff"] < 1e-3 and gs["n_clipped"] == 3 and gn["n_on"] == 0)
T("the real runs/ptdom/HOLD was not created by the tests", (R / "HOLD").exists() == hold_before and not hold_before)
print("A1 AUX GATE", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)})")
sys.exit(0 if all(ok) else 1)
