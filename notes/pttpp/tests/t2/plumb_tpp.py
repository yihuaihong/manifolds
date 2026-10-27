"""Plumbing test of the pttpp selection (runs/pttpp/maybe_select.sh + pick_tpp.py, the read-out in the ladder-tpp worktree)
on a fake root: synthetic finished stage-A runs, the stand-in generator (standin_gen.py: real rendering, --test-only = the real
sbatch --test-only, --submit = fake flags only). Nothing is submitted. Scenarios (designed -> asserted):
  P0  stage A incomplete -> --check exit 1, nothing decided
  P1  7% tpp7-190m-t0.745: none 9.6e-3 beats 4.8e-3 by 5 (OPEN upper) -> X at 1.92e-2; win-ctx tied within 2, lower
      in-training bpb at 2.4e-3 -> C at 2.4e-3; the X / C jobs pass --test-only and are (fake) submitted
  P2  the X runs (none@1.92e-2) 3 points below 9.6e-3 -> closed -> C at 9.6e-3
  P3  (fresh) the X runs 4 points above 9.6e-3 -> still open -> STOP, nothing submitted for none
  P4  2% tpp2-190m-t0.745 via pt2main_summary: none tied (lower offline bpb at 9.6e-3) -> C at 9.6e-3; win-ctx 6e-4 beats
      1.2e-3 by 5 (OPEN lower) -> X at 3e-4
  P5  HOLD -> nothing; a changed frozen file -> FROZEN FILE CHANGED, nothing; the lock held by a running job -> pending mark
  P6  --check after everything was submitted -> exit 1 (nothing to do)
    python plumb_tpp.py        -> PLUMB PASS / FAIL"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

REAL = T.R
SRC = Path(os.environ.get("PLUMB_SRC", str(REAL)))   # where pick_tpp.py / maybe_select.sh come from (T1: scale_work/tpp/t1)
RO = Path("/scratch/yh6210/llmadd-burst-tpp/tools")
W = HERE / "plumb"
PY = T.PY
fails = []


def check(c, what):
    print(("PASS " if c else "FAIL ") + what, flush=True)
    if not c:
        fails.append(what)


def write_run(R, gid, arm, lr, seed, area_frac, bpb):
    """a finished run of the design with a flat curve h = area_frac (so area = 100 x area_frac) at every eval"""
    c = T.GROUPS[gid]
    F = T.facts(gid, arm, lr, seed, "A")
    d, nl, nh, dff = F["dims"]
    steps = sorted(set(range(0, F["S"], F["E"])) | {F["S"] - 1})
    ev = F["events"]
    curve = []
    for st in steps:
        inwin = bool(ev) and ev[0][0] <= st < ev[1][0]
        e = {"step": st, "tokens": st * 16384, **{f"held_{b}": (0.0 if st == 0 else area_frac) for b in ("b11", "b12", "b13", "b22", "b23", "b33")},
             "lm": F["lm0"] if st == 0 else 3.0, "val_loss": 3.0, "val_bits_per_byte": bpb, "geo": 0.1 if arm != "none" else None,
             "prior_on": int(inwin), "prior_on_frac": F["on_frac"] if st == F["S"] - 1 else 0.0,
             "prior_used_frac": F["on_frac"] if st == F["S"] - 1 else 0.0}
        if inwin:
            e.update(grad_lm=1.0, grad_prior=0.2, grad_cos=0.1)
        curve.append(e)
    a = {"tokenization": "whole", "target": "none" if arm == "none" else "helix3", "site": "ctx",
         "schedule": "whole" if arm == "none" else "window", "window": "0.05,0.4", "arith_lines": 4 if c["setting"] == "7" else 1,
         "lr": float(lr), "lam": 5.0, "tokens": float(F["T"]), "bs": 32, "seq": 512, "d_model": d, "n_layers": nl, "n_heads": nh,
         "d_ff": dff, "seed": seed, "tag": f"_{F['tag']}", "geo_layers": None, "eval_every": F["E"], "n_eval": 150, "untie": False}
    r = {"args": a, "steps": F["S"], "n_params": F["n_params"], "grad_ckpt_on": F["ckpt_on"], "eval_every_steps": F["E"],
         "lm_chunk": 0, "code_sha": T.PIN7_SHA if c["setting"] == "7" else T.WANT_SHA2["pretrain_range.py"], "prior_events": ev,
         "curve": curve}
    if c["setting"] == "7":
        out = R / "out" / F["tag"]
        out.mkdir(parents=True, exist_ok=True)
        (out / f"run_whole_{a['target']}_seed{seed}_{F['tag']}.json").write_text(json.dumps(r))
        (out / f"check_s{seed}.json").write_text(json.dumps({"pass": True}))
    else:
        task = R / "out2" / F["name"]
        (task / F["tag"]).mkdir(parents=True, exist_ok=True)
        f = task / F["tag"] / f"run_whole_{a['target']}_seed{seed}_{F['tag']}.json"
        f.write_text(json.dumps(r))
        (task / "check.json").write_text(json.dumps({"pass": True, "checks": []}))
        (task / "offline_lm.json").write_text(json.dumps({str(f.resolve()): {"tag": f"_{F['tag']}", "seed": seed, "max_windows": 0,
                                                                           "orig_minus_logged": 0.0, "bpb_full": bpb}}))


def fresh(name):
    root = W / name
    shutil.rmtree(root, ignore_errors=True)
    R = root / "pttpp"
    R.mkdir(parents=True)
    for c in ("code7", "code2"):
        os.symlink(REAL / c, R / c)
    shutil.copy(REAL / "AUDIT.md", R / "AUDIT.md")
    for f in ("pick_tpp.py", "maybe_select.sh"):
        shutil.copy(SRC / f, R / f)
        (R / f).chmod(0o644)
    gen = HERE / "standin_gen.py"
    (R / "FROZEN.txt").write_text("".join(f"{T.sha(p)} {p}\n" for p in (R / "pick_tpp.py", R / "maybe_select.sh", gen, RO / "pttpp_summary.py")))
    env = dict(os.environ, PTTPP_DIR=str(R), PTTPP_ROOT=str(root), PTTPP_GEN=str(gen), PTTPP_READOUT=str(RO))
    env.pop("SLURM_JOB_ID", None)
    return root, R, env


def mark_submitted_A(R, gid):
    """stage A as after the real --submit: its job files (rendered into the fake root) and SUBMITTED flags"""
    old, T.R = T.R, R
    try:
        for name, txt, F in T.render(gid):
            assert F["stage"] == "A"
            (R / f"job_{name}.sbatch").write_text(txt)
            (R / f"SUBMITTED_{name}").write_text("FAKE A\n")
    finally:
        T.R = old


def hook(R, env, gid):
    return subprocess.run(["bash", str(R / "maybe_select.sh"), gid], env=env, capture_output=True, text=True)


def check_rc(R, env, gid):
    return subprocess.run([PY, str(R / "pick_tpp.py"), "--group", gid, "--check"], env=env, capture_output=True, text=True).returncode


def decs(R, gid):
    f = R / "decisions" / f"{gid}.json"
    return {(e["arm"], e["stage"]): e.get("lr") for e in json.loads(f.read_text())["log"]} if f.exists() else {}


G7, G2g = "tpp7-190m-t0.745", "tpp2-190m-t0.745"
# ---------------- P0 / P1 / P2
root, R, env = fresh("p12")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):   # stage A, one run missing in each arm (none@4.8e-3 seed 3, win-ctx@1.2e-3 seed 2)
    for lr, ar in (("4.8e-3", .50), ("9.6e-3", .55)):
        if not (lr == "4.8e-3" and s == 3):
            write_run(R, G7, "none", lr, s, ar, 1.20)
    if s != 2:
        write_run(R, G7, "win-ctx", "1.2e-3", s, .880, 1.150)
    write_run(R, G7, "win-ctx", "2.4e-3", s, .885, 1.140)
check(check_rc(R, env, G7) == 1, "P0: stage A incomplete -> --check exit 1")
h = hook(R, env, G7)
check(decs(R, G7) == {}, "P0: the hook decides nothing while stage A is incomplete")
write_run(R, G7, "none", "4.8e-3", 3, .50, 1.20)
write_run(R, G7, "win-ctx", "1.2e-3", 2, .880, 1.150)
check(check_rc(R, env, G7) == 0, "P1: stage A complete -> --check exit 0")
h = hook(R, env, G7)
d = decs(R, G7)
check(d == {("none", "X"): "1.92e-2", ("win-ctx", "C"): "2.4e-3"}, f"P1: decisions none X 1.92e-2, win-ctx C 2.4e-3 ({d})")
sub = (R / "SUBMIT_CALLED.txt").read_text().split("\n") if (R / "SUBMIT_CALLED.txt").exists() else []
check(sum(" X tpp7-190m-t0.745-lr1.92e-2-none-s" in " " + l for l in sub) == 4 and sum(" C tpp7-190m-t0.745-lr2.4e-3-win-ctx-s" in " " + l for l in sub) == 4,
      f"P1: 4 X jobs + 4 C jobs (fake) submitted after --test-only ({len([l for l in sub if l])} lines)")
lg = (R / f"select_{G7}.log").read_text()
check("--test-only rc 0" in lg and lg.count("OK  tpp7-190m-t0.745") >= 8, "P1: the real sbatch --test-only passed for the 8 new jobs")
jt = (R / "job_tpp7-190m-t0.745-lr2.4e-3-win-ctx-s5.sbatch").read_text()
check("--save_model" in jt and "--seed 5 " in jt and "--lr 2.4e-3 " in jt, "P1: the C job of win-ctx is seed 5 at 2.4e-3 with --save_model")
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .52, 1.25)
h = hook(R, env, G7)
d = decs(R, G7)
check(d.get(("none", "C")) == "9.6e-3" and ("none", "STOP") not in d, f"P2: X worse -> closed -> none C 9.6e-3 ({d})")
check(check_rc(R, env, G7) == 1, "P6: after every stage is decided and submitted -> --check exit 1")
# ---------------- P3
root, R, env = fresh("p3")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "4.8e-3", s, .50, 1.20)
    write_run(R, G7, "none", "9.6e-3", s, .55, 1.20)
    write_run(R, G7, "win-ctx", "1.2e-3", s, .880, 1.150)
    write_run(R, G7, "win-ctx", "2.4e-3", s, .885, 1.140)
hook(R, env, G7)
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .59, 1.25)
hook(R, env, G7)
d = decs(R, G7)
lg = (R / f"select_{G7}.log").read_text()
check(("none", "STOP") in d and ("none", "C") not in d and "still OPEN at the upper edge after one extension" in lg,
      f"P3: X better again -> STOP for none, no C ({d})")
# ---------------- P4 (2%: pt2main_summary)
root, R, env = fresh("p4")
mark_submitted_A(R, G2g)
for s in (0, 1, 2, 3):
    write_run(R, G2g, "none", "4.8e-3", s, .30, 1.20)
    write_run(R, G2g, "none", "9.6e-3", s, .305, 1.19)
    write_run(R, G2g, "win-ctx", "6e-4", s, .80, 1.15)
    write_run(R, G2g, "win-ctx", "1.2e-3", s, .75, 1.15)
h = hook(R, env, G2g)
d = decs(R, G2g)
check(d == {("none", "C"): "9.6e-3", ("win-ctx", "X"): "3e-4"}, f"P4: 2% decisions none C 9.6e-3, win-ctx X 3e-4 ({d})")
ro = (R / f"readout_{G2g}.txt").read_text() if (R / f"readout_{G2g}.txt").exists() else ""
check("pt2main pre-registered read-out" in ro and "OPEN BRACKET at the lower edge" in ro, "P4: the 2% selection is pt2main_summary's own output")
sub = (R / "SUBMIT_CALLED.txt").read_text() if (R / "SUBMIT_CALLED.txt").exists() else ""
check(sub.count(" X tpp2-190m-t0.745-lr3e-4-win-ctx-s") == 4 and sub.count(" C tpp2-190m-t0.745-lr9.6e-3-none-s") == 4,
      "P4: 4 X + 4 C 2% jobs (fake) submitted")
# ---------------- P5
root, R, env = fresh("p5")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .5, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
(R / "HOLD").write_text("")
h = hook(R, env, G7)
check("HOLD present" in h.stdout and decs(R, G7) == {}, "P5: HOLD -> nothing")
(R / "HOLD").unlink()
p = R / "pick_tpp.py"
p.chmod(0o644)
p.write_text(p.read_text() + "\n# changed\n")
hook(R, env, G7)
check("FROZEN FILE CHANGED" in (R / f"select_{G7}.log").read_text() and decs(R, G7) == {}, "P5: a changed frozen file -> nothing")
shutil.copy(SRC / "pick_tpp.py", p)
(R / "locks" / G7).mkdir(parents=True)
(R / "locks" / G7 / "job").write_text(os.environ.get("SLURM_JOB_ID", "20284046") + "\n")   # a running job (this host's)
h = hook(R, env, G7)
check("pending mark left" in h.stdout and (R / "locks" / f"{G7}.pending").exists() and decs(R, G7) == {},
      "P5: lock held by a running job -> pending mark, nothing decided")
shutil.rmtree(R / "locks" / G7)
(R / "locks" / G7).mkdir()
(R / "locks" / G7 / "job").write_text("1\n")   # a job that is not running: stale lock taken over
hook(R, env, G7)
check(decs(R, G7) == {("none", "C"): "4.8e-3", ("win-ctx", "C"): "1.2e-3"} and not (R / "locks" / G7).exists(),
      f"P5: stale lock taken over and released; ties -> C at the lower bpb / lower lr ({decs(R, G7)})")
# ---------------- P7 (T1): a 7% run JSON without its job's check_s<seed>.json is not finished -> no decision, no STOP
root, R, env = fresh("p7")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .5, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
ck = R / "out" / T.tag_of(G7, "none", "9.6e-3") / "check_s2.json"
ck.unlink()
check(check_rc(R, env, G7) == 0, "P7: win-ctx is ready (its runs and checks are complete) -> --check exit 0")
hook(R, env, G7)
d = decs(R, G7)
check(d == {("win-ctx", "C"): "1.2e-3"}, f"P7: none waits for its missing check (no STOP), win-ctx decided ({d})")
ck.write_text(json.dumps({"pass": True}))
hook(R, env, G7)
check(decs(R, G7).get(("none", "C")) == "4.8e-3", f"P7: once the check is written, none is decided ({decs(R, G7)})")
# ---------------- P8 (T1): a pending mark set while a selection runs -> the holder repeats the step
root, R, env = fresh("p8")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .5, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
wrap = R / "py_wrap.sh"
wrap.write_text("#!/bin/bash\n"
                f'case "$*" in *--decide*) [ -e {R}/.marked ] || {{ touch {R}/.marked {R}/locks/{G7}.pending; }};; esac\n'
                f'exec {PY} "$@"\n')
wrap.chmod(0o755)
hook(R, dict(env, PTTPP_PY=str(wrap)), G7)
so = (R / f"select_{G7}.log.stdout").read_text() if (R / f"select_{G7}.log.stdout").exists() else ""
check(so.count("nothing to submit") + so.count("--submit rc") >= 2 and not (R / "locks" / f"{G7}.pending").exists()
      and not (R / "locks" / G7).exists(), f"P8: a mark set during the selection -> a second round, mark consumed, lock released")
# ---------------- P9 (T2): REOPEN + X appended by an amendment -> after the X runs, rule 1 again: C at the new pick
root, R, env = fresh("p9")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .040 if lr == "4.8e-3" else .041, 1.20 if lr == "4.8e-3" else 1.19)   # both at the floor
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
hook(R, env, G7)
check(decs(R, G7) == {("none", "C"): "9.6e-3", ("win-ctx", "C"): "1.2e-3"}, f"P9: floor tie -> C at the edge (as at 93M@0.745) ({decs(R, G7)})")
f = R / "decisions" / f"{G7}.json"
d = json.loads(f.read_text())
d["log"] += [{"seq": len(d["log"]), "arm": "none", "stage": "REOPEN", "amend": "T2"},
             {"seq": len(d["log"]) + 1, "arm": "none", "stage": "X", "lr": "1.92e-2", "amend": "T2"}]
f.write_text(json.dumps(d))
check(check_rc(R, env, G7) == 0, "P9: after REOPEN + X: the X jobs are not submitted yet -> --check exit 0")
hook(R, env, G7)
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(sub.count(" X tpp7-190m-t0.745-lr1.92e-2-none-s") == 4, "P9: the 4 X jobs are submitted by the hook")
check(check_rc(R, env, G7) == 1, "P9: X runs not finished, everything submitted -> --check exit 1")
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .120, 1.18)
hook(R, env, G7)
ed = json.loads(f.read_text())["log"]
check([(e["arm"], e["stage"], e.get("lr")) for e in ed][-1] == ("none", "STOP", "1.92e-2"),
      f"P9: the X lr beats both by > 2 at the upper edge -> still open -> STOP ({[(e['arm'], e['stage'], e.get('lr')) for e in ed]})")
for s in (0, 1, 2, 3):   # a second fresh case: the X lr ties -> C at the lower bpb
    pass
root, R, env = fresh("p9b")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .040 if lr == "4.8e-3" else .041, 1.20 if lr == "4.8e-3" else 1.19)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
hook(R, env, G7)
f = R / "decisions" / f"{G7}.json"
d = json.loads(f.read_text())
d["log"] += [{"seq": len(d["log"]), "arm": "none", "stage": "REOPEN", "amend": "T2"},
             {"seq": len(d["log"]) + 1, "arm": "none", "stage": "X", "lr": "1.92e-2", "amend": "T2"}]
f.write_text(json.dumps(d))
hook(R, env, G7)
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .045, 1.17)
hook(R, env, G7)
ed = [(e["arm"], e["stage"], e.get("lr")) for e in json.loads(f.read_text())["log"]]
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(ed[-1] == ("none", "C", "1.92e-2") and sub.count(" C tpp7-190m-t0.745-lr1.92e-2-none-s") == 4,
      f"P9b: X ties (within 2) with the lowest bpb -> new C at 1.92e-2, its 4 confirm jobs submitted ({ed})")
# ---------------- P10 (T2): the cost guard stops a submission that would exceed the budget
root, R, env = fresh("p10")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .5, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
hook(R, dict(env, PLUMB_COST_MAX="180,100"), G7)
lg = (R / f"select_{G7}.log").read_text()
check("COST GUARD" in lg and not (R / "SUBMIT_CALLED.txt").exists() and decs(R, G7), "P10: over budget -> decisions recorded, nothing submitted, COST GUARD logged")
hook(R, env, G7)
check((R / "SUBMIT_CALLED.txt").exists() and "within" in (R / f"select_{G7}.log").read_text(), "P10: within budget -> submitted on the next hook")
print("PLUMB", "PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
sys.exit(1 if fails else 0)
