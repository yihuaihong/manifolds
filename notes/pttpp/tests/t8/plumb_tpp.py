"""Plumbing test of the pttpp selection (runs/pttpp/maybe_select.sh + pick_tpp.py, the read-out in the ladder-tpp worktree)
on a fake root: synthetic finished stage-A runs, the stand-in generator (standin_gen.py: real rendering, --test-only = the real
sbatch --test-only, --submit = fake flags only). Nothing is submitted. Scenarios (designed -> asserted):
  P0  stage A incomplete -> --check exit 1, nothing decided
  P1  7% tpp7-190m-t0.745: none 9.6e-3 beats 4.8e-3 by 5 (OPEN upper) -> X at 1.92e-2; win-ctx tied within 2, lower
      in-training bpb at 2.4e-3 -> C at 2.4e-3; the X / C jobs pass --test-only and are (fake) submitted
  P2  the X runs (none@1.92e-2) 3 points below 9.6e-3 -> closed -> C at 9.6e-3
  P3  (fresh) the X runs 4 points above 9.6e-3 -> still open -> STOP, nothing submitted for none
  P4  2% tpp2-190m-t0.745 via pt2main_summary: none tied (lower offline bpb at 9.6e-3 = the upper edge) -> T6: X at
      1.92e-2, then C at 9.6e-3; win-ctx 6e-4 beats 1.2e-3 by 5 (OPEN lower) -> X at 3e-4, then C
  P5  HOLD -> nothing; a changed frozen file -> FROZEN FILE CHANGED, nothing; the lock held by a running job -> pending mark
  P6  --check after everything was submitted -> exit 1 (nothing to do)
  P7-P14 (T1-T4): see the scenario comments below. T5 (fourth audit): P15 a job killed while it held the lock and requeued
      with the same id takes its own old lock over (another live owner is still respected); P16 a crash of --check is logged
      and returned (rc 2), not read as "nothing to do"; P17 DISK GUARD keeps the decisions, the next call of the group submits.
      T6: none's pick at the upper grid edge is extended also when closed (P4 2%, P9 / P9b 7%: X automatically, one extension
      only; P9c: a pre-T6 C at the upper edge + REOPEN + X by hand); the lower edge (P5) and win-ctx (P1) are unchanged
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
# ---------------- P4 (2%: pt2main_summary; T6: none's tie at the upper edge -> X one step up, then rule 1 -> C)
root, R, env = fresh("p4")
mark_submitted_A(R, G2g)
for s in (0, 1, 2, 3):
    write_run(R, G2g, "none", "4.8e-3", s, .30, 1.20)
    write_run(R, G2g, "none", "9.6e-3", s, .305, 1.19)
    write_run(R, G2g, "win-ctx", "6e-4", s, .80, 1.15)
    write_run(R, G2g, "win-ctx", "1.2e-3", s, .75, 1.15)
h = hook(R, env, G2g)
d = decs(R, G2g)
check(d == {("none", "X"): "1.92e-2", ("win-ctx", "X"): "3e-4"}, f"P4 (T6): 2% none tied, pick 9.6e-3 = the upper edge -> X 1.92e-2; win-ctx X 3e-4 ({d})")
ro = (R / f"readout_{G2g}.txt").read_text() if (R / f"readout_{G2g}.txt").exists() else ""
check("pt2main pre-registered read-out" in ro and "OPEN BRACKET at the lower edge" in ro, "P4: the 2% selection is pt2main_summary's own output")
lg = (R / f"select_{G2g}.log").read_text()
check("T6: none's pick at the upper grid edge" in lg, "P4 (T6): the X decision says why")
sub = (R / "SUBMIT_CALLED.txt").read_text() if (R / "SUBMIT_CALLED.txt").exists() else ""
check(sub.count(" X tpp2-190m-t0.745-lr3e-4-win-ctx-s") == 4 and sub.count(" X tpp2-190m-t0.745-lr1.92e-2-none-s") == 4 and " C " not in sub,
      "P4: 4 + 4 2% X jobs (fake) submitted after the real --test-only, no C yet")
for s in (0, 1, 2, 3):
    write_run(R, G2g, "none", "1.92e-2", s, .30, 1.21)
    write_run(R, G2g, "win-ctx", "3e-4", s, .70, 1.15)
hook(R, env, G2g)
d = decs(R, G2g)
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(d.get(("none", "C")) == "9.6e-3" and d.get(("win-ctx", "C")) == "6e-4" and sub.count(" C tpp2-190m-t0.745-lr9.6e-3-none-s") == 4,
      f"P4 (T6): after the X runs (2% tie, lowest offline bpb at 9.6e-3) -> C 9.6e-3, 4 confirm jobs; win-ctx closed -> C 6e-4 ({d})")
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
# ---------------- P9 (T6): none's floor tie with the pick at the upper edge -> X automatically (was C, T2 then added X by hand)
def floor9(name):
    root, R, env = fresh(name)
    mark_submitted_A(R, G7)
    for s in (0, 1, 2, 3):
        for lr in ("4.8e-3", "9.6e-3"):
            write_run(R, G7, "none", lr, s, .040 if lr == "4.8e-3" else .041, 1.20 if lr == "4.8e-3" else 1.19)   # both at the floor
        for lr in ("1.2e-3", "2.4e-3"):
            write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
    return root, R, env


root, R, env = floor9("p9")
hook(R, env, G7)
f = R / "decisions" / f"{G7}.json"
check(decs(R, G7) == {("none", "X"): "1.92e-2", ("win-ctx", "C"): "1.2e-3"}, f"P9 (T6): floor tie at the upper edge -> none X 1.92e-2, win-ctx C ({decs(R, G7)})")
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(sub.count(" X tpp7-190m-t0.745-lr1.92e-2-none-s") == 4 and sub.count(" C tpp7-190m-t0.745-lr9.6e-3-none-s") == 0,
      "P9 (T6): the 4 X jobs are submitted by the hook, no none C")
check(check_rc(R, env, G7) == 1, "P9: X runs not finished, everything submitted -> --check exit 1")
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .120, 1.18)
hook(R, env, G7)
ed = json.loads(f.read_text())["log"]
check([(e["arm"], e["stage"], e.get("lr")) for e in ed][-1] == ("none", "STOP", "1.92e-2"),
      f"P9: the X lr beats both by > 2 at the upper edge -> still open -> STOP ({[(e['arm'], e['stage'], e.get('lr')) for e in ed]})")
root, R, env = floor9("p9b")
hook(R, env, G7)
f = R / "decisions" / f"{G7}.json"
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .045, 1.17)
hook(R, env, G7)
ed = [(e["arm"], e["stage"], e.get("lr")) for e in json.loads(f.read_text())["log"]]
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(ed[-1] == ("none", "C", "1.92e-2") and sub.count(" C tpp7-190m-t0.745-lr1.92e-2-none-s") == 4 and ("none", "X", "2.4e-2") not in ed
      and len([e for e in ed if e[1] == "X" and e[0] == "none"]) == 1,
      f"P9b (T6): X ties (within 2) with the lowest bpb -> C at 1.92e-2 (the new upper edge: no second extension), 4 confirm jobs ({ed})")
# ---------------- P9c (T2 / T6 by hand): a pick made before T6 (C at the upper edge, its confirm jobs submitted) + REOPEN + X
root, R, env = floor9("p9c")
f = R / "decisions" / f"{G7}.json"
f.parent.mkdir(exist_ok=True)
f.write_text(json.dumps({"group": G7, "log": [{"seq": 0, "arm": "none", "stage": "C", "lr": "9.6e-3"},
                                              {"seq": 1, "arm": "win-ctx", "stage": "C", "lr": "1.2e-3"}]}))   # as the pre-T6 code wrote it
hook(R, env, G7)
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(sub.count(" C tpp7-190m-t0.745-lr9.6e-3-none-s") == 4 and len(decs(R, G7)) == 2, "P9c: the pre-T6 decisions -> their C jobs submitted, nothing new decided")
d = json.loads(f.read_text())
d["log"] += [{"seq": 2, "arm": "none", "stage": "REOPEN", "amend": "T6"}, {"seq": 3, "arm": "none", "stage": "X", "lr": "1.92e-2", "amend": "T6"}]
f.write_text(json.dumps(d))
check(check_rc(R, env, G7) == 0, "P9c: after REOPEN + X the X jobs are not submitted yet -> --check exit 0")
hook(R, env, G7)
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(sub.count(" X tpp7-190m-t0.745-lr1.92e-2-none-s") == 4 and len(json.loads(f.read_text())["log"]) == 4, "P9c: the hook submits the 4 X jobs, decides nothing")
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .010, 1.25)   # the X lr worse by 3 points: closed, the pick stays 9.6e-3
hook(R, env, G7)
ed = [(e["arm"], e["stage"], e.get("lr")) for e in json.loads(f.read_text())["log"]]
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(ed[-1] == ("none", "C", "9.6e-3") and sub.count(" C tpp7-190m-t0.745-lr9.6e-3-none-s") == 4,
      f"P9c: X worse -> closed -> C at 9.6e-3 again; its confirm jobs are not submitted twice ({ed})")
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
# ---------------- P11 (T3): a stage-A run that failed on its own (no run JSON, no check), listed in OWN_FAILURES -> counts
root, R, env = fresh("p11")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        if not (lr == "9.6e-3" and s == 1):
            write_run(R, G7, "none", lr, s, .50 if lr == "4.8e-3" else .55, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
check(check_rc(R, env, G7) == 0 and decs(R, G7) == {}, "P11: win-ctx ready, nothing decided yet")
hook(R, env, G7)
check(decs(R, G7) == {("win-ctx", "C"): "1.2e-3"}, f"P11: none waits for its missing run ({decs(R, G7)})")
(R / "OWN_FAILURES.txt").write_text(f"{T.tag_of(G7, 'none', '9.6e-3')} 1\n")
hook(R, env, G7)
d = decs(R, G7)
check(d.get(("none", "X")) == "2.4e-3" and ("none", "C") not in d,
      f"P11: listed as an own failure -> counts with area 0 -> 9.6e-3 mean 41.2 < 4.8e-3 50.0 - 2 -> open at the lower edge -> X 2.4e-3 ({d})")
# ---------------- P12 (T3): lock owners: a dead pid on this host -> taken over; a live job's lock older than 2 h -> taken over
root, R, env = fresh("p12b")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .5, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
import socket
(R / "locks" / G7).mkdir(parents=True)
(R / "locks" / G7 / "job").write_text(f"pid:{socket.gethostname()}:999999\n")
hook(R, env, G7)
check(len(decs(R, G7)) == 2 and "took over the lock of pid:" in (R / f"select_{G7}.log").read_text(), "P12: a dead pid's lock is taken over")
root, R, env = fresh("p12c")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .5, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
(R / "locks" / G7).mkdir(parents=True)
(R / "locks" / G7 / "job").write_text(os.environ.get("SLURM_JOB_ID", "20284046") + "\n")
subprocess.run(["touch", "-d", "3 hours ago", str(R / "locks" / G7)], check=True)
hook(R, env, G7)
check(decs(R, G7) == {} and (R / "locks" / f"{G7}.pending").exists(), "P12 (T4): a live job's lock, however old, is not taken over -> pending")
root, R, env = fresh("p12d")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .5, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
(R / "locks" / G7).mkdir(parents=True)
(R / "locks" / G7 / "job").write_text(f"pid:{socket.gethostname()}:{os.getpid()}\n")   # this live process
h = hook(R, env, G7)
check(decs(R, G7) == {} and (R / "locks" / f"{G7}.pending").exists() and "pending mark left" in h.stdout,
      "P12: a live pid's fresh lock -> pending mark, nothing decided")
# ---------------- P13 (T4): an owner-less lock: fresh -> pending; older than 10 min -> taken over (renamed away)
root, R, env = fresh("p13")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        write_run(R, G7, "none", lr, s, .5, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
(R / "locks" / G7).mkdir(parents=True)
h = hook(R, env, G7)
check(decs(R, G7) == {} and "just made" in h.stdout, "P13: an owner-less fresh lock -> pending, nothing decided")
subprocess.run(["touch", "-d", "20 minutes ago", str(R / "locks" / G7)], check=True)
hook(R, env, G7)
check(len(decs(R, G7)) == 2 and list((R / "locks").glob(f"{G7}.stale.*")) and "took over the lock of ?" in (R / f"select_{G7}.log").read_text(),
      "P13: an owner-less lock older than 10 min -> renamed away, taken over, decided")
# ---------------- P14 (T4): OWN_FAILURES with malformed lines -> logged, never a crash; a listed run still queued -> not counted
root, R, env = fresh("p14")
mark_submitted_A(R, G7)
for s in (0, 1, 2, 3):
    for lr in ("4.8e-3", "9.6e-3"):
        if not (lr == "9.6e-3" and s == 1):
            write_run(R, G7, "none", lr, s, .50 if lr == "4.8e-3" else .55, 1.2)
    for lr in ("1.2e-3", "2.4e-3"):
        write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
tg = T.tag_of(G7, "none", "9.6e-3")
(R / "OWN_FAILURES.txt").write_text(f"# header\n{tg} s1   # OOM\nthis line is wrong\n  # indented comment\n")
(R / f"SUBMITTED_{tg}-s1").write_text(f"Submitted batch job {os.environ.get('SLURM_JOB_ID', '20284046')}\n")   # a job that runs
rc = check_rc(R, env, G7)
hook(R, env, G7)
lg = (R / f"select_{G7}.log").read_text()
check(rc == 0 and "line ignored" in lg and "still queued or running: not counted yet" in lg and ("none", "C") not in decs(R, G7)
      and ("none", "X") not in decs(R, G7), f"P14: bad lines logged; a listed run whose job still runs is not counted ({decs(R, G7)})")
(R / f"SUBMITTED_{tg}-s1").write_text("Submitted batch job 1\n")   # gone
hook(R, env, G7)
check(decs(R, G7).get(("none", "X")) == "2.4e-3", f"P14: once its job is gone the listed run counts (area 0) -> X 2.4e-3 ({decs(R, G7)})")


def ready7(name):
    root, R, env = fresh(name)
    mark_submitted_A(R, G7)
    for s in (0, 1, 2, 3):
        for lr in ("4.8e-3", "9.6e-3"):
            write_run(R, G7, "none", lr, s, .5, 1.2)
        for lr in ("1.2e-3", "2.4e-3"):
            write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
    return root, R, env


LIVE = os.environ.get("SLURM_JOB_ID", "20284046")   # a job that squeue shows running (this session's host job)
assert subprocess.run(["squeue", "-h", "-j", LIVE, "-t", "R", "-o", "%i"], capture_output=True, text=True).stdout.strip() == LIVE, LIVE
# ---------------- P15 (T5, audit F1): a job killed while it held the lock, requeued with the same id, finds its own old lock
root, R, env = ready7("p15")
(R / "locks" / G7).mkdir(parents=True)
(R / "locks" / G7 / "job").write_text(LIVE + "\n")   # its own id: squeue shows it running (it is this very job)
h = hook(R, dict(env, SLURM_JOB_NAME="llmadd_tpp-test", SLURM_JOB_ID=LIVE), G7)
lg = (R / f"select_{G7}.log").read_text()
check(len(decs(R, G7)) == 2 and f"job {LIVE}: took over the lock of {LIVE}" in lg and not (R / "locks" / G7).exists()
      and not (R / "locks" / f"{G7}.pending").exists(), f"P15: the requeued job takes over its own old lock, selects, releases ({decs(R, G7)}; {h.stdout.strip()[:120]})")
# the same lock, but the caller is another pttpp job: the live owner is respected
root, R, env = ready7("p15b")
(R / "locks" / G7).mkdir(parents=True)
(R / "locks" / G7 / "job").write_text(LIVE + "\n")
subprocess.run(["touch", "-d", "3 hours ago", str(R / "locks" / G7)], check=True)
h = hook(R, dict(env, SLURM_JOB_NAME="llmadd_tpp-test", SLURM_JOB_ID="424242"), G7)
check(decs(R, G7) == {} and (R / "locks" / G7).exists() and (R / "locks" / f"{G7}.pending").exists() and f"held by {LIVE}" in h.stdout,
      f"P15: another job's caller finds a live owner's lock (3 h old) -> pending mark, nothing decided ({h.stdout.strip()[:120]})")
# and a hand caller (pid owner) in front of a live job's lock
h = hook(R, env, G7)
check(decs(R, G7) == {} and (R / "locks" / G7).exists() and f"held by {LIVE}" in h.stdout, "P15: a hand caller respects the live owner too")
# ---------------- P16 (T5, audit F4): a crash of --check (here: the generator's import fails) is logged and returned, not 'nothing to do'
root, R, env = ready7("p16")
h = hook(R, dict(env, PLUMB_GEN_CRASH="1"), G7)
lg = (R / f"select_{G7}.log").read_text() if (R / f"select_{G7}.log").exists() else ""
check(h.returncode == 2 and "--check CRASHED: AssertionError" in lg and "pick_tpp.py --check FAILED (rc 2" in lg and decs(R, G7) == {}
      and check_rc(R, dict(env, PLUMB_GEN_CRASH="1"), G7) == 2, f"P16: --check crash -> logged twice, hook rc 2, nothing decided (rc {h.returncode})")
h = hook(R, env, G7)
check(h.returncode == 0 and len(decs(R, G7)) == 2, "P16: once the cause is gone the next call selects")
check(subprocess.run([PY, str(R / "pick_tpp.py"), "--group", T.BRIDGE, "--check"], env=env, capture_output=True, text=True).returncode == 1,
      "P16: the bridge (no selection) is still 'nothing to do' (rc 1)")
# ---------------- P17 (T5, audit F6): DISK GUARD keeps the decisions, submits nothing; the next call of the group submits
root, R, env = ready7("p17")
fb = R / "fakebin"
fb.mkdir()
(fb / "myquota").write_text("#!/bin/bash\ncat \"$(dirname \"$0\")/quota.txt\"\n")
(fb / "myquota").chmod(0o755)
(fb / "quota.txt").write_text("Quota for user yh6210\n/scratch     4.90TB  5.00TB (98.00%)   12345  (1.00%)\n")
(fb / "py").write_text(f'#!/bin/bash\nexport PATH="{fb}:$PATH"\nexec {PY} "$@"\n')   # after BASH_ENV (cluster.sh) has set PATH
(fb / "py").chmod(0o755)
envq = dict(env, PTTPP_PY=str(fb / "py"))
assert "98.00%" in subprocess.run([str(fb / "py"), "-c", "import subprocess; print(subprocess.run(['myquota'], capture_output=True, text=True).stdout)"],
                                  capture_output=True, text=True, env=envq).stdout
hook(R, envq, G7)
lg = (R / f"select_{G7}.log").read_text()
check(len(decs(R, G7)) == 2 and "DISK GUARD: /scratch at 98.00%" in lg and "the next hook call of this group retries" in lg
      and "maybe_select.sh tpp7-190m-t0.745' by hand" in lg and not (R / "SUBMIT_CALLED.txt").exists(),
      "P17: DISK GUARD -> decisions recorded, nothing submitted, the log says how it is retried")
check(check_rc(R, envq, G7) == 0, "P17: the unsubmitted decided runs make --check say 'something to do'")
(fb / "quota.txt").write_text("Quota for user yh6210\n/scratch     4.70TB  5.00TB (94.00%)   12345  (1.00%)\n")
hook(R, envq, G7)
sub = (R / "SUBMIT_CALLED.txt").read_text() if (R / "SUBMIT_CALLED.txt").exists() else ""
check(sub.count(" C tpp7-190m-t0.745-") == 8 and len(decs(R, G7)) == 2, f"P17: below the guard the next call submits the 8 C jobs, no new decision ({sub.count(' C ')})")
# ---------------- P20 (T7): a STOP after one extension, withdrawn by the user's approval: REOPEN + a second X one step further
import importlib   # noqa: E402
TN = importlib.import_module(os.environ.get("PLUMB_GENMOD", "make_jobs_pttpp"))   # the generator under test


def stop_after_x(name):   # none opens at the upper edge, X 1.92e-2 opens again -> STOP (as 294M@0.745 none)
    root, R, env = fresh(name)
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
    assert decs(R, G7).get(("none", "STOP")) == "1.92e-2", decs(R, G7)
    return root, R, env


def add(R, *entries):
    f = R / "decisions" / f"{G7}.json"
    d = json.loads(f.read_text())
    d["log"] += [dict(e, seq=len(d["log"]) + i) for i, e in enumerate(entries)]
    f.write_text(json.dumps(d))


def gen_ok(R):   # the generator under test on this fake root: decisions() / runs_of() accept the log?
    old, TN.R = TN.R, R
    try:
        TN.runs_of(G7)
        return True
    except AssertionError:
        return False
    finally:
        TN.R = old


root, R, env = stop_after_x("p20")
nsub = (R / "SUBMIT_CALLED.txt").read_text().count("\n")
check(check_rc(R, env, G7) == 1, "P20: after the STOP nothing to do (--check exit 1)")
add(R, {"arm": "none", "stage": "REOPEN", "amend": "T7"}, {"arm": "none", "stage": "X", "lr": "3.84e-2", "amend": "T7"})
check(gen_ok(R) and check_rc(R, env, G7) == 0, "P20: REOPEN (amend) of the STOP + X 3.84e-2 accepted; its X jobs are to be submitted -> --check exit 0")
old_, TN.R = TN.R, R
eff_ = TN.effective(G7)["none"]
xs_ = [r for r in TN.runs_of(G7) if r[0] == "X" and r[1] == "none"]
TN.R = old_
check(eff_ == {"X": "3.84e-2", "C": None, "STOP": False} and [r[2] for r in xs_] == ["1.92e-2"] * 4 + ["3.84e-2"] * 4,
      f"P20: effective = X 3.84e-2, no C, no STOP; runs_of keeps both X stages ({eff_})")
hook(R, env, G7)
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(sub.count(" X tpp7-190m-t0.745-lr3.84e-2-none-s") == 4 and sub.count("\n") == nsub + 4 and len(json.loads((R / "decisions" / f"{G7}.json").read_text())["log"]) == 5,
      "P20: the hook submits only the 4 new X jobs (after the real --test-only), decides nothing")
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "3.84e-2", s, .65, 1.30)   # better again by > 2 at the new upper edge
hook(R, env, G7)
ed = [(e["arm"], e["stage"], e.get("lr")) for e in json.loads((R / "decisions" / f"{G7}.json").read_text())["log"]]
check(ed[-1] == ("none", "STOP", "3.84e-2") and sum(e[1] == "X" for e in ed) == 2, f"P20: still open at the upper edge -> STOP again, no third X ({ed})")
root, R, env = stop_after_x("p20b")
add(R, {"arm": "none", "stage": "REOPEN", "amend": "T7"}, {"arm": "none", "stage": "X", "lr": "3.84e-2", "amend": "T7"})
hook(R, env, G7)
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "3.84e-2", s, .58, 1.30)   # within 2 of 1.92e-2 (.59): tie -> the lower in-training bpb = 1.92e-2
hook(R, env, G7)
ed = [(e["arm"], e["stage"], e.get("lr")) for e in json.loads((R / "decisions" / f"{G7}.json").read_text())["log"]]
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(ed[-1] == ("none", "C", "1.92e-2") and sub.count(" C tpp7-190m-t0.745-lr1.92e-2-none-s") == 4,
      f"P20b: the second X ties -> closed -> C at 1.92e-2 (lower bpb), 4 confirm jobs ({ed})")
# the generator refuses the other forms
for nm, ents in (("no amend", [{"arm": "none", "stage": "REOPEN"}]),
                 ("X two steps on", [{"arm": "none", "stage": "REOPEN", "amend": "T7"}, {"arm": "none", "stage": "X", "lr": "7.68e-2", "amend": "T7"}]),
                 ("X in the other direction", [{"arm": "none", "stage": "REOPEN", "amend": "T7"}, {"arm": "none", "stage": "X", "lr": "2.4e-3", "amend": "T7"}]),
                 ("C after the STOP", [{"arm": "none", "stage": "C", "lr": "1.92e-2"}]),
                 ("X after the STOP without REOPEN", [{"arm": "none", "stage": "X", "lr": "3.84e-2"}])):
    root, R, env = fresh("p20x")
    (R / "decisions").mkdir()
    (R / "decisions" / f"{G7}.json").write_text(json.dumps({"group": G7, "log": [
        {"seq": 0, "arm": "none", "stage": "X", "lr": "1.92e-2"}, {"seq": 1, "arm": "none", "stage": "STOP", "lr": "1.92e-2"}]}))
    add(R, *ents)
    check(not gen_ok(R), f"P20: the generator refuses '{nm}'")
root, R, env = fresh("p20y")   # a third X after a second REOPEN of a second STOP is refused too (one approval, one step)
(R / "decisions").mkdir()
(R / "decisions" / f"{G7}.json").write_text(json.dumps({"group": G7, "log": [
    {"seq": 0, "arm": "none", "stage": "X", "lr": "1.92e-2"}, {"seq": 1, "arm": "none", "stage": "STOP"},
    {"seq": 2, "arm": "none", "stage": "REOPEN", "amend": "T7"}, {"seq": 3, "arm": "none", "stage": "X", "lr": "3.84e-2", "amend": "T7"},
    {"seq": 4, "arm": "none", "stage": "STOP"}, {"seq": 5, "arm": "none", "stage": "REOPEN", "amend": "x"},
    {"seq": 6, "arm": "none", "stage": "X", "lr": "7.68e-2", "amend": "x"}]}))
check(gen_ok(R), "P20: a further approved REOPEN + X one more step is accepted by the generator's grammar (each needs its own amendment)")
print("PLUMB", "PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
sys.exit(1 if fails else 0)
