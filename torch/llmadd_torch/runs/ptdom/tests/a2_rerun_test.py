"""Amendment A2: the re-run path after a HOLD (make_jobs_ptdom.rerun_plan / render(rr) / main --rerun / --submit refusals),
without submitting anything: sbatch is replaced in-process by a recorder that never calls Slurm (only `sbatch --test-only`
is run for real, on an -r1 job file written to a temp dir); every directory written is a temp dir (runs/ptdom gets nothing).
  1. rerun_plan on constructed SUBMITTED_ flags / out dirs / job logs: a HOLD exit -> -r1 ('HOLD exit'); a job still in the
     queue -> refused; a finished run (75 evals) -> refused (a run is data); an unfinished run -> -r1; a cell never submitted
     -> not listed (the normal --submit); a cell whose -r1 also exited at HOLD -> -r2; squeue failing -> refused.
  2. render(rr=1) differs from render(rr=0) only in the name lines (job name, OUT, --out of ARGS, the re-run comment); the
     intent (CHECK_PY's) is identical; bash -n; sbatch --test-only accepts it.
  3. main() on a temp copy of runs/ptdom (code symlinked, the 92 job files copied, sbatch recorded): every name with a
     SUBMITTED_ flag is refused by --submit (printed); a name without one is submitted (recorded); --rerun lists the -r1
     names and --rerun --submit submits exactly them; HOLD blocks --submit; the real runs/ptdom gets no file.
  4. the CLI on the real runs/ptdom: --stage A --rerun --no-write --print -> 0 jobs (nothing was ever submitted)."""
import contextlib
import difflib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "ptdom"
sys.path.insert(0, str(B)); sys.path.insert(0, str(R / "code" / "tools"))
import make_jobs_ptdom as M   # noqa: E402
import make_rb6_tasks as P    # noqa: E402

ok = []


def T(name, cond, val=None):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val is not None else ""))


before = sorted(p.name for p in R.iterdir())
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # ---------------------------------------------------------------- 1. rerun_plan
    rd, logs = tmp / "r", tmp / "logs"
    rd.mkdir(); logs.mkdir()

    def flag(a, lr, s, k, jid):
        (rd / f"SUBMITTED_{M.job_name(a, lr, s, k)}").write_text(f"Submitted batch job {jid}\n")

    def log(a, lr, s, k, jid, txt):
        (logs / f"llmadd_ptdom-{a}-lr{lr}-s{s}{f'-r{k}' if k else ''}_{jid}.out").write_text(txt)

    def runjson(a, lr, s, k, n):
        tag = M.tag_of(a, lr)
        d = rd / "out" / M.job_name(a, lr, s, k) / tag
        d.mkdir(parents=True)
        (d / f"run_whole_none_seed{s}_{tag}.json").write_text(json.dumps({"curve": [{"step": i} for i in range(n)]}))
    flag("none", "1e-3", 0, 0, 111); log("none", "1e-3", 0, 0, 111, "HOLD present: exiting\n")
    flag("none", "2e-3", 0, 0, 112)
    flag("only-stars", "1e-3", 0, 0, 113); runjson("only-stars", "1e-3", 0, 0, 75)
    flag("only-stars", "2e-3", 0, 0, 114); runjson("only-stars", "2e-3", 0, 0, 40); log("only-stars", "2e-3", 0, 0, 114, "## 2. training\n")
    flag("none", "8e-3", 0, 0, 115); flag("none", "8e-3", 0, 1, 116); log("none", "8e-3", 0, 1, 116, "HOLD present: exiting\n")
    todo = [("none", "1e-3", 0), ("none", "2e-3", 0), ("only-stars", "1e-3", 0), ("only-stars", "2e-3", 0), ("none", "4e-3", 0),
            ("none", "8e-3", 0)]
    plan, refused = M.rerun_plan(todo, rd, ({"112", "999"}, {"other"}), logs)
    got = {(a, lr, s): (k, why) for a, lr, s, k, why in plan}
    T("rerun_plan: HOLD exit -> -r1 ('HOLD exit'); unfinished (40 evals) -> -r1; -r1 also HOLD-exited -> -r2; never submitted -> "
      "not listed; queued -> refused; finished (75 evals) -> refused",
      set(got) == {("none", "1e-3", 0), ("only-stars", "2e-3", 0), ("none", "8e-3", 0)}
      and got[("none", "1e-3", 0)][0] == 1 and got[("none", "1e-3", 0)][1].startswith("HOLD exit")
      and got[("only-stars", "2e-3", 0)] == (1, "unfinished run JSON (40 evals)") and got[("none", "8e-3", 0)][0] == 2
      and got[("none", "8e-3", 0)][1].startswith("HOLD exit") and len(refused) == 2
      and any("job 112 is in the queue" in x for x in refused) and any("its run finished (75 evals)" in x for x in refused), (plan, refused))
    for x in plan:
        print(f"    re-run {x}")
    for x in refused:
        print(f"    refused {x}")
    plan2, _ = M.rerun_plan(todo, rd, (set(), {"llmadd_ptdom-none-lr1e-3-s0"}), logs)
    T("rerun_plan: a job found in the queue by its NAME (id not listed) is refused too", ("none", "1e-3", 0) not in {x[:3] for x in plan2})
    try:
        M.rerun_plan(todo, rd, None, logs)
        T("rerun_plan refuses when squeue fails", False)
    except SystemExit as e:
        T("rerun_plan refuses when squeue fails", "squeue failed" in str(e))
    # ---------------------------------------------------------------- 2. render(rr)
    RF = M.refs()
    commit = (R / "code" / "COMMIT").read_text().split()[0]
    n0, _, t0, a0, I0 = M.render(P, "only-stars", "1e-3", 0, RF[0], commit)
    n1, _, t1, a1, I1 = M.render(P, "only-stars", "1e-3", 0, RF[0], commit, rr=1)
    dl = [l for l in difflib.unified_diff(t0.splitlines(), t1.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    T(f"render(rr=1): name {n1}; vs rr=0 only the job name, OUT=, the ARGS --out path and the re-run comment differ ({len(dl)} diff "
      "lines); the intent is identical; the trainer --tag is unchanged",
      n1 == "ptdom-53m-k2-only-stars-lr1e-3-s0-r1" and I0 == I1 and a1 == a0.replace(f"/{n0}/", f"/{n1}/") and len(dl) == 7
      and sum(l.startswith("+# A2 re-run 1 of ptdom-53m-k2-only-stars-lr1e-3-s0") for l in dl) == 1
      and "#SBATCH --job-name=llmadd_ptdom-only-stars-lr1e-3-s0-r1" in t1 and "--tag _ptdom-53m-k2-only-stars-lr1e-3 " in a1, dl)
    print("    " + "\n    ".join(l[:200] for l in dl))
    jf = tmp / "job_r1.sbatch"
    jf.write_text(t1)
    T("the -r1 job file: bash -n", subprocess.run(["bash", "-n", str(jf)]).returncode == 0)
    o = subprocess.run(["sbatch", "--test-only", str(jf)], capture_output=True, text=True)
    T("the -r1 job file: sbatch --test-only accepts it (nothing submitted)", o.returncode == 0 and "to start at" in (o.stdout + o.stderr),
      (o.stdout + o.stderr).strip()[:200])
    print("    " + (o.stdout + o.stderr).strip()[:300])
    # ---------------------------------------------------------------- 3. main() on a temp copy of runs/ptdom, sbatch recorded
    TR = tmp / "ptdom"
    TR.mkdir()
    (TR / "code").symlink_to(R / "code")
    for f in R.glob("job_*.sbatch"):
        shutil.copy(f, TR / f.name)
    calls = []
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd and cmd[0] == "sbatch" and "--test-only" not in cmd:
            calls.append(Path(cmd[-1]).name)
            return subprocess.CompletedProcess(cmd, 1, "", "recorded by the test: not submitted")
        return real_run(cmd, *a, **k)
    M.subprocess.run = fake_run
    M.R = TR

    def main(*argv):
        sys.argv = ["make_jobs_ptdom.py", *argv]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                M.main()
            err = None
        except (AssertionError, SystemExit) as e:
            err = e
        return buf.getvalue(), err
    for a, lr, s in M.cells("A"):
        (TR / f"SUBMITTED_{M.job_name(a, lr, s)}").write_text(f"Submitted batch job {9_000_000 + len(calls)}\n")
    (TR / "SUBMITTED_ptdom-53m-k2-none-lr4e-3-s0").unlink()
    out, err = main("--stage", "A", "--submit")
    refl = [l for l in out.splitlines() if "refused (SUBMITTED_" in l]
    T("--submit: the 91 names with a SUBMITTED_ flag are refused (printed), the one without a flag is submitted (recorded, not sent)",
      err is None and len(refl) == 91 and calls == ["job_ptdom-53m-k2-none-lr4e-3-s0.sbatch"], (err, len(refl), calls[:3]))
    (TR / "SUBMITTED_ptdom-53m-k2-none-lr4e-3-s0").write_text("Submitted batch job 9000999\n")
    calls.clear()
    out, err = main("--stage", "A", "--rerun", "--no-write", "--print")
    rl = [l for l in out.splitlines() if l.startswith("re-run: ")]
    T("--rerun --print: every one of the 92 cells (flagged, not queued, no run) -> its -r1 name; 92 jobs rendered (A2 re-runs)",
      err is None and len(rl) == 92 and all(l.split(" -> ")[1].split(" ")[0].endswith("-r1") for l in rl)
      and "ptdom stage A (A2 re-runs): 92 jobs rendered" in out and not list(TR.glob("job_*-r1.sbatch")), (err, len(rl)))
    print("    " + rl[0])
    out, err = main("--stage", "A", "--rerun", "--submit")
    T("--rerun --submit: exactly the 92 -r1 job files are submitted (recorded, not sent), in seed-major order",
      err is None and len(calls) == 92 and all(c.endswith("-r1.sbatch") for c in calls) and calls[0].endswith("-s0-r1.sbatch")
      and calls[-1].endswith("-s3-r1.sbatch"), (err, calls[:2]))
    (TR / "HOLD").write_text("test\n")
    calls.clear()
    out, err = main("--stage", "A", "--rerun", "--submit")
    T("HOLD present -> --submit refused (AssertionError), nothing recorded", isinstance(err, AssertionError) and "HOLD" in str(err) and not calls)
    M.subprocess.run = real_run
    M.R = R
# ---------------------------------------------------------------- 4. the CLI on the real runs/ptdom
o = subprocess.run([M.PY, str(B / "make_jobs_ptdom.py"), "--stage", "A", "--rerun", "--no-write", "--print"], capture_output=True, text=True, cwd=B)
T("real runs/ptdom: --stage A --rerun --no-write --print -> 0 jobs (no SUBMITTED_ flag exists: nothing was ever submitted)",
  o.returncode == 0 and "ptdom stage A (A2 re-runs): 0 jobs rendered" in o.stdout and not list(R.glob("SUBMITTED_*")), (o.stdout + o.stderr)[-300:])
T("runs/ptdom unchanged by this test (no new file)", sorted(p.name for p in R.iterdir()) == before)
print("A2 RERUN", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)})")
sys.exit(0 if all(ok) else 1)
