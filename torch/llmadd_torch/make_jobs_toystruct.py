"""Torch jobs toystruct (2026-10-16, user decision: run BOTH toy structure-prior experiments now, small and decisive,
before any 53M version: S4 = the cube rotation group, and dimensional analysis). Code: llmadd-burst
tools/toy_struct.py (generalises toy_grok.py; --task mod is its known-answer path, tools/test_toy_struct.py).
Intent spec and PRE-REGISTERED rules: runs/toystruct/AUDIT.md.
One job per (task, fraction): 5 seeds as 5 parallel processes, all arms per process (paired by seed: same init,
same split), then the summary (pre-registered tests) of that cell, once every process has finished.
Budget (steps) and --time: see BUDGET / MS_PER_STEP below (AUDIT.md "Budget" and "Cost").
Account torch_pr_625_general on the public L40S partition. Writes job files only; --submit submits jobs not yet
submitted (SUBMITTED_<tag> flag files). Kill switch: a job exits at start while runs/toystruct/HOLD exists.
Default = the design as approved (S4 fractions 0.3 / 0.6, six arms per task, window [0, 2000)). Options (each the
user's call; AUDIT.md "Open decisions"), every one written into the job name so cells never mix in one out/ dir:
  --always        S4 cells also run R@always, R_sign@always, specrand@always (rescue-capacity positive control, +3 arms)
  --s4fracs F,G   S4 train fractions instead of 0.3,0.6 (e.g. 0.45,0.6; a fraction needs a BUDGET entry)
  --oh            dim cells also run specrand_oh (rank-6 control for onehot, rule (d), +1 arm)
  --posctl        also job_s4-f0.9-posctl (none at frac 0.9, 5 seeds, 25k steps; descriptive only)"""
import hashlib, math, shutil, subprocess, sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/scratch/yh6210/llmadd-burst")
R = B / "runs" / "toystruct"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
FILES = ("toy_struct.py", "test_toy_struct.py", "toy_grok.py")
SEEDS = (0, 1, 2, 3, 4)
ARMS = {"s4": "none,R,R_sign,absR,specrand,shuf", "dim": "none,expo,quot,onehot,specrand,shuf"}
SPLIT = {"s4": "question", "dim": "class"}   # explicit in every job (toy_struct --split auto gives the same)
# PRE-REGISTERED (AUDIT.md "Budget"): 0.3 / 0.6 before the first pilot; 0.45 (a candidate S4 fraction) set to 50k on
# 2026-10-16 before its none pilot ran. The none-pilot rule may double a fraction's budget once.
BUDGET = {0.3: 50000, 0.45: 50000, 0.6: 25000, 0.9: 25000}
MS_PER_STEP = 23.0   # GPU, per process with 5 processes sharing one GPU: measured toy-multifreq (toy_grok, same model, on
#                      gh015 = A100/H100, not L40S, without --threads 1; 573 s per 25k-step run incl. evals every 100);
#                      this code on CPU, 1 thread: AUDIT.md "Cost". The first finished job recalibrates (ms_per_step).
MARGIN = 2.0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def snapshot() -> str:
    code = R / "code" / "tools"
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    if not code.exists():
        code.mkdir(parents=True)
        for f in FILES:
            shutil.copy2(REPO / "tools" / f, code / f)
        dirty = git("status", "--porcelain", "--", *[f"tools/{f}" for f in FILES])
        (R / "code" / "COMMIT").write_text(
            git("rev-parse", "--short", "HEAD") + (" + UNCOMMITTED working-tree files (see sha256 below):\n" + dirty if dirty else "")
            + "\n" + "".join(f"sha256[:12] {sha(code / f)} tools/{f}\n" for f in FILES))
    for f in FILES:   # a stale snapshot would run silently: refuse
        assert sha(code / f) == sha(REPO / "tools" / f), (f"snapshot {code / f} differs from the repo; nothing submitted yet? "
                                                         f"then delete {R / 'code'} and rerun")
    return sha(code / "toy_struct.py")


def opt(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    R.mkdir(parents=True, exist_ok=True)
    code_sha = snapshot()
    n = 0
    s4fracs = [float(x) for x in opt("--s4fracs", "0.3,0.6").split(",")]
    always, oh = "--always" in sys.argv, "--oh" in sys.argv
    cells = []
    for f in s4fracs:
        assert f in BUDGET, f"no pre-registered budget for frac {f}"
        cells.append(("s4", f, ARMS["s4"] + (",R@always,R_sign@always,specrand@always" if always else ""), BUDGET[f],
                      "-always" if always else ""))
    for f in (0.3, 0.6):
        cells.append(("dim", f, ARMS["dim"] + (",specrand_oh" if oh else ""), BUDGET[f], "-oh" if oh else ""))
    if "--posctl" in sys.argv:
        cells.append(("s4", 0.9, "none", BUDGET[0.9], "-posctl"))
    keep = {f"job_{t}-f{f}{s}.sbatch" for t, f, _, _, s in cells}
    for old in R.glob("job_*.sbatch"):   # a job file of another option set must not be submitted by mistake
        if old.name not in keep and not (R / f"SUBMITTED_{old.stem[4:]}").exists():
            old.unlink(); print("removed stale", old.name)
    for task, frac, arms, steps, suffix in cells:
        n_arms = len(arms.split(","))
        est_h = n_arms * steps * MS_PER_STEP / 1000 / 3600
        hrs = max(1, math.ceil(MARGIN * est_h))
        tag = f"{task}-f{frac}{suffix}"
        out = R / "out" / tag
        job = R / f"job_{tag}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_toystruct-{tag}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={hrs:02d}:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# toy structure prior, task {task}, train fraction {frac} ({SPLIT[task]} split), budget {steps} steps, arms {arms}, seeds {','.join(map(str, SEEDS))}
# estimate {est_h:.2f} h per process ({n_arms} arms x {steps} steps x {MS_PER_STEP} ms), --time = {MARGIN:g}x rounded up
# code = runs/toystruct/code (COMMIT; toy_struct.py sha256[:12] {code_sha}); intent spec + rules runs/toystruct/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/toystruct/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p {out}
# targets, CKA matrix (targets + initial W_E), closure fractions, every build-time assert (fails the job if one fails)
{PY} toy_struct.py --task {task} --check --frac {frac} --split {SPLIT[task]} --seeds {','.join(map(str, SEEDS))} > {out}/check.txt 2>&1
# finished runs are skipped on a requeue (run_<arm>_seed<s>.json, written atomically); a run cut by preemption restarts from step 0
pids=""
for s in {' '.join(map(str, SEEDS))}; do
  {PY} toy_struct.py --task {task} --frac {frac} --split {SPLIT[task]} --steps {steps} --seeds $s --arms {arms} --win 0,2000 --lam 1 \\
    --eval_every 100 --threads 1 --out {out} >> {out}/log_seed$s.txt 2>&1 &
  pids="$pids $!"
done
fail=0
for p in $pids; do wait $p || fail=1; done
# the pre-registered tests, once, after every seed process has finished
{PY} toy_struct.py --summarize {out}
if [ $fail -ne 0 ]; then echo "a seed process failed: see {out}/log_seed*.txt"; exit 1; fi
""")
        n += 1
        flag = R / f"SUBMITTED_{tag}"
        print(f"{job.name}: --time {hrs:02d}:00:00 (estimate {est_h:.2f} h, {n_arms} arms)")
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text(o.stdout)
            else:
                print(tag, o.stderr.strip())
    print(n, "jobs written to", R, "(not submitted)" if "--submit" not in sys.argv else "")


if __name__ == "__main__":
    main()
