"""2026-10-19 F84 补充（定稿） follow-ups: the cheapest next steps of notes/batch1_build/audit_toy_optimiser_sweeps.md §6,
plus the coordinator's Latin addition (c). Intent spec, cells and the PRE-REGISTERED reading: runs/toyf84fu/AUDIT.md.
  g1  S4 0.6 same-device control: none, R@always, R_sign@always at lr 1e-2 wd 1, one job on one L40S (the existing
      best-vs-best paired none on the L40S with R on the Burst A100).
  g2  S4 0.6 upper lr edge: none at lr 3e-2 (wd 1, 3); R@always at lr 3e-2 wd 1.
  g3  dim 0.3 none: lr 1e-2 wd 0.03; lr 2e-2 wd 0.1 and wd 1.
  g4  dim 0.3 prior lr scan: F84's dim prior arm expo (window [0, 2000), lam 1) and specrand (same schedule) at lr 3e-3
      and 1e-2, wd 1 (= F84's default wd).
  g5  S4 0.2 / 0.3 / 0.45 prior lr scan (coordinator (c); toyshape runs R@always at lr 1e-3 / 1e-2 only, 50k, and no
      specrand): R@always and specrand@always at lr 3e-3 / 1e-2 / 3e-2, wd 1, 200k steps (= toyshape's none budget),
      toyshape's seeds (S4 0.2: 0, 1, 2, 4, 5).
Budgets / splits as the existing cells: S4 question split, 100k steps at 0.6 (toynone2) and 200k at 0.2-0.45 (toyshape
none); dim class split frac 0.3, 200k steps. 5 seed processes per job, one L40S.
Code: tools/toy_struct.py at the pin (git archive of llmadd-burst 398935a; sha256[:12] b74a78cb5472, the file of
toynone2 / toynone3 / toyprior_lr / Burst 480-482), unmodified, run through code/hits_wrap.py (logs per-question
held-out hits; A/B bitwise-equal otherwise; runs/toyf84fu/preflight). Every job re-checks both shas before any GPU work
(exit 3) and refuses a non-L40S GPU (exit 4).
    python make_jobs_toyf84fu.py                       # render (and create the code snapshot once)
    python make_jobs_toyf84fu.py --print [--groups g1,g4]   # list what --submit would sbatch (no sbatch call)
    python make_jobs_toyf84fu.py --submit [--groups g1,g4]  # the user's call: sbatch every job (of those groups) without
                                                            # a SUBMITTED_<name> flag
HOLD: a pending job exits at start when runs/toyf84fu/HOLD exists (no scancel needed)."""
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/scratch/yh6210/llmadd-burst")
R = B / "runs" / "toyf84fu"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
PIN_COMMIT, PIN_SHA = "398935a", "b74a78cb5472"
WRAP_SHA = "c2384fbe7fb4"   # sha256[:12] of runs/toyf84fu/code/hits_wrap.py (written by hand, A/B-tested in preflight/)
CODE = R / "code"
sha12 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
args = sys.argv[1:]
GROUPS = {"g1", "g2", "g3", "g4", "g5"}
sel = set(GROUPS)
if "--groups" in args:
    i = args.index("--groups")
    sel = set(args[i + 1].split(",")) if i + 1 < len(args) else set()
    assert sel and sel <= GROUPS, f"--groups takes a comma list of {sorted(GROUPS)}: {args[i + 1:i + 2]}"
    del args[i:i + 2]
    assert "--submit" in args or "--print" in args, "--groups only with --submit or --print"
assert set(args) <= {"--submit", "--print"} and not {"--submit", "--print"} <= set(args), f"bad option(s): {args}"

if not (CODE / "tools" / "toy_struct.py").exists():
    (CODE / "tools").mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "-C", str(REPO), "archive", PIN_COMMIT, "tools/toy_struct.py", "tools/test_toy_struct.py"],
                         capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(CODE)
    full = subprocess.run(["git", "-C", str(REPO), "rev-parse", PIN_COMMIT], capture_output=True, text=True, check=True).stdout.strip()
    (CODE / "COMMIT").write_text(f"{PIN_COMMIT} (git archive of llmadd-burst {full}): tools/toy_struct.py sha256[:12] "
                                 f"{sha12(CODE / 'tools/toy_struct.py')}, tools/test_toy_struct.py sha256[:12] "
                                 f"{sha12(CODE / 'tools/test_toy_struct.py')}; the toy_struct.py of toynone2 / toynone3 / "
                                 f"toyprior_lr / Burst 480-482. Run through code/hits_wrap.py (not from git; sha in the generator).\n")
assert sha12(CODE / "tools" / "toy_struct.py") == PIN_SHA, "snapshot toy_struct.py is not the pinned file"
assert (CODE / "hits_wrap.py").exists(), "code/hits_wrap.py missing (see runs/toyf84fu/AUDIT.md)"
assert sha12(CODE / "hits_wrap.py") == WRAP_SHA, f"code/hits_wrap.py changed: {sha12(CODE / 'hits_wrap.py')} != {WRAP_SHA}"

S4_06 = ("s4", 0.6, "question", 100000)
DIM = ("dim", 0.3, "class", 200000)
s4low = lambda f: ("s4", f, "question", 200000)
SEEDS = {("s4", 0.2): "0 1 2 4 5"}   # toyshape's rule: the first five seeds whose split passes (seed 3 fails at 0.2)
WIN = "--win 0,2000 --lam 1"         # F84's schedule for windowed arms (expo, specrand on dim); defaults, explicit here
JOBS = [  # (group, cell, lr, wd, arms)
    ("g1", S4_06, "1e-2", "1", "none,R@always,R_sign@always"),
    ("g2", S4_06, "3e-2", "1", "none,R@always"),
    ("g2", S4_06, "3e-2", "3", "none"),
    ("g3", DIM, "1e-2", "0.03", "none"),
    ("g3", DIM, "2e-2", "0.1", "none"),
    ("g3", DIM, "2e-2", "1", "none"),
    ("g4", DIM, "3e-3", "1", "expo,specrand"),
    ("g4", DIM, "1e-2", "1", "expo,specrand"),
] + [("g5", s4low(f), lr, "1", "R@always,specrand@always") for f in (0.2, 0.3, 0.45) for lr in ("3e-3", "1e-2", "3e-2")]
MS = {("s4", "none"): 3.1, ("s4", "prior"): 3.8, ("dim", "none"): 2.7, ("dim", "prior"): 2.75}   # L40S ms/step/process, 5 per GPU

names, total_h = set(), 0.0
for g, (task, frac, split, steps), lr, wd, arms in JOBS:
    name = f"{g}-{task}-f{frac}-lr{lr}-wd{wd}"
    assert name not in names, name
    names.add(name)
    seeds = SEEDS.get((task, frac), "0 1 2 3 4")
    out = R / "out" / name
    est_h = sum(steps * MS[(task, "none" if a == "none" else "prior")] for a in arms.split(",")) / 3.6e6
    total_h += est_h
    job = R / f"job_{name}.sbatch"
    cmd = (f"{PY} ../hits_wrap.py --task {task} --frac {frac} --split {split} --steps {steps} --seeds $s --arms {arms} {WIN} "
           f"--lr {lr} --wd {wd} --eval_every 100 --threads 1 --out {out}")
    job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_toyf84fu-{name}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# toyf84fu group {g}: task {task}, frac {frac} ({split} split), lr {lr}, wd {wd}, {steps} steps, arms {arms}, seeds {seeds}
# estimate {est_h:.2f} h per job (5 seed processes in parallel on one L40S); intent spec + pre-registered reading runs/toyf84fu/AUDIT.md
set -eu
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {CODE}/tools
[ "$(sha256sum toy_struct.py | cut -c1-12)" = "{PIN_SHA}" ] || {{ echo "toy_struct.py is not the pinned file"; exit 3; }}
[ "$(sha256sum ../hits_wrap.py | cut -c1-12)" = "{WRAP_SHA}" ] || {{ echo "hits_wrap.py changed"; exit 3; }}
gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader)
echo "$gpu"
mkdir -p {out}
echo "$(date -Is) job ${{SLURM_JOB_ID:-none}} restart ${{SLURM_RESTART_COUNT:-0}} host $(hostname) gpu $gpu" >> {out}/gpu.txt
case "$gpu" in *L40S*) ;; *) echo "not an L40S: $gpu"; exit 4;; esac
# finished runs are skipped on a requeue (run_<arm>_seed<s>.json, written atomically); a run cut by preemption restarts from step 0
pids=""
for s in {seeds}; do
  {cmd} >> {out}/log_seed$s.txt 2>&1 &
  pids="$pids $!"
done
fail=0
for p in $pids; do wait $p || fail=1; done
{PY} toy_struct.py --summarize {out}
[ $fail -eq 0 ] || {{ echo "a seed process failed: see {out}/log_seed*.txt"; exit 1; }}
""")
    flag = R / f"SUBMITTED_{name}"
    if "--print" in args and g in sel and not flag.exists():
        print("would submit", name)
    if "--submit" in args and g in sel and not flag.exists():
        o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
        if o.returncode == 0:
            flag.write_text(o.stdout)
            print("submitted", name, o.stdout.strip())
        else:
            print(o.stderr)
print(len(JOBS), "jobs", f"estimate {total_h:.1f} L40S-h (sum of per-job wall estimates; --time 2 h each)")
