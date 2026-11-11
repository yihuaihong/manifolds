"""2026-10-19 (F84 correction, every arm at its own best lr): the S4 prior arms (R / R_sign / specrand, always on) at lr
3e-3 and 1e-2 (wd 1, 100k steps, 5 seeds), to compare best-vs-best with the tuned none of runs/toynone2. Intent: runs/toynone/AUDIT.md.
2026-10-18 RE-RUN of the lr axis: in runs/toynone the warmup line hard-coded lr 1e-3, so --lr was inert (every lr ran at 1e-3;
wd and the 4x budget were valid). Fixed in tools/toy_struct.py (warmup uses args.lr; lr_now logged and asserted). Here only
lr != 1e-3.
Torch jobs (2026-10-17, user: make "without the prior it is not learned" harder, F84): the toy none arm with its
optimiser swept and a 4x budget, at the two cells where F84 reads (c): S4 frac 0.6 (question split; F84: none 0/5 in 25k,
R@always 5/5) and dim frac 0.3 (class split; none 0/5 in 50k, expo 4/5).
Grid: lr {3e-4, 1e-3, 3e-3, 1e-2} x weight decay {0.1, 1, 3}; seeds 0-4; S4 100k steps, dim 200k steps; arm none only.
Code: tools/toy_struct.py with --lr / --wd (defaults = the toy_grok optimiser; the default path is bitwise identical to the
F84 snapshot on CPU, checked 2026-10-17). One job per (task, lr, wd), 5 seed processes. Intent spec: runs/toynone/AUDIT.md.
    python make_jobs_toynone.py [--submit]"""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
R = B / "runs" / "toyprior_lr"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    (R / "code" / "tools").mkdir(parents=True)
    for f in ("toy_struct.py", "test_toy_struct.py"):
        shutil.copy(REPO / "tools" / f, R / "code" / "tools" / f)
    sha = subprocess.run(["sha256sum", str(R / "code/tools/toy_struct.py")], capture_output=True, text=True).stdout[:12]
    (R / "code" / "COMMIT").write_text(f"{commit} + toy_struct.py (sha256[:12] {sha}) with --lr / --wd\n")
CELLS = {"s4": (0.6, "question", 100000)}
LRS, WDS = ("3e-3", "1e-2"), ("1",)   # the none optimum region (toynone2: S4 none 5/5 at lr 1e-2 wd 1)
n = 0
for task, (frac, split, steps) in CELLS.items():
    for lr in LRS:
        for wd in WDS:
            name = f"{task}-f{frac}-lr{lr}-wd{wd}"
            out = R / "out" / name
            job = R / f"job_{name}.sbatch"
            job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_toypriorlr-{name}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# toy none arm, task {task}, frac {frac} ({split} split), lr {lr}, wd {wd}, {steps} steps, seeds 0-4; intent spec runs/toynone/AUDIT.md (amendment 2026-10-18)
set -eu
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p {out}
pids=""
for s in 0 1 2 3 4; do
  {PY} toy_struct.py --task {task} --frac {frac} --split {split} --steps {steps} --seeds $s --arms R@always,R_sign@always,specrand@always --win 0,2000 --lam 1 --lr {lr} --wd {wd} \\
    --eval_every 100 --threads 1 --out {out} >> {out}/log_seed$s.txt 2>&1 &
  pids="$pids $!"
done
fail=0
for p in $pids; do wait $p || fail=1; done
{PY} toy_struct.py --summarize {out}
[ $fail -eq 0 ] || {{ echo "a seed process failed"; exit 1; }}
""")
            n += 1
            flag = R / f"SUBMITTED_{name}"
            if "--submit" in sys.argv and not flag.exists():
                o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
                if o.returncode == 0:
                    flag.write_text(o.stdout)
                else:
                    print(o.stderr)
print(n, "jobs")
