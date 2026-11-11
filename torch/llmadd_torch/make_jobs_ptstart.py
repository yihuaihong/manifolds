"""Torch jobs: start / end of the helix window in from-scratch pretraining (7% arithmetic, in-context eval, saved
weights). Vary start with end .4: (0, .4), (.05, .4), (.2, .4); vary end with start .1: (.1, .2), (.1, 1.0); seeds 0-3.
(.1, .4) and none are in pt7c (seeds 0-3); seeds 4, 5 of both are added here to replicate F65 (audit check iv).
Intent spec: runs/ptstart/AUDIT.md. --submit submits the jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptstart"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
jobs = [(f"w{w.replace(',', '-')}", "helix_window", w, s) for w in ("0,0.4", "0.05,0.4", "0.2,0.4", "0.1,0.2", "0.1,1.0") for s in range(4)]
jobs += [("none", "none", "0.1,0.4", s) for s in (4, 5)] + [("w0.1-0.4", "helix_window", "0.1,0.4", s) for s in (4, 5)]
for tag, arm, w, seed in jobs:
    job = R / f"job_{tag}_s{seed}.sbatch"
    job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptstart-{tag}-s{seed}
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# window start/end sweep; code = llmadd-burst commit in code/COMMIT; intent spec in runs/ptstart/AUDIT.md
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --out {R}/out/{tag} --arm {arm} --window {w} --seed {seed} --arith_lines 4 --tokens 3e8 --eval_every 250 --ctx_eval --save_model"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py $ARGS
""")
    flag = R / f"SUBMITTED_{tag}_s{seed}"
    if "--submit" in sys.argv and not flag.exists():
        o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
        print(job.name, o.stdout.strip() or o.stderr.strip())
        if o.returncode == 0:
            flag.write_text(o.stdout)
print(len(jobs), "jobs")
