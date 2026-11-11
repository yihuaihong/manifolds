"""Torch jobs: rainbow step 2 (tools/pretrain_rainbow.py) in from-scratch pretraining; arms none, singles,
all three, all-but-one, all shuffled, all @norm; seeds 0-3. --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptrainbow"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
assert (R / "code").exists()
if False:
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
# 2026-10-14 replication (unexpected result: the number helix, not the months circle, makes held-out months
# generalise, and adding the months circle on top removes it; seeds 0-3 ranged 9-78%): seeds 4-7 for the arms
# involved, same code snapshot (runs/ptrainbow/code), public L40S with the 625 account
ARMS = ["none", "helix", "months", "helix+months", "helix+days", "helix+days+months", "days+months", "helix_shuf"]
SEEDS = range(4, 8)
SEEDS_FOR = {"helix_shuf": range(8)}   # audit 2026-10-14: wrong-geometry control for the months effect (training dynamics)
for arm in ARMS:
    for s in SEEDS_FOR.get(arm, SEEDS):
        tag = f"{arm.replace('+', '-').replace('@', '-')}-s{s}"
        job = R / f"job_{tag}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptrainbow-{tag}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# rainbow step 2 (pretraining); code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --out {R}/out --priors {arm} --seed {s} --save_model"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_rainbow.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_rainbow.py $ARGS
""")
        flag = R / f"SUBMITTED_{tag}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            print(job.name, o.stdout.strip() or o.stderr.strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
