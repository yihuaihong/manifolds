"""Torch jobs: rainbow step 2 (tools/pretrain_rainbow.py) in from-scratch pretraining; arms none, singles,
all three, all-but-one, all shuffled, all @norm; seeds 0-3. --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptrainbow"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
ARMS = ["none", "helix", "days", "months", "helix+days+months", "days+months", "helix+months", "helix+days",
        "helix_shuf+days_shuf+months_shuf", "helix+days+months@norm", "helix+days+months@bal"]
for arm in ARMS:
    for s in range(4):
        tag = f"{arm.replace('+', '-').replace('@', '-')}-s{s}"
        job = R / f"job_{tag}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptrainbow-{tag}
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=a100_tandon,h100_tandon
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
