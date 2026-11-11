"""Torch jobs: tokenisation control (tools/pretrain_digits.py). Whole-number vs digit-split tokens x priors
none / helix / helix window (.05, .4) / digitcirc / digitcirc window (.05, .4) / helix_shuf, 7% arithmetic,
unordered held-out split, seeds 0-3. --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
DATA = {"whole": B / "pretrain_data", "digits": B / "pretrain_data_digits"}
R = B / "runs" / "ptdig"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
conds = [("none", "--arm none"), ("helix", "--arm helix"), ("helix-w05", "--arm helix --window 0.05,0.4"),
         ("digitcirc", "--arm digitcirc"), ("digitcirc-w05", "--arm digitcirc --window 0.05,0.4"), ("helix_shuf", "--arm helix_shuf")]
for tk in DATA:
    for c, extra in conds:
        for s in range(4):
            tag = f"{tk}-{c}-s{s}"
            job = R / f"job_{tag}.sbatch"
            job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptdig-{tag}
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# tokenisation control; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {DATA[tk]} --tokenizer {TOK} --out {R}/out/{tk}-{c} --tokenization {tk} {extra} --seed {s} --tag _{c} --save_model"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_digits.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_digits.py $ARGS
""")
            flag = R / f"SUBMITTED_{tag}"
            if "--submit" in sys.argv and not flag.exists():
                o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
                print(job.name, o.stdout.strip() or o.stderr.strip())
                if o.returncode == 0:
                    flag.write_text(o.stdout)
