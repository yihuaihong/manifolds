"""Torch jobs: from-scratch pretraining pilot with a helix prior (pretrain_helix.py). --submit submits the jobs not yet
submitted once the token files exist. Arms: none / helix / helix_shuf / helix_window; seeds from --seeds."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
seeds = next((a.split("=")[1] for a in sys.argv if a.startswith("--seeds=")), "0").split(",")
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ready = (B / "pretrain_data/train.bin").exists() and (B / "pretrain_data/meta.json").exists()
for seed in seeds:
    for arm in ("none", "helix", "helix_shuf", "helix_window"):
        name = f"pt-{arm}-s{seed}"; R = B / "runs" / name
        if not (R / "code").exists():
            shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
            (R / "code" / "COMMIT").write_text(commit + "\n")
        (R / "job.sbatch").write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_{name}
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# from-scratch pretraining pilot; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py --data {B}/pretrain_data --tokenizer {TOK} --out {R}/out --arm {arm} --seed {seed} --tokens 3e8 --eval_every 500
""")
        if "--submit" in sys.argv and ready and not (R / "SUBMITTED").exists():
            o = subprocess.run(["sbatch", str(R / "job.sbatch")], capture_output=True, text=True)
            print(name, o.stdout.strip() or o.stderr.strip())
            if o.returncode == 0:
                (R / "SUBMITTED").write_text(o.stdout)
        elif "--submit" in sys.argv:
            print(name, "submitted earlier" if (R / "SUBMITTED").exists() else "waiting for the data")
