"""Torch jobs: pretraining reruns with the in-context held-out eval (--ctx_eval) and saved weights (2026-10-13).
pt7c = 4 arithmetic lines per sequence (~7% of tokens), arms none / helix / helix_shuf / helix_window, seeds 0-3;
pt2c = 1 line (~2%), arms none / helix / helix_shuf, seeds 0-3. One run dir per batch; --submit submits unsubmitted jobs."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
batches = {"pt7c": (4, 250, ("none", "helix", "helix_shuf", "helix_window")), "pt2c": (1, 500, ("none", "helix", "helix_shuf"))}
for name, (lines, every, arms) in batches.items():
    R = B / "runs" / name
    if not (R / "code").exists():
        shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
        (R / "code" / "COMMIT").write_text(commit + "\n")
    for arm in arms:
        for seed in range(4):
            job = R / f"job_{arm}_s{seed}.sbatch"
            job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_{name}-{arm}-s{seed}
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# pretraining rerun with the in-context held-out eval; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py --data {B}/pretrain_data --tokenizer {TOK} --out {R}/out --arm {arm} --seed {seed} --arith_lines {lines} --tokens 3e8 --eval_every {every} --ctx_eval --save_model
""")
            flag = R / f"SUBMITTED_{arm}_s{seed}"
            if "--submit" in sys.argv and not flag.exists():
                o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
                print(job.name, o.stdout.strip() or o.stderr.strip())
                if o.returncode == 0:
                    flag.write_text(o.stdout)
