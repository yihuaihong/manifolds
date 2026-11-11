"""Torch jobs for the broader-evaluation batch (2026-10-12). Writes runs/<name>/{code,job.sbatch}; with --submit,
submits every job whose checkpoint is fully downloaded and that has not been submitted yet (runs/<name>/SUBMITTED).

B1  new operations on the same operands (mul / mod / max) x 7B 9B, 42B, 84B, 210B: none / helix / helix_shuf, 3 seeds
B2  add (42B, 84B) and sub (42B) with the GPT-3 arithmetic + word-problem evals and learning curves
B3  learning curves at 9B / 7000 pairs, 10 epochs: none and helix (also the helix ceiling at 10 epochs)
Code: a snapshot of llmadd-burst/tools (commit in code/COMMIT); results are copied back to the repo afterwards.
"""
import shutil
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
CK = {"9B": "stage1-step2000-tokens9B", "210B": "stage1-step50000-tokens210B", "21B": "stage1-step5000-tokens21B", "42B": "stage1-step10000-tokens42B", "84B": "stage1-step20000-tokens84B",
      "210B": "stage1-step50000-tokens210B"}
G3 = f"--gpt3_arith {B}/data/arithmetic"
jobs = []  # multiplicative geometries (log / factor / helix_factor)
MUL = "--task mul --train_pairs 900 --test_pairs 500"
for ck in ("42B", "84B"):
    jobs.append((f"fx-mul-{ck}", ck, f"{MUL} {G3} --geoms none,helix,log,factor,factor_shuf,helix_factor --seeds 0,1,2", "06:00:00"))
    jobs.append((f"fx-mixed-{ck}", ck, f"--task mixed {G3} --geoms none,factor,helix_factor --seeds 0,1,2", "08:00:00"))
jobs.append(("fx-add-42B", "42B", f"--task add {G3} --geoms none,helix,factor,helix_factor --seeds 0,1,2", "06:00:00"))

commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
for name, ck, args, tl in jobs:
    R = B / "runs" / name
    if not (R / "code").exists():
        shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
        (R / "code" / "COMMIT").write_text(commit + "\n")
    model = B / "models/OLMo-2-1124-7B" / CK[ck]
    (R / "job.sbatch").write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_{name}
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time={tl}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# broader-evaluation batch; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache
export PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python manifold_train.py --model {model} --out {R}/out --mode lora --lam 20 --save_preds {args}
""")
    ready = (model / "config.json").exists() and len(list(model.glob("model-*.safetensors"))) == 6 and not list(model.glob("**/*.incomplete"))
    if "--submit" in sys.argv and ready and not (R / "SUBMITTED").exists():
        out = subprocess.run(["sbatch", str(R / "job.sbatch")], capture_output=True, text=True)
        print(name, out.stdout.strip() or out.stderr.strip())
        if out.returncode == 0:
            (R / "SUBMITTED").write_text(out.stdout)
    elif "--submit" in sys.argv:
        print(name, "submitted earlier" if (R / "SUBMITTED").exists() else "waiting for the checkpoint")
