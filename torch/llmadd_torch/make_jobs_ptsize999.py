"""Torch jobs (2026-10-14, user: unify on 0..999 -> the size ladder moves from 10..99 (ptsize) to the unified setup):
tools/pretrain_range.py, operands 0..999, helix3, 3e8 tokens, same data / split / evals as pt999; OLMo-2-style models of
190M (d 768, 12 layers, lr 6e-4) and 505M (d 1024, 24 layers, lr 3e-4) as in ptsize; prior layers default to 1/3 and
2/3 of the depth (4,8 / 8,16; 53M: 2,4). Arms, seeds 0-3:
  190M: 7% none, gate-ctx, gate-text, shuf-ctx; 2% none, gate-ctx, gate-text
  505M: 7% none, gate-ctx, gate-text
Account torch_pr_625_general on the public L40S partition. --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptsize999"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
G = "--schedule gate --gate_thr 0.5"
ARM = {"none": "--target none", "gate-ctx": f"--target helix3 {G} --site ctx", "gate-text": f"--target helix3 {G} --site text",
       "shuf-ctx": f"--target helix3_shuf {G} --site ctx", "shuf-text": f"--target helix3_shuf {G} --site text"}
SIZES = {"190m": "--d_model 768 --n_layers 12 --n_heads 12 --d_ff 3072 --lr 6e-4",
         "500m": "--d_model 1024 --n_layers 24 --n_heads 16 --d_ff 4096 --lr 3e-4"}
jobs = []   # (tag, args, seeds)
for a in ("none", "gate-ctx", "gate-text", "shuf-ctx"):
    jobs.append((f"190m-7pct-{a}", ARM[a] + " --arith_lines 4 " + SIZES["190m"], range(4)))
for a in ("none", "gate-ctx", "gate-text"):
    jobs.append((f"190m-2pct-{a}", ARM[a] + " --arith_lines 1 " + SIZES["190m"], range(4)))
    jobs.append((f"500m-7pct-{a}", ARM[a] + " --arith_lines 4 " + SIZES["500m"], range(4)))
# 2026-10-15 (user): window arms for the ladder -- the main recipe is moving to the 5% window (F79: the step-0 gate
# costs as much LM as the whole-run prior); the snapshot (fdfadfd) already supports --schedule window
WIN = "--target helix3 --schedule window --window 0.05,0.4 --site ctx"
for sz in ("190m", "500m"):
    jobs.append((f"{sz}-7pct-win-ctx", WIN + " --arith_lines 4 " + SIZES[sz], range(4)))
n = 0
for tag, extra, seeds in jobs:
    for s in seeds:
        jt = f"{tag}-s{s}"
        job = R / f"job_{jt}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptsize999-{jt}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# unified 0..999 setup; code = llmadd-burst commit in code/COMMIT; intent spec runs/ptsize999/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/ptsize999/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {extra} --seed {s} --tag _{tag} {'--save_model ' if True else ''}--ckpt_every 2000"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS
""")
        n += 1
        flag = R / f"SUBMITTED_{jt}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text(o.stdout)
            else:
                print(jt, o.stderr.strip())
print(n, "jobs")
