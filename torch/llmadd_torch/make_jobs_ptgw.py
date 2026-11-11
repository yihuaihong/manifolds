"""Torch jobs (2026-10-14, user-approved): WINDOW vs DELAYED GATE across settings (runs/ptgw, intent spec AUDIT.md).
Unified 0..999 setup (tools/pretrain_range.py, 53M, helix3 prior on the operands in the lines = ctx, 7% arithmetic).
Schedules (all start at 5% of training):
  W     window (0.05, 0.4)
  D05   delayed gate: from 5%, on while the lowest per-bucket in-batch EMA < 0.5 (the existing gate rule)
  D09h  delayed gate: from 5%, threshold 0.9, hysteresis 0.1 (a closed gate reopens only below 0.8)
Settings: main (lr 1e-3, layers 2,4, 3e8 tokens); lr 5e-4; prior on all 6 layers; 6e8 tokens (the window and the
gate's start are fractions of training, so they move with the budget).
None references with saved models where none exist yet: lr 5e-4 and 6e8 tokens (main / all-layers use pt999's
7pct-none, 8 seeds, saved). Seeds 0-3, every run saves its model (offline LM re-evaluation, tools/offline_lm.py).
Free references already queued: pt999 7pct-gate-ctx (gate from step 0), 7pct-whole-ctx, sens-lr5e-4-gate-ctx,
sens-Lall-gate-ctx; pt999b 7pct-win-ctx (= main-W on another snapshot, no saved models).
Account torch_pr_625_general on the public L40S partition. --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptgw"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "tools"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    assert not dirty, f"tools/ has uncommitted changes, commit first:\n{dirty}"
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
SETTINGS = {"main": "", "lr5e-4": "--lr 5e-4", "Lall": "--geo_layers 1,2,3,4,5,6", "tok6e8": "--tokens 6e8"}
SCHED = {"W": "--target helix3 --schedule window --window 0.05,0.4",
         "D05": "--target helix3 --schedule gate --gate_start 0.05 --gate_thr 0.5",
         "D09h": "--target helix3 --schedule gate --gate_start 0.05 --gate_thr 0.9 --gate_hyst 0.1",
         "none": "--target none"}
cells = [(st, sc) for st in SETTINGS for sc in ("W", "D05", "D09h")] + [("lr5e-4", "none"), ("tok6e8", "none")]
n = 0
R.mkdir(parents=True, exist_ok=True)
for st, sc in cells:
    tag = f"{st}-{sc}"
    hours = 16 if st == "tok6e8" else 8
    for s in range(4):
        jt = f"{tag}-s{s}"
        job = R / f"job_{jt}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptgw-{jt}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={hours:02d}:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# window vs delayed gate across settings; code = llmadd-burst commit in code/COMMIT; intent spec runs/ptgw/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/ptgw/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {SCHED[sc]} --site ctx {SETTINGS[st]} --arith_lines 4 --seed {s} --tag _{tag} --save_model --ckpt_every 2000"
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
