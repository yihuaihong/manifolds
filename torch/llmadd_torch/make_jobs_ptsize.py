"""Torch jobs (P2, 2026-10-14): size ladder for the pretraining result (F71 at 53M). Same data, split, evals and
3e8 tokens; OLMo-2-style models of 190M (d 768, 12 layers) and 505M (d 1024, 24 layers) parameters (tied 100k-token
embedding included; the F71 model is 53M), lr 6e-4 / 3e-4. Arms: none, gate-ctx (the default recipe: in-context prior, self-timed gate),
ctx-shuf (wrong geometry) at 7% arithmetic; none, gate-ctx at 2%. Seeds 0-3. Resumable checkpoints every 2000 steps.
Account torch_pr_625_general on the public L40S partition.
--submit submits the jobs not yet submitted (runs/ptsize/SUBMITTED_<tag>)."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptsize"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
SIZES = {"190m": "--d_model 768 --n_layers 12 --n_heads 12 --d_ff 3072 --lr 6e-4",
         "500m": "--d_model 1024 --n_layers 24 --n_heads 16 --d_ff 4096 --lr 3e-4"}
GATE = "--arm helix_gate --gate_thr 0.5 --gate_start 0 --geo_site ctx"
ARMS = [("none", "--arm none", 4), ("gate-ctx", GATE, 4), ("ctx-shuf", "--arm helix_shuf --geo_site ctx", 4),
        ("2pct-none", "--arm none", 1), ("2pct-gate-ctx", GATE, 1)]
n = 0
for sz, cfg in SIZES.items():
    for arm, extra, lines in ARMS:
        for seed in range(4):
            tag = f"{sz}-{arm}-s{seed}"
            job = R / f"job_{tag}.sbatch"
            job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptsize-{tag}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# size ladder; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --out {R}/out/{sz}-{arm} {extra} {cfg} --seed {seed} --tag _{sz}-{arm} --split unordered --arith_lines {lines} --tokens 3e8 --eval_every 250 --ctx_eval --save_model --ckpt_every 2000"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py $ARGS
""")
            n += 1
            flag = R / f"SUBMITTED_{tag}"
            if "--submit" in sys.argv and not flag.exists():
                o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
                print(job.name, o.stdout.strip() or o.stderr.strip())
                if o.returncode == 0:
                    flag.write_text(o.stdout)
print(n, "jobs")
