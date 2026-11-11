"""Torch jobs (P1, 2026-10-14): experiment B (operands 0..999, whole tokens) with the prior only while it helps:
self-timed gate (on while the in-batch first-line exact match EMA < 0.5) and window (0.05, 0.4), for helix3 and the
per-place circles; val bits/byte logged as the language-modelling cost (whole-run priors cost +1-2% in ptrange).
References from ptrange (whole run): none, helix3, digitcirc. Seeds 0-3. Account torch_pr_625_general, public L40S.
--submit submits the jobs not yet submitted (runs/ptrangeg/SUBMITTED_<tag>)."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptrangeg"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
conds = [("helix3-gate", "--target helix3 --schedule gate --gate_thr 0.5"),
         ("helix3-w0.05-0.4", "--target helix3 --schedule window --window 0.05,0.4"),
         ("digitcirc-gate", "--target digitcirc --schedule gate --gate_thr 0.5")]
n = 0
for tg, extra in conds:
    for s in range(4):
        tag = f"whole-{tg}-s{s}"
        job = R / f"job_{tag}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptrangeg-{tag}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# experiment B with gated / windowed priors; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/whole-{tg} {extra} --seed {s} --tag _{tg} --save_model"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS
""")
        n += 1
        flag = R / f"SUBMITTED_{tag}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            print(job.name, o.stdout.strip() or o.stderr.strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
print(n, "jobs")
