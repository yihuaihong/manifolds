"""Torch jobs (2026-10-14, F77 audit follow-up): two control targets for the teacher-target result, same setup and code
snapshot as ptteach (runs/ptteach/code; 7% arithmetic, ctx site, unordered split, seeds 0-3), 625 account on L40S:
  ctl-helix-noisy   hand helix + noise on its columns, CKA (as trained) to the helix 0.625 = the teacher k8 target's
  ctl-digit-onehot  units one-hot + tens one-hot: a hand-made number code that is not a helix (CKA to helix 0.620)
--submit submits the jobs not yet submitted (runs/ptteach/SUBMITTED_<cond>_s<seed>)."""
import subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"; T = B / "teacher"
R = B / "runs" / "ptteach"
assert (R / "code" / "tools" / "pretrain_helix.py").exists()
conds = [("ctl-helix-noisy", f"--arm helix --geo_site ctx --target_file {T}/control_helix_noisy.npz --target_key feat"),
         ("ctl-digit-onehot", f"--arm helix --geo_site ctx --target_file {T}/control_digit_onehot.npz --target_key feat")]
for tag, extra in conds:
    for seed in range(4):
        job = R / f"job_{tag}_s{seed}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptctl-{tag}-s{seed}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# F77 control targets; code = runs/ptteach/code (COMMIT there)
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --out {R}/out/{tag} {extra} --seed {seed} --tag _{tag} --split unordered --arith_lines 4 --tokens 3e8 --eval_every 250 --ctx_eval --save_model"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py $ARGS
""")
        flag = R / f"SUBMITTED_{tag}_s{seed}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            print(job.name, o.stdout.strip() or o.stderr.strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
