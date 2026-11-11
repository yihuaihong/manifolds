"""Torch jobs: raising the final / out-of-distribution performance of the helix prior in from-scratch pretraining
(7% arithmetic, in-context eval incl. the strict subset and the harder end-point tests, saved weights), seeds 0-3:
  baselines   none; helix_window (.1, .4) on the prefix site (as in pt7c, now with all eval sets)
  ctx site    helix, helix_window (.1, .4), helix_shuf      (geometry on the operands inside the training lines)
  ans site    helix, helix_window (.1, .4), helix_shuf      (geometry of the sum on the token that predicts the answer)
  self-timed  helix_gate (prior while first-line in-batch accuracy < .5, from step 0) on the prefix and on the ctx site
Intent spec: runs/ptstart/AUDIT.md. --submit submits the jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptfinal"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
conds = [("none", "--arm none"), ("prefix-window", "--arm helix_window --window 0.1,0.4"),
         ("ctx-helix", "--arm helix --geo_site ctx"), ("ctx-window", "--arm helix_window --window 0.1,0.4 --geo_site ctx"),
         ("ctx-shuf", "--arm helix_shuf --geo_site ctx"),
         ("ans-helix", "--arm helix --geo_site ans"), ("ans-window", "--arm helix_window --window 0.1,0.4 --geo_site ans"),
         ("ans-shuf", "--arm helix_shuf --geo_site ans"),
         ("gate-prefix", "--arm helix_gate --gate_thr 0.5 --gate_start 0"),
         ("gate-ctx", "--arm helix_gate --gate_thr 0.5 --gate_start 0 --geo_site ctx")]
for tag, extra in conds:
    for seed in range(4):
        job = R / f"job_{tag}_s{seed}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptfinal-{tag}-s{seed}
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# final-performance batch; code = llmadd-burst commit in code/COMMIT; intent spec in runs/ptstart/AUDIT.md
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --out {R}/out/{tag} {extra} --seed {seed} --arith_lines 4 --tokens 3e8 --eval_every 250 --ctx_eval --save_model"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py $ARGS
""")
        flag = R / f"SUBMITTED_{tag}_s{seed}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            print(job.name, o.stdout.strip() or o.stderr.strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
print(len(conds) * 4, "jobs")
