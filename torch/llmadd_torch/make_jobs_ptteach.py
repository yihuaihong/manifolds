"""Torch jobs: automatic targets. The prior's target geometry is read out of a trained model instead of hand-designed
(tools/teacher_targets.py: states of "Q: x", x = 10..99, from the atlas, top-k PCs or all dims) and imposed on the
operands in the training lines (--geo_site ctx), same setup as ptu (unordered split, 3e8 tokens, seeds 0-3).
7% arithmetic (4 lines):
  t16k8 / t16full: OLMo-2-7B main, layer 16, top-8 PCs / all 4096 dims;
  t16k8-shuf / t16full-shuf: the same targets with the values permuted (farthest of 3000 draws; recorded CKA);
  t9Bk8: OLMo-2-7B at 9B tokens, layer 16, top-8 PCs (an early teacher, before its addition works).
2% arithmetic (1 line): 2pct-t16k8, 2pct-t16k8-shuf.
After the audit: none-t16k8 (target metrics for none), t16full-nohelix (helix span projected out of the teacher).
References from ptu (same args, training path unchanged): none, ctx-helix, ctx-shuf; 2pct-none, 2pct-ctx-helix.
--submit submits the jobs not yet submitted (runs/ptteach/SUBMITTED_<cond>_s<seed>)."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
T = B / "teacher"
R = B / "runs" / "ptteach"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
M, E = f"--target_file {T}/olmo2_7b_main_L16.npz", f"--target_file {T}/olmo2_7b_9B_L16.npz"
C = "--geo_site ctx"
conds = [("t16k8", f"--arm helix {C} {M} --target_key feat_k8", 4),
         ("t16full", f"--arm helix {C} {M} --target_key feat_full", 4),
         ("t16k8-shuf", f"--arm helix_shuf {C} {M} --target_key feat_k8", 4),
         ("t16full-shuf", f"--arm helix_shuf {C} {M} --target_key feat_full", 4),
         ("t9Bk8", f"--arm helix {C} {E} --target_key feat_k8", 4),
         ("2pct-t16k8", f"--arm helix {C} {M} --target_key feat_k8", 1),
         # added after the audit (runs/ptteach/AUDIT.md): none with the target metrics logged (training identical to
         # ptu none), the teacher with the helix span projected out, and the 2% shuffled-teacher control at ctx
         ("none-t16k8", f"--arm none {C} {M} --target_key feat_k8", 4),
         ("t16full-nohelix", f"--arm helix {C} --target_file {T}/olmo2_7b_main_L16_nohelix.npz --target_key feat_full_nohelix", 4),
         ("2pct-t16k8-shuf", f"--arm helix_shuf {C} {M} --target_key feat_k8", 1)]
for tag, extra, lines in conds:
    for seed in range(4):
        job = R / f"job_{tag}_s{seed}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptteach-{tag}-s{seed}
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# automatic (teacher) targets; code = llmadd-burst commit in code/COMMIT; intent spec runs/ptteach/AUDIT.md
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
sha256sum pretrain_helix.py
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --out {R}/out/{tag} {extra} --seed {seed} --tag _{tag} --split unordered --arith_lines {lines} --tokens 3e8 --eval_every 250 --ctx_eval --save_model"
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
