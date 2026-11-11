"""Torch jobs (2026-10-14, user: unify the main-text ablations on operands 0..999): WHERE and WHEN the prior acts, in
the unified setup of runs/pt999 (tools/pretrain_range.py, 53M, helix3, 7% arithmetic, 3e8 tokens), 4 seeds each:
  site at a fixed schedule: ctx window (0.05, 0.4); the prefix site ('Q: x' prompts at position 0, as in F69 / F71)
    whole / window (0.05, 0.4) / gate. pt999 has ctx whole / gate, text gate / window (0.05, 0.4), none, shuffles.
  onset of the text prior (F68 at 10..99; F78: a step-0 text prior costs +0.12..0.15 nats of val loss): windows
    (0, 0.4), (0.1, 0.4), (0.2, 0.4); (0.05, 0.4) is in pt999.
AUDIT ADDITIONS (2026-10-14, adversarial audit of this batch): the prefix prior's gradient is ~3.9x the LM's at init
(ctx 1.3x), so a slower prefix may be strength, not site -> strength-matched prefix window (lambda 5 * 4.82 / 14.43 = 1.7);
no shuffled control at the prefix site -> helix3_shuf prefix window. Both 4 seeds, models saved.
A separate batch because pt999 jobs were already running on their snapshot (fdfadfd) when --site prefix was added; the
ctx / text training paths are identical between the two snapshots (CPU check, scratchpad/test_prefix.sh). Analyse
with tools/pt999_summary.py runs/pt999/out runs/pt999b/out (tags are unique across the two).
Account torch_pr_625_general on the public L40S partition. --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "pt999b"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "tools"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    assert not dirty, f"tools/ has uncommitted changes, commit first:\n{dirty}"
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
G = "--schedule gate --gate_thr 0.5"
ARM = {"win-ctx": "--target helix3 --schedule window --window 0.05,0.4 --site ctx",
       "whole-prefix": "--target helix3 --schedule whole --site prefix",
       "win-prefix": "--target helix3 --schedule window --window 0.05,0.4 --site prefix",
       "gate-prefix": f"--target helix3 {G} --site prefix",
       "win0-text": "--target helix3 --schedule window --window 0,0.4 --site text",
       "win10-text": "--target helix3 --schedule window --window 0.1,0.4 --site text",
       "win20-text": "--target helix3 --schedule window --window 0.2,0.4 --site text",
       "win-prefix-shuf": "--target helix3_shuf --schedule window --window 0.05,0.4 --site prefix --save_model",
       "win-prefix-lam1.7": "--target helix3 --schedule window --window 0.05,0.4 --site prefix --lam 1.7 --save_model"}
n = 0
for a, extra in ARM.items():
    tag = f"7pct-{a}"
    for s in range(4):
        jt = f"{tag}-s{s}"
        job = R / f"job_{jt}.sbatch"
        R.mkdir(parents=True, exist_ok=True)
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_pt999b-{jt}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# unified 0..999 setup, site x timing; code = llmadd-burst commit in code/COMMIT; intent spec runs/pt999b/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/pt999b/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {extra} --arith_lines 4 --seed {s} --tag _{tag} --ckpt_every 2000"
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
