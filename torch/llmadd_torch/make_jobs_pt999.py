"""Torch jobs (2026-10-14, user decisions: unify on operands 0..999 with the helix3 target; sensitivity + layer sweep):
the unified pretraining setup = tools/pretrain_range.py (53M, FineWeb-Edu + arithmetic lines with operands 0..999,
per-bucket unordered held-out pairs, 3e8 tokens, whole-number tokens).
MAIN (8 seeds): 7% (4 lines): none, gate-ctx, gate-text, shuf-ctx (helix3_shuf, gate, ctx), shuf-text;
                2% (1 line): none, gate-ctx, gate-text.
SENSITIVITY (4 seeds, 7%, one factor at a time from the base lr 1e-3 / bs 32 / lambda 5 / layers 2,4):
  lr 5e-4 and 2e-3, batch 64 (same tokens): none, gate-ctx, gate-text; lambda 1 and 20: gate-ctx, gate-text;
  LAYERS (appendix): {1}, {4,5}, {1..6}: gate-ctx, gate-text.
AUDIT ADDITIONS (2026-10-14, adversarial audit of this batch, runs/pt999/AUDIT.md): positive control 7pct whole-run
helix3 on the operands (8 seeds; F75 ran on A100/H100); helix3_lenshuf gate on the operands (4 seeds: keeps the
digit-count grouping, destroys the within-length geometry); untied input / output embeddings, none and gate-ctx (4 seeds:
with tied embeddings the prior also moves the answer classifier rows of the number tokens).
TEXT WINDOW (2026-10-14, pttext results audit at 10..99: a text prior switched on at step 0 -- whole run or gate --
costs +0.12..0.15 nats of final val loss, the window (0.05, 0.4) only +0.01..0.03 with the same speed-up): 7% and 2%
helix3 text prior in the window (0.05, 0.4), 8 seeds each.
Account torch_pr_625_general on the public L40S partition. --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "pt999"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
G = "--schedule gate --gate_thr 0.5"
ARM = {"none": "--target none", "gate-ctx": f"--target helix3 {G} --site ctx", "gate-text": f"--target helix3 {G} --site text",
       "shuf-ctx": f"--target helix3_shuf {G} --site ctx", "shuf-text": f"--target helix3_shuf {G} --site text",
       "whole-ctx": "--target helix3 --schedule whole --site ctx", "lenshuf-ctx": f"--target helix3_lenshuf {G} --site ctx",
       "untie-none": "--target none --untie", "untie-gate-ctx": f"--target helix3 {G} --site ctx --untie",
       "win-text": "--target helix3 --schedule window --window 0.05,0.4 --site text"}
jobs = []   # (tag, args, seeds)
for a in ("none", "gate-ctx", "gate-text", "shuf-ctx", "shuf-text"):
    jobs.append((f"7pct-{a}", ARM[a] + " --arith_lines 4", range(8)))
for a in ("none", "gate-ctx", "gate-text"):
    jobs.append((f"2pct-{a}", ARM[a] + " --arith_lines 1", range(8)))
for name, extra, arms in (("lr5e-4", "--lr 5e-4", ("none", "gate-ctx", "gate-text")), ("lr2e-3", "--lr 2e-3", ("none", "gate-ctx", "gate-text")),
                          ("bs64", "--bs 64", ("none", "gate-ctx", "gate-text")), ("lam1", "--lam 1", ("gate-ctx", "gate-text")),
                          ("lam20", "--lam 20", ("gate-ctx", "gate-text")), ("L1", "--geo_layers 1", ("gate-ctx", "gate-text")),
                          ("L45", "--geo_layers 4,5", ("gate-ctx", "gate-text")), ("Lall", "--geo_layers 1,2,3,4,5,6", ("gate-ctx", "gate-text"))):
    for a in arms:
        jobs.append((f"sens-{name}-{a}", ARM[a] + f" --arith_lines 4 {extra}", range(4)))
jobs.append(("7pct-whole-ctx", ARM["whole-ctx"] + " --arith_lines 4", range(8)))
for lines, share in ((4, "7pct"), (1, "2pct")):
    jobs.append((f"{share}-win-text", ARM["win-text"] + f" --arith_lines {lines}", range(8)))
# 2026-10-16 (next-phase T2; rule in runs/ptlr/AUDIT.md): 2% window on the operands, paired by seed with 2pct-none / 2pct-gate-ctx
jobs.append(("2pct-win-ctx", "--target helix3 --schedule window --window 0.05,0.4 --site ctx --arith_lines 1", range(8)))
# 2026-10-17 (F82 results audit, M4): the 2% main setting has no shuffled-geometry control; the same windows with the
# helix3_shuf target at both sites (8 seeds, paired by seed with 2pct-none / 2pct-win-text / 2pct-win-ctx)
for site in ("text", "ctx"):
    jobs.append((f"2pct-win-shuf-{site}", f"--target helix3_shuf --schedule window --window 0.05,0.4 --site {site} --arith_lines 1", range(8)))
for a in ("lenshuf-ctx", "untie-none", "untie-gate-ctx"):
    jobs.append((f"7pct-{a}", ARM[a] + " --arith_lines 4", range(4)))
n = 0
for tag, extra, seeds in jobs:
    for s in seeds:
        jt = f"{tag}-s{s}"
        job = R / f"job_{jt}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_pt999-{jt}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# unified 0..999 setup; code = llmadd-burst commit in code/COMMIT; intent spec runs/pt999/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/pt999/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {extra} --seed {s} --tag _{tag} {'--save_model ' if tag.startswith(('7pct', '2pct')) else ''}--ckpt_every 2000"
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
