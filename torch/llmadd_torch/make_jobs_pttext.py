"""Torch jobs (P1, 2026-10-14): natural-text prior. The prior acts on two-digit numbers that occur naturally in the
FineWeb text of each training row (--geo_site text), never on the arithmetic lines, i.e. the version a real pretraining
run could use without synthetic data. Same setup as ptu (unordered split, 3e8 tokens), seeds 0-3, 7% (4 lines) and
2% (1 line) arithmetic. Arms: text (whole run), text window (0.05, 0.4), text + self-timed gate, text shuffled (whole).
References from ptu: none, ctx-helix, gate-ctx, ctx-shuf (7%); 2pct-none, 2pct-ctx-helix (2%); plus 2pct-gate-ctx here.
Account <slurm-account> on the public L40S partition (user-approved 2026-10-14).
--submit submits the jobs not yet submitted (runs/pttext/SUBMITTED_<cond>_s<seed>)."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "pttext"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
T = "--geo_site text"
arms = [("text", f"--arm helix {T}"), ("text-w0.05-0.4", f"--arm helix_window --window 0.05,0.4 {T}"),
        ("text-gate", f"--arm helix_gate --gate_thr 0.5 --gate_start 0 {T}"), ("text-shuf", f"--arm helix_shuf {T}"),
        # audit 2026-10-14: none with the text-site CKA logged (baseline for cka_text; training identical to none)
        ("text-none", f"--arm none {T}")]
jobs = [(c, e, 4) for c, e in arms] + [("2pct-" + c, e, 1) for c, e in arms] + \
       [("2pct-gate-ctx", "--arm helix_gate --gate_thr 0.5 --gate_start 0 --geo_site ctx", 1)]
for tag, extra, lines in jobs:
    for seed in range(8):   # audit 2026-10-14: 8 seeds (4 cannot separate shuffled / 2% arms from none)
        job = R / f"job_{tag}_s{seed}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_pttext-{tag}-s{seed}
#SBATCH --account=<slurm-account>
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# natural-text prior; code = llmadd-burst commit in code/COMMIT; intent spec runs/pttext/AUDIT.md
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
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
print(len(jobs) * 8, "jobs")
