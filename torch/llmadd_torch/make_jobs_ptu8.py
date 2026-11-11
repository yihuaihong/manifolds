"""Torch jobs: from-scratch pretraining with the UNORDERED held-out split (--split unordered: both orders of 500
pairs never trained -> uheld, ~985 truly unseen problems; uswap = 300 pairs with only the other order trained), so
speed and final accuracy measure generalisation, not recall of b + a (experiment-audit 2026-10-13, AUDIT.md).
7% arithmetic (4 lines), seeds 0-3:
  none; helix and helix_shuf (whole run, prefix site); helix windows (0,.4) (.05,.4) (.1,.4) (.2,.4) (.1,1.0);
  helix_shuf window (.1,.4) (wrong-geometry timing control: same aux-gradient shock at 10%);
  ctx site: helix, window (.1,.4), helix_shuf; ans site: helix, window (.1,.4), helix_shuf;
  helix_gate (thr .5, from step 0) on the prefix and on the ctx site.
2% arithmetic (1 line), seeds 0-3: none, helix, helix_shuf, window (.1,.4), ctx helix.
--submit submits the jobs not yet submitted (runs/ptu/SUBMITTED_<cond>_s<seed>)."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptu"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
assert (R / "code").exists()   # reuse the existing snapshot
if False:
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
W = "--arm helix_window --window"
conds7 = [("none", "--arm none"), ("helix", "--arm helix"), ("shuf", "--arm helix_shuf"),
          ("w0-0.4", f"{W} 0,0.4"), ("w0.05-0.4", f"{W} 0.05,0.4"), ("w0.1-0.4", f"{W} 0.1,0.4"), ("w0.2-0.4", f"{W} 0.2,0.4"),
          ("w0.1-1.0", f"{W} 0.1,1.0"), ("shufw0.1-0.4", "--arm helix_shuf_window --window 0.1,0.4"),
          ("ctx-helix", "--arm helix --geo_site ctx"), ("ctx-w0.1-0.4", f"{W} 0.1,0.4 --geo_site ctx"), ("ctx-shuf", "--arm helix_shuf --geo_site ctx"),
          ("ans-helix", "--arm helix --geo_site ans"), ("ans-w0.1-0.4", f"{W} 0.1,0.4 --geo_site ans"), ("ans-shuf", "--arm helix_shuf --geo_site ans"),
          ("gate-prefix", "--arm helix_gate --gate_thr 0.5 --gate_start 0"),
          ("gate-ctx", "--arm helix_gate --gate_thr 0.5 --gate_start 0 --geo_site ctx")]
conds2 = [("2pct-none", "--arm none"), ("2pct-helix", "--arm helix"), ("2pct-shuf", "--arm helix_shuf"),
          ("2pct-w0.1-0.4", f"{W} 0.1,0.4"), ("2pct-ctx-helix", "--arm helix --geo_site ctx")]
# 2026-10-13 (user-approved P1): seeds 4-7 for the headline arms of F71, on the public L40S partition so they run in
# parallel with the A100/H100 queue; same code snapshot (runs/ptu/code), same output folders, new seed numbers only.
SEEDS = range(4, 8)
KEEP7 = {"none", "ctx-helix", "gate-ctx", "w0.05-0.4", "shufw0.1-0.4", "ctx-shuf"}
KEEP2 = {"2pct-none", "2pct-ctx-helix", "2pct-w0.1-0.4", "2pct-shuf"}
jobs = [(c, e, 4) for c, e in conds7 if c in KEEP7] + [(c, e, 1) for c, e in conds2 if c in KEEP2]
assert len(jobs) == 10, [j[0] for j in jobs]
for tag, extra, lines in jobs:
    for seed in SEEDS:
        job = R / f"job_{tag}_s{seed}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptu-{tag}-s{seed}
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# unordered-split pretraining batch; code = llmadd-burst commit in code/COMMIT; intent spec runs/ptstart/AUDIT.md
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
print(len(jobs) * len(SEEDS), "jobs")
