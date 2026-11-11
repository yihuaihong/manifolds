"""Torch jobs: rainbow fine-tuning (tools/rainbow_train.py) on OLMo-2-7B at 84B (all three single priors help there:
F14 add +10, F24 days +11, months +7.5) and 210B (add window closed, helix hurts; days / months still benefit).
Arms: none, each single prior, all three, all-but-one (x3), all three shuffled; one job per (checkpoint, seed);
plus rank 32 for none and all three (capacity check). --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
CK = {"84B": "stage1-step20000-tokens84B", "210B": "stage1-step50000-tokens210B"}
ARMS = "none,helix,days,months,helix+days+months,days+months,helix+months,helix+days,helix_shuf+days_shuf+months_shuf"
R = B / "runs" / "rainbow"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
KEY = "none,helix,days,months,helix+days+months,helix+days+months@norm,days+months"
jobs = [(f"{ck}-s{s}", ck, ARMS, s, 8) for ck in CK for s in (0, 1, 2)] + [(f"{ck}-r32", ck, "none,helix+days+months", "0,1,2", 32) for ck in CK]
# audit (2026-10-13): lambda normalised by the number of priors, and seeds 3-4 of the key arms (power)
jobs += [(f"{ck}-norm", ck, "helix+days+months@norm", "0,1,2", 8) for ck in CK] + [(f"{ck}-s{s}", ck, KEY, s, 8) for ck in CK for s in (3, 4)]
for tag, ck, arms, seeds, rank in jobs:
    job = R / f"job_{tag}.sbatch"
    out = R / "out" / (ck if rank == 8 else f"{ck}-r32")
    job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_rainbow-{tag}
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# rainbow fine-tuning; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python rainbow_train.py --model {B}/models/OLMo-2-1124-7B/{CK[ck]} --out {out} --arms {arms} --seeds {seeds} --rank {rank}
""")
    flag = R / f"SUBMITTED_{tag}"
    if "--submit" in sys.argv and not flag.exists():
        o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
        print(job.name, o.stdout.strip() or o.stderr.strip())
        if o.returncode == 0:
            flag.write_text(o.stdout)
