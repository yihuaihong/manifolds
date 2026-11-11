"""Torch jobs (2026-10-14, user-approved): the day / month fine-tuning experiments (F24 / F70) re-run with an
EQUIVALENCE-CLASS held-out split (concept_train.py --split class): the original 60/40 combo split left an equivalent
trained combo (same item, same shift mod n) for 100% of the test problems. OLMo-2-7B checkpoints where the circle prior
helped in F24 (days 84B / 210B / 630B, months 42B / 84B / 210B); geometries none / circle / circle_shuf / line (the
domain's defaults), LoRA as in F24 (lambda 20, layers 4,8,12,16, 280 steps), seeds 0-2. 625 account, public L40S.
--submit submits the jobs not yet submitted (runs/ftstrict/SUBMITTED_<tag>)."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
M = B / "models/OLMo-2-1124-7B"
CK = {"42B": "stage1-step10000-tokens42B", "84B": "stage1-step20000-tokens84B", "210B": "stage1-step50000-tokens210B",
      "630B": "stage1-step150000-tokens630B"}
R = B / "runs" / "ftstrict"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
JOBS = [("days", c) for c in ("84B", "210B", "630B")] + [("months", c) for c in ("42B", "84B", "210B")]
for dom, ck in JOBS:
    tag = f"{dom}-{ck}"
    job = R / f"job_{tag}.sbatch"
    job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ftstrict-{tag}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# equivalence-class split re-run of the day / month fine-tuning; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python concept_train.py --model {M / CK[ck]} --out {R}/out/{tag} --domain {dom} --split class --seeds 0,1,2
""")
    flag = R / f"SUBMITTED_{tag}"
    if "--submit" in sys.argv and not flag.exists():
        o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
        print(job.name, o.stdout.strip() or o.stderr.strip())
        if o.returncode == 0:
            flag.write_text(o.stdout)
print(len(JOBS), "jobs")
