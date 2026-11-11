"""Torch jobs: operation-conditioned priors (tools/opcond_train.py) on OLMo-2-7B at 42B (F52: shared helix add +13 /
sub +19 / mul -12) and 84B; one job per (checkpoint, seed). --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
CK = {"42B": "stage1-step10000-tokens42B", "84B": "stage1-step20000-tokens84B"}
ARMS = ("none,prefix_helix,add_helix+sub_helix,add_helix+sub_helix+mul_log,add_helix+sub_helix+mul_factor,mul_log,mul_factor,"
        "add_helix_shuf+sub_helix_shuf+mul_log_shuf")
R = B / "runs" / "opcond"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
EXTRA = "mul_log_shuf,mul_factor_shuf,add_helix_shuf+sub_helix_shuf,add_helix+sub_helix+mul_log@norm"
KEY = "none,prefix_helix,add_helix+sub_helix,mul_log,mul_factor,add_helix+sub_helix+mul_log"
plan = [(f"{ck}-s{s}", ck, ARMS, s) for ck in CK for s in (0, 1, 2)]
plan += [(f"{ck}-extra-s{s}", ck, EXTRA, s) for ck in CK for s in (0, 1, 2)] + [(f"{ck}-s{s}", ck, KEY, s) for ck in CK for s in (3, 4)]
for tag, ck, arms, s in plan:
    if True:
        job = R / f"job_{tag}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_opcond-{tag}
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# operation-conditioned priors; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python opcond_train.py --model {B}/models/OLMo-2-1124-7B/{CK[ck]} --out {R}/out/{ck} --arms {arms} --seeds {s}
""")
        flag = R / f"SUBMITTED_{tag}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            print(job.name, o.stdout.strip() or o.stderr.strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
