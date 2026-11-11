"""Torch jobs: manifold atlas (tools/manifold_atlas.py) on every downloaded OLMo-2-7B checkpoint of the timeline.
--submit submits the checkpoints that are fully downloaded and not yet submitted (runs/atlas/SUBMITTED_<tag>)."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
M = B / "models/OLMo-2-1124-7B"
CK = {"rand": "stage1-step2000-tokens9B", "1B": "stage1-step150-tokens1B", "4B": "stage1-step900-tokens4B", "9B": "stage1-step2000-tokens9B",
      "21B": "stage1-step5000-tokens21B", "42B": "stage1-step10000-tokens42B", "84B": "stage1-step20000-tokens84B",
      "210B": "stage1-step50000-tokens210B", "630B": "stage1-step150000-tokens630B", "3896B": "stage1-step928646-tokens3896B",
      "main": "main"}
R = B / "runs" / "atlas"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
for tag, rev in CK.items():
    model = M / rev
    job = R / f"job_{tag}.sbatch"
    job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_atlas-{tag}
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# manifold atlas; code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib OMP_NUM_THREADS=16
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python manifold_atlas.py --model {model} --tag {tag} --out {R}/out{" --random_init" if tag == "rand" else ""}
""")
    ready = (model / ".complete").exists() or ((model / "config.json").exists() and len(list(model.glob("model-*.safetensors"))) == 6)
    flag = R / f"SUBMITTED_{tag}"
    if "--submit" in sys.argv and ready and not flag.exists():
        o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
        print(tag, o.stdout.strip() or o.stderr.strip())
        if o.returncode == 0:
            flag.write_text(o.stdout)
    elif "--submit" in sys.argv:
        print(tag, "submitted earlier" if flag.exists() else "waiting for the checkpoint")
