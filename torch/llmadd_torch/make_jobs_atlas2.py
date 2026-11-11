"""Torch jobs: manifold atlas v2 (36 domains: the 18 of v1 + 18 general-knowledge / tokenizer-independent ones,
tools/manifold_domains.py) on
  - the OLMo-2-7B pretraining timeline (same 11 checkpoints as v1, incl. the random-init control),
  - the OLMo-2-1B timeline (step 0 = its real initialisation, 1B .. 4T tokens, main),
  - the Pythia-6.9B timeline (4B .. 300B tokens; different data: the Pile),
  - final checkpoints of other families: SmolLM2-1.7B, Qwen2.5-7B, Qwen3-8B-Base, OLMo-3-7B, Mistral-7B.
--submit submits the models that are fully downloaded and not yet submitted (runs/atlas2/SUBMITTED_<tag>).
--l40s: the same jobs on the public L40S partition (48 GB is enough for every model here), so the atlas runs in parallel
with the pretraining jobs that fill the Tandon A100/H100 quota (job_l40s_<tag>.sbatch, SUBMITTED_l40s_<tag>)."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
M7 = B / "models/OLMo-2-1124-7B"; X = B / "models/xmodel"
CK = {"rand": "stage1-step2000-tokens9B", "1B": "stage1-step150-tokens1B", "4B": "stage1-step900-tokens4B", "9B": "stage1-step2000-tokens9B",
      "21B": "stage1-step5000-tokens21B", "42B": "stage1-step10000-tokens42B", "84B": "stage1-step20000-tokens84B",
      "210B": "stage1-step50000-tokens210B", "630B": "stage1-step150000-tokens630B", "3896B": "stage1-step928646-tokens3896B",
      "main": "main"}
MODELS = {f"olmo2-7b-{t}": M7 / r for t, r in CK.items()}
MODELS.update({d.name: d for d in sorted(X.iterdir()) if d.is_dir() and d.name not in ("pythia-160m", "Qwen2.5-0.5B")} if X.exists() else {})
R = B / "runs" / "atlas2"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
L40S = "--l40s" in sys.argv
PART = "l40s_public" if L40S else "a100_tandon,h100_tandon"
PFX = "l40s_" if L40S else ""
for tag, model in MODELS.items():
    job = R / f"job_{PFX}{tag}.sbatch"
    job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_atlas2-{tag}
#SBATCH --account=<slurm-account>
#SBATCH --partition={PART}
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# manifold atlas v2; code = llmadd-burst commit in code/COMMIT; intent spec runs/atlas2/AUDIT.md
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib OMP_NUM_THREADS=16
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python manifold_atlas.py --model {model} --tag {tag} --out {R}/out{" --random_init" if tag == "olmo2-7b-rand" else ""}
""")
    ready = (model / ".complete").exists() or ((model / "config.json").exists() and len(list(model.glob("model-*.safetensors"))) == 6)
    flag = R / f"SUBMITTED_{PFX}{tag}"
    if "--submit" in sys.argv and ready and not flag.exists():
        o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
        print(tag, o.stdout.strip() or o.stderr.strip())
        if o.returncode == 0:
            flag.write_text(o.stdout)
    elif "--submit" in sys.argv and not flag.exists():
        print(tag, "waiting for the download")
print(len(MODELS), "models")
