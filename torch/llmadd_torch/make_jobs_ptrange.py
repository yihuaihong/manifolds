"""Torch jobs: experiment B (tools/pretrain_range.py): operands 0..999, whole vs digit tokens x targets none / helix2 /
helix3 / digitcirc / digitcirc_shuf (in-context prior, whole run), seeds 0-3. --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
DATA = {"whole": B / "pretrain_data", "digits": B / "pretrain_data_digits"}
R = B / "runs" / "ptrange"
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
for tk in DATA:
    for tg in ("none", "helix2", "helix3", "digitcirc", "digitcirc_shuf", "digitcirc_lenshuf", "helix3_lenshuf", "length"):
        for s in range(4):
            tag = f"{tk}-{tg}-s{s}"
            job = R / f"job_{tag}.sbatch"
            job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptrange-{tag}
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# experiment B (number range 0..999); code = llmadd-burst commit in code/COMMIT
set -eu
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="--data {DATA[tk]} --tokenizer {TOK} --out {R}/out/{tk}-{tg} --tokenization {tk} --target {tg} --seed {s} --save_model"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS
""")
            flag = R / f"SUBMITTED_{tag}"
            if "--submit" in sys.argv and not flag.exists():
                o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
                print(job.name, o.stdout.strip() or o.stderr.strip())
                if o.returncode == 0:
                    flag.write_text(o.stdout)
