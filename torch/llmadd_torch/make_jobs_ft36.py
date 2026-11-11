"""Torch jobs (2026-10-14, user: "test all 36 manifolds with priors"): tools/ft36.py, the fine-tuning prior screen over
every atlas domain. OLMo-2-7B at 42B tokens (the checkpoint with the clearest number-helix fine-tuning gain, F10-F14)
and 210B (days / months gains, F40); per domain none / prior / prior_shuf x seeds 0-2, LoRA r 8, lambda 20, layers
4,8,12,16, 280 steps; splits without equivalent leakage (cyclic: (item, k mod n) classes). 6 jobs per checkpoint,
6 domains each (the two costliest, numbers and years, in different jobs). Intent spec runs/ft36/AUDIT.md.
Account torch_pr_625_general on the public L40S partition. --submit submits jobs not yet submitted."""
import shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
R = B / "runs" / "ft36"
CK = {"42B": "stage1-step10000-tokens42B", "210B": "stage1-step50000-tokens210B"}
DOMAINS = ("numbers number_words ordinals years days months hours letters greek planets elements presidents pitch keys "
           "colours stars cities emotions countries country_size landmarks us_cities people geo_periods animals "
           "animal_size languages kinship scalar_adj ranks life_stages zodiac chinese_zodiac compass units prefixes").split()
assert len(DOMAINS) == 36
GROUPS = [DOMAINS[i::6] for i in range(6)]   # round robin: numbers (0) and years (3) land in different groups
assert not any({"numbers", "years"} <= set(g) for g in GROUPS)
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "tools"], capture_output=True, text=True).stdout.strip()
if not (R / "code").exists():   # audit 2026-10-14: the snapshot must be a commit, not a dirty working tree
    assert not dirty, f"tools/ has uncommitted changes, commit first:\n{dirty}"
    shutil.copytree(REPO / "tools", R / "code" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (R / "code" / "COMMIT").write_text(commit + "\n")
n = 0
for ck, rev in CK.items():
    for gi, group in enumerate(GROUPS):
        tag = f"{ck}-g{gi}"
        job = R / f"job_{tag}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ft36-{tag}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# 36-domain fine-tuning prior screen; code = llmadd-burst commit in code/COMMIT; intent spec runs/ft36/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/ft36/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
# finished runs are skipped (readable run_<arm>_seed<s>.json), so a requeued job continues where it stopped
/scratch/yh6210/miniconda3/envs/open_r1/bin/python ft36.py --model {B}/models/OLMo-2-1124-7B/{rev} --out {R}/out/{ck} --domains {",".join(group)} --seeds 0,1,2
""")
        n += 1
        flag = R / f"SUBMITTED_{tag}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text(o.stdout)
            else:
                print(tag, o.stderr.strip())
print(n, "jobs;", [len(g) for g in GROUPS])
