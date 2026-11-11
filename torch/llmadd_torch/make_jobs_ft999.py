"""Torch jobs (2026-10-22, LLM_addition Figure 1 bottom row; user: "可以的，没问题，请你直接开始跑就行"): ft999 = LoRA
fine-tuning of OLMo-2-7B on three-digit addition (operands 0..999, sums 0..1998) with the helix3 prior over 0..999
(manifold_train.py --add3). PRIMARY 9B (stage1-step2000-tokens9B, as F64 / viz), SECONDARY 42B (stage1-step10000-tokens42B);
arms none / helix3 / helix3_shuf x seeds 0-2 = 18 one-run jobs. F64 setting: 7000 train pairs, 10 epochs, bs 16, lambda 20,
layers 4,8,12,16 (A slot of "Q: x"), curves every 50 steps; 2000 held-out pairs (unordered-disjoint). Layer-16 states of
0..999 every 50 steps for seed 0 of each arm at 9B. Intent spec runs/ft999/AUDIT.md.
Code = git archive of tools/ at the committed ft999 HEAD (llmadd-burst-ft999 worktree), sha of manifold_train.py asserted
in every job. A100 only (guarded). --test-only runs sbatch --test-only; --submit submits jobs not yet submitted."""
import hashlib, subprocess, sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/scratch/yh6210/llmadd-burst-ft999")
R = B / "runs" / "ft999"
CK = {"9B": "stage1-step2000-tokens9B", "42B": "stage1-step10000-tokens42B"}
ARMS = ("none", "helix3", "helix3_shuf")
SEEDS = (0, 1, 2)
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "tools"], capture_output=True, text=True, check=True).stdout.strip()
code = R / "code"
if not code.exists():   # the snapshot is a commit, never a dirty working tree
    assert not dirty, f"tools/ has uncommitted changes, commit first:\n{dirty}"
    code.mkdir(parents=True)
    arc = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", "tools"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(code)], input=arc, check=True)
    (code / "COMMIT").write_text(commit + "\n")
snap_commit = (code / "COMMIT").read_text().strip()
assert snap_commit == commit, f"snapshot is {snap_commit}, HEAD is {commit}: refresh the snapshot only while no job has started"
sha = hashlib.sha256((code / "tools" / "manifold_train.py").read_bytes()).hexdigest()[:12]
head_sha = hashlib.sha256(subprocess.run(["git", "-C", str(REPO), "show", "HEAD:tools/manifold_train.py"], capture_output=True, check=True).stdout).hexdigest()[:12]
assert sha == head_sha, (sha, head_sha)
(R / "out").mkdir(exist_ok=True)
jobs = []
for ck, rev in CK.items():
    for arm in ARMS:
        for s in SEEDS:
            tag = f"{ck}-{arm}-s{s}"
            states = "--save_states_every 50" if (ck == "9B" and s == 0) else ""
            job = R / f"job_{tag}.sbatch"
            job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ft999-{tag}
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --nice=20
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# ft999 (Figure 1 bottom row): LoRA on three-digit addition 0..999 + helix3 prior over 0..999; intent spec runs/ft999/AUDIT.md
# code = runs/ft999/code (git archive of tools/ at commit {commit}); a requeued job restarts its single run from scratch
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/ft999/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
[ "$(cat ../COMMIT)" = "{commit}" ] || {{ echo "COMMIT mismatch: exiting"; exit 3; }}
[ "$(sha256sum manifold_train.py | cut -c1-12)" = "{sha}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
MODEL={B}/models/OLMo-2-1124-7B/{rev}
OUT={R}/out/{ck}/{arm}-s{s}
EXTRA=""
if [ -z "${{SLURM_JOB_ID:-}}" ] && [ "${{FT999_SMOKE:-0}}" = 1 ]; then
  # local CPU smoke of this exact command (never under slurm): tiny random OLMo-2, a few steps, scratch output
  MODEL=$FT999_SMOKE_MODEL; OUT=$FT999_SMOKE_OUT/{tag}
  EXTRA="--train_pairs 64 --test_pairs 200 --epochs 1 --curve_every 2 --save_states_every 2 --grad_check_every 2"
else
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU: $GPU"
  case "$GPU" in *A100*) ;; *) echo "not an A100: exiting"; exit 4;; esac
fi
echo "ft999 {tag} commit {commit} code_sha {sha} model $MODEL out $OUT"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python manifold_train.py --model $MODEL --out $OUT --mode lora --lam 20 --save_preds --fixed_test --add3 --train_pairs 7000 --test_pairs 2000 --epochs 10 --curve_every 50 --grad_check_every 250 {states} --geoms {arm} --seeds {s} $EXTRA
""")
            jobs.append((tag, job))
mode = "--submit" if "--submit" in sys.argv else ("--test-only" if "--test-only" in sys.argv else None)
for tag, job in jobs:
    flag = R / f"SUBMITTED_{tag}"
    if mode == "--test-only":
        o = subprocess.run(["sbatch", "--test-only", str(job)], capture_output=True, text=True)
        print(tag, o.returncode, (o.stdout + o.stderr).strip().splitlines()[-1] if (o.stdout + o.stderr).strip() else "")
    elif mode == "--submit" and not flag.exists():
        o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
        if o.returncode == 0:
            flag.write_text(o.stdout); print(tag, o.stdout.strip())
        else:
            print(tag, "FAILED", o.stderr.strip())
print(len(jobs), "jobs; commit", commit, "code_sha", sha)
