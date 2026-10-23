"""Generate (and with --submit, submit) the sbatch jobs of the downstream evaluation (tools/downstream_eval.py).
Intent spec / pre-registration: llmadd_torch/runs/downstream/AUDIT.md. One job per model group; a job skips models that are
already evaluated with the same eval code (or not finished yet), so re-running a group later only adds the new models:

    python tools/make_downstream_jobs.py --groups rb6_505m --submit      # e.g. when runs/rb6_505m has finished
    # if downstream_eval.py changed since the known-answer files were written, first re-run the KA job and chain:
    python tools/make_downstream_jobs.py --groups ka --submit; python tools/make_downstream_jobs.py --groups rb6_505m --submit --after <ka job id>

Every group job runs the read-out selftest and the known-answer gate (all KA files of the current eval code, every
load-check model) before scoring anything. The code is snapshotted into runs/downstream/code_<commit>/ (sha256-pinned in
every sbatch; the first batch of 2026-10-22 runs from runs/downstream/code = commit 2dc8375); runs/downstream/HOLD makes
a pending job exit at start (no scancel needed). Comparisons for new groups must be pre-registered (AUDIT.md amendment)
before their scores are read.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/scratch/yh6210/llmadd_torch")
R = ROOT / "runs/downstream"
RUNS = ROOT / "runs"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
FILES = ("downstream_eval.py", "pretrain_helix.py", "numlib.py", "downstream_readout.py")
KA_ARCH = {"53m": RUNS / "pt999/out/7pct-none/run_whole_none_seed0_7pct-none.json",
           "190m": RUNS / "rb6_190m/out/rb6tb-190m-2pct-k2-none-lr6e-4-s0/rb6tb-190m-2pct-k2-none-lr6e-4/"
                          "run_whole_none_seed0_rb6tb-190m-2pct-k2-none-lr6e-4.json",
           "505m": RUNS / "ptlr/out/500m-lr1.2e-3-7pct-none/run_whole_none_seed0_500m-lr1.2e-3-7pct-none.json"}
# one model per (family, training-code version): ptlr a3bce3ab63c1, ptsize999 / pt999 79e9b410dcc3, rb6_190m 93e22bf4c4c0
LOADCHECK = [RUNS / "ptlr/out/500m-lr1.2e-3-7pct-none/model_whole_none_seed0_500m-lr1.2e-3-7pct-none.pt",
             RUNS / "ptlr/out/53m-lr4e-3-7pct-none/model_whole_none_seed0_53m-lr4e-3-7pct-none.pt",
             RUNS / "ptsize999/out/500m-7pct-none/model_whole_none_seed0_500m-7pct-none.pt",
             RUNS / "pt999/out/7pct-none/model_whole_none_seed0_7pct-none.pt",
             RUNS / "rb6_190m/out/rb6tb-190m-2pct-k2-none-lr6e-4-s0/rb6tb-190m-2pct-k2-none-lr6e-4/"
                    "model_whole_none_seed0_rb6tb-190m-2pct-k2-none-lr6e-4.pt"]
# group -> (model globs, time limit)
GROUPS = {
    "p505a": ([f"{RUNS}/ptlr/out/500m-*/model_*.pt"], "16:00:00"),
    "p505b": ([f"{RUNS}/ptsize999/out/500m-*/model_*.pt"], "12:00:00"),
    "r190": ([f"{RUNS}/rb6_190m/out/*/*/model_*.pt"], "08:00:00"),
    "desc": ([f"{RUNS}/ptlr/out/53m-*/model_*.pt", f"{RUNS}/ptlr/out/190m-*/model_*.pt", f"{RUNS}/pt999/out/7pct-*/model_*.pt",
              f"{RUNS}/ptsize999/out/190m-7pct-*/model_*.pt"], "08:00:00"),
    # later (not submitted now): the same command when these runs finish
    "rb6_505m": ([f"{RUNS}/rb6_505m/out/*/*/model_*.pt"], "24:00:00"),
    "pt2s505": ([f"{RUNS}/pt2s505/out/*/*/model_*.pt"], "24:00:00"),
    "rb6_190m_ext": ([f"{RUNS}/rb6_190m_ext/out/*/*/model_*.pt"], "12:00:00"),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True, help="comma list of: ka," + ",".join(GROUPS))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--after", default=None, help="job id: submit with --dependency=afterok:<id> (e.g. the known-answer job)")
    a = ap.parse_args()
    tools = Path(__file__).resolve().parent
    commit = subprocess.run(["git", "-C", str(tools), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(tools), "status", "--porcelain", "--", *FILES], capture_output=True, text=True).stdout.strip()
    assert not dirty, f"commit the eval code first: {dirty}"   # before touching the snapshot (pending jobs pin it)
    # A2: one snapshot per commit (code_<commit>): a later batch never rewrites the code a pending job pins (the 2026-10-22
    # batch 20678044 / 71 / 76 / 80 / 84 runs from runs/downstream/code = commit 2dc8375)
    from downstream_readout import LOADCHECK_NAMES
    assert tuple(f"{p.parent.name}_s0" for p in LOADCHECK) == LOADCHECK_NAMES, "LOADCHECK != the gate's LOADCHECK_NAMES"
    code = R / f"code_{commit}"; code.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy(tools / f, code / f)
    (code / "COMMIT").write_text(f"git {commit} (branch downstream)\n" + "".join(f"{f} {sha(code / f)}\n" for f in FILES))
    (R / "jobs").mkdir(exist_ok=True); (R / "logs").mkdir(exist_ok=True); (R / "out").mkdir(exist_ok=True)
    pins = "\n".join(f'[ "$(sha256sum {f} | cut -c1-12)" = "{sha(code / f)}" ] || {{ echo "## {f} sha mismatch: exiting"; exit 3; }}'
                     for f in FILES)
    for g in a.groups.split(","):
        if g == "ka":
            body = "\n".join([
                f'"$PY" downstream_eval.py --out_dir "$R/out" --hf gpt2 --prefix harness --max_len 1024',
                f'"$PY" downstream_eval.py --out_dir "$R/out" --hf gpt2 --prefix eos --max_len 1024',
                f'"$PY" downstream_eval.py --out_dir "$R/out" --random_init {" ".join(str(p) for p in KA_ARCH.values())}',
                '# loading path (Amendment A1): in-training bpb replica on one model per training-code version, NO benchmark score',
                f'"$PY" downstream_eval.py --out_dir "$R/out" --load_check_only {" ".join(str(p) for p in LOADCHECK)}',
                '"$PY" downstream_readout.py --out_dir "$R/out" --ka_gate   # exit 1 on a failed check (afterok dependents never run)'])
            tlim = "04:00:00"
        else:
            globs, tlim = GROUPS[g]
            body = "\n".join([
                '# pre-registered known-answer gate (AUDIT.md): no project model is scored unless every check passed',
                '"$PY" downstream_readout.py --selftest > /dev/null || { echo "## read-out selftest FAILED: exiting"; exit 5; }',
                '"$PY" downstream_readout.py --out_dir "$R/out" --ka_gate || { echo "## known-answer gate FAILED: exiting"; exit 5; }',
                f'"$PY" downstream_eval.py --out_dir "$R/out" ' + " ".join(f"'{x}'" for x in globs),
                '# refresh the pre-registered read-out with everything evaluated so far (the last job to finish writes the full one)',
                '"$PY" downstream_readout.py --out_dir "$R/out" --write "$R/readout" > /dev/null'])
        job = R / "jobs" / f"job_{g}_{commit}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_down_{g}
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --nice=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={tlim}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={R}/logs/%x_%j.out
# downstream eval, group {g}; generated by tools/make_downstream_jobs.py (change the generator, not this file);
# intent spec / pre-registration {R}/AUDIT.md; code = {code} (COMMIT); models are read-only inputs.
set -euo pipefail
R={R}; PY={PY}
if [ -e "$R/HOLD" ]; then echo "HOLD present: exiting"; exit 0; fi
cd "{code}"
{pins}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1); echo "GPU: $GPU"
case "$GPU" in *A100*) ;; *) echo "## not an A100: exiting"; exit 3;; esac
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 >> "$R/logs/gpu_{g}_${{SLURM_JOB_ID:-x}}.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
echo "## start $(date -u +%FT%TZ) $(hostname) job ${{SLURM_JOB_ID:-}} restart ${{SLURM_RESTART_COUNT:-0}}"
{body}
echo "## done $(date -u +%FT%TZ)"
""")
        print("wrote", job)
        if a.submit:
            dep = [f"--dependency=afterok:{a.after}", "--kill-on-invalid-dep=yes"] if a.after else []
            out = subprocess.run(["sbatch", *dep, str(job)], capture_output=True, text=True)
            print(out.stdout.strip(), out.stderr.strip())


if __name__ == "__main__":
    main()
