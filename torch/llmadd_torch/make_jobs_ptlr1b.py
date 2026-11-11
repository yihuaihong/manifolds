"""Torch B200 jobs (2026-10-23, ladder-1b; user: 「我想用上这个 B200，没问题，我们暂时用这个 setup，先排着」; coordinator's go
2026-10-23): a 1B point on F87's 7% single-prior size ladder (runs/ptlr: 53M / 190M / 505M on L40S). Everything as F87's
ptlr cells -- the same trainer snapshot (llmadd-burst commit 64abc19, tools/pretrain_range.py code_sha a3bce3ab63c1; the
extracted tools/ are asserted equal to runs/ptlr/code/tools file by file), data, split, evals, 3e8 tokens = 18,310 steps of
32 x 512, 1% warmup, AdamW wd 0.1, lambda 5, the window (0.05, 0.4) at the operands (ctx) -- except: the model size, the lr,
the GPU type (B200, preemptible: a preempted job is requeued with its id and resumes from its ckpt), --ckpt_every 500
(F87: 2000; resume granularity only, the B200 smoke showed the resumed segment identical at its first eval), and
--save_model only at the picks (seeds 2-7; stage A = seeds 0-1 has none: the read-out uses the in-training bpb).
  1B = the GPT-2-large shape after 190M (GPT-2 small shape) and 505M (GPT-2 medium shape): d 1280, 36 layers, 20 heads of
       64, d_ff 5120 (4d); 1,072,259,840 parameters with the tied 100,278-row embedding, N_nonemb 943,904,000.
Intent spec and pre-registered rules: runs/ptlr_1b/AUDIT.md. Stages: A = none {2.4e-3, 4.8e-3, 9.6e-3} and win-ctx {3e-4,
6e-4, 1.2e-3}, seeds 0-1 (12 runs); B = seeds 2-3 at each arm's pick (picks.json, written by maybe_select.sh after stage
A); C = confirm seeds 4-7 at the picks, only when runs/ptlr_1b/APPROVE_C exists (the coordinator's approval).
Every run is queued under two accounts (torch_pr_520_tandon_priority: b200_tandon_prem or b200; torch_pr_625_general:
b200); tools/run_guard.py lets the first copy compute it and the other exit (F87 practice). At most 4 runs at a time: job
names llmadd_ptlr1b-slot<k> with --dependency=singleton (k = run index mod 4; both copies of a run share its slot).
    python make_jobs_ptlr1b.py [--test-only | --submit]"""
import filecmp
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptlr_1b"
F87 = B / "runs" / "ptlr" / "code"   # F87's own snapshot: the extracted tools/ must equal it file by file
PIN, PIN_SHA = "64abc19", "a3bce3ab63c1"
ACCOUNTS = ("torch_pr_520_tandon_priority", "torch_pr_625_general")   # submitted in this order
HOURS = "05:00:00"    # one segment (a requeued job restarts the clock); a run is ~3.0 B200-h (smoke: 0.55 s/step)
SLOTS = 4
CKPT_EVERY = "500"
ARM = {"none": "--target none", "win-ctx": "--target helix3 --schedule window --window 0.05,0.4 --site ctx"}
MODEL = "--d_model 1280 --n_layers 36 --n_heads 20 --d_ff 5120"
GRID = {"none": ("2.4e-3", "4.8e-3", "9.6e-3"), "win-ctx": ("3e-4", "6e-4", "1.2e-3")}   # AUDIT.md: why these
WANT = {"none": {"--target": "none"},
        "win-ctx": {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx"}}
WANT_SIZE = {"--d_model": "1280", "--n_layers": "36", "--n_heads": "20", "--d_ff": "5120"}
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"


def cells():
    """[(lr, arm, seed)] in a fixed order: stage A, then B / C from picks.json (appended, so slots stay stable)."""
    out = [(lr, a, s) for a in ("none", "win-ctx") for lr in GRID[a] for s in range(2)]
    pk = R / "picks.json"
    if pk.exists():
        p = json.loads(pk.read_text())
        assert p["status"] == "final" and p["none"] in GRID["none"] and p["win-ctx"] in GRID["win-ctx"], p
        out += [(p[a], a, s) for s in (2, 3) for a in ("none", "win-ctx")]
        if (R / "APPROVE_C").exists():
            out += [(p[a], a, s) for s in range(4, 8) for a in ("none", "win-ctx")]
    return out


def flags(args):
    """{--flag: value or True} of an ARGS string; asserts that no flag is given twice."""
    t, d = args.split(), {}
    for i, w in enumerate(t):
        if w.startswith("--"):
            assert w not in d, f"{w} given twice: {args}"
            d[w] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True
    return d


def render_args(lr, a, s):
    """the trainer ARGS of a run, with every intended flag asserted (and nothing else)"""
    assert lr in GRID[a] and 0 <= s <= 7, (lr, a, s)
    tag, save = f"1b-lr{lr}-7pct-{a}", s >= 2
    args = (f"--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {ARM[a]} "
            f"--arith_lines 4 {MODEL} --lr {lr} --seed {s} --tag _{tag}{' --save_model' if save else ''} --ckpt_every {CKPT_EVERY}")
    f = flags(args)
    want = {"--lr": lr, "--arith_lines": "4", "--seed": str(s), "--tag": f"_{tag}", "--out": f"{R}/out/{tag}",
            "--save_model": True if save else None, "--ckpt_every": CKPT_EVERY, "--tokenization": "whole",
            "--data": f"{B}/pretrain_data", "--tokenizer": str(TOK), "--lam": None, "--geo_layers": None, "--bs": None,
            "--tokens": None, "--seq": None, "--eval_every": None, "--n_eval": None, "--gate_start": None,
            "--gate_hyst": None, "--untie": None, "--stop_after": None, "--grad_ckpt": None, "--lm_chunk_tokens": None,
            "--dry_inputs": None, **WANT[a], **WANT_SIZE}
    for k, v in want.items():
        assert f.get(k) == v, (tag, s, k, f.get(k), v)
    assert float(f["--lr"]) == float(lr) and set(f) <= {k for k, v in want.items() if v is not None}, (tag, s, sorted(f))
    if a == "none":
        assert not ({"--schedule", "--window", "--site", "--gate_thr"} & set(f)), (tag, s, sorted(f))
    return tag, args


def snapshot():
    if not (R / "code").exists():   # once; temp dir + rename, so a failed extraction leaves no snapshot
        tmp = R / "code.tmp"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        arc = subprocess.run(["git", "-C", str(REPO), "archive", PIN, "tools"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
        (tmp / "COMMIT").write_text(PIN + "\n")
        tmp.rename(R / "code")
    sha = hashlib.sha256((R / "code" / "tools" / "pretrain_range.py").read_bytes()).hexdigest()[:12]
    assert sha == PIN_SHA and (R / "code" / "COMMIT").read_text().strip() == PIN, (sha, PIN_SHA)
    c = filecmp.dircmp(R / "code" / "tools", F87 / "tools", ignore=["__pycache__"])
    assert not (c.left_only or c.right_only or c.diff_files or c.funny_files), (c.left_only, c.right_only, c.diff_files)
    assert all(filecmp.cmp(R / "code/tools" / f, F87 / "tools" / f, shallow=False) for f in c.common_files)


def job_text(i, lr, a, s):
    tag, args = render_args(lr, a, s)
    jt = f"{tag}-s{s}"
    return jt, f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptlr1b-slot{i % SLOTS}
#SBATCH --dependency=singleton
#SBATCH --account={ACCOUNTS[1]}
#SBATCH --gres=gpu:1
#SBATCH --constraint=b200
#SBATCH --comment="preemption=yes"
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time={HOURS}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={B}/logs/llmadd_ptlr1b-{jt}_%j.out
# ladder-1b run {jt}: the 1B point of F87's 7% size ladder on B200; code = llmadd-burst commit in code/COMMIT; intent spec
# runs/ptlr_1b/AUDIT.md. Preemptible: a preempted job is requeued (same job id) and re-enters here; the trainer resumes
# from its ckpt (every {CKPT_EVERY} steps) and run_guard lets the same job id continue. The account line is overridden at
# submission for the second copy (make_jobs_ptlr1b.py); the slot name + singleton keep at most {SLOTS} runs at a time.
set -eu
# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/ptlr_1b/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
# the snapshot must still be the pinned trainer (code_sha of every result); exit before any GPU work otherwise
[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{PIN_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
echo "=== {jt} start $(date -Is) job ${{SLURM_JOB_ID:-none}} account ${{SLURM_JOB_ACCOUNT:-?}} partition ${{SLURM_JOB_PARTITION:-?}} restart ${{SLURM_RESTART_COUNT:-0}} node $(hostname)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader)"; echo "$GPU"
# one GPU type for every 1B run (criterion (f)): exit before any training on anything else
case "$GPU" in *B200*) ;; *) echo "not a B200: exiting"; exit 4;; esac
ARGS="{args}"
{PY} pretrain_range.py $ARGS --dry_inputs
{PY} pretrain_range.py $ARGS
echo "=== {jt} trainer returned $(date -Is)"
# after stage A's last run: the pre-registered selection (picks.json) and stage B (+ C if APPROVE_C), once (lock)
bash {R}/maybe_select.sh || echo "maybe_select.sh failed (rc $?): selection not run"
"""


def main():
    snapshot()
    R.mkdir(parents=True, exist_ok=True)
    cs, seen, n, bad = cells(), set(), 0, 0
    for i, (lr, a, s) in enumerate(cs):
        jt, txt = job_text(i, lr, a, s)
        assert jt not in seen, jt; seen.add(jt)
        job = R / f"job_{jt}.sbatch"
        job.write_text(txt)
        n += 1
        for acct in ACCOUNTS:
            short = acct.split("_")[2]
            flag = R / f"SUBMITTED_{jt}_{short}"
            cmd = ["sbatch", f"--account={acct}", str(job)]
            if "--test-only" in sys.argv:
                o = subprocess.run(["sbatch", "--test-only"] + cmd[1:], capture_output=True, text=True)
                lines = (o.stdout + o.stderr).strip().splitlines()
                last = next((x for x in lines if "to start at" in x), lines[-1] if lines else "")
                ok = o.returncode == 0 and "to start at" in last
                bad += not ok
                print(f"{'OK ' if ok else 'BAD'} {jt} {short}: {last}")
            elif "--submit" in sys.argv and not flag.exists():
                o = subprocess.run(cmd, capture_output=True, text=True)
                if o.returncode == 0:
                    flag.write_text(o.stdout)
                    print(jt, short, o.stdout.strip())
                else:
                    bad += 1
                    print("FAILED", jt, short, o.stderr.strip())
    print(n, "runs,", len(ACCOUNTS) * n, "job copies")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
