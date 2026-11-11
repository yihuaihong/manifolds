"""Torch jobs (2026-10-16, next-phase batch 1, plan notes/next_phase_synthesis.md T1 + T3; user: every arm, the prior
too, gets its own best lr): lr brackets in the unified 0..999 setup of runs/pt999 (tools/pretrain_range.py, helix3,
operands in the lines = ctx, 7% = 4 arithmetic lines, 3e8 tokens, L40S). Intent spec + pre-registered rules:
runs/ptlr/AUDIT.md. Seeds 0-3, every model saved.
  53M (d384 x 6; pt999 base lr 1e-3): none at lr 4e-3 and 8e-3; gate-ctx at 4e-3; win-ctx (window 0.05,0.4) at lr
    2e-3 and 4e-3. Reused (L40S, other snapshots, ctx / window path regression-identical): none / gate-ctx at 5e-4,
    1e-3, 2e-3 (pt999 sens-lr5e-4-*, 7pct-*, sens-lr2e-3-*); win-ctx at 5e-4 and 1e-3 (ptgw lr5e-4-W, main-W).
    none@8e-3 was conditional (iff none@4e-3 > none@2e-3 + 2); queued now (audit 2026-10-16: that trigger is inside the
    seed noise, paired none 2e-3 - 1e-3 on seeds 0-3 = +4.8 / +3.1 / -2.1 / +25.7, and each trigger costs a ~2-day
    queue trip; user: maximise parallelism).
  190M (d768 x 12; ptsize999 lr 6e-4): none at lr 1.2e-3 and 2.4e-3; win-ctx at 1.2e-3.
    Reused: ptsize999 190m-7pct-{none,gate-ctx,win-ctx} at 6e-4.
The 2% (main setting) lr bracket is NOT here: Burst pt2main (tools/make_pt2main_tasks.py, tasks 350-357: 53M and 190M
2%, none and win-ctx, each with its own none on Burst). Here 2% only as T2 (runs/pt999 2pct-win-ctx, lr 1e-3).
Conditional cells: the CONDITIONAL block below CELLS (pre-registered in AUDIT.md, not generated).
Code snapshot = tools/ at commit PIN (git archive, so a later HEAD, e.g. Rev A, cannot change it); its
pretrain_range.py is the ptgw snapshot (code_sha a3bce3ab63c1). Tags are unique across pt999 / pt999b / ptgw /
ptsize999 (tools/pt999_summary.py asserts a unique (tag, seed) across directories).
Account <slurm-account> on the public L40S partition. --submit submits jobs not yet submitted."""
import hashlib, shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptlr"
PIN, PIN_SHA = "64abc19", "a3bce3ab63c1"   # commit; sha256[:12] of its tools/pretrain_range.py (= code_sha in every result)
if not (R / "code").exists():   # once; written to a temp dir and renamed, so a failed extraction leaves no snapshot
    tmp = R / "code.tmp"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    arc = subprocess.run(["git", "-C", str(REPO), "archive", PIN, "tools"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
    (tmp / "COMMIT").write_text(PIN + "\n")
    tmp.rename(R / "code")
code_sha = hashlib.sha256((R / "code" / "tools" / "pretrain_range.py").read_bytes()).hexdigest()[:12]
assert code_sha == PIN_SHA and (R / "code" / "COMMIT").read_text().strip() == PIN, (code_sha, PIN_SHA)
G = "--schedule gate --gate_thr 0.5"
ARM = {"none": "--target none", "gate-ctx": f"--target helix3 {G} --site ctx",
       "win-ctx": "--target helix3 --schedule window --window 0.05,0.4 --site ctx"}
# size: (model args without --lr, time limit); 53M 0.96 L40S-h per run, 190M 2.6 L40S-h (gradient checkpointing on)
SIZES = {"53m": ("", "03:00:00"),
         "190m": ("--d_model 768 --n_layers 12 --n_heads 12 --d_ff 3072", "06:00:00"),
         "500m": ("--d_model 1024 --n_layers 24 --n_heads 16 --d_ff 4096", "13:00:00")}   # ptsize999 500m: 6.4 L40S-h
CELLS = [("53m", "4e-3", "none"), ("53m", "4e-3", "gate-ctx"), ("53m", "2e-3", "win-ctx"), ("53m", "4e-3", "win-ctx"),
         ("190m", "1.2e-3", "none"), ("190m", "2.4e-3", "none"), ("190m", "1.2e-3", "win-ctx"),
         ("53m", "8e-3", "none")]   # appended after the audit (2026-10-16), before any submission
# CONDITIONAL (pre-registered in runs/ptlr/AUDIT.md; NOT generated -- uncomment only when its rule has fired or the user
# has queued it up front, once):
CELLS += [("190m", "4.8e-3", "none")]   # queued up front (user 2026-10-16: maximise parallelism; every arm at its own best lr)
CELLS += [("190m", "2.4e-3", "win-ctx")]   # queued up front, same decision, before any result
# 2026-10-17, after the first read-out (tools/ptscale_summary.py), before any confirm seed exists -- the pre-registered
# open-bracket and confirm-seed rules (AUDIT.md amendment 3): none's bracket is open at the upper edge at both sizes
# (53M 8e-3: 65.8 > 4e-3: 57.9; 190M 4.8e-3: 64.3 > 1.2e-3: 56.4) -> one more doubling; confirm seeds 4-7 for none at the
# current best and the next lr, and for win-ctx at the tied best lrs; 505M: none was run at 3e-4 only (17.1 vs 190M none
# 64.3) -> none at 4x and 16x, and win-ctx at 4x (the 190M optimum ratios). Entries: (size, lr, arm[, seeds]).
CELLS += [("53m", "1.6e-2", "none"), ("53m", "8e-3", "none", range(4, 8)), ("53m", "1.6e-2", "none", range(4, 8)),
          ("53m", "2e-3", "win-ctx", range(4, 8)), ("53m", "1e-3", "win-ctx", range(4, 8)),
          ("190m", "9.6e-3", "none"), ("190m", "4.8e-3", "none", range(4, 8)), ("190m", "1.2e-3", "win-ctx", range(4, 8)),
          ("500m", "1.2e-3", "none"), ("500m", "4.8e-3", "none"), ("500m", "1.2e-3", "win-ctx")]
# 2026-10-18, after the read-out (amendment 4): 505M none's bracket is open at the top (4.8e-3: 57.6 > 1.2e-3: 47.3)
# -> none@9.6e-3; win-ctx's best (1.2e-3) is clear -> its confirm seeds 4-7 now (none's follow once its bracket closes)
CELLS += [("500m", "9.6e-3", "none"), ("500m", "1.2e-3", "win-ctx", range(4, 8))]
# 2026-10-19, after the read-out (amendment 5): 505M none's bracket is still open at the top (9.6e-3: 62.1 > 4.8e-3: 57.6)
# -> none@1.92e-2 (the pre-registered doubling); none@9.6e-3 confirm seeds 4-7 queued up front in the same step (user:
# maximise parallelism, arXiv freeze 10-21), so the confirm seeds exist whichever of the two wins
CELLS += [("500m", "1.92e-2", "none"), ("500m", "9.6e-3", "none", range(4, 8))]
# intent of every cell, asserted on the rendered ARGS below (argparse keeps the LAST of a repeated flag: none allowed)
WANT = {"none": {"--target": "none"},
        "gate-ctx": {"--target": "helix3", "--schedule": "gate", "--gate_thr": "0.5", "--site": "ctx"},
        "win-ctx": {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx"}}
WANT_SIZE = {"53m": {"--d_model": None, "--n_layers": None, "--n_heads": None, "--d_ff": None},
             "190m": {"--d_model": "768", "--n_layers": "12", "--n_heads": "12", "--d_ff": "3072"},
             "500m": {"--d_model": "1024", "--n_layers": "24", "--n_heads": "16", "--d_ff": "4096"}}


def flags(args):
    """{--flag: value or True} of an ARGS string; asserts that no flag is given twice."""
    t, d = args.split(), {}
    for i, w in enumerate(t):
        if w.startswith("--"):
            assert w not in d, f"{w} given twice: {args}"
            d[w] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True
    return d


n, tags = 0, set()
R.mkdir(parents=True, exist_ok=True)
for cell in CELLS:
    sz, lr, a = cell[:3]
    seeds = cell[3] if len(cell) > 3 else range(4)
    tag = f"{sz}-lr{lr}-7pct-{a}"
    assert (tag, tuple(seeds)) not in tags; tags.add((tag, tuple(seeds)))
    model, hours = SIZES[sz]
    extra = " ".join(x for x in (ARM[a], "--arith_lines 4", model, f"--lr {lr}") if x)
    for s in seeds:
        jt = f"{tag}-s{s}"
        args = (f"--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {extra} --seed {s} "
                f"--tag _{tag} --save_model --ckpt_every 2000")
        f = flags(args)
        want = {"--lr": lr, "--arith_lines": "4", "--seed": str(s), "--tag": f"_{tag}", "--out": f"{R}/out/{tag}",
                "--save_model": True, "--ckpt_every": "2000", "--tokenization": "whole", "--data": f"{B}/pretrain_data",
                "--lam": None, "--geo_layers": None, "--bs": None, "--tokens": None, "--gate_start": None,
                "--gate_hyst": None, "--untie": None, **WANT[a], **WANT_SIZE[sz]}
        for k, v in want.items():
            assert f.get(k) == v, (jt, k, f.get(k), v)
        assert float(f["--lr"]) == float(lr) and set(f) <= set(want) | {"--tokenizer", "--target", "--schedule", "--window",
                                                                         "--site", "--gate_thr"}, (jt, sorted(f))
        if a == "none":
            assert not ({"--schedule", "--window", "--site", "--gate_thr"} & set(f)), (jt, sorted(f))
        job = R / f"job_{jt}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptlr-{jt}
#SBATCH --account=<slurm-account>
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={hours}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# lr bracket (best lr per arm), unified 0..999 setup; code = llmadd-burst commit in code/COMMIT; intent spec runs/ptlr/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/ptlr/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
# the snapshot must still be the pinned trainer (code_sha of every result); exit before any GPU work otherwise
[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{PIN_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="{args}"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS
""")
        n += 1
        flag = R / f"SUBMITTED_{jt}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text(o.stdout)
            else:
                print(jt, o.stderr.strip())
print(n, "jobs")
