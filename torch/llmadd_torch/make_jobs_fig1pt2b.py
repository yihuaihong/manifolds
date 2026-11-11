"""Torch jobs (2026-10-22, fig1_pt2b): re-runs of F86's 2% main setting pt2b (53M from scratch, --arith_lines 1, 3e8
tokens = 18310 updates, helix3 prior on the operands = site ctx, window 0.05,0.4, lambda 5) for the paper's new Figure 1,
seed 0, with the model state saved after 0, 916 (window opens), 3662 (20%), 7324 (window closes), 9324 (+2000), 18309
(the state of the last logged eval's forward) and 18310 (final) updates (tools/pretrain_range.py Rev E --save_at):
  none          @ lr 8e-3   (none's area-best lr in F86)
  win-ctx       @ lr 2e-3   (the prior's area-best lr in F86)
  win-shuf-ctx  @ lr 2e-3   (helix3_shuf, the per-seed whole-table permutation of F82 addendum / F86's pt2long shuffle
                             control; at the PRIOR's lr 2e-3: no pt2b shuffle cell exists in F86)
Every other trainer argument = make_pt2main_tasks.py render() for pt2b (data / tokenizer = the sha-identical Torch files).
Intent spec: runs/fig1_pt2b/AUDIT.md. Export: tools/fig1_export.py (not run by these jobs).
Code: runs/fig1_pt2b/code = git archive of tools/ at the COMMITTED HEAD of the fig1-export worktree (llmadd-burst-fig1);
code/COMMIT records HEAD and the sha256[:12] of the trainer and every module it imports; the generator refuses to render
when the snapshot differs from HEAD; every sbatch re-checks the trainer's sha (exit 3), runs --dry_inputs and requires
the sha256 of its 9 DRY lines = the Burst / Torch seed-0 value of pt2main (exit 3), and exits 0 when runs/fig1_pt2b/HOLD
exists. Account torch_pr_520_tandon_priority, a100_tandon, QoS gpu48, --nice=50.
    python make_jobs_fig1pt2b.py [--print] [--test-only] [--submit]"""
import hashlib, shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/scratch/yh6210/llmadd-burst-fig1")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "fig1_pt2b"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
DEPS = ("pretrain_range.py", "pretrain_helix.py", "numlib.py", "digit_split_data.py", "run_guard.py")
BRANCH = "fig1-export"
STEPS = 18310                                   # int(3e8 // (32 * 512))
SAVE_AT = (0, 916, 3662, 7324, 9324, 18309, 18310)
# the trainer's window rule (prior on iff 0.05 * steps <= s < 0.4 * steps): first on step 916, first off step 7324
assert [s for s in range(STEPS) if 0.05 * STEPS <= s < 0.4 * STEPS][0] == 916
assert [s for s in range(STEPS) if 0.05 * STEPS <= s < 0.4 * STEPS][-1] + 1 == 7324 and 0.2 * STEPS == 3662
DRY_SHA = "379354e22b42e615e89aee1398d72e768837c900d1db089c60f88e8649007346"   # make_pt2main_tasks.DRY[pretrain_data, 53m, ctx, 0]
DRY_GREP = "^(line:|held_b|ood_b|\\{'held_b11'|model:|train row)"
DATA_SHA = ("4de2ed19093c37b302b6a1ced56136e6a8b805e6d6b48057a77316811704ff11",   # make_pt2main_tasks.DATA['pretrain_data']
            "d6f744a99b988ed6928fb588356cc8fe3d61bcf348b8d06fc0eece25c624a4c8")
git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout
head = git("rev-parse", "--short", "HEAD").decode().strip()
assert git("rev-parse", "--abbrev-ref", "HEAD").decode().strip() == BRANCH, "the worktree must be on " + BRANCH
assert subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", "tools/"]).returncode == 0, \
    "tools/ has uncommitted changes in the worktree: commit them (the snapshot is the committed code)"
R.mkdir(parents=True, exist_ok=True)
if not (R / "code").exists():   # once; temp dir + rename
    tmp = R / "code.tmp"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=git("archive", head, "tools"), check=True)
    (tmp / "COMMIT").write_text(f"{head} ({BRANCH}, committed; git archive of tools/); sha256[:12]: "
                                + ", ".join(f"{f} {sha(tmp / 'tools' / f)}" for f in DEPS) + "\n")
    tmp.rename(R / "code")
PIN_SHA = sha(R / "code" / "tools" / "pretrain_range.py")
assert f"pretrain_range.py {PIN_SHA}" in (R / "code" / "COMMIT").read_text()
for f in DEPS:   # the snapshot = the committed HEAD (trainer and the modules it imports)
    assert hashlib.sha256(git("show", f"HEAD:tools/{f}")).hexdigest()[:12] == sha(R / "code" / "tools" / f), f

W = "--schedule window --window 0.05,0.4"
CELLS = [("none", "8e-3", "--target none", {"--target": "none", "--schedule": None, "--window": None, "--site": None}),
         ("win-ctx", "2e-3", f"--target helix3 {W} --site ctx",
          {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx"}),
         ("win-shuf-ctx", "2e-3", f"--target helix3_shuf {W} --site ctx",
          {"--target": "helix3_shuf", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx"})]


def flags(args):
    """{--flag: value or True} of an ARGS string; asserts that no flag is given twice."""
    t, d = args.split(), {}
    for i, w in enumerate(t):
        if w.startswith("--"):
            assert w not in d, f"{w} given twice: {args}"
            d[w] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True
    return d


existing = {p.name for p in B.glob("runs/*/out/*")}
for arm, lr, ARM, WANT in CELLS:
    tag = f"fig1-pt2b-lr{lr}-{arm}"
    assert tag not in existing or (R / "out" / tag).exists(), tag
    sv = ",".join(map(str, SAVE_AT))
    # = make_pt2main_tasks.render() for pt2b (53m, 3e8 = default --tokens), + --save_at; Torch data / tokenizer
    args = (f"--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {ARM} --arith_lines 1 "
            f"--lr {lr} --seed 0 --tag _{tag} --save_model --ckpt_every 2000 --save_at {sv}")
    f = flags(args)
    want = {"--data": f"{B}/pretrain_data", "--tokenizer": str(TOK), "--tokenization": "whole", "--out": f"{R}/out/{tag}",
            "--arith_lines": "1", "--lr": lr, "--seed": "0", "--tag": f"_{tag}", "--save_model": True, "--ckpt_every": "2000",
            "--save_at": sv, "--tokens": None, "--lam": None, "--geo_layers": None, "--bs": None, "--seq": None,
            "--eval_every": None, "--n_eval": None, "--d_model": None, "--n_layers": None, "--n_heads": None, "--d_ff": None,
            "--untie": None, "--grad_ckpt": None, "--lm_chunk_tokens": None, "--stop_after": None, "--dry_inputs": None, **WANT}
    for k, v in want.items():
        assert f.get(k) == v, (tag, k, f.get(k), v)
    assert set(f) <= {k for k, v in want.items() if v is not None}, (tag, sorted(f))
    jn = f"llmadd_fig1-{arm}-s0"
    job = R / f"job_{arm}-s0.sbatch"
    job.write_text(f"""#!/bin/bash
#SBATCH --job-name={jn}
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --nice=50
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# fig1_pt2b: {arm} @ lr {lr}, seed 0, pt2b args + --save_at {sv}; code = runs/fig1_pt2b/code (COMMIT); intent spec runs/fig1_pt2b/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/fig1_pt2b/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{PIN_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
(cd {B}/pretrain_data && sha256sum -c -) <<'SHA' || exit 3
{DATA_SHA[0]}  train.bin
{DATA_SHA[1]}  val.bin
SHA
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ARGS="{args}"
mkdir -p {R}/out/{tag}
{PY} pretrain_range.py $ARGS --dry_inputs > {R}/out/{tag}/dry_inputs.txt 2>&1 || {{ cat {R}/out/{tag}/dry_inputs.txt; exit 3; }}
cat {R}/out/{tag}/dry_inputs.txt
grep -qxF 'model: 52.7M parameters; gradient checkpointing off' {R}/out/{tag}/dry_inputs.txt || {{ echo "## not the 53M model"; exit 3; }}
got=$(set +o pipefail; grep -E "{DRY_GREP}" {R}/out/{tag}/dry_inputs.txt | head -n 9 | sha256sum | cut -d' ' -f1)
[ "$got" = {DRY_SHA} ] || {{ echo "## dry_inputs rows differ from pt2main seed 0 (sha256 $got)"; exit 3; }}
echo "## dry_inputs rows = pt2main seed 0 ({DRY_SHA[:12]})"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 120 > {R}/out/{tag}/gpu.csv &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
{PY} pretrain_range.py $ARGS
ls -la {R}/out/{tag}/save_at
""")
    if "--print" in sys.argv:
        print(f"{jn}: {args}")
    flag = R / f"SUBMITTED_{arm}-s0"
    if "--test-only" in sys.argv:
        o = subprocess.run(["sbatch", "--test-only", str(job)], capture_output=True, text=True)
        print(jn, (o.stdout + o.stderr).strip())
    if "--submit" in sys.argv and not flag.exists():
        o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
        print(jn, (o.stdout + o.stderr).strip())
        if o.returncode == 0:
            flag.write_text(o.stdout)
print(f"{len(CELLS)} jobs rendered in {R}; code_sha {PIN_SHA} (HEAD {head}); save_at {SAVE_AT}; "
      f"submitted: {'yes' if '--submit' in sys.argv else 'no'}")
