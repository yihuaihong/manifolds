"""Torch jobs (2026-10-21, ptheld2 = wall 1 follow-up; F90): with H stratified by VALUE x text frequency, does a number
prior let a 53M model pretrained from scratch add with three-digit numbers it never saw in any arithmetic line, beyond
what none reaches? tools/pretrain_range.py Rev D --held_operands 54 --held_strat value_freq: H = 6 numbers per hundreds
block 100-199 .. 900-999 (3 from the block's rarer, 3 from its commoner 50 by text-site count), never an operand or an
answer token in a training line, still in the FineWeb text. 7% share (--arith_lines 4), 53M (d384 x 6), 3e8 tokens, eval
every 250 steps, every model saved. Intent spec and PRE-REGISTERED rules: runs/ptheld2/AUDIT.md.
  none @ lr 2e-3 and 8e-3 (seeds 0-7); none @ 8e-3 with 4x the tokens (1.2e9; seeds 0-3; rescue criterion (a)).
  Prior arms (helix3, window 0.05,1.0 = on from 5% to the end, lambda 1, lr 2e-3; seeds 0-7):
    win-emb-always   the embedding rows of 0..999 (reaches H without text)
    win-emb-decay    the same with --lam_decay (lambda 1 -> 0 linearly over the window)
    win-text-always  numbers in the text (reaches H through its natural occurrences)
    win-ctx-always   operands in the lines (never touches H: the non-specific control)
  SUPPLEMENTARY (only with --supp; rendered for the user's approval, submitted only with --supp --submit; amendment SUPP in
  AUDIT.md, before any result): (i) rescue_standard (a), none's lr x wd bracket on this H: none @ 1.6e-2, none @ 8e-3 wd
  0.01 and wd 1.0 (seeds 0-3); (ii) rescue_standard (d), the remapped-target control of the emb arms:
  win-emb-hremap-always = helix3_hremap (h at the helix position of a seen u_h) at the emb site, window 0.05,1.0,
  lambda 1, lr 2e-3 (seeds 0-7).
  SUPPLEMENTARY 2 (only with --supp2; amendment AUD2 in AUDIT.md, before any result; the user's call): (b)
  rescue_standard (b), the prior arms' own lr sweep: win-emb-always, win-emb-decay, win-text-always at lr 1e-3 and 4e-3
  (seeds 0-3); (d) the same-schedule control of win-emb-decay: win-emb-hremap-decay = helix3_hremap at the emb site with
  --lam_decay, window 0.05,1.0, lambda 1, lr 2e-3 (seeds 0-7).
  SUPPLEMENTARY 3 (only with --supp3, which needs --site tandon; amendment WD3 in AUDIT.md, written AFTER the wd 1.0
  result was read, before any of these runs; the user's call): rescue_standard (a) completed on A100 (the batch's GPU):
  none @ 8e-3 wd 0.3 (the wd axis becomes 0.01 / 0.1 / 0.3), none @ 1e-3 (the lr axis' lower edge) and none @ 2e-3 with 4x
  the tokens (budget at none's best lr on layer A), seeds 0-3 each (12 jobs).
Code: runs/ptheld2/code = git archive of tools/ at the COMMITTED HEAD of the ptheld2-build worktree (llmadd-burst-p2);
code/COMMIT records HEAD and the sha256[:12] of the trainer and of every module it imports. The generator refuses to render
when the snapshot's trainer differs from HEAD's or the working tree's, or an imported module differs from HEAD's; every
cell's flags are asserted (intended values present, no other flag, no flag twice); every sbatch re-checks the trainer's
sha at start (exit 3) and exits 0 when runs/ptheld2/HOLD exists.
Account torch_pr_625_general on the public L40S partition, QoS gpu48. --submit submits jobs not yet submitted
(SUBMITTED_<job> flags); --print also prints every job's ARGS. Without --submit nothing is submitted.
    python make_jobs_ptheld2.py [--print] [--site tandon] [--supp] [--supp2] [--supp3] [--submit]"""
import hashlib, shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/scratch/yh6210/llmadd-burst-p2")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptheld2"
# --site tandon (amendment SITE, user decision 2026-10-21: the public L40S queue estimated a start on 10-27): every cell
# on A100 (a100_tandon, account torch_pr_520_tandon_priority) in runs/ptheld2t (its own out / HOLD / SUBMITTED), with the
# L40S batch's code snapshot copied byte for byte; the trainer args are unchanged except the --out root
TANDON = "--site" in sys.argv and sys.argv[sys.argv.index("--site") + 1] == "tandon"
assert "--site" not in sys.argv or TANDON, "--site tandon is the only alternative site"
if TANDON:
    R0, R = R, B / "runs" / "ptheld2t"
    if not (R / "code").exists():
        R.mkdir(parents=True, exist_ok=True); shutil.copytree(R0 / "code", R / "code.tmp"); (R / "code.tmp").rename(R / "code")
    for f in ("pretrain_range.py", "pretrain_helix.py", "numlib.py", "digit_split_data.py", "run_guard.py", "COMMIT"):
        q = "tools/" + f if f != "COMMIT" else f
        assert sha(R / "code" / q) == sha(R0 / "code" / q), f
ACCOUNT, PARTITION, JOBP = (("torch_pr_520_tandon_priority", "a100_tandon", "ptheld2t") if TANDON else
                            ("torch_pr_625_general", "l40s_public", "ptheld2"))
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
DEPS = ("pretrain_range.py", "pretrain_helix.py", "numlib.py", "digit_split_data.py", "run_guard.py")   # what it imports
BRANCH = "ptheld2-build"
git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout
head = git("rev-parse", "--short", "HEAD").decode().strip()
assert git("rev-parse", "--abbrev-ref", "HEAD").decode().strip() == BRANCH, "the worktree must be on " + BRANCH
assert subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", "tools/"]).returncode == 0, \
    "tools/ has uncommitted changes in the worktree: commit them (the snapshot is the committed code)"
if not (R / "code").exists():   # once; written to a temp dir and renamed, so a failed extraction leaves no snapshot
    tmp = R / "code.tmp"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=git("archive", head, "tools"), check=True)
    (tmp / "COMMIT").write_text(f"{head} ({BRANCH}, committed; git archive of tools/); sha256[:12]: "
                                + ", ".join(f"{f} {sha(tmp / 'tools' / f)}" for f in DEPS) + "\n")
    tmp.rename(R / "code")
PIN_SHA = sha(R / "code" / "tools" / "pretrain_range.py")
assert PIN_SHA == sha(REPO / "tools" / "pretrain_range.py"), \
    f"code/ snapshot {PIN_SHA} != working tree {sha(REPO / 'tools' / 'pretrain_range.py')}: delete code/ (before any submission)"
assert f"pretrain_range.py {PIN_SHA}" in (R / "code" / "COMMIT").read_text()
for f in DEPS:   # the snapshot = the committed HEAD (trainer and the modules it imports)
    assert hashlib.sha256(git("show", f"HEAD:tools/{f}")).hexdigest()[:12] == sha(R / "code" / "tools" / f), f

W = "--schedule window --window 0.05,1.0 --lam 1"
ARM = {"none": "--target none", "win-emb-always": f"--target helix3 {W} --site emb",
       "win-emb-decay": f"--target helix3 {W} --site emb --lam_decay", "win-text-always": f"--target helix3 {W} --site text",
       "win-ctx-always": f"--target helix3 {W} --site ctx", "win-emb-hremap-always": f"--target helix3_hremap {W} --site emb",
       "win-emb-hremap-decay": f"--target helix3_hremap {W} --site emb --lam_decay"}
PRI = {"--target": "helix3", "--schedule": "window", "--window": "0.05,1.0", "--lam": "1", "--lam_decay": None}
WANT = {"none": {"--target": "none", "--schedule": None, "--window": None, "--site": None, "--lam": None, "--lam_decay": None},
        "win-emb-always": {**PRI, "--site": "emb"}, "win-emb-decay": {**PRI, "--site": "emb", "--lam_decay": True},
        "win-text-always": {**PRI, "--site": "text"}, "win-ctx-always": {**PRI, "--site": "ctx"},
        "win-emb-hremap-always": {**PRI, "--target": "helix3_hremap", "--site": "emb"},
        "win-emb-hremap-decay": {**PRI, "--target": "helix3_hremap", "--site": "emb", "--lam_decay": True}}
S8, S4 = range(8), range(4)
# (group, arm, lr, token multiple, seeds, wd or None)
CELLS = [("NONE", "none", "2e-3", 1, S8, None), ("NONE", "none", "8e-3", 1, S8, None), ("NONE", "none", "8e-3", 4, S4, None),
         ("PRIOR", "win-emb-always", "2e-3", 1, S8, None), ("PRIOR", "win-emb-decay", "2e-3", 1, S8, None),
         ("PRIOR", "win-text-always", "2e-3", 1, S8, None), ("CTRL", "win-ctx-always", "2e-3", 1, S8, None)]
if "--supp" in sys.argv:   # amendment SUPP (AUDIT.md): the user's call; never submitted without --supp
    CELLS += [("SUPP-A", "none", "1.6e-2", 1, S4, None), ("SUPP-A", "none", "8e-3", 1, S4, "0.01"),
              ("SUPP-A", "none", "8e-3", 1, S4, "1.0"), ("SUPP-D", "win-emb-hremap-always", "2e-3", 1, S8, None)]
if "--supp2" in sys.argv:   # amendment AUD2 (AUDIT.md): the user's call; never submitted without --supp2
    CELLS += [("SUPP2-B", a, lr, 1, S4, None) for a in ("win-emb-always", "win-emb-decay", "win-text-always") for lr in ("1e-3", "4e-3")]
    CELLS += [("SUPP2-D", "win-emb-hremap-decay", "2e-3", 1, S8, None)]
if "--supp3" in sys.argv:   # amendment WD3 (AUDIT.md, post hoc: after the wd 1.0 result): A100 only, the user's call
    assert TANDON, "--supp3 needs --site tandon (amendment WD3: the cells complete the A100 batch runs/ptheld2t)"
    CELLS += [("SUPP3-A", "none", "8e-3", 1, S4, "0.3"), ("SUPP3-X", "none", "1e-3", 1, S4, None),
              ("SUPP3-X", "none", "2e-3", 4, S4, None)]
HOURS = {1: ("03:00:00", 1.1), 4: ("09:00:00", 4.3)}   # time limit, planned L40S-h per run (ptheld: 0.98 h measured at 1x)
N_HELD = 54


def flags(args):
    """{--flag: value or True} of an ARGS string; asserts that no flag is given twice."""
    t, d = args.split(), {}
    for i, w in enumerate(t):
        if w.startswith("--"):
            assert w not in d, f"{w} given twice: {args}"
            d[w] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True
    return d


existing = {p.name for p in B.glob("runs/*/out/*")} | {p.name for p in B.glob("runs/*/out_*/*")}
n, hours, tags, rows = 0, {}, set(), []
R.mkdir(parents=True, exist_ok=True)
for grp, a, lr, mult, seeds, wd in CELLS:
    tag = f"53m-lr{lr}-7pct-h{N_HELD}vf-{a}" + (f"-wd{wd}" if wd else "") + (f"-{mult}x" if mult != 1 else "")
    # a tag once, never one of another batch's out dirs
    assert tag not in tags and (tag not in existing or (R / "out" / tag).exists()), tag
    tags.add(tag)
    limit, h = HOURS[mult]
    extra = " ".join(x for x in (ARM[a], "--arith_lines 4", f"--held_operands {N_HELD}", "--held_strat value_freq",
                                 f"--lr {lr}", f"--wd {wd}" if wd else "", f"--tokens {3e8 * mult:.1e}" if mult != 1 else "") if x)
    for s in seeds:
        jt = f"{tag}-s{s}"
        args = (f"--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {extra} --seed {s} "
                f"--tag _{tag} --save_model --ckpt_every 2000")
        f = flags(args)
        want = {"--lr": lr, "--arith_lines": "4", "--held_operands": str(N_HELD), "--held_strat": "value_freq",
                "--seed": str(s), "--tag": f"_{tag}", "--out": f"{R}/out/{tag}", "--save_model": True, "--ckpt_every": "2000",
                "--tokenization": "whole", "--data": f"{B}/pretrain_data", "--tokenizer": str(TOK), "--wd": wd,
                "--tokens": f"{3e8 * mult:.1e}" if mult != 1 else None, "--geo_layers": None, "--bs": None,
                "--gate_thr": None, "--gate_start": None, "--gate_hyst": None, "--untie": None, "--d_model": None,
                "--n_layers": None, "--seq": None, "--text_min": None, "--eval_every": None, "--lam_rule": None, **WANT[a]}
        for k, v in want.items():
            assert f.get(k) == v, (jt, k, f.get(k), v)
        assert set(f) <= set(want), (jt, sorted(set(f) - set(want)))
        assert float(f["--lr"]) == float(lr) and (mult == 1 or float(f["--tokens"]) == 3e8 * mult)
        assert a == "none" or float(f["--lam"]) == 1.0
        job = R / f"job_{jt}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_{JOBP}-{jt}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={limit}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# ptheld2 (wall 1 follow-up, {grp}): {a} @ lr {lr}{f', wd {wd}' if wd else ''}{f', {mult}x tokens' if mult != 1 else ''}, H = 54 value x frequency; code = runs/{JOBP}/code (COMMIT); intent spec runs/ptheld2/AUDIT.md{' (amendment SITE)' if TANDON else ''}
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/{JOBP}/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
# the snapshot must still be the rendered trainer (code_sha of every result); exit before any GPU work otherwise
[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{PIN_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="{args}"
{PY} pretrain_range.py $ARGS --dry_inputs
{PY} pretrain_range.py $ARGS
""")
        n += 1; hours[grp] = hours.get(grp, 0) + h
        if "--print" in sys.argv:
            print(f"{jt}: {extra} --seed {s}")
        flag = R / f"SUBMITTED_{jt}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text(o.stdout)
            else:
                print(jt, o.stderr.strip())
    rows.append((grp, tag, len(seeds), limit, h * len(seeds)))
for r in rows:
    print(f"{r[0]:5s} {r[1]:44s} x{r[2]}  limit {r[3]}  {r[4]:.1f} L40S-h")
print(n, "jobs;", ", ".join(f"{g} {v:.1f}" for g, v in hours.items()), f"L40S-h; total {sum(hours.values()):.1f}; code_sha {PIN_SHA} "
      f"(HEAD {head}); submitted: {'yes' if '--submit' in sys.argv else 'no (render only)'}")
