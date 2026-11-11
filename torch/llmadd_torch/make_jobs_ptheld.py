"""Torch jobs (2026-10-19, wall 1 "data missing / long tail"; WRITING_HANDOFF 2026-10-19: the discussion may say which
walls the prior breaks, only with experiments): can a number prior let a 53M model pretrained from scratch add with numbers
it never saw in any arithmetic line? tools/pretrain_range.py Rev C --held_operands 50: H = 50 three-digit numbers per seed
(10 per quintile of their text count), never an operand or an answer token in a training line, still in the FineWeb text.
7% share (--arith_lines 4), 53M (d384 x 6), 3e8 tokens, seeds 0-3, every model saved. Intent spec and PRE-REGISTERED
rules: runs/ptheld/AUDIT.md.
  MAIN (7 cells x 4 seeds): none @ lr 2e-3 and 8e-3 (8e-3 = none's best at 7%, ptlr); win-ctx (operands in the lines:
    never touches H), win-text (numbers in the text: reaches H through the text), win-emb (the embedding rows 0..999:
    reaches H without text) @ 2e-3; win-emb-hremap (Rev C1, fresh audit finding 1: helix3 with only H remapped,
    h -> helix3(u_h), at the emb site: the mechanism test, predicted wrong answer u_h + y) @ 2e-3; win-text-farshuf
    (helix3 of a FAR permutation of all of 0..999 at the text site: the geometry-specificity control of win-text) @ 2e-3.
    Window 0.05,0.4, lambda 5.
  LC (4 cells x 4 seeds; lesson c: "cannot" must survive the baseline's own lr x wd sweep with the optimum bracketed and a
    4x budget): none @ 1.6e-2; none @ 8e-3 x wd 0.01 and wd 1.0; none @ 8e-3 with 4x the tokens (1.2e9).
  CONF (only with --confirm_seeds; the user's call, BEFORE any result is read): seeds 4-7 of the five R1 cells (none @ 2e-3,
    none @ 8e-3, win-ctx, win-text, win-emb), for the n = 8 paired tests of AUDIT.md R1(c).
  (win-emb-farshuf, an earlier optional cell, is superseded by win-emb-hremap.)
Code: tools/ at the repo HEAD (git archive) + the WORKING-TREE tools/pretrain_range.py (Rev C is uncommitted: the repo is
public and the agent does not commit). code/COMMIT records HEAD and the sha256[:12] of the trainer and of every module it
imports. The generator refuses to render when code/tools/pretrain_range.py differs from the repo's working tree (refresh
= delete code/, only before any submission); every sbatch re-checks the sha at start and exits 3 before any GPU work.
Account torch_pr_625_general on the public L40S partition, QoS gpu48. --submit submits jobs not yet submitted
(SUBMITTED_<job> flags); a pending job exits at start when runs/ptheld/HOLD exists.
    python make_jobs_ptheld.py [--confirm_seeds] [--submit]"""
import hashlib, shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "ptheld"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
DEPS = ("pretrain_range.py", "pretrain_helix.py", "numlib.py", "digit_split_data.py", "run_guard.py")   # what it imports
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                      check=True).stdout.strip()
if not (R / "code").exists():   # once; written to a temp dir and renamed, so a failed extraction leaves no snapshot
    tmp = R / "code.tmp"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    arc = subprocess.run(["git", "-C", str(REPO), "archive", head, "tools"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
    shutil.copy(REPO / "tools" / "pretrain_range.py", tmp / "tools" / "pretrain_range.py")
    committed = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", "tools/pretrain_range.py"]).returncode == 0
    (tmp / "COMMIT").write_text(f"{head} (git archive of tools/) + tools/pretrain_range.py from the working tree ("
                                + ("= HEAD's, committed" if committed else "Rev C1, UNCOMMITTED") + "); sha256[:12]: "
                                + ", ".join(f"{f} {sha(tmp / 'tools' / f)}" for f in DEPS) + "\n")
    tmp.rename(R / "code")
PIN_SHA = sha(R / "code" / "tools" / "pretrain_range.py")
assert PIN_SHA == sha(REPO / "tools" / "pretrain_range.py"), \
    f"code/ snapshot {PIN_SHA} != working tree {sha(REPO / 'tools' / 'pretrain_range.py')}: delete code/ (before any submission)"
assert f"pretrain_range.py {PIN_SHA}" in (R / "code" / "COMMIT").read_text()
for f in DEPS[1:]:   # the imported modules of the snapshot = the repo HEAD's (the trainer was regression-tested with them)
    want = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:tools/{f}"], capture_output=True, check=True).stdout
    assert hashlib.sha256(want).hexdigest()[:12] == sha(R / "code" / "tools" / f), f

W = "--schedule window --window 0.05,0.4"
ARM = {"none": "--target none", "win-ctx": f"--target helix3 {W} --site ctx", "win-text": f"--target helix3 {W} --site text",
       "win-emb": f"--target helix3 {W} --site emb", "win-emb-hremap": f"--target helix3_hremap {W} --site emb",
       "win-text-farshuf": f"--target helix3_farshuf {W} --site text"}
WANT = {"none": {"--target": "none", "--schedule": None, "--window": None, "--site": None},
        "win-ctx": {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx"},
        "win-text": {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "text"},
        "win-emb": {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "emb"},
        "win-emb-hremap": {"--target": "helix3_hremap", "--schedule": "window", "--window": "0.05,0.4", "--site": "emb"},
        "win-text-farshuf": {"--target": "helix3_farshuf", "--schedule": "window", "--window": "0.05,0.4", "--site": "text"}}
SEEDS, CONF_SEEDS = range(4), range(4, 8)
# (group, arm, lr, wd or None, token multiple, seeds)
CELLS = [("MAIN", "none", "2e-3", None, 1, SEEDS), ("MAIN", "none", "8e-3", None, 1, SEEDS),
         ("MAIN", "win-ctx", "2e-3", None, 1, SEEDS), ("MAIN", "win-text", "2e-3", None, 1, SEEDS),
         ("MAIN", "win-emb", "2e-3", None, 1, SEEDS), ("MAIN", "win-emb-hremap", "2e-3", None, 1, SEEDS),
         ("MAIN", "win-text-farshuf", "2e-3", None, 1, SEEDS),
         ("LC", "none", "1.6e-2", None, 1, SEEDS), ("LC", "none", "8e-3", "0.01", 1, SEEDS),
         ("LC", "none", "8e-3", "1.0", 1, SEEDS), ("LC", "none", "8e-3", None, 4, SEEDS)]
# PLR (2026-10-20, amendment PLR in AUDIT.md, before any result; report/audits/rescue_standard.md criterion (b): the
# prior arm gets its own lr sweep): win-text and win-emb at lr 1e-3 and 4e-3, bracketing the MAIN 2e-3
CELLS += [("PLR", a, lr, None, 1, SEEDS) for a in ("win-text", "win-emb") for lr in ("1e-3", "4e-3")]
if "--confirm_seeds" in sys.argv:   # the user's call (AUDIT.md open decisions), before any result is read
    CELLS += [("CONF", a, lr, None, 1, CONF_SEEDS) for a, lr in
              (("none", "2e-3"), ("none", "8e-3"), ("win-ctx", "2e-3"), ("win-text", "2e-3"), ("win-emb", "2e-3"))]
HOURS = {1: ("03:00:00", 1.1), 4: ("09:00:00", 4.3)}   # time limit, planned L40S-h per run (0.96 measured + ~9% H evals + counting)
N_HELD = 50


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
for grp, a, lr, wd, mult, seeds in CELLS:
    tag = f"53m-lr{lr}-7pct-h{N_HELD}-{a}" + (f"-wd{wd}" if wd else "") + (f"-{mult}x" if mult != 1 else "")
    # a tag once (CONF: the confirm seeds of a MAIN tag), never one of another batch's out dirs
    assert (tag in tags) == (grp == "CONF") and (tag not in existing or (R / "out" / tag).exists()), tag
    tags.add(tag)
    limit, h = HOURS[mult]
    extra = " ".join(x for x in (ARM[a], "--arith_lines 4", f"--held_operands {N_HELD}", f"--lr {lr}",
                                 f"--wd {wd}" if wd else "", f"--tokens {3e8 * mult:.1e}" if mult != 1 else "") if x)
    for s in seeds:
        jt = f"{tag}-s{s}"
        args = (f"--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {extra} --seed {s} "
                f"--tag _{tag} --save_model --ckpt_every 2000")
        f = flags(args)
        want = {"--lr": lr, "--arith_lines": "4", "--held_operands": str(N_HELD), "--seed": str(s), "--tag": f"_{tag}",
                "--out": f"{R}/out/{tag}", "--save_model": True, "--ckpt_every": "2000", "--tokenization": "whole",
                "--data": f"{B}/pretrain_data", "--tokenizer": str(TOK), "--wd": wd,
                "--tokens": f"{3e8 * mult:.1e}" if mult != 1 else None, "--lam": None, "--geo_layers": None, "--bs": None,
                "--gate_thr": None, "--gate_start": None, "--gate_hyst": None, "--untie": None, "--d_model": None,
                "--n_layers": None, "--seq": None, "--text_min": None, **WANT[a]}
        for k, v in want.items():
            assert f.get(k) == v, (jt, k, f.get(k), v)
        assert set(f) <= set(want), (jt, sorted(set(f) - set(want)))
        assert float(f["--lr"]) == float(lr) and (mult == 1 or float(f["--tokens"]) == 3e8 * mult)
        job = R / f"job_{jt}.sbatch"
        job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptheld-{jt}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={limit}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# wall 1 (held operands, {grp}): {a} @ lr {lr}{f', wd {wd}' if wd else ''}{f', {mult}x tokens' if mult != 1 else ''}; code = runs/ptheld/code (COMMIT); intent spec runs/ptheld/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/ptheld/HOLD exists
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
        flag = R / f"SUBMITTED_{jt}"
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text(o.stdout)
            else:
                print(jt, o.stderr.strip())
    rows.append((grp, tag, len(seeds), limit, h * len(seeds)))
for r in rows:
    print(f"{r[0]:5s} {r[1]:42s} x{r[2]}  limit {r[3]}  {r[4]:.1f} L40S-h")
print(n, "jobs;", ", ".join(f"{g} {v:.1f}" for g, v in hours.items()), f"L40S-h; total {sum(hours.values()):.1f}; code_sha {PIN_SHA}")
