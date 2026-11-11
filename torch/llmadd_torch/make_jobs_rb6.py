"""Torch jobs (2026-10-17, rainbow6 PILOT; NOT submitted by the agent): six manifolds pretrained at once at 53M
(tools/pretrain_range.py Rev B, --domains). User decisions 2026-10-17: numbers 0..999 (helix3), months (C12),
weekdays (C7), clock hours (C24, number tokens), S4 cube rotations (R, nonce single tokens), dimension lattice
{-1,0,1}^3 (exponent vectors, nonce single tokens); every arm sees the SAME data mixture, arms differ only in which
priors are on; data option 1 = numbers stay at the 2% main setting (--arith_lines 1) and the other five domains are
ADDED on top (--extra_lines K domain lines per row, cycled over the five). Intent spec + pre-registered pilot rule +
main-design sketch: runs/rb6/AUDIT.md (amendments A1, A2). Summary: tools/rainbow6_summary.py.
Amendment A2 (2026-10-17, user decisions, before any run): the numbers line is the FIRST line after the text in every
row (the default path's pairs / formats / text of the same seed), so numbers compare directly with the pt999 / pt2b
2pct cells; the domain lines follow it in random order; domain evals in that position; held-out-format evals (3-4
paraphrase templates per domain, never trained); all6 = lambda 5 PER ACTIVE PRIOR (--lam_rule each, the best
single-domain setup; norm stays selectable). The Burst version of the same pilot: tools/make_rb6_tasks.py (tasks
400-407, tag rb6pb-).
  PILOT  {none, all6} x {K = 1, 2 extra lines per row} x seeds 0, 1 = 8 runs; 53M (d384 x 6), 3e8 tokens, lr 1e-3,
         bs 32, seq 512; all6 = helix3 on the operands + the five domain priors on their item tokens (ctx), one shared
         window (0.05, 0.4), lambda rule LAM_RULE below (each = 5 per active prior: user decision 2026-10-17, A2;
         norm = 5 / number of active priors stays selectable; the none arm does not depend on it); --save_model,
         --ckpt_every 2000.
  MAIN   (sketch, not generated; --print-main lists it): none / all6 / 6 single-prior arms / all6-shuf x seeds 0-3 at
         the density the pilot selects, after the user approves.
Conventions as make_jobs_ptlr.py: code snapshot = tools/ at commit PIN (git archive into runs/rb6/code, COMMIT file);
every sbatch re-checks the sha256[:12] of the snapshot's pretrain_range.py (= code_sha in every result) and of
toy_struct.py (the S4 / dim tables, imported by the trainer); HOLD kill switch; SUBMITTED_<job> flags; --submit
submits jobs not yet submitted. PIN is empty until Rev B is committed: then the snapshot is not made and --submit is
refused (the job files are still rendered and their args asserted)."""
import hashlib, shutil, subprocess, sys
from pathlib import Path
B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
R = B / "runs" / "rb6"
PIN = ""                                  # commit of Rev B (set it after the commit, before --submit)
PIN_SHA = "cc984148b15a"                  # sha256[:12] of Rev B tools/pretrain_range.py (= code_sha of every result)
                                          # (A1: e231a219007f + the audit fixes of 2026-10-17; A2: numbers line first,
                                          # domain evals in that position, held-out-format evals; default path unchanged)
TOY_SHA = "6248613ac6d2"                     # sha256[:12] of tools/toy_struct.py at that commit
if PIN and not (R / "code").exists():     # once; written to a temp dir and renamed
    tmp = R / "code.tmp"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    arc = subprocess.run(["git", "-C", str(REPO), "archive", PIN, "tools"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
    (tmp / "COMMIT").write_text(PIN + "\n")
    tmp.rename(R / "code")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:12]
if PIN:
    assert sha(R / "code/tools/pretrain_range.py") == PIN_SHA and sha(R / "code/tools/toy_struct.py") == TOY_SHA
    assert (R / "code/COMMIT").read_text().strip() == PIN
else:   # before the commit: the working tree must still be the reviewed Rev B
    assert sha(REPO / "tools/pretrain_range.py") == PIN_SHA and sha(REPO / "tools/toy_struct.py") == TOY_SHA, \
        "working tree != the reviewed Rev B: update PIN_SHA / TOY_SHA only after a re-review"
    print("PIN not set: no code snapshot; job files rendered for review, --submit refused")
ALL6 = "numbers,months,weekdays,clock,s4,dim"
# USER DECISION 2026-10-17 (A2; audit finding 2): each = lambda 5 on every active prior (the best single-domain setup;
# the RB2 design in notes/next_phase_designs.json: "lam 5 per prior"); norm (lambda 5 split over the active priors,
# F70 '@norm') stays selectable here. The none arm (which alone decides the pilot gate) does not depend on it.
LAM_RULE = "each"
W = f"--schedule window --window 0.05,0.4 --site ctx --lam_rule {LAM_RULE}"
ARM = {"none": "--target none",
       "all6": f"--target helix3 {W} --prior_domains {ALL6}",
       # main-design arms (sketch; rendered and asserted by --print-main, never generated here)
       "all6-shuf": f"--target helix3_shuf {W} --prior_domains {ALL6} --prior_shuf {ALL6}",
       "only-numbers": f"--target helix3 {W} --prior_domains numbers",
       **{f"only-{d}": f"--target none {W} --prior_domains {d}" for d in ("months", "weekdays", "clock", "s4", "dim")}}
WANT = {"none": {"--target": "none"},
        "all6": {"--target": "helix3", "--prior_domains": ALL6},
        "all6-shuf": {"--target": "helix3_shuf", "--prior_domains": ALL6, "--prior_shuf": ALL6},
        "only-numbers": {"--target": "helix3", "--prior_domains": "numbers"},
        **{f"only-{d}": {"--target": "none", "--prior_domains": d} for d in ("months", "weekdays", "clock", "s4", "dim")}}
WIN = {"--schedule": "window", "--window": "0.05,0.4", "--site": "ctx", "--lam_rule": LAM_RULE}
PILOT = [(k, a, s) for k in (1, 2) for a in ("none", "all6") for s in (0, 1)]
MAIN_ARMS = ["none", "all6", "only-numbers", "only-months", "only-weekdays", "only-clock", "only-s4", "only-dim", "all6-shuf"]
HOURS = "03:00:00"   # 53M numbers-only 0.95-0.97 L40S-h; + domain / held-out-format evals, gradient logging (~20%,
                     # FLOP estimate, A2) = ~1.2 L40S-h; 2.5x


def flags(args):
    """{--flag: value or True} of an ARGS string; asserts that no flag is given twice."""
    t, d = args.split(), {}
    for i, w in enumerate(t):
        if w.startswith("--"):
            assert w not in d, f"{w} given twice: {args}"
            d[w] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True
    return d


def render(phase, k, a, s):
    tag = f"rb6{phase}-53m-2pct-k{k}-{a}"
    jt = f"{tag}-s{s}"
    extra = f"{ARM[a]} --arith_lines 1 --domains {ALL6} --extra_lines {k} --lr 1e-3"
    args = (f"--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {extra} --seed {s} "
            f"--tag _{tag} --save_model --ckpt_every 2000")
    f = flags(args)
    want = {"--lr": "1e-3", "--arith_lines": "1", "--domains": ALL6, "--extra_lines": str(k), "--seed": str(s),
            "--tag": f"_{tag}", "--out": f"{R}/out/{tag}", "--save_model": True, "--ckpt_every": "2000",
            "--tokenization": "whole", "--data": f"{B}/pretrain_data", "--tokenizer": str(TOK),
            # the main recipe's defaults stay defaults (53M, 3e8 tokens, bs 32, lambda 5, layers 2,4): never overridden
            "--lam": None, "--geo_layers": None, "--bs": None, "--tokens": None, "--seq": None, "--d_model": None,
            "--n_layers": None, "--gate_start": None, "--gate_hyst": None, "--untie": None, "--cyc_held": None,
            "--cyc_heldE": None, "--s4_train": None, "--dim_train": None, "--dom_eval_rows": None, "--dom_seen": None,
            **WANT[a], **({} if a == "none" else WIN)}
    for kk, v in want.items():
        assert f.get(kk) == v, (jt, kk, f.get(kk), v)
    assert set(f) <= set(want) | {"--prior_domains", "--prior_shuf"}, (jt, sorted(set(f) - set(want)))
    if a == "none":
        assert not ({"--schedule", "--window", "--site", "--prior_domains", "--prior_shuf", "--lam_rule"} & set(f)), (jt, sorted(f))
    return tag, jt, args


if "--print-main" in sys.argv:   # the main design (sketch, for approval): arms x seeds 0-3 at the selected density
    k = int(sys.argv[sys.argv.index("--print-main") + 1])
    for a in MAIN_ARMS:
        for s in range(4):
            tag, jt, args = render("m", k, a, s)
            reuse = a in ("none", "all6") and s in (0, 1)   # identical config: the pilot run is reused, not re-run
            print(jt, "|", f"REUSE pilot rb6p-53m-2pct-k{k}-{a}-s{s}" if reuse else ARM[a])
    print(len(MAIN_ARMS) * 4 - 4, "new main runs + 4 reused pilot runs (not generated)")
    sys.exit(0)

n, tags = 0, set()
R.mkdir(parents=True, exist_ok=True)
for k, a, s in PILOT:
    tag, jt, args = render("p", k, a, s)
    assert jt not in tags; tags.add(jt)
    job = R / f"job_{jt}.sbatch"
    job.write_text(f"""#!/bin/bash
#SBATCH --job-name=llmadd_rb6-{jt}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={HOURS}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# rainbow6 pilot (six manifolds, 2% numbers + {k} domain line(s) per row); code = llmadd-burst commit in code/COMMIT;
# intent spec runs/rb6/AUDIT.md
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/rb6/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
# the snapshot must still be the reviewed Rev B (code_sha of every result) and its toy_struct; exit before GPU work
[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{PIN_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
[ "$(sha256sum toy_struct.py | cut -c1-12)" = "{TOY_SHA}" ] || {{ echo "toy_struct sha mismatch: exiting"; exit 3; }}
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
ARGS="{args}"
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS --dry_inputs
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_range.py $ARGS
""")
    n += 1
    flag = R / f"SUBMITTED_{jt}"
    # --arms none[,all6]: submit only these arms (the none arm alone decides the gate; all6 can wait for LAM_RULE)
    arms = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else ["none", "all6"]
    assert set(arms) <= {"none", "all6"}, arms
    if "--submit" in sys.argv and not flag.exists() and a in arms:
        assert PIN and (R / "code").exists(), "commit Rev B and set PIN first"
        o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
        if o.returncode == 0:
            flag.write_text(o.stdout)
        else:
            print(jt, o.stderr.strip())
print(n, "jobs")
