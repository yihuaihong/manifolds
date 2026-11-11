"""Torch jobs of runs/ptdom, Amendment A3 + A4 (2026-11-04; user decision 2026-11-04; A4 = the coordinator's decisions on
the A3 audit, before any submission, AUDIT.md "Amendment A4": trainer Rev G (the full-span k-way read-out, the tie-aware
k-way known answer), every batch on B200 (16 singleton slot chains, 4 per batch), no saved model in stage 1, none's lr
extensions at both edges (lower 1e-3 then 5e-4, upper 1.6e-2), the extensions' seeds 4-7 run with them, per-batch HOLD_<X>
files, one GPU pilot job per batch (--pilot)). Screen 24 candidate domains in 53M
pretraining, each prior ALONE vs none, in four self-contained batches (identical data for every arm of a batch and seed,
the batch's own none arm, ONE GPU type per batch), two stages; the six that pass later join numbers and dimensions in a
six-manifold run. Intent spec and PRE-REGISTERED screen / verdict: runs/ptdom/AUDIT.md "Amendment A3" (written before any
run; supersedes the A1 / A2 lr design; the A2 per-cell auxiliary-loss gate, the shortcut baselines, the shared-site
statistics and the Tier L tags stay).
Batches (--domains numbers,<six>; K = 2 domain lines per row cycling the six, the numbers line (2%) first after the text):
  A   months, stars, landmarks, languages, colours, cities                (the A2 build's data and templates)   B200 (A4)
  B1  countries, animals, country_size, animal_size, people, units         (PAIR2 comparisons)                   B200
  B2  zodiac, compass, planets, elements, prefixes, ranks                  (OFFSET k-step: zodiac / planets /    B200
                                                                            elements; PAIR2: compass / prefixes / ranks)
  B3  chinese_zodiac, kinship, life_stages, geo_periods, greek, scalar_adj (OFFSET: chinese_zodiac / geo_periods / B200 (A4)
                                                                            greek; PAIR2: kinship / life_stages / scalar_adj)
  (letters, years, number_words are not used: harmful in fine-tuning, F81)
Arms per batch: none; only-<d> for its six domains (--target none, window (0.05, 0.4), ctx, lambda 1, lam_rule each).
Stages (AUDIT.md A3; select_ptdom.py --batch X decides from the finished runs):
  1   seeds 0-3: prior arms at lr 1e-3 / 2e-3, none at 2e-3 / 4e-3 / 8e-3 (60 runs per batch; A4: no saved model)
  2   the advancing domains (stage-1 screen): seeds 4-7 at the picks (prior and none), and the extension lr where a
      stage-1 area argmax is at a grid edge (none's lower edge -> 1e-3, A4 upper edge -> 1.6e-2; prior's upper edge ->
      4e-3, lower edge -> 5e-4), A4: at seeds 0-7 (the selection reads seeds 0-3; seeds 4-7 there skip a 2b round)
  2b  (A4) none's second lower extension 5e-4 at seeds 0-7 where 1e-3 is still none's argmax after stage 2, and seeds 4-7
      at a pick that the re-selection moved to an lr without them
  2c  (A4) seeds 4-7 at a pick that the re-selection after 2b moved to an lr without them (no further extension)
  A5 (AUDIT.md "Amendment A5"): stages 2b / 2c also carry the fixB cells F = none at a prior's final pick lr (1e-3 / 5e-4)
      outside the lr range none ran for that domain, at seeds 0-7; CHECK_PY's double-path tolerances = ptheld Amendment DP's
      (eval 0 <= 0.05, later <= 0.5 nat per scored token), a run failing only them is valid data (flagged)
Code: runs/ptdom/code = git archive of tools/ at the COMMITTED HEAD of branch ptdom (worktree /scratch/yh6210/llmadd-burst-ptdom,
not pushed; Rev F); code/COMMIT records it and the sha256[:12] of every file the jobs use; the generator refuses a snapshot
that differs from HEAD or from WANT_SHA. Every sbatch: HOLD kill switch, the snapshot shas (exit 3), a GPU-type guard (the
batch's), the data sha256, --dry_inputs with the input checks + the DRY row hash of the batch and seed (CPU reference,
exit 3 before training), training (--requeue resumes from the checkpoint: L40S every 2000 steps, B200 every 500 steps,
pt300x's B200 protocol: preemptible, singleton slot chains), offline_lm.py, CHECK_PY with the run's intent -> check.json
(exit 4 on a failed check), gpu.csv. Never submits without --submit.
    python make_jobs_ptdom.py --plan
    python make_jobs_ptdom.py --batch X --stage 1 [--print] [--test-only N] [--submit]
    python make_jobs_ptdom.py --batch X --stage 2 --picks runs/ptdom/picks_<X>_after1.json [--print] [--test-only N] [--submit]
    python make_jobs_ptdom.py --batch X --stage 2b --picks runs/ptdom/picks_<X>_after2.json [--print] [--test-only N] [--submit]
    python make_jobs_ptdom.py --batch X --stage 2c --picks runs/ptdom/picks_<X>_after3.json [--print] [--test-only N] [--submit]
    python make_jobs_ptdom.py --batch X --pilot [--print] [--test-only N] [--submit]    # A4: the GPU pilot of batch X
    python make_jobs_ptdom.py --batch X --stage 1|2|2b [--picks ...] --rerun [--print] [--test-only N] [--submit]   # re-runs (A2)
    python make_jobs_ptdom.py --snapshot [--resnapshot]
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/scratch/yh6210/llmadd-burst-ptdom"); BRANCH = "ptdom"
R = B / "runs" / "ptdom"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
DATA = B / "pretrain_data"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
# the trainer (A4: Rev G, commit 11f61b9), what it imports, offline_lm, and the generator module whose constants / flags() are reused
WANT_SHA = {"pretrain_range.py": "d87a1502cee8", "toy_struct.py": "d182ce548bcc", "pretrain_helix.py": "25d5a35e6f17",
            "numlib.py": "d09bb47b04c2", "digit_split_data.py": "742b922c8125", "run_guard.py": "b944dc859c7a",
            "offline_lm.py": "226f4cac70ea", "ft36.py": "2f84907b4725", "manifold_domains.py": "0607a9d693da",
            "manifold_atlas.py": "43efc295ecc0", "make_rb6_tasks.py": "7f689a95f000", "readout_rules.py": "e91ee8a8a846"}
DEP = ("toy_struct.py", "ft36.py", "manifold_domains.py", "manifold_atlas.py")   # the run JSON's dep_sha (Rev E / F)
DATA_SHA = ("4de2ed19093c37b302b6a1ced56136e6a8b805e6d6b48057a77316811704ff11",   # make_rb6_tasks.DATA (train.bin, val.bin)
            "d6f744a99b988ed6928fb588356cc8fe3d61bcf348b8d06fc0eece25c624a4c8")
# ---------------------------------------------------------------- design (AUDIT.md Amendment A3)
BATCHES = {"A": ("months", "stars", "landmarks", "languages", "colours", "cities"),
           "B1": ("countries", "animals", "country_size", "animal_size", "people", "units"),
           "B2": ("zodiac", "compass", "planets", "elements", "prefixes", "ranks"),
           "B3": ("chinese_zodiac", "kinship", "life_stages", "geo_periods", "greek", "scalar_adj")}
HWOF = {"A": "b200", "B1": "b200", "B2": "b200", "B3": "b200"}   # one GPU type per batch (the user); A4: every batch on B200
KSTEP = ("months", "zodiac", "planets", "elements", "chinese_zodiac", "geo_periods", "greek")   # the k-way read-out domains
NEVER = ("letters", "years", "number_words")   # harmful in fine-tuning (F81): never a candidate
ALL24 = tuple(d for b in BATCHES.values() for d in b)
assert len(set(ALL24)) == 24 and not set(NEVER) & set(ALL24) and set(KSTEP) <= set(ALL24)
COMP = tuple(d for d in ALL24 if d not in KSTEP)   # the comparison domains (2-way read-out; Tier L)
# GPU types. L40S: the A2 header (account 625, l40s_public, gpu48, nice 20); plan 1.3 L40S-h per run (A2's basis: L40S /
# A100-40GB = 1.31 at 53M 2%, rb6 53M K 2 = 3,075 s on A100 -> 1.12 h on L40S, ptdom's eval rows ~1.1x rb6's + the other-
# option forwards). B200: pt300x's 294M protocol (account 520 priority, --constraint b200, preemption=yes, singleton slots,
# ckpt every 500 steps, --open-mode=append, 3 h); plan 0.6 B200-h per run: pt300x 294M on B200 = 4,007 s of training
# (13 runs, runs/ptdom/tests/a3_b200_timing.py) vs 294M on L40S 14,645 s (pttpp: 0.7551 s/step + 10.92 s/eval) -> B200 /
# L40S = 0.274 at 294M -> 0.33 h of training at 53M + 0.1 h dry / offline / checks = 0.43 h; planned 0.6 because a 53M
# step is launch-bound on a B200 (the size ratio of L40S need not carry over); the first finished job calibrates.
HW = {"l40s": dict(gpu="L40S", plan_h=1.3, time="03:00:00", ckpt=2000, cpus=8, mem="64G"),
      "b200": dict(gpu="B200", plan_h=0.6, time="03:00:00", ckpt=500, cpus=16, mem="128G")}
SLOTS = 4                        # A4: B200 singleton slots per batch, 16 chains in all: llmadd_ptdom-b200-slot<0..3> (A),
SLOT0 = {"A": 0, "B1": 4, "B2": 8, "B3": 12}   # <4..7> (B1), <8..11> (B2), <12..15> (B3)
K = 2
GRID1 = {"none": ("2e-3", "4e-3", "8e-3"), "prior": ("1e-3", "2e-3")}   # stage 1 (A3)
# stage-2 extensions: the first step beyond a stage-1 grid edge (A3; A4 adds none's upper edge), and A4's second step (none's
# lower edge only: 5e-4 when 1e-3 is still the argmax after stage 2); none's pick or argmax at an outermost lr it ran
# (5e-4 / 1.6e-2) after the allowed steps -> NOT FINAL
EXT_LR = {("none", "lower"): "1e-3", ("none", "upper"): "1.6e-2", ("prior", "upper"): "4e-3", ("prior", "lower"): "5e-4"}
EXT2 = {("none", "lower"): "5e-4"}
OUTER = {"none": ("5e-4", "1.6e-2")}
# A5 fixB (AUDIT.md A5.2; audit_a4 important 2): the lrs a prior's final pick can take outside none's stage-1 grid range (the
# prior's grid and its one-step extensions below 2e-3 / above 8e-3): none is run there at seeds 0-7 when none has not run that
# lr for the domain (stage 2b / 2c 'F' cells), and none is re-selected including it before the verdict
FIXB_LR = tuple(lr for lr in GRID1["prior"] + (EXT_LR[("prior", "lower")], EXT_LR[("prior", "upper")])
                if float(lr) < float(GRID1["none"][0]) or float(lr) > float(GRID1["none"][-1]))
assert FIXB_LR == ("1e-3", "5e-4"), FIXB_LR
STAGES = ("1", "2", "2b", "2c")   # stage written by picks after<k>: after1 -> 2, after2 -> 2b, after3 -> 2c
LADDER = ("6.25e-5", "1.25e-4", "2.5e-4", "5e-4", "1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2", "3.2e-2", "6.4e-2")
SEL, CONF = (0, 1, 2, 3), (4, 5, 6, 7)
W = "--schedule window --window 0.05,0.4 --site ctx --lam_rule each"
# the primary read-out per domain (metrics key; area = the selection and decision metric): k-step = the held-out k-way
# choice (months: tier S); comparisons = every held-out question (both orders), the 2-way choice; the seen set of the same
# scoring (the memorisation check)
# A4 (Rev G): k-step = the FULL-SPAN k-way choice (<d>_readout_kspan: every item's ' <item>\n' summed log-prob, ft36 / F81);
# the first-token k-way choice (<d>_readout_kway) is the secondary line (READ1)
READ = {d: (f"{d}_readout_kspan" if d in KSTEP else f"{d}_readout") for d in ALL24}
SEEN = {d: (f"{d}_seen_kspan" if d in KSTEP else f"{d}_seen_2way") for d in ALL24}
READ1 = {d: f"{d}_readout_kway" for d in KSTEP}
GEO_TOL, CKA_RISE = -1e-3, 0.1   # = make_rb6m_tasks (float CKA > 1 at a perfect fit; the CKA-rise sanity bar)
# A5 (AUDIT.md A5.1 = runs/ptheld/AUDIT.md Amendment DP's rule; audit_a4 blocking 1): the GPU-numerics tolerances of the two
# scoring double paths (2-way: lp_dp_maxdiff_*; k-span: lpk_dp_maxdiff_* and lpk_own_maxdiff_*): eval 0 (near-uniform
# model, where an index error costs ~11 nats) <= DP_TOL0; every later eval <= DP_PER_TOK nat per scored token, the token
# count bounded by the domain's LONGEST span ' <item>\n' (SPAN_TOK, incl. the newline; the trainer logs only the maximum
# over a set's first 4 rows, so the per-row count is not in the run JSON): DP_PER_TOK x SPAN_TOK[d] per domain.
# SPAN_TOK = tests/a5/span_tok.py (the snapshot's tables and the jobs' tokenizer, as the trainer builds the spans; = every
# k-step domain's logged kspan_tokens and every domain's item_tokens histogram in the four GPU pilots).
DP_TOL0, DP_PER_TOK = 0.05, 0.5
SPAN_TOK = {"months": 2, "stars": 5, "landmarks": 7, "languages": 4, "colours": 3, "cities": 5, "countries": 6, "animals": 5,
            "country_size": 6, "animal_size": 5, "people": 8, "units": 3, "zodiac": 4, "compass": 2, "planets": 3, "elements": 5,
            "prefixes": 3, "ranks": 3, "chinese_zodiac": 3, "kinship": 4, "life_stages": 2, "geo_periods": 4, "greek": 4,
            "scalar_adj": 4}
assert set(SPAN_TOK) == set(ALL24) and min(SPAN_TOK.values()) >= 2
DOM_KEYS = {"months": ("n_train", "n_held", "n_heldE", "n_readout", "n_readoutE", "readout", "n_items"),
            "comp": ("n_train", "n_held", "n_readout", "readout", "n_items", "n_comparisons", "n_keys", "n_held_keys",
                     "split_seed", "target", "raw_gram"),
            "kstep": ("n_train", "n_held", "n_readout", "readout", "n_items", "n_questions", "n_keys", "n_held_keys",
                      "split_seed", "target", "cyclic")}


def dom_kind(d):
    return "months" if d == "months" else "kstep" if d in KSTEP else "comp"


def arms(batch):
    return ("none",) + tuple(f"only-{d}" for d in BATCHES[batch])


def grid(arm):
    return GRID1["none" if arm == "none" else "prior"]


def refs_path(batch):
    """CPU references per batch (tests/refs_ptdom.py for A (the A2 build's file, valid under Rev F: A3 regression);
    tests/a3/refs_a3.py for B1-B3)."""
    return R / ("ptdom_refs.json" if batch == "A" else f"ptdom_refs_{batch}.json")


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def snapshot():
    """runs/ptdom/code = git archive of tools/ at the committed HEAD of the ptdom worktree (once); afterwards every file
    = COMMIT's sha = the file of that commit, and = WANT_SHA. -> the snapshot's commit. --resnapshot replaces it only while
    no job of the batch was submitted (no SUBMITTED_* flag)."""
    R.mkdir(parents=True, exist_ok=True)
    if "--resnapshot" in sys.argv and (R / "code").exists():
        assert not list(R.glob("SUBMITTED_*")), "a job was submitted: the snapshot is frozen"
        shutil.rmtree(R / "code")
    if not (R / "code").exists():
        head = git("rev-parse", "--short", "HEAD").decode().strip()
        assert git("rev-parse", "--abbrev-ref", "HEAD").decode().strip() == BRANCH, "the worktree must be on " + BRANCH
        assert subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", "tools/"]).returncode == 0, \
            "tools/ has uncommitted changes in the worktree: commit them (the snapshot is the committed code)"
        tmp = R / "code.tmp"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=git("archive", head, "tools"), check=True)
        (tmp / "COMMIT").write_text(f"{head} ({BRANCH}, committed, not pushed; git archive of tools/); sha256[:12]: "
                                    + ", ".join(f"{f} {sha(tmp / 'tools' / f)}" for f in WANT_SHA) + "\n")
        tmp.rename(R / "code")
    cm = (R / "code" / "COMMIT").read_text()
    commit = cm.split()[0]
    for f, v in WANT_SHA.items():
        s_ = sha(R / "code" / "tools" / f)
        assert f"{f} {s_}" in cm and s_ == v, (f, s_, v)
        assert hashlib.sha256(git("show", f"{commit}:tools/{f}")).hexdigest()[:12] == s_, (f, commit)
    return commit


def refs(batch):
    """{seed: {dry: [sha, n], lm0, dsha0, dom, table_sha}} of a batch (empty if absent)."""
    p = refs_path(batch)
    if not p.exists():
        return {}
    j = json.loads(p.read_text())
    assert batch == "A" or j.get("batch") == batch, (p, j.get("batch"))
    return {int(k): v for k, v in j["seeds"].items()}


def domains_arg(batch):
    return "numbers," + ",".join(BATCHES[batch])


def tag_of(batch, arm, lr):
    return f"ptdom-{batch}-{arm}-lr{lr}"


def arm_flags(arm):
    return "--target none" if arm == "none" else f"--target none {W} --prior_domains {arm[5:]} --lam 1"


def want_flags(arm):
    """The arm's flags as flags() parses them (None = absent)."""
    if arm == "none":
        return {"--target": "none", "--schedule": None, "--window": None, "--site": None, "--lam_rule": None,
                "--prior_domains": None, "--prior_shuf": None, "--lam": None}
    return {"--target": "none", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx", "--lam_rule": "each",
            "--prior_domains": arm[5:], "--prior_shuf": None, "--lam": "1"}


# A4 GPU pilot (one short job per batch, B200; not part of any result): the batch's own data, domains, eval sets and a
# prior arm (the gradient split is logged on the window's evals) on a compressed schedule: 400 steps (--tokens 32 x 512 x
# 400), evals every 50 steps (9 evals, 3 on the window 20..159), a checkpoint every 100 steps, --save_model (the stage-2
# path: offline_lm runs); output runs/ptdom/pilot/
PILOT = {"A": ("only-stars", "1e-3"), "B1": ("only-animals", "1e-3"), "B2": ("only-elements", "1e-3"), "B3": ("only-chinese_zodiac", "1e-3")}
PILOT_STEPS, PILOT_EVAL, PILOT_CKPT = 400, 50, 100


def saves(stage, pilot=False):
    """A4: stage 1 saves no model (disk; models are saved in stage 2 and later only); the pilot saves one."""
    return pilot or stage != "1"


def args_of(batch, arm, lr, seed, out, save=True, pilot=False):
    tag = tag_of(batch, arm, lr)
    ck = PILOT_CKPT if pilot else HW[HWOF[batch]]['ckpt']
    return (f"--data {DATA} --tokenizer {TOK} --tokenization whole --out {out}/{tag} {arm_flags(arm)} --arith_lines 1 "
            f"--domains {domains_arg(batch)} --extra_lines {K} --lr {lr} --seed {seed} --tag _{tag}{' --save_model' if save else ''} "
            f"--ckpt_every {ck}" + (f" --tokens {32 * 512 * PILOT_STEPS} --eval_every {PILOT_EVAL}" if pilot else ""))


def intent(P, batch, arm, lr, seed, ref, save=True, pilot=False):
    """Everything CHECK_PY asserts on the run JSON (also used by the CPU tests of CHECK_PY). A4: save (stage 1: no saved
    model, no offline LM) and the pilot's compressed schedule (steps, evals, window events)."""
    doms = BATCHES[batch]
    pri = () if arm == "none" else (arm[5:],)
    assert set(pri) <= set(doms)
    steps, ev_ = (PILOT_STEPS, PILOT_EVAL) if pilot else (P.STEPS, P.EVAL_EVERY)
    events, on_frac = P.window_events(steps) if pri else ([], 0.0)
    if pri:
        assert events == ([[20, 1], [160, 0]] if pilot else [[916, 1], [7324, 0]]), events
    lam_rule = "each" if pri else "norm"
    hw = HW[HWOF[batch]]
    return {"batch": batch, "code_sha": WANT_SHA["pretrain_range.py"], "dep_sha": {f: WANT_SHA[f] for f in DEP}, "steps": steps,
            "n_params": P.N_PARAMS, "grad_ckpt_on": False, "geo_layers": [2, 4], "read_layers": [2, 4], "lm_chunk": 0,
            "eval_every_steps": ev_, "lam_rule": lam_rule, "save_model": bool(save), "pilot": bool(pilot),
            "lam_domains": {d: (1.0 if d in pri else 0.0) for d in ("numbers",) + doms}, "K": K, "ext": list(doms),
            "comp": [d for d in doms if d in COMP], "kstep": [d for d in doms if d in KSTEP],
            "tables": [d for d in doms if d != "months"], "active": list(pri), "geo_tol": GEO_TOL,
            "evals": len(range(0, steps, ev_)) + ((steps - 1) % ev_ != 0), "events": events,
            "on_frac": on_frac, "lm0_ref": ref.get("lm0"), "data_sha0": ref.get("dsha0"), "dom": ref.get("dom"),
            "table_sha": ref.get("table_sha"), "n_fmt": {"months": 4} if "months" in doms else {}, "cka_rise": CKA_RISE,
            "dp_tol0": DP_TOL0, "dp_tol": {d: DP_PER_TOK * SPAN_TOK[d] for d in doms},   # A5.1
            "numbers_share": [0.019, 0.024], "read": {d: READ[d] for d in doms}, "seen": {d: SEEN[d] for d in doms},
            "gpu": hw["gpu"],
            "args": {"data": None, "tokenizer": str(TOK), "out": None, "tokenization": "whole", "target": "none",
                     "site": "ctx", "schedule": "window" if pri else "whole", "window": "0.05,0.4", "arith_lines": 1,
                     "lr": float(lr), "lam": 1.0 if pri else 5.0, "tokens": float(32 * 512 * PILOT_STEPS) if pilot else 3e8,
                     "bs": P.BS, "seq": P.SEQ, "d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536, "seed": seed,
                     "tag": f"_{tag_of(batch, arm, lr)}", "save_model": bool(save), "ckpt_every": PILOT_CKPT if pilot else hw["ckpt"],
                     "geo_layers": None, "eval_every": ev_, "n_eval": 150,
                     "grad_ckpt": "auto", "lm_chunk_tokens": -1, "untie": False, "gate_thr": 0.5, "gate_start": 0.0,
                     "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256, "stop_after": -1, "dry_inputs": False,
                     "domains": domains_arg(batch), "extra_lines": K, "prior_domains": ",".join(pri), "prior_shuf": "",
                     "lam_rule": lam_rule, "cyc_held": 0.4, "cyc_heldE": 0.5, "s4_train": 0.6, "dim_train": 0.5,
                     "dom_eval_rows": 200, "dom_seen": 150}}


# The intent checks of one run (the rb6 CHECK_PY of make_rb6m_tasks.py generalised to the ptdom batches, + Rev E / F): run
# python - <run json> <offline_lm json> <gpu.csv> <COMMIT> <data dir> <check.json> '<intent json>' "$ARGS"
CHECK_PY = r'''import csv, json, math, statistics, sys
from pathlib import Path
run, offl, gpu, pinned, data, out_json = sys.argv[1:7]
I = json.loads(sys.argv[7])
JARGS = sys.argv[8].split()   # A1: the job's own ARGS (the trainer's command line): the intended lambda comes from it
EXT, COMP, KST = tuple(I["ext"]), tuple(I["comp"]), tuple(I["kstep"])
ALLD = ("numbers",) + EXT
ACT, TOL = I["active"], I["geo_tol"]
GEOK = {f"geo_{d}" for d in EXT}   # A3: the prior values' keys exactly (a prefix test 'geo_' would hit geo_periods' read-outs)
mean = statistics.fmean
checks = []
def chk(name, ok, value):
    checks.append({"check": name, "ok": bool(ok), "value": value})
r = json.loads(Path(run).read_text()); c = r["curve"]; a = r["args"]
want = dict(I["args"], data=data, out=str(Path(run).parent))
diff = {k: [a.get(k), v] for k, v in want.items() if a.get(k) != v}
extra = sorted(set(a) - set(want))
chk(f"run args = intent ({len(want)} fields, no other field)", not diff and not extra, {"diff": diff, "extra": extra})
# A1: the job's ARGS (the command line the trainer ran with) parsed here: every flag = the run's args (a second path to the
# intent's args table), and the INTENDED lambda per domain from it (lam_rule each: lambda_d = --lam for every
# --prior_domains entry, 0 for every other domain; no prior flags (none) -> 0 everywhere)
jf, i_ = {}, 0
while i_ < len(JARGS):
    k_ = JARGS[i_]
    if not k_.startswith("--"):
        jf.setdefault("__bad__", []).append(k_); i_ += 1; continue
    if i_ + 1 < len(JARGS) and not JARGS[i_ + 1].startswith("--"):
        jf[k_[2:]] = JARGS[i_ + 1]; i_ += 2
    else:
        jf[k_[2:]] = True; i_ += 1
def same_(v, x):
    if v is True or isinstance(x, bool):
        return v is True and x is True
    if isinstance(x, (int, float)):
        try:
            return float(v) == float(x)
        except ValueError:
            return False
    return v == str(x)
jd = {k: [v, a.get(k)] for k, v in jf.items() if not same_(v, a.get(k))}
chk(f"the job's ARGS ({len(jf)} flags) = the run's args", bool(jf) and not jd, jd)
jpri = [x for x in str(jf.get("prior_domains", "")).split(",") if x]
want_lam = {d: (float(jf["lam"]) if d in jpri and jf.get("lam_rule") == "each" and "lam" in jf else 0.0) for d in ALLD}
chk("lambda per domain from the job's ARGS " + json.dumps(want_lam) + " = the logged lam_domains and domains.<d>.lam",
    r.get("lam_domains") == want_lam and all(r["domains"][d].get("lam") == want_lam[d] for d in ALLD) and set(jpri) <= set(ALLD),
    [r.get("lam_domains"), {d: r["domains"][d].get("lam") for d in ALLD}])
chk("code_sha " + I["code_sha"], r.get("code_sha") == I["code_sha"], r.get("code_sha"))
chk("dep_sha " + json.dumps(I["dep_sha"]), r.get("dep_sha") == I["dep_sha"], r.get("dep_sha"))
keys = ("steps", "n_params", "grad_ckpt_on", "geo_layers", "read_layers", "lm_chunk", "eval_every_steps", "lam_rule")
got = {k: r.get(k) for k in keys}
chk("run facts = intent " + json.dumps({k: I[k] for k in keys}), all(got[k] == I[k] for k in keys), got)
chk("lambda per domain = intent " + json.dumps(I["lam_domains"]), r.get("lam_domains") == I["lam_domains"], r.get("lam_domains"))
dp = {d: [r["domains"][d].get(k) for k in ("prior", "shuf", "lam")] for d in ALLD}
dw = {d: [d in ACT, False, I["lam_domains"][d]] for d in ALLD}
chk("domains.<d> prior / shuf / lam = intent (active " + (",".join(ACT) or "none") + "; numbers target none)",
    dp == dw and r["domains"]["numbers"].get("target") == "none" and set(r["domains"]) == set(ALLD), dp)
st = [e["step"] for e in c]
chk(f"finished: {I['evals']} evals, increasing, the last at step {I['steps'] - 1}",
    len(c) == I["evals"] and st == sorted(set(st)) and st[-1] == I["steps"] - 1, [len(c), st[-1]])
# a diverged run is data, not a failed check: onset = the first eval with a non-finite LM / val / prior value
fin = lambda v: v is None or math.isfinite(v)
bad = [e["step"] for e in c if not (fin(e["lm"]) and fin(e["val_loss"]) and fin(e["geo"]) and fin(e.get("grad_prior"))
                                     and all(fin(e.get(f"geo_{d}")) for d in EXT))]
diverged = bool(bad)
pre = [e for e in c if not bad or e["step"] < bad[0]]
chk(f"data_sha after step 0 = the CPU reference {I['data_sha0']} (the same data in every arm of the batch and seed)",
    c[0].get("data_sha") == I["data_sha0"], c[0].get("data_sha"))
chk("step-0 lm within 1e-3 of the CPU reference (same init and first batch; no prior at step 0)",
    abs(c[0]["lm"] - I["lm0_ref"]) <= 1e-3, c[0]["lm"] - I["lm0_ref"])
chk("the numbers line is the first line after the text in every row", abs(c[-1]["first_line_frac"]["numbers"] - 1) < 1e-12,
    c[-1]["first_line_frac"])
ts = c[-1]["token_share"]["numbers"]
chk(f"numbers token share in {I['numbers_share']} (the 2% setting)", I["numbers_share"][0] <= ts <= I["numbers_share"][1], ts)
sl = c[-1]["slot_frac"]
chk(f"domain lines per slot after the numbers line: K = {I['K']} slots, every slot within 1/K +- 0.05 of every domain's lines",
    set(sl) == set(EXT) and all(len(v) == I["K"] and all(abs(x - 1 / I["K"]) <= 0.05 for x in v) for v in sl.values()), sl)
dm = {d: {k: r["domains"][d].get(k) for k in I["dom"][d]} for d in EXT}
chk("domain splits = the CPU reference " + json.dumps(I["dom"]), dm == I["dom"], dm)
tsh = {d: r["domains"][d].get("table", {}).get("sha") for d in I["tables"]}
chk("domain tables (items, target, questions) = the reference " + json.dumps(I["table_sha"]), tsh == I["table_sha"], tsh)
nf = {d: len(r["domains"][d].get("formats", {})) for d in I["n_fmt"]}
fk = [f"{d}_p{t}_{s}" for d in I["n_fmt"] for t in range(1, I["n_fmt"][d] + 1) for s in ("held", "seen")]
chk(f"held-out-format templates {I['n_fmt']}, scored at every eval ({len(fk)} sets)",
    nf == I["n_fmt"] and all(e.get(k) is not None for e in c for k in fk), nf)
# Rev E / F / G: the choice read-outs at every eval (the hard subset only where it is not empty) and the gradient split;
# the double paths of the 2-way and the k-span scoring
rk = ([f"{d}_held_2way" for d in COMP] + [f"{d}_seen_2way" for d in COMP]
      + [f"{d}_held_2way_hard" for d in COMP if r["domains"][d].get("n_hard")]
      + [f"{d}_{s_}_{w_}" for d in KST for s_ in ("held", "seen") for w_ in ("kway", "kspan")]
      + (["months_heldE_kway", "months_heldE_kspan"] if "months" in KST else []))
chk(f"choice read-outs at every eval ({len(rk)} sets: 2-way held / seen / heuristic-hard (non-empty) of the comparisons, "
    f"k-way full-span (Rev G, the primary) and first-token held / seen of the k-step domains)",
    all(isinstance(e.get(k), float) for e in c for k in rk), [k for k in rk if any(not isinstance(e.get(k), float) for e in c)])
# The double paths of the 2-way and the k-span scoring (Rev E / G): 2-way = the span log-prob vs the direct per-position path on
# the first 4 rows of every set; k-span = the same on the first 4 candidate rows, and the answer's own candidate row (= the eval
# row) vs the eval forward's span log-prob; every set of every eval before a divergence. A5 (AUDIT.md A5.1; audit_a4 blocking
# 1): the fields are one check; the two tolerances (ptheld Amendment DP's rule: eval 0 <= dp_tol0, every later eval <= 0.5 nat
# per scored token = dp_tol[d]) are the 'GPU-numerics tolerance' checks: a run whose only failed checks are these (and / or the
# CKA-rise bar) is VALID data, flagged (readout_ptdom.load); A4's '< 0.25 nats at every eval' is superseded.
num_ = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
dom_of = lambda n: max((d for d in EXT if n.startswith(d + "_")), key=len)   # '<d>_<set>' -> d (domain names contain '_')
dp2 = [(f"lp_dp_maxdiff_{d}_{s}", d) for d in COMP for s in ("held", "seen")]
kn_ = sorted(k[len("lpk_dp_maxdiff_"):] for k in c[0] if k.startswith("lpk_dp_maxdiff_")) if KST else []
kneed = {f"{d}_{s_}" for d in KST for s_ in ("held", "seen")}
dpk = [(f"{p_}_{n}", dom_of(n)) for n in kn_ for p_ in ("lpk_dp_maxdiff", "lpk_own_maxdiff")]
dmiss = sorted({k for e in pre for k, _ in dp2 + dpk if not num_(e.get(k))})
chk(f"double-path fields logged (finite) at every eval before a divergence: 2-way {len(dp2)} sets, k-span {len(dpk)} (dp and own row); "
    "every k-way domain's held / seen", bool(pre) and not dmiss and kneed <= set(kn_), [dmiss[:6], sorted(kneed - set(kn_))])
def dp_over(sets):   # A5.1: (key, step, value, tolerance) over the tolerance; the max at eval 0, later, and later / tolerance
    ov, m0, mL, rL = [], 0.0, 0.0, 0.0
    for e in pre:
        for k, d in sets:
            v = e.get(k)
            if not num_(v):
                continue   # the fields check above
            t_ = I["dp_tol0"] if e["step"] == 0 else I["dp_tol"][d]
            if e["step"] == 0:
                m0 = max(m0, v)
            else:
                mL, rL = max(mL, v), max(rL, v / t_)
            if v > t_:
                ov.append([k, e["step"], round(v, 4), t_])
    return {"n_over": len(ov), "over": ov[:6], "max_eval0": m0, "max_later": mL, "max_later_over_tol": rL}
tol_txt = "; ".join(f"{d} {I['dp_tol'][d]:g}" for d in EXT)
if COMP:
    o2_ = dp_over(dp2)
    chk(f"2-way scoring double path (GPU-numerics tolerance, A5.1): |span log-prob - the direct per-position path| (first 4 rows of "
        f"every set) <= {I['dp_tol0']} nats at eval 0 and <= 0.5 nat per scored token at every later eval ({tol_txt})",
        o2_["n_over"] == 0, o2_)
if KST:
    ok_ = dp_over(dpk)
    chk(f"k-span scoring double paths (GPU-numerics tolerance, A5.1): |candidate span log-prob - the direct per-position path| (first "
        f"4 rows) and |the answer's candidate row - the eval forward| <= {I['dp_tol0']} nats at eval 0 and <= 0.5 nat per scored "
        f"token at every later eval ({tol_txt})", ok_["n_over"] == 0, ok_)
kdp = max([e[k] for e in pre for k, _ in dpk if num_(e.get(k))], default=None)
kb = sum(v for e in c for k, v in e.items() if k.startswith("kway_em_bad_"))
kt = sum(v for e in c for k, v in e.items() if k.startswith("kway_em_tie_"))
chk("k-way known answer (Rev G, tie-aware): no exact-match hit whose first-token k-way choice is another item with a logit more "
    f"than one bf16 step away (ties within it: {kt}, logged)", kb == 0 and all(
        any(k.startswith("kway_em_bad_") for k in e) for e in c) if KST else kb == 0, [kb, kt])
gk_ = ("grad_T", "grad_lm_share", "grad_total", "clip_coef", "grad_T_reldiff")
chk("gradient split logged at every eval (" + ", ".join(gk_) + ")", all(e.get(k) is not None for e in c for k in gk_),
    [k for k in gk_ if any(e.get(k) is None for e in c)])
# A1 / A2: the decomposition double path on the on-window evals only (outside the window grad_T is grad_lm), the MAX; none
# has no window: every eval (grad_T = grad_lm vs the backward's own norm)
n_on_ = sum(1 for e in pre if e.get("prior_on"))
rd = [e["grad_T_reldiff"] for e in pre if e.get("grad_T_reldiff") is not None and (e.get("prior_on") or not I["events"])]
chk("decomposition double path: max |grad_T - grad_total| / grad_total < 0.05 over the "
    + ("on-window evals (logged at >= 90% of them)" if I["events"] else "evals (none: no window)"),
    bool(rd) and (not I["events"] or (n_on_ > 0 and len(rd) >= 0.9 * n_on_)) and max(rd) < 0.05,
    [statistics.median(rd) if rd else None, max(rd) if rd else None, len(rd), n_on_])
cka_true = None
if I["events"]:
    on_frac = c[-1]["prior_on_frac"]
    chk(f"prior_events == {I['events']}", r.get("prior_events") == I["events"], r.get("prior_events"))
    chk(f"prior_on_frac == {I['on_frac']:.6f}", abs(on_frac - I["on_frac"]) < 1e-9, on_frac)
    chk("prior_on_domains: the active prior (" + ",".join(ACT) + ") on exactly at the evals with prior_on, every other domain never",
        all(len(e["prior_on_domains"]) == len(ALLD) and all(e["prior_on_domains"][d] == (e["prior_on"] if d in ACT else 0) for d in ALLD)
            for e in c), {d: sum(e["prior_on_domains"][d] for e in c) for d in ALLD})
    chk("numbers prior off: never applied, no numbers loss term (geo == 0 at every eval), no weighted numbers gradient",
        c[-1]["prior_used_frac_numbers"] == 0 and all(e["geo"] == 0 for e in c) and all("gnorm_prior_numbers" not in e for e in c),
        c[-1]["prior_used_frac_numbers"])
    used = {d: c[-1][f"prior_used_frac_{d}"] for d in EXT}
    chk("the active prior applied on >= 95% of the on-steps; every inactive one never",
        all((u >= 0.95 * on_frac - 1e-12) if d in ACT else (u == 0) for d, u in used.items()), used)
    ng = {d: sum(e.get(f"geo_{d}") is not None for e in c) for d in EXT}
    chk("geo_<d> logged at >= 95% of the evals for the active prior, never for an inactive one",
        all((n >= 0.95 * len(c)) if d in ACT else (n == 0) for d, n in ng.items()), ng)
    on = [e for e in pre if e["prior_on"]]
    gk = {d: sum(f"gnorm_prior_{d}" in e for e in on) for d in ALLD}
    chk("weighted prior gradients at the on-evals from the active prior only (at >= 90% of the on-evals)",
        all((n >= 0.9 * len(on)) if d in ACT else (n == 0) for d, n in gk.items()) and all(
            not any(k.startswith("gnorm_prior_") for k in e) for e in c if not e["prior_on"]), gk)
    lr_ = {d: [e[f"gnorm_prior_{d}"] / e[f"gnorm_true_{d}"] for e in on if e.get(f"gnorm_prior_{d}") is not None
               and e.get(f"gnorm_true_{d}")] for d in ACT}
    chk("gnorm_prior_<d> / gnorm_true_<d> = the lambda of the job's ARGS (rel. 1e-3) at every on-eval (the lambda that weighted "
        "the prior's gradient; with the on-window decomposition check: the lambda of the backward)",
        all(vs and want_lam[d] > 0 and all(abs(v / want_lam[d] - 1) < 1e-3 for v in vs) for d, vs in lr_.items()),
        {d: [round(min(vs), 5), round(max(vs), 5), len(vs)] if vs else None for d, vs in lr_.items()})
    chk(f"geo > {TOL} for the active prior and 0 < grad_prior < inf at every on-eval (before a divergence)",
        (bool(on) or diverged) and all(all(e.get(f"geo_{d}") is None or e[f"geo_{d}"] > TOL for d in ACT)
                                      and 0 < e.get("grad_prior", 0) < math.inf for e in on), len(on))
    if on:
        we = on[-1]
        cka_true = {d: round(we[f"cka_{d}"] - c[0][f"cka_{d}"], 4) for d in ACT}
        chk(f"item-state CKA to the target rises by >= {I['cka_rise']} from step 0 to the window's last eval (step {we['step']})",
            diverged or all(v >= I["cka_rise"] for v in cka_true.values()), cka_true)
else:
    chk("no prior ever on (prior_on_frac 0, no events, no domain prior on or applied, no prior loss term or gradient; LM share 1)",
        c[-1]["prior_on_frac"] == 0 and r.get("prior_events") == [] and all(
            not any(e["prior_on_domains"].values()) for e in c) and all(c[-1][f"prior_used_frac_{d}"] == 0 for d in ALLD)
        and all(e["geo"] == 0 and not any(k in GEOK or k.startswith("gnorm_prior_") for k in e) and e["grad_lm_share"] == 1.0
                for e in c), c[-1]["prior_on_frac"])
o = json.loads(Path(offl).read_text()).get(str(Path(run).resolve())) if Path(offl).exists() else None
if I["save_model"]:
    chk("offline_lm on the full val, |orig_minus_logged| < 0.002 (skipped if the run diverged)",
        o is not None and o["max_windows"] == 0 and (diverged or abs(o["orig_minus_logged"]) < 0.002), o and o["orig_minus_logged"])
else:   # A4: stage 1 saves no model (disk): no offline LM; the bpb tie-break and the LM rows use the in-training val bpb
    mf_ = sorted(p_.name for p_ in Path(run).parent.glob("model_*.pt"))
    chk("stage 1 (A4): no saved model and no offline LM, by design", not mf_ and not Path(offl).exists(), mf_)
m = r.get("metrics", {})
rate = [(q["sec"] - p["sec"]) / (q["step"] - p["step"]) for p, q in zip(c, c[1:]) if q["sec"] > p["sec"] and q["step"] > p["step"]]
util, mem, gname = [], [], set()
for row in csv.reader(open(gpu)) if Path(gpu).exists() else []:
    try:
        util.append(int(row[2].split()[0])); mem.append(int(row[4].split()[0])); gname.add(row[1].strip())
    except (IndexError, ValueError):
        pass
chk(f"GPU = {I['gpu']} in every nvidia-smi sample (the batch's GPU type)", bool(gname) and all(I["gpu"] in g for g in gname), sorted(gname))
M = lambda k, f="area": m.get(k, {}).get(f)
on_e = [e for e in pre if e.get("prior_on")]
med = lambda k: statistics.median([e[k] for e in on_e if e.get(k) is not None]) if on_e else None
pg = [e["grad_prior"] / e["grad_lm"] for e in on_e if e.get("grad_prior") is not None and e.get("grad_lm")]
sn = [1 / (1 + v) for v in pg]   # A2: the LM share by norm a / (a + b) (the gate's quantity; info only, the gate is the read-out's)
import importlib.metadata as ilm, platform
def ver(p):
    try:
        return ilm.version(p)
    except ilm.PackageNotFoundError:
        return None
versions = {"python": platform.python_version(), **{p: ver(p) for p in ("torch", "transformers", "numpy", "tokenizers")}}
info = {"versions": versions, "batch": I["batch"], "tag": a["tag"], "seed": a["seed"], "lr": a["lr"], "active": ACT,
        "read_area": {d: M(I["read"][d]) for d in EXT}, "read_final3": {d: M(I["read"][d], "final3") for d in EXT},
        "seen_final3": {d: M(I["seen"][d], "final3") for d in EXT}, "numbers_area": M("numbers"),
        "hard_final3": {d: M(f"{d}_readout_hard", "final3") for d in COMP},
        "heuristic": {d: [r["domains"][d].get("heuristic_ansfreq"), r["domains"][d].get("heuristic_shorter")] for d in COMP},
        "final_cka": {d: c[-1].get(f"cka_{d}") for d in EXT}, "cka_rise_true": cka_true,
        "grad_window_median": {"lm_share": med("grad_lm_share"), "clip_coef": med("clip_coef"),
                               "prior_over_lm": statistics.median(pg) if pg else None, "n_on_evals": len(on_e),
                               "lm_share_norm": statistics.median(sn) if sn else None,
                               "frac_prior_gt_lm": (sum(v > 1 for v in pg) / len(pg)) if pg else None},
        "ties2": sum(v for e in c for k, v in e.items() if k.startswith("ties2_")),
        "tiesk": sum(v for e in c for k, v in e.items() if k.startswith("tiesk_")), "kway_em_tie": kt,
        "read_area_first_token_kway": {d: M(f"{d}_readout_kway") for d in KST},
        "kspan_dp_max": kdp if KST else None,
        "lm0": c[0]["lm"], "final_val_bpb": c[-1]["val_bits_per_byte"], "bpb_full": o and o["bpb_full"], "diverged": diverged,
        "diverged_at_step": bad[0] if bad else None, "sec_per_step_median": statistics.median(rate) if rate else None,
        "projected_run_h": statistics.median(rate) * r["steps"] / 3600 if rate else None,
        "gpu_util_median": statistics.median(util) if util else None, "gpu_mem_max_MiB": max(mem) if mem else None,
        "gpu_samples": len(util), "gpu_names": sorted(gname), "commit": Path(pinned).read_text().split()[0] if Path(pinned).exists() else None}
ok = all(k["ok"] for k in checks)
# A5.1: the checks whose failure alone leaves the run VALID data (flagged; readout_ptdom.load): the two GPU-numerics tolerances and
# the CKA-rise sanity bar (A1); the exit code stays 4 on any failed check (the loader, not the exit code, decides validity)
FLAGC = ("2-way scoring double path (GPU-numerics tolerance", "k-span scoring double paths (GPU-numerics tolerance",
         "item-state CKA to the target rises")
flag_only = not ok and all(k["ok"] or k["check"].startswith(FLAGC) for k in checks)
Path(out_json).write_text(json.dumps({"pass": ok, "valid_data_flagged": flag_only, "checks": checks, "info": info}, indent=1))
for k in checks:
    print(("PASS " if k["ok"] else "FAIL ") + k["check"] + f"  ({k['value']})")
print(json.dumps(info))
print("CHECKS", "PASS" if ok else "FAIL" + (" (only GPU-numerics tolerance / CKA-rise checks failed: the run is VALID data, "
                                            "flagged; AUDIT.md A5.1)" if flag_only else ""))
sys.exit(0 if ok else 4)
'''


def dry_grep(P):
    """The DRY_GREP lines of --dry_inputs (make_rb6_tasks.DRY_GREP + the Rev E site-rule and other-option lines + the Rev F
    k-way candidate lines; batch A has no k-way candidate line, so its DRY rows and reference are unchanged)."""
    assert P.DRY_GREP.endswith(")")
    return P.DRY_GREP[:-1] + "|  site rule: |  other-option line |  k-way candidates )"


def dry_block(batch, arm):
    """The arm-dependent input checks of --dry_inputs: no numbers target; per domain prior / shuf / lam in 'domains:'."""
    pri = () if arm == "none" else (arm[5:],)
    specs = " ".join(f"{d}:{str(d in pri).lower()}:false:{json.dumps(1.0 if d in pri else 0.0)}" for d in ("numbers",) + BATCHES[batch])
    return ("""! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a numbers prior target (every ptdom arm has none)"; exit 3; }
dl=$(grep -m1 '^domains: ' "$OUT/dry_inputs.txt" || true)
has() { grep -qF -e "$1," -e "$1}" <<< "$blk"; }
for sp in """ + specs + """; do
  IFS=: read -r d p s l <<< "$sp"
  blk=$(grep -oE "\\"$d\\": \\{[^}]*\\}" <<< "$dl" || true)
  has "\\"prior\\": $p" && has "\\"shuf\\": $s" && has "\\"lam\\": $l" ||
    { echo "## domain $d not prior $p / shuf $s / lam $l in the domains: line"; exit 3; }
done
grep -qF '"numbers": {"target": "none", ' <<< "$dl" || { echo "## numbers target not none"; exit 3; }""")


def job_name(batch, arm, lr, seed, rr=0, pilot=False):
    """The run / out-dir name of a cell; rr > 0: its rr-th re-run (A2); A4 pilot: ptdom-pilot-<batch>-<arm>-lr<lr>-s<seed>."""
    if pilot:
        return f"ptdom-pilot-{batch}-{arm}-lr{lr}-s{seed}" + (f"-r{rr}" if rr else "")
    return f"{tag_of(batch, arm, lr)}-s{seed}" + (f"-r{rr}" if rr else "")


def slurm_name(batch, arm, lr, seed, rr=0, slot=None, pilot=False):
    """The Slurm job name: L40S = the run's; B200 = its singleton slot (pt300x's protocol); A4 pilot: llmadd_ptdom-pilot-<batch>."""
    if pilot:
        return f"llmadd_ptdom-pilot-{batch}"
    return f"llmadd_ptdom-b200-slot{slot}" if HWOF[batch] == "b200" else f"llmadd_{job_name(batch, arm, lr, seed, rr)}"


def hold_paths(batch):
    """A4: the global kill switch and the batch's own (select_ptdom writes HOLD_<batch> when an arm fails the gate in every
    stage-1 cell: only that batch stops)."""
    return R / "HOLD", R / f"HOLD_{batch}"


def render(P, batch, arm, lr, seed, ref, commit, rr=0, slot=None, stage="1", pilot=False):
    tag = tag_of(batch, arm, lr)
    name = job_name(batch, arm, lr, seed, rr, pilot)
    hwk = HWOF[batch]
    hw = HW[hwk]
    save = saves(stage, pilot)
    OUT = (R / "pilot" if pilot else R / "out") / name
    args = args_of(batch, arm, lr, seed, OUT, save, pilot)
    f = P.flags(args)
    want = {"--data": str(DATA), "--tokenizer": str(TOK), "--tokenization": "whole", "--out": f"{OUT}/{tag}",
            "--arith_lines": "1", "--domains": domains_arg(batch), "--extra_lines": str(K), "--lr": lr, "--seed": str(seed),
            "--tag": f"_{tag}", "--save_model": True if save else None, "--ckpt_every": str(PILOT_CKPT if pilot else hw["ckpt"]),
            "--tokens": str(32 * 512 * PILOT_STEPS) if pilot else None, "--eval_every": str(PILOT_EVAL) if pilot else None,
            **want_flags(arm)}
    for k, v in want.items():
        assert f.get(k) == v, (name, k, f.get(k), v)
    assert set(f) == {k for k, v in want.items() if v is not None}, (name, sorted(f))   # every other flag = its default
    assert arm in arms(batch) and lr in LADDER and seed in SEL + CONF and stage in STAGES
    assert pilot or (slot is not None) == (hwk == "b200") and (slot is None or SLOT0[batch] <= slot < SLOT0[batch] + SLOTS), (batch, slot)
    I = intent(P, batch, arm, lr, seed, ref, save, pilot)
    for k in ("lm0_ref", "data_sha0", "dom", "table_sha"):
        assert I[k] is not None, f"{name}: no CPU reference {k} for batch {batch} seed {seed}: run the batch's refs first"
    doms = BATCHES[batch]
    assert set(I["dom"]) == set(doms) and all(set(I["dom"][d]) == set(DOM_KEYS[dom_kind(d)]) for d in doms), (name, I["dom"])
    assert set(I["table_sha"]) == set(I["tables"]), (name, I["table_sha"])
    dry, n_dry = ref["dry"]
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    runfile = f"run_whole_none_seed{seed}_{tag}.json"
    assert hw["plan_h"] < P.hours(hw["time"]), (hw["time"], hw["plan_h"])
    G = dry_grep(P)
    n_site = len([d for d in doms if d != "months"])
    dep_chk = "\n".join(f'[ "$(sha256sum {fn} | cut -c1-12)" = "{v}" ] || {{ echo "## {fn} sha mismatch: exiting"; exit 3; }}'
                        for fn, v in WANT_SHA.items())
    desc = "none (no prior)" if arm == "none" else f"only the {arm[5:]} prior (lambda 1, window (0.05, 0.4), ctx)"
    rr_line = (f"# A2 re-run {rr} of {job_name(batch, arm, lr, seed, 0, pilot)} (its last submission exited before the run finished, e.g. "
               f"at HOLD; make_jobs_ptdom.py --rerun): a fresh run in its own out dir\n" if rr else "")
    if pilot:
        rr_line += (f"# A4 GPU PILOT of batch {batch} (not part of any result; output runs/ptdom/pilot/): {PILOT_STEPS} steps (--tokens), "
                    f"evals every {PILOT_EVAL} steps, ckpt every {PILOT_CKPT}, --save_model; every in-job check on a B200\n")
    elif not save:
        rr_line += "# A4 stage 1: no --save_model (disk; models are saved from stage 2 on): no offline LM; CHECK_PY asserts no model file\n"
    sname = slurm_name(batch, arm, lr, seed, rr, slot, pilot)
    if hwk == "b200":   # pt300x's 294M header (make_jobs_pt300x.job_x), this batch's slot and log name
        head = f"""#SBATCH --job-name={sname}
#SBATCH --dependency=singleton
#SBATCH --account=<slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --constraint=b200
#SBATCH --comment="preemption=yes"
#SBATCH --nice=20
#SBATCH --cpus-per-task={hw['cpus']}
#SBATCH --mem={hw['mem']}
#SBATCH --time={'01:00:00' if pilot else hw['time']}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={B}/logs/{sname if pilot else 'llmadd_' + name}_%j.out"""
        resume = (f"Preemptible: a preempted job is requeued (same id), re-runs the input checks and the trainer resumes from its ckpt "
                  f"(every {PILOT_CKPT if pilot else hw['ckpt']} steps); "
                  + ("the pilot's own name (no slot)." if pilot else
                     f"singleton slot {slot} (batch {batch}: slots {SLOT0[batch]}..{SLOT0[batch] + SLOTS - 1}; 16 chains over the four batches)."))
    else:
        head = f"""#SBATCH --job-name={sname}
#SBATCH --account=<slurm-account>
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --nice=20
#SBATCH --cpus-per-task={hw['cpus']}
#SBATCH --mem={hw['mem']}
#SBATCH --time={hw['time']}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out"""
        resume = f"A requeued job re-runs the input checks and resumes from its ckpt (every {hw['ckpt']} steps)."
    tl = (f"""#!/bin/bash
{head}
# ptdom A3 / A4 batch {batch} stage {'pilot' if pilot else stage}: {desc} @ lr {lr}, 53M, seed {seed}; data = numbers (2%) + K 2 lines of
# {' / '.join(doms)} per row, identical in every arm of the batch and seed; code = runs/ptdom/code (COMMIT {commit}); intent
# spec and pre-registered screen / verdict runs/ptdom/AUDIT.md (Amendments A3, A4); generated by llmadd_torch/make_jobs_ptdom.py
# (change the generator, not this file). {resume}
# 1. HOLD, snapshot shas, the GPU type ({hw['gpu']}), data sha256; --dry_inputs: model line, no numbers target, per domain
#    prior / shuf / lam, numbers first, content scan, the site rule ({n_site} domains asserted), DRY rows = the CPU reference (exit 3)
# 2. training ({'--save_model ' if save else 'no saved model, '}--ckpt_every {PILOT_CKPT if pilot else hw['ckpt']}); 3. {'offline_lm.py (full val)' if save else 'no offline LM (stage 1)'};
# 4. CHECK_PY (make_jobs_ptdom.py) with this run's intent and its ARGS -> $OUT/check.json (exit 4 on a fail); gpu.csv every 60 s
{rr_line}set -euo pipefail
R={R}; OUT={OUT}; TAG={tag}; SEED={seed}; DATA={DATA}; TOK={TOK}; PY={PY}
# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/ptdom/HOLD (every batch) or
# runs/ptdom/HOLD_{batch} (A4: this batch only) exists
if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{batch}" ]; then echo "HOLD present: exiting"; exit 0; fi
cd "$R/code/tools"
{dep_chk}
grep -q "pretrain_range.py {WANT_SHA['pretrain_range.py']}" "$R/code/COMMIT" || {{ echo "## COMMIT does not name the trainer {WANT_SHA['pretrain_range.py']}"; exit 3; }}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU: $GPU"
case "$GPU" in *{hw['gpu']}*) ;; *) echo "## not a {hw['gpu']} (batch {batch}'s GPU type, one type per batch): exiting"; exit 3;; esac
mkdir -p "$OUT"
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
[ -s "$OUT/gpu.csv" ] || echo "timestamp, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]" > "$OUT/gpu.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader -l 60 >> "$OUT/gpu.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
{{ date -u +%FT%TZ; hostname; echo "SLURM_JOB_ID ${{SLURM_JOB_ID:-}} restart ${{SLURM_RESTART_COUNT:-0}}"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "cpus $(nproc)"; }} >> "$OUT/node.txt" 2>&1 || true

echo "## 1. inputs $(date -u +%FT%TZ)"
(cd "$DATA" && sha256sum -c -) <<'SHA' || exit 3
{DATA_SHA[0]}  train.bin
{DATA_SHA[1]}  val.bin
SHA
ARGS="{args}"
"$PY" pretrain_range.py $ARGS --dry_inputs > "$OUT/dry_inputs.txt" 2>&1 || {{ cat "$OUT/dry_inputs.txt"; exit 3; }}
cat "$OUT/dry_inputs.txt"
grep -qxF '{P.MODEL_LINE}' "$OUT/dry_inputs.txt" || {{ echo "## not the intended model size / gradient checkpointing state"; exit 3; }}
""" + dry_block(batch, arm) + "\n" + f"""grep -q '^first line after the text (share of rows): {{"numbers": 1.0, ' "$OUT/dry_inputs.txt" || {{ echo "## the numbers line is not first in every row"; exit 3; }}
grep -q '^content scan: 640 training rows, no held-out prompt of ' "$OUT/dry_inputs.txt" || {{ echo "## no content scan"; exit 3; }}
[ "$(grep -cE '^  site rule: the prior and the CKA read-out use the LAST token of each item name .*: [0-9]+ sites in [0-9]+ lines asserted; ' "$OUT/dry_inputs.txt")" = {n_site} ] || {{ echo "## the site rule is not asserted for the {n_site} domains with item-name sites"; exit 3; }}
n=$(set +o pipefail; grep -cE "{G}" "$OUT/dry_inputs.txt")
got=$(set +o pipefail; grep -E "{G}" "$OUT/dry_inputs.txt" | sha256sum | cut -d' ' -f1)
[ "$n" = {n_dry} ] && [ "$got" = {dry} ] || {{ echo "## dry_inputs differ from the CPU reference rows ($n lines, sha256 $got)"; exit 3; }}
echo "## dry_inputs rows = the reference ({dry[:12]}, {n_dry} lines)"

echo "## 2. training $(date -u +%FT%TZ)"
"$PY" pretrain_range.py $ARGS
""" + (f"""echo "## 3. offline LM $(date -u +%FT%TZ)"
"$PY" offline_lm.py --data "$DATA" --tokenizer "$TOK" --out "$OUT/offline_lm.json" "$OUT/$TAG"
""" if save else """echo "## 3. offline LM: not run (stage 1, A4: no saved model)"
""") + f"""echo "## 4. intent checks $(date -u +%FT%TZ)"
"$PY" - "$OUT/$TAG/{runfile}" "$OUT/offline_lm.json" "$OUT/gpu.csv" "$R/code/COMMIT" "$DATA" \\
  "$OUT/check.json" '{ij}' "$ARGS" <<'PY' | tee "$OUT/check.txt"
{CHECK_PY}PY
echo "## done $(date -u +%FT%TZ)"
""")
    return name, tag, tl, args, I


def edge_step(lr, side):
    i = LADDER.index(lr) + (1 if side == "upper" else -1)
    assert 0 <= i < len(LADDER), (lr, side)
    return LADDER[i]


def cells(batch, stage, picks=None):
    """[(arm, lr, seed)] of a batch's stage, in submission order (seed-major: a partial stage stays paired and balanced).
    A4: an extension cell (E) runs at seeds 0-7 (the selection reads 0-3; seeds 4-7 there skip a round when it is picked);
    a confirm cell (C) at seeds 4-7. Stage 2: E = first-step extensions (EXT_LR), C = the stage-1 picks (grid lrs); 2b: E =
    none's second lower step (EXT2), C = moved picks; 2c: C only. A5: F = the fixB none cells (seeds 0-7) in 2b / 2c."""
    if stage == "1":
        return [(a, lr, s) for s in SEL for a in arms(batch) for lr in grid(a)]
    pk = json.loads(Path(picks).read_text())
    assert pk.get("status") == "decided" and pk.get("design") == "ptdom" and pk.get("amendment") == "A5", (pk.get("status"), pk.get("amendment"))
    assert pk.get("batch") == batch and pk.get("stage_written") == stage, (pk.get("batch"), pk.get("stage_written"), batch, stage)
    xs = [tuple(v) for v in pk.get("E", [])]
    fs = [tuple(v) for v in pk.get("F", [])]   # A5 fixB (AUDIT.md A5.2): none at a prior's final pick outside none's lr range
    cs = [tuple(v) for v in pk["C"]]
    for a, lr in fs:
        assert stage in ("2b", "2c") and a == "none" and lr in FIXB_LR and (a, lr) not in xs, (stage, a, lr)
    kind_ = lambda a: "none" if a == "none" else "prior"
    side = lambda a, lr: "lower" if float(lr) < float(grid(a)[0]) else "upper"
    for a, lr in xs:   # stage 2: one factor-2 step beyond a stage-1 grid edge (EXT_LR); 2b: none's second lower step only
        assert a in arms(batch) and lr == (EXT_LR if stage == "2" else EXT2).get((kind_(a), side(a, lr))), (stage, a, lr)
    assert stage != "2c" or not xs, "stage 2c has no extension"
    ext_all = lambda a: [v for (k, _), v in list(EXT_LR.items()) + list(EXT2.items()) if k == kind_(a)]
    ok_lr = lambda a, lr: lr in grid(a) or lr in ext_all(a)
    assert len(set(cs)) == len(cs) and all(a in arms(batch) and ok_lr(a, lr) for a, lr in cs), cs
    assert stage != "2" or all(lr in grid(a) for a, lr in cs), "stage 2 confirms the stage-1 picks (grid lrs)"
    out = [(a, lr, s) for s in SEL + CONF for a, lr in xs + fs] + [(a, lr, s) for s in CONF for a, lr in cs]
    assert len(set(out)) == len(out), "a cell asked twice"
    return out


def queue():
    """A2: (job ids, job names) of this user's jobs in the Slurm queue (pending or running), or None when squeue fails (a
    re-run is then refused)."""
    o = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "yh6210"), "-o", "%i %j"], capture_output=True, text=True)
    if o.returncode != 0:
        return None
    rows = [l.split() for l in o.stdout.splitlines() if l.strip()]
    return {r[0] for r in rows}, {r[1] for r in rows if len(r) > 1}


def rerun_plan(batch, todo, rdir, q, logdir, n_evals=75):
    """A2, the re-run path (e.g. after HOLD: a pending job that starts while runs/ptdom/HOLD exists exits 0 before training,
    and its SUBMITTED_ flag stays). For each cell (arm, lr, seed): k = the latest submitted name; the cell gets the re-run
    name job_name(.., k + 1) iff its latest name has a SUBMITTED_ flag, that job is not in the queue (the job id of the flag;
    for L40S also the job name; B200 jobs are named by their slot, so by id only; q None -> refused), and its out dir holds
    no FINISHED run JSON (a finished run is data, never re-run). -> ([(arm, lr, seed, k + 1, reason)], [refusals])"""
    if q is None:
        raise SystemExit("--rerun: squeue failed; a re-run needs the queue state (refused)")
    ids, names = q
    out, refused = [], []
    for a, lr, s in todo:
        k = 0
        while (rdir / f"SUBMITTED_{job_name(batch, a, lr, s, k + 1)}").exists():
            k += 1
        name = job_name(batch, a, lr, s, k)
        flag = rdir / f"SUBMITTED_{name}"
        if not flag.exists():
            continue
        txt = flag.read_text().split()
        jid = txt[-1] if txt and txt[-1].isdigit() else None
        by_name = HWOF[batch] != "b200" and f"llmadd_{name}" in names
        if jid is None or jid in ids or by_name:
            refused.append(f"{name}: " + ("no job id in its SUBMITTED_ flag" if jid is None else f"job {jid} is in the queue"))
            continue
        tag = tag_of(batch, a, lr)
        rj = rdir / "out" / name / tag / f"run_whole_none_seed{s}_{tag}.json"
        ne = None
        if rj.exists():
            try:
                ne = len(json.loads(rj.read_text())["curve"])
            except (json.JSONDecodeError, KeyError):
                ne = -1
        if ne is not None and ne >= n_evals:
            refused.append(f"{name}: its run finished ({ne} evals): a run is data, never re-run")
            continue
        lg = logdir / f"llmadd_{name}_{jid}.out"
        held = lg.exists() and "HOLD present: exiting" in lg.read_text(errors="replace")
        why = ("HOLD exit (" + lg.name + ")") if held else (f"unfinished run JSON ({ne} evals)" if ne is not None else
                                                             f"no run JSON (log {lg.name}{'' if lg.exists() else ' absent'})")
        out.append((a, lr, s, k + 1, why))
    return out, refused


def stage2_expect(m):
    """A4: stage 2 / 2b / 2c runs of one batch with m advancing domains under a FLAT lr response (every argmax equally likely;
    the pick = the argmax): prior: 4 confirm seeds + the extension at 8 seeds (the 2-point grid's argmax is always at an
    edge); none (shared by the batch's advancing domains): 4 per distinct stage-1 pick (3 (1 - (2/3)^m)), the 1e-3
    extension (8) when some advancing domain's argmax is 2e-3 (1 - (2/3)^m), the 1.6e-2 extension (8) when some argmax is
    8e-3 (the same), the 5e-4 step (8) when 1e-3 then wins for some domain (1 - (5/6)^m); moved picks need no new seeds
    (their extension ran at 8 seeds; a top-two tie broken by bpb towards another grid lr is not modelled). -> (min,
    expected, max) runs. A5 fixB adds none@1e-3 / none@5e-4 (8 runs each, shared) where a prior's final pick lies below none's
    range for its domain and no other domain asked that cell: per domain (flat) the prior's final pick is 1e-3 or 5e-4 with
    p 1/4 each; none@1e-3 is then asked by a domain with p 1/3 (extension) + 1/4 x 2/3 (fixB) = 1/2 instead of 1/3, none@5e-4
    with p 1/6 + 1/4 x 5/6 = 3/8 instead of 1/6 (independent domains): + 8 ((2/3)^m - (1/2)^m) + 8 ((5/6)^m - (5/8)^m)."""
    if m == 0:
        return 0, 0.0, 0
    e = 12 * m + 4 * 3 * (1 - (2 / 3) ** m) + 2 * 8 * (1 - (2 / 3) ** m) + 8 * (1 - (5 / 6) ** m)
    e += 8 * ((2 / 3) ** m - (1 / 2) ** m) + 8 * ((5 / 6) ** m - (5 / 8) ** m)   # A5 fixB
    return 12 * m + 4, e, 12 * m + 4 * min(m, 3) + 24 + 16


def plan():
    """The design table and the cost (printed; AUDIT.md A3 'Cost')."""
    print(f"ptdom A5: 53M, 3e8 tokens (18,310 steps of 32 x 512, 75 evals), K {K}, numbers line 2%; stage 1 seeds 0-3: "
          f"none {' / '.join(GRID1['none'])}, prior arms {' / '.join(GRID1['prior'])}; every batch on B200 ({SLOTS} singleton slots "
          f"per batch, {SLOTS * len(BATCHES)} in all); stage 1 saves no model")
    tot = {"l40s": 0.0, "b200": 0.0}
    n1 = 0
    for b, doms in BATCHES.items():
        hw = HW[HWOF[b]]
        n = len(cells(b, "1"))
        n1 += n
        tot[HWOF[b]] += n * hw["plan_h"]
        print(f"  batch {b:3s} {hw['gpu']:5s} {', '.join(doms):70s} stage 1: {n} runs = {n * hw['plan_h']:.1f} {hw['gpu']}-h")
    print(f"  stage 1 total: {n1} runs = {tot['b200']:.1f} B200-h (plan {HW['b200']['plan_h']} B200-h per run)")
    for frac in (0.3, 0.5, 0.7):
        m = 6 * frac
        lo, ex, hi = (0.0, 0.0, 0.0)
        hrs = {"l40s": [0.0, 0.0, 0.0], "b200": [0.0, 0.0, 0.0]}
        for b in BATCHES:
            mn, e, mx = stage2_expect(m)
            for i, v in enumerate((mn, e, mx)):
                hrs[HWOF[b]][i] += v * HW[HWOF[b]]["plan_h"]
            lo += mn; ex += e; hi += mx
        print(f"  stage 2 / 2b at {100 * frac:.0f}% advancing ({4 * m:.1f} of 24 domains, {m:.1f} per batch; A4: extensions at 8 seeds, A5: fixB none cells, "
              f"none at both edges + 5e-4): min {lo:.0f} / expected {ex:.0f} / max {hi:.0f} runs = "
              f"B200-h {hrs['b200'][0]:.0f} / {hrs['b200'][1]:.0f} / {hrs['b200'][2]:.0f}")


def main():
    if "--plan" in sys.argv:
        plan()
        return
    if "--snapshot" in sys.argv:   # create / verify runs/ptdom/code only (the CPU references run on it)
        print("snapshot", snapshot(), (R / "code" / "COMMIT").read_text().strip())
        return
    batch, stage = opt("--batch"), opt("--stage")
    pilot = "--pilot" in sys.argv
    assert batch in BATCHES, "--batch A | B1 | B2 | B3 (--plan for the table)"
    if pilot:   # A4: the one GPU pilot job of the batch (stage-1-like cell, compressed schedule; not part of any result)
        assert stage is None and "--rerun" not in sys.argv, "--pilot takes no --stage / --rerun"
        stage = "1"
    assert stage in STAGES, "--stage 1 | 2 | 2b | 2c"
    commit = snapshot()
    sys.path.insert(0, str(R / "code" / "tools"))
    import make_rb6_tasks as P   # the snapshot's rb6 constants: STEPS, EVAL_EVERY, N_PARAMS, MODEL_LINE, flags(), ...
    assert Path(P.__file__).resolve().parent == (R / "code" / "tools").resolve()
    assert (P.BS, P.SEQ, P.STEPS, P.EVAL_EVERY, P.N_PARAMS) == (32, 512, 18310, 250, 52672128)
    RF = refs(batch)
    todo = [PILOT[batch] + (0,)] if pilot else cells(batch, stage, opt("--picks"))
    rerun = "--rerun" in sys.argv
    if rerun:   # A2: only the cells whose latest submission exited before its run finished, under the next name
        rp, refused = rerun_plan(batch, todo, R, queue(), B / "logs")
        for x in refused:
            print(f"re-run refused: {x}")
        for a, lr, s, k, why in rp:
            print(f"re-run: {job_name(batch, a, lr, s, k - 1)} -> {job_name(batch, a, lr, s, k)} ({why})")
        todo = [(a, lr, s, k) for a, lr, s, k, _ in rp]
    else:
        todo = [(a, lr, s, 0) for a, lr, s in todo]
    existing = {p.name for p in B.glob("runs/*/out/*") if p.parent.parent != R}
    JD = R / "pilot" if pilot else R   # A4: the pilot's job files, flags and outputs live in runs/ptdom/pilot/
    JD.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (a, lr, s, rr) in enumerate(todo):
        assert s in RF, f"no CPU reference for batch {batch} seed {s}: run the batch's refs first"
        slot = SLOT0[batch] + i % SLOTS if HWOF[batch] == "b200" and not pilot else None
        name, tag, text, args, I = render(P, batch, a, lr, s, RF[s], commit, rr, slot, stage, pilot)
        assert name not in existing, f"{name}: an out dir of that name exists in another batch"
        job, flag = JD / f"job_{name}.sbatch", JD / f"SUBMITTED_{name}"
        if flag.exists():   # a submitted job file is never rewritten
            if job.read_text() != text:
                print(f"WARNING: {job.name} was submitted and differs from this rendering: kept, not rewritten")
        elif "--no-write" not in sys.argv:
            job.write_text(text)
        jobs.append((name, job, flag, args))
    if "--print" in sys.argv:
        for name, job, flag, args in jobs:
            print(f"{name}{' (SUBMITTED)' if flag.exists() else ''}: {args}")
    if "--test-only" in sys.argv:
        n_t = int(opt("--test-only", "0")) or len(jobs)
        for name, job, flag, args in jobs[:n_t]:
            o = subprocess.run(["sbatch", "--test-only", str(job)], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
    if "--submit" in sys.argv:
        for h_ in hold_paths(batch):   # A4: the global kill switch and the batch's own
            assert not h_.exists(), f"{h_} exists: remove it to submit (the user decides)"
        for name, job, flag, args in jobs:
            if flag.exists():   # A2: a job name is submitted once; a re-run is a new name (--rerun)
                print(f"{name}: refused (SUBMITTED_{name} exists: a name is submitted once; a re-run is a new name: --rerun)")
                continue
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
    print(f"ptdom A5 batch {batch} {'PILOT' if pilot else 'stage ' + stage}{' (A2 re-runs)' if rerun else ''}: {len(jobs)} jobs rendered in {JD}; code {commit}, "
          f"trainer {WANT_SHA['pretrain_range.py']}; {HW[HWOF[batch]]['gpu']}; submitted: {'yes' if '--submit' in sys.argv else 'no'}")


if __name__ == "__main__":
    main()
