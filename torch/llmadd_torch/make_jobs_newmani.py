"""Torch jobs (2026-10-20; revised 2026-10-21 after the build audit, before any submission): new-manifold toy screen,
STAGE 1 (scout). Design: LLM_addition_review/notes/new_manifolds_screen_design.md; intent spec, cells, PRE-REGISTERED
reading (+ Amendment A1), costs: runs/newmani/AUDIT.md; audit: notes/batch1_build/audit_newmani_build.md. Rescue standard
(report/audits/rescue_standard.md, Amendment 1): every candidate gets an assignment prior (Tier L) AND a shape-only prior
(Tier S) with their controls; none gets the full lr x wd grid at 4 B; the priors get their own lr sweep. NOT submitted by
the agent (--submit is the user's).

Device (criterion f): every arm on an L40S (l40s_public, gpu48), 5 single-thread processes per GPU as toyshape. Measured
toyshape L40S rates (ms/step/process, 5 per GPU) are linear in the training rows: one L40S ~ the whole 32-CPU cpu_short
cap, and gpu48 allows 16 GPUs / 48 h per user (AUDIT.md "Device").

Tasks (toy_struct --task box / grp, opt-in additions; every pre-existing path bitwise = d182ce548bcc):
  box5 = --task box --box=-2:2,-2:2,-2:2 (125 items, mul / div, relation-class split); 2i, d57, a5, s5 = --task grp
  (whole twin-orbit class split). Fractions / seeds: preflight/seed_scan.json (the design's m ladder 6 / 12 / 24 training
  cells per item; the first 3 seeds whose split passes coverage + closure 1.0 + shuf far, the fraction moved up when < 3
  pass). Stage 2's fresh seeds are pre-registered in preflight/stage2_seeds.json (not rendered here).
Per (task, m):
  none   lr {1e-3, 3e-3, 1e-2, 3e-2} x wd {0.1, 1, 3}, 3 seeds, 4 B = 200k steps            (criterion a)
  prior  lr {1e-3, 3e-3, 1e-2}, wd 1, lam 1, 3 seeds, B = 50k, all @always (criterion b):
           assign  <true>@always          Tier L (CKA to the true matrix target)
           specrand@always, shuf@always   Tier L controls (same spectrum / far permutation, criterion d)
           shape   shape:<true>@always    Tier S (spectrum + norms only, permutation-invariant)
           shape:sphere@always            Tier S control (flat, constant norm, rank r' != r: the smallest non-isotypic rank)
           shape:line@always              Tier S control (open chain, rank r)
  positive controls (design 3.1): 2i / d57 / s5 at frac 0.4, none lr 1e-3 wd 1 (A5's m 24 cell is f 0.4); box5 (audit I3)
  at frac 0.4 and 0.6, none lr 1e-3 wd 1 and lr 1e-2 wd 0.1 (dim's best); 3 seeds, 200k.
  extension (rule 2: lr 1e-1 x wd {0.1, 1, 3}, wd 10 x the 4 lrs), APPEND-ONLY (audit I5): --ext (every (task, m) cell) or
  --ext_cells box5-m6,2i-m12 (those cells). Extension runs go to their own jobs '<task>-m<m>-ext-j<k>', packed on their
  own, so the default jobs never change; once an extension job is submitted it is rendered on every later call.
Every run: --shape_log --eval_every 100 --threads 1 --ckpt_dir <cell>/ckpt --ckpt_every 10000 --deadline <job end - 15 min>
(a requeued / time-limited run resumes from its checkpoint; a finished run is skipped: its JSON exists).
Jobs: per group the runs are packed (largest first) into lanes of one process each, 5 lanes per job (one L40S);
--time = 2 x the job's planned time (rate model below), at least 2 h. Account torch_pr_625_general.
Append-only guard: every job file is rendered in memory first; if any SUBMITTED job's file would change (or be deleted),
the generator stops before writing anything.
Kill switch: runs/newmani/HOLD. Options:
    python make_jobs_newmani.py [--ext | --ext_cells a,b]                       render (+ the code snapshot once)
    python make_jobs_newmani.py [...] --print [--wave 1|2] [--only a,b] [--nice N]   list what --submit would sbatch
    python make_jobs_newmani.py [...] --submit [--wave 1|2] [--only a,b] [--nice N]  the user's call (skips jobs in squeue)
    python make_jobs_newmani.py --status            per job: runs done / partial, the Slurm state (sacct), and separately
                                                    the jobs that ENDED with runs missing (deadline exit 0, cancel, ...)
    python make_jobs_newmani.py --submit --only a,b --redo   resubmit ended jobs whose runs are incomplete (not if queued)
wave 1 = box5 + 2i (the design's top 2 + their positive controls); wave 2 = d57, a5, s5 (+ positive controls).

Added 2026-10-25 (user approval; AUDIT.md Amendment A2, "Stage 2 (box5 m6, Tier L)", "Diagnostics (post hoc)"),
APPEND-ONLY like the extension (every default / Stage-1 job file stays byte-identical; sticky once submitted):
    --stage2   Stage 2 of box5 m6 Tier L on the pre-registered seeds (preflight/stage2_seeds.json, box5|6 = 100, 103,
               108, 122, 126): none 12 configs + the 7 rule-2 extension configs at 4 B; expo@always, specrand@always,
               shuf@always at lr {1e-3, 3e-3, 1e-2}, wd 1, lam 1, at B. Jobs 's2-box5-m6-j<k>', out_stage2/ (wave 3)
    --diag     post hoc positive-control diagnostic P1: 2i and s5 at frac 0.4 (their Stage-1 positive-control seeds
               0, 1, 2), none lr 3e-3 wd 3 and lr 1e-2 wd 1, 200k. Jobs 'diag-pos-j<k>', out_diag/ (wave 4)
    --diag_q   the CONDITIONAL P2 (the coordinator triggers it only if both 2i configs of P1 read 0/3): 2i frac 0.4
               --split question, none lr 1e-2 wd 1, seeds 0, 1, 2, 200k. Job 'diag-q-2i-j0', out_diag/ (wave 4)
    --a3       Amendment A3 (2026-10-25, POST HOC): box5 m6 none at lr {3e-3, 1e-2, 3e-2} x wd {0.03, 0.01, 0}, the Stage-2
               seeds, 4 B (job group a3-box5-m6), and A3-long: lr 1e-2 x wd {0.03, 0.01}, the same seeds, 16 B (group
               a3long-box5-m6); out_a3/ (wave 5)
--wave 1|2 never selects them; submit them with --only <names>."""
import hashlib, json, math, os, shutil, subprocess, sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/scratch/yh6210/llmadd-burst-newmani")
R = Path(os.environ.get("NEWMANI_R", str(B / "runs" / "newmani")))   # NEWMANI_R: a scratch copy for the generator's own tests
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
PIN_SHA = "ea797e5321d3"   # sha256[:12] of the audited tools/toy_struct.py (= code_sha of every result)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
argv = sys.argv[1:]
OPTS = {"--ext", "--submit", "--print", "--status", "--redo", "--stage2", "--diag", "--diag_q", "--a3"}
only = wave = ext_cells = None


def take(flag, check, msg):
    if flag not in argv:
        return None
    i = argv.index(flag)
    assert i + 1 < len(argv) and check(argv[i + 1]), msg
    v = argv[i + 1]; del argv[i: i + 2]
    return v


only = take("--only", lambda v: not v.startswith("--"), "--only needs a comma-separated list of job names")
only = None if only is None else set(only.split(","))
wave = take("--wave", lambda v: v in ("1", "2"), "--wave 1|2")
wave = None if wave is None else int(wave)
NICE = take("--nice", str.isdigit, "--nice N")   # sbatch --nice=N (scheduling only, no effect on runs)
NICE = None if NICE is None else int(NICE)
ext_cells = take("--ext_cells", lambda v: not v.startswith("--"), "--ext_cells needs a comma-separated list like box5-m6,2i-m12")
assert set(argv) <= OPTS, f"unknown option(s) {set(argv) - OPTS}"
EXT, SUBMIT, PRINT, STATUS, REDO = "--ext" in argv, "--submit" in argv, "--print" in argv, "--status" in argv, "--redo" in argv
STAGE2, DIAG, DIAG_Q = "--stage2" in argv, "--diag" in argv, "--diag_q" in argv
A3 = "--a3" in argv
assert not (SUBMIT and DIAG_Q and only is None), "--diag_q renders the CONDITIONAL P2: --submit needs --only <names> (review 2026-10-25)"
assert not (SUBMIT and PRINT), "--submit or --print"
assert not (EXT and ext_cells), "--ext (every cell) or --ext_cells (some cells)"
assert (only is None and wave is None and NICE is None) or SUBMIT or PRINT, "--only / --wave / --nice only with --submit / --print"
assert not REDO or (SUBMIT and only), "--redo only with --submit --only <jobs>"

# ---------------------------------------------------------------------------------------------------- code snapshot
assert sha(REPO / "tools" / "toy_struct.py") == PIN_SHA or (R / "code").exists(), \
    f"worktree tools/toy_struct.py is {sha(REPO / 'tools' / 'toy_struct.py')}, not the pinned {PIN_SHA}"
if not (R / "code").exists():   # once; via a temp dir, so a failed copy leaves no snapshot
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tmp = R / "code.tmp"
    shutil.rmtree(tmp, ignore_errors=True); (tmp / "tools").mkdir(parents=True)
    for f in ("toy_struct.py", "test_toy_struct.py", "toy_grok.py"):
        shutil.copy(REPO / "tools" / f, tmp / "tools" / f)
    for f in ("test_newmani.py", "regress_default.py", "seed_scan.py", "latin_ref_dump.py", "smoke_jobs.py", "b1_toyshape.py",
              "test_generator.py"):
        if (R / "preflight" / f).exists():
            shutil.copy(R / "preflight" / f, tmp / "tools" / f)
    (tmp / "COMMIT").write_text(
        f"worktree /scratch/yh6210/llmadd-burst-newmani, branch newmani-screen at {commit} (= origin/main when created) + "
        f"uncommitted tools/toy_struct.py (sha256[:12] {sha(tmp / 'tools/toy_struct.py')}: newmani box / grp tasks, read-outs, "
        f"checkpoint / resume; opt-in, default path bitwise = d182ce548bcc); test_toy_struct.py {sha(tmp / 'tools/test_toy_struct.py')}; "
        f"toy_grok.py {sha(tmp / 'tools/toy_grok.py')}\n")
    tmp.rename(R / "code")
assert sha(R / "code" / "tools" / "toy_struct.py") == PIN_SHA, ("snapshot is not the pinned file", sha(R / "code/tools/toy_struct.py"))
SCAN = json.loads((R / "preflight" / "seed_scan.json").read_text())
assert SCAN["toy_struct_sha"] == PIN_SHA and SCAN["K"] == 3, ("seed_scan.json from another toy_struct", SCAN["toy_struct_sha"])

# ---------------------------------------------------------------------------------------------------- design
TASKS = {"box5": {"cli": "--task box --box=-2:2,-2:2,-2:2", "true": "expo", "T": 4, "wave": 1,
                  "pos": {"pos": [("1e-3", "1"), ("1e-2", "0.1")], "pos6": [("1e-3", "1"), ("1e-2", "0.1")]}},
         "2i": {"cli": "--task grp --group 2i", "true": "quat4", "T": 3, "wave": 1, "pos": {"pos": [("1e-3", "1")]}},
         "d57": {"cli": "--task grp --group d57", "true": "rho1", "T": 3, "wave": 2, "pos": {"pos": [("1e-3", "1")]}},
         "a5": {"cli": "--task grp --group a5", "true": "rot3", "T": 3, "wave": 2, "pos": {}},
         "s5": {"cli": "--task grp --group s5", "true": "std4", "T": 3, "wave": 2, "pos": {"pos": [("1e-3", "1")]}}}
LEVELS = (6, 12, 24)
BUDGET = 50000                      # B (prior arms); none at 4 B
NONE_LR, NONE_WD = ("1e-3", "3e-3", "1e-2", "3e-2"), ("0.1", "1", "3")
FLAT_EXT = [("1e-1", w) for w in NONE_WD] + [(lr, "10") for lr in NONE_LR]   # rule 2 (toy_struct.WALL_FLAT_EXT)
PRIOR_LR, PRIOR_WD = ("1e-3", "3e-3", "1e-2"), "1"
ROLES = lambda t: [("assign", f"{t}@always"), ("specrand", "specrand@always"), ("shuf", "shuf@always"),
                   ("shape", f"shape:{t}@always"), ("sphere", "shape:sphere@always"), ("line", "shape:line@always")]
PROCS = 5                           # lanes (single-thread processes) per L40S job
PROCS_OF = {"box5-pos": 6,          # box5's 12 positive-control runs: one job, 6 lanes (one f 0.4 + one f 0.6 run each)
            "diag-pos": 6}          # 2026-10-25: the 12 P1 diagnostic runs: 2 jobs x 6 lanes, one run per lane
TARGET_H = 6.0                      # planned hours per lane (packing target)
TARGET_OF = {"diag-pos": 2.5,       # 2026-10-25: P1 in 2 short jobs (the pipeline-vs-config question gates the 2i / s5 reading)
             "a3-box5-m6": 2.5, "a3long-box5-m6": 2.0}   # 2026-10-25 A3: 2 + 2 jobs (one 16 B run per lane)
CKPT_EVERY = 10000
ALL_GROUPS = [f"{t}-m{m}" for t in TASKS for m in LEVELS]
if ext_cells is not None:
    ext_cells = set(ext_cells.split(","))
    assert ext_cells <= set(ALL_GROUPS), f"--ext_cells: unknown cells {sorted(ext_cells - set(ALL_GROUPS))} (valid: {ALL_GROUPS})"
# extension groups: requested now, plus every group with a submitted extension job (sticky: the guard keeps them identical)
EXT_GROUPS = set(ALL_GROUPS) if EXT else set(ext_cells or ())
EXT_GROUPS |= {g for g in ALL_GROUPS if list(R.glob(f"SUBMITTED_{g}-ext-j*"))}


def ms_step(rows_eff: float, shape: bool) -> float:
    """Planning L40S rate, ms/step/process with 5 processes per GPU, fitted to toyshape's measured mod cells (n_train 638 ->
    4.71, 1276 -> 7.66, 1915 -> 11.24, 2553 -> 14.63, 3830 -> 22.40; eval over the 12.8k-question table every 100 steps
    included): 1.2 + 5.54e-3 x rows (over-predicts every point by <= 0.7). rows_eff = n_train x T / 3 (box T 4). Shape
    arms +1.0 (toyshape: +0.6 to +1.5)."""
    return 1.2 + 5.54e-3 * rows_eff + (1.0 if shape else 0.0)


def flags(args):
    t, d = args.split(), {}
    for i, w in enumerate(t):
        if w.startswith("--"):
            k, eq, v = w.partition("=")
            assert k not in d, f"{k} given twice: {args}"
            d[k] = v if eq else (t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True)
    return d


units = []   # (job group, out dir, arm, seed, steps, est hours, cli args)


def add(group, task, frac, seeds, rows, cellname, arm, lr, wd, steps, prior, root="out", split="class"):
    assert root in ("out", "out_stage2", "out_diag", "out_a3") and split in ("class", "question"), (root, split)
    out = R / root / cellname
    cfg = TASKS[task]
    for s in seeds:
        a = (f"{cfg['cli']} --frac {frac} --split {split} --steps {steps} --arms {arm} --lr {lr} --wd {wd}" + (" --lam 1" if prior else "")
             + f" --shape_log --eval_every 100 --threads 1 --out {out} --ckpt_dir {out}/ckpt --ckpt_every {CKPT_EVERY} --seeds {s}")
        want = {**flags(cfg["cli"]), "--frac": str(frac), "--split": split, "--steps": str(steps), "--arms": arm, "--lr": lr,
                "--wd": wd, **({"--lam": "1"} if prior else {}), "--shape_log": True, "--eval_every": "100", "--threads": "1",
                "--out": str(out), "--ckpt_dir": f"{out}/ckpt", "--ckpt_every": str(CKPT_EVERY), "--seeds": str(s)}
        assert flags(a) == want, (cellname, arm, flags(a), want)
        units.append((group, out, arm, s, steps, steps * ms_step(rows[s], ":" in arm) / 3.6e6, a))


for task, cfg in TASKS.items():
    for lvl in [f"m{m}" for m in LEVELS] + list(cfg["pos"]):
        cell = SCAN["cells"][f"{task}|{lvl[1:] if lvl.startswith('m') else lvl}"]
        frac, seeds = cell["frac"], [x["seed"] for x in cell["seeds"]]
        rows = {x["seed"]: x["n_train"] * cfg["T"] / 3 for x in cell["seeds"]}
        group = f"{task}-{lvl}" if lvl not in cfg["pos"] else f"{task}-pos"   # a task's positive controls pack together
        if lvl in cfg["pos"]:
            for lr, wd in cfg["pos"][lvl]:
                add(group, task, frac, seeds, rows, f"{task}-{lvl}-f{frac}-none-lr{lr}-wd{wd}", "none", lr, wd, 4 * BUDGET, False)
            continue
        for lr, wd in [(lr, wd) for lr in NONE_LR for wd in NONE_WD]:
            add(group, task, frac, seeds, rows, f"{task}-{lvl}-f{frac}-none-lr{lr}-wd{wd}", "none", lr, wd, 4 * BUDGET, False)
        if group in EXT_GROUPS:   # append-only: the extension runs live in their own group / jobs
            for lr, wd in FLAT_EXT:
                add(f"{group}-ext", task, frac, seeds, rows, f"{task}-{lvl}-f{frac}-none-lr{lr}-wd{wd}", "none", lr, wd, 4 * BUDGET, False)
        for lr in PRIOR_LR:
            for role, arm in ROLES(cfg["true"]):
                add(group, task, frac, seeds, rows, f"{task}-{lvl}-f{frac}-prior-lr{lr}-wd{PRIOR_WD}", arm, lr, PRIOR_WD, BUDGET, True)
# ---------------------------------------------------------------------------------------------------- Stage 2 + diagnostics
# (2026-10-25, user approval; AUDIT.md "Stage 2 (box5 m6, Tier L)" and "Diagnostics (post hoc)"). Append-only: their own
# groups / jobs / out roots; rendered only with --stage2 / --diag / --diag_q, and on every call once one of their jobs is
# SUBMITTED (sticky, like the extension), so the append-only guard keeps them identical.
S2J = json.loads((R / "preflight" / "stage2_seeds.json").read_text())
assert S2J["toy_struct_sha"] == PIN_SHA, ("stage2_seeds.json from another toy_struct", S2J["toy_struct_sha"])
S2_CELL = S2J["cells"]["box5|6"]
S2_SEEDS = [x["seed"] for x in S2_CELL["seeds"]]
assert S2_SEEDS == [100, 103, 108, 122, 126] and S2_CELL["frac"] == SCAN["cells"]["box5|6"]["frac"] == 0.067, (S2_SEEDS, S2_CELL["frac"])
assert not set(S2_SEEDS) & {x["seed"] for x in SCAN["cells"]["box5|6"]["seeds"]}, "a Stage-2 seed is a Stage-1 seed"
assert all(x["closure"] == 1.0 and x["nullity"] == 3 for x in S2_CELL["seeds"]), "a Stage-2 split fails the seed rule"
S2_ROLES = [("assign", "expo@always"), ("specrand", "specrand@always"), ("shuf", "shuf@always")]   # Tier L only
DIAG_CFG = [("3e-3", "3"), ("1e-2", "1")]       # results audit P1
DIAGQ_CFG, DIAGQ_SEEDS = ("1e-2", "1"), [0, 1, 2]   # results audit P2 (conditional; question split)
NEW_GROUPS = {"s2-box5-m6": 3, "diag-pos": 4, "diag-q-2i": 4, "a3-box5-m6": 5, "a3long-box5-m6": 5}   # group -> wave (never --wave 1|2)
A3_GRID = [(lr, wd) for lr in ("3e-3", "1e-2", "3e-2") for wd in ("0.03", "0.01", "0")]   # AUDIT.md Amendment A3
A3L_GRID, A3L_STEPS = [("1e-2", "0.03"), ("1e-2", "0.01")], 16 * BUDGET                    # A3-long: 16 B = 800k
sticky = lambda g: bool(list(R.glob(f"SUBMITTED_{g}-j*")))
S2_ON, DIAG_ON, DIAGQ_ON = STAGE2 or sticky("s2-box5-m6"), DIAG or sticky("diag-pos"), DIAG_Q or sticky("diag-q-2i")
A3_ON, A3L_ON = A3 or sticky("a3-box5-m6"), A3 or sticky("a3long-box5-m6")
if S2_ON:
    fr2 = S2_CELL["frac"]
    rows2 = {x["seed"]: x["n_train"] * TASKS["box5"]["T"] / 3 for x in S2_CELL["seeds"]}
    for lr, wd in [(lr, wd) for lr in NONE_LR for wd in NONE_WD] + FLAT_EXT:
        add("s2-box5-m6", "box5", fr2, S2_SEEDS, rows2, f"box5-m6-f{fr2}-none-lr{lr}-wd{wd}", "none", lr, wd, 4 * BUDGET, False,
            root="out_stage2")
    for lr in PRIOR_LR:
        for role, arm in S2_ROLES:
            add("s2-box5-m6", "box5", fr2, S2_SEEDS, rows2, f"box5-m6-f{fr2}-prior-lr{lr}-wd{PRIOR_WD}", arm, lr, PRIOR_WD, BUDGET, True,
                root="out_stage2")
if A3_ON or A3L_ON:   # Amendment A3: the Stage-2 cell and seeds, none only; every other flag as the Stage-2 none runs
    fr2 = S2_CELL["frac"]
    rows2 = {x["seed"]: x["n_train"] * TASKS["box5"]["T"] / 3 for x in S2_CELL["seeds"]}
    assert not set(A3_GRID) & set([(lr, wd) for lr in NONE_LR for wd in NONE_WD] + FLAT_EXT), "an A3 config repeats a Stage-2 one"
    if A3_ON:
        for lr, wd in A3_GRID:
            add("a3-box5-m6", "box5", fr2, S2_SEEDS, rows2, f"box5-m6-f{fr2}-none-lr{lr}-wd{wd}", "none", lr, wd, 4 * BUDGET, False, root="out_a3")
    if A3L_ON:
        for lr, wd in A3L_GRID:
            add("a3long-box5-m6", "box5", fr2, S2_SEEDS, rows2, f"box5-m6-f{fr2}-none-lr{lr}-wd{wd}-16B", "none", lr, wd, A3L_STEPS, False,
                root="out_a3")
if DIAG_ON:
    for task in ("2i", "s5"):
        cell = SCAN["cells"][f"{task}|pos"]
        dseeds = [x["seed"] for x in cell["seeds"]]
        assert dseeds == [0, 1, 2] and cell["frac"] == 0.4, (task, dseeds, cell["frac"])   # the Stage-1 positive control's split
        rows = {x["seed"]: x["n_train"] * TASKS[task]["T"] / 3 for x in cell["seeds"]}
        for lr, wd in DIAG_CFG:
            add("diag-pos", task, cell["frac"], dseeds, rows, f"{task}-pos-f{cell['frac']}-none-lr{lr}-wd{wd}", "none", lr, wd,
                4 * BUDGET, False, root="out_diag")
if DIAGQ_ON:
    nq = int(0.4 * 120 * 120)   # toy_struct.split question mode: int(frac * N) training questions; 2I's table is 120 x 120
    add("diag-q-2i", "2i", 0.4, DIAGQ_SEEDS, {s: nq for s in DIAGQ_SEEDS}, f"2i-pos-f0.4-question-none-lr{DIAGQ_CFG[0]}-wd{DIAGQ_CFG[1]}",
        "none", *DIAGQ_CFG, 4 * BUDGET, False, root="out_diag", split="question")
# every (out dir, arm, seed) exactly once (two processes writing one run JSON / checkpoint would race)
keys = [(str(u[1]), u[2], u[3]) for u in units]
assert len(keys) == len(set(keys)), "a run is rendered twice"
n_pos = sum(3 * len(v) for c in TASKS.values() for v in c["pos"].values())
n_expected = len(TASKS) * len(LEVELS) * 3 * (12 + 6 * 3) + n_pos + 3 * 7 * len(EXT_GROUPS)
n_expected += S2_ON * 5 * (12 + 7 + 3 * len(S2_ROLES)) + DIAG_ON * 2 * len(DIAG_CFG) * 3 + DIAGQ_ON * len(DIAGQ_SEEDS)
n_expected += A3_ON * 5 * len(A3_GRID) + A3L_ON * 5 * len(A3L_GRID)
assert len(units) == n_expected, (len(units), n_expected)
assert all((u[0] in NEW_GROUPS) == (u[1].parent.name != "out") for u in units), "Stage-2 / diagnostic runs only in their own roots"


def hours(h):
    return max(2, math.ceil(2 * h))   # --time = 2 x planned, >= 2 h


def job_body(name, lanes, plan_h, comment):
    limit_h = hours(plan_h)
    assert limit_h <= 47, (name, limit_h)
    body = [f"""#!/bin/bash
#SBATCH --job-name=llmadd_newmani-{name}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={limit_h:02d}:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# {comment}; planned {plan_h:.2f} h (L40S rate model), --time = 2 x planned; intent spec + PRE-REGISTERED reading runs/newmani/AUDIT.md
set -u
# kill switch (no scancel from the agent): a pending job exits at start when runs/newmani/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
# the snapshot must still be the pinned trainer (= code_sha of every result); exit before any GPU work otherwise
[ "$(sha256sum toy_struct.py | cut -c1-12)" = "{PIN_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader)
mkdir -p {R}/gpu
echo "$(date -Is) job ${{SLURM_JOB_ID:-none}} restart ${{SLURM_RESTART_COUNT:-0}} host $(hostname) gpu $gpu" >> {R}/gpu/{name}.txt
case "$gpu" in *L40S*) ;; *) echo "not an L40S: $gpu (criterion f: one device type for every arm)"; exit 4;; esac
# deadline: no new run starts after it, and a running one checkpoints at its next eval boundary (toy_struct --deadline);
# runs left unfinished make the job end COMPLETED with runs missing: make_jobs_newmani.py --status lists it, --redo resubmits
DL=$(( $(date +%s) + {limit_h * 3600} - 900 ))
"""]
    dirs = sorted({str(u[1]) for ln in lanes for u in ln})
    body.append("mkdir -p " + " ".join(dirs) + "\n")
    body.append('pids=""\n')
    for i, ln in enumerate(lanes):
        cmds = []
        for (_, out, arm, s, steps, est, a) in ln:
            js = f"{out}/run_{arm}_seed{s}.json"
            cmds.append(f'  if [ ! -e "{js}" ] && [ "$(date +%s)" -lt "$DL" ]; then {PY} toy_struct.py {a} --deadline "$DL" '
                        f'>> "{out}/log_{arm}_seed{s}.txt" 2>&1 || r=1; fi')
        body.append(f"# lane {i}: {len(ln)} runs, planned {sum(u[5] for u in ln):.2f} h\n(\n  r=0\n" + "\n".join(cmds) + "\n  exit $r\n) &\npids=\"$pids $!\"\n")
    body.append("""fail=0
for p in $pids; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo "a run failed: see the log_<arm>_seed<s>.txt files of this job's cells"; exit 1; }
""")
    return "".join(body), limit_h


jobs = []   # (name, body, planned h, limit h, wave, units)
for group in dict.fromkeys(u[0] for u in units):
    procs = PROCS_OF.get(group, PROCS)
    slow = procs / PROCS   # the rate model is for 5 processes sharing a GPU; more processes run proportionally slower
    us = sorted([(*u[:5], u[5] * slow, u[6]) for u in units if u[0] == group], key=lambda u: (-u[5], str(u[1]), u[2], u[3]))
    tot = sum(u[5] for u in us)
    nj = max(1, math.ceil(tot / (procs * TARGET_OF.get(group, TARGET_H))))
    lanes = [[] for _ in range(nj * procs)]
    load = [0.0] * len(lanes)
    for u in us:   # LPT: the largest run to the least loaded lane
        k = min(range(len(lanes)), key=lambda i: (load[i], i))
        lanes[k].append(u); load[k] += u[5]
    task = group.split("-")[0]
    stage = ("Stage 1" if group not in NEW_GROUPS else "Stage 2 (Tier L, pre-registered seeds; A2)" if group.startswith("s2-")
             else "Amendment A3 (POST HOC, low-wd none)" if group.startswith("a3") else "POST HOC diagnostic (not part of the pre-registered reading)")
    for j in range(nj):
        ln = [sorted(lanes[j * procs + i], key=lambda u: (u[2] != "none", str(u[1]), u[2], u[3])) for i in range(procs)]
        ln = [x for x in ln if x]
        plan = max(sum(u[5] for u in x) for x in ln)
        name = f"{group}-j{j}"
        body, lim = job_body(name, ln, plan, f"newmani {stage} {group}, job {j + 1}/{nj}, {sum(len(x) for x in ln)} runs in {len(ln)} lanes")
        jobs.append((name, body, plan, lim, NEW_GROUPS[group] if group in NEW_GROUPS else TASKS[task]["wave"], [u for x in ln for u in x]))
names = [j[0] for j in jobs]
assert len(names) == len(set(names))
assert sorted((str(u[1]), u[2], u[3]) for j in jobs for u in j[5]) == sorted(keys), "every run in exactly one job"
# ---- append-only guard (audit I5): nothing is written if a submitted job's file would change or disappear
submitted = {p.name[len("SUBMITTED_"):] for p in R.glob("SUBMITTED_*")}
conflict = [n for n, body, *_ in jobs if n in submitted and (R / f"job_{n}.sbatch").exists() and (R / f"job_{n}.sbatch").read_text() != body]
gone = sorted(submitted - set(names))
assert not conflict and not gone, (f"REFUSED (append-only): this call would change the files of submitted jobs {conflict} or drop "
                                   f"submitted jobs {gone}; render extensions with --ext_cells into new jobs, never rewrite submitted ones")
for n, body, *_ in jobs:
    p = R / f"job_{n}.sbatch"
    if not p.exists() or p.read_text() != body:
        p.write_text(body)
for p in R.glob("job_*.sbatch"):   # stale job files of another option set (never submitted ones only)
    if p.stem[4:] not in names and p.stem[4:] not in submitted:
        p.unlink()
if only is not None:
    assert only <= set(names), f"--only: unknown job names {sorted(only - set(names))}"
sel = [j for j in jobs if (only is None or j[0] in only) and (wave is None or j[4] == wave)]


def queued_names():
    """Names of this user's jobs in the Slurm queue (pending / running / requeued); None if squeue fails.
    NEWMANI_FAKE_SQUEUE (comma list): the generator's own tests only."""
    if os.environ.get("NEWMANI_FAKE_SQUEUE") is not None:
        return set(os.environ["NEWMANI_FAKE_SQUEUE"].split(",")) - {""}
    o = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"], capture_output=True, text=True)
    return None if o.returncode else {x.strip() for x in o.stdout.splitlines()}


def slurm_state(name):
    """Latest Slurm state of a submitted job (from the job ids in its SUBMITTED_ flag; sacct), or None."""
    ids = [w for w in (R / f"SUBMITTED_{name}").read_text().split() if w.isdigit()]
    if not ids:
        return None
    o = subprocess.run(["sacct", "-j", ids[-1], "-X", "-n", "-P", "-o", "State"], capture_output=True, text=True)
    st = o.stdout.strip().splitlines()
    return st[-1].split()[0] if o.returncode == 0 and st else None


done_of = lambda us: sum((u[1] / f"run_{u[2]}_seed{u[3]}.json").exists() for u in us)
if STATUS:
    ended_missing = []
    for name, body, plan, lim, wv, us in jobs:
        done = done_of(us)
        ck = sum((u[1] / "ckpt" / f"ckpt_{u[2]}_seed{u[3]}.pt").exists() for u in us)
        st = slurm_state(name) if name in submitted else "not submitted"
        print(f"{name}: {done}/{len(us)} runs done, {ck} checkpointed partial, Slurm {st}")
        if st not in (None, "not submitted", "PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED") and done < len(us):
            ended_missing.append((name, st, done, len(us)))
    print("\nENDED WITH RUNS MISSING (a deadline exit 0 ends COMPLETED; resubmit: --submit --only <names> --redo):")
    for name, st, done, n in ended_missing:
        print(f"  {name}: Slurm {st}, {done}/{n} runs done")
    if not ended_missing:
        print("  none")
if PRINT or SUBMIT:
    q = queued_names()
    assert q is not None or PRINT, "squeue failed: refusing to submit (cannot check for queued copies)"
    for name, body, plan, lim, wv, us in sel:
        flag = R / f"SUBMITTED_{name}"
        done = done_of(us)
        if flag.exists() and not (REDO and done < len(us)):   # --redo: resubmit a submitted job whose runs are incomplete
            continue
        if q is not None and f"llmadd_newmani-{name}" in q:   # audit I5: never two copies of one job
            print(f"skip {name}: already in the queue")
            continue
        if PRINT:
            print("would submit", name, f"(planned {plan:.2f} h, --time {lim} h)" + (f" --nice={NICE}" if NICE is not None else ""))
        else:
            o = subprocess.run(["sbatch"] + ([f"--nice={NICE}"] if NICE is not None else []) + [str(R / f"job_{name}.sbatch")],
                               capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text((flag.read_text() if flag.exists() else "") + o.stdout)
                print("submitted", name, o.stdout.strip(), f"({done}/{len(us)} runs already done)")
            else:
                print(name, o.stderr.strip())
gh = lambda js: sum(j[2] for j in js)
for wv in (1, 2):
    js = [j for j in jobs if j[4] == wv]
    print(f"wave {wv}: {len(js)} jobs, {sum(len(j[5]) for j in js)} runs, planned {gh(js):.1f} L40S-h (sum of job times), "
          f"--time sum {sum(j[3] for j in js)} h; at the 16-GPU cap ~{gh(js) / 16:.1f} h wall if nothing else of ours runs")
for wv, lab in ((3, "Stage 2 (box5 m6 Tier L)"), (4, "post hoc diagnostics"), (5, "Amendment A3 (low-wd none, post hoc)")):   # printed only when rendered
    js = [j for j in jobs if j[4] == wv]
    if js:
        pr = sum(u[5] for j in js for u in j[5])   # process-hours (rate model; PROCS_OF slow-down included)
        print(f"wave {wv} = {lab}: {len(js)} jobs {[j[0] for j in js]}, {sum(len(j[5]) for j in js)} runs, planned {gh(js):.1f} L40S-h "
              f"(sum of job times; {pr:.1f} process-h), --time sum {sum(j[3] for j in js)} h; seeds "
              + "; ".join(f"{g}: {sorted({u[3] for j in js for u in j[5] if j[0].startswith(g)})}" for g in NEW_GROUPS
                          if any(j[0].startswith(g) for j in js)))
print(f"total: {len(jobs)} jobs, {len(units)} runs (extension groups {sorted(EXT_GROUPS) or 'none'}); planned {gh(jobs):.1f} L40S-h; "
      "per task: " + ", ".join(f"{t} {gh([j for j in jobs if j[0].startswith(t + '-')]):.1f}" for t in TASKS))
