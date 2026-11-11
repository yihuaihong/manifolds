"""Torch jobs (2026-10-19, wall 2 = inductive bias / sample-complexity threshold; user: discussion section "which walls a
prior breaks", WRITING_HANDOFF.md 2026-10-19): does a SHAPE-ONLY prior (permutation-invariant over the items: the items'
centred-Gram spectrum + norm profile [+ row profile], never which item sits where) let the one-layer toy learn where
none cannot? Intent spec, PRE-REGISTERED rule, costs: runs/toyshape/AUDIT.md (revised after the independent audit of
2026-10-19, before any run). NOT submitted by the agent (--submit is the user's).

Code: tools/toy_struct.py with the opt-in wall-2 additions (shape:<target> / shapeprof:<target> arms, the 'line' and
'sphere' controls, --shape_log, --mod_twins, --summarize_wall; every pre-existing path bitwise identical to b74a78cb5472,
regression in runs/toyshape/preflight). The file is uncommitted (public repo, no agent commits), so the snapshot is a
copy pinned by sha256 (PIN_SHA): the generator refuses a repo file / snapshot with another sha, and every job re-checks
the snapshot before any GPU work.

Cells (seeds 0-4; S4 0.2 uses 0,1,2,4,5 because seed 3's split leaves an item that is never an answer in training --
toy_struct's split assert -- rule "the first five seeds whose split passes", AUDIT.md):
  mod p 113 (question split), train fraction {0.05, 0.1, 0.15, 0.2, 0.3}; prior budget B 25k, none 4 B = 100k.
  s4 (question split), {0.2, 0.3, 0.45}; B 50k, none 200k.
  none: lr {1e-3, 3e-3, 1e-2, 3e-2} x wd {0.1, 1, 3} (one job per (task, frac, lr); the 3 wd run in sequence per seed).
  --ext: the pre-registered one-step extension of a FLAT cell (rule 2: lr 1e-1 x wd {0.1, 1, 3} and wd 10 x the 4 lrs)
    queued up front for the likely wall cells EXT_CELLS (open decision 3); without it the extension is conditional.
  prior arms (lr {1e-3, 1e-2}, wd 1, lam 1), each always-on and in the [0, 2000) window, two jobs per (variant, frac, lr):
    main: shape = 'shape:<true>' (spec + norm); wrongshape = 'shape:line' (same rank, spectrum ~ 1/k^2); assign = the
          existing CKA prior to the fixed target (mod 'always' / 'w0' = toy_grok's; s4 'R@always' / 'R');
    ctl:  rankctl = 'shape:sphere' (flat, constant norm, rank r' = mod r + 1 / s4 12); shapeprof = 'shapeprof:<true>'
          (spec + norm + row profile; default since the audit, --noprof drops it).
  mod variants: circle (--freqs 1) and torus (--freqs 1,2,3,4,5, F59).
Every run: --shape_log, --eval_every 100, --threads 1, 5 seed processes per GPU; every mod run: --mod_twins (test_fresh =
held-out questions whose commuted twin is not in training; the split stays toy_grok's question split, AUDIT.md).
Conditional cells (bracket extension beyond a non-flat edge, PRE-REGISTERED): the NONE_EXT block below, not generated.
Account <slurm-account>, l40s_public, gpu48. Kill switch: runs/toyshape/HOLD. --submit submits jobs not yet
submitted; --only name1,name2 restricts --submit to those jobs (calibrators first, AUDIT.md "Cost").

--cpu (2026-10-20, AUDIT.md amendment "CPU move", before any result): the SAME runs (cells, arms, seeds, lrs, wds,
budgets, --ext, every toy_struct flag) as CPU jobs in runs/toyshape_cpu, a separate dir (own HOLD, SUBMITTED_ flags,
out/; code/ = a byte copy of runs/toyshape/code, sha pinned). The GPU path (no --cpu) renders byte-identical job files.
  - one toy_struct process per run = (cell dir, arm, seed): the GPU job's arm list / wd sequence is split (every run
    re-seeds at its start, so a run does not depend on what ran before it in its process; A/B in toyshape_cpu/preflight);
  - runs are packed (LPT) into lanes = one single-thread process per CPU, runs in sequence; lanes into jobs of LANES[part];
    the GPU calibrator jobs' runs (CALIB) fill the first jobs ("calib-*"), the rest follow ("b*");
  - --part cpu_short (default; QoS cpu_short: MaxWall 6 h, per user cpu 32 / mem 120G): a lane is at most 3 h estimated,
    --time = 2 x the job's longest lane (<= 6 h). A run whose own estimate exceeds 3 h cannot run there: refused, with
    the list (use --part cs). --part cs (QoS cpu48: MaxWall 48 h, per user cpu 3000): lanes up to the longest run;
  - each job: CUDA_VISIBLE_DEVICES='' + an assert that torch sees no GPU; 1-thread BLAS; the CPU model is recorded in
    runs/toyshape_cpu/CPU_MODEL by the first job and every job exits (code 4, before any run, its SUBMITTED_ flag
    removed so that --submit resubmits it) on another model (same hardware for every arm, rescue criterion f).
  - CPU vs GPU numerics differ (float32 matmul paths): toyshape_cpu results are compared only within themselves.
    python make_jobs_toyshape.py [--ext] [--noprof] [--cpu [--part cpu_short|cs]] [--submit [--only a,b]]"""
import hashlib, re, shutil, subprocess, sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/home/yh6210/Research_Projects/llmadd-burst")
RG = B / "runs" / "toyshape"   # the GPU batch (its code snapshot is the source of the CPU snapshot)
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
PIN_SHA = "d182ce548bcc"   # sha256[:12] of the audited tools/toy_struct.py (= code_sha of every result)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
OPTS = {"--ext", "--noprof", "--submit", "--only", "--cpu"}
argv = sys.argv[1:]
only = None
if "--only" in argv:
    i = argv.index("--only")
    assert i + 1 < len(argv) and not argv[i + 1].startswith("--"), "--only needs a comma-separated list of job names"
    only = set(argv[i + 1].split(",")); del argv[i: i + 2]
    assert "--submit" in argv, "--only only restricts --submit"
PART = None
if "--part" in argv:
    i = argv.index("--part")
    assert i + 1 < len(argv) and argv[i + 1] in ("cpu_short", "cs"), "--part needs cpu_short or cs"
    PART = argv[i + 1]; del argv[i: i + 2]
    assert "--cpu" in argv, "--part is a --cpu option"
assert set(argv) <= OPTS, f"unknown option(s) {set(argv) - OPTS}"
EXT, PROF, SUBMIT, CPU = "--ext" in argv, "--noprof" not in argv, "--submit" in argv, "--cpu" in argv
PART = PART or ("cpu_short" if CPU else None)
R = B / "runs" / "toyshape_cpu" if CPU else RG
if CPU and not (R / "code").exists():   # once; a byte copy of the GPU batch's snapshot, via a temp dir
    src = RG / "code"
    assert sha(src / "tools" / "toy_struct.py") == PIN_SHA, ("GPU snapshot is not the pinned file", sha(src / "tools/toy_struct.py"))
    tmp = R / "code.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("__pycache__"))
    for f in sorted(src.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            assert sha(tmp / f.relative_to(src)) == sha(f), f
    (tmp / "COMMIT").write_text((src / "COMMIT").read_text() + "2026-10-20: copied unchanged (every file sha256-equal) from "
                                f"runs/toyshape/code for the CPU move (AUDIT.md amendment 'CPU move'); toy_struct.py {PIN_SHA}\n")
    tmp.rename(R / "code")
assert CPU or sha(REPO / "tools" / "toy_struct.py") == PIN_SHA or (R / "code").exists(), \
    f"repo tools/toy_struct.py is {sha(REPO / 'tools' / 'toy_struct.py')}, not the pinned {PIN_SHA}"
if not (R / "code").exists():   # once; via a temp dir, so a failed copy leaves no snapshot
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tmp = R / "code.tmp"
    shutil.rmtree(tmp, ignore_errors=True); (tmp / "tools").mkdir(parents=True)
    for f in ("toy_struct.py", "test_toy_struct.py", "toy_grok.py"):
        shutil.copy(REPO / "tools" / f, tmp / "tools" / f)
    pre = R / "preflight"
    for f in ("test_toyshape.py", "regress_default.py", "double_path.py"):
        if (pre / f).exists():
            shutil.copy(pre / f, tmp / "tools" / f)
    (tmp / "COMMIT").write_text(f"{commit} + uncommitted tools/toy_struct.py (sha256[:12] {sha(tmp / 'tools/toy_struct.py')}: wall-2 "
                                f"shape arms + line / sphere controls, opt-in); test_toy_struct.py {sha(tmp / 'tools/test_toy_struct.py')}; "
                                f"toy_grok.py {sha(tmp / 'tools/toy_grok.py')}\n")
    tmp.rename(R / "code")
assert sha(R / "code" / "tools" / "toy_struct.py") == PIN_SHA, ("snapshot is not the pinned file", sha(R / "code/tools/toy_struct.py"))

SEEDS = {("s4", 0.2): "0,1,2,4,5"}   # default "0,1,2,3,4"
TASKS = {"mod": {"B": 25000, "fracs": (0.05, 0.1, 0.15, 0.2, 0.3), "variants": {"circle": "1", "torus": "1,2,3,4,5"},
                 "split": "question"},
         "s4": {"B": 50000, "fracs": (0.2, 0.3, 0.45), "variants": {"R": None}, "split": "question"}}
NONE_LR, NONE_WD = ("1e-3", "3e-3", "1e-2", "3e-2"), ("0.1", "1", "3")
PRIOR_LR, PRIOR_WD = ("1e-3", "1e-2"), "1"
ROLES = {"mod": ["shape:circle@always", "shape:line@always", "always", "shape:circle", "shape:line", "w0"],
         "s4": ["shape:R@always", "shape:line@always", "R@always", "shape:R", "shape:line", "R"]}
CTL = {t: ["shape:sphere@always", "shape:sphere"] for t in TASKS}
PROF_ARMS = {"mod": ["shapeprof:circle@always", "shapeprof:circle"], "s4": ["shapeprof:R@always", "shapeprof:R"]}
MODX = {"mod": "--freqs 1 --mod_twins "}   # mod none: toy_grok's circle for the logged cka_circle; test_fresh = twin-free held out
# rule 2's one-step extension of a FLAT cell (= toy_struct.WALL_FLAT_EXT): job name suffix -> configs (lr, wd)
FLAT_EXT = {"lr1e-1": [("1e-1", w) for w in NONE_WD], "wd10": [(lr, "10") for lr in NONE_LR]}
EXT_CELLS = [("mod", 0.05), ("mod", 0.1), ("mod", 0.15), ("s4", 0.2), ("s4", 0.3)]   # the likely wall cells (AUDIT.md)
# CONDITIONAL (PRE-REGISTERED, AUDIT.md rule 2; NOT generated -- append only when the rule has fired on a finished none
# sweep): (task, frac, name suffix, [(lr, wd), ...]) none cells one x3 step beyond the edge the best config sits on, e.g.
# NONE_EXT = [("s4", 0.3, "lr1e-1", [("1e-1", w) for w in NONE_WD])]   # best none at lr 3e-2 above the floor
NONE_EXT = []
# planning rates, ms per step per process with 5 processes sharing one L40S (AUDIT.md "Cost"): s4 measured 3.0 (none) /
# 3.7 (CKA prior) in runs/toynone2, toyprior_lr; mod and the shape arms' CPU eigvalsh are NOT measured on a GPU node yet
MS = {"none": {"mod": 6.0, "s4": 3.0}, "prior": {"mod": 7.5, "s4": 4.5}}
WORST = {"mod": 25.0, "s4": 10.0}   # time limits = 1.5 x this per-step rate (mod: toy_grok on A100 measured 22.9; s4: 2.7x the CKA arm)
from math import isclose  # noqa: E402
assert all(isclose(float(a), b) and isclose(float(w), c) for (a, w), (b, c) in
           zip(FLAT_EXT["lr1e-1"] + FLAT_EXT["wd10"], [(1e-1, x) for x in (0.1, 1.0, 3.0)] + [(x, 10.0) for x in (1e-3, 3e-3, 1e-2, 3e-2)]))
# ---- --cpu only (2026-10-20). CPU rate = ms of process CPU time per step on one core (setup CPU subtracted), measured
# on the session host cl014 (Xeon Platinum 8592+), toy_struct d182ce548bcc, --threads 1, 1-thread BLAS, 2 processes at
# once: runs/toyshape_cpu/preflight/calib.log. Per (task, kind) {frac: ms}, kind = none / cka (assign) / shape /
# shapeprof; the max over the arms and lrs measured of that kind; window arms take their @always rate (conservative).
CPU_MS = None   # filled from the calibration below
GPU_CALIB = ("mod-torus-f0.05-prior-lr1e-2", "mod-torus-f0.05-prior-lr1e-2-ctl", "mod-f0.05-none-lr1e-2", "s4-R-f0.2-prior-lr1e-2")
QOS = {"cpu_short": {"wall_h": 6, "cpu": 32, "mem_g": 120}, "cs": {"wall_h": 48, "cpu": 3000, "mem_g": 6000}}   # sacctmgr qos cpu_short / cpu48
LANES = {"cpu_short": 16, "cs": 32}   # single-thread processes (= CPUs) per job
MARGIN = 2.0                          # --time = MARGIN x the job's longest lane estimate, whole hours
MEM_G = 1.5                           # per process (peak RSS measured <= 0.8 GB)


def flags(args):
    t, d = args.split(), {}
    for i, w in enumerate(t):
        if w.startswith("--"):
            assert w not in d, f"{w} given twice: {args}"
            d[w] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True
    return d


def hours(sec):
    """--time in whole hours: 1.5 x the worst-case seconds, at least 1 h."""
    return max(1, -(-int(1.5 * sec) // 3600))


def job(name, lines, limit_h, comment):
    p = R / f"job_{name}.sbatch"
    body = f"""#!/bin/bash
#SBATCH --job-name=llmadd_toyshape-{name}
#SBATCH --account=<slurm-account>
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={limit_h:02d}:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# {comment}; intent spec + PRE-REGISTERED rule runs/toyshape/AUDIT.md; code = runs/toyshape/code (COMMIT)
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/toyshape/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
# the snapshot must still be the pinned trainer (= code_sha of every result); exit before any GPU work otherwise
[ "$(sha256sum toy_struct.py | cut -c1-12)" = "{PIN_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
nvidia-smi --query-gpu=name --format=csv,noheader
{lines}
"""
    p.write_text(body)
    return p


jobs, n_runs, est = [], 0, {}
units = []   # --cpu: one toy_struct process per run


def cpu_ms(task, kind, frac):
    """ms per step on one core: linear in frac between measured fracs; below the smallest measured frac its rate,
    above the largest the line through the two largest (or the only point)."""
    pts = sorted(CPU_MS[(task, kind)].items())
    if frac <= pts[0][0] or len(pts) == 1:
        return pts[0][1] if frac <= pts[0][0] else pts[0][1] * frac / pts[0][0]
    for (f0, m0), (f1, m1) in zip(pts, pts[1:]):
        if frac <= f1:
            return m0 + (m1 - m0) * (frac - f0) / (f1 - f0)
    return m0 + (m1 - m0) * (frac - f0) / (f1 - f0)


def unit(gpu_name, out, arm, args, seed, steps, task, frac, kind, log):
    assert f" --steps {steps} " in f" {args} " and f" --arms {arm} " in f" {args} "
    units.append({"gpu": gpu_name, "out": str(out), "arm": arm, "args": args, "seed": seed, "log": log,
                  "h": steps * cpu_ms(task, kind, frac) / 1000 / 3600, "task": task, "frac": frac, "kind": kind})


def pack(us, cap_h):
    """Best-fit decreasing: each run (longest first) into the fullest lane it still fits (<= cap_h), else a new lane."""
    lanes = []
    for u in sorted(us, key=lambda u: -u["h"]):
        fit = [l for l in lanes if l[0] + u["h"] <= cap_h + 1e-9]
        lane = max(fit, key=lambda l: l[0]) if fit else None
        if lane is None:
            lane = [0.0, []]; lanes.append(lane)
        lane[0] += u["h"]; lane[1].append(u)
    return sorted(lanes, key=lambda l: -l[0])


def cpu_job(name, lanes, comment):
    K, top = len(lanes), lanes[0][0]
    limit_h = max(1, -(-int(MARGIN * top * 3600) // 3600))
    assert limit_h <= QOS[PART]["wall_h"] and K <= QOS[PART]["cpu"], (name, limit_h, K)
    mem = -(-int(K * MEM_G * 10) // 10) + 2
    assert mem <= QOS[PART]["mem_g"], (name, mem)
    cells = sorted({u["out"] for l in lanes for u in l[1]})
    lane_lines = []
    for k, (load, us) in enumerate(lanes):
        seq = "; ".join(f"{PY} toy_struct.py {u['args']} --seeds {u['seed']} >> {u['log']} 2>&1 || r=1" for u in us)
        lane_lines.append(f"# lane {k}: {len(us)} runs, {load:.2f} h estimated\n( r=0; {seq}; exit $r ) &\npids=\"$pids $!\"")
    p = R / f"job_{name}.sbatch"
    body = f"""#!/bin/bash
#SBATCH --job-name=llmadd_toyshapecpu-{name}
#SBATCH --account=<slurm-account>
#SBATCH --partition={PART}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={K}
#SBATCH --mem={mem}G
#SBATCH --time={limit_h:02d}:00:00
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# {comment}; intent spec + PRE-REGISTERED rule runs/toyshape/AUDIT.md (amendment "CPU move"); code = runs/toyshape_cpu/code (COMMIT)
# {K} lanes = {K} single-thread processes, each running its runs in sequence; longest lane {top:.2f} h estimated, --time = {MARGIN:g} x
set -eu
# kill switch (no scancel from the agent): a pending job exits at start when runs/toyshape_cpu/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code/tools
# the snapshot must still be the pinned trainer (= code_sha of every result); exit before any run otherwise
[ "$(sha256sum toy_struct.py | cut -c1-12)" = "{PIN_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
# CPU only (AUDIT.md amendment "CPU move"): no GPU visible to any process; 1-thread BLAS (--threads 1 sets torch's pool)
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{PY} -c 'import torch; assert not torch.cuda.is_available(), "a GPU is visible"; print("torch", torch.__version__, "cuda available", torch.cuda.is_available(), flush=True)' || {{ echo "GPU visible or torch broken: exiting"; exit 5; }}
# same hardware for every run (rescue criterion f): the first job records the CPU model, every job must match it
model="$(lscpu | sed -n 's/^Model name: *//p' | head -n 1)"
echo "host $(hostname) nproc $(nproc) cpu model: $model"
[ -n "$model" ] || {{ echo "CPU model unknown: exiting"; exit 4; }}
if [ ! -e {R}/CPU_MODEL ]; then
  printf '%s\\n' "$model" > {R}/.CPU_MODEL.$$ && {{ ln {R}/.CPU_MODEL.$$ {R}/CPU_MODEL 2>/dev/null || true; }}; rm -f {R}/.CPU_MODEL.$$
fi
[ "$(cat {R}/CPU_MODEL)" = "$model" ] || {{ echo "CPU model differs from runs/toyshape_cpu/CPU_MODEL ($(cat {R}/CPU_MODEL)): exiting before any run; SUBMITTED_{name} removed (rerun --submit)"; rm -f {R}/SUBMITTED_{name}; exit 4; }}
[ "$(nproc)" -ge {K} ] || {{ echo "fewer CPUs than lanes: exiting"; exit 6; }}
mkdir -p {' '.join(cells)}
pids=""
{chr(10).join(lane_lines)}
fail=0
for p in $pids; do wait $p || fail=1; done
# per-cell summaries (other jobs write the same cells: serialised; non-fatal; the read-out is --summarize_wall over out/)
for d in {' '.join(cells)}; do
  flock $d/.summarize.lock {PY} toy_struct.py --summarize $d > /dev/null 2>&1 || echo "summarize $d failed (non-fatal)"
done
[ $fail -eq 0 ] || {{ echo "a lane failed"; exit 1; }}
"""
    p.write_text(body)
    return p


def none_job(task, frac, Bp, split, seeds, name, cfgs, comment):
    """One job: 5 seed processes, each running the configs cfgs = [(lr, wd), ...] in sequence."""
    global n_runs
    outs = {c: R / "out" / f"{task}-f{frac}-none-lr{c[0]}-wd{c[1]}" for c in cfgs}
    cmd = {}
    for (lr, wd) in cfgs:
        a = (f"--task {task} --frac {frac} --split {split} --steps {4 * Bp} --arms none --lr {lr} --wd {wd} "
             + MODX.get(task, "") + f"--shape_log --eval_every 100 --threads 1 --out {outs[(lr, wd)]}")
        want = {"--task": task, "--frac": str(frac), "--split": split, "--steps": str(4 * Bp), "--arms": "none", "--lr": lr,
                "--wd": wd, "--shape_log": True, "--eval_every": "100", "--threads": "1", "--out": str(outs[(lr, wd)]),
                **({"--freqs": "1", "--mod_twins": True} if task == "mod" else {})}
        assert flags(a) == want, (name, flags(a), want)
        cmd[(lr, wd)] = a
    if CPU:   # one process per (config, seed) run; log name as on the GPU (one config per cell dir)
        for c in cfgs:
            for s in seeds.split(","):
                unit(name, outs[c], "none", cmd[c], s, 4 * Bp, task, frac, "none", f"{outs[c]}/log_seed{s}.txt")
        n_runs += 5 * len(cfgs)
        return
    per_seed = "; ".join(f"{PY} toy_struct.py {cmd[c]} --seeds $s >> {outs[c]}/log_seed$s.txt 2>&1 || r=1" for c in cfgs)
    mk = " ".join(str(o) for o in outs.values())
    summ = "\n".join(f"{PY} toy_struct.py --summarize {o}" for o in outs.values())
    lines = f"""mkdir -p {mk}
pids=""
for s in {seeds.replace(',', ' ')}; do
  ( r=0; {per_seed}; exit $r ) &
  pids="$pids $!"
done
fail=0
for p in $pids; do wait $p || fail=1; done
{summ}
[ $fail -eq 0 ] || {{ echo "a seed process failed"; exit 1; }}"""
    steps = len(cfgs) * 4 * Bp
    jobs.append((name, job(name, lines, hours(steps * WORST[task] / 1000), comment)))
    est[name] = steps * MS["none"][task] / 1000 / 3600
    n_runs += 5 * len(cfgs)


for task, cfg in TASKS.items():
    Bp, split = cfg["B"], cfg["split"]
    for frac in cfg["fracs"]:
        seeds = SEEDS.get((task, frac), "0,1,2,3,4")
        sl = seeds.split(",")
        assert len(sl) == 5 and len(set(sl)) == 5
        groups = [(f"lr{lr}", [(lr, wd) for wd in NONE_WD], "base") for lr in NONE_LR]   # ---- none sweep
        if EXT and (task, frac) in EXT_CELLS:
            groups += [(k, v, "flat extension, rule 2, queued up front (--ext)") for k, v in FLAT_EXT.items()]
        groups += [(k, v, "CONDITIONAL extension, rule 2") for (t, f, k, v) in NONE_EXT if (t, f) == (task, frac)]
        for suffix, cfgs, why in groups:
            none_job(task, frac, Bp, split, seeds, f"{task}-f{frac}-none-{suffix}", cfgs,
                     f"wall 2 none sweep ({why}), {task} frac {frac} ({split} split), configs "
                     + " ".join(f"lr{a}/wd{b}" for a, b in cfgs) + f", {4 * Bp} steps (= 4 B), seeds {seeds}")
        for var, freqs in cfg["variants"].items():   # ---- prior arms
            for lr in PRIOR_LR:
                out = R / "out" / f"{task}-{var}-f{frac}-lr{lr}-wd{PRIOR_WD}"
                for opt, arms in (("", ROLES[task]), ("-ctl", CTL[task] + (PROF_ARMS[task] if PROF else []))):
                    name = f"{task}-{var}-f{frac}-prior-lr{lr}{opt}"
                    a = (f"--task {task} --frac {frac} --split {split} --steps {Bp} --arms {','.join(arms)} --win 0,2000 --lam 1 "
                         f"--lr {lr} --wd {PRIOR_WD} " + (f"--freqs {freqs} --mod_twins " if task == "mod" else "")
                         + f"--shape_log --eval_every 100 --threads 1 --out {out}")
                    want = {"--task": task, "--frac": str(frac), "--split": split, "--steps": str(Bp), "--arms": ",".join(arms),
                            "--win": "0,2000", "--lam": "1", "--lr": lr, "--wd": PRIOR_WD, "--shape_log": True, "--eval_every": "100",
                            "--threads": "1", "--out": str(out), **({"--freqs": freqs, "--mod_twins": True} if task == "mod" else {})}
                    assert (task == "mod") == (freqs is not None)
                    assert flags(a) == want, (name, flags(a), want)
                    if CPU:   # one process per (arm, seed) run: the arm list split, every other flag as rendered above
                        for arm in arms:
                            a1 = a.replace(f" --arms {','.join(arms)} ", f" --arms {arm} ")
                            assert flags(a1) == {**want, "--arms": arm}, (name, arm)
                            kind = arm.split(":")[0] if ":" in arm else "cka"
                            for s in sl:
                                unit(name, out, arm, a1, s, Bp, task, frac, kind, f"{out}/log{opt}_seed{s}_{arm}.txt")
                        n_runs += 5 * len(arms)
                        continue
                    lines = f"""mkdir -p {out}
pids=""
for s in {seeds.replace(',', ' ')}; do
  {PY} toy_struct.py {a} --seeds $s >> {out}/log{opt}_seed$s.txt 2>&1 &
  pids="$pids $!"
done
fail=0
for p in $pids; do wait $p || fail=1; done
{PY} toy_struct.py --summarize {out}
[ $fail -eq 0 ] || {{ echo "a seed process failed"; exit 1; }}"""
                    steps = len(arms) * Bp
                    jobs.append((name, job(name, lines, hours(steps * WORST[task] / 1000),
                                           f"wall 2 prior arms, {task} {var} frac {frac}, lr {lr} wd {PRIOR_WD}, {Bp} steps (= B), seeds {seeds}, arms {','.join(arms)}")))
                    est[name] = steps * MS["prior"][task] / 1000 / 3600
                    n_runs += 5 * len(arms)
if CPU:   # ---- pack the runs: the GPU calibrator jobs' runs first ("calib-*"), then the rest ("b*")
    assert units and len(units) == n_runs
    gpu_order = list(dict.fromkeys(u["gpu"] for u in units))
    assert set(GPU_CALIB) <= set(gpu_order), GPU_CALIB
    longest = max(u["h"] for u in units)
    cap = QOS[PART]["wall_h"] / MARGIN if PART == "cpu_short" else max(longest, QOS["cpu_short"]["wall_h"] / MARGIN)
    too_long = [u for u in units if u["h"] > cap + 1e-9]
    if too_long:
        by_gpu = {}
        for u in too_long:
            by_gpu.setdefault(u["gpu"], []).append(u["h"])
        sys.exit(f"--part {PART}: {len(too_long)} runs are estimated above the {cap:g} h lane cap (= MaxWall "
                 f"{QOS[PART]['wall_h']} h / margin {MARGIN:g}); toy_struct has no mid-run checkpoint, so they cannot run "
                 "there:\n" + "\n".join(f"  {g}: {len(v)} runs, {min(v):.1f}-{max(v):.1f} h each" for g, v in by_gpu.items())
                 + "\nuse --part cs (QoS cpu48, MaxWall 48 h) for the whole batch (same hardware for every arm)")
    plan = []   # (job name, lanes)
    for tag, us in (("calib", [u for u in units if u["gpu"] in GPU_CALIB]), ("b", [u for u in units if u["gpu"] not in GPU_CALIB])):
        lanes = pack(us, cap)
        for j in range(0, len(lanes), LANES[PART]):
            plan.append((f"{tag}-{j // LANES[PART]:02d}" if tag == "calib" else f"{tag}{j // LANES[PART]:03d}", lanes[j: j + LANES[PART]]))
    for name, lanes in plan:
        gp = sorted({u["gpu"] for l in lanes for u in l[1]}, key=gpu_order.index)
        jobs.append((name, cpu_job(name, lanes, f"wall 2 on CPU ({PART}), {sum(len(l[1]) for l in lanes)} runs of the GPU jobs "
                                    + (" ".join(gp) if len(gp) <= 6 else f"{' '.join(gp[:6])} ... ({len(gp)} GPU jobs)"))))
        est[name] = (len(lanes), lanes[0][0], sum(l[0] for l in lanes))
names = [n for n, _ in jobs]
assert len(names) == len(set(names))
# every (cell dir, arm) is produced by exactly one job (two jobs writing one run JSON would race); --cpu: every
# (cell dir, arm, seed) by exactly one lane, and the set of runs is the GPU batch's (cell dir, arm) x its 5 seeds
outs = {}
for n, p in jobs:
    if CPU:
        found = re.findall(r"--arms (\S+)[^;&\n]*? --out (\S+) --seeds (\d+) ", p.read_text())
        assert found, n
        for a, o, s in found:
            assert "," not in a and (o, a, s) not in outs, ((o, a, s), n, outs.get((o, a, s)))
            outs[(o, a, s)] = n
        continue
    found = re.findall(r"--arms (\S+)[^;&\n]*? --out (\S+) --seeds", p.read_text())
    assert found, n
    for a, o in found:
        for arm in a.split(","):
            assert (o, arm) not in outs, ((o, arm), n, outs.get((o, arm)))
            outs[(o, arm)] = n
if CPU:
    assert set(outs) == {(u["out"], u["arm"], u["seed"]) for u in units} and len(outs) == n_runs, "a run is missing or extra"
for p in R.glob("job_*.sbatch"):   # stale job files of another option set (never submitted ones only)
    if p.stem[4:] not in names and not (R / f"SUBMITTED_{p.stem[4:]}").exists():
        p.unlink()
if only is not None:
    assert only <= set(names), f"--only: unknown job names {sorted(only - set(names))}"
if SUBMIT:
    for name, p in jobs:
        flag = R / f"SUBMITTED_{name}"
        if (only is None or name in only) and not flag.exists():
            o = subprocess.run(["sbatch", str(p)], capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text(o.stdout)
            else:
                print(name, o.stderr.strip())
if CPU:   # CPU-hours, and the wall-clock under the per-user CPU cap (jobs in submission order, a later job starts as
    # soon as its CPUs fit under the cap; node availability / queue wait NOT included)
    by = {}
    for u in units:
        g = f"{u['task']} {'none' if u['kind'] == 'none' else 'prior'}"
        by[g] = by.get(g, 0) + u["h"]
    cap_cpu, t, running, starts = QOS[PART]["cpu"], 0.0, [], []
    for name, _ in jobs:
        K, top, _s = est[name]
        while sum(k for _, k in running) + K > cap_cpu:
            t = min(e for e, _ in running); running = [(e, k) for e, k in running if e > t]
        running.append((t + top, K)); starts.append(t)
    span = max(e for e, _ in running)
    limits = sorted({int(re.search(r"--time=(\d+):", p.read_text()).group(1)) for _, p in jobs})
    print(f"--cpu --part {PART}: {len(jobs)} jobs ({sum(1 for n, _ in jobs if n.startswith('calib'))} calibrator jobs first), "
          f"{sum(est[n][0] for n, _ in jobs)} lanes, {n_runs} runs (prof {PROF}, ext {EXT}); {sum(by.values()):.0f} CPU-h estimated ("
          + ", ".join(f"{k} {v:.0f}" for k, v in sorted(by.items())) + f"); longest run {max(u['h'] for u in units):.2f} h, "
          f"longest lane {max(est[n][1] for n, _ in jobs):.2f} h, --time {limits} h; under the per-user cap of {cap_cpu} CPUs "
          f"the batch takes {span:.1f} h of running time at the estimated rates (x{MARGIN:g} = {MARGIN * span:.1f} h if every run is "
          f"{MARGIN:g}x slower), plus queue waits")
    sys.exit(0)
tot = sum(est.values())
by = {}
for k, v in est.items():
    g = ("none" if "-none-" in k else "prior") + " " + k.split("-")[0] + (" ctl" if k.endswith("-ctl") else "") \
        + (" ext" if k.endswith(("-none-lr1e-1", "-none-wd10")) else "")
    by[g] = by.get(g, 0) + v
print(f"{len(jobs)} jobs, {n_runs} runs (prof {PROF}, ext {EXT}); planning estimate {tot:.1f} L40S-h ("
      + ", ".join(f"{k} {v:.1f}" for k, v in sorted(by.items())) + ")")
