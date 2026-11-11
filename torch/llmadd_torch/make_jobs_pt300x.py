"""Torch jobs pt300x (2026-11-02; the user's decision of 2026-11-02, relayed by the coordinator: two more sizes on F87's
fixed-token ladder, 「可以多用点其他gpu吗，现在越快越好」). Intent spec and PRE-REGISTERED analysis: runs/pt300x/AUDIT.md.
F87's 7% ladder (3e8 tokens, every arm at its own best lr, 8 seeds; 53M / 190M / 505M on L40S, 1.07B on B200) gets two
more sizes at the same 3e8 tokens, the pttpp shapes:
  93M  = d512 x 10, 8 heads, d_ff 2048 (N_nonemb 41,964,032)   on L40S (account 625, l40s_public, gpu48, nice 20)
  294M = d832 x 19, 13 heads, d_ff 3328 (N_nonemb 210,500,160) on B200 (account 520, preemptible, 8 singleton slots
                                                                 llmadd_p3x-b200-slot0..7, ckpt every 500 steps, resume)
Protocol = F87 (ptlr) exactly, except the shape and the lr grid: 3e8 tokens = 18,310 steps of 32 x 512 (the trainer's
default --tokens 3e8, as F87's jobs: no --tokens / --eval_every flag), eval every 250 steps (75 evals), 1% warmup, window
(0.05, 0.4), lambda 5, ctx site, --arith_lines 4 (7%); arms 'none' and 'win-ctx' = F87's flags.
Trainer and job body = pttpp's 7% cells: runs/pt300x/code7 = `git archive 64abc19 tools/` (pretrain_range.py
a3bce3ab63c1), asserted equal to runs/pttpp/code7 (= F87's runs/ptlr/code) file by file; the job body is
make_jobs_pttpp.job7's (HOLD kill switch, trainer sha, GPU-name guard, data sha256, INTENT line, --dry_inputs, training,
CHECK7 verbatim on the run JSON) with this batch's paths, no selection hook, and the B200 header of pttpp's B200 jobs.
The frozen pttpp generator is imported read-only (sha256[:12] and the absence of top-level calls asserted); its args7() /
intent7() are called for every run and their outputs asserted equal to this batch's up to the --out directory.
Stages (AUDIT.md "lr selection"): A = 4 lrs x 2 arms x seeds 0-3 per size (64 runs); select_pt300x.py (by hand) applies
F87 rule 1 and appends X (one ladder step beyond an open edge, seeds 0-3, once per arm), C (confirm seeds 4-7 at the pick)
or STOP to runs/pt300x/decisions.json; this generator then renders those runs too. A submitted job file is never
rewritten (SUBMITTED_<run> flags).
    python make_jobs_pt300x.py --plan                                         design, cost, wall time
    python make_jobs_pt300x.py [--stage A|X|C] [--size S] [--arm A] [--print] [--test-only | --submit]
    python make_jobs_pt300x.py --check [--stage A]                            job files = rendering, snapshot, frozen shas
    python make_jobs_pt300x.py --names [--stage A] [--size S]                 run names in submission order (submit_pt300x.sh)
"""
import ast
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
# read-only inputs (sha256[:12]): the pttpp generator (FROZEN.txt, T8) and what it imports / reads at import time
FROZEN = {"make_jobs_pttpp.py": "b4bfa6eb248d", "make_jobs_ptlr1b.py": "8dd7b3407459", "make_jobs_pt2s505.py": "a5a7fb60eedc",
          "make_jobs_rb6scale.py": "19bc055c2d09", "pttpp_refs.json": "d10d720c1242", "pt2s505_refs.json": "479e394771e9",
          "rb6scale_refs.json": "b65472eb3d29", "runs/pttpp/code7/tools/pretrain_range.py": "a3bce3ab63c1",
          "runs/pttpp/readout/tools/readout_rules.py": "e91ee8a8a846"}
for _f, _v in FROZEN.items():
    assert sha(HERE / _f) == _v, (_f, sha(HERE / _f), _v, "a read-only input changed")
_t = ast.parse((HERE / "make_jobs_pttpp.py").read_text())   # importing it must run no main() and no top-level call
_g = [n for n in _t.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
assert [ast.unparse(n) for n in _t.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)] == \
    ["sys.path.insert(0, str(HERE))"], "make_jobs_pttpp: a top-level call other than its sys.path line"
assert len(_g) == 1 and [ast.unparse(x) for x in _g[0].body] == ["main()"], "make_jobs_pttpp: __main__ guard"
sys.path.insert(0, str(HERE))
import make_jobs_pttpp as M   # noqa: E402  (read-only: SIZES, LADDER, args7, intent7, CHECK7, lm0_ref, the job body)

B = M.B
P = B / "runs" / "pt300x"
TOK, PY, DATA, PIN7, PIN7_SHA = M.TOK, M.PY, M.DATA, M.PIN7, M.PIN7_SHA
assert PIN7_SHA == "a3bce3ab63c1" and P.parent == B / "runs"
S, E, TOK_STEP = M.STEPS_3E8, M.EVAL_3E8, M.TOK_STEP          # 18310 steps, eval every 250, 32 x 512
TOKENS_ARG = 3e8                                                # the trainer's default (F87 passes no --tokens)
assert int(TOKENS_ARG // TOK_STEP) == S == 18310 and E == 250

# ---------------------------------------------------------------- design (pre-registered: runs/pt300x/AUDIT.md)
HW = {"93m": "l40s", "294m": "b200"}            # one GPU type per size (the user, 2026-11-02), never mixed inside a size
ARMS = ("none", "win-ctx")
SEL, CONF = (0, 1, 2, 3), (4, 5, 6, 7)
GRID = {"93m": {"none": ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"), "win-ctx": ("6e-4", "1.2e-3", "2.4e-3", "4.8e-3")},
        "294m": {"none": ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"), "win-ctx": ("3e-4", "6e-4", "1.2e-3", "2.4e-3")}}
for _s, _g2 in GRID.items():   # every grid = 4 consecutive rungs of the factor-2 ladder
    for _a, _lrs in _g2.items():
        _i = M.LADDER.index(_lrs[0])
        assert _lrs == M.LADDER[_i:_i + 4], (_s, _a, _lrs)
for _k in HW:
    assert _k in M.SIZES and M.SIZES[_k][4] > 0
assert M.SIZES["93m"][:4] == (512, 10, 8, 2048) and M.SIZES["294m"][:4] == (832, 19, 13, 3328)
assert M.SIZES["93m"][5] == 41964032 and M.SIZES["294m"][5] == 210500160
SLOTS = 8                       # B200 singleton slots llmadd_p3x-b200-slot0..7 (one account: 520)
WINDOW_EVENTS = [[916, 1], [7324, 0]]   # the window at 18310 steps (= readout_rules.W_EVENTS; intent_x asserts it = make_jobs_pttpp.window_events)
# cost (AUDIT.md "Cost"): training seconds per run at 18310 steps / 75 evals, from run JSONs (curve[-1]['sec'], medians):
#   93M L40S: pttpp 93M@0.745 (1908 steps, 736 s) and 93M@0.318 (814 steps, 451 s) -> 0.2605 s/step + 3.186 s/eval -> 5009 s
#   294M B200: log-log in n_params through the B200 505M bridge (5768 s) and ptlr_1b 1.07B (10395 s), exponent 0.783 -> 3773 s
#   (the other route, pttpp 294M L40S 0.7551 s/step + 10.92 s/eval x the bridge's B200 / L40S ratio 0.2506, gives 3670 s)
TRAIN_S = {"93m": 5009.0, "294m": 3773.0}
START_S = 180                   # start-up, --dry_inputs and CHECK7 (pttpp 93M log: 870 s wall vs 735 s training)


def step_out(lr, edge):
    i = M.LADDER.index(lr)
    assert (edge == "upper" and i + 1 < len(M.LADDER)) or (edge == "lower" and i > 0), (lr, edge)
    return M.LADDER[i + 1] if edge == "upper" else M.LADDER[i - 1]


def tag_of(size, arm, lr):
    return f"p3x-{size}-lr{lr}-{arm}"


def target_of(arm):
    return "none" if arm == "none" else "helix3"


def run_file(size, arm, lr, seed):
    """the run JSON / check file names relative to <runs root>/pt300x/out/<tag>/"""
    tag = tag_of(size, arm, lr)
    return tag, f"run_whole_{target_of(arm)}_seed{seed}_{tag}.json", f"check_s{seed}.json"


# ---------------------------------------------------------------- decisions (select_pt300x.py --write appends; append-only)
def decisions(path=None):
    """[{seq, size, arm, stage X | C | STOP, lr, ...}] validated: per (size, arm) at most one X (one ladder step beyond a
    stage-A edge), then at most one C (lr in the grid or the X lr) or a STOP; nothing after a C or a STOP"""
    f = Path(path) if path else P / "decisions.json"
    log = json.loads(f.read_text())["log"] if f.exists() else []
    assert [e["seq"] for e in log] == list(range(len(log))), "decision seqs"
    st = {(s, a): {"X": None, "C": None, "STOP": False} for s in GRID for a in ARMS}
    for e in log:
        k = (e["size"], e["arm"])
        assert k in st and e["stage"] in ("X", "C", "STOP"), e
        x = st[k]
        assert x["C"] is None and not x["STOP"], ("entry after a C or a STOP", e)
        g = GRID[e["size"]][e["arm"]]
        if e["stage"] == "X":
            assert x["X"] is None and e["lr"] in (step_out(g[-1], "upper"), step_out(g[0], "lower")), e
            x["X"] = e["lr"]
        elif e["stage"] == "C":
            assert e["lr"] in g + ((x["X"],) if x["X"] else ()), e
            x["C"] = e["lr"]
        else:
            x["STOP"] = True
    return log, st


def runs_of(size, path=None):
    """[(stage, arm, lr, seed)]: stage A (arm-major, lr ascending, seed), then the decided X / C runs in decision order"""
    out = [("A", a, lr, s) for a in ARMS for lr in GRID[size][a] for s in SEL]
    for e in decisions(path)[0]:
        if e["size"] == size and e["stage"] in ("X", "C"):
            out += [(e["stage"], e["arm"], e["lr"], s) for s in (SEL if e["stage"] == "X" else CONF)]
    assert len(set(out)) == len(out), size
    return out


def slot_of(size, i, stage, arm, seed):
    """B200 slot: stage A by index (i % 8: none's 16 runs fill the first two waves), X / C by arm and seed"""
    if HW[size] != "b200":
        return None
    return i % SLOTS if stage == "A" else (ARMS.index(arm) * 4 + seed % 4) % SLOTS


# ---------------------------------------------------------------- per run facts
def fmt_h(h):
    return f"{int(h):02d}:{int(round(60 * (h % 1))):02d}:00"


def facts(size, arm, lr, seed, stage):
    assert size in GRID and arm in ARMS and stage in ("A", "X", "C") and (seed in SEL) == (stage != "C"), (size, arm, stage, seed)
    d, nl, nh, dff, n_params, nne = M.SIZES[size]
    hw = HW[size]
    ev, on = M.window_events(S) if arm != "none" else ([], 0.0)
    est_h = (TRAIN_S[size] + START_S) / 3600
    tlim_h = 3.0 if hw == "b200" else max(1.0, math.ceil(2.5 * est_h * 2) / 2)   # B200 = pttpp's bridge; L40S = pttpp's rule
    tag = tag_of(size, arm, lr)
    return dict(gid=f"pt300x-{size}", size=size, arm=arm, lr=lr, seed=seed, stage=stage, setting="7", hw=hw, line=None,
                S=S, E=E, T=S * TOK_STEP, tokens_arg=TOKENS_ARG, evals=M.n_evals(S, E), events=ev, on_frac=on,
                dims=(d, nl, nh, dff), n_params=n_params, nne=nne, tpp=S * TOK_STEP / nne, warm=max(1, S // 100),
                layers=[nl // 3, 2 * nl // 3], ckpt_on=n_params > 150e6, est_h=est_h, tlim=fmt_h(tlim_h),
                save=(stage == "C"), ckpt_every=500 if hw == "b200" else 0, lm0=M.lm0_ref("7", size, seed), tag=tag,
                name=f"{tag}-s{seed}")


def args_x(F):
    """the trainer ARGS: F87's 3e8 form (no --tokens / --eval_every) = make_jobs_pttpp.args7 of the same facts in its
    3e8 (bridge) form up to the --out directory; every intended flag asserted, nothing else"""
    out = P / "out" / F["tag"]
    args = (f"--data {DATA} --tokenizer {TOK} --tokenization whole --out {out} {M.ARM7[F['arm']]} --arith_lines 4 "
            f"{M.model_flags(F['dims'])} --lr {F['lr']} --seed {F['seed']} --tag _{F['tag']}"
            + (" --save_model" if F["save"] else "") + (f" --ckpt_every {F['ckpt_every']}" if F["ckpt_every"] else ""))
    ref = M.args7(dict(F, gid=M.BRIDGE))   # pttpp's own assertion of every flag runs here too
    assert args == ref.replace(f" --out {M.R / 'out' / F['tag']} ", f" --out {out} "), (F["name"], args, ref)
    f = M.M1.flags(args)
    d, nl, nh, dff = F["dims"]
    want = {"--data": str(DATA), "--tokenizer": str(TOK), "--tokenization": "whole", "--out": str(out), "--arith_lines": "4",
            "--d_model": str(d), "--n_layers": str(nl), "--n_heads": str(nh), "--d_ff": str(dff), "--lr": F["lr"],
            "--seed": str(F["seed"]), "--tag": f"_{F['tag']}", "--save_model": True if F["save"] else None,
            "--ckpt_every": str(F["ckpt_every"]) if F["ckpt_every"] else None, **M.WANT7[F["arm"]]}
    for k, v in want.items():
        assert f.get(k) == v, (F["name"], k, f.get(k), v)
    assert set(f) == {k for k, v in want.items() if v is not None}, (F["name"], sorted(f))
    assert "--tokens" not in f and "--eval_every" not in f and float(f["--lr"]) == float(F["lr"])
    return args


def intent_x(F):
    """the run JSON this job must produce (CHECK7 compares every field) = make_jobs_pttpp.intent7 of the same facts with
    this batch's --out"""
    I = M.intent7(F)
    assert I["args"]["out"] == str(M.R / "out" / F["tag"]), I["args"]["out"]
    I["args"]["out"] = str(P / "out" / F["tag"])
    a = I["args"]
    assert (a["tokens"], a["eval_every"], a["arith_lines"], a["lam"], a["window"], a["site"], a["bs"], a["seq"]) == \
        (3e8, 250, 4, 5.0, "0.05,0.4", "ctx", 32, 512), a
    assert I["steps"] == 18310 and I["evals"] == 75 and I["events"] == (WINDOW_EVENTS if F["arm"] != "none" else [])
    assert I["lm0_ref"] is not None, (F["name"], "no step-0 lm reference")
    return I


# ---------------------------------------------------------------- the job (make_jobs_pttpp.job7's body, this batch's paths)
def job_x(F, slot=None):
    args = args_x(F)
    I = intent_x(F)
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    tag, rf, cf = run_file(F["size"], F["arm"], F["lr"], F["seed"])
    runjson, check = P / "out" / tag / rf, P / "out" / tag / cf
    if F["hw"] == "b200":
        assert slot is not None and 0 <= slot < SLOTS
        head = f"""#SBATCH --job-name=llmadd_p3x-b200-slot{slot}
#SBATCH --dependency=singleton
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint=b200
#SBATCH --comment="preemption=yes"
#SBATCH --nice=20
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time={F['tlim']}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={B}/logs/llmadd_{F['name']}_%j.out"""
        gpu = "B200"
    else:
        assert F["hw"] == "l40s" and slot is None
        head = f"""#SBATCH --job-name=llmadd_{F['name']}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --nice=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={F['tlim']}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out"""
        gpu = "L40S"
    resume = ("Preemptible: a preempted job is requeued (same id) and the trainer resumes from its ckpt (every 500 steps); "
              f"singleton slots keep at most {SLOTS} B200 runs of this batch at a time." if F["hw"] == "b200" else
              "No ckpt (--ckpt_every 0): a requeued job restarts the run from step 0.")
    w = F["events"]
    return f"""#!/bin/bash
{head}
# pt300x run {F['name']}: {F['size']} at 3e8 tokens (F87's fixed-token 7% ladder, pttpp shape), stage {F['stage']}; {F['arm']} lr {F['lr']} seed {F['seed']}.
# Intent spec and pre-registered analysis runs/pt300x/AUDIT.md; generated by llmadd_torch/make_jobs_pt300x.py (change the
# generator, not this file); trainer = F87's snapshot (code7/COMMIT, pretrain_range.py {PIN7_SHA}). {resume}
set -eu
# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/pt300x/HOLD exists
if [ -e {P}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {P}/code7/tools
# the snapshot must still be the pinned trainer (code_sha of every result); exit before any GPU work otherwise
[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{PIN7_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
echo "=== {F['name']} start $(date -Is) job ${{SLURM_JOB_ID:-none}} account ${{SLURM_JOB_ACCOUNT:-?}} partition ${{SLURM_JOB_PARTITION:-?}} restart ${{SLURM_RESTART_COUNT:-0}} node $(hostname)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"; echo "$GPU"
# one GPU type per size (the design's): exit before any training on anything else
case "$GPU" in *{gpu}*) ;; *) echo "not a {gpu}: exiting"; exit 4;; esac
(cd {DATA} && sha256sum -c -) <<'SHA' || exit 3
{M.J.DATA_SHA[0]}  train.bin
{M.J.DATA_SHA[1]}  val.bin
SHA
echo "INTENT {F['name']}: {F['size']} N_nonemb {F['nne']}, tokens {F['T']} = {F['S']} steps x {TOK_STEP} (tpp {F['tpp']:.5f}), eval every {F['E']} steps -> {F['evals']} evals, warmup {F['warm']} steps, prior window {'steps ' + str(w[0][0]) + '..' + str(w[1][0] - 1) if w else 'none (no prior)'}, save_model {F['save']}"
ARGS="{args}"
{PY} pretrain_range.py $ARGS --dry_inputs
{PY} pretrain_range.py $ARGS
echo "=== {F['name']} trainer returned $(date -Is)"
# intent checks of the run JSON (every argument, steps, evals, window events, step-0 lm) -> {check.name}; a failed check
# withholds the run in select_pt300x.py / readout_pt300x.py; the job exits with the check's code
RC=0
{PY} - "{runjson}" "{check}" '{ij}' <<'PY' || RC=$?
{M.CHECK7}PY
# no selection hook: runs/pt300x/select_pt300x.py is run by the coordinator by hand (AUDIT.md "lr selection")
exit $RC
"""


def render(size, path=None):
    """[(run name, job text, facts)] of a size's current stages (B200 runs with their slot)"""
    out = []
    for i, (st, a, lr, s) in enumerate(runs_of(size, path)):
        F = facts(size, a, lr, s, st)
        out.append((F["name"], job_x(F, slot_of(size, i, st, a, s)), F))
    return out


# ---------------------------------------------------------------- snapshot, cost, main
def snapshot():
    """runs/pt300x/code7 = git archive 64abc19 tools/ (make_jobs_pttpp._extract: once, temp dir + rename), asserted equal
    file by file to runs/pttpp/code7 and to F87's runs/ptlr/code, COMMIT verbatim; then read-only"""
    dst = P / "code7"
    M._extract(dst, PIN7, M.R / "code7", (M.R / "code7" / "COMMIT").read_text())
    M._extract(dst, PIN7, M.F87_CODE, (M.F87_CODE / "COMMIT").read_text())
    assert sha(dst / "tools" / "pretrain_range.py") == PIN7_SHA
    for root, dirs, files in os.walk(dst):
        for n in dirs + files:
            p = Path(root) / n
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    dst.chmod(dst.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def plan():
    print(f"{'size':5s} {'hw':5s} {'shape':22s} {'N_nonemb':>12s} {'steps':>6s} {'tokens':>12s} {'tpp':>7s} {'evals':>5s} "
          f"{'warm':>4s} {'window':>9s} {'grid none':>32s} {'grid win-ctx':>30s} {'h/run':>6s} {'limit':>8s}")
    tot = {}
    for size in GRID:
        F = facts(size, "win-ctx", GRID[size]["win-ctx"][0], 0, "A")
        d, nl, nh, dff = F["dims"]
        w = F["events"]
        print(f"{size:5s} {F['hw']:5s} {f'd{d} x {nl}, {nh} heads, ff {dff}':22s} {F['nne']:12,d} {F['S']:6d} {F['T']:12,d} "
              f"{F['tpp']:7.3f} {F['evals']:5d} {F['warm']:4d} {str(w[0][0]) + '-' + str(w[1][0] - 1):>9s} "
              f"{' / '.join(GRID[size]['none']):>32s} {' / '.join(GRID[size]['win-ctx']):>30s} {F['est_h']:6.3f} {F['tlim']:>8s}")
        nA, nC, nX = 32, 8, 8
        tot[F["hw"]] = (nA * F["est_h"], nC * F["est_h"], nX * F["est_h"])
    for hw, (a, c, x) in tot.items():
        print(f"GPU-h {hw}: stage A {a:.1f} + C {c:.1f} = {a + c:.1f}; worst case + X on both arms {x:.1f} -> {a + c + x:.1f}")
    l40s, b200 = facts("93m", "none", "2.4e-3", 0, "A")["est_h"], facts("294m", "none", "2.4e-3", 0, "A")["est_h"]
    for name, h, conc in (("93M L40S", l40s, 12), ("294M B200", b200, SLOTS)):
        waves = [math.ceil(32 / conc), 1, 1]
        print(f"wall {name} at {conc} concurrent: A {waves[0]} waves + X 1 + C 1 = {sum(waves) * h:.1f} h of running "
              f"(A alone {waves[0] * h:.1f} h), plus queue waits and the selection by hand")


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    if "--plan" in sys.argv:
        plan()
        return
    au = P / "AUDIT.md"
    assert au.exists() and "## Pre-registered analysis" in au.read_text(), "runs/pt300x/AUDIT.md (the pre-registration) first"
    sizes = [opt("--size")] if opt("--size") else list(GRID)
    stage, arm = opt("--stage"), opt("--arm")
    assert all(s in GRID for s in sizes) and stage in (None, "A", "X", "C") and arm in (None,) + ARMS, (sizes, stage, arm)
    if "--names" in sys.argv:   # the selection's run names in rendering (= submission) order; writes nothing
        for size in sizes:
            for name, txt, F in render(size):
                if not ((stage and F["stage"] != stage) or (arm and F["arm"] != arm)):
                    print(name)
        return
    snapshot()
    existing = {p.name for p in B.glob("runs/*/out*/*") if p.parent.parent != P}   # other batches' run dirs
    bad, jobs = 0, []
    for size in sizes:
        for name, txt, F in render(size):
            if (stage and F["stage"] != stage) or (arm and F["arm"] != arm):
                continue
            assert name not in existing and F["tag"] not in existing, name
            job, flag = P / f"job_{name}.sbatch", P / f"SUBMITTED_{name}"
            if "--check" in sys.argv:   # read-only: every job file of the selection = this rendering
                if not job.exists() or job.read_text() != txt:
                    print(f"{'MISSING' if not job.exists() else 'DIFFERS'}: {job.name}")
                    bad += 1
            elif flag.exists():
                if not job.exists():
                    print(f"WARNING: {job.name} was submitted but its file is missing: not rewritten")
                elif job.read_text() != txt:
                    print(f"WARNING: {job.name} was submitted and differs from this rendering: kept, not rewritten")
            else:
                job.write_text(txt)
            jobs.append((name, job, flag, F))
    if "--print" in sys.argv:
        for name, job, flag, F in jobs:
            print(name, F["stage"], F["hw"], [l for l in job.read_text().splitlines() if l.startswith("ARGS=")][0])
    for name, job, flag, F in jobs:
        if "--test-only" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", "--test-only", str(job)], capture_output=True, text=True)
            lines = (o.stdout + o.stderr).strip().splitlines()
            line = next((x for x in lines if "to start at" in x), lines[-1] if lines else "")
            ok = o.returncode == 0 and "to start at" in line
            bad += not ok
            print(f"{'OK ' if ok else 'BAD'} {name}: {line}")
        elif "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text(o.stdout)
                print(name, o.stdout.strip())
            else:
                bad += 1
                print("FAILED", name, o.stderr.strip())
                break
    print(f"pt300x {', '.join(sizes)}{' stage ' + stage if stage else ''}{' arm ' + arm if arm else ''}: {len(jobs)} jobs in {P}; "
          f"{'checked' if '--check' in sys.argv else 'test-only' if '--test-only' in sys.argv else 'submitted' if '--submit' in sys.argv else 'not submitted'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
