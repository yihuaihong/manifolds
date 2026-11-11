"""Torch jobs ptsz (2026-11-05; the user's decision of 2026-11-05, relayed by the coordinator: P1 of the writing session's
request incl. the ~750M size, all on L40S; priority ptdom > new sizes > ptshare -> every job of this batch --nice=1000; 2026-11-05 amendment N1: --nice=100, the user).
Intent spec and PRE-REGISTERED analysis: runs/ptsz/AUDIT.md.
The fixed-3e8-token 7% size ladder (F87 ptlr / ptlr_1b, F99 pt300x) gets four more L40S points, at the same 3e8 tokens:
  136m = d640 x 11, 10 heads, d_ff 2560 (N_nonemb  72,118,400)   new, never seen (93M-190M geometric midpoint +0.045)
  294m = d832 x 19, 13 heads, d_ff 3328 (N_nonemb 210,500,160)   the pttpp shape; an independent L40S re-run of F99's B200 294M
  391m = d960 x 20, 15 heads, d_ff 3840 (N_nonemb 294,989,760)   new, never seen (294M-505M midpoint +0.013)
  731m = d1152 x 29, 18 heads, d_ff 4608 (N_nonemb 615,911,040)  new, never seen (505M-1.07B midpoint -0.001)
(exact counts: scale_work/ptsz/shapes_ptsz.py, the trainer's own build_model on the meta device; names = rounded totals).
All on L40S (account torch_pr_625_general, l40s_public, QOS gpu48, --nice=100 (amendment N1), GPU-name guard 'L40S'); one GPU type for the
whole batch. Protocol = pt300x (= F87 ptlr) exactly, except shape, lr grid, --nice and (runs over ~4 h) --ckpt_every 2000:
3e8 tokens = 18,310 steps of 32 x 512 (the trainer's default --tokens 3e8: no --tokens / --eval_every flag), eval every 250
steps (75 evals), 1% warmup, window (0.05, 0.4), lambda 5, ctx site, --arith_lines 4 (7%); arms 'none' / 'win-ctx' = F87's.
Trainer and job body = pt300x's (= make_jobs_pttpp.job7's 7% body): runs/ptsz/code7 = `git archive 64abc19 tools/`
(pretrain_range.py a3bce3ab63c1), asserted equal to runs/pttpp/code7 (= F87's runs/ptlr/code) file by file; HOLD kill
switch, trainer sha, GPU-name guard, data sha256, INTENT line, --dry_inputs, training, CHECK7 verbatim; no selection hook.
The frozen pttpp generator is imported read-only (sha256[:12] and the absence of top-level calls asserted); its args7() /
intent7() are called for every run and their outputs asserted equal to this batch's up to the --out directory (they take
the shape from the facts, so the new shapes need no change there; this module's SIZES extends make_jobs_pttpp.SIZES).
Step-0 lm references of the new shapes: llmadd_torch/ptsz_refs.json (scale_work/ptsz/refs_ptsz.py, CPU); 294m keeps
pttpp_refs.json '7|294m' (frozen, never written).
Stages (AUDIT.md "lr selection"): A = 4 lrs x 2 arms x seeds 0-3 per size (128 runs); select_ptsz.py (by hand) applies
F87 rule 1 and appends X (one ladder step beyond an open edge, seeds 0-3, once per arm), C (seeds 4-7 at the pick) or STOP
to runs/ptsz/decisions.json; this generator then renders those runs too. A submitted job file is never rewritten.
    python make_jobs_ptsz.py --plan                                     design, grids, cost, wall time
    python make_jobs_ptsz.py [--stage A|X|C] [--size S] [--arm A] [--print] [--test-only | --submit]
    python make_jobs_ptsz.py --check [--stage A]                        job files = rendering, snapshot, frozen shas
    python make_jobs_ptsz.py --names [--stage A] [--size S]             run names in submission order (submit_ptsz.sh)
    python make_jobs_ptsz.py --smoke [--test-only]                      write runs/ptsz/smoke/job_smoke731.sbatch (AUDIT.md "Smoke")
    python make_jobs_ptsz.py --smoke-xs [--test-only]                   write runs/ptsz/smoke/job_smoke731xs.sbatch (AUDIT.md "Amendment 4")
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
# read-only inputs (sha256[:12]): pt300x's list (the pttpp generator and what it imports / reads) + pt300x's own generator,
# decisions (the 93M / 294M picks used for the grids and the reused cells) and trainer snapshot
FROZEN = {"make_jobs_pttpp.py": "b4bfa6eb248d", "make_jobs_ptlr1b.py": "8dd7b3407459", "make_jobs_pt2s505.py": "a5a7fb60eedc",
          "make_jobs_rb6scale.py": "19bc055c2d09", "pttpp_refs.json": "d10d720c1242", "pt2s505_refs.json": "479e394771e9",
          "rb6scale_refs.json": "b65472eb3d29", "runs/pttpp/code7/tools/pretrain_range.py": "a3bce3ab63c1",
          "runs/pttpp/readout/tools/readout_rules.py": "e91ee8a8a846", "make_jobs_pt300x.py": "e8fd563c335a",
          "runs/pt300x/decisions.json": "22532f929c7a", "runs/pt300x/code7/tools/pretrain_range.py": "a3bce3ab63c1"}
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
P = B / "runs" / "ptsz"
TOK, PY, DATA, PIN7, PIN7_SHA = M.TOK, M.PY, M.DATA, M.PIN7, M.PIN7_SHA
assert PIN7_SHA == "a3bce3ab63c1" and P.parent == B / "runs"
S, E, TOK_STEP = M.STEPS_3E8, M.EVAL_3E8, M.TOK_STEP          # 18310 steps, eval every 250, 32 x 512
TOKENS_ARG = 3e8                                                # the trainer's default (F87 passes no --tokens)
assert int(TOKENS_ARG // TOK_STEP) == S == 18310 and E == 250
V = M.V

# ---------------------------------------------------------------- sizes (pre-registered: runs/ptsz/AUDIT.md "Design")
# (d_model, n_layers, n_heads, d_ff, n_params, N_nonemb): the trainer's own build_model on the meta device
# (scale_work/ptsz/shapes.out; the six existing sizes reproduced make_jobs_pttpp.SIZES to the parameter there)
NEW_SHAPES = {"136m": (640, 11, 10, 2560, 136296320, 72118400), "391m": (960, 20, 15, 3840, 391256640, 294989760),
              "731m": (1152, 29, 18, 4608, 731431296, 615911040)}
SIZES = dict(M.SIZES)
SIZES.update(NEW_SHAPES)
for _k, (_d, _l, _h, _ff, _n, _nn) in SIZES.items():
    assert _d == 64 * _h and _ff == 4 * _d and _n - V * _d == _nn, _k
    assert _k == (f"{round(_n / 1e6)}m" if _k != "1b" else "1b"), (_k, _n)   # names = rounded totals (1.07B: '1b')
LADDER9 = sorted(SIZES, key=lambda k: SIZES[k][5])
assert LADDER9 == ["53m", "93m", "136m", "190m", "294m", "391m", "505m", "731m", "1b"], LADDER9
for _a, _b in zip(LADDER9, LADDER9[1:]):   # width and depth grow together along the ladder (the shape rule)
    assert SIZES[_a][0] < SIZES[_b][0] and SIZES[_a][1] < SIZES[_b][1], (_a, _b)
MODEL_LINE = {k: f"model: {v[4] / 1e6:.1f}M parameters; gradient checkpointing {'on' if v[4] > 150e6 else 'off'}"
              for k, v in SIZES.items()}
assert all(MODEL_LINE[k] == M.MODEL_LINE[k] for k in M.SIZES)
BATCH = ("731m", "391m", "294m", "136m")       # this batch's sizes in submission order (the long pole first)
NEW3 = ("136m", "391m", "731m")                # never seen before registration (the secondary test)
REGISTERED = ("136m", "294m", "391m", "731m")  # GRID / iteration order (ascending N)
assert sorted(BATCH) == sorted(REGISTERED) and set(NEW3) < set(REGISTERED)
ARMS = ("none", "win-ctx")
SEL, CONF = (0, 1, 2, 3), (4, 5, 6, 7)
LADDER = M.LADDER
WINDOW_EVENTS = [[916, 1], [7324, 0]]   # the window at 18310 steps (= readout_rules.W_EVENTS; intent_x asserts it)

# ---------------------------------------------------------------- lr grids (pre-registered: AUDIT.md "lr grids")
# the existing 3e8-token 7% picks (none / win-ctx): F87 ptlr (53M, 190M, 505M: make_jobs_pttpp.OPT3E8, 505M none = the
# geometric mean of its 8-seed tie 4.8e-3 / 9.6e-3), ptlr_1b (1.07B), pt300x decisions.json (93M L40S, 294M B200)
PICK3E8 = {"none": {"53m": 8e-3, "93m": 2.4e-3, "190m": 4.8e-3, "294m": 9.6e-3, "505m": math.sqrt(4.8e-3 * 9.6e-3), "1b": 2.4e-3},
           "win-ctx": {"53m": 2e-3, "93m": 1.2e-3, "190m": 1.2e-3, "294m": 6e-4, "505m": 1.2e-3, "1b": 6e-4}}
for _a in ARMS:
    for _k, _v in M.OPT3E8[("7", _a)].items():
        assert PICK3E8[_a][_k] == _v, (_a, _k)
_dec = {(e["size"], e["arm"]): e["lr"] for e in json.loads((B / "runs/pt300x/decisions.json").read_text())["log"] if e["stage"] == "C"}
assert _dec == {("93m", "none"): "2.4e-3", ("93m", "win-ctx"): "1.2e-3", ("294m", "none"): "9.6e-3", ("294m", "win-ctx"): "6e-4"}, _dec
for (_k, _a), _lr in _dec.items():
    assert PICK3E8[_a][_k] == float(_lr)
NEIGH = {"136m": ("93m", "190m"), "391m": ("294m", "505m"), "731m": ("505m", "1b"),     # the new sizes
         "93m": ("53m", "190m"), "294m": ("190m", "505m")}                              # pt300x's (known answers)


def lr_pred(size, arm, picks=None):
    """log-linear interpolation in ln N_nonemb between the neighbouring sizes' 3e8 picks (pt300x AUDIT 'lr grids')"""
    pk = picks or PICK3E8[arm]
    lo, hi = NEIGH[size]
    x, x0, x1 = (math.log(SIZES[k][5]) for k in (size, lo, hi))
    f = (x - x0) / (x1 - x0)
    assert 0 < f < 1, (size, f)
    return math.exp(math.log(pk[lo]) + f * (math.log(pk[hi]) - math.log(pk[lo])))


def grid_rule(pred):
    """4 consecutive factor-2 ladder rungs, two at or below the interpolated lr and two above it: L[k-1], L[k] <= pred <
    L[k+1], L[k+2] (make_jobs_pttpp.grid_rule's convention 'L[k] <= pred'; a pred on a rung, within 1e-9 in log, counts as
    that rung)"""
    k = max(i for i, v in enumerate(LADDER) if math.log(float(v)) <= math.log(pred) + 1e-9)
    assert 1 <= k and k + 2 < len(LADDER), pred
    return LADDER[k - 1:k + 3]


GRID = {"136m": {"none": ("1.2e-3", "2.4e-3", "4.8e-3", "9.6e-3"), "win-ctx": ("6e-4", "1.2e-3", "2.4e-3", "4.8e-3")},
        "294m": {"none": ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"), "win-ctx": ("3e-4", "6e-4", "1.2e-3", "2.4e-3")},   # pt300x's, unchanged
        "391m": {"none": ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"), "win-ctx": ("3e-4", "6e-4", "1.2e-3", "2.4e-3")},
        "731m": {"none": ("1.2e-3", "2.4e-3", "4.8e-3", "9.6e-3"), "win-ctx": ("3e-4", "6e-4", "1.2e-3", "2.4e-3")}}
assert tuple(GRID) == REGISTERED
PT300X_GRID = {"93m": {"none": ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"), "win-ctx": ("6e-4", "1.2e-3", "2.4e-3", "4.8e-3")},
               "294m": {"none": ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"), "win-ctx": ("3e-4", "6e-4", "1.2e-3", "2.4e-3")}}
for _s, _g2 in GRID.items():
    for _a, _lrs in _g2.items():
        _i = LADDER.index(_lrs[0])
        assert _lrs == LADDER[_i:_i + 4], (_s, _a, _lrs)                    # 4 consecutive rungs of the factor-2 ladder
        if _s in NEW3:
            assert _lrs == grid_rule(lr_pred(_s, _a)), (_s, _a, _lrs, grid_rule(lr_pred(_s, _a)))   # the rule's output
        alt = {**PICK3E8[_a], "505m": 9.6e-3} if _a == "none" else PICK3E8[_a]   # 505M none as the REUSED pick instead
        for _p in {lr_pred(_s, _a), lr_pred(_s, _a, alt)}:                   # straddles the interpolation, >= 1 rung each side
            assert float(_lrs[0]) < _p * (1 + 1e-9) and float(_lrs[-1]) > _p * (1 - 1e-9), (_s, _a, _p)
            assert sum(float(v) <= _p * (1 + 1e-9) for v in _lrs) >= 1 and sum(float(v) > _p * (1 + 1e-9) for v in _lrs) >= 1, (_s, _a, _p)
assert GRID["294m"] == PT300X_GRID["294m"]
# known answers: the rule reproduces pt300x's grids (93M both arms, 294M none); 294M win-ctx (interpolation exactly on the
# 1.2e-3 rung) was set one rung lower in pt300x by its stated reasons (AUDIT.md "lr grids"): kept unchanged here
assert grid_rule(lr_pred("93m", "none")) == PT300X_GRID["93m"]["none"] and grid_rule(lr_pred("93m", "win-ctx")) == PT300X_GRID["93m"]["win-ctx"]
assert grid_rule(lr_pred("294m", "none")) == PT300X_GRID["294m"]["none"]
assert abs(lr_pred("294m", "win-ctx") / 1.2e-3 - 1) < 1e-12 and grid_rule(lr_pred("294m", "win-ctx")) != PT300X_GRID["294m"]["win-ctx"]
assert abs(lr_pred("136m", "win-ctx") / 1.2e-3 - 1) < 1e-12   # the one on-rung case among the new cells (93M = 190M = 1.2e-3)

# ---------------------------------------------------------------- cost, ckpt, time limit (AUDIT.md "Cost")
# training seconds per run at 18310 steps / 75 evals (curve[-1]['sec']), L40S (scale_work/ptsz/timing.out):
#   136m (grad ckpt off): log-log in n_params through 53M (3470 s) and 93M (4975 s), win-ctx medians, both off (exponent 0.630) -> 6317 s
#   294m: pttpp 294M L40S at the two tpp lines (9570 / 4083 steps) -> 0.7596 s/step + 10.70 s/eval (win-ctx; none 14665) -> 14711 s
#   391m: log-log in n_params through 294M (14711 s) and 505M (23170 s, ptlr win-ctx median), exponent 0.838 -> 18695 s
#   731m: the smoke (runs/ptsz/smoke, AUDIT.md "Smoke"); before it: the same log-log extrapolated -> 31581 s
TRAIN_S = {"136m": 6317.0, "294m": 14711.0, "391m": 18695.0, "731m": 33177.0, "93m": 5009.0}   # 93m: pt300x's (known answer)
# amendment 4 (2026-11-06, before any ptsz training job): 731m = the smoke 22011938's measurement (smoke_report_22011938.json
# est.win_s 33176.6 s: 18310 t + 75 e with t 1.7032 s/step, e 23.20 s/eval, x the F87 505M L40S win-ctx / none ratio), was
# the log-log extrapolation 31581 s; the time limit 22:30 -> 23:30 (the same 2.5x rule); no other size changes
# amendment 4 (the user's decision 2026-11-06): the 731m jobs (both arms, every stage) run with the CUDA caching allocator's
# expandable segments; numerics unchanged (allocator only), against fragmentation (smoke 22011938: the resumed in-window eval
# reserved 46.83 of 47.70 GB with 40.30 GB allocated); verified by the smoke job_smoke731xs (AUDIT.md "Amendment 4")
XS_SIZES = ("731m",)   # amendment 7 (2026-11-10, the user): the setting for every 731m run (both arms, all seeds, all stages); amendment 5 had removed it after the XS smoke's margin failure, amendment 6 (seed-2 OOM) brings it back
XS_ENV = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
START_S = 180                   # start-up, --dry_inputs and CHECK7 (pt300x's)
CKPT_GB = {k: v[4] * 12 / 1e9 for k, v in SIZES.items()}     # model fp32 + 2 Adam moments
CKPT_S_PER_GB = 3.0             # ckpt write seconds per GB (before the smoke: a guess; the smoke times the 731m writes)
CKPT_RULE_H = 4.0               # --ckpt_every 2000 (F87's L40S value) when the estimated run exceeds ~4 h; else 0 (pt300x)
GPU48_MAXWALL_H = 48.0          # sacctmgr show qos gpu48: MaxWall 2-00:00:00 (l40s_public MaxTime UNLIMITED)


def ckpt_every_of(size):
    return 2000 if (TRAIN_S[size] + START_S) / 3600 > CKPT_RULE_H else 0


def n_ckpt(ck):
    """ckpt writes of a run: (s + 1) % ck == 0 and s < steps - 1 (the trainer's rule)"""
    return sum(1 for s in range(S) if ck and (s + 1) % ck == 0 and s < S - 1)


def fmt_h(h):
    return f"{int(h):02d}:{int(round(60 * (h % 1))):02d}:00"


def step_out(lr, edge):
    i = LADDER.index(lr)
    assert (edge == "upper" and i + 1 < len(LADDER)) or (edge == "lower" and i > 0), (lr, edge)
    return LADDER[i + 1] if edge == "upper" else LADDER[i - 1]


def tag_of(size, arm, lr):
    return f"psz-{size}-lr{lr}-{arm}"


def target_of(arm):
    return "none" if arm == "none" else "helix3"


def run_file(size, arm, lr, seed):
    """the run JSON / check file names relative to <runs root>/ptsz/out/<tag>/"""
    tag = tag_of(size, arm, lr)
    return tag, f"run_whole_{target_of(arm)}_seed{seed}_{tag}.json", f"check_s{seed}.json"


# ---------------------------------------------------------------- step-0 lm references
REFS = B / "ptsz_refs.json"     # this batch's CPU references of the new shapes (scale_work/ptsz/refs_ptsz.py table --write)
REFS_SHA = "831d6694cd6f"       # written 2026-11-05 16:46 (scale_work/ptsz/refs_table_write.txt: ALL IDENTITIES HOLD); frozen


def refs():
    if not REFS.exists():
        return {"lm0": {}}
    assert sha(REFS) == REFS_SHA, ("ptsz_refs.json changed", sha(REFS))
    return json.loads(REFS.read_text())


def lm0_ref(size, seed):
    """pttpp_refs.json (frozen) for the sizes it has (294m and the reused ones), ptsz_refs.json for the new shapes only"""
    if size in M.SIZES:
        return M.lm0_ref("7", size, seed)
    r = refs()
    assert not any(k.split("|")[1] in M.SIZES for k in r["lm0"]), "ptsz_refs.json must not shadow pttpp_refs.json"
    v = r["lm0"].get(f"7|{size}", {}).get(str(seed))
    return None if v is None else float(v)


# ---------------------------------------------------------------- decisions (select_ptsz.py --write appends; append-only)
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


# ---------------------------------------------------------------- per run facts
def facts(size, arm, lr, seed, stage, ka=False):
    """everything a job is rendered from; ka=True: a known-answer rendering of another size of the ladder (tests only)"""
    assert (size in GRID or (ka and size in TRAIN_S)) and arm in ARMS and stage in ("A", "X", "C") \
        and (seed in SEL) == (stage != "C"), (size, arm, stage, seed)
    d, nl, nh, dff, n_params, nne = SIZES[size]
    ev, on = M.window_events(S) if arm != "none" else ([], 0.0)
    ck = ckpt_every_of(size)
    est_h = (TRAIN_S[size] + START_S + n_ckpt(ck) * CKPT_GB[size] * CKPT_S_PER_GB) / 3600
    tlim_h = max(1.0, math.ceil(2.5 * est_h * 2) / 2)      # pt300x's (= pttpp's) L40S rule
    assert tlim_h <= GPU48_MAXWALL_H, (size, tlim_h)
    tag = tag_of(size, arm, lr)
    return dict(gid=f"ptsz-{size}", size=size, arm=arm, lr=lr, seed=seed, stage=stage, setting="7", hw="l40s", line=None,
                S=S, E=E, T=S * TOK_STEP, tokens_arg=TOKENS_ARG, evals=M.n_evals(S, E), events=ev, on_frac=on,
                dims=(d, nl, nh, dff), n_params=n_params, nne=nne, tpp=S * TOK_STEP / nne, warm=max(1, S // 100),
                layers=[nl // 3, 2 * nl // 3], ckpt_on=n_params > 150e6, est_h=est_h, tlim=fmt_h(tlim_h),
                save=(stage == "C"), ckpt_every=ck, lm0=lm0_ref(size, seed), tag=tag, name=f"{tag}-s{seed}")


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
    assert a["ckpt_every"] == F["ckpt_every"] and a["save_model"] == F["save"] and a["d_model"] == F["dims"][0]
    assert I["steps"] == 18310 and I["evals"] == 75 and I["events"] == (WINDOW_EVENTS if F["arm"] != "none" else [])
    assert I["n_params"] == SIZES[F["size"]][4] and I["n_nonemb"] == SIZES[F["size"]][5]
    assert I["lm0_ref"] is not None, (F["name"], "no step-0 lm reference (scale_work/ptsz/refs_ptsz.py table --write first)")
    return I


# ---------------------------------------------------------------- the job (make_jobs_pttpp.job7's body via pt300x, this batch's paths)
def header(F):
    return f"""#SBATCH --job-name=llmadd_{F['name']}
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --nice=100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={F['tlim']}
#SBATCH --requeue
""" + ("#SBATCH --open-mode=append\n" if F["ckpt_every"] else "") + f"#SBATCH --output={B}/logs/%x_%j.out"


def job_x(F):
    args = args_x(F)
    I = intent_x(F)
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    assert F["hw"] == "l40s"
    tag, rf, cf = run_file(F["size"], F["arm"], F["lr"], F["seed"])
    runjson, check = P / "out" / tag / rf, P / "out" / tag / cf
    gpu = "L40S"
    resume = (f"Resumable: --ckpt_every {F['ckpt_every']} (a requeued job, same id, resumes from the last ckpt; log appended)."
              if F["ckpt_every"] else "No ckpt (--ckpt_every 0): a requeued job restarts the run from step 0.")
    w = F["events"]
    # amendment 4: the allocator setting for the XS sizes only; every other size renders byte for byte as before
    xs = (f"export {XS_ENV}   # amendment 4 (runs/ptsz/AUDIT.md): allocator only, numerics unchanged\n"
          f'echo "ALLOC: PYTORCH_CUDA_ALLOC_CONF=${{PYTORCH_CUDA_ALLOC_CONF:-unset}}"\n') if F["size"] in XS_SIZES else ""
    return f"""#!/bin/bash
{header(F)}
# ptsz run {F['name']}: {F['size']} at 3e8 tokens (the fixed-token 7% ladder, all L40S), stage {F['stage']}; {F['arm']} lr {F['lr']} seed {F['seed']}.
# Intent spec and pre-registered analysis runs/ptsz/AUDIT.md; generated by llmadd_torch/make_jobs_ptsz.py (change the
# generator, not this file); trainer = F87's snapshot (code7/COMMIT, pretrain_range.py {PIN7_SHA}). {resume}
set -eu
# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/ptsz/HOLD exists
if [ -e {P}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {P}/code7/tools
# the snapshot must still be the pinned trainer (code_sha of every result); exit before any GPU work otherwise
[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{PIN7_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
{xs}echo "=== {F['name']} start $(date -Is) job ${{SLURM_JOB_ID:-none}} account ${{SLURM_JOB_ACCOUNT:-?}} partition ${{SLURM_JOB_PARTITION:-?}} restart ${{SLURM_RESTART_COUNT:-0}} node $(hostname)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"; echo "$GPU"
# one GPU type for the whole batch (the design's): exit before any training on anything else
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
# withholds the run in select_ptsz.py / readout_ptsz.py; the job exits with the check's code
RC=0
{PY} - "{runjson}" "{check}" '{ij}' <<'PY' || RC=$?
{M.CHECK7}PY
# no selection hook: runs/ptsz/select_ptsz.py is run by the coordinator by hand (AUDIT.md "lr selection")
exit $RC
"""


def render(size, path=None):
    """[(run name, job text, facts)] of a size's current stages"""
    out = []
    for st, a, lr, s in runs_of(size, path):
        F = facts(size, a, lr, s, st)
        out.append((F["name"], job_x(F), F))
    return out


# ---------------------------------------------------------------- the 731m smoke job (AUDIT.md "Smoke"; one job, by the build agent)
SMOKE = P / "smoke"


def smoke_job():
    """one L40S job: 731m memory (gradient checkpointing auto, the prior's in-window eval step with its two extra gradient
    passes, a ckpt resume) and speed (s/step, s/eval, ckpt write), plus a short 136m memory check (grad ckpt off). Every
    segment = the trainer's own main() with a rendered stage-A ARGS (asserted), --out -> runs/ptsz/smoke/<seg>, and only
    the listed test flags changed: --window 0,0.4 puts the prior on from step 0 and --eval_every 100 makes evals 100 / 200
    in-window evals with the previous step's gradients still allocated (the trainer's worst memory point: LM and prior
    gradient passes before zero_grad); --ckpt_every 100 + a second call = a resume (B resumes A's step-199 ckpt and reaches
    the in-window eval at 300 with the loaded checkpoint still referenced); C / D = the real none ARGS (D with 11 evals)
    for s/step and s/eval; E = 136m (grad ckpt off). Fallbacks (only if A fails): A2 / B2 with expandable_segments, A3 with
    --lm_chunk_tokens 8192 (both are deviations; reported, never used without the user)."""
    def seg_args(size, arm, lr, seed, out, repl, extra):
        F = facts(size, arm, lr, seed, "A")
        a = args_x(F).replace(f"--out {P / 'out' / F['tag']} ", f"--out {SMOKE / out} ")
        assert f"--out {SMOKE / out} " in a and str(P / "out") not in a
        for old, new in repl:
            assert a.count(old) == 1, (old, a)
            a = a.replace(old, new)
        assert M.M1.flags(a + extra)   # parses (no repeated flag)
        return a + extra
    w = ("--window 0.05,0.4", "--window 0,0.4")
    ck731 = (" --ckpt_every 2000", " --ckpt_every 100")
    assert ckpt_every_of("731m") == 2000 and ckpt_every_of("136m") == 0
    ab = " --eval_every 100"
    segs = [("A", seg_args("731m", "win-ctx", "6e-4", 0, "A", [w, ck731], ab + " --stop_after 210"), ""),
            ("B", seg_args("731m", "win-ctx", "6e-4", 0, "A", [w, ck731], ab + " --stop_after 310"), ""),   # resumes A's step-199 ckpt
            ("C", seg_args("731m", "none", "2.4e-3", 0, "C", [], " --stop_after 500"), ""),
            ("D", seg_args("731m", "none", "2.4e-3", 0, "D", [], " --eval_every 25 --stop_after 250"), ""),
            ("E", seg_args("136m", "win-ctx", "1.2e-3", 0, "E", [w], ab + " --stop_after 210"), "")]
    xs = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    fb = [("A2", seg_args("731m", "win-ctx", "6e-4", 0, "A2", [w, ck731], ab + " --stop_after 210"), xs),
          ("B2", seg_args("731m", "win-ctx", "6e-4", 0, "A2", [w, ck731], ab + " --stop_after 310"), xs),
          ("A3", seg_args("731m", "win-ctx", "6e-4", 0, "A3", [w, ck731], ab + " --stop_after 210 --lm_chunk_tokens 8192"), "")]
    line = lambda n, a, env: f'seg {n} env {env}{PY} {SMOKE}/smoke_seg.py {n} {a}' if env else f'seg {n} {PY} {SMOKE}/smoke_seg.py {n} {a}'
    body = "\n".join(line(n, a, env) + (" && RA=1 || RA=0" if n == "A" else "") for n, a, env in segs[:1])
    body += "\nif [ $RA = 1 ]; then\n  " + line(*segs[1]) + "\nelse\n  echo '=== A failed: fallbacks A2 / B2 (expandable_segments), A3 (--lm_chunk_tokens 8192)'\n  " + \
        line(*fb[0]) + " && " + line(*fb[1]) + "\n  " + line(*fb[2]) + "\nfi\n" + "\n".join(line(*x) for x in segs[2:])
    return f"""#!/bin/bash
#SBATCH --job-name=llmadd_psz-smoke731
#SBATCH --account=torch_pr_625_general
#SBATCH --partition=l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --nice=100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output={B}/logs/%x_%j.out
# ptsz smoke (runs/ptsz/AUDIT.md "Smoke"): 731m on L40S, memory with gradient checkpointing auto (in-window eval steps with
# the LM / prior gradient passes; a resume from a ckpt) and speed; 136m (grad ckpt off) memory. Test runs only: their run
# JSONs live in runs/ptsz/smoke and never enter a selection or read-out. Generated by make_jobs_ptsz.py --smoke.
set -u
if [ -e {P}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {P}/code7/tools
[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{PIN7_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
echo "=== smoke731 start $(date -Is) job ${{SLURM_JOB_ID:-none}} node $(hostname)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"; echo "$GPU"
case "$GPU" in *L40S*) ;; *) echo "not a L40S: exiting"; exit 4;; esac
(cd {DATA} && sha256sum -c -) <<'SHA' || exit 3
{M.J.DATA_SHA[0]}  train.bin
{M.J.DATA_SHA[1]}  val.bin
SHA
mkdir -p {SMOKE}
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -lms 1000 > {SMOKE}/gpu_${{SLURM_JOB_ID:-x}}.csv &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
seg() {{ n=$1; shift; echo "=== segment $n start $(date -Is)"; "$@"; rc=$?; echo "=== segment $n rc $rc $(date -Is)"; return $rc; }}
{body}
echo "=== smoke731 end $(date -Is)"
"""


def smoke_xs_job():
    """amendment 4: the 731m memory under the XS allocator setting, one short L40S job, the real jobs' export line at the top
    (every segment inherits it). XA = smoke A (win-ctx 6e-4 seed 0, --window 0,0.4 --eval_every 100 --ckpt_every 100
    --stop_after 210: in-window grad_info evals 100 / 200, two ckpt writes); XB = smoke B (the same command to 310: resumes XA's
    step-199 ckpt, the loaded checkpoint still referenced, in-window eval 300) = the segment that failed the margin in
    22011938; XC = the real none 2.4e-3 ARGS to 260 (evals 0 / 250). Same smoke_seg.py; out dirs runs/ptsz/smoke/XA, XC."""
    base = smoke_job()
    def seg_line(name):   # the segment lines A / B / C of the asserted smoke rendering, re-targeted to the XS dirs
        ls = [l.strip() for l in base.splitlines() if l.strip().startswith(f"seg {name} ")]
        assert len(ls) == 1, (name, ls)
        return ls[0].removesuffix(" && RA=1 || RA=0")
    a, b, c = seg_line("A"), seg_line("B"), seg_line("C")
    assert f"--out {SMOKE / 'A'} " in a and f"--out {SMOKE / 'A'} " in b and f"--out {SMOKE / 'C'} " in c
    assert a.endswith(" --stop_after 210") and b.endswith(" --stop_after 310") and c.endswith(" --stop_after 500")
    xa = a.replace("seg A ", "seg XA ", 1).replace(f"smoke_seg.py A ", "smoke_seg.py XA ", 1).replace(f"--out {SMOKE / 'A'} ", f"--out {SMOKE / 'XA'} ")
    xb = b.replace("seg B ", "seg XB ", 1).replace(f"smoke_seg.py B ", "smoke_seg.py XB ", 1).replace(f"--out {SMOKE / 'A'} ", f"--out {SMOKE / 'XA'} ")
    xc = c.replace("seg C ", "seg XC ", 1).replace(f"smoke_seg.py C ", "smoke_seg.py XC ", 1).replace(f"--out {SMOKE / 'C'} ", f"--out {SMOKE / 'XC'} ")
    xc = xc.removesuffix(" --stop_after 500") + " --stop_after 260"
    for x in (xa, xb, xc):
        assert " env " not in x and str(SMOKE / "A ") not in x and str(SMOKE / "C ") not in x, x
    head = base.split("\nseg() {")[0]
    assert head.count("#SBATCH --job-name=llmadd_psz-smoke731\n") == 1 and head.count("#SBATCH --time=02:00:00\n") == 1
    head = head.replace("#SBATCH --job-name=llmadd_psz-smoke731\n", "#SBATCH --job-name=llmadd_psz-smoke731xs\n").replace(
        "#SBATCH --time=02:00:00\n", "#SBATCH --time=01:00:00\n").replace(
        "# ptsz smoke (runs/ptsz/AUDIT.md \"Smoke\")", "# ptsz smoke XS (runs/ptsz/AUDIT.md \"Amendment 4\"; the first smoke's header follows)").replace(
        "Generated by make_jobs_ptsz.py --smoke.", "Generated by make_jobs_ptsz.py --smoke-xs.").replace(
        'echo "=== smoke731 start', f'export {XS_ENV}\necho "ALLOC: PYTORCH_CUDA_ALLOC_CONF=${{PYTORCH_CUDA_ALLOC_CONF:-unset}}"\necho "=== smoke731 start')
    assert head.count(f"export {XS_ENV}\n") == 1 and "llmadd_psz-smoke731xs" in head and "--smoke-xs." in head
    seg = 'seg() { n=$1; shift; echo "=== segment $n start $(date -Is)"; "$@"; rc=$?; echo "=== segment $n rc $rc $(date -Is)"; return $rc; }'
    assert seg in base
    return head + "\n" + seg + "\n" + xa + "\n" + xb + "\n" + xc + '\necho "=== smoke731 end $(date -Is)"\n'


SMOKE_SEG = '''"""ptsz smoke segment (make_jobs_ptsz.py --smoke writes this file): the trainer's own main() (cwd = runs/ptsz/code7/tools)
with the given args on the GPU; torch.save is timed (ckpt writes); then the CUDA peak memory. Prints one 'SMOKE {json}' line."""
import json
import sys
import time

import torch

sys.path.insert(0, ".")
import pretrain_range as pr   # noqa: E402

seg, args = sys.argv[1], sys.argv[2:]
saves = []
_save = torch.save


def timed_save(obj, f, *a, **k):
    t = time.time()
    _save(obj, f, *a, **k)
    saves.append(round(time.time() - t, 2))


torch.save = timed_save
pr.torch.save = timed_save
sys.argv = ["pretrain_range.py"] + args
t0, err = time.time(), None
try:
    pr.main()
except torch.cuda.OutOfMemoryError as e:
    err = "OOM: " + str(e)[:400]
except Exception as e:   # noqa: BLE001
    err = repr(e)[:400]
p = torch.cuda.get_device_properties(0)
print("SMOKE", json.dumps({"seg": seg, "ok": err is None, "err": err, "wall": round(time.time() - t0, 1), "ckpt_save_s": saves,
                           "max_alloc": torch.cuda.max_memory_allocated(), "max_reserved": torch.cuda.max_memory_reserved(),
                           "total": p.total_memory, "gpu": p.name, "args": args}), flush=True)
sys.exit(0 if err is None else 5)
'''


# ---------------------------------------------------------------- snapshot, cost, main
def snapshot():
    """runs/ptsz/code7 = git archive 64abc19 tools/ (make_jobs_pttpp._extract: once, temp dir + rename), asserted equal
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
    print(f"{'size':5s} {'shape':24s} {'n_params':>12s} {'N_nonemb':>12s} {'ln N':>7s} {'tpp':>7s} {'ckpt':>4s} {'every':>5s} "
          f"{'grid none':>32s} {'grid win-ctx':>30s} {'h/run':>6s} {'limit':>8s}")
    tot = {}
    for size in REGISTERED:
        F = facts(size, "win-ctx", GRID[size]["win-ctx"][0], 0, "A")
        d, nl, nh, dff = F["dims"]
        print(f"{size:5s} {f'd{d} x {nl}, {nh} heads, ff {dff}':24s} {F['n_params']:12,d} {F['nne']:12,d} {math.log(F['nne']):7.3f} "
              f"{F['tpp']:7.3f} {'on' if F['ckpt_on'] else 'off':>4s} {F['ckpt_every']:5d} {' / '.join(GRID[size]['none']):>32s} "
              f"{' / '.join(GRID[size]['win-ctx']):>30s} {F['est_h']:6.2f} {F['tlim']:>8s}")
        for a in ARMS:
            print(f"      {a:8s} interpolated {lr_pred(size, a):.3e}" + (f" (505M none = 9.6e-3 instead: "
                  f"{lr_pred(size, a, {**PICK3E8[a], '505m': 9.6e-3}):.3e})" if a == "none" and "505m" in NEIGH[size] else "")
                  + f"; neighbours {NEIGH[size][0]} {PICK3E8[a][NEIGH[size][0]]:.3e} / {NEIGH[size][1]} {PICK3E8[a][NEIGH[size][1]]:.3e}")
        tot[size] = (32 * F["est_h"], 8 * F["est_h"], 8 * F["est_h"])
    A_, C_, X_ = (sum(v[i] for v in tot.values()) for i in range(3))
    for size, (a, c, x) in tot.items():
        print(f"GPU-h {size}: stage A {a:6.1f} + C {c:5.1f} = {a + c:6.1f}; worst case + X on both arms {x:5.1f} -> {a + c + x:6.1f}")
    print(f"GPU-h total: stage A {A_:.1f} + C {C_:.1f} = {A_ + C_:.1f}; worst case + X {X_:.1f} -> {A_ + C_ + X_:.1f} (L40S-h)")
    for conc in (16, 12, 8):
        # stage A in BATCH order on `conc` GPUs (list scheduling, longest first), then X (worst case), then C
        free = [0.0] * conc
        for size in BATCH:
            for _ in range(32):
                i = free.index(min(free))
                free[i] += tot[size][0] / 32
        a_end = max(free)
        long = max(tot[s][0] / 32 for s in BATCH)
        print(f"wall at {conc} concurrent L40S: stage A {a_end:.1f} h (last 731m run ends by then), + C {long:.1f} h "
              f"(+ X {long:.1f} h worst case) -> {a_end + long:.1f}-{a_end + 2 * long:.1f} h of running, plus queue waits and the "
              f"selection by hand")
    gb = {s: CKPT_GB[s] for s in REGISTERED if ckpt_every_of(s)}
    print("transient ckpt per running run (model + 2 Adam moments): " + ", ".join(f"{s} {v:.1f} GB" for s, v in gb.items())
          + f"; stage C models (fp32): " + ", ".join(f"{s} 8 x {SIZES[s][4] * 4 / 1e9:.2f} GB" for s in REGISTERED)
          + f" = {sum(8 * SIZES[s][4] * 4 / 1e9 for s in REGISTERED):.1f} GB")


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    if "--plan" in sys.argv:
        plan()
        return
    if "--smoke-xs" in sys.argv:   # amendment 4: write runs/ptsz/smoke/job_smoke731xs.sbatch (submitted by hand)
        snapshot()
        jf, sf = SMOKE / "job_smoke731xs.sbatch", SMOKE / "smoke_seg.py"
        assert sf.read_text() == SMOKE_SEG, "smoke_seg.py differs from this rendering"
        txt = smoke_xs_job()
        flag = SMOKE / "SUBMITTED_smoke731xs"
        if flag.exists():
            assert jf.read_text() == txt, "the submitted XS smoke differs from this rendering"
        else:
            jf.write_text(txt)
        print(jf)
        if "--test-only" in sys.argv:
            o = subprocess.run(["sbatch", "--test-only", str(jf)], capture_output=True, text=True)
            print((o.stdout + o.stderr).strip())
        return
    if "--smoke" in sys.argv:
        snapshot()
        SMOKE.mkdir(parents=True, exist_ok=True)
        jf, sf = SMOKE / "job_smoke731.sbatch", SMOKE / "smoke_seg.py"
        txt = smoke_job()
        flag = SMOKE / "SUBMITTED_smoke731"
        if flag.exists():
            assert jf.read_text() == txt and sf.read_text() == SMOKE_SEG, "the submitted smoke differs from this rendering"
        else:
            jf.write_text(txt)
            sf.write_text(SMOKE_SEG)
        print(jf)
        if "--test-only" in sys.argv:
            o = subprocess.run(["sbatch", "--test-only", str(jf)], capture_output=True, text=True)
            print((o.stdout + o.stderr).strip())
        return
    au = P / "AUDIT.md"
    assert au.exists() and "## Pre-registered analysis" in au.read_text(), "runs/ptsz/AUDIT.md (the pre-registration) first"
    sizes = [opt("--size")] if opt("--size") else list(BATCH)
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
    print(f"ptsz {', '.join(sizes)}{' stage ' + stage if stage else ''}{' arm ' + arm if arm else ''}: {len(jobs)} jobs in {P}; "
          f"{'checked' if '--check' in sys.argv else 'test-only' if '--test-only' in sys.argv else 'submitted' if '--submit' in sys.argv else 'not submitted'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
