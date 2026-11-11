"""Torch jobs pt2sz (2026-11-05; the user's decision of 2026-11-05, relayed by the coordinator: the request's part P2, 2%
arithmetic at a fixed 3e8 tokens on Torch A100, sizes 93M and 294M, no 1.07B; priority ptdom > new sizes > ptshare, so every
job of this batch gets --nice=1000). Intent spec and PRE-REGISTERED analysis: runs/pt2sz/AUDIT.md.
FINDINGS F98 (pt2same) compares the 2% prior gain at 3e8 tokens on one cluster (Torch A100-80GB) at three sizes, 53M / 190M
(pt2same) and 505M (pt2s505). This batch adds two sizes on the same cluster, GPU type, trainer and job body, the pttpp shapes:
  93M  = d512 x 10, 8 heads, d_ff 2048 (N_nonemb 41,964,032; gradient checkpointing off)
  294M = d832 x 19, 13 heads, d_ff 3328 (N_nonemb 210,500,160; gradient checkpointing on)
Job body = pt2same's (= pt2s505's + the A100-80GB guard + the INTENT line): HOLD kill switch, the snapshot's shas, the GPU
guard (exit 3 on anything but an A100-80GB), the data sha256, --dry_inputs (model line, 2 operand sites per row, target table,
the 9 DRY rows' sha256 = the Burst pt2s190 rows of the seed with this model line = this batch's CPU dry run,
llmadd_torch/pt2sz_refs.json), training (--save_model --ckpt_every 2000, --requeue resumes), offline_lm.py (full val),
make_pt2main_tasks.CHECK_PY verbatim with this run's intent -> check.json; then (stage A / X only, seeds 0-3) the model file
is removed after a passing check (pttpp job2's rule: offline_lm has used it; AUDIT.md "Disk"); the job exits with the
check's code. Account <slurm-account>, a100_tandon, gpu48, nice 1000; job names llmadd_p2z-<run>.
Trainer = pt2same's / pt2s505's snapshot: runs/pt2sz/code = `git archive 7f1d06b tools/` (pretrain_range.py
93e22bf4c4c0), asserted equal to runs/pt2s505/code and runs/pt2same/code file by file.
lr selection = pt300x's protocol (AUDIT.md "lr selection"): stage A = 4 consecutive ladder rungs per arm x seeds 0-3;
select_pt2sz.py (by hand) applies F87 rule 1 and appends X (one rung beyond an open edge, seeds 0-3, once per arm), C
(confirm seeds 4-7 at the pick) or STOP to runs/pt2sz/decisions.json (append-only); this generator then renders those runs.
Reused read-only (sha256[:12] asserted; importing them runs no main(), asserted by ast): make_jobs_pt2same.py (its intent()
is the known answer of this batch's intent at 53M / 190M; GPU guard, account, the 2% picks), make_jobs_pttpp.py (SIZES,
MODEL_LINE, LADDER, intent2, dry2, _extract), make_jobs_pt2s505.py (intent / render: known answers at 505M), make_jobs_rb6scale.
A submitted job file is never rewritten (SUBMITTED_<run> flags). This generator never submits unless --submit is given.
    python make_jobs_pt2sz.py --plan                                     design, grids, cost, wall time
    python make_jobs_pt2sz.py [--stage A|X|C] [--size S] [--arm A] [--print] [--test-only | --submit] [--run N1,N2]
      (--run: only these runs are tested / submitted, e.g. one run lost to an infrastructure failure, AUDIT.md rule 5)
    python make_jobs_pt2sz.py --check [--stage A]                        job files = rendering, snapshot, frozen shas
    python make_jobs_pt2sz.py --names [--stage A] [--size S]             run names in submission order (submit_pt2sz.sh)
"""
import ast
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
# read-only inputs (sha256[:12]) = runs/pt2same/FROZEN.sha256's values and pt300x's
FROZEN = {"make_jobs_pt2same.py": "84b1f1103231", "make_jobs_pttpp.py": "b4bfa6eb248d", "make_jobs_pt2s505.py": "a5a7fb60eedc",
          "make_jobs_rb6scale.py": "19bc055c2d09", "make_jobs_ptlr1b.py": "8dd7b3407459", "pt2s505_refs.json": "479e394771e9",
          "pttpp_refs.json": "d10d720c1242", "rb6scale_refs.json": "b65472eb3d29",
          "runs/pt2same/code/tools/pretrain_range.py": "93e22bf4c4c0", "runs/pt2same/code/tools/make_pt2main_tasks.py": "a2e8a9621383",
          "runs/pt2same/code/tools/readout_rules.py": "e91ee8a8a846", "runs/pt2same/code/tools/offline_lm.py": "226f4cac70ea",
          "runs/pt2s505/picks_confirm.json": "713ee399cce4"}
for _f, _v in FROZEN.items():
    assert sha(HERE / _f) == _v, (_f, sha(HERE / _f), _v, "a read-only input changed")
for _f, _ok in (("make_jobs_pt2same.py", ("sys.path.insert(0, str(HERE))",)),):   # importing it runs no main()
    _t = ast.parse((HERE / _f).read_text())
    _calls = [ast.unparse(n) for n in _t.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    _g = [n for n in _t.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
    assert tuple(_calls) == _ok and len(_g) == 1 and [ast.unparse(x) for x in _g[0].body] == ["main()"], (_f, _calls)
sys.path.insert(0, str(HERE))
import make_jobs_pt2same as PS   # noqa: E402  (asserts its own frozen inputs; imports make_jobs_pttpp / pt2s505 / rb6scale)
TP, P2, J = PS.TP, PS.P2, PS.J

B = PS.B
R = B / "runs" / "pt2sz"
BLOCK, PREFIX = "pt2sz", "p2z"
TOK, PY, DATA = J.TOK, J.PY, J.DATA
assert B == HERE and DATA == B / "pretrain_data" and R.parent == B / "runs"
WANT_SHA = PS.WANT_SHA                       # the 2% trainer 93e22bf4c4c0 and every module it imports (pt2same's = pt2s505's)
assert WANT_SHA is P2.WANT_SHA is TP.WANT_SHA2 and WANT_SHA["pretrain_range.py"] == "93e22bf4c4c0"
PIN2, PIN2_CODE = PS.PIN2, PS.PIN2_CODE      # 7f1d06b, runs/pt2s505/code
EXTRA_SHA = PS.EXTRA_SHA                     # make_pt2main_tasks / readout_rules / pt2main_summary of the snapshot
ACCOUNT, PARTITION, QOS, GPU_CASE = PS.ACCOUNT, PS.PARTITION, PS.QOS, PS.GPU_CASE
assert (ACCOUNT, PARTITION, QOS, GPU_CASE) == ("<slurm-account>", "a100_tandon", "gpu48", "*A100*80GB*")
NICE = 1000                                  # the user (2026-11-05): ptdom > new sizes > ptshare (2000); pt2same had 20
assert PS.NICE == 20 and NICE != PS.NICE

# ---------------------------------------------------------------- design (pre-registered: runs/pt2sz/AUDIT.md)
SIZES = ("93m", "294m")
ARMS = ("none", "win-ctx")
SEL, CONF = (0, 1, 2, 3), (4, 5, 6, 7)
DIMS = {z: TP.SIZES[z] for z in SIZES}       # (d_model, n_layers, n_heads, d_ff, n_params, N_nonemb)
assert DIMS["93m"] == (512, 10, 8, 2048, 93306368, 41964032) and DIMS["294m"] == (832, 19, 13, 3328, 293931456, 210500160)
MODEL_LINE = {z: TP.MODEL_LINE[z] for z in SIZES}
assert MODEL_LINE == {"93m": "model: 93.3M parameters; gradient checkpointing off",
                      "294m": "model: 293.9M parameters; gradient checkpointing on"}
STEPS, TOK_STEP, EVAL_EVERY, N_EVALS = 18310, 32 * 512, 250, 75
TOKENS_ARG = 3e8                             # the trainer's default (pt2same / pt2s505 pass no --tokens / --eval_every)
assert int(TOKENS_ARG // TOK_STEP) == STEPS == TP.STEPS_3E8 and EVAL_EVERY == TP.EVAL_3E8
WINDOW_EVENTS = [[916, 1], [7324, 0]]
# the factor-2 ladder of pttpp / pt300x, plus one rung below it (7.5e-5) so that an X below a 1.5e-4 edge can be rendered
LADDER = ("7.5e-5",) + TP.LADDER
assert all(abs(float(b) / float(a) - 2) < 1e-9 for a, b in zip(LADDER, LADDER[1:]))
# the 2% picks at 3e8 tokens of the neighbouring sizes (none / win-ctx): 53M and 190M = F86's (pt2same re-ran them on
# Torch at these lrs), 505M = pt2s505's (picks_confirm.json); asserted against both sources and make_jobs_pttpp.OPT3E8
PICKS_3E8 = {"53m": {"none": "8e-3", "win-ctx": "2e-3"}, "190m": {"none": "4.8e-3", "win-ctx": "6e-4"},
             "505m": {"none": "4.8e-3", "win-ctx": "6e-4"}}
assert {z: PICKS_3E8[z] for z in PS.SIZES} == PS.PICKS
assert json.loads((B / "runs/pt2s505/picks_confirm.json").read_text())["best"] == PICKS_3E8["505m"]
assert all(float(PICKS_3E8[z][a]) == TP.OPT3E8[("2", a)][z] for z in PICKS_3E8 for a in ARMS)
GRID = {"93m": {"none": ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"), "win-ctx": ("3e-4", "6e-4", "1.2e-3", "2.4e-3")},
        "294m": {"none": ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"), "win-ctx": ("1.5e-4", "3e-4", "6e-4", "1.2e-3")}}


def lr_interp(size, arm):
    """the pre-registered interpolation (pttpp's rule without its token factor, as pt300x): log-linear in ln N_nonemb
    between the 3e8 2% picks of the neighbouring sizes"""
    nn = {z: TP.SIZES[z][5] for z in PICKS_3E8}
    lo = max((z for z in nn if nn[z] < TP.SIZES[size][5]), key=nn.get)
    hi = min((z for z in nn if nn[z] > TP.SIZES[size][5]), key=nn.get)
    f = (math.log(TP.SIZES[size][5]) - math.log(nn[lo])) / (math.log(nn[hi]) - math.log(nn[lo]))
    a, b = float(PICKS_3E8[lo][arm]), float(PICKS_3E8[hi][arm])
    return math.exp(math.log(a) + f * (math.log(b) - math.log(a))), lo, hi


for _s, _g2 in GRID.items():   # every grid: 4 consecutive rungs, the interpolated lr inside with >= 1 rung strictly on each side
    for _a, _lrs in _g2.items():
        _i = LADDER.index(_lrs[0])
        assert _lrs == LADDER[_i:_i + 4], (_s, _a, _lrs)
        _p = lr_interp(_s, _a)[0]
        assert any(float(v) < _p * (1 - 1e-9) for v in _lrs) and any(float(v) > _p * (1 + 1e-9) for v in _lrs), (_s, _a, _p)
assert all(abs(lr_interp(s, a)[0] / v - 1) < 1e-4 for (s, a), v in {("93m", "none"): 6.1264e-3, ("93m", "win-ctx"): 1.0664e-3,
           ("294m", "none"): 4.8e-3, ("294m", "win-ctx"): 6e-4}.items()), [lr_interp(s, a)[0] for s in SIZES for a in ARMS]
# cost (AUDIT.md "Cost"): A100-80GB wall hours per run at 18310 steps / 75 evals, from run JSONs (curve[-1]['sec'], medians)
# + start-up / dry run / offline_lm / check (sacct elapsed - training: 53M 150 s, 190M 170 s, tpp2-294m 185 s, 505M 200 s):
#   93M: log-log in n_params through pt2same 53M (2300 s) and 190M (6740 s) -> 3702 s (+160 s) = 1.07 h; the other route
#        (53M A100 x the 7% L40S 93M / 53M ratio 1.4325 at 3e8) gives 3295 s (+160 s) = 0.96 h -> the larger, 1.07 h
#   294M: pttpp tpp2-294m on A100-80GB (9570 steps, 75 evals, 5708 s) scaled by steps 18310 / 9570 -> 10921 s (+190 s) =
#        3.09 h; the eval-split route (190M A100: 0.353 s/step + 3.49 s/eval, scaled) gives 2.98 h -> the larger, 3.09 h
PLAN_H = {"93m": 1.07, "294m": 3.09}
TLIM = {"93m": "03:00:00", "294m": "08:00:00"}   # 2.5 x plan, rounded up to the half hour (pt2same's / pttpp's rule)
for _s in SIZES:
    _h = math.ceil(2.5 * PLAN_H[_s] * 2) / 2
    assert TLIM[_s] == f"{int(_h):02d}:{int(round(60 * (_h % 1))):02d}:00", (_s, _h)
SUBMIT_ORDER = ("294m", "93m")   # the user / coordinator: 294m stage A first (longer runs), then 93m


def step_out(lr, edge):
    i = LADDER.index(lr)
    assert (edge == "upper" and i + 1 < len(LADDER)) or (edge == "lower" and i > 0), (lr, edge)
    return LADDER[i + 1] if edge == "upper" else LADDER[i - 1]


def tag_of(size, arm, lr):
    return f"{PREFIX}-{size}-lr{lr}-{arm}"


def target_of(arm):
    return "none" if arm == "none" else "helix3"


def run_paths(size, arm, lr, seed, root=None):
    """(run JSON, check.json, model file) of one run under <root>/pt2sz/out/<run>/ (= the job's $OUT)"""
    tag = tag_of(size, arm, lr)
    out = (Path(root) / "pt2sz" if root else R) / "out" / f"{tag}-s{seed}"
    base = f"whole_{target_of(arm)}_seed{seed}_{tag}"
    return out / tag / f"run_{base}.json", out / "check.json", out / tag / f"model_{base}.pt"


# ---------------------------------------------------------------- references (CPU, scale_work/pt2sz/refs_pt2sz.py)
REFS = B / "pt2sz_refs.json"   # {"lm0": {"2|93m": {seed: v}, "2|294m": ...}, "dry2": {"93m": {seed: sha256}, ...}}


def refs():
    return json.loads(REFS.read_text()) if REFS.exists() else {"lm0": {}, "dry2": {}}


def lm0_ref(size, seed):
    v = refs()["lm0"].get(f"2|{size}", {}).get(str(seed))
    return None if v is None else float(v)


# ---------------------------------------------------------------- decisions (select_pt2sz.py --write appends; append-only)
def decisions(path=None):
    """[{seq, size, arm, stage X | C | STOP, lr, ...}] validated: per (size, arm) at most one X (one ladder rung beyond a
    stage-A edge), then at most one C (lr in the grid or the X lr) or a STOP; nothing after a C or a STOP"""
    f = Path(path) if path else R / "decisions.json"
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
    """[(stage, arm, lr, seed)]: stage A seed-major (seed, arm, lr ascending: the first wave shows every cell), then the
    decided X / C runs in decision order"""
    out = [("A", a, lr, s) for s in SEL for a in ARMS for lr in GRID[size][a]]
    for e in decisions(path)[0]:
        if e["size"] == size and e["stage"] in ("X", "C"):
            out += [(e["stage"], e["arm"], e["lr"], s) for s in (SEL if e["stage"] == "X" else CONF)]
    assert len(set(out)) == len(out), size
    return out


# ---------------------------------------------------------------- per run facts, ARGS, intent
def fmt_h(h):
    return f"{int(h):02d}:{int(round(60 * (h % 1))):02d}:00"


def facts(size, arm, lr, seed, stage, G2=None, tag=None, lm0="ref"):
    """everything a job is rendered from; size may also be 53m / 190m / 505m (known answers: pt2same / pt2s505)"""
    assert arm in ARMS and stage in ("A", "X", "C") and (seed in SEL) == (stage != "C"), (size, arm, stage, seed)
    d, nl, nh, dff, n_params, nne = TP.SIZES[size]
    ev, on = TP.window_events(STEPS) if arm != "none" else ([], 0.0)
    if G2 is not None:
        assert (ev, on) == (G2.window_events(STEPS) if arm != "none" else ([], 0.0))
    tag = tag or tag_of(size, arm, lr)
    return dict(size=size, arm=arm, lr=lr, seed=seed, stage=stage, S=STEPS, E=EVAL_EVERY, T=TOKENS_ARG, evals=N_EVALS,
                events=ev, on_frac=on, dims=(d, nl, nh, dff), n_params=n_params, nne=nne, tpp=STEPS * TOK_STEP / nne,
                warm=max(1, STEPS // 100), layers=[nl // 3, 2 * nl // 3], ckpt_on=n_params > 150e6, ckpt_every=2000,
                lm0=lm0_ref(size, seed) if lm0 == "ref" else lm0, tag=tag, name=f"{tag}-s{seed}",
                plan_h=PLAN_H.get(size), tlim=TLIM.get(size), keep_model=(stage == "C"))


def model_flags(dims):
    d, nl, nh, dff = dims
    return f"--d_model {d} --n_layers {nl} --n_heads {nh} --d_ff {dff}"


def args_of(G2, F, out):
    """the trainer ARGS (pt2same's / pt2s505's 3e8 form: no --tokens / --eval_every; --save_model --ckpt_every 2000); every
    flag asserted (make_pt2main_tasks.render's want table), nothing else"""
    args = (f"--data {DATA} --tokenizer {TOK} --tokenization whole --out {out}/{F['tag']} {G2.ARM[F['arm']]} --arith_lines 1 "
            f"{model_flags(F['dims'])} --lr {F['lr']} --seed {F['seed']} --tag _{F['tag']} --save_model --ckpt_every 2000")
    f = G2.flags(args)
    d, nl, nh, dff = F["dims"]
    want = {"--data": str(DATA), "--tokenizer": str(TOK), "--tokenization": "whole", "--out": f"{out}/{F['tag']}",
            "--arith_lines": "1", "--lr": F["lr"], "--seed": str(F["seed"]), "--tag": f"_{F['tag']}", "--save_model": True,
            "--ckpt_every": "2000", "--tokens": None, "--lam": None, "--geo_layers": None, "--bs": None, "--seq": None,
            "--eval_every": None, "--n_eval": None, "--gate_thr": None, "--gate_start": None, "--gate_hyst": None,
            "--untie": None, "--grad_ckpt": None, "--lm_chunk_tokens": None, "--stop_after": None, "--dry_inputs": None,
            **G2.WANT[F["arm"]], "--d_model": str(d), "--n_layers": str(nl), "--n_heads": str(nh), "--d_ff": str(dff)}
    for k, v in want.items():
        assert f.get(k) == v, (F["name"], k, f.get(k), v)
    assert set(f) <= {k for k, v in want.items() if v is not None}, (F["name"], sorted(f))
    assert float(f["--lr"]) == float(F["lr"])
    return args


def intent_of(G2, F):
    """CHECK_PY's intent = make_jobs_pttpp.intent2 (frozen) with the 3e8 facts (tokens 3e8, eval every 250, ckpt every
    2000); known answers asserted in known_answers(): = make_jobs_pt2s505.intent at 505M, = make_jobs_pt2same.intent at
    53M / 190M"""
    I = TP.intent2(G2, F)
    a = I["args"]
    assert (a["tokens"], a["eval_every"], a["arith_lines"], a["lam"], a["window"], a["site"], a["bs"], a["seq"], a["ckpt_every"],
            a["save_model"]) == (3e8, 250, 1, 5.0, "0.05,0.4", "ctx", 32, 512, 2000, True), a
    assert I["steps"] == STEPS and I["evals"] == N_EVALS and I["events"] == (WINDOW_EVENTS if F["arm"] != "none" else [])
    assert I["code_sha"] == WANT_SHA["pretrain_range.py"]
    return I


def known_answers(G2):
    """this batch's ARGS / intent builders reproduce the submitted pt2s505 (505M) and pt2same (53M, 190M) jobs' exactly
    (up to out / tag, which are passed); called by main() and the tests"""
    pin = WANT_SHA["pretrain_range.py"]
    for arm, lr, seed in (("none", "4.8e-3", 0), ("win-ctx", "6e-4", 5), ("none", "1.92e-2", 2), ("win-ctx", "3e-4", 3)):
        name, tag, text, args, I = P2.render(G2, arm, lr, seed, pin)
        F = facts("505m", arm, lr, seed, "C" if seed >= 4 else "A", G2, tag=tag, lm0=P2.LM0[("505m", "pretrain_data")][seed])
        assert intent_of(G2, F) == I, ("intent != make_jobs_pt2s505.intent", name)
        assert args_of(G2, F, P2.R / "out" / name) == args, ("ARGS != make_jobs_pt2s505's", name)
        assert text == (P2.R / f"job_{name}.sbatch").read_text(), name   # the submitted pt2s505 job file
    for size in PS.SIZES:
        for arm in ARMS:
            lr = PICKS_3E8[size][arm]
            for seed in (0, 6):
                tag = PS.tag_of(size, arm, lr)
                F = facts(size, arm, lr, seed, "C" if seed >= 4 else "A", G2, tag=tag, lm0=PS.lm0_ref(G2, size, seed))
                assert intent_of(G2, F) == PS.intent(G2, size, arm, lr, seed), ("intent != make_jobs_pt2same.intent", tag, seed)
                job = (PS.R / f"job_{tag}-s{seed}.sbatch").read_text()
                theirs = [l for l in job.splitlines() if l.startswith('ARGS="')][0][6:-1]
                mine = args_of(G2, F, PS.R / "out" / f"{tag}-s{seed}")
                if size == "53m":   # pt2same's 53M passes no model flags (the trainer default = d384 x 6)
                    mine = mine.replace(f" {model_flags(F['dims'])}", "")
                assert mine == theirs, ("ARGS != the submitted pt2same job's", tag, seed, mine, theirs)
    return True


# ---------------------------------------------------------------- the job (pt2same's body + the stage A / X model removal)
def g2():
    """make_pt2main_tasks of this batch's snapshot (CHECK_PY, DRY_GREP, DATA, ARM, WANT, flags, hours, window_events)"""
    d = str(R / "code" / "tools")
    if d not in sys.path:
        sys.path.insert(0, d)
    import make_pt2main_tasks as G2
    assert Path(G2.__file__).resolve().parent == (R / "code" / "tools").resolve(), G2.__file__
    return G2


def render_one(G2, F):
    """(job text, ARGS, intent) of one run; every intended fact asserted"""
    assert F["size"] in SIZES and F["lr"] in LADDER
    OUT = R / "out" / F["name"]
    args = args_of(G2, F, OUT)
    I = intent_of(G2, F)
    assert I["lm0_ref"] is not None, (F["name"], "no step-0 lm reference: run scale_work/pt2sz/refs_pt2sz.py table --write first")
    dry = TP.dry2(G2, F["size"], F["seed"])   # the Burst pt2s190 rows of the seed with this model line
    rd = refs()["dry2"].get(F["size"], {}).get(str(F["seed"]))
    assert rd == dry, (F["name"], "the Torch CPU dry run of this size and seed != the Burst pt2s190 rows + model line", rd, dry)
    assert F["plan_h"] * 2 <= G2.hours(F["tlim"]) <= F["plan_h"] * 3 + 0.5, (F["size"], F["tlim"])
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    site_assert = ("""[ "$(grep -c '^train row [012] (len 512, 2 prior sites at ' "$OUT/dry_inputs.txt")" = 3 ] """
                   """|| { echo "## not 1 arithmetic line (2 operand sites) per row"; exit 3; }""")
    target_assert = ("""! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a prior target in a none run"; exit 3; }"""
                     if F["arm"] == "none" else
                     """grep -qxF 'target features (1000, 10)' "$OUT/dry_inputs.txt" || { echo "## not the helix3 table (1000 x 10: cos 2, cos/sin 5 10 100 1000, x / 1000)"; exit 3; }""")
    run_json, _, model = run_paths(F["size"], F["arm"], F["lr"], F["seed"])
    assert run_json.parent == OUT / F["tag"]
    dep_chk = "\n".join(f'[ "$(sha256sum {fn} | cut -c1-12)" = "{v}" ] || {{ echo "## {fn} sha mismatch: exiting"; exit 3; }}'
                        for fn, v in WANT_SHA.items())
    d, nl, nh, dff = F["dims"]
    ev = F["events"]
    win = f"steps {ev[0][0]}..{ev[1][0] - 1}" if ev else "none (no prior)"
    keep = ("the model is kept (stage C)" if F["keep_model"] else
            f"the model is removed after a passing check (stage {F['stage']}, seed {F['seed']} <= 3)")
    about = textwrap.fill(f"pt2sz (2% size comparison on one cluster, two more sizes for F98): run {F['name']}, stage "
                          f"{F['stage']} of the lr selection (AUDIT.md 'lr selection'); {G2.ARM_DESC[F['arm']]}, lr {F['lr']}, "
                          f"seed {F['seed']}; {F['size']} (d{d} x {nl}, {nh} heads, d_ff {dff}), 2% arithmetic (1 line per row), "
                          "lambda 5, 3e8 tokens; pt2same's trainer and job body on Torch A100-80GB. Intent spec and "
                          "pre-registered analysis runs/pt2sz/AUDIT.md; generated by llmadd_torch/make_jobs_pt2sz.py (change "
                          "the generator, not this file).", 120, initial_indent="# ", subsequent_indent="# ",
                          break_on_hyphens=False, break_long_words=False)
    drop = ("" if F["keep_model"] else
            f'# stage {F["stage"]} (seed {F["seed"]} <= 3): the model is not kept once offline_lm and the checks used it (AUDIT.md "Disk")\n'
            f'if [ "$RC" = 0 ]; then rm -f "$OUT/$TAG/{model.name}"; echo "## model removed (stage {F["stage"]})"; fi\n')
    text = f"""#!/bin/bash
#SBATCH --job-name=llmadd_{F['name']}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --qos={QOS}
#SBATCH --gres=gpu:1
#SBATCH --nice={NICE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={F['tlim']}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
{about}
# 1. HOLD, snapshot shas, A100-80GB, data sha256; --dry_inputs: model line, 2 operand sites per row, target table, the 9 rows'
#    sha256 = the Burst pt2s190 rows of this seed with this model line = pt2sz_refs.json (exit 3 before training)
# 2. training (--save_model --ckpt_every 2000; a requeued job resumes); 3. offline_lm.py (full val);
# 4. make_pt2main_tasks.CHECK_PY with this run's intent -> $OUT/check.json; gpu.csv every 60 s; 5. {keep};
# the job exits with the check's code (4 on a fail)
set -euo pipefail
R={R}; OUT={OUT}; TAG={F['tag']}; SEED={F['seed']}; DATA={DATA}; TOK={TOK}; PY={PY}
# kill switch (no scancel from the agent): a pending job exits at start when runs/pt2sz/HOLD exists
if [ -e "$R/HOLD" ]; then echo "HOLD present: exiting"; exit 0; fi
cd "$R/code/tools"
{dep_chk}
grep -q "pretrain_range.py {WANT_SHA['pretrain_range.py']}" "$R/code/COMMIT" || {{ echo "## COMMIT does not name the trainer {WANT_SHA['pretrain_range.py']}"; exit 3; }}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU: $GPU"
case "$GPU" in {GPU_CASE}) ;; *) echo "## not an A100-80GB (this batch's one GPU type: Torch a100_tandon): exiting"; exit 3;; esac
mkdir -p "$OUT"
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
[ -s "$OUT/gpu.csv" ] || echo "timestamp, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]" > "$OUT/gpu.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader -l 60 >> "$OUT/gpu.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
{{ date -u +%FT%TZ; hostname; echo "SLURM_JOB_ID ${{SLURM_JOB_ID:-}} restart ${{SLURM_RESTART_COUNT:-0}}"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "cpus $(nproc)"; }} >> "$OUT/node.txt" 2>&1 || true

echo "## 1. inputs $(date -u +%FT%TZ)"
(cd "$DATA" && sha256sum -c -) <<'SHA' || exit 3
{G2.DATA['pretrain_data'][0]}  train.bin
{G2.DATA['pretrain_data'][1]}  val.bin
SHA
echo "INTENT {F['name']}: {F['size']} d{d} x {nl} ({nh} heads, d_ff {dff}), {F['n_params']} params, N_nonemb {F['nne']}, grad ckpt {'on' if F['ckpt_on'] else 'off'}; tokens 3e8 = {STEPS} steps x {TOK_STEP}, eval every {EVAL_EVERY} steps -> {I['evals']} evals, warmup {F['warm']} steps, prior window {win}; {F['arm']} lr {F['lr']} seed {F['seed']}; stage {F['stage']}; step-0 lm ref {I['lm0_ref']:.6f}; {keep}"
ARGS="{args}"
"$PY" pretrain_range.py $ARGS --dry_inputs > "$OUT/dry_inputs.txt" 2>&1 || {{ cat "$OUT/dry_inputs.txt"; exit 3; }}
cat "$OUT/dry_inputs.txt"
grep -qxF '{MODEL_LINE[F["size"]]}' "$OUT/dry_inputs.txt" || {{ echo "## not the intended model size / gradient checkpointing state"; exit 3; }}
{site_assert}
{target_assert}
got=$(set +o pipefail; grep -E "{G2.DRY_GREP}" "$OUT/dry_inputs.txt" | head -n 9 | sha256sum | cut -d' ' -f1)
[ "$got" = {dry} ] || {{ echo "## dry_inputs differ from the reference rows (sha256 $got)"; exit 3; }}
echo "## dry_inputs rows = the reference ({dry[:12]})"

echo "## 2. training $(date -u +%FT%TZ)"
"$PY" pretrain_range.py $ARGS
echo "## 3. offline LM $(date -u +%FT%TZ)"
"$PY" offline_lm.py --data "$DATA" --tokenizer "$TOK" --out "$OUT/offline_lm.json" "$OUT/$TAG"
echo "## 4. intent checks $(date -u +%FT%TZ)"
RC=0
"$PY" - "$OUT/$TAG/{run_json.name}" "$OUT/offline_lm.json" "$OUT/gpu.csv" "$R/code/COMMIT" "$DATA" \\
  "$OUT/check.json" '{ij}' <<'PY' | tee "$OUT/check.txt" || RC=$?
{G2.CHECK_PY}PY
{drop}echo "## done $(date -u +%FT%TZ) (check rc $RC)"
exit $RC
"""
    return text, args, I


def render(size, path=None, G2=None):
    """[(run name, job text, facts)] of a size's current stages (stage A, then the decided X / C)"""
    G2 = G2 or g2()
    out = []
    for st, a, lr, s in runs_of(size, path):
        F = facts(size, a, lr, s, st, G2)
        out.append((F["name"], render_one(G2, F)[0], F))
    return out


# ---------------------------------------------------------------- snapshot, plan, main
def snapshot():
    """runs/pt2sz/code = git archive 7f1d06b tools/ (make_jobs_pttpp._extract: once, temp dir + rename), asserted equal file
    by file to runs/pt2s505/code and runs/pt2same/code, COMMIT verbatim; every trainer module = WANT_SHA, the read-out
    modules = EXTRA_SHA; then read-only (chmod a-w)"""
    R.mkdir(parents=True, exist_ok=True)
    dst = R / "code"
    TP._extract(dst, PIN2, PIN2_CODE, (PIN2_CODE / "COMMIT").read_text())
    TP._extract(dst, PIN2, PS.R / "code", (PS.R / "code" / "COMMIT").read_text())
    cm = (dst / "COMMIT").read_text()
    for f, v in {**WANT_SHA, **EXTRA_SHA}.items():
        assert sha(dst / "tools" / f) == v and f"{f} {v}" in cm, (f, sha(dst / "tools" / f), v)
    for root, dirs, files in os.walk(dst):
        for n in dirs + files:
            p = Path(root) / n
            if not p.is_symlink():
                p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    dst.chmod(dst.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return cm.split()[0]


def plan():
    print(f"{'size':5s} {'shape':24s} {'n_params':>12s} {'N_nonemb':>12s} {'tpp':>6s} {'layers':>7s} {'ckpt':>4s} "
          f"{'grid none (interp)':>40s} {'grid win-ctx (interp)':>38s} {'h/run':>6s} {'limit':>8s}")
    for z in SIZES:
        d, nl, nh, dff, n, nne = TP.SIZES[z]
        pn, pw = lr_interp(z, "none"), lr_interp(z, "win-ctx")
        print(f"{z:5s} {f'd{d} x {nl}, {nh} heads, ff {dff}':24s} {n:12,d} {nne:12,d} {STEPS * TOK_STEP / nne:6.2f} "
              f"{str([nl // 3, 2 * nl // 3]):>7s} {'on' if n > 150e6 else 'off':>4s} "
              f"{'/'.join(GRID[z]['none']) + f' ({pn[0]:.3g})':>40s} {'/'.join(GRID[z]['win-ctx']) + f' ({pw[0]:.3g})':>38s} "
              f"{PLAN_H[z]:6.2f} {TLIM[z]:>8s}")
    print("stage    93m runs  294m runs  A100-h (93m x 1.07 + 294m x 3.09)")
    rows = [("A", 32, 32), ("C", 8, 8), ("A + C", 40, 40), ("X worst case (all 4 arms)", 8, 8), ("A + X + C worst case", 48, 48)]
    for k, a, b in rows:
        print(f"{k:28s} {a:3d} {b:9d}   {a * PLAN_H['93m'] + b * PLAN_H['294m']:6.1f}")
    for conc in (8, 16):
        wa = sum(math.ceil(n / conc) * PLAN_H[z] for z, n in (("294m", 32), ("93m", 32)))
        print(f"wall of running at {conc} concurrent A100s (no queue waits): stage A ~{32 * (PLAN_H['93m'] + PLAN_H['294m']) / conc:.1f} h "
              f"(waves: {wa:.1f} h), X + C (294m-bound) ~{2 * PLAN_H['294m']:.1f} h; plus the selection by hand and queue waits")


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    if "--plan" in sys.argv:
        plan()
        return
    au = R / "AUDIT.md"
    assert au.exists() and "## Pre-registered analysis" in au.read_text(), "runs/pt2sz/AUDIT.md (the pre-registration) first"
    sizes = [opt("--size")] if opt("--size") else list(SUBMIT_ORDER)
    stage, arm = opt("--stage"), opt("--arm")
    assert all(s in GRID for s in sizes) and stage in (None, "A", "X", "C") and arm in (None,) + ARMS, (sizes, stage, arm)
    if "--names" in sys.argv:   # the selection's run names in rendering (= submission) order; writes nothing
        G2 = g2()
        for size in sizes:
            for name, txt, F in render(size, G2=G2):
                if not ((stage and F["stage"] != stage) or (arm and F["arm"] != arm)):
                    print(name)
        return
    head = snapshot()
    G2 = g2()
    known_answers(G2)
    existing = {p.name for p in B.glob("runs/*/out*/*") if p.parent.parent != R}   # other batches' run dirs
    bad, jobs = 0, []
    for size in sizes:
        for name, txt, F in render(size, G2=G2):
            if (stage and F["stage"] != stage) or (arm and F["arm"] != arm):
                continue
            assert name not in existing and F["tag"] not in existing, name
            job, flag = R / f"job_{name}.sbatch", R / f"SUBMITTED_{name}"
            if "--check" in sys.argv:   # read-only: every job file of the selection = this rendering
                if not job.exists() or job.read_text() != txt:
                    print(f"{'MISSING' if not job.exists() else 'DIFFERS'}: {job.name}")
                    bad += 1
            elif flag.exists():
                if not job.exists():
                    print(f"WARNING: {job.name} was submitted but its file is missing: not rewritten")
                elif job.read_text() != txt:
                    print(f"WARNING: {job.name} was submitted and differs from this rendering: kept, not rewritten")
            elif not job.exists() or job.read_text() != txt:
                job.write_text(txt)
            jobs.append((name, job, flag, F))
    if "--check" not in sys.argv:
        names = {f"job_{n}.sbatch" for n, *_ in jobs}
        if not (stage or arm or opt("--size")):   # a full rendering: no job file that this rendering does not make
            stray = sorted(p.name for p in R.glob("job_*.sbatch") if p.name not in names)
            assert not stray, ("job files not made by this rendering", stray)
    if "--print" in sys.argv:
        for name, job, flag, F in jobs:
            print(name, F["stage"], [l for l in job.read_text().splitlines() if l.startswith("ARGS=")][0])
    runs = opt("--run").split(",") if opt("--run") else None   # --test-only / --submit only these runs
    assert runs is None or set(runs) <= {n for n, *_ in jobs}, ("--run names not in this selection", runs)
    for name, job, flag, F in jobs:
        if runs and name not in runs:
            continue
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
    print(f"pt2sz {', '.join(sizes)}{' stage ' + stage if stage else ''}{' arm ' + arm if arm else ''}: {len(jobs)} jobs in {R}; "
          f"code {head}, trainer {WANT_SHA['pretrain_range.py']}; "
          f"{'checked' if '--check' in sys.argv else 'test-only' if '--test-only' in sys.argv else 'submitted' if '--submit' in sys.argv else 'not submitted'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
