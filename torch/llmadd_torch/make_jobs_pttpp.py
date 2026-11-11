"""Torch jobs (2026-10-26, user-approved; intent spec and PRE-REGISTERED rules: runs/pttpp/AUDIT.md): the prior's gain at a
FIXED number of training tokens per non-embedding parameter (tpp). F87's 7% ladder trains every size on 3e8 tokens, so the
larger sizes get fewer tokens per parameter (53M 21 / 190M 2.65 / 505M 0.745 / 1.07B 0.318): "the gain grows with size" and
"the gain grows as training gets shorter" are confounded. Two tpp lines, equal to 505M@3e8 (0.745) and 1.07B@3e8 (0.318):
  7% (F87's ptlr design: 4 arithmetic lines per row, helix3 window (0.05, 0.4) prior on the operands = win-ctx, lambda 5)
    0.745: 93M, 190M, 294M (L40S), 1.07B (B200, 7.0e8 tokens);  reused: 505M@3e8 = F87 ptlr 500m
    0.318: 93M, 190M, 294M, 505M (L40S);                        reused: 1.07B@3e8 = F87 ptlr_1b
    bridge: 505M@3e8 on B200 at F87's picks (none@9.6e-3, win-ctx@1.2e-3), seeds 0-3 (is L40S -> B200 comparable?)
  2% (F86's pt2main design: 1 arithmetic line per row, the same prior)
    0.745: 190M, 294M (A100);                                   reused: 505M@3e8 = pt2s505
New sizes (the ladder's shape rule: head dim 64, d_ff = 4d, width and depth both strictly between the neighbours, N_nonemb
closest to the neighbours' geometric midpoint): 93M = d512 x 10 (8 heads), 294M = d832 x 19 (13 heads).
Tokens = whole steps of 32 x 512: S = round(18310 x N_nonemb / N_nonemb(reference size)), --tokens S x 16384; evals keep
~75 per run: --eval_every round(250 x S / 18310); the window (0.05, 0.4), the 1% warmup and the cosine are fractions of S
in the trainer (unchanged code). Per (setting, size, line) = one group: stage A = 2 lrs per arm (the pre-registered
interpolation, GRID below) x the selection seeds (0-3; 1.07B 0-1); the selection (runs/pttpp/maybe_select.sh, at the end of
every job) picks each arm's lr by F87 rule 1 (7%) / pt2main rule 1 (2%); an OPEN bracket adds one lr beyond the edge
(stage X, once per arm); still open after it -> STOP (the user decides); closed -> stage C = the confirm seeds at the pick
(4-7; 1.07B 2-7). Stage X / C cells come from runs/pttpp/decisions/<group>.json (written by pick_tpp.py), appended.
Trainers (unchanged): 7% = F87's snapshot (llmadd-burst 64abc19, pretrain_range.py a3bce3ab63c1; runs/pttpp/code7 asserted
= runs/ptlr/code file by file); 2% = pt2s505's snapshot (7f1d06b, trainer 93e22bf4c4c0; runs/pttpp/code2 = runs/pt2s505/code).
Reused read-only generators (sha256[:12] asserted; importing them runs no main(), asserted by ast): make_jobs_ptlr1b.py
(flags(), ARM, WANT), make_jobs_pt2s505.py (its intent() / burst_dry_expected() are the known answers of intent2() / dry2()
here, asserted), make_jobs_rb6scale.py (WANT_SHA, DATA_SHA, accounts, BURST). This module also has
make_pt2main_tasks.py's interface for the 2% groups (SEEDS, BLOCKS, LM0, DRY, tasks()), so
    python tools/pt2main_summary.py --gen /scratch/yh6210/llmadd_torch/make_jobs_pttpp.py --res runs/pttpp/out2
reads them with pt2main's own pre-registered rules (the 2% selection).
    python make_jobs_pttpp.py --plan                                   design table, cost, disk
    python make_jobs_pttpp.py [--group G | --all] [--print] [--test-only] [--submit]
(--all = every group's current stages; a submitted job file is never rewritten; SUBMITTED_<run> flags)"""
import ast
import filecmp
import hashlib
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
FROZEN = {"make_jobs_ptlr1b.py": "8dd7b3407459", "make_jobs_pt2s505.py": "a5a7fb60eedc", "make_jobs_rb6scale.py": "19bc055c2d09",
          "pt2s505_refs.json": "479e394771e9", "rb6scale_refs.json": "b65472eb3d29"}
for _f, _v in FROZEN.items():
    assert sha(HERE / _f) == _v, (_f, sha(HERE / _f), _v, "a read-only generator / reference file changed")
for _f, _ok in (("make_jobs_ptlr1b.py", ()), ("make_jobs_pt2s505.py", ("sys.path.insert(0, str(Path(__file__).resolve().parent))",
                "BURST_DRY.update({s: f'3{78 + s}-pt2s190-lr4.8e-3-none-s{s}' for s in CONFIRM})")), ("make_jobs_rb6scale.py", ())):
    _t = ast.parse((HERE / _f).read_text())
    _calls = [ast.unparse(n) for n in _t.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    _g = [n for n in _t.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
    assert tuple(_calls) == _ok and len(_g) == 1 and [ast.unparse(x) for x in _g[0].body] == ["main()"], (_f, _calls)
sys.path.insert(0, str(HERE))
import make_jobs_ptlr1b as M1   # noqa: E402  (F87 / ptlr_1b: flags(), ARM, WANT, TOK, PY, the 64abc19 pin)
import make_jobs_pt2s505 as P2  # noqa: E402  (pt2s505: the 2% job body; its intent() / burst_dry_expected() = known answers)
import make_jobs_rb6scale as J  # noqa: E402  (WANT_SHA of the 2% trainer, DATA_SHA, the A100 account)

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pttpp"
REPO = Path("/scratch/yh6210/llmadd-burst")   # holds 64abc19 and 7f1d06b (git archive of tools/)
TOK, PY, DATA = M1.TOK, M1.PY, J.DATA
assert TOK == J.TOK and PY == J.PY and DATA == B / "pretrain_data"
PIN7, PIN7_SHA, F87_CODE = M1.PIN, M1.PIN_SHA, B / "runs" / "ptlr" / "code"           # 64abc19 / a3bce3ab63c1
PIN2, PIN2_CODE = "7f1d06b", B / "runs" / "pt2s505" / "code"                           # 2% trainer 93e22bf4c4c0
WANT_SHA2 = J.WANT_SHA
assert PIN7_SHA == "a3bce3ab63c1" and WANT_SHA2["pretrain_range.py"] == "93e22bf4c4c0"
V = 100278                                    # tied embedding rows: N_nonemb = n_params - V * d_model
STEPS_3E8, TOK_STEP, EVAL_3E8 = 18310, 32 * 512, 250
REFS = B / "pttpp_refs.json"                  # CPU / GPU step-0 lm references and the 2% DRY row hashes (tests, AUDIT.md)

# ---------------------------------------------------------------- design (pre-registered: runs/pttpp/AUDIT.md "Design")
# size: (d_model, n_layers, n_heads, d_ff, n_params, N_nonemb); n_params as the trainer counts them (tied embedding once),
# verified on CPU (meta-device count and the --dry_inputs model line)
SIZES = {"53m": (384, 6, 6, 1536, 52672128, 14165376), "93m": (512, 10, 8, 2048, 93306368, 41964032),
         "190m": (768, 12, 12, 3072, 190297344, 113283840), "294m": (832, 19, 13, 3328, 293931456, 210500160),
         "505m": (1024, 24, 16, 4096, 505437184, 402752512), "1b": (1280, 36, 20, 5120, 1072259840, 943904000)}
for _k, (_d, _l, _h, _ff, _n, _nn) in SIZES.items():
    assert _d == 64 * _h and _ff == 4 * _d and _n - V * _d == _nn, _k
ORDER = list(SIZES)
for _a, _b in zip(ORDER, ORDER[1:]):   # width and depth grow together along the ladder
    assert SIZES[_a][0] < SIZES[_b][0] and SIZES[_a][1] < SIZES[_b][1], (_a, _b)
LINES = {"0.745": "505m", "0.318": "1b"}   # tpp line -> the size whose 3e8-token run defines it
MODEL_LINE = {k: f"model: {v[4] / 1e6:.1f}M parameters; gradient checkpointing {'on' if v[4] > 150e6 else 'off'}"
              for k, v in SIZES.items()}
LADDER = ("1.5e-4", "3e-4", "6e-4", "1.2e-3", "2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2", "3.84e-2", "7.68e-2")
# best lrs at 3e8 tokens (F87 / F86 and addenda): 7% none 53M 8e-3, 190M 4.8e-3, 505M 4.8e-3 = 9.6e-3 on 8 seeds (geometric
# mean used), 1.07B 2.4e-3; 7% win-ctx 53M 2e-3, 190M 1.2e-3, 505M 1.2e-3, 1.07B 6e-4; 2% none 53M 8e-3, 190M 4.8e-3, 505M
# 4.8e-3; 2% win-ctx 53M 2e-3, 190M 6e-4, 505M 6e-4
OPT3E8 = {("7", "none"): {"53m": 8e-3, "190m": 4.8e-3, "505m": math.sqrt(4.8e-3 * 9.6e-3), "1b": 2.4e-3},
          ("7", "win-ctx"): {"53m": 2e-3, "190m": 1.2e-3, "505m": 1.2e-3, "1b": 6e-4},
          ("2", "none"): {"53m": 8e-3, "190m": 4.8e-3, "505m": 4.8e-3},
          ("2", "win-ctx"): {"53m": 2e-3, "190m": 6e-4, "505m": 6e-4}}
ALPHA = 0.5   # lr_pred(N, T) = lr*(N, 3e8) x (3e8 / T)^ALPHA (AUDIT.md: the geometric middle of no token dependence and
              # pt2long's none, 4x tokens -> lr / 4; pt2long's prior: lr / 2 = ALPHA 0.5)
ARMS = ("none", "win-ctx")
ARM7 = {a: M1.ARM[a] for a in ARMS}
WANT7 = {a: M1.WANT[a] for a in ARMS}
# groups in submission order: id -> setting, size, line, hardware, selection seeds, confirm seeds, the pre-registered grid
GROUPS = {}
for _set, _hw, _cells in (("7", "l40s", (("93m", "0.745"), ("190m", "0.745"), ("294m", "0.745"), ("93m", "0.318"),
                                         ("190m", "0.318"), ("294m", "0.318"), ("505m", "0.318"))),
                          ("7", "b200", (("1b", "0.745"),)), ("2", "a100", (("190m", "0.745"), ("294m", "0.745")))):
    for _sz, _ln in _cells:
        GROUPS[f"tpp{_set}-{_sz}-t{_ln}"] = dict(setting=_set, size=_sz, line=_ln, hw=_hw,
                                                 sel=(0, 1) if _sz == "1b" else (0, 1, 2, 3),
                                                 conf=(2, 3, 4, 5, 6, 7) if _sz == "1b" else (4, 5, 6, 7))
GRID = {"tpp7-93m-t0.745": {"none": ("9.6e-3", "1.92e-2"), "win-ctx": ("2.4e-3", "4.8e-3")},
        "tpp7-190m-t0.745": {"none": ("4.8e-3", "9.6e-3"), "win-ctx": ("1.2e-3", "2.4e-3")},
        "tpp7-294m-t0.745": {"none": ("4.8e-3", "9.6e-3"), "win-ctx": ("1.2e-3", "2.4e-3")},
        "tpp7-93m-t0.318": {"none": ("1.92e-2", "3.84e-2"), "win-ctx": ("4.8e-3", "9.6e-3")},
        "tpp7-190m-t0.318": {"none": ("9.6e-3", "1.92e-2"), "win-ctx": ("2.4e-3", "4.8e-3")},
        "tpp7-294m-t0.318": {"none": ("9.6e-3", "1.92e-2"), "win-ctx": ("2.4e-3", "4.8e-3")},
        "tpp7-505m-t0.318": {"none": ("9.6e-3", "1.92e-2"), "win-ctx": ("1.2e-3", "2.4e-3")},
        "tpp7-1b-t0.745": {"none": ("1.2e-3", "2.4e-3"), "win-ctx": ("3e-4", "6e-4")},
        "tpp2-190m-t0.745": {"none": ("4.8e-3", "9.6e-3"), "win-ctx": ("6e-4", "1.2e-3")},
        "tpp2-294m-t0.745": {"none": ("4.8e-3", "9.6e-3"), "win-ctx": ("6e-4", "1.2e-3")}}
assert list(GRID) == list(GROUPS)
BRIDGE = "tppb-505m-3e8"   # 505M at 3e8 tokens on B200, F87's picks, seeds 0-3, no selection (AUDIT.md "Hardware bridge")
BRIDGE_CELLS = (("none", "9.6e-3"), ("win-ctx", "1.2e-3"))
# reused cells (AUDIT.md "Reused cells"): (setting, size, tpp label) -> {arm: (source dir, tag, pick lr)}, 8 seeds each
REUSED = {("7", "505m", "0.745"): {"none": ("ptlr/out", "500m-lr9.6e-3-7pct-none", "9.6e-3"),
                                    "win-ctx": ("ptlr/out", "500m-lr1.2e-3-7pct-win-ctx", "1.2e-3")},
          ("7", "1b", "0.318"): {"none": ("ptlr_1b/out", "1b-lr2.4e-3-7pct-none", "2.4e-3"),
                                  "win-ctx": ("ptlr_1b/out", "1b-lr6e-4-7pct-win-ctx", "6e-4")},
          ("7", "190m", "2.65"): {"none": ("ptlr/out", "190m-lr4.8e-3-7pct-none", "4.8e-3"),
                                   "win-ctx": ("ptlr/out", "190m-lr1.2e-3-7pct-win-ctx", "1.2e-3")},
          ("7", "53m", "21.2"): {"none": ("ptlr/out", "53m-lr8e-3-7pct-none", "8e-3"),
                                  "win-ctx": ("ptlr/out", "53m-lr2e-3-7pct-win-ctx", "2e-3")},
          ("2", "505m", "0.745"): {"none": ("pt2s505/out", "pt2s505-lr4.8e-3-none", "4.8e-3"),
                                    "win-ctx": ("pt2s505/out", "pt2s505-lr6e-4-win-ctx", "6e-4")}}
# per-step / per-eval seconds (training loop; measured on the F87 / ptlr_1b / pt2s505 run JSONs: 53M L40S 0.180 s / step,
# 190M 0.482, 505M 1.176; 1.07B B200 0.528 incl. its ckpt writes; 505M A100 0.798; the others log-log interpolated in
# n_params at the same gradient-checkpointing state, AUDIT.md "Cost"), + 180 s start-up per run
SEC = {("l40s", "93m"): (0.245, 3.5), ("l40s", "190m"): (0.482, 6.9), ("l40s", "294m"): (0.717, 10.5),
       ("l40s", "505m"): (1.176, 18.5), ("b200", "505m"): (0.266, 4.0), ("b200", "1b"): (0.528, 7.6),
       ("a100", "190m"): (0.327, 4.7), ("a100", "294m"): (0.487, 7.1)}
START_S = 180


def lr_fmt(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def steps_of(size, line):
    """whole steps: round(18310 x N_nonemb / N_nonemb(line's reference size)), half up, exact rational arithmetic"""
    fr = Fraction(STEPS_3E8 * SIZES[size][5], SIZES[LINES[line]][5])
    return math.floor(fr + Fraction(1, 2))


def eval_every_of(S):
    return max(1, math.floor(Fraction(EVAL_3E8 * S, STEPS_3E8) + Fraction(1, 2)))


def n_evals(S, E):
    return len(range(0, S, E)) + ((S - 1) % E != 0)


def window_events(S, window="0.05,0.4"):
    """the trainer's rule (pretrain_range.prior_on: w0 * steps <= s < w1 * steps, floats) -> events, on fraction"""
    w0, w1 = (float(v) for v in window.split(","))
    on = [s for s in range(S) if w0 * S <= s < w1 * S]
    return [[on[0], 1], [on[-1] + 1, 0]], len(on) / S


def lr_pred(setting, arm, size, S):
    """the pre-registered interpolation: log-linear in ln N_nonemb between the 3e8 optima of the neighbouring sizes, times
    (18310 / S)^ALPHA"""
    tab = OPT3E8[(setting, arm)]
    ks = sorted(tab, key=lambda k: SIZES[k][5])
    x = math.log(SIZES[size][5])
    if size in tab:
        l3 = tab[size]
    else:
        lo = max((k for k in ks if SIZES[k][5] < SIZES[size][5]), key=lambda k: SIZES[k][5])
        hi = min((k for k in ks if SIZES[k][5] > SIZES[size][5]), key=lambda k: SIZES[k][5])
        f = (x - math.log(SIZES[lo][5])) / (math.log(SIZES[hi][5]) - math.log(SIZES[lo][5]))
        l3 = math.exp(math.log(tab[lo]) + f * (math.log(tab[hi]) - math.log(tab[lo])))
    return l3 * (STEPS_3E8 / S) ** ALPHA


def grid_rule(setting, arm, size, S):
    """the two ladder lrs that straddle lr_pred: L[k] <= pred < L[k+1]"""
    p = lr_pred(setting, arm, size, S)
    k = max(i for i, v in enumerate(LADDER) if float(v) <= p)
    return LADDER[k], LADDER[k + 1]


for _g, _c in GROUPS.items():   # the hard-coded grids ARE the rule's output (pre-registered numbers, asserted)
    _S = steps_of(_c["size"], _c["line"])
    for _a in ARMS:
        assert GRID[_g][_a] == grid_rule(_c["setting"], _a, _c["size"], _S), (_g, _a, GRID[_g][_a], grid_rule(_c["setting"], _a, _c["size"], _S))


def step_out(lr, edge):
    i = LADDER.index(lr)
    return LADDER[i + 1] if edge == "upper" else LADDER[i - 1]


def tag_of(gid, arm, lr):
    return f"{gid}-lr{lr}-{arm}"


# ---------------------------------------------------------------- references, snapshots, decisions
def refs():
    return json.loads(REFS.read_text()) if REFS.exists() else {"lm0": {}, "dry2": {}}


def lm0_ref(setting, size, seed):
    v = refs()["lm0"].get(f"{setting}|{size}", {}).get(str(seed))
    return None if v is None else float(v)


git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def _extract(dst, commit, cmp_dir, commit_text):
    if not dst.exists():   # once; temp dir + rename, so a failed extraction leaves no snapshot
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=git("archive", commit, "tools"), check=True)
        (tmp / "COMMIT").write_text(commit_text)
        tmp.rename(dst)
    c = filecmp.dircmp(dst / "tools", cmp_dir / "tools", ignore=["__pycache__"])
    assert not (c.left_only or c.right_only or c.diff_files or c.funny_files), (dst, c.left_only, c.right_only, c.diff_files)
    assert all(filecmp.cmp(dst / "tools" / f, cmp_dir / "tools" / f, shallow=False) for f in c.common_files), dst
    assert (dst / "COMMIT").read_text() == (cmp_dir / "COMMIT").read_text(), dst


def snapshots():
    """runs/pttpp/code7 = git archive of 64abc19 tools/ (= F87's runs/ptlr/code file by file); code2 = git archive of
    7f1d06b tools/ (= pt2s505's runs/pt2s505/code file by file); COMMIT files copied verbatim (asserted equal)"""
    R.mkdir(parents=True, exist_ok=True)
    _extract(R / "code7", PIN7, F87_CODE, (F87_CODE / "COMMIT").read_text())
    _extract(R / "code2", PIN2, PIN2_CODE, (PIN2_CODE / "COMMIT").read_text())
    assert sha(R / "code7/tools/pretrain_range.py") == PIN7_SHA
    for f, v in WANT_SHA2.items():
        assert sha(R / "code2/tools" / f) == v, f


def g2():
    """make_pt2main_tasks of the 2% snapshot (CHECK_PY, DRY_GREP, DATA, ARM, WANT, flags, hours, the 190M DRY rows)"""
    d = str(R / "code2" / "tools")
    if d not in sys.path:
        sys.path.insert(0, d)
    import make_pt2main_tasks as G2
    assert Path(G2.__file__).resolve().parent == (R / "code2" / "tools").resolve(), G2.__file__
    return G2


def decisions(gid):
    """pick_tpp.py's decisions for a group: {"log": [{"seq", "arm", "stage" X | C | STOP | REOPEN, "lr", ...}]}, append-only.
    Amendment T2: REOPEN (an amendment entry: "amend" names it) withdraws the arm's current C; a later C may follow. At most
    one X per arm, at most one C per arm between REOPENs, nothing after a STOP; the seqs are 0, 1, 2, ... in order.
    Amendment T7 (the user's approval, 2026-10-27): a REOPEN (amend) may also withdraw a STOP; then one more X may follow
    (runs_of: one ladder step beyond the arm's last X, in the same direction); nothing else after a STOP."""
    f = R / "decisions" / f"{gid}.json"
    d = json.loads(f.read_text()) if f.exists() else {"log": []}
    assert [e["seq"] for e in d["log"]] == list(range(len(d["log"]))), (gid, "decision seqs")
    st = {a: {"X": None, "C": None, "STOP": False, "more": False} for a in ARMS}
    for e in d["log"]:
        a, k = e["arm"], e["stage"]
        assert k in ("X", "C", "STOP", "REOPEN") and a in ARMS and (not st[a]["STOP"] or (k == "REOPEN" and e.get("amend"))), e
        if k == "X":
            assert (st[a]["X"] is None or st[a]["more"]) and st[a]["C"] is None and e["lr"] in LADDER, e
            st[a]["X"], st[a]["more"] = e["lr"], False
        elif k == "C":
            assert st[a]["C"] is None and e["lr"] in LADDER, e
            st[a]["C"] = e["lr"]
        elif k == "REOPEN":
            assert (st[a]["C"] is not None or st[a]["STOP"]) and e.get("amend"), e
            if st[a]["STOP"]:   # T7: a STOP withdrawn by the user's approval: one more X may follow
                assert st[a]["X"] is not None, e
                st[a]["STOP"], st[a]["more"] = False, True
            st[a]["C"] = None
        else:
            st[a]["STOP"] = True
    return d


def effective(gid):
    """{arm: {"X": lr or None, "C": lr or None (the current one), "STOP": bool}} folded over the decision log"""
    st = {a: {"X": None, "C": None, "STOP": False} for a in ARMS}
    for e in decisions(gid)["log"]:
        a, k = e["arm"], e["stage"]
        if k in ("X", "C"):
            st[a][k] = e["lr"]
        elif k == "REOPEN":   # T2: withdraws a C; T7: or a STOP
            st[a]["C"], st[a]["STOP"] = None, False
        else:
            st[a]["STOP"] = True
    return st


def runs_of(gid):
    """[(stage, arm, lr, seed)] of a group, stage A first, then the decided stages in decision order"""
    c = GROUPS[gid]
    out = [("A", a, lr, s) for a in ARMS for lr in GRID[gid][a] for s in c["sel"]]
    lastx = {}
    for e in sorted(decisions(gid)["log"], key=lambda e: e["seq"]):
        if e["stage"] == "X":
            g_ = GRID[gid][e["arm"]]
            if e["arm"] not in lastx:
                assert e["lr"] not in g_ and e["lr"] in (step_out(g_[1], "upper"), step_out(g_[0], "lower")), e
            else:   # T7: the second extension continues the first one's direction by one ladder step
                prev = lastx[e["arm"]]
                assert e["lr"] == step_out(prev, "upper" if float(prev) > float(g_[1]) else "lower"), (e, prev)
            lastx[e["arm"]] = e["lr"]
            out += [("X", e["arm"], e["lr"], s) for s in c["sel"]]
        elif e["stage"] == "C":   # every C ever decided (a withdrawn C's jobs were submitted; Amendment T2): once each
            out += [r_ for r_ in (("C", e["arm"], e["lr"], s) for s in c["conf"]) if r_ not in out]
    return out


def bridge_runs():
    return [("A", a, lr, s) for a, lr in BRIDGE_CELLS for s in range(4)]


def b200_order():
    """every B200 run in a fixed order (slot = index mod SLOTS): tpp7-1b stage A, the bridge, then tpp7-1b's decided stages"""
    one = [("tpp7-1b-t0.745",) + r for r in runs_of("tpp7-1b-t0.745")]
    return one[:8] + [(BRIDGE,) + r for r in bridge_runs()] + one[8:]


SLOTS = 8   # Amendment T8 (the user, 2026-10-27: more GPUs, the same hardware): 4 -> 8 singleton slots; index < 4 keeps its slot,
            # the job text is otherwise unchanged (its comment 'at most 4 B200 runs' predates T8 and is kept verbatim)


# ---------------------------------------------------------------- per run facts
def facts(gid, arm, lr, seed, stage):
    """everything a job is rendered from (asserted against the design)"""
    if gid == BRIDGE:
        size, setting, hw, S, E = "505m", "7", "b200", STEPS_3E8, EVAL_3E8
        line = None
    else:
        c = GROUPS[gid]
        size, setting, hw, line = c["size"], c["setting"], c["hw"], c["line"]
        S = steps_of(size, line)
        E = eval_every_of(S)
    d, nl, nh, dff, n_params, nne = SIZES[size]
    ev, on = window_events(S) if arm != "none" else ([], 0.0)
    t, e = SEC[(hw, size)]
    est_h = (S * t + n_evals(S, E) * e + START_S) / 3600
    tlim_h = {"b200": 9.0 if size == "1b" else 3.0}.get(hw, max(1.0, math.ceil(2.5 * est_h * 2) / 2))
    save = (stage == "C") if setting == "7" else True
    return dict(gid=gid, arm=arm, lr=lr, seed=seed, stage=stage, size=size, setting=setting, hw=hw, line=line, S=S, E=E,
                T=S * TOK_STEP, tokens_arg=3e8 if gid == BRIDGE else float(S * TOK_STEP), evals=n_evals(S, E), events=ev, on_frac=on, dims=(d, nl, nh, dff), n_params=n_params, nne=nne,
                tpp=S * TOK_STEP / nne, warm=max(1, S // 100), layers=[nl // 3, 2 * nl // 3], ckpt_on=n_params > 150e6,
                est_h=est_h, tlim=f"{int(tlim_h):02d}:{int(round(60 * (tlim_h % 1))):02d}:00", save=save,
                ckpt_every=500 if hw == "b200" else 0, lm0=lm0_ref(setting, size, seed),
                tag=tag_of(gid, arm, lr), name=f"{tag_of(gid, arm, lr)}-s{seed}")


def model_flags(dims):
    d, nl, nh, dff = dims
    return f"--d_model {d} --n_layers {nl} --n_heads {nh} --d_ff {dff}"


# ---------------------------------------------------------------- 7% (F87 trainer) and the bridge
def args7(F):
    """the trainer ARGS of a 7% / bridge run, every intended flag asserted (and nothing else)"""
    out = R / "out" / F["tag"]
    tok = "" if F["gid"] == BRIDGE else f" --tokens {F['T']} --eval_every {F['E']}"
    args = (f"--data {DATA} --tokenizer {TOK} --tokenization whole --out {out} {ARM7[F['arm']]} --arith_lines 4 "
            f"{model_flags(F['dims'])}{tok} --lr {F['lr']} --seed {F['seed']} --tag _{F['tag']}"
            + (" --save_model" if F["save"] else "") + (f" --ckpt_every {F['ckpt_every']}" if F["ckpt_every"] else ""))
    f = M1.flags(args)
    d, nl, nh, dff = F["dims"]
    want = {"--data": str(DATA), "--tokenizer": str(TOK), "--tokenization": "whole", "--out": str(out), "--arith_lines": "4",
            "--d_model": str(d), "--n_layers": str(nl), "--n_heads": str(nh), "--d_ff": str(dff), "--lr": F["lr"],
            "--seed": str(F["seed"]), "--tag": f"_{F['tag']}", "--save_model": True if F["save"] else None,
            "--ckpt_every": str(F["ckpt_every"]) if F["ckpt_every"] else None,
            "--tokens": None if F["gid"] == BRIDGE else str(F["T"]), "--eval_every": None if F["gid"] == BRIDGE else str(F["E"]),
            "--lam": None, "--geo_layers": None, "--bs": None, "--seq": None, "--n_eval": None, "--gate_thr": None,
            "--gate_start": None, "--gate_hyst": None, "--untie": None, "--stop_after": None, "--grad_ckpt": None,
            "--lm_chunk_tokens": None, "--dry_inputs": None, "--site": None if F["arm"] == "none" else "ctx", **WANT7[F["arm"]]}
    for k, v in want.items():
        assert f.get(k) == v, (F["name"], k, f.get(k), v)
    assert set(f) <= {k for k, v in want.items() if v is not None}, (F["name"], sorted(f))
    assert float(f["--lr"]) == float(F["lr"]) and F["lr"] in LADDER + ("8e-3", "2e-3")
    if F["gid"] != BRIDGE:
        assert int(f["--tokens"]) // TOK_STEP == F["S"] and int(f["--tokens"]) % TOK_STEP == 0   # as the trainer: int(tokens // 16384)
    return args


def intent7(F):
    """the run JSON this job must produce (CHECK7 compares every field; the known answer: F87's 505M runs, tests)"""
    d, nl, nh, dff = F["dims"]
    a = {"data": str(DATA), "tokenizer": str(TOK), "out": str(R / "out" / F["tag"]), "tokenization": "whole",
         "target": "none" if F["arm"] == "none" else "helix3", "site": "ctx", "n_prefix": 256, "text_min": 16,
         "ckpt_every": F["ckpt_every"], "stop_after": -1, "seed": F["seed"], "tag": f"_{F['tag']}", "d_model": d, "n_layers": nl,
         "n_heads": nh, "d_ff": dff, "seq": 512, "bs": 32, "tokens": F["tokens_arg"], "arith_lines": 4, "lr": float(F["lr"]),
         "lam": 5.0, "schedule": "whole" if F["arm"] == "none" else "window", "window": "0.05,0.4", "gate_thr": 0.5,
         "gate_start": 0.0, "gate_hyst": 0.0, "geo_layers": None, "eval_every": F["E"], "n_eval": 150, "save_model": F["save"],
         "dry_inputs": False, "grad_ckpt": "auto", "lm_chunk_tokens": -1, "untie": False}
    return {"code_sha": PIN7_SHA, "steps": F["S"], "n_params": F["n_params"], "n_nonemb": F["nne"], "grad_ckpt_on": F["ckpt_on"],
            "geo_layers": F["layers"], "read_layers": F["layers"], "lm_chunk": 0, "eval_every_steps": F["E"], "evals": F["evals"],
            "events": F["events"], "on_frac": F["on_frac"], "lm0_ref": F["lm0"], "V": V, "tpp": F["tpp"], "args": a}


CHECK7 = r'''import json, math, sys
from pathlib import Path
run, out_json = sys.argv[1:3]
I = json.loads(sys.argv[3])
checks = []
def chk(name, ok, value):
    checks.append({"check": name, "ok": bool(ok), "value": value})
r = json.loads(Path(run).read_text()); c = r["curve"]; a = r["args"]
diff = {k: [a.get(k), v] for k, v in I["args"].items() if a.get(k) != v}
extra = sorted(set(a) - set(I["args"]))
chk(f"run args = intent ({len(I['args'])} fields, no other)", not diff and not extra, [diff, extra])
chk("code_sha " + I["code_sha"], r.get("code_sha") == I["code_sha"], r.get("code_sha"))
keys = ("steps", "n_params", "grad_ckpt_on", "geo_layers", "read_layers", "lm_chunk", "eval_every_steps")
got = {k: r.get(k) for k in keys}
chk("run facts = intent " + json.dumps({k: I[k] for k in keys}), all(got[k] == I[k] for k in keys), got)
chk(f"N_nonemb = n_params - V x d_model = {I['n_nonemb']}", r.get("n_params", 0) - I["V"] * a["d_model"] == I["n_nonemb"], r.get("n_params"))
st = [e["step"] for e in c]
want_st = sorted(set(range(0, I["steps"], I["eval_every_steps"])) | {I["steps"] - 1})
chk(f"finished: {I['evals']} evals at the intended steps, the last at step {I['steps'] - 1}",
    st == want_st and len(st) == I["evals"], [len(c), st[-1] if st else None])
fin = lambda v: v is None or math.isfinite(v)
bad = [e["step"] for e in c if not (fin(e["lm"]) and fin(e["val_loss"]) and fin(e["geo"]) and fin(e.get("grad_prior")))]
if I["events"]:
    chk(f"prior_events == {I['events']}", r.get("prior_events") == I["events"], r.get("prior_events"))
    chk(f"prior_on_frac == {I['on_frac']:.6f} == prior_used_frac (ctx)", abs(c[-1]["prior_on_frac"] - I["on_frac"]) < 1e-9
        and abs(c[-1]["prior_used_frac"] - I["on_frac"]) < 1e-9, [c[-1]["prior_on_frac"], c[-1]["prior_used_frac"]])
    on = [e for e in c if e["prior_on"] and e["geo"] is not None and (not bad or e["step"] < bad[0])]
    chk("geo > 0 and 0 < grad_prior < inf at every eval where the prior acted (before a divergence)",
        (bool(on) or bool(bad)) and all(e["geo"] > 0 and 0 < e.get("grad_prior", 0) < math.inf for e in on), len(on))
else:
    chk("prior never on", c[-1]["prior_on_frac"] == 0 and c[-1].get("prior_used_frac", 0) == 0 and r.get("prior_events") == [],
        c[-1]["prior_on_frac"])
if I["lm0_ref"] is not None:
    chk("step-0 lm within 1e-3 of the same-seed reference (same init and first batch)", abs(c[0]["lm"] - I["lm0_ref"]) <= 1e-3,
        c[0]["lm"] - I["lm0_ref"])
B = ("b11", "b12", "b13", "b22", "b23", "b33")
held = lambda e: sum(e[f"held_{b}"] for b in B) / len(B)
tok = r["steps"] * a["bs"] * a["seq"]
info = {"tag": a["tag"], "seed": a["seed"], "lr": a["lr"], "tokens_consumed": tok, "steps": r["steps"],
        "eval_every_steps": r.get("eval_every_steps"), "evals": len(c), "warmup_steps": max(1, r["steps"] // 100),
        "tpp_consumed": tok / (r["n_params"] - I["V"] * a["d_model"]), "prior_events": r.get("prior_events"),
        "area": round(100 * sum(held(e) for e in c) / len(c), 2), "diverged_at_step": bad[0] if bad else None,
        "final_val_bpb": c[-1]["val_bits_per_byte"], "lm0": c[0]["lm"]}
ok = all(k["ok"] for k in checks)
Path(out_json).write_text(json.dumps({"pass": ok, "checks": checks, "info": info}, indent=1))
for k in checks:
    print(("PASS " if k["ok"] else "FAIL ") + k["check"] + f"  ({k['value']})")
print("CONSUMED", json.dumps(info))
print("CHECKS", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 4)
'''


def job7(F, slot=None):
    """a 7% / bridge job: F87's job body (L40S) or ptlr_1b's (B200) + the data sha, the intent line, CHECK7, the hook"""
    args = args7(F)
    I = intent7(F)
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    target = "none" if F["arm"] == "none" else "helix3"
    runjson = R / "out" / F["tag"] / f"run_whole_{target}_seed{F['seed']}_{F['tag']}.json"
    check = R / "out" / F["tag"] / f"check_s{F['seed']}.json"
    w = F["events"]
    hook = ("# the hardware bridge has no selection: no hook\n" if F["gid"] == BRIDGE else
            "# the pre-registered selection of this group (once its runs are complete; lock, decisions, sha checks inside)\n"
            f"bash {R}/maybe_select.sh {F['gid']} || echo \"maybe_select.sh failed (rc $?): selection not run\"\n")
    if F["hw"] == "b200":
        head = f"""#SBATCH --job-name=llmadd_tpp-b200-slot{slot}
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
#SBATCH --output={B}/logs/llmadd_tpp-{F['name']}_%j.out"""
        gpu = "B200"
    else:
        assert F["hw"] == "l40s" and slot is None
        head = f"""#SBATCH --job-name=llmadd_tpp-{F['name']}
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
    what = ("hardware bridge: 505M at 3e8 tokens on B200 at F87's pick" if F["gid"] == BRIDGE else
            f"{F['size']} at tpp {F['line']} line, 7% (F87 ptlr design), stage {F['stage']}")
    resume = ("Preemptible: a preempted job is requeued (same id) and the trainer resumes from its ckpt (every 500 steps); "
              "singleton slots keep at most 4 B200 runs at a time." if F["hw"] == "b200" else
              "No ckpt (--ckpt_every 0, short run): a requeued job restarts the run from step 0.")
    return f"""#!/bin/bash
{head}
# pttpp run {F['name']}: {what}; {F['arm']} lr {F['lr']} seed {F['seed']}. Intent spec and pre-registered rules
# runs/pttpp/AUDIT.md; generated by llmadd_torch/make_jobs_pttpp.py (change the generator, not this file); trainer = F87's
# snapshot (code7/COMMIT, pretrain_range.py {PIN7_SHA}). {resume}
set -eu
# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/pttpp/HOLD exists
if [ -e {R}/HOLD ]; then echo "HOLD present: exiting"; exit 0; fi
cd {R}/code7/tools
# the snapshot must still be the pinned trainer (code_sha of every result); exit before any GPU work otherwise
[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{PIN7_SHA}" ] || {{ echo "code_sha mismatch: exiting"; exit 3; }}
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
echo "=== {F['name']} start $(date -Is) job ${{SLURM_JOB_ID:-none}} account ${{SLURM_JOB_ACCOUNT:-?}} partition ${{SLURM_JOB_PARTITION:-?}} restart ${{SLURM_RESTART_COUNT:-0}} node $(hostname)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"; echo "$GPU"
# one GPU type per hardware arm (the design's): exit before any training on anything else
case "$GPU" in *{gpu}*) ;; *) echo "not a {gpu}: exiting"; exit 4;; esac
(cd {DATA} && sha256sum -c -) <<'SHA' || exit 3
{J.DATA_SHA[0]}  train.bin
{J.DATA_SHA[1]}  val.bin
SHA
echo "INTENT {F['name']}: {F['size']} N_nonemb {F['nne']}, tokens {F['T']} = {F['S']} steps x {TOK_STEP} (tpp {F['tpp']:.5f}), eval every {F['E']} steps -> {F['evals']} evals, warmup {F['warm']} steps, prior window {'steps ' + str(w[0][0]) + '..' + str(w[1][0] - 1) if w else 'none (no prior)'}, save_model {F['save']}"
ARGS="{args}"
{PY} pretrain_range.py $ARGS --dry_inputs
{PY} pretrain_range.py $ARGS
echo "=== {F['name']} trainer returned $(date -Is)"
# intent checks of the run JSON (every argument, steps, evals, window events, step-0 lm) -> {check.name}; the hook runs
# whatever the check says (a failed run withholds its cell in the read-out); the job then exits with the check's code
RC=0
{PY} - "{runjson}" "{check}" '{ij}' <<'PY' || RC=$?
{CHECK7}PY
{hook}exit $RC
"""


# ---------------------------------------------------------------- 2% (pt2s505's job body, pt2main's CHECK_PY)
def dry2(G2, size, seed):
    """the 9 DRY rows' sha256 expected at <size> from the Burst pt2s190 dry log of the seed: make_jobs_pt2s505's
    burst_dry_expected with the model line as a parameter (with 505M's line it equals P2.burst_dry_expected: tests)"""
    import re
    if size == "190m":
        return G2.DRY[("pretrain_data", "190m", "ctx", seed)]
    txt = (J.BURST / "results" / P2.BURST_DRY[seed] / "dry_inputs.txt").read_text().splitlines()
    g = re.compile(G2.DRY_GREP)
    sel = [l for l in txt if g.match(l)][:9]
    h190 = hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest()
    assert h190 == G2.DRY[("pretrain_data", "190m", "ctx", seed)], (P2.BURST_DRY[seed], "emulation != the recorded DRY")
    m190 = G2.SIZES["190m"]["model_line"]
    assert sum(l == m190 for l in sel) == 1, P2.BURST_DRY[seed]
    return hashlib.sha256(("\n".join(MODEL_LINE[size] if l == m190 else l for l in sel) + "\n").encode()).hexdigest()


def intent2(G2, F, code_sha=None):
    """CHECK_PY's intent of a 2% run (make_jobs_pt2s505.intent with the tokens / eval_every / ckpt_every / size facts as
    parameters; with 505M's facts at 3e8 it returns P2.intent's dict exactly: tests)"""
    d, nl, nh, dff = F["dims"]
    site = G2.WANT[F["arm"]]["--site"] or "ctx"
    return {"code_sha": code_sha or WANT_SHA2["pretrain_range.py"], "steps": F["S"], "n_params": F["n_params"],
            "grad_ckpt_on": F["ckpt_on"], "geo_layers": F["layers"], "read_layers": F["layers"], "lm_chunk": 0,
            "eval_every_steps": F["E"], "evals": F["evals"], "events": F["events"], "on_frac": F["on_frac"], "site": site,
            "lm0_ref": F["lm0"],
            "args": {"tokenization": "whole", "target": G2.WANT[F["arm"]]["--target"], "site": site,
                     "schedule": G2.WANT[F["arm"]]["--schedule"] or "whole", "window": "0.05,0.4", "arith_lines": 1,
                     "lr": float(F["lr"]), "lam": float(G2.WANT[F["arm"]].get("--lam") or 5.0), "tokens": float(F["T"]),
                     "bs": G2.BS, "seq": G2.SEQ, "d_model": d, "n_layers": nl, "n_heads": nh, "d_ff": dff, "seed": F["seed"],
                     "tag": f"_{F['tag']}", "save_model": True, "ckpt_every": F["ckpt_every"], "geo_layers": None,
                     "eval_every": F["E"], "n_eval": 150, "grad_ckpt": "auto", "lm_chunk_tokens": -1, "untie": False,
                     "gate_thr": 0.5, "gate_start": 0.0, "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256, "stop_after": -1,
                     "dry_inputs": False}}


def args2(G2, F):
    OUT = R / "out2" / F["name"]
    tok = "" if F["T"] == 3e8 else f" --tokens {F['T']} --eval_every {F['E']}"
    args = (f"--data {DATA} --tokenizer {TOK} --tokenization whole --out {OUT}/{F['tag']} {G2.ARM[F['arm']]} --arith_lines 1 "
            f"{model_flags(F['dims'])}{tok} --lr {F['lr']} --seed {F['seed']} --tag _{F['tag']} --save_model")
    f = G2.flags(args)
    d, nl, nh, dff = F["dims"]
    want = {"--data": str(DATA), "--tokenizer": str(TOK), "--tokenization": "whole", "--out": f"{OUT}/{F['tag']}",
            "--arith_lines": "1", "--lr": F["lr"], "--seed": str(F["seed"]), "--tag": f"_{F['tag']}", "--save_model": True,
            "--ckpt_every": None, "--tokens": str(F["T"]), "--eval_every": str(F["E"]), "--lam": None, "--geo_layers": None,
            "--bs": None, "--seq": None, "--n_eval": None, "--gate_thr": None, "--gate_start": None, "--gate_hyst": None,
            "--untie": None, "--grad_ckpt": None, "--lm_chunk_tokens": None, "--stop_after": None, "--dry_inputs": None,
            **G2.WANT[F["arm"]], "--d_model": str(d), "--n_layers": str(nl), "--n_heads": str(nh), "--d_ff": str(dff)}
    for k, v in want.items():
        assert f.get(k) == v, (F["name"], k, f.get(k), v)
    assert set(f) <= {k for k, v in want.items() if v is not None}, (F["name"], sorted(f))
    assert int(f["--tokens"]) // TOK_STEP == F["S"] and int(f["--tokens"]) % TOK_STEP == 0
    return OUT, args


def job2(G2, F):
    """a 2% job: make_jobs_pt2s505.render's body (HOLD, snapshot shas, A100, data sha256, --dry_inputs checks + the DRY
    row hash, training, offline_lm.py, CHECK_PY verbatim) with this run's facts; + the hook; stage A / X (seeds 0-3): the
    model is deleted after a passing check (offline_lm has used it; AUDIT.md "Disk")"""
    OUT, args = args2(G2, F)
    I = intent2(G2, F)
    assert I["lm0_ref"] is not None, f"{F['name']}: no step-0 lm reference; run the CPU references first"
    dry = dry2(G2, F["size"], F["seed"])
    rd = refs()["dry2"].get(F["size"], {}).get(str(F["seed"]))
    assert rd == dry, (F["name"], "the Torch CPU dry run of this size and seed != the Burst pt2s190 rows + model line", rd, dry)
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    target = G2.WANT[F["arm"]]["--target"]
    runfile = f"run_whole_{target}_seed{F['seed']}_{F['tag']}.json"
    site_assert = ("""[ "$(grep -c '^train row [012] (len 512, 2 prior sites at ' "$OUT/dry_inputs.txt")" = 3 ] """
                   """|| { echo "## not 1 arithmetic line (2 operand sites) per row"; exit 3; }""")
    target_assert = ("""! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a prior target in a none run"; exit 3; }"""
                     if F["arm"] == "none" else
                     """grep -qxF 'target features (1000, 10)' "$OUT/dry_inputs.txt" || { echo "## not the helix3 table (1000 x 10: cos 2, cos/sin 5 10 100 1000, x / 1000)"; exit 3; }""")
    dep_chk = "\n".join(f'[ "$(sha256sum {fn} | cut -c1-12)" = "{v}" ] || {{ echo "## {fn} sha mismatch: exiting"; exit 3; }}'
                        for fn, v in WANT_SHA2.items())
    keep = F["seed"] >= 4
    drop = ("" if keep else
            f'# stage {F["stage"]} (seed {F["seed"]} <= 3): the model is not kept once offline_lm and the checks used it (AUDIT.md "Disk")\n'
            f'if [ "$RC" = 0 ]; then rm -f "$OUT/$TAG/model_whole_{target}_seed{F["seed"]}_{F["tag"]}.pt"; echo "## model removed (stage {F["stage"]})"; fi\n')
    w = F["events"]
    return OUT, f"""#!/bin/bash
#SBATCH --job-name=llmadd_tpp-{F['name']}
#SBATCH --account={J.ACCOUNT}
#SBATCH --partition={J.PARTITION}
#SBATCH --qos={J.QOS}
#SBATCH --gres=gpu:1
#SBATCH --nice={J.NICE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={F['tlim']}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# pttpp run {F['name']}: {F['size']} at the tpp {F['line']} line, 2% (F86's pt2main design, as pt2s505), stage {F['stage']};
# {F['arm']} lr {F['lr']} seed {F['seed']}. Intent spec and pre-registered rules runs/pttpp/AUDIT.md; generated by
# llmadd_torch/make_jobs_pttpp.py (change the generator). Body = make_jobs_pt2s505's: 1. HOLD, snapshot shas, A100, data
# sha256; --dry_inputs: model line, 2 operand sites per row, target table, the 9 rows' sha256 (exit 3 before training);
# 2. training (no ckpt: a requeued job restarts the run); 3. offline_lm.py (full val); 4. make_pt2main_tasks.CHECK_PY with
# this run's intent -> $OUT/check.json; 5. the group's selection hook; the job exits with the check's code
set -euo pipefail
R={R}; OUT={OUT}; TAG={F['tag']}; SEED={F['seed']}; DATA={DATA}; TOK={TOK}; PY={PY}
# kill switch (no scancel from the agent): a pending job exits at start when runs/pttpp/HOLD exists
if [ -e "$R/HOLD" ]; then echo "HOLD present: exiting"; exit 0; fi
cd "$R/code2/tools"
{dep_chk}
grep -q "pretrain_range.py {WANT_SHA2['pretrain_range.py']}" "$R/code2/COMMIT" || {{ echo "## COMMIT does not name the trainer {WANT_SHA2['pretrain_range.py']}"; exit 3; }}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU: $GPU"
case "$GPU" in *A100*) ;; *) echo "## not an A100 (the design's GPU; time limits are A100-based): exiting"; exit 3;; esac
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
echo "INTENT {F['name']}: {F['size']} N_nonemb {F['nne']}, tokens {F['T']} = {F['S']} steps x {TOK_STEP} (tpp {F['tpp']:.5f}), eval every {F['E']} steps -> {F['evals']} evals, warmup {F['warm']} steps, prior window {'steps ' + str(w[0][0]) + '..' + str(w[1][0] - 1) if w else 'none (no prior)'}"
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
"$PY" - "$OUT/$TAG/{runfile}" "$OUT/offline_lm.json" "$OUT/gpu.csv" "$R/code2/COMMIT" "$DATA" \\
  "$OUT/check.json" '{ij}' <<'PY' | tee "$OUT/check.txt" || RC=$?
{G2.CHECK_PY}PY
{drop}echo "## 5. selection hook $(date -u +%FT%TZ)"
bash "$R/maybe_select.sh" {F['gid']} || echo "maybe_select.sh failed (rc $?): selection not run"
echo "## done $(date -u +%FT%TZ) (check rc $RC)"
exit $RC
"""


# ---------------------------------------------------------------- make_pt2main_tasks' interface (the 2% groups)
SEEDS = range(4)
LM0, DRY = {}, {}
GROUPS2 = [g for g, c in GROUPS.items() if c["setting"] == "2"]
BLOCKS = []
for _i, _g in enumerate(GROUPS2):
    _c = GROUPS[_g]
    _S = steps_of(_c["size"], _c["line"])
    BLOCKS.append((_i, _g, _c["size"], "pretrain_data", str(_S * TOK_STEP), facts(_g, "none", GRID[_g]["none"][0], 0, "A")["est_h"],
                   facts(_g, "none", GRID[_g]["none"][0], 0, "A")["tlim"], [(a, lr) for a in ARMS for lr in GRID[_g][a]]))
    _l = [lm0_ref("2", _c["size"], s) for s in range(8)]
    if None not in _l:
        LM0[(_c["size"], "pretrain_data")] = tuple(_l)


def tasks():
    """[(name, tag, text, args, block, arm, lr, seed)] of every 2% run (stage A seed-major, then the decided stages)"""
    G2 = g2()
    out = []
    for gid in GROUPS2:
        rs = runs_of(gid)
        rs = sorted([r for r in rs if r[0] == "A"], key=lambda r: (r[3], ARMS.index(r[1]), float(r[2]))) + [r for r in rs if r[0] != "A"]
        for st, a, lr, s in rs:
            F = facts(gid, a, lr, s, st)
            _, txt = job2(G2, F)
            out.append((F["name"], F["tag"], txt, args2(G2, F)[1], gid, a, lr, s))
    return out


# ---------------------------------------------------------------- rendering / submission
def render(gid):
    """[(run name, job file text, facts)] of a group's current stages (B200 runs with their slot)"""
    out = []
    if GROUPS.get(gid, {}).get("hw") == "a100":
        G2 = g2()
        for st, a, lr, s in runs_of(gid):
            F = facts(gid, a, lr, s, st)
            out.append((F["name"], job2(G2, F)[1], F))
        return out
    if gid == BRIDGE or GROUPS[gid]["hw"] == "b200":
        for i, (g, st, a, lr, s) in enumerate(b200_order()):
            if g == gid:
                F = facts(g, a, lr, s, st)
                out.append((F["name"], job7(F, slot=i % SLOTS), F))
        return out
    for st, a, lr, s in runs_of(gid):
        F = facts(gid, a, lr, s, st)
        out.append((F["name"], job7(F), F))
    return out


def all_groups():
    return list(GROUPS)[:-3] + ["tpp7-1b-t0.745", BRIDGE] + GROUPS2


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def planned_hours():
    """GPU-h by bucket ('b200', 'other' = L40S + A100) of every run decided or still to come under the design: stage A,
    the decided X runs, every decided C, and C for each arm without a C yet (Amendment T2 cost guard)"""
    tot = {"b200": 0.0, "other": 0.0}
    for gid in GROUPS:
        st = effective(gid)
        rs = runs_of(gid)
        for a in ARMS:
            if st[a]["C"] is None and not st[a]["STOP"]:
                rs = rs + [("C", a, GRID[gid][a][0], s) for s in GROUPS[gid]["conf"]]
        for stg, a, lr, s in rs:
            F = facts(gid, a, lr, s, stg)
            tot["b200" if F["hw"] == "b200" else "other"] += F["est_h"]
    for stg, a, lr, s in bridge_runs():
        tot["b200"] += facts(BRIDGE, a, lr, s, stg)["est_h"]
    return tot


COST_MAX = {"b200": 180.0, "other": 310.0}   # Amendment T2 (coordinator 2026-10-26): stop before exceeding; T7 (the user, 2026-10-27): other 300 -> 310


def plan():
    print(f"{'group':20s} {'size':5s} {'N_nonemb':>12s} {'steps':>6s} {'tokens':>11s} {'tpp':>8s} {'E':>4s} {'evals':>5s} "
          f"{'warm':>4s} {'window':>13s} {'grid none':>18s} {'grid win-ctx':>16s} {'h/run':>6s} {'limit':>8s}")
    tot = {}
    for gid in all_groups():
        rs = bridge_runs() if gid == BRIDGE else runs_of(gid)
        F = facts(gid, rs[0][1], rs[0][2], rs[0][3], "A")
        Fw = facts(gid, "win-ctx", (dict(BRIDGE_CELLS)["win-ctx"] if gid == BRIDGE else GRID[gid]["win-ctx"][0]), 0, "A")
        g = ({"none": ("9.6e-3",), "win-ctx": ("1.2e-3",)} if gid == BRIDGE else GRID[gid])
        n_sel = len(rs)
        n_conf = 0 if gid == BRIDGE else 2 * len(GROUPS[gid]["conf"])
        tot[F["hw"]] = tot.get(F["hw"], 0) + (n_sel + n_conf) * F["est_h"]
        pn = "" if gid == BRIDGE else f" (pred {lr_pred(GROUPS[gid]['setting'], 'none', F['size'], F['S']):.2e})"
        pw = "" if gid == BRIDGE else f" (pred {lr_pred(GROUPS[gid]['setting'], 'win-ctx', F['size'], F['S']):.2e})"
        print(f"{gid:20s} {F['size']:5s} {F['nne']:12,d} {F['S']:6d} {F['T']:11,d} {F['tpp']:8.5f} {F['E']:4d} {F['evals']:5d} "
              f"{F['warm']:4d} {str(Fw['events'][0][0]) + '-' + str(Fw['events'][1][0]):>13s} {'/'.join(g['none']) + pn:>18s} "
              f"{'/'.join(g['win-ctx']) + pw:>16s} {F['est_h']:6.2f} {F['tlim']:>8s}   runs A {n_sel} + C {n_conf}")
    print("planned GPU-h (stage A + C, no stage X):", {k: round(v, 1) for k, v in tot.items()})


def main():
    snapshots()
    if "--plan" in sys.argv:
        plan()
        return
    au = R / "AUDIT.md"
    assert au.exists() and "## Pre-registered rules" in au.read_text(), "runs/pttpp/AUDIT.md (the pre-registration) must exist first"
    groups = all_groups() if "--all" in sys.argv else [opt("--group")]
    assert all(g in GROUPS or g == BRIDGE for g in groups), groups
    existing = {p.name for p in B.glob("runs/*/out*/*") if p.parent.parent != R}   # other batches' run dirs
    bad, jobs = 0, []
    for gid in groups:
        for name, txt, F in render(gid):
            assert name not in existing and F["tag"] not in existing, name   # never another batch's dir name
            job, flag = R / f"job_{name}.sbatch", R / f"SUBMITTED_{name}"
            if flag.exists():
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
        if "--test-only" in sys.argv and not flag.exists():   # submitted jobs are not tested again
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
    print(f"pttpp {', '.join(groups)}: {len(jobs)} jobs rendered in {R}; "
          f"{'test-only' if '--test-only' in sys.argv else 'submitted' if '--submit' in sys.argv else 'not submitted'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
