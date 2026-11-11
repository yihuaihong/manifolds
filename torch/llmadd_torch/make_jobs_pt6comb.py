"""Torch jobs of runs/pt6comb: six manifold priors trained together (pre-registered in runs/pt6comb/AUDIT.md; the user's goal
'six manifolds added together, each helps, none interferes'; design by the LLM_addition coordinator 2026-11-10; composition B
decided by the coordinator / user 2026-11-10).

Composition (--comp; one batch = one --domains, identical data in every arm of a seed, B200 only):
  B (decided, submittable)  numbers (helix3 on the operands) + s4 + dim (the Rev B rainbow domains, F88) + cities + countries +
                            people (the ptdom Rev E / F comparison domains, F100 / F101)  -> --domains numbers,s4,dim,cities,countries,people
  A (kept as a parameter only; NOT submittable)  numbers + months + cities + countries + people + X (--x, a ptdom domain)
Data per row = FineWeb-Edu text + one numbers line (2%, always first after the text) + K = 2 lines of the five other domains
(the trainer's cycling rule over its fixed domain order), 53M, 3e8 tokens = 18,310 steps of 32 x 512, 75 evals; ptdom's
stage-1 job body (no saved model, no offline LM).
Arms: none; all6 (the six priors at once: --target helix3 + the five domain priors, lambda 1 each, lam_rule each, one window
(0.05, 0.4) = steps 916..7323, site ctx); only-<m> for each of the six manifolds m (only-numbers = --target helix3 alone).
Parts (AUDIT.md "Design"):
  L1    lr selection, seeds 0-3: none @ 2e-3 / 4e-3 / 8e-3, all6 @ 1e-3 / 2e-3 / 4e-3                         24 runs
  L1x   the conditional extensions decided by select_pt6comb.py --after L1 (ptdom2 A1: all6 x2 above where its pick or
        area argmax is 4e-3; none x2 above / x1/2 below where its pick or argmax is at that edge; fixB: none at all6's pick
        when it lies outside none's range), once, seeds 0-3                                                  0-12 runs
  C     confirmation, fresh seeds 8-15 (never read by a selection): none@N, all6@P, only-<m>@2e-3 for the six m, and (AUDIT A2)
        the same-lr pair none@2e-3 / all6@2e-3 (merged with none@N / all6@P when N / P = 2e-3)                  64-80 runs
  smoke ONE B200 job (llmadd_pt6comb-smoke, <= 1 h; not part of any result): all6@2e-3 and none@2e-3, seed 0, --stop_after
        8000 (the whole prior window and three evals after it), then smoke/smoke_check.py (AUDIT.md "GPU smoke")
Every job is rendered by make_jobs_ptdom.render (the frozen ptdom generator, imported read-only, sha256[:12] asserted; ptdom2's
technique: a handful of module names re-pointed for one call), then exact, counted text edits; CHECK_PY_6 = ptdom's CHECK_PY
with exact, counted edits for the numbers prior and the s4 / dim read-outs (tests/render_diff.py prints the full diff).
Run / out-dir name: pt6comb-<comp>-<arm>-lr<lr>-s<seed>[-r<k>]; Slurm: llmadd_pt6comb-b200-slot<0..7> (singleton chains, ptdom's /
pt300x's B200 protocol); logs llmadd_torch/logs/llmadd_<run>_<jobid>.out.
    python make_jobs_pt6comb.py --plan
    python make_jobs_pt6comb.py --snapshot
    python make_jobs_pt6comb.py --part L1 [--comp B] [--check] [--print] [--test-only N] [--submit] [--no-write]
    python make_jobs_pt6comb.py --part L1x|C --picks runs/pt6comb/picks_after_<L1|L1x>.json [--check] [--print] [--test-only N] [--submit]
    python make_jobs_pt6comb.py --smoke [--check] [--print] [--test-only] [--submit]
    python make_jobs_pt6comb.py --part X [--picks f] --rerun [--print] [--submit]      (ptdom's A2 re-run path: <name>-r<k>)
Never submits without --submit; --submit refuses while runs/pt6comb/HOLD or HOLD_<part> exists, for composition A, and never
re-submits a name.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pt6comb"
R0 = B / "runs" / "ptdom"                      # the frozen ptdom batch: read only
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
MJ_FILE, MJ_SHA = B / "make_jobs_ptdom.py", "6061b299129c"
M2_FILE, M2_SHA = B / "make_jobs_ptdom2.py", "62d680144854"
assert sha(MJ_FILE) == MJ_SHA, f"make_jobs_ptdom.py changed: {sha(MJ_FILE)} != {MJ_SHA}"
assert sha(M2_FILE) == M2_SHA, f"make_jobs_ptdom2.py changed: {sha(M2_FILE)} != {M2_SHA}"
sys.path.insert(0, str(B))
import make_jobs_ptdom as MJ    # noqa: E402  (read only: render, CHECK_PY, constants)
import make_jobs_ptdom2 as M2   # noqa: E402  (read only: _edit, the ptdom2 technique)
assert Path(MJ.__file__).resolve() == MJ_FILE.resolve() and Path(M2.__file__).resolve() == M2_FILE.resolve()

COMMIT = "11f61b9"                              # the ptdom trainer (Rev G); runs/pt6comb/code = git archive of its tools/
WANT_SHA, PY, LADDER, HW = MJ.WANT_SHA, MJ.PY, MJ.LADDER, MJ.HW
assert WANT_SHA["pretrain_range.py"] == "d87a1502cee8"
PREFIX = "pt6comb"
_edit = M2._edit                                # exact, counted replacement (asserts the count)

# ---------------------------------------------------------------- design (AUDIT.md "Design")
# the trainer's fixed domain order (pretrain_range.py EXTRA + ANCHOR + NEWDOM: the RNG children are indexed by it; EXT is built
# in this order whatever the order of --domains)
TRAINER_ORDER = (("months", "weekdays", "clock", "s4", "dim") + ("stars", "landmarks", "languages", "colours", "cities")
                 + MJ.BATCHES["B1"] + MJ.BATCHES["B2"] + MJ.BATCHES["B3"])
TOY = ("s4", "dim")                             # the Rev B toy domains (rb6 / F88): exact-match read-outs, nonce single-token items
COMP_B = ("s4", "dim", "cities", "countries", "people")
COMP_A_FIXED = ("months", "cities", "countries", "people")
X_ALLOWED = ("landmarks", "colours", "stars", "compass", "prefixes", "ranks", "scalar_adj")   # composition A's X (not submittable)


def comp_doms(comp, x=None):
    """The five non-number domains of a composition, in the trainer's order."""
    if comp == "B":
        assert x is None, "composition B has no X"
        doms = COMP_B
    else:
        assert comp == "A" and x in X_ALLOWED, (comp, x)
        doms = COMP_A_FIXED + (x,)
    out = tuple(d for d in TRAINER_ORDER if d in doms)
    assert len(out) == 5 and set(out) == set(doms), doms
    return out


def comp_key(comp, x=None):
    """The batch key handed to make_jobs_ptdom.render (injected into its BATCHES / HWOF / SLOT0 for one call)."""
    return f"C6{comp}" + (f"-{x}" if x else "")


def manifolds(comp, x=None):
    return ("numbers",) + comp_doms(comp, x)


def arms(comp, x=None):
    return ("none", "all6") + tuple(f"only-{m}" for m in manifolds(comp, x))


def pri_of(arm, comp, x=None):
    """The active priors of an arm (numbers = the helix3 prior on the operands)."""
    if arm == "none":
        return ()
    if arm == "all6":
        return manifolds(comp, x)
    assert arm.startswith("only-") and arm[5:] in manifolds(comp, x), arm
    return (arm[5:],)


GRID_L1 = {"none": ("2e-3", "4e-3", "8e-3"), "all6": ("1e-3", "2e-3", "4e-3")}
SINGLE_LR = "2e-3"                               # the six single-prior arms (F100's prior pick; not selected here)
step_up = lambda lr: LADDER[LADDER.index(lr) + 1]
step_down = lambda lr: LADDER[LADDER.index(lr) - 1]
EXT_UP = {"all6": step_up(GRID_L1["all6"][-1]), "none": step_up(GRID_L1["none"][-1])}   # once each (ptdom2 A1)
EXT_DOWN = {"none": step_down(GRID_L1["none"][0])}                                        # none only (ptdom2 A1.1 (i))
assert EXT_UP == {"all6": "8e-3", "none": "1.6e-2"} and EXT_DOWN == {"none": "1e-3"}
# fixB (ptdom A5.2 / ptdom2 A1.1 (b)): none at all6's pick when the pick lies outside none's range; all6 has no lower extension,
# so the only possible fixB lr is all6's lowest grid lr below none's lowest (1e-3, = none's down step)
FIXB_LR = tuple(lr for lr in GRID_L1["all6"] + (EXT_UP["all6"],)
                if float(lr) < float(GRID_L1["none"][0]) or float(lr) > float(EXT_UP["none"]))
assert FIXB_LR == ("1e-3",), FIXB_LR
SEL = (0, 1, 2, 3)                              # selection seeds
CONF = tuple(range(8, 16))                      # confirmation seeds (fresh: never read by any selection)
PARTS = ("L1", "L1x", "C", "smoke")
N_SLOTS = 8                                     # singleton B200 chains llmadd_pt6comb-b200-slot0..7, shared by every part
SMOKE_STOP = 8000                               # evals 0..8000 every 250 (33); on-window 1000..7250 (26); 7500..8000 off (3)
SMOKE = (("all6", "2e-3", 0), ("none", "2e-3", 0))
# plan hours per run on B200: ptdom's stage-1 medians were A 0.46 (months k-span), B1 0.35 (six comparisons) (ptdom2
# tests/walltime.out); composition B has three comparisons + s4 / dim (+ their held-out-format sets) and seven gradient passes
# at every eval (dom_grads: the LM and every domain's true prior, in every arm) -> planned 0.45; the GPU smoke calibrates
PLAN_H = 0.45
K = 2
# the read-out keys the job's CHECK info line prints (the verdicts read readout_pt6comb's own definitions)
READ6 = {**{d: MJ.READ[d] for d in MJ.ALL24}, "s4": "s4_readout", "dim": "dim_readout"}
SEEN6 = {**{d: MJ.SEEN[d] for d in MJ.ALL24}, "s4": "s4_seen", "dim": "dim_seen"}
N_FMT = {"months": 4, "s4": 3, "dim": 3}        # the trainer's held-out-format templates (PARA) per domain
_DOM_KIND0 = MJ.dom_kind                         # make_jobs_ptdom.dom_kind captured before any re-pointing
DOM_KEYS6 = {**MJ.DOM_KEYS, "toy": ("n_train", "n_held", "n_readout", "readout", "n_items", "split_seed", "split_mode", "train_frac")}


def dom_kind6(d):
    return "toy" if d in TOY else _DOM_KIND0(d)


def domains_arg(comp, x=None):
    return "numbers," + ",".join(comp_doms(comp, x))


def opt(name, default=None):
    i = sys.argv.index(name) if name in sys.argv else None
    return sys.argv[i + 1] if i is not None and i + 1 < len(sys.argv) else default


# ---------------------------------------------------------------- names
def tag6(comp, arm, lr, x=None):
    return f"{PREFIX}-{comp}{('-' + x) if x else ''}-{arm}-lr{lr}"


def name6(comp, arm, lr, seed, rr=0, smoke=False, x=None):
    return (f"{PREFIX}-smoke-" if smoke else f"{PREFIX}-") + f"{comp}{('-' + x) if x else ''}-{arm}-lr{lr}-s{seed}" + (f"-r{rr}" if rr else "")


def slot_name(slot):
    return f"llmadd_{PREFIX}-b200-slot{slot}"


# ---------------------------------------------------------------- the trainer's command line, intent, input checks
W = MJ.W   # "--schedule window --window 0.05,0.4 --site ctx --lam_rule each"


def arm_flags6(arm, comp, x=None):
    pri = pri_of(arm, comp, x)
    if not pri:
        return "--target none"
    tgt = "helix3" if "numbers" in pri else "none"
    return f"--target {tgt} {W} --prior_domains {','.join(pri)} --lam 1"


def want_flags6(arm, comp, x=None, smoke=False):
    pri = pri_of(arm, comp, x)
    w = ({"--target": "none", "--schedule": None, "--window": None, "--site": None, "--lam_rule": None, "--prior_domains": None,
          "--prior_shuf": None, "--lam": None} if not pri else
         {"--target": "helix3" if "numbers" in pri else "none", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx",
          "--lam_rule": "each", "--prior_domains": ",".join(pri), "--prior_shuf": None, "--lam": "1"})
    if smoke:
        w["--stop_after"] = str(SMOKE_STOP)
    return w


def args_of6(comp, arm, lr, seed, out, x=None, smoke=False):
    """The trainer's command line = ptdom's layout (make_jobs_ptdom.args_of) with this composition's --domains, the arm's prior
    flags and no --save_model (ptdom's stage-1 body)."""
    tag = tag6(comp, arm, lr, x)
    ck = HW["b200"]["ckpt"]
    return (f"--data {MJ.DATA} --tokenizer {MJ.TOK} --tokenization whole --out {out}/{tag} {arm_flags6(arm, comp, x)} --arith_lines 1 "
            f"--domains {domains_arg(comp, x)} --extra_lines {K} --lr {lr} --seed {seed} --tag _{tag} --ckpt_every {ck}"
            + (f" --stop_after {SMOKE_STOP}" if smoke else ""))


def intent6(P, comp, arm, lr, seed, ref, x=None, smoke=False):
    """Everything CHECK_PY_6 asserts on the run JSON (make_jobs_ptdom.intent generalised: the numbers prior may be active; the
    toy domains s4 / dim with their n_fmt and split facts; no table sha for months / s4 / dim)."""
    doms = comp_doms(comp, x)
    pri = pri_of(arm, comp, x)
    events, on_frac = P.window_events(P.STEPS) if pri else ([], 0.0)
    if pri:
        assert events == [[916, 1], [7324, 0]], events
    lam_rule = "each" if pri else "norm"
    hw = HW["b200"]
    I = {"batch": comp_key(comp, x), "code_sha": WANT_SHA["pretrain_range.py"], "dep_sha": {f: WANT_SHA[f] for f in MJ.DEP},
         "steps": P.STEPS, "n_params": P.N_PARAMS, "grad_ckpt_on": False, "geo_layers": [2, 4], "read_layers": [2, 4], "lm_chunk": 0,
         "eval_every_steps": P.EVAL_EVERY, "lam_rule": lam_rule, "save_model": False, "pilot": bool(smoke),
         "lam_domains": {d: (1.0 if d in pri else 0.0) for d in ("numbers",) + doms}, "K": K, "ext": list(doms),
         "comp": [d for d in doms if d in MJ.COMP], "kstep": [d for d in doms if d in MJ.KSTEP], "toy": [d for d in doms if d in TOY],
         "tables": [d for d in doms if d in MJ.ALL24 and d != "months"], "active": list(pri), "geo_tol": MJ.GEO_TOL,
         "evals": len(range(0, P.STEPS, P.EVAL_EVERY)) + ((P.STEPS - 1) % P.EVAL_EVERY != 0), "events": events,
         "on_frac": on_frac, "lm0_ref": ref.get("lm0"), "data_sha0": ref.get("dsha0"), "dom": ref.get("dom"),
         "table_sha": ref.get("table_sha"), "n_fmt": {d: N_FMT[d] for d in doms if d in N_FMT}, "cka_rise": MJ.CKA_RISE,
         "dp_tol0": MJ.DP_TOL0, "dp_tol": {d: MJ.DP_PER_TOK * MJ.SPAN_TOK[d] for d in doms if d in MJ.ALL24},
         "numbers_share": [0.019, 0.024], "read": {d: READ6[d] for d in doms}, "seen": {d: SEEN6[d] for d in doms}, "gpu": hw["gpu"],
         "args": {"data": None, "tokenizer": str(MJ.TOK), "out": None, "tokenization": "whole",
                  "target": "helix3" if "numbers" in pri else "none", "site": "ctx", "schedule": "window" if pri else "whole",
                  "window": "0.05,0.4", "arith_lines": 1, "lr": float(lr), "lam": 1.0 if pri else 5.0, "tokens": 3e8,
                  "bs": P.BS, "seq": P.SEQ, "d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536, "seed": seed,
                  "tag": f"_{tag6(comp, arm, lr, x)}", "save_model": False, "ckpt_every": hw["ckpt"], "geo_layers": None,
                  "eval_every": P.EVAL_EVERY, "n_eval": 150, "grad_ckpt": "auto", "lm_chunk_tokens": -1, "untie": False,
                  "gate_thr": 0.5, "gate_start": 0.0, "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256,
                  "stop_after": SMOKE_STOP if smoke else -1, "dry_inputs": False, "domains": domains_arg(comp, x), "extra_lines": K,
                  "prior_domains": ",".join(pri), "prior_shuf": "", "lam_rule": lam_rule, "cyc_held": 0.4, "cyc_heldE": 0.5,
                  "s4_train": 0.6, "dim_train": 0.5, "dom_eval_rows": 200, "dom_seen": 150}}
    return I


def dry_block6(comp, arm, x=None):
    """The arm-dependent input checks of --dry_inputs: the numbers target (helix3 table (1000, 10) / none, rb6's check), per
    domain prior / shuf / lam in 'domains:' (ptdom's block)."""
    pri = pri_of(arm, comp, x)
    tgt = "helix3" if "numbers" in pri else "none"
    ta = ("""grep -qxF 'target features (1000, 10)' "$OUT/dry_inputs.txt" || { echo "## not a helix3 table (1000 x 10) (this arm's numbers prior)"; exit 3; }"""
          if tgt == "helix3" else
          """! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a numbers prior target (this arm has none)"; exit 3; }""")
    specs = " ".join(f"{d}:{str(d in pri).lower()}:false:{json.dumps(1.0 if d in pri else 0.0)}" for d in ("numbers",) + comp_doms(comp, x))
    return (ta + """
dl=$(grep -m1 '^domains: ' "$OUT/dry_inputs.txt" || true)
has() { grep -qF -e "$1," -e "$1}" <<< "$blk"; }
for sp in """ + specs + """; do
  IFS=: read -r d p s l <<< "$sp"
  blk=$(grep -oE "\\"$d\\": \\{[^}]*\\}" <<< "$dl" || true)
  has "\\"prior\\": $p" && has "\\"shuf\\": $s" && has "\\"lam\\": $l" ||
    { echo "## domain $d not prior $p / shuf $s / lam $l in the domains: line"; exit 3; }
done
grep -qF '"numbers": {"target": \"""" + tgt + """\", ' <<< "$dl" || { echo "## numbers target not """ + tgt + """"; exit 3; }""")


# ---------------------------------------------------------------- CHECK_PY_6 (exact, counted edits of ptdom's CHECK_PY)
GEO_NAME6 = "geo > {TOL} for every active prior (numbers: geo) and 0 < grad_prior < inf at every on-eval"
NUM_ON_CHECK = "numbers prior applied on every on-step"
TOY_CHECK = "toy read-outs (s4 / dim, Rev B exact match) at every eval"


def _check_py_6(cp):
    # 1. the numbers target is the arm's (helix3 for all6 / only-numbers), not always none
    cp = _edit(cp, 'chk("domains.<d> prior / shuf / lam = intent (active " + (",".join(ACT) or "none") + "; numbers target none)",\n'
                   '    dp == dw and r["domains"]["numbers"].get("target") == "none" and set(r["domains"]) == set(ALLD), dp)\n',
               'chk("domains.<d> prior / shuf / lam = intent (active " + (",".join(ACT) or "none") + "; numbers target " + I["args"]["target"] + ")",\n'
               '    dp == dw and r["domains"]["numbers"].get("target") == I["args"]["target"] and set(r["domains"]) == set(ALLD), dp)\n')
    # 2. the toy domains' read-outs (s4 / dim) and their per-question hits (the read-out's s4_hard / dim strict sets need them)
    cp = _edit(cp, '# The double paths of the 2-way and the k-span scoring (Rev E / G): 2-way = the span log-prob vs the direct per-position path on\n',
               'TOYD = tuple(I.get("toy", ()))   # pt6comb: the Rev B toy domains (exact match; per-question hits for s4_hard / the strict dim set)\n'
               'if TOYD:\n'
               '    tk_ = [f"{d}_{s_}" for d in TOYD for s_ in ("held", "seen")]\n'
               '    nq_ = {d: len(r["domains"][d].get("eval_held", {}).get("q", [])) for d in TOYD}\n'
               '    tb_ = [k for k in tk_ if any(not isinstance(e.get(k), float) for e in c)] + [f"hits_{d}_held" for d in TOYD if not nq_[d] or any(\n'
               '        not (isinstance(e.get(f"hits_{d}_held"), str) and len(e[f"hits_{d}_held"]) == nq_[d] and e[f"hits_{d}_held"].isdigit()) for e in c)]\n'
               '    chk(f"' + TOY_CHECK + ' ({len(tk_)} sets) and the per-question hits hits_<d>_held (one digit per held question, every eval)",\n'
               '        not tb_, tb_)\n'
               '# The double paths of the 2-way and the k-span scoring (Rev E / G): 2-way = the span log-prob vs the direct per-position path on\n')
    # 3. the GPU-numerics tolerance text lists the domains that have one (s4 / dim have no scoring double path)
    cp = _edit(cp, 'tol_txt = "; ".join(f"{d} {I[\'dp_tol\'][d]:g}" for d in EXT)\n',
               'tol_txt = "; ".join(f"{d} {I[\'dp_tol\'][d]:g}" for d in EXT if d in I["dp_tol"])\n')
    # 4. the numbers prior: applied on every on-step when active (rb6's check), else off (ptdom's check, unchanged)
    cp = _edit(cp, '    chk("numbers prior off: never applied, no numbers loss term (geo == 0 at every eval), no weighted numbers gradient",\n'
                   '        c[-1]["prior_used_frac_numbers"] == 0 and all(e["geo"] == 0 for e in c) and all("gnorm_prior_numbers" not in e for e in c),\n'
                   '        c[-1]["prior_used_frac_numbers"])\n',
               '    if "numbers" in ACT:   # pt6comb: the numbers prior (helix3 on the operands, ctx) is active (rb6\'s check + its value logged)\n'
               '        chk("' + NUM_ON_CHECK + ' (ctx: prior_used_frac == prior_used_frac_numbers == prior_on_frac); its value geo logged "\n'
               '            "(finite) at every eval before a divergence", abs(c[-1]["prior_used_frac"] - on_frac) < 1e-9\n'
               '            and abs(c[-1]["prior_used_frac_numbers"] - on_frac) < 1e-9 and all(num_(e.get("geo")) for e in pre),\n'
               '            [c[-1]["prior_used_frac"], c[-1]["prior_used_frac_numbers"], on_frac])\n'
               '    else:\n'
               '        chk("numbers prior off: never applied, no numbers loss term (geo == 0 at every eval), no weighted numbers gradient",\n'
               '            c[-1]["prior_used_frac_numbers"] == 0 and all(e["geo"] == 0 for e in c) and all("gnorm_prior_numbers" not in e for e in c),\n'
               '            c[-1]["prior_used_frac_numbers"])\n')
    # 5. the geo check covers every active prior; the numbers prior's value is 'geo'
    cp = _edit(cp, '    chk(f"geo > {TOL} for the active prior and 0 < grad_prior < inf at every on-eval (before a divergence)",\n'
                   '        (bool(on) or diverged) and all(all(e.get(f"geo_{d}") is None or e[f"geo_{d}"] > TOL for d in ACT)\n',
               '    gv_ = lambda e, d: e.get("geo") if d == "numbers" else e.get(f"geo_{d}")   # pt6comb: the numbers prior\'s value is geo\n'
               '    chk(f"' + GEO_NAME6 + ' (before a divergence)",\n'
               '        (bool(on) or diverged) and all(all(gv_(e, d) is None or gv_(e, d) > TOL for d in ACT)\n')
    # 6. the CKA-rise bar: numbers = the operands' CKA to helix3 (cka_ops_helix3, rb6's)
    cp = _edit(cp, '        cka_true = {d: round(we[f"cka_{d}"] - c[0][f"cka_{d}"], 4) for d in ACT}\n',
               '        ck_ = lambda e, d: e["cka_ops_helix3"] if d == "numbers" else e[f"cka_{d}"]   # pt6comb: numbers = cka_ops_helix3 (rb6)\n'
               '        cka_true = {d: round(ck_(we, d) - ck_(c[0], d), 4) for d in ACT}\n')
    # 7. info: the numbers prior's CKA and the toy read-outs' areas
    cp = _edit(cp, '"final_cka": {d: c[-1].get(f"cka_{d}") for d in EXT}, "cka_rise_true": cka_true,',
               '"final_cka": {d: c[-1].get(f"cka_{d}") for d in EXT}, "cka_rise_true": cka_true, "final_cka_ops_helix3": c[-1].get("cka_ops_helix3"),')
    return cp


CHECK_PY_6 = _check_py_6(MJ.CHECK_PY)
ORIG = {k: getattr(MJ, k) for k in ("intent", "want_flags", "dry_block", "tag_of", "arms", "args_of", "render", "CHECK_PY",
                                     "dom_kind", "DOM_KEYS")}


# ---------------------------------------------------------------- rendering through make_jobs_ptdom.render
_ctx = {}   # (comp, x, smoke) of the call in progress (read by the re-pointed functions)


@contextmanager
def patched(comp, x=None, smoke=False):
    """Re-point make_jobs_ptdom's module names for one render() call (restored afterwards; the injected batch key removed)."""
    key = comp_key(comp, x)
    names = ("R", "tag_of", "job_name", "slurm_name", "args_of", "intent", "want_flags", "dry_block", "arms", "saves", "STAGES",
             "SEL", "CONF", "SLOT0", "SLOTS", "CHECK_PY", "dom_kind", "DOM_KEYS")
    keep = {k: getattr(MJ, k) for k in names}
    assert key not in MJ.BATCHES and key not in MJ.HWOF
    try:
        _ctx.update(comp=comp, x=x, smoke=smoke)
        MJ.BATCHES[key] = comp_doms(comp, x)
        MJ.HWOF[key] = "b200"
        MJ.R = R
        MJ.tag_of = lambda b_, a_, l_: tag6(comp, a_, l_, x)
        MJ.job_name = lambda b_, a_, l_, s_, rr=0, pilot=False: name6(comp, a_, l_, s_, rr, smoke, x)
        MJ.slurm_name = lambda b_, a_, l_, s_, rr=0, slot=None, pilot=False: slot_name(slot)
        MJ.args_of = lambda b_, a_, l_, s_, out, save=True, pilot=False: args_of6(comp, a_, l_, s_, out, x, smoke)
        MJ.intent = lambda P_, b_, a_, l_, s_, ref, save=True, pilot=False: intent6(P_, comp, a_, l_, s_, ref, x, smoke)
        MJ.want_flags = lambda a_: want_flags6(a_, comp, x, smoke)
        MJ.dry_block = lambda b_, a_: dry_block6(comp, a_, x)
        MJ.arms = lambda b_: arms(comp, x)
        MJ.saves = lambda stage, pilot=False: False
        MJ.STAGES = PARTS
        MJ.SEL, MJ.CONF = SEL, tuple(range(4, 16))
        MJ.SLOT0, MJ.SLOTS = {key: 0}, N_SLOTS
        MJ.CHECK_PY = CHECK_PY_6
        MJ.dom_kind = dom_kind6
        MJ.DOM_KEYS = DOM_KEYS6
        yield key
    finally:
        for k, v in keep.items():
            setattr(MJ, k, v)
        MJ.BATCHES.pop(key, None)
        MJ.HWOF.pop(key, None)
        _ctx.clear()


def desc6(arm, comp, x=None):
    pri = pri_of(arm, comp, x)
    if not pri:
        return "none (no prior)"
    if arm == "all6":
        return "all6: the six priors " + " + ".join(pri) + " (lambda 1 each, window (0.05, 0.4), ctx; numbers = helix3 on the operands)"
    return f"only the {pri[0]} prior (lambda 1, window (0.05, 0.4), ctx" + ("; helix3 on the operands)" if pri[0] == "numbers" else ")")


def render6(P, part, comp, arm, lr, seed, ref, commit, rr=0, slot=0, x=None, smoke=False):
    """-> (name, tag, text, args, intent) of one pt6comb cell, rendered by make_jobs_ptdom.render + counted edits."""
    assert part in PARTS and arm in arms(comp, x) and lr in LADDER and (smoke or slot is not None)
    with patched(comp, x, smoke) as key:
        desc0 = (f"only the {arm[5:]} prior (lambda 1, window (0.05, 0.4), ctx)" if arm != "none" else "none (no prior)")
        name, tag, text, args, I = MJ.render(P, key, arm, lr, seed, ref, commit, rr, slot, part, False)
    assert name == name6(comp, arm, lr, seed, rr, smoke, x) and tag == tag6(comp, arm, lr, x)
    doms = comp_doms(comp, x)
    n5 = len([d for d in doms if d != "months"])          # make_jobs_ptdom.render's n_site (every non-months domain)
    n_site = len([d for d in doms if d in MJ.ALL24 and d != "months"])   # the domains that print a site-rule line (Rev E / F)
    text = _edit(text, f"# ptdom A3 / A4 batch {key} stage {part}: {desc0} @ lr",
                 f"# pt6comb part {part} (the ptdom A3-A6.1 / ptdom2 stage-1 job body), composition {comp}{(' X=' + x) if x else ''}: "
                 f"{desc6(arm, comp, x)} @ lr")
    text = _edit(text, "; code = runs/ptdom/code (COMMIT", "; code = runs/pt6comb/code (COMMIT")
    text = _edit(text, "# spec and pre-registered screen / verdict runs/ptdom/AUDIT.md (Amendments A3, A4); generated by llmadd_torch/make_jobs_ptdom.py\n",
                 "# spec and pre-registered verdict runs/pt6comb/AUDIT.md; rendered by llmadd_torch/make_jobs_pt6comb.py through make_jobs_ptdom.render\n")
    text = _edit(text, f"; 16 chains over the four batches).", f"; {N_SLOTS} chains shared by every pt6comb part).")
    text = _edit(text, "# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/ptdom/HOLD (every batch) or\n"
                       f"# runs/ptdom/HOLD_{key} (A4: this batch only) exists\n"
                       f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{key}" ]; then echo "HOLD present: exiting"; exit 0; fi\n',
                 "# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/pt6comb/HOLD (every part) or\n"
                 f"# runs/pt6comb/HOLD_{part} (this part only) exists\n"
                 f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{part}" ]; then echo "HOLD present: exiting"; exit 0; fi\n')
    text = _edit(text, "--dry_inputs: model line, no numbers target, per domain",
                 "--dry_inputs: model line, the numbers target (" + ("helix3" if "numbers" in pri_of(arm, comp, x) else "none") + "), per domain")
    text = _edit(text, "# 4. CHECK_PY (make_jobs_ptdom.py) with this run's intent", "# 4. CHECK_PY_6 (make_jobs_pt6comb.py: ptdom's CHECK_PY + counted edits) with this run's intent")
    text = _edit(text, f"the site rule ({n5} domains asserted)",f"the site rule ({n_site} domains asserted: the comparisons; s4 / dim / months: "
                 "single-token sites asserted by the trainer at step 0)")
    text = _edit(text, f'"$OUT/dry_inputs.txt")" = {n5} ] || {{ echo "## the site rule is not asserted for the {n5} domains with item-name sites"; exit 3; }}',
                 f'"$OUT/dry_inputs.txt")" = {n_site} ] || {{ echo "## the site rule is not asserted for the {n_site} domains with item-name sites"; exit 3; }}')
    # the run JSON's name carries the numbers target (pretrain_range.py:574 run_<tokenization>_<target>_seed<s><tag>.json):
    # ptdom's render hard-codes 'none' (every ptdom arm had --target none); a numbers-prior arm writes run_whole_helix3_...
    tgt = I["args"]["target"]
    if tgt != "none":
        text = _edit(text, f'"$OUT/$TAG/run_whole_none_seed{seed}_{tag}.json"', f'"$OUT/$TAG/run_whole_{tgt}_seed{seed}_{tag}.json"')
    if smoke:   # the smoke's bodies live in runs/pt6comb/smoke/ (run by the smoke job with bash; their #SBATCH lines are inert)
        n_ = text.count(f"{R}/out/{name}")
        assert n_ == 2, n_   # OUT= and the --out of ARGS
        text = text.replace(f"{R}/out/{name}", f"{R}/smoke/{name}")
        args = args.replace(f"{R}/out/{name}", f"{R}/smoke/{name}")
    return name, tag, text, args, I


# ---------------------------------------------------------------- snapshot, references
def snapshot():
    """runs/pt6comb/code = git archive of tools/ at COMMIT (11f61b9, the ptdom trainer); asserted = WANT_SHA, = the commit's files
    and byte-identical to runs/ptdom/code/tools (the frozen ptdom snapshot), every file. -> commit."""
    R.mkdir(parents=True, exist_ok=True)
    if not (R / "code").exists():
        tmp = R / "code.tmp"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        arc = subprocess.run(["git", "-C", str(MJ.REPO), "archive", COMMIT, "tools"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
        (tmp / "COMMIT").write_text(f"{COMMIT} (ptdom, committed, not pushed; git archive of tools/ = runs/ptdom/code); sha256[:12]: "
                                    + ", ".join(f"{f} {sha(tmp / 'tools' / f)}" for f in WANT_SHA) + "\n")
        tmp.rename(R / "code")
    cm = (R / "code" / "COMMIT").read_text()
    assert cm.split()[0] == COMMIT
    for f, v in WANT_SHA.items():
        s_ = sha(R / "code" / "tools" / f)
        assert f"{f} {s_}" in cm and s_ == v, (f, s_, v)
        assert hashlib.sha256(subprocess.run(["git", "-C", str(MJ.REPO), "show", f"{COMMIT}:tools/{f}"], capture_output=True,
                                             check=True).stdout).hexdigest()[:12] == s_, (f, COMMIT)
    mine = sorted(p.relative_to(R / "code" / "tools") for p in (R / "code" / "tools").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    theirs = sorted(p.relative_to(R0 / "code" / "tools") for p in (R0 / "code" / "tools").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert mine == theirs, "runs/pt6comb/code/tools and runs/ptdom/code/tools list different files"
    for p in mine:
        assert (R / "code" / "tools" / p).read_bytes() == (R0 / "code" / "tools" / p).read_bytes(), p
    return COMMIT


def refs_path(comp, x=None):
    return R / f"pt6comb_refs_{comp}{('-' + x) if x else ''}.json"


def refs6(comp, x=None):
    """CPU references per seed (tests/refs_pt6comb.py; runs/pt6comb/pt6comb_refs_<comp>.json, written once, chmod 444)."""
    p = refs_path(comp, x)
    if not p.exists():
        return {}
    j = json.loads(p.read_text())
    assert j.get("design") == PREFIX and j.get("comp") == comp and j.get("x") == x and j.get("domains") == domains_arg(comp, x), (p, j.get("comp"))
    return {int(k): v for k, v in j["seeds"].items()}


def load_P():
    snapshot()
    sys.path.insert(0, str(R / "code" / "tools"))
    import make_rb6_tasks as P
    assert Path(P.__file__).resolve().parent in ((R / "code" / "tools").resolve(), (R0 / "code" / "tools").resolve())
    assert (P.BS, P.SEQ, P.STEPS, P.EVAL_EVERY, P.N_PARAMS) == (32, 512, 18310, 250, 52672128)
    return P


# ---------------------------------------------------------------- cells per part
def read_picks(path, stage):
    """A picks file of select_pt6comb.py (written once, chmod 444): design pt6comb, decided, the stage it writes."""
    pk = json.loads(Path(path).read_text())
    assert pk.get("design") == PREFIX and pk.get("status") == "decided" and pk.get("stage_written") == stage, \
        (pk.get("design"), pk.get("status"), pk.get("stage_written"), stage)
    assert not (Path(path).stat().st_mode & 0o222), f"{path} is writable: a decision file is chmod 444"
    return pk


def cells(part, comp="B", picks=None, x=None):
    """[(arm, lr, seed)] of a part in submission order (seed-major, so a partial part stays paired)."""
    if part == "L1":
        return [(a, lr, s) for s in SEL for a in ("none", "all6") for lr in GRID_L1[a]]
    if part == "smoke":
        return list(SMOKE)
    pk = read_picks(picks, part)
    assert pk.get("comp") == comp and pk.get("x") == x, (pk.get("comp"), comp)
    if part == "L1x":
        E = [tuple(v) for v in pk["E"]]
        ok_ = {("all6", EXT_UP["all6"]), ("none", EXT_UP["none"]), ("none", EXT_DOWN["none"])} | {("none", lr) for lr in FIXB_LR}
        assert E and len(set(E)) == len(E) and set(E) <= ok_, E
        assert not any(lr in GRID_L1[a] for a, lr in E), E
        return [(a, lr, s) for s in SEL for a, lr in E]
    assert part == "C"
    P_, N_ = pk["pick_all6"], pk["pick_none"]
    assert P_ in GRID_L1["all6"] + (EXT_UP["all6"],) and N_ in GRID_L1["none"] + (EXT_UP["none"], EXT_DOWN["none"]) + FIXB_LR, (P_, N_)
    C = c_cells(P_, N_, comp, x)   # A2: + the same-lr pair none@2e-3 / all6@2e-3, merged with none@N / all6@P when equal
    assert [tuple(v) for v in pk["C"]] == C and len(set(C)) == len(C), (pk["C"], C)
    return [(a, lr, s) for s in CONF for a, lr in C]


def c_cells(P_, N_, comp="B", x=None):
    """AUDIT.md Amendment A2: the confirmation cells none@N, all6@P, the same-lr pair none@2e-3 / all6@2e-3 (the interference contrast;
    merged with none@N / all6@P when N / P = 2e-3: no cell twice) and the six only-<m>@2e-3, in this order."""
    out = []
    for c in [("none", N_), ("all6", P_), ("none", SINGLE_LR), ("all6", SINGLE_LR)] + [(f"only-{m}", SINGLE_LR) for m in manifolds(comp, x)]:
        if c not in out:
            out.append(c)
    return out


def hold_paths(part):
    return R / "HOLD", R / f"HOLD_{part}"


# ---------------------------------------------------------------- plan / cost
def plan():
    n1 = len(cells("L1"))
    nx = 4 * 3                       # max: all6 8e-3, none 1.6e-2, none 1e-3 (= fixB's lr) at seeds 0-3
    nC_lo, nC_hi = 8 * 8, 8 * 10   # A2: + none@2e-3 / all6@2e-3 unless N / P = 2e-3 (8-10 cells x 8 seeds)
    h = PLAN_H
    print(f"pt6comb plan (composition B: --domains {domains_arg('B')}; 53M, 3e8 tokens, B200, {N_SLOTS} singleton slot chains; ptdom's "
          f"stage-1 body: no model, no offline LM; plan {h} B200-h per run, calibrated by the GPU smoke):")
    print(f"  L1     {n1:3d} runs = {n1 * h:5.1f} B200-h (seeds 0-3: none {' / '.join(GRID_L1['none'])}, all6 {' / '.join(GRID_L1['all6'])})")
    print(f"  L1x    0-{nx} runs = 0-{nx * h:.1f} B200-h (once: all6 {EXT_UP['all6']}, none {EXT_UP['none']} / {EXT_DOWN['none']}; fixB "
          f"none@{FIXB_LR[0]} = none's down step)")
    print(f"  C      {nC_lo}-{nC_hi} runs = {nC_lo * h:.1f}-{nC_hi * h:.1f} B200-h (seeds 8-15: none@N, all6@P, A2: none@{SINGLE_LR} / all6@{SINGLE_LR} "
          f"(merged when N / P = {SINGLE_LR}), six only-<m>@{SINGLE_LR}; 72 when one of P / N is {SINGLE_LR})")
    print(f"  smoke  1 job (2 bodies x {SMOKE_STOP} steps) <= 1 B200-h (not part of any result)")
    print(f"  total: min {n1 + nC_lo} runs ({(n1 + nC_lo) * h:.1f} B200-h), max {n1 + nx + nC_hi} runs ({(n1 + nx + nC_hi) * h:.1f} B200-h) + the smoke")
    print(f"  wall clock with {N_SLOTS} chains always running: L1 ~{n1 * h / N_SLOTS:.1f} h, L1x <= {nx * h / N_SLOTS:.1f} h, C "
          f"~{nC_lo * h / N_SLOTS:.1f}-{nC_hi * h / N_SLOTS:.1f} h (+ queue waits)")
    print("  disk: no saved models; run dirs ~1-2 MB each (~0.2 GB in all)")


# ---------------------------------------------------------------- main
def main():
    if "--plan" in sys.argv:
        plan()
        return
    if "--snapshot" in sys.argv:
        print("snapshot", snapshot(), (R / "code" / "COMMIT").read_text().strip())
        return
    smoke = "--smoke" in sys.argv
    part = "smoke" if smoke else (opt("--part") or opt("--stage"))
    assert part in PARTS, "--part L1 | L1x | C (or --smoke); --plan for the cost"
    comp, x = opt("--comp", "B"), opt("--x")
    comp_doms(comp, x)
    if "--submit" in sys.argv:
        assert comp == "B", "composition A is a parameter only (coordinator 2026-11-10): not submittable"
    commit = snapshot()
    P = load_P()
    todo = cells(part, comp, opt("--picks"), x)
    rerun = "--rerun" in sys.argv
    assert not (rerun and smoke)
    if rerun:
        todo = rerun_plan(todo, comp, x)
    else:
        todo = [(a, lr, s, 0) for a, lr, s in todo]
    others = {p.name for p in B.glob("runs/*/out/*") if p.parent.parent != R}
    JD = R / "smoke" if smoke else R
    JD.mkdir(parents=True, exist_ok=True)
    RF = refs6(comp, x)
    jobs = []
    for i, (a, lr, s, rr) in enumerate(todo):
        assert s in RF, f"no CPU reference for composition {comp} seed {s}: run runs/pt6comb/tests/refs_pt6comb.py first"
        name, tag, text, args, I = render6(P, part, comp, a, lr, s, RF[s], commit, rr, i % N_SLOTS, x, smoke)
        assert name not in others, f"{name}: an out dir of that name exists in another batch"
        jobs.append((name, text, args, I))
    if smoke:
        jobs = [smoke_job(jobs)]
    written = []
    for name, text, args, I in jobs:
        job, flag = JD / f"job_{name}.sbatch", JD / f"SUBMITTED_{name}"
        if flag.exists():
            if job.read_text() != text:
                print(f"WARNING: {job.name} was submitted and differs from this rendering: kept, not rewritten")
        elif "--no-write" not in sys.argv:
            job.write_text(text)
            written.append(job)
    if "--check" in sys.argv:
        for name, text, args, I in jobs:
            assert subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0, f"{name}: bash -n"
        print(f"check: {len(jobs)} job texts parse (bash -n); every render assertion of make_jobs_ptdom.render held")
    if "--print" in sys.argv:
        for name, text, args, I in jobs:
            print(f"{name}{' (SUBMITTED)' if (JD / f'SUBMITTED_{name}').exists() else ''}: {args}")
    if "--test-only" in sys.argv:
        n_t = int(opt("--test-only")) if str(opt("--test-only")).isdigit() else 0
        for name, text, args, I in jobs[: n_t or len(jobs)]:
            o = subprocess.run(["sbatch", "--test-only", str(JD / f"job_{name}.sbatch")], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
    if "--submit" in sys.argv:
        for h_ in hold_paths(part):
            assert not h_.exists(), f"{h_} exists: remove it to submit (the user decides)"
        for name, text, args, I in jobs:
            flag = JD / f"SUBMITTED_{name}"
            if flag.exists():
                print(f"{name}: refused (SUBMITTED_{name} exists: a name is submitted once; a re-run is a new name: --rerun)")
                continue
            o = subprocess.run(["sbatch", str(JD / f"job_{name}.sbatch")], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
    print(f"pt6comb part {part} composition {comp}{(' X=' + x) if x else ''}{' (re-runs)' if rerun else ''}: {len(jobs)} jobs rendered "
          f"in {JD} ({len(written)} written); code {commit}, trainer {WANT_SHA['pretrain_range.py']}; B200; submitted: "
          f"{'yes' if '--submit' in sys.argv else 'no'}")


def rerun_plan(todo, comp, x=None):
    """ptdom's A2 re-run path on pt6comb names: the latest name has a SUBMITTED_ flag, its job id is not in the queue, its out dir
    holds no finished run JSON (a finished run is data, never re-run) -> the next name <name>-r<k+1>."""
    q = MJ.queue()
    if q is None:
        raise SystemExit("--rerun: squeue failed; a re-run needs the queue state (refused)")
    ids, _ = q
    out = []
    for a, lr, s in todo:
        k = 0
        while (R / f"SUBMITTED_{name6(comp, a, lr, s, k + 1, x=x)}").exists():
            k += 1
        flag = R / f"SUBMITTED_{name6(comp, a, lr, s, k, x=x)}"
        if not flag.exists():
            continue
        txt = flag.read_text().split()
        jid = txt[-1] if txt and txt[-1].isdigit() else None
        if jid is None or jid in ids:
            print(f"re-run refused: {name6(comp, a, lr, s, k, x=x)}: " + ("no job id" if jid is None else f"job {jid} is in the queue"))
            continue
        tg = tag6(comp, a, lr, x)
        rj = R / "out" / name6(comp, a, lr, s, k, x=x) / tg / f"run_whole_{'helix3' if 'numbers' in pri_of(a, comp, x) else 'none'}_seed{s}_{tg}.json"
        if rj.exists():
            try:
                if len(json.loads(rj.read_text())["curve"]) >= 75:
                    print(f"re-run refused: {name6(comp, a, lr, s, k, x=x)}: its run finished (a run is data, never re-run)")
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"re-run: {name6(comp, a, lr, s, k, x=x)} -> {name6(comp, a, lr, s, k + 1, x=x)}")
        out.append((a, lr, s, k + 1))
    return out


def smoke_job(bodies):
    """The ONE smoke job: B200, 1 h, job name llmadd_pt6comb-smoke (not a slot chain); it runs the two rendered bodies (each the
    pt6comb job body of its cell with --stop_after; HOLD, snapshot shas, B200 guard, data sha, dry_inputs + the input checks + DRY
    rows = the CPU reference, training, CHECK_PY_6) from runs/pt6comb/smoke/body_<name>.sh in turn with bash, then
    runs/pt6comb/smoke/smoke_check.py (the pass rules registered in AUDIT.md 'GPU smoke' before it ran)."""
    sd = R / "smoke"
    names = [n for n, *_ in bodies]
    if "--no-write" not in sys.argv:
        sd.mkdir(parents=True, exist_ok=True)
        for n, text, *_ in bodies:
            bf = sd / f"body_{n}.sh"
            if (sd / f"SUBMITTED_{PREFIX}-smoke").exists():   # a submitted smoke's bodies are never rewritten
                if bf.read_text() != text:
                    print(f"WARNING: {bf.name} belongs to the submitted smoke and differs from this rendering: kept")
            else:
                bf.write_text(text)
    run_lines = "\n".join(f'bash "$SD/body_{n}.sh" > "$SD/body_{n}.log" 2>&1; echo "## body {n} rc $?"; tail -n 3 "$SD/body_{n}.log"'
                          for n in names)
    hw = HW["b200"]
    text = f"""#!/bin/bash
#SBATCH --job-name=llmadd_{PREFIX}-smoke
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint=b200
#SBATCH --comment="preemption=yes"
#SBATCH --nice=20
#SBATCH --cpus-per-task={hw['cpus']}
#SBATCH --mem={hw['mem']}
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={B}/logs/llmadd_{PREFIX}-smoke_%j.out
# pt6comb GPU smoke (runs/pt6comb/AUDIT.md 'GPU smoke'; not part of any result): two runs on the real schedule up to
# --stop_after {SMOKE_STOP} (evals 0..{SMOKE_STOP} every 250; on-window evals 1000..7250; 7500..{SMOKE_STOP} after the window): {', '.join(names)}.
# Each body = the pt6comb job body of its cell (rendered by make_jobs_pt6comb.py through make_jobs_ptdom.render; no saved model, the
# stop_after flag in ARGS and intent); its CHECK fails by design exactly the truncation checks (75 evals, prior_on_frac);
# smoke_check.py asserts that, the six priors' window / sites / fit, the LM share, data identity of the two arms, memory, speed.
set -u
SD={sd}; PY={PY}
if [ -e "{R}/HOLD" ] || [ -e "{R}/HOLD_smoke" ]; then echo "HOLD present: exiting"; exit 0; fi
date -u +%FT%TZ; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
{run_lines}
export PYTHONPATH={B}/pylib
"$PY" "$SD/smoke_check.py" {' '.join(names)} | tee "$SD/smoke_check.out"
# the truncated runs' resume checkpoints (~0.6 GB each; --stop_after leaves them) are not needed: remove exactly these two files
{chr(10).join(f'rm -f "$SD/{n}/{n.replace("pt6comb-smoke-", "pt6comb-").rsplit("-s", 1)[0]}/ckpt_whole_' + ('helix3' if '-all6-' in n else 'none') + f'_seed0_{n.replace("pt6comb-smoke-", "pt6comb-").rsplit("-s", 1)[0]}.pt"' for n in names)}
echo "## smoke done $(date -u +%FT%TZ)"
"""
    return f"{PREFIX}-smoke", text, "; ".join(a for _, _, a, _ in bodies), [I for *_, I in bodies]


if __name__ == "__main__":
    main()
