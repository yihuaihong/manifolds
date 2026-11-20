"""Torch jobs of runs/pt6v2: the second six-manifold combination (pre-registered in runs/pt6v2/AUDIT.md before any pt6v2
result; design = the user's decision 2026-11-18, single stage, mixed GPU types; build 2026-11-19).

Composition = a parameter: numbers + five domains from ALLOWED (the trainer's order), fixed at pick time by --set_comp
(runs/pt6v2/COMPOSITION.json, written once, chmod 444); PLACEHOLDER (dim, cities, countries, compass, hcube) for build / tests.
Everything else = pt6comb composition B's C job body (53M, 3e8 tokens, numbers line 2% first, K = 2 lines cycling over the
five, window (0.05, 0.4), site ctx, lam_rule each, lambda 1, numbers prior helix3) on Rev H (runs/pt6v2/code = git archive of
llmadd-burst-ptdom3 tools/ at ef7b8854, asserted byte-identical to runs/ptdom3/code/tools).
Cells (one stage, seeds 16-23, 13 x 8 = 104 runs; AUDIT.md Amendment V1 added none@4e-3): all6@1e-3 (P, pt6comb's pick), all6@2e-3,
none@{2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2},
only-<m>@2e-3 for the six m; + one pipeline canary (all6@2e-3 seed 16, 400 steps, its own name / out dir runs/pt6v2/pilot/).
GPU type per SEED (--alloc b200:16,17 l40s:18,19,20 a100:21,22,23): all 13 cells of a seed on one type; the job body asserts
the seed's type (exit 3 otherwise) and the intent / check.json carry it ("device").
Every job is rendered by make_jobs_ptdom.render (frozen, sha asserted) with module names re-pointed for one call, then exact,
counted edits (make_jobs_ptdom2._edit); CHECK_PY = pt6comb's CHECK_PY_6 (no toy3 domain) or ptdom3's CHECK_PY_3 (a toy3 domain).
    python make_jobs_pt6v2.py --plan [--comp d1,..,d5] [--alloc ...]
    python make_jobs_pt6v2.py --snapshot
    python make_jobs_pt6v2.py --set_comp d1,d2,d3,d4,d5 --by "<who, date>"       (pick time; once)
    python make_jobs_pt6v2.py --alloc ... [--comp ...] [--canary_dev l40s] [--check] [--print] [--write] [--test-only N] [--submit]
--submit: only the COMPOSITION.json composition, with its CPU references, every seed allocated, no HOLD, a name submitted once;
the canary is submitted first in the same call.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pt6v2"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
FROZEN = {"make_jobs_ptdom.py": "6061b299129c", "make_jobs_ptdom2.py": "62d680144854", "make_jobs_pt6comb.py": "01d1ad3d2867",
          "make_jobs_ptdom3.py": "1046e44f3409"}
for f_, v_ in FROZEN.items():
    assert sha(B / f_) == v_, f"{f_} changed: {sha(B / f_)} != {v_}"
sys.path.insert(0, str(B))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_ptdom as MJ     # noqa: E402  (read only)
import make_jobs_ptdom2 as M2    # noqa: E402  (read only: _edit, R_DOMS)
import make_jobs_pt6comb as M6   # noqa: E402  (read only: CHECK_PY_6, N_FMT, READ6 / SEEN6, TRAINER_ORDER)
import make_jobs_ptdom3 as M3    # noqa: E402  (read only: CHECK_PY_3, READSET, DEP, STEP0_MARK, Rev H shas)
sys.argv = _argv
for m_ in (MJ, M2, M6, M3):
    assert Path(m_.__file__).resolve().parent == B.resolve()
_edit = M2._edit

REPO, COMMIT = M3.REPO, M3.COMMIT                 # Rev H (branch ptdom3, ef7b8854, not pushed)
WANT_SHA = M3.WANT_SHA
assert COMMIT == "ef7b8854" and WANT_SHA["pretrain_range.py"] == "a398fccbfe93"
R3 = B / "runs" / "ptdom3"                        # its validated snapshot: runs/pt6v2/code/tools must equal it byte for byte
PY, LADDER, K = MJ.PY, MJ.LADDER, 2
PREFIX, KEY = "pt6v2", "V6"

# ---------------------------------------------------------------- design (AUDIT.md "Design")
TOY, TOY3 = M6.TOY, M3.TOY3
TRAINER_ORDER = M6.TRAINER_ORDER + TOY3           # pretrain_range.py EXTRA + ANCHOR + NEWDOM + TOY3 (Rev H)
PART_R = tuple(M2.R_DOMS)                         # stars, landmarks, compass, prefixes, ranks, scalar_adj
ALLOWED = ("s4", "dim", "cities", "countries", "people", "months") + PART_R + TOY3
assert len(set(ALLOWED)) == 18 == len(ALLOWED) and set(ALLOWED) <= set(TRAINER_ORDER), (ALLOWED, set(ALLOWED) - set(TRAINER_ORDER))
PLACEHOLDER = ("dim", "cities", "countries", "compass", "hcube")
SOURCE = {**{d: "pt6comb-B" for d in M6.COMP_B}, "months": "ptdom-A", **{d: f"ptdom-{b}" for d, b in M2.R_DOMS.items()},
          **{d: "ptdom3-T3" for d in TOY3}}       # each domain's source batch (its read-out and exposure reference)
assert set(SOURCE) == set(ALLOWED)
SOURCE_DOMS = {"pt6comb-B": M6.COMP_B, **{f"ptdom-{b}": tuple(MJ.BATCHES[b]) for b in ("A", "B2", "B3")}, "ptdom3-T3": TOY3}
SEEDS = tuple(range(16, 24))                      # fresh (pt6comb used 0-15)
HALVES = (SEEDS[:4], SEEDS[4:])                   # the A2.1 GATE per half
P_LR, SAME_LR = "1e-3", "2e-3"
NONE_LRS = ("2e-3", "4e-3", "8e-3", "1.6e-2", "3.2e-2")   # Amendment V1 (2026-11-19): + 4e-3
CANARY = ("all6", "2e-3", 16)
N_SLOTS = 8
CKPT = 500                                        # every device (pt6comb's B200 value): the trainer args are device-independent
assert MJ.HW["b200"]["ckpt"] == CKPT and all(lr in LADDER for lr in NONE_LRS + (P_LR, SAME_LR))
EXTRA_NONE_OK = ("6.4e-2",)   # AUDIT.md Amendment V2 (conditional; opt-in --extra_none 6.4e-2): none@6.4e-2 for all 8 seeds
assert all(x in LADDER and float(x) > float(max(NONE_LRS, key=float)) for x in EXTRA_NONE_OK)
# GPU types (AUDIT.md "GPU types"). B200: pt6comb's header / protocol unchanged (MJ.HW["b200"]). Measured 53M training on B200:
# pt6comb 1,091-1,120 s, ptdom3 965-970 s (runs/pt6v2/tests/walltime_scan.out) -> plan 0.45 h incl. start-up / dry / check.
# No ptdom-style 53M run exists on L40S / A100: A100 = rb6 53M K 2 3,075 s (make_jobs_ptdom.py:82) x 1.1 (ptdom's eval rows)
# + 0.1 h = 1.05 h -> limit 3 h (2.9x); L40S = that x 1.31 (L40S / A100 at 53M, make_jobs_ptdom.py:81) = 1.35 h -> limit 4 h (3.0x).
HW6 = {"b200": MJ.HW["b200"],
       "l40s": dict(gpu="L40S", plan_h=1.35, time="04:00:00", ckpt=CKPT, cpus=8, mem="64G"),
       "a100": dict(gpu="A100", plan_h=1.05, time="03:00:00", ckpt=CKPT, cpus=8, mem="64G")}
PLAN_H = {"b200": 0.45, "l40s": 1.35, "a100": 1.05}
# 2026-11-19 coordinator (pending decision 6): the A100 guard accepts any A100, 40GB or 80GB (53M fits; same architecture)
GPU_CASE = {"b200": "*B200*", "l40s": "*L40S*", "a100": "*A100*"}
GPU_TXT = {"b200": "a B200", "l40s": "an L40S", "a100": "an A100 (40GB or 80GB)"}
A100_HEAD = ("#SBATCH --account=<slurm-account>\n#SBATCH --partition=a100_tandon\n")   # make_jobs_pt2sz.py:71
L40S_HEAD = ("#SBATCH --account=<slurm-account>\n#SBATCH --partition=l40s_public\n")          # make_jobs_ptdom.py render
READ_V2 = {**M6.READ6, **M3.READ3}
SEEN_V2 = {**M6.SEEN6, **M3.SEEN3}
DOM_KEYS_V2 = {**MJ.DOM_KEYS, "toy": M6.DOM_KEYS6["toy"], "toy3": M6.DOM_KEYS6["toy"]}
_KIND0 = MJ.dom_kind


def dom_kind(d):
    return "toy3" if d in TOY3 else ("toy" if d in TOY else _KIND0(d))


def comp_doms(comp):
    """The five non-number domains (a list / tuple / 'a,b,..' string, 'numbers' ignored), in the trainer's order."""
    xs = [d for d in (comp.split(",") if isinstance(comp, str) else comp) if d and d != "numbers"]
    assert len(xs) == 5 and len(set(xs)) == 5 and set(xs) <= set(ALLOWED), f"composition {xs}: five distinct domains of {ALLOWED}"
    return tuple(d for d in TRAINER_ORDER if d in xs)


def domains_arg(doms):
    return "numbers," + ",".join(doms)


def manifolds(doms):
    return ("numbers",) + tuple(doms)


def arms(doms):
    return ("none", "all6") + tuple(f"only-{m}" for m in manifolds(doms))


def pri_of(arm, doms):
    if arm == "none":
        return ()
    if arm == "all6":
        return manifolds(doms)
    assert arm.startswith("only-") and arm[5:] in manifolds(doms), arm
    return (arm[5:],)


def cells(doms, extra=None):
    """-> [(arm, lr)] of the 13 cells of a seed (AUDIT.md "Design" + Amendment V1); 14 with extra = '6.4e-2' (Amendment V2, opt-in:
    none@6.4e-2 right after the none grid)."""
    assert extra is None or extra in EXTRA_NONE_OK, f"--extra_none {extra}: one of {EXTRA_NONE_OK}"
    c = ([("all6", P_LR), ("all6", SAME_LR)] + [("none", lr) for lr in NONE_LRS + ((extra,) if extra else ())]
         + [(f"only-{m}", SAME_LR) for m in manifolds(doms)])
    nc = 13 + bool(extra)
    assert len(c) == nc and len(set(c)) == nc
    return c


def parse_alloc(words):
    """'b200:16,17 l40s:18,19,20 a100:21,22,23' (or a list of such words) -> {seed: device}; every seed of SEEDS exactly once."""
    words = words.split() if isinstance(words, str) else words
    al = {}
    for w in words:
        dev, ss = w.split(":")
        assert dev in HW6, f"--alloc device {dev}: one of {sorted(HW6)}"
        for s in ss.split(","):
            assert int(s) not in al, f"seed {s} allocated twice"
            al[int(s)] = dev
    assert sorted(al) == list(SEEDS), f"--alloc covers seeds {sorted(al)} != {list(SEEDS)}"
    return al


def opt(name, default=None):
    i = sys.argv.index(name) if name in sys.argv else None
    return sys.argv[i + 1] if i is not None and i + 1 < len(sys.argv) else default


def opt_list(name):
    i = sys.argv.index(name) if name in sys.argv else None
    if i is None:
        return None
    out = []
    for w in sys.argv[i + 1:]:
        if w.startswith("--"):
            break
        out.append(w)
    return out


# ---------------------------------------------------------------- names
def tag_v2(arm, lr):
    return f"{PREFIX}-{arm}-lr{lr}"


def name_v2(arm, lr, seed, rr=0, canary=False):
    return (f"{PREFIX}-canary-" if canary else f"{PREFIX}-") + f"{arm}-lr{lr}-s{seed}" + (f"-r{rr}" if rr else "")


def slot_name(slot):
    return f"llmadd_{PREFIX}-b200-slot{slot}"


# ---------------------------------------------------------------- command line, intent, input checks
def arm_flags(arm, doms):
    pri = pri_of(arm, doms)
    if not pri:
        return "--target none"
    return f"--target {'helix3' if 'numbers' in pri else 'none'} {MJ.W} --prior_domains {','.join(pri)} --lam 1"


def want_flags(arm, doms):
    pri = pri_of(arm, doms)
    if not pri:
        return {"--target": "none", "--schedule": None, "--window": None, "--site": None, "--lam_rule": None, "--prior_domains": None,
                "--prior_shuf": None, "--lam": None}
    return {"--target": "helix3" if "numbers" in pri else "none", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx",
            "--lam_rule": "each", "--prior_domains": ",".join(pri), "--prior_shuf": None, "--lam": "1"}


def args_raw(doms, arm, lr, seed, out, canary=False, tag=None):
    """pt6comb's trainer command line (make_jobs_pt6comb.args_of6) for any domain list (the refs' source mixes use it too)."""
    tag = tag or tag_v2(arm, lr)
    ck = MJ.PILOT_CKPT if canary else CKPT
    return (f"--data {MJ.DATA} --tokenizer {MJ.TOK} --tokenization whole --out {out}/{tag} {arm_flags(arm, doms)} --arith_lines 1 "
            f"--domains {domains_arg(doms)} --extra_lines {K} --lr {lr} --seed {seed} --tag _{tag} --ckpt_every {ck}"
            + (f" --tokens {32 * 512 * MJ.PILOT_STEPS} --eval_every {MJ.PILOT_EVAL}" if canary else ""))


def has3(doms):
    return any(d in TOY3 for d in doms)


def intent_v2(P, doms, arm, lr, seed, ref, dev, canary=False):
    """make_jobs_pt6comb.intent6 for any composition (key order kept) + ptdom3's toy3 facts (readset, Rev H dep_sha) + the device."""
    pri = pri_of(arm, doms)
    steps, ev_ = (MJ.PILOT_STEPS, MJ.PILOT_EVAL) if canary else (P.STEPS, P.EVAL_EVERY)
    events, on_frac = P.window_events(steps) if pri else ([], 0.0)
    if pri:
        assert events == ([[20, 1], [160, 0]] if canary else [[916, 1], [7324, 0]]), events
    lam_rule = "each" if pri else "norm"
    hw = HW6[dev]
    I = {"batch": KEY, "code_sha": WANT_SHA["pretrain_range.py"], "dep_sha": {f: WANT_SHA[f] for f in (M3.DEP if has3(doms) else MJ.DEP)},
         "steps": steps, "n_params": P.N_PARAMS, "grad_ckpt_on": False, "geo_layers": [2, 4], "read_layers": [2, 4], "lm_chunk": 0,
         "eval_every_steps": ev_, "lam_rule": lam_rule, "save_model": False, "pilot": bool(canary),
         "lam_domains": {d: (1.0 if d in pri else 0.0) for d in manifolds(doms)}, "K": K, "ext": list(doms),
         "comp": [d for d in doms if d in MJ.COMP], "kstep": [d for d in doms if d in MJ.KSTEP],
         "toy": [d for d in doms if d in TOY or d in TOY3]}
    if has3(doms):
        I["readset"] = {d: M3.READSET[d] for d in doms if d in TOY3}
    I.update({"tables": [d for d in doms if d in MJ.ALL24 and d != "months"], "active": list(pri), "geo_tol": MJ.GEO_TOL,
              "evals": len(range(0, steps, ev_)) + ((steps - 1) % ev_ != 0), "events": events,
              "on_frac": on_frac, "lm0_ref": ref.get("lm0"), "data_sha0": ref.get("dsha0"), "dom": ref.get("dom"),
              "table_sha": ref.get("table_sha"), "n_fmt": {d: M6.N_FMT[d] for d in doms if d in M6.N_FMT}, "cka_rise": MJ.CKA_RISE,
              "dp_tol0": MJ.DP_TOL0, "dp_tol": {d: MJ.DP_PER_TOK * MJ.SPAN_TOK[d] for d in doms if d in MJ.ALL24},
              "numbers_share": [0.019, 0.024], "read": {d: READ_V2[d] for d in doms}, "seen": {d: SEEN_V2[d] for d in doms},
              "gpu": hw["gpu"],
              "args": {"data": None, "tokenizer": str(MJ.TOK), "out": None, "tokenization": "whole",
                       "target": "helix3" if "numbers" in pri else "none", "site": "ctx", "schedule": "window" if pri else "whole",
                       "window": "0.05,0.4", "arith_lines": 1, "lr": float(lr), "lam": 1.0 if pri else 5.0,
                       "tokens": float(32 * 512 * MJ.PILOT_STEPS) if canary else 3e8,
                       "bs": P.BS, "seq": P.SEQ, "d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536, "seed": seed,
                       "tag": f"_{tag_v2(arm, lr)}", "save_model": False, "ckpt_every": MJ.PILOT_CKPT if canary else CKPT,
                       "geo_layers": None, "eval_every": ev_, "n_eval": 150, "grad_ckpt": "auto", "lm_chunk_tokens": -1,
                       "untie": False, "gate_thr": 0.5, "gate_start": 0.0, "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256,
                       "stop_after": -1, "dry_inputs": False, "domains": domains_arg(doms), "extra_lines": K,
                       "prior_domains": ",".join(pri), "prior_shuf": "", "lam_rule": lam_rule, "cyc_held": 0.4, "cyc_heldE": 0.5,
                       "s4_train": 0.6, "dim_train": 0.5, "dom_eval_rows": 200, "dom_seen": 150},
              "device": dev})
    return I


def dry_block_v2(doms, arm):
    """make_jobs_pt6comb.dry_block6 for any composition (text identical; tests/regress_pt6comb.py asserts it for B) + ptdom3's
    read-out-set grep for the toy3 domains."""
    pri = pri_of(arm, doms)
    tgt = "helix3" if "numbers" in pri else "none"
    ta = ("""grep -qxF 'target features (1000, 10)' "$OUT/dry_inputs.txt" || { echo "## not a helix3 table (1000 x 10) (this arm's numbers prior)"; exit 3; }"""
          if tgt == "helix3" else
          """! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a numbers prior target (this arm has none)"; exit 3; }""")
    specs = " ".join(f"{d}:{str(d in pri).lower()}:false:{json.dumps(1.0 if d in pri else 0.0)}" for d in manifolds(doms))
    txt = (ta + """
dl=$(grep -m1 '^domains: ' "$OUT/dry_inputs.txt" || true)
has() { grep -qF -e "$1," -e "$1}" <<< "$blk"; }
for sp in """ + specs + """; do
  IFS=: read -r d p s l <<< "$sp"
  blk=$(grep -oE "\\"$d\\": \\{[^}]*\\}" <<< "$dl" || true)
  has "\\"prior\\": $p" && has "\\"shuf\\": $s" && has "\\"lam\\": $l" ||
    { echo "## domain $d not prior $p / shuf $s / lam $l in the domains: line"; exit 3; }
done
grep -qF '"numbers": {"target": \"""" + tgt + """\", ' <<< "$dl" || { echo "## numbers target not """ + tgt + """"; exit 3; }""")
    t3 = [d for d in doms if d in TOY3]
    if t3:
        txt += ("""
for d in """ + " ".join(f"{d}:{M3.READSET[d]}" for d in t3) + """; do
  IFS=: read -r d s <<< "$d"
  grep -qE "^\\[$d\\] \\{.*\\"readout\\": \\"$s\\"" "$OUT/dry_inputs.txt" || { echo "## $d read-out set is not $s"; exit 3; }
done""")
    return txt


# ---------------------------------------------------------------- rendering through make_jobs_ptdom.render
@contextmanager
def patched(doms, dev, canary=False, seeds=SEEDS):
    names = ("R", "WANT_SHA", "HW", "tag_of", "job_name", "slurm_name", "args_of", "intent", "want_flags", "dry_block", "arms",
             "saves", "STAGES", "SEL", "CONF", "SLOT0", "SLOTS", "CHECK_PY", "dom_kind", "DOM_KEYS")
    keep = {k: getattr(MJ, k) for k in names}
    assert KEY not in MJ.BATCHES and KEY not in MJ.HWOF
    try:
        MJ.BATCHES[KEY] = tuple(doms)
        MJ.HWOF[KEY] = dev
        MJ.R, MJ.WANT_SHA, MJ.HW = R, WANT_SHA, HW6
        MJ.tag_of = lambda b_, a_, l_: tag_v2(a_, l_)
        MJ.job_name = lambda b_, a_, l_, s_, rr=0, pilot=False: name_v2(a_, l_, s_, rr, canary)
        MJ.slurm_name = lambda b_, a_, l_, s_, rr=0, slot=None, pilot=False: (
            slot_name(slot) if (dev == "b200" and not canary) else f"llmadd_{name_v2(a_, l_, s_, rr, canary)}")
        MJ.args_of = lambda b_, a_, l_, s_, out, save=True, pilot=False: args_raw(doms, a_, l_, s_, out, canary)
        MJ.intent = lambda P_, b_, a_, l_, s_, ref, save=True, pilot=False: intent_v2(P_, doms, a_, l_, s_, ref, dev, canary)
        MJ.want_flags = lambda a_: want_flags(a_, doms)
        MJ.dry_block = lambda b_, a_: dry_block_v2(doms, a_)
        MJ.arms = lambda b_: arms(doms)
        MJ.saves = lambda stage, pilot=False: False
        MJ.STAGES = ("V", "canary")
        MJ.SEL, MJ.CONF = (), tuple(seeds)
        MJ.SLOT0, MJ.SLOTS = {KEY: 0}, N_SLOTS
        MJ.CHECK_PY = M3.CHECK_PY_3 if has3(doms) else M6.CHECK_PY_6
        MJ.dom_kind = dom_kind
        MJ.DOM_KEYS = DOM_KEYS_V2
        yield KEY
    finally:
        for k, v in keep.items():
            setattr(MJ, k, v)
        MJ.BATCHES.pop(KEY, None)
        MJ.HWOF.pop(KEY, None)


def desc_v2(arm, doms):
    pri = pri_of(arm, doms)
    if not pri:
        return "none (no prior)"
    if arm == "all6":
        return "all6: the six priors " + " + ".join(pri) + " (lambda 1 each, window (0.05, 0.4), ctx; numbers = helix3 on the operands)"
    return f"only the {pri[0]} prior (lambda 1, window (0.05, 0.4), ctx" + ("; helix3 on the operands)" if pri[0] == "numbers" else ")")


RO_FILE = R / "readout_pt6v2.py"


def render_v2(P, doms, arm, lr, seed, ref, commit, dev, slot=None, rr=0, canary=False, seeds=SEEDS):
    """-> (name, tag, text, args, intent) of one pt6v2 cell: make_jobs_ptdom.render + counted edits."""
    stage = "canary" if canary else "V"
    assert arm in arms(doms) and lr in LADDER and dev in HW6 and (canary or seed in seeds)
    slot_ = slot if (dev == "b200" and not canary) else None
    assert (slot_ is not None) == (dev == "b200" and not canary), (dev, slot)
    with patched(doms, dev, canary, seeds) as key:
        desc0 = (f"only the {arm[5:]} prior (lambda 1, window (0.05, 0.4), ctx)" if arm != "none" else "none (no prior)")
        hw = HW6[dev]
        name, tag, text, args, I = MJ.render(P, key, arm, lr, seed, ref, commit, rr, slot_, stage, canary)
    assert name == name_v2(arm, lr, seed, rr, canary) and tag == tag_v2(arm, lr)
    n5 = len([d for d in doms if d != "months"])                       # make_jobs_ptdom.render's n_site
    n_site = len([d for d in doms if d in MJ.ALL24 and d != "months"])  # the domains that print a site-rule line
    text = _edit(text, f"# ptdom A3 / A4 batch {key} stage {'pilot' if canary else stage}: {desc0} @ lr",
                 f"# pt6v2 {'PIPELINE CANARY' if canary else 'main'} (pt6comb's C job body on Rev H), composition {domains_arg(doms)}, "
                 f"{dev.upper()}: {desc_v2(arm, doms)} @ lr")
    text = _edit(text, "; code = runs/ptdom/code (COMMIT", "; code = runs/pt6v2/code (COMMIT")
    text = _edit(text, "# spec and pre-registered screen / verdict runs/ptdom/AUDIT.md (Amendments A3, A4); generated by llmadd_torch/make_jobs_ptdom.py\n",
                 "# spec and pre-registered verdict runs/pt6v2/AUDIT.md; rendered by llmadd_torch/make_jobs_pt6v2.py through make_jobs_ptdom.render\n")
    if canary:
        text = _edit(text, f"# A4 GPU PILOT of batch {key} (not part of any result; output runs/ptdom/pilot/): {MJ.PILOT_STEPS} steps (--tokens), "
                           f"evals every {MJ.PILOT_EVAL} steps, ckpt every {MJ.PILOT_CKPT}, --save_model; every in-job check on a B200\n",
                     f"# pt6v2 PIPELINE CANARY (not part of any result; output runs/pt6v2/pilot/): {MJ.PILOT_STEPS} steps (--tokens), evals "
                     f"every {MJ.PILOT_EVAL} steps, ckpt every {MJ.PILOT_CKPT}, no saved model; every in-job check on {GPU_TXT[dev]}\n")
        if dev != "b200":
            text = _edit(text, f"#SBATCH --time={hw['time']}\n", "#SBATCH --time=01:00:00\n")
    elif dev == "b200":
        text = _edit(text, "; 16 chains over the four batches).", f"; {N_SLOTS} chains shared by every pt6v2 B200 job).")
    if dev == "a100":
        text = _edit(text, L40S_HEAD, A100_HEAD)
    hk = "canary" if canary else dev
    text = _edit(text, "# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/ptdom/HOLD (every batch) or\n"
                       f"# runs/ptdom/HOLD_{key} (A4: this batch only) exists\n"
                       f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{key}" ]; then echo "HOLD present: exiting"; exit 0; fi\n',
                 "# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/pt6v2/HOLD (every job) or\n"
                 f"# runs/pt6v2/HOLD_{hk} (the {'canary' if canary else dev.upper() + ' jobs'} only) exists\n"
                 f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{hk}" ]; then echo "HOLD present: exiting"; exit 0; fi\n')
    text = _edit(text, f'case "$GPU" in *{hw["gpu"]}*) ;; *) echo "## not a {hw["gpu"]} (batch {key}\'s GPU type, one type per batch): exiting"; exit 3;; esac\n',
                 f'case "$GPU" in {GPU_CASE[dev]}) ;; *) echo "## not {GPU_TXT[dev]}: seed {seed} is allocated to {dev.upper()} '
                 f'(one GPU type for every cell of a seed, AUDIT.md GPU types): exiting"; exit 3;; esac\n'
                 f'echo "DEVICE {dev} (seed {seed}\'s allocated GPU type)"\n')
    text = _edit(text, "--dry_inputs: model line, no numbers target, per domain",
                 "--dry_inputs: model line, the numbers target (" + ("helix3" if "numbers" in pri_of(arm, doms) else "none") + "), per domain")
    text = _edit(text, "# 4. CHECK_PY (make_jobs_ptdom.py) with this run's intent",
                 ("# 4. CHECK_PY_3 (make_jobs_ptdom3.py: ptdom's CHECK_PY + pt6comb's 7 + ptdom3's 1 counted edits)" if has3(doms) else
                  "# 4. CHECK_PY_6 (make_jobs_pt6comb.py: ptdom's CHECK_PY + counted edits)") + " with this run's intent")
    text = _edit(text, f"the site rule ({n5} domains asserted)", f"the site rule ({n_site} domains asserted: the comparisons; s4 / dim / months / "
                 "toy3: single-token sites asserted by the trainer)")
    text = _edit(text, f'"$OUT/dry_inputs.txt")" = {n5} ] || {{ echo "## the site rule is not asserted for the {n5} domains with item-name sites"; exit 3; }}',
                 f'"$OUT/dry_inputs.txt")" = {n_site} ] || {{ echo "## the site rule is not asserted for the {n_site} domains with item-name sites"; exit 3; }}')
    tgt = I["args"]["target"]
    if tgt != "none":   # pretrain_range.py run_<tokenization>_<target>_seed<s><tag>.json (pt6comb's edit)
        text = _edit(text, f'"$OUT/$TAG/run_whole_none_seed{seed}_{tag}.json"', f'"$OUT/$TAG/run_whole_{tgt}_seed{seed}_{tag}.json"')
    post = M3.STEP0_MARK if has3(doms) else ""
    if canary:
        if RO_FILE.exists():
            post += (f'[ "$(sha256sum {RO_FILE} | cut -c1-12)" = "{sha(RO_FILE)}" ] || {{ echo "## readout_pt6v2.py changed since rendering"; exit 3; }}\n'
                     f'echo "## 2b. canary read-out $(date -u +%FT%TZ)"\n'
                     f'"$PY" {RO_FILE} --canary "$OUT" | tee "$OUT/canary_readout.txt" || echo "## CANARY READ-OUT FAILED"\n')
        else:
            post += 'echo "## CANARY READ-OUT NOT RENDERED: runs/pt6v2/readout_pt6v2.py did not exist at render time"\n'
    if post:
        text = _edit(text, '"$PY" pretrain_range.py $ARGS\n', '"$PY" pretrain_range.py $ARGS\n' + post)
    for bad in ("runs/ptdom/", "runs/pt6comb", "make_jobs_ptdom.py)", f"HOLD_{key}"):
        assert bad not in text, bad
    return name, tag, text, args, I


SMOKE_DEVS = ("l40s", "a100")


def smoke_dir(dev):
    return R / "pilot" / f"smoke_{dev}"


def render_smoke(P, doms, ref, commit, dev):
    """A GPU smoke (phase 2b, 2026-11-19; not part of any result; pass rules in runs/pt6v2/AUDIT_phase2b.md): the canary's job
    (all6@2e-3 seed 16, 400 steps, every in-job check) on `dev`, its own Slurm name llmadd_pt6v2-smoke-<dev> and out dir
    runs/pt6v2/pilot/smoke_<dev>; no read-out. -> (sname, text)."""
    global RO_FILE
    a, lr, s = CANARY
    keep, RO_FILE = RO_FILE, R / "NO_READOUT_IN_A_SMOKE"      # the smoke never calls readout_pt6v2.py
    try:
        name, tag, text, args, I = render_v2(P, doms, a, lr, s, ref, commit, dev, None, 0, True)
    finally:
        RO_FILE = keep
    sname = f"pt6v2-smoke-{dev}"
    old, new = str(R / "pilot" / name), str(smoke_dir(dev))
    assert text.count(old) >= 2 and str(new) not in text, text.count(old)
    text = text.replace(f"{old}/", f"{new}/").replace(f"{old};", f"{new};").replace(f"{old} ", f"{new} ").replace(f"{old}\n", f"{new}\n")
    assert old not in text, "the canary's out dir is left in the smoke text"
    text = _edit(text, f"#SBATCH --job-name=llmadd_{name}\n", f"#SBATCH --job-name=llmadd_{sname}\n")
    text = _edit(text, "# pt6v2 PIPELINE CANARY (not part of any result; output runs/pt6v2/pilot/)",
                 f"# pt6v2 GPU SMOKE on {dev.upper()} (phase 2b: device guard + step rate; not part of any result; output "
                 f"runs/pt6v2/pilot/smoke_{dev}/)")
    text = _edit(text, 'echo "## CANARY READ-OUT NOT RENDERED: runs/pt6v2/readout_pt6v2.py did not exist at render time"\n',
                 'echo "## smoke: no read-out (runs/pt6v2/AUDIT_phase2b.md)"\n')
    text = _edit(text, f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_canary" ]; then', f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_smoke" ]; then')
    text = _edit(text, "# runs/pt6v2/HOLD_canary (the canary only) exists\n", "# runs/pt6v2/HOLD_smoke (the smokes only) exists\n")
    for bad in ("pt6v2-canary", "pilot/pt6v2-", "readout_pt6v2"):
        assert bad not in text, bad
    return sname, text


# ---------------------------------------------------------------- snapshot, references, composition
def snapshot():
    """runs/pt6v2/code = git archive of tools/ at ef7b8854 (Rev H); asserted = WANT_SHA (= the commit's files) and every file
    byte-identical to runs/ptdom3/code/tools (the validated Rev H snapshot). -> commit."""
    R.mkdir(parents=True, exist_ok=True)
    if not (R / "code").exists():
        tmp = R / "code.tmp"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        arc = subprocess.run(["git", "-C", str(REPO), "archive", COMMIT, "tools"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
        (tmp / "COMMIT").write_text(f"{COMMIT} (ptdom3 Rev H, committed, not pushed; git archive of tools/ = runs/ptdom3/code); sha256[:12]: "
                                    + ", ".join(f"{f} {sha(tmp / 'tools' / f)}" for f in WANT_SHA) + "\n")
        tmp.rename(R / "code")
    cm = (R / "code" / "COMMIT").read_text()
    assert cm.split()[0] == COMMIT
    for f, v in WANT_SHA.items():
        s_ = sha(R / "code" / "tools" / f)
        assert f"{f} {s_}" in cm and s_ == v, (f, s_, v)
        assert hashlib.sha256(subprocess.run(["git", "-C", str(REPO), "show", f"{COMMIT}:tools/{f}"], capture_output=True,
                                             check=True).stdout).hexdigest()[:12] == s_, (f, COMMIT)
    T_, T3_ = R / "code" / "tools", R3 / "code" / "tools"
    ls = lambda t: sorted(p.relative_to(t) for p in t.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert ls(T_) == ls(T3_), "runs/pt6v2/code/tools and runs/ptdom3/code/tools list different files"
    for p in ls(T_):
        assert (T_ / p).read_bytes() == (T3_ / p).read_bytes(), p
    return COMMIT


def refs_path(doms):
    return R / f"pt6v2_refs_{'-'.join(doms)}.json"


def refs_v2(doms):
    """CPU references of a composition (tests/refs_pt6v2.py table --write; written once, chmod 444): ({seed: ref}, canary ref)."""
    p = refs_path(doms)
    if not p.exists():
        return {}, {}
    j = json.loads(p.read_text())
    assert j.get("design") == PREFIX and j.get("domains") == domains_arg(doms), p
    return {int(k): v for k, v in j["seeds"].items()}, j.get("canary", {})


COMP_FILE = R / "COMPOSITION.json"
CELLS_FILE = R / "CELLS.json"   # AUDIT.md Amendment V2: the cell set of the written batch (read by readout_pt6v2.py / submit_pt6v2.sh)


def decided_comp():
    if not COMP_FILE.exists():
        return None
    j = json.loads(COMP_FILE.read_text())
    return comp_doms(j["domains"])


def load_P():
    snapshot()
    sys.path.insert(0, str(R / "code" / "tools"))
    import make_rb6_tasks as P
    assert Path(P.__file__).resolve().parent == (R / "code" / "tools").resolve()
    assert (P.BS, P.SEQ, P.STEPS, P.EVAL_EVERY, P.N_PARAMS) == (32, 512, 18310, 250, 52672128)
    return P


# ---------------------------------------------------------------- plan
def lines_per_domain(n_doms, steps=18310, bs=32):
    return steps * bs * K / n_doms


def plan(doms, alloc, extra=None):
    nd = {dev: sum(1 for s in SEEDS if alloc[s] == dev) for dev in HW6} if alloc else None
    nc = len(cells(doms, extra))
    assert nc == 13 + bool(extra)
    print(f"pt6v2 plan: composition {domains_arg(doms)}; seeds {SEEDS[0]}-{SEEDS[-1]}; {nc} cells x 8 seeds = {nc * 8} runs + 1 canary")
    for a, lr in cells(doms, extra):
        print(f"  {a:18s} @ {lr}")
    if nd:
        tot = 0.0
        for dev, n in nd.items():
            h = nc * n * PLAN_H[dev]
            tot += h
            print(f"  {dev:5s}: {n} seeds x {nc} = {nc * n:3d} runs x {PLAN_H[dev]} h = {h:5.1f} {dev.upper()}-h (limit {HW6[dev]['time']})")
        print(f"  total {tot:.1f} GPU-h; canary ~0.1 h on its device")
    else:
        for dev in HW6:
            print(f"  per seed on {dev}: {nc} runs x {PLAN_H[dev]} h = {nc * PLAN_H[dev]:.1f} {dev.upper()}-h (limit {HW6[dev]['time']})")
    import glob
    import os
    szs = []
    for d in sorted(glob.glob(str(B / "runs/pt6comb/out/pt6comb-B-*-s1[0-5]")))[:6]:
        szs.append(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs))
    if szs:
        m = max(szs)
        print(f"  disk: a finished pt6comb C out dir = {min(szs) / 1e6:.0f}-{m / 1e6:.0f} MB (n {len(szs)}) -> {nc * 8 + 1} dirs <= {(nc * 8 + 1) * m / 1e9:.1f} GB "
              f"(ckpt files while running)")
    L = lines_per_domain(5)
    dref = R / "pt6v2_domrefs.json"
    dj = json.loads(dref.read_text())["domains"] if dref.exists() else {}
    print(f"  exposures (descriptive; FLAG outside 0.67x-1.5x of the source batch): domain lines per run {L:,.0f} (5 domains)")
    for d in doms:
        src = SOURCE[d]
        Ls = lines_per_domain(len(SOURCE_DOMS[src]))
        nt = (dj.get(d, {}).get(str(SEEDS[0])) or {}).get("dom", {}).get("n_train")
        r = L / Ls
        print(f"    {d:11s} source {src:10s} ({len(SOURCE_DOMS[src])} domains: {Ls:,.0f} lines) ratio {r:.2f}x"
              + (f"; ~{L / nt:,.0f} lines per training fact (n_train {nt}, seed {SEEDS[0]})" if nt else "")
              + ("  FLAG" if not 0.67 <= r <= 1.5 else ""))


# ---------------------------------------------------------------- main
def main():
    comp_arg = opt("--comp")
    if "--set_comp" in sys.argv:
        doms = comp_doms(opt("--set_comp"))
        by = opt("--by")
        assert by, "--set_comp needs --by '<who decided, date>'"
        assert not COMP_FILE.exists(), f"{COMP_FILE} exists (written once): {COMP_FILE.read_text()}"
        COMP_FILE.write_text(json.dumps({"design": PREFIX, "domains": list(doms), "domains_arg": domains_arg(doms), "by": by}, indent=1) + "\n")
        COMP_FILE.chmod(0o444)
        print("written", COMP_FILE, domains_arg(doms))
        return
    dec = decided_comp()
    doms = comp_doms(comp_arg) if comp_arg else (dec or PLACEHOLDER)
    al_w = opt_list("--alloc")
    alloc = parse_alloc(al_w) if al_w else None
    extra = opt("--extra_none")
    assert ("--extra_none" in sys.argv) == (extra in EXTRA_NONE_OK), f"--extra_none {extra}: one of {EXTRA_NONE_OK} (AUDIT.md Amendment V2)"
    if extra:
        print(f"extra_none: {extra} (AUDIT.md Amendment V2: none@{extra} for all 8 seeds)")
    if "--plan" in sys.argv:
        plan(doms, alloc, extra)
        return
    if "--snapshot" in sys.argv:
        print("snapshot", snapshot(), (R / "code" / "COMMIT").read_text().strip())
        return
    if "--smoke" in sys.argv:   # phase 2b GPU smokes: placeholder (or --comp) composition, never the batch; no --alloc
        devs = opt("--smoke").split(",")
        assert devs and set(devs) <= set(SMOKE_DEVS), f"--smoke {devs}: of {SMOKE_DEVS}"
        assert not alloc and "--fake_refs" not in sys.argv
        commit = snapshot()
        P = load_P()
        _, RC = refs_v2(doms)
        assert RC, f"no CPU reference for the canary of {domains_arg(doms)} (tests/refs_pt6v2.py table --write first)"
        for dev in devs:
            sname, text = render_smoke(P, doms, RC, commit, dev)
            job, flag = R / "pilot" / f"job_smoke_{dev}.sbatch", R / "pilot" / f"SUBMITTED_smoke_{dev}"
            assert subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0, f"{sname}: bash -n"
            if "--write" in sys.argv or "--submit" in sys.argv:
                if flag.exists():
                    assert job.read_text() == text, f"{job} was submitted and differs from this rendering"
                else:
                    job.parent.mkdir(parents=True, exist_ok=True)
                    job.write_text(text)
            if "--print" in sys.argv:
                print(text)
            if "--test-only" in sys.argv:
                o = subprocess.run(["sbatch", "--test-only", str(job)], capture_output=True, text=True)
                print(sname, (o.stdout + o.stderr).strip())
            if "--submit" in sys.argv:
                if flag.exists():
                    print(f"{sname}: refused (SUBMITTED_smoke_{dev} exists)")
                    continue
                o = subprocess.run(["sbatch", "--parsable", str(job)], capture_output=True, text=True)
                print(sname, (o.stdout + o.stderr).strip())
                if o.returncode == 0:
                    flag.write_text(o.stdout)
            print(f"smoke {dev}: {sname}, composition {domains_arg(doms)}, out {smoke_dir(dev)}, job file {job}")
        return
    assert alloc, "--alloc b200:<seeds> l40s:<seeds> a100:<seeds> (every seed 16-23 once)"
    submit = "--submit" in sys.argv
    if submit:
        assert dec is not None and doms == dec, f"--submit: only the COMPOSITION.json composition ({dec}); got {doms}"
    commit = snapshot()
    P = load_P()
    RF, RC = refs_v2(doms)
    fake = "--fake_refs" in sys.argv     # tests only: render without CPU references (never written, never submitted)
    assert not (fake and (submit or "--write" in sys.argv))
    if fake:
        f0 = {"lm0": 0.0, "dsha0": "FAKE", "dom": {d: {k: 0 for k in DOM_KEYS_V2[dom_kind(d)]} for d in doms},
              "table_sha": {d: "FAKE" for d in doms if d in MJ.ALL24 and d != "months"}, "dry": ["0" * 64, 0]}
        RF, RC = {s: f0 for s in SEEDS}, f0
    cdev = opt("--canary_dev") or "l40s"      # AUDIT.md pending decision: the canary device (default L40S, l40s_public)
    others = {p.name for p in B.glob("runs/*/out/*") if p.parent.parent != R}
    jobs = []
    a, lr, s = CANARY
    assert RC, f"no CPU reference for the canary of {domains_arg(doms)}: run runs/pt6v2/tests/run_refs_pt6v2.sh comp first"
    jobs.append(render_v2(P, doms, a, lr, s, RC, commit, cdev, None, 0, True))
    i = 0
    for s in SEEDS:
        assert s in RF, f"no CPU reference for {domains_arg(doms)} seed {s}: run runs/pt6v2/tests/run_refs_pt6v2.sh comp first"
        for a, lr in cells(doms, extra):
            dev = alloc[s]
            slot = (i % N_SLOTS) if dev == "b200" else None
            i += dev == "b200"
            name, tag, text, args, I = render_v2(P, doms, a, lr, s, RF[s], commit, dev, slot)
            assert name not in others, f"{name}: an out dir of that name exists in another batch"
            jobs.append((name, tag, text, args, I))
    nj = 8 * len(cells(doms, extra)) + 1
    assert len(jobs) == nj == (113 if extra else 105) and len({j[0] for j in jobs}) == nj   # 13 (V2: 14) x 8 + the canary
    manifest = {"design": PREFIX, "domains_arg": domains_arg(doms), "none_lrs": list(NONE_LRS) + ([extra] if extra else []),
                "extra_none": extra, "cells": [list(c_) for c_ in cells(doms, extra)], "n_runs": nj - 1, "seeds": list(SEEDS),
                "canary": list(CANARY)}
    subm = sorted(p_.name for p_ in R.glob(f"SUBMITTED_{PREFIX}-*"))
    m_disk = json.loads(CELLS_FILE.read_text()) if CELLS_FILE.exists() else None
    m_state = "missing" if m_disk is None else ("same" if m_disk == manifest else "differ")
    if "--write" in sys.argv and m_state != "same":
        assert not subm, f"{CELLS_FILE} is {m_state} but {len(subm)} jobs were submitted: the cell set is frozen once a job is submitted"
        if CELLS_FILE.exists():
            CELLS_FILE.chmod(0o644)
        CELLS_FILE.write_text(json.dumps(manifest, indent=1) + "\n")
        CELLS_FILE.chmod(0o444)
        print(f"written {CELLS_FILE}: {len(manifest['cells'])} cells x 8 seeds, none grid {manifest['none_lrs']}")
        m_state = "same"
    written = []
    for name, tag, text, args, I in jobs:
        job, flag = R / f"job_{name}.sbatch", R / f"SUBMITTED_{name}"
        if flag.exists():
            if job.read_text() != text:
                print(f"WARNING: {job.name} was submitted and differs from this rendering: kept, not rewritten")
        elif "--write" in sys.argv:
            job.write_text(text)
            written.append(job)
    if "--check" in sys.argv:
        for name, tag, text, args, I in jobs:
            assert subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0, f"{name}: bash -n"
        print(f"check: {len(jobs)} job texts parse (bash -n); every render assertion of make_jobs_ptdom.render held")
    if "--print" in sys.argv:
        for name, tag, text, args, I in jobs:
            print(f"{name} [{I['device']}]{' (SUBMITTED)' if (R / f'SUBMITTED_{name}').exists() else ''}: {args}")
    if "--disk" in sys.argv:   # phase 2b (submit_pt6v2.sh): the on-disk job files of the unsubmitted jobs vs this rendering
        st_ = {"same": 0, "missing": 0, "differ": 0}
        for name, tag, text, args, I in jobs:
            job = R / f"job_{name}.sbatch"
            if not (R / f"SUBMITTED_{name}").exists():
                st_["missing" if not job.exists() else ("same" if job.read_text() == text else "differ")] += 1
        print(f"disk: unsubmitted job files {st_}")
        print(f"cells manifest: {m_state} ({CELLS_FILE.name}; {len(manifest['cells'])} cells x 8 seeds, none grid {manifest['none_lrs']})")
    if "--test-only" in sys.argv:
        n_t = int(opt("--test-only")) if str(opt("--test-only")).isdigit() else 0
        for name, *_ in jobs[: n_t or len(jobs)]:
            o = subprocess.run(["sbatch", "--test-only", str(R / f"job_{name}.sbatch")], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
    if submit:
        assert RO_FILE.exists(), "--submit: runs/pt6v2/readout_pt6v2.py must exist (the canary calls it)"
        assert m_state == "same", f"--submit: {CELLS_FILE} is {m_state} (run --check --write with the same --extra_none first)"
        for h_ in [R / "HOLD"] + [R / f"HOLD_{d}" for d in sorted(set(alloc.values()))] + [R / "HOLD_canary"]:
            assert not h_.exists(), f"{h_} exists: remove it to submit (the user decides)"
        (R / "ALLOC.json").write_text(json.dumps({str(k): v for k, v in sorted(alloc.items())}) + "\n") if not (R / "ALLOC.json").exists() else None
        assert json.loads((R / "ALLOC.json").read_text()) == {str(k): v for k, v in sorted(alloc.items())}, "--alloc differs from ALLOC.json"
        stale = [n_ for n_, t_, text_, *_ in jobs if not (R / f"SUBMITTED_{n_}").exists()
                 and (not (R / f"job_{n_}.sbatch").exists() or (R / f"job_{n_}.sbatch").read_text() != text_)]
        assert not stale, f"--submit: {len(stale)} job files missing or differing from this rendering (run --check --write first): {stale[:3]}"
        for name, *_ in jobs:
            flag = R / f"SUBMITTED_{name}"
            if flag.exists():
                print(f"{name}: refused (SUBMITTED_{name} exists: a name is submitted once)")
                continue
            o = subprocess.run(["sbatch", str(R / f"job_{name}.sbatch")], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
    print(f"pt6v2 composition {domains_arg(doms)}{' (PLACEHOLDER)' if doms == PLACEHOLDER and dec is None else ''}: {len(jobs)} jobs rendered "
          f"({len(written)} written); code {commit}, trainer {WANT_SHA['pretrain_range.py']}; alloc "
          f"{ {d: [s for s in SEEDS if alloc[s] == d] for d in HW6} }; canary on {cdev}; submitted: {'yes' if submit else 'no'}")


if __name__ == "__main__":
    main()
