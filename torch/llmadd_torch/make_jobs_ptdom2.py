"""Torch jobs of runs/ptdom2 (pre-registered in runs/ptdom2/AUDIT.md; user-approved design of 2026-11-08 relayed by the
LLM_addition coordinator). ptdom2 re-uses ptdom's job body exactly: every job is rendered by make_jobs_ptdom.render (the frozen
generator, imported read-only, sha256[:12] asserted) with a handful of module names re-pointed for the duration of one call
(run / job / slot names, out dir, seeds 8-15, the --prior_shuf flag of a shuffled-target arm); the text is then edited by
exact, counted replacements of the comment lines that name runs/ptdom and of the HOLD line (HOLD_<part> instead of HOLD_<batch>).
A shuffled-target arm (only-<d>-shuf) differs from its true-target twin only in --prior_shuf <d> and in CHECK_PY_SHUF (= ptdom's
CHECK_PY with exact, counted edits: domains.<d>.shuf, the shuffled-target facts, the lambda check of a shuffled prior, the
CKA-rise bar on the SHUFFLED target); every other arm gets ptdom's CHECK_PY byte for byte.

Parts (AUDIT.md "Design"):
  S    (A2 / A3) F100's four PASS domains on FRESH seeds 8-15, lr 2e-3: only-<d>, only-<d>-shuf, and none at F100's none pick
       (countries, people: none@2e-3, batch B1; cities, months: none@4e-3, batch A) + A's none@2e-3 (same-lr lines) = 88 runs
  S0   OPTIONAL, descriptive only (A2.1): only-<d>-shuf@2e-3 at seeds 0-7; --submit refuses it without --allow_s0 (A3.3)
  R1   re-screen of six near-miss domains, lr selection: only-<d>@4e-3 at seeds 0-3 (stars, landmarks: A; compass,
       prefixes, ranks: B2; scalar_adj: B3); ptdom's 1e-3 / 2e-3 (ranks also 5e-4) seeds 0-3 are re-used        24 runs
  R1x  the conditional extensions decided by select_ptdom2.py --after R1 (prior 8e-3 where 4e-3 is the argmax; none one
       step (x2) above its batch's highest ptdom lr where that is an R domain's none argmax), seeds 0-3        0-36 runs
  R2   confirmation on fresh seeds 8-15 at the final picks: only-<d>@P, only-<d>-shuf@P, none@N (shared per batch and lr)
                                                                                                        0-144 runs
  pilot  ONE B200 job (llmadd_ptdom2-pilot): only-compass-shuf@2e-3 (B2) and only-countries-shuf@2e-3 (B1) at seed 0 with
       --stop_after 2500 (11 evals, 7 of them in the prior window) -> runs/ptdom2/pilot/ (not part of any result)
Run / out-dir name: ptdom2-<batch>-<arm>-lr<lr>-s<seed>[-r<k>]; Slurm: llmadd_ptdom2-b200-slot<0..N_SLOTS-1> (singleton
chains, pt300x's / ptdom's B200 protocol); logs llmadd_torch/logs/llmadd_<run>_<jobid>.out.
    python make_jobs_ptdom2.py --plan
    python make_jobs_ptdom2.py --snapshot
    python make_jobs_ptdom2.py --part S|R1 [--check] [--print] [--test-only N] [--submit] [--no-write]
    python make_jobs_ptdom2.py --part R1x|R2 --picks runs/ptdom2/picks_R_after<k>.json [--check] [--print] [--test-only N] [--submit]
    python make_jobs_ptdom2.py --pilot [--check] [--print] [--test-only] [--submit]
    python make_jobs_ptdom2.py --part X --rerun [--print] [--submit]      (ptdom's A2 re-run path: <name>-r<k>)
    (--stage S|R1|R1x|R2 is accepted as an alias of --part)
Never submits without --submit; --submit refuses while runs/ptdom2/HOLD or HOLD_<part> exists and never re-submits a name.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "ptdom2"
R0 = B / "runs" / "ptdom"                      # the frozen ptdom batch: read only
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
MJ_FILE, MJ_SHA = B / "make_jobs_ptdom.py", "6061b299129c"
assert sha(MJ_FILE) == MJ_SHA, f"make_jobs_ptdom.py changed: {sha(MJ_FILE)} != {MJ_SHA}"
sys.path.insert(0, str(B))
import make_jobs_ptdom as MJ   # noqa: E402  (read only: render, intent, CHECK_PY, dry_block, constants)
assert Path(MJ.__file__).resolve() == MJ_FILE.resolve()

COMMIT = "11f61b9"                              # the ptdom trainer commit (Rev G); runs/ptdom2/code = git archive of its tools/
WANT_SHA, PY, BATCHES, KSTEP, COMP, HWOF, HW, LADDER = MJ.WANT_SHA, MJ.PY, MJ.BATCHES, MJ.KSTEP, MJ.COMP, MJ.HWOF, MJ.HW, MJ.LADDER
assert WANT_SHA["pretrain_range.py"] == "d87a1502cee8" and all(v == "b200" for v in HWOF.values())
PREFIX = "ptdom2"
PARTS = ("S", "S0", "R1", "R1pre", "R1x", "R2", "pilot")
# ---------------------------------------------------------------- design (AUDIT.md "Design"; the user's 2026-11-08 decision)
R_DOMS = {"stars": "A", "landmarks": "A", "compass": "B2", "prefixes": "B2", "ranks": "B2", "scalar_adj": "B3"}
S_DOMS = {"countries": "B1", "people": "B1", "cities": "A", "months": "A"}
S_LR = "2e-3"                                    # F100's pick of all four S domains (prior 2e-3)
# A3.1 (audit_a2 blocking 1): F100's none pick per S domain (runs/ptdom/readout.json, asserted by readout_ptdom2): the PASS
# contrast of Part S is against it; A's none@2e-3 stays (shuffled - none at the same lr, the descriptive same-lr PASS line)
S_NONE = {"countries": "2e-3", "people": "2e-3", "cities": "4e-3", "months": "4e-3"}
# the lrs ptdom ran (seeds 0-3 at least), re-used by R1's selection; select_ptdom2.py asserts that exactly these exist on disk
PRIOR_LRS0 = {"stars": ("1e-3", "2e-3"), "landmarks": ("1e-3", "2e-3"), "compass": ("1e-3", "2e-3"),
              "prefixes": ("1e-3", "2e-3"), "ranks": ("5e-4", "1e-3", "2e-3"), "scalar_adj": ("1e-3", "2e-3")}
NONE_LRS0 = {"A": ("1e-3", "2e-3", "4e-3", "8e-3"), "B2": ("1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2"), "B3": ("2e-3", "4e-3", "8e-3")}
R1_LR = "4e-3"                                   # added to every R prior grid
R1X_PRIOR = "8e-3"                               # once, where 4e-3 is the prior's area argmax
step_up = lambda lr: LADDER[LADDER.index(lr) + 1]
step_down = lambda lr: LADDER[LADDER.index(lr) - 1]
NONE_EXT = {b: step_up(max(v, key=float)) for b, v in NONE_LRS0.items()}   # once, where the batch's top lr is an R domain's none pick / argmax
NONE_LOW = {b: step_down(min(v, key=float)) for b, v in NONE_LRS0.items()}   # A1.1a: once, where the lowest is (pick or argmax, A1.2)
assert NONE_EXT == {"A": "1.6e-2", "B2": "3.2e-2", "B3": "1.6e-2"} and step_up(R1_LR) == R1X_PRIOR
assert NONE_LOW == {"A": "5e-4", "B2": "5e-4", "B3": "1e-3"}
# A1.10: the extensions the pre-registered rule already asks from ptdom's data ALONE (none units only: stars' / landmarks' none
# pick and argmax 8e-3 = A's top; prefixes' none pick and argmax 1e-3 = B2's lowest), run with R1 at seeds 0-3 (part R1pre);
# readout_ptdom2.decide_R re-derives them (none_asks_from_ptdom) and refuses if they differ
R1PRE = (("A", "none", "1.6e-2"), ("B2", "none", "5e-4"))
SEL = (0, 1, 2, 3)                               # selection seeds (ptdom's stage-1 seeds)
S_SEEDS = tuple(range(8))                        # A2: only the optional, descriptive seeds 0-7 pairing (part S0)
S_SEEDS2 = tuple(range(8, 16))                   # A2 (blocking 1): Part S's verdict seeds (fresh: F100 selected seeds 0-7 of the true arm)
R2_SEEDS = tuple(range(8, 16))                   # Part R2: fresh seeds, never used by any selection
# A1.9: every part runs ptdom's stage-1 body (no saved model, no offline LM; the bpb tie-break uses the in-training bpb)
SAVE = {"S": False, "S0": False, "R1": False, "R1pre": False, "R1x": False, "R2": False, "pilot": False}
N_SLOTS = 8                                      # singleton B200 chains llmadd_ptdom2-b200-slot0..7, shared by every part
PILOT = (("B2", "only-compass-shuf", "2e-3", 0), ("B1", "only-countries-shuf", "2e-3", 0))
PILOT_STOP = 2500                                # evals 0, 250, .., 2500 (11); on-window evals 1000 .. 2500 (7; window 916..7323)
# plan hours per run on B200 (median wall time of ptdom's jobs, '## 1. inputs' to '## done', 52-60 jobs per batch and kind;
# runs/ptdom2/tests/walltime.out) rounded up; ptdom's own plan basis was 0.6 B200-h (the upper column)
PLAN_H = {"A": 0.46, "B1": 0.35, "B2": 0.39, "B3": 0.37}   # A1.9: ptdom's stage-1 (no-save) medians
PLAN_H_PTDOM = 0.6
assert set(R_DOMS) | set(S_DOMS) <= set(MJ.COMP) | {"months"} and not set(R_DOMS) & set(S_DOMS)
assert all(d in BATCHES[b] for d, b in list(R_DOMS.items()) + list(S_DOMS.items()))


def opt(name, default=None):
    i = sys.argv.index(name) if name in sys.argv else None
    return sys.argv[i + 1] if i is not None and i + 1 < len(sys.argv) else default


# ---------------------------------------------------------------- arms and names
def is_shuf(arm):
    return arm.endswith("-shuf")


def base(arm):
    """only-<d>-shuf -> only-<d>; other arms unchanged."""
    return arm[: -len("-shuf")] if is_shuf(arm) else arm


def dom(arm):
    return None if arm == "none" else base(arm)[5:]


def arms2(batch):
    return ORIG["arms"](batch) + tuple(f"only-{d}-shuf" for d in BATCHES[batch])


def tag2(batch, arm, lr):
    return f"{PREFIX}-{batch}-{arm}-lr{lr}"


def name2(batch, arm, lr, seed, rr=0, pilot=False):
    return (f"{PREFIX}-pilot-" if pilot else f"{PREFIX}-") + f"{batch}-{arm}-lr{lr}-s{seed}" + (f"-r{rr}" if rr else "")


def slot_name(slot):
    return f"llmadd_{PREFIX}-b200-slot{slot}"


# ---------------------------------------------------------------- CHECK_PY of a shuffled-target arm (exact, counted edits of ptdom's)
def _edit(txt, old, new, n=1):
    assert txt.count(old) == n, (old[:80], txt.count(old), n)
    return txt.replace(old, new)


def _check_py_shuf(cp):
    cp = _edit(cp, 'ACT, TOL = I["active"], I["geo_tol"]\n',
               'ACT, TOL = I["active"], I["geo_tol"]\n'
               'SHUF = tuple(I.get("shuf", ()))   # ptdom2: the active priors on the SHUFFLED target (--prior_shuf)\n')
    cp = _edit(cp, 'dw = {d: [d in ACT, False, I["lam_domains"][d]] for d in ALLD}\n'
                   'chk("domains.<d> prior / shuf / lam = intent (active " + (",".join(ACT) or "none") + "; numbers target none)",\n'
                   '    dp == dw and r["domains"]["numbers"].get("target") == "none" and set(r["domains"]) == set(ALLD), dp)\n',
               'dw = {d: [d in ACT, d in SHUF, I["lam_domains"][d]] for d in ALLD}\n'
               'chk("domains.<d> prior / shuf / lam = intent (active " + (",".join(ACT) or "none") + "; shuffled " + (",".join(SHUF) or "none") + "; numbers target none)",\n'
               '    dp == dw and r["domains"]["numbers"].get("target") == "none" and set(r["domains"]) == set(ALLD), dp)\n'
               'jsh_ = [x for x in str(jf.get("prior_shuf", "")).split(",") if x]\n'
               'chk("ptdom2 shuffled target: --prior_shuf of the job\'s ARGS = the intent\'s shuf (" + ",".join(SHUF) + "), a subset of the active "\n'
               '    "priors; domains.<d>.shuf_cka_to_true < 0.15 for each (the trainer\'s far permutation of the item -> target rows)",\n'
               '    jsh_ == list(SHUF) and bool(SHUF) and set(SHUF) <= set(ACT) and set(jsh_) <= set(jpri)\n'
               '    and all(isinstance(r["domains"][d].get("shuf_cka_to_true"), float) and r["domains"][d]["shuf_cka_to_true"] < 0.15 for d in SHUF),\n'
               '    {"args": jsh_, "shuf_cka_to_true": {d: r["domains"][d].get("shuf_cka_to_true") for d in SHUF}})\n')
    cp = _edit(cp, 'cka_true = None\n', 'cka_true = cka_shuf = None\n')
    cp = _edit(cp, '               and e.get(f"gnorm_true_{d}")] for d in ACT}\n'
                   '    chk("gnorm_prior_<d> / gnorm_true_<d> = the lambda of the job\'s ARGS (rel. 1e-3) at every on-eval (the lambda that weighted "\n'
                   '        "the prior\'s gradient; with the on-window decomposition check: the lambda of the backward)",\n'
                   '        all(vs and want_lam[d] > 0 and all(abs(v / want_lam[d] - 1) < 1e-3 for v in vs) for d, vs in lr_.items()),\n'
                   '        {d: [round(min(vs), 5), round(max(vs), 5), len(vs)] if vs else None for d, vs in lr_.items()})\n',
               '               and e.get(f"gnorm_true_{d}")] for d in ACT if d not in SHUF}\n'
               '    chk("gnorm_prior_<d> / gnorm_true_<d> = the lambda of the job\'s ARGS (rel. 1e-3) at every on-eval (the lambda that weighted "\n'
               '        "the prior\'s gradient; with the on-window decomposition check: the lambda of the backward; ptdom2: true-target priors only)",\n'
               '        all(vs and want_lam[d] > 0 and all(abs(v / want_lam[d] - 1) < 1e-3 for v in vs) for d, vs in lr_.items()),\n'
               '        {d: [round(min(vs), 5), round(max(vs), 5), len(vs)] if vs else None for d, vs in lr_.items()})\n'
               '    gs_ = {d: [e[f"gnorm_prior_{d}"] for e in on if e.get(f"gnorm_prior_{d}") is not None] for d in SHUF}\n'
               '    chk("ptdom2 shuffled prior: its weighted gradient gnorm_prior_<d> finite > 0 at every on-eval where logged and the lambda of the "\n'
               '        "job\'s ARGS > 0 (in a shuffled arm gnorm_true_<d> is the TRUE target\'s term, so the ratio is not lambda there; the lambda is "\n'
               '        "tied by the ARGS / lam_domains check and the on-window decomposition double path)",\n'
               '        all(want_lam[d] > 0 and vs and all(num_(v) and v > 0 for v in vs) for d, vs in gs_.items()),\n'
               '        {d: [min(vs), max(vs), len(vs)] if vs and all(num_(v) for v in vs) else vs[:3] for d, vs in gs_.items()})\n')
    cp = _edit(cp, '        cka_true = {d: round(we[f"cka_{d}"] - c[0][f"cka_{d}"], 4) for d in ACT}\n'
                   '        chk(f"item-state CKA to the target rises by >= {I[\'cka_rise\']} from step 0 to the window\'s last eval (step {we[\'step\']})",\n'
                   '            diverged or all(v >= I["cka_rise"] for v in cka_true.values()), cka_true)\n',
               '        cka_true = {d: round(we[f"cka_{d}"] - c[0][f"cka_{d}"], 4) for d in ACT}\n'
               '        g0_ = {d: next((e[f"geo_{d}"] for e in c if num_(e.get(f"geo_{d}"))), None) for d in SHUF}\n'
               '        cka_shuf = {d: (round(g0_[d] - we[f"geo_{d}"], 4) if g0_[d] is not None and num_(we.get(f"geo_{d}")) else None) for d in SHUF}\n'
               '        rise_ = {d: (cka_shuf[d] if d in SHUF else cka_true[d]) for d in ACT}\n'
               '        chk(f"item-state CKA to the target rises by >= {I[\'cka_rise\']} from step 0 to the window\'s last eval (step {we[\'step\']}) "\n'
               '            "(ptdom2: a shuffled prior on its SHUFFLED target = 1 - geo_<d> on the training batch from its first logged value; "\n'
               '            "the true-target eval CKA rise is info)",\n'
               '            diverged or all(v is not None and v >= I["cka_rise"] for v in rise_.values()), {"rise": rise_, "true_target_eval_cka_rise": cka_true})\n')
    cp = _edit(cp, '"cka_rise_true": cka_true,', '"cka_rise_true": cka_true, "cka_rise_shuf": cka_shuf,')
    return cp


CHECK_PY_SHUF = _check_py_shuf(MJ.CHECK_PY)
# make_jobs_ptdom's own functions, captured at import (before any re-pointing)
ORIG = {k: getattr(MJ, k) for k in ("intent", "want_flags", "dry_block", "tag_of", "arms", "args_of", "render", "CHECK_PY")}


# ---------------------------------------------------------------- rendering through make_jobs_ptdom.render
def args_of2(batch, arm, lr, seed, out, save=True, pilot=False):
    """The trainer's command line of a ptdom2 cell = ptdom's (MJ.args_of of the true-target twin) with the ptdom2 tag, plus
    --prior_shuf <d> for a shuffled arm and --stop_after for the pilot."""
    a = ORIG["args_of"](batch, base(arm), lr, seed, out, save, False)
    t0, t1 = MJ.tag_of(batch, base(arm), lr), tag2(batch, arm, lr)   # MJ.tag_of: ptdom's, or tag2 while re-pointed
    a = _edit(a, f"/{t0} ", f"/{t1} ")
    a = _edit(a, f"--tag _{t0}", f"--tag _{t1}")
    a = _edit(a, f" --prior_domains {dom(arm)} --lam 1", f" --prior_domains {dom(arm)} --prior_shuf {dom(arm)} --lam 1") if is_shuf(arm) else a
    return a + (f" --stop_after {PILOT_STOP}" if pilot else "")


def intent2(P, batch, arm, lr, seed, ref, save=True, pilot=False):
    """ptdom's intent of the true-target twin (MJ.intent) with the ptdom2 tag; a shuffled arm adds args.prior_shuf and the key
    'shuf' (read by CHECK_PY_SHUF only); the pilot adds args.stop_after (its CHECK fails exactly the truncation checks)."""
    I = ORIG["intent"](P, batch, base(arm), lr, seed, ref, save, False)
    I["args"]["tag"] = f"_{tag2(batch, arm, lr)}"
    if is_shuf(arm):
        I["args"]["prior_shuf"] = dom(arm)
        I["shuf"] = [dom(arm)]
    if pilot:
        I["args"]["stop_after"] = PILOT_STOP
        I["pilot"] = True
    return I


def want_flags2(arm, pilot=False):
    w = dict(ORIG["want_flags"](base(arm)))
    if is_shuf(arm):
        w["--prior_shuf"] = dom(arm)
    if pilot:
        w["--stop_after"] = str(PILOT_STOP)
    return w


def dry_block2(batch, arm):
    """ptdom's input-check block of the twin; a shuffled arm's domain is 'prior true / shuf true / lam 1.0'."""
    t = ORIG["dry_block"](batch, base(arm))
    return _edit(t, f" {dom(arm)}:true:false:1.0", f" {dom(arm)}:true:true:1.0") if is_shuf(arm) else t


@contextmanager
def patched(part, pilot=False, shuf=False):
    """Re-point make_jobs_ptdom's module names for one render() call; restored afterwards."""
    keep = {k: getattr(MJ, k) for k in ("R", "tag_of", "job_name", "slurm_name", "args_of", "intent", "want_flags", "arm_flags",
                                         "dry_block", "arms", "saves", "STAGES", "SEL", "CONF", "SLOT0", "SLOTS", "CHECK_PY")}
    try:
        MJ.R = R
        MJ.tag_of = tag2
        MJ.job_name = lambda b_, a_, l_, s_, rr=0, pilot=False: name2(b_, a_, l_, s_, rr, pilot_)
        MJ.slurm_name = lambda b_, a_, l_, s_, rr=0, slot=None, pilot=False: slot_name(slot)
        MJ.args_of = lambda b_, a_, l_, s_, out, save=True, pilot=False: args_of2(b_, a_, l_, s_, out, save, pilot_)
        MJ.intent = lambda P_, b_, a_, l_, s_, ref, save=True, pilot=False: intent2(P_, b_, a_, l_, s_, ref, save, pilot_)
        MJ.want_flags = lambda a_: want_flags2(a_, pilot_)
        MJ.dry_block = dry_block2
        MJ.arms = arms2
        MJ.saves = lambda stage, pilot=False: SAVE[stage]
        MJ.STAGES = PARTS
        MJ.SEL, MJ.CONF = SEL, tuple(range(4, 16))
        MJ.SLOT0, MJ.SLOTS = {b: 0 for b in BATCHES}, N_SLOTS
        MJ.CHECK_PY = CHECK_PY_SHUF if shuf else keep["CHECK_PY"]
        yield
    finally:
        for k, v in keep.items():
            setattr(MJ, k, v)


pilot_ = False   # set by render2 around a pilot body


def render2(P, part, batch, arm, lr, seed, ref, commit, rr=0, slot=0, pilot=False):
    """-> (name, tag, text, args, intent) of one ptdom2 cell, rendered by make_jobs_ptdom.render."""
    global pilot_
    assert part in PARTS and arm in arms2(batch) and (pilot or slot is not None)
    pilot_ = pilot
    try:
        with patched(part, pilot, is_shuf(arm)):
            name, tag, text, args, I = MJ.render(P, batch, arm, lr, seed, ref, commit, rr, slot, part, False)
    finally:
        pilot_ = False
    assert name == name2(batch, arm, lr, seed, rr, pilot) and tag == tag2(batch, arm, lr)
    # exact, counted edits of the comment lines that name ptdom and of the kill switch (HOLD_<part>)
    text = _edit(text, f"# ptdom A3 / A4 batch {batch} stage {part}:", f"# ptdom2 part {part} (the ptdom A3-A6.1 job body), batch {batch}:")
    text = _edit(text, "; code = runs/ptdom/code (COMMIT", "; code = runs/ptdom2/code (COMMIT")
    text = _edit(text, "# spec and pre-registered screen / verdict runs/ptdom/AUDIT.md (Amendments A3, A4); generated by llmadd_torch/make_jobs_ptdom.py\n",
                 "# spec and pre-registered verdict runs/ptdom2/AUDIT.md; rendered by llmadd_torch/make_jobs_ptdom2.py through make_jobs_ptdom.render\n")
    text = _edit(text, f"; 16 chains over the four batches).", f"; {N_SLOTS} chains shared by every ptdom2 part).")
    text = _edit(text, "# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/ptdom/HOLD (every batch) or\n"
                       f"# runs/ptdom/HOLD_{batch} (A4: this batch only) exists\n"
                       f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{batch}" ]; then echo "HOLD present: exiting"; exit 0; fi\n',
                 "# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/ptdom2/HOLD (every part) or\n"
                 f"# runs/ptdom2/HOLD_{part} (this part only) exists\n"
                 f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{part}" ]; then echo "HOLD present: exiting"; exit 0; fi\n')
    if pilot:   # the pilot's bodies live in runs/ptdom2/pilot/ (run by the pilot job with bash; their #SBATCH lines are inert)
        n_ = text.count(f"{R}/out/{name}")
        assert n_ == 2, n_   # OUT= and the --out of ARGS
        text = text.replace(f"{R}/out/{name}", f"{R}/pilot/{name}")
        args = args.replace(f"{R}/out/{name}", f"{R}/pilot/{name}")
    return name, tag, text, args, I


def twin_render(P0, part, batch, arm, lr, seed, ref0, commit, slot0):
    """The ptdom job a ptdom2 cell is compared with (tests/render_diff.py): make_jobs_ptdom.render UNPATCHED, the true-target
    twin's arm, the same lr, ptdom's stage with the same save setting (1: no model; 2: model + offline LM), a ptdom seed."""
    return MJ.render(P0, batch, base(arm), lr, seed, ref0, commit, 0, slot0, "2" if SAVE[part] else "1", False)


# ---------------------------------------------------------------- snapshot, references
def snapshot():
    """runs/ptdom2/code = git archive of tools/ at COMMIT (11f61b9, the ptdom trainer); asserted = WANT_SHA, = the commit's files
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
    assert mine == theirs, "runs/ptdom2/code/tools and runs/ptdom/code/tools list different files"
    for p in mine:
        assert (R / "code" / "tools" / p).read_bytes() == (R0 / "code" / "tools" / p).read_bytes(), p
    return COMMIT


def refs2(batch):
    """CPU references per seed: ptdom's (seeds 0-7; runs/ptdom/ptdom_refs*.json, read only) + ptdom2's (seeds 8-15;
    runs/ptdom2/ptdom2_refs_<batch>.json, tests/refs_ptdom2.py). A seed in both is refused."""
    out = dict(MJ.refs(batch))
    p = R / f"ptdom2_refs_{batch}.json"
    if p.exists():
        j = json.loads(p.read_text())
        assert j.get("batch") == batch and j.get("design") == PREFIX, (p, j.get("batch"))
        new = {int(k): v for k, v in j["seeds"].items()}
        assert not set(new) & set(out), f"{p}: seeds {sorted(set(new) & set(out))} also in ptdom's references"
        assert all(s in R2_SEEDS for s in new), sorted(new)
        out.update(new)
    return out


def load_P():
    snapshot()
    sys.path.insert(0, str(R / "code" / "tools"))
    import make_rb6_tasks as P
    assert Path(P.__file__).resolve().parent in ((R / "code" / "tools").resolve(), (R0 / "code" / "tools").resolve())
    assert (P.BS, P.SEQ, P.STEPS, P.EVAL_EVERY, P.N_PARAMS) == (32, 512, 18310, 250, 52672128)
    return P


# ---------------------------------------------------------------- cells per part
def read_picks(path, stage):
    """A picks file of select_ptdom2.py (written once, read only): design ptdom2, the stage it writes."""
    pk = json.loads(Path(path).read_text())
    assert pk.get("design") == PREFIX and pk.get("status") == "decided" and pk.get("stage_written") == stage, \
        (pk.get("design"), pk.get("status"), pk.get("stage_written"), stage)
    assert not (Path(path).stat().st_mode & 0o222), f"{path} is writable: a decision file is chmod 444"
    return pk


def cells(part, picks=None):
    """[(batch, arm, lr, seed)] of a part in submission order (seed-major, so a partial part stays paired)."""
    if part == "S":   # A2 / A3: seeds 8-15: only-d, only-d-shuf at 2e-3; none at 2e-3 (B1, A) and at F100's 4e-3 (A)
        nones = list(dict.fromkeys([(b, S_LR) for b in S_DOMS.values()] + [(S_DOMS[d], lr) for d, lr in S_NONE.items()]))
        return [x for s in S_SEEDS2 for x in [(b, a, S_LR, s) for d, b in S_DOMS.items() for a in (f"only-{d}", f"only-{d}-shuf")]
                + [(b, "none", lr, s) for b, lr in nones]]
    if part == "S0":   # A2: OPTIONAL, descriptive only (the old pairing with ptdom's selected true arm at seeds 0-7)
        return [(b, f"only-{d}-shuf", S_LR, s) for s in S_SEEDS for d, b in S_DOMS.items()]
    if part == "R1":
        return [(b, f"only-{d}", R1_LR, s) for s in SEL for d, b in R_DOMS.items()]
    if part == "R1pre":
        return [(b, a, lr, s) for s in SEL for b, a, lr in R1PRE]
    if part == "pilot":
        return list(PILOT)
    pk = read_picks(picks, part)
    if part == "R1x":
        E = [tuple(x) for x in pk["E"]]
        assert pk.get("amendment") == "A1", "an R1x picks file must carry amendment A1"
        fixb_ok = lambda b, lr: any(lr in PRIOR_LRS0[d] + (R1_LR,) for d, bb in R_DOMS.items() if bb == b)
        for b, a, lr in E:   # A1: none up / down once, fixB at an R prior lr of the batch; the prior's 8e-3; never a cell already run
            assert (a == "none" and (lr in (NONE_EXT[b], NONE_LOW[b]) or fixb_ok(b, lr)) and lr not in NONE_LRS0[b]
                    and (b, a, lr) not in R1PRE) or (a != "none" and R_DOMS.get(dom(a)) == b and lr == R1X_PRIOR and not is_shuf(a)), (b, a, lr)
        assert len(set(E)) == len(E)
        return [(b, a, lr, s) for s in SEL for b, a, lr in E]
    assert part == "R2"
    C = [tuple(x) for x in pk["C"]]
    for b, a, lr in C:
        assert lr in LADDER and (a == "none" or R_DOMS.get(dom(a)) == b), (b, a, lr)
    fin = {d: v for d, v in pk["domains"].items() if v["status"] == "FINAL"}
    want = set()
    for d, v in fin.items():
        b = R_DOMS[d]
        want |= {(b, f"only-{d}", v["pick_prior"]), (b, f"only-{d}-shuf", v["pick_prior"]), (b, "none", v["pick_none"])}
    assert set(C) == want and len(set(C)) == len(C), (sorted(set(C) ^ want))
    s_cells = set(cells("S"))   # A2: a none cell Part S already runs (A none@2e-3 at seeds 8-15) is not rendered again
    return [(b, a, lr, s) for s in R2_SEEDS for b, a, lr in C if (b, a, lr, s) not in s_cells]


def hold_paths(part):
    return R / "HOLD", R / f"HOLD_{part}"


# ---------------------------------------------------------------- plan / cost
def plan():
    h = lambda b: PLAN_H[b]
    nS, hS = len(cells("S")), sum(h(b) for b, *_ in cells("S"))
    nS0, hS0 = len(cells("S0")), sum(h(b) for b, *_ in cells("S0"))
    nR1, hR1 = len(cells("R1")), sum(h(b) for b, *_ in cells("R1"))
    nP, hP = len(cells("R1pre")), sum(h(b) for b, *_ in cells("R1pre"))
    # R1x max: every R prior at 8e-3 + per batch the none directions R1pre did not use (A down, B2 up, B3 up and down = fixB 1e-3)
    used = {(b, "up" if lr == NONE_EXT[b] else "down") for b, _, lr in R1PRE}
    xn = [(b, dr) for b in sorted(set(R_DOMS.values())) for dr in ("up", "down") if (b, dr) not in used]
    hX = sum(4 * h(b) for d, b in R_DOMS.items()) + sum(4 * h(b) for b, _ in xn)
    nX = 4 * (len(R_DOMS) + len(xn))
    batches = sorted(set(R_DOMS.values()))
    h2_prior = sum(16 * h(b) for b in R_DOMS.values())
    h2n_lo, h2n_hi = sum(8 * h(b) for b in batches), sum(8 * h(b) * sum(1 for d in R_DOMS if R_DOMS[d] == b) for b in batches)
    print(f"ptdom2 plan (53M, 3e8 tokens, B200, {N_SLOTS} singleton slot chains; A1.9: every part ptdom's stage-1 body (no model, no "
          f"offline LM); per-run hours {PLAN_H} = ptdom's measured stage-1 medians, ptdom's plan basis {PLAN_H_PTDOM}):")
    print(f"  S      {nS:3d} runs = {hS:5.1f} B200-h ({nS * PLAN_H_PTDOM:.1f} at {PLAN_H_PTDOM}) (A2 / A3: seeds 8-15; only-d, only-d-shuf x 4 at 2e-3; none@2e-3 B1 / A, none@4e-3 A)")
    print(f"  S0     {nS0:3d} runs = {hS0:5.1f} B200-h (A2: OPTIONAL, descriptive only: the shuffled arm at seeds 0-7; not in the totals)")
    print(f"  R1     {nR1:3d} runs = {hR1:5.1f} B200-h ({nR1 * PLAN_H_PTDOM:.1f})")
    print(f"  R1pre  {nP:3d} runs = {hP:5.1f} B200-h (A1.10: {', '.join(f'{b} none@{lr}' for b, _, lr in R1PRE)}, seeds 0-3)")
    print(f"  R1x    0-{nX} runs = 0-{hX:.1f} B200-h (prior 8e-3 for each of 6 domains at most; none: the directions R1pre left, {xn}; fixB = one of them)")
    print(f"  R2     every R domain FINAL: {16 * len(R_DOMS) + 8 * len(batches)}-{16 * len(R_DOMS) + 8 * len(R_DOMS)} runs = "
          f"{h2_prior + h2n_lo:.1f}-{h2_prior + h2n_hi:.1f} B200-h (one / six distinct none lrs); 16 runs less per NOT FINAL / GATE domain")
    lo = hS + hR1 + hP
    hi = lo + hX + h2_prior + h2n_hi
    nom = lo + h2_prior + h2n_lo + 4 * h("B3")   # (A2: if A's none pick is 2e-3, R2 re-uses Part S's none@2e-3 cell: 8 runs fewer)
    print(f"  total: min {nS + nR1 + nP} runs ({lo:.1f} B200-h, every R domain NOT FINAL / GATE), nominal ~{nS + nR1 + nP + 4 + 16 * 6 + 8 * 3} "
          f"runs ({nom:.1f} B200-h: no prior extension, the B3 fixB / lower cell, one none pick per batch), max "
          f"{nS + nR1 + nP + nX + 16 * 6 + 8 * 6} runs ({hi:.1f} B200-h)")
    print(f"  wall clock with {N_SLOTS} chains always running: S + R1 + R1pre ~{lo / N_SLOTS:.1f} h, R1x <= {hX / N_SLOTS:.1f} h, R2 "
          f"{(h2_prior + h2n_lo) / N_SLOTS:.1f}-{(h2_prior + h2n_hi) / N_SLOTS:.1f} h (+ queue waits; ptdom: 384 runs in ~3.5 days on 16 chains)")
    print("  disk: no saved models (A1.9); run dirs ~1-2 MB each")


# ---------------------------------------------------------------- main
def main():
    if "--plan" in sys.argv:
        plan()
        return
    if "--snapshot" in sys.argv:
        print("snapshot", snapshot(), (R / "code" / "COMMIT").read_text().strip())
        return
    pilot = "--pilot" in sys.argv
    part = "pilot" if pilot else (opt("--part") or opt("--stage"))   # --stage: an alias of --part
    assert part in PARTS, "--part S | R1 | R1x | R2 (or --pilot); --plan for the cost"
    commit = snapshot()
    P = load_P()
    todo = cells(part, opt("--picks"))
    rerun = "--rerun" in sys.argv
    assert not (rerun and pilot)
    if rerun:   # ptdom's A2 path: a cell whose latest submission exited before its run finished -> <name>-r<k+1>
        todo = rerun_plan(todo)
    else:
        todo = [(b, a, lr, s, 0) for b, a, lr, s in todo]
    others = {p.name for p in B.glob("runs/*/out/*") if p.parent.parent != R}
    JD = R / "pilot" if pilot else R
    JD.mkdir(parents=True, exist_ok=True)
    jobs, RF = [], {}
    for i, (b, a, lr, s, rr) in enumerate(todo):
        RF.setdefault(b, refs2(b))
        assert s in RF[b], f"no CPU reference for batch {b} seed {s}: run runs/ptdom2/tests/refs_ptdom2.py first"
        name, tag, text, args, I = render2(P, part, b, a, lr, s, RF[b][s], commit, rr, i % N_SLOTS, pilot)
        assert name not in others, f"{name}: an out dir of that name exists in another batch"
        jobs.append((name, text, args, I))
    if pilot:
        jobs = [pilot_job(jobs)]
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
        assert part != "S0" or "--allow_s0" in sys.argv, "--part S0 is optional and descriptive (A2.1): submit it only on the coordinator's explicit decision (--allow_s0)"
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
    print(f"ptdom2 part {part}{' (re-runs)' if rerun else ''}: {len(jobs)} jobs rendered in {JD} ({len(written)} written); code {commit}, "
          f"trainer {WANT_SHA['pretrain_range.py']}; B200; submitted: {'yes' if '--submit' in sys.argv else 'no'}")


def rerun_plan(todo):
    """ptdom's A2 re-run path on ptdom2 names (MJ.rerun_plan's rules, re-implemented for the ptdom2 names: the latest name has
    a SUBMITTED_ flag, its job id is not in the queue, its out dir holds no finished run JSON)."""
    q = MJ.queue()
    if q is None:
        raise SystemExit("--rerun: squeue failed; a re-run needs the queue state (refused)")
    ids, _ = q
    out = []
    for b, a, lr, s in todo:
        k = 0
        while (R / f"SUBMITTED_{name2(b, a, lr, s, k + 1)}").exists():
            k += 1
        flag = R / f"SUBMITTED_{name2(b, a, lr, s, k)}"
        if not flag.exists():
            continue
        txt = flag.read_text().split()
        jid = txt[-1] if txt and txt[-1].isdigit() else None
        if jid is None or jid in ids:
            print(f"re-run refused: {name2(b, a, lr, s, k)}: " + ("no job id" if jid is None else f"job {jid} is in the queue"))
            continue
        tg = tag2(b, a, lr)
        rj = R / "out" / name2(b, a, lr, s, k) / tg / f"run_whole_none_seed{s}_{tg}.json"
        if rj.exists():
            try:
                if len(json.loads(rj.read_text())["curve"]) >= 75:
                    print(f"re-run refused: {name2(b, a, lr, s, k)}: its run finished (a run is data, never re-run)")
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"re-run: {name2(b, a, lr, s, k)} -> {name2(b, a, lr, s, k + 1)}")
        out.append((b, a, lr, s, k + 1))
    return out


def pilot_job(bodies):
    """The ONE pilot job: B200, 1 h, job name llmadd_ptdom2-pilot (not a slot chain); it writes the two rendered bodies to
    runs/ptdom2/pilot/body_<name>.sh, runs them in turn with bash (each: HOLD, snapshot shas, B200 guard, data sha, dry_inputs +
    the input checks + DRY rows = the CPU reference, training to --stop_after, CHECK_PY / CHECK_PY_SHUF with the cell's intent),
    then runs/ptdom2/pilot/pilot_check.py (the pilot's pre-registered pass rules, AUDIT.md 'GPU pilot')."""
    pd = R / "pilot"
    names = [n for n, *_ in bodies]
    if "--no-write" not in sys.argv:
        pd.mkdir(parents=True, exist_ok=True)
        for n, text, *_ in bodies:
            bf = pd / f"body_{n}.sh"
            if (pd / f"SUBMITTED_{PREFIX}-pilot").exists():   # a submitted pilot's bodies are never rewritten
                if bf.read_text() != text:
                    print(f"WARNING: {bf.name} belongs to the submitted pilot and differs from this rendering: kept")
            else:
                bf.write_text(text)
    run_lines = "\n".join(f'bash "$PD/body_{n}.sh" > "$PD/body_{n}.log" 2>&1; echo "## body {n} rc $?"; tail -n 3 "$PD/body_{n}.log"'
                          for n in names)
    hw = HW["b200"]
    text = f"""#!/bin/bash
#SBATCH --job-name=llmadd_{PREFIX}-pilot
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
#SBATCH --output={B}/logs/llmadd_{PREFIX}-pilot_%j.out
# ptdom2 GPU pilot (runs/ptdom2/AUDIT.md 'GPU pilot'; not part of any result): two short runs on the real schedule up to
# --stop_after {PILOT_STOP} (evals 0..{PILOT_STOP} every 250; on-window evals 1000..{PILOT_STOP}): {', '.join(names)}.
# Each body = the ptdom2 job body of its cell (rendered by make_jobs_ptdom2.py through make_jobs_ptdom.render; save_model off,
# the stop_after flag in ARGS and intent); its CHECK fails by design exactly the truncation checks (75 evals, prior_events,
# prior_on_frac); pilot_check.py asserts that and the shuffled-target effect, data identity with ptdom's twin run, speed, memory.
set -u
PD={pd}; PY={PY}
if [ -e "{R}/HOLD" ] || [ -e "{R}/HOLD_pilot" ]; then echo "HOLD present: exiting"; exit 0; fi
date -u +%FT%TZ; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
{run_lines}
export PYTHONPATH={B}/pylib
"$PY" "$PD/pilot_check.py" {' '.join(names)} | tee "$PD/pilot_check.out"
echo "## pilot done $(date -u +%FT%TZ)"
"""
    return f"{PREFIX}-pilot", text, "; ".join(a for _, _, a, _ in bodies), [I for *_, I in bodies]


if __name__ == "__main__":
    main()
