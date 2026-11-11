"""ptdom read-out, Amendments A3 + A4 (PRE-REGISTERED in runs/ptdom/AUDIT.md "Amendment A3" and "Amendment A4" before any
run; A3 supersedes the A1 / A2 lr design and keeps the A2 per-cell auxiliary-loss gate, the shortcut baselines, the
shared-site statistics and the Tier L tags; A4 = the coordinator's decisions on the A3 audit): 24 candidate domains in four
batches (make_jobs_ptdom.BATCHES), per domain 'only-d at its own best lr' vs 'none at its best lr FOR d' within d's batch,
seeds paired.
A4 changes (where they differ, A4 holds; the A3 text below is kept and marked):
  - k-step read-out = the FULL-SPAN k-way choice (<d>_readout_kspan, trainer Rev G); the first-token choice is printed as
    a secondary line.
  - none's extensions at both edges: lower 2e-3 -> 1e-3, then 5e-4 if 1e-3 is still the argmax; upper 8e-3 -> 1.6e-2; a
    none unit whose pick or area argmax sits at an outermost lr it ran (5e-4 / 1.6e-2) after the allowed steps -> NOT FINAL.
    Every extension runs at seeds 0-7 (the selection reads 0-3); rounds: stage 2 (after 1), 2b (after 2), 2c (after 3).
  - the screen also needs the SAME-LR paired gain at 2e-3 (both arms ran it, seeds 0-3) > 0 in 4/4 seeds; it is printed
    with every stage-2 verdict.
  - the shortcut condition (iii) on the POOLED hard-subset questions of the seeds (>= SC_MIN; else 'not testable':
    recorded, does not block, reported).
  - the verdict adds the selection-free condition: seeds 4-7 all positive and their one-sided one-sample t p, Holm over
    every stage-2 domain of all batches, < SF_ALPHA (calibrated by tests/a4/a4_null_mc.py); the A2 gate at the pick is
    judged on seeds 0-3 and on seeds 4-7 separately; FLOOR uses the shortcut / elimination baselines; the strict read-out
    prints PROVISIONAL (no PASS) unless all four batches' decision chains are complete and agree with the picks files.
A5 changes (AUDIT.md "Amendment A5", before any stage-1 submission; A5 holds where it differs):
  - a run whose only failed checks are the GPU-numerics double-path tolerances (CHECK_PY, ptheld Amendment DP's rule) and /
    or the CKA-rise bar is VALID data, flagged (listed by the screen, select and the read-out), never dropped (A5.1).
  - fixB (A5.2): when a prior's final pick lies outside the lr range none ran for that domain, none is run there (seeds
    0-7, decision list F, stages 2b / 2c) and re-selected including it (F87 rule 1) before the verdict; no verdict while
    those cells are missing. Picks files carry amendment 'A5'.
  - --allow_partial: the comparison sub-read-out problems make the read-out PROVISIONAL and are printed (audit_a4 minor 3);
    the screen notes that NOT ADVANCE where none prefers a higher lr than the prior's stage-1 grid is not evidence of no
    effect (A5.3).
A6 (AUDIT.md "A6 — geo overshoot at a perfect fit", written before any B2 result was read; A6 holds where it differs):
  - a run whose only failed checks are A5.1's set and the geo / grad_prior check is VALID data, flagged A6 (listed by the
    screen, select and the read-out), iff geo_a6() verifies from the run JSON: grad_prior finite > 0 at every eval with
    prior_on; at every such eval with geo_<active> <= -0.001: 0.995 <= cka_<active> <= 1.01 (A6.1: the bar 0.999 -> 0.995
    and the cap 1.01) and geo_<active> >= -0.005; no inactive
    domain has a geo / prior-gradient field and the separate checks for that passed. Otherwise the run stays invalid. The
    check names are matched by prefix; a6_guard() asserts them in the CHECK_PY of every rendered job file.

  python readout_ptdom.py [--root runs/ptdom/out] [--allow_partial] [--json out.json]      (stage-2 verdicts, all batches)
  python readout_ptdom.py --screen X [--root ...]                                           (the stage-1 screen of batch X)
  python readout_ptdom.py --selftest

Runs: <root>/ptdom-<batch>-<arm>-lr<lr>-s<seed>[-r<k>]/ (make_jobs_ptdom.py): <tag>/run_whole_none_seed<s>_<tag>.json,
check.json, offline_lm.json. VALID run = finished (75 evals) and check.json pass, or its only failed checks are the CKA-rise
sanity bar (a weak prior: data) and / or (A5.1) the GPU-numerics double-path tolerances (flagged) and / or (A6, verified by
geo_a6) the geo / grad_prior check at a geo overshoot of a perfect fit (flagged). A diverged run is data (its area counts; its bpb is the worst). Any other failed check ->
EXCLUDED with the reason. A2 re-runs: the loader reads a cell's latest dir; integrity() refuses a cell with a FINISHED run
in an earlier dir.
Primary metric = area (100 x mean over the 75 evals) of the domain's held-out read-out: k-step domains (months, zodiac,
planets, elements, chinese_zodiac, geo_periods, greek) = the k-way choice over the domain's items (A4: the full-span choice
<d>_readout_kspan; A3's first-token <d>_readout_kway is the secondary line; months: tier S); comparison domains = the 2-way
choice over every held-out question, both orders (<d>_readout). final3 secondary.
A2 auxiliary-loss gate (unchanged; the user's rule 'the prior is only an auxiliary loss; the LM gradient must dominate'),
per (arm, lr) CELL: per run the median over its on-window evals of a / (a + b) (a = grad_lm, b = grad_prior, pre-clip
norms); a cell FAILS if the median of its run medians is < 0.5 or a run is unverifiable (a missing gradient field at an
on-window eval, no on-window eval, the logged grad_lm_share off its re-derivation by > 1e-3). A failing cell is excluded
from its arm's selection; an arm whose every stage-1 cell fails -> GATE (no pick; select writes HOLD).
STAGE 1 (seeds 0-3; prior 1e-3 / 2e-3, none 2e-3 / 4e-3 / 8e-3). Per domain d of a batch: prior pick P = F87 rule 1
(readout_rules.best_lr 'top2': the argmax of the mean area; top two within 2 points -> the lower mean bpb, offline
full-val if every run of both has it) over the admissible prior lrs on d's read-out; none pick N likewise over none's grid
on d's read-out. SCREEN (lenient, pre-registered): d ADVANCES iff (i) gain_s = area(only-d@P, s) - area(none@N, s) > 0 in
4/4 seeds, (ii) the gate passes at the picked cell (seeds 0-3), (iii) comparison domains: the shortcut condition: the
mean over seeds 0-3 (seeds with a non-empty hard subset) of only-d@P's hard-subset FINAL3 > the mean over the same seeds of
the training-frequency shortcut baseline on that subset (anchor templates: A1's anchor-conditional win rate; templates
without an anchor: the option's training win rate); every seed's hard subset empty -> (iii) fails (the answer-frequency
line solves every held-out question: the read-out cannot separate the prior from the shortcut). Every domain is reported.
STAGE 2 (advancing domains only): seeds 4-7 at P and at N; ONE extension lr at seeds 0-3 where the stage-1 area argmax is at
a grid edge: none's lower edge (2e-3) -> 1e-3; the prior's lower edge (1e-3) -> 5e-4, its upper edge (2e-3) -> 4e-3 (with
two prior lrs the argmax is always at an edge); none's upper edge (8e-3): no extension (the user's list). The unit is then
re-selected over grid + extension (F87 rule 1, seeds 0-3; a gate-failing extension cell excluded); a moved pick gets seeds
4-7 there (stage 2b). No second extension; a final argmax at the extension lr is tagged EDGE.
VERDICT (advancing domains, seeds 0-7 paired at the final picks; Holm over ALL advancing domains of ALL batches together):
  PASS ('shown effective in pretraining') iff (a) the 90% paired seed-bootstrap CI of the mean gain (10,000 resamples,
  default_rng(0), percentile) excludes 0; (b) both seed halves positive (seeds 0-3 and 4-7); (c) the two-sided one-sample
  t-test of the 8 gains, Holm-adjusted, p < 0.05 (so the per-seed t interval excludes 0); and the mean > 0. A failing pick
  (gate over seeds 0-7) -> 'AUX GATE FAILED at the pick' instead of a verdict (stays in the Holm family with its t p).
  Comparison domains: a PASS must also meet the shortcut condition on 8 seeds (hard-subset final3 > the baseline), else
  NOT SEPARABLE (from a training-frequency shortcut); they carry [Tier L] (the target carries the answers). A NOT SHOWN
  on a domain where >= 20% of the items share their site token reads 'NOT SHOWN with this site (many shared site
  tokens)'. Tags (never change the verdict): FLOOR, CEILING, SEEN < 80, HARM, EDGE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
B = HERE.parent.parent
TOOLS = str(HERE / "code" / "tools") if (HERE / "code" / "tools").exists() else "/scratch/yh6210/llmadd-burst-ptdom/tools"
sys.path.insert(0, TOOLS)
sys.path.insert(0, str(B))
import readout_rules as RR      # noqa: E402  (best_lr, TIE: the shared F87 definitions)
import make_jobs_ptdom as MJ    # noqa: E402  (the design constants: batches, grids, extensions, read-out keys)

TOK = "/scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
PYLIB = "/scratch/yh6210/llmadd_torch/pylib"
BATCHES, HWOF, KSTEP, COMP, ALL24 = MJ.BATCHES, MJ.HWOF, MJ.KSTEP, MJ.COMP, MJ.ALL24
GRID1, EXT_LR, EXT2, OUTER, LADDER, SEL, CONF = MJ.GRID1, MJ.EXT_LR, MJ.EXT2, MJ.OUTER, MJ.LADDER, MJ.SEL, MJ.CONF
READ, SEEN, READ1 = MJ.READ, MJ.SEEN, MJ.READ1
EVAL_KEY = {d: (f"{d}_held_kspan" if d in KSTEP else f"{d}_held_2way") for d in ALL24}   # the read-out's eval key (step 0)
DOM_BATCH = {d: b for b, ds in BATCHES.items() for d in ds}
N_EVALS, N_BOOT, ALPHA_T = 75, 10000, 0.05
AUX_MIN = 0.5                      # A2: the gate statistic a / (a + b) < 0.5 <=> the prior's gradient norm exceeds the LM's
GRAD_FIELDS = ("grad_lm", "grad_prior", "grad_cos", "grad_lm_share", "clip_coef")   # A2: required at every on-window eval
DP_TOL = 1e-3                      # logged grad_lm_share vs its re-derivation from grad_lm, grad_prior, grad_cos
SHARED_FRAC = 0.2                  # A3: >= 20% of the items share their site token -> the 'with this site' wording (A1's two)
ANC_HELD = 0.4                     # = the trainer's ANC_HELD (ft36 make_split)
CKA_CHECK = "item-state CKA to the target rises"
# A5.1 (pre-registered before any stage-1 run): a run whose ONLY failed checks are these GPU-numerics tolerances of the scoring
# double paths (and / or the CKA-rise bar) is VALID data, FLAGGED (listed by the screen and the read-out), never dropped
DP_FLAG_CHECKS = ("2-way scoring double path (GPU-numerics tolerance", "k-span scoring double paths (GPU-numerics tolerance")
# A6 (AUDIT.md "A6", pre-registered before any B2 result was read): the geo / grad_prior check, matched by its exact rendered
# name prefix. In CHECK_PY the name is an f-string, chk(f"geo > {TOL} for the active prior ...") with TOL = the intent's
# geo_tol (-0.001 in every job), so a6_guard() asserts the template, the TOL line and geo_tol in every rendered job file. A run
# whose only failed checks are A5.1's set plus this one is VALID data, FLAGGED A6, iff geo_a6() verifies from the run JSON:
# (1) grad_prior finite > 0 at every eval with prior_on; (2) at every such eval with geo_<active> <= GEO_TOL: cka_<active>
# (the logged eval-prompt item-state CKA, unrounded) in [GEO_CKA_MIN, GEO_CKA_MAX] and geo_<active> >= GEO_FLOOR (a hard floor:
# a real sign bug still fails); (3) no inactive domain has a geo / prior-gradient field, and the separate checks A6_KEEP are
# present (once) and passed. Otherwise the run stays invalid.
# A6.1 (tests/a6/audit_a6.md minors 1-2): the bar 0.999 -> 0.995 (the eval CKA's own bf16 noise reaches 0.998 at real geo < 0
# evals) and an upper sanity cap 1.01 (CKA <= 1 in exact arithmetic; the largest real 2-layer overshoot is +1.04e-3)
GEO_TOL, GEO_FLOOR, GEO_CKA_MIN, GEO_CKA_MAX = -1e-3, -5e-3, 0.995, 1.01
_GEO_REST = " for the active prior and 0 < grad_prior < inf at every on-eval"
GEO_FLAG_CHECK = f"geo > {GEO_TOL}" + _GEO_REST   # 'geo > -0.001 for the active prior and 0 < grad_prior < inf at every on-eval'
GEO_TEMPLATE = 'chk(f"geo > {TOL}' + _GEO_REST    # its source text in CHECK_PY
A6_KEEP = ("the active prior applied on >= 95% of the on-steps; every inactive one never",
           "geo_<d> logged at >= 95% of the evals for the active prior, never for an inactive one",
           "weighted prior gradients at the on-evals from the active prior only",
           "numbers prior off: never applied, no numbers loss term")
assert GEO_FLAG_CHECK == "geo > -0.001 for the active prior and 0 < grad_prior < inf at every on-eval" and GEO_TOL == MJ.GEO_TOL
FIXB = "A5 fixB"   # A5.2: none run at a prior's final pick lr outside the lr range none ran for that domain
NAME = re.compile(r"^ptdom-(A|B1|B2|B3)-(none|only-[a-z_]+)-lr([0-9.e-]+)-s(\d+)(?:-r(\d+))?$")   # A2: -r<k> = the k-th re-run
GATE_PICK_FAIL = "AUX GATE FAILED at the pick"
NOT_FINAL = "NOT FINAL (none's pick or argmax at the outermost lr it may reach: lr-confounded, no verdict)"
SAME_LR = "2e-3"                   # A4: the lr both arms run in stage 1 (the same-lr paired gain, screen condition (iv))
SC_MIN = 30                        # A4: the shortcut condition needs >= 30 pooled hard-subset questions, else 'not testable'
SF_ALPHA = 0.10                    # A4: seeds 4-7 one-sided t p (Holm over every stage-2 domain) < SF_ALPHA (a4_null_mc.py)
med = lambda v: float(np.median(v)) if len(v) else None
assert GRID1 == {"none": ("2e-3", "4e-3", "8e-3"), "prior": ("1e-3", "2e-3")}
assert EXT_LR == {("none", "lower"): "1e-3", ("none", "upper"): "1.6e-2", ("prior", "upper"): "4e-3", ("prior", "lower"): "5e-4"}
assert EXT2 == {("none", "lower"): "5e-4"} and OUTER == {"none": ("5e-4", "1.6e-2")}


def grid(arm):
    return GRID1["none" if arm == "none" else "prior"]


def kind(arm):
    return "none" if arm == "none" else "prior"


# ------------------------------------------------------------------------------------------------------------ gradients (A2)
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def window_split(c, div=None):
    """A2: per on-window eval of a prior run (prior_on == 1, before a divergence; EVERY such eval, so a missing field is
    seen): a = grad_lm, b = grad_prior, the LM share by norm a / (a + b), clip_coef, the logged grad_lm_share (= a / |g_lm +
    sum lambda g_d|) and its re-derivation a / sqrt(a^2 + b^2 + 2ab cos), and the missing / invalid fields."""
    rows = []
    for e in c:
        if not e.get("prior_on") or (div is not None and e["step"] >= div):
            continue
        miss = [k for k in GRAD_FIELDS if not _num(e.get(k))]
        a, b, cs = e.get("grad_lm"), e.get("grad_prior"), e.get("grad_cos")
        if not miss and (a <= 0 or b < 0):
            miss.append("grad_lm > 0 / grad_prior >= 0")
        T = math.sqrt(max(a * a + b * b + 2 * a * b * cs, 0.0)) if not miss else None
        rows.append({"step": e["step"], "a": a, "b": b, "share": a / (a + b) if not miss else None, "clip": e.get("clip_coef"),
                     "logged": e.get("grad_lm_share"), "rederived": (a / T) if T else None, "missing": miss})
    return rows


def window_grad(c, div=None):
    """A2: a run's gradient split over its on-window evals (median a / (a + b) = the gate's run value; counts; the double
    path's largest disagreement; descriptive medians)."""
    rows = window_split(c, div)
    ok = [r for r in rows if not r["missing"]]
    sh = [r["share"] for r in ok]
    dp = [abs(r["logged"] - r["rederived"]) if r["rederived"] is not None else math.inf for r in ok]
    return {"n_on": len(rows), "n_ok": len(ok), "n_missing": len(rows) - len(ok),
            "missing": sorted({k for r in rows for k in r["missing"]}), "share": med(sh), "share_min": min(sh) if sh else None,
            "n_b_gt_a": sum(r["b"] > r["a"] for r in ok), "n_clip_lt1": sum(r["clip"] < 1 for r in ok),
            "dp_maxdiff": max(dp) if dp else None, "lm_share_T": med([r["logged"] for r in ok]),
            "prior_over_lm": med([r["b"] / r["a"] for r in ok]), "clip_coef": med([r["clip"] for r in ok]),
            "last_step": c[-1]["step"] if c else None}


def run_unverifiable(g):
    why = []
    if not g or not g["n_on"]:
        return ["no on-window eval"]
    if g["n_missing"]:
        why.append(f"missing gradient fields ({', '.join(g['missing'])}) at {g['n_missing']} of {g['n_on']} on-window evals")
    if g["dp_maxdiff"] is not None and g["dp_maxdiff"] > DP_TOL:
        why.append(f"logged grad_lm_share vs a / sqrt(a^2 + b^2 + 2ab cos): max diff {g['dp_maxdiff']:.1e} > {DP_TOL}")
    return why


def cell_gate(rs):
    """A2: one (arm, lr) cell from its runs: FAILS if the median over its runs of each run's on-window median a / (a + b)
    is < AUX_MIN, or any run is unverifiable."""
    rs = sorted(rs, key=lambda r: r["seed"])
    per = {r["seed"]: r["grad"]["share"] for r in rs}
    why = [f"s{r['seed']}: {w}" for r in rs for w in run_unverifiable(r["grad"])]
    v = [x for x in per.values() if x is not None]
    m = med(v)
    if m is not None and m < AUX_MIN:
        why.insert(0, f"median LM share a / (a + b) {m:.3f} < {AUX_MIN}")
    n_ok = sum(r["grad"]["n_ok"] for r in rs if r["grad"])
    dp = [r["grad"]["dp_maxdiff"] for r in rs if r["grad"] and r["grad"]["dp_maxdiff"] is not None]
    return {"ok": not why, "why": why, "n_runs": len(rs), "seeds": [r["seed"] for r in rs], "median": m,
            "min": min(v) if v else None, "max": max(v) if v else None, "per_run": per, "n_on": n_ok,
            "frac_b_gt_a": sum(r["grad"]["n_b_gt_a"] for r in rs if r["grad"]) / n_ok if n_ok else None,
            "frac_clip_lt1": sum(r["grad"]["n_clip_lt1"] for r in rs if r["grad"]) / n_ok if n_ok else None,
            "dp_maxdiff": max(dp) if dp else None}


def aux_gate(R_, batch, seeds=SEL):
    """A2 per cell, one batch: every prior (arm, lr) cell with FINISHED runs at the given seeds (valid or not: the gradient
    log is a fact about the update). arms_all_failed: every stage-1 lr of the arm has a gated cell and every gated cell of the
    arm fails (no pick; HOLD)."""
    cells = {}
    for arm in MJ.arms(batch)[1:]:
        for lr in sorted({k[2] for k in R_ if k[:2] == (batch, arm)}, key=float):
            rs = [r for k, r in R_.items() if k[:3] == (batch, arm, lr) and k[3] in seeds and r.get("finished")]
            if rs:
                cells[f"{arm}@{lr}"] = cell_gate(rs)
    failed = [k for k, x in cells.items() if not x["ok"]]
    all_failed = [arm for arm in MJ.arms(batch)[1:] if all(f"{arm}@{lr}" in cells for lr in grid(arm))
                  and all(not x["ok"] for k, x in cells.items() if k.split("@")[0] == arm)]
    return {"batch": batch, "ok": not failed, "failed": failed, "cells": cells, "arms_all_failed": all_failed,
            "min_share": AUX_MIN, "seeds": list(seeds)}


def gate_report(g, title="auxiliary-loss gate (A2, per cell)"):
    L = [f"{title}, batch {g['batch']}: per (arm, lr) cell the median over its finished runs (seeds {g['seeds']}) of the run's "
         f"on-window median LM share by norm a / (a + b) (a = grad_lm, b = grad_prior, pre-clip); a cell fails < {g['min_share']} "
         f"or with an unverifiable run (a missing gradient field, no on-window eval, logged vs re-derived share > {DP_TOL})"]
    f3 = lambda v: "-" if v is None else f"{v:.3f}"
    pc = lambda v: "-" if v is None else f"{100 * v:.0f}%"
    for key, x in g["cells"].items():
        dm = "-" if x["dp_maxdiff"] is None else f"{x['dp_maxdiff']:.1e}"
        L.append(f"  {key:28s} runs {x['n_runs']} (seeds {','.join(map(str, x['seeds']))}): median LM share {f3(x['median'])} "
                 f"[min {f3(x['min'])}, max {f3(x['max'])}]; on-window evals {x['n_on']}: b > a {pc(x['frac_b_gt_a'])}, "
                 f"clip < 1 {pc(x['frac_clip_lt1'])}; double path {dm}" + (f"  FAIL: {'; '.join(x['why'])}" if x["why"] else ""))
    if g["ok"]:
        L.append("AUX GATE passed (every cell)")
    else:
        L.append("AUX GATE: failing cells (excluded from their arm's selection): " + ", ".join(g["failed"]))
        if g["arms_all_failed"]:
            L.append("AUX GATE FAILED: " + ", ".join(g["arms_all_failed"]) + " (every stage-1 cell of the arm fails: no pick; HOLD)")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------ loading
def cell_dirs(root):
    """{(batch, arm, lr, seed): [(k, dir), ...] sorted by k}; k = 0 for the first submission, k for its k-th re-run."""
    out = {}
    for d in sorted(Path(root).glob("ptdom-*-s*")):
        m = NAME.match(d.name)
        if m:
            out.setdefault((m.group(1), m.group(2), m.group(3), int(m.group(4))), []).append((int(m.group(5) or 0), d))
    return {k: sorted(v) for k, v in out.items()}


def run_json(d, batch, arm, lr, seed):
    tag = MJ.tag_of(batch, arm, lr)
    return Path(d) / tag / f"run_whole_none_seed{seed}_{tag}.json"


@lru_cache(maxsize=None)
def a6_guard(job_dir=str(HERE)):
    """A6: the check names the loader matches by prefix for A6 occur in the CHECK_PY of EVERY rendered job file
    (job_ptdom-*.sbatch in job_dir), so a renamed check cannot silently stop matching: the geo check's f-string template
    GEO_TEMPLATE once, the TOL line, the intent's geo_tol == GEO_TOL (so the rendered name starts with GEO_FLAG_CHECK), and
    every A6_KEEP name once as a literal chk("<name>... -> the number of files checked (AssertionError otherwise)."""
    files = sorted(Path(job_dir).glob("job_ptdom-*.sbatch"))
    assert files, f"A6 guard: no rendered job file job_ptdom-*.sbatch in {job_dir}"
    sep = "<<'PY' | tee \"$OUT/check.txt\"\n"
    for f in files:
        txt = f.read_text()
        m = re.search(r"\"\$OUT/check.json\" '(\{.*\})' \"\$ARGS\" <<'PY'", txt)
        assert m and txt.count(sep) == 1, f"A6 guard: {f.name}: no CHECK_PY heredoc / intent found"
        cp = txt.split(sep, 1)[1].split("\nPY\n", 1)[0]
        tol = json.loads(m.group(1)).get("geo_tol")
        assert cp.count(GEO_TEMPLATE) == 1 and 'ACT, TOL = I["active"], I["geo_tol"]' in cp, \
            f"A6 guard: {f.name}: the geo / grad_prior check is not named '{GEO_TEMPLATE}...' in its CHECK_PY"
        assert tol == GEO_TOL and f"geo > {tol}" + _GEO_REST == GEO_FLAG_CHECK, f"A6 guard: {f.name}: geo_tol {tol} != {GEO_TOL}"
        miss = [p for p in A6_KEEP if cp.count(f'chk("{p}') != 1]
        assert not miss, f"A6 guard: {f.name}: the separate checks {miss} are not named so (once) in its CHECK_PY"
    return len(files)


def geo_a6(r, ck, arm):
    """A6: verify from the run JSON r (and its check.json ck) that a failed geo / grad_prior check is a geo overshoot at a
    perfect fit -> {'why': [reasons A6 does not apply; empty = it applies], 'flag': the A6 flag record}. 'Every eval with
    prior_on' is read literally: every curve entry with prior_on, also after a divergence (stricter than the check's 'before
    a divergence'; the same set for a run that did not diverge). Interpretations (AUDIT.md A6): a non-finite geo_<active> at
    such an eval is unverifiable (A6 does not apply); the failure must be explained (some eval with geo_<active> <= GEO_TOL
    or a grad_prior not finite > 0), else A6 does not apply; the active prior must be one domain, the same in domains.<d>.prior,
    args.prior_domains and the arm name."""
    c, why = r["curve"], []
    doms = r.get("domains", {})
    act = sorted(x for x, v in doms.items() if v.get("prior"))
    jpri = [x for x in str(r.get("args", {}).get("prior_domains") or "").split(",") if x]
    if len(act) != 1 or jpri != act or arm != f"only-{act[0]}":
        return {"why": [f"not one active prior (domains.<d>.prior {act}, args.prior_domains {jpri}, arm {arm})"], "flag": None}
    d = act[0]
    gk, kk = f"geo_{d}", f"cka_{d}"
    on = [e for e in c if e.get("prior_on")]
    if not on:
        why.append("no eval with prior_on")
    gp_bad = [e["step"] for e in on if not (_num(e.get("grad_prior")) and e["grad_prior"] > 0)]
    if gp_bad:
        why.append(f"(1) grad_prior not finite > 0 at {len(gp_bad)} of {len(on)} evals with prior_on (steps {gp_bad[:5]})")
    nf = [e["step"] for e in on if e.get(gk) is not None and not _num(e[gk])]
    if nf:
        why.append(f"{gk} not finite at {len(nf)} evals with prior_on (steps {nf[:5]}): unverifiable")
    low = [{"step": e["step"], "geo": e[gk], "cka": e.get(kk)} for e in on if _num(e.get(gk)) and e[gk] <= GEO_TOL]
    for x in low:
        if x["geo"] < GEO_FLOOR:
            why.append(f"(2) step {x['step']}: {gk} {x['geo']:.6f} < the floor {GEO_FLOOR}")
        if not (_num(x["cka"]) and x["cka"] >= GEO_CKA_MIN):
            why.append(f"(2) step {x['step']}: {kk} {x['cka']} < {GEO_CKA_MIN} (not a perfect fit)")
        elif x["cka"] > GEO_CKA_MAX:   # A6.1: above any float overshoot (CKA <= 1 in exact arithmetic): a computation anomaly
            why.append(f"(2) step {x['step']}: {kk} {x['cka']} > {GEO_CKA_MAX} (above any float overshoot: an anomaly)")
    if on and not low and not gp_bad and not nf:
        why.append(f"the check failed, but no eval with prior_on has {gk} <= {GEO_TOL} and grad_prior is finite > 0 at all of "
                   "them: the failure is not the A6 case")
    ina = [x for x in doms if x != d]
    bad3 = sorted({f"geo_{x}" for e in c for x in ina if e.get(f"geo_{x}") is not None}
                  | {f"gnorm_prior_{x}" for e in c for x in ina if f"gnorm_prior_{x}" in e}
                  | ({"geo (the numbers prior) != 0"} if any(e.get("geo") not in (None, 0) for e in c) else set()))
    if bad3:
        why.append(f"(3) an inactive domain has a geo / prior-gradient field: {bad3[:6]}")
    ckn = [(k_["check"], k_["ok"]) for k_ in ck.get("checks", [])]
    for p in A6_KEEP:
        m = [o for n, o in ckn if n.startswith(p)]
        if m != [True]:
            why.append(f"(3) the separate check '{p} ...' is not present once and passed ({m})")
    gp = [e["grad_prior"] for e in on if _num(e.get("grad_prior"))]
    return {"why": why, "flag": {"active": d, "n_on": len(on), "grad_prior_min": min(gp) if gp else None, "evals": low}}


def load(root):
    """{(batch, arm, lr, seed): rec} of every cell under root (the latest run dir of the cell); rec['valid'] / ['why'] /
    ['finished']; rec['earlier_finished'] lists earlier dirs of the cell that hold a FINISHED run. A6: a6_guard() first; a
    valid A6 run carries rec['geo_flag']."""
    a6_guard()
    out = {}
    for (batch, arm, lr, seed), dirs in cell_dirs(root).items():
        doms = BATCHES[batch]
        k_, d = dirs[-1]
        f = run_json(d, batch, arm, lr, seed)
        early = []
        for k0, d0 in dirs[:-1]:
            f0 = run_json(d0, batch, arm, lr, seed)
            try:
                if f0.exists() and len(json.loads(f0.read_text())["curve"]) == N_EVALS:
                    early.append(d0.name)
            except (json.JSONDecodeError, KeyError):
                pass
        rec = {"batch": batch, "arm": arm, "lr": lr, "seed": seed, "dir": str(d), "rerun": k_,
               "superseded": [d0.name for _, d0 in dirs[:-1]], "earlier_finished": early, "valid": False, "why": "", "finished": False}
        out[batch, arm, lr, seed] = rec
        if not f.exists():
            rec["why"] = "no run JSON"
            continue
        r = json.loads(f.read_text())
        c = r["curve"]
        ck = json.loads((d / "check.json").read_text()) if (d / "check.json").exists() else None
        off = json.loads((d / "offline_lm.json").read_text()) if (d / "offline_lm.json").exists() else {}
        oe = off.get(str(f.resolve())) or (next(iter(off.values())) if len(off) == 1 else None)
        fails = [k["check"] for k in (ck or {}).get("checks", []) if not k["ok"]]
        dpf = [{"check": k["check"].split(" (")[0], "value": k.get("value")} for k in (ck or {}).get("checks", [])
               if not k["ok"] and k["check"].startswith(DP_FLAG_CHECKS)]
        div = next((e["step"] for e in c if not all(v is None or math.isfinite(v) for v in
                                                      (e.get("lm"), e.get("val_loss"), e.get("geo")))), None)
        cmp_ = [x for x in doms if x in COMP]
        rec.update(code_sha=r.get("code_sha"), args=r["args"], n_evals=len(c), diverged_at=div, finished=len(c) == N_EVALS,
                   area={x: r["metrics"].get(READ[x], {}).get("area") for x in doms},
                   final3={x: r["metrics"].get(READ[x], {}).get("final3") for x in doms},
                   area1={x: r["metrics"].get(READ1[x], {}).get("area") for x in doms if x in KSTEP},
                   kshort={x: (100 * r["domains"][x]["heuristic_shortest_kspan"] if r["domains"][x].get("heuristic_shortest_kspan")
                               is not None else None) for x in doms if x in KSTEP},
                   seen3={x: r["metrics"].get(SEEN[x], {}).get("final3") for x in doms},
                   hard={x: r["metrics"].get(f"{x}_readout_hard", {}).get("area") for x in cmp_},
                   hard3={x: r["metrics"].get(f"{x}_readout_hard", {}).get("final3") for x in cmp_},
                   n_hard={x: r["domains"][x].get("n_hard") for x in cmp_},
                   chance={x: 100 * r["domains"][x]["chance_readout"] for x in doms},
                   heur={x: [100 * r["domains"][x].get("heuristic_ansfreq", 0), 100 * r["domains"][x].get("heuristic_shorter", 0)]
                         for x in cmp_},
                   split_seed={x: r["domains"][x].get("split_seed") for x in doms},
                   step0={x: 100 * c[0][EVAL_KEY[x]] for x in doms},
                   numbers=r["metrics"]["numbers"]["area"], data_sha=[e.get("data_sha") for e in c], lm0=c[0]["lm"],
                   bpb_intr=c[-1]["val_bits_per_byte"] if div is None else math.inf,
                   bpb_off=(RR.offline_bpb(oe, diverged=div is not None) if oe else None),
                   fails=fails, grad=window_grad(c, div), cmp={})
        if div is not None:
            rec["bpb_off"] = math.inf
        for x in cmp_:   # the per-question data of the comparison read-out (compact), for run_subsets
            D_ = r["domains"].get(x, {})
            eh = D_.get("eval_held")
            if eh and all(f"hits2_{x}_held" in e for e in c):
                rec["cmp"][x] = {"q": eh["q"], "nctx": eh["contexts"], "masks": eh.get("masks", {}),
                                 "split_seed": D_.get("split_seed"), "table_sha": D_.get("table", {}).get("sha"),
                                 "heur": [D_.get("heuristic_ansfreq"), D_.get("heuristic_shorter")],
                                 "hits": [e[f"hits2_{x}_held"] for e in c], "logged": [e.get(f"{x}_held_2way") for e in c],
                                 "logged_hard": [e.get(f"{x}_held_2way_hard") for e in c],
                                 "area": rec["area"][x], "area_hard": rec["hard"][x]}
        geo_f = [x for x in fails if x.startswith(GEO_FLAG_CHECK)]   # A6
        a6 = None
        if ck is None:
            rec["why"] = "no check.json"
        elif len(c) != N_EVALS:
            rec["why"] = f"unfinished ({len(c)} evals)"
        elif fails and not all(x.startswith((CKA_CHECK, GEO_FLAG_CHECK) + DP_FLAG_CHECKS) for x in fails):
            rec["why"] = "failed check: " + "; ".join(fails)
        elif geo_f and (a6 := geo_a6(r, ck, arm))["why"]:   # A6: the geo / grad_prior check failed and is not the A6 case
            rec["why"] = "failed check: " + "; ".join(fails) + " (A6 does not apply: " + "; ".join(a6["why"]) + ")"
        else:   # A5.1: only the CKA-rise bar and / or the GPU-numerics double-path tolerances failed -> valid data, flagged;
            # A6: and / or the geo / grad_prior check at a verified geo overshoot of a perfect fit -> valid data, flagged A6
            rec["valid"] = True
            rec["dp_flag"] = dpf
            rec["geo_flag"] = a6["flag"] if a6 else None
            g6 = rec["geo_flag"]
            rec["why"] = ("valid" + (" (CKA-rise check failed: a weak prior, data)" if any(x.startswith(CKA_CHECK) for x in fails) else "")
                          + (" (FLAG A5.1: GPU-numerics double-path tolerance exceeded: " + "; ".join(x["check"] for x in dpf)
                             + "; valid data)" if dpf else "")
                          + (f" (FLAG A6: geo overshoot at a perfect fit: " + "; ".join(
                              f"geo_{g6['active']} {x['geo']:.6f} >= {GEO_FLOOR} with cka_{g6['active']} {x['cka']:.6f} in "
                              f"[{GEO_CKA_MIN}, {GEO_CKA_MAX}] at step {x['step']}" for x in g6["evals"])
                             + f"; grad_prior finite > 0 at all {g6['n_on']} evals with prior_on; valid data)" if g6 else ""))
    return out


def integrity(R_):
    """Per batch: the same data per seed across every valid run (data_sha at every eval; step-0 lm within 1e-3); one
    code_sha over every valid run; no cell with a FINISHED run in an earlier dir than the one read (a run is data)."""
    probs, V = [], [r for r in R_.values() if r["valid"]]
    for k, r in sorted(R_.items()):
        if r.get("earlier_finished"):
            probs.append(f"{k[0]} {k[1]}@{k[2]} s{k[3]}: a finished run in {r['earlier_finished']} and a later re-run dir {Path(r['dir']).name}")
    shas = {r["code_sha"] for r in V}
    if len(shas) > 1:
        probs.append(f"code_sha not unique: {sorted(shas)}")
    for b in sorted({r["batch"] for r in V}):
        for s in sorted({r["seed"] for r in V if r["batch"] == b}):
            rs = [r for r in V if r["seed"] == s and r["batch"] == b]
            if len({tuple(r["data_sha"]) for r in rs}) != 1:
                probs.append(f"batch {b} seed {s}: data_sha series differ across runs (the data must be identical in every arm)")
            if max(r["lm0"] for r in rs) - min(r["lm0"] for r in rs) > 1e-3:
                probs.append(f"batch {b} seed {s}: step-0 lm differs by > 1e-3 across runs")
    return probs


# ------------------------------------------------------------------------------------------------------------ question meta
def _md():
    sys.path.insert(0, PYLIB)
    import ft36
    from manifold_atlas import primary as md_primary
    from manifold_domains import domains as md_domains
    from transformers import AutoTokenizer
    return ft36, md_primary, md_domains(), AutoTokenizer.from_pretrained(TOK)


def domain_meta(doms=None):
    """Per domain (from the snapshot: ft36 question builders on manifold_domains.domains(), the primary atlas target, the
    jobs' tokenizer): items, token count of ' ' + name, the SITE token (the last token of ' ' + name: the trainer's site),
    the items sharing their site token, the questions and the table sha by the trainer's formula (Rev E comparisons /
    Rev F k-step). Comparison q = (anchor or None, option x, option y, answer index, key); k-step q = (x, k, answer, key).
    months (the rb6 path) has no entry."""
    ft36, md_primary, MD, tok = _md()
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    out = {}
    for d in [x for x in (doms or ALL24) if x != "months"]:
        it_, tn = MD[d]["items"], md_primary(MD[d], d)
        F_ = np.asarray(MD[d]["targets"][tn], dtype=np.float64)
        names = [enc(" " + it) for it in it_]
        site = [n[-1] for n in names]
        cnt = Counter(site)
        shared = [i for i, t in enumerate(site) if cnt[t] > 1]
        m = {"items": it_, "ntok": [len(n) for n in names], "site": site, "shared": shared, "target": tn,
             "groups": {tok.decode([t]): [it_[i] for i in range(len(it_)) if site[i] == t] for t in sorted({site[i] for i in shared})}}
        if d in KSTEP:
            Q_, cyc = ft36.offset_questions(d, MD[d])
            m.update(kind="kstep", cyclic=cyc, n=len(it_),
                     q=[(Q["xk"][0], Q["xk"][1], Q["ans"], tuple(Q["key"])) for Q in Q_],
                     sha=hashlib.sha256(json.dumps({"items": it_, "target": tn, "F": np.round(F_, 9).tolist(),
                                                    "q": [[Q["prompt"], it_[Q["ans"]]] for Q in Q_]}).encode()).hexdigest()[:12])
        else:
            Q_ = ft36.pair_questions(d, MD[d])
            rt, wr = (lambda Q: Q["opts"][Q["ans"]]), (lambda Q: Q["opts"][1 - Q["ans"]])
            has_a = all(len(Q["occ"]) == 3 for Q in Q_)
            assert has_a or all(len(Q["occ"]) == 2 for Q in Q_), d
            assert all(tuple(Q["opts"]) == (Q["occ"][0][0], Q["occ"][1][0]) for Q in Q_), d
            m.update(kind="comp", has_anchor=has_a,
                     q=[(Q["occ"][2][0] if has_a else None, Q["opts"][0], Q["opts"][1], Q["ans"], tuple(Q["key"])) for Q in Q_],
                     sha=hashlib.sha256(json.dumps({"items": it_, "target": tn, "F": np.round(F_, 9).tolist(),
                                                    "q": [[Q["prompt"], it_[rt(Q)], it_[wr(Q)]] for Q in Q_]}).encode()).hexdigest()[:12])
        out[d] = m
    return out


def split_info(m, split_seed):
    """A comparison domain's split (ft36.split of the option-pair keys, 40%) and per held-out question in the trainer's
    order: the build's heuristic lines and masks (answer frequency, shorter option, hard = both wrong or tied), the
    training-frequency SHORTCUT baseline (A1, audit heur2.py's rule): for held 'a: b or c' the option with the higher
    training win rate with anchor a (wins / appearances as an option); A3, a template without an anchor: the option's win
    rate over every training question; unseen or equal rates -> the answer-frequency line (1 / 0 / 0.5) -- and the unique-
    site flag (no item of the question shares its site token)."""
    import ft36
    Q, n = m["q"], len(m["items"])
    keys = [q[4] for q in Q]
    hk = ft36.split(keys, ANC_HELD, split_seed)
    train = [i for i, k in enumerate(keys) if k not in hk]
    held = [i for i, k in enumerate(keys) if k in hk]
    rt, wr = (lambda i: Q[i][1 + Q[i][3]]), (lambda i: Q[i][2 - Q[i][3]])
    acnt = np.bincount([rt(i) for i in train], minlength=n)
    af = [0.5 if acnt[rt(i)] == acnt[wr(i)] else float(acnt[rt(i)] > acnt[wr(i)]) for i in held]
    ln = m["ntok"]
    shr = [0.5 if ln[rt(i)] == ln[wr(i)] else float(ln[rt(i)] < ln[wr(i)]) for i in held]
    af_w = [bool(acnt[rt(i)] <= acnt[wr(i)]) for i in held]
    sh_w = [bool(ln[rt(i)] >= ln[wr(i)]) for i in held]
    win, app = Counter(), Counter()
    for i in train:
        a = Q[i][0]   # None without an anchor: the win rate over every training question
        app[a, Q[i][1]] += 1; app[a, Q[i][2]] += 1; win[a, rt(i)] += 1
    h3 = []
    for j, i in enumerate(held):
        a, r_, w_ = Q[i][0], rt(i), wr(i)
        if app[a, r_] and app[a, w_] and win[a, r_] * app[a, w_] != win[a, w_] * app[a, r_]:
            h3.append(float(win[a, r_] * app[a, w_] > win[a, w_] * app[a, r_]))
        else:
            h3.append(af[j])
    sset = set(m["shared"])
    uniq = [not ({x for x in Q[i][:3] if x is not None} & sset) for i in held]
    return {"held": held, "train": train, "heur_af": float(np.mean(af)), "heur_sh": float(np.mean(shr)),
            "masks": {"ansfreq_wrong": af_w, "shorter_wrong": sh_w, "hard": [a and b for a, b in zip(af_w, sh_w)]},
            "h3": h3, "uniq": uniq}


def kstep_split(m, split_seed):
    """A k-step domain's split re-derived (A3; a second implementation of ft36.main's make_split, not the trainer's
    off_split): ft36.split of the keys (40%); cyclic: held keys with k mod n = 0 given back; linear: held keys given back
    (the smallest offending key first, repeatedly) until every start item and every k of a held question is trained ->
    (held question indices, train indices) in question order."""
    import ft36
    Q = m["q"]
    held = set(ft36.split([q[3] for q in Q], ANC_HELD, split_seed))
    if m["cyclic"]:
        held = {h for h in held if h[1] != 0}
    else:
        while True:
            tr = [q for q in Q if q[3] not in held]
            xs, ks = {q[0] for q in tr}, {q[1] for q in tr}
            off = sorted({q[3] for q in Q if q[3] in held and (q[0] not in xs or q[1] not in ks)})
            if not off:
                break
            held.discard(off[0])
    return [i for i, q in enumerate(Q) if q[3] in held], [i for i, q in enumerate(Q) if q[3] not in held]


def elim_baseline(m, split_seed):
    """A3, descriptive (k-step domains): the Latin-square ELIMINATION shortcut: for a held 'k after x', choose uniformly
    among the items that are never the training answer after the same start x nor for the same shift (cyclic: k mod n;
    linear: k); its expected accuracy (0 when the right answer is eliminated) -- a non-geometric route to above-chance
    k-way accuracy."""
    held, train = kstep_split(m, split_seed)
    Q, n, cyc = m["q"], m["n"], m["cyclic"]
    sh = (lambda k: k % n) if cyc else (lambda k: k)
    by_x, by_k = {}, {}
    for i in train:
        x, k, a, _ = Q[i]
        by_x.setdefault(x, set()).add(a); by_k.setdefault(sh(k), set()).add(a)
    acc = []
    for i in held:
        x, k, a, _ = Q[i]
        cand = set(range(n)) - by_x.get(x, set()) - by_k.get(sh(k), set())
        acc.append(1 / len(cand) if a in cand else 0.0)
    return 100 * float(np.mean(acc)) if acc else None


def run_subsets(x, m):
    """One run, one comparison domain (x = rec['cmp'][d]): the sub-read-outs recomputed from the per-question hits of
    every eval, with the checks tying them to the trainer (table sha, the eval question order = the re-derived held order,
    masks and heuristic lines = the re-derived ones, the full / hard 2-way series = the logged ones at every eval, their
    areas = the metrics)."""
    probs = []
    if x["table_sha"] != m["sha"]:
        probs.append(f"table sha {x['table_sha']} != the snapshot's {m['sha']}")
    si = split_info(m, x["split_seed"])
    q, nq, nc = list(x["q"]), len(x["q"]), x["nctx"]
    if q != si["held"][:nq]:
        probs.append("eval questions != the re-derived held order")
    for k, v in x["masks"].items():
        if v != "".join("1" if b else "0" for b in si["masks"][k][:nq]):
            probs.append(f"mask {k} != the re-derived one")
    if x["heur"][0] is not None and (abs(x["heur"][0] - si["heur_af"]) > 1e-12 or abs(x["heur"][1] - si["heur_sh"]) > 1e-12):
        probs.append(f"heuristic lines {x['heur']} != the re-derived {[si['heur_af'], si['heur_sh']]}")
    if any(len(s) != nq or not s.isdigit() for s in x["hits"]):
        probs.append(f"hits2 strings are not {nq} digits at every eval")
        return None, probs
    H = np.stack([np.frombuffer(s.encode(), dtype=np.uint8) for s in x["hits"]]).astype(np.int64) - 48
    idx = {"all": np.arange(nq), "hard": np.flatnonzero(si["masks"]["hard"][:nq]), "uniq": np.flatnonzero(si["uniq"][:nq])}
    ser = {k: (H[:, v].sum(1) / (nc * len(v))) if len(v) else None for k, v in idx.items()}
    for k, lg in (("all", x["logged"]), ("hard", x["logged_hard"])):
        if ser[k] is not None and not all(L is not None and abs(a - L) < 1e-9 for a, L in zip(ser[k], lg)):
            probs.append(f"recomputed {k} 2-way series != the logged one")
        if ser[k] is None and any(L is not None for L in lg):
            probs.append(f"empty {k} subset but a logged series")
    area = {k: (100 * float(np.mean(v)) if v is not None else None) for k, v in ser.items()}
    fin3 = {k: (100 * float(np.mean(v[-3:])) if v is not None else None) for k, v in ser.items()}
    if x["area"] is not None and abs(area["all"] - x["area"]) > 1e-6:
        probs.append(f"recomputed area {area['all']} != metrics {x['area']}")
    if x["area_hard"] is not None and area["hard"] is not None and abs(area["hard"] - x["area_hard"]) > 1e-6:
        probs.append(f"recomputed hard area {area['hard']} != metrics {x['area_hard']}")
    h3 = np.asarray(si["h3"][:nq])
    return {"full": area["all"], "hard": area["hard"], "hard3": fin3["hard"], "uniq": area["uniq"], "uniq_final3": fin3["uniq"],
            "n_q": nq, "n_hard": len(idx["hard"]), "n_uniq": len(idx["uniq"]),
            "h3_held": 100 * float(h3.mean()), "h3_hard": 100 * float(h3[idx["hard"]].mean()) if len(idx["hard"]) else None}, probs


def subsets_all(R_, meta):
    """{(batch, arm, lr, seed): {d: run_subsets}} over every run with the per-question data; problems (strict refuses)."""
    subs, probs = {}, []
    for k, r in R_.items():
        for d, x in r.get("cmp", {}).items():
            s, p = run_subsets(x, meta[d])
            if s is not None:
                subs.setdefault(k, {})[d] = s
            probs += [f"{k[0]} {k[1]}@{k[2]} s{k[3]} {d}: {v}" for v in p]
    return subs, probs


def site_summary(meta):
    return {d: {"n_items": len(m["items"]), "n_shared": len(m["shared"]), "n_tokens": len(m["groups"]), "groups": m["groups"],
                "frac": len(m["shared"]) / len(m["items"])} for d, m in meta.items()}


# ------------------------------------------------------------------------------------------------------------ selection
def select_unit(R_, batch, arm, d, lrs, excluded=(), seeds=SEL):
    """F87 rule 1 on seeds for unit (arm, d) over lrs minus the gate-failed lrs `excluded`. Refuses ({'missing'}) when an
    lr lacks a valid run at a seed (excluded or not). No admissible lr -> pick None. -> pick, argmax (the first maximum of
    the mean area over the admissible lrs), means, the tie-break used."""
    cells, miss = {}, []
    for lr in lrs:
        rs = [R_.get((batch, arm, lr, s)) for s in seeds]
        if not all(r and r["valid"] for r in rs):
            miss += [f"{batch} {arm}@{lr} s{s}" for s, r in zip(seeds, rs) if not (r and r["valid"])]
            continue
        if lr not in excluded:
            cells[lr] = {"area": [r["area"][d] for r in rs], "off": [r["bpb_off"] for r in rs], "intr": [r["bpb_intr"] for r in rs]}
    if miss:
        return {"missing": miss}
    adm = sorted(cells, key=float)
    exc = sorted(set(excluded) & set(lrs), key=float)
    if not adm:
        return {"grid": [], "excluded": exc, "mean": {}, "pick": None, "near": [], "tie_break": None, "argmax": None}
    b = RR.best_lr(cells, near_rule="top2")
    k = max(range(len(adm)), key=lambda j: b["mean"][adm[j]])
    return {"grid": adm, "excluded": exc, "mean": {x: round(v, 4) for x, v in b["mean"].items()}, "pick": b["best"],
            "near": b["near"], "tie_break": ("offline" if b["offline"] else "in-training") if len(b["near"]) > 1 else None,
            "argmax": adm[k]}


def extension(arm, argmax):
    """A3 / A4: the first extension lr of a unit whose stage-1 area argmax is at a grid edge (A3: none at its lower edge only;
    A4: none at both edges)."""
    g = grid(arm)
    if argmax is None:
        return None
    side = "lower" if argmax == g[0] else "upper" if argmax == g[-1] else None
    return EXT_LR.get((kind(arm), side)) if side else None


def next_ext(arm, argmax, exts):
    """A4: the next extension of a unit from the extensions it has run (in order) and its current area argmax: the first
    step at a stage-1 grid edge (extension); none only: 5e-4 when 1e-3 (its first lower step) is still the argmax (EXT2);
    otherwise none (the prior and none's upper edge: one step)."""
    if not exts:
        return extension(arm, argmax)
    if kind(arm) == "none" and exts == [EXT_LR[("none", "lower")]] and argmax == exts[-1]:
        return EXT2[("none", "lower")]
    return None


def not_final(arm, u, exts):
    """A4: a none unit whose pick or area argmax sits at an outermost lr it ran (5e-4 / 1.6e-2) once its allowed steps are
    used: none may be better beyond, so the comparison is lr-confounded -> NOT FINAL (no verdict)."""
    if kind(arm) != "none" or not u or u.get("pick") is None:
        return False
    return any(lr in exts and lr in (u["pick"], u["argmax"]) for lr in OUTER["none"])


def gate_bad(gate):
    bad = {}
    for key in gate["failed"]:
        a_, l_ = key.split("@")
        bad.setdefault(a_, set()).add(l_)
    return bad


def shortcut(rows, n_min=SC_MIN):
    """A4: the shortcut condition on the POOLED hard-subset questions of the given seeds. rows = per seed (the prior's
    hard-subset final3 or None, the shortcut baseline's accuracy on that subset or None, n_hard); both accuracies pooled
    over the seeds weighted by n_hard (every question of a seed has the same number of contexts). Testable iff the pooled
    count >= n_min; then met iff the prior's pooled final3 > the baseline's. ok: True / False / None (not testable:
    recorded, never blocks; A3 failed an empty subset and judged on any count)."""
    use = [(p, h, n) for p, h, n in rows if n and p is not None and h is not None]
    N = int(sum(n for _, _, n in use))
    pp = float(sum(p * n for p, _, n in use) / N) if N else None
    hh = float(sum(h * n for _, h, n in use) / N) if N else None
    base = {"prior": pp, "base": hh, "n_pooled": N, "n_seeds": len(use), "n_hard": [n for _, _, n in rows], "n_min": n_min}
    if N < n_min:
        return dict(base, ok=None, testable=False, why=f"not testable: {N} pooled hard-subset questions < {n_min} (answer "
                                                         "frequency / length solve the others)")
    return dict(base, ok=pp > hh, testable=True, why=None if pp > hh else f"pooled hard-subset final3 {pp:.1f} <= the shortcut baseline {hh:.1f}")


def comp_rows(R_, subs, batch, arm_lr_prior, arm_lr_none, d, seeds):
    """Per seed (prior hard final3, baseline hard accuracy (from the none run's split = the same split), n_hard)."""
    out = []
    for s in seeds:
        rp = R_.get((batch,) + arm_lr_prior + (s,))
        sp = subs.get((batch,) + arm_lr_prior + (s,), {}).get(d)
        sn = subs.get((batch,) + arm_lr_none + (s,), {}).get(d)
        nh = rp["n_hard"].get(d) if rp else None
        base = (sn or sp or {}).get("h3_hard")
        out.append((rp["hard3"].get(d) if rp else None, base, nh))
    return out


def same_lr(R_, batch, d, lr=SAME_LR, seeds=SEL):
    """A4: the paired gain only-d@lr - none@lr per seed (both valid; the gate does not enter: the runs are data)."""
    out = []
    for s in seeds:
        a, b = R_.get((batch, f"only-{d}", lr, s)), R_.get((batch, "none", lr, s))
        out.append(a["area"][d] - b["area"][d] if a and b and a["valid"] and b["valid"] else None)
    return out


def screen(R_, batch, subs, meta=None):
    """Stage-1 decision of one batch (select_ptdom.py --batch X --after 1; the read-out re-derives it): the A2 gate per
    cell (seeds 0-3), the picks, the screen of every domain, the stage-2 cells (C: seeds 4-7 at the picks; E: the
    extensions, A4 at seeds 0-7). A4 adds (iv) the same-lr gain at 2e-3 > 0 in 4/4 seeds; (iii) pooled, 'not testable'
    never blocks. -> picks dict."""
    gate = aux_gate(R_, batch, SEL)
    bad = gate_bad(gate)
    units, doms, C, E = {}, {}, [], []
    for d in BATCHES[batch]:
        up = select_unit(R_, batch, f"only-{d}", d, grid(f"only-{d}"), bad.get(f"only-{d}", set()))
        un = select_unit(R_, batch, "none", d, grid("none"))
        for k_, u in ((f"only-{d}|{d}", up), (f"none|{d}", un)):
            if "missing" in u:
                raise SystemExit(f"selection refused: unit {batch} {k_} lacks valid runs {u['missing']}")
            units[k_] = u
        row = {"kind": "kstep" if d in KSTEP else "comp", "pick_prior": up["pick"], "pick_none": un["pick"],
               "argmax_prior": up["argmax"], "argmax_none": un["argmax"]}
        if up["pick"] is None:
            row.update(decision="GATE", why=["AUX GATE FAILED (every stage-1 cell of the prior arm fails: no pick)"], gain=None)
            doms[d] = row
            continue
        g = [R_[(batch, f"only-{d}", up["pick"], s)]["area"][d] - R_[(batch, "none", un["pick"], s)]["area"][d] for s in SEL]
        g3 = [R_[(batch, f"only-{d}", up["pick"], s)]["final3"][d] - R_[(batch, "none", un["pick"], s)]["final3"][d] for s in SEL]
        gs = same_lr(R_, batch, d)
        pg = gate["cells"].get(f"only-{d}@{up['pick']}")
        why = []
        if not all(v > 0 for v in g):
            why.append(f"paired gain positive in {sum(v > 0 for v in g)}/4 seeds (needs 4/4)")
        if not all(v is not None and v > 0 for v in gs):
            why.append(f"same-lr gain at {SAME_LR} positive in {sum(v is not None and v > 0 for v in gs)}/4 seeds (needs 4/4; A4)")
        if pg is None or not pg["ok"]:
            why.append("the aux gate fails at the picked cell")
        sc = None
        if d in COMP:
            sc = shortcut(comp_rows(R_, subs, batch, (f"only-{d}", up["pick"]), ("none", un["pick"]), d, SEL))
            if sc["ok"] is False:
                why.append("shortcut: " + sc["why"])
        ep, en = extension(f"only-{d}", up["argmax"]), extension("none", un["argmax"])
        sg = gate["cells"].get(f"only-{d}@{SAME_LR}")
        row.update(decision="ADVANCE" if not why else "NOT ADVANCE", why=why, gain=g, gain_mean=float(np.mean(g)),
                   gain_final3=float(np.mean(g3)), wins=sum(v > 0 for v in g), shortcut=sc,
                   same_lr=gs, same_lr_mean=float(np.mean(gs)) if all(v is not None for v in gs) else None,
                   same_lr_wins=sum(v is not None and v > 0 for v in gs), same_lr_gate_ok=sg["ok"] if sg else None,
                   gate_median=pg["median"] if pg else None, ext_prior=ep, ext_none=en,
                   prior_area=float(np.mean([R_[(batch, f"only-{d}", up["pick"], s)]["area"][d] for s in SEL])),
                   none_area=float(np.mean([R_[(batch, "none", un["pick"], s)]["area"][d] for s in SEL])),
                   chance=float(np.mean([R_[(batch, "none", un["pick"], s)]["chance"][d] for s in SEL])))
        # A5.3 (audit_a4 minor 6; descriptive, never a rule): a NOT ADVANCE where none's stage-1 argmax lies above the prior's
        # stage-1 grid (none prefers a higher lr than the prior was run at in stage 1; the prior's 4e-3 extension only comes
        # after advancing) is not evidence of no effect
        row["note_none_higher_lr"] = (row["decision"] == "NOT ADVANCE" and un["argmax"] is not None
                                      and float(un["argmax"]) > float(grid(f"only-{d}")[-1]))
        if row["decision"] == "ADVANCE":
            C += [(f"only-{d}", up["pick"]), ("none", un["pick"])]
            E += [(f"only-{d}", ep)] + ([("none", en)] if en else [])
        doms[d] = row
    uniq = lambda xs: sorted(set(xs), key=lambda t: (t[0], float(t[1])))
    return {"design": "ptdom", "amendment": "A5", "status": "decided", "batch": batch, "after": "1", "stage_written": "2",
            "units": units, "domains": doms, "C": [list(c) for c in uniq(C)], "E": [list(e) for e in uniq(E)],
            "ADVANCE": [d for d in BATCHES[batch] if doms[d]["decision"] == "ADVANCE"],
            "GATE": [d for d in BATCHES[batch] if doms[d]["decision"] == "GATE"],
            "aux_gate_failed_cells": gate["failed"], "aux_gate_arms_all_failed": gate["arms_all_failed"]}


def asked(batch, prev):
    """A4: {(arm, lr, seed)} asked so far: stage 1, every E (A5: and F) of the decisions prev at seeds 0-7, every C at seeds 4-7."""
    out = set(MJ.cells(batch, "1"))
    for p in prev:
        out |= {(a, lr, s) for a, lr in map(tuple, p["E"] + p.get("F", [])) for s in SEL + CONF}
        out |= {(a, lr, s) for a, lr in map(tuple, p["C"]) for s in CONF}
    return out


def dp_flags(R_, batch=None):
    """A5.1: the valid runs flagged by a GPU-numerics double-path tolerance (valid data, listed): [(key, [check names])]."""
    return [(k, [x["check"] for x in r["dp_flag"]]) for k, r in sorted(R_.items())
            if r.get("valid") and r.get("dp_flag") and (batch is None or k[0] == batch)]


def geo_flags(R_, batch=None):
    """A6: the valid runs flagged by a geo overshoot at a perfect fit (valid data, listed): [(key, [the overshoot evals])]."""
    return [(k, [f"geo_{r['geo_flag']['active']} {x['geo']:.6f} with cka_{r['geo_flag']['active']} {x['cka']:.6f} at step {x['step']}"
                 for x in r["geo_flag"]["evals"]]) for k, r in sorted(R_.items())
            if r.get("valid") and r.get("geo_flag") and (batch is None or k[0] == batch)]


A6_LINE = (f"FLAGGED (A6, valid data): {{n}} runs failed only the geo / grad_prior check (with A5.1's set) at a geo overshoot of a "
           f"perfect fit (geo_<active> in [{GEO_FLOOR}, {GEO_TOL}] with cka_<active> in [{GEO_CKA_MIN}, {GEO_CKA_MAX}]; grad_prior "
           f"finite > 0)")


def unit_history(prev, d, kind_):
    """A4: the extension lrs unit kind_ ('prior' / 'none') of domain d has been given by the decisions prev, in order (the
    screen's first step, then each later decision's 'next'); every asked extension has run before the next decision."""
    r1 = prev[0]["domains"][d]
    exts = [x for x in [r1["ext_prior" if kind_ == "prior" else "ext_none"]] if x]
    for p in prev[1:]:
        nx = p["final"][d][kind_]["next"]
        if nx:
            exts.append(nx)
    return exts


def chain_step(R_, batch, prev):
    """A4: the decision after round k = len(prev) + 1 (select_ptdom.py --after k): every advancing domain's two units
    re-selected over grid + the extensions they ran (F87 rule 1, seeds 0-3; a gate-failing prior cell excluded); E = the
    next extension where one is due (none: 5e-4 when 1e-3 is still the argmax), at seeds 0-7; C = a pick without seeds
    4-7 so far, at seeds 4-7; per unit EDGE (prior: the argmax is its extension), NOT FINAL (none: pick or argmax at an
    outermost lr it ran, no step left). Writes stage 2b (after 2) / 2c (after 3); complete when nothing is asked.
    A5 fixB (AUDIT.md A5.2; audit_a4 important 2; = the auditor's simulated fixB): once none's own extension chain for d is
    done (no next step) and the prior's final pick P (fixed from after 2 on: its one extension has run) lies outside the lr
    range none ran for d (its grid + its own extensions; other domains' extensions are not part of d's range, A2's rule),
    none is run at P at seeds 0-7 (F, unless every one of those cells was asked before, e.g. as another domain's extension)
    and, once it has, none is re-selected by F87 rule 1 (seeds 0-3) over its range + P: that pick is none's final pick for
    d; NOT FINAL also when P is an outermost lr (5e-4) and the re-selected pick or argmax is P (the simulated rule). The
    fixB lr never triggers a further extension. A pending fixB = no verdict for d."""
    k = len(prev) + 1
    p1 = prev[0]
    gate = aux_gate(R_, batch, SEL)
    bad = gate_bad(gate)
    done = asked(batch, prev)
    pf = prev[-1].get("final") if k > 2 else None
    final, E, C, F = {}, [], [], []
    for d in p1["ADVANCE"]:
        f = {}
        for kind_, arm in (("prior", f"only-{d}"), ("none", "none")):
            exts = unit_history(prev, d, kind_)
            lrs = sorted(set(grid(arm)) | set(exts), key=float)
            u = select_unit(R_, batch, arm, d, lrs, bad.get(arm, set()) if arm != "none" else set())
            if "missing" in u:
                raise SystemExit(f"selection refused: unit {batch} {arm}|{d} lacks valid runs {u['missing']}")
            nx = next_ext(arm, u["argmax"], exts)
            old = pf[d][kind_]["pick"] if pf else p1["domains"][d]["pick_prior" if kind_ == "prior" else "pick_none"]
            x = {"pick": u["pick"], "argmax": u["argmax"], "exts": exts, "next": nx, "moved": u["pick"] != old,
                 "edge": kind_ == "prior" and bool(exts) and u["argmax"] == exts[-1],
                 "not_final": nx is None and not_final(arm, u, exts), "unit": u,
                 "ext_excluded": [x_ for x_ in exts if x_ in u["excluded"]]}
            if kind_ == "none":   # A5 fixB
                pp = f["prior"]["pick"]
                need = nx is None and pp is not None and (float(pp) < float(lrs[0]) or float(pp) > float(lrs[-1]))
                x.update(range=[lrs[0], lrs[-1]], fixb_lr=pp if need else None, fixb=None, fixb_pending=False,
                         a4={"pick": u["pick"], "argmax": u["argmax"], "not_final": x["not_final"]})
                if need and all(("none", pp, s) in done for s in SEL + CONF):
                    ub = select_unit(R_, batch, arm, d, sorted(set(lrs) | {pp}, key=float))
                    if "missing" in ub:
                        raise SystemExit(f"selection refused: unit {batch} none|{d} (fixB @{pp}) lacks valid runs {ub['missing']}")
                    x.update(pick=ub["pick"], argmax=ub["argmax"], unit=ub, fixb=pp, moved=ub["pick"] != old,
                             not_final=x["not_final"] or (pp in OUTER["none"] and pp in (ub["pick"], ub["argmax"])))
                elif need:
                    x["fixb_pending"] = True
                    if ("none", pp) not in F:
                        F.append(("none", pp))
            f[kind_] = x
            if nx and (arm, nx) not in E and any((arm, nx, s) not in done for s in SEL + CONF):
                E.append((arm, nx))
            if x["pick"] is not None and (arm, x["pick"]) not in C and any((arm, x["pick"], s) not in done for s in CONF):
                C.append((arm, x["pick"]))
        final[d] = f
    F = [x for x in F if x not in E]   # a cell is asked once (E runs seeds 0-7 too)
    C = [x for x in C if x not in F]   # F runs seeds 4-7 too
    srt = lambda xs: [list(t) for t in sorted(xs, key=lambda t: (t[0], float(t[1])))]
    assert k < 4 or not (E or C or F), "the decision chain must end after stage 2c"
    return {"design": "ptdom", "amendment": "A5", "status": "decided", "batch": batch, "after": str(k),
            "stage_written": (MJ.STAGES[k] if (E or C or F) else None), "final": final, "E": srt(E), "C": srt(C), "F": srt(F),
            "complete": not E and not C and not F, "aux_gate_failed_cells": gate["failed"]}


def asks(batch, p):
    """The (arm, lr, seed) cells decision p asks for (E and A5's F at seeds 0-7, C at seeds 4-7)."""
    return ([(a, lr, s) for a, lr in map(tuple, p["E"] + p.get("F", [])) for s in SEL + CONF]
            + [(a, lr, s) for a, lr in map(tuple, p["C"]) for s in CONF])


def decide_batch(R_, batch, subs):
    """The read-out's own re-derivation of a batch's decision chain: the stage-1 screen, then a decision after every round
    whose asked cells are all present (valid). -> (chain [after1, after2, ...], complete, missing cells): complete when the
    last decision asks for nothing."""
    chain = [screen(R_, batch, subs)]
    while True:
        last = chain[-1]
        miss = [c for c in asks(batch, last) if not ((batch,) + c in R_ and R_[(batch,) + c]["valid"])]
        if miss:
            return chain, False, miss
        if not last["E"] and not last["C"] and not last.get("F"):
            return chain, True, []
        chain.append(chain_step(R_, batch, chain))


# ------------------------------------------------------------------------------------------------------------ decision
def boot_ci(g, n=N_BOOT, seed=0):
    g = np.asarray(g, float)
    idx = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    m = g[idx].mean(1)
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def t_test(g):
    g = np.asarray(g, float)
    n, mu, sd = len(g), float(g.mean()), float(g.std(ddof=1))
    if sd == 0:
        return mu, (0.0 if mu != 0 else 1.0), (mu, mu)
    t = mu / (sd / math.sqrt(n))
    p = float(2 * stats.t.sf(abs(t), n - 1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return mu, p, (mu - h, mu + h)


def sf_test(g):
    """A4: the selection-free test on seeds 4-7 (not used by any selection or screen step): the one-sided one-sample t p of
    H1 mean > 0 (df n - 1); fewer than 2 values -> 1; zero spread -> 0 if the mean is > 0, else 1."""
    g = np.asarray(g, float)
    if len(g) < 2:
        return 1.0
    mu, sd = float(g.mean()), float(g.std(ddof=1))
    if sd == 0:
        return 0.0 if mu > 0 else 1.0
    return float(stats.t.sf(mu / (sd / math.sqrt(len(g))), len(g) - 1))


def holm(ps):
    """Holm-adjusted p-values (step-down, monotone, capped at 1), in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def rule(g, p_holm, p_sf_holm, sf_alpha=None):
    """The pre-registered conditions for gains g (8 seeds, seed order 0..7) -> (bool PASS, details). A3: (a) the 8-seed
    bootstrap 90% CI excludes 0, (b) both halves positive, (c) the 8-seed two-sided t p (Holm) < 0.05, the mean > 0. A4
    adds (d) the selection-free condition: seeds 4-7 all positive and their one-sided t p (Holm over every stage-2
    domain) < SF_ALPHA."""
    sf_alpha = SF_ALPHA if sf_alpha is None else sf_alpha
    g = np.asarray(g, float)
    lo, hi = boot_ci(g)
    h1, h2 = float(np.mean(g[:4])), float(np.mean(g[4:]))
    a3 = lo > 0 and h1 > 0 and h2 > 0 and p_holm < ALPHA_T and float(np.mean(g)) > 0
    sf = len(g) == 8 and bool((g[4:] > 0).all()) and p_sf_holm < sf_alpha
    return a3 and sf, {"boot90": [lo, hi], "halves": [h1, h2], "a3_conditions": bool(a3), "selection_free": bool(sf),
                       "sf_all4": bool(len(g) == 8 and (g[4:] > 0).all())}


def verdict_text(v, shared_frac):
    if v == "NOT SHOWN" and shared_frac is not None and shared_frac >= SHARED_FRAC:
        return "NOT SHOWN with this site (many shared site tokens)"
    if v == "NOT SEPARABLE":
        return "NOT SEPARABLE from a training-frequency shortcut"
    return v


def readout(R_, subs, picks_files=(), allow_partial=False, sites=None, elim=None, batches=tuple(BATCHES), family=tuple(BATCHES),
            sprobs=()):
    """Integrity; per batch the re-derived decision chain (screen, after 2, after 3, ...) compared with the picks files;
    the verdicts of every advancing domain with Holm over all of them; the screen table of every domain; side effects.
    A4: a FINAL verdict needs every batch of `family` (the design: all four; a test may name its own, printed) read with a
    complete decision chain, every decision's picks file present and equal to the re-derivation; otherwise every verdict
    is PROVISIONAL (no PASS) with the reasons, and the Holm family is printed as incomplete (never shrunk silently).
    A5: the comparison sub-read-out problems (sprobs, only reachable with --allow_partial) make it PROVISIONAL too; a domain
    whose fixB none cells have not run gets no verdict ('NO VERDICT ... fixB'); the runs flagged by a GPU-numerics
    double-path tolerance (valid data) are listed. A6: the runs flagged by a geo overshoot at a perfect fit (valid data) are
    listed (out['geo_flags']); no rule reads either flag."""
    probs = integrity(R_)
    if probs and not allow_partial:
        raise SystemExit("integrity: " + "; ".join(probs))
    pk = [json.loads(Path(p).read_text()) for p in picks_files]
    out = {"integrity": probs, "picks_consistency": [], "screen": {}, "chain": {}, "domains": {}, "side": {}, "gates": {},
           "sites": sites or {}, "elim": elim or {}, "batches_read": [], "provisional": [], "family": list(family)}
    assert set(family) <= set(BATCHES), family
    rows, adv = {}, []
    for b in [b_ for b_ in BATCHES if b_ in batches or b_ in family]:   # A4: every batch of the family is read (or reported missing)
        if not any(k[0] == b for k in R_):
            out["provisional"].append(f"batch {b}: no run on disk (its screen and stage 2 are missing)")
            continue
        try:
            chain, complete, miss = decide_batch(R_, b, subs)
        except SystemExit as e:
            out["screen"][b] = {"error": str(e)}
            out["provisional"].append(f"batch {b}: {e}")
            continue
        p1 = chain[0]
        out["batches_read"].append(b)
        out["screen"][b] = p1
        out["chain"][b] = chain[1:]
        out["gates"][b] = aux_gate(R_, b, SEL)
        if not complete:
            out["provisional"].append(f"batch {b}: the decision chain stops after {len(chain)}: {len(miss)} asked cells missing "
                                      f"(e.g. {', '.join(f'{a}@{lr} s{s}' for a, lr, s in miss[:4])})")
        fb = {}
        for f in pk:
            if f.get("batch") == b:
                fb.setdefault(str(f.get("after")), []).append(f)
        for i, mine in enumerate(chain):
            k = str(i + 1)
            fs = fb.pop(k, [])
            if not fs:
                if mine["E"] or mine["C"] or mine.get("F") or k == "1":
                    out["picks_consistency"].append({"batch": b, "after": k, "agree": False, "differ": ["no picks file"]})
                continue
            f = fs[0]
            same = (len(fs) == 1 and f["C"] == mine["C"] and f["E"] == mine["E"] and f.get("F", []) == mine.get("F", [])
                    and f.get("amendment") == "A5" and (k != "1" or f["ADVANCE"] == mine["ADVANCE"]))
            dflt = lambda x: [] if x == "F" else None
            out["picks_consistency"].append({"batch": b, "after": k, "agree": same, "differ": [] if same else
                                             [x for x in ("C", "E", "F", "ADVANCE", "amendment") if f.get(x, dflt(x)) != mine.get(x, dflt(x))]
                                             or ["duplicate files"]})
        for k, fs in fb.items():   # a picks file beyond the re-derived chain
            out["picks_consistency"].append({"batch": b, "after": k, "agree": False, "differ": ["no such decision in the re-derivation"]})
        last = chain[-1] if len(chain) > 1 else None
        for d in p1["ADVANCE"]:
            r1 = p1["domains"][d]
            if last is None:
                rows[d] = {"batch": b, "pick_prior": r1["pick_prior"], "pick_none": r1["pick_none"], "edge": [], "not_final": False,
                           "fixb": None, "fixb_pending": False, "fixb_lr": None, "none_range": None}
            else:
                F_ = last["final"][d]
                rows[d] = {"batch": b, "pick_prior": F_["prior"]["pick"], "pick_none": F_["none"]["pick"],
                           "edge": ["prior"] if F_["prior"]["edge"] else [], "not_final": F_["none"]["not_final"],
                           "exts": {"prior": F_["prior"]["exts"], "none": F_["none"]["exts"]},
                           "fixb": F_["none"].get("fixb"), "fixb_pending": bool(F_["none"].get("fixb_pending")),
                           "fixb_lr": F_["none"].get("fixb_lr"), "none_range": F_["none"].get("range")}
            rows[d].update(complete=complete, screen=r1)
            adv.append(d)
    for x in out["picks_consistency"]:
        if not x["agree"]:
            out["provisional"].append(f"picks file batch {x['batch']} after {x['after']}: {', '.join(x['differ'])}")
    out["provisional"] = [v for v in out["provisional"] if v]
    if set(out["batches_read"]) != set(family):
        out["provisional"].append(f"batches read {out['batches_read']} != the Holm family's batches {list(family)}")
    out["sub_problems"] = list(sprobs)
    if sprobs:   # A5 (audit_a4 minor 3): with --allow_partial the comparison sub-read-out problems make the read-out PROVISIONAL
        out["provisional"].append(f"comparison sub-read-out problems ({len(sprobs)}; --allow_partial): " + "; ".join(sprobs[:3]))
    for d in adv:
        if rows[d]["fixb_pending"]:
            out["provisional"].append(f"{d}: {FIXB}: none has not run at the prior's final pick {rows[d]['fixb_lr']} (outside none's "
                                      f"range {rows[d]['none_range']} for {d}): no verdict until it has")
    out["dp_flags"] = [[list(k), v] for k, v in dp_flags(R_)]   # A5.1: valid data, flagged
    out["geo_flags"] = [[list(k), v] for k, v in geo_flags(R_)]   # A6: valid data, flagged
    final_ok = not out["provisional"] and not (probs and allow_partial)
    ps8, psf, psf_neg = [], [], []
    for d in adv:
        x = rows[d]
        b = x["batch"]
        P_ = [R_.get((b, f"only-{d}", x["pick_prior"], s)) for s in SEL + CONF]
        N_ = [R_.get((b, "none", x["pick_none"], s)) for s in SEL + CONF]
        have = [s for s, a, c in zip(SEL + CONF, P_, N_) if a and c and a["valid"] and c["valid"]]
        sel = {s: (R_[(b, f"only-{d}", x["pick_prior"], s)], R_[(b, "none", x["pick_none"], s)]) for s in have}
        g = np.array([sel[s][0]["area"][d] - sel[s][1]["area"][d] for s in have])
        g3 = np.array([sel[s][0]["final3"][d] - sel[s][1]["final3"][d] for s in have])
        g47 = np.array([sel[s][0]["area"][d] - sel[s][1]["area"][d] for s in have if s in CONF])
        g1 = (np.array([sel[s][0]["area1"][d] - sel[s][1]["area1"][d] for s in have]) if d in KSTEP and have
              and all(sel[s][0].get("area1", {}).get(d) is not None and sel[s][1].get("area1", {}).get(d) is not None for s in have) else None)
        cg = lambda ss: cell_gate([r for k, r in R_.items() if k[:3] == (b, f"only-{d}", x["pick_prior"]) and k[3] in ss and r.get("finished")])
        both = sorted({k[2] for k in R_ if k[:2] == (b, f"only-{d}")} & {k[2] for k in R_ if k[:2] == (b, "none")}, key=float)
        x.update(seeds=have, gain=g, gain_final3=g3, g47=g47, gain1=g1, prior=[sel[s][0]["area"][d] for s in have],
                 none=[sel[s][1]["area"][d] for s in have],
                 seen=[min(sel[s][0]["seen3"][d], sel[s][1]["seen3"][d]) for s in have],
                 chance=float(np.mean([sel[s][1]["chance"][d] for s in have])) if have else None,
                 heur=(np.mean([sel[s][1]["heur"][d] for s in have], 0).tolist() if d in COMP and have else None),
                 step0=float(np.mean([sel[s][1]["step0"][d] for s in have])) if have else None,
                 kshort=(float(np.mean([sel[s][1]["kshort"][d] for s in have])) if d in KSTEP and have
                         and all(sel[s][1].get("kshort", {}).get(d) is not None for s in have) else None),
                 pick_gate=cg(SEL + CONF), pick_gate_sel=cg(SEL), pick_gate_conf=cg(CONF),
                 same_lr={lr: [v for v in same_lr(R_, b, d, lr, SEL + CONF) if v is not None] for lr in both})
        if d in COMP:
            x["shortcut"] = shortcut(comp_rows(R_, subs, b, (f"only-{d}", x["pick_prior"]), ("none", x["pick_none"]), d, have))
            sp = [subs.get((b, f"only-{d}", x["pick_prior"], s), {}).get(d) for s in have]
            sn = [subs.get((b, "none", x["pick_none"], s), {}).get(d) for s in have]
            ok_ = have and all(sp) and all(sn)
            x["sub"] = None if not ok_ else {
                "h3_held": float(np.mean([v["h3_held"] for v in sn])), "n_uniq": float(np.mean([v["n_uniq"] for v in sn])),
                "n_q": float(np.mean([v["n_q"] for v in sn])),
                "uniq_gain": (float(np.mean([a["uniq"] - c["uniq"] for a, c in zip(sp, sn)]))
                              if all(a["uniq"] is not None and c["uniq"] is not None for a, c in zip(sp, sn)) else None)}
        ps8.append(t_test(g)[1] if len(g) >= 2 else 1.0)
        psf.append(sf_test(g47) if len(g47) == 4 else 1.0)
        psf_neg.append(sf_test(-g47) if len(g47) == 4 else 1.0)
    ph, phsf, phsfn = holm(ps8), holm(psf), holm(psf_neg)
    for i, d in enumerate(adv):
        x = rows[d]
        g = x["gain"]
        mu, p, tci = t_test(g) if len(g) >= 2 else (float("nan"), 1.0, (float("nan"),) * 2)
        res = {"batch": x["batch"], "pick_prior": x["pick_prior"], "pick_none": x["pick_none"], "n_seeds": len(g), "gain_area": mu,
               "per_seed": [round(float(v), 3) for v in g], "gain_final3": float(np.mean(x["gain_final3"])) if len(g) else None,
               "prior_area": float(np.mean(x["prior"])) if len(g) else None, "none_area": float(np.mean(x["none"])) if len(g) else None,
               "t_p": p, "t_p_holm": ph[i], "t95": list(tci), "wins": int((g > 0).sum()), "chance": x["chance"],
               "sf_p": psf[i], "sf_p_holm": phsf[i], "seeds_4_7_mean": float(np.mean(x["g47"])) if len(x["g47"]) else None,
               "seeds_4_7_wins": int((x["g47"] > 0).sum()), "heuristic_ansfreq_shorter": x["heur"], "step0_none": x["step0"],
               "tier": "L" if d in COMP else None, "pick_gate": x["pick_gate"], "pick_gate_sel": x["pick_gate_sel"],
               "pick_gate_conf": x["pick_gate_conf"], "shortcut": x.get("shortcut"), "sub": x.get("sub"), "edge": x["edge"],
               "not_final": x["not_final"], "exts": x.get("exts"), "holm_family": len(adv), "same_lr": x["same_lr"],
               "same_lr_screen": x["screen"].get("same_lr"),
               "gain_first_token_kway": float(np.mean(x["gain1"])) if x["gain1"] is not None and len(x["gain1"]) else None}
        gate_ok = x["pick_gate_sel"]["ok"] and x["pick_gate_conf"]["ok"]
        hok = False
        res.update(fixb=x["fixb"], fixb_pending=x["fixb_pending"], fixb_lr=x["fixb_lr"], none_range=x["none_range"])
        if x["fixb_pending"]:   # A5.2: never a verdict (not even a provisional one) before none has run at the prior's pick lr
            res.update(boot90=None, halves=None, verdict=f"NO VERDICT ({FIXB}: none not yet run at the prior's pick {x['fixb_lr']}, "
                                                         f"outside none's range {x['none_range']} for {d})")
        elif x["not_final"]:
            res.update(boot90=None, halves=None, verdict=NOT_FINAL)
        elif not gate_ok:
            res.update(boot90=None, halves=None, verdict=GATE_PICK_FAIL)
        elif len(g) == 8:
            ok, det = rule(g, ph[i], phsf[i])
            hok, _ = rule(-g, ph[i], phsfn[i])
            v = "PASS" if ok else "NOT SHOWN"
            res["three_conditions"] = det["a3_conditions"]
            res["conditions_met"] = ok
            if v == "PASS" and d in COMP and x["shortcut"]["ok"] is False:
                v = "NOT SEPARABLE"
            res.update(det, verdict=v)
        else:
            res.update(boot90=None, halves=None, verdict=f"PROVISIONAL n={len(g)} of 8")
        if not final_ok and res["verdict"] in ("PASS", "NOT SHOWN", "NOT SEPARABLE"):
            res["would_be"] = res["verdict"]
            res["verdict"] = "PROVISIONAL (" + res["verdict"] + " if final; see the PROVISIONAL reasons)"
        res["verdict_text"] = verdict_text(res["verdict"], (sites or {}).get(d, {}).get("frac"))
        # A4: FLOOR = both arms at or below the best known non-geometric line + 5: chance, the heuristic lines, the
        # training-frequency shortcut baseline on every held-out question (comparisons), the elimination baseline and the
        # shortest-span line of the full-span scoring (k-step)
        lines = [x["chance"] or 0] + (x["heur"] or []) + ([x["sub"]["h3_held"]] if x.get("sub") else []) + \
            ([(elim or {}).get(d)] if d in KSTEP and (elim or {}).get(d) is not None else []) + \
            ([x["kshort"]] if x.get("kshort") is not None else [])   # Rev G: the k-span's shortest-span line
        floor = max(lines) + 5
        tags = []
        if len(g) and res["prior_area"] <= floor and res["none_area"] <= floor:
            tags.append(f"FLOOR (both <= {floor:.1f})")
        if len(g) and res["none_area"] >= 95:
            tags.append("CEILING")
        if len(g) and min(x["seen"]) < 80:
            tags.append("SEEN < 80")
        if hok:
            tags.append("HARM")
        if x["edge"]:
            tags.append("EDGE (prior: the final argmax is its extension lr)")
        if x["fixb"]:
            tags.append(f"FIXB (A5.2: none also run at the prior's pick {x['fixb']}, outside its range {x['none_range']}; none "
                        f"re-selected including it)")
        if d in COMP and x.get("shortcut") and x["shortcut"]["ok"] is None:
            tags.append("SHORTCUT NOT TESTABLE (" + x["shortcut"]["why"] + ")")
        res["tags"] = tags
        res["floor_line"] = floor
        out["domains"][d] = res
    out["final"] = final_ok
    out["side"] = side_effects(R_, out)
    return out


def pair_diff(R_, a, b, f):
    """Paired a - b of f(run) over the seeds where both cells have a valid run (descriptive; A2: a diverged run is the
    worst, never dropped: prior diverged -> +inf, none diverged -> -inf, both -> 0; a missing value (None) is skipped)."""
    ss = [s for s in SEL + CONF if a + (s,) in R_ and b + (s,) in R_ and R_[a + (s,)]["valid"] and R_[b + (s,)]["valid"]]
    v, da, db = [], 0, 0
    for s in ss:
        x, y = f(R_[a + (s,)]), f(R_[b + (s,)])
        if x is None or y is None:
            continue
        ia, ib = not math.isfinite(x), not math.isfinite(y)
        da += ia; db += ib
        v.append(0.0 if ia and ib else math.inf if ia else -math.inf if ib else x - y)
    return {"n": len(v), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None,
            "wins": int(sum(x > 0 for x in v)), "n_div_a": da, "n_div_b": db}


def side_effects(R_, o):
    """Descriptive only, per batch and prior arm at its stage-1 pick: the same-lr LM row (prior - none at every lr both ran:
    2e-3, and 1e-3 when none's extension ran), vs none at the domain's pick (lr-confounded), the numbers-line area, the
    gradient split (FLAG at a seed's median LM share < 0.5)."""
    out = {}
    off = lambda r: 1e3 * r["bpb_off"] if r["bpb_off"] is not None else None
    for b in o["batches_read"]:
        p1 = o["screen"][b]
        none_lrs = sorted({k[2] for k in R_ if k[:2] == (b, "none")}, key=float)
        for d in BATCHES[b]:
            arm, lp = f"only-{d}", p1["domains"][d]["pick_prior"]
            if lp is None:
                continue
            P_, N_ = (b, arm, lp), (b, "none", p1["domains"][d]["pick_none"])
            both = sorted({k[2] for k in R_ if k[:2] == (b, arm)} & set(none_lrs), key=float)
            gs = [R_[P_ + (s,)]["grad"] for s in SEL + CONF if P_ + (s,) in R_ and R_[P_ + (s,)]["valid"]]
            ls = [x["share"] for x in gs if x["share"] is not None]
            n_ok = sum(x["n_ok"] for x in gs)
            out[f"{b}:{arm}"] = {
                "pick": lp, "bpb_off_x1e3_vs_none_pick": pair_diff(R_, P_, N_, off),
                "same_lr": {lr: {"bpb_off_x1e3": pair_diff(R_, (b, arm, lr), (b, "none", lr), off),
                                 "bpb_intr_x1e3": pair_diff(R_, (b, arm, lr), (b, "none", lr), lambda r: 1e3 * r["bpb_intr"])} for lr in both},
                "numbers_area_vs_none_pick": pair_diff(R_, P_, N_, lambda r: r["numbers"]),
                "grad": {"lm_share_median": med(ls), "lm_share_min": min(ls) if ls else None,
                         "frac_b_gt_a": sum(x["n_b_gt_a"] for x in gs) / n_ok if n_ok else None,
                         "frac_clip_lt1": sum(x["n_clip_lt1"] for x in gs) / n_ok if n_ok else None,
                         "FLAG_prior_dominates": bool(ls and min(ls) < AUX_MIN)}}
    return out


f1 = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"
s1 = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.1f}"


def screen_report(p1, sites=None, elim=None):
    """The stage-1 screen table of one batch (every domain)."""
    b = p1["batch"]
    L = [f"stage-1 screen, batch {b} (A3 + A4, pre-registered: ADVANCE iff the paired gain > 0 in 4/4 seeds, the same-lr gain at "
         f"{SAME_LR} > 0 in 4/4 seeds (A4), the aux gate passes at the picked cell, and (comparisons) the prior's pooled hard-subset "
         f"final3 beats the training-frequency shortcut baseline when >= {SC_MIN} pooled questions (A4; else not testable, not blocking))"]
    for d in BATCHES[b]:
        r = p1["domains"][d]
        tier = "[Tier L] " if d in COMP else "[k-step] "
        if r["decision"] == "GATE":
            L.append(f"  {d:15s} GATE {tier}{'; '.join(r['why'])}")
            continue
        sc = r.get("shortcut")
        st = (sites or {}).get(d)
        scs = ""
        if sc:
            scs = (f"; shortcut: prior pooled hard final3 {f1(sc['prior'])} vs baseline {f1(sc['base'])} on {sc['n_pooled']} pooled "
                   f"questions (n_hard per seed {sc['n_hard']})" + (" NOT TESTABLE" if sc["ok"] is None else ""))
        L.append(f"  {d:15s} {r['decision']:11s} {tier}only-{d}@{r['pick_prior']} (argmax {r['argmax_prior']}) vs none@{r['pick_none']} "
                 f"(argmax {r['argmax_none']}): gain {s1(r['gain_mean'])} per seed {[round(v, 2) for v in r['gain']]} wins {r['wins']}/4; "
                 f"same lr @{SAME_LR} {s1(r['same_lr_mean'])} wins {r['same_lr_wins']}/4"
                 + ("" if r.get("same_lr_gate_ok") in (None, True) else " (only-d@2e-3 fails the gate)")
                 + f"; prior {f1(r['prior_area'])}, none {f1(r['none_area'])}, chance {f1(r['chance'])}; final3 gain {s1(r['gain_final3'])}; "
                 f"gate median {('%.3f' % r['gate_median']) if r['gate_median'] is not None else '-'}"
                 + scs + (f"; elimination baseline {f1((elim or {}).get(d))}" if d in KSTEP and elim else "")
                 + (f"; site tokens shared by {st['n_shared']}/{st['n_items']} items" if st else "")
                 + (f"; stage 2: extensions prior {r['ext_prior']}, none {r['ext_none'] or '-'} (seeds 0-7)" if r["decision"] == "ADVANCE" else "")
                 + (f"  [{'; '.join(r['why'])}]" if r["why"] else "")
                 + (f"  NOTE (A5.3): none's stage-1 argmax {r['argmax_none']} is above the prior's stage-1 grid: this NOT ADVANCE "
                    "is not evidence of no effect" if r.get("note_none_higher_lr") else ""))
    L.append(f"  -> ADVANCE: {p1['ADVANCE'] or 'none'}; stage 2 cells: C (seeds 4-7) {p1['C']}, E (seeds 0-7) {p1['E']}; GATE {p1['GATE']}")
    L.append(f"  (A5.3) NOT ADVANCE = no reliable stage-1 gain under the lenient screen. Where none prefers a higher lr than the "
             f"prior's stage-1 grid ({' / '.join(GRID1['prior'])}; the prior's 4e-3 extension runs only after advancing), a NOT "
             f"ADVANCE is NOT evidence of no effect (a shared trend towards high lr pushes the screened gain down).")
    return "\n".join(L)


def chain_report(b, chain):
    """A4: the decisions after stage 2 / 2b / 2c of one batch."""
    L = []
    for p in chain:
        L.append(f"decision after round {p['after']} of batch {b}: " + (f"stage {p['stage_written']}: E (seeds 0-7) {p['E']}, C (seeds 4-7) "
                                                                       f"{p['C']}, F ({FIXB}: none at a prior's pick outside none's range, "
                                                                       f"seeds 0-7) {p.get('F', [])}" if not p["complete"]
                                                                       else "complete (nothing more to run)"))
        for d, f in p["final"].items():
            for kd in ("prior", "none"):
                x = f[kd]
                fb = ""
                if kd == "none" and x.get("fixb_lr"):
                    fb = (f" {FIXB}: the prior's pick {x['fixb_lr']} is outside none's range {x['range']} -> "
                          + (f"none@{x['fixb_lr']} asked (F; no verdict until it has run)" if x["fixb_pending"] else
                             f"none re-selected including {x['fixb']} (A4 pick {x['a4']['pick']}, argmax {x['a4']['argmax']})"))
                L.append(f"    {d:15s} {kd:5s} pick {x['pick']} argmax {x['argmax']} extensions {x['exts'] or '-'}"
                         + (f" next {x['next']}" if x["next"] else "") + (" MOVED" if x["moved"] else "") + (" EDGE" if x["edge"] else "")
                         + (" NOT FINAL" if x["not_final"] else "") + (f" (gate-excluded {x['ext_excluded']})" if x["ext_excluded"] else "") + fb)
    return "\n".join(L)


def report(o):
    fam = o.get("family", list(BATCHES))
    L = ["ptdom read-out (A3 + A4, pre-registered: runs/ptdom/AUDIT.md Amendments A3 / A4; area = primary, final3 secondary; "
         f"batches read {o['batches_read']}; Holm family = every advancing domain of {fam}"
         + ("" if list(fam) == list(BATCHES) else " (A TEST FAMILY, not the design's)") + ")"]
    if not o.get("final"):
        L.append("PROVISIONAL: no final verdict (no PASS) until every batch's decision chain is complete and agrees with its picks "
                 "files; reasons: " + ("; ".join(o["provisional"]) or "integrity problems (allow_partial)"))
    if o["integrity"]:
        L.append("INTEGRITY: " + "; ".join(o["integrity"]))
    if o.get("sub_problems"):   # A5 (audit_a4 minor 3)
        L.append(f"COMPARISON SUB-READ-OUT PROBLEMS ({len(o['sub_problems'])}; --allow_partial; PROVISIONAL): " + "; ".join(o["sub_problems"][:10]))
    if o.get("dp_flags"):   # A5.1
        L.append(f"FLAGGED (A5.1, valid data): {len(o['dp_flags'])} runs exceed a GPU-numerics double-path tolerance: "
                 + "; ".join(f"{k[0]} {k[1]}@{k[2]} s{k[3]} ({', '.join(v)})" for k, v in o["dp_flags"][:20]))
    if o.get("geo_flags"):   # A6
        L.append(A6_LINE.format(n=len(o["geo_flags"])) + ": "
                 + "; ".join(f"{k[0]} {k[1]}@{k[2]} s{k[3]} ({', '.join(v)})" for k, v in o["geo_flags"][:20]))
    for x in o["picks_consistency"]:
        L.append(f"picks consistency (batch {x['batch']}, after {x['after']}): {'agree' if x['agree'] else 'DIFFER ' + str(x['differ'])}")
    for b in o["batches_read"]:
        L.append(gate_report(o["gates"][b], "auxiliary-loss gate (A2 (i), seeds 0-3)"))
        L.append(screen_report(o["screen"][b], o.get("sites"), o.get("elim")))
        if o["chain"].get(b):
            L.append(chain_report(b, o["chain"][b]))
    L.append(f"stage-2 verdicts (Holm over the {len(o['domains'])} advancing domains of the batches read; A4: PASS needs the A3 "
             f"conditions AND seeds 4-7 all positive with one-sided t p_Holm < {SF_ALPHA})")
    for d, r in o["domains"].items():
        b = r.get("boot90")
        tier = "[Tier L] " if d in COMP else "[k-step] "
        L.append(f"{d:15s} {r['verdict_text']} {tier}batch {r['batch']}: only-{d}@{r['pick_prior']} vs none@{r['pick_none']}: gain "
                 f"{s1(r['gain_area'])} (prior {f1(r['prior_area'])}, none {f1(r['none_area'])}; chance {f1(r['chance'])}) n={r['n_seeds']} "
                 f"wins {r['wins']}; boot90 {('[%+.1f, %+.1f]' % tuple(b)) if b else '-'}; halves "
                 f"{('%+.1f / %+.1f' % tuple(r['halves'])) if r.get('halves') else '-'}; t95 [{s1(r['t95'][0])}, {s1(r['t95'][1])}] "
                 f"p {r['t_p']:.4f} Holm {r['t_p_holm']:.4f}; seeds 4-7 alone {s1(r['seeds_4_7_mean'])} wins {r['seeds_4_7_wins']}/4 "
                 f"one-sided p {r['sf_p']:.4f} Holm {r['sf_p_holm']:.4f}; final3 {s1(r['gain_final3'])}"
                 + (f"  [{', '.join(r['tags'])}]" if r["tags"] else ""))
        L.append(f"                per seed {r['per_seed']}")
        sl = "; ".join(f"@{lr} {s1(float(np.mean(v)))} ({sum(x_ > 0 for x_ in v)}/{len(v)})" for lr, v in r["same_lr"].items() if v)
        L.append(f"                same-lr gain (A4; lr-matched, the screen's condition at {SAME_LR} on seeds 0-3: "
                 f"{[round(v, 2) if v is not None else None for v in (r['same_lr_screen'] or [])]}): {sl or '-'}"
                 + (f"; first-token k-way gain (secondary) {s1(r['gain_first_token_kway'])}" if d in KSTEP else ""))
        pg, ps_, pc_ = r["pick_gate"], r["pick_gate_sel"], r["pick_gate_conf"]
        fm_ = lambda g_: ("%.3f" % g_["median"]) if g_["median"] is not None else "-"
        L.append(f"                aux gate at the pick: seeds 0-3 {fm_(ps_)}{'' if ps_['ok'] else ' FAIL'}, seeds 4-7 {fm_(pc_)}"
                 f"{'' if pc_['ok'] else ' FAIL'} (A4: each half separately), all {fm_(pg)}"
                 + ("" if ps_["ok"] and pc_["ok"] else "  " + "; ".join(ps_["why"] + pc_["why"])))
        if d in COMP and r.get("shortcut"):
            sc = r["shortcut"]
            L.append(f"                [Tier L] shortcut condition (pooled, 8 seeds): prior hard final3 {f1(sc['prior'])} vs baseline "
                     f"{f1(sc['base'])} on {sc['n_pooled']} questions: "
                     + ("NOT TESTABLE (< %d)" % sc["n_min"] if sc["ok"] is None else "met" if sc["ok"] else "NOT met (" + str(sc["why"]) + ")"))
    L.append("A PASS = 'the prior with this target helps' (no dose-matched shuffled control in this screen): not geometry-specific; "
             "comparison domains: Tier L (the target carries the answers: 'given the structure, the model uses it'). k-step domains "
             "(A4): the full-span k-way choice; a single-token item's candidate is the prior site's own token, and with tied "
             "embeddings the prior can move that output row directly (not so for a multi-token name).")
    L.append("side effects (descriptive; the same-lr row is lr-matched, the rest lr-confounded)")
    fm = lambda p: "-" if not p or p["mean"] is None else f"{p['mean']:+.2f} ({p['wins']}/{p['n']})"
    for k, x in o["side"].items():
        G_ = x["grad"]
        L.append(f"  {k:28s}@{x['pick']}: offline bpb x1e3 vs none@pick {fm(x['bpb_off_x1e3_vs_none_pick'])}; same lr "
                 + ("; ".join(f"@{lr} {fm(v['bpb_off_x1e3'])} (in-training {fm(v['bpb_intr_x1e3'])})" for lr, v in x["same_lr"].items()) or "-")
                 + f"; numbers area {fm(x['numbers_area_vs_none_pick'])}; LM share median "
                 + (f"{G_['lm_share_median']:.3f}" if G_["lm_share_median"] is not None else "-")
                 + ("  FLAG: the prior dominates the update in a seed" if G_["FLAG_prior_dominates"] else ""))
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------ self-test
def fake_run(root, batch, arm, lr, seed, area, final3=None, seen=99.0, numbers=20.0, bpb=1.2, chk_fail=(), evals=N_EVALS,
             lm_share=0.9, clip=0.5, cos=0.0, logged_off=0.0, missing=(), rr=0, diverge_at=None, hard3=None, n_hard=40):
    """A minimal run dir with the fields load() reads (area: {domain: value} for the batch's six). On the window evals
    (3 < i < 30) of a prior arm: grad_lm a = 1, grad_prior b = 1 / lm_share - 1 (a / (a + b) = lm_share), grad_cos cos, the
    logged grad_lm_share = a / sqrt(a^2 + b^2 + 2ab cos) + logged_off, clip_coef clip; the fields in `missing` are absent
    there. hard3: {d: hard-subset final3} (default final3 - 3); n_hard: the hard subset's size (0: empty; a dict per domain
    allowed). A4: the k-step domains also get the first-token k-way area (READ1 = area - 1)."""
    doms = BATCHES[batch]
    tag = MJ.tag_of(batch, arm, lr)
    d = Path(root) / (f"{tag}-s{seed}" + (f"-r{rr}" if rr else ""))
    (d / tag).mkdir(parents=True, exist_ok=True)
    f = d / tag / f"run_whole_none_seed{seed}_{tag}.json"
    on = arm != "none"
    gp = 1.0 / lm_share - 1.0
    lg = 1.0 / math.sqrt(max(1.0 + gp * gp + 2 * gp * cos, 1e-24)) + logged_off
    win = lambda i: on and 3 < i < 30
    curve = [{"step": i, "lm": (math.nan if diverge_at is not None and i >= diverge_at else 11.6 + 1e-4 * seed), "val_loss": 4.0,
              "geo": 0.0, "data_sha": f"sha{batch}{seed}-{i}", "val_bits_per_byte": bpb + 0.001, "prior_on": int(win(i)),
              "grad_lm_share": lg if win(i) else 1.0, "clip_coef": clip if win(i) else 1.0, "grad_lm": 1.0,
              "grad_prior": gp if win(i) else None, "grad_cos": cos if win(i) else None,
              **{EVAL_KEY[x]: 0.5 for x in doms}} for i in range(evals)]
    for e in curve:
        if e["prior_on"]:
            for k in missing:
                e.pop(k, None)
    f3 = final3 or area
    nh = n_hard if isinstance(n_hard, dict) else {x: n_hard for x in doms}
    met = {READ[x]: {"area": area[x], "final3": f3[x]} for x in doms}
    met.update({READ1[x]: {"area": area[x] - 1.0, "final3": f3[x] - 1.0} for x in doms if x in KSTEP})
    met.update({SEEN[x]: {"final3": seen} for x in doms}, numbers={"area": numbers})
    met.update({f"{x}_readout_hard": ({"area": area[x] - 5.0, "final3": (hard3 or {}).get(x, f3[x] - 3.0)} if nh.get(x, 40) else {})
                for x in doms if x in COMP})
    dm = {x: {"chance_readout": (1 / 12 if x in KSTEP else 0.5), "heuristic_ansfreq": 0.55, "heuristic_shorter": 0.52,
              "n_hard": nh.get(x, 40), "split_seed": 7} for x in doms}
    f.write_text(json.dumps({"code_sha": "c0de", "args": {"lr": float(lr), "seed": seed}, "curve": curve, "metrics": met, "domains": dm}))
    checks = [{"check": "run args = intent", "ok": True}] + [{"check": c, "ok": False} for c in chk_fail]
    (d / "check.json").write_text(json.dumps({"pass": not chk_fail, "checks": checks}))
    (d / "offline_lm.json").write_text(json.dumps({str(f.resolve()): {"bpb_full": bpb, "max_windows": 0, "orig_minus_logged": 0.0}}))


A6_CHECK_NAME = GEO_FLAG_CHECK + " (before a divergence)"   # A6: the geo / grad_prior check's name as CHECK_PY renders it


def patch_a6(d, batch, arm, lr, seed, geo=-0.00115, cka=1.0, at=(10,), gp_zero_at=(), geo_nan_at=(), fail=None, keep=A6_KEEP,
             inactive=None, active=None, extra_active=None):
    """Selftest / plumbing (A6): turn an existing fake_run dir d of a prior arm into a run with the active prior's fields:
    domains.<d>.prior and args.prior_domains = the arm's domain; geo_<d> 0.3 and cka_<d> 0.7 at every eval except (geo, cka)
    at the evals `at` (the window is 3 < i < 30); grad_prior 0 at gp_zero_at; geo_<d> NaN at geo_nan_at; geo_<inactive>
    logged at eval 10. check.json: the separate checks `keep` (A6_KEEP names) passed, and `fail` failed (default: the geo /
    grad_prior check as CHECK_PY names it). A6.1 (audit_a6 minor 8): active = the domain written as the active prior (default
    the arm's: another one = an arm-name mismatch); extra_active = a second active prior (domains.<x>.prior and
    args.prior_domains)."""
    fail = (A6_CHECK_NAME,) if fail is None else fail
    dom = active or arm[5:]
    tag = MJ.tag_of(batch, arm, lr)
    f = Path(d) / tag / f"run_whole_none_seed{seed}_{tag}.json"
    j = json.loads(f.read_text())
    j["args"]["prior_domains"] = dom + (f",{extra_active}" if extra_active else "")
    for x, v in j["domains"].items():
        v["prior"] = x in (dom, extra_active)
    for i, e in enumerate(j["curve"]):
        e[f"geo_{dom}"], e[f"cka_{dom}"] = (geo, cka) if i in at else (0.3, 0.7)
        if i in gp_zero_at:
            e["grad_prior"] = 0.0
        if i in geo_nan_at:
            e[f"geo_{dom}"] = math.nan
        if inactive and i == 10:
            e[f"geo_{inactive}"] = 0.2
    f.write_text(json.dumps(j))
    ck = json.loads((Path(d) / "check.json").read_text())
    ck["checks"] = ck["checks"][:1] + [{"check": p + " (...)", "ok": True} for p in keep] + ck["checks"][1:] + [
        {"check": x, "ok": False} for x in fail]
    ck["pass"] = all(k["ok"] for k in ck["checks"])
    (Path(d) / "check.json").write_text(json.dumps(ck))


def fake_a6(root, batch, arm, lr, seed, area, chk_fail=(), rr=0, **a6):
    """Selftest / plumbing (A6): fake_run (other failed checks chk_fail) + patch_a6 (its keyword arguments a6)."""
    fake_run(root, batch, arm, lr, seed, area, chk_fail=chk_fail, rr=rr)
    patch_a6(Path(root) / (f"{MJ.tag_of(batch, arm, lr)}-s{seed}" + (f"-r{rr}" if rr else "")), batch, arm, lr, seed, **a6)


def plant_subs(R_, h3_hard=None):
    """Selftest: planted comparison sub-read-outs for every run (the real ones come from subsets_all): baseline h3 60 / hard
    40 unless h3_hard[d] is given."""
    out = {}
    for k, r in R_.items():
        if "area" not in r:
            continue
        out[k] = {d: {"h3_held": 60.0, "h3_hard": (h3_hard or {}).get(d, 40.0), "n_uniq": 500, "n_q": 1200, "uniq": r["area"][d]}
                  for d in BATCHES[k[0]] if d in COMP}
    return out


def _raises(f, text):
    try:
        f()
    except SystemExit as e:
        return text in str(e)
    return False


def selftest():
    """A4 + A5 known answers (the A3 self-test is kept in superseded_A3/; A5: fixB with a reused and with new none cells (F, NO
    VERDICT while pending, NOT FINAL / PASS after), the generator's F validation, a differing F, --allow_partial sub-read-out
    problems -> PROVISIONAL, the A5.3 note, the A5.1 loader rule): Holm; the extension rules (both none edges, the second
    lower step, NOT FINAL); the pooled shortcut condition with 'not testable'; the selection-free test; the verdict rule with
    the seeds 4-7 condition; a planted two-batch design through screen -> stage 2 -> after 2 -> stage 2b -> after 3 -> the
    read-out (the same-lr condition, a moved none pick needing 2b seeds, a none argmax still at 5e-4 / 1.6e-2 -> NOT FINAL, the
    prior's extension winning (EDGE), A3 PASS conditions with a negative confirm seed -> NOT SHOWN, NOT SEPARABLE on the pooled 8
    seeds, PASS with the shortcut not testable, the gate at the pick judged per half); PROVISIONAL for a missing batch, a
    missing / differing picks file and missing stage-2b cells; the generator's stage 2 / 2b cells; the audit_a1 gate
    counter-examples; the loader and integrity."""
    ok = []

    def T(name, cond):
        ok.append((name, bool(cond)))
        print(("PASS " if cond else "FAIL ") + name)

    def stepdown(ps, alpha):
        order, rej = sorted(range(len(ps)), key=lambda i: ps[i]), set()
        for j, i in enumerate(order):
            if ps[i] > alpha / (len(ps) - j):
                break
            rej.add(i)
        return rej
    rs_, agree = np.random.default_rng(3), True
    for _ in range(2000):
        ps = (10 ** rs_.uniform(-5, 0, 9)).tolist()
        adj = holm(ps)
        agree &= all({i for i in range(9) if adj[i] <= al} == stepdown(ps, al) for al in (0.01, 0.05, 0.1))
        agree &= all(a_ >= p_ for a_, p_ in zip(adj, ps))
    T("Holm adjusted p (<= alpha) = the step-down rule at alpha .01 / .05 / .10 on 2000 random 9-vectors; adjusted >= raw", agree)
    T("Holm known answer (by hand): [.01, .04, .03, .2, .001, .5] -> [.05, .12, .12, .4, .006, .5]",
      np.allclose(holm([0.01, 0.04, 0.03, 0.2, 0.001, 0.5]), [0.05, 0.12, 0.12, 0.4, 0.006, 0.5]))
    # the extension rules (A4)
    T("extension (first step): prior 1e-3 -> 5e-4, 2e-3 -> 4e-3; none 2e-3 -> 1e-3, 8e-3 -> 1.6e-2 (A4), 4e-3 -> none; no pick -> none",
      (extension("only-x", "1e-3"), extension("only-x", "2e-3"), extension("none", "2e-3"), extension("none", "8e-3"),
       extension("none", "4e-3"), extension("only-x", None)) == ("5e-4", "4e-3", "1e-3", "1.6e-2", None, None))
    T("next_ext: none after 1e-3 with the argmax still 1e-3 -> 5e-4; argmax 2e-3 -> none; after 1e-3 + 5e-4 -> none; after 1.6e-2 "
      "-> none; the prior after its one step -> none",
      (next_ext("none", "1e-3", ["1e-3"]), next_ext("none", "2e-3", ["1e-3"]), next_ext("none", "5e-4", ["1e-3", "5e-4"]),
       next_ext("none", "1.6e-2", ["1.6e-2"]), next_ext("only-x", "4e-3", ["4e-3"]), next_ext("none", "2e-3", [])) ==
      ("5e-4", None, None, None, None, "1e-3"))
    U = lambda p, a: {"pick": p, "argmax": a}
    T("not_final: none pick or argmax at 5e-4 (ran) / 1.6e-2 (ran) -> True; at 1e-3 or 8e-3 -> False; the prior never",
      (not_final("none", U("5e-4", "5e-4"), ["1e-3", "5e-4"]), not_final("none", U("5e-4", "1e-3"), ["1e-3", "5e-4"]),
       not_final("none", U("1e-3", "5e-4"), ["1e-3", "5e-4"]), not_final("none", U("1e-3", "1e-3"), ["1e-3", "5e-4"]),
       not_final("none", U("1.6e-2", "8e-3"), ["1.6e-2"]), not_final("none", U("8e-3", "8e-3"), ["1.6e-2"]),
       not_final("only-x", U("5e-4", "5e-4"), ["5e-4"])) == (True, True, True, False, True, False, False))
    # the pooled shortcut condition (A4)
    sc1 = shortcut([(72.0, 70.0, 20), (71.0, 69.0, 18), (None, None, 0), (75.0, 71.0, 22)])
    sc2 = shortcut([(None, None, 0)] * 4)
    sc3 = shortcut([(80.0, 40.0, 2), (90.0, 40.0, 0), (85.0, 40.0, 5), (None, None, 0)])
    sc4 = shortcut([(60.0, 70.0, 10), (65.0, 68.0, 12), (70.0, 69.0, 8), (69.0, 70.0, 10)])
    T(f"shortcut (pooled): 60 questions, pooled 72.8 vs 70.1 -> met; every subset empty -> NOT TESTABLE (ok None, A3: failed); 7 "
      f"pooled questions -> NOT TESTABLE; 40 questions, 65.8 <= 69.3 -> not met",
      sc1["ok"] is True and sc1["n_pooled"] == 60 and abs(sc1["prior"] - (72 * 20 + 71 * 18 + 75 * 22) / 60) < 1e-9
      and sc2["ok"] is None and sc3["ok"] is None and sc3["n_pooled"] == 7 and sc4["ok"] is False and sc4["testable"]
      and abs(sc4["base"] - (70 * 10 + 68 * 12 + 69 * 8 + 70 * 10) / 40) < 1e-9)
    # the selection-free test and the verdict rule (A4)
    p_ref = float(stats.ttest_1samp([1.0, 2.0, 3.0, 4.0], 0.0, alternative="greater").pvalue)
    T(f"sf_test = scipy's one-sided one-sample t ({p_ref:.4f}); zero spread positive -> 0; < 2 values -> 1",
      abs(sf_test([1.0, 2.0, 3.0, 4.0]) - p_ref) < 1e-12 and sf_test([2.0] * 4) == 0.0 and sf_test([1.0]) == 1.0)
    G = {"pos": np.array([20, 22, 18, 21, 19, 23, 20, 21.0]), "zero": np.array([1.0, -1.2, 0.4, -0.3, 0.8, -0.9, 0.2, -0.5]),
         "halves": np.array([9.0, 8.0, 10.0, 9.5, -1.0, -2.0, -0.5, -1.5]), "f98": np.array([0.4, 4.3, 2.0, 2.1, 0.2, -1.6, 1.3, 1.2]),
         "neg1": np.array([4.0, 5.0, 4.0, 6.0, 3.0, 4.0, -0.5, 5.0]), "noisy": np.array([6.0, 5.0, 7.0, 6.0, 0.2, 9.0, 0.1, 0.3]),
         "cx": np.array([6.0, 5.0, 7.0, 6.0, 0.1, 0.2, 0.1, 0.3])}
    pr = lambda k: (t_test(G[k])[1], sf_test(G[k][4:]))
    T("rule: a consistent positive -> PASS; zero -> fails; halves disagree -> fails; the F98 case (t interval includes 0) -> fails",
      rule(G["pos"], *pr("pos"))[0] and not rule(G["zero"], *pr("zero"))[0] and not rule(G["halves"], *pr("halves"))[0]
      and not rule(G["f98"], *pr("f98"))[0])
    T(f"rule (A4): the A3 conditions hold but a seeds 4-7 gain is negative -> fails (selection-free condition); seeds 4-7 all "
      f"positive but noisy (one-sided p {pr('noisy')[1]:.3f} >= {SF_ALPHA}) -> fails; audit_a3's cx_halves case (seeds 4-7 "
      f"+0.1..+0.3, one-sided p {pr('cx')[1]:.4f}) passes alone and fails at Holm over 9 domains when its rank gives x 9",
      rule(G["neg1"], *pr("neg1"))[1]["a3_conditions"] and not rule(G["neg1"], *pr("neg1"))[0]
      and rule(G["noisy"], *pr("noisy"))[1]["a3_conditions"] and not rule(G["noisy"], *pr("noisy"))[0]
      and rule(G["cx"], *pr("cx"))[0] and not rule(G["cx"], pr("cx")[0], min(1.0, 9 * pr("cx")[1]))[0])
    ps9 = [t_test(G["pos"])[1]] + [0.5] * 8
    T("Holm across batches on the 8-seed t (A3 kept) and on the selection-free p (A4)", rule(G["pos"], holm(ps9)[0], holm([pr("pos")[1]] + [0.5] * 8)[0])[0])
    # ---- the planted two-batch design (A4). Per domain: none's lr profile NL, the prior's PL (its own read-out), the gain G
    # (8 seeds, added to the prior at every lr); areas = NB + profile (+ G) + 0.01 s; a prior arm's other read-outs NB - 1.
    NB = {d: 62.0 + 3 * i for i, d in enumerate(BATCHES["A"] + BATCHES["B2"])}
    G8 = [3.0, 2.0, 4.0, 2.5, 3.0, 3.5, 2.0, 4.0]
    NL0 = {"5e-4": -9.0, "1e-3": -7.0, "2e-3": -4.0, "4e-3": 0.0, "8e-3": -5.0, "1.6e-2": -9.0}   # none argmax 4e-3 (inside)
    PL0 = {"5e-4": -8.0, "1e-3": 0.0, "2e-3": -1.0, "4e-3": -6.0}                                # prior argmax 1e-3 (5e-4 loses)
    SPEC = {
        "months": (NL0, PL0, G8),                                                   # ADVANCE -> PASS (k-step)
        "stars": ({"5e-4": -3.0, "1e-3": 3.0, "2e-3": 0.0, "4e-3": -3.0, "8e-3": -5.0, "1.6e-2": -9.0},
                  {"5e-4": -9.0, "1e-3": -3.0, "2e-3": 0.0, "4e-3": 3.0}, G8),      # prior 4e-3 wins (EDGE); none 1e-3 -> 5e-4 loses
        "landmarks": (NL0, PL0, [3.0, -0.5, 2.0, 1.0] + G8[4:]),                     # 3/4 -> NOT ADVANCE
        "languages": ({**NL0, "2e-3": -2.5}, {**PL0, "2e-3": -6.0}, [2.0] * 8),     # same-lr gain -1.5 -> NOT ADVANCE (A4 iv)
        "colours": (NL0, PL0, [2.0] * 8),                                           # every prior cell fails the gate -> GATE
        "cities": ({"5e-4": -9.0, "1e-3": -7.0, "2e-3": -6.0, "4e-3": -1.0, "8e-3": 0.0, "1.6e-2": -0.5}, {**PL0, "2e-3": -3.0},
                   [2.0, 1.5, 2.5, 2.0, 2.0, 2.5, 1.5, 3.0]),   # shortcut not testable -> PASS + tag; none: bpb picks 4e-3, then 8e-3 (2b C)
        "zodiac": ({"5e-4": 6.0, "1e-3": 3.0, "2e-3": 0.0, "4e-3": -3.0, "8e-3": -5.0, "1.6e-2": -9.0},
                   {"5e-4": -5.0, "1e-3": 0.0, "2e-3": -3.0, "4e-3": -6.0}, [8.0] * 8),   # none 2e-3 -> 1e-3 -> 5e-4 still best: NOT FINAL
        "compass": ({"5e-4": -9.0, "1e-3": -7.0, "2e-3": -5.0, "4e-3": -3.0, "8e-3": 0.0, "1.6e-2": 3.0},
                    {"5e-4": -5.0, "1e-3": 0.0, "2e-3": -3.0, "4e-3": -6.0}, [8.0] * 8),  # none 8e-3 -> 1.6e-2 best: NOT FINAL
        "planets": ({**NL0, "8e-3": 1.0, "4e-3": -2.5, "1.6e-2": -3.0}, PL0, [4.0, 5.0, 4.0, 6.0, 3.0, 4.0, -0.5, 5.0]),   # A3 yes, seed 6 < 0 -> NOT SHOWN
        "elements": (NL0, {**PL0, "1e-3": -3.0, "2e-3": 0.0}, [3.0, 4.0, 3.0, 5.0, 4.0, 3.0, 5.0, 4.0]),   # PASS
        "prefixes": (NL0, PL0, [3.0, 4.0, 3.0, 5.0, 4.0, 3.0, 5.0, 4.0]),           # pooled 8-seed shortcut fails -> NOT SEPARABLE
        "ranks": ({"5e-4": -9.0, "1e-3": -1.0, "2e-3": 0.0, "4e-3": -1.5, "8e-3": -5.0, "1.6e-2": -9.0}, PL0,
                  [3.0, 4.0, 3.0, 5.0, 4.0, 3.0, 5.0, 4.0]),                         # none: bpb picks 4e-3, after 1e-3 the pick moves to 2e-3 (2b C)
    }
    BPB = {("B2", "none", "4e-3"): 1.19, ("B2", "none", "1e-3"): 1.21, ("A", "none", "4e-3"): 1.19, ("A", "none", "1.6e-2"): 1.21}

    def area_of(b, arm, lr, s):
        if arm == "none":
            return {x: NB[x] + SPEC[x][0][lr] + 0.01 * s for x in BATCHES[b]}
        d = arm[5:]
        return {x: (NB[x] + SPEC[x][1][lr] + SPEC[x][2][s] + 0.01 * s if x == d else NB[x] - 1.0) for x in BATCHES[b]}

    def bpb_of(b, arm, lr):   # a run's bpb is shared by every read-out of it: it only matters where two lrs tie (ranks, cities)
        return BPB.get((b, arm, lr), 1.2)
    share_of = lambda b, arm, lr, s: 0.3 if arm == "only-colours" else 0.9
    hard3_of = lambda b, arm, lr, s: ({"prefixes": 5.0} if arm == "only-prefixes" and s >= 4 else {"prefixes": 70.0} if arm == "only-prefixes" else {})
    nh_of = lambda b: ({x: (3 if x == "cities" else 20) for x in BATCHES[b]})

    def mk(root, b, arm, lr, s, **kw):
        fake_run(root, b, arm, lr, s, area_of(b, arm, lr, s), bpb=bpb_of(b, arm, lr), lm_share=kw.pop("lm_share", share_of(b, arm, lr, s)),
                 hard3=hard3_of(b, arm, lr, s), n_hard=nh_of(b), **kw)

    def run_cells(root, b, cs):
        for a, lr, s in cs:
            mk(root, b, a, lr, s)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "out"
        for b in ("A", "B2"):
            run_cells(root, b, MJ.cells(b, "1"))
        R_ = load(root)
        T("loader: 2 x 60 stage-1 runs, all valid; the k-step first-token area read (A4 secondary)",
          len(R_) == 120 and all(r["valid"] for r in R_.values()) and R_[("A", "none", "4e-3", 0)]["area1"]["months"] == NB["months"] - 1.0)
        subs = plant_subs(R_)
        sA, sB = screen(R_, "A", subs), screen(R_, "B2", subs)
        DA, DB = sA["domains"], sB["domains"]
        T("screen A: months / stars / cities ADVANCE; landmarks 3/4 -> NOT ADVANCE; languages: picks gain +2 in 4/4 but the same-lr "
          "gain at 2e-3 -1.5 -> NOT ADVANCE (A4 iv); colours: every prior cell fails the gate -> GATE; cities: 12 pooled hard "
          "questions -> shortcut NOT TESTABLE, does not block (A4)",
          (DA["months"]["decision"], DA["stars"]["decision"], DA["landmarks"]["decision"], DA["languages"]["decision"],
           DA["colours"]["decision"], DA["cities"]["decision"]) == ("ADVANCE", "ADVANCE", "NOT ADVANCE", "NOT ADVANCE", "GATE", "ADVANCE")
          and "3/4" in DA["landmarks"]["why"][0] and "same-lr" in DA["languages"]["why"][0] and len(DA["languages"]["why"]) == 1
          and np.allclose(DA["languages"]["same_lr"], [-1.5] * 4) and DA["cities"]["shortcut"]["ok"] is None
          and DA["cities"]["shortcut"]["n_pooled"] == 12 and sA["GATE"] == ["colours"])
        T("screen A: picks and extensions: stars prior 2e-3 -> E 4e-3, none 2e-3 -> E 1e-3; months prior 1e-3 -> E 5e-4, none 4e-3 "
          "(inside: none); cities none argmax 8e-3 -> E 1.6e-2 (A4); E at seeds 0-7, C = the picks",
          (DA["stars"]["pick_prior"], DA["stars"]["pick_none"], DA["stars"]["ext_prior"], DA["stars"]["ext_none"],
           DA["months"]["ext_prior"], DA["months"]["ext_none"]) == ("2e-3", "2e-3", "4e-3", "1e-3", "5e-4", None)
          and sA["E"] == [["none", "1e-3"], ["none", "1.6e-2"], ["only-cities", "5e-4"], ["only-months", "5e-4"], ["only-stars", "4e-3"]]
          and sA["C"] == [["none", "2e-3"], ["none", "4e-3"], ["only-cities", "1e-3"], ["only-months", "1e-3"], ["only-stars", "2e-3"]])
        T("screen B2: all six ADVANCE; none E: 1e-3 (zodiac, ranks: argmax 2e-3) and 1.6e-2 (compass, planets: argmax 8e-3), once each; "
          "ranks' none pick 4e-3 by the bpb tie-break (2e-3 within 2)",
          sB["ADVANCE"] == list(BATCHES["B2"]) and ["none", "1e-3"] in sB["E"] and ["none", "1.6e-2"] in sB["E"]
          and sum(e[0] == "none" for e in sB["E"]) == 2 and DB["ranks"]["pick_none"] == "4e-3" and DB["ranks"]["argmax_none"] == "2e-3")
        # generator plumbing: stage 2 = E at seeds 0-7 + C at seeds 4-7
        pf = {}
        for nm, obj in (("A1", sA), ("B21", sB)):
            pf[nm] = Path(tmp) / f"picks_{nm}.json"
            pf[nm].write_text(json.dumps(obj))
        c2 = MJ.cells("A", "2", pf["A1"])
        T(f"generator: stage 2 of A = 5 E x seeds 0-7 + 5 C x seeds 4-7 = {len(c2)} cells",
          len(c2) == 60 and {(a, lr) for a, lr, s in c2 if s < 4} == {tuple(e) for e in sA["E"]}
          and {(a, lr) for a, lr, s in c2 if s >= 4} == {tuple(e) for e in sA["E"]} | {tuple(c) for c in sA["C"]})
        bad_pk = dict(sA, E=[["none", "5e-4"]])
        (Path(tmp) / "bad.json").write_text(json.dumps(bad_pk))
        (Path(tmp) / "a3.json").write_text(json.dumps(dict(sA, amendment="A3")))
        T("generator: refuses a second-step extension (none@5e-4) in stage 2 and an A3 picks file",
          _raises_any(lambda: MJ.cells("A", "2", Path(tmp) / "bad.json")) and _raises_any(lambda: MJ.cells("A", "2", Path(tmp) / "a3.json")))
        for b, nm in (("A", "A1"), ("B2", "B21")):
            run_cells(root, b, MJ.cells(b, "2", pf[nm]))
        R_ = load(root)
        subs = plant_subs(R_)
        a2A, a2B = chain_step(R_, "A", [sA]), chain_step(R_, "B2", [sB])
        fA, fB = a2A["final"], a2B["final"]
        T("after 2, A: stars' prior extension 4e-3 wins -> pick 4e-3 (MOVED, EDGE), its seeds 4-7 ran with the extension (no C); "
          "stars' none 1e-3 wins by 3 -> next 5e-4 (stage 2b E); months / cities keep 1e-3 (5e-4 loses)",
          (fA["stars"]["prior"]["pick"], fA["stars"]["prior"]["moved"], fA["stars"]["prior"]["edge"], fA["stars"]["none"]["pick"],
           fA["stars"]["none"]["next"], fA["months"]["prior"]["pick"], fA["cities"]["prior"]["pick"]) == ("4e-3", True, True, "1e-3", "5e-4", "1e-3", "1e-3")
          and a2A["E"] == [["none", "5e-4"]] and a2A["stage_written"] == "2b")
        T("after 2, A: cities' none (stage 1: 8e-3 argmax, 4e-3 picked by bpb) -> 1.6e-2 enters the top two -> the pick moves to "
          "8e-3, which has no seeds 4-7 -> C none@8e-3 (stage 2b); not NOT FINAL (pick and argmax 8e-3)",
          (fA["cities"]["none"]["pick"], fA["cities"]["none"]["moved"], fA["cities"]["none"]["not_final"]) == ("8e-3", True, False)
          and a2A["C"] == [["none", "8e-3"]])
        T("after 2, B2: zodiac none 1e-3 best -> 5e-4 next; compass none 1.6e-2 best -> NOT FINAL (no upper step left); planets none "
          "1.6e-2 loses -> 8e-3 kept; ranks none: 1e-3 within 2 of 2e-3 -> the bpb tie-break moves the pick 4e-3 -> 2e-3, whose "
          "seeds 4-7 already ran as zodiac's confirm cell (shared none cells) -> no C",
          fB["zodiac"]["none"]["next"] == "5e-4" and fB["compass"]["none"]["not_final"] and fB["compass"]["none"]["pick"] == "1.6e-2"
          and fB["planets"]["none"]["pick"] == "8e-3" and not fB["planets"]["none"]["not_final"]
          and (fB["ranks"]["none"]["pick"], fB["ranks"]["none"]["moved"], fB["ranks"]["none"]["next"]) == ("2e-3", True, None)
          and a2B["E"] == [["none", "5e-4"]] and a2B["C"] == [])
        for nm, obj in (("A2", a2A), ("B22", a2B)):
            pf[nm] = Path(tmp) / f"picks_{nm}.json"
            pf[nm].write_text(json.dumps(obj))
        c2b, c2bA = MJ.cells("B2", "2b", pf["B22"]), MJ.cells("A", "2b", pf["A2"])
        T(f"generator: stage 2b of B2 = none@5e-4 x seeds 0-7 ({len(c2b)} cells); of A = none@5e-4 x seeds 0-7 + none@8e-3 x seeds 4-7 "
          f"({len(c2bA)})", len(c2b) == 8 and len(c2bA) == 12)
        o_part = readout(load(root), plant_subs(load(root)), [pf[k] for k in ("A1", "B21", "A2", "B22")], batches=("A", "B2"), family=("A", "B2"))
        T("read-out: stage 2b cells missing -> PROVISIONAL, no PASS", not o_part["final"]
          and not any(r["verdict"] == "PASS" for r in o_part["domains"].values()) and any("stops after" in v for v in o_part["provisional"]))
        for b, nm in (("A", "A2"), ("B2", "B22")):
            run_cells(root, b, MJ.cells(b, "2b", pf[nm]))
        R_ = load(root)
        subs = plant_subs(R_)
        a3A, a3B = chain_step(R_, "A", [sA, a2A]), chain_step(R_, "B2", [sB, a2B])
        T("after 3: stars' none 5e-4 loses -> 1e-3 final (closed); zodiac's none 5e-4 still best -> NOT FINAL; nothing more to run "
          "(complete, no stage 2c)", a3A["final"]["stars"]["none"]["pick"] == "1e-3" and not a3A["final"]["stars"]["none"]["not_final"]
          and a3B["final"]["zodiac"]["none"]["not_final"] and a3A["complete"] and a3B["complete"] and a3A["C"] == a3B["C"] == [])
        for nm, obj in (("A3", a3A), ("B23", a3B)):
            pf[nm] = Path(tmp) / f"picks_{nm}.json"
            pf[nm].write_text(json.dumps(obj))
        files = [pf[k] for k in ("A1", "B21", "A2", "B22", "A3", "B23")]
        o_fam = readout(R_, subs, files, batches=("A", "B2"))
        T("read-out with the design's family (all four batches) while only A and B2 are on disk -> PROVISIONAL (batches B1 / B3 "
          "missing), no PASS; the Holm family is not shrunk silently", not o_fam["final"]
          and any("batch B1" in v for v in o_fam["provisional"]) and not any(r["verdict"] == "PASS" for r in o_fam["domains"].values())
          and all(r["verdict"].startswith("PROVISIONAL") or r["verdict"] in (NOT_FINAL, GATE_PICK_FAIL) for r in o_fam["domains"].values()))
        o = readout(R_, subs, files, batches=("A", "B2"), family=("A", "B2"),
                    sites={"compass": {"frac": 0.0, "n_shared": 0, "n_items": 8}, "ranks": {"frac": 0.25, "n_shared": 2, "n_items": 8}})
        txt = report(o)
        print(txt)
        V = {d: o["domains"][d]["verdict"] for d in o["domains"]}
        T("read-out (test family A + B2): final; the six picks files agree with the re-derivation", o["final"]
          and len(o["picks_consistency"]) == 6 and all(x["agree"] for x in o["picks_consistency"]))
        T("Holm family = the 9 advancing domains of both batches (NOT FINAL ones included: conservative)",
          sorted(o["domains"]) == sorted(["months", "stars", "cities"] + list(BATCHES["B2"])) and all(r["holm_family"] == 9 for r in o["domains"].values()))
        T("months / elements / ranks PASS (8 consistent seeds, seeds 4-7 all positive, Holm over 9); stars PASS at the moved prior "
          "pick 4e-3 vs none 1e-3 with the EDGE tag; ranks at none's moved pick 2e-3 (its 2b seeds)",
          V["months"] == V["elements"] == V["ranks"] == V["stars"] == "PASS" and o["domains"]["stars"]["pick_prior"] == "4e-3"
          and o["domains"]["stars"]["pick_none"] == "1e-3" and any(t_.startswith("EDGE") for t_ in o["domains"]["stars"]["tags"])
          and o["domains"]["ranks"]["pick_none"] == "2e-3" and np.allclose(o["domains"]["stars"]["per_seed"], G8))
        T("zodiac (none still best at 5e-4) and compass (at 1.6e-2) -> NOT FINAL, no verdict", V["zodiac"] == V["compass"] == NOT_FINAL)
        T("planets: the A3 conditions hold, seed 6's gain is negative -> NOT SHOWN (A4 selection-free condition)",
          V["planets"] == "NOT SHOWN" and o["domains"]["planets"]["three_conditions"] and not o["domains"]["planets"]["selection_free"])
        T("prefixes: the screen's pooled shortcut met (seeds 0-3: 70 > 40), over 8 seeds pooled 37.5 <= 40 -> NOT SEPARABLE",
          o["screen"]["B2"]["domains"]["prefixes"]["shortcut"]["ok"] is True and V["prefixes"] == "NOT SEPARABLE"
          and abs(o["domains"]["prefixes"]["shortcut"]["prior"] - 37.5) < 1e-9)
        T("cities: PASS with the tag SHORTCUT NOT TESTABLE (24 pooled questions < 30)", V["cities"] == "PASS"
          and any(t_.startswith("SHORTCUT NOT TESTABLE") for t_ in o["domains"]["cities"]["tags"]) and o["domains"]["cities"]["shortcut"]["n_pooled"] == 24)
        T("verdict lines: the same-lr gain (screen 2e-3 and every lr both arms ran), the seeds 4-7 test, the first-token k-way line "
          "(k-step), Tier L / k-step tags",
          all(f"{d:15s}" in txt for d in o["domains"]) and "same-lr gain (A4" in txt and "seeds 4-7 alone" in txt
          and "first-token k-way gain" in txt and np.allclose(o["domains"]["months"]["same_lr_screen"], [g_ + 3 for g_ in G8[:4]]))
        T("FLOOR (A4): the line includes the shortcut baseline (planted h3_held 60) and the elimination baseline",
          o["domains"]["stars"]["floor_line"] == 65.0)
        # A6: the same design with every run of only-months (A), only-planets and only-elements (B2) flagged A6 (a geo overshoot
        # -0.00115 at cka 1.0, the geo / grad_prior check failed): every run stays valid, and the screens, the decision chains,
        # the gates, the verdicts and the report are exactly those of the clean runs; only the A6 list differs
        import shutil
        root6 = Path(tmp) / "out_a6"
        shutil.copytree(root, root6)
        n6 = 0
        for k_, dirs_ in cell_dirs(root6).items():
            if k_[1] in ("only-months", "only-planets", "only-elements"):
                patch_a6(dirs_[-1][1], *k_)
                n6 += 1
        R6 = load(root6)
        o6 = readout(R6, plant_subs(R6), files, batches=("A", "B2"), family=("A", "B2"),
                     sites={"compass": {"frac": 0.0, "n_shared": 0, "n_items": 8}, "ranks": {"frac": 0.25, "n_shared": 2, "n_items": 8}})
        js_ = lambda x: json.dumps({k_: v_ for k_, v_ in x.items() if k_ != "geo_flags"}, sort_keys=True,
                                   default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
        t6 = report(o6).split("\n")
        T(f"A6: {n6} runs of only-months / only-planets / only-elements flagged A6 -> all valid; screens, chains, gates, verdicts "
          f"and the report identical to the clean runs (only the FLAGGED (A6) line added); geo_flags lists exactly them",
          n6 > 0 and all(r_["valid"] for r_ in R6.values()) and sorted(R6) == sorted(R_)
          and js_(o6) == js_(o) and o["geo_flags"] == [] and len(o6["geo_flags"]) == n6
          and {tuple(k_[1:2]) for k_, _ in o6["geo_flags"]} == {("only-months",), ("only-planets",), ("only-elements",)}
          and [l_ for l_ in t6 if not l_.startswith("FLAGGED (A6")] == report(o).split("\n")
          and sum(l_.startswith("FLAGGED (A6, valid data): %d runs" % n6) for l_ in t6) == 1)
        # a missing / differing picks file -> PROVISIONAL
        o_m = readout(R_, subs, [f for f in files if f != pf["B22"]], batches=("A", "B2"), family=("A", "B2"))
        tam = Path(tmp) / "picks_tamper.json"
        tam.write_text(json.dumps(dict(a3A, C=[["none", "8e-3"]])))
        o_t = readout(R_, subs, [f for f in files if f != pf["A3"]] + [tam], batches=("A", "B2"), family=("A", "B2"))
        T("a missing picks file (B2 after 2) or one that differs from the re-derivation (A after 3) -> PROVISIONAL, no PASS (A4; A3 "
          "printed DIFFER and went on)", not o_m["final"] and not o_t["final"]
          and not any(r["verdict"] == "PASS" for x_ in (o_m, o_t) for r in x_["domains"].values())
          and any("no picks file" in v for v in o_m["provisional"]) and any("DIFFER" not in v and "C" in v for v in o_t["provisional"]))
        # the gate at the pick per half (A4): elements' confirm seeds at a / (a + b) 0.45 -> seeds 4-7 fail while the 8-seed median
        # (0.9 x 4, 0.45 x 4) = 0.675 passes
        for s in CONF:
            mk(root, "B2", "only-elements", o["domains"]["elements"]["pick_prior"], s, lm_share=0.45)
        R2 = load(root)
        o3 = readout(R2, plant_subs(R2), files, batches=("A", "B2"), family=("A", "B2"))
        T(f"the gate at the pick per half (A4): elements seeds 4-7 median 0.45 < 0.5 while the 8-seed median "
          f"{o3['domains']['elements']['pick_gate']['median']:.3f} passes -> '{GATE_PICK_FAIL}'",
          o3["domains"]["elements"]["verdict"] == GATE_PICK_FAIL and o3["domains"]["elements"]["pick_gate"]["ok"]
          and not o3["domains"]["elements"]["pick_gate_conf"]["ok"] and o3["domains"]["elements"]["pick_gate_sel"]["ok"])
        T("A5 fixB with the none cell already asked (batch A / B2 above: none@1e-3 ran as another domain's extension): months, "
          "cities, compass, planets, prefixes (prior pick 1e-3 below none's range) are re-selected including none@1e-3 at once (no "
          "F, the FIXB tag), their picks unchanged (1e-3 is not none's best there)",
          all(o["domains"][d_]["fixb"] == "1e-3" and any(t_.startswith("FIXB") for t_ in o["domains"][d_]["tags"])
              for d_ in ("months", "cities", "compass", "planets", "prefixes"))
          and not any(p_.get("F") for p_ in (a2A, a2B, a3A, a3B)) and o["domains"]["stars"]["fixb"] is None
          and (o["domains"]["months"]["pick_none"], o["domains"]["prefixes"]["pick_none"]) == ("4e-3", "4e-3"))
    # ---- A5 fixB with new none cells (audit_a4 important 2, its CX1 made exact): batch B3. chinese_zodiac: NO prior effect, a
    # shared lr trend (+3 per halving, both arms), none@4e-3 lucky +4.5 on seeds 0-3 (its stage-1 argmax inside the grid: no
    # lower extension), the prior's same-lr 2e-3 cell +0.4 on seeds 0-3 ((iv) met): the prior extends to 5e-4 and wins there
    # (EDGE); A4 would PASS it against none@4e-3. kinship: a true +4 at every lr, the prior's final pick 1e-3, none's best
    # 4e-3. Both picks lie below none's range [2e-3, 8e-3] -> F none@5e-4 / none@1e-3 at seeds 0-7 (stage 2b); no verdict until
    # they have run; then chinese_zodiac's none moves to 5e-4 -> NOT FINAL (5e-4 is outermost), kinship stays PASS (FIXB tag).
    # geo_periods: none's stage-1 argmax 8e-3, the prior below none -> NOT ADVANCE with the A5.3 note.
    TR = lambda lr: 3.0 * -math.log2(float(lr) / 2e-3)
    jit = lambda arm, lr, s: 0.05 * ((3 * s + (0 if arm == "none" else 1) + LADDER.index(lr)) % 4)
    NK = {"5e-4": -6.0, "1e-3": -3.0, "2e-3": -1.0, "4e-3": 0.0, "8e-3": -4.0, "1.6e-2": -8.0}
    PK = {"5e-4": -5.0, "1e-3": 0.0, "2e-3": -3.0, "4e-3": -6.0}
    NG = {"5e-4": -9.0, "1e-3": -7.0, "2e-3": -6.0, "4e-3": -3.0, "8e-3": 0.0, "1.6e-2": -2.0}
    base3 = {d_: 60.0 + 2 * i_ for i_, d_ in enumerate(BATCHES["B3"])}

    def area3(arm, lr, s):
        out_ = {}
        for d_ in BATCHES["B3"]:
            v = base3[d_] + jit(arm, lr, s)
            if d_ == "chinese_zodiac":
                v += TR(lr) + (4.5 if arm == "none" and lr == "4e-3" and s in SEL else 0.0) + \
                    (0.4 if arm == f"only-{d_}" and lr == "2e-3" and s in SEL else 0.0)
            elif d_ == "kinship":
                v += (NK[lr] if arm == "none" else (PK[lr] + 4.0 if arm == f"only-{d_}" else NK.get(lr, 0.0) - 1.0))
            elif d_ == "geo_periods":
                v += NG[lr] if arm == "none" else NG[lr] - 1.0
            else:
                v += -1.0 if arm == f"only-{d_}" else 0.0
            out_[d_] = v
        return out_
    bpb3 = lambda arm, lr: 1.19 if (arm, lr) == ("none", "4e-3") else 1.2
    with tempfile.TemporaryDirectory() as tmp:
        root3 = Path(tmp) / "out"
        run3 = lambda cs: [fake_run(root3, "B3", a_, lr_, s_, area3(a_, lr_, s_), bpb=bpb3(a_, lr_)) for a_, lr_, s_ in cs]
        run3(MJ.cells("B3", "1"))
        R3 = load(root3)
        s3 = screen(R3, "B3", plant_subs(R3))
        D3 = s3["domains"]
        T("A5 B3 screen: chinese_zodiac ADVANCES with no prior effect (gain +1.5 in 4/4, same-lr +0.4 in 4/4; none argmax 4e-3 inside "
          "the grid: no none extension) and kinship (gain +4); geo_periods NOT ADVANCE with the A5.3 note (none's argmax 8e-3 above "
          "the prior's grid); life_stages NOT ADVANCE without it",
          s3["ADVANCE"] == ["chinese_zodiac", "kinship"] and D3["chinese_zodiac"]["ext_none"] is None
          and D3["chinese_zodiac"]["argmax_none"] == "4e-3" and D3["chinese_zodiac"]["pick_prior"] == "1e-3"
          and D3["geo_periods"]["note_none_higher_lr"] and not D3["life_stages"]["note_none_higher_lr"]
          and "NOTE (A5.3)" in screen_report(s3) and "is NOT evidence of no effect" in screen_report(s3))
        pf3 = {"1": Path(tmp) / "picks_B3_after1.json"}
        pf3["1"].write_text(json.dumps(s3))
        run3(MJ.cells("B3", "2", pf3["1"]))
        R3 = load(root3)
        b2_ = chain_step(R3, "B3", [s3])
        fz, fk = b2_["final"]["chinese_zodiac"], b2_["final"]["kinship"]
        T("A5 B3 after 2: chinese_zodiac's prior moves to 5e-4 (EDGE), kinship's stays 1e-3; both below none's range ['2e-3', '8e-3'] "
          "with none's own chain done -> F = none@1e-3 and none@5e-4 (seeds 0-7, stage 2b), fixB pending; no E, no C",
          (fz["prior"]["pick"], fz["prior"]["edge"], fk["prior"]["pick"]) == ("5e-4", True, "1e-3")
          and fz["none"]["fixb_pending"] and fk["none"]["fixb_pending"] and fz["none"]["range"] == ["2e-3", "8e-3"]
          and b2_["F"] == [["none", "5e-4"], ["none", "1e-3"]] and b2_["E"] == b2_["C"] == [] and b2_["stage_written"] == "2b"
          and not b2_["complete"])
        pf3["2"] = Path(tmp) / "picks_B3_after2.json"
        pf3["2"].write_text(json.dumps(b2_))
        c2b3 = MJ.cells("B3", "2b", pf3["2"])
        bad_f = {"F in stage 2": dict(s3, F=[["none", "1e-3"]]), "F at a prior arm": dict(b2_, F=[["only-kinship", "1e-3"]]),
                 "F at 2e-3 (inside none's grid)": dict(b2_, F=[["none", "2e-3"]]), "an A4 picks file": dict(b2_, amendment="A4")}
        refused = {}
        for nm_, obj in bad_f.items():
            p_ = Path(tmp) / "bad_f.json"
            p_.write_text(json.dumps(obj))
            refused[nm_] = _raises_any(lambda: MJ.cells("B3", "2" if nm_ == "F in stage 2" else "2b", p_))
        T(f"generator: stage 2b of B3 = the F cells none@1e-3 / 5e-4 x seeds 0-7 ({len(c2b3)}); refuses {sorted(refused)}",
          sorted(c2b3) == sorted(("none", lr_, s_) for lr_ in ("1e-3", "5e-4") for s_ in SEL + CONF) and all(refused.values()))
        o_p = readout(load(root3), plant_subs(load(root3)), [pf3["1"], pf3["2"]], batches=("B3",), family=("B3",))
        T("A5 read-out before the F cells ran: NO VERDICT for chinese_zodiac and kinship (fixB pending), PROVISIONAL with the reasons",
          not o_p["final"] and all(o_p["domains"][d_]["verdict"].startswith("NO VERDICT (A5 fixB") for d_ in ("chinese_zodiac", "kinship"))
          and sum("A5 fixB: none has not run" in v for v in o_p["provisional"]) == 2)
        # what A4 would have said for chinese_zodiac: the false PASS of audit_a4's CX1
        ga4 = np.array([area3("only-chinese_zodiac", "5e-4", s)["chinese_zodiac"] - area3("none", "4e-3", s)["chinese_zodiac"] for s in SEL + CONF])
        a4_pass = rule(ga4, t_test(ga4)[1], sf_test(ga4[4:]))[0] and not fz["none"]["a4"]["not_final"]
        run3(c2b3)
        R3 = load(root3)
        b3_ = chain_step(R3, "B3", [s3, b2_])
        pf3["3"] = Path(tmp) / "picks_B3_after3.json"
        pf3["3"].write_text(json.dumps(b3_))
        o3_ = readout(R3, plant_subs(R3), [pf3["1"], pf3["2"], pf3["3"]], batches=("B3",), family=("B3",))
        V3 = {d_: o3_["domains"][d_]["verdict"] for d_ in o3_["domains"]}
        T(f"A5 after 2b: chinese_zodiac's none re-selected over its range + 5e-4 -> pick 5e-4 = the outermost lr -> NOT FINAL (A4 rule "
          f"alone on the same runs: PASS {a4_pass}); kinship's none + 1e-3 -> stays 4e-3 -> PASS with the FIXB tag; chain complete, "
          f"read-out final on the test family",
          a4_pass and b3_["complete"] and b3_["final"]["chinese_zodiac"]["none"]["fixb"] == "5e-4"
          and b3_["final"]["chinese_zodiac"]["none"]["pick"] == "5e-4" and V3["chinese_zodiac"] == NOT_FINAL
          and b3_["final"]["kinship"]["none"]["pick"] == "4e-3" and V3["kinship"] == "PASS" and o3_["final"]
          and any(t_.startswith("FIXB") for t_ in o3_["domains"]["kinship"]["tags"]))
        tam3 = Path(tmp) / "picks_B3_after2_noF.json"
        tam3.write_text(json.dumps(dict(b2_, F=[])))
        o_t3 = readout(R3, plant_subs(R3), [pf3["1"], tam3, pf3["3"]], batches=("B3",), family=("B3",))
        o_sp = readout(R3, plant_subs(R3), [pf3["1"], pf3["2"], pf3["3"]], batches=("B3",), family=("B3",), allow_partial=True,
                       sprobs=["B3 none@2e-3 s0 kinship: table sha x != the snapshot's y"])
        T("A5: a picks file whose F differs from the re-derivation -> PROVISIONAL ('F'); --allow_partial with a comparison "
          "sub-read-out problem -> PROVISIONAL (no PASS), the problem printed (audit_a4 minor 3)",
          not o_t3["final"] and any("after 2: F" in v for v in o_t3["provisional"]) and not o_sp["final"]
          and not any(r_["verdict"] == "PASS" for r_ in o_sp["domains"].values())
          and any("sub-read-out problems" in v for v in o_sp["provisional"]) and "COMPARISON SUB-READ-OUT PROBLEMS" in report(o_sp))
    # A5.1: the loader keeps a run whose only failed checks are the GPU-numerics double-path tolerances (flagged)
    with tempfile.TemporaryDirectory() as tmp:
        r5 = Path(tmp) / "dp"
        b5 = {d_: 50.0 for d_ in BATCHES["B2"]}
        T2W, TKS = DP_FLAG_CHECKS[0] + ", A5.1): ... <= 0.05 nats at eval 0 ...", DP_FLAG_CHECKS[1] + ", A5.1): ..."
        fake_run(r5, "B2", "none", "2e-3", 0, b5, chk_fail=(T2W,))
        fake_run(r5, "B2", "none", "4e-3", 0, b5, chk_fail=(T2W, TKS, CKA_CHECK + " by >= 0.1"))
        fake_run(r5, "B2", "none", "8e-3", 0, b5, chk_fail=("double-path fields logged (finite) at every eval before a divergence",))
        fake_run(r5, "B2", "only-zodiac", "1e-3", 0, b5, chk_fail=(TKS, "run args = intent"))
        L5 = load(r5)
        fl5 = dp_flags(L5)
        T("A5.1 loader: only the 2-way GPU-numerics tolerance failed -> valid, FLAGGED; 2-way + k-span tolerances + CKA -> valid, "
          "flagged; a missing double-path field -> invalid; a tolerance + another check -> invalid; dp_flags lists the two",
          (L5[("B2", "none", "2e-3", 0)]["valid"], L5[("B2", "none", "4e-3", 0)]["valid"], L5[("B2", "none", "8e-3", 0)]["valid"],
           L5[("B2", "only-zodiac", "1e-3", 0)]["valid"]) == (True, True, False, False)
          and "FLAG A5.1" in L5[("B2", "none", "2e-3", 0)]["why"] and [k_[2] for k_, _ in fl5] == ["2e-3", "4e-3"]
          and len(fl5[1][1]) == 2)
    # A6: the loader keeps a run whose only failed checks are A5.1's set and the geo / grad_prior check at a geo overshoot of a
    # perfect fit, verified from the run JSON (flagged A6); every other case stays invalid
    with tempfile.TemporaryDirectory() as tmp:
        r6 = Path(tmp) / "a6"
        b6 = {d_: 50.0 for d_ in BATCHES["B2"]}
        T2W = DP_FLAG_CHECKS[0] + ", A5.1): ... <= 0.05 nats at eval 0 ..."
        OTHER = "data_sha after step 0 = the CPU reference 4174d3fb0e6447dd (the same data in every arm of the batch and seed)"
        C6 = {   # (arm, lr, seed): (patch / fake_run keywords, valid?, a phrase of the reason when invalid)
            ("only-planets", "1e-3", 0): ({}, True, None),                                     # the trigger: -0.00115 at cka 1.0
            ("only-planets", "1e-3", 1): ({"geo": -0.0012, "cka": 0.95}, False, "not a perfect fit"),
            ("only-planets", "1e-3", 2): ({"geo": -0.006, "cka": 1.0}, False, "< the floor -0.005"),
            ("only-planets", "1e-3", 3): ({"geo": 0.3, "cka": 0.7, "gp_zero_at": (12,)}, False, "(1) grad_prior not finite > 0 at 1 of 26"),
            ("only-planets", "2e-3", 0): ({"chk_fail": (OTHER,)}, False, None),                # A6-like + another failed check
            ("only-planets", "2e-3", 1): ({"chk_fail": (T2W, CKA_CHECK + " by >= 0.1")}, True, None),   # A6 + A5.1 (+ CKA rise)
            ("only-planets", "2e-3", 2): ({"geo": -0.005, "cka": 0.995}, True, None),          # both lower boundaries are inside (A6.1)
            ("only-planets", "2e-3", 3): ({"geo": 0.3, "cka": 0.7}, False, "not the A6 case"),  # a failure nothing explains
            ("only-elements", "1e-3", 0): ({"inactive": "zodiac"}, False, "(3) an inactive domain has a geo"),
            ("only-elements", "1e-3", 1): ({"keep": A6_KEEP[:2] + A6_KEEP[3:]}, False, "(3) the separate check 'weighted prior"),
            ("only-elements", "1e-3", 2): ({"gp_zero_at": (20,)}, False, "(1) grad_prior not finite > 0"),   # overshoot + grad_prior 0
            ("only-elements", "1e-3", 3): ({"geo_nan_at": (20,)}, False, "not finite"),
            ("only-elements", "2e-3", 0): ({"geo": -0.0012, "cka": 0.9949}, False, "0.9949 < 0.995 (not a perfect fit)"),   # A6.1: just below the bar
            ("only-elements", "2e-3", 1): ({"geo": -0.0050001, "cka": 1.0}, False, "< the floor"),          # just below the floor
            ("only-elements", "2e-3", 2): ({"at": (10, 20), "geo": -0.002, "cka": 0.9995}, True, None),     # two overshoot evals
            # A6.1 (bar 0.995, cap 1.01): the coordinator's cases 0.9975 / 0.994 / 1.02, the upper boundary 1.01; audit_a6 minor 8:
            # two active priors, an arm-name mismatch, a separate check present twice
            ("only-elements", "2e-3", 3): ({"geo": -0.0012, "cka": 0.9975}, True, None),
            ("only-zodiac", "1e-3", 0): ({"geo": -0.0012, "cka": 0.994}, False, "0.994 < 0.995 (not a perfect fit)"),
            ("only-zodiac", "1e-3", 1): ({"geo": -0.0012, "cka": 1.02}, False, "1.02 > 1.01 (above any float overshoot"),
            ("only-zodiac", "1e-3", 2): ({"geo": -0.0012, "cka": 1.01}, True, None),
            ("only-zodiac", "1e-3", 3): ({"extra_active": "planets"}, False, "not one active prior"),
            ("only-zodiac", "2e-3", 0): ({"active": "planets"}, False, "not one active prior"),
            ("only-zodiac", "2e-3", 1): ({"keep": A6_KEEP + A6_KEEP[:1]}, False, "is not present once and passed ([True, True])"),
        }
        for (a_, lr_, s_), (kw_, _, _) in C6.items():
            fake_a6(r6, "B2", a_, lr_, s_, b6, **kw_)
        fake_run(r6, "B2", "none", "2e-3", 0, b6)
        L6 = load(r6)
        got = {k_[1:]: (L6[k_]["valid"], L6[k_]["why"]) for k_ in L6 if k_[1] != "none"}
        bad6 = [k_ for k_, (kw_, v_, w_) in C6.items() if got[k_][0] != v_ or (w_ is not None and w_ not in got[k_][1])
                or (not v_ and w_ is not None and "A6 does not apply" not in got[k_][1])]
        g6, d6 = geo_flags(L6), dp_flags(L6)
        T("A6 loader (synthetic run JSONs): -0.00115 at cka 1.0 -> valid, FLAGGED A6; -0.0012 at cka 0.95, -0.006 at cka 1.0, "
          "grad_prior 0 at one on-eval, an A6-like failure + another failed check -> invalid; A6 + A5.1 (2-way tolerance, CKA rise) "
          "-> valid, both flags listed; geo -0.005 at cka 0.995 (the lower boundaries) -> valid; a failed check nothing explains, a "
          "geo field of an inactive domain, a separate check missing, an overshoot + grad_prior 0, a NaN geo, cka 0.9949, geo "
          "-0.0050001 -> invalid with the reason; two overshoot evals -> valid, both listed. A6.1: cka 0.9975 and 1.01 (the cap) -> "
          "valid; cka 0.994 and 1.02 -> invalid; two active priors, an arm-name mismatch, a separate check present twice -> invalid "
          f"(audit_a6 minor 8) (wrong: {bad6})",
          not bad6 and L6[("B2", "none", "2e-3", 0)]["valid"]
          and [k_[1:] for k_, _ in g6] == sorted(k_ for k_, (_, v_, _) in C6.items() if v_)
          and [k_[1:] for k_, _ in d6] == [("only-planets", "2e-3", 1)] and len(d6[0][1]) == 1
          and "FLAG A6: geo overshoot at a perfect fit: geo_planets -0.001150 >= -0.005 with cka_planets 1.000000 in [0.995, 1.01] at step 10"
          in got[("only-planets", "1e-3", 0)][1] and "FLAG A5.1" in got[("only-planets", "2e-3", 1)][1]
          and "FLAG A6" in got[("only-planets", "2e-3", 1)][1] and "CKA-rise check failed" in got[("only-planets", "2e-3", 1)][1]
          and len(dict(g6)[("B2", "only-elements", "2e-3", 2)]) == 2
          and got[("only-planets", "2e-3", 0)][1].startswith("failed check: ") and "A6 does not apply" not in got[("only-planets", "2e-3", 0)][1]
          and L6[("B2", "only-planets", "1e-3", 0)]["geo_flag"]["n_on"] == 26 and L6[("B2", "none", "2e-3", 0)].get("geo_flag") is None)
        # the check name: the loader's prefix = CHECK_PY's f-string rendered with the generator's geo_tol; the guard on the real
        # rendered job files, and on tampered copies (a renamed check, another geo_tol, a renamed separate check, no file)
        src = next(l_ for l_ in MJ.CHECK_PY.split("\n") if GEO_TEMPLATE in l_)
        rendered = src.split('chk(f"', 1)[1].split('",', 1)[0].replace("{TOL}", f"{MJ.GEO_TOL}")
        n_jobs = a6_guard()
        jf = sorted(HERE.glob("job_ptdom-*.sbatch"))[0]
        jt = jf.read_text()
        tam6 = {"renamed": jt.replace(GEO_TEMPLATE, 'chk(f"geo >= {TOL} for the active prior and 0 < grad_prior < inf at every on-eval'),
                "geo_tol": jt.replace('"geo_tol":-0.001', '"geo_tol":-0.002'),
                "keep": jt.replace(f'chk("{A6_KEEP[1]}', 'chk("geo_<d> logged for the active prior'),
                "twice": jt.replace("PY\necho \"## done", f"# {GEO_TEMPLATE}\nPY\necho \"## done", 1)}
        rej = {}
        for nm_, t_ in tam6.items():
            dd_ = Path(tmp) / f"g_{nm_}"
            dd_.mkdir()
            (dd_ / jf.name).write_text(t_)
            rej[nm_] = t_ != jt and _raises_any(lambda: a6_guard(str(dd_)))
        (Path(tmp) / "g_empty").mkdir()
        rej["empty"] = _raises_any(lambda: a6_guard(str(Path(tmp) / "g_empty")))
        T(f"A6 check name: CHECK_PY's f-string rendered with geo_tol {MJ.GEO_TOL} = '{rendered}' = A6_CHECK_NAME; a6_guard passes on "
          f"the {n_jobs} rendered job files of runs/ptdom and rejects a renamed check, another geo_tol, a renamed separate check, "
          f"the template twice, no job file ({rej})", rendered == A6_CHECK_NAME and rendered.startswith(GEO_FLAG_CHECK)
          and n_jobs == len(list(HERE.glob("job_ptdom-*.sbatch"))) >= 240 and all(rej.values()) and len(rej) == 5)
    # the audit_a1 gate counter-examples, per cell (A2), through select_unit / screen exclusion
    with tempfile.TemporaryDirectory() as tmp:
        r3 = Path(tmp) / "gate"
        base = {d: 50.0 for d in BATCHES["B1"]}
        for s in SEL:
            fake_run(r3, "B1", "only-countries", "1e-3", s, dict(base, countries=60 + 0.01 * s), lm_share=0.3)   # CX1: the area argmax at 0.3
            fake_run(r3, "B1", "only-countries", "2e-3", s, dict(base, countries=55 + 0.01 * s), lm_share=0.9)
            fake_run(r3, "B1", "only-animals", "1e-3", s, base, lm_share=0.4)                  # CX2: b = 1.5 a at cos 0
            fake_run(r3, "B1", "only-animals", "2e-3", s, base, lm_share=0.4, cos=-0.9)        # CX3: opposing gradients
            fake_run(r3, "B1", "only-people", "1e-3", s, base, lm_share=0.9, missing=("grad_cos",) if s == 1 else ())   # CX7
            fake_run(r3, "B1", "only-people", "2e-3", s, base, lm_share=0.5)                   # b = a exactly: passes
            fake_run(r3, "B1", "only-units", "1e-3", s, base, lm_share=0.9, logged_off=0.2 if s == 2 else 0.0)
            fake_run(r3, "B1", "only-units", "2e-3", s, base, lm_share=0.9)
        Rg = load(r3)
        gt = aux_gate(Rg, "B1")
        gc = gt["cells"]
        T(f"aux gate per cell (audit_a1 CX1 / CX2 / CX3 / CX7): countries@1e-3 0.3 fails, @2e-3 0.9 passes; animals 0.4 at cos 0 and "
          f"at cos -0.9 fail (every cell -> the arm fails); a missing grad_cos fails people@1e-3, b = a passes; a logged share off its "
          f"re-derivation fails units@1e-3 (failed {gt['failed']})",
          sorted(gt["failed"]) == sorted(["only-countries@1e-3", "only-animals@1e-3", "only-animals@2e-3", "only-people@1e-3", "only-units@1e-3"])
          and gt["arms_all_failed"] == ["only-animals"] and abs(gc["only-animals@2e-3"]["median"] - 0.4) < 1e-12
          and any("missing gradient fields (grad_cos)" in w for w in gc["only-people@1e-3"]["why"])
          and any("logged grad_lm_share" in w for w in gc["only-units@1e-3"]["why"]) and gc["only-people@2e-3"]["ok"])
        u = select_unit(Rg, "B1", "only-countries", "countries", grid("only-countries"), gate_bad(gt).get("only-countries", set()))
        T("CX1 exclusion: countries@1e-3 (the area argmax) fails the gate -> excluded; the pick is 2e-3 (the only admissible lr); "
          "its argmax 2e-3 is the upper edge -> extension 4e-3", (u["pick"], u["grid"], u["excluded"], extension("only-countries", u["argmax"]))
          == ("2e-3", ["2e-3"], ["1e-3"], "4e-3"))
        # loader / integrity
        r2 = Path(tmp) / "ex"
        base3 = {d: 50.0 for d in BATCHES["B3"]}
        fake_run(r2, "B3", "none", "8e-3", 0, base3, chk_fail=("run args = intent",))
        fake_run(r2, "B3", "only-greek", "1e-3", 0, base3, chk_fail=(CKA_CHECK + " by >= 0.1",))
        fake_run(r2, "B3", "none", "4e-3", 0, base3, evals=40)
        fake_run(r2, "B3", "only-kinship", "2e-3", 0, base3, evals=40)
        fake_run(r2, "B3", "only-kinship", "2e-3", 0, base3, rr=1)
        L2 = load(r2)
        T("loader: a failed check excludes; a CKA-only fail stays valid; an unfinished run is excluded; a -r1 dir supersedes an "
          "unfinished first dir", (L2[("B3", "none", "8e-3", 0)]["valid"], L2[("B3", "only-greek", "1e-3", 0)]["valid"],
                                   L2[("B3", "none", "4e-3", 0)]["valid"], L2[("B3", "only-kinship", "2e-3", 0)]["valid"],
                                   L2[("B3", "only-kinship", "2e-3", 0)]["rerun"]) == (False, True, False, True, 1))
        fake_run(r2, "B3", "none", "2e-3", 1, base3); fake_run(r2, "B3", "only-greek", "2e-3", 1, base3)
        fake_run(r2, "B1", "none", "2e-3", 1, base)
        p_ = r2 / "ptdom-B3-none-lr2e-3-s1" / "ptdom-B3-none-lr2e-3" / "run_whole_none_seed1_ptdom-B3-none-lr2e-3.json"
        j = json.loads(p_.read_text()); j["curve"][5]["data_sha"] = "other"; p_.write_text(json.dumps(j))
        T("integrity: different data within a batch and seed is reported; different batches are not compared",
          any("batch B3 seed 1: data_sha" in x for x in integrity(load(r2))) and not any("batch B1" in x for x in integrity(load(r2))))
    n_ok = sum(v for _, v in ok)
    print(f"SELFTEST {'PASS' if n_ok == len(ok) else 'FAIL'} ({n_ok}/{len(ok)})")
    return n_ok == len(ok)


def _raises_any(f):
    try:
        f()
    except (AssertionError, SystemExit):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / "out"))
    ap.add_argument("--picks", nargs="*", default=None, help="picks files to compare with (default: every picks_<X>_after<k>.json)")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--screen", choices=tuple(BATCHES), help="print the stage-1 screen of one batch only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    R_ = load(a.root)
    doms = [d for b in BATCHES for d in BATCHES[b] if any(k[0] == b for k in R_)]
    meta = domain_meta(doms)
    subs, sprobs = subsets_all(R_, meta)
    if sprobs and not a.allow_partial:
        raise SystemExit("comparison sub-read-outs: " + "; ".join(sprobs[:20]))
    elim = {}
    for d in doms:
        if d in KSTEP and d != "months":
            ss_ = sorted({r["split_seed"].get(d) for r in R_.values() if "split_seed" in r and r["seed"] in SEL} - {None})
            elim[d] = float(np.mean([elim_baseline(meta[d], s_) for s_ in ss_])) if ss_ else None
    if a.screen:
        p1 = screen(R_, a.screen, subs)
        print(gate_report(aux_gate(R_, a.screen, SEL)))
        fl = dp_flags(R_, a.screen)
        print(f"FLAGGED (A5.1, valid data): {len(fl)} runs of batch {a.screen} exceed a GPU-numerics double-path tolerance"
              + (": " + "; ".join(f"{k[1]}@{k[2]} s{k[3]} ({', '.join(v)})" for k, v in fl) if fl else ""))
        g6 = geo_flags(R_, a.screen)   # A6
        print(A6_LINE.format(n=len(g6)) + f" in batch {a.screen}"
              + (": " + "; ".join(f"{k[1]}@{k[2]} s{k[3]} ({', '.join(v)})" for k, v in g6) if g6 else ""))
        print(screen_report(p1, site_summary(meta), elim))
        return
    picks = a.picks if a.picks is not None else sorted(str(p) for p in HERE.glob("picks_*_after*.json"))
    o = readout(R_, subs, picks, a.allow_partial, sites=site_summary(meta), elim=elim, sprobs=sprobs)   # A5: sprobs -> PROVISIONAL
    print(report(o))
    if a.json:
        conv = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
        Path(a.json).write_text(json.dumps(o, indent=1, default=conv))


if __name__ == "__main__":
    main()
