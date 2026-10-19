"""rainbow6 summary (runs/rb6, tools/pretrain_range.py Rev B --domains; intent spec runs/rb6/AUDIT.md).

    python tools/rainbow6_summary.py <runs/rb6/out> [<more dirs>] [--ref <glob or dir> ...] [--json out.json] [--allow_partial]
        [--a6_partial] [--a7_partial]

Reads every run_*.json under the directories (one sub-directory per tag; tag = rb6<phase>-53m-2pct-k<K>-<arm>).
Per arm x domain x K: area (100 x mean over all evals), final3 (100 x mean of the last 3 evals), t90 / t50 (M tokens to
the first eval >= 0.9 / 0.5, linear interpolation; None = never), computed HERE from the curves and checked against the
run JSON's own 'metrics' (double path; |diff| < 1e-9 asserted); the per-question hit counts (hits_<set>) are checked
against the curve's set and sub-split accuracies at every eval (second double path). numbers = the 6-bucket held-out
mean (readout_rules); a domain = its READ-OUT set (run JSON domains.<d>.readout): months / weekdays tier S (held-out
classes), clock tier S WITH WRAP (h + k >= 24; a no-wrap clock question is a plain addition the numbers lines train),
s4 the question split without the commuting-twin-trained and the copy (identity-input) cells, dim the relation classes
without the negation-twin-trained and the copy (classes {x, 0}) cells; tier E (cyclic: the equivalent shift k +- n of
the same item is trained; clock with wrap) as the pre-registered alternative; seen sets as the upper-bound check.
Paired by seed against the same K's none.
Integrity asserts (every run): one code_sha; finished (last eval at steps - 1; --allow_partial demotes this to a warning
and marks every gate verdict PROVISIONAL); same steps; the data mixture identical across the arms of one (seed, K)
(data_sha at every eval; step-0 lm within 1e-3); the window on exactly round-trip [0.05, 0.4) of the steps; none never
on; every active domain prior applied on >= 95% of the on-steps (silent-prior check, warning); no non-finite lm /
val_loss / geo; the gate needs >= 2 none seeds per K, the same seeds at every K.
PILOT GATE (pre-registered in runs/rb6/AUDIT.md, amendment A1 of 2026-10-17 before any run; decided on the none arm
only): per domain, the read-out set's final3 of the none seeds, with a 90% CI from a cluster bootstrap (resampling the
independence units of each seed's split: cyclic (item, k mod n) classes, s4 / dim relation classes; seeds averaged;
4000 draws, rng 0) over the per-question accuracy of the last 3 evals. floor = chance + 5 points (chance = max(uniform,
majority training answer) on the read-out set, seeds averaged). MEASURABLE iff CI within [floor, 90]; FLOOR iff CI upper
< floor; CEILING iff CI lower > 90; BORDERLINE otherwise; INVALID iff the none seen set's final3 < 80 (the eval / data
path is suspect; overrides). Cyclic domains not MEASURABLE on tier S are MEASURABLE (tier E) iff tier E passes the same
rule. Density: K* = the K with the most MEASURABLE non-number domains (tier E counts); tie -> the smaller K (A4: K 1,
2, 5). Main design at K*:
MEASURABLE = two-sided own-domain claims, FLOOR = one-sided ("the prior makes it learnable"; a null is uninterpretable),
CEILING = speed (area, t90) only, BORDERLINE / INVALID = to the user before the main design.
all6 (descriptive, never a gate): the same CI and bounds on all6 (in range / no room above / at floor), paired
differences to none per seed; per-domain geo and item-state CKA at the window's end, bpb vs none, NaN.
Amendment A2 (2026-10-17, user decisions, before any run): (a) the numbers line is the FIRST line after the text in every
row (the default path's pairs, formats and text offsets of the same seed; the text shortened at the far left by the
domain lines), so the numbers cells (the default path's eval sets) are set up to compare with the 2% main setting's
cells of the same lr: pt999 / pt2b 2pct none@1e-3 for none, win-ctx@1e-3 (helix3, window (0.05, 0.4), lambda 5) for
the numbers prior of all6 (lam_rule each: lambda 5 per active prior). Asserted: first_line_frac numbers = 1 in every run.
(b) Domain evals place the question where training does (slot 1): text + one numbers training line + the question.
(c) Held-out-FORMAT evals: per domain 3-4 paraphrase templates never used in training (run JSON domains.<d>.formats),
scored on the same questions and contexts with the same rule; per template the read-out set's final3 / area
(<d>_p<t>_readout) and the seen set; tables: training template vs each paraphrase, the drop, and the all6 - none
difference per template (format robustness of a prior effect). Descriptive (no pre-registered gate).
Amendment A3 (2026-10-18, fixes of the A2 audit, before any run; descriptive tables only, no gate change): the numbers
comparison is NOT clean on every bucket and the prior arms do not get the same optimiser step, so the numbers table
(--ref: the main setting's run JSONs, read here and asserted comparable: same model / data size / lr / eval settings,
the prior arms also the same target / window / site / lambda; paired by seed, since a seed draws the same numbers lines)
prints (1) the area over all 6 buckets, over b11 / b12 / b22 (clock lines with h + k < 24, 'k hours after h:00 is
(h+k):00', are extra addition statements of these buckets), over b13 / b23 / b33 (no domain line states an addition
with a 3-digit operand) and b33 alone (both operands 100..999: no domain line contains these tokens and no domain prior
acts on them; clock / months / weekdays lines contain the numbers 0..47); (2) per prior arm the window medians of the
global clip (the trainer clips the TOTAL gradient, LM + every active prior, to norm 1.0): the pre-clip total
T = sqrt(grad_lm^2 + grad_prior^2 + 2 grad_lm grad_prior grad_cos), the clip factor 1 / max(1, T), the share of the
on-evals with T > 1, the numbers prior's weighted gradient after the clip and relative to T, the LM gradient after the
clip. An all6 - win-ctx difference is the joint effect of the domain lines, the five other priors and this shared
clip, not by itself interference. (3) Held-out formats: a template that contains the training template's prompt up to
its last input slot verbatim (TRAIN_PROMPT) is NEAR (marked *); the drop is given over all templates and over the far
ones only.
Amendment A5 (2026-10-19, the MAIN design at K = 2, seeds 0-3; pre-registered in runs/rb6/AUDIT.md A5 before any main run).
(1) Fix: the per-seed lists of the 'per arm x set' table and of JSON cells are in seed order (they were in load order: the
pilot's K2 prior arms printed [s1, s0]). (2) New sections, printed only when a main-design arm is present (none,
all6-lam1, all6-lam0.3, only-<d> x 6, all6-shuf, none-lr8e-3) or with --main_readout, so the pilot's output is otherwise
unchanged: the STRICT dim read-out (no_negtwin_nonid minus every question with any trained same-answer token-transform
twin: swap / negate / swap+negate / other op; the tables of toy_struct re-implemented and asserted against every run's
classes and masks; per eval from the per-question hits), the answer-elimination heuristic level next to every floor
verdict, the gate per none-type arm (incl. none@8e-3), a per-arm matrix of the main read-outs, the pre-declared families
(F1, F1b, FE, F2-F8; paired by seed: t test, exact sign-flip, BH within each family), the headline claims, the lambda-0.3
non-inferiority lines and the gradient split per prior arm (window medians of grad_lm / T, the clip factor, grad_prior /
grad_lm; FLAG if the prior gradient exceeds the LM gradient). Seeds 0-3 decide; the confirm seeds 4-7 (none, all6-lam1,
none-lr8e-3) confirm (same sign and >= 3/4 of them, readout_rules.survives) with the 8-seed pooled value (descriptive). (3) The A3
numbers table pairs a reference only with rb6 cells of its own lr and kind (numbers prior helix3 on -> prior reference,
off -> none reference, shuffled -> none), and its ref - ref yardstick only references of one lr.
A5 after the fresh build audit (2026-10-19; user decisions): a headline claim must be significant in BOTH F1 (all6-lam1 -
none@1e-3) and F1b (all6-lam1 - none@8e-3), same sign; the pilot's tier-E harm hypothesis (formed on seeds 0, 1) is
decided on the fresh seeds 2-7 only (family FE), with seeds 0-3 and 0-7 printed beside for F1 / F1b / FE; every gate /
matrix / heuristic verdict is SEED-LEVEL (t interval across seeds; the item-bootstrap CI, which has no seed-to-seed
variance, is printed as descriptive); tier E has no heuristic line (n/a); F8 flags a mean LM cost > +4e-3 bpb; a dropped
seed or a missing pre-declared test fails loudly (unless --allow_partial / --main_readout); a mix of main-design runs with
the pilot's K 1 / 5 cells or its lambda-5 'all6' arm is refused (or, with --main_readout, excluded with a note); the
library versions recorded by the tasks' checks are compared. Read-out command: runs/rb6/AUDIT.md A5.
A5.1 (2026-10-19, after the re-review of the post-audit fixes, before any main result): (1) seed completeness: every
main-design arm must have its pre-registered seeds (0-3; 0-7 for none, all6-lam1, none-lr8e-3); a missing run makes every
family test, gate line, matrix row, confirm block and headline that needs it PROVISIONAL (n = k of K) - never decided,
never 'confirmed' - and the read-out refuses unless --allow_partial (--main_readout no longer relaxes this); a final status
line says complete / PROVISIONAL. (2) Run identity (main(), before any table): each K = 2 main-design run's JSON must match
make_rb6m_tasks.intent(arm, seed) (every trainer argument incl. lr / lam / target / prior_domains / prior_shuf /
extra_lines / tag, lam_domains, the active and shuffled prior sets, code_sha, dep_sha, the logged multiplier
gnorm_prior / gnorm_true = lambda) and its check.json must PASS (a shuffled arm failing only the CKA rise stays valid, A5
decision 3); otherwise the run is EXCLUDED with the reason printed; a lambda ratio that cannot be computed prints NOT
CHECKED. (3) Tier-E rows outside FE that contain the discovery seeds 0, 1 of none / all6-lam1 are marked exploratory; F5's
descriptive group is its real seeds 2-3; FE's headline labels say 'vs none' / 'vs none@8e-3'.
A6 (2026-10-19, user decisions after the main read-out; pre-registered in amendment A6 of
report/audits/rb6_A5_main_design.md BEFORE the A6 runs): the arms all6-lam1-lr4e-3 / -lr8e-3 and none-lr3e-3 / -lr1.6e-2 (tasks 494-509) are read only by the A6 section (a6_readout),
appended after the A5 sections, which are computed without them (byte-identical); run identity, seed completeness and
--allow_partial as A5.1; a diverged run is data (listed; values over all evals count, its bpb the worst).
A6.1 (2026-10-19, before any A6 result; the fresh audit of A6, report notes/batch1_build/audit_rb6_a6.md): (5)'s claim is
HOLDS / DOES NOT HOLD / NOT ESTABLISHED by every none lr's seed-level gate (the lrs that break FLOOR named), PROVISIONAL
(n=k of K) without a direction when a none run is missing; dim also shows none@8e-3's confirm seeds 4-7 (descriptive);
BH's m in every family line; a diverged run's gradient split 'pre-onset evals only (k of K)', no FLAG from < half the
on-evals; the finiteness assert back for non-A6 runs; strict mode refuses before any A6 conclusion; --a6_partial relaxes
the A6 section only (the A5 section stays strict and byte-identical; --allow_partial marks it PROVISIONAL by A5.1: 2 lines).
A7 (2026-10-19, user decisions after the A6 read-out; pre-registered in amendment A7 of report/audits/rb6_A5_main_design.md
BEFORE the A7 runs): none@3.2e-2 and all6@5e-4 at seeds 0-3 and none@1.6e-2 at seeds 4-7 (tasks 510-521) are read only by
the A7 section (a7_readout: S4 reliability vs the per-seed Latin-propagation level, speed to 25%, best vs best on s4,
none's s4 bracket over five lrs, all6's best lr over four and F1c updated), appended after A6; every section above is
computed without them (byte-identical); --a7_partial relaxes only the A7 section.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

B = ("b11", "b12", "b13", "b22", "b23", "b33")
EXTRA = ("months", "weekdays", "clock", "s4", "dim")
CYCLIC = ("months", "weekdays", "clock")
TAG = re.compile(r"^rb6(?P<phase>[a-z]+)-(?P<size>\w+)-(?P<share>\w+)-k(?P<k>\d+)-(?P<arm>.+)$")
SUBS = {"clock_held": ("wrap", "modwrap", "nowrap_numtrain", "nowrap_numheld"),
        "s4_held": ("fresh", "comm_twin_trained", "nonid", "fresh_nonid"),
        "dim_held": ("no_negtwin", "negtwin_trained", "nonid", "no_negtwin_nonid", "times", "per"),
        "months_held": ("wrap", "modwrap", "k_ge_n"), "weekdays_held": ("wrap", "modwrap", "k_ge_n")}
CEIL, MARGIN, SEEN_MIN, LM0_TOL, N_BOOT, CI = 90.0, 5.0, 80.0, 1e-3, 4000, 0.90
# A3: numbers bucket groups (see the docstring) and the training templates' prompts up to the last input slot, in PARA's
# slot notation (pretrain_range.dom_q: '5 months after March is', '9 hours after 14:00 is', ' gez then cuk is',
# ' zaj times roz is' / ' zaj per roz is')
NUM_GROUPS = (("all6", B), ("b11/12/22", ("b11", "b12", "b22")), ("b13/23/33", ("b13", "b23", "b33")), ("b33", ("b33",)))
TRAIN_PROMPT = {"months": ("{k} {u} after{I}",), "weekdays": ("{k} {u} after{I}",), "clock": ("{k} {u} after {I}:00",),
                "s4": ("{A} then{B}",), "dim": ("{A} times{B}", "{A} per{B}")}
# A3: a main-setting reference run must match the rb6 runs on these args (and steps); a prior reference also PRIOR_REF
REF_SAME = ("tokenization", "lr", "tokens", "bs", "seq", "d_model", "n_layers", "n_heads", "d_ff", "arith_lines", "n_eval",
            "eval_every", "untie", "geo_layers")
PRIOR_REF = {"target": "helix3", "schedule": "window", "window": "0.05,0.4", "site": "ctx", "lam": 5.0}


def is_near(d, tpl):
    """A3: held-out-format template tpl (a str, or dim's (times, per) pair) keeps the training prompt of domain d."""
    return any(t in p for p in ([tpl] if isinstance(tpl, str) else tpl) for t in TRAIN_PROMPT[d])


def num_area(r, bs):
    """100 x mean over all evals of the held-out accuracy averaged over the buckets bs (bs = B: the 'numbers' area)."""
    return 100 * float(np.mean([np.mean([e[f"held_{b}"] for b in bs]) for e in r["curve"]]))


def clip_stats(r):
    """A3: the global clip during the window. Per on-eval with the gradient decomposition logged (before clipping):
    T = |g_lm + g_prior| from grad_lm, grad_prior (all active priors, weighted) and grad_cos; c = 1 / max(1, T) (the
    trainer's clip_grad_norm_(.., 1.0)); the numbers prior's weighted gradient after the clip c |g_num| and relative to
    T (rb6: gnorm_prior_numbers; the default path: grad_prior, its only prior); the LM gradient after the clip c grad_lm.
    Medians over those evals, frac_clipped = share with T > 1; None if no such eval (none arms)."""
    rows = []
    for e in r["curve"]:
        if not e.get("prior_on") or e.get("grad_prior") is None or e.get("grad_lm") is None:
            continue
        gl, gp, cs = e["grad_lm"], e["grad_prior"], e["grad_cos"]
        T = math.sqrt(max(gl * gl + gp * gp + 2 * gl * gp * cs, 0.0))
        gn = e.get("gnorm_prior_numbers") if r["args"].get("domains") else gp
        c = 1 / max(1.0, T)
        rows.append((T, c, None if gn is None else c * gn, None if gn is None else gn / T, c * gl))
    if not rows:
        return None
    med = lambda i: (lambda v: float(np.median(v)) if v else None)([x[i] for x in rows if x[i] is not None])
    return {"n": len(rows), "T": med(0), "clip": med(1), "num_after_clip": med(2), "num_rel_T": med(3), "lm_after_clip": med(4),
            "frac_clipped": float(np.mean([x[0] > 1 for x in rows]))}


def load_refs(pats, runs):
    """A3: the main setting's run JSONs (globs or directories) -> {tag: {seed: run}}; asserted comparable with the rb6
    runs (REF_SAME, steps; prior references also PRIOR_REF), default path (no --domains), finished, one run per seed."""
    files = []
    for p in pats:
        hits = sorted(glob.glob(p))
        assert hits, f"--ref {p}: nothing found"
        for h in map(Path, hits):
            files += [h] if h.is_file() else sorted(h.rglob("run_*.json"))
    refs, a0, st0 = defaultdict(dict), runs[0]["r"]["args"], runs[0]["r"]["steps"]
    lrs = {x["r"]["args"]["lr"] for x in runs}   # A5: the main design has two lrs (none-lr8e-3); a reference is paired
    for f in sorted(set(f.resolve() for f in files)):   # only with the rb6 cells of its own lr (main(): rb6_kind)
        r = json.loads(f.read_text()); a = r["args"]
        assert not a.get("domains"), (str(f), "a --domains run is not a main-setting reference")
        bad = {k: [a.get(k), a0.get(k)] for k in REF_SAME if a.get(k) != a0.get(k) and not (k == "lr" and a.get(k) in lrs)}
        if r["steps"] != st0:
            bad["steps"] = [r["steps"], st0]
        kind = "none" if a["target"] == "none" else "prior"
        if kind == "prior":
            bad.update({k: [a.get(k), v] for k, v in PRIOR_REF.items() if a.get(k) != v})
        assert not bad, (str(f), "reference not comparable with the rb6 runs [ref, rb6 / wanted]", bad)
        assert r["curve"][-1]["step"] == r["steps"] - 1, (str(f), "unfinished reference")
        tag = a["tag"].lstrip("_")
        assert a["seed"] not in refs[tag], (str(f), "two reference runs of one (tag, seed)")
        refs[tag][a["seed"]] = dict(r=r, kind=kind)
    return refs


def series(r, key):
    if key == "numbers":
        return [(e["step"], float(np.mean([e[f"held_{b}"] for b in B]))) for e in r["curve"]]
    if key == STRICT:   # A5: the strict dim set, from the per-question hits (no curve key of its own)
        return strict_series(r)
    return [(e["step"], e[key]) for e in r["curve"] if e.get(key) is not None]


def metrics(r, key):
    """Own implementation (double path of pretrain_range.curve_metrics)."""
    pst = r["args"]["bs"] * r["args"]["seq"]
    pts = [(s * pst, v) for s, v in series(r, key)]
    if not pts:
        return None

    def t_th(th):
        prev = None
        for tk, v in pts:
            if v >= th:
                return tk / 1e6 if prev is None else (prev[0] + (th - prev[1]) / (v - prev[1]) * (tk - prev[0])) / 1e6
            prev = (tk, v)
        return None
    vals = [v for _, v in pts]
    return {"area": 100 * float(np.mean(vals)), "final": 100 * vals[-1], "final3": 100 * float(np.mean(vals[-3:])),
            "t50": t_th(0.5), "t90": t_th(0.9), "n_evals": len(vals)}


def readout_key(r, d, tier="S"):
    """Curve key of domain d's read-out set (tier S, or E for the cyclic fallback) and the per-question view of it:
    (key, eval set name, sub-split mask name or None)."""
    suf = r["domains"][d]["readout" if tier == "S" else "readoutE"]
    base = "heldE" if suf.startswith("heldE") else "held"
    sub = suf[len(base) + 1:] or None
    return f"{d}_{suf}", f"{d}_{base}", sub


def eval_info(r, d, nm):
    """domains.<d>.eval_<nm>; a held-out-format set (A2: eval_p<t>_<set>) takes its questions, units and masks from
    the training-template set it names (same_questions_as), its own contexts / row count."""
    info = r["domains"][d][f"eval_{nm}"]
    if "same_questions_as" in info:
        base = r["domains"][d][info["same_questions_as"]]
        assert base["questions"] == info["questions"] and base["contexts"] == info["contexts"], (d, nm)
        info = dict(base, **info)
    return info


def para_sets(r, d):
    """A2: the held-out-format eval set names of domain d (eval_p<t>_<set>), e.g. ['p1_held', 'p1_seen', ...]."""
    return [k[len("eval_"):] for k in r["domains"][d] if re.match(r"^eval_p\d+_", k)]


def per_question(r, d, base, sub):
    """(per-question accuracy over the last 3 evals, bootstrap unit per question) on the set base restricted to sub."""
    info = eval_info(r, d, base[len(d) + 1:])
    nctx, cls = info["contexts"], np.array(info["cls"])
    last = r["curve"][-3:]
    acc = np.mean([[int(c) for c in e[f"hits_{base}"]] for e in last], 0) / nctx
    m = np.ones(len(cls), dtype=bool) if sub is None else np.array([c == "1" for c in info["masks"][sub]])
    return acc[m], cls[m]


def boot_ci(rs, d, tier="S", seed=0):
    """Point (mean over seeds of the read-out final3) and the 90% cluster-bootstrap CI (units resampled within each
    seed's split, seeds averaged), in points; per-seed final3."""
    rng = np.random.default_rng(seed)
    draws, per_seed = np.zeros(N_BOOT), []
    for r in rs:
        key, base, sub = readout_key(r, d, tier)
        acc, cls = per_question(r, d, base, sub)
        f3 = metrics(r, key)["final3"]
        assert abs(100 * acc.mean() - f3) < 1e-9, (d, tier, "per-question hits != curve final3", 100 * acc.mean(), f3)
        per_seed.append(f3)
        units = np.unique(cls)
        s_u = np.array([acc[cls == u].sum() for u in units]); n_u = np.array([(cls == u).sum() for u in units])
        ix = rng.integers(0, len(units), size=(N_BOOT, len(units)))
        draws += 100 * s_u[ix].sum(1) / n_u[ix].sum(1)
    draws /= len(rs)
    lo, hi = np.quantile(draws, [(1 - CI) / 2, 1 - (1 - CI) / 2])
    return float(np.mean(per_seed)), float(lo), float(hi), per_seed


def verdict_of(lo, hi, floor):
    if hi < floor:
        return "FLOOR"
    if lo > CEIL:
        return "CEILING"
    if lo >= floor and hi <= CEIL:
        return "MEASURABLE"
    return "BORDERLINE"


def load(roots):
    runs = []
    for p in sorted({p.resolve() for root in roots for p in Path(root).rglob("run_*.json")}):
        r = json.loads(p.read_text())
        tag = r["args"]["tag"].lstrip("_")
        m = TAG.match(tag)
        assert m, tag
        assert r["args"]["domains"], (p, "not a --domains run")
        runs.append(dict(r=r, path=p, tag=tag, k=int(m["k"]), arm=m["arm"], phase=m["phase"], seed=r["args"]["seed"]))
        assert runs[-1]["k"] == r["args"]["extra_lines"], (p, "tag K != --extra_lines")
    assert runs, f"no run_*.json under {roots}"
    return runs


def check(runs, partial, a6_partial=False, a7_partial=False):
    """Integrity asserts; returns a list of warnings (soft checks). A6.1: an A6 arm's non-finite values are a DIVERGED
    warning (data), every other run keeps the finiteness assert; --a6_partial lets only A6 runs be unfinished."""
    warn = []
    shas = {x["r"]["code_sha"] for x in runs}
    assert len(shas) == 1, f"mixed code_sha {shas}"
    steps = {x["r"]["steps"] for x in runs}
    assert len(steps) == 1, f"mixed steps {steps}"
    keys = defaultdict(list)
    for x in runs:   # a pilot run reused in the main design is the same (arm, K, seed): never two copies
        keys[(x["arm"], x["k"], x["seed"])].append(x)
    assert all(len(v) == 1 for v in keys.values()), "duplicate (arm, K, seed)"
    for x in runs:
        r = x["r"]; c = r["curve"]
        a6arm = x["arm"] in A6_ARMS + A7_ARMS   # A7: its lr arms are A6's kind (a diverged run is data)
        if c[-1]["step"] != r["steps"] - 1:
            msg = f"{x['tag']} s{x['seed']}: unfinished (last eval {c[-1]['step']} of {r['steps']})"
            assert partial or (a6_partial and a6arm and not is_a7(x)) or (a7_partial and is_a7(x)), \
                msg + " (--allow_partial to look anyway; verdicts are then PROVISIONAL)"
            warn.append(msg)
        div = diverged_at(r) if a6arm else None   # A6: a diverged A6 run is data (pt2main / readout_rules), listed, never
        if div is not None:                       # an integrity failure (A6.1: the A6 arms only)
            warn.append(f"{x['tag']} s{x['seed']}: DIVERGED at step {div} (data: its values over all evals count, its bpb is"
                        " the worst)")
        for e in c:
            if not a6arm:   # A6.1: every non-A6 run keeps the A5 finiteness assert (4090d62)
                for k_ in ("lm", "val_loss") + tuple(f"geo_{d}" for d in EXTRA):
                    v = e.get(k_)
                    assert v is None or math.isfinite(v), (x["tag"], x["seed"], e["step"], k_, v)
            for d in EXTRA:   # second double path: per-question hits vs the set / sub-split accuracies of this eval
                for nm in ("held", "heldE", "seen") + tuple(para_sets(r, d)):
                    if f"eval_{nm}" not in r["domains"][d]:
                        continue
                    info = eval_info(r, d, nm)
                    h = np.array([int(ch) for ch in e[f"hits_{d}_{nm}"]]) / info["contexts"]
                    assert len(h) == info["questions"] == len(info["cls"]), (x["tag"], d, nm)
                    assert abs(h.mean() - e[f"{d}_{nm}"]) < 1e-9, (x["tag"], e["step"], d, nm, "hits != accuracy")
                    for sub, ms in info["masks"].items():
                        mk = np.array([ch == "1" for ch in ms])
                        want = e[f"{d}_{nm}_{sub}"]
                        assert (want is None) == (not mk.any()) and (want is None or abs(h[mk].mean() - want) < 1e-9), \
                            (x["tag"], e["step"], d, nm, sub, "hits != sub-split accuracy")
        # A2 layout: the numbers line is the first line after the text in every row; every domain has its formats
        assert abs(c[-1]["first_line_frac"]["numbers"] - 1) < 1e-12, (x["tag"], "numbers not always the first line")
        for d in EXTRA:
            assert r["domains"][d].get("formats") and len(para_sets(r, d)) == 2 * len(r["domains"][d]["formats"]), (x["tag"], d)
        pri = [d for d in r["domains"] if r["domains"][d].get("prior")]
        last = c[-1]
        if not pri:
            assert last["prior_on_frac"] == 0 and all(v == 0 for v in last["prior_on_domains"].values()), (x["tag"], "none was on")
        else:
            w0, w1 = (float(v) for v in r["args"]["window"].split(","))
            want = sum(1 for s in range(last["step"] + 1) if w0 * r["steps"] <= s < w1 * r["steps"])
            got = round(last["prior_on_frac"] * (last["step"] + 1))
            assert got == want, (x["tag"], x["seed"], "window steps", got, want)
            for d in pri:
                u = round(last[f"prior_used_frac_{d}"] * (last["step"] + 1))
                if u < 0.95 * want:
                    warn.append(f"{x['tag']} s{x['seed']}: prior {d} applied on {u}/{want} on-steps (< 95%)")
        # the metrics written by the trainer = this script's (double path)
        pairs = [(d, d if d in ("numbers", "clock_held_wrap") else f"{d}_held") for d in ("numbers",) + EXTRA + ("clock_held_wrap",)]
        pairs += [(f"{d}_{t}", readout_key(r, d, "S" if t == "readout" else "E")[0]) for d in EXTRA
                  for t in ("readout", "readoutE") if t in r["domains"][d]]
        pairs += [(f"{d}_p{t}_{w}", f"{d}_p{t}_{r['domains'][d]['readout'] if w == 'readout' else 'seen'}") for d in EXTRA
                  for t in range(1, len(r["domains"][d]["formats"]) + 1) for w in ("readout", "seen")]
        for name, key in pairs:
            mine, theirs = metrics(r, key), r["metrics"].get(name)
            assert theirs is not None and theirs.get("set", key) == key, (x["tag"], name, key)
            for k_ in ("area", "final", "final3"):
                assert abs(mine[k_] - theirs[k_]) < 1e-9, (x["tag"], name, k_, mine[k_], theirs[k_])
            for k_ in ("t50", "t90"):
                assert (mine[k_] is None) == (theirs[k_] is None) and (mine[k_] is None or abs(mine[k_] - theirs[k_]) < 1e-9)
    by = defaultdict(list)   # same data mixture across arms: per (K, seed) the data hashes and the step-0 lm agree
    for x in runs:
        by[(x["k"], x["seed"])].append(x)
    for (k, s), xs in by.items():
        ref = xs[0]["r"]["curve"]
        for x in xs[1:]:
            c = x["r"]["curve"]
            n = min(len(c), len(ref))
            assert [e["data_sha"] for e in c[:n]] == [e["data_sha"] for e in ref[:n]], (k, s, x["tag"], "data mixture differs")
            # step 0 is before the window: same data, same init; GPU kernels need not be bitwise reproducible across jobs
            assert abs(c[0]["lm"] - ref[0]["lm"]) < LM0_TOL, (k, s, x["tag"], "step-0 lm differs", c[0]["lm"], ref[0]["lm"])
    return warn


def fmt(v, w=5, p=1):
    return " " * (w - 1) + "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:{w}.{p}f}"


def domain_gate(rs, d, label):
    """Verdict of the read-out set of domain d for the runs rs (one arm, one K): tier S, then the cyclic tier-E
    fallback; the seen-set check; per-seed values."""
    ch = 100 * float(np.mean([r["domains"][d]["chance_readout"] for r in rs]))
    f_s, lo, hi, ps = boot_ci(rs, d, "S")
    v = verdict_of(lo, hi, ch + MARGIN)
    row = {"set": readout_key(rs[0], d)[0], "chance": ch, "floor": ch + MARGIN, "final3_S": f_s, "ci_S": [lo, hi],
           "per_seed_S": ps, "both_seeds_in_range_S": all(ch + MARGIN <= p <= CEIL for p in ps), "verdict_S": v,
           "verdict": v, "tier": "S" if v == "MEASURABLE" else None}
    if d in CYCLIC and v != "MEASURABLE":
        chE = 100 * float(np.mean([r["domains"][d]["chance_readoutE"] for r in rs]))
        f_e, loE, hiE, psE = boot_ci(rs, d, "E")
        vE = verdict_of(loE, hiE, chE + MARGIN)
        row.update(setE=readout_key(rs[0], d, "E")[0], chanceE=chE, final3_E=f_e, ci_E=[loE, hiE], per_seed_E=psE, verdict_E=vE)
        if vE == "MEASURABLE":
            row.update(verdict="MEASURABLE (tier E)", tier="E")
    seen = [metrics(r, f"{d}_seen")["final3"] for r in rs]
    row["seen_final3"] = seen
    if label == "none" and min(seen) < SEEN_MIN:
        row.update(verdict=f"INVALID (seen {min(seen):.0f} < {SEEN_MIN:.0f})", tier=None)
    return row


def rb6_kind(sd):
    """A3 numbers table: which main-setting reference an rb6 cell pairs with: 'prior' if its numbers prior is helix3 (on),
    'none' if the numbers prior is off; None (no pairing) for a shuffled numbers target (A5: only-<domain> arms pair with
    the none reference, all6-shuf with none of them; the pilot's none / all6 / all6-lam1 pair as before)."""
    ks = set()
    for r in sd.values():
        n = r["domains"]["numbers"]
        ks.add("none" if not n["prior"] else ("prior" if n["target"] == "helix3" else None))
    assert len(ks) == 1, ks
    return ks.pop()


# ==================================================================================================================
# Amendment A5 (2026-10-19, the MAIN design, pre-registered before any main run; runs/rb6/AUDIT.md A5): K = 2, 4 seeds;
# arms none, all6-lam1 (= "all6"), all6-lam0.3, only-<d> x 6, all6-shuf, none-lr8e-3. The sections below are printed only
# when a main-design arm is present (or --main_readout): the pilot's output is unchanged.
MAIN_K, ALL6M = 2, "all6-lam1"
DOM6 = ("numbers",) + EXTRA
SINGLES = tuple(f"only-{d}" for d in DOM6)
PILOT_ARMS = ("none", "all6", "all6-lam1")
MAIN_ARMS = ("none", ALL6M, "all6-lam0.3", *SINGLES, "all6-shuf", "none-lr8e-3")
A6_ARMS = ("all6-lam1-lr4e-3", "all6-lam1-lr8e-3", "none-lr3e-3", "none-lr1.6e-2")   # A6: read by a6_readout only
A7_ARMS = ("none-lr3.2e-2", "all6-lam1-lr5e-4")   # A7: + none-lr1.6e-2 at seeds 4-7 (is_a7); read by a7_readout only
TIER_E = ("months", "weekdays")   # A5 fix 1: the pilot's tier-E harm hypothesis, decided on the fresh seeds 2-7 only
STRICT = "dim_held_strict"    # A5: the main dim read-out (no same-answer token-transform twin trained; see strict_mask)
# the main read-out of every domain (pilot gate at K* = 2: months / weekdays MEASURABLE on tier E only, clock / dim
# MEASURABLE, s4 FLOOR -> one-sided vs a floor baseline): (curve key, statistic)
READ = {"numbers": ("numbers", "area"), "months": ("months_heldE", "final3"), "weekdays": ("weekdays_heldE", "final3"),
        "clock": ("clock_held_wrap", "final3"), "s4": ("s4_held_fresh_nonid", "final3"), "dim": (STRICT, "final3")}
Q_BH, NI = 0.05, {"numbers": 3.0, "other": 5.0}   # BH level; A5 non-inferiority margins (points) for "the gain is kept"
DECIDE, CONFIRM = (0, 1, 2, 3), (4, 5, 6, 7)       # readout_rules.MAIN / CONFIRM: seeds 0-3 decide, 4-7 (the confirm
#                                                   blocks of none / all6-lam1 / none-lr8e-3) confirm; 0-7 pooled descriptive
FRESH, ALL8 = (2, 3, 4, 5, 6, 7), tuple(range(8))   # A5 fix 1: seeds never seen by the pilot (0, 1 = discovery seeds)
GROUPS = {"0-3": DECIDE, "2-7": FRESH, "4-7": CONFIRM, "0-7": ALL8, "2-3": (2, 3)}
CONF_ARMS = ("none", ALL6M, "none-lr8e-3")   # the arms with the confirm seeds 4-7 (tasks 470-477, 490-493)
DISCOVERY = (0, 1)                            # the pilot's runs of none / all6-lam1 (the tier-E hypothesis was formed on them)
ENV311 = {"torch": "2.10.0", "transformers": "4.57.6"}   # task 331-env311's log (torch 2.10.0+cu128)


def expected_seeds(arm):
    """A5.1: the pre-registered seed set of a main-design arm (seeds 0-3; 0-7 for the confirm arms)."""
    return set(ALL8) if arm in CONF_ARMS else set(DECIDE)
LM_FLAG = 0.004                                    # A1 R5: offline bpb cost flag (F8: mean > +4e-3 bpb, whatever p says)
WINNER = ("none@8e-3 was selected on pt2b seeds 0-3 (area 23.1 there vs 13.8 on seeds 4-7), which share the init and the"
          " numbers data with these seeds: numbers biased in none@8e-3's favour (F1b conservative, F6 inflated)")
RANGE_N = {1: (0, 10), 2: (10, 100), 3: (100, 1000)}
BUCKETS_N = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
CYC_N = {"months": 12, "weekdays": 7, "clock": 24}
_TAB, _CACHE = {}, {}


def toy_tables():
    """The s4 / dim question tables exactly as toy_struct.build_task builds them at the pin (6248613ac6d2): row order,
    answers, relation classes (re-implemented here; toy_struct at HEAD may differ from the pinned one). Checked against
    every run's eval_held classes and masks in split_of()."""
    if _TAB:
        return _TAB
    mats = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), dtype=np.int64)
            for r_, (c_, s_) in enumerate(zip(perm, sg)):
                M[r_, c_] = s_
            if round(np.linalg.det(M)) == 1:
                mats.append(M)
    assert len(mats) == 24
    key = {tuple(M.ravel()): i for i, M in enumerate(mats)}
    mul = np.array([[key[tuple((mats[b] @ mats[a]).ravel())] for b in range(24)] for a in range(24)])   # "a then b"
    a, b = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
    X4 = np.stack([a.ravel(), b.ravel()], 1)
    Y4 = mul[X4[:, 0], X4[:, 1]]
    nc = Y4 != mul[X4[:, 1], X4[:, 0]]
    ids = {}
    cls4 = np.array([ids.setdefault((p, q) if c else (min(p, q), max(p, q)), len(ids)) for (p, q), c in zip(X4.tolist(), nc)])
    E = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)
    ek = {tuple(v): i for i, v in enumerate(E)}
    rows, ys = [], []
    for op, s in ((27, 1), (28, -1)):
        for i in range(27):
            for j in range(27):
                c = E[i] + s * E[j]
                if np.abs(c).max() <= 1:
                    rows.append((i, op, j)); ys.append(ek[tuple(c)])
    XD, YD = np.array(rows), np.array(ys)
    assert len(XD) == 686
    ids = {}
    clsD = np.array([ids.setdefault((min(p, q), max(p, q)) if op == 27 else (min(c, q), max(c, q)), len(ids))
                     for (p, op, q), c in zip(XD.tolist(), YD.tolist())])
    _TAB.update(s4=dict(X=X4, Y=Y4, cls=cls4, n=24, e=int(np.flatnonzero((mul == np.arange(24)).all(1))[0])),
                dim=dict(X=XD, Y=YD, cls=clsD, n=27, E=E, neg=np.array([ek[tuple(-v)] for v in E]),
                         row={tuple(x): i for i, x in enumerate(XD.tolist())}))
    return _TAB


def numbers_held(seed):
    """pretrain_range.held_out(default_rng(seed)) re-implemented: the unordered held-out numbers pairs (clock's forbidden
    no-wrap lines); checked against domains.numbers.held_pairs."""
    rng, out = np.random.default_rng(seed), set()
    for la, lb in BUCKETS_N:
        n, s = (20 if (la, lb) == (1, 1) else 150), set()
        while len(s) < n:
            a, b = int(rng.integers(*RANGE_N[la])), int(rng.integers(*RANGE_N[lb]))
            s.add((min(a, b), max(a, b)) if la == lb else (a, b))
        out |= {(min(a, b), max(a, b)) for a, b in s}
    return out


def bits(v):
    return "".join("1" if x else "0" for x in v)


def split_of(r, d):
    """A5: a domain's training questions and held-out sets rebuilt from the run JSON (cyclic: every (i, k) minus the tier S
    / E held sets minus clock's forbidden lines; s4 / dim: the table minus the held set), asserted against the run's own
    record (n_train, classes, masks)."""
    ck = (id(r), "split", d)
    if ck in _CACHE and _CACHE[ck][0] is r:   # the cache holds r, so its id cannot be reused while cached
        return _CACHE[ck][1]
    D = r["domains"][d]
    held = [tuple(q) if d in CYC_N else int(q) for q in D["eval_held"]["q"]]
    if d in CYC_N:
        n = CYC_N[d]
        heldE = [tuple(q) for q in D["eval_heldE"]["q"]]
        allq = [(i, k) for i in range(n) for k in list(range(1, n)) + list(range(n + 1, 2 * n))]
        forbid = set()
        if d == "clock":
            nh = numbers_held(r["args"]["seed"])
            assert len(nh) == r["domains"]["numbers"]["held_pairs"], "numbers held pairs"
            forbid = {(i, k) for i, k in allq if i + k < 24 and (min(i, k), max(i, k)) in nh}
        ex = set(held) | set(heldE) | forbid
        train = [q for q in allq if q not in ex]
        assert len(train) == D["n_train"], (d, len(train), D["n_train"])
        assert D["eval_held"]["cls"] == [i * n + k % n for i, k in held] and D["eval_heldE"]["cls"] == [i * n + k % n for i, k in heldE]
        out = dict(train=train, held=held, heldE=heldE, n=n)
    else:
        T = toy_tables()[d]
        hs = set(held)
        train = [q for q in range(len(T["X"])) if q not in hs]
        assert len(train) == D["n_train"] and len(held) == D["n_held"], (d, len(train), D["n_train"])
        assert D["eval_held"]["cls"] == [int(T["cls"][q]) for q in held], (d, "relation classes != the rebuilt table")
        X, Y, mk = T["X"], T["Y"], D["eval_held"]["masks"]
        trs = set(train)
        if d == "s4":
            nonid = [not (X[q, 0] == T["e"] or X[q, 1] == T["e"]) for q in held]
            fresh = [int(T["cls"][q]) not in {int(T["cls"][j]) for j in train} for q in held]
            assert mk["nonid"] == bits(nonid) and mk["fresh"] == bits(fresh), "s4 masks != the rebuilt table"
        else:
            copy = [Y[q] == X[q, 0] or Y[q] == X[q, 2] or (X[q, 1] == 28 and X[q, 0] == X[q, 2]) for q in held]
            tw = [T["row"][(X[q, 0], 55 - X[q, 1], T["neg"][X[q, 2]])] in trs for q in held]
            assert mk["nonid"] == bits([not c for c in copy]) and mk["negtwin_trained"] == bits(tw), "dim masks != the rebuilt table"
        out = dict(train=train, held=held, n=T["n"])
    _CACHE[ck] = (r, out)
    return out


def strict_mask(r):
    """A5: the strict dim read-out over eval_held: no_negtwin_nonid AND no same-answer token-transform twin in training.
    Twins of a held question (a, op, b) with answer c: every (x, op2, y) with (x, y) = (a, b) or (b, a), either input
    negated or not, op2 = times or per (15 transforms besides the question itself) that exists and has answer c. The
    algebraic identities of the exponent group (times: b*a, a/(-b), b/(-a); per: a*(-b), (-b)*a, (-b)/(-a)) are asserted
    to be among them for every question; twins in the question's own relation class are asserted untrained (class split).
    Returns (mask, counts by twin kind)."""
    ck = (id(r), "strict")
    if ck in _CACHE and _CACHE[ck][0] is r:   # the cache holds r, so its id cannot be reused while cached
        return _CACHE[ck][1]
    T, sp = toy_tables()["dim"], split_of(r, "dim")
    X, Y, neg, row, cls = T["X"], T["Y"], T["neg"], T["row"], T["cls"]
    trs = set(sp["train"])
    ro = np.array([c == "1" for c in r["domains"]["dim"]["eval_held"]["masks"]["no_negtwin_nonid"]])
    keep, kinds = [], defaultdict(int)
    for q, m in zip(sp["held"], ro):
        a, op, b = (int(v) for v in X[q])
        c = int(Y[q])
        na, nb = int(neg[a]), int(neg[b])
        ident = ({(b, 27, a): "swap b*a", (a, 28, nb): "negation twin a/(-b)", (b, 28, na): "swap+negate b/(-a)"} if op == 27
                 else {(a, 27, nb): "negation twin a*(-b)", (nb, 27, a): "swap+negate (-b)*a", (nb, 28, na): "swap+negate (-b)/(-a)"})
        for t in ident:
            assert t in row and Y[row[t]] == c, ("identity twin missing or another answer", q, t)
        twins = {}   # distinct trained same-answer twin rows j -> label (identity form, else the first transform reaching j)
        for (x1, x2), n1, n2, op2 in itertools.product(((a, b), (b, a)), (0, 1), (0, 1), (27, 28)):
            t = (int(neg[x1]) if n1 else x1, op2, int(neg[x2]) if n2 else x2)
            j = row.get(t)
            if j is None or j == q or Y[j] != c:
                continue
            assert not (cls[j] == cls[q] and j in trs), "a twin of the own relation class is trained"
            if j in trs and j not in twins:
                tj = tuple(int(v) for v in X[j])
                twins[j] = ident.get(tj) or ("coincidental " + ("swap+" if (x1, x2) == (b, a) else "") + ("negA+" if n1 else "")
                                             + ("negB+" if n2 else "") + ("op" if op2 != op else "same op"))
        if m:
            for lab in twins.values():
                kinds[lab] += 1
        keep.append(bool(m) and not twins)
    _CACHE[ck] = (r, (np.array(keep), dict(kinds)))
    return _CACHE[ck][1]


def strict_series(r):
    """Per-eval accuracy on the strict dim set from the per-question hit counts (hits_dim_held / contexts)."""
    m, _ = strict_mask(r)
    nctx = r["domains"]["dim"]["eval_held"]["contexts"]
    return [(e["step"], float(np.mean([int(ch) for ch, k in zip(e["hits_dim_held"], m) if k])) / nctx) for e in r["curve"]]


def heuristic(r, d, key):
    """A5 (audit 2026-10-19): accuracy of the answer-elimination heuristic on a read-out set, in points: answer uniformly
    among the answers NOT given by a trained question with the same start item (cyclic) / the same row or column (s4,
    a Latin square) / the same first input and operation (dim, among the answers possible for them). Tier E = 0 by
    construction (its answer is trained for the same item)."""
    ck = (id(r), "heur", key)
    if ck in _CACHE and _CACHE[ck][0] is r:   # the cache holds r, so its id cannot be reused while cached
        return _CACHE[ck][1]
    sp = split_of(r, d)
    acc = []
    if d in CYC_N:
        n = sp["n"]
        frm = defaultdict(set)
        for i, k in sp["train"]:
            frm[i].add((i + k) % n)
        tier = "heldE" if key.startswith(f"{d}_heldE") else "held"
        qs = sp[tier]
        msk = [i + k >= n for i, k in qs] if key.endswith("_wrap") else [True] * len(qs)
        for (i, k), m in zip(qs, msk):
            if m:
                cand = set(range(n)) - frm[i]
                acc.append(1 / len(cand) if (i + k) % n in cand else 0.0)
    else:
        T = toy_tables()[d]
        X, Y = T["X"], T["Y"]
        if key == STRICT:
            msk = strict_mask(r)[0]
        else:
            sub = key[len(f"{d}_held_"):]
            msk = np.array([c == "1" for c in r["domains"][d]["eval_held"]["masks"][sub]])
        for q, m in zip(sp["held"], msk):
            if not m:
                continue
            if d == "s4":
                used = {int(Y[j]) for j in sp["train"] if X[j, 0] == X[q, 0] or X[j, 1] == X[q, 1]}
                cand = set(range(24)) - used
            else:
                same = [j for j in range(len(X)) if X[j, 0] == X[q, 0] and X[j, 1] == X[q, 1]]
                trs = set(sp["train"])
                cand = {int(Y[j]) for j in same} - {int(Y[j]) for j in same if j in trs}
            acc.append(1 / len(cand) if int(Y[q]) in cand else 0.0)
    _CACHE[ck] = (r, 100 * float(np.mean(acc)) if acc else None)
    return _CACHE[ck][1]


def set_chance(r, d, key):
    """chance = max(uniform, majority training answer) on a read-out set (the run's own value; recomputed for the strict
    dim set)."""
    if key != STRICT:
        D = r["domains"][d]
        return D["chance_readoutE"] if key.startswith(f"{d}_heldE") else D["chance_readout"]
    T, sp = toy_tables()["dim"], split_of(r, "dim")
    maj = int(np.bincount(T["Y"][sp["train"]], minlength=27).argmax())
    m = strict_mask(r)[0]
    return max(1 / 27, float(np.mean([T["Y"][q] == maj for q, k in zip(sp["held"], m) if k])))


def set_question_view(r, d, key):
    """(per-question accuracy over the last 3 evals, bootstrap unit) on a read-out set given by its curve key."""
    base = f"{d}_heldE" if key.startswith(f"{d}_heldE") else f"{d}_held"
    sub = key[len(base) + 1:] or None
    if key == STRICT:
        acc, cls = per_question(r, d, base, None)
        m = strict_mask(r)[0]
        return acc[m], cls[m]
    return per_question(r, d, base, sub)


def boot_key(rs, d, key, seed=0):
    """boot_ci for any read-out set (A5: incl. the strict dim set): point, 90% cluster-bootstrap CI, per-seed final3."""
    rng = np.random.default_rng(seed)
    draws, per_seed = np.zeros(N_BOOT), []
    for r in rs:
        acc, cls = set_question_view(r, d, key)
        f3 = metrics(r, key)["final3"]
        assert abs(100 * acc.mean() - f3) < 1e-9, (d, key, "per-question hits != curve final3", 100 * acc.mean(), f3)
        per_seed.append(f3)
        units = np.unique(cls)
        s_u = np.array([acc[cls == u].sum() for u in units]); n_u = np.array([(cls == u).sum() for u in units])
        ix = rng.integers(0, len(units), size=(N_BOOT, len(units)))
        draws += 100 * s_u[ix].sum(1) / n_u[ix].sum(1)
    draws /= len(rs)
    lo, hi = np.quantile(draws, [(1 - CI) / 2, 1 - (1 - CI) / 2])
    return float(np.mean(per_seed)), float(lo), float(hi), per_seed


def seed_ci(vals, level=CI):
    """A5 fix 2: the seed-level interval: mean over seeds and the two-sided `level` Student t interval of the mean (df n -
    1; seeds are the independent units). (mean, None, None) for one seed; (mean, mean, mean) for identical values."""
    v = np.asarray(vals, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, None, None
    h = t_ppf(0.5 + level / 2, len(v) - 1) * float(v.std(ddof=1)) / math.sqrt(len(v))
    return m, m - h, m + h


def gate_key(rs, d, key):
    """A1 rule 1 on one read-out set, decided at the SEED level (A5 fix 2): the mean over seeds of final3 and its 90% t
    interval across seeds vs [floor, 90] (floor = chance + 5, chance seeds-averaged); the item-bootstrap CI (items resampled
    within seeds: no seed-to-seed variance) and its verdict are descriptive only. The heuristic line, also at the seed
    level: the per-seed difference final3 - heuristic, its 90% t interval and the number of seeds above; tier E: n/a."""
    ch = 100 * float(np.mean([set_chance(r, d, key) for r in rs]))
    f3_i, lo_i, hi_i, ps = boot_key(rs, d, key)
    m, lo, hi = seed_ci(ps)
    out = {"set": key, "seeds_n": len(rs), "chance": ch, "floor": ch + MARGIN, "final3": m, "per_seed": ps,
           "seed_ci": [lo, hi], "verdict": verdict_of(lo, hi, ch + MARGIN) if lo is not None else "n/a (one seed)",
           "item_ci": [lo_i, hi_i], "item_verdict": verdict_of(lo_i, hi_i, ch + MARGIN)}
    if key.startswith(f"{d}_heldE"):   # tier E: its answer is trained for the same item (shift k -+ n): copying that twin
        out.update(heuristic=None, heuristic_per_seed=None, seeds_above=None,   # scores 100% by construction, elimination 0
                   above_heuristic="n/a (tier E: copying the trained equivalent shift scores 100% by construction)")
        return out
    hs = [heuristic(r, d, key) for r in rs]
    dh = [p - h for p, h in zip(ps, hs)]
    _, lo_h, hi_h = seed_ci(dh)
    above = ("n/a (one seed)" if lo_h is None else "yes (seed-level lower bound > heuristic)" if lo_h > 0 else
             "no (seed-level upper bound < heuristic)" if hi_h < 0 else "undecided (the seed-level interval covers it)")
    out.update(heuristic=float(np.mean(hs)), heuristic_per_seed=hs, seeds_above=int(sum(v > 0 for v in dh)),
               above_heuristic=above, heuristic_diff_ci=[lo_h, hi_h])
    return out


# ---- paired inference (A5): t test (Student t via the regularised incomplete beta), exact sign-flip, BH -------------
def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b) (continued fraction, Numerical Recipes 6.4)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    p = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return 1 - p if t > 0 else p


def t_ppf(q, df):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2


def paired(diffs, side):
    """Paired-by-seed inference on per-seed differences: mean, sd, t, p (side 'two' or 'greater'), the exact sign-flip p
    (all 2^n sign patterns; floor 2 / 2^n two-sided, 1 / 2^n one-sided: .125 / .0625 at n = 4), 95% CI (two-sided) or
    95% lower bound (one-sided), 90% lower bound (non-inferiority), wins."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    out = {"n": n, "diffs": d.tolist(), "mean": float(d.mean()) if n else None, "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    if n < 2:
        return dict(out, sd=None, t=None, p_t=None, p_sf=None, ci=None, lb90=None)
    sd = float(d.std(ddof=1))
    stat = (lambda v: abs(v.mean())) if side == "two" else (lambda v: v.mean())
    pats = np.array(list(itertools.product((1, -1), repeat=n)))
    p_sf = float(np.mean([stat(d * s) >= stat(d) - 1e-12 for s in pats]))
    if sd <= 1e-9 * max(1.0, abs(float(d.mean()))):   # identical differences (up to float noise): t undefined; p = its
        mu = float(d.mean())                           # limit (0 for a nonzero difference in the tested direction, else 1),
        lim = 0.0 if (mu > 1e-12 or (side == "two" and mu < -1e-12)) else 1.0   # the interval collapses to the mean
        return dict(out, sd=0.0, t=None, p_t=lim, p_sf=p_sf, ci=[mu, mu] if side == "two" else [mu, None], lb90=mu,
                    degenerate=True)
    se = sd / math.sqrt(n)
    t = float(d.mean() / se)
    df = n - 1
    p_t = 2 * (1 - t_cdf(abs(t), df)) if side == "two" else 1 - t_cdf(t, df)
    ci = ([d.mean() - t_ppf(0.975, df) * se, d.mean() + t_ppf(0.975, df) * se] if side == "two"
          else [d.mean() - t_ppf(0.95, df) * se, None])
    return dict(out, sd=sd, t=t, p_t=float(min(1.0, max(0.0, p_t))), p_sf=p_sf, ci=[float(v) if v is not None else None for v in ci],
                lb90=float(d.mean() - t_ppf(0.90, df) * se))


def bh(ps):
    """Benjamini-Hochberg adjusted p-values (None stays None, not counted in m)."""
    idx = [i for i, p in enumerate(ps) if p is not None]
    m, out = len(idx), [None] * len(ps)
    order = sorted(idx, key=lambda i: ps[i])
    run = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, ps[i] * m / rank)
        out[i] = min(1.0, run)
    return out


def offline_bpb(path, r):
    """bpb_full of offline_lm.py (full val) for this run: results/<task>/offline_lm.json next to the run's tag dir, matched
    by tag, seed and code_sha (the record's key is the Burst path). None if absent."""
    f = Path(path).parent.parent / "offline_lm.json"
    if not f.exists():
        return None
    hits = [v for v in json.loads(f.read_text()).values() if v.get("tag") == r["args"]["tag"] and v.get("seed") == r["args"]["seed"]
            and v.get("code_sha") == r["code_sha"]]
    return hits[0]["bpb_full"] if len(hits) == 1 else None


def mval(x, dom):
    """The main read-out of domain dom for one run record x (numbers: area; a domain: final3 of READ[dom]; 'bpb': offline
    bpb_full x 1000; 'seen': mean seen final3 over the five domains)."""
    r = x["r"]
    if dom == "bpb":
        v = offline_bpb(x["path"], r)
        return None if v is None else 1000 * v
    if dom == "seen":
        return float(np.mean([metrics(r, f"{d}_seen")["final3"] for d in EXTRA]))
    key, stat = READ[dom]
    return metrics(r, key)[stat]


def grad_split(r):
    """Window medians at the on-evals (before the clip): LM share grad_lm / T, clip factor 1 / max(1, T), grad_prior /
    grad_lm, share of the on-evals with grad_prior > grad_lm."""
    rows = []
    div = diverged_at(r)   # A6: a diverged run's split = its on-evals before the divergence (CHECK_PY's rule)
    for e in r["curve"]:
        if not e.get("prior_on") or e.get("grad_prior") is None or not e.get("grad_lm") or (div is not None and e["step"] >= div):
            continue
        gl, gp, cs = e["grad_lm"], e["grad_prior"], e["grad_cos"]
        T = math.sqrt(max(gl * gl + gp * gp + 2 * gl * gp * cs, 0.0))
        rows.append((gl / T if T else None, 1 / max(1.0, T), gp / gl, T))
    if not rows:
        return None
    med = lambda i: float(np.median([v[i] for v in rows if v[i] is not None]))
    return {"n": len(rows), "lm_share": med(0), "clip": med(1), "prior_over_lm": med(2), "T": med(3),
            "frac_prior_gt_lm": float(np.mean([v[2] > 1 for v in rows]))}


def run_versions(x):
    """A5 fix 8: the library versions the task's check recorded (results/<task>/check.json info.versions; None for runs whose
    check predates the record, e.g. the pilot's)."""
    f = Path(x["path"]).parent.parent / "check.json"
    try:
        v = json.loads(f.read_text())["info"].get("versions")
    except (OSError, ValueError, KeyError):
        return None
    return json.dumps(v, sort_keys=True) if v else None


def run_identity(runs, arms=MAIN_ARMS):
    """A5.1 (re-review 2026-10-19): every K = 2 main-design run must be what its tag says. Its run JSON is compared with
    make_rb6m_tasks.intent(arm, seed) (the intent table the task's own check used): every trainer argument (lr, lam, target,
    prior_domains, prior_shuf, extra_lines = K, tag, seed, ...), lam_domains, the active and the shuffled prior sets
    (domains.<d>.prior / shuf), code_sha and toy_struct's dep_sha; the logged multiplier of every unshuffled active prior
    (gnorm_prior_<d> / gnorm_true_<d> = lambda, rel. 1e-3; 'not checked' where gnorm_true is missing); and check.json next
    to the run must PASS - except a shuffled arm whose only failing check is the CKA rise (A5 decision 3: a finding, the
    run stays valid). Other runs (the pilot's other cells) are not checked here. -> (kept runs, info for the read-out)."""
    import make_rb6m_tasks as G   # tools/: the generator of the submitted tasks (its intent table)
    kept, info = [], {"excluded": [], "valid_notes": [], "not_checked": [], "checked": 0}
    for x in runs:
        if x["k"] != MAIN_K or x["arm"] not in arms:
            kept.append(x)
            continue
        r, a, s = x["r"], x["arm"], x["seed"]
        lab = f"{a} s{s} ({Path(x['path']).parent.parent.name})"
        I = G.intent(a, s)
        bad, ra = [], r["args"]
        dif = {k: [ra.get(k), v] for k, v in I["args"].items() if ra.get(k) != v}
        if dif:
            bad.append(f"trainer args differ from the tag's intent [run, intent]: {dif}")
        if r.get("code_sha") != I["code_sha"] or r.get("dep_sha", {}).get("toy_struct.py") != I["toy_sha"]:
            bad.append(f"code_sha / toy_struct dep_sha {r.get('code_sha')} / {r.get('dep_sha')} != {I['code_sha']} / {I['toy_sha']}")
        if r.get("lam_domains") != I["lam_domains"]:
            bad.append(f"lam_domains {r.get('lam_domains')} != {I['lam_domains']}")
        act = sorted(d for d in DOM6 if r["domains"][d].get("prior"))
        shf = sorted(d for d in DOM6 if r["domains"][d].get("shuf"))
        if act != sorted(I["active"]) or shf != sorted(I["shuf"]):
            bad.append(f"active / shuffled priors {act} / {shf} != {sorted(I['active'])} / {sorted(I['shuf'])}")
        on = [e for e in r["curve"] if e.get("prior_on")]
        for d in I["active"]:
            if d in I["shuf"]:
                continue
            rat = [e[f"gnorm_prior_{d}"] / e[f"gnorm_true_{d}"] for e in on
                   if e.get(f"gnorm_prior_{d}") is not None and e.get(f"gnorm_true_{d}")]
            if any(abs(v / I["lam_domains"][d] - 1) >= 1e-3 for v in rat):
                bad.append(f"logged multiplier of {d} != lambda {I['lam_domains'][d]}: {min(rat):.5f}..{max(rat):.5f}")
            if len(rat) < len(on):
                info["not_checked"].append(f"{lab} {d}: lambda ratio NOT CHECKED at {len(on) - len(rat)} of {len(on)} on-evals"
                                           " (gnorm_prior / gnorm_true missing)")
        f = Path(x["path"]).parent.parent / "check.json"
        try:
            cj = json.loads(f.read_text())
        except (OSError, ValueError):
            cj = None
            bad.append("no readable check.json next to the run (its task's intent checks did not run or were not published)")
        if cj is not None and not cj.get("pass"):
            fail = [c["check"] for c in cj.get("checks", []) if not c.get("ok")]
            if I["shuf"] and fail and all(c.startswith("CKA to the target USED") for c in fail):
                info["valid_notes"].append(f"{lab}: its check FAILED only the CKA rise of the shuffled priors: a finding, the run"
                                           " stays valid (A5 decision 3)")
            else:
                bad.append(f"its check FAILED: {fail}")
        info["checked"] += 1
        if bad:
            info["excluded"].append((lab, bad))
        else:
            kept.append(x)
    return kept, info


def main_readout(runs, cells, out, partial, forced=False, ident=None):
    """A5 main-design read-out (pre-registered; AUDIT.md A5 + A5.1). Prints the sections and fills out['main']. Only the
    K = 2 cells of the main-design arms are read (MAIN_ARMS); anything else is excluded with a printed note (main() refuses
    such a mix unless --main_readout is given). A5.1: every arm must have its pre-registered seeds (expected_seeds); a
    missing run makes every verdict that needs it PROVISIONAL (n = k of K, never 'confirmed'). strict = not --allow_partial
    (--main_readout no longer relaxes it): a missing seed, a dropped seed or an incomplete family then fails loudly."""
    K = MAIN_K
    strict = not partial
    keep = {(k, a): sd for (k, a), sd in cells.items() if k == K and a in MAIN_ARMS}
    excl = sorted(f"K{k} {a}" for (k, a) in cells if (k, a) not in keep)
    runs = [x for x in runs if (x["k"], x["arm"]) in keep]
    cells = keep
    xs = {(x["k"], x["arm"], x["seed"]): x for x in runs}
    arms = sorted({a for _, a in cells})
    M = out["main"] = {"K": K, "arms": arms, "excluded": excl, "gate": {}, "matrix": {}, "families": {},
                       "noninferiority": {}, "grad": {}, "dim_strict": {}, "headline": {}, "versions": {}}
    print(f"\n== A5 MAIN-DESIGN READ-OUT (pre-registered 2026-10-19, runs/rb6/AUDIT.md A5) at K = {K}; arms {arms}")
    if excl:
        print(f"  EXCLUDED from the A5 sections (not main-design K = {K} cells; the pilot's lambda-5 arm is 'all6', the main"
              f" all6 is all6-lam1): {excl}")
    if partial:
        print("  PROVISIONAL: unfinished runs or an incomplete batch (--allow_partial)")
    M["identity"] = ident
    if ident is not None:   # A5.1: run identity and check.json (main(): run_identity)
        print(f"  run identity (tag's intent = run JSON; check.json PASS): {ident['checked']} main-design run(s) checked,"
              f" {len(ident['excluded'])} EXCLUDED")
        for lab, why in ident["excluded"]:
            print(f"  EXCLUDED run {lab}: " + "; ".join(why))
        for t in ident["valid_notes"] + ident["not_checked"]:
            print("  NOTE " + t)
    problems = []
    missing = {a: sorted(expected_seeds(a) - set(cells.get((K, a), {}))) for a in MAIN_ARMS}
    missing = {a: v for a, v in missing.items() if v}
    M["missing_seeds"] = missing
    M["provisional"] = bool(partial or missing)
    if missing:   # A5.1: a run that is not there (not finished / not published / excluded) is never silently left out
        msg = ("seed set incomplete (pre-registered: seeds 0-3, and 4-7 for " + ", ".join(CONF_ARMS) + "): missing "
               + "; ".join(f"{a} {v}" for a, v in sorted(missing.items())))
        print(f"  PROVISIONAL: {msg}")
        problems.append(msg)
    for a in ("none", ALL6M):
        if (K, a) in cells:
            print(f"  {a}: seeds {sorted(cells[(K, a)])} (0, 1 = the pilot's runs; 2-7 = main-design runs under the pilot tag "
                  f"{next(iter(cells[(K, a)].values()))['args']['tag'].lstrip('_')}: A5 reuse)")
    vs = defaultdict(list)
    for x in runs:
        vs[run_versions(x)].append(f"{x['arm']} s{x['seed']}")
    M["versions"] = {str(k): v for k, v in vs.items()}
    rec = [k for k in vs if k is not None]
    env_ok = [all(str(json.loads(k).get(p, "")).startswith(v) for p, v in ENV311.items()) for k in rec]
    print(f"  library versions (check.json info.versions): {len(rec)} distinct recorded set(s) {rec}; not recorded for "
          f"{len(vs.get(None, []))} run(s) (checks before A5, e.g. the pilot's seeds 0, 1)" + ("  WARNING: runs differ in versions" if len(rec) > 1 else "")
          + (f"; = env311 at task 331 (torch {ENV311['torch']}+cu128, transformers {ENV311['transformers']}): "
             f"{'yes' if all(env_ok) else 'NO'}" if rec else ""))
    print("  dim read-out = the STRICT set (no_negtwin_nonid without any question whose same-answer token-transform twin -"
          " swap / negate / swap+negate / other op - is trained); the pilot's no_negtwin_nonid is reported beside it")
    for (k, a), sd in sorted(cells.items()):
        for s in sorted(sd):
            m, kinds = strict_mask(sd[s])
            M["dim_strict"][f"k{k}|{a}|s{s}"] = {"n_strict": int(m.sum()), "n_no_negtwin_nonid": sd[s]["domains"]["dim"]["n_readout"],
                                                 "dropped_by_kind": kinds}
    for s in sorted({x["seed"] for x in runs}):
        vv = [v for k_, v in M["dim_strict"].items() if k_.endswith(f"|s{s}")]
        assert all(v == vv[0] for v in vv), f"strict dim set differs across the arms of seed {s}"
        print(f"  seed {s}: strict dim set {vv[0]['n_strict']} of the {vv[0]['n_no_negtwin_nonid']} no_negtwin_nonid questions"
              f" (distinct trained same-answer twins of those questions, by form: {dict(sorted(vv[0]['dropped_by_kind'].items()))})")
    # ---- 1. gate per baseline arm, at the seed level (A5 fix 2), + the heuristic line
    sets = [("months", "months_held"), ("months", "months_heldE"), ("weekdays", "weekdays_held"), ("weekdays", "weekdays_heldE"),
            ("clock", "clock_held_wrap"), ("clock", "clock_heldE_wrap"), ("s4", "s4_held_fresh_nonid"),
            ("dim", "dim_held_no_negtwin_nonid"), ("dim", STRICT)]
    print(f"\n== A5 gate per baseline arm, SEED-LEVEL (A1 rule 1 on the mean over seeds of final3 with its {CI:.0%} t interval"
          f" across seeds vs [chance + {MARGIN:.0f}, {CEIL:.0f}]; the item-bootstrap CI (items within seeds, no seed variance) is"
          " DESCRIPTIVE only) + the answer-elimination HEURISTIC line (seed level: final3 - heuristic per seed, its t interval,"
          " seeds above; tier E: n/a)")
    for (k, a), sd in sorted(cells.items()):
        if a not in ("none", "none-lr8e-3"):
            continue
        groups = [("", [s for s in sorted(sd) if s in DECIDE])]   # seeds 0-3 decide; 2-7 fresh; 4-7 confirm; 0-7 pooled
        if any(s in CONFIRM for s in sd) or set(CONFIRM) <= expected_seeds(a):
            groups += [(f" [{g}]", [s for s in sorted(sd) if s in GROUPS[g]]) for g in ("2-7", "4-7", "0-7")]
        for d, key in sets:
            for lab, ss in groups:
                if not ss:
                    continue
                rs = [sd[s] for s in ss]
                g = gate_key(rs, d, key)
                seen = [metrics(r, f"{d}_seen")["final3"] for r in rs]
                if min(seen) < SEEN_MIN:
                    g["verdict"] = f"INVALID (seen {min(seen):.0f} < {SEEN_MIN:.0f})"
                g["seeds"] = ss
                need = sorted(set(GROUPS[lab.strip(" []")] if lab else DECIDE) & expected_seeds(a))
                g["provisional"] = len(ss) < len(need)
                if g["provisional"]:
                    g["verdict"] = f"PROVISIONAL (n={len(ss)} of {len(need)}): {g['verdict']}"
                M["gate"][f"k{k}|{a}{lab.replace(' ', '')}|{key}"] = g
                sc = g["seed_ci"]
                print(f"  K{k} {a + lab:18s} {key:26s} floor {fmt(g['floor'])} final3 {fmt(g['final3'])} seed-level "
                      f"[{fmt(sc[0])},{fmt(sc[1])}] seeds {ss} [{' '.join(fmt(v) for v in g['per_seed'])}] -> {g['verdict']} | "
                      f"item-bootstrap CI (descriptive) [{fmt(g['item_ci'][0])},{fmt(g['item_ci'][1])}] -> {g['item_verdict']}")
                print(f"  {'':48s} heuristic " + ("n/a" if g["heuristic"] is None else
                      f"{fmt(g['heuristic'])} [{' '.join(fmt(v) for v in g['heuristic_per_seed'])}] seeds above {g['seeds_above']}/"
                      f"{len(ss)}") + f" -> above the heuristic: {g['above_heuristic']}")
    if (K, "none") in cells and (K, "none-lr8e-3") in cells:
        print(f"  F6 verdicts (seed level, seeds 0-3), none@8e-3 vs none@1e-3 at K{K}: " + "; ".join(
            f"{key} {M['gate'][f'k{K}|none|{key}']['verdict']} -> {M['gate'][f'k{K}|none-lr8e-3|{key}']['verdict']}"
            + (" CHANGED" if M['gate'][f'k{K}|none|{key}']['verdict'] != M['gate'][f'k{K}|none-lr8e-3|{key}']['verdict'] else "")
            for _, key in sets))
    print(f"\n== A5 main read-outs per arm at K{K} (mean over seeds 0-3; [{CI:.0%} seed-level t interval]; F = its upper bound <"
          " floor, H = above the heuristic at the seed level (tier E: n/a); numbers = area)")
    for a in arms:
        ss = [s for s in sorted(cells[(K, a)]) if s in DECIDE]
        if not ss:
            continue
        rs = [cells[(K, a)][s] for s in ss]
        row = {}
        for d in DOM6:
            key, stat = READ[d]
            if d == "numbers":
                v = [metrics(r, key)[stat] for r in rs]   # (A5.1: the matrix row is marked PROVISIONAL below if seeds are missing)
                m, lo, hi = seed_ci(v)
                row[d] = {"mean": m, "seed_ci": [lo, hi], "per_seed": v}
                continue
            g = gate_key(rs, d, key)
            row[d] = {"mean": g["final3"], "seed_ci": g["seed_ci"], "per_seed": g["per_seed"], "floor": g["floor"],
                      "heuristic": g["heuristic"], "above_heuristic": g["above_heuristic"], "item_ci": g["item_ci"]}
        M["matrix"][a] = row

        def cell_txt(d, v):
            lo, hi = v["seed_ci"]
            t = f"{d} {fmt(v['mean'])} [{fmt(lo)},{fmt(hi)}]"
            if d != "numbers":
                t += ("F" if hi is not None and hi < v["floor"] else " ") + ("H" if v["above_heuristic"].startswith("yes") else " ")
            return t
        need = sorted(set(DECIDE) & expected_seeds(a))
        M.setdefault("matrix_provisional", {})[a] = len(ss) < len(need)
        print(f"  {a:12s} seeds {ss} " + " | ".join(cell_txt(d, v) for d, v in row.items())
              + (f"  PROVISIONAL (n={len(ss)} of {len(need)})" if len(ss) < len(need) else ""))
    # ---- 2. the pre-declared families (paired by seed; t test, exact sign-flip, BH within each family and seed group)
    def have(a):
        return (K, a) in cells

    s4_side_8e3 = "two"
    if have("none-lr8e-3"):
        s4_side_8e3 = "greater" if M["gate"].get(f"k{K}|none-lr8e-3|s4_held_fresh_nonid", {}).get("verdict", "").endswith(
            "FLOOR") else "two"   # (a PROVISIONAL gate decides the side provisionally; its tests are PROVISIONAL anyway)
    D4 = ("numbers", "clock", "s4", "dim")   # F1 / F1b: the domains whose effect was predicted before the pilot
    # (fid, title, tests (metric, arm a, arm b, side), decision seed group, extra seed groups printed beside)
    fam = [
        ("F1", "joint own-domain effect: all6-lam1 - none (s4 one-sided: none at the floor)",
         [(d, ALL6M, "none", "greater" if d == "s4" else "two") for d in D4], "0-3", ("2-7", "0-7")),
        ("F1b", "each arm at its own best lr: all6-lam1@1e-3 - none@8e-3 (s4 one-sided iff none@8e-3's seed-level s4 gate is FLOOR)",
         [(d, ALL6M, "none-lr8e-3", s4_side_8e3 if d == "s4" else "two") for d in D4], "0-3", ("2-7", "0-7")),
        ("FE", "the pilot's tier-E harm hypothesis, decided on the FRESH seeds 2-7 (seeds 0, 1 formed it): all6-lam1 - none and"
               " all6-lam1 - none@8e-3 on months / weekdays tier E",
         [(d, ALL6M, b, "two") for b in ("none", "none-lr8e-3") for d in TIER_E], "2-7", ("0-3", "0-7")),
        ("F2", "single prior vs none, own domain (s4 one-sided)",
         [(d, f"only-{d}", "none", "greater" if d == "s4" else "two") for d in DOM6], "0-3", ()),
        ("F3", "stacking: all6-lam1 - single, own domain (> 0 synergy, < 0 interference)",
         [(d, ALL6M, f"only-{d}", "two") for d in DOM6], "0-3", ()),
        ("F4", "geometry specificity: all6-lam1 - all6-shuf, own domain", [(d, ALL6M, "all6-shuf", "two") for d in DOM6], "0-3", ()),
        ("F5", "dose: all6-lam0.3 - all6-lam1 (harm: bpb, tier E, seen, numbers; gains)",
         [(d, "all6-lam0.3", ALL6M, "two") for d in ("bpb",) + DOM6 + ("seen",)], "0-3", ("2-3",)),
        ("F6", "baseline lr: none@8e-3 - none@1e-3", [(d, "none-lr8e-3", "none", "two") for d in DOM6 + ("bpb",)], "0-3", ()),
        ("F7", "unrelated-task harm: single prior - none on every OTHER domain",
         [(e, f"only-{d}", "none", "two") for d in DOM6 for e in DOM6 if e != d], "0-3", ()),
        ("F8", "LM cost: offline bpb (full val) x 1000, prior arm - none",
         [("bpb", a, "none", "two") for a in (ALL6M, "all6-lam0.3", *SINGLES, "all6-shuf")], "0-3", ()),
    ]

    def test(aa, bb, dom, side, grp):
        """One paired test on the seeds of group grp both arms have; a seed whose metric is missing (offline bpb) is
        printed and counted as a problem (A5 fix 7)."""
        common = sorted(set(cells[(K, aa)]) & set(cells[(K, bb)]) & set(GROUPS[grp]))
        va = {s: mval(xs[(K, aa, s)], dom) for s in common}
        vb = {s: mval(xs[(K, bb, s)], dom) for s in common}
        seeds = [s for s in common if va[s] is not None and vb[s] is not None]
        if len(seeds) < len(common):
            msg = f"{aa} - {bb} {dom} seeds {grp}: DROPPED seeds {sorted(set(common) - set(seeds))} (no {dom} value: offline_lm.json missing?)"
            print("  WARNING " + msg)
            problems.append(msg)
        if not seeds:
            return None
        st = paired([va[s] - vb[s] for s in seeds], side)
        need = sorted(set(GROUPS[grp]) & expected_seeds(aa) & expected_seeds(bb))   # A5.1: the pre-registered seeds
        return dict(st, metric=dom, a=aa, b=bb, side=side, group=grp, seeds=seeds, n_common=len(common), n_required=len(need),
                    provisional=len(seeds) < len(need),
                    a_mean=float(np.mean([va[s] for s in seeds])), b_mean=float(np.mean([vb[s] for s in seeds])))

    def with_bh(rows):
        for row, q in zip(rows, bh([x["p_t"] for x in rows])):
            row["p_bh"] = q
            row["significant"] = q is not None and q < Q_BH
            row["robust"] = bool(row["significant"] and (row["wins"] == row["n"] if row["mean"] > 0 else row["losses"] == row["n"]))
        return rows

    def lab_of(dom):
        return {"bpb": "offline bpb x1e3", "seen": "seen (5 domains)"}.get(dom, f"{dom} {READ[dom][0]}" if dom in READ else dom)

    def row_txt(x, pre):
        ci = x["ci"]
        ci_t = "" if ci is None else (f" 95% CI [{fmt(ci[0], 6)},{fmt(ci[1], 6)}]" if x["side"] == "two" else f" 95% LB {fmt(ci[0], 6)}")
        return (f"{pre}{x['a']:12s} - {x['b']:12s} seeds {x['group']} {fmt(x['a_mean'], 6)} vs {fmt(x['b_mean'], 6)} Δ {fmt(x['mean'], 6)} "
                f"[{' '.join(fmt(v, 6) for v in x['diffs'])}] n {x['n']} {'1-sided' if x['side'] != 'two' else '2-sided'} t "
                f"{fmt(x['t'], 6, 2)} p {fmt(x['p_t'], 6, 4)} BH {fmt(x['p_bh'], 6, 4)} sign-flip {fmt(x['p_sf'], 6, 4)} "
                f"wins {x['wins']}/{x['n']}{ci_t} -> {('PROVISIONAL (n=' + str(x['n']) + ' of ' + str(x['n_required']) + '): ') if x['provisional'] else ''}"
                f"{'SIGNIFICANT' + (' (robust)' if x['robust'] else '') if x['significant'] else 'n.s.'}"
                + (" [identical differences in every seed: t undefined, p = its limit]" if x.get("degenerate") else ""))

    print(f"\n== A5 families (paired by seed, K{K}): mean Δ [per seed in seed order], paired t (df n-1), exact sign-flip p (floor"
          f" 2/2^n two-sided, 1/2^n one-sided: .125 / .0625 at n = 4, so it can never reach .05 there and is not a decision rule),"
          f" BH within the family and seed group; SIGNIFICANT = BH-adjusted t p < {Q_BH} on the family's DECISION seeds (F1, F1b"
          f" and F2-F8: 0-3; FE: the fresh seeds 2-7); 'robust' (all seeds in the direction of the mean) is reported, not a rule;"
          f" the other seed groups are descriptive; PROVISIONAL (n=k of K) = a pre-registered seed is missing: not decided")
    for fid, title, tests, dec, extra in fam:
        present = [t for t in tests if have(t[1]) and have(t[2])]
        if not present:
            print(f"  {fid} {title}: not computed (arms missing)")
            continue
        if len(present) < len(tests):
            msg = f"{fid}: only {len(present)} of its {len(tests)} pre-declared tests (missing arms {sorted({a for t in tests for a in t[1:3] if not have(a)})})"
            print("  WARNING family incomplete: " + msg)
            problems.append(msg)
        grp_rows = {}
        for grp in (dec,) + tuple(extra):
            rows = [r_ for r_ in (test(aa, bb, dom, side, grp) for dom, aa, bb, side in present) if r_ is not None]
            if grp == dec and len(rows) < len(present):
                msg = f"{fid}: {len(present) - len(rows)} test(s) without any seed of group {dec}"
                print("  WARNING " + msg)
                problems.append(msg)
            grp_rows[grp] = with_bh(rows)
        rows = grp_rows[dec]
        conf = {}   # the confirm seeds 4-7 of a test decided on seeds 0-3 (readout_rules.survives)
        if dec == "0-3":
            for x in rows:
                c_ = test(x["a"], x["b"], x["metric"], x["side"], "4-7")
                if c_ is None:
                    continue
                sg = 1 if x["mean"] >= 0 else -1
                agree = sg * c_["mean"] > 0 and (c_["wins"] if sg > 0 else c_["losses"]) >= 0.75 * c_["n"]
                c_["status"] = ("nothing to confirm (n.s. on seeds 0-3)" if not x["significant"] and not x["provisional"] else
                                f"PROVISIONAL (decision seeds n={x['n']} of {x['n_required']}): unconfirmed" if x["provisional"] else
                                f"PROVISIONAL (n={c_['n']} of {c_['n_required']} confirm seeds): unconfirmed, so far "
                                f"{'agrees' if agree else 'disagrees'}" if c_["provisional"] else
                                "CONFIRMED on seeds 4-7" if agree else "NOT confirmed (seeds 4-7 disagree)")
                conf[(x["a"], x["b"], x["metric"])] = c_
        for x in rows:
            x["confirm"] = conf.get((x["a"], x["b"], x["metric"]))
            x["extra"] = {g: next((y for y in grp_rows[g] if (y["a"], y["b"], y["metric"]) == (x["a"], x["b"], x["metric"])), None)
                          for g in extra}
        M["families"][fid] = {"title": title, "decision_seeds": dec, "tests": rows}
        print(f"  {fid} {title} ({len(rows)} tests; decision seeds {dec})")
        for x in rows:
            note = ""
            if x["metric"] == "numbers" and "none-lr8e-3" in (x["a"], x["b"]):
                note = f"  [winner's curse: {WINNER}]"
            if (x["metric"] in TIER_E and fid != "FE" and {x["a"], x["b"]} & {"none", ALL6M}
                    and set(x["seeds"]) & set(DISCOVERY)):   # A5.1: tier-E rows on the pilot's seeds 0, 1
                note += ("  [tier E incl. the DISCOVERY seeds 0, 1 of none / all6-lam1 (the pilot formed the tier-E hypothesis"
                         " on them): exploratory; the tier-E test is FE on seeds 2-7]")
                x["discovery_seeds"] = True
            if fid == "F8" and x["mean"] is not None and x["mean"] > 1000 * LM_FLAG:
                note += f"  LM COST FLAG (mean > +{LM_FLAG:g} bpb, A1 R5)"
                x["lm_flag"] = True
            print(row_txt(x, f"    {lab_of(x['metric']):30s} ") + note)
            for g, y in x["extra"].items():
                if y is not None:
                    print(row_txt(y, f"    {'':30s} [{'descriptive' if g != dec else 'decision'}] "))
            c_ = x["confirm"]
            if c_ is not None:
                print(f"    {'':30s} confirm seeds {c_['seeds']}: Δ {fmt(c_['mean'], 6)} [{' '.join(fmt(v, 6) for v in c_['diffs'])}] wins "
                      f"{c_['wins']}/{c_['n']} t p {fmt(c_['p_t'], 6, 4)} -> {c_['status']}")
            elif dec == "0-3" and {x["a"], x["b"]} <= set(CONF_ARMS):
                print(f"    {'':30s} confirm seeds 4-7: none yet (blocks 470-477 / 490-493) -> unconfirmed (PROVISIONAL)")
    if problems:
        print(f"  {len(problems)} problem(s): {problems}")
        assert not strict, ("A5 read-out: a pre-registered seed is missing, a seed was dropped or a pre-declared test is missing"
                            " (n and the BH m would change): " + "; ".join(problems)
                            + " (--allow_partial to look at an incomplete batch anyway: verdicts PROVISIONAL)")
    # ---- 3. headline claims (user decision 2026-10-19: in BOTH F1 and F1b, same sign)
    fams = M["families"]

    def find(fid, a, b, dom):
        return next((x for x in fams.get(fid, {}).get("tests", []) if (x["a"], x["b"], x["metric"]) == (a, b, dom)), None)
    print("\n== A5 HEADLINE claims (pre-registered, user decision 2026-10-19): a headline must be SIGNIFICANT in BOTH F1"
          " (all6-lam1 - none@1e-3) and F1b (all6-lam1 - none@8e-3), with the same sign; numbers / clock / s4 / dim decided on"
          " seeds 0-3, confirmed iff the confirm seeds 4-7 of BOTH F1 and F1b agree, withdrawn if either disagrees; months /"
          " weekdays tier E (the pilot's hypothesis)"
          " decided on the fresh seeds 2-7 (FE); 'robust' is reported, not required")
    for d in DOM6:
        if d in TIER_E:
            x1, x2, how = find("FE", ALL6M, "none", d), find("FE", ALL6M, "none-lr8e-3", d), "FE, seeds 2-7"
            l1, l2 = "FE vs none", "FE vs none@8e-3"
        else:
            x1, x2, how = find("F1", ALL6M, "none", d), find("F1b", ALL6M, "none-lr8e-3", d), "F1 / F1b, seeds 0-3"
            l1, l2 = "F1", "F1b"
        sig = lambda x: "SIGNIFICANT" if x["significant"] else "n.s."
        if x1 is None or x2 is None:
            st = f"PROVISIONAL: not decidable yet ({how}: {l1 if x1 is None else l2} missing)"
        elif x1["provisional"] or x2["provisional"]:   # A5.1: a missing pre-registered seed: never a decision
            st = (f"PROVISIONAL ({how}: {l1} n={x1['n']} of {x1['n_required']}, {l2} n={x2['n']} of {x2['n_required']}):"
                  f" not decided (so far {l1} {sig(x1)}, {l2} {sig(x2)})")
        elif not (x1["significant"] and x2["significant"] and (x1["mean"] > 0) == (x2["mean"] > 0)):
            st = (f"not a headline ({how}: {l1} {sig(x1)} (n {x1['n']}), {l2} {sig(x2)} (n {x2['n']})"
                  + (", opposite signs" if x1["significant"] and x2["significant"] else "") + ")")
        else:
            dirn = "gain" if x1["mean"] > 0 else "harm"
            cs = {"F1": x1.get("confirm"), "F1b": x2.get("confirm")}
            bad = [f for f, c_ in cs.items() if c_ is not None and c_["status"].startswith("NOT confirmed")]
            miss = [f for f, c_ in cs.items() if c_ is None or c_["status"].startswith("PROVISIONAL")]
            if d in TIER_E:
                st = f"HEADLINE {dirn} ({how}; decided on fresh seeds, no separate confirmation)"
            elif bad:
                st = f"not a headline: significant in F1 and F1b ({how}) but the confirm seeds 4-7 of {'/'.join(bad)} disagree"
            elif miss:
                st = f"HEADLINE {dirn} ({how}), unconfirmed: PROVISIONAL (confirm seeds 4-7 of {'/'.join(miss)} missing or incomplete)"
            else:
                st = f"HEADLINE {dirn} ({how}), confirmed on seeds 4-7 (F1 and F1b)"
        M["headline"][d] = st
        print(f"  {d:9s} {READ[d][0]:22s} {st}"
              + (f"  [numbers F1b: {WINNER}]" if d == "numbers" and x2 is not None else ""))
    # ---- 4. non-inferiority of the gains at lambda 0.3 (A5: 'the gain is kept')
    if have("all6-lam0.3") and have(ALL6M):
        print(f"  F5 non-inferiority (gain kept at lambda 0.3): the 90% one-sided lower bound of all6-lam0.3 - all6-lam1 > -margin"
              f" (numbers {NI['numbers']:.0f} area points, domains {NI['other']:.0f} final3 points); read only where F1 (FE for tier E)"
              f" shows a significant gain")
        for x in fams.get("F5", {}).get("tests", []):
            if x["metric"] not in DOM6:
                continue
            g1 = find("FE", ALL6M, "none", x["metric"]) if x["metric"] in TIER_E else find("F1", ALL6M, "none", x["metric"])
            mg = NI["numbers"] if x["metric"] == "numbers" else NI["other"]
            ok = x["lb90"] is not None and x["lb90"] > -mg
            gain = bool(g1 and g1["significant"] and g1["mean"] > 0)
            prov = x["provisional"] or bool(g1 and g1["provisional"])
            M["noninferiority"][x["metric"]] = {"lb90": x["lb90"], "margin": mg, "kept": ok, "F1_gain": gain, "provisional": prov}
            print(f"    {x['metric']:9s} lb90 {fmt(x['lb90'], 6)} vs -{mg:.0f}: {'kept' if ok else 'not shown'}"
                  f" (F1 gain: {'yes' if gain else 'no'})" + (" PROVISIONAL (seeds missing)" if prov else "")
                  + ("  [tier E incl. the discovery seeds 0, 1 of all6-lam1: exploratory]" if x["metric"] in TIER_E else ""))
    # ---- 5. gradient split per prior arm (user principle: the prior is auxiliary)
    print("\n== A5 gradient split at the on-evals, every prior arm and seed (window medians before the global clip): LM share"
          " grad_lm / T, clip factor 1/max(1, T), grad_prior / grad_lm, share of on-evals with grad_prior > grad_lm; FLAG ="
          " the prior gradient exceeds the LM gradient (median ratio > 1 in any seed)")
    for (k, a), sd in sorted(cells.items()):
        gs = {s: grad_split(sd[s]) for s in sorted(sd)}
        if not any(gs.values()):
            continue
        flag = any(g and g["prior_over_lm"] > 1 for g in gs.values())
        M["grad"][f"k{k}|{a}"] = {"per_seed": gs, "flag": flag}   # (A5.1: the status line follows the loop)
        print(f"  K{k} {a:12s} " + " | ".join(
            f"s{s} LM share {fmt(g['lm_share'], 4, 2)} c {fmt(g['clip'], 4, 2)} prior/LM {fmt(g['prior_over_lm'], 5, 2)}"
            f" (>1 on {g['frac_prior_gt_lm']:.2f})" for s, g in gs.items() if g) + ("  FLAG: prior > LM" if flag else ""))


def diverged_at(r):
    """A6: step of the first eval with a non-finite lm, val_loss, geo, grad_prior or geo_<d> (None if none): the trainer's /
    CHECK_PY's / readout_rules' divergence onset (+ the domain priors' values)."""
    fin = lambda v: v is None or math.isfinite(v)
    for e in r["curve"]:
        if not all(fin(e.get(k)) for k in ("lm", "val_loss", "geo", "grad_prior") + tuple(f"geo_{d}" for d in EXTRA)):
            return e["step"]
    return None


def final_bpb(r):
    """The last eval's in-training val bits per byte (readout_rules.best_lr's fallback tie-break)."""
    return r["curve"][-1]["val_bits_per_byte"] if r["curve"] else None


def a6_readout(runs, out, partial, ident6=None, warn6=(), a6_partial=False):
    """A6 read-out (pre-registered 2026-10-19 before the A6 runs; amendment A6 of report/audits/rb6_A5_main_design.md),
    printed after the A5 sections when an A6 arm is given. Uses the A6 arms (all6-lam1 at lr 4e-3 / 8e-3, none at lr
    3e-3 / 1.6e-2) with the A5 cells none, none-lr8e-3 and all6-lam1, seeds 0-3 only. (1) all6's best lr over {1e-3, 4e-3, 8e-3} = readout_rules.best_lr on the
    numbers area (ties within 2 -> lower offline bpb), bracket = readout_rules.bracket; (2) F1c = all6@best - none@8e-3 on
    F1b's four domains with F1b's tests / BH / s4 side, next to F1 / F1b, saying whether it changes a domain's reading (the
    headline rule stays F1 & F1b); (3) F9 = all6@8e-3 - none@8e-3 on the six domains + offline bpb; (4) gradient split and LM
    cost (F8c: offline bpb all6@lr - none@1e-3, flag > +4e-3) of the new prior arms; (5) none's best lr PER DOMAIN over
    {1e-3, 3e-3, 8e-3, 1.6e-2} (best_lr on that domain's read-out, bracket), the s4 / dim 'at the floor at every lr' claim
    (every none lr FLOOR at the seed level), and F1d = all6@best - none@best-on-that-domain (descriptive); (6) the
    winner's-curse note on numbers rows with none@8e-3. A diverged run is data: its values over ALL evals count (area, final3
    of the broken model), its offline bpb counts as the worst in a tie-break, a paired row on its bpb is NOT TESTED (the
    diverged side named; F8c raises the LM COST FLAG when the prior arm diverged), its gradient split is that of its on-evals
    before the divergence; it is listed, never excluded. Seeds 0-3 of
    every arm used are required: a missing one makes every verdict that needs it PROVISIONAL and the read-out refuses
    unless --allow_partial / --a6_partial.
    A6.1 (2026-10-19, before any A6 result; the fresh audit of A6): (5)'s claim is three-way by the seed-level gate of every
    none lr - HOLDS (every lr FLOOR), DOES NOT HOLD (an lr whose seed CI lies above the floor: MEASURABLE / CEILING, or
    BORDERLINE only by crossing the ceiling; each such lr named with its level), NOT ESTABLISHED (none above the floor, an
    lr BORDERLINE across the floor; named) - and PROVISIONAL (n=k of K) with the current per-lr status when a none run is
    missing; dim also prints none@8e-3's confirm seeds 4-7 gate (descriptive); every BH family line prints its m; a
    diverged run's gradient split is labelled 'pre-onset evals only (k of K)' and raises no FLAG from < half the on-evals;
    an unfinished A6 run counts as missing; strict mode refuses BEFORE any conclusion (header, identity, missing seeds,
    refusal); --a6_partial relaxes only the A6 section (the A5 section stays strict and byte-identical)."""
    import contextlib
    import io
    import readout_rules as RR   # the one best_lr / bracket definition (pt2main_AUDIT rule 1, readout_rules.TIE = 2)
    K = MAIN_K
    strict = not (partial or a6_partial)
    ALL6_LR = {"1e-3": ALL6M, "4e-3": "all6-lam1-lr4e-3", "8e-3": "all6-lam1-lr8e-3"}
    NONE_LR = {"1e-3": "none", "3e-3": "none-lr3e-3", "8e-3": "none-lr8e-3", "1.6e-2": "none-lr1.6e-2"}
    use = set(ALL6_LR.values()) | set(NONE_LR.values())
    runs = [x for x in runs if x["k"] == K and x["arm"] in use]
    # A6.1: an unfinished run (only reachable with --allow_partial / --a6_partial) counts as missing (A5.1 rule 1)
    unfin = [x for x in runs if x["r"]["curve"][-1]["step"] != x["r"]["steps"] - 1]
    runs = [x for x in runs if x not in unfin]
    xs = {(x["arm"], x["seed"]): x for x in runs}
    cells = defaultdict(dict)
    for x in runs:
        cells[x["arm"]][x["seed"]] = x["r"]
    A = out["a6"] = {"excluded": [], "missing_seeds": {}, "diverged": {}, "all6_best": None, "families": {}, "none_best": {},
                     "claims": {}, "claim_status": {}, "grad": {}, "unfinished": [], "provisional": False,
                     "mode": "--allow_partial" if partial else "--a6_partial" if a6_partial else "strict"}
    print("\n== A6 READ-OUT (pre-registered 2026-10-19 BEFORE the A6 runs, amendment A6 of report/audits/rb6_A5_main_design.md):"
          " each arm at its own best lr (all6-lam1 at 1e-3 / 4e-3 / 8e-3; none at 1e-3 / 3e-3 / 8e-3 / 1.6e-2); K = 2, seeds 0-3; the A5 rules and"
          " headline are unchanged; these hypotheses came from the main read-out and are tested only on the new runs")
    print("  mode: " + {"strict": "strict (an incomplete A6 batch is refused before any A6 conclusion is printed)",
                        "--a6_partial": "--a6_partial (the A6 section may be incomplete: every verdict that needs a missing run is"
                                        " PROVISIONAL; the A5 section above is the strict one)",
                        "--allow_partial": "--allow_partial (A6 and A5 may be incomplete; A6 verdicts that need a missing run are"
                                           " PROVISIONAL)"}[A["mode"]])
    if partial:   # A6.1: what --allow_partial changed in the A5 section above (A5.1: any --allow_partial read-out is PROVISIONAL)
        print("  NOTE --allow_partial: by A5.1 the A5 section above is marked PROVISIONAL whatever its seeds: with every A5 seed"
              " present it differs from the strict output in exactly 2 lines - the added line '  PROVISIONAL: unfinished runs or"
              " an incomplete batch (--allow_partial)' under the A5 MAIN-DESIGN header, and the A5 status line"
              " 'PROVISIONAL (--allow_partial)' instead of 'complete (...)'; no A5 number changes. --a6_partial keeps the A5"
              " section strict (byte-identical) and relaxes only the A6 section")
    problems = []
    if ident6 is not None:
        print(f"  run identity of the A6 runs (tag's intent = run JSON; check.json PASS): {ident6['checked']} checked,"
              f" {len(ident6['excluded'])} EXCLUDED")
        for lab, why in ident6["excluded"]:
            print(f"  EXCLUDED run {lab}: " + "; ".join(why))
            A["excluded"].append([lab, why])
        for t in ident6["valid_notes"] + ident6["not_checked"]:
            print("  NOTE " + t)
    for w in warn6:
        print("  WARNING " + w)
    if unfin:
        A["unfinished"] = sorted(f"{x['arm']} s{x['seed']}" for x in unfin)
        print(f"  UNFINISHED run(s) counted as missing (A5.1 rule 1): {A['unfinished']}")
    missing = {a: sorted(set(DECIDE) - set(cells.get(a, {}))) for a in sorted(use)}
    missing = {a: v for a, v in missing.items() if v}
    A["missing_seeds"] = missing
    if missing:
        msg = "A6 seed set incomplete (seeds 0-3 of every arm used): missing " + "; ".join(f"{a} {v}" for a, v in missing.items())
        print(f"  PROVISIONAL: {msg}")
        problems.append(msg)
    div = {f"{a} s{s}": diverged_at(r) for a, sd in cells.items() for s, r in sd.items() if diverged_at(r) is not None}
    A["diverged"] = div
    print("  diverged runs (data, never excluded: area / final3 over ALL evals of the run as trained, offline bpb = the worst"
          f" in a tie-break): {div if div else 'none'}")
    prov = lambda arms: any(a in missing for a in arms)
    # A6.1: the conclusions below are computed into a buffer; strict mode with a problem prints only the refusal
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _a6_body(A, out, cells, xs, div, missing, problems, prov, RR, K)
    A["provisional"] = bool(missing or problems or partial or a6_partial)
    if problems and strict:
        print(f"  {len(problems)} A6 problem(s): {problems}")
        print("  A6 read-out REFUSED (strict): no A6 conclusion is printed for an incomplete batch; --a6_partial (the A5"
              " section stays strict) or --allow_partial to look anyway, every verdict that needs a missing run PROVISIONAL")
    assert not (problems and strict), ("A6 read-out: " + "; ".join(problems) + " (--a6_partial or --allow_partial to look"
                                       " at an incomplete A6 batch anyway: verdicts PROVISIONAL)")
    print(buf.getvalue(), end="")
    if problems:
        print(f"  {len(problems)} A6 problem(s): {problems}")
    print(f"\n== A6 read-out status: {'PROVISIONAL (' + ('missing seeds ' + str(missing) if missing else 'problems, --allow_partial or --a6_partial') + ')' if A['provisional'] else 'complete (seeds 0-3 of every arm used)'}")


def _a6_body(A, out, cells, xs, div, missing, problems, prov, RR, K):
    """a6_readout's conclusions (1)-(5) (A6.1: printed by a6_readout only when the read-out may proceed)."""
    ALL6_LR = {"1e-3": ALL6M, "4e-3": "all6-lam1-lr4e-3", "8e-3": "all6-lam1-lr8e-3"}
    NONE_LR = {"1e-3": "none", "3e-3": "none-lr3e-3", "8e-3": "none-lr8e-3", "1.6e-2": "none-lr1.6e-2"}

    def seeds_of(a):
        return [s for s in sorted(cells.get(a, {})) if s in DECIDE]

    def lr_cells(arms, dom):
        """readout_rules.best_lr input: {lr: {area: [metric per seed 0-3], off: [offline bpb], intr: [in-training bpb]}}"""
        cl = {}
        for lr, a in arms.items():
            ss = seeds_of(a)
            if not ss:
                continue
            off = [offline_bpb(xs[(a, s)]["path"], cells[a][s]) for s in ss]
            cl[lr] = {"area": [mval(xs[(a, s)], dom) for s in ss], "off": off, "intr": [final_bpb(cells[a][s]) for s in ss],
                      "seeds": ss}
        return cl

    # ---- (1) all6's best lr (ONE criterion, fixed before the runs: numbers area, ties within 2 -> offline bpb)
    cl = lr_cells(ALL6_LR, "numbers")
    best = None
    if len(cl) >= 1:
        sel = RR.best_lr(cl)
        lrs = sorted(cl, key=float)
        br = RR.bracket(lrs, sel["mean"], sel["best"])
        best = sel["best"]
        A["all6_best"] = {"best": best, "arm": ALL6_LR[best], "near": sel["near"], "mean_area": sel["mean"], "bpb": sel["bpb"],
                          "offline": sel["offline"], "bracket": br, "provisional": prov(ALL6_LR.values()) or len(cl) < 3}
        print(f"\n== A6 (1) all6's best lr over {list(ALL6_LR)} (readout_rules.best_lr: the highest mean numbers area over seeds 0-3;"
              f" lrs within {RR.TIE:g} of the top tied -> the lower mean {'offline' if sel['offline'] else 'in-training'} bpb, a"
              " non-finite one the worst; the same rule that picked none@8e-3 in pt2b; no per-domain selection)")
        for lr in lrs:
            print(f"  all6@{lr:7s} ({ALL6_LR[lr]:17s}) numbers area {fmt(sel['mean'][lr])} [{' '.join(fmt(v) for v in cl[lr]['area'])}]"
                  f" seeds {cl[lr]['seeds']}" + (f" bpb {sel['bpb'][lr]:.4f}" if lr in sel["bpb"] else "")
                  + (" (tied)" if lr in sel["near"] and len(sel["near"]) > 1 else ""))
        print(f"  -> all6's best lr = {best}" + (f"; bracket {br[0].upper()} at the {br[1]} edge" if br else "; bracket closed")
              + ("  PROVISIONAL (seeds or lrs missing)" if A["all6_best"]["provisional"] else ""))
    # ---- (2)-(5) families
    gate8 = out.get("main", {}).get("gate", {}).get(f"k{K}|none-lr8e-3|s4_held_fresh_nonid", {}).get("verdict", "")
    if not gate8 and seeds_of("none-lr8e-3"):
        gate8 = gate_key([cells["none-lr8e-3"][s] for s in seeds_of("none-lr8e-3")], "s4", READ["s4"][0])["verdict"]
    s4side = "greater" if gate8.endswith("FLOOR") else "two"
    D4 = ("numbers", "clock", "s4", "dim")

    def test(a, b, dom, side, label):
        need = list(DECIDE)
        ss = [s for s in need if s in cells.get(a, {}) and s in cells.get(b, {})]
        va = {s: mval(xs[(a, s)], dom) for s in ss}
        vb = {s: mval(xs[(b, s)], dom) for s in ss}
        miss = [s for s in ss if va[s] is None or vb[s] is None]
        if miss:   # a value that is not there (no offline_lm record): a problem, as A5's DROPPED seeds
            problems.append(f"{label}: {a} - {b} {dom}: DROPPED seeds {miss} (no value: offline_lm.json missing?)")
        ss = [s for s in ss if s not in miss]
        nonfin = [s for s in ss if not (math.isfinite(va[s]) and math.isfinite(vb[s]))]   # a diverged run's bpb: data
        if nonfin or not ss:
            dd = [va[s] - vb[s] for s in ss]
            return {"n": len(ss), "diffs": dd, "mean": float(np.mean(dd)) if dd else None, "wins": sum(v > 0 for v in dd),
                    "losses": sum(v < 0 for v in dd), "sd": None, "t": None, "p_t": None, "p_sf": None, "ci": None,
                    "metric": dom, "a": a, "b": b, "side": side, "seeds": ss, "n_required": len(need),
                    "provisional": len(ss) < len(need), "diverged_seeds": nonfin,
                    "diverged_a": [s for s in nonfin if not math.isfinite(va[s])],
                    "diverged_b": [s for s in nonfin if not math.isfinite(vb[s])],
                    "a_mean": float(np.mean([va[s] for s in ss])) if ss else None,
                    "b_mean": float(np.mean([vb[s] for s in ss])) if ss else None}
        st = paired([va[s] - vb[s] for s in ss], side)
        return dict(st, metric=dom, a=a, b=b, side=side, seeds=ss, n_required=len(need), provisional=len(ss) < len(need),
                    diverged_seeds=[], diverged_a=[], diverged_b=[], a_mean=float(np.mean([va[s] for s in ss])),
                    b_mean=float(np.mean([vb[s] for s in ss])))

    def with_bh(rows):
        for row, q in zip(rows, bh([x["p_t"] for x in rows])):
            row["p_bh"] = q
            row["significant"] = q is not None and q < Q_BH and not row["provisional"]
            row["bh_m"] = sum(x["p_t"] is not None for x in rows)
        return rows

    def bh_m(rows):
        """A6.1: BH's m of a family (rows without a p - NOT TESTED / n < 2 - are not counted: the others' BH is milder)"""
        m = sum(x["p_t"] is not None for x in rows)
        return f"BH m = {m} of {len(rows)} rows" + (f", {len(rows) - m} without a p (NOT TESTED / n < 2): the others' BH milder"
                                                  if m < len(rows) else "")

    def lab_of(dom):
        return {"bpb": "offline bpb x1e3"}.get(dom, f"{dom} {READ[dom][0]}" if dom in READ else dom)

    def row_txt(x, pre, decide=True):
        ci = x.get("ci")
        ci_t = "" if ci is None else (f" 95% CI [{fmt(ci[0], 6)},{fmt(ci[1], 6)}]" if x["side"] == "two" else f" 95% LB {fmt(ci[0], 6)}")
        dv = " and ".join(f"{x[k]} at seeds {x['diverged_' + k]}" for k in ("a", "b") if x.get("diverged_" + k))
        v = ("PROVISIONAL (n=" + str(x["n"]) + " of " + str(x["n_required"]) + "): " if x["provisional"] else "") + (
            (f"NOT TESTED: DIVERGED {dv} (a non-finite value = the worst: that run's LM is broken; no t test)"
             if x.get("diverged_seeds") else "SIGNIFICANT" if x["significant"] else "n.s.") if decide else
            ("descriptive" + (f" [DIVERGED {dv}]" if x.get("diverged_seeds") else "")))
        note = f"  [winner's curse: {WINNER}]" if x["metric"] == "numbers" and "none-lr8e-3" in (x["a"], x["b"]) else ""
        return (f"{pre}{x['a']:17s} - {x['b']:13s} {fmt(x['a_mean'], 6)} vs {fmt(x['b_mean'], 6)} Δ {fmt(x['mean'], 6)} "
                f"[{' '.join(fmt(d, 6) for d in x['diffs'])}] n {x['n']} {'1-sided' if x['side'] != 'two' else '2-sided'} p "
                f"{fmt(x.get('p_t'), 6, 4)} BH {fmt(x.get('p_bh'), 6, 4)} sign-flip {fmt(x.get('p_sf'), 6, 4)} wins {x['wins']}/{x['n']}"
                f"{ci_t} -> {v}{note}")

    fams = out.get("main", {}).get("families", {})
    old = {fid: {x["metric"]: x for x in fams.get(fid, {}).get("tests", [])} for fid in ("F1", "F1b")}
    if best is not None and "none-lr8e-3" in cells:
        a_best = ALL6_LR[best]
        rows = with_bh([test(a_best, "none-lr8e-3", d, s4side if d == "s4" else "two", "F1c") for d in D4])
        A["families"]["F1c"] = rows
        print(f"\n== A6 (2) F1c = all6@best ({best}: {a_best}) - none@8e-3 on F1b's four domains, seeds 0-3 (F1b's tests, BH within"
              f" F1c: {bh_m(rows)}; s4 {'one-sided' if s4side == 'greater' else 'two-sided'} as F1b by none@8e-3's seed-level s4 gate); F1 and F1b"
              " (A5, unchanged) beside; the HEADLINE rule stays F1 & F1b"
              + ("" if best == "1e-3" else f"; all6's lr {best} was chosen on numbers on these same seeds 0-3: F1c numbers also"
                 " carries a selection effect in all6's favour (besides the winner's curse of none@8e-3)"))
        for x in rows:
            d = x["metric"]
            print(row_txt(x, f"    {lab_of(d):30s} F1c "))
            for fid in ("F1", "F1b"):
                y = old[fid].get(d)
                print(f"    {'':30s} {fid:3s} " + (f"{y['a']} - {y['b']} Δ {fmt(y['mean'], 6)} n {y['n']} -> "
                      f"{('PROVISIONAL ' if y.get('provisional') else '')}{'SIGNIFICANT' if y['significant'] else 'n.s.'}" if y else "not computed"))
            y = old["F1b"].get(d)
            if best == "1e-3":
                ch = "no change: all6's best lr is 1e-3, so F1c = F1b by construction"
            elif y is None or x["provisional"] or y.get("provisional"):
                ch = "PROVISIONAL: cannot compare (F1b or F1c incomplete)"
            else:
                rd = lambda z: ("gain" if z["mean"] > 0 else "harm") if z["significant"] else "n.s."
                ch = (f"no change (F1b {rd(y)}, F1c {rd(x)})" if rd(x) == rd(y) else
                      f"CHANGES the best-vs-best reading: F1b {rd(y)} -> F1c {rd(x)} (the A5 headline, F1 & F1b, is not changed by it)")
            x["reading"] = ch
            print(f"    {'':30s} reading: {ch}")
    if "all6-lam1-lr8e-3" in cells and "none-lr8e-3" in cells:
        rows = with_bh([test("all6-lam1-lr8e-3", "none-lr8e-3", d, s4side if d == "s4" else "two", "F9") for d in DOM6 + ("bpb",)])
        A["families"]["F9"] = rows
        print("\n== A6 (3) F9 = all6@8e-3 - none@8e-3, the same lr (clock is the question), the six domains + offline bpb, seeds"
              f" 0-3, BH within F9 ({bh_m(rows)}; s4 side as F1b)")
        for x in rows:
            print(row_txt(x, f"    {lab_of(x['metric']):30s} " + ("<- the question " if x["metric"] == "clock" else "")))
    rows = with_bh([test(ALL6_LR[lr], "none", "bpb", "two", "F8c") for lr in ("4e-3", "8e-3")
                    if ALL6_LR[lr] in cells and "none" in cells])
    if rows:
        A["families"]["F8c"] = rows
        print(f"\n== A6 (4) LM cost of the new prior arms (F8c: offline bpb x1e3, all6@lr - none@1e-3 as F8, seeds 0-3, BH within"
              f" F8c: {bh_m(rows)}; LM COST FLAG"
              f" when the mean > +{1000 * LM_FLAG:g} whatever p says) and the gradient split (window medians before the global clip;"
              " FLAG = the prior gradient exceeds the LM gradient); all6@1e-3 beside for reference")
        for x in rows:
            fl = bool(x.get("diverged_a")) or (x["mean"] is not None and math.isfinite(x["mean"]) and x["mean"] > 1000 * LM_FLAG)
            x["lm_flag"] = fl
            print(row_txt(x, f"    {lab_of('bpb'):30s} ") + ((f"  LM COST FLAG (the prior arm DIVERGED at seeds {x['diverged_a']})"
                  if x.get("diverged_a") else f"  LM COST FLAG (mean > +{LM_FLAG:g} bpb, A1 R5)") if fl else ""))
    for a in (ALL6M, "all6-lam1-lr4e-3", "all6-lam1-lr8e-3"):
        sd = cells.get(a, {})
        gs = {s: grad_split(sd[s]) for s in seeds_of(a)}
        if not any(gs.values()):
            continue
        # A6.1: a diverged run's split uses its on-evals before the onset only (k of K, K = every on-eval with both
        # gradients logged); from fewer than half of them it raises no FLAG (the early window has a larger prior / LM ratio)
        n_on = {s: sum(1 for e in sd[s]["curve"] if e.get("prior_on") and e.get("grad_prior") is not None and e.get("grad_lm"))
                for s in gs}
        short = sorted(s for s, g in gs.items() if f"{a} s{s}" in div and (g is None or g["n"] < 0.5 * n_on[s]))
        flag = any(g and g["prior_over_lm"] > 1 for s, g in gs.items() if s not in short)
        quiet = [s for s in short if gs[s] and gs[s]["prior_over_lm"] > 1]
        A["grad"][a] = {"per_seed": gs, "flag": flag, "n_on_evals": n_on, "pre_onset_lt_half": short}

        def seg(s, g):
            dv = f"{a} s{s}" in div
            pre = (f" [DIVERGED at step {div[f'{a} s{s}']}: pre-onset evals only ({g['n'] if g else 0} of {n_on[s]})"
                   + ("; fewer than half: no FLAG from this seed" if s in short else "") + "]") if dv else ""
            if not g:
                return f"s{s} no on-eval before the divergence{pre}"
            return (f"s{s} LM share {fmt(g['lm_share'], 4, 2)} c {fmt(g['clip'], 4, 2)} prior/LM {fmt(g['prior_over_lm'], 5, 2)}"
                    f" (>1 on {g['frac_prior_gt_lm']:.2f}){pre}")
        print(f"  {a:17s} " + " | ".join(seg(s, g) for s, g in gs.items() if g or f"{a} s{s}" in div)
              + ("  FLAG: prior > LM" if flag else "")
              + (f"  (prior/LM > 1 at s{quiet} only from < half the on-evals: not a FLAG)" if quiet else "")
              + ("  (reference, A5)" if a == ALL6M else ""))
    # ---- (5) none's best lr per domain, the s4 / dim floor claim, F1d
    print(f"\n== A6 (5) none's best lr PER DOMAIN over {list(NONE_LR)} (readout_rules.best_lr on that domain's read-out, seeds"
          " 0-3; ties within 2 -> lower bpb; bracket OPEN = the best at an edge by > 2), the seed-level gate of every none lr, and"
          " F1d = all6@best - none@best-on-that-domain (DESCRIPTIVE: a per-domain choice for none only, conservative for the prior)")
    f1d = []
    for d in DOM6:
        cl = lr_cells(NONE_LR, d)
        if not cl:
            continue
        sel = RR.best_lr(cl)
        lrs = sorted(cl, key=float)
        br = RR.bracket(lrs, sel["mean"], sel["best"])
        gates = {}
        if d != "numbers":
            for lr in lrs:
                a = NONE_LR[lr]
                gates[lr] = gate_key([cells[a][s] for s in cl[lr]["seeds"]], d, READ[d][0])
        nb = {"best": sel["best"], "arm": NONE_LR[sel["best"]], "mean": sel["mean"], "near": sel["near"], "bracket": br,
              "verdicts": {lr: g["verdict"] for lr, g in gates.items()},
              "provisional": prov(NONE_LR.values()) or len(cl) < len(NONE_LR)}
        A["none_best"][d] = nb
        print(f"  {d:9s} {READ[d][0]:22s} " + " | ".join(
            f"{lr} {fmt(sel['mean'][lr])}" + (f" {gates[lr]['verdict']}" if lr in gates else "") for lr in lrs)
            + f" -> none's best {sel['best']}" + (f"; bracket {br[0].upper()} at the {br[1]} edge" if br else "; bracket closed")
            + ("  PROVISIONAL" if nb["provisional"] else ""))
        if d in ("s4", "dim"):
            # A6.1: three-way by the seed-level gate of EVERY none lr (A6: 'holds only if every lr is FLOOR; otherwise the
            # level is stated'): an lr whose seed CI lies above the floor (lo >= floor: MEASURABLE, CEILING, or BORDERLINE
            # only by crossing the ceiling) breaks it -> DOES NOT HOLD; else an lr BORDERLINE across the floor -> NOT
            # ESTABLISHED (not shown to be at the floor, not shown to be learned); else HOLDS. A missing none run ->
            # PROVISIONAL (n=k of K) with the current per-lr status, no direction. The named lrs are the ones that break FLOOR.
            def lvl(lr):
                g = gates[lr]
                lo, hi = g["seed_ci"]
                return (f"{lr} {fmt(g['final3']).strip()} [{fmt(lo).strip()}, {fmt(hi).strip()}] vs floor {fmt(g['floor']).strip()}"
                        f" -> {g['verdict']}" + (f" (n {len(cl[lr]['seeds'])} of {len(DECIDE)})" if len(cl[lr]["seeds"]) < len(DECIDE)
                                                 else ""))
            above = lambda g: g["seed_ci"][0] is not None and g["seed_ci"][0] >= g["floor"]
            broke = [lr for lr in lrs if above(gates[lr])]
            border = [lr for lr in lrs if gates[lr]["verdict"] == "BORDERLINE" and lr not in broke]
            undecided = [lr for lr in lrs if gates[lr]["verdict"] not in ("FLOOR", "BORDERLINE", "MEASURABLE", "CEILING")]
            stem = f"'{d} not learnable without the prior'"
            hs = sorted({fmt(g["heuristic"]).strip() for g in gates.values() if g.get("heuristic") is not None})
            if nb["provisional"] or undecided:
                status = "PROVISIONAL"
                nrun = sum(len(cl[lr]["seeds"]) for lr in lrs)
                miss_t = "; ".join(f"{a} {missing[a]}" for a in NONE_LR.values() if a in missing) or "none"
                claim = (f"{stem}: PROVISIONAL (n={nrun} of {len(NONE_LR) * len(DECIDE)} none runs; missing {miss_t}): not decided;"
                         " so far " + ("every available lr FLOOR: " if all(gates[lr]["verdict"] == "FLOOR" for lr in lrs) else "")
                         + "; ".join(lvl(lr) for lr in lrs))
            elif broke:
                status = "DOES NOT HOLD"
                claim = (f"{stem} DOES NOT HOLD: none is above the floor at " + "; ".join(lvl(lr) for lr in broke)
                         + ("; also BORDERLINE across the floor: " + "; ".join(lvl(lr) for lr in border) if border else ""))
            elif border:
                status = "NOT ESTABLISHED"
                claim = (f"{stem} NOT ESTABLISHED: no none lr is above the floor, but BORDERLINE across the floor at "
                         + "; ".join(lvl(lr) for lr in border) + " (the seed CI spans the floor: not shown to be at the floor,"
                         " not shown to be learned); every other lr FLOOR")
            else:
                status = "HOLDS"
                claim = f"{stem} HOLDS: none is at FLOOR (seed level) at every lr {lrs}"
            if hs and status != "HOLDS":
                claim += f" (heuristic level {', '.join(hs)})"
            if br and br[0] == "open" and status != "HOLDS":
                claim += f"; none's bracket is OPEN at the {br[1]} edge: its best on {d} may lie beyond the grid"
            A["claims"][d] = claim
            A["claim_status"][d] = status
            print(f"    {claim}")
            if d == "dim":   # A6.1: the seeds-0-3 MEASURABLE of none@8e-3 (A5) beside its confirm seeds 4-7 (descriptive)
                g47 = out.get("main", {}).get("gate", {}).get(f"k{K}|none-lr8e-3[4-7]|{READ['dim'][0]}")
                src = "A5 gate"
                if g47 is None:
                    s47 = [s for s in CONFIRM if s in cells.get("none-lr8e-3", {})]
                    g47 = gate_key([cells["none-lr8e-3"][s] for s in s47], "dim", READ["dim"][0]) if len(s47) >= 2 else None
                    src = f"seeds {s47}"
                g03 = gates.get("8e-3")
                if g47 is not None and g03 is not None:
                    same = g47["verdict"] == g03["verdict"]
                    A["dim_none8e-3_seeds4-7"] = {"final3": g47["final3"], "seed_ci": g47["seed_ci"], "verdict": g47["verdict"],
                                                  "floor": g47["floor"], "seeds_0_3_verdict": g03["verdict"], "replicates": same}
                    print(f"    descriptive (not part of the claim): none@8e-3's dim on its confirm seeds 4-7 ({src}) "
                          f"{fmt(g47['final3']).strip()} [{fmt(g47['seed_ci'][0]).strip()}, {fmt(g47['seed_ci'][1]).strip()}] vs floor"
                          f" {fmt(g47['floor']).strip()} -> {g47['verdict']}; seeds 0-3 {fmt(g03['final3']).strip()} -> {g03['verdict']}:"
                          + (" the seeds 0-3 verdict replicates on 4-7" if same else
                             " the seeds 0-3 verdict does NOT replicate on 4-7 (8e-3 was chosen on pt2b seeds 0-3, which share"
                             " the init and data with these seeds 0-3)"))
                else:
                    print("    descriptive: none@8e-3's dim on its confirm seeds 4-7 not available")
        if best is not None:
            x = test(ALL6_LR[best], NONE_LR[sel["best"]], d, "two", "F1d")
            x["significant"] = False
            x["p_bh"] = None
            disc = d in TIER_E and (ALL6_LR[best] == ALL6M or NONE_LR[sel["best"]] == "none")
            f1d.append(dict(x, discovery_seeds=disc))
            print(row_txt(x, f"    {'F1d':4s} ", decide=False)
                  + ("  [tier E incl. the DISCOVERY seeds 0, 1 of none / all6-lam1: exploratory]" if disc else ""))
    A["families"]["F1d"] = f1d


def is_a7(x):
    """A7: a run of the A7 block (none@3.2e-2 / all6@5e-4 at seeds 0-3, none@1.6e-2 at the fresh seeds 4-7): read by the A7
    section only; the A5 / A6 sections are computed without them (byte-identical)."""
    return x["arm"] in A7_ARMS or (x["arm"] == "none-lr1.6e-2" and x["seed"] in CONFIRM)


def s4_latin_level(r):
    """A7: the full Latin-square propagation level of this run's s4 read-out set (READ['s4'], held_fresh_nonid), in points:
    the trained cells' answers as givens; naked singles and hidden singles (per row / column) to a fixpoint; a derived
    cell scores 1, any other held cell 1 / (its remaining candidates) if its answer is among them. Only the Latin-square
    property, no group law (the F88 result audit's computation, notes/batch1_build/audit_rb6m_results.md: 32.5 over
    seeds 0-3, 31.5 over 0-7). A seed's level is the same for every arm (same split)."""
    ck = (id(r), "latin")
    if ck in _CACHE and _CACHE[ck][0] is r:
        return _CACHE[ck][1]
    sp = split_of(r, "s4")
    T = toy_tables()["s4"]
    X, Y = T["X"], T["Y"]
    n = 24
    K = {(int(X[j, 0]), int(X[j, 1])): int(Y[j]) for j in sp["train"]}
    held = [(int(X[q, 0]), int(X[q, 1])) for q in sp["held"]]
    ans = {(int(X[q, 0]), int(X[q, 1])): int(Y[q]) for q in sp["held"]}
    cand = {c: set(range(n)) - {v for (a, b), v in K.items() if a == c[0] or b == c[1]} for c in held}

    def place(c, v):
        K[c] = v
        cand[c] = {v}
        for c2 in cand:
            if c2 != c and (c2[0] == c[0] or c2[1] == c[1]):
                cand[c2].discard(v)
    changed = True
    while changed:
        changed = False
        for c, cs in list(cand.items()):   # naked singles
            if len(cs) == 1 and c not in K:
                place(c, next(iter(cs)))
                changed = True
        for line in range(n):   # hidden singles per row / column
            for axis in (0, 1):
                cells = [c for c in cand if c[axis] == line and c not in K]
                for v in range(n):
                    if any(K.get((line, j) if axis == 0 else (j, line)) == v for j in range(n)):
                        continue
                    where = [c for c in cells if v in cand[c]]
                    if len(where) == 1:
                        place(where[0], v)
                        changed = True
    mk = [ch == "1" for ch in r["domains"]["s4"]["eval_held"]["masks"][READ["s4"][0][len("s4_held_"):]]]
    acc = []
    for c, m in zip(held, mk):
        if not m:
            continue
        if c in K:
            assert K[c] == ans[c], ("s4 Latin propagation derived a wrong answer", c)
            acc.append(1.0)
        else:
            acc.append(1 / len(cand[c]) if ans[c] in cand[c] else 0.0)
    v = 100 * float(np.mean(acc))
    _CACHE[ck] = (r, v)
    return v


def first_step_ge(r, key, th):
    """A7: the first eval step at which the read-out key reaches th (a fraction); None = never (censored at the budget)."""
    return next((s for s, v in series(r, key) if v is not None and v >= th), None)


def fisher_greater(ka, na, kb, nb):
    """A7: one-sided Fisher exact p for 'group a has a larger success share than group b' (hypergeometric upper tail)."""
    tot, m = na + nb, ka + kb
    pmf = lambda x: math.comb(m, x) * math.comb(tot - m, na - x) / math.comb(tot, na)
    return float(min(1.0, sum(pmf(x) for x in range(ka, min(m, na) + 1))))


def barnard_greater(ka, na, kb, nb, grid=2001):
    """A7: one-sided Barnard exact p (pooled-variance Wald z, a larger for a; the nuisance share maximised over a grid on
    (0, 1)); the same statistic as scipy.stats.barnard_exact(pooled=True, alternative='greater')."""
    def z(xa, xb):
        pa, pb, p = xa / na, xb / nb, (xa + xb) / (na + nb)
        v = p * (1 - p) * (1 / na + 1 / nb)
        return 0.0 if v == 0 else (pa - pb) / math.sqrt(v)
    z0 = z(ka, kb)
    ext = [(xa, xb) for xa in range(na + 1) for xb in range(nb + 1) if z(xa, xb) >= z0 - 1e-12]
    best = 0.0
    for pi in np.linspace(1e-6, 1 - 1e-6, grid):
        pr = sum(math.comb(na, xa) * pi ** xa * (1 - pi) ** (na - xa) * math.comb(nb, xb) * pi ** xb * (1 - pi) ** (nb - xb)
                 for xa, xb in ext)
        best = max(best, pr)
    return float(min(1.0, best))


def gehan_perm(ta, tb):
    """A7: exact permutation Gehan test (censored times: None = never reached, censored at the budget, later than every
    event) for 'a reaches it earlier'. U = sum over (i in a, j in b) of +1 if a_i < b_j (a_i an event), -1 if b_j < a_i
    (b_j an event), 0 otherwise; p = share of the C(n, |a|) relabellings with U >= U_obs. -> (U, p)"""
    inf = float("inf")
    pool = [inf if t is None else float(t) for t in list(ta) + list(tb)]
    na = len(ta)

    def U(ia):
        ib = [k for k in range(len(pool)) if k not in ia]
        u = 0
        for i in ia:
            for j in ib:
                x, y = pool[i], pool[j]
                u += 1 if x < y else -1 if y < x else 0
        return u
    u0 = U(tuple(range(na)))
    perms = list(itertools.combinations(range(len(pool)), na))
    return u0, float(np.mean([U(c) >= u0 for c in perms]))


def sign_greater(pairs):
    """A7: exact one-sided sign test over paired seeds (a_t, b_t), None = never: 'a earlier' wins; ties (equal, incl. both
    never) dropped. -> (wins, losses, ties, p)"""
    inf = float("inf")
    w = l_ = t_ = 0
    for a, b in pairs:
        a, b = inf if a is None else a, inf if b is None else b
        w, l_, t_ = (w + 1, l_, t_) if a < b else (w, l_ + 1, t_) if b < a else (w, l_, t_ + 1)
    n = w + l_
    p = float(sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n) if n else 1.0
    return w, l_, t_, p


def km_median(ts):
    """A7: the time by which half the seeds reached it (the ceil(n/2)-th smallest; None = not reached by then)."""
    s = sorted(float("inf") if t is None else t for t in ts)
    if not s:
        return None
    v = s[(len(s) + 1) // 2 - 1]
    return None if v == float("inf") else v


def a7_readout(runs, out, partial, ident7=None, warn7=(), a7_partial=False):
    """A7 read-out (pre-registered 2026-10-19 before the A7 runs; amendment A7 of report/audits/rb6_A5_main_design.md),
    printed after the A6 section when an A7 run is given. Hypothesis (formed on the A6 read-out; tested only with the new
    runs): 'the prior makes S4 learned reliably and early; without it S4 is learned only at a high lr, in some seeds and
    late'. (1) reliability: seeds with s4 read-out final3 >= that seed's Latin-propagation level (s4_latin_level), all6@1e-3
    vs none@1.6e-2 at the fresh seeds 4-7 (primary) and vs none@3.2e-2 at seeds 0-3 (same seeds): one-sided Fisher exact
    (decision, BH within the family), Barnard exact and the paired seed-level t on final3 beside; (2) speed: the first
    eval step with s4 read-out >= 25% (censored at the budget), the same pairs: exact permutation Gehan (decision, BH),
    sign test over paired seeds, medians; (3) best vs best: all6@best over {5e-4, 1e-3, 4e-3, 8e-3} (numbers area, A6's
    rule) - none@best over {1e-3, 3e-3, 8e-3, 1.6e-2, 3.2e-2} by the s4 AREA, on s4 area and final3 at seeds 0-3 and, where
    both cells have them, 4-7 (two-sided, BH within the family); (4) none's s4 bracket over the five lrs (by area and by
    final3), divergence as A6; (5) all6's best lr over the four lrs and F1c updated. A diverged run is data (A6). A missing
    / excluded / unfinished run -> PROVISIONAL; strict refuses before any conclusion (A6.1)."""
    import contextlib
    import io
    import readout_rules as RR
    K = MAIN_K
    strict = not (partial or a7_partial)
    ALL6_LR = {"5e-4": "all6-lam1-lr5e-4", "1e-3": ALL6M, "4e-3": "all6-lam1-lr4e-3", "8e-3": "all6-lam1-lr8e-3"}
    NONE_LR = {"1e-3": "none", "3e-3": "none-lr3e-3", "8e-3": "none-lr8e-3", "1.6e-2": "none-lr1.6e-2", "3.2e-2": "none-lr3.2e-2"}
    use = set(ALL6_LR.values()) | set(NONE_LR.values())
    need = {a: set(DECIDE) for a in use}
    need[ALL6M] |= set(CONFIRM)
    need["none-lr1.6e-2"] |= set(CONFIRM)
    runs = [x for x in runs if x["k"] == K and x["arm"] in use and x["seed"] in need[x["arm"]]]
    unfin = [x for x in runs if x["r"]["curve"][-1]["step"] != x["r"]["steps"] - 1]
    runs = [x for x in runs if x not in unfin]
    xs = {(x["arm"], x["seed"]): x for x in runs}
    cells = defaultdict(dict)
    for x in runs:
        cells[x["arm"]][x["seed"]] = x["r"]
    A = out["a7"] = {"excluded": [], "missing_seeds": {}, "diverged": {}, "unfinished": [], "levels": {}, "families": {},
                     "all6_best": None, "none_s4_best": None, "provisional": False,
                     "mode": "--allow_partial" if partial else "--a7_partial" if a7_partial else "strict"}
    print("\n== A7 READ-OUT (pre-registered 2026-10-19 BEFORE the A7 runs, amendment A7 of report/audits/rb6_A5_main_design.md):"
          " 'the prior makes S4 learned reliably and early; without it S4 is learned only at a high lr, in some seeds and late'"
          " - formed on the A6 read-out, tested with the new runs only (none@1.6e-2 seeds 4-7, none@3.2e-2 and all6@5e-4"
          " seeds 0-3); the A5 / A6 rules and headlines are unchanged")
    print("  mode: " + {"strict": "strict (an incomplete A7 batch is refused before any A7 conclusion is printed)",
                        "--a7_partial": "--a7_partial (the A7 section may be incomplete: every verdict that needs a missing run is"
                                        " PROVISIONAL; the A5 / A6 sections above are the strict ones)",
                        "--allow_partial": "--allow_partial (A7 verdicts that need a missing run are PROVISIONAL)"}[A["mode"]])
    problems = []
    if ident7 is not None:
        print(f"  run identity of the A7 runs (tag's intent = run JSON; check.json PASS): {ident7['checked']} checked,"
              f" {len(ident7['excluded'])} EXCLUDED")
        for lab, why in ident7["excluded"]:
            print(f"  EXCLUDED run {lab}: " + "; ".join(why))
            A["excluded"].append([lab, why])
        for t in ident7["valid_notes"] + ident7["not_checked"]:
            print("  NOTE " + t)
    for w in warn7:
        print("  WARNING " + w)
    if unfin:
        A["unfinished"] = sorted(f"{x['arm']} s{x['seed']}" for x in unfin)
        print(f"  UNFINISHED run(s) counted as missing (A5.1 rule 1): {A['unfinished']}")
    missing = {a: sorted(need[a] - set(cells.get(a, {}))) for a in sorted(use)}
    missing = {a: v for a, v in missing.items() if v}
    A["missing_seeds"] = missing
    if missing:
        msg = "A7 seed set incomplete: missing " + "; ".join(f"{a} {v}" for a, v in missing.items())
        print(f"  PROVISIONAL: {msg}")
        problems.append(msg)
    div = {f"{a} s{s}": diverged_at(r) for a, sd in cells.items() for s, r in sd.items() if diverged_at(r) is not None}
    A["diverged"] = div
    print("  diverged runs (data, never excluded: accuracies over ALL evals count; offline bpb = the worst in a tie-break):"
          f" {div if div else 'none'}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _a7_body(A, out, cells, xs, missing, problems, RR, K, ALL6_LR, NONE_LR)
    A["provisional"] = bool(missing or problems or partial or a7_partial)
    if problems and strict:
        print(f"  {len(problems)} A7 problem(s): {problems}")
        print("  A7 read-out REFUSED (strict): no A7 conclusion is printed for an incomplete batch; --a7_partial (the A5 / A6"
              " sections stay strict) or --allow_partial to look anyway, every verdict that needs a missing run PROVISIONAL")
    assert not (problems and strict), ("A7 read-out: " + "; ".join(problems) + " (--a7_partial or --allow_partial to look"
                                       " at an incomplete A7 batch anyway: verdicts PROVISIONAL)")
    print(buf.getvalue(), end="")
    if problems:
        print(f"  {len(problems)} A7 problem(s): {problems}")
    print(f"\n== A7 read-out status: {'PROVISIONAL (' + ('missing seeds ' + str(missing) if missing else 'problems, --allow_partial or --a7_partial') + ')' if A['provisional'] else 'complete (every A7 seed present)'}")


def _a7_body(A, out, cells, xs, missing, problems, RR, K, ALL6_LR, NONE_LR):
    """a7_readout's conclusions (1)-(5), printed only when the read-out may proceed."""
    S4K = READ["s4"][0]
    have = lambda a, ss: [s for s in ss if s in cells.get(a, {})]
    f3 = lambda a, s: metrics(cells[a][s], S4K)["final3"]
    ar = lambda a, s: metrics(cells[a][s], S4K)["area"]
    # per-seed levels (the same split for every arm of a seed): Latin propagation (decision) and one-step elimination
    lev, el = {}, {}
    for s in sorted({s for sd in cells.values() for s in sd}):
        r = next(sd[s] for sd in cells.values() if s in sd)
        lev[s], el[s] = s4_latin_level(r), heuristic(r, "s4", S4K)
    A["levels"] = {"latin": lev, "elimination": el}
    print("\n== A7 s4 levels per seed (the read-out set's own; the same for every arm of a seed): full Latin-square propagation"
          " (decision threshold of (1)) | one-step elimination (the summary's heuristic, beside): "
          + " ".join(f"s{s} {lev[s]:.1f} | {el[s]:.1f}" for s in sorted(lev))
          + (f"; mean seeds 0-3 {np.mean([lev[s] for s in DECIDE if s in lev]):.1f}" if all(s in lev for s in DECIDE) else "")
          + (f", 0-7 {np.mean([lev[s] for s in DECIDE + CONFIRM if s in lev]):.1f}" if all(s in lev for s in DECIDE + CONFIRM) else ""))

    def bh_rows(rows, key="p", out_key="p_bh"):
        for row, q in zip(rows, bh([x[key] for x in rows])):
            row[out_key] = q
        return sum(x[key] is not None for x in rows)

    # ---- (1) reliability
    comps = [("R1", ALL6M, "none-lr1.6e-2", CONFIRM, "primary: the fresh seeds 4-7"),
             ("R2", ALL6M, "none-lr3.2e-2", DECIDE, "the second none, seeds 0-3 (same seeds)")]
    rows = []
    for lab, a, b, seeds, what in comps:
        sa, sb = have(a, seeds), have(b, seeds)
        ss = [s for s in seeds if s in sa and s in sb]
        va = {s: f3(a, s) for s in sa}
        vb = {s: f3(b, s) for s in sb}
        ka, kb = sum(va[s] >= lev[s] for s in sa), sum(vb[s] >= lev[s] for s in sb)
        prov = len(sa) < len(seeds) or len(sb) < len(seeds)
        st = paired([va[s] - vb[s] for s in ss], "greater")
        rows.append({"id": lab, "a": a, "b": b, "seeds": list(seeds), "what": what, "a_vals": va, "b_vals": vb, "k_a": ka,
                     "n_a": len(sa), "k_b": kb, "n_b": len(sb), "p": fisher_greater(ka, len(sa), kb, len(sb)) if sa and sb else None,
                     "p_barnard": barnard_greater(ka, len(sa), kb, len(sb)) if sa and sb else None, "t": st, "provisional": prov})
    m = bh_rows(rows)
    for x in rows:
        x["t_p"] = x["t"]["p_t"]
    mt = bh_rows(rows, "t_p", "t_bh")
    print(f"\n== A7 (1) reliability: seeds whose s4 read-out final3 ({S4K}) >= that seed's Latin-propagation level; one-sided"
          f" (the hypothesis' direction): Fisher exact on the counts = the decision (BH m = {m} of {len(rows)}), Barnard exact and"
          f" the paired seed-level t on final3 beside (t BH m = {mt}); all6@1e-3's seeds 0-7 were seen when the hypothesis was formed,"
          " the none runs are new")
    for x in rows:
        sig = x["p_bh"] is not None and x["p_bh"] < Q_BH and not x["provisional"]
        x["significant"] = sig
        tt = x["t"]
        verdict = ("PROVISIONAL (n=" + f"{x['n_a']}+{x['n_b']} of {2 * len(x['seeds'])}): not decided" if x["provisional"] else
                   "MORE RELIABLE with the prior (SIGNIFICANT)" if sig else "n.s.")
        print(f"  {x['id']} {x['a']} - {x['b']} seeds {x['seeds'][0]}-{x['seeds'][-1]} ({x['what']}): above the level"
              f" {x['k_a']}/{x['n_a']} vs {x['k_b']}/{x['n_b']} | final3 [" + " ".join(fmt(x["a_vals"].get(s)) for s in x["seeds"])
              + "] vs [" + " ".join(fmt(x["b_vals"].get(s)) for s in x["seeds"]) + "] | levels ["
              + " ".join(fmt(lev.get(s)) for s in x["seeds"]) + f"] | Fisher p {fmt(x['p'], 6, 4)} BH {fmt(x['p_bh'], 6, 4)}"
              f" Barnard p {fmt(x['p_barnard'], 6, 4)} | t Δ {fmt(tt['mean'], 6)} p {fmt(tt['p_t'], 6, 4)} BH {fmt(x['t_bh'], 6, 4)}"
              f" sign-flip {fmt(tt['p_sf'], 6, 4)} -> {verdict}")
    A["families"]["R"] = rows
    # ---- (2) speed
    srows = []
    for lab, a, b, seeds, what in [("S" + c[0][1:],) + c[1:] for c in comps]:
        sa, sb = have(a, seeds), have(b, seeds)
        ta = {s: first_step_ge(cells[a][s], S4K, 0.25) for s in sa}
        tb = {s: first_step_ge(cells[b][s], S4K, 0.25) for s in sb}
        u, p = gehan_perm([ta[s] for s in sa], [tb[s] for s in sb]) if sa and sb else (None, None)
        w, l_, t_, ps = sign_greater([(ta[s], tb[s]) for s in seeds if s in ta and s in tb])
        srows.append({"id": lab, "a": a, "b": b, "seeds": list(seeds), "what": what, "t_a": ta, "t_b": tb, "U": u, "p": p,
                      "sign": [w, l_, t_, ps], "median_a": km_median(list(ta.values())), "median_b": km_median(list(tb.values())),
                      "provisional": len(sa) < len(seeds) or len(sb) < len(seeds)})
    m = bh_rows(srows)
    stp = lambda t: "never" if t is None else str(t)
    print(f"\n== A7 (2) speed: the first eval step with s4 read-out >= 25% (censored at the budget, step {next(iter(next(iter(cells.values())).values()))['steps'] - 1}:"
          f" 'never'); one-sided (all6 earlier): exact permutation Gehan test = the decision (BH m = {m} of {len(srows)}), the sign test"
          " over paired seeds beside; median = the step by which half the seeds reached it")
    for x in srows:
        sig = x["p_bh"] is not None and x["p_bh"] < Q_BH and not x["provisional"]
        x["significant"] = sig
        w, l_, t_, ps = x["sign"]
        verdict = ("PROVISIONAL: not decided" if x["provisional"] else "EARLIER with the prior (SIGNIFICANT)" if sig else "n.s.")
        print(f"  {x['id']} {x['a']} - {x['b']} seeds {x['seeds'][0]}-{x['seeds'][-1]}: steps [" + " ".join(stp(x["t_a"].get(s, "-"))
              if s in x["t_a"] else "-" for s in x["seeds"]) + "] vs [" + " ".join(stp(x["t_b"][s]) if s in x["t_b"] else "-"
                                                                                   for s in x["seeds"])
              + f"] medians {stp(x['median_a'])} vs {stp(x['median_b'])} | Gehan U {x['U']} p {fmt(x['p'], 6, 4)} BH"
              f" {fmt(x['p_bh'], 6, 4)} | sign test {w} earlier / {l_} later / {t_} tied, p {fmt(ps, 6, 4)} -> {verdict}")
    A["families"]["S"] = srows

    # ---- (5) all6's best lr over the four lrs (A6 (1)'s rule) and (4) none's s4 bracket (by area, also final3)
    def lr_cells(arms, val):
        cl = {}
        for lr, a in arms.items():
            ss = have(a, DECIDE)
            if ss:
                cl[lr] = {"area": [val(a, s) for s in ss], "off": [offline_bpb(xs[(a, s)]["path"], cells[a][s]) for s in ss],
                          "intr": [final_bpb(cells[a][s]) for s in ss], "seeds": ss}
        return cl
    cl = lr_cells(ALL6_LR, lambda a, s: metrics(cells[a][s], "numbers")["area"])
    sel = RR.best_lr(cl)
    lrs = sorted(cl, key=float)
    br = RR.bracket(lrs, sel["mean"], sel["best"])
    best = sel["best"]
    a6b = (out.get("a6") or {}).get("all6_best") or {}
    A["all6_best"] = {"best": best, "near": sel["near"], "mean_area": sel["mean"], "bpb": sel["bpb"], "bracket": br,
                      "a6_best": a6b.get("best"), "provisional": any(ALL6_LR[lr] in missing for lr in ALL6_LR) or len(cl) < 4}
    print(f"\n== A7 (5) all6's best lr over {list(ALL6_LR)} (A6 (1)'s rule: the numbers area, seeds 0-3, ties within {RR.TIE:g}"
          " -> lower bpb)")
    for lr in lrs:
        print(f"  all6@{lr:5s} ({ALL6_LR[lr]:17s}) numbers area {fmt(sel['mean'][lr])} [{' '.join(fmt(v) for v in cl[lr]['area'])}]"
              f" seeds {cl[lr]['seeds']}" + (" (tied)" if lr in sel["near"] and len(sel["near"]) > 1 else ""))
    print(f"  -> all6's best lr = {best}" + (f"; bracket {br[0].upper()} at the {br[1]} edge" if br else "; bracket closed")
          + (f"; A6's best was {a6b.get('best')}: " + ("UNCHANGED" if a6b.get("best") == best else "CHANGED") if a6b else "")
          + ("  PROVISIONAL (seeds or lrs missing)" if A["all6_best"]["provisional"] else ""))
    gate8 = out.get("main", {}).get("gate", {}).get(f"k{K}|none-lr8e-3|s4_held_fresh_nonid", {}).get("verdict", "")
    s4side = "greater" if gate8.endswith("FLOOR") else "two"
    fams = out.get("main", {}).get("families", {})
    old = {x["metric"]: x for x in fams.get("F1b", {}).get("tests", [])}

    def test(a, b, dom, side, seeds=DECIDE, val=None):
        ss = [s for s in seeds if s in cells.get(a, {}) and s in cells.get(b, {})]
        g = val or (lambda arm, s: mval(xs[(arm, s)], dom))
        va, vb = {s: g(a, s) for s in ss}, {s: g(b, s) for s in ss}
        miss = [s for s in ss if va[s] is None or vb[s] is None]
        if miss:
            problems.append(f"A7: {a} - {b} {dom}: DROPPED seeds {miss} (no value)")
        ss = [s for s in ss if s not in miss]
        st = paired([va[s] - vb[s] for s in ss], side)
        return dict(st, metric=dom, a=a, b=b, side=side, seeds=ss, n_required=len(seeds), provisional=len(ss) < len(seeds),
                    a_mean=float(np.mean([va[s] for s in ss])) if ss else None, b_mean=float(np.mean([vb[s] for s in ss])) if ss else None)

    def row_txt(x, pre):
        v = ("PROVISIONAL (n=" + str(x["n"]) + " of " + str(x["n_required"]) + "): " if x["provisional"] else "") + (
            "SIGNIFICANT" if x["significant"] else "n.s.")
        ci = x.get("ci")
        ci_t = "" if ci is None else (f" 95% CI [{fmt(ci[0], 6)},{fmt(ci[1], 6)}]" if x["side"] == "two" else f" 95% LB {fmt(ci[0], 6)}")
        return (f"{pre}{x['a']:17s} - {x['b']:13s} {fmt(x['a_mean'], 6)} vs {fmt(x['b_mean'], 6)} Δ {fmt(x['mean'], 6)} "
                f"[{' '.join(fmt(d, 6) for d in x['diffs'])}] n {x['n']} {'1-sided' if x['side'] != 'two' else '2-sided'} p "
                f"{fmt(x.get('p_t'), 6, 4)} BH {fmt(x.get('p_bh'), 6, 4)} sign-flip {fmt(x.get('p_sf'), 6, 4)} wins {x['wins']}/{x['n']}"
                f"{ci_t} -> {v}")
    a_best = ALL6_LR[best]
    rows = [test(a_best, "none-lr8e-3", d, s4side if d == "s4" else "two") for d in ("numbers", "clock", "s4", "dim")]
    m = sum(x["p_t"] is not None for x in rows)
    for x, q in zip(rows, bh([x["p_t"] for x in rows])):
        x["p_bh"], x["significant"] = q, q is not None and q < Q_BH and not x["provisional"]
        y = old.get(x["metric"])
        rd = lambda z: ("gain" if z["mean"] > 0 else "harm") if z["significant"] else "n.s."
        x["reading"] = ("F1c unchanged: all6's best lr is still 1e-3, so F1c = F1b by construction" if best == "1e-3" else
                        "PROVISIONAL: cannot compare" if y is None or x["provisional"] else
                        f"no change (F1b {rd(y)}, F1c {rd(x)})" if rd(x) == rd(y) else
                        f"CHANGES the best-vs-best reading: F1b {rd(y)} -> F1c {rd(x)} (the A5 headline is not changed by it)")
    A["families"]["F1c_a7"] = rows
    print(f"  F1c updated = all6@best ({best}: {a_best}) - none@8e-3, seeds 0-3, F1b's tests (BH m = {m} of {len(rows)}; s4"
          f" {'one-sided' if s4side == 'greater' else 'two-sided'})" + ("" if best == "1e-3" else f"; all6's lr {best} was chosen on"
                                                                         " numbers on these seeds 0-3"))
    for x in rows:
        print(row_txt(x, f"    {x['metric']:9s} ") + f"  | {x['reading']}")
    # (4) none's s4 bracket over the five lrs
    print(f"\n== A7 (4) none's s4 lr bracket over {list(NONE_LR)} (seeds 0-3): by the s4 AREA (the A7 criterion, used by (3)) and"
          " by the s4 final3 (A6 (5)'s), each with readout_rules.best_lr (ties within 2 -> lower bpb, a diverged run the worst)"
          " and the bracket (OPEN = the best at an edge by > 2); the seed-level gate of every lr")
    nb = {}
    for crit, val in (("area", ar), ("final3", f3)):
        c2 = lr_cells(NONE_LR, val)
        s2 = RR.best_lr(c2)
        l2 = sorted(c2, key=float)
        b2 = RR.bracket(l2, s2["mean"], s2["best"])
        nb[crit] = {"best": s2["best"], "mean": s2["mean"], "near": s2["near"], "bracket": b2,
                    "provisional": any(NONE_LR[lr] in missing for lr in NONE_LR) or len(c2) < len(NONE_LR)}
        print(f"  by {crit:6s}: " + " | ".join(f"{lr} {fmt(s2['mean'][lr])}" for lr in l2) + f" -> none's best {s2['best']}"
              + (f"; bracket {b2[0].upper()} at the {b2[1]} edge" + (" (none's s4 may peak beyond the grid; reported as OPEN)"
                                                                     if b2[0] == "open" and b2[1] == "upper" else "") if b2 else "; bracket closed")
              + ("  PROVISIONAL" if nb[crit]["provisional"] else ""))
    gates = {lr: gate_key([cells[a][s] for s in have(a, DECIDE)], "s4", S4K) for lr, a in NONE_LR.items() if have(a, DECIDE)}
    print("  seed-level gate: " + " | ".join(f"{lr} {fmt(g['final3']).strip()} [{fmt(g['seed_ci'][0]).strip()}, {fmt(g['seed_ci'][1]).strip()}]"
                                             f" {g['verdict']}" for lr, g in gates.items()))
    nb["gates"] = {lr: g["verdict"] for lr, g in gates.items()}
    A["none_s4_best"] = nb
    # ---- (3) best vs best on s4
    nbest = nb["area"]["best"]
    na_ = NONE_LR[nbest]
    brows = [test(a_best, na_, crit, "two", DECIDE, val) for crit, val in (("s4 area", ar), ("s4 final3", f3))]
    for x in brows:
        x["group"] = "0-3"
    both47 = have(a_best, CONFIRM) and have(na_, CONFIRM)
    if both47:
        for crit, val in (("s4 area", ar), ("s4 final3", f3)):
            x = test(a_best, na_, crit, "two", CONFIRM, val)
            x["group"] = "4-7"
            brows.append(x)
    m = sum(x["p_t"] is not None for x in brows)
    for x, q in zip(brows, bh([x["p_t"] for x in brows])):
        x["p_bh"], x["significant"] = q, q is not None and q < Q_BH and not x["provisional"]
    A["families"]["B"] = brows
    print(f"\n== A7 (3) best vs best on s4: all6@best ({best}, by the numbers area) - none@best on s4 ({nbest}, by the s4 area),"
          f" two-sided paired by seed, BH m = {m} of {len(brows)}; seeds 0-3 = the seeds both lrs were chosen on; seeds 4-7 = fresh"
          + ("" if both47 else f" (not available: all6@{best} or none@{nbest} has no seeds 4-7)"))
    for x in brows:
        print(row_txt(x, f"    {x['metric']:9s} seeds {x['group']} ") + ("  [both lrs chosen on these seeds]" if x["group"] == "0-3" else ""))


def main():
    ap = argparse.ArgumentParser(description="rainbow6 (rb6) summary; see the module docstring")
    ap.add_argument("roots", nargs="+", help="directories with the rb6 run JSONs (searched recursively)")
    ap.add_argument("--ref", action="append", default=[], help="A3: glob or directory of the 2%% main setting's run JSONs"
                    " at the same lr (pt999 / pt2b 2pct none and win-ctx); repeatable")
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--a6_partial", action="store_true", help="A6.1: let the A6 section be incomplete (verdicts that need a"
                    " missing A6-section run PROVISIONAL) while the A5 section stays strict (byte-identical)")
    ap.add_argument("--a7_partial", action="store_true", help="A7: let the A7 section be incomplete (verdicts that need a"
                    " missing A7-section run PROVISIONAL) while the A5 / A6 sections stay strict (byte-identical)")
    ap.add_argument("--main_readout", action="store_true", help="A5: print the main-design read-out sections (strict dim"
                    " set, heuristic levels, families) also when only pilot arms are present")
    opt = ap.parse_args()
    partial = opt.allow_partial
    runs = load(opt.roots)
    main_mode = any(x["arm"] not in PILOT_ARMS for x in runs)
    mix = sorted({f"K{x['k']} {x['arm']}" for x in runs if x["k"] != MAIN_K or x["arm"] not in MAIN_ARMS + A6_ARMS + A7_ARMS})
    # A5 fix 6: the main design's read-out refuses the pilot's other cells (K 1 / 5, and the lambda-5 arm 'all6', not the
    # main all6 = all6-lam1) unless --main_readout is given (then they are excluded from the A5 sections with a note)
    assert not (main_mode and mix and not opt.main_readout), (
        f"main-design runs mixed with pilot-only cells {mix}: pass only the A5 read-out command's directories (runs/rb6/AUDIT.md"
        " A5), or add --main_readout to exclude them from the A5 sections")
    ident = ident6 = ident7 = None
    a7runs = [x for x in runs if is_a7(x)]   # A7: split off first; every section above A7 is computed without them
    a7_given = bool(a7runs)
    runs = [x for x in runs if not is_a7(x)]
    assert runs, "only A7 runs given: the A7 read-out needs the A5 / A6 cells"
    a6_given = any(x["arm"] in A6_ARMS for x in runs)
    if main_mode or opt.main_readout:   # A5.1: only runs whose tag's intent, run JSON and check.json agree enter any table
        runs, ident = run_identity(runs)
        if a6_given:
            runs, ident6 = run_identity(runs, A6_ARMS)
        if a7_given:
            a7runs, ident7 = run_identity(a7runs, A7_ARMS + ("none-lr1.6e-2",))
        assert runs, "no run left after the run-identity check (A5.1)"
    # integrity over every run (incl. A6 / A7: the same data per seed in every arm); A7 runs' warnings are theirs only
    warn_full = check(runs + a7runs, partial, opt.a6_partial, opt.a7_partial)
    a7keys = {f"{x['tag']} s{x['seed']}" for x in a7runs}
    warn7 = [w for w in warn_full if w.split(":")[0] in a7keys]
    warn_all = [w for w in warn_full if w.split(":")[0] not in a7keys]
    a6runs = [x for x in runs if x["arm"] in A6_ARMS]   # A6: the A5 / pilot sections are computed WITHOUT the A6 runs
    a6tags = {x["tag"] for x in a6runs}
    all_runs = runs
    runs = [x for x in runs if x["arm"] not in A6_ARMS]
    warn = [w for w in warn_all if w.split(" ")[0] not in a6tags]
    assert runs, "only A6 runs given: the A6 read-out needs the A5 cells none, none-lr8e-3 and all6-lam1"
    refs = load_refs(opt.ref, runs)
    out = {"warnings": warn, "cells": {}, "gate": {}, "paired": {}, "all6_range": {}, "provisional": any("unfinished" in w for w in warn)}
    cells = defaultdict(dict)   # (K, arm) -> seed -> run
    for x in runs:
        cells[(x["k"], x["arm"])][x["seed"]] = x["r"]
    Ks = sorted({k for k, _ in cells})
    print(f"rainbow6 summary: {len(runs)} runs, code_sha {runs[0]['r']['code_sha']}, K {Ks}, arms {sorted({a for _, a in cells})}")
    for w in warn:
        print("WARNING", w)
    r0 = runs[0]["r"]
    ro_sets = [readout_key(r0, d, t)[0] for d in EXTRA for t in ("S", "E") if t == "S" or d in CYCLIC]
    sets = (["numbers"] + ro_sets + [f"{d}_{t}" for d in EXTRA for t in ("held", "heldE", "seen")])
    sets = list(dict.fromkeys(sets))
    print("\n== per arm x set: area / final3 / t90 (M tokens; '-' never), mean over seeds [per-seed final3]")
    for (k, a), sd in sorted(cells.items()):
        for st in sets:
            ms = [metrics(sd[s], st) for s in sorted(sd)]   # A5 fix: seed order (was load order: K2 prior arms [s1, s0])
            if any(m is None for m in ms):
                continue
            ar, f3 = [m["area"] for m in ms], [m["final3"] for m in ms]
            t9 = [m["t90"] for m in ms]
            t9m = None if any(t is None for t in t9) else float(np.median(t9))
            out["cells"][f"k{k}|{a}|{st}"] = {"seeds": sorted(sd), "area": ar, "final3": f3, "t90": t9}
            print(f"  K{k} {a:14s} {st:24s} area {fmt(np.mean(ar))} final3 {fmt(np.mean(f3))} t90 {fmt(t9m, 6)} "
                  f"[{' '.join(fmt(v) for v in f3)}]")
    print("\n== paired by seed vs the same K's none (Δ area, Δ final3; per seed)")
    for (k, a), sd in sorted(cells.items()):
        if a == "none" or (k, "none") not in cells:
            continue
        nd = cells[(k, "none")]
        for st in sets:
            seeds = sorted(set(sd) & set(nd))
            pr = [(metrics(sd[s], st), metrics(nd[s], st)) for s in seeds]
            if not pr or any(p is None or q is None for p, q in pr):
                continue
            da = [p["area"] - q["area"] for p, q in pr]; df = [p["final3"] - q["final3"] for p, q in pr]
            out["paired"][f"k{k}|{a}|{st}"] = {"seeds": seeds, "d_area": da, "d_final3": df}
            print(f"  K{k} {a:14s} {st:24s} Δarea {fmt(np.mean(da), 6)} [{' '.join(fmt(v, 6) for v in da)}]  "
                  f"Δfinal3 {fmt(np.mean(df), 6)} [{' '.join(fmt(v, 6) for v in df)}] wins {sum(v > 0 for v in da)}/{len(da)}")
    print("\n== sub-splits of the held-out sets (final3, mean over seeds)")
    for (k, a), sd in sorted(cells.items()):
        for base, subs in SUBS.items():
            vals = []
            for sub in subs:
                ms = [metrics(sd[s], f"{base}_{sub}") for s in sorted(sd)]
                vals.append(None if any(m is None for m in ms) else float(np.mean([m["final3"] for m in ms])))
            if any(v is not None for v in vals):
                print(f"  K{k} {a:14s} {base:14s} " + "  ".join(f"{s_} {fmt(v)}" for s_, v in zip(subs, vals)))
    tag = " PROVISIONAL (unfinished runs)" if out["provisional"] else ""
    print(f"\n== PILOT GATE{tag} (none only): read-out final3, 90% cluster-bootstrap CI vs [chance + {MARGIN:.0f}, {CEIL:.0f}];"
          f" cyclic fallback tier E; INVALID if the seen set < {SEEN_MIN:.0f}")
    meas = {}
    none_seeds = {k: sorted(cells[(k, "none")]) for k in Ks if (k, "none") in cells}
    if not partial:
        assert all(len(v) >= 2 for v in none_seeds.values()), f"the gate needs >= 2 none seeds per K: {none_seeds}"
        assert len({tuple(v) for v in none_seeds.values()}) <= 1, f"none seeds differ across K: {none_seeds}"
    for k in Ks:
        if (k, "none") not in cells:
            continue
        nd = [cells[(k, "none")][s] for s in none_seeds[k]]
        meas[k] = {}
        f_n = [metrics(r, "numbers")["final3"] for r in nd]   # numbers: descriptive (no per-question hits stored)
        print(f"  K{k} numbers   none final3 {fmt(np.mean(f_n))} [{' '.join(fmt(v) for v in f_n)}] (descriptive)")
        out["gate"][f"k{k}|numbers"] = {"final3": float(np.mean(f_n)), "per_seed": f_n}
        for d in EXTRA:
            g = domain_gate(nd, d, "none")
            meas[k][d] = g["tier"]
            out["gate"][f"k{k}|{d}"] = g
            e_txt = (f" | E {g['setE']} {fmt(g['final3_E'])} CI [{fmt(g['ci_E'][0])},{fmt(g['ci_E'][1])}] -> {g['verdict_E']}"
                     if "verdict_E" in g else "")
            print(f"  K{k} {d:9s} {g['set']:24s} floor {fmt(g['floor'])} final3 {fmt(g['final3_S'])} CI [{fmt(g['ci_S'][0])},"
                  f"{fmt(g['ci_S'][1])}] seeds [{' '.join(fmt(v) for v in g['per_seed_S'])}] -> {g['verdict_S']}{e_txt}"
                  f" | seen [{' '.join(fmt(v) for v in g['seen_final3'])}] => {g['verdict']}{tag}")
    if meas:
        cnt = {k: sum(1 for d in EXTRA if meas[k].get(d)) for k in meas}
        bord = {k: [d for d in EXTRA if out["gate"][f"k{k}|{d}"]["verdict"] == "BORDERLINE"] for k in meas}
        kstar = max(sorted(cnt), key=lambda k: (cnt[k], -k))
        verd = {d: out["gate"][f"k{kstar}|{d}"]["verdict"] for d in EXTRA}
        out["density"] = {"measurable_per_K": cnt, "borderline_per_K": bord, "K_star": kstar, "verdicts_at_K_star": verd,
                          "two_sided": [d for d in EXTRA if meas[kstar].get(d)],
                          "one_sided_floor": [d for d in EXTRA if verd[d] == "FLOOR"],
                          "speed_only_ceiling": [d for d in EXTRA if verd[d] == "CEILING"],
                          "to_the_user": [d for d in EXTRA if verd[d] == "BORDERLINE" or verd[d].startswith("INVALID")]}
        print(f"  measurable non-number domains per K: {cnt} (borderline {bord}) -> K* = {kstar} (tie -> the smaller K){tag}. Main design"
              f" at K*: two-sided {out['density']['two_sided']}, one-sided (floor) {out['density']['one_sided_floor']}, speed"
              f" only (ceiling) {out['density']['speed_only_ceiling']}, to the user {out['density']['to_the_user']}; every"
              f" domain stays in the mixture")
        if main_mode:   # A5.1: in the main design this section is the pilot's (item-bootstrap) rule, descriptive only
            print("  (main design: the PILOT GATE lines above use the pilot's item-bootstrap rule and are DESCRIPTIVE; the"
                  " seed-level A5 gate below decides)")
    print("\n== all6 vs the same bounds (descriptive, never a gate): is there room for an effect?")
    for (k, a), sd in sorted(cells.items()):
        if a == "none" or len(sd) < 1:
            continue
        rs = [sd[s] for s in sorted(sd)]
        for d in EXTRA:
            g = domain_gate(rs, d, a)
            nd = cells.get((k, "none"), {})
            dl = [metrics(sd[s], g["set"])["final3"] - metrics(nd[s], g["set"])["final3"] for s in sorted(sd) if s in nd]
            room = {"MEASURABLE": "in range", "CEILING": "above 90: no room", "FLOOR": "at the floor",
                    "BORDERLINE": "borderline"}[g["verdict_S"]]
            out["all6_range"][f"k{k}|{a}|{d}"] = dict(g, delta_vs_none=dl, room=room)
            print(f"  K{k} {a:10s} {d:9s} {g['set']:24s} final3 {fmt(g['final3_S'])} CI [{fmt(g['ci_S'][0])},{fmt(g['ci_S'][1])}]"
                  f" -> {room}; Δ vs none per seed [{' '.join(fmt(v, 6) for v in dl)}]; seen [{' '.join(fmt(v) for v in g['seen_final3'])}]")
    print("\n== numbers (A2 + A3) vs the 2% main setting at the same lr (--ref; paired by seed: a seed draws the same numbers"
          " lines, formats and text in both). area = 100 x mean over all evals of the held-out accuracy over: all6 = the 6"
          " buckets (the 'numbers' area) | b11/12/22 (clock lines with h + k < 24 are extra additions of these buckets) |"
          " b13/23/33 (no domain line states an addition with a 3-digit operand) | b33 (no domain line or domain prior"
          " touches either operand). Prior arms, window medians of the GLOBAL clip (clip_grad_norm_ 1.0 on the LM + every"
          " active prior): pre-clip total T, clip factor c = 1/max(1, T), share of the on-evals with T > 1, the numbers"
          " prior's weighted |g| after the clip (c |g_num|) and relative to T, the LM |g| after the clip")
    out["numbers"] = {"cells": {}, "paired": {}}

    def numrow(r):
        return {"area": {g: num_area(r, bs) for g, bs in NUM_GROUPS}, "final3": metrics(r, "numbers")["final3"],
                "clip": clip_stats(r)}

    def clip_txt(cs):
        if not cs:
            return "no prior on"
        return (f"T {fmt(cs['T'], 6, 2)} c {fmt(cs['clip'], 5, 3)} clipped {cs['frac_clipped']:.2f} numbers after clip "
                f"{fmt(cs['num_after_clip'], 5, 3)} (/T {fmt(cs['num_rel_T'], 4, 2)}) LM after clip {fmt(cs['lm_after_clip'], 5, 3)}")
    ncells = [(f"K{k} {a}", sd, f"lam_rule {sorted({r['args']['lam_rule'] for r in sd.values()})} numbers lambda "
               f"{sorted({r['domains']['numbers']['lam'] for r in sd.values()})} text share "
               f"{100 * np.mean([sd[s]['curve'][-1]['token_share']['text'] for s in sorted(sd)]):.2f}%")
              for (k, a), sd in sorted(cells.items())]
    ncells += [(f"ref {tag}", {s: x["r"] for s, x in v.items()}, f"{next(iter(v.values()))['kind']} reference, code_sha "
                f"{sorted({x['r']['code_sha'] for x in v.values()})}") for tag, v in sorted(refs.items())]
    if not refs:
        print("  (no --ref given: the main setting's cells are not shown)")
    for name, sd, note in ncells:
        per = {s: numrow(r) for s, r in sorted(sd.items())}
        out["numbers"]["cells"][name] = per
        print(f"  {name:32s} " + " | ".join(f"{g} {fmt(np.mean([p['area'][g] for p in per.values()]))}" for g, _ in NUM_GROUPS)
              + f" | final3 {fmt(np.mean([p['final3'] for p in per.values()]))} seeds {sorted(per)} | {note}")
        for s, p in per.items():
            if p["clip"]:
                print(f"  {'':32s}   s{s} {clip_txt(p['clip'])}")
    for (k, a), sd in sorted(cells.items()):   # paired by seed: rb6 none - each none reference, all6 - each prior reference
        for tag, v in sorted(refs.items()):
            if next(iter(v.values()))["kind"] != rb6_kind(sd) or {x["r"]["args"]["lr"] for x in v.values()} != {
                    r["args"]["lr"] for r in sd.values()}:
                continue
            seeds = sorted(set(sd) & set(v))
            if not seeds:
                print(f"  K{k} {a} - {tag}: no common seed")
                continue
            dl = {g: [num_area(sd[s], bs) - num_area(v[s]["r"], bs) for s in seeds] for g, bs in NUM_GROUPS}
            out["numbers"]["paired"][f"k{k}|{a}-{tag}"] = {"seeds": seeds, "d_area": dl}
            print(f"  K{k} {a} - {tag} Δarea (seeds {seeds}): " + " | ".join(
                f"{g} {fmt(np.mean(x), 6)} [{' '.join(fmt(y, 6) for y in x)}]" for g, x in dl.items()))
    tags = sorted(refs)
    for i, t1 in enumerate(tags):   # yardstick: two references of one kind at a common seed = same data, another run
        for t2 in tags[i + 1:]:
            if next(iter(refs[t1].values()))["kind"] != next(iter(refs[t2].values()))["kind"] or (
                    {x["r"]["args"]["lr"] for x in refs[t1].values()} != {x["r"]["args"]["lr"] for x in refs[t2].values()}):
                continue   # A5: two lrs of one kind are not 'one setting, two runs'
            seeds = sorted(set(refs[t1]) & set(refs[t2]))
            if seeds:
                dl = {g: [num_area(refs[t1][s]["r"], bs) - num_area(refs[t2][s]["r"], bs) for s in seeds] for g, bs in NUM_GROUPS}
                out["numbers"]["paired"][f"ref|{t1}-{t2}"] = {"seeds": seeds, "d_area": dl}
                print(f"  yardstick {t1} - {t2} (one setting, two runs; seeds {seeds}): " + " | ".join(
                    f"{g} {fmt(np.mean(x), 6)} [{' '.join(fmt(y, 6) for y in x)}]" for g, x in dl.items()))
    print("  reading: none - none ref = the domain lines in the row (their LM loss, extra small additions in b11/12/22, the"
          " numbers line L tokens earlier); all6 - prior ref = that + five other priors + the shared global clip (c, numbers"
          " after clip): a lower all6 is not by itself interference. b33 is the cleanest bucket, b13/23/33 the next; a bucket"
          " group near the floor in the none rows can only separate the prior arms; the yardstick = run-to-run spread")
    print("\n== held-out-FORMAT evals (A2 + A3; descriptive): read-out set final3 (mean over seeds) of the training template vs"
          " each paraphrase template p<t> (never trained; * = NEAR: it contains the training prompt up to the last input slot),"
          " the mean drop over all templates and over the far ones, and the seen sets")
    out["formats"] = {}
    for (k, a), sd in sorted(cells.items()):
        rs = [sd[s_] for s_ in sorted(sd)]
        for d in EXTRA:
            nT = len(rs[0]["domains"][d]["formats"])
            ro = rs[0]["domains"][d]["readout"]
            base = [metrics(r, f"{d}_{ro}")["final3"] for r in rs]
            para = [[metrics(r, f"{d}_p{t}_{ro}")["final3"] for r in rs] for t in range(1, nT + 1)]
            bs_ = [metrics(r, f"{d}_seen")["final3"] for r in rs]
            ps_ = [[metrics(r, f"{d}_p{t}_seen")["final3"] for r in rs] for t in range(1, nT + 1)]
            tpls = rs[0]["domains"][d]["formats"]
            near = [is_near(d, tpls[f"p{t}"]) for t in range(1, nT + 1)]
            drop = float(np.mean(base) - np.mean([np.mean(p) for p in para]))
            far = [np.mean(p) for p, n_ in zip(para, near) if not n_]
            drop_far = float(np.mean(base) - np.mean(far)) if far else None
            out["formats"][f"k{k}|{a}|{d}"] = {"set": f"{d}_{ro}", "train_template": base, "para": para, "seen_train_template": bs_,
                                              "seen_para": ps_, "mean_drop": drop, "mean_drop_far": drop_far, "near": near,
                                              "templates": tpls}
            print(f"  K{k} {a:14s} {d:9s} {ro:22s} train-tpl {fmt(np.mean(base))} | " + " ".join(
                f"p{t + 1}{'*' if near[t] else ' '} {fmt(np.mean(p))}" for t, p in enumerate(para))
                + f" | drop {fmt(drop, 6)} far {fmt(drop_far, 6)} | seen train-tpl {fmt(np.mean(bs_))} paras "
                + " ".join(fmt(np.mean(p)) for p in ps_))
    for (k, a), sd in sorted(cells.items()):   # the prior effect per template (paired by seed vs the same K's none)
        if a == "none" or (k, "none") not in cells:
            continue
        nd = cells[(k, "none")]
        seeds = sorted(set(sd) & set(nd))
        for d in EXTRA:
            r0_ = sd[seeds[0]]
            ro = r0_["domains"][d]["readout"]
            keys = [f"{d}_{ro}"] + [f"{d}_p{t}_{ro}" for t in range(1, len(r0_["domains"][d]["formats"]) + 1)]
            dl = {kk: [metrics(sd[s_], kk)["final3"] - metrics(nd[s_], kk)["final3"] for s_ in seeds] for kk in keys}
            out["formats"][f"k{k}|{a}-none|{d}"] = dl
            tpls = r0_["domains"][d]["formats"]
            print(f"  K{k} {a}-none {d:9s} Δfinal3 train-tpl {fmt(np.mean(dl[keys[0]]), 6)} | " + " ".join(
                f"p{i + 1}{'*' if is_near(d, tpls[f'p{i + 1}']) else ' '} {fmt(np.mean(dl[kk]), 6)}" for i, kk in enumerate(keys[1:]))
                + f" (seeds {seeds})")
    print("\n== all6 sanity (descriptive): window-end geo / CKA to the true target; final bpb vs none")
    for (k, a), sd in sorted(cells.items()):
        if a == "none":
            continue
        for s, r in sorted(sd.items()):
            c = r["curve"]
            on = [e for e in c if e.get("prior_on")]
            we = on[-1] if on else None
            nd = cells.get((k, "none"), {}).get(s)
            dbpb = None if nd is None else 1000 * (c[-1]["val_bits_per_byte"] - nd["curve"][-1]["val_bits_per_byte"])
            g = " ".join(f"{d}:{fmt(we.get(f'geo_{d}'), 4, 2)}/{fmt(we.get(f'cka_{d}'), 4, 2)}" for d in EXTRA) if we else "never on"
            gn = we.get("geo") if we else None
            print(f"  K{k} {a:14s} s{s} numbers geo {fmt(gn, 4, 2)} cka_ops {fmt(we.get('cka_ops_helix3') if we else None, 4, 2)} | "
                  f"{g} | Δbpb x1e-3 {fmt(dbpb, 6)}")
    print("\n== token share / first-line share (final eval, mean over runs of each K)")
    for k in Ks:
        xs = [x["r"]["curve"][-1] for x in runs if x["k"] == k]
        ts = {t: np.mean([e["token_share"][t] for e in xs]) for t in xs[0]["token_share"]}
        fl = {t: np.mean([e["first_line_frac"][t] for e in xs]) for t in xs[0]["first_line_frac"]}
        print(f"  K{k} tokens %: " + " ".join(f"{t} {100 * v:.3f}" for t, v in ts.items()))
        print(f"  K{k} first line: " + " ".join(f"{t} {v:.3f}" for t, v in fl.items()))
        sl = {d: np.mean([e["slot_frac"][d] for e in xs], 0) for d in xs[0]["slot_frac"]}
        print(f"  K{k} domain line in slot 1 / 2 / .. after the numbers line: " + " ".join(f"{d} {'/'.join(f'{v:.3f}' for v in w)}"
              for d, w in sl.items()))
    if opt.main_readout or main_mode or any(a not in PILOT_ARMS for _, a in cells):   # A5: with main-design arms (A5.1: also
        #   when the run-identity check excluded all of them) or when asked
        main_readout(runs, cells, out, partial, forced=opt.main_readout, ident=ident)
        print(f"\n== A5 read-out status: {'PROVISIONAL (' + ('missing seeds ' + str(out['main']['missing_seeds']) if out['main']['missing_seeds'] else '--allow_partial') + ')' if out['main']['provisional'] else 'complete (every pre-registered seed present)'}")
    if a6_given:   # A6 (pre-registered before the A6 runs): appended; the sections above are those of the runs without A6
        a6_readout(all_runs, out, partial, ident6, [w for w in warn_all if w.split(" ")[0] in a6tags], a6_partial=opt.a6_partial)
    if a7_given:   # A7 (pre-registered before the A7 runs): appended; the sections above are those of the runs without A7
        a7_readout(all_runs + a7runs, out, partial, ident7, warn7, a7_partial=opt.a7_partial)
    if opt.json:
        Path(opt.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
