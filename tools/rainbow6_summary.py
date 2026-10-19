"""rainbow6 summary (runs/rb6, tools/pretrain_range.py Rev B --domains; intent spec runs/rb6/AUDIT.md).

    python tools/rainbow6_summary.py <runs/rb6/out> [<more dirs>] [--ref <glob or dir> ...] [--json out.json] [--allow_partial]

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


def check(runs, partial):
    """Integrity asserts; returns a list of warnings (soft checks)."""
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
        if c[-1]["step"] != r["steps"] - 1:
            msg = f"{x['tag']} s{x['seed']}: unfinished (last eval {c[-1]['step']} of {r['steps']})"
            assert partial, msg + " (--allow_partial to look anyway; verdicts are then PROVISIONAL)"
            warn.append(msg)
        for e in c:
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
GROUPS = {"0-3": DECIDE, "2-7": FRESH, "4-7": CONFIRM, "0-7": ALL8}
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
    for e in r["curve"]:
        if not e.get("prior_on") or e.get("grad_prior") is None or not e.get("grad_lm"):
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


def main_readout(runs, cells, out, partial, forced=False):
    """A5 main-design read-out (pre-registered; AUDIT.md A5). Prints the sections and fills out['main']. Only the K = 2
    cells of the main-design arms are read (MAIN_ARMS); anything else is excluded with a printed note (main() refuses such
    a mix unless --main_readout is given). strict = not (--allow_partial or --main_readout): an incomplete family or a
    dropped seed then fails loudly."""
    K = MAIN_K
    strict = not (partial or forced)
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
    for a in ("none", ALL6M):
        if (K, a) in cells:
            print(f"  {a}: seeds {sorted(cells[(K, a)])} (0, 1 = the pilot's runs; 2-7 = main-design runs under the pilot tag "
                  f"{next(iter(cells[(K, a)].values()))['args']['tag'].lstrip('_')}: A5 reuse)")
    vs = defaultdict(list)
    for x in runs:
        vs[run_versions(x)].append(f"{x['arm']} s{x['seed']}")
    M["versions"] = {str(k): v for k, v in vs.items()}
    rec = [k for k in vs if k is not None]
    print(f"  library versions (check.json info.versions): {len(rec)} distinct recorded set(s) {rec}; not recorded for "
          f"{len(vs.get(None, []))} run(s) (checks before A5, e.g. the pilot's)" + ("  WARNING: runs differ in versions" if len(rec) > 1 else ""))
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
        if any(s in CONFIRM for s in sd):
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
                v = [metrics(r, key)[stat] for r in rs]
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
        print(f"  {a:12s} seeds {ss} " + " | ".join(cell_txt(d, v) for d, v in row.items()))
    # ---- 2. the pre-declared families (paired by seed; t test, exact sign-flip, BH within each family and seed group)
    def have(a):
        return (K, a) in cells

    s4_side_8e3 = "two"
    if have("none-lr8e-3"):
        s4_side_8e3 = "greater" if M["gate"].get(f"k{K}|none-lr8e-3|s4_held_fresh_nonid", {}).get("verdict") == "FLOOR" else "two"
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
         [(d, "all6-lam0.3", ALL6M, "two") for d in ("bpb",) + DOM6 + ("seen",)], "0-3", ("2-7",)),
        ("F6", "baseline lr: none@8e-3 - none@1e-3", [(d, "none-lr8e-3", "none", "two") for d in DOM6 + ("bpb",)], "0-3", ()),
        ("F7", "unrelated-task harm: single prior - none on every OTHER domain",
         [(e, f"only-{d}", "none", "two") for d in DOM6 for e in DOM6 if e != d], "0-3", ()),
        ("F8", "LM cost: offline bpb (full val) x 1000, prior arm - none",
         [("bpb", a, "none", "two") for a in (ALL6M, "all6-lam0.3", *SINGLES, "all6-shuf")], "0-3", ()),
    ]
    problems = []

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
        return dict(st, metric=dom, a=aa, b=bb, side=side, group=grp, seeds=seeds, n_common=len(common),
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
                f"wins {x['wins']}/{x['n']}{ci_t} -> {'SIGNIFICANT' + (' (robust)' if x['robust'] else '') if x['significant'] else 'n.s.'}"
                + (" [identical differences in every seed: t undefined, p = its limit]" if x.get("degenerate") else ""))

    print(f"\n== A5 families (paired by seed, K{K}): mean Δ [per seed in seed order], paired t (df n-1), exact sign-flip p (floor"
          f" 2/2^n two-sided, 1/2^n one-sided: .125 / .0625 at n = 4, so it can never reach .05 there and is not a decision rule),"
          f" BH within the family and seed group; SIGNIFICANT = BH-adjusted t p < {Q_BH} on the family's DECISION seeds (F1, F1b"
          f" and F2-F8: 0-3; FE: the fresh seeds 2-7); 'robust' (all seeds in the direction of the mean) is reported, not a rule;"
          f" the other seed groups are descriptive")
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
                c_["status"] = ("nothing to confirm (n.s. on seeds 0-3)" if not x["significant"] else
                                "CONFIRMED on seeds 4-7" if sg * c_["mean"] > 0 and (c_["wins"] if sg > 0 else c_["losses"]) >= 0.75 * c_["n"]
                                else "NOT confirmed (seeds 4-7 disagree)")
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
            elif dec == "0-3" and x["significant"] and {x["a"], x["b"]} <= {ALL6M, "none", "none-lr8e-3"}:
                print(f"    {'':30s} confirm seeds 4-7: none yet (blocks 470-477 / 490-493) -> unconfirmed")
    if problems:
        print(f"  {len(problems)} family problem(s): {problems}")
        assert not strict, ("A5 families: a seed was dropped or a pre-declared test is missing (the BH m would change): "
                            + "; ".join(problems) + " (--allow_partial to look at an incomplete batch anyway)")
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
        else:
            x1, x2, how = find("F1", ALL6M, "none", d), find("F1b", ALL6M, "none-lr8e-3", d), "F1 / F1b, seeds 0-3"
        if x1 is None or x2 is None:
            st = f"not decidable yet ({how}: {'F1' if x1 is None else 'F1b'} missing)"
        elif not (x1["significant"] and x2["significant"] and (x1["mean"] > 0) == (x2["mean"] > 0)):
            st = (f"not a headline ({how}: F1 {'SIGNIFICANT' if x1['significant'] else 'n.s.'} (n {x1['n']}), F1b "
                  f"{'SIGNIFICANT' if x2['significant'] else 'n.s.'} (n {x2['n']})"
                  + (", opposite signs" if x1["significant"] and x2["significant"] else "") + ")")
        else:
            dirn = "gain" if x1["mean"] > 0 else "harm"
            cs = {"F1": x1.get("confirm"), "F1b": x2.get("confirm")}
            bad = [f for f, c_ in cs.items() if c_ is not None and not c_["status"].startswith("CONFIRMED")]
            miss = [f for f, c_ in cs.items() if c_ is None]
            if d in TIER_E:
                st = f"HEADLINE {dirn} ({how}; decided on fresh seeds, no separate confirmation)"
            elif bad:
                st = f"not a headline: significant in F1 and F1b ({how}) but the confirm seeds 4-7 of {'/'.join(bad)} disagree"
            elif miss:
                st = f"HEADLINE {dirn} ({how}), unconfirmed (confirm seeds 4-7 of {'/'.join(miss)} missing)"
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
            M["noninferiority"][x["metric"]] = {"lb90": x["lb90"], "margin": mg, "kept": ok, "F1_gain": gain}
            print(f"    {x['metric']:9s} lb90 {fmt(x['lb90'], 6)} vs -{mg:.0f}: {'kept' if ok else 'not shown'}"
                  f" (F1 gain: {'yes' if gain else 'no'})")
    # ---- 5. gradient split per prior arm (user principle: the prior is auxiliary)
    print("\n== A5 gradient split at the on-evals, every prior arm and seed (window medians before the global clip): LM share"
          " grad_lm / T, clip factor 1/max(1, T), grad_prior / grad_lm, share of on-evals with grad_prior > grad_lm; FLAG ="
          " the prior gradient exceeds the LM gradient (median ratio > 1 in any seed)")
    for (k, a), sd in sorted(cells.items()):
        gs = {s: grad_split(sd[s]) for s in sorted(sd)}
        if not any(gs.values()):
            continue
        flag = any(g and g["prior_over_lm"] > 1 for g in gs.values())
        M["grad"][f"k{k}|{a}"] = {"per_seed": gs, "flag": flag}
        print(f"  K{k} {a:12s} " + " | ".join(
            f"s{s} LM share {fmt(g['lm_share'], 4, 2)} c {fmt(g['clip'], 4, 2)} prior/LM {fmt(g['prior_over_lm'], 5, 2)}"
            f" (>1 on {g['frac_prior_gt_lm']:.2f})" for s, g in gs.items() if g) + ("  FLAG: prior > LM" if flag else ""))


def main():
    ap = argparse.ArgumentParser(description="rainbow6 (rb6) summary; see the module docstring")
    ap.add_argument("roots", nargs="+", help="directories with the rb6 run JSONs (searched recursively)")
    ap.add_argument("--ref", action="append", default=[], help="A3: glob or directory of the 2%% main setting's run JSONs"
                    " at the same lr (pt999 / pt2b 2pct none and win-ctx); repeatable")
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--main_readout", action="store_true", help="A5: print the main-design read-out sections (strict dim"
                    " set, heuristic levels, families) also when only pilot arms are present")
    opt = ap.parse_args()
    partial = opt.allow_partial
    runs = load(opt.roots)
    main_mode = any(x["arm"] not in PILOT_ARMS for x in runs)
    mix = sorted({f"K{x['k']} {x['arm']}" for x in runs if x["k"] != MAIN_K or x["arm"] not in MAIN_ARMS})
    # A5 fix 6: the main design's read-out refuses the pilot's other cells (K 1 / 5, and the lambda-5 arm 'all6', not the
    # main all6 = all6-lam1) unless --main_readout is given (then they are excluded from the A5 sections with a note)
    assert not (main_mode and mix and not opt.main_readout), (
        f"main-design runs mixed with pilot-only cells {mix}: pass only the A5 read-out command's directories (runs/rb6/AUDIT.md"
        " A5), or add --main_readout to exclude them from the A5 sections")
    warn = check(runs, partial)
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
    if opt.main_readout or any(a not in PILOT_ARMS for _, a in cells):   # A5: only with main-design arms (or asked)
        main_readout(runs, cells, out, partial, forced=opt.main_readout)
    if opt.json:
        Path(opt.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
