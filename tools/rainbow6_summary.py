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
rule. Density: K* = the K with more MEASURABLE non-number domains (tier E counts); tie -> K = 1. Main design at K*:
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
"""
from __future__ import annotations

import argparse
import glob
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
    for f in sorted(set(f.resolve() for f in files)):
        r = json.loads(f.read_text()); a = r["args"]
        assert not a.get("domains"), (str(f), "a --domains run is not a main-setting reference")
        bad = {k: [a.get(k), a0.get(k)] for k in REF_SAME if a.get(k) != a0.get(k)}
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


def main():
    ap = argparse.ArgumentParser(description="rainbow6 (rb6) summary; see the module docstring")
    ap.add_argument("roots", nargs="+", help="directories with the rb6 run JSONs (searched recursively)")
    ap.add_argument("--ref", action="append", default=[], help="A3: glob or directory of the 2%% main setting's run JSONs"
                    " at the same lr (pt999 / pt2b 2pct none and win-ctx); repeatable")
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow_partial", action="store_true")
    opt = ap.parse_args()
    partial = opt.allow_partial
    runs = load(opt.roots)
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
            ms = [metrics(r, st) for r in sd.values()]
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
                ms = [metrics(r, f"{base}_{sub}") for r in sd.values()]
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
        print(f"  measurable non-number domains per K: {cnt} (borderline {bord}) -> K* = {kstar} (tie -> 1){tag}. Main design"
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
               f"{100 * np.mean([r['curve'][-1]['token_share']['text'] for r in sd.values()]):.2f}%") for (k, a), sd in sorted(cells.items())]
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
            if next(iter(v.values()))["kind"] != ("none" if a == "none" else "prior"):
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
            if next(iter(refs[t1].values()))["kind"] != next(iter(refs[t2].values()))["kind"]:
                continue
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
    if opt.json:
        Path(opt.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
