"""Read-out of the 190M rainbow follow-up batch rb6_190m_ext (Torch, llmadd_torch/runs/rb6_190m_ext), PRE-REGISTERED in
llmadd_torch/runs/rb6_190m_ext/AUDIT.md before any result of the batch. A thin second wrapper: the metric definitions,
gate, s4_hard, paired test, BH, gradient split and integrity checks are tools/rainbow6_summary.py's, the Torch run
identity is tools/rb6scale_summary.py's torch_identity (both used as they are; neither file is changed, so the rb6_190m
read-out stays byte-identical). Everything is descriptive on seeds 0-3 (the lr-selection seeds of the main batch): no
confirmatory claim, no size-trend claim (only 190M runs are read).
Cells (190M, seeds 0-3; labelled arm@lr by the actual lr, never "none@best"):
  main batch (runs/rb6_190m/out)     none@6e-4, none@4.8e-3, all6-lam1@6e-4, only-numbers@6e-4
  this batch (runs/rb6_190m_ext/out) all6-lam1@1.2e-3, only-numbers@1.2e-3, only-clock@6e-4, all6-minus-clock@6e-4 (the
                                     numbers, months, weekdays, s4, dim priors; the clock prior off), none@9.6e-3,
                                     none@1.92e-2
Output: per cell the A5 READ metric per domain (numbers area; months / weekdays tier-E final3; clock wrap final3; s4
fresh non-id final3; dim strict final3), s4_hard area (+ A8.2 'learned' at the last eval), offline bpb x1e3, seen, the
seed-level gate (90% t interval vs [chance + 5, 90]; A5's INVALID rule for none cells) and the gradient split of the prior
cells; then the pre-registered questions (rules: AUDIT.md "Read-out"):
  Q1 (lr)          all6 - only-numbers per domain at 6e-4 and at 1.2e-3 side by side (paired by seed); the clock reading
                   r = d(1.2e-3) / d(6e-4): PERSISTS (r >= 2/3 and >= 3/4 seeds negative), LARGELY GONE (r <= 1/3),
                   else PARTIAL / UNDETERMINED; plus the within-arm lr contrasts (1.2e-3 - 6e-4)
  Q2 (attribution) clock of only-clock vs only-numbers vs none@6e-4, all6-minus-clock vs all6 vs only-numbers (all 6e-4);
                   f = (mc - on) / (all6 - on) on the seed means: f >= 2/3 -> ~ all6 (the gap is NOT from the clock
                   prior), f <= 1/3 -> ~ only-numbers (the clock prior, with the others, is involved), else UNDETERMINED;
                   |all6 - on| < 10 points -> UNDETERMINED (no gap to attribute); per-seed f_s printed (seeds in the
                   same third); plus what all6-minus-clock / only-clock do to the other domains
  Q3 (none, high lr) for s4 (fresh non-id final3) and dim (strict final3) the gate of every none config over {6e-4,
                   4.8e-3, 9.6e-3, 1.92e-2}: FLOOR -> AT THE FLOOR; seed-level lower bound >= floor (MEASURABLE /
                   CEILING / BORDERLINE above the floor) -> LEFT THE FLOOR; otherwise UNDETERMINED; INVALID not counted;
                   s4_hard area + A8.2 learned k/4 beside it; the best tested none per metric (highest mean; INVALID
                   excluded; tie -> the lower lr) is the 190M comparison for S4 / dim: all6@6e-4 (and all6@1.2e-3) - that
                   none, labelled "best tested, chosen on these seeds 0-3".
A missing / excluded / unfinished run -> PROVISIONAL (n=k of 4) on every line that needs it; without --allow_partial the
read-out refuses. A diverged run is data (A6 rule: its values over all evals count, its bpb is NOT TESTED).
    python tools/rb6scale_ext_summary.py <runs/rb6_190m/out> <runs/rb6_190m_ext/out> [--allow_partial] [--json out.json]
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rainbow6_summary as R6     # noqa: E402
import rb6scale_summary as S      # noqa: E402
import make_rb6m_tasks as G       # noqa: E402  (the module object rb6scale_summary.torch_identity reads)

SIZE, SEEDS = "190m", (0, 1, 2, 3)
DOMS = R6.DOM6
# the arm the generator (llmadd_torch/make_jobs_rb6_190m_ext.py) adds to the SPEC table in memory; the same tuple here
# (torch_identity checks the run against make_rb6m_tasks.intent(<arm>))
NEW_ARMS = {"all6-minus-clock": ("helix3", ("numbers", "months", "weekdays", "s4", "dim"), (), "1", "1e-3")}
ACTIVE = {"none": (), "all6-lam1": DOMS, "only-numbers": ("numbers",), "only-clock": ("clock",),
          "all6-minus-clock": ("numbers", "months", "weekdays", "s4", "dim")}
CELLS = {   # cell label -> (arm, lr as in the tag); MAIN = runs/rb6_190m, the rest runs/rb6_190m_ext
    "none@6e-4": ("none", "6e-4"), "none@4.8e-3": ("none", "4.8e-3"), "all6-lam1@6e-4": ("all6-lam1", "6e-4"),
    "only-numbers@6e-4": ("only-numbers", "6e-4"),
    "all6-lam1@1.2e-3": ("all6-lam1", "1.2e-3"), "only-numbers@1.2e-3": ("only-numbers", "1.2e-3"),
    "only-clock@6e-4": ("only-clock", "6e-4"), "all6-minus-clock@6e-4": ("all6-minus-clock", "6e-4"),
    "none@9.6e-3": ("none", "9.6e-3"), "none@1.92e-2": ("none", "1.92e-2")}
MAIN = ("none@6e-4", "none@4.8e-3", "all6-lam1@6e-4", "only-numbers@6e-4")
NONE_GRID = ("none@6e-4", "none@4.8e-3", "none@9.6e-3", "none@1.92e-2")
ARM_RE = re.compile(r"^(none|all6-lam1|only-numbers|only-clock|all6-minus-clock)-lr(.+)$")
LABEL = "[seeds 0-3 (the main batch's lr-selection seeds): descriptive, no confirmatory claim]"
METRICS = DOMS + ("s4_hard", "bpb", "seen")
THIRD = (1 / 3, 2 / 3)
MIN_GAP = 10.0
NOTES = [
    "lrs: none's 6e-4 / 4.8e-3 and the prior arms' 6e-4 are reused from pt2s190 (chosen on the numbers area, seeds 0-3); "
    "1.2e-3 / 9.6e-3 / 1.92e-2 are one grid step (x2) further; no cell is 'at its best lr' for a domain",
    "Q3 'best tested' = the best of the four tested none lrs on that metric, chosen on the same seeds 0-3 (optimistic for "
    "none); not 'none at its best lr', not comparable with A8.3 (53M, fresh seeds 8-15)",
    "no size-trend claim (F87): only 190M runs are read here; 53M values are not compared",
]


def register():
    for a, spec in NEW_ARMS.items():
        assert a not in G.SPEC or G.SPEC[a] == spec, (a, G.SPEC.get(a))
        G.SPEC[a] = spec
    assert G.SPEC["only-clock"] == ("none", ("clock",), (), "1", "1e-3"), G.SPEC["only-clock"]
    for a, act in ACTIVE.items():
        assert tuple(G.SPEC[a][1]) == tuple(act), (a, G.SPEC[a])


def fmt(v, w=6, p=1):
    return S.fmt(v, w, p)


def floor_label(g):
    """Q3's reading of one seed-level gate dict {'verdict', 'seed_ci', 'floor'} (pre-registered, AUDIT.md)."""
    v = g["verdict"]
    if v.startswith("INVALID"):
        return "INVALID (not counted)"
    lo = g["seed_ci"][0]
    if v == "FLOOR":
        return "AT THE FLOOR"
    if lo is None:
        return "n/a (one seed)"
    if lo >= g["floor"]:
        return "LEFT THE FLOOR"
    return "UNDETERMINED (BORDERLINE: the seed interval straddles the floor)"


def floor_answer(labels):
    """{cell: floor_label} -> the Q3 answer for one metric."""
    valid = {c: l for c, l in labels.items() if not l.startswith(("INVALID", "n/a"))}
    left = [c for c, l in valid.items() if l == "LEFT THE FLOOR"]
    if left:
        return f"YES: {', '.join(left)} left the floor"
    if valid and all(l == "AT THE FLOOR" for l in valid.values()):
        return f"NO: every tested valid none config is at the floor ({', '.join(valid)})"
    if not valid:
        return "NOT DETERMINED: no valid none config"
    return ("UNDETERMINED: none left the floor; " + ", ".join(c for c, l in valid.items() if l.startswith("UNDETERMINED"))
            + " straddle(s) it")


def best_tested(means, invalid=()):
    """{cell: mean} -> the best tested none cell (highest mean; INVALID cells excluded; a tie within 1e-9 -> the lower lr)."""
    cand = [(c, m) for c, m in means.items() if c not in invalid and m is not None and math.isfinite(m)]
    if not cand:
        return None
    top = max(m for _, m in cand)
    tied = [c for c, m in cand if m >= top - 1e-9]
    return min(tied, key=lambda c: float(CELLS[c][1]))


def thirds(f, near_hi, near_lo, mid):
    """f >= 2/3 -> near_hi; f <= 1/3 -> near_lo; else mid."""
    return near_hi if f >= THIRD[1] - 1e-9 else near_lo if f <= THIRD[0] + 1e-9 else mid   # 1e-9: float noise at a boundary


def attribution(mc, a6, on):
    """Q2 (pre-registered): per-seed or mean clock values of all6-minus-clock (mc), all6 (a6), only-numbers (on) ->
    (label, f). f = (mc - on) / (a6 - on): the share of the all6 - only-numbers gap that removing the clock prior leaves."""
    gap = a6 - on
    if abs(gap) < MIN_GAP:
        return f"UNDETERMINED (|all6 - only-numbers| = {abs(gap):.1f} < {MIN_GAP:.0f} points: no gap to attribute)", None
    f = (mc - on) / gap
    lab = thirds(f, "~ all6: the clock gap is NOT from the clock prior (removing it leaves the gap)",
                 "~ only-numbers: the clock prior (together with the other four domain priors) is involved in the gap",
                 "UNDETERMINED (in between)")
    return lab, f


def persistence(d12, d6, diffs12):
    """Q1 (pre-registered): r = d(1.2e-3) / d(6e-4) of all6 - only-numbers on clock; PERSISTS needs r >= 2/3 and at least
    3 of 4 seeds with the same sign as d(6e-4)."""
    if d6 is None or d12 is None or abs(d6) < MIN_GAP:
        return "UNDETERMINED (no 6e-4 gap of >= 10 points)", None
    r = d12 / d6
    same = sum((x < 0) == (d6 < 0) and x != 0 for x in diffs12)
    if r >= THIRD[1] - 1e-9:
        lab = "PERSISTS at 1.2e-3" if same >= 3 else f"UNDETERMINED (r >= 2/3 but only {same}/{len(diffs12)} seeds in the direction)"
    elif r <= THIRD[0] + 1e-9:
        lab = "LARGELY GONE at 1.2e-3"
    else:
        lab = "PARTIAL / UNDETERMINED"
    return lab, r


def load_cells(roots, allow_partial):
    """-> cells {label: {seed: x}}, ignored, excluded, warnings (rainbow6_summary.load / check, torch_identity)."""
    runs = R6.load(roots)
    cells, ignored, excluded = defaultdict(dict), [], []
    for x in runs:
        m = R6.TAG.match(x["tag"])
        am = ARM_RE.match(x["arm"])
        lab = f"{x['tag']} s{x['seed']} ({Path(x['path']).parent.parent.name})"
        cell = None
        if m["size"] == SIZE and am:
            cell = next((c for c, (a, l_) in CELLS.items() if a == am[1] and float(l_) == float(am[2])), None)
        if cell is None or x["seed"] not in SEEDS or x["k"] != 2:
            ignored.append(lab)
            continue
        assert x["seed"] not in cells[cell], f"two runs of {cell} s{x['seed']}"
        x["cell"] = cell
        cells[cell][x["seed"]] = x
    for c in list(cells):
        arm, lr = CELLS[c]
        for s, x in list(cells[c].items()):
            bad = S.torch_identity(x, SIZE, arm, lr)
            if bad:
                excluded.append((f"{c} s{s} ({Path(x['path']).parent.parent.name})", bad))
                del cells[c][s]
    xs = [x for c in cells.values() for x in c.values()]
    R6.A6_ARMS = R6.A6_ARMS + tuple(sorted({x["arm"] for x in xs} - set(R6.A6_ARMS)))   # a diverged run is data (A6)
    warn = R6.check(xs, allow_partial) if xs else []
    for c in cells:   # unfinished = missing (A5.1 rule 1)
        cells[c] = {s: x for s, x in cells[c].items() if x["r"]["curve"][-1]["step"] == x["r"]["steps"] - 1}
    return cells, ignored, excluded, warn


def main():
    ap = argparse.ArgumentParser(description="rb6_190m_ext read-out; see the module docstring")
    ap.add_argument("roots", nargs="+", help="runs/rb6_190m/out and runs/rb6_190m_ext/out (searched recursively)")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--json", default=None)
    opt = ap.parse_args()
    register()
    cells, ignored, excluded, warn = load_cells(opt.roots, opt.allow_partial)
    n_in = sum(len(v) for v in cells.values())
    print(f"rb6_190m_ext read-out (190M): {n_in} runs in the design (10 cells x 4 seeds = 40), {len(ignored)} ignored, "
          f"{len(excluded)} excluded; mode {'PARTIAL (--allow_partial: every verdict PROVISIONAL)' if opt.allow_partial else 'strict'}")
    print("  cells: " + ", ".join(CELLS) + f" {LABEL}")
    for n_ in NOTES:
        print(f"  NOTE (pre-registered): {n_}")
    for lab in ignored:
        print(f"  IGNORED (not a design cell / seed): {lab}")
    for lab, why in excluded:
        print(f"  EXCLUDED run {lab}: " + "; ".join(why))
    for w in warn:
        print(f"  WARNING {w}")
    print("  data identity: per seed every run has the same data_sha at every eval and the same step-0 lm "
          "(rainbow6_summary.check)")
    missing = {c: [s for s in SEEDS if s not in cells.get(c, {})] for c in CELLS}
    miss_txt = [f"{c} {v}" for c, v in missing.items() if v]
    if miss_txt and not opt.allow_partial:
        raise AssertionError("rb6_190m_ext read-out REFUSED (strict): missing / excluded / unfinished seeds " + "; ".join(miss_txt)
                             + " (--allow_partial to look anyway: PROVISIONAL)")
    out = {"cells": {}, "families": {}, "readings": {}, "ignored": ignored, "excluded": [[a, b] for a, b in excluded],
           "warnings": warn, "missing": missing, "notes": NOTES, "grad_split": {}}
    prov = lambda *cs: any(missing[c] for c in cs)

    # ---- per cell (descriptive)
    print("\n== 190M: per cell, the A5 READ metric per domain, s4_hard area, offline bpb x1e3, seen; mean over seeds [per seed]; "
          f"the seed-level gate (90% t interval vs [chance + 5, 90]); none cells: A5's INVALID rule (seen final3 < {R6.SEEN_MIN:.0f})")
    gates = {}
    for c in CELLS:
        cs = cells.get(c, {})
        seeds = sorted(cs)
        arm = CELLS[c][0]
        print(f"  {c:22s} (active priors: {','.join(ACTIVE[arm]) or 'none'}) "
              + (f"PROVISIONAL (n={len(seeds)} of 4)" if missing[c] else "n=4"))
        for mt in METRICS:
            vals = [S.value(cs[s], mt) for s in seeds]
            ok = [v for v in vals if v is not None and math.isfinite(v)]
            mean = float(np.mean(ok)) if vals and len(ok) == len(vals) else None
            extra, g, seen_d = "", None, None
            if mt in R6.EXTRA and seeds:
                g = R6.gate_key([cs[s]["r"] for s in seeds], mt, R6.READ[mt][0])
                seen_d = [R6.metrics(cs[s]["r"], f"{mt}_seen")["final3"] for s in seeds]
                if arm == "none" and min(seen_d) < R6.SEEN_MIN:
                    g["verdict"] = f"INVALID (seen {min(seen_d):.0f} < {R6.SEEN_MIN:.0f})"
                ci = g["seed_ci"]
                extra = (f" | seed CI [{fmt(ci[0], 5)},{fmt(ci[1], 5)}] vs floor {g['floor']:.1f} -> {g['verdict']}; seen "
                         f"[{' '.join(fmt(v, 5) for v in seen_d)}]")
                gates[(c, mt)] = g
            if mt == "s4_hard" and seeds:
                hm = [R6.s4_hard_metrics(cs[s]["r"]) for s in seeds]
                extra = f" | learned (A8.2, last eval vs the exact filler chance, alpha .01) {sum(h['learned'] for h in hm)}/{len(hm)}"
            if mt == "bpb" and any(R6.diverged_at(cs[s]["r"]) is not None for s in seeds):
                extra = " | DIVERGED seed(s): " + ",".join(f"s{s}" for s in seeds if R6.diverged_at(cs[s]["r"]) is not None)
            print(f"    {mt:8s} {fmt(mean)} [{' '.join(fmt(v, 5) for v in vals)}]{extra}")
            out["cells"][f"{c}|{mt}"] = {"seeds": seeds, "values": vals, "mean": mean, "seen_per_seed": seen_d,
                                         "gate": g and {k: g[k] for k in ("verdict", "seed_ci", "floor", "chance")},
                                         "provisional": bool(missing[c])}
        if arm != "none" and seeds:   # the gradient split of every prior cell (A5 / manifold_aux_loss_principle)
            rows, flag = [], []
            for s in seeds:
                gsp = R6.grad_split(cs[s]["r"])
                if gsp is None:
                    rows.append(f"s{s} NOT LOGGED")
                    out["grad_split"][f"{c}|s{s}"] = "NOT LOGGED"
                    continue
                flag += [s] if gsp["prior_over_lm"] > 1 else []
                rows.append(f"s{s} LM share {gsp['lm_share']:.2f} prior/LM {gsp['prior_over_lm']:.2f} clip {gsp['clip']:.2f} "
                            f"prior>LM {gsp['frac_prior_gt_lm']:.2f}")
                out["grad_split"][f"{c}|s{s}"] = gsp
            print("    gradient split (window medians at the on-evals): " + "; ".join(rows))
            print("    " + (f"FLAG: median prior / LM > 1 at seeds {flag} (A5: the prior is auxiliary)" if flag else
                            "no seed with a median prior / LM > 1 (A5 FLAG not raised)"))
    h = {s: R6.s4_hard(x["r"]) for c in cells.values() for s, x in c.items()}
    print("    s4_hard per seed (n / exact filler chance / Latin level; the same for every cell of a seed): "
          + " ".join(f"s{s} {v['n']}/{v['chance']:.1f}/{v['level']:.1f}" for s, v in sorted(h.items())))

    def family(name, desc, items, side_of=lambda a, b, m: "two", also=()):
        """items [(a, b, metric)] -> paired rows (A5 style), BH within the family, printed and stored; a row is
        PROVISIONAL when a or b (or a cell in also: Q3's selection grid) misses a seed."""
        res = []
        for a, b, mt in items:
            ca, cb = cells.get(a, {}), cells.get(b, {})
            seeds = [s for s in SEEDS if s in ca and s in cb]
            va = [S.value(ca[s], mt) for s in seeds]
            vb = [S.value(cb[s], mt) for s in seeds]
            side = side_of(a, b, mt)
            bad = [s for s, p, q in zip(seeds, va, vb) if p is None or q is None or not (math.isfinite(p) and math.isfinite(q))]
            if bad or len(seeds) < 2:
                res.append(dict(a=a, b=b, m=mt, side=side, seeds=seeds, p_t=None,
                                note=f"NOT TESTED: no finite value at seeds {bad} (missing offline bpb or DIVERGED)" if bad
                                else f"NOT TESTED: {len(seeds)} paired seed(s)"))
                continue
            res.append(dict(a=a, b=b, m=mt, side=side, seeds=seeds, **R6.paired([p - q for p, q in zip(va, vb)], side)))
        for r_, q in zip(res, R6.bh([r_.get("p_t") for r_ in res])):
            r_["p_bh"] = q
        print(f"  -- {name}: {desc}")
        for r_ in res:
            head = f"    {r_['m']:8s} {r_['a'] + ' - ' + r_['b']:44s}"
            if r_.get("note"):
                print(head + " " + r_["note"])
                continue
            pv = prov(r_["a"], r_["b"], *also)
            r_["provisional"] = pv
            ci = r_.get("ci") or [None, None]
            sig = " sig. at BH .05 (descriptive)" if (r_["p_bh"] is not None and r_["p_bh"] < 0.05 and not pv) else ""
            flag = " LM COST FLAG (mean > +4)" if r_["m"] == "bpb" and r_["mean"] is not None and r_["mean"] > 4 else ""
            print(f"{head} {fmt(r_['mean'], 7)} [{' '.join(fmt(v, 6) for v in r_['diffs'])}] wins {r_['wins']}/{r_['n']} "
                  f"t-p {fmt(r_['p_t'], 5, 3)} ({r_['side']}) sf-p {fmt(r_['p_sf'], 5, 3)} BH {fmt(r_['p_bh'], 5, 3)} "
                  f"95% CI [{fmt(ci[0], 6)},{fmt(ci[1], 6)}]{sig}{flag}"
                  + (f" PROVISIONAL (n={r_['n']} of 4)" if pv else ""))
        out["families"][name] = res
        return {(r_["a"], r_["b"], r_["m"]): r_ for r_ in res}

    mean_of = lambda c, mt: out["cells"].get(f"{c}|{mt}", {}).get("mean")
    # ---- Q1
    print(f"\n== Q1 (lr): all6 - only-numbers at 6e-4 and at 1.2e-3, paired by seed {LABEL}")
    print("  side by side (means): domain   all6@6e-4  only-num@6e-4  diff | all6@1.2e-3  only-num@1.2e-3  diff")
    for d in DOMS:
        a6, o6, a12, o12 = (mean_of(c, d) for c in ("all6-lam1@6e-4", "only-numbers@6e-4", "all6-lam1@1.2e-3", "only-numbers@1.2e-3"))
        df = lambda p, q: None if p is None or q is None else p - q
        print(f"    {d:9s} {fmt(a6, 9)} {fmt(o6, 13)} {fmt(df(a6, o6), 7)} | {fmt(a12, 11)} {fmt(o12, 15)} {fmt(df(a12, o12), 7)}")
    q1 = family("Q1", "all6 - only-numbers per lr (same lr within a pair); the clock rows are the pre-registered Q1 rows",
                [(f"all6-lam1@{lr}", f"only-numbers@{lr}", d) for lr in ("6e-4", "1.2e-3") for d in DOMS])
    family("Q1-lr", "lr 1.2e-3 - lr 6e-4 within an arm (descriptive)",
           [(f"{a}@1.2e-3", f"{a}@6e-4", d) for a in ("all6-lam1", "only-numbers") for d in DOMS])
    r6, r12 = q1.get(("all6-lam1@6e-4", "only-numbers@6e-4", "clock")), q1.get(("all6-lam1@1.2e-3", "only-numbers@1.2e-3", "clock"))
    if r6 and r12 and not r6.get("note") and not r12.get("note"):
        lab, r = persistence(r12["mean"], r6["mean"], r12["diffs"])
    else:
        lab, r = "NOT DETERMINED (a clock row is not tested)", None
    pv = prov("all6-lam1@6e-4", "only-numbers@6e-4", "all6-lam1@1.2e-3", "only-numbers@1.2e-3")
    print(f"  Q1 READING (clock, pre-registered): d(6e-4) {fmt(r6 and r6.get('mean'), 6)}, d(1.2e-3) {fmt(r12 and r12.get('mean'), 6)}, "
          f"r = {fmt(r, 5, 2)} -> {lab}" + (" PROVISIONAL" if pv else "")
          + " [two lrs per prior arm: still not 'all6 at its best lr'; descriptive]")
    out["readings"]["Q1"] = {"d6": r6 and r6.get("mean"), "d12": r12 and r12.get("mean"), "r": r, "label": lab, "provisional": pv}

    # ---- Q2
    print(f"\n== Q2 (attribution of the clock gap at 6e-4) {LABEL}")
    for c in ("none@6e-4", "only-numbers@6e-4", "only-clock@6e-4", "all6-minus-clock@6e-4", "all6-lam1@6e-4"):
        print(f"    clock  {c:24s} {fmt(mean_of(c, 'clock'))} [{' '.join(fmt(v, 5) for v in out['cells'].get(c + '|clock', {}).get('values', []))}]")
    q2 = family("Q2", "clock (wrap final3) at 6e-4: the clock prior alone, and all6 without the clock prior",
                [("only-clock@6e-4", "none@6e-4", "clock"), ("only-clock@6e-4", "only-numbers@6e-4", "clock"),
                 ("only-numbers@6e-4", "none@6e-4", "clock"), ("all6-minus-clock@6e-4", "all6-lam1@6e-4", "clock"),
                 ("all6-minus-clock@6e-4", "only-numbers@6e-4", "clock"), ("all6-lam1@6e-4", "only-numbers@6e-4", "clock")])
    mc, a6, on = (out["cells"].get(f"{c}|clock", {}) for c in ("all6-minus-clock@6e-4", "all6-lam1@6e-4", "only-numbers@6e-4"))
    pv = prov("all6-minus-clock@6e-4", "all6-lam1@6e-4", "only-numbers@6e-4")
    if all(x.get("mean") is not None for x in (mc, a6, on)):
        lab, f = attribution(mc["mean"], a6["mean"], on["mean"])
        ss = [s for s in SEEDS if s in mc["seeds"] and s in a6["seeds"] and s in on["seeds"]]
        val = lambda x, s: x["values"][x["seeds"].index(s)]
        per = {s: attribution(val(mc, s), val(a6, s), val(on, s)) for s in ss}
        same = sum(per[s][0] == lab for s in ss)
        print(f"  Q2 READING (pre-registered): gap all6 - only-numbers {a6['mean'] - on['mean']:+.1f}; all6-minus-clock - "
              f"only-numbers {mc['mean'] - on['mean']:+.1f}; f = {fmt(f, 5, 2)} -> {lab}" + (" PROVISIONAL" if pv else ""))
        print("    per seed f_s: " + " ".join(f"s{s} {fmt(per[s][1], 5, 2)}" for s in ss)
              + f"; {same}/{len(ss)} seeds give the same reading (descriptive)")
        out["readings"]["Q2"] = {"label": lab, "f": f, "gap": a6["mean"] - on["mean"], "per_seed": {s: per[s][1] for s in ss},
                                 "seeds_same": same, "provisional": pv}
    else:
        print("  Q2 READING: NOT DETERMINED (a cell has no complete mean)")
        out["readings"]["Q2"] = {"label": "NOT DETERMINED", "provisional": True}
    family("Q2-other", "what the two new prior cells do elsewhere (descriptive): all6-minus-clock - all6, only-clock - none@6e-4",
           [("all6-minus-clock@6e-4", "all6-lam1@6e-4", d) for d in DOMS + ("s4_hard", "bpb") if d != "clock"]
           + [("only-clock@6e-4", "none@6e-4", d) for d in DOMS + ("bpb",) if d != "clock"])

    # ---- Q3
    print(f"\n== Q3 (none at higher lrs: S4 / dim at 190M) {LABEL}")
    ans, best = {}, {}
    for mt in ("s4", "dim"):
        labels = {}
        for c in NONE_GRID:
            g = gates.get((c, mt))
            labels[c] = floor_label(g) if g else "n/a (no run)"
            ci = g["seed_ci"] if g else [None, None]
            print(f"    {mt:4s} {c:14s} {fmt(mean_of(c, mt))} seed CI [{fmt(ci[0], 5)},{fmt(ci[1], 5)}] vs floor "
                  f"{(g['floor'] if g else float('nan')):.1f}: {g['verdict'] if g else '-'} -> {labels[c]}")
        ans[mt] = floor_answer(labels)
        print(f"  Q3 ANSWER {mt}: {ans[mt]}" + (" PROVISIONAL" if prov(*NONE_GRID) else ""))
        out["readings"][f"Q3_{mt}"] = {"labels": labels, "answer": ans[mt], "provisional": prov(*NONE_GRID)}
    for c in NONE_GRID:
        hv = out["cells"].get(f"{c}|s4_hard", {})
        cs = cells.get(c, {})
        k = sum(R6.s4_hard_metrics(cs[s]["r"])["learned"] for s in sorted(cs))
        print(f"    s4_hard {c:14s} area {fmt(hv.get('mean'))} [{' '.join(fmt(v, 5) for v in hv.get('values', []))}] learned {k}/{len(cs)}")
    for mt in ("s4", "s4_hard", "dim"):
        inval = {c for c in NONE_GRID if (gates.get((c, "s4" if mt == "s4_hard" else mt)) or {}).get("verdict", "").startswith("INVALID")}
        b = best_tested({c: mean_of(c, mt) for c in NONE_GRID}, inval)
        best[mt] = b
        print(f"  best tested none on {mt}: {b} (highest mean over {', '.join(NONE_GRID)}; INVALID excluded {sorted(inval) or '-'}; "
              "chosen on these seeds 0-3: optimistic for none)")
    out["readings"]["Q3_best"] = dict(best, provisional=prov(*NONE_GRID))
    s4_floor = lambda c: (gates.get((c, "s4")) or {}).get("verdict") == "FLOOR"
    items = [(f"all6-lam1@{lr}", best[mt], mt) for lr in ("6e-4", "1.2e-3") for mt in ("s4", "s4_hard", "dim") if best[mt]]
    family("Q3", "all6 - the best tested none per metric (s4 / s4_hard one-sided greater iff that none's s4 gate is FLOOR, "
           "A5's F1b rule; dim two-sided)", items,
           side_of=lambda a, b, m: "greater" if m in ("s4", "s4_hard") and s4_floor(b) else "two", also=NONE_GRID)
    status = "PROVISIONAL" if (opt.allow_partial or miss_txt or excluded) else "complete"
    print(f"\n== rb6_190m_ext read-out status: {status}" + (f" (missing: {'; '.join(miss_txt)})" if miss_txt else "") + f" {LABEL}")
    out["status"] = status
    if opt.json:
        Path(opt.json).write_text(json.dumps(out, indent=1, default=lambda o: None))


if __name__ == "__main__":
    main()
