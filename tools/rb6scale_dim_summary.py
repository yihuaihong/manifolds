"""Selection and read-out of the 190M dim closing batch rb6_190m_dim (Torch, llmadd_torch/runs/rb6_190m_dim), PRE-REGISTERED
in llmadd_torch/runs/rb6_190m_dim/AUDIT.md before any run of the batch was submitted. A new thin wrapper: the metric
definitions, gate, paired test, gradient split, clip statistics and integrity checks are tools/rainbow6_summary.py's, the
Torch run identity is tools/rb6scale_summary.py's torch_identity, the lr choice is tools/rb6scale_select.select (second
path: readout_rules.best_lr), the bracket readout_rules.bracket and the reading readout_rules.survives (pt2main's rule 2);
none of these files is changed (the rb6_190m and rb6_190m_ext read-outs stay byte-identical). Only 190M runs are read: no
size comparison.
Cells (190M; labelled arm@lr by the actual lr):
  none@{6e-4, 4.8e-3, 9.6e-3, 1.92e-2, 3.84e-2}, seeds 0-3  (runs/rb6_190m/out, runs/rb6_190m_ext/out; 3.84e-2 = Stage D1)
  none@<best_dim>, seeds 4-7                                (Stage D2, runs/rb6_190m_dim/out)
  all6-lam1@6e-4, seeds 0-3 (runs/rb6_190m/out) and 4-7     (Stage D2)
Modes:
  select   the pre-registered choice of none's dim lr on seeds 0-3: the highest mean dim strict final3 over the five lrs;
           every lr within 2 points of the top is tied -> the lower mean in-training ("online") bpb (the last eval's
           val_bits_per_byte; a non-finite value is the worst) -> the lower lr. Bracket: OPEN (upper edge) when the pick is
           3.84e-2 and its mean is more than 2 points above 1.92e-2's (= readout_rules.bracket 'open upper'; asserted).
           A diverged run is data (its values count; its bpb is the worst). Complete = all 20 runs present, finished,
           identity clean (check.json PASS) and rainbow6_summary.check passes; otherwise status 'incomplete', NO pick
           (the selection job then submits nothing). Writes --json (never over an existing file).
  readout  the selection recomputed (with --picks: asserted equal to picks.json, else the read-out refuses); per cell the
           values per seed; the primary row dim all6-lam1@6e-4 - none@best_dim on seeds 0-3 and on the confirm seeds 4-7,
           read by readout_rules.survives (pt2main rule 2: mean > 0, >= 3/4 wins, paired t p < .05 on seeds 0-3, the
           confirm seeds 4-7 the same sign with >= 3/4 wins) -> SURVIVES / NOT SIGNIFICANT / REVERSED / DIRECTION UNSTABLE
           / OPEN (bracket open: not best-vs-best) / INCOMPLETE; the same contrast on s4 and numbers (descriptive, no
           reading); the gradient split and the clip statistics of every all6-lam1@6e-4 seed; the lm curve of the
           3.84e-2 seeds (divergence / soft blow-up). A missing / excluded / unfinished run: strict mode refuses,
           --allow_partial prints PROVISIONAL.
    python tools/rb6scale_dim_summary.py select  <roots> --json picks.json
    python tools/rb6scale_dim_summary.py readout <roots> [--picks picks.json] [--allow_partial] [--json out.json]
(<roots> = runs/rb6_190m/out runs/rb6_190m_ext/out runs/rb6_190m_dim/out; other cells there are listed as IGNORED.)
"""
import argparse
import hashlib
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
import rb6scale_select as SEL     # noqa: E402
import readout_rules as RR        # noqa: E402

SIZE = "190m"
NONE_GRID = ("6e-4", "4.8e-3", "9.6e-3", "1.92e-2", "3.84e-2")   # as spelled in the tags, ascending
NEW_LR = "3.84e-2"                                             # Stage D1
ALL6 = ("all6-lam1", "6e-4")
SEL_SEEDS, CONF_SEEDS = (0, 1, 2, 3), (4, 5, 6, 7)
TIE = 2.0
assert SEL.TIE == TIE == RR.TIE and list(NONE_GRID) == sorted(NONE_GRID, key=float)
ARM_RE = re.compile(r"^(none|all6-lam1)-lr(.+)$")
EXPECT_DIVERGE = ("none-lr1.92e-2", "none-lr3.84e-2")   # the high-lr none cells: a non-finite value is data (A6 rule)
TOOL_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
RULE = ("none's dim lr: the highest mean dim strict final3 over seeds 0-3 among none@{6e-4, 4.8e-3, 9.6e-3, 1.92e-2, "
        "3.84e-2}; lrs within 2 points of the top are tied -> the lower mean in-training bpb (non-finite = worst) -> the "
        "lower lr; OPEN (upper edge) iff the pick is 3.84e-2 and beats 1.92e-2 by > 2 points")
NOTES = [
    "use: this reading only decides whether the dim cell may go back into Table 1; the user has decided NOT to put it there "
    "for now; no size comparison (only 190M runs are read)",
    "selection: none's dim lr is chosen on seeds 0-3, the same seeds as the primary row on 0-3 (a winner's curse in none's "
    "favour, i.e. against all6); the confirm seeds 4-7 are selection-free",
    "lrs: all6-lam1's 6e-4 and none's 6e-4 / 4.8e-3 come from pt2s190 (numbers area, seeds 0-3); 9.6e-3 / 1.92e-2 / 3.84e-2 "
    "are x2 steps; all6-lam1's lr is not tuned for dim (1.2e-3 gave 76.0 vs 89.4 at 6e-4, rb6_190m_ext)",
    "S4 (descriptive row): already known from rb6_190m_ext, none@1.92e-2 is above all6-lam1@6e-4 on S4 (60.5 vs 44.5); the "
    "none lr here is chosen on dim, not on S4",
]
OPEN_NOTE = ("bracket OPEN at the upper edge: none@3.84e-2 is the pick and beats 1.92e-2 by > 2 points, so the comparison is "
             "with the best TESTED none, not best-vs-best; no further lr is added automatically (pre-registered)")


def fmt(v, w=6, p=1):
    return S.fmt(v, w, p)


def load(roots):
    """-> runs {(arm, lr, seed): x} of the design's arms (none / all6-lam1 at 190M, K 2, lr spelled as in the grid,
    seeds 0-7), ignored [labels]; asserts one run per key."""
    runs, ignored = {}, []
    for x in R6.load(roots):
        m, am = R6.TAG.match(x["tag"]), ARM_RE.match(x["arm"])
        lab = f"{x['tag']} s{x['seed']} ({Path(x['path']).parent.parent.name})"
        ok = m["size"] == SIZE and am and x["k"] == 2 and x["seed"] in SEL_SEEDS + CONF_SEEDS and (
            (am[1] == "none" and am[2] in NONE_GRID) or (am[1], am[2]) == ALL6)
        if not ok:
            ignored.append(lab)
            continue
        key = (am[1], am[2], x["seed"])
        assert key not in runs, ("two runs of one cell", key, lab)
        x["cell"] = f"{am[1]}@{am[2]}"
        runs[key] = x
    return runs, ignored


def problems_of(runs, keys):
    """identity (torch_identity incl. check.json PASS) and finished, per key; missing keys. -> {key: [problems]}"""
    out = {}
    for k in keys:
        if k not in runs:
            out[k] = ["missing"]
            continue
        x = runs[k]
        bad = S.torch_identity(x, SIZE, k[0], k[1])
        if x["r"]["curve"][-1]["step"] != x["r"]["steps"] - 1:
            bad.append(f"unfinished (last eval {x['r']['curve'][-1]['step']} of {x['r']['steps']})")
        if bad:
            out[k] = bad
    return out


def integrity(xs, partial):
    """rainbow6_summary.check over the runs (a diverged high-lr none run is data, A6 rule; any other non-finite value
    stops); -> warnings"""
    R6.A6_ARMS = R6.A6_ARMS + tuple(a for a in EXPECT_DIVERGE if a not in R6.A6_ARMS)
    return R6.check(xs, partial) if xs else []


def dim_seen(x):
    return R6.metrics(x["r"], "dim_seen")["final3"]


def choose(dim, bpb):
    """dim / bpb: {lr: [per-seed values over seeds 0-3]} for the five none lrs -> the pre-registered pick (rb6scale_select
    .select; second path readout_rules.best_lr with in-training bpb, asserted equal), the tied lrs, the means and the
    bracket (OPEN iff the pick is 3.84e-2 and beats 1.92e-2 by > 2; = readout_rules.bracket 'open upper', asserted)."""
    grid = list(NONE_GRID)
    pick, tied, m, b, _ = SEL.select(dim, bpb, grid)
    rr = RR.best_lr({lr: {"area": dim[lr], "off": [None] * len(dim[lr]), "intr": bpb[lr]} for lr in grid}, near_rule="within")
    assert rr["best"] == pick and rr["near"] == tied and not rr["offline"], ("two selection paths differ", pick, rr)
    open_ = pick == NEW_LR and m[NEW_LR] > m["1.92e-2"] + TIE
    br = RR.bracket(sorted(grid, key=float), m, pick)
    assert (br == ("open", "upper")) == open_, ("bracket paths differ", br, open_)
    lower_open = br == ("open", "lower")
    state = (OPEN_NOTE if open_ else "OPEN at the lower edge (pick 6e-4 beats 4.8e-3 by > 2): not best-vs-best" if lower_open
             else "closed" + (f" (note: the dim argmax is at the {br[1]} edge by <= 2 points)" if br and br[0] == "note" else ""))
    return dict(pick=pick, tied=tied, mean=m, mean_bpb=b, open=bool(open_ or lower_open), bracket=state)


def select_table(runs):
    """-> (status, problems, table, choice or None) of the selection over the 20 runs none@grid x seeds 0-3"""
    keys = [("none", lr, s) for lr in NONE_GRID for s in SEL_SEEDS]
    probs = problems_of(runs, keys)
    problems = [f"none@{k[1]} s{k[2]}: {p}" for k, ps in sorted(probs.items(), key=lambda kv: (float(kv[0][1]), kv[0][2]))
                for p in ps]
    table = {}
    for lr in NONE_GRID:
        xs = {s: runs[("none", lr, s)] for s in SEL_SEEDS if ("none", lr, s) in runs}
        table[lr] = {"seeds": sorted(xs), "dim": [S.value(xs[s], "dim") for s in sorted(xs)],
                     "bpb": [R6.final_bpb(xs[s]["r"]) for s in sorted(xs)],
                     "seen_dim": [dim_seen(xs[s]) for s in sorted(xs)],
                     "diverged": [s for s in sorted(xs) if R6.diverged_at(xs[s]["r"]) is not None]}
        table[lr]["invalid_A5"] = bool(table[lr]["seen_dim"]) and min(table[lr]["seen_dim"]) < R6.SEEN_MIN
    if problems:
        return "incomplete", problems, table, None, []
    warn = integrity([runs[k] for k in keys], False)
    ch = choose({lr: table[lr]["dim"] for lr in NONE_GRID}, {lr: table[lr]["bpb"] for lr in NONE_GRID})
    return "picked", [], table, ch, warn


def print_selection(table, ch):
    print(f"  selection rule (pre-registered): {RULE}")
    for lr in NONE_GRID:
        t = table[lr]
        mean = float(np.mean(t["dim"])) if len(t["dim"]) == len(SEL_SEEDS) else None
        mb = float(np.mean([SEL.worst(v) for v in t["bpb"]])) if len(t["bpb"]) == len(SEL_SEEDS) else None
        print(f"    none@{lr:8s} dim strict final3 [{' '.join(fmt(v, 5) for v in t['dim'])}] mean {fmt(mean, 5)} | "
              f"in-training bpb [{' '.join(fmt(v, 6, 4) for v in t['bpb'])}] mean {fmt(mb, 6, 4)} | dim seen "
              f"[{' '.join(fmt(v, 5) for v in t['seen_dim'])}]" + (" A5-INVALID (seen < 80; reported, not excluded)"
                                                                   if t["invalid_A5"] else "")
              + (f" | DIVERGED seeds {t['diverged']} (data)" if t["diverged"] else ""))
    if ch:
        print(f"  -> best_dim none@{ch['pick']} (top mean {max(ch['mean'].values()):.1f}; tied within {TIE:g}: "
              f"{['none@' + lr for lr in ch['tied']]}; their mean in-training bpb "
              f"{ {lr: round(v, 4) for lr, v in ch['mean_bpb'].items() if lr in ch['tied']} }); bracket: {ch['bracket']}")


def cmd_select(opt):
    out_p = Path(opt.json)
    assert not out_p.exists(), f"{out_p} exists: the pre-registered selection is never overwritten"
    print(f"rb6_190m_dim selection (tool sha256[:12] {TOOL_SHA}); roots {opt.roots}")
    runs, ignored = load(opt.roots)
    print(f"  {len(runs)} design runs read, {len(ignored)} other runs ignored")
    status, problems, table, ch, warn = select_table(runs)
    for p in problems:
        print("PROBLEM " + p)
    for w in warn:
        print("WARNING " + w)
    print_selection(table, ch)
    out = {"batch": "rb6_190m_dim", "size": SIZE, "tool": "rb6scale_dim_summary.py", "tool_sha": TOOL_SHA, "status": status,
           "rule": RULE, "grid": list(NONE_GRID), "seeds": list(SEL_SEEDS), "problems": problems, "warnings": warn,
           "table": table, "pick": {}, "open": None, "bracket": None}
    if ch:
        out.update(pick={"none_dim": ch["pick"]}, open=ch["open"], bracket=ch["bracket"], tied=ch["tied"], mean=ch["mean"],
                   mean_bpb=ch["mean_bpb"])
        print(f"PICK none_dim {ch['pick']}; open {ch['open']}")
    else:
        print("INCOMPLETE: no pick (Stage D2 is not submitted; the problems above go to the user)")
    out_p.write_text(json.dumps(out, indent=1, default=lambda o: None))


def verdict(ch, sv, main, conf_complete):
    """the pre-registered reading of the primary row (AUDIT.md 'Reading'): ch = the selection (None = incomplete), sv =
    readout_rules.survives on seeds 0-3 (+ 4-7 when complete), main = R6.paired on seeds 0-3."""
    if ch is None:
        return "INCOMPLETE (data missing: no selection on seeds 0-3, no reading)"
    if ch["open"]:
        return f"OPEN (bracket open: all6 - the best TESTED none, not best-vs-best; vs it: {sv['verdict']})"
    if not sv["main_ok"]:
        g, p, w, n = sv["main"]
        if g <= 0:
            return ("REVERSED (none > all6 on seeds 0-3, paired t p < .05)" if p < .05 else
                    "NOT SIGNIFICANT (mean <= 0 on seeds 0-3)")
        if w < 0.75 * n:
            return f"DIRECTION UNSTABLE (only {w}/{n} seed wins on seeds 0-3)"
        return f"NOT SIGNIFICANT (paired t p {p:.3f} >= .05 on seeds 0-3)"
    if not conf_complete:
        return "INCOMPLETE (passes on seeds 0-3; the confirm seeds 4-7 are missing: PROVISIONAL)"
    if sv["confirm_ok"]:
        return "SURVIVES (seeds 0-3: mean > 0, >= 3/4 wins, p < .05; confirm seeds 4-7: same sign, >= 3/4 wins)"
    c = sv["confirm"]
    return f"DIRECTION UNSTABLE (the confirm seeds 4-7 disagree: mean {c[0]:+.1f}, wins {c[2]}/{c[3]})"


def cmd_readout(opt):
    print(f"rb6_190m_dim read-out (190M; tool sha256[:12] {TOOL_SHA})")
    print(f"  flags used: mode readout; allow_partial {opt.allow_partial}; picks {opt.picks or '- (selection recomputed only)'};"
          f" json {opt.json or '-'}; roots {opt.roots}")
    for n_ in NOTES:
        print(f"  NOTE (pre-registered): {n_}")
    runs, ignored = load(opt.roots)
    status, sel_probs, table, ch, _ = select_table(runs)
    pk = None
    if opt.picks:
        pk = json.loads(Path(opt.picks).read_text())
        mine = ch["pick"] if ch else None
        theirs = pk.get("pick", {}).get("none_dim")
        assert pk.get("status") == status and theirs == mine and (ch is None or pk.get("open") == ch["open"]), \
            (f"--picks {opt.picks} disagrees with the recomputed selection: status {pk.get('status')} vs {status}, pick "
             f"{theirs} vs {mine}: read-out REFUSED")
        print(f"  --picks {opt.picks}: status {pk['status']}, none_dim {theirs}, open {pk.get('open')}, written by tool "
              f"{pk.get('tool_sha')} = the recomputed selection (agrees)")
    best = ch["pick"] if ch else None
    conf_cells = ([("none", best)] if best else []) + [ALL6]
    keys = [("none", lr, s) for lr in NONE_GRID for s in SEL_SEEDS] + [(*ALL6, s) for s in SEL_SEEDS] + \
        [(a, lr, s) for a, lr in conf_cells for s in CONF_SEEDS]
    stray = [k for k in runs if k not in keys]
    for k in sorted(stray, key=lambda k: (k[0], float(k[1]), k[2])):
        ignored.append(f"{k[0]}@{k[1]} s{k[2]} (not a design cell: seeds 4-7 only for none@best_dim and all6-lam1@6e-4)")
    probs = problems_of(runs, keys)
    excluded = {k: p for k, p in probs.items() if p != ["missing"] and not (len(p) == 1 and p[0].startswith("unfinished"))}
    unfinished = [k for k, p in probs.items() if any(q.startswith("unfinished") for q in p)]
    missing = [k for k, p in probs.items() if p == ["missing"]] + [k for k in excluded]
    if best is None:
        missing += [("none", "<best_dim>", s) for s in CONF_SEEDS]
    use = {k: runs[k] for k in keys if k in runs and k not in excluded}
    print(f"  {len(use)} runs in the design (none@5 lrs x seeds 0-3 = 20, all6-lam1@6e-4 x seeds 0-7 = 8, none@best_dim x "
          f"seeds 4-7 = 4: 32), {len(ignored)} ignored, {len(excluded)} excluded, {len(missing)} missing; mode "
          + ("PARTIAL (--allow_partial: every verdict PROVISIONAL)" if opt.allow_partial else "strict"))
    for lab in ignored:
        print(f"  IGNORED {lab}")
    for k, p in excluded.items():
        print(f"  EXCLUDED {k[0]}@{k[1]} s{k[2]}: " + "; ".join(p))
    miss_txt = [f"{k[0]}@{k[1]} s{k[2]}" for k in missing] + [f"{k[0]}@{k[1]} s{k[2]} (unfinished)" for k in unfinished]
    if miss_txt and not opt.allow_partial:
        raise AssertionError("rb6_190m_dim read-out REFUSED (strict): missing / excluded / unfinished " + "; ".join(miss_txt)
                             + " (--allow_partial to look anyway: PROVISIONAL)")
    warn = integrity(list(use.values()), opt.allow_partial)
    for w in warn:
        print(f"  WARNING {w}")
    print("  data identity: per seed every run has the same data_sha at every eval and the same step-0 lm (rainbow6_summary.check)")
    use = {k: x for k, x in use.items() if x["r"]["curve"][-1]["step"] == x["r"]["steps"] - 1}   # unfinished = missing
    out = {"tool_sha": TOOL_SHA, "selection": {"status": status, "problems": sel_probs, "table": table,
           "choice": ch}, "picks_file": opt.picks, "cells": {}, "rows": {}, "grad": {}, "ignored": ignored,
           "excluded": [[f"{k[0]}@{k[1]} s{k[2]}", p] for k, p in excluded.items()], "missing": miss_txt, "warnings": warn,
           "notes": NOTES}

    print(f"\n== selection of none's dim lr (seeds 0-3): status {status}")
    for p in sel_probs:
        print(f"  PROBLEM {p}")
    print_selection(table, ch)

    print("\n== per cell: dim strict final3, s4 fresh non-id final3, numbers area, offline bpb x1e3, dim seen; mean [per seed]")
    cells = [(f"none@{lr}", "none", lr, SEL_SEEDS) for lr in NONE_GRID] + [("all6-lam1@6e-4", *ALL6, SEL_SEEDS)]
    cells += ([(f"none@{best} (confirm)", "none", best, CONF_SEEDS)] if best else []) + \
        [("all6-lam1@6e-4 (confirm)", *ALL6, CONF_SEEDS)]
    for lab, a, lr, seeds in cells:
        xs = {s: use[(a, lr, s)] for s in seeds if (a, lr, s) in use}
        ss = sorted(xs)
        print(f"  {lab:26s} seeds {list(seeds)} " + ("n=4" if len(ss) == 4 else f"PROVISIONAL (n={len(ss)} of 4)"))
        for mt in ("dim", "s4", "numbers", "bpb", "dim_seen"):
            vals = [dim_seen(xs[s]) if mt == "dim_seen" else S.value(xs[s], mt) for s in ss]
            ok = [v for v in vals if v is not None and math.isfinite(v)]
            mean = float(np.mean(ok)) if vals and len(ok) == len(vals) else None
            extra = ""
            if mt == "dim" and len(ss) >= 2:
                g = R6.gate_key([xs[s]["r"] for s in ss], "dim", R6.READ["dim"][0])
                extra = f" | seed CI [{fmt(g['seed_ci'][0], 5)},{fmt(g['seed_ci'][1], 5)}] vs floor {g['floor']:.1f} -> {g['verdict']}"
            if mt == "bpb" and any(R6.diverged_at(xs[s]["r"]) is not None for s in ss):
                extra = " | DIVERGED seed(s): " + ",".join(f"s{s}" for s in ss if R6.diverged_at(xs[s]["r"]) is not None)
            print(f"    {mt:8s} {fmt(mean)} [{' '.join(fmt(v, 5) for v in vals)}]{extra}")
            out["cells"][f"{lab}|{mt}"] = {"seeds": ss, "values": vals, "mean": mean}

    print(f"\n== none@{NEW_LR} lm curve (training lm at the evals: step 0, 1000, 5000, 9000, 13000, last; val_loss last; min; "
          "max rise above the running minimum; DIVERGED step)")
    out["lm_curve"] = {}
    for s in SEL_SEEDS:
        x = use.get(("none", NEW_LR, s))
        if x is None:
            print(f"    s{s} -")
            continue
        c_ = x["r"]["curve"]
        at = lambda st: next((e["lm"] for e in c_ if e["step"] >= st), None)
        lms = [(e["step"], e["lm"]) for e in c_ if e["lm"] is not None and math.isfinite(e["lm"])]
        mn = min(lms, key=lambda z: z[1]) if lms else (None, None)
        run_min, rise, rise_st = float("inf"), 0.0, None
        for st, v in lms:
            run_min = min(run_min, v)
            if v - run_min > rise:
                rise, rise_st = v - run_min, st
        div = R6.diverged_at(x["r"])
        out["lm_curve"][s] = dict(last=c_[-1]["lm"], val_last=c_[-1]["val_loss"], min=mn, max_rise=rise, diverged_at=div)
        print(f"    s{s} " + " ".join(fmt(at(st), 6, 3) for st in (0, 1000, 5000, 9000, 13000)) + f" {fmt(c_[-1]['lm'], 6, 3)}; "
              f"val_loss last {fmt(c_[-1]['val_loss'], 6, 3)}; min {fmt(mn[1], 6, 3)} ({mn[0]}); max rise {rise:.3f} ({rise_st})"
              + (" <- inspect: a rise > 0.5 nats above the running minimum (possible soft blow-up)" if rise > 0.5 else "")
              + (f"; DIVERGED at step {div} (data: its values count, its bpb is the worst / NOT TESTED)" if div is not None
                 else "; finite throughout"))

    def row(mt, seeds):
        if best is None:
            return None
        a = {s: use[(*ALL6, s)] for s in seeds if (*ALL6, s) in use}
        b = {s: use[("none", best, s)] for s in seeds if ("none", best, s) in use}
        ss = [s for s in seeds if s in a and s in b]
        va = [S.value(a[s], mt) for s in ss]
        vb = [S.value(b[s], mt) for s in ss]
        if len(ss) < 2 or any(v is None or not math.isfinite(v) for v in va + vb):
            return dict(seeds=ss, a=va, b=vb, note=f"NOT TESTED ({len(ss)} paired seeds or a non-finite value)")
        return dict(seeds=ss, a=va, b=vb, complete=len(ss) == len(seeds), **R6.paired([p - q for p, q in zip(va, vb)], "two"))

    def show(lab, r_):
        if r_ is None:
            print(f"    {lab}: NOT DETERMINED (no selection)")
            return
        if r_.get("note"):
            print(f"    {lab}: {r_['note']}")
            return
        ci = r_.get("ci") or [None, None]
        print(f"    {lab}: {fmt(r_['mean'], 7)} [{' '.join(fmt(v, 6) for v in r_['diffs'])}] wins {r_['wins']}/{r_['n']} t-p "
              f"{fmt(r_['p_t'], 5, 3)} (two-sided) sf-p {fmt(r_['p_sf'], 5, 3)} 95% CI [{fmt(ci[0], 6)},{fmt(ci[1], 6)}]"
              + ("" if r_["complete"] else f" PROVISIONAL (n={r_['n']} of 4)"))

    print(f"\n== PRIMARY (pre-registered): dim strict final3, all6-lam1@6e-4 - none@{best or '<best_dim>'} (none's lr chosen "
          "on dim, seeds 0-3)")
    r03, r47 = row("dim", SEL_SEEDS), row("dim", CONF_SEEDS)
    show("seeds 0-3 (selection seeds)", r03)
    show("seeds 4-7 (confirm, selection-free)", r47)
    sv = None
    if r03 and not r03.get("note") and r03["complete"]:
        conf_complete = bool(r47 and not r47.get("note") and r47["complete"])
        sv = RR.survives(r03["a"], r03["b"], *((r47["a"], r47["b"]) if conf_complete else ()))
        assert abs(sv["main"][1] - r03["p_t"]) < 1e-6 or (math.isnan(sv["main"][1]) and r03.get("degenerate")), \
            ("paired t: readout_rules vs rainbow6_summary", sv["main"][1], r03["p_t"])
        lab = verdict(ch, sv, r03, conf_complete)
    else:
        lab = verdict(ch, None, None, False) if ch is None else "INCOMPLETE (seeds 0-3 of the primary row are not complete: PROVISIONAL)"
    prov = bool(miss_txt) or opt.allow_partial
    print(f"  READING: {lab}" + (" [PROVISIONAL]" if prov else ""))
    if ch and ch["open"]:
        print(f"    NOTE {OPEN_NOTE}")
    out["rows"]["dim"] = {"0-3": r03, "4-7": r47}
    out["reading"] = {"label": lab, "provisional": prov, "survives": sv and {k: v for k, v in sv.items()}}

    print(f"\n== the same contrast on s4 and numbers (descriptive, no reading; none's lr chosen on dim, not on these)")
    for mt in ("s4", "numbers"):
        for nm, seeds in (("0-3", SEL_SEEDS), ("4-7", CONF_SEEDS)):
            r_ = row(mt, seeds)
            show(f"{mt:8s} seeds {nm}", r_)
            out["rows"][f"{mt}|{nm}"] = r_
    print("    NOTE s4: none@1.92e-2 is already known to be above all6-lam1@6e-4 on s4 at seeds 0-3 (60.5 vs 44.5, rb6_190m_ext)")

    print("\n== gradient split and clip of the prior cell all6-lam1@6e-4 (window medians at the on-evals; clip = 1 / max(1, T), "
          "frac clipped = share of on-evals with T > 1)")
    flag = []
    for s in SEL_SEEDS + CONF_SEEDS:
        x = use.get((*ALL6, s))
        if x is None:
            print(f"    s{s} -")
            continue
        g, c = R6.grad_split(x["r"]), R6.clip_stats(x["r"])
        out["grad"][s] = {"split": g, "clip": c}
        if g is None:
            print(f"    s{s} NOT LOGGED")
            continue
        flag += [s] if g["prior_over_lm"] > 1 else []
        print(f"    s{s} LM share {g['lm_share']:.2f} prior/LM {g['prior_over_lm']:.2f} prior>LM {g['frac_prior_gt_lm']:.2f} "
              f"clip {g['clip']:.2f}" + (f" frac clipped {c['frac_clipped']:.2f} (n {c['n']})" if c else " clip stats NOT LOGGED"))
    print("    " + (f"FLAG: median prior / LM > 1 at seeds {flag} (the prior is auxiliary)" if flag else
                    "no seed with a median prior / LM > 1 (FLAG not raised)"))
    st = "PROVISIONAL" if prov else "complete"
    print(f"\n== rb6_190m_dim read-out status: {st}" + (f" (missing: {'; '.join(miss_txt)})" if miss_txt else ""))
    out["status"] = st
    if opt.json:
        Path(opt.json).write_text(json.dumps(out, indent=1, default=lambda o: None))


def main():
    ap = argparse.ArgumentParser(description="rb6_190m_dim selection / read-out; see the module docstring")
    ap.add_argument("mode", choices=("select", "readout"))
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--picks", default=None, help="readout: the selection job's picks.json (asserted = the recomputed one)")
    ap.add_argument("--allow_partial", action="store_true", help="readout: look at an incomplete batch (PROVISIONAL)")
    opt = ap.parse_args()
    if opt.mode == "select":
        assert opt.json and not opt.picks and not opt.allow_partial, "select: --json only"
        cmd_select(opt)
    else:
        cmd_readout(opt)


if __name__ == "__main__":
    main()
