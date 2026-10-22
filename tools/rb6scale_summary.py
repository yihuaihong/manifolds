"""Model-size ladder of the six-prior rainbow pretraining (rb6, F88 at 53M): the descriptive per-size read-out of the Torch
runs (190M: llmadd_torch/runs/rb6_190m, 505M: runs/rb6_505m) next to the 53M Burst cells, PRE-REGISTERED in
llmadd_torch/runs/rb6_190m/AUDIT.md (and runs/rb6_505m/AUDIT.md). A thin wrapper around tools/rainbow6_summary.py: its
metric definitions (A5 READ), integrity check, strict dim set, s4_hard (A8.3), seed-level gate, paired test and BH are
used as they are; rainbow6_summary's own output is unchanged (this file does not edit it).

Cells per size (roles): none@low = none at the prior arms' lr, none@best = none at its pt2 numbers-best lr, all6 = the six
priors (lambda 1 each), only-numbers = the numbers prior alone; seeds 0-3.
  53m   none@1e-3 (rb6pb-53m-2pct-k2-none), none@8e-3 (rb6mb-...-none-lr8e-3), all6-lam1@1e-3 (rb6pb-...-all6-lam1),
        only-numbers@1e-3 (rb6mb-...-only-numbers): the F88 cells (Burst; identity by rainbow6_summary.run_identity)
  190m  none@6e-4, none@4.8e-3, all6-lam1@6e-4, only-numbers@6e-4 (tags rb6tb-190m-2pct-k2-<arm>-lr<lr>; the pt2s190 lrs)
  505m  none@<all6's pick>, none@<none's pick>, all6-lam1@<all6's pick>, only-numbers@<all6's pick> (--picks: the JSON of
        tools/rb6scale_select.py over the 505M sweep)
Output: per size and cell the A5 READ metric per domain (mean over seeds 0-3, per-seed values in seed order, the seed-level
90% t interval and gate verdict), s4_hard area, offline bpb x1e3, seen, the gradient split; the sizes side by side (no
statistic across sizes: F87); per size the paired families S1 / S1b / SE / S2 / S3 / SH / SL, every line "on the
lr-selection seeds 0-3: descriptive, no confirmatory claim". A missing / excluded / unfinished run -> PROVISIONAL (n=k of 4)
on every line that needs it; without --allow_partial the read-out refuses. A diverged Torch run is data (A6 rule).
    python tools/rb6scale_summary.py <dirs with run_*.json ...> [--picks picks.json] [--allow_partial] [--json out.json]
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
import rainbow6_summary as R6   # noqa: E402
import make_rb6m_tasks as G      # noqa: E402

SEEDS = (0, 1, 2, 3)
ROLES = ("none_low", "none_best", "all6", "only_numbers")
ROLE_LAB = {"none_low": "none@low", "none_best": "none@best", "all6": "all6-lam1", "only_numbers": "only-numbers"}
ROLE_BASE = {"none_low": "none", "none_best": "none", "all6": "all6-lam1", "only_numbers": "only-numbers"}
DOMS = R6.DOM6   # numbers, months, weekdays, clock, s4, dim
DESIGN53 = {"none_low": ("none", "1e-3"), "none_best": ("none-lr8e-3", "8e-3"), "all6": ("all6-lam1", "1e-3"),
            "only_numbers": ("only-numbers", "1e-3")}   # 53M arm names (as in the F88 tags) and lrs
# the Torch sizes: model facts (the generator's, written again here as a second path) and their lrs per role
SIZE_FACTS = {"190m": dict(dims=(768, 12, 12, 3072), n_params=190297344, layers=[4, 8]),
              "505m": dict(dims=(1024, 24, 16, 4096), n_params=505437184, layers=[8, 16])}
TORCH_SHA = {"code_sha": "93e22bf4c4c0", "toy_struct.py": "d182ce548bcc"}
DESIGN = {"190m": {"none_low": "6e-4", "none_best": "4.8e-3", "all6": "6e-4", "only_numbers": "6e-4"}}
TORCH_ARM = re.compile(r"^(none|all6-lam1|only-numbers)-lr(.+)$")
LABEL = "[on the lr-selection seeds 0-3: descriptive, no confirmatory claim]"
FAMILIES = [   # (name, contrast (role a, role b), rows)
    ("S1", ("all6", "none_low"), ("numbers", "clock", "s4", "dim")),
    ("S1b", ("all6", "none_best"), ("numbers", "clock", "s4", "dim")),
    ("SE", None, (("all6", "none_low", "months"), ("all6", "none_low", "weekdays"), ("all6", "none_best", "months"),
                  ("all6", "none_best", "weekdays"))),
    ("S2", ("only_numbers", "none_low"), DOMS),
    ("S3", ("all6", "only_numbers"), DOMS),
    ("SH", None, (("all6", "none_low", "s4_hard"), ("all6", "none_best", "s4_hard"))),
    ("SL", None, (("all6", "none_low", "bpb"), ("only_numbers", "none_low", "bpb"))),
]


def fmt(v, w=6, p=1):
    return "-".rjust(w) if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:{w}.{p}f}"


def torch_identity(x, size, base, lr):
    """The run JSON of a Torch scale run against its intent (a second path next to the job's CHECK_PY): every trainer
    argument of make_rb6m_tasks.intent(base) with the size's model dims, lr and tag; the run facts; code / toy shas; the
    lambda per domain, the active / shuffled priors, the logged multiplier; check.json PASS. -> list of problems."""
    r, s = x["r"], x["seed"]
    F = SIZE_FACTS[size]
    I = G.intent(base, s)
    want = dict(I["args"], d_model=F["dims"][0], n_layers=F["dims"][1], n_heads=F["dims"][2], d_ff=F["dims"][3],
                lr=float(lr), tag=f"_{x['tag']}")
    bad = []
    dif = {k: [r["args"].get(k), v] for k, v in want.items() if r["args"].get(k) != v}
    if dif:
        bad.append(f"trainer args differ from the intent [run, intent]: {dif}")
    facts = {"code_sha": TORCH_SHA["code_sha"], "steps": I["steps"], "n_params": F["n_params"], "grad_ckpt_on": True,
             "geo_layers": F["layers"], "read_layers": F["layers"], "lm_chunk": 0, "eval_every_steps": I["eval_every_steps"],
             "lam_rule": I["lam_rule"]}
    fd = {k: [r.get(k), v] for k, v in facts.items() if r.get(k) != v}
    if fd:
        bad.append(f"run facts differ [run, intent]: {fd}")
    if r.get("dep_sha", {}).get("toy_struct.py") != TORCH_SHA["toy_struct.py"]:
        bad.append(f"toy_struct dep_sha {r.get('dep_sha')} != {TORCH_SHA['toy_struct.py']}")
    if r.get("lam_domains") != I["lam_domains"]:
        bad.append(f"lam_domains {r.get('lam_domains')} != {I['lam_domains']}")
    act = sorted(d for d in DOMS if r["domains"][d].get("prior"))
    shf = sorted(d for d in DOMS if r["domains"][d].get("shuf"))
    if act != sorted(I["active"]) or shf != sorted(I["shuf"]):
        bad.append(f"active / shuffled priors {act} / {shf} != {sorted(I['active'])} / {sorted(I['shuf'])}")
    on = [e for e in r["curve"] if e.get("prior_on")]
    div = R6.diverged_at(r)
    for d in I["active"]:
        rat = [e[f"gnorm_prior_{d}"] / e[f"gnorm_true_{d}"] for e in on if (div is None or e["step"] < div)
               and e.get(f"gnorm_prior_{d}") is not None and e.get(f"gnorm_true_{d}")]
        if not rat or any(abs(v / I["lam_domains"][d] - 1) >= 1e-3 for v in rat):
            bad.append(f"logged multiplier of {d} != lambda {I['lam_domains'][d]} (or never logged): "
                       f"{[round(min(rat), 5), round(max(rat), 5)] if rat else None}")
    f = Path(x["path"]).parent.parent / "check.json"
    try:
        cj = json.loads(f.read_text())
    except (OSError, ValueError):
        cj = None
        bad.append("no readable check.json next to the run (the job's intent checks did not run)")
    if cj is not None and not cj.get("pass"):
        bad.append(f"its check FAILED: {[c['check'][:80] for c in cj.get('checks', []) if not c.get('ok')]}")
    return bad


def assign(runs, design):
    """-> cells {size: {role: {seed: x}}}, ignored [labels], torch_keys {(size, role, seed)} (Torch runs)."""
    cells = defaultdict(lambda: defaultdict(dict))
    ignored, info = [], {}
    for x in runs:
        size = R6.TAG.match(x["tag"])["size"]
        x["size"] = size
        # the cell is the TAG's (arm, lr); the run's own arguments are checked against it by the identity checks
        # (rainbow6_summary.run_identity / torch_identity): a mislabelled run is EXCLUDED there, not ignored here
        role, base, lr = None, None, None
        if size == "53m":
            role = next((ro for ro, (a, l_) in DESIGN53.items() if a == x["arm"]), None)
            base, lr = (x["arm"], DESIGN53[role][1]) if role else (None, None)
        else:
            m = TORCH_ARM.match(x["arm"])
            if m and size in design:
                base, lr = m[1], m[2]
                role = next((ro for ro in ROLES if ROLE_BASE[ro] == base and float(design[size][ro]) == float(lr)), None)
        lab = f"{x['tag']} s{x['seed']} ({Path(x['path']).parent.parent.name})"
        if role is None or x["seed"] not in SEEDS or x["k"] != 2:
            ignored.append(lab)
            continue
        assert x["seed"] not in cells[size][role], f"two runs of {size} {role} s{x['seed']}"
        cells[size][role][x["seed"]] = x
        info[id(x)] = (size, role, ROLE_BASE[role] if size != "53m" else x["arm"], lr)
    return cells, ignored, info


def value(x, metric):
    """One run's read-out value: an A5 READ domain, 's4_hard' (A8.3 area), 'bpb' (offline x1e3, None if absent) or 'seen'."""
    if metric == "s4_hard":
        return R6.s4_hard_metrics(x["r"])["area"]
    return R6.mval(x, metric)


def main():
    ap = argparse.ArgumentParser(description="rb6 model-size ladder read-out; see the module docstring")
    ap.add_argument("roots", nargs="+", help="directories with the run JSONs (searched recursively): the Torch out/ dirs and "
                    "the 53M Burst results/<task> dirs")
    ap.add_argument("--picks", default=None, help="505M: the JSON of tools/rb6scale_select.py (status 'picked')")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--json", default=None)
    opt = ap.parse_args()
    design = dict(DESIGN)
    if opt.picks:
        pk = json.loads(Path(opt.picks).read_text())
        assert pk.get("status") == "picked" and pk["size"] == "505m", ("not a completed 505M selection", pk.get("status"))
        a_, n_ = pk["pick"]["all6-lam1"], pk["pick"]["none"]
        design["505m"] = {"none_low": a_, "none_best": n_, "all6": a_, "only_numbers": a_}
    for sz, d in design.items():
        assert float(d["none_low"]) != float(d["none_best"]), (sz, "none@low = none@best: one cell, not two")
    runs = R6.load(opt.roots)
    cells, ignored, info = assign(runs, design)
    sizes = [s for s in ("53m", "190m", "505m") if s in cells]
    print(f"rb6 scale summary: sizes {sizes}; {sum(len(v) for c in cells.values() for v in c.values())} runs in the design, "
          f"{len(ignored)} ignored; mode {'PARTIAL (--allow_partial: every verdict PROVISIONAL)' if opt.allow_partial else 'strict'}")
    print(f"  design lrs: 53m {dict((ROLE_LAB[k], v[1]) for k, v in DESIGN53.items())}; "
          + "; ".join(f"{s} {dict((ROLE_LAB[k], v) for k, v in design[s].items())}" for s in design))
    print("  NOTE (pre-registered): the Torch sizes' lrs are reused from the pt2 single-prior 2% setting (chosen on the numbers"
          " area over seeds 0-3); none is not tuned for S4 or any other domain; seeds 0-3 share the init and the numbers lines"
          " with the lr-selection runs (winner's curse on the numbers rows). No size-trend claim (F87).")
    for lab in ignored:
        print(f"  IGNORED (not a design cell / seed): {lab}")
    out = {"sizes": sizes, "design": {"53m": {k: v[1] for k, v in DESIGN53.items()}, **design}, "ignored": ignored,
           "excluded": [], "warnings": [], "cells": {}, "families": {}, "missing": {}}
    # ---- run identity (53M: rainbow6_summary.run_identity, the F88 read-out's; Torch: torch_identity)
    excluded = []
    if "53m" in cells:
        xs = [x for c in cells["53m"].values() for x in c.values()]
        kept, ident = R6.run_identity(xs, arms=("none", "none-lr8e-3", "all6-lam1", "only-numbers"))
        keep = {id(x) for x in kept}
        for lab, why in ident["excluded"]:
            excluded.append(("53m", lab, why))
        for n_ in ident["not_checked"]:
            print("  NOTE", n_)
        for ro in list(cells["53m"]):
            cells["53m"][ro] = {s: x for s, x in cells["53m"][ro].items() if id(x) in keep}
    for sz in sizes:
        if sz == "53m":
            continue
        for ro in list(cells[sz]):
            for s, x in list(cells[sz][ro].items()):
                _, _, base, lr = info[id(x)]
                bad = torch_identity(x, sz, base, lr)
                if bad:
                    excluded.append((sz, f"{ROLE_LAB[ro]} s{s} ({Path(x['path']).parent.parent.name})", bad))
                    del cells[sz][ro][s]
    for sz, lab, why in excluded:
        print(f"  EXCLUDED run {sz} {lab}: " + "; ".join(why))
        out["excluded"].append([sz, lab, why])
    # ---- integrity per size (rainbow6_summary.check; a Torch run's non-finite values are data: the A6 rule)
    torch_arms = tuple(sorted({x["arm"] for sz in sizes if sz != "53m" for c in cells[sz].values() for x in c.values()}))
    R6.A6_ARMS = R6.A6_ARMS + tuple(a for a in torch_arms if a not in R6.A6_ARMS)
    for sz in sizes:
        xs = [x for c in cells[sz].values() for x in c.values()]
        if xs:
            w = R6.check(xs, opt.allow_partial)
            for m_ in w:
                print(f"  WARNING {sz} {m_}")
                out["warnings"].append(f"{sz} {m_}")
    unfinished = {id(x) for sz in sizes for c in cells[sz].values() for x in c.values()
                  if x["r"]["curve"][-1]["step"] != x["r"]["steps"] - 1}
    for sz in sizes:   # unfinished = missing (A5.1 rule 1)
        for ro in cells[sz]:
            cells[sz][ro] = {s: x for s, x in cells[sz][ro].items() if id(x) not in unfinished}
    # data identity ACROSS sizes: per seed the same data_sha at every eval (the batches are size-independent)
    by_seed = defaultdict(list)
    for sz in sizes:
        for c in cells[sz].values():
            for s, x in c.items():
                by_seed[s].append((sz, x))
    for s, xs in sorted(by_seed.items()):
        ref = [e["data_sha"] for e in xs[0][1]["r"]["curve"]]
        for sz, x in xs[1:]:
            got = [e["data_sha"] for e in x["r"]["curve"]]
            n = min(len(ref), len(got))
            assert got[:n] == ref[:n], (s, sz, x["tag"], "the data differ across sizes (data_sha series)")
    print(f"  data identity: per seed every run of every size has the same data_sha at every eval ({len(by_seed)} seeds)")
    # ---- missing seeds
    missing = {sz: {ro: [s for s in SEEDS if s not in cells[sz].get(ro, {})] for ro in ROLES} for sz in sizes}
    out["missing"] = missing
    miss_txt = [f"{sz} {ROLE_LAB[ro]} {v}" for sz in sizes for ro, v in missing[sz].items() if v]
    if miss_txt and not opt.allow_partial:
        print("  missing seeds: " + "; ".join(miss_txt))
        raise AssertionError("rb6 scale read-out REFUSED (strict): missing / excluded / unfinished seeds " + "; ".join(miss_txt)
                             + " (--allow_partial to look anyway: PROVISIONAL)")
    prov = lambda sz, ros: [ro for ro in ros if missing[sz][ro]]

    # ---- per size: the cells (descriptive)
    METRICS = DOMS + ("s4_hard", "bpb", "seen")
    for sz in sizes:
        print(f"\n== {sz}: per cell, the A5 READ metric per domain (numbers area; months / weekdays tier-E final3; clock wrap; "
              "s4 fresh non-id; dim strict), s4_hard area, offline bpb x1e3, seen; mean over seeds [per seed, seed order] "
              "and the seed-level gate (90% t interval vs [chance + 5, 90])")
        for ro in ROLES:
            c = cells[sz].get(ro, {})
            seeds = sorted(c)
            lr = DESIGN53[ro][1] if sz == "53m" else design[sz][ro]
            tag = f"PROVISIONAL (n={len(seeds)} of 4)" if missing[sz][ro] else f"n=4"
            print(f"  {ROLE_LAB[ro]:13s} lr {lr:7s} {tag}")
            for mt in METRICS:
                vals = [value(c[s], mt) for s in seeds]
                ok = [v for v in vals if v is not None and math.isfinite(v)]
                mean = float(np.mean(ok)) if len(ok) == len(vals) and vals else None
                gate = ""
                g = None
                if mt in R6.EXTRA and seeds:
                    key = R6.READ[mt][0]
                    g = R6.gate_key([c[s]["r"] for s in seeds], mt, key)
                    ci = g["seed_ci"]
                    gate = (f" | seed CI [{fmt(ci[0], 5)},{fmt(ci[1], 5)}] vs floor {g['floor']:.1f} -> {g['verdict']}; "
                            f"heuristic {fmt(g['heuristic'], 4) if g.get('heuristic') is not None else 'n/a'}")
                print(f"    {mt:8s} {fmt(mean)} [{' '.join(fmt(v, 5) for v in vals)}]{gate}")
                out["cells"][f"{sz}|{ro}|{mt}"] = {"lr": lr, "seeds": seeds, "values": vals, "mean": mean,
                                                    "gate": g and {k: g[k] for k in ("verdict", "seed_ci", "floor", "chance")},
                                                    "provisional": bool(missing[sz][ro])}
            gs = [R6.grad_split(c[s]["r"]) for s in seeds]
            if any(gs):
                print("    grad split (window medians: LM share / prior over LM): "
                      + " ".join(f"s{s} {g['lm_share']:.2f}/{g['prior_over_lm']:.2f}" if g else f"s{s} -" for s, g in zip(seeds, gs)))
        if sz != "53m":
            h = {s: R6.s4_hard(x["r"]) for c in cells[sz].values() for s, x in c.items()}
            print("    s4_hard per seed (n / chance / Latin level, the same for every arm of a seed): "
                  + " ".join(f"s{s} {v['n']}/{v['chance']:.1f}/{v['level']:.1f}" for s, v in sorted(h.items())))

    # ---- sizes side by side (descriptive; no statistic across sizes)
    print("\n== sizes side by side (means over seeds 0-3; no statistic across sizes: F87)")
    print("  " + "cell".ljust(14) + "metric".ljust(9) + "".join(s.rjust(9) for s in sizes))
    for ro in ROLES:
        for mt in METRICS:
            row = [out["cells"].get(f"{sz}|{ro}|{mt}", {}).get("mean") for sz in sizes]
            pv = ["*" if missing[sz][ro] else " " for sz in sizes]
            print("  " + ROLE_LAB[ro].ljust(14) + mt.ljust(9) + "".join(f"{fmt(v, 8)}{p}" for v, p in zip(row, pv)))
    print("  (* = PROVISIONAL: fewer than 4 seeds)")

    # ---- paired families per size
    for sz in sizes:
        print(f"\n== {sz}: paired by seed, A5 style (paired t df n-1, exact sign-flip p, 95% CI, BH q .05 within the family) {LABEL}")
        s4_floor = {}
        for ro in ("none_low", "none_best"):
            gk = out["cells"].get(f"{sz}|{ro}|s4", {}).get("gate")
            s4_floor[ro] = bool(gk) and gk["verdict"] == "FLOOR"
        for fam, con, rows in FAMILIES:
            items = [(con[0], con[1], d) for d in rows] if con else list(rows)
            res = []
            for a, b, mt in items:
                ca, cb = cells[sz].get(a, {}), cells[sz].get(b, {})
                seeds = [s for s in SEEDS if s in ca and s in cb]
                va = [value(ca[s], mt) for s in seeds]
                vb = [value(cb[s], mt) for s in seeds]
                side = "greater" if (mt in ("s4", "s4_hard") and b in s4_floor and s4_floor[b]) else "two"
                bad = [s for s, p, q in zip(seeds, va, vb) if p is None or q is None or not (math.isfinite(p) and math.isfinite(q))]
                if bad:
                    res.append(dict(a=a, b=b, m=mt, side=side, seeds=seeds, p_t=None,
                                    note=f"NOT TESTED: no finite value at seeds {bad} (missing offline bpb or DIVERGED)"))
                    continue
                pr = R6.paired([p - q for p, q in zip(va, vb)], side)
                res.append(dict(a=a, b=b, m=mt, side=side, seeds=seeds, **pr))
            adj = R6.bh([r_.get("p_t") for r_ in res])
            for r_, q in zip(res, adj):
                r_["p_bh"] = q
            print(f"  -- {fam}" + (f" ({ROLE_LAB[con[0]]} - {ROLE_LAB[con[1]]})" if con else ""))
            for r_ in res:
                pv = prov(sz, (r_["a"], r_["b"]))
                lab = f"{ROLE_LAB[r_['a']]} - {ROLE_LAB[r_['b']]}"
                head = f"    {r_['m']:8s} {lab:27s}"
                if r_.get("note"):
                    print(head + " " + r_["note"])
                    continue
                ci = r_.get("ci") or [None, None]
                flag = " LM COST FLAG (mean > +4)" if r_["m"] == "bpb" and r_["mean"] is not None and r_["mean"] > 4 else ""
                pmark = f" PROVISIONAL (n={r_['n']} of 4)" if pv else ""
                sig = (" sig. at BH .05 (descriptive)" if r_["p_bh"] is not None and r_["p_bh"] < 0.05 else "") if not pv else ""
                print(f"{head} {fmt(r_['mean'], 7)} [{' '.join(fmt(v, 6) for v in r_['diffs'])}] wins {r_['wins']}/{r_['n']} "
                      f"t-p {fmt(r_['p_t'], 5, 3)} ({r_['side']}) sf-p {fmt(r_['p_sf'], 5, 3)} BH {fmt(r_['p_bh'], 5, 3)} "
                      f"95% CI [{fmt(ci[0], 6)},{fmt(ci[1], 6)}]{sig}{flag}{pmark}")
            out["families"][f"{sz}|{fam}"] = res
    status = "PROVISIONAL" if (opt.allow_partial or miss_txt or excluded) else "complete"
    print(f"\n== rb6 scale read-out status: {status}" + (f" (missing: {'; '.join(miss_txt)})" if miss_txt else "")
          + f" {LABEL}")
    out["status"] = status
    if opt.json:
        Path(opt.json).write_text(json.dumps(out, indent=1, default=lambda o: None))


if __name__ == "__main__":
    main()
