"""pt6v2 composition pick: runs/pt6v2/AUDIT.md "Composition pick rule (coordinator, registered 2026-11-19 ~05:55 EST)".

    python runs/pt6v2/pick_pt6v2.py [--allow_partial] [--json OUT] [--selftest]

The rule, as implemented (read-only on every source; nothing is written except --json OUT):
  candidates  Part R: landmarks, compass, prefixes, ranks, scalar_adj (stars is NOT FINAL: out);
              toy3: cyc12, torus, hcube, d6, box5, caesar (caesar counts; its source read-out prints it separately).
              Fallbacks in order: people (F102 HELPS 8/8, FLOOR), months (F101 PASS).
  metric      c's PRIMARY area. Part R: readout_ptdom2's run record area[c] (load_all). toy3: readout_ptdom3.dom_numbers:
              tf_area (twin-free) for hcube / d6, area (full read-out set) otherwise.
  evidence    Part R seeds 8-15 (whatever exists), toy3 seeds 0-2 (screen + replication); valid runs only
              (ptdom2: record 'valid'; ptdom3: status VALID or FLAGGED).
  p           the prior lr in {1e-3, 2e-3} with the highest mean over its evidence seeds.
  n           the none lr with the highest mean among every none lr present at the evidence seeds.
  g_s         only-c@p - none@n, paired by seed (the seeds where both exist).
  E1          no FLOOR / CEILING tag on p vs n (the source read-out's definitions, on the paired seeds' means):
              Part R = readout_ptdom2.floor_tags (FLOOR: both <= max(chance, the heuristic lines, the shortcut baseline) + 5;
              CEILING: none >= 95); toy3 = readout_ptdom3's (FLOOR: both <= chance + 5, chance = 100 / n_items on the twin-free
              set, 100 x chance_readout on the full set; CEILING: none >= 95).
  E2          >= 3 paired seeds.
  E3          n is not the highest none lr present, unless that lr is >= 3.2e-2 (pt6v2's top none lr).
  E4          mean g > 0 and >= 2/3 of the paired seeds positive.
  E5          d = mean g / sd(g, ddof 1) >= 0.5.
  pick        the two eligible candidates with the highest d (tie: larger mean g); fewer than two -> fallbacks in order.
              composition = numbers + dim + cities + countries + the two picks (printed in the trainer's order, the form
              make_jobs_pt6v2.py --set_comp takes).
  V2          if a PICKED candidate's n is 3.2e-2: pt6v2 adds none@6.4e-2 (Amendment V2, conditional).
  pick time   strict mode (default) refuses (exit 2) unless: Part R2 complete (every evidence-seed run of the cells used here
              and of picks_R_after2.json's R2 cell list 'C' is valid, or that run failed twice) and every toy3 replication run
              the rule requires has finished (derived from the seed-0 screen, the rule's "toy3 replication" bullet).

Interpretations (the rule does not spell these out; each is printed in the report):
  I1 "present at the evidence seeds": an lr counts when its number of evidence seeds with a valid run is more than half of
     the best-covered lr of the same arm (prior lrs {1e-3, 2e-3} / none lrs). Effect: a toy3 seed-0-only screen cell does not
     compete with cells replicated at seeds 0-2; at Part R seeds 8-15 every lr counts (all were run at all 8 seeds).
  I2 a tie of the means for p or n goes to the higher lr (pt6v2's N_m tie rule).
  I3 toy3 replication cells: a domain qualifies iff its seed-0 prior pick over {1e-3, 2e-3} (readout_ptdom3.pick: F87 rule 1,
     tie -> lower val bits/byte) beats its seed-0 best none over the SCREEN none grid {2e-3, 4e-3, 8e-3, 1.6e-2} on the
     primary area and is not FLOOR (both <= chance + 5). Required: only-d@p0 seeds 1, 2; none@1.6e-2 seeds 1, 2;
     none@3.2e-2 seeds 0, 1, 2; none@n0 seeds 1, 2 when a qualifying domain's n0 is below 1.6e-2. "Finished" = a run JSON
     with status VALID / FLAGGED / INVALID (an INVALID run is finished and kept out of the numbers).
  I4 "failed twice" (Part R2 only): the cell-seed has no valid run, its SUBMITTED markers in runs/ptdom2 name >= 2 attempts,
     every attempt's sacct state is terminal (none PENDING / RUNNING / ..., none unknown). Each attempt and state is printed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

B = Path("/scratch/yh6210/llmadd_torch")
R6 = B / "runs" / "pt6v2"
R2DIR = B / "runs" / "ptdom2"
R3DIR = B / "runs" / "ptdom3"
PICKS = R2DIR / "picks_R_after2.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def _imports():
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        for p in (str(B), str(R2DIR)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import readout_ptdom2 as r2     # noqa: E402  (read only: load_all, R_DOMS, floor_tags; imports readout_ptdom as RO)
        import readout_ptdom3 as r3     # noqa: E402  (read only: run_json, status, dom_numbers, pick, PRIMARY_TF, TOY3)
    finally:
        sys.argv = argv
    return r2, r3


R2, RO3 = _imports()
RO = R2.RO
assert Path(R2.__file__).resolve() == (R2DIR / "readout_ptdom2.py").resolve()
assert Path(RO3.__file__).resolve() == (B / "readout_ptdom3.py").resolve()

# ---------------------------------------------------------------- the rule's constants (AUDIT.md, pick rule)
CAND_R = ("landmarks", "compass", "prefixes", "ranks", "scalar_adj")
OUT_R = {"stars": "NOT FINAL in picks_R_after2.json: out by the rule"}
CAND_T = ("cyc12", "torus", "hcube", "d6", "box5", "caesar")
BASE = ("dim", "cities", "countries")
FALLBACKS = ("people", "months")
FALLBACK_WHY = {"people": "F102 HELPS 8/8, FLOOR", "months": "F101 PASS (ends below chance)"}
PRIOR_LRS = ("1e-3", "2e-3")
EVID = {"ptdom2": tuple(range(8, 16)), "ptdom3": (0, 1, 2)}
TOP_NONE = 3.2e-2
MIN_SEEDS, D_MIN = 3, 0.5
SCREEN_NONE = ("2e-3", "4e-3", "8e-3", "1.6e-2")
REPL_SEEDS = (1, 2)
REPL_SHARED = tuple(("none", "1.6e-2", s) for s in (1, 2)) + tuple(("none", "3.2e-2", s) for s in (0, 1, 2))
OKS = ("VALID", "FLAGGED")
ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUE_HOLD",
          "REQUEUE_FED", "SIGNALING", "STAGE_OUT", "STOPPED"}
E_KEYS = ("E1", "E2", "E3", "E4", "E5")
assert set(CAND_R) | set(OUT_R) == set(R2.R_DOMS) and tuple(RO3.TOY3) == CAND_T and set(RO3.PRIMARY_TF) == {"hcube", "d6"}
assert tuple(RO3.GRID["none"]) == SCREEN_NONE
num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
mean = lambda xs: (lambda v: float(np.mean(v)) if v else None)(list(xs))


# ---------------------------------------------------------------- composition order (make_jobs_pt6v2's comp_doms)
def comp_order(five):
    """-> (the five domains in the trainer's order, where the order came from). make_jobs_pt6v2.comp_doms when it imports
    (it is the --set_comp parser); else the same order rebuilt from make_jobs_ptdom.BATCHES (make_jobs_pt6comb.TRAINER_ORDER)
    + TOY3, and a note."""
    try:
        argv, sys.argv = sys.argv, [sys.argv[0]]
        try:
            import make_jobs_pt6v2 as MJ6   # noqa: E402
        finally:
            sys.argv = argv
        assert set(five) <= set(MJ6.ALLOWED), f"{five} not in ALLOWED"
        return list(MJ6.comp_doms(list(five))), f"make_jobs_pt6v2.comp_doms (sha {sha(B / 'make_jobs_pt6v2.py')})"
    except Exception as ex:   # the file is being edited by another session: fall back to the same order
        BT = RO.MJ.BATCHES
        order = ("months", "weekdays", "clock", "s4", "dim") + ("stars", "landmarks", "languages", "colours", "cities") \
            + tuple(BT["B1"]) + tuple(BT["B2"]) + tuple(BT["B3"]) + tuple(RO3.TOY3)
        assert len(five) == 5 and len(set(five)) == 5 and set(five) <= set(order), five
        return [d for d in order if d in five], f"FALLBACK order (make_jobs_pt6v2 import failed: {ex!r})"[:200]


# ---------------------------------------------------------------- toy3 loader (any seed; readout_ptdom3's parsing)
NAME3 = re.compile(rf"^{RO3.PREFIX}-{RO3.KEY}-(none|only-[a-z0-9]+)-lr([0-9.e-]+)-s(\d+)(?:-r(\d+))?$")


def load3(root=None):
    """{(arm, lr, seed): {dir, rr, status, note, n: {d: readout_ptdom3.dom_numbers}, marks}} over every seed. readout_ptdom3.load's
    rules (the highest -r<k> with a dir is kept; the run JSON's identity must match the directory name; status from
    readout_ptdom3.status). The tag may be the seed-free screen tag or carry '-s<seed>'."""
    root = Path(root or (R3DIR / "out"))
    keep = {}
    for d in sorted(root.glob(f"{RO3.PREFIX}-{RO3.KEY}-*")):
        m = NAME3.match(d.name)
        if not m:
            continue
        k, rr = (m.group(1), m.group(2), int(m.group(3))), int(m.group(4) or 0)
        if k not in keep or rr > keep[k][0]:
            keep[k] = (rr, d)
    runs = {}
    for (arm, lr, seed), (rr, d) in sorted(keep.items()):
        f = RO3.run_json(d)
        r, why = None, ""
        if f is not None:
            try:
                r = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError) as ex:
                r, why = None, f"unreadable run JSON ({ex.__class__.__name__})"
        if r is not None:
            a_ = r.get("args", {})
            base = f"_{RO3.PREFIX}-{RO3.KEY}-{arm}-lr{lr}"
            exp_pd = "" if arm == "none" else arm[5:]
            if (a_.get("seed"), a_.get("lr"), a_.get("prior_domains")) != (seed, float(lr), exp_pd) or a_.get("tag") not in (base, f"{base}-s{seed}"):
                r, why = None, f"identity of the run JSON differs from its directory name {d.name}"
        act = () if arm == "none" else (arm[5:],)
        if r is not None:
            stt, note = RO3.status(d, r, act)
            # coordinator 2026-11-19 (audit_pick_pt6v2.md important 2): the job writes the run JSON before check.json, so
            # "no check.json" is a run still finishing, not an invalid one
            if stt == "INVALID" and "no check.json" in str(note):
                stt, note = "UNFINISHED", f"{note} (run JSON written, check.json not yet: still finishing)"
        elif f is None:
            stt, note = "UNFINISHED", "no run JSON"
        else:
            stt, note = ("UNFINISHED" if why.startswith("unreadable") else "INVALID"), why
        rec = {"dir": str(d), "rr": rr, "status": stt, "note": note, "n": {}, "marks": []}
        if r is not None and stt != "UNFINISHED":
            for dom in RO3.TOY3:
                n, mk = RO3.dom_numbers(r, dom)
                rec["n"][dom] = n
                rec["marks"] += mk
        runs[(arm, lr, seed)] = rec
    return runs


def t_key(d):
    return "tf_area" if d in RO3.PRIMARY_TF else "area"


def t_val(rec, d):
    if not rec or rec.get("status") not in OKS:
        return None
    v = (rec.get("n", {}).get(d) or {}).get(t_key(d))
    return float(v) if num(v) else None


def t_chance(runs3, d):
    key = "chance_u" if d in RO3.PRIMARY_TF else "chance"
    for rec in runs3.values():
        v = (rec.get("n", {}).get(d) or {}).get(key)
        if rec.get("status") in OKS and num(v):
            return float(v)
    return None


# ---------------------------------------------------------------- candidates
def vals_R(R_, b, arm, lr, d, seeds):
    out = {}
    for s in seeds:
        r = R_.get((b, arm, lr, s))
        if r and r.get("valid") and num((r.get("area") or {}).get(d)):
            out[s] = float(r["area"][d])
    return out


def gather_R(R_, subs, picks):
    C = {}
    for d in CAND_R:
        b, ev = R2.R_DOMS[d], EVID["ptdom2"]
        prior = {lr: v for lr in PRIOR_LRS for v in [vals_R(R_, b, f"only-{d}", lr, d, ev)] if v}
        nl = sorted({k[2] for k in R_ if k[0] == b and k[1] == "none" and k[3] in ev}, key=float)
        none = {lr: v for lr in nl for v in [vals_R(R_, b, "none", lr, d, ev)] if v}
        grid = ((picks.get("domains", {}).get(d) or {}).get("unit_none") or {}).get("grid") or []

        def fl(p, n, seeds, pm, nm, b=b, d=d):
            x = R2.floor_tags(R_, subs, b, [(f"only-{d}", p), ("none", n)], d, list(seeds))
            return None if x is None else {"line": float(x["line"]), "tags": list(x["tags"]), "lines": dict(zip(x["line_names"], map(float, x["lines"])))}
        C[d] = {"cand": d, "src": "ptdom2", "batch": b, "metric": "area", "evid": ev, "prior": prior, "none": none, "floor": fl,
                "src_grid": sorted(grid, key=float), "src_grid_name": "R1 none grid (seeds 0-3, picks_R_after2.json)"}
    return C


def gather_T(runs3):
    C = {}
    for d in CAND_T:
        ev = EVID["ptdom3"]
        prior = {lr: v for lr in PRIOR_LRS for v in [{s: x for s in ev for x in [t_val(runs3.get((f"only-{d}", lr, s)), d)] if x is not None}] if v}
        nl = sorted({k[1] for k in runs3 if k[0] == "none" and k[2] in ev}, key=float)
        none = {lr: v for lr in nl for v in [{s: x for s in ev for x in [t_val(runs3.get(("none", lr, s)), d)] if x is not None}] if v}
        ch = t_chance(runs3, d)

        def fl(p, n, seeds, pm, nm, ch=ch):
            if ch is None:
                return None
            tags = []
            if pm <= ch + 5 and nm <= ch + 5:
                tags.append(f"FLOOR (both <= chance {ch:.1f} + 5)")
            if nm >= 95:
                tags.append("CEILING (none >= 95)")
            return {"line": ch + 5, "tags": tags}
        # audit important 4: the 3.2e-2 extension counts in the grid only where it has a valid run for d at some seed
        ext = [lr for lr in ("3.2e-2",) if any(t_val(r_, d) is not None for k_, r_ in runs3.items() if k_[:2] == ("none", lr))]
        C[d] = {"cand": d, "src": "ptdom3", "batch": "T3", "metric": "twin-free area" if d in RO3.PRIMARY_TF else "area (full set)",
                "evid": ev, "prior": prior, "none": none, "floor": fl, "chance": ch,
                "src_grid": list(SCREEN_NONE) + ext, "src_grid_name": "ptdom3 none grid (screen + valid 3.2e-2 runs)"}
    return C


def present(cells):
    """I1: lrs whose evidence-seed coverage is more than half of the arm's best-covered lr."""
    cov = {lr: len(v) for lr, v in cells.items() if v}
    if not cov:
        return []
    mx = max(cov.values())
    return sorted([lr for lr, k in cov.items() if 2 * k > mx], key=float)


def best_lr(cells, lrs):
    """Highest mean over its evidence seeds; I2: a tie (1e-9) goes to the higher lr."""
    return max(lrs, key=lambda lr: (round(mean(cells[lr].values()), 9), float(lr)))


def evaluate(c):
    row = {"cand": c["cand"], "src": c["src"], "batch": c["batch"], "metric": c["metric"], "p": None, "n": None, "seeds": [],
           "g": {}, "mean_g": None, "wins": None, "k": 0, "sd": None, "d": None, "E": {k: False for k in E_KEYS},
           "why": {k: "n/a" for k in E_KEYS}, "eligible": False, "floor": None,
           "prior_means": {lr: [round(mean(v.values()), 4), len(v)] for lr, v in c["prior"].items()},
           "none_means": {lr: [round(mean(v.values()), 4), len(v)] for lr, v in c["none"].items()},
           "present_prior": present(c["prior"]), "present_none": present(c["none"]), "note": ""}
    pp, npr = row["present_prior"], row["present_none"]
    if not pp or not npr:
        miss = ("no only-%s run at 1e-3 / 2e-3" % c["cand"]) if not pp else "no none run"
        row["why"] = {k: f"n/a: {miss} at the evidence seeds" for k in E_KEYS}
        row["reason"] = f"OUT: {miss} at the evidence seeds"
        return row
    p, n = best_lr(c["prior"], pp), best_lr(c["none"], npr)
    seeds = sorted(set(c["prior"][p]) & set(c["none"][n]))
    g = {s: c["prior"][p][s] - c["none"][n][s] for s in seeds}
    gv = np.array([g[s] for s in seeds], dtype=float)
    k = len(seeds)
    row.update(p=p, n=n, seeds=seeds, g={s: float(v) for s, v in g.items()}, k=k)
    top = max(npr, key=float)
    if k:
        mg, wins = float(gv.mean()), int((gv > 0).sum())
        sd = float(gv.std(ddof=1)) if k >= 2 else None
        if sd is None:
            d = None
        elif sd > 0:
            d = mg / sd
        else:
            d = math.inf if mg > 0 else (-math.inf if mg < 0 else None)
        row.update(mean_g=mg, wins=wins, sd=sd, d=d)
        pm, nm = mean(c["prior"][p][s] for s in seeds), mean(c["none"][n][s] for s in seeds)
        fl = c["floor"](p, n, seeds, pm, nm)
        row["floor"] = fl
        row["pn_means_paired"] = [pm, nm]
        if fl is None:
            row["E"]["E1"], row["why"]["E1"] = False, "n/a: FLOOR line not computable"
        else:
            row["E"]["E1"] = not fl["tags"]
            row["why"]["E1"] = ("; ".join(fl["tags"]) if fl["tags"] else "no tag") + f" (line {fl['line']:.1f}; means p {pm:.1f} / n {nm:.1f}" + ("; lines " + ", ".join(f"{k} {v:.1f}" for k, v in fl["lines"].items()) if fl.get("lines") else "") + ")"
        row["E"]["E4"] = mg > 0 and 3 * wins >= 2 * k
        row["why"]["E4"] = f"mean g {mg:+.2f}, wins {wins}/{k} (need > 0 and >= 2/3)"
        row["E"]["E5"] = d is not None and d >= D_MIN
        row["why"]["E5"] = (f"d {fmt_d(d)} (need >= {D_MIN})" if d is not None else "n/a: d needs >= 2 paired seeds and sd > 0 or mean != 0")
    else:
        for e in ("E1", "E4", "E5"):
            row["why"][e] = "n/a: no paired seed"
    row["E"]["E2"] = k >= MIN_SEEDS
    row["why"]["E2"] = f"{k} paired seeds ({'>=' if k >= MIN_SEEDS else '<'} {MIN_SEEDS})"
    # coordinator 2026-11-19 (AUDIT.md "Pick rule readings"): E3 on the batch's whole none grid at any seed (the source grid
    # + the lrs present), i.e. "the none optimum is bracketed"; the literal evidence-seed reading is kept as a note
    grid = c.get("src_grid") or []
    top = max(list(npr) + list(grid), key=float)
    at_top = float(n) >= float(top) - 1e-12
    row["E"]["E3"] = (not at_top) or float(top) >= TOP_NONE - 1e-12
    lit = (n == max(npr, key=float)) and float(n) < TOP_NONE - 1e-12
    row["why"]["E3"] = ((f"n {n} = the top of the batch's none grid ({', '.join(sorted(set(npr) | set(grid), key=float))}) and < 3.2e-2"
                         if not row["E"]["E3"] else
                         (f"n {n} below the batch's top none lr {top}" if not at_top else f"n {n} = the batch's top, >= 3.2e-2"))
                        + (f" [literal reading would fail: n = the highest lr at the evidence seeds ({', '.join(npr)})]" if lit and row["E"]["E3"] else ""))
    if grid:
        row["note"] = (f"descriptive: n {n} vs the {c['src_grid_name']} {grid}: "
                       + ("at its top" if float(n) >= max(float(x) for x in grid) - 1e-12 else "not at its top"))
    row["eligible"] = all(row["E"].values())
    fails = [f"{e} ({row['why'][e]})" for e in E_KEYS if not row["E"][e]]
    row["reason"] = "IN (eligible)" if row["eligible"] else "OUT: " + "; ".join(fails)
    return row


def fmt_d(d):
    return "-" if d is None else ("inf" if d == math.inf else ("-inf" if d == -math.inf else f"{d:.2f}"))


def choose(rows, comp_fn=comp_order):
    el = [r for r in rows if r["eligible"]]
    key = lambda r: (-round(r["d"], 9), -round(r["mean_g"], 9))
    el_s = sorted(el, key=key)   # stable: equal keys keep the candidate order (printed as an unresolved tie)
    ties = []
    for a, b in zip(el_s, el_s[1:]):
        if round(a["d"], 9) == round(b["d"], 9):
            ties.append(f"tie on d {fmt_d(a['d'])}: {a['cand']} (mean g {a['mean_g']:+.3f}) vs {b['cand']} (mean g {b['mean_g']:+.3f})"
                        + (" -> larger mean g first" if key(a) != key(b) else " -> UNRESOLVED (equal mean g): candidate order"))
    picks = [r["cand"] for r in el_s[:2]]
    fb = list(FALLBACKS[:2 - len(picks)])
    comp, src = comp_fn(list(BASE) + picks + fb)
    v2 = [r["cand"] for r in el_s[:2] if r["n"] is not None and abs(float(r["n"]) - TOP_NONE) < 1e-12]
    return {"picks": picks, "fallbacks_used": fb, "composition": comp, "comp_order_src": src, "extra_none_6.4e-2": bool(v2),
            "v2_by": v2, "ranking": [[r["cand"], r["d"], r["mean_g"]] for r in el_s], "ties": ties}


# ---------------------------------------------------------------- pick-time completeness
def repl_required(runs3):
    """I3: -> (required {(arm, lr, seed)}, per-domain qualification info, undetermined [(d, cells)])."""
    req, info, undet = set(), {}, []
    for d in CAND_T:
        need0 = [(f"only-{d}", lr) for lr in PRIOR_LRS] + [("none", lr) for lr in SCREEN_NONE]
        unf = [f"{a}@{lr}" for a, lr in need0 if (runs3.get((a, lr, 0)) or {}).get("status", "MISSING") in ("UNFINISHED", "MISSING")]
        row = lambda a, lr: (lr, {"area": t_val(runs3.get((a, lr, 0)), d),
                                  "bpb": ((runs3.get((a, lr, 0)) or {}).get("n", {}).get(d) or {}).get("bpb")},
                             t_val(runs3.get((a, lr, 0)), d) is not None)
        pp = RO3.pick([row(f"only-{d}", lr) for lr in PRIOR_LRS])
        pn = RO3.pick([row("none", lr) for lr in SCREEN_NONE])
        ch = t_chance(runs3, d)
        x = {"p0": pp[0] if pp else None, "p0_val": pp[1]["area"] if pp else None, "n0": pn[0] if pn else None,
             "n0_val": pn[1]["area"] if pn else None, "chance": ch, "unfinished_s0": unf}
        if unf:
            x.update(status="UNDETERMINED", qualifies=None)
            undet.append((d, unf))
        else:
            floor = bool(pp and pn and ch is not None and pp[1]["area"] <= ch + 5 and pn[1]["area"] <= ch + 5)
            q = bool(pp and pn and pp[1]["area"] > pn[1]["area"] and not floor)
            x.update(floor=floor, qualifies=q, status="QUALIFIES" if q else
                     ("FLOOR" if floor else ("no valid seed-0 prior / none" if not (pp and pn) else "prior <= none at seed 0")))
            if q:
                req |= {(f"only-{d}", pp[0], s) for s in REPL_SEEDS}
                if float(pn[0]) < 1.6e-2 - 1e-12:
                    req |= {("none", pn[0], s) for s in REPL_SEEDS}
        info[d] = x
    if any(v.get("qualifies") for v in info.values()):
        req |= set(REPL_SHARED)
    return req, info, undet


def t_problems(runs3):
    req, info, undet = repl_required(runs3)
    probs, rows = [], []
    for d, cells in undet:
        probs.append(f"toy3 {d}: seed-0 screen not finished ({', '.join(cells)}): its replication cells cannot be derived")
    for cell in sorted(req, key=lambda c: (c[0], float(c[1]), c[2])):
        rec = runs3.get(cell) or {}
        st = rec.get("status", "MISSING")
        rows.append((cell, st))
        # audit important 1 / 3: a required cell must be VALID / FLAGGED; an INVALID one refuses (strict) unless it already
        # had >= 2 attempts (its kept dir is -r1 or later), the toy3 analogue of Part R's I4
        if st in ("UNFINISHED", "MISSING"):
            probs.append(f"toy3 replication {cell[0]}@{cell[1]} s{cell[2]}: {st}")
        elif st not in OKS and int(rec.get("rr") or 0) < 1:
            probs.append(f"toy3 replication {cell[0]}@{cell[1]} s{cell[2]}: {st} ({rec.get('note')}): re-run it once "
                         f"(make_jobs_ptdom3r.py, -r1) before picking; a second failure is excused")
    return probs, rows, info


def markers(name, root=R2DIR):
    """SUBMITTED_<name>[-r<k>] markers in runs/ptdom2 -> [(rr, job id or None)]."""
    rx = re.compile(rf"^SUBMITTED_{re.escape(name)}(?:-r(\d+))?$")
    out = []
    for f in Path(root).glob(f"SUBMITTED_{name}*"):
        m = rx.match(f.name)
        if m:
            j = re.search(r"Submitted batch job (\d+)", f.read_text())
            out.append((int(m.group(1) or 0), j.group(1) if j else None))
    return sorted(out)


def sacct_states(ids):
    ids = sorted({i for i in ids if i})
    if not ids:
        return {}
    try:
        o = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(ids)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    st = {}
    for ln in o.stdout.splitlines():
        if "|" in ln:
            j, s = ln.split("|", 1)
            st[j.strip()] = (s.split() or ["UNKNOWN"])[0]
    return st


def r2_problems(R_, picks, C_R, markers_fn=markers, states_fn=sacct_states):
    """Part R2 complete: every evidence-seed run of picks 'C' (R2's cells) and of the cells used here is valid, or failed
    twice (I4). -> (problems, excused lines)."""
    need = {tuple(x) for x in picks.get("C", [])}
    for d, c in C_R.items():
        need |= {(c["batch"], f"only-{d}", lr) for lr in c["prior"]} | {(c["batch"], "none", lr) for lr in c["none"]}
    miss = [(cell, s) for cell in sorted(need) for s in EVID["ptdom2"] if not (R_.get(cell + (s,)) or {}).get("valid")]
    att = {(cell, s): markers_fn(f"ptdom2-{cell[0]}-{cell[1]}-lr{cell[2]}-s{s}") for cell, s in miss}
    states = states_fn([j for a in att.values() for _, j in a])
    probs, exc = [], []
    for (cell, s), a in att.items():
        st = [(rr, j, states.get(j, "UNKNOWN") if j else "UNKNOWN") for rr, j in a]
        txt = ", ".join(f"r{rr} {j} {x}" for rr, j, x in st) or "no SUBMITTED marker"
        rec = R_.get(cell + (s,))
        why = (rec or {}).get("why") or ("no run record" if rec is None else "not valid")
        line = f"ptdom2 {cell[0]} {cell[1]}@{cell[2]} s{s}: no valid run ({why}); attempts: {txt}"
        if len(st) >= 2 and all(x not in ACTIVE and x != "UNKNOWN" for _, _, x in st):
            exc.append("EXCUSED (failed twice) " + line)
        else:
            probs.append(line)
    return probs, exc


def picks_problems(picks):
    p = []
    if not (picks.get("complete") and picks.get("stage_written") == "R2"):
        p.append(f"picks_R_after2.json: complete {picks.get('complete')} stage_written {picks.get('stage_written')} (want True / R2)")
    for d in CAND_R:
        if (picks.get("domains", {}).get(d) or {}).get("status") != "FINAL":
            p.append(f"picks_R_after2.json: {d} status {(picks.get('domains', {}).get(d) or {}).get('status')} != FINAL (the rule's candidate list)")
    for d in OUT_R:
        if (picks.get("domains", {}).get(d) or {}).get("status") == "FINAL":
            p.append(f"picks_R_after2.json: {d} is FINAL now (the rule put it out as NOT FINAL)")
    return p


# ---------------------------------------------------------------- run + report
def run(R_, subs, picks, runs3, strict, markers_fn=markers, states_fn=sacct_states, comp_fn=comp_order, inputs=None):
    C_R, C_T = gather_R(R_, subs, picks), gather_T(runs3)
    pr = picks_problems(picks)
    r2p, r2x = r2_problems(R_, picks, C_R, markers_fn, states_fn)
    tp, trows, tinfo = t_problems(runs3)
    probs = pr + r2p + tp
    if probs and strict:
        L = ["pt6v2 composition pick: REFUSED (strict mode; inputs incomplete). Re-run at pick time, or --allow_partial for a "
             "partial standing (not a pick).", f"{len(probs)} problem(s):"] + [f"  - {x}" for x in probs] + [f"  {x}" for x in r2x]
        return 2, "\n".join(L), None
    rows = [evaluate(C_R[d]) for d in CAND_R] + [evaluate(C_T[d]) for d in CAND_T]
    ch = choose(rows, comp_fn)
    obj = {"composition": ch["composition"], "picks": ch["picks"], "fallbacks_used": ch["fallbacks_used"],
           "extra_none_6.4e-2": ch["extra_none_6.4e-2"], "table": [strip(r) for r in rows], "inputs": inputs or {},
           "complete": not probs, "mode": "strict" if strict else "allow_partial", "problems": probs, "excused": r2x,
           "v2_by": ch["v2_by"], "ranking": ch["ranking"], "ties": ch["ties"], "comp_order_src": ch["comp_order_src"],
           "toy3_replication": {"qualification": tinfo, "required": [[list(c), s] for c, s in trows]},
           "out_by_rule": OUT_R}
    tm = t_marks(runs3)
    obj["toy3_marks"] = tm
    txt = report(rows, ch, probs, r2x, trows, tinfo, strict, inputs or {})
    if tm:
        txt += ("\ntoy3 MARKS (logged, not blocking: a usable run without its primary value drops out of the evidence):\n"
                + "\n".join(f"  {x}" for x in tm))
    return 0, txt, obj


def t_marks(runs3):
    """VALID / FLAGGED toy3 runs at the evidence seeds with no primary value for a domain this pick reads (e.g. the twin-free
    read-out of hcube / d6 unavailable at a replication seed): readout_ptdom3's marks for it."""
    out = []
    for (arm, lr, s), rec in sorted(runs3.items(), key=lambda kv: (kv[0][0], float(kv[0][1]), kv[0][2])):
        if rec.get("status") not in OKS or s not in EVID["ptdom3"]:
            continue
        miss = [d for d in (CAND_T if arm == "none" else (arm[5:],)) if d in CAND_T and t_val(rec, d) is None]
        if miss:
            mk = [m for m in rec.get("marks", []) if any(m.startswith(d + ":") for d in miss)]
            out.append(f"{arm}@{lr} s{s}: no primary value for {', '.join(miss)}" + (f" ({'; '.join(mk)[:300]})" if mk else ""))
    return out


def strip(r):
    x = {k: v for k, v in r.items() if k != "floor"}
    x["floor"] = r["floor"]
    x["g"] = {str(s): v for s, v in r["g"].items()}
    x["d"] = None if r["d"] is None else (r["d"] if math.isfinite(r["d"]) else fmt_d(r["d"]))
    return x


def seeds_txt(ss):
    if not ss:
        return "-"
    if ss == list(range(ss[0], ss[-1] + 1)) and len(ss) > 2:
        return f"{ss[0]}-{ss[-1]}"
    return ",".join(map(str, ss))


def report(rows, ch, probs, r2x, trows, tinfo, strict, inputs):
    f2 = lambda v: "-" if v is None else f"{v:+.2f}"
    L = ["pt6v2 composition pick (runs/pt6v2/AUDIT.md 'Composition pick rule', registered 2026-11-19 ~05:55 EST)",
         f"mode: {'STRICT (pick time)' if strict else 'ALLOW_PARTIAL: inputs incomplete -> a PARTIAL STANDING, NOT the pick'}"
         if probs else f"mode: {'strict' if strict else 'allow_partial'}; inputs complete",
         "inputs: " + ", ".join(f"{k} {v}" for k, v in inputs.items()), ""]
    if probs:
        L.append(f"INCOMPLETE ({len(probs)}):")
        L += [f"  - {x}" for x in probs]
    L += [f"  {x}" for x in r2x]
    L.append("")
    L.append("toy3 replication (I3; seed-0 screen, primary area): " + "; ".join(
        f"{d} {v['status']}" + (f" (p0 {v['p0']} {v['p0_val']:.1f} vs n0 {v['n0']} {v['n0_val']:.1f}, chance {v['chance']:.1f})"
                                if v.get("p0") and v.get("n0") and v.get("chance") is not None else "") for d, v in tinfo.items()))
    if trows:
        L.append("  required cells: " + ", ".join(f"{c[0]}@{c[1]} s{c[2]} {s}" for c, s in trows))
    L.append("")
    L.append("stars: OUT (" + OUT_R["stars"] + "); s4: OUT (F102 NOT SHOWN; the user: swap it)")
    hdr = f"{'cand':10s} {'src':6s} {'p':5s} {'n':6s} {'k':>2s} {'paired seeds':12s} {'mean g':>7s} {'wins':>5s} {'d':>6s}  E1 E2 E3 E4 E5  in/out"
    L += [hdr, "-" * len(hdr)]
    for r in rows:
        e = " ".join(("ok" if r["E"][k] else "NO") for k in E_KEYS)
        wt = "-" if r["wins"] is None else "%d/%d" % (r["wins"], r["k"])
        L.append(f"{r['cand']:10s} {r['src']:6s} {str(r['p'] or '-'):5s} {str(r['n'] or '-'):6s} {r['k']:>2d} {seeds_txt(r['seeds']):12s} "
                 f"{f2(r['mean_g']):>7s} {wt:>5s} {fmt_d(r['d']):>6s}  {e}  {r['reason']}")
        L.append("    per-seed g: " + (", ".join(f"s{s} {v:+.1f}" for s, v in r["g"].items()) or "-") + f"   [{r['metric']}]")
        L.append("    means (n seeds): prior " + ", ".join(f"{lr} {m:.1f} ({k})" for lr, (m, k) in r["prior_means"].items())
                 + " | none " + ", ".join(f"{lr} {m:.1f} ({k})" for lr, (m, k) in r["none_means"].items())
                 + f" | present (I1): prior {r['present_prior']}, none {r['present_none']}")
        L.append("    " + " | ".join(f"{k}: {r['why'][k]}" for k in E_KEYS))
        if r.get("note"):
            L.append(f"    {r['note']}")
    L.append("")
    L.append("eligible ranking (d, then mean g): " + (", ".join(f"{c} d {fmt_d(d)} g {g:+.2f}" for c, d, g in ch["ranking"]) or "none"))
    L += [f"  {t}" for t in ch["ties"]]
    L.append(f"PICKS: {ch['picks'] or 'none'}")
    L.append("FALLBACKS USED: " + (", ".join(f"{x} ({FALLBACK_WHY[x]})" for x in ch["fallbacks_used"]) or "none"))
    L.append(f"COMPOSITION: numbers + {', '.join(ch['composition'])}   (--set_comp {','.join(ch['composition'])}; order: {ch['comp_order_src']})")
    L.append("AMENDMENT V2 (none@6.4e-2, 8 runs): " + (f"TRIGGERED (n = 3.2e-2 for picked {', '.join(ch['v2_by'])})"
                                                     if ch["extra_none_6.4e-2"] else "not triggered (no picked candidate has n = 3.2e-2)"))
    e3 = [r["cand"] for r in rows if not r["E"]["E3"] and r["n"]]
    if e3:
        L.append("E3 note: E3 fails for " + ", ".join(e3) + " (n = the highest none lr run at the evidence seeds, < 3.2e-2); "
                 "the descriptive line under each row places n on the source's full none grid.")
    if probs:
        L.append("PARTIAL: the inputs are incomplete; the picks / composition above are a standing, NOT the pick.")
    L.append("Interpretations I1-I4: see the module docstring (present = coverage > half of the best-covered lr; p / n ties -> "
             "higher lr; toy3 replication from the screen none grid; failed twice = >= 2 terminal attempts).")
    return "\n".join(L)


def build_subs(R_):
    meta = RO.domain_meta(list(CAND_R))
    bs = {R2.R_DOMS[d] for d in CAND_R}
    Rsub = {k: dict(v, cmp={x: y for x, y in v.get("cmp", {}).items() if x in meta}) for k, v in R_.items()
            if v.get("cmp") and k[0] in bs and k[3] in EVID["ptdom2"]}
    return RO.subsets_all(Rsub, meta)


def inputs_shas():
    return {"readout_ptdom2.py": sha(R2.__file__), "readout_ptdom.py": sha(RO.__file__), "readout_ptdom3.py": sha(RO3.__file__),
            "picks_R_after2.json": sha(PICKS), "make_jobs_pt6v2.py": sha(B / "make_jobs_pt6v2.py"), "pick_pt6v2.py": sha(__file__)}


# ---------------------------------------------------------------- self-test (planted worlds)
def _fake_R(spec, heur=None):
    """spec {(b, arm, lr): {seed: {d: area}}} -> a readout_ptdom2-like R_ (the fields gather_R / floor_tags read)."""
    R_ = {}
    for (b, arm, lr), per in spec.items():
        for s, areas in per.items():
            R_[(b, arm, lr, s)] = {"batch": b, "arm": arm, "lr": lr, "seed": s, "valid": True, "area": dict(areas),
                                   "chance": {d: 50.0 for d in areas}, "heur": {d: list((heur or {}).get(d, (50.0, 50.0))) for d in areas}}
    return R_


def _fake_T(spec, chance=None):
    """spec {(arm, lr, seed): {d: primary area}} -> a load3-like runs3."""
    ch = {"cyc12": 100 / 12, "torus": 100 / 36, "hcube": 100 / 32, "d6": 100 / 12, "box5": 4.0, "caesar": 100 / 26, **(chance or {})}
    out = {}
    for k, per in spec.items():
        out[k] = {"status": "VALID", "n": {d: {"area": v, "tf_area": v, "chance": ch[d], "chance_u": ch[d], "bpb": 1.0} for d, v in per.items()}}
    return out


def _world_R(cands, heur=None):
    """cands {d: {'prior': {lr: [v per seed 8..]}, 'none': {lr: [..]}}} -> R_ (none runs shared within a batch)."""
    spec = {}
    for d, x in cands.items():
        b = R2.R_DOMS[d]
        for arm, cells in (("only-" + d, x["prior"]), ("none", x["none"])):
            for lr, vs in cells.items():
                for i, v in enumerate(vs):
                    if v is not None:
                        spec.setdefault((b, arm, lr), {}).setdefault(8 + i, {})[d] = float(v)
    return _fake_R(spec, heur)


def _world_T(cands):
    """cands {d: {'prior': {lr: [v at s0, s1, s2]}, 'none': {lr: [..]}}} -> runs3 (none runs shared)."""
    spec = {}
    for d, x in cands.items():
        for arm, cells in (("only-" + d, x["prior"]), ("none", x["none"])):
            for lr, vs in cells.items():
                for s, v in enumerate(vs):
                    if v is not None:
                        spec.setdefault((arm, lr, s), {})[d] = float(v)
    return _fake_T(spec)


PICKS_FAKE = {"complete": True, "stage_written": "R2", "C": [],
              "domains": {**{d: {"status": "FINAL", "unit_none": {"grid": ["1e-3", "2e-3", "4e-3", "8e-3"]}} for d in CAND_R},
                          # E3 (source-grid reading): prefixes' source grid tops at 4e-3, so its n 4e-3 is truncated
                          "prefixes": {"status": "FINAL", "unit_none": {"grid": ["1e-3", "2e-3", "4e-3"]}},
                          "stars": {"status": "NOT FINAL"}}}
add = lambda base, g: [None if (b is None or x is None) else b + x for b, x in zip(base, g)]


def selftest():
    T_, n_ok = [], 0

    def ok(name, cond, val=""):
        nonlocal n_ok
        T_.append(name)
        n_ok += bool(cond)
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"   got: {str(val)[:300]}"))

    fixed = lambda five: ([d for d in ("dim", "cities", "countries", "people", "months", "landmarks", "compass", "prefixes", "ranks",
                                       "scalar_adj") + CAND_T if d in five], "test order")
    nos = lambda: (lambda name: [], lambda ids: {})
    # ---- world 1: eligible, FLOOR, CEILING, truncated none (E3), d < 0.5, E2, the I1 majority rule, V2 trigger
    b8 = [80.0] * 8
    WR = {"ranks": {"prior": {"1e-3": add(b8, [3, 2, 4, 3, 2, 3, 4, 3])}, "none": {"1e-3": [70.0] * 8, "2e-3": b8, "4e-3": [75.0] * 8}},
          "prefixes": {"prior": {"1e-3": add([86.0] * 8, [3, 2, 4, 3, 2, 3, 4, 3])}, "none": {"1e-3": [70.0] * 8, "2e-3": [80.0] * 8, "4e-3": [86.0] * 8}},
          "compass": {"prior": {"1e-3": add(b8, [1, 1, 1, 1, 1, 1, -5, 0.5])}, "none": {"1e-3": [70.0] * 8, "2e-3": b8, "4e-3": [75.0] * 8}},
          "landmarks": {"prior": {"2e-3": add([88.0] * 8, [2, 2.5, 1.5, 2, 2, 2.5, 1.5, 2])}, "none": {"2e-3": [80.0] * 8, "4e-3": [88.0] * 8, "8e-3": [85.0] * 8}},
          "scalar_adj": {"prior": {"1e-3": [98.0] * 8}, "none": {"2e-3": [96.0, 96.5, 95.5, 96, 96, 96.5, 95.5, 96], "4e-3": [90.0] * 8}}}
    WR["scalar_adj"]["prior"]["1e-3"] = add(WR["scalar_adj"]["none"]["2e-3"], [1, 2, 1.5, 2, 1, 2, 1.5, 1])
    # B2's none runs are shared: give each B2 domain its own values in the same none runs
    R1 = _world_R(WR, heur={"landmarks": (90.0, 60.0)})
    WT = {"torus": {"prior": {"1e-3": [90.0, 88.0, 91.0], "2e-3": [95.0, None, None]},
                    "none": {"2e-3": [10.0, None, None], "8e-3": [99.0, None, None], "1.6e-2": [70.0, 69.0, 71.0], "3.2e-2": [80.0, 78.0, 79.0]}},
          "hcube": {"prior": {"2e-3": [90.0, 91.0, None]}, "none": {"1.6e-2": [70.0, 70.0, 70.0], "3.2e-2": [80.0, 80.0, 80.0]}},
          "cyc12": {"prior": {"1e-3": [5.0, 6.0, 7.0]}, "none": {"1.6e-2": [3.0, 3.0, 4.0], "3.2e-2": [4.0, 4.0, 4.0]}},
          "d6": {"prior": {"1e-3": [1.0, None, None]}, "none": {"1.6e-2": [4.0, 4.0, 4.0], "3.2e-2": [5.0, 5.0, 5.0]}}}
    T1 = _world_T(WT)
    code, txt, o = run(R1, {}, PICKS_FAKE, T1, strict=False, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    rw = {r["cand"]: r for r in o["table"]}
    el = {c for c, r in rw.items() if r["eligible"]}
    ok("W1 eligible set = {ranks, torus}", el == {"ranks", "torus"}, el)
    ok("W1 ranks: p 1e-3, n 2e-3 (the none argmax, not the top lr 4e-3), 8 seeds, mean g +3.00, wins 8/8, E3 ok",
       (rw["ranks"]["p"], rw["ranks"]["n"], rw["ranks"]["k"], round(rw["ranks"]["mean_g"], 6), rw["ranks"]["wins"], rw["ranks"]["E"]["E3"])
       == ("1e-3", "2e-3", 8, 3.0, 8, True), rw["ranks"])
    g = np.array([3, 2, 4, 3, 2, 3, 4, 3.0])
    ok("W1 ranks: d = mean / sd(ddof 1)", abs(float(rw["ranks"]["d"]) - g.mean() / g.std(ddof=1)) < 1e-9, rw["ranks"]["d"])
    ok("W1 prefixes: n 4e-3 = the highest none lr present (< 3.2e-2) -> E3 NO, the only failure",
       rw["prefixes"]["n"] == "4e-3" and not rw["prefixes"]["E"]["E3"] and all(rw["prefixes"]["E"][k] for k in ("E1", "E2", "E4", "E5")), rw["prefixes"])
    ok("W1 compass: E4 ok (mean > 0, wins 7/8) but d < 0.5 -> E5 NO",
       rw["compass"]["E"]["E4"] and not rw["compass"]["E"]["E5"] and rw["compass"]["d"] < 0.5 and rw["compass"]["wins"] == 7, rw["compass"])
    ok("W1 landmarks: FLOOR from a heuristic line (90 + 5 = 95; means 90 / 88) -> E1 NO, the only failure",
       not rw["landmarks"]["E"]["E1"] and "FLOOR" in rw["landmarks"]["why"]["E1"] and all(rw["landmarks"]["E"][k] for k in ("E2", "E3", "E4", "E5")),
       rw["landmarks"])
    ok("W1 scalar_adj: CEILING (none mean >= 95) -> E1 NO", not rw["scalar_adj"]["E"]["E1"] and "CEILING" in rw["scalar_adj"]["why"]["E1"],
       rw["scalar_adj"]["why"])
    ok("W1 torus: I1 drops the seed-0-only cells (none@8e-3 s0 = 99, only-torus@2e-3 s0 = 95): p 1e-3, n 3.2e-2, 3 seeds",
       (rw["torus"]["p"], rw["torus"]["n"], rw["torus"]["k"]) == ("1e-3", "3.2e-2", 3) and "8e-3" not in rw["torus"]["present_none"], rw["torus"])
    ok("W1 torus: n = the highest present but = 3.2e-2 -> E3 ok", rw["torus"]["E"]["E3"], rw["torus"]["why"]["E3"])
    ok("W1 hcube: 2 paired seeds -> E2 NO", rw["hcube"]["k"] == 2 and not rw["hcube"]["E"]["E2"], rw["hcube"])
    ok("W1 cyc12: FLOOR (both <= chance 8.3 + 5) -> E1 NO", not rw["cyc12"]["E"]["E1"] and "FLOOR" in rw["cyc12"]["why"]["E1"], rw["cyc12"]["why"])
    ok("W1 d6: one paired seed and no prior over none -> out", not rw["d6"]["eligible"] and rw["d6"]["k"] == 1, rw["d6"])
    ok("W1 caesar / box5 without runs -> out with 'no only-<d> run'", "no only-caesar run" in rw["caesar"]["reason"], rw["caesar"]["reason"])
    ok("W1 picks: torus (d inf, constant g) then ranks; no fallback",
       o["picks"] == ["torus", "ranks"] and o["fallbacks_used"] == [], (o["picks"], o["fallbacks_used"]))
    ok("W1 V2 triggers (picked torus has n = 3.2e-2)", o["extra_none_6.4e-2"] is True and o["v2_by"] == ["torus"], o["v2_by"])
    ok("W1 composition = dim, cities, countries + picks in the given order", o["composition"] == ["dim", "cities", "countries", "ranks", "torus"],
       o["composition"])
    ok("W1 report has every row and E-column", all(f"{c:10s}" in txt for c in CAND_R + CAND_T) and "E1 E2 E3 E4 E5" in txt, txt[:400])
    # torus g is constant (10, 10, 10)? 90-80, 88-78, 91-79 = 10, 10, 12 -> finite d; check the ordering by d explicitly
    ok("W1 ranking sorted by d", [x[0] for x in o["ranking"]] == sorted([x[0] for x in o["ranking"]], key=lambda c: -rw[c]["d"]), o["ranking"])

    # ---- world 2: a tie on d decided by the larger mean g; an eligible n = 3.2e-2 candidate not picked -> no V2
    v = [1, 2, 3, 2, 1, 2, 3, 2]
    WR2 = {"ranks": {"prior": {"1e-3": add(b8, v)}, "none": {"2e-3": b8, "4e-3": [70.0] * 8}},
           "scalar_adj": {"prior": {"1e-3": add(b8, [2 * x for x in v])}, "none": {"2e-3": b8, "4e-3": [70.0] * 8}},
           "compass": {"prior": {"1e-3": add(b8, [5, 6, 5, 6, 5, 6, 5, 6])}, "none": {"2e-3": b8, "4e-3": [70.0] * 8}}}
    WR2["prefixes"] = {"prior": {"1e-3": add(b8, [5, 6, 5, 6, 5, 6, 5, 6])}, "none": {"2e-3": b8, "4e-3": [70.0] * 8}}
    WR2.pop("prefixes")
    WT2 = {"hcube": {"prior": {"1e-3": [81.0, 80.5, 83.0]}, "none": {"1.6e-2": [70.0] * 3, "3.2e-2": [80.0] * 3}}}
    _, txt2, o2 = run(_world_R(WR2), {}, PICKS_FAKE, _world_T(WT2), strict=False, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    r2 = {r["cand"]: r for r in o2["table"]}
    ok("W2 ranks and scalar_adj tie on d (g scaled x2)", abs(r2["ranks"]["d"] - r2["scalar_adj"]["d"]) < 1e-9, (r2["ranks"]["d"], r2["scalar_adj"]["d"]))
    ok("W2 compass highest d; the tie goes to scalar_adj (larger mean g); ranks out of the two",
       o2["picks"] == ["compass", "scalar_adj"], o2["picks"])
    ok("W2 the tie is printed", any("tie on d" in t and "larger mean g" in t for t in o2["ties"]) and "tie on d" in txt2, o2["ties"])
    ok("W2 hcube eligible with n = 3.2e-2 but not picked -> V2 not triggered",
       r2["hcube"]["eligible"] and r2["hcube"]["n"] == "3.2e-2" and o2["extra_none_6.4e-2"] is False, (r2["hcube"]["reason"], o2["v2_by"]))

    # ---- world 3: fewer than two eligible -> fallbacks in order
    WR3 = {"ranks": {"prior": {"1e-3": add(b8, v)}, "none": {"2e-3": b8, "4e-3": [70.0] * 8}}}
    _, _, o3 = run(_world_R(WR3), {}, PICKS_FAKE, {}, strict=False, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    ok("W3 one eligible -> + people", o3["picks"] == ["ranks"] and o3["fallbacks_used"] == ["people"]
       and o3["composition"] == ["dim", "cities", "countries", "people", "ranks"], o3)
    _, txt4, o4 = run({}, {}, PICKS_FAKE, {}, strict=False, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    ok("W3 none eligible -> people, months", o4["picks"] == [] and o4["fallbacks_used"] == ["people", "months"]
       and set(o4["composition"]) == {"dim", "cities", "countries", "people", "months"} and not o4["extra_none_6.4e-2"], o4["composition"])
    ok("W3 fallback reasons printed", "people (F102 HELPS 8/8, FLOOR)" in txt4 and "months (F101 PASS" in txt4, txt4[-600:])
    # E4 alone: wins < 2/3
    WR5 = {"ranks": {"prior": {"1e-3": add(b8, [6, 6, 6, 6, -1, -1, -1, 1])}, "none": {"2e-3": b8, "4e-3": [70.0] * 8}}}
    _, _, o5 = run(_world_R(WR5), {}, PICKS_FAKE, {}, strict=False, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    r5 = {r["cand"]: r for r in o5["table"]}["ranks"]
    ok("W3 wins 5/8 < 2/3 -> E4 NO", r5["wins"] == 5 and not r5["E"]["E4"] and not r5["eligible"], r5["why"]["E4"])

    # ---- world 4: pick-time completeness (toy3 replication derivation, Part R2 seed 15, strict exit 2)
    S0 = {("none", "2e-3", 0): {"torus": 15.0, "hcube": 60.0, "cyc12": 0.5, "box5": 12.0, "caesar": 0.4, "d6": 1.0},
          ("none", "4e-3", 0): {"torus": 18.0, "hcube": 70.0, "cyc12": 0.4, "box5": 9.0, "caesar": 0.2, "d6": 1.2},
          ("none", "8e-3", 0): {"torus": 46.0, "hcube": 65.0, "cyc12": 1.2, "box5": 11.0, "caesar": 0.5, "d6": 4.0},
          ("none", "1.6e-2", 0): {"torus": 74.0, "hcube": 69.0, "cyc12": 5.8, "box5": 20.0, "caesar": 2.3, "d6": 9.3},
          ("only-torus", "1e-3", 0): {"torus": 83.0}, ("only-torus", "2e-3", 0): {"torus": 80.0},
          ("only-hcube", "1e-3", 0): {"hcube": 72.0}, ("only-hcube", "2e-3", 0): {"hcube": 75.0},
          ("only-cyc12", "1e-3", 0): {"cyc12": 2.7}, ("only-cyc12", "2e-3", 0): {"cyc12": 2.8},
          ("only-box5", "1e-3", 0): {"box5": 15.0}, ("only-box5", "2e-3", 0): {"box5": 18.0},
          ("only-caesar", "1e-3", 0): {"caesar": 1.0}, ("only-caesar", "2e-3", 0): {"caesar": 1.5},
          ("only-d6", "1e-3", 0): {"d6": 2.0}}
    T4 = _fake_T(S0)
    req, info, undet = repl_required(T4)
    want = {("only-torus", "1e-3", 1), ("only-torus", "1e-3", 2), ("only-hcube", "2e-3", 1), ("only-hcube", "2e-3", 2),
            ("none", "4e-3", 1), ("none", "4e-3", 2)} | set(REPL_SHARED)
    ok("W4 replication: torus@1e-3 and hcube@2e-3 qualify; hcube's seed-0 best none 4e-3 adds none@4e-3 s1, s2; shared none cells; "
       "cyc12 FLOOR, box5 prior <= none, caesar FLOOR out", req == want, sorted(req ^ want))
    ok("W4 d6 undetermined (only-d6@2e-3 s0 missing)", [d for d, _ in undet] == ["d6"] and info["d6"]["status"] == "UNDETERMINED", undet)
    T4f = dict(T4)
    T4f[("only-d6", "2e-3", 0)] = {"status": "VALID", "n": {"d6": {"area": 5.0, "tf_area": 5.0, "chance": 100 / 12, "chance_u": 100 / 12, "bpb": 1.0}}}
    for c in want:
        T4f[c] = {"status": "VALID", "n": {d: {"area": 50.0, "tf_area": 50.0, "chance": 5.0, "chance_u": 5.0, "bpb": 1.0} for d in CAND_T}}
    T4f[("only-hcube", "2e-3", 2)] = {"status": "UNFINISHED", "n": {}}
    R4 = _world_R(WR3)
    code_s, txt_s, o_s = run(R4, {}, PICKS_FAKE, T4f, strict=True, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    ok("W4 strict: an unfinished replication run -> exit 2, names it, no pick", code_s == 2 and o_s is None
       and "toy3 replication only-hcube@2e-3 s2: UNFINISHED" in txt_s, txt_s)
    code_p, txt_p, o_p = run(R4, {}, PICKS_FAKE, T4f, strict=False, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    ok("W4 --allow_partial: exit 0, marked PARTIAL", code_p == 0 and o_p["complete"] is False and "NOT the pick" in txt_p, txt_p[:300])
    # coordinator 2026-11-19 (audit_pick_pt6v2.md important 1 / 3): a first-attempt INVALID required cell refuses; after a
    # re-run (kept dir -r1) a second INVALID is excused
    T4f[("only-hcube", "2e-3", 2)] = {"status": "INVALID", "rr": 0, "note": "planted", "n": {}}
    code_i, txt_i, o_i = run(R4, {}, PICKS_FAKE, T4f, strict=True, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    ok("W4 strict: an INVALID first-attempt replication run -> exit 2, asks for one re-run", code_i == 2 and o_i is None
       and "re-run it once" in txt_i, txt_i[:400])
    T4f[("only-hcube", "2e-3", 2)] = {"status": "INVALID", "rr": 1, "note": "planted", "n": {}}
    code_i, _, o_i = run(R4, {}, PICKS_FAKE, T4f, strict=True, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    ok("W4 strict: an INVALID re-run (-r1) replication run is excused", code_i == 0 and o_i["complete"], o_i and o_i["problems"])
    # Part R2: ranks s15 missing
    R4m = {k: v for k, v in R4.items() if not (k[1] == "only-ranks" and k[3] == 15)}
    mk = lambda att: (lambda name: att if name == "ptdom2-B2-only-ranks-lr1e-3-s15" else [])
    for att, st, want_code, label in (([(0, "1"), (1, "2")], {"1": "FAILED", "2": "TIMEOUT"}, 0, "failed twice -> excused"),
                                      ([(0, "1"), (1, "2")], {"1": "CANCELLED", "2": "PENDING"}, 2, "second attempt pending -> refused"),
                                      ([(0, "1")], {"1": "FAILED"}, 2, "failed once -> refused"),
                                      ([(0, "1"), (1, None)], {"1": "FAILED"}, 2, "an attempt without a job id -> refused")):
        c_, t_, o_ = run(R4m, {}, PICKS_FAKE, T4f, strict=True, markers_fn=mk(att), states_fn=lambda ids, st=st: st, comp_fn=fixed)
        ok(f"W4 Part R2 seed 15 missing, {label}", c_ == want_code and ("EXCUSED" in (t_ if o_ is None else "\n".join(o_["excused"]))) == (want_code == 0),
           t_[:500])
    pk_bad = json.loads(json.dumps(PICKS_FAKE))
    pk_bad["domains"]["ranks"]["status"] = "NOT FINAL"
    c_, t_, _ = run(R4, {}, pk_bad, T4f, strict=True, markers_fn=nos()[0], states_fn=nos()[1], comp_fn=fixed)
    ok("W4 strict: a candidate no longer FINAL in the picks file -> refused", c_ == 2 and "ranks status NOT FINAL" in t_, t_)
    # composition order from make_jobs_pt6v2 (or its fallback) is accepted shape
    comp, src = comp_order(["torus", "dim", "ranks", "cities", "countries"])
    ok("W5 comp_order: trainer order, 5 distinct", len(comp) == 5 and comp.index("dim") < comp.index("cities") < comp.index("ranks") < comp.index("torus"),
       (comp, src))
    Tm = _fake_T({("none", "1.6e-2", 1): {"torus": 70.0}, ("only-hcube", "1e-3", 2): {"torus": 1.0}})
    Tm[("only-hcube", "1e-3", 2)]["marks"] = ["hcube: twin-free read-out unavailable: AssertionError()"]
    tm = t_marks(Tm)
    ok("W6 toy3 marks: a usable run without its primary value is listed with its readout_ptdom3 mark",
       len(tm) == 2 and any("only-hcube@1e-3 s2: no primary value for hcube (hcube: twin-free" in x for x in tm), tm)
    print(f"SELFTEST {'PASS' if n_ok == len(T_) else 'FAIL'} {n_ok}/{len(T_)}")
    return 0 if n_ok == len(T_) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    R_ = R2.load_all()
    picks = json.loads(PICKS.read_text())
    subs, sprobs = build_subs(R_)
    runs3 = load3()
    inp = {**inputs_shas(), "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "n_ptdom2_records": len(R_), "n_ptdom3_runs": len(runs3), "sub_problems": len(sprobs)}
    code, text, obj = run(R_, subs, picks, runs3, strict=not a.allow_partial, inputs=inp)
    print(text)
    if sprobs:
        print(f"note: {len(sprobs)} comparison sub-read-out problem(s) (the FLOOR line's shortcut baseline may be missing for those runs): "
              + "; ".join(sprobs[:3]))
    if a.json and obj is not None:
        conv = lambda v: v.tolist() if isinstance(v, np.ndarray) else (float(v) if isinstance(v, np.floating) else str(v))
        Path(a.json).write_text(json.dumps(obj, indent=1, default=conv))
    sys.exit(code)


if __name__ == "__main__":
    main()
