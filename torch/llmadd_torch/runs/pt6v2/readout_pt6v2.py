#!/usr/bin/env python3
"""pt6v2 read-out (runs/pt6v2/AUDIT.md "Verdict rule"; phase-2a notes in AUDIT_phase2a.md).

Composition: runs/pt6v2/COMPOSITION.json when it exists, else make_jobs_pt6v2.PLACEHOLDER (numbers + dim, cities, countries,
compass, hcube). Seeds 16-23. Cells: all6@1e-3 (P), all6@2e-3, none@{2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2} (4e-3: AUDIT.md Amendment V1),
only-<m>@2e-3.

Each domain's primary read-out = its source batch's, imported read only and sha-asserted (SRC):
  numbers, s4 (s4_hard), dim (strict), cities / countries / people: readout_pt6comb.series_of; FLOOR = readout_pt6comb.floor_line
    (per run; the line = the mean over the none@N_m runs);
  months + Part R (stars, landmarks, compass, prefixes, ranks, scalar_adj): metrics[readout_ptdom.READ[d]] (= readout_ptdom2.load2's
    area / final3); FLOOR = readout_ptdom2.floor_tags (max of the mean lines + 5) on adapter records;
  toy3: readout_ptdom3.dom_numbers; PRIMARY = the twin-free set for hcube and d6, the full set for cyc12 / torus / box5 / caesar;
    d6 prints its AAE split; caesar is printed in a separate block with readout_ptdom3.CAESAR_NOTE (it counts in the headline);
    FLOOR = both arms <= the none runs' chance (twin-free: 100 / n_items) + 5.
Verdict (AUDIT.md, unchanged): N_m = the none lr with the highest seed-mean of m's primary area (a tie -> the higher lr);
g = all6@1e-3 - none@N_m; HELPS iff one-sided paired t p < 0.05 and g > 0 in >= 6 of 8 seeds; A2 interference at the same lr 2e-3
(h = all6@2e-3 - only-m@2e-3, u = only-m@2e-3 - none@2e-3) and the HELPS rule = readout_pt6comb.manifold_verdict (inherited as is).
Flags, never conditions: 'none at its top lr 3.2e-2' (N_m = 3.2e-2), 'none at its lowest lr 2e-3' (N_m = 2e-3; V1), 'prior at its
lowest lr' (always). Descriptive: g vs every none lr and vs the mean6-picked none; a NOT SHOWN manifold's line starts with the
same-lr gain all6@2e-3 - none@2e-3 and u (V1 item 2). A3: AUX GATE FAILED at all6@2e-3 / an incomplete pair -> interference words
WITHHELD. The report prints the two qualifier lines of V1 item 4 (N_m picked on the same seeds; pt6comb's 'reading' line).
GATE (A2.1) per half (16-19, 20-23) at all6@1e-3. Headline: k of 6 HELPS with interference not detected; Holm a reference column.
Device per seed + a 'mixed devices' note (never a condition). Strict: a missing / invalid core run or a problem -> REFUSED;
--allow_partial prints PROVISIONAL.

Amendment V3 (post-hoc, user-approved 2026-11-19; runs/pt6v2/POSTHOC_V3.json): none@6.4e-2 at seeds 16-23 outside CELLS.json.
Without --posthoc_none these runs are left out of every number (the registered read-out, the same with or without them on disk);
--posthoc_none 6.4e-2 prints the registered block unchanged, then block 3 (the verdict on the grid + 6.4e-2, I2 rules for the extra
lr; a missing run of it -> that block PROVISIONAL only); --json puts it under the key "posthoc_V3".

Usage:  readout_pt6v2.py [--root DIR] [--allow_partial] [--json OUT] [--posthoc_none 6.4e-2]
        readout_pt6v2.py --canary OUT_DIR       (AUDIT.md "Pipeline canary": a number for each of the 6 manifolds)
        readout_pt6v2.py --selftest             (test 4: planted worlds + mutations)
        readout_pt6v2.py --known                (test 5: known answers on real pt6comb / ptdom2 / ptdom3 runs)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse          # noqa: E402
import hashlib           # noqa: E402
import importlib.util    # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import re                # noqa: E402
import shutil            # noqa: E402
import subprocess        # noqa: E402  (x_sacct: read-only sacct for I2 'failed twice')
import tempfile          # noqa: E402
from functools import lru_cache   # noqa: E402
from pathlib import Path          # noqa: E402

import numpy as np       # noqa: E402

HERE = Path(__file__).resolve().parent
B = Path("/scratch/yh6210/llmadd_torch")
V2 = B / "runs" / "pt6v2"
SRC = {"readout_pt6comb": (B / "runs/pt6comb/readout_pt6comb.py", "05ea474d9c9b"),
       "readout_ptdom2": (B / "runs/ptdom2/readout_ptdom2.py", "3f78507d5d47"),
       "readout_ptdom3": (B / "readout_ptdom3.py", "9abd89ad75a7"),
       "readout_ptdom": (B / "runs/ptdom/readout_ptdom.py", "1e2f17522e47")}


def sha12(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


for _n, (_p, _s) in SRC.items():
    assert sha12(_p) == _s, f"source read-out {_p} changed: sha256[:12] {sha12(_p)} != {_s} (AUDIT_phase2a.md)"
MJ2_FILE, MJ2_SHA = B / "make_jobs_pt6v2.py", "bb15d248afd8"   # frozen after the V2 build (AUDIT.md "V2 build"; V1: 2f10290f95b2)
assert sha12(MJ2_FILE) == MJ2_SHA, f"{MJ2_FILE} changed: sha256[:12] {sha12(MJ2_FILE)} != {MJ2_SHA} (AUDIT.md Amendment V1: frozen)"
for _p in (B, B / "runs/ptdom", B / "runs/ptdom2", B / "runs/pt6comb"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import readout_ptdom as RO       # noqa: E402
import readout_pt6comb as PC     # noqa: E402
import readout_ptdom2 as P2      # noqa: E402
import readout_ptdom3 as P3      # noqa: E402
import make_jobs_pt6v2 as MJ2    # noqa: E402  (read only; owned by the phase-2b builder: names, cells, composition)

for _n, (_p, _s) in SRC.items():
    assert Path(sys.modules[_n].__file__).resolve() == _p.resolve(), f"{_n} imported from {sys.modules[_n].__file__}, not {_p}"
assert Path(MJ2.__file__).resolve() == MJ2_FILE.resolve(), f"make_jobs_pt6v2 imported from {MJ2.__file__}, not {MJ2_FILE}"

PREFIX = MJ2.PREFIX
SEEDS, HALVES, P_LR, SAME_LR, NONE_LRS, CANARY = MJ2.SEEDS, MJ2.HALVES, MJ2.P_LR, MJ2.SAME_LR, MJ2.NONE_LRS, MJ2.CANARY
assert PREFIX == "pt6v2" and SEEDS == tuple(range(16, 24)) and HALVES == ((16, 17, 18, 19), (20, 21, 22, 23))
assert P_LR == "1e-3" and SAME_LR == "2e-3" and NONE_LRS == ("2e-3", "4e-3", "8e-3", "1.6e-2", "3.2e-2")   # Amendment V1
TOP_LR, BOT_LR = "3.2e-2", "2e-3"
assert TOP_LR == max(NONE_LRS, key=float) and BOT_LR == min(NONE_LRS, key=float)
EXTRA_NONE = MJ2.EXTRA_NONE_OK
assert EXTRA_NONE == ("6.4e-2",)   # AUDIT.md Amendment V2 (conditional, opt-in)
CELLS_FILE = MJ2.CELLS_FILE


def none_grid():
    """The none lr grid of the written batch: CELLS.json (make_jobs_pt6v2.py --write) or, without it, the default 5-lr grid."""
    if not CELLS_FILE.exists():
        return NONE_LRS
    g = tuple(json.loads(CELLS_FILE.read_text())["none_lrs"])
    assert g in (NONE_LRS,) + tuple(NONE_LRS + (x,) for x in EXTRA_NONE), f"{CELLS_FILE}: none grid {g} is not a registered grid"
    return g


POSTHOC_FILE = V2 / "POSTHOC_V3.json"   # AUDIT.md Amendment V3 (post-hoc, user-approved 2026-11-19): none@6.4e-2, seeds 16-23
V3_TAG = "extended grid (post-hoc V3, user-approved 2026-11-19)"
V3_RULE = ("reading rule (AUDIT.md Amendment V3): the registered verdict above stays the registered one; this extended verdict is "
           "reported beside it; a manifold whose extended g changes sign or loses HELPS: report both, the paper wording follows the "
           "extended one")


def posthoc_lrs(path=None):
    """The none lrs that POSTHOC_V3.json adds outside the CELLS.json grid (() without the file)."""
    p = Path(path or POSTHOC_FILE)
    if not p.exists():
        return ()
    j = json.loads(p.read_text())
    assert j.get("amendment") == "V3" and j.get("arm") == "none" and j.get("lr") in EXTRA_NONE, f"{p}: not an Amendment V3 record"
    return (j["lr"],)


def split_posthoc(R_, ph_lrs, grid):
    """-> (the registered records, the post-hoc ones). A post-hoc record = none at an lr of ph_lrs outside the CELLS.json grid; flag
    off, readout() drops them before every registered number (load, problems, NOT VALID)."""
    ph = {k for k in R_ if k[0] == "none" and k[1] in ph_lrs and k[1] not in grid}
    return {k: r for k, r in R_.items() if k not in ph}, {k: R_[k] for k in sorted(ph)}


def posthoc_problems(R_all, R_reg):
    """The integrity problems that only the post-hoc runs bring (Amendment V3: they make the extended block PROVISIONAL)."""
    base = set(PC.integrity(R_reg, set(SEEDS)))
    return [p for p in PC.integrity(R_all, set(SEEDS)) if p not in base]


def top_flag(lr):
    return f"none at its top lr {lr}"
N_EVALS = PC.N_EVALS
ALPHA, MIN_WINS = PC.ALPHA, PC.MIN_WINS
assert (N_EVALS, ALPHA, MIN_WINS, PC.INT_FRAC) == (75, 0.05, 6, 0.5)
PRIMARY_TF = tuple(P3.PRIMARY_TF)
TOY3 = tuple(P3.TOY3)
SEPARATE = ("caesar",)
TOP_FLAG = "none at its top lr 3.2e-2"
BOT_FLAG = "none at its lowest lr 2e-3"   # Amendment V1 item 1 (label only)
LOW_FLAG = "prior at its lowest lr"
I2_FLAG = "I2: N_m not valid at a seed:"   # Amendment V2 item I2 (a flag, never a condition)
assert TOP_FLAG == top_flag(TOP_LR)
# Tier L (AUDIT.md "V2 build"): the prior's target carries the answers, so a HELPS reads "given the structure, the model can use it".
# Sources: ptdom AUDIT 145-156 (anchors yes, months no), ptdom2 AUDIT 81 (Part R six + countries / people / cities), ptdom3 AUDIT 14-24
# (toy3: all yes) + tests/byteid_23439548.log (caesar), pt6comb AUDIT 232-235 (s4 / dim; the tied-row caveat).
TIER_L = {"s4": True, "dim": True, "cities": True, "countries": True, "people": True, "months": False,
          "stars": True, "landmarks": True, "compass": True, "prefixes": True, "ranks": True, "scalar_adj": True,
          "cyc12": True, "torus": True, "hcube": True, "d6": True, "box5": True, "caesar": True}
NONCE_TIED = ("s4", "dim", "cyc12", "torus", "hcube", "d6", "box5")
assert set(TIER_L) == set(MJ2.ALLOWED) and set(NONCE_TIED) <= set(TIER_L)
Q_SAME_SEEDS = ("qualifier: N_m is picked on the same seeds that g is computed on (the none lr with the highest mean of these seeds), "
                "which favours none; the rule is conservative only within the none grid (see the none-at-its-lowest / top-lr flags)")
Q_READING = ("reading: comparisons and s4 / dim are Tier L (the targets carry the answers); s4 / dim items are single nonce tokens "
             "whose site is the read-out candidate's own (tied) embedding row (AUDIT 'Caveats'); NOT SHOWN / INTERFERENCE at 8 seeds are "
             "statements about this rule, never 'no effect' / 'harm' in general; 'interference not detected' is never 'shown not to "
             "interfere' (A2)")   # = runs/pt6comb/readout.txt's reading line, verbatim (V1 item 4)
Q_TAIL = ("NOT SHOWN / INTERFERENCE at 8 seeds are statements about this rule, never 'no effect' / 'harm' in general; 'interference "
          "not detected' is never 'shown not to interfere' (A2)")
assert Q_READING.endswith("; " + Q_TAIL)


def reading_line(mani):
    """The composition's Tier L reading line (V2 build; pt6comb's line generalised to every ALLOWED domain, same tail)."""
    doms = [d for d in mani if d != "numbers"]
    tl, nl, nt = ([d for d in doms if TIER_L.get(d)], [d for d in doms if TIER_L.get(d) is False], [d for d in doms if d not in TIER_L])
    s = ("reading: comparisons" + "".join(f", {d}" for d in tl) + " are Tier L (the prior's target carries the answers: a HELPS "
         "reads 'given the structure, the model can use it')")
    if nl:
        s += f"; {' / '.join(nl)} not Tier L (the target does not carry the answers: a HELPS there is not that reading)"
    if nt:
        s += f"; {' / '.join(nt)}: Tier L not registered (not a pt6v2 candidate)"
    nz = [d for d in doms if d in NONCE_TIED]
    if nz:
        s += f"; {' / '.join(nz)} items are single nonce tokens whose site is the read-out candidate's own (tied) embedding row (AUDIT 'Caveats')"
    if "caesar" in doms:
        s += "; caesar items are real letter tokens (its own block)"
    return s + "; " + Q_TAIL
GATE_TXT = ("all6 GATE (the LM share a / (a + b) of all6@1e-3 < 0.5 on seeds 16-19 or 20-23: the prior is not an auxiliary loss "
            "there): no HELPS / interference words")
ALLOWED_FAILS = (RO.CKA_CHECK, PC.GEO_FLAG6) + tuple(RO.DP_FLAG_CHECKS)
assert set(P3.FLAGC) == set(RO.DP_FLAG_CHECKS) | {RO.CKA_CHECK}, "ptdom3's flag-only checks != pt6comb's"
assert MJ2.M6.GEO_NAME6 == PC.M6.GEO_NAME6 and (P3.GEO_TOL, P3.GEO_FLOOR, P3.CKA_MIN, P3.CKA_MAX) == (
    PC.GEO_TOL, PC.GEO_FLOOR, PC.GEO_CKA_MIN, PC.GEO_CKA_MAX)
_num = PC._num
mean = lambda xs: float(np.mean(xs)) if len(xs) else None   # noqa: E731


def route(d):
    """-> 'pc' (pt6comb-B read-out), 'ro' (months / Part R: readout_ptdom READ), 'p3' (toy3: readout_ptdom3)."""
    if d == "numbers" or MJ2.SOURCE.get(d) == "pt6comb-B":
        return "pc"
    if d in TOY3:
        return "p3"
    assert d in MJ2.SOURCE, f"{d}: no source batch"
    return "ro"


def primary_set(d):
    r = route(d)
    if r == "pc":
        return {"numbers": "held b11..b33", "s4": "s4_hard", "dim": "dim strict"}.get(d, "held 2-way")
    if r == "ro":
        return RO.READ[d]
    return f"{P3.READSET[d]}" + (" twin-free" if d in PRIMARY_TF else "")


def composition(path=None):
    """-> (the five non-number domains, source text)."""
    f = Path(path) if path else MJ2.COMP_FILE
    if f.exists():
        J = json.loads(f.read_text())
        return MJ2.comp_doms(J["domains"]), f"{f.name} (by {J.get('by')})"
    return MJ2.comp_doms(MJ2.PLACEHOLDER), "PLACEHOLDER (no COMPOSITION.json yet)"


# ------------------------------------------------------------------------------------------------------------ per-domain metrics
@lru_cache(maxsize=None)
def meta1(d):
    return RO.domain_meta([d])[d]


def _x_cmp(r, d):
    """readout_pt6comb.run_metrics' per-question record of a 2-way domain (for readout_ptdom.run_subsets)."""
    c, m_, D_ = r["curve"], r["metrics"], r["domains"][d]
    eh = D_.get("eval_held")
    return {"q": eh["q"], "nctx": eh["contexts"], "masks": eh.get("masks", {}), "split_seed": D_.get("split_seed"),
            "table_sha": D_.get("table", {}).get("sha"), "heur": [D_.get("heuristic_ansfreq"), D_.get("heuristic_shorter")],
            "hits": [e[f"hits2_{d}_held"] for e in c], "logged": [e.get(f"{d}_held_2way") for e in c],
            "logged_hard": [e.get(f"{d}_held_2way_hard") for e in c], "area": m_[RO.READ[d]]["area"],
            "area_hard": m_.get(f"{d}_readout_hard", {}).get("area")}


def primary_of(d, n):
    """toy3: the primary (area, final3) from readout_ptdom3.dom_numbers' dict: twin-free for PRIMARY_TF, else the full set."""
    return (n.get("tf_area"), n.get("tf_final3")) if d in PRIMARY_TF else (n.get("area"), n.get("final3"))


def dom_metrics(r, d):
    """-> (dict, problems): manifold d's primary series / area / final3 on run r, its source's FLOOR inputs, secondary numbers."""
    c, m_, probs = r["curve"], r["metrics"], []
    o = {"route": route(d), "series": None, "area": None, "final3": None, "floor": None, "fl": None, "p3": None, "sec": {}}
    if o["route"] == "pc":
        s = PC.series_of(r, d)
        if len(s) != len(c):
            probs.append(f"{d}: {len(s)} series values for {len(c)} evals")
        o["series"], o["area"], o["final3"] = s, (100 * mean(s) if s else None), (100 * mean(s[-3:]) if s else None)
        ref = m_["numbers"]["area"] if d == "numbers" else (m_[RO.READ[d]]["area"] if d in PC.CMPD else None)
        if ref is not None and abs(o["area"] - ref) > 1e-6:
            probs.append(f"{d} area {o['area']} != metrics {ref}")
        if d in ("s4", "dim"):
            mk = {"s4": "fresh_nonid", "dim": "no_negtwin_nonid"}[d]
            a_ = PC.hits_area(r, d, mk)
            if abs(a_ - m_[f"{d}_readout"]["area"]) > 1e-6:
                probs.append(f"{d}_readout area re-derived from the hits {a_} != metrics {m_[f'{d}_readout']['area']}")
            o["sec"][f"{d}_readout"] = m_[f"{d}_readout"]["area"]
        if d == "s4":
            h = PC.RS.s4_hard(r)
            pre_ = PC.RS.S4_HARD_PRE.get(r["args"]["seed"])
            if pre_ is not None and ((h["n"], h["n_readout"]) != pre_[:2] or abs(h["chance"] - pre_[2]) > 1e-9):
                probs.append(f"s4_hard ({h['n']}, {h['n_readout']}, {h['chance']}) != rb6's precomputation {pre_}")
        sub = None
        if d in PC.CMPD:
            sub, p_ = RO.run_subsets(_x_cmp(r, d), meta1(d))
            probs += [f"{d} sub-read-out: {v}" for v in p_]
            o["sec"]["hard_area"] = m_.get(f"{d}_readout_hard", {}).get("area")
        o["floor"] = PC.floor_line(r, d, sub)
    elif o["route"] == "ro":
        mm = m_.get(RO.READ[d]) or {}
        s = [e.get(RO.EVAL_KEY[d]) for e in c]
        o["series"], o["area"], o["final3"] = s, mm.get("area"), mm.get("final3")
        if not s or not all(_num(v) for v in s) or not _num(o["area"]):
            probs.append(f"{d}: read-out {RO.READ[d]} / {RO.EVAL_KEY[d]} missing or non-finite")
        elif abs(100 * mean(s) - o["area"]) > 1e-6:
            probs.append(f"{d}: metrics {RO.READ[d]} area {o['area']} != 100 x mean({RO.EVAL_KEY[d]}) {100 * mean(s)}")
        D_ = r["domains"][d]
        fl = {"chance": 100 * D_["chance_readout"], "heur": None, "kshort": None, "h3": None}
        if d in P2.COMP:
            fl["heur"] = [100 * D_.get("heuristic_ansfreq", 0), 100 * D_.get("heuristic_shorter", 0)]
            if D_.get("eval_held") and all(f"hits2_{d}_held" in e for e in c):
                sub, p_ = RO.run_subsets(_x_cmp(r, d), meta1(d))
                probs += [f"{d} sub-read-out: {v}" for v in p_]
                fl["h3"] = sub["h3_held"] if sub else None
            o["sec"]["hard_area"] = m_.get(f"{d}_readout_hard", {}).get("area")
        if d in P2.KSTEP and D_.get("heuristic_shortest_kspan") is not None:
            fl["kshort"] = 100 * D_["heuristic_shortest_kspan"]
        o["fl"] = fl
    else:
        n, marks = P3.dom_numbers(r, d)
        probs += marks
        o["p3"] = n
        o["area"], o["final3"] = primary_of(d, n)
        o["series"] = [x for _, x in P3.series(r, d)] if d not in PRIMARY_TF else None
        if not _num(o["area"]):
            probs.append(f"{d}: primary read-out ({primary_set(d)}) unavailable")
        o["sec"] = {"full_area": n.get("area"), "tf_area": n.get("tf_area"), "n_tf": n.get("n_tf"), "n_twin": n.get("n_twin")}
        if d == "d6":
            o["sec"].update(aae_area=n.get("aae_area"), noaae_area=n.get("noaae_area"), n_aae=n.get("n_aae"))
    return o, probs


def run_metrics_v2(r, mani):
    out = {k: {} for k in ("series", "area", "final3", "floor", "fl", "p3", "sec", "cka_win")}
    probs = []
    for m in mani:
        try:
            o, p = dom_metrics(r, m)
        except Exception as ex:   # noqa: BLE001  (a problem: the strict read-out refuses)
            o, p = None, [f"{m}: read-out failed: {ex!r}"[:240]]
        probs += p
        for k in ("series", "area", "final3", "floor", "fl", "p3", "sec"):
            out[k][m] = o[k] if o else None
    c = r["curve"]
    on = [e for e in c if e.get("prior_on")]
    we = on[-1] if on else next((e for e in c if e["step"] >= 7250), c[-1])
    for m in mani:
        out["cka_win"][m] = we.get("cka_ops_helix3") if m == "numbers" else we.get(f"cka_{m}")
    out["steps"] = [e["step"] for e in c]
    return out, probs


# ------------------------------------------------------------------------------------------------------------ loading / validity
def spec_v2(doms, none_lrs=NONE_LRS):
    return {"prefix": PREFIX, "mani": MJ2.manifolds(doms), "pri_of": lambda a: MJ2.pri_of(a, doms), "domains_arg": MJ2.domains_arg(doms),
            "tag": MJ2.tag_v2, "arms": MJ2.arms(doms), "lrs": set(none_lrs) | {P_LR, SAME_LR}, "n_evals": N_EVALS, "guard": a6_guard_v2}


def spec_pc():
    """pt6comb composition B with pt6comb's names (test 5: this loader = readout_pt6comb.load on pt6comb's runs)."""
    M6 = PC.M6
    return {"prefix": "pt6comb-B", "mani": PC.MANI, "pri_of": lambda a: M6.pri_of(a, "B"), "domains_arg": M6.domains_arg("B"),
            "tag": lambda a, lr: M6.tag6("B", a, lr), "arms": M6.arms("B"), "lrs": set(M6.LADDER), "n_evals": PC.N_EVALS,
            "guard": PC.a6_guard6}


@lru_cache(maxsize=None)
def a6_guard_v2():
    """The check names the loader matches by prefix occur once in both check scripts a pt6v2 job can render (CHECK_PY_6 without a
    toy3 domain, CHECK_PY_3 with one), geo_tol = -0.001; every rendered job file's CHECK_PY is one of them (if any exists yet)."""
    tmpl = 'chk(f"' + MJ2.M6.GEO_NAME6
    for nm, cp in (("CHECK_PY_6", MJ2.M6.CHECK_PY_6), ("CHECK_PY_3", MJ2.M3.CHECK_PY_3)):
        assert cp.count(tmpl) == 1, f"A6 guard: {nm}: the geo check template"
        miss = [p for p in PC.KEEP6 + (MJ2.M6.NUM_ON_CHECK, PC.NUM_OFF) if cp.count(f'chk("{p}') != 1]
        assert not miss, f"A6 guard: {nm}: the separate checks {miss} are not named so (once)"
    assert MJ2.MJ.GEO_TOL == PC.GEO_TOL
    sep = "<<'PY' | tee \"$OUT/check.txt\"\n"
    n = 0
    for f in sorted(V2.glob("job_*.sbatch")) + sorted((V2 / "pilot").glob("job_*.sbatch")):
        txt = f.read_text()
        if txt.count(sep) != 1:
            continue
        cp = txt.split(sep, 1)[1].split("\nPY\n", 1)[0] + "\n"
        assert cp in (MJ2.M6.CHECK_PY_6, MJ2.M3.CHECK_PY_3), f"A6 guard: {f.name}: its CHECK_PY is neither CHECK_PY_6 nor CHECK_PY_3"
        n += 1
    return n


def geo_a6_v2(r, ck, arm, spec):
    """readout_pt6comb.geo_a6_multi with the composition a parameter (test 5: = geo_a6_multi on composition B)."""
    c, why = r["curve"], []
    mani = spec["mani"]
    doms = r.get("domains", {})
    pri = tuple(spec["pri_of"](arm))
    act = tuple(x for x in mani if doms.get(x, {}).get("prior"))
    jpri = [x for x in str(r.get("args", {}).get("prior_domains") or "").split(",") if x]
    if not pri or act != pri or tuple(jpri) != pri:
        return {"why": [f"the active priors {act} / args {jpri} != the arm's {pri}"], "flag": None}
    on = [e for e in c if e.get("prior_on")]
    if not on:
        why.append("no eval with prior_on")
    gp_bad = [e["step"] for e in on if not (_num(e.get("grad_prior")) and e["grad_prior"] > 0)]
    if gp_bad:
        why.append(f"(1) grad_prior not finite > 0 at {len(gp_bad)} of {len(on)} evals with prior_on (steps {gp_bad[:5]})")
    low = []
    for d in pri:
        gk, kk = ("geo", "cka_ops_helix3") if d == "numbers" else (f"geo_{d}", f"cka_{d}")
        nf = [e["step"] for e in on if e.get(gk) is not None and not _num(e[gk])]
        if nf:
            why.append(f"{gk} not finite at {len(nf)} evals with prior_on (steps {nf[:5]}): unverifiable")
        for e in on:
            if _num(e.get(gk)) and e[gk] <= PC.GEO_TOL:
                x = {"prior": d, "step": e["step"], "geo": e[gk], "cka": e.get(kk)}
                low.append(x)
                if x["geo"] < PC.GEO_FLOOR:
                    why.append(f"(2) step {x['step']}: {gk} {x['geo']:.6f} < the floor {PC.GEO_FLOOR}")
                if not (_num(x["cka"]) and x["cka"] >= PC.GEO_CKA_MIN):
                    why.append(f"(2) step {x['step']}: {kk} {x['cka']} < {PC.GEO_CKA_MIN} (not a perfect fit)")
                elif x["cka"] > PC.GEO_CKA_MAX:
                    why.append(f"(2) step {x['step']}: {kk} {x['cka']} > {PC.GEO_CKA_MAX} (above any float overshoot: an anomaly)")
    if on and not low and not gp_bad:
        why.append(f"the check failed, but no active prior's value is <= {PC.GEO_TOL} at an eval with prior_on and grad_prior is finite > 0: "
                   "the failure is not the A6 case")
    ina = [x for x in mani if x not in pri and x != "numbers"]
    bad3 = sorted({f"geo_{x}" for e in c for x in ina if e.get(f"geo_{x}") is not None}
                  | {f"gnorm_prior_{x}" for e in c for x in ina if f"gnorm_prior_{x}" in e})
    if "numbers" not in pri and (any(e.get("geo") not in (None, 0) for e in c) or any("gnorm_prior_numbers" in e for e in c)):
        bad3.append("the inactive numbers prior has a value / gradient")
    if bad3:
        why.append(f"(3) an inactive prior has a geo / prior-gradient field: {bad3[:6]}")
    ckn = [(k_["check"], k_["ok"]) for k_ in ck.get("checks", [])]
    for p in PC.KEEP6 + ((PC.M6.NUM_ON_CHECK,) if "numbers" in pri else (PC.NUM_OFF,)):
        m = [o for n, o in ckn if n.startswith(p)]
        if m != [True]:
            why.append(f"(3) the separate check '{p} ...' is not present once and passed ({m})")
    gp = [e["grad_prior"] for e in on if _num(e.get("grad_prior"))]
    return {"why": why, "flag": {"active": list(pri), "n_on": len(on), "grad_prior_min": min(gp) if gp else None, "evals": low}}


def cell_dirs(root, prefix):
    rx = re.compile(rf"^{re.escape(prefix)}-(none|all6|only-[a-z0-9_]+)-lr([0-9.e-]+)-s(\d+)(?:-r(\d+))?$")
    out = {}
    for d in sorted(Path(root).glob(f"{prefix}-*-s*")):
        m = rx.match(d.name)
        if m and d.is_dir():
            out.setdefault((m.group(1), m.group(2), int(m.group(3))), []).append((int(m.group(4) or 0), d))
    return {k: sorted(v) for k, v in out.items()}


def run_json(d, spec, arm, lr, seed):
    tag = spec["tag"](arm, lr)
    tgt = "helix3" if "numbers" in spec["pri_of"](arm) else "none"
    return Path(d) / tag / f"run_whole_{tgt}_seed{seed}_{tag}.json"


def identity(r, arm, lr, seed, spec):
    a = r.get("args", {})
    pri = tuple(spec["pri_of"](arm))
    want = {"seed": seed, "lr": float(lr), "tag": f"_{spec['tag'](arm, lr)}", "target": "helix3" if "numbers" in pri else "none",
            "prior_domains": ",".join(pri), "domains": spec["domains_arg"], "prior_shuf": ""}
    bad = {k: [a.get(k), v] for k, v in want.items() if a.get(k) != v}
    bad.update({f"domains.{d}.prior": [r["domains"].get(d, {}).get("prior"), d in pri] for d in spec["mani"]
                if r["domains"].get(d, {}).get("prior") != (d in pri)})
    return bad


def device_of(d):
    """The run's GPU (gpu.csv's name column; else check.json info.device): 'b200' / 'l40s' / 'a100' / ... or the raw name."""
    name = None
    try:
        name = Path(d, "gpu.csv").read_text().splitlines()[1].split(",")[1].strip()
    except (OSError, IndexError):
        try:
            name = json.loads(Path(d, "check.json").read_text()).get("info", {}).get("device")
        except (OSError, ValueError, AttributeError):
            name = None
    low = str(name or "").lower()
    return next((k for k in ("b200", "h200", "h100", "l40s", "a100") if k in low), name)


def load_v2(root, spec, guard=True):
    """{(arm, lr, seed): rec} = readout_pt6comb.load's records and validity rules with the composition a parameter (spec)."""
    if guard:
        spec["guard"]()
    out, mani = {}, spec["mani"]
    for (arm, lr, seed), dirs in cell_dirs(root, spec["prefix"]).items():
        k_, d = dirs[-1]
        rec = {"arm": arm, "lr": lr, "seed": seed, "dir": str(d), "rerun": k_, "earlier_finished": [], "valid": False, "why": "",
               "finished": False, "problems": [], "device": device_of(d)}
        out[arm, lr, seed] = rec
        if arm not in spec["arms"] or lr not in spec["lrs"]:
            rec["why"] = f"not a {spec['prefix']} arm / lr ({arm}, {lr})"
            continue
        for k0, d0 in dirs[:-1]:
            f0 = run_json(d0, spec, arm, lr, seed)
            try:
                if f0.exists() and len(json.loads(f0.read_text())["curve"]) == spec["n_evals"]:
                    rec["earlier_finished"].append(d0.name)
            except (json.JSONDecodeError, KeyError):
                pass
        f = run_json(d, spec, arm, lr, seed)
        if not f.exists():
            rec["why"] = "no run JSON"
            continue
        r = json.loads(f.read_text())
        c = r["curve"]
        ck = json.loads((d / "check.json").read_text()) if (d / "check.json").exists() else None
        fails = [k["check"] for k in (ck or {}).get("checks", []) if not k["ok"]]
        dpf = [k["check"].split(" (")[0] for k in (ck or {}).get("checks", []) if not k["ok"] and k["check"].startswith(RO.DP_FLAG_CHECKS)]
        div = next((e["step"] for e in c if not all(v is None or math.isfinite(v) for v in (e.get("lm"), e.get("val_loss"), e.get("geo"),
                    e.get("grad_prior"), *(e.get(f"geo_{x}") for x in mani if x != "numbers")))), None)
        rec.update(code_sha=r.get("code_sha"), n_evals=len(c), finished=len(c) == spec["n_evals"], diverged_at=div, fails=fails,
                   data_sha=[e.get("data_sha") for e in c], lm0=c[0]["lm"],
                   bpb_intr=c[-1]["val_bits_per_byte"] if div is None else math.inf,
                   val_loss=c[-1]["val_loss"] if div is None else math.inf, grad=RO.window_grad(c, div))
        idb = identity(r, arm, lr, seed, spec)
        a6 = None
        geo_f = [x for x in fails if x.startswith(PC.GEO_FLAG6)]
        if idb:
            rec["why"] = f"identity: the run's record differs from its directory name {idb}"
        elif ck is None:
            rec["why"] = "no check.json"
        elif len(c) != spec["n_evals"]:
            rec["why"] = f"unfinished ({len(c)} evals)"
        elif fails and not all(x.startswith(ALLOWED_FAILS) for x in fails):
            rec["why"] = "failed check: " + "; ".join(fails)
        elif geo_f and (a6 := geo_a6_v2(r, ck, arm, spec))["why"]:
            rec["why"] = "failed check: " + "; ".join(fails) + " (A6 does not apply: " + "; ".join(a6["why"]) + ")"
        else:
            rec["valid"] = True
            rec["dp_flag"] = dpf
            rec["geo_flag"] = a6["flag"] if a6 else None
            rec["why"] = ("valid" + (" (CKA-rise check failed: a weak prior, data)" if any(x.startswith(RO.CKA_CHECK) for x in fails) else "")
                          + (f" (FLAG A5.1: {'; '.join(dpf)}; valid data)" if dpf else "")
                          + (f" (FLAG A6: geo overshoot at a perfect fit {rec['geo_flag']['evals']}; valid data)" if a6 else ""))
        if rec["finished"]:
            mt, probs = run_metrics_v2(r, mani)
            rec.update(mt)
            rec["problems"] = probs
    return out


# I2 fix 2 (diff review 2026-11-19, runs/pt6v2/tests/audit_diff_I2.md item 4): the state of an EXTRA_NONE (Amendment V2) run
X_ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUE_HOLD",
            "REQUEUE_FED", "SIGNALING", "STAGE_OUT", "STOPPED"}   # = pick_pt6v2.ACTIVE (its I4 'failed twice')


def x_state(R_, k, terminal=frozenset()):
    """'valid'; 'invalid' = a finished run that failed its checks (a full curve with check.json written, or an identity mismatch), or a
    terminal failed attempt (k in terminal: failed twice, x_terminal); 'missing' = no run directory, no run JSON, an unfinished curve
    (running, preempted / requeued, or crashed once) or a full curve whose check.json is not written yet. Strict refuses a missing one."""
    r = R_.get(k)
    if r and r.get("valid"):
        return "valid"
    if k in terminal:
        return "invalid"
    if r and r.get("finished") and r.get("why") != "no check.json":
        return "invalid"
    return "missing"


def x_markers(name, root=V2):
    """SUBMITTED_<name>[-r<k>] markers in runs/pt6v2 -> [(rr, job id or None)] (pick_pt6v2.markers on this batch's directory; read only)."""
    rx = re.compile(rf"^SUBMITTED_{re.escape(name)}(?:-r(\d+))?$")
    out = []
    for f in Path(root).glob(f"SUBMITTED_{name}*"):
        m = rx.match(f.name)
        if m:
            j = re.search(r"Submitted batch job (\d+)", f.read_text())
            out.append((int(m.group(1) or 0), j.group(1) if j else None))
    return sorted(out)


def x_sacct(ids):
    """{job id: sacct state} (pick_pt6v2.sacct_states); {} when sacct is not reachable (every attempt then counts as UNKNOWN)."""
    ids = sorted({i for i in ids if i})
    if not ids:
        return {}
    try:
        o = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(ids)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return {j.strip(): (s.split() or ["UNKNOWN"])[0] for j, s in (ln.split("|", 1) for ln in o.stdout.splitlines() if "|" in ln)}


def x_terminal(R_, seeds, lrs=EXTRA_NONE, markers_fn=x_markers, states_fn=x_sacct):
    """The EXTRA_NONE cell-seeds that x_state calls missing but that failed twice (pick_pt6v2 I4: >= 2 SUBMITTED attempts, every sacct
    state terminal, none unknown) -> (set of keys, a line per checked cell-seed). Such a run counts as invalid, not missing."""
    keys = [("none", lr, s) for lr in lrs for s in seeds if x_state(R_, ("none", lr, s)) == "missing"]
    att = {k: markers_fn(f"{PREFIX}-none-lr{k[1]}-s{k[2]}") for k in keys}
    st = states_fn([j for a in att.values() for _, j in a])
    out, lines = set(), []
    for k, a in att.items():
        x = [(rr, j, st.get(j, "UNKNOWN") if j else "UNKNOWN") for rr, j in a]
        term = len(x) >= 2 and all(y not in X_ACTIVE and y != "UNKNOWN" for *_, y in x)
        if term:
            out.add(k)
        lines.append(f"none@{k[1]} s{k[2]}: {(R_.get(k) or {}).get('why') or 'no run'}; attempts: "
                     + (", ".join(f"r{rr} {j} {y}" for rr, j, y in x) or "no SUBMITTED marker")
                     + (" -> failed twice: invalid" if term else " -> missing"))
    return out, lines


# ------------------------------------------------------------------------------------------------------------ verdicts
def desc(g):
    g = np.asarray(g, float)
    p, w = RO.sf_test(g), int((g > 0).sum())
    return {"mean": float(g.mean()), "wins": w, "n": len(g), "p": p, "helps": bool(p < ALPHA and w >= MIN_WINS)}


def floor_tags_v2(R_, m, seeds, P, N):
    """FLOOR / CEILING of manifold m at all6@P vs none@N by its source batch's definition. -> (tags, line).
    N: one lr, or {seed: lr} (I2: a seed whose N_m run is not valid uses its fallback none lr)."""
    Ns = N if isinstance(N, dict) else {s: N for s in seeds}
    NL_ = N if isinstance(N, str) else "N_m"
    rec = lambda a, lr, s: R_[a, Ns[s] if a == "none" else lr, s]   # noqa: E731
    A = lambda a, lr: [rec(a, lr, s)["area"][m] for s in seeds]   # noqa: E731
    ma, mn = mean(A("all6", P)), mean(A("none", NL_))
    rt = route(m)
    if rt == "pc":
        line = mean([rec("none", NL_, s)["floor"][m] for s in seeds])
    elif rt == "ro":
        R2, subs = {}, {}
        for a, lr in (("all6", P), ("none", NL_)):
            for s in seeds:
                fl = rec(a, lr, s)["fl"][m]
                R2["v2", a, lr, s] = {"valid": True, "batch": "v2", "arm": a, "lr": lr, "seed": s, "area": {m: rec(a, lr, s)["area"][m]},
                                      "chance": {m: fl["chance"]}, "heur": {m: fl["heur"]}, "kshort": {m: fl["kshort"]}}
                if fl.get("h3") is not None:
                    subs["v2", a, lr, s] = {m: {"h3_held": fl["h3"]}}
        t = P2.floor_tags(R2, subs, "v2", [("all6", P), ("none", NL_)], m, list(seeds))
        return (["FLOOR"] if any(x.startswith("FLOOR") for x in t["tags"]) else []) + (
            ["CEILING"] if any(x.startswith("CEILING") for x in t["tags"]) else []), t["line"]
    else:
        key = "chance_u" if m in PRIMARY_TF else "chance"
        line = mean([rec("none", NL_, s)["p3"][m][key] for s in seeds]) + 5
    return (["FLOOR"] if ma <= line and mn <= line else []) + (["CEILING"] if mn >= 95 else []), line


def verdict_v2(R_, mani, seeds=SEEDS, halves=HALVES, P=P_LR, same=SAME_LR, none_lrs=NONE_LRS, allow_partial=False, gate_txt=GATE_TXT,
               xterm=frozenset()):
    """The pt6v2 verdicts (module docstring). R_: {(arm, lr, seed): rec} with area / final3 / floor / fl / p3 / grad / device.
    I2 (AUDIT.md "toy3 readiness"): a none lr of Amendment V2 (EXTRA_NONE, 6.4e-2) is not a core cell. An invalid run of it (x_state:
    a finished run that failed its checks, or failed twice = a key in xterm) drops that lr from that seed's N_m candidates only: the
    seed stays in every verdict and uses its best valid none lr (flagged). A missing one (no run, unfinished, requeued, no check.json
    yet) is refused in strict mode and treated like an invalid one under --allow_partial (PROVISIONAL). The lr stays in the N_m grid
    while valid at more than half of the read seeds (pick_pt6v2's I1 'present'). Ranking (review fix 1): the grid lrs are ranked on the
    seeds where every one of them is valid (paired); a seed's fallback ranks its valid lrs on the seeds where all of those are valid.
    With every run valid this is the registered rule (seed means over the read seeds)."""
    xl = [lr for lr in none_lrs if lr in EXTRA_NONE]
    cells = [("all6", P)] + [("none", lr) for lr in none_lrs if lr not in xl] + [(f"only-{m}", same) for m in mani]
    core = list(dict.fromkeys(cells))
    pair = [c for c in (("none", same), ("all6", same)) if c not in core]
    vok = lambda c, s: bool((R_.get((c[0], c[1], s)) or {}).get("valid"))   # noqa: E731
    miss = [f"{a}@{lr} s{s}" for a, lr in core for s in seeds if not vok((a, lr), s)]
    xs = lambda lr, s: x_state(R_, ("none", lr, s), xterm)   # noqa: E731  (I2 fix 2)
    miss += [f"none@{lr} s{s} ({(R_.get(('none', lr, s)) or {}).get('why') or 'no run'})" for lr in xl for s in seeds if xs(lr, s) == "missing"]
    ok_s = [s for s in seeds if all(vok(c, s) for c in core)]
    miss_pair = [f"{a}@{lr} s{s}" for a, lr in pair for s in ok_s if not vok((a, lr), s)]
    pair_ok = not miss_pair
    res = {"mani": list(mani), "P": P, "same": same, "none_lrs": list(none_lrs), "seeds": ok_s, "seeds_wanted": list(seeds),
           "missing": miss, "missing_pair": miss_pair, "pair_ok": pair_ok, "m": {}}
    if miss and not allow_partial:
        return res
    if len(ok_s) < 2:
        res["empty"] = True
        return res
    xbad = {lr: {s: ((R_.get(("none", lr, s)) or {}).get("why") or "no run") + {"missing": "; missing, --allow_partial"}.get(
        xs(lr, s), "; failed twice" if ("none", lr, s) in xterm else "") for s in ok_s if not vok(("none", lr), s)} for lr in xl}
    xin = [lr for lr in xl if 2 * (len(ok_s) - len(xbad[lr])) > len(ok_s)]   # present: valid at more than half of the read seeds
    res["extra_none"] = {lr: {"not_valid": xbad[lr], "in_grid": lr in xin, "n_valid": len(ok_s) - len(xbad[lr])} for lr in xl}
    NL_all, none_lrs = tuple(none_lrs), tuple(lr for lr in none_lrs if lr not in xl or lr in xin)   # the N_m grid
    res["none_grid"] = list(none_lrs)
    A = lambda a, lr, m, f="area": [R_[a, lr, s][f][m] for s in ok_s]   # noqa: E731
    D = lambda c1, c2, m, f="area": [x - y for x, y in zip(A(*c1, m, f), A(*c2, m, f))]   # noqa: E731
    Dn = lambda Ns, m, f="area": [R_["all6", P, s][f][m] - R_["none", Ns[s], s][f][m] for s in ok_s]   # noqa: E731  (I2: per seed)
    # review fix 1 (paired ranking): the means of lrs are taken on the seeds where every one of lrs is valid
    mu_on = lambda lrs, m: {lr: mean([R_["none", lr, s]["area"][m] for s in ok_s if all(vok(("none", x), s) for x in lrs)])   # noqa: E731
                            for lr in lrs}
    mu6_on = lambda lrs: {lr: mean([mu_on(lrs, m)[lr] for m in mani]) for lr in lrs}   # noqa: E731
    pick = lambda d_: max(d_, key=lambda lr: (round(d_[lr], 9), float(lr)))   # noqa: E731
    n_seed = lambda N_, rank: {s: N_ if vok(("none", N_), s) else pick(   # noqa: E731  (I2: the seed's best valid none lr, paired)
        rank(tuple(lr for lr in none_lrs if vok(("none", lr), s)))) for s in ok_s}
    TOP_, BOT_ = (max(none_lrs, key=float), min(none_lrs, key=float)) if len(none_lrs) > 1 else (None, None)   # V2: the grid's edges
    for x_ in res["extra_none"].values():   # the seeds N_m is ranked on (= the read seeds when every grid lr is valid)
        x_["rank_seeds"] = [s for s in ok_s if all(vok(("none", lr), s) for lr in none_lrs)]
    mu = {m: mu_on(none_lrs, m) for m in mani}
    mu6 = mu6_on(none_lrs)
    N6 = pick(mu6)
    N6s = n_seed(N6, mu6_on)
    res["N6"], res["mu6"] = N6, mu6
    for m in mani:
        N = pick(mu[m])
        Ns = n_seed(N, lambda lrs, m=m: mu_on(lrs, m))
        o = f"only-{m}"
        g = Dn(Ns, m)
        h = D(("all6", same), (o, same), m) if pair_ok else None
        u = D((o, same), ("none", same), m) if pair_ok else None
        v = PC.manifold_verdict(g, h, u)
        v["N"], v["none_means"], v["N_per_seed"] = N, mu[m], Ns
        fb = {s: x for s, x in Ns.items() if x != N}
        v["flags"] = ([top_flag(TOP_)] if N == TOP_ else []) + ([BOT_FLAG] if N == BOT_ else []) + [LOW_FLAG] + (
            [f"{I2_FLAG} none@{N} at " + ", ".join(f"s{s} (uses none@{x})" for s, x in fb.items())] if fb else [])
        v["vs_lr"] = {lr: (desc(dl) if len(dl) >= 2 else None) for lr in NL_all
                      for dl in [[R_["all6", P, s]["area"][m] - R_["none", lr, s]["area"][m] for s in ok_s if vok(("none", lr), s)]]}
        v["vs_mean6"] = dict(desc(Dn(N6s, m)), N6=N6)
        v["g_same_lr"] = desc(D(("all6", same), ("none", same), m)) if pair_ok else None
        v["final3"] = {"g": mean(Dn(Ns, m, "final3")),
                       "wins_g": sum(x > 0 for x in Dn(Ns, m, "final3"))}
        v["levels"] = {"all6@P": mean(A("all6", P, m)), "none@N": mean([R_["none", Ns[s], s]["area"][m] for s in ok_s]), o: mean(A(o, same, m)),
                       "all6@same": mean(A("all6", same, m)) if pair_ok else None}
        v["tags"], v["floor_line"] = floor_tags_v2(R_, m, ok_s, P, N if not fb else Ns)
        ser = [(R_["all6", P, s].get("series", {}).get(m), R_["none", Ns[s], s].get("series", {}).get(m), R_["all6", P, s].get("steps"))
               for s in ok_s]
        if all(a_ is not None and b_ is not None and st for a_, b_, st in ser):
            ws = [PC.window_series_split(a_, b_, st) for a_, b_, st in ser]
            v["window_g"] = {"in": mean([w["in"] for w in ws]), "out": mean([w["out"] for w in ws])}
        else:
            v["window_g"] = None
        v["word"] = "HELPS" if v["helps"] else "NOT SHOWN"
        res["m"][m] = v
    for m, ph in zip(mani, RO.holm([res["m"][m]["p_help"] for m in mani])):
        res["m"][m]["p_help_holm"] = ph
        res["m"][m]["helps_holm"] = bool(ph < ALPHA and res["m"][m]["wins_g"] >= MIN_WINS)
    hv = {f"{h[0]}-{h[-1]}": [s for s in h if s in ok_s] for h in halves}
    g_h = {k: RO.cell_gate([R_["all6", P, s] for s in v]) for k, v in hv.items() if v}
    res["gate_all6"] = {k: {"ok": g["ok"], "median": g["median"], "why": g["why"]} for k, g in g_h.items()}
    res["gate_failed"] = any(not g["ok"] for g in g_h.values())
    if pair_ok:
        g_2 = {k: RO.cell_gate([R_["all6", same, s] for s in v]) for k, v in hv.items() if v}
        res["gate_all6_same_lr"] = {k: {"ok": g["ok"], "median": g["median"], "why": g["why"]} for k, g in g_2.items()}
        res["gate_same_lr_failed"] = any(not g["ok"] for g in g_2.values())
    else:
        res["gate_all6_same_lr"], res["gate_same_lr_failed"] = None, None
    for m in mani:
        v = res["m"][m]
        v["int_word"] = ("interference WITHHELD (the same-lr pair is incomplete)" if not pair_ok else
                         f"AUX GATE FAILED at all6@{same}" if res["gate_same_lr_failed"] else
                         f"interference not detected (by {v['noint_by']})" if v["noint"] else "INTERFERENCE DETECTED")
    res["arms"] = {}
    for a, lr in list(dict.fromkeys(core + [("none", lr) for lr in xl] + pair)):
        if not all(vok((a, lr), s) for s in ok_s):
            continue
        rs = [R_[a, lr, s] for s in ok_s]
        res["arms"][f"{a}@{lr}"] = {"lm_share": RO.cell_gate(rs)["median"] if a != "none" else None,
                                    "val_loss": mean([r["val_loss"] for r in rs]), "bpb": mean([r["bpb_intr"] for r in rs])}
    per_seed = [mean([res["m"][m]["g"][i] for m in mani]) for i in range(len(ok_s))]
    mu_, _, ci = RO.t_test(per_seed)
    res["mean6_gain"] = {"per_seed": per_seed, "mean": mu_, "ci95": list(ci)}
    devs = {s: sorted({str(R_[a, lr, s].get("device")) for a, lr in core + [("none", lr) for lr in xl] + pair if (a, lr, s) in R_})
            for s in ok_s}
    alld = sorted({x for v in devs.values() for x in v})
    res["devices"] = devs
    res["device_note"] = ((f"mixed devices across the read seeds ({'; '.join(f'{x}: ' + ','.join(str(s) for s in ok_s if x in devs[s]) for x in alld)})"
                           if len(alld) > 1 else f"one device ({alld[0] if alld else '?'})")
                          + (f"; seeds with cells on more than one device: {[s for s in ok_s if len(devs[s]) > 1]}"
                             if any(len(devs[s]) > 1 for s in ok_s) else "") + " (a note, never a condition)")
    helps_ = [m for m in mani if res["m"][m]["helps"]]
    if res["gate_failed"]:
        hl = {"k": None, "list": [], "helps": helps_, "text": gate_txt}
    elif not pair_ok or res["gate_same_lr_failed"]:
        why = (f"WITHHELD: the same-lr pair is incomplete ({', '.join(miss_pair[:4])})" if not pair_ok else
               f"AUX GATE FAILED at all6@{same} (the arm of the interference contrast): interference words withheld")
        hl = {"k": None, "list": [], "helps": helps_, "text": f"{why}; HELPS (all6@{P} vs none@N_m): {len(helps_)} of {len(mani)}: "
              f"{', '.join(helps_) or 'none'}"}
    else:
        ok = [m for m in mani if res["m"][m]["helps"] and res["m"][m]["noint"]]
        hl = {"k": len(ok), "list": ok, "helps": helps_, "text": f"{len(ok)} of {len(mani)} HELPS with interference not detected "
              f"(at the same lr {same}): {', '.join(ok) or 'none'}"}
    res["headline"] = hl
    return res


# ------------------------------------------------------------------------------------------------------------ report
f1 = lambda v, p=1: "-" if v is None else f"{v:+.{p}f}"   # noqa: E731
f0 = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"    # noqa: E731


def report(res, R_, comp_src="", prov=(), probs=()):
    mani, P, same, NL, ss = res["mani"], res["P"], res["same"], res["none_lrs"], res["seeds"]
    L = [f"pt6v2 read-out (runs/pt6v2/AUDIT.md 'Verdict rule'; readout_pt6v2.py sha256[:12] {sha12(__file__)}; "
         f"make_jobs_pt6v2.py {sha12(MJ2.__file__)})",
         f"composition: {', '.join(mani)} [{comp_src}]; read seeds {ss} of {res['seeds_wanted']}; P = all6@{P}; none lrs {NL}; same lr {same}"]
    if prov:
        L.append("PROVISIONAL (--allow_partial): " + "; ".join(prov))
    cols = [("all6", P), ("all6", same)] + [("none", lr) for lr in NL] + [("only", same)]
    hdr = "  | manifold    set                       " + "".join(f"{a + '@' + lr:>12s}" for a, lr in cols)
    L += ["", "== 1. every metric computed (seed means; area = 100 x mean over the evals of the primary read-out; f3 = the last 3) ==", hdr]
    def val(a, lr, m, f, k=None):   # I2: an extra none lr (Amendment V2) is the mean over its valid seeds; a crashed record -> '-'
        ss_ = [s for s in ss if (R_.get((a, lr, s)) or {}).get("valid")] if (a == "none" and lr in EXTRA_NONE) else ss
        if not ss_ or not all((a, lr, s) in R_ and f in R_[a, lr, s] for s in ss_):
            return None
        xs = [(R_[a, lr, s][f][m] if k is None else (R_[a, lr, s][f][m] or {}).get(k)) for s in ss_]
        return mean(xs) if all(x is not None for x in xs) else None
    for m in mani:
        cc = [(f"only-{m}" if a == "only" else a, lr) for a, lr in cols]
        for f, k, lab in (("area", None, "area"), ("final3", None, "f3")) + tuple(
                ("sec", k, k) for k in (("full_area", "tf_area", "n_tf") + (("aae_area", "noaae_area", "n_aae") if m == "d6" else ()))
                if route(m) == "p3") + ((("sec", "hard_area", "hard_area"),) if m in P2.COMP else ()) + (
                (("sec", f"{m}_readout", f"{m}_readout"),) if m in ("s4", "dim") else ()):
            L.append(f"  | {m:<11s} {(primary_set(m) if lab == 'area' else '') + ' ' + lab:<26s}"
                     + "".join(f"{f0(val(a, lr, m, f, k)):>12s}" for a, lr in cc))
    L.append("  | LM (val loss / bpb at the end; LM share a/(a+b) median):   " + "; ".join(
        f"{k} {f0(v['val_loss'], 4)} / {f0(v['bpb'], 4)}" + (f" share {f0(v['lm_share'], 3)}" if v["lm_share"] is not None else "")
        for k, v in res["arms"].items()))
    L += ["", f"== 2. verdicts (AUDIT.md: HELPS iff one-sided paired t p < {ALPHA} and g > 0 in >= {MIN_WINS} of {len(res['seeds_wanted'])} "
          f"seeds; N_m = the none lr with the highest mean area of m, a tie -> the higher lr) =="]
    main_ = [m for m in mani if m not in SEPARATE]
    sep_ = [m for m in mani if m in SEPARATE]

    def vline(m, pre):
        v = res["m"][m]
        ns = v["word"] == "NOT SHOWN"   # V1 item 2: the same-lr gain all6@same - none@same and u first
        gs = v["g_same_lr"]
        first = ((f"[NOT SHOWN at P: same-lr gain all6@{same} - none@{same} {f1(gs['mean'])} ({gs['wins']}/{gs['n']}, p {gs['p']:.3f}), "
                  f"u {f1(v['mean_u'])}] ") if gs else "[NOT SHOWN at P: same-lr gain - (the same-lr pair is incomplete), u -] ") if ns else ""
        x = (f"{pre}{m:<11s} {first}N_m = none@{v['N']}: g {f1(v['mean_g'])} (90% boot [{f1(v['boot90_g'][0])}, {f1(v['boot90_g'][1])}]) "
             f"wins {v['wins_g']}/{len(ss)} p {v['p_help']:.4f} (Holm {v['p_help_holm']:.4f}) -> {v['word']}"
             + "".join(f" [{t}]" for t in v["tags"]) + "".join(f" [{t}]" for t in v["flags"])
             + f" | A2: h {f1(v['mean_h'])} u {f1(v['mean_u'])} margin {f1(v['margin'])} p(h<0) "
             + (f"{v['p_int']:.3f}" if v["p_int"] is not None else "-") + f" -> {v['int_word']}")
        y = (f"{pre}{'':<11s} descriptive: g vs none@" + "; ".join(f"{lr} {f1(d['mean'])} ({d['wins']}/{d['n']}, p {d['p']:.3f})" if d else f"{lr} -"
                                                                  for lr, d in v["vs_lr"].items())
             + f" | vs the mean6-picked none@{v['vs_mean6']['N6']}: {f1(v['vs_mean6']['mean'])} ({v['vs_mean6']['wins']}/{len(ss)}, p "
             f"{v['vs_mean6']['p']:.3f}) -> {'HELPS' if v['vs_mean6']['helps'] else 'NOT SHOWN'} (descriptive)"
             + (f" | all6@{same} - none@{same} {f1(v['g_same_lr']['mean'])}" if v["g_same_lr"] else "")
             + f" | final3 g {f1(v['final3']['g'])} ({v['final3']['wins_g']}/{len(ss)})"
             + (f" | window in {f1(v['window_g']['in'])} out {f1(v['window_g']['out'])}" if v["window_g"] else "")
             + f" | FLOOR line {f0(v['floor_line'])}; none means " + ", ".join(f"{lr} {f0(x_)}" for lr, x_ in v["none_means"].items()))
        return [x, y]
    for m in main_:
        L += vline(m, "  ")
    for m in sep_:
        L += [f"  [separate] {m} (read out separately): {P3.CAESAR_NOTE}"] + vline(m, "  [separate] ")
    if "d6" in mani:
        L.append("  d6: AAE flag: the read-out holds (a, a) -> e cells (a != e); its AAE / noAAE areas are in table 1")
    L.append(f"  GATE (A2.1) all6@{P} per half: " + "; ".join(f"{k}: {'ok' if g['ok'] else 'FAILED'} (median {f0(g['median'], 3)})"
                                                         for k, g in res["gate_all6"].items()))
    if res["gate_all6_same_lr"] is not None:
        L.append(f"  A3 gate all6@{same} per half: " + "; ".join(f"{k}: {'ok' if g['ok'] else 'FAILED'} (median {f0(g['median'], 3)})"
                                                            for k, g in res["gate_all6_same_lr"].items()))
    else:
        L.append(f"  A3: the same-lr pair is incomplete ({', '.join(res['missing_pair'][:6])}): interference WITHHELD")
    mg = res["mean6_gain"]
    L.append(f"  mean6 gain (descriptive) {f1(mg['mean'], 2)} 95% t CI [{f1(mg['ci95'][0], 2)}, {f1(mg['ci95'][1], 2)}]; "
             f"the mean6-picked none@{res['N6']}")
    L.append("  devices per seed: " + "; ".join(f"s{s} {','.join(v)}" for s, v in res["devices"].items()) + f"; {res['device_note']}")
    for lr, x in (res.get("extra_none") or {}).items():   # I2
        if x["not_valid"]:
            L.append(f"  Amendment V2 (I2): none@{lr} not valid at " + ", ".join(f"s{s} ({w})" for s, w in x["not_valid"].items())
                     + (f": those seeds' N_m candidates are the other none lrs (the seeds stay in every verdict); the N_m grid is ranked on "
                        f"the {len(x['rank_seeds'])} seeds where every grid lr is valid (paired: s{', s'.join(map(str, x['rank_seeds']))})"
                        if x["in_grid"] else
                        f": valid at {x['n_valid']} of {len(ss)} read seeds (<= half): none@{lr} left out of the N_m grid {res['none_grid']}"))
    if probs:
        L.append(f"  problems ({len(probs)}): " + "; ".join(list(probs)[:8]))
    L += [f"  {Q_SAME_SEEDS} (none grid {res.get('none_grid', NL)})", f"  {reading_line(mani)}"]
    L += ["", "HEADLINE: " + res["headline"]["text"] + (" (caesar counted; see its separate block)" if sep_ else "")
          + (" [PROVISIONAL]" if prov else "")]
    return "\n".join(L)


def readout_from(R_, mani, comp_src="", probs=(), allow_partial=False, **kw):
    res = verdict_v2(R_, mani, allow_partial=allow_partial, **kw)
    prov = []
    if res["missing"]:
        prov.append(f"{len(res['missing'])} core runs missing / invalid: {res['missing'][:6]}")
    if probs:
        prov.append(f"{len(probs)} problems")
    if prov and not allow_partial:
        raise SystemExit("REFUSED (strict read-out): " + "; ".join(prov + list(probs)[:10]) + " (--allow_partial prints PROVISIONAL)")
    if res.get("empty"):
        return res, "PROVISIONAL: fewer than 2 complete seeds: no verdicts; " + "; ".join(prov)
    res["provisional"] = prov
    return res, report(res, R_, comp_src, prov, probs)


def extended_from(R_all, res_r, mani, lr, probs_new=(), xterm=frozenset(), xlines=()):
    """Amendment V3 (post-hoc, user-approved 2026-11-19): the verdict on the registered none grid + lr (the post-hoc runs) with the I2
    rules for lr (an invalid run -> that seed's other none lrs; a missing one -> this block PROVISIONAL; lr in the N_m grid iff valid
    at more than half of the seeds). res_r (the registered verdict) is only read. -> (block text, the "posthoc_V3" JSON value)."""
    grid = tuple(res_r["none_lrs"]) + (lr,)
    res_e = verdict_v2(R_all, mani, allow_partial=True, none_lrs=grid, xterm=frozenset(xterm))
    pv = ([f"{len(res_e['missing'])} runs missing / invalid: {res_e['missing'][:6]}"] if res_e["missing"] else []) + (
        [f"{len(probs_new)} problems the post-hoc runs bring: {list(probs_new)[:4]}"] if probs_new else []) + (
        ["the registered block is PROVISIONAL"] if res_r.get("provisional") else [])
    out = {"tag": V3_TAG, "lr": lr, "grid": list(grid), "provisional": pv, "rule": V3_RULE, "change": {}, "res": res_e}
    L = ["", f"== 3. {V3_TAG}: none grid {list(grid)} (runs/pt6v2/POSTHOC_V3.json; blocks 1-2 above = the registered read-out) =="]
    L += [f"  I2 attempts (post-hoc none@{lr}): {x}" for x in xlines]
    if res_e.get("empty"):
        out["provisional"] = pv + ["fewer than 2 complete seeds"]
        L.append("  PROVISIONAL: fewer than 2 complete seeds: no extended verdicts")
        return "\n".join(L), out
    ss = res_e["seeds"]
    sgn = lambda x: ((x > 0) - (x < 0)) if _num(x) else None   # noqa: E731
    for m in mani:
        v, r = res_e["m"][m], res_r["m"][m]
        c = {"N_reg": r["N"], "N_ext": v["N"], "g_reg": r["mean_g"], "g_ext": v["mean_g"],
             "dg": (v["mean_g"] - r["mean_g"]) if _num(v["mean_g"]) and _num(r["mean_g"]) else None,
             "word_reg": r["word"], "word_ext": v["word"], "helps_holm_reg": r["helps_holm"], "helps_holm_ext": v["helps_holm"]}
        c["sign_flip"] = None not in (sgn(c["g_reg"]), sgn(c["g_ext"])) and sgn(c["g_reg"]) != sgn(c["g_ext"])
        c["lost_helps"] = r["word"] == "HELPS" and v["word"] != "HELPS"
        c["gained_helps"] = r["word"] != "HELPS" and v["word"] == "HELPS"
        out["change"][m] = c
        L.append(f"  {m:<11s} N_m = none@{v['N']}: g {f1(v['mean_g'])} (90% boot [{f1(v['boot90_g'][0])}, {f1(v['boot90_g'][1])}]) "
                 f"wins {v['wins_g']}/{len(ss)} p {v['p_help']:.4f} (Holm {v['p_help_holm']:.4f}) -> {v['word']}"
                 + "".join(f" [{t}]" for t in v["flags"]) + f" | final3 g {f1(v['final3']['g'])} ({v['final3']['wins_g']}/{len(ss)})"
                 + f" | vs registered: N_m {r['N']} -> {v['N']}, g {f1(r['mean_g'])} -> {f1(v['mean_g'])} (change {f1(c['dg'])}), "
                 f"{r['word']} -> {v['word']}" + (" [g CHANGES SIGN]" if c["sign_flip"] else "") + (" [LOSES HELPS]" if c["lost_helps"] else "")
                 + (" [GAINS HELPS]" if c["gained_helps"] else ""))
    x = (res_e.get("extra_none") or {}).get(lr)
    if x:
        L.append(f"  Amendment V2 (I2) rules on post-hoc none@{lr}: valid at {x['n_valid']} of {len(ss)} read seeds -> "
                 + (f"in the N_m grid (ranked paired on s{', s'.join(map(str, x['rank_seeds']))})" if x["in_grid"] else
                    "left out of the N_m grid (valid at <= half of the seeds): the extended verdict = the registered grid's")
                 + ("; not valid at " + ", ".join(f"s{s} ({w})" for s, w in x["not_valid"].items()) if x["not_valid"] else ""))
    mg, mr = res_e["mean6_gain"], res_r["mean6_gain"]
    L.append(f"  mean6 gain (descriptive) {f1(mg['mean'], 2)} 95% t CI [{f1(mg['ci95'][0], 2)}, {f1(mg['ci95'][1], 2)}]; the mean6-picked "
             f"none@{res_e['N6']} (registered: {f1(mr['mean'], 2)}, none@{res_r['N6']})")
    chg = [m for m, c in out["change"].items() if c["sign_flip"] or c["lost_helps"]]
    L.append("  changed vs registered: " + ("; ".join(f"{m} (" + ", ".join(w for w, b in (("g changes sign", out["change"][m]["sign_flip"]),
                                                                                           ("loses HELPS", out["change"][m]["lost_helps"])) if b) + ")"
                                                      for m in chg) + ": report both; the paper wording follows the extended verdict"
                                           if chg else "no manifold's g changes sign or loses HELPS"))
    L += [f"  {V3_RULE}", "EXTENDED HEADLINE: " + res_e["headline"]["text"] + (f" [PROVISIONAL: {'; '.join(pv)}]" if pv else "")]
    return "\n".join(L), out


def readout(root=None, allow_partial=False, guard=True, comp=None, posthoc=None):
    doms, src = composition(comp)
    NL = none_grid()   # V2: CELLS.json's none grid (default 5 lrs; + 6.4e-2 under Amendment V2)
    PH = tuple(x for x in posthoc_lrs() if x not in NL)   # V3: post-hoc none lrs (POSTHOC_V3.json); without --posthoc_none: left out
    assert posthoc is None or posthoc in PH, f"--posthoc_none {posthoc}: not a post-hoc lr of {POSTHOC_FILE} outside the grid {list(NL)}"
    spec = spec_v2(doms, NL + ((posthoc,) if posthoc else ()))
    root_ = Path(root or (V2 / "out"))
    R_all = load_v2(root_, spec, guard)
    R_, R_ph = split_posthoc(R_all, PH, NL)   # V3: every registered number below is read from R_ only
    probs = list(PC.integrity(R_, set(SEEDS)))
    rx_ = re.compile(rf"^{PREFIX}-none-lr([0-9.e-]+)-s\d+")
    probs += [f"{p_.name}: a none lr outside the CELLS.json grid {list(NL)}" for p_ in sorted(root_.glob(f"{PREFIX}-none-lr*"))
              if rx_.match(p_.name) and rx_.match(p_.name)[1] not in NL + PH]
    xterm, xlines = x_terminal(R_, SEEDS, [lr for lr in NL if lr in EXTRA_NONE])   # I2 fix 2: failed twice -> invalid, else missing
    for ln in xlines:
        print("I2 attempts: " + ln)
    res, txt = readout_from(R_, spec["mani"], src, probs, allow_partial, none_lrs=NL, xterm=xterm)
    inv = [f"{k[0]}@{k[1]} s{k[2]}: {r['why']}" for k, r in sorted(R_.items()) if not r["valid"]]
    if inv:
        txt += "\nNOT VALID: " + "; ".join(inv[:20])
    if posthoc and not res.get("empty"):   # V3: block 3 after the unchanged registered text
        xt3, xl3 = x_terminal(R_all, SEEDS, [posthoc])
        t3, res["posthoc_V3"] = extended_from(R_all, res, spec["mani"], posthoc, posthoc_problems(R_all, R_), set(xterm) | xt3, xl3)
        inv3 = [f"none@{k[1]} s{k[2]}: {r['why']}" for k, r in R_ph.items() if not r["valid"]]
        txt += t3 + ("\n  post-hoc NOT VALID: " + "; ".join(inv3) if inv3 else "")
    return res, txt


def canary(out):
    """AUDIT.md 'Pipeline canary': a number for each manifold of the canary run under out. -> (rc, text, {m: area})."""
    out = Path(out)
    fs = sorted(out.glob("*/run_whole_*_seed*_*.json"))
    if len(fs) != 1:
        return 1, f"CANARY READ-OUT: FAIL: {len(fs)} run JSONs under {out} (want 1)", {}
    r = json.loads(fs[0].read_text())
    doms = tuple(x for x in str(r["args"]["domains"]).split(",") if x and x != "numbers")
    mani = ("numbers",) + doms
    L = [f"canary read-out of {fs[0]} (readout_pt6v2.py {sha12(__file__)}): {len(r['curve'])} evals, last step {r['curve'][-1]['step']}"]
    try:
        want, src = composition()
        if tuple(sorted(want)) != tuple(sorted(doms)):
            L.append(f"NOTE: the canary's domains {doms} != the composition the read-out uses ({want}, {src})")
    except AssertionError as ex:
        L.append(f"NOTE: composition: {ex}")
    ck = out / "check.json"
    if ck.exists():
        C = json.loads(ck.read_text())
        fails = [k["check"][:90] for k in C.get("checks", []) if not k["ok"]]
        L.append(f"check.json: pass {C.get('pass')}; failed: {fails or 'none'}"
                 + (f" (not flag-only: {[x for x in fails if not x.startswith(ALLOWED_FAILS)]})" if fails else ""))
    else:
        L.append("check.json: missing")
    nums = {}
    for m in mani:
        try:
            o, p = dom_metrics(r, m)
        except Exception as ex:   # noqa: BLE001
            o, p = None, [repr(ex)[:200]]
        if o and _num(o["area"]):
            nums[m] = o["area"]
            ex_ = (f" floor line {f0(o['floor'])}" if o["floor"] is not None else "") + (
                f" chance {f0(o['fl']['chance'])}" if o["fl"] else "") + (
                f" full {f0(o['p3'].get('area'))} tf {f0(o['p3'].get('tf_area'))} n_tf {o['p3'].get('n_tf')}" if o["p3"] else "")
            L.append(f"  {m:<11s} [{route(m)}: {primary_set(m)}] area {o['area']:.2f} final3 {f0(o['final3'], 2)}{ex_}"
                     + (f"  marks: {p[:2]}" if p else ""))
        else:
            L.append(f"  {m:<11s} NO NUMBER: {p[:3]}")
    ok = len(nums) == len(mani)
    L.append(f"CANARY READ-OUT: {len(nums)} of {len(mani)} manifolds read: {'PASS' if ok else 'FAIL'}")
    return (0 if ok else 1), "\n".join(L), nums


# ======================================================================================================= MUTANTS / self-test
# Everything below this marker is test code; mutations apply only to the code above it.
MUTANTS = [
    ("a tie goes to the LOWER lr", "(round(d_[lr], 9), float(lr))", "(round(d_[lr], 9), -float(lr))"),
    ("N_m = the mean6-picked none", "N = pick(mu[m])", "N = N6"),
    ("the top-lr flag on the wrong lr", "[top_flag(TOP_)] if N == TOP_ else []", "[top_flag(TOP_)] if N == none_lrs[0] else []"),
    ("V2: the top-lr flag fixed at 3.2e-2", "(max(none_lrs, key=float), min(none_lrs, key=float)) if len", "(TOP_LR, min(none_lrs, key=float)) if len"),
    ("V2: N_m ignores none@6.4e-2", "pick = lambda d_: max(d_, key=", "pick = lambda d_: max([x for x in d_ if x not in EXTRA_NONE], key="),
    ("V2: months read as Tier L", '"months": False', '"months": True'),
    ("twin-free primary only for hcube", "PRIMARY_TF = tuple(P3.PRIMARY_TF)", "PRIMARY_TF = ('hcube',)"),
    ("GATE over all seeds, not per half", 'hv = {f"{h[0]}-{h[-1]}": [s for s in h if s in ok_s] for h in halves}', 'hv = {"all": list(ok_s)}'),
    ("no refusal", "if prov and not allow_partial:", "if False:"),
    ("caesar not separate", 'SEPARATE = ("caesar",)', "SEPARATE = ()"),
    ("the device note in the headline", 'res["headline"] = hl', 'res["headline"] = dict(hl, text=hl["text"] + " " + res["device_note"])'),
    ("V1: the low-edge flag on the top lr", "([BOT_FLAG] if N == BOT_ else [])", "([BOT_FLAG] if N == TOP_ else [])"),
    ("V1: no same-lr gain first on a NOT SHOWN line", 'ns = v["word"] == "NOT SHOWN"', 'ns = v["word"] == "HELPS"'),
    ("I2: none@6.4e-2 a core cell (an invalid run drops its seed)", "xl = [lr for lr in none_lrs if lr in EXTRA_NONE]", "xl = []"),
    ("I2: none@6.4e-2 kept in the grid at any number of valid seeds", "2 * (len(ok_s) - len(xbad[lr])) > len(ok_s)", "len(ok_s) - len(xbad[lr]) > 0"),
    ("I2: a seed without its N_m run falls back to the grid's lowest lr", "rank(tuple(lr for lr in none_lrs if vok((\"none\", lr), s)))",
     "{min(none_lrs, key=float): 0.0}"),
    ("I2 fix 1: unpaired ranking (each lr over its own valid seeds)", "if all(vok((\"none\", x), s) for x in lrs)", "if vok((\"none\", lr), s)"),
    ("I2 fix 2: a started but unfinished run counts as invalid", "if r and r.get(\"finished\") and r.get(\"why\") != \"no check.json\":", "if r:"),
    ("I2 fix 2: a full curve without check.json counts as invalid", "and r.get(\"why\") != \"no check.json\":", ":"),
    ("I2 fix 2: failed twice still counts as missing", "    if k in terminal:\n        return \"invalid\"", "    if False:\n        return \"invalid\""),
    ("I2 fix 2: one terminal attempt counts as failed twice", "term = len(x) >= 2 and", "term = len(x) >= 1 and"),
    ("I2 fix 2: an active attempt counts as terminal", "all(y not in X_ACTIVE and y != \"UNKNOWN\" for *_, y in x)", "all(y != \"UNKNOWN\" for *_, y in x)"),
    ("V3: flag off keeps the post-hoc runs in the registered records", "ph = {k for k in R_ if k[0] == \"none\" and k[1] in ph_lrs and",
     "ph = {k for k in R_ if False and"),
    ("V3: a sign change of g not reported", "c[\"sign_flip\"] = None not in", "c[\"sign_flip\"] = False and None not in"),
    ("V3: a missing post-hoc run not PROVISIONAL", "pv = ([f\"{len(res_e['missing'])} runs missing", "pv = ([] and [f\"{len(res_e['missing'])} runs missing"),
]
E_ = [0.0, 0.3, -0.2, 0.1, 0.4, -0.1, 0.2, -0.3]
GN = [3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 2.0, 3.0]
HN = [0.1, -0.2, 0.3, 0.0, 0.1, -0.1, 0.2, 0.0]
BASE = {"2e-3": 40.0, "4e-3": 43.0, "8e-3": 45.0, "1.6e-2": 50.0, "3.2e-2": 48.0}   # 4e-3: Amendment V1


def fake_rec(arm, lr, s, area, mani, share=0.9, dev="b200"):
    steps = list(range(0, 18310, 250)) + [18309]
    on = [{"prior_on": 1, "step": t, "grad_lm": 1.0, "grad_prior": (1 - share) / share, "grad_cos": 0.0, "clip_coef": 1.0,
           "grad_lm_share": 1.0 / math.sqrt(1 + ((1 - share) / share) ** 2)} for t in steps if PC.WIN[0] <= t <= PC.WIN[1]] if arm != "none" else []
    return {"arm": arm, "lr": lr, "seed": s, "valid": True, "finished": True, "why": "planted", "area": dict(area), "final3": dict(area),
            "series": {m: [area[m] / 100] * len(steps) for m in mani}, "steps": steps,
            "floor": {m: 5.0 if route(m) == "pc" else None for m in mani},
            "fl": {m: {"chance": 20.0, "heur": [10.0, 10.0], "kshort": None, "h3": 20.0} if route(m) == "ro" else None for m in mani},
            "p3": {m: {"chance": 8.3, "chance_u": 5.0} if route(m) == "p3" else None for m in mani},
            "sec": {m: {} for m in mani}, "cka_win": {m: 0.5 for m in mani}, "device": dev, "val_loss": 4.0, "bpb_intr": 1.2,
            "data_sha": [f"d{s}"] * len(steps), "lm0": 11.6, "code_sha": "x", "problems": [], "earlier_finished": [],
            "dir": f"planted/{arm}-lr{lr}-s{s}",
            "grad": RO.window_grad(on) if on else {"n_on": 0, "n_ok": 0, "n_missing": 0, "missing": [], "share": None, "share_min": None,
                                                    "n_b_gt_a": 0, "n_clip_lt1": 0, "dp_maxdiff": None, "clip_coef": None}}


def world(mani, mod=None, share=None, dev=None, drop=()):
    """The default planted world, every manifold: none@{2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2} = 40 / 43 / 45 / 50 / 48 + e_s; all6@1e-3 = 50 + e_s
    + GN_s (HELPS vs N_m = 1.6e-2); all6@2e-3 = the same; only-m@2e-3 = none@2e-3 except m = all6@2e-3 - HN_s (no interference).
    mod(L) edits the levels L[(arm, lr)][m] (8-lists); share {(arm, lr, s): LM share}; dev {s: device}; drop: (arm, lr, s) left out."""
    L = {("none", lr): {m: [b + e for e in E_] for m in mani} for lr, b in BASE.items()}
    L["all6", P_LR] = {m: [50 + e + g for e, g in zip(E_, GN)] for m in mani}
    L["all6", SAME_LR] = {m: list(v) for m, v in L["all6", P_LR].items()}
    for m in mani:
        L[f"only-{m}", SAME_LR] = {k: list(L["none", SAME_LR][k]) for k in mani}
        L[f"only-{m}", SAME_LR][m] = [a - h for a, h in zip(L["all6", SAME_LR][m], HN)]
    if mod:
        mod(L)
    R_ = {}
    for (a, lr), lv in L.items():
        for i, s in enumerate(SEEDS):
            if (a, lr, s) not in drop:
                R_[a, lr, s] = fake_rec(a, lr, s, {m: lv[m][i] for m in mani}, mani, share=(share or {}).get((a, lr, s), 0.9),
                                        dev=(dev or {}).get(s, "b200"))
    return R_


def selftest(quiet=False):
    """Test 4 (planted worlds). -> (n checks, failures)."""
    n, fails = [0], []

    def chk(name, cond):
        n[0] += 1
        if not cond:
            fails.append(name)
        if not quiet:
            print(f"  {'ok ' if cond else 'FAIL'} {name}")
    mani = MJ2.manifolds(MJ2.comp_doms(MJ2.PLACEHOLDER))
    m0, m1, m2, m3, m4, m5 = mani
    chk("placeholder = numbers, dim, cities, countries, compass, hcube (routes pc pc pc pc ro p3)",
        set(mani) == {"numbers", "dim", "cities", "countries", "compass", "hcube"} and [route(m) for m in
                                                                                        ("numbers", "dim", "cities", "compass", "hcube")] == ["pc", "pc", "pc", "ro", "p3"])
    chk("names: cell_dirs parses make_jobs_pt6v2.name_v2; tag = tag_v2", all(
        re.match(rf"^{PREFIX}-(none|all6|only-[a-z0-9_]+)-lr([0-9.e-]+)-s(\d+)(?:-r(\d+))?$", MJ2.name_v2(a, lr, 17, rr)).groups()[:3]
        == (a, lr, "17") for a, lr in MJ2.cells(mani[1:]) for rr in (0, 2)) and not re.match(
        rf"^{PREFIX}-(none|all6|only-[a-z0-9_]+)-lr", MJ2.name_v2("all6", "2e-3", 16, canary=True)))
    # default world
    res, txt = readout_from(world(mani), mani, "planted")
    chk("default: N_m = 1.6e-2 for every m; HELPS everywhere; 6 of 6",
        all(res["m"][m]["N"] == "1.6e-2" and res["m"][m]["word"] == "HELPS" for m in mani) and res["headline"]["k"] == 6)
    chk("default: 'prior at its lowest lr' always, no top-lr flag", all(res["m"][m]["flags"] == [LOW_FLAG] for m in mani))
    chk("default: the report puts table 1 before the verdicts", txt.index("== 1.") < txt.index("== 2.") < txt.index("HEADLINE"))

    # N_m = the per-manifold max; HELPS judged against N_m; the 3.2e-2 flag
    def mod1(L):
        L["none", "8e-3"][m1] = [60 + e for e in E_]
        L["none", "3.2e-2"][m2] = [70 + e for e in E_]
        L["all6", P_LR][m3] = [47 + e + 0.5 * g for e, g in zip(E_, GN)]
    res, _ = readout_from(world(mani, mod1), mani)
    chk("N_m = the per-manifold max (1.6e-2 / 8e-3 / 3.2e-2)", (res["m"][m0]["N"], res["m"][m1]["N"], res["m"][m2]["N"]) == ("1.6e-2", "8e-3", "3.2e-2"))
    chk("HELPS judged against N_m: m1, m2 NOT SHOWN", res["m"][m1]["word"] == res["m"][m2]["word"] == "NOT SHOWN")
    chk("HELPS vs N_m, not vs none@2e-3: m3 NOT SHOWN though vs none@2e-3 it HELPS",
        res["m"][m3]["word"] == "NOT SHOWN" and res["m"][m3]["vs_lr"]["2e-3"]["helps"])
    chk("top-lr flag iff N_m = 3.2e-2", TOP_FLAG in res["m"][m2]["flags"] and all(TOP_FLAG not in res["m"][m]["flags"] for m in (m0, m1, m3)))
    chk("headline counts HELPS + no interference: 3 of 6", res["headline"]["k"] == 3 and res["headline"]["list"] == [m0, m4, m5])

    # V2 (Amendment V2, opt-in none@6.4e-2): N_m over every none lr present; the top-lr flag on the grid's actual top
    def mod_v2(L):
        L["none", "6.4e-2"] = {m: [(72 if m == m2 else 40) + e for e in E_] for m in mani}
        L["none", "3.2e-2"][m3] = [71 + e for e in E_]
    rv2, tv2 = readout_from(world(mani, mod_v2), mani, none_lrs=NONE_LRS + EXTRA_NONE)
    chk("V2: N_m over the 6-lr grid (m2 -> 6.4e-2, m3 -> 3.2e-2, m0 -> 1.6e-2)",
        (rv2["m"][m2]["N"], rv2["m"][m3]["N"], rv2["m"][m0]["N"]) == ("6.4e-2", "3.2e-2", "1.6e-2"))
    chk("V2: 'none at its top lr 6.4e-2' iff N_m = 6.4e-2; 3.2e-2 no longer flagged as the top",
        top_flag("6.4e-2") in rv2["m"][m2]["flags"] and all(not any(f.startswith("none at its top lr") for f in rv2["m"][m]["flags"])
                                                             for m in mani if m != m2))
    chk("V2: the report shows the 6-lr none grid", "6.4e-2" in tv2.split("HEADLINE")[0])
    rl = reading_line(mani)
    chk("V2: placeholder reading line: comparisons + the 5 domains Tier L; dim / hcube nonce-tied; pt6comb's tail",
        rl.startswith("reading: comparisons, ") and all(d in rl.split(" are Tier L")[0] for d in mani if d != "numbers")
        and "dim / hcube items are single nonce tokens" in rl and rl.endswith("; " + Q_TAIL) and "not Tier L" not in rl)
    chk("V2: every ALLOWED domain: Tier L iff not months; the line names it",
        all((f", {d} are Tier L" in reading_line(("numbers", d))) == (d != "months") for d in MJ2.ALLOWED)
        and "months not Tier L" in reading_line(("numbers", "months")))

    # I2 (Amendment V2): a crashed / invalid none@6.4e-2 run drops only from that seed's N_m candidates, not the whole seed
    def bad_rec(lr, s, why="failed check: diverged at step 300 (planted)", finished=True):   # fix 2: invalid = finished, checks failed
        return {"arm": "none", "lr": lr, "seed": s, "valid": False, "finished": finished, "why": why, "problems": [], "earlier_finished": [],
                "device": "b200", "dir": f"planted/none-lr{lr}-s{s}"}
    Wi = world(mani, mod_v2)
    Wi["none", "6.4e-2", 17] = bad_rec("6.4e-2", 17)
    ri, ti = readout_from(Wi, mani, none_lrs=NONE_LRS + EXTRA_NONE)
    i17 = ri["seeds"].index(17) if 17 in ri["seeds"] else None
    chk("I2: an invalid none@6.4e-2 at s17 -> strict read-out on all 8 seeds; m2 N_m = 6.4e-2; s17 uses none@1.6e-2 (g = GN)",
        ri["seeds"] == list(SEEDS) and ri["m"][m2]["N"] == "6.4e-2" and ri["m"][m2]["N_per_seed"][17] == "1.6e-2"
        and i17 is not None and abs(ri["m"][m2]["g"][i17] - GN[i17]) < 1e-9
        and all(abs(ri["m"][m2]["g"][i] - (50 + E_[i] + GN[i] - 72 - E_[i])) < 1e-9 for i in range(8) if i != i17))
    chk("I2: the fallback flag on m2 only; the report names the invalid run; 6.4e-2's mean over its 7 valid seeds",
        any(f.startswith(I2_FLAG) and "s17 (uses none@1.6e-2)" in f for f in ri["m"][m2]["flags"])
        and all(not any(f.startswith(I2_FLAG) for f in ri["m"][m]["flags"]) for m in mani if m != m2)
        and "Amendment V2 (I2): none@6.4e-2 not valid at s17" in ti and ri["extra_none"]["6.4e-2"]["n_valid"] == 7
        and abs(ri["m"][m2]["none_means"]["6.4e-2"] - (72 + mean([e for i, e in enumerate(E_) if i != i17]))) < 1e-9
        and ri["m"][m2]["vs_lr"]["6.4e-2"]["n"] == 7 and ri["extra_none"]["6.4e-2"]["rank_seeds"] == [x for x in SEEDS if x != 17])
    # review fix 2: a started run that is not finished (running / requeued / crashed once), has no run JSON, or has a full curve but no
    # check.json yet is MISSING: strict refuses, --allow_partial reads it as not valid (PROVISIONAL); failed twice -> invalid, no refusal
    for lab, why_, fin_ in (("unfinished (requeued / running)", "unfinished (31 evals)", False), ("no run JSON", "no run JSON", False),
                            ("a full curve, check.json not written yet", "no check.json", True)):
        Wm = world(mani, mod_v2)
        Wm["none", "6.4e-2", 20] = bad_rec("6.4e-2", 20, why_, fin_)
        try:
            readout_from(Wm, mani, none_lrs=NONE_LRS + EXTRA_NONE)
            refused, rtxt = False, ""
        except SystemExit as ex:
            refused, rtxt = str(ex).startswith("REFUSED"), str(ex)
        rm, tm = readout_from(Wm, mani, allow_partial=True, none_lrs=NONE_LRS + EXTRA_NONE)
        chk(f"I2 fix 2: none@6.4e-2 s20 {lab} -> MISSING: strict REFUSED (names it); --allow_partial PROVISIONAL, s20 on none@1.6e-2",
            refused and f"none@6.4e-2 s20 ({why_})" in rtxt and "PROVISIONAL" in tm and rm["seeds"] == list(SEEDS)
            and rm["m"][m2]["N_per_seed"][20] == "1.6e-2" and "missing, --allow_partial" in rm["extra_none"]["6.4e-2"]["not_valid"][20])
        rt, tt = readout_from(Wm, mani, none_lrs=NONE_LRS + EXTRA_NONE, xterm={("none", "6.4e-2", 20)})
        chk(f"I2 fix 2: the same {lab} run failed twice (xterm) -> invalid: strict reads 8 seeds, s20 on none@1.6e-2, no PROVISIONAL",
            rt["seeds"] == list(SEEDS) and rt["m"][m2]["N_per_seed"][20] == "1.6e-2" and "PROVISIONAL" not in tt
            and "failed twice" in rt["extra_none"]["6.4e-2"]["not_valid"][20])
    Wt = world(mani, mod_v2, drop={("none", "6.4e-2", 21)})
    Wt["none", "6.4e-2", 20] = bad_rec("6.4e-2", 20, "unfinished (31 evals)", False)
    Wt["none", "6.4e-2", 22] = bad_rec("6.4e-2", 22, "unfinished (12 evals)", False)
    Wt["none", "6.4e-2", 23] = bad_rec("6.4e-2", 23, "unfinished (5 evals)", False)
    Wt["none", "6.4e-2", 19] = bad_rec("6.4e-2", 19)   # finished, checks failed: invalid, not looked up
    mk = {f"{PREFIX}-none-lr6.4e-2-s20": [(0, "101"), (1, "102")], f"{PREFIX}-none-lr6.4e-2-s21": [(0, "111"), (1, "112")],
          f"{PREFIX}-none-lr6.4e-2-s22": [(0, "121")], f"{PREFIX}-none-lr6.4e-2-s23": [(0, "131"), (1, "132")]}
    stt = {"101": "FAILED", "102": "TIMEOUT", "111": "FAILED", "112": "RUNNING", "121": "FAILED", "131": "CANCELLED"}   # 132 unknown
    asked, mnames = [], []
    xt, xln = x_terminal(Wt, SEEDS, markers_fn=lambda n: mnames.append(n) or mk.get(n, []), states_fn=lambda ids: asked.extend(ids) or stt)
    chk("I2 fix 2: failed twice = >= 2 SUBMITTED attempts, every sacct state terminal, none unknown (s20 yes; s21 one RUNNING, s22 one "
        "attempt, s23 one unknown: missing); a finished invalid run is not looked up",
        xt == {("none", "6.4e-2", 20)} and len(xln) == 4 and sorted(mnames) == sorted(mk) and "132" in asked
        and any(x.startswith("none@6.4e-2 s20") and x.endswith("failed twice: invalid") for x in xln))
    # review fix 1 (the auditor's counter-example, cmp_old_new.out (3)): 6.4e-2 invalid at s16-18; at s19-23 none@3.2e-2 beats it by 2;
    # unpaired seed means 72.5 (3.2e-2, 8 seeds) vs 78 (6.4e-2, 5 seeds) picked 6.4e-2; paired on s19-23 the pick is 3.2e-2
    def mod_flip(L):
        L["none", "3.2e-2"][m2] = [60, 60, 60, 80, 80, 80, 80, 80]
        L["none", "6.4e-2"] = {m: ([78] * 8 if m == m2 else [40] * 8) for m in mani}
        L["all6", P_LR][m2] = [62, 62, 62, 82, 82, 82, 82, 82]
    Wf = world(mani, mod_flip)
    for s in (16, 17, 18):
        Wf["none", "6.4e-2", s] = bad_rec("6.4e-2", s)
    rf, tf = readout_from(Wf, mani, none_lrs=NONE_LRS + EXTRA_NONE)
    chk("I2 fix 1: the auditor's counter-example -> N_m = 3.2e-2 at every seed (paired on s19-23: 80 vs 78), g = 2 at every seed",
        rf["m"][m2]["N"] == "3.2e-2" and all(rf["m"][m2]["N_per_seed"][s] == "3.2e-2" for s in SEEDS)
        and all(abs(x - 2) < 1e-9 for x in rf["m"][m2]["g"]) and rf["extra_none"]["6.4e-2"]["rank_seeds"] == [19, 20, 21, 22, 23]
        and abs(rf["m"][m2]["none_means"]["3.2e-2"] - 80) < 1e-9 and "ranked on the 5 seeds where every grid lr is valid" in tf)
    Wd = world(mani, mod_v2, drop={("none", "6.4e-2", 17)})
    try:
        readout_from(Wd, mani, none_lrs=NONE_LRS + EXTRA_NONE)
        refused = False
    except SystemExit as ex:
        refused = str(ex).startswith("REFUSED")
    rd, td = readout_from(Wd, mani, allow_partial=True, none_lrs=NONE_LRS + EXTRA_NONE)
    chk("I2: no none@6.4e-2 run at s17 -> strict REFUSED; --allow_partial keeps s17 (fallback none@1.6e-2)",
        refused and rd["seeds"] == list(SEEDS) and rd["m"][m2]["N_per_seed"][17] == "1.6e-2" and "PROVISIONAL" in td)
    Wh = world(mani, mod_v2)
    for s in (16, 17, 18, 19):
        Wh["none", "6.4e-2", s] = bad_rec("6.4e-2", s)
    rh, th = readout_from(Wh, mani, none_lrs=NONE_LRS + EXTRA_NONE)
    chk("I2: none@6.4e-2 valid at only 4 of 8 seeds -> out of the N_m grid (m2 -> 1.6e-2; m3 -> 3.2e-2 flagged as the top); 8 seeds",
        rh["seeds"] == list(SEEDS) and rh["none_grid"] == list(NONE_LRS) and rh["m"][m2]["N"] == "1.6e-2"
        and TOP_FLAG in rh["m"][m3]["flags"] and "left out of the N_m grid" in th)
    W5 = world(mani, mod_v2)
    for s in (16, 17, 18):
        W5["none", "6.4e-2", s] = bad_rec("6.4e-2", s)
    r5, _ = readout_from(W5, mani, none_lrs=NONE_LRS + EXTRA_NONE)
    chk("I2: valid at 5 of 8 -> stays in the grid; m2 N_m = 6.4e-2 with s16-18 on none@1.6e-2",
        r5["none_grid"] == list(NONE_LRS + EXTRA_NONE) and r5["m"][m2]["N"] == "6.4e-2"
        and [r5["m"][m2]["N_per_seed"][s] for s in SEEDS] == ["1.6e-2"] * 3 + ["6.4e-2"] * 5)

    # the mean6 world: the mean6-picked none would give HELPS, the per-manifold best does not
    def mod2(L):
        for m in (m0, m1, m2, m3):
            L["none", "3.2e-2"][m] = [80 + e for e in E_]
            L["all6", P_LR][m] = [82 + e + g for e, g in zip(E_, GN)]
        L["none", "3.2e-2"][m4] = [40 + e for e in E_]
        L["all6", P_LR][m4] = [47 + e + 0.5 * g for e, g in zip(E_, GN)]
    res, _ = readout_from(world(mani, mod2), mani)
    chk("mean6 world: the mean6-picked none = 3.2e-2", res["N6"] == "3.2e-2")
    chk("mean6 world: m4 NOT SHOWN vs N_m = 1.6e-2, HELPS vs the mean6 pick (descriptive only)",
        res["m"][m4]["N"] == "1.6e-2" and res["m"][m4]["word"] == "NOT SHOWN" and res["m"][m4]["vs_mean6"]["helps"])

    # ties -> the higher lr
    def mod3(L):
        L["none", "8e-3"][m5] = list(L["none", "1.6e-2"][m5])
        L["none", "3.2e-2"][m4] = list(L["none", "1.6e-2"][m4])
    res, _ = readout_from(world(mani, mod3), mani)
    chk("a tie goes to the higher lr (8e-3 = 1.6e-2 -> 1.6e-2; 1.6e-2 = 3.2e-2 -> 3.2e-2 + its flag)",
        res["m"][m5]["N"] == "1.6e-2" and res["m"][m4]["N"] == "3.2e-2" and TOP_FLAG in res["m"][m4]["flags"])

    # Amendment V1: N_m can be 4e-3; the low-edge flag iff N_m = 2e-3; a NOT SHOWN line starts with the same-lr gain and u;
    # the two qualifier lines
    def mod6(L):
        L["none", "4e-3"][m1] = [62 + e for e in E_]
        L["none", "2e-3"][m2] = [64 + e for e in E_]
        L["all6", P_LR][m3] = [47 + e + 0.5 * g for e, g in zip(E_, GN)]   # NOT SHOWN vs 1.6e-2; all6@2e-3 (= 50 + GN) is kept
    res, txt = readout_from(world(mani, mod6), mani)
    chk("V1: N_m can be 4e-3 (m1), with g vs none@4e-3 among the descriptive columns",
        res["m"][m1]["N"] == "4e-3" and res["m"][m1]["word"] == "NOT SHOWN" and "4e-3" in res["m"][m1]["vs_lr"]
        and abs(res["m"][m1]["mean_g"] - res["m"][m1]["vs_lr"]["4e-3"]["mean"]) < 1e-9)
    chk("V1: low-edge flag iff N_m = 2e-3 (m2 only; not on 4e-3 / 1.6e-2; none in the default world)",
        res["m"][m2]["N"] == "2e-3" and BOT_FLAG in res["m"][m2]["flags"] and TOP_FLAG not in res["m"][m2]["flags"]
        and all(BOT_FLAG not in res["m"][m]["flags"] for m in mani if m != m2)
        and all(BOT_FLAG not in v["flags"] for v in readout_from(world(mani), mani)[0]["m"].values()))
    lines = {m: next(ln for ln in txt.splitlines() if ln.startswith(f"  {m:<11s} ")) for m in mani}
    v3 = res["m"][m3]
    pre3 = lines[m3].split("N_m = ")[0]
    chk("V1: a NOT SHOWN line starts with the same-lr gain all6@2e-3 - none@2e-3 and u (m3: both ~+13); HELPS lines do not",
        v3["word"] == "NOT SHOWN" and f"same-lr gain all6@{SAME_LR} - none@{SAME_LR} {f1(v3['g_same_lr']['mean'])}" in pre3
        and f"u {f1(v3['mean_u'])}" in pre3 and v3["g_same_lr"]["mean"] > 10 and v3["mean_u"] > 10
        and all("same-lr gain" not in lines[m].split("N_m = ")[0] for m in (m0, m4, m5))
        and all("same-lr gain" in lines[m].split("N_m = ")[0] for m in (m1, m2)))
    rw6, tw6 = readout_from(world(mani, mod6, drop={("all6", SAME_LR, 18)}), mani)
    chk("V1: a NOT SHOWN line with an incomplete same-lr pair says so (no number)",
        "same-lr gain - (the same-lr pair is incomplete), u -" in next(ln for ln in tw6.splitlines() if ln.startswith(f"  {m3:<11s} ")))
    hl = txt.index("HEADLINE")
    chk("V1: the two qualifier lines (N_m on the same seeds; pt6comb's reading line) before the headline",
        0 < txt.index(Q_SAME_SEEDS) < hl and 0 < txt.index(reading_line(mani)) < hl)

    # twin-free primary (hcube, d6); full set otherwise
    nd = {"area": 10.0, "final3": 11.0, "tf_area": 20.0, "tf_final3": 21.0}
    chk("twin-free primary for hcube and d6, the full set for cyc12 / torus / box5 / caesar",
        all(primary_of(d, nd) == (20.0, 21.0) for d in ("hcube", "d6")) and all(primary_of(d, nd) == (10.0, 11.0) for d in
                                                                             ("cyc12", "torus", "box5", "caesar")))
    # caesar separate
    mc = MJ2.manifolds(MJ2.comp_doms(("dim", "cities", "countries", "compass", "caesar")))
    res, txt = readout_from(world(mc), mc)
    chk("caesar read out separately with its note; counted in the headline",
        "[separate] caesar (read out separately)" in txt and P3.CAESAR_NOTE in txt
        and not any(ln.startswith("  caesar") for ln in txt.splitlines()) and res["headline"]["k"] == 6)
    # device note
    al = MJ2.parse_alloc("b200:16,17,18 l40s:19,20,21 a100:22,23")
    r1, t1 = readout_from(world(mani), mani)
    r2, t2 = readout_from(world(mani, dev=al), mani)
    chk("the device note changes no word", r1["headline"] == r2["headline"] and all(
        (r1["m"][m]["word"], r1["m"][m]["int_word"]) == (r2["m"][m]["word"], r2["m"][m]["int_word"]) for m in mani))
    chk("'mixed devices' printed only for the mixed allocation", "mixed devices" in t2 and "mixed devices" not in t1)
    # per-half GATE
    lo2 = {("all6", P_LR, s): 0.3 for s in HALVES[1]}
    lo1 = {("all6", P_LR, s): 0.3 for s in HALVES[0]}
    lo_2s = {("all6", P_LR, s): 0.3 for s in (16, 17)}
    rg2, _ = readout_from(world(mani, share=lo2), mani)
    rg1, _ = readout_from(world(mani, share=lo1), mani)
    rg0, _ = readout_from(world(mani, share=lo_2s), mani)
    chk("GATE per half: 20-23 low -> GATE text; 16-19 low -> GATE text; two low seeds of a half (median 0.6) -> ok",
        rg2["gate_failed"] and rg2["headline"]["text"] == GATE_TXT and rg1["gate_failed"] and not rg0["gate_failed"]
        and rg0["headline"]["k"] == 6)

    # A2 / A3 (inherited)
    def mod4(L):
        L["all6", SAME_LR][m0] = [40 + e + 2 for e in E_]
        L[f"only-{m0}", SAME_LR][m0] = [40 + e + 12 + h for e, h in zip(E_, HN)]
        L[f"only-{m1}", SAME_LR][m1] = [40 + e - 1 for e in E_]
        L["all6", SAME_LR][m1] = [40 + e - 1 + x for e, x in zip(E_, [-2, 1.5, -1, 2, -1.5, 1, -0.5, 0.2])]
    res, _ = readout_from(world(mani, mod4), mani)
    v0, v1 = res["m"][m0], res["m"][m1]
    chk("A2: interference detected (h << -0.5 u, p small) -> out of the headline, HELPS kept",
        v0["helps"] and v0["noint"] is False and v0["int_word"] == "INTERFERENCE DETECTED" and m0 not in res["headline"]["list"])
    chk("A2: u <= 0 -> only the t clause (margin None, noint by p)", v1["margin"] is None and v1["noint"] and v1["noint_by"] == "p" and v1["neg_u_note"])
    ra3, ta3 = readout_from(world(mani, share={("all6", SAME_LR, s): 0.3 for s in HALVES[0]}), mani)
    chk("A3: AUX GATE FAILED at all6@2e-3 (headline + every m)", "AUX GATE FAILED at all6@2e-3" in ra3["headline"]["text"]
        and all(ra3["m"][m]["int_word"] == "AUX GATE FAILED at all6@2e-3" for m in mani) and ra3["headline"]["k"] is None)
    rw, _ = readout_from(world(mani, drop={("all6", SAME_LR, 18)}), mani)
    chk("A3: an incomplete pair -> WITHHELD (HELPS words kept on 8 seeds)", "WITHHELD" in rw["headline"]["text"]
        and len(rw["seeds"]) == 8 and all("WITHHELD" in rw["m"][m]["int_word"] for m in mani) and rw["headline"]["helps"] == list(mani))
    # FLOOR / CEILING (each source's definition)

    def mod5(L):
        for (a, lr), lv in L.items():
            lv[m0] = [2.0 + 0.1 * e for e in E_]
            lv[m4] = [15.0 + 0.1 * e for e in E_]
            lv[m5] = [6.0 + 0.1 * e for e in E_]
        L["none", "1.6e-2"][m1] = [96 + 0.1 * e for e in E_]
        L["all6", P_LR][m1] = [97 + 0.1 * e for e in E_]
    res, _ = readout_from(world(mani, mod5), mani)
    chk("FLOOR (pc line 5 / ro P2.floor_tags line 25 / p3 twin-free chance 5 + 5) and CEILING (none >= 95)",
        "FLOOR" in res["m"][m0]["tags"] and "FLOOR" in res["m"][m4]["tags"] and abs(res["m"][m4]["floor_line"] - 25) < 1e-9
        and "FLOOR" in res["m"][m5]["tags"] and abs(res["m"][m5]["floor_line"] - 10) < 1e-9 and "CEILING" in res["m"][m1]["tags"]
        and not res["m"][m2]["tags"])
    # refusal / allow_partial
    try:
        readout_from(world(mani, drop={("none", "8e-3", 19)}), mani)
        refused = False
    except SystemExit as ex:
        refused = str(ex).startswith("REFUSED")
    rp, tp = readout_from(world(mani, drop={("none", "8e-3", 19)}), mani, allow_partial=True)
    chk("a missing core run -> REFUSED; --allow_partial -> PROVISIONAL on 7 seeds", refused and "PROVISIONAL" in tp and len(rp["seeds"]) == 7)
    try:
        readout_from(world(mani), mani, probs=["planted problem"])
        refused = False
    except SystemExit:
        refused = True
    chk("a problem -> REFUSED", refused)

    # Amendment V3 (post-hoc none@6.4e-2 outside CELLS.json): flag off = the registered read-out with or without the runs on disk
    def mod_v3(L):
        L["none", "6.4e-2"] = {m: [(72 if m == m2 else 40) + e for e in E_] for m in mani}
    r0, t0 = readout_from(world(mani), mani)
    W3 = world(mani, mod_v3)
    W3r, W3p = split_posthoc(W3, ("6.4e-2",), NONE_LRS)
    r3, t3 = readout_from(W3r, mani)
    chk("V3: split_posthoc takes the 8 none@6.4e-2 records out; the registered read-out = the world without them (text and JSON)",
        len(W3p) == 8 and not any(k[1] == "6.4e-2" for k in W3r) and t3 == t0
        and json.dumps(r3, default=str, sort_keys=True) == json.dumps(r0, default=str, sort_keys=True)
        and posthoc_lrs(V2 / "tests" / "no_such_POSTHOC.json") == ())
    tx, ox = extended_from(W3, r3, mani, "6.4e-2")
    cx = ox["change"]
    chk("V3: extended block: m2 N_m 1.6e-2 -> 6.4e-2, g +3.25 -> -18.75, HELPS lost, 'report both'; m0 unchanged; not PROVISIONAL",
        ox["res"]["m"][m2]["N"] == "6.4e-2" and cx[m2]["N_reg"] == "1.6e-2" and abs(cx[m2]["g_reg"] - 3.25) < 1e-9
        and abs(cx[m2]["g_ext"] + 18.75) < 1e-9 and cx[m2]["sign_flip"] and cx[m2]["lost_helps"] and ox["res"]["m"][m2]["word"] == "NOT SHOWN"
        and cx[m0]["N_ext"] == "1.6e-2" and not cx[m0]["sign_flip"] and not cx[m0]["lost_helps"] and cx[m0]["word_ext"] == "HELPS"
        and not ox["provisional"] and "report both" in tx and V3_TAG in tx and "PROVISIONAL" not in tx
        and json.dumps(r3, default=str, sort_keys=True) == json.dumps(r0, default=str, sort_keys=True))
    W3m = world(mani, mod_v3, drop={("none", "6.4e-2", 20), ("none", "6.4e-2", 21)})
    r3m, t3m = readout_from(split_posthoc(W3m, ("6.4e-2",), NONE_LRS)[0], mani)   # strict: must not refuse
    tm3, om3 = extended_from(W3m, r3m, mani, "6.4e-2")
    chk("V3: 2 post-hoc runs missing -> registered strict read-out unchanged; extended PROVISIONAL, 6.4e-2 in the grid (6 of 8), s20 on 1.6e-2",
        t3m == t0 and om3["provisional"] and "PROVISIONAL" in tm3.split("EXTENDED HEADLINE")[1]
        and om3["res"]["m"][m2]["N"] == "6.4e-2" and om3["res"]["m"][m2]["N_per_seed"][20] == "1.6e-2")
    return n[0], fails


def mutation_tests():
    """Each mutant of the code above the MUTANTS marker must make the self-test fail."""
    src = Path(__file__).read_text()
    mark = "# " + "=" * 103 + " MUTANTS / self-test"
    assert src.count(mark) == 1, "the MUTANTS marker"
    head, tail = src.split(mark)
    out, tmp = [], Path(tempfile.mkdtemp(prefix="mut_", dir=str(V2 / "tests")))
    try:
        for i, (lab, old, new) in enumerate(MUTANTS):
            assert head.count(old) == 1, f"mutant {lab}: {head.count(old)} occurrences of {old!r}"
            f = tmp / f"mut{i}.py"
            f.write_text(head.replace(old, new) + mark + tail)
            sp = importlib.util.spec_from_file_location(f"pt6v2_mut{i}", f)
            mod = importlib.util.module_from_spec(sp)
            try:
                sp.loader.exec_module(mod)
                k, fl = mod.selftest(quiet=True)
                killed, why = bool(fl), (fl[:2] if fl else "SURVIVED")
            except BaseException as ex:   # noqa: BLE001  (a crash also kills the mutant)
                killed, why = True, f"raised {ex!r}"[:120]
            out.append((lab, killed, why))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# ------------------------------------------------------------------------------------------------------------ test 5 (known answers)
PC_KNOWN = {   # runs/pt6comb/readout.txt (seeds 8-15, all6@1e-3 vs none@1.6e-2): g, wins, h, u, verdict, FLOOR line
    "numbers": (52.6, 8, 0.1, 52.6, "HELPS", 5.0), "s4": (9.3, 4, 0.3, 38.6, "NOT SHOWN", 29.9), "dim": (41.4, 8, 1.6, 53.7, "HELPS", 19.9),
    "cities": (7.1, 8, 2.1, 7.0, "HELPS", 73.6), "countries": (6.4, 8, 1.0, 5.9, "HELPS", 77.2), "people": (3.0, 8, 0.3, 1.4, "HELPS", 99.9)}
P3_KNOWN = [   # runs/ptdom3/tests/readout_partial_20261118T1405.txt (seed 0): (dir, domain, key, value)
    ("ptdom3-T3-none-lr2e-3-s0", "cyc12", "area", 0.5), ("ptdom3-T3-none-lr2e-3-s0", "cyc12", "final3", 0.0),
    ("ptdom3-T3-none-lr1.6e-2-s0", "cyc12", "area", 5.8), ("ptdom3-T3-none-lr1.6e-2-s0", "cyc12", "final3", 12.6),
    ("ptdom3-T3-none-lr1.6e-2-s0", "cyc12", "tf_area", 5.5), ("ptdom3-T3-none-lr1.6e-2-s0", "cyc12", "n_tf", 53),
    ("ptdom3-T3-only-cyc12-lr4e-3-s0", "cyc12", "area", 3.9), ("ptdom3-T3-only-cyc12-lr4e-3-s0", "cyc12", "tf_area", 4.1),
    ("ptdom3-T3-none-lr1.6e-2-s0", "torus", "area", 74.4), ("ptdom3-T3-none-lr1.6e-2-s0", "torus", "tf_area", 74.0),
    ("ptdom3-T3-none-lr1.6e-2-s0", "torus", "n_tf", 450),
    ("ptdom3-T3-only-torus-lr1e-3-s0", "torus", "area", 82.9), ("ptdom3-T3-only-torus-lr1e-3-s0", "torus", "tf_area", 82.7),
    ("ptdom3-T3-none-lr1.6e-2-s0", "hcube", "area", 79.7), ("ptdom3-T3-none-lr1.6e-2-s0", "hcube", "tf_area", 75.7),
    ("ptdom3-T3-none-lr1.6e-2-s0", "hcube", "n_tf", 66),
    ("ptdom3-T3-none-lr1.6e-2-s0", "d6", "area", 9.3), ("ptdom3-T3-none-lr1.6e-2-s0", "d6", "tf_area", 4.8),
    ("ptdom3-T3-none-lr1.6e-2-s0", "d6", "noaae_area", 8.4), ("ptdom3-T3-none-lr1.6e-2-s0", "d6", "aae_area", 26.5),
    ("ptdom3-T3-none-lr1.6e-2-s0", "d6", "n_tf", 28)]


def known():
    """Test 5. -> (n checks, failures)."""
    n, fails = [0], []

    def chk(name, cond, info=""):
        n[0] += 1
        if not cond:
            fails.append(name)
        print(f"  {'ok ' if cond else 'FAIL'} {name}" + (f"  [{info}]" if info else ""), flush=True)
    # --- pt6comb C: this loader + verdict vs readout_pt6comb.load / verdict and readout.txt
    root = B / "runs/pt6comb/out"
    print("pt6comb: loading with readout_pt6comb.load and load_v2(spec_pc) ...", flush=True)
    Rp = PC.load(root, guard=True)
    Rm = load_v2(root, spec_pc(), guard=True)
    chk(f"pt6comb: same cells ({len(Rp)})", set(Rp) == set(Rm))
    dv = [k for k in Rp if (Rp[k]["valid"], Rp[k]["why"]) != (Rm[k]["valid"], Rm[k]["why"])]
    chk(f"pt6comb: valid / why identical on every cell ({sum(r['valid'] for r in Rp.values())} valid)", not dv, str(dv[:3]))
    dd, nn = [], 0
    for k, r in Rp.items():
        if r.get("finished"):
            for m in PC.MANI:
                for f in ("area", "final3", "floor"):
                    nn += 1
                    if abs(r[f][m] - Rm[k][f][m]) > 1e-9:
                        dd.append((k, m, f, r[f][m], Rm[k][f][m]))
            if r["problems"] != Rm[k]["problems"] or r["grad"] != Rm[k]["grad"]:
                dd.append((k, "problems/grad"))
    chk(f"pt6comb: per-run area / final3 / FLOOR line = readout_pt6comb.run_metrics ({nn} numbers) and problems / grad", not dd, str(dd[:3]))
    ng, gd = 0, []
    for (arm, lr, s), r in sorted(Rm.items()):
        if arm == "none" or not r.get("finished"):
            continue
        d = Path(r["dir"])
        ck = json.loads((d / "check.json").read_text())
        rj = json.loads(run_json(d, spec_pc(), arm, lr, s).read_text())
        a, b = PC.geo_a6_multi(rj, ck, arm), geo_a6_v2(rj, ck, arm, spec_pc())
        ng += 1
        if a != b:
            gd.append((arm, lr, s))
        if ng >= 24:
            break
    chk(f"pt6comb: geo_a6_v2 = geo_a6_multi on {ng} prior runs", not gd, str(gd[:3]))
    chk("pt6comb: integrity problems identical", PC.integrity(Rp, set(range(8, 16))) == PC.integrity(Rm, set(range(8, 16))))
    rp = PC.verdict(Rp, {"pick_all6": "1e-3", "pick_none": "1.6e-2", "flags": []})
    rv = verdict_v2(Rm, PC.MANI, seeds=tuple(range(8, 16)), halves=(tuple(range(8, 12)), tuple(range(12, 16))), P="1e-3", same="2e-3",
                    none_lrs=("1.6e-2",))
    keys = ("mean_g", "wins_g", "p_help", "mean_h", "mean_u", "margin", "p_int", "helps", "noint", "noint_by", "p_help_holm", "tags")
    bad = [(m, k_, rp["m"][m][k_], rv["m"][m][k_]) for m in PC.MANI for k_ in keys
           if not (rp["m"][m][k_] == rv["m"][m][k_] or (isinstance(rp["m"][m][k_], float) and abs(rp["m"][m][k_] - rv["m"][m][k_]) < 1e-9))]
    chk("pt6comb: verdict_v2 = readout_pt6comb.verdict per manifold (g, wins, p, h, u, margin, p_int, words, Holm, tags)", not bad, str(bad[:3]))
    chk("pt6comb: FLOOR line = readout_pt6comb.verdict's", all(abs(rp["m"][m]["levels"]["floor_line"] - rv["m"][m]["floor_line"]) < 1e-9 for m in PC.MANI))
    chk("pt6comb: gates and the headline list = readout_pt6comb.verdict's", rp["gate_failed"] == rv["gate_failed"]
        and rp["gate_same_lr_failed"] == rv["gate_same_lr_failed"] and rp["headline"]["list"] == rv["headline"]["list"],
        f"{rv['headline']['text']}")
    txtbad = []
    for m, (g, w, h, u, word, fl) in PC_KNOWN.items():
        v = rv["m"][m]
        got = (round(v["mean_g"], 1), v["wins_g"], round(v["mean_h"], 1), round(v["mean_u"], 1), v["word"], round(v["floor_line"], 1))
        if got != (g, w, h, u, word, fl):
            txtbad.append((m, got))
    chk("pt6comb: = runs/pt6comb/readout.txt (g, wins, h, u, word, FLOOR line per manifold; 5 of 6; people FLOOR)", not txtbad
        and rv["headline"]["k"] == 5 and "FLOOR" in rv["m"]["people"]["tags"]
        and abs(rv["mean6_gain"]["mean"] - 19.98) < 0.005 and [round(x, 2) for x in rv["mean6_gain"]["ci95"]] == [16.27, 23.70], str(txtbad[:3]))
    rl = [ln.strip() for ln in (B / "runs/pt6comb/readout.txt").read_text().splitlines() if ln.strip().startswith("reading: ")]
    chk("pt6comb: Q_READING = runs/pt6comb/readout.txt's reading line verbatim (V1 item 4)", rl == [Q_READING], str(rl)[:120])
    # --- the canary path on a real pt6comb run
    k8 = ("all6", "1e-3", 8)
    rc, ctext, nums = canary(Path(Rm[k8]["dir"]))
    chk("canary path on pt6comb all6@1e-3 s8: 6 numbers = the loader's areas", rc == 0 and all(abs(nums[m] - Rp[k8]["area"][m]) < 1e-9 for m in PC.MANI),
        ctext.splitlines()[-1])
    # --- ptdom2 Part R + months: dom_metrics vs readout_ptdom2.load2
    print("ptdom2: loading with readout_ptdom2.load2 ...", flush=True)
    R2 = P2.load2(B / "runs/ptdom2/out", "ptdom2", True)
    want = set(P2.R_DOMS) | {"months"}
    nr, b2, doms_seen = 0, [], set()
    for (b, arm, lr, s), rec in sorted(R2.items()):
        if not (rec["valid"] and rec.get("finished")):
            continue
        ds = [d for d in P2.BATCHES[b] if d in want]
        if not ds:
            continue
        rj = json.loads(P2.run_json2(Path(rec["dir"]), "ptdom2", b, arm, lr, s).read_text())
        for d in ds:
            o, p = dom_metrics(rj, d)
            nr += 1
            doms_seen.add(d)
            sub = RO.run_subsets(rec["cmp"][d], meta1(d))[0] if d in rec.get("cmp", {}) else None
            ref = (rec["area"][d], rec["final3"][d], rec["chance"][d], rec["heur"].get(d), rec["kshort"].get(d) if d in P2.KSTEP else None,
                   sub["h3_held"] if sub else None)
            got = (o["area"], o["final3"], o["fl"]["chance"], o["fl"]["heur"], o["fl"]["kshort"], o["fl"]["h3"])
            if any((x is None) != (y is None) or (x is not None and np.max(np.abs(np.asarray(x, float) - np.asarray(y, float))) > 1e-9)
                   for x, y in zip(ref, got)) or p:
                b2.append((b, arm, lr, s, d, ref, got, p[:1]))
    chk(f"ptdom2: area / final3 / chance / heuristic / kshort / h3 = readout_ptdom2.load2 on {nr} (run, domain) pairs "
        f"({sorted(doms_seen)})", nr > 0 and not b2, str(b2[:2]))
    chk("ptdom2: every Part R domain + months covered", doms_seen == want, str(sorted(want - doms_seen)))
    # --- ptdom3 screen: = the saved partial read-out text
    b3 = []
    for dn, d, key, val in P3_KNOWN:
        dd_ = B / "runs/ptdom3/out" / dn
        f = P3.run_json(dd_)
        if f is None:
            b3.append((dn, d, key, "no run JSON"))
            continue
        o, p = dom_metrics(json.loads(f.read_text()), d)
        x = {"area": o["sec"]["full_area"], "final3": o["p3"]["final3"], "tf_area": o["sec"]["tf_area"], "n_tf": o["sec"]["n_tf"],
             "aae_area": o["sec"].get("aae_area"), "noaae_area": o["sec"].get("noaae_area")}[key]
        if x is None or (round(x, 1) if key != "n_tf" else x) != val:
            b3.append((dn, d, key, x, val))
        prim = o["p3"]["tf_area"] if d in PRIMARY_TF else o["p3"]["area"]
        if o["area"] != prim:
            b3.append((dn, d, "primary", o["area"], prim))
    chk(f"ptdom3: = readout_partial_20261118T1405.txt ({len(P3_KNOWN)} numbers; primary = twin-free for hcube / d6)", not b3, str(b3[:3]))
    rc, ctext, nums = canary(B / "runs/ptdom3/pilot/ptdom3-canary-T3-alltoy3-lr2e-3-s0")
    chk("canary path on the ptdom3 canary run: a number for each of its 7 manifolds", rc == 0 and len(nums) == 7, ctext.splitlines()[-1])
    return n[0], fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root")
    ap.add_argument("--comp")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--canary")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--posthoc_none", choices=EXTRA_NONE, help="AUDIT.md Amendment V3: add block 3, the grid + this post-hoc none lr")
    a = ap.parse_args()
    print(f"readout_pt6v2.py sha256[:12] {sha12(__file__)}; sources " + ", ".join(f"{k} {v[1]}" for k, v in SRC.items())
          + f"; make_jobs_pt6v2.py {sha12(MJ2.__file__)}", flush=True)
    if a.canary:
        rc, txt, _ = canary(a.canary)
        print(txt)
        sys.exit(rc)
    if a.selftest:
        k, fl = selftest()
        print(f"SELFTEST (test 4): {k - len(fl)} of {k} checks pass" + (f"; FAILED: {fl}" if fl else ""))
        mt = mutation_tests()
        for lab, killed, why in mt:
            print(f"  mutant '{lab}': {'killed' if killed else 'SURVIVED'} ({why})")
        nk = sum(x[1] for x in mt)
        print(f"MUTATIONS: {nk} of {len(mt)} killed")
        sys.exit(0 if not fl and nk == len(mt) else 1)
    if a.known:
        k, fl = known()
        print(f"KNOWN ANSWERS (test 5): {k - len(fl)} of {k} checks pass" + (f"; FAILED: {fl}" if fl else ""))
        sys.exit(0 if not fl else 1)
    if a.posthoc_none:
        print(f"--posthoc_none {a.posthoc_none}: block 3 = {V3_TAG} ({POSTHOC_FILE})", flush=True)
    res, txt = readout(a.root, a.allow_partial, comp=a.comp, posthoc=a.posthoc_none)
    print(txt)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
