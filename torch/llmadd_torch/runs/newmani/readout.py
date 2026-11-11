"""runs/newmani Stage 1 read-out: the PRE-REGISTERED reading of AUDIT.md ("Reading"), from the run JSONs. No training.
Uses the pinned toy_struct (code/tools) for compare_values / upto / late_change / auc / medians (one definition each).

    python readout.py                       # reads runs/newmani/out, writes readout.md / readout.json next to this file
    python readout.py --root DIR            # another root (tests); outputs go to DIR/..
    python readout.py --selftest            # synthetic cells with known verdicts (must pass before any reading is used)
    python readout.py --tools DIR           # toy_struct from DIR instead of code/tools (pre-snapshot tests only)
    python readout.py --a2 [--root DIR]     # Amendment A2 (2026-10-25): readout_a2.md / .json = the A1 text + per cell an
                                            # A1 | A2 table + both selections (post hoc for Stage 1; the default call above
                                            # is the pre-registered A1 reading, byte-identical to c1ce6fd75bd3's)
    python readout.py --a2 --trainable_thr X   # sensitivity only (X != 0.99 is labelled 'SENSITIVITY', never the rule)

Definitions (one place; AUDIT.md "Metrics"):
- B = the prior arms' budget (50k); none runs 4 B. t_grok = first eval with held-out test >= 0.95 (inf = never).
- AUC_B / t_grok_B = toy_struct.upto(): the evals at steps 0, 100, ..., B - 100 of any run (prior and none read alike).
- final = the last eval of the run; final_strict / final_nonimplied = its test_strict / test_nonimplied.
- late gain / CE drop = toy_struct.late_change() over the last 25 % of 4 B (none).
- chance = max(1 / n_out, median majority-of-train baseline); floor line = max(chance, median one-step elimination
  heuristic on the strict set) + 0.05 (design 3.1).
- implied / non-implied = held-out cells a pure Latin-square filler derives / does not (runs/toyf84fu/latin.py's
  definition; toy_struct.nm_latin); assoc closure = toy_struct.closure (groups: associativity fixpoint; box: linear)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ARGS = sys.argv[1:]
TOOLS = Path(ARGS[ARGS.index("--tools") + 1]) if "--tools" in ARGS else HERE / "code" / "tools"
sys.path.insert(0, str(TOOLS))
import numpy as np  # noqa: E402
import toy_struct as ts  # noqa: E402

AUC_MIN, CAP, RISE_ACC, RISE_CE = ts.AUC_MIN, 0.20, 0.02, 0.01
NONE_GRID = [(lr, wd) for lr in (1e-3, 3e-3, 1e-2, 3e-2) for wd in (0.1, 1.0, 3.0)]
FLAT_EXT = [(1e-1, wd) for wd in (0.1, 1.0, 3.0)] + [(lr, 10.0) for lr in (1e-3, 3e-3, 1e-2, 3e-2)]
PRIOR_LRS = (1e-3, 3e-3, 1e-2)
ROLE_OF = lambda arm, true: {f"{true}@always": "assign", "specrand@always": "specrand", "shuf@always": "shuf",
                             f"shape:{true}@always": "shape", "shape:sphere@always": "sphere", "shape:line@always": "line"}.get(arm)
TIERS = {"L": ("assign", ("specrand", "shuf")), "S": ("shape", ("sphere", "line"))}
SHARES = ("neg_share", "coset3_share", "coset19_share", "refl_share", "parity_share", "drop_share")
N_SEEDS = 3
# Amendment A1 (2026-10-21, before any newmani result; audit B1 + F89): the best none config's last-25 % test-CE drop
# counts only while that config's late held-out CE is <= log(n_out) + CE_GATE_NATS (near-uniform or better). A memoriser
# that is confidently wrong on held-out (CE 13-55 nats) drifts by nats with no change in accuracy: there the component is n/a.
CE_GATE_NATS = 1.0
# positive controls (design 3.1; box5 audit I3): the none configs expected in every 'pos*' level; OK iff >= 1 (level,
# config) groks on >= 2/3 of its seeds with every seed present
POS_CFG = {"box5": ((1e-3, 1.0), (1e-2, 0.1))}
POS_DEFAULT = ((1e-3, 1.0),)
med = ts.median_vals
# Amendment A2 (2026-10-25, before Stage 2 and before the box5-m6 extension is read; results audit F1): a none config is
# TRAINABLE iff the median over its seeds of final_train >= TRAINABLE_THR. Rule 1's CE component and rule 2's edge test
# are read on the REFERENCE config = the trainable config with the most groks, then the highest median AUC 4B (then the
# lower lr); every other rule-1 part stays over every config; no trainable config -> the wall is n/a (never a wall).
# The paired-test comparator is NOT restricted (AUDIT.md A2 item 5).
TRAINABLE_THR = 0.99
A2_THR = float(ARGS[ARGS.index("--trainable_thr") + 1]) if "--trainable_thr" in ARGS else TRAINABLE_THR
assert 0.0 < A2_THR <= 1.0, A2_THR
assert "--trainable_thr" not in ARGS or "--a2" in ARGS, "--trainable_thr only with --a2 (it changes nothing else)"


def trainable(rows_train, thr):
    """A2: {config: median final_train} -> the configs at or above thr (sorted)."""
    return sorted(k for k, v in rows_train.items() if v >= thr)


def reference_config(rows, ok):
    """A2's rule-1/2 reference config among the trainable configs ok (None if there is none); A1's order of keys."""
    return max(ok, key=lambda k: (rows[k]["grok"], rows[k]["auc4"], -k[0])) if ok else None


def bracket_of(rows, bk, chance):
    """Rule 2 (bracket) of a none sweep: flat (every config 0 groks, AUC 4B <= chance + AUC_MIN) -> closed only with the
    extension; else closed iff the best config bk is not on an edge of the grid that ran (bk None: A2 n/a)."""
    flat = all(x["grok"] == 0 and x["auc4"] <= chance + AUC_MIN for x in rows.values())
    if flat:
        return "flat over the extended grid" if all(g in rows for g in FLAT_EXT) else "OPEN (flat: rule-2 extension lr 1e-1 / wd 10 needed)"
    if bk is None:
        return "n/a (no trainable none config)"
    lrs, wds = sorted({k[0] for k in rows}), sorted({k[1] for k in rows})
    edges = [e for e, hit in (("lr top", bk[0] == lrs[-1]), ("lr bottom", bk[0] == lrs[0]), ("wd top", bk[1] == wds[-1]),
                              ("wd bottom", bk[1] == wds[0])) if hit]
    return "closed" if not edges else "OPEN at " + ", ".join(edges)


def late_ce(r, B4):
    """Mean held-out CE over the last evals before B4 (the same evals late_change() ends on); inf if non-finite / absent."""
    e = r.get("eval_every", 100)
    v = [c for c in r["curve"] if c["step"] < B4 and c["step"] % e == 0]
    late = [c for c in v if c["step"] >= 0.75 * B4]
    k = min(5, len(late) // 2)
    if k == 0:
        return math.inf
    x = float(np.mean([c["test_ce"] for c in late[-k:]]))
    return x if math.isfinite(x) else math.inf


def wall_reasons(rows, bk, n_out, floor, gate=True, cap_key="final_max"):
    """Rule-1 wall reasons (scout version) of a none sweep {config: row}; row keys grok, n, final_max, gain, ce_drop,
    ce_late, strict. gate=False = the rule before Amendment A1 (CE component always on). Returns (reasons, ce info).
    bk = the config the CE component is read on (A1: best none; A2: the reference config, None = no trainable config ->
    the CE component n/a). cap_key 'final_max' = the scout's every-seed cap; 'final_med' = G1's median cap (Stage 2)."""
    assert cap_key in ("final_max", "final_med"), cap_key
    thr = math.log(n_out) + CE_GATE_NATS
    on = bk is not None and ((not gate) or rows[bk]["ce_late"] <= thr)
    fires = on and rows[bk]["ce_drop"] >= RISE_CE
    why = {}
    g = [k for k, x in rows.items() if x["grok"] > math.floor(0.2 * x["n"])]
    if g:
        why["grok"] = "none groks: " + ", ".join(f"lr {k[0]:g} wd {k[1]:g} {rows[k]['grok']}/{rows[k]['n']}" for k in g)
    c = [k for k, x in rows.items() if x[cap_key] > CAP]
    if c and cap_key == "final_max":
        why["cap"] = f"a none seed above {CAP:.2f}: " + ", ".join(f"lr {k[0]:g} wd {k[1]:g} max {rows[k]['final_max']:.2f}" for k in c)
    elif c:
        why["cap"] = f"median final above {CAP:.2f}: " + ", ".join(f"lr {k[0]:g} wd {k[1]:g} median {rows[k]['final_med']:.2f}" for k in c)
    ra = [k for k, x in rows.items() if x["gain"] >= RISE_ACC]
    if ra:
        why["gain"] = "still rising (accuracy): " + ", ".join(f"lr {k[0]:g} wd {k[1]:g} gain {rows[k]['gain']:+.3f}" for k in ra)
    if fires:
        why["ce"] = (f"best config lr {bk[0]:g} wd {bk[1]:g} test CE still falling: drop {rows[bk]['ce_drop']:+.3f} nats "
                     f"(late CE {rows[bk]['ce_late']:.2f} <= gate {thr:.2f})")
    if floor is not None:
        st = [k for k, x in rows.items() if x["strict"] is not None and x["strict"] > floor]
        if st:
            why["strict"] = "strict set above the floor line: " + ", ".join(f"lr {k[0]:g} wd {k[1]:g} {rows[k]['strict']:.3f}" for k in st)
    ce = {"gate": thr, "late_ce_best": rows[bk]["ce_late"] if bk is not None else None, "on": bool(on),
          "drop_best": rows[bk]["ce_drop"] if bk is not None else None, "fires": bool(fires)}
    return why, ce


def none_row(rr, B, B4):
    """One none config's row (its runs rr, seed order): groks in 4 B, final test median / max, final strict / non-implied
    medians, AUC 4B / AUC_B medians, last-25 % gain / CE drop medians, late CE median. Shared by Stage 1 and Stage 2."""
    lc = [ts.late_change(r, B4) for r in rr]
    return {"n": len(rr), "grok": sum(r["t_grok"] is not None for r in rr), "final_med": med([r["final_test"] for r in rr]),
            "final_max": max(r["final_test"] for r in rr), "strict": med([final_of(r, "test_strict") for r in rr]),
            "nonimp": med([final_of(r, "test_nonimplied") for r in rr]), "auc4": med([ts.auc(r) for r in rr]),
            "aucB": med([ts.upto(r, B)[1] for r in rr]), "gain": med([g for g, _ in lc]), "ce_drop": med([c for _, c in lc]),
            "ce_late": med([late_ce(r, B4) for r in rr])}


def task_key(r):
    return r["group"] if r["task"] == "grp" else "box" + str(r["box"][0][1] * 2 + 1)


def level_of(r, scan):
    for k, c in scan["cells"].items():
        if k.split("|")[0] == task_key(r) and abs(c["frac"] - r["frac"]) < 1e-12:
            return str(c["m_level"])
    return f"f{r['frac']}"


def final_of(r, key):
    v = r["curve"][-1].get(key)
    return None if v is None else float(v)


def read(root: Path, scan: dict, a2: bool = False, thr: float = TRAINABLE_THR, measure: str = "final") -> tuple[str, dict]:
    """a2=False: the pre-registered A1 reading (text and JSON exactly as c1ce6fd75bd3). a2=True: the same text, plus per cell
    an A1 | A2 table and J['cells'][..]['a2'], plus the A2 selection; thr / measure = A2's trainability rule (registered:
    0.99 on 'final' = final_train; other values are sensitivity checks, labelled)."""
    assert measure in ("final", "late"), measure
    rs = [json.loads(p.read_text()) for p in sorted(root.glob("*/run_*.json"))]
    L = [f"# newmani Stage 1 read-out ({root})\n\n"]
    if not rs:
        return "".join(L) + "no runs\n", {"root": str(root), "n_runs": 0}
    shas = sorted({str(r.get("code_sha")) for r in rs})
    J = {"root": str(root), "n_runs": len(rs), "code_sha": shas, "cells": {}, "tasks": {}}
    if len(shas) != 1:
        return "".join(L) + f"NO VERDICT: {len(rs)} runs from {len(shas)} code_sha values {shas}\n", J
    L.append(f"{len(rs)} runs, code_sha {shas[0]}. Stage 1 is a SCOUT (3 seeds): it selects (task, m) cells for Stage 2 and never "
             "supports a 'rescue' claim (rescue_standard.md: fresh confirm seeds, closed brackets, paired tests at n >= 5).\n\n")
    cells = {}
    for r in rs:
        assert r["task"] in ts.NM_TASKS, ("not a newmani run", r["task"])
        cells.setdefault((task_key(r), level_of(r, scan)), []).append(r)
    for (tk, lvl) in sorted(cells, key=lambda k: (k[0], not k[1].startswith("pos"), int(k[1]) if k[1].isdigit() else 99, k[1])):
        sc = scan["cells"].get(f"{tk}|{lvl}", {})
        exp = {x["seed"] for x in sc["seeds"]} if "seeds" in sc else None
        J["cells"][f"{tk}|{lvl}"] = read_cell(tk, lvl, cells[(tk, lvl)], L, exp, a2=(thr, measure) if a2 else None)
    L.append("\n## Selection for Stage 2 (pre-registered, per task and tier)\n\n| task | positive control | tier L (assign) | tier S (shape) |\n|---|---|---|---|\n")
    for tk in sorted({k[0] for k in cells}):
        J["tasks"][tk] = sel = select(tk, {lv: J["cells"][f"{tk}|{lv}"] for (t, lv) in cells if t == tk})
        L.append(f"| {tk} | {sel['pos']} | {sel['L']} | {sel['S']} |\n")
    if a2:   # the same selection rule on the A2 walls / tiers (the positive controls are not part of A2)
        lab = ("" if (thr, measure) == (TRAINABLE_THR, "final") else
               f" -- SENSITIVITY (threshold {thr:g} on {measure} train), NOT the registered A2")
        L.append(f"\n## Selection under Amendment A2 (trainable = median {'final_train' if measure == 'final' else 'last-25 % mean train'} "
                 f">= {thr:g}; post hoc for Stage 1, F94's pre-registered verdicts stand){lab}\n\n"
                 "| task | positive control | tier L: A1 (pre-registered) | tier L: A2 | tier S: A1 (pre-registered) | tier S: A2 |\n"
                 "|---|---|---|---|---|---|\n")
        J["a2"] = {"thr": thr, "measure": measure, "registered": not lab, "tasks": {}}
        for tk in sorted({k[0] for k in cells}):
            sub = {}
            for (t, lv) in cells:
                if t != tk:
                    continue
                c = J["cells"][f"{tk}|{lv}"]
                sub[lv] = {**c, "tier": c["a2"]["tier"], "wall": c["a2"]["wall"]} if "a2" in c else c
            J["a2"]["tasks"][tk] = s2 = select(tk, sub)
            s1 = J["tasks"][tk]
            assert s2["pos"] == s1["pos"], (tk, s1["pos"], s2["pos"])
            L.append(f"| {tk} | {s1['pos']} | {s1['L']} | {s2['L']} | {s1['S']} | {s2['S']} |\n")
    return "".join(L), J


def read_cell(tk, lvl, rs, L, exp_seeds=None, a2=None):
    C = {"task": tk, "level": lvl}
    fr = sorted({r["frac"] for r in rs}); assert len(fr) == 1, fr
    by_seed = {}
    for r in rs:   # split facts: one split per seed, identical in every run of that seed
        f = {k: r.get(k) for k in ("split_sha", "n_train", "m_per_item", "closure", "closure_latin", "nullity", "latin_share",
                                   "latin_level", "chance_nonimp", "elim_strict", "chance_majority", "n_mask", "n_out")}
        assert by_seed.setdefault(r["seed"], f) == f, ("split facts differ within a seed", tk, lvl, r["seed"])
    n_out = by_seed[min(by_seed)]["n_out"]
    chance = max(1 / n_out, med([f["chance_majority"] for f in by_seed.values()]))
    floor = max(chance, med([f["elim_strict"] for f in by_seed.values()])) + AUC_MIN
    C.update({"frac": fr[0], "chance": chance, "floor_line": floor, "splits": by_seed})
    L.append(f"\n## {tk}, m level {lvl} (frac {fr[0]})\n\nsplits (per seed): " + "; ".join(
        f"s{s} n_train {f['n_train']} (m {f['m_per_item']:.2f}), assoc closure {f['closure']:.3f}"
        + (f", assoc+latin {f['closure_latin']:.3f}" if f.get("closure_latin") is not None else f", nullity {f['nullity']}")
        + f", Latin-only implied {f['latin_share']:.3f} (n {f['n_mask']['implied']} implied / {f['n_mask']['nonimplied']} non-implied), "
          f"strict n {f['n_mask']['strict']}" for s, f in sorted(by_seed.items()))
        + f"\n\nchance {chance:.4f}; floor line {floor:.4f}\n\n")
    none = {}
    prior = {}
    for r in rs:
        if r["arm"] == "none":
            none.setdefault((r["lr"], r["wd"]), {})[r["seed"]] = r
        else:
            role = ROLE_OF(r["arm"], r["true_target"])
            assert role is not None and r["wd"] == 1.0 and r["lam"] == 1.0, ("unexpected prior run", r["arm"], r["wd"], r["lam"])
            prior.setdefault(role, {}).setdefault(r["lr"], {})[r["seed"]] = r
    seeds_ok = lambda d: len(d) == N_SEEDS and (exp_seeds is None or set(d) == exp_seeds)   # audit I1: every seed present
    if lvl.startswith("pos"):   # positive control (design 3.1): none must learn here, else check the pipeline
        C["pos"] = {}
        for cfg in POS_CFG.get(tk, POS_DEFAULT):
            d = none.get(cfg, {})
            g = sum(r["t_grok"] is not None for r in d.values())
            st = ("INCOMPLETE" if not seeds_ok(d) else "groks" if g >= math.ceil(2 * N_SEEDS / 3) else "fails")
            C["pos"][f"lr{cfg[0]:g}-wd{cfg[1]:g}"] = {"status": st, "grok": g, "n": len(d)}
            L.append(f"positive control none lr {cfg[0]:g} wd {cfg[1]:g}: groks {g}/{len(d)} (seeds {sorted(d)}), final test median "
                     f"{med([r['final_test'] for r in d.values()]) if d else float('nan'):.3f} -> {st}\n")
        return C
    Bs = sorted({r["steps"] for d in prior.values() for x in d.values() for r in x.values()})
    B4s = sorted({r["steps"] for d in none.values() for r in d.values()})
    if len(Bs) != 1 or len(B4s) != 1 or B4s[0] != 4 * Bs[0]:
        L.append(f"NO VERDICT: prior budgets {Bs}, none budgets {B4s} (needs one B and none at 4 B)\n")
        C["verdict"] = "no verdict (budgets)"; return C
    B, B4 = Bs[0], B4s[0]
    # ---- none sweep
    rows = {}
    L.append("| none lr | wd | n | groks in 4B | final test median / max | final strict median | final non-implied median | AUC 4B | AUC_B "
             "| last 25%: gain / CE drop |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for k in sorted(none):
        x = rows[k] = none_row([none[k][s] for s in sorted(none[k])], B, B4)
        L.append(f"| {k[0]:g} | {k[1]:g} | {x['n']} | {x['grok']}/{x['n']} | {x['final_med']:.3f} / {x['final_max']:.3f} | {x['strict']:.3f} "
                 f"| {x['nonimp']:.3f} | {x['auc4']:.3f} | {x['aucB']:.3f} | {x['gain']:+.3f} / {x['ce_drop']:+.3f} (late CE {x['ce_late']:.2f}) |\n")
    n = min(x["n"] for x in rows.values())
    missing = [f"lr {g[0]:g} wd {g[1]:g}" for g in NONE_GRID if g not in rows] + \
        [f"lr {k[0]:g} wd {k[1]:g} seeds {sorted(none[k])}" for k in rows if not seeds_ok(none[k])]
    bk = max(rows, key=lambda k: (rows[k]["grok"], rows[k]["auc4"], -k[0]))
    whyd, ce = wall_reasons(rows, bk, n_out, floor, gate=True)
    old, _ = wall_reasons(rows, bk, n_out, floor, gate=False)   # the rule before Amendment A1, reported only
    why = list(whyd.values())
    wall = not why
    ce["alone_vetoes"] = set(whyd) == {"ce"}
    ce["ungated_alone_would_veto"] = set(old) == {"ce"}
    bracket = bracket_of(rows, bk, chance)
    nb = max(rows, key=lambda k: (rows[k]["aucB"], -k[0]))
    C.update({"B": B, "none": {f"lr{k[0]:g}-wd{k[1]:g}": v for k, v in rows.items()}, "wall": wall, "wall_why": why,
              "provisional": missing, "none_best": list(bk), "bracket": bracket, "none_best_aucB": list(nb), "ce": ce,
              "wall_ungated": not old})
    L.append(f"\n**wall (Stage-1 scout: every none config 0 groks in 4B, every seed final <= {CAP:.2f}, not rising, strict <= floor line): "
             f"{wall}**" + ("" if wall else " (" + "; ".join(why) + ")") + f"; best none {bk[0]:g} / {bk[1]:g}, bracket {bracket}"
             + (f"; PROVISIONAL (missing {missing})" if missing else "") + "\n\n")
    L.append(f"CE component (Amendment A1): best config late CE {ce['late_ce_best']:.2f} vs gate log(n_out) + {CE_GATE_NATS:g} = "
             f"{ce['gate']:.2f} -> {'on' if ce['on'] else 'n/a'}; drop {ce['drop_best']:+.3f}; "
             + ("**the CE component ALONE vetoes this wall candidate**" if ce["alone_vetoes"] else "does not alone veto")
             + ("; the ungated pre-A1 rule would have vetoed it on CE alone" if ce["ungated_alone_would_veto"] else "") + "\n\n")
    # ---- prior roles: best lr by median AUC_B
    L.append("| role | tier | lr chosen (AUC_B per lr) | lr bracket | groks in B | AUC_B | final test | final strict | final non-implied "
             "| own objective end | CKA to the true target end | wrong-answer shares (end) |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    pick = {}
    for role in ("assign", "specrand", "shuf", "shape", "sphere", "line"):
        if role not in prior:
            continue
        st = {lr: med([ts.upto(r, B)[1] for r in d.values()]) for lr, d in prior[role].items()}
        lr = max(st, key=lambda k: (st[k], -k))
        rr = [prior[role][lr][s] for s in sorted(prior[role][lr])]
        gl = {k: sum(ts.upto(r, B)[0] < math.inf for r in d.values()) for k, d in prior[role].items()}
        inc = [f"lr {k:g} missing" for k in PRIOR_LRS if k not in prior[role]] + \
              [f"lr {k:g} seeds {sorted(d)}" for k, d in sorted(prior[role].items()) if not seeds_ok(d)]
        if inc:
            lb = "INCOMPLETE"
        elif max(st.values()) - min(st.values()) < 1e-9 or (max(gl.values()) == 0 and max(st.values()) <= floor):
            lb = "flat (no optimum: every lr at the same AUC_B or at the floor)"   # audit M6
        else:
            lb = "closed" if min(PRIOR_LRS) < lr < max(PRIOR_LRS) else "OPEN at " + ("bottom" if lr == min(PRIOR_LRS) else "top")
        tg = [ts.upto(r, B)[0] for r in rr]
        x = pick[role] = {"lr": lr, "aucB_per_lr": {f"{k:g}": v for k, v in st.items()}, "lr_bracket": lb, "n": len(rr),
                          "incomplete": inc,
                          "grokB": sum(t < math.inf for t in tg), "aucB": st[lr], "final": med([r["final_test"] for r in rr]),
                          "strict": med([final_of(r, "test_strict") for r in rr]), "nonimp": med([final_of(r, "test_nonimplied") for r in rr]),
                          "geo": med([r["curve"][-1]["geo"] for r in rr if r["curve"][-1].get("geo") is not None] or [float("nan")]),
                          "cka_true": med([r["curve"][-1].get(f"cka_{r['true_target']}", float("nan")) for r in rr]),
                          "shares": {k: med([r["curve"][-1][k] for r in rr if r["curve"][-1].get(k) is not None] or [float("nan")])
                                     for k in SHARES if any(k in r["curve"][-1] for r in rr)}}
        tier = {"assign": "L", "shape": "S", "specrand": "L control", "shuf": "L control", "sphere": "S control", "line": "S control"}[role]
        L.append(f"| {role} | {tier} | {lr:g} (" + ", ".join(f"{k:g}: {v:.3f}" for k, v in sorted(st.items())) + f") | {lb} | {x['grokB']}/{x['n']} "
                 f"| {x['aucB']:.3f} | {x['final']:.3f} | {x['strict']:.3f} | {x['nonimp']:.3f} | {x['geo']:.3f} | {x['cka_true']:.3f} | "
                 + ", ".join(f"{k} {v:.2f}" for k, v in x["shares"].items()) + " |\n")
    C["prior"] = pick
    C["prior_missing"] = [r for r in ("assign", "specrand", "shuf", "shape", "sphere", "line") if r not in pick]
    # ---- descriptive paired tests (n = 3: exact one-sided p >= 0.05, can never pass alpha 0.025)
    L.append(f"\npaired AUC_B vs best none (lr {nb[0]:g} wd {nb[1]:g}), descriptive at n = 3: ")
    tests = {}
    for role, x in pick.items():
        sd = sorted(set(prior[role][x["lr"]]) & set(none[nb]))
        if not sd:
            continue
        a = [ts.upto(prior[role][x["lr"]][s], B) for s in sd]; b = [ts.upto(none[nb][s], B) for s in sd]
        t = ts.compare_values([u[0] for u in a], [u[0] for u in b], [u[1] for u in a], [u[1] for u in b], chance)
        tests[role] = {"n": t["n"], "auc_wins": t["auc"]["wins"], "auc_med": [t["auc"]["med_a"], t["auc"]["med_b"]], "p": t["auc"]["p"]}
        L.append(f"{role} {t['auc']['med_a']:.3f} vs {t['auc']['med_b']:.3f} (wins {t['auc']['wins']}/{t['n']}, p {t['auc']['p']:.3f}); ")
    C["tests"] = tests
    # ---- tier verdicts (pre-registered, AUDIT.md "Reading" 3)
    C["tier"] = {}
    for tier, (role, _) in TIERS.items():
        v = tier_verdict(tier, pick, missing, wall)
        C["tier"][tier] = v
        L.append(f"\n\n**tier {tier} ({role})**: {v}")
    L.append("\n")
    if a2 is not None:
        C["a2"] = a2_block(a2, none, rows, bk, nb, n_out, floor, chance, missing, pick, C, L)
    return C


def tier_verdict(tier, pick, missing, wall, na=None):
    """Tier verdict of a cell (pre-registered, AUDIT.md "Reading" 4; audit I1: CANDIDATE / floor only with every arm of the
    tier complete). na = A2's reason when the wall is n/a (no trainable none config); A1 never passes it."""
    role, ctls = TIERS[tier]
    x = pick.get(role)
    gap = [f"{a} missing" for a in (role,) + ctls if a not in pick] + \
          [f"{a}: " + ", ".join(pick[a]["incomplete"]) for a in (role,) + ctls if a in pick and pick[a]["incomplete"]]
    if missing:
        return f"PROVISIONAL (none grid incomplete: {', '.join(missing)}); tentative: " + ("A2 n/a" if na else "wall" if wall else "not a wall")
    if na:
        return f"not a wall cell (A2 n/a: {na})"
    if not wall:
        return "not a wall cell: acceleration reading only"
    if gap:
        return "INCOMPLETE (prior arms: " + "; ".join(gap) + ")"
    if x["grokB"] >= math.ceil(2 * N_SEEDS / 3):
        cg = [c for c in ctls if pick[c]["grokB"] >= math.ceil(2 * N_SEEDS / 3)]
        return "CANDIDATE" + (f" (but control {', '.join(cg)} also groks >= 2/3: specificity doubtful)" if cg else "")
    return f"floor for tier {tier} ({role} groks {x['grokB']}/{x['n']})"


def train_level(r, measure="final"):
    """A2's per-run training-fit measure: 'final' = final_train (the registered one); 'late' = the mean train accuracy
    over the evals in the last 25 % of the run (a sensitivity check only)."""
    if measure == "final":
        assert r.get("final_train") is not None, (r["arm"], r["seed"], "final_train missing")
        return float(r["final_train"])
    v = [c["train"] for c in r["curve"] if c["step"] >= 0.75 * r["steps"]]
    assert v, (r["arm"], r["seed"])
    return float(np.mean(v))


def a2_block(a2, none, rows, bk, nb, n_out, floor, chance, missing, pick, C, L):
    """Amendment A2 (AUDIT.md) on a Stage-1 cell, printed beside A1: trainable configs, the reference config, the CE
    component, the wall, the bracket, the tiers, and the paired-test comparators (all configs = the rule; trainable only =
    descriptive)."""
    thr, measure = a2
    tr = {k: med([train_level(r, measure) for r in none[k].values()]) for k in rows}
    ok = trainable(tr, thr)
    ref = reference_config(rows, ok)
    whyd, ce = wall_reasons(rows, ref, n_out, floor, gate=True)
    na = None if ref is not None else "no trainable none config"
    wall = (ref is not None) and not whyd
    bracket = bracket_of(rows, ref, chance)
    tiers = {t: tier_verdict(t, pick, missing, wall, na=na) for t in TIERS}
    nbt = max(ok, key=lambda k: (rows[k]["aucB"], -k[0])) if ok else None
    cf = lambda k: "n/a" if k is None else f"lr {k[0]:g} wd {k[1]:g}"
    cel = lambda c: ("n/a (no reference config)" if c["late_ce_best"] is None else
                     f"late CE {c['late_ce_best']:.2f} vs gate {c['gate']:.2f} -> {'on' if c['on'] else 'n/a'}; drop {c['drop_best']:+.3f}"
                     + ("; FIRES" if c["fires"] else ""))
    a1_wall = C["wall"]
    lab = "" if (thr, measure) == (TRAINABLE_THR, "final") else f" -- SENSITIVITY (threshold {thr:g} on {measure} train), NOT the registered A2"
    L.append(f"\n**A1 | A2** (Amendment A2: trainable = median {'final_train' if measure == 'final' else 'last-25 % mean train'} >= {thr:g}; "
             f"post hoc for Stage 1){lab}: trainable {len(ok)}/{len(rows)} configs; median train per config: "
             + ", ".join(f"{cf(k)} {tr[k]:.3f}" for k in sorted(rows)) + "\n\n"
             "| reading | A1 (pre-registered) | A2 |\n|---|---|---|\n"
             f"| rule-1/2 reference config (CE component, edge test) | {cf(bk)} (median train {tr[bk]:.3f}) | {cf(ref)}"
             + (f" (median train {tr[ref]:.3f})" if ref is not None else "") + " |\n"
             f"| CE component | {cel(C['ce'])} | {cel(ce)} |\n"
             f"| wall | {a1_wall}" + ("" if a1_wall else " (" + "; ".join(C["wall_why"]) + ")") + " | "
             + ("n/a (no trainable none config)" if na else f"{wall}" + ("" if wall else " (" + "; ".join(whyd.values()) + ")")) + " |\n"
             f"| bracket | {C['bracket']} | {bracket} |\n"
             + "".join(f"| tier {t} | {C['tier'][t]} | {tiers[t]} |\n" for t in TIERS)
             + f"| paired-test comparator (highest AUC_B) | {cf(nb)}, all configs | {cf(nb)}, all configs (A2 item 5); "
               f"trainable only {cf(nbt)} (descriptive) |\n")
    return {"thr": thr, "measure": measure, "registered": not lab, "train_med": {f"lr{k[0]:g}-wd{k[1]:g}": v for k, v in tr.items()},
            "trainable": [list(k) for k in ok], "ref": None if ref is None else list(ref), "ce": ce, "wall": bool(wall), "wall_na": na,
            "wall_why": list(whyd.values()), "bracket": bracket, "tier": tiers, "comparator": list(nb),
            "comparator_trainable": None if nbt is None else list(nbt),
            "ce_alone_vetoes": set(whyd) == {"ce"}}


def select(tk, cells):
    """Per task and tier: the CANDIDATE level closest to m = 12 (ties -> the smaller m); else 'no wall at any m' (no wall
    cell at any level) or 'floor' (wall cells exist, the prior fails at all of them). Positive control first."""
    lv = {int(k): c for k, c in cells.items() if k.isdigit()}
    pc = {f"{k}|{cf}": v for k, c in cells.items() if k.startswith("pos") for cf, v in c.get("pos", {}).items()}
    if tk == "a5" and 24 in lv and "lr0.001-wd1" in lv[24].get("none", {}):   # A5's m 24 cell is f 0.4 (its positive control)
        x = lv[24]["none"]["lr0.001-wd1"]
        pc["24|lr0.001-wd1"] = {"status": "INCOMPLETE" if x["n"] < N_SEEDS else "groks" if x["grok"] >= math.ceil(2 * N_SEEDS / 3) else "fails"}
    exp_pos = {"box5": 4, "2i": 1, "d57": 1, "s5": 1, "a5": 1}.get(tk, 0)   # (level, config) cells the design renders
    if not exp_pos:
        out = {"pos": "-"}
    elif any(v["status"] == "groks" for v in pc.values()):
        out = {"pos": "OK (" + ", ".join(k for k, v in pc.items() if v["status"] == "groks") + ")"}
    elif len(pc) < exp_pos or any(v["status"] == "INCOMPLETE" for v in pc.values()):
        out = {"pos": f"INCOMPLETE ({len(pc)}/{exp_pos} cells; " + ", ".join(f"{k} {v['status']}" for k, v in pc.items()) + ")"}
    else:
        out = {"pos": "FAILED: check the pipeline, do not read"}
    for tier in TIERS:
        vt = {m: c.get("tier", {}).get(tier, "no verdict") for m, c in lv.items()}
        cand = [m for m, v in vt.items() if v.startswith("CANDIDATE")]
        open_ = [m for m, v in vt.items() if not v.startswith(("CANDIDATE", "floor", "not a wall"))]
        if set(lv) != {6, 12, 24} or open_:   # audit I1: the choice is made only when every level has a complete verdict
            out[tier] = ("INCOMPLETE: levels present " + str(sorted(lv)) + ("; no complete verdict at m " + str(sorted(open_)) if open_ else "")
                         + (f"; candidates so far at m {sorted(cand)}" if cand else "") + "; nothing is selected for this task yet")
        elif out["pos"].startswith("INCOMPLETE"):
            out[tier] = "INCOMPLETE: positive control incomplete; nothing is selected for this task yet"
        elif cand:
            m = min(cand, key=lambda m: (abs(m - 12), m))
            out[tier] = f"select m {m} for Stage 2" + (f" (candidates at m {sorted(cand)})" if len(cand) > 1 else "") + \
                        (" -- " + vt[m][len("CANDIDATE"):].strip() if vt[m] != "CANDIDATE" else "")
        elif not any(c.get("wall") for c in lv.values()):
            out[tier] = "no wall at any m (none learns or is not at the floor): acceleration reading only"
        else:
            out[tier] = "floor (wall cells at m " + ", ".join(str(m) for m, c in sorted(lv.items()) if c.get("wall")) + "; the prior fails there)"
        if out["pos"].startswith("FAILED"):
            out[tier] = "NOT READ (positive control failed): " + out[tier]
    return out


# ---------------------------------------------------------------------------------------------------- self-test
def selftest():
    import shutil
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="nm_readout_selftest_")) / "out"
    B = 2000
    scan = {"cells": {f"2i|{m}": {"frac": f, "m_level": m} for m, f in ((6, 0.067), (12, 0.1), (24, 0.2))}}
    scan["cells"]["2i|pos"] = {"frac": 0.4, "m_level": "pos"}
    split = lambda seed: {"split_sha": f"sha{seed}", "n_train": 1400, "m_per_item": 11.7, "closure": 1.0, "closure_latin": 1.0, "nullity": None,
                          "latin_share": 0.0, "latin_level": 0.01, "chance_nonimp": 0.01, "elim_strict": 0.0103, "chance_majority": 0.0074,
                          "n_mask": {"implied": 0, "nonimplied": 13000, "strict": 12600}, "n_out": 120}

    def curve(steps, f, strict=None, ce=None, train=None):
        ev = list(range(0, steps, 100)) + ([steps - 1] if (steps - 1) % 100 else [])
        return [{"step": s, "test": f(s), "train": (train(s) if train else 1.0), "test_ce": (ce(s) if ce else 4.0),
                 "test_strict": (strict(s) if strict else f(s)),
                 "test_nonimplied": f(s), "geo": 0.01, "cka_quat4": 0.3, "neg_share": 0.1} for s in ev]

    def write(cell, frac, arm, seed, lr, wd, steps, f, strict=None, ce=None, sha="synthetic", over=None, train=None):
        c = curve(steps, f, strict, ce, train)
        tg = next((x["step"] for x in c if x["test"] >= 0.95), None)
        r = {"task": "grp", "group": "2i", "arm": arm, "seed": seed, "lr": lr, "wd": wd, "lam": 1.0, "frac": frac, "steps": steps,
             "eval_every": 100, "t_grok": tg, "final_test": c[-1]["test"], "final_train": c[-1]["train"], "curve": c, "code_sha": sha,
             "true_target": "quat4", **split(seed), **(over or {})}
        d = root / cell; d.mkdir(parents=True, exist_ok=True)
        (d / f"run_{arm}_seed{seed}.json").write_text(json.dumps(r))

    flat = lambda v: (lambda s: v)
    grok_at = lambda t: (lambda s: 1.0 if s >= t else 0.01)

    def make(m, frac, none_fn=lambda lr, wd, s: flat(0.01), roles=None, grid=None, ce=None, strict=None, keep=lambda role, lr, s: True,
             none_ce=None, none_train=None):
        for (lr, wd) in (grid or NONE_GRID):
            for s in range(3):
                write(f"m{m}-none-{lr}-{wd}", frac, "none", s, lr, wd, 4 * B, none_fn(lr, wd, s), strict,
                      none_ce(lr, wd, s) if none_ce else ce, train=none_train(lr, wd, s) if none_train else None)
        roles = roles or {}
        for role, arm in (("assign", "quat4@always"), ("specrand", "specrand@always"), ("shuf", "shuf@always"),
                          ("shape", "shape:quat4@always"), ("sphere", "shape:sphere@always"), ("line", "shape:line@always")):
            for lr in PRIOR_LRS:
                for s in range(3):
                    if keep(role, lr, s):
                        write(f"m{m}-prior-{lr}", frac, arm, s, lr, 1.0, B, roles.get(role, lambda lr, s: flat(0.01))(lr, s))

    # m6: wall; assign groks 3/3 at lr 3e-3 (best), shape fails -> L CANDIDATE, S floor
    make(6, 0.067, roles={"assign": lambda lr, s: grok_at(800) if lr == 3e-3 else flat(0.3)})
    # m12: wall; assign groks 2/3, shape groks 3/3 but sphere also 2/3 -> L CANDIDATE, S CANDIDATE (control also groks)
    make(12, 0.1, roles={"assign": lambda lr, s: grok_at(900) if s < 2 else flat(0.02), "shape": lambda lr, s: grok_at(1200),
                         "sphere": lambda lr, s: grok_at(1500) if s < 2 else flat(0.01)})
    # m24: none groks in one config -> not a wall cell
    make(24, 0.2, none_fn=lambda lr, wd, s: grok_at(5000) if (lr, wd) == (1e-2, 1.0) else flat(0.01),
         roles={"assign": lambda lr, s: grok_at(500)})
    for s in range(3):
        write("pos", 0.4, "none", s, 1e-3, 1.0, 4 * B, grok_at(3000))
    txt, J = read(root, scan)
    c6, c12, c24 = J["cells"]["2i|6"], J["cells"]["2i|12"], J["cells"]["2i|24"]
    assert c6["wall"] and c6["tier"]["L"] == "CANDIDATE" and c6["tier"]["S"].startswith("floor"), (c6["wall_why"], c6["tier"])
    assert c6["prior"]["assign"]["lr"] == 3e-3 and c6["prior"]["assign"]["lr_bracket"] == "closed"
    assert c6["bracket"].startswith("OPEN (flat"), c6["bracket"]
    assert c12["tier"]["L"] == "CANDIDATE" and "sphere also groks" in c12["tier"]["S"], c12["tier"]
    assert not c24["wall"] and c24["tier"]["L"].startswith("not a wall"), c24["tier"]
    assert J["tasks"]["2i"]["pos"].startswith("OK"), J["tasks"]["2i"]["pos"]
    assert c6["prior"]["shape"]["lr_bracket"].startswith("flat"), c6["prior"]["shape"]["lr_bracket"]   # M6: no optimum, no direction
    assert J["tasks"]["2i"]["L"].startswith("select m 12") and "candidates at m [6, 12]" in J["tasks"]["2i"]["L"], J["tasks"]["2i"]
    assert J["tasks"]["2i"]["S"].startswith("select m 12") and "specificity doubtful" in J["tasks"]["2i"]["S"], J["tasks"]["2i"]
    print("selftest A: L candidate at m 6 and 12 -> m 12 (closest to 12); S candidate with a grokking control flagged; m 24 not a "
          "wall (one none config groks); flat none grid -> bracket OPEN until the extension; best prior lr 3e-3 = closed bracket")
    # B: every way out of the wall, one cell each (m levels reused under fresh roots)
    cases = {
        "seed above cap": dict(none_fn=lambda lr, wd, s: flat(0.25) if (lr, wd, s) == (3e-3, 0.1, 2) else flat(0.01)),
        "rising": dict(none_fn=lambda lr, wd, s: (lambda st: 0.01 + 0.1 * max(0, st - 6000) / 2000)),
        "CE drop": dict(ce=lambda st: 4.0 - 0.1 * st / 8000),   # last-25 % drop ~0.019 nats (>= 0.01)
        "strict above floor": dict(strict=lambda st: 0.08),
    }
    for name, kw in cases.items():
        shutil.rmtree(root, ignore_errors=True)
        make(6, 0.067, roles={"assign": lambda lr, s: grok_at(800)}, **kw)
        _, J = read(root, scan)
        c = J["cells"]["2i|6"]
        assert not c["wall"] and c["tier"]["L"].startswith("not a wall"), (name, c["wall_why"])
        if name == "CE drop":   # late CE 3.9 <= log(120) + 1 = 5.79: the CE component is on and alone vetoes (printed)
            assert c["ce"]["on"] and c["ce"]["alone_vetoes"] and "ALONE vetoes" in _, c["ce"]
        print(f"selftest B '{name}': not a wall cell ({c['wall_why'][0][:70]}...)")
    # C: incomplete grid -> PROVISIONAL; pos control failure -> task not read; mixed code_sha -> no verdict
    shutil.rmtree(root, ignore_errors=True)
    make(6, 0.067, roles={"assign": lambda lr, s: grok_at(800)}, grid=NONE_GRID[:-1])
    for s in range(3):
        write("pos", 0.4, "none", s, 1e-3, 1.0, 4 * B, flat(0.02))
    _, J = read(root, scan)
    v = J["cells"]["2i|6"]["tier"]["L"]
    assert v.startswith("PROVISIONAL (none grid incomplete") and "CANDIDATE" not in v and J["cells"]["2i|6"]["provisional"], v
    assert J["tasks"]["2i"]["pos"].startswith("FAILED") and J["tasks"]["2i"]["L"].startswith("NOT READ")
    write("pos", 0.4, "none", 0, 1e-3, 1.0, 4 * B, flat(0.02), sha="other")
    txt, J = read(root, scan)
    assert "NO VERDICT" in txt
    print("selftest C: missing none config -> PROVISIONAL (never CANDIDATE); failed positive control -> task NOT READ; mixed code_sha -> NO VERDICT")
    shutil.rmtree(root, ignore_errors=True)   # D: levels without a verdict (no none runs at m 12, m 24 absent) -> INCOMPLETE
    make(6, 0.067, none_fn=lambda lr, wd, s: grok_at(3000))
    for s in range(3):
        write("m12-prior-0.003", 0.1, "quat4@always", s, 3e-3, 1.0, B, grok_at(500))
    _, J = read(root, scan)
    assert all(J["tasks"]["2i"][t].startswith("INCOMPLETE") for t in "LS"), J["tasks"]["2i"]
    print("selftest D: a level without none runs / a missing level -> INCOMPLETE (never 'no wall')")
    # E: audit I1, the four scenarios of the audit's chk_readout.py (all three levels walls; none grid complete)
    levels = ((6, 0.067), (12, 0.1), (24, 0.2))
    grk = lambda lr, s: grok_at(800)

    def pos3(seeds=(0, 1, 2)):
        for s in seeds:
            write("pos", 0.4, "none", s, 1e-3, 1.0, 4 * B, grok_at(3000))
    E = {"assign one seed at one lr": dict(keep=lambda role, lr, s: role != "assign" or (lr == 1e-2 and s == 0), roles={"assign": grk}),
         "assign missing": dict(keep=lambda role, lr, s: role != "assign"),
         "control specrand 1/3 seeds": dict(keep=lambda role, lr, s: role != "specrand" or s == 0, roles={"assign": grk})}
    for name, kw in E.items():
        shutil.rmtree(root, ignore_errors=True)
        for m, f in levels:
            make(m, f, **kw)
        pos3()
        _, J = read(root, scan)
        vs = [J["cells"][f"2i|{m}"]["tier"]["L"] for m, _ in levels]
        assert all(v.startswith("INCOMPLETE (prior arms") for v in vs), (name, vs)
        assert J["tasks"]["2i"]["L"].startswith("INCOMPLETE"), (name, J["tasks"]["2i"])
        assert not J["tasks"]["2i"]["L"].startswith(("select", "floor")), J["tasks"]["2i"]
        print(f"selftest E '{name}': every level tier L '{vs[0][:60]}...', selection '{J['tasks']['2i']['L'][:50]}...'")
    shutil.rmtree(root, ignore_errors=True)
    for m, f in levels:
        make(m, f, roles={"assign": grk})
    pos3(seeds=(0,))
    _, J = read(root, scan)
    assert J["tasks"]["2i"]["pos"].startswith("INCOMPLETE") and all(J["tasks"]["2i"][t].startswith("INCOMPLETE") for t in "LS"), J["tasks"]["2i"]
    print(f"selftest E 'positive control 1/3 seeds': pos '{J['tasks']['2i']['pos'][:50]}', selection INCOMPLETE")
    # F: Amendment A1 (audit B1): a memoriser with late CE ~22 nats and a falling CE (drop ~1.5 nats) is a wall; the
    # ungated rule would have vetoed it on CE alone (printed); case B 'CE drop' above covers the gate-on side
    shutil.rmtree(root, ignore_errors=True)
    make(6, 0.067, ce=lambda st: 30.0 - 8.0 * st / 8000, roles={"assign": grk})
    txt, J = read(root, scan)
    c = J["cells"]["2i|6"]
    assert c["wall"] and not c["ce"]["on"] and c["ce"]["ungated_alone_would_veto"] and not c["wall_ungated"], c["ce"]
    assert "the ungated pre-A1 rule would have vetoed it on CE alone" in txt and c["tier"]["L"] == "CANDIDATE", c["tier"]
    print(f"selftest F: late CE {c['ce']['late_ce_best']:.1f} > gate {c['ce']['gate']:.2f}: CE component n/a, wall True "
          f"(the ungated rule: not a wall, CE alone; printed)")
    # G: box5 positive controls: 2 levels (f 0.4, 0.6) x 2 configs (lr 1e-3 wd 1, lr 1e-2 wd 0.1)
    sc2 = {"cells": {"box5|pos": {"frac": 0.4, "m_level": "pos"}, "box5|pos6": {"frac": 0.6, "m_level": "pos6"}}}
    bo = {"task": "box", "box": [[-2, 2]] * 3, "true_target": "expo", "n_out": 125}
    for name, gro, seeds_lr2, want in (("one config learns", (0.6, 1e-2), (0, 1, 2), "OK"), ("none learns", None, (0, 1, 2), "FAILED"),
                                       ("a config incomplete", None, (0,), "INCOMPLETE")):
        shutil.rmtree(root, ignore_errors=True)
        for lvl, f in (("pos", 0.4), ("pos6", 0.6)):
            for lr, wd in ((1e-3, 1.0), (1e-2, 0.1)):
                for s in (seeds_lr2 if lr == 1e-2 else (0, 1, 2)):
                    write(f"{lvl}-{lr}", f, "none", s, lr, wd, 4 * B, grok_at(3000) if gro == (f, lr) else flat(0.02), over=bo)
        _, J = read(root, sc2)
        assert J["tasks"]["box5"]["pos"].startswith(want), (name, J["tasks"]["box5"]["pos"])
        print(f"selftest G box5 positive control '{name}': {J['tasks']['box5']['pos'][:60]}")
    # H: Amendment A2 (2026-10-25; results audit F1). Crashed configs: train 0.02, held-out at chance 0.012 (above the
    # memorisers' 0.01, so they win A1's AUC ranking), late CE ~ log n_out; memorisers: train 1.0, late CE 30 (gate n/a).
    import re
    import subprocess
    crashed = lambda lr, wd: lr == 3e-2 and wd >= 1.0
    falling = lambda c0: (lambda st: c0 - 0.1 * st / 8000)   # last-25 % drop ~0.019 nats (>= 0.01), late CE ~ c0 - 0.1
    H = {
        # A1 picks a crashed config whose CE drift alone vetoes; A2's reference is a memoriser (CE n/a) -> a wall
        "crashed config as best": dict(none_fn=lambda lr, wd, s: flat(0.012 if crashed(lr, wd) else 0.01),
                                       none_ce=lambda lr, wd, s: falling(4.9) if crashed(lr, wd) else flat(30.0),
                                       none_train=lambda lr, wd, s: flat(0.02 if crashed(lr, wd) else 1.0), roles={"assign": grk}),
        # nothing fits its training set: A1 reads a wall (CANDIDATE); A2 is n/a, never a wall
        "all configs crashed": dict(none_fn=lambda lr, wd, s: flat(0.012), none_ce=lambda lr, wd, s: flat(4.8),
                                    none_train=lambda lr, wd, s: flat(0.02), roles={"assign": grk}),
        # mixed: the crashed configs win A1's ranking (flat CE, no veto); A2's reference (lr 1e-2 wd 1, the best trainable
        # AUC) has a falling CE inside the gate -> A2 vetoes on CE (A2 does not just switch the CE component off)
        "mixed, trainable reference still rising": dict(
            none_fn=lambda lr, wd, s: flat(0.012 if crashed(lr, wd) else 0.011 if (lr, wd) == (1e-2, 1.0) else 0.01),
            none_ce=lambda lr, wd, s: flat(4.8) if crashed(lr, wd) else falling(4.0) if (lr, wd) == (1e-2, 1.0) else flat(30.0),
            none_train=lambda lr, wd, s: flat(0.02 if crashed(lr, wd) else 1.0), roles={"assign": grk}),
        # mixed: a crashed config with one seed above the cap still vetoes under A2 (crashed configs count in the other parts)
        "mixed, crashed config above the cap": dict(
            none_fn=lambda lr, wd, s: flat(0.25 if (lr, wd, s) == (3e-2, 1.0, 0) else 0.012 if crashed(lr, wd) else 0.01),
            none_ce=lambda lr, wd, s: flat(4.8) if crashed(lr, wd) else flat(30.0),
            none_train=lambda lr, wd, s: flat(0.02 if crashed(lr, wd) else 1.0), roles={"assign": grk}),
    }
    want = {"crashed config as best": (False, True), "all configs crashed": (True, None),
            "mixed, trainable reference still rising": (True, False), "mixed, crashed config above the cap": (False, False)}
    for name, kw in H.items():
        shutil.rmtree(root, ignore_errors=True)
        make(6, 0.067, **kw)
        t1, J1 = read(root, scan)
        t2, J2 = read(root, scan, a2=True)
        c1, c2 = J1["cells"]["2i|6"], J2["cells"]["2i|6"]
        x = c2["a2"]
        w1, w2 = want[name]
        assert c1["wall"] == w1 and (x["wall_na"] is not None if w2 is None else x["wall"] == w2 and x["wall_na"] is None), (name, c1["wall_why"], x)
        # the A1 part of the --a2 output is the A1 output (text and JSON)
        assert re.sub(r"\n## Selection under Amendment A2.*\Z", "", re.sub(r"\n\*\*A1 \| A2\*\*.*?(?=\n## )", "", t2, flags=re.S), flags=re.S) == t1, name
        assert {k: v for k, v in c2.items() if k != "a2"} == c1, name
        assert "**A1 | A2**" in t2 and "| CE component |" in t2 and "Selection under Amendment A2" in t2, name
        if name == "crashed config as best":
            assert tuple(c1["none_best"]) == (3e-2, 1.0) and c1["ce"]["alone_vetoes"], c1["ce"]
            assert x["ref"] == [1e-3, 0.1] and not x["ce"]["on"] and len(x["trainable"]) == 10, x
            assert c1["tier"]["L"].startswith("not a wall") and x["tier"]["L"] == "CANDIDATE" and x["tier"]["S"].startswith("floor"), x["tier"]
            assert x["comparator"] == c1["none_best_aucB"] == [3e-2, 1.0] and x["comparator_trainable"] == [1e-3, 0.1], x   # A2 item 5
        elif name == "all configs crashed":
            assert c1["tier"]["L"] == "CANDIDATE" and x["ref"] is None and x["trainable"] == [] and x["ce"]["late_ce_best"] is None, (c1["tier"], x)
            assert all(v.startswith("not a wall cell (A2 n/a: no trainable none config)") for v in x["tier"].values()), x["tier"]
            assert x["bracket"] == c1["bracket"] and x["bracket"].startswith("OPEN (flat"), x["bracket"]   # the flat test is unchanged
            assert "n/a (no trainable none config)" in t2
        elif name == "mixed, trainable reference still rising":
            assert tuple(c1["none_best"]) in ((3e-2, 1.0), (3e-2, 3.0)) and not c1["ce"]["fires"], c1["ce"]
            assert x["ref"] == [1e-2, 1.0] and x["ce"]["on"] and x["ce"]["fires"] and x["ce_alone_vetoes"], x
            assert x["tier"]["L"].startswith("not a wall") and c1["tier"]["L"] == "CANDIDATE", (c1["tier"], x["tier"])
        else:
            assert any("above 0.20" in w for w in c1["wall_why"]) and any("above 0.20" in w for w in x["wall_why"]), (c1["wall_why"], x["wall_why"])
            assert all(v.startswith("not a wall") for v in x["tier"].values()), x["tier"]
        print(f"selftest H (A2) '{name}': A1 wall {c1['wall']} (best none {c1['none_best']}), A2 wall "
              f"{'n/a' if x['wall_na'] else x['wall']} (reference {x['ref']}, trainable {len(x['trainable'])}/12); A1 text / JSON unchanged")
    # H5: the threshold is consumed (function and CLI): a config at median train 0.7 with the best AUC among the rest and a
    # falling CE is untrainable at 0.99 (A2 wall) and the reference at 0.5 (A2 vetoes on CE); the CLI writes a SENSITIVITY file
    shutil.rmtree(root, ignore_errors=True)
    make(6, 0.067, none_fn=lambda lr, wd, s: flat(0.011 if (lr, wd) == (1e-2, 1.0) else 0.01),
         none_ce=lambda lr, wd, s: falling(4.0) if (lr, wd) == (1e-2, 1.0) else flat(30.0),
         none_train=lambda lr, wd, s: flat(0.7 if (lr, wd) == (1e-2, 1.0) else 1.0), roles={"assign": grk})
    ta, Ja = read(root, scan, a2=True, thr=TRAINABLE_THR)
    tb, Jb = read(root, scan, a2=True, thr=0.5)
    xa, xb = Ja["cells"]["2i|6"]["a2"], Jb["cells"]["2i|6"]["a2"]
    assert xa["wall"] and xa["ref"] != [1e-2, 1.0] and xa["registered"] and "SENSITIVITY" not in ta, xa
    assert not xb["wall"] and xb["ref"] == [1e-2, 1.0] and not xb["registered"] and "SENSITIVITY (threshold 0.5" in tb, xb
    sc_dir = root.parent / "cli"; (sc_dir / "preflight").mkdir(parents=True, exist_ok=True)
    (sc_dir / "preflight" / "seed_scan.json").write_text(json.dumps(scan))
    shutil.copy(Path(__file__), sc_dir / "readout.py"); (sc_dir / "code").symlink_to(TOOLS.parent)
    o = subprocess.run([sys.executable, str(sc_dir / "readout.py"), "--a2", "--trainable_thr", "0.5", "--root", str(root)], capture_output=True, text=True)
    assert o.returncode == 0, o.stderr[-800:]
    assert "A2 trainability threshold used: 0.5 on final_train (SENSITIVITY)" in o.stdout and (root.parent / "readout_a2_sens_thr0.5.md").exists()
    assert "SENSITIVITY (threshold 0.5" in (root.parent / "readout_a2_sens_thr0.5.md").read_text()
    o = subprocess.run([sys.executable, str(sc_dir / "readout.py"), "--trainable_thr", "0.5", "--root", str(root)], capture_output=True, text=True)
    assert o.returncode != 0 and "--trainable_thr only with --a2" in o.stderr, o.stderr[-300:]
    print("selftest H5 (A2 threshold consumed): 0.99 -> the 0.7-train config is not trainable, wall; 0.5 -> it is the reference, "
          "CE vetoes; the CLI --trainable_thr 0.5 writes a SENSITIVITY file and prints the value; without --a2 it is refused")
    shutil.rmtree(root.parent, ignore_errors=True)
    print("READOUT SELFTEST PASSED")


if __name__ == "__main__":
    if "--selftest" in ARGS:
        selftest()
    else:
        root = Path(ARGS[ARGS.index("--root") + 1]).resolve() if "--root" in ARGS else HERE / "out"
        scan = json.loads((HERE / "preflight" / "seed_scan.json").read_text())
        A2 = "--a2" in ARGS
        txt, J = read(root, scan, a2=True, thr=A2_THR) if A2 else read(root, scan)
        dest = HERE if "--root" not in ARGS else root.parent
        # the default call = the pre-registered A1 reading (readout.md / .json); --a2 never overwrites them
        stem = "readout" if not A2 else "readout_a2" if A2_THR == TRAINABLE_THR else f"readout_a2_sens_thr{A2_THR:g}"
        (dest / f"{stem}.md").write_text(txt)
        clean = lambda x: ({str(k): clean(v) for k, v in x.items()} if isinstance(x, dict) else [clean(v) for v in x]
                           if isinstance(x, (list, tuple)) else "inf" if isinstance(x, float) and math.isinf(x) else x)
        (dest / f"{stem}.json").write_text(json.dumps(clean(J), indent=1))
        print(txt)
        if A2:
            print(f"A2 trainability threshold used: {A2_THR:g} on final_train ({'registered' if A2_THR == TRAINABLE_THR else 'SENSITIVITY'}); "
                  f"wrote {dest / stem}.md / .json")
