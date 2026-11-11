"""runs/newmani Amendment A3 read-out (POST HOC, registered 2026-10-25 11:37 EDT before any low-wd run; AUDIT.md "Amendment
A3"): does the box5 m6 Stage-2 wall (PASS, G1-G4) survive none's low-wd direction? From the run JSONs; no training.

    python readout_a3.py                       # out_stage2/ + out_a3/ -> readout_a3.md / .json next to this file
    python readout_a3.py --root_s2 D --root_a3 E   # other roots (tests); outputs go to E/..
    python readout_a3.py --selftest

Reuses readout_stage2.read_stage2 (the Stage-2 verdict and the assign pick; it must be PASS) and readout.py's shared
definitions (none_row, train_level, trainable, reference_config, wall_reasons, bracket_of). Stage-2 readers are unchanged.
  A3 main  none lr {3e-3, 1e-2, 3e-2} x wd {0.03, 0.01, 0}, the 5 Stage-2 seeds, 4 B (out_a3/box5-m6-f0.067-none-lr*-wd*)
  A3-long  none lr 1e-2 x wd {0.03, 0.01}, the same seeds, 16 B (cells suffixed -16B)
Union G1 over the 19 Stage-2 + 9 A3 configs (G1's four conditions per config, the A2 reference config's CE with A1's gate,
the bracket over the union grid), under both training-fit measures (A2 final_train; A2.1 last-25 % mean); label per
measure: WALL / OPEN_WD_BOTTOM (G1 fails, the reference is at wd 0 and rising, no new config groks > 1/5 or has median
final > 0.20) / FAIL; the two labels differ -> UNDETERMINED. A3-long only qualifies a WALL (never upgrades)."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ARGS = sys.argv[1:]
assert set(a for a in ARGS if a.startswith("--")) <= {"--root_s2", "--root_a3", "--selftest"}, f"unknown option(s) {ARGS}"
_argv, sys.argv = sys.argv, [sys.argv[0]]   # the imported readers parse their own argv at import: give them none
_sp = importlib.util.spec_from_file_location("readout_stage2", HERE / "readout_stage2.py")
S2 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(S2)
sys.argv = _argv
R1, ts, med, AUC_MIN = S2.R1, S2.ts, S2.med, S2.AUC_MIN
SEEDS, PIN = S2.SEEDS, S2.PIN
A3_GRID = [(lr, wd) for lr in (3e-3, 1e-2, 3e-2) for wd in (0.03, 0.01, 0.0)]
A3L_GRID = [(1e-2, 0.03), (1e-2, 0.01)]
L_MULT = 16                                  # A3-long budget = 16 B
NEED_BAD = math.floor(0.2 * len(SEEDS))      # a config "groks" in the wall sense when > 1 of 5
assert len(A3_GRID) == 9 and not set(A3_GRID) & set(S2.NONE_ALL) and NEED_BAD == 1
WALL_TXT = ("低 wd 方向也是墙：Tier L 挽救在 wd ∈ {0, 0.01, 0.03, 0.1, 1, 3, 10} × lr {1e-3..1e-1} 的范围内成立 "
            "(post hoc, within the 4 B budget)")
OPEN_TXT = "区间仍开着（wd 下沿）：不叫挽救"
FAIL_TXT = "G1 在低 wd 方向不成立 → 不是挽救（墙依赖于没测的优化器方向）"


def local_bracket(rows, ref, chance):
    """Amendment A3.1: rule 2 on an irregular (L-shaped) grid. Flat -> as readout.bracket_of; else the reference is on an
    edge when an axis neighbour (the previous / next lr of the grid's lr list at its wd, the previous / next wd at its lr)
    did not run, or it is a global extreme."""
    flat = all(x["grok"] == 0 and x["auc4"] <= chance + AUC_MIN for x in rows.values())
    if flat or ref is None:
        return R1.bracket_of(rows, ref, chance)
    lrs, wds = sorted({k[0] for k in rows}), sorted({k[1] for k in rows})
    i, j = lrs.index(ref[0]), wds.index(ref[1])
    edges = []
    for name, ax, idx, step in (("lr bottom", lrs, i, -1), ("lr top", lrs, i, +1), ("wd bottom", wds, j, -1), ("wd top", wds, j, +1)):
        q = idx + step
        if q < 0 or q >= len(ax):
            edges.append(name)
        else:
            nb = (ax[q], ref[1]) if name.startswith("lr") else (ref[0], ax[q])
            if nb not in rows:
                edges.append(f"{name} (neighbour lr {nb[0]:g} wd {nb[1]:g} not run)")
    return "closed" if not edges else "OPEN at " + ", ".join(edges)


def load_a3(root: Path, cell: dict, B: int, pin: str | None, device: str):
    exp = {x["seed"]: x for x in cell["seeds"]}
    n4, nl, bad = {}, {}, []
    for p in sorted(root.glob("*/run_*.json")):
        r = json.loads(p.read_text())
        k = (r["lr"], r["wd"])
        why = [w for w, c in (("arm", r["arm"] != "none"), ("not box5", not (r.get("task") == "box" and r.get("box") == [[-2, 2]] * 3)),
                              ("frac", abs(r.get("frac", -1) - cell["frac"]) > 1e-12), ("split", r.get("split") != "class"),
                              ("seed", r["seed"] not in exp), ("split_sha", r["seed"] in exp and r.get("split_sha") != exp[r["seed"]]["split_sha"]),
                              ("code_sha", pin is not None and r.get("code_sha") != pin), ("device", r.get("device") != device),
                              ("config / steps", not ((k in A3_GRID and r["steps"] == 4 * B) or (k in A3L_GRID and r["steps"] == L_MULT * B)))) if c]
        if why:
            bad.append(f"{p.parent.name}/{p.name}: {', '.join(why)}"); continue
        d = (n4 if r["steps"] == 4 * B else nl).setdefault(k, {})
        assert r["seed"] not in d, ("duplicate", k, r["seed"], r["steps"])
        d[r["seed"]] = r
    miss4 = [f"lr {k[0]:g} wd {k[1]:g}: " + (f"seeds {sorted(n4[k])}" if k in n4 else "absent") for k in A3_GRID if set(n4.get(k, {})) != set(SEEDS)]
    missl = [f"lr {k[0]:g} wd {k[1]:g} (16 B): " + (f"seeds {sorted(nl[k])}" if k in nl else "absent") for k in A3L_GRID if set(nl.get(k, {})) != set(SEEDS)]
    return n4, nl, bad, miss4, missl


def read_a3(root_s2: Path, root_a3: Path, cell: dict, B_expected: int | None = S2.B_REG, pin: str | None = PIN) -> tuple[str, dict]:
    L = [f"# newmani Amendment A3 read-out (POST HOC): box5 m6 none, low wd ({root_s2} + {root_a3})\n\n"]
    J = {"root_s2": str(root_s2), "root_a3": str(root_a3), "verdict": None}
    t2, J2 = S2.read_stage2(root_s2, cell, B_expected, pin)
    J["stage2_verdict"] = J2["verdict"]
    if J2["verdict"] in ("NO VERDICT", "INCOMPLETE", "NO VERDICT (no runs)"):
        J["verdict"] = f"INCOMPLETE (Stage 2: {J2['verdict']})"
        return "".join(L) + f"**{J['verdict']}**\n", J
    if not J2["verdict"].startswith("PASS"):
        J["verdict"] = f"n/a (Stage 2 is not PASS: {J2['verdict']})"
        return "".join(L) + f"**{J['verdict']}**\n", J
    B, chance, floor = J2["B"], J2["chance"], J2["floor_line"]
    dev = J2["device"][0]
    s2none = {}
    for p in sorted(root_s2.glob("*/run_none_seed*.json")):
        r = json.loads(p.read_text())
        s2none.setdefault((r["lr"], r["wd"]), {})[r["seed"]] = r
    assert set(s2none) == set(S2.NONE_ALL) and all(set(d) == set(SEEDS) for d in s2none.values()), "Stage-2 none grid (read_stage2 passed it)"
    n4, nl, bad, miss4, missl = load_a3(root_a3, cell, B, pin, dev)
    J.update({"a3_configs_read": sorted([list(k) for k in n4]), "a3long_configs_read": sorted([list(k) for k in nl]), "foreign": bad,
              "missing": miss4, "missing_long": missl})
    L.append(f"Stage 2: {J2['verdict']}\n\nA3 configs read (4 B): " + (", ".join(f"lr {k[0]:g} wd {k[1]:g} ({len(n4[k])} seeds)" for k in sorted(n4)) or "none")
             + "\n\nA3-long configs read (16 B): " + (", ".join(f"lr {k[0]:g} wd {k[1]:g} ({len(nl[k])} seeds)" for k in sorted(nl)) or "none") + "\n\n")
    if bad:
        J["verdict"] = "NO VERDICT"
        return "".join(L) + f"**NO VERDICT**: foreign A3 runs {bad[:10]}\n", J
    if miss4:
        J["verdict"] = "INCOMPLETE"
        return "".join(L) + "**INCOMPLETE: nothing is read**: " + "; ".join(miss4) + "\n", J
    B4, n_out = 4 * B, 125
    union = {**s2none, **n4}
    assert len(union) == 28
    rows, tr, trl = {}, {}, {}
    for k in sorted(union):
        rr = [union[k][s] for s in SEEDS]
        rows[k] = R1.none_row(rr, B, B4)
        tr[k] = med([R1.train_level(r) for r in rr])
        trl[k] = med([R1.train_level(r, "late") for r in rr])
    cf = lambda k: "n/a" if k is None else f"lr {k[0]:g} wd {k[1]:g}"
    new_learns = [k for k in A3_GRID if rows[k]["grok"] > NEED_BAD or rows[k]["final_med"] > R1.CAP]
    g = {}
    for m, tt in (("final", tr), ("late", trl)):
        ok = R1.trainable(tt, R1.TRAINABLE_THR)
        ref = R1.reference_config(rows, ok)
        whyd, ce = R1.wall_reasons(rows, ref, n_out, floor, gate=True, cap_key="final_med")
        br = local_bracket(rows, ref, chance)   # A3.1
        closed = br in ("closed", "flat over the extended grid")
        g1 = ref is not None and not whyd and closed
        rising = ref is not None and (rows[ref]["gain"] >= R1.RISE_ACC or ce["fires"])
        lab = "WALL" if g1 else "OPEN_WD_BOTTOM" if (ref is not None and ref[1] == 0.0 and rising and not new_learns) else "FAIL"
        g[m] = {"trainable": [list(k) for k in ok], "ref": None if ref is None else list(ref), "ce": ce, "why": list(whyd.values()),
                "bracket": br, "g1": g1, "label": lab, "fails_on": sorted(whyd) + ([] if closed else ["bracket"])}
    # ---- tables
    L.append("| none config (new = A3) | n | groks in 4B | median final train / last-25 % mean | final test median / max | final strict median "
             "| AUC 4B | AUC_B | last 25%: gain / CE drop (late CE) | wall conditions (groks <= 1, final <= 0.20, gain < 0.02, strict <= floor) "
             "| A3.1 descriptive: reference (A2 / A2.1), CE fires, median max held-out at any eval, non-finite test_ce evals |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|\n")
    per = {}
    for k in sorted(union, key=lambda k: (k not in n4, k)):
        x = rows[k]
        c = {"grok": x["grok"] <= NEED_BAD, "final": x["final_med"] <= R1.CAP, "gain": x["gain"] < R1.RISE_ACC, "strict": x["strict"] <= floor}
        isref = [g["final"]["ref"] == list(k), g["late"]["ref"] == list(k)]
        fires = [isref[0] and g["final"]["ce"]["fires"], isref[1] and g["late"]["ce"]["fires"]]
        mx = med([max(c_["test"] for c_ in union[k][s]["curve"]) for s in SEEDS])
        nonfin = sum(not math.isfinite(c_.get("test_ce", math.nan)) for s in SEEDS for c_ in union[k][s]["curve"])
        if k in n4:
            per[f"lr{k[0]:g}-wd{k[1]:g}"] = {**c, "all": all(c.values()), "is_ref": isref, "ce_fires": fires, "max_any_eval_med": mx,
                                             "nonfinite_ce_evals": nonfin}
        L.append(f"| {cf(k)}{' (new)' if k in n4 else ''} | {x['n']} | {x['grok']}/{x['n']} | {tr[k]:.3f} / {trl[k]:.3f} | {x['final_med']:.3f} / "
                 f"{x['final_max']:.3f} | {x['strict']:.3f} | {x['auc4']:.3f} | {x['aucB']:.3f} | {x['gain']:+.3f} / {x['ce_drop']:+.3f} "
                 f"({x['ce_late']:.2f}) | " + ("all met" if all(c.values()) else "FAILS: " + ", ".join(n for n, v in c.items() if not v)) + " | "
                 f"{'/'.join('yes' if v else 'no' for v in isref)}, {'/'.join('yes' if v else 'no' for v in fires)}, {mx:.3f}"
                 + (" **> 0.20**" if mx > R1.CAP else "") + f", {nonfin} |\n")
    J["per_new_config"] = per
    cel = lambda c: ("n/a" if c["late_ce_best"] is None else f"late CE {c['late_ce_best']:.2f} vs gate {c['gate']:.2f} -> "
                     f"{'on' if c['on'] else 'n/a'}; drop {c['drop_best']:+.3f}" + (" FIRES" if c["fires"] else ""))
    L.append(f"\nchance {chance:.4f}; floor line {floor:.4f}\n\n| union G1 (28 configs) | A2: trainable = median final_train >= 0.99 | "
             "A2.1: trainable = median last-25 % mean >= 0.99 |\n|---|---|---|\n"
             f"| trainable configs | {len(g['final']['trainable'])} | {len(g['late']['trainable'])} |\n"
             f"| reference config | {cf(g['final']['ref'] and tuple(g['final']['ref']))} | {cf(g['late']['ref'] and tuple(g['late']['ref']))} |\n"
             f"| CE component | {cel(g['final']['ce'])} | {cel(g['late']['ce'])} |\n"
             f"| wall reasons | {'; '.join(g['final']['why']) or 'none'} | {'; '.join(g['late']['why']) or 'none'} |\n"
             f"| bracket (union grid) | {g['final']['bracket']} | {g['late']['bracket']} |\n"
             f"| **union G1 / label** | **{'PASS' if g['final']['g1'] else 'FAIL'} / {g['final']['label']}** | "
             f"**{'PASS' if g['late']['g1'] else 'FAIL'} / {g['late']['label']}** |\n\n")
    # ---- G2, descriptive: assign (Stage 2's pick) vs each new config and vs the union comparator
    a = J2["prior"]["assign"]
    thr = floor - AUC_MIN

    def test(k, key):
        nt = {s: ts.upto(union[k][s], B)[0] for s in SEEDS}
        na = {s: ts.upto(union[k][s], B, key)[1] for s in SEEDS}
        aa = a["auc"] if key == "test" else a["auc_strict"]
        t = ts.compare_values([a["tg"][s] for s in SEEDS], [nt[s] for s in SEEDS], [aa[s] for s in SEEDS], [na[s] for s in SEEDS], thr)
        return {"pass": t["auc"]["pass"], "med_a": t["auc"]["med_a"], "med_b": t["auc"]["med_b"], "wins": t["auc"]["wins"], "p": t["auc"]["p"]}

    aucBs = {k: med([ts.upto(union[k][s], B, "test_strict")[1] for s in SEEDS]) for k in union}
    comp = {"test": max(union, key=lambda k: (rows[k]["aucB"], -k[0])), "strict": max(union, key=lambda k: (aucBs[k], -k[0]))}
    G2 = {}
    L.append(f"G2 (descriptive): assign lr {a['lr']:g} (Stage 2's pick) vs none, paired AUC_B on the first B steps\n\n"
             "| none | all held-out: assign / none, wins, p, pass | strict: assign / none, wins, p, pass |\n|---|---|---|\n")
    for name, ks in [(f"union comparator ({cf(comp['test'])} / {cf(comp['strict'])})", (comp["test"], comp["strict"]))] + \
                    [(cf(k) + " (new)", (k, k)) for k in A3_GRID]:
        tt, tsr = test(ks[0], "test"), test(ks[1], "test_strict")
        G2[name] = {"test": tt, "strict": tsr}
        L.append(f"| {name} | {tt['med_a']:.3f} / {tt['med_b']:.3f}, {tt['wins']}/5, {tt['p']:.4f}, {tt['pass']} | "
                 f"{tsr['med_a']:.3f} / {tsr['med_b']:.3f}, {tsr['wins']}/5, {tsr['p']:.4f}, {tsr['pass']} |\n")
    J["g2_descriptive"] = G2
    # ---- A3-long (16 B)
    long = {}
    L.append("\nA3-long (16 B = %d steps):\n\n| none config | seeds | groks within 16 B | t_grok median | final test median | last 25 %% of 16 B: "
             "gain | median final train |\n|---|---|---|---|---|---|---|\n" % (L_MULT * B))
    for k in A3L_GRID:
        d = nl.get(k, {})
        if not d:
            L.append(f"| {cf(k)} | none | - | - | - | - | - |\n"); continue
        rr = [d[s] for s in sorted(d)]
        gk = sum(r["t_grok"] is not None for r in rr)
        tgm = S2.cmed([ts.cens(r) for r in rr])
        tg2 = sorted(ts.cens(r) for r in rr)[NEED_BAD] if len(rr) > NEED_BAD else math.inf   # A3.1: when >= 2 seeds have grokked
        long[f"lr{k[0]:g}-wd{k[1]:g}"] = x = {"n": len(rr), "grok": gk, "tg_med": tgm, "tg_2nd": tg2, "final": med([r["final_test"] for r in rr]),
                                             "gain": med([ts.late_change(r, L_MULT * B)[0] for r in rr]),
                                             "train": med([R1.train_level(r) for r in rr]), "learns": gk > NEED_BAD}
        L.append(f"| {cf(k)} | {sorted(d)} | {gk}/{len(rr)} | {S2.fmt_tg(tgm)} | {x['final']:.3f} | {x['gain']:+.3f} | {x['train']:.3f} |\n")
    J["a3long"] = long
    learns = [k for k, x in long.items() if x["learns"]]
    # ---- verdict
    agree = g["final"]["label"] == g["late"]["label"]
    J["agree"] = agree
    lab = g["final"]["label"]
    if not agree:
        v = (f"UNDETERMINED (the reference config depends on the training-fit measure; A2.1): final_train -> {g['final']['label']} "
             f"(G1 {'PASS' if g['final']['g1'] else 'FAIL'}), last-25 % mean -> {g['late']['label']} (G1 {'PASS' if g['late']['g1'] else 'FAIL'})")
    elif lab == "OPEN_WD_BOTTOM":
        v = f"{OPEN_TXT} (reference {cf(tuple(g['final']['ref']))}; fails on {', '.join(g['final']['fails_on'])})"
    elif lab == "FAIL":
        v = f"{FAIL_TXT} (fails on {', '.join(g['final']['fails_on'])}" + (f"; new configs that learn: {', '.join(cf(k) for k in new_learns)}" if new_learns else "") + ")"
    else:
        if learns:   # A3.1: triggered as soon as >= 2 seeds grok, also while A3-long is incomplete
            v = (f"{WALL_TXT} -- QUALIFIER (A3-long): none learns box5 m6 within 16 B at " + ", ".join(
                f"{k} ({long[k]['grok']}/{long[k]['n']} seeds, >= 2/5 reached at t_grok {S2.fmt_tg(long[k]['tg_2nd'])} = {long[k]['tg_2nd'] / B:.1f} B)"
                for k in learns) + ": 'rescue' only as 'within the 4 B budget'; rescue (4 B) vs 加速档 is the user's call"
                 + (f" (A3-long not complete: {'; '.join(missl)})" if missl else ""))
        elif missl:
            v = f"PROVISIONAL (A3-long not complete: {'; '.join(missl)}): {WALL_TXT}"
        else:
            v = f"{WALL_TXT}; A3-long: none " + ", ".join(f"{long[k]['grok']}/5" for k in long) + " within 16 B at lr 1e-2 wd {0.03, 0.01}"
    if lab in ("FAIL", "OPEN_WD_BOTTOM") and agree:   # A3.1, descriptive: the 加速档 ratio (Stage 2's definition, over the union)
        tg_none = min(S2.cmed([ts.cens(union[k][s]) for s in SEEDS]) for k in union)
        ratio = math.inf if math.isinf(tg_none) else tg_none / a["tg_med"]
        J["accel_ratio_union"] = ratio
        v += f" [descriptive: none's earliest median t_grok over the 28 configs {S2.fmt_tg(tg_none)} / assign's {S2.fmt_tg(a['tg_med'])} = {ratio:.1f}]"
    J["verdict"] = v
    J["labels"] = [g["final"]["label"], g["late"]["label"]]
    J["g1"] = g
    L.append(f"\n**A3 verdict (post hoc): {v}**\n\nF94 and the Stage-2 pre-registered verdict are not rewritten; A3 can only remove the "
             "'rescue' label, never add one.\n")
    return "".join(L), J


def J_rows_probe(r2, r3, cell, B):
    """selftest only: the union none rows of a synthetic root pair (to show what the global edge rule would say)."""
    rows = {}
    for root in (r2, r3):
        d = {}
        for p in sorted(root.glob("*/run_none_seed*.json")):
            r = json.loads(p.read_text())
            if r["steps"] == 4 * B:
                d.setdefault((r["lr"], r["wd"]), []).append(r)
        rows.update({k: R1.none_row(sorted(v, key=lambda r: r["seed"]), B, 4 * B) for k, v in d.items()})
    return rows


# ---------------------------------------------------------------------------------------------------- self-test
def selftest():
    import shutil
    import tempfile
    top = Path(tempfile.mkdtemp(prefix="nm_a3_selftest_"))
    r2, r3 = top / "s2", top / "a3"
    B = 2000
    cell = {"frac": 0.067, "seeds": [{"seed": s, "split_sha": f"sha{s}", "n_train": 918} for s in SEEDS]}
    flat = lambda v: (lambda st: v)
    grok_at = lambda t, lo=0.008: (lambda st: 1.0 if st >= t else lo)
    crashed = lambda lr, wd: lr >= 3e-2 and wd >= 1.0

    def write(root, cdir, arm, seed, lr, wd, steps, f, ce=None, train=None, over=None, strict=None):
        ev = list(range(0, steps, 100)) + ([steps - 1] if (steps - 1) % 100 else [])
        c = [{"step": s, "test": f(s), "train": (train(s) if train else 1.0), "test_ce": (ce(s) if ce else 30.0), "test_strict": (strict(s) if strict else min(f(s), 1.0)),
              "test_nonimplied": f(s), "cka_expo": 0.5} for s in ev]
        r = {"task": "box", "box": [[-2, 2]] * 3, "arm": arm, "seed": seed, "lr": lr, "wd": wd, "lam": 1.0, "frac": 0.067, "split": "class",
             "steps": steps, "eval_every": 100, "t_grok": next((x["step"] for x in c if x["test"] >= 0.95), None), "final_test": c[-1]["test"],
             "final_train": c[-1]["train"], "curve": c, "code_sha": "synthetic", "device": "cuda", "split_sha": f"sha{seed}", "n_train": 918,
             "chance_majority": 0.011, "elim_strict": 0.0085, "n_out": 125, "closure": 1.0, "nullity": 3, **(over or {})}
        d = root / cdir; d.mkdir(parents=True, exist_ok=True)
        (d / f"run_{arm}_seed{seed}.json").write_text(json.dumps(r))

    def stage2(shuf_groks=False):
        for (lr, wd) in S2.NONE_ALL:
            for s in SEEDS:
                write(r2, f"none-{lr}-{wd}", "none", s, lr, wd, 4 * B, flat(0.012 if crashed(lr, wd) else 0.008),
                      ce=flat(4.9 if crashed(lr, wd) else 30.0), train=flat(0.02 if crashed(lr, wd) else 1.0))
        for arm, fn in (("expo@always", lambda lr: grok_at(400) if lr == 1e-3 else grok_at(900)), ("specrand@always", lambda lr: flat(0.008)),
                        ("shuf@always", lambda lr: grok_at(400) if shuf_groks else flat(0.008))):
            for lr in S2.LRS:
                for s in SEEDS:
                    write(r2, f"prior-{lr}", arm, s, lr, 1.0, B, fn(lr))

    mem = dict(f=lambda lr, wd, s: flat(0.008), ce=lambda lr, wd, s: flat(30.0), train=lambda lr, wd, s: flat(1.0))

    def a3(main=None, long=None, skip=lambda lr, wd, s, L: False, over=None):
        m = {**mem, **(main or {})}
        for (lr, wd) in A3_GRID:
            for s in SEEDS:
                if not skip(lr, wd, s, False):
                    write(r3, f"none-{lr}-{wd}", "none", s, lr, wd, 4 * B, m["f"](lr, wd, s), ce=m["ce"](lr, wd, s), train=m["train"](lr, wd, s),
                          over=(over or {}).get((lr, wd, s)), strict=m["strict"](lr, wd, s) if "strict" in m else None)
        lg = {**mem, **(long or {})}
        for (lr, wd) in A3L_GRID:
            for s in SEEDS:
                if not skip(lr, wd, s, True):
                    write(r3, f"none-{lr}-{wd}-16B", "none", s, lr, wd, L_MULT * B, lg["f"](lr, wd, s), ce=lg["ce"](lr, wd, s), train=lg["train"](lr, wd, s))

    def run(name, want, shuf_groks=False, **kw):
        shutil.rmtree(r2, ignore_errors=True); shutil.rmtree(r3, ignore_errors=True)
        stage2(shuf_groks); a3(**kw)
        txt, J = read_a3(r2, r3, cell, B_expected=None, pin=None)
        assert J["verdict"].startswith(want), (name, J["verdict"], txt[-3000:])
        print(f"selftest '{name}': {J['verdict'][:150]}")
        return txt, J

    t, J = run("all wall", "低 wd 方向也是墙")
    assert J["agree"] and "A3-long: none 0/5, 0/5 within 16 B" in J["verdict"] and len(J["a3_configs_read"]) == 9 and len(J["a3long_configs_read"]) == 2
    assert all(x["all"] for x in J["per_new_config"].values()) and J["g1"]["final"]["bracket"] == "flat over the extended grid"
    assert "A3 configs read (4 B): lr 0.003 wd 0 (5 seeds)" in t and "| union comparator" in t
    t, J = run("one new config groks", "G1 在低 wd 方向不成立",
               main={"f": lambda lr, wd, s: grok_at(6000) if (lr, wd) == (1e-2, 0.03) and s in (100, 103, 108) else flat(0.008)})
    assert not J["per_new_config"]["lr0.01-wd0.03"]["grok"] and "new configs that learn: lr 0.01 wd 0.03" in J["verdict"]
    rise = lambda st: 0.01 + 0.12 * max(0, st - 6000) / 2000   # 0.01 until 0.75 x 4 B, then up to 0.13 (no grok, final <= 0.20)
    t, J = run("wd 0 rising", "区间仍开着（wd 下沿）", main={"f": lambda lr, wd, s: rise if (lr, wd) == (1e-2, 0.0) else flat(0.008)})
    assert J["g1"]["final"]["ref"] == [1e-2, 0.0] and "gain" in J["g1"]["final"]["fails_on"] and J["labels"] == ["OPEN_WD_BOTTOM"] * 2
    t, J = run("wd 0 config at the cap", "G1 在低 wd 方向不成立",
               main={"f": lambda lr, wd, s: (lambda st: 0.01 + 0.3 * max(0, st - 6000) / 2000) if (lr, wd) == (1e-2, 0.0) else flat(0.008)})
    assert J["labels"] == ["FAIL", "FAIL"]   # rising at wd 0 but also median final > 0.20: 'not a wall', not 'open'
    t, J = run("A3-long groks", "低 wd 方向也是墙",
               long={"f": lambda lr, wd, s: grok_at(20000) if wd == 0.01 and s != 126 else flat(0.008)})
    assert "QUALIFIER (A3-long): none learns box5 m6 within 16 B at lr0.01-wd0.01 (4/5 seeds, >= 2/5 reached at t_grok 20000 = 10.0 B)" in J["verdict"], J["verdict"]
    t, J = run("A3-long incomplete", "PROVISIONAL (A3-long not complete", skip=lambda lr, wd, s, L: L and s == 126)
    t, J = run("A3-long exactly 2/5", "低 wd 方向也是墙", long={"f": lambda lr, wd, s: grok_at(20000 if s == 100 else 24000) if wd == 0.03 and s in (100, 103) else flat(0.008)})
    assert "lr0.01-wd0.03 (2/5 seeds, >= 2/5 reached at t_grok 24000 = 12.0 B)" in J["verdict"], J["verdict"]
    t, J = run("A3-long incomplete, already 2 groks", "低 wd 方向也是墙", skip=lambda lr, wd, s, L: L and s == 126,
               long={"f": lambda lr, wd, s: grok_at(20000) if wd == 0.01 and s in (100, 103) else flat(0.008)})
    assert "QUALIFIER (A3-long)" in J["verdict"] and "A3-long not complete" in J["verdict"] and not J["verdict"].startswith("PROVISIONAL")
    # A3.1 (review I1): a non-flat union with the reference at lr 3e-3 wd 0.03, whose lr neighbour 1e-3 never ran at wd 0.03:
    # the global edge test says 'closed' (WALL); the local test says 'OPEN at lr bottom (neighbour ... not run)' -> FAIL
    t, J = run("L-shaped grid edge", "G1 在低 wd 方向不成立",
               main={"f": lambda lr, wd, s: flat(0.10) if (lr, wd) == (3e-3, 0.03) else flat(0.008), "strict": lambda lr, wd, s: flat(0.008)})
    assert J["g1"]["final"]["fails_on"] == ["bracket"] and J["g1"]["late"]["fails_on"] == ["bracket"], J["g1"]["final"]["fails_on"]   # the edge alone
    assert J["g1"]["final"]["ref"] == [3e-3, 0.03] and "neighbour lr 0.001 wd 0.03 not run" in J["g1"]["final"]["bracket"], J["g1"]["final"]["bracket"]
    assert R1.bracket_of({k: v for k, v in J_rows_probe(r2, r3, cell, B).items()}, (3e-3, 0.03), 0.011) == "closed"   # the global rule would close it
    assert "descriptive: none's earliest median t_grok" in J["verdict"] and "0.100" in t
    t, J = run("A3 main incomplete", "INCOMPLETE", skip=lambda lr, wd, s, L: (not L) and (lr, wd) == (3e-2, 0.0) and s == 100)
    assert "lr 0.03 wd 0: seeds [103, 108, 122, 126]" in t
    osc = lambda st: 1.0 if (st >= 7900 or (st // 500) % 2 == 0) else 0.4
    t, J = run("A2.1 disagreement", "UNDETERMINED",
               main={"f": lambda lr, wd, s: flat(0.011) if (lr, wd) == (1e-2, 0.03) else flat(0.008),
                     "ce": lambda lr, wd, s: (lambda st: 4.0 - 0.1 * st / 8000) if (lr, wd) == (1e-2, 0.03) else flat(30.0),
                     "train": lambda lr, wd, s: osc if (lr, wd) == (1e-2, 0.03) else flat(1.0)})
    assert J["labels"] == ["FAIL", "WALL"], J["labels"]
    t, J = run("foreign wd", "NO VERDICT", over={(3e-3, 0.0, 100): {"wd": 0.05}})
    t, J = run("foreign steps", "NO VERDICT", over={(3e-3, 0.0, 100): {"steps": 16 * B}})
    t, J = run("Stage 2 not PASS", "n/a (Stage 2 is not PASS", shuf_groks=True)
    shutil.rmtree(r2, ignore_errors=True); shutil.rmtree(r3, ignore_errors=True)
    stage2(); a3()
    txt, J = read_a3(r2, r3, cell, B_expected=None)   # the registered pin: synthetic code_sha -> Stage 2 NO VERDICT -> INCOMPLETE
    assert J["verdict"].startswith("INCOMPLETE (Stage 2: NO VERDICT"), J["verdict"]
    print("selftest 'code_sha != pin': INCOMPLETE (Stage 2: NO VERDICT)")
    shutil.rmtree(top, ignore_errors=True)
    print("A3 READOUT SELFTEST PASSED")


if __name__ == "__main__":
    cell = json.loads((HERE / "preflight" / "stage2_seeds.json").read_text())["cells"]["box5|6"]
    assert tuple(x["seed"] for x in cell["seeds"]) == SEEDS
    if "--selftest" in ARGS:
        selftest()
    else:
        r2 = Path(ARGS[ARGS.index("--root_s2") + 1]).resolve() if "--root_s2" in ARGS else HERE / "out_stage2"
        r3 = Path(ARGS[ARGS.index("--root_a3") + 1]).resolve() if "--root_a3" in ARGS else HERE / "out_a3"
        txt, J = read_a3(r2, r3, cell)
        dest = HERE if "--root_a3" not in ARGS else r3.parent
        (dest / "readout_a3.md").write_text(txt)
        clean = lambda x: ({str(k): clean(v) for k, v in x.items()} if isinstance(x, dict) else [clean(v) for v in x]
                           if isinstance(x, (list, tuple)) else "inf" if isinstance(x, float) and math.isinf(x) else x)
        (dest / "readout_a3.json").write_text(json.dumps(clean(J), indent=1))
        print(txt)
        print(f"read: Stage 2 {r2}, A3 {r3}; A3 configs {J.get('a3_configs_read')}; A3-long configs {J.get('a3long_configs_read')}; "
              f"wrote {dest / 'readout_a3.md'} / .json")
