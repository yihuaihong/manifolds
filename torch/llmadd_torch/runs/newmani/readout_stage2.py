"""runs/newmani STAGE 2 read-out: box5 m6, Tier L (PRE-REGISTERED 2026-10-25, before any Stage-2 run exists; AUDIT.md
"Stage 2 (box5 m6, Tier L)": the design's G1-G4 with Amendment A1 and A2). From the run JSONs; no training.
Uses readout.py (Stage 1) for the shared definitions (none_row, late_ce, wall_reasons, bracket_of, trainable,
reference_config) and the pinned toy_struct for upto / auc / compare_values / median.

    python readout_stage2.py                 # reads runs/newmani/out_stage2, writes readout_stage2.md / .json next to this file
    python readout_stage2.py --root DIR      # another root (tests); outputs go to DIR/..
    python readout_stage2.py --selftest      # synthetic cells with known verdicts (must pass before any reading is used)

The cell: --task box --box=-2:2,-2:2,-2:2, frac 0.067, class split, seeds 100, 103, 108, 122, 126 (preflight/
stage2_seeds.json, box5|6; every run's split_sha must equal that file's). B = 50,000.
  none   the 12 Stage-1 configs + the 7 rule-2 extension configs (lr 1e-1 x wd {0.1, 1, 3}, wd 10 x 4 lrs), 4 B
  prior  expo@always (assign, Tier L), specrand@always, shuf@always; lr {1e-3, 3e-3, 1e-2}, wd 1, lam 1, B
Missing or extra data -> INCOMPLETE / NO VERDICT (never a G verdict).
G1 wall: every none config (all 19, crashed ones included) groks on <= floor(0.2 n) = 1 of 5 seeds within 4 B, median final
   held-out <= 0.20, median last-25 % gain < 0.02, median final strict <= the floor line; and the A2 reference config
   (trainable = median final_train >= 0.99; most groks, then AUC 4B) has a last-25 % CE drop < 0.01 nats while A1's gate
   is on (late CE <= log n_out + 1; else n/a). No trainable config -> n/a (fails). Bracket closed: flat (every config
   0 groks and AUC 4B <= chance + 0.05) over the extended grid, or the reference config not on an edge of the grid.
G2 rescue: assign at its best lr (median AUC_B, all held-out) groks within B on >= 4/5 seeds, and its paired AUC_B test
   (compare_values: median difference >= 0.05, wins >= 4/5, exact one-sided MWU p < 0.025, median above the floor line)
   vs the none comparator passes on all held-out AND on test_strict. Comparator (A2 item 5): per read-out set the none
   config with the highest median AUC_B on that set over EVERY config (ties -> lower lr).
G3 specific: assign beats specrand AND shuf (each at its own best lr by median AUC_B) by the same test on both sets.
G4 transferable: assign's median t_grok_B <= 0.4 B (20k) and every seed's n_train <= 1,800.
Verdict: G1-G4 -> PASS (next rainbow round); G1 fails, G2-G4 pass and none's earliest median t_grok (4 B, every config) /
   assign's median t_grok_B >= 3 -> 加速档 (the user's call); else NOT PASSED. A1's G1 is printed beside A2's.
Amendment A2.1 (2026-10-25, coordinator; before any Stage-2 result): the whole verdict is computed twice, with trainable =
   median final_train >= 0.99 (A2, registered) and with trainable = median last-25 % mean train >= 0.99. If the two G1s
   (pass / fail) or the two verdict categories (PASS / 加速档 / other) differ, the verdict is "UNDETERMINED (the reference
   config depends on the training-fit measure)" with both results listed: never PASS, never 加速档."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ARGS = sys.argv[1:]
assert set(a for a in ARGS if a.startswith("--")) <= {"--root", "--selftest"}, f"unknown option(s) {ARGS}"
_spec = importlib.util.spec_from_file_location("readout", HERE / "readout.py")
R1 = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]   # readout.py parses its own argv at import: give it none
_spec.loader.exec_module(R1)
sys.argv = _argv
ts, med, AUC_MIN = R1.ts, R1.med, R1.AUC_MIN

SEEDS = (100, 103, 108, 122, 126)        # AUDIT.md "Stage 2 seeds", box5 m6 (asserted = stage2_seeds.json)
B_REG = 50000
PIN = "ea797e5321d3"                      # the trainer every Stage-2 run must carry (code_sha)
NONE_ALL = list(R1.NONE_GRID) + list(R1.FLAT_EXT)
ROLES = {"expo@always": "assign", "specrand@always": "specrand", "shuf@always": "shuf"}
LRS = R1.PRIOR_LRS
G4_FRAC, G4_NTRAIN, ACC_RATIO = 0.4, 1800, 3.0
NEED = math.ceil(0.8 * len(SEEDS))       # 4 of 5
assert len(NONE_ALL) == 19 and NEED == 4


def cmed(v):
    return med(sorted(v))   # inf-safe median (toy_struct.median_vals sorts; inf = never)


def fmt_tg(v):
    return "never" if math.isinf(v) else f"{v:.0f}"


def read_stage2(root: Path, cell: dict, B_expected: int | None = B_REG, pin: str | None = PIN) -> tuple[str, dict]:
    """cell = stage2_seeds.json's box5|6 entry (frac, seeds with split_sha / n_train). B_expected / pin None: selftest only."""
    rs = [json.loads(p.read_text()) for p in sorted(root.glob("*/run_*.json"))]
    L = [f"# newmani Stage 2 read-out: box5 m6, Tier L ({root})\n\n"]
    J = {"root": str(root), "n_runs": len(rs), "verdict": None}
    exp_seeds = {x["seed"]: x for x in cell["seeds"]}
    assert tuple(sorted(exp_seeds)) == SEEDS, ("not the pre-registered Stage-2 seeds", sorted(exp_seeds))
    if not rs:
        J["verdict"] = "NO VERDICT (no runs)"; return "".join(L) + "no runs\n", J
    shas = sorted({str(r.get("code_sha")) for r in rs})
    devs = sorted({str(r.get("device")) for r in rs})
    bad = [f"{r['arm']} seed {r['seed']}: " + ", ".join(w for w, c in (
               ("not box5", not (r.get("task") == "box" and r.get("box") == [[-2, 2]] * 3)),
               ("frac", abs(r.get("frac", -1) - cell["frac"]) > 1e-12), ("split", r.get("split") != "class"),
               ("seed not pre-registered", r["seed"] not in exp_seeds),
               ("split_sha", r["seed"] in exp_seeds and r.get("split_sha") != exp_seeds[r["seed"]]["split_sha"]),
               ("arm", r["arm"] != "none" and r["arm"] not in ROLES),
               ("prior wd / lam", r["arm"] != "none" and (r.get("wd") != 1.0 or r.get("lam") != 1.0))) if c)
           for r in rs]
    bad = [b for b in bad if not b.endswith(": ")]
    J.update({"code_sha": shas, "device": devs})
    if len(shas) != 1 or len(devs) != 1 or bad or (pin is not None and shas != [pin]):
        J["verdict"] = "NO VERDICT"
        return "".join(L) + (f"NO VERDICT: {len(rs)} runs; code_sha {shas}; device {devs}; unexpected runs: {bad[:10]}\n"), J
    none, prior = {}, {}
    for r in rs:
        if r["arm"] == "none":
            d = none.setdefault((r["lr"], r["wd"]), {})
        else:
            d = prior.setdefault(ROLES[r["arm"]], {}).setdefault(r["lr"], {})
        assert r["seed"] not in d, ("duplicate run", r["arm"], r["lr"], r["wd"], r["seed"])
        d[r["seed"]] = r
    missing = [f"none lr {k[0]:g} wd {k[1]:g}: " + (f"seeds {sorted(none[k])}" if k in none else "absent")
               for k in NONE_ALL if set(none.get(k, {})) != set(SEEDS)]
    missing += [f"none lr {k[0]:g} wd {k[1]:g}: not in the registered grid" for k in none if k not in NONE_ALL]
    missing += [f"{role} lr {lr:g}: " + (f"seeds {sorted(prior[role][lr])}" if lr in prior.get(role, {}) else "absent")
                for role in ROLES.values() for lr in LRS if set(prior.get(role, {}).get(lr, {})) != set(SEEDS)]
    missing += [f"{role} lr {lr:g}: not in the registered lrs" for role in prior for lr in prior[role] if lr not in LRS]
    Bs = sorted({r["steps"] for d in prior.values() for x in d.values() for r in x.values()})
    B4s = sorted({r["steps"] for d in none.values() for r in d.values()})
    if not missing and (len(Bs) != 1 or len(B4s) != 1 or B4s[0] != 4 * Bs[0] or (B_expected is not None and Bs[0] != B_expected)):
        missing.append(f"budgets: prior {Bs}, none {B4s} (needs B = {B_expected} and none at 4 B)")
    J["missing"] = missing
    L.append(f"{len(rs)} runs; code_sha {shas[0]}; device {devs[0]}; seeds {list(SEEDS)} (pre-registered; split_sha = "
             f"stage2_seeds.json for every run)\n\n")
    if missing:
        J["verdict"] = "INCOMPLETE"
        L.append(f"**INCOMPLETE: nothing is read** ({len(missing)} gaps): " + "; ".join(missing[:40]) + "\n")
        return "".join(L), J
    B, B4 = Bs[0], B4s[0]
    by_seed = {}
    for r in rs:
        f = {k: r.get(k) for k in ("split_sha", "n_train", "chance_majority", "elim_strict", "n_out", "closure", "nullity")}
        assert by_seed.setdefault(r["seed"], f) == f, ("split facts differ within a seed", r["seed"])
    n_out = by_seed[SEEDS[0]]["n_out"]
    assert n_out == 125 and all(f["closure"] == 1.0 and f["nullity"] == 3 for f in by_seed.values()), by_seed
    chance = max(1 / n_out, med([f["chance_majority"] for f in by_seed.values()]))
    floor = max(chance, med([f["elim_strict"] for f in by_seed.values()])) + AUC_MIN
    ntr = {s: f["n_train"] for s, f in sorted(by_seed.items())}
    J.update({"B": B, "chance": chance, "floor_line": floor, "n_train": ntr})
    L.append(f"B {B}, none 4 B {B4}; n_train per seed {ntr}; chance {chance:.4f}; floor line {floor:.4f}; A2 trainable = median "
             f"final_train >= {R1.TRAINABLE_THR:g}; A1 gate log(n_out) + {R1.CE_GATE_NATS:g} = {math.log(n_out) + R1.CE_GATE_NATS:.2f}\n\n")
    # ---- none sweep (19 configs x 5 seeds)
    rows, tr, trl, aucBs, tg4, tstat = {}, {}, {}, {}, {}, {}
    L.append("| none lr | wd | n | groks in 4B | median final train | final test median / max | final strict median | AUC 4B | AUC_B "
             "| AUC_B strict | last 25%: gain / CE drop (late CE) | t_grok median (4B) |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for k in sorted(none):
        rr = [none[k][s] for s in SEEDS]
        rows[k] = x = R1.none_row(rr, B, B4)
        tr[k] = med([R1.train_level(r) for r in rr])
        trl[k] = med([R1.train_level(r, "late") for r in rr])   # descriptive (review 2026-10-25): the last-25 % mean train
        tstat[k] = {s: (r["final_train"], R1.train_level(r, "late"), min(c["train"] for c in r["curve"] if c["step"] >= 0.75 * r["steps"]))
                    for s, r in zip(SEEDS, rr)}
        aucBs[k] = med([ts.upto(r, B, "test_strict")[1] for r in rr])
        tg4[k] = cmed([ts.cens(r) for r in rr])
        L.append(f"| {k[0]:g} | {k[1]:g} | {x['n']} | {x['grok']}/{x['n']} | {tr[k]:.3f} | {x['final_med']:.3f} / {x['final_max']:.3f} "
                 f"| {x['strict']:.3f} | {x['auc4']:.3f} | {x['aucB']:.3f} | {aucBs[k]:.3f} | {x['gain']:+.3f} / {x['ce_drop']:+.3f} "
                 f"({x['ce_late']:.2f}) | {fmt_tg(tg4[k])} |\n")
    cf = lambda k: "n/a" if k is None else f"lr {k[0]:g} wd {k[1]:g}"
    ok = R1.trainable(tr, R1.TRAINABLE_THR)
    ok_late = R1.trainable(trl, R1.TRAINABLE_THR)
    g1 = {}
    for rule, ref in (("A2", R1.reference_config(rows, ok)), ("A1", max(rows, key=lambda k: (rows[k]["grok"], rows[k]["auc4"], -k[0]))),
                      ("late", R1.reference_config(rows, ok_late))):   # 'late' = Amendment A2.1's second measure (a verdict gate)
        whyd, ce = R1.wall_reasons(rows, ref, n_out, floor, gate=True, cap_key="final_med")
        na = ref is None
        wall = not na and not whyd
        br = R1.bracket_of(rows, ref, chance)
        closed = br == "closed" or br == "flat over the extended grid"
        g1[rule] = {"ref": None if na else list(ref), "ref_train": None if na else tr[ref], "ref_train_late": None if na else trl[ref],
                    "ce": ce, "why": list(whyd.values()),
                    "na": na, "wall": wall, "bracket": br, "closed": closed, "pass": wall and closed,
                    "fails_on": (["n/a"] if na else sorted(whyd) + ([] if closed else ["bracket"]))}
    L.append(f"\ntrainable none configs (A2): {len(ok)}/{len(rows)}; not trainable: " + ", ".join(f"{cf(k)} ({tr[k]:.3f})" for k in sorted(rows) if k not in ok) + "\n\n")
    L.append("| G1 | A2 (registered) | A1 (pre-A2, for the record) |\n|---|---|---|\n")
    cel = lambda c: ("n/a (no reference config)" if c["late_ce_best"] is None else
                     f"late CE {c['late_ce_best']:.2f} vs gate {c['gate']:.2f} -> {'on' if c['on'] else 'n/a'}; drop {c['drop_best']:+.3f}"
                     + (" FIRES" if c["fires"] else ""))
    L.append(f"| reference config | {cf(g1['A2']['ref'] and tuple(g1['A2']['ref']))}"
             + (f" (train {g1['A2']['ref_train']:.3f})" if not g1["A2"]["na"] else "") + f" | {cf(tuple(g1['A1']['ref']))} "
             f"(train {g1['A1']['ref_train']:.3f}) |\n| CE component | {cel(g1['A2']['ce'])} | {cel(g1['A1']['ce'])} |\n"
             f"| wall | " + " | ".join("n/a (no trainable none config)" if g["na"] else f"{g['wall']}" + ("" if g["wall"] else " (" + "; ".join(g["why"]) + ")")
                                     for g in (g1["A2"], g1["A1"])) + " |\n"
             f"| bracket | {g1['A2']['bracket']} | {g1['A1']['bracket']} |\n| **G1** | **{'PASS' if g1['A2']['pass'] else 'FAIL'}** | "
             f"{'PASS' if g1['A1']['pass'] else 'FAIL'} |\n\n")
    gl = g1["late"]
    J["g1_measures_differ"] = gl["pass"] != g1["A2"]["pass"]
    L.append(f"Amendment A2.1 second measure (a gate, see the verdict): G1 with trainable = median last-25 % mean train >= {R1.TRAINABLE_THR:g} "
             f"({len(ok_late)}/{len(rows)} configs): reference {cf(gl['ref'] and tuple(gl['ref']))}, CE {cel(gl['ce'])}, bracket {gl['bracket']} "
             f"-> **{'n/a' if gl['na'] else 'PASS' if gl['pass'] else 'FAIL'}**" + ("" if gl["pass"] or gl["na"] else f" (fails on {', '.join(gl['fails_on'])})")
             + (" -- **THE REGISTERED AND THE LAST-25 % G1 DIFFER (A2.1: the verdict is UNDETERMINED)**" if J["g1_measures_differ"] else
                " (A2.1 also compares the verdict categories: see the verdict block)") + "\n\n")
    L.append("per-seed training fit (final / last-25 % mean / last-25 % min) of the reference configs (A2, A1, last-25 %):\n\n"
             "| config | role | " + " | ".join(f"s{s}" for s in SEEDS) + " |\n|---|---|" + "---|" * len(SEEDS) + "\n")
    for rule in ("A2", "A1", "late"):
        k = g1[rule]["ref"]
        if k is not None:
            k = tuple(k)
            L.append(f"| {cf(k)} | {rule} reference | " + " | ".join("%.3f / %.3f / %.3f" % tstat[k][s] for s in SEEDS) + " |\n")
    L.append("\n")
    # ---- prior arms: best lr by median AUC_B (all held-out)
    pick = {}
    L.append("| role | lr chosen (median AUC_B per lr) | groks in B | t_grok_B median | AUC_B | AUC_B strict | final test | final strict "
             "| CKA to expo (end) |\n|---|---|---|---|---|---|---|---|---|\n")
    for role in ("assign", "specrand", "shuf"):
        st = {lr: med([ts.upto(r, B)[1] for r in d.values()]) for lr, d in prior[role].items()}
        lr = max(st, key=lambda k: (st[k], -k))
        rr = {s: prior[role][lr][s] for s in SEEDS}
        u = {s: ts.upto(r, B) for s, r in rr.items()}
        us = {s: ts.upto(r, B, "test_strict")[1] for s, r in rr.items()}
        pick[role] = x = {"lr": lr, "aucB_per_lr": {f"{k:g}": v for k, v in st.items()}, "tg": {s: u[s][0] for s in SEEDS},
                          "auc": {s: u[s][1] for s in SEEDS}, "auc_strict": us, "grokB": sum(u[s][0] < math.inf for s in SEEDS),
                          "tg_med": cmed([u[s][0] for s in SEEDS]), "final": med([r["final_test"] for r in rr.values()]),
                          "final_strict": med([R1.final_of(r, "test_strict") for r in rr.values()]),
                          "cka": med([r["curve"][-1].get("cka_expo", float("nan")) for r in rr.values()])}
        L.append(f"| {role} | {lr:g} (" + ", ".join(f"{k:g}: {v:.3f}" for k, v in sorted(st.items())) + f") | {x['grokB']}/5 | "
                 f"{fmt_tg(x['tg_med'])} "
                 f"| {med(list(x['auc'].values())):.3f} | {med(list(us.values())):.3f} | {x['final']:.3f} | {x['final_strict']:.3f} | {x['cka']:.3f} |\n")
    # ---- paired tests (read on the first B steps, paired by seed)
    thr_chance = floor - AUC_MIN   # compare_values' 'above chance + AUC_MIN' = above the floor line

    def test(a_tg, a_auc, b_tg, b_auc):
        t = ts.compare_values([a_tg[s] for s in SEEDS], [b_tg[s] for s in SEEDS], [a_auc[s] for s in SEEDS], [b_auc[s] for s in SEEDS], thr_chance)
        return {"pass": t["auc"]["pass"], "med_a": t["auc"]["med_a"], "med_b": t["auc"]["med_b"], "wins": t["auc"]["wins"], "p": t["auc"]["p"]}

    nb = {"test": max(rows, key=lambda k: (rows[k]["aucB"], -k[0])), "strict": max(rows, key=lambda k: (aucBs[k], -k[0]))}
    nbt = {"test": max(ok, key=lambda k: (rows[k]["aucB"], -k[0])) if ok else None,
           "strict": max(ok, key=lambda k: (aucBs[k], -k[0])) if ok else None}
    nside = lambda k, key: ({s: ts.upto(none[k][s], B)[0] for s in SEEDS}, {s: ts.upto(none[k][s], B, key)[1] for s in SEEDS})
    a = pick["assign"]
    tests = {}
    for setn, key, aa in (("test", "test", a["auc"]), ("strict", "test_strict", a["auc_strict"])):
        tests[f"G2 vs none {setn}"] = test(a["tg"], aa, *nside(nb[setn], key))
        if nbt[setn] is not None and nbt[setn] != nb[setn]:
            tests[f"(descriptive) vs trainable-only none {setn}"] = test(a["tg"], aa, *nside(nbt[setn], key))
        for c in ("specrand", "shuf"):
            cc = pick[c]
            tests[f"G3 vs {c} {setn}"] = test(a["tg"], aa, cc["tg"], cc["auc"] if setn == "test" else cc["auc_strict"])
    L.append(f"\npaired AUC_B tests (compare_values' AUC part; floor line {floor:.4f}; none comparator per set = the highest median "
             f"AUC_B over every config: all held-out {cf(nb['test'])}, strict {cf(nb['strict'])}; trainable only: "
             f"{cf(nbt['test'])} / {cf(nbt['strict'])})\n\n| test | assign median | other median | wins | p | pass |\n|---|---|---|---|---|---|\n")
    for name, t in tests.items():
        L.append(f"| {name} | {t['med_a']:.3f} | {t['med_b']:.3f} | {t['wins']}/5 | {t['p']:.4f} | {t['pass']} |\n")
    L.append("\nper-seed training fit (final / last-25 % mean / last-25 % min) of the G2 comparators: "
             + "; ".join(f"{setn} {cf(nb[setn])}: " + ", ".join("s%d %.3f / %.3f / %.3f" % ((s,) + tstat[nb[setn]][s]) for s in SEEDS)
                         for setn in ("test", "strict")) + "\n")
    g2 = {"grok": a["grokB"] >= NEED, "test": tests["G2 vs none test"]["pass"], "strict": tests["G2 vs none strict"]["pass"]}
    g2["pass"] = all(g2.values())
    g3 = {k: tests[k]["pass"] for k in tests if k.startswith("G3")}
    g3["pass"] = all(g3.values())
    g4 = {"tg_med": a["tg_med"], "tg_ok": a["tg_med"] <= G4_FRAC * B, "n_train_max": max(ntr.values()), "n_ok": max(ntr.values()) <= G4_NTRAIN}
    g4["pass"] = g4["tg_ok"] and g4["n_ok"]
    none_tg = min(tg4.values())
    ratio = math.inf if math.isinf(none_tg) and not math.isinf(a["tg_med"]) else (none_tg / a["tg_med"] if a["tg_med"] > 0 else math.inf)
    if math.isinf(a["tg_med"]):
        ratio = 0.0
    J.update({"g1": g1, "g2": g2, "g3": g3, "g4": g4, "tests": tests, "prior": {k: {kk: vv for kk, vv in v.items()} for k, v in pick.items()},
              "none_comparator": {k: list(v) for k, v in nb.items()}, "trainable": [list(k) for k in ok], "accel_ratio": ratio,
              "none_earliest_tg": none_tg})

    def verdict(g1pass, g1na=False, fails_on=()):
        if g1na:   # A2 item 4: nothing fits its training set -> no wall and no 加速档 reading (the baseline did not train)
            return "NOT PASSED (G1 n/a: no none config fits its training set; no 加速档 reading; check the none runs)"
        if g1pass and g2["pass"] and g3["pass"] and g4["pass"]:
            return "PASS (G1-G4): Tier L on box5 m6 -> next rainbow round"
        if not g1pass and g2["pass"] and g3["pass"] and g4["pass"] and ratio >= ACC_RATIO:
            never = math.isinf(none_tg)
            return (f"加速档 (G1 fails on {', '.join(fails_on)}; G2-G4 pass; "
                    + ("no none config's median t_grok is finite in 4 B (none never groks), ratio inf" if never else
                       f"none's earliest median t_grok / assign's = {ratio:.1f} >= {ACC_RATIO:g}") + "): the user's call")
        fail = [g for g, p in (("G1", g1pass), ("G2", g2["pass"]), ("G3", g3["pass"]), ("G4", g4["pass"])) if not p]
        return "NOT PASSED (" + ", ".join(fail) + " fail" + (f"; 加速档 ratio {ratio:.1f} < {ACC_RATIO:g}" if fail == ["G1"] else "") + ")"

    J["verdict_A2_final_train_only"] = verdict(g1["A2"]["pass"], g1["A2"]["na"], g1["A2"]["fails_on"])
    J["verdict_A1"] = verdict(g1["A1"]["pass"], False, g1["A1"]["fails_on"])
    J["verdict_late"] = verdict(g1["late"]["pass"], g1["late"]["na"], g1["late"]["fails_on"])
    # Amendment A2.1 (coordinator, 2026-10-25, before any Stage-2 result): the two training-fit measures must agree on G1
    # and on the verdict category, else UNDETERMINED (never PASS, never 加速档)
    cat = lambda v: "PASS" if v.startswith("PASS") else "加速档" if v.startswith("加速档") else "other"
    agree = g1["A2"]["pass"] == g1["late"]["pass"] and cat(J["verdict_A2_final_train_only"]) == cat(J["verdict_late"])
    J["a21"] = {"agree": agree, "g1": [g1["A2"]["pass"], g1["late"]["pass"]], "cat": [cat(J["verdict_A2_final_train_only"]), cat(J["verdict_late"])]}
    g1s = lambda g: "n/a" if g["na"] else "PASS" if g["pass"] else "FAIL on " + ", ".join(g["fails_on"])
    # the UNDETERMINED text names categories and G1 states only (no verbatim PASS / 加速档 verdict inside it; review 2026-10-25)
    J["verdict"] = J["verdict_A2_final_train_only"] if agree else (
        "UNDETERMINED (the reference config depends on the training-fit measure; Amendment A2.1): "
        f"final_train measure: G1 {g1s(g1['A2'])}, category {J['a21']['cat'][0]}; "
        f"last-25 % mean measure: G1 {g1s(g1['late'])}, category {J['a21']['cat'][1]}")
    assert agree == (not J["verdict"].startswith("UNDETERMINED")), (agree, J["verdict"])
    assert agree == (J["a21"]["g1"][0] == J["a21"]["g1"][1] and J["a21"]["cat"][0] == J["a21"]["cat"][1])
    assert not agree or J["verdict"] == J["verdict_A2_final_train_only"] == J["verdict_late"] or cat(J["verdict_late"]) == cat(J["verdict"])
    L.append(f"\n| G | result | detail |\n|---|---|---|\n"
             f"| G1 wall (A2) | {'PASS' if g1['A2']['pass'] else 'FAIL'} | wall {'n/a' if g1['A2']['na'] else g1['A2']['wall']}, bracket {g1['A2']['bracket']} |\n"
             f"| G2 rescue | {'PASS' if g2['pass'] else 'FAIL'} | assign lr {a['lr']:g}: groks {a['grokB']}/5 (need {NEED}); AUC_B vs none: all held-out "
             f"{g2['test']}, strict {g2['strict']} |\n"
             f"| G3 specific | {'PASS' if g3['pass'] else 'FAIL'} | " + ", ".join(f"{k[3:]} {v}" for k, v in g3.items() if k != "pass") + " |\n"
             f"| G4 transferable | {'PASS' if g4['pass'] else 'FAIL'} | median t_grok_B {fmt_tg(g4['tg_med'])} "
             f"(<= {G4_FRAC * B:.0f}: {g4['tg_ok']}); max n_train {g4['n_train_max']} (<= {G4_NTRAIN}: {g4['n_ok']}) |\n\n"
             f"**Verdict (A2 + A2.1): {J['verdict']}**\n\n"
             f"- read with trainable = median final_train >= 0.99 alone (A2; not the verdict): G1 {g1s(g1['A2'])} -> {J['verdict_A2_final_train_only']}\n"
             f"- read with trainable = median last-25 % mean train >= 0.99 alone (A2.1; not the verdict): G1 {g1s(g1['late'])} -> {J['verdict_late']}\n"
             f"- A2.1: the two measures {'agree' if agree else 'DISAGREE -> UNDETERMINED'}\n\n"
             f"For the record, with A1's G1: {J['verdict_A1']}\n\n"
             "Wording (pre-registered): 'not learned' only within this one-layer toy, these optimisers and the 4 B budget; the target "
             "encodes the table (Tier L: 'given the structure, the model learns to use it'); 'geometry-specific' only from G3; one task, "
             "one tier, two read-out sets (multiplicity: 5 tasks x 2 sets screened in Stage 1). Tier S is not in Stage 2 (Stage-1 floor).\n")
    return "".join(L), J


# ---------------------------------------------------------------------------------------------------- self-test
def selftest():
    import shutil
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="nm_s2_selftest_")) / "out"
    B = 2000
    cell = {"frac": 0.067, "seeds": [{"seed": s, "split_sha": f"sha{s}", "n_train": 918} for s in SEEDS]}
    flat = lambda v: (lambda st: v)
    grok_at = lambda t, lo=0.01: (lambda st: 1.0 if st >= t else lo)

    def write(cell_dir, arm, seed, lr, wd, steps, f, strict=None, ce=None, train=None, over=None):
        ev = list(range(0, steps, 100)) + ([steps - 1] if (steps - 1) % 100 else [])
        c = [{"step": s, "test": f(s), "train": (train(s) if train else 1.0), "test_ce": (ce(s) if ce else 30.0),
              "test_strict": (strict(s) if strict else f(s)), "test_nonimplied": f(s), "cka_expo": 0.5} for s in ev]
        r = {"task": "box", "box": [[-2, 2]] * 3, "arm": arm, "seed": seed, "lr": lr, "wd": wd, "lam": 1.0, "frac": 0.067, "split": "class",
             "steps": steps, "eval_every": 100, "t_grok": next((x["step"] for x in c if x["test"] >= 0.95), None), "final_test": c[-1]["test"],
             "final_train": c[-1]["train"], "curve": c, "code_sha": "synthetic", "device": "cuda", "split_sha": f"sha{seed}", "n_train": 918,
             "chance_majority": 0.011, "elim_strict": 0.0085, "n_out": 125, "closure": 1.0, "nullity": 3, **(over or {})}
        d = root / cell_dir; d.mkdir(parents=True, exist_ok=True)
        (d / f"run_{arm}_seed{seed}.json").write_text(json.dumps(r))

    crashed = lambda lr, wd: lr >= 3e-2 and wd >= 1.0   # synthetic: lr 3e-2 / 1e-1 at wd >= 1 cannot fit training
    mem_none = dict(f=lambda lr, wd, s: flat(0.012 if crashed(lr, wd) else 0.008), ce=lambda lr, wd, s: flat(4.9 if crashed(lr, wd) else 30.0),
                    train=lambda lr, wd, s: flat(0.02 if crashed(lr, wd) else 1.0))

    def make(none=None, roles=None, skip=lambda arm, lr, wd, s: False, strict_assign=None):
        nn = {**mem_none, **(none or {})}
        for (lr, wd) in NONE_ALL:
            for s in SEEDS:
                if not skip("none", lr, wd, s):
                    write(f"none-{lr}-{wd}", "none", s, lr, wd, 4 * B, nn["f"](lr, wd, s), ce=nn["ce"](lr, wd, s), train=nn["train"](lr, wd, s),
                          strict=(nn["strict"](lr, wd, s) if "strict" in nn else None))
        ro = {"expo@always": lambda lr, s: grok_at(400) if lr == 1e-3 else grok_at(900), "specrand@always": lambda lr, s: flat(0.008),
              "shuf@always": lambda lr, s: flat(0.008), **(roles or {})}
        for arm, fn in ro.items():
            for lr in LRS:
                for s in SEEDS:
                    if not skip(arm, lr, 1.0, s):
                        write(f"prior-{lr}", arm, s, lr, 1.0, B, fn(lr, s),
                              strict=(strict_assign(lr, s) if strict_assign and arm == "expo@always" else None))

    def run(name, want, **kw):
        shutil.rmtree(root, ignore_errors=True)
        make(**kw)
        txt, J = read_stage2(root, cell, B_expected=None, pin=None)
        assert J["verdict"].startswith(want), (name, J["verdict"], txt[-2500:])
        return txt, J

    # 1 PASS: none memorises (below chance) or crashes; assign groks 5/5 at 400 steps; controls at the floor
    t, J = run("pass", "PASS (G1-G4)")
    assert not J["g1_measures_differ"] and J["a21"]["agree"] and J["verdict_late"] == J["verdict"] == J["verdict_A2_final_train_only"] and "per-seed training fit" in t
    assert "the two measures agree" in t
    assert J["g1"]["A2"]["pass"] and J["g1"]["A2"]["bracket"] == "flat over the extended grid" and not J["g1"]["A2"]["ce"]["on"]
    assert J["none_comparator"]["test"] in ([3e-2, 1.0], [1e-1, 1.0]) and J["prior"]["assign"]["lr"] == 1e-3 and J["g2"]["pass"] and J["g3"]["pass"]
    print(f"selftest 1 pass: {J['verdict']} (comparator {J['none_comparator']['test']}, all configs; A2 reference {J['g1']['A2']['ref']})")
    # 2 the F1 situation: the crashed config wins A1's ranking and its CE drifts inside the gate -> A1's G1 fails, A2's passes
    t, J = run("F1", "PASS (G1-G4)", none={"ce": lambda lr, wd, s: (lambda st: 4.9 - 0.1 * st / 8000) if crashed(lr, wd) else flat(30.0)})
    assert not J["g1"]["A1"]["pass"] and J["g1"]["A1"]["ce"]["fires"] and J["g1"]["A2"]["pass"] and J["verdict_A1"].startswith("加速档")   # A1: G1 fails on CE noise
    print(f"selftest 2 F1: A2 {J['verdict'][:20]}..., A1 {J['verdict_A1']}")
    # 3 G1 fails (none config lr 1e-2 wd 1 groks on 3/5 seeds at 6000 steps, after B) -> 加速档 (6000 / 400 = 15 >= 3)
    t, J = run("G1 fail -> accel", "加速档", none={"f": lambda lr, wd, s: grok_at(6000, 0.008) if (lr, wd) == (1e-2, 1.0) and s in (100, 103, 108)
                                                   else flat(0.012 if crashed(lr, wd) else 0.008)})
    assert not J["g1"]["A2"]["pass"] and "none groks" in " ".join(J["g1"]["A2"]["why"]) and abs(J["accel_ratio"] - 15) < 1e-9, J["accel_ratio"]
    print(f"selftest 3 G1 fail: {J['verdict']}")
    # 3b G1 fails, assign not 3x faster than none (none groks 3/5 at 800: ratio 2) -> NOT PASSED (G1)
    t, J = run("G1 fail, no accel", "NOT PASSED (G1 fail; 加速档 ratio 2.0", none={"f": lambda lr, wd, s: grok_at(800, 0.008) if (lr, wd) == (1e-2, 1.0)
                                                                            and s in (100, 103, 108) else flat(0.012 if crashed(lr, wd) else 0.008)})
    assert J["g2"]["pass"] and J["none_comparator"]["test"] == [1e-2, 1.0], J["none_comparator"]   # the grokking config is the comparator
    print(f"selftest 3b G1 fail without speed-up: {J['verdict']}")
    # 3c G1 fails on the CE of a trainable reference (falling inside the gate); none never groks -> 加速档 (ratio inf)
    t, J = run("G1 CE", "加速档", none={"f": lambda lr, wd, s: flat(0.012 if crashed(lr, wd) else 0.011 if (lr, wd) == (1e-2, 3.0) else 0.008),
                                                "ce": lambda lr, wd, s: (lambda st: 4.0 - 0.1 * st / 8000) if (lr, wd) == (1e-2, 3.0) else flat(4.9 if crashed(lr, wd) else 30.0)})
    assert J["g1"]["A2"]["ref"] == [1e-2, 3.0] and J["g1"]["A2"]["ce"]["fires"]
    print(f"selftest 3c G1 fails on the trainable reference's CE: {J['verdict']}")
    # 4 G2 fails: assign groks on 3/5 seeds only
    t, J = run("G2 grok", "NOT PASSED (G2", roles={"expo@always": lambda lr, s: grok_at(400) if s in (100, 103, 108) else flat(0.008)})
    assert not J["g2"]["grok"]
    print(f"selftest 4 G2 (3/5 groks): {J['verdict']}")
    # 4b G2 fails on test_strict only (assign learns the non-strict cells only)
    t, J = run("G2 strict", "NOT PASSED (G2", strict_assign=lambda lr, s: flat(0.008))
    assert J["g2"]["grok"] and J["g2"]["test"] and not J["g2"]["strict"]
    print(f"selftest 4b G2 (strict set at the floor): {J['verdict']}")
    # 5 G3 fails: shuf also groks
    t, J = run("G3", "NOT PASSED (G3", roles={"shuf@always": lambda lr, s: grok_at(400)})
    assert J["g2"]["pass"] and not J["g3"]["pass"]
    print(f"selftest 5 G3 (shuf groks too): {J['verdict']}")
    # 6 G4 fails: assign groks only at 1200 > 0.4 B = 800
    t, J = run("G4", "NOT PASSED (G4", roles={"expo@always": lambda lr, s: grok_at(1200)})
    assert J["g2"]["pass"] and not J["g4"]["tg_ok"]
    print(f"selftest 6 G4 (t_grok 1200 > 0.4 B): {J['verdict']}")
    # 7 no trainable none config -> G1 n/a -> NOT PASSED
    t, J = run("no trainable", "NOT PASSED (G1 n/a", none={"train": lambda lr, wd, s: flat(0.02)})
    assert J["g1"]["A2"]["na"] and J["g1"]["A1"]["pass"] and "n/a (no trainable none config)" in t
    print(f"selftest 7 no trainable none config: A2 {J['verdict']}; A1 {J['verdict_A1'][:20]}")
    # 8 incomplete data: a missing extension config, a missing seed of a prior arm, a missing lr
    for nm, sk in (("extension config missing", lambda arm, lr, wd, s: arm == "none" and (lr, wd) == (1e-1, 3.0)),
                   ("prior seed missing", lambda arm, lr, wd, s: arm == "shuf@always" and lr == 3e-3 and s == 126),
                   ("assign lr missing", lambda arm, lr, wd, s: arm == "expo@always" and lr == 1e-2)):
        t, J = run(nm, "INCOMPLETE", skip=sk)
        assert "nothing is read" in t and J["missing"], nm
        print(f"selftest 8 '{nm}': INCOMPLETE ({J['missing'][0]})")
    # 9 foreign data -> NO VERDICT: a non-registered seed, another code_sha, a wrong split_sha
    for nm, over in (("seed 4 (Stage 1)", {"seed": 4}), ("code_sha", {"code_sha": "other"}), ("split_sha", {"split_sha": "x"})):
        shutil.rmtree(root, ignore_errors=True)
        make()
        p = root / "prior-0.001" / "run_expo@always_seed100.json"
        r = json.loads(p.read_text()); r.update(over)
        if "seed" in over:
            p = p.with_name("run_expo@always_seed4.json")
        p.write_text(json.dumps(r))
        txt, J = read_stage2(root, cell, B_expected=None, pin=None)
        assert J["verdict"] == "NO VERDICT", (nm, J["verdict"])
        print(f"selftest 9 '{nm}': NO VERDICT")
    # 10 the registered budget is enforced in the real reading (B 2000 here != 50000)
    shutil.rmtree(root, ignore_errors=True)
    make()
    txt, J = read_stage2(root, cell, pin=None)
    assert J["verdict"] == "INCOMPLETE" and "budgets" in J["missing"][0], J["missing"]
    print("selftest 10 budget: B != 50,000 -> INCOMPLETE in the registered reading")
    txt, J = read_stage2(root, cell, B_expected=None)   # the registered pin: these synthetic runs carry code_sha 'synthetic'
    assert J["verdict"] == "NO VERDICT", J["verdict"]
    print("selftest 10b code_sha != the pin ea797e5321d3 -> NO VERDICT in the registered reading")
    # 11 (review 2026-10-25) an oscillating config passes final_train (1.0 at the last eval) but not the last-25 % mean
    # (~0.7); it has the best trainable AUC 4B and a CE falling inside the gate: the registered G1 fails on its CE (verdict
    # 加速档, none never groks, reason named), the descriptive last-25 % G1 passes -> 'differ' flagged, both verdicts printed
    osc = lambda st: 1.0 if (st >= 7900 or (st // 500) % 2 == 0) else 0.4
    t, J = run("oscillating reference", "UNDETERMINED (the reference config depends",
               none={"f": lambda lr, wd, s: flat(0.012 if crashed(lr, wd) else 0.011 if (lr, wd) == (1e-2, 3.0) else 0.008),
                     "ce": lambda lr, wd, s: (lambda st: 4.0 - 0.1 * st / 8000) if (lr, wd) == (1e-2, 3.0) else flat(4.9 if crashed(lr, wd) else 30.0),
                     "train": lambda lr, wd, s: osc if (lr, wd) == (1e-2, 3.0) else flat(0.02 if crashed(lr, wd) else 1.0)})
    assert J["g1"]["A2"]["ref"] == [1e-2, 3.0] and not J["g1"]["A2"]["pass"] and J["g1"]["late"]["pass"] and J["g1_measures_differ"], J["g1"]
    assert J["verdict_A2_final_train_only"].startswith("加速档 (G1 fails on ce;") and "none never groks" in J["verdict_A2_final_train_only"] and J["verdict_late"].startswith("PASS")
    assert "THE REGISTERED AND THE LAST-25 % G1 DIFFER" in t and "DISAGREE -> UNDETERMINED" in t and "| lr 0.01 wd 3 | A2 reference | 1.000 / 0." in t
    assert "category 加速档" in J["verdict"] and "category PASS" in J["verdict"] and "next rainbow round" not in J["verdict"]   # both listed, no verbatim PASS
    assert ("alone (A2; not the verdict): G1 FAIL on ce -> " + J["verdict_A2_final_train_only"]) in t and ("not the verdict): G1 PASS -> " + J["verdict_late"]) in t
    print(f"selftest 11 (A2.1) 加速档 vs PASS: {J['verdict'][:62]}... (registered 加速档, last-25 % PASS)")
    # 12 (A2.1) PASS -> UNDETERMINED: config X = lr 1e-2 wd 3 has the best trainable AUC 4B and a CE falling inside the gate;
    # its final_train is 0.98 (last eval only; last-25 % mean 0.999): A2 leaves it out (memoriser reference, CE n/a: PASS),
    # the last-25 % measure takes it as the reference (CE fires: 加速档) -> UNDETERMINED
    xX = lambda lr, wd: (lr, wd) == (1e-2, 3.0)
    dip = lambda st: 0.98 if st >= 7999 else 1.0
    noneX = {"f": lambda lr, wd, s: flat(0.012 if crashed(lr, wd) else 0.011 if xX(lr, wd) else 0.008),
             "ce": lambda lr, wd, s: (lambda st: 4.0 - 0.1 * st / 8000) if xX(lr, wd) else flat(4.9 if crashed(lr, wd) else 30.0),
             "train": lambda lr, wd, s: dip if xX(lr, wd) else flat(0.02 if crashed(lr, wd) else 1.0)}
    t, J = run("PASS -> UNDETERMINED", "UNDETERMINED (the reference config depends", none=noneX)
    assert J["verdict_A2_final_train_only"].startswith("PASS") and J["verdict_late"].startswith("加速档") and J["g1"]["late"]["ref"] == [1e-2, 3.0], (J["verdict_A2_final_train_only"], J["g1"])
    print(f"selftest 12 (A2.1) PASS vs 加速档: UNDETERMINED (registered {J['verdict_A2_final_train_only'][:12]}, last-25 % {J['verdict_late'][:22]})")
    # 13 (A2.1) disagreement, neither is PASS: (a) the same none sweep but shuf groks too -> NOT PASSED (G3) vs NOT PASSED
    # (G1, G3): G1 differs -> UNDETERMINED; (b) every trainable config oscillates (final 1.0, last-25 % mean ~0.7) and X's CE
    # falls: 加速档 vs NOT PASSED (G1 n/a: nothing fits under the last-25 % measure) -> UNDETERMINED, never 加速档
    t, J = run("disagree, both NOT PASSED", "UNDETERMINED (the reference config depends", none=noneX, roles={"shuf@always": lambda lr, s: grok_at(400)})
    assert J["verdict_A2_final_train_only"].startswith("NOT PASSED (G3") and J["verdict_late"].startswith("NOT PASSED (G1, G3") and J["a21"]["cat"] == ["other", "other"]
    print(f"selftest 13a (A2.1) G1 differs, both NOT PASSED: UNDETERMINED ({J['verdict_A2_final_train_only']} / {J['verdict_late']})")
    t, J = run("disagree, 加速档 vs NOT PASSED", "UNDETERMINED (the reference config depends",
               none={**noneX, "train": lambda lr, wd, s: flat(0.02) if crashed(lr, wd) else osc})
    assert J["verdict_A2_final_train_only"].startswith("加速档") and J["verdict_late"].startswith("NOT PASSED (G1 n/a") and J["a21"]["g1"] == [False, False]
    print(f"selftest 13b (A2.1) 加速档 vs NOT PASSED (G1 n/a): UNDETERMINED, not 加速档")
    # 14 (A2.1, bracket path; review 2026-10-25) a non-flat cell: Y = lr 3e-3 wd 1 holds held-out 0.1 (> chance + 0.05, strict
    # 0.008, no grok) and has the best AUC 4B; it fits its training set at the last eval only (last-25 % mean ~0.7). A2's
    # reference Y is interior -> bracket closed -> PASS; the last-25 % reference is the lr 1e-3 wd 0.1 corner -> OPEN -> 加速档
    yY = lambda lr, wd: (lr, wd) == (3e-3, 1.0)
    t, J = run("bracket path", "UNDETERMINED (the reference config depends",
               none={"f": lambda lr, wd, s: flat(0.1 if yY(lr, wd) else 0.012 if crashed(lr, wd) else 0.008),
                     "strict": lambda lr, wd, s: flat(0.008),
                     "train": lambda lr, wd, s: osc if yY(lr, wd) else flat(0.02 if crashed(lr, wd) else 1.0)})
    assert J["g1"]["A2"]["bracket"] == "closed" and J["g1"]["late"]["bracket"].startswith("OPEN at") and J["verdict_A2_final_train_only"].startswith("PASS")
    assert J["verdict_late"].startswith("加速档 (G1 fails on bracket") and not J["g1"]["late"]["ce"]["fires"], (J["g1"]["late"], J["verdict_late"])
    print(f"selftest 14 (A2.1) bracket closed vs {J['g1']['late']['bracket']}: UNDETERMINED (PASS vs 加速档 via the edge test)")
    # 15 (A2.1) PASS vs G1 n/a: every non-crashed config oscillates (final 1.0, last-25 % mean ~0.7), no CE fall -> A2 PASS,
    # the last-25 % measure has no trainable config -> NOT PASSED (G1 n/a) -> UNDETERMINED
    t, J = run("PASS vs n/a", "UNDETERMINED (the reference config depends",
               none={"train": lambda lr, wd, s: flat(0.02) if crashed(lr, wd) else osc})
    assert J["verdict_A2_final_train_only"].startswith("PASS") and J["g1"]["late"]["na"] and "G1 n/a, category other" in J["verdict"]
    print("selftest 15 (A2.1) PASS vs G1 n/a: UNDETERMINED")
    shutil.rmtree(root.parent, ignore_errors=True)
    print("STAGE 2 READOUT SELFTEST PASSED")


if __name__ == "__main__":
    S2 = json.loads((HERE / "preflight" / "stage2_seeds.json").read_text())
    cell = S2["cells"]["box5|6"]
    assert tuple(x["seed"] for x in cell["seeds"]) == SEEDS and cell["frac"] == 0.067, cell["frac"]
    if "--selftest" in ARGS:
        selftest()
    else:
        root = Path(ARGS[ARGS.index("--root") + 1]).resolve() if "--root" in ARGS else HERE / "out_stage2"
        txt, J = read_stage2(root, cell)
        dest = HERE if "--root" not in ARGS else root.parent
        (dest / "readout_stage2.md").write_text(txt)
        clean = lambda x: ({str(k): clean(v) for k, v in x.items()} if isinstance(x, dict) else [clean(v) for v in x]
                           if isinstance(x, (list, tuple)) else "inf" if isinstance(x, float) and math.isinf(x) else x)
        (dest / "readout_stage2.json").write_text(json.dumps(clean(J), indent=1))
        print(txt)
        print(f"seeds {list(SEEDS)}; A2 trainable threshold {R1.TRAINABLE_THR:g}; wrote {dest / 'readout_stage2.md'} / .json")
