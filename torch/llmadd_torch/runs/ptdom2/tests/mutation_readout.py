"""ptdom2 test: the read-out self-test bites. Each mutation (an exact, counted text edit of a temp copy of readout_ptdom2.py, run
from the directory of the readout under test so its imports resolve) must make the self-test FAIL (or crash); the unmutated copy
must PASS. The readout under test = the one beside this tests/ directory (runs/ptdom2, or a working copy such as
runs/ptdom2/wip_a4). A4: + the A3 audit's nine blind spots (y1-y8, y11) and its caught ones (y9, y10, y12), + the A4 rules (z*);
run in parallel (MUT_WORKERS, default 4). -> mutation_readout.out"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R2D = Path(__file__).resolve().parent.parent
SRC = (R2D / "readout_ptdom2.py").read_text()
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
F3 = "    if any(v is None or not v[1] > 0 for v in f3s.values()):"
GSN = "    gsn = pair(R_, b, S_[0], NS_[0], P, NS_[1], d, seeds, [])[0]"
MUT = {
    "m0 none (control)": [],
    "m1 wins >= 5 instead of 6": [("WINS_MIN = 6 ", "WINS_MIN = 5 ")],
    "m2 halves condition dropped": [('"halves_gt0": h1 > 0 and h2 > 0', '"halves_gt0": True')],
    "m3 bootstrap condition dropped": [('"boot90_lo_gt0": lo > 0', '"boot90_lo_gt0": True')],
    "m4 no Holm (raw p)": [("    h1 = dict(zip(ok_fam, RO.holm([p1[d] for d in ok_fam]))) if ok_fam else {}\n", "    h1 = dict(p1)\n"),
                           ("    h2 = dict(zip(ok_fam, RO.holm([p2[d] for d in ok_fam]))) if ok_fam else {}\n", "    h2 = dict(p2)\n")],
    "m5 geometry gain against none instead of the shuffled arm": [
        ('    gs, _, sv = pair(R_, b, T_[0], S_[0], P, P, d, seeds, problems)', '    gs, _, sv = pair(R_, b, T_[0], N_[0], P, N, d, seeds, problems)')],
    "m6 R verdict on the selection seeds 0-3": [('x = contrast_domain(R_, subs, b, d, v["pick_prior"], v["pick_none"], R2_SEEDS, problems,',
                                                 'x = contrast_domain(R_, subs, b, d, v["pick_prior"], v["pick_none"], SEL, problems,')],
    "m7 NOT FINAL never (an edge after the one step counts as final)": [('            x.update(status="NOT FINAL", why=list(x["edges"]))',
                                                                          '            x["status"] = "FINAL"')],
    "m8 the prior extension never asked": [('            if after == "R1" and top_p == M2.R1_LR:', '            if False:')],
    "m9 the none upper step never asked": [('            if after == "R1" and float(hi_n) <= float(max(M2.NONE_LRS0[b], key=float)):', '            if False:')],
    "m14 the none lower step never asked (A1.1a)": [('            if after == "R1" and float(lo_n) >= float(min(M2.NONE_LRS0[b], key=float)):', '            if False:')],
    "m15 fixB never asked (A1.1b)": [('        if not (min(rng) <= float(P) <= max(rng)):', '        if False:')],
    "m16 the trigger on the argmax only (A1.2 dropped)": [('        at = lambda u, lr: lr in (u["pick"], u["argmax"])', '        at = lambda u, lr: lr == u["argmax"]')],
    "m17 the gate at the pick ignored (A1.7)": [('        if not (gt["prior@P s8-11"]["ok"] and gt["prior@P s12-15"]["ok"]):', '        if False:')],
    "m18 A6 literal for shuffled arms (A1.4 dropped)": [('A6_SHUF = "adjacent"', 'A6_SHUF = "literal"')],
    "m19 outside problems ignored (A2.4)": [('    final = len(ok_fam) == len(fam) and not problems and not ext_probs', '    final = len(ok_fam) == len(fam) and not problems')],
    "m20 the reading lines swapped (A2.2)": [('           (False, True): "GEOMETRY-SPECIFIC without PASS: true > shuffled, but no gain vs none shown (the shuffled prior may hurt)",',
                                               '           (False, True): "PASS and GEOMETRY-SPECIFIC: the gain depends on the correct geometry",')],
    "m21 the window split over every eval (A2.2)": [('        w = [WIN[0] <= st <= WIN[1] for st in x["steps"]]', '        w = [True for st in x["steps"]]')],
    "m22 no FLOOR / CEILING in the domain rows (A2.3)": [('           "floor": floor_tags(R_, subs, b, [T_, N_], d, seeds) if g is not None else None,',
                                                          '           "floor": None,')],
    "m23 A3.2 FLOOR condition dropped": [('    if any(tg.startswith("FLOOR") for fl in (x.get("floor"), x.get("floor_geo")) if fl for tg in fl["tags"]):',
                                          '    if False:')],
    "m24 A3.2 / A4.1 final3 condition dropped": [(F3, "    if False:")],
    "m25 A4.2a shuffled-worse condition dropped": [("        if not hi_ >= 0:", "        if False:")],
    "m26 A3.2 dose condition dropped": [('    if lt is None or lsh is None or abs(lt - lsh) > LM_DIFF:', '    if False:')],
    "m27 A3.1 S PASS against none@2e-3 (not F100's pick)": [('        N = M2.S_NONE[d]   # A3.1', '        N = lr   # A3.1')],
    "x1 WIN one eval wider (audit_a2)": [('WIN = (916, 7323)', 'WIN = (916, 7500)')],
    "x6 the shuffled arm's LM share from the true cell (audit_a2)": [('f"only-{d}-shuf@{P}": cg(S_[0], P, seeds)["median"],', 'f"only-{d}-shuf@{P}": cg(T_[0], P, seeds)["median"],')],
    "x7 the FLOOR > 100 note dropped (audit_a2)": [('f"tag holds whenever both arms are <= {line:.1f} (A3.3)]" if line > 100 else ""))', '"" if False else ""))')],
    "x8 CEILING never (audit_a2)": [('    if ref >= 95:', '    if False:')],
    "x9 no FLOOR for true vs shuffled (audit_a2)": [('           "floor_geo": floor_tags(R_, subs, b, [T_, S_], d, seeds) if gs is not None else None,', '           "floor_geo": None,')],
    "x11 the S ptdom2-source check dropped (audit_a2)": [('        if any(nonp2.values()):', '        if False:')],
    "x13 the cka double path always true (audit_a2)": [('        ok_ = [None not in (rs["cka_wlast"].get(d), rn["cka_wlast"].get(d), rt["cka_wlast"].get(d)) and rs["cka_wlast"][d] <= rn["cka_wlast"][d] + 0.10\n               and rs["cka_wlast"][d] <= rt["cka_wlast"][d] - 0.2 for rt, rs, rn in zip(recs[T_], recs[S_], recs[NS_])]',
                                                        '        ok_ = [True for rt, rs, rn in zip(recs[T_], recs[S_], recs[NS_])]')],
    "m10 gate exclusion dropped": [('        bad = {lr for lr, g in gate.items() if not g["ok"]}', '        bad = set()')],
    "m11 S on the selected seeds 0-7 (A2.1 dropped)": [('        x = contrast_domain(R_, subs, b, d, lr, N, R2_SEEDS, problems, N_same=lr)',
                                                       '        x = contrast_domain(R_, subs, b, d, lr, N, M2.S_SEEDS, problems, N_same=lr)')],
    "m12 Part R never refuses a missing run": [('    if (problems or ext_probs) and not allow_partial:\n        raise SystemExit("Part R verdict refused: "',
                                                '    if False:\n        raise SystemExit("Part R verdict refused: "')],
    "m13 loader identity check dropped": [("        if idbad:\n", "        if False:\n")],
    # ---- A4: the A3 audit's mutations (tests/audit_a3.md, minor 4; y1-y8 and y11 were not caught by the A3 self-test)
    "y1 check_s_none never refuses (always [])": [("    sn = M2.S_NONE if s_none is None else s_none\n", "    return []\n")],
    "y2 A4.1 final3: one gain's lower bound suffices ('and' -> 'or')": [(F3, F3.replace("if any(", "if all("))],
    "y3 A4.1 final3: only the vs-none gain checked": [(F3, F3.replace("for v in f3s.values()", 'for v in [f3s["gain_final3"]]'))],
    "y4 A4.2a the bar loosened (upper bound >= -1)": [("        if not hi_ >= 0:", "        if not hi_ >= -1.0:")],
    "y5 LM_DIFF 0.1 -> 0.5": [("SHARE_MIN, LM_DIFF = 0.5, 0.1", "SHARE_MIN, LM_DIFF = 0.5, 0.5")],
    "y6 shuffled - none VALUE against none@N (label still N_same)": [(GSN, "    gsn = pair(R_, b, S_[0], N_[0], P, N_[1], d, seeds, [])[0]")],
    "y7 the sentence kept when not final (--allow_partial / problems)": [
        ("""            v1, v2 = f"PROVISIONAL ({v1} if final; {x['would_be_holm']})", f"PROVISIONAL ({v2} if final)"\n            reading = None\n""",
         """            v1, v2 = f"PROVISIONAL ({v1} if final; {x['would_be_holm']})", f"PROVISIONAL ({v2} if final)"\n""")],
    "y8 the sentence kept at a gate failure at the pick": [
        ("            v1 = v2 = RO.GATE_PICK_FAIL\n            reading = None\n", "            v1 = v2 = RO.GATE_PICK_FAIL\n")],
    "y9 the same-lr PASS line against none@N (= the PASS contrast)": [
        ('            gl = pair(R_, b, f"only-{d}", "none", lr, lr, d, R2_SEEDS, [])[0]', '            gl = pair(R_, b, f"only-{d}", "none", lr, N, d, R2_SEEDS, [])[0]')],
    "y10 the dose condition reads true vs none (ls[2])": [("    lt, lsh = (ls[0], ls[1]) if len(ls) >= 2 else (None, None)",
                                                          "    lt, lsh = (ls[0], ls[2]) if len(ls) >= 3 else (None, None)")],
    "y11 S's FLOOR (true vs none) against none@2e-3, not F100's none": [
        ('           "floor": floor_tags(R_, subs, b, [T_, N_], d, seeds) if g is not None else None,',
         '           "floor": floor_tags(R_, subs, b, [T_, NS_], d, seeds) if g is not None else None,')],
    "y12 the gate at the pick: s8-15 only (halves dropped)": [
        ('        if not (gt["prior@P s8-11"]["ok"] and gt["prior@P s12-15"]["ok"]):   # A1.7',
         '        if not gt["prior@P s8-15"]["ok"]:   # A1.7')],
    # ---- A4's own rules
    "z1 A4.2b (the shuffled arm's share of the gain) dropped": [("    if g_ is None or gs_ is None or not gs_ >= SHARE_MIN * g_:", "    if False:")],
    "z2 A4.2b at 0.2 instead of 0.5": [("SHARE_MIN, LM_DIFF = 0.5, 0.1", "SHARE_MIN, LM_DIFF = 0.2, 0.1")],
    "z3 A4.1 back to A3's point rule (final3 means > 0)": [(F3, F3.replace("v[1] > 0", "v[0] > 0"))],
    "z4 A4.2a back to A3's point rule (mean >= -1)": [("        if not hi_ >= 0:", "        if not mn(sn) >= -1.0:")],
    "z5 A4.3 the gate needs the descriptive none again": [("    have_ts = ok_arm[T_] and ok_arm[S_]", "    have_ts = ok_arm[T_] and ok_arm[S_] and ok_arm[NS_]")],
    "z6 A4.3 the source check counts missing runs again": [('if v_ is not None and v_ != "ptdom2"]', 'if v_ != "ptdom2"]')],
    "z7 A4.4 the window check ignored": [("        elif winbad:\n", "        elif False:\n")],
    "z8 A4.4 the reduced-Holm label always": [('                                  if len(ok_fam) < len(fam) else f"Holm over all', '                                  if True else f"Holm over all')],
    "z9 A4.3 a missing descriptive none refuses the verdict": [(GSN, GSN.replace("seeds, [])[0]", "seeds, problems)[0]"))],
    "z10 the closing paragraph keeps A2.2's sentence": [
        ('''"a domain's reading line states the geometry dependence only when PASS, GEOMETRY-SPECIFIC and every A3.2 / A4 condition hold "''',
         '''"the gain depends on the correct geometry only when both hold (A2.2) "''')],
    "z11 A4.3 the same-lr line silently absent when its none is missing": [
        ('''                x["same_lr_pass"] = {"missing": x.get("ns_missing") or [f"a run of only-{d}@{lr} is missing / invalid"]}''', "                pass")],
    "z12 A4.3 the 'descriptive only' gap also for a verdict-arm none (N = N_same)": [("    ns_miss = None if (ok_arm[NS_] or NS_ == N_) else [", "    ns_miss = None if ok_arm[NS_] else [")],
    "a5c A5c (harm interval) disabled": [("        if not lo5 >= -HARM_MAX:", "        if False:")],
    "a5d A5d (absorption interval) disabled": [("        if not hi5 <= SHARE_MIN * tn5:", "        if False:")],
    "a5c2 A5c on the mean instead of the interval": [("        if not lo5 >= -HARM_MAX:", "        if not float(np.mean(sn)) >= -HARM_MAX:")],
    "a5d2 A5d on the mean instead of the interval": [("        if not hi5 <= SHARE_MIN * tn5:", "        if not float(np.mean(sn)) <= SHARE_MIN * tn5:")],
}


def run(item):
    i, (name, edits) = item
    t = SRC
    for a, b in edits:
        assert t.count(a) == 1, (name, a[:60], t.count(a))
        t = t.replace(a, b)
    tmp = R2D / f"_mut_readout_{os.getpid()}_{i}.py"
    tmp.write_text(t)
    try:
        o = subprocess.run([PY, "-B", str(tmp), "--selftest"], capture_output=True, text=True, cwd=str(R2D),
                           env=dict(os.environ, PYTHONPATH="/scratch/yh6210/llmadd_torch/pylib", PYTHONDONTWRITEBYTECODE="1"), timeout=1800)
    finally:
        tmp.unlink()
    last = (o.stdout.strip().splitlines() or ["(no output)"])[-1]
    passed = o.returncode == 0 and "SELFTEST PASS" in last
    fails = [l[5:90] for l in o.stdout.splitlines() if l.startswith("FAIL ")]
    good = passed == name.startswith("m0")
    return good, (f"{'OK  ' if good else 'BAD '} {name}: self-test {'PASS' if passed else 'FAIL / crash (rc %d)' % o.returncode} -- {last[:80]}"
                  + (f"; failing cases: {fails[:3]}" if fails else ""))


for name, edits in MUT.items():   # every anchor exists exactly once before anything runs
    for a, _ in edits:
        assert SRC.count(a) == 1, (name, a[:60], SRC.count(a))
print(f"readout under test: {R2D / 'readout_ptdom2.py'}", flush=True)
ok_all = True
with ThreadPoolExecutor(max_workers=int(os.environ.get("MUT_WORKERS", "4"))) as ex:
    for good, line in ex.map(run, enumerate(MUT.items())):
        ok_all &= good
        print(line, flush=True)
print(f"MUTATION READOUT {'PASS' if ok_all else 'FAIL'} ({len(MUT) - 1} mutations each caught; the control passes)" if ok_all else "MUTATION READOUT FAIL")
