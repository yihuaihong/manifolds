from pathlib import Path
p = Path("plumb_tpp.py")
s = p.read_text()
old4 = s[s.index("# ---------------- P4 (2%: pt2main_summary)"):s.index("# ---------------- P5\n")]
new4 = '''# ---------------- P4 (2%: pt2main_summary; T6: none's tie at the upper edge -> X one step up, then rule 1 -> C)
root, R, env = fresh("p4")
mark_submitted_A(R, G2g)
for s in (0, 1, 2, 3):
    write_run(R, G2g, "none", "4.8e-3", s, .30, 1.20)
    write_run(R, G2g, "none", "9.6e-3", s, .305, 1.19)
    write_run(R, G2g, "win-ctx", "6e-4", s, .80, 1.15)
    write_run(R, G2g, "win-ctx", "1.2e-3", s, .75, 1.15)
h = hook(R, env, G2g)
d = decs(R, G2g)
check(d == {("none", "X"): "1.92e-2", ("win-ctx", "X"): "3e-4"}, f"P4 (T6): 2% none tied, pick 9.6e-3 = the upper edge -> X 1.92e-2; win-ctx X 3e-4 ({d})")
ro = (R / f"readout_{G2g}.txt").read_text() if (R / f"readout_{G2g}.txt").exists() else ""
check("pt2main pre-registered read-out" in ro and "OPEN BRACKET at the lower edge" in ro, "P4: the 2% selection is pt2main_summary's own output")
lg = (R / f"select_{G2g}.log").read_text()
check("T6: none's pick at the upper grid edge" in lg, "P4 (T6): the X decision says why")
sub = (R / "SUBMIT_CALLED.txt").read_text() if (R / "SUBMIT_CALLED.txt").exists() else ""
check(sub.count(" X tpp2-190m-t0.745-lr3e-4-win-ctx-s") == 4 and sub.count(" X tpp2-190m-t0.745-lr1.92e-2-none-s") == 4 and " C " not in sub,
      "P4: 4 + 4 2% X jobs (fake) submitted after the real --test-only, no C yet")
for s in (0, 1, 2, 3):
    write_run(R, G2g, "none", "1.92e-2", s, .30, 1.21)
    write_run(R, G2g, "win-ctx", "3e-4", s, .70, 1.15)
hook(R, env, G2g)
d = decs(R, G2g)
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(d.get(("none", "C")) == "9.6e-3" and d.get(("win-ctx", "C")) == "6e-4" and sub.count(" C tpp2-190m-t0.745-lr9.6e-3-none-s") == 4,
      f"P4 (T6): after the X runs (2% tie, lowest offline bpb at 9.6e-3) -> C 9.6e-3, 4 confirm jobs; win-ctx closed -> C 6e-4 ({d})")
'''
s = s.replace(old4, new4)
old9 = s[s.index("# ---------------- P9 (T2)"):s.index("# ---------------- P10 (T2)")]
new9 = '''# ---------------- P9 (T6): none's floor tie with the pick at the upper edge -> X automatically (was C, T2 then added X by hand)
def floor9(name):
    root, R, env = fresh(name)
    mark_submitted_A(R, G7)
    for s in (0, 1, 2, 3):
        for lr in ("4.8e-3", "9.6e-3"):
            write_run(R, G7, "none", lr, s, .040 if lr == "4.8e-3" else .041, 1.20 if lr == "4.8e-3" else 1.19)   # both at the floor
        for lr in ("1.2e-3", "2.4e-3"):
            write_run(R, G7, "win-ctx", lr, s, .88, 1.15)
    return root, R, env


root, R, env = floor9("p9")
hook(R, env, G7)
f = R / "decisions" / f"{G7}.json"
check(decs(R, G7) == {("none", "X"): "1.92e-2", ("win-ctx", "C"): "1.2e-3"}, f"P9 (T6): floor tie at the upper edge -> none X 1.92e-2, win-ctx C ({decs(R, G7)})")
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(sub.count(" X tpp7-190m-t0.745-lr1.92e-2-none-s") == 4 and sub.count(" C tpp7-190m-t0.745-lr9.6e-3-none-s") == 0,
      "P9 (T6): the 4 X jobs are submitted by the hook, no none C")
check(check_rc(R, env, G7) == 1, "P9: X runs not finished, everything submitted -> --check exit 1")
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .120, 1.18)
hook(R, env, G7)
ed = json.loads(f.read_text())["log"]
check([(e["arm"], e["stage"], e.get("lr")) for e in ed][-1] == ("none", "STOP", "1.92e-2"),
      f"P9: the X lr beats both by > 2 at the upper edge -> still open -> STOP ({[(e['arm'], e['stage'], e.get('lr')) for e in ed]})")
root, R, env = floor9("p9b")
hook(R, env, G7)
f = R / "decisions" / f"{G7}.json"
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .045, 1.17)
hook(R, env, G7)
ed = [(e["arm"], e["stage"], e.get("lr")) for e in json.loads(f.read_text())["log"]]
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(ed[-1] == ("none", "C", "1.92e-2") and sub.count(" C tpp7-190m-t0.745-lr1.92e-2-none-s") == 4 and ("none", "X", "2.4e-2") not in ed
      and len([e for e in ed if e[1] == "X" and e[0] == "none"]) == 1,
      f"P9b (T6): X ties (within 2) with the lowest bpb -> C at 1.92e-2 (the new upper edge: no second extension), 4 confirm jobs ({ed})")
# ---------------- P9c (T2 / T6 by hand): a pick made before T6 (C at the upper edge, its confirm jobs submitted) + REOPEN + X
root, R, env = floor9("p9c")
f = R / "decisions" / f"{G7}.json"
f.parent.mkdir(exist_ok=True)
f.write_text(json.dumps({"group": G7, "log": [{"seq": 0, "arm": "none", "stage": "C", "lr": "9.6e-3"},
                                              {"seq": 1, "arm": "win-ctx", "stage": "C", "lr": "1.2e-3"}]}))   # as the pre-T6 code wrote it
hook(R, env, G7)
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(sub.count(" C tpp7-190m-t0.745-lr9.6e-3-none-s") == 4 and len(decs(R, G7)) == 2, "P9c: the pre-T6 decisions -> their C jobs submitted, nothing new decided")
d = json.loads(f.read_text())
d["log"] += [{"seq": 2, "arm": "none", "stage": "REOPEN", "amend": "T6"}, {"seq": 3, "arm": "none", "stage": "X", "lr": "1.92e-2", "amend": "T6"}]
f.write_text(json.dumps(d))
check(check_rc(R, env, G7) == 0, "P9c: after REOPEN + X the X jobs are not submitted yet -> --check exit 0")
hook(R, env, G7)
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(sub.count(" X tpp7-190m-t0.745-lr1.92e-2-none-s") == 4 and len(json.loads(f.read_text())["log"]) == 4, "P9c: the hook submits the 4 X jobs, decides nothing")
for s in (0, 1, 2, 3):
    write_run(R, G7, "none", "1.92e-2", s, .010, 1.25)   # the X lr worse by 3 points: closed, the pick stays 9.6e-3
hook(R, env, G7)
ed = [(e["arm"], e["stage"], e.get("lr")) for e in json.loads(f.read_text())["log"]]
sub = (R / "SUBMIT_CALLED.txt").read_text()
check(ed[-1] == ("none", "C", "9.6e-3") and sub.count(" C tpp7-190m-t0.745-lr9.6e-3-none-s") == 4,
      f"P9c: X worse -> closed -> C at 9.6e-3 again; its confirm jobs are not submitted twice ({ed})")
'''
s = s.replace(old9, new9)
s = s.replace('''  P4  2% tpp2-190m-t0.745 via pt2main_summary: none tied (lower offline bpb at 9.6e-3) -> C at 9.6e-3; win-ctx 6e-4 beats
      1.2e-3 by 5 (OPEN lower) -> X at 3e-4''', '''  P4  2% tpp2-190m-t0.745 via pt2main_summary: none tied (lower offline bpb at 9.6e-3 = the upper edge) -> T6: X at
      1.92e-2, then C at 9.6e-3; win-ctx 6e-4 beats 1.2e-3 by 5 (OPEN lower) -> X at 3e-4, then C''')
s = s.replace('''      and returned (rc 2), not read as "nothing to do"; P17 DISK GUARD keeps the decisions, the next call of the group submits''',
 '''      and returned (rc 2), not read as "nothing to do"; P17 DISK GUARD keeps the decisions, the next call of the group submits.
      T6: none's pick at the upper grid edge is extended also when closed (P4 2%, P9 / P9b 7%: X automatically, one extension
      only; P9c: a pre-T6 C at the upper edge + REOPEN + X by hand); the lower edge (P5) and win-ctx (P1) are unchanged''')
p.write_text(s)
print("ok", s.count("P9c"), s.count("(T6)"))
