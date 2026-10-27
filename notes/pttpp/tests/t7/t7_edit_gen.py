from pathlib import Path
p = Path("/scratch/yh6210/llmadd-burst-tpp/notes/pttpp/make_jobs_pttpp.py")
s = p.read_text()
rep = [
('''    one X per arm, at most one C per arm between REOPENs, nothing after a STOP; the seqs are 0, 1, 2, ... in order."""''',
'''    one X per arm, at most one C per arm between REOPENs, nothing after a STOP; the seqs are 0, 1, 2, ... in order.
    Amendment T7 (the user's approval, 2026-10-27): a REOPEN (amend) may also withdraw a STOP; then one more X may follow
    (runs_of: one ladder step beyond the arm's last X, in the same direction); nothing else after a STOP."""'''),
('''    st = {a: {"X": None, "C": None, "STOP": False} for a in ARMS}
    for e in d["log"]:
        a, k = e["arm"], e["stage"]
        assert k in ("X", "C", "STOP", "REOPEN") and a in ARMS and not st[a]["STOP"], e
        if k == "X":
            assert st[a]["X"] is None and st[a]["C"] is None and e["lr"] in LADDER, e
            st[a]["X"] = e["lr"]
        elif k == "C":
            assert st[a]["C"] is None and e["lr"] in LADDER, e
            st[a]["C"] = e["lr"]
        elif k == "REOPEN":
            assert st[a]["C"] is not None and e.get("amend"), e
            st[a]["C"] = None
        else:
            st[a]["STOP"] = True
    return d''',
'''    st = {a: {"X": None, "C": None, "STOP": False, "more": False} for a in ARMS}
    for e in d["log"]:
        a, k = e["arm"], e["stage"]
        assert k in ("X", "C", "STOP", "REOPEN") and a in ARMS and (not st[a]["STOP"] or (k == "REOPEN" and e.get("amend"))), e
        if k == "X":
            assert (st[a]["X"] is None or st[a]["more"]) and st[a]["C"] is None and e["lr"] in LADDER, e
            st[a]["X"], st[a]["more"] = e["lr"], False
        elif k == "C":
            assert st[a]["C"] is None and e["lr"] in LADDER, e
            st[a]["C"] = e["lr"]
        elif k == "REOPEN":
            assert (st[a]["C"] is not None or st[a]["STOP"]) and e.get("amend"), e
            if st[a]["STOP"]:   # T7: a STOP withdrawn by the user's approval: one more X may follow
                assert st[a]["X"] is not None, e
                st[a]["STOP"], st[a]["more"] = False, True
            st[a]["C"] = None
        else:
            st[a]["STOP"] = True
    return d'''),
('''        elif k == "REOPEN":
            st[a]["C"] = None
        else:
            st[a]["STOP"] = True
    return st''',
'''        elif k == "REOPEN":   # T2: withdraws a C; T7: or a STOP
            st[a]["C"], st[a]["STOP"] = None, False
        else:
            st[a]["STOP"] = True
    return st'''),
('''    out = [("A", a, lr, s) for a in ARMS for lr in GRID[gid][a] for s in c["sel"]]
    for e in sorted(decisions(gid)["log"], key=lambda e: e["seq"]):
        if e["stage"] == "X":
            assert e["lr"] not in GRID[gid][e["arm"]] and e["lr"] in (step_out(GRID[gid][e["arm"]][1], "upper"),
                                                                       step_out(GRID[gid][e["arm"]][0], "lower")), e
            out += [("X", e["arm"], e["lr"], s) for s in c["sel"]]''',
'''    out = [("A", a, lr, s) for a in ARMS for lr in GRID[gid][a] for s in c["sel"]]
    lastx = {}
    for e in sorted(decisions(gid)["log"], key=lambda e: e["seq"]):
        if e["stage"] == "X":
            g_ = GRID[gid][e["arm"]]
            if e["arm"] not in lastx:
                assert e["lr"] not in g_ and e["lr"] in (step_out(g_[1], "upper"), step_out(g_[0], "lower")), e
            else:   # T7: the second extension continues the first one's direction by one ladder step
                prev = lastx[e["arm"]]
                assert e["lr"] == step_out(prev, "upper" if float(prev) > float(g_[1]) else "lower"), (e, prev)
            lastx[e["arm"]] = e["lr"]
            out += [("X", e["arm"], e["lr"], s) for s in c["sel"]]'''),
('''COST_MAX = {"b200": 180.0, "other": 300.0}   # Amendment T2 (coordinator 2026-10-26): stop before exceeding''',
'''COST_MAX = {"b200": 180.0, "other": 310.0}   # Amendment T2 (coordinator 2026-10-26): stop before exceeding; T7 (the user, 2026-10-27): other 300 -> 310'''),
]
for a, b in rep:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
p.write_text(s)
print("ok")
