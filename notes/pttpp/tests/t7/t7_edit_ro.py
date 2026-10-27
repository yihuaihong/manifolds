from pathlib import Path
p = Path("/scratch/yh6210/llmadd-burst-tpp/tools/pttpp_summary.py")
s = p.read_text()
rep = [
('''    """{(arm, stage): entry} of the decision log folded in seq order: X / C / STOP set, REOPEN (Amendment T2) withdraws the
    arm's current C (a later C replaces it)"""
    f = D.Rd / "decisions" / f"{gid}.json"
    log = sorted(json.loads(f.read_text())["log"], key=lambda e: e["seq"]) if f.exists() else []
    out = {}
    for e in log:
        if e["stage"] == "REOPEN":
            out.pop((e["arm"], "C"), None)
        else:
            out[(e["arm"], e["stage"])] = e
    return out''',
'''    """{(arm, stage): entry} of the decision log folded in seq order: X / C / STOP set, REOPEN (Amendment T2) withdraws the
    arm's current C (a later C replaces it); T7: a REOPEN after a STOP (the user's approval) withdraws the STOP; (arm, "X")
    is the arm's last X entry, x_lrs() lists them all"""
    out = {}
    for e in decision_log(D, gid):
        if e["stage"] == "REOPEN":
            out.pop((e["arm"], "C"), None)
            out.pop((e["arm"], "STOP"), None)
        else:
            out[(e["arm"], e["stage"])] = e
    return out


def decision_log(D, gid):
    f = D.Rd / "decisions" / f"{gid}.json"
    return sorted(json.loads(f.read_text())["log"], key=lambda e: e["seq"]) if f.exists() else []


def x_lrs(D, gid, arm):
    """every X lr of the arm (T7: a user-approved second extension adds a second one)"""
    return {e["lr"] for e in decision_log(D, gid) if e["arm"] == arm and e["stage"] == "X"}'''),
('''    grid (stage A + the X lr) on the group's selection seeds -> dict(status, best, open, ...)"""''',
'''    grid (stage A + the X lrs) on the group's selection seeds -> dict(status, best, open, ...)"""'''),
('''    lrs = sorted(set(G.GRID[gid][arm]) | ({dec[(arm, "X")]["lr"]} if (arm, "X") in dec else set()), key=float)''',
'''    lrs = sorted(set(G.GRID[gid][arm]) | x_lrs(D, gid, arm), key=float)'''),
]
for a, b in rep:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
p.write_text(s)
print("ok")
