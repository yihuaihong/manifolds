"""Amendment T6 item 2 (coordinator): every group whose none arm has its current C at the UPPER edge of its stage-A grid
and no X yet gets REOPEN + X none@<one ladder step up> (seeds = the group's selection seeds), as T2 / T5 did by hand.
Found by the rule over every group (expected: tpp7-190m-t0.318, tpp7-294m-t0.318; 294M@0.745 is STOP and left to the
user). Each group's entries are written under its lock (owner pid:<host>:<pid>), atomically, and checked through the
generator's decisions() / effective() / runs_of(); the cost guard must hold after all of them. Structure only is printed
(no basis, no metric: blind). The X jobs are then submitted by the hook (bash maybe_select.sh <group>).
    python t6_decide.py [--dry]"""
import datetime
import json
import os
import socket
import sys

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

DRY = "--dry" in sys.argv
assert not (T.R / "HOLD").exists(), "HOLD is set"
todo = []
for gid in T.GROUPS:
    eff = T.effective(gid)["none"]
    lo, hi = T.GRID[gid]["none"]
    if eff["C"] == hi and eff["X"] is None and not eff["STOP"]:
        todo.append((gid, hi, T.step_out(hi, "upper")))
    print(f"{gid}: none {eff}; grid {lo} / {hi}" + ("  -> T6: REOPEN + X " + T.step_out(hi, "upper") if todo and todo[-1][0] == gid else ""))
assert [g for g, _, _ in todo] == ["tpp7-190m-t0.318", "tpp7-294m-t0.318"], todo   # what the coordinator and the check found
ph0 = T.planned_hours()
if DRY:
    print("dry run: nothing written", todo)
    sys.exit(0)
for gid, hi, x in todo:
    L = T.R / "locks" / gid
    L.mkdir()   # fails if a selection holds it: rerun later
    try:
        (L / "job").write_text(f"pid:{socket.gethostname()}:{os.getpid()}\n")
        f = T.R / "decisions" / f"{gid}.json"
        d = json.loads(f.read_text())
        eff = T.effective(gid)
        assert eff["none"] == {"X": None, "C": hi, "STOP": False}, eff
        now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        n = len(d["log"])
        d["log"] += [{"seq": n, "arm": "none", "stage": "REOPEN", "amend": "T6", "time": now,
                      "why": "Amendment T6 (coordinator): none's pick at the upper grid edge is always extended one step up; "
                             "the C at " + hi + " is withdrawn until F87 rule 1 is applied again"},
                     {"seq": n + 1, "arm": "none", "stage": "X", "lr": x, "amend": "T6", "time": now,
                      "why": "Amendment T6: one ladder step above the upper edge, the selection seeds; then rule 1 over the three "
                             "lrs; open again at either edge -> STOP (the user)"}]
        tmp = f.with_name(f"{f.name}.t6.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(f)
        eff2 = T.effective(gid)
        assert eff2["none"] == {"X": x, "C": None, "STOP": False} and eff2["win-ctx"] == eff["win-ctx"], eff2
        print("written", gid, [(e["seq"], e["arm"], e["stage"], e.get("lr"), e.get("amend")) for e in d["log"]])
        print("   X runs:", [r for r in T.runs_of(gid) if r[0] == "X"])
    finally:
        (L / "job").unlink(missing_ok=True)
        L.rmdir()
ph1 = T.planned_hours()
assert all(ph1[k] <= T.COST_MAX[k] for k in ph1), (ph1, T.COST_MAX)
print("planned GPU-h before", {k: round(v, 2) for k, v in ph0.items()}, "after", {k: round(v, 2) for k, v in ph1.items()}, "COST_MAX", T.COST_MAX)
