"""Amendment T7 item 2 (the user's approval, 2026-10-27, relayed by the coordinator): 294M@0.745 none, STOP after its X
1.92e-2, gets one more step: REOPEN (withdraws the STOP; amend T7) + X none@3.84e-2 (seeds 0-3), under the group's lock
(owner pid:<host>:<pid>), atomically; checked through the T7 generator's decisions() / effective() / runs_of() and the
cost guard (COST_MAX other 310). Prints structure only (no basis, no metric: blind). The X jobs are then submitted by the
hook (bash maybe_select.sh tpp7-294m-t0.745). Afterwards the unchanged selection applies F87 rule 1 over the four lrs;
open again -> STOP (no automatic further step).
    python t7_decide.py [--dry]"""
import datetime
import json
import os
import socket
import sys

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

gid, x = "tpp7-294m-t0.745", "3.84e-2"
assert T.COST_MAX == {"b200": 180.0, "other": 310.0}, ("deploy the T7 generator first", T.COST_MAX)
assert not (T.R / "HOLD").exists(), "HOLD is set"
eff = T.effective(gid)
assert eff["none"] == {"X": "1.92e-2", "C": None, "STOP": True}, eff
assert T.GRID[gid]["none"] == ("4.8e-3", "9.6e-3") and T.step_out("1.92e-2", "upper") == x
ph0 = T.planned_hours()
print("before:", [(e["seq"], e["arm"], e["stage"], e.get("lr"), e.get("amend")) for e in T.decisions(gid)["log"]], {k: round(v, 2) for k, v in ph0.items()})
if "--dry" in sys.argv:
    sys.exit(0)
L = T.R / "locks" / gid
L.mkdir()   # fails if a selection holds it: rerun later
try:
    (L / "job").write_text(f"pid:{socket.gethostname()}:{os.getpid()}\n")
    f = T.R / "decisions" / f"{gid}.json"
    d = json.loads(f.read_text())
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    n = len(d["log"])
    d["log"] += [{"seq": n, "arm": "none", "stage": "REOPEN", "amend": "T7", "time": now,
                  "why": "Amendment T7: the user approved one more step above the X (2026-10-27); the STOP is withdrawn"},
                 {"seq": n + 1, "arm": "none", "stage": "X", "lr": x, "amend": "T7", "time": now,
                  "why": "Amendment T7: one ladder step above the first X, seeds 0-3; then F87 rule 1 over the four lrs; "
                         "open again -> STOP (the user), no automatic further step"}]
    tmp = f.with_name(f"{f.name}.t7.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(f)
    eff2 = T.effective(gid)
    assert eff2["none"] == {"X": x, "C": None, "STOP": False} and eff2["win-ctx"] == eff["win-ctx"], eff2
    xs = [r for r in T.runs_of(gid) if r[0] == "X" and r[1] == "none"]
    assert [r[2] for r in xs] == ["1.92e-2"] * 4 + [x] * 4, xs
    print("written", [(e["seq"], e["arm"], e["stage"], e.get("lr"), e.get("amend")) for e in d["log"]])
finally:
    (L / "job").unlink(missing_ok=True)
    L.rmdir()
ph1 = T.planned_hours()
assert all(ph1[k] <= T.COST_MAX[k] for k in ph1), (ph1, T.COST_MAX)
print("planned GPU-h before", {k: round(v, 2) for k, v in ph0.items()}, "after", {k: round(v, 2) for k, v in ph1.items()}, "COST_MAX", T.COST_MAX)
