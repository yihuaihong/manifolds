"""Amendment T5 item 2(a) (coordinator, after the fourth audit's F3): 93M@0.318 none gets the same one-step extension as
93M@0.745 none (T2 item 3): append REOPEN (none's C at 3.84e-2) + X none@7.68e-2 (one ladder step above the grid's upper
edge, seeds 0-3) to decisions/tpp7-93m-t0.318.json, under the group's lock (owner pid:<host>:<pid>, the T4 format of a
selection by hand), atomically; validated by the generator's decisions() / effective() / runs_of(); the cost guard
(planned_hours vs COST_MAX) is printed before and after and must hold. Prints structure only (seq / arm / stage / lr):
no basis, no metric (the batch is blind). The X jobs are then submitted by the unchanged hook (bash maybe_select.sh
tpp7-93m-t0.318), which also applies the cost and disk guards and --test-only."""
import datetime
import json
import os
import socket
import sys

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

gid = "tpp7-93m-t0.318"
L = T.R / "locks" / gid
L.parent.mkdir(exist_ok=True)
assert not (T.R / "HOLD").exists(), "HOLD is set: remove it first (the hook must submit the X jobs right after)"
L.mkdir()   # fails if a selection holds it: rerun later
try:
    (L / "job").write_text(f"pid:{socket.gethostname()}:{os.getpid()}\n")
    f = T.R / "decisions" / f"{gid}.json"
    d = json.loads(f.read_text())
    eff = T.effective(gid)
    assert T.GRID[gid]["none"] == ("1.92e-2", "3.84e-2"), T.GRID[gid]
    assert eff["none"] == {"X": None, "C": "3.84e-2", "STOP": False}, eff
    assert T.step_out(T.GRID[gid]["none"][1], "upper") == "7.68e-2" and "7.68e-2" in T.LADDER
    ph0 = T.planned_hours()
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    n = len(d["log"])
    d["log"] += [{"seq": n, "arm": "none", "stage": "REOPEN", "amend": "T5", "time": now,
                  "why": "Amendment T5 item 2(a) (coordinator, fourth audit F3): the one-step extension that 93M@0.745 none got in "
                         "T2 item 3; the C at the upper grid edge 3.84e-2 is withdrawn until F87 rule 1 is applied again"},
                 {"seq": n + 1, "arm": "none", "stage": "X", "lr": "7.68e-2", "amend": "T5", "time": now,
                  "why": "Amendment T5 item 2(a): one ladder step above the upper edge, seeds 0-3; then rule 1 over "
                         "{1.92e-2, 3.84e-2, 7.68e-2}; open again at either edge -> STOP (the user)"}]
    tmp = f.with_name(f"{f.name}.t5.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(f)
    eff = T.effective(gid)
    assert eff["none"] == {"X": "7.68e-2", "C": None, "STOP": False} and eff["win-ctx"]["C"] == "9.6e-3" and not eff["win-ctx"]["STOP"], eff
    ph1 = T.planned_hours()
    assert all(ph1[k] <= T.COST_MAX[k] for k in ph1), (ph1, T.COST_MAX)
    print("written", [(e["seq"], e["arm"], e["stage"], e.get("lr"), e.get("amend")) for e in d["log"]])
    print("X runs:", [r for r in T.runs_of(gid) if r[0] == "X"])
    print("planned GPU-h before", {k: round(v, 2) for k, v in ph0.items()}, "after", {k: round(v, 2) for k, v in ph1.items()},
          "COST_MAX", T.COST_MAX)
finally:
    (L / "job").unlink(missing_ok=True)
    L.rmdir()
