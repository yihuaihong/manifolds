"""Amendment T2 item 3: append REOPEN (none's C) + X none@3.84e-2 to decisions/tpp7-93m-t0.745.json, under the group's lock,
atomically; validated by the generator's decisions() / effective()."""
import datetime, json, os, sys
from pathlib import Path
sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T
gid = "tpp7-93m-t0.745"
L = T.R / "locks" / gid
L.parent.mkdir(exist_ok=True)
L.mkdir()   # fails if a selection holds it: rerun later
try:
    (L / "job").write_text(os.environ.get("SLURM_JOB_ID", "none") + "\n")
    f = T.R / "decisions" / f"{gid}.json"
    d = json.loads(f.read_text())
    eff = T.effective(gid)
    assert eff["none"] == {"X": None, "C": "1.92e-2", "STOP": False}, eff
    assert T.step_out(T.GRID[gid]["none"][1], "upper") == "3.84e-2"
    now = datetime.datetime.now().isoformat(timespec="seconds")
    n = len(d["log"])
    d["log"] += [{"seq": n, "arm": "none", "stage": "REOPEN", "amend": "T2", "time": now,
                  "why": "Amendment T2 item 3: both none lrs at the floor (3.44 / 3.81), the edge pick 1.92e-2 came from the bpb tie-break; one step up first"},
                 {"seq": n + 1, "arm": "none", "stage": "X", "lr": "3.84e-2", "amend": "T2", "time": now,
                  "why": "Amendment T2 item 3: the pre-registered extension above the edge, seeds 0-3"}]
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(f)
    eff = T.effective(gid)
    assert eff["none"] == {"X": "3.84e-2", "C": None, "STOP": False} and eff["win-ctx"]["C"] == "2.4e-3", eff
    print("written", [(e["seq"], e["arm"], e["stage"], e.get("lr")) for e in d["log"]])
    print("X runs:", [r for r in T.runs_of(gid) if r[0] == "X"])
finally:
    (L / "job").unlink(missing_ok=True); L.rmdir()
