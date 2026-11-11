"""ladder-1b selection step (runs/ptlr_1b/AUDIT.md, 'Selection and stages'); called by maybe_select.sh.
  --check-complete   exit 0 iff all 12 stage-A runs (none / win-ctx x 3 lrs x seeds 0-1) have a complete run JSON
  --pick <json>      read ptscale_summary.py --json output: if both 1B arms have a best lr (status ok) and neither is an
                     F87 open bracket (edge pick beating its neighbour by > 2), write picks.json (status final) and exit 0;
                     else write the reason to select.log, touch select.done (no automatic stage B), exit 1.
                     An edge pick within 2 points is reported 'OPEN (edge pick)' in select.log; stage B still proceeds."""
import json
import os
import sys
import time
from pathlib import Path

R = Path(os.environ.get("PTLR1B_DIR", "/scratch/yh6210/llmadd_torch/runs/ptlr_1b"))   # override: tests only
GRID = {"none": ("2.4e-3", "4.8e-3", "9.6e-3"), "win-ctx": ("3e-4", "6e-4", "1.2e-3")}
TARGET = {"none": "none", "win-ctx": "helix3"}


def log(msg):
    with open(R / "select.log", "a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}\n")
    print(msg)


def complete(f):
    try:
        r = json.loads(Path(f).read_text())
        return r["curve"][-1]["step"] >= r["steps"] - 1
    except Exception:
        return False


if sys.argv[1] == "--check-complete":
    miss = [f"{a}@{lr} s{s}" for a in GRID for lr in GRID[a] for s in (0, 1)
            if not complete(R / f"out/1b-lr{lr}-7pct-{a}/run_whole_{TARGET[a]}_seed{s}_1b-lr{lr}-7pct-{a}.json")]
    print(f"stage A: {12 - len(miss)}/12 complete" + (f"; missing {', '.join(miss)}" if miss else ""))
    sys.exit(1 if miss else 0)

assert sys.argv[1] == "--pick", sys.argv
sel = json.loads(Path(sys.argv[2]).read_text())
picks, why = {}, []
for a in ("none", "win-ctx"):
    e = sel.get(f"1B|{a}")
    if e is None or e["status"] != "ok":
        why.append(f"{a}: {'missing' if e is None else e['status']}" + (f" ({e['reasons'][0]})" if e and e.get("reasons") else ""))
        continue
    assert tuple(e["grid"]) == GRID[a], (a, e["grid"])
    b = e["best"]
    if e["open"]:
        why.append(f"{a}: OPEN bracket at the {e['open']} edge (best {b} beats its neighbour by > 2): extending the grid "
                   f"needs the user")
        continue
    if b in (GRID[a][0], GRID[a][-1]):
        log(f"{a}: OPEN (edge pick): best {b} is at the {'lower' if b == GRID[a][0] else 'upper'} edge of {GRID[a]} within 2 "
            f"points of its neighbour (F87 rule: not an open bracket); reported, stage B proceeds")
    picks[a] = b
if why:
    log("NO PICKS (no automatic stage B): " + "; ".join(why))
    (R / "select.done").write_text("no picks\n")
    sys.exit(1)
(R / "picks.json").write_text(json.dumps({"status": "final", **picks, "selection_json": sys.argv[2],
                                          "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
log(f"picks: none@{picks['none']}, win-ctx@{picks['win-ctx']} -> stage B (seeds 2-3)"
    + (" and C (seeds 4-7, APPROVE_C present)" if (R / "APPROVE_C").exists() else "; stage C waits for APPROVE_C"))
