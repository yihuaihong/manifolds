"""Amendment B2 selection step (runs/ptlr_1b/AUDIT.md "Amendment B2"); called by maybe_select_b2.sh.
  --check-complete   exit 0 iff the 12 stage-A runs and the 2 B2 runs (none@1.2e-3, seeds 0-1) have a complete run JSON;
                     the count goes to select_b2.log
  --pick <json>      read ptscale_summary.py --json output (F87 rule 1 re-applied by the read-out to none {1.2e-3, 2.4e-3,
                     4.8e-3, 9.6e-3}, seeds 0-1):
                     none: status ok and exactly that grid; an open bracket (lower: 1.2e-3 beats 2.4e-3 by > 2) -> NO
                     PICKS, stop (the user decides); otherwise the read-out's best (2.4e-3 -> closed; 1.2e-3 within 2 by
                     the bpb tie-break -> an edge pick within 2 points, not open by F87 rule 1 -> closed).
                     win-ctx: status ok, grid {3e-4, 6e-4, 1.2e-3}, not open, best 6e-4 (readout_A's pick; else stop).
                     -> picks_b2.json (status final) and exit 0, or the reason in select_b2.log + select_b2.done, exit 1."""
import json
import os
import sys
import time
from pathlib import Path

R = Path(os.environ.get("PTLR1B_DIR", "/scratch/yh6210/llmadd_torch/runs/ptlr_1b"))   # override: tests only
GRID = {"none": ("1.2e-3", "2.4e-3", "4.8e-3", "9.6e-3"), "win-ctx": ("3e-4", "6e-4", "1.2e-3")}
STAGE_A = {"none": ("2.4e-3", "4.8e-3", "9.6e-3"), "win-ctx": ("3e-4", "6e-4", "1.2e-3")}
TARGET = {"none": "none", "win-ctx": "helix3"}
WIN_PICK = "6e-4"


def log(msg):
    with open(R / "select_b2.log", "a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}\n")
    print(msg)


def complete(f):
    try:
        r = json.loads(Path(f).read_text())
        return r["curve"][-1]["step"] >= r["steps"] - 1
    except Exception:
        return False


if sys.argv[1] == "--check-complete":
    runs = [(a, lr, s) for a in STAGE_A for lr in STAGE_A[a] for s in (0, 1)] + [("none", "1.2e-3", 0), ("none", "1.2e-3", 1)]
    miss = [f"{a}@{lr} s{s}" for a, lr, s in runs
            if not complete(R / f"out/1b-lr{lr}-7pct-{a}/run_whole_{TARGET[a]}_seed{s}_1b-lr{lr}-7pct-{a}.json")]
    log(f"B2 check: {len(runs) - len(miss)}/{len(runs)} complete" + (f"; missing {', '.join(miss)}" if miss else ""))
    sys.exit(1 if miss else 0)

assert sys.argv[1] == "--pick", sys.argv
sel = json.loads(Path(sys.argv[2]).read_text())
picks, why = {}, []
for a in ("none", "win-ctx"):
    e = sel.get(f"1B|{a}")
    if e is None or e["status"] != "ok":
        why.append(f"{a}: {'missing' if e is None else e['status']}" + (f" ({e['reasons'][0]})" if e and e.get("reasons") else ""))
        continue
    if tuple(e["grid"]) != GRID[a]:
        why.append(f"{a}: grid {e['grid']} is not the pre-registered {list(GRID[a])}")
        continue
    b = e["best"]
    if e["open"]:
        why.append(f"{a}: OPEN bracket at the {e['open']} edge (best {b} beats its neighbour by > 2): no further automatic "
                   f"step, the user decides")
        continue
    if a == "win-ctx" and b != WIN_PICK:
        why.append(f"win-ctx: best {b}, not readout_A's {WIN_PICK} (its data did not change): check before any stage B")
        continue
    if b in (GRID[a][0], GRID[a][-1]):
        log(f"{a}: edge pick within 2 points (not open by F87 rule 1): best {b} at the "
            f"{'lower' if b == GRID[a][0] else 'upper'} edge of {list(GRID[a])}; bracket closed, stage B proceeds")
    picks[a] = b
if why:
    log("NO PICKS (no stage B): " + "; ".join(why))
    (R / "select_b2.done").write_text("no picks\n")
    sys.exit(1)
(R / "picks_b2.json").write_text(json.dumps({"status": "final", **picks, "selection_json": sys.argv[2],
                                             "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
log(f"B2 picks: none@{picks['none']}, win-ctx@{picks['win-ctx']} -> stage B (seeds 2-3, 520 only)"
    + ("; APPROVE_C present: stage C is `make_jobs_ptlr1b_b2.py --stage C --submit`" if (R / "APPROVE_C").exists()
       else "; stage C waits for APPROVE_C"))
