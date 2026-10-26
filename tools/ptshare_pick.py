"""ptshare: the pre-registered expansion / confirm trigger (report/audits/ptshare_AUDIT.md "Selection, expansion, confirm
seeds"), applied to the JSON of tools/ptshare_summary.py --json (its "select" block: per share k and arm the status of rule
1 on seeds 0-3). Round n (1 = after the 72 main jobs, 2 = after round 1's expansion jobs):
  final       best lr with a closed bracket   -> confirm seeds 4-7 at the best lr (k = 2% / 4% / 14%; 7% has them in ptlr)
  open        open bracket, not yet expanded  -> round 1: one factor-2 step outward (ptshare_design.EXPAND), seeds 0-3;
                                                 round 2: STOP (an expansion happens once, after round 1 only)
  stop        open again after the expansion  -> STOP: the user's call (nothing submitted for this arm)
  incomplete / withheld / missing             -> nothing (the reason is printed)
Writes <picks.json> {"round", "cells": [{k, arm, lr, seeds, why}], "stops", "waiting"} only when there is a cell to submit;
always prints the decision. Exit 0.
    python ptshare_pick.py --round <1|2> <selection.json> <picks.json>"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ptshare_design as D   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--round", type=int, required=True, choices=(1, 2))
ap.add_argument("selection")
ap.add_argument("picks")
A = ap.parse_args()
sel = json.loads(Path(A.selection).read_text())["select"]
cells, stops, waiting = [], [], []
for k in D.NEW:
    for arm in D.ARMS:
        s = sel.get(str(k), {}).get(arm)
        lab = f"{D.share_label(k)} {arm}"
        if s is None:
            waiting.append(f"{lab}: absent from the selection")
        elif s["status"] == "final":
            cells.append({"k": k, "arm": arm, "lr": s["best"], "seeds": list(D.CONFIRM), "why": "confirm"})
        elif s["status"] == "open" and A.round == 1 and not s["expanded"]:
            cells.append({"k": k, "arm": arm, "lr": D.EXPAND[arm][s["open"]], "seeds": list(D.MAIN), "why": "expand"})
        elif s["status"] in ("open", "stop"):
            stops.append(f"{lab}: best {s['best']} with the bracket OPEN at the {s['open']} edge of {s['grid']}"
                         + (" after the one-step expansion" if s["expanded"] else " (round 2 does not expand)")
                         + ": STOP, the next lr and the confirm seeds are the user's call")
        else:
            waiting.append(f"{lab}: {s['status']} ({s['why']})")
res = {"round": A.round, "cells": cells, "stops": stops, "waiting": waiting}
for c in cells:
    print(f"round {A.round}: {c['why']:7s} {D.share_label(c['k'])} {c['arm']}@{c['lr']} seeds {c['seeds']}")
for x in stops:
    print(f"round {A.round}: STOP {x}")
for x in waiting:
    print(f"round {A.round}: nothing for {x}")
if cells:
    Path(A.picks).write_text(json.dumps(res, indent=1))
print(json.dumps({"round": A.round, "n_cells": len(cells), "n_stops": len(stops), "n_waiting": len(waiting)}))
