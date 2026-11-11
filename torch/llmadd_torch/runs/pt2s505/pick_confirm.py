"""pt2s505: the pre-registered confirm-seed trigger (runs/pt2s505/AUDIT.md "Confirm seeds"), applied to the JSON that
tools/pt2main_summary.py --gen make_jobs_pt2s505.py --res runs/pt2s505/out --json <selection.json> writes (pt2main rule 1's
best lr per arm, the arms without one and why, the open brackets).
  status 'final'     both none and win-ctx have a best lr (complete, not WITHHELD) and neither bracket is open
                     -> picks_confirm.json {"status": "final", "block": "pt2s505", "best": {...}}: the confirm stage runs
  status 'not final' otherwise (the reason: incomplete / WITHHELD / OPEN at an edge) -> nothing is submitted
    python pick_confirm.py <selection.json> <picks_confirm.json>   (exit 0 when final, 1 otherwise)"""
import json
import sys
from pathlib import Path

sel, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
b = sel.get("pt2s505")
why = []
if b is None:
    why.append("the read-out has no pt2s505 block")
else:
    for arm in ("none", "win-ctx"):
        if arm not in b["best"]:
            why.append(f"{arm}: no best lr ({b['nobest'].get(arm, 'absent')})")
        elif arm in b["open"]:
            why.append(f"{arm}: best lr {b['best'][arm]} but the bracket is OPEN at the {b['open'][arm]} edge (rule 1: NOT FINAL; "
                       "the next lr beyond it is the conditional cell, the user's call)")
res = {"block": "pt2s505", "status": "final" if not why else "not final", "best": (b or {}).get("best", {}),
       "open": (b or {}).get("open", {}), "nobest": (b or {}).get("nobest", {}), "why": why}
if not why:
    out.write_text(json.dumps(res, indent=1))
print(json.dumps(res))
sys.exit(0 if not why else 1)
