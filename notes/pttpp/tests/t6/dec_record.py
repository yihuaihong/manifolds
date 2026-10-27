"""T5: a structural copy of decisions/tpp7-93m-t0.318.json for git (seq / arm / stage / lr / amend / time / job / why):
the basis of the automatic entries (selection-seed means) is left out, so the record carries no metric."""
import json
import sys
from pathlib import Path

g = sys.argv[2] if len(sys.argv) > 2 else "tpp7-93m-t0.318"
d = json.loads(Path(f"/scratch/yh6210/llmadd_torch/runs/pttpp/decisions/{g}.json").read_text())
keep = ("seq", "arm", "stage", "lr", "amend", "time", "job", "why")
out = {"group": g, "note": "structure only (basis omitted: blind); the full log is runs/pttpp/decisions/" + g + ".json",
       "log": [{k: e[k] for k in keep if k in e} for e in d["log"]]}
for e in out["log"]:
    if not e.get("amend"):
        e.pop("why", None)   # an automatic entry's 'why' may quote the basis
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
print([(e["seq"], e["arm"], e["stage"], e.get("lr"), e.get("amend")) for e in out["log"]])
