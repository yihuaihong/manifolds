"""ptdom2: the frozen manifest of the re-used ptdom runs (read only): per cell (R1's selection cells at seeds 0-3, Part S's
true-target arm only-<d>@2e-3 at seeds 0-7) its run dir, the sha256[:16] of its run JSON and check.json, validity and area of
the domain(s) it serves are NOT stored (no result is written into the pre-registration). --write -> runs/ptdom2/reused_ptdom.json
(written once, chmod 444); without --write: compare with the written file."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/scratch/yh6210/llmadd_torch/runs/ptdom2")
sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
import readout_ptdom2 as RO2
L = RO2.load2(RO2.R0 / "out", "ptdom")
cells, probs = {}, []
for k in RO2.reused_cells():
    r = L.get(k)
    if r is None or not r["valid"]:
        probs.append(f"{k}: missing or invalid in ptdom")
        continue
    d = Path(r["dir"])
    cells["|".join(map(str, k))] = {"dir": str(d), "run_sha": RO2.file_sha(RO2.run_json2(d, "ptdom", *k)), "check_sha": RO2.file_sha(d / "check.json")}
print(f"{len(cells)} re-used ptdom cells; problems: {probs or 'none'}")
if "--write" in sys.argv:
    assert not RO2.MANIFEST.exists() and not probs
    RO2.MANIFEST.write_text(json.dumps({"design": "ptdom2", "what": "the re-used ptdom cells (read only)", "cells": cells}, indent=1))
    RO2.MANIFEST.chmod(0o444)
    print("written", RO2.MANIFEST, RO2.file_sha(RO2.MANIFEST))
else:
    old = json.loads(RO2.MANIFEST.read_text())["cells"]
    print("MANIFEST", "AGREES" if old == cells else "DIFFERS")
