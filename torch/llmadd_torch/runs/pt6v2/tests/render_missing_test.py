#!/usr/bin/env python3
"""MISSING-key test for the two FINDINGS renderers: delete keys from a read-out JSON, render, and check that each deleted key prints
'MISSING <key>' and that the renderer does not crash. Usage: render_missing_test.py pt6v2 IN.json | partR IN.json"""
import sys

sys.dont_write_bytecode = True

import copy                    # noqa: E402
import importlib.util          # noqa: E402
import json                    # noqa: E402
from pathlib import Path       # noqa: E402

B = Path("/scratch/yh6210/llmadd_torch/runs")
kind, src = sys.argv[1], Path(sys.argv[2])
J = json.loads(src.read_text())


def load(p):
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fails = []
if kind == "pt6v2":
    R = load(B / "pt6v2/render_findings_pt6v2.py")
    J2 = copy.deepcopy(J)
    del J2["m"]["dim"]["ci95_g"]
    del J2["m"]["cities"]["levels"]["none@N"]
    del J2["m"]["countries"]["final3"]
    del J2["mean6_gain"]
    del J2["devices"]
    del J2["arms"]["all6@1e-3"]["lm_share"]
    md = R.render(J2, "mutated")
    want = ["MISSING m.dim.ci95_g", "MISSING m.cities.levels.none@N", "MISSING m.countries.final3", "MISSING mean6_gain",
            "MISSING devices", "MISSING arms.all6@1e-3.lm_share"]
else:
    R = load(B / "ptdom2/render_findings_partR.py")
    J2 = copy.deepcopy(J)
    x = J2["R"]["rows"]["landmarks"]
    del x["rule_gain"]["boot90"]
    del x["rule_geo"]["wins"]
    del x["floor"]["means"]
    del x["lm_share"]
    del x["split"]
    del x["tier"]
    md = R.render(J2, "mutated")
    want = ["MISSING R.rows.landmarks.rule_gain.boot90", "MISSING R.rows.landmarks.rule_geo.wins", "MISSING R.rows.landmarks.floor.means",
            "MISSING R.rows.landmarks.lm_share", "MISSING R.rows.landmarks.split", "MISSING R.rows.landmarks.tier"]
for w in want:
    if w not in md:
        fails.append(w)
out = src.with_name(src.stem + "_missing_test.md")
out.write_text(md)
print(f"{kind}: {len(want) - len(fails)} of {len(want)} deleted keys print MISSING" + (f"; NOT FOUND: {fails}" if fails else "") + f" ({out})")
sys.exit(1 if fails else 0)
