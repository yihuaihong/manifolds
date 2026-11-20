#!/usr/bin/env python3
"""Write sample pt6v2 read-out JSONs from readout_pt6v2's own planted worlds (test input for render_findings_pt6v2.py).

No pt6v2 run exists yet and --known / --selftest write no JSON, so this driver imports readout_pt6v2 READ ONLY, builds planted
worlds with its world() / readout_from() and dumps res exactly as readout_pt6v2.main does (json.dumps(res, indent=1, default=str)).
The numbers are planted, not results. Usage: sample_json_pt6v2.py OUT_DIR
"""
import sys

sys.dont_write_bytecode = True

import json                     # noqa: E402
from pathlib import Path        # noqa: E402

sys.path.insert(0, "/scratch/yh6210/llmadd_torch/runs/pt6v2")
import readout_pt6v2 as R       # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
mani = R.MJ2.manifolds(R.MJ2.comp_doms(R.MJ2.PLACEHOLDER))
m0, m1, m2, m3, m4, m5 = mani
E_, GN = R.E_, R.GN


def rich(L):
    """mod6 of the self-test (N_m = 4e-3 / 2e-3, a NOT SHOWN) + a top-lr pick, a FLOOR (route pc) and a CEILING."""
    L["none", "4e-3"][m1] = [62 + e for e in E_]
    L["none", "2e-3"][m2] = [64 + e for e in E_]
    L["all6", R.P_LR][m3] = [47 + e + 0.5 * g for e, g in zip(E_, GN)]
    L["none", "3.2e-2"][m4] = [70 + e for e in E_]
    for k in list(L):
        if k[0] == "none" or k == ("all6", R.P_LR):
            L[k][m0] = [3.0 + 0.1 * e for e in E_]
    L["none", "1.6e-2"][m5] = [96 + 0.1 * e for e in E_]
    L["all6", R.P_LR][m5] = [97 + 0.1 * e + 0.2 * g for e, g in zip(E_, GN)]


worlds = {
    "default": dict(w=R.world(mani), kw={}),
    "rich": dict(w=R.world(mani, rich, dev={20: "l40s", 21: "l40s"}), kw={}),
    "partial": dict(w=R.world(mani, rich, drop={("all6", R.SAME_LR, 18), ("none", "8e-3", 23)}), kw={"allow_partial": True}),
}
for name, x in worlds.items():
    res, txt = R.readout_from(x["w"], mani, "planted (sample_json_pt6v2.py)", **x["kw"])
    (out / f"sample_pt6v2_{name}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (out / f"sample_pt6v2_{name}.txt").write_text(txt + "\n")
    print(name, "->", out / f"sample_pt6v2_{name}.json", "| HEADLINE:", res.get("headline", {}).get("text"))
