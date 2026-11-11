"""Structural check of a finished ft999 GPU run (format only): curve steps, states file, grad_log, eval keys."""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1]); arm, s = d.name.rsplit("-s", 1)
r = json.loads((d / f"run_{arm}_seed{s}.json").read_text())
steps = [c["step"] for c in r["curve"]]
assert steps == list(range(0, 4351, 50)), (steps[:3], steps[-3:], len(steps))
assert r["steps"] == 4380 and r["lam"] == 20.0 and r["add3"]["n_test"] == 2000 and r["add3"]["test_sha"] == "420909b9ca714352"
print("curve points", len(steps), "steps", r["steps"], "grad_log steps", [g["step"] for g in r["grad_log"]])
z = d / f"states_{arm}_seed{s}.npz"
if z.exists():
    f = np.load(z)
    print("states", f["states"].shape, f["states"].dtype, "steps", f["steps"][:3], "...", f["steps"][-1], "layer", int(f["layer"]),
          "numbers", f["numbers"][0], "..", f["numbers"][-1], "finite", bool(np.isfinite(f["states"].astype(np.float32)).all()))
    v = np.load("/scratch/yh6210/llmadd_torch/runs/viz-9B-n7000-helix/out/states_helix_seed0.npz")
    print("viz format", {k: (v[k].shape, v[k].dtype) for k in v.files}, "ours", {k: (f[k].shape, f[k].dtype) for k in f.files})
print("STRUCTURE OK")
