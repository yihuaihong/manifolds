"""Checks on the CPU smoke outputs of ft999 (tiny random OLMo-2, 4 steps): file format, curves, grad A/B, lambda A/B."""
import json
from pathlib import Path

import numpy as np

S = Path("/scratch/yh6210/llmadd_torch/runs/ft999_dev/smoke")
runs, states = {}, {}
for name in ("9B-none-s0", "9B-helix3-s0", "9B-helix3_shuf-s0", "lam0-helix3-s0"):
    arm = name.split("-")[1]
    d = S / name
    r = json.loads((d / f"run_{arm}_seed0.json").read_text())
    z = np.load(d / f"states_{arm}_seed0.npz")
    runs[name], states[name] = r, z
    assert z["states"].shape == (3, 1000, 64) and z["states"].dtype == np.float16, z["states"].shape
    assert list(z["steps"]) == [0, 2, 4] and int(z["layer"]) == 16 and np.array_equal(z["numbers"], np.arange(1000))
    assert [c["step"] for c in r["curve"]] == [0, 2, 4], r["curve"]
    assert all({"test", "test_lt1000", "test_ge1000"} <= set(c) for c in r["curve"])
    assert {"test", "test_lt1000", "test_ge1000", "train", "train_lt1000", "train_ge1000"} <= set(r["eval"])
    assert r["steps"] == 4 and r["add3"]["lo3"] == 0 and r["add3"]["n_test"] == 200
    assert len(r["preds"]["test"]["hit"]) == 200
    g = r["grad_log"]
    assert [x["step"] for x in g] == [0, 2], g
    print(f"{name:18s} lam {r['lam']:4.0f} geo_loss(ep0) {r['train_log'][0]['geo_loss']:.4f} ce {r['train_log'][0]['ce']:.5f} "
          f"grad_log step0 {g[0]}")
    if arm != "none":
        assert all(x["grad_geo"] > 0 for x in g), g
    for k in ("cka_helix3_all1000", "cka_helix3_shuf_all1000"):
        assert k in r["geometry"]["L16"], r["geometry"]["L16"].keys()
# all runs share one test set
assert len({r["add3"]["test_sha"] for r in runs.values()}) == 1
# lambda A/B: lam 0 reproduces none exactly; lam 20 does not
s_none, s_l0, s_h, s_sh = (states[k]["states"] for k in ("9B-none-s0", "lam0-helix3-s0", "9B-helix3-s0", "9B-helix3_shuf-s0"))
assert np.array_equal(s_none[0], s_h[0]) and np.array_equal(s_none[0], s_sh[0]), "step-0 states differ between arms"
d_l0 = float(np.abs(s_none.astype(np.float32) - s_l0.astype(np.float32)).max())
d_h = float(np.abs(s_none[1:].astype(np.float32) - s_h[1:].astype(np.float32)).max())
d_sh = float(np.abs(s_h[1:].astype(np.float32) - s_sh[1:].astype(np.float32)).max())
ce_eq = runs["9B-none-s0"]["train_log"][0]["ce"] == runs["lam0-helix3-s0"]["train_log"][0]["ce"]
print(f"lambda A/B: max|states none - helix3@lam0| = {d_l0:.3g} (want 0), ce equal {ce_eq}; "
      f"max|states none - helix3@lam20| = {d_h:.3g} (want > 0); max|helix3 - helix3_shuf| = {d_sh:.3g} (want > 0)")
assert d_l0 == 0.0 and ce_eq and d_h > 0 and d_sh > 0
# the prior moves the geometry in its direction: CKA to helix3 over 0..999 at L16 after 4 steps
for k in ("9B-none-s0", "9B-helix3-s0", "9B-helix3_shuf-s0", "lam0-helix3-s0"):
    gg = runs[k]["geometry"]["L16"]
    print(f"{k:18s} final L16 CKA helix3 {gg['cka_helix3_all1000']:.4f} helix3_shuf {gg['cka_helix3_shuf_all1000']:.4f}")
print("SMOKE CHECKS OK")
