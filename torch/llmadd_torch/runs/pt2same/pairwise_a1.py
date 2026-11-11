"""pt2same Amendment A1 item 1 (pre-registered, descriptive): the three pairwise contrasts of the 8-seed paired gains,
Delta505 - Delta53, Delta190 - Delta53, Delta505 - Delta190, with 90% seed-bootstrap CIs (seeds resampled within each
size, arms paired), and the OLS slope weights. Uses readout_pt2same's own loader (imported, unchanged)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_pt2same as RO  # noqa: E402

data = RO.load(RO.ROOTS)
D = {}
for z in RO.SIZES:
    cn, cp = data[("torch", z, "none")], data[("torch", z, "win-ctx")]
    assert all(st == "ok" for st in cn["status"] + cp["status"]), (z, cn["status"], cp["status"])
    D[z] = np.array(cp["area"]) - np.array(cn["area"])
x = np.log([RO.N_NONEMB[z] for z in RO.SIZES])
w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
print("OLS slope weights:", ", ".join(f"{z} {wi:+.3f}" for z, wi in zip(RO.SIZES, w)))
rng = np.random.default_rng(0)
for a, b in (("505m", "53m"), ("190m", "53m"), ("505m", "190m")):
    est = D[a].mean() - D[b].mean()
    boot = [D[a][rng.integers(0, 8, 8)].mean() - D[b][rng.integers(0, 8, 8)].mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [5, 95])
    h = [D[a][:4].mean() - D[b][:4].mean(), D[a][4:].mean() - D[b][4:].mean()]
    print(f"Delta{a[:-1]} - Delta{b[:-1]}: {est:+.2f} [90% {lo:+.2f}, {hi:+.2f}]  (seeds 0-3 {h[0]:+.2f}, 4-7 {h[1]:+.2f})")
