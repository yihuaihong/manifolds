"""Read-only: do the lattice priors (dim, torus, box5) share representation? Uses logged per-eval cka_<d> and gcos_<a>|<b> in pt6v2 run JSONs."""
import glob
import json
from pathlib import Path

import numpy as np

OUT = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2/out")
SEEDS = range(16, 24)
W0, W1 = 916, 7323
ARMS = [("none", "2e-3"), ("only-numbers", "2e-3"), ("only-dim", "2e-3"), ("only-cities", "2e-3"), ("only-countries", "2e-3"),
        ("only-torus", "2e-3"), ("only-box5", "2e-3"), ("all6", "2e-3"), ("all6", "1e-3"), ("none", "1.6e-2"), ("none", "3.2e-2")]
DOMS = ("numbers", "dim", "cities", "countries", "torus", "box5")
CK = {"numbers": "cka_ops_helix3", "dim": "cka_dim", "cities": "cka_cities", "countries": "cka_countries", "torus": "cka_torus", "box5": "cka_box5"}


def load(arm, lr, s):
    f = glob.glob(str(OUT / f"pt6v2-{arm}-lr{lr}-s{s}" / "*" / "run_whole_*.json"))
    assert len(f) == 1, (arm, lr, s, f)
    return json.loads(Path(f[0]).read_text())["curve"]


C = {(a, l, s): load(a, l, s) for a, l in ARMS for s in SEEDS}
steps = np.array([e["step"] for e in C[("all6", "2e-3", 16)]])
iw = int(np.searchsorted(steps, W1))   # first eval at/after window close
ins = (steps >= W0) & (steps <= W1)
print("window-close eval step:", int(steps[iw]))


def ser(a, l, key):
    return np.array([[np.nan if e.get(key) is None else float(e[key]) for e in C[(a, l, s)]] for s in SEEDS])


print("\n[1] CKA of each domain's representation to its own target geometry (mean over 8 seeds): at window close -> final3")
hdr = "arm".ljust(22) + "".join(d.rjust(18) for d in DOMS)
print(hdr)
for a, l in ARMS:
    row = f"{a}@{l}".ljust(22)
    for d in DOMS:
        x = ser(a, l, CK[d])
        row += f"{np.nanmean(x[:, iw]):8.3f}->{np.nanmean(x[:, -3:]):6.3f}  "
    print(row)

print("\n[2] cross-domain CKA (representation transfer without that domain's prior): only-X@2e-3 minus none@2e-3, at window close / final3; wins of 8")
for a in ("only-dim", "only-torus", "only-box5", "only-numbers", "only-cities", "only-countries"):
    out = []
    for d in DOMS:
        if a == f"only-{d}":
            continue
        x, n = ser(a, "2e-3", CK[d]), ser("none", "2e-3", CK[d])
        dw, df = x[:, iw] - n[:, iw], x[:, -3:].mean(1) - n[:, -3:].mean(1)
        out.append(f"{d} {np.nanmean(dw):+.3f}({int((dw > 0).sum())})/{np.nanmean(df):+.3f}({int((df > 0).sum())})")
    print(f"  {a:15s}", " | ".join(out))

print("\n[3] gradient cosine between priors, all6@2e-3 and all6@1e-3, mean over in-window evals and seeds (n evals with a value)")
keys = sorted(k for k in C[("all6", "2e-3", 16)][iw - 2] if k.startswith("gcos_"))
for l in ("2e-3", "1e-3"):
    print(f"  all6@{l}:")
    for k in keys:
        x = ser("all6", l, k)
        v = x[:, ins]
        print(f"    {k:24s} {np.nanmean(v):+.3f}  (seed means {np.round(np.nanmean(v, 1), 3).tolist()}; n {int(np.isfinite(v).sum())})")
print("\n  gcos logged outside the window? all6@2e-3 after-window non-null count:",
      {k: int(np.isfinite(ser('all6', '2e-3', k)[:, steps > W1]).sum()) for k in keys[:3]})
print("  gcos in only-dim@2e-3 in-window (diagnostic gradients of priors that are off?):",
      {k: round(float(np.nanmean(ser('only-dim', '2e-3', k)[:, ins])), 3) for k in keys if 'dim' in k})
