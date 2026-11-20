"""Descriptive partial means for Part R2 over the seeds that exist (no verdict): prior pick vs none pick, vs shuffled,
and same-lr 2e-3 / prior-lr gains, area and final3. Read-only."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/scratch/yh6210/llmadd_torch/runs/ptdom2")
import readout_ptdom2 as R2

R_ = R2.load_all()
picks = json.loads(Path("/scratch/yh6210/llmadd_torch/runs/ptdom2/picks_R_after2.json").read_text())["domains"]
SEEDS = list(range(8, 16))


def vals(b, arm, lr, d, field):
    out = {}
    for s in SEEDS:
        r = R_.get((b, arm, lr, s))
        if r and r.get("valid") and r.get(field, {}).get(d) is not None:
            out[s] = r[field][d]
    return out


def gain(b, a1, l1, a2, l2, d, field="area"):
    x, y = vals(b, a1, l1, d, field), vals(b, a2, l2, d, field)
    ss = sorted(set(x) & set(y))
    if not ss:
        return "n 0"
    g = np.array([x[s] - y[s] for s in ss])
    return f"mean {g.mean():+.2f} wins {int((g > 0).sum())}/{len(g)} seeds {ss} per-seed {[round(v, 1) for v in g]}"


for d, b in R2.R_DOMS.items():
    v = picks[d]
    if v["status"] != "FINAL":
        print(d, v["status"]); continue
    P, N = v["pick_prior"], v["pick_none"]
    print(f"== {d} (batch {b}) prior@{P} none@{N}")
    for f in ("area", "final3"):
        print(f"  {f:6s} prior-none     ", gain(b, f"only-{d}", P, "none", N, d, f))
        print(f"  {f:6s} prior-shuf     ", gain(b, f"only-{d}", P, f"only-{d}-shuf", P, d, f))
    for lr in sorted({P, "2e-3", N}, key=float):
        print(f"  area   same-lr {lr:6s}", gain(b, f"only-{d}", lr, "none", lr, d))
    nl = sorted({k[2] for k in R_ if k[:2] == (b, "none")}, key=float)
    print("  none means by lr (seeds 8-15 present):", {lr: (round(np.mean(list(vals(b, 'none', lr, d, 'area').values())), 1), len(vals(b, 'none', lr, d, 'area'))) for lr in nl if vals(b, 'none', lr, d, 'area')})
    pl = sorted({k[2] for k in R_ if k[:2] == (b, f"only-{d}")}, key=float)
    print("  prior means by lr (seeds 8-15 present):", {lr: (round(np.mean(list(vals(b, f'only-{d}', lr, d, 'area').values())), 1), len(vals(b, f'only-{d}', lr, d, 'area'))) for lr in pl if vals(b, f'only-{d}', lr, d, 'area')})
