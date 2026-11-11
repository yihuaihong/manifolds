"""pt2sz AUDIT item (before any pt2sz run): the false-SHOWN rate of the PRIMARY rule (a)(b)(c) under a no-growth null, with
the seed and size scatter of the existing 2% Torch cells (F98's 53M / 190M / 505M, real run JSONs via readout_pt2sz's own
loader), as pt300x's tests/audit_build.md did for its 6-point rule. Also the rate of F98's own rule (joint bootstrap + halves)
on five points, and, for reference, the SHOWN rate when the two new points sit on the old line (not a null).
Scatter estimates (from the 3 old cells, 8 seeds each):
  sigma   pooled within-size SD of the per-seed gains (df 21); rho = mean cross-size correlation of the seed residuals
  tau     size-level SD beyond seed noise: tau_flat from the spread of the 3 means around their mean (df 2), tau_line from
          their residuals around the 3-point OLS line (df 1); each = sqrt(max(0, s^2 - sigma^2 / 8))
Scenarios (gain = the same true mean at every size = no growth; per-seed noise N(0, sigma^2) with the cross-size seed
correlation rho, or the observed seed residuals resampled):
  U   unconditional: all five cells drawn (true mean 50.31 + u_z, u_z ~ N(0, tau^2)): the rule's operating characteristic
  C   conditional on the three observed old cells (their real per-seed gains fixed): the two new cells drawn at a flat level
      (50.31 = the old cells' mean; 47.99 = 53M's; 52.98 = 505M's) + u_z: the realistic chance of a SHOWN with no growth
      beyond what the old cells already contain
  L   reference (not a null): the new cells on the line through the old cells (+ u_z)
Every simulated read-out uses readout_pt2sz's own statistics (slope_block / verdict / f98_stats on synthetic cells; the
bootstrap draws are the read-out's fixed default_rng(0) ones), so the rates are those of the registered code.
    python sim_null.py [N_SIM=2000]"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/yh6210/llmadd_torch/runs/pt2sz")
import readout_pt2sz as RD   # noqa: E402

N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
ROOT = Path("/scratch/yh6210/llmadd_torch/runs")
old = {s: RD.finish(RD.reused_cell(ROOT, s)) for s in RD.OLD}
assert all(c["ok"] for c in old.values()), [c["problems"] for c in old.values()]
D = {s: c["d"] for s, c in old.items()}
x = {s: math.log(RD.TP.SIZES[s][5]) for s in RD.SIZES5}
res = np.stack([D[s] - D[s].mean() for s in RD.OLD])          # (3, 8) seed residuals
sigma = math.sqrt((res ** 2).sum() / (3 * 7))
C = np.corrcoef(res)
rho = float(np.mean([C[i, j] for i in range(3) for j in range(3) if i < j]))
m3 = np.array([D[s].mean() for s in RD.OLD])
x3 = np.array([x[s] for s in RD.OLD])
mu = float(m3.mean())
b3 = float(RD.ols(x3, m3))
a3 = mu - b3 * x3.mean()
s2_flat = float(((m3 - mu) ** 2).sum() / 2)
s2_line = float(((m3 - (a3 + b3 * x3)) ** 2).sum() / 1)
tau_flat = math.sqrt(max(0.0, s2_flat - sigma ** 2 / 8))
tau_line = math.sqrt(max(0.0, s2_line - sigma ** 2 / 8))
print(f"old cells: mean gains {', '.join(f'{RD.LABEL[s]} {D[s].mean():+.2f}' for s in RD.OLD)}; mean {mu:.2f}; 3-point slope {b3:+.3f}")
print(f"seed scatter: pooled SD of the per-seed gains sigma {sigma:.2f} (per size {', '.join(f'{RD.LABEL[s]} {D[s].std(ddof=1):.2f}' for s in RD.OLD)}); "
      f"mean cross-size seed correlation rho {rho:+.2f} ({', '.join(f'{C[i, j]:+.2f}' for i in range(3) for j in range(3) if i < j)})")
print(f"size scatter beyond seed noise: tau_flat {tau_flat:.2f} (spread of the 3 means around their mean, df 2), "
      f"tau_line {tau_line:.2f} (around the 3-point line, df 1); seed SE of a mean sigma/sqrt(8) = {sigma / math.sqrt(8):.2f}")


def cell(size, d):
    return {"size": size, "x": x[size], "d": np.asarray(d, float), "ok": True}


def verdicts(cells):
    main = RD.slope_block(cells)
    h1, h2 = RD.slope_block(cells, seeds=RD.H1), RD.slope_block(cells, seeds=RD.H2)
    v, crit = RD.verdict(main, h1, h2)
    f = RD.f98_stats(cells)
    return v.startswith("SHOWN (increase)"), v.startswith("SHOWN"), crit, f["rule"].startswith("SHOWN (+"), main["b"]


def noise(rng, n_cells, mode):
    if mode == "normal":
        cov = sigma ** 2 * (rho * np.ones((n_cells, n_cells)) + (1 - rho) * np.eye(n_cells))
        return rng.multivariate_normal(np.zeros(n_cells), cov, size=8).T            # (cells, 8): seed s correlated across sizes
    flat = res.ravel()
    return rng.choice(flat, size=(n_cells, 8), replace=True) * math.sqrt(24 / 21)     # resampled, df-corrected


def run(name, make, n=N_SIM, seed=1):
    rng = np.random.default_rng(seed)
    inc = any_ = f98 = 0
    crit_fail = {"a": 0, "b": 0, "c": 0}
    bs = []
    for _ in range(n):
        cells = make(rng)
        i, a_, c, f, b = verdicts(cells)
        inc += i; any_ += a_; f98 += f; bs.append(b)
        for k, v in c.items():
            crit_fail[k] += not v
    se = lambda p: math.sqrt(p * (1 - p) / n)
    print(f"  {name:72s} SHOWN (increase) {inc / n:.3f} (+-{se(inc / n):.3f}); any SHOWN {any_ / n:.3f}; F98's rule SHOWN (+) "
          f"{f98 / n:.3f}; median b {np.median(bs):+.2f}; fail (a) {crit_fail['a'] / n:.2f} (b) {crit_fail['b'] / n:.2f} (c) {crit_fail['c'] / n:.2f}")
    return dict(inc=inc / n, any=any_ / n, f98=f98 / n, b_med=float(np.median(bs)), fail={k: v / n for k, v in crit_fail.items()})


out = {}
print(f"\n== {N_SIM} simulated read-outs per scenario (primary = (a) within-size seed bootstrap, (b) halves, (c) residual t df 3)")
for tau_name, tau in (("tau 0", 0.0), (f"tau_line {tau_line:.2f}", tau_line), (f"tau_flat {tau_flat:.2f}", tau_flat)):
    for mode in ("normal", "resampled"):
        def U(rng, tau=tau, mode=mode):
            e = noise(rng, 5, mode)
            u = rng.normal(0, tau, 5) if tau > 0 else np.zeros(5)
            return [cell(s, mu + u[k] + e[k]) for k, s in enumerate(RD.SIZES5)]
        out[f"U|{tau_name}|{mode}"] = run(f"U  unconditional null, {tau_name}, noise {mode}", U)
for level_name, level in (("old mean 50.31", mu), ("53M level 47.99", float(D["53m"].mean())), ("505M level 52.98", float(D["505m"].mean()))):
    for tau_name, tau in (("tau 0", 0.0), (f"tau_flat {tau_flat:.2f}", tau_flat)):
        for mode in ("normal", "resampled"):
            def Cn(rng, level=level, tau=tau, mode=mode):
                e = noise(rng, 2, mode)
                u = rng.normal(0, tau, 2) if tau > 0 else np.zeros(2)
                new = {s: level + u[k] + e[k] for k, s in enumerate(RD.NEW)}
                return [cell(s, D[s] if s in D else new[s]) for s in RD.SIZES5]
            out[f"C|{level_name}|{tau_name}|{mode}"] = run(f"C  conditional, new cells flat at the {level_name}, {tau_name}, {mode}", Cn)
for tau_name, tau in (("tau 0", 0.0), (f"tau_flat {tau_flat:.2f}", tau_flat)):
    def L(rng, tau=tau):
        e = noise(rng, 2, "normal")
        u = rng.normal(0, tau, 2) if tau > 0 else np.zeros(2)
        new = {s: a3 + b3 * x[s] + u[k] + e[k] for k, s in enumerate(RD.NEW)}
        return [cell(s, D[s] if s in D else new[s]) for s in RD.SIZES5]
    out[f"L|{tau_name}"] = run(f"L  reference (not a null): new cells on the old line, {tau_name}, normal", L)
# the grid of new-cell means (pt300x audit's crit_region): which (93M, 294M) mean gains read SHOWN given the old cells,
# with each new cell's seed residuals = the pooled residual shape of the old cells (fixed), no size scatter
grid = np.arange(40.0, 60.5, 1.0)
reg = np.zeros((len(grid), len(grid)), bool)
for i, g93 in enumerate(grid):
    for j, g294 in enumerate(grid):
        cells = [cell(s, D[s] if s in D else ({"93m": g93, "294m": g294}[s] + res[k % 3])) for k, s in enumerate(RD.SIZES5)]
        reg[i, j] = verdicts(cells)[0]
print(f"\n== grid: 93M and 294M 8-seed mean gains 40..60 (step 1; seed residuals 93M = 190M's, 294M = 53M's), old cells fixed: "
      f"SHOWN (increase) in {reg.sum()} of {reg.size} combinations")
print("  93M \\ 294M  " + "".join(f"{int(g) % 10}" for g in grid) + "   (294M 40 .. 60 left to right; S = SHOWN (increase))")
for i in range(len(grid)):
    print(f"  93M {grid[i]:4.0f}:   " + "".join("S" if reg[i, j] else "." for j in range(len(grid))))
j50 = list(grid).index(float(round(mu)))
flat = [float(grid[i]) for i in range(len(grid)) if reg[i, j50]]
print(f"  with 294M at {round(mu)} (the old mean): SHOWN for 93M in {[flat[0], flat[-1]] if flat else 'no value'}")
import json   # noqa: E402
Path(__file__).with_name("sim_null.json").write_text(json.dumps(
    dict(n_sim=N_SIM, sigma=sigma, rho=rho, tau_flat=tau_flat, tau_line=tau_line, mu=mu, b3=b3, scenarios=out,
         grid=dict(values=grid.tolist(), shown=reg.astype(int).tolist())), indent=1))
print("written", Path(__file__).with_name("sim_null.json"))
