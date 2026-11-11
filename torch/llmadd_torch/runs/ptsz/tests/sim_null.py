"""Pre-registration simulation (AUDIT.md "False-SHOWN rates", written before any ptsz run; pt300x's build audit did the same
for F99): how often do the registered PRIMARY (8 L40S points, (a)(b)(c), (c) at df 6) and SECONDARY (the three never-seen
sizes, (a)(b)) read 'SHOWN (increase)' when the gain does NOT grow with size?
The decision functions are readout_ptsz's own (slope_block with its fixed bootstrap draws, verdict); halves by OLS (= the
read-out's point estimates). Inputs from the real reused runs (readout_ptsz.reused_cell, the KA-checked cells):
  seed scatter: the 32 within-cell centred per-seed gains of the four reused L40S cells (53M, 93M, 190M, 505M), x sqrt(8/7)
                (undo the centring), drawn with replacement, 8 per simulated cell;
  size scatter tau (extra, beyond the seed noise of an 8-seed mean): tau6 from the OLS residuals of F99's six points (df 4),
                tau4 from the four L40S points (df 2), also 0 and 3.41 (the pt300x build audit's value: the 4-point
                residual SD of F97 FIXSLOPE).
Scenarios (no growth = one constant gain level mu0 for every size):
  conditional null: the four reused L40S cells fixed at their real per-seed gains (they are known); the four new cells
                    simulated at mu0 = the mean of the four L40S cell means (26.85) [and mu0 = F99's six-cell mean, 28.84];
  unconditional null: all eight cells simulated at mu0 (the rule's own false-positive rate);
  secondary: the three never-seen cells of the same draws (fixed cells play no part);
  alternatives (power, for context): the new cells on F99's six-point line (+2.787 / e-fold) or on the L40S-only line (+2.161).
    python tests/sim_null.py [N_SIM]       (default 2000 per scenario; numpy default_rng(12345))"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import readout_ptsz as RO   # noqa: E402

ROOT = Path("/scratch/yh6210/llmadd_torch/runs")
NSIM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
G = RO.G
X = {c: math.log(G.SIZES[RO.SIZE[c]][5]) for c in RO.L40S8 + RO.B200_2}


def main():
    t0 = time.time()
    cells = {c: RO.finish(RO.reused_cell(ROOT, c)) for c in RO.REUSED}
    assert all(c["ok"] for c in cells.values())
    obs4 = [cells[c] for c in RO.L40S4]
    dev = np.concatenate([c["d"] - c["d"].mean() for c in obs4]) * math.sqrt(8 / 7)
    seedvar_mean = float(np.mean([np.var(c["d"], ddof=1) / 8 for c in obs4]))

    def tau_from(cs, df):
        x = np.array([c["x"] for c in cs]); y = np.array([c["d"].mean() for c in cs])
        b, a = np.polyfit(x, y, 1)
        s2 = float(np.sum((y - a - b * x) ** 2) / df)
        sv = float(np.mean([np.var(c["d"], ddof=1) / 8 for c in cs]))
        return math.sqrt(max(0.0, s2 - sv)), math.sqrt(s2), sv, (a, b)
    tau6, s6, sv6, line6 = tau_from([cells[c] for c in RO.F99_6], 4)
    tau4, s4, sv4, line4 = tau_from(obs4, 2)
    mu4 = float(np.mean([c["d"].mean() for c in obs4]))
    mu6 = float(np.mean([cells[c]["d"].mean() for c in RO.F99_6]))
    print(f"# inputs: seed deviations n {len(dev)} (SD {dev.std():.2f}; SD of an 8-seed mean {math.sqrt(seedvar_mean):.2f}); "
          f"F99 six-point residual SD {s6:.2f} -> tau6 {tau6:.2f}; L40S four-point residual SD {s4:.2f} -> tau4 {tau4:.2f}; "
          f"mu0 {mu4:.2f} (four L40S means) / {mu6:.2f} (F99's six); lines: six {line6[1]:+.3f}, L40S four {line4[1]:+.3f}")
    rng = np.random.default_rng(12345)
    new4 = list(G.REGISTERED)

    def sim(cid, mean):
        return dict(cid=cid, size=RO.SIZE[cid], hw="L40S", x=X[cid], d=mean + rng.choice(dev, 8))

    def decide(cs8):
        main = RO.slope_block(cs8)
        h1 = dict(b=float(RO.ols([c["x"] for c in cs8], [c["d"][:4].mean() for c in cs8])))
        h2 = dict(b=float(RO.ols([c["x"] for c in cs8], [c["d"][4:].mean() for c in cs8])))
        v, cr = RO.verdict(main, h1, h2)
        s3 = [c for c in cs8 if c["cid"] in RO.NEW3]
        m3 = RO.slope_block(s3)
        g1 = dict(b=float(RO.ols([c["x"] for c in s3], [c["d"][:4].mean() for c in s3])))
        g2 = dict(b=float(RO.ols([c["x"] for c in s3], [c["d"][4:].mean() for c in s3])))
        v2, cr2 = RO.verdict(m3, g1, g2, use_c=False)
        return v, v2, main["b"], m3["b"], cr

    rows = []
    taus = (("0", 0.0), ("tau6", tau6), ("tau4", tau4), ("3.41", 3.41))
    scen = [("conditional null, mu0 = L40S-4 mean", "cond", lambda c: mu4),
            ("conditional null, mu0 = F99-6 mean", "cond", lambda c: mu6),
            ("unconditional null (all 8 simulated)", "uncond", lambda c: mu4),
            ("alternative: new cells on F99's six-point line", "cond", lambda c: line6[0] + line6[1] * X[c]),
            ("alternative: new cells on the L40S-only line", "cond", lambda c: line4[0] + line4[1] * X[c])]
    for name, mode, mean_of in scen:
        for tn, tau in taus:
            P = {"SHOWN (increase)": 0, "SHOWN (decrease)": 0}
            P2 = {"SHOWN (increase)": 0, "SHOWN (decrease)": 0}
            fails = {"a": 0, "b": 0, "c": 0}
            bs = []
            for _ in range(NSIM):
                cs = []
                for cid in RO.L40S8:
                    if mode == "cond" and cid in RO.L40S4:
                        cs.append(cells[cid])
                    else:
                        cs.append(sim(cid, mean_of(cid) + tau * rng.standard_normal()))
                v, v2, b, b3, cr = decide(cs)
                bs.append(b)
                for k in P:
                    P[k] += v.startswith(k)
                    P2[k] += v2.startswith(k)
                for k in fails:
                    fails[k] += not cr[k]
            r = dict(scenario=name, tau=tn, tau_value=round(tau, 3), n=NSIM,
                     primary_shown_inc=P["SHOWN (increase)"] / NSIM, primary_shown_dec=P["SHOWN (decrease)"] / NSIM,
                     secondary_shown_inc=P2["SHOWN (increase)"] / NSIM, secondary_shown_dec=P2["SHOWN (decrease)"] / NSIM,
                     fail_a=fails["a"] / NSIM, fail_b=fails["b"] / NSIM, fail_c=fails["c"] / NSIM,
                     b_median=float(np.median(bs)))
            rows.append(r)
            print(f"{name:48s} tau {tn:5s} ({tau:4.2f}): PRIMARY SHOWN inc {r['primary_shown_inc']:.3f} dec "
                  f"{r['primary_shown_dec']:.3f} [fail a {r['fail_a']:.2f} b {r['fail_b']:.2f} c {r['fail_c']:.2f}; median b "
                  f"{r['b_median']:+.2f}] | SECONDARY SHOWN inc {r['secondary_shown_inc']:.3f} dec {r['secondary_shown_dec']:.3f}",
                  flush=True)
    (HERE / "sim_null.json").write_text(json.dumps({"n_sim": NSIM, "tau6": tau6, "tau4": tau4, "mu4": mu4, "mu6": mu6,
                                                    "line6": line6, "line4": line4, "rows": rows}, indent=1))
    print(f"# {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
