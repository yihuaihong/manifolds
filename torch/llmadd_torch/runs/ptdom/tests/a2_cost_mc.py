"""Amendment A2: the expected cost of the A2 extension rule (F87 rule 1's 2-point margin for every unit; none's area argmax
at the LOWER edge of its grid triggers its one extension without the margin; STOP after X by the margin only), simulated
with the REAL selection functions (readout_ptdom.decide -> select_unit -> readout_rules.best_lr 'top2' / bracket and
extend_rule) on synthetic stage-A / X runs, as the A1 audit's stop_mc.py did for the A1 rule.

Per iteration: every stage-A cell x seeds 0-3 (92 runs) gets, per domain, area = response(lr) + N(0, (2 sigma)^2) per seed
(so the 4-seed mean has sd sigma; independent across domains, arms and seeds) and one bpb per run (1.2 + N(0, 0.003^2);
the tie-break); decide(stage A) -> the X cells get seeds 0-3 from the same response -> decide(AX) -> the C cells and STOPs.
The auxiliary-loss gate passes everywhere (a constant passing gradient record). Responses (x = log2(lr / centre); prior
centre 1e-3, none centre 4e-3 = the grid centres): flat; a peak of -c x^2 points; 'confound': none rising by 1 point per
halving toward low lr (the case the none exception exists for) with a weak prior peak.
Cost = 92 + 4 |X| + 4 |C| runs (C = the distinct confirm cells of A and AX) x 1.3 L40S-h.

  python runs/ptdom/tests/a2_cost_mc.py [N] [--rule A1]     (default 2000 iterations per scenario; --rule A1: the superseded
                                                             A1 rule, tests/a1_rule_ref.py, on the same responses)
"""
import math
import sys
from pathlib import Path

import numpy as np

R = Path("/scratch/yh6210/llmadd_torch/runs/ptdom")
sys.path.insert(0, str(R))
import readout_ptdom as RO   # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--") and a != "A1"]
N = int(args[0]) if args else 2000
RULE = "A1" if "--rule" in sys.argv and sys.argv[sys.argv.index("--rule") + 1] == "A1" else "A2"
if RULE == "A1":
    sys.path.insert(0, str(R / "tests"))
    import a1_rule_ref as DEC   # noqa: E402
else:
    DEC = RO
PLAN_H = 1.3
GRAD_OK = {"n_on": 26, "n_ok": 26, "n_missing": 0, "missing": [], "share": 0.95, "share_min": 0.95, "n_b_gt_a": 0,
           "n_clip_lt1": 0, "dp_maxdiff": 0.0, "lm_share_T": 0.95, "prior_over_lm": 0.05, "clip_coef": 1.0, "last_step": 18309}
SCEN = [("flat, sigma 0.5", 0.5, ("peak", 0.0), ("peak", 0.0)),
        ("flat, sigma 1.5", 1.5, ("peak", 0.0), ("peak", 0.0)),
        ("weak peak 0.25 pt/step^2, sigma 1", 1.0, ("peak", 0.25), ("peak", 0.25)),
        ("peak 0.5 pt/step^2, sigma 1", 1.0, ("peak", 0.5), ("peak", 0.5)),
        ("confound: none +1 pt per halving toward low lr, prior weak peak, sigma 1", 1.0, ("peak", 0.25), ("slope", 1.0))]


def resp(kind, lr, centre):
    x = math.log2(float(lr) / centre)
    return -kind[1] * x * x if kind[0] == "peak" else -kind[1] * x


def one(rng, sigma, kp, kn):
    R_ = {}

    def add(arm, lr, seeds=RO.SEL):
        kind, centre = (kn, 4e-3) if arm == "none" else (kp, 1e-3)
        mu = 50.0 + resp(kind, lr, centre)
        for s in seeds:
            bp = 1.2 + rng.normal(0, 0.003)
            R_[(arm, lr, s)] = {"arm": arm, "lr": lr, "seed": s, "valid": True, "finished": True,
                                "area": {d: mu + rng.normal(0, 2 * sigma) for d in RO.SIX}, "bpb_off": bp, "bpb_intr": bp,
                                "grad": GRAD_OK}
    for arm in RO.ARMS:
        for lr in RO.GRID[arm]:
            add(arm, lr)
    pa = DEC.decide(R_)
    for arm, lr in pa["X"]:
        add(arm, lr)
    pax = DEC.decide(R_, pa) if pa["X"] else None
    fin = pax or pa
    nC = len(pa["C"]) + (len(pax["C"]) if pax else 0)
    return {"X_units": {k for k, u in pa["units"].items() if u["decision"] == "X"}, "STOP": set(fin["STOP"]),
            "nX": len(pa["X"]), "nC": nC, "runs": 92 + 4 * len(pa["X"]) + 4 * nC,
            "nf_dom": {d for d in RO.SIX if f"only-{d}|{d}" in fin["STOP"] or f"none|{d}" in fin["STOP"]}}


def main():
    print(f"{RULE} extension rule, expected cost ({'readout_ptdom' if RULE == 'A2' else 'tests/a1_rule_ref'}.decide(); {N} iterations per scenario; 1.3 L40S-h per run)")
    print("| scenario | prior unit P(X) | prior unit P(STOP) | none unit P(X) | none unit P(STOP) | P(a domain NOT FINAL) | "
          "E[X runs] | E[C runs] | E[runs] | E[L40S-h] | p95 L40S-h | max L40S-h |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, sigma, kp, kn in SCEN:
        rng = np.random.default_rng(20261104)
        res = [one(rng, sigma, kp, kn) for _ in range(N)]
        pu = [f"only-{d}|{d}" for d in RO.SIX]
        nu = [f"none|{d}" for d in RO.SIX]
        pX = np.mean([np.mean([k in r["X_units"] for k in pu]) for r in res])
        pS = np.mean([np.mean([k in r["STOP"] for k in pu]) for r in res])
        nX = np.mean([np.mean([k in r["X_units"] for k in nu]) for r in res])
        nS = np.mean([np.mean([k in r["STOP"] for k in nu]) for r in res])
        nf = np.mean([len(r["nf_dom"]) / 6 for r in res])
        runs = np.array([r["runs"] for r in res])
        h = runs * PLAN_H
        print(f"| {name} | {pX:.3f} | {pS:.3f} | {nX:.3f} | {nS:.3f} | {nf:.3f} | {4 * np.mean([r['nX'] for r in res]):.1f} | "
              f"{4 * np.mean([r['nC'] for r in res]):.1f} | {runs.mean():.1f} | {h.mean():.1f} | {np.percentile(h, 95):.1f} | {h.max():.1f} |")
    print("A1 rule for comparison (the A1 audit's stop_mc.py, audit_a1.md): flat sigma 0.5: prior P(X) .66 / P(STOP) .16, none .41 / "
          ".07, NOT FINAL ~.22; flat sigma 1.5: .67 / .19, .43 / .09, ~.25; weak peak sigma 1: .61 / .06, .17 / 0, ~.06")
    print(f"{RULE} COST MC done")


if __name__ == "__main__":
    main()
