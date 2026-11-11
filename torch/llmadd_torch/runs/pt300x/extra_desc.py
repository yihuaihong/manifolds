"""pt300x AUDIT 'Submission part 2' promised descriptive outputs (coordinator's own path, double-checking the results
audit): per prior cell at its pick, the LM share of the gradient over on-window evals (pre-clip norms; share by norm
a / (a + b)), the share of on-window evals with prior > LM, the clip coefficient derived as in the pttpp read-out; and the
area gain without bucket b11 per cell plus its OLS slope on ln N_nonemb (6 sizes). Never changes the verdict."""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "/scratch/yh6210/llmadd_torch/runs/pttpp/readout/tools")
import readout_rules as RR  # noqa: E402

R = "/scratch/yh6210/llmadd_torch/runs"
CELLS = {  # size: (N_nonemb, none glob, prior glob)
    "53M": (14165376, f"{R}/ptlr/out/53m-lr8e-3-7pct-none/run_*seed*.json", f"{R}/ptlr/out/53m-lr2e-3-7pct-win-ctx/run_*seed*.json"),
    "93M": (41964032, f"{R}/pt300x/out/p3x-93m-lr2.4e-3-none/run_*seed*.json", f"{R}/pt300x/out/p3x-93m-lr1.2e-3-win-ctx/run_*seed*.json"),
    "190M": (113283840, f"{R}/ptlr/out/190m-lr4.8e-3-7pct-none/run_*seed*.json", f"{R}/ptlr/out/190m-lr1.2e-3-7pct-win-ctx/run_*seed*.json"),
    "294M": (210500160, f"{R}/pt300x/out/p3x-294m-lr9.6e-3-none/run_*seed*.json", f"{R}/pt300x/out/p3x-294m-lr6e-4-win-ctx/run_*seed*.json"),
    "505M": (402752512, f"{R}/ptlr/out/500m-lr9.6e-3-7pct-none/run_*seed*.json", f"{R}/ptlr/out/500m-lr1.2e-3-7pct-win-ctx/run_*seed*.json"),
    "1.07B": (943904000, f"{R}/ptlr_1b/out/1b-lr2.4e-3-7pct-none/run_*seed*.json", f"{R}/ptlr_1b/out/1b-lr6e-4-7pct-win-ctx/run_*seed*.json"),
}
NO_B11 = RR.B5   # without b11 (readout_rules.B5)


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        if f.endswith(".claim"):
            continue
        r = json.load(open(f))
        s = r["args"]["seed"]
        assert s not in out, (pat, s)
        out[s] = r
    assert sorted(out) == list(range(8)), (pat, sorted(out))
    return out


xs, g6, gnb = [], [], []
print("size   LM share (norm) med [min seed]  prior>LM evals  clip med (min, clipped %)  gain 6-bucket  gain no-b11")
for z, (nne, pn, pp) in CELLS.items():
    n, p = load(pn), load(pp)
    sh, gt, cl = [], [], []
    for s in range(8):
        on = [e for e in p[s]["curve"] if e.get("prior_on") and e.get("grad_lm") is not None]
        a = np.array([e["grad_lm"] for e in on]); b = np.array([e["grad_prior"] for e in on]); c = np.array([e["grad_cos"] for e in on])
        sh.append(np.median(a / (a + b))); gt.append(np.mean(b > a))
        tot = np.sqrt(a ** 2 + b ** 2 + 2 * a * b * c); cl.append(np.minimum(1.0, 1.0 / (tot + 1e-6)))
    clc = np.concatenate(cl)
    d6 = np.mean([RR.area(p[s]) - RR.area(n[s]) for s in range(8)])
    dnb = np.mean([RR.area(p[s], NO_B11) - RR.area(n[s], NO_B11) for s in range(8)])
    xs.append(np.log(nne)); g6.append(d6); gnb.append(dnb)
    print(f"{z:6s} {np.median(sh):.2f} [{min(sh):.2f}]                  {100 * np.mean(gt):4.0f}%          "
          f"{np.median(clc):.2f} ({clc.min():.2f}, {100 * np.mean(clc < 1):.0f}%)              {d6:+6.1f}        {dnb:+6.1f}")
print(f"OLS slope on ln N_nonemb: 6-bucket {np.polyfit(xs, g6, 1)[0]:+.3f} (read-out +2.787), no-b11 {np.polyfit(xs, gnb, 1)[0]:+.3f}")
