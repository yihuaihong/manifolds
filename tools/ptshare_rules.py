"""The decision rules of the arithmetic-share dose read-out (tools/ptshare_summary.py; pre-registered in
report/audits/ptshare_AUDIT.md, rule 5 as amended by the user on 2026-10-26 before any ptshare run). Pure functions, unit-tested
by tools/ptshare_selftest.py:
  floor_ceiling(area_mean, final_median)   'floor' (mean area < 5 and median final held6 < 0.10) / 'ceiling' (mean area >= 90
                                           and median final held6 >= 0.95) / None, per arm of a share (8 seeds)
  uninformative(none, prior)               both arms at the floor or both at the ceiling -> the share leaves every slope
  primary(cen_t50)                         {k: none's t50-censored count over 8 seeds} -> 'dA' if the 2% (k 1) or 4% (k 2)
                                           share has >= 3 of 8, else 'lR50'; one metric for the whole slope
  ols(ks, ys) / slope_bound(w, bounds)     OLS slope of y on ln(k) and its weights; the direction in which bounded points
                                           (+1: true y >= y, -1: true y <= y) can move the true slope (0 none, None both ways)
  sign_status(entry, want)                 'same' / 'opposite' / 'undetermined' (no slope, zero, or a bound that lets the
                                           true slope cross zero)
  decide(p8, p47, s8, n_informative,       rule 5 -> {"label", "zh", "why"}; allow_bound=True is reading B (log R_0.5 for
         allow_bound)                      every share, a censored R used as the bound it is, conservatively)
Rule 5 has two readings, printed side by side until the user picks one (ptshare_design.RULE5_READING; the user's rule names
log R_0.5 for the shares >= 7% and Delta area for a censored 2% / 4% share, but one slope needs one metric):
  A  P = Delta area for the whole slope when the 2% or 4% share switches, else log R_0.5 (a bounded P: not determinable)
  B  P = log R_0.5 for every share; a censored none median at 2% / 4% makes R a lower bound there, so the slope an upper
     bound: with such a share only a negative claim ("the less arithmetic, the larger") can be shown, a positive one is "not
     shown"; S = Delta area
A slope entry is {"slope": float or None, "ci": [lo, hi], "bound": 0 / +1 / -1 / None, "push": {"1": f, "-1": f}, "ci_cons":
[lo, hi]}: push = the share of bootstrap resamples with a censored median that bounds that resample's slope from below (+1) /
above (-1); ci_cons = the percentiles with the resamples bounded from above set to -inf (lower end) and those bounded from
below set to +inf (upper end)."""
import math

import numpy as np

FLOOR_AREA, FLOOR_FINAL, CEIL_AREA, CEIL_FINAL = 5.0, 0.10, 90.0, 0.95
SWITCH_SHARES, SWITCH_CEN = (1, 2), 3


def floor_ceiling(area_mean, final_median):
    if area_mean < FLOOR_AREA and final_median < FLOOR_FINAL:
        return "floor"
    if area_mean >= CEIL_AREA and final_median >= CEIL_FINAL:
        return "ceiling"
    return None


def uninformative(none_fc, prior_fc):
    return none_fc is not None and none_fc == prior_fc


def primary(cen_t50):
    return "dA" if any(cen_t50.get(k, 0) >= SWITCH_CEN for k in SWITCH_SHARES) else "lR50"


def ols(ks, ys):
    x = np.log(np.asarray(ks, float))
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    return float(w @ np.asarray(ys, float)), w


def slope_bound(w, bounds):
    dirs = {int(np.sign(wi)) * b for wi, b in zip(w, bounds) if b}
    return 0 if not dirs else (dirs.pop() if len(dirs) == 1 else None)


def sign_status(s, want):
    if s is None or s.get("slope") is None or s.get("bound", 0) is None:
        return "undetermined"
    sg = int(np.sign(s["slope"]))
    if sg == 0 or s.get("bound", 0) not in (0, sg):
        return "undetermined"
    return "same" if sg == want else "opposite"


def fmt(s):
    if s is None or s.get("slope") is None:
        return "- (" + (s or {}).get("why", "no slope") + ")"
    mark = {0: "", 1: " (lower bound: true >= this)", -1: " (upper bound: true <= this)",
            None: " (bounds both ways: sign open)"}[s.get("bound", 0)]
    return (f"{s['slope']:+7.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] per ln(share), "
            f"{s['slope'] * math.log(2):+.3f} per doubling{mark}")


def decide(p8, p47, s8, n_informative, allow_bound=False):
    """rule 5: P's 8-seed slope CI excludes 0 (main test), P's confirm-only slope (guardrail) and S's 8-seed slope with the
    same sign -> shown; P's CI includes 0 -> not shown; guardrail or S opposite / undeterminable -> direction unstable.
    Censoring (both readings): the claim's side must also hold on the conservative CI 'ci_cons' of the slope entry, in which
    every bootstrap resample whose censored medians bound it towards the other side is set to -inf / +inf, so a censored
    median can only make the test stricter. A P slope that is itself a bound: reading A (allow_bound False) not
    determinable; reading B (allow_bound True) counts when the bound can only push it further from 0 on the claim's side"""
    if n_informative < 3:
        return {"label": "not shown", "zh": "未显示", "why": f"only {n_informative} informative shares (< 3): no slope"}
    if p8 is None or p8.get("slope") is None or p8.get("bound", 0) is None:
        return {"label": "not shown", "zh": "未显示", "why": f"P's 8-seed slope is not determinable: {fmt(p8).strip()}"}
    if p8["ci"][0] <= 0 <= p8["ci"][1]:
        return {"label": "not shown", "zh": "未显示", "why": f"P's 8-seed slope {fmt(p8).strip()}: the 90% CI includes 0"}
    side = -1 if p8["ci"][1] < 0 else 1
    cons = p8.get("ci_cons", p8["ci"])
    if not ((side < 0 and cons[1] < 0) or (side > 0 and cons[0] > 0)):
        return {"label": "not shown", "zh": "未显示",
                "why": f"P's 8-seed slope {fmt(p8).strip()}: with the censoring-bounded resamples set against it the 90% CI "
                       f"is [{cons[0]:+.3f}, {cons[1]:+.3f}] and includes 0"}
    if p8.get("bound", 0) != 0 and not (allow_bound and p8["bound"] == side):
        return {"label": "not shown", "zh": "未显示",
                "why": f"P's 8-seed slope is a censoring bound that leaves its sign open: {fmt(p8).strip()}"}
    want = int(np.sign(p8["slope"]))
    g, s = sign_status(p47, want), sign_status(s8, want)
    if g != "same" or s != "same":
        return {"label": "direction unstable", "zh": "方向不稳",
                "why": f"P {fmt(p8).strip()}; guardrail (P on seeds 4-7) {g} sign: {fmt(p47).strip()}; S (8 seeds) {s} sign: "
                       f"{fmt(s8).strip()}"}
    d = "the less arithmetic, the larger the gain" if want < 0 else "the more arithmetic, the larger the gain"
    z = "算术越少，增益越大" if want < 0 else "算术越多，增益越大"
    return {"label": f"shown: {d}", "zh": f"显示了（{z}）",
            "why": f"P {fmt(p8).strip()}; guardrail same sign ({p47['slope']:+.3f}); S same sign ({s8['slope']:+.3f})"}
