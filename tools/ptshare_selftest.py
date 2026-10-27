"""Self-test of the arithmetic-share dose read-out (tools/ptshare_rules.py, tools/ptshare_summary.py, tools/ptshare_pick.py),
written before any ptshare run (report/audits/ptshare_AUDIT.md "Tests").
Part 1, unit tests of the rules: decide() on designed slope entries (monotone / no trend / only P significant / only S
significant / 8 seeds significant but the confirm seeds reversed / S reversed / CI includes 0 / undeterminable signs / bounded
P / fewer than 3 informative shares / the positive direction), primary() (the metric switch at 3/8), slope_bound, sign_status,
floor_ceiling, uninformative.
Part 2, end to end on fake runs (designed curves on the structure of real ptlr run JSONs; fake references): the read-out's
printed and JSON numbers against an independent computation here (area, t50 / t90, R, Delta area, OLS slope), the verdicts
of designed scenarios, the metric switch, floor / ceiling shares, the side-by-side columns (Delta area, R_0.5, R_0.9 with CIs
and censored counts for seeds 0-3 / 4-7 / 8; slopes of every metric), the selection states (final, open -> expansion pick,
open after the expansion -> STOP, incomplete, withheld) and ptshare_pick.py's picks.
    python tools/ptshare_selftest.py [--work <dir>] [--nboot 2000]      -> 'PTSHARE SELFTEST PASS (n checks)'"""
import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ptshare_design as D   # noqa: E402
import ptshare_rules as Q    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/scratch/yh6210/llmadd_torch/scale_work/ptshare/selftest_runs")
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--templates", default="/scratch/yh6210/llmadd_torch/runs/ptlr/out")
A = ap.parse_args()
PY = sys.executable
N = [0]


def check(cond, what):
    N[0] += 1
    if not cond:
        print("FAIL:", what)
        sys.exit(1)


# ---------------------------------------------------------------- part 1: rules
def E(s, lo, hi, b=0):
    return {"slope": s, "ci": [lo, hi], "bound": b}


L = "the less arithmetic, the larger the gain"
cases = [("monotone (all three negative)", E(-0.6, -0.9, -0.3), E(-0.5, -1.0, 0.1), E(-8, -20, 3), 4, "显示了（算术越少，增益越大）"),
         ("no trend", E(0.02, -0.2, 0.25), E(0.1, -0.4, 0.5), E(-1, -9, 7), 4, "未显示"),
         ("only P significant, S same sign n.s.", E(-0.4, -0.7, -0.1), E(-0.3, -0.9, 0.3), E(-2, -11, 6), 4, "显示了（算术越少，增益越大）"),
         ("only S significant", E(-0.1, -0.4, 0.2), E(-0.2, -0.6, 0.2), E(-15, -25, -5), 4, "未显示"),
         ("8 seeds significant, confirm reversed", E(-0.5, -0.8, -0.2), E(0.1, -0.5, 0.7), E(-9, -18, -1), 4, "方向不稳"),
         ("S reversed", E(-0.5, -0.8, -0.2), E(-0.4, -0.9, 0.1), E(4, -3, 11), 4, "方向不稳"),
         ("CI includes 0 (point estimate printed)", E(-0.3, -0.62, 0.01), E(-0.3, -0.8, 0.2), E(-5, -12, 2), 4, "未显示"),
         ("S sign undeterminable (a bound towards 0)", E(-0.5, -0.8, -0.2), E(-0.4, -0.9, 0.1), E(-3, -9, 2, +1), 4, "方向不稳"),
         ("S bounded away from 0 (sign settled)", E(-0.5, -0.8, -0.2), E(-0.4, -0.9, 0.1), E(-3, -9, 2, -1), 4, "显示了（算术越少，增益越大）"),
         ("P bounded (not determinable)", E(-0.5, -0.8, -0.2, -1), E(-0.4, -0.9, 0.1), E(-3, -9, 2), 4, "未显示"),
         ("P missing", {"slope": None, "why": "x"}, E(-0.4, -0.9, 0.1), E(-3, -9, 2), 4, "未显示"),
         ("guardrail missing", E(-0.5, -0.8, -0.2), {"slope": None, "why": "x"}, E(-3, -9, 2), 4, "方向不稳"),
         ("fewer than 3 informative shares", E(-0.6, -0.9, -0.3), E(-0.5, -1.0, 0.1), E(-8, -20, 3), 2, "未显示"),
         ("positive direction", E(0.6, 0.3, 0.9), E(0.5, -0.1, 1.0), E(8, -3, 20), 3, "显示了（算术越多，增益越大）")]
for name, p8, p47, s8, ninf, want in cases:
    v = Q.decide(p8, p47, s8, ninf)
    check(v["zh"] == want, f"decide: {name}: {v['zh']} != {want} ({v['why']})")


def EB(s, lo, hi, b, cons=None):
    return {"slope": s, "ci": [lo, hi], "bound": b, "ci_cons": cons or [lo, hi]}


g47, s8n = E(-0.4, -0.9, 0.1), E(-3, -9, 2)
for name, p8, want in (("B: upper bound, CI below 0 (conservative)", EB(-0.5, -0.8, -0.2, -1), "显示了（算术越少，增益越大）"),
                       ("B: lower bound, CI below 0 (sign open)", EB(-0.5, -0.8, -0.2, 1), "未显示"),
                       ("B: upper bound, but the resamples bounded from below reach 0", EB(-0.5, -0.8, -0.2, -1, [-0.8, 0.05]), "未显示"),
                       ("B: exact point, conservative CI still below 0", EB(-0.5, -0.8, -0.2, 0, [-0.8, -0.1]), "显示了（算术越少，增益越大）"),
                       ("B: upper bound, CI above 0 (sign open)", EB(0.5, 0.2, 0.8, -1), "未显示"),
                       ("B: lower bound, CI above 0 (conservative)", EB(0.5, 0.2, 0.8, 1), "方向不稳")):
    v = Q.decide(p8, g47, s8n, 4, allow_bound=True)
    check(v["zh"] == want, f"decide reading B: {name}: {v['zh']} != {want} ({v['why']})")
check(Q.decide(EB(-0.5, -0.8, -0.2, -1), g47, s8n, 4)["zh"] == "未显示", "reading A: a bounded P is not determinable")
check(Q.decide(EB(0.5, 0.2, 0.8, 0, [-0.1, 0.8]), E(0.4, -0.1, 0.9), E(3, -2, 9), 4)["zh"] == "未显示",
      "reading A: a positive claim with resamples bounded from above reaching 0 -> not shown")
check(Q.decide(EB(0.5, 0.2, 0.8, 0, [0.1, 0.8]), E(0.4, -0.1, 0.9), E(3, -2, 9), 4)["zh"] == "显示了（算术越多，增益越大）",
      "reading A: a positive claim whose conservative CI stays above 0 -> shown")
check(Q.primary({1: 3, 2: 0}) == "dA" and Q.primary({1: 0, 2: 3}) == "dA" and Q.primary({1: 2, 2: 2}) == "lR50"
      and Q.primary({1: 8, 2: 8, 4: 8}) == "dA" and Q.primary({}) == "lR50", "primary(): switch at >= 3/8 at 2% or 4%")
_, w = Q.ols([1, 2, 4, 8], [0, 0, 0, 0])
check(np.allclose(w, [-0.4325, -0.1442, 0.1442, 0.4325], atol=1e-3), f"ols weights {w}")
check(Q.slope_bound(w, [1, 0, 0, 0]) == -1 and Q.slope_bound(w, [0, 0, 0, 1]) == 1 and Q.slope_bound(w, [1, 0, 0, 1]) is None
      and Q.slope_bound(w, [-1, 0, 0, 0]) == 1 and Q.slope_bound(w, [0, 0, 0, 0]) == 0, "slope_bound directions")
b, _ = Q.ols([1, 2, 4, 8], [math.log(8), math.log(4), math.log(2), 0.0])
check(abs(b + 1.0) < 1e-12, f"ols slope of log(8/k) on ln k = -1 ({b})")
check(Q.sign_status(E(-0.5, -1, 0, -1), -1) == "same" and Q.sign_status(E(-0.5, -1, 0, 1), -1) == "undetermined"
      and Q.sign_status(E(0.5, 0, 1), -1) == "opposite" and Q.sign_status(E(0.0, -1, 1), -1) == "undetermined", "sign_status")
check(Q.floor_ceiling(4.9, 0.05) == "floor" and Q.floor_ceiling(4.9, 0.2) is None and Q.floor_ceiling(90, 0.95) == "ceiling"
      and Q.floor_ceiling(89.9, 1.0) is None and Q.floor_ceiling(50, 0.5) is None, "floor_ceiling thresholds")
check(Q.uninformative("floor", "floor") and Q.uninformative("ceiling", "ceiling") and not Q.uninformative("floor", None)
      and not Q.uninformative(None, None) and not Q.uninformative(None, "ceiling"), "uninformative")
rf = D.refs()
check(rf is not None and all(sorted(rf[sec]) == list(D.LINES) and all(sorted(rf[sec][k]) == list(range(8)) for k in D.LINES)
                             for sec in ("lm0", "share", "dry"))
      and all(abs(rf["share"][k][s]["share"] / k - rf["share"][1][s]["share"]) < 0.002 for k in D.LINES for s in range(8)),
      "ptshare_refs.json via ptshare_design.refs(): 4 shares x 8 seeds, share proportional to k (within 0.2 points per line)")
print(f"part 1: {N[0]} rule checks pass")

# ---------------------------------------------------------------- part 2: fake runs
T = Path(A.templates)
TPL = {"none": json.loads((T / "53m-lr8e-3-7pct-none/run_whole_none_seed0_53m-lr8e-3-7pct-none.json").read_text()),
       "win-ctx": json.loads((T / "53m-lr2e-3-7pct-win-ctx/run_whole_helix3_seed0_53m-lr2e-3-7pct-win-ctx.json").read_text())}
STEPS = [e["step"] for e in TPL["none"]["curve"]]
assert len(STEPS) == 75 and STEPS[-1] == 18309
TOKM = 32 * 512 / 1e6
BKT = ("b11", "b12", "b13", "b22", "b23", "b33")
LM0 = {k: {s: round(11.6 + 0.0013 * s + 0.0007 * k, 6) for s in range(8)} for k in D.LINES}


def h(step, s0, s1, hmax):
    if step <= s0:
        return 0.0
    if step >= s1:
        return hmax
    return hmax * (step - s0) / (s1 - s0)


def mine_area(c):
    return 100 * float(np.mean([h(st, *c) for st in STEPS]))


def mine_t(c, th):
    prev = None
    for st in STEPS:
        v = h(st, *c)
        if v >= th:
            if prev is None:
                return st * TOKM
            return (prev[0] + (th - prev[1]) / (v - prev[1]) * (st - prev[0])) * TOKM
        prev = (st, v)
    return math.inf


def write_run(root, src, k, arm, lr, seed, c, bpb=1.22, lm0=None, drop_last=False, tag=None, noprior=False, diverge=None):
    r = copy.deepcopy(TPL[arm])
    tag = tag or ((D.tag(k, arm, lr) if k != 4 else f"53m-lr{lr}-7pct-{arm}") if src != "ptgw" else "main-W")
    out = Path(root) / src / "out" / tag
    out.mkdir(parents=True, exist_ok=True)
    r["args"].update(arith_lines=k, lr=float(lr), seed=seed, tag=f"_{tag}", out=str(out))
    for i, e in enumerate(r["curve"]):
        v = h(e["step"], *c)
        for b_ in BKT:
            e[f"held_{b_}"] = v
        e["val_bits_per_byte"] = bpb
        if noprior:
            e["grad_prior"] = None
        if diverge is not None and e["step"] >= diverge:   # NaN from here on (rule 2: counted with its area)
            for b_ in BKT:
                e[f"held_{b_}"] = 0.0
            e["lm"] = e["val_loss"] = float("nan")
            e["geo"] = float("nan")
            if e.get("grad_prior") is not None:
                e["grad_prior"] = float("nan")
        if i == 0:
            e["lm"] = LM0[k][seed] if lm0 is None else lm0
    if drop_last:
        r["curve"] = r["curve"][:-10]
    tgt = "none" if arm == "none" else "helix3"
    (out / f"run_whole_{tgt}_seed{seed}_{tag}.json").write_text(json.dumps(r))


# design: centre lrs best; the two side lrs are the same curve delayed (area lower by > 2)
def jit(seed):
    return 50 * ((seed * 7) % 5 - 2)   # -100 .. +100 steps, a fixed pattern per seed


def build(root, none_c, prior_c, special=None):
    """none_c / prior_c: {k: function(seed) -> (s0, s1, hmax)} at the centre lr; the side lrs get the same curve delayed by
    3000 steps and a worse in-training bpb (1.25 vs 1.22, so a tie-break never picks them). special: {(k, arm, '*', s):
    {'c'}} replaces the centre curve before the delay; {(k, arm, lr, s): {'c' | 'skip' | 'lm0'}} acts on one run after it;
    {('extra', k, arm): {lr: function(seed)}} adds an expansion lr (seeds 0-3)"""
    shutil.rmtree(root, ignore_errors=True)
    special = special or {}
    for k in D.LINES:
        for arm, cf in (("none", none_c), ("win-ctx", prior_c)):
            g = D.GRID[arm]
            for lr in g:
                for s in range(8):
                    if lr != g[1] and s >= 4:
                        continue
                    c = special.get((k, arm, "*", s), {}).get("c") or cf[k](s)
                    if lr != g[1]:
                        c = (c[0] + 3000, c[1] + 3000, c[2])
                    key = (k, arm, lr, s)
                    if key in special:
                        c = special[key].get("c", c)
                        if special[key].get("skip"):
                            continue
                    src = "ptshare" if k != 4 else ("ptgw" if (arm == "win-ctx" and lr == "1e-3" and s < 4) else "ptlr")
                    write_run(root, src, k, arm, lr, s, c, bpb=1.22 if lr == g[1] else 1.25,
                              lm0=special.get(key, {}).get("lm0"))
            for lr, c_ in special.get(("extra", k, arm), {}).items():
                for s in range(4):
                    write_run(root, "ptshare", k, arm, lr, s, c_(s))
    refs = {"lm0": {str(k): {str(s): v for s, v in d.items()} for k, d in LM0.items()},
            "share": {str(k): {str(s): {"share": RSH[k][s]} for s in range(8)} for k in D.LINES}, "dry": {}, "meta": {}}
    (Path(root) / "refs.json").write_text(json.dumps(refs))


def run(root, tagname, hide=False):
    js = Path(root) / f"{tagname}.json"
    o = subprocess.run([PY, str(HERE / "ptshare_summary.py"), "--root", str(root), "--refs", str(Path(root) / "refs.json"),
                        "--json", str(js), "--nboot", str(A.nboot)] + ([] if hide else ["--show_both"]), capture_output=True, text=True)
    if o.returncode:
        print(o.stdout[-3000:], o.stderr[-3000:])
        check(False, f"{tagname}: ptshare_summary.py failed")
    return o.stdout, json.loads(js.read_text())


RSH = {k: {s: v["share"] for s, v in d.items()} for k, d in D.refs()["share"].items()}   # the real measured shares
L = {k: D.share_label(k) for k in D.LINES}                                                   # '2.12%', '4.25%', '8.50%', '17.00%'
check(list(L.values()) == ["2.12%", "4.25%", "8.50%", "17.00%"], f"measured labels {L}")


def VA(J):
    return J["verdict"]["readings"]["A"]


def VB(J):
    return J["verdict"]["readings"]["B"]


def pick(root, sel_json, rnd):
    pk = Path(root) / f"picks_r{rnd}.json"
    pk.unlink(missing_ok=True)
    o = subprocess.run([PY, str(HERE / "ptshare_pick.py"), "--round", str(rnd), str(sel_json), str(pk)], capture_output=True, text=True)
    check(o.returncode == 0, f"ptshare_pick round {rnd}: {o.stderr[-500:]}")
    return o.stdout, (json.loads(pk.read_text()) if pk.exists() else None)


W = Path(A.work)
P_FAST = {k: (lambda s: (1500 + jit(s), 3500 + jit(s), 1.0)) for k in D.LINES}   # prior: t50 ~ step 2500, area ~86 (below the ceiling)
NONE_T50 = {1: 12000, 2: 8000, 4: 5000, 8: 3000}                                 # none's t50 step falls with the share


def none_mono(k):
    return lambda s: (NONE_T50[k] - 1000 + 3 * jit(s), NONE_T50[k] + 1000 + 3 * jit(s), 1.0)


# ---- scenario A: monotone, no switch -> P = log R_0.5, shown (the less arithmetic, the larger)
root = W / "A"
build(root, {k: none_mono(k) for k in D.LINES}, P_FAST)
write_run(root, "ptlr", 4, "none", "8e-3", 0, (0, 1000, 1.0), tag="53m-lr8e-3-7pct-none-dup")   # same args, another tag
out, J = run(root, "A")
check(VA(J)["zh"] == "显示了（算术越少，增益越大）" and VA(J)["P"] == "dA" and VA(J)["S"] == "lR50" and VB(J)["zh"] == VA(J)["zh"]
      and VB(J)["P"] == "lR50" and J["verdict"]["reading"] == "A" and "VERDICT (reading A" in out
      and "增益随占比下降而增大" in (J["verdict"]["wording"] or ""), f"A verdict {J['verdict']}")
check("attribution (mandatory): area none slope" in out and "area win-ctx slope" in out
      and abs(J["slopes"]["8"]["area_prior"]["slope"] - J["slopes"]["8"]["area_none"]["slope"] - J["slopes"]["8"]["dA"]["slope"]) < 1e-9,
      "A: the mandatory attribution line; Delta area slope = prior slope - none slope")
check("covariate (descriptive; lambda fixed" in out and set(J["covariate"]["first_on_prior_over_lm_median"]) == set(L.values())
      and "covariate at one lr (descriptive): win-ctx@2e-3" in out and set(J["covariate_same_lr"]["prior_over_lm_median"]) == set(L.values()),
      "A: the gradient-ratio covariate lines (selected lr; one lr)")
check("是保守的" in J["verdict"]["wording"] and "同向" not in J["verdict"]["wording"] and f"高占比端（{L[4]} / {L[8]}）" in J["verdict"]["wording"],
      "A: the bias wording (high end enlarges, low end conservative)")
check(all(lab in out for lab in L.values()) and not any(x + "pct" in out for x in ("2", "4", "14")) and "7pct " not in out,
      "A: every output names the shares by their measured share")
check(all(J["shares"][str(k)]["prior_t50_after_window"] > 0 for k in D.LINES) and "the prior's t50 is" in out
      and "after the window opens" in out, "A: the prior-t50-after-the-window line")
check("left out" in out and "is not the cell's pre-registered tag: 1" in out
      and abs(J["shares"]["4"]["groups"]["0-3"]["area_none"] - np.mean([mine_area(none_mono(4)(s)) for s in range(4)])) < 1e-9,
      "A: a same-args run under another tag in ptlr is left out, not merged")
check(set(J["loso"]["dA"]) == {L[1], L[2], L[4], L[8]} and all(v < 0 for v in J["loso"]["lR50"].values()),
      "A: leave-one-share-out slopes printed")
for k in D.LINES:
    sel = J["select"][str(k)]
    check(sel["none"]["status"] == "final" and sel["none"]["best"] == "8e-3" and sel["win-ctx"]["best"] == "2e-3", f"A select {k} {sel}")
    g = J["shares"][str(k)]["groups"]["8"]
    an = np.mean([mine_area(none_mono(k)(s)) for s in range(8)])
    aw = np.mean([mine_area(P_FAST[k](s)) for s in range(8)])
    check(abs(g["area_none"] - an) < 1e-9 and abs(g["area_prior"] - aw) < 1e-9 and abs(g["dA"] - (aw - an)) < 1e-9,
          f"A areas {k}: {g['area_none']} vs {an}, {g['area_prior']} vs {aw}")
    tn = sorted(mine_t(none_mono(k)(s), 0.5) for s in range(8))
    tw = sorted(mine_t(P_FAST[k](s), 0.5) for s in range(8))
    r_mine = np.median(tn) / np.median(tw)
    check(abs(g["R50"]["ratio"] - r_mine) < 1e-9, f"A R_0.5 {k}: {g['R50']['ratio']} vs {r_mine}")
    t9 = np.median([mine_t(none_mono(k)(s), 0.9) for s in range(8)]) / np.median([mine_t(P_FAST[k](s), 0.9) for s in range(8)])
    check(abs(g["R90"]["ratio"] - t9) < 1e-9, f"A R_0.9 {k}")
ys = [math.log(J["shares"][str(k)]["groups"]["8"]["R50"]["ratio"]) for k in D.LINES]
b_m, _ = Q.ols(list(D.LINES), ys)
check(abs(J["slopes"]["8"]["lR50"]["slope"] - b_m) < 1e-9 and J["slopes"]["8"]["lR50"]["ci"][1] < 0, f"A slope {J['slopes']['8']['lR50']}")
# the side-by-side columns: every share x group line has Delta area, R_0.5, R_0.9 with a CI and censored counts; every slope
lines = [l for l in out.splitlines() if " seeds " in l and "none@" in l and "win-ctx@" in l]
check(len(lines) == 12 and all("dArea" in l and "R_0.5" in l and "R_0.9" in l and l.count("cen ") == 2 and l.count("[") == 3
                               for l in lines), "A: the per-share columns (Delta area, R_0.5, R_0.9 with CI and censored counts)")
for gname in ("0-3", "4-7", "8"):
    for lab in ("Delta area", "log R_0.5", "log R_0.9", "area none", "area win-ctx"):
        check(any(l.strip().startswith(f"seeds {gname}") and lab in l and "per ln(share)" in l for l in out.splitlines()),
              f"A: slope line {gname} {lab}")
check("the 7% sub-grid selection" not in out, "A: the 7% selection = F87's lrs (no warning)")
print(f"scenario A (monotone; reading A, P = Delta area): {VA(J)['zh']}; slope {J['slopes']['8']['dA']['slope']:+.3f}")
_, pk = pick(root, root / "A.json", 1)
check(pk is not None and sorted((c["k"], c["arm"], c["lr"], c["why"]) for c in pk["cells"]) ==
      sorted((k, a, D.GRID[a][1], "confirm") for k in D.NEW for a in D.ARMS) and all(c["seeds"] == [4, 5, 6, 7] for c in pk["cells"]),
      f"A picks {pk}")

# ---- scenario B: the switch (2% none never reaches 0.5 in 3/8 seeds) -> P = Delta area; and 2/8 -> log R_0.5
for n_cen, wantP in ((3, "dA"), (2, "lR50"), (4, "dA")):
    root = W / f"B{n_cen}"
    sp = {(1, "none", "*", s): {"c": (9000, 16000, 0.45)} for s in range(n_cen)}
    build(root, {k: none_mono(k) for k in D.LINES}, P_FAST, sp)
    out, J = run(root, f"B{n_cen}")
    check(J["shares"]["1"]["none_t50_censored_8"] == n_cen and VA(J)["P"] == "dA" and VB(J)["P"] == "lR50", f"B{n_cen}: P {J['verdict']}")
    check(VA(J)["zh"] == "显示了（算术越少，增益越大）" and VB(J)["zh"] == "显示了（算术越少，增益越大）", f"B{n_cen} verdict {J['verdict']}")
    check((J["slopes"]["8"]["lR50"]["bound"] == -1 and L[1] in J["slopes"]["8"]["lR50"]["bounds"]) == (n_cen >= 4),
          f"B{n_cen}: a censored 2% none median (>= 4/8) makes the log R slope an upper bound {J['slopes']['8']['lR50']}")
    check(f"{L[1]} {n_cen}/8" in out, f"B{n_cen}: none's censored t50 count is printed")
    print(f"scenario B ({n_cen}/8 of 2% none censored at t50): A: P = {VA(J)['P']}, {VA(J)['zh']}; B: {VB(J)['zh']}")

# ---- scenario C: no trend (same curves at every share) -> not shown
root = W / "C"
flat = {k: (lambda s: (6000 + 6 * jit(s), 8000 + 6 * jit(s), 1.0)) for k in D.LINES}
pj = {k: (lambda s, k=k: (1000 + 4 * jit((s + k) % 8), 2000 + 4 * jit((s + k) % 8), 1.0)) for k in D.LINES}
build(root, flat, pj)
out, J = run(root, "C")
check(VA(J)["zh"] == "未显示" and VB(J)["zh"] == "未显示", f"C verdict {J['verdict']}")
print(f"scenario C (no trend): {VA(J)['zh']} ({VA(J)['why'][:110]})")

# ---- scenario F: 8 seeds negative, the confirm seeds reversed (P = Delta area via the switch) -> direction unstable
root = W / "F"
T50F = {1: 12000, 2: 8000, 4: 5000, 8: 3000}
T50G = {1: 3500, 2: 4000, 4: 4500, 8: 5000}   # seeds 4-7: none slower with more arithmetic -> gain grows with the share


def none_f(k):
    return lambda s: ((T50F[k] if s < 4 else T50G[k]) - 1000, (T50F[k] if s < 4 else T50G[k]) + 1000, 1.0) if not (k == 1 and s < 3) \
        else (9000, 16000, 0.45)


build(root, {k: none_f(k) for k in D.LINES}, P_FAST)
out, J = run(root, "F")
check(VA(J)["P"] == "dA" and J["slopes"]["8"]["dA"]["ci"][1] < 0 and J["slopes"]["4-7"]["dA"]["slope"] > 0
      and VA(J)["zh"] == "方向不稳" and not VB(J)["zh"].startswith("显示了"), f"F verdict {J['verdict']} / {J['slopes']['4-7']['dA']}")
print(f"scenario F (confirm seeds reversed): A {VA(J)['zh']}, B {VB(J)['zh']}")

# ---- scenario F2: the secondary metric reversed (P = log R_0.5 falls with the share, Delta area rises) -> direction unstable
root = W / "F2"
PRI = {1: (500, 1100, 0.6), 2: (800, 1800, 0.75), 4: (1100, 2700, 0.9), 8: (1500, 3500, 1.0)}
build(root, flat, {k: (lambda s, k=k: (PRI[k][0] + jit(s), PRI[k][1] + jit(s), PRI[k][2])) for k in D.LINES})
out, J = run(root, "F2")
check(VA(J)["P"] == "dA" and J["slopes"]["8"]["lR50"]["ci"][1] < 0 and J["slopes"]["8"]["dA"]["slope"] > 0
      and VA(J)["zh"] == "方向不稳" and VB(J)["zh"] == "方向不稳" and "S (8 seeds) opposite sign" in VA(J)["why"], f"F2 verdict {J['verdict']}")
print(f"scenario F2 (secondary reversed): {VA(J)['zh']}")

# ---- scenario N: a win-ctx run without prior gradients (the prior silently off) -> that cell withheld
root = W / "N"
build(root, {k: none_mono(k) for k in D.LINES}, P_FAST)
write_run(root, "ptshare", 2, "win-ctx", "2e-3", 1, P_FAST[2](1), noprior=True)
out, J = run(root, "N")
check(J["select"]["2"]["win-ctx"]["status"] == "withheld" and "prior not acting" in out and J["verdict"]["zh"] == "未完成",
      f"N: prior not acting -> withheld {J['select']['2']['win-ctx']}")
print("scenario N (a prior run with no prior gradient): withheld")

# ---- scenario Dv: a diverged win-ctx run at a side lr is counted (rule 2), not withheld by the prior-activity check;
# and without --show_both no reading's verdict is printed or written (the reading is chosen before any result)
root = W / "Dv"
build(root, {k: none_mono(k) for k in D.LINES}, P_FAST)
c4 = P_FAST[2](1)
write_run(root, "ptshare", 2, "win-ctx", "4e-3", 1, (c4[0] + 3000, c4[1] + 3000, c4[2]), bpb=1.25, diverge=3000)
out, J = run(root, "Dv")
check(J["select"]["2"]["win-ctx"]["status"] == "final" and "4e-3" in J["select"]["2"]["win-ctx"].get("diverged", {})
      and "prior not acting" not in out, f"Dv: a diverged prior run counts, not withheld {J['select']['2']['win-ctx']}")
out, J = run(root, "Dv_hidden", hide=True)
check(J["verdict"]["reading"] == "A" and set(J["verdict"]["readings"]) == {"A"} and "reading B (" not in out
      and "VERDICT (reading A" in out, "without --show_both only the chosen reading A is computed and printed")
print("scenario Dv (a diverged prior run; only reading A without --show_both): OK")

# ---- scenario G: floor at 2% (both arms) -> 2% left out; slope over 3 shares
root = W / "G"
fl = {(1, a, "*", s): {"c": (15000, 18000, 0.02)} for a in D.ARMS for s in range(8)}
build(root, {k: none_mono(k) for k in D.LINES}, P_FAST, fl)
out, J = run(root, "G")
check(J["shares"]["1"]["uninformative"] and J["shares"]["1"]["floor"] == {"none": True, "prior": True}
      and f"uninformative shares left out of every slope: {L[1]}" in out and set(J["slopes"]["8"]["dA"]["per_share"]) == {L[2], L[4], L[8]},
      f"G floor {J['shares']['1'].get('floor')}")
print(f"scenario G (2% both at the floor): left out; {VA(J)['zh']}")

# ---- scenario H: the prior alone at the ceiling at 14% -> note; both at the ceiling -> 14% uninformative
root = W / "H"
ce = {(8, "win-ctx", lr, s): {"c": (900, 1250, 1.0)} for lr in ("2e-3",) for s in range(8)}
build(root, {k: none_mono(k) for k in D.LINES}, P_FAST, ce)
out, J = run(root, "H")
check(J["shares"]["8"]["prior_ceiling_only"] and f"the prior alone is at the ceiling at {L[8]}:" in out
      and not any(J["shares"][str(k)]["prior_ceiling_only"] for k in (1, 2, 4)), "H: prior-only ceiling note (14% only)")
root = W / "H2"
ce2 = dict(ce)
ce2.update({(8, "none", "8e-3", s): {"c": (900, 1300, 1.0)} for s in range(8)})
build(root, {k: none_mono(k) for k in D.LINES}, P_FAST, ce2)
out, J = run(root, "H2")
check(J["shares"]["8"]["uninformative"] and L[8] not in J["slopes"]["8"]["dA"]["per_share"], "H2: both at the ceiling -> left out")
print("scenario H (ceilings): prior-only note; both -> left out")

# ---- scenario S: selection states and the picks
root = W / "S"
sp = {(8, "none", "1.6e-2", s): {"c": (2000 - 1000, 2000 + 1000, 1.0)} for s in range(4)}   # 14% none best at the upper edge
sp.update({(2, "win-ctx", "1e-3", 3): {"skip": True}})                                     # 4% win-ctx incomplete
sp.update({(1, "none", "4e-3", 2): {"lm0": LM0[1][2] + 0.01}})                              # 2% none withheld (step-0 lm)
build(root, {k: none_mono(k) for k in D.LINES}, P_FAST, sp)
out, J = run(root, "S")
s8n, s4w, s1n = J["select"]["8"]["none"], J["select"]["2"]["win-ctx"], J["select"]["1"]["none"]
check(s8n["status"] == "open" and s8n["open"] == "upper" and s8n["best"] == "1.6e-2", f"S: 14% none open {s8n}")
check(s4w["status"] == "incomplete" and s1n["status"] == "withheld", f"S: incomplete / withheld {s4w['status']} {s1n['status']}")
check(J["verdict"]["zh"] == "未完成", f"S: no verdict while selections are open ({J['verdict']})")
pout, pk = pick(root, root / "S.json", 1)
cells = {(c["k"], c["arm"], c["lr"], c["why"]) for c in pk["cells"]}
check((8, "none", "3.2e-2", "expand") in cells and (8, "win-ctx", "2e-3", "confirm") in cells and not any(c[:2] in ((2, "win-ctx"), (1, "none")) for c in cells)
      and f"nothing for {L[2]} win-ctx: incomplete" in pout and f"nothing for {L[1]} none: withheld" in pout, f"S picks round 1 {cells}")
_, pk2 = pick(root, root / "S.json", 2)
check(pk2 is None or (8, "none", "3.2e-2", "expand") not in {(c["k"], c["arm"], c["lr"], c["why"]) for c in pk2["cells"]}, "S: round 2 never expands")
for name, c_, want in (("closed", lambda s: (5000, 7000, 1.0), ("final", "1.6e-2")), ("open again", lambda s: (1000, 1800, 1.0), ("stop", "3.2e-2"))):
    sp2 = dict(sp)
    sp2[("extra", 8, "none")] = {"3.2e-2": c_}
    root2 = W / f"S_{name.replace(' ', '_')}"
    build(root2, {k: none_mono(k) for k in D.LINES}, P_FAST, sp2)
    _, J2 = run(root2, "S2")
    s = J2["select"]["8"]["none"]
    check((s["status"], s["best"]) == want and s["expanded"], f"S expansion {name}: {s}")
    pout2, pk3 = pick(root2, root2 / "S2.json", 2)
    if want[0] == "final":
        check(pk3 is not None and (8, "none", "1.6e-2", "confirm") in {(c["k"], c["arm"], c["lr"], c["why"]) for c in pk3["cells"]},
              "S: closed after the expansion -> confirm at the best lr")
    else:
        check(f"STOP {L[8]} none" in pout2 and (pk3 is None or all(c[:2] != (8, "none") for c in
                                                                 [(c["k"], c["arm"]) for c in pk3["cells"]])), "S: open again -> STOP")
    print(f"scenario S ({name} after the expansion): {s['status']} at {s['best']}")
print(f"PTSHARE SELFTEST PASS ({N[0]} checks)")
