"""Selftest of tools/pttpp_summary.py (runs/pttpp/AUDIT.md "Read-out tests"), before any pttpp result. Synthetic roots in
a scratch dir (the batch's layout: pttpp/out, pttpp/out2, pttpp/decisions, the reused ptlr / ptlr_1b / pt2s505 dirs) with
designed curves; every run passes the read-out's integrity checks unless a scenario breaks one on purpose. Curves: every
held-out bucket = h(t) = min(plateau, 0.5 t / t50) (t50 = the designed crossing of 0.5; 0.9 at 1.8 t50 when plateau >= 0.9),
so R_0.5 = t50(none) / t50(prior) and Delta area follow from the design. Scenarios (designed -> asserted on the printed
read-out): size effect / no size effect / only one seed group significant (8 seeds shown, confirm seeds opposite) / only
one metric (the other metric opposite) / floor cell / censored cell / token effect at fixed size / bridge pass and fail /
an integrity failure; plus unit tests of the verdict rule, the column check (every cell and slope table prints Delta
area, R_0.5 and R_0.9 / both metrics), and the known answer on the REAL reused cells (F87 / F86 numbers).
    python tools/pttpp_selftest.py [--work <scratch dir>] [--keep]        -> SELFTEST PASS / FAIL"""
import argparse
import hashlib
import io
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pttpp_summary as S   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/tmp/pttpp_selftest")
ap.add_argument("--gen", default=S.GEN_DEFAULT)
ap.add_argument("--keep", action="store_true")
A = ap.parse_args()
WORK = Path(A.work)
G = S.load_gen(A.gen)
fails, npass = [], 0


def check(cond, what):
    global npass
    if cond:
        npass += 1
        print("PASS", what, flush=True)
    else:
        fails.append(what)
        print("FAIL", what, flush=True)


def jitter(*key, sd=0.08):
    h = int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16) / 2 ** 32
    return math.exp(sd * math.sqrt(2) * _erfinv(2 * h - 1))


def _erfinv(y):
    a = 0.147
    s = math.copysign(1, y)
    ln = math.log(max(1e-12, 1 - y * y))
    return s * math.sqrt(math.sqrt((2 / (math.pi * a) + ln / 2) ** 2 - ln / a) - (2 / (math.pi * a) + ln / 2))


def run_json(setting, size, S_, E, arm, lr, seed, tag, t50, plateau, out, bpb=1.10, lm0=None, tokens_arg=None, code=None):
    d, nl, nh, dff, n_params, nne = G.SIZES[size]
    steps = sorted(set(range(0, S_, E)) | {S_ - 1})
    ev, on = G.window_events(S_) if arm != "none" else ([], 0.0)
    bud = S_ * 16384 / 1e6
    curve = []
    for st in steps:
        t = st * 16384 / 1e6
        h = min(plateau, 0.5 * t / t50) if t50 > 0 else plateau
        inwin = bool(ev) and ev[0][0] <= st < ev[1][0]
        e = {"step": st, "tokens": st * 16384, **{f"held_{b}": h for b in S.R.B}, **{f"seen_{b}": 1.0 for b in S.R.B},
             "lm": lm0 if st == 0 else 3.0, "val_loss": 3.0, "val_bits_per_byte": bpb, "geo": 0.1 if arm != "none" else None,
             "prior_on": int(inwin), "prior_on_frac": (on if st == S_ - 1 else min(on, 0.5)) if arm != "none" else 0.0,
             "prior_used_frac": (on if st == S_ - 1 else 0.0) if arm != "none" else 0.0}
        if inwin:
            e.update(grad_lm=1.0, grad_prior=0.2, grad_cos=0.1)
        curve.append(e)
    a = {"tokenization": "whole", "target": "none" if arm == "none" else "helix3", "site": "ctx", "schedule": "whole" if arm == "none" else "window",
         "window": "0.05,0.4", "arith_lines": 4 if setting == "7" else 1, "lr": float(lr), "lam": 5.0,
         "tokens": tokens_arg if tokens_arg is not None else float(S_ * 16384), "bs": 32, "seq": 512, "d_model": d, "n_layers": nl,
         "n_heads": nh, "d_ff": dff, "seed": seed, "tag": f"_{tag}", "geo_layers": None, "eval_every": E, "n_eval": 150, "untie": False}
    r = {"args": a, "steps": S_, "n_params": n_params, "grad_ckpt_on": n_params > 150e6, "eval_every_steps": E, "lm_chunk": 0,
         "code_sha": code or (S.SNAP7 if setting == "7" else S.SNAP2), "prior_events": ev, "curve": curve}
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"run_whole_{a['target']}_seed{seed}_{tag}.json"
    f.write_text(json.dumps(r))
    return f, bud


def put_new(root, gid, arm, lr, seed, t50f, plateau, bpb=1.10, bad_lm0=False):
    c = G.GROUPS[gid]
    S_ = G.steps_of(c["size"], c["line"])
    E = G.eval_every_of(S_)
    tag = G.tag_of(gid, arm, lr)
    bud = S_ * 16384 / 1e6
    lm0 = G.lm0_ref(c["setting"], c["size"], seed) + (0.01 if bad_lm0 else 0.0)
    if c["setting"] == "7":
        out = root / "pttpp" / "out" / tag
        run_json("7", c["size"], S_, E, arm, lr, seed, tag, t50f * bud, plateau, out, bpb, lm0)
        (out / f"check_s{seed}.json").write_text(json.dumps({"pass": True}))
    else:
        task = root / "pttpp" / "out2" / f"{tag}-s{seed}"
        f, _ = run_json("2", c["size"], S_, E, arm, lr, seed, tag, t50f * bud, plateau, task / tag, bpb, lm0)
        (task / "check.json").write_text(json.dumps({"pass": True}))
        (task / "offline_lm.json").write_text(json.dumps({str(f.resolve()): {"tag": f"_{tag}", "seed": seed, "max_windows": 0,
                                                                           "orig_minus_logged": 0.0, "bpb_full": bpb}}))


def put_reused(root, ck, arm, seed, t50f, plateau, gkey=None):
    src, tag, lr = G.REUSED[ck][arm]
    size = ck[1]
    S_, E = G.STEPS_3E8, G.EVAL_3E8
    bud = S_ * 16384 / 1e6
    lm0 = G.lm0_ref(ck[0], size, seed)
    if ck[0] == "7":
        run_json("7", size, S_, E, arm, lr, seed, tag, t50f * bud, plateau, root / src / tag, 1.1, lm0, tokens_arg=3e8)
    else:
        task = root / src / f"{tag}-s{seed}"
        f, _ = run_json("2", size, S_, E, arm, lr, seed, tag, t50f * bud, plateau, task / tag, 1.1, lm0, tokens_arg=3e8)
        (task / "check.json").write_text(json.dumps({"pass": True}))
        (task / "offline_lm.json").write_text(json.dumps({str(f.resolve()): {"tag": f"_{tag}", "seed": seed, "max_windows": 0,
                                                                           "orig_minus_logged": 0.0, "bpb_full": 1.1}}))


def put_bridge(root, arm, seed, t50f, plateau):
    lr = dict(G.BRIDGE_CELLS)[arm]
    tag = G.tag_of(G.BRIDGE, arm, lr)
    out = root / "pttpp" / "out" / tag
    run_json("7", "505m", G.STEPS_3E8, G.EVAL_3E8, arm, lr, seed, tag, t50f * G.STEPS_3E8 * 16384 / 1e6, plateau, out, 1.1,
             G.lm0_ref("7", "505m", seed), tokens_arg=3e8)
    (out / f"check_s{seed}.json").write_text(json.dumps({"pass": True}))


def build(name, cellfun, bridge_none_shift=0.0, mutate=None):
    """cellfun(setting, size, tpp, arm, seed) -> (t50 as a fraction of the cell's budget, plateau). Every new group: both
    grid lrs on the selection seeds (the second lr = the same curve, higher in-training / offline bpb: an edge tie, so the
    first lr is the pick and the bracket is not open), the C decision at the first lr, its confirm seeds."""
    root = WORK / name
    shutil.rmtree(root, ignore_errors=True)
    for gid, c in G.GROUPS.items():
        lo, hi = G.GRID[gid]["none"][0], G.GRID[gid]["win-ctx"][0]
        picks = {"none": G.GRID[gid]["none"][0], "win-ctx": G.GRID[gid]["win-ctx"][0]}
        for arm in G.ARMS:
            for lr in G.GRID[gid][arm]:
                for s in c["sel"]:
                    t, p = cellfun(c["setting"], c["size"], c["line"], arm, s)
                    put_new(root, gid, arm, lr, s, t, p, bpb=1.10 if lr == picks[arm] else 1.20)
            for s in c["conf"]:
                t, p = cellfun(c["setting"], c["size"], c["line"], arm, s)
                put_new(root, gid, arm, picks[arm], s, t, p)
        dd = root / "pttpp" / "decisions"
        dd.mkdir(parents=True, exist_ok=True)
        (dd / f"{gid}.json").write_text(json.dumps({"group": gid, "log": [{"seq": i, "arm": a, "stage": "C", "lr": picks[a]}
                                                                         for i, a in enumerate(G.ARMS)]}))
    for ck in G.REUSED:
        for arm in G.ARMS:
            for s in range(8):
                t, p = cellfun(ck[0], ck[1], ck[2], arm, s)
                put_reused(root, ck, arm, s, t, p)
    for arm in G.ARMS:
        for s in range(4):
            t, p = cellfun("7", "505m", "0.745", arm, s)
            if arm == "none":
                t = t * (1 + bridge_none_shift)
            put_bridge(root, arm, s, t, p)
    if mutate:
        mutate(root)
    buf = io.StringIO()
    S.main(["--root", str(root), "--gen", A.gen], out=buf)
    txt = buf.getvalue()
    (WORK / f"{name}.txt").write_text(txt)
    return txt


def line_of(txt, start):
    m = [l for l in txt.splitlines() if l.strip().startswith(start)]
    return m[0] if m else ""


N0 = G.SIZES["93m"][5]
FP = 0.04   # the prior's t50 as a fraction of the budget


def base(fn_of):
    """prior: t50 = FP x budget, plateau 1; none: t50 = fn_of(...) x budget, plateau 1 (jitter per cell / arm / seed)"""
    def f(st, size, tpp, arm, s):
        if arm == "win-ctx":
            return FP * jitter(st, size, tpp, arm, s), 1.0
        return fn_of(st, size, tpp, s) * jitter(st, size, tpp, arm, s), 1.0
    return f


def lnN(size):
    return math.log(G.SIZES[size][5] / N0)


# ---------------------------------------------------------------- unit tests of the verdict rule
def sl(b, lo, hi):
    return {"b": b, "lo": lo, "hi": hi, "n": 4, "cells": []}


V = S.verdict
check(V(sl(.2, .1, .3), sl(.1, -.1, .3), sl(5, 1, 9), S.Q1_DIRS)[1] == "shown", "verdict: CI > 0, guard +, other + -> shown")
check(V(sl(-.2, -.3, -.1), sl(-.1, -.3, .1), sl(-5, -9, -1), S.Q1_DIRS)[1] == "shown"
      and "smaller" in V(sl(-.2, -.3, -.1), sl(-.1, -.3, .1), sl(-5, -9, -1), S.Q1_DIRS)[0], "verdict: CI < 0 -> shown (smaller)")
check(V(sl(.2, .1, .3), sl(-.1, -.3, .1), sl(5, 1, 9), S.Q1_DIRS)[1] == "unstable", "verdict: 8 seeds shown, confirm-only opposite -> unstable")
check(V(sl(.2, .1, .3), sl(.1, -.1, .3), sl(-5, -9, 1), S.Q1_DIRS)[1] == "unstable", "verdict: other metric opposite -> unstable")
check(V(sl(.2, -.1, .3), sl(.1, -.1, .3), sl(5, 1, 9), S.Q1_DIRS)[1] == "not shown", "verdict: CI includes 0 -> not shown")
check("other sign" in V(sl(.2, -.1, .3), sl(-.1, -.3, .1), sl(5, 1, 9), S.Q1_DIRS)[0]
      and V(sl(.2, -.1, .3), sl(-.1, -.3, .1), sl(5, 1, 9), S.Q1_DIRS)[1] == "not shown", "verdict: CI includes 0 and guard opposite -> not shown (noted)")
check(V(None, None, None, S.Q1_DIRS)[1] == "undeterminable", "verdict: no slope -> not determinable")
check(V(sl(.2, .1, .3), None, sl(5, 1, 9), S.Q1_DIRS)[1] == "unchecked", "verdict: guard not computable -> not fully shown")
check(V(sl(.2, .1, .3), sl(0.0, -.1, .1), sl(5, 1, 9), S.Q1_DIRS)[1] == "unstable", "verdict: a zero guard counts as not the same sign")
check(V(sl(-.2, -.3, -.1), sl(-.1, -.2, 0), sl(-1, -2, -.5), S.Q2_DIRS)[0].endswith("fewer tokens (same size)"), "verdict: Q2 negative = fewer tokens")
check(V({"b": float("nan"), "lo": float("nan"), "hi": float("nan"), "bad": "non-finite"}, None, None, S.Q1_DIRS)[1] == "undeterminable",
      "verdict: a non-finite slope -> not determinable")
x = np.log([1.0, 2.0, 4.0, 8.0])
check(abs(S.fit_slope(x, 3 * x + 1) - 3) < 1e-12 and abs(S.fit_slope(np.r_[x, x], np.r_[2 * x, 2 * x + 5], groups=[0] * 4 + [1] * 4) - 2) < 1e-12,
      "fit_slope: OLS and fixed-effects slopes on exact lines")

# ---------------------------------------------------------------- scenarios
sizeeff = base(lambda st, size, tpp, s: 0.15 * math.exp(0.25 * lnN(size)))   # R_0.5 = (0.15 / 0.04) N^0.25: grows with N
t = build("size_effect", sizeeff)
for ln in ("7% tpp 0.745 (all sizes)", "7% tpp 0.318 (all sizes)"):
    check("SHOWN: the gain is larger at larger sizes" in line_of(t, f"-> {ln}"), f"size_effect: {ln} -> shown, larger")
check("SHOWN on both lines" in line_of(t, "=> Q1 7%"), "size_effect: Q1 7% shown on both lines")
check("bridge PASS" in t, "size_effect: bridge PASS (B200 curves = L40S curves)")
m = re.search(r"7% tpp 0.745, all sizes \[PRIMARY\]\n.*\n.*\n    ln R_0.5\s+([+-][\d.]+)", t)
check(m is not None and abs(float(m.group(1)) - 0.25) < 0.05, f"size_effect: ln R_0.5 slope ~ the designed 0.25 ({m.group(1) if m else None})")
check(all(h in t for h in ("Delta area [90% CI]", "R_0.5 [90% CI]", "R_0.9 [90% CI]")), "columns: the cell table prints Delta area, R_0.5 and R_0.9")
tabs = re.findall(r"    metric .*\n((?:    (?:Delta area|ln R_0.5|ln R_0.9).*\n)+)", t)
check(len(tabs) >= 9 and all(("Delta area" in x and "ln R_0.5" in x and "ln R_0.9" in x) for x in tabs),
      f"columns: every slope table ({len(tabs)}) prints Delta area, ln R_0.5 and ln R_0.9 for all three seed sets")
rows = [l for l in t.splitlines() if re.match(r"  \S.* (0-3|4-7|0-7)\s+\|", l) or re.match(r"\s+(4-7|0-7)\s+\|", l)]
check(len(rows) >= 3 * 15 and all(l.count("|") >= 4 and "[" in l.split("|")[1] and "[" in l.split("|")[2] for l in rows),
      f"columns: every cell row ({len(rows)}) has Delta area and R_0.5 with CIs (seeds 0-3 / 4-7 / 0-7)")
check("LM share (norm) median 0.83" in t and "clip coefficient median" in t, "gradient report printed for the prior cells (designed share 1 / 1.2)")
check("2% tpp 0.745" in t and "-> 2% tpp 0.745 (all sizes): SHOWN" in t, "size_effect: 2% line shown (primary Delta area)")

t = build("no_effect", base(lambda st, size, tpp, s: 0.15 * jitter("ne", size, tpp, sd=0.02)))
check("NOT SHOWN" in line_of(t, "-> 7% tpp 0.745 (all sizes)") and "NOT SHOWN" in line_of(t, "-> 7% tpp 0.318 (all sizes)"),
      "no_effect: both 7% lines -> not shown (CI includes 0)")
check("Q1 7%: NOT SHOWN" in line_of(t, "=> Q1 7%"), "no_effect: Q1 7% not shown")


def flip(st, size, tpp, arm, s):   # 2%: seeds 0-3 gain grows steeply with N, seeds 4-7 declines mildly (Delta area)
    if st == "7":
        return sizeeff(st, size, tpp, arm, s)
    if arm == "win-ctx":
        return FP, 1.0
    k = lnN(size) - lnN("190m")
    return (0.10 + 0.25 * k if s < 4 else 0.30 - 0.03 * k) * jitter(st, size, tpp, arm, s, sd=0.01), 1.0


t = build("confirm_flip", flip)
l2 = line_of(t, "-> 2% tpp 0.745 (all sizes)")
check("DIRECTION UNSTABLE" in l2 and "confirm-only" in l2, f"confirm_flip: 2% 8 seeds shown, confirm seeds opposite -> unstable ({l2.strip()[:120]})")


def metflip(st, size, tpp, arm, s):   # 7%: R_0.5 grows with N, Delta area falls (the prior's plateau drops with N)
    k = lnN(size) / lnN("1b")
    if arm == "win-ctx":
        return FP * jitter(st, size, tpp, arm, s), 1.0 - 0.4 * k
    return 0.15 * math.exp(0.25 * lnN(size)) * jitter(st, size, tpp, arm, s), 1.0


t = build("metric_flip", metflip)
l7 = line_of(t, "-> 7% tpp 0.745 (all sizes)")
check("DIRECTION UNSTABLE" in l7 and "other metric" in l7, f"metric_flip: R_0.5 shown, Delta area opposite -> unstable ({l7.strip()[:120]})")


def floorcens(st, size, tpp, arm, s):
    if st == "7" and size == "93m" and tpp == "0.318":
        return 0.5, 0.03                     # both arms flat at 3%: floor
    if st == "7" and size == "190m" and tpp == "0.318" and arm == "none":
        return 1.6, 1.0                      # none never reaches 0.5: R_0.5 is a lower bound
    return sizeeff(st, size, tpp, arm, s)


t = build("floor_censored", floorcens)
check("** FLOOR" in "\n".join(l for l in t.splitlines() if "N_nonemb 42.0M; tpp 0.3178" in l), "floor: 93M tpp 0.318 labelled FLOOR")
s318 = t.split("7% tpp 0.318, all sizes")[1].split("-> 7% tpp 0.318")[0]
check("93m@0.318" not in s318 and "190m@0.318 R >=" in s318, "floor / censored: 93M@0.318 out of every slope; 190M@0.318 out of the R slopes (R >=)")
check(re.search(r"Delta area\s+[+-][\d.]+ \[.*190m@0.318", s318) is not None, "censored: 190M@0.318 stays in the Delta area slope")
check(">=" in "\n".join(l for l in t.splitlines() if "7% 190m tpp 0.318" in l), "censored: the cell row prints R_0.5 as '>=' (a bound)")


def tokeff(st, size, tpp, arm, s):   # R_0.5 larger with fewer tokens at the same size: none's t50 fraction ~ T^-0.3
    if arm == "win-ctx":
        return FP * jitter(st, size, tpp, arm, s), 1.0
    T = (G.STEPS_3E8 if tpp not in G.LINES else G.steps_of(size, tpp)) * 16384
    return min(0.6, 0.15 * (T / 3e8) ** -0.3) * jitter(st, size, tpp, arm, s), 1.0


t = build("token_effect", tokeff)
check("SHOWN: the gain is larger with fewer tokens" in line_of(t, "-> Q2 7% pooled"), "token_effect: Q2 pooled -> shown, fewer tokens")
t = build("bridge_fail", sizeeff, bridge_none_shift=1.5)
check("bridge FAIL" in t and "[PRIMARY]" in line_of(t, "7% tpp 0.745, L40S only"), "bridge_fail: FAIL -> the L40S-only slope is the primary")
check("[descriptive]" in line_of(t, "7% tpp 0.745, all sizes"), "bridge_fail: the all-size slope becomes descriptive")


def breakone(root):   # an integrity failure in one confirm run of 294M@0.745: its cell is withheld
    f = sorted((root / "pttpp/out/tpp7-294m-t0.745-lr4.8e-3-none").glob("run_*seed5_*.json"))[0]
    r = json.loads(f.read_text())
    r["curve"][0]["lm"] += 0.01
    f.write_text(json.dumps(r))


t = build("integrity", sizeeff, mutate=breakone)
check("WITHHELD" in line_of(t, "7% 294m tpp 0.745") and "294m@0.745" not in t.split("7% tpp 0.745, all sizes")[1].split("->")[0]
      and "PROVISIONAL" in t, "integrity: a bad step-0 lm withholds the cell, it leaves the slopes, the read-out says PROVISIONAL")

# ---------------------------------------------------------------- known answer: the REAL reused cells (F87 / F86 numbers)
D = S.Data(G, S.ROOT_DEFAULT)
cells = S.build_cells(D)
KA = {("7", "505m", "0.745"): {"0-3": 27.4, "4-7": 36.9, "0-7": 32.2, "R0-3": 4.76},
      ("7", "1b", "0.318"): {"0-3": 38.8, "4-7": 33.5, "0-7": 36.1, "R0-3": 5.69, "R4-7": 4.67, "R0-7": 5.18},
      ("7", "190m", "2.65"): {"0-3": 25.4, "4-7": 21.6, "0-7": 23.5, "R0-3": 4.56},
      ("7", "53m", "21.2"): {"0-3": 23.4, "4-7": 22.2, "0-7": 22.8, "R0-3": 4.75},
      ("2", "505m", "0.745"): {"0-3": 52.0, "4-7": 53.9, "0-7": 53.0}}
for ck, want in KA.items():
    c = cells[ck]
    if c["status"] != "final":
        check(False, f"known answer {ck}: cell {c['status']} {c['why'][:2]}")
        continue
    got = {lab: S.cell_line(c, seeds)["area"][0] for seeds, lab in S.SETS}
    for lab in ("0-3", "4-7", "0-7"):
        check(abs(got[lab] - want[lab]) < 0.051, f"known answer {ck} Delta area seeds {lab}: {got[lab]:+.2f} (FINDINGS {want[lab]:+.1f})")
        if "R" + lab in want:
            r = S.cell_line(c, dict((l, s) for s, l in S.SETS)[lab])["R0.5"][0]["ratio"]
            check(abs(r - want["R" + lab]) < 0.006, f"known answer {ck} R_0.5 seeds {lab}: {r:.3f} (FINDINGS {want['R' + lab]})")
check(not [k for k in D.runs if k[0].startswith("R") and D.runs[k]["bad"]], "known answer: no integrity failure on the real reused runs")

print(f"SELFTEST {'PASS' if not fails else 'FAIL'}: {npass} passed, {len(fails)} failed" + (f": {fails}" if fails else ""))
if not A.keep:
    for d in WORK.iterdir():
        if d.is_dir():
            shutil.rmtree(d)
sys.exit(1 if fails else 0)
