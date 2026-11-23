#!/usr/bin/env python3
"""pt6v2 Amendment V7b read (post-hoc, exploratory; NO check gates it). Arms, paired by seed 16..23:
  all6dropS = V7b (out_v7b/, torus / box5 lines replaced in their slots by text fillers from step 7324),
  all6drop  = V7  (out_v7/,  torus / box5 lines left out, text extends left: dim's slot-1 share shifts),
  all6, only-dim = the registered @2e-3 runs (out/).
Per-run metrics = out_v7/v7read.py's (its load() / f3() / at(), exec'd read only) + dim mean over the evals in 7324..12000 and
12001..18309, the final bits-per-byte, and dim's slot-1 share (cumulative slot_frac at the end; over steps 7251..end, from the
cumulative slot_frac at 7250 and the end and the cycling rule's dim line counts).
Summary:
  S_b = (all6dropS - only-dim) / (all6 - only-dim), dim strict final3, ratio of seed means, seed-bootstrap 95% CI (10000, seed 0);
        the same for V7's all6drop (F109: 0.77);
  cost all6 - all6dropS: dim final3 and dim mean over 7324..12000 and 12001..18309 (paired, bootstrap CI);
  slot effect all6drop - all6dropS: the same metrics, paired per seed, bootstrap 95% CI;
  torus / box5 held collapse (mean over the evals >= 7324, final3), numbers / cities / countries final3, bpb: paired diffs;
  device and check status per run: flags only.
The decision rule (AUDIT.md Amendment V7b, verbatim) is printed with the numbers; it is applied by the reader.
usage: v7bread.py [--json <out.json>] [--test-alias]   (--test-alias: SMOKE ONLY, all6dropS := the V7 dirs; S_b must = V7's share
and the slot effect must be exactly 0)"""
import json
import sys
from pathlib import Path

import numpy as np

V7R = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2/out_v7/v7read.py")
_s = V7R.read_text()
G = {"__name__": "v7read_prefix", "__file__": str(V7R)}
exec(compile(_s[:_s.index("RUN, missing = {}, []")], str(V7R), "exec"), G)   # v4final prefix, load, f3, at, device (read only)
load, f3, at, SEEDS, R = G["load"], G["f3"], G["at"], G["SEEDS"], G["R"]
DROP_AT, WCLOSE, SEG = 7324, 7250, ((7324, 12000), (12001, 18309))
TEST = "--test-alias" in sys.argv
ARMS = {"all6dropS": (lambda s: R / "out_v7" / f"pt6v2-all6drop-lr2e-3-s{s}") if TEST else (lambda s: R / "out_v7b" / f"pt6v2-all6dropS-lr2e-3-s{s}"),
        "all6drop": lambda s: R / "out_v7" / f"pt6v2-all6drop-lr2e-3-s{s}",
        "all6": lambda s: R / "out" / f"pt6v2-all6-lr2e-3-s{s}",
        "only-dim": lambda s: R / "out" / f"pt6v2-only-dim-lr2e-3-s{s}"}
RULE = ("Report S_b with its CI and the paired slot effect; if the slot effect's CI includes 0 and S_b's CI includes 0.77, F109's 77% "
        "stands and its qualifier 2 is replaced by the measured slot effect; otherwise the paper's 77% is replaced by S_b. No check "
        "gates the read-out.")
B_N, B_SEED = 10000, 0


def n_lines(a, d, s):   # dim lines in steps 0..s by the cycling rule (every kept line is counted in slot_frac)
    ext = [x for x in a["domains"].split(",") if x != "numbers"]
    bs, K = a["bs"], a["extra_lines"]
    i = np.arange((s + 1) * bs * K)
    return int(((i % len(ext)) == ext.index(d)).sum())


def extra(d):
    fs = sorted(d.glob("*/run_whole_*.json"))
    r = json.loads(fs[0].read_text())
    c, a = r["curve"], r["args"]
    st = [e["step"] for e in c]
    e_end, e_w = c[-1], c[st.index(WCLOSE)]
    f_end, f_w = e_end["slot_frac"]["dim"][0], e_w["slot_frac"]["dim"][0]
    n_end, n_w = n_lines(a, "dim", e_end["step"]), n_lines(a, "dim", WCLOSE)
    flags = []
    f = d / "check_v7b.json"
    if f.exists():
        k = json.loads(f.read_text())
        if not k.get("pass"):
            flags.append("check_v7b.json: " + "; ".join(x["check"][:50] for x in k["checks"] if not x["ok"])[:200])
    return {"bpb": float(e_end["val_bits_per_byte"]), "slot1_end": float(f_end),
            "slot1_post": float((f_end * n_end - f_w * n_w) / (n_end - n_w)), "flags_b": flags}


RUN, missing = {}, []
for a, p in ARMS.items():
    for s in SEEDS:
        x = load(p(s))
        if x is None:
            missing.append(f"{a} s{s}")
        else:
            x.update(extra(p(s)))
            RUN[(a, s)] = x
print(("## TEST ALIAS (smoke only): all6dropS := the V7 all6drop dirs\n" if TEST else "") + "missing / unfinished:", missing or "none")
row = {}
for (a, s), x in sorted(RUN.items()):
    post = x["st"] >= DROP_AT
    v = {"dim_f3": f3(x["S"]["dim"]), "dim_area": 100 * float(np.mean(x["S"]["dim"])), "dim_wclose": at(x, "dim", WCLOSE),
         "dim_drop": at(x, "dim", WCLOSE) - f3(x["S"]["dim"])}
    for k_, (lo, hi) in zip(("dim_seg1", "dim_seg2"), SEG):
        m = (x["st"] >= lo) & (x["st"] <= hi)
        v[k_] = 100 * float(np.mean(x["S"]["dim"][m]))
    v.update({f"{d}_post": 100 * float(np.mean(x["S"][d][post])) for d in ("torus", "box5")})
    v.update({f"{d}_f3": f3(x["S"][d]) for d in ("torus", "box5", "numbers", "cities", "countries")})
    v.update(bpb=x["bpb"], slot1_end=x["slot1_end"], slot1_post=x["slot1_post"], dev=x["dev"], flags=x["flags"] + x["flags_b"])
    row[(a, s)] = v
cols = ["dim_f3", "dim_seg1", "dim_seg2", "dim_wclose", "torus_post", "box5_post", "numbers_f3", "cities_f3", "countries_f3", "bpb",
        "slot1_end", "slot1_post"]
print(f"{'arm':9s} {'seed':>4s} " + " ".join(f"{c:>11s}" for c in cols) + "  device / flags")
for (a, s), v in sorted(row.items()):
    print(f"{a:9s} {s:4d} " + " ".join(f"{v[c]:11.4f}" if c in ("bpb", "slot1_end", "slot1_post") else f"{v[c]:11.2f}" for c in cols)
          + f"  {v['dev']} {v['flags'] or ''}")
rng = np.random.default_rng(B_SEED)
BOOT = {}


def boot_idx(n):
    if n not in BOOT:
        BOOT[n] = rng.integers(0, n, size=(B_N, n))
    return BOOT[n]


def paired(a, b, c):
    ss = [s for s in SEEDS if (a, s) in row and (b, s) in row]
    if len(ss) < 2:
        return None
    d = np.array([row[(a, s)][c] - row[(b, s)][c] for s in ss])
    bm = d[boot_idx(len(ss))].mean(1)
    return {"n": len(ss), "mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "pos": int((d > 0).sum()), "neg": int((d < 0).sum()), "per_seed": dict(zip(ss, d.round(3).tolist()))}


def share(arm):   # (arm - only-dim) / (all6 - only-dim), dim final3, ratio of seed means over the seeds in all three arms
    ss = [s for s in SEEDS if all((x, s) in row for x in (arm, "all6", "only-dim"))]
    if len(ss) < 2:
        return None
    num = np.array([row[(arm, s)]["dim_f3"] - row[("only-dim", s)]["dim_f3"] for s in ss])
    den = np.array([row[("all6", s)]["dim_f3"] - row[("only-dim", s)]["dim_f3"] for s in ss])
    ix = boot_idx(len(ss))
    bs_ = num[ix].mean(1) / den[ix].mean(1)
    return {"n": len(ss), "S": float(num.mean() / den.mean()), "ci95": [float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))],
            "num": float(num.mean()), "den": float(den.mean()), "seeds": ss}


OUTJ = {"test_alias": TEST, "rule": RULE, "rows": {f"{a}/s{s}": v for (a, s), v in row.items()}, "paired": {}, "share": {}}
print("\nshare of all6's dim final3 advantage over only-dim (ratio of seed means; seed-bootstrap 95% CI, 10000)")
for arm, lab in (("all6dropS", "S_b (V7b, slot-matched)"), ("all6drop", "S   (V7, F109)")):
    sh = share(arm)
    OUTJ["share"][arm] = sh
    if sh:
        print(f"  {lab:24s} {sh['S']:.3f}  CI [{sh['ci95'][0]:.3f}, {sh['ci95'][1]:.3f}]  (n={sh['n']}; {arm} - only-dim "
              f"{sh['num']:+.2f}, all6 - only-dim {sh['den']:+.2f})")
    else:
        print(f"  {lab:24s} n/a ({arm} missing)")
print("\npaired by seed (A - B, points; bpb in bits/byte; slot1 as a fraction): mean, sd, bootstrap 95% CI, sign counts")
GROUPS = [("all6", "all6dropS", "cost of the drop, slot-matched", ("dim_f3", "dim_seg1", "dim_seg2", "dim_area")),
          ("all6drop", "all6dropS", "slot effect (V7 - V7b)", ("dim_f3", "dim_seg1", "dim_seg2", "dim_area", "slot1_post")),
          ("all6dropS", "all6", "torus / box5 collapse, damage, bpb, slot", ("torus_post", "box5_post", "torus_f3", "box5_f3",
                                                                            "numbers_f3", "cities_f3", "countries_f3", "bpb", "slot1_end", "slot1_post")),
          ("all6drop", "all6", "V7 for reference", ("dim_f3", "dim_seg1", "dim_seg2", "torus_post", "box5_post", "bpb", "slot1_post")),
          ("all6dropS", "only-dim", "V7b vs only-dim", ("dim_f3", "dim_seg1", "dim_seg2"))]
for a, b, lab, cs in GROUPS:
    print(f" {lab}")
    for c in cs:
        p = paired(a, b, c)
        OUTJ["paired"][f"{a}-{b}/{c}"] = p
        if p:
            fm = ".4f" if c in ("bpb", "slot1_end", "slot1_post") else ".2f"
            print(f"  {a:9s} - {b:9s} {c:11s} n={p['n']} {p['mean']:+{fm}} (sd {p['sd']:{fm}}) CI [{p['ci95'][0]:+{fm}}, "
                  f"{p['ci95'][1]:+{fm}}] +{p['pos']}/-{p['neg']}")
        else:
            print(f"  {a:9s} - {b:9s} {c:11s} n/a")
print("\ndim slot-1 share per arm (mean over seeds): end (cumulative) / over steps 7251..end")
for a in ARMS:
    v = [row[(a, s)] for s in SEEDS if (a, s) in row]
    if v:
        print(f"  {a:9s} n={len(v)}  {np.mean([x['slot1_end'] for x in v]):.4f} / {np.mean([x['slot1_post'] for x in v]):.4f}")
se, sb = OUTJ["paired"].get("all6drop-all6dropS/dim_f3"), OUTJ["share"].get("all6dropS")
print("\nDECISION RULE (AUDIT.md Amendment V7b): " + RULE)
if se and sb:
    inc0 = se["ci95"][0] <= 0 <= se["ci95"][1]
    inc77 = sb["ci95"][0] <= 0.77 <= sb["ci95"][1]
    print(f"  slot effect (dim final3) {se['mean']:+.2f} CI [{se['ci95'][0]:+.2f}, {se['ci95'][1]:+.2f}] includes 0: {inc0}; "
          f"S_b {sb['S']:.3f} CI [{sb['ci95'][0]:.3f}, {sb['ci95'][1]:.3f}] includes 0.77: {inc77} -> "
          + ("F109's 77% stands; qualifier 2 -> the measured slot effect" if inc0 and inc77 else "the paper's 77% -> S_b"))
    OUTJ["rule_inputs"] = {"slot_effect_ci_includes_0": inc0, "S_b_ci_includes_0.77": inc77}
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(OUTJ, indent=1))
