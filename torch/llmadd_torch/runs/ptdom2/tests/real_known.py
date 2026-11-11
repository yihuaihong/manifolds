"""ptdom2 test on REAL ptdom data (read only) with planted ptdom2 cells (temp dirs; nothing written in runs/ptdom or runs/ptdom2):
  1. load2(runs/ptdom/out, 'ptdom') = readout_ptdom.load(runs/ptdom/out) on every ptdom cell (every record field but 'why' /
     'source'): ptdom2's loader is ptdom's.
  2. Part S (A2: the verdict on seeds 8-15, three ptdom2 arms): the seeds 8-15 cells are copies of the real ptdom runs of seeds
     0-7 relabelled (only-<d>@2e-3 as the true arm, none@2e-3, and a shuffled arm = the true copy minus a PLANTED delta in area,
     final3, hard final3 and the per-eval series). verdict_S must (a) give true - none = the real only-<d> minus none values read
     independently from runs/ptdom/out, (b) true - shuffled = the planted deltas (1e-9), (c) the planted GEOMETRY-SPECIFIC
     verdicts (+3 everywhere -> x 4; 0 -> none; countries +3 only -> countries), (d) the descriptive seeds 0-7 line = part S0's
     planted deltas against ptdom's selected true arm, labelled as selected; the window split sums to the gain.
  3. R1 on real data: planted only-<d>@4e-3 seeds 0-3 (copies of the real only-<d>@2e-3 runs with area + 20 / - 20); decide_R's
     picks must equal an independent re-implementation of F87 rule 1 (top2 within 2 points -> lower mean in-training bpb; the
     planted cells excluded only by the gate) on the same values; the none side is the real ptdom data (its R1x none
     extensions are printed: the pre-registered rule applied to existing data).
-> real_known.out
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # A4: the readout beside this tests/ dir (runs/ptdom2 or a working copy)
sys.argv = [sys.argv[0]]
import readout_ptdom2 as RO2   # noqa: E402
RO, M2 = RO2.RO, RO2.M2
print(f"readout under test: {RO2.__file__}", flush=True)
R0 = RO2.R0 / "out"
res = []


def T(name, ok, info=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info != "" else ""), flush=True)


# ---------------------------------------------------------------- 1. load2 = load
L0 = RO.load(R0)
L2 = RO2.load2(R0, "ptdom")
skip = ("why", "source")
diff = [k for k in L0 if k not in L2 or {f: v for f, v in L0[k].items() if f not in skip} != {f: v for f, v in L2[k].items() if f not in skip and f in L0[k]}]
extra = sorted({f for r in L2.values() for f in r} - {f for r in L0.values() for f in r} - set(skip))
T(f"1. load2 = readout_ptdom.load on all {len(L0)} ptdom cells (every field of load's records; load2 adds {extra})",
  not diff and set(L0) == set(L2) and sum(r["valid"] for r in L0.values()) == sum(r["valid"] for r in L2.values()) == 384,
  f"{len(diff)} differ; valid {sum(r['valid'] for r in L2.values())}")


def plant(root2, src_dir, prefix_src, batch, arm_src, arm_new, lr_src, lr_new, seed, d, delta, d3=None, new_seed=None):
    """Copy a real run dir into root2 under the ptdom2 name of (batch, arm_new, lr_new, seed); identity fields rewritten;
    metrics of d: area - delta, final3 - d3 (default delta), hard final3 - d3."""
    t_src, t_new = RO2.tag_of(prefix_src, batch, arm_src, lr_src), RO2.tag_of("ptdom2", batch, arm_new, lr_new)
    ns = seed if new_seed is None else new_seed   # A2: a copy relabelled as another seed (its data_sha series stays the source's)
    dst = Path(root2) / f"{t_new}-s{ns}"
    (dst / t_new).mkdir(parents=True)
    r = json.loads((Path(src_dir) / t_src / f"run_whole_none_seed{seed}_{t_src}.json").read_text())
    r["args"]["tag"] = f"_{t_new}"
    r["args"]["lr"] = float(lr_new)
    r["args"]["seed"] = ns
    dm = M2.dom(arm_new)
    r["args"]["prior_shuf"] = dm if M2.is_shuf(arm_new) else ""
    for x, v in r["domains"].items():
        v["shuf"] = bool(M2.is_shuf(arm_new) and x == dm)
    k = RO.READ[d]
    d3 = delta if d3 is None else d3
    r["metrics"][k]["area"] -= delta
    r["metrics"][k]["final3"] -= d3
    for e in r["curve"]:   # A2: the per-eval series too (area = 100 x its mean; the window split reads it)
        e[RO.EVAL_KEY[d]] -= delta / 100
    if d in RO.COMP and r["metrics"].get(f"{d}_readout_hard", {}).get("final3") is not None:
        r["metrics"][f"{d}_readout_hard"]["final3"] -= d3
    r.get("domains", {}).get(d, {}).pop("eval_held", None)   # no per-question data: the planted values are not hit strings
    (dst / t_new / f"run_whole_none_seed{ns}_{t_new}.json").write_text(json.dumps(r))
    shutil.copy(Path(src_dir) / "check.json", dst / "check.json")
    return dst


def direct(batch, d, s, arm="only", lr="2e-3"):
    """An independent read of a real ptdom run's primary metric."""
    a = f"only-{d}" if arm == "only" else arm
    t = f"ptdom-{batch}-{a}-lr{lr}"
    f = R0 / f"{t}-s{s}" / t / f"run_whole_none_seed{s}_{t}.json"
    m = json.loads(f.read_text())["metrics"]
    return m[{"months": "months_readout_kspan"}.get(d, f"{d}_readout")]


# ---------------------------------------------------------------- 2. Part S on real data (A2)
scen = {"+3 everywhere": {d: [3.0 + 0.5 * (s % 2) for s in range(8)] for d in M2.S_DOMS},
        "0 everywhere": {d: [0.0] * 8 for d in M2.S_DOMS},
        "countries +3 only": {d: ([3.0 + 0.5 * (s % 2) for s in range(8)] if d == "countries" else [0.0] * 8) for d in M2.S_DOMS}}
want = {"+3 everywhere": ["GEOMETRY-SPECIFIC"] * 4, "0 everywhere": ["NOT SHOWN"] * 4,
        "countries +3 only": ["GEOMETRY-SPECIFIC", "NOT SHOWN", "NOT SHOWN", "NOT SHOWN"]}
for nm, dl in scen.items():
    with tempfile.TemporaryDirectory() as tmp:
        for d, b in M2.S_DOMS.items():
            for s in range(8):
                rec = L2[(b, f"only-{d}", "2e-3", s)]
                # the verdict cells at seeds 8-15: copies of ptdom's seed-(s) true arm, none@2e-3 and a shuffled arm = true - delta
                plant(tmp, rec["dir"], "ptdom", b, f"only-{d}", f"only-{d}", "2e-3", "2e-3", s, d, 0.0, new_seed=s + 8)
                plant(tmp, rec["dir"], "ptdom", b, f"only-{d}", f"only-{d}-shuf", "2e-3", "2e-3", s, d, dl[d][s], new_seed=s + 8)
                plant(tmp, rec["dir"], "ptdom", b, f"only-{d}", f"only-{d}-shuf", "2e-3", "2e-3", s, d, dl[d][s])   # part S0 (descriptive)
        for b, lr in (("B1", "2e-3"), ("A", "2e-3"), ("A", "4e-3")):   # A3: A's none@4e-3 (F100's pick for cities / months)
            for s in range(8):
                rec = L2[(b, "none", lr, s)]
                plant(tmp, rec["dir"], "ptdom", b, "none", "none", lr, lr, s, "countries" if b == "B1" else "cities", 0.0, new_seed=s + 8)
        Rm = {**L2, **RO2.load2(tmp, "ptdom2", guard=False)}
        probs = RO.integrity(Rm) + RO2.series_problems(Rm)
        o = RO2.verdict_S(Rm, {})
        rd = {d: o["rows"][d] for d in M2.S_DOMS}
        ind_t = {d: [direct(b, d, s)["area"] for s in range(8)] for d, b in M2.S_DOMS.items()}
        ind_n = {d: [direct(b, d, s, arm="none", lr=M2.S_NONE[d])["area"] for s in range(8)] for d, b in M2.S_DOMS.items()}
        ind_n2 = {d: [direct(b, d, s, arm="none", lr="2e-3")["area"] for s in range(8)] for d, b in M2.S_DOMS.items()}
        T(f"2. S [{nm}]: the verdict contrasts read the seeds 8-15 cells: true - none = the real only-<d>@2e-3 minus none at F100's pick "
          "(countries / people 2e-3, cities / months 4e-3; A3.1) of the source seeds read independently (months: its k-span), true - "
          "shuffled = the planted deltas; cities / months' same-lr line = the real only-<d> minus none@2e-3; every arm a ptdom2 run",
          all(np.allclose(rd[d]["gain"], np.array(ind_t[d]) - np.array(ind_n[d]), atol=1e-9) and np.allclose(rd[d]["geo_gain"], dl[d], atol=1e-9)
              for d in M2.S_DOMS) and all(abs(rd[d]["same_lr_pass"]["mean"] - float(np.mean(np.array(ind_t[d]) - np.array(ind_n2[d])))) < 1e-9
                                          for d in ("cities", "months")) and not o["problems"] and not probs, f"integrity / series {probs or 'ok'}")
        T(f"2. S [{nm}]: the descriptive seeds 0-7 line pairs ptdom's selected true arm with part S0 (= the planted deltas), labelled as selected",
          all(np.allclose(rd[d]["seeds0_7_selected"]["per_seed"], np.round(dl[d], 2)) and "selected by F100" in rd[d]["seeds0_7_selected"]["label"]
              for d in M2.S_DOMS))
        T(f"2. S [{nm}]: GEOMETRY-SPECIFIC verdicts {[rd[d]['geometry'] for d in M2.S_DOMS]} = planted {want[nm]}; the window split sums to the gain",
          [rd[d]["geometry"] for d in M2.S_DOMS] == want[nm] and all(
              np.allclose(np.array(rd[d]["split"]["vs none"]["in"]) + np.array(rd[d]["split"]["vs none"]["out"]), rd[d]["gain"]) for d in M2.S_DOMS))

# ---------------------------------------------------------------- 3. R1 selection on real data (planted 4e-3)
def f87(cells):
    """An independent F87 rule 1 ('top2' within 2 points -> lower mean in-training bpb (offline only if every run of both has
    it), then the lower lr)."""
    m = {lr: float(np.mean(v["area"])) for lr, v in cells.items()}
    order = sorted(m, key=lambda lr: (-m[lr], float(lr)))
    near = order[:2] if len(order) > 1 and m[order[0]] - m[order[1]] <= 2.0 else order[:1]
    off = all(x is not None for lr in near for x in cells[lr]["off"])
    bp = {lr: float(np.mean(cells[lr]["off" if off else "intr"])) for lr in near}
    pick = min(near, key=lambda lr: (bp[lr], float(lr)))
    return pick, max(m, key=lambda lr: (m[lr], -float(lr))), m


der = RO2.none_asks_from_ptdom(L2)
T("3. A1.10: on ptdom's data ALONE the pre-registered rule asks exactly R1PRE = A none@1.6e-2 (stars' / landmarks' none pick / argmax "
  "8e-3 = A's top) and B2 none@5e-4 (prefixes' none pick / argmax 1e-3 = B2's lowest); check_r1pre finds no problem",
  der == sorted(M2.R1PRE, key=lambda t: (t[0], float(t[2]))) and RO2.check_r1pre(L2) == [], der)
for d, b in M2.R_DOMS.items():
    g = list(M2.NONE_LRS0[b])
    cn = {lr: {"area": [L2[(b, "none", lr, s)]["area"][d] for s in M2.SEL], "off": [L2[(b, "none", lr, s)]["bpb_off"] for s in M2.SEL],
               "intr": [L2[(b, "none", lr, s)]["bpb_intr"] for s in M2.SEL]} for lr in g}
    pn, an, mnn = f87(cn)
    print(f"    {d}: none over batch {b}'s ptdom lrs {g}: pick {pn}, argmax {an} (means {', '.join(f'{k} {x:.1f}' for k, x in mnn.items())})"
          + (" -> up" if g[-1] in (pn, an) else "") + (" -> down" if g[0] in (pn, an) else ""))
for sign in (+20.0, -20.0):
    with tempfile.TemporaryDirectory() as tmp:
        for d, b in M2.R_DOMS.items():
            for s in M2.SEL:
                rec = L2[(b, f"only-{d}", "2e-3", s)]
                plant(tmp, rec["dir"], "ptdom", b, f"only-{d}", f"only-{d}", "2e-3", "4e-3", s, d, -sign)
        for b, _, lr in M2.R1PRE:   # the R1pre cells: copies of the batch's nearest real none run, 20 points lower (never the argmax)
            src_lr = {"1.6e-2": "8e-3", "5e-4": "1e-3"}[lr]
            for s in M2.SEL:
                rec = L2[(b, "none", src_lr, s)]
                dd = plant(tmp, rec["dir"], "ptdom", b, "none", "none", src_lr, lr, s, "stars" if b == "A" else "compass", 0.0)
                f = next((dd / RO2.tag_of("ptdom2", b, "none", lr)).glob("run_*.json"))
                j = json.loads(f.read_text())
                for x in M2.R_DOMS:
                    if x in j["metrics"].get(RO.READ[x], {}) or RO.READ[x] in j["metrics"]:
                        j["metrics"][RO.READ[x]]["area"] -= 20.0
                f.write_text(json.dumps(j))
        Rm = {**L2, **RO2.load2(tmp, "ptdom2", guard=False)}
        pk = RO2.decide_R(Rm, "R1")
        ok, lines = True, []
        E_ = {tuple(x) for x in pk["E"]}
        for d, b in M2.R_DOMS.items():
            v = pk["domains"][d]
            cp = {lr: {"area": [Rm[(b, f"only-{d}", lr, s)]["area"][d] for s in M2.SEL], "off": [Rm[(b, f"only-{d}", lr, s)]["bpb_off"] for s in M2.SEL],
                       "intr": [Rm[(b, f"only-{d}", lr, s)]["bpb_intr"] for s in M2.SEL]} for lr in v["prior_lrs"] if lr not in v["gate_excluded"]}
            gn = sorted(set(M2.NONE_LRS0[b]) | {lr for bb, _, lr in M2.R1PRE if bb == b}, key=float)
            cn = {lr: {"area": [Rm[(b, "none", lr, s)]["area"][d] for s in M2.SEL], "off": [Rm[(b, "none", lr, s)]["bpb_off"] for s in M2.SEL],
                       "intr": [Rm[(b, "none", lr, s)]["bpb_intr"] for s in M2.SEL]} for lr in gn}
            pp, ap, mp = f87(cp)
            pn, an, mnn = f87(cn)
            ok &= (pp, ap, pn, an) == (v["pick_prior"], v["argmax_prior"], v["pick_none"], v["argmax_none"]) and v["none_lrs"] == gn
            ok &= (v["ext_prior"] == "8e-3") == ("4e-3" in (pp, ap))   # A1.2: pick or argmax at the top 4e-3
            up_due = gn[-1] in (pn, an) and float(gn[-1]) <= float(max(M2.NONE_LRS0[b], key=float))
            dn_due = gn[0] in (pn, an) and float(gn[0]) >= float(min(M2.NONE_LRS0[b], key=float))
            ok &= ((b, "none", M2.NONE_EXT[b]) in E_) >= up_due and ((b, "none", M2.NONE_LOW[b]) in E_) >= dn_due
            outside = float(pp) < float(gn[0]) and not dn_due
            ok &= (v["fixb"] == pp) == outside
            lines.append(f"{d}: prior {pp} (argmax {ap}) none {pn} (argmax {an}) over {gn}; ext prior {v['ext_prior']}, none {v['ext_none']}, "
                         f"fixB {v['fixb']}, status {v['status']}")
        T(f"3. R1 on real ptdom data with the planted 4e-3 cells at 2e-3 {sign:+.0f} and the R1pre cells: decide_R's picks / argmaxes / asks = "
          f"an independent F87 rule 1 and the A1 edge rules on the same values (gate on the real gradient fields)", ok, "; ".join(lines))
        if sign > 0:
            T("3. planted +20 at 4e-3: every R prior asks for 8e-3 (R1x)", all(v["ext_prior"] == "8e-3" for v in pk["domains"].values()))
        else:
            T("3. planted -20 at 4e-3: no prior asks for 8e-3; R1pre's cells are not asked again", all(v["ext_prior"] is None for v in pk["domains"].values())
              and not E_ & {tuple(x) for x in M2.R1PRE})
            print("    asks after R1 here (with ptdom's none data and these planted prior cells): " + str(sorted(E_)))
print(f"REAL KNOWN {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
