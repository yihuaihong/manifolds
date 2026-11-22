#!/usr/bin/env python
"""pt6v2 Amendment V6 aggregate (exploratory, descriptive): reads out_v6/<label>_step<step>/result.json of the 12 V5
checkpoints (probe_v6.TARGETS) and prints all6 vs only-dim side by side per seed and step, the per-seed all6 - only-dim
differences, their mean over the seeds present, and the number of seeds that entered it (no silent averaging over a
missing seed). No verdict rule; every number is reported. Writes out_v6/aggregate.{json,txt}.
A2 rows (code review B3): the drop from each arm's OWN clean value (clean - ablated), so the difference column of a
drop row is the difference in differences DiD = all6 drop - only-dim drop; absolute values follow as '.abs' rows.
Held acc (all held rows) is the primary A2 row; dim strict (n=58 at step 18309) is printed next to it.
  python -B runs/pt6v2/probe_v6_aggregate.py [--tag _smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
OUT = R / "out_v6"
STEPS = (7323, 12000, 18309)
NAN = float("nan")


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return NAN
        d = d[k]
    return NAN if d is None else d


def is_null(s):
    return s == "random" or s.startswith("null_")


def rows(r):
    """The flat per-checkpoint numbers (name -> value) that the tables compare, in print order (primary first)."""
    o = {"acc.dim_strict": r["dim_strict_acc"]}
    o |= {f"acc.{k}": v for k, v in r["acc"].items()}
    tp = r["A1"].get("torus_pair")
    if tp:
        o["A1.torus_pair.chosen_mean_seen_r2"] = tp["mean_seen_r2"]; o["A1.torus_pair.dtab_mean_seen_r2"] = tp["dtab_mean_seen_r2"]
    for k, v in r["A1"]["r2"].items():
        o[f"A1.r2.{k}"] = v["r2"]; o[f"A1.r2.{k}.held"] = v["held_r2"]
        o[f"A1.r2.{k}.shuf_max"] = g(v, "r2_shuf", "max"); o[f"A1.r2.{k}.z_shuf"] = g(v, "r2_z_shuf")
        if "strict_r2" in v:
            o[f"A1.r2.{k}.strict"] = v["strict_r2"]
    for k, v in r["A1"]["overlap"].items():   # primary: class-mean PCs; secondary: decoder column space
        for kk in ("pc3", "pc6"):
            o[f"A1.ov.{k}.{kk}"] = v[kk]; o[f"A1.ov.{k}.{kk}_rowshuf_max"] = g(v, f"{kk}_null_rowshuffle", "max")
            o[f"A1.ov.{k}.{kk}_z"] = g(v, f"{kk}_z_rowshuffle"); o[f"A1.ov.{k}.{kk}_cov95"] = g(v, f"{kk}_null_covmatched", "p95")
    for k, v in r["A1"]["overlap"].items():
        o[f"A1.ov.{k}.dec(2nd)"] = v["dec"]; o[f"A1.ov.{k}.dec_z_shuf(2nd)"] = g(v, "dec_z_shuffled")
        o[f"A1.ov.{k}.dec_cov95(2nd)"] = g(v, "dec_null_covmatched", "p95")
        if "dec_dtab" in v:
            o[f"A1.ov.{k}.dec_dtab(2nd)"] = v["dec_dtab"]
    for k, v in r["A1"]["transfer"].items():
        o[f"A1.tr.{k}"] = v["r2"]; o[f"A1.tr.{k}.shuf_max"] = g(v, "null_shuffled_src", "max"); o[f"A1.tr.{k}.z_shuf"] = g(v, "z_shuffled_src")
        o[f"A1.tr.{k}.randproj95"] = g(v, "null_random_proj", "p95")
        if v.get("strict_r2") == v.get("strict_r2") and v.get("strict_r2") is not None:
            o[f"A1.tr.{k}.strict"] = v["strict_r2"]
    for nm, rec in r["A2"].items():
        mets = [("held", "acc")] + ([("strict", "strict")] if "strict" in rec["clean"] else [])
        for lab, met in mets:
            c0 = rec["clean"][met]
            o[f"A2.{nm}.{lab}.clean"] = c0
            for h, hv in rec["conds"].items():
                for kk, cr in hv.items():
                    if kk == "k0":
                        continue
                    pre = f"A2.{nm}.{lab}.{h}.{kk}"
                    dc = cr.get("dose_cmp", {}).get(met, {})
                    for s, v in cr.items():
                        if s == "dose_cmp" or not v:
                            continue
                        if is_null(s):
                            o[f"{pre}.{s}.drop"] = c0 - g(v, met, "mean"); o[f"{pre}.{s}.dose"] = g(v, "dose", "mean")
                        else:
                            o[f"{pre}.{s}.drop"] = c0 - g(v, met); o[f"{pre}.{s}.dose"] = g(v, "dose")
                            o[f"{pre}.{s}.drop_per_dose"] = g(dc, s, "drop_per_dose")
                            o[f"{pre}.{s}.excess_vs_null_fit"] = g(dc, s, "excess_vs_null_fit")
            for h, hv in rec["conds"].items():   # absolute values after the drops
                for kk, cr in hv.items():
                    if kk == "k0":
                        continue
                    for s, v in cr.items():
                        if s != "dose_cmp" and v:
                            o[f"A2.{nm}.{lab}.{h}.{kk}.{s}.abs"] = g(v, met, "mean") if is_null(s) else g(v, met)
    for nm, rec in r["A3"].items():
        sub = "strict" if nm == "dim_held" else "all"
        for l_, lv in rec["layers"].items():
            o[f"A3.{nm}.margin.L{l_}"] = g(lv, sub, "margin"); o[f"A3.{nm}.top1.L{l_}"] = g(lv, sub, "top1")
        for p_, v in rec["dla"].get(sub, {}).items():
            o[f"A3.{nm}.dla.{p_}"] = v
    for d, v in r["A4"].items():
        o[f"A4.{d}.cka_unembed_coords"] = v["cka_unembed_vs_coords"]
        o[f"A4.{d}.cka_classmean_unembed"] = v["cka_classmean_finalnorm_vs_unembed"]
        o[f"A4.{d}.cka_classmean_coords"] = g(v, "cka_classmean_finalnorm_vs_coords")
        for kk in ("cka_unembed_vs_coords_dtab", "cka_classmean_finalnorm_vs_coords_dtab"):
            if kk in v:
                o[f"A4.{d}.{kk}"] = v[kk]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    got, missing, pairs = {}, [], {}
    for s in (16, 17):
        for arm in ("all6", "only-dim"):
            for st in STEPS:
                p = OUT / f"{arm}_s{s}_step{st}{a.tag}" / "result.json"
                if p.exists():
                    r = json.loads(p.read_text())
                    assert r["arm"] == arm, (p, r["arm"])
                    got[arm, s, st] = rows(r)
                    if r["A1"].get("torus_pair"):
                        pairs[f"{arm}_s{s}_step{st}"] = r["A1"]["torus_pair"]["chosen"]
                else:
                    missing.append(str(p))
    keys = list(dict.fromkeys(k for v in got.values() for k in v))
    L = [f"probe_v6 aggregate: {len(got)} of 12 checkpoints" + (f"; MISSING {missing}" if missing else "")]
    L.append(f"torus character pair chosen per checkpoint (B1; DTAB = [[0,1],[1,0]]): {json.dumps(pairs)}")
    L.append("columns per step: s16 all6 | s16 only-dim | s17 all6 | s17 only-dim || all6 - only-dim at s16 | at s17 | mean over "
             "the seeds present | n seeds in that mean. On A2 '.drop' rows (clean - ablated, each arm's own clean) the difference "
             "is the DiD (all6 drop - only-dim drop).")
    table = {}
    for st in STEPS:
        L.append(f"== step {st}")
        for k in keys:
            v = [got.get((arm, s, st), {}).get(k, NAN) for s in (16, 17) for arm in ("all6", "only-dim")]
            ds = [v[0] - v[1], v[2] - v[3]]
            ok = [x for x in ds if x == x]
            dd = float(np.mean(ok)) if ok else NAN
            table[f"{st}|{k}"] = dict(values=v, diff_per_seed=ds, diff_mean=dd, n_seeds=len(ok))
            L.append(f"  {k:58s} " + " ".join(f"{x:7.3f}" for x in v) + " || " + " ".join(f"{x:+7.3f}" for x in ds) +
                     f" | {dd:+7.3f} n{len(ok)}")
    (OUT / f"aggregate{a.tag}.json").write_text(json.dumps(dict(missing=missing, torus_pairs=pairs, table=table), indent=1))
    (OUT / f"aggregate{a.tag}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
