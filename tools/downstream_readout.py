"""Pre-registered read-out of the downstream evaluation (tools/downstream_eval.py); rules: llmadd_torch/runs/downstream/
AUDIT.md (committed before any score). Per comparison and benchmark: d_s = prior - none (percentage points) per common
seed 0-3, mean, SD, 95% t-CI, wins; verdict (non-inferiority margin 1.0 point):
  1. "at chance: uninformative" if both arms' means are within 2 points of chance;
  2. "non-inferior" if the CI lower bound >= -margin;  3. "worse" if the CI upper bound < 0 and the mean < -margin;
  4. else "undetermined";  fewer than 4 common seeds -> "INCOMPLETE".
Reference = the group's none cell with the lowest mean bpb over seeds 0-3 (offline full-val bpb_full when present for
every seed of every none cell, else the final logged in-training val_bits_per_byte); prior - reference is descriptive.

    python tools/downstream_readout.py --selftest                       # fake inputs; must pass before the real read-out
    python tools/downstream_readout.py --out_dir llmadd_torch/runs/downstream/out --write llmadd_torch/runs/downstream/readout
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import tempfile
from pathlib import Path

BENCH = ("blimp", "lambada", "hellaswag", "arc_easy", "piqa")
PRIMARY = {"blimp": "acc", "lambada": "acc", "hellaswag": "acc_norm", "arc_easy": "acc", "piqa": "acc"}
CHANCE = {"blimp": 50.0, "lambada": 0.0, "hellaswag": 25.0, "piqa": 50.0}   # arc_easy: from the eval JSON (mean 1/n)
MARGIN = {b: 1.0 for b in BENCH}
AT_CHANCE = 2.0
SEEDS = (0, 1, 2, 3)
T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646}
EXPECT_N = {"blimp": 67000, "lambada": 5153, "hellaswag": 10042, "arc_easy": 2376, "piqa": 1838}

G505, G190 = "505M 7%", "190M rainbow"
COMPARISONS = [   # pre-registered (AUDIT.md); cells = <runs family>/<model dir name>
    dict(id="P1", role="primary", group=G505, label="win-ctx@1.2e-3 vs none@1.2e-3",
         prior="ptlr/500m-lr1.2e-3-7pct-win-ctx", none="ptlr/500m-lr1.2e-3-7pct-none"),
    dict(id="P2", role="primary", group=G505, label="win-ctx@3e-4 vs none@3e-4",
         prior="ptsize999/500m-7pct-win-ctx", none="ptsize999/500m-7pct-none"),
    dict(id="P3", role="primary", group=G190, label="all6-lam1@6e-4 vs none@6e-4",
         prior="rb6_190m/rb6tb-190m-2pct-k2-all6-lam1-lr6e-4", none="rb6_190m/rb6tb-190m-2pct-k2-none-lr6e-4"),
    dict(id="D1", role="descriptive", group=G505, label="gate-ctx@3e-4 vs none@3e-4",
         prior="ptsize999/500m-7pct-gate-ctx", none="ptsize999/500m-7pct-none"),
    dict(id="D2", role="descriptive", group=G505, label="gate-text@3e-4 vs none@3e-4",
         prior="ptsize999/500m-7pct-gate-text", none="ptsize999/500m-7pct-none"),
    dict(id="D3", role="descriptive", group=G190, label="only-numbers@6e-4 vs none@6e-4",
         prior="rb6_190m/rb6tb-190m-2pct-k2-only-numbers-lr6e-4", none="rb6_190m/rb6tb-190m-2pct-k2-none-lr6e-4"),
]
NONES = {G505: ["ptsize999/500m-7pct-none", "ptlr/500m-lr1.2e-3-7pct-none", "ptlr/500m-lr4.8e-3-7pct-none",
                "ptlr/500m-lr9.6e-3-7pct-none", "ptlr/500m-lr1.92e-2-7pct-none"],
         G190: ["rb6_190m/rb6tb-190m-2pct-k2-none-lr6e-4", "rb6_190m/rb6tb-190m-2pct-k2-none-lr4.8e-3"]}
DESC_FAMILIES = ("ptlr", "pt999", "ptsize999")   # 53M / 190M 7% seed-0 cells (descriptive)


# ------------------------------------------------------------------------------------------------------- loading
def load(out_dir: Path) -> dict:
    """cell -> seed -> record (project models only; smoke / known-answer files excluded)."""
    cells = {}
    for f in sorted(out_dir.glob("*/*.json")):
        if f.parent.name in ("known_answer", "smoke"):
            continue
        r = json.loads(f.read_text())
        if r.get("kind") != "project" or r.get("limit"):
            continue
        cell = f"{f.parent.name}/{Path(r['model']).parent.name}"
        assert r["seed"] not in cells.get(cell, {}), ("two records for one cell and seed", cell, r["seed"])
        cells.setdefault(cell, {})[r["seed"]] = r
    return cells


def score(r: dict, b: str) -> float:
    return r["scores"][b][PRIMARY[b]]


def chance(b: str, recs) -> float:
    if b == "arc_easy":
        cs = {round(r["scores"][b]["chance"], 6) for r in recs}
        assert len(cs) == 1, cs
        return cs.pop()
    return CHANCE[b]


def bpb_of(r: dict):
    """(offline full-val bpb or None, logged final in-training bpb)."""
    rj = str(Path(r["run_json"]).resolve())
    for d in (Path(r["model"]).parent, Path(r["model"]).parent.parent):
        p = d / "offline_lm.json"
        if p.exists():
            e = json.loads(p.read_text()).get(rj)
            if e and e.get("max_windows", 0) == 0:
                return e["bpb_full"], r["val_bpb_logged_final"]
    return None, r["val_bpb_logged_final"]


# ------------------------------------------------------------------------------------------------------ statistics
def tci(d):
    n = len(d)
    m = statistics.fmean(d)
    if n < 2:
        return m, None, None, None
    sd = statistics.stdev(d)
    h = T975[n - 1] * sd / math.sqrt(n)
    return m, sd, m - h, m + h


def verdict(d, mp, mn, ch, margin):
    if len(d) < len(SEEDS):
        return "INCOMPLETE"
    if abs(mp - ch) <= AT_CHANCE and abs(mn - ch) <= AT_CHANCE:
        return "at chance: uninformative"
    m, sd, lo, hi = tci(d)
    if lo >= -margin:
        return "non-inferior" + (" (small cost within margin: CI < 0)" if hi < 0 else "")
    if hi < 0 and m < -margin:
        return "worse"
    return "undetermined"


HITS = {"blimp": "hits", "lambada": "hits", "hellaswag": "hits_acc_norm", "arc_easy": "hits_acc", "piqa": "hits_acc"}
NBOOT = 2000


def boot2(A, B, common, be, nboot=NBOOT, seed=0):
    """Amendment A1, descriptive: seed x item bootstrap of the mean paired difference (points): resample the common seeds
    and the items (both with replacement); 95% percentile interval. Covers item sampling and (crudely, 4 seeds) seed
    variation; not a verdict."""
    import numpy as np
    k = HITS[be]
    if not all(k in A[s]["scores"][be] and k in B[s]["scores"][be] for s in common):
        return None
    D = np.stack([np.frombuffer(A[s]["scores"][be][k].encode(), np.uint8).astype(np.int8)
                  - np.frombuffer(B[s]["scores"][be][k].encode(), np.uint8).astype(np.int8) for s in common])
    for s in common:   # the hit strings reproduce the stored scores (double path for the bootstrap input)
        for X in (A, B):
            assert abs(100 * (np.frombuffer(X[s]["scores"][be][k].encode(), np.uint8) == 49).mean() - score(X[s], be)) < 1e-6
    rng = np.random.default_rng(seed)
    n = D.shape[1]
    vals = np.empty(nboot)
    for i in range(nboot):
        it = rng.integers(0, n, n)
        sd_ = rng.integers(0, len(common), len(common))
        vals[i] = 100 * D[:, it].mean(1)[sd_].mean()
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "nboot": nboot,
            "item_only_sd_of_mean": float(100 * D.mean(0).std(ddof=1) / math.sqrt(n))}


def paired(cells, a, b, bench, seeds=SEEDS, boot=False):
    A, B = cells.get(a, {}), cells.get(b, {})
    common = [s for s in seeds if s in A and s in B]
    out = {}
    for be in bench:
        d = [score(A[s], be) - score(B[s], be) for s in common]
        if not d:
            out[be] = {"n": 0, "verdict": "INCOMPLETE"}
            continue
        mp = statistics.fmean(score(A[s], be) for s in common)
        mn = statistics.fmean(score(B[s], be) for s in common)
        ch = chance(be, [A[s] for s in common] + [B[s] for s in common])
        m, sd, lo, hi = tci(d)
        out[be] = {"n": len(d), "seeds": common, "d": d, "mean": m, "sd": sd, "lo": lo, "hi": hi,
                   "wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "prior_mean": mp, "none_mean": mn,
                   "chance": ch, "verdict": verdict(d, mp, mn, ch, MARGIN[be]),
                   "boot2": boot2(A, B, common, be) if boot else None}
    lr = [math.log(A[s]["scores"]["lambada"]["ppl"] / B[s]["scores"]["lambada"]["ppl"]) for s in common]
    if lr:
        m, sd, lo, hi = tci(lr)
        out["lambada_ppl_ratio"] = {"n": len(lr), "per_seed": [math.exp(x) for x in lr], "geo_mean": math.exp(m),
                                    "lo": math.exp(lo) if lo is not None else None, "hi": math.exp(hi) if hi is not None else None}
    return out


def reference(cells, group):
    rows = {}
    for c in NONES[group]:
        recs = [cells[c][s] for s in SEEDS if s in cells.get(c, {})]
        rows[c] = (recs, [bpb_of(r) for r in recs])
    full = all(len(recs) == len(SEEDS) and all(o is not None for o, _ in bp) for recs, bp in rows.values())
    src = "offline bpb_full" if full else "logged final val_bits_per_byte"
    if not all(len(recs) == len(SEEDS) for recs, _ in rows.values()):   # A2: chosen on partial cells -> provisional
        src += " (PROVISIONAL: a none cell lacks seeds 0-3)"
    mean_bpb = {c: statistics.fmean((o if full else l) for o, l in bp) for c, (recs, bp) in rows.items() if bp}
    best = min(mean_bpb, key=mean_bpb.get) if mean_bpb else None
    return best, src, mean_bpb


def check_pair(cells, cp) -> list:
    """Amendment A2: identity of the two arms from the records, not only the directory names: the prior arm trains with a
    target (not none), the none arm without; same lr, same training code, same n_params, same architecture; the model file
    name carries the run's target and seed. Returns problems (the read-out stops on any)."""
    P, N = cells.get(cp["prior"], {}), cells.get(cp["none"], {})
    bad = []
    for s in sorted(set(P) & set(N)):
        p, n = P[s], N[s]
        for r in (p, n):
            if f"_{r['target']}_seed{r['seed']}_" not in Path(r["model"]).name:
                bad.append(f"{cp['id']} s{s}: model file {Path(r['model']).name} does not carry target {r['target']} seed {r['seed']}")
        if p["target"] == "none" or n["target"] != "none":
            bad.append(f"{cp['id']} s{s}: targets prior {p['target']} / none {n['target']}")
        for k in ("lr", "train_code_sha", "n_params", "d_model", "n_layers", "seed"):
            if p.get(k) != n.get(k):
                bad.append(f"{cp['id']} s{s}: {k} differs ({p.get(k)} vs {n.get(k)})")
    return bad


def analyse(cells: dict) -> dict:
    res = {"comparisons": [], "reference": {}, "cells": {}, "desc7": [], "checks": {}}
    shas = {r["eval_code_sha"] for c in cells.values() for r in c.values()}
    bad_n = [(c, s, b) for c, v in cells.items() for s, r in v.items() for b in BENCH if r["scores"][b]["n"] != EXPECT_N[b]]
    setup = {(r["prefix"], r["max_len"], r.get("dtype")) for c in cells.values() for r in c.values()}
    res["checks"] = {"eval_code_shas": sorted(shas), "n_mismatch": bad_n, "setups": sorted(map(list, setup)),
                     "max_double_path": max((r["scores"][b]["double_path"]["max_abs_dloglik"] for c in cells.values()
                                             for r in c.values() for b in BENCH), default=None),
                     "n_trunc": sum(r["tok_stats"][b]["n_trunc"] for c in cells.values() for r in c.values() for b in BENCH)}
    for g in (G505, G190):
        best, src, mb = reference(cells, g)
        res["reference"][g] = {"cell": best, "bpb_source": src, "mean_bpb": mb}
    an = []   # anomaly flags (AUDIT.md: outside [chance - 3, 100] -> inspect; Amendment A1: load check, double path)
    for c, v in cells.items():
        for s, r in v.items():
            for b in BENCH:
                ch = r["scores"][b].get("chance", CHANCE.get(b, 0.0))
                for m in ("acc", "acc_norm"):   # A2: every accuracy metric, not only the primary one
                    v = r["scores"][b].get(m)
                    if v is not None and not (ch - 3 <= v <= 100):
                        an.append(f"{c} s{s} {b} {m} {v:.2f} outside [chance {ch:.1f} - 3, 100]")
                if r["scores"][b]["double_path"].get("greedy_mismatch", 0):
                    an.append(f"{c} s{s} {b}: double path greedy mismatch {r['scores'][b]['double_path']['greedy_mismatch']}/24")
            lc = r.get("load_check") or {}
            if abs(lc.get("replica_minus_logged", 0.0)) > 2e-3:
                an.append(f"{c} s{s}: load check |replica - logged| {lc['replica_minus_logged']:+.2e} > 2e-3")
    res["anomalies"] = an
    ident = [x for cp in COMPARISONS for x in check_pair(cells, cp)]
    assert not ident, ("arm identity check failed (Amendment A2)", ident)
    res["checks"]["arm_identity"] = "PASS (target / lr / train code / n_params / arch / file name, every common seed)"
    for cp in COMPARISONS:
        e = dict(cp, stats=paired(cells, cp["prior"], cp["none"], BENCH, boot=True))
        ref = res["reference"][cp["group"]]["cell"]
        e["ref_cell"] = ref
        e["vs_ref"] = paired(cells, cp["prior"], ref, BENCH) if ref else None
        res["comparisons"].append(e)
    for c, v in sorted(cells.items()):
        s03 = [v[s] for s in SEEDS if s in v]
        row = {"seeds": sorted(v), "lr": next(iter(v.values()))["lr"], "d_model": next(iter(v.values()))["d_model"]}
        for b in BENCH:
            xs = [score(r, b) for r in s03]
            row[b] = {"mean03": statistics.fmean(xs) if xs else None, "sd03": statistics.stdev(xs) if len(xs) > 1 else None,
                      "n03": len(xs), "mean_all": statistics.fmean(score(r, b) for r in v.values()), "n_all": len(v)}
        row["lambada_ppl_mean03"] = statistics.fmean(r["scores"]["lambada"]["ppl"] for r in s03) if s03 else None
        res["cells"][c] = row
    # 53M / 190M 7% seed-0 cells: prior - none at the same lr (the cell named with its arm replaced by none), seed 0 only
    for c, v in sorted(cells.items()):
        fam, name = c.split("/", 1)
        if fam not in DESC_FAMILIES or "7pct-" not in name or name.startswith("500m") or 0 not in v:
            continue
        pre, arm = name.split("7pct-", 1)
        if arm in ("none", "untie-none"):
            continue
        nc = f"{fam}/{pre}7pct-" + ("untie-none" if arm.startswith("untie-") else "none")
        if 0 not in cells.get(nc, {}):
            continue
        p, q = v[0], cells[nc][0]
        assert p["d_model"] == q["d_model"], (c, nc)
        if p["lr"] != q["lr"]:   # only same-lr pairs (the fair comparison)
            continue
        # the task: BLiMP + LAMBADA for every size, HellaSwag / ARC-E / PIQA for 190M / 505M (Amendment A1 item 7)
        bl = BENCH if p["d_model"] >= 768 else ("blimp", "lambada")
        res["desc7"].append({"cell": c, "none": nc, "d_model": p["d_model"], "lr": p["lr"],
                             "d": {b: score(p, b) - score(q, b) for b in bl}, "prior": {b: score(p, b) for b in bl},
                             "none_score": {b: score(q, b) for b in bl}})
    return res


# ------------------------------------------------------------------------------------------------------ report
def f(x, k=2):
    return "–" if x is None else f"{x:+.{k}f}" if k < 0 else f"{x:.{k}f}"


def report(res: dict, ka: dict | None = None) -> str:
    L = ["# Downstream read-out (pre-registered rules: runs/downstream/AUDIT.md)", ""]
    ck = res["checks"]
    L += [f"eval code sha(s): {ck['eval_code_shas']}; setups (prefix, max_len, dtype): {ck['setups']}; "
          f"n mismatches: {len(ck['n_mismatch'])}; truncated sequences: {ck['n_trunc']}; max double-path |dloglik|: "
          f"{ck['max_double_path']:.1e}; arm identity: {ck.get('arm_identity')}; records: "
          f"{sum(len(v) for v in res['cells'].values())} models in {len(res['cells'])} cells"
          if ck["max_double_path"] is not None else "no records", ""]
    if ka:
        L += ["## Known-answer checks", ""] + ka["lines"] + [""]
    L += ["## Anomaly flags", ""] + ([f"- {x}" for x in res.get("anomalies", [])] or ["- none"]) + [""]
    for g, r in res["reference"].items():
        L.append(f"Reference ({g}): {r['cell']} (lowest mean bpb over seeds 0-3, source: {r['bpb_source']}; "
                 + ", ".join(f"{c.split('/')[-1]} {v:.4f}" for c, v in r["mean_bpb"].items()) + ")")
    L.append("")
    for role in ("primary", "descriptive"):
        L += [f"## {role.upper()} comparisons (prior - none, same lr, same seeds; points; 95% t-CI; margin 1.0)", "",
              "\"undetermined\" means n = 4 seeds cannot tell; it is NOT \"no cost\". The seed x item bootstrap interval is "
              "descriptive (Amendment A1), not a verdict.", "",
              "| id | comparison | benchmark | prior | none | mean d | 95% t-CI | seed x item bootstrap | wins (ties) | per seed | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in res["comparisons"]:
            if e["role"] != role:
                continue
            for b in BENCH:
                s = e["stats"][b]
                if not s["n"]:
                    L.append(f"| {e['id']} | {e['group']} {e['label']} | {b} | | | | | | | | INCOMPLETE (no records) |")
                    continue
                v = s["verdict"] if role == "primary" else f"({s['verdict']}; descriptive)"
                bt = s.get("boot2")
                L.append(f"| {e['id']} | {e['group']} {e['label']} | {b} {PRIMARY[b]} | {s['prior_mean']:.2f} | "
                         f"{s['none_mean']:.2f} | {s['mean']:+.2f} | [{f(s['lo'])}, {f(s['hi'])}] | "
                         + (f"[{bt['lo']:.2f}, {bt['hi']:.2f}]" if bt else "–") + f" | {s['wins']}/{s['n']} ({s['ties']} ties) | "
                         + " ".join(f"{x:+.2f}" for x in s["d"]) + f" | {v} |")
            p = e["stats"].get("lambada_ppl_ratio")
            if p:
                L.append(f"| {e['id']} | | LAMBADA ppl ratio prior/none (descriptive) | | | x{p['geo_mean']:.3f} | "
                         f"[{f(p['lo'], 3)}, {f(p['hi'], 3)}] | | | " + " ".join(f"{x:.3f}" for x in p["per_seed"]) + " | |")
        L.append("")
    L += ["## Gap to the reference none (LM-optimal lr; prior - reference, paired by seed; lr differs: descriptive)", "",
          "| id | prior cell | reference | " + " | ".join(BENCH) + " |", "|---|---|---|" + "---|" * len(BENCH)]
    for e in res["comparisons"]:
        if e["vs_ref"] is None:
            continue
        cells_ = []
        for b in BENCH:
            s = e["vs_ref"][b]
            cells_.append("–" if not s["n"] else f"{s['mean']:+.2f} [{f(s['lo'])}, {f(s['hi'])}] {s['wins']}/{s['n']}")
        L.append(f"| {e['id']} | {e['prior'].split('/')[-1]} | {e['ref_cell'].split('/')[-1]} | " + " | ".join(cells_) + " |")
    L += ["", "## Every cell (mean over seeds 0-3 +- SD; all-seed mean when more seeds exist)", "",
          "| cell | lr | seeds | " + " | ".join(f"{b} {PRIMARY[b]}" for b in BENCH) + " | LAMBADA ppl |",
          "|---|---|---|" + "---|" * (len(BENCH) + 1)]
    for c, r in res["cells"].items():
        cs = []
        for b in BENCH:
            x = r[b]
            if r["d_model"] < 768 and b not in ("blimp", "lambada"):   # A1 item 7: 53M rows show BLiMP / LAMBADA only
                cs.append("(53M: not reported)"); continue
            t = f"{x['mean03']:.2f} ± {f(x['sd03'])}" if x["mean03"] is not None else "–"
            if x["n_all"] > x["n03"]:
                t += f" (all {x['n_all']}: {x['mean_all']:.2f})"
            cs.append(t)
        L.append(f"| {c} | {r['lr']:g} | {','.join(map(str, r['seeds']))} | " + " | ".join(cs)
                 + f" | {f(r['lambada_ppl_mean03'], 1)} |")
    L += ["", "## 53M / 190M 7% cells, seed 0 only (prior - none at the same lr; single models, no CI, descriptive)", "",
          "| cell | none | lr | " + " | ".join(BENCH) + " |", "|---|---|---|" + "---|" * len(BENCH)]
    for e in res["desc7"]:
        L.append(f"| {e['cell']} | {e['none'].split('/')[-1]} | {e['lr']:g} | "
                 + " | ".join(f"{e['d'][b]:+.2f} ({e['prior'][b]:.1f} vs {e['none_score'][b]:.1f})" if b in e["d"] else "–"
                              for b in BENCH) + " |")
    return "\n".join(L) + "\n"


def eval_sha() -> str:
    """sha of the downstream_eval.py next to this file (the code snapshot the jobs run)."""
    import hashlib
    return hashlib.sha256((Path(__file__).resolve().parent / "downstream_eval.py").read_bytes()).hexdigest()[:12]


def tol(n: int, p: float) -> float:
    """Amendment A1: random-init tolerance = max(3 points, 3 binomial SE at chance p), in points."""
    return max(3.0, 300 * math.sqrt(p * (1 - p) / n))


def cluster_se(hits, eq, uid) -> tuple:
    """Amendment A3: cluster-robust (CR1) SE, in points, of the pooled accuracy on the equal-token pairs, clusters =
    BLiMP paradigms: p = sum_g s_g / N; SE = sqrt(G / (G - 1) * sum_g (s_g - n_g p)^2) / N over the G paradigms with
    >= 1 equal-token pair (s_g hits among their n_g equal-token pairs, N = sum_g n_g). -> (SE, G, N, p in points)."""
    import numpy as np
    hits, eq, uid = np.asarray(hits, bool), np.asarray(eq, bool), np.asarray(uid)
    us = [u for u in dict.fromkeys(uid.tolist()) if (eq & (uid == u)).any()]
    n = np.array([(eq & (uid == u)).sum() for u in us], float)
    s = np.array([(hits & eq & (uid == u)).sum() for u in us], float)
    N, G = n.sum(), len(us)
    p = s.sum() / N
    return 100 * math.sqrt(G / (G - 1) * float(((s - n * p) ** 2).sum())) / N, G, int(N), 100 * p


def blimp_cluster_se(r: dict) -> tuple:
    """A3: rebuild each pair's paradigm and the equal-token mask exactly as downstream_eval does (sorted paradigm files,
    1000 pairs each; a BLiMP sentence's scored tokens = its tokens without special tokens, the prefix <eos> is not scored)
    with the random-init run's tokenizer; cross-check against the stored n_equal_len / acc_equal_len; -> cluster_se."""
    import glob
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(json.loads(Path(r["arch_from"]).read_text())["args"]["tokenizer"])
    files = sorted(glob.glob(str(Path(r["data"]) / "blimp/*/*.parquet")))
    assert len(files) == 67, len(files)
    eq, uid = [], []
    for f in files:
        df = pd.read_parquet(f)
        g = tok(df.sentence_good.tolist(), add_special_tokens=False)["input_ids"]
        b = tok(df.sentence_bad.tolist(), add_special_tokens=False)["input_ids"]
        eq += [len(x) == len(y) for x, y in zip(g, b)]; uid += df.UID.tolist()
    bl = r["scores"]["blimp"]
    hits = np.frombuffer(bl["hits"].encode(), np.uint8) == 49
    eq = np.array(eq)
    assert len(hits) == len(eq) == 67000, (len(hits), len(eq))
    assert int(eq.sum()) == bl["n_equal_len"], ("equal-token mask differs from the eval's", int(eq.sum()), bl["n_equal_len"])
    se, G, N, p = cluster_se(hits, eq, uid)
    assert abs(p - bl["acc_equal_len"]) < 1e-9, ("rebuilt equal-token accuracy differs from the stored one", p, bl["acc_equal_len"])
    return se, G, N, p


LOAD_GATE = 2e-3   # bpb, |in-training replica - logged final| for the load-check-only models (Amendment A1)
LOADCHECK_NAMES = ("500m-lr1.2e-3-7pct-none_s0", "53m-lr4e-3-7pct-none_s0", "500m-7pct-none_s0", "7pct-none_s0",
                   "rb6tb-190m-2pct-k2-none-lr6e-4_s0")   # = make_downstream_jobs.LOADCHECK (asserted there)


def known_answer(out_dir: Path, sha: str | None = None, blimp_se=blimp_cluster_se) -> dict:
    """Pass / fail of the pre-registered known-answer criteria (AUDIT.md + Amendment A1). Only full-run files (limit 0)
    written by the current eval code count (sha None: the sha of the downstream_eval.py next to this file)."""
    lines, ok = [], True
    ka = out_dir / "known_answer"
    sha = eval_sha() if sha is None else sha

    def get(p: Path):
        if not p.exists():
            lines.append(f"- MISSING {p.name}"); return None
        r = json.loads(p.read_text())
        if r.get("eval_code_sha") != sha or r.get("limit"):
            lines.append(f"- STALE {p.name} (eval code {r.get('eval_code_sha')} != {sha}, or a smoke run)"); return None
        return r
    r = get(ka / "hf_gpt2_harness.json")
    if r is None:
        ok = False
    else:
        s = r["scores"]
        c = [("GPT-2 LAMBADA acc within 32.56 +- 1.0", abs(s["lambada"]["acc"] - 32.56) <= 1.0, s["lambada"]["acc"]),
             ("GPT-2 LAMBADA ppl within 40.06 +- 5%", abs(s["lambada"]["ppl"] / 40.0554 - 1) <= 0.05, s["lambada"]["ppl"]),
             ("GPT-2 BLiMP in [81, 87]", 81 <= s["blimp"]["acc"] <= 87, s["blimp"]["acc"])]
        for name, p, v in c:
            ok &= p; lines.append(f"- {'PASS' if p else 'FAIL'} {name}: {v:.2f}")
        lines.append("- GPT-2 (harness prefix) descriptive: " + ", ".join(
            f"{b} {s[b][PRIMARY[b]]:.2f}" for b in ("hellaswag", "arc_easy", "piqa")))
    r = get(ka / "hf_gpt2_eos.json")   # descriptive only (not gating)
    if r is not None:
        s = r["scores"]
        lines.append("- GPT-2 (eos prefix, the project setting) descriptive: " + ", ".join(
            f"{b} {s[b][PRIMARY[b]]:.2f}" for b in BENCH) + f", LAMBADA ppl {s['lambada']['ppl']:.2f}")
    for name in ("randinit_d384_L6_eos.json", "randinit_d768_L12_eos.json", "randinit_d1024_L24_eos.json"):
        r = get(ka / name)
        if r is None:
            ok = False; continue
        s = r["scores"]
        hs, ar, pq, bl = s["hellaswag"], s["arc_easy"], s["piqa"], s["blimp"]
        cse, G, _, _ = blimp_se(r)   # A3 (post hoc): pairs share their paradigm's word contrast -> paradigm-clustered SE
        c = [("HellaSwag acc_norm", hs["acc_norm"], 25.0, tol(hs["n"], 0.25)),
             ("ARC-E acc", ar["acc"], ar["chance"], tol(ar["n"], ar["chance"] / 100)),
             ("PIQA acc, equal-token pairs (A1)", pq["acc_equal_len"], 50.0, tol(pq["n_equal_len"], 0.5)),
             (f"BLiMP equal-token pairs (A3: max(3, 3 x cluster SE {cse:.2f}, G {G}))", bl["acc_equal_len"], 50.0,
              max(3.0, 3 * cse))]
        res_ = [(f"{n} (|x - {ch:.1f}| <= {t:.1f})", v is not None and abs(v - ch) <= t, v) for n, v, ch, t in c]
        res_.append(("LAMBADA acc <= 1", s["lambada"]["acc"] <= 1.0, s["lambada"]["acc"]))
        ok &= all(p_ for _, p_, _ in res_)
        lines.append(f"- {name[:-5]}: " + "; ".join(f"{'PASS' if p_ else 'FAIL'} {n} {v:.2f}" for n, p_, v in res_)
                     + f"; descriptive: PIQA all pairs {pq['acc']:.2f} (fewest-tokens heuristic "
                     f"{pq['length_heuristic']['fewest_tokens_acc']:.2f}, n equal {pq['n_equal_len']}), BLiMP all pairs "
                     f"{bl['acc']:.2f} (shorter-good share {bl['length_heuristic']['shorter_good_share']:.1f}, n equal "
                     f"{bl['n_equal_len']})")
    # A2: the full expected set (one model per training-code version, make_downstream_jobs.LOADCHECK); a missing one fails
    lcs = [ka / f"loadcheck_{n}.json" for n in LOADCHECK_NAMES]
    for p in lcs:
        r = get(p)
        if r is None:
            ok = False; continue
        d = r["load_check"]["replica_minus_logged"]
        ok &= abs(d) < LOAD_GATE
        lines.append(f"- {'PASS' if abs(d) < LOAD_GATE else 'FAIL'} load check {p.stem} (train code {r.get('train_code_sha')}): "
                     f"in-training bpb replica - logged = {d:+.2e} (|.| < {LOAD_GATE:g})")
    lines.append(f"- KNOWN-ANSWER OVERALL (eval code {sha}): {'PASS' if ok else 'FAIL'}")
    return {"pass": ok, "lines": lines}


# ------------------------------------------------------------------------------------------------------ selftest
def _fake_rec(model, seed, lr, d_model, sc, bpb=1.0, ppl=100.0, chance_arc=25.03):
    S = {}
    for b in BENCH:
        S[b] = {"n": EXPECT_N[b], PRIMARY[b]: sc[b], "double_path": {"max_abs_dloglik": 1e-5}}
    S["lambada"]["ppl"] = ppl
    S["arc_easy"]["chance"] = chance_arc
    return {"kind": "project", "limit": 0, "model": model, "run_json": model.replace("model_", "run_").replace(".pt", ".json"),
            "seed": seed, "lr": lr, "d_model": d_model, "eval_code_sha": "fake", "prefix": "eos", "max_len": 512,
            "target": "none" if "_none_seed" in model else "helix3", "train_code_sha": "t", "n_params": d_model, "n_layers": 1,
            "dtype": "fp32", "val_bpb_logged_final": bpb, "scores": S,
            "tok_stats": {b: {"n_trunc": 0} for b in BENCH}}


def selftest() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="downstream_selftest_"))
    try:
        out = tmp / "out"; runs = tmp / "runs"
        base = {"blimp": 70.0, "lambada": 15.0, "hellaswag": 30.0, "arc_easy": 40.0, "piqa": 60.0}

        def put(fam, cell, seed, lr, dm, sc, bpb=1.0, ppl=100.0, offline=None):
            md = runs / fam / "out" / cell; md.mkdir(parents=True, exist_ok=True)
            mp = str(md / f"model_whole_{'none' if 'none' in cell.split('-') else 'helix3'}_seed{seed}_{cell}.pt")
            r = _fake_rec(mp, seed, lr, dm, sc, bpb, ppl)
            (out / fam).mkdir(parents=True, exist_ok=True)
            (out / fam / f"{Path(mp).stem}.json").write_text(json.dumps(r))
            if offline is not None:   # offline_lm.json in the model dir, keyed by the resolved run json
                p = md / "offline_lm.json"
                d = json.loads(p.read_text()) if p.exists() else {}
                d[str(Path(r["run_json"]).resolve())] = {"bpb_full": offline, "max_windows": 0}
                p.write_text(json.dumps(d))

        # P1: non-inferior on blimp (d = +0.1 .. +0.4), worse on lambada (d ~ -3, tight), undetermined on hellaswag (noisy),
        #     at chance on piqa (both ~50.5), arc: small cost within margin (d ~ -0.3 tight)
        dP1 = {"blimp": [0.1, 0.2, 0.3, 0.4], "lambada": [-3.0, -3.2, -2.8, -3.1], "hellaswag": [3.0, -4.0, 2.0, -5.0],
               "arc_easy": [-0.3, -0.25, -0.35, -0.3], "piqa": [0.5, -0.5, 0.2, 0.1]}
        for s in range(4):
            nn = dict(base, piqa=50.5); pr = {b: nn[b] + dP1[b][s] for b in BENCH}
            put("ptlr", "500m-lr1.2e-3-7pct-none", s, 1.2e-3, 1024, nn, bpb=1.10, ppl=100)
            put("ptlr", "500m-lr1.2e-3-7pct-win-ctx", s, 1.2e-3, 1024, pr, ppl=110)
        # P2: seed 3 of the prior missing -> INCOMPLETE everywhere
        for s in range(4):
            put("ptsize999", "500m-7pct-none", s, 3e-4, 1024, base, bpb=1.05)
            if s < 3:
                put("ptsize999", "500m-7pct-win-ctx", s, 3e-4, 1024, base)
        for cell, bp in (("500m-lr4.8e-3-7pct-none", 1.07), ("500m-lr9.6e-3-7pct-none", 1.08), ("500m-lr1.92e-2-7pct-none", 1.2)):
            for s in range(4):
                put("ptlr", cell, s, 0.0, 1024, base, bpb=bp)
        # 190M: offline bpb present for every seed of both nones: the reference must follow it (4.8e-3 lower offline,
        # 6e-4 lower logged)
        for s in range(4):
            put("rb6_190m", "rb6tb-190m-2pct-k2-none-lr6e-4", s, 6e-4, 768, base, bpb=1.00, offline=1.20)
            put("rb6_190m", "rb6tb-190m-2pct-k2-none-lr4.8e-3", s, 4.8e-3, 768, base, bpb=1.10, offline=1.15)
            put("rb6_190m", "rb6tb-190m-2pct-k2-all6-lam1-lr6e-4", s, 6e-4, 768, {b: base[b] - 1.5 + 0.1 * s for b in BENCH})
        # desc 53M seed 0
        put("ptlr", "53m-lr4e-3-7pct-none", 0, 4e-3, 384, base)
        put("ptlr", "53m-lr4e-3-7pct-win-ctx", 0, 4e-3, 384, dict(base, blimp=71.0, hellaswag=20.0))
        res = analyse(load(out))
        st = {e["id"]: e for e in res["comparisons"]}
        v = {b: st["P1"]["stats"][b]["verdict"] for b in BENCH}
        assert v["blimp"] == "non-inferior", v
        assert v["lambada"] == "worse", v
        assert v["hellaswag"] == "undetermined", v
        assert v["arc_easy"].startswith("non-inferior (small cost"), v
        assert v["piqa"] == "at chance: uninformative", v
        assert abs(st["P1"]["stats"]["lambada"]["mean"] - (-3.025)) < 1e-9
        assert st["P1"]["stats"]["blimp"]["wins"] == 4 and st["P1"]["stats"]["lambada"]["wins"] == 0
        m, sd, lo, hi = tci([0.1, 0.2, 0.3, 0.4])
        assert abs(lo - (0.25 - 3.1824 * statistics.stdev([0.1, 0.2, 0.3, 0.4]) / 2)) < 1e-9
        assert abs(st["P1"]["stats"]["lambada_ppl_ratio"]["geo_mean"] - 1.1) < 1e-9
        assert all(st["P2"]["stats"][b]["verdict"] == "INCOMPLETE" for b in BENCH), st["P2"]["stats"]
        assert st["P3"]["stats"]["blimp"]["verdict"] == "worse", st["P3"]["stats"]["blimp"]   # d = -1.5..-1.2, tight
        assert all(st[k]["stats"]["blimp"]["n"] == 0 for k in ("D1", "D2", "D3"))
        assert res["reference"][G505]["cell"] == "ptsize999/500m-7pct-none", res["reference"][G505]
        assert res["reference"][G505]["bpb_source"].startswith("logged")
        assert res["reference"][G190]["cell"] == "rb6_190m/rb6tb-190m-2pct-k2-none-lr4.8e-3", res["reference"][G190]
        assert res["reference"][G190]["bpb_source"] == "offline bpb_full"
        assert st["P1"]["ref_cell"] == "ptsize999/500m-7pct-none"
        assert len(res["desc7"]) == 1 and abs(res["desc7"][0]["d"]["blimp"] - 1.0) < 1e-9 and "hellaswag" not in res["desc7"][0]["d"]
        txt = report(res)
        assert "| P1 |" in txt and "INCOMPLETE" in txt and "worse" in txt
        # A2 arm identity: a swapped target, a different lr / training code or a file name without the run's target is caught
        cells_ = load(out)
        assert not check_pair(cells_, COMPARISONS[0])
        x = json.loads(json.dumps(cells_)); x["ptlr/500m-lr1.2e-3-7pct-win-ctx"]["1"]["target"] = "none"
        assert check_pair({c: {int(k): v for k, v in d.items()} for c, d in x.items()}, COMPARISONS[0])
        x = json.loads(json.dumps(cells_)); x["ptlr/500m-lr1.2e-3-7pct-none"]["2"]["lr"] = 3e-4
        assert check_pair({c: {int(k): v for k, v in d.items()} for c, d in x.items()}, COMPARISONS[0])
        x = json.loads(json.dumps(cells_)); x["ptlr/500m-lr1.2e-3-7pct-none"]["0"]["model"] = "/x/model_whole_helix3_seed0_y.pt"
        assert check_pair({c: {int(k): v for k, v in d.items()} for c, d in x.items()}, COMPARISONS[0])
        x = json.loads(json.dumps(cells_)); x["ptlr/500m-lr1.2e-3-7pct-none"]["3"]["train_code_sha"] = "other"
        assert check_pair({c: {int(k): v for k, v in d.items()} for c, d in x.items()}, COMPARISONS[0])
        # anomaly flag: the 53M win-ctx cell's HellaSwag below chance - 3
        assert any("53m-lr4e-3-7pct-win-ctx" in x and "hellaswag" in x for x in res["anomalies"]), res["anomalies"]
        assert not any("500m" in x for x in res["anomalies"]), res["anomalies"]
        # seed x item bootstrap: A right on the first 600 of 1000 items, B on the first 590 -> d = +1.0 at every seed
        A = {s: {"scores": {"piqa": {"acc": 60.0, "hits_acc": "1" * 600 + "0" * 400}}} for s in range(4)}
        B = {s: {"scores": {"piqa": {"acc": 59.0, "hits_acc": "1" * 590 + "0" * 410}}} for s in range(4)}
        bt = boot2(A, B, list(range(4)), "piqa", nboot=500)
        assert bt["lo"] < 1.0 < bt["hi"] and bt["hi"] - bt["lo"] < 1.5, bt
        A[2]["scores"]["piqa"]["acc"] = 61.0   # hit string no longer reproduces the stored score -> assert
        try:
            boot2(A, B, list(range(4)), "piqa", nboot=10); raise RuntimeError("boot2 accepted inconsistent hits")
        except AssertionError:
            pass
        # known-answer gate: passing fake files -> PASS; one criterion broken at a time -> FAIL; stale / smoke files -> FAIL
        kd = out / "known_answer"; kd.mkdir()
        g2 = {"lambada": {"acc": 32.9, "ppl": 41.5}, "blimp": {"acc": 83.0}, "hellaswag": {"acc_norm": 31.0},
              "arc_easy": {"acc": 43.0}, "piqa": {"acc": 62.0}}
        ri = {"hellaswag": {"acc_norm": 26.0, "n": 10042}, "arc_easy": {"acc": 23.0, "chance": 25.03, "n": 2376},
              "piqa": {"acc": 53.0, "acc_equal_len": 51.0, "n_equal_len": 781, "n": 1838,
                       "length_heuristic": {"fewest_tokens_acc": 53.9}},
              "blimp": {"acc": 58.0, "acc_equal_len": 49.0, "n_equal_len": 30000,
                        "length_heuristic": {"shorter_good_share": 30.0, "equal_len_share": 50.0}},
              "lambada": {"acc": 0.0, "ppl": 9e5}}
        RI = ("randinit_d384_L6_eos.json", "randinit_d768_L12_eos.json", "randinit_d1024_L24_eos.json")
        lc = {"replica_minus_logged": 3e-8}

        # A3 cluster SE: hand-computed case (2 paradigms; equal-token pairs 4 + 2; hits 4/4 and 0/2 -> p = 2/3)
        se, G, N, p = cluster_se([1, 1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 0], ["a"] * 4 + ["b"] * 3)
        want = 100 * math.sqrt(2 / 1 * ((4 - 4 * 2 / 3) ** 2 + (0 - 2 * 2 / 3) ** 2)) / 6
        assert (G, N) == (2, 6) and abs(p - 200 / 3) < 1e-9 and abs(se - want) < 1e-9, (se, want, G, N, p)
        FAKE_SE = lambda r: (2.0, 67, 30000, r["scores"]["blimp"]["acc_equal_len"])   # tol = max(3, 6.0) = 6.0

        def wr(name, s, sha="fake", limit=0, key="scores"):
            (kd / name).write_text(json.dumps({key: s, "eval_code_sha": sha, "limit": limit}))
        wr("hf_gpt2_harness.json", g2); [wr(n, ri) for n in RI]; [wr(f"loadcheck_{n}.json", lc, key="load_check") for n in LOADCHECK_NAMES]
        assert known_answer(out, sha="fake", blimp_se=FAKE_SE)["pass"], known_answer(out, sha="fake", blimp_se=FAKE_SE)["lines"]
        assert abs(tol(781, 0.5) - 300 * math.sqrt(0.25 / 781)) < 1e-12 and tol(10042, 0.25) == 3.0
        for name, s, path, bad in (("hf_gpt2_harness.json", g2, ("lambada", "acc"), 30.0),
                                   ("hf_gpt2_harness.json", g2, ("lambada", "ppl"), 43.0),
                                   ("hf_gpt2_harness.json", g2, ("blimp", "acc"), 75.0),
                                   (RI[1], ri, ("piqa", "acc_equal_len"), 56.0),   # tol(781) = 5.37
                                   (RI[2], ri, ("hellaswag", "acc_norm"), 28.5),
                                   (RI[0], ri, ("arc_easy", "acc"), 21.0),
                                   (RI[0], ri, ("blimp", "acc_equal_len"), 43.9),   # A3: tol = max(3, 3 x 2.0) = 6.0
                                   (RI[0], ri, ("lambada", "acc"), 1.5)):
            s2 = json.loads(json.dumps(s)); s2[path[0]][path[1]] = bad; wr(name, s2)
            assert not known_answer(out, sha="fake", blimp_se=FAKE_SE)["pass"], (name, path, bad)
            wr(name, s)
        s2 = json.loads(json.dumps(ri)); s2["piqa"]["acc"] = 54.0; wr(RI[0], s2)   # all-pairs PIQA is descriptive now (A1)
        assert known_answer(out, sha="fake", blimp_se=FAKE_SE)["pass"]
        wr(RI[0], ri)
        wr(f"loadcheck_{LOADCHECK_NAMES[2]}.json", {"replica_minus_logged": 2.5e-3}, key="load_check")
        assert not known_answer(out, sha="fake", blimp_se=FAKE_SE)["pass"]
        wr(f"loadcheck_{LOADCHECK_NAMES[2]}.json", lc, key="load_check")
        (kd / f"loadcheck_{LOADCHECK_NAMES[4]}.json").unlink(); assert not known_answer(out, sha="fake", blimp_se=FAKE_SE)["pass"]   # A2: 4 of 5
        wr(f"loadcheck_{LOADCHECK_NAMES[4]}.json", lc, key="load_check")
        wr(RI[1], ri, sha="old"); assert not known_answer(out, sha="fake", blimp_se=FAKE_SE)["pass"]; wr(RI[1], ri)
        wr(RI[1], ri, limit=20); assert not known_answer(out, sha="fake", blimp_se=FAKE_SE)["pass"]; wr(RI[1], ri)
        (kd / RI[2]).unlink(); assert not known_answer(out, sha="fake", blimp_se=FAKE_SE)["pass"]; wr(RI[2], ri)
        assert known_answer(out, sha="fake", blimp_se=FAKE_SE)["pass"]
        print("---- report on the FAKE selftest data ----\n" + txt)
        print("SELFTEST PASS")
    finally:
        shutil.rmtree(tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/scratch/yh6210/llmadd_torch/runs/downstream/out")
    ap.add_argument("--write", default=None, help="prefix for <prefix>.md / <prefix>.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ka_gate", action="store_true", help="exit 1 unless every pre-registered known-answer check passes")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    out = Path(a.out_dir)
    ka = known_answer(out)
    if a.ka_gate:
        print("\n".join(ka["lines"]))
        raise SystemExit(0 if ka["pass"] else 1)
    res = analyse(load(out))
    res["known_answer"] = ka
    txt = report(res, ka)
    print(txt)
    if a.write:
        import os
        for ext, body in ((".json", json.dumps(res, indent=1, default=str)), (".md", txt)):   # atomic (A2)
            tmp = Path(a.write + ext + f".tmp{os.getpid()}"); tmp.write_text(body); tmp.replace(a.write + ext)
        print("wrote", a.write + ".md", a.write + ".json")


if __name__ == "__main__":
    main()
