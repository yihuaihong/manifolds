"""Summary of the unified 0..999 batches (runs/pt999, runs/ptsize999; tools/pretrain_range.py).

One cell per --tag (audit 2026-10-14: ptrange_summary.py groups by tokenization / target / schedule, which merges the
ctx, text, 2%, sensitivity and size cells, and a later seed silently replaced an earlier one). Per cell:
  n       finished seeds (unfinished runs are listed and left out)
  area    mean over evals of the 6-bucket mean held-out exact match (x100); also per bucket in the second table
  t90     tokens (M) until the 6-bucket mean first reaches 0.9, linear interpolation between evals (median over the
          seeds that reach it; "k/n" = how many reach it)
  final   held-out exact match per bucket at the end; b13 / b23 / b33 also split by sum < 1000 (one answer token) and
          >= 1000 (first answer chunk "1xx"), audit item on tied embeddings
  bpb     val bits per byte at the end;  used = share of steps on which the prior acted;  cka_ops / cka_emb = helix3 CKA
          of the operand states (fixed read-out layers) / of the number tokens' embedding rows, at the end
Paired comparisons (same seed = same init, data stream and held-out split): every cell vs its none cell (same arithmetic
share / optimisation setting; lambda and layer cells vs 7pct-none), and gate vs shuffled / lenshuf / whole-run at the
same site. Reported: mean difference, paired t-test p, wins = seeds where the arm is higher.

    python tools/pt999_summary.py /scratch/yh6210/llmadd_torch/runs/pt999/out [/scratch/yh6210/llmadd_torch/runs/ptsize999/out]
Burst results (2026-10-16, pt2main): a returned task dir results/<task>/ holds <tag>/run_*.json, so pass the task dirs,
e.g. results/3[5-9][0-9]-pt2*; tools/pt2main_summary.py does that and applies the batch's pre-registered rules.
The shared read-out rules (diverged, best lr, survives tuning, "N x") are tools/readout_rules.py; area and t90 here stay
an independent implementation on purpose (pt2main_summary / ptscale_summary check their cell areas against this output).
"""
import collections
import glob
import json
import sys

import numpy as np
from scipy import stats

B = ("b11", "b12", "b13", "b22", "b23", "b33")
ARMS = ("untie-gate-ctx", "untie-none", "lenshuf-ctx", "whole-ctx", "win-text", "gate-ctx", "gate-text",
        "win-shuf-ctx",   # pt2main (window + helix3_shuf on the operands); before shuf-ctx, which it ends with
        "win-shuf-text",   # pt999 2pct (window + helix3_shuf on text numbers); before shuf-text, which it ends with
        "win-ctx-lam0.3", "win-ctx-lam1", "win-ctx-lam2",   # pt2main lambda sweep (2026-10-17)
        "shuf-ctx", "shuf-text",
        "win-prefix-shuf", "win-prefix-lam1.7",
        "win-ctx", "whole-prefix", "win-prefix", "gate-prefix", "win0-text", "win10-text", "win20-text",
        "none")

cells = collections.defaultdict(dict)
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/*/run_*.json")):
        r = json.load(open(f))
        a = r["args"]
        tag = a["tag"].lstrip("_")
        if r["curve"][-1]["step"] < r["steps"] - 1:
            print("unfinished:", f)
            continue
        assert a["seed"] not in cells[tag], f"two runs for {tag} seed {a['seed']}: {f}"
        cells[tag][a["seed"]] = r


def held(e):
    return np.mean([e[f"held_{b}"] for b in B])


def area(r, b=None):
    return 100 * np.mean([(e[f"held_{b}"] if b else held(e)) for e in r["curve"]])


def t90(r):
    prev, per_step = None, r["args"]["bs"] * r["args"]["seq"]
    for e in r["curve"]:
        v, tk = held(e), e["step"] * per_step
        if v >= 0.9:
            if prev is None:
                return tk / 1e6
            t0, v0 = prev
            return (t0 + (0.9 - v0) / (v - v0) * (tk - t0)) / 1e6
        prev = (tk, v)
    return np.nan


def split_arm(tag):
    arm = next(x for x in ARMS if tag == x or tag.endswith("-" + x))
    return tag[: len(tag) - len(arm)].rstrip("-"), arm


def reference(tag):
    pre, arm = split_arm(tag)
    if arm in ("none", "untie-none"):
        return None
    ref = f"{pre}-{'untie-none' if arm.startswith('untie') else 'none'}"
    if ref not in cells and pre.startswith("sens-"):   # lambda / layer cells: none does not depend on them
        ref = "7pct-none"
    return ref


def paired(t1, t2, f):
    s = sorted(set(cells[t1]) & set(cells[t2]))
    if len(s) < 2:
        return f"n={len(s)}"
    x = np.array([f(cells[t1][k]) for k in s]); y = np.array([f(cells[t2][k]) for k in s])
    p = stats.ttest_rel(x, y).pvalue if np.std(x - y) > 0 else float("nan")
    return f"{np.mean(x - y):+6.1f} (p {p:.3f}, wins {int((x > y).sum())}/{len(s)})"


def fin(r, k):
    v = r["curve"][-1].get(k)
    return np.nan if v is None else v


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return np.mean(xs) if xs else np.nan


print("== cells: area, t90 (M tokens), final held-out per bucket (x100), b33 by sum < / >= 1000, bpb, prior use, CKA")
for tag in sorted(cells):
    rs = list(cells[tag].values())
    ar = [area(r) for r in rs]
    tt = [t90(r) for r in rs]
    reach = [t for t in tt if np.isfinite(t)]
    fb = " ".join(f"{b}:{100 * mean([fin(r, f'held_{b}') for r in rs]):3.0f}" for b in B)
    sp = "/".join(f"{100 * mean([fin(r, f'held_b33_{k}') for r in rs]):3.0f}" for k in ("lt1000", "ge1000"))
    line = (f"  {tag:24s} n={len(rs)} area {np.mean(ar):5.1f}±{np.std(ar):4.1f} t90 "
            f"{(f'{np.median(reach):5.0f}' if reach else '    -')} ({len(reach)}/{len(rs)}) | {fb} | b33 {sp} | "
            f"bpb {mean([fin(r, 'val_bits_per_byte') for r in rs]):.4f} used {mean([fin(r, 'prior_used_frac') for r in rs]):.2f} "
            f"cka_ops {mean([fin(r, 'cka_ops_helix3') for r in rs]):.2f} cka_emb {mean([fin(r, 'cka_emb_helix3') for r in rs]):.2f}")
    ref = reference(tag)
    if ref and ref in cells:
        line += f" | vs {ref}: Δarea {paired(tag, ref, area)} Δbpb {paired(tag, ref, lambda r: 1000 * fin(r, 'val_bits_per_byte'))} (x1e-3)"
    print(line)

print("\n== area per bucket (x100)")
for tag in sorted(cells):
    rs = list(cells[tag].values())
    print(f"  {tag:24s} " + " ".join(f"{b}:{np.mean([area(r, b) for r in rs]):5.1f}" for b in B))

print("\n== paired Δarea: gate vs shuffled / lenshuf / whole-run; site at the same schedule; text prior onset")
for a, b in (("gate-ctx", "shuf-ctx"), ("gate-text", "shuf-text"), ("gate-ctx", "lenshuf-ctx"), ("whole-ctx", "gate-ctx"),
             ("gate-ctx", "gate-text"), ("win-text", "gate-text"),
             # site at the same schedule, and the onset of the text prior
             ("whole-ctx", "whole-prefix"), ("win-ctx", "win-prefix"), ("gate-ctx", "gate-prefix"), ("win-ctx", "win-text"),
             ("win-text", "win-prefix"), ("gate-text", "gate-prefix"),
             ("win-text", "win0-text"), ("win-text", "win10-text"), ("win-text", "win20-text"),
             ("win-prefix", "win-prefix-shuf"), ("win-ctx", "win-prefix-lam1.7"), ("win-prefix-lam1.7", "win-prefix"),
             ("win-ctx", "win-shuf-ctx"), ("win-text", "win-shuf-text")):
    for pre in sorted({split_arm(t)[0] for t in cells}):
        t1, t2 = f"{pre}-{a}", f"{pre}-{b}"
        if t1 in cells and t2 in cells:
            print(f"  {t1:24s} - {t2:24s} {paired(t1, t2, area)}")
