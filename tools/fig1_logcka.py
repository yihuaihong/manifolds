"""Figure 1, all seeds from existing logs (2026-10-22; no new runs): the CKA curves that pretrain_range.py logs at every
eval, for F86's pt2b cells at their area-best lr (none@8e-3, win-ctx@2e-3, seeds 0-7; Burst results/<task>/<tag>/run_*.json),
at the primary eval of each Figure 1 checkpoint (the nearest; at the window edges 916 / 7324 the latest eval <= S) and the
eval on the other side of it, plus the full curves.
Logged keys (pretrain_range.py, eval step s):
  cka_ops_helix3  mean over read_layers [2, 4] of pretrain_helix.cka(states at the operand positions of step s's TRAINING
                  batch, centred_gram(helix3(operand values))): 32 rows x 1 line x 2 operands = 64 positions (the operand's
                  last = only token), each line after its row's FineWeb text in one of the 4 formats; the forward of step s
                  = the state after s updates; bf16 autocast
  cka_emb_helix3  cka(input-embedding rows of the number tokens 0..999, K1000) after step s's update (state after s + 1
                  updates); tied embeddings: = cka_unemb_helix3
  held6           mean of the 6 held-out bucket accuracies (the area's per-eval value), after step s's update
    python tools/fig1_logcka.py --results <llmadd-burst results dir> --out <dir> [--extra <run json> ...]"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

CKPT = (0, 916, 3662, 7324, 9324, 18310)   # the export's update counts (the state after S updates)
CELLS = (("none", "8e-3"), ("win-ctx", "2e-3"))
B6 = ("b11", "b12", "b13", "b22", "b23", "b33")


EDGES = (916, 7324)   # the window's first on / first off step (prior_events of the prior arms)


def nearest(steps, S):
    """the primary logged eval for the state after S updates: at a window edge (S = 916 / 7324: the state the regime
    BEFORE S produced) the latest eval <= S, so the eval is in the same regime (audit 2026-10-22: the nearest eval to
    916 is 1000, already 84 prior updates into the window); otherwise the nearest eval (ties: the earlier); S = 18310 ->
    the last eval 18309"""
    if S in EDGES:
        return max(s for s in steps if s <= S)
    return min(steps, key=lambda s: (abs(s - S), s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="label=run json: extra curves (e.g. the Torch re-runs)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = []
    for arm, lr in CELLS:
        tag = f"pt2b-lr{lr}-{arm}"
        for p in sorted(glob.glob(f"{a.results}/*-{tag}-s[0-9]/{tag}/run_*.json")):
            r = json.loads(Path(p).read_text())
            assert r["args"]["tag"] == f"_{tag}" and r["args"]["arith_lines"] == 1 and r["steps"] == 18310 and r["read_layers"] == [2, 4]
            assert r["curve"][-1]["step"] == 18309 and len(r["curve"]) == 75, p
            runs.append((f"burst:{arm}@{lr}", arm, lr, int(r["args"]["seed"]), r, p))
    for e in a.extra:
        lab, p = e.split("=", 1)
        r = json.loads(Path(p).read_text())
        runs.append((lab, lab, f"{r['args']['lr']:g}", int(r["args"]["seed"]), r, p))
    with open(out / "fig1_logged_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "arm", "lr", "seed", "step", "tokens", "cka_ops_helix3", "cka_emb_helix3", "held6", "geo", "prior_on", "file"])
        for lab, arm, lr, sd, r, p in runs:
            for e in r["curve"]:
                w.writerow([lab, arm, lr, sd, e["step"], e["tokens"], e.get("cka_ops_helix3"), e.get("cka_emb_helix3"),
                            float(np.mean([e[f"held_{b}"] for b in B6])), e.get("geo"), e.get("prior_on"), p])
    rows = []
    for S in CKPT:
        for lab in dict.fromkeys(x[0] for x in runs):
            sel = [x for x in runs if x[0] == lab]
            st = sorted(e["step"] for e in sel[0][4]["curve"])
            s0 = nearest(st, S)
            # the nearest eval, and the eval on the other side of S (the window edges 916 / 7324 fall between evals)
            other = [max([s for s in st if s <= S], default=None), min([s for s in st if s >= S], default=None)]
            for s in [s0] + [s for s in dict.fromkeys(other) if s is not None and s != s0]:
              for key in ("cka_ops_helix3", "cka_emb_helix3", "held6"):
                v = []
                for _, _, _, sd, r, _ in sel:
                    e = {q["step"]: q for q in r["curve"]}[s]
                    v.append(float(np.mean([e[f"held_{b}"] for b in B6])) if key == "held6" else e[key])
                e0 = {q["step"]: q for q in sel[0][4]["curve"]}[s]
                rows.append({"S": S, "eval_step": s, "primary": int(s == s0), "prior_on_at_eval": e0.get("prior_on"),
                             "state_updates": s if key == "cka_ops_helix3" else s + 1, "source": lab, "key": key, "n_seeds": len(v),
                             "seeds": ",".join(str(x[3]) for x in sel), "mean": float(np.mean(v)), "min": float(min(v)),
                             "max": float(max(v)), "values": ";".join(f"{x:.4f}" for x in v)})
    with open(out / "fig1_logged_cka_at_ckpts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"S {r['S']:5d} eval {r['eval_step']:5d}{'*' if r['primary'] else ' '} {r['source']:28s} {r['key']:15s} "
              f"n {r['n_seeds']} mean {r['mean']:.3f} [{r['min']:.3f}, {r['max']:.3f}]")
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
