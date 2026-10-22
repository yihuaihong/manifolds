"""Read-out of ft999 (2026-10-22): LoRA fine-tuning of OLMo-2-7B on three-digit addition (operands 0..999, sums 0..1998)
with the helix3 prior over 0..999 (manifold_train.py --add3), arms none / helix3 / helix3_shuf, 3 seeds, two checkpoints.

PRIMARY checkpoint = 9B (stage1-step2000-tokens9B); SECONDARY = 42B (stage1-step10000-tokens42B), reported beside it and
never swapped in for the primary (pre-registered in runs/ft999/AUDIT.md).

Per checkpoint, arm and split (all = every held-out pair, lt1000 = sum < 1000, ge1000 = sum >= 1000):
  area   = mean of the held-out accuracy over the curve evaluations with step > 0 (every 50 steps; step 0 = base is
           excluded because it is identical for every arm)
  final  = held-out accuracy after the last step (the final evaluation of the run)
  to50 / to90 = first evaluated step at which the seed-mean curve reaches 0.5 / 0.9 (descriptive)
Paired comparisons (helix3 - none, helix3 - helix3_shuf; seed s of both arms shares data order and adapter init): the
per-seed differences, their mean, a paired t-test (df = n - 1 = 2) and the number of seeds with a positive difference.
With n = 3 seeds the smallest possible two-sided sign-test p is 0.25 and the t-test has 2 degrees of freedom: the
comparisons are descriptive; a claim needs all 3 seeds in one direction AND a difference far outside the seed spread.

    python tools/ft999_summary.py [--root /scratch/yh6210/llmadd_torch/runs/ft999/out]
    python tools/ft999_summary.py --selftest      # fake curves with known answers
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

CKPTS = (("9B", "PRIMARY"), ("42B", "secondary"))
ARMS = ("none", "helix3", "helix3_shuf")
SEEDS = (0, 1, 2)
SPLITS = (("all", "test"), ("lt1000", "test_lt1000"), ("ge1000", "test_ge1000"))
PAIRS = (("helix3", "none"), ("helix3", "helix3_shuf"))
LAM = 20.0


def t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p of Student's t (scipy if present; else the exact closed forms for df 1 and 2)."""
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except ImportError:
        if df == 2:
            return float(1 - abs(t) / math.sqrt(2 + t * t))
        if df == 1:
            return float(1 - 2 * math.atan(abs(t)) / math.pi)
        raise


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    n = len(d)
    out = {"diffs": d.tolist(), "mean": float(d.mean()), "n": n, "n_pos": int((d > 0).sum())}
    if n >= 2:
        sd = d.std(ddof=1)
        if sd > 0:
            out["t"] = float(d.mean() / (sd / math.sqrt(n)))
            out["p"] = t_two_sided_p(out["t"], n - 1)
        else:
            out["t"], out["p"] = (math.inf if d.mean() != 0 else 0.0), (0.0 if d.mean() != 0 else 1.0)
    return out


def load(root: Path, ckpt: str, arm: str, seed: int) -> dict | None:
    p = root / ckpt / f"{arm}-s{seed}" / f"run_{arm}_seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def run_metrics(r: dict) -> dict:
    curve = [c for c in r["curve"] if c["step"] > 0]
    assert curve, "empty curve"
    m = {}
    for sp, key in SPLITS:
        m[f"area_{sp}"] = float(np.mean([c[key] for c in curve]))
        m[f"final_{sp}"] = float(r["eval"][key])
    m["steps"] = [c["step"] for c in curve]
    m["curve_all"] = [c["test"] for c in curve]
    m["train_final"] = float(r["eval"]["train"])
    return m


def first_reach(steps: list, ys: np.ndarray, thr: float):
    hit = np.nonzero(ys >= thr)[0]
    return int(steps[hit[0]]) if len(hit) else None


def summarize(root: Path, strict_lam: bool = True) -> dict:
    res, shas = {}, set()
    for ckpt, role in CKPTS:
        res[ckpt] = {"role": role, "arms": {}, "pairs": {}, "missing": []}
        for arm in ARMS:
            runs = {}
            for s in SEEDS:
                r = load(root, ckpt, arm, s)
                if r is None:
                    res[ckpt]["missing"].append(f"{arm}-s{s}")
                    continue
                assert r["geom"] == arm and r["seed"] == s and r.get("add3"), (ckpt, arm, s)
                assert not strict_lam or r["lam"] == LAM, ("lam", r["lam"])
                shas.add(r["add3"]["test_sha"])
                runs[s] = run_metrics(r)
            res[ckpt]["arms"][arm] = runs
        for a, b in PAIRS:
            A, B = res[ckpt]["arms"][a], res[ckpt]["arms"][b]
            common = [s for s in SEEDS if s in A and s in B]
            if not common:
                continue
            res[ckpt]["pairs"][f"{a}-{b}"] = {
                f"{kind}_{sp}": paired(np.array([A[s][f"{kind}_{sp}"] for s in common]), np.array([B[s][f"{kind}_{sp}"] for s in common]))
                for kind in ("area", "final") for sp, _ in SPLITS} | {"seeds": common}
    assert len(shas) <= 1, f"runs do not share one held-out test set: {shas}"
    res["test_sha"] = next(iter(shas)) if shas else None
    return res


def fmt(v: float) -> str:
    return f"{100 * v:5.1f}"


def report(res: dict) -> str:
    L = [f"# ft999 read-out (held-out accuracy in %, test set sha {res['test_sha']}); statistic: paired t-test over seeds, "
         "df = n - 1 = 2 for n = 3 -> descriptive only (min two-sided sign-test p = 0.25)\n"]
    for ckpt, role in CKPTS:
        R = res[ckpt]
        L.append(f"\n## {ckpt} ({role}){'  MISSING: ' + ', '.join(R['missing']) if R['missing'] else ''}\n")
        L.append("| arm | n | split | area mean [per seed] | final mean [per seed] | to50 | to90 | train final |")
        L.append("|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            runs = R["arms"][arm]
            if not runs:
                L.append(f"| {arm} | 0 | - | - | - | - | - | - |")
                continue
            ss = sorted(runs)
            steps = runs[ss[0]]["steps"]
            same = all(runs[s]["steps"] == steps for s in ss)
            mc = np.mean([runs[s]["curve_all"] for s in ss], 0) if same else None
            for k, (sp, _) in enumerate(SPLITS):
                ar = [runs[s][f"area_{sp}"] for s in ss]; fi = [runs[s][f"final_{sp}"] for s in ss]
                r50 = r90 = ""
                if k == 0 and mc is not None:
                    r50, r90 = first_reach(steps, mc, 0.5), first_reach(steps, mc, 0.9)
                    r50, r90 = ("not reached" if r50 is None else r50), ("not reached" if r90 is None else r90)
                tr = fmt(np.mean([runs[s]["train_final"] for s in ss])) if k == 0 else ""
                L.append(f"| {arm} | {len(ss)} | {sp} | {fmt(np.mean(ar))} [{', '.join(fmt(v).strip() for v in ar)}] | "
                         f"{fmt(np.mean(fi))} [{', '.join(fmt(v).strip() for v in fi)}] | {r50} | {r90} | {tr} |")
        for name, P in R["pairs"].items():
            L.append(f"\n{ckpt} {name} (seeds {P['seeds']}; difference in percentage points):")
            for kind in ("area", "final"):
                for sp, _ in SPLITS:
                    q = P[f"{kind}_{sp}"]
                    tp = f"t = {q['t']:.2f}, p = {q['p']:.3f} (df {q['n'] - 1})" if "t" in q else "t n/a"
                    L.append(f"  {kind:5s} {sp:6s}: mean {100 * q['mean']:+6.1f} | per seed "
                             f"[{', '.join(f'{100 * v:+.1f}' for v in q['diffs'])}] | {q['n_pos']}/{q['n']} positive | {tp}")
    L.append("\nPrimary claim = 9B only; 42B is secondary and is not substituted for 9B.")
    return "\n".join(L)


def selftest() -> None:
    """Fake runs with known curves: helix3 = none + 0.2 (all steps, all splits) except seed 2 (+0.1); shuf = none - 0.05."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        steps = [0, 50, 100, 150]
        base = {"none": 0.3, "helix3": 0.5, "helix3_shuf": 0.25}
        for ckpt, _ in CKPTS:
            for arm in ARMS:
                for s in SEEDS:
                    off = base[arm] + (-0.1 if (arm == "helix3" and s == 2) else 0.0) + 0.01 * s
                    curve = [{"step": t, "test": 0.0 if t == 0 else off + t / 1000,
                              "test_lt1000": 0.0 if t == 0 else off + 0.1 + t / 1000,
                              "test_ge1000": 0.0 if t == 0 else off - 0.1 + t / 1000} for t in steps]
                    ev = {"test": off + 0.2, "test_lt1000": off + 0.3, "test_ge1000": off + 0.1, "train": 0.9}
                    d = root / ckpt / f"{arm}-s{s}"; d.mkdir(parents=True)
                    (d / f"run_{arm}_seed{s}.json").write_text(json.dumps(
                        {"geom": arm, "seed": s, "lam": LAM, "curve": curve, "eval": ev, "add3": {"test_sha": "abc"}}))
        res = summarize(root)
        R = res["9B"]
        # area over steps 50, 100, 150 = off + 0.1
        assert abs(R["arms"]["none"][0]["area_all"] - 0.4) < 1e-9, R["arms"]["none"][0]["area_all"]
        assert abs(R["arms"]["helix3"][1]["area_lt1000"] - (0.5 + 0.01 + 0.1 + 0.1)) < 1e-9
        assert abs(R["arms"]["helix3_shuf"][2]["final_ge1000"] - (0.25 + 0.02 + 0.1)) < 1e-9
        q = R["pairs"]["helix3-none"]["area_all"]
        assert np.allclose(q["diffs"], [0.2, 0.2, 0.1]) and q["n_pos"] == 3 and abs(q["mean"] - 0.5 / 3) < 1e-9, q
        d = np.array([0.2, 0.2, 0.1]); t = d.mean() / (d.std(ddof=1) / math.sqrt(3))
        assert abs(q["t"] - t) < 1e-9 and abs(q["p"] - (1 - abs(t) / math.sqrt(2 + t * t))) < 1e-6, (q, t)
        q2 = R["pairs"]["helix3-helix3_shuf"]["final_all"]
        assert np.allclose(q2["diffs"], [0.25, 0.25, 0.15]), q2
        # to50 of the helix3 mean curve: mean offset 0.5 + 0.01 - 0.0333 = 0.4767 -> 50 gives 0.5267 >= 0.5
        assert first_reach([50, 100, 150], np.array([0.4, 0.45, 0.5]), 0.5) == 150 and first_reach([50], np.array([0.1]), 0.5) is None
        # a missing run is reported, not silently averaged
        (root / "42B" / "helix3-s1" / "run_helix3_seed1.json").unlink()
        res = summarize(root)
        assert res["42B"]["missing"] == ["helix3-s1"] and res["42B"]["pairs"]["helix3-none"]["seeds"] == [0, 2]
        txt = report(res)
        assert "MISSING: helix3-s1" in txt and "9B (PRIMARY)" in txt
        # a run on a different test set is refused
        p = root / "9B" / "none-s0" / "run_none_seed0.json"
        r = json.loads(p.read_text()); r["add3"]["test_sha"] = "zzz"; p.write_text(json.dumps(r))
        try:
            summarize(root)
            raise AssertionError("different test sets were not refused")
        except AssertionError as e:
            assert "share one held-out test set" in str(e), e
        print(txt)
        print("\nselftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/yh6210/llmadd_torch/runs/ft999/out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    res = summarize(Path(args.root))
    print(report(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
