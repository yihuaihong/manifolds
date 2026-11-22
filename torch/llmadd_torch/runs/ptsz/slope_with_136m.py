"""Descriptive, read-only companion of readout_ptsz.py (written 2026-11-19 by the prep agent; NOT part of the registered
analysis and never a criterion): the registered size-slope numbers of the area gain (win-ctx - none, paired by seed),
including 136M once its cell is final, plus the median / trimmed magnitudes that the FINDINGS F99 addendum requires next to
any slope magnitude. The registered verdict comes ONLY from readout_ptsz.py; this script repeats its statistics
(readout_ptsz.slope_block / verdict / robust, imported, not re-implemented) so FINDINGS / paper text can quote the
magnitude together with median, trimmed and leave-one-seed-out numbers in one place.

It never calls readout_ptsz.readout() (so it never writes FIRST_READOUT.json) and writes nothing unless --json OUT is given.

  python runs/ptsz/slope_with_136m.py [--root RUNS] [--json OUT]

Output is labelled PARTIAL while any of the eight L40S cells is not final; then the 7-point L40S slope (without 136M) and a
PROVISIONAL 136M (stage-A seeds 0-3 at the best complete lr per arm, plain argmax of the mean, not F87 rule 1) are printed
for orientation only."""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_ptsz as RO   # noqa: E402
S, G = RO.S, RO.G

F99_ADD = dict(mean=2.79, median=1.82, trimmed=2.44, loso=2.47, l40s4_mean=2.16, l40s4_median=1.01, l40s4_trimmed=1.17)


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, out=lambda *x, **y: print(*x, **y), **k)
    return r, buf.getvalue()


def trimmed(d):
    return float(np.sort(d)[1:-1].mean())


def block(cells, name, use_c=True, res=None):
    """the registered statistics of one set of cells (8 seeds each): slope, (a)(b)(c), robustness and ratios"""
    main = RO.slope_block(cells)
    h1, h2 = RO.slope_block(cells, seeds=RO.H1), RO.slope_block(cells, seeds=RO.H2)
    v, crit = RO.verdict(main, h1, h2, use_c=use_c)
    rb = RO.robust(cells)
    print(f"\n## {name}: {len(cells)} cells ({', '.join(RO.LABEL[c['cid']] for c in sorted(cells, key=lambda c: c['x']))})")
    print(f"  rule applied as registered: {v}")
    r = rb["ratios"] or {}
    ext, (ce, se_) = rb["loso_ext"][0], rb["loso_ext"][1]
    print(f"  magnitude line (F99 addendum: quote the mean slope with the median and trimmed ones, or the direction only):")
    print(f"    mean-gain slope {rb['mean']:+.2f} / median-gain slope {rb['median']:+.2f} (ratio {r.get('median', float('nan')):.2f}) / "
          f"trimmed slope {rb['trimmed']:+.2f} (ratio {r.get('trimmed', float('nan')):.2f}) / leave-one-seed-out extreme toward 0 "
          f"{ext:+.2f} (-{RO.LABEL[ce]} s{se_}; ratio {r.get('loso', float('nan')):.2f}) points per e-fold of N_nonemb")
    print(f"    'few seeds' label (any ratio < {RO.RATIO}): {'YES: the slope rests on a few seeds' if rb['label'] else 'no'}")
    if res is not None:
        res[name] = dict(slope=main, h1=h1["b"], h2=h2["b"], verdict=v, crit=crit,
                         robust={k: rb[k] for k in ("mean", "median", "trimmed", "loso_ext", "ratios", "label", "late", "late_rel")})
    return main, rb


def provisional_136m(root, own):
    """seeds 0-3 of stage A: per arm the decided C lr if any, else the complete lr with the highest mean area"""
    log, state = G.decisions(root / "ptsz" / "decisions.json")
    pick, area = {}, {}
    for arm in G.ARMS:
        sel, _ = quiet(S.select_arm, root, "136m", arm, state[("136m", arm)]["X"], own, state[("136m", "none")]["X"])
        recs = sel.get("recs", {})
        full = {lr: np.array([recs[(lr, s)]["area"] for s in G.SEL]) for lr in sel.get("grid", ())
                if all((lr, s) in recs and recs[(lr, s)]["status"] == "ok" for s in G.SEL)}
        if not full:
            return None
        lr = state[("136m", arm)]["C"] or (sel.get("best") if sel.get("status") == "ok" else max(full, key=lambda k: full[k].mean()))
        if lr not in full:
            return None
        pick[arm], area[arm] = lr, full[lr]
        pick[arm + "_n_lrs_complete"] = f"{len(full)}/{len(sel.get('grid', ()))}"
    return dict(pick=pick, none=area["none"], prior=area["win-ctx"], d=area["win-ctx"] - area["none"])


def main():
    root = Path(opt("--root", str(G.B / "runs")))
    res = {"root": str(root), "readout_sha": S.sha(HERE / "readout_ptsz.py"), "this_sha": S.sha(Path(__file__))}
    (log, cells), _ = quiet(RO.load_cells, root)
    own = S.own_failures(root, lambda *a: None)
    l40s = [cells[c] for c in RO.L40S8]
    final = all(c["ok"] for c in l40s)
    tag = "FINAL (all eight L40S cells final)" if final else "PARTIAL (not every L40S cell final; registered primary NOT DETERMINABLE)"
    res["status"] = "final" if final else "partial"
    print(f"# slope_with_136m.py — descriptive, read-only; {tag}")
    print(f"# root {root}; readout_ptsz {res['readout_sha']}; decisions {len(log)} entries; the registered verdict is readout_ptsz.py's")
    print("\n## cells (area points; gain = win-ctx - none, paired by seed; 8 seeds)")
    print("  cell        ln N   none  prior   gain mean  median  trimmed   s0-3   s4-7  status")
    rows = {}
    for cid in RO.L40S8 + RO.B200_2:
        c = cells[cid]
        if not c["ok"]:
            why = "; ".join(c["notfinal"] + c["problems"]) or "not available"
            print(f"  {RO.LABEL[cid]:<10} {c['x']:6.3f}  {'not final: ' + why}")
            continue
        d = c["d"]
        rows[cid] = dict(lnN=c["x"], none=float(c["an"].mean()), prior=float(c["ap"].mean()), gain=float(d.mean()),
                         median=float(np.median(d)), trimmed=trimmed(d), h1=float(d[list(RO.H1)].mean()), h2=float(d[list(RO.H2)].mean()))
        r = rows[cid]
        print(f"  {RO.LABEL[cid]:<10} {c['x']:6.3f} {r['none']:6.1f} {r['prior']:6.1f}   {r['gain']:+6.1f} {r['median']:+7.1f} {r['trimmed']:+8.1f} "
              f"{r['h1']:+6.1f} {r['h2']:+6.1f}  {'reused' if cid in RO.REUSED else 'new'}{' (B200, descriptive only)' if cid in RO.B200_2 else ''}")
    res["cells"] = rows

    if final:
        block(l40s, "PRIMARY (registered): eight L40S sizes", res=res)
        block([cells[c] for c in RO.NEW3], "SECONDARY (registered, (a)(b) only): 136M, 391M, 731M", use_c=False, res=res)
    else:
        seven = [c for c in l40s if c["ok"]]
        block(seven, f"PARTIAL L40S ({len(seven)} of 8 sizes; NOT the registered primary, orientation only)", res=res)
        p = provisional_136m(root, own)
        if p is None:
            print("\n## PROVISIONAL 136M: not available (an arm has no complete stage-A lr)")
        else:
            res["provisional_136m"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in p.items()}
            x = [c["x"] for c in seven] + [cells["136m"]["x"]]
            ys = {"mean": [rows[c["cid"]]["gain"] for c in seven] + [float(p["d"].mean())],
                  "median": [rows[c["cid"]]["median"] for c in seven] + [float(np.median(p["d"]))],
                  "trimmed": [rows[c["cid"]]["trimmed"] for c in seven] + [float(np.sort(p["d"])[1:-1].mean())]}
            pt = {k: float(RO.ols(x, v)) for k, v in ys.items()}
            res["provisional_8pt_point_slopes"] = pt
            print(f"\n## PROVISIONAL 136M (stage-A seeds 0-3 only; picks none {p['pick']['none']} ({p['pick']['none_n_lrs_complete']} lrs complete), "
                  f"win-ctx {p['pick']['win-ctx']} ({p['pick']['win-ctx_n_lrs_complete']} lrs complete); NOT the registered cell)")
            print(f"  none {p['none'].mean():.1f}, prior {p['prior'].mean():.1f}, gain {p['d'].mean():+.1f} (seeds {np.round(p['d'], 1).tolist()}); "
                  f"AUDIT expected none 58-68, prior 88-91")
            print(f"  8-point point slopes with this provisional 136M (no intervals; 136M has 4 seeds): mean {pt['mean']:+.2f}, "
                  f"median {pt['median']:+.2f}, trimmed {pt['trimmed']:+.2f}")

    f6 = [cells[c] for c in RO.F99_6]
    if all(c["ok"] for c in f6):
        print("\n## F99 reference (the six F99 sizes, recomputed here; FINDINGS F99 addendum: mean +2.79, median +1.82, trimmed +2.44, "
              "LOSO +2.47; L40S-4 mean +2.16, median +1.01, trimmed +1.17)")
        _, rb6 = block(f6, "F99 six sizes (53M, 93M, 190M, 294M-B200, 505M, 1.07B; mixed hardware, as F99)", res=res)
        _, rb4 = block([cells[c] for c in RO.L40S4], "F99 L40S-4 (53M, 93M, 190M, 505M)", res=res)
        okref = abs(rb6["mean"] - F99_ADD["mean"]) < 0.01 and abs(rb6["median"] - F99_ADD["median"]) < 0.01 \
            and abs(rb6["trimmed"] - F99_ADD["trimmed"]) < 0.01
        print(f"  reference check vs the F99 addendum (tol 0.01): {'matches' if okref else 'DIFFERS: investigate before quoting'}")
        res["f99_reference_matches"] = bool(okref)
    if opt("--json"):
        Path(opt("--json")).write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print(f"\n# json written {opt('--json')}")


if __name__ == "__main__":
    main()
