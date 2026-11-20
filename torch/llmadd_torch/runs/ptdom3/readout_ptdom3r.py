"""ptdom3 toy3 replication read-out (multi-seed; EXPLORATORY tier: no verdict words beyond the pick rule's E1-E5 and the labels
'preliminary (1 seed)' / 'replicated at k seeds' + the project grading 'supports').

Reads seeds 0-2 of the ptdom3 toy3 runs runs/ptdom3/out/ptdom3-T3-<arm>-lr<lr>-s<seed>[-r<k>]: the seed-0 screen (22 runs) and the
replication stage (runs/ptdom3/HANDOFF_rep.md; 11 runs: none@1.6e-2 s1,2; none@3.2e-2 s0,1,2; only-torus@1e-3 s1,2;
only-hcube@2e-3 s1,2; only-box5@1e-3 s1,2). Every per-run number comes from readout_ptdom3.py's own functions (frozen, sha
9abd89ad75a7, imported read-only): run_json, status, dom_numbers (area / final3 / twin-free tf_area / tf_final3, split_masks at
the run's own seed), grad_numbers (LM share), pick constants (PRIMARY_TF, TOY3, GRID).

Per toy3 domain d (detailed for the replicated torus / hcube / box5; one line each for the seed-0-only cyc12 / d6 / caesar):
  metric     d's PRIMARY area: tf_area (twin-free) for hcube / d6, area (full read-out set) otherwise (as the screen read-out);
             the other metric is printed on the same cells as 'secondary'.
  p, N       chosen EXACTLY as runs/pt6v2/pick_pt6v2.py (sha 7a2e73faa8c8; its docstring I1 / I2) chooses p and n: evidence seeds
             0-2, valid runs only (VALID / FLAGGED); an lr is 'present' when its number of evidence seeds with a value is more than
             half of the best-covered lr of the same arm (prior lrs {1e-3, 2e-3} / every none lr); the highest mean over its seeds
             (rounded 1e-9) wins, a tie goes to the higher lr. A run JSON without check.json is UNFINISHED (still finishing).
  g_s        only-d@p - none@N, paired by seed (seeds where both exist); mean, wins, Cohen d = mean / sd(ddof 1) (= the rule's d),
             paired bootstrap 90% CI: resample the k per-seed differences g_s with replacement (B 20000, numpy default_rng(0)),
             percentile 5 / 95 of the resampled means (k = 3 gives only 10 distinct resamples: coarse; k < 2: none).
  final3 g   the same pairs on final3 (tf_final3 for hcube / d6). same-lr gains: only-d@lr - none@lr per seed where both exist.
  tags       FLOOR = both paired means <= chance + 5 (chance = 100 / n_items on the twin-free set, 100 x chance_readout on the full
             set; the chance value taken as pick_pt6v2.t_chance takes it); CEILING = the none mean >= 95. Also per seed.
  E1-E5      the pick rule's (runs/pt6v2/AUDIT.md "Composition pick rule" + "Pick rule readings"; E3 on the ptdom3 none grid =
             the screen grid + 3.2e-2 where it has a valid run for d). V2 flag: N = 3.2e-2 -> Amendment V2 (none@6.4e-2) would
             trigger if d is picked.
  label      k paired seeds = 1 -> 'preliminary (1 seed)'; k >= 2 -> 'replicated at k seeds'; grading (project rule): 'supports'
             iff k >= 2, every g_s has the same sign and |mean g| > the seed range (max g_s - min g_s).
  LM share   readout_ptdom3.grad_numbers of the prior run per seed (median grad_lm_share over the prior-on evals); < 1/2 marked.
  cross-check  runs pick_pt6v2.py --allow_partial --json (or reads --pick_json) and asserts p / n / paired seeds / g / mean g /
             wins / k / d / E1-E5 / eligible equal for each of the 6 toy3 domains; a mismatch is printed in a banner and the
             exit code is 3 (after every output is written).
  seed-0 consistency  the seed-0 cells' numbers vs readout_ptdom3.report()'s screen read-out (same dirs, statuses, areas).

    python runs/ptdom3/readout_ptdom3r.py                    strict: exit 2 unless the 11 replication runs and the needed
                                                             screen runs are VALID / FLAGGED; writes <out>.txt / .json
    python runs/ptdom3/readout_ptdom3r.py --allow_partial    PROVISIONAL read-out of whatever exists ("complete": false)
        [--out PREFIX] [--pick_json PICK.json | --no_crosscheck] [--root runs/ptdom3]
    python runs/ptdom3/readout_ptdom3r.py --selftest         planted worlds (pick_pt6v2's own evaluate on the same worlds)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "ptdom3"
PICK_PY = B / "runs" / "pt6v2" / "pick_pt6v2.py"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
RO3_SHA, PICK_SHA = "9abd89ad75a7", "7a2e73faa8c8"

if str(B) not in sys.path:
    sys.path.insert(0, str(B))
import readout_ptdom3 as RO3  # noqa: E402  (frozen; read only)

assert Path(RO3.__file__).resolve() == (B / "readout_ptdom3.py").resolve()
assert sha(RO3.__file__) == RO3_SHA, f"readout_ptdom3.py sha {sha(RO3.__file__)} != the pinned {RO3_SHA}"

TOY3 = tuple(RO3.TOY3)
REP_P = {"torus": "1e-3", "hcube": "2e-3", "box5": "1e-3"}            # the replicated prior cells (HANDOFF_rep, AUDIT.md 263-266)
REP_DOMS = tuple(REP_P)
REP11 = ([("none", "1.6e-2", s) for s in (1, 2)] + [("none", "3.2e-2", s) for s in (0, 1, 2)]
         + [(f"only-{d}", lr, s) for d, lr in REP_P.items() for s in (1, 2)])
SCREEN_NONE = tuple(RO3.GRID["none"])                                  # 2e-3, 4e-3, 8e-3, 1.6e-2
PRIOR_LRS = ("1e-3", "2e-3")                                           # the pick rule's prior lrs
SCREEN_NEEDED = [("none", lr, 0) for lr in SCREEN_NONE] + [(f"only-{d}", lr, 0) for d in REP_DOMS for lr in PRIOR_LRS]
EVID = (0, 1, 2)
TOP_NONE, MIN_SEEDS, D_MIN = 3.2e-2, 3, 0.5
OKS = ("VALID", "FLAGGED")
E_KEYS = ("E1", "E2", "E3", "E4", "E5")
BOOT_B, BOOT_SEED = 20000, 0
assert len(REP11) == 11 and len(SCREEN_NEEDED) == 10
NAME3 = re.compile(rf"^{RO3.PREFIX}-{RO3.KEY}-(none|only-[a-z0-9]+)-lr([0-9.e-]+)-s(\d+)(?:-r(\d+))?$")
num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
mean = lambda xs: (lambda v: float(np.mean(v)) if v else None)(list(xs))
lrk = lambda lr: float(lr)
cell_txt = lambda c: f"{c[0]}@{c[1]} s{c[2]}"


def pkey(d):
    return ("tf_area", "tf_final3") if d in RO3.PRIMARY_TF else ("area", "final3")


def skey(d):
    return ("area", "final3") if d in RO3.PRIMARY_TF else ("tf_area", "tf_final3")


def mname(key):
    return "twin-free area" if key.startswith("tf_") else "area (full set)"


# ---------------------------------------------------------------- loader (pick_pt6v2.load3's rules, readout_ptdom3's functions)
def load(root=None):
    """{(arm, lr, seed): {dir, rr, status, note, n: {d: dom_numbers}, marks, g: grad_numbers}}: the highest -r<k> per cell; the
    run JSON's identity (seed, lr, prior_domains, tag = the screen tag or with -s<seed>) must match the dir name; status from
    readout_ptdom3.status; 'no check.json' -> UNFINISHED (the job writes the run JSON first)."""
    root = Path(root or (R / "out"))
    keep = {}
    for dd in sorted(root.glob(f"{RO3.PREFIX}-{RO3.KEY}-*")):
        m = NAME3.match(dd.name)
        if not m:
            continue
        k, rr = (m.group(1), m.group(2), int(m.group(3))), int(m.group(4) or 0)
        if k not in keep or rr > keep[k][0]:
            keep[k] = (rr, dd)
    runs = {}
    for (arm, lr, seed), (rr, dd) in sorted(keep.items()):
        f = RO3.run_json(dd)
        r, why = None, ""
        if f is not None:
            try:
                r = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError) as ex:
                r, why = None, f"unreadable run JSON ({ex.__class__.__name__})"
        if r is not None:
            a_ = r.get("args", {})
            base = f"_{RO3.PREFIX}-{RO3.KEY}-{arm}-lr{lr}"
            if (a_.get("seed"), a_.get("lr"), a_.get("prior_domains")) != (seed, float(lr), "" if arm == "none" else arm[5:]) \
                    or a_.get("tag") not in (base, f"{base}-s{seed}"):
                r, why = None, f"identity of the run JSON differs from its directory name {dd.name}"
        act = () if arm == "none" else (arm[5:],)
        if r is not None:
            stt, note = RO3.status(dd, r, act)
            if stt == "INVALID" and "no check.json" in str(note):
                stt, note = "UNFINISHED", f"{note} (run JSON written, check.json not yet: still finishing)"
        elif f is None:
            stt, note = "UNFINISHED", "no run JSON"
        else:
            stt, note = ("UNFINISHED" if why.startswith("unreadable") else "INVALID"), why
        rec = {"dir": str(dd), "rr": rr, "status": stt, "note": note, "n": {}, "marks": [], "g": {}}
        if r is not None and stt != "UNFINISHED":
            for d in TOY3:
                n, mk = RO3.dom_numbers(r, d)
                n.update(tf_window(r, d))
                rec["n"][d] = n
                rec["marks"] += mk
            rec["g"] = RO3.grad_numbers(r)
        runs[(arm, lr, seed)] = rec
    return runs


def tf_window(r, d, win=None):
    """twin-free in-window / out-of-window area: readout_ptdom3.twin_numbers on the run with its curve restricted to the evals
    inside (outside) the prior window, i.e. readout_ptdom3's own in / out definition (steps win[0] <= step < win[1]; dom_numbers'
    'in' / 'out') applied to the twin-free series. The full-set 'in' / 'out' come from dom_numbers itself."""
    win = win or RO3.WIN
    s = RO3.series(r, d)
    out = {"tf_in": None, "tf_out": None}
    if not s or not all(num(x) for _, x in s):
        return out
    for key, inside in (("tf_in", True), ("tf_out", False)):
        keep = [i for i, (st, _) in enumerate(s) if (win[0] <= st < win[1]) == inside]
        if not keep:
            continue
        rr = {**r, "curve": [r["curve"][i] for i in keep]}
        out[key] = RO3.twin_numbers(rr, d, [s[i][1] for i in keep], [])["tf_area"]
    return out


def val(rec, d, key):
    if not rec or rec.get("status") not in OKS:
        return None
    v = (rec.get("n", {}).get(d) or {}).get(key)
    return float(v) if num(v) else None


def chance(runs, d):
    """pick_pt6v2.t_chance: the first VALID / FLAGGED run (sorted cell order) with a value."""
    key = "chance_u" if d in RO3.PRIMARY_TF else "chance"
    for rec in runs.values():
        v = (rec.get("n", {}).get(d) or {}).get(key)
        if rec.get("status") in OKS and num(v):
            return float(v)
    return None


def chance_for(runs, d, key):
    k2 = "chance_u" if key.startswith("tf_") else "chance"
    vals = sorted({round(float((r.get("n", {}).get(d) or {}).get(k2)), 9) for r in runs.values()
                   if r.get("status") in OKS and num((r.get("n", {}).get(d) or {}).get(k2))})
    return vals


# ---------------------------------------------------------------- the rule's p / N (independent re-implementation of I1 / I2)
def present(cells):
    cov = {lr: len(v) for lr, v in cells.items() if v}
    if not cov:
        return []
    mx = max(cov.values())
    return sorted([lr for lr, k in cov.items() if 2 * k > mx], key=lrk)


def best(cells, lrs):
    return max(lrs, key=lambda lr: (round(mean(cells[lr].values()), 9), float(lr))) if lrs else None


def boot90(gv, B_=BOOT_B, seed=BOOT_SEED):
    if len(gv) < 2:
        return None
    rng = np.random.default_rng(seed)
    a = np.asarray(gv, float)
    m = a[rng.integers(0, len(a), (B_, len(a)))].mean(1)
    return [float(np.percentile(m, 5)), float(np.percentile(m, 95))]


def cohen(gv):
    k = len(gv)
    if k < 2:
        return None, None
    a = np.asarray(gv, float)
    mg, sd = float(a.mean()), float(a.std(ddof=1))
    if sd > 0:
        return sd, mg / sd
    return sd, (math.inf if mg > 0 else (-math.inf if mg < 0 else None))


def grade(g):
    v = list(g.values())
    k = len(v)
    if k == 0:
        return {"label": "no paired seed", "supports": False, "why": "no paired seed"}
    if k == 1:
        return {"label": "preliminary (1 seed)", "supports": False, "why": "1 seed"}
    mg, rg = float(np.mean(v)), float(max(v) - min(v))
    same = all(x > 0 for x in v) or all(x < 0 for x in v)
    sup = same and abs(mg) > rg
    why = (f"{'same' if same else 'mixed'} sign, |mean| {abs(mg):.2f} {'>' if abs(mg) > rg else '<='} seed range {rg:.2f}")
    return {"label": f"replicated at {k} seeds", "supports": sup, "direction": ("+" if mg > 0 else "-") if sup else None,
            "range": rg, "why": why}


def cells_of(runs, d, key):
    prior = {lr: v for lr in PRIOR_LRS for v in [{s: x for s in EVID for x in [val(runs.get((f"only-{d}", lr, s)), d, key)] if x is not None}] if v}
    nl = sorted({k[1] for k in runs if k[0] == "none" and k[2] in EVID}, key=lrk)
    none = {lr: v for lr in nl for v in [{s: x for s in EVID for x in [val(runs.get(("none", lr, s)), d, key)] if x is not None}] if v}
    return prior, none


def paired(runs, d, p, n, key, fkey):
    """per-seed numbers of only-d@p vs none@n on (key, fkey)."""
    out = {"seeds": [], "g": {}, "f3": {"prior": {}, "none": {}, "g": {}}, "prior": {}, "none": {}}
    if p is None or n is None:
        return out
    for s in EVID:
        a, b = val(runs.get((f"only-{d}", p, s)), d, key), val(runs.get(("none", n, s)), d, key)
        if a is not None:
            out["prior"][s] = a
        if b is not None:
            out["none"][s] = b
        if a is not None and b is not None:
            out["seeds"].append(s)
            out["g"][s] = a - b
            fa, fb = val(runs.get((f"only-{d}", p, s)), d, fkey), val(runs.get(("none", n, s)), d, fkey)
            if fa is not None and fb is not None:
                out["f3"]["prior"][s], out["f3"]["none"][s], out["f3"]["g"][s] = fa, fb, fa - fb
    return out


def window_split(runs, d, p, n, key, fkey, seeds):
    """per paired seed: g inside / outside the prior window (readout_ptdom3's in / out; tf_in / tf_out on the twin-free key), and
    the levels of both arms (area, in, out, final3) so the reader sees where each arm moves."""
    ink, outk = ("tf_in", "tf_out") if key.startswith("tf_") else ("in", "out")
    out = {"in_key": ink, "out_key": outk, "in": {"g": {}}, "out": {"g": {}}, "levels": {"prior": {}, "none": {}}}
    for s in seeds:
        cp, cn = runs.get((f"only-{d}", p, s)), runs.get(("none", n, s))
        for arm, c in (("prior", cp), ("none", cn)):
            out["levels"][arm][s] = {"area": val(c, d, key), "in": val(c, d, ink), "out": val(c, d, outk), "final3": val(c, d, fkey)}
        for w, k_ in (("in", ink), ("out", outk)):
            a, b = val(cp, d, k_), val(cn, d, k_)
            if a is not None and b is not None:
                out[w]["g"][s] = a - b
    for w in ("in", "out"):
        st = stats(out[w]["g"])
        out[w].update({x: st[x] for x in ("k", "mean_g", "wins")})
    for arm in ("prior", "none"):
        out.setdefault("level_means", {})[arm] = {f: mean(v[f] for v in out["levels"][arm].values() if v[f] is not None)
                                                  for f in ("area", "in", "out", "final3")}
    return out


def stats(g):
    gv = [g[s] for s in sorted(g)]
    k = len(gv)
    if not k:
        return {"k": 0, "mean_g": None, "wins": None, "sd": None, "d": None, "boot90": None}
    sd, d = cohen(gv)
    return {"k": k, "mean_g": float(np.mean(gv)), "wins": int(sum(x > 0 for x in gv)), "sd": sd, "d": d, "boot90": boot90(gv)}


def tags_for(pm, nm, ch):
    t = []
    if num(ch) and num(pm) and num(nm) and pm <= ch + 5 and nm <= ch + 5:
        t.append(f"FLOOR (both <= chance {ch:.1f} + 5)")
    if num(nm) and nm >= 95:
        t.append("CEILING (none >= 95)")
    return t


def evaluate(runs, d):
    key, fkey = pkey(d)
    prior, none = cells_of(runs, d, key)
    pp, npr = present(prior), present(none)
    p, n = best(prior, pp), best(none, npr)
    ch = chance(runs, d)
    pr = paired(runs, d, p, n, key, fkey)
    st = stats(pr["g"])
    k = st["k"]
    row = {"metric": mname(key), "key": key, "fkey": fkey, "chance": ch, "chance_values_seen": chance_for(runs, d, key),
           "p": p, "N": n, "present_prior": pp, "present_none": npr,
           "prior_means": {lr: [mean(v.values()), len(v)] for lr, v in prior.items()},
           "none_means": {lr: [mean(v.values()), len(v)] for lr, v in none.items()},
           "paired_seeds": pr["seeds"], "g": pr["g"], **st, "grade": grade(pr["g"]),
           "final3": {**pr["f3"], **{x: y for x, y in stats(pr["f3"]["g"]).items() if x in ("k", "mean_g", "wins")}},
           "E": {e: False for e in E_KEYS}, "why": {e: "n/a" for e in E_KEYS}, "eligible": False, "tags": [], "tags_per_seed": {},
           "window": window_split(runs, d, p, n, key, fkey, pr["seeds"]) if p is not None and n is not None else None}
    if p is None or n is None:
        miss = f"no only-{d} run at 1e-3 / 2e-3" if p is None else "no none run"
        row["why"] = {e: f"n/a: {miss} at the evidence seeds" for e in E_KEYS}
        return row
    if k:
        pm, nm = mean(pr["prior"][s] for s in pr["seeds"]), mean(pr["none"][s] for s in pr["seeds"])
        row["pn_means_paired"] = [pm, nm]
        row["tags"] = tags_for(pm, nm, ch)
        row["tags_per_seed"] = {s: tags_for(pr["prior"][s], pr["none"][s], ch) for s in pr["seeds"]}
        row["E"]["E1"] = ch is not None and not row["tags"]
        row["why"]["E1"] = ("n/a: no chance" if ch is None else ("; ".join(row["tags"]) or "no tag")) + f" (means p {pm:.1f} / N {nm:.1f})"
        row["E"]["E4"] = st["mean_g"] > 0 and 3 * st["wins"] >= 2 * k
        row["why"]["E4"] = f"mean g {st['mean_g']:+.2f}, wins {st['wins']}/{k}"
        row["E"]["E5"] = st["d"] is not None and st["d"] >= D_MIN
        row["why"]["E5"] = f"d {fmt_d(st['d'])}" if st["d"] is not None else "n/a: d needs >= 2 paired seeds"
    else:
        for e in ("E1", "E4", "E5"):
            row["why"][e] = "n/a: no paired seed"
    row["E"]["E2"] = k >= MIN_SEEDS
    row["why"]["E2"] = f"{k} paired seeds"
    ext = ["3.2e-2"] if any(val(r_, d, key) is not None for k_, r_ in runs.items() if k_[:2] == ("none", "3.2e-2")) else []
    grid = list(SCREEN_NONE) + ext
    top = max(list(npr) + grid, key=lrk)
    at_top = float(n) >= float(top) - 1e-12
    row["E"]["E3"] = (not at_top) or float(top) >= TOP_NONE - 1e-12
    row["why"]["E3"] = (f"N {n} below the grid top {top}" if not at_top else
                        (f"N {n} = the grid top, >= 3.2e-2" if row["E"]["E3"] else f"N {n} = the grid top {top} (< 3.2e-2)"))
    row["grid"], row["N_at_top"] = grid, at_top
    row["v2_flag"] = abs(float(n) - TOP_NONE) < 1e-12
    row["eligible"] = all(row["E"].values())
    # same-lr gains (descriptive): only-d@lr - none@lr where both exist
    row["same_lr"] = {}
    for lr in RO3.GRID["prior"]:
        gg = {s: a - b for s in EVID for a in [val(runs.get((f"only-{d}", lr, s)), d, key)]
              for b in [val(runs.get(("none", lr, s)), d, key)] if a is not None and b is not None}
        if gg:
            row["same_lr"][lr] = {"g": gg, "mean": mean(gg.values()), "k": len(gg)}
    return row


def fmt_d(d):
    return "-" if d is None else ("inf" if d == math.inf else ("-inf" if d == -math.inf else f"{d:.2f}"))


def secondary(runs, d, p, n):
    key, fkey = skey(d)
    pr = paired(runs, d, p, n, key, fkey)
    st = stats(pr["g"])
    return {"metric": mname(key), "key": key, "p": p, "N": n, "g": pr["g"], "prior": pr["prior"], "none": pr["none"],
            **{x: st[x] for x in ("k", "mean_g", "wins", "d", "boot90")},
            "final3_g": pr["f3"]["g"], "final3_mean_g": mean(pr["f3"]["g"].values())}


def seed_info(runs, d):
    """per seed: twin-free n / twins / read-out n from every valid run at that seed (a disagreement between arms is a mark)."""
    out, marks = {}, []
    for s in EVID:
        seen = {}
        for (arm, lr, s_), rec in runs.items():
            if s_ != s or rec.get("status") not in OKS:
                continue
            n = rec.get("n", {}).get(d) or {}
            if n.get("n_tf") is not None:
                seen[f"{arm}@{lr}"] = (n["n_tf"], n["n_twin"])
        if seen:
            vals = set(seen.values())
            if len(vals) > 1:
                marks.append(f"{d} s{s}: twin-free / twin counts differ between arms {seen}")
            ntf, ntw = sorted(vals)[0]
            out[s] = {"n_tf": ntf, "n_twin": ntw, "n_readout": ntf + ntw, "n_runs": len(seen)}
    return out, marks


def lm_rows(runs, d, p):
    out = {}
    for s in EVID:
        rec = runs.get((f"only-{d}", p, s)) if p else None
        if rec and rec.get("status") in OKS and rec.get("g"):
            g = rec["g"]
            out[s] = {k: g.get(k) for k in ("lm_share", "clip", "clip_lt1", "n", "over")}
    return out


def run_marks(runs, d, p, n):
    out = []
    for c in [(f"only-{d}", p, s) for s in EVID] + [("none", n, s) for s in EVID]:
        rec = runs.get(c)
        if rec and c[1]:
            out += [f"{cell_txt(c)}: {m}" for m in rec.get("marks", []) if m.startswith(d + ":")]
    return out


# ---------------------------------------------------------------- completeness (strict)
def excused_cell(c, rec):
    """pick_pt6v2.t_problems (lines 405-407): a replication cell that is INVALID after >= 2 attempts (kept dir -r1 or later) is
    excused; UNFINISHED / MISSING never are. Screen cells are not excused (unchanged)."""
    return c in REP11 and rec is not None and rec["status"] not in OKS + ("UNFINISHED", "MISSING") and int(rec.get("rr") or 0) >= 1


def problems(runs):
    out = []
    for c in REP11 + SCREEN_NEEDED:
        rec = runs.get(c)
        st = rec["status"] if rec else "MISSING"
        if st in OKS or excused_cell(c, rec):
            continue
        txt = f"{'replication' if c in REP11 else 'screen'} {cell_txt(c)}: {st}" + (f" ({rec['note']})" if rec and rec.get("note") else "")
        if c in REP11 and st not in ("UNFINISHED", "MISSING"):
            txt += ": re-run it once (make_jobs_ptdom3r.py, -r1) before reading; a second failure is excused (as pick_pt6v2)"
        out.append(txt)
    return out


def excused(runs):
    return [f"replication {cell_txt(c)}: {runs[c]['status']} at attempt -r{runs[c]['rr']} ({runs[c].get('note')}): excused "
            "(second failed attempt, as pick_pt6v2 lines 405-407); the cell has no value" for c in REP11 if excused_cell(c, runs.get(c))]


def submitted_check(path=R / "submitted.tsv"):
    """the replication rows in submitted.tsv vs REP11 (a mark if they differ; a -r<k> re-run is fine)."""
    if not Path(path).exists():
        return ["submitted.tsv not found"]
    names = [ln.split("\t")[0] for ln in Path(path).read_text().splitlines() if ln.strip()]
    rep = set()
    for nm in names:
        m = NAME3.match(nm)
        if m and not (int(m.group(3)) == 0 and m.group(2) in SCREEN_NONE + RO3.GRID["prior"]):
            rep.add((m.group(1), m.group(2), int(m.group(3))))
    exp = set(REP11)
    out = []
    if rep - exp:
        out.append(f"submitted.tsv has replication rows outside the 11: {sorted(rep - exp)}")
    if exp - rep:
        out.append(f"submitted.tsv lacks replication rows: {sorted(exp - rep)}")
    return out


# ---------------------------------------------------------------- cross-check vs pick_pt6v2.py
def run_pick(keep_json):
    """pick_pt6v2.py --allow_partial --json <fresh temp path> (read-only on every source) -> (its JSON or None, stdout tail, rc).
    The JSON is read only from a temp dir created for this call and only when rc == 0 (pick's run() returns 0 or 2 = refused,
    no JSON); it is then copied to keep_json. A stale keep_json is removed first, so it is never read back."""
    Path(keep_json).unlink(missing_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="pick_", dir=str(R / "tests")))
    out_json = tmp / "pick.json"
    try:
        try:
            p = subprocess.run([PY, str(PICK_PY), "--allow_partial", "--json", str(out_json)], capture_output=True, text=True, timeout=1200,
                               cwd=str(B), env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
        except (OSError, subprocess.TimeoutExpired) as ex:
            return None, repr(ex), -1
        tail = (p.stdout + p.stderr)[-3000:]
        if p.returncode != 0 or not out_json.exists():
            return None, tail, p.returncode if p.returncode != 0 else -2
        try:
            obj = json.loads(out_json.read_text())
        except json.JSONDecodeError as ex:
            return None, tail + f"\n[pick JSON unreadable: {ex}]", -3
        shutil.copyfile(out_json, keep_json)
        return obj, tail, 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pick_inprocess(out_root):
    """pick_pt6v2's own functions (load3 -> gather_T -> evaluate -> strip) on out_root: the cross-check for a non-default --root
    (pick_pt6v2.py itself has no --root; it always reads runs/ptdom3/out)."""
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        if str(PICK_PY.parent) not in sys.path:
            sys.path.insert(0, str(PICK_PY.parent))
        import pick_pt6v2 as PK  # noqa: E402
    finally:
        sys.argv = argv
    assert sha(PK.__file__) == PICK_SHA, f"pick_pt6v2.py sha {sha(PK.__file__)} != the pinned {PICK_SHA}"
    return {"table": [PK.strip(PK.evaluate(c)) for c in PK.gather_T(PK.load3(out_root)).values()], "complete": None,
            "problems": None}


def snapshot(out_root):
    """{path: (mtime_ns, size)} of every check.json + every run dir name under out_root: what can change a status between two loads
    (a run counts only once its check.json exists; a new -r<k> dir replaces the cell). A still-running job's run JSON is rewritten at
    every eval but stays UNFINISHED for both loads, so it is not part of the snapshot."""
    out = {}
    for dd in Path(out_root).glob(f"{RO3.PREFIX}-{RO3.KEY}-*"):
        out[dd.name] = "dir"
        f = dd / "check.json"
        try:
            st = f.stat()
            out[f"{dd.name}/check.json"] = (st.st_mtime_ns, st.st_size)
        except OSError:
            pass
    return out


def cross_check(rows, pick_obj):
    """rows {d: evaluate()} vs pick JSON 'table' rows of src ptdom3 -> (status, diffs, n_fields)."""
    if not pick_obj:
        return "NOT RUN", ["no pick JSON"], 0
    if pick_obj.get("_failed") is not None:
        return "PICK FAILED", [f"pick_pt6v2.py exited rc {pick_obj['_failed']} (or wrote no / an unreadable JSON): nothing to compare; "
                               "see <out>.pick.txt"], 0
    tab = {r["cand"]: r for r in pick_obj.get("table", []) if r.get("src") == "ptdom3"}
    diffs, nf = [], 0
    close = lambda a, b: (a is None and b is None) or (num(a) and num(b) and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)))

    def dval(v):
        return math.inf if v == "inf" else (-math.inf if v == "-inf" else v)
    for d in TOY3:
        a, b = rows.get(d), tab.get(d)
        if a is None or b is None:
            diffs.append(f"{d}: missing on {'this read-out' if a is None else 'the pick'} side")
            continue
        pairs = [("p", a["p"], b.get("p")), ("n", a["N"], b.get("n")), ("seeds", a["paired_seeds"], b.get("seeds")),
                 ("k", a["k"], b.get("k")), ("wins", a["wins"], b.get("wins")), ("eligible", a["eligible"], b.get("eligible"))]
        pairs += [(e, a["E"][e], (b.get("E") or {}).get(e)) for e in E_KEYS]
        for nm, x, y in pairs:
            nf += 1
            if x != y:
                diffs.append(f"{d}: {nm} {x!r} (here) != {y!r} (pick)")
        for nm, x, y in (("mean_g", a["mean_g"], b.get("mean_g")), ("d", a["d"], dval(b.get("d")))):
            nf += 1
            if not (close(x, y) or (x in (math.inf, -math.inf) and x == y)):
                diffs.append(f"{d}: {nm} {x!r} (here) != {y!r} (pick)")
        gb = {int(s): v for s, v in (b.get("g") or {}).items()}
        nf += 1
        if set(gb) != set(a["g"]) or not all(close(a["g"][s], gb[s]) for s in a["g"]):
            diffs.append(f"{d}: per-seed g {a['g']} (here) != {gb} (pick)")
    return ("EQUAL" if not diffs else "MISMATCH"), diffs, nf


# ---------------------------------------------------------------- seed-0 consistency vs the screen read-out
def screen_consistency(runs, screen_res):
    """the seed-0 cells read here vs readout_ptdom3.report()'s rows (status, area, tf_area, final3)."""
    diffs, n = [], 0
    for d, t in screen_res.items():
        for w in t["rows"]:
            rec = runs.get((w["arm"], w["lr"], 0))
            if rec is None:
                diffs.append(f"{d} {w['arm']}@{w['lr']} s0: in the screen, not here")
                continue
            n += 1
            if rec["status"] != w["status"]:
                diffs.append(f"{d} {w['arm']}@{w['lr']} s0: status {rec['status']} here vs {w['status']} in the screen")
            for k in ("area", "tf_area", "final3", "tf_final3"):
                x, y = (rec.get("n", {}).get(d) or {}).get(k), w["n"].get(k)
                if not ((x is None and y is None) or (num(x) and num(y) and abs(x - y) < 1e-9)):
                    diffs.append(f"{d} {w['arm']}@{w['lr']} s0 {k}: {x} here vs {y} in the screen")
    return diffs, n


# ---------------------------------------------------------------- report
f1 = lambda v: "-" if v is None else f"{v:.1f}"
f2 = lambda v: "-" if v is None else f"{v:.2f}"
sg = lambda v: "-" if v is None else f"{v:+.1f}"
sg2 = lambda v: "-" if v is None else f"{v:+.2f}"


def analyze(runs, strict, pick_obj=None, pick_note="", screen_res=None, inputs=None, sub_marks=()):
    probs = problems(runs)
    if probs and strict:
        L = ["ptdom3 replication read-out: REFUSED (strict; inputs incomplete). Re-run when the 11 replication runs have finished, "
             "or --allow_partial for a PROVISIONAL read-out.", f"{len(probs)} problem(s):"] + [f"  - {x}" for x in probs]
        return 2, "\n".join(L), None
    rows = {d: evaluate(runs, d) for d in TOY3}
    dom = {}
    for d in TOY3:
        r = rows[d]
        si, sm = seed_info(runs, d)
        dom[d] = {"primary": r, "secondary": secondary(runs, d, r["p"], r["N"]), "replicated": d in REP_P, "rep_lr": REP_P.get(d),
                  "seed_info": si, "lm": lm_rows(runs, d, r["p"]), "marks": sm + run_marks(runs, d, r["p"], r["N"]),
                  "screen_only": d not in REP_P}
        if d in REP_P and r["p"] != REP_P[d]:
            dom[d]["marks"].insert(0, f"the rule's p {r['p']} != the replicated cell {REP_P[d]} (expected only while the replication "
                                      "seeds are missing; the paired gain uses the rule's p)")
        if len(r["chance_values_seen"]) > 1:
            dom[d]["marks"].append(f"chance differs across runs {r['chance_values_seen']}; the first valid run's value is used (as pick_pt6v2)")
        lmv = [x["lm_share"] for x in dom[d]["lm"].values() if num(x.get("lm_share"))]
        dom[d]["lm_median"] = float(np.median(lmv)) if lmv else None
        dom[d]["lm_flag"] = [s for s, x in dom[d]["lm"].items() if num(x.get("lm_share")) and x["lm_share"] < 0.5]
    cc_status, cc_diffs, cc_n = cross_check(rows, pick_obj) if pick_obj is not None else ("NOT RUN", [pick_note or "skipped"], 0)
    sc_diffs, sc_n = screen_consistency(runs, screen_res) if screen_res is not None else (["not run"], 0)
    obj = {"complete": not probs, "mode": "strict" if strict else "allow_partial", "problems": probs, "excused": excused(runs),
           "inputs": inputs or {},
           "rep11": [list(c) for c in REP11], "screen_needed": [list(c) for c in SCREEN_NEEDED], "submitted_marks": list(sub_marks),
           "runs": {cell_txt(k): {"status": v["status"], "note": v["note"], "rr": v["rr"], "dir": v["dir"]}
                    for k, v in sorted(runs.items(), key=lambda kv: (kv[0][0], lrk(kv[0][1]), kv[0][2]))},
           "domains": dom,
           "cross_check": {"status": cc_status, "diffs": cc_diffs, "n_fields": cc_n, "pick_sha": sha(PICK_PY), "pick_sha_pinned": PICK_SHA,
                           "pick_complete": (pick_obj or {}).get("complete"), "pick_problems": (pick_obj or {}).get("problems"),
                           "note": pick_note},
           "screen_consistency": {"diffs": sc_diffs, "n_cells": sc_n},
           "boot": {"B": BOOT_B, "seed": BOOT_SEED, "method": "paired bootstrap over seeds: resample the k per-seed g with replacement, "
                    "percentile 5 / 95 of the resampled means"},
           "grading": "supports iff k >= 2, every per-seed g has the same sign and |mean g| > max g - min g"}
    txt = report(obj)
    code = 3 if cc_status in ("MISMATCH", "PICK FAILED") else 0
    return code, txt, obj


def report(o):
    D = o["domains"]
    L = []
    if not o["complete"]:
        L += ["*" * 100, "PROVISIONAL (--allow_partial): inputs incomplete; every number below is a partial standing, not the read-out.",
              "*" * 100]
    L += ["ptdom3 toy3 replication read-out (EXPLORATORY; labels: E1-E5 of the pt6v2 pick rule, 'preliminary (1 seed)' / "
          "'replicated at k seeds', grading 'supports')",
          "inputs: " + ", ".join(f"{k} {v}" for k, v in o["inputs"].items())]
    if o["problems"]:
        L.append(f"INCOMPLETE ({len(o['problems'])}):")
        L += [f"  - {x}" for x in o["problems"]]
    L += [f"  MARK {x}" for x in o["submitted_marks"]]
    L += [f"  EXCUSED {x}" for x in o.get("excused", [])]
    st = {}
    for k, v in o["runs"].items():
        st.setdefault(v["status"], []).append(k)
    L.append("runs found: " + "; ".join(f"{s} {len(v)}" for s, v in sorted(st.items())))
    for s in ("INVALID", "UNFINISHED"):
        for k in st.get(s, []):
            L.append(f"  {s} {k}: {o['runs'][k]['note']}")
    for d in REP_DOMS + tuple(x for x in TOY3 if x not in REP_P):
        x = D[d]
        r, sc = x["primary"], x["secondary"]
        L.append("")
        if x["screen_only"]:
            L.append(f"[{d}] seed-0 screen only (not replicated: FLOOR at seed 0 per the replication rule); PRIMARY {r['metric']}; "
                     f"p {r['p']} N {r['N']}; g " + ", ".join(f"s{s} {sg(v)}" for s, v in r["g"].items())
                     + f"; tags {', '.join(r['tags']) or 'none'}; E1-E5 " + " ".join(("ok" if r["E"][e] else "NO") for e in E_KEYS)
                     + f"; {r['grade']['label']}" + ("  [CAESAR: real letter tokens]" if d == "caesar" else ""))
            continue
        L.append(f"[{d}] PRIMARY {r['metric']} on {d}_{RO3.READSET[d]}; chance {f1(r['chance'])}; replicated cell only-{d}@{x['rep_lr']}")
        L.append("  twin-free n per seed (read-out questions with no trained triple / triple twins / read-out n): "
                 + ", ".join(f"s{s} {v['n_tf']} / {v['n_twin']} / {v['n_readout']}" for s, v in x["seed_info"].items()))
        cells = [(f"only-{d}", lr) for lr in RO3.GRID["prior"]] + [("none", lr) for lr in sorted(o_none_lrs(o), key=lrk)]
        L.append(f"  {'cell':<20}" + "".join(f"{'s' + str(s):>8}" for s in EVID) + f"{'mean':>8}{'k':>3}   ({r['metric']})")
        for arm, lr in cells:
            vs = x.get("_cells", {}).get(f"{arm}@{lr}") or {}
            if not vs:
                continue
            m = mean(vs.values())
            tag = " <- p" if arm != "none" and lr == r["p"] else (" <- N" if arm == "none" and lr == r["N"] else "")
            L.append(f"  {arm + '@' + lr:<20}" + "".join(f"{f1(vs.get(s)):>8}" for s in EVID) + f"{f1(m):>8}{len(vs):>3}{tag}")
        L.append(f"  rule's p {r['p']} (present {r['present_prior']}), N {r['N']} (present none {r['present_none']}; ties -> higher lr)")
        L.append(f"  GAIN g = only-{d}@{r['p']} - none@{r['N']} (paired): " + (", ".join(f"s{s} {sg2(v)}" for s, v in r["g"].items()) or "-")
                 + f" | mean {sg2(r['mean_g'])}, wins {r['wins']}/{r['k']}, d {fmt_d(r['d'])}, boot90 "
                 + (f"[{sg2(r['boot90'][0])}, {sg2(r['boot90'][1])}]" if r["boot90"] else "-"))
        f3 = r["final3"]
        L.append(f"  final3 g: " + (", ".join(f"s{s} {sg2(v)}" for s, v in f3["g"].items()) or "-") + f" | mean {sg2(f3.get('mean_g'))}, "
                 f"wins {f3.get('wins')}/{f3.get('k')}")
        w = r.get("window")
        if w:
            for nm, lab in (("in", f"in-window g ({w['in_key']}, steps {RO3.WIN[0]}..{RO3.WIN[1] - 1})"), ("out", f"out-window g ({w['out_key']})")):
                L.append(f"  {lab}: " + (", ".join(f"s{s} {sg2(v)}" for s, v in w[nm]["g"].items()) or "-")
                         + f" | mean {sg2(w[nm]['mean_g'])}, wins {w[nm]['wins']}/{w[nm]['k']}")
            for arm, c in (("prior", f"only-{d}@{r['p']}"), ("none", f"none@{r['N']}")):
                L.append(f"  levels {c:<18} (in-win / out-win / final3): " + ", ".join(
                    f"s{s} {f1(v['in'])} / {f1(v['out'])} / {f1(v['final3'])}" for s, v in w["levels"][arm].items())
                    + f" | means {f1(w['level_means'][arm]['in'])} / {f1(w['level_means'][arm]['out'])} / {f1(w['level_means'][arm]['final3'])}")
        L.append("  same-lr gains (descriptive): " + ("; ".join(f"{lr}: " + ", ".join(f"s{s} {sg(v)}" for s, v in w["g"].items())
                                                          for lr, w in r["same_lr"].items()) or "none"))
        L.append(f"  tags (paired means): {', '.join(r['tags']) or 'none'}; per seed: "
                 + ", ".join(f"s{s} {'/'.join(t) or 'none'}" for s, t in r["tags_per_seed"].items()))
        L.append("  LM share (only-%s@%s, median over prior-on evals): " % (d, r["p"])
                 + ", ".join(f"s{s} {f2(v['lm_share'])} (clip {f2(v['clip'])}, clip<1 {f2(v['clip_lt1'])})" for s, v in x["lm"].items())
                 + (f"  LM<1/2 at seeds {x['lm_flag']}" if x["lm_flag"] else ""))
        L.append("  E1-E5: " + " | ".join(f"{e} {'ok' if r['E'][e] else 'NO'} ({r['why'][e]})" for e in E_KEYS)
                 + f"  -> {'eligible' if r['eligible'] else 'not eligible'} by the pick rule")
        L.append(f"  none grid {r.get('grid')}; N at the grid top: {r.get('N_at_top')}; "
                 + ("V2 FLAG: N = 3.2e-2 -> Amendment V2 (none@6.4e-2) would trigger if picked" if r.get("v2_flag") else "V2 flag: no (N != 3.2e-2)"))
        gr = r["grade"]
        L.append(f"  LABEL: {gr['label']}" + (f"; grading: {'SUPPORTS (' + gr['direction'] + ')' if gr['supports'] else 'not supports'} "
                                              f"({gr['why']})" if r["k"] >= 2 else ""))
        L.append(f"  secondary ({sc['metric']}, same cells): g " + (", ".join(f"s{s} {sg(v)}" for s, v in sc["g"].items()) or "-")
                 + f" | mean {sg2(sc['mean_g'])}, wins {sc['wins']}/{sc['k']}, d {fmt_d(sc['d'])}; final3 mean {sg2(sc['final3_mean_g'])}")
        for m in x["marks"]:
            L.append(f"  MARK {m}")
    cc = o["cross_check"]
    L.append("")
    if cc["status"] in ("MISMATCH", "PICK FAILED"):
        L += ["!" * 100, f"CROSS-CHECK vs pick_pt6v2.py: {cc['status']} ({len(cc['diffs'])} item(s)); a run finishing between the two loads "
              "also causes a MISMATCH: re-run once before reading", *[f"  !! {x}" for x in cc["diffs"]], "!" * 100]
    else:
        L.append(f"CROSS-CHECK vs pick_pt6v2.py (sha {cc['pick_sha']}{'' if cc['pick_sha'] == cc['pick_sha_pinned'] else ' != pinned ' + cc['pick_sha_pinned']}, "
                 f"--allow_partial; pick complete {cc['pick_complete']}): {cc['status']} ({cc['n_fields']} fields over 6 toy3 domains: p, n, "
                 "paired seeds, per-seed g, mean g, wins, k, d, E1-E5, eligible)" + (f" [{'; '.join(cc['diffs'])}]" if cc["status"] != "EQUAL" else ""))
    sc_ = o["screen_consistency"]
    L.append(f"SEED-0 CONSISTENCY vs readout_ptdom3.report() (screen read-out): "
             + ("EQUAL" if not sc_["diffs"] else "DIFFERS: " + " | ".join(sc_["diffs"][:10])) + f" ({sc_['n_cells']} cell-domain rows)")
    L.append(f"bootstrap: {o['boot']['method']} (B {o['boot']['B']}, default_rng({o['boot']['seed']})); grading: {o['grading']}")
    if not o["complete"]:
        L.append("PROVISIONAL: inputs incomplete (see INCOMPLETE above).")
    return "\n".join(L)


def o_none_lrs(o):
    return o.get("_none_lrs", [])


def attach_cells(obj, runs):
    """per-seed primary values of every cell, for the report table and the JSON (cells[<arm>@<lr>][seed])."""
    nl = sorted({k[1] for k in runs if k[0] == "none"}, key=lrk)
    obj["_none_lrs"] = nl
    for d, x in obj["domains"].items():
        key = x["primary"]["key"]
        cells = {}
        for arm, lr in [(f"only-{d}", lr) for lr in RO3.GRID["prior"]] + [("none", lr) for lr in nl]:
            vs = {s: v for s in EVID for v in [val(runs.get((arm, lr, s)), d, key)] if v is not None}
            if vs:
                cells[f"{arm}@{lr}"] = vs
        x["_cells"] = cells


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return "inf" if o > 0 else ("-inf" if o < 0 else "nan")
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def full(runs, strict, pick_obj=None, pick_note="", screen_res=None, inputs=None, sub_marks=()):
    code, _, obj = analyze(runs, strict, pick_obj, pick_note, screen_res, inputs, sub_marks)
    if obj is None:
        return code, _, None
    attach_cells(obj, runs)
    obj["_txt_cells"] = True
    return code, report(obj), obj


# ---------------------------------------------------------------- self-test (planted worlds)
CH = {"cyc12": 100 / 12, "torus": 100 / 36, "hcube": 100 / 32, "d6": 100 / 12, "box5": 4.0, "caesar": 100 / 26}


def _world(spec, status=None):
    """spec {(arm, lr, seed): {d: primary value}} -> runs (every metric = the value; final3 = value - 1)."""
    out = {}
    for k, per in sorted(spec.items()):
        n = {d: {"area": v, "tf_area": v, "final3": v - 1, "tf_final3": v - 1, "chance": CH[d], "chance_u": CH[d], "bpb": 1.0,
                 "n_tf": 66, "n_twin": 302, "in": v + 2, "out": v - 2, "tf_in": v + 2, "tf_out": v - 2} for d, v in per.items()}
        out[k] = {"dir": "", "rr": 0, "status": (status or {}).get(k, "VALID"), "note": "", "n": n, "marks": [],
                  "g": {"lm_share": 0.9, "clip": 1.0, "clip_lt1": 0.0, "n": 27, "over": "prior-on evals"}}
    return out


def _full_world(gain, none16=70.0, none32=65.0, d="torus", p="1e-3", extra=None, drop=()):
    """the 33-run layout: screen s0 (4 none + 3 prior lrs) + the 11 replication cells; only-d@p = none@1.6e-2 + gain[s]."""
    spec = {}
    for dd in TOY3:
        for lr in SCREEN_NONE:
            spec.setdefault(("none", lr, 0), {})[dd] = 20.0
        for s in EVID:
            spec.setdefault(("none", "1.6e-2", s), {})[dd] = none16 + s
            spec.setdefault(("none", "3.2e-2", s), {})[dd] = none32 + s
        for lr in RO3.GRID["prior"]:
            spec.setdefault((f"only-{dd}", lr, 0), {})[dd] = 30.0
    for dd, lr in REP_P.items():
        for s in (1, 2):
            spec[(f"only-{dd}", lr, s)] = {dd: 30.0}
    for s in EVID:
        spec[(f"only-{d}", p, s)] = {d: none16 + s + gain[s]}
    for k, v in (extra or {}).items():
        spec.setdefault(k, {}).update(v)
    for k in drop:
        spec.pop(k, None)
    return _world(spec)


def selftest():
    T_, n_ok = [], 0

    def ok(name, cond, val=""):
        nonlocal n_ok
        T_.append(name)
        n_ok += bool(cond)
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"   got: {str(val)[:400]}"))

    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        sys.path.insert(0, str(PICK_PY.parent))
        import pick_pt6v2 as PK  # noqa: E402  (selftest only: the rule's own evaluate on the same planted worlds)
    finally:
        sys.argv = argv
    ok("S0 pick_pt6v2.py sha = the pinned 7a2e73faa8c8", sha(PK.__file__) == PICK_SHA, sha(PK.__file__))

    def pick_obj(runs):
        return {"table": [PK.strip(PK.evaluate(c)) for c in PK.gather_T(runs).values()]}

    def both(runs, strict=False):
        code, txt, o = full(runs, strict, pick_obj=pick_obj(runs))
        return code, txt, o
    # W1 positive gain, replicated at 3 seeds
    c, t, o = both(_full_world([10.0, 12.0, 11.0]))
    r = o["domains"]["torus"]["primary"]
    ok("W1 positive: p 1e-3, N 1.6e-2 (above 3.2e-2), g [10, 12, 11], mean 11, wins 3/3, d 11, E all ok, eligible",
       (r["p"], r["N"], [round(r["g"][s], 9) for s in EVID], round(r["mean_g"], 9), r["wins"], round(r["d"], 9), r["eligible"])
       == ("1e-3", "1.6e-2", [10.0, 12.0, 11.0], 11.0, 3, 11.0, True), (r["p"], r["N"], r["g"], r["mean_g"], r["d"], r["E"]))
    ok("W1 label 'replicated at 3 seeds', grading SUPPORTS (+) (|11| > range 2)", r["grade"]["label"] == "replicated at 3 seeds"
       and r["grade"]["supports"] and r["grade"]["direction"] == "+", r["grade"])
    lo, hi = r["boot90"]
    ok("W1 boot90 inside [min g, max g] and around the mean", 10 <= lo <= 11 <= hi <= 12, r["boot90"])
    ok("W1 final3 g = g (planted final3 = value - 1 on both arms)", all(abs(r["final3"]["g"][s] - r["g"][s]) < 1e-9 for s in EVID), r["final3"])
    ok("W1 cross-check vs pick_pt6v2's own evaluate: EQUAL", o["cross_check"]["status"] == "EQUAL" and c == 0, o["cross_check"])
    ok("W1 report prints the per-seed cell table and the LABEL line", "only-torus@1e-3" in t and "LABEL: replicated at 3 seeds" in t, t[:400])
    # W2 null: mixed sign, d < 0.5
    c, t, o = both(_full_world([2.0, -1.0, 0.5]))
    r = o["domains"]["torus"]["primary"]
    ok("W2 null: mean +0.5, wins 2/3, d 0.33 -> E5 NO, E4 ok; grading not supports (mixed sign)",
       abs(r["mean_g"] - 0.5) < 1e-9 and r["wins"] == 2 and not r["E"]["E5"] and r["E"]["E4"] and not r["grade"]["supports"]
       and o["cross_check"]["status"] == "EQUAL", (r["mean_g"], r["d"], r["E"], r["grade"]))
    # W3 FLOOR: hcube (twin-free primary) both sides <= chance + 5
    w = _full_world([1.0, 1.0, 1.0], d="hcube", p="2e-3", none16=2.0, none32=1.0)
    c, t, o = both(w)
    r = o["domains"]["hcube"]["primary"]
    ok("W3 FLOOR: hcube both means <= 3.1 + 5 -> FLOOR tag, E1 NO; key tf_area", any(x.startswith("FLOOR") for x in r["tags"])
       and not r["E"]["E1"] and r["key"] == "tf_area" and o["cross_check"]["status"] == "EQUAL", (r["tags"], r["E"], r["key"]))
    # W4 a missing seed (hcube s2 not run): k 2, E2 NO; strict refuses (exit 2), allow_partial writes complete false
    w = _full_world([15.0, 14.0, 16.0], d="hcube", p="2e-3", drop=[("only-hcube", "2e-3", 2)])
    c, t, o = both(w)
    r = o["domains"]["hcube"]["primary"]
    ok("W4 missing seed: hcube k 2 (s0, s1), E2 NO, label 'replicated at 2 seeds', complete false, problem listed",
       r["k"] == 2 and r["paired_seeds"] == [0, 1] and not r["E"]["E2"] and r["grade"]["label"] == "replicated at 2 seeds"
       and not o["complete"] and any("only-hcube@2e-3 s2: MISSING" in x for x in o["problems"]) and o["cross_check"]["status"] == "EQUAL",
       (r["k"], o["problems"]))
    ok("W4 PROVISIONAL banner in --allow_partial", t.startswith("*" * 100) and "PROVISIONAL" in t[:300], t[:300])
    c2, t2, o2 = full(w, True)
    ok("W4 strict mode refuses (exit 2, no JSON)", c2 == 2 and o2 is None and "REFUSED" in t2, (c2, t2[:300]))
    w_inv = dict(_full_world([15.0, 14.0, 16.0]))
    w_inv[("none", "3.2e-2", 1)] = dict(w_inv[("none", "3.2e-2", 1)], status="INVALID", note="failed: x")
    c3, t3, _ = full(w_inv, True)
    ok("W4b strict refuses an INVALID replication run", c3 == 2 and "none@3.2e-2 s1: INVALID" in t3, t3[:300])
    c4, _, o4 = full(_full_world([15.0, 14.0, 16.0]), True)
    ok("W4c strict on the complete 33-run world: exit 0, complete true", c4 == 0 and o4["complete"], (c4, o4 and o4["problems"]))
    # W5 best none at 3.2e-2 -> V2 flag, E3 ok
    w = _full_world([20.0, 22.0, 21.0], d="box5", none16=40.0, none32=50.0)
    c, t, o = both(w)
    r = o["domains"]["box5"]["primary"]
    ok("W5 best none at 3.2e-2: N 3.2e-2, N at grid top, E3 ok (top >= 3.2e-2), V2 FLAG printed",
       r["N"] == "3.2e-2" and r["N_at_top"] and r["E"]["E3"] and r["v2_flag"] and "V2 FLAG" in t and o["cross_check"]["status"] == "EQUAL",
       (r["N"], r["E"], r["v2_flag"]))
    ok("W5 g is against none@3.2e-2: (40 + s + [20, 22, 21]) - (50 + s) = [10, 12, 11]",
       [round(r["g"][s], 9) for s in EVID] == [10.0, 12.0, 11.0], r["g"])
    # W6 I1: a seed-0-only prior lr with a higher value does not compete with the 3-seed cell; a single-seed world competes
    w = _full_world([5.0, 5.0, 5.0], extra={("only-torus", "2e-3", 0): {"torus": 99.0}})
    r = both(w)[2]["domains"]["torus"]["primary"]
    ok("W6 I1: only-torus@2e-3 s0 99 (1 seed) is not present vs 1e-3 (3 seeds): p 1e-3", r["p"] == "1e-3" and r["present_prior"] == ["1e-3"], r)
    w = _full_world([5.0, 5.0, 5.0], extra={("only-torus", "2e-3", 0): {"torus": 99.0}}, drop=[("only-torus", "1e-3", 1), ("only-torus", "1e-3", 2)])
    c, t, o = both(w)
    r = o["domains"]["torus"]["primary"]
    ok("W6 I1 at one seed each: both present, p 2e-3 (higher mean); MARK p != the replicated cell; k 1 'preliminary (1 seed)', boot none",
       r["p"] == "2e-3" and any("!= the replicated cell" in m for m in o["domains"]["torus"]["marks"])
       and r["grade"]["label"] == "preliminary (1 seed)" and r["boot90"] is None and r["d"] is None and o["cross_check"]["status"] == "EQUAL",
       (r["p"], o["domains"]["torus"]["marks"]))
    # W7 tie -> higher lr (none@1.6e-2 = none@3.2e-2 at every seed)
    w = _full_world([5.0, 5.0, 5.0], none16=60.0, none32=60.0)
    r = both(w)[2]["domains"]["torus"]["primary"]
    ok("W7 tie of none means -> higher lr (3.2e-2)", r["N"] == "3.2e-2", r["N"])
    # W8 the cross-check catches a planted disagreement
    w = _full_world([10.0, 12.0, 11.0])
    po = pick_obj(w)
    for x in po["table"]:
        if x["cand"] == "torus":
            x["g"]["1"] += 0.5
            x["E"]["E5"] = False
    c, t, o = full(w, False, pick_obj=po)
    ok("W8 a planted pick disagreement -> MISMATCH banner, exit 3", c == 3 and o["cross_check"]["status"] == "MISMATCH"
       and "!" * 100 in t and "torus: E5" in t and "torus: per-seed g" in t, (c, o["cross_check"]))
    # W9 loader (the real readout_ptdom3 functions on fake run JSONs): seed-free tag at s1 -> VALID; no check.json -> UNFINISHED;
    #    identity mismatch -> INVALID; the highest -r<k> is read; equal to pick_pt6v2.load3 on the same dir
    root = Path(tempfile.mkdtemp(prefix="ro3r_selftest_"))
    try:
        steps = list(range(0, 18310, 250))[:74] + [RO3.LAST]
        PASS = {"pass": True, "checks": []}

        def put(arm, lr, s, rr=0, check=True, tag=None, seed_arg=None):
            r = RO3._fake_run({"torus": [0.5] * 75}, steps, lr=lr, arm=arm)
            r["args"]["seed"] = s if seed_arg is None else seed_arg
            if tag:
                r["args"]["tag"] = tag
            name = f"ptdom3-T3-{arm}-lr{lr}-s{s}" + (f"-r{rr}" if rr else "")
            t_ = r["args"]["tag"].lstrip("_")
            dd = root / "out" / name / t_
            dd.mkdir(parents=True)
            (dd / f"run_whole_none_seed{s}_{t_}.json").write_text(json.dumps(r))
            if check:
                (dd.parent / "check.json").write_text(json.dumps(PASS))
        put("none", "1.6e-2", 1)
        put("only-torus", "1e-3", 2, check=False)
        put("none", "3.2e-2", 1, seed_arg=2)
        put("only-torus", "1e-3", 1)
        put("only-torus", "1e-3", 1, rr=1, tag="_ptdom3-T3-only-torus-lr1e-3-s1")
        L = load(root / "out")
        st = {cell_txt(k): v["status"] for k, v in L.items()}
        ok("W9 loader: s1 seed-free tag VALID; no check.json UNFINISHED; identity mismatch INVALID; -r1 read (tag with -s1 VALID)",
           st == {"none@1.6e-2 s1": "VALID", "none@3.2e-2 s1": "INVALID", "only-torus@1e-3 s1": "VALID", "only-torus@1e-3 s2": "UNFINISHED"}
           and L[("only-torus", "1e-3", 1)]["rr"] == 1, st)
        n1 = L[("none", "1.6e-2", 1)]["n"]["torus"]
        ok("W9b loader window split: constant 0.5 series -> in = out = 50 (full set); tf_in = tf_out = tf_area (readout_ptdom3 functions)",
           abs(n1["in"] - 50) < 1e-9 and abs(n1["out"] - 50) < 1e-9 and "tf_in" in n1
           and (n1["tf_area"] is None or (abs(n1["tf_in"] - n1["tf_area"]) < 1e-9 and abs(n1["tf_out"] - n1["tf_area"]) < 1e-9)), n1)
        rr_ = RO3._fake_run({"torus": [1.0 if RO3.WIN[0] <= st < RO3.WIN[1] else 0.0 for st in steps]}, steps, lr="1.6e-2", arm="none")
        tw = tf_window(rr_, "torus")
        ok("W9c tf_window on a series 1 inside the window, 0 outside: tf_in 100 / tf_out 0 (or None if the fake run has no split)",
           (tw["tf_in"] is None and tw["tf_out"] is None) or (abs(tw["tf_in"] - 100) < 1e-9 and abs(tw["tf_out"]) < 1e-9), tw)
        L2 = PK.load3(root / "out")
        ok("W9 loader = pick_pt6v2.load3 on the same dir (statuses, rr, areas)",
           {k: (v["status"], v["rr"], (v["n"].get("torus") or {}).get("area")) for k, v in L.items()}
           == {k: (v["status"], v["rr"], (v["n"].get("torus") or {}).get("area")) for k, v in L2.items()}, (L, L2))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    # W11 strict mode and an INVALID replication cell: first attempt refuses with the re-run hint; -r1 is excused (pick 405-407)
    w = _full_world([15.0, 14.0, 16.0])
    w[("only-torus", "1e-3", 2)] = dict(w[("only-torus", "1e-3", 2)], status="INVALID", note="failed: planted", rr=0, n={})
    c, t, o = full(w, True)
    ok("W11a INVALID replication cell at -r0: strict refuses (exit 2) with the re-run hint", c == 2 and o is None
       and "only-torus@1e-3 s2: INVALID" in t and "re-run it once" in t and "-r1" in t, t[:400])
    w[("only-torus", "1e-3", 2)]["rr"] = 1
    c, t, o = full(w, True, pick_obj=pick_obj(w))
    r = o["domains"]["torus"]["primary"] if o else {}
    ok("W11b the same cell at -r1: excused (strict exit 0, complete true, EXCUSED printed), torus k 2, cross-check EQUAL",
       c == 0 and o["complete"] and len(o["excused"]) == 1 and "EXCUSED replication only-torus@1e-3 s2" in t and r["k"] == 2
       and o["cross_check"]["status"] == "EQUAL", (c, o and o["excused"], r.get("k"), o and o["cross_check"]))
    w2 = _full_world([15.0, 14.0, 16.0])
    w2[("none", "3.2e-2", 1)] = dict(w2[("none", "3.2e-2", 1)], status="UNFINISHED", note="running", rr=1, n={})
    ok("W11c UNFINISHED at -r1 is not excused: strict refuses", full(w2, True)[0] == 2)
    # W12 pick failed (rc != 0, no JSON) -> PICK FAILED banner, exit 3
    c, t, o = full(_full_world([10.0, 12.0, 11.0]), False, pick_obj={"_failed": 2})
    ok("W12 pick_pt6v2 failed: PICK FAILED banner, exit 3", c == 3 and o["cross_check"]["status"] == "PICK FAILED" and "PICK FAILED" in t,
       (c, o["cross_check"]))
    # W13 window split on planted levels (in = v + 2, out = v - 2 on both arms -> in / out g = g) and levels printed
    o = full(_full_world([10.0, 12.0, 11.0]), False, pick_obj=pick_obj(_full_world([10.0, 12.0, 11.0])))[2]
    wv = o["domains"]["torus"]["primary"]["window"]
    ok("W13 window split: in / out g = g per seed, keys in / out (full set), levels per seed for both arms",
       wv["in_key"] == "in" and [round(wv["in"]["g"][s], 9) for s in EVID] == [10.0, 12.0, 11.0]
       and [round(wv["out"]["g"][s], 9) for s in EVID] == [10.0, 12.0, 11.0] and wv["in"]["wins"] == 3
       and abs(wv["levels"]["prior"][0]["in"] - 82.0) < 1e-9 and abs(wv["levels"]["none"][0]["final3"] - 69.0) < 1e-9, wv)
    wh = full(_full_world([1.0, 2.0, 3.0], d="hcube", p="2e-3"), False)[2]["domains"]["hcube"]["primary"]["window"]
    ok("W13b twin-free primary (hcube) uses tf_in / tf_out", wh["in_key"] == "tf_in" and wh["out_key"] == "tf_out", wh)
    # W10 JSON round trip (inf d) and the renderer's input keys exist
    w = _full_world([5.0, 5.0, 5.0])
    o = full(w, False, pick_obj=pick_obj(w))[2]
    js = json.loads(json.dumps(jsonable(o)))
    ok("W10 JSON: d inf (sd 0, mean > 0) serialised as 'inf'; cross-check EQUAL with the pick's 'inf'",
       js["domains"]["torus"]["primary"]["d"] == "inf" and o["cross_check"]["status"] == "EQUAL", (js["domains"]["torus"]["primary"]["d"], o["cross_check"]))
    print(f"SELFTEST {'PASS' if n_ok == len(T_) else 'FAIL'} {n_ok}/{len(T_)}")
    return 0 if n_ok == len(T_) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--out", help="output prefix (writes <prefix>.txt and <prefix>.json); default runs/ptdom3/tests/readout_rep_<final|partial>_<ts>")
    ap.add_argument("--pick_json", help="read this pick_pt6v2.py --json output instead of running pick_pt6v2.py")
    ap.add_argument("--no_crosscheck", action="store_true")
    ap.add_argument("--root", default=str(R))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    strict = not a.allow_partial
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M")
    prefix = Path(a.out) if a.out else R / "tests" / f"readout_rep_{'final' if strict else 'partial'}_{ts}"
    out_root = Path(a.root) / "out"
    runs = load(out_root)
    if strict and problems(runs):
        code, txt, _ = analyze(runs, True)
        print(txt)
        sys.exit(code)
    pick_obj, note = None, ""
    if a.pick_json:
        pick_obj, note = json.loads(Path(a.pick_json).read_text()), f"read {a.pick_json} (sha {sha(a.pick_json)})"
    elif not a.no_crosscheck and Path(a.root).resolve() != R.resolve():
        # pick_pt6v2.py has no --root (it always reads runs/ptdom3/out): run its own functions in-process on this --root
        pick_obj, note = pick_inprocess(out_root), f"pick_pt6v2 functions in-process (load3 / gather_T / evaluate) on --root {a.root}"
    elif not a.no_crosscheck:
        # order: snapshot -> pick -> this read-out's load -> snapshot. A run that finishes in between (jobs still running) makes the
        # two loads see different files: then retry once; the changed files are reported either way.
        pj = Path(str(prefix) + ".pick.json")
        for attempt in (1, 2):
            s1 = snapshot(out_root)
            pick_obj, tail, rc = run_pick(pj)
            runs = load(out_root)
            s2 = snapshot(out_root)
            changed = sorted(k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k))
            note = (f"ran pick_pt6v2.py --allow_partial --json <fresh temp> (rc {rc}; kept as {pj if pick_obj is not None else 'nothing'}); "
                    f"attempt {attempt}; "
                    + (f"inputs changed during the pick: {changed[:6]}{' ...' if len(changed) > 6 else ''}" if changed
                       else "inputs unchanged during the pick (both loads saw the same files)"))
            Path(str(prefix) + ".pick.txt").write_text(tail)
            if pick_obj is None:
                pick_obj = {"_failed": rc}
            if not changed:
                break
        if strict and problems(runs):
            code, txt, _ = analyze(runs, True)
            print(txt)
            sys.exit(code)
    else:
        note = "--no_crosscheck"
    _, screen_res = RO3.report(Path(a.root))
    inputs = {"readout_ptdom3r.py": sha(__file__), "readout_ptdom3.py": sha(RO3.__file__), "pick_pt6v2.py": sha(PICK_PY),
              "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "n_runs": len(runs), "root": a.root}
    code, txt, obj = full(runs, strict, pick_obj, note, screen_res, inputs, submitted_check(Path(a.root) / "submitted.tsv"))
    if obj is None:
        print(txt)
        sys.exit(code)
    obj["outputs"] = {"txt": str(prefix) + ".txt", "json": str(prefix) + ".json"}
    Path(str(prefix) + ".txt").write_text(txt + "\n")
    Path(str(prefix) + ".json").write_text(json.dumps(jsonable(obj), indent=1))
    print(txt)
    print(f"wrote {prefix}.txt / .json")
    sys.exit(code)


if __name__ == "__main__":
    main()
