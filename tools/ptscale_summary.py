"""Pre-registered read-out of the Torch lr brackets runs/ptlr (53M and 190M at 7%) with their reference cells (pt999
sens-lr* / 7pct-*, ptgw main-W / lr5e-4-W, ptrangeg whole-helix3-w0.05-0.4, ptsize999 190m-* / 500m-*, pt999b
7pct-win-ctx). Intent spec and rules: runs/ptlr/AUDIT.md (incl. the 2026-10-16 amendments); every definition (area, t90 /
t50, diverged, unstable, best lr, open bracket, survives tuning, "N x", bootstrap) is tools/readout_rules.py, the one set
shared with tools/pt2main_summary.py. Written and tested on fake results before any ptlr result existed.

Cells are keyed on r["args"], never on tags: size (d_model, n_layers, n_heads, d_ff) x arm x lr, arm = none (target
none) / gate-ctx (helix3, schedule gate, thr 0.5, no start / hysteresis, site ctx) / win-ctx (helix3, window 0.05,0.4,
site ctx). Every other argument must equal the ptlr base (arith_lines 4, lam 5, bs 32, seq 512, 3e8 tokens, default
prior layers, eval every 250 steps x 150 items, whole-number tokenization, tied embeddings; absent keys of older
snapshots take the trainer's defaults); runs that differ (sens-lam / -L / -bs64, 2%, text site, shuffled targets, ...)
are counted and left out.
Sources: the run dirs in the order given (default runs/{ptlr,ptgw,pt999,ptsize999,ptrangeg,pt999b}/out); a cell's
seeds 0-3 come from the FIRST dir that has all four finished, its confirm seeds 4-7 from the first dir that has all four
(e.g. pt999 7pct-none's own seeds 4-7, or batch 2's T1b dir passed as an extra dir); the same cell in other dirs is
printed as a replicate with the paired difference. So 53M W@1e-3 = ptgw main-W (same code as ptlr) once its 4 seeds
are in, until then ptrangeg (same flags, older snapshot; marked).
Rules (AUDIT numbering):
  1/6 best lr per (size, arm) = readout_rules.best_lr over seeds 0-3 (8-seed cells truncated to 0-3), tie-break as
      pre-registered: the top two lrs if within 2 points (near_rule 'top2'); the shared reading (every lr within 2) is
      printed as PROPOSED when it picks another lr, pending the user's sign-off. bpb tie-break offline (full val, double
      path < 0.002; --offline files, matched by the resolved run path, else by its last 4 parts <source>/out/<cell>/<file>
      so a moved runs root still matches, and checked against the run's tag / seed / logged final bpb) if every run of
      every tied lr has it, else in-training for all ('*'); readout_rules.bracket: open / edge note; a cell with an
      unfinished run is not selectable ('incomplete'; a lower-priority source with all four seeds is used meanwhile)
      unless the run is listed in OWN_FAILURES (an error of its own: its area over all 75 eval slots with the missing
      evals scored 0, bpb worst; a listed run without any run JSON counts as area 0 when its tag has one cell; an entry
      that matches nothing is printed as a WARNING); infrastructure losses are resubmitted
  2   diverged (first non-finite lm / val_loss / geo / grad_prior) and unstable (final val_loss > the same seed's
      none@base lr + 0.5; base lr 1e-3 at 53M, 6e-4 at 190M) are listed and kept in the mean
  3   53M: readout_rules.survives(W@best, none@best) (primary) and gate-ctx@best (secondary), confirm seeds 4-7 of both
      best cells when they exist; best-vs-best numbers without them are PROVISIONAL (winner's curse); an open bracket
      of either arm adds 'NOT FINAL' (the AUDIT's conditional cells / 'bracket not closed' rule)
  4   R_0.9 = readout_rules.speedup of t90 none@best over t90 arm@best (censored at the 300M budget; '>=' for a
      censored none median; '<=' for a censored prior median is an interpretation, rule 4 is silent there) with
      readout_rules.boot_ci; R_0.5 alongside; the lrs picked by median t90 (readout_rules.best_lr_t90) also printed,
      descriptive (the pre-registered ratio is at the area-best lrs)
  5   lr-flat (descriptive): the arm's mean-area spread over its lrs <= 3 points
  7   190M: survives(prior@best, none@best) AND the R_0.5 bootstrap 90% CI above 1; W@best primary, gate-ctx at its
      single lr 6e-4 secondary (untuned: conservative for the prior); a gap < 10 needs seeds 4-7 before any claim
  8   size trend: OLS slope of log R vs log N_nonemb over the sizes where none has >= 2 lrs (tuned) and the arm a
      cell, R = R_0.5 (rule 7's ratio; 190M none rarely reaches 0.9), R_0.9 printed when no size's ratio is a bound;
      90% CI from 10000 seed bootstraps (numpy default_rng(0); seeds resampled within each size, arms paired);
      'grows' / 'shrinks' (a trend) only if the CI excludes 0; the gate-ctx slope is secondary and, while its 190M cell has
      a single lr, labelled untuned on its own line
  9-11 T2, 53M 2% at lr 1e-3 (pt999 2pct-none / 2pct-gate-ctx / 2pct-win-ctx, seeds 0-7, the first source with all
      eight): rule 9 = Delta area (win-ctx - none) >= +30, >= 7/8 wins, exact two-sided sign-flip p < .05
      (readout_rules.signflip_p) and offline (full val) Delta bpb <= +0.015 (in-training printed, descriptive; rule 9
      'incomplete' without offline for all 16 runs); rule 10 = paired mean win-ctx - gate-ctx < -10 -> trade-off, no
      winner; rule 11 = no ratio at 2%, the Delta is 'at lr 1e-3 for both arms'. Every run must pass
      readout_rules.check_2pct (else WITHHELD)
Validity (AUDIT 'Controls', before any number is used): step-0 lm vs the same-seed reference (and the size-seed median)
within 1e-3, code_sha of the source's snapshot, steps 18310 / 75 evals, grad_ckpt_on iff > 150M parameters, W:
prior_on_frac 0.349973 and prior_events [[916, 1], [7324, 0]] where logged, none: prior_used_frac 0. A failure withholds
every verdict that uses the cell ('WITHHELD: <reason>'), as pt2main_summary's rule 6. Ceiling (seen_bXY near 1 at the
end): printed for every counted non-diverged run below 0.9, descriptive only (the AUDIT names no threshold, and
reference cells sit below any 'near 1': pt999 sens-lr5e-4-none s1 0.75, ptsize999 190m-7pct-none s0 0.88).
ladder-1b (2026-10-22, runs/ptlr_1b/AUDIT.md): a 1B size (d1280 x 36, 20 heads, d_ff 5120; N_nonemb 943.9M) read from
<root>/ptlr_1b/out when that dir exists (its OWN_FAILURES.txt too). 1B's lrs are selected on seeds 0-1 (SEL); seeds 2-3
run only at the picks, so rule 7 / 4 / 8 use seeds 0-3 at the picks and 'incomplete' until seeds 2-3 exist; other 1B lrs
are listed with their seeds 0-1. Seed 0's step-0 lm is also checked against the B200 smoke value (LM0_1B). --json writes
each (size, arm)'s best-lr status for the ptlr_1b selection job. Without ptlr_1b/out the output is byte-identical.
    python tools/ptscale_summary.py [--root /scratch/yh6210/llmadd_torch/runs] [--dirs d1 d2 ...] [--extra_dirs ...]
                                    [--offline f.json ...] [--own_failures runs/ptlr/OWN_FAILURES.txt] [--json sel.json]
"""
import argparse
import collections
import glob
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_rules as R   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/scratch/yh6210/llmadd_torch/runs")
ap.add_argument("--dirs", nargs="*", help="run dirs in priority order (default: <root>/{ptlr,ptgw,pt999,ptsize999,ptrangeg,pt999b}/out)")
ap.add_argument("--extra_dirs", nargs="*", default=[], help="inserted right after ptlr (e.g. batch 2's confirm-seed dir)")
ap.add_argument("--offline", nargs="*", help="offline_lm.py output files (default: <root>/*/offline_lm*.json, <root>/*/out/offline_lm*.json)")
ap.add_argument("--own_failures", help="'<tag> <seed>' lines: unfinished runs that stopped on an error of their own "
                                       "(default <root>/ptlr/OWN_FAILURES.txt)")
ap.add_argument("--json", help="ladder-1b: also write each (size, arm)'s best-lr status / pick / open edge here (for the "
                               "runs/ptlr_1b selection job); the printed read-out is unchanged")
A = ap.parse_args()
root = Path(A.root)
DIRS = [Path(d) for d in (A.dirs if A.dirs is not None else
                          [root / f"{x}/out" for x in ("ptlr", "ptgw", "pt999", "ptsize999", "ptrangeg", "pt999b")]
                          # 2026-10-22 ladder-1b: the 1B cells (runs/ptlr_1b, B200), last in priority (no other source
                          # has 1B runs); only when present, so the read-out of the older results is unchanged
                          + [root / "ptlr_1b/out"] * (root / "ptlr_1b/out").exists())]
if A.extra_dirs:
    DIRS = DIRS[:1] + [Path(d) for d in A.extra_dirs] + DIRS[1:]
OFFLINE = A.offline if A.offline is not None else sorted(glob.glob(f"{root}/*/offline_lm*.json") + glob.glob(f"{root}/*/out/offline_lm*.json"))
OWNF = Path(A.own_failures) if A.own_failures else root / "ptlr/OWN_FAILURES.txt"

V = 100278                                    # vocabulary rows (tied embeddings): N_nonemb = n_params - V * d_model
SIZES = {(384, 6, 6, 1536): ("53M", 14165376, "1e-3"), (768, 12, 12, 3072): ("190M", 113283840, "6e-4"),
         (1024, 24, 16, 4096): ("500M", 402752512, "3e-4")}   # name, N_nonemb, base lr (divergence reference)
# 2026-10-22 ladder-1b (runs/ptlr_1b/AUDIT.md): the next GPT-2-shaped rung (d 1280, 36 layers, 20 heads of 64, d_ff 4d),
# 1,072,259,840 parameters (tied embedding counted once); base lr = the lowest none lr of its grid. Its lrs are selected
# on seeds 0-1 (SEL); seeds 2-3 run at the picks only, so a 1B pick has seeds 0-3, the other lrs 0-1; confirm seeds 4-7.
SIZES[(1280, 36, 20, 5120)] = ("1B", 943904000, "2.4e-3")
SEL = {"1B": (0, 1)}
# step-0 lm of seeds 0-3 (runs/ptlr/AUDIT.md 'Controls'; lr- and arm-independent: measured before the first update)
LM0 = {"53M": (11.60809, 11.59907, 11.57956, 11.59710), "190M": (11.65420, 11.67774, 11.65268, 11.64366)}
LM0_1B = {0: 11.73940}   # ladder-1b: seed 0 from the B200 smoke (scale_work/b200/smoke, 4 runs agree to all digits)
SNAP = {"ptlr": {"a3bce3ab63c1"}, "ptgw": {"a3bce3ab63c1"}, "pt999": {"79e9b410dcc3"}, "ptsize999": {"79e9b410dcc3"},
        "ptrangeg": {None}, "pt999b": {"44ce3fef0c29"}, "ptlr_1b": {"a3bce3ab63c1"}}
SNAP_OTHER = {"a3bce3ab63c1", "79e9b410dcc3"}  # a dir not named above (e.g. batch 2's T1b)
DEFAULTS = {"site": "ctx", "untie": False, "text_min": 16, "n_prefix": 256, "gate_start": 0.0, "gate_hyst": 0.0,
            "geo_layers": None, "tokenization": "whole"}
BASE = {"tokenization": "whole", "arith_lines": 4, "lam": 5.0, "bs": 32, "seq": 512, "tokens": 3e8, "geo_layers": None,
        "eval_every": 250, "n_eval": 150, "untie": False, "text_min": 16, "n_prefix": 256}
W_EVENTS, W_ON = R.W_EVENTS, R.W_ON
CEIL = 0.9   # ceiling control: descriptive threshold of this read-out (the AUDIT gives none)
STEPS, EVALS = 18310, 75
ARMS = ("none", "gate-ctx", "win-ctx")
PRIORS = ("win-ctx", "gate-ctx")   # primary, secondary


def fmt_lr(lr):
    m, e = f"{lr:.1e}".split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def arm_of(a):
    g = lambda k: a.get(k, DEFAULTS.get(k))
    if a["target"] == "none":
        return "none" if a.get("schedule", "whole") == "whole" else None
    if a["target"] != "helix3" or g("site") != "ctx":
        return None
    if a.get("schedule") == "window" and a.get("window") == "0.05,0.4":
        return "win-ctx"
    if a.get("schedule") == "gate" and a.get("gate_thr") == 0.5 and g("gate_start") == 0 and g("gate_hyst") == 0:
        return "gate-ctx"
    return None


def why_out(a):
    """why a run is not in the ptlr grid (None if it is)"""
    if arm_of(a) is None:
        return "arm (target / schedule / site)"
    for k, v in BASE.items():
        if a.get(k, DEFAULTS.get(k)) != v:
            return f"{k}={a.get(k, DEFAULTS.get(k))}"
    if (a["d_model"], a["n_layers"], a["n_heads"], a["d_ff"]) not in SIZES:
        return "size"
    return None


# ---------------------------------------------------------------- collect
own = set()
for of_ in [OWNF] + ([] if A.own_failures else [root / "ptlr_1b/OWN_FAILURES.txt"]):   # ladder-1b: its own list
    if of_.exists():
        for line in of_.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                t, s = line.split()
                own.add((t, int(s)))
off_entries, off_tail = {}, collections.defaultdict(list)   # resolved run path -> entry; its last 4 parts -> [entries]
for f in OFFLINE:
    for k, v in json.loads(Path(f).read_text()).items():
        off_entries[k] = v
        off_tail[Path(k).parts[-4:]].append(v)


def off_entry(f, r):
    """the offline_lm entry of a run file: by its resolved path, else by <source>/out/<cell>/<file> (a moved runs root);
    None unless it is the same run (tag, seed and, when logged, the final in-training bpb)"""
    pth = Path(f).resolve()
    e = off_entries.get(str(pth))
    if e is None:
        c = off_tail.get(pth.parts[-4:], [])
        e = c[0] if len(c) == 1 or (c and all(x == c[0] for x in c)) else None
    if e is None or e.get("tag") != r["args"]["tag"] or e.get("seed") != r["args"]["seed"]:
        return None
    lb, rb = e.get("bpb_logged_final"), r["curve"][-1].get("val_bits_per_byte") if r["curve"] else None
    if lb is not None and rb is not None and not (lb == rb or (math.isnan(lb) and math.isnan(rb))):
        return None
    return e


cells = collections.defaultdict(dict)   # (size, arm, lr) -> {source: {seed: (file, run, own_failure)}}
t2_runs = collections.defaultdict(dict)  # 2% (--arith_lines 1) runs of the grid's arms: (size, arm, lr) -> {source: {seed: (file, run)}}
left_out, notes, src_tag = collections.Counter(), [], collections.defaultdict(set)
unfin, run_seen, own_used = [], {}, set()   # unfinished runs not in OWN_FAILURES; (tag, seed) -> finished; used entries
for d in DIRS:
    src = d.parent.name if d.name == "out" else d.name
    for f in sorted(glob.glob(f"{d}/*/run_*.json")):
        r = json.loads(Path(f).read_text())
        a = r["args"]
        run_seen[(a["tag"].lstrip("_"), a["seed"])] = run_seen.get((a["tag"].lstrip("_"), a["seed"]), False) or R.finished(r)
        w = why_out(a)
        if w:
            left_out[(src, w)] += 1
            if w == "arith_lines=1" and why_out(dict(a, arith_lines=4)) is None:
                k2 = (SIZES[(a["d_model"], a["n_layers"], a["n_heads"], a["d_ff"])][0], arm_of(a), fmt_lr(a["lr"]))
                if R.finished(r):
                    assert a["seed"] not in t2_runs[k2].get(src, {}), f"two 2% runs of {k2} seed {a['seed']} in {d}: {f}"
                    t2_runs[k2].setdefault(src, {})[a["seed"]] = (f, r)
                elif k2 == ("53M", k2[1], "1e-3"):
                    notes.append(f"unfinished 2% run (T2 cells need all 8 seeds): {f} (last eval {r['curve'][-1]['step']})")
            continue
        size = SIZES[(a["d_model"], a["n_layers"], a["n_heads"], a["d_ff"])][0]
        key, tag, seed = (size, arm_of(a), fmt_lr(a["lr"])), a["tag"].lstrip("_"), a["seed"]
        of = False
        if not R.finished(r):
            if (tag, seed) in own:
                of = True
                own_used.add((tag, seed))
                notes.append(f"own failure (counts, missing evals scored 0): {f} (last eval {r['curve'][-1]['step']})")
            else:
                unfin.append((key, src, seed, f, r))
                continue
        elif (tag, seed) in own:
            own_used.add((tag, seed))
            notes.append(f"WARNING: {OWNF} lists {tag} {seed}, but {f} is finished: the entry is ignored")
        assert seed not in cells[key].get(src, {}), f"two runs of {key} seed {seed} in {d}: {f}"
        cells[key].setdefault(src, {})[seed] = (f, r, of)
        src_tag[(key, src)].add(tag)
for (key, src), tags in src_tag.items():
    assert len(tags) == 1, f"{key} in {src} under several tags {tags}: keyed on args, one tag per cell and source"
# OWN_FAILURES entries without an unfinished run JSON: a run that crashed before writing one counts as area 0 (all 75
# eval slots missing) when its tag names exactly one cell; anything else is printed, never silently unused
for tag, seed in sorted(own - own_used):
    hit = [(key, src) for (key, src), tags in src_tag.items() if tag in tags]
    if seed not in R.MAIN + R.CONFIRM:
        notes.append(f"WARNING: {OWNF} lists {tag} {seed}: seed outside 0-7, ignored")
    elif run_seen.get((tag, seed)) is not None:
        notes.append(f"WARNING: {OWNF} lists {tag} {seed}, which is not in the ptlr grid: ignored")
    elif len(hit) != 1:
        notes.append(f"WARNING: {OWNF} lists {tag} {seed}: no run JSON and {len(hit)} cells with that tag: ignored")
    else:
        key, src = hit[0]
        sib = next(iter(cells[key][src].values()))[1]
        r = {"args": dict(sib["args"], seed=seed), "steps": sib["steps"], "curve": [], "code_sha": sib.get("code_sha")}
        cells[key][src][seed] = (f"<no run JSON: {tag} seed {seed}>", r, True)
        notes.append(f"own failure without a run JSON (counts as area 0, bpb worst): {tag} seed {seed} ({src})")


def pick(key, seeds):
    """(source, {seed: (file, run, own)}) of the first source with every seed, else None"""
    for src, runs in cells.get(key, {}).items():
        if all(s in runs for s in seeds):
            return src, {s: runs[s] for s in seeds}
    return None


main = {k: pick(k, R.MAIN) for k in cells}
conf = {k: pick(k, R.CONFIRM) for k in cells}
selp = {k: pick(k, SEL[k[0]]) if k[0] in SEL else main[k] for k in cells}   # ladder-1b: the lr-selection seeds
for key, src, seed, f, r in unfin:   # after the picks, so the note says what the cell uses meanwhile
    pk = main.get(key) if seed in R.MAIN else conf.get(key) if seed in R.CONFIRM else None
    which = "seeds 0-3" if seed in R.MAIN else "seeds 4-7" if seed in R.CONFIRM else None
    if key[0] in SEL and seed in SEL[key[0]]:
        pk, which = selp.get(key), f"selection seeds {SEL[key[0]][0]}-{SEL[key[0]][-1]}"
    use = (f"seeds outside 0-7 are not used" if which is None else
           f"its cell's {which} are not selectable (incomplete)" if pk is None else
           f"its cell's {which} come from {pk[0]} meanwhile (the first source with all four)")
    notes.append(f"unfinished: {f} (last eval {r['curve'][-1]['step']} of {r['steps']}): {use}; it counts once it "
                 f"finishes, or as an own failure if listed in {OWNF if key[0] not in SEL or A.own_failures else root / 'ptlr_1b/OWN_FAILURES.txt'}")


def area_of(fr):
    """readout_rules.area; an own failure: over all 75 eval slots, the missing evals scored 0 (AUDIT rule 2)"""
    f, r, of = fr
    return 100 * sum(R.held(e) for e in r["curve"]) / EVALS if of else R.area(r)


def bpb_of(fr):
    f, r, of = fr
    if of:
        return math.inf, math.inf
    ob = R.offline_bpb(off_entry(f, r), R.diverged_at(r) is not None)
    return ob, r["curve"][-1]["val_bits_per_byte"]


# ---------------------------------------------------------------- validity (AUDIT 'Controls')
bad = collections.defaultdict(list)   # (key, seed) -> reasons, for the counted runs (seeds 0-3 and 4-7)
lm0 = collections.defaultdict(dict)
ceil_low = []
for k in cells:
    for pk in (main[k], conf[k]) + ((selp[k],) if k[0] in SEL and main[k] is None else ()):   # ladder-1b: seeds 0-1
        if not pk:
            continue
        src, rs = pk
        for s, (f, r, of) in rs.items():
            c = r["curve"]
            if not c:   # an own failure without a run JSON
                continue
            lm0[(k[0], s)][(k, s)] = c[0]["lm"]
            if not of and R.diverged_at(r) is None and all(f"seen_{b}" in c[-1] for b in R.B):
                ms = min(c[-1][f"seen_{b}"] for b in R.B)
                if ms < CEIL:
                    ceil_low.append(f"{k[0]} {k[1]}@{k[2]} seed {s}: min final seen_bXY {ms:.2f} ({f})")
            want = SNAP.get(src, SNAP_OTHER)
            if r.get("code_sha") not in want:
                bad[(k, s)].append(f"code_sha {r.get('code_sha')} (want {sorted(want, key=str)}) in {f}")
            if not of and not (r["steps"] == STEPS and len(c) == EVALS and c[-1]["step"] == STEPS - 1):
                bad[(k, s)].append(f"steps {r['steps']} / {len(c)} evals (want {STEPS} / {EVALS}) in {f}")
            if r.get("grad_ckpt_on") is not None and r.get("n_params") is not None and r["grad_ckpt_on"] != (r["n_params"] > 150e6):
                bad[(k, s)].append(f"grad_ckpt_on {r['grad_ckpt_on']} at {r['n_params']} parameters in {f}")
            if r.get("n_params") is not None:
                assert r["n_params"] - V * r["args"]["d_model"] == SIZES[(r["args"]["d_model"], r["args"]["n_layers"],
                                                                          r["args"]["n_heads"], r["args"]["d_ff"])][1], f
            if k[1] == "win-ctx" and not of:
                if abs(c[-1].get("prior_on_frac", -1) - W_ON) > 1e-6:
                    bad[(k, s)].append(f"prior_on_frac {c[-1].get('prior_on_frac')} (want {W_ON:.6f}) in {f}")
                if "prior_events" in r and r["prior_events"] != W_EVENTS:
                    bad[(k, s)].append(f"prior_events {r['prior_events']} in {f}")
            if k[1] == "none" and c[-1].get("prior_used_frac", 0) != 0:
                bad[(k, s)].append(f"none run with prior_used_frac {c[-1].get('prior_used_frac')} in {f}")
for (size, s), d in lm0.items():
    med = float(np.median(list(d.values())))
    for ks, v in d.items():
        if not (math.isfinite(v) and abs(v - med) <= 1e-3):
            bad[ks].append(f"step-0 lm {v:.6f} is {v - med:+.4f} off the {size} seed-{s} median (pairing broken)")
        if size == "1B" and s in LM0_1B and not abs(v - LM0_1B[s]) <= 1e-3:
            bad[ks].append(f"step-0 lm {v:.6f} is {v - LM0_1B[s]:+.4f} off the B200 smoke reference {LM0_1B[s]}")
        if size in LM0 and s < 4 and not abs(v - LM0[size][s]) <= 1e-3:
            bad[ks].append(f"step-0 lm {v:.6f} is {v - LM0[size][s]:+.4f} off the reference {LM0[size][s]}")

for n in notes:
    print("NOTE", n)
for ks in sorted(bad, key=str):
    for why in bad[ks]:
        print(f"INTEGRITY {ks[0][0]} {ks[0][1]}@{ks[0][2]} seed {ks[1]}: {why}")
for c in ceil_low:
    print(f"CEILING (descriptive, threshold {CEIL} is this read-out's; the AUDIT names none) {c}")
print("== sources (priority order): runs in the ptlr grid / left out (reason: count)")
for d in DIRS:
    src = d.parent.name if d.name == "out" else d.name
    n_in = sum(len(v.get(src, {})) for v in cells.values())
    lo = {w: c for (s_, w), c in left_out.items() if s_ == src}
    print(f"  {src:10s} {str(d):60s} {n_in:3d} in grid; left out {sum(lo.values())}"
          + (f" ({', '.join(f'{w}: {c}' for w, c in sorted(lo.items()))})" if lo else "") + ("" if d.exists() else "  (missing)"))
if not cells:
    sys.exit("no run in the ptlr grid")

# ---------------------------------------------------------------- double path: pt999_summary's cell areas
checked = 0
for d in DIRS:
    src = d.parent.name if d.name == "out" else d.name
    keys = [k for k in cells if src in cells[k]]
    if not keys or not d.exists():
        continue
    o = subprocess.run([sys.executable, str(HERE / "pt999_summary.py"), str(d)], capture_output=True, text=True)
    if o.returncode:
        print(f"  (pt999_summary.py cannot read {src}'s tags: {o.stderr.strip().splitlines()[-1][:80]}; double path skipped there)")
        continue
    theirs = {m.group(1): float(m.group(2)) for m in re.finditer(r"^  (\S+)\s+n=\d+ area\s+(-?[\d.]+)±", o.stdout, re.M)}
    for k in keys:
        tag = next(iter(src_tag[(k, src)]))
        rs = [fr for fr in cells[k][src].values() if not fr[2]]
        if not rs:
            continue
        mine = np.mean([R.area(r) for _, r, _ in rs])
        if abs(theirs.get(tag, np.inf) - mine) > 0.05:
            sys.exit(f"double path: area of {src}/{tag}: pt999_summary {theirs.get(tag)} vs here {mine:.2f}")
        checked += 1
print(f"(double path: {checked} cell means agree with pt999_summary.py within 0.05)")


# ---------------------------------------------------------------- per cell
def vals(k, f, seeds=R.MAIN):
    pk = main[k] if seeds == R.MAIN else conf[k] if seeds == R.CONFIRM else selp[k] if seeds == SEL.get(k[0]) else None
    return None if pk is None else np.array([f(pk[1][s]) for s in seeds])


def withheld(keys, seeds=R.MAIN):
    return [f"{k[0]} {k[1]}@{k[2]} seed {s}: {why}" for k in keys for s in seeds for why in bad.get((k, s), [])]


def say_withheld(what, reasons):
    print(f"  {what}: WITHHELD: {reasons[0]}" + (f" (+{len(reasons) - 1} more)" if len(reasons) > 1 else ""))


def tt(k, theta, seeds=R.MAIN):
    return vals(k, lambda fr: R.t_theta(fr[1], theta), seeds)


def med_txt(t):
    return (f"{np.nanmedian(t):5.0f}" if np.isfinite(t).any() else "    -") + f" {int(np.isfinite(t).sum())}/{len(t)}"


ref_run = {}   # (size, seed) -> the counted none@base-lr run of that seed (the 'unstable' reference)
for size, _, base_lr in SIZES.values():
    for pk in ((selp.get((size, "none", base_lr)),) if size in SEL else ()) + (main.get((size, "none", base_lr)),
                                                                                 conf.get((size, "none", base_lr))):
        for s, fr in (pk[1].items() if pk else ()):
            if fr[1]["curve"]:
                ref_run[(size, s)] = fr[1]
ORDER = {v[0]: i for i, v in enumerate(SIZES.values())}

print("\n== cells (seeds 0-3 from the first source with all four): area, area5 (no b11), t90 / t50 (M tokens, median of "
      "the seeds that reach it, k/4), bpb (offline; '*' = in-training), divergence / instability; seeds 4-7; replicates")
grid = collections.defaultdict(dict)   # (size, arm) -> {lr: key}
for k in sorted(cells, key=lambda k: (ORDER[k[0]], ARMS.index(k[1]), float(k[2]))):
    grid[(k[0], k[1])][k[2]] = k
    pk = main[k]
    lab = f"{k[0]} {k[1]}@{k[2]}"
    if pk is None and k[0] in SEL and selp[k] is not None:   # ladder-1b: an lr with its selection seeds only
        ss = SEL[k[0]]
        src, rs = selp[k]
        ar = vals(k, area_of, ss)
        bp = [bpb_of(rs[s]) for s in ss]
        offl = all(b[0] is not None for b in bp)
        dv = [(s, R.diverged_at(rs[s][1])) for s in ss if R.diverged_at(rs[s][1]) is not None]
        un = [s for s in ss if not rs[s][2] and R.unstable(rs[s][1], ref_run.get((k[0], s)))]
        ow = [s for s in ss if rs[s][2]]
        print(f"  {lab:22s} {src:9s} {next(iter(src_tag[(k, src)])):26s} area {ar.mean():5.1f}±{ar.std():4.1f} "
              f"[selection seeds {ss[0]}-{ss[-1]} only] t90 {med_txt(tt(k, 0.9, ss))} t50 {med_txt(tt(k, 0.5, ss))} "
              f"bpb {np.mean([R.worst(b[0] if offl else b[1]) for b in bp]):.4f}{'' if offl else '*'}"
              + (f" | diverged {', '.join(f's{s}@{st}' for s, st in dv)}" if dv else "")
              + (f" | unstable (final val_loss > none@base + 0.5) s{', s'.join(map(str, un))}" if un else "")
              + (f" | own failure s{', s'.join(map(str, ow))}" if ow else "")
              + (" | [integrity: verdicts on this cell are withheld]" if withheld([k], ss) else ""))
        continue
    if pk is None:
        have = {src: sorted(rs) for src, rs in cells[k].items()}
        print(f"  {lab:22s} incomplete: seeds {have}")
        continue
    src, rs = pk
    ar, a5 = vals(k, area_of), vals(k, lambda fr: R.area(fr[1], R.B5) if not fr[2] else area_of(fr))
    bp = [bpb_of(rs[s]) for s in R.MAIN]
    offl = all(b[0] is not None for b in bp)
    bmean = np.mean([R.worst(b[0] if offl else b[1]) for b in bp])
    dv = [(s, R.diverged_at(rs[s][1])) for s in R.MAIN if R.diverged_at(rs[s][1]) is not None]
    un = [s for s in R.MAIN if not rs[s][2] and R.unstable(rs[s][1], ref_run.get((k[0], s)))]
    ow = [s for s in R.MAIN if rs[s][2]]
    cf = vals(k, area_of, R.CONFIRM)
    print(f"  {lab:22s} {src:9s} {next(iter(src_tag[(k, src)])):26s} area {ar.mean():5.1f}±{ar.std():4.1f} area5 {a5.mean():5.1f} "
          f"t90 {med_txt(tt(k, 0.9))} t50 {med_txt(tt(k, 0.5))} bpb {bmean:.4f}{'' if offl else '*'}"
          + (f" | diverged {', '.join(f's{s}@{st}' for s, st in dv)}" if dv else "")
          + (f" | unstable (final val_loss > none@base + 0.5) s{', s'.join(map(str, un))}" if un else "")
          + (f" | own failure s{', s'.join(map(str, ow))}" if ow else "")
          + (f" | seeds 4-7 ({conf[k][0]}) {cf.mean():5.1f}" if cf is not None else "")
          + (f" | selection seeds {SEL[k[0]][0]}-{SEL[k[0]][-1]} {vals(k, area_of, SEL[k[0]]).mean():5.1f}"
             if k[0] in SEL else "")
          + (" | [integrity: verdicts on this cell are withheld]" if withheld([k], R.MAIN + R.CONFIRM) else ""))
    for s2, rs2 in cells[k].items():
        if s2 == src or (conf[k] and s2 == conf[k][0] and not set(rs2) & set(rs)):
            continue   # the counted source, or the source of the confirm seeds only
        sh = sorted(set(rs2) & set(rs))
        x = [area_of(rs2[s]) for s in sorted(rs2)]
        txt = (f"; paired vs counted on seeds {sh}: {R.fmt_paired(R.paired([area_of(rs2[s]) for s in sh], [area_of(rs[s]) for s in sh]))}"
               if len(sh) > 1 else "")
        print(f"      replicate {s2}/{next(iter(src_tag[(k, s2)]))}: seeds {sorted(rs2)} mean area {np.mean(x):5.1f}{txt}")


# ---------------------------------------------------------------- rules
def best_of(size, arm):
    """('ok', best lr, selection) / ('incomplete' | 'withheld', reasons, None)"""
    g = grid.get((size, arm), {})
    if not g:
        return "missing", [], None
    ss, pkd = (SEL[size], selp) if size in SEL else (R.MAIN, main)   # ladder-1b: 1B selects on seeds 0-1
    if any(pkd[k] is None for k in g.values()):
        return "incomplete", [], None
    wh = withheld(list(g.values()), ss)
    if wh:
        return "withheld", wh, None
    inp = {lr: {"area": vals(k, area_of, ss), "off": [bpb_of(pkd[k][1][s])[0] for s in ss],
                "intr": [bpb_of(pkd[k][1][s])[1] for s in ss]} for lr, k in g.items()}
    sel = R.best_lr(inp, near_rule="top2")   # runs/ptlr/AUDIT.md rule 1 as pre-registered
    sel["proposed"] = R.best_lr(inp, near_rule="within")
    return "ok", sel["best"], sel


def ratio_line(label, kn, ka, theta):
    tn, ta = tt(kn, theta), tt(ka, theta)
    bud = R.budget_M(main[kn][1][0][1])
    sp, ci = R.speedup(tn, ta, bud), R.boot_ci(tn, ta, bud)
    print(f"  {label} R_{theta} = t{int(100 * theta)} none@{kn[2]} / {ka[1]}@{ka[2]}: {sp['text']}"
          + (f" ({ci['text']})" if sp["ratio"] is not None else "")
          + f"; medians {'>=' if R.censored_median(tn) else ''}{sp['med_ref']:.0f}M / {'>=' if R.censored_median(ta) else ''}"
          f"{sp['med_arm']:.0f}M; censored at {bud:.0f}M: none {sp['cen_ref']}/4, {ka[1]} {sp['cen_arm']}/4")
    return sp, ci


best, openbr, status = {}, {}, {}
for size, nne, base_lr in SIZES.values():
    if not any(k[0] == size for k in cells):
        continue
    print(f"\n## {size} (N_nonemb {nne / 1e6:.1f}M; divergence / instability reference: none@{base_lr} of the same seed)")
    for arm in ARMS:
        st, b, sel = best_of(size, arm)
        status[(size, arm)] = (st, b if st in ("ok", "withheld") else None)
        lrs = sorted(grid.get((size, arm), {}), key=float)
        if st == "missing":
            continue
        if st == "incomplete":
            print(f"  best lr {arm}: incomplete (grid {', '.join(lrs)})")
            continue
        if st == "withheld":
            say_withheld(f"best lr {arm}", b)
            best[(size, arm)] = ("withheld", b)
            continue
        best[(size, arm)] = ("ok", b)
        br = R.bracket(lrs, sel["mean"], b)
        if br and br[0] == "open":
            openbr[(size, arm)] = br[1]
        spread = max(sel["mean"].values()) - min(sel["mean"].values())
        gtxt = ", ".join(f"{lr} {sel['mean'][lr]:.1f}" for lr in lrs)
        print(f"  best lr {arm}: {b} (grid {gtxt}; tied (the top two if within 2): "
              f"{', '.join(sel['near'])}{'' if len(sel['near']) == 1 or sel['offline'] else '; tie broken on the in-training bpb *'})"
              + (f"  OPEN BRACKET at the {br[1]} edge: bracket not closed {'above' if br[1] == 'upper' else 'below'} {b}"
                 if br and br[0] == "open" else "")
              + (f"  (note: area argmax at the {br[1]} edge by <= 2: not an open bracket)" if br and br[0] == "note" else "")
              + ("" if len(lrs) > 1 else "  (single lr: untuned)")
              + (f"  [PROPOSED shared tie-break (every lr within 2: {', '.join(sel['proposed']['near'])}), pending the "
                 f"user's sign-off: {sel['proposed']['best']}]" if sel["proposed"]["best"] != b else "")
              + (f"  [rule 5: {'lr-flat' if spread <= 3 else 'not lr-flat'}, spread {spread:.1f}]" if len(lrs) > 1 else ""))
    bn = best.get((size, "none"))
    for arm in PRIORS:
        ba = best.get((size, arm))
        if not grid.get((size, arm)):
            continue
        lab = f"{arm}@best - none@best"
        if not bn or not ba:
            print(f"  {lab}: incomplete")
            continue
        if bn[0] == "withheld" or ba[0] == "withheld":
            say_withheld(lab, (bn[1] if bn[0] == "withheld" else []) + (ba[1] if ba[0] == "withheld" else []))
            continue
        kn, ka = grid[(size, "none")][bn[1]], grid[(size, arm)][ba[1]]
        if size in SEL:   # ladder-1b: the test is on seeds 0-3 at the picks; seeds 2-3 run only there
            print(f"  ({size}: the lrs were selected on seeds {SEL[size][0]}-{SEL[size][-1]}; the best-vs-best test below is "
                  f"on seeds 0-3 at the picks, confirm seeds 4-7)")
            if main[kn] is None or main[ka] is None:
                print(f"  {arm}@{ba[1]} - none@{bn[1]}: seeds 0-3 at the picks incomplete (seeds 2-3 pending)")
                continue
            wh = withheld([kn, ka], R.MAIN)
            if wh:
                say_withheld(f"{lab} (seeds 0-3)", wh)
                continue
        has_c = conf[kn] is not None and conf[ka] is not None
        wh = withheld([kn, ka], R.CONFIRM) if has_c else []
        if wh:
            say_withheld(f"{lab} (confirm seeds)", wh)
            continue
        sv = R.survives(vals(ka, area_of), vals(kn, area_of),
                        *((vals(ka, area_of, R.CONFIRM), vals(kn, area_of, R.CONFIRM)) if has_c else ()))
        prim = "primary" if arm == "win-ctx" else "secondary" + (" (its lr is untuned: conservative for the prior)"
                                                                 if len(grid[(size, arm)]) == 1 else "")
        print(f"  {arm}@{ba[1]} - none@{bn[1]} ({prim}): area {R.fmt_paired(sv['main'])}"
              + ("" if sv["confirm"] else "  PROVISIONAL (winner's curse: the lrs were selected on these seeds)")
              + (f"\n    confirm seeds 4-7 ({conf[ka][0]} / {conf[kn][0]}): area {R.fmt_paired(sv['confirm'])} -> "
                 f"{'agrees' if sv['confirm_ok'] else 'does NOT agree'}" if sv["confirm"] else "")
              + "".join(f"\n    NOT FINAL: {a}'s bracket is open at the {openbr[(size, a)]} edge (best lr {b_}; the AUDIT's "
                        f"conditional cells / 'bracket not closed' rule apply)" for a, b_ in (("none", bn[1]), (arm, ba[1]))
                        if (size, a) in openbr))
        sp9, _ = ratio_line("   ", kn, ka, 0.9)
        sp5, ci5 = ratio_line("   ", kn, ka, 0.5)
        if len(grid[(size, "none")]) < 2:
            print(f"    -> ({arm}) not a tuning test: none ran at a single lr (untuned) at {size}")
        elif size == "53M":
            print(f"    -> rule 3 ({arm}): speed-up {sv['verdict']}")
        else:
            ok5 = sp5["ratio"] is not None and ci5["lo"] > 1
            v = sv["verdict"] if ok5 or not sv["main_ok"] else "does NOT survive tuning (the R_0.5 90% CI does not exclude 1)"
            print(f"    -> rule 7 ({arm}): speed-up at {size} {v}; R_0.5 CI {'above 1' if ok5 else 'not above 1'}"
                  + ("; gap < 10: seeds 4-7 (8 runs) before any claim" if sv["main"][0] < 10 and not sv["confirm"] else ""))
        # descriptive: the lrs picked by median t90 (the pt2long headline's selection)
        sel_t = {}
        ss, pkd = (SEL[size], selp) if size in SEL else (R.MAIN, main)   # ladder-1b: over the selection seeds
        for a in ("none", arm):
            g = grid[(size, a)]
            med = {lr: R.median_at_budget(tt(k, 0.9, ss), R.budget_M(pkd[k][1][ss[0]][1])) for lr, k in g.items()}
            sel_t[a] = R.best_lr_t90(med, {lr: vals(k, area_of, ss).mean() for lr, k in g.items()})
        if (sel_t["none"], sel_t[arm]) != (bn[1], ba[1]):
            kt_n, kt_a = grid[(size, "none")][sel_t["none"]], grid[(size, arm)][sel_t[arm]]
            if main[kt_n] is None or main[kt_a] is None:
                print(f"    (descriptive, lrs by median t90 on seeds {ss[0]}-{ss[-1]}: none@{sel_t['none']} / "
                      f"{arm}@{sel_t[arm]}; no ratio: seeds 2-3 ran only at the area picks)")
            else:
                ratio_line("    (descriptive, lrs by median t90)", kt_n, kt_a, 0.9)

if A.json:   # ladder-1b: the selection job's input (runs/ptlr_1b/AUDIT.md); open = the edge of an open bracket
    Path(A.json).write_text(json.dumps({f"{s_}|{a_}": {"status": st, "best": b if st == "ok" else None,
                                                        "reasons": b if st == "withheld" else None,
                                                        "open": openbr.get((s_, a_)),
                                                        "grid": sorted(grid.get((s_, a_), {}), key=float)}
                                        for (s_, a_), (st, b) in status.items()}, indent=1))

# ---------------------------------------------------------------- rule 8: size trend
print("\n## size trend (rule 8): slope of log R vs log N_nonemb, best-vs-best, sizes where none has >= 2 lrs")
for arm in PRIORS:
    pts = []
    for (dm, nl, nh, dff), (size, nne, _) in SIZES.items():
        bn, ba = best.get((size, "none")), best.get((size, arm))
        if not grid.get((size, "none")) or not grid.get((size, arm)):
            continue
        if len(grid[(size, "none")]) < 2:
            print(f"  {arm}: {size} left out (none at a single lr: untuned)")
            continue
        # results audit 2026-10-19: a size whose none or arm bracket is still open has a PROVISIONAL best lr, so it
        # does not enter the trend (the 505M point alone made R_0.5 'grow')
        if (size, "none") in openbr or (size, arm) in openbr:
            ob = [a_ for a_ in ("none", arm) if (size, a_) in openbr]
            print(f"  {arm}: {size} left out (NOT FINAL: open bracket of {', '.join(ob)})")
            continue
        if not bn or not ba or bn[0] != "ok" or ba[0] != "ok":
            print(f"  {arm}: {size} {'withheld' if (bn and bn[0] == 'withheld') or (ba and ba[0] == 'withheld') else 'incomplete'}")
            pts = None
            break
        if size in SEL and (main[grid[(size, "none")][bn[1]]] is None or main[grid[(size, arm)][ba[1]]] is None):
            print(f"  {arm}: {size} incomplete (seeds 0-3 at the picks: seeds 2-3 pending)")   # ladder-1b
            pts = None
            break
        pts.append((size, nne, grid[(size, "none")][bn[1]], grid[(size, arm)][ba[1]]))
    if pts is None or len(pts) < 2:
        print(f"  {arm}: fewer than 2 sizes -> no slope" if pts is not None else f"  {arm}: no slope")
        continue
    for theta in (0.5, 0.9):
        T = [(tt(kn, theta), tt(ka, theta), R.budget_M(main[kn][1][0][1])) for _, _, kn, ka in pts]
        sps = [R.speedup(tn, ta, bud) for tn, ta, bud in T]
        x = np.log([nne for _, nne, _, _ in pts])
        untuned = [p[0] for p in pts if len(grid[(p[0], arm)]) == 1]
        lab = (f"  {arm} ({'primary' if arm == 'win-ctx' else 'secondary'} arm"
               + (f"; {', '.join(untuned)} {arm} at a single lr: UNTUNED, not a finding" if untuned else "")
               + f"), {'R_0.5 (the rule-8 ratio)' if theta == 0.5 else 'R_0.9 (descriptive)'}: R "
               + ", ".join(f"{p[0]} {sp['text'].split(' ')[0]}" for p, sp in zip(pts, sps)))
        if any(sp["ratio"] is None or sp["mark"] for sp in sps):
            print(lab + " -> slope not determinable (a ratio is a bound or missing)")
            continue
        y = np.log([sp["ratio"] for sp in sps])
        slope = np.polyfit(x, y, 1)[0]
        bs, rng = [], np.random.default_rng(0)
        for _ in range(R.N_BOOT):
            yb = []
            for tn, ta, bud in T:
                i = rng.integers(0, len(tn), len(tn))
                yb.append(np.log(R.median_at_budget(tn[i], bud) / R.median_at_budget(ta[i], bud)))
            bs.append(np.polyfit(x, yb, 1)[0])
        lo, hi = np.percentile(bs, 5), np.percentile(bs, 95)
        call = "grows with size (a trend)" if lo > 0 else "shrinks with size (a trend)" if hi < 0 else "no trend (CI includes 0)"
        if arm != "win-ctx":
            call = f"secondary{', untuned' if untuned else ''}: {call}"
        print(lab + f" -> slope {slope:+.3f} per log N_nonemb (90% CI {lo:+.3f} to {hi:+.3f}, seed bootstrap; n=4 per size: "
              f"indicative only) -> " + (call if theta == 0.5 else f"[{call}; descriptive]"))


# ---------------------------------------------------------------- T2: 53M 2% at lr 1e-3 (rules 9-11)
print("\n## T2, 53M 2% (--arith_lines 1), all arms at lr 1e-3, seeds 0-7 (rules 9-11; validity readout_rules.check_2pct)")
S8 = tuple(range(8))


def pick8(key):
    for src, rs in t2_runs.get(key, {}).items():
        if all(s in rs for s in S8):
            return src, rs
    return None


T2 = {arm: pick8(("53M", arm, "1e-3")) for arm in ARMS}
t2bad = {}
for arm in ARMS:
    if T2[arm] is None or T2["none"] is None:
        continue
    t2bad[arm] = [f"{T2[arm][0]} 2% {arm} seed {s}: {why}" for s in S8
                  for why in R.check_2pct(T2[arm][1][s][1], arm, T2["none"][1][s][1]["args"])]
    for why in t2bad[arm]:
        print(f"INTEGRITY {why}")


def t2vals(arm, f):
    return np.array([f(*T2[arm][1][s]) for s in S8])


def t2bpb(arm):
    """(offline bpb per seed or None if any is missing, in-training bpb per seed)"""
    rs = [T2[arm][1][s] for s in S8]
    off = [R.offline_bpb(off_entry(f, r), R.diverged_at(r) is not None) for f, r in rs]
    return (None if any(v is None for v in off) else np.array([R.worst(v) for v in off]),
            np.array([R.worst(r["curve"][-1]["val_bits_per_byte"]) for _, r in rs]))


for arm in ARMS:
    pk = T2[arm]
    if pk is None:
        have = {src: sorted(rs) for src, rs in t2_runs.get(("53M", arm, "1e-3"), {}).items()}
        print(f"  2% {arm:9s} incomplete: finished seeds {have or '{}'} (needs 0-7 in one source)")
        continue
    ar = t2vals(arm, lambda f, r: R.area(r))
    t9 = t2vals(arm, lambda f, r: R.t_theta(r, 0.9))
    off, intr = t2bpb(arm)
    used = t2vals(arm, lambda f, r: r["curve"][-1].get("prior_used_frac", np.nan))
    print(f"  2% {arm:9s} {pk[0]:9s} {next(iter(pk[1].values()))[1]['args']['tag'].lstrip('_'):14s} area {ar.mean():5.1f}±{ar.std():4.1f} "
          f"t90 reached {int(np.isfinite(t9).sum())}/8  used {np.nanmean(used):.2f}  bpb offline "
          + (f"{off.mean():.4f}" if off is not None else "-") + f" in-training {intr.mean():.4f}"
          + ("  [check if < 40: the AUDIT's expected range is 55-72]" if arm == "win-ctx" and ar.mean() < 40 else ""))


def t2_withheld(arms):
    return [w for a in arms for w in t2bad.get(a, [])]


if T2["none"] is None or T2["win-ctx"] is None:
    print("  rule 9 (win-ctx - none): incomplete")
elif t2_withheld(("none", "win-ctx")):
    say_withheld("rule 9 (win-ctx - none)", t2_withheld(("none", "win-ctx")))
else:
    d = t2vals("win-ctx", lambda f, r: R.area(r)) - t2vals("none", lambda f, r: R.area(r))
    wins, p_sf = int((d > 0).sum()), R.signflip_p(d)
    (ow, iw), (on, in_) = t2bpb("win-ctx"), t2bpb("none")
    dbo = None if ow is None or on is None else float(np.mean(ow - on))
    gate = ""
    if T2["gate-ctx"] is not None and not t2_withheld(("gate-ctx",)):
        og, ig = t2bpb("gate-ctx")
        gate = (f"; gate-ctx - none {np.mean(og - on):+.4f} offline (the plan's +0.045)" if og is not None and on is not None
                else f"; gate-ctx - none {np.mean(ig - in_):+.4f} in-training* (the plan's +0.045 is offline)")
    parts = {"Delta area >= +30": d.mean() >= 30, ">= 7/8 wins": wins >= 7, "sign-flip p < .05": p_sf < .05,
             "offline Delta bpb <= +0.015": None if dbo is None else dbo <= 0.015}
    print(f"  rule 9: win-ctx - none area {d.mean():+.1f} (per seed {', '.join(f'{v:+.1f}' for v in d)}; wins {wins}/8; "
          f"exact two-sided sign-flip p {p_sf:.4f}); Delta bpb offline " + ("-" if dbo is None else f"{dbo:+.4f}")
          + f", in-training {np.mean(iw - in_):+.4f} (65k tokens, descriptive){gate}")
    if parts["offline Delta bpb <= +0.015"] is None:
        print("    -> rule 9 incomplete: offline (full val) bpb missing for some 2pct-none / 2pct-win-ctx run (offline_lm.py "
              "of the ptlr snapshot, files via --offline); the in-training bpb does not decide it")
    elif all(parts.values()):
        print("    -> rule 9 met: the 2% row uses the window; the gate is written as \"degenerates to a whole-run prior at 2%\""
              + (f" (gate used {t2vals('gate-ctx', lambda f, r: r['curve'][-1].get('prior_used_frac', np.nan)).mean():.2f})"
                 if T2["gate-ctx"] is not None else ""))
    else:
        print(f"    -> rule 9 NOT met ({', '.join(k for k, v in parts.items() if not v)} failed): T2 does not support "
              f"switching the 2% row to the window")
if T2["gate-ctx"] is None or T2["win-ctx"] is None:
    print("  rule 10 (win-ctx - gate-ctx): incomplete")
elif t2_withheld(("gate-ctx", "win-ctx")):
    say_withheld("rule 10 (win-ctx - gate-ctx)", t2_withheld(("gate-ctx", "win-ctx")))
else:
    d = t2vals("win-ctx", lambda f, r: R.area(r)) - t2vals("gate-ctx", lambda f, r: R.area(r))
    print(f"  rule 10: win-ctx - gate-ctx area paired mean {d.mean():+.1f} (wins {int((d > 0).sum())}/8) -> "
          + ("< -10: a trade-off is reported, no winner" if d.mean() < -10 else ">= -10: no trade-off by rule 10"))
if T2["none"] is not None:
    na = t2vals("none", lambda f, r: R.area(r)).mean()
    nt = int(np.isfinite(t2vals("none", lambda f, r: R.t_theta(r, 0.9))).sum())
    print(f"  rule 11: no speed ratio at 2% (none's area {na:.1f}" + (" < 20" if na < 20 else " >= 20: rule 11's premise no longer "
          "holds; still no ratio here, the user's call") + f"; none reaches 0.9 in {nt}/8); every Delta above is \"at lr 1e-3 "
          "for both arms\", not a tuned comparison (the 2% best-vs-best is Burst pt2main)")
