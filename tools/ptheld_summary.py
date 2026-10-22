"""Pre-registered read-out of the Torch runs/ptheld (wall 1, "data missing / long tail": can a number prior let a 53M model
add with numbers H it never saw in an arithmetic line?). Intent spec and rules: llmadd_torch/runs/ptheld/AUDIT.md (R1-R5,
amended 2026-10-19 after the fresh-agent audit, before any ptheld result existed). Written and tested on fake results
(--selftest) before any result existed; the rules below are the AUDIT's, in code.

Cells are keyed on r["args"] (arm = target x site; lr; wd; tokens), every other argument must equal the ptheld base (53M,
seq 512, bs 32, 7% = --arith_lines 4, --held_operands 50, lambda 5, window 0.05,0.4 for the prior arms, eval every 250 x
150, tied embeddings, whole-number tokens). final3 = the mean over the LAST THREE evals of a run (1x: steps 17750 / 18000 /
18309; 4x: 72750 / 73000 / 73241). held6 = the mean of the six filtered held_bXY. acc_H1 = H_one exact match; logp_H1 =
H_one teacher-forced log-prob of the answer (nats, summed over the answer tokens and the newline).
  informative seed (controls and LC, R1 b / d): held6 final3 >= 0.80, not diverged (readout_rules.diverged_at), not
      unstable (final val_loss <= the same seed's none@2e-3 + 0.5 nats; readout_rules.unstable); otherwise 'uninformative'
      and left out of (b) / (d) / the paired tests. An X seed (win-text / win-emb) that is not informative counts as a
      failure of R1(a) and keeps its measured value in the paired tests.
  R1 (X in win-text, win-emb): (a) acc_H1 final3 >= 0.50 in >= 3/4 of X's seeds; (b) for Y in none@2e-3, none@8e-3,
      win-ctx@2e-3: >= 3/4 of Y's seeds informative and every informative seed <= 0.10; (c) X - Y over the seeds where Y is
      informative (>= 3): wins >= ceil(3/4 m), one-sided paired t; p_X = max over the three Y (intersection-union: X must
      beat every Y); Holm over the two X: the smaller p_X < .025, the larger < .05; (d) LC cells (none@1.6e-2,
      none@8e-3-wd0.01, none@8e-3-wd1, none@8e-3-4x): 'holds' (>= 3/4 informative, all <= 0.10), 'one seed' (exactly one
      informative seed > 0.10: <= 0.50 -> the claim stands, qualified; > 0.50 -> as 'fails'), 'fails' (>= 2 seeds > 0.10 ->
      'within this budget / faster', never 'cannot'), 'uninformative' / 'missing' -> PROVISIONAL.
      Bracket on logp_H1 final3 (the continuous score; exact match sits at the floor): lr axis none@2e-3 / 8e-3 / 1.6e-2
      (wd 0.1), wd axis none@8e-3 wd 0.01 / 0.1 / 1.0; OPEN when the best mean is at an edge and beats its neighbour by
      > 0.25 nats with >= 3/4 paired wins -> one more step outward (lr 1e-3 / 3.2e-2, wd 0.001 / 3.0) before any
      'cannot'. Also on acc_H1 when the best mean exceeds 0.05. 4x trend: none@8e-3-4x - none@8e-3 on logp_H1 final3
      >= +0.5 nats with >= 3/4 wins, or the OLS slope of logp_H1 over the last quarter of the 4x run > 0 in >= 3/4 seeds
      -> 'cannot' only 'within 4x the budget'. Transient: curve max >= 0.5 in >= 3/4 seeds with final3 < 0.5.
  R2: if win-ctx fails (b), X also needs X - win-ctx paired mean >= +0.30 (>= 3/4 wins, p inside the same Holm).
  R3 (mechanism; arm win-emb-hremap, u_h = held_operands.remap_H): S = mean over h of the share of h's H_one rows whose
      greedy answer equals u_h + y; at the window end (the last eval with the prior on, step 7250) and at final3 (the
      three evals pooled). p = h-level randomisation test: the trainer's own u-sampler re-drawn N times on other streams
      (u_h is exchangeable with its decoys under H0), p = (1 + #{S_decoy >= S}) / (1 + N). 'supports' iff >= 3/4 seeds
      have p < .01 AND S(hremap) > S(win-emb) AND S(hremap) > S(none@2e-3) (same u); 'against' iff p >= .05 in >= 3/4
      seeds; else 'inconclusive'; a seed whose hremap acc_H1 >= 0.90 (or < 10 h with a wrong row) is 'not testable'.
      final3 supports -> 'persistent'; only the window end -> 'while the prior is on'. The re-draw with stream
      6400 + seed must reproduce the logged remap_H (double path; else WITHHELD).
      R3b (emb geometry): win-emb - win-emb-hremap on acc_H1 final3, paired mean >= +0.20 and >= 3/4 wins -> "h's own
      helix value"; else "a prior on the number embeddings, not specific to h's value". R3c (text): win-text -
      win-text-farshuf, the same rule -> "helix geometry"; farshuf acc_H1 >= 0.5 in >= 3/4 -> "not geometry-specific".
      farshuf's share of wrong answers = pi_far(h) + y: descriptive only (the audit: a shuffle of ALL numbers is not
      predicted to give pi(h) + y).
  R4 (descriptive): acc_H1 per stratum q1..q5; per seed the partial Spearman of per-h accuracy (12 rows x final3) on
      log10(text-site count), controlling for the value h and h's share of 4-digit answers (rank-residual correlation);
      high- vs low-count h inside the value bands 100-399 / 400-699 / 700-999; the contrast win-text - win-emb per h on
      the same partial Spearman (win-emb's own rho is NOT a null: count and value are rank-correlated -0.86 over
      100..999). win-text also against its logged exposure (held_exposure.st_on).
  R5 (descriptive): H_both, H_ans single / chunk, held6, val bits per byte vs none@2e-3, win-emb acc_H1 at the window end.
  R1-best (amendment PLR, 2026-10-20; implemented 2026-10-21 before any PLR result was read): per prior arm (win-text,
      win-emb), the lr in {1e-3, 2e-3, 4e-3} with the highest mean H_one area over seeds 0-3 (area = the mean of acc_H1
      over all evals of a run, the evals being evenly spaced); means within 0.02 of the top -> the lower mean offline
      bpb (val_bits_per_byte final3); bracket closed iff 2e-3 is chosen, else OPEN at that edge. R1 is then re-run with
      those two cells as X on seeds 0-3 of every cell (the selection seeds: a winner's-curse-biased reading). The
      CONF seeds (4-7) exist only at 2e-3: if 2e-3 is chosen, R1 with that X cut to seeds 4-7 (the controls as in R1)
      is the confirmation, else "CONF seeds not run at the chosen lr". R1 itself (the prior arms at 2e-3, all seeds) is unchanged.
Validity (AUDIT 'Controls'; a failure withholds every verdict that uses the cell): code_sha, steps / evals, base args,
held_lines_scanned = 128 x (step + 1), held_rej > 0, pred_mismatch = 0 (else R3 withheld for the run),
logp_dp_maxdiff <= 0.05, prior_on_frac 6408 / 18310 and prior_events [[916, 1], [7324, 0]] for window arms (1x), none
never on, emb arms grad_prior_emb_rows = 1000; per seed across arms: H, counts, H_one pairs, remap_H, perm_far_H equal,
step-0 lm within 1e-3, held_data_sha equal at every common eval step.
    python tools/ptheld_summary.py [--root /scratch/yh6210/llmadd_torch/runs/ptheld/out] [--code_sha <sha>] [--n_rand 10000]
    python tools/ptheld_summary.py --selftest
"""
import argparse
import glob
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_rules as RR   # noqa: E402

BUCK = ("b11", "b12", "b13", "b22", "b23", "b33")
BASE = dict(arith_lines=4, held_operands=50, lam=5.0, bs=32, seq=512, d_model=384, n_layers=6, n_heads=6, d_ff=1536,
            tokenization="whole", n_eval=150, eval_every=250, geo_layers=None, untie=False, text_min=16)
ARMS = {("none", None): "none", ("helix3", "ctx"): "win-ctx", ("helix3", "text"): "win-text", ("helix3", "emb"): "win-emb",
        ("helix3_hremap", "emb"): "win-emb-hremap", ("helix3_farshuf", "text"): "win-text-farshuf"}
X_CELLS = ("win-text@2e-3", "win-emb@2e-3")
Y_CELLS = ("none@2e-3", "none@8e-3", "win-ctx@2e-3")
LC_CELLS = ("none@1.6e-2", "none@8e-3-wd0.01", "none@8e-3-wd1", "none@8e-3-4x")
LR_AXIS = ("none@2e-3", "none@8e-3", "none@1.6e-2")
WD_AXIS = ("none@8e-3-wd0.01", "none@8e-3", "none@8e-3-wd1")
OUTWARD = {"none@2e-3": "lr 1e-3", "none@1.6e-2": "lr 3.2e-2", "none@8e-3-wd0.01": "wd 0.001", "none@8e-3-wd1": "wd 3.0"}
STEPS = {3e8: 18310, 1.2e9: 73242}
WIN_END = 7250
ALPHA = (0.025, 0.05)
LINES_PER_STEP = 32 * 4


def lrs(x):
    m, e = f"{x:.1e}".split("e")
    return f"{float(m):g}e{int(e)}"


def cell_of(a):
    """cell name of a run's args, or None (not a ptheld cell)"""
    arm = ARMS.get((a["target"], None if a["target"] == "none" else a.get("site", "ctx")))
    if arm is None:
        return None
    wd = a.get("wd", 0.1)
    return (f"{arm}@{lrs(a['lr'])}" + (f"-wd{wd:g}" if wd != 0.1 else "") + ("-4x" if a["tokens"] == 1.2e9 else ""))


def n_evals(steps):
    return len(range(0, steps, 250)) + (0 if (steps - 1) % 250 == 0 else 1)


def final3(r, k):
    return float(np.mean([e[k] for e in r["curve"][-3:]]))


def held6(e):
    return float(np.mean([e[f"held_{b}"] for b in BUCK]))


def finite(v):
    """a logged number: present, numeric (not bool), finite"""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_run(r, code_sha):
    """reasons why a run is not valid (empty = valid); r3_ok False if its greedy answers may not be used"""
    a, bad, r3_ok = r["args"], [], True
    if code_sha and r.get("code_sha") != code_sha:
        bad.append(f"code_sha {r.get('code_sha')} != {code_sha}")
    for k, v in BASE.items():
        if a.get(k, v) != v:
            bad.append(f"arg {k}={a.get(k)} != {v}")
    if a["tokens"] not in STEPS or r["steps"] != STEPS[a["tokens"]]:
        bad.append(f"steps {r['steps']} for tokens {a['tokens']}")
    elif len(r["curve"]) != n_evals(r["steps"]) or not RR.finished(r):
        bad.append(f"evals {len(r['curve'])} != {n_evals(r['steps'])} or unfinished")
    if a["target"] != "none" and (a.get("schedule"), a.get("window")) != ("window", "0.05,0.4"):
        bad.append(f"schedule {a.get('schedule')} {a.get('window')}")
    for e in r["curve"]:
        if e.get("held_lines_scanned") != LINES_PER_STEP * (e["step"] + 1):
            bad.append(f"held_lines_scanned {e.get('held_lines_scanned')} at step {e['step']}"); break
        # AUDIT R05 (2026-10-22, external review): a missing or non-finite diagnostic is never a pass. The emb row count is
        # logged exactly when the prior is on (prior_on), so it is required there; prior_on itself must be logged.
        if a["site"] == "emb" and a["target"] != "none" and (e.get("prior_on") is None or
                                                             (e["prior_on"] and e.get("grad_prior_emb_rows") != 1000)):
            bad.append(f"grad_prior_emb_rows {e.get('grad_prior_emb_rows')} (prior_on {e.get('prior_on')}) at step {e['step']}"); break
        # AUDIT amendment DP (2026-10-20): step 0 (near-uniform model) <= 0.05 for every H set = the span check the hits
        # share; later evals <= 0.5 nat per scored token (H_one 1.0, H_both / H_ans 1.5) = two bf16 ulps per token on GPU
        dp_tol = {"H_one": 1.0, "H_both": 1.5, "H_ans": 1.5}
        dp_bad = [s for s in dp_tol if not (finite(e.get(f"logp_dp_maxdiff_{s}")) and
                                            e[f"logp_dp_maxdiff_{s}"] <= (0.05 if e["step"] == 0 else dp_tol[s]))]
        if dp_bad:
            bad.append(f"logp double path {[(s, e.get(f'logp_dp_maxdiff_{s}')) for s in dp_bad]} at step {e['step']}"); break
        if any(not finite(e.get(f"pred_mismatch_{s}")) or e[f"pred_mismatch_{s}"] != 0 for s in ("H_one", "H_both", "H_ans")):
            r3_ok = False   # a missing count cannot vouch for the greedy answers either
    if not r["curve"][-1].get("held_rej", 0) > 0:
        bad.append("held_rej 0")
    last = r["curve"][-1]
    if a["target"] == "none":
        if last.get("prior_used_frac", 0) != 0:
            bad.append(f"none prior_used_frac {last.get('prior_used_frac')}")
    elif a["tokens"] == 3e8:
        if abs(last.get("prior_on_frac", -1) - 6408 / 18310) > 1e-9 or r.get("prior_events") != [[916, 1], [7324, 0]]:
            bad.append(f"window prior_on_frac {last.get('prior_on_frac')} events {r.get('prior_events')}")
    return bad, r3_ok


def load(root, code_sha):
    """{cell: {seed: run}}, problems {cell: [reasons]}, r3 flags {(cell, seed): ok}"""
    runs, probs, r3ok, other = {}, {}, {}, 0
    for f in sorted(glob.glob(str(Path(root) / "*" / "run_*.json"))):
        r = json.loads(Path(f).read_text())
        c = cell_of(r["args"])
        if c is None:
            other += 1; continue
        s = r["args"]["seed"]
        if s in runs.get(c, {}):
            probs.setdefault(c, []).append(f"seed {s} twice ({f})"); continue
        bad, ok = check_run(r, code_sha)
        runs.setdefault(c, {})[s] = r; r3ok[(c, s)] = ok
        if bad:
            probs.setdefault(c, []).extend(f"s{s}: {b}" for b in bad)
    if other:
        print(f"[note] {other} run JSONs are not ptheld cells (left out)")
    # per seed across cells: the same H, counts, eval pairs, remap, far perm, step-0 lm, data stream
    seeds = sorted({s for c in runs for s in runs[c]})
    for s in seeds:
        rs = [(c, runs[c][s]) for c in runs if s in runs[c]]
        c0, r0 = rs[0]
        h0 = r0["held_operands"]
        for c, r in rs[1:]:
            h = r["held_operands"]
            for k in ("H", "count_text_site", "remap_H", "perm_far_H"):
                if h.get(k) != h0.get(k):
                    probs.setdefault(c, []).append(f"s{s}: held_operands.{k} differs from {c0}")
            if h["sets"]["H_one"]["pairs"] != h0["sets"]["H_one"]["pairs"]:
                probs.setdefault(c, []).append(f"s{s}: H_one pairs differ from {c0}")
            if abs(r["curve"][0]["lm"] - r0["curve"][0]["lm"]) > 1e-3:
                probs.setdefault(c, []).append(f"s{s}: step-0 lm {r['curve'][0]['lm']:.4f} vs {r0['curve'][0]['lm']:.4f} ({c0})")
            sha0 = {e["step"]: e.get("held_data_sha") for e in r0["curve"]}
            diff = [e["step"] for e in r["curve"] if e["step"] in sha0 and e.get("held_data_sha") != sha0[e["step"]]]
            if diff:
                probs.setdefault(c, []).append(f"s{s}: held_data_sha differs from {c0} from step {diff[0]}")
    return runs, probs, r3ok


# ---------------------------------------------------------------------------------------------------------------- R1
def informative(r, ref):
    return (held6_final3(r) >= 0.80 and RR.diverged_at(r) is None and not RR.unstable(r, ref))


def held6_final3(r):
    return float(np.mean([held6(e) for e in r["curve"][-3:]]))


def ptest(d):
    """one-sided paired t p (H1: mean > 0); constant differences -> 0 if positive else 1"""
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan")
    if np.allclose(d, d[0]):
        return 0.0 if d[0] > 0 else 1.0
    return float(stats.ttest_1samp(d, 0.0, alternative="greater").pvalue)


def need_wins(m):
    return math.ceil(0.75 * m)


def r1(runs, probs, xcells=X_CELLS):
    out, lines = {}, []
    ref = runs.get("none@2e-3", {})
    get = lambda c, s, k="H_one": final3(runs[c][s], k)
    missing = [c for c in tuple(xcells) + Y_CELLS if c not in runs or len(runs[c]) < 3]
    if missing:
        lines.append(f"R1 INCOMPLETE: missing cells {missing}")
        return {"status": "incomplete"}, lines
    withheld = [c for c in tuple(xcells) + Y_CELLS if c in probs]
    if withheld:
        lines.append(f"R1 WITHHELD: invalid runs in {withheld}")
        return {"status": "withheld"}, lines
    info = {c: {s: informative(r, ref.get(s)) for s, r in runs[c].items()} for c in Y_CELLS + LC_CELLS if c in runs}
    yb = {}
    for y in Y_CELLS:
        ss = sorted(runs[y]); inf = [s for s in ss if info[y][s]]
        vals = {s: get(y, s) for s in ss}
        ok_n = len(inf) >= need_wins(len(ss))
        yb[y] = "holds" if ok_n and all(vals[s] <= 0.10 for s in inf) else ("uninformative" if not ok_n else "fails")
        lines.append(f"R1(b) {y}: acc_H1 final3 {[round(vals[s], 3) for s in ss]}, informative {len(inf)}/{len(ss)} -> {yb[y]}")
    px, detail = {}, {}
    for x in xcells:
        ss = sorted(runs[x])
        vals = {s: get(x, s) for s in ss}
        inf_x = {s: held6_final3(runs[x][s]) >= 0.80 and RR.diverged_at(runs[x][s]) is None for s in ss}
        a_ok = sum(vals[s] >= 0.50 and inf_x[s] for s in ss) >= need_wins(len(ss))
        cmax = {s: max(e["H_one"] for e in runs[x][s]["curve"]) for s in ss}
        transient = (not a_ok) and sum(cmax[s] >= 0.5 for s in ss) >= need_wins(len(ss))
        ps, cs = {}, {}
        for y in Y_CELLS:
            common = [s for s in ss if s in runs[y] and info[y][s]]
            d = [vals[s] - get(y, s) for s in common]
            wins = sum(v > 0 for v in d)
            ps[y] = ptest(d) if len(common) >= 3 else float("nan")
            cs[y] = (len(common) >= 3 and wins >= need_wins(len(common)), float(np.mean(d)) if d else float("nan"), wins, len(common))
        r2_need = yb["win-ctx@2e-3"] == "fails"
        r2_ok = (not r2_need) or (cs["win-ctx@2e-3"][1] >= 0.30 and cs["win-ctx@2e-3"][0])
        px[x] = max(ps.values()) if all(not math.isnan(p) for p in ps.values()) else float("nan")
        detail[x] = dict(vals=vals, a=a_ok, transient=transient, cs=cs, ps=ps, r2_need=r2_need, r2_ok=r2_ok,
                         wins_ok=all(v[0] for v in cs.values()))
        lines.append(f"R1(a) {x}: acc_H1 final3 {[round(vals[s], 3) for s in ss]} (held6 >= 0.80 {sum(inf_x.values())}/{len(ss)}) "
                     f"-> {'holds' if a_ok else ('TRANSIENT' if transient else 'fails')}")
        for y in Y_CELLS:
            ok, md, w, m = cs[y]
            lines.append(f"R1(c) {x} - {y}: mean {md:+.3f}, wins {w}/{m}, one-sided paired t p {ps[y]:.4f}")
    # Holm over the two X (each X's p = max over its Y: intersection-union)
    order = sorted(xcells, key=lambda x: (math.isnan(px[x]), px[x]))
    holm, prev = {}, True
    for i, x in enumerate(order):
        holm[x] = prev and not math.isnan(px[x]) and px[x] < ALPHA[i]
        prev = holm[x]
    # (d) lesson c
    lc, lc_words = {}, []
    for c in LC_CELLS:
        if c not in runs:
            lc[c] = "missing"; continue
        if c in probs:
            lc[c] = "withheld"; continue
        ss = sorted(runs[c]); inf = [s for s in ss if info[c][s]]
        vals = {s: get(c, s) for s in ss}
        over = [s for s in inf if vals[s] > 0.10]
        if len(inf) < need_wins(len(ss)):
            lc[c] = "uninformative"
        elif not over:
            lc[c] = "holds"
        elif len(over) == 1 and vals[over[0]] <= 0.50:
            lc[c] = f"one seed (s{over[0]} {vals[over[0]]:.2f})"
        else:
            lc[c] = "fails"
        lines.append(f"R1(d) {c}: acc_H1 final3 {[round(vals[s], 3) for s in ss]}, informative {len(inf)}/{len(ss)} -> {lc[c]}")
    br = bracket(runs, probs, info)
    lines += br["lines"]
    NONE_Y = Y_CELLS[:2]
    for x in xcells:
        d_ = detail[x]
        # (b): both none cells hold; win-ctx holds, or fails and R2's extra margin holds (site specificity)
        allb = all(yb[y] == "holds" for y in NONE_Y) and (yb["win-ctx@2e-3"] == "holds" or
                                                          (yb["win-ctx@2e-3"] == "fails" and d_["r2_ok"]))
        if d_["a"] and allb and d_["wins_ok"] and holm[x]:
            if any(v == "fails" for v in lc.values()):
                v = "the prior makes H usable within this budget / faster (an LC cell of none also uses H)"
            elif any(v in ("missing", "uninformative", "withheld") for v in lc.values()) or br["open"] or br["trend_incomplete"]:
                v = "HOLDS, PROVISIONAL (at wd 0.1, 3e8 tokens, lr 2e-3 / 8e-3; lesson c incomplete" + \
                    (", bracket OPEN" if br["open"] else "") + (", 4x trend on < 3 seeds" if br["trend_incomplete"] else "") + ")"
            else:
                v = "HOLDS" + (" (" + "; ".join(f"{c}: {s}" for c, s in lc.items() if s.startswith("one seed")) + ")"
                               if any(s.startswith("one seed") for s in lc.values()) else "")
            if br["trend"]:
                v += "; 'cannot' only within 4x the budget (H likelihood still rising)"
            if yb["win-ctx@2e-3"] == "fails":
                v += f"; R2: win-ctx also uses H, X - win-ctx {d_['cs']['win-ctx@2e-3'][1]:+.2f}"
        elif any(yb[y] == "uninformative" for y in Y_CELLS):
            v = "NOT ESTABLISHED: uninformative control(s) " + ", ".join(y for y in Y_CELLS if yb[y] == "uninformative")
        elif d_["a"] and any(yb[y] == "fails" for y in Y_CELLS):
            v = "no wall claim: a control uses H (" + ", ".join(y for y in Y_CELLS if yb[y] == "fails") + \
                f"); the prior raises acc_H1 by {np.mean([d_['cs'][y][1] for y in Y_CELLS]):+.2f} (paired)"
        elif (not d_["a"]) and all(d_["cs"][y][0] for y in NONE_Y) and max(d_["ps"][y] for y in NONE_Y) < 0.05:
            v = f"partial: the prior raises H accuracy to {np.mean(list(d_['vals'].values())):.0%}"
        elif d_["transient"]:
            v = "transient (curve max >= 0.5, final3 < 0.5): not the claim"
        else:
            v = "fails"
        if x.startswith("win-emb@") and v.startswith("HOLDS"):
            v += " [wording: injecting h's value coordinates into its embedding substitutes for task exposure]"
        out[x] = v
        lines.append(f"R1 {x}: {v}  (p_X = max_Y p = {px[x]:.4f}, Holm alpha {ALPHA[order.index(x)]}{'; R2 needed' if d_['r2_need'] else ''})")
    out.update(lc={c: v for c, v in lc.items()}, yb=yb, bracket_open=br["open"], trend=br["trend"], status="done")
    return out, lines


PLR_LRS = ("1e-3", "2e-3", "4e-3")
SEL_SEEDS, CONF_SEEDS = (0, 1, 2, 3), (4, 5, 6, 7)


def h_area(r):
    return float(np.mean([e["H_one"] for e in r["curve"]]))


def seeds_view(runs, seeds, cells=None):
    """runs with the given cells (default: all) cut to `seeds`; a cell left with no seed is dropped (-> 'missing')"""
    v = {c: ({s: r for s, r in rs.items() if s in seeds} if cells is None or c in cells else dict(rs)) for c, rs in runs.items()}
    return {c: rs for c, rs in v.items() if rs}


def r1_best(runs, probs, o1):
    """amendment PLR: each prior arm at its best lr (seeds 0-3), then R1 with the same thresholds"""
    lines, best, br = [], {}, {}
    for arm in ("win-text", "win-emb"):
        cells = [f"{arm}@{lr}" for lr in PLR_LRS]
        miss = [c for c in cells if c not in runs or not all(s in runs[c] for s in SEL_SEEDS)]
        bad = [c for c in cells if c in probs]
        if miss or bad:
            lines.append(f"R1-best {arm}: " + (f"INCOMPLETE (missing / short of seeds 0-3: {miss})" if miss else f"WITHHELD (invalid runs in {bad})"))
            return {"status": "incomplete" if miss else "withheld"}, lines
        area = {c: float(np.mean([h_area(runs[c][s]) for s in SEL_SEEDS])) for c in cells}
        # AUDIT fix (2026-10-21, review audit_ptheld_r1best.md, before any PLR result): the tie-break bpb is the in-training
        # val_bits_per_byte final3 (65k tokens): ptheld computed no offline bpb, a deviation from the amendment's wording;
        # a non-finite bpb counts as the worst (RR.worst), never as a tie winner
        bpb = {c: float(np.mean([RR.worst(final3(runs[c][s], "val_bits_per_byte")) for s in SEL_SEEDS])) for c in cells}
        f3 = {c: float(np.mean([final3(runs[c][s], "H_one") for s in SEL_SEEDS])) for c in cells}
        top = max(area.values())
        tied = [c for c in cells if area[c] >= top - 0.02]
        b = min(tied, key=lambda c: (bpb[c], cells.index(c)))
        best[arm] = b
        i = cells.index(b)
        br[arm] = "closed" if 0 < i < len(cells) - 1 else f"OPEN ({'bottom' if i == 0 else 'top'} edge, lr {PLR_LRS[i]})"
        lines.append(f"R1-best {arm}: H_one area (seeds 0-3, the pre-registered selection score; includes the window) "
                     + ", ".join(f"{c.split('@')[1]} {area[c]:.3f}" for c in cells)
                     + "; acc_H1 final3 (seeds 0-3, what R1 tests; shown, not used) " + ", ".join(f"{c.split('@')[1]} {f3[c]:.3f}" for c in cells)
                     + "; in-training bpb " + ", ".join(f"{c.split('@')[1]} {bpb[c]:.4f}" for c in cells)
                     + f"; within 0.02 of the top: {[c.split('@')[1] for c in tied]} -> best {b} (bracket {br[arm]})"
                     + ("; tie decided by the in-training bpb (offline bpb not computed), then the lower lr" if len(tied) > 1 else ""))
    xb = tuple(best[a] for a in ("win-text", "win-emb"))
    # the selection view cuts only X to seeds 0-3; the controls read the same seeds as R1 (paired tests pair on 0-3)
    o_sel, l_sel = r1(seeds_view(runs, SEL_SEEDS, cells=xb), probs, xcells=xb)
    lines += ["R1-best [seeds 0-3, the selection seeds] " + l for l in l_sel]
    out = {"status": "done", "best": best, "bracket": br, "sel": o_sel}
    # CONF: only the chosen X is cut to seeds 4-7 (the paired tests then pair on 4-7; Y's (b), LC, bracket and 4x trend
    # read the same seeds as R1: they are not selected); an X chosen at another lr has no CONF seeds and is left out
    # (one X -> Holm alpha .025)
    conf_x = tuple(x for x in xb if x.endswith("@2e-3"))
    if conf_x:
        o_c, l_c = r1(seeds_view(runs, CONF_SEEDS, cells=conf_x), probs, xcells=conf_x)
        lines += ["R1-best [seeds 4-7, CONF] " + l for l in l_c]
        out["conf"] = {x: o_c.get(x) for x in conf_x}
    for x in xb:
        if not x.endswith("@2e-3"):
            lines.append(f"R1-best {x}: CONF seeds not run at the chosen lr (selection-seed reading only)")
    # rescue (rescue_standard.md) per arm = every part holds, unqualified (not PROVISIONAL, no 'within'):
    #   R1 at 2e-3 HOLDS; R1-best on the selection seeds HOLDS; CONF agrees (:= the CONF verdict HOLDS; no CONF run ->
    #   not met); (d) the arm's shuffle control (text: win-text-farshuf, emb: win-emb-hremap, both @2e-3) is informative
    #   in >= 3/4 of its seeds and < 0.50 on acc_H1 final3 in >= 3/4 of its informative seeds
    clean = lambda v: isinstance(v, str) and v.startswith("HOLDS") and "PROVISIONAL" not in v and "within" not in v
    ref = runs.get("none@2e-3", {})
    out["rescue"] = {}
    for arm, x, ctl in (("win-text", best["win-text"], "win-text-farshuf@2e-3"), ("win-emb", best["win-emb"], "win-emb-hremap@2e-3")):
        failed = []
        if not clean(o1.get(f"{arm}@2e-3")):
            failed.append(f"R1 {arm}@2e-3: {o1.get(f'{arm}@2e-3', o1.get('status'))}")
        if not clean(o_sel.get(x)):
            failed.append(f"R1-best (seeds 0-3) {x}: {o_sel.get(x, o_sel.get('status'))}")
        cv = out.get("conf", {}).get(x)
        if not clean(cv):
            failed.append(f"CONF {x}: {cv if cv is not None else 'not run at the chosen lr'}")
        if ctl not in runs or ctl in probs:
            failed.append(f"(d) {ctl}: missing / withheld")
        else:
            ss = sorted(runs[ctl]); inf = [s for s in ss if informative(runs[ctl][s], ref.get(s))]
            low = sum(final3(runs[ctl][s], "H_one") < 0.50 for s in inf)
            if not (len(inf) >= need_wins(len(ss)) and low >= need_wins(len(inf))):
                failed.append(f"(d) {ctl}: informative {len(inf)}/{len(ss)}, < 0.50 in {low}/{len(inf)}")
        out["rescue"][arm] = "YES" if not failed else "NO"
        lines.append(f"R1-best rescue {arm}: " + ("YES (tier L: the helix target gives h's value)" if not failed else "NO (" + "; ".join(failed) + ")"))
    return out, lines


def bracket(runs, probs, info):
    lines, opened, trend = [], [], False
    for axis in (LR_AXIS, WD_AXIS):
        if not all(c in runs and c not in probs for c in axis):
            lines.append(f"bracket {axis}: incomplete"); continue
        for key, thr, minv in (("logp_H_one", 0.25, None), ("H_one", 0.02, 0.05)):
            means = {c: np.mean([final3(r, key) for s, r in runs[c].items() if info.get(c, {}).get(s, True)]) for c in axis}
            best = max(axis, key=lambda c: means[c])
            if minv is not None and means[best] <= minv:
                lines.append(f"bracket {key} {list(axis)}: all <= {minv} (floor; uninformative)"); continue
            if best in (axis[0], axis[-1]):
                nb = axis[1]
                common = [s for s in runs[best] if s in runs[nb]]
                d = [final3(runs[best][s], key) - final3(runs[nb][s], key) for s in common]
                op = np.mean(d) > thr and sum(v > 0 for v in d) >= need_wins(len(d))
                lines.append(f"bracket {key} {list(axis)}: means {[round(means[c], 3) for c in axis]}, best {best} at the edge, "
                             f"vs {nb} {np.mean(d):+.3f} ({sum(v > 0 for v in d)}/{len(d)}) -> {'OPEN: run ' + OUTWARD[best] if op else 'closed (edge note)'}")
                if op:
                    opened.append(OUTWARD[best])
            else:
                lines.append(f"bracket {key} {list(axis)}: means {[round(means[c], 3) for c in axis]}, best {best} inside -> closed")
    c4, c1 = "none@8e-3-4x", "none@8e-3"
    common = sorted(s for s in runs.get(c4, {}) if s in runs.get(c1, {}))
    trend_inc = c4 in runs and c1 in runs and len(common) < 3
    if c4 in runs and c1 in runs and c4 not in probs and c1 not in probs and len(common) >= 3:   # no vacuous rise on 0 seeds
        d = [final3(runs[c4][s], "logp_H_one") - final3(runs[c1][s], "logp_H_one") for s in common]
        sl = []   # the OLS rise of logp_H1 over the last quarter of the 4x run (slope x the quarter's token span)
        for s in common:
            cv = runs[c4][s]["curve"]; q = cv[-max(3, len(cv) // 4):]
            tk = [e["tokens"] / 1e9 for e in q]
            sl.append(np.polyfit(tk, [e["logp_H_one"] for e in q], 1)[0] * (tk[-1] - tk[0]))
        rise = ((np.mean(d) >= 0.5 and sum(v > 0 for v in d) >= need_wins(len(d)))
                or sum(v >= 0.25 for v in sl) >= need_wins(len(sl)))
        trend = bool(rise)
        lines.append(f"4x trend: logp_H1 final3 4x - 1x {[round(v, 3) for v in d]}, fitted rise over the last quarter (nats) "
                     f"{[round(v, 3) for v in sl]} -> {'STILL RISING' if rise else 'flat'}")
    else:
        lines.append("4x trend: incomplete")
    return {"lines": lines, "open": opened, "trend": trend, "trend_incomplete": trend_inc}


# ---------------------------------------------------------------------------------------------------------------- R3
def ansvals(s):
    return [s] if s < 1000 else [s // 10, s % 10]


def remap_cands(H, pairs):
    Hs = set(H)
    part = {h: sorted({b if a == h else a for a, b in pairs if h in (a, b)}) for h in H}
    return {h: np.array([u for u in range(100, 1000) if u not in Hs and abs(u - h) >= 100 and (u - h) % 10 != 0
                         and all(not any(v in Hs for v in ansvals(u + y)) for y in part[h])], dtype=np.int64) for h in H}


def remap_draw(H, cands, g):
    """the trainer's sampler (pretrain_range.py Rev C1): h ascending, uniform over the unused candidates"""
    used, u = set(), []
    for h in sorted(H):
        c = cands[h]
        c = c[~np.isin(c, list(used))] if used else c
        u.append(int(c[int(g.integers(len(c)))])); used.add(u[-1])
    return np.array(u, dtype=np.int64)


def s_stat(preds, hidx, y, U, nh):
    """S per decoy row of U [K, nh]: mean over h of the share of h's rows (pooled over the evals of preds [E, n]) with
    pred = u_h + y"""
    want = U[:, hidx] + y[None]                                   # [K, n]
    m = (preds[None] == want[:, None]).mean(1)                    # [K, n]: share over the evals
    cnt = np.bincount(hidx, minlength=nh)
    tot = np.stack([np.bincount(hidx, weights=row, minlength=nh) for row in m])
    return (tot / cnt[None]).mean(1)


def r3(runs, probs, r3ok, n_rand):
    lines, out = [], {}
    arm, cmp_ = "win-emb-hremap@2e-3", ("win-emb@2e-3", "none@2e-3")
    if arm not in runs or any(c not in runs for c in cmp_):
        return {"status": "incomplete"}, ["R3 INCOMPLETE"]
    if any(c in probs for c in (arm,) + cmp_):
        return {"status": "withheld"}, [f"R3 WITHHELD: invalid runs in {[c for c in (arm,) + cmp_ if c in probs]}"]
    res = {}
    for tp in ("window_end", "final3"):
        per = {}
        for s in sorted(runs[arm]):
            r = runs[arm][s]
            if not all(r3ok.get((c, s), False) for c in (arm,) + cmp_ if s in runs[c]):
                per[s] = ("withheld (pred_mismatch)", None); continue
            ho = r["held_operands"]; H = ho["H"]; pairs = ho["sets"]["H_one"]["pairs"]
            cands = remap_cands(H, pairs)
            u_true = remap_draw(H, cands, np.random.default_rng(6400 + s))
            if u_true.tolist() != ho["remap_H"]:
                per[s] = ("WITHHELD: re-drawn remap != logged remap_H", None); continue
            hpos = {h: i for i, h in enumerate(H)}
            hidx = np.array([hpos[a] if a in hpos else hpos[b] for a, b in pairs])
            y = np.array([b if a in hpos else a for a, b in pairs])
            true = np.array([a + b for a, b in pairs])

            def evals(rr):
                cv = rr["curve"]
                return [e for e in cv if e["step"] == WIN_END] if tp == "window_end" else cv[-3:]
            ev = evals(r)
            if not ev:
                per[s] = ("no eval at the window end", None); continue
            P = np.array([e["pred_H_one"] for e in ev])
            acc = float(np.mean([e["H_one"] for e in ev]))
            nh_wrong = len(set(hidx[(P != true[None]).any(0)].tolist()))
            S = float(s_stat(P, hidx, y, u_true[None], len(H))[0])
            g = np.random.default_rng(10_000_000 + 1000 * s + (0 if tp == "window_end" else 1))
            D = np.stack([remap_draw(H, cands, g) for _ in range(n_rand)])
            Sd = np.concatenate([s_stat(P, hidx, y, D[i:i + 500], len(H)) for i in range(0, n_rand, 500)])
            p = (1 + int((Sd >= S - 1e-12).sum())) / (1 + n_rand)
            Sc = {c: float(s_stat(np.array([e["pred_H_one"] for e in evals(runs[c][s])]), hidx, y, u_true[None], len(H))[0])
                  for c in cmp_ if s in runs[c] and evals(runs[c][s])}
            wrong = P != true[None]
            ud = ((P % 10) == ((u_true[hidx] + y) % 10)[None]) & wrong & (P >= 0)
            testable = acc < 0.90 and nh_wrong >= 10
            per[s] = (("ok" if testable else "not testable"),
                      dict(S=S, p=p, Sc=Sc, acc=acc, nh_wrong=nh_wrong, decoy_mean=float(Sd.mean()),
                           units=float(ud.sum() / max(1, wrong.sum()))))
        ok = {s: v[1] for s, v in per.items() if v[0] == "ok"}
        n = len(per)
        sup = sum(d["p"] < 0.01 and all(d["S"] > d["Sc"].get(c, -1) for c in cmp_) for d in ok.values())
        agn = sum(d["p"] >= 0.05 for d in ok.values())
        v = "supports" if sup >= need_wins(n) else ("against" if agn >= need_wins(n) else "inconclusive")
        res[tp] = v
        for s, (st, d) in per.items():
            if d is None:
                lines.append(f"R3 {tp} s{s}: {st}"); continue
            lines.append(f"R3 {tp} s{s}: S {d['S']:.3f} (decoys {d['decoy_mean']:.3f}), p {d['p']:.4f}; same u on win-emb "
                         f"{d['Sc'].get('win-emb@2e-3', float('nan')):.3f}, none {d['Sc'].get('none@2e-3', float('nan')):.3f}; "
                         f"acc_H1 {d['acc']:.3f}, h with a wrong row {d['nh_wrong']}; units digit = u+y among wrong {d['units']:.3f}"
                         f"{'' if st == 'ok' else ' -> NOT TESTABLE'}")
        lines.append(f"R3 {tp}: {v} ({sup} seeds p < .01 and above both controls, {agn} seeds p >= .05, of {n})")
    out["window_end"], out["final3"] = res["window_end"], res["final3"]
    out["verdict"] = ("supports (persistent)" if res["final3"] == "supports" else
                      "supports while the prior is on" if res["window_end"] == "supports" else
                      "against" if res["window_end"] == "against" and res["final3"] == "against" else "inconclusive")
    lines.append(f"R3: {out['verdict']}")
    # R3b / R3c geometry specificity
    for x, ctl, name in (("win-emb@2e-3", arm, "R3b"), ("win-text@2e-3", "win-text-farshuf@2e-3", "R3c")):
        if x not in runs or ctl not in runs or x in probs or ctl in probs:
            out[name] = "incomplete"; lines.append(f"{name}: incomplete"); continue
        common = sorted(s for s in runs[x] if s in runs[ctl])
        d = [final3(runs[x][s], "H_one") - final3(runs[ctl][s], "H_one") for s in common]
        # R3b: the control (hremap) must stay < 0.5 in >= 3/4 seeds; R3c: farshuf >= 0.5 in >= 3/4 seeds -> not specific
        n_lo = sum(final3(runs[ctl][s], "H_one") < 0.5 for s in common)
        ctl_hi = (n_lo < need_wins(len(common))) if name == "R3b" else (len(common) - n_lo >= need_wins(len(common)))
        spec = np.mean(d) >= 0.20 and sum(v > 0 for v in d) >= need_wins(len(d))
        word = {"R3b": ("h's own helix value", "a prior on the number embeddings, not specific to h's value"),
                "R3c": ("helix geometry", "a number prior (not geometry-specific)")}[name]
        out[name] = word[0] if spec and not ctl_hi else word[1]
        lines.append(f"{name} {x} - {ctl}: {[round(v, 3) for v in d]} (mean {np.mean(d):+.3f}){'; control >= 0.5' if ctl_hi else ''} -> {out[name]}")
    fs = "win-text-farshuf@2e-3"
    if fs in runs and fs not in probs:
        for s, r in sorted(runs[fs].items()):
            ho = r["held_operands"]; pairs = ho["sets"]["H_one"]["pairs"]; pf = dict(zip(ho["H"], ho["perm_far_H"]))
            P = np.array([e["pred_H_one"] for e in r["curve"][-3:]])
            tgt = np.array([pf[a] + b if a in pf else pf[b] + a for a, b in pairs])
            true = np.array([a + b for a, b in pairs]); wrong = P != true[None]
            lines.append(f"R3 descriptive {fs} s{s}: share of wrong final3 answers = pi_far(h) + y: "
                         f"{float(((P == tgt[None]) & wrong).sum() / max(1, wrong.sum())):.3f} (not a mechanism test)")
    return out, lines


# ---------------------------------------------------------------------------------------------------------------- R4/R5
def per_h_acc(r):
    ho = r["held_operands"]; H = ho["H"]; pairs = ho["sets"]["H_one"]["pairs"]; hpos = {h: i for i, h in enumerate(H)}
    hidx = np.array([hpos[a] if a in hpos else hpos[b] for a, b in pairs])
    hits = np.array([[c == "1" for c in e["hits_H_one"]] for e in r["curve"][-3:]]).mean(0)
    acc = np.bincount(hidx, weights=hits, minlength=len(H)) / np.bincount(hidx, minlength=len(H))
    s4 = np.bincount(hidx, weights=np.array([a + b >= 1000 for a, b in pairs], float), minlength=len(H)) / np.bincount(hidx, minlength=len(H))
    cnt = np.array([ho["count_text_site"][h] for h in H], float)
    return np.array(H), acc, s4, cnt


def partial_spearman(v, x, covs):
    """Pearson of the rank residuals of v and x after OLS on the ranks of covs (+ intercept); nan if v is constant"""
    rv, rx = stats.rankdata(v), stats.rankdata(x)
    if np.ptp(rv) == 0 or np.ptp(rx) == 0:
        return float("nan")
    Z = np.column_stack([np.ones(len(v))] + [stats.rankdata(c) for c in covs])
    res = lambda t: t - Z @ np.linalg.lstsq(Z, t, rcond=None)[0]
    a, b = res(rv), res(rx)
    return float(a @ b / math.sqrt((a @ a) * (b @ b))) if (a @ a) > 0 and (b @ b) > 0 else float("nan")


def r4_r5(runs, probs):
    lines = []
    for c in sorted(runs):
        if c in probs:
            continue
        ss = sorted(runs[c])
        q = [np.mean([final3(runs[c][s], f"H_one_q{k}") for s in ss]) for k in range(1, 6)]
        rho, band = [], {}
        pooled = []
        for s in ss:
            H, acc, s4, cnt = per_h_acc(runs[c][s])
            rho.append(partial_spearman(acc, np.log10(cnt), [H, s4]))
            pooled += list(zip(H, acc, np.log10(cnt)))
        P = np.array(pooled)
        for lo, hi in ((100, 400), (400, 700), (700, 1000)):
            m = (P[:, 0] >= lo) & (P[:, 0] < hi)
            if m.sum() >= 4:
                med = np.median(P[m, 2]); hi_ = m & (P[:, 2] > med); lo_ = m & (P[:, 2] <= med)
                band[f"{lo}-{hi - 1}"] = round(float(P[hi_, 1].mean() - P[lo_, 1].mean()), 3) if hi_.any() and lo_.any() else None
        lines.append(f"R4 {c}: acc_H1 final3 q1..q5 {[round(v, 3) for v in q]}; partial rho(acc, log count | value, 4-digit share) "
                     f"per seed {[round(v, 2) for v in rho]}; high - low count inside value bands {band}")
    a, b = "win-text@2e-3", "win-emb@2e-3"
    if a in runs and b in runs and a not in probs and b not in probs:
        rr = []
        for s in sorted(set(runs[a]) & set(runs[b])):
            H, acc_a, s4, cnt = per_h_acc(runs[a][s]); _, acc_b, _, _ = per_h_acc(runs[b][s])
            rr.append(partial_spearman(acc_a - acc_b, np.log10(cnt), [H, s4]))
        lines.append(f"R4 contrast {a} - {b}: partial rho per seed {[round(v, 2) for v in rr]} (prediction > 0)")
        ex = []
        for s, r in sorted(runs[a].items()):
            H, acc, _, _ = per_h_acc(r)
            e_ = np.log10(1 + np.array([r["held_exposure"]["st_on"][h] for h in H]))
            ex.append(stats.spearmanr(acc, e_)[0] if np.ptp(acc) > 0 and np.ptp(e_) > 0 else float("nan"))
        lines.append(f"R4 {a} vs prior-on exposure st_on: rho per seed {[round(float(v), 2) for v in ex]}")
    ref = runs.get("none@2e-3", {})
    for c in sorted(runs):
        if c in probs:
            continue
        ss = sorted(runs[c])
        f = lambda k: np.mean([final3(runs[c][s], k) for s in ss])
        dbpb = [final3(runs[c][s], "val_bits_per_byte") - final3(ref[s], "val_bits_per_byte") for s in ss if s in ref]
        we = [e["H_one"] for s in ss for e in runs[c][s]["curve"] if e["step"] == WIN_END]
        lines.append(f"R5 {c}: H_both {f('H_both'):.3f}, H_ans single {f('H_ans_single'):.3f} chunk {f('H_ans_chunk'):.3f}, "
                     f"held6 {np.mean([held6_final3(runs[c][s]) for s in ss]):.3f}, logp_H1 {f('logp_H_one'):.2f}, "
                     f"bpb - none@2e-3 {np.mean(dbpb) if dbpb else float('nan'):+.4f}; acc_H1 at step {WIN_END} "
                     f"{np.mean(we) if we else float('nan'):.3f}")
    return lines


def summarise(root, code_sha, n_rand):
    runs, probs, r3ok = load(root, code_sha)
    lines = [f"cells: " + ", ".join(f"{c} ({len(v)} seeds)" for c, v in sorted(runs.items()))]
    for c, b in sorted(probs.items()):
        lines.append(f"INVALID {c}: {b[:4]}{' ...' if len(b) > 4 else ''}")
    o1, l1 = r1(runs, probs)
    ob, lb = r1_best(runs, probs, o1)
    o3, l3 = r3(runs, probs, r3ok, n_rand)
    return {"R1": o1, "R1best": ob, "R3": o3, "problems": probs}, lines + l1 + lb + l3 + r4_r5(runs, probs)


# ---------------------------------------------------------------------------------------------------------------- selftest
def fake_run(tpl, cell, seed, rng, acc, logp, held=0.97, hremap_share=0.0, sha="fake", data_sha_tag="", diverge=False, bpb=1.22):
    """a fake run JSON on the template's H / pairs (one template for every seed; remap_H re-drawn for the seed)"""
    arm, rest = cell.split("@")
    lr_s = rest.split("-wd")[0].split("-4x")[0]
    wd = float(rest.split("-wd")[1].split("-")[0]) if "-wd" in rest else 0.1
    tokens = 1.2e9 if rest.endswith("-4x") else 3e8
    tgt, site = {"none": ("none", "ctx"), "win-ctx": ("helix3", "ctx"), "win-text": ("helix3", "text"),
                 "win-emb": ("helix3", "emb"), "win-emb-hremap": ("helix3_hremap", "emb"),
                 "win-text-farshuf": ("helix3_farshuf", "text")}[arm]
    a = dict(BASE, target=tgt, site=site, lr=float(lr_s), tokens=tokens, seed=seed,
             schedule="whole" if tgt == "none" else "window", window="0.05,0.4")
    if wd != 0.1:
        a["wd"] = wd
    ho = json.loads(json.dumps(tpl["held_operands"]))
    H, pairs = ho["H"], ho["sets"]["H_one"]["pairs"]
    ho["remap_H"] = remap_draw(H, remap_cands(H, pairs), np.random.default_rng(6400 + seed)).tolist()
    u = dict(zip(H, ho["remap_H"]))
    steps = STEPS[tokens]
    evs = list(range(0, steps, 250)) + [steps - 1]
    curve = []
    for k, st in enumerate(evs):
        fr = min(1.0, st / (0.4 * steps))
        acc_k = acc * fr
        hits = rng.random(len(pairs)) < acc_k
        preds = []
        for (x, y_), hit in zip(pairs, hits):
            h, y = (x, y_) if x in u else (y_, x)
            if hit:
                preds.append(x + y_)
            elif rng.random() < hremap_share:
                preds.append(u[h] + y)
            else:
                preds.append(int(rng.integers(0, 1999)))
        e = dict(step=st, tokens=st * 16384, lm=11.5 if st == 0 else 3.0, val_loss=3.2 if not (diverge and k > 10) else float("nan"),
                 val_bits_per_byte=bpb, geo=None, grad_prior=None, H_one=float(hits.mean()), hits_H_one="".join("1" if v else "0" for v in hits),
                 pred_H_one=preds, logp_H_one=logp * (0.5 + 0.5 * fr) + float(rng.normal(0, 0.05)), H_both=0.0, H_ans_single=0.0, H_ans_chunk=0.0,
                 held_lines_scanned=LINES_PER_STEP * (st + 1), held_rej=10 + st, held_data_sha=f"sha{seed}{data_sha_tag}{st}",
                 prior_used_frac=0.0 if tgt == "none" else 0.35, prior_on_frac=6408 / 18310 if tgt != "none" else 0.0,
                 logp_dp_maxdiff_H_one=0.0, logp_dp_maxdiff_H_both=0.0, logp_dp_maxdiff_H_ans=0.0,
                 pred_mismatch_H_one=0, pred_mismatch_H_both=0, pred_mismatch_H_ans=0,
                 prior_on=int(tgt != "none" and 916 <= st < 7324))   # as the trainer logs them (AUDIT R05)
        for kq in range(1, 6):
            e[f"H_one_q{kq}"] = acc_k
        for b in BUCK:
            e[f"held_{b}"] = held * fr if st > 0 else 0.0
        if site == "emb" and tgt != "none" and e["prior_on"]:   # the trainer logs it only while the prior is on
            e["grad_prior_emb_rows"] = 1000
        curve.append(e)
    ex = {"st": tpl["held_exposure"]["st"], "st_on": tpl["held_exposure"]["st_on"], "raw": tpl["held_exposure"]["raw"]}
    return dict(args=a, steps=steps, curve=curve, code_sha=sha, held_operands=ho, held_exposure=ex,
                prior_events=[] if tgt == "none" else [[916, 1], [7324, 0]])


def selftest(tpl_path, n_rand):
    tpl = json.loads(Path(tpl_path).read_text())
    ho = tpl["held_operands"]
    # double path against the trainer: the re-drawn remap of the template's own seed = the logged one
    s0 = tpl["args"]["seed"]
    redraw = remap_draw(ho["H"], remap_cands(ho["H"], ho["sets"]["H_one"]["pairs"]), np.random.default_rng(6400 + s0))
    assert redraw.tolist() == ho["remap_H"], "re-drawn remap != the trainer's remap_H"
    print(f"selftest: remap re-draw = the trainer's remap_H of seed {s0} ({len(redraw)} h)")
    MAINC = ("none@2e-3", "none@8e-3", "win-ctx@2e-3", "win-text@2e-3", "win-emb@2e-3", "win-emb-hremap@2e-3",
             "win-text-farshuf@2e-3")

    PLRC = tuple(f"{a}@{lr}" for a in ("win-text", "win-emb") for lr in ("1e-3", "4e-3"))
    CONFC = X_CELLS + Y_CELLS   # the cells with CONF seeds 4-7 in the real design

    def scenario(name, spec, check, lc=True, plr=False, conf=False, drop=None, **kw):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(1)
            for c in MAINC + (LC_CELLS if lc else ()) + (PLRC if plr else ()):
                acc, logp, extra = spec.get(c, (0.02, -9.0, {}))
                for s in (range(8) if conf and c in CONFC else range(4)):
                    if s in (drop or {}).get(c, ()):
                        continue
                    ex = dict(extra)
                    if callable(ex.get("held")):
                        ex["held"] = ex["held"](s)
                    if callable(acc):
                        a_ = acc(s)
                    else:
                        a_ = acc
                    r = fake_run(tpl, c, s, rng, a_, logp, **ex)
                    od = Path(d) / c.replace("@", "_")
                    od.mkdir(exist_ok=True)
                    (od / f"run_{c}_seed{s}.json").write_text(json.dumps(r))
            o, lines = summarise(d, "fake", n_rand)
        ok = check(o)
        print(f"selftest {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("\n".join(lines)); raise SystemExit(1)
        return lines

    # S1 the emb prior works, with the remap mechanism; text partial; controls and LC at the floor
    L = scenario("S1 prior works", {"win-emb@2e-3": (0.75, -1.0, {}), "win-text@2e-3": (0.30, -5.0, {}),
                                    "win-emb-hremap@2e-3": (0.10, -6.0, {"hremap_share": 0.6})},
                 lambda o: o["R1"]["win-emb@2e-3"].startswith("HOLDS") and o["R1"]["win-text@2e-3"].startswith("partial")
                 and o["R3"]["verdict"] == "supports (persistent)" and o["R3"]["R3b"] == "h's own helix value")
    print("\n".join("    " + l for l in L if l.startswith(("R1 ", "R3:", "R3b"))))
    # S2 nothing works: R1 fails, R3 against
    scenario("S2 nothing", {"win-emb-hremap@2e-3": (0.02, -9.0, {"hremap_share": 0.0})},
             lambda o: o["R1"]["win-emb@2e-3"] == "fails" and o["R1"]["win-text@2e-3"] == "fails" and o["R3"]["verdict"] == "against")
    # S3 a control that never learned seen addition in 2 seeds: not established (no vacuous pass)
    scenario("S3 uninformative control", {"win-emb@2e-3": (0.75, -1.0, {}),
                                          "none@8e-3": (0.0, -9.0, {"held": lambda s: 0.5 if s < 2 else 0.97})},
             lambda o: o["R1"]["win-emb@2e-3"].startswith("NOT ESTABLISHED") and o["R1"]["yb"]["none@8e-3"] == "uninformative")
    # S4 LC: one seed of none@1.6e-2 at 0.3 -> qualified HOLDS; two seeds -> within this budget
    scenario("S4a LC one seed", {"win-emb@2e-3": (0.75, -1.0, {}), "none@1.6e-2": (lambda s: 0.3 if s == 0 else 0.0, -9.0, {})},
             lambda o: o["R1"]["win-emb@2e-3"].startswith("HOLDS (none@1.6e-2: one seed"))
    scenario("S4b LC fails", {"win-emb@2e-3": (0.75, -1.0, {}), "none@1.6e-2": (lambda s: 0.3 if s < 2 else 0.02, -9.0, {})},
             lambda o: o["R1"]["win-emb@2e-3"].startswith("the prior makes H usable within this budget"))
    # S5 the bracket is open on the continuous score (none@1.6e-2 best by 1 nat) -> PROVISIONAL; LC missing -> PROVISIONAL
    scenario("S5 bracket open", {"win-emb@2e-3": (0.75, -1.0, {}), "none@1.6e-2": (0.02, -8.0, {})},
             lambda o: "bracket OPEN" in o["R1"]["win-emb@2e-3"] and o["R1"]["bracket_open"] == ["lr 3.2e-2"])
    scenario("S5b LC missing", {"win-emb@2e-3": (0.75, -1.0, {})}, lambda o: "PROVISIONAL" in o["R1"]["win-emb@2e-3"], lc=False)
    # S6 a broken data stream in one arm -> WITHHELD
    scenario("S6 data sha", {"win-emb@2e-3": (0.75, -1.0, {"data_sha_tag": "x"})},
             lambda o: o["R1"]["status"] == "withheld" and any("held_data_sha" in b for b in o["problems"]["win-emb@2e-3"]))
    # S7 a 4x none whose H likelihood still rises -> 'within 4x'
    scenario("S7 4x trend", {"win-emb@2e-3": (0.75, -1.0, {}), "none@8e-3-4x": (0.02, -7.5, {})},
             lambda o: o["R1"]["trend"] and "within 4x" in o["R1"]["win-emb@2e-3"])
    # S8 R2: win-ctx uses H (0.2) but X beats it by >= 0.30 -> HOLDS with the R2 note; S9 by < 0.30 -> no wall claim
    scenario("S8 R2 margin met", {"win-emb@2e-3": (0.75, -1.0, {}), "win-ctx@2e-3": (0.20, -6.0, {})},
             lambda o: o["R1"]["win-emb@2e-3"].startswith("HOLDS") and "R2: win-ctx also uses H" in o["R1"]["win-emb@2e-3"])
    scenario("S9 R2 margin missed", {"win-emb@2e-3": (0.75, -1.0, {}), "win-ctx@2e-3": (0.60, -2.0, {})},
             lambda o: o["R1"]["win-emb@2e-3"].startswith("no wall claim"))
    # R1-best (amendment PLR). B0: no PLR cells -> R1-best INCOMPLETE, R1 as in S1
    scenario("B0 PLR not landed", {"win-emb@2e-3": (0.75, -1.0, {}), "win-text@2e-3": (0.30, -5.0, {})},
             lambda o: o["R1best"]["status"] == "incomplete" and o["R1"]["win-emb@2e-3"].startswith("HOLDS")
             and o["R1"]["win-text@2e-3"].startswith("partial"))
    # B1: emb best at the top edge 4e-3 (OPEN), text best at 2e-3 (closed); the selection-seed R1 uses those cells;
    # CONF only for the 2e-3 pick; R1 itself unchanged (emb at 2e-3 = 0.30 -> partial)
    L = scenario("B1 emb best at 4e-3", {"win-emb@1e-3": (0.20, -6.0, {}), "win-emb@2e-3": (0.30, -5.0, {}), "win-emb@4e-3": (0.80, -1.0, {}),
                                        "win-text@1e-3": (0.10, -7.0, {}), "win-text@2e-3": (0.35, -5.0, {}), "win-text@4e-3": (0.20, -6.0, {})},
                 lambda o: o["R1best"]["best"] == {"win-text": "win-text@2e-3", "win-emb": "win-emb@4e-3"}
                 and o["R1best"]["bracket"]["win-emb"].startswith("OPEN (top") and o["R1best"]["bracket"]["win-text"] == "closed"
                 and o["R1best"]["sel"]["win-emb@4e-3"].startswith("HOLDS") and o["R1best"]["sel"]["win-text@2e-3"].startswith("partial")
                 and list(o["R1best"]["conf"]) == ["win-text@2e-3"] and o["R1best"]["conf"]["win-text@2e-3"].startswith("partial")
                 and o["R1"]["win-emb@2e-3"].startswith("partial"), plr=True, conf=True)
    print("\n".join("    " + l for l in L if l.startswith("R1-best") and ("best" in l.split(":")[0] or "CONF seeds not run" in l)))
    tr = [l for l in L if l.startswith("R1-best [seeds 4-7, CONF] 4x trend")]
    assert len(tr) == 1 and "[]" not in tr[0] and tr[0].endswith("flat"), tr   # the CONF trend reads the R1 seeds, not 0 seeds
    print("selftest B1b CONF trend not vacuous: PASS")
    # B2: areas within 0.02 -> the lower offline bpb wins (2e-3 has the lower area but the lower bpb)
    scenario("B2 tie -> lower bpb", {"win-emb@1e-3": (0.20, -6.0, {}), "win-emb@2e-3": (0.70, -1.0, {"bpb": 1.20}),
                                     "win-emb@4e-3": (0.715, -1.0, {"bpb": 1.30})},
             lambda o: o["R1best"]["best"]["win-emb"] == "win-emb@2e-3" and o["R1best"]["bracket"]["win-emb"] == "closed", plr=True)
    # B3: an X at a non-2e-3 lr never reaches the CONF reading (no vacuous pass on zero seeds)
    scenario("B3 no vacuous CONF", {"win-emb@4e-3": (0.80, -1.0, {}), "win-text@1e-3": (0.80, -1.0, {})},
             lambda o: "conf" not in o["R1best"] and o["R1best"]["bracket"]["win-text"].startswith("OPEN (bottom"), plr=True, conf=True)
    # B2b: the tie goes to the lower bpb even at the higher lr (B2 cannot tell bpb from 'the lower lr')
    scenario("B2b tie -> lower bpb at the higher lr", {"win-emb@2e-3": (0.715, -1.0, {"bpb": 1.30}), "win-emb@4e-3": (0.70, -1.0, {"bpb": 1.20})},
             lambda o: o["R1best"]["best"]["win-emb"] == "win-emb@4e-3", plr=True)
    # B5: a non-finite bpb never wins a tie
    scenario("B5 nan bpb loses", {"win-emb@2e-3": (0.70, -1.0, {"bpb": 1.30}), "win-emb@4e-3": (0.715, -1.0, {"bpb": float("nan")})},
             lambda o: o["R1best"]["best"]["win-emb"] == "win-emb@2e-3", plr=True)
    # B4: CONF reads seeds 4-7 (X good on 0-3, poor on 4-7 -> selection HOLDS, CONF partial)
    scenario("B4 CONF reads seeds 4-7", {"win-text@2e-3": (lambda s: 0.80 if s < 4 else 0.30, -1.0, {})},
             lambda o: o["R1best"]["sel"]["win-text@2e-3"].startswith("HOLDS") and o["R1best"]["conf"]["win-text@2e-3"].startswith("partial")
             and o["R1best"]["rescue"]["win-text"] == "NO", plr=True, conf=True)
    # B6: a 4x cell with 2 seeds -> PROVISIONAL, never a plain HOLDS
    scenario("B6 4x on 2 seeds", {"win-emb@2e-3": (0.75, -1.0, {})},
             lambda o: "PROVISIONAL" in o["R1"]["win-emb@2e-3"] and "4x trend on < 3 seeds" in o["R1"]["win-emb@2e-3"],
             drop={"none@8e-3-4x": (2, 3)})
    # B7: the rescue conjunction: emb holds everywhere with an informative, failing hremap -> YES; text partial -> NO
    scenario("B7 rescue conjunction", {"win-emb@2e-3": (0.75, -1.0, {}), "win-text@2e-3": (0.30, -5.0, {}),
                                       "win-emb-hremap@2e-3": (0.10, -6.0, {"hremap_share": 0.6})},
             lambda o: o["R1best"]["rescue"] == {"win-text": "NO", "win-emb": "YES"}, plr=True, conf=True)
    print("selftest: all scenarios pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/yh6210/llmadd_torch/runs/ptheld/out")
    ap.add_argument("--code_sha", default=None, help="expected code_sha (default: sha256[:12] of runs/ptheld/code/tools/pretrain_range.py)")
    ap.add_argument("--n_rand", type=int, default=10000)
    ap.add_argument("--selftest", default=None, metavar="RUN_JSON", help="a Rev C1 CPU run JSON as the template of the fakes")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.selftest, min(args.n_rand, 2000))
    else:
        import hashlib
        sha = args.code_sha or hashlib.sha256(Path("/scratch/yh6210/llmadd_torch/runs/ptheld/code/tools/pretrain_range.py").read_bytes()).hexdigest()[:12]
        o, lines = summarise(args.root, sha, args.n_rand)
        print("\n".join(lines))
