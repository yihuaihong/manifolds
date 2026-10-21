"""Pre-registered read-out of the Torch runs/ptheld2 (wall 1 follow-up: H = 54 three-digit numbers, 6 per hundreds block,
3 rare + 3 common by text-site count inside the block; prior arms on from 5% to the end of training at lambda 1). Intent
spec and rules: llmadd_torch/runs/ptheld2/AUDIT.md (written 2026-10-21 before any run; co-primary layers fixed by the user
the same day, before any run). Written and tested on FAKE results only (--selftest) before any result existed.

Cells (keyed on r["args"]): none@2e-3, none@8e-3 (seeds 0-7), none@8e-3-4x (seeds 0-3), win-emb-always@2e-3,
win-emb-decay@2e-3, win-text-always@2e-3, win-ctx-always@2e-3 (seeds 0-7). final3 = the mean of the last three evals
(1x: 17750 / 18000 / 18309; 4x: 72750 / 73000 / 73241). acc_L = H_one_<L> (exact match on the rows of layer L, the same
rows / forward as H_one). bpb = val_bits_per_byte final3. held6 = mean of the six filtered held_bXY.
Co-primary layers (user, 2026-10-21): A = hi (blocks 7-9, 18 h), B = lo_rare (the rare halves of all blocks, 27 h). Every
rule runs per layer; the verdicts are printed separately, never merged.
  informative seed (as ptheld): held6 final3 >= 0.80, not diverged, not unstable (final val_loss <= the same seed's
      none@2e-3 + 0.5 nats). A prior-arm seed failing the held6 / divergence gate counts as a failure of (b) and keeps its
      measured value in the paired tests.
  Selection (per layer, seeds 0-3 only): the prior arm (win-emb-always / win-emb-decay / win-text-always) with the highest
      mean acc_L final3; arms within 0.02 of the top -> the lower mean bpb. Confirmation on seeds 4-7 of that arm only.
  Rescue reading (per layer, confirm seeds; report/audits/rescue_standard.md, tier L):
      (a) every none config (none@2e-3, none@8e-3 on seeds 4-7; none@8e-3-4x on its seeds 0-3): >= ceil(3/4 n) seeds
          informative AND acc_L final3 <= 0.10 ('uninformative' if fewer than ceil(3/4 n) informative seeds);
      (b) the chosen arm: acc_L final3 >= 0.50 (and the held6 / divergence gate) in >= 3/4 of seeds 4-7;
      (c) chosen minus each of none@2e-3, none@8e-3, win-ctx-always@2e-3 over the seeds 4-7 where the control is
          informative (>= 3): wins >= ceil(3/4 m), one-sided paired t; p_L = max of the three (intersection-union);
          specificity: chosen - win-ctx-always paired mean >= +0.30;
      Holm over the two layers: the smaller p_L < .0125, the other < .025 (and only if the first passed).
      RESCUE (tier L) iff (a), (b), (c) with its Holm threshold, and specificity. Printed with the rescue_standard
      criteria this batch does not run: (a) none's lr x wd bracket, (b) the prior arm's own lr sweep, (d) a shuffled target.
  Beside it (no claim): held6 of the chosen arm, its bpb minus none@2e-3 (paired), the median grad_lm_share / clip_coef
  over the on-evals per prior-arm seed (FLAG when < 0.5: the prior gradient exceeds the LM's).
  Descriptive: every arm's full-H acc_H1 and every layer (v1..v9, rare, common, hi, lo_rare) final3 over seeds 0-3 / 4-7 /
  0-7; acc at step 7250 (ptheld's window end) vs the final eval vs final3 (the window 0.05,1.0 ends with the run); logp per
  layer; the arm a selection on seeds 4-7 alone would pick (shown, never used).
Validity (a failure withholds every verdict that uses the cell): ptheld_summary.check_run's checks with amendment DP,
adapted to ptheld2 (code_sha, base args with held_operands 54 / held_strat value_freq / lambda 1 / window 0.05,1.0,
steps, evals, held_lines_scanned, held_rej, emb rows, prior_on_frac 17394 / 18310 and prior_events [[916, 1]], none never
on), plus the Rev D logs (layers and their logp at every eval; lam_now = the formula; grad_total / clip_coef /
grad_lm_share at every eval; H = 54, 3 + 3 per block, block = h // 100, layer_rows hi 216 / lo_rare 324); per seed across
cells: H, counts, H_one pairs equal, step-0 lm within 1e-3, held_data_sha equal at every common eval step.
    python tools/ptheld2_summary.py [--root /scratch/yh6210/llmadd_torch/runs/ptheld2/out] [--code_sha <sha>]
    python tools/ptheld2_summary.py --selftest <a Rev D value_freq CPU run JSON>
"""
import argparse
import glob
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_rules as RR   # noqa: E402
from ptheld_summary import BUCK, final3, held6, held6_final3, lrs, n_evals, need_wins, ptest   # noqa: E402

BASE = dict(arith_lines=4, held_operands=54, held_strat="value_freq", bs=32, seq=512, d_model=384, n_layers=6, n_heads=6,
            d_ff=1536, tokenization="whole", n_eval=150, eval_every=250, geo_layers=None, untie=False, text_min=16, wd=0.1)
STEPS = {3e8: 18310, 1.2e9: 73242}
S_ON, N_ON = 916, 18310 - 916     # window 0.05,1.0 at 18,310 steps: on from step 916 to the end
WIN_END_PTHELD = 7250             # ptheld's window end (descriptive comparison only)
LINES_PER_STEP = 32 * 4
PRIOR_X = ("win-emb-always@2e-3", "win-emb-decay@2e-3", "win-text-always@2e-3")
CTRL = "win-ctx-always@2e-3"
NONES = ("none@2e-3", "none@8e-3", "none@8e-3-4x")
Y_PAIRED = ("none@2e-3", "none@8e-3", CTRL)
MAIN_CELLS = NONES + PRIOR_X + (CTRL,)
# amendment SUPP (2026-10-21, before any result; rendered by make_jobs_ptheld2.py --supp, the user's call): rescue_standard
# (a) none's lr x wd bracket on this H (seeds 0-3) and (d) the remapped-target control of the emb arms (seeds 0-7)
LC = ("none@1.6e-2", "none@8e-3-wd0.01", "none@8e-3-wd1")
HREMAP = "win-emb-hremap-always@2e-3"
LR_AXIS, WD_AXIS = ("none@2e-3", "none@8e-3", "none@1.6e-2"), ("none@8e-3-wd0.01", "none@8e-3", "none@8e-3-wd1")
WD_OF = {"none@8e-3-wd0.01": 0.01, "none@8e-3-wd1": 1.0}
ALL_CELLS = MAIN_CELLS + LC + (HREMAP,)
LAYERS = (("A", "hi"), ("B", "lo_rare"))
ALL_LAYERS = tuple(f"v{j}" for j in range(1, 10)) + ("rare", "common", "hi", "lo_rare")
SEL, CONF = (0, 1, 2, 3), (4, 5, 6, 7)
SEEDS = {c: (SEL if c == "none@8e-3-4x" or c in LC else SEL + CONF) for c in ALL_CELLS}
HOLM = (0.0125, 0.025)
FLOOR, BAR, SPEC, TIE = 0.10, 0.50, 0.30, 0.02
BR_LOGP, BR_ACC, BR_ACC_MIN, D_MARGIN = 0.25, 0.02, 0.05, 0.20   # bracket margins (as ptheld R1); (d) margin (as ptheld R3b)


def cell_of(a):
    """cell name of a run's args, or None (not a ptheld2 cell)"""
    if a["target"] == "none":
        wd = a.get("wd", 0.1)
        return f"none@{lrs(a['lr'])}" + (f"-wd{wd:g}" if wd != 0.1 else "") + ("-4x" if a["tokens"] == 1.2e9 else "")
    if a["target"] == "helix3_hremap" and a.get("site") == "emb" and not a.get("lam_decay", False):
        return f"win-emb-hremap-always@{lrs(a['lr'])}" + ("-4x" if a["tokens"] == 1.2e9 else "")
    if a["target"] != "helix3":
        return None
    arm = {("emb", False): "win-emb-always", ("emb", True): "win-emb-decay", ("text", False): "win-text-always",
           ("ctx", False): "win-ctx-always"}.get((a.get("site", "ctx"), bool(a.get("lam_decay", False))))
    return None if arm is None else f"{arm}@{lrs(a['lr'])}" + ("-4x" if a["tokens"] == 1.2e9 else "")


def lam_want(c, step):
    """the lambda an eval's step must have consumed (prior on)"""
    if c.startswith("win-emb-decay"):
        return 1.0 * (1.0 - (step - S_ON) / N_ON)
    return 1.0


def check_run(r, code_sha, c):
    """reasons why a run is not valid (empty = valid)"""
    a, bad = r["args"], []
    if code_sha and r.get("code_sha") != code_sha:
        bad.append(f"code_sha {r.get('code_sha')} != {code_sha}")
    for k, v in BASE.items():
        v = WD_OF.get(c, v) if k == "wd" else v   # the SUPP wd cells
        if a.get(k, v if k not in ("held_strat", "held_operands") else None) != v:
            bad.append(f"arg {k}={a.get(k)} != {v}")
    if c not in ALL_CELLS:
        bad.append(f"cell {c} is not a ptheld2 cell")
    if a["tokens"] not in STEPS or r["steps"] != STEPS[a["tokens"]]:
        bad.append(f"steps {r['steps']} for tokens {a['tokens']}")
    elif len(r["curve"]) != n_evals(r["steps"]) or not RR.finished(r):
        bad.append(f"evals {len(r['curve'])} != {n_evals(r['steps'])} or unfinished")
    prior = a["target"] != "none"
    if prior and ((a.get("schedule"), a.get("window")) != ("window", "0.05,1.0") or a.get("lam") != 1.0):
        bad.append(f"schedule {a.get('schedule')} {a.get('window')} lam {a.get('lam')}")
    ho = r.get("held_operands", {})
    H = ho.get("H", [])
    blk_ok = (len(H) == 54 and ho.get("block") == [h // 100 for h in H] and len(ho.get("half", [])) == 54 and
              all(sum(1 for b, hf in zip(ho["block"], ho["half"]) if (b, hf) == (j, x)) == 3
                  for j in range(1, 10) for x in ("rare", "common")))
    if not blk_ok or (ho.get("layer_rows") or {}).get("hi") != 216 or (ho.get("layer_rows") or {}).get("lo_rare") != 324:
        bad.append("H construction (54 h, 3 + 3 per block, block = h // 100, layer rows hi 216 / lo_rare 324)")
    dp_tol = {"H_one": 1.0, "H_both": 1.5, "H_ans": 1.5}
    mks = {L: np.array([c_ == "1" for c_ in m]) for L, m in ((ho.get("sets") or {}).get("H_one") or {}).get("masks", {}).items()}
    for e in r["curve"][-3:]:   # double path of the layers read: the logged H_one_<L> = the per-row hits on the layer's mask
        hits = np.array([c_ == "1" for c_ in e.get("hits_H_one", "")])
        for L in ("hi", "lo_rare"):
            if L not in mks or len(hits) != len(mks[L]) or abs(float(hits[mks[L]].mean()) - e.get(f"H_one_{L}", -1)) > 1e-9:
                bad.append(f"H_one_{L} at step {e['step']} != its rows' hits"); break
    for e in r["curve"]:
        st = e["step"]
        if e.get("held_lines_scanned") != LINES_PER_STEP * (st + 1):
            bad.append(f"held_lines_scanned {e.get('held_lines_scanned')} at step {st}"); break
        if a.get("site") == "emb" and prior and e.get("grad_prior_emb_rows") not in (None, 1000):
            bad.append(f"grad_prior_emb_rows {e.get('grad_prior_emb_rows')} at step {st}"); break
        # amendment DP (ptheld, 2026-10-20): step 0 <= 0.05 for every H set; later <= 1.0 / 1.5 / 1.5 nats
        dp_bad = [s for s in dp_tol if not e.get(f"logp_dp_maxdiff_{s}", 0.0) <= (0.05 if st == 0 else dp_tol[s])]
        if dp_bad:
            bad.append(f"logp double path {[(s, e.get(f'logp_dp_maxdiff_{s}')) for s in dp_bad]} at step {st}"); break
        miss = [k for k in [f"H_one_{L}" for L in ALL_LAYERS] + [f"logp_H_one_{L}" for L in ALL_LAYERS] +
                ["lam_now", "grad_total", "clip_coef", "grad_lm_share"] if e.get(k) is None]
        if miss:
            bad.append(f"Rev D keys missing at step {st}: {miss[:4]}"); break
        on = bool(e.get("prior_on"))
        if not prior or not on:
            ok_l = e["lam_now"] == 0.0
        else:   # the text site skips batches with < 16 numbers: no prior term there (lam_now 0)
            ok_l = abs(e["lam_now"] - lam_want(c, st)) <= 1e-9 or (a.get("site") == "text" and e["lam_now"] == 0.0)
        if not ok_l:
            bad.append(f"lam_now {e['lam_now']} at step {st} (prior_on {int(on)})"); break
    last = r["curve"][-1]
    if not last.get("held_rej", 0) > 0:
        bad.append("held_rej 0")
    if not prior:
        if last.get("prior_used_frac", 0) != 0:
            bad.append(f"none prior_used_frac {last.get('prior_used_frac')}")
    elif abs(last.get("prior_on_frac", -1) - N_ON / 18310) > 1e-9 or r.get("prior_events") != [[S_ON, 1]]:
        bad.append(f"window prior_on_frac {last.get('prior_on_frac')} events {r.get('prior_events')}")
    return bad


def load(root, code_sha):
    """{cell: {seed: run}}, problems {cell: [reasons]}"""
    runs, probs, other = {}, {}, 0
    for f in sorted(glob.glob(str(Path(root) / "*" / "run_*.json"))):
        r = json.loads(Path(f).read_text())
        c = cell_of(r["args"])
        if c is None:
            other += 1; continue
        s = r["args"]["seed"]
        if s in runs.get(c, {}):
            probs.setdefault(c, []).append(f"seed {s} twice ({f})"); continue
        runs.setdefault(c, {})[s] = r
        bad = check_run(r, code_sha, c)
        if bad:
            probs.setdefault(c, []).extend(f"s{s}: {b}" for b in bad)
    if other:
        print(f"[note] {other} run JSONs are not ptheld2 cells (left out)")
    for s in sorted({s for c in runs for s in runs[c]}):   # per seed across cells
        rs = [(c, runs[c][s]) for c in runs if s in runs[c]]
        c0, r0 = rs[0]
        h0 = r0["held_operands"]
        sha0 = {e["step"]: e.get("held_data_sha") for e in r0["curve"]}
        for c, r in rs[1:]:
            h = r["held_operands"]
            for k in ("H", "count_text_site", "block", "half", "remap_H"):
                if h.get(k) != h0.get(k):
                    probs.setdefault(c, []).append(f"s{s}: held_operands.{k} differs from {c0}")
            if h["sets"]["H_one"]["pairs"] != h0["sets"]["H_one"]["pairs"]:
                probs.setdefault(c, []).append(f"s{s}: H_one pairs differ from {c0}")
            if abs(r["curve"][0]["lm"] - r0["curve"][0]["lm"]) > 1e-3:
                probs.setdefault(c, []).append(f"s{s}: step-0 lm {r['curve'][0]['lm']:.4f} vs {r0['curve'][0]['lm']:.4f} ({c0})")
            diff = [e["step"] for e in r["curve"] if e["step"] in sha0 and e.get("held_data_sha") != sha0[e["step"]]]
            if diff:
                probs.setdefault(c, []).append(f"s{s}: held_data_sha differs from {c0} from step {diff[0]}")
    return runs, probs


def acc(r, L):
    return final3(r, f"H_one_{L}")


def ref_of(runs, s):
    return runs.get("none@2e-3", {}).get(s)


def informative(runs, c, s):
    r = runs[c][s]
    return held6_final3(r) >= 0.80 and RR.diverged_at(r) is None and not RR.unstable(r, ref_of(runs, s))


def x_gate(r):
    return held6_final3(r) >= 0.80 and RR.diverged_at(r) is None


def needed(runs, probs, cells_seeds):
    """'incomplete' / 'withheld' / None for the (cell, seeds) a reading uses"""
    miss = [f"{c} s{s}" for c, ss in cells_seeds for s in ss if s not in runs.get(c, {})]
    if miss:
        return "incomplete", f"INCOMPLETE (missing {miss[:6]}{' ...' if len(miss) > 6 else ''})"
    bad = sorted({c for c, _ in cells_seeds if c in probs})
    if bad:
        return "withheld", f"WITHHELD (invalid runs in {bad})"
    return None, ""


def select(runs, L, seeds):
    """the prior arm with the highest mean acc_L final3 over `seeds`; within TIE of the top -> lower mean bpb"""
    m = {x: float(np.mean([acc(runs[x][s], L) for s in seeds])) for x in PRIOR_X}
    b = {x: float(np.mean([final3(runs[x][s], "val_bits_per_byte") for s in seeds])) for x in PRIOR_X}
    top = max(m.values())
    tied = [x for x in PRIOR_X if m[x] >= top - TIE]
    return min(tied, key=lambda x: (b[x], PRIOR_X.index(x))), m, b, tied


def layer_reading(runs, probs, L):
    """selection on seeds 0-3, then (a) / (b) / (c) / specificity on seeds 4-7 for layer L; p_L for the Holm step"""
    lines, out = [], {"layer": L, "p": float("nan")}
    st, msg = needed(runs, probs, [(x, SEL) for x in PRIOR_X])
    if st:
        out.update(status=st, verdict=msg); lines.append(f"[{L}] selection: {msg}"); return out, lines
    x, m, b, tied = select(runs, L, SEL)
    out["chosen"] = x
    lines.append(f"[{L}] selection (seeds 0-3): mean acc_{L} final3 " + ", ".join(f"{k.split('@')[0]} {v:.3f}" for k, v in m.items())
                 + "; bpb " + ", ".join(f"{k.split('@')[0]} {v:.4f}" for k, v in b.items())
                 + f"; within {TIE} of the top {[t.split('@')[0] for t in tied]} -> chosen {x}")
    st, msg = needed(runs, probs, [(x, CONF), ("none@2e-3", CONF), ("none@8e-3", CONF), ("none@8e-3-4x", SEL), (CTRL, CONF)])
    if st:
        out.update(status=st, verdict=msg); lines.append(f"[{L}] confirmation: {msg}"); return out, lines
    fails = []
    # (a) the floor of every none config
    fa = {}
    for y in NONES:
        ss = SEEDS[y] if y.endswith("-4x") else CONF
        inf = [s for s in ss if informative(runs, y, s)]
        low = [s for s in inf if acc(runs[y][s], L) <= FLOOR]
        need = math.ceil(0.75 * len(ss))
        fa[y] = "uninformative" if len(inf) < need else ("holds" if len(low) >= need else "not at the floor")
        lines.append(f"[{L}] (a) {y} seeds {list(ss)}: acc_{L} final3 {[round(acc(runs[y][s], L), 3) for s in ss]}, "
                     f"informative {len(inf)}/{len(ss)}, informative and <= {FLOOR}: {len(low)} (need {need}) -> {fa[y]}")
        if fa[y] != "holds":
            fails.append(f"(a) {y} {fa[y]}")
    # (b) the chosen arm on the confirm seeds
    va = {s: acc(runs[x][s], L) for s in CONF}
    okb = [s for s in CONF if va[s] >= BAR and x_gate(runs[x][s])]
    hb = {s: held6_final3(runs[x][s]) for s in CONF}
    b_ok = len(okb) >= need_wins(len(CONF))
    lines.append(f"[{L}] (b) {x} seeds 4-7: acc_{L} final3 {[round(va[s], 3) for s in CONF]}, held6 {[round(hb[s], 3) for s in CONF]}; "
                 f">= {BAR} with the gate in {len(okb)}/4 -> {'holds' if b_ok else 'fails'}")
    if not b_ok:
        fails.append(f"(b) {x} >= {BAR} in {len(okb)}/4")
    # (c) paired tests (intersection-union) and the specificity margin
    ps, c_ok, spec = {}, True, None
    for y in Y_PAIRED:
        ss = [s for s in CONF if informative(runs, y, s)]
        if len(ss) < 3:
            lines.append(f"[{L}] (c) {x} - {y}: only {len(ss)} informative control seeds -> fails"); ps[y] = float("nan"); c_ok = False
            continue
        d = [va[s] - acc(runs[y][s], L) for s in ss]
        w = sum(v > 0 for v in d)
        ps[y] = ptest(d)
        ok = w >= need_wins(len(ss))
        c_ok &= ok
        lines.append(f"[{L}] (c) {x} - {y} over seeds {ss}: mean {np.mean(d):+.3f}, wins {w}/{len(ss)} (need {need_wins(len(ss))}), "
                     f"one-sided paired t p {ps[y]:.4g}")
        if y == CTRL:
            spec = float(np.mean(d))
    p_l = max(ps.values()) if all(not math.isnan(v) for v in ps.values()) else float("nan")
    out.update(p=p_l, c_wins=c_ok, spec=spec, fa=fa, b_ok=b_ok)
    if not c_ok:
        fails.append("(c) paired wins")
    if spec is None or spec < SPEC:
        fails.append(f"not specific: {x} - {CTRL} = {'n/a' if spec is None else f'{spec:+.3f}'} < +{SPEC}")
    lines.append(f"[{L}] (c) p_{L} = max over the three controls = {p_l:.4g}; specificity {x} - {CTRL} "
                 f"{'n/a' if spec is None else f'{spec:+.3f}'} (need >= +{SPEC})")
    out.update(status="done", fails=fails)
    out["std_a"], la = std_a(runs, probs, L)
    out["std_d"], ld = std_d(runs, probs, L, x)
    lines += la + ld
    # beside the verdict: held6, LM cost, gradient split of the chosen arm
    db = [final3(runs[x][s], "val_bits_per_byte") - final3(runs["none@2e-3"][s], "val_bits_per_byte") for s in CONF]
    lines.append(f"[{L}] beside: {x} held6 final3 {[round(hb[s], 3) for s in CONF]} (seen addition preserved: "
                 f"{'yes' if all(v >= 0.80 for v in hb.values()) else 'NO'}); bpb - none@2e-3 (paired) {np.mean(db):+.4f} "
                 f"{[round(v, 4) for v in db]}; grad split {grad_split_txt(runs, x, CONF)}")
    return out, lines


def floor_state(runs, y, ss, L):
    """(a)'s floor rule on cell y over seeds ss: 'holds' / 'not at the floor' / 'uninformative', and the counts"""
    inf = [s for s in ss if informative(runs, y, s)]
    low = [s for s in inf if acc(runs[y][s], L) <= FLOOR]
    need = math.ceil(0.75 * len(ss))
    return ("uninformative" if len(inf) < need else ("holds" if len(low) >= need else "not at the floor")), inf, low, need


def std_a(runs, probs, L):
    """rescue_standard (a), amendment SUPP: the LC cells at the floor (seeds 0-3, the (a) rule) and none's optimum
    bracketed on both axes (seeds 0-3): an axis is OPEN when its best cell is at an edge and beats its neighbour by
    > 0.25 nats on logp_H_one_<L> final3 (or, when the best mean acc_L > 0.05, by > 0.02 on acc_L) with >= 3/4 paired wins"""
    if not any(c in runs for c in LC):
        return "NOT RUN", [f"[{L}] rescue_standard (a) bracket: NOT RUN (cells {list(LC)} absent)"]
    st, msg = needed(runs, probs, [(c, SEL) for c in LC + ("none@2e-3", "none@8e-3")])
    if st:
        return msg, [f"[{L}] rescue_standard (a) bracket: {msg}"]
    lines, bad = [], []
    for y in LC:
        fs, inf, low, need = floor_state(runs, y, SEL, L)
        lines.append(f"[{L}] rescue_standard (a) {y} seeds 0-3: acc_{L} final3 {[round(acc(runs[y][s], L), 3) for s in SEL]}, "
                     f"informative {len(inf)}/4, informative and <= {FLOOR}: {len(low)} (need {need}) -> {fs}")
        if fs != "holds":
            bad.append(f"{y} {fs}")
    for name, axis in (("lr", LR_AXIS), ("wd", WD_AXIS)):
        for key, margin, gate in ((f"logp_H_one_{L}", BR_LOGP, None), (f"H_one_{L}", BR_ACC, BR_ACC_MIN)):
            m = {c: float(np.mean([final3(runs[c][s], key) for s in SEL])) for c in axis}
            best = max(axis, key=lambda c: m[c])
            i = axis.index(best)
            state = "closed"
            if (gate is None or m[best] > gate) and i in (0, len(axis) - 1):
                nb = axis[1]
                d = [final3(runs[best][s], key) - final3(runs[nb][s], key) for s in SEL]
                if np.mean(d) > margin and sum(v > 0 for v in d) >= 3:
                    state = f"OPEN at {best} (+{np.mean(d):.3f} vs {nb})"
                    bad.append(f"{name} axis {state} on {key}")
            lines.append(f"[{L}] rescue_standard (a) {name} axis on {key} (seeds 0-3): " + ", ".join(f"{c} {m[c]:.3f}" for c in axis)
                         + f" -> {state}" + ("" if gate is None or m[best] > gate else f" (acc rule off: best mean <= {gate})"))
    return ("met" if not bad else "not met: " + "; ".join(bad)), lines


def remap_share(r, L, evs=3):
    """share of layer-L rows whose greedy answer (pred_H_one) equals u_h + y, pooled over the last `evs` evals"""
    ho = r["held_operands"]
    u = dict(zip(ho["H"], ho["remap_H"]))
    m = np.array([c_ == "1" for c_ in ho["sets"]["H_one"]["masks"][L]])
    want = np.array([(u[a] + b) if a in u else (u[b] + a) for a, b in ho["sets"]["H_one"]["pairs"]])
    ps = [np.array(e["pred_H_one"]) for e in r["curve"][-evs:] if e.get("pred_H_one") is not None]
    return float(np.mean([(p == want)[m].mean() for p in ps])) if ps else float("nan")


def std_d(runs, probs, L, x):
    """rescue_standard (d), amendment SUPP: for an emb arm, win-emb-hremap-always (h at the helix position of u_h) must
    fail: acc_L final3 < 0.50 in >= 3/4 of seeds 4-7 AND chosen - hremap paired mean >= +0.20 with >= 3/4 wins"""
    if not x.startswith("win-emb"):
        return "NOT RUN", [f"[{L}] rescue_standard (d): NOT RUN (no shuffled / remapped control at the text site for {x})"]
    if HREMAP not in runs:
        return "NOT RUN", [f"[{L}] rescue_standard (d): NOT RUN ({HREMAP} absent)"]
    st, msg = needed(runs, probs, [(HREMAP, CONF)])
    if st:
        return msg, [f"[{L}] rescue_standard (d): {msg}"]
    vh = {s: acc(runs[HREMAP][s], L) for s in CONF}
    d = [acc(runs[x][s], L) - vh[s] for s in CONF]
    low, w = sum(v < BAR for v in vh.values()), sum(v > 0 for v in d)
    ok = low >= need_wins(4) and np.mean(d) >= D_MARGIN and w >= need_wins(4)
    sh = {c: [round(remap_share(runs[c][s], L), 3) for s in CONF] for c in (HREMAP, x, "none@2e-3")}
    lines = [f"[{L}] rescue_standard (d) {HREMAP} seeds 4-7: acc_{L} final3 {[round(vh[s], 3) for s in CONF]} (< {BAR} in {low}/4); "
             f"{x} - hremap mean {np.mean(d):+.3f}, wins {w}/4 (need mean >= +{D_MARGIN}) -> {'met' if ok else 'not met'}",
             f"[{L}]   descriptive: share of rows answered u_h + y (final3): " + "; ".join(f"{c} {v}" for c, v in sh.items())]
    return ("met" if ok else f"not met: hremap < {BAR} in {low}/4, {x} - hremap {np.mean(d):+.3f}, wins {w}/4"), lines


def holm(outs):
    """Holm over the two layers on p_L; fills verdict per layer"""
    order = sorted(outs, key=lambda L: (math.isnan(outs[L]["p"]), outs[L]["p"]))
    prev = True
    for i, L in enumerate(order):
        o = outs[L]
        thr = HOLM[i]
        o["holm_alpha"] = thr
        o["holm_pass"] = prev and not math.isnan(o["p"]) and o["p"] < thr
        prev = o["holm_pass"]
        if o.get("status") != "done":
            continue
        f = list(o["fails"])
        if not o["holm_pass"]:
            f.append(f"Holm: p_{L} {o['p']:.4g} not < {thr}" + ("" if i == 0 or outs[order[0]]["holm_pass"] else " (first layer failed)"))
        std = (f"rescue_standard: (a) none's lr x wd bracket {o['std_a']}; (b) the prior arm's own lr sweep NOT RUN; "
               f"(d) remapped-target control {o['std_d']} -- a rescue claim needs (a), (b), (d) met")
        std_fail = [k for k in ("std_a", "std_d") if o[k].startswith("not met")]
        if f:
            o["verdict"] = "no rescue: " + "; ".join(f) + f" [{std}]"
        elif std_fail:
            o["verdict"] = f"no rescue (ptheld2 criteria hold with {o['chosen']}, rescue_standard not met) [{std}]"
        else:
            o["verdict"] = f"RESCUE (tier L) on layer {L} with {o['chosen']} [{std}]"
    return order


def grad_split(r):
    """medians over the evals with a prior term in the loss (lam_now > 0): grad_lm_share, clip_coef; None if none"""
    rows = [(e["grad_lm_share"], e["clip_coef"]) for e in r["curve"] if e.get("lam_now", 0) > 0 and e.get("grad_lm_share") is not None]
    if not rows:
        return None
    return float(np.median([v[0] for v in rows])), float(np.median([v[1] for v in rows])), len(rows)


def grad_split_txt(runs, c, seeds):
    t = []
    for s in seeds:
        g = grad_split(runs[c][s]) if s in runs.get(c, {}) else None
        t.append(f"s{s} -" if g is None else f"s{s} LM share {g[0]:.3f} clip {g[1]:.3f}{' FLAG' if g[0] < 0.5 else ''}")
    return "; ".join(t)


def at_step(r, k, step):
    e = [x for x in r["curve"] if x["step"] == step]
    return e[0].get(k) if e else None


def descriptive(runs, probs):
    lines = ["== descriptive (no claim): final3 means over seeds 0-3 | 4-7 | all; '*' = the cell has invalid runs"]
    fmt = lambda v: "  -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
    mean_over = lambda c, k, ss: (float(np.mean([final3(runs[c][s], k) for s in ss if s in runs[c]]))
                                  if any(s in runs.get(c, {}) for s in ss) else None)
    for c in ALL_CELLS:
        if c not in runs:
            lines.append(f"{c}: no runs"); continue
        star = "*" if c in probs else ""
        ks = ["H_one"] + [f"H_one_{L}" for L in ALL_LAYERS]
        lines.append(f"{c}{star} (seeds {sorted(runs[c])}): " + "  ".join(
            f"{k.replace('H_one_', '') if k != 'H_one' else 'full'} {fmt(mean_over(c, k, SEL))}|{fmt(mean_over(c, k, CONF))}|"
            f"{fmt(mean_over(c, k, SEL + CONF))}" for k in ks))
        lines.append(f"    logp final3 (all seeds): " + "  ".join(
            f"{k.replace('logp_H_one_', '') if k != 'logp_H_one' else 'full'} {mean_over(c, k, SEL + CONF):.2f}"
            for k in ["logp_H_one"] + [f"logp_H_one_{L}" for L in ("hi", "lo_rare", "rare", "common")]))
        for L in ("H_one", "H_one_hi", "H_one_lo_rare"):
            w = [at_step(runs[c][s], L, WIN_END_PTHELD) for s in sorted(runs[c])]
            fe = [runs[c][s]["curve"][-1].get(L) for s in sorted(runs[c])]
            f3 = [final3(runs[c][s], L) for s in sorted(runs[c])]
            lines.append(f"    {L}: step {WIN_END_PTHELD} (ptheld's window end) {[fmt(v) for v in w]}; final eval "
                         f"{[fmt(v) for v in fe]}; final3 {[fmt(v) for v in f3]}")
        lines.append(f"    held6 final3 {[round(held6_final3(runs[c][s]), 3) for s in sorted(runs[c])]}; bpb final3 "
                     f"{[round(final3(runs[c][s], 'val_bits_per_byte'), 4) for s in sorted(runs[c])]}")
        if not c.startswith("none"):
            lines.append(f"    gradient split (on-evals): {grad_split_txt(runs, c, sorted(runs[c]))}")
    for _, L in LAYERS:   # what a selection on the confirm seeds alone would pick: shown only, never used
        if all(s in runs.get(x, {}) for x in PRIOR_X for s in CONF):
            x4, m4, _, _ = select(runs, L, CONF)
            lines.append(f"[{L}] a selection on seeds 4-7 alone would pick {x4} (" + ", ".join(
                f"{k.split('@')[0]} {v:.3f}" for k, v in m4.items()) + ") -- NOT used (the verdict uses the 0-3 pick)")
    return lines


def summarise(root, code_sha):
    runs, probs = load(root, code_sha)
    lines = [f"cells: {{{', '.join(f'{c}: {len(v)}' for c, v in sorted(runs.items()))}}}"]
    for c, b in sorted(probs.items()):
        lines.append(f"INVALID {c}: {b[:4]}{' ...' if len(b) > 4 else ''}")
    outs = {}
    for name, L in LAYERS:
        o, l_ = layer_reading(runs, probs, L)
        outs[L] = o; lines += l_
    order = holm(outs)
    for name, L in LAYERS:
        o = outs[L]
        lines.append(f"VERDICT layer {name} (H_one_{L}): {o['verdict']}  (p_{L} {o['p']:.4g}, Holm alpha {o.get('holm_alpha')}, "
                     f"rank {order.index(L) + 1} of 2)")
    return {"layers": outs, "problems": probs}, lines + descriptive(runs, probs)


# ---------------------------------------------------------------------------------------------------------------- selftest
def fake_run(tpl, cell, seed, rng, accs, held=0.97, bpb=1.22, sha="fake", data_sha_tag="", diverge=False, bad_lines=False,
             lm_share=0.9, lam_bad=False, layer_bad=False, logp_shift=0.0, hremap_share=0.0):
    """a fake ptheld2 run JSON on the template's H / rows; accs = final accuracies of the four row groups
    {'hr': blocks 7-9 rare, 'hc': 7-9 common, 'lr': blocks 1-6 rare, 'lc': 1-6 common}; every layer's accuracy is
    computed from per-row hits and the template's own masks (the trainer's computation), ramping up over training;
    greedy answers (last three evals): the sum on a hit, else u_h + y with probability hremap_share, else a random number"""
    arm, rest = cell.split("@")
    tokens = 1.2e9 if rest.endswith("-4x") else 3e8
    lr = float(rest.split("-4x")[0].split("-wd")[0])
    wd = float(rest.split("-wd")[1].split("-")[0]) if "-wd" in rest else 0.1
    prior = arm != "none"
    a = dict(BASE, target="helix3_hremap" if arm == "win-emb-hremap-always" else ("helix3" if prior else "none"),
             site={"none": "ctx", "win-emb-always": "emb", "win-emb-decay": "emb", "win-text-always": "text", "win-ctx-always": "ctx",
                   "win-emb-hremap-always": "emb"}[arm], lr=lr, tokens=tokens, seed=seed,
             lam=1.0 if prior else 5.0, schedule="window" if prior else "whole", window="0.05,1.0" if prior else "0.05,0.4")
    a.pop("wd")
    if wd != 0.1:
        a["wd"] = wd
    if arm == "win-emb-decay":
        a["lam_decay"] = True
    ho = json.loads(json.dumps(tpl["held_operands"]))
    u = dict(zip(ho["H"], ho["remap_H"]))
    pairs = ho["sets"]["H_one"]["pairs"]
    mk = {k: np.array([c_ == "1" for c_ in v]) for k, v in ho["sets"]["H_one"]["masks"].items()}
    hi, rare = mk["hi"], mk["rare"]
    grp = {"hr": hi & rare, "hc": hi & ~rare, "lr": ~hi & rare, "lc": ~hi & ~rare}
    order = {g: rng.permutation(np.flatnonzero(m)) for g, m in grp.items()}
    steps = STEPS[tokens]
    evs = list(range(0, steps, 250)) + [steps - 1]
    curve = []
    for k, st in enumerate(evs):
        fr = min(1.0, st / (0.4 * steps))
        hits = np.zeros(len(hi), dtype=bool)
        for g, idx in order.items():
            hits[idx[:int(round(accs[g] * fr * len(idx)))]] = True
        on = prior and st >= S_ON
        lam = (0.0 if not on else (1.0 * (1.0 - (st - S_ON) / N_ON) if arm == "win-emb-decay" else 1.0))
        e = dict(step=st, tokens=st * 16384, lm=11.5 if st == 0 else 3.0,
                 val_loss=3.2 if not (diverge and k > 10) else float("nan"), val_bits_per_byte=bpb, geo=None, grad_prior=None,
                 H_one=float(hits.mean()), hits_H_one="".join("1" if v else "0" for v in hits), logp_H_one=-9.0 + 8.0 * float(hits.mean()),
                 held_lines_scanned=LINES_PER_STEP * (st + 1) + (1 if bad_lines and k == 5 else 0), held_rej=10 + st,
                 held_data_sha=f"sha{seed}{data_sha_tag}{st}", prior_on=int(on), prior_used_frac=0.0 if not prior else 0.9,
                 prior_on_frac=(N_ON / 18310 if st == steps - 1 else 0.5) if prior else 0.0,
                 logp_dp_maxdiff_H_one=0.0, logp_dp_maxdiff_H_both=0.0, logp_dp_maxdiff_H_ans=0.0, pred_mismatch_H_one=0,
                 lam_now=lam + (0.5 if lam_bad and on else 0.0), grad_total=1.2, clip_coef=1 / 1.2,
                 grad_lm_share=lm_share if on else 1.0)
        for L in ALL_LAYERS:
            e[f"H_one_{L}"] = float(hits[mk[L]].mean())
            e[f"logp_H_one_{L}"] = -9.0 + 8.0 * e[f"H_one_{L}"] + logp_shift
        if k >= len(evs) - 3:
            e["pred_H_one"] = [(x + y) if hit else ((u[x] + y) if x in u else (u[y] + x)) if rng.random() < hremap_share
                               else int(rng.integers(0, 1999)) for (x, y), hit in zip(pairs, hits)]
        for b in BUCK:
            e[f"held_{b}"] = held * fr if st > 0 else 0.0
        if layer_bad and st == steps - 1:   # a logged layer that does not match its rows (read-out double path)
            e["H_one_hi"] += 0.01
        if a["site"] == "emb" and prior:
            e["grad_prior_emb_rows"] = 1000
        curve.append(e)
    return dict(args=a, steps=steps, curve=curve, code_sha=sha, held_operands=ho, prior_events=[[S_ON, 1]] if prior else [])


def selftest(tpl_path):
    tpl = json.loads(Path(tpl_path).read_text())
    ho = tpl["held_operands"]
    assert ho.get("held_strat") == "value_freq" and ho["layer_rows"]["hi"] == 216, "template must be a Rev D value_freq run (H = 54)"
    mk = {k: np.array([c_ == "1" for c_ in v]) for k, v in ho["sets"]["H_one"]["masks"].items()}
    # double path on the template: the layers' rows = the rows of their h (blocks / halves from held_operands)
    hrow = np.repeat(np.array(ho["H"]), 12)
    blk = dict(zip(ho["H"], ho["block"])); half = dict(zip(ho["H"], ho["half"]))
    assert (mk["hi"] == np.array([blk[h] >= 7 for h in hrow])).all() and (mk["lo_rare"] == np.array([half[h] == "rare" for h in hrow])).all()
    assert all((mk[f"v{j}"] == np.array([blk[h] == j for h in hrow])).all() for j in range(1, 10))
    print(f"selftest: template {Path(tpl_path).name}: layer masks = the rows of their h (hi {int(mk['hi'].sum())}, lo_rare "
          f"{int(mk['lo_rare'].sum())} rows)")
    NONE_ACC = {"hr": 0.02, "hc": 0.04, "lr": 0.03, "lc": 0.40}
    CTX_ACC = {"hr": 0.04, "hc": 0.06, "lr": 0.05, "lc": 0.45}
    LOW = {"hr": 0.20, "hc": 0.25, "lr": 0.25, "lc": 0.6}
    GOOD = {"hr": 0.80, "hc": 0.85, "lr": 0.82, "lc": 0.9}

    def jit(d, s, ci):   # small seed- and cell-dependent jitter (no constant paired differences)
        return {g: min(1.0, max(0.0, v + 0.01 * ((s * 7 + ci * 3 + len(g)) % 3 - 1))) for g, v in d.items()}

    def scenario(name, spec, check, skip=(), supp=False, **kw):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(1)
            for c in (ALL_CELLS if supp else MAIN_CELLS):
                for s in SEEDS[c]:
                    if (c, s) in skip:
                        continue
                    sp = spec.get(c, {})
                    acc_ = sp.get("acc", NONE_ACC if c.startswith("none") else (CTX_ACC if c == CTRL else LOW))
                    if callable(acc_):
                        acc_ = acc_(s)
                    ex = {k: (v(s) if callable(v) else v) for k, v in sp.get("kw", {}).items()}
                    r = fake_run(tpl, c, s, rng, jit(acc_, s, ALL_CELLS.index(c) + s), **ex)
                    od = Path(d) / c.replace("@", "_")
                    od.mkdir(exist_ok=True)
                    (od / f"run_{c}_seed{s}.json").write_text(json.dumps(r))
            o, lines = summarise(d, "fake")
        ok = check(o)
        print(f"selftest {name}: {'PASS' if ok else 'FAIL'}")
        for l_ in lines:
            if l_.startswith(("VERDICT", "INVALID")) or "would pick" in l_ or "selection (seeds 0-3)" in l_:
                print("    " + l_[:260])
        if not ok:
            print("\n".join(lines)); raise SystemExit(1)
        return o, lines

    V = lambda o, L: o["layers"][L].get("verdict", "")
    # S1 the emb prior works on both layers, none and win-ctx at the floor -> RESCUE on A and B
    scenario("S1 rescue (both layers)", {"win-emb-always@2e-3": {"acc": GOOD}},
             lambda o: V(o, "hi").startswith("RESCUE") and V(o, "lo_rare").startswith("RESCUE")
             and o["layers"]["hi"]["chosen"] == "win-emb-always@2e-3" and "NOT RUN" in V(o, "hi"))
    # S2 none not at the floor (none@2e-3 at ~0.25 on both layers) -> no rescue, (a) named
    scenario("S2 none not at the floor", {"win-emb-always@2e-3": {"acc": GOOD}, "none@2e-3": {"acc": LOW}},
             lambda o: all(V(o, L).startswith("no rescue") and "(a) none@2e-3 not at the floor" in V(o, L) for L in ("hi", "lo_rare")))
    # S3 win-ctx-always rises too (0.65): prior - ctx < 0.30 -> not specific
    scenario("S3 win-ctx also rises", {"win-emb-always@2e-3": {"acc": GOOD},
                                       CTRL: {"acc": {"hr": 0.62, "hc": 0.66, "lr": 0.64, "lc": 0.8}}},
             lambda o: all(V(o, L).startswith("no rescue") and "not specific" in V(o, L) for L in ("hi", "lo_rare")))
    # S4a an invalid run (wrong held_lines_scanned) of the chosen arm on a confirm seed -> WITHHELD
    scenario("S4a invalid run", {"win-emb-always@2e-3": {"acc": GOOD, "kw": {"bad_lines": lambda s: s == 5}}},
             lambda o: all(V(o, L).startswith("WITHHELD") for L in ("hi", "lo_rare"))
             and any("held_lines_scanned" in b for b in o["problems"]["win-emb-always@2e-3"]))
    # S4b a missing control run (none@8e-3 seed 6) -> INCOMPLETE; S4c a wrong consumed lambda -> WITHHELD
    scenario("S4b missing run", {"win-emb-always@2e-3": {"acc": GOOD}},
             lambda o: all(V(o, L).startswith("INCOMPLETE") for L in ("hi", "lo_rare")), skip={("none@8e-3", 6)})
    scenario("S4c lam_now wrong", {"win-emb-always@2e-3": {"acc": GOOD}, "win-emb-decay@2e-3": {"kw": {"lam_bad": True}}},
             lambda o: all(V(o, L).startswith("WITHHELD") for L in ("hi", "lo_rare"))
             and any("lam_now" in b for b in o["problems"]["win-emb-decay@2e-3"]))
    # S4d a logged H_one_hi that does not match its rows' hits (layer double path) -> WITHHELD
    scenario("S4d layer != its rows", {"win-emb-always@2e-3": {"acc": GOOD}, "none@2e-3": {"kw": {"layer_bad": lambda s: s == 4}}},
             lambda o: all(V(o, L).startswith("WITHHELD") for L in ("hi", "lo_rare"))
             and any("!= its rows' hits" in b for b in o["problems"]["none@2e-3"]))
    # S5 the selection on 0-3 picks win-text-always; on 4-7 win-emb-always would be best: confirm uses the 0-3 pick only
    scenario("S5 selection only on 0-3", {"win-text-always@2e-3": {"acc": lambda s: GOOD if s < 4 else LOW},
                                          "win-emb-always@2e-3": {"acc": lambda s: {"hr": 0.6, "hc": 0.6, "lr": 0.6, "lc": 0.7} if s < 4 else GOOD}},
             lambda o: all(o["layers"][L]["chosen"] == "win-text-always@2e-3" and V(o, L).startswith("no rescue")
                           and "(b) win-text-always@2e-3" in V(o, L) for L in ("hi", "lo_rare")))
    # S6 layer A rescues, layer B does not (none is not at the floor on the rare halves: its rare 1xx-6xx at 0.35)
    o6, l6 = scenario("S6 A rescue, B not", {"win-emb-always@2e-3": {"acc": GOOD},
                                             **{y: {"acc": {"hr": 0.02, "hc": 0.04, "lr": 0.35, "lc": 0.5}} for y in NONES}},
                      lambda o: V(o, "hi").startswith("RESCUE") and V(o, "lo_rare").startswith("no rescue")
                      and "(a) none@2e-3 not at the floor" in V(o, "lo_rare"))
    assert sum(l_.startswith("VERDICT") for l_ in l6) == 2   # two separate verdicts
    # S7 tie within 0.02 on seeds 0-3 -> the lower bpb (win-emb-decay, bpb 1.20 vs 1.25) is chosen
    scenario("S7 tie -> lower bpb", {"win-emb-always@2e-3": {"acc": GOOD, "kw": {"bpb": 1.25}},
                                     "win-emb-decay@2e-3": {"acc": {g: v - 0.01 for g, v in GOOD.items()}, "kw": {"bpb": 1.20}}},
             lambda o: all(o["layers"][L]["chosen"] == "win-emb-decay@2e-3" and V(o, L).startswith("RESCUE") for L in ("hi", "lo_rare")))
    # S8 two confirm seeds of none@8e-3 never learn seen addition -> (a) uninformative (no vacuous floor)
    scenario("S8 uninformative none", {"win-emb-always@2e-3": {"acc": GOOD}, "none@8e-3": {"kw": {"held": lambda s: 0.5 if s in (4, 5) else 0.97}}},
             lambda o: all("(a) none@8e-3 uninformative" in V(o, L) for L in ("hi", "lo_rare")))
    # S9 the prior gradient dominates in the text arm: the FLAG is printed beside / in the descriptive lines
    _, l9 = scenario("S9 gradient FLAG", {"win-emb-always@2e-3": {"acc": GOOD}, "win-text-always@2e-3": {"kw": {"lm_share": 0.4}}},
                     lambda o: V(o, "hi").startswith("RESCUE"))
    assert any("win-text-always" in l_ for l_ in l9) and any("FLAG" in l_ for l_ in l9), "FLAG not printed"
    print("selftest S9: FLAG printed for the text arm (LM share 0.4)")
    # amendment SUPP: the LC cells and the remapped-target control run
    HRM = {"acc": {"hr": 0.10, "hc": 0.12, "lr": 0.10, "lc": 0.5}, "kw": {"hremap_share": 0.5}}
    # S10 supp run: LC at the floor, bracket closed, hremap fails -> RESCUE with (a) met, (d) met, (b) NOT RUN
    scenario("S10 supp: (a) and (d) met", {"win-emb-always@2e-3": {"acc": GOOD}, HREMAP: HRM},
             lambda o: all(V(o, L).startswith("RESCUE") and "bracket met" in V(o, L) and "control met" in V(o, L)
                           and "lr sweep NOT RUN" in V(o, L) for L in ("hi", "lo_rare")), supp=True)
    # S11 none@1.6e-2 answers H (0.3 on every group) -> (a) not met -> no rescue although the ptheld2 criteria hold
    scenario("S11 supp: LC not at the floor", {"win-emb-always@2e-3": {"acc": GOOD}, HREMAP: HRM,
                                               "none@1.6e-2": {"acc": {"hr": 0.3, "hc": 0.3, "lr": 0.3, "lc": 0.6}}},
             lambda o: all(V(o, L).startswith("no rescue (ptheld2 criteria hold") and "none@1.6e-2 not at the floor" in V(o, L)
                           for L in ("hi", "lo_rare")), supp=True)
    # S12 the remapped target works as well as the true one -> (d) not met
    scenario("S12 supp: hremap also works", {"win-emb-always@2e-3": {"acc": GOOD}, HREMAP: {"acc": GOOD}},
             lambda o: all(V(o, L).startswith("no rescue (ptheld2") and "control not met" in V(o, L) for L in ("hi", "lo_rare")),
             supp=True)
    # S13 none's likelihood of H is best at the lr edge 1.6e-2 by 1 nat (exact match still at the floor) -> bracket OPEN
    scenario("S13 supp: bracket open", {"win-emb-always@2e-3": {"acc": GOOD}, HREMAP: HRM, "none@1.6e-2": {"kw": {"logp_shift": 1.0}}},
             lambda o: all("lr axis OPEN at none@1.6e-2" in V(o, L) for L in ("hi", "lo_rare")), supp=True)
    # S14 the text arm is chosen: (d) has no control at the text site -> NOT RUN, the rest as S10
    scenario("S14 supp: text chosen", {"win-text-always@2e-3": {"acc": GOOD}, HREMAP: HRM},
             lambda o: all(V(o, L).startswith("RESCUE") and "control NOT RUN" in V(o, L) for L in ("hi", "lo_rare")), supp=True)
    print("selftest: all scenarios pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/yh6210/llmadd_torch/runs/ptheld2/out")
    ap.add_argument("--code_sha", default=None, help="expected code_sha (default: sha256[:12] of runs/ptheld2/code/tools/pretrain_range.py)")
    ap.add_argument("--selftest", default=None, metavar="RUN_JSON", help="a Rev D value_freq CPU run JSON as the template of the fakes")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.selftest)
        sys.exit(0)
    sha = args.code_sha or hashlib.sha256(Path("/scratch/yh6210/llmadd_torch/runs/ptheld2/code/tools/pretrain_range.py")
                                          .read_bytes()).hexdigest()[:12]
    _, lines = summarise(args.root, sha)
    print("\n".join(lines))
