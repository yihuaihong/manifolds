"""Pre-registered read-out of the Torch runs/ptheld2 (wall 1 follow-up: H = 54 three-digit numbers, 6 per hundreds block,
3 rare + 3 common by text-site count inside the block; prior arms on from 5% to the end of training at lambda 1). Intent
spec and rules: llmadd_torch/runs/ptheld2/AUDIT.md (written 2026-10-21 before any run; co-primary layers fixed by the user
the same day; amendments SUPP, AUD, AUD2 before any result). Written and tested on FAKE results only (--selftest).

Cells (keyed on r["args"]): none@2e-3, none@8e-3 (seeds 0-7), none@8e-3-4x (seeds 0-3), win-emb-always@2e-3,
win-emb-decay@2e-3, win-text-always@2e-3, win-ctx-always@2e-3 (seeds 0-7). Rendered, the user's call: SUPP = none@1.6e-2,
none@8e-3-wd0.01, none@8e-3-wd1 (seeds 0-3), win-emb-hremap-always@2e-3 (seeds 0-7); SUPP-2 = the three prior arms at lr
1e-3 and 4e-3 (seeds 0-3), win-emb-hremap-decay@2e-3 (seeds 0-7). final3 = the mean of the last three evals (1x: 18000 /
18250 / 18309; 4x: 72750 / 73000 / 73241). acc_L = H_one_<L> (exact match on the rows of layer L, the same rows / forward
as H_one). bpb = val_bits_per_byte final3 (a non-finite bpb is the worst, RR.worst). held6 = mean of the six held_bXY.
win-ctx-always never gives h a value label (its target Gram has seen operands only), but its gradient still reaches E[h]
through attention when h is in the text of the row: the non-specific control, not "cannot touch H".
Co-primary layers (user, 2026-10-21): A = hi (blocks 7-9, 18 h): large h AND (for 35% of its rows) a two-token answer, not
value alone; B = lo_rare = rare by standalone occurrences in the text (the text-site count, 27 h), not token exposure (in
block 1 Spearman(text-site, raw token count) = -0.18: year prefixes). Every rule runs per layer; the two verdicts are
printed separately, never merged. Descriptive only: B' = B without block 1, every layer by sum < 1000 / >= 1000, the
'+' / '=' contexts of H in the text (--textarith), (e)-agreement of the selection seeds.
  informative seed (as ptheld): held6 final3 >= 0.80, not diverged, not unstable (final val_loss <= the same seed's
      none@2e-3 + 0.5 nats; no none@2e-3 reference -> not informative). A prior-arm seed failing the held6 / divergence
      gate counts as a failure of C2 and keeps its measured value in the paired tests.
  Selection (per layer, seeds 0-3 only): the prior arm (win-emb-always / win-emb-decay / win-text-always, at 2e-3) with
      the highest mean acc_L final3; arms within 0.02 of the top -> the lower mean bpb. Confirmation on seeds 4-7 only.
  Core criteria (per layer, confirm seeds):
      C1 floor: every none config (none@2e-3, none@8e-3 on seeds 4-7; none@8e-3-4x on its seeds 0-3): >= ceil(3/4 n) of
          its seeds informative AND every informative seed acc_L final3 <= 0.10 (as ptheld R1(b)); fewer informative ->
          'uninformative' (NOT ESTABLISHED);
      C2 bar: the chosen arm acc_L final3 >= 0.50 with the held6 / divergence gate in >= 3/4 of seeds 4-7;
      C3 paired: chosen minus each of none@2e-3, none@8e-3, win-ctx-always@2e-3 over the seeds 4-7 where the control is
          informative (>= 3, else 'uninformative'): wins >= ceil(3/4 m), one-sided paired t; p_L = the largest of the three
          (intersection-union); specificity: chosen - win-ctx-always paired mean >= +0.30;
      Holm over the two layers: the smaller p_L < .0125, the other < .025, and only if the first passed.
  rescue_standard (report/audits/rescue_standard.md; tier L at most):
      RS(a) bracket of none (SUPP; amendment WD3, POST HOC after the wd 1.0 result): every none configuration that ran
          passes C1's floor rule on seeds 0-3 (LC = none@1.6e-2, wd 0.01, wd 0.3 required; none@1e-3, none@2e-3-4x when run)
          and neither axis is OPEN on seeds 0-3, means and paired differences over informative seeds only (>= 3 per cell,
          else 'uninformative'). lr axis (wd 0.1: [1e-3 if run] / 2e-3 / 8e-3 / 1.6e-2): OPEN = the best cell at an edge
          beats its neighbour by > 0.02 on acc_L (best mean > 0.05) or > 0.25 nats on logp_H_one_<L>, >= 3/4 paired wins.
          wd axis (lr 8e-3: 0.01 / 0.1 / 0.3; wd 1.0 reported, not on the axis): the acc rule, or the median first-token
          rank >= 2x better, or the logp rule only when the seen held_b33 logp is not worse (>= -0.05 nats; audit I4).
          OPEN -> PROVISIONAL (more cells needed), a configuration not at the floor -> not met (no rescue);
      RS(b) the prior arm's own lr sweep (SUPP-2): the chosen arm at lr 1e-3 / 2e-3 / 4e-3, seeds 0-3, mean acc_L final3:
          2e-3 best or within 0.02 of the best -> met (bracket closed); another lr better -> 'open: best lr X != 2e-3,
          CONF not run there'; cells absent -> NOT RUN ('(b) open: prior at one lr');
      RS(d) a remapped-target control in the SAME schedule: win-emb-hremap-always for win-emb-always, win-emb-hremap-decay
          for win-emb-decay, none for win-text-always (NOT RUN); >= 3/4 of its seeds 4-7 informative (else
          'uninformative'), acc_L < 0.50 in >= 3/4 of the informative ones, chosen - control paired mean >= +0.20 with
          >= 3/4 wins over them.
  Verdict per layer: any SUPP / SUPP-2 run the reading uses invalid -> WITHHELD; core missing / invalid -> INCOMPLETE /
      WITHHELD; a core criterion fails -> "no rescue: ..." (only 'uninformative' failures -> "NOT ESTABLISHED: ...");
      core holds and RS(a) / RS(b) / RS(d) all met -> "RESCUE (tier L)"; RS(a) and RS(d) met, RS(b) not run ->
      "PROVISIONAL rescue (tier L; (b) open: prior at one lr)"; any RS 'not met' -> "no rescue (core criteria hold;
      rescue_standard not met)"; otherwise (not run / incomplete / uninformative / open) -> "PROVISIONAL (partial)".
      Never RESCUE unless all three RS criteria are met. The verdict line carries the tier-L, layer and emb wording.
  Beside it (no claim): held6 of the chosen arm, its bpb minus none@2e-3 (paired), the median grad_lm_share, prior / LM
  norm ratio, the prior's norm on the embedding and clip_coef over the on-evals per seed -- GLOBAL norms, descriptive
  only: they cannot show whether the prior dominates the 1000 number rows under AdamW (audit I1), no threshold.
Validity (a failure withholds every verdict that uses the cell): ptheld_summary.check_run's checks with amendment DP,
adapted to ptheld2 (code_sha, base args, steps, evals, held_lines_scanned, held_rej, emb rows, prior_on_frac 17394 / 18310
and prior_events [[916, 1]], none never on), plus: the layers and their logp at every eval; lam_now = the formula when the
prior term is in the loss and 0 otherwise, consistent with the logged geo (text site: the prior applied iff the batch was
usable, geo not None; never with < 16 text numbers; prior_used_frac >= 0.9 x prior_on_frac; emb / ctx: always applied);
grad_total / clip_coef / grad_lm_share at every eval; H = 54, 3 + 3 per block, block = h // 100, the halves re-derived from
count_text_site (rank by count, ties by value, lower 50 = rare) = the logged half labels; the layer masks re-derived from
the pairs and those halves; H_one_<L> and logp_H_one_<L> of the final3 evals = the means of hits_H_one / logp_rows_H_one on
the mask; per seed against the reference cell none@2e-3: H, counts, block, half, remap, H_one pairs equal, step-0 lm within
1e-3, held_data_sha equal at every common eval step (a mismatch is charged to the non-reference cell).
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
TEXT_MIN = 16
PRIOR_X = ("win-emb-always@2e-3", "win-emb-decay@2e-3", "win-text-always@2e-3")
CTRL = "win-ctx-always@2e-3"
NONES = ("none@2e-3", "none@8e-3", "none@8e-3-4x")
Y_PAIRED = ("none@2e-3", "none@8e-3", CTRL)
MAIN_CELLS = NONES + PRIOR_X + (CTRL,)
REF = "none@2e-3"   # the reference cell of the per-seed cross-arm checks (audit m2)
# amendment SUPP (rendered by make_jobs_ptheld2.py --supp, the user's call): rescue_standard (a) none's lr x wd bracket on
# this H (seeds 0-3) and (d) the remapped-target control of win-emb-always (seeds 0-7)
# amendment WD3 (2026-10-22, POST HOC: written after the wd 1.0 result was read, before any wd 0.3 / 1e-3 / 2e-3-4x run;
# make_jobs_ptheld2.py --site tandon --supp3): the wd axis is 0.01 / 0.1 / 0.3 at lr 8e-3; wd 1.0 is reported as it is
# (uninformative: the LM is broken) and no longer blocks the axis; none@1e-3 = the lr axis' lower-edge check; none@2e-3-4x =
# the budget at none's best lr on layer A. Every none configuration that ran passes the C1 floor rule (rescue_standard (a):
# "none fails at every configuration").
LC = ("none@1.6e-2", "none@8e-3-wd0.01", "none@8e-3-wd0.3")
WD1 = "none@8e-3-wd1"
LR_LOW, NONE_4X2 = "none@1e-3", "none@2e-3-4x"
EXTRA_NONE = (LR_LOW, NONE_4X2)
HREMAP = "win-emb-hremap-always@2e-3"
# amendment AUD2 / SUPP-2 (make_jobs_ptheld2.py --supp2): rescue_standard (b) the prior arms' lr sweep (seeds 0-3) and the
# same-schedule control of win-emb-decay (seeds 0-7)
SWEEP_LRS = ("1e-3", "4e-3")
SWEEP = tuple(f"{x.split('@')[0]}@{lr}" for x in PRIOR_X for lr in SWEEP_LRS)
HREMAP_DECAY = "win-emb-hremap-decay@2e-3"
D_CONTROL = {"win-emb-always@2e-3": HREMAP, "win-emb-decay@2e-3": HREMAP_DECAY}   # same schedule; none for the text arm
LR_AXIS, WD_AXIS = ("none@2e-3", "none@8e-3", "none@1.6e-2"), ("none@8e-3-wd0.01", "none@8e-3", "none@8e-3-wd0.3")
WD_OF = {"none@8e-3-wd0.01": 0.01, "none@8e-3-wd1": 1.0, "none@8e-3-wd0.3": 0.3}
SUPP_CELLS = LC + (WD1,) + EXTRA_NONE + (HREMAP,) + SWEEP + (HREMAP_DECAY,)
ALL_CELLS = MAIN_CELLS + SUPP_CELLS
LAYERS = (("A", "hi"), ("B", "lo_rare"))
ALL_LAYERS = tuple(f"v{j}" for j in range(1, 10)) + ("rare", "common", "hi", "lo_rare")
SEL, CONF = (0, 1, 2, 3), (4, 5, 6, 7)
SEEDS = {c: (SEL if c in ("none@8e-3-4x", WD1) + LC + EXTRA_NONE + SWEEP else SEL + CONF) for c in ALL_CELLS}
HOLM = (0.0125, 0.025)
FLOOR, BAR, SPEC, TIE = 0.10, 0.50, 0.30, 0.02
BR_LOGP, BR_ACC, BR_ACC_MIN, D_MARGIN = 0.25, 0.02, 0.05, 0.20   # bracket margins (as ptheld R1); (d) margin (as ptheld R3b)
# WD3: the wd axis' OPEN rule (audit I4: H logp is confounded by calibration under heavy wd): acc (best mean > 0.05, margin
# 0.02), OR the median first-token rank on the layer (rank1_rows_H_one, pooled over the final3 evals; best at an edge with a
# paired mean log2((rank_best + 1) / (rank_nb + 1)) <= -1, i.e. >= 2x better, and >= 3/4 paired wins), OR logp (margin
# 0.25) only when the best cell's seen held_b33 logp final3 is not worse than its neighbour's (paired mean >= -0.05 nats)
WD_RANK_LOG2, WD_SEEN_TOL = -1.0, 0.05
WORDING = {"tier": "tier L = given the structure, the model learns to use it (never 'the model discovered the structure')",
           "hi": "layer A = large h AND (for 35% of its rows) a two-token answer, not value alone",
           "lo_rare": "layer B = rare by standalone occurrences in the text (text-site count), not by token exposure",
           "emb": "emb arm: injecting h's value coordinates into its embedding substitutes for task exposure",
           "scope": ("within the none configurations tested (lr x wd cross at wd 0.1 / lr 8e-3, <= 4x tokens) and only while "
                     "the prior keeps h's coordinates in the embedding (decay: lambda < 0.1 only in the last 9.5% of steps)"),
           "wd3": ("RS(a) completed after the read-out (amendment WD3: wd 0.3 / none@1e-3 / none@2e-3-4x added after the wd 1.0 "
                   "result was read; their criteria fixed before their results)"),
           "never": ("do not write: 'rescue' of any tier before RS(a) is met; 'the model discovered the structure' / 'learned "
                     "h's value'; 'the prior makes the unlearnable learnable' without the tested-configurations and "
                     "coordinates-kept qualifiers; 'long tail / rare numbers' (A is defined by value; B is no rescue); 'still "
                     "usable after the prior fades / persistent'; 'auxiliary loss / LM gradient dominates' for an emb arm; "
                     "'none is 0' or 'none <= 0.10 in every seed'; 'wd 1.0 none did not learn seen addition'; B's gain as a "
                     "rescue; numbers against F90 without 'L40S vs A100, H chosen by quintile'; 'helix geometry does the "
                     "addition in the middle layers'; an hremap share without its layer / seeds")}
# claims a verdict head (the text before its first ' [') must never contain (audit (ii) 'do not write', asserted in holm())
FORBIDDEN = ("discovered the structure", "learned h's value", "learned the value", "unlearnable", "long tail", "rare numbers",
             "persistent", "after the prior fades", "auxiliary loss", "gradient dominates", "none is 0", "in every seed",
             "did not learn seen addition", "middle layers")


def finite(v):
    """a logged number: present, numeric (not bool), finite (AUDIT R05, 2026-10-22)"""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def cell_of(a):
    """cell name of a run's args, or None (not a ptheld2 cell)"""
    x4 = "-4x" if a["tokens"] == 1.2e9 else ""
    if a["target"] == "none":
        wd = a.get("wd", 0.1)
        return f"none@{lrs(a['lr'])}" + (f"-wd{wd:g}" if wd != 0.1 else "") + x4
    dec = bool(a.get("lam_decay", False))
    if a["target"] == "helix3_hremap" and a.get("site") == "emb":
        return f"win-emb-hremap-{'decay' if dec else 'always'}@{lrs(a['lr'])}" + x4
    if a["target"] != "helix3":
        return None
    arm = {("emb", False): "win-emb-always", ("emb", True): "win-emb-decay", ("text", False): "win-text-always",
           ("ctx", False): "win-ctx-always"}.get((a.get("site", "ctx"), dec))
    return None if arm is None else f"{arm}@{lrs(a['lr'])}" + x4


def lam_want(c, step):
    """the lambda an eval's step must have consumed (prior term in the loss)"""
    if "-decay@" in c:
        return 1.0 * (1.0 - (step - S_ON) / N_ON)
    return 1.0


def halves_from_counts(cnt):
    """{number: 'rare' / 'common'} for 100..999: per hundreds block, ranked by text-site count, ties by value, lower 50 rare
    (re-derived independently of the trainer's labels; audit m6)"""
    out = {}
    for b in range(1, 10):
        vals = sorted(range(100 * b, 100 * b + 100), key=lambda v: (cnt[v], v))
        out.update({v: "rare" for v in vals[:50]}); out.update({v: "common" for v in vals[50:]})
    return out


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
    site = a.get("site")
    if prior and ((a.get("schedule"), a.get("window")) != ("window", "0.05,1.0") or a.get("lam") != 1.0):
        bad.append(f"schedule {a.get('schedule')} {a.get('window')} lam {a.get('lam')}")
    ho = r.get("held_operands", {})
    H = ho.get("H", [])
    blk_ok = (len(H) == 54 and ho.get("block") == [h // 100 for h in H] and len(ho.get("half", [])) == 54 and
              all(sum(1 for b, hf in zip(ho["block"], ho["half"]) if (b, hf) == (j, x)) == 3
                  for j in range(1, 10) for x in ("rare", "common")))
    if not blk_ok or (ho.get("layer_rows") or {}).get("hi") != 216 or (ho.get("layer_rows") or {}).get("lo_rare") != 324:
        bad.append("H construction (54 h, 3 + 3 per block, block = h // 100, layer rows hi 216 / lo_rare 324)")
    mks = {L: np.array([c_ == "1" for c_ in m]) for L, m in ((ho.get("sets") or {}).get("H_one") or {}).get("masks", {}).items()}
    if blk_ok:   # double path of the layer masks (audit M6 / m6): halves re-derived from the counts, rows from the pairs
        half_re = halves_from_counts(ho["count_text_site"])
        if [half_re[h] for h in H] != ho["half"]:
            bad.append("half labels != the halves re-derived from count_text_site")
        Hs = set(H)
        prs = ho["sets"]["H_one"]["pairs"]
        hrow = [x if x in Hs else y for x, y in prs]
        want = {"hi": np.array([h >= 700 for h in hrow]), "lo_rare": np.array([half_re[h] == "rare" for h in hrow])}
        if any(((x in Hs) + (y in Hs)) != 1 for x, y in prs) or any(L not in mks or len(mks[L]) != len(prs)
                                                                      or (mks[L] != want[L]).any() for L in want):
            bad.append("layer masks != the masks re-derived from the H_one pairs and the count-derived halves")
    for e in r["curve"][-3:]:   # the layers read: H_one_<L> = hits on the mask; logp_H_one_<L> = logp_rows on the mask
        hits = np.array([c_ == "1" for c_ in e.get("hits_H_one", "")])
        lpr = np.array(e.get("logp_rows_H_one") or [], dtype=float)
        for L in ("hi", "lo_rare"):
            if L not in mks or len(hits) != len(mks[L]) or abs(float(hits[mks[L]].mean()) - e.get(f"H_one_{L}", -1)) > 1e-9:
                bad.append(f"H_one_{L} at step {e['step']} != its rows' hits"); break
            if len(lpr) != len(mks[L]) or abs(float(lpr[mks[L]].mean()) - e.get(f"logp_H_one_{L}", 1e9)) > 1e-3:
                bad.append(f"logp_H_one_{L} at step {e['step']} != the mean of its rows' logp"); break
        # WD3: the wd axis also reads the first-token ranks and the seen held_b33 logp of these evals
        if len(e.get("rank1_rows_H_one") or []) != len(hits) or e.get("logp_held_b33") is None:
            bad.append(f"rank1_rows_H_one / logp_held_b33 missing at step {e['step']}")
    dp_tol = {"H_one": 1.0, "H_both": 1.5, "H_ans": 1.5}
    for e in r["curve"]:
        st = e["step"]
        if e.get("held_lines_scanned") != LINES_PER_STEP * (st + 1):
            bad.append(f"held_lines_scanned {e.get('held_lines_scanned')} at step {st}"); break
        # AUDIT R05 (2026-10-22, external review): a missing or non-finite diagnostic is never a pass (see ptheld_summary)
        if site == "emb" and prior and (e.get("prior_on") is None or (e["prior_on"] and e.get("grad_prior_emb_rows") != 1000)):
            bad.append(f"grad_prior_emb_rows {e.get('grad_prior_emb_rows')} (prior_on {e.get('prior_on')}) at step {st}"); break
        # amendment DP (ptheld, 2026-10-20): step 0 <= 0.05 for every H set; later <= 1.0 / 1.5 / 1.5 nats
        dp_bad = [s for s in dp_tol if not (finite(e.get(f"logp_dp_maxdiff_{s}")) and
                                            e[f"logp_dp_maxdiff_{s}"] <= (0.05 if st == 0 else dp_tol[s]))]
        if not finite(e.get("pred_mismatch_H_one")) or e["pred_mismatch_H_one"] != 0:
            bad.append(f"pred_mismatch_H_one {e.get('pred_mismatch_H_one')} at step {st}"); break
        if dp_bad:
            bad.append(f"logp double path {[(s, e.get(f'logp_dp_maxdiff_{s}')) for s in dp_bad]} at step {st}"); break
        miss = [k for k in [f"H_one_{L}" for L in ALL_LAYERS] + [f"logp_H_one_{L}" for L in ALL_LAYERS] +
                ["lam_now", "grad_total", "clip_coef", "grad_lm_share"] if e.get(k) is None]
        if miss:
            bad.append(f"Rev D keys missing at step {st}: {miss[:4]}"); break
        on = bool(e.get("prior_on"))
        applied = e["lam_now"] > 0
        if not prior or not on:
            ok_l = e["lam_now"] == 0.0
        elif site == "text":   # the text site skips unusable batches (text_ok): applied iff geo was computed (usable)
            usable = e.get("geo") is not None
            few = e.get("text_numbers") is not None and e["text_numbers"] < TEXT_MIN
            ok_l = (applied == usable) and not (applied and few) and (not applied or abs(e["lam_now"] - lam_want(c, st)) <= 1e-9)
        else:   # emb / ctx: the prior is always applicable
            ok_l = applied and e.get("geo") is not None and abs(e["lam_now"] - lam_want(c, st)) <= 1e-9
        if not ok_l:
            bad.append(f"lam_now {e['lam_now']} at step {st} (prior_on {int(on)}, geo {e.get('geo')}, "
                       f"text_numbers {e.get('text_numbers')})"); break
    last = r["curve"][-1]
    if not last.get("held_rej", 0) > 0:
        bad.append("held_rej 0")
    if not prior:
        if last.get("prior_used_frac", 0) != 0:
            bad.append(f"none prior_used_frac {last.get('prior_used_frac')}")
    else:
        if abs(last.get("prior_on_frac", -1) - N_ON / 18310) > 1e-9 or r.get("prior_events") != [[S_ON, 1]]:
            bad.append(f"window prior_on_frac {last.get('prior_on_frac')} events {r.get('prior_events')}")
        used, onf = last.get("prior_used_frac", -1), last.get("prior_on_frac", -1)
        if (site == "text" and not used >= 0.9 * onf) or (site != "text" and abs(used - onf) > 1e-9):
            bad.append(f"prior applied on {used:.4f} of the steps, on {onf:.4f} ({site} site)")
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
    for s in sorted({s for c in runs for s in runs[c]}):   # per seed against the fixed reference cell (audit m2)
        present = [c for c in ALL_CELLS if s in runs.get(c, {})]
        c0 = REF if REF in present else present[0]
        r0 = runs[c0][s]
        h0 = r0["held_operands"]
        sha0 = {e["step"]: e.get("held_data_sha") for e in r0["curve"]}
        for c in present:
            if c == c0:
                continue
            r, h = runs[c][s], runs[c][s]["held_operands"]
            for k in ("H", "count_text_site", "block", "half", "remap_H"):
                if h.get(k) != h0.get(k):
                    probs.setdefault(c, []).append(f"s{s}: held_operands.{k} differs from the reference {c0}")
            if h["sets"]["H_one"]["pairs"] != h0["sets"]["H_one"]["pairs"]:
                probs.setdefault(c, []).append(f"s{s}: H_one pairs differ from the reference {c0}")
            if abs(r["curve"][0]["lm"] - r0["curve"][0]["lm"]) > 1e-3:
                probs.setdefault(c, []).append(f"s{s}: step-0 lm {r['curve'][0]['lm']:.4f} vs {r0['curve'][0]['lm']:.4f} ({c0})")
            diff = [e["step"] for e in r["curve"] if e["step"] in sha0 and e.get("held_data_sha") != sha0[e["step"]]]
            if diff:
                probs.setdefault(c, []).append(f"s{s}: held_data_sha differs from the reference {c0} from step {diff[0]}")
    return runs, probs


def acc(r, L):
    return final3(r, f"H_one_{L}")


def ref_of(runs, s):
    return runs.get(REF, {}).get(s)


def informative(runs, c, s):
    r, ref = runs[c][s], ref_of(runs, s)
    if ref is None:   # audit M5: no none@2e-3 reference -> the unstable gate cannot run -> not informative (never skipped)
        return False
    return held6_final3(r) >= 0.80 and RR.diverged_at(r) is None and not RR.unstable(r, ref)


def x_gate(r):
    return held6_final3(r) >= 0.80 and RR.diverged_at(r) is None


def needed(runs, probs, cells_seeds):
    """'withheld' / 'incomplete' / None for the (cell, seeds) a reading uses; an invalid run wins over a missing one
    (audit m1: SUPP partial AND invalid -> WITHHELD)"""
    bad = sorted({c for c, _ in cells_seeds if c in probs})
    if bad:
        return "withheld", f"WITHHELD (invalid runs in {bad})"
    miss = [f"{c} s{s}" for c, ss in cells_seeds for s in ss if s not in runs.get(c, {})]
    if miss:
        return "incomplete", f"INCOMPLETE (missing {miss[:6]}{' ...' if len(miss) > 6 else ''})"
    return None, ""


def select(runs, L, seeds, cands=PRIOR_X):
    """the arm with the highest mean acc_L final3 over `seeds`; within TIE of the top -> lower mean bpb (non-finite = worst)"""
    m = {x: float(np.mean([acc(runs[x][s], L) for s in seeds])) for x in cands}
    b = {x: float(np.mean([RR.worst(final3(runs[x][s], "val_bits_per_byte")) for s in seeds])) for x in cands}
    top = max(m.values())
    tied = [x for x in cands if m[x] >= top - TIE]
    return min(tied, key=lambda x: (b[x], cands.index(x))), m, b, tied


def floor_state(runs, y, ss, L):
    """the C1 floor rule on cell y over seeds ss (as ptheld R1(b)): 'holds' iff >= ceil(3/4 n) seeds informative and every
    informative seed <= 0.10; 'uninformative' if fewer informative; else 'not at the floor'"""
    inf = [s for s in ss if informative(runs, y, s)]
    high = [s for s in inf if acc(runs[y][s], L) > FLOOR]
    need = math.ceil(0.75 * len(ss))
    return ("uninformative" if len(inf) < need else ("holds" if not high else "not at the floor")), inf, high, need


def layer_reading(runs, probs, L):
    """selection on seeds 0-3, then C1 / C2 / C3 (+ specificity) on seeds 4-7 for layer L, p_L for the Holm step, and the
    rescue_standard states RS(a) / RS(b) / RS(d)"""
    lines, out = [], {"layer": L, "p": float("nan")}
    st, msg = needed(runs, probs, [(x, SEL) for x in PRIOR_X])
    if st:
        out.update(status=st, verdict=msg); lines.append(f"[{L}] selection: {msg}"); return out, lines
    x, m, b, tied = select(runs, L, SEL)
    out["chosen"] = x
    lines.append(f"[{L}] selection (seeds 0-3): mean acc_{L} final3 " + ", ".join(f"{k.split('@')[0]} {v:.3f}" for k, v in m.items())
                 + "; bpb " + ", ".join(f"{k.split('@')[0]} {v:.4f}" for k, v in b.items())
                 + f"; within {TIE} of the top {[t.split('@')[0] for t in tied]} -> chosen {x}")
    # none@2e-3 seeds 0-3 too: the 'unstable' gate of none@8e-3-4x (seeds 0-3) needs them (audit M5: never skipped silently)
    st, msg = needed(runs, probs, [(x, CONF), (REF, SEL + CONF), ("none@8e-3", CONF), ("none@8e-3-4x", SEL), (CTRL, CONF)])
    if st:
        out.update(status=st, verdict=msg); lines.append(f"[{L}] confirmation: {msg}"); return out, lines
    fails = []   # (kind, text): kind 'uninf' (an uninformative control: NOT ESTABLISHED) or 'fail'
    for y in NONES:   # C1 the floor of every none config
        ss = SEEDS[y] if y.endswith("-4x") else CONF
        fs, inf, high, need = floor_state(runs, y, ss, L)
        lines.append(f"[{L}] C1 {y} seeds {list(ss)}: acc_{L} final3 {[round(acc(runs[y][s], L), 3) for s in ss]}, "
                     f"informative {len(inf)}/{len(ss)} (need {need}), informative seeds > {FLOOR}: {high} -> {fs}")
        if fs != "holds":
            fails.append(("uninf" if fs == "uninformative" else "fail", f"C1 {y} {fs}"))
    va = {s: acc(runs[x][s], L) for s in CONF}   # C2 the chosen arm on the confirm seeds
    okb = [s for s in CONF if va[s] >= BAR and x_gate(runs[x][s])]
    hb = {s: held6_final3(runs[x][s]) for s in CONF}
    b_ok = len(okb) >= need_wins(len(CONF))
    lines.append(f"[{L}] C2 {x} seeds 4-7: acc_{L} final3 {[round(va[s], 3) for s in CONF]}, held6 {[round(hb[s], 3) for s in CONF]}; "
                 f">= {BAR} with the gate in {len(okb)}/4 -> {'holds' if b_ok else 'fails'}")
    if not b_ok:
        fails.append(("fail", f"C2 {x} >= {BAR} in {len(okb)}/4"))
    ps, spec = {}, None   # C3 paired tests (intersection-union) and the specificity margin
    for y in Y_PAIRED:
        ss = [s for s in CONF if informative(runs, y, s)]
        if len(ss) < 3:
            lines.append(f"[{L}] C3 {x} - {y}: only {len(ss)} informative control seeds -> uninformative"); ps[y] = float("nan")
            fails.append(("uninf", f"C3 uninformative control {y} ({len(ss)} informative seeds)"))
            continue
        d = [va[s] - acc(runs[y][s], L) for s in ss]
        w = sum(v > 0 for v in d)
        ps[y] = ptest(d)
        if w < need_wins(len(ss)):
            fails.append(("fail", f"C3 {x} - {y} wins {w}/{len(ss)}"))
        lines.append(f"[{L}] C3 {x} - {y} over seeds {ss}: mean {np.mean(d):+.3f}, wins {w}/{len(ss)} (need {need_wins(len(ss))}), "
                     f"one-sided paired t p {ps[y]:.4g}")
        if y == CTRL:
            spec = float(np.mean(d))
    p_l = max(ps.values()) if all(not math.isnan(v) for v in ps.values()) else float("nan")
    if spec is not None and spec < SPEC:
        fails.append(("fail", f"not specific: {x} - {CTRL} = {spec:+.3f} < +{SPEC}"))
    lines.append(f"[{L}] C3 p_{L} = max over the three controls = {p_l:.4g}; specificity {x} - {CTRL} "
                 f"{'n/a' if spec is None else f'{spec:+.3f}'} (need >= +{SPEC})")
    out.update(status="done", fails=fails, p=p_l, spec=spec)
    for k, fn in (("rs_a", std_a), ("rs_b", std_b), ("rs_d", std_d)):
        out[k], l_ = fn(runs, probs, L, x)
        lines += l_
    agree = [s for s in SEL if acc(runs[x][s], L) >= BAR and x_gate(runs[x][s])]   # rescue_standard (e), descriptive
    lines.append(f"[{L}] descriptive (e): {x} on the selection seeds 0-3 >= {BAR} with the gate in {len(agree)}/4 "
                 f"({[round(acc(runs[x][s], L), 3) for s in SEL]}); the verdict uses the confirm seeds only (AUDIT)")
    db = [final3(runs[x][s], "val_bits_per_byte") - final3(runs[REF][s], "val_bits_per_byte") for s in CONF]
    lines.append(f"[{L}] beside: {x} held6 final3 {[round(hb[s], 3) for s in CONF]} (seen addition preserved: "
                 f"{'yes' if all(v >= 0.80 for v in hb.values()) else 'NO'}); bpb - none@2e-3 (paired) {np.mean(db):+.4f} "
                 f"{[round(v, 4) for v in db]}; grad split {grad_split_txt(runs, x, CONF)}")
    return out, lines


def layer_mask(r, L):
    return np.array([c_ == "1" for c_ in r["held_operands"]["sets"]["H_one"]["masks"][L]])


def rank_med(r, L):
    """the median first-token rank (rank1_rows_H_one: 0 = argmax) over layer L's rows, pooled over the final3 evals"""
    m = layer_mask(r, L)
    return float(np.median(np.concatenate([np.asarray(e["rank1_rows_H_one"])[m] for e in r["curve"][-3:]])))


def edge_test(runs, axis, inf_of, val, margin, gate=None, inclusive=False):
    """the bracket rule on one axis and one score (val(run) -> higher = better; means / paired differences over the
    informative seeds 0-3): OPEN iff the best cell is at an edge, its mean exceeds `gate` (if given), and it beats its
    neighbour by a paired mean > margin (>= with inclusive) with >= 3/4 paired wins. -> (state, means, best, neighbour, d)"""
    m = {c: float(np.mean([val(runs[c][s]) for s in inf_of[c]])) for c in axis}
    best = max(axis, key=lambda c: m[c])
    i = axis.index(best)
    if gate is not None and not m[best] > gate:
        return f"closed (best mean <= {gate})", m, best, None, []
    if i not in (0, len(axis) - 1):
        return "closed", m, best, None, []
    nb = axis[1] if i == 0 else axis[-2]
    both = [s for s in SEL if s in inf_of[best] and s in inf_of[nb]]
    d = [val(runs[best][s]) - val(runs[nb][s]) for s in both]
    big = d and (np.mean(d) >= margin if inclusive else np.mean(d) > margin)
    if big and sum(v > 0 for v in d) >= need_wins(len(d)):
        return f"OPEN at {best} ({np.mean(d):+.3f} vs {nb})", m, best, nb, d
    return "closed", m, best, nb, d


def std_a(runs, probs, L, x):
    """RS(a), amendments SUPP / AUD2 / WD3: every none configuration that ran passes the C1 floor rule on seeds 0-3 (the LC
    cells none@1.6e-2, wd 0.01, wd 0.3 are required; none@1e-3 and none@2e-3-4x when they ran) and none's optimum is
    bracketed on both axes, over informative seeds only (audit m4):
      lr axis (wd 0.1): none@1e-3 (if run) / 2e-3 / 8e-3 / 1.6e-2, the AUD2 rules (acc with best mean > 0.05 and margin
          0.02, or logp margin 0.25 nats); with none@1e-3 run, 2e-3 is no longer an edge (WD3 lower-edge check);
      wd axis (lr 8e-3): 0.01 / 0.1 / 0.3, the WD3 rules (acc as above, or the median first-token rank >= 2x better, or logp
          only when the seen held_b33 logp is not worse); wd 1.0 is reported as it is and is not on the axis."""
    if not any(c in runs for c in LC):
        return "NOT RUN", [f"[{L}] RS(a) bracket: NOT RUN (cells {list(LC)} absent)"]
    extra = tuple(c for c in EXTRA_NONE if c in runs)
    st, msg = needed(runs, probs, [(c, SEL) for c in LC + extra + (REF, "none@8e-3")])
    if st:
        return msg, [f"[{L}] RS(a) bracket: {msg}"]
    lines, bad, opens, uninf = [], [], [], []
    for y in LC + extra:
        fs, inf, high, need = floor_state(runs, y, SEL, L)
        lines.append(f"[{L}] RS(a) {y} seeds 0-3: acc_{L} final3 {[round(acc(runs[y][s], L), 3) for s in SEL]}, "
                     f"informative {len(inf)}/4 (need {need}), informative seeds > {FLOOR}: {high} -> {fs}")
        if fs == "uninformative":
            uninf.append(f"{y} uninformative")
        elif fs != "holds":
            bad.append(f"{y} {fs}")
    for c in EXTRA_NONE:
        if c not in runs:
            lines.append(f"[{L}] RS(a) {c}: not run (WD3: {'the lr lower-edge check' if c == LR_LOW else 'budget at lr 2e-3'}"
                         " is applied only when it ran)")
    lr_axis = ((LR_LOW,) if LR_LOW in runs else ()) + LR_AXIS
    inf_of = {c: [s for s in SEL if s in runs.get(c, {}) and informative(runs, c, s)] for c in set(lr_axis + WD_AXIS)}
    acc_v = lambda r: acc(r, L)
    logp_v = lambda r: final3(r, f"logp_H_one_{L}")
    rank_v = lambda r: -math.log2(rank_med(r, L) + 1)
    for name, axis in (("lr", lr_axis), ("wd", WD_AXIS)):
        if any(len(inf_of[c]) < 3 for c in axis):
            uninf.append(f"{name} axis ({ {c: len(inf_of[c]) for c in axis} } informative seeds)")
            lines.append(f"[{L}] RS(a) {name} axis: < 3 informative seeds in a cell -> uninformative"); continue
        rules = [("acc", acc_v, BR_ACC, BR_ACC_MIN, False), ("logp", logp_v, BR_LOGP, None, False)]
        if name == "wd":
            rules.insert(1, ("rank", rank_v, -WD_RANK_LOG2, None, True))
        for rn, val, margin, gate, incl in rules:
            state, m, best, nb, d = edge_test(runs, axis, inf_of, val, margin, gate, incl)
            note = ""
            if name == "wd" and rn == "logp" and state.startswith("OPEN"):   # WD3 / audit I4: the seen-logp gate
                both = [s for s in SEL if s in inf_of[best] and s in inf_of[nb]]
                ds = float(np.mean([final3(runs[best][s], "logp_held_b33") - final3(runs[nb][s], "logp_held_b33") for s in both]))
                if ds < -WD_SEEN_TOL:
                    note = f" -- not counted: seen held_b33 logp {ds:+.3f} nats vs {nb} (calibration, audit I4)"
                    state = "closed (logp not counted)"
                else:
                    note = f" (seen held_b33 logp {ds:+.3f} vs {nb}: counted)"
            if state.startswith("OPEN"):
                opens.append(f"{name} axis {state} on {rn}")
            shown = (lambda v: f"{-v:.2f} log2(rank+1)") if rn == "rank" else (lambda v: f"{v:.3f}")
            lines.append(f"[{L}] RS(a) {name} axis on {rn} (informative seeds 0-3): " + ", ".join(f"{c} {shown(m[c])}" for c in axis)
                         + f" -> {state}{note}")
    if WD1 in runs:   # WD3: reported as it is, not on the axis
        r1 = runs[WD1]
        ref = {s: ref_of(runs, s) for s in r1}
        dv = [round(r1[s]["curve"][-1]["val_loss"] - ref[s]["curve"][-1]["val_loss"], 3) if ref[s] else None for s in sorted(r1)]
        ninf = sum(informative(runs, WD1, s) for s in r1)
        why = ("uninformative: the LM is broken (val_loss > none@2e-3 + 0.5 nats)" if all(v is not None and v > RR.UNSTABLE_NATS for v in dv)
               else "uninformative" if ninf < math.ceil(0.75 * len(r1)) else "informative (reported only)")
        lines.append(f"[{L}] RS(a) {WD1} (WD3: reported, not on the wd axis): informative {ninf}/{len(r1)}; val_loss - none@2e-3 "
                     f"{dv}; held6 {[round(held6_final3(r1[s]), 3) for s in sorted(r1)]}; acc_{L} "
                     f"{[round(acc(r1[s], L), 3) for s in sorted(r1)]} -> {why} (never 'did not learn seen addition')")
    if NONE_4X2 in runs:   # WD3: descriptive budget line at none's best lr on layer A
        ss = [s for s in SEL if s in runs[NONE_4X2] and s in runs.get(REF, {})]
        lines.append(f"[{L}] descriptive {NONE_4X2} vs {REF} (seeds {ss}): acc_{L} "
                     f"{[round(acc(runs[NONE_4X2][s], L), 3) for s in ss]} vs {[round(acc(runs[REF][s], L), 3) for s in ss]}; "
                     f"logp_{L} {[round(logp_v(runs[NONE_4X2][s]), 2) for s in ss]} vs {[round(logp_v(runs[REF][s]), 2) for s in ss]}")
    if bad:
        return "not met: " + "; ".join(bad), lines
    if opens:
        return "open: " + "; ".join(opens), lines
    return ("uninformative: " + "; ".join(uninf) if uninf else "met"), lines


def std_b(runs, probs, L, x):
    """RS(b), amendment AUD2 / SUPP-2: the chosen arm's own lr sweep on seeds 0-3 (1e-3 / 2e-3 / 4e-3)"""
    arm = x.split("@")[0]
    cells = [f"{arm}@{lr}" for lr in SWEEP_LRS]
    if not any(c in runs for c in cells):
        return "NOT RUN", [f"[{L}] RS(b) prior-arm lr sweep: NOT RUN ({cells} absent; the prior at one lr)"]
    st, msg = needed(runs, probs, [(c, SEL) for c in cells])
    if st:
        return msg, [f"[{L}] RS(b) prior-arm lr sweep: {msg}"]
    allc = [f"{arm}@1e-3", x, f"{arm}@4e-3"]
    m = {c: float(np.mean([acc(runs[c][s], L) for s in SEL])) for c in allc}
    best = max(allc, key=lambda c: m[c])
    closed = m[x] >= m[best] - TIE
    lines = [f"[{L}] RS(b) {arm} lr sweep (seeds 0-3, mean acc_{L} final3): " + ", ".join(f"{c.split('@')[1]} {m[c]:.3f}" for c in allc)
             + f" -> best {best.split('@')[1]}; " + ("2e-3 within 0.02 of the best: bracket closed, met" if closed
                                                     else f"open: best lr {best.split('@')[1]} != 2e-3, CONF not run there")]
    return ("met" if closed else f"open: best lr {best.split('@')[1]} != 2e-3, CONF not run there"), lines


def remap_share(r, L, evs=3):
    """share of layer-L rows whose greedy answer (pred_H_one) equals u_h + y, pooled over the last `evs` evals"""
    ho = r["held_operands"]
    u = dict(zip(ho["H"], ho["remap_H"]))
    m = np.array([c_ == "1" for c_ in ho["sets"]["H_one"]["masks"][L]])
    want = np.array([(u[a] + b) if a in u else (u[b] + a) for a, b in ho["sets"]["H_one"]["pairs"]])
    ps = [np.array(e["pred_H_one"]) for e in r["curve"][-evs:] if e.get("pred_H_one") is not None]
    return float(np.mean([(p == want)[m].mean() for p in ps])) if ps else float("nan")


def std_d(runs, probs, L, x):
    """RS(d), amendments SUPP / AUD2: the remapped-target control in the chosen arm's own schedule must fail: >= 3/4 of its
    seeds 4-7 informative (else 'uninformative'), acc_L < 0.50 in >= 3/4 of the informative ones, chosen - control paired
    mean >= +0.20 with >= 3/4 wins over them"""
    ctl = D_CONTROL.get(x)
    if ctl is None:
        return f"NOT RUN (no remapped / shuffled control in the schedule of {x})", [f"[{L}] RS(d): NOT RUN (no remapped / shuffled control in the schedule of {x})"]
    if ctl not in runs:
        return f"NOT RUN ({ctl} absent)", [f"[{L}] RS(d): NOT RUN ({ctl} absent; the same-schedule control of {x})"]
    st, msg = needed(runs, probs, [(ctl, CONF)])
    if st:
        return msg, [f"[{L}] RS(d): {msg}"]
    inf = [s for s in CONF if informative(runs, ctl, s)]
    vh = {s: acc(runs[ctl][s], L) for s in CONF}
    if len(inf) < need_wins(4):
        return (f"uninformative: {ctl} informative in {len(inf)}/4",
                [f"[{L}] RS(d) {ctl}: informative seeds {inf} of 4-7 (held6 {[round(held6_final3(runs[ctl][s]), 3) for s in CONF]}) "
                 f"-> uninformative"])
    d = [acc(runs[x][s], L) - vh[s] for s in inf]
    low, w = sum(vh[s] < BAR for s in inf), sum(v > 0 for v in d)
    ok = low >= need_wins(len(inf)) and np.mean(d) >= D_MARGIN and w >= need_wins(len(inf))
    sh = {c: [round(remap_share(runs[c][s], L), 3) for s in CONF] for c in (ctl, x, REF)}
    lines = [f"[{L}] RS(d) {ctl} informative seeds {inf}: acc_{L} final3 {[round(vh[s], 3) for s in inf]} (< {BAR} in {low}); "
             f"{x} - control mean {np.mean(d):+.3f}, wins {w}/{len(inf)} (need mean >= +{D_MARGIN}) -> {'met' if ok else 'not met'}",
             f"[{L}]   descriptive: share of rows answered u_h + y (final3): " + "; ".join(f"{c} {v}" for c, v in sh.items())]
    return ("met" if ok else f"not met: control < {BAR} in {low}/{len(inf)}, {x} - control {np.mean(d):+.3f}, wins {w}/{len(inf)}"), lines


def holm(outs):
    """Holm over the two layers on p_L, then the verdict per layer (see the module docstring)"""
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
            kind = "uninf" if math.isnan(o["p"]) else "fail"
            f.append((kind, f"Holm: p_{L} {o['p']:.4g} not < {thr}" + ("" if i == 0 or outs[order[0]]["holm_pass"] else " (first layer failed)")))
        rs = {k: o[k] for k in ("rs_a", "rs_b", "rs_d")}
        std = (f"RS(a) none's lr x wd bracket {rs['rs_a']}; RS(b) prior-arm lr sweep {rs['rs_b']}; "
               f"RS(d) same-schedule remapped control {rs['rs_d']}")
        x = o["chosen"]
        words = WORDING[L] + ("; " + WORDING["tier"] + ("; " + WORDING["emb"] if x.startswith("win-emb") else ""))
        words += "; " + WORDING["scope"] + ("" if rs["rs_a"].startswith("NOT RUN") else "; " + WORDING["wd3"])
        subst = [t for k, t in f if k == "fail"]
        if any(v.startswith("WITHHELD") for v in rs.values()):
            v = f"WITHHELD (invalid SUPP runs) [{std}]"
        elif subst:
            v = f"no rescue: {'; '.join(t for _, t in f)} [{std}]"
        elif f:
            v = f"NOT ESTABLISHED (uninformative controls): {'; '.join(t for _, t in f)} [{std}]"
        elif any(s_.startswith("not met") for s_ in rs.values()):
            v = f"no rescue (core criteria hold with {x}; rescue_standard not met) [{std}]"
        elif all(s_ == "met" for s_ in rs.values()):
            v = f"RESCUE (tier L) on layer {L} with {x} [{std}] [wording: {words}]"
        elif rs["rs_a"] == "met" and rs["rs_d"] == "met" and rs["rs_b"] == "NOT RUN":
            v = f"PROVISIONAL rescue (tier L; (b) open: prior at one lr) on layer {L} with {x} [{std}] [wording: {words}]"
        else:
            v = (f"PROVISIONAL (partial; core criteria hold with {x}; rescue_standard criteria not run / incomplete / "
                 f"uninformative / open -> not a rescue) [{std}] [wording: {words}]")
        head_ = v.split(" [")[0].lower()   # audit (ii) 'do not write': the claim itself never carries these phrases
        assert not any(p_ in head_ for p_ in FORBIDDEN), ("forbidden wording in a verdict", v[:120])
        o["verdict"] = v
    return order


GRAD_CAVEAT = ("gradient split = GLOBAL L2 norms over all parameters, descriptive only (no criterion): it cannot show whether "
               "the prior dominates the update of the 1000 number embedding rows (AdamW normalises per coordinate), and the "
               "LM share |g_LM| / |g_LM + g_prior| is 0.71 for equal orthogonal norms, so no share threshold means 'the "
               "prior exceeds the LM' (audit I1)")


def grad_split(r):
    """medians over the evals with a prior term in the loss (lam_now > 0): grad_lm_share, clip_coef, grad_prior / grad_lm
    (global norms), grad_prior_emb (the prior's gradient norm on the embedding, emb site); None if no such eval"""
    rows = [(e["grad_lm_share"], e["clip_coef"], (e.get("grad_prior") or float("nan")) / max(e.get("grad_lm") or float("nan"), 1e-12),
             e.get("grad_prior_emb", float("nan"))) for e in r["curve"] if e.get("lam_now", 0) > 0 and e.get("grad_lm_share") is not None]
    if not rows:
        return None
    return tuple(float(np.nanmedian([v[i] for v in rows])) if not all(math.isnan(v[i]) for v in rows) else float("nan")
                 for i in range(4)) + (len(rows),)


def grad_split_txt(runs, c, seeds):
    t = []
    for s in seeds:
        g = grad_split(runs[c][s]) if s in runs.get(c, {}) else None
        t.append(f"s{s} -" if g is None else f"s{s} LM share {g[0]:.3f} prior/LM {g[2]:.2f}"
                 + ("" if math.isnan(g[3]) else f" prior-on-emb {g[3]:.3g}") + f" clip {g[1]:.3f}")
    return "; ".join(t) + " (global norms, descriptive)"


def at_step(r, k, step):
    e = [x for x in r["curve"] if x["step"] == step]
    return e[0].get(k) if e else None


TEXTARITH = None   # {number: count of '+' / '=' contexts in train.bin} (tools/ptheld2_textarith.py), set by --textarith


def row_masks(r):
    """per-row masks of a run's H_one rows: the two layers, B' = lo_rare without block 1, sum < 1000 / >= 1000"""
    ho = r["held_operands"]
    mk = {k: np.array([c_ == "1" for c_ in v]) for k, v in ho["sets"]["H_one"]["masks"].items()}
    sums = np.array([a + b for a, b in ho["sets"]["H_one"]["pairs"]])
    return {"hi": mk["hi"], "lo_rare": mk["lo_rare"], "lo_rare_nob1": mk["lo_rare"] & ~mk["v1"], "full": np.ones(len(sums), bool)}, \
        sums < 1000


def rows_final3(r, m):
    """mean over the final3 evals of the hit rate on the rows in mask m (from hits_H_one; nan if m is empty)"""
    if not m.any():
        return float("nan")
    return float(np.mean([np.array([c_ == "1" for c_ in e["hits_H_one"]])[m].mean() for e in r["curve"][-3:]]))


def split_lines(runs):
    """audit I2 / I3 / M3 (descriptive, pre-registered 2026-10-21 before any result): B' = the rare halves of blocks 2-9
    (block 1's text-site 'rare' is not token-rare: year prefixes); every layer split by sum < 1000 (one answer token) /
    >= 1000 (two tokens: layer A mixes value with answer length), next to the seen references held_b33 / held_b23
    lt1000 / ge1000; block 1's Spearman(text-site count, raw token count); the '+' / '=' contexts of H in the text"""
    lines = ["== descriptive splits (no claim): final3, seeds 0-3 | 4-7; lt = sum < 1000 (one answer token), ge = sum >= 1000"]
    fmt = lambda v: "  -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
    any_r = next((runs[c][s] for c in ALL_CELLS if c in runs for s in sorted(runs[c])), None)
    if any_r is None:
        return lines
    ho = any_r["held_operands"]
    if ho.get("count_raw"):
        from scipy import stats as _st
        st_, raw_ = np.array(ho["count_text_site"]), np.array(ho["count_raw"])
        rho = {b: float(_st.spearmanr(st_[100 * b:100 * b + 100], raw_[100 * b:100 * b + 100])[0]) for b in range(1, 10)}
        lines.append("  B = 'rare by standalone occurrences in the text' (text-site count), NOT by token exposure: "
                     "Spearman(text-site count, raw token count) per block " + ", ".join(f"b{b} {v:+.2f}" for b, v in rho.items())
                     + " (block 1: 4-digit years split as '17x' + 'd')")
    m0, lt0 = row_masks(any_r)
    lines.append("  share of rows with sum >= 1000 (two answer tokens): " + ", ".join(f"{k} {float((~lt0)[m].mean()):.2f}" for k, m in m0.items()))
    for c in ALL_CELLS:
        if c not in runs:
            continue
        parts = []
        for k in ("hi", "lo_rare", "lo_rare_nob1", "full"):
            for tag, sel in (("lt", True), ("ge", False), ("all", None)):
                v = []
                for ss in (SEL, CONF):
                    xs = []
                    for s in ss:
                        if s in runs[c]:
                            m, lt = row_masks(runs[c][s])
                            xs.append(rows_final3(runs[c][s], m[k] if sel is None else (m[k] & (lt if sel else ~lt))))
                    v.append(float(np.nanmean(xs)) if xs and not all(math.isnan(x) for x in xs) else None)
                parts.append(f"{k}/{tag} {fmt(v[0])}|{fmt(v[1])}")
        ref = [f"{b}_{t} {fmt(float(np.mean([final3(runs[c][s], f'{b}_{t}') for s in sorted(runs[c])])))}"
               for b in ("held_b33", "held_b23") for t in ("lt1000", "ge1000")
               if all(runs[c][s]["curve"][-1].get(f"{b}_{t}") is not None for s in runs[c])]
        lines.append(f"  {c}: " + "  ".join(parts) + "; seen refs (all seeds) " + ", ".join(ref))
    if TEXTARITH is not None:   # audit M3: arithmetic statements with h in the natural text (every arm sees them)
        seeds = sorted({s for c in runs for s in runs[c]})
        for s in seeds:
            r = next(runs[c][s] for c in ALL_CELLS if c in runs and s in runs[c])
            H, half = r["held_operands"]["H"], r["held_operands"]["half"]
            grp = {"hi": [h for h in H if h >= 700], "lo_rare": [h for h, hf in zip(H, half) if hf == "rare"],
                   "lo_rare_nob1": [h for h, hf in zip(H, half) if hf == "rare" and h >= 200], "all H": H}
            lines.append(f"  seed {s}: h next to '+' / '=' in train.bin text (tools/ptheld2_textarith.py): " + "; ".join(
                f"{k} total {sum(TEXTARITH[h] for h in v)} median {int(np.median([TEXTARITH[h] for h in v]))}" for k, v in grp.items()))
    else:
        lines.append("  ('+' / '=' contexts of H in the text: not computed; pass --textarith runs/ptheld2/textarith.json)")
    return lines


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
    lines.append("    note: " + GRAD_CAVEAT)
    lines += split_lines(runs)
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
    lines.append("WORDING: " + WORDING["never"])
    return {"layers": outs, "problems": probs}, lines + descriptive(runs, probs)




# ---------------------------------------------------------------------------------------------------------------- selftest
ARM_SITE = {"none": "ctx", "win-emb-always": "emb", "win-emb-decay": "emb", "win-text-always": "text", "win-ctx-always": "ctx",
            "win-emb-hremap-always": "emb", "win-emb-hremap-decay": "emb"}


def fake_run(tpl, cell, seed, rng, accs, held=0.97, bpb=1.22, sha="fake", data_sha_tag="", diverge=False, bad_lines=False,
             lm_share=0.9, lam_bad=False, layer_bad=False, logp_shift=0.0, hremap_share=0.0, mask_bad=False,
             text_inert=False, half_swap=False, val_last=None, rank_base=20, seen_logp=-0.73):
    """a fake ptheld2 run JSON on the template's H / rows; accs = final accuracies of the four row groups
    {'hr': blocks 7-9 rare, 'hc': 7-9 common, 'lr': blocks 1-6 rare, 'lc': 1-6 common}; every layer's accuracy (and its
    logp) is computed from per-row hits / logp and the template's own masks (the trainer's computation), ramping up over
    training; greedy answers (last three evals): the sum on a hit, else u_h + y with probability hremap_share, else random"""
    arm, rest = cell.split("@")
    tokens = 1.2e9 if rest.endswith("-4x") else 3e8
    lr = float(rest.split("-4x")[0].split("-wd")[0])
    wd = float(rest.split("-wd")[1].split("-")[0]) if "-wd" in rest else 0.1
    prior = arm != "none"
    a = dict(BASE, target="helix3_hremap" if "hremap" in arm else ("helix3" if prior else "none"), site=ARM_SITE[arm], lr=lr,
             tokens=tokens, seed=seed, lam=1.0 if prior else 5.0, schedule="window" if prior else "whole",
             window="0.05,1.0" if prior else "0.05,0.4")
    a.pop("wd")
    if wd != 0.1:
        a["wd"] = wd
    if arm.endswith("-decay"):
        a["lam_decay"] = True
    ho = json.loads(json.dumps(tpl["held_operands"]))
    msk = ho["sets"]["H_one"]["masks"]
    if mask_bad:   # a layer mask that is not the rows of its h
        msk["hi"] = ("0" if msk["hi"][0] == "1" else "1") + msk["hi"][1:]
    if half_swap:   # labels AND masks swapped consistently for one rare / common pair of block 9 (wrong w.r.t. the counts)
        H = ho["H"]
        hr_ = next(h for h, b, hf in zip(H, ho["block"], ho["half"]) if b == 9 and hf == "rare")
        hc_ = next(h for h, b, hf in zip(H, ho["block"], ho["half"]) if b == 9 and hf == "common")
        ho["half"] = ["common" if h == hr_ else ("rare" if h == hc_ else hf) for h, hf in zip(H, ho["half"])]
        Hs = set(H)
        rows = [x if x in Hs else y for x, y in ho["sets"]["H_one"]["pairs"]]
        rare_rows = "".join("1" if (dict(zip(H, ho["half"]))[h] == "rare") else "0" for h in rows)
        msk["rare"] = msk["lo_rare"] = rare_rows
        msk["common"] = "".join("0" if c_ == "1" else "1" for c_ in rare_rows)
    mk = {k: np.array([c_ == "1" for c_ in v]) for k, v in msk.items()}
    u = dict(zip(ho["H"], ho["remap_H"]))
    pairs = ho["sets"]["H_one"]["pairs"]
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
        lam = 0.0 if not on else lam_want(cell, st)
        if text_inert and on:
            lam = 0.0
        lrows = -9.0 + 8.0 * hits.astype(float) + logp_shift
        onf = (N_ON / 18310 if st == steps - 1 else 0.5) if prior else 0.0
        e = dict(step=st, tokens=st * 16384, lm=11.5 if st == 0 else 3.0,
                 val_loss=(val_last if (val_last is not None and k == len(evs) - 1) else 3.2) if not (diverge and k > 10) else float("nan"),
                 val_bits_per_byte=bpb, geo=0.5 if prior else 0.0, grad_prior=None,
                 H_one=float(hits.mean()), hits_H_one="".join("1" if v else "0" for v in hits), logp_H_one=float(lrows.mean()),
                 held_lines_scanned=LINES_PER_STEP * (st + 1) + (1 if bad_lines and k == 5 else 0), held_rej=10 + st,
                 held_data_sha=f"sha{seed}{data_sha_tag}{st}", prior_on=int(on), prior_on_frac=onf, prior_used_frac=onf,
                 text_numbers=120 if ARM_SITE[arm] == "text" else None,
                 logp_dp_maxdiff_H_one=0.0, logp_dp_maxdiff_H_both=0.0, logp_dp_maxdiff_H_ans=0.0, pred_mismatch_H_one=0,
                 lam_now=lam + (0.5 if lam_bad and on else 0.0), grad_total=1.2, clip_coef=1 / 1.2,
                 grad_lm_share=lm_share if lam > 0 else 1.0, logp_held_b33=seen_logp)
        for L in ALL_LAYERS:
            e[f"H_one_{L}"] = float(hits[mk[L]].mean())
            e[f"logp_H_one_{L}"] = float(lrows[mk[L]].mean())
        if k >= len(evs) - 3:
            e["logp_rows_H_one"] = [round(float(v), 4) for v in lrows]
            e["rank1_rows_H_one"] = [0 if v else int(rank_base) for v in hits]
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


def unit_tests():
    """known-answer checks of the Holm step and the verdict ladder, final3 and the NaN tie-break (audit I4)"""
    def mk(p, fails=(), a="met", b="met", d="met"):
        return {"p": p, "status": "done", "fails": list(fails), "rs_a": a, "rs_b": b, "rs_d": d, "chosen": "win-emb-always@2e-3"}
    n = 0
    for (pa, pb), want in (((0.010, 0.020), (True, True)), ((0.015, 0.020), (False, False)), ((0.020, 0.010), (True, True)),
                           ((0.030, 0.010), (False, True)), ((0.013, 0.012), (True, True)), ((float("nan"), 0.012), (False, True)),
                           ((0.012, 0.030), (True, False)), ((0.014, float("nan")), (False, False))):
        outs = {"hi": mk(pa), "lo_rare": mk(pb)}
        holm(outs)
        got = tuple(outs[L]["verdict"].startswith("RESCUE") for L in ("hi", "lo_rare"))
        assert got == want, ("Holm", pa, pb, got, want, outs["hi"]["holm_alpha"], outs["lo_rare"]["holm_alpha"])
        n += 1
    for (a, b, d), want in ((("met", "met", "met"), "RESCUE (tier L)"),
                            (("met", "NOT RUN", "met"), "PROVISIONAL rescue (tier L; (b) open: prior at one lr)"),
                            (("NOT RUN", "NOT RUN", "met"), "PROVISIONAL (partial"), (("INCOMPLETE (x)", "met", "met"), "PROVISIONAL (partial"),
                            (("met", "open: best lr 4e-3 != 2e-3, CONF not run there", "met"), "PROVISIONAL (partial"),
                            (("met", "met", "uninformative: c"), "PROVISIONAL (partial"), (("WITHHELD (x)", "met", "met"), "WITHHELD"),
                            (("met", "met", "not met: y"), "no rescue (core"), (("not met: z", "NOT RUN", "NOT RUN"), "no rescue (core"),
                            (("WITHHELD (x)", "NOT RUN", "not met: y"), "WITHHELD"), (("met", "WITHHELD (x)", "met"), "WITHHELD")):
        outs = {"hi": mk(0.001, a=a, b=b, d=d), "lo_rare": mk(0.001)}
        holm(outs)
        assert outs["hi"]["verdict"].startswith(want), ((a, b, d), outs["hi"]["verdict"][:80], want)
        assert outs["hi"]["verdict"].startswith("RESCUE") == (want == "RESCUE (tier L)")
        n += 1
    outs = {"hi": mk(0.001, fails=[("uninf", "C1 none@8e-3 uninformative")]), "lo_rare": mk(0.001, fails=[("fail", "C2 x")])}
    holm(outs)
    assert outs["hi"]["verdict"].startswith("NOT ESTABLISHED") and outs["lo_rare"]["verdict"].startswith("no rescue: C2"), outs
    n += 1
    vals = [0.1] * 71 + [0.2, 0.4, 0.6, 0.9]   # final3 = the last three evals, of the layer's own key (every key present)
    r = {"curve": [{"H_one": 0.07, **{f"H_one_{L_}": 0.05 + 0.01 * i for i, L_ in enumerate(ALL_LAYERS)},
                    "H_one_hi": v, "H_one_lo_rare": 1 - v, "H_one_rare": 1 - v} for v in vals]}   # rare = lo_rare (same rows)
    assert abs(acc(r, "hi") - (0.4 + 0.6 + 0.9) / 3) < 1e-12, ("final3 / layer key", acc(r, "hi"))
    assert abs(acc(r, "lo_rare") - (0.6 + 0.4 + 0.1) / 3) < 1e-12, ("final3 / layer key", acc(r, "lo_rare"))
    n += 1
    for nan_arm in PRIOR_X:   # a non-finite bpb inside the tie never wins (RR.worst; ptheld B5)
        runs = {x: {s: {"curve": [{"H_one_hi": 0.80, "val_bits_per_byte": float("nan") if x == nan_arm else
                                   (1.21 if x == PRIOR_X[-1] and nan_arm != PRIOR_X[-1] else 1.22)}] * 3} for s in SEL} for x in PRIOR_X}
        x, _, _, _ = select(runs, "hi", SEL)
        assert x != nan_arm, ("NaN bpb won the tie", nan_arm)
        n += 1
    return n


def selftest(tpl_path):
    tpl = json.loads(Path(tpl_path).read_text())
    ho = tpl["held_operands"]
    assert ho.get("held_strat") == "value_freq" and ho["layer_rows"]["hi"] == 216, "template must be a Rev D value_freq run (H = 54)"
    mk = {k: np.array([c_ == "1" for c_ in v]) for k, v in ho["sets"]["H_one"]["masks"].items()}
    hrow = np.repeat(np.array(ho["H"]), 12)
    blk = dict(zip(ho["H"], ho["block"])); half = dict(zip(ho["H"], ho["half"]))
    assert (mk["hi"] == np.array([blk[h] >= 7 for h in hrow])).all() and (mk["lo_rare"] == np.array([half[h] == "rare" for h in hrow])).all()
    assert all((mk[f"v{j}"] == np.array([blk[h] == j for h in hrow])).all() for j in range(1, 10))
    hrc = halves_from_counts(ho["count_text_site"])
    assert [hrc[h] for h in ho["H"]] == ho["half"], "the template's half labels != the count-derived halves"
    print(f"selftest: template {Path(tpl_path).name}: layer masks = the rows of their h (hi {int(mk['hi'].sum())}, lo_rare "
          f"{int(mk['lo_rare'].sum())} rows); half labels = the count-derived halves")
    print(f"selftest: {unit_tests()} unit checks pass (Holm known answers incl. the chain, the verdict ladder, final3, NaN tie)")
    NONE_ACC = {"hr": 0.02, "hc": 0.04, "lr": 0.03, "lc": 0.40}
    CTX_ACC = {"hr": 0.04, "hc": 0.06, "lr": 0.05, "lc": 0.45}
    LOW = {"hr": 0.20, "hc": 0.25, "lr": 0.25, "lc": 0.6}
    GOOD = {"hr": 0.80, "hc": 0.85, "lr": 0.82, "lc": 0.9}
    FLAT = lambda v: {g: v for g in ("hr", "hc", "lr", "lc")}
    HRM = {"acc": {"hr": 0.10, "hc": 0.12, "lr": 0.10, "lc": 0.5}, "kw": {"hremap_share": 0.5}}
    GROUPS = {"a": LC, "d": (HREMAP,), "b": SWEEP, "e": (HREMAP_DECAY,), "w": (WD1,), "x": EXTRA_NONE}
    count = [0]

    def jit(d, s, ci):   # small seed- and cell-dependent jitter (no constant paired differences)
        return {g: min(1.0, max(0.0, v + 0.01 * ((s * 7 + ci * 3 + len(g)) % 3 - 1))) for g, v in d.items()}

    def scenario(name, spec, check, skip=(), supp=""):
        count[0] += 1
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(1)
            for c in MAIN_CELLS + tuple(c for g in supp for c in GROUPS[g]):
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
            if l_.startswith(("VERDICT", "INVALID")):
                print("    " + l_[:230])
        if not ok:
            print("\n".join(lines)); raise SystemExit(1)
        return o, lines

    V = lambda o, L: o["layers"][L].get("verdict", "")
    BOTH = ("hi", "lo_rare")
    EA = "win-emb-always@2e-3"
    # --- core criteria (main cells only: every rescue_standard criterion NOT RUN -> never RESCUE)
    scenario("S1 core holds, SUPP absent -> PROVISIONAL", {EA: {"acc": GOOD}},
             lambda o: all(V(o, L).startswith("PROVISIONAL (partial") and "RESCUE" not in V(o, L).split("[")[0] for L in BOTH))
    scenario("S2 none not at the floor", {EA: {"acc": GOOD}, "none@2e-3": {"acc": LOW}},
             lambda o: all(V(o, L).startswith("no rescue: ") and "C1 none@2e-3 not at the floor" in V(o, L) for L in BOTH))
    scenario("S3 win-ctx also rises", {EA: {"acc": GOOD}, CTRL: {"acc": {"hr": 0.62, "hc": 0.66, "lr": 0.64, "lc": 0.8}}},
             lambda o: all(V(o, L).startswith("no rescue") and "not specific" in V(o, L) for L in BOTH))
    scenario("S4a invalid run", {EA: {"acc": GOOD, "kw": {"bad_lines": lambda s: s == 5}}},
             lambda o: all(V(o, L).startswith("WITHHELD") for L in BOTH) and any("held_lines_scanned" in b for b in o["problems"][EA]))
    scenario("S4b missing run", {EA: {"acc": GOOD}}, lambda o: all(V(o, L).startswith("INCOMPLETE") for L in BOTH), skip={("none@8e-3", 6)})
    scenario("S4c lam_now wrong", {EA: {"acc": GOOD}, "win-emb-decay@2e-3": {"kw": {"lam_bad": True}}},
             lambda o: all(V(o, L).startswith("WITHHELD") for L in BOTH) and any("lam_now" in b for b in o["problems"]["win-emb-decay@2e-3"]))
    scenario("S4d layer != its rows", {EA: {"acc": GOOD}, "none@2e-3": {"kw": {"layer_bad": lambda s: s == 4}}},
             lambda o: all(V(o, L).startswith("WITHHELD") for L in BOTH) and any("!= its rows' hits" in b for b in o["problems"]["none@2e-3"]))
    scenario("S5 selection only on 0-3", {"win-text-always@2e-3": {"acc": lambda s: GOOD if s < 4 else LOW},
                                          EA: {"acc": lambda s: FLAT(0.6) if s < 4 else GOOD}},
             lambda o: all(o["layers"][L]["chosen"] == "win-text-always@2e-3" and V(o, L).startswith("no rescue")
                           and "C2 win-text-always@2e-3" in V(o, L) for L in BOTH))
    o6, l6 = scenario("S6 A passes, B not (SUPP absent)", {EA: {"acc": GOOD}, **{y: {"acc": {"hr": 0.02, "hc": 0.04, "lr": 0.35, "lc": 0.5}} for y in NONES}},
                      lambda o: V(o, "hi").startswith("PROVISIONAL") and V(o, "lo_rare").startswith("no rescue")
                      and "C1 none@2e-3 not at the floor" in V(o, "lo_rare"))
    assert sum(l_.startswith("VERDICT") for l_ in l6) == 2   # two separate verdicts
    scenario("S7 tie -> lower bpb", {EA: {"acc": GOOD, "kw": {"bpb": 1.25}},
                                     "win-emb-decay@2e-3": {"acc": {g: v - 0.01 for g, v in GOOD.items()}, "kw": {"bpb": 1.20}}},
             lambda o: all(o["layers"][L]["chosen"] == "win-emb-decay@2e-3" and V(o, L).startswith("PROVISIONAL") for L in BOTH))
    scenario("S8 uninformative none -> NOT ESTABLISHED", {EA: {"acc": GOOD}, "none@8e-3": {"kw": {"held": lambda s: 0.5 if s in (4, 5) else 0.97}}},
             lambda o: all(V(o, L).startswith("NOT ESTABLISHED") and "C1 none@8e-3 uninformative" in V(o, L) for L in BOTH))
    _, l9 = scenario("S9 gradient split descriptive", {EA: {"acc": GOOD}, "win-text-always@2e-3": {"kw": {"lm_share": 0.4}}},
                     lambda o: V(o, "hi").startswith("PROVISIONAL"))
    assert any("LM share 0.400" in l_ and "global norms, descriptive" in l_ for l_ in l9) and not any("FLAG" in l_ for l_ in l9)
    assert any("GLOBAL L2 norms" in l_ and "no criterion" in l_ for l_ in l9)
    scenario("S23 floor strict: one informative none seed at 0.6", {EA: {"acc": GOOD}, "none@8e-3": {"acc": lambda s: FLAT(0.6) if s == 7 else NONE_ACC}},
             lambda o: all(V(o, L).startswith("no rescue") and "C1 none@8e-3 not at the floor" in V(o, L) for L in BOTH))
    scenario("S24 4x not at the floor", {EA: {"acc": GOOD}, "none@8e-3-4x": {"acc": FLAT(0.3)}},
             lambda o: all(V(o, L).startswith("no rescue") and "C1 none@8e-3-4x not at the floor" in V(o, L) for L in BOTH))
    scenario("S25 chosen arm fails the held6 gate in 2 confirm seeds", {EA: {"acc": GOOD, "kw": {"held": lambda s: 0.5 if s in (4, 5) else 0.97}}},
             lambda o: all(V(o, L).startswith("no rescue") and f"C2 {EA} >= 0.5 in 2/4" in V(o, L) for L in BOTH))
    scenario("S26 chosen arm at 0.40 (below the 0.50 bar)", {EA: {"acc": FLAT(0.40)}},
             lambda o: all(o["layers"][L]["chosen"] == EA and V(o, L).startswith("no rescue") and f"C2 {EA} >= 0.5 in 0/4" in V(o, L) for L in BOTH))
    # --- rescue_standard: SUPP (a = LC, d = hremap-always), SUPP-2 (b = the prior arms' lr sweep, e = hremap-decay)
    scenario("S10 RS(a), RS(d) met, RS(b) not run -> PROVISIONAL rescue", {EA: {"acc": GOOD}, HREMAP: HRM},
             lambda o: all(V(o, L).startswith("PROVISIONAL rescue (tier L; (b) open: prior at one lr)") for L in BOTH), supp="ad")
    scenario("S19 RS(a), RS(b), RS(d) met -> RESCUE", {EA: {"acc": GOOD}, HREMAP: HRM},
             lambda o: all(V(o, L).startswith("RESCUE (tier L)") and "given the structure" in V(o, L) and "injecting h's value" in V(o, L)
                           for L in BOTH) and "two-token answer" in V(o, "hi") and "standalone occurrences" in V(o, "lo_rare"), supp="adb")
    scenario("S20 RS(b) open: the arm is better at 4e-3", {EA: {"acc": GOOD}, HREMAP: HRM, "win-emb-always@4e-3": {"acc": FLAT(0.97)}},
             lambda o: all(V(o, L).startswith("PROVISIONAL (partial") and "open: best lr 4e-3 != 2e-3, CONF not run there" in V(o, L)
                           for L in BOTH), supp="adb")
    scenario("S21 decay chosen, its same-schedule control absent", {"win-emb-decay@2e-3": {"acc": GOOD}, HREMAP: HRM},
             lambda o: all(o["layers"][L]["chosen"] == "win-emb-decay@2e-3" and V(o, L).startswith("PROVISIONAL (partial")
                           and f"{HREMAP_DECAY} absent" in V(o, L) for L in BOTH), supp="adb")
    scenario("S21b decay chosen, hremap-decay fails -> RESCUE", {"win-emb-decay@2e-3": {"acc": GOOD}, HREMAP: {"acc": GOOD}, HREMAP_DECAY: HRM},
             lambda o: all(V(o, L).startswith("RESCUE (tier L)") and "win-emb-decay@2e-3" in V(o, L) for L in BOTH), supp="adbe")
    scenario("S22 the control broke seen addition -> RS(d) uninformative", {EA: {"acc": GOOD},
             HREMAP: {"acc": FLAT(0.02), "kw": {"held": 0.5}}},
             lambda o: all(V(o, L).startswith("PROVISIONAL (partial") and "control uninformative" in V(o, L) for L in BOTH), supp="adb")
    scenario("S27 RS(d) margin missed (chosen 0.6, control 0.45)", {EA: {"acc": FLAT(0.6)}, HREMAP: {"acc": FLAT(0.45)}},
             lambda o: all(V(o, L).startswith("no rescue (core criteria hold") and "control not met" in V(o, L) for L in BOTH), supp="adb")
    scenario("S11 an LC cell not at the floor", {EA: {"acc": GOOD}, HREMAP: HRM, "none@1.6e-2": {"acc": {"hr": 0.3, "hc": 0.3, "lr": 0.3, "lc": 0.6}}},
             lambda o: all(V(o, L).startswith("no rescue (core criteria hold") and "none@1.6e-2 not at the floor" in V(o, L) for L in BOTH), supp="adb")
    scenario("S12 the remapped control works as well", {EA: {"acc": GOOD}, HREMAP: {"acc": GOOD}},
             lambda o: all(V(o, L).startswith("no rescue (core") and "control not met" in V(o, L) for L in BOTH), supp="adb")
    scenario("S13 bracket open", {EA: {"acc": GOOD}, HREMAP: HRM, "none@1.6e-2": {"kw": {"logp_shift": 1.0}}},
             lambda o: all("lr axis OPEN at none@1.6e-2" in V(o, L) for L in BOTH), supp="adb")
    scenario("S32 an LC cell uninformative -> PROVISIONAL, not no rescue", {EA: {"acc": GOOD}, HREMAP: HRM, "none@8e-3-wd0.3": {"kw": {"val_last": 4.5}}},
             lambda o: all(V(o, L).startswith("PROVISIONAL (partial") and "bracket uninformative" in V(o, L) for L in BOTH), supp="adb")
    scenario("S14 text chosen -> RS(d) NOT RUN", {"win-text-always@2e-3": {"acc": GOOD}, HREMAP: HRM},
             lambda o: all(V(o, L).startswith("PROVISIONAL (partial") and "control NOT RUN" in V(o, L) for L in BOTH), supp="adb")
    scenario("S15 SUPP invalid", {EA: {"acc": GOOD}, HREMAP: {**HRM, "kw": {"hremap_share": 0.5, "bad_lines": lambda s: s == 5}}},
             lambda o: all(V(o, L).startswith("WITHHELD (invalid SUPP") for L in BOTH), supp="adb")
    scenario("S16 SUPP incomplete", {EA: {"acc": GOOD}, HREMAP: HRM},
             lambda o: all(V(o, L).startswith("PROVISIONAL (partial") and "bracket INCOMPLETE" in V(o, L) for L in BOTH),
             supp="adb", skip={("none@1.6e-2", 2)})
    scenario("S28 SUPP incomplete AND invalid -> WITHHELD", {EA: {"acc": GOOD}, HREMAP: HRM, "none@1.6e-2": {"kw": {"bad_lines": lambda s: s == 1}}},
             lambda o: all(V(o, L).startswith("WITHHELD (invalid SUPP") for L in BOTH), supp="adb", skip={("none@1.6e-2", 2)})
    scenario("S29 a SUPP run's data stream differs -> charged to it, not to the main cells",
             {EA: {"acc": GOOD}, HREMAP: HRM, "none@1.6e-2": {"kw": {"data_sha_tag": lambda s: "x" if s == 0 else ""}}},
             lambda o: "none@1.6e-2" in o["problems"] and not any(c in o["problems"] for c in MAIN_CELLS)
             and all(V(o, L).startswith("WITHHELD (invalid SUPP") for L in BOTH), supp="adb")
    scenario("S17 none@2e-3 seed missing", {EA: {"acc": GOOD}},
             lambda o: all(V(o, L).startswith("INCOMPLETE") and "none@2e-3 s1" in V(o, L) for L in BOTH), skip={("none@2e-3", 1)})
    scenario("S18 layer mask != its h", {EA: {"acc": GOOD}, "none@8e-3": {"kw": {"mask_bad": lambda s: s == 6}}},
             lambda o: all(V(o, L).startswith("WITHHELD") for L in BOTH) and any("layer masks" in b for b in o["problems"]["none@8e-3"]))
    scenario("S30 the text prior never applied (lam_now 0 on usable batches)", {EA: {"acc": GOOD}, "win-text-always@2e-3": {"kw": {"text_inert": True}}},
             lambda o: all(V(o, L).startswith("WITHHELD") for L in BOTH) and any("lam_now" in b for b in o["problems"]["win-text-always@2e-3"]))
    scenario("S31 half labels and masks agree but not with the counts", {EA: {"acc": GOOD}, "none@8e-3": {"kw": {"half_swap": lambda s: s == 5}}},
             lambda o: all(V(o, L).startswith("WITHHELD") for L in BOTH) and any("half labels" in b for b in o["problems"]["none@8e-3"]))
    # --- amendment WD3 (wd axis 0.01 / 0.1 / 0.3; wd 1.0 reported; none@1e-3 lower edge; none@2e-3-4x); the real pattern:
    # wd 1.0 uninformative (the LM broken: val_loss +0.75 nats), layer B's none not at the floor
    W1 = {"none@8e-3-wd1": {"kw": {"val_last": 3.95}}, **{y: {"acc": {"hr": 0.02, "hc": 0.04, "lr": 0.35, "lc": 0.5}}
                                                       for y in NONES + LC + EXTRA_NONE}}
    WOK = {EA: {"acc": GOOD}, HREMAP: HRM, **W1}
    H35 = {"hr": 0.02, "hc": 0.04, "lr": 0.35, "lc": 0.5}
    scenario("W1 wd 0.3 informative and at the floor -> layer A RESCUE (tier L), B no rescue", WOK,
             lambda o: V(o, "hi").startswith("RESCUE (tier L)") and "amendment WD3" in V(o, "hi") and "lr 1e-3" not in V(o, "hi")[:20]
             and "within the none configurations tested" in V(o, "hi") and V(o, "lo_rare").startswith("no rescue")
             and "uninformative: the LM is broken" not in V(o, "hi"), supp="adbwx")
    scenario("W2 wd 0.3 uninformative -> PROVISIONAL", {**WOK, "none@8e-3-wd0.3": {"acc": H35, "kw": {"val_last": 4.5}}},
             lambda o: V(o, "hi").startswith("PROVISIONAL (partial") and "none@8e-3-wd0.3 uninformative" in V(o, "hi"), supp="adbwx")
    scenario("W3 wd 0.3 OPEN on acc (0.085, at the floor) -> PROVISIONAL", {**WOK, "none@8e-3-wd0.3": {"acc": {"hr": 0.085, "hc": 0.085, "lr": 0.35, "lc": 0.5}}},
             lambda o: V(o, "hi").startswith("PROVISIONAL (partial") and "wd axis OPEN at none@8e-3-wd0.3" in V(o, "hi") and "on acc" in V(o, "hi"),
             supp="adbwx")
    scenario("W3b wd 0.3 OPEN on the first-token rank (4x better) -> PROVISIONAL", {**WOK, "none@8e-3-wd0.3": {"acc": H35, "kw": {"rank_base": 4}}},
             lambda o: V(o, "hi").startswith("PROVISIONAL (partial") and "wd axis OPEN at none@8e-3-wd0.3" in V(o, "hi") and "on rank" in V(o, "hi"),
             supp="adbwx")
    scenario("W3c wd 0.3 better on H logp but worse on seen logp -> logp not counted -> RESCUE",
             {**WOK, "none@8e-3-wd0.3": {"acc": H35, "kw": {"logp_shift": 1.0, "seen_logp": -2.6}}},
             lambda o: V(o, "hi").startswith("RESCUE (tier L)"), supp="adbwx")
    scenario("W3d wd 0.3 better on H logp, seen logp unchanged -> OPEN -> PROVISIONAL",
             {**WOK, "none@8e-3-wd0.3": {"acc": H35, "kw": {"logp_shift": 1.0}}},
             lambda o: V(o, "hi").startswith("PROVISIONAL (partial") and "wd axis OPEN at none@8e-3-wd0.3" in V(o, "hi") and "on logp" in V(o, "hi"),
             supp="adbwx")
    scenario("W4 wd 0.3 with an informative seed > 0.10 -> no rescue", {**WOK, "none@8e-3-wd0.3": {"acc": lambda s: FLAT(0.3) if s == 2 else H35}},
             lambda o: V(o, "hi").startswith("no rescue (core criteria hold") and "none@8e-3-wd0.3 not at the floor" in V(o, "hi"), supp="adbwx")
    scenario("W5 none@1e-3 best on the lr axis and OPEN -> PROVISIONAL", {**WOK, "none@1e-3": {"acc": {"hr": 0.085, "hc": 0.085, "lr": 0.35, "lc": 0.5}}},
             lambda o: V(o, "hi").startswith("PROVISIONAL (partial") and "lr axis OPEN at none@1e-3" in V(o, "hi"), supp="adbwx")
    scenario("W6 none@2e-3-4x with an informative seed > 0.10 -> no rescue (every none configuration that ran)",
             {**WOK, "none@2e-3-4x": {"acc": lambda s: FLAT(0.3) if s == 1 else H35}},
             lambda o: V(o, "hi").startswith("no rescue (core criteria hold") and "none@2e-3-4x not at the floor" in V(o, "hi"), supp="adbwx")
    scenario("W7 wd 0.3 not run yet (wd 1.0 only, the post-read state) -> PROVISIONAL", {EA: {"acc": GOOD}, HREMAP: HRM, **W1},
             lambda o: V(o, "hi").startswith("PROVISIONAL (partial") and "none@8e-3-wd0.3" in V(o, "hi") and "INCOMPLETE" in V(o, "hi"),
             supp="dbw" + "a", skip={("none@8e-3-wd0.3", s_) for s_ in SEL})
    print(f"selftest: all {count[0]} scenarios pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/yh6210/llmadd_torch/runs/ptheld2/out")
    ap.add_argument("--code_sha", default=None, help="expected code_sha (default: sha256[:12] of runs/ptheld2/code/tools/pretrain_range.py)")
    ap.add_argument("--selftest", default=None, metavar="RUN_JSON", help="a Rev D value_freq CPU run JSON as the template of the fakes")
    ap.add_argument("--textarith", default="/scratch/yh6210/llmadd_torch/runs/ptheld2/textarith.json",
                    help="per-number '+' / '=' context counts (tools/ptheld2_textarith.py); descriptive, skipped if absent")
    args = ap.parse_args()
    if Path(args.textarith).exists():
        TEXTARITH = json.loads(Path(args.textarith).read_text())["count"]
    if args.selftest:
        selftest(args.selftest)
        sys.exit(0)
    sha = args.code_sha or hashlib.sha256(Path("/scratch/yh6210/llmadd_torch/runs/ptheld2/code/tools/pretrain_range.py")
                                          .read_bytes()).hexdigest()[:12]
    _, lines = summarise(args.root, sha)
    print("\n".join(lines))
