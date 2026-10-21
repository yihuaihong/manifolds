"""Experiment B: a number prior that covers 0..999 and generalises beyond the trained range?

From-scratch pretraining as in pretrain_helix.py / pretrain_digits.py, but the arithmetic covers operands 0..999 with
one-, two- and three-digit operands equally often (pairs drawn on the fly), so 0-9 arithmetic and three-digit
arithmetic are trained, and four-digit operands (1000..9999) are never seen. Numbers are tokenised as whole tokens
(OLMo-2: 0..999 are single tokens, "1234" -> "123" + "4") or digit by digit (--tokenization digits; data from
digit_split_data.py).
Held out (never trained, either order): for every operand-length bucket (1+1, 1+2, 1+3, 2+2, 2+3, 3+3) a set of
unordered pairs; training pairs are drawn uniformly over buckets and rejected if held out.
Prior (--target; lam * (1 - CKA)) on the operands INSIDE the training lines (F69: best site), state at the operand's
last token, at --geo_layers; target Gram from the operand values in the batch:
  none
  helix2     periods 2, 5, 10, 100 + linear (x / 100): the original 10..99 helix, applied to 0..999
  helix3     periods 2, 5, 10, 100, 1000 + linear (x / 1000) (F18's helix3)
  digitcirc  one period-10 circle per decimal place (units, tens, hundreds) + log10(x + 1) + number of digits
             (one-hot): the base-10 positional code (Levy & Geva 2024); extends to any length by adding places
  digitcirc_shuf  digitcirc of permuted values (wrong geometry AND wrong number of digits)
  digitcirc_lenshuf / helix3_lenshuf  the target of values permuted WITHIN 0..9, 10..99, 100..999: keeps the
             number-of-digits grouping, destroys the per-place / helix geometry (audit 2026-10-13: ~80% of the
             digitcirc Gram is the digit-count grouping)
  length     number of digits (one-hot) + log10(x + 1) only
Evaluation (training position, answer + newline teacher-forced, exact match over all of them, so an answer that is
a prefix of the prediction does not count): held-out pairs per bucket (b11 = 0..9 + 0..9, b22, b33, b13, b23, ...),
"seen" pairs per bucket (a fixed rehearsal pool of 100 pairs per bucket that makes up 5% of the arithmetic lines,
so they are certainly trained), and never-trained lengths b44, b41, b42. The held-out split depends on the seed.
Not comparable across tokenisations: OOD answers (whole: "13525" = "135" + "25", answer chunks never trained),
per-token val loss (bits per byte is logged for that), and the arithmetic share of tokens (whole ~8.5%, digits ~11%).
At every eval: gradient norm of the LM loss and of the prior (before clipping) and their cosine.
Audit 2026-10-14 (pt999): the helix3 transfer read-out (cka_ops_helix3) at fixed layers (1/3, 2/3 of the depth) for every
arm, whatever --geo_layers; CKA of the number tokens' input and output embedding rows (0..999) to helix3 (with tied
embeddings the prior also moves the answer classifier rows; --untie is the control); held_b13/b23/b33 also split into
sums < 1000 (one answer token) and >= 1000 (first chunk "1xx"); the text site skips numbers glued to a preceding letter
("B1", "VO2") and logs how many sites fall in 0-9 / 10-99 / 100-999; evals and the gate's EMA on a token scale
(--bs 64: every 125 steps, EMA 0.98^2; bs 32 unchanged); LM loss in chunks when the full fp32 logits would not fit;
gradient checkpointing above 150M parameters; code_sha (sha256 of this file) in every result.
Rev B 2026-10-17 (rainbow6, opt-in --domains; see the block above main()): months / weekdays / clock / s4 / dim lines
added to every row (--extra_lines), per-domain splits, eval sets in the training position and priors on their item
tokens with one shared schedule. Without --domains the trainer is the pinned a3bce3ab63c1 (CPU regression: per-step
loss and gradient norm, curves, saved model and --dry_inputs output identical; the run JSON's args omit the Rev B flags).
Rev B amendment A2 2026-10-17 (user decisions; --domains only): the numbers line is always the first line after the text
(as in the 2% main setting; same pairs, formats and text offsets as the default path of the same seed, the text cut at
its start by the domain lines' length) and the domain lines follow it in random order; domain evals in that position
(text + one numbers training line + the question); held-out-FORMAT evals (3-4 paraphrase templates per domain, never in
a training line) on the same questions and contexts.
Rev C 2026-10-19 (wall 1, "data missing / long tail"; opt-in, intent spec llmadd_torch/runs/ptheld/AUDIT.md). Without
--held_operands, --site emb and --wd the trainer is the pinned one (default path bit-identical; the run JSON's args omit
the Rev C flags at their defaults).
  --held_operands N  a per-seed set H of N three-digit numbers (100..999), N / 5 from each quintile of their text-site
             count in train.bin (counted in-run with the text site's own rule), NEVER in a training arithmetic line:
             not as an operand and not as an answer token (sums >= 1000 are the tokens s // 10 + s % 10, so a held
             h in 100..199 also removes the sums 10h..10h+9; digits 0..9 are never held). Training pairs are redrawn
             within their bucket until clean, the rehearsal pool and the held / seen eval sets keep clean pairs only, and
             every training batch is content-scanned (assert: no H token in the lines). H still occurs in the text.
             New eval sets in the training position: H_one (one operand in H, the other seen, every answer token seen),
             H_both (both operands in H), H_ans (sum = h, or a 4-digit sum whose first token is h); per-row hits and
             greedy answers (two decode steps) for the shuffled-value diagnostic; per-number text exposure.
  --site emb  the prior on the embedding rows of the number tokens 0..999 (every one, H too; with tied embeddings
             these are also the output rows): 1 - CKA(W_E[0..999], target) on the weights, no hidden states.
  --wd       AdamW weight decay (default 0.1 = the pinned value), for the none arm's lr x wd sweep (lesson c).
  Rev C1 2026-10-19 (fresh audit of Rev C; --held_operands only):
  --target helix3_hremap  helix3 with ONLY H remapped: x not in H keeps helix3(x), h gets helix3(u_h), u_h a seen
             three-digit number with |u_h - h| >= 100, u_h != h (mod 10), all u_h distinct, and u_h + y token-clean for
             every H_one partner y of h (drawn on stream 6400 + seed; logged for every --held_operands run as remap_H).
             The mechanism test: a model that computes with the prior's geometry answers u_h + y for h + y.
  H sets also log a continuous score (the answer's teacher-forced log-prob, per row for H_one; the first answer
  token's log-prob and rank), a running sha256 of the training batches (identical across the arms of a seed) and the
  cosine of the embedding rows of h and u_h.
Rev D 2026-10-21 (ptheld2, wall 1 follow-up; opt-in, intent spec llmadd_torch/runs/ptheld2/AUDIT.md). Without --held_strat
value_freq and --lam_decay the trainer is Rev C1 (default path and every Rev C / C1 path bit-identical; the run JSON's
args omit the two flags at their defaults).
  --held_strat value_freq  (needs --held_operands N, N = 18 k; ptheld2: 54) H = 2 k numbers per hundreds block 100-199 ..
             900-999: inside a block the 100 numbers are ranked by their text-site count (ties by value, as the
             quintile strata), split into a lower ("rare") and upper ("common") half of 50, and k are drawn from each half
             on the held stream 6000 + seed (blocks ascending, rare then common). Every Rev C guarantee is unchanged (H
             never in a line, scanned every step, data sha, same H for every arm of a seed). Logged per h: block, half,
             count; H_one masks v1..v9 (block), rare, common, hi (blocks 7-9) and lo_rare (the rare halves = rare); for
             each of these layers the accuracy H_one_<layer> and the mean answer log-prob logp_H_one_<layer> at every eval
             (the same rows and forward as H_one; rows per layer in held_operands.layer_rows).
  --lam_decay  (needs --schedule window) inside the window lambda_t = lambda (1 - p), p = (s - s_on) / (s_off - s_on),
             s_on / s_off = the first step with the prior on / off (s_off = steps when the window ends with the run):
             lambda at the first on step, lambda / (s_off - s_on) at the last one.
  With either flag, every eval also logs the consumed lambda (lam_now; 0 when no prior term is in that step's loss), the
  pre-clip norm of the total gradient as clip_grad_norm_ saw it (grad_total), the clip coefficient min(1, 1 / (grad_total
  + 1e-6)) (clip_coef), and the LM share grad_lm / |g_lm + g_prior| (grad_lm_share; g_prior lambda_t-weighted; 1.0 when
  the loss is the LM alone) with |g_lm + g_prior| itself (grad_T; double path of grad_total).

    python tools/pretrain_range.py --data <bins> --tokenizer <olmo2> --out <dir> --tokenization digits --target digitcirc --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
import torch.utils.checkpoint

import numlib as nl
from digit_split_data import expand, split_table
from pretrain_helix import build_model, centred_gram, cka

FORMATS = ("Q: {a} + {b} = ", "{a} plus {b} equals ", "The sum of {a} and {b} is ", "Compute {a} + {b}. Result: ")
BUCKETS = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
RANGE = {1: (0, 10), 2: (10, 100), 3: (100, 1000), 4: (1000, 10000)}
# Rev C (wall 1, --held_operands): RNG streams (+ seed) 6000 the draw of H, 6100 the H eval pairs, 6200 their text ends
HELD_STRATA = 5       # H: N / 5 numbers from each quintile of the text-site count of 100..999 (ties by value)
HELD_ONE_PER = 12     # H_one rows per held number: partner of 1 / 2 / 3 digits x 2 draws x h first / h second
HELD_BOTH_N = 150     # H_both rows (two different held numbers, either order)
HELD_ANS_PER = 3      # H_ans rows per held number with sum h (+3 with a 4-digit sum whose first token is h, h < 200)
REVC_DEFAULTS = {"held_operands": 0, "wd": 0.1}   # Rev C flags: kept out of the run JSON's args at these values
HELD_BLOCKS = 9       # Rev D (--held_strat value_freq): the hundreds blocks 100-199 .. 900-999, two halves each
REVD_DEFAULTS = {"held_strat": "freq", "lam_decay": False}   # Rev D flags: kept out of the run JSON's args at these values


def features(target: str, x: np.ndarray, perm=None, perm_len=None) -> np.ndarray:
    x = np.asarray(x)
    if target.endswith("_lenshuf"):
        x = perm_len[x]; target = target[: -len("_lenshuf")]
    elif target.endswith("_farshuf"):   # Rev C: the caller passes the far permutation as perm (main: --held_operands)
        x = perm[x]; target = target[: -len("_farshuf")]
    elif target.endswith("_hremap"):   # Rev C1: the caller passes the H-only remap (identity off H) as perm
        x = perm[x]; target = target[: -len("_hremap")]
    elif target.endswith("_shuf"):
        x = perm[x]; target = target[: -len("_shuf")]
    if target == "length":
        nd = np.array([len(str(int(v))) for v in x])
        return np.concatenate([np.log10(x + 1.0)[:, None], np.eye(3)[np.minimum(nd, 3) - 1]], 1)
    if target == "helix2":
        return nl.features(x)[0]
    if target == "helix3":
        return np.concatenate([nl.features(x, periods=(2, 5, 10, 100, 1000), linear=False)[0], (x / 1000.0)[:, None]], 1)
    if target == "digitcirc":
        cols = []
        for place in (1, 10, 100):
            d = (x // place) % 10
            cols += [np.cos(2 * np.pi * d / 10), np.sin(2 * np.pi * d / 10)]
        nd = np.array([len(str(int(v))) for v in x])
        return np.concatenate([np.stack(cols, 1), np.log10(x + 1.0)[:, None], np.eye(3)[np.minimum(nd, 3) - 1]], 1)
    raise ValueError(target)


def gate_step(is_open, m, thr, hyst):
    """One step of the self-timed gate on the lowest per-bucket EMA m: an open gate stays open while m < thr; a closed
    gate reopens only when m < thr - hyst (hysteresis against chattering). hyst = 0 is the stateless rule m < thr."""
    return m < thr if is_open else m < thr - hyst


def held_out(rng, per_bucket=150):
    """Unordered held-out pairs per bucket (1+1: 20 of the 55 unordered pairs)."""
    held = {}
    for la, lb in BUCKETS:
        n = 20 if (la, lb) == (1, 1) else per_bucket
        s = set()
        while len(s) < n:
            a, b = int(rng.integers(*RANGE[la])), int(rng.integers(*RANGE[lb]))
            s.add((min(a, b), max(a, b)) if la == lb else (a, b))
        held[(la, lb)] = sorted(s)
    return held


# ==================================================================================================================
# Rev B (2026-10-17, rainbow6): several manifolds in one pretraining run. Opt-in only (--domains); without it none of
# the code below runs and the default path is the pinned trainer (sha256 a3bce3ab63c1) step for step.
#   numbers   the arithmetic lines above (--arith_lines; 2% main setting = 1), prior helix3 on the operands
#   months    "5 months after March is August"      circle Z12 on the start month   (k = 1..23, k != 12)
#   weekdays  "3 days after Monday is Thursday"     circle Z7 on the start day      (k = 1..13, k != 7)
#   clock     "9 hours after 14:00 is 23:00"        circle Z24 on the start hour    (k = 1..47, k != 24)
#             The hours 0..23 and the shifts are the SAME token ids as the numbers' operands 0..999: the numbers
#             prior (helix3, arithmetic lines only) and the clock prior (C24, clock lines only) act on the same rows of
#             the tied embedding in different contexts. A clock line without wrap (h + k < 24) states the addition fact
#             h + k: training clock lines whose unordered pair is a held-out numbers pair are dropped (no leak into the
#             numbers held-out evals), and the clock held-out evals are split into wrap / no-wrap (trained addition
#             pair) / no-wrap (held-out addition pair).
#   s4        "lij then mip is buc"                 the 24 cube rotations (toy_struct.cube_group), c = "a then b"
#             (R_c = R_b R_a); prior R (faithful 3-d matrix rep, 9-d) on a and b
#   dim       "zaj times pij is kep" / "kep per pij is zaj"   exponent vectors {-1,0,1}^3 (toy_struct dim task):
#             times = e_a + e_b, per = e_a - e_b, only inside the box; prior expo (3-d) on the two inputs
# Row layout (amendment A2, 2026-10-17, user: numbers results must compare directly with the 2% main setting): text,
# then the numbers line(s) (--arith_lines), then the --extra_lines domain lines in random order (stream 5000 + seed).
# The numbers line is ALWAYS the first line after the text, as in the default path, and (causal attention) sees what it
# sees there: the default path's pair and format (streams 1000 / 2000, consumed identically) after the END of the
# default path's text chunk (the same offset draw; the first len(domain lines) tokens of that chunk are dropped). The
# only differences for the numbers line: its absolute position is earlier by the domain lines' length, the text before
# it is shorter by as much, and the row goes on after it (domain lines, which it cannot see). The numbers eval sets are
# the default path's (built before this block). A domain line sits right after the numbers line (slot 1) or after
# another domain line (slot 2 for K = 2); the slot shares are logged (slot_frac). Domain evals (every set) place the
# question in slot 1: text + one numbers line (a TRAINING pair and format, drawn like draw_pair from the domain's
# eval-context stream; for clock never the question's own addition i + k / i + k mod 24) + the question at the row end.
# Held-out-format evals (A2, user): PARA below, 3-4 paraphrase templates per domain used in NO training line (asserted),
# scored on the same questions and the SAME contexts (text end and numbers line) as the training template, same scoring.
# Held-out splits (per seed): cyclic = F80 / F81 class split (fraction --cyc_held of the (item, k mod n) classes, every
# k of a held class held out = tier S, the primary) + tier E (of the trained classes, fraction --cyc_heldE have one of
# their two k held out: the equivalent shift k +- n is trained for the same item); s4 = question split, train fraction
# --s4_train (F84); dim = relation-class split, train fraction --dim_train, with the negation-twin bookkeeping (F84).
# Read-out sets (DINFO readout / readoutE, audit 2026-10-17): months / weekdays tier S (E); clock tier S (E) with wrap;
# s4 held, fresh (commuting twin not trained) and no copy cell (no identity input); dim held, no trained negation twin
# and no copy cell (not a relation class {x, 0}); chance on the read-out set itself. Every eval also stores the per-
# question hit counts of each domain eval set (hits_<set>; questions / bootstrap units / masks in DINFO eval_<set>).
# RNG streams (registry, + seed): 1000 pairs, 2000 rows, 3000 prefix, 4000 share (Rev A), 4100 domains (one
# SeedSequence; spawned children: 0 = training domain lines, 1-5 = the split of EXTRA[i], 6-10 = the shuffled target of
# EXTRA[i], 11-15 = the eval contexts of EXTRA[i]: seen subset, then per set and context the numbers line and the text
# end), 5000 line order (of the domain lines after the numbers line), 10**6 + step logging.
EXTRA =("months", "weekdays", "clock", "s4", "dim")   # fixed order: the RNG children are indexed by it
ALL_DOMAINS = ("numbers",) + EXTRA
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
CYCLIC = {"months": 12, "weekdays": 7, "clock": 24}
UNIT = {"months": ("month", "months"), "weekdays": ("day", "days"), "clock": ("hour", "hours")}
# nonce items: OLMo-2 single tokens " xyz" (consonant-vowel-consonant, lowercase), not numbers, chosen 2026-10-17 as the
# rarest such tokens in the 4e8-token FineWeb-Edu train bin after removing words / abbreviations / names / vulgar ones,
# assigned to the two domains by a fixed permutation; value = occurrences in train.bin (re-counted by --dry_inputs).
S4_NAMES = {"lij": 4, "mip": 4, "buc": 7, "cuk": 8, "pob": 13, "poj": 13, "cif": 14, "gev": 16, "gez": 18, "tej": 20,
            "geg": 23, "nok": 26, "ler": 27, "sik": 27, "bek": 28, "zab": 32, "kes": 42, "rin": 46, "kem": 48, "dap": 57,
            "fot": 68, "rez": 78, "vad": 80, "diz": 85}
DIM_NAMES = {"zaj": 2, "pij": 5, "kep": 9, "zun": 12, "jel": 15, "fod": 16, "jed": 26, "mav": 29, "rej": 29, "tav": 29,
             "tud": 29, "poz": 30, "ked": 34, "roz": 39, "zar": 41, "vak": 42, "vog": 43, "nen": 44, "dob": 48, "mec": 50,
             "kob": 60, "geb": 68, "gef": 68, "tup": 68, "zag": 72, "baj": 78, "fas": 82}
DOM_ARGS = ("domains", "extra_lines", "prior_domains", "prior_shuf", "lam_rule", "cyc_held", "cyc_heldE", "s4_train",
            "dim_train", "dom_eval_rows", "dom_seen")   # Rev B flags: kept out of the run JSON's args when --domains is off
# Held-out-FORMAT templates (amendment A2, 2026-10-17, user): EVALUATION ONLY. Asserted at build time: no training line
# (every domain's training pool, the numbers formats) is an instance of any template (slots as wildcards) and no rendered
# paraphrase prompt occurs inside a training line; --dry_inputs also scans the training rows for them. Slots: {k} the
# shift, {u} its unit (singular for k = 1, as in training), {I} the start item, {A} / {B} the two inputs (s4 / dim; dim:
# (times, per) template pairs), {J} / {C} the answer. Items render as in the training lines (" March", " gez"; clock
# hours without a space), so the items and the scored span (answer through the newline; clock "{J}:00\n") are the
# training line's tokens (asserted per question). Eval keys <d>_p<t>_<set> (t = 1.., set held / seen).
PARA = {"months": ("What month is {k} {u} after{I}?{J}\n", "Date math:{I} + {k} {u} ={J}\n",
                   "Starting from{I}, count forward {k} {u}:{J}\n", "If it is{I} now, in {k} {u} it will be{J}\n"),
        "weekdays": ("What day is {k} {u} after{I}?{J}\n", "Day math:{I} + {k} {u} ={J}\n",
                     "Starting from{I}, count forward {k} {u}:{J}\n", "If it is{I} today, in {k} {u} it will be{J}\n"),
        "clock": ("{k} {u} after {I}:00 the time is {J}:00\n", "Time math: {I}:00 + {k} {u} = {J}:00\n",
                  "Starting from {I}:00, count forward {k} {u}: {J}:00\n", "If it is {I}:00 now, in {k} {u} it will be {J}:00\n"),
        "s4": ("{A} followed by{B} gives{C}\n", "Apply{A} and then{B} to get{C}\n", "Combining{A} with{B} yields{C}\n"),
        "dim": (("{A} multiplied by{B} gives{C}\n", "{A} divided by{B} gives{C}\n"),
                ("Multiply{A} by{B}:{C}\n", "Divide{A} by{B}:{C}\n"),
                ("{A} x{B} ={C}\n", "{A} /{B} ={C}\n"))}
PARA_SETS = ("held", "seen")   # the eval sets every template is scored on (the read-out sub-splits are masks of held)


def circle(n: int) -> np.ndarray:
    a = 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(a), np.sin(a)], 1)


def gram_std(F: np.ndarray) -> np.ndarray:
    """Centred Gram of column-standardised features (constant columns dropped): the prior's convention
    (pretrain_helix.centred_gram), in numpy."""
    F = np.asarray(F, np.float64)
    F = F[:, F.std(0) > 1e-9]
    F = (F - F.mean(0)) / F.std(0)
    F = F - F.mean(0)
    return F @ F.T


def cka_np(A: np.ndarray, B: np.ndarray) -> float:
    G, K = gram_std(A), gram_std(B)
    return float((G * K).sum() / (np.linalg.norm(G) * np.linalg.norm(K)))


def cka_cov(A: np.ndarray, B: np.ndarray) -> float:
    """cka_np through feature covariances (<G_A, G_B> = |A'B|^2 for centred columns): the same value without the n x n
    Grams (Rev C: the far-permutation search over 5,000 text-site values)."""
    st = lambda F: (lambda F: (F - F.mean(0)) / F.std(0))(np.asarray(F, np.float64)[:, np.asarray(F).std(0) > 1e-9])
    A, B = st(A), st(B)
    A, B = A - A.mean(0), B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def far_perm(F: np.ndarray, rng, refs, n_try: int = 3000, thr: float = 0.15):
    """(max CKA to the references, permutation q): the first q with max_ref CKA(F[q], ref) < thr, else the farthest of
    n_try (toy_struct.far_perm / pretrain_rainbow.wrong, on the domain's own stream)."""
    Gs = [gram_std(R) for R in refs]
    best = (2.0, None)
    for _ in range(n_try):
        q = rng.permutation(len(F))
        G1 = gram_std(F[q])
        c = max(float((G0 * G1).sum() / (np.linalg.norm(G0) * np.linalg.norm(G1))) for G0 in Gs)
        if c < best[0]:
            best = (c, q)
        if c < thr:
            break
    return best


def cyclic_split(n: int, rng, frac_s: float, frac_e: float, forbid=frozenset(), tries: int = 2000):
    """Questions (i, k), k in 1..2n-1, k != n; class (i, k mod n). Tier S: round(frac_s * #classes) classes held out
    with both their k; tier E: of the trained classes with no forbidden question (forbid = questions that may not be
    trained, e.g. clock lines stating a held-out addition pair), round(frac_e * #) have ONE of their two k (drawn) held
    out. Redrawn until every non-zero shift is trained on >= 2 items and every item is a trained start (as F80).
    Forbidden questions of trained classes are dropped from training (their class stays trained through the other k)."""
    classes = [(i, e) for i in range(n) for e in range(1, n)]
    for _ in range(tries):
        p = rng.permutation(len(classes))
        S = {classes[j] for j in p[:int(round(frac_s * len(classes)))]}
        trc = [c for c in classes if c not in S]
        if all(sum(1 for _, e in trc if e == e0) >= 2 for e0 in range(1, n)) and {i for i, _ in trc} == set(range(n)):
            break
    else:
        raise RuntimeError("no valid cyclic split")
    ok_e = [(i, e) for i, e in trc if (i, e) not in forbid and (i, e + n) not in forbid]
    q = rng.permutation(len(ok_e))
    E = {ok_e[j]: int(rng.integers(2)) for j in q[:int(round(frac_e * len(ok_e)))]}   # 0: k = e held, 1: k = e + n held
    held_s, held_e, train = [], [], []
    for i in range(n):
        for k in list(range(1, n)) + list(range(n + 1, 2 * n)):
            c = (i, k % n)
            if c in S:
                held_s.append((i, k))
            elif c in E and (k >= n) == bool(E[c]):
                held_e.append((i, k))
            elif (i, k) not in forbid:
                train.append((i, k))
    return train, held_s, held_e, sorted(S)


def cyclic_closure(n: int, train_classes: set, held: list) -> dict:
    """Share of held-out (i, k) whose answer follows from trained facts alone: by composing two trained shifts from i
    (i -> j -> answer, e1 + e2 = k mod n), by any chain of trained shifts, and through the inverse relation (the answer's
    class (j, -k mod n) trained: 'j + (n - e) = i')."""
    reach = {}
    for i in range(n):   # states (item, accumulated shift mod n) reachable from (i, 0) along trained classes
        seen, todo = {(i, 0)}, [(i, 0)]
        while todo:
            v, acc = todo.pop()
            for e in range(1, n):
                if (v, e) in train_classes:
                    w = ((v + e) % n, (acc + e) % n)
                    if w not in seen:
                        seen.add(w); todo.append(w)
        reach[i] = seen
    two = lambda i, e: any((i, e1) in train_classes and ((i + e1) % n, (e - e1) % n) in train_classes
                           for e1 in range(1, n) if (e - e1) % n)
    return {"closure_2hop": float(np.mean([two(i, k % n) for i, k in held])) if held else None,
            "closure_chain": float(np.mean([(((i + k) % n), k % n) in reach[i] for i, k in held])) if held else None,
            "inverse_trained": float(np.mean([(((i + k) % n), (-k) % n) in train_classes for i, k in held])) if held else None}


def curve_metrics(curve: list, key, per_step_tokens: int) -> dict:
    """area = 100 x mean over all evals; final = the last eval (x100) and the mean of the last 3; t50 / t90 = M tokens
    until the value first reaches 0.5 / 0.9 (linear interpolation between evals, None if never): readout_rules'
    definitions, for one series key(eval) in [0, 1] (None evals skipped)."""
    pts = [(e["step"] * per_step_tokens, key(e)) for e in curve]
    pts = [(t, v) for t, v in pts if v is not None]
    if not pts:
        return {}

    def t_th(th):
        prev = None
        for tk, v in pts:
            if v >= th:
                return tk / 1e6 if prev is None else (prev[0] + (th - prev[1]) / (v - prev[1]) * (tk - prev[0])) / 1e6
            prev = (tk, v)
        return None
    return {"area": 100 * float(np.mean([v for _, v in pts])), "final": 100 * pts[-1][1],
            "final3": 100 * float(np.mean([v for _, v in pts[-3:]])), "t50": t_th(0.5), "t90": t_th(0.9), "n_evals": len(pts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenization", choices=("whole", "digits"), required=True)
    ap.add_argument("--target", choices=("none", "helix2", "helix3", "helix3_shuf", "digitcirc", "digitcirc_shuf",
                                         "digitcirc_lenshuf", "helix3_lenshuf", "length", "helix3_farshuf", "helix3_hremap"),
                    required=True,
                    help="helix3_farshuf (Rev C, needs --held_operands): helix3 of a FAR permutation of 0..999 (the best of 3000 "
                         "draws, stream 6300 + seed, by max(CKA to helix3 over 0..999, over the text site's value mix)); "
                         "helix3_hremap (Rev C1, needs --held_operands): helix3 with only H remapped, h -> helix3(u_h)")
    ap.add_argument("--site", choices=("ctx", "text", "prefix", "emb"), default="ctx",
                    help="ctx = the operands inside the training lines; text = numbers 0..999 that occur NATURALLY in the "
                         "FineWeb text of each row (never the arithmetic lines; whole-number tokens only), 2026-10-14; "
                         "prefix = the last token of separate 'Q: x' prompts at position 0 (F69 / F71's prefix site), x "
                         "drawn like the operands (lengths 1-3 equally often) from its own random stream; emb (Rev C) = "
                         "the embedding rows of the number tokens 0..999 (all of them; tied: also the output rows)")
    ap.add_argument("--held_operands", type=int, default=0,
                    help="Rev C: N three-digit numbers (per seed, N / 5 per text-count quintile) never in a training line, "
                         "as operand or answer token; 0 = off (the pinned trainer)")
    ap.add_argument("--wd", type=float, default=0.1, help="Rev C: AdamW weight decay (0.1 = the pinned value)")
    ap.add_argument("--held_strat", choices=("freq", "value_freq"), default="freq",
                    help="Rev D (needs --held_operands): freq = N / 5 per quintile of the text-site count (Rev C); value_freq "
                         "= N / 9 per hundreds block, half from its rarer and half from its commoner 50 numbers (N = 18 k)")
    ap.add_argument("--lam_decay", action="store_true",
                    help="Rev D (needs --schedule window): lambda_t = lambda (1 - p), p = linear progress through the window")
    ap.add_argument("--n_prefix", type=int, default=256, help="--site prefix: 'Q: x' prompts per step")
    ap.add_argument("--text_min", type=int, default=16, help="--site text: skip the prior on batches with fewer numbers")
    ap.add_argument("--ckpt_every", type=int, default=0, help="resumable checkpoint every N steps (0 = off)")
    ap.add_argument("--stop_after", type=int, default=-1, help="testing only: exit after this step (simulated preemption)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tokens", type=float, default=3e8)
    ap.add_argument("--arith_lines", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--schedule", choices=("whole", "window", "gate"), default="whole",
                    help="when the prior is on: whole run; --window w0,w1 (fractions of training); or self-timed gate = on "
                         "while the LOWEST per-bucket EMA of in-batch exact match (answer + newline) on the first arithmetic "
                         "line of each row (the line right after the text, as in the evals) is below --gate_thr")
    ap.add_argument("--window", default="0.05,0.4")
    ap.add_argument("--gate_thr", type=float, default=0.5)
    ap.add_argument("--gate_start", type=float, default=0.0,
                    help="--schedule gate: no prior before this fraction of training (same boundary as a window's start); "
                         "2026-10-14: delayed gate vs window")
    ap.add_argument("--gate_hyst", type=float, default=0.0,
                    help="--schedule gate: once closed, reopen only when the lowest bucket EMA < gate_thr - gate_hyst")
    ap.add_argument("--geo_layers", default=None)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--n_eval", type=int, default=150, help="held-out items per bucket in the eval sets")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--dry_inputs", action="store_true")
    ap.add_argument("--grad_ckpt", choices=("auto", "on", "off"), default="auto",
                    help="gradient checkpointing of the transformer blocks (same math, less memory, ~1/3 slower); auto = on "
                         "above 150M parameters (2026-10-14: the 505M model ran out of memory on a 48 GB L40S at bs 32)")
    ap.add_argument("--lm_chunk_tokens", type=int, default=-1,
                    help="LM loss over chunks of this many tokens, logits recomputed in backward (0 = off: one full logits "
                         "tensor as before); -1 = auto: 8192 when the fp32 logits would exceed 8 GB (audit 2026-10-14: bs 64 "
                         "x 512 needs ~45 GB for them on a 48 GB L40S)")
    ap.add_argument("--untie", action="store_true",
                    help="separate input and output embeddings (control: with tied embeddings the prior on the number "
                         "tokens also moves the answer classifier rows of the same tokens)")
    # Rev B (rainbow6): opt-in multi-domain pretraining; nothing below is used unless --domains is given
    ap.add_argument("--domains", default="", help="comma list of " + ",".join(ALL_DOMAINS) + " (must include numbers); "
                    "the other domains add --extra_lines lines per row (cycled over the domains)")
    ap.add_argument("--extra_lines", type=int, default=0, help="--domains: extra domain lines per row")
    ap.add_argument("--prior_domains", default="", help="--domains: domains whose prior is on (numbers iff --target is "
                    "helix3 / helix3_shuf); one shared schedule (--schedule window / whole)")
    ap.add_argument("--prior_shuf", default="", help="--domains: domains whose prior uses the shuffled target (numbers "
                    "iff --target helix3_shuf; the others a far permutation of the items, CKA to the true target < 0.15)")
    ap.add_argument("--lam_rule", choices=("norm", "each"), default="norm",
                    help="--domains: norm = every active prior gets lam / (number of active priors), so the total prior "
                         "weight is lam whatever the arm (F70 '@norm'); each = every active prior gets lam")
    ap.add_argument("--cyc_held", type=float, default=0.4, help="--domains: held-out share of (item, k mod n) classes")
    ap.add_argument("--cyc_heldE", type=float, default=0.5, help="--domains: share of the trained classes with one k held")
    ap.add_argument("--s4_train", type=float, default=0.6, help="--domains: s4 question split, train fraction")
    ap.add_argument("--dim_train", type=float, default=0.5, help="--domains: dim relation-class split, train fraction")
    ap.add_argument("--dom_eval_rows", type=int, default=200, help="--domains: min rows per eval set (contexts x questions)")
    ap.add_argument("--dom_seen", type=int, default=150, help="--domains: trained questions per seen eval set")
    args = ap.parse_args()
    DOM = bool(args.domains)
    lst = lambda s: [v for v in s.split(",") if v]
    DOMS, DPRI_ALL, DSHUF = lst(args.domains), lst(args.prior_domains), lst(args.prior_shuf)
    if not DOM:
        assert not (args.extra_lines or DPRI_ALL or DSHUF), "--extra_lines / --prior_domains / --prior_shuf need --domains"
    else:
        assert len(set(DOMS)) == len(DOMS) and set(DOMS) <= set(ALL_DOMAINS) and "numbers" in DOMS, DOMS
        assert set(DPRI_ALL) <= set(DOMS) and set(DSHUF) <= set(DPRI_ALL) and len(set(DPRI_ALL)) == len(DPRI_ALL), (DPRI_ALL, DSHUF)
        assert ("numbers" in DPRI_ALL) == (args.target != "none"), "numbers in --prior_domains iff --target != none"
        assert args.target in ("none", "helix3", "helix3_shuf"), "--domains: numbers target none / helix3 / helix3_shuf"
        assert ("numbers" in DSHUF) == (args.target == "helix3_shuf"), "numbers in --prior_shuf iff --target helix3_shuf"
        assert (args.extra_lines >= 1) == (len(DOMS) > 1), "--extra_lines >= 1 iff a domain besides numbers"
        assert args.site == "ctx" and args.tokenization == "whole", "--domains: ctx site and whole-number tokens only"
        assert args.schedule in ("window", "whole"), "--domains: one shared window / whole schedule (the gate EMA is numbers-only)"
    HO = args.held_operands > 0   # Rev C (wall 1)
    REVC = HO or args.site == "emb" or args.wd != REVC_DEFAULTS["wd"]   # Rev C logging (lr_now / wd_now) only then
    VF = args.held_strat == "value_freq"   # Rev D
    REVD = VF or args.lam_decay            # Rev D logging (lam_now, grad_total, clip_coef, grad_lm_share) only then
    if HO:
        assert args.tokenization == "whole" and not DOM, "--held_operands: whole-number tokens, no --domains"
        if not VF:
            assert args.held_operands % HELD_STRATA == 0 and args.held_operands <= 300, args.held_operands
        else:   # k per half per block, k <= 25 (a half has 50 numbers)
            assert args.held_operands % (2 * HELD_BLOCKS) == 0 and args.held_operands <= 2 * HELD_BLOCKS * 25, args.held_operands
    assert HO or not VF, "--held_strat value_freq needs --held_operands"
    assert not args.lam_decay or (args.schedule == "window" and args.target != "none" and not DOM), \
        "--lam_decay: a window prior on the numbers (no --domains)"
    if args.site == "emb":
        assert args.tokenization == "whole" and not DOM, "--site emb: whole-number tokens (0..999 single tokens), no --domains"
    assert HO or args.target != "helix3_farshuf", "--target helix3_farshuf needs --held_operands (its search uses the text counts)"
    assert HO or args.target != "helix3_hremap", "--target helix3_hremap needs --held_operands (it remaps H)"
    assert args.wd >= 0, args.wd
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"run_{args.tokenization}_{args.target}_seed{args.seed}{args.tag}.json"
    if not args.dry_inputs:   # the same run may be queued twice (two accounts / partitions): see run_guard.py
        import run_guard
        if not run_guard.claim(path):
            return
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train = np.memmap(Path(args.data) / "train.bin", dtype=np.uint32, mode="r")
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint32, mode="r")
    meta = json.loads((Path(args.data) / "meta.json").read_text()) if (Path(args.data) / "meta.json").exists() else {}
    digits = args.tokenization == "digits"
    assert digits == ("digit_tokens" in meta), "data and --tokenization must match"
    table = split_table(tok)
    raw = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    enc = (lambda s: expand(np.array(raw(s)), table).tolist()) if digits else raw

    # ---------------------------------------------------------------- pieces of a line, so operand positions are known
    pieces = []
    for f in FORMATS:
        pre, rest = f.split("{a}"); mid, post = rest.split("{b}")
        pieces.append((enc(pre) if pre else [], enc(mid), enc(post)))

    def line(a, b, f, with_answer=True):
        """Token ids of a line and the index of the LAST token of a and of b (and the answer span)."""
        pre, mid, post = pieces[f]
        A, B = enc(str(a)), enc(str(b))
        ids = pre + A + mid + B + post
        ia, ib = len(pre) + len(A) - 1, len(pre) + len(A) + len(mid) + len(B) - 1
        S = enc(str(a + b)) if with_answer else []
        return ids + S + (enc("\n") if with_answer else []), ia, ib, len(S)

    rng_split = np.random.default_rng(args.seed)   # the held-out split is drawn per seed
    held = held_out(rng_split)
    held_set = {(min(a, b), max(a, b)) for v in held.values() for a, b in v}
    rehearsal = {}   # fixed pairs per bucket that make up 5% of the arithmetic lines: certainly trained
    for la, lb in BUCKETS:
        prs = set()
        while len(prs) < min(100, 25 if (la, lb) == (1, 1) else 100):
            a, b = int(rng_split.integers(*RANGE[la])), int(rng_split.integers(*RANGE[lb]))
            if (min(a, b), max(a, b)) not in held_set:
                prs.add((a, b))
        rehearsal[(la, lb)] = sorted(prs)
    rehearsal_all = [q for v in rehearsal.values() for q in v]
    for _ in range(200):   # piecewise tokenisation = whole-string tokenisation
        a, b, f = int(rng_split.integers(0, 1000)), int(rng_split.integers(0, 1000)), int(rng_split.integers(4))
        ids, ia, ib, ns = line(a, b, f)
        assert ids == enc(FORMATS[f].format(a=a, b=b) + f"{a + b}\n"), (a, b, f)
        assert tok.decode(enc(str(a))[-1:]) == tok.decode([ids[ia]]) and tok.decode(enc(str(b))[-1:]) == tok.decode([ids[ib]])
    data_rng = np.random.default_rng(1000 + args.seed)

    def draw_pair():
        if data_rng.random() < 0.05:
            return rehearsal_all[int(data_rng.integers(len(rehearsal_all)))]
        while True:
            la, lb = BUCKETS[int(data_rng.integers(len(BUCKETS)))]
            if data_rng.random() < 0.5:
                la, lb = lb, la
            a, b = int(data_rng.integers(*RANGE[la])), int(data_rng.integers(*RANGE[lb]))
            if (min(a, b), max(a, b)) not in held_set:
                return a, b

    # ---------------------------------------------------------------- eval sets (training position, teacher forcing)
    vr = np.random.default_rng(11)
    def evalset(pairs, fmt=0, g=None):   # g: Rev C's own text-offset stream (default: vr, as pinned)
        rows, lens = [], []
        for a, b in pairs:
            ids, _, _, ns = line(a, b, fmt)   # scored span = answer tokens + the newline (seq - 1, as in training)
            n = args.seq - len(ids)
            o = int((vr if g is None else g).integers(0, len(val) - n))
            rows.append(np.concatenate([np.asarray(val[o:o + n], dtype=np.int64), ids]))
            lens.append(ns + 1)
        return torch.tensor(np.stack(rows), device=dev), torch.tensor(lens, device=dev), np.array([a + b for a, b in pairs])
    er = np.random.default_rng(5)
    evalsets = {}
    for (la, lb), prs in held.items():
        both = prs + [(b, a) for a, b in prs if a != b]
        sel = [both[i] for i in er.permutation(len(both))[:args.n_eval]]
        evalsets[f"held_b{la}{lb}"] = evalset(sel)
    for (la, lb), prs in rehearsal.items():
        evalsets[f"seen_b{la}{lb}"] = evalset(prs)
    for la, lb in ((4, 4), (4, 1), (4, 2)):   # never-trained lengths
        evalsets[f"ood_b{la}{lb}"] = evalset([(int(er.integers(*RANGE[la])), int(er.integers(*RANGE[lb]))) for _ in range(args.n_eval)])
    max_ans = max(int(v[1].max()) for v in evalsets.values())

    # ---------------------------------------------------------------- Rev B: domain lines, splits, targets, eval sets
    EXT = [d for d in EXTRA if d in DOMS]       # the enabled non-number domains, in the fixed order
    DPRI = [d for d in EXT if d in DPRI_ALL]    # their active priors
    lam_num = args.lam                          # the numbers prior's weight (default path: --lam)
    DLINE, DPOOL, DINFO, DTAB, DTAB_ACT, DHELD, dom_evalsets = {}, {}, {}, {}, {}, {}, {}
    if DOM:
        import toy_struct as ts
        dom_ss = np.random.SeedSequence(4100 + args.seed).spawn(16)
        dom_rng = np.random.default_rng(dom_ss[0])          # training domain lines
        order_rng = np.random.default_rng(5000 + args.seed)  # line order within a row
        lam_each = args.lam / len(DPRI_ALL) if (args.lam_rule == "norm" and DPRI_ALL) else args.lam
        LAMD = {d: (lam_each if d in DPRI_ALL else 0.0) for d in DOMS}
        if "numbers" in DPRI_ALL:
            lam_num = LAMD["numbers"]
        for nm in list(S4_NAMES) + list(DIM_NAMES):
            assert nm.isascii() and nm.isalpha() and nm.islower() and len(enc(" " + nm)) == 1, nm
        assert not (set(S4_NAMES) & set(DIM_NAMES)) and len(S4_NAMES) == 24 and len(DIM_NAMES) == 27
        S4N, DIMN = list(S4_NAMES), list(DIM_NAMES)

        def pieces_line(pieces):
            """pieces [(text, role)], role None / ('site', item) / 'ans' -> (ids, [(pos, item)], answer start, answer
            span length = answer tokens through the newline). Asserts single-token items and that the pieces tokenise
            like the whole line."""
            ids, sites, a0 = [], [], None
            for text, role in pieces:
                t = enc(text)
                if role == "ans":
                    a0 = len(ids)
                elif role is not None:
                    assert len(t) == 1, (text, t)
                    sites.append((len(ids), role[1]))
                ids += t
            assert ids == enc("".join(p for p, _ in pieces)), pieces
            return ids, sites, a0, len(ids) - a0

        def dom_q(d, q):
            """The line of question q of domain d, its answer string and its input items (for the decode asserts)."""
            if d in CYCLIC:
                n = CYCLIC[d]; i, k = q; j = (i + k) % n; u = UNIT[d][int(k != 1)]
                if d == "clock":
                    return pieces_line([(f"{k} {u} after ", None), (f"{i}", ("site", i)), (":00 is ", None), (f"{j}", "ans"),
                                        (":00\n", None)]), f"{j}:00", [str(i)]
                nm = MONTHS if d == "months" else WEEKDAYS
                return pieces_line([(f"{k} {u} after", None), (f" {nm[i]}", ("site", i)), (" is", None), (f" {nm[j]}", "ans"),
                                    ("\n", None)]), nm[j], [nm[i]]
            Dd = DTASK[d]; x = Dd["X"][q]; c = int(Dd["Y"][q])
            if d == "s4":
                a, b = int(x[0]), int(x[1])
                return pieces_line([(f" {S4N[a]}", ("site", a)), (" then", None), (f" {S4N[b]}", ("site", b)), (" is", None),
                                    (f" {S4N[c]}", "ans"), ("\n", None)]), S4N[c], [S4N[a], S4N[b]]
            a, op, b = int(x[0]), int(x[1]), int(x[2])
            return pieces_line([(f" {DIMN[a]}", ("site", a)), (" times" if op == 27 else " per", None),
                                (f" {DIMN[b]}", ("site", b)), (" is", None), (f" {DIMN[c]}", "ans"), ("\n", None)]), DIMN[c], [DIMN[a], DIMN[b]]

        PLINE, DCTX = {}, {}   # A2: held-out-format lines (d, template, q); text length of every eval row (dry display)

        def para_tpl(d, t, q):
            """Template t of domain d for question q (dim: the times / per member of the pair)."""
            return PARA[d][t] if d != "dim" else PARA[d][t][int(DTASK[d]["X"][q][1] == 28)]

        def para_q(d, t, q):
            """A2: the line of question q of domain d in held-out-format template t (PARA): (ids, sites, answer start,
            answer span length), as pieces_line; the slot texts are those of dom_q, so items and answer are the
            training line's tokens."""
            if d in CYCLIC:
                n = CYCLIC[d]; i, k = q; j = (i + k) % n
                nm = (lambda v: f"{v}") if d == "clock" else (lambda v: f" {(MONTHS if d == 'months' else WEEKDAYS)[v]}")
                slot = {"k": (str(k), None), "u": (UNIT[d][int(k != 1)], None), "I": (nm(i), ("site", i)), "J": (nm(j), "ans")}
            else:
                x = DTASK[d]["X"][q]; c = int(DTASK[d]["Y"][q]); N = S4N if d == "s4" else DIMN
                a, b = (int(x[0]), int(x[1])) if d == "s4" else (int(x[0]), int(x[2]))
                slot = {"A": (f" {N[a]}", ("site", a)), "B": (f" {N[b]}", ("site", b)), "C": (f" {N[c]}", "ans")}
            pcs = []
            for i_, p in enumerate(re.split(r"\{(\w)\}", para_tpl(d, t, q))):
                txt, role = slot[p] if i_ % 2 else (p, None)
                if not txt:
                    continue
                if role is None and pcs and pcs[-1][1] is None:   # plain text is tokenised as one piece (" 5 months")
                    pcs[-1] = (pcs[-1][0] + txt, None)
                else:
                    pcs.append((txt, role))
            return pieces_line(pcs)

        def ctx_pair(g, d, q):
            """A2: the numbers line of a domain eval context: a TRAINING pair and format, drawn as draw_pair does (5%
            rehearsal pool, else a bucket uniformly, both orders, never a held-out pair) from the domain's eval-context
            stream g; for clock a pair stating the question's own addition (i + k, i + k mod 24) is redrawn. -> (a, b,
            format, number of redraws)"""
            bad = {(min(q[0], q[1]), max(q[0], q[1])), (min(q[0], q[1] % 24), max(q[0], q[1] % 24))} if d == "clock" else set()
            nr = 0
            while True:
                if g.random() < 0.05:
                    a, b = rehearsal_all[int(g.integers(len(rehearsal_all)))]
                else:
                    while True:
                        la, lb = BUCKETS[int(g.integers(len(BUCKETS)))]
                        if g.random() < 0.5:
                            la, lb = lb, la
                        a, b = int(g.integers(*RANGE[la])), int(g.integers(*RANGE[lb]))
                        if (min(a, b), max(a, b)) not in held_set:
                            break
                f = int(g.integers(len(FORMATS)))
                if (min(a, b), max(a, b)) not in bad:
                    return a, b, f, nr
                nr += 1

        DTASK ={d: ts.build_task(d) for d in ("s4", "dim") if d in EXT}
        for d in EXT:
            ix = EXTRA.index(d)
            sr = np.random.default_rng(dom_ss[1 + ix])
            info, masks = {}, {}
            if d in CYCLIC:
                n = CYCLIC[d]
                forbid = frozenset()   # no-wrap clock lines state h + k: never train a held-out numbers pair this way
                if d == "clock":
                    forbid = frozenset((i, k) for i in range(24) for k in range(1, 24) if i + k < 24 and (min(i, k), max(i, k)) in held_set)
                train_q, held_s, held_e, S = cyclic_split(n, sr, args.cyc_held, args.cyc_heldE, forbid)
                excl = sorted(set(forbid) - {(i, k) for i, k in held_s})   # forbidden questions of trained classes
                assert not any(i + k < 24 and (min(i, k), max(i, k)) in held_set for i, k in train_q) or d != "clock"
                trc = {(i, k % n) for i, k in train_q}
                assert not (trc & set(S)), "tier-S class trained"
                assert all((i, k) not in set(train_q) and (i, k % n) in trc for i, k in held_e), "tier E: held, class trained"
                assert all(sum(1 for _, e in trc if e == e0) >= 2 for e0 in range(1, n)), "a shift trained on < 2 items"
                assert {i for i, _ in train_q} == set(range(n)) and {(i + k) % n for i, k in train_q} == set(range(n))
                ans_tr = np.bincount([(i + k) % n for i, k in train_q], minlength=n)
                maj = int(ans_tr.argmax())
                info.update(n_items=n, k_range=[1, 2 * n - 1, "k != n"], n_classes=n * (n - 1), n_held_classes=len(S),
                            n_excluded_numbers_leak=len(excl), chance_uniform=1 / n,
                            chance_majority=float(np.mean([(i + k) % n == maj for i, k in held_s])),
                            **cyclic_closure(n, trc, held_s))
                if d == "clock":   # the clock read-out: tier S with wrap (i + k >= 24; no-wrap = plain addition)
                    hw = [(i, k) for i, k in held_s if i + k >= 24]
                    info.update(n_held_wrap=len(hw), chance_wrap=max(1 / n, float(np.mean([(i + k) % n == maj for i, k in hw]))))
                    # indirect route into the numbers held-out evals (caveat, not dropped): a trained clock line with
                    # k > 24 and no wrap after the period shift ('29 hours after 3:00 is 8:00') is the shifted twin of the
                    # addition i + (k - 24), which may be a held-out numbers pair
                    info.update(n_numbers_held_shift_twin=sum(1 for i, k in train_q if k > n and i + k - n < n
                                                              and (min(i, k - n), max(i, k - n)) in held_set))
                # read-out sets (suffix of the eval keys) and their chance (max(uniform, majority training answer))
                info.update(readout="held_wrap" if d == "clock" else "held", readoutE="heldE_wrap" if d == "clock" else "heldE")
                sets = {"held": held_s, "heldE": held_e}
                for nm_, qs in sets.items():
                    # modwrap: the start plus the shift reduced mod n passes the period (k >= n with i + k - n < n is a
                    # wrap only before the reduction: 'subtract n, then a plain step')
                    m = {"wrap": np.array([i + k >= n for i, k in qs]), "k_ge_n": np.array([k >= n for i, k in qs]),
                         "modwrap": np.array([i + k % n >= n for i, k in qs])}
                    if d == "clock":
                        nw = np.array([i + k < 24 for i, k in qs])
                        hp = np.array([(min(i, k), max(i, k)) in held_set for i, k in qs])
                        m.update(nowrap_numtrain=nw & ~hp, nowrap_numheld=nw & hp)
                    masks[nm_] = m
                    ro = m["wrap"] if d == "clock" else np.ones(len(qs), dtype=bool)
                    ans_ = np.array([(i + k) % n for i, k in qs])[ro]
                    info[f"chance_readout{'E' if nm_ == 'heldE' else ''}"] = max(1 / n, float(np.mean(ans_ == maj)))
                    info[f"n_readout{'E' if nm_ == 'heldE' else ''}"] = int(ro.sum())
                if d == "clock":
                    assert info["chance_readout"] == info["chance_wrap"]
                DTAB[d] = circle(n)
            else:
                Dd = DTASK[d]
                frac, mode = (args.s4_train, "question") if d == "s4" else (args.dim_train, "class")
                base = int(sr.integers(2 ** 31))
                for att in range(200):   # toy_struct.split asserts every item is a trained input and answer
                    try:
                        _, tr, te = ts.split(Dd, frac, base + att, mode)
                        break
                    except AssertionError:
                        continue
                else:
                    raise RuntimeError(f"no valid {d} split")
                train_q, held_s = [int(i) for i in tr], [int(i) for i in te]
                rows_ = [tuple(r) for r in Dd["X"].tolist()]
                assert not ({rows_[i] for i in train_q} & {rows_[i] for i in held_s}), "held-out question in training"
                info.update(n_items=Dd["n_items"], split_mode=mode, train_frac=frac, split_seed=base + att,
                            chance_uniform=1 / Dd["n_items"], chance_majority=ts.majority_chance(Dd, tr, te), **ts.closure(Dd, tr, te))
                X_all, Y_all = Dd["X"], Dd["Y"]
                Xte, Yte = X_all[te], Y_all[te]
                ia, ib = (0, 1) if d == "s4" else (0, 2)   # columns of the two input items
                # copy cells (audit 2026-10-17): the answer is one of the inputs (s4: an input is the identity, a e = a,
                # e b = b; dim: x times 0 = 0 times x = x per 0 = x), or dim's x per x = 0. A model that only copies
                # answers them; they are kept in the held-out set but out of the read-out (nonid)
                copy_all = (Y_all == X_all[:, ia]) | (Y_all == X_all[:, ib])
                if d == "dim":
                    copy_all |= (X_all[:, 1] == 28) & (X_all[:, 0] == X_all[:, 2])
                if d == "s4":
                    e_ = int(np.flatnonzero((Dd["mul"] == np.arange(24)).all(1))[0])
                    assert (Dd["targets"]["R"][e_] == np.eye(3).ravel()).all()
                    assert (copy_all == ((X_all[:, 0] == e_) | (X_all[:, 1] == e_))).all(), "s4 copy cells = identity cells"
                else:   # dim: the copy cells are exactly the relation classes {x, 0} (both class-key rules)
                    z_ = int(np.flatnonzero((Dd["E"] == 0).all(1))[0])
                    a_, op_, b_ = X_all[:, 0], X_all[:, 1], X_all[:, 2]
                    rules = ((op_ == 27) & ((b_ == z_) | (a_ == z_))) | ((op_ == 28) & ((b_ == z_) | (a_ == b_)))
                    keyz = np.array([z_ in ((min(a, b), max(a, b)) if o == 27 else (min(c, b), max(c, b)))
                                     for (a, o, b, _), c in zip(X_all.tolist(), Y_all.tolist())])
                    assert (copy_all == rules).all() and (copy_all == keyz).all(), "dim copy cells = classes {x, 0}"
                nonid = ~copy_all[te]
                maj = int(np.bincount(Y_all[tr], minlength=Dd["n_items"]).argmax())
                if d == "s4":
                    fr = ts.fresh_mask(Dd, tr, te)
                    masks["held"] = {"fresh": fr, "comm_twin_trained": ~fr, "nonid": nonid, "fresh_nonid": fr & nonid}
                    info.update(n_comm_twin_trained=int((~fr).sum()), n_copy_held=int((~nonid).sum()), readout="held_fresh_nonid")
                    ro = fr & nonid
                    DTAB[d] = Dd["targets"]["R"]
                else:
                    tw = ts.negtwin_mask(Dd, tr, te)
                    cls = Dd["cls"]
                    assert not (set(cls[tr]) & set(cls[te])), "relation class split across train / held-out"
                    masks["held"] = {"negtwin_trained": tw, "no_negtwin": ~tw, "times": Dd["X"][te, 1] == 27,
                                     "per": Dd["X"][te, 1] == 28, "nonid": nonid, "no_negtwin_nonid": ~tw & nonid}
                    info.update(n_classes=int(cls.max()) + 1, n_negtwin_trained=int(tw.sum()), n_copy_held=int((~nonid).sum()),
                                readout="held_no_negtwin_nonid")
                    ro = ~tw & nonid
                    DTAB[d] = Dd["targets"]["expo"]
                assert ro.sum() >= 50, (d, int(ro.sum()))
                info.update(chance_readout=max(1 / Dd["n_items"], float(np.mean(Yte[ro] == maj))), n_readout=int(ro.sum()))
                sets = {"held": held_s}
            info["chance"] = max(info["chance_uniform"], info["chance_majority"])
            for q in train_q + [q for qs in sets.values() for q in qs]:   # lines, decode asserts
                (ids, sites, a0, al), astr, ins = dom_q(d, q)
                assert tok.decode(ids[a0:]).strip() == astr and [tok.decode([ids[p]]).strip() for p, _ in sites] == ins, (d, q)
                DLINE[d, q] = (ids, sites, a0, al)
            heldc = {tuple(DLINE[d, q][0][:DLINE[d, q][2]]) for qs in sets.values() for q in qs}
            assert not (heldc & {tuple(DLINE[d, q][0][:DLINE[d, q][2]]) for q in train_q}), "held-out prompt in training"
            DHELD[d] = heldc
            DPOOL[d] = train_q
            F = DTAB[d]
            refs = [F, Dd["targets"]["R_sign"]] if d == "s4" else [F]
            c_far, qperm = far_perm(F, np.random.default_rng(dom_ss[6 + ix]), refs)
            assert d == "s4" or c_far < 0.15, (d, c_far)
            assert abs(cka_np(F, F) - 1) < 1e-9 and abs(float(cka(torch.tensor(F), centred_gram(F, "cpu"))) - 1) < 1e-4, d
            DTAB_ACT[d] = F[qperm] if d in DSHUF else F
            info.update(n_train=len(train_q), **{f"n_{k}": len(v) for k, v in sets.items()}, shuf_cka_to_true=c_far,
                        shuf_cka_refs=["R", "R_sign"] if d == "s4" else ["true"], target_dim=int(F.shape[1]),
                        prior=d in DPRI, shuf=d in DSHUF, lam=LAMD[d])
            er_d = np.random.default_rng(dom_ss[11 + ix])   # eval contexts / seen subset
            sets["seen"] = [train_q[j] for j in er_d.permutation(len(train_q))[:args.dom_seen]]
            nT = len(PARA[d])
            for t_ in range(nT):   # A2: held-out-format lines of the evaluated questions; scored span = the training line's
                for q in [q for nm_ in PARA_SETS for q in sets[nm_]]:
                    PLINE[d, t_, q] = para_q(d, t_, q)
                    ids, sites, a0, al = PLINE[d, t_, q]
                    assert ids[a0:] == DLINE[d, q][0][DLINE[d, q][2]:] and al == DLINE[d, q][3], (d, t_, q, "answer span")
                    assert [v for _, v in sites] == [v for _, v in DLINE[d, q][1]] and \
                        [ids[p] for p, _ in sites] == [DLINE[d, q][0][p] for p, _ in DLINE[d, q][1]], (d, t_, q, "items")
                    assert tuple(ids[:a0]) != tuple(DLINE[d, q][0][:DLINE[d, q][2]]), (d, t_, q, "paraphrase = training prompt")
            n_rej = 0
            for nm_, qs in sets.items():
                nctx = min(8, max(1, math.ceil(args.dom_eval_rows / len(qs))))
                assert nctx <= 9   # per-question hit counts are stored as one digit each (hits_<set>)
                # A2: one context per (context index, question): a numbers TRAINING line and the end of a val text chunk;
                # every template of the set (training template and paraphrases) uses the SAME contexts (paired)
                ctx = []
                for _c in range(nctx):
                    for j, q in enumerate(qs):
                        a_, b_, f_, r_ = ctx_pair(er_d, d, q)
                        n_rej += r_
                        ctx.append((j, line(a_, b_, f_)[0], int(er_d.integers(args.seq, len(val) + 1))))
                mk = {k: np.asarray(v, dtype=bool)[[j for j, _, _ in ctx]] for k, v in masks.get(nm_, {}).items()}
                variants = [(f"{d}_{nm_}", lambda q: DLINE[d, q], True)]
                if nm_ in PARA_SETS:   # paraphrase sets: no CKA sites (the item-state read-out stays on the training template)
                    variants += [(f"{d}_p{t_ + 1}_{nm_}", lambda q, t_=t_: PLINE[d, t_, q], False) for t_ in range(nT)]
                for name, get, cka_sites in variants:
                    rows, lens, sites_ev, qi, ntx = [], [], [], [], []
                    for j, nids, oe in ctx:
                        ids, sites, a0, al = get(qs[j])
                        n_ = args.seq - len(nids) - len(ids)   # text, the numbers line, the question at the row end
                        if cka_sites:
                            sites_ev += [(len(rows), n_ + len(nids) + p, v) for p, v in sites]
                        rows.append(np.concatenate([np.asarray(val[oe - n_:oe], dtype=np.int64), np.asarray(nids, dtype=np.int64),
                                                    np.asarray(ids, dtype=np.int64)]))
                        lens.append(al); qi.append(j); ntx.append(n_)
                    dom_evalsets[name] = (torch.tensor(np.stack(rows), device=dev), torch.tensor(lens, device=dev),
                                          np.array(sites_ev, dtype=np.int64).reshape(-1, 3), mk, d, np.array(qi, dtype=np.int64), len(qs))
                    DCTX[name] = ntx
                # per question (in the order of hits_<set>): the question, its independence unit for the bootstrap
                # (cyclic: the (item, k mod n) class, both k held together; s4 / dim: toy_struct's relation class) and
                # its sub-split masks
                cls_q = ([int(i) * CYCLIC[d] + int(k) % CYCLIC[d] for i, k in qs] if d in CYCLIC
                         else [int(DTASK[d]["cls"][q]) for q in qs])
                info[f"eval_{nm_}"] = {"questions": len(qs), "contexts": nctx, "rows": len(ctx),
                                       "q": [list(map(int, q)) if d in CYCLIC else int(q) for q in qs], "cls": cls_q,
                                       "masks": {k: "".join("1" if b else "0" for b in v) for k, v in masks.get(nm_, {}).items()}}
                if nm_ in PARA_SETS:   # same questions, units, masks and contexts as eval_<set> (not repeated)
                    for t_ in range(nT):
                        info[f"eval_p{t_ + 1}_{nm_}"] = {"questions": len(qs), "contexts": nctx, "rows": len(ctx),
                                                         "same_questions_as": f"eval_{nm_}"}
            info.update(context_layout="text + 1 numbers training line + the question (slot 1)", n_ctx_pairs_redrawn=n_rej,
                        formats={f"p{t_ + 1}": PARA[d][t_] for t_ in range(nT)})
            DINFO[d] = info
        dom_max_ans = max(int(v[1].max()) for v in dom_evalsets.values()) if dom_evalsets else 0
        allh = set().union(*DHELD.values()) if DHELD else set()   # no held-out prompt of any domain is a training prompt
        assert not (allh & {tuple(DLINE[d, q][0][:DLINE[d, q][2]]) for d in EXT for q in DPOOL[d]}), "cross-domain leak"
        # A2: no held-out-format template ever occurs in a training line. (1) No training line (every domain's training
        # pool; the 4 numbers formats with 1-, 2-, 3-digit operands) is an instance of any template (slots as
        # wildcards); (2) no rendered paraphrase prompt (text up to the answer) occurs inside any training line string
        # nor as a token sub-sequence of one; (3) the --dry_inputs content scan looks for them in the training rows.
        tr_lines = [DLINE[d, q][0] for d in EXT for q in DPOOL[d]] + [line(a, b, f)[0] for f in range(len(FORMATS))
                                                                      for a, b in ((0, 0), (7, 538), (999, 999), (45, 6))]
        tr_str = [tok.decode(t_) for t_ in tr_lines]
        for d in EXT:
            tpls = [tp for t_ in PARA[d] for tp in ((t_,) if isinstance(t_, str) else t_)]
            for tp in tpls:
                rx = re.compile("".join(".+?" if i_ % 2 else re.escape(p) for i_, p in enumerate(re.split(r"\{(\w)\}", tp))))
                hit = [s_ for s_ in tr_str if rx.search(s_)]
                assert not hit, (d, tp, "a training line is an instance of a held-out-format template", hit[:3])
        def tok_inside(pb, b_):
            """pb (int64 bytes of token ids) is a token sub-sequence of b_ (an occurrence aligned to a token)."""
            k_ = b_.find(pb)
            while k_ >= 0:
                if k_ % 8 == 0:
                    return True
                k_ = b_.find(pb, k_ + 1)
            return False
        tr_all = "\x00".join(tr_str)   # one buffer; the separators (NUL / token id -1) occur in no prompt
        tr_bytes = np.asarray([-1], dtype=np.int64).tobytes().join(np.asarray(t_, dtype=np.int64).tobytes() for t_ in tr_lines)
        for (d, t_, q), (ids, _, a0, _) in PLINE.items():
            assert tok.decode(ids[:a0]) not in tr_all, (d, t_, q, "paraphrase prompt inside a training line")
            assert not tok_inside(np.asarray(ids[:a0], dtype=np.int64).tobytes(), tr_bytes), (d, t_, q, "paraphrase prompt tokens in a training line")
        DPARA = {d: {np.asarray(PLINE[d_, t_, q][0][:PLINE[d_, t_, q][2]], dtype=np.int64).tobytes()
                     for (d_, t_, q) in PLINE if d_ == d} for d in EXT}   # for the --dry_inputs content scan
        DINFO["numbers"] = {"target": args.target, "prior": "numbers" in DPRI_ALL, "shuf": "numbers" in DSHUF,
                            "lam": LAMD["numbers"], "held_pairs": len(held_set), "chance": 0.0,
                            "helix3_shuf_cka_to_true": cka_np(features("helix3_shuf", np.arange(1000), np.random.default_rng(124 + args.seed).permutation(1000)),
                                                              features("helix3", np.arange(1000)))}
        print("domains:", json.dumps({d: {k: v for k, v in i.items() if not isinstance(v, dict)} for d, i in DINFO.items()}), flush=True)

    xs_all = np.arange(1000)
    perm = np.random.default_rng(124 + args.seed).permutation(1000)
    perm_len = np.arange(1000)
    for lo, hi in ((0, 10), (10, 100), (100, 1000)):
        perm_len[lo:hi] = lo + np.random.default_rng(125 + args.seed + lo).permutation(hi - lo)
    if args.dry_inputs:
        ids, ia, ib, ns = line(7, 538, 0)
        print("line:", [tok.decode([t]) for t in ids], "a at", ia, tok.decode([ids[ia]]), "b at", ib, tok.decode([ids[ib]]), "answer tokens", ns)
        for k in ("held_b11", "held_b33", "ood_b44"):
            x, L = evalsets[k][0][0], int(evalsets[k][1][0])
            print(f"{k}: ...{tok.decode(x[-20:].tolist())!r} answer {[tok.decode([t]) for t in x[-L:].tolist()]}")
        print({k: (tuple(v[0].shape), int(v[1].max())) for k, v in evalsets.items()}, "held unordered", {k: len(v) for k, v in held.items()})
        if args.target != "none":
            F = features(args.target, xs_all, perm, perm_len); print("target features", F.shape)
        # continues to the training rows below (model built, no training), then returns

    # ---------------------------------------------------------------- model
    model = build_model(args, tok, dev)
    n_params = sum(q.numel() for q in model.parameters())   # the tied embedding counts once
    ckpt_on = args.grad_ckpt == "on" or (args.grad_ckpt == "auto" and n_params > 150e6)
    if ckpt_on:   # train mode only (evaluate() runs in eval mode); the prior's forward hooks see the same states
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"model: {n_params / 1e6:.1f}M parameters; gradient checkpointing {'on' if ckpt_on else 'off'}", flush=True)
    geo_layers = [int(v) for v in args.geo_layers.split(",")] if args.geo_layers else [args.n_layers // 3, 2 * args.n_layers // 3]
    read_layers = [args.n_layers // 3, 2 * args.n_layers // 3]   # transfer read-out: the same layers for every arm
    store = {}
    for L in sorted(set(geo_layers) | set(read_layers)):
        model.model.layers[L - 1].register_forward_hook(lambda _m, _i, o, L=L: store.__setitem__(L, nl._hidden(o)))
    steps = int(args.tokens // (args.bs * args.seq))
    warm = max(1, steps // 100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd, fused=(dev == "cuda"))
    sched = lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = np.random.default_rng(2000 + args.seed)
    Ftab = features(args.target, xs_all, perm, perm_len) if args.target != "none" else None

    nl_len = len(enc("\n"))
    if args.site == "text" or HO:   # Rev C: --held_operands counts H in the text with the same rule
        assert args.tokenization == "whole", "--site text needs whole-number tokens (0..999 are single tokens)"
        dec = [tok.decode([i]) for i in range(len(tok))]
        numv = np.array([int(t) if t.isascii() and t.isdigit() and len(t) <= 3 and (t == "0" or t[0] != "0") else -1 for t in dec])
        digit0 = np.array([len(t) > 0 and t[0].isdigit() for t in dec])
        digit1 = np.array([len(t) > 0 and t[-1].isdigit() for t in dec])
        # not part of a longer number, a decimal, a range / date / ratio / time / exponent, a list index or a digit group
        tsep = np.array([t.strip() in (".", ",", "/", "-", "\u2013", "\u2014", ":", "e", "E", "x") for t in dec])
        glued = np.array([t.endswith((".", "#", "$")) for t in dec])
        alpha1 = np.array([len(t) > 0 and t[-1].isalpha() for t in dec])   # "B1", "VO2", "T64": labels, not quantities
        space = np.array([len(t) > 0 and t.strip() == "" for t in dec])
        print(f"text site: {int((numv >= 0).sum())} number tokens 0..999 in the vocabulary", flush=True)

    def text_numbers(t):
        """Positions and values of standalone numbers 0..999 in a text chunk t (whole-number tokens)."""
        t = np.asarray(t)
        v = numv[t]
        nxt, nxt2 = np.append(t[1:], t[-1]), np.append(t[2:], [t[-1], t[-1]])
        prv, prv2 = np.insert(t[:-1], 0, t[0]), np.insert(t[:-2], 0, [t[0], t[0]])
        ok = ((v >= 0) & ~digit0[nxt] & ~((tsep[nxt] | space[nxt]) & digit0[nxt2]) & ~digit1[prv]
              & ~((tsep[prv] | space[prv]) & digit1[prv2]) & ~glued[prv] & ~alpha1[prv])
        ok[:2] = False; ok[-2:] = False   # context unknown at the edges; the next token would be the arithmetic line
        idx = np.flatnonzero(ok)
        return idx, v[idx]

    # ---------------------------------------------------------------- Rev C (wall 1): held operands H (--held_operands)
    HINFO, hsets = {}, {}
    if HO:
        t_c = time.time()
        num_ids = np.array([enc(str(v))[0] for v in range(1000)])
        assert (numv[num_ids] == np.arange(1000)).all() and len(set(num_ids.tolist())) == 1000
        cnt_st, cnt_raw, C_ = np.zeros(1000, dtype=np.int64), np.zeros(1000, dtype=np.int64), 20_000_000
        for i in range(0, len(train), C_):   # text-site count over the whole train bin, with the text site's own rule:
            # 2 tokens of overlap on each side, so every position is judged once with its true neighbours
            lo, hi = max(0, i - 2), min(len(train), i + C_ + 2)
            t_ = np.asarray(train[lo:hi], dtype=np.int64)
            ti, tv = text_numbers(t_)
            keep = (ti + lo >= i) & (ti + lo < i + C_)
            cnt_st += np.bincount(tv[keep], minlength=1000)[:1000]
            cnt_raw += np.bincount(t_[i - lo:min(i + C_, len(train)) - lo], minlength=len(tok))[num_ids]
        cand = np.arange(100, 1000)
        order = cand[np.lexsort((cand, cnt_st[cand]))]   # by text-site count, ties by value
        strata = np.array_split(order, HELD_STRATA)
        hr = np.random.default_rng(6000 + args.seed)
        if not VF:
            H = sorted(int(v) for st_ in strata for v in hr.choice(st_, args.held_operands // HELD_STRATA, replace=False))
        else:   # Rev D: k per half of every hundreds block (ranked by text-site count, ties by value), same stream
            k_vf, vf, vf_edges = args.held_operands // (2 * HELD_BLOCKS), {}, {}
            for blk in range(1, HELD_BLOCKS + 1):
                nb = np.arange(100 * blk, 100 * blk + 100)
                ob = nb[np.lexsort((nb, cnt_st[nb]))]
                halves = {"rare": ob[:50], "common": ob[50:]}
                assert cnt_st[halves["rare"]].max() <= cnt_st[halves["common"]].min() and len(set(ob.tolist())) == 100
                vf_edges[blk] = {hf: [int(cnt_st[p].min()), int(cnt_st[p].max())] for hf, p in halves.items()}
                for hf, part in halves.items():
                    for v in hr.choice(part, k_vf, replace=False):
                        vf[int(v)] = (blk, hf)
            H = sorted(vf)
            assert len(H) == args.held_operands == len(vf) and all(100 <= h <= 999 and vf[h][0] == h // 100 for h in H)
            for blk in range(1, HELD_BLOCKS + 1):   # k rare + k common in every block
                for hf in ("rare", "common"):
                    assert sum(1 for h in H if vf[h] == (blk, hf)) == k_vf, (blk, hf)
        Hs = frozenset(H)
        stratum = {int(v): k for k, st_ in enumerate(strata) for v in st_}
        H_tok = np.array(sorted(int(num_ids[h]) for h in H), dtype=np.int64)
        ansvals = [[int(numv[t]) for t in enc(str(s_))] for s_ in range(1999)]   # number values of a sum's tokens
        assert all(ansvals[s_] == ([s_] if s_ < 1000 else [s_ // 10, s_ % 10]) for s_ in range(1999))
        assert not (Hs & set(range(10))), "a held digit would be the last token of every 10th sum >= 1000"
        clean = lambda a, b: a not in Hs and b not in Hs and not any(v in Hs for v in ansvals[a + b])
        ck_ = np.random.default_rng(6001 + args.seed)   # clean() == no H token among the line's tokens (every format)
        tests_ = [(int(ck_.integers(1000)), int(ck_.integers(1000))) for _ in range(3000)]
        tests_ += [(h, int(ck_.integers(1000))) for h in H] + [(a, h - a) for h in H for a in (0, h // 2)]
        tests_ += [(999, 10 * h + 5 - 999) for h in H if h < 200]
        for a, b in tests_:
            for f in range(len(FORMATS)):
                assert clean(a, b) == (not np.isin(line(a, b, f)[0], H_tok).any()), (a, b, f)
        assert sum(not clean(a, b) for a, b in tests_) > len(H), "the clean test never failed"
        n_reh0 = len(rehearsal_all)
        rehearsal_all[:] = [q for q in rehearsal_all if clean(*q)]   # in place: draw_pair_ho draws from this list
        pre0, mid0, post0 = pieces[0]

        def row_pair(x, ln):
            """(a, b) of a format-0 eval row (the line at the row end; ln = answer tokens + the newline)."""
            eB = len(x) - ln - len(post0) - 1
            eA = eB - len(mid0) - 1
            assert x[eB + 1:eB + 1 + len(post0)] == post0 and x[eA + 1:eB] == mid0 and x[eA - len(pre0):eA] == pre0
            return int(numv[x[eA]]), int(numv[x[eB]])
        filt = {}
        for k in [k for k in evalsets if k.startswith(("held_b", "seen_b"))]:   # seen operands only (clean pairs)
            ids_, lens_, sums_ = evalsets[k]
            prs = [row_pair(x, ln) for x, ln in zip(ids_.tolist(), lens_.tolist())]
            assert [a + b for a, b in prs] == sums_.tolist(), k
            m = np.array([clean(a, b) for a, b in prs])
            evalsets[k] = (ids_[torch.tensor(m, device=dev)], lens_[torch.tensor(m, device=dev)], sums_[m])
            filt[k] = [int(len(m)), int(m.sum())]
        hp, hv = np.random.default_rng(6100 + args.seed), np.random.default_rng(6200 + args.seed)
        seen_num = {L: [v for v in range(*RANGE[L]) if v not in Hs] for L in (1, 2, 3)}
        one, one_m = [], {f"p{L}": [] for L in (1, 2, 3)}
        one_m.update(hfirst=[], **{f"q{k + 1}": [] for k in range(HELD_STRATA)})
        VF_LAYERS = [f"v{j}" for j in range(1, HELD_BLOCKS + 1)] + ["rare", "common", "hi", "lo_rare"] if VF else []
        one_m.update({k: [] for k in VF_LAYERS})   # Rev D: block / half layers of H_one (masks of the same rows)
        for h in H:   # one operand in H, the other seen, every answer token seen
            for L in (1, 2, 3):
                for _ in range(HELD_ONE_PER // 6):
                    for first in (True, False):
                        while True:
                            y = seen_num[L][int(hp.integers(len(seen_num[L])))]
                            if not any(v in Hs for v in ansvals[h + y]):
                                break
                        one.append((h, y) if first else (y, h))
                        for L_ in (1, 2, 3):
                            one_m[f"p{L_}"].append(L_ == L)
                        one_m["hfirst"].append(first)
                        for k in range(HELD_STRATA):
                            one_m[f"q{k + 1}"].append(stratum[h] == k)
                        if VF:
                            for j in range(1, HELD_BLOCKS + 1):
                                one_m[f"v{j}"].append(vf[h][0] == j)
                            one_m["rare"].append(vf[h][1] == "rare"); one_m["common"].append(vf[h][1] == "common")
                            one_m["hi"].append(vf[h][0] >= 7); one_m["lo_rare"].append(vf[h][1] == "rare")
        if VF:   # every layer = the rows of its h (12 per h); blocks partition H_one, rare / common too
            hrow = np.repeat(np.array(H), HELD_ONE_PER)
            assert len(hrow) == len(one) and all(h_ in pr_ for h_, pr_ in zip(hrow.tolist(), one))
            M_ = {k: np.array(one_m[k]) for k in VF_LAYERS}
            assert (sum(M_[f"v{j}"].astype(int) for j in range(1, HELD_BLOCKS + 1)) == 1).all() and (M_["rare"] ^ M_["common"]).all()
            assert (M_["hi"] == (hrow >= 700)).all() and (M_["lo_rare"] == M_["rare"]).all()
            for k in VF_LAYERS:
                hk = {h for h in H if (vf[h][0] == int(k[1:]) if k[0] == "v" and k[1:].isdigit() else
                                       vf[h][0] >= 7 if k == "hi" else vf[h][1] == ("common" if k == "common" else "rare"))}
                assert set(hrow[M_[k]].tolist()) == hk and M_[k].sum() == HELD_ONE_PER * len(hk), k
        both = []
        while len(both) < HELD_BOTH_N:   # both operands in H (answer tokens unrestricted)
            a, b = (int(v) for v in hp.choice(H, 2, replace=False))
            both.append((a, b))
        ans, ans_m = [], {"single": [], "chunk": []}
        for h in H:   # the answer is h: sum h (one token), and for h < 200 a 4-digit sum whose first token is h
            for kind in ("single", "chunk"):
                if kind == "chunk" and h >= 200:
                    continue
                for _ in range(HELD_ANS_PER):
                    while True:
                        s_ = h if kind == "single" else 10 * h + int(hp.integers(min(10, 1999 - 10 * h)))
                        a = int(hp.integers(max(0, s_ - 999), min(999, s_) + 1)); b = s_ - a
                        if a not in Hs and b not in Hs and (kind == "single" or ansvals[s_][1] not in Hs):
                            break
                    ans.append((a, b)); ans_m["single"].append(kind == "single"); ans_m["chunk"].append(kind == "chunk")
        for nm_, prs, mk_ in (("H_one", one, one_m), ("H_both", both, {}), ("H_ans", ans, ans_m)):
            ids_, lens_, sums_ = evalset(prs, g=hv)
            mk_ = {k: np.array(v, dtype=bool) for k, v in mk_.items()}
            if nm_ == "H_both":
                mk_ = {"lt1000": sums_ < 1000, "ge1000": sums_ >= 1000,
                       "ans_tok_in_H": np.array([any(v in Hs for v in ansvals[s_]) for s_ in sums_.tolist()])}
            hsets[nm_] = (ids_, lens_, sums_, np.array(prs, dtype=np.int64), mk_)
            assert int(lens_.max()) <= max_ans, nm_
            for (a, b), x, ln in zip(prs, ids_.tolist(), lens_.tolist()):   # decode: operands, answer, H where intended
                assert row_pair(x, ln) == (a, b) and tok.decode(x[-ln:]) == f"{a + b}\n" and len(x) == args.seq
                assert (a in Hs) + (b in Hs) == {"H_one": 1, "H_both": 2, "H_ans": 0}[nm_], (nm_, a, b)
                assert nm_ != "H_ans" or any(v in Hs for v in ansvals[a + b])
                assert nm_ != "H_one" or not any(v in Hs for v in ansvals[a + b])
        # the far permutation (target helix3_farshuf; logged for every arm, the diagnostic's pi): the best of 3000 draws
        # by max(CKA to helix3 over 0..999, over 5,000 values drawn like the text site's) -- a plain random permutation
        # keeps ~0.14 over the text mix (the pt999 helix3_shuf of seed 1: 0.20), because small values dominate there
        T3 = features("helix3", xs_all)
        smp = np.random.default_rng(0).choice(1000, 5000, p=cnt_st / cnt_st.sum())   # the text site's value mix
        assert abs(cka_cov(T3[perm], T3) - cka_np(T3[perm], T3)) < 1e-9 and abs(cka_cov(T3[perm[smp]], T3[smp]) -
                                                                                  cka_np(T3[perm[smp]], T3[smp])) < 1e-9
        far_c = lambda q: (cka_cov(T3[q], T3), cka_cov(T3[q[smp]], T3[smp]))
        fg, far = np.random.default_rng(6300 + args.seed), (9.0, None, None)
        for _ in range(3000):
            q_ = fg.permutation(1000); c_ = far_c(q_)
            if max(c_) < far[0]:
                far = (max(c_), q_, c_)
        perm_far = far[1]
        assert far[0] < 0.10, far[2]
        if args.target == "helix3_farshuf":   # the target table (built above with the pt999 perm as a placeholder)
            Ftab = features(args.target, xs_all, perm_far, perm_len)
            assert np.array_equal(Ftab, T3[perm_far]) and abs(cka_np(Ftab, T3) - far[2][0]) < 1e-9
        # Rev C1: the H-only remap (target helix3_hremap; logged for every arm, the mechanism test's u_h). Candidates of h
        # (ascending): seen three-digit u, |u - h| >= 100, u != h (mod 10), u + y token-clean for every H_one partner y of
        # h; drawn for h in H ascending, each u used once (tools/ptheld_summary.py re-draws it: double path + decoys)
        part_ = {h: sorted({b if a == h else a for a, b in one if h in (a, b)}) for h in H}
        rcand = {h: [u for u in range(100, 1000) if u not in Hs and abs(u - h) >= 100 and (u - h) % 10 != 0
                     and all(not any(v in Hs for v in ansvals[u + y]) for y in part_[h])] for h in H}
        rg_, used_, u_of = np.random.default_rng(6400 + args.seed), set(), {}
        for h in H:
            c_ = [u for u in rcand[h] if u not in used_]
            assert len(c_) >= 100, (h, len(c_))
            u_of[h] = c_[int(rg_.integers(len(c_)))]; used_.add(u_of[h])
        rmap = np.arange(1000); rmap[H] = [u_of[h] for h in H]
        assert all(rmap[x] == x for x in range(1000) if x not in Hs) and len(set(rmap[H].tolist())) == len(H)
        assert not (set(rmap[H].tolist()) & Hs) and all(abs(int(rmap[h]) - h) >= 100 for h in H)
        if args.target == "helix3_hremap":   # the target table: true helix3 off H, helix3(u_h) on h
            Ftab = features(args.target, xs_all, rmap, perm_len)
            assert np.array_equal(Ftab, T3[rmap]) and np.array_equal(np.delete(Ftab, H, 0), np.delete(T3, H, 0))
            assert all(np.array_equal(Ftab[h], T3[u_of[h]]) and not np.array_equal(Ftab[h], T3[h]) for h in H)
        pi_t = {"helix3_farshuf": perm_far, "helix3_hremap": rmap}.get(args.target, perm)   # the arm's shuffled value
        h_cka_all, h_cka_text = far_c(perm)
        HINFO = {"N": len(H), "H": H, "stratum": [stratum[h] for h in H], "n_strata": HELD_STRATA,
                 "stratum_count_edges": [[int(cnt_st[st_].min()), int(cnt_st[st_].max())] for st_ in strata],
                 "count_rule": "text site rule (text_numbers) over all of train.bin; raw = every occurrence of the token",
                 "train_bin_tokens": int(len(train)), "count_text_site": cnt_st.tolist(), "count_raw": cnt_raw.tolist(),
                 "perm_H": [int(perm[h]) for h in H], "helix3_shuf_cka_to_true": h_cka_all,
                 "helix3_shuf_cka_to_true_text_mix": h_cka_text, "perm_far_H": [int(perm_far[h]) for h in H],
                 "perm_far_cka_to_true": far[2][0], "perm_far_cka_to_true_text_mix": far[2][1], "perm_far_sha":
                 hashlib.sha256(perm_far.astype(np.int64).tobytes()).hexdigest()[:12], "pi_H": [int(pi_t[h]) for h in H],
                 "pi_of": {"helix3_farshuf": "perm_far", "helix3_hremap": "remap"}.get(args.target, "perm (pt999 helix3_shuf)"),
                 "remap_H": [int(rmap[h]) for h in H], "remap_sha": hashlib.sha256(rmap.astype(np.int64).tobytes()).hexdigest()[:12],
                 "remap_rule": "u_h: seen 100..999, |u - h| >= 100, u != h mod 10, distinct, u + y token-clean for h's "
                               "H_one partners; stream 6400 + seed, h ascending, uniform over the unused candidates",
                 "remap_n_cand": [len(rcand[h]) for h in H],
                 "rehearsal_pool": [n_reh0, len(rehearsal_all)],
                 "eval_filtered_rows": filt,
                 "sets": {k: {"pairs": v[3].tolist(), "masks": {m: "".join("1" if b_ else "0" for b_ in mv) for m, mv in v[4].items()}}
                          for k, v in hsets.items()}, "count_sec": round(time.time() - t_c, 1)}
        if VF:   # Rev D: per h its block, half and text-site count; the layers' rows (masks in sets.H_one.masks)
            HINFO.update(held_strat="value_freq", block=[vf[h][0] for h in H], half=[vf[h][1] for h in H],
                         count=[int(cnt_st[h]) for h in H], per_half=k_vf, half_count_edges=vf_edges,
                         layer_rows={k: int(np.sum(hsets["H_one"][4][k])) for k in VF_LAYERS},
                         layer_h={k: int(len({int(h) for h in hrow[hsets["H_one"][4][k]]})) for k in VF_LAYERS},
                         layer_def={"v<j>": "block j = h in 100 j .. 100 j + 99", "rare / common": "lower / upper 50 of the "
                                    "block by text-site count", "hi": "blocks 7-9", "lo_rare": "the rare halves (= rare)"})
            print(f"held_strat value_freq: {k_vf} rare + {k_vf} common per block; " + "; ".join(
                f"b{blk} " + " ".join(f"{h}{'r' if vf[h][1] == 'rare' else 'c'}({int(cnt_st[h])})" for h in H if vf[h][0] == blk)
                for blk in range(1, HELD_BLOCKS + 1)) + f"; layer rows {HINFO['layer_rows']}", flush=True)
        print(f"held operands: N {len(H)} {H}; text-site counts {[int(cnt_st[h]) for h in H]}; rehearsal {n_reh0} -> "
              f"{len(rehearsal_all)}; eval rows kept {filt}; H_one {len(one)} H_both {len(both)} H_ans {len(ans)}; "
              f"helix3_shuf CKA to helix3 {h_cka_all:.4f} (text mix {h_cka_text:.4f}); far permutation {far[2][0]:.4f} "
              f"({far[2][1]:.4f}){'; the target' if args.target == 'helix3_farshuf' else ''}; H remap u_h {HINFO['remap_H']} "
              f"(sha {HINFO['remap_sha']}){'; the target' if args.target == 'helix3_hremap' else ''}; "
              f"{time.time() - t_c:.0f}s", flush=True)

        def draw_pair_ho():
            """draw_pair without H: the 5% rehearsal pool holds clean pairs only; otherwise a bucket as in draw_pair,
            the operands redrawn WITHIN it until the line is clean (counted in dyn['held']['rej']), then a held-out
            pair is rejected (bucket redrawn) as in draw_pair."""
            if data_rng.random() < 0.05:
                return rehearsal_all[int(data_rng.integers(len(rehearsal_all)))]
            while True:
                la, lb = BUCKETS[int(data_rng.integers(len(BUCKETS)))]
                if data_rng.random() < 0.5:
                    la, lb = lb, la
                while True:
                    a, b = int(data_rng.integers(*RANGE[la])), int(data_rng.integers(*RANGE[lb]))
                    if clean(a, b):
                        break
                    dyn["held"]["rej"] += 1
                if (min(a, b), max(a, b)) not in held_set:
                    return a, b

        def decode_val(t1, t2):
            """The greedy answer as a number: the first token if it is digits, then the second if it is digits too
            (-1: the first token is not a number)."""
            s_ = dec[t1]
            if not (s_.isascii() and s_.isdigit()):
                return -1
            return int(s_ + (dec[t2] if dec[t2].isascii() and dec[t2].isdigit() else ""))

    def batch(s=0):
        """Rows of text + arithmetic lines; ops = (row, operand position, value); ans = (row, first answer position,
        number of answer tokens, first line of the row?)."""
        rows, ops, ans, tmeta = [], [], [], []
        if DOM:   # Rev B: per-domain sites (row, position, item), token counts per line type, first line per row
            batch.dsite = {d: [] for d in EXT}; batch.ntok = {t: 0 for t in ("text",) + tuple(DOMS)}; batch.first = []
            batch.ntext = []; batch.slot = []
        if HO:   # Rev C: text exposure of the numbers 0..999 in this batch (text-site rule / raw tokens)
            batch.h_st, h_txt, batch.h_ntext = np.zeros(1000, dtype=np.int64), [], []
        for r in range(args.bs):
            parts, offs = [], []
            for _ in range(args.arith_lines):
                a, b = draw_pair_ho() if HO else draw_pair()
                ids, ia, ib, ns = line(a, b, int(rng.integers(len(FORMATS))))
                parts.append((ids, ia, ib, a, b))
            if DOM:
                dom_row(s, r, parts, rows, ops, ans)
                continue
            L = sum(len(p[0]) for p in parts)
            n = args.seq - L
            o = int(rng.integers(0, len(train) - n))
            row = list(np.asarray(train[o:o + n], dtype=np.int64)) + sum((p[0] for p in parts), [])
            if args.site == "text":
                ti, tv = text_numbers(np.asarray(train[o:o + n], dtype=np.int64))
                tmeta += [(r, int(i), int(v)) for i, v in zip(ti, tv)]
            if HO:   # content scan (assert): no H token anywhere in the arithmetic lines of a training row
                assert not np.isin(np.asarray(row[n:], dtype=np.int64), H_tok).any(), ("H token in a training line", r)
                h_txt.append(np.asarray(train[o:o + n], dtype=np.int64)); batch.h_ntext.append(n)
                batch.h_st += np.bincount(text_numbers(h_txt[-1])[1], minlength=1000)[:1000]
            off = n
            for j, (ids, ia, ib, a, b) in enumerate(parts):
                ops.append((r, off + ia, a)); ops.append((r, off + ib, b))
                na = len(enc(str(a + b)))
                bk = BUCKETS.index(tuple(sorted((len(str(a)), len(str(b))))))
                ans.append((r, off + len(ids) - nl_len - na, na, int(j == 0), bk))
                off += len(ids)
            rows.append(row)
        batch.tmeta = np.array(tmeta, dtype=np.int64).reshape(-1, 3)
        if HO:
            batch.h_raw = np.bincount(np.concatenate(h_txt), minlength=len(tok))[num_ids]
            batch.h_lines = args.bs * args.arith_lines
            batch.h_np = np.array(rows, dtype=np.int64)   # Rev C1: hashed into dyn['held']['data_sha'] (the model's input)
        if DOM:
            batch.np = np.array(rows)
            batch.dsite = {d: np.array(v, dtype=np.int64).reshape(-1, 3) for d, v in batch.dsite.items()}
        return torch.tensor(np.array(rows), device=dev), np.array(ops), np.array(ans)

    def dom_row(s, r, parts, rows, ops, ans):
        """Rev B row (amendment A2): text, the numbers lines (parts, drawn as in the default path) FIRST, then
        --extra_lines domain lines (domain cycled over EXT by the global line index, question drawn from the domain's
        training pool, stream 4100) in random order (stream 5000). The text offset is the default path's draw (stream
        2000, the same bound seq - len(numbers lines)) and the text is the END of the default path's chunk (its first
        len(domain lines) tokens dropped): the numbers line sees the default path's context, shortened at the far left."""
        ext = []
        for j in range(args.extra_lines):
            d = EXT[((s * args.bs + r) * args.extra_lines + j) % len(EXT)]
            ext.append((d, DPOOL[d][int(dom_rng.integers(len(DPOOL[d])))]))
        ext = [ext[i] for i in order_rng.permutation(len(ext))]
        items = [("numbers", p) for p in parts] + ext
        n0 = args.seq - sum(len(p[0]) for p in parts)   # the default path's text length (text, then the numbers line)
        o = int(rng.integers(0, len(train) - n0))        # the default path's draw: same stream, same bound
        n = n0 - sum(len(DLINE[t, p][0]) for t, p in ext)
        row = list(np.asarray(train[o + n0 - n:o + n0], dtype=np.int64))
        off = n
        batch.ntok["text"] += n; batch.first.append(items[0][0]); batch.ntext.append(n)
        batch.slot += [(t, j) for j, (t, _) in enumerate(ext)]   # slot j + 1 after the numbers line(s)
        for j, (t, p) in enumerate(items):
            if t == "numbers":
                ids, ia, ib, a, b = p
                ops.append((r, off + ia, a)); ops.append((r, off + ib, b))
                na = len(enc(str(a + b)))
                bk = BUCKETS.index(tuple(sorted((len(str(a)), len(str(b))))))
                ans.append((r, off + len(ids) - nl_len - na, na, int(j == 0), bk))
            else:
                ids, sites, _, _ = DLINE[t, p]
                batch.dsite[t] += [(r, off + q, v) for q, v in sites]
            row += ids; off += len(ids); batch.ntok[t] += len(ids)
        assert len(row) == args.seq
        rows.append(row)

    def prior(ops, table=None, layers=None):
        table = Ftab if table is None else table
        Ls = geo_layers if layers is None else layers
        r = torch.tensor(ops[:, 0], device=dev); pos = torch.tensor(ops[:, 1], device=dev)
        Fb = table[ops[:, 2]]
        K = centred_gram(Fb, dev)
        return sum(1 - cka(store[L][r, pos], K) for L in Ls) / len(Ls)

    def text_ok():
        tm = batch.tmeta
        return len(tm) >= args.text_min and len(set(tm[:, 2].tolist())) >= 4

    def prefix_vals(g):
        lens = g.integers(1, 4, size=args.n_prefix)   # operand lengths 1-3 equally often, as in the lines
        return np.array([int(g.integers(*RANGE[int(l)])) for l in lens])

    # audit 2026-10-14: with gradient checkpointing the backward recomputation re-fires the hooks after the prefix forward
    assert not (ckpt_on and args.site == "prefix"), "--site prefix with gradient checkpointing is not supported"
    if args.site == "prefix":   # "Q: x" = one token per number after "Q: " (whole tokens), all prompts the same length
        PRE = {v: enc(f"Q: {v}") for v in range(1000)}
        assert len({len(t) for t in PRE.values()}) == 1 and all(PRE[v][-1] == enc(str(v))[-1] for v in (0, 7, 42, 999))
        prng = np.random.default_rng(3000 + args.seed)   # own stream: the training batches are those of every other site

    def site_prior(ops, g=None):
        """1 - CKA at the prior's site (ctx: operands in the lines; text: numbers in the text part; prefix: last token
        of separate 'Q: x' prompts). The prefix forward overwrites the hooked states, so the training forward's states
        are put back for the read-outs."""
        if args.site == "ctx":
            return prior(ops)
        if args.site == "text":
            return prior(batch.tmeta)
        if args.site == "emb":   # Rev C: the embedding rows of the number tokens 0..999 (H too) vs the target, on the weights
            return 1 - cka(model.get_input_embeddings().weight[num_tok], K_emb)
        vals = prefix_vals(prng if g is None else g)
        keep = dict(store)
        model(input_ids=torch.tensor([PRE[int(v)] for v in vals], device=dev))
        L_ = len(PRE[0]) - 1
        out_ = prior(np.stack([np.arange(len(vals)), np.full(len(vals), L_), vals], 1))
        store.update(keep)
        return out_

    Fref = features("helix3", xs_all)   # transfer read-out for every arm (incl. none): operands in the lines vs helix3
    if args.tokenization == "whole":   # embedding read-out: the rows of the number tokens 0..999 vs helix3
        assert all(len(enc(str(v))) == 1 for v in range(1000))
        num_tok = torch.tensor([enc(str(v))[0] for v in range(1000)], device=dev)
        K1000 = centred_gram(Fref, dev)
    if args.site == "emb" and Ftab is not None:   # Rev C: the emb site's target Gram over 0..999 (shuffled target: Ftab is)
        K_emb = centred_gram(Ftab, dev)
        if args.target == "helix3_hremap":   # Rev C1: the target Gram rows of h and u_h are the same row
            assert all(torch.equal(K_emb[h], K_emb[int(rmap[h])]) for h in H)
    if HO:   # Rev C1: the embedding rows of h, of u_h, and of a decoy (u of the next h, cyclic) for emb_cos_hu
        hu_tok = [num_tok[torch.tensor(np.asarray(v_, dtype=np.int64), device=dev)]
                  for v_ in (H, rmap[H], np.roll(rmap[H], 1))]

    def ans_score(lg, x, ln):
        """Rev C1 (--held_operands): per eval row scored as in evaluate() (lg = logits of the last max_ans + 1 positions):
        the answer's teacher-forced log-prob (answer tokens + the newline, nats, summed), the first answer token's
        log-prob and its rank over the vocabulary (number of tokens with a strictly larger logit; 0 = the argmax)."""
        lgf = lg[:, :-1].float()
        tgt = x[:, -max_ans:]
        lp = lgf.gather(-1, tgt[..., None])[..., 0] - torch.logsumexp(lgf, -1)
        need = torch.arange(max_ans, device=dev)[None] >= (max_ans - ln)[:, None]
        ar, j_ = torch.arange(len(x), device=dev), max_ans - ln
        g1 = lgf[ar, j_]
        rank1 = (g1 > g1.gather(1, tgt[ar, j_][:, None])).sum(1)
        return (lp * need).sum(1), lp[ar, j_], rank1

    def ans_logp_direct(x, ln):
        """ans_score's log-prob on a second path (double path, logged every eval, asserted on CPU): the final hidden
        states of the whole row, then the head at each answer position p - 1 by explicit index."""
        hs_ = model.model(input_ids=x, use_cache=False).last_hidden_state
        out_ = []
        for r_ in range(len(x)):
            S_, L_ = x.shape[1], int(ln[r_])
            out_.append(sum(float(torch.log_softmax(model.lm_head(hs_[r_, p_ - 1]).float(), -1)[x[r_, p_]])
                            for p_ in range(S_ - L_, S_)))
        return out_

    @torch.no_grad()
    def evaluate(step) -> dict:
        model.eval()
        res = {"step": step, "tokens": step * args.bs * args.seq}
        with torch.autocast(dev, dtype=torch.bfloat16):
            for name, (ids, lens, sums) in evalsets.items():
                hits, lps_ = [], []
                for i in range(0, len(ids), 250):
                    x, ln = ids[i:i + 250], lens[i:i + 250]
                    lg = model(input_ids=x, logits_to_keep=max_ans + 1).logits
                    pred = lg[:, :-1].argmax(-1)
                    tgt = x[:, -max_ans:]
                    ok = torch.ones(len(x), dtype=torch.bool, device=dev)
                    for k in range(max_ans):
                        need = ln > (max_ans - 1 - k)
                        ok &= ~need | (pred[:, k] == tgt[:, k])
                    hits.append(ok)
                    if HO:   # Rev C1: the continuous score on the seen sets too (the scale of logp_H_one)
                        lps_.append(ans_score(lg, x, ln)[0])
                hit = torch.cat(hits).cpu().numpy()
                res[name] = float(hit.mean())
                if HO:
                    res[f"logp_{name}"] = float(torch.cat(lps_).mean())
                if name.startswith("held_b") and name.endswith("3"):   # sums < 1000: one answer token; >= 1000: "1xx" + digit
                    for tag, m in (("lt1000", sums < 1000), ("ge1000", sums >= 1000)):
                        res[f"{name}_{tag}"] = float(hit[m].mean()) if m.any() else None
            for name, (ids, lens, sums, _, mk) in hsets.items():   # Rev C: H sets, same scoring + greedy answers
                hits, preds, sc_, dpd = [], [], [], 0.0
                for i in range(0, len(ids), 250):
                    x, ln = ids[i:i + 250], lens[i:i + 250]
                    lg = model(input_ids=x, logits_to_keep=max_ans + 1).logits
                    pred = lg[:, :-1].argmax(-1)
                    tgt = x[:, -max_ans:]
                    ok = torch.ones(len(x), dtype=torch.bool, device=dev)
                    for k in range(max_ans):
                        need = ln > (max_ans - 1 - k)
                        ok &= ~need | (pred[:, k] == tgt[:, k])
                    hits.append(ok)
                    sc_.append(ans_score(lg, x, ln))   # Rev C1: log-prob of the answer, of its first token; rank
                    if i == 0:   # double path on the first 8 rows
                        dpd = max(abs(a_ - b_) for a_, b_ in zip(ans_logp_direct(x[:8], ln[:8]), sc_[-1][0][:8].tolist()))
                    # greedy answer: the first answer token from the prompt (lg index max_ans - ln = the position before
                    # the answer), then the next token after it (index + 1) with the predicted first token in place
                    ar, j_ = torch.arange(len(x), device=dev), max_ans - ln
                    t1 = lg[ar, j_].argmax(-1)
                    x2 = x.clone(); x2[ar, x.shape[1] - ln] = t1
                    t2 = model(input_ids=x2, logits_to_keep=max_ans + 1).logits[ar, j_ + 1].argmax(-1)
                    preds += [decode_val(a_, b_) for a_, b_ in zip(t1.tolist(), t2.tolist())]
                hit = torch.cat(hits).cpu().numpy()
                res[name] = float(hit.mean())
                for k, m in mk.items():
                    res[f"{name}_{k}"] = float(hit[m].mean()) if m.any() else None
                res[f"hits_{name}"] = "".join("1" if h_ else "0" for h_ in hit.tolist())
                res[f"pred_{name}"] = preds
                # a scored hit decodes to its sum (the newline after a one-token answer is not a digit): logged, and
                # asserted in the CPU tests (a GPU run is not stopped on a bf16 tie)
                res[f"pred_mismatch_{name}"] = int(sum(1 for p_, s_, h_ in zip(preds, sums.tolist(), hit.tolist()) if h_ and p_ != s_))
                # Rev C1: the continuous score (lesson c's bracket and the 4x trend read it where exact match sits at 0)
                lp_, lp1_, rk_ = (torch.cat([c_[j] for c_ in sc_]).cpu() for j in range(3))
                res[f"logp_{name}"] = float(lp_.mean()); res[f"logp1_{name}"] = float(lp1_.mean())
                res[f"rank1med_{name}"] = float(rk_.float().median())
                if VF and name == "H_one":   # Rev D: the continuous score per layer (accuracy: H_one_<layer> above)
                    for k in VF_LAYERS:
                        res[f"logp_H_one_{k}"] = float(lp_[torch.from_numpy(mk[k])].mean())
                if name == "H_one":
                    res["logp_rows_H_one"] = [round(float(v), 4) for v in lp_.tolist()]
                    res["rank1_rows_H_one"] = [int(v) for v in rk_.tolist()]
                # known answers: a hit's first token is the argmax (rank 0); log-probs are <= 0; the two paths agree
                res[f"rank_hit_bad_{name}"] = int(((rk_ != 0) & torch.tensor(hit)).sum())
                res[f"logp_dp_maxdiff_{name}"] = float(dpd)
                if dev == "cpu":   # the CPU tests (bf16 ties on GPU are logged, not fatal)
                    assert res[f"rank_hit_bad_{name}"] == 0 and float(lp_.max()) <= 1e-5 and dpd < 2e-2, (name, res[f"rank_hit_bad_{name}"], dpd)
                    assert res[f"pred_mismatch_{name}"] == 0, (name, res[f"pred_mismatch_{name}"])
            if HO:   # Rev C1: the remap arm pulls the embedding rows of h and u_h together (decoy: the u of another h)
                E_ = model.get_input_embeddings().weight.float()
                res["emb_cos_hu"] = float(Fn.cosine_similarity(E_[hu_tok[0]], E_[hu_tok[1]], dim=-1).mean())
                res["emb_cos_hu_decoy"] = float(Fn.cosine_similarity(E_[hu_tok[0]], E_[hu_tok[2]], dim=-1).mean())
            vrr = np.random.default_rng(7)
            vl, nats, nbytes = [], 0.0, 0
            for _ in range(8):
                o = vrr.integers(0, len(val) - args.seq - 1, size=16)
                x = torch.tensor(np.stack([val[i:i + args.seq + 1].astype(np.int64) for i in o]), device=dev)
                ce = Fn.cross_entropy(model(input_ids=x[:, :-1]).logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="sum")
                vl.append(float(ce) / x[:, 1:].numel()); nats += float(ce)
                nbytes += sum(len(tok.decode(row).encode("utf-8")) for row in x[:, 1:].tolist())
            res["val_loss"] = float(np.mean(vl))
            res["val_bits_per_byte"] = nats / math.log(2) / nbytes   # comparable across tokenisations
            if DOM:   # Rev B: per-domain exact match (answer through the newline) + item-state CKA to the TRUE target
                states = {d: {L: [] for L in read_layers} for d in EXT}
                for name, (ids, lens, sites_ev, mk, d, qi_, nq_) in dom_evalsets.items():
                    hits = []
                    for i in range(0, len(ids), 250):
                        x, ln = ids[i:i + 250], lens[i:i + 250]
                        lg = model(input_ids=x, logits_to_keep=dom_max_ans + 1).logits
                        pred = lg[:, :-1].argmax(-1)
                        tgt = x[:, -dom_max_ans:]
                        ok = torch.ones(len(x), dtype=torch.bool, device=dev)
                        for k in range(dom_max_ans):
                            need = ln > (dom_max_ans - 1 - k)
                            ok &= ~need | (pred[:, k] == tgt[:, k])
                        hits.append(ok)
                        m = sites_ev[(sites_ev[:, 0] >= i) & (sites_ev[:, 0] < i + 250)]
                        for L in (read_layers if len(m) else ()):   # item states of these prompts (the hooks of this
                            # forward); A2: the held-out-format sets have no sites (CKA on the training template only)
                            states[d][L].append((store[L][torch.tensor(m[:, 0] - i, device=dev), torch.tensor(m[:, 1], device=dev)].float(), m[:, 2]))
                    hit = torch.cat(hits).cpu().numpy()
                    res[name] = float(hit.mean())
                    for k, msk in mk.items():
                        res[f"{name}_{k}"] = float(hit[msk].mean()) if msk.any() else None
                    # per question: correct contexts (0..contexts), one digit each, question order of DINFO eval_<set>
                    res[f"hits_{name}"] = "".join(map(str, np.bincount(qi_[hit.astype(bool)], minlength=nq_).tolist()))
                for d in EXT:
                    cs = []
                    for L in read_layers:
                        H = torch.cat([a for a, _ in states[d][L]]); v = np.concatenate([b for _, b in states[d][L]])
                        cs.append(float(cka(H, centred_gram(DTAB[d][v], dev)))); res[f"cka_{d}@L{L}"] = cs[-1]
                    res[f"cka_{d}"] = float(np.mean(cs))
        model.train()
        return res

    w0, w1 = (float(v) for v in args.window.split(","))
    if args.lam_decay:   # Rev D: the window's first on step and first off step (prior_on's own comparisons)
        s_on0 = next(s for s in range(steps + 1) if w0 * steps <= s)
        s_on1 = next(s for s in range(s_on0, steps + 1) if not s < w1 * steps)
        assert 0 <= s_on0 < s_on1 <= steps, (s_on0, s_on1)
        print(f"lam_decay: lambda {lam_num} x (1 - (s - {s_on0}) / {s_on1 - s_on0}) at the window steps {s_on0}..{s_on1 - 1}", flush=True)

    def lam_at(s):
        """Rev D: the prior's weight in step s's loss (default: lam_num, the same float object as before)."""
        return lam_num * (1.0 - (s - s_on0) / (s_on1 - s_on0)) if args.lam_decay else lam_num
    dyn = {"ema1": 0.0, "on": 0, "ema_b": np.zeros(len(BUCKETS)), "toggles": 0, "last_on": None,
           "gate_open": True, "events": []}   # events: [step, 1 = prior switched on / 0 = off]
    assert args.schedule == "gate" or (args.gate_start == 0 and args.gate_hyst == 0), "--gate_start / --gate_hyst need --schedule gate"
    assert 0 <= args.gate_hyst < args.gate_thr, (args.gate_hyst, args.gate_thr)
    if DOM:   # Rev B counters (saved with dyn): tokens and first lines per line type, steps each domain prior was applied,
        # and a running sha256 of every training batch (the data mixture must be identical across arms of one seed and K)
        dyn.update(dtok={t: 0 for t in ("text",) + tuple(DOMS)}, dfirst={t: 0 for t in DOMS}, dused={d: 0 for d in DOMS},
                   data_sha="", dslot={d: [0] * args.extra_lines for d in EXT})   # A2: domain lines per slot after numbers
    if HO:   # Rev C counters (saved with dyn): candidate pairs redrawn for H, lines content-scanned, text exposure of 0..999
        # (text-site rule: all steps / prior-on steps; raw token occurrences)
        dyn["held"] = {"rej": 0, "lines": 0, "st": np.zeros(1000, dtype=np.int64), "st_on": np.zeros(1000, dtype=np.int64),
                       "raw": np.zeros(1000, dtype=np.int64), "data_sha": ""}   # Rev C1: running sha256 of the batches

    def show_line(ids, sites=(), a0=None):
        """Decoded tokens with the prior sites in ⟦ ⟧ and the scored span (answer through the newline) in « »."""
        out = ""
        for j, t in enumerate(ids):
            s_ = tok.decode([t])
            out += ("«" if j == a0 else "") + (f"⟦{s_}⟧" if j in sites else s_)
        return out + ("»" if a0 is not None else "")

    def dom_dry():
        """--dry_inputs, Rev B part (experiment-audit vii): per-domain split / chance / closure; example lines (training
        template, and the first held-out question in every held-out-format template) with the prior sites in ⟦ ⟧ and the
        scored span in « »; eval rows in their position (A2: text + one numbers training line + the question) and the
        same rows in every held-out format; training rows with each domain in each slot after the numbers line; token,
        first-line and slot shares over 50 batches; a content scan of 20 batches for held-out prompts and held-out-
        format prompts; the nonce tokens re-counted in train.bin."""
        for d in EXT:
            i_ = DINFO[d]
            print(f"[{d}] " + json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ('q', 'cls', 'masks')}
                                             if k.startswith('eval_') else v) for k, v in i_.items()}), flush=True)
            for q in DPOOL[d][:2]:
                ids, sites, a0, al = DLINE[d, q]
                print(f"  train line {q}: {show_line(ids, {p for p, _ in sites}, a0)!r} sites {sites} answer span {al}")
            qh = i_["eval_held"]["q"][0]
            qh = tuple(qh) if d in CYCLIC else qh
            for t_ in range(len(PARA[d])):
                ids, sites, a0, al = PLINE[d, t_, qh]
                print(f"  fmt p{t_ + 1} line {qh}: {show_line(ids, {p for p, _ in sites}, a0)!r} answer span {al} = the training line's")
            for nm_ in ("held", "heldE", "seen"):
                names = [f"{d}_{nm_}"] + ([f"{d}_p{t_ + 1}_{nm_}" for t_ in range(len(PARA[d]))] if nm_ == "held" else [])
                for name in names:
                    if name not in dom_evalsets:
                        continue
                    X_, Ln, Se, mk = dom_evalsets[name][:4]
                    for rr in range(1 if name != f"{d}_held" else 2):
                        row = X_[rr].tolist(); L_ = int(Ln[rr]); nt_ = DCTX[name][rr]
                        st = {int(p) for r_, p, _ in Se.tolist() if r_ == rr}
                        lo = nt_ - 6
                        print(f"  eval {name[len(d) + 1:]} row {rr} (len {len(row)}, text {nt_} tokens, numbers line at {nt_}.., "
                              f"sites at {sorted(st)}, scored {L_} tokens at {len(row) - L_}..{len(row) - 1}): "
                              f"...{show_line(row[lo:], {p - lo for p in st}, len(row) - L_ - lo)!r}")
        # training rows with each domain in each slot after the numbers line (slot 1 = the eval position)
        g, K_ = {}, args.extra_lines
        for _ in range(20):
            x_, ops_, _ = batch(0)
            for r_ in range(args.bs):
                assert batch.first[r_] == "numbers"
                for t0, j0 in batch.slot[r_ * K_:(r_ + 1) * K_]:
                    if (t0, j0) not in g:
                        row = x_[r_].tolist()
                        st = set(int(p) for p in ops_[ops_[:, 0] == r_][:, 1]) | {int(p) for d in EXT for rr, p, _ in batch.dsite[d].tolist() if rr == r_}
                        lo = batch.ntext[r_] - 6
                        g[t0, j0] = (f"  train row, slot {j0 + 1} = {t0} (len {len(row)}, text {batch.ntext[r_]} tokens, numbers line at "
                                     f"{batch.ntext[r_]}.., sites at {sorted(st)}): ...{show_line(row[lo:], {p - lo for p in st})!r}")
        for t0 in EXT:
            for j0 in range(K_):
                print(g.get((t0, j0), f"  (no row with {t0} in slot {j0 + 1} in 20 batches)"))
        ntok, nfirst, scan, scan_p, nlines = {t: 0 for t in ("text",) + tuple(DOMS)}, {t: 0 for t in DOMS}, 0, 0, 0
        nslot = {d: [0] * K_ for d in EXT}
        hb = {d: [np.asarray(h, dtype=np.int64).tobytes() for h in DHELD[d]] for d in EXT}
        for b_ in range(50):
            x_, _, _ = batch(b_)
            for t, v in batch.ntok.items():
                ntok[t] += v
            for t in batch.first:
                nfirst[t] += 1
            for t, j in batch.slot:
                nslot[t][j] += 1
            if b_ < 20:   # content scan: no held-out prompt (token ids up to the answer) and no held-out-format prompt
                # (A2) in the lines part of a row; occurrences in the text are counted
                for row, n_txt in zip(batch.np, batch.ntext):
                    for part, key in ((row[n_txt:], "lines"), (row[:n_txt], "text")):
                        rb = np.asarray(part, dtype=np.int64).tobytes()
                        for d in EXT:
                            for hs, what in ((hb[d], "held-out prompt"), (DPARA[d], "held-out-format prompt")):
                                for h in hs:
                                    k_ = rb.find(h)
                                    while k_ >= 0:
                                        assert key == "text" or k_ % 8, (d, what + " inside the lines of a training row")
                                        if what == "held-out prompt":
                                            scan += int(key == "text" and k_ % 8 == 0)
                                        else:
                                            scan_p += int(key == "text" and k_ % 8 == 0)
                                        k_ = rb.find(h, k_ + 1)
                    nlines += 1
        tot = sum(ntok.values())
        print("token share over 50 batches (%): " + json.dumps({t: round(100 * v / tot, 3) for t, v in ntok.items()}))
        print("first line after the text (share of rows): " + json.dumps({t: round(v / (50 * args.bs), 3) for t, v in nfirst.items()}))
        print("slot share of each domain's lines after the numbers line (slot 1, 2, ..) over 50 batches: "
              + json.dumps({d: [round(c / max(1, sum(v)), 3) for c in v] for d, v in nslot.items()}))
        print(f"content scan: {nlines} training rows, no held-out prompt of {EXT} in their lines; {scan} in their text; "
              f"no held-out-format prompt ({sum(len(v) for v in DPARA.values())}) in their lines; {scan_p} in their text")
        if {"s4", "dim"} & set(EXT):   # nonce tokens re-counted in the train bin (= the values in S4_NAMES / DIM_NAMES)
            want = {**({nm: c for nm, c in S4_NAMES.items()} if "s4" in EXT else {}), **({nm: c for nm, c in DIM_NAMES.items()} if "dim" in EXT else {})}
            ids_ = np.array([enc(" " + nm)[0] for nm in want])
            cnt = np.zeros(len(ids_), dtype=np.int64)
            for i in range(0, len(train), 50_000_000):
                cnt += np.bincount(np.asarray(train[i:i + 50_000_000]), minlength=len(tok))[ids_]
            got = dict(zip(want, cnt.tolist()))
            assert got == want, {k: (got[k], want[k]) for k in want if got[k] != want[k]}
            print(f"nonce tokens: {len(want)} single tokens, occurrences in train.bin ({len(train)} tokens) as recorded: max {max(got.values())}")

    def prior_on(s):
        if Ftab is None and not DPRI:   # Rev B: domain priors share the schedule (DPRI is empty without --domains)
            return False
        if args.schedule == "window":
            return w0 * steps <= s < w1 * steps
        if args.schedule == "gate":   # audit 2026-10-14: every operand-length bucket must be learned, not the average
            if s < args.gate_start * steps:   # delayed onset: the same boundary as a window's w0 * steps <= s
                return False
            m = float(dyn["ema_b"].min())
            if s == math.ceil(args.gate_start * steps) and s > 0 and m >= args.gate_thr:
                print(f"WARNING: at the gate's onset (step {s}) the lowest bucket EMA {m:.3f} >= {args.gate_thr}: it does not open", flush=True)
            if not args.gate_hyst:
                return m < args.gate_thr   # stateless (the original gate)
            dyn["gate_open"] = gate_step(dyn["gate_open"], m, args.gate_thr, args.gate_hyst)
            return dyn["gate_open"]
        return True

    def ho_dry():
        """--dry_inputs, Rev C part (experiment-audit vii): H with stratum, text counts and shuffled value; two eval rows
        of every H set in their position (⟦ ⟧ = an H token, « » = the scored span); a content scan of 50 training
        batches on the model's input tensor (H tokens in the lines = 0, in the text > 0) with decoded text snippets
        around H numbers."""
        print(f"[held] N {HINFO['N']}; count edges per stratum {HINFO['stratum_count_edges']}; rehearsal pool "
              f"{HINFO['rehearsal_pool']}; eval rows kept {HINFO['eval_filtered_rows']}; counted in {HINFO['count_sec']}s")
        for nm_, pp, ca, cm in (("helix3_shuf (pt999 perm)", perm, HINFO["helix3_shuf_cka_to_true"], HINFO["helix3_shuf_cka_to_true_text_mix"]),
                                ("far perm", perm_far, HINFO["perm_far_cka_to_true"], HINFO["perm_far_cka_to_true_text_mix"])):
            d_ = np.array([abs(int(pp[h]) - h) for h in H])
            print(f"[held] {nm_}: CKA to helix3 {ca:.4f} over 0..999, {cm:.4f} over the text-site mix; |pi(h) - h| min "
                  f"{d_.min()} median {int(np.median(d_))}; {int((d_ < 20).sum())} h with |pi(h) - h| < 20, "
                  f"{sum(int(pp[h]) % 10 == h % 10 for h in H)} with pi(h) = h mod 10")
        d_ = np.array([abs(int(rmap[h]) - h) for h in H])
        print(f"[held] H remap (helix3_hremap; sha {HINFO['remap_sha']}): CKA(helix3[remap], helix3) {cka_np(T3[rmap], T3):.4f}; "
              f"|u_h - h| min {d_.min()} median {int(np.median(d_))}; candidates per h {min(HINFO['remap_n_cand'])}.."
              f"{max(HINFO['remap_n_cand'])}")
        print(f"[held] this arm's pi = {HINFO['pi_of']}")
        for h, k in zip(H, HINFO["stratum"]):
            print(f"  h {h}: q{k + 1}, text-site count {int(cnt_st[h])}, raw {int(cnt_raw[h])}, pi(h) {int(pi_t[h])}, "
                  f"u_h {int(rmap[h])}" + (f", block {vf[h][0]} {vf[h][1]} (half counts {vf_edges[vf[h][0]][vf[h][1]]})" if VF else ""))
        Ht = set(H_tok.tolist())
        for name, (ids, lens, sums, prs, mk) in hsets.items():
            for rr in range(2):
                row, L_ = ids[rr].tolist(), int(lens[rr])
                lo = len(row) - 20
                hp_ = {j - lo for j in range(lo, len(row)) if row[j] in Ht}
                print(f"  eval {name} row {rr} (len {len(row)}, pair {tuple(prs[rr].tolist())}, H tokens at {sorted(j + lo for j in hp_)}, "
                      f"scored {L_} tokens at {len(row) - L_}..{len(row) - 1}): ...{show_line(row[lo:], hp_, 20 - L_)!r}")
        nl_, hl_, ht_, hs_, ex_ = 0, 0, 0, np.zeros(1000, dtype=np.int64), []
        for b_ in range(50):
            x_, ops_, _ = batch(b_)
            for r_ in range(args.bs):
                row, n_ = x_[r_].cpu().numpy(), batch.h_ntext[r_]
                hl_ += int(np.isin(row[n_:], H_tok).sum()); ht_ += int(np.isin(row[:n_], H_tok).sum()); nl_ += args.arith_lines
                ti, tv = text_numbers(row[:n_])
                hs_ += np.bincount(tv, minlength=1000)[:1000]
                for i_, v_ in zip(ti, tv):
                    if v_ in Hs and len(ex_) < 4:
                        ex_.append(show_line(row[max(0, i_ - 8):i_ + 5].tolist(), {min(8, i_)}))
        assert hl_ == 0, hl_
        print(f"content scan: {nl_} training lines in 50 batches, H tokens in the lines {hl_}; H tokens in the text {ht_} "
              f"(text-site occurrences {int(hs_[H].sum())}, H numbers seen at least once {int((hs_[H] > 0).sum())}/{len(H)}); e.g. {ex_}")

    if args.dry_inputs:   # decoded training rows: prior sites marked with ⟦ ⟧ (ctx: operands; text: numbers in the text)
        if args.site == "prefix":
            vals = prefix_vals(np.random.default_rng(0))
            print("prefix prompts:", [tok.decode(PRE[int(v)][:-1]) + "⟦" + tok.decode(PRE[int(v)][-1:]) + "⟧" for v in vals[:6]],
                  "lengths 1/2/3:", np.bincount([len(str(int(v))) for v in vals], minlength=4)[1:].tolist())
        x, ops, ans = batch()
        for r in range(3):
            row = x[r].tolist()
            site = ops[ops[:, 0] == r][:, 1].tolist() if args.site == "ctx" else batch.tmeta[batch.tmeta[:, 0] == r][:, 1].tolist()
            if DOM:   # Rev B: the domain items too
                site = sorted(site + [int(p) for d in EXT for rr, p, _ in batch.dsite[d].tolist() if rr == r])
            show = sorted(set(range(max(0, len(row) - 40), len(row))) | {j for i in site for j in range(i - 6, i + 7) if 0 <= j < len(row)})
            txt, prev = "", None
            for j in show:
                txt += (" ... " if prev is not None and j != prev + 1 else "") + (f"⟦{tok.decode([row[j]])}⟧" if j in site else tok.decode([row[j]]))
                prev = j
            print(f"train row {r} (len {len(row)}, {len(site)} prior sites at {site[:12]}): {txt!r}")
        if args.site == "text":
            ns, ok, bins = [], 0, np.zeros(3)
            for _ in range(50):
                batch(); ns.append(len(batch.tmeta)); ok += text_ok()
                bins += np.histogram(batch.tmeta[:, 2], bins=[0, 10, 100, 1000])[0]
            print(f"text numbers per batch over 50 batches: median {int(np.median(ns))}, min {min(ns)}; prior applied on {ok}/50 "
                  f"batches; values 0-9 / 10-99 / 100-999: {np.round(bins / bins.sum(), 3).tolist()}")
        if DOM:
            dom_dry()
        if args.site == "emb" and Ftab is not None:
            with torch.no_grad():
                print(f"emb site: prior on the {len(num_tok)} embedding rows of the number tokens 0..999 (no row positions); "
                      f"1 - CKA at init {float(site_prior(None)):.4f}; target {args.target} {tuple(Ftab.shape)}")
        if HO:
            ho_dry()
        return
    # evals and the gate's EMA on a token scale (audit 2026-10-14): --eval_every counts steps of 32 rows; bs 32 unchanged
    eval_every = max(1, round(args.eval_every * 32 / args.bs))
    ema_k, ema_w = (0.98, 0.02) if args.bs == 32 else (0.98 ** (args.bs / 32), 1 - 0.98 ** (args.bs / 32))
    lm_chunk = args.lm_chunk_tokens if args.lm_chunk_tokens >= 0 else (8192 if args.bs * args.seq * len(tok) * 4 > 8e9 else 0)
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    print(f"code {code_sha}; eval every {eval_every} steps; EMA {ema_k:.4f}; LM loss chunks of {lm_chunk or 'all'} tokens", flush=True)

    def lm_forward(x):
        """Mean next-token CE over the batch and a function giving the logits at (rows, positions). lm_chunk > 0: the
        logits are formed lm_chunk tokens at a time under activation checkpointing, so the full fp32 [bs, seq, vocab]
        tensor never exists; same loss up to summation order."""
        if not lm_chunk:
            o = model(input_ids=x, labels=x)
            return o.loss, (lambda rr, pp: o.logits[rr, pp])
        h = model.model(input_ids=x, use_cache=False).last_hidden_state
        hs, tg = h[:, :-1].reshape(-1, h.shape[-1]), x[:, 1:].reshape(-1)
        ce = lambda a, b: Fn.cross_entropy(model.lm_head(a).float(), b, reduction="sum")
        tot = sum(torch.utils.checkpoint.checkpoint(ce, hs[i:i + lm_chunk], tg[i:i + lm_chunk], use_reentrant=False)
                  for i in range(0, len(tg), lm_chunk))

        def at(rr, pp):
            with torch.autocast(dev, dtype=torch.bfloat16):
                return model.lm_head(h[rr, pp])
        return tot / len(tg), at

    # ---------------------------------------------------------------- Rev B helpers (used only with --domains)
    def item_name(d, v):
        return {"months": MONTHS, "weekdays": WEEKDAYS}[d][v] if d in ("months", "weekdays") else (
            str(v) if d == "clock" else (S4N if d == "s4" else DIMN)[v])

    def dom_prior(d, table):
        """1 - CKA of the states at domain d's item sites in this batch (mean over --geo_layers) vs table[item]; None
        when the batch has fewer than 4 sites or 3 distinct items (no usable Gram)."""
        m = batch.dsite[d]
        if len(m) < 4 or len(set(m[:, 2].tolist())) < 3:
            return None
        return prior(m, table)

    def dom_grads(lm_loss, ops, on, usable, gl, dgeo):
        """Before clipping, every arm: gradient norms of the LM loss and of each domain's TRUE prior term (1 - CKA,
        unweighted, whether or not it is on) and their cosines; for the priors applied this step, the weighted active
        term (shuffled target where set) and the sum of them (grad_prior / grad_cos, the default path's keys)."""
        params = [q for q in model.parameters() if q.requires_grad]

        def vec(t):
            g_ = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            return torch.cat([(a if a is not None else torch.zeros_like(q)).flatten().float() for a, q in zip(g_, params)])
        cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))
        with torch.autocast(dev, dtype=torch.bfloat16):
            true = {"numbers": prior(ops, Fref)}
            for d in EXT:
                true[d] = dgeo[d] if (d in dgeo and d not in DSHUF) else dom_prior(d, DTAB[d])
        v = {"lm": vec(lm_loss)}
        for d, t in true.items():
            if t is not None:
                v[d] = vec(t)
        info = {"grad_lm": float(v["lm"].norm())}
        names = [d for d in DOMS if d in v]
        for i, a in enumerate(names):
            info[f"gnorm_true_{a}"] = float(v[a].norm()); info[f"gcos_lm|{a}"] = cos(v["lm"], v[a])
            for b in names[i + 1:]:
                info[f"gcos_{a}|{b}"] = cos(v[a], v[b])
        act = {}
        if on and usable and Ftab is not None:
            act["numbers"] = lam_num * v["numbers"] if "numbers" not in DSHUF else vec(lam_num * gl)
        for d, g_ in dgeo.items():
            if g_ is not None:
                act[d] = LAMD[d] * v[d] if d not in DSHUF else vec(LAMD[d] * g_)
        tot = None
        for d, a in act.items():
            info[f"gnorm_prior_{d}"] = float(a.norm()); info[f"gcos_lm|prior_{d}"] = cos(v["lm"], a)
            tot = a if tot is None else tot + a
        if tot is not None:
            info.update(grad_prior=float(tot.norm()), grad_cos=cos(v["lm"], tot))
        return info

    def dom_eval_log(s, on, dgeo):
        """Per-domain fields of an eval entry (before evaluate(), on the training batch's states)."""
        lg = {}
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            for d in EXT:
                if d in DPRI:   # the active prior's value; when off (or too few sites) its fit is still logged
                    g_ = dgeo.get(d)
                    if g_ is None:
                        g_ = dom_prior(d, DTAB_ACT[d])
                    lg[f"geo_{d}"] = float(g_) if g_ is not None else None
        lg.update({f"prior_used_frac_{d}": dyn["dused"][d] / (s + 1) for d in DOMS})
        lg["prior_on_domains"] = {d: int(bool(on) and d in DPRI_ALL) for d in DOMS}
        lg["dom_sites"] = {d: int(len(batch.dsite[d])) for d in EXT}
        tot = sum(dyn["dtok"].values()); nr = sum(dyn["dfirst"].values())
        lg["token_share"] = {t: v / tot for t, v in dyn["dtok"].items()}
        lg["first_line_frac"] = {t: v / nr for t, v in dyn["dfirst"].items()}
        # A2: share of each domain's lines in slot 1 / 2 / .. after the numbers line (slot 1 = the eval position)
        lg["slot_frac"] = {d: [c / max(1, sum(v)) for c in v] for d, v in dyn["dslot"].items()}
        lg["data_sha"] = dyn["data_sha"][:16]
        return lg

    def dom_metrics(curve):
        """Read-outs per domain (curve_metrics): numbers = the 6-bucket held-out mean (readout_rules.area / t_theta);
        a domain = its primary held-out set (cyclic: tier S), plus tier E, the seen set and the item-state CKA."""
        pst = args.bs * args.seq
        mt = {"numbers": curve_metrics(curve, lambda e: float(np.mean([e[f"held_b{a}{b}"] for a, b in BUCKETS])), pst)}
        for d in EXT:
            for k in ("held", "heldE", "seen"):
                if f"{d}_{k}" in dom_evalsets:
                    mt[d if k == "held" else f"{d}_{k}"] = curve_metrics(curve, lambda e, n=f"{d}_{k}": e.get(n), pst)
            mt[d]["chance"] = 100 * DINFO[d]["chance"]
            mt[d]["final_cka"] = curve[-1].get(f"cka_{d}")
            if d == "clock":   # the clock gate's set: tier S (and E) with wrap
                for k in ("held", "heldE"):
                    mt[f"clock_{k}_wrap"] = curve_metrics(curve, lambda e, n=f"clock_{k}_wrap": e.get(n), pst)
                mt["clock_held_wrap"]["chance"] = 100 * DINFO["clock"]["chance_wrap"]
            # the read-out set(s) of the gate and the claims (DINFO readout / readoutE; s4 / dim without copy cells)
            for k in ("readout", "readoutE"):
                if k in DINFO[d]:
                    mt[f"{d}_{k}"] = curve_metrics(curve, lambda e, n=f"{d}_{DINFO[d][k]}": e.get(n), pst)
                    mt[f"{d}_{k}"].update(set=f"{d}_{DINFO[d][k]}", chance=100 * DINFO[d][f"chance_{k}"])
            for t_ in range(len(PARA[d])):   # A2: held-out-format templates: the same read-out sub-split, and seen
                pre = f"{d}_p{t_ + 1}"
                mt[f"{pre}_readout"] = curve_metrics(curve, lambda e, n=f"{pre}_{DINFO[d]['readout']}": e.get(n), pst)
                mt[f"{pre}_readout"].update(set=f"{pre}_{DINFO[d]['readout']}", chance=100 * DINFO[d]["chance_readout"])
                mt[f"{pre}_seen"] = curve_metrics(curve, lambda e, n=f"{pre}_seen": e.get(n), pst)
        return mt

    curve, t0 = [], time.time()
    if HO and args.target in ("helix3_shuf", "helix3_farshuf", "helix3_hremap"):   # Rev C: the table is helix3 of the logged pi
        assert np.array_equal(Ftab, features("helix3", xs_all)[pi_t]), "shuffled target table != helix3[pi]"
    model.train()
    ck = out / f"ckpt_{args.tokenization}_{args.target}_seed{args.seed}{args.tag}.pt"
    start = 0
    if args.ckpt_every and ck.exists():   # resume after preemption / requeue
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["rng"]; data_rng.bit_generator.state = st["data_rng"]
        if args.site == "prefix":
            prng.bit_generator.state = st["prng"]
        if DOM:
            dom_rng.bit_generator.state = st["dom_rng"]; order_rng.bit_generator.state = st["order_rng"]
        dyn.update(st["dyn"]); curve = st["curve"]; start = st["step"] + 1
        torch.set_rng_state(st["torch_rng"].cpu())
        if dev == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        print(f"resumed from {ck.name} at step {start} of {steps}", flush=True)
    for s in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(s)
        x, ops, ans = batch(s)
        if HO:   # Rev C1: running hash of the model's input (the arms of a seed must see the same data)
            dyn["held"]["data_sha"] = hashlib.sha256(dyn["held"]["data_sha"].encode() + batch.h_np.tobytes()).hexdigest()
            if s == 0:
                assert (batch.h_np == x.cpu().numpy()).all()
        if DOM:   # Rev B: running hash of the data, token / first-line counts per line type
            dyn["data_sha"] = hashlib.sha256(dyn["data_sha"].encode() + batch.np.astype(np.int64).tobytes()).hexdigest()
            for t, v in batch.ntok.items():
                dyn["dtok"][t] += v
            for t in batch.first:
                dyn["dfirst"][t] += 1
            for t, j in batch.slot:
                dyn["dslot"][t][j] += 1
            if s == 0:   # every domain site decodes to its item; the stored rows are the model's input
                assert (batch.np == x.cpu().numpy()).all()
                for d in EXT:
                    for r_, p_, v_ in batch.dsite[d][:8].tolist():
                        assert tok.decode([int(x[r_, p_])]).strip() == item_name(d, v_), (d, r_, p_, v_)
        if s == 0:   # operand positions point at the operands' last tokens; answer spans decode to the sums
            for r_, p_, v_ in ops[:16].tolist():
                assert tok.decode([int(x[r_, p_])]) == tok.decode(enc(str(v_))[-1:]), (r_, p_, v_)
            for (r_, p_, n_, _, _), (_, _, a_), (_, _, b_) in zip(ans[:8].tolist(), ops[0:16:2].tolist(), ops[1:16:2].tolist()):
                assert tok.decode(x[r_, p_:p_ + n_].tolist()).strip() == str(a_ + b_), (r_, p_, n_, a_, b_)
            if args.site == "text":   # text positions decode to their values and lie before the arithmetic lines
                first_op = {int(r_): int(p_) for r_, p_, _ in ops[0::2 * args.arith_lines].tolist()}
                for r_, p_, v_ in batch.tmeta[:64].tolist():
                    assert tok.decode([int(x[r_, p_])]) == str(v_) and p_ < first_op[r_], (r_, p_, v_)
                show = [tok.decode(x[r_, max(0, p_ - 6):p_ + 3].tolist()) for r_, p_, _ in batch.tmeta[:6].tolist()]
                print(f"text site: {len(batch.tmeta)} numbers in the first batch, e.g. {show}", flush=True)
        on = prior_on(s)
        usable = args.site != "text" or text_ok()
        lam_t = lam_at(s) if on else lam_num   # Rev D: = lam_num unless --lam_decay (then only inside the window)
        with torch.autocast(dev, dtype=torch.bfloat16):
            lm_loss, logits_at = lm_forward(x)
            gl = site_prior(ops) if (on and usable and Ftab is not None) else torch.zeros((), device=dev)
            loss = lm_loss + (lam_t * gl if (on and usable and Ftab is not None) else 0.0)
            dgeo = {}
            if DOM and on:   # Rev B: every active domain prior on its own sites (same schedule), weight LAMD[d]
                for d in DPRI:
                    dgeo[d] = dom_prior(d, DTAB_ACT[d])
                    if dgeo[d] is not None:
                        loss = loss + LAMD[d] * dgeo[d]
        with torch.no_grad():   # in-batch exact match per line (teacher forced): the gate's signal
            # answer AND the newline after it, as in the evals (a prefix of a longer prediction is not correct)
            A = torch.tensor(ans, device=dev)
            span = A[:, 2] + nl_len
            kk = torch.arange(int(span.max()), device=dev)
            pos = (A[:, 1:2] + kk).clamp(max=x.shape[1] - 1)
            valid = kk[None] < span[:, None]
            rr = A[:, 0:1].expand_as(pos)
            ok = ((logits_at(rr, pos - 1).argmax(-1) == x[rr, pos]) | ~valid).all(1)
            first = A[:, 3] == 1
            first_acc = float(ok[first].float().mean()) if (not DOM or bool(first.any())) else dyn["ema1"]
            for bk in range(len(BUCKETS)):   # per-bucket EMA over first lines (the evals' position)
                m = first & (A[:, 4] == bk)
                if bool(m.any()):
                    dyn["ema_b"][bk] = ema_k * dyn["ema_b"][bk] + ema_w * float(ok[m].float().mean())
        dyn["ema1"] = ema_k * dyn["ema1"] + ema_w * first_acc
        dyn["on"] += int(on)
        dyn["used"] = dyn.get("used", 0) + int(on and usable)
        if DOM:
            dyn["dused"]["numbers"] += int(on and usable and Ftab is not None)
            for d, g_ in dgeo.items():
                dyn["dused"][d] += int(g_ is not None)
        if dyn["last_on"] is not None and on != dyn["last_on"]:
            dyn["events"].append([s, int(on)])
        dyn["toggles"] += int(dyn["last_on"] is not None and on != dyn["last_on"]); dyn["last_on"] = on
        if HO:   # Rev C: lines content-scanned (batch() asserts no H token) and the text exposure of 0..999
            dyn["held"]["lines"] += batch.h_lines; dyn["held"]["st"] += batch.h_st; dyn["held"]["raw"] += batch.h_raw
            if on and usable and Ftab is not None:
                dyn["held"]["st_on"] += batch.h_st
        grad_info = {}
        if DOM and (s % eval_every == 0 or s == steps - 1):   # Rev B: every arm, LM and all true priors (+ the active ones)
            grad_info = dom_grads(lm_loss, ops, on, usable, gl, dgeo)
        elif (s % eval_every == 0 or s == steps - 1) and on and usable:   # LM vs prior pull, before clipping
            params = [q for q in model.parameters() if q.requires_grad]
            g1 = torch.autograd.grad(lm_loss, params, retain_graph=True, allow_unused=True)
            g2 = torch.autograd.grad(lam_t * gl, params, retain_graph=True, allow_unused=True)   # Rev D: lam_t (= args.lam by default)
            v1 = torch.cat([(g if g is not None else torch.zeros_like(q)).flatten().float() for g, q in zip(g1, params)])
            v2 = torch.cat([(g if g is not None else torch.zeros_like(q)).flatten().float() for g, q in zip(g2, params)])
            grad_info = dict(grad_lm=float(v1.norm()), grad_prior=float(v2.norm()), grad_cos=float(v1 @ v2 / (v1.norm() * v2.norm() + 1e-12)))
            if REVD:   # Rev D: |g_lm + g_prior| (the decomposition's total; grad_total below is the backward's) and the LM share
                grad_info["grad_T"] = float((v1 + v2).norm())
                grad_info["grad_lm_share"] = grad_info["grad_lm"] / max(grad_info["grad_T"], 1e-12)
            del g1, g2, v1, v2
            if args.site == "emb":   # Rev C: the emb prior moves exactly the 1000 number rows of the embedding (H too)
                ge = torch.autograd.grad(lam_t * gl, model.get_input_embeddings().weight, retain_graph=True)[0]
                nz = ge.abs().sum(1) > 0
                assert int(nz.sum()) == 1000 and bool(nz[num_tok].all()), ("emb prior rows", int(nz.sum()))
                grad_info.update(grad_prior_emb_rows=int(nz.sum()), grad_prior_emb=float(ge.float().norm()))
                if HO:
                    grad_info["grad_prior_emb_H"] = float(ge[torch.tensor(H_tok, device=dev)].float().norm())
                del ge
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # Rev D: its pre-clip total norm is logged
        opt.step()
        if s % eval_every == 0 or s == steps - 1:
            if Ftab is not None and not (on and usable) and usable:   # prior off: still log the fit at its site
                with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                    gl = site_prior(ops, np.random.default_rng(10 ** 6 + s))   # logging only: not the prior's stream
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):   # comparable across arms (incl. none)
                cka_ops_h3 = float(1 - prior(ops, Fref, read_layers))
                emb = {}
                if args.tokenization == "whole":
                    emb = dict(cka_emb_helix3=float(cka(model.get_input_embeddings().weight[num_tok], K1000)),
                               cka_unemb_helix3=float(cka(model.get_output_embeddings().weight[num_tok], K1000)))
            tbins = (np.histogram(batch.tmeta[:, 2], bins=[0, 10, 100, 1000])[0].tolist()
                     if args.site == "text" else None)   # text sites with values 0-9 / 10-99 / 100-999 in this batch
            dom_log = dom_eval_log(s, on, dgeo) if DOM else {}
            r = evaluate(s); r.update(lm=float(lm_loss), geo=float(gl) if usable else None, sec=time.time() - t0,
                                      prior_on=int(on), prior_used_frac=dyn["used"] / (s + 1), cka_ops_helix3=cka_ops_h3,
                                      text_numbers=int(len(batch.tmeta)) if args.site == "text" else None, text_bins=tbins, **emb,
                                      prior_on_frac=dyn["on"] / (s + 1), ema_first_line=dyn["ema1"],
                                      ema_bucket_min=float(dyn["ema_b"].min()),
                                      ema_bucket={f"b{la}{lb}": float(v) for (la, lb), v in zip(BUCKETS, dyn["ema_b"])},
                                      prior_toggles=dyn["toggles"], **grad_info, **dom_log)
            if REVC:   # Rev C: the consumed optimiser values (lesson 2026-10-18: a new flag must be seen to act)
                r.update(lr_now=opt.param_groups[0]["lr"], wd_now=opt.param_groups[0]["weight_decay"])
                assert r["wd_now"] == args.wd and r["lr_now"] == args.lr * sched(s), (r["wd_now"], r["lr_now"])
            if REVD:   # Rev D: the consumed lambda, the clip, the LM share of the update (manifold_aux_loss_principle)
                applied = bool(on and usable and Ftab is not None)
                gt_ = float(g_total)
                r.update(lam_now=float(lam_t) if applied else 0.0, grad_total=gt_, clip_coef=min(1.0, 1.0 / (gt_ + 1e-6)))
                if not applied:   # the loss is the LM alone: its gradient is the total
                    r["grad_lm_share"] = 1.0
                want_lam = (args.lam * (1.0 - (s - math.ceil(w0 * steps)) / (math.ceil(w1 * steps) - math.ceil(w0 * steps)))
                            if args.lam_decay else args.lam) if applied else 0.0
                assert abs(r["lam_now"] - want_lam) <= 1e-12 * max(1.0, args.lam), (s, r["lam_now"], want_lam)
                if applied and r.get("grad_T") is not None:   # double path: decomposition vs the backward's own norm
                    r["grad_T_reldiff"] = abs(r["grad_T"] - gt_) / max(gt_, 1e-12)
                    if dev == "cpu":
                        assert r["grad_T_reldiff"] < 0.05, (s, r["grad_T"], gt_)
            if HO:
                r.update(held_rej=dyn["held"]["rej"], held_lines_scanned=dyn["held"]["lines"],
                         held_data_sha=dyn["held"]["data_sha"][:16])
            curve.append(r)
            print(json.dumps(r), flush=True)
            res_ = {"args": {k: v for k, v in vars(args).items() if (DOM or k not in DOM_ARGS)
                             and not (k in REVC_DEFAULTS and v == REVC_DEFAULTS[k])
                             and not (k in REVD_DEFAULTS and v == REVD_DEFAULTS[k])}, "steps": steps,
                    "geo_layers": geo_layers, "read_layers": read_layers,
                    "n_params": n_params, "grad_ckpt_on": ckpt_on, "lm_chunk": lm_chunk,
                    "eval_every_steps": eval_every, "code_sha": code_sha, "prior_events": dyn["events"],
                    "curve": curve}
            if HO:   # Rev C: H, its counts, the eval sets' pairs and masks, the exposure during training
                res_.update(held_operands=HINFO, held_exposure={k: dyn["held"][k].tolist() for k in ("st", "st_on", "raw")})
            if DOM:   # Rev B: domain set-up and the per-domain read-outs of this curve
                res_.update(domains=DINFO, lam_domains=LAMD, lam_rule=args.lam_rule, nonce={"s4": S4N, "dim": DIMN},
                            dep_sha={"toy_struct.py": hashlib.sha256((Path(__file__).parent / "toy_struct.py").read_bytes()).hexdigest()[:12]},
                            metrics=dom_metrics(curve))
            path.write_text(json.dumps(res_, indent=1))
        if args.ckpt_every and (s + 1) % args.ckpt_every == 0 and s < steps - 1:
            tmp = ck.with_name(ck.name + ".tmp")
            st_ = {"model": model.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                   "data_rng": data_rng.bit_generator.state, "dyn": dict(dyn), "curve": curve, "step": s,
                   "prng": prng.bit_generator.state if args.site == "prefix" else None,
                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if dev == "cuda" else None}
            if DOM:
                st_.update(dom_rng=dom_rng.bit_generator.state, order_rng=order_rng.bit_generator.state)
            torch.save(st_, tmp)
            tmp.replace(ck)
        if s == args.stop_after:
            print(f"stop_after {s}: exiting (test)", flush=True)
            return
    if ck.exists():
        ck.unlink()
    if args.save_model:
        torch.save(model.state_dict(), out / f"model_{args.tokenization}_{args.target}_seed{args.seed}{args.tag}.pt")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
