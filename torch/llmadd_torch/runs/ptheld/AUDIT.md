# ptheld — wall 1 (data missing / long tail): can a number prior let a model add with numbers it never saw in the task? (2026-10-19)

Context: WRITING_HANDOFF 2026-10-19 ("先验能破哪些墙", wall ①: hold out a batch of numbers that never enter an arithmetic
line and occur only in the text; can a text / embedding prior let the model add with them; the ctx prior cannot reach them
and is the control). Builds on F78 (text prior, 10..99), F82 / F83 (sites, 0..999), ptlr (lr brackets: 53M none's best at
7% is 8e-3, win-ctx 1e-3 / 2e-3 tied), F84 addendum (lesson c). No result exists: runs/ptheld/out does not exist; nothing
was submitted.

Question. A 53M model pretrained from scratch on FineWeb + 7% arithmetic over operands 0..999. A per-seed set H of 50
three-digit numbers never occurs in ANY training arithmetic line (not as an operand, not as an answer token) but still
occurs in the FineWeb text. Does a helix3 prior that reaches H (at the text site: numbers in the text, H included when
they occur; at the new emb site: the embedding rows of all 1000 number tokens, every step, no text needed) let the model
answer h + y for h in H? The ctx prior (operands in the lines) never touches H and is the control for "a prior was on,
but not on H". The mechanism test (Rev C1, after the fresh audit): an emb prior whose target is true helix3 for every
seen number but puts each h at the helix position of another, seen number u_h; a model that computes with the prior's
geometry answers u_h + y for h + y.

## Code and provenance
- Trainer: llmadd-burst `tools/pretrain_range.py` Rev C + Rev C1 (opt-in flags `--held_operands`, `--site emb`, `--wd`,
  the targets `helix3_farshuf` and `helix3_hremap`, which need `--held_operands`; docstring blocks "Rev C" and "Rev C1");
  code_sha 3c99f6f8cb17. UNCOMMITTED working-tree change (public repo; the agent does not commit). Without these options
  the trainer is the pinned one: CPU regression below (bit-identical to a3bce3ab63c1 = 64abc19 / ptlr / ptgw and to
  cc984148b15a = c434d29 / Rev B, for the default path and --domains).
- Rev C1 (2026-10-19, the fresh audit's fixes; --held_operands only): target helix3_hremap; logged for every H run: the
  remap (`remap_H`, `remap_sha`, `remap_n_cand`), the continuous score of every H set (`logp_*`, `logp1_*`, `rank1med_*`,
  per row for H_one), `logp_<set>` on the seen / held sets, its double path (`logp_dp_maxdiff_*`) and known answers
  (`rank_hit_bad_*`), `emb_cos_hu` / `emb_cos_hu_decoy`, the running data sha (`held_data_sha`). Rev C runs of the
  existing arms are unchanged on every Rev C key (models bitwise): test 9.
- Job code = snapshot `runs/ptheld/code/tools/` (git archive of HEAD's tools/ + the working-tree pretrain_range.py);
  `code/COMMIT` = HEAD + sha256[:12] of the trainer and every module it imports (and whether the trainer is HEAD's).
  Generator `llmadd_torch/make_jobs_ptheld.py`: refuses to render when the snapshot's trainer differs from the working
  tree, or an imported module differs from HEAD; asserts every cell's flags (intended values present, no other flag, no
  flag twice); every sbatch re-checks the trainer's sha256 at start (exit 3) and honours runs/ptheld/HOLD.
- Read-out: llmadd-burst `tools/ptheld_summary.py` (uncommitted, written and tested on fake results before any result:
  `--selftest <a Rev C1 CPU run JSON>`, 11 scenarios, test 12). R1-R5 below are its rules; where this file and the code
  disagree, that is a bug to fix before reading results.

## Setting (= ptlr / pt999 53M 7%, except H)
53M (d384 x 6), seq 512, bs 32, 3e8 tokens = 18,310 steps, 1% warmup, cosine to 0.1x, AdamW (0.9, 0.95), wd 0.1 (LC cells
vary it), clip 1.0, bf16 autocast, lambda 5, hidden-state sites at layers 2 and 4, --arith_lines 4 (7%), L40S. Evals every
250 steps + the last: 75 evals (4x: 73,242 steps, 294 evals). Window (0.05, 0.4): prior on at steps 916..7323 (the last
eval with the prior on: step 7250 = "the window end"). Same seed = same init, same H, same remap, same data stream and
held-out split for every arm (arms differ only in the prior / lr / wd / budget), so arms are paired by seed; the data
stream is checked (running sha of the model's input, `held_data_sha`, equal across the arms of a seed at every common
eval step).
Why 7% and not the 2% main setting: the wall-1 question needs a model that HAS learned addition on the seen numbers, so
that failure on H is attributable to H's absence; at 2% none is at 17% held-out after 3e8 tokens (F82), at 7% none@8e-3
reaches 94-100% on the seen held-out buckets (ptlr).

## H: which numbers, and why (--held_operands 50)
- **3-digit only (100..999), 10 per quintile of the text-site count.** Reasons:
  1. Digits 0..9 are a token of every 4-digit answer (OLMo-2: s >= 1000 is the tokens s // 10 and s % 10, checked for all
     1000..1998). Holding a digit out would remove 10% of all sums >= 1000 from training and change every seen bucket;
     there are only 10 of them and b11 has 55 unordered pairs. Asserted: no digit in H.
  2. The long tail is in the 3-digit numbers: text-site counts in the 4e8-token train bin are 36..50,548 (median 172) for
     100..999, while 0..9 / 10..99 are all frequent (totals 1.24M / 1.46M vs 0.45M for all of 100..999). Wall 1 is about
     data missing / the long tail, so the frequency axis is wanted.
  3. 50 of 900 three-digit numbers = 5.6%: the seen training distribution barely changes (b33 loses ~11% of its pairs,
     b13 / b23 ~5.6%); the bucket mix is kept exactly (operands are redrawn within the drawn bucket).
- **Frequency is confounded with magnitude** (fresh audit, finding 4; checked on counts.json): Spearman(value, text-site
  count) over 100..999 = -0.86, inside H -0.92 / -0.88 / -0.87 / -0.89 (seeds 0-3); inside the value bands 100-399 /
  400-699 / 700-999 the counts of H span only 0.5-2.2 decades. The strata q1..q5 are therefore mostly a magnitude axis
  (q1 = rare = large values, more 4-digit answers; q5 = frequent = small values, 55% in 100..199). H is kept as drawn;
  R4 is descriptive and controls for value and answer length (below).
- **Strata** = quintiles of 100..999 sorted by text-site count (ties by value): count ranges 36-83 / 84-133 / 134-239 /
  239-514 / 516-50,548. H drawn per seed from stream 6000 + seed (numpy choice without replacement within each stratum).
- **Answer tokens.** A sum s < 1000 is one token (s); s >= 1000 is two (s // 10 in 100..199, then a digit). So h in
  100..199 (4-6 per seed) is also the first token of the sums 10h..10h+9: those sums never occur in training either, and
  are evaluated (H_ans "chunk").
- **Counts** (per number 0..999, in every run JSON `held_operands.count_text_site` / `count_raw`): text-site = the text
  site's own rule (text_numbers: a standalone quantity, not part of a longer number, decimal, range, date, label "B1"),
  over all of train.bin with 2 tokens of overlap between chunks; raw = every occurrence of the token (also as a chunk of a
  longer number). They differ a lot for year prefixes: 201 has 756 text-site and 345,589 raw occurrences ("2015" =
  "201" + "5"). Strata use the text-site count (what the text prior sees). Double path: an independent re-implementation
  (scratch count_text.py, vectorised rule) gives the same 1000 + 1000 counts (0 differ; totals 3,156,568 / 9,614,594).
  Exposure during training (text of the rows actually seen, all steps / prior-on steps / raw) is logged per number
  (`held_exposure`); a number's expected text-site occurrences in one run ~ 0.7 x its count (2.8e8 text tokens of 4e8).
- **H per seed** (seeds 0-3; median text-site count 192 / 195 / 206 / 164, min 52-58, max 1,260-13,771; 4-6 in 100..199):
  printed by --dry_inputs with stratum, both counts, pi(h) and u_h.

## What is removed, and how it is checked
- Training lines: a candidate pair is clean iff a, b not in H and no token of a + b is in H. `clean()` is checked at build
  against the token-level definition (no H token among the line's tokens, all 4 formats) on 3,000 random pairs + pairs
  with an H operand / an H sum / an H first chunk (asserted equal; the test must hit unclean pairs). draw_pair_ho = the
  default draw (5% rehearsal pool, else a uniform bucket, both orders, held-out pairs rejected by redrawing the bucket),
  with operands redrawn WITHIN the bucket until clean; redraws counted (`held_rej`, logged every eval).
- Rehearsal pool (525 pairs, 5% of lines): unclean pairs dropped (seed 0: 525 -> 476).
- Content scan, every training step: `assert not isin(row[text_end:], H tokens)` for every row (batch()); lines scanned
  counted (`held_lines_scanned` = 128 per step). --dry_inputs repeats it on the model's input tensor for 50 batches (H in
  the lines = 0, asserted) and counts H tokens in the text part. The fresh audit's independent leak scan (1,536 decoded
  lines, regex over the 4 formats, re-tokenised): 0 H operands / sums / first chunks, 0 H tokens; H 53x in the text.
- H stays in the text: nothing in the text is touched. Natural "a + b = c" lines in FineWeb text may contain an H number
  (FineWeb-Edu 400M has ~260 correct natural additions, 2f99b06): part of the text by design, negligible.
- Eval sets of the default path: held_bXY and seen_bXY keep only clean pairs (rows parsed back from their tokens, sums
  asserted; kept / total in `held_operands.eval_filtered_rows`). ood_b44 / b41 / b42 are untouched (4-digit operands, eval
  only, their chunks may be H tokens; not used by any rule).

## Arms (MAIN: seeds 0-3, --save_model; window 0.05,0.4; lambda 5)
| tag | args beyond the base (+ --arith_lines 4 --held_operands 50) | reaches H? | role |
|---|---|---|---|
| 53m-lr2e-3-7pct-h50-none | --target none --lr 2e-3 | - | baseline |
| 53m-lr8e-3-7pct-h50-none | --target none --lr 8e-3 | - | baseline at its best 7% lr (ptlr) |
| 53m-lr2e-3-7pct-h50-win-ctx | --target helix3 --schedule window --window 0.05,0.4 --site ctx --lr 2e-3 | no (operands only) | control: prior on, not on H |
| 53m-lr2e-3-7pct-h50-win-text | same, --site text | through the text, when h occurs | test (the informative route) |
| 53m-lr2e-3-7pct-h50-win-emb | same, --site emb | all 1000 rows, every step, no text | test (value injection) |
| 53m-lr2e-3-7pct-h50-win-emb-hremap | --target helix3_hremap, --site emb | all rows; h gets u_h's helix position | mechanism test (R3) + geometry control of win-emb (R3b) |
| 53m-lr2e-3-7pct-h50-win-text-farshuf | --target helix3_farshuf, --site text | as win-text, all numbers shuffled | geometry control of win-text (R3c) |
- emb site: `lam * (1 - CKA(W_E[num tokens 0..999], K))`, K = the column-standardised centred Gram of the target over
  0..999 (the convention of every other site, pretrain_helix.centred_gram). On the weights, no hidden states, no row
  positions. With tied embeddings (the default) these are ALSO the output rows of 0..999: the emb prior shapes the answer
  classifier of H as well (relevant for H_ans). In-run assert at every eval with the prior on: the prior's gradient on the
  embedding is non-zero on exactly 1000 rows, all of them number rows (`grad_prior_emb_rows`), H's share logged
  (`grad_prior_emb_H`).
- helix3_hremap (Rev C1; fresh audit finding 1): target table T[x] = helix3(x) for x not in H, T[h] = helix3(u_h). u_h:
  a seen three-digit number (not in H), |u_h - h| >= 100, u_h != h (mod 10) (the units digit of u_h + y differs from the
  true one), all u_h distinct, and u_h + y token-clean for every H_one partner y of h (the predicted wrong answer is a
  trained answer); drawn on stream 6400 + seed, h ascending, uniformly over the unused candidates (262-355 candidates per
  h at seed 0). Asserted before training: T = helix3[remap], T = helix3 off H, T[h] = helix3(u_h) != helix3(h), the emb
  site's target Gram rows of h and u_h identical. Logged for every H run (`remap_H`, the same for every arm of a seed), so
  the same u is scored on win-emb and none. Behaviour (test 10): `emb_cos_hu` (mean cosine of the embedding rows of h and
  u_h) 0.002 -> 0.26 -> 0.47 in 24 tiny-model steps vs 0.04-0.06 for the decoy pairing (h with the u of another h).
- helix3_farshuf = helix3 of a FAR permutation pi of ALL of 0..999: the best of 3000 draws (stream 6300 + seed) by
  max(CKA to helix3 over 0..999, CKA over 5,000 values drawn like the text site's, i.e. by text-site count); asserted
  < 0.10; the table asserted = helix3[pi] before training. Why far and not the pt999 helix3_shuf: over the text site's
  value mix the pt999 permutation keeps CKA 0.12-0.20 (seeds 0-3), the far one 0.07-0.08 (0.006-0.009 over 0..999).
  Fresh audit finding 1: because every SEEN number is shuffled too, the circuit learns on the shuffled geometry (F82:
  shuf-text still helps at 7%), so a model using the geometry maps h to a seen x' with pi(x') near pi(h), NOT to
  pi(h) + y: the pi(h) + y statistic is not a mechanism test and is kept as a descriptive line only. The arm's role is
  the geometry-specificity control of win-text (R3c).

LC (lesson c; none only; seeds 0-3, --save_model): none @ lr 1.6e-2; none @ 8e-3 with --wd 0.01 and --wd 1.0; none @ 8e-3
with --tokens 1.2e9 (4x; 73,242 steps, 294 evals; window irrelevant).
CONF (OPTIONAL, `make_jobs_ptheld.py --confirm_seeds`, the user's call, made BEFORE any result is read): seeds 4-7 of
none@2e-3, none@8e-3, win-ctx, win-text, win-emb (20 jobs, 22.0 L40S-h); R1 then uses n = 8 per cell.
(win-emb-farshuf, an earlier optional cell, is superseded by win-emb-hremap.)

## Metrics (unique definitions; run-JSON keys)
- H_one (PRIMARY): 600 rows per seed = 12 per h: partner y seen (not in H) with 1 / 2 / 3 digits x 2 draws x h first /
  h second; every answer token seen (a + b and its tokens not in H). Exact match of all answer tokens AND the newline,
  teacher forced, argmax, format 0 ("Q: a + b = ") at the END of a 512-token row after a val-text chunk = the training
  position and the scoring of held_bXY (pretrain_range.py evaluate()). Masks: p1 / p2 / p3 (partner length), hfirst,
  q1..q5 (stratum of h). Per-row hits `hits_H_one` (string, row order = `held_operands.sets.H_one.pairs`).
- acc_H1 = H_one at an eval; **final3** = mean of the LAST THREE evals of the run (1x: steps 17750, 18000, 18309; 4x:
  72750, 73000, 73241); curve max reported.
- logp_H1 (CONTINUOUS, Rev C1; fresh audit finding 3) = `logp_H_one`: mean over the rows of the teacher-forced log-prob
  (nats) of the answer tokens AND the newline, summed per row, same rows / positions as acc_H1; per row in
  `logp_rows_H_one`. Also `logp1_H_one` (first answer token only) and `rank1med_H_one` (median rank of the correct first
  answer token over the vocabulary, 0 = argmax; per row `rank1_rows_H_one`). The same score on every H set and on the
  held / seen sets (`logp_held_b33`, ..., the scale of a learned answer). Double path every eval: the first 8 rows of
  each H set recomputed from the final hidden states with the head at each answer position by explicit index
  (`logp_dp_maxdiff_<set>`; asserted < 0.02 on CPU, read-out withholds a run > 0.05); known answers: a hit's first
  token has rank 0 (`rank_hit_bad_<set>`, asserted 0 on CPU), log-probs <= 0.
- H_both: 150 rows, two different H numbers; masks lt1000 / ge1000 / ans_tok_in_H. H_ans: 3 rows per h with a + b = h
  (a, b seen) + 3 rows per h < 200 with a 4-digit sum whose first token is h; masks single / chunk. Secondary.
- Seen reference: held6 = mean of the 6 filtered held_bXY (held-out pairs of seen operands, both orders never trained);
  held_b13 / b23 / b33 = the partner-length-matched seen sets for H_one p1 / p2 / p3.
- Greedy answer `pred_<set>` per row: first answer token = argmax after the prompt, second = argmax after it with the
  predicted first token in place; the value = the digits of the two tokens (-1 if the first is not a number). A scored hit
  must decode to its sum: `pred_mismatch_<set>` logged every eval, must be 0 (asserted on CPU; a nonzero count on GPU
  withholds that run's R3).
- `emb_cos_hu` / `emb_cos_hu_decoy`: mean cosine of the input-embedding rows (h, u_h) / (h, u of the previous h), every
  eval, every H run.
- Exposure: `held_exposure.st` / `st_on` / `raw` = text-site occurrences in the training rows over all steps / over
  prior-on steps / raw token occurrences, per number 0..999.

## PRE-REGISTERED rules (written 2026-10-19, before any run; amended the same day after the fresh-agent audit, still before any run)
Amendments (fresh audit findings -> rule): 1 -> R3 rebuilt on win-emb-hremap, farshuf demoted to a control (R3c); 2 ->
"informative seed" gate for every control and LC seed; 3 -> the bracket and the 4x trend read logp_H1; 4 -> R4 controls
for value and answer length; 5 -> intersection-union + Holm over the two X instead of Holm over six tests; optional CONF
seeds; 6 -> R3 p by an h-level randomisation; 7 -> one-LC-seed wording, final3 steps of the 4x cell, read-out in code;
8 -> win-emb wording; 9 -> data sha.
Units: seed = the unit of replication; a cell's n = its finished seeds (4, or 8 with CONF); "3/4" = at least ceil(3/4 n).
**Informative seed** (controls Y and LC cells): held6 final3 >= 0.80 AND not diverged (readout_rules.diverged_at: a
non-finite lm / val_loss / geo / grad_prior at any eval) AND not unstable (final val_loss <= the same seed's none@2e-3
+ 0.5 nats). A seed that is not informative is reported as "uninformative" and counts NEITHER for nor against "none
cannot" (it is left out of (b), (d) and the paired tests). An X seed that fails the held6 / divergence gate counts as a
failure of R1(a) and keeps its measured value in the paired tests (conservative for the prior).
R1. "The prior lets the model use items never seen in the task" holds for X in {win-text, win-emb} iff ALL of
  (a) acc_H1 final3(X) >= 0.50 in >= 3/4 of X's seeds;
  (b) for each control Y in {none@2e-3, none@8e-3, win-ctx@2e-3}: >= 3/4 of Y's seeds informative, and every informative
      seed has acc_H1 final3(Y) <= 0.10. Fewer informative seeds: "NOT ESTABLISHED (uninformative control Y)", no claim;
  (c) for each Y, over the seeds where Y is informative (>= 3): X - Y > 0 in >= 3/4 of them and the one-sided paired t p
      of X - Y; p_X = the largest of the three p (intersection-union test: X must beat every Y, so no correction within X);
      Holm over the two X: the smaller p_X < .025, the larger < .05 (n = 4: t > 3.18 / 2.35, paired mean / sd >= 1.59 /
      1.18; the old Holm over six tests needed t > 4.86);
  (d) lesson c, per LC cell (none@1.6e-2, none@8e-3 wd 0.01, none@8e-3 wd 1.0, none@8e-3 4x), informative seeds only:
      all <= 0.10 (>= 3/4 informative) -> holds; exactly one informative seed > 0.10 and <= 0.50 -> the claim stands,
      qualified "(none reached v on H in 1 of n seeds of <cell>)"; exactly one > 0.50, or >= 2 seeds > 0.10 -> "the prior
      makes H usable within this budget / faster", never "none cannot"; < 3/4 informative or not run -> the claim is
      PROVISIONAL and reads "at wd 0.1, 3e8 tokens, lr 2e-3 / 8e-3".
  Bracket (the continuous score; exact match sits at the floor for every none cell): on mean logp_H1 final3 over the
  informative seeds, the lr axis none@2e-3 / 8e-3 / 1.6e-2 (wd 0.1) and the wd axis none@8e-3 wd 0.01 / 0.1 / 1.0. If the
  best cell is at an edge and beats its neighbour by > 0.25 nats (paired mean) with >= 3/4 paired wins: OPEN -> one more
  step outward (lr 1e-3 / 3.2e-2; wd 0.001 / 3.0) runs before any "cannot" wording (the claim reads PROVISIONAL until
  then). The same rule on acc_H1 when the best cell's mean exceeds 0.05 (margin 0.02).
  4x trend: logp_H1 final3 of none@8e-3-4x minus none@8e-3 >= +0.5 nats with >= 3/4 paired wins, OR the OLS rise of
  logp_H1 over the last quarter of the 4x run (slope x the quarter's token span) >= 0.25 nats in >= 3/4 seeds -> "none's
  H likelihood still rises with budget": any "cannot" reads "within 4x the budget".
  If (a) holds and a control Y fails (b): "the prior raises H accuracy by D" (paired), no wall claim. If (a) fails but X
  beats both none cells as in (c) at p < .05 each: "partial: the prior raises H accuracy to N%". Curve max >= 0.5 in
  >= 3/4 seeds with final3 < 0.5: "transient", not the claim.
  Wording for win-emb (fresh audit finding 8; the F81 caveat "the target carries the answer"): the emb prior writes h's
  value coordinates straight into h's embedding, so a win-emb success reads "injecting h's value coordinates into its
  embedding substitutes for task exposure"; win-text, which reaches h only through its natural occurrences, is the
  informative route for "a prior learned from text makes unseen items usable".
R2. Site specificity: if win-ctx fails R1(b) (an informative seed > 0.10), the effect on H is not specific to reaching H;
  X then additionally needs X - win-ctx paired mean >= +0.30 (>= 3/4 wins, its p inside R1(c)) for any R1-type wording,
  which then names the win-ctx effect.
R3. Mechanism, arm win-emb-hremap (u_h = `remap_H`). Per seed, at two time points: the window end (the eval at step
  7250, the prior on) and final3 (the last three evals pooled). S = mean over h of the share of h's 12 H_one rows (x the
  evals of the time point) whose greedy answer equals u_h + y. p = h-level randomisation test: the trainer's u-sampler
  re-drawn 10,000 times on other streams (10^7 + 1000 seed + time point) gives decoy maps u'; under H0 (the answers do
  not depend on u) the drawn u is exchangeable with its decoys, p = (1 + #{S(u') >= S(u)}) / (1 + 10,000) (all 12 rows
  of an h share one u', so rows are not treated as independent; fresh audit finding 6). Empirical controls: S with the
  same u on win-emb and on none@2e-3 of that seed. A time point "supports" iff in >= 3/4 seeds p < .01 AND S(hremap) >
  S(win-emb) AND S(hremap) > S(none@2e-3); "against" iff p >= .05 in >= 3/4 seeds; otherwise "inconclusive". A seed
  whose hremap acc_H1 at the time point is >= 0.90 or that has < 10 h with a wrong row is "not testable". Verdict:
  final3 supports -> "supports (persistent)"; only the window end -> "supports while the prior is on"; both against ->
  "against"; else "inconclusive". Validity: the read-out re-draws u with stream 6400 + seed and must reproduce
  `remap_H` (else WITHHELD); pred_mismatch > 0 on any of the three arms of a seed withholds that seed.
  Secondary (descriptive): the share of wrong answers whose units digit is that of u_h + y vs the same share on win-emb.
R3b. Geometry of h at the emb site: win-emb - win-emb-hremap on acc_H1 final3, paired mean >= +0.20 with >= 3/4 wins
  (and hremap < 0.5 in >= 3/4 seeds) -> win-emb's wording may say "h's own helix value"; otherwise "a prior on the number
  embeddings, not specific to h's value".
R3c. Geometry at the text site: win-text - win-text-farshuf, the same rule -> "helix geometry"; otherwise (or farshuf
  acc_H1 >= 0.5 in >= 3/4 seeds) "a number prior (not geometry-specific)". farshuf's share of wrong answers equal to
  pi_far(h) + y: descriptive only (not a mechanism test, see the arms).
R4. Frequency vs accuracy (descriptive, no claim at n = 4; fresh audit finding 4): per arm, acc_H1 final3 per stratum
  q1..q5 (pooled seeds); per seed the PARTIAL Spearman between per-h accuracy (12 rows x final3) and log10(text-site
  count), controlling for the value h and h's share of 4-digit answers among its H_one rows (rank residuals after OLS
  on the covariates' ranks); per arm the high- minus low-count difference inside the value bands 100-399 / 400-699 /
  700-999 (split at the band's median count, seeds pooled); the per-h contrast win-text - win-emb on the same partial
  Spearman (prediction: > 0, the text prior reaches frequent h more often); win-text also against its own prior-on
  exposure `st_on`. win-emb's own rho is NOT a null (count and value are rank-correlated -0.86).
R5. Secondary, descriptive: H_both, H_ans (single / chunk); held6 per arm (does H or the prior hurt seen addition);
  in-training bpb vs none@2e-3; acc_H1 at the window end vs final3 (does win-emb's H accuracy survive the window's end).

## Predictions / expected ranges (outside -> check the run before using it)
| cell | acc_H1 final3 | basis | check if |
|---|---|---|---|
| none @ 2e-3 / 8e-3 | 0-0.05 | h never in a line; its embedding shaped by text only | > 0.2 (leak?) |
| win-ctx @ 2e-3 | 0-0.10 | prior never touches H | > 0.3 |
| win-text @ 2e-3 | 0.05-0.6, rising with the stratum | H reached only when it occurs (stratum q1: ~10-20 occurrences in the window) | |
| win-emb @ 2e-3 | 0.2-0.9 | every H row on the helix every window step | 1.0 everywhere (too good: check H in lines) |
| win-emb-hremap @ 2e-3 | 0-0.3; at the window end S >> decoys if R3's mechanism holds | h sits at u_h's helix position | > 0.6 |
| win-text-farshuf | 0-0.1 | | > 0.4 |
logp_H1: no prior expectation of its scale at 53M (the tiny CPU model sits at -20 to -25 nats); report it next to
`logp_held_b13` / `b23` / `b33` (the learned scale). emb_cos_hu: rises in the hremap arm during the window, decoy flat.
Seen: held6 final 0.94-1.0 for every arm (ptlr none@8e-3 / win-ctx@2e-3); bpb 1.20-1.26; window arms prior_on_frac 6408 /
18310, prior_events [[916, 1], [7324, 0]]; none prior_used_frac 0; held_rej > 0; held_lines_scanned = 128 x (step + 1).
H_ans: ~0 for none (h never an answer in a line); win-emb may exceed 0 (tied output rows). LC: <= 0.05.

## Train vs eval conditions
| dimension | train | eval | same? |
|---|---|---|---|
| sequence position | 4 lines at the end of a 512-token row after FineWeb text | one line at the row end after val text | yes (the first line also follows text) |
| template | 4 formats, random | format 0 | subset |
| value range | operands 0..999 except H; answers without H tokens | H_one: one operand in H (by design never trained), answer tokens seen; held/seen bXY: seen | intended generalisation test, named as such |
| data / split | train.bin text, held-out pairs and H never in lines | val.bin text | disjoint lines by construction; H occurs in both texts |
| model state | train, bf16 autocast | eval mode, bf16 autocast | yes |

## Controls and validity (checked by tools/ptheld_summary.py before any number is used; a failure withholds every verdict using the cell)
- Every run: code_sha = the snapshot's; base args; steps 18,310 (4x: 73,242); 75 evals (4x: 294);
  `held_lines_scanned` = 128 x (step + 1) at every eval; `held_rej` > 0; pred_mismatch = 0 (else R3 withheld for the
  seed); `logp_dp_maxdiff_H_one` <= 0.05; window arms prior_on_frac 6408 / 18310 and prior_events [[916, 1], [7324, 0]];
  none prior_used_frac 0; emb arms grad_prior_emb_rows = 1000 wherever logged; wd_now / lr_now (asserted in-run).
- Per seed across all arms: `held_operands.H`, `count_text_site`, `remap_H`, `perm_far_H` and the H_one pairs identical;
  step-0 lm within 1e-3 (same init and batch); `held_data_sha` identical at every common eval step (also 1x vs 4x).
- Positive control: held6 (seen) high in every arm (the informative gate). Negative control: none on H_one and H_ans near
  0. Positive control of the H_one rows themselves (fresh audit): a ptlr model that trained WITH H (none@8e-3 seed 0)
  scores 0.995 on exactly ptheld's seed-0 H_one rows (p1 / p2 / p3 1.000 / 1.000 / 0.985, every q >= 0.992), so the rows
  are scorable and the 0.50 bar reachable.

## Known limits
4 seeds (CONF optional); one lr for the prior arms (2e-3; conservative for the prior); lambda 5 is strong for the text
site (F82 M7: lambda 1 was better at 7%); one share (7%); 53M only; tied embeddings only (no --untie control for win-emb);
H only 3-digit; frequency confounded with magnitude (R4); the text prior acts on H only when h occurs in the batch (rare
h: a handful of steps); exposure differs from count by sampling; the hremap arm tests the mechanism at the emb site only.

## Tests (CPU, 2026-10-19, before submission; scripts and logs in the session scratchpad walls/wall1/, Rev C1 in walls/wall1/fix/)
Tiny model for training tests (d64 x 3 layers, seq 128, bs 8, 24 steps, evals at steps 0 / 8 / 16 / 23, n_eval 20);
--dry_inputs also at the production shape (53M, seq 512, bs 32, n_eval 150, the rendered ARGS unchanged except --out).
Tests 1-8 ran on Rev C (09ed9837fea7); tests 9-13 on the final trainer 3c99f6f8cb17.
1. **Regression of the default path** (compare.py: run JSON equal except code_sha / wall-clock sec and the out dir, every
   eval value float-equal, saved model bitwise, stdout equal except the code line; --dry_inputs stdout byte-identical).
   Against a3bce3ab63c1 (64abc19, the ptlr / ptgw / pt2main trainer): r1 none; r2 helix3 window ctx; r3 helix3 gate at
   the text site (lambda 1; exercises the text tables now also built for --held_operands); r4 helix3_shuf at the prefix
   site; r6 none with an explicit `--wd 0.1` (= no flag: the JSON args omit it). Against cc984148b15a (c434d29, Rev B /
   rb6): r5 --domains numbers,months with both priors. Result for trainer 0a53f32ca023: r1, r2, r3, r4, r6 IDENTICAL
   (models 36/36 tensors equal, curves equal, dry stdout byte-identical); r5 identical except `dep_sha` (toy_struct.py
   6248613ac6d2 at c434d29 vs b74a78cb5472 at HEAD = the 2026-10-18 toy lr fix, not this change; its curves, model and
   dry stdout are identical). Re-run on 09ed9837fea7 (r1, r3, r4): IDENTICAL (1b). The fresh audit added x2 (text-site
   window with --untie, seed 3) and x3 (helix3_shuf at the ctx site, whole schedule, lr 3e-3, seed 1): IDENTICAL on
   09ed9837fea7. Final trainer: test 13.
2. **A/B: each new flag changes behaviour, and its consumed value is logged / asserted** (trainer 0a53f32ca023; the final
   trainer differs only by the far-shuffle branch and one assert):
   - `--wd 1.0` vs default (none, seed 0): 36/36 model tensors differ, lm at step 23 10.6167 vs 10.6005; `wd_now` 1.0 at
     every eval (asserted = --wd; `lr_now` asserted = lr x schedule). Fresh audit: `--wd 0.5` vs 0.1 on the held path
     with the emb site: 25/25 tensors differ, final lm 8.567 vs 8.517, `wd_now` 0.5 / 0.1 logged at every eval.
   - `--site emb` vs `--site ctx` (helix3, window 0.1,0.6, seed 1): 36/36 tensors differ; CKA(embedding rows, helix3)
     0.025 -> 0.184 (step 8) -> 0.470 (step 23) under emb vs 0.025 -> 0.051 under ctx; `geo` at step 0 = 0.9746 =
     1 - cka_emb_helix3 (0.0254); `grad_prior_emb_rows` = 1000 at the on-eval (asserted).
   - `--held_operands 50` vs 0 (none, seed 0): 36/36 tensors differ; step-0 lm 11.5178 vs 11.5169 (the step-0 batch
     differs: unclean lines redrawn); `held_rej` 2 -> 64 over 24 steps (8% of 768 lines needed a redraw);
     `held_lines_scanned` = 32 x (step + 1); H, both count vectors, the eval pairs and masks in `held_operands`.
3. **Fault injection (each must be caught by an in-run assert):** F1 H allowed into the lines (draw_pair_ho's clean test
   removed) -> `AssertionError: ('H token in a training line', 6)` at step 0; F2 `--wd` inert (optimizer hard-coded
   0.1, the 2026-10-18 failure mode) -> `AssertionError: (0.1, 0.001)` at the first eval; F3 emb site silently = ctx
   -> `AssertionError: ('emb prior rows', 493)` at the first on-eval; F4 far table never installed: see 1b.
4. **Known answers / double paths:** text counts in-run = independent count_text.py (0 of 1000 + 1000 differ); clean()
   = the token-level definition on 3,000 + 150 pairs x 4 formats (asserted in every run); every H eval row decodes to its
   pair and to "{a+b}\n" with the intended number of H operands (asserted); seen held / seen rows parsed back from the
   tokens with sums asserted; a scored hit decodes to its sum (`pred_mismatch` 0 at every eval of every CPU run);
   cka_cov = cka_np (asserted, 1e-9).
5. **Generator:** bash -n clean; fault injection on a scratch copy: win-emb rendered with --site text ->
   `AssertionError: ('53m-lr2e-3-7pct-h50-win-emb-s0', '--site', 'text', 'emb')`; a repeated --lr -> "given twice"; a
   duplicate cell -> tag assert; --schedule on none -> assert; a tampered snapshot -> the generator refuses
   ("code/ snapshot ... != working tree") and the rendered sbatch exits 3 ("code_sha mismatch") before any GPU work;
   HOLD present -> the sbatch exits 0 at start. Rev C1: test 14.
1b. **Trainer 09ed9837fea7** (after adding helix3_farshuf): r1 none, r3 text-site gate, r4 helix3_shuf prefix
   re-run against the same a3bce3ab63c1 outputs: IDENTICAL (models 36/36 equal, curves, logs, dry stdout byte-identical).
   A4 helix3_farshuf vs helix3_shuf (held path, text site, window 0.1,0.6, --text_min 2, seed 0): 36/36 tensors differ;
   prior applied at step 8 (geo 0.131 vs 0.246); `pi_of` perm_far vs perm, the far perm logged in both (sha
   38a0b1817759, CKA 0.0056 / 0.084). F4 (the far table never installed: the pt999 placeholder stays) ->
   `AssertionError: shuffled target table != helix3[pi]` before step 0.
6. **Smoke of every rendered cell** (Rev C, seed 0; the sbatch's ARGS with the snapshot trainer, tiny-model overrides
   appended): --dry_inputs of all 10 cells clean; training of none@8e-3, win-ctx, win-text, win-text-farshuf, win-emb:
   24 steps, every Rev C key present, pred_mismatch 0, held_lines_scanned = 32 x (step + 1), step-0 lm 11.5178 identical
   in all five (same init and batch), emb arm grad_prior_emb_rows 1000 and CKA(embedding rows, helix3) 0.026 -> 0.758
   vs 0.074 for win-ctx. At this tiny shape (seq 128) a batch has < 16 text numbers, so the text prior is never applied
   in the win-text / win-text-farshuf smoke (geo None; their curves equal); the text prior on the held path is covered by
   A4 and by the production-shape dry run (prior applicable on 50/50 batches). Rev C1: test 9.
7. **Production-shape --dry_inputs** (53M, the rendered ARGS, CPU): win-text-farshuf seeds 0-3 and win-emb seed 0.
   Eval rows kept after removing H (seed 0): held_b11 37/37, b12 145/150, b13 128/150, b22 144/150, b23 136/150, b33
   130/150; seen 25/25, 100/100, 88, 96, 81, 86 of 100 (seeds 1-3 similar: b33 131-138); rehearsal pool 525 -> 476-484.
   Content scan over 50 batches (6,400 lines): H tokens in the lines 0 (all seeds); H tokens in the text 200-829, text-site
   occurrences 25-59, 15-17 of 50 H seen at least once. Text site: median 111-120 numbers per batch (min 58-68), shares
   0-9 / 10-99 / 100-999 = 0.37-0.39 / 0.46-0.49 / 0.14-0.15 (= pt999). Emb site at the 53M init: 1 - CKA = 0.948.
   Rev C1: test 11.
8. **Resume** (Rev C; held path, none, seed 0, --ckpt_every 8 --stop_after 15, then the same command: "resumed from ...
   at step 16 of 24") vs the uninterrupted run: curves (incl. hits / greedy answers / held_rej / lines scanned),
   held_exposure and the model (36/36 tensors) identical; only args.ckpt_every and the wall-clock count_sec differ.
   Rev C1: test 11.
9. **Rev C1 smoke + equivalence** (fix/tests.sh S; the rendered sbatch ARGS of every seed-0 cell, snapshot trainer
   3c99f6f8cb17, tiny overrides): --dry_inputs of all 11 cells clean (content scan 0 H tokens in the lines; the remap
   line printed; this arm's pi = remap only for win-emb-hremap, perm_far only for farshuf). Training of none@8e-3,
   win-emb, win-emb-hremap: every Rev C1 key present, pred_mismatch 0, rank_hit_bad 0, logp_dp_maxdiff <= 1e-4 at every
   eval. Rev C (09ed9837fea7) vs Rev C1 on the Rev C smoke runs of none@8e-3 and win-emb (cmp_common.py): EVERY Rev C
   key of the run JSON equal (curves incl. hits / greedy answers, held_operands, exposure), models 36/36 tensors
   bitwise equal: Rev C1 adds logging only for the existing arms.
10. **A/B, behaviour of the new target and the new logs:**
   - `--target helix3_hremap` vs `helix3` (emb site, the rendered cells, seed 0): 36/36 model tensors differ;
     emb_cos_hu (h, u_h) 0.002 -> 0.259 -> 0.448 -> 0.475 (steps 0 / 8 / 16 / 23) vs the decoy pairing 0.039 -> 0.057
     -> 0.057 -> 0.056; under helix3 the same pairs go 0.002 -> -0.034 -> -0.080 -> -0.087 (decoy 0.039 -> -0.011);
     `pi_of` remap, the table asserted = helix3[remap]. (Raw cosines share a common direction at lr 8e-3: none@8e-3
     goes 0.12 -> 0.71 for the pairs and 0.16 -> 0.75 for the decoy, hence the decoy column.)
   - logp_H_one moves with training (hremap -25.43 -> -22.69 -> -21.15 -> -20.68; none@8e-3 -25.36 -> -15.93;
     rank1med 50,021 -> 1,805 / 3,892); the seen scale is logged (logp_held_b33 -31.4 -> -25.4 / -19.1).
   - `held_data_sha` identical across none@8e-3, win-emb and win-emb-hremap of seed 0 at all 4 evals (74a41c44... ->
     198b17f9...), step-0 lm 11.517814 in all three; seed 1 differs at 4 of 4 evals.
   - Fault injection (each caught by an in-run assert before or at the first eval): F5 remap table never installed
     (the placeholder stays) -> the emb site's Gram-row assert (rows of h and u_h differ) before step 0; F6 log-prob
     span off by one -> `AssertionError: ('H_one', 0, 11.345...)` (double path 11.3 nats) at step 0; F7 data sha over
     rows that are not the model's input -> the step-0 row assert; F8 remap u_h = h -> the remap assert at build.
11. **Resume and production shape:** win-emb-hremap seed 0 stopped at step 15 and resumed ("resumed from ... at step 16
   of 24") = the uninterrupted run on every key (curves incl. logp rows, data sha, emb_cos_hu; models 36/36 equal);
   only args.ckpt_every differs. --dry_inputs at the production shape (53M, rendered ARGS) of win-emb-hremap seeds 0-3:
   remap sha 662e98eb90ef / 644db0679089 / 3addf58b303b / 9d63c2f8f27b, candidates per h 262-371, |u_h - h| min
   105-115, median 287-357; CKA(helix3[remap], helix3) 0.885-0.892; emb site 1 - CKA at the 53M init 0.949-0.952;
   content scan 6,400 lines: 0 H tokens in the lines, 190-782 in the text.
12. **Read-out** (tools/ptheld_summary.py sha 6e9098f26a7c, `--selftest` on the Rev C1 seed-0 run JSON): the re-drawn
   remap = the trainer's remap_H (seeds 0 and 1, and the logged candidate counts); 11 fake scenarios PASS: S1 emb works
   with the remap mechanism (R1 win-emb HOLDS, win-text partial, R3 supports (persistent), R3b "h's own helix value");
   S2 nothing works (R1 fails, R3 against); S3 two seeds of none@8e-3 never learn seen addition -> NOT ESTABLISHED (no
   vacuous pass); S4a one LC seed at 0.30 -> HOLDS qualified; S4b two LC seeds -> "within this budget"; S5 none@1.6e-2
   best on logp by 1 nat -> bracket OPEN (lr 3.2e-2), PROVISIONAL; S5b LC missing -> PROVISIONAL; S6 one arm's data sha
   differs -> WITHHELD; S7 the 4x cell's logp +1.5 nats -> "within 4x"; S8 win-ctx 0.20, X - win-ctx 0.55 -> HOLDS with
   the R2 note; S9 X - win-ctx 0.15 -> no wall claim.
13. **Regression of the default path, final trainer 3c99f6f8cb17** (fix/reg2.sh, against the pinned outputs on disk):
   r1, r2, r3, r4, r6 IDENTICAL vs a3bce3ab63c1 (models 36/36, curves, logs, dry stdout byte-identical); r5 --domains vs
   cc984148b15a identical except dep_sha (as in test 1); x2 and x3 (the fresh audit's configs): models 36/36 equal and
   every training value equal, but read-out floats (cka_emb_helix3, cka_unemb_helix3, grad_cos) differ in the 8th digit
   from the audit's pinned run. Cause: the host, not the code -- the PINNED trainer re-run now reproduces the same
   8th-digit values, and pinned vs Rev C1 run under the same conditions are identical in every value (x3: two pinned
   runs and one Rev C1 run; x2: one pinned run vs the Rev C1 run).
14. **Generator (Rev C1)** on scratch copies (fix/gentest.py): 44 jobs, bash -n clean, the hremap cell's ARGS
   (--target helix3_hremap --schedule window --window 0.05,0.4 --site emb ...); `--confirm_seeds` -> 64 jobs (CONF =
   seeds 4-7 of the five R1 cells, 22.0 L40S-h); no SUBMITTED flag written; refused: hremap rendered with --site text,
   a CONF cell without a MAIN tag, a duplicate MAIN cell. code/COMMIT records "Rev C1, UNCOMMITTED".

## Jobs (rendered by make_jobs_ptheld.py, NOT submitted; L40S, gpu48; planned 1.1 L40S-h per 3e8-token run = 0.96 measured in ptlr + ~9% for the H evals and the in-run count)
| group | tag | runs | limit | L40S-h |
|---|---|---|---|---|
| MAIN | 53m-lr2e-3-7pct-h50-none | 4 | 03:00 | 4.4 |
| MAIN | 53m-lr8e-3-7pct-h50-none | 4 | 03:00 | 4.4 |
| MAIN | 53m-lr2e-3-7pct-h50-win-ctx | 4 | 03:00 | 4.4 |
| MAIN | 53m-lr2e-3-7pct-h50-win-text | 4 | 03:00 | 4.4 |
| MAIN | 53m-lr2e-3-7pct-h50-win-emb | 4 | 03:00 | 4.4 |
| MAIN | 53m-lr2e-3-7pct-h50-win-emb-hremap | 4 | 03:00 | 4.4 |
| MAIN | 53m-lr2e-3-7pct-h50-win-text-farshuf | 4 | 03:00 | 4.4 |
| LC | 53m-lr1.6e-2-7pct-h50-none | 4 | 03:00 | 4.4 |
| LC | 53m-lr8e-3-7pct-h50-none-wd0.01 | 4 | 03:00 | 4.4 |
| LC | 53m-lr8e-3-7pct-h50-none-wd1.0 | 4 | 03:00 | 4.4 |
| LC | 53m-lr8e-3-7pct-h50-none-4x | 4 | 09:00 | 17.2 |
| CONF (optional) | seeds 4-7 of none@2e-3, none@8e-3, win-ctx, win-text, win-emb | 20 | 03:00 | 22.0 |
44 jobs: MAIN 30.8, LC 30.4, total 61.2 L40S-h (+ CONF 22.0 if chosen). Submission (the user's call): `python
make_jobs_ptheld.py --submit` (with CONF: `--confirm_seeds --submit`).
Post-start checks (first eval of every run): step-0 lm identical across the arms of a seed; `held_operands.H` and
`remap_H` equal across arms; `held_data_sha` equal across arms at step 0; win arms prior_on_frac and prior_events as
above; grad_prior / grad_lm at the first on-eval (step 1000) logged (pt999 text window: 3.9; the emb site's ratio at
53M is unknown -- report it); `logp_dp_maxdiff_*` small on GPU.

## Audit status (experiment-audit, 2026-10-19, before submission)
i double path ✓ (counts in-run vs independent: 0/2000 differ; greedy vs teacher-forced: pred_mismatch 0; cka_cov = cka_np;
clean() = token-level; logp two paths every eval; the read-out re-draws the remap = the trainer's). ii asserts ✓ (per-step
content scan; clean() equivalence; eval-row decode; lr / wd consumed; emb prior rows; far permutation < 0.10 and table =
helix3[pi]; remap table / Gram rows; data sha = the model's input; logp known answers on CPU). iii known answers ✓
(default path bit-identical; 8 fault injections caught; decoded rows; positive control 0.995 on the H_one rows) --
result-level positive / negative controls only after the runs. iv - (no results).
v ✓ fresh-agent adversarial audit (2026-10-19, of Rev C 09ed9837fea7): code and data isolation clean (independent leak
scan, two more regression configs, a --wd A/B, the positive control); 9 findings, all accepted and fixed in Rev C1 /
this file:
| # | finding (severity) | fix |
|---|---|---|
| 1 | R3 cannot discriminate: farshuf shuffles the SEEN numbers too, so pi(h) + y is not the prediction (HIGH) | target helix3_hremap + MAIN arm win-emb-hremap; R3 rebuilt on u_h + y; farshuf = the geometry control of win-text (R3c), its pi(h) + y share descriptive |
| 2 | R1(b) / (d) can pass vacuously: a control seed that never learned seen addition scores ~0 on H (MED-HIGH) | the informative-seed gate (held6 >= 0.80, not diverged, not unstable) for every Y and LC seed; < 3/4 informative -> NOT ESTABLISHED / PROVISIONAL |
| 3 | lesson-c bracket at the floor: exact match is ~0 for every none cell (MED) | logp_H1 (+ first-token log-prob and rank) logged in-run with a double path; bracket and 4x trend on it with pre-registered margins |
| 4 | strata are mostly a magnitude axis (Spearman -0.86) (MED) | R4: partial Spearman controlling value and 4-digit share, within-band contrasts, the win-text - win-emb contrast; "win-emb rho ~ 0" no longer a null |
| 5 | R1(c) power at n = 4: Holm over 6 needs t > 4.86 (LOW-MED) | intersection-union within X + Holm over the two X (t > 3.18 / 2.35); optional CONF seeds 4-7 (user decision before any result) |
| 6 | R3 rows counted as independent (LOW) | h-level randomisation test over re-drawn decoy maps |
| 7 | pre-registration gaps: one LC seed > 0.10; final3 of the 4x cell; read-out not written (LOW) | wording for exactly one seed; final3 = last three evals (4x: 72750 / 73000 / 73241; the 4x cell has 294 evals, not 293 as written before); tools/ptheld_summary.py + selftest |
| 8 | win-emb wording: the emb prior writes h's value into its embedding (LOW) | pre-registered wording; R3b asks whether h's own value is what helps |
| 9 | data identity across arms checked only at step 0 (LOW) | `held_data_sha` every eval; equal across the arms of a seed (test 10) and checked by the read-out |
Not changed: the double text count per job (dry run + training, ~2.7 L40S-h idle over 44 jobs), accepted.
vi ✓ (train / eval table; silent filtering: kept / total rows per set logged; silent prior failure: emb rows assert,
prior_on / used fractions, remap asserts, emb_cos_hu; side effects: regression of seven default configs + --domains;
job rendering: bash -n + generator faults). vii ✓ (--dry_inputs: three training rows with the prior sites ⟦ ⟧, two rows
of every H set with H tokens ⟦ ⟧ and the scored span « », H in the text with snippets, the content scan, h with u_h).

## Open decisions (the user's)
1. LC (lesson c) costs as much as MAIN (30.4 vs 30.8 L40S-h). Without it every "none cannot" stays PROVISIONAL (R1 d). A
   cheaper subset: the 4x cell alone (17.2) or the lr / wd cells alone (13.2).
2. CONF seeds 4-7 for the five R1 cells (22.0 L40S-h): power for an uneven win-text effect; must be decided before any
   result is read (a later add-on would be a data-dependent stopping rule).
3. Shuffle: the far permutation instead of the pt999 helix3_shuf for win-text-farshuf (now a geometry control only).
4. No --untie control for win-emb / win-emb-hremap (the emb prior also shapes the output rows of H).
5. Lambda 5 at the text site (as specified) although F82 M7 found lambda 1 better at 7%; a lambda-1 win-text cell = 4.4 L40S-h.
6. The prior arms run at one lr (2e-3, conservative for the prior); "every arm at its own best lr" would add cells.
7. Rev C / C1 and the read-out are uncommitted (public repo; the agent does not commit). The snapshot runs/ptheld/code
   holds the trainer (sha 3c99f6f8cb17); commit before submission, then refresh the snapshot so COMMIT records it.

## Amendment PLR (2026-10-20, before any result of this batch was read; report/audits/rescue_standard.md criterion (b))
- Added cells (group PLR): win-text and win-emb at lr 1e-3 and 4e-3, seeds 0-3 (16 jobs; same code snapshot, only --lr
  differs from the MAIN 2e-3 cells).
- Reading: R1 stays as pre-registered (the prior arms at 2e-3). Beside it, "R1-best": each prior arm's best lr among
  {1e-3, 2e-3, 4e-3} by the mean H_one area over seeds 0-3 (ties within 0.02 -> lower offline bpb), with the bracket
  state; R1-best is evaluated with the same thresholds. A "rescue" in the sense of rescue_standard.md needs R1-best (and
  R1) to hold, none bracketed (a), the far-shuffle control failing (d), the CONF seeds agreeing (e).
- tools/ptheld_summary.py does not know the PLR tags yet: the read-out of R1-best is to be added (and tested on fake
  results) before these results are read.

## Amendment DP (2026-10-20, before any outcome of this batch was read; diagnosis notes/batch1_build/diag_ptheld_logp_double_path.md, made blind from the code and the double-path fields only)
- Finding: the two logp paths compute the same quantity (same rows, span, targets; bf16 logits under autocast, fp32
  log-softmax) and differ only in kernel shapes (path A: 250-row forward + head GEMM, pretrain_range.py:1302-1310; path B:
  8-row forward + per-position head GEMV, :1261-1270), so on GPU their logits round to neighbouring bf16 values: the
  logged differences are quantised at bf16 ulps (mostly 0.0625 / 0.125), 0.005-0.020 at step 0, flat after ~1,250 steps
  (64 runs: median 0.055, p90 0.13, max 0.75). The 0.05 tolerance came from CPU tests (logits near 0) and fails 64 / 64
  GPU runs without indicating an error. The check exists to catch a span / row / index error (signature: a whole
  token's log-prob, ~11 nats).
- Rule (replaces "logp_dp_maxdiff_H_one <= 0.05" in ptheld_summary.check_run): (i) at step 0 (near-uniform model, where a
  span error costs ~11 nats): logp_dp_maxdiff_<set> <= 0.05 for H_one, H_both and H_ans; (ii) at every later eval:
  logp_dp_maxdiff_<set> <= 0.5 nat per scored token (H_one 1.0; H_both, H_ans 1.5), i.e. at most two bf16 ulps per token
  for |logit| < 64. A failure of (i) or (ii) makes the run INVALID for every rule (stricter than needed: the hits share
  only the step-0 span check). The accuracy hits keep their own checks pred_mismatch_<set> = 0 and rank_hit_bad_<set> = 0.
  The tolerances follow from bf16 arithmetic and token counts only, not from any outcome.
- Not done (optional fp32 recompute of the final eval): not run; GPU vs fp32 argmax near-ties at the final eval are not
  measured.

### Amendment PLR — implementation note (2026-10-21, before any PLR result existed: all 16 PLR jobs still PENDING)
- tools/ptheld_summary.py `r1_best` (branch r1best -> main):
  - "H_one area" = the mean of acc_H1 over all evals of a run (evals evenly spaced every 250 steps), averaged over
    seeds 0-3.
  - Ties: means within 0.02 of the top -> the lower mean val_bits_per_byte final3; if still tied, the lower lr.
  - Bracket: closed iff 2e-3 is chosen; else OPEN at that edge.
- R1 is re-run with the two chosen cells as X on seeds 0-3 of every cell. These are the selection seeds, so the reading
  is biased by the winner's curse; it is printed as such.
- CONF: if an arm's pick is 2e-3, that X is cut to seeds 4-7, and every control reads the same seeds as R1. An X picked
  at 1e-3 / 4e-3 has no CONF seeds: "CONF seeds not run at the chosen lr". With one X left, its Holm alpha is .025.
- Guards:
  - an X / Y cell with < 3 seeds = R1 INCOMPLETE (no vacuous pass at need_wins(0) = 0);
  - the 4x trend needs >= 3 common seeds.
  Both only bind on cut views; the main R1 output on the real data is unchanged, and the diff old -> new is the one
  added R1-best line.
- Selftest:
  - B0: PLR not landed -> INCOMPLETE, R1 as in S1;
  - B1: emb best at 4e-3 = OPEN top, text best 2e-3 closed, CONF only for text; B1b: CONF trend not vacuous;
  - B2: tie -> lower bpb;
  - B3: no vacuous CONF;
  - S1-S9 unchanged.

### Amendment PLR — review fixes (2026-10-21, notes/batch1_build/audit_ptheld_r1best.md; still before any PLR result: 16/16 PENDING)
- Tie-break bpb: ptheld computed no offline bpb, so the tie-break uses the in-training val_bits_per_byte final3
  (65k tokens). This is a deviation from the amendment's wording, labelled as such in the output. A non-finite bpb
  counts as the worst (RR.worst), never as a tie winner.
- The selection score stays the pre-registered H_one area. It includes the prior window, while R1 tests final3; the
  per-lr acc_H1 final3 is now printed beside the area (shown, not used; no rule change).
- The selection view cuts only X to seeds 0-3. The controls (b), LC, bracket and 4x trend read the same seeds as R1,
  which removes the bracket wording that contradicted the main R1.
- A 4x cell present on < 3 common seeds makes a HOLDS verdict PROVISIONAL ("4x trend on < 3 seeds"); before, the
  qualifier was silently dropped.
- Rescue conjunction, defined here before any PLR result. Rescue = every part holds, with no "PROVISIONAL" and no
  "within":
  - R1 at 2e-3 HOLDS;
  - R1-best on the selection seeds HOLDS;
  - CONF agrees, defined as: the CONF verdict HOLDS; no CONF run = not met;
  - (d): the arm's shuffle control (text: win-text-farshuf, emb: win-emb-hremap) is informative in >= 3/4 of its seeds
    and < 0.50 in >= 3/4 of the informative ones.
- Known, not changed: the partial branch does not apply R2. On the main-batch data no control passes (b) on 8 seeds,
  so no HOLDS or rescue is reachable whatever PLR shows.
- Selftest: B2b (bpb, not the lower lr, decides), B4 (CONF reads seeds 4-7), B5 (nan bpb loses), B6 (4x on 2 seeds ->
  PROVISIONAL), B7 (rescue conjunction); all 22 scenarios PASS. The real-data diff vs the original read-out is still
  the one added R1-best line.
