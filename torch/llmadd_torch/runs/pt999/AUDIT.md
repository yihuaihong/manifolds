# pt999 — unified 0..999 setup: main arms + sensitivity + layer sweep (2026-10-14)

User decisions (2026-10-14): unify all new pretraining experiments on operands 0..999; helix3 is the main target
(per-place circles = alternative, one sentence); run a cheap 53M sensitivity check (lr x0.5 / x2, batch 64,
lambda 1 / 5 / 20) comparing none / gate on operands / natural-text prior; a layer sweep for the appendix.

Code: tools/pretrain_range.py at the commit in code/COMMIT (fdfadfd since the audit fixes; a282c0b before, no job ran on it;
per-run code_sha 79e9b410dcc3 in every result). Generator: llmadd_torch/make_jobs_pt999.py.

## Questions
1. MAIN (8 seeds; 7% = 4 arithmetic lines per row, 2% = 1 line): does the self-gated helix3 prior speed up learning
   of held-out sums (both orders never trained) when it acts (a) on the operands inside the arithmetic lines (ctx) or
   (b) on numbers 0..999 that occur naturally in the FineWeb text of the row (text; never on the arithmetic lines)?
   Controls: helix3_shuf (same values, permuted rows of the target table; far permutation) at both sites.
2. SENSITIVITY (4 seeds, 7%): is the speed-up still there at lr 5e-4 / 2e-3, batch 64 (same 3e8 tokens, half the
   steps), lambda 1 / 20? One factor at a time from the base (lr 1e-3, bs 32, lambda 5, layers 2,4).
3. LAYERS (appendix, 4 seeds, 7%): prior at layer {1}, {4,5}, {1..6} vs the default {2,4}.

## Setup (shared by all arms)
- 53M OLMo-2-style model from scratch (d384, 6 layers), seq 512, 3e8 tokens, AdamW (0.9, 0.95), wd 0.1, 1% warmup,
  cosine to 0.1x. FineWeb-Edu text chunk + `arith_lines` lines appended at the end of each row. Operand-length
  buckets (1,1) .. (3,3) drawn equally often; 4 formats. Whole-number tokens (OLMo-2 tokenizer: 0..999 = 1 token).
- Held-out: per seed, unordered pairs held out per bucket; neither order is ever trained (draw_pair rejects them).
- Gate (--schedule gate): prior on while min over the 6 buckets of the EMA(0.98) of in-batch first-line exact match
  (answer tokens + newline, teacher forced) is < 0.5. Same gate for ctx and text sites.

## Metric: held_bXY (main), seen_bXY, ood_b4X
- Definition: exact match of all answer tokens AND the newline, teacher forced, greedy argmax at each position;
  implementation evaluate() in pretrain_range.py. Mean over n_eval = 150 problems per bucket (b11: 37, all held).
- Measures: whether the model computes sums it never saw (held) at the position where training lines sit.
- Token position: the eval line is placed at the END of a 512-token row after a val-text chunk (same as the last
  training line; the first training line also follows text). Scored positions = the answer tokens and newline.
- Derived: t90 = tokens until held_bXY >= 0.9 (per bucket); area = mean of the 6-bucket average over the curve.
- Expected (F75, whole-run prior, A100/H100): none area 38 (bimodal 15-54), helix3 whole-run 88; gate expected
  close to whole-run. Text site at 0..999 unknown (10..99, pttext: about 5x faster to 90% than none).

## Metric: val_bits_per_byte (LM cost)
- CE on 8 x 16 fixed val windows (65k tokens) / bytes; comparable across arms. F75 whole-run cost +0.012-0.022.

## Metric: cka_ops_helix3 (transfer read-out, every arm incl. none)
- CKA (z-scored columns, centred Grams) between hidden states at the operand tokens in the lines and helix3
  features of their values, at geo_layers. For text-site arms this says whether text-number geometry transfers to
  the operands in the arithmetic lines.

## Train vs eval conditions
| dimension | train | eval | same? |
|---|---|---|---|
| sequence position | 4 (or 1) lines at the end of a 512-token row after text | one line at the end after val text | yes for the last line |
| template | 4 formats, random | format 0 ("Q: a + b = ") | subset |
| value range | operands 0..999, buckets balanced | same buckets; ood_b4X = 4-digit (never trained) | yes (ood is labelled extrapolation) |
| data / split | train text; held pairs never in lines | val text; held pairs | disjoint by construction |
| model state | train, bf16 autocast | eval, bf16 autocast | yes |

## Controls
- Positive: whole-run helix3 ctx (F75) must reproduce a large gain; ceiling: seen_bXY near 1 at the end.
- Negative: helix3_shuf at both sites (should not give the clean gain; F77 showed shuffled targets can be bimodal).
- Hardware: all pt999 jobs run on L40S; compare arms only within pt999 (A100 vs L40S runs diverge from ~step 250).

## Known limits
- Sensitivity / layer cells have 4 seeds (enough for large effects only; none is bimodal).
- The text filter keeps list indices ("4. Ultrasonic ...") and units ("20kHz"); it drops decimals, ranges, dates,
  times, exponents, digit groups and numbers glued to "." "#" "$".

## Audit 2026-10-14 (fresh general-purpose subagent, read-only; scripts in scratchpad/audit_pt999) — fixed before any job ran
Verdict: no silent mis-measurement in the training / eval code (eval and gate alignment, ctx / text sites, split,
resume bit-identical, 140 sbatch files with unique result and checkpoint paths). Items and what was done:
1. BLOCKING for analysis (not for training): tools/ptrange_summary.py groups by tokenization/target/schedule and
   overwrites seeds -> it would merge ctx / text / 2% / sensitivity cells. NEW tools/pt999_summary.py: one cell per
   --tag, asserts (tag, seed) unique, n per cell, paired vs the matching none cell; checked on fake results.
2. IMPORTANT: no in-batch positive control (F75 ran on A100/H100). ADDED 7pct-whole-ctx (helix3 whole run, 8 seeds).
3. IMPORTANT: text sites are 43% values 0-9, 44% 10-99, 14% 100-999 (each 3-digit value ~0.02 times per batch);
   helix3 vs value-identity CKA on text batches 0.375; shuffled-vs-true 0.17-0.28. => report the text arms PER BUCKET;
   gate-text vs shuf-text isolates only the geometry beyond "values are distinguishable". Logged now: text_bins.
4. IMPORTANT: tied embeddings — the prior's gradient reaches the number tokens' embedding rows, which are also the answer
   classifier rows. ADDED: cka_emb_helix3 / cka_unemb_helix3 at every eval; held_b13/b23/b33 split by sum < 1000 (one
   answer token) and >= 1000 ("1xx" + digit); 7pct-untie-none / 7pct-untie-gate-ctx (4 seeds) as the control.
5. IMPORTANT: bs 64 needs ~45 GB for the fp32 logits on a 48 GB L40S (the 505M model did OOM at bs 32, ptsize job
   19782846). FIXED: LM loss in 8192-token chunks with recomputed logits when the fp32 logits would exceed 8 GB (only
   bs 64 here), gradient checkpointing above 150M parameters (ptsize999). CPU checks: see below.
6. minor: the prior's gradient dominates the clipped update when on (init: text 15.8 vs LM 3.7; ctx 4.8). Main
   comparisons: gate vs shuffled at the same site (same dilution) as well as vs none; report grad ratios.
7. minor: 7% eval line follows text at ~500 while in training position ~500 is the 4th line (after 3 lines).
   Planned offline check with the saved 7pct models (eval line at the first-line position + 3 seen lines).
8. minor: layer sweep changes per-layer weight (mean over layers). FIXED the read-out: cka_ops_helix3 at fixed layers
   (1/3, 2/3 of the depth) for every arm.
9. minor: bs 64 evals every 8.2M tokens and a 2x longer gate EMA. FIXED: evals and EMA on a token scale (bs 64: every
   125 steps, EMA 0.98^2); bs 32 unchanged.
10. minor: natural text contains 13-17 of the 20 b11 held-out pairs per seed as "a + b = c" (1-2 exposures each vs ~5800
    for trained pairs; b12 2-8/150; b13+ ~0). => "never trained" is not strictly true for b11; say so.
11. minor: shuffles are plain permutations (CKA to true: whole table 0.01, ctx batch 0.09-0.12: far in practice) and
    also break the digit-count grouping. ADDED 7pct-lenshuf-ctx (helix3_lenshuf gate, 4 seeds).
12. minor: "7%" = 8.5% of tokens (1 line = 2.1%); b11 n = 35-39 by seed; text filter kept letter-glued numbers ("B1",
    "VO2", ~19% of sites next to letters). FIXED: numbers glued to a preceding letter are skipped.
13/14. minor: CKA Grams in bf16 under autocast (~1e-3 on GPU); run_guard exits 0 if squeue fails (unreachable here).
Snapshot: code/ refreshed while all jobs were PENDING; every result JSON now carries code_sha (sha256 of
pretrain_range.py, first 12 hex) to pin the version per run.
CPU checks of the fixes (tiny model, bs 32 x seq 128, scratchpad/test_audit_fixes.sh): default ctx path vs the old code
bit-identical (155/155 values, same eval steps); chunked LM loss vs one logits tensor: loss equal to 1e-6 at step 0,
largest later difference grad_prior 1% after 19 steps (trajectory drift from summation order); --untie doubles the
embedding parameters and separates emb / unemb CKA; bs 64 evals every 2 steps (5 x 32 / 64) with EMA 0.9604; text site
logs value bins (dry run at seq 512: 0-9 / 10-99 / 100-999 = 0.39 / 0.45 / 0.16; median 120 numbers per batch, was 128
before the letter-glued filter). Gradient checkpointing on vs off: bit-identical curves (ctx and text, bs 4).
New arms submitted at fdfadfd: 7pct-whole-ctx (8 seeds), 7pct-lenshuf-ctx, 7pct-untie-none, 7pct-untie-gate-ctx (4 each).
