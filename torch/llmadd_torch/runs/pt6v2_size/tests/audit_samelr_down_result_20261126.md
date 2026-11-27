# pt6v2-size SAMELR downstream: result audit (exp-auditor, fresh context, 2026-11-26; saved by the main session)

Scripts: /scratch/yh6210/tmp_audit/pt6v2_size_down_result/ (recompute.py -> recompute_out.txt)

Blocking issues: none.

## (a) Reproduction
All numbers reproduce, recomputed from the 24 per-model JSONs:
- the primary metric is re-scored from the hit strings;
- BLiMP is the mean over its 67 paradigms;
- t(0.975, 3) = 3.18245 gives a max CI difference of 4e-5 against the stored values.

15/15 verdicts agree: 9 non-inferior (190M 5, 505M 3, 1.07B 1), 6 undetermined, 0 worse. The largest |mean| among the undetermined is 0.558 (1.07B ARC-E, positive). The LAMBADA ppl ratios 1.028 / 1.052 / 1.077 reproduce.

## (b) Validity
- **Arms match the model files.**
  - all6 has six domains and prior_events [[916,1],[7324,0]]; none has '' and [].
  - lr, seed, train code and n_params match within each pair. Model size and mtime are unchanged since the eval.
  - The load check rules out arm swaps: |replica - logged| <= 1.8e-5 bpb, while the smallest between-arm difference is 8.2e-4.
- **The 1.07B preemptions left no damage.**
  - 24743956 ran in three segments of 2 models each; 24764663 scored none s18 and s19; 24764662 found all 8 done and only rewrote the read-out.
  - The segments do not overlap, and each model was scored exactly once. Models interrupted by preemption were re-scored from scratch.
  - Writes are atomic (tmp -> replace). No .tmp files, duplicates or limit != 0 files.
- **Same eval device.** Every segment ran on an A100-SXM4-80GB, fp32 with TF32 off. Double-path |dloglik| <= 7.8e-5; greedy mismatches 0.
- **Checks.** Load checks 24/24 pass; n_trunc, decode mismatch and ctx_not_prefix are all 0; every benchmark uses its full n; every KA gate passes.

## Important
"make no benchmark worse" reads as "no cost". 1.07B has 4 undetermined verdicts, with CI lower bounds of -1.88 (BLiMP), -2.01 (PIQA) and -2.25 (ARC-E). Use the quantified sentence below.

## Minor
- Training devices are mixed within some pairs (flag only; this weakens pairing and does not set a direction).
- HellaSwag is only 2.3-3.9 points above chance.

## (c) LAMBADA ppl ratio: mention it in a table or one appendix sentence
- Geometric-mean ratios all6 / none: 1.03 [0.95, 1.11], 1.05 [0.94, 1.18], 1.08 [0.95, 1.22]. All CIs include 1.
- 10 of 12 pairs are above 1. In-training val bpb is higher for all6 in 11 of 12 pairs.

## Possible true effects that were missed
- A small same-lr LM cost that the accuracy verdicts do not capture.
- 1.07B HellaSwag 4/4 wins, CI [-0.12, +0.56].

## Strongest supported statement
"At the same learning rate, the six priors change no benchmark's mean score by more than 0.6 points at any size: across the 15 size x benchmark comparisons (190M, 505M, 1.07B x five benchmarks, four paired seeds each), mean differences range from -0.45 to +0.56 points with no consistent sign (7 positive, 8 negative), none is worse, 9 are non-inferior at a 1-point margin (all 5 at 190M, 3 at 505M, 1 at 1.07B), and the other 6 have 95% CIs too wide to decide."

Scope: 300M-token models, zero-shot, same lr 6e-4.
