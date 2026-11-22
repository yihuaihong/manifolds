# Result audit: pt6v2 Amendment V5 probe, V5-S1 to V5-S3 (exp-auditor, 2026-11-20; read only, no reruns; saved verbatim by the coordinator)

## Summary: two numbers in S1 do not reproduce, S2 mostly reproduces, S3 holds

Inputs: only the existing JSON and OOF npz files. Nothing was rerun and no project file was changed. Scripts and outputs are in /scratch/yh6210/tmp_audit/pt6v2_v5/ (check_v5.py, check_v5_transport.py and their *_out.txt).

Terms:
- **all6**: the arm with all six priors on.
- **only-dim**: the arm with only the dim prior on.
- **strict**: the held questions selected by strict_mask (58 for s16, 77 for s17).
- **transport residual R²**: the answer information at the answer position beyond Ê(a)±Ê(b).

## Verdicts

### 1. Important: S1, behaviour part. s16 reproduces; the "16" for s17 does not (mislabelled)

- **s16** reproduces. In the probe forward on the 7323 checkpoint the strict gap goes from +1.7 to +25.9 at 18309, so DiD = +24.1. The logged eval at 7250 also gives +24.1.
- **s17's "16" is not computed from 7323.** In v5pair.py, `at()` takes the last eval at or before 7323, which is 7250. From there the gap goes +6.5 → +22.1, so DiD = +15.6.
- **On the real 7323 checkpoint** (the one the representation DiD uses), s17's behaviour DiD is only +3.9. The gap is already +18.2 at 7323, because only-dim is at 76.6% on that checkpoint, against 85.7% at 7250 and 79.2% at 7500.
- **The baseline moves this number a lot.** A single eval on 77 questions jitters by about ±4 points. With 7500 as the baseline the DiD is +10.4; with 7000 it is +23.4.
- **Fix, either of:**
  - "24 and 16 points (DiD from the last prior-on eval, step 7250, to 18309)";
  - or, more stable, a window baseline: "21 and 18 points (DiD from the mean of the last four prior-on evals to step 18309)". With the mean of the last 3 evals as the end point instead, it is +21.0 / +20.8.
- **Also state that these are V5 reruns.** The registered same-seed pairs give +12.1 / +10.4, so the s16 V5 effect is about twice as large.

### 2. Important: S1, representation part. "at every layer … 0.96–0.99" does not reproduce

- At 18309 (4 runs × strict_A / A / pooled), item R² is 0.870–0.917 at L0 and 0.900–0.984 at L1. Only L2–6 fall in 0.962–0.990.
- Change it to "at layers 2–6 in both models (ridge R² 0.96–0.99 at step 18309)".
- "DiD within ±0.012 at L2–6, both seeds" reproduces: max|DiD| = 0.0116 (s17 strict_A L6), and all 30 cells are ≥ 0.
- Because s17's behaviour DiD is only +3.9 on the same checkpoint, pair s17's "behaviour drops, representation does not" with the 7323→12000 segment as well. There the behaviour DiD is +13.0 (gap +18.2 → +31.2), and the item DiD at L2–6 is +0.000 to +0.007.

### 3. Minor: S2 reproduces, with one phrase slightly too broad

- **Strict questions:** R² at L5–6 is 0.965–0.989, and nearest-lattice accuracy is 100% in 12/12 cells.
- **Strict questions only-dim answers wrongly** (s16 n = 2 / 12 / 17, s17 n = 18 / 38 / 31): R² 0.957–0.992, lattice 100% in all of them.
- **71% / 60%** reproduces (70.7 / 59.7); the probe forward and the logged evals agree.
- **Scope of "the questions it answers wrongly".** If that means all misses (held + seen, up to 120 questions), only-dim s17 L6 @12000 has R² 0.956 and lattice 99.2% (1 wrong row out of 120). So write "on the strict questions it answers wrongly", or set the threshold at R² ≥ 0.95. Everything else can stay as written.
- **Checked the other way:** on miss rows the decoded lattice point equals the correct answer c in 100% of cases. It equals a only when c == a (identity questions), so the decoder is not copying the input.

### 4. S3 holds and does not need weakening

- **The main alternative is ruled out on the miss rows.** That alternative is "' is' only carries a and b". On only-dim's miss rows, the transport residual R² is +0.53 to +0.97 in both seeds, both ops and every cell with n ≥ 10. This matches the hit rows (+0.51 to +0.97).
- **The loss is located at the state→token readout.** The answer is a single word token plus "\n" (scored at positions 510..511), and the logits at ' is' predict that word. After L6 there is only the final norm and the lm_head. So "the answer is present at L5–6 but the output token is wrong" logically places the loss in the state→token readout.
- **Wording:** "not a loss" is right; "unchanged" is not. A frozen decoder fitted at 7323 drops to R² 0.69–0.76 at 18309 in both arms, so both arms re-encode to the same degree.

## Effects that may be missed or wrongly included

- **The item L0–L1 DiD is negative, the opposite direction.** It is −0.03 to −0.046 for s17 and −0.004 to −0.007 for s16 L1, so all6 loses more in the early layers. It is small and is not part of the conclusion, but keeping "every layer" would imply DiD ≈ 0 at L0–1.
- **The frozen-decoder DiD has opposite signs in the two seeds** (s17 L5–6 +0.05 to +0.06, s16 L6 −0.04), so it is not an effect.
- **Answer position L4: only-dim is above all6 at all three checkpoints.** For s17, R² is 0.91–0.93 against 0.71–0.75, and lattice accuracy 0.93–0.96 against 0.53–0.63. This already exists during the window, so it is not a post-window effect and has nothing to do with S1–S3.

## Bottom line

- **Most likely mis-measurement:** S1's s17 "16 points, DiD from step 7323" is in fact computed from eval 7250. On the 7323 checkpoint that the representation DiD uses it is only +3.9, and depending on which eval is the baseline the number moves between +4 and +23.
- **True effects that may have been measured or written away:** none suppressed by a threshold or a control. The window-mean baseline gives a larger s17 effect (+18 / +21) than the 16 in the sentence. The transport confound is ruled out on the miss rows, so S3 can be written more strongly than in the draft.
- **Strongest conclusion the evidence supports:**
  - In the V5 reruns (seeds 16, 17), after the prior window closes only-dim loses 21 and 18 more strict-dim points than all6 (mean of the last four prior-on evals → step 18309).
  - Yet in both models the item tokens keep linearly decodable lattice coordinates at layers 2–6 (ridge R² 0.96–0.99 at 18309; all6 − only-dim DiD 0.000–0.012).
  - The ' is' state at layers 5–6 decodes the correct answer at all three checkpoints (R² ≥ 0.96, 100% nearest-lattice on strict questions). This includes the strict questions only-dim gets wrong (n = 17 and 31 at 18309; 100%), where the state carries answer information beyond the transported inputs (residual R² 0.6–0.95).
  - So the post-window loss sits in reading the token out of a state that still holds the answer.
  - Where the draft S1–S3 is weaker: it does not use the transport residual evidence, which turns S3 from an inference into a direct measurement.

(Coordinator check, out_v5/v5win.py: window-baseline behaviour DiD s16 +21.6 (final3 +21.0), s17 +18.2 (final3 +20.8), matching the auditor.)
