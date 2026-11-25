# pt6v2_size seed-16 result audit (190M / 505M / 1.07B), 2026-11-24

exp-auditor (fresh context), result layer. Saved by the main session because the auditor is read-only. Scripts and outputs
are in /scratch/yh6210/tmp_audit/pt6v2_size_s16/: indep.py, noise53.py, reach53.py and their *_out.txt.

## Reproduction
- **Recomputed independently** from the run JSONs, without sizeread.py or v4final.py; the strict dim set was rebuilt (n=58).
- **Reproduced:**
  - every number in sentences 1–3;
  - the area numbers in sentence 5;
  - sentence 6 for all6.
- **Sentence 4 has two errors** (see Important 1–2).
- **Data consistency:**
  - every run has 75 evals and ends at step 18309;
  - split seeds and data_sha sequences are identical across arms within each size;
  - no s16 main restarted or hit an OOM.

## Blocking (type 3: a cell cannot answer its question)

### 1. 1.07B none grid
The 1.07B none grid {1.2e-3 .. 9.6e-3} sits one step lower than the grids at the other sizes. As a result it can measure neither the dim/box5 magnitudes nor the cities/countries signs (indep_out.txt:83-98).

**Evidence:**
- For 5 of the 6 manifolds, the none area optimum is at the top edge, 9.6e-3.
- dim is at the floor at all four lrs: area 4.4–10.0, final3 ≤ 10.9. For reference, the heuristic floor is 14.3 and chance is 3.7.
- box5 final3 is ≤ 26.6, against 51–55 at 505M at the same lrs.
- cities and countries none final3 rise monotonically with lr (86.2→96.1 and 87.9→97.2).

**Affected numbers:**
- sentence 1: dim +78.8, box5 +59.2, and torus +18.7 (its optimum is also at the edge);
- sentence 2: cities +0.84, countries +0.50;
- sentence 3: cities −2.0, countries −2.3;
- sentence 4: 1.07B 7–51%.

**Unaffected:**
- numbers +50.9 (its optimum, 4.8e-3, is interior);
- the sign of the dim and box5 gains: all6 area is 88.8 / 84.0, while no none run at any size or lr exceeds 44.0 / 44.4.

**Next:** re-read after none@1.92e-2 and 3.84e-2 (REP 1B). Until then, write no 1.07B magnitudes and no cross-size magnitude comparison.

### 2. Sentence 5, "joint lift on dim shrinks with size"
The comparison is saturated and within seed noise, so it cannot carry this sentence.

**Evidence:**
- The area differences reproduce: +4.57, +3.20 and −1.84.
- At 190M and above, only-dim no longer loses dim after the window. Its post-window mean is 95.3 / 96.3 / 98.1, against 82.5 at 53M seed 16.
- final3 is near the ceiling in both arms.
- At 53M, the 8-seed sd of the all6 − only-dim dim-area difference is 7.3 (range +2.8 to +25.6; s16 is +4.3). That sd is larger than the whole cross-size spread of 6.4.

**Can write:** at seed 16, only-dim keeps dim at 190M–1.07B (final3 95–99). So the 53M joint advantage (final3 +25.0, 8 seeds) does not appear at these sizes. "Shrinks with size" cannot be written.

### 3. Sentence 6, bpb
lr is confounded with the prior here.

**Evidence:**
- The prior arms run at 6e-4, below every none lr.
- At 505M and 1.07B, none bpb rises monotonically with lr.
- only-dim is also below every none run.
- At 53M at the same lr (2e-3), all6 − none bpb is −0.0004 over 8 seeds, with 7/8 positive.
- At 505M, all6full (1.1108) is above none@2.4e-3 (1.1070).

**Can write:** "no LM cost is visible: all6 bpb is 1.125 / 1.096 / 1.084, below every tested none lr (best 1.141 / 1.107 / 1.092)". Do not write that the priors lower bpb.

## Important
1. **505M reach in sentence 4:** with the running-3 definition, reach is 0.341 / 0.082 / never / never / 0.382 / 0.096, so the range is 8–38%, not 7–38%. The 7% is dim's single-eval reach.
2. **"As at 53M" in sentence 4 compares two definitions.**
   - The 53M 8–30% (F105 补记二) uses 8-seed mean curves, all6@1e-3 and the none@N_m final3 as threshold. It reproduces as 30 / 8 / 14 / 10 / 27 / 10.
   - With this batch's definition (per seed, running-3, best-final3 none), 53M all6@1e-3 gives 10–56%.
   - So 190M's 8–30% is not slower than 53M; either compare like with like or drop the clause.
3. **The cities/countries counts (4/6, 5/6, never) are single-seed noise.** They rest on differences of 0.3–2.5 points.
   - At 53M, the cities final3 seed sd is 1.16 for all6 and 1.2 for best none.
   - Within a run, the sd over the last 10 evals is 0.2–1.3.
   - At 53M, all6@2e-3 also never reaches on cities in 1/8 seeds.
   - Settle these with the REP seeds.
4. **Grid edges at 190M and 505M:** 190M numbers is at the low edge and torus at the high edge; 505M dim is at the high edge and still rising (34.8→42.1). REP closes these. 505M dim +49.1 may shrink a few points; its sign won't change.

## Minor
- Devices are mixed across arms at 190M/505M; each run uses one device.
- The check.json failures are all expected:
  - the step-0 lm reference is 53M's;
  - offline_lm is null (not computed);
  - the 1.07B none decomposition double path is 0.076 against a 0.05 tolerance. This is diagnostic only: the code_1b diff touches only the diagnostic, and training is unchanged.
- The prior lr, 6e-4, is untuned, so the gains are more likely lower bounds.

## Possibly missed true effects
- **cities/countries area gains:** positive at all three sizes (+1.0 to +5.9, 6/6), consistent with 53M 8/8. The single-seed final3 counts reduced them to "4/6" and "never".
- **Sentence 5:** it turns "only-dim stops forgetting dim at ≥190M" into "the joint lift shrinks".
- **Sentence 4:** "as at 53M" makes 190M look slower than it is.

## Strongest supported statement
At 190M and 505M (seed 16; prior lr 6e-4, untuned), the six priors raise the curve area on all six manifolds above the best no-prior lr:

| manifold | 190M | 505M |
|---|---|---|
| numbers | +51 | +49 |
| dim | +48 | +49 |
| box5 | +41 | +42 |
| torus | +19 | +15 |
| cities | +5.9 | +1.0 |
| countries | +3.7 | +2.6 |

- numbers, dim, torus and box5 also end higher (+1.7 to +55).
- At 190M every manifold reaches the best no-prior final after 8–30% of training, not slower than 53M under the same definition.
- At 505M, four manifolds reach it within 8–38%.
- At 1.07B: numbers +50.9. The signs for dim, box5 and torus hold; their magnitudes await none@1.92e-2/3.84e-2.
- No LM cost is visible at any size.
