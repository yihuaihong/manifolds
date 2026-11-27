# pt6v2 Amendment V9 result audit (exp-auditor, fresh context, 2026-11-25; saved by the main session)

Scripts / outputs: /scratch/yh6210/tmp_audit/pt6v2_v9_result/ (indep.py -> indep_out.txt, dose.py -> dose_out.txt, prewin.py)

## Reproduction
All numbers reproduce, recomputed from the 32 run JSONs (dim strict mask = audited v4final.dim_strict):

| quantity | value |
|---|---|
| area all6 - shuf | +18.05 [+10.40, +25.70], 8/8 |
| h_shuf | -4.55 [-14.60, +5.51], 3/8 |
| h6 | +13.50 |
| h4 | +1.49 |
| final3 all6 - shuf | +34.51 [+21.17, +47.84], 8/8 |
| dose: grad_prior median ratio | 0.1298 / 0.1428 = 0.909 |

The 4 arms x 8 seeds all finished (75 evals each). Within a seed, every run has the same held sets, strict mask, eval steps, per-eval data_sha and code a398fccbfe93. The args differ only in prior_shuf / prior_domains / target.

## Does the control isolate what it claims? Yes
- **Shuffle:** a permutation of the true target (pretrain_range.py:1151, F[qperm]; qperm = rng.permutation via far_perm :363). The training loss and geo use the shuffled target; eval CKA uses the true target.
- **Effect at window end:**
  - CKA to the shuffled target is about 0.99, to the true target 0.11 / 0.08; all6 reaches 0.996 / 0.998 on the true target.
  - torus / box5 final3 drop to 0.3 / 1.1.
- **Gradients are comparable:**
  - Pre-clip applied-term norms, taken at the same eval step and batch; clip_coef = 1 throughout.
  - At the first prior-on eval the ratios are torus 1.28x, box5 0.84x, total 1.02x; the 1.4-1.7x seen at step 0 is outside the window.

No blocking issue.

## Important
1. **18.0 is not the size of the geometry effect.** The shuffled target is a contradictory geometry: box5's natural CKA drops from 0.74 to 0.08. shuf - all4 = -6.0 [-15.8, +3.7], 2/8, which is n.s. The net geometry gain over having no torus/box5 priors is all6 - all4 = +12.0 [+3.8, +20.2], 7/8 (final3 +22.0). Use 12.0 if the text gives a geometry effect size.
2. **Dose is uneven per domain.**
   - Total 91% (85-103% cumulative); box5 73% (0.50-0.98 per seed); torus 98%.
   - Across seeds, the dose ratio correlates with all6 - shuf at r = -0.77: the s20-23 seeds (ratio >= 0.95) give +12.2 (4/4), the rest +23.9.
   - The correlation comes from the all6 side (within all6, dim vs its own grad_prior r = +0.71). Within shuf, more gradient goes with lower dim (r = -0.66), so it does not support a dose reading. State box5 = 73% in the methods.
3. **Devices are mixed.**
   - shufTB: s16 A100, s17-18 H100, s19-23 B200. all6 / only-dim: B200 (s23 A100). all4: A100/H100.
   - Same-seed runs diverge by step 250, before the prior is on, so pairing shares only the init and data order; the t-CI stays valid.
   - On the device-consistent s19-22: all6 - shuf +13.5 (4/4), h_shuf +4.0, close to h4 +2.8, so the reading is still geometry.
   - The negative h_shuf is driven by s16 / s23. Do not write "the shuffled prior is harmful".

## Minor
- Each batch's shuffled spectrum equals the true one only in distribution.
- The gcos_dim|torus/box5 logged in the shuf runs refers to the true (unapplied) term.
- The effect is attributable to torus + grid jointly only.

## Supplementary
The gap opens after the window. Dim area by arm (all6 / shuf / only-dim / all4):

| | all6 | shuf | only-dim | all4 |
|---|---|---|---|---|
| after the window | 78.8 | 53.1 | 58.8 | 61.3 |
| inside the window | 87.2 | 79.5 | 82.8 | 83.0 |

## Strongest supported statement
Shuffling the torus and grid targets (same lambda and target spectrum, 91% of the total prior gradient) removes all of the joint lift on dimensions:
- shuffled vs dims prior alone: -4.5 [-14.6, +5.5], where a dose explanation predicts +13.5;
- true vs shuffled: +18.0 [+10.4, +25.7] (8/8) averaged over training and +34.5 [+21.2, +47.8] (8/8) at the end;
- extra prior gradient alone brings no gain: shuffled has 1.36x all4's prior gradient yet is -6.0 [-15.8, +3.7] relative to all4.

Qualifiers to state together: box5 gradient 73%; the effect is attributable to torus + grid jointly; devices are mixed, but the device-consistent seeds agree.
