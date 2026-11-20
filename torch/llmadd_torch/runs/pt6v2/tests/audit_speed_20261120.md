# Result audit: pt6v2 post-hoc speed / transfer sentences S1–S7 (2026-11-20)

Auditor: exp-auditor (result review, read-only). I recomputed every number independently and did not import speed*.py or repr_check.py.
- Scripts: /scratch/yh6210/tmp_audit/pt6v2_speed/audit_speed.py (S1–S5), audit_repr.py (S6–S7) and audit_lib.py (the loader, i.e. the first 115 lines of audit_speed.py).
- Outputs: audit_speed_out.txt and audit_repr_out.txt in the same directory.

Sanity checks:
- 112 runs and 75 evals each were loaded. Prior-on evals are 1000–7250 in every prior run and absent from every none run.
- My areas match the runs' own metrics to 4e-14. My dim strict areas match readout_V3 to 1e-14.
- N_m reproduces over all 6 none lrs.
- pt6comb dim h matches readout.json to 1e-14.

## Verdicts

| | numbers | wording |
|---|---|---|
| S1 | reproduce: 3.94 / 4.4 / 2.88 / 3.14 / 4.1 / 5.0; per-seed median 3.77 / 3.20 / 2.00 / 2.67 / 3.61 / 4.67; at 90% the fastest none lr = N_m in all six | **change**: see below |
| S2 | reproduce | **change**: all6 is also below none@3.2e-2 |
| S3 | reproduce (segment means) | **change**: name the definition and drop box5 from "mainly after" |
| S4 | reproduce | correct |
| S5 | reproduce; Welch p 0.005 | **change**: "the two swapped companions" |
| S6 | reproduce exactly (default key = mean of L2, L4) | **change**: "cross-family ≤ 0.06" holds only at the layer average |
| S7 | reproduce | CKA part correct, not saturation; **gcos part weak by construction** |

### S1
**What the baseline is.** The "chance" in speed.py is the FLOOR-line baseline, max(chance, heuristic): cities 67.2 (ansfreq), countries 71.8, dim 15.5.
- Either call it "above the heuristic floor", or use true chance (cities / countries 50, dim 6.0). With true chance, cities becomes 2.57× (T 87.5; step 1750 vs 4500) and countries 2.14× (T 89.7; 1750 vs 3750), so the range is 2.1–5.0×. The other four manifolds do not change.
- At 90%, the fastest none lr is N_m under both baselines.

**"Best none lr" means the area-best lr (N_m).** For cities and countries the final-best lr is 1.6e-2. Against it the ratios are larger (5.1 / 5.7 with the FLOOR baseline, 3.1 / 3.7 with true chance), but the fastest none lr is then 4e-3, not the reference.

**Grid resolution.** At 250-step resolution each ratio sits at the lower edge of its bracket:
- brackets: numbers 3.88–4.19, dim 4.2–5.5, cities 2.75–3.29, countries 3.0–3.67, torus 4.0–4.56, box5 4.86–5.83;
- interpolated ratios: 4.01 / 4.38 / 3.13 / 3.32 / 4.28 / 5.64.

So the stated numbers are conservative.

**Per-seed median.** It drops the seeds where none never reaches T: numbers 2, cities 1, countries 1, box5 1. Counting those seeds as infinitely slow gives 2.25–5.17×, so dropping them is also conservative.

### S2
- "98% at window close" is the first eval after the window, step 7500 (97.98). The last in-window eval is 7250 (97.07), and the peak is 98.2 at 7750.
- The best none mean-curve peak is 94.6 (1.6e-2). Per seed, all6's peak beats the best none peak over all 6 lrs in 7/8 seeds.
- final3 is 90.36. That is below none@1.6e-2 (−3.0, all6 higher in 1/8) **and below none@3.2e-2 (−1.1, 3/8)**. It is above 2e-3 / 4e-3 / 8e-3 (+74.2 / +35.2 / +6.0) and above 6.4e-2 (+16.2).
- Change the sentence to "below the two best none lrs (by 3.0 and 1.1)".

### S3
**Name the definition.** The numbers are segment means: in = 26 evals (1000–7250), after = 45 evals (7500–18309). F105 uses contributions (sum/75), which give dim +1.5 / +12.0, torus −0.1 / +4.5 and box5 +1.8 / +5.6. F105's own note forbids mixing the two definitions, so the sentence must name which one it uses.

**dim and torus support the claim:**
- dim: after-window +19.9 (8/8); final3 +25.0 (8/8); area h +13.5 [+7.4, +19.6].
- torus: after-window +7.5 (6/8); final3 +11.7 (8/8); area h +4.4 [+0.6, +8.2].
- Drops from 7500 to final3, only-m vs all6@2e-3: dim 32.9 vs 12.4, torus 21.9 vs 9.8. Paired, only-m drops more in 8/8 seeds for both.

**box5 does not.** After-window +9.3 (4/8), final3 +9.5 (5/8), area h +7.4 [−2.0, +16.7]. The drops are 29.5 vs 27.5, about equal. Take box5 out of the "mainly after" list, or say it explicitly as "box5 noisy, 4/8".

### S4
- all6@1e-3 − none@2e-3: mean +0.0035 (+0.29%), median +0.0046 (+0.38%), higher in 7/8 seeds.
- The stronger statement uses the same lr. all6@2e-3 − none@2e-3 has mean −0.0004 (−0.03%) and median +0.0038, higher in 7/8.
- all6@1e-3 has lower bpb than none@4e-3 (1.2133 vs 1.2150).

### S5
**Both numbers match the readouts:**
- pt6v2 h +13.50 [+7.37, +19.63];
- pt6comb h +1.59 [−4.42, +7.60], 95% t.

**Wording.** The companions in both batches also include numbers, cities and countries. Only S4 and people were swapped for torus and box5, so say "the two swapped companions".

**Additions worth making:**
- Welch p 0.0054; Mann-Whitney p 0.010.
- only-dim is unchanged across batches (area 63.5 vs 64.1). The change is in the all6 arm (65.1 → 77.6).
- In pt6comb, h is small both in-window and after (+0.9 / +2.1, segment means).

**Remaining confounds:** trainer Rev G vs Rev H, seeds 8–15 vs 16–23, and the device mix.

### S6
**Site.** `cka_<d>` is exactly the mean of `cka_<d>@L2` and `cka_<d>@L4` (max diff 0 over all evals; pretrain_range.py:1934–1941). These are the item-token sites of the training-template eval prompts. read_layers = geo_layers = [2, 4] (pretrain_range.py:1318–1319), so these are the layers the prior's 1 − CKA loss is trained on (:2249–2258).

**Tokens.** The nonce tokens of dim, box5 and torus are pairwise disjoint (0 overlap). So the transfer is across distinct tokens, not the same embedding rows being trained.

**Transfer by layer** (only-X − none@2e-3 at step 7500; final3 in parentheses where given; all 8/8 seeds unless noted):

| pair | layer average | L2 | L4 |
|---|---|---|---|
| box5 → dim | +0.159 | +0.080 | +0.239 |
| dim → box5 | +0.183 | +0.118 | +0.247 |
| cities → countries | +0.222 | +0.128 | +0.316 |
| countries → cities | +0.237 | +0.189 | +0.285 |
| torus → dim | +0.084 (7/8) | +0.022, n.s. | +0.146 (7/8) |
| torus → box5 | +0.090 (7/8) | +0.045, n.s. | +0.135 (7/8) |

- dim ↔ box5 and cities ↔ countries survive at both layers, and the effect is about 3× larger at L4.
- Torus → dim / box5 is an L4 effect only.

**Cross-family effects:**
- Layer average: the largest are countries → box5 +0.062 (6/8) and countries → dim +0.064 at final3 (8/8), so "≤ 0.06" is borderline.
- L4: cross-family reaches +0.10–0.11 (countries → dim / box5) and numbers → dim +0.076.
- Within-family transfer is still larger at every layer.

**"Nothing raises torus"** holds at all layers: the maximum is +0.046 (4/8), and only-numbers lowers torus (−0.04 to −0.05, 1/8).

**Wording.**
- Write "cross-family ≤ 0.06 at the layer average (≤ 0.11 at L4)".
- "Similar geometry" cannot be separated from "similar task": dim and box5 are both add / subtract on exponent-vector lattices with two inverse ops. "Priors on structurally similar domains" is the exact description.

### S7
**Numbers.** final3, all6@2e-3 vs only-d:

| domain | all6 | only-d | paired diff [95% CI] |
|---|---|---|---|
| dim | 0.975 | 0.971 | +0.0045 [−0.003, +0.012] |
| torus | 0.977 | 0.975 | +0.0024 [−0.005, +0.010] |
| box5 | 0.968 | 0.969 | −0.0014 [−0.010, +0.008] |

L2 and L4 separately give the same result.

**Not saturation.** Both arms fall 0.018–0.029 from 7500 to final3 (seed SD about 0.007). Within runs, CKA moves with accuracy after the window (Spearman +0.45 to +0.82). Equality is bounded at about ±0.01 on a scale where none sits at 0.15–0.67. So "retained equally at the CKA site" is a real, bounded equality.

**Scope.** CKA at this site does not rank accuracy across arms:
- dim: none@3.2e-2 has CKA 0.20 at 53.5% accuracy, while only-dim has CKA 0.97 at 52.3%;
- torus: none@1.6e-2 has CKA 0.10 at 93.3%.

So the sentence "not a loss of geometry at the CKA site" is correct. It must not be widened to "not a representational loss".

**The gcos sentence is weak by construction.** It is measured in all6, where every prior is already converged:
- gnorm_true is about 13× smaller than in the none model (dim 0.066 vs 0.84);
- the mean cosine at the first in-window eval (step 1000) is +0.005.

So near-zero cosines are expected for any pair. "≤ 0.05" is true only as a mean over 8 seeds × 26 evals (max 0.0501); seed means reach 0.12 and single evals reach 0.55.

**Missed true effect (supports S6).** In the none model, where neither prior is trained, the true-prior gradients align for exactly the transfer pairs: dim|box5 +0.118 and cities|countries +0.125 in-window, against +0.02–0.03 for dim|torus, torus|box5 and dim|cities. Before the window the corresponding values are +0.04, +0.04 and ≤ 0.03. Recommend dropping the gcos clause from S7 and adding this to S6.

## Possible missed or under-stated true effects
- **S1, a threshold-free version.** all6@1e-3's 8-seed mean curve reaches 100% of N_m's final3 at steps 5500 / 1500 / 2500 / 1750 / 5000 / 1750. That is 30 / 8 / 14 / 10 / 27 / 10% of training, and every seed reaches it (8/8 per manifold).
- **S4.** At the same lr, all6@2e-3 has no mean bpb cost (−0.03%).
- **S6.** The gradient-alignment evidence in the none model (above), and the ×3 larger transfer at L4.

## Strongest supported statements (pt6v2, 8 seeds, the 6-manifold composition, lr as given)
- **Speed.** all6@1e-3 reaches 90% of the best no-prior lr's final level above the heuristic floor 2.9–5.0× sooner (2.1–5.0× above true chance; per-seed median 2.0–4.7×). It matches that lr's final level by 8–30% of training in every seed.
- **Window.** For dim and torus, the gain over the single prior arises after the window (final3 +25.0 and +11.7, both 8/8). The single-prior models lose about 2.5× more after the window (8/8), while their own-target CKA at the trained layers is retained equally (|Δ| ≤ 0.01).
- **Transfer.** Training only one prior raises the CKA of its partner domain, across disjoint tokens:
  - dim ↔ box5: +0.16 / +0.18 (L4 +0.24 / +0.25);
  - cities ↔ countries: +0.22 / +0.24 (L4 +0.32 / +0.29).

  All are 8/8. In the no-prior model these pairs' prior gradients are already aligned (+0.12, vs ≤ 0.03 for other pairs).
