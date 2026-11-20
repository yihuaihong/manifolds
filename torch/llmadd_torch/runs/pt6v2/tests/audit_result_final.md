# pt6v2 final read-out: result audit (2026-11-19)

The auditor (exp-auditor) did not take part in the design. Code names used here:
- pt6v2: the batch that trains all six manifold priors jointly.
- all6: all six priors in one run. none: no prior. only-m: one prior on its own.
- N_m: for each manifold, the no-prior learning rate (lr) whose 8-seed mean area is highest.
- g = all6@1e-3 − none@N_m. h = all6@2e-3 − only-m@2e-3. u = only-m@2e-3 − none@2e-3.
- final3: the mean of the last 3 evals. V3: the post-hoc amendment that adds none@6.4e-2.

Materials checked: tests/readout_final_20261119T2001.{txt,json}, tests/render_final_20261119T2001.md, and the 104 run JSONs under out/. I recomputed everything independently, without importing the read-out; the dim strict set was rebuilt from its definition. Scripts and outputs: /scratch/yh6210/tmp_audit/pt6v2_final/{indep,lmcheck,extra}.py and *_out.txt.

## 1. Reproduction: no number is wrong or unreproducible
- **Validity.** All 104 runs pass: identity (seed, lr, prior_domains), 75 evals each, check.json all passing, no non-finite values. Each seed has one data_sha and runs on one device (s16–22 on B200, s23 on A100).
- **Areas.** My per-run areas differ from each run's own metrics by 0.
  - dim strict was rebuilt from scratch, giving 58–95 questions per seed. The per-seed g values match the read-out exactly.
- **JSON comparison.** All fields agree to within 1e-6:
  - for each of the 6 manifolds: N_m, g (mean and per seed), wins, p, Holm, boot90, h, u, p_int, final3, window in/out, the 5 none means, g vs each lr, and the same-lr g;
  - val_loss, bpb and share for all 13 arms;
  - mean6;
  - the render's 95% CIs.
- **The one difference** is the FLOOR line for cities/countries. I did not rebuild h3_held, so I only have a lower bound (72.2/76.8 vs the read-out's 74.5/78.6). Both arms sit far above either line, so the tags are unaffected.
- **Every number in the sentence is correct:**
  - g +46.9 / +33.9 / +6.3 / +5.8 / +22.0 / +19.7, each 8/8;
  - the largest Holm p is .00220 (cities). "≤ .0022" holds only after rounding; strictly it should read ≤ .0023;
  - h runs from −0.51 to +13.50;
  - share 0.7065;
  - the bpb gap is +0.0035;
  - 52.67M parameters.

## 2. Wording to change (correcting facts or claiming more, not adding qualifiers)
- **[Important] "at that manifold's best learning rate".**
  - The problem: for dim and box5, N_m = 3.2e-2 is the top of the grid, and none is still rising there (dim 37.7→47.3, box5 31.0→41.0, about +10 per doubling). The optimum is not bracketed.
  - V3 has only one run started, none@6.4e-2-s16 (20:13), with no output yet (material missing).
  - Suggested wording, which is stronger and already true: "at each of the five no-prior learning rates tried (2e-3–3.2e-2)".
    - All six manifolds win against all 5 lrs in 8/8 seeds.
    - They also win 8/8 against each seed's own best none lr, which is stricter than N_m: g +45.6 / +33.9 / +4.5 / +4.5 / +19.9 / +18.9, with smallest per-seed margins +37.8 / +18.8 / +2.5 / +3.8 / +16.3 / +4.8.
  - If none gains about another +10 at 6.4e-2, dim g drops to about 24 and box5 g to about 10. box5 s20 (+5.2) could flip. The direction most likely holds. Re-read these two cells once V3 produces numbers.
- **[Important] LM "within 0.004 of the best no-prior model".**
  - The +0.0035 mean reproduces, but one point pulls it down: none@2e-3 s22 has bpb 1.2387, while the other seven are 1.199–1.210.
    - only-numbers and only-cities at 2e-3 are also high for s22. There is no val spike above 0.05.
  - Per seed, all6@1e-3 is worse in 7/8 seeds. The median gap is +0.0046; without s22 it is +0.0097.
  - At the same lr (all6@2e-3 vs none@2e-3) the mean is −0.0004 and the median +0.0038, also driven by s22.
  - Suggested wording, robust and no weaker: "final bpb 1.213, between the two best no-prior lrs (1.210 at 2e-3, 1.215 at 4e-3)". This holds for medians too (1.2127 vs 1.2063 / 1.2155).
- **[Important] "interference not detected" undersells a real effect.**
  - dim trained with the other five priors beats dim's prior alone by +13.5: 8/8 seeds, one-sided p .0006, Holm over 6 tests .0037.
  - torus, countries and cities have h > 0 in 6/8 seeds, p .014–.043, but none survives Holm.
  - numbers has h −0.51: 7/8 seeds negative, one-sided p .040 uncorrected, Holm .24. Under the registered margin rule (−24.9) this counts as not detected.
  - Suggested wording: "joint training costs at most 0.5 point on any manifold and lifts dim by +13.5 over its prior alone (8/8)".
- **[Minor] Share definitions.**
  - 0.71 is the median of ‖g_LM‖/(‖g_LM‖+‖g_prior‖). Per run it ranges 0.689–0.730. The median on-window clip_coef is 1.0 (no clipping), which fills the render's MISSING.
  - The logged grad_lm_share is a/|g_total|, which gives 0.92. The paper should state which definition it uses and match F100/F101.

## 3. Qualifiers not to add, and scope
- **"N_m chosen on the same seeds"** biases against the claim, not for it: taking the max of the none means can only raise the none baseline, so g is if anything underestimated. The per-seed oracle check is already 8/8. Do not write this as a qualifier.
- **"prior at its lowest lr"** is the same: all6 was not lr-tuned.
- **The result does not depend on all6 using 1e-3.** At 2e-3, all6 still beats none@N_m by +45.0 / +30.3 / +6.4 / +5.7 / +19.5 / +15.9.
- **Scope, not a qualifier.**
  - torus and box5 gain mainly inside the prior window: in-window +18.5 / +14.5 vs out-of-window +3.4 / +5.2.
  - At the end of training (final3), torus is −3.0 (1/8; 0/8 vs the oracle) and box5 is −1.6 (5/8).
  - The other four still win at the end: +33.9 / +30.7 / +3.7 / +3.7, winning in 8 / 8 / 7 / 7 of 8 seeds.
  - The original sentence only claims area, so it is correct. Just don't write elsewhere that all six are "better at the end".

- **最可能测错的地方** (where it is most likely mismeasured): the dim and box5 g values use the grid-top none@3.2e-2 as the baseline, and the none optimum is not bracketed. If V3's none@6.4e-2 is higher, both g values are overstated, though their direction most likely holds.
- **可能被测没或写没的真效应** (real effects that may have been measured or written out): "interference not detected" hides dim's positive transfer (+13.5, 8/8, Holm .0037). "at the best lr" is weaker than what holds (8/8 against all 5 lrs and against each seed's best lr). Writing "N_m chosen on the same seeds" as a qualifier would point the wrong way.
- **证据撑得住的最强结论** (the strongest conclusion the evidence supports): "Trained together (all six priors at lr 1e-3, 53M LM, 8 seeds), each of the six manifold priors gives a higher read-out curve area than the no-prior model at every one of the five learning rates tried (2e-3–3.2e-2), in 8/8 seeds and also against each seed's best (g vs the best mean lr +46.9, +33.9, +6.3, +5.8, +22.0, +19.7; Holm p ≤ .0023). Joint training costs at most 0.5 point on any manifold vs its prior alone and lifts dim by +13.5 (8/8), at an LM gradient share of 0.71 and a final bpb between the two best no-prior lrs (1.213 vs 1.210/1.215)." This is stronger than the planned sentence. The one thing pending is that the dim/box5 none optimum lies above the grid; V3 settles it.
