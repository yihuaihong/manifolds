# ptgw — window vs delayed gate across settings (2026-10-14, user-approved)

Question (user): is "a self-timed gate that starts at 5%" or "a window that starts at 5%" the better schedule for the
manifold prior? Both start at 5%, so they differ only in WHEN THE PRIOR STOPS. Existing evidence (workflow
wf_76559c97, scratchpad/wf_gate_window; F79): in the main setting a delayed gate with the existing rule would close at
12-14% of training with held-out 3+3 at ~52% (read off the window runs: identical trajectories until it closes); the
step-0 gate that closed at that state lost b33 speed (gate - whole b33_last5 -3.5, cross-hardware, n=4) but the same
gate on all layers did not (pt999 sens-Lall, n=3); a fixed end at 40% may cut learning short when learning is slower.
So the question is which schedule is never bad across settings.

Code: tools/pretrain_range.py at code/COMMIT. New: --gate_start (no prior before this fraction; same boundary as a
window's w0 * steps <= s), --gate_hyst (a closed gate reopens only when the lowest bucket EMA < gate_thr - gate_hyst;
gate_step() is a pure function), prior_events ([step, on/off] at every switch) saved in checkpoints and results.
Generator: llmadd_torch/make_jobs_ptgw.py. Summary: tools/ptgw_summary.py. Offline LM: tools/offline_lm.py.

## Arms (ctx site = operands in the lines, helix3, 7% arithmetic, L40S, seeds 0-3, every model saved)
| setting | change from main | W | D05 | D09h | none |
|---|---|---|---|---|---|
| main | lr 1e-3, layers 2,4, 3e8 tokens | x4 | x4 | x4 | pt999 7pct-none (8, saved) |
| lr5e-4 | --lr 5e-4 | x4 | x4 | x4 | x4 (new, saved) |
| Lall | --geo_layers 1..6 | x4 | x4 | x4 | pt999 7pct-none |
| tok6e8 | --tokens 6e8 (36,620 steps) | x4 | x4 | x4 | x4 (new, saved) |
W = --schedule window --window 0.05,0.4; D05 = --schedule gate --gate_start 0.05 --gate_thr 0.5; D09h = the same with
--gate_thr 0.9 --gate_hyst 0.1. 56 runs, ~70 L40S GPU-hours (tok6e8 runs take ~2 h).
Free references (other snapshots; ctx path identical by regression test; descriptive only): pt999 7pct-gate-ctx (gate
from step 0), 7pct-whole-ctx, sens-lr5e-4-{none,gate-ctx}, sens-Lall-gate-ctx; pt999b 7pct-win-ctx.

## Metrics (unique definitions in tools/ptgw_summary.py)
- b33_area (PRIMARY speed): mean over all evals of held_b33 (held-out 3+3-digit sums, both orders never trained,
  exact match of answer + newline, teacher forced, at the training position) x 100. A speed metric: at 3e8 tokens the
  gate runs were still catching up at the end, so last-eval values mostly measure speed, not final ability.
- bpb_offline (PRIMARY LM cost): tools/offline_lm.py, bits per byte of the saved final model on ALL of val.bin (2M
  tokens, non-overlapping 512-token windows, the same for every model), eval mode, bf16 autocast. Double path: the
  same script recomputes the in-training metric on its original 8 x 16 windows and must match the logged final value
  (|diff| < 0.002 on GPU), else the offline numbers are not used.
- secondary: area6 (6-bucket curve mean), b33_t90 (first eval after which held_b33 >= 0.9 for good; steps, tokens and
  LR-weighted progress), b33_last5, prior_on_frac, toggles, first close step and held_b33 at the first close.

## Pre-registered decision rule (written before any run)
Per setting, paired by seed, D - W for D05 and D09h. Report mean, paired t p and exact sign-flip p (smallest
possible = 0.125 at n = 4), wins. Margins: b33_area 2 points, area6 1 point, bpb_offline 0.004.
- D is "worse" in a setting if b33_area (or area6) is lower by more than the margin in >= 3/4 seeds with paired p < .05.
- D is "non-inferior" in a setting if the 90% CI of the b33_area difference lies above -2 and bpb_offline lies below
  +0.004; otherwise "undecided" for that setting.
- Overall: per schedule, worst-case regret on b33_area across the 4 settings (best schedule mean - its mean). The
  schedule with the smallest worst-case regret wins if no setting shows it "worse" and its bpb is non-inferior. If
  the worst-case regrets are within 2 points: tie -> default W (fewer parts, no task signal needed), and the result
  goes to the user. A D09h run with > 20 toggles counts as a failure of that run's gate (reported, kept in the stats).
- Validity: all runs reach the last step; one code_sha across ptgw; D runs have prior on at the first eval after 5%;
  until a D run first closes, its curve may differ from W only by GPU nondeterminism.

## Predictions (written before results)
- main: D05 b33_area below W by ~1-4 points (centre ~3); D09h ~ W (it would close near 29-53%).
- Lall: D05 ~ W (the step-0 gate did not lose b33 there).
- lr5e-4: learning is slower; W's fixed end at 40% may stop the prior while b33 is still low (the step-0 gate first
  closed at 31-37% with b33 ~0.5) -> D09h may beat W here.
- tok6e8: windows and the gate start are fractions, so they scale with the budget; expected D05 < W as in main.
- LM: |D - W| <= 0.003 bits/byte everywhere (both start at 5%), i.e. not separable at n = 4.

## Train vs eval
As runs/pt999/AUDIT.md (same evaluate(), held-out split, position, template subset). Only the schedule, lr, layers or
budget change between arms.

## Known limits
4 seeds per cell (a 2-point b33_area difference may stay undecided); one target, site and arithmetic share; 53M only;
val.bin has only 2M tokens; all onsets are fractions of training (a state-triggered onset is not tested here).

## Rule amendment 2026-10-15, written BEFORE any ptgw result existed (runs/ptgw/out did not exist; no job had started)
Adversarial audit (workflow wf_5d35198d: 3 reviewers + 10 verifiers; scratchpad/audit_ptgw). Gate code: no error found
(synthetic-EMA checks of closing, the hysteresis band, re-opening and resume in the closed state are bit-identical; the
window path is unchanged). All serious findings concern the analysis; this amendment replaces the rule above where
they differ, and tools/ptgw_summary.py implements it (it prints one DECISION line).
1. Tests: "paired p" = two-sided paired t (scipy ttest_rel); exact sign-flip p is reported only. "3/4 seeds" means
   at least 3 of the 4 per-seed differences beyond the margin, with the mean on the same side.
2. Symmetric "worse": in a setting, schedule S is WORSE THAN T if, paired by seed, S - T is below -margin in >= 3/4
   seeds, mean < 0 and p < .05, on b33_area (margin 2) or area6 (margin 1); or if S's bpb_offline exceeds T's by more
   than 0.004 in >= 3/4 seeds, mean > 0.004 and p < .05. This applies to W, D05 and D09h alike, against each other.
   A schedule is ELIGIBLE if it is worse than no other schedule in any setting.
3. Regret: per setting, on the seeds common to the three schedules (validity requires all four), best mean b33_area
   minus the schedule's mean; worst-case regret = max over the 4 settings. Units: fraction of training (as
   registered); a token-equivalent version (tok6e8 regret x 2) is reported as a sensitivity check.
4. Decision table:
   a. no eligible schedule -> NO WINNER: the per-setting trade-offs go to the user (no default to W);
   b. the eligible schedule with the smallest worst-case regret wins if every other eligible schedule's worst-case
      regret is larger by > 2 points;
   c. otherwise (eligible schedules within 2 points): W if W is among them (fewer parts, no task signal), else the
      choice between D05 and D09h goes to the user;
   d. if the winner differs between the fraction and the token-equivalent units -> UNIT-DEPENDENT, to the user.
   Every outcome is reported to the user with the per-setting table; none of them changes the paper's main method
   without the user's decision.
5. Non-inferiority of D vs W per setting (90% t-CI of D - W b33_area above -2, bpb_offline 90% t-CI below +0.004) is
   reported as DESCRIPTIVE only: with 4 seeds it discriminates only when the paired SD of b33_area is <= 1.5; the
   actual paired SD and CI half-width are printed.
6. Validity (the summary exits with an error unless --allow_invalid; then everything is labelled descriptive): all 56
   runs finished with seeds 0-3; one code_sha across ptgw, equal to sha256 of the snapshot's pretrain_range.py
   (a3bce3ab63c1); every run's args match its cell (schedule, window, gate_start / thr / hyst, lr, geo_layers, tokens,
   target, site, arith_lines); every D run's first prior event is [ceil(0.05 * steps), 1]; an offline entry is used
   only if |orig_minus_logged| < 0.002, it covers all of val.bin, it was computed from this run file and its logged
   final bpb equals the run's. Missing / invalid offline entries make the LM part of the decision PROVISIONAL.
7. Descriptive extras: per-seed differences; paired SD; per-run toggles (onset switch excluded when compared with
   step-0 gates; D09h runs with > 20 toggles flagged; D05 toggles reported); the largest |D - W| in held_b33 before
   D's first close (GPU-nondeterminism floor); tok6e8 D - W b33_area over its first 18,310 steps (= the first 3e8
   tokens); runs whose gate never closed (D09h may stay open in slow settings: the in-batch EMA peaks at 0.88-0.97).
8. Prediction notes (corrections, not changes): the closest same-metric analog of main D05 - W is the step-0 gate vs
   window in ptrangeg, b33_area -9.45 (90% CI -14.3 to -4.6); the lr5e-4 "first close at 31-37%" came from the TEXT
   site gate (pt999 sens-lr5e-4-gate-text), the ctx version had not run yet.
Analysis tools: run from the repo (tools/ptgw_summary.py, tools/offline_lm.py) at commit 3440149, not from the training snapshot (training code unchanged, 2f99b06).
