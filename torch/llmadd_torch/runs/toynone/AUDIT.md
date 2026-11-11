# toynone: is "not learned without the prior" (F84) an artefact of the toy optimiser / budget? (2026-10-17)

User: "可以的：想把「学不会」说得更硬：给 none 扫 lr、加长训练".
The cells are F84's two rule-(c) readings:
- S4 frac 0.6 (question split): F84 none 0/5 in 25k steps; R@always 5/5, median t_grok 10.2k.
- dim frac 0.3 (class split): F84 none 0/5 in 50k steps; expo 4/5.

Design:
- Arm: none only.
- Grid: lr {3e-4, 1e-3, 3e-3, 1e-2} x weight decay {0.1, 1, 3}; seeds 0-4.
- Budget: 4x F84, i.e. S4 100k steps, dim 200k.
- Everything else is the F84 setting: one-layer d128, full batch, betas (0.9, 0.98), 10-step warmup, same splits per seed.
- The prior arms are not re-tuned. That is conservative for the claim: their optimiser stays at the default.

Metrics (definitions as in toy_struct.py / F84):
- t_grok = first eval with held-out test >= 0.95, censored at the budget;
- AUC = mean held-out accuracy over every eval of the budget;
- final test; for dim also test_nonegtwin.

PRE-REGISTERED reading (written before any run):
- "Not learned without the prior, robust to the optimiser and a 4x budget" holds for a task iff, in EVERY one of the 12
  (lr, wd) configs, at most 1/5 seeds grok within the 4x budget.
- Otherwise report, per config: grok count, median t_grok, the AUC curve.
- Then compare the best none config with the prior arm on two measures:
  - grok step at the prior's default optimiser: S4 R@always median 10.2k; dim expo median 4.8k;
  - steps-to-0.95 ratio (censored median).
- If the best none needs more than 4x the prior arm's steps, the claim becomes "the prior speeds learning by > 4x",
  not "not learned".
- A config whose runs diverge (non-finite loss) counts as not learning, and is reported.

Checks:
- The default path of the new --lr / --wd flags is bitwise identical to the F84 snapshot: none and R curves on CPU,
  300 steps.
- Every run JSON records lr / wd / steps / split.
- The 24 jobs render with the intended flags (generator).

## 2026-10-18 amendment: the lr axis of this batch is INVALID
- **What went wrong.** The toy's warmup line set the optimiser lr to a hard-coded 1e-3 at every step, so --lr had no effect.
  All four "lr" configs ran at lr 1e-3. The tell: results were identical across lr within each wd.
- **Still valid.** The weight-decay axis (0.1 / 1 / 3) and the 4x budget at lr 1e-3. There, none groks 0/5 in every config
  for S4 0.6 (100k steps) and dim 0.3 (200k steps).
- **Fix.** tools/toy_struct.py now uses args.lr in the warmup, logs lr_now at every eval and asserts it after the warmup.
  Checked: lr 1e-3 is bitwise identical to the F84 snapshot; lr 1e-2 now differs.
- **Re-run.** The lr != 1e-3 configs go to runs/toynone2 (same grid, same seeds, same pre-registered reading).
- **Why it was missed.** The regression only tested that the default path was unchanged, never that the new flag takes
  effect. The fresh-agent audit was skipped for this "small" change.
- Aggregation (re-audit 2026-10-18): the invalid lr != 1e-3 cells were moved to out_INVALID_lr_inert/ (README inside).
  Combined tables key on code_sha: b74a78cb5472 for runs/toynone2, and 6248613ac6d2 only at lr 1e-3.
  Next to t_grok, report AUC and final test (a high-lr run can grok and then fall back below 0.95), and scan every curve
  for non-finite ce.
- A 1-seed CPU probe by the auditor (lr 1e-2, wd 0.1, S4 0.6) grokked at 16k steps, against R@always's median of 10.2k.
  If toynone2 confirms this, the F84 reading becomes "the prior speeds learning by ~1.6x at the best none optimiser",
  not "not learned".

## 2026-10-19 result of toynone2 (partial: 16/18 cells) and the follow-up
- S4 0.6: none groks at lr 3e-3 (2-3/5 per wd) and at lr 1e-2 (wd 1: 5/5, median t_grok 4.6k steps; wd 3: 5/5, 5.1k).
  The pre-registered reading "not learned without the prior, robust to the optimiser" DOES NOT HOLD for S4. At lr 1e-2,
  none is faster than R@always at the default lr (median 10.2k).
- dim 0.3: 1/5 at lr 1e-2 wd 0.1 (the other lr 1e-2 cells are still running). Holds so far.
- Follow-up (every arm at its own best lr): runs/toyprior_lr. R@always, R_sign@always and specrand@always at lr 3e-3
  and 1e-2, wd 1, 100k steps, seeds 0-4, compared best-vs-best with the tuned none (same t_grok / AUC definitions).
- F84's S4 "rescue" reading is suspended until then.
- 2026-10-19: dim none's best config sat at the top lr edge (1e-2: AUC 0.18-0.22, 1/5 grok at wd 0.1), so lr 3e-2 x wd {0.1, 1, 3} is added (runs/toynone3). Bracket rule (lesson c).
