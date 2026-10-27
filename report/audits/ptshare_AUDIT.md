# ptshare — the arithmetic-share dose at 53M (Torch L40S; 2026-10-26)

User approval (relayed by the coordinator, 2026-10-26): run the dose, queued "after this batch" (the fixed-tokens-per-parameter
batch, nice 20) -> `--nice=2000`; rb6_190m_dim (nice 5000) stays last. The same day, before any job of this batch, the user
relaxed the decision rule ("我感觉应该不用这么严格？"): rule 5 below is that rule. This file was written BEFORE any job of
this batch was submitted; no ptshare result exists. Copy: llmadd_torch/runs/ptshare/AUDIT.md (this text + later sections).

## Question
At one size (53M), with one trainer, one GPU type and one lr-selection rule, does the gain of the helix3 window prior (win-ctx)
depend on the share of arithmetic in pretraining? F86 (2%, Burst A100, pt2main) and F87 (7%, Torch L40S, ptlr) suggest a
larger best-vs-best area gain at 2% (+49 to +53) than at 7% (+23 to +36), mostly because none barely learns at 2%; but the two
come from different batches (cluster, GPU, lr grids), and F86's 505M addendum forbids reading them side by side as "the less
arithmetic, the larger the gain". Here only the share changes.

## Share mechanism (the same for every share)
- The trainer's existing `--arith_lines k` (tools/pretrain_range.py `batch()`): every 512-token row is a FineWeb-Edu chunk
  followed by k arithmetic lines. F82 / F86's "2%" is k = 1 and F82 / F87's "7%" is k = 4: the same mechanism (lines per row,
  not a ratio parameter). No trainer change, no new flag.
- Dose: k = 1, 2, 4, 8, labelled 2 / 4 / 7 / 14% (the F82 / F87 names kept for k = 1 / 4). Measured token shares
  (tools/arith_share.py: the run's own first 640 rows replayed from the trainer's random streams, CPU, seeds 0-7; per seed in
  tools/ptshare_refs.json): see "Tests". The labels are nominal: k = 4 is 8.5% of the tokens and k = 8 is 17%.
- The line-length distribution does not depend on k (formats, operand buckets and the 5% rehearsal pool are the same), so the
  expected share is proportional to k: ln(share) = ln(k) + const, and every slope below is per unit ln(share) (also printed
  per doubling).
- Every job checks its own share: `--dry_inputs` logged to a file, 2k operand sites in each printed row, the target line only
  for win-ctx, then tools/arith_share.py replays rows 0-2 from the same streams and must reproduce their operand positions
  exactly (exit 3 before training otherwise) and prints the run's share.

## Cells
- Trainer: tools/ at commit 64abc19 (code_sha a3bce3ab63c1) = runs/ptlr/code; runs/ptshare/code is asserted file-by-file
  identical to runs/ptlr/code. 53M (d384 x 6, 52,672,128 parameters), 3e8 tokens = 18,310 steps of 32 x 512, AdamW (0.9,
  0.95) wd 0.1, 1% warmup, cosine to 0.1x, lambda 5, prior layers 2, 4, eval every 250 steps (75 evals), `--save_model
  --ckpt_every 2000`; L40S (l40s_public, torch_pr_625_general, gpu48, 3 h limit, `--requeue`): all as ptlr 53M.
- Arms: none (`--target none`); win-ctx (`--target helix3 --schedule window --window 0.05,0.4 --site ctx`).
- Grids, seeds 0-3, for k = 1, 2, 8: none 4e-3 / 8e-3 / 1.6e-2; win-ctx 1e-3 / 2e-3 / 4e-3 (72 jobs). Why: centred on the
  7% best lrs (F87 53M: none 8e-3, win-ctx 2e-3, both brackets closed, confirm seeds agreeing); F86 pt2b (2%, 53M) selected
  the same two lrs on Burst; factor 2 = ptlr's step.
- k = 4 is reused, not re-run: ptlr none@8e-3 and win-ctx@2e-3 (seeds 0-7), and for the same rule on the same 3-lr sub-grid
  ptlr none@4e-3 / 1.6e-2 and win-ctx@4e-3 (seeds 0-3), win-ctx@1e-3 (ptgw main-W seeds 0-3, same snapshot; ptlr seeds 4-7).
  The rule must reproduce F87's 8e-3 / 2e-3 there (the read-out prints a WARNING otherwise; it does: tested on the real cells).
- Tags `53m-lr<lr>-<2pct|4pct|14pct>-<arm>` (used by no other batch), out `runs/ptshare/out/<tag>/`.
- Generator llmadd_torch/make_jobs_ptshare.py: the job text is make_jobs_ptlr.py's own template (sha asserted; taken from its
  source with ast, not copied) plus exactly these edits: job name, `--nice=2000`, two comments, the HOLD path, an L40S guard,
  and the dry-run block above. Design constants: tools/ptshare_design.py of the read-out snapshot runs/ptshare/readout (git
  archive of branch arith-dose; COMMIT lists every read-out file's sha256[:12], re-checked by every job and selection job).

## Selection, expansion, confirm seeds (automatic; pre-registered)
- Rule 1 = runs/ptlr/AUDIT.md rule 1 as pre-registered: best lr per (share, arm) = argmax of the mean area over seeds 0-3;
  the top two within 2 points -> the lower mean in-training bpb (these jobs run no offline LM; marked '*'), exact tie -> the
  lower lr. Bracket = readout_rules.bracket: OPEN iff the best lr is at a grid edge and beats its neighbour by > 2 points; an
  area argmax at an edge by <= 2 is a note. INTERPRETATION of "选在边缘时自动往外扩一档": "at the edge" = this open-bracket
  definition (F86 / F87 practice), so a flat edge (the prior is lr-flat) does not trigger an expansion.
- Round 1 (CPU job `llmadd_ptshare-select-r1`, cpu_short, `--dependency=afterany` on the 72): per (share, arm)
  - closed -> confirm seeds 4-7 at the best lr (4 jobs);
  - open -> one factor-2 step outward on seeds 0-3 (none 2e-3 / 3.2e-2, win-ctx 5e-4 / 8e-3; 4 jobs), plus the round-2
    selection job (afterany on the expansion jobs);
  - incomplete (a run missing or unfinished) / withheld (integrity) -> nothing for that arm; the reason goes to select.log.
- Round 2: closed after the expansion -> confirm seeds 4-7 at the (possibly new) best; still open -> STOP: nothing is
  submitted for that arm and the user decides (the next lr and its confirm seeds). Round 2 never expands.
- Each selection job: tools/ptshare_summary.py --json -> tools/ptshare_pick.py -> make_jobs_ptshare.py --stage auto
  --test-only, and --submit only if every test passed. Confirm seeds are selection-free (the lrs were picked on seeds 0-3).

## Metrics (one definition each; tools/readout_rules.py unchanged)
- held6 = mean over the 6 buckets of held-out exact match (answer tokens + newline, teacher forced, argmax), format 0 at the
  end of a 512-token val-text row (pretrain_range.py evaluate(); pretrain_range has no --ctx_eval flag, that was
  pretrain_helix's). area = 100 x mean over the 75 evals of held6 (readout_rules.area). t_theta = tokens (M) until held6
  first reaches theta, linear interpolation, censored at the 300M budget (readout_rules.t_theta).
- Delta area = area(win-ctx@best) - area(none@best), paired by seed. R_theta = median t_theta(none@best) / median
  t_theta(win-ctx@best) with censored runs at the budget (readout_rules.speedup: '>=' when none's median is censored, '<='
  when the prior's is, no ratio when both are).
- 90% CIs: paired seed bootstrap, 10000 resamples, numpy default_rng(0): of the mean Delta area, of R (readout_rules.boot_ci).
- Slopes: OLS over the shares of y on x = ln(k), y in {Delta area, log R_0.5, log R_0.9, area none, area win-ctx}; 90% CI from
  10000 bootstrap resamples (default_rng(0)), seeds resampled within each share (the two arms paired by seed, the shares
  independent; across shares a seed shares the held-out split, the initialisation and the SEQUENCE of arithmetic pairs
  (data_rng: the first three pairs of k = 1 rows 0-2 are the first three of the k = 8 row 0), but the rows differ, so the
  shares are correlated positively and independent resampling makes the CI wider, i.e. conservative), the same resampled
  indices for every metric; censored runs at the budget. A bounded R (a censored median) makes the slope a bound in the
  direction of that share's OLS weight.
- Held-out set and position are identical across the shares: the split is drawn from default_rng(seed), the eval rows from
  default_rng(11) / (5), none of which depends on --arith_lines (code); checked on the CPU dry runs: for every seed the eval
  lines (line / held_b11 / held_b33 / ood_b44 / eval-set shapes) are identical across k = 1, 2, 4, 8.

## Train vs eval conditions (only the share differs between the shares; the arms differ only in the prior)
| dimension | train | eval | same? |
|---|---|---|---|
| position | k lines at the end of a 512-token row after FineWeb text (k = 1: one line, at the end) | one line at the end after val text | k = 1: yes; k > 1: the last line's place, the first line's context (text before it) |
| template | 4 formats, random | format 0 ("Q: a + b = ") | subset |
| values | operands 0..999, length buckets balanced, 5% rehearsal pool | the 6 buckets' held-out pairs (both orders never trained) | yes |
| split | train.bin text; held pairs rejected by draw_pair | val.bin text; held pairs | disjoint by construction; identical across the shares |
| model state | train, bf16 autocast, no gradient checkpointing (53M) | eval mode, bf16 autocast | yes |
| what changes with k | k x 32 lines per step (2k operand sites per row for the prior), 512 - L text tokens | nothing | the dose |
| prior strength at fixed lambda 5 | the CKA is over 2k x 32 operand sites per batch (64 at k = 1, 512 at k = 8); |g_prior| / |g_lm| at the first on-eval was 3.5-7.8 at 2% (pt999 2pct-win-ctx, snapshot 79e9b410dcc3, lr 1e-3) vs 1.3-2.5 at 7% (ptgw main-W, lr 1e-3) (audit item 3) | - | not matched: lambda is fixed by design; the read-out prints the gradient ratios per share |

## Read-out (tools/ptshare_summary.py; written and tested on fake results before any ptshare run)
- Rule 2: a diverged run (first non-finite lm / val_loss / geo / grad_prior) keeps its area; 'unstable' (final val_loss >
  the same seed's none@4e-3 of the same share + 0.5 nats) is listed; a run that stopped on an error of its own, listed in
  runs/ptshare/OWN_FAILURES.txt, counts with the missing evals scored 0 (infrastructure losses are resubmitted).
- Rule 3, per share at the selected lrs, side by side for seeds 0-3 / 4-7 / all 8: both arms' absolute areas; Delta area
  [90% CI], wins, paired t p; R_0.5 [90% CI] and R_0.9 [90% CI] with the censored counts of each arm; median final held6;
  in-training bpb; floor / ceiling flags; for the prior arm the LM share of the pre-clip gradient norm |g_lm| / (|g_lm| +
  |g_prior|), the share of prior-on evals with the prior's norm above the LM's, and the clip coefficient min(1, 1 / |g_lm +
  g_prior|) (max_norm 1.0) derived from the logged grad_lm / grad_prior / grad_cos (the trainer logs them at the prior-on
  evals only; no clip value is logged directly).
- Rule 4: the slope table, every metric for seeds 0-3 / 4-7 / 8. The primary metric only decides which CI the verdict reads;
  every quantity is printed.
- Rule 5, the verdict (user 2026-10-26):
  1. Primary metric: log R_0.5 for the shares >= 7%; for the 2% and 4% shares Delta area if none's t50 is censored in >= 3 of
     its 8 seeds (the user's words: per share). One slope needs one metric, so the rule has two readings; both are computed
     and printed, and the verdict stays PENDING until the user picks one (tools/ptshare_design.py RULE5_READING), before any
     ptshare result (pre-submission audit 2026-10-26, item 1):
     - reading A: when the switch fires at 2% or 4%, P = Delta area for the whole slope (so 7% and 14% are read by Delta area,
       where the prior sits near its ceiling: their area gain is compressed, which pushes the slope towards "the less
       arithmetic, the larger the gain"); otherwise P = log R_0.5. A bounded P is not determinable.
     - reading B: P = log R_0.5 for every share (as the user names for >= 7%); a censored none median at 2% / 4% makes R a lower
       bound there and, with the negative OLS weight of the low shares, the slope an upper bound: with such a share reading B
       can only show "the less arithmetic, the larger the gain" (a positive slope is then "未显示"). S = Delta area.
     - censoring in the bootstrap (both readings): a claim must also hold on the conservative CI, in which every resample
       whose censored medians bound its slope towards the other side is set to -inf / +inf, so censoring can only make the
       test stricter (audit re-check: a 5% tolerance would have let the one-sided level reach about 10%).
     - while the reading is unset, no reading's verdict is printed or written (only the definitions); the choice is made
       before any ptshare result, and must be made before `--submit` (it needs a new read-out commit, RO_COMMIT, and a
       rebuilt runs/ptshare/readout, which running jobs read).
     S = the other of the two in each reading. R_0.9 never decides.
  2. Main test: P's slope over all 8 seeds per share; its 90% CI excludes 0 -> shown.
  3. Guardrail against the selection bias: P's slope on the confirm seeds 4-7 only, same sign (no significance needed).
  4. S's 8-seed slope, same sign (no significance needed).
  All three -> "显示了（算术越少，增益越大）" for a negative slope ("算术越多，增益越大" for a positive one). P's CI includes 0 ->
  "未显示" with the point estimate and CI. Guardrail or S with the opposite sign -> "方向不稳". INTERPRETATIONS: a sign
  that a censoring bound leaves open (the bound can move that slope across 0) is not "the same sign" -> "方向不稳"; P's 8-seed
  slope containing a bounded point is not determinable -> "未显示".
- Floor / ceiling (pre-registered; 8 seeds at the selected lrs): an arm is at the floor if its mean area < 5 and its median
  final held6 < 0.10, at the ceiling if its mean area >= 90 and its median final held6 >= 0.95 (the window opens at step 916,
  so the area cannot exceed about 95). A share with both arms at the floor or both at the ceiling is "无信息" and leaves every
  slope; fewer than 3 informative shares -> no slope -> "未显示". A share where the prior alone is at the ceiling (expected
  at 14%): its area gain is compressed by the ceiling and is read by R (noted on the output); when P = Delta area, the verdict
  then also rests on S = log R_0.5 having the same sign (rule 5.4).
- Which arm moves the gain is read from the two absolute areas and their slopes (descriptive).
- No confirmatory comparison with F86 / F87 numbers (other batches); the 7% row is the ptlr cells themselves.

## Validity (before any number is used; a failure withholds every verdict using the cell)
code_sha a3bce3ab63c1; 18310 steps / 75 evals; 52,672,128 parameters, no gradient checkpointing; args = the ptlr 53M base +
the share (save_model, ckpt_every 2000, data, tokenizer, defaults); win-ctx prior_on_frac 6408 / 18310 and prior_events
[[916, 1], [7324, 0]]; none never on; step-0 lm within 1e-3 of the CPU reference of its (share, seed) (tools/ptshare_refs.json)
and of the (share, seed) median over the counted cells; win-ctx: the prior acts (26 on-evals, grad_prior > 0 at every one,
the median geo inside the window below a quarter of the geo at the last pre-window eval, step 750; the real 53M window-ctx
runs are >= 40x apart), a check skipped for a diverged run (rule 2 counts it with its area); each cell only under its pre-registered tag
(the reused 7% cells pinned per source: a same-args run under another tag is left out, not merged). Ceiling of the seen
pairs is not checked (F87 printed it only).

## Expected ranges (outside -> look at the run before using it)
| cell | expected | basis |
|---|---|---|
| 2% none@best | area 10-30; t50 censored in many seeds (P = Delta area likely) | F86 pt2b 53M 18.4 (8 seeds, Burst); pt999 2pct-none@1e-3 9.4 |
| 2% win-ctx@best | area 60-75 | F86 pt2b 69.9 |
| 4% | between 2% and 7% | |
| 7% (reused) | none@8e-3 65.8, win-ctx@2e-3 89.1 (seeds 0-3) | F87 |
| 14% none@best | area 65-85 | more arithmetic per step |
| 14% win-ctx@best | area 89-93 (prior at the ceiling likely) | 7% prior 89.1 |
Every win-ctx run: prior_on_frac 0.350; bpb 1.19-1.27; none: prior never on.

## Known limits
- (added by the pre-submission amendment, 2026-10-26 19:29 EDT) lambda is fixed, so the prior / LM gradient ratio changes
  with the share (first on-eval 3.5-7.8 at 2%, 1.3-2.5 at 7% in earlier batches): reported per share as a covariate. The
  nominal shares are measured as 2.12 / 4.25 / 8.50 / 17.00%: every read-out output uses the measured values. Both ends
  are biased towards "the lower the share, the larger the gain" (2% / 4%: R censored; 8.5% / 17%: Delta area compressed). [2026-10-26 20:22 EDT text amendment: half of this is wrong, see "Text-only amendment" at the end]
- At >= 7% the prior is at its speed limit: the window opens at 15.0M tokens and the 7% prior's t50 is 19.9-24.2M (one to
  two evals later; real ptlr / ptgw runs). There R_0.5 is about t50(none) / 20M and Delta area about 89 - area(none), so a
  change of the gain with the share mostly measures a change of none's deficit, not of the prior's effect (audit item 2).
  The read-out prints, per share, the prior's t50 after the window opens and its area headroom, both arms' absolute areas
  and log t50 slopes, and a note at the shares where the prior's t50 is within 2 evals of the window's opening.
- The edge rule ("选在边缘时自动往外扩一档") is read as the open-bracket rule (best at an edge AND > 2 points above its
  neighbour); a best lr at an edge within 2 points is confirmed there without expansion (audit item 5): pending the user's
  confirmation.
- The share changes at a fixed 3e8-token budget: more arithmetic = fewer text tokens per row (about 501 / 490 / 469 / 425
  text tokens at k = 1 / 2 / 4 / 8).
- Position: the eval line sits at the end of a val-text row. At k = 1 that is exactly the training line's place and context;
  at k > 1 the first training line follows text (at another position) and the last one sits at the end (after other lines).
  The mismatch grows with k and is part of the dose (as at 7% in F82 / F87).
- Only lr is tuned (warmup 1%, wd, batch fixed). The 7% grid in ptlr was wider (none at 6 lrs) than 3 (+1) here: the 7% none
  is at least as well tuned as the others.
- 4 seeds select the lr; the confirm seeds guard the sign; the slope CI does not include the lr-selection uncertainty (as
  F87 rule 8). 4 points on the x axis.
- The tie-break uses the in-training bpb (65k tokens, noisy), as ptlr.
- 2% none is near its floor and bimodal: its R_0.5 is probably a bound, hence the switch.

## Cost and storage
72 + 24 = 96 jobs x 0.98 L40S-h (ptlr 53M measured: training 3438-3541 s, plus start-up and the dry block) = 94 L40S-h; with
every possible expansion (6 x 4 jobs) at most 118 L40S-h; CPU selection jobs < 0.5 h each. Storage: one saved model per run
(about 210 MB) = 20-25 GB; checkpoints are deleted at the end of each run. The coordinator's estimate was about 70 L40S-h.

## Tests before submission (scripts and logs: llmadd_torch/scale_work/ptshare/; all before any ptshare job)
| test | result |
|---|---|
| CPU references (refs_ptshare.py, CPU job 21028579; every job's own ARGS, --out moved) | for k = 1 / 2 / 4 / 8 x seeds 0-7: the dry rows identical across the two arms, 2k operand sites in each row, the target line only in win-ctx; arith_share.py's replay = rows 0-2 in all 64 dry runs; k = 4 rows = the ptlr GPU job logs (none@8e-3 s0-7) and k = 1 rows = the pt999 2pct-none GPU logs (s0-7), byte for byte; the eval lines (line, held_b11, held_b33, ood_b44, eval-set shapes, model) identical across the four shares for every seed; step-0 lm identical across the arms (seed 0), k = 4 within 5.2e-5 of ptlr's GPU values and k = 1 within 4.4e-5 of the pt999 2% GPU values (8 seeds each) -> tools/ptshare_refs.json. ALL IDENTITIES HOLD |
| measured shares (seeds 0-7, 640 rows each) | k = 1: 2.09-2.15% (mean 2.12%); k = 2: 4.21-4.28% (4.25%); k = 4: 8.47-8.54% (8.50%); k = 8: 16.96-17.08% (17.00%); 10.8-10.9 arithmetic tokens per line at every k |
| jobcheck_ptshare.py (every job the generator can render: 72 main + every expansion / confirm cell = 240) | each equals its ptlr 53M job (same arm and seed) byte for byte after undoing the listed edits and the design substitutions (runs/ptshare -> runs/ptlr, tag, --arith_lines, --lr); the raw token differences are exactly: job name, --nice=2000, the two comments, HOLD / code / out paths, the L40S guard, the dry block, --arith_lines, --lr; bash -n clean (+ the 2 selection job texts); render() refuses a wrong lr, k = 4, a gate arm, seed 8, a repeated --arith_lines, an unintended --lam. JOBCHECK PASS |
| e2e_ptshare.py (the rendered job's bash on CPU, fake nvidia-smi, training line replaced by an echo) | k1 none@8e-3 s0, k8 win-ctx@2e-3 s1, k2 none@3.2e-2 s5 reach the training line with the share printed (2.15 / 17.00 / 4.23%) and the replay = the dry rows; an A100 -> exit 3; ARGS with k + 1 lines -> exit 3; a win-ctx job without its target -> exit 3; arith_share.py edited after the snapshot -> exit 3; HOLD -> exit 0. E2E PASS |
| ptshare_selftest.py (rules unit tests, the references loader, fake runs end to end) | 105 checks: decide() on monotone / no trend / only P significant / only S significant / 8 seeds significant but confirm reversed / S reversed / CI includes 0 / undeterminable or settled bounds / bounded P / < 3 informative shares / positive direction; the switch at 3/8 (and not at 2/8); areas, R_0.5, R_0.9, Delta area and the OLS slope = an independent computation (1e-9); every per-share line carries Delta area, R_0.5, R_0.9 with CIs and censored counts for seeds 0-3 / 4-7 / 8, and the slope table every metric; designed verdicts (A shown, B switch, C not shown, F confirm reversed -> unstable, F2 S reversed -> unstable, G 2% floor left out, H prior-only ceiling note / both at the ceiling left out); selection final / open -> expansion pick / open again -> STOP / closed after the expansion -> confirm / incomplete / withheld; round 2 never expands; reading B on designed bounds (conservative / sign open / resamples against the claim) and a censored 2% median (4/8) giving an upper-bound slope; a win-ctx run without prior gradients -> withheld; a same-args 7% run under another tag -> left out; leave-one-share-out and the prior-t50-after-the-window lines printed. PASS |
| selplumb_ptshare.py (the selection job's bash on fake runs, generator calls echoed) | all final -> 6 confirm picks, --test-only then --submit; mixed -> the expansion + the closed arms' confirms, nothing for the incomplete / withheld arms; no runs or a stale picks file -> nothing submitted; a failing --test-only -> no --submit; a tampered read-out file -> exit 3 before the read-out. SELPLUMB PASS |
| real 7% cells (ptshare_summary.py on runs/ptlr + ptgw with the CPU references) | integrity clean; the sub-grid rule picks none 8e-3 / win-ctx 2e-3 (= F87); Delta area +23.4 [+20.3, +26.5] 4/4 (seeds 0-3), +22.2 (4-7), +22.8 8/8; R_0.5 4.75x, R_0.9 3.13x [2.75, 6.28]: F87's pre-registered numbers reproduced by an independent read-out; the prior's pre-clip LM share median 0.82, clip coefficient 1.000 (the total pre-clip norm never exceeds 1 at the on-evals) |
| regression of the existing read-outs | no existing file changed (git diff origin/main: new files only); ptscale_summary.py (129 lines) and pt2main_summary.py (143 lines) outputs byte-identical between origin/main and this branch |
| template pin (template_pin.py) | make_jobs_ptlr.py's G / ARM / WANT / flags / job f-string are pinned by the sha256[:12] of their unparsed source (c9b834f3679a), not the file: a copy with an appended CELLS amendment passes, a changed template line or ARM fails (audit item 7) |

## Pre-submission adversarial audit (2026-10-26; fresh general-purpose subagent, read-only)
No job-level bug (snapshots, job texts, eval set, seed pairing, R censoring, bound directions all checked). Items and actions:
1. (blocking for the verdict, not the jobs) the global switch to Delta area turns the user's "log R_0.5 for >= 7%" and "14%:
   read by R" into a note -> rule 5 now has readings A and B, both printed, verdict PENDING until the user picks one.
2. (important) at >= 7% the prior is at the window's speed limit -> known limit + descriptive lines (above); to the user.
3. (important) the prior's effective strength changes with k at fixed lambda -> train / eval table row; to the user.
4. (important) the position mismatch grows with k -> already a known limit; quantifiable later by an offline eval of the
   saved models at the training positions (not in this batch).
5. (minor-important) the edge rule interpretation -> known limit, pending the user's confirmation.
6. (minor) the data-rows sentence was wrong -> corrected (shared pair sequence; independent resampling is conservative).
7. (minor) brittleness: the ptlr template pinned by its source parts (an append-only amendment of make_jobs_ptlr.py no longer
   stops a selection job); same-args runs under another tag are left out instead of crashing the read-out. Still by design:
   an incomplete / withheld arm is never retried automatically, and select_r2 is submitted once (both go to the user).
Re-check of the fixes (same auditor): items 1 and 7 fixed; two new issues of the prior-activity check (a diverged run would
have been withheld against rule 2; 'last on-eval geo < first' sits in the noise at high k) and a loose 5% tolerance in
reading B -> fixed as described above (selftest 105 checks: a diverged prior run counts; readings hidden while unset).
8. (minor) the ceiling threshold 90 misses a compressed 88.9 -> kept as pre-registered; the area headroom is printed per share.
9. (minor) slope lines now carry the censored medians and the resample shares bounded each way; the prior-activity check is
   in the validity list; leave-one-share-out slopes are printed (descriptive).

## Pre-submission amendment (2026-10-26 19:29 EDT; before any ptshare job or result)
Decided by the coordinator, to whom the user delegated the technical calls (the user may overrule; an overrule comes as a
further amendment). Supersedes the corresponding text above where they differ.
1. Cost: about 95 L40S-h accepted (nice 2000, low priority, L40S); with every expansion at most about 118 L40S-h, also
   accepted. Never more: the generator refuses any stage that would bring the batch above 120 training jobs
   (ptshare_design.MAX_JOBS = 72 main + 6 arms x 4 expansion + 6 arms x 4 confirm = 118 L40S-h at 0.98 h).
2. Rule 5 = reading A, fixed now (ptshare_design.RULE5_READING = "A"):
   - P = Delta area for the whole slope (defined at every share). The 3/8 switch is no longer used; none's t50 censored
     counts are still printed.
   - S = log R_0.5 with a censored R used as its bound (reading B's treatment), direction agreement only; a bound can
     support one direction only, and a bound that leaves the direction open counts as "不一致" (-> "方向不稳").
   - Main test, guardrail (P on seeds 4-7 only, same sign) and the verdict words are unchanged.
   - Both ends are biased the same way: at 2% / 4% none's R is censored (a lower bound); at 8.5% / 17% the prior sits near
     its ceiling and the area gain is compressed. Both enlarge "the lower the share, the larger the gain". [2026-10-26 20:22 EDT text amendment: half of this is wrong, see "Text-only amendment" at the end]
     A shown negative
     slope is therefore written "增益随占比下降而增大，其中多少来自 none 的缺口、多少来自先验，看两组 arm 的绝对面积", and the
     attribution line (both arms' absolute-area slopes with CIs and the per-share absolute areas) is mandatory with it.
3. Edge rule confirmed: expand only when the best lr is at a grid edge and beats its neighbour by more than 2 points (F87's
   open-bracket definition).
4. Known limits accepted (added to "Known limits" below by this amendment):
   - At high shares the prior is near its ceiling (above).
   - lambda is fixed, so the prior / LM gradient ratio changes with the share (first on-eval: 2% 3.5-7.8, 7% 1.3-2.5, earlier
     batches); the read-out reports it per share as a covariate (median first-on-eval and all-on-eval |g_prior| / |g_lm|,
     and the slope of its log against ln(share)), descriptive. [2026-10-26 20:22 EDT: the ratio also depends on lr; see "Text-only amendment"]
   - The nominal 2 / 4 / 7 / 14% are measured as 2.12 / 4.25 / 8.50 / 17.00% (CPU replay, seeds 0-7); every read-out output
     names a share by its measured value (ptshare_design.share_label). The nominal names remain only in tags and job names.
Code: read-out branch arith-dose (this amendment's commit), ptshare_selftest.py 109 checks PASS (reading A with P = Delta area
in every scenario, the mandatory attribution line and its identity Delta-area slope = prior slope - none slope, the covariate
line, measured labels in every output, only the chosen reading computed without --show_both); the job texts are unchanged
except that they point to the new read-out snapshot; minimal fresh audit of the read-out diff after submission.

## Text-only amendment (2026-10-26 20:22 EDT; after submission, before any ptshare result or read-out)
Coordinator, after the post-submission minimal audit found the bias sentence half wrong. Only text and descriptive lines change;
the verdict logic is line-for-line unchanged (ptshare_rules.py, ptshare_design.py, ptshare_pick.py: no diff; in
ptshare_summary.py only the wording string, the comments, the covariate print lines, one new descriptive covariate line and
three nominal labels in printed notes). No new audit (coordinator): text-only, and the regression below shows identical numbers
and verdicts.
1. Bias sentence (replaces the two marked sentences above; the printed wording uses it):
   「高占比端（8.50% / 17.00%）先验接近上限，面积差被压缩，这会放大『占比越低增益越大』。低占比端（2.12% / 4.25%）R 被删失，
   只是下界，这会让次指标 R 的负斜率显得更小，是保守的。主指标面积差不用 R。」
2. Covariate: the prior / LM gradient ratio changes with lr (8.50%: 1.58 / 1.18 / 0.95 at lr 1e-3 / 2e-3 / 4e-3), so across
   shares with different selected lrs it mixes lr and share. The read-out prints the ratio at each share's selected lr and,
   separately, at one lr for every share (win-ctx@2e-3, the 8.50% selected lr, seeds 0-3; none@8e-3 has no prior gradient);
   both descriptive. The "7% 1.3-2.5" quoted above is ptgw main-W at lr 1e-3.
3. P = Delta area at every share is confirmed as the coordinator's intent.
Regression (llmadd_torch/scale_work/ptshare/reg_text.py; old = runs/ptshare/readout_a4de096, kept): on all 15 selftest scenario
roots and the real data the JSON is identical except the text fields, and every differing printed line is a changed text line;
selftest 110 checks PASS (new: the wording says the low end is conservative, the one-lr covariate line).
