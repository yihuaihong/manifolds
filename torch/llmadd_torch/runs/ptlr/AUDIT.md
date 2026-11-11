# ptlr — lr brackets: every arm at its own best lr, 53M and 190M (2026-10-16)

Plan: ~/Research_Projects/LLM_addition_review/notes/next_phase_synthesis.md, batch 1, T1 (53M bracket) and T3 (190M);
design notes next_phase_designs.json (scaling E1 / E2, lowshare E0). User decisions 2026-10-16: the 2% share
(--arith_lines 1) becomes the MAIN setting and the 5% window (0.05, 0.4) the main recipe; every arm, the prior too,
gets its own best lr. This batch brackets lr at 7% (the setting where none is off the floor, so a best lr for none is
measurable); the 2% main-setting row at lr 1e-3 is T2 (2pct-win-ctx, in runs/pt999, rule below). The 2% lr bracket
(the main setting's best-vs-best) is NOT in this batch: Burst pt2main (llmadd-burst tools/make_pt2main_tasks.py,
intent spec report/audits/pt2main_AUDIT.md), tasks 350-353 (53M 2%: none and win-ctx at 1e-3 / 2e-3 / 4e-3) and
354-357 (190M 2%: none at 6e-4 / 1.2e-3 / 2.4e-3, win-ctx at 6e-4 / 1.2e-3), self-contained on Burst (its own none
cells, no pairing with Torch). Any 2% "speed-up survives tuning" claim comes from pt2main only, never from T2
(T2's none is at the untuned lr 1e-3). If pt2main does not run, the 2% best lr is open: say so.

Question: is the speed-up of the helix3 prior an artefact of an under-tuned baseline? (53M none is still rising
at lr 2e-3: area 35.0 / 50.9 / 58.8 at 5e-4 / 1e-3 / 2e-3 on seeds 0-3, while gate-ctx is flat 85.7 / 86.9 / 87.2;
190M none ran at one lr, 6e-4: 39.4 +- 12.2.) Answer it best-vs-best, with the selection rule fixed before any result.

Code: tools/pretrain_range.py at the commit in code/COMMIT = 64abc19, extracted with `git archive` (a later HEAD
cannot change it); code_sha a3bce3ab63c1 = the ptgw snapshot (all of tools/ identical to runs/ptgw/code except the
analysis scripts offline_lm.py / ptgw_summary.py). Generator: llmadd_torch/make_jobs_ptlr.py (asserts the snapshot
sha and every cell's args). No trainer change. Analysis: tools/pt999_summary.py (per cell) and the new
tools/ptscale_summary.py (A1: best-lr selection, KM t_theta, bootstrap; written and audited before results).

## Cells (L40S, seeds 0-3, --save_model, --ckpt_every 2000; 7% = --arith_lines 4; ctx = operands in the lines)
| tag | args beyond the pt999 base | time | runs |
|---|---|---|---|
| 53m-lr4e-3-7pct-none | --target none --lr 4e-3 | 03:00 | 4 |
| 53m-lr4e-3-7pct-gate-ctx | --target helix3 --schedule gate --gate_thr 0.5 --site ctx --lr 4e-3 | 03:00 | 4 |
| 53m-lr2e-3-7pct-win-ctx | --target helix3 --schedule window --window 0.05,0.4 --site ctx --lr 2e-3 | 03:00 | 4 |
| 53m-lr4e-3-7pct-win-ctx | same window, --lr 4e-3 | 03:00 | 4 |
| 190m-lr1.2e-3-7pct-none | --target none --d_model 768 --n_layers 12 --n_heads 12 --d_ff 3072 --lr 1.2e-3 | 06:00 | 4 |
| 190m-lr2.4e-3-7pct-none | same model, --lr 2.4e-3 | 06:00 | 4 |
| 190m-lr1.2e-3-7pct-win-ctx | window as above, same model, --lr 1.2e-3 | 06:00 | 4 |
| 53m-lr8e-3-7pct-none | --target none --lr 8e-3 | 03:00 | 4 |
Base (pt999): 53M d384 x 6, seq 512, bs 32, 3e8 tokens = 18,310 steps, 1% warmup (183 steps), cosine to 0.1x,
AdamW (0.9, 0.95) wd 0.1, clip 1.0, lambda 5, prior layers 1/3 and 2/3 of the depth (53M 2,4; 190M 4,8), bf16 autocast.
190M: gradient checkpointing on (auto above 150M parameters). 32 runs; 20 x 0.96 + 12 x 2.6 = 50.4 L40S-h.
Time limits ~3x / ~2.3x the measured run time (53M 0.95-0.97 h, 190M 2.58-2.61 h), short so the jobs backfill.
53m-lr8e-3-7pct-none was conditional (iff none@4e-3 > none@2e-3 + 2) in the first version of this file; moved into the
batch on 2026-10-16 after the pre-submission audit, before any submission or result: that trigger is inside the seed
noise (paired none 2e-3 - 1e-3 area on seeds 0-3 = +4.8 / +3.1 / -2.1 / +25.7), so whether it fired would be close to
random, and each trigger costs a ~2-day Torch queue trip (user: maximise parallelism). 3.8 L40S-h.

CONDITIONAL (listed in make_jobs_ptlr.py as a comment, not generated; added when the rule fires, or up front if the
user queues it now; both are the user's call, 10.4 L40S-h each):
- 190m-lr4.8e-3-7pct-none x 4 iff none@2.4e-3 is the best 190M none lr by more than 2 points (same noise caveat as
  above: 190M none SD 12.2, so a 2-point trigger is close to random).
- Not pre-registered: 190m-lr2.4e-3-7pct-win-ctx x 4. If W@1.2e-3 beats W@6e-4 by > 2 points, the prior's 190M bracket
  is open; that goes to the user (no automatic cell).
- 53M: if none@8e-3 is the best none lr by > 2 points, report "bracket not closed above 8e-3" (no 1.6e-2).

## Reused references (same data, split, evals; L40S; pairing by seed = same init, data stream, held-out split)
| arm @ lr | where | snapshot (code_sha) | seeds | model saved | status 2026-10-16 |
|---|---|---|---|---|---|
| 53M none @ 5e-4 / 1e-3 / 2e-3 | pt999 sens-lr5e-4-none / 7pct-none / sens-lr2e-3-none | 79e9b410dcc3 | 4 / 8 / 4 | no / yes / no | done |
| 53M gate-ctx @ 5e-4 / 1e-3 / 2e-3 | pt999 sens-lr5e-4-gate-ctx / 7pct-gate-ctx / sens-lr2e-3-gate-ctx | 79e9b410dcc3 | 4 / 8 / 4 | no / yes / no | done |
| 53M win-ctx @ 5e-4 / 1e-3 | ptgw lr5e-4-W / main-W | a3bce3ab63c1 (= ptlr) | 4 / 4 | yes | queued (1 of 8 done) |
| 53M win-ctx @ 1e-3 (descriptive) | pt999b 7pct-win-ctx | 44ce3fef0c29 | 4 | no | queued |
| 190M none / gate-ctx @ 6e-4 | ptsize999 190m-7pct-none / -gate-ctx | 79e9b410dcc3 | 4 | yes | done |
| 190M win-ctx @ 6e-4 | ptsize999 190m-7pct-win-ctx | 79e9b410dcc3 | 4 | yes | queued |
Cross-snapshot pairing is justified: from 79e9b410dcc3 to a3bce3ab63c1 the none / default-gate / ctx-window training
paths are unchanged (0279f37: ctx path 117/117 values identical on CPU; 2f99b06: default gate bit-identical to
7dd06f2, 117/117; the diff adds only --site prefix, --gate_start / --gate_hyst with defaults that reproduce the old
rule, and prior_events logging); pretrain_helix / numlib / digit_split_data / run_guard are unchanged; step-0 lm is
identical across snapshots (11.60809 / 11.59907 / 11.57956 / 11.59710 for seeds 0-3 in pt999, ptrangeg and ptgw).

## Metrics (unique definitions; implementations tools/pt999_summary.py, reused by tools/ptscale_summary.py)
- held_bXY: exact match of ALL answer tokens AND the newline, teacher forced, argmax at each position, of a held-out
  pair (both orders never trained) in format 0 ("Q: a + b = "), placed at the END of a 512-token row after a val-text
  chunk (the training lines' position); n = 150 per bucket (b11: 37). pretrain_range.py:evaluate().
- area (PRIMARY): 100 x mean over ALL evals of the 6-bucket mean of held_b11..held_b33 (evals every 250 steps = 4.1M
  tokens plus the last step: 75 evals at 3e8 tokens). pt999_summary.py:area. A speed metric (the integral of the
  curve), not final ability. Same quantity as ptgw's "area6"; NOT ptgw's b33_area (3+3 bucket only).
- t90: tokens (M) until the 6-bucket held mean first reaches 0.9, linear interpolation between evals
  (pt999_summary.py:t90); t_theta the same at theta = 0.5. Runs that never reach theta are CENSORED at 300M tokens
  (never averaged as inf).
- per-bucket final: held_bXY at the last eval (step 18,309); b13 / b23 / b33 also split by sum < 1000 / >= 1000.
- bpb: final in-training val_bits_per_byte (8 x 16 fixed val windows, 65k tokens: noisy); bpb_offline =
  offline_lm.py from the ptlr snapshot (runs/ptlr/code/tools, 64abc19, sha256 226f4cac70ea; NOT the repo's, which
  batch 2 changes; the pt999 snapshot has no offline_lm.py, so the same file also serves T2's 2pct-win-ctx /
  2pct-none / 2pct-gate-ctx; record its sha256 next to each output json) on ALL of val.bin, used only if its double
  path |orig_minus_logged| < 0.002.
- prior_on_frac / prior_used_frac at the end; grad_lm / grad_prior while the prior is on (spike diagnosis).

## Pre-registered rules (written 2026-10-16, before any ptlr result; ptlr/out does not exist)
Revised the same day, still before any submission or result: none@8e-3 moved from conditional into the batch; rule 2
made explicit for unfinished runs; the args keying below.
Cells are keyed on r["args"] (target, schedule, window, site, arith_lines, lr, d_model / n_layers), never on parsed
tags: only ptlr tags carry the lr (pt999 7pct-none / sens-lr2e-3-none, ptgw main-W, ptsize999 190m-7pct-none do not).
T1, 53M (arms none, gate-ctx, win-ctx; lrs 5e-4, 1e-3, 2e-3, 4e-3, and 8e-3 for none):
1. Best lr per arm = argmax over its lrs of the mean area over seeds 0-3 (8-seed cells truncated to seeds 0-3, so
   every lr has the same n; seeds 4-7 of a selected 1e-3 cell stay fresh for confirmation). If the top two are
   within 2 points: the lower bpb wins (bpb_offline when both cells saved models, else the in-training final bpb for
   both; say which).
2. A diverged run (NaN loss, or final val_loss > the same seed's none@1e-3 final val_loss + 0.5 nats) counts as a
   failure of that lr: it stays in the mean with its area (not excluded) and is reported. A cell with fewer than 4
   finished runs is not selectable until the missing seed is accounted for (pt999_summary.py prints "unfinished:" and
   SKIPS such a run, so its n can drop silently): a run that stopped with an error of its own (not preemption, node
   failure or a HOLD) is a failure of that lr, its area taken over all 75 eval slots with the missing evals scored 0,
   never dropped; an infrastructure loss is resubmitted (it resumes from its ckpt).
3. "Speed-up survives tuning" iff W@best - none@best area > 0 with paired t p < .05 and >= 3/4 wins on seeds 0-3,
   AND the same sign with >= 3/4 wins on confirm seeds 4-7 (batch 2, T1b: seeds 4-7 of none@best and W@best; for a
   1e-3 cell that already has them, the existing seeds). gate-ctx@best vs none@best: the same test, secondary.
   Until the confirm seeds are in, any best-vs-best number is labelled PROVISIONAL (winner's curse).
4. The paper's "N x faster" = R_0.9 = KM median t90(none@best) / KM median t90(W@best), seed-bootstrap 90% CI,
   censored runs at the budget; if none's KM median is censored, report a lower bound. R_0.5 alongside.
5. Descriptive: the prior is "lr-flat" if its area spread over its lrs is <= 3 points (gate-ctx so far 1.5).
T3, 190M (none at 6e-4, 1.2e-3, 2.4e-3 [, 4.8e-3 if triggered]; W at 6e-4, 1.2e-3; gate-ctx at 6e-4 only):
6. Best lr per arm by rule 1-2 (divergence reference: the same seed's 190m-7pct-none@6e-4).
7. "Survives tuning at 190M" iff prior@best - none@best area > 0 (paired t p < .05, >= 3/4 wins) AND the R_0.5
   bootstrap 90% CI excludes 1; tested for W@best (primary) and gate-ctx@6e-4 (secondary; its lr is not tuned:
   conservative for the prior, say so). If the best-vs-best gap is < 10 points: seeds 4-7 (8 runs) before any claim.
8. Size: only through ptscale_summary's slope of log R vs log N_nonemb, called "grows" / "shrinks" only if its 90%
   CI excludes 0, and called a trend.
T2, 53M 2% (runs/pt999 2pct-win-ctx, seeds 0-7, snapshot 79e9b410dcc3; references 2pct-none, 2pct-gate-ctx done,
2pct-win-text queued; all lr 1e-3):
9. Δarea(2pct-win-ctx - 2pct-none) >= +30, >= 7/8 wins, exact sign-flip p < .05, and offline Δbpb <= +0.015 (gate:
   +0.045) -> the 2% row uses the window; the gate is written as "degenerates to a whole-run prior at 2%" (used 0.84).
10. win-ctx area < 2pct-gate-ctx area - 10 (paired mean) -> a trade-off is reported, no winner.
11. No speed RATIOS at 2% (none's area 9.4 < 20; t90 not reached). The 2% best lr is not tested here (Burst
    pt2main, see the top): T2's Δarea is "at lr 1e-3 for both arms", never written as a tuned comparison.

## Predictions / expected ranges (outside -> check the run before using it)
| cell | expected area | basis | check if |
|---|---|---|---|
| 53M none@4e-3 | ~58 (45-68); P(> none@2e-3 + 2) ~ 0.3 | 35.0 / 50.9 / 58.8 rising, flattening | < 30, NaN, loss spike not recovered |
| 53M none@8e-3 | ~50 (25-68) or diverged; P(best by > 2) ~ 0.15 | 8x the base lr with 183 warmup steps | NaN, loss spike not recovered (a failure by rule 2, still reported) |
| 53M gate-ctx@4e-3 | 84-89 | 85.7 / 86.9 / 87.2 | < 80 |
| 53M W@2e-3, W@4e-3 | 86-91 | W@1e-3 88.9 (ptrangeg, older snapshot), W@5e-4 s1 87.3 (ptgw) | < 82 |
| 190M none@1.2e-3 | ~55 (30-80), bimodal | none@6e-4 39.4 (s0 19.1) | NaN |
| 190M none@2.4e-3 | 35-85, higher spike risk | | NaN |
| 190M W@1.2e-3 | 87-92 | gate-ctx@6e-4 90.0 | < 80 |
| 2pct-win-ctx | 55-72; Δbpb vs none +0.005..+0.015 | 2pct-gate-ctx 59.1, 2pct-win-text 65.3 (2 seeds) | < 40 |
Other expected values: final bpb 53M 1.19-1.24, 190M 1.11-1.15; W prior_on_frac = 0.350 (prior on at steps
916..7323 = 6408 of 18,310); gate-ctx@4e-3 used 0.07-0.25; none used 0; t90 W 53M ~45-80M tokens.

## Train vs eval conditions (identical to runs/pt999/AUDIT.md; only lr / size differ between arms)
| dimension | train | eval | same? |
|---|---|---|---|
| sequence position | 4 lines at the end of a 512-token row after FineWeb text | one line at the end after val text | yes for the last line; the first line also follows text |
| template | 4 formats, random | format 0 ("Q: a + b = ") | subset |
| value range | operands 0..999, buckets balanced | same buckets (ood_b4X = 4-digit, labelled extrapolation) | yes |
| data / split | train.bin text; held pairs never in lines (draw_pair rejects them) | val.bin text; held pairs | disjoint by construction (b11: 13-17 of 20 pairs occur in natural text, F-noted) |
| model state | train, bf16 autocast, grad ckpt (190M) | eval mode, bf16 autocast | yes |

## Controls and validity (checked on the results, before any number is used)
- Known answer, per run: step-0 lm (lr-independent: measured before the first update) equals the same-seed reference
  within 1e-3: 53M 7% seeds 0-3 11.60809 / 11.59907 / 11.57956 / 11.59710; 190M seeds 0-3 11.65420 / 11.67774 /
  11.65268 / 11.64366; 2% seeds 0-7 11.61647 / 11.58723 / 11.56547 / 11.60211 / 11.59495 / 11.56812 / 11.58452 / 11.59349.
- Every run: code_sha a3bce3ab63c1 (ptlr) / 79e9b410dcc3 (2pct-win-ctx); args per tag as in the table; steps 18,310;
  75 evals; grad_ckpt_on True for 190M, False for 53M; W runs prior_on_frac 0.350 and (ptlr) prior_events
  [[916, 1], [7324, 0]]; none runs prior_used_frac 0.
- Ceiling: seen_bXY near 1 at the end for every non-diverged run.
- Negative control: the shuffled prior is NOT repeated at the new lrs; geometry specificity rests on 7pct-shuf-ctx at
  1e-3 (area -23.6 vs none). Say so if a best lr other than 1e-3 is used for the prior.

## Known limits
4 seeds per cell (none is bimodal, SD 7.8 at 53M, 12.2 at 190M): selection on seeds 0-3 inflates the gap, hence the
confirm seeds; pt999 sens-* cells saved no models (in-training bpb only); one share (7%) for the bracket while 2% is
now the main setting (the 2% bracket is Burst pt2main, another site and another piece; a Torch 2% bracket exists only
if the user adds one); 190M gate-ctx only at 6e-4; the 190M W bracket has two points; all at 3e8 tokens (the best lr
falls with budget). A kill between the final ckpt delete and the model save (pretrain_range.py:594-596, tiny window)
leaves a finished JSON without a model: that run then has no bpb_offline; use the in-training bpb for its cell and say so.

## Audit (minimal mode, 2026-10-16, before submission)
i double path: - (at analysis: offline bpb vs logged bpb, |diff| < 0.002). ii asserts: done (generator asserts the
snapshot sha + COMMIT and, per run, every intended flag, the absence of unintended flags and no repeated flag; every
sbatch re-checks sha256[:12] of code/tools/pretrain_range.py = a3bce3ab63c1 at start and exits 3 before any GPU work
otherwise).
iii known answers: --dry_inputs on CPU for seed 0 of every new tag is byte-identical to its reference's dry run
(53m none = pt999 7pct-none; 53m W / gate = ptgw main-W, which prints the same rows; 190m none / W = ptsize999
190m-7pct-none / -win-ctx; 2pct-win-ctx = 2pct-none + the "target features" line). Step-0 lm on CPU (--stop_after 0,
--lm_chunk_tokens 8192, --out in the scratchpad): 53m-lr4e-3-7pct-none s0 (ptlr snapshot) 11.608102798 = pt999
7pct-none s0 (pt999 snapshot) 11.608102798 bit-identical, GPU reference 11.60809; 2pct-win-ctx s0 11.616468 vs GPU
2pct-none s0 11.61647, prior_on 0 at step 0 (window opens at 916). 190M not run on CPU (host memory). Post-start:
step-0 lm in the first eval of every run vs the values above. iv: -. v: independent pre-submission audit 2026-10-16
(nothing blocking). Acted on: the 2% bracket lives in Burst pt2main (pointer above; checked in its generator: 53M none /
win-ctx at 1e-3 / 2e-3 / 4e-3, 190M none at 3 lrs, win-ctx at 2); none@8e-3 queued now; runtime sha check in every
sbatch; offline_lm pinned to the snapshot; unfinished-run rule (rule 2); cells keyed on args. Re-verified after the
fixes: 32 jobs, bash -n clean; the 28 earlier sbatch files are byte-identical to their first rendering once the 2
sha-check lines are removed; the sha check passes on the snapshot and exits 3 on a tampered copy; fault injection
(wrong site, repeated --lr, --schedule on none, duplicate tag, wrong snapshot sha) still stops the generator; tag
53m-lr8e-3-7pct-none occurs nowhere in runs/* or the Burst task tags; its seed-0 --dry_inputs log is byte-identical
to 53m-lr4e-3-7pct-none's and pt999 7pct-none's; its CPU step-0 lm 11.608102798 = 53m-lr4e-3-7pct-none s0 (GPU
reference 11.60809); pt999_summary.py runs over fake copies with the new tag.
vi: checked (position / template / range table above; gradient-checkpointing hooks: bit-identical test in pt999
audit; silent prior failure: prior_on_frac / used asserted on results). vii: dry_inputs prints 3 decoded training rows
with the prior sites marked and the eval lines (held_b11 / held_b33 / ood_b44) at the end of a val-text row.

## 2026-10-16 amendment (before any submission or result)
User: maximise parallelism; every arm, the prior too, at its own best lr. The two CONDITIONAL 190M cells are queued up
front: none@4.8e-3 and win-ctx@2.4e-3, seeds 0-3 (2 x 10.4 L40S-h). Their trigger rules no longer gate submission; the
best-lr selection rule (argmax mean area over seeds 0-3; ties within 2 -> lower offline bpb) is unchanged and now runs over
the wider grids. 40 ptlr jobs + 8 pt999 2pct-win-ctx jobs.

## 2026-10-16 amendment 2: the read-out is code, one definition set (before any ptlr result)
Written and tested before any ptlr result existed (runs/ptlr/out does not exist; T2 2pct-win-ctx and ptgw main-W have no
run yet). Read-out: llmadd-burst `tools/ptscale_summary.py`; every definition comes from llmadd-burst
`tools/readout_rules.py`, the single definition set shared with the Burst pt2main read-out (report/audits/pt2main_AUDIT.md,
amendment 3). After the read-out's own audit (same day, before any result) the code applies every rule as pre-registered;
where the rule text was silent it fills the gap with a choice marked **INTERPRETATION** below, and where the shared set
would change a rule it prints the alternative as **PROPOSED**; both wait for the user's sign-off.
- Rule 1 tie-break, as pre-registered: the top two lrs if within 2 area points -> the lower mean bpb, offline when every
  run of both has a valid full-val entry (|orig_minus_logged| < 0.002; files passed with --offline, matched by the run
  path or, for a moved runs root, by <source>/out/<cell>/<file>, and checked against tag / seed / logged final bpb),
  else the in-training bpb for both (marked '*'); a non-finite bpb is the worst. **PROPOSED** (pt2main's rule 1):
  every lr within 2 of the top is tied; printed next to the best lr when it picks another one.
- Rule 2: diverged = the first eval with a non-finite lm / val_loss / geo / grad_prior; final val_loss > the same seed's
  none@base lr + 0.5 nats is flagged 'unstable'; both stay in the mean with their area, as before. A run that stopped on
  an error of its own is listed in runs/ptlr/OWN_FAILURES.txt ('<tag> <seed>'): its area over all 75 eval slots with
  the missing evals scored 0, bpb worst; a listed run that crashed before writing any JSON counts as area 0 when its tag
  names one cell; an entry that matches nothing (unknown tag, seed outside 0-7, a finished run) is printed as a WARNING;
  any other unfinished run keeps its cell 'incomplete' (not selectable), and the note says which source the cell uses
  meanwhile.
- Rule 3 / 7: "survives tuning" = mean > 0, >= 3/4 wins, paired t p < .05 on seeds 0-3, and a paper claim needs the
  confirm seeds 4-7 (same sign, >= 3/4 wins); until they exist 'survives (unconfirmed)' and PROVISIONAL (= rule 3).
  Rule 7 adds: the R_0.5 bootstrap 90% CI above 1. INTERPRETATION: with none at a single lr the line reads 'not a
  tuning test'.
- Rule 4: the KM-median wording becomes the shared definition: median t90 with a run that never reaches 0.9 counted
  as +inf (censored at the 300M budget); '>=' when the none median is censored (>= half of its runs), no ratio when
  both are; INTERPRETATION (rule 4 is silent): '<=' when only the prior's is; this equals the Kaplan-Meier median for the common censoring time (asserted in
  code); seed-bootstrap 90% CI with 'n=4: indicative only'. R_0.9 and R_0.5 at the area-best lrs (rule 4); the lrs picked
  by median t90 are printed as a descriptive line.
- Rule 8: INTERPRETATION (rule 8 says "log R"): R = R_0.5 (the ratio rule 7 already uses; 190M none@6e-4 reaches 0.9
  in 1/4 seeds), over the sizes where none has >= 2 lrs; R_0.9 printed when no size's ratio is a bound (descriptive);
  CI: seeds resampled within each size, arms paired, 10000 resamples, default_rng(0). The gate-ctx slope is secondary
  and its line says 'UNTUNED, not a finding' while its 190M cell has a single lr.
- Cells are keyed on args (as above); a cell's seeds 0-3 come from the first source with all four, in the order ptlr,
  (batch 2's confirm dir), ptgw, pt999, ptsize999, ptrangeg, pt999b. INTERPRETATION: 53M W@1e-3 = ptgw main-W once its
  4 seeds are in, until then ptrangeg (same flags, older snapshot); the others print as replicates with the paired
  difference. The
  'Controls' are enforced: step-0 lm (vs the values above and the size-seed median, 1e-3), code_sha per source, 18310
  steps / 75 evals, grad_ckpt_on, W prior_on_frac / prior_events, none prior_used_frac 0; a failure withholds every
  verdict using the cell ('WITHHELD: <reason>'). Ceiling (seen_bXY near 1): INTERPRETATION, descriptive only: every
  counted non-diverged run below 0.9 is printed, nothing withheld (no threshold above, and reference cells already sit
  below any 'near 1': pt999 sens-lr5e-4-none s1 0.75, ptsize999 190m-7pct-none s0 0.88).
- A rule 3 / 7 verdict whose none or prior arm has an open bracket also prints 'NOT FINAL' (the conditional cells and
  'bracket not closed' above); the verdict itself is unchanged.
- T2 rules 9-11 are in ptscale_summary.py (section '## T2'; added after the read-out's audit, before any T2 result):
  cells keyed on args (53M, --arith_lines 1, lr 1e-3; none / gate-ctx / win-ctx), seeds 0-7 from the first source with
  all eight; every run must pass readout_rules.check_2pct (the 'Controls' for 2pct-*: code_sha 79e9b410dcc3, 18310
  steps / 75 evals, the same seed's 2pct-none args except target / schedule, step-0 lm vs the 2% values, win-ctx
  prior_on_frac 6408 / 18310, none never on), else WITHHELD. Rule 9 = Delta area >= +30, >= 7/8 wins, exact two-sided
  sign-flip p < .05 (all 256 sign patterns; = ptgw_summary.signflip_p; INTERPRETATION: two-sided) and offline Delta bpb
  <= +0.015 (rule 9 'incomplete' until all 16 offline values exist; in-training printed as descriptive; gate - none
  printed as the +0.045 reference); rule 10 = paired mean win-ctx - gate-ctx < -10 -> 'trade-off, no winner'; rule 11
  = no ratio, none's area and t90 count printed, every Delta labelled 'at lr 1e-3 for both arms'. The xsite P1b-win-ctx
  rule uses the same check_2pct.
Tests on fake results (scratch `readout/fake_ptlr.py`: fake ptlr / ptgw / ptsize999 190m-7pct-win-ctx runs from the same
seed's real run JSONs with designed linear curves; the real pt999 / ptrangeg / ptsize999 cells unchanged), designed ->
printed: 53M none best 4e-3 (66.1; none@8e-3 31.6 with an own failure s0, a divergence s2 at step 5000 and an unstable
s3, all listed); gate-ctx best 4e-3 by the in-training tie-break over 5e-4 / 1e-3 / 2e-3 ('*'); W best 2e-3 by the
offline tie-break over 4e-3 (edge note); W@1e-3 = ptgw main-W, ptrangeg as replicate (-0.6); W - none +25.0 ->
'survives (unconfirmed)'; R_0.9 = R_0.5 = 4.06x (designed 183.6 / 45.225 = 4.060); 190M none best 2.4e-3, W 6e-4 by the
offline tie-break, R_0.5 4.06x -> slope 0.000, CI -0.050..+0.050 -> no trend; trend variant (190M W slower):
R_0.5 2.03x -> slope -0.333 (designed ln 0.5 / ln(113.28M / 14.17M) = -0.333) -> 'shrinks with size (a trend)'; confirm
seeds 4-7 (extra dir) +24.3 4/4 -> 'SURVIVES tuning (confirmed)'; confirm seeds against -> 'does NOT survive (the confirm
seeds 4-7 disagree)'; without OWN_FAILURES.txt -> none@8e-3 incomplete -> no 53M verdict, no slope; a step-0 lm +0.005 in
gate-ctx@4e-3 s1 -> gate-ctx's best lr and its verdict WITHHELD, the rest unchanged. The ptrangeg gate replicate of
pt999 7pct-gate-ctx prints +0.0 (1/4 wins), the known replicate spread (+0.39 / +0.33 / -1.38 / +0.52).
After the read-out's audit (designed -> printed): gate-ctx best 4e-3 by the pre-registered top-two tie-break (2e-3 /
4e-3, in-training *; same lr as before); T2 fake (2pct-win-ctx from the real 2pct-gate-ctx runs, window args, designed
linear curves): +58.84 / 8/8 / Delta bpb +0.010 -> +58.8, sign-flip p 0.0078 (= 2/256), 'rule 9 met', rule 10 +9.1 ->
no trade-off; Delta bpb +0.020 -> 'rule 9 NOT met (offline Delta bpb)'; win-ctx at 45.65 -> rule 10 -13.4 -> 'trade-off,
no winner' (rule 9 still met, +36.3); win-ctx s3 prior_on_frac 0.40 -> rules 9 and 10 WITHHELD; no offline for
2pct-none -> rule 9 'incomplete'; planted controls (code_sha in W@4e-3 s0, prior_on_frac in W@2e-3 s1, 74 evals in
190M none@1.2e-3 s2) -> 53M W best lr / rule 3 and 190M none best lr / rule 7 / slope WITHHELD; OWN_FAILURES with a run
that has no JSON (W@2e-3 s3) -> area 68.3 (= 91.1 x 3/4), W best 4e-3 with an open bracket, plus WARNINGs for seed 9 and
an unknown tag; the fake root copied elsewhere -> the offline tie-breaks unchanged (path-tail match); ptgw main-W s2
unfinished -> 'its cell's seeds 0-3 come from ptrangeg meanwhile'.

## 2026-10-17 amendment 3: cells added after the first read-out (tools/ptscale_summary.py), under the open-bracket and confirm-seed rules, before any of them ran
- **none's bracket is open at the upper edge at both sizes.**
  - 53M: 8e-3 gives 65.8, above 4e-3 at 57.9.
  - 190M: 4.8e-3 gives 64.3, above 1.2e-3 at 56.4.
  - Added for seeds 0-3: 53M none@1.6e-2 and 190M none@9.6e-3.
- **Confirm seeds 4-7**, for none at the current best lr and the next one, and for win-ctx at its tied best lrs:
  - 53M none@8e-3, none@1.6e-2, win-ctx@2e-3 and win-ctx@1e-3;
  - 190M none@4.8e-3 and win-ctx@1.2e-3.
  - If 190M none@9.6e-3 becomes the best lr, its confirm seeds follow.
- **505M (T4 of the plan).** none ran only at 3e-4 (17.1), far below the 190M none optimum (64.3). Added for seeds 0-3:
  none@1.2e-3, none@4.8e-3 and win-ctx@1.2e-3 (4x and 16x the default, the 190M optimum ratios); time limit 13 h.
- 44 jobs, about 135 L40S-h. The selection rules are unchanged. The verdicts stay PROVISIONAL / NOT FINAL until the
  brackets close and the confirm seeds agree.

## 2026-10-18 amendment 4 (after the read-out, before these cells ran)
- 505M none: the bracket is open at the upper edge (4.8e-3: 57.6 > 1.2e-3: 47.3), so none@9.6e-3 is added for seeds 0-3.
- 505M win-ctx@1.2e-3: its confirm seeds 4-7 run now. none's confirm seeds follow once its bracket closes.
- 53M and 190M are final under the rules: the brackets are closed and the confirm seeds agree.

## Amendment 5 (2026-10-19, after the 505M read-out, before any of the cells below ran)
- Read-out (tools/ptscale_summary.py, ptlr + ptsize999): 505M none 3e-4 17.1, 1.2e-3 47.3, 4.8e-3 57.6, 9.6e-3 62.1 (seeds 0-3); win-ctx 1.2e-3 89.6 (seeds 4-7 89.8). none's bracket is open at the upper edge -> the pre-registered doubling: none@1.92e-2, seeds 0-3.
- Queued up front in the same step (user 2026-10-16 "maximise parallelism"; arXiv freeze 10-21): none@9.6e-3 confirm seeds 4-7. If 1.92e-2 beats 9.6e-3 by > 2 (rule 1), its own confirm seeds 4-7 are the next step and the 9.6e-3 seeds 4-7 are reported as descriptive; if not, they are the confirm seeds of the selected cell.
- Same code snapshot (code_sha a3bce3ab63c1), only --lr / --seed differ; 505M at 1.92e-2 may diverge: a diverged run is data (rule 'diverged'), not an exclusion.
- Until these finish, 505M stays PROVISIONAL (F87).
