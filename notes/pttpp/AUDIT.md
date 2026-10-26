# pttpp — the prior's gain at a fixed number of tokens per parameter (2026-10-26)

User approval 2026-10-26 (relayed by the coordinator), for this design. Read-out rule change by the user the same day,
before anything ran (「我感觉应该不用这么严格？」): the verdict rules in "Pre-registered rules" below. Written BEFORE any job
of this batch was submitted (only CPU reference / test jobs ran: scale_work/tpp).

## Question
F87's 7% ladder (53M / 190M / 505M / 1.07B, FINDINGS F87 and addenda 1-3) trains every size on 3e8 tokens, so the tokens per
non-embedding parameter (tpp) fall with size: 21.2 / 2.65 / 0.745 / 0.318. "The prior's gain grows with size" and "the less
trained the model, the more the prior helps" cannot be told apart there (none's area falls 66 -> 53 while the prior stays
at ~89). Two questions:
- **Q1 (size at a fixed tpp)**: at the same tpp, does the gain still change with size?
- **Q2 (tokens at a fixed size)**: at the same size, is the gain larger with fewer tokens?

## Design (make_jobs_pttpp.py is the one source; `--plan` prints the table below)
**Sizes.** The ladder's shape rule (head dim 64, d_ff = 4d, width and depth both grow, OLMo-2 blocks, tied 100,278-row
embedding counted once). The two new rungs are the shapes strictly between the neighbours' width and depth whose N_nonemb
is closest (in log) to the neighbours' geometric midpoint (scale_work/tpp: every candidate enumerated, exact counts on the
meta device; the CPU dry runs print the model line):

| name | shape | n_params | N_nonemb | midpoint target | grad ckpt |
|---|---|---|---|---|---|
| 53M (F87) | d384 x 6, 6 heads | 52,672,128 | 14,165,376 | | off |
| **93M (new)** | **d512 x 10, 8 heads, d_ff 2048** | 93,306,368 | 41,964,032 | 40.06M (log +0.046) | off |
| 190M (F87) | d768 x 12, 12 heads | 190,297,344 | 113,283,840 | | on |
| **294M (new)** | **d832 x 19, 13 heads, d_ff 3328** | 293,931,456 | 210,500,160 | 213.60M (log -0.015) | on |
| 505M (F87) | d1024 x 24, 16 heads | 505,437,184 | 402,752,512 | | on |
| 1.07B (ptlr_1b) | d1280 x 36, 20 heads | 1,072,259,840 | 943,904,000 | | on |

Prior / read-out layers are the trainer default (n_layers // 3, 2 n_layers // 3: 93M 3, 6; 294M 6, 12).

**Lines and tokens.** tpp line 0.745 = 505M@3e8 (18310 x 16384 / 402,752,512 = 0.74485); line 0.318 = 1.07B@3e8 (0.31782).
Steps S = round(18310 x N_nonemb / N_nonemb(line's size)) (half up, exact rational), `--tokens S x 16384` (whole steps; the
trainer's steps = int(tokens // 16384)). Evals: `--eval_every round(250 S / 18310)` -> ~75 evals per run (as F87's 75), so
area (the mean over all evals) and t_theta are on the same footing. The window (0.05, 0.4), the 1% warmup (steps // 100) and
the cosine to 0.1x are fractions of S inside the trainer (unchanged code, lines asserted by flagtest_tpp.py).

| group | hw | S | tokens | tpp | eval_every | evals | warmup | window steps | none grid (pred) | win-ctx grid (pred) | h/run | limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tpp7-93m-t0.745 | L40S | 1908 | 31,260,672 | 0.74494 | 26 | 75 | 19 | 96-763 | 9.6e-3 / 1.92e-2 (1.90e-2) | 2.4e-3 / 4.8e-3 (4.74e-3) | 0.25 | 1:00 |
| tpp7-190m-t0.745 | L40S | 5150 | 84,377,600 | 0.74483 | 70 | 75 | 51 | 258-2059 | 4.8e-3 / 9.6e-3 (9.05e-3) | 1.2e-3 / 2.4e-3 (2.26e-3) | 0.88 | 2:30 |
| tpp7-294m-t0.745 | L40S | 9570 | 156,794,880 | 0.74487 | 131 | 75 | 95 | 479-3827 | 4.8e-3 / 9.6e-3 (7.86e-3) | 1.2e-3 / 2.4e-3 (1.66e-3) | 2.17 | 5:30 |
| tpp7-1b-t0.745 | B200 | 42912 | 703,070,208 | 0.74485 | 586 | 75 | 429 | 2146-17164 | 1.2e-3 / 2.4e-3 (1.57e-3) | 3e-4 / 6e-4 (3.92e-4) | 6.50 | 9:00 |
| tpp7-93m-t0.318 | L40S | 814 | 13,336,576 | 0.31781 | 11 | 75 | 8 | 41-325 | 1.92e-2 / 3.84e-2 (2.91e-2) | 4.8e-3 / 9.6e-3 (7.26e-3) | 0.18 | 1:00 |
| tpp7-190m-t0.318 | L40S | 2197 | 35,995,648 | 0.31775 | 30 | 75 | 21 | 110-878 | 9.6e-3 / 1.92e-2 (1.39e-2) | 2.4e-3 / 4.8e-3 (3.46e-3) | 0.49 | 1:30 |
| tpp7-294m-t0.318 | L40S | 4083 | 66,895,872 | 0.31779 | 56 | 74 | 40 | 205-1633 | 9.6e-3 / 1.92e-2 (1.20e-2) | 2.4e-3 / 4.8e-3 (2.54e-3) | 1.08 | 3:00 |
| tpp7-505m-t0.318 | L40S | 7813 | 128,008,192 | 0.31783 | 107 | 75 | 78 | 391-3125 | 9.6e-3 / 1.92e-2 (1.04e-2) | 1.2e-3 / 2.4e-3 (1.84e-3) | 2.99 | 7:30 |
| tppb-505m-3e8 (bridge) | B200 | 18310 | 299,991,040 | 0.74485 | 250 | 75 | 183 | 916-7323 | 9.6e-3 (F87 pick) | 1.2e-3 (F87 pick) | 1.49 | 3:00 |
| tpp2-190m-t0.745 | A100 | 5150 | 84,377,600 | 0.74483 | 70 | 75 | 51 | 258-2059 | 4.8e-3 / 9.6e-3 (9.05e-3) | 6e-4 / 1.2e-3 (1.13e-3) | 0.62 | 2:00 |
| tpp2-294m-t0.745 | A100 | 9570 | 156,794,880 | 0.74487 | 131 | 75 | 95 | 479-3827 | 4.8e-3 / 9.6e-3 (6.64e-3) | 6e-4 / 1.2e-3 (8.30e-4) | 1.49 | 4:00 |

**lr grids (two per arm; the rule, asserted in the generator).** lr_pred(N, T) = lr*(N, 3e8) x (3e8 / T)^0.5, where lr*(N, 3e8)
is log-linear in ln N_nonemb between the 3e8-token picks of the neighbouring sizes (7% none 53M 8e-3, 190M 4.8e-3, 505M
6.79e-3 = the geometric mean of its 8-seed tie 4.8e-3 / 9.6e-3, 1.07B 2.4e-3; 7% win-ctx 2e-3 / 1.2e-3 / 1.2e-3 / 6e-4; 2% none
8e-3 / 4.8e-3 / 4.8e-3, 2% win-ctx 2e-3 / 6e-4 / 6e-4). The token exponent 0.5 is the geometric middle between "no token
dependence" and pt2long's none (4x tokens -> lr / 4, exponent 1); pt2long's prior moved by exponent 0.5. The grid is the two
ladder lrs (..., 1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2, 3.84e-2, x2 steps) that straddle lr_pred.

**Seeds and stages (per group).** Stage A: both arms x both grid lrs x the selection seeds (seeds 0-3; 1.07B 0-1, as
ptlr_1b, for cost). Selection per arm (below). Stage X: an OPEN bracket (the pick at a grid edge beating its neighbour by > 2
area points, readout_rules.bracket) adds one lr one ladder step beyond that edge, selection seeds, once per arm. Still open
after X -> STOP for that arm: nothing more is submitted, the user decides. Closed (incl. an edge pick within 2 points, "not
open" in F87's sense) -> stage C: the confirm seeds at the pick (4-7; 1.07B 2-7, as ptlr_1b's B + C). Every cell then has 8
seeds at each arm's pick. Arms proceed independently.

**Reused cells (8 seeds each, picks as published).** 7%: 505M@0.745 = F87 ptlr 500m none@9.6e-3 / win-ctx@1.2e-3 (L40S);
1.07B@0.318 = ptlr_1b none@2.4e-3 / win-ctx@6e-4 (B200, lrs picked on seeds 0-1); 190M@2.65 = ptlr 190m none@4.8e-3 /
win-ctx@1.2e-3; 53M@21.2 = ptlr 53m none@8e-3 / win-ctx@2e-3 (Q2 / regression only). 2%: 505M@0.745 = pt2s505 none@4.8e-3 /
win-ctx@6e-4 (A100).

**Hardware and code.**
- 7% new cells on L40S (account 625, l40s_public, gpu48), the same hardware as F87's 53M-505M; 1.07B@0.745 and the bridge on
  B200 (account 520 only, --constraint=b200 --comment="preemption=yes", --requeue, ckpt every 500 steps, 4 singleton slots
  llmadd_tpp-b200-slot0..3), as ptlr_1b. 2% on A100 (account 520, a100_tandon, gpu48), as pt2s505. Every job --nice=20.
- Trainers unchanged: 7% runs/pttpp/code7 = `git archive 64abc19 tools/` (pretrain_range.py a3bce3ab63c1), asserted equal to
  F87's runs/ptlr/code file by file; 2% runs/pttpp/code2 = `git archive 7f1d06b tools/` (93e22bf4c4c0), asserted equal to
  runs/pt2s505/code file by file. Every job re-checks the trainer sha (2%: every module, as pt2s505) and the data sha256.
- Jobs: 7% = F87 / ptlr_1b's job body + a GPU-type guard, the data sha256, an INTENT line (tokens, steps, tpp, eval_every,
  evals, warmup, window), CHECK7 (every trainer argument = the intent, code_sha, steps, n_params, N_nonemb, grad ckpt, layers,
  evals at the intended steps, window events / on_frac, none never on, step-0 lm vs the reference; prints the consumed
  values) -> out/<tag>/check_s<seed>.json, then the selection hook. 2% = pt2s505's job body (dry-input checks with the DRY row
  hash, offline_lm, make_pt2main_tasks.CHECK_PY verbatim with this run's intent) + the hook.

## Deviations from F87 / pt2s505 (all decided before any result)
1. `--tokens` / `--eval_every` non-default (the design). Both flags exist in both trainers; flagtest_tpp.py shows that they
   take effect and that the window scales with S.
2. L40S / A100 runs: no ckpt (`--ckpt_every` 0; F87 2000, pt2s505 2000). Runs are 0.2-3.0 h; a requeued job restarts the
   run from step 0. Resume granularity only (no numerics); saves ~100 GB of transient ckpts at the 16-GPU gpu48 cap.
3. `--save_model` only in stage C (7%; F87 saved every run); 2%: every run saves (offline_lm needs it), stage A / X models
   (seeds 0-3) are deleted after a passing check.
4. 7% jobs add the data sha256, the GPU guard, CHECK7 and the hook; every job `--nice=20` (F87 none, ptlr_1b none).
5. The bridge: F87's 505M args on B200 with `--ckpt_every 500` and no `--save_model`.
6. 1.07B@0.745 selects on seeds 0-1 (as ptlr_1b); the other cells on 0-3.

## Data
pretrain_data: train.bin 400M tokens (uint32), val.bin 2M. Every batch is random windows (rng 2000+seed) of train.bin plus
freshly drawn arithmetic lines (rng 1000+seed): no epochs. The stream of a shorter run is the prefix of a longer run's
(flagtest_tpp.py: lr 0, 40 vs 80 steps, the logged lm of every common step identical, both trainers, both arms), so at a
fixed size the cells of different tpp see nested data. Text drawn / train.bin size (~90% of a row is text): 93M 0.07-0.03x,
190M 0.19x / 0.08x, 294M 0.35x / 0.15x, 505M 0.68x / 0.29x, 1.07B 1.58x / 0.68x: the 1.07B@0.745 run sees each text token ~1.6
times on average (F87's 3e8 runs 0.68x). Repetition below ~4x is known to cost little loss; the arithmetic lines are drawn
fresh, and the held-out pairs are never trained at any budget. It is a size-confounded factor on the 0.745 line and is
named in every write-up of Q1 there. The pretrain_data_3B set is NOT used: its val.bin differs (eval contexts and LM numbers
would not compare).

## Metrics (unique definitions: tools/readout_rules.py, unchanged)
- area = 100 x mean over all evals of the 6-bucket held-out exact match (F87's; ~75 evals in every cell).
- t_theta: tokens until the bucket mean first reaches theta (linear interpolation); censored at the run's own budget.
- R_theta = median t_theta(none) / median t_theta(prior) (readout_rules.speedup: a median is censored when >= half its runs
  are; censored none median -> '>=' a lower bound; censored prior median -> no ratio unless only the prior's is: '<=').
- Delta area = mean over seeds of (prior - none) area at the picks.
- Every table prints Delta area, R_0.5 and R_0.9 side by side for seeds 0-3, 4-7 and 0-7, each with a 90% CI (Delta area:
  paired seed bootstrap; R: readout_rules.boot_ci) and the censored counts (none / prior). Every slope table prints the
  slopes of Delta area, ln R_0.5 and ln R_0.9 for all three seed sets. The primary metric only decides which CI the verdict
  reads.

## Pre-registered rules
**S1. Selection (stage A / X).** 7%: F87 rule 1 as pre-registered (argmax of the mean area over the selection seeds; top two
within 2 points -> the lower mean in-training bpb, '*'); 2%: pt2main rule 1 by pt2main_summary.py itself (every lr within 2 ->
the lower offline full-val bpb). Open bracket = readout_rules.bracket. A diverged run keeps its area (data); unstable runs
(final val_loss > the same seed's none at the group's lower none lr + 0.5) are flagged; an own failure (OWN_FAILURES.txt) is
scored over all eval slots with the missing evals 0; any integrity failure withholds the arm (-> STOP, the user).
**S2. Integrity (every counted run; the read-out withholds every verdict using a failing cell).** check json present and
passing; code_sha; steps / eval spacing / evals at the intended steps; dims, n_params, N_nonemb, grad ckpt; win-ctx window
events and on_frac for its S, none never on; arith_lines / lam / bs / seq / n_eval / window / site / untie / tokenization;
step-0 lm within 1e-3 of the (setting, size, seed) reference (GPU values of F87 / ptlr_1b / ptsize999 + Burst pt2s190 /
pt2s505 runs where they exist, else this batch's CPU references) and of that key's median over every counted run (pairing);
2%: a valid offline_lm entry (pt2main rule 6).
**S3. Floor.** A cell is uninformative (printed, left out of every slope and the regression) if both arms' 8-seed mean area is
< 5 (chance 0 + 5), or the prior's 8-seed t50 median is censored (it does not reach 50% in at least half its runs).
**S4. R censoring.** A cell enters an ln R_theta slope only if its 8-seed R_theta is a point value (neither median censored);
a '>=' cell is listed next to the slope and stays in the Delta area slope. In bootstrap resamples, censored runs sit at the
budget (F87's convention). Fewer than 2 cells -> that slope is "not determinable".
**S5. Slopes.** OLS. Q1: per tpp line, x = ln N_nonemb (7%: 0.745 = 93M, 190M, 294M, 505M, 1.07B; 0.318 = 93M, 190M, 294M,
505M, 1.07B; 2%: 0.745 = 190M, 294M, 505M). Q2: x = ln T (total tokens), pooled within size (size fixed effects; every size with
>= 2 informative cells: 93M, 190M (2.65 / 0.745 / 0.318), 294M, 505M, 1.07B (0.318 / 0.745)); per-size slopes descriptive. 90% CI:
joint paired seed bootstrap (one resample of the seed indices for every cell at once, arms paired; seed s = the same data
stream and held-out split in every cell of a setting), 10000 resamples, numpy default_rng(0). Seed sets: 0-7 (main), 4-7
(guard), 0-3 (descriptive).
**V1-V4. Verdicts (user 2026-10-26).**
- V1. One primary metric per setting: 7% ln R_0.5, 2% Delta area. The other metric is reported; only its 8-seed point estimate
  must have the same sign.
- V2. Main test: the 8-seed slope's 90% CI excludes 0 -> "shown".
- V3. Selection-bias guard: the slope on the confirm seeds 4-7 only must have the same sign (point estimate; no significance).
- V4. All three -> "SHOWN (direction)". Main CI includes 0 -> "NOT SHOWN" with the point estimate and CI (a guard / other
  metric of the other sign is noted). Main CI excludes 0 but the guard or the other metric has the other sign -> "DIRECTION
  UNSTABLE". A guard or other-metric slope that cannot be computed -> "NOT FULLY SHOWN" (not "shown"). Q1 direction: + = larger
  gain at larger sizes; Q2: - = larger gain with fewer tokens. The same rules for Q2.
- Q1 at 7% combines the two lines: both SHOWN the same way -> SHOWN; one line SHOWN, the other not -> "shown on the <line>
  line only"; a line UNSTABLE or the two lines opposite -> DIRECTION UNSTABLE; else NOT SHOWN.
- Every verdict is PROVISIONAL while a designed cell is not final (incomplete / withheld / STOP).
**H. Hardware bridge.** 505M@3e8 on B200 at F87's picks, seeds 0-3, vs F87's L40S runs of the same seeds: PASS iff |B200 -
L40S| mean area <= 2.0 (win-ctx) and <= 8.0 (none, bimodal: one seed changing mode moves a 4-seed mean by ~7), the gap
(win-ctx - none) has the same sign and differs by <= 8.0, and |ln R_0.5(B200) - ln R_0.5(L40S)| <= 0.20 with both point values.
PASS -> the Q1 7% primary slope uses every size; FAIL / withheld -> the primary is the L40S-only slope (1.07B left out) and
the all-size slope is descriptive; both are always printed. Q2 is within size (1.07B: both cells B200) and does not depend on
the bridge. The bridge detects only gross shifts (4 seeds).
**D. Descriptive.** gain ~ a + b ln N_nonemb + c ln T over every informative 7% cell (incl. 53M@21.2, 190M@2.65), both metrics,
same bootstrap; c reads equally as "more arithmetic problems seen" (lines seen = 4 x 32 x S; the prior's t50 is printed in
tokens and in lines seen; it sat at ~20-24M tokens at every size at 3e8). Gradient split per prior cell (LM share by norm and
by squared norm, share of on-evals with prior > LM, first on-eval prior / LM, the clip coefficient derived from the logged
pre-clip norms and cosine: total = sqrt(a^2 + b^2 + 2ab cos), clip = min(1, 1 / (total + 1e-6)); the trainer does not log
it). **No 2% vs 7% comparison.**

## Expected ranges (outside -> check the run before using it)
- prior (win-ctx) area at 0.745: 70-92 at >= 190M; 93M@0.318 may sit at the floor (13M tokens; the prior reached 50% at
  ~20-24M tokens at 3e8). none at the short budgets: low (10-60), bimodal.
- R_0.5: the 0.745 / 0.318 cells below 505M may be '>=' (none's t50 censored): S4 then shrinks the R slopes (a known risk of
  V1's 7% primary; see "Open points").
- final in-training bpb: above F87's 3e8 values at the short budgets.
- Divergence risk: 93M@0.318 none@3.84e-2 (8 warmup steps), the 1.92e-2 cells.

## Cost and disk
- Planned (stage A + C, no X; the generator's --plan): L40S 193 h, B200 142 h (1.07B 20 x 6.5 + bridge 8 x 1.5), A100 51 h.
  One X stage adds 4 runs of its cell (1.07B: 2 x 6.5 = 13 B200-h). Rates measured on F87 / ptlr_1b / pt2s505 run JSONs
  (53M L40S 0.180 s/step, 190M 0.482, 505M 1.176; 1.07B B200 0.528; 505M A100 0.798), others log-log interpolated.
- Kept models (stage C; 2%: C only): 93M 6.0, 190M 12.2, 294M 18.8, 505M 16.2, 1.07B 51.5 (7%) + 2% 15.5 -> ~120 GB. Transient:
  B200 ckpts <= 4 x 12.9 GB (+ one copy while writing); 2% stage-A models until their check (<= 16 x 1.2 GB). myquota
  2026-10-26: /scratch 4.69 / 5.00 TB (93.8%). The hook submits nothing while /scratch >= 97% (DISK GUARD in the log).

## Selection and stages (automation)
Every job of a group ends with `bash runs/pttpp/maybe_select.sh <group>` (not the bridge):
1. HOLD -> nothing. Every frozen file (FROZEN.txt: the read-out snapshot runs/pttpp/readout/tools, pick_tpp.py,
   maybe_select.sh, the generator, the references, the code snapshots' key files) has its recorded sha256[:12]; a mismatch
   logs "FROZEN FILE CHANGED" and nothing is selected or submitted.
2. `pick_tpp.py --check` (no lock): an arm without a C / STOP decision whose stage A (+ X) runs are all finished, or a decided
   run not yet submitted -> continue; else exit.
3. The group's lock (stale-lock takeover when its job is not running); a job finding the lock held leaves a pending mark, and
   the holder repeats the step before releasing it.
4. `pick_tpp.py --decide`: the read-out (7% readout/tools/pttpp_summary.py --select; 2% readout/tools/pt2main_summary.py --gen
   make_jobs_pttpp.py --res out2) -> per arm incomplete (wait) / withheld (STOP) / closed (C at the pick) / open (X one ladder
   step beyond the edge; open again after X: STOP) -> decisions/<group>.json (append-only) -> the disk guard ->
   `make_jobs_pttpp.py --group <g> --test-only` (unsubmitted jobs) and only if all OK `--submit`. Logs select_<group>.log
   (+ .stdout), readout_<group>.txt.
A run that fails on its own never reaches the hook: the coordinator adds it to OWN_FAILURES.txt (or resubmits an
infrastructure loss) and runs `bash maybe_select.sh <group>` by hand.

## Open points (for the user; the rules above stand until changed before a result is read)
- V1 makes 7%'s primary ln R_0.5. At the short budgets none may not reach 50% in half its runs; S4 then leaves such cells out
  of the R slopes, and a line can become "not determinable" while its Delta area slope exists.
- B200 bridge: kept as designed (the user asked whether it is needed; withdrawing it before it runs changes only H).

## Tests before submission (scripts and outputs: llmadd_torch/scale_work/tpp; git copies notes/pttpp/tests)
| test | result |
|---|---|
| shapes.py (meta-device counts of every candidate shape) | 93M = d512 x 10 (log +0.046 from the 53M-190M midpoint), 294M = d832 x 19 (-0.015); F87's four sizes reproduced to the parameter |
| refs_tpp.py on CPU (job 21028572; refs_table.txt) | ALL IDENTITIES HOLD: every dry run prints the designed model line (93.3M off / 293.9M on / 1072.3M on / 505.4M on / 190.3M on), 3 rows with 8 (7%) or 2 (2%) operand sites, the target line only for win-ctx; 7% dry logs (93M, 294M, 1.07B, bridge; seeds 0-1, both arms) = F87's 505M job logs line by line except the model line; 2% DRY row hashes (190M, 294M; seeds 0-7, both arms) = dry2() (190M = make_pt2main_tasks.DRY); step-0 lm identical across arms; CPU - GPU calibration 7% 190M s0-1 -3.7e-5 / -5.8e-5, 505M s0 -1.8e-4, 2% 190M s0-7 within 1.1e-4 -> llmadd_torch/pttpp_refs.json (sha256[:12] d10d720c1242): GPU references where they exist, CPU for 93M / 294M (7%, 2%) |
| flagtest_tpp.py (job 21029607; both trainers) | FLAGTEST PASS x2: --tokens / --eval_every take effect (steps 40 / 80, eval spacing 5 / 10 / 250, evals at the intended steps, window events [[2,1],[16,0]] / [[4,1],[32,0]] = make_jobs_pttpp.window_events); --eval_every changes the eval count (9 vs 2); lr 0, 40 vs 80 steps: the logged lm of every common step identical and distinct across steps (none and win-ctx): the data stream of a shorter run is the prefix of a longer one. A first version at bs 4 failed because the trainer scales --eval_every by 32 / bs (every job runs bs 32; CHECK7 asserts the consumed spacing) |
| intent_ka.py | INTENT-KA PASS: intent7 with F87's 505M facts = the args and facts of F87's real run JSONs (none s0, win-ctx s1) field by field; intent2 with pt2s505's facts = make_jobs_pt2s505.intent(); dry2 with 505M's line = make_jobs_pt2s505.burst_dry_expected, seeds 0-7 |
| jobcheck_tpp.py (200 job texts: the 160 stage-A files + an X and a C rendering per arm and group) | JOBCHECK PASS, 1440 checks: vs the nearest submitted job (F87 ptlr 190M / 505M, ptlr_1b, pt2s505, F87's 505M for the bridge) ARGS differ only in the design's items (--out, --tag, --lr, dims, + --tokens / --eval_every, - --ckpt_every / --save_model; bridge: --ckpt_every 2000 -> 500, no --save_model; stage C adds --save_model); headers (account, partition / constraint, --nice=20, --time = the design); CHECK7 / CHECK_PY verbatim; the 2% intent differs only in size / token / tag fields and equals the design's facts; planted errors caught 7/7 (extra --lam, --eval_every 250, --tokens one step short, hook removed, --nice removed, a 2% intent field, a CHECK_PY line) |
| e2e_tpp.py (job 21029519; rendered jobs run end to end on CPU with tiny overrides, fake nvidia-smi, stub hook) | 5/5 PASS: 7% L40S stage A (rc 0, CHECK7 PASS, consumed values printed, no model), B200 stage C (--save_model: model saved), 2% A100 stage A (CHECK_PY PASS, model removed after the check) and stage C (model kept), a planted intent error (rc 4, the hook still called with the group) |
| pttpp_selftest.py (ladder-tpp tools/) | SELFTEST PASS 56/56: verdict rule units (shown / not shown / 8 seeds shown but confirm opposite / other metric opposite / not computable / zero guard / non-finite); synthetic roots: size effect (both 7% lines SHOWN, designed ln R_0.5 slope 0.25 read +0.262), no effect (NOT SHOWN), confirm seeds opposite (2%: DIRECTION UNSTABLE), other metric opposite (7%: DIRECTION UNSTABLE), floor cell (left out of every slope), censored cell (out of the R slopes, in the Delta area slope, printed '>='), token effect (Q2 SHOWN, fewer tokens), bridge FAIL (L40S-only becomes primary), an integrity failure (cell WITHHELD, PROVISIONAL); every cell row and slope table prints Delta area, R_0.5 and R_0.9 for seeds 0-3 / 4-7 / 0-7; known answers on the REAL reused cells = FINDINGS (505M +27.4 / +36.9 / +32.2, R_0.5 4.76; 1.07B +38.8 / +33.5 / +36.1, R_0.5 5.69 / 4.67 / 5.18; 190M +25.4 / +21.6 / +23.5, 4.56; 53M +23.4 / +22.2 / +22.8, 4.75; pt2s505 +52.0 / +53.9 / +53.0), no integrity failure on them |
| plumb_tpp.py (fake root, stand-in generator: real rendering and sbatch --test-only, fake submission) | PLUMB PASS 17/17: incomplete -> nothing; 7% open upper -> X 1.92e-2, edge tie -> C (8 jobs pass the real --test-only); X worse -> C at the interior pick; X better again -> STOP; 2% through pt2main_summary: tie -> C at the lower offline bpb, open lower -> X 3e-4; HOLD; a changed frozen file; a lock held by a running job -> pending mark; a stale lock taken over and released; nothing left -> --check exit 1 |
| regress_tpp.sh (existing read-outs, ladder-tpp vs the worktree / snapshot they were frozen in) | identical: pt2main (Burst, 143 lines), pt2s505 (--gen, text + json), ptscale_summary (scale-ladder), rb6scale_summary 190M (text + json), rb6scale_ext (text + json), rb6scale_dim (both stop at the not-yet-written rb6_190m_dim/picks.json; text identical after path normalisation), ptlr_1b's frozen snapshot = ladder-1b. ladder-tpp only adds tools/pttpp_summary.py and tools/pttpp_selftest.py |

## Frozen before submission (sha256[:12]; chmod a-w; FROZEN.txt, checked by maybe_select.sh)
make_jobs_pttpp.py a20881aec69f, pick_tpp.py 4d86fa1056b5, maybe_select.sh 37d4d7635ec1, pttpp_refs.json d10d720c1242;
read-out snapshot runs/pttpp/readout = `git archive 712817c tools/` (branch ladder-tpp): pttpp_summary.py 2df9b0ca3f09,
readout_rules.py e91ee8a8a846, pt999_summary.py 01e2fec07499, pt2main_summary.py 68c3b7c26de2; the read-only generators
make_jobs_ptlr1b.py 8dd7b3407459, make_jobs_pt2s505.py a5a7fb60eedc, make_jobs_rb6scale.py 19bc055c2d09; trainers code7
pretrain_range.py a3bce3ab63c1, code2 pretrain_range.py 93e22bf4c4c0 / make_pt2main_tasks.py a2e8a9621383 / offline_lm.py
226f4cac70ea. The read-out on the real root before submission (scale_work/tpp/readout_presubmit.txt): 80 reused runs, no
integrity line, the reused cells at their FINDINGS values, every new cell NOTFINAL, Q1 7% waiting for the bridge.

## Submitted (2026-10-26 ~18:30 EDT; submit_A.log)
`make_jobs_pttpp.py --all --test-only`: 160/160 OK (testonly_A.txt; the B200 copies estimated to start 2026-10-29 on
b200_tandon_prem). myquota at submission: /scratch 4.70 / 5.00 TB (93.9%). `--all --submit`: 160 jobs, 21032456-21033186
(SUBMITTED_<run>): 7% L40S stage A 7 x 16, 1.07B@0.745 stage A 8 (21033028-21033036), the bridge 8 (21033037-21033044, B200
slots after the 1.07B stage A in each slot), 2% A100 stage A 2 x 16. Stages X / C follow automatically (decisions/,
select_<group>.log). Kill switch: touch runs/pttpp/HOLD. The fresh post-submission audit (read-only) was started at once.

## Amendment T1 (2026-10-26 evening; after the fresh post-submission build audit, before any pttpp result was read)
The audit (read-only, fresh general-purpose agent; 13 items) found no wrong job. Seen so far: only the automatic selection of
tpp7-93m-t0.745 (decisions/: none C 1.92e-2 with stage-A areas 3.44 / 3.81, win-ctx C 2.4e-3); no read-out of any cell.
Checked by the build session against the code before acting:
1. **Fixed (bug, could stop an arm for good).** pick_tpp.py: a 7% run counted as finished once its run JSON reached the last
   step, before its job wrote check_s<seed>.json (2-4 s later); a selection by another job in that gap read 'check json
   missing' -> WITHHELD -> STOP (append-only). Now a 7% run is finished only with its check file, as the 2% check.json.
   Plumbing P7: the arm waits, no STOP, and is decided once the check exists.
2. **Fixed (robustness).** maybe_select.sh: a pending mark set after the holder's last check was lost; squeue errors let a
   running holder's lock be taken over. Now the holder releases, re-checks the mark and repeats (<= 5 rounds, logged
   otherwise); only squeue's "Invalid job id" counts as gone (P8). pick_tpp.py retries the read-out 3 x 30 s; the read-out
   skips a run JSON that does not parse (a running job rewrites it at every eval, not atomically) with a NOTE (a finished
   run's JSON is never rewritten); TMPDIR / XDG_RUNTIME_DIR are not passed to submitted jobs.
3. **Fixed (descriptive output).** The regression gain ~ ln N + ln T printed pinv's minimum-norm c with a CI on a rank-2
   design (the four reused cells all have T = 3e8). Now: rank < 3 or |corr(ln N, ln T)| >= 0.98 -> "NOT IDENTIFIED".
4. **Added, DESCRIPTIVE ONLY (the verdict rules V1-V4 are unchanged; whether any of this gates a verdict is the user's
   decision):** a "## sensitivity" section per Q1 line and Q2: (a) the R slopes with the '>=' cells at their bound (a bound at
   the small end biases the slope upward); (b) every slope without b11 (its held pairs occur in the natural text, so its
   dose grows with T; the AUDIT's "never trained" holds for the arithmetic lines only); (c) lr selection: the seeds-0-3
   slope with one cell's arm at each alternative lr within 2 area points of the pick, or at the lowest-median-t50 lr (new
   cells: the stage A / X lrs; reused cells: F87 ptlr / ptsize999 and pt2s505 lrs with seeds 0-3; 1.07B's alternatives
   have seeds 0-1 only and are not varied), with the slope range and the number of sign changes. Per cell a facts line:
   steps, warmup steps, eval spacing in tokens, tpp by total parameters, the token at which the window opens, the prior's
   t50 / opening and t50 - opening, Delta area and R_0.5 without b11.
5. Code snapshots chmod a-w (code7, code2). The 7% jobs check only the trainer's sha at start (as F87 / ptlr_1b); the 2%
   jobs check every module.
Tests after the changes: selftest 65/65 (new: the non-identified regression, the sensitivity lines, an lr swap that changes
the sign is counted and flagged, 2% / 7% cells with the same label kept apart, 1.07B alternatives not swapped, a partial
run JSON skipped with the verdict unchanged), plumbing 21/21 (P7, P8 new), the read-out on the real root; regression of the
other read-outs unchanged (only tools/pttpp_*.py differ). Frozen anew: see "Frozen (T1)" below; the jobs already submitted
call maybe_select.sh by path, so they use the new files.
**Points for the user (not acted on; the build session does not change the verdict rules or the design):**
- (#1) The slope CIs cover seed noise only; with 2 lrs per arm and the prior's area near its ceiling, the prior's lr is
  chosen by the 65k-token in-training bpb, and F87 showed that a tied choice can move R_0.5 by up to ~0.27 in ln. Section
  4(c) prints that range. Option: require that no swap in 4(c) changes the sign for "SHOWN".
- (#2) R_0.5 is partly tied to the window: the prior's 3e8 t50 (20-24M) sits 1.4-2.8x after the window opens (15M), and the
  opening moves with T (0.05 S). A ln R slope on ln T (Q2) and on ln N along a tpp line (Q1, T ∝ N) partly reflects this
  design choice; Delta area (normalised time) is not affected. Option: read Q2 on Delta area, or treat ln R and Delta area as
  two separate questions.
- (#3) A floor tie closes a bracket: at tpp7-93m-t0.745 none's two lrs are at the floor (3.44 / 3.81), the in-training bpb
  picked the upper edge 1.92e-2 and stage C started; 3.84e-2 (the predicted optimum sits at the grid's top) was not tried.
  The same may happen at 93M / 190M@0.318. Option: none@3.84e-2 seeds 0-3 for that cell (~1 L40S-h), and/or a rule "every lr
  of an arm < 7 area points -> STOP".
- (#5) The bridge tolerance (|Delta ln R_0.5| <= 0.20) allows the 1.07B end point alone to move a 5-cell slope by ~0.05 (3 cells:
  ~0.14). The L40S-only slope is always printed; option: write a size effect only when both slopes agree in sign.
- (#8 / #9) At a fixed tpp the steps also grow with size (814-42,912; warmup 8-429 steps), and the tpp by total parameters
  rises from 0.335 (93M) to 0.656 (1.07B) on the 0.745 line: printed per cell, not corrected.
