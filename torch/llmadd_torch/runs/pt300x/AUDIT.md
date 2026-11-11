# pt300x — two more sizes on the fixed-token 7% ladder (2026-11-02)

User decision 2026-11-02 (relayed by the coordinator): add two sizes to F87's fixed-token ladder at the same 3e8 tokens;
hardware: 「可以多用点其他gpu吗，现在越快越好」 (one GPU type per size, never mixed inside a size). Written BEFORE any job of
this batch was submitted: only CPU dry runs and tests ran (runs/pt300x/tests). Status at writing: built and tested, nothing
submitted.

## Question
F87's 7% ladder (3e8 tokens, every arm at its own best lr, 8 seeds; F87 + addenda 1-3) has four sizes: 53M, 190M, 505M on
L40S and 1.07B on B200. Its post hoc area-gain slope on ln N_nonemb is +3.25 per e-fold (FINDINGS F97 addendum FIXSLOPE):
seed-only 90% CI [+2.1, +4.5], but [+0.1, +6.4] when the spread of the four sizes around the line is counted (t, df 2).
With four points the size-level interval cannot be tight. Two more sizes at the same 3e8 tokens, the pttpp shapes:
93M (d512 x 10, N_nonemb 41.96M) and 294M (d832 x 19, N_nonemb 210.5M).
- **Q**: on the fixed 3e8-token 7% ladder with six sizes (53M, 93M, 190M, 505M from L40S; 294M and 1.07B from B200; the
  four old cells reused), does the 8-seed paired area gain (prior - none, readout_rules.area) increase with ln N_nonemb?
- **Already observed when this was registered**: 4 of the 6 points (53M, 190M, 505M, 1.07B: F87, its addenda, F97 FIXSLOPE
  with its slope, both seed halves and the 4-point residual interval). The test is therefore only partly confirmatory: the
  two new sizes and criterion (c) are the new evidence. Nothing of 93M or 294M at 3e8 tokens exists.

## Design (make_jobs_pt300x.py is the one source; `--plan` prints the table)
| size | hw | shape | n_params | N_nonemb | tpp at 3e8 | ln N | source | picks (none / win-ctx) |
|---|---|---|---|---|---|---|---|---|
| 53M | L40S | d384 x 6, 6 heads | 52,672,128 | 14,165,376 | 21.18 | 16.466 | F87 ptlr (reused) | 8e-3 / 2e-3 |
| **93M** | **L40S** | **d512 x 10, 8 heads, d_ff 2048** | 93,306,368 | 41,964,032 | 7.149 | 17.552 | **new** | stage A below |
| 190M | L40S | d768 x 12, 12 heads | 190,297,344 | 113,283,840 | 2.648 | 18.545 | F87 ptlr (reused) | 4.8e-3 / 1.2e-3 |
| **294M** | **B200** | **d832 x 19, 13 heads, d_ff 3328** | 293,931,456 | 210,500,160 | 1.425 | 19.165 | **new** | stage A below |
| 505M | L40S | d1024 x 24, 16 heads | 505,437,184 | 402,752,512 | 0.745 | 19.814 | F87 ptlr (reused) | 9.6e-3 / 1.2e-3 |
| 1.07B | B200 | d1280 x 36, 20 heads | 1,072,259,840 | 943,904,000 | 0.318 | 20.666 | ptlr_1b (reused) | 2.4e-3 / 6e-4 |

Shapes, n_params and N_nonemb are make_jobs_pttpp.SIZES (verified there on the meta device and in the dry runs; the CPU
dry runs here print "model: 93.3M parameters; gradient checkpointing off" / "293.9M ...; on"). The reused cells are
make_jobs_pttpp.REUSED (one run per seed 0-7, 64 run JSONs, the F97 FIXSLOPE cells).

**Protocol = F87 (ptlr) exactly, except shape, lr grid and (294M) hardware.** 3e8 tokens = 18,310 steps x 32 x 512
(299,991,040 tokens; the trainer's default --tokens 3e8, int(3e8 // 16384) = 18310: F87's jobs pass no --tokens /
--eval_every, nor do these), eval every 250 steps -> 75 evals, warmup 1% = 183 steps, cosine to 0.1x, AdamW (0.9, 0.95)
wd 0.1, clip 1.0, bf16 autocast, lambda 5, window (0.05, 0.4) = prior on at steps 916..7323 (prior_events [[916, 1],
[7324, 0]], on_frac 6408 / 18310), ctx site (the operands), --arith_lines 4 (7%), prior / read-out layers n_layers // 3 and
2 n_layers // 3 (93M 3, 6; 294M 6, 12), gradient checkpointing auto (93M off, 294M on). Arms 'none' (--target none) and
'win-ctx' (--target helix3 --schedule window --window 0.05,0.4 --site ctx) = F87's flags (make_jobs_pttpp.ARM7). Data
pretrain_data (train.bin 400M tokens, val.bin 2M; sha256 checked by every job), random windows (rng 2000 + seed) +
fresh arithmetic lines (rng 1000 + seed): seed s = the same data stream and held-out split at every size (the init differs by shape).

**Code.** Trainer: runs/pt300x/code7 = `git archive 64abc19 tools/` (via make_jobs_pttpp._extract), asserted equal file by
file to runs/pttpp/code7 and to F87's runs/ptlr/code (`diff -r`: identical), COMMIT verbatim, chmod a-w;
pretrain_range.py a3bce3ab63c1 (code_sha of every result). Job body = make_jobs_pttpp.job7's 7% body: HOLD kill switch,
trainer sha check (exit 3), the GPU-name guard (exit 4), data sha256 (exit 3), the INTENT line, `--dry_inputs` (decoded
training / eval rows in the log), training, CHECK7 verbatim on the run JSON -> out/<tag>/check_s<seed>.json (every trainer
argument = the intent, code_sha, steps, n_params, N_nonemb, grad ckpt, layers, 75 evals at the intended steps, window
events / on_frac, none never on, step-0 lm within 1e-3 of the (size, seed) reference in pttpp_refs.json); the job exits with
the check's code. No selection hook (the selection is run by hand). The generator imports make_jobs_pttpp read-only
(sha256[:12] b4bfa6eb248d = its FROZEN.txt line since Amendment T8; the task text quoted ad538e40275c, the T7 version),
asserts that the import runs no main(), calls its args7() / intent7() for every run and asserts this batch's ARGS / intent
equal to them up to the --out directory.

## Deviations from F87 (decided before any result; tests/args_diff.out lists every differing argument)
1. Shape (the pttpp shapes) and the lr grids (below).
2. 294M on B200 (below, "Hardware").
3. `--ckpt_every`: L40S 0 (F87 2000; pttpp's 7% L40S rule), B200 500 (= the pttpp bridge, the pttpp 1.07B and ptlr_1b
   jobs). Resume granularity only: ptlr_1b's B200 smoke showed the first resumed eval identical to the uninterrupted one
   (--stop_after resume: lm, val_loss, geo, cka, ema, prior_on_frac equal to all digits; a real preemption resumed
   correctly). On L40S a requeued job restarts the run from step 0 (run_guard lets the same job id continue).
4. `--save_model` only in stage C (seeds 4-7 at the picks; pttpp's 7% rule; F87 saved every run): F87 rule 1 at 7% uses the
   in-training bpb, so no stage-A model is needed.
5. `--nice=20` on every job (pttpp); the data sha256, GPU guard, INTENT line and CHECK7 (pttpp's 7% body).
Argument diffs (rendered ARGS): vs pttpp 93M / 294M (same shape, 0.745 line): --tokens / --eval_every absent (the 3e8
default), --lr, --out, --tag, and on B200 --ckpt_every 500; vs the pttpp B200 bridge (505M at 3e8 on B200): only the shape
(--d_model / --n_layers / --n_heads / --d_ff), --lr, --out, --tag; vs F87 ptlr 190M: shape, --lr, --out, --tag,
--save_model (stage C only), --ckpt_every (0 / 500 vs 2000); vs ptlr_1b 1.07B stage C: shape, --lr, --out, --tag. Nothing
else; CHECK7 byte-identical to pttpp's.

## Hardware (one GPU type per size, never mixed inside a size)
- **93M on L40S**: account <slurm-account>, l40s_public, QOS gpu48, --nice=20, 8 CPUs / 64 GB, time 04:00 (pttpp's
  rule max(1, ceil(5 x est) / 2) h on the 1.44 h estimate), --requeue: as 53M / 190M / 505M.
- **294M on B200**: as the pttpp B200 jobs (1.07B@0.745 and the bridge), header line for line: account
  <slurm-account>, no --partition, `--constraint=b200 --comment="preemption=yes"` (the preemptible B200 pools),
  --gres=gpu:1, --nice=20, 16 CPUs / 128 GB, time 03:00 (= the bridge's; one segment, a requeue restarts the clock),
  --requeue, --open-mode=append, per-run log file; 8 singleton slots `llmadd_p3x-b200-slot0..7` with
  --dependency=singleton (this batch's own names, separate from pttpp's llmadd_tpp-b200-slot*); ckpt every 500 steps and
  resume after preemption (the trainer reloads model, optimiser, rng states and curve); GPU-name guard 'B200' (exit 4 on
  anything else). Stage A: slot = run index mod 8 (none's 16 runs fill the first two waves); X / C: slot = 4 x arm index +
  seed mod 4.
- **Why B200 for 294M is acceptable.** The pttpp hardware bridge (505M at 3e8 tokens, B200 vs L40S, F87's picks, seeds 0-3;
  runs/pttpp/readout_full_20261030T2118.txt, '## hardware bridge') passed every pre-registered tolerance: none -4.0
  (tolerance 8; per seed -13.0, +1.0, -2.7, -1.5), win-ctx +0.2 (tolerance 2), gap +31.7 (B200) vs +27.4 (L40S) (same
  sign, difference 4.3 <= 8), ln R_0.5 5.14x vs 4.76x (|difference| 0.08 <= 0.20). 294M sits near the middle of the six
  ln N values (19.165 vs their mean 18.701; Sxx 11.651), so a hardware offset moves the slope little: a +1 point offset of
  the gain at 294M moves the six-point slope by +0.040, at 1.07B by +0.169; the bridge's +4.3 gap difference applied to
  294M alone gives +0.17, to 1.07B alone +0.72, to both +0.90.
- **Pre-registered hardware sensitivities (descriptive, not part of the decision rule)**: (i) the slope with a B200
  indicator (294M and 1.07B) as a covariate, gain ~ a + b ln N_nonemb + c [B200], b with the same seed bootstrap and its
  residual t interval (df 3), c printed; (ii) the slope over the L40S sizes only (53M, 93M, 190M, 505M; Sxx 6.104), same
  bootstrap, residual t (df 2).
- Capacity note: the gpu48 per-user cap (16 GPUs, across accounts and partitions) also covers B200 jobs that get gpu48
  (pttpp T8); 8 B200 + ~12 L40S can exceed it, in which case the scheduler holds some jobs (no failure).

## lr grids (four rungs of the factor-2 ladder per arm; asserted in the generator)
Interpolation at 3e8 tokens, log-linear in ln N_nonemb between the neighbouring sizes' picks (pttpp's rule without its
token factor, since the tokens are the same):
| cell | neighbours' 3e8 picks | interpolated | pttpp picks of the same shape (fewer tokens) | grid | X if open |
|---|---|---|---|---|---|
| 93M none | 53M 8e-3, 190M 4.8e-3 | 6.13e-3 | 1.92e-2 @ 31.3M tokens, 7.68e-2 @ 13.3M (none's area at the floor in both cells, pttpp T2 / T3: uninformative) | 2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2 | 1.2e-3 / 3.84e-2 |
| 93M win-ctx | 53M 2e-3, 190M 1.2e-3 | 1.53e-3 | 2.4e-3 @ 31.3M, 9.6e-3 @ 13.3M | 6e-4, 1.2e-3, 2.4e-3, 4.8e-3 | 3e-4 / 9.6e-3 |
| 294M none | 190M 4.8e-3, 505M 9.6e-3 | 6.73e-3 (5.69e-3 with 505M's 8-seed tie 4.8e-3 / 9.6e-3 as its geometric mean) | 1.92e-2 @ 156.8M (after two extensions), 1.92e-2 @ 66.9M | 2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2 | 1.2e-3 / 3.84e-2 |
| 294M win-ctx | 190M 1.2e-3, 505M 1.2e-3 | 1.2e-3 | 2.4e-3 @ 156.8M, 1.2e-3 @ 66.9M | 3e-4, 6e-4, 1.2e-3, 2.4e-3 | 1.5e-4 / 4.8e-3 |
Why these grids: each straddles the interpolated value with at least one rung on either side; none's grids reach two
rungs above it because none's 3e8 pick rose again from 190M (4.8e-3) to 505M (9.6e-3, an 8-seed tie with 4.8e-3) and the
pttpp picks of both shapes at fewer tokens sit at 1.92e-2 (fewer tokens -> a higher or equal lr, so 1.92e-2 bounds them
from above at 3e8); win-ctx's grids include the pttpp win-ctx pick of the same shape at the 0.745 line (2.4e-3 for both)
and reach the 1.07B pick (6e-4; 294M: one rung below it) at their lower end.

## lr selection (pre-registered; runs/pt300x/select_pt300x.py, run by hand by the coordinator; no hooks)
Per size and arm independently, F87 rule 1 exactly (runs/ptlr/AUDIT.md rules 1-2; readout_rules.best_lr with near_rule
'top2', in-training bpb, readout_rules.bracket):
1. Stage A: the 4 grid lrs x seeds 0-3 (2 sizes x 2 arms x 4 lrs x 4 seeds = 64 runs).
2. Best lr = argmax of the mean area over seeds 0-3; if the top two are within 2 area points, the one with the lower mean
   final in-training bpb (val_bits_per_byte at the last eval; a non-finite bpb is the worst; an exact tie -> the lower lr).
3. Open bracket = the pick is at an edge of the arm's grid and beats its neighbour by more than 2 points -> stage X: one
   ladder rung beyond that edge, seeds 0-3, once per arm; then rule 2 over the five lrs. Still open after the X -> STOP:
   nothing more for that arm, the user decides (an amendment). An edge pick within 2 points of its neighbour is closed.
4. Closed -> stage C: seeds 4-7 at the pick (--save_model). The cell's 8 seeds = seeds 0-3 of the pick's stage A / X runs +
   seeds 4-7 of C. Stage C = 2 arms x 4 seeds x 2 sizes = 16 runs; X adds 4 runs per extended arm (at most 16).
5. A diverged run keeps its area (data) and counts against its lr; final val_loss > the same seed's none at the arm's lowest
   none lr + 0.5 nats is flagged 'unstable' (kept). An own failure (the run stopped on an error of its own; the coordinator
   lists it in runs/pt300x/OWN_FAILURES.txt) is scored over all 75 eval slots with the missing evals 0, bpb worst; an
   infrastructure loss (preemption, node failure, HOLD) is resubmitted, not listed.
6. A run whose CHECK7 failed, whose recomputed area differs from the check's by more than 0.006 (double path), or that
   fails the integrity re-check (every training-defining argument, code_sha, steps, n_params, grad ckpt, layers, evals,
   window events / on_frac, step-0 lm) withholds its arm: nothing is decided; the coordinator investigates.
`select_pt300x.py` prints every arm's grid means, tie-break and decision; `--write` appends the new X / C / STOP entries to
runs/pt300x/decisions.json (append-only; the generator validates the whole log: one X one rung beyond a stage-A edge, then
one C in the grid or at the X lr, or a STOP; nothing after a C or a STOP) and writes picks.json, then prints the commands
`make_jobs_pt300x.py --stage X|C --size S --arm A --test-only` and `... --submit` that the coordinator runs by hand.

## Pre-registered analysis (runs/pt300x/readout_pt300x.py; written and tested before any pt300x run)
- **Cells**: the six sizes at each arm's pick, seeds 0-7, gain_s = area(prior, s) - area(none, s) (paired by seed; area =
  readout_rules.area = 100 x mean over all 75 evals of the 6-bucket held-out exact match). New cells: the C decisions of
  decisions.json, recomputed by the selection rule and asserted equal; seeds 0-3 at the pick from stage A / X, 4-7 from C.
  Reused cells: make_jobs_pttpp.REUSED, asserted equal to the FINDINGS F97 FIXSLOPE values (gain, none, prior to 0.05).
- **Primary statistic**: the OLS slope b of the six 8-seed mean gains on ln N_nonemb (equal weights).
- **Decision rule.** "SHOWN (increase)" (or "(decrease)" for b < 0) iff ALL of:
  - (a) the 90% seed-bootstrap CI of b excludes 0 on the side of b. Bootstrap: seeds resampled with replacement within each
    size, the two arms paired (the gain per seed), 10000 resamples; numpy default_rng(0), one (10000, 8) index draw per size
    in ascending N order; percentile interval (5th, 95th).
  - (b) the slopes on seeds 0-3 and on seeds 4-7 (point estimates, the same six sizes) both have the sign of b.
  - (c) the size-level interval from the OLS residuals of the six points, b +- t(0.95, df 4) x se(b), se(b) = sqrt(SSR / 4 /
    Sxx), excludes 0. This is the criterion the four-point ladder could not meet.
  Otherwise "NOT SHOWN" with the numbers and the criteria that failed. Any cell not final (an arm without its C, a STOP, a
  run missing or unfinished) or with an integrity problem -> "NOT DETERMINABLE" (no slope is read). A new cell whose 8-seed
  none or prior mean lies outside its expected range by more than 15 points -> the verdict line carries "INVESTIGATE BEFORE
  READING" and the coordinator records the investigation here before anyone reads the verdict.
- **Descriptive (printed, no verdict)**: the same slope with 1.07B left out (five sizes; note: with 294M now on B200 these
  five are not L40S-only, the L40S-only slope is (ii) below); hardware sensitivities (i) B200 indicator and (ii) L40S sizes
  only (above); the six leave-one-size-out slopes; none and prior arms separately (mean area on ln N, same bootstrap); the
  ln R_0.5 slope (secondary, sign only: F87 rule 8's metric, R_0.5 = median t50(none) / median t50(prior) with censored runs
  at the budget, readout_rules.speedup; only cells with a point value enter); lr-swap sensitivity within the tie band (one
  arm of one cell at another grid lr within 2 area points of its pick on seeds 0-3 -> the seeds-0-3 slope; the number of sign
  changes; new cells: the stage A / X lrs; reused L40S cells: F87's ptlr grid dirs with seeds 0-3; 1.07B's alternatives
  have seeds 0-1 only and are not varied); the four reused points alone (= F97 FIXSLOPE).
- **What the verdict may be called.** A fixed-token statement: at 3e8 tokens tpp falls from 21.2 to 0.318 along the ladder,
  so size and training sufficiency stay confounded (FIXSLOPE confounds 1-7 apply: 1.07B on B200 with lrs picked on seeds
  0-1, different lr grids, fixed 1% warmup, bimodal none, fixed lambda, the prior near the area ceiling). "SHOWN (increase)"
  = the area gain at 3e8 tokens increases with ln N_nonemb across six sizes, with the size-level interval; it is not a
  speed-up claim (area is not time; F87 rule 8's ln R verdict stays separate and is reported by sign only) and not "the
  prior's effect grows with size" (the gain rose because none's area fell at 3e8 in F97).
- **Rough expectation (not a rule)**: with the four-point residual SD (3.41 area points, from FIXSLOPE's df-2 interval), six
  points give se(b) ~ 1.00 and a df-4 half-width ~ 2.13, so (c) excludes 0 if b stays near +3.25 and the new points scatter
  no more than the old ones; a 93M gain near 53M's / 190M's (the flat 53M -> 190M segment) lowers b.

## Expected ranges (outside -> check the run before using it; the read-out flags +-15)
- 93M: none ~60-70 (between 53M 66.1 and 190M 66.5 at 3e8; bimodal), prior ~89 -> gain ~19-29.
- 294M: none ~55-67 (between 190M 66.5 and 505M 57.5), prior ~89-90 -> gain ~22-35.
- Every run: code_sha a3bce3ab63c1, 18310 steps, 75 evals, prior_events [[916, 1], [7324, 0]] / on_frac 0.349973 (win-ctx),
  none never on; step-0 lm = pttpp_refs.json within 1e-3 (93M seeds 0-7: 11.6255 / 11.6025 / 11.6372 / 11.6178 / 11.5911 /
  11.6218 / 11.6184 / 11.6036; 294M: 11.6846 / 11.6604 / 11.6818 / 11.6756 / 11.6600 / 11.6808 / 11.6914 / 11.6864; the
  pttpp L40S runs of both shapes matched them within 1.2e-4; the bridge's B200 runs matched L40S references within 5e-5).
- Divergence risk: none@1.92e-2 (and an X at 3.84e-2).

## Metrics (unique definitions: tools/readout_rules.py of the frozen pttpp read-out snapshot, sha256[:12] e91ee8a8a846)
area (above); t_theta (tokens until the bucket mean first reaches theta, linear interpolation, censored at the 300M budget);
R_0.5 (readout_rules.speedup); in-training bpb = val_bits_per_byte at the last eval (8 x 16 fixed val windows). Train vs
eval conditions: identical to F87 (runs/ptlr/AUDIT.md "Train vs eval conditions"); only the shape and lr differ.

## Cost and wall time (make_jobs_pt300x.py --plan)
Per run, from run JSONs (curve[-1]['sec'], medians) + 180 s start-up / dry run / check:
- 93M on L40S: pttpp 93M@0.745 (1908 steps, 736 s) and 93M@0.318 (814 steps, 451 s), 75 evals each -> 0.2605 s/step +
  3.19 s/eval -> 5009 s at 18310 steps -> **1.44 L40S-h per run**.
- 294M on B200: log-log in n_params through the B200 505M bridge (18310 steps, 5768 s = 1.60 h) and ptlr_1b 1.07B (10395 s =
  2.89 h): exponent 0.783 -> 3773 s; the other route (pttpp 294M L40S 0.755 s/step + 10.9 s/eval -> 14644 s at 18310 steps,
  x the bridge's B200 / L40S ratio 0.251) gives 3670 s -> **1.10 B200-h per run** (the larger value).
| | stage A | C | A + C | worst case + X on both arms |
|---|---|---|---|---|
| 93M L40S | 32 x 1.44 = 46.1 | 8 x 1.44 = 11.5 | 57.7 | +11.5 -> 69.2 |
| 294M B200 | 32 x 1.10 = 35.1 | 8 x 1.10 = 8.8 | 43.9 | +8.8 -> 52.7 |
Wall time of running (queue waits and the selection by hand come on top): 93M at ~12 concurrent L40S: stage A 3 waves = 4.3
h, X 1.4 h, C 1.4 h -> ~7.2 h; 294M on 8 B200 slots: stage A 4 waves = 4.4 h, X 1.1 h, C 1.1 h -> ~6.6 h (+ preemption losses,
at most 500 steps ~ 100 s per preemption). B200 preemptible queues have taken 1-2 days to start (ptlr_1b / pttpp T8); the
--test-only estimates at writing are below.
Disk: stage C models 93M 8 x 0.37 GB, 294M 8 x 1.18 GB (~12.4 GB); transient 294M ckpts <= 8 x ~3.5 GB (+ one copy while
writing) ~ 32 GB; run JSONs small. /scratch at writing 4.71 / 5.00 TB (94.19%); submit_pt300x.sh refuses at >= 97%.

## Submission (stage A only; the coordinator)
`bash runs/pt300x/submit_pt300x.sh` checks (no HOLD; `make_jobs_pt300x.py --check --stage A`: every stage-A job file = its
rendering, the snapshot, the frozen shas; 64 runs; none submitted yet; myquota /scratch < 97%) and prints the 64 sbatch
lines (B200 294M in slot order, then L40S 93M); `--go` submits them together and writes SUBMITTED_<run> flags (the sbatch
output). X / C: by hand after `select_pt300x.py --write`. Kill switch: `touch runs/pt300x/HOLD` (a pending or requeued
job exits at start; a job that ends on HOLD exits 0 = COMPLETED without a run JSON: the selection shows it as missing).

## Tests before submission (runs/pt300x/tests; outputs *.out next to the scripts)
| test | result |
|---|---|
| render + `bash -n` (tests/render.out) | 64 stage-A job files written by `make_jobs_pt300x.py`; `bash -n` clean on all 64; `--check --stage A` OK (file = rendering); 32 L40S 93M (time 04:00), 32 B200 294M (slots 0-7, 4 jobs each, time 03:00) |
| args_diff.py | ARGS DIFF PASS: the argument diffs listed in "Deviations" (12 pairings: pttpp 93M / 294M stage A and C, the pttpp B200 bridge, F87 190M, ptlr_1b 1.07B C); nothing unexpected; CHECK7 byte-identical to pttpp's job and to make_jobs_pttpp.CHECK7; the other job lines differ only in names, paths, comments, time limit and the absent hook |
| dry_digest.py (CPU, rendered stage-A ARGS, --out redirected) | DRY DIGEST PASS 4/4: the --dry_inputs section (eval rows, model line, target line, 3 decoded training rows of the first batch) of 93M none / win-ctx and 294M none / win-ctx seed 0 is line for line the pttpp job log's of the same shape, arm and seed (sha256[:16] a3192b8a71181936, ef66a699e4826073, ce8af7afdb623bab, 5d4742770e61f1e5); with pttpp's flagtest (a shorter run's stream is the prefix of a longer one's) the data stream is identical up to the step count |
| check7_ka.py | CHECK7 KA PASS 21/21: CHECK7 with this batch's 3e8 intent form (only size / lr / seed / out / tag / save / ckpt set to the runs') passes on real F87 190M none s0, F87 505M win-ctx s1 and ptlr_1b 1.07B win-ctx s5 run JSONs; planted lr x2, eval_every 26, tokens 31.3M, steps 18309, shifted window events, step-0 reference +0.002 -> rc 4 each |
| select_pt300x.py --selftest | SELECT SELFTEST PASS 22/22 (synthetic roots, real CHECK7 files, the real generator): incomplete -> wait; interior pick -> C; top-two tie -> lower in-training bpb (F87's top-two rule, not every-lr-within-2); open upper -> X 3.84e-2 and open lower (294M win-ctx) -> X 1.5e-4 (rendered on B200 slots 4-7); X worse -> C at the interior pick (C jobs seeds 4-7 with --save_model); X better again -> STOP, nothing after it; CHECK7 failure, double-path mismatch, step-0 lm off -> WITHHELD; an unfinished run waits; an own failure scored 0; 5 malformed decision logs refused |
| readout_pt300x.py --selftest | READOUT SELFTEST PASS 18/18: positive slope (designed 3 read +3.000 on 8 seeds and both halves; SHOWN increase; arms -3 / 0; B200-indicator b 3, c 0; L40S-only 3; ln R_0.5 slope positive; the one tied lr swap listed, no sign change); zero slope -> NOT SHOWN (a); negative -> SHOWN decrease; halves disagree (+6 / -1.5) -> NOT SHOWN (b); residual interval includes 0 (b 1, size scatter +-8, seed noise +-0.1) -> NOT SHOWN (c) with (a) and (b) passing; B200 offset +6 -> (i) b 2, c 6, (ii) 2; a reused run with another code_sha, a failed CHECK7, a missing C decision, a missing C run, a C decision that is not the recomputed pick -> NOT DETERMINABLE; prior ~73 -> INVESTIGATE BEFORE READING; designed reused cells with the FINDINGS check on -> NOT DETERMINABLE |
| readout_pt300x.py --ka (real reused runs) | KA PASS 12/12: the 64 reused runs pass the integrity re-check; gains / none / prior per size = F97 (22.79 / 23.53 / 32.19 / 36.13); slope +3.252; within-size bootstrap [+2.106, +4.480]; residual t df 2 [+0.102, +6.402]; halves +3.077 / +3.426; leave-one-out +6.011 / +3.072 / +3.185 / +2.566; arms none -3.125 (FINDINGS prints -3.13), prior +0.127 |
| regress_frozen.py | REGRESSION PASS: runs/pttpp/FROZEN.txt 26/26 before and after; the frozen make_jobs_pttpp re-renders all 302 submitted pttpp job files byte for byte in the same process that imported and used make_jobs_pt300x; make_jobs_ptlr1b re-renders its 12 job files; make_jobs_ptlr.py (85c3bf083bc9, writes its jobs at import, not run) and runs/ptlr job files untouched |
| sbatch --test-only (2 jobs; tests/testonly.out) | both accepted (nothing submitted): p3x-93m-lr4.8e-3-none-s0 on l40s_public, estimated start 2026-11-03T08:49 (it would preempt one job, 21595074); p3x-294m-lr4.8e-3-none-s0 on b200_tandon_prem, estimated start 2026-11-03T01:42 |
| submit_pt300x.sh (no --go; tests/submit_dry.out) | checks pass, 64 sbatch lines printed (32 B200 then 32 L40S), nothing submitted |

## Audit status (experiment-audit)
i double path: area recomputed from the run JSON vs CHECK7's printed area (<= 0.006) for every new run; the four reused
cells vs FINDINGS. ii asserts: generator (frozen shas, import without main, ARGS / intent = pttpp's functions, grids = ladder
rungs, shapes, steps / evals / window), every job (trainer sha, data sha256, GPU name, CHECK7), selection and read-out
(integrity re-check of every counted run, decision log validation). iii known answers: dry digests = pttpp logs, CHECK7 on
real F87 / ptlr_1b runs, F97 FIXSLOPE reproduced by the read-out, designed selftests. iv: - (no result yet). v: **pending** —
the fresh general-purpose adversarial audit could not be started by this build agent (no subagent tool); the coordinator
starts it (read-only, blind) before or right after submission. vi: trap list checked (positions / template / range = F87;
the prior's activity asserted per run; no silent filtering: own failures scored, missing runs block). vii: every job prints
the decoded training and eval rows (--dry_inputs) before training.

## Files at writing (sha256[:12]; 2026-11-02 ~18:40 EST, before any submission)
make_jobs_pt300x.py e8fd563c335a; runs/pt300x: select_pt300x.py c4f07df1e18c, readout_pt300x.py c595c9dbb70c,
submit_pt300x.sh c3fbf808e3fd; tests/args_diff.py e4352b3e042e, check7_ka.py 47bd2531f6c2, dry_digest.py eb4a292b42df,
regress_frozen.py 4e9192dc8f78; code7/tools/pretrain_range.py a3bce3ab63c1; 64 stage-A job files = the generator's
rendering (`make_jobs_pt300x.py --check --stage A`). Read-only inputs asserted by the generator: make_jobs_pttpp.py
b4bfa6eb248d, make_jobs_ptlr1b.py 8dd7b3407459, make_jobs_pt2s505.py a5a7fb60eedc, make_jobs_rb6scale.py 19bc055c2d09,
pttpp_refs.json d10d720c1242, pt2s505_refs.json 479e394771e9, rb6scale_refs.json b65472eb3d29, runs/pttpp/code7
pretrain_range.py a3bce3ab63c1, runs/pttpp/readout/tools/readout_rules.py e91ee8a8a846. A change to the selection or the
read-out after this point is an amendment written here before any pt300x result is read.

## Submission, part 1 (coordinator, 2026-11-02 19:18 EST)
Only the 32 stage-A 93M jobs (L40S) were submitted (SUBMITTED_ flags; generator --check passed; no HOLD). The 32 294M
stage-A jobs are held: the fresh build audit (tests/audit_build.md, important items 1-2) found that 294M on B200 carries the
505M bridge's B200 offset (gain +4.3 on B200, same direction and size as the ~6-point difference 294M is meant to
discriminate) and that the 6-point rule as registered would very likely read SHOWN even with no growth from 53M to 294M.
Both points go to the user (hardware for 294M; an added new-point criterion) before the 294M jobs are rendered/submitted
and before any 93M result is read.

## Submission, part 2 and the user's decisions (coordinator, 2026-11-02 20:02 EST; no result read)
- The user (2026-11-02, after the build audit's items 1-2 were put to them): "B：仍用 B200；不加，等结果出来再说". So 294M
  stays on B200, and no new-point criterion is added: the registered rule (a)(b)(c) stands unchanged.
- The 32 stage-A 294M jobs were submitted (8 singleton slot chains llmadd_p3x-b200-slot0..7; SUBMITTED_ flags).
- Carried into the FINDINGS entry whatever the outcome (from tests/audit_build.md, written before any result): (i) the
  6-point verdict rests mostly on the 4 points seen before registration (new points ~23% of the slope weight; criterion
  (c) passes in a simulated no-growth-to-294M scenario 74-90% of the time), so a SHOWN verdict is only weakly confirmatory;
  (ii) 294M runs on B200, whose bridge offset (+4.3 gain at 505M) has the direction and size of the difference 294M is
  meant to discriminate (~6 points); the L40S-only slope (53M, 93M, 190M, 505M) is printed next to the verdict.
- Added descriptive outputs (audit item 3, before any result): per prior cell the LM share of the gradient (median, min
  over seeds), the share of on-window evals with prior > LM, and the clip coefficient, as in the pttpp read-out; and the
  'without b11' area gain per cell and for the slope. Computed by a separate descriptive script if readout_pt300x.py does
  not print them; they never change the verdict.
