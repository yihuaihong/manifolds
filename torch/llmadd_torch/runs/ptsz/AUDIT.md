# ptsz — four more L40S sizes on the fixed-token 7% ladder (2026-11-05)

User decision 2026-11-05 (relayed by the coordinator, notes/reply_manifolds_sizes_2026-11-05.md; the writing session's request
P1): add points to the fixed-3e8-token 7% size ladder, all on L40S, including the ~750M size; priority ptdom > new sizes >
ptshare, so every job of this batch is submitted with `--nice=1000` (**amendment N1, the user: `--nice=100`**; see the
amendment section at the end). Written BEFORE any training job of this batch was
submitted: only CPU reference / dry runs, the 731m smoke job (test runs, never counted) and tests ran. Built by a build agent;
the fresh adversarial audit (item v) is the coordinator's and is **pending**.

## Question
F99 (pt300x, 2026-11-04) read "SHOWN (increase)" for the six-point slope of the 8-seed area gain on ln N_nonemb (+2.79 / e-fold,
residual t df 4 [+0.85, +4.73]) with two qualifiers this batch is meant to remove:
- **I1**: 4 of F99's 6 points were seen before registration; (c) was already met by them, and the two new points only failed
  to break it (their own slope +0.14 / e-fold).
- **I2**: 294M and 1.07B ran on B200; the L40S-only slope (53M, 93M, 190M, 505M) is +2.16 with a residual interval (df 2)
  [-2.54, +6.86] that includes 0.

Questions (fixed 3e8 tokens, 7% arithmetic, every arm at its own best lr, 8 seeds, all L40S):
- **PRIMARY (for I2)**: across the eight L40S sizes 53M, 93M, 136M, 190M, 294M, 391M, 505M, 731M, does the 8-seed paired area
  gain (win-ctx - none, readout_rules.area) increase with ln N_nonemb?
- **SECONDARY (for I1, confirmatory)**: across the three never-seen sizes 136M, 391M, 731M only, does it increase?
- **Already observed at registration**: 4 of the 8 primary points (53M, 190M, 505M: F87; 93M: F99) and F99's B200 294M (the
  L40S 294M re-run here is an independent selection and run of the same shape, not a never-seen size). The secondary uses
  none of them. Nothing of 136M, 391M, 731M, or of 294M on L40S at 3e8 tokens, exists.

## Design (make_jobs_ptsz.py is the one source; `--plan` prints the table)
| size | shape | n_params | N_nonemb | ln N | tpp at 3e8 | grad ckpt | source | picks (none / win-ctx) |
|---|---|---|---|---|---|---|---|---|
| 53M | d384 x 6, 6 heads | 52,672,128 | 14,165,376 | 16.466 | 21.18 | off | F87 ptlr (reused) | 8e-3 / 2e-3 |
| 93M | d512 x 10, 8 heads | 93,306,368 | 41,964,032 | 17.552 | 7.149 | off | pt300x (reused) | 2.4e-3 / 1.2e-3 |
| **136M** | **d640 x 11, 10 heads, d_ff 2560** | **136,296,320** | **72,118,400** | 18.094 | 4.160 | off | **new** | stage A below |
| 190M | d768 x 12, 12 heads | 190,297,344 | 113,283,840 | 18.545 | 2.648 | on | F87 ptlr (reused) | 4.8e-3 / 1.2e-3 |
| **294M** | **d832 x 19, 13 heads, d_ff 3328** | 293,931,456 | 210,500,160 | 19.165 | 1.425 | on | **new (L40S re-run)** | stage A below |
| **391M** | **d960 x 20, 15 heads, d_ff 3840** | **391,256,640** | **294,989,760** | 19.502 | 1.017 | on | **new** | stage A below |
| 505M | d1024 x 24, 16 heads | 505,437,184 | 402,752,512 | 19.814 | 0.745 | on | F87 ptlr (reused) | 9.6e-3 / 1.2e-3 |
| **731M** | **d1152 x 29, 18 heads, d_ff 4608** | **731,431,296** | **615,911,040** | 20.239 | 0.487 | on | **new** | stage A below |
| (294M-B200) | d832 x 19 | 293,931,456 | 210,500,160 | 19.165 | 1.425 | on | pt300x (descriptive) | 9.6e-3 / 6e-4 |
| (1.07B) | d1280 x 36, 20 heads | 1,072,259,840 | 943,904,000 | 20.666 | 0.318 | on | ptlr_1b, B200 (descriptive) | 2.4e-3 / 6e-4 |

**Shapes.** The coordinator's shapes, all valid for the trainer (head dim 64: heads x 64 = d; d_ff = 4d; width and depth strictly
between the neighbours'). Exact counts with the trainer's own `pretrain_helix.build_model` (called as `pretrain_range.main()`
calls it, the real tokenizer, V = 100,278, tied embedding counted once) on the meta device: scale_work/ptsz/shapes_ptsz.py ->
shapes.out, which first reproduces make_jobs_pttpp.SIZES of the six existing sizes to the parameter. Names = the rounded totals
(136m, 391m, 731m: the provisional names were already right). Log distance (N_nonemb) to the geometric midpoint of the
neighbouring existing points: 136M +0.045 (93M-190M; the nearest valid shape), 391M +0.013 (294M-505M; nearest), 731M -0.001
(505M-1.07B; nearest). The CPU dry runs print `model: 136.3M parameters; gradient checkpointing off`, `391.3M ...; on`,
`731.4M ...; on` (grad ckpt auto: on above 150M parameters).

**Protocol = pt300x (= F87 ptlr) exactly, except shape, lr grid, --nice and --ckpt_every.** 3e8 tokens = 18,310 steps x 32 x 512
(the trainer's default --tokens 3e8; no --tokens / --eval_every flag), eval every 250 steps -> 75 evals, warmup 1% = 183
steps, cosine to 0.1x, AdamW (0.9, 0.95) wd 0.1, clip 1.0, bf16 autocast, lambda 5, window (0.05, 0.4) = prior on at steps
916..7323 (prior_events [[916, 1], [7324, 0]], on_frac 6408 / 18310), ctx site, --arith_lines 4 (7%), prior / read-out layers
n_layers // 3 and 2 n_layers // 3 (136M 3, 7; 294M 6, 12; 391M 6, 13; 731M 9, 19), gradient checkpointing auto. Arms 'none'
and 'win-ctx' = F87's flags (make_jobs_pttpp.ARM7). Data pretrain_data (sha256 checked by every job), seed s = the same data
stream and held-out split at every size.

**Code.** Trainer: runs/ptsz/code7 = `git archive 64abc19 tools/` (make_jobs_pttpp._extract), asserted equal file by file to
runs/pttpp/code7 and to F87's runs/ptlr/code (`diff -r`: identical), COMMIT verbatim, chmod a-w; pretrain_range.py
a3bce3ab63c1. Job body = pt300x's L40S body (= make_jobs_pttpp.job7's 7% body): HOLD kill switch (runs/ptsz/HOLD), trainer
sha check (exit 3), GPU-name guard 'L40S' (exit 4), data sha256 (exit 3), the INTENT line, `--dry_inputs`, training, CHECK7
verbatim on the run JSON -> out/<tag>/check_s<seed>.json; exit with the check's code; no selection hook. The generator imports
make_jobs_pttpp read-only (b4bfa6eb248d, no top-level call, asserted), calls its args7() / intent7() for every run and
asserts this batch's ARGS / intent equal to them up to the --out directory; pttpp's helpers take the shape from the facts, so
the new shapes only needed this batch's own SIZES table (make_jobs_pttpp.SIZES + the three new shapes, the shape rule and the
ladder order asserted). Read-only inputs asserted by sha256[:12]: pt300x's list + make_jobs_pt300x.py e8fd563c335a,
runs/pt300x/decisions.json 22532f929c7a, runs/pt300x/code7 pretrain_range.py a3bce3ab63c1; ptsz_refs.json 831d6694cd6f
once written.

**Step-0 lm references** (CHECK7 and the integrity re-check compare the run's step-0 lm within 1e-3): 294m keeps pttpp_refs.json
'7|294m' (frozen, never written). The new shapes: llmadd_torch/ptsz_refs.json (new file, read-only, sha256[:12] 831d6694cd6f),
written by scale_work/ptsz/refs_ptsz.py the pttpp way (refs_tpp.py): CPU step 0 of the trainer's own main() with the jobs'
own ARGS (scale_work/step0_ng.py: bf16 autocast, no_grad, 2 threads), 8 seeds, only after every identity held
(refs_table_write.txt "ALL IDENTITIES HOLD"): every dry run's model line / 8 operand sites per row / target line; dry
sections = F87's 505M job logs (seeds 0-1) and = pttpp's 294M job logs (seed 0) except the model line; step-0 lm identical
across arms; **known answers of the method**: recomputed CPU values of 294M seeds 0, 1 and 93M seed 0 equal pttpp_refs.json's
CPU values to the last bit (diff 0.0). No GPU value of the new shapes exists; the smoke gives one for 731M and 136M seed 0
(below).

## Deviations from pt300x (decided before any result; tests/args_diff.out lists every differing line)
1. Shapes (above) and lr grids (below).
2. **294M on L40S** (pt300x: B200): the user's "all on L40S"; an independent selection and run (its own stage A / X / C).
3. **`--nice=1000`** on every job (pt300x 20): the user's priority ptdom > new sizes > ptshare. [amendment N1: --nice=100]
4. **`--ckpt_every 2000` (F87's L40S value) for every size whose estimated run exceeds 4 h** (294M 4.16 h, 391M 5.3 h, 731M
   ~8.9 h: a requeue resumes instead of restarting from step 0), with `--open-mode=append` on those jobs (the log keeps every
   segment, as pt300x's B200 resume jobs); 136M (1.8 h) keeps pt300x's L40S rule (no ckpt, a requeue restarts). Resume
   granularity only: the trainer restores model, optimiser, all rng states, dyn and curve (pretrain_range.py:492-498); F87's
   53M-505M L40S runs used the same 2000. CHECK7 compares args.ckpt_every like every other argument.
5. Time limits by pt300x's L40S rule max(1, ceil(5 x est) / 2) h: 136M 05:00, 294M 10:30, 391M 13:30, 731M 22:30 (gpu48 MaxWall
   48 h; l40s_public MaxTime UNLIMITED, PreemptMode OFF).
6. Selection (select_ptsz.py) = select_pt300x.py + two fixes of the pt300x build audit (tests/audit_build.md there):
   M6 an OWN_FAILURES.txt name that is no run of this batch, or a listed run that finished with a passing check, prints a
   WARNING; M7 'unstable' is judged against the arm's lowest none lr INCLUDING a decided X (pt300x: grid[0]). Neither changes
   a pick.
7. The read-out computes the LM gradient share / clip / 'without b11' itself (pt300x needed a separate extra_desc.py; F99 I3).
Nothing else: the trainer, data, tokens, evals, window, lambda, site, layers rule, arms, seeds, CHECK7, stage rules and the
decision-log validation are pt300x's.

## Hardware
All on L40S, one GPU type for the whole batch: account <slurm-account>, l40s_public, QOS gpu48 (16 GPUs per user across
the user's gpu48 jobs, shared with ptdom / ptshare), `--nice=1000` [amendment N1: --nice=100], 8 CPUs / 64 GB, --requeue; GPU-name guard 'L40S' (exit 4
before any training on anything else). Reused L40S cells: 53M / 190M / 505M (F87, L40S logs) and 93M (pt300x, all L40S; F99's
results audit verified every log). The B200 cells (F99's 294M and 1.07B) enter only descriptive lines.

## lr grids (four consecutive rungs of the factor-2 ladder per arm; asserted in the generator)
pt300x's derivation at 3e8 tokens: log-linear interpolation in ln N_nonemb between the neighbouring sizes' 3e8 picks.
**Rule (made explicit and asserted):** grid = L[k-1], L[k], L[k+1], L[k+2] with L[k] <= interpolated < L[k+1]
(make_jobs_pttpp.grid_rule's convention; an interpolation on a rung counts as that rung), i.e. two rungs at or below it and
two above; every grid also straddles the interpolation computed with 505M none = 9.6e-3 (its REUSED pick) instead of the
geometric mean, with at least one rung on each side (asserted). Known answers: the rule reproduces pt300x's grids for 93M (both
arms) and 294M none; pt300x's 294M win-ctx (interpolation exactly on 1.2e-3) was set one rung lower by its own stated reasons
and is kept unchanged here for the 294M re-run (the user's spec).
Existing 3e8 picks (checked against the run dirs and decision files: F87 ptlr dirs with 8 seeds at 53M 8e-3 / 2e-3, 190M
4.8e-3 / 1.2e-3, 505M 9.6e-3 / 1.2e-3 = make_jobs_pttpp.REUSED / OPT3E8; ptlr_1b 2.4e-3 / 6e-4; runs/pt300x/decisions.json C
entries 93M 2.4e-3 / 1.2e-3 and 294M 9.6e-3 / 6e-4, asserted by the generator). 505M none enters the interpolation as the
geometric mean sqrt(4.8e-3 x 9.6e-3) = 6.79e-3 (make_jobs_pttpp.OPT3E8: 9.6e-3's 8-seed mean 57.5 = 4.8e-3's 4-seed mean 57.6).
| cell | neighbours' 3e8 picks | interpolated (505M none = 9.6e-3 instead) | grid | X if open |
|---|---|---|---|---|
| 136M none | 93M 2.4e-3, 190M 4.8e-3 | 3.50e-3 | 1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3 | 6e-4 / 1.92e-2 |
| 136M win-ctx | 93M 1.2e-3, 190M 1.2e-3 | 1.2e-3 (on the rung) | 6e-4, 1.2e-3, 2.4e-3, 4.8e-3 | 3e-4 / 9.6e-3 |
| 294M none | (pt300x: 190M 4.8e-3, 505M) | 5.69e-3 (6.73e-3) | 2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2 (pt300x's) | 1.2e-3 / 3.84e-2 |
| 294M win-ctx | (pt300x: 190M 1.2e-3, 505M 1.2e-3) | 1.2e-3 (on the rung) | 3e-4, 6e-4, 1.2e-3, 2.4e-3 (pt300x's) | 1.5e-4 / 4.8e-3 |
| 391M none | 294M 9.6e-3 (B200), 505M 6.79e-3 | 8.02e-3 (9.60e-3) | 2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2 | 1.2e-3 / 3.84e-2 |
| 391M win-ctx | 294M 6e-4 (B200), 505M 1.2e-3 | 8.60e-4 | 3e-4, 6e-4, 1.2e-3, 2.4e-3 | 1.5e-4 / 4.8e-3 |
| 731M none | 505M 6.79e-3, 1.07B 2.4e-3 | 4.04e-3 (4.81e-3) | 1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3 | 6e-4 / 1.92e-2 |
| 731M win-ctx | 505M 1.2e-3, 1.07B 6e-4 | 8.49e-4 | 3e-4, 6e-4, 1.2e-3, 2.4e-3 | 1.5e-4 / 4.8e-3 |
Notes: 391M's interpolation uses F99's B200 294M picks (the only 294M picks that exist at registration). 731M none with
505M = 9.6e-3 would give 4.81e-3, a hair above the 4.8e-3 rung (balanced grid 2.4e-3..1.92e-2); the geometric-mean convention
gives 1.2e-3..9.6e-3; both contain 2.4e-3, 4.8e-3, 9.6e-3 and the registered grid straddles both values. The none picks along
the ladder are noisy (93M's stage A had 2.4e-3 and 9.6e-3 within 0.35 points; F99's results audit: "none's pick jumps back
and forth, dominated by noise"), so the balanced grid (two rungs each side) is the one that minimises the chance of a STOP.

## lr selection (pre-registered; runs/ptsz/select_ptsz.py, run by hand by the coordinator; no hooks)
Per size and arm independently, F87 rule 1 exactly, as pt300x (readout_rules.best_lr with near_rule 'top2', in-training bpb,
readout_rules.bracket):
1. Stage A: the 4 grid lrs x seeds 0-3 (4 sizes x 2 arms x 4 lrs x 4 seeds = 128 runs).
2. Best lr = argmax of the mean area over seeds 0-3; if the top two are within 2 area points, the one with the lower mean
   final in-training bpb (a non-finite bpb is the worst; an exact tie -> the lower lr).
3. Open bracket = the pick is at an edge of the arm's grid and beats its neighbour by more than 2 points -> stage X: one rung
   beyond that edge, seeds 0-3, once per arm; then rule 2 over the five lrs. Still open after the X -> STOP (the user decides;
   the read-out is then NOT DETERMINABLE for that cell). An edge pick within 2 points of its neighbour is closed.
4. Closed -> stage C: seeds 4-7 at the pick (--save_model). The cell's 8 seeds = seeds 0-3 of the pick's stage A / X runs +
   seeds 4-7 of C. C = 4 sizes x 2 arms x 4 seeds = 32 runs; X adds 4 runs per extended arm (at most 32).
5. A diverged run keeps its area and counts against its lr; final val_loss > the same seed's none at the arm's lowest none lr
   (incl. a decided X; M7) + 0.5 nats is flagged 'unstable' (kept). ~~An own failure (runs/ptsz/OWN_FAILURES.txt) is scored
   over all 75 eval slots with the missing evals 0, bpb worst~~ [replaced by amendment I5: an own failure that recurs on one
   resubmission, e.g. a deterministic OOM, is listed in OWN_FAILURES.txt and makes its size cell NOT DETERMINABLE; never
   scored]; an infrastructure loss is resubmitted, not listed.
6. A run whose CHECK7 failed, whose recomputed area differs from the check's by more than 0.006 (double path), or that fails
   the integrity re-check withholds its arm: nothing is decided; the coordinator investigates.
`--write` appends the new X / C / STOP entries to runs/ptsz/decisions.json (append-only; the generator validates the whole
log) and writes picks.json, then prints `make_jobs_ptsz.py --stage X|C --size S --arm A --test-only` / `--submit`.

## Pre-registered analysis (runs/ptsz/readout_ptsz.py; written and self-tested before any ptsz run)
- **Cells**: each size at each arm's pick, seeds 0-7, gain_s = area(win-ctx, s) - area(none, s) (paired by seed; area =
  readout_rules.area = 100 x mean over all 75 evals of the 6-bucket held-out exact match). New cells (136M, 294M, 391M, 731M):
  the C decisions of runs/ptsz/decisions.json, recomputed by the selection rule and asserted equal. Reused L40S cells: F87
  53M / 190M / 505M (make_jobs_pttpp.REUSED) and pt300x 93M (its decision C lrs, its CHECK7 files must pass, double path);
  every reused run passes the same integrity re-check; with the FINDINGS check on, their 8-seed none / prior / gain must equal
  FINDINGS F99's table to 0.05. The B200 cells (F99's 294M, 1.07B) are loaded the same way for descriptive lines only.
- **PRIMARY**: the OLS slope b of the eight L40S 8-seed mean gains on ln N_nonemb (equal weights). **"SHOWN (increase)"** (or
  "(decrease)" for b < 0) iff ALL of:
  - (a) the 90% seed-bootstrap CI of b excludes 0 on the side of b (seeds resampled with replacement within each size, the two
    arms paired; 10000 resamples; numpy default_rng(0), one (10000, 8) index draw per size in ascending N order; percentile
    (5th, 95th));
  - (b) the slopes on seeds 0-3 and on seeds 4-7 (point estimates, the same eight sizes) both have the sign of b;
  - (c) the size-level interval b +- t(0.95, df 6) x se(b), se(b) = sqrt(SSR / 6 / Sxx) from the OLS residuals of the eight
    points, excludes 0.
  Otherwise "NOT SHOWN" with the numbers and the failed criteria.
- **SECONDARY (the confirmatory test for F99 I1; reported under its own name, "VERDICT (secondary, new sizes only)")**: the
  same slope over 136M, 391M, 731M only (294M excluded: its B200 twin was seen), criteria (a) and (b) only (the same bootstrap
  convention over three sizes); its df-1 residual interval is printed but is not a criterion.
- **NOT DETERMINABLE**: a primary cell not final (no C, a STOP, a run missing or unfinished) or with an integrity problem ->
  the primary reads NOT DETERMINABLE (no slope is read); the secondary likewise for its three cells (it does not need 294M or
  the reused cells). **INVESTIGATE**: a new cell whose 8-seed none or prior mean lies outside its expected range by more than
  15 points -> both verdict lines carry "INVESTIGATE BEFORE READING" and the coordinator records the investigation here
  before anyone reads the verdict.
- **Descriptive (printed, no verdict)**: the 10-point all-hardware slope (the eight L40S + F99's B200 294M and 1.07B; also with a
  B200 indicator: b and c); the hardware bridge at 294M (L40S - B200 per arm and in gain, 8-seed means, seed bootstrap 90%
  CI; pt300x's 505M bridge had B200 - L40S none -4.0, gain +4.3); the eight leave-one-size-out slopes; none and prior arms
  separately; the ln R_0.5 slope (sign only, F87 rule 8's metric, readout_rules.speedup); per prior cell the LM share of the
  pre-clip gradient (median over seeds of the per-seed median a / (a + b) over on-window evals; min seed), the share of
  on-window evals with prior > LM and the clip coefficient (flag below 0.6 or prior > LM in most evals: the project's
  auxiliary-loss rule); the 'without b11' gain per cell (readout_rules.B5) and its primary / secondary slopes; lr swaps within
  the tie band (seeds 0-3; F99 M3's limitation applies to the reused cells); for reference F99's six cells and the four reused
  L40S cells alone, reproduced.
- **What the verdicts may be called.** Fixed-token statements: at 3e8 tokens tpp falls from 21.2 to 0.49 along the L40S ladder,
  so size and training sufficiency stay confounded (F99 I4 is not addressed by this batch; neither is the prior-dose
  confound: the LM share rose 0.81 -> 0.91 along F99's ladder). "SHOWN (increase)" (primary) = the area gain at 3e8 tokens
  increases with ln N_nonemb across eight L40S sizes with the size-level interval; with the secondary also SHOWN, the
  increase is also seen on three sizes never seen before registration. Not a speed-up claim, not "the prior's effect grows
  with size" (F99: the gain rose because none's area fell). The primary still contains the four seen points (see the
  conditional false-SHOWN rate below); the secondary is the confirmatory one.
  [Amendment after the fresh build audit, before any result: primary SHOWN with secondary NOT SHOWN = only "the new sizes
  did not fall below the old trend" (audit I3); a secondary SHOWN is reported with its false-SHOWN rates (0.07-0.22) beside
  it (audit I2, the user: no rule change); ~~a verdict whose median / trimmed-mean slopes or leave-one-seed-out range
  (printed, I1) disagree with it in sign rests on a few late none seeds~~ [amendment 2, I-0: a magnitude rule: when the
  median-gain slope / mean slope, the trimmed-mean slope / mean slope, or the leave-one-seed-out extreme toward 0 / mean
  slope is below 0.5, the read-out prints "the slope rests on a few seeds" next to the verdict and FINDINGS reports the
  verdict with that label, the three ratios and the late counts. **No label does not mean robust: in simulation only 4-23%
  of false SHOWN carry the label** (audit of amendment 2, m6: primary 0.04-0.13, secondary 0.07-0.23); the label is read
  for SHOWN verdicts only (near b = 0 the ratios explode and it fires without information).]

## False-SHOWN rates (simulation, before any result; tests/sim_null.py -> sim_null.out / sim_null.json)
The read-out's own decision functions (slope_block with its fixed bootstrap draws, verdict) on simulated cells; 2000
simulations per row. Seed scatter: the 32 within-cell centred per-seed gains of the four reused L40S cells (SD 6.96; SD of an
8-seed mean 2.46), drawn with replacement. Size scatter beyond the seed noise: tau6 1.85 (F99's six-point residuals, df 4),
tau4 3.12 (the four L40S points, df 2), also 0 and 3.41 (pt300x's build audit value). No growth = one constant gain mu0.
| scenario | tau 0 | tau6 1.85 | tau4 3.12 | 3.41 |
|---|---|---|---|---|
| PRIMARY false SHOWN (increase), conditional on the four seen L40S points, new cells at mu0 = their mean 26.85 | 0.117 | 0.127 | 0.143 | 0.142 |
| same, new cells at mu0 = F99's six-cell mean 28.77 | 0.382 | 0.322 | 0.273 | 0.299 |
| PRIMARY false SHOWN (increase), unconditional (all eight simulated; the rule's own rate) | 0.032 | 0.032 | 0.049 | 0.056 |
| SECONDARY false SHOWN (increase) (three new cells; conditional = unconditional) | 0.074 | 0.115 | 0.186 | 0.216 |
| SECONDARY false SHOWN (decrease) | 0.056 | 0.131 | 0.186 | 0.197 |
| power: PRIMARY SHOWN (increase), new cells on F99's six-point line (+2.79) | 0.964 | 0.877 | 0.727 | 0.693 |
| power: SECONDARY SHOWN (increase), same | 0.632 | 0.587 | 0.556 | 0.578 |
| power: PRIMARY / SECONDARY, new cells on the L40S-only line (+2.16) | 0.863 / 0.494 | 0.711 / 0.467 | 0.567 / 0.473 | 0.550 / 0.492 |
Reading (recorded before any result, for the coordinator and the user; the registered rules are NOT changed by this build):
- The primary's own false-positive rate is near its nominal one-sided 5% (0.03-0.06), but given the four points already seen
  it is 0.12-0.14 if "no growth" sits at their mean, and 0.27-0.38 if it sits at F99's six-cell mean (the new sizes lie at
  larger ln N on average than the seen ones, so a flat level above the small sizes' reads as growth). So a primary "SHOWN"
  alone still leans on the seen points (F99 I1 is reduced, not removed, by the primary).
- **The secondary's (a) is a seed-only interval** and (b) is weak with three points, so with realistic size scatter its false
  SHOWN (increase) rate is 0.12-0.22 (tau 1.85-3.41; 0.07 with no size scatter), about two to four times the nominal 5-10%,
  and its power against F99's line is only 0.56-0.63. Its df-1 residual interval (printed, not a criterion) is the only
  size-level check and has almost no power at df 1. The secondary as specified therefore cannot by itself turn "I1" into a
  confirmed increase at a 5% error rate; a secondary "SHOWN" should be reported with this rate next to it. (If the user wants a
  stricter secondary, it has to be an amendment written here before any ptsz result is read; this build does not change it.)

## Expected ranges (outside -> check the run before using it; the read-out flags +-15)
- 136M: none ~58-68 (between 93M 60.9 and 190M 66.5), prior ~88-91 -> gain ~20-32.
- 294M (L40S): none ~55-67 (F99's B200 294M 60.8; between 190M 66.5 and 505M 57.5), prior ~88-91 -> gain ~22-35.
- 391M: none ~53-63 (between 294M 60.8 and 505M 57.5), prior ~88-91 -> gain ~26-37.
- 731M: none ~50-60 (between 505M 57.5 and 1.07B 53.3), prior ~88-91 -> gain ~29-40.
- Every run: code_sha a3bce3ab63c1, 18310 steps, 75 evals, prior_events [[916, 1], [7324, 0]] / on_frac 0.349973 (win-ctx),
  none never on; step-0 lm = the reference within 1e-3 (136M seeds 0-7: 11.6575 / 11.6390 / 11.6719 / 11.6547 / 11.6359 /
  11.6549 / 11.6393 / 11.6464; 391M: 11.7264 / 11.7023 / 11.7123 / 11.7058 / 11.6916 / 11.7441 / 11.6995 / 11.6954; 731M:
  11.7453 / 11.7297 / 11.7373 / 11.7874 / 11.7259 / 11.7214 / 11.7096 / 11.7518; 294M: pttpp_refs.json).
- Divergence risk: none at 1.92e-2 (294M, 391M) and any X at 3.84e-2.

## Metrics (unique definitions: tools/readout_rules.py of the frozen pttpp read-out snapshot, sha256[:12] e91ee8a8a846)
area (above); t_theta (tokens until the bucket mean first reaches theta, linear interpolation, censored at the 300M budget);
R_0.5 (readout_rules.speedup); in-training bpb = val_bits_per_byte at the last eval; LM share = grad_lm / (grad_lm +
grad_prior) of the trainer's grad_info (pre-clip norms, on-window evals). Train vs eval conditions: identical to F87 / pt300x
(runs/ptlr/AUDIT.md "Train vs eval conditions"); only the shape and lr differ.

## Smoke (731m on L40S; one GPU job submitted by the build agent; make_jobs_ptsz.py --smoke -> runs/ptsz/smoke)
Job llmadd_psz-smoke731 (account 625, l40s_public, gpu48, --nice=1000 in the file; resubmitted with `sbatch --nice=100`, see the
amendment, 2 h, GPU guard L40S, HOLD, trainer sha, data sha256);
every segment = the trainer's own main() via smoke/smoke_seg.py (torch.save timed, then torch.cuda peak memory), a rendered
stage-A ARGS (asserted) with --out -> runs/ptsz/smoke/<seg> and only these test flags changed:
- A: 731m win-ctx 6e-4 seed 0, `--window 0,0.4 --eval_every 100 --ckpt_every 100 --stop_after 210`: the prior on from step 0,
  so evals 100 / 200 are in-window evals with the previous step's gradients still allocated (the trainer's memory peak: the
  LM and prior gradient passes of grad_info before zero_grad, gradient checkpointing auto = on), plus two ckpt writes;
- B: the same command, `--stop_after 310`: resumes A's step-199 ckpt (the loaded checkpoint stays referenced in main()) and
  reaches the in-window eval at 300; B's step-200 lm vs A's (the resume test of the ptlr_1b smoke);
- C: 731m none 2.4e-3 seed 0, the real ARGS + `--stop_after 500` (evals 0 / 250 / 500); D: the same + `--eval_every 25
  --stop_after 250` (11 evals) -> s/step and s/eval; E: 136m win-ctx (grad ckpt off), `--window 0,0.4 --eval_every 100
  --stop_after 210` -> memory;
- only if A fails: A2 / B2 with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, A3 with --lm_chunk_tokens 8192 (both would be
  deviations; reported to the user, never used without them).
The smoke's run JSONs never enter a selection or read-out (other directory, other tags than any counted run). Its result:
"Smoke result" below (tests/smoke_report.py).

## Smoke result (2026-11-05 18:28 EST; superseded in part by the amendment: 21999324 was cancelled by the coordinator and
resubmitted as **22011938** at nice 100; run `tests/smoke_report.py 22011938`)
**The smoke job 21999324 (submitted 16:24) had not started after 2 h** (PENDING, reason None, priority 11,626 with
--nice=1000; ~1,000 of the 1,491 pending l40s_public jobs rank above it). [Later cancelled by the coordinator, resubmitted as
22011938 at nice 100.] Hence:
- **Cost**: from the existing L40S run JSONs (the table below; consistent with ptlr 505M 6.4 h, 53M 1.0 h, 190M 2.6 h, pt300x
  93M 1.4 h per run), the 731M per-run time an extrapolation (8.9 h), not a measurement.
- **731M memory on L40S is NOT verified.** Second-path evidence from existing nvidia-smi logs (reserved memory, no memory
  pressure): 190M 30,911 MiB, 294M 38,149 MiB, 505M 40,063 MiB (2% jobs on A100 80 GB, gpu.csv), 1.07B 60,472 MiB (B200
  smoke, scale_work/b200/smoke/gpu_usage_r*.log); linear in n_params between 505M and 1.07B -> ~48,200 MiB at 731M, above
  the L40S's 46,068 MiB. Reserved memory under no pressure overstates the need (the caching allocator frees cached blocks
  before an OOM), and 505M did run on L40S in F87, but the margin is small or negative: the trainer's peak is the in-window
  eval step (grad_info: two extra full gradients and their flattened copies while the previous step's grads still exist,
  ~4 x 2.9 GB at 731M, on top of params + grads + Adam 11.7 GB and the full 32 x 512 x 100,278 logits path, lm_chunk 0) and a
  resumed run keeps the loaded checkpoint referenced (+2.9 GB). An OOM would hit at step 1000 (the first in-window eval),
  ~30 min into every 731M win-ctx run.
- The smoke, once it runs, answers this (22011938, nice 100, submitted before any batch job at the same nice 100):
  `python runs/ptsz/tests/smoke_report.py 22011938` prints memory / speed / resume / step-0 calibration and writes
  smoke/SMOKE_VERDICT.json; **submit_ptsz.sh refuses to submit until that verdict passes, unless `--no-smoke`**. If A fails,
  the job runs A2 / B2 (expandable_segments) and A3 (--lm_chunk_tokens 8192): those would be deviations for the user to choose
  (or a smaller ~730M shape would not help: the logits path is size-independent). If the smoke's speed differs from 8.9 h,
  TRAIN_S['731m'] in the generator (and so the 731M time limit, 22:30 now) should be updated before the 731M jobs are
  submitted (no job file is submitted yet, so re-rendering is allowed).

## Cost and wall time (make_jobs_ptsz.py --plan)
Training seconds per run at 18,310 steps / 75 evals, L40S, from run JSONs (curve[-1]['sec'] medians; scale_work/ptsz/
timing_ptsz.py -> timing.out) + 180 s start-up / dry run / check + the ckpt writes:
- 136M (grad ckpt off): log-log in n_params through 53M (3470 s) and 93M (4975 s), both off -> 6317 s -> **1.80 L40S-h**.
- 294M: pttpp 294M L40S at its two tpp lines (9570 / 4083 steps, 75 / 74 evals) -> 0.7596 s/step + 10.70 s/eval -> 14711 s
  -> **4.16 L40S-h** (with 9 ckpt writes).
- 391M: log-log in n_params through 294M (14711 s) and F87 505M (23170 s), exponent 0.838 -> 18695 s -> **5.28 L40S-h**.
- 731M: the same log-log extrapolated -> 31581 s -> **8.89 L40S-h** before the smoke (replaced by the smoke's measurement if
  it ran: "Smoke result").
- The win-ctx arm costs ~1% more (F87 505M L40S: 23170 / 22995 s); ckpt writes ~3 s / GB (a guess before the smoke; 9 per run).
| size | per run | stage A (32) | C (8) | A + C | worst case + X on both arms (8) |
|---|---|---|---|---|---|
| 136M | 1.80 h | 57.8 | 14.4 | 72.2 | 86.6 |
| 294M | 4.16 h | 133.2 | 33.3 | 166.5 | 199.8 |
| 391M | 5.28 h | 168.9 | 42.2 | 211.1 | 253.4 |
| 731M | 8.89 h | 284.4 | 71.1 | 355.5 | 426.6 |
| total | | 644.3 | 161.1 | **805.4 L40S-h** | 966.4 |
Wall time of running (list scheduling in submission order; queue waits and the selections by hand come on top): at 16
concurrent L40S (the gpu48 per-user cap, shared with the user's ptdom / ptshare jobs) stage A ends after ~40 h and C (+ X)
adds ~9 h (18 h) -> **~49-58 h**; at 12 concurrent ~63-72 h; at 8 ~89-98 h. The 731M stage A (the long pole, submitted
first) takes 2 waves of ~8.9 h at 16 concurrent, so its selection is possible after ~18 h and its C ends ~27 h in.
**Queue at writing (2026-11-05 ~17:00; before amendment N1)**: l40s_public 178 running / 1,491 pending; a --nice=1000 job of ours has priority
11,626, below ~1,000 of the pending jobs (the user's ptshare jobs at nice 2000 have 12,265 from age); sbatch --test-only
estimates a start on 2026-11-11 (tests/testonly.out; backfill estimates are pessimistic but the queue is long). The wall time
is therefore dominated by the queue, not by the 805 GPU-h.

## Disk
myquota at writing: /scratch 4.69 / 5.00 TB (93.9%; ~150 GB below the 97% line submit_ptsz.sh enforces, ~300 GB to full).
- Transient ckpts (model + 2 Adam moments, deleted at the end of each run; one extra copy while writing): 731M 8.8 GB, 391M
  4.7 GB, 294M 3.5 GB per running job. 16 concurrent 731M runs = 141 GB (+ up to 8.8 GB per simultaneous write). This is the
  one disk risk of the batch: with other batches' growth it can reach the 97% line during the 731M stage A. The coordinator
  should check myquota before submission and while the 731M wave runs (HOLD stops pending / requeued jobs).
- Stage C models (fp32): 136M 8 x 0.55, 294M 8 x 1.18, 391M 8 x 1.57, 731M 8 x 2.93 GB = 49.7 GB. Run JSONs small.

## Submission (stage A only; the coordinator, after the fresh audit)
`bash runs/ptsz/submit_ptsz.sh` checks (no HOLD; the 731m smoke passed = smoke/SMOKE_VERDICT.json pass, written by
`tests/smoke_report.py JOBID`, unless `--no-smoke` (the coordinator's explicit acceptance of an unverified 731M memory fit);
`make_jobs_ptsz.py --check --stage A`: every stage-A job file = its rendering,
the snapshot, the frozen shas, the step-0 references; 32 runs per size, 128 in all; none submitted yet; every job --nice=100
(amendment N1) on l40s_public with its own llmadd_psz- name; myquota /scratch < 97%) and prints the 128 sbatch lines in the order 731M,
391M, 294M, 136M; `--go` submits them and writes SUBMITTED_<run> flags. X / C: by hand after `select_ptsz.py --write`.
Kill switch: `touch runs/ptsz/HOLD` (a pending or requeued job exits at start; a running one is not stopped).

## Tests before submission (runs/ptsz/tests; outputs *.out next to the scripts)
| test | result |
|---|---|
| shapes (scale_work/ptsz/shapes.out) | the trainer's build_model on the meta device reproduces make_jobs_pttpp.SIZES of the six existing sizes to the parameter; 136M 136,296,320 / 72,118,400, 391M 391,256,640 / 294,989,760, 731M 731,431,296 / 615,911,040; each the nearest valid shape to its geometric midpoint |
| CPU references (scale_work/ptsz: jobs 21999437 + 22000205 on cpu_short via cpu_run.sh; refs_table.txt, refs_table_write.txt) | ALL IDENTITIES HOLD: 16 dry runs (4 sizes x 2 arms x seeds 0-1) with the jobs' own ARGS: model line as designed, 8 operand sites in each of 3 rows, target line only for win-ctx; every dry section = F87's 505M job log of the same arm and seed except the model line; seed 0 also = pttpp's 294M job log (294M including the model line); step-0 lm identical across arms (seed 0, all sizes); known answers 294M s0 / s1 and 93M s0 = pttpp_refs.json's CPU values, diff 0.0 (bitwise); lm0 seeds 0-7 of 136M / 391M / 731M -> ptsz_refs.json 831d6694cd6f (new, read-only; pttpp_refs.json d10d720c1242 unchanged, asserted before and after). First pass: seeds 4-7 failed in refs_ptsz.job_args (facts() requires stage C for seeds 4-7; fixed to their stage-C ARGS, rerun) |
| render + `bash -n` (tests/render.out) | 128 stage-A job files written by `make_jobs_ptsz.py --stage A`; `bash -n` clean on all 128; `--check --stage A` OK; all `--nice=1000` (re-rendered with --nice=100 after amendment N1), `l40s_public`; time 136M 05:00, 294M 10:30, 391M 13:30, 731M 22:30; `--ckpt_every 2000` + `--open-mode=append` on 294M / 391M / 731M, none on 136M; no `--save_model` in stage A |
| args_diff.py | ARGS DIFF PASS. (1) Equivalence with pt300x's SUBMITTED jobs after the substitution p3x- -> psz-, runs/pt300x -> runs/ptsz, *_pt300x.py -> *_ptsz.py: 93M (known answer, 4 jobs: A none / win-ctx, X, C) differs in exactly 6 lines (`--nice` 20 -> 1000 [after amendment N1: 20 -> 100; tests/args_diff.out], 4 comment lines), ARGS and intent JSON identical; 294M (4 jobs: A none / win-ctx, C none / win-ctx) differs in the hardware header (B200 slot / dependency / constraint / comment / account 520 / 16 CPUs / 128G / 03:00 -> L40S name / account 625 / partition / qos / 8 CPUs / 64G / 10:30 / open-mode / output), the comment lines, the GPU guard B200 -> L40S, and ARGS / intent only in ckpt_every 500 -> 2000; CHECK7 identical in all. (2) ARGS vs pttpp 294M, F87 190M / 505M, ptlr_1b 1.07B C, pt300x 93M: only shape / lr / seed / tag / out / --tokens / --eval_every (pttpp) / --save_model / --ckpt_every. (3) CHECK7 byte-identical to make_jobs_pttpp.CHECK7 in all 128 stage-A renderings |
| dry_digest.py | DRY DIGEST PASS 8/8: seed 0 of every size x arm, CPU --dry_inputs with the rendered stage-A ARGS (asserted = the job file's ARGS line): 294M none / win-ctx = pttpp's 294M job logs line for line (sha256[:16] ce8af7afdb623bab / 5d4742770e61f1e5); 136M / 391M / 731M = the same except the model line (without it e04f706634090844 / 070b276d8bb6fd58 for every size), model lines 136.3M off / 391.3M on / 731.4M on |
| check7_ka.py | CHECK7 KA PASS 36/36: CHECK7 with this batch's intent form passes on real F87 190M none s0 (L40S, ckpt 2000), F87 505M win-ctx s1, pt300x 93M none C s5 (ckpt 0) and pt300x 294M win-ctx C s6 (B200, ckpt 500); planted lr x2, eval_every 26, tokens 31.3M, steps 18309, shifted window events, step-0 reference +0.002, ckpt_every flipped, d_model / n_params of 731M -> rc 4 each |
| select_ptsz.py --selftest | SELECT SELFTEST PASS 25/25 (synthetic roots, real CHECK7 files, the real generator): incomplete -> wait; interior pick -> C; top-two tie -> lower in-training bpb; open upper (136M none) -> X 1.92e-2 and open lower (294M win-ctx) -> X 1.5e-4 (L40S, nice 1000 -> 100 after N1, ckpt 2000, no --save_model); X worse -> C (seeds 4-7, --save_model, 136M without --ckpt_every); X better again -> STOP, nothing after it; CHECK7 failure, double-path mismatch, step-0 lm off (731M against ptsz_refs.json) -> WITHHELD; unfinished -> wait; an own failure scored 0 (replaced by I5, see the amendment); M6 warnings (misspelled name; a listed run that passed); M7 unstable reference after a lower X; 6 malformed decision logs refused (one with a size of another batch). One test-code bug found and fixed on the way (the 136M C check matched the comment line "No ckpt (--ckpt_every 0)"; now it reads the ARGS line) |
| readout_ptsz.py --selftest | READOUT SELFTEST PASS 27/27: designed slope +3 on the 10 cells -> primary SHOWN (+3.000 on 8 seeds and both halves, df 6, cells = the eight L40S in ascending N) and secondary SHOWN (+3.000, df 1, exactly 136M / 391M / 731M, criteria a, b only); arms -3 / 0; 10-point 3 (indicator c 0); bridge 0; leave-one-out +3; without-b11 +3; ln R_0.5 > 0; LM share 0.833 everywhere, clip 0.962; one tied swap; zero slope -> both NOT SHOWN (a); negative -> both SHOWN decrease; halves disagree -> both NOT SHOWN (b); primary residual interval includes 0 -> NOT SHOWN (c) with (a)(b) passing, the secondary still decided by (a)(b) only; old sizes sloped but new sizes flat -> secondary NOT SHOWN (a) b 0.000; B200 offset +6 -> primary unaffected (2.000), indicator c 6.000, bridge gain -6.00; a B200 cell missing -> primary still determined, 10-point "not available"; reused run with another code_sha -> primary NOT DETERMINABLE, secondary still determined; a failed pt300x 93M CHECK7 file -> NOT DETERMINABLE; a failed 391M C check -> both NOT DETERMINABLE; 731M win-ctx without C -> both NOT DETERMINABLE; 294M none without C -> primary NOT DETERMINABLE, secondary SHOWN; a missing C run, a C decision != the recomputed pick -> NOT DETERMINABLE; prior ~70 -> both verdict lines carry INVESTIGATE BEFORE READING; FINDINGS check on with designed cells -> NOT DETERMINABLE; LM share 0.40 in 731M -> FLAG, verdict unchanged |
| readout_ptsz.py --ka (real reused runs; tests/readout_ka.out) | KA PASS 19/19: the 96 reused runs (53M, 93M, 190M, 505M L40S; 294M, 1.07B B200; pt300x runs also CHECK7 + double path) pass the integrity re-check; 8-seed gain / none / prior of the six F99 cells = pt300x's audit_results to 4 decimals (22.7927 / 28.8741 / 23.5299 / 29.1067 / 32.1885 / 36.1292); F99's six-point slope +2.7871, bootstrap [+1.7098, +3.9214], residual df 4 [+0.8455, +4.7287], p 0.0377, halves +2.8064 / +2.7677, leave-one-out, arms -2.6947 / +0.0924, ln R_0.5 +0.0301, LM shares 0.81 / 0.86 / 0.88 / 0.89 / 0.91 / 0.91, without-b11 slope +2.517; the L40S-only four-point slope +2.1608 [+0.5840, +4.0296], residual df 2 [-2.5355, +6.8571] (= F99 I2) |
| select / readout on the real root before submission (tests/select_presubmit.out, readout_presubmit.out) | every arm waits (16 runs missing each); primary and secondary NOT DETERMINABLE (no C decision); the six reused cells print F99's numbers |
| sim_null.py (tests/sim_null.out) | the false-SHOWN table above (2000 simulations per row, 503 s) |
| regress_frozen.py | REGRESSION PASS: runs/pttpp/FROZEN.txt 26/26 before and after; the frozen make_jobs_pttpp re-renders all 302 submitted pttpp job files byte for byte in the same process that imported and fully rendered make_jobs_ptsz; make_jobs_ptlr1b re-renders its 12; make_jobs_pt300x re-renders its 84 submitted job files; pt300x decisions / picks / select / readout, pttpp_refs.json and make_jobs_ptlr.py unchanged; no runs/ptlr job file modified since 2026-11-05 |
| sbatch --test-only (2 jobs; tests/testonly.out) | both accepted (nothing submitted): psz-731m-lr2.4e-3-none-s0 and psz-136m-lr6e-4-win-ctx-s3 on l40s_public, estimated start 2026-11-11 (the queue, see "Cost") |
| submit_ptsz.sh (no --go; tests/submit_dry.out) | without options it refuses: the 731m smoke has not passed (smoke/SMOKE_VERDICT.json missing); with --no-smoke the checks pass (/scratch 94.03%) and 128 sbatch lines are printed in the order 731M (32), 391M (32), 294M (32), 136M (32); nothing submitted |
| the smoke (runs/ptsz/smoke; job 21999324) | **not run**: still PENDING after 2 h (priority 11,626 behind ~1,000 jobs); see "Smoke result" |

## Audit status (experiment-audit)
i double path: area recomputed from the run JSON vs CHECK7's printed area (<= 0.006) for every new and pt300x run; nvidia-smi
memory vs torch peak in the smoke; the reused cells vs FINDINGS F99 (read-out) and pt300x's audit_results (KA). ii asserts:
generator (frozen shas incl. ptsz_refs.json, import without main, ARGS / intent = pttpp's functions, shapes / names / ladder
order, the grid rule and its straddle, steps / evals / window, the ckpt rule, the gpu48 wall limit), every job (trainer sha,
data sha256, GPU name, CHECK7), selection and read-out (integrity re-check of every counted run, decision-log validation, the
pt300x picks = its decisions.json). iii known answers: shapes = the existing six; CPU step-0 method bitwise on 294M / 93M; dry
digests = pttpp / F87 logs; pt300x job texts reproduced up to the intended differences; CHECK7 on real F87 / pt300x / ptlr_1b
runs + planted failures; F99 reproduced by the read-out; designed self-tests. iv: - (no result yet). v: **pending** — the
fresh general-purpose adversarial audit is the coordinator's (this build agent did not start one, by instruction). vi: trap
list (positions / template / range = F87; the prior's activity asserted per run; no silent filtering: own failures -> NOT DETERMINABLE (I5),
missing runs block; the secondary's weak error control is recorded above). vii: every job prints the decoded training and eval
rows (--dry_inputs) before training; the CPU dry logs (scale_work/ptsz/refs_out) show them for every size and arm.

## Files at writing (sha256[:12]; 2026-11-05 ~18:35 EST, before any training submission)
make_jobs_ptsz.py ae1a75c91d3e; runs/ptsz: select_ptsz.py a9bca4a4b1b8, readout_ptsz.py d66129bde8ac, submit_ptsz.sh
68cea25b58f1; tests/args_diff.py b32001611a4c, check7_ka.py efeeaa26ddbd, dry_digest.py 6aaef417c37c, regress_frozen.py
0c40ac20cb4c, sim_null.py d18a05205c90, smoke_report.py 90c040c09ebd; scale_work/ptsz: shapes_ptsz.py b3cfdad9ded4,
refs_ptsz.py c9a4c1e6088a, timing_ptsz.py 669e818817b7; llmadd_torch/ptsz_refs.json 831d6694cd6f (read-only); smoke:
job_smoke731.sbatch 08cf626a08f8, smoke_seg.py c62eff092e5c (job 21999324, cancelled; now 22011938, SUBMITTED_smoke731); code7/tools/pretrain_range.py
a3bce3ab63c1; 128 stage-A job files = the generator's rendering (`make_jobs_ptsz.py --check --stage A`). Nothing of the
batch's training is submitted (no SUBMITTED_psz-* flag). A change to the selection or the read-out after this point is an
amendment written here before any ptsz result is read; TRAIN_S['731m'] (cost / time limit only) may still follow the smoke.

## Amendment after the fresh build audit (before any result)
The fresh adversarial build audit (runs/ptsz/tests/audit_build.md, 2026-11-05; no blocking item) went to the user through
the coordinator. Written 2026-11-06 before any ptsz training job was submitted and before any ptsz result exists (no run of
this batch has trained; the smoke 22011938 is still pending). The registered PRIMARY and SECONDARY slope criteria are
unchanged; the NOT DETERMINABLE conditions are widened by I5 (below). The user's decisions, as relayed by the coordinator:
- **N1 (the user: "P1 jobs use --nice=100, not 1000").** Done by the coordinator: both `#SBATCH --nice` lines of
  make_jobs_ptsz.py (the 128 training jobs and the smoke rendering), the docstring, and submit_ptsz.sh's header checks now
  say 100; the 128 stage-A job files are re-rendered (`--stage A`, `--check --stage A` pass; all say `--nice=100`). Verified
  by the build agent (grep: 128 x `#SBATCH --nice=100`; submit_ptsz.sh checks `--nice=100`). Deviation 3 now reads nice 100.
- **Smoke resubmission (audit M2).** The coordinator cancelled 21999324 (nice 1000, an estimated start around 11-11) and
  resubmitted the same file as **22011938** with `sbatch --nice=100 smoke/job_smoke731.sbatch` (runs/ptsz/SUBMITTED_smoke731).
  The file's own directive still says `--nice=1000` (the command line overrides it; Slurm holds the script as submitted);
  the generator's smoke rendering now says 100, so `make_jobs_ptsz.py --smoke` would refuse (explicitly) to treat the
  on-disk file as its rendering: harmless, since the smoke is never re-rendered. Read it with
  `python runs/ptsz/tests/smoke_report.py 22011938`. At 2026-11-06 it is PENDING, priority 12,474.
- **I4 (smoke criterion).** Done by the coordinator: tests/smoke_report.py requires max_reserved <= total - 2 GiB for the
  731m segments A-D (it was `< total`, which the allocator always meets). The audit's other suggestion (nvidia-smi max
  memory.used <= 44 GiB) is printed by smoke_report.py but not a criterion; the coordinator reads it.
- **I1 (the user: "add to P1, print only").** readout_ptsz.py prints, next to the primary verdict, next to the secondary
  verdict and with the 10-point descriptive slope: the slope of the per-cell MEDIAN gain; the slope of the trimmed-mean gain
  (the min and the max seed dropped per cell); the leave-one-seed-out range (every cell x seed left out: min and max, and
  which cell / seed); the count of none seeds with area < 45 per cell (late learners). Never a criterion; the verdict rules
  are unchanged. Known answers (readout --ka, the four reused L40S cells; the coordinator's independent numbers): median
  slope +1.0059 (want +1.006), trimmed +1.1673 (+1.167), leaving out 505M seed 5 +1.2126 (+1.213; the minimum of the range);
  mean slope +2.1608; F99's six cells median slope +1.8228 (the audit's +1.82). Self-test: a planted single late-learning
  none seed (731m seed 6, none area 25 points lower) moves the mean slopes (primary 0 -> +0.44, secondary 0 -> +1.26) while
  the median and trimmed slopes stay 0.000, and the leave-one-seed-out minimum leaves out exactly 731m seed 6. ~~Reading: a
  verdict whose median / trimmed slopes or leave-one-seed-out range fail to agree in sign with it rests on a few late none
  seeds (audit I1), and the FINDINGS entry says so.~~ [replaced by amendment 2, I-0 (a sign rule never fires on the audit's
  own example): the magnitude label "the slope rests on a few seeds".]
- **I5 (own failures; pre-registered here, enforced in code).** I5 replaces F87 rule 2's own-failure-scored-0 for this batch;
  decided by the coordinator, put to the user. [Amendment 2, I-A / I-B, tightens which entries count: see below.] A deterministic OOM, or any other own failure that recurs
  on one resubmission, makes that SIZE cell NOT DETERMINABLE: never an own failure scored 0 (F87 rule 2's scoring no longer
  applies to this batch), never one arm alone on a different path. Procedure: a run whose job ended without a finished run
  JSON is resubmitted once by hand (check / remove the stale `.claim` first, audit M8; for a ckpt run, the ckpt resumes);
  if the same failure recurs, the coordinator lists the run in runs/ptsz/OWN_FAILURES.txt. Enforcement: select_ptsz.py
  marks BOTH arms of a size with a listed run 'nd' (NOT DETERMINABLE, nothing decided, no X / C / STOP written);
  readout_ptsz.py makes that cell NOT DETERMINABLE (so the primary, and the secondary if the size is 136M / 391M / 731M);
  select / readout also print an **ALERT** for any run whose job (from its run_guard `.claim` file, one sacct call) ended
  without a finished run JSON (status 'dead'; it still counts as waiting), so an OOM at the first in-window eval is not a
  silent 'running'. **Fallbacks** (PYTORCH_CUDA_ALLOC_CONF=expandable_segments, --lm_chunk_tokens 8192) are decided by the
  user and applied to both arms of the size (all its stages), as a generator amendment written here first. Partly
  automatic: --lm_chunk_tokens in one arm alone fails CHECK7 and the integrity re-check (lm_chunk_tokens / lm_chunk are
  compared), so that arm is withheld; expandable_segments is an environment variable the run JSON does not record, so its
  both-arms rule is applied by hand by the coordinator (the generator would set it per size, never per arm).
- **Secondary (the user: "no change").** The SECONDARY rule stays as registered ((a)(b) over 136M / 391M / 731M; its df-1
  interval printed only). Its false-SHOWN (increase) rates under no growth go next to its verdict in FINDINGS: 0.07 / 0.12 /
  0.19 / 0.22 at size scatter tau 0 / 1.85 / 3.12 / 3.41 (this build's simulation), 0.07-0.20 (the audit's independent
  simulation of the same model), 0.08-0.17 (the audit's simulation including the lr selection). readout_ptsz.py prints this
  line under the secondary verdict.
Other audit items:
- I2 (the secondary has no size-level error control): carried, as the secondary decision above (rates next to its verdict;
  calling it "confirmatory" in FINDINGS needs those rates beside it).
- I3 (the primary depends on where a flat gain would sit): carried; "What the verdicts may be called" gains: primary SHOWN
  with secondary NOT SHOWN = only "the new sizes did not fall below the old trend". The read-out prints this line under the
  verdicts. The audit's mu0 scan (0.01 at 24 ... 0.58 at 32) goes with the primary's conditional false-SHOWN rates.
- M1 (lr grids differ between old and new cells; selection bias <= 0.5 points): not acted on (recorded; below tau).
- M2 (stale smoke id / nice): fixed (this section; AUDIT text updated to 22011938 / nice 100).
- M3 ("double path" CHECK7 area vs R.area is not independent): carried: the results audit recomputes every counted run
  from the run JSONs with its own code (as F99's did).
- M4 (relative prior / read-out depth differs by shape; 391M almost as deep as 294M): not acted on (recorded).
- M5 (391M has CPU step-0 references only): carried: a GPU-CPU difference > 1e-3 would withhold every 391M run explicitly;
  the first 391M run's CHECK7 shows it within 30 min.
- M6 (the bpb tie-break effectively picks the lower lr): not acted on (the same rule in every cell, = F87).
- M7 (disk: the 97% line is checked only at submission; failed runs leave 8.8 GB ckpts; ENOSPC truncates run JSONs): carried:
  the coordinator watches myquota during the 731M wave and deletes the ckpt of a run declared failed; the new DEAD alert
  lists such runs.
- M8 (run_guard + slow squeue can make a resubmitted job exit at once): carried: before any resubmission check / remove the
  stale `.claim` and grep the new log for "run guard:".
- M9 (/scratch purge ~60 days, reads do not refresh atime; the reused F87 JSONs date from 10-16 .. 10-22): carried: the
  read-out must run before ~12-15, or the coordinator archives the 64 reused F87 / ptlr_1b and the 32 pt300x reused run JSONs
  (+ check files) first.
- M10 (`sec` restarts on resume): not acted on (cost estimates only).
- M11 (the +-15 INVESTIGATE band is wide): not acted on (it is meant to catch gross failures only).
Tests rerun after the amendment (2026-11-06 ~01:05 EST; outputs in runs/ptsz/tests):
| test | result |
|---|---|
| readout_ptsz.py --ka (readout_ka.out) | KA PASS 24/24 (the 19 before + I1: median +1.0059, trimmed +1.1673, -505M s5 +1.2126 = the range's minimum, mean +2.1608; F99-six median +1.8228) |
| readout_ptsz.py --selftest (readout_selftest.out) | READOUT SELFTEST PASS 31/31 (the 27 before + I1 robustness at the designed slope; the planted late none seed; its leave-one-seed-out / late-count lines; I5 recurrent own failure -> both verdicts NOT DETERMINABLE) |
| select_ptsz.py --selftest (select_selftest.out) | SELECT SELFTEST PASS 26/26 (I5: a listed own failure -> both arms 'nd', nothing written; M6 warnings still printed; a dead OUT_OF_MEMORY job -> 'dead' + ALERT, a RUNNING one -> plain running; the X-job check now expects --nice=100) |
| render --check (render_amend.out) | 128 stage-A files = the rendering; `bash -n` clean; 128 x `--nice=100`, `l40s_public`; time limits unchanged |
| args_diff.py (args_diff.out) | ARGS DIFF PASS (93M vs pt300x: 6 lines, now `--nice` 20 -> 100; ARGS / intent / CHECK7 unchanged) |
| select / readout on the real root (select_presubmit.out, readout_presubmit.out) | every arm waits; both verdicts NOT DETERMINABLE; no ALERT (no claim files yet) |
| regress_frozen.py | REGRESSION PASS (FROZEN 26/26; pttpp 302, ptlr_1b 12, pt300x 84 job files re-rendered byte for byte) |
| submit_ptsz.sh, no --go (submit_dry.out) | refuses without the smoke verdict (22011938 not run yet); with --no-smoke: checks pass (/scratch 94.14%), 128 lines 731M / 391M / 294M / 136M x 32; nothing submitted |
Not rerun (inputs unchanged: ARGS, intent, CHECK7, references): dry_digest, check7_ka, CPU references, sim_null.

Files after the amendment (sha256[:12]): make_jobs_ptsz.py 63002a20b835 (the coordinator's N1 edit), select_ptsz.py
da5833367c67, readout_ptsz.py cf7a9a9d2626, submit_ptsz.sh 38c1f51658fb (N1), tests/smoke_report.py 098a807f3764 (I4);
unchanged: tests/args_diff.py b32001611a4c, check7_ka.py efeeaa26ddbd, dry_digest.py 6aaef417c37c, regress_frozen.py
0c40ac20cb4c, sim_null.py d18a05205c90; ptsz_refs.json 831d6694cd6f; smoke/job_smoke731.sbatch 08cf626a08f8 (says nice
1000; submitted as 22011938 with --nice=100), smoke_seg.py c62eff092e5c; code7 pretrain_range.py a3bce3ab63c1. The
pre-amendment select / readout / AUDIT are kept in scale_work/ptsz/*.pre_amend. Nothing of the batch's training is submitted.

## Amendment 2 after the audit of the amendment (before any result)
The fresh audit of the first amendment (runs/ptsz/tests/audit_amend.md, 2026-11-06; no blocking item) went to the
coordinator, who decided the fixes below. Written 2026-11-06 ~02:00 EST before any ptsz training job was submitted and before
any ptsz result exists. The slope criteria of the PRIMARY and the SECONDARY are unchanged (the audit's pre / post equivalence
test: every decision field identical on 10 synthetic roots and 48 selections).
- **I-0 (the robustness reading rule only compared signs, so it never fired; the audit's own example, the four reused L40S
  cells, passed it).** Now a magnitude rule, printed only, next to the verdicts (primary, secondary, 10-point): the ratios
  median-gain slope / mean slope, trimmed-mean slope / mean slope and leave-one-seed-out extreme toward 0 / mean slope (the
  minimum for a positive mean slope, the maximum for a negative one); **"the slope rests on a few seeds"** when any ratio is
  below 0.5 (none for a zero mean slope). The verdicts are unchanged. No label does not mean robust: in simulation only
  4-23% of false SHOWN carry the label (amendment 3). The read-out also prints the left-out seed's none /
  prior areas against the cell's medians (which arm deviates, audit m3: the label does not say "late none seeds"). Known
  answers (--ka, real runs): the four reused L40S cells give 0.47 / 0.54 / 0.56 -> the label (the audit's example);
  F99's six cells 0.65 / 0.88 / 0.88 -> no label. Self-tests: no late seed -> no label; the planted late seed -> the label in
  all three places; the real four-cell example printed through the read-out's own print path -> the label.
- **I-A (OWN_FAILURES validation; enforced in select_ptsz.own_failures, used by select and the read-out).** An entry is
  `<run> <job id>:<STATE> <job id>:<STATE> <ISO 8601 timestamp with zone>  # comment` and is honoured only if
  - the run is a run of this batch (M6);
  - (a) the run has NO finished run JSON with a passing CHECK7; otherwise the pre-amendment WARNING is printed again ("is
    listed as an own failure but finished with a passing check: entry ignored") and the entry is ignored;
  - (b) the two job ids differ, both states are FAILED or OUT_OF_MEMORY (an infrastructure loss is resubmitted, not listed),
    and, when sacct answers, they equal sacct's states;
  - (c) the timestamp is zoned and before the first read-out of a ptsz result, and the entry was in the file at that
    read-out: **the first readout_ptsz.py call that sees any final new cell writes runs/ptsz/FIRST_READOUT.json** (time,
    final cells, the OWN_FAILURES.txt entries and sha256[:12], the read-out's sha); an entry added later is ignored with a
    WARNING, an entry removed later prints a WARNING (the file is append-only). Every read-out prints the file's sha, the
    number of entries, the honoured ones and the first read-out time.
  Every other entry prints a WARNING and is ignored (the run then simply counts as missing / dead -> the cell NOT FINAL).
- **I-B (stale-claim repeat).** A recurrence requires two DIFFERENT SLURM job ids whose logs
  (logs/llmadd_<run>_<job>.out) both show the trainer's training call (its `code a3bce3ab63c1; eval every` line, printed only
  by the training call, not by --dry_inputs) and no run-guard / HOLD / guard exit ("run guard:", "HOLD present", "not a
  L40S", "code_sha mismatch"); otherwise the entry is ignored with a WARNING. select's ALERT now names the claim's job and its
  state AND the latest job of the run by name (one sacct call over this user's jobs since 2026-11-05) and says whether they are
  the same; a different latest job means it never reached the training call (the claim is rewritten there) -> "NOT a
  recurrence". A resubmission still pending / running -> plain 'running', no ALERT.
- **Minor.**
  - m1 DEAD coverage: a run with no JSON and no claim whose latest job (by name) ended prints a WARNING naming the job and its
    state (it exited before the trainer: HOLD, a sha / GPU / data guard, or the --dry_inputs call); a finished JSON without a
    check file whose job ended prints a WARNING; sacct exiting non-zero, raising or timing out prints a WARNING (never silent).
  - m4 LATE: the absolute bar (none area < 45) is kept; the size-relative count (none area < the cell's none median - 15)
    is printed next to it, both labelled.
  - m5: the secondary's rate line now gives the increase and the decrease rates.
  - m7: a 'dead' run makes its cell NOT FINAL (was PROBLEM).
  - m6 (fallbacks): noted: an --lm_chunk_tokens fallback would also need the integrity re-check and the CHECK7 intent
    amended (they compare lm_chunk_tokens -1 / lm_chunk 0), for both arms, and it puts that size on another loss path than
    the reused cells; the user decides before any such run.
  - m2: done in the amendment section ("I5 replaces F87 rule 2's own-failure-scored-0 for this batch; decided by the
    coordinator, put to the user"; the slope criteria are unchanged, the NOT DETERMINABLE conditions widened).
  - m8 docs: AUDIT's args_diff row (nice) annotated; smoke/SUBMITTED_smoke731 gained a line (21999324 cancelled, 22011938 at
    nice 100; the old line kept). tests/testonly.out still holds the nice-1000 start estimates (not rerun); args_diff's header
    allowance stays generic (submit_ptsz.sh asserts nice 100 exactly).
  - m3 / m6 / m8 otherwise: not acted on beyond the above.
Tests rerun after amendment 2 (2026-11-06 ~02:00 EST; runs/ptsz/tests):
| test | result |
|---|---|
| readout_ptsz.py --ka (readout_ka.out) | KA PASS 27/27 (+ I-0: four reused L40S cells 0.4655 / 0.5402 / 0.5612 -> label; F99 six 0.654 / 0.876 / 0.885 -> no label; late 505M 1 / 1, 1.07B 2 absolute) |
| readout_ptsz.py --selftest (readout_selftest.out) | READOUT SELFTEST PASS 39/39 (+ no-label case; the planted late seed labelled 3x; the real four-cell example labelled through out_robust; a valid own-failure entry -> both verdicts NOT DETERMINABLE; a bare-name entry -> WARNING, the cell only NOT FINAL ('dead' -> NOT FINAL); an entry for a finished, passing run -> the restored WARNING, verdict SHOWN; FIRST_READOUT.json written by the first read-out with a final cell, not by one without; an entry appended after it -> WARNING, ignored) |
| select_ptsz.py --selftest (select_selftest.out) | SELECT SELFTEST PASS 39/39 (+ I-A: bare name, same job id twice, a TIMEOUT state, an unzoned timestamp, a state sacct contradicts -> WARNING, ignored; a valid entry -> both arms nd; the restored WARNING; I-B: a resubmission that exited at the run guard -> ignored; FIRST_READOUT freeze; ALERT: claim = latest -> "the same job", a later FAILED job with a stale claim -> "NOT the claim's job", a PENDING resubmission -> no ALERT; m1: an exit before the claim -> WARNING with job and state; sacct non-zero exit / timeout -> WARNING) |
| select / readout on the real root (select_presubmit.out, readout_presubmit.out) | every arm waits; both verdicts NOT DETERMINABLE; the live sacct answered (674 jobs, 12 s), no WARNING / ALERT; no FIRST_READOUT.json written |
| render --check (render_amend.out) | 128 files = the rendering, all --nice=100 |
| regress_frozen.py | REGRESSION PASS |
| submit_ptsz.sh, no --go (submit_dry.out) | refuses without the smoke verdict; --no-smoke: checks pass (/scratch 94.19%), 128 lines; nothing submitted |
Files after amendment 2 (sha256[:12]): select_ptsz.py 6b5bb0ea3c8f, readout_ptsz.py 5825302449b8, smoke/SUBMITTED_smoke731
e70f16faaca5, SUBMITTED_smoke731 fe097dc147be; unchanged since the first amendment: make_jobs_ptsz.py 63002a20b835,
submit_ptsz.sh 38c1f51658fb, tests/smoke_report.py 098a807f3764, ptsz_refs.json 831d6694cd6f. Previous versions:
scale_work/ptsz/*.amend1 (after amendment 1) and *.pre_amend. Nothing of the batch's training is submitted.

## Amendment 3 after the audit of amendment 2 (before any result)
The fresh audit of amendment 2 (runs/ptsz/tests/audit_amend2.md, 2026-11-06; no blocking item; the verdicts and every number
they use are unchanged across the three versions) went to the coordinator, who asked for these fixes. Written 2026-11-06
~03:30 EST before any ptsz training job was submitted and before any ptsz result exists (no runs/ptsz/out, no
FIRST_READOUT.json, readout.json or readout.txt). The slope criteria are unchanged.
- **R1 (important): `readout_ptsz.py --json` crashed on the first complete read-out** (the leave-one-seed-out dict had tuple
  keys; json.dumps never applies default= to keys). Fixed: robust()['loso'] is keyed 'cid|sN'; the extreme / min / max stay
  (value, (cid, seed)) values (lists in JSON). Tests: the full CLI path `readout_ptsz.py --root R --json OUT` on a COMPLETE
  synthetic root (reused cells = FINDINGS F99 so main()'s FINDINGS check passes; primary, secondary and 10-point all final)
  runs, writes the JSON and reloads it; a recursive walk finds no non-string dict key anywhere in the read-out's result.
  select's picks.json / decisions.json carry string keys only (lr strings, size / arm names; the per-run records are dropped
  before writing) and are written by its selftest's --write runs.
- **dead misjudgement**: the claim's own job is consulted first: while it is still alive (pending / running / requeued), the
  run is 'running' whatever a later duplicate job did (e.g. a resubmission that exited at the run guard and then FAILED); no
  ALERT. A claim job unknown to sacct prints a WARNING. With no claim file the ALERT says "no claim file to compare".
- **negative slope**: the seed whose none / prior areas are printed is the one giving the leave-one-seed-out extreme TOWARD 0
  (the maximum for a negative mean slope, the minimum for a positive one), the same extreme the I-0 ratio uses; tested with a
  planted seed in a negative-slope root.
- **OWN_FAILURES consistency** (select_ptsz.own_failures):
  - a run with a finished run JSON, also one whose CHECK7 failed or that has no check file, is not "an own failure without a
    finished run" (I5): the entry is refused with a WARNING (a failed check withholds the arm by rule 6, the coordinator
    investigates);
  - an entry whose timestamp is later than the current read-out / selection time is refused;
  - FIRST_READOUT.json: written atomically (tmp + rename) and made read-only (mode 444); a file that cannot be parsed makes
    the read-out refuse (RuntimeError -> `REFUSED:` and exit 2) and select honour no entry (WARNING); if it is missing while
    runs/ptsz/readout.json or readout.txt already shows a final new cell, the read-out refuses loudly instead of writing a
    new one (an empty readout.txt, i.e. the shell redirect of the first read-out itself, does not count);
  - an honoured entry prints a NOTE with each job's last error line (so the coordinator can see it is the same failure; audit
    amend2 m5);
  - the once-per-run warning memory is reset by each select / read-out run.
- **Recorded, not changed**: once FIRST_READOUT.json exists, a true recurrence that happens later cannot be honoured; that
  cell stays NOT FINAL, i.e. NOT DETERMINABLE either way (audit amend2 m2 ii). The label is read for SHOWN verdicts only, and
  "no label" is not evidence of robustness (only 4-23% of false SHOWN carry it; m6). select's docstring now describes the
  entry format and the sacct lookup (m8); the no-op M6 loop in select_arm is removed; the read-out's docstring says it is not
  read-only (FIRST_READOUT.json).
Tests rerun after amendment 3 (2026-11-06 ~03:25 EST; runs/ptsz/tests):
| test | result |
|---|---|
| readout_ptsz.py --ka (readout_ka.out) | KA PASS 27/27 (the leave-505M-s5 lookup now by the key '505m|s5') |
| readout_ptsz.py --selftest (readout_selftest.out) | READOUT SELFTEST PASS 46/46 (+ R1 CLI --json on a complete root, reloaded; no non-string key anywhere; FIRST_READOUT.json mode 444; refusal when readout.txt shows a final new cell and FIRST_READOUT.json is missing; an empty readout.txt does not block; refusal on an unparsable FIRST_READOUT.json; negative slope -> the extreme toward 0 is the maximum and its seed's arms are printed) |
| select_ptsz.py --selftest (select_selftest.out) | SELECT SELFTEST PASS 45/45 (+ the NOTE with the last error lines; an unparsable FIRST_READOUT.json -> no entry honoured; a future timestamp -> ignored; an entry for a finished run whose CHECK7 failed -> WARNING, not honoured; the claim's job RUNNING with a later duplicate FAILED -> running, no ALERT; a claim job unknown to sacct -> WARNING). One test-only issue on the way: the NOTE was deduplicated across runs; the warning memory is now reset per run |
| select / readout on the real root, readout with --json (select_presubmit.out, readout_presubmit.out) | every arm waits; both verdicts NOT DETERMINABLE; no WARNING / ALERT; the --json path ran; no FIRST_READOUT.json written |
| render --check (render_amend.out) | 128 files = the rendering, all --nice=100 |
| regress_frozen.py | REGRESSION PASS |
| submit_ptsz.sh, no --go (submit_dry.out) | refuses without the smoke verdict; --no-smoke: checks pass (/scratch 94.43%), 128 lines; nothing submitted |
Files after amendment 3 (sha256[:12]): select_ptsz.py d00fd67f042f, readout_ptsz.py cc44adc8329d; unchanged:
make_jobs_ptsz.py 63002a20b835, submit_ptsz.sh 38c1f51658fb, tests/smoke_report.py 098a807f3764, ptsz_refs.json
831d6694cd6f, smoke/SUBMITTED_smoke731 e70f16faaca5, SUBMITTED_smoke731 fe097dc147be. Previous versions:
scale_work/ptsz/*.amend2, *.amend1, *.pre_amend. Nothing of the batch's training is submitted.

Coordinator (2026-11-06): amendment 3 is not sent to another fresh audit. It fixes audit_amend2.md R1 and minors; the coordinator
checked the R1 fix (readout_ptsz.py:376 converts the loso keys to 'cid|sN'; selftest 'R1: the CLI --json path on a complete root
... reloads' PASS, READOUT SELFTEST 46/46). Submission still waits for the smoke 22011938 verdict (submit_ptsz.sh enforces it).

## Amendment 4: the user's decision on the 731M memory margin (2026-11-06 ~16:30 EST, before any ptsz training job and before the XS smoke ran)
**Trigger.** The smoke 22011938 ran every segment without error or OOM, but failed the margin criterion from the first
amendment (I4): `731m reserved <= total - 2 GiB`. Segment B, the resumed in-window eval, reserved 46.83 GB of 47.70 GB
(allocated 40.30 GB). A reserved 43.85 GB, C 42.60 GB and D 42.60 GB were within the margin. submit_ptsz.sh therefore
refused to submit. The coordinator put the choice to the user: (A, recommended) expandable segments on both 731M arms plus a
short resume smoke before submission; (B) submit unchanged, so that a deterministic OOM on a resumed run makes 731M NOT
DETERMINABLE under I5. The loss-chunking fallback was named as a larger deviation and was not recommended.
**User decision (2026-11-06, verbatim):** "ok:1. 731M 显存：建议加 expandable_segments、补一个从断点续跑的短测试再提交；也可以照原样提交，代价是续跑时一旦 OOM，按规则 731M 就记为「无法判定」。" This is option A.
**Changes** (make_jobs_ptsz.py 63002a20b835 -> abee61db9ef3; the previous copy is scale_work/ptsz/make_jobs_ptsz.py.amend3):
- XS_SIZES = ("731m",). Every 731M job (both arms; stages A, X and C, since they render through the same job_x) exports
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the trainer calls and echoes it ("ALLOC: ..." in the log).
  This changes only how the CUDA caching allocator maps memory. The trainer, its arguments, the data, CHECK7 and the intent
  JSON are unchanged, and the run JSON does not record the setting; the job log does. The other sizes, and the reused
  pt300x cells, run without it. The setting is uniform within a size, so it is the same for both arms of every comparison.
- TRAIN_S['731m'] 31581 -> 33177 s: the smoke's measurement (smoke_report_22011938.json est.win_s; 9.33 h per run with start-up
  and 9 ckpt writes). The first smoke section already asked for this update before the 731M jobs are submitted. The 731M time
  limit goes from 22:30 to 23:30 under the same 2.5x rule (< gpu48 MaxWall 48 h); ckpt_every stays 2000.
- `--smoke-xs` renders runs/ptsz/smoke/job_smoke731xs.sbatch (80764f3ea66e): job llmadd_psz-smoke731xs, 1 h, nice 100,
  L40S, the first smoke's header (HOLD, trainer sha, GPU guard, data sha) plus the export line. Its segments are the first
  smoke's lines, asserted, with only the name and out dir changed. XA = A (731m win-ctx 6e-4 s0, --window 0,0.4
  --eval_every 100 --ckpt_every 100 --stop_after 210). XB = B (the same command to 310; it resumes XA's step-199 ckpt). XC =
  C (the real none 2.4e-3 ARGS, --stop_after 260 instead of 500). smoke_seg.py (c62eff092e5c) is unchanged. These are test
  runs in runs/ptsz/smoke/XA and XC and never enter a selection or read-out.
**XS smoke criteria (tests/smoke_report_xs.py 5e1f993f90f1; registered here before the job runs).** PASS iff all of:
- (1) the log shows `ALLOC: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- (2) XA, XB and XC ran without error, and XB resumed ('resumed from' in XB's part of the log);
- (3) max_reserved <= total - 2 GiB in XA, XB and XC;
- (4) the setting takes effect: the reserved - allocated gap of XA is below A's in 22011938, and XB's below B's (same
  commands without the setting);
- (5) the first smoke's non-memory criteria stay true (A / B / C D E ok, resumed, step-0 lm within 1e-3).

The report writes smoke_report_<jid>.json and smoke/SMOKE_VERDICT.json, the file submit_ptsz.sh reads; the first smoke's
record stays in smoke_report_22011938.json. Speed under the setting is printed (XA and XB wall time against A and B) but is
not a criterion.
**If the XS smoke fails** nothing is submitted, and the choice goes back to the user (B, or the loss-chunking fallback on
both 731M arms).
**Tests (before submission):**
| test | result |
|---|---|
| `make_jobs_ptsz.py --stage A --size 731m` then `--check --stage A` | the 32 731M files re-rendered; 128 files = the rendering |
| byte diff against the pre-amendment copies (scale_work/ptsz/jobs_all_amend3/) | the 96 136M / 294M / 391M files byte-identical; each 731M file differs only in `#SBATCH --time` (22:30 -> 23:30) and the two added lines (export, echo) |
| XS smoke rendering against the first smoke | header: job name, time, nice 100 (the first file kept its 1000 from before N1, its sbatch ran with --nice=100), two comment lines and the export/echo lines; segments: only the segment name, the out dir (A -> XA, C -> XC) and XC's --stop_after 500 -> 260 |
| tests/regress_frozen.py | REGRESSION PASS |
| tests/args_diff.py | output identical to tests/args_diff.out |
| tests/smoke_report_xs_test.py (smoke_report_xs_test.out) | XS REPORT TEST PASS: cases built from the real 22011938 log: unchanged numbers -> FAIL on (3) and (4) only; reserved = alloc + 0.5 GiB -> PASS; no ALLOC line -> FAIL on (1) only; no resume line -> FAIL on (2) only; the real SMOKE_VERDICT.json untouched |
| `make_jobs_ptsz.py --smoke-xs --test-only` | accepted (backfill estimate 2026-11-09 06:10, pessimistic) |
| submit_ptsz.sh (no --go) | refuses: SMOKE_VERDICT.json is still 22011938's FAIL |
torch 2.10.0+cu128 (open_r1) supports expandable segments; criterion (4) checks that the setting takes effect.
**Audit:** experiment-audit, minimal mode (asserts and tests above). A fresh adversarial audit of this amendment starts after
the XS smoke is submitted; its blocking / important items are checked before the 128 stage-A jobs are submitted.

## Amendment 4.1: the fresh audit of amendment 4 (tests/audit_amend4.md, 2026-11-06), fixed before the XS smoke ran
The audit found no blocking item. It found one important item and seven minor ones, and confirmed four things: the setting
reaches only the 32 731M jobs and both trainer calls (re-exported on requeue); torch 2.10 reads PYTORCH_CUDA_ALLOC_CONF (an
invalid value raises on CPU, and it takes precedence over PYTORCH_ALLOC_CONF); XA / XB / XC equal 22011938's A / B / C up to
--out and XC's --stop_after; the time limit is computed correctly and submit_ptsz.sh reads the new verdict. Written
2026-11-06 ~18:00 EST; XS smoke 22129757 still PENDING (QOSMaxGRESPerUser: the user's 16-GPU cap). No ptsz training job is
submitted.
- **R1 (important), fixed.** Criterion (5) read only 22011938's JSON (a constant), so "numerics unchanged" was never tested
  on the GPU. The coordinator checked smoke_report_xs.py:68 (a4 copy) and confirmed it. New (5): XA's step-0 lm (log) and
  XC's (run JSON) within 2e-4 (the pttpp calibration tolerance) of 22011938's 11.74520206451416 (A = C = D, bit for bit;
  bitwise equality printed); XB's step-200 lm = XA's step-200 lm within 2e-4 (22011938: bit for bit, 6.374513626098633);
  XA@100 / XA@200 / XC@250 within 0.5 nat of 22011938's A / A / C (ptheld Amendment DP's later-eval tolerance; printed). The
  old constant criterion becomes (6) and covers only what is not rerun (C D E ok, resume, step-0).
- **m1, fixed.** (4) needs at least a halving of the reserved - allocated gap (XA <= 0.5 x A's 6.03 GiB, XB <= 0.5 x B's
  6.08 GiB), no longer just "below".
- **m2, fixed.** (2) also needs no 'run guard' line, every segment's max_allocated >= 0.8 x the same segment of 22011938 (the
  731m computation really ran), and XB's 'resumed from ... at step 200'.
- **m3, coordinator step (no code).** Before the first select_ptsz.py run that uses a 731M cell, the coordinator checks that
  every 731M stage log contains "ALLOC: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" (grep over logs/llmadd_psz-731m-*);
  a log without it is investigated before use.
- **m4, recorded.** submit_ptsz.sh checks only pass = true, not the job id or the amendment. The verdict now carries
  "amendment": "4.1"; the coordinator checks it by eye before `--go`. submit_ptsz.sh is unchanged (38c1f51658fb).
- **m5, recorded.** Cost at the measured 731M time: 9.33 h per run -> 731M stage A 298.6 L40S-h, A + C ~373 h, batch total
  ~823 L40S-h (was 805). The Deviations list gains amendment 4 (731M allocator setting).
- **m6, recorded.** smoke/A and smoke/XA keep ~17 GB of test ckpts (/scratch ~94%); delete after the XS verdict only with the
  user's OK.
- **m7.** The XS smoke waits on QOSMaxGRESPerUser (the user's 16-GPU cap shared with ptdom / pt2sz / ptshare), not on L40S.
Tests: tests/smoke_report_xs_test.py (546169b51cdb) -> smoke_report_xs_test.out: **XS REPORT TEST PASS 10/10**. same ->
FAIL (3), (4); good -> PASS; noalloc -> (1); noresume, runguard, tiny XB -> (2); gapweak (gap 0.6 x) -> (4) only;
lm0shift (+0.5, the audit's case), resumeoff (+1e-3), no XC run JSON -> (5) only. The real SMOKE_VERDICT.json is untouched.
Files: tests/smoke_report_xs.py 5e1f993f90f1 -> **f92e98ba92dc** (previous: scale_work/ptsz/smoke_report_xs.py.a4);
make_jobs_ptsz.py abee61db9ef3, job_smoke731xs.sbatch 80764f3ea66e, smoke_seg.py c62eff092e5c, every job file unchanged.
Coordinator: 4.1 applies the audit's own findings to the report and adds test cases. It is not sent to another audit.

## XS smoke result (job 22129757, gl037, 2026-11-07 14:54-15:17 EST; tests/smoke_report_xs.py f92e98ba92dc -> tests/smoke_report_xs_22129757.out, smoke/SMOKE_VERDICT.json)
**FAIL by the registered criteria. Nothing is submitted; the choice goes back to the user (Amendment 4).**

Criteria met:
- (1) setting on;
- (2) XA / XB / XC ran, no run guard; XB resumed at step 200;
- (5) numerics unchanged: step-0 lm of XA and XC = 11.74520206451416 bit for bit; XB@200 = XA@200 bit for bit; later evals
  within 7.2e-3 of 22011938;
- (6) the first smoke's criteria.

Criteria failed:
- (3) XB reserved 43.86 GiB, 0.56 GiB headroom against the 2 GiB margin. XA 42.36 GiB and XC 42.32 GiB are within it.
- (4) the gap did not shrink, it grew: XA 7.57 GiB against A's 6.02; XB 6.35 against 6.08; XC 9.37 against C's 6.69.
  The setting raises peak reserved memory here.

Peak allocated is unchanged by the setting: XA 34.79, XB 37.51, XC 32.94 GiB, against 34.81 / 37.53 / 32.98 GiB in
22011938, of a 44.42 GiB card. nvidia-smi max 45,449 of 46,068 MiB (22011938: 45,193). No OOM in either smoke.

Reading: expandable segments does not fix the margin. In both smokes only the resumed in-window eval misses it, and resume
happens only after a requeue: l40s_public has PreemptMode OFF, so only a node failure or a manual requeue triggers it.

The XS job queued ~21 h on QOSMaxGRESPerUser (the user's 16-GPU cap).

## Amendment 5: the user's decision after the failed XS smoke (2026-11-08 ~12:40 EST, before any ptsz training job)
**User decision (2026-11-08):** "可以的，没问题：…都按你所推荐的做先". The recommendation put to the user on 2026-11-07
after the XS smoke failed: option A, i.e. remove the setting, submit the 731M jobs unchanged with the measured time limit, and
register now that a 731M run which OOMs after a resume is re-run from step 0, the same rule for both arms. Option B (loss
chunking + another smoke) and option C (drop 731M) were not chosen.
**Changes** (make_jobs_ptsz.py abee61db9ef3 -> 464d2f5b53c6): `XS_SIZES = ()`. The 32 731M job files are re-rendered.
Against the pre-amendment-4 copies (scale_work/ptsz/jobs_all_amend3/) they differ only in `#SBATCH --time` (22:30 ->
23:30, TRAIN_S['731m'] = the smoke's measurement, amendment 4); no job file has the export line; the 96 other files are
unchanged. `--check --stage A`: 128 files = the rendering. Amendment 4's smoke-xs rendering and report stay as records.
**Submission:** `submit_ptsz.sh --no-smoke --go`. --no-smoke is the documented explicit acceptance that the 731M L40S
memory fit on the resumed in-window eval is not verified (both smokes: the fresh-run segments meet the 2 GiB margin, the
resumed one misses it by ~1.4 GiB; no OOM in either; peak allocated 37.5 of 44.4 GiB).
**Resume-OOM contingency (pre-registered here, before any ptsz training job; both arms, every stage of 731M):**
- Trigger: a 731M job that ends FAILED / OUT_OF_MEMORY with a CUDA OutOfMemoryError in its log after a
  'resumed from ... at step N' line of the same attempt (i.e. the failure is on the resumed path).
- Action (the coordinator, by hand): move that run's ckpt file(s) out of its out dir into runs/ptsz/ckpt_quarantine/<run>/
  (kept, not deleted), remove the stale `.claim` (audit M8), resubmit the same job file once. The run then starts from step 0
  with unchanged arguments, data and seed. This is the one I5 resubmission of that run.
- If that from-scratch attempt fails with an OOM (or any failure recurs), I5 applies unchanged: the run goes to
  OWN_FAILURES.txt and the 731M cell is NOT DETERMINABLE.
- The re-run counts like any run; nothing else changes (no flag, no allocator setting, no other arm treated differently).
- An OOM without a preceding resume line is not this trigger. I5 applies to it directly (resubmit once; recurrence -> OWN_FAILURES).
**m3 of amendment 4.1 is void** (no ALLOC line any more). **Audit:** no fresh audit. Amendment 5 returns the 731M files to
their audited pre-amendment-4 content, except the one constant amendment 4's audit checked (TRAIN_S / the time limit),
and adds a by-hand procedure that changes no code.
- Submission (amendment 5): `submit_ptsz.sh --no-smoke --go` ran from ~12:55 to 20:01 EST on 2026-11-08 (slurmctld was down
  for parts of it; sbatch calls hung). Exit 0. 128 SUBMITTED_ flags; 128 llmadd_psz- jobs queued, 32 per size, 731M first
  (log: runs/ptsz/submit_go_20261108.log; first job 22346557 area, last 22347418). /scratch 95.77% at the check.

## Incident: 731M none seed 2 OOM on the fresh path (2026-11-09 ~00:35 EST)
- All four 731M none s2 jobs failed within ~2.5 min: 22346531 (1.2e-3), 22346535 (2.4e-3), 22346539 (4.8e-3) and the 9.6e-3
  one, on nodes gl065 / gl067 / gl031. Each hit torch.OutOfMemoryError in the first training backward right after the
  step-0 eval: "Tried to allocate 6.12 GiB ... 25.95 GiB allocated, 12.27 GiB reserved but unallocated". This is
  fragmentation, and it is deterministic for seed 2. No resume line, so it is NOT the amendment-5 trigger; I5 applies.
- 731M none s0 / s1 / s3 at all four lrs are running (11 jobs). The 16 win-ctx jobs are pending, behind the user's 16-GPU cap.
- scontrol hold on the pending 731M jobs failed ("Unspecified error"). Nothing was cancelled.
- Diagnostic smoke submitted, before any decision, not a result: smoke/job_smoke731s2.sbatch = job_smoke731xs.sbatch with
  --seed 2 and out dirs S2A / S2C (expandable_segments on). Job: Submitted batch job 22378673. It tests whether the setting
  removes the seed-2 failure (S2C = the failing none 2.4e-3 s2 path) and the in-window / resume path at seed 2.
- The I5 fallback is the user's decision (both arms, all stages of 731M).

## Amendment 6: the user's decision on the seed-2 OOM (2026-11-09 ~01:10 EST, before any ptsz result is read)
**User decision (verbatim):** "ok：- A（我推荐）：现在先停掉 731M 的全部 28 个作业（正在跑的才跑了 10～25 分钟，损失约 3 个 GPU 小时）。等诊断测试确认 expandable_segments 能消除这个 OOM、数值不变，就给 731M 的全部 32 个 run 开这个设置后重交，两组、所有 seed 用同一设置。"
**Steps:**
1. Every queued or running 731M job is cancelled. The IDs are in runs/ptsz/cancel_731m_ids.txt; the controller was slow, so
   the cancels were retried one by one. No 731M run had reached its first ckpt or finished.
2. Diagnostic smoke 22378673 (smoke/job_smoke731s2.sbatch c6b894bd4e5a) is read by tests/smoke_report_s2.py (188240296866).
   The criteria are registered here before the job runs. PASS iff all of:
   - (1) the setting is on;
   - (2) S2A / S2B / S2C ran, with no OOM and no run guard;
   - (3) S2B resumed at step 200, and its step-200 lm equals S2A's within 2e-4;
   - (4) the step-0 lm of S2A and S2C is within 2e-4 of the CPU reference lm0_ref('731m', 2) = 11.737274169921875;
   - (5) S2C reached step 260.
   The memory margin is printed but is not a criterion (the user's decision is the setting for all 731M runs).
3. If PASS:
   - set `XS_SIZES = ("731m",)` (amendment 4's rendering: export plus an ALLOC echo line, time limit 23:30);
   - move the superseded 731M files (job_psz-731m-*.sbatch, SUBMITTED_psz-731m-*, out/psz-731m-*) into
     runs/ptsz/superseded_a6/, kept and not deleted;
   - re-render the 32 731M stage-A files and check them against the amendment-4 rendering (scale_work/ptsz/jobs731_amend3
     plus the two lines and the time limit);
   - submit them with `make_jobs_ptsz.py --stage A --size 731m --submit`. submit_ptsz.sh cannot be used: it requires none of
     the 128 to be submitted.

   The 96 jobs of the other sizes are untouched. The amendment-4.1 m3 step (grep the ALLOC line in every 731M log before its
   selection) applies again. The amendment-5 resume-OOM contingency stays in force, with the setting on.
4. If FAIL: nothing is resubmitted, and the choice goes back to the user.
**Numerics:** the XS smoke 22129757 showed the setting leaves step-0 lm and the resume bit-identical at seed 0. (4) and (3)
check the same at seed 2.

## Amendment 7: submit 731M with the setting in parallel with the seed-2 smoke (the user, 2026-11-10 ~15:40 EST; before any 731M result)
**User decision (verbatim):** "这个可以：1. 731M 现在就提交，和短测试同时跑：开着 expandable_segments 直接把 32 个 run 交上去，不再等短测试排到。…" Reason: the diagnostic smoke 22378673 had been pending ~38 h on l40s_public.
**Changes:**
- make_jobs_ptsz.py 464d2f5b53c6 -> 0a360a50e19f: XS_SIZES = ("731m",). Previous copy: scale_work/ptsz/make_jobs_ptsz.py.amend6.
- The amendment-5 / 6 731M files are moved to runs/ptsz/superseded_a6/ and kept: 32 job files, 32 SUBMITTED_ flags, 4 out
  dirs from the cancelled / failed attempts. Those dirs hold no ckpt and no finished run.
- The 32 731M stage-A files are re-rendered. Against scale_work/ptsz/jobs731_amend3/ they differ only in the time limit
  (23:30) and the export / echo lines; every 731M file has the export line and no 136M file does. `--check --stage A`:
  128 files = the rendering.
- Submitted with `make_jobs_ptsz.py --stage A --size 731m --submit` (log submit_731m_a7_20261110.log).
**If the smoke 22378673 FAILS** (Amendment 6 criteria): the 731M jobs are cancelled (by the user if the agent's scancel is
refused), and the choice goes back to the user. If it passes, nothing changes. The amendment-4.1 m3 check (the ALLOC line in
every 731M log) and the amendment-5 resume-OOM contingency apply.

## pt2sz deadline (the user, 2026-11-10): "2. 给 pt2sz 定一个截止日：比如 11-13 还没开跑，论文的 2% 尺寸部分就只用已有的 3 个尺寸（F98），不再等它。"
