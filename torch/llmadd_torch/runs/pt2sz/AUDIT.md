# pt2sz — two more sizes for the 2% size comparison on one cluster (93M, 294M; 2026-11-05)

User decision 2026-11-05 (relayed by the coordinator; LLM_addition_review/notes/reply_manifolds_sizes_2026-11-05.md, the
writing session's request part P2): 2% arithmetic at a fixed 3e8 tokens on Torch A100, sizes 93M and 294M, no 1.07B; each
arm picks its own lr, then confirm seeds, 8 seeds; priority ptdom > new sizes > ptshare, so every job gets `--nice=1000`.
Written BEFORE any job of this batch was submitted: only CPU work ran (the references, tests/). Generator
llmadd_torch/make_jobs_pt2sz.py; the generator never submits by itself (runs/pt2sz/submit_pt2sz.sh --go, by the coordinator).

## Question
FINDINGS F98 (pt2same; Torch A100-80GB, trainer 93e22bf4c4c0, 3e8 tokens, 2%, each arm at a fixed pick): the prior's area
gain is +48.0 / +49.9 / +53.0 at 53M / 190M / 505M (8/8 each), pre-registered verdict "SHOWN (+)", slope +1.44 per e-fold
of N_nonemb [+0.22, +2.64] (seed error only) but marginal (per-seed slopes t [-0.05, +2.92], p .11; driven by
Delta505 - Delta53). **Q:** with two more sizes on the same cluster, GPU type, trainer and job body (five points: 53M,
93M, 190M, 294M, 505M), does the 8-seed paired area gain (win-ctx - none, readout_rules.area) increase with ln N_nonemb at
a fixed 3e8 tokens?
- **Already observed when this was registered:** 3 of the 5 points (F98's cells, with their slope, halves, per-seed
  slopes). The test is only partly confirmatory: the two new sizes and criterion (c) are the new evidence; the simulated
  false-SHOWN rate below says how much the new points can change.

## Design (make_jobs_pt2sz.py is the one source; `--plan` prints the table)
| size | shape | n_params | N_nonemb | ln N | tpp at 3e8 | prior / read-out layers | grad ckpt | source | picks (none / win-ctx) |
|---|---|---|---|---|---|---|---|---|---|
| 53M | d384 x 6, 6 heads | 52,672,128 | 14,165,376 | 16.466 | 21.18 | 2, 4 | off | pt2same (reused) | 8e-3 / 2e-3 |
| **93M** | **d512 x 10, 8 heads, d_ff 2048** | 93,306,368 | 41,964,032 | 17.552 | 7.149 | 3, 6 | off | **new** | stage A below |
| 190M | d768 x 12, 12 heads | 190,297,344 | 113,283,840 | 18.545 | 2.648 | 4, 8 | on | pt2same (reused) | 4.8e-3 / 6e-4 |
| **294M** | **d832 x 19, 13 heads, d_ff 3328** | 293,931,456 | 210,500,160 | 19.165 | 1.425 | 6, 12 | on | **new** | stage A below |
| 505M | d1024 x 24, 16 heads | 505,437,184 | 402,752,512 | 19.814 | 0.745 | 8, 16 | on | pt2s505 (reused) | 4.8e-3 / 6e-4 |

Shapes, n_params and N_nonemb = make_jobs_pttpp.SIZES (the pttpp / pt300x shapes; the CPU dry runs print "model: 93.3M
parameters; gradient checkpointing off" / "model: 293.9M parameters; gradient checkpointing on"). The reused cells are F98's
exactly (readout_pt2same.run_paths: pt2same 53M / 190M, pt2s505 505M; 8 seeds each, Torch A100-SXM4-80GB).

**Protocol = pt2same's (= pt2s505's = F86's pt2main) exactly, except the shape, the lr grid and the stages.** Arms
(make_pt2main_tasks.ARM): none = `--target none`; win-ctx = `--target helix3 --schedule window --window 0.05,0.4 --site ctx`
(lambda 5 = the trainer default). `--arith_lines 1` (2%), 3e8 tokens = 18,310 steps of 32 x 512 on pretrain_data (the
trainer default: no --tokens / --eval_every flag, as pt2same / pt2s505), eval every 250 steps (75 evals), warmup 183,
window steps 916..7323 (prior_events [[916, 1], [7324, 0]], on_frac 6408 / 18310), `--save_model --ckpt_every 2000`; every
other flag = the trainer default (asserted). Seed s = the same data stream (rng 2000+s / 1000+s) and held-out split (rng s)
as seed s of every reused cell (the init differs by shape). Model flags `--d_model --n_layers --n_heads --d_ff` as pt2same's
190M and pt2s505's 505M (pt2same's 53M passes none: the trainer default shape).

**Code.** Trainer: runs/pt2sz/code = `git archive 7f1d06b tools/` (make_jobs_pttpp._extract), asserted equal file by file
to runs/pt2s505/code and runs/pt2same/code, COMMIT verbatim, chmod a-w; pretrain_range.py 93e22bf4c4c0 and every module it
imports (WANT_SHA) re-checked in every job. Job body = make_jobs_pt2same.render's body (pt2s505's + the A100-80GB guard +
the INTENT line); the generator imports make_jobs_pt2same / pttpp / pt2s505 / rb6scale read-only (sha256[:12] asserted,
the import runs no main(), asserted by ast). Its ARGS / intent builders reproduce the SUBMITTED pt2s505 (505M) and pt2same
(53M, 190M) jobs exactly (`known_answers()`, run by every rendering): make_jobs_pt2s505.render's whole job file, its ARGS
and intent; make_jobs_pt2same.intent and the ARGS lines of its job files. The intent = make_jobs_pttpp.intent2 (frozen) with
the 3e8 facts.

## Differences from pt2same's jobs (decided before any result; tests/args_diff.out, tests/render_check.out)
1. Shape (the pttpp shapes) and the lr grid / stages (below).
2. `--nice=1000` (the user, 2026-11-05: ptdom (B200, nice 20) > new sizes > ptshare (nice 2000)); pt2same had 20.
3. Job names `llmadd_p2z-<run>` (run = `p2z-<size>-lr<lr>-<arm>-s<seed>`), out dirs runs/pt2sz/out/<run>/<tag>.
4. **Stage A / X models are not kept** (pttpp job2's rule): offline_lm.py has used the model and the checks have passed, so
   `rm -f` of the model file runs only after a passing check (RC 0); stage C (seeds 4-7) keeps its model. The read-out uses
   only run JSONs and check.json (area, in-training bpb; offline bpb is in check.json). To do this the CHECK_PY call
   captures its rc (`... | tee "$OUT/check.txt" || RC=$?`, pttpp job2's form) and the job ends `exit $RC`: a failed check
   still exits 4, as in pt2same (where `set -e` exited at the check).
5. Time limits 93M 03:00:00, 294M 08:00:00 (2.5 x the 1.07 / 3.09 h plan, rounded up to the half hour: pt2same's rule).
6. The INTENT line names the stage and the model rule instead of a Burst twin (there is none).
Argument diffs (emulated run args, tests/args_diff.out): vs pt2same 190M and pt2s505 505M at the same lr / seed: only
d_model, n_layers, n_heads, d_ff, out, tag; vs pt2same 53M: + lr; vs pttpp tpp2-294m (the same shape at 156.8M tokens):
tokens, eval_every (the 3e8 defaults here), ckpt_every (2000 here as pt2same; pttpp's 2% jobs had none), out, tag.

## Hardware
Torch a100_tandon (every node NVIDIA A100-SXM4-80GB), account torch_pr_520_tandon_priority, qos gpu48, 1 GPU, 8 CPUs, 64G,
`--requeue` (TIMEOUT is not requeued, preemption is and resumes from the 2000-step ckpt), `--nice=1000`; the GPU guard
`*A100*80GB*` exits 3 on anything else (A100-40GB, L40S, H100, ...): one GPU type for the whole batch and for every
reused cell (F98's 48 runs: all A100-SXM4-80GB, re-checked by the read-out from node.txt).

## lr grids (four consecutive rungs of the factor-2 ladder per arm; asserted in the generator)
Interpolation at 3e8 tokens, log-linear in ln N_nonemb between the neighbouring sizes' 2% 3e8 picks (pt300x's rule; the
picks checked against their sources by the generator: 53M / 190M = pt2same's PICKS = F86's, 505M = pt2s505 picks_confirm.json,
all = make_jobs_pttpp.OPT3E8['2']):
| cell | neighbours' 2% 3e8 picks | interpolated | grid | rungs below / above the interpolation | X if open |
|---|---|---|---|---|---|
| 93M none | 53M 8e-3, 190M 4.8e-3 (f = 0.522) | 6.13e-3 | 2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2 | 2 / 2 | 1.2e-3 / 3.84e-2 |
| 93M win-ctx | 53M 2e-3, 190M 6e-4 | 1.07e-3 | 3e-4, 6e-4, 1.2e-3, 2.4e-3 | 2 / 2 | 1.5e-4 / 4.8e-3 |
| 294M none | 190M 4.8e-3, 505M 4.8e-3 (f = 0.489) | 4.8e-3 (on a rung) | 2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2 | 1 / 2 | 1.2e-3 / 3.84e-2 |
| 294M win-ctx | 190M 6e-4, 505M 6e-4 | 6e-4 (on a rung) | 1.5e-4, 3e-4, 6e-4, 1.2e-3 | 2 / 1 | 7.5e-5 / 2.4e-3 |
Why these grids (the generator asserts: 4 consecutive rungs, at least one rung strictly below and one strictly above the
interpolated value): where the interpolation falls between two rungs (93M), two rungs on each side. Where it falls on a rung
(294M: both neighbours share the pick), the fourth rung goes to the side where the neighbours' area argmax lay at 2% 3e8:
- none: at 190M (Burst, F86) and 505M (pt2s505) the area argmax was 9.6e-3, one rung ABOVE the pick (the pick came from the
  bpb tie-break: 190M 28.78 vs 26.84, 505M 25.6 vs 24.7, seeds 0-3), and 505M fell at 1.92e-2 (15.5): two rungs above;
- win-ctx: at 190M (6e-4 78.56 > 1.2e-3 76.91 > 2.4e-3 72.86) and 505M (3e-4 78.1 > 6e-4 76.7 > 1.2e-3 75.2) the area fell
  monotonically with lr and the argmax sat at the LOWER grid edge (a note, not open): two rungs below; the pttpp 294M
  win-ctx pick at the 0.745 line (fewer tokens: a higher or equal lr) was 6e-4 after an X to 3e-4.
The ladder gets one rung below make_jobs_pttpp.LADDER (7.5e-5) so that an X below 294M win-ctx's 1.5e-4 edge can be rendered.
Note: the 53M picks (8e-3 / 2e-3) are off this ladder (F86's pt2b grid); the new grids are on pttpp's ladder.

## lr selection (pre-registered; runs/pt2sz/select_pt2sz.py, run by hand by the coordinator; no hooks) = pt300x's protocol
Per size and arm independently, F87 rule 1 exactly (runs/ptlr/AUDIT.md rules 1-2; readout_rules.best_lr with near_rule
'top2', the FINAL IN-TRAINING bpb, readout_rules.bracket):
1. Stage A: the 4 grid lrs x seeds 0-3 (2 sizes x 2 arms x 4 lrs x 4 seeds = 64 runs).
2. Best lr = argmax of the mean area over seeds 0-3; if the top two are within 2 area points, the one with the lower mean
   final in-training bpb (val_bits_per_byte at the last eval; non-finite = worst; an exact tie -> the lower lr).
3. Open bracket = the pick is at an edge of the arm's grid and beats its neighbour by more than 2 points -> stage X: one
   ladder rung beyond that edge, seeds 0-3, once per arm; then rule 2 over the five lrs. Still open after the X -> STOP:
   nothing more for that arm, the user decides (an amendment). An edge pick within 2 points of its neighbour is closed.
4. Closed -> stage C: seeds 4-7 at the pick (model kept). The cell's 8 seeds = seeds 0-3 of the pick's stage A / X runs +
   seeds 4-7 of C. C = 2 arms x 4 seeds x 2 sizes = 16 runs; X adds 4 runs per extended arm (at most 16).
5. A diverged run keeps its area (data) and counts against its lr; final val_loss > the same seed's none run at the lowest
   none lr run so far + 0.5 nats is flagged 'unstable' (kept). An own failure (the run stopped on an error of its own; the
   coordinator lists it in runs/pt2sz/OWN_FAILURES.txt; a name that is not a selection run prints a WARNING) is scored over
   all 75 eval slots with the missing evals 0, bpb worst; an infrastructure loss (node failure, TIMEOUT, a preemption that
   was not requeued, HOLD) is not listed but resubmitted as the same job file by the coordinator: rename its flag to
   SUBMITTED_<run>.lost<k> (the job id history is kept), then `make_jobs_pt2sz.py --stage <S> --size <z> --run <run>
   --submit`; the trainer resumes from the last 2000-step ckpt (run_guard hands the claim of a job that is no longer
   running to the new one). A job that fails its own checks (exit 3 / 4) is not re-run without the user: its arm is
   withheld (rule 6).
6. A run whose check.json (CHECK_PY) failed, whose recomputed area differs from the check's by more than 0.006 (double path;
   the check rounds to 2 decimals), or that fails the integrity re-check (every training-defining argument of this batch's
   intent, code_sha, steps, n_params, grad ckpt, layers, 75 evals at the intended steps, window events / on_frac / use,
   step-0 lm within 1e-3 of pt2sz_refs.json) withholds its arm: nothing is decided; the coordinator investigates.
`select_pt2sz.py` prints every arm's grid means, the tie-break and the decision; `--write` appends the new X / C / STOP
entries to runs/pt2sz/decisions.json (append-only; the generator validates the whole log: one X one rung beyond a stage-A
edge, then one C in the grid or at the X lr, or a STOP; nothing after a C or a STOP) and writes picks.json, then prints
the commands `make_jobs_pt2sz.py --stage X|C --size S --arm A --test-only` / `--submit` that the coordinator runs by hand.
- **Rule choice (stated before any result).** F87 rule 1 (top two, in-training bpb) is the coordinator's instruction and
  pt300x's protocol. F98's reused picks came from pt2main rule 1 (every lr within 2 of the top, OFFLINE full-val bpb;
  53M / 190M on Burst grids, 505M on Torch). The 2% jobs also write the offline bpb, so select_pt2sz.py prints pt2main rule
  1's pick next to each decision (descriptive, never used); on pt2s505's own 505M runs both rules pick the same lrs (none
  4.8e-3, win-ctx 6e-4). The asymmetry (old cells: pt2main rule 1 on 3-5-rung grids, 53M / 190M picked on Burst; new
  cells: F87 rule 1 on 4-rung grids on Torch) is a known difference of this design, written into the FINDINGS entry.

## References (llmadd_torch/pt2sz_refs.json, sha256[:12] bb92a90edc26; a NEW file: pttpp_refs.json is frozen and untouched)
CPU job (scale_work/pt2sz/refs_pt2sz.py + refs_cpu.sh via ~/research_tools/cputest/cpu_run.sh, cpu_short, 16 CPUs, account
torch_pr_625_general; job 21999483 COMPLETED, refs_table.txt "ALL IDENTITIES HOLD"), pttpp's method on this batch's snapshot
and the jobs' own ARGS:
- `--dry_inputs`, both sizes x both arms x seeds 0-7: rc 0, the model line, 3 rows x 2 operand sites, the target line only
  for win-ctx; the 9 DRY rows' sha256 identical across the arms and = make_jobs_pttpp.dry2 (the Burst pt2s190 rows of the
  seed with this model line) for all 16 (size, seed); the whole dry file = pt2same 190M's dry_inputs.txt of the seed except
  the model line (16/16), and at 294M = pttpp tpp2-294m's dry_inputs.txt line for line, model line included (8/8).
- Step-0 lm (scale_work/step0_ng.py, CPU bf16, no_grad), seeds 0-7 (+ win-ctx seed 0 = none seed 0 bitwise, both sizes).
  - **294M at 3e8 tokens vs the tpp line:** the CPU value with this batch's ARGS equals pttpp_refs.json '2|294m' (CPU, the
    tpp line's ARGS: 156.8M tokens, eval every 131) to the last bit at all 8 seeds (difference 0.0): step 0 does not depend
    on the token count. CPU - GPU (the pttpp tpp2-294m A100-80GB runs, one distinct value per seed) -5.5e-5 .. +1.0e-4.
    Reference = the GPU values (the jobs' own hardware and trainer; pttpp's rule): 11.674556 / 11.678276 / 11.709671 /
    11.683458 / 11.671468 / 11.706556 / 11.688425 / 11.689414.
  - **93M:** no 2% 93M GPU run exists: reference = the CPU values 11.626397 / 11.613168 / 11.613724 / 11.628060 /
    11.598959 / 11.625436 / 11.611862 / 11.613997 (the CPU method's calibration: 190M 2% |CPU - GPU| <= 1.1e-4, 294M <=
    1.0e-4, 505M 7% 1.8e-4; CHECK_PY's tolerance 1e-3).

## Pre-registered analysis (runs/pt2sz/readout_pt2sz.py; written and tested before any pt2sz run)
- **Cells**: the five Torch-A100 2% 3e8 cells at each arm's pick, seeds 0-7, gain_s = area(win-ctx, s) - area(none, s)
  (paired by seed; area = readout_rules.area = 100 x mean over all 75 evals of the 6-bucket held-out exact match, the
  definition F98 used). Reused: F98's cells through readout_pt2same's own run_paths / load_run, each run re-checked
  (integrity as above with that cell's lr; check.json passing; area double path; finite area: pt2same Amendment A3;
  node.txt A100-SXM4-80GB), and asserted equal to F98 (the 8-seed none / prior / gain means to 0.005 and every per-seed gain
  to 1e-9 against runs/pt2same/readout.json). New: the C decisions of decisions.json, recomputed by the selection rule and
  asserted equal; seeds 0-3 at the pick from stage A / X, 4-7 from C.
- **Primary statistic**: the OLS slope b of the five 8-seed mean gains on ln N_nonemb (equal weights; weights 53M -0.262,
  93M -0.108, 190M +0.034, 294M +0.122, 505M +0.214: the new points carry 31% of the |weights|).
- **Decision rule.** "SHOWN (increase)" (or "(decrease)" for b < 0) iff ALL of:
  - (a) the 90% seed-bootstrap CI of b excludes 0 on b's side. Bootstrap = pt300x's: seeds resampled with replacement
    within each size, the two arms paired (the gain per seed), 10000 resamples; numpy default_rng(0), one (10000, 8) index
    draw per size in ascending N order; percentile interval (5th, 95th).
  - (b) the slopes on seeds 0-3 and on seeds 4-7 (point estimates, the same five sizes) both have the sign of b.
  - (c) the size-level interval from the OLS residuals of the five points, b +- t(0.95, df 3) x se(b),
    se(b) = sqrt(SSR / 3 / Sxx), excludes 0.
  Otherwise "NOT SHOWN" with the numbers and the criteria that failed. Any cell not final (an arm without its C, a STOP, a
  run missing or unfinished), with an integrity problem, or a reused cell that is not F98's -> "NOT DETERMINABLE" (no slope
  is read). A new cell whose 8-seed none, prior or gain mean lies outside its expected range by more than 15 points -> the
  verdict line carries "INVESTIGATE BEFORE READING" and the coordinator records the investigation here before anyone reads
  the verdict. "SHOWN" is a direction only (no minimum effect size); never "constant" / "no size effect" (no equivalence
  test is registered).
- **F98's own statistics on the five points** (printed next to the primary, no second verdict; as F98 / pt2same defined
  them): the slope with the JOINT paired seed bootstrap (pt2same's: one (10000, 8) draw of default_rng(0) shared by every
  size and both arms) and pt2same's rule ("SHOWN (+/-)" iff that CI excludes 0 and both halves share its sign); the 8
  per-seed slopes (positives x/8, t interval df 7 at 90% and its two-sided p, exact sign-flip p: F98 audit I2); the slope of
  the per-size median gains.
- **Descriptive (printed, no verdict)**: the two new points against the line through the three old ones (prediction,
  residual, joint seed bootstrap 90%) and the slope between the two new points alone (joint and within-size 90%); none and
  prior arms separately (mean area on ln N; F98: the 2% rise came from the prior arm); the five leave-one-size-out slopes;
  the slope weights; the gain without b11 (readout_rules.B5) per cell and its slope; per prior cell the LM share of the
  gradient in the window (pttpp's grad_stats: pre-clip norms a / (a + b) at on-evals, median per seed then median / min over
  seeds; plus the pooled median, F98's definition; the share of on-evals with prior > LM; the derived clip coefficient;
  flagged when the LM share < 0.6); the expected-range table.
- **What the verdict may be called.** A fixed-token, fixed-lambda statement: at 3e8 tokens tpp falls from 21.2 to 0.745
  along the five sizes, so size, training sufficiency and the prior's relative dose stay confounded (F98 / F99 I4). "SHOWN
  (increase)" = the 2% area gain at 3e8 tokens increases with ln N_nonemb across five sizes on one cluster, with the
  size-level interval; not a speed-up claim, not "the prior helps larger models more", and never compared with 7% (F98
  can't-write 8, F99 can't-write 8). Since 3 of 5 points were seen before registration, a SHOWN is only partly confirmatory
  (see the simulation below); the new points' own information is the "new vs old line" block.
- **Known differences between the old and new cells, written before any result**: lr rule and grids (above); the 53M /
  190M picks were made on Burst seeds 0-3 that share data with Torch seeds 0-3 (F98 I6: a winner's curse on 53M none that
  biases the slope positive); fixed lambda 5 is not a fixed dose (F98: prior / LM gradient ratio 0.48 / 0.35 / 0.32); seed
  4-7 eval items overlap partly with seeds 0-3 (R27). F98's I1-I6 carry over.

## Simulated false-SHOWN rate of the primary under a no-growth null (tests/sim_null.py; tests/sim_null.out, .json)
Written before any pt2sz run (CPU job 22003014, 2000 simulated read-outs per scenario, each through readout_pt2sz's own
slope_block / verdict / f98_stats with its fixed bootstrap draws). Scatter from F98's three cells (real run JSONs): seed SD
of the per-seed gain sigma 5.48 (53M 5.51, 190M 4.81, 505M 6.05; seed SE of a size mean 1.94), mean cross-size seed
correlation rho +0.25 (pairs -0.08 / +0.24 / +0.60), size scatter beyond seed noise tau_flat 1.60 (the three means around
their mean) / tau_line 0.00 (around their own line). Null = the same true gain at every size.
| scenario | P(primary SHOWN (increase)) | P(any SHOWN) | P(F98's rule SHOWN (+)) | median b |
|---|---|---|---|---|
| U unconditional (all five cells drawn), tau 0 / 1.60, normal or resampled noise | **0.025-0.036** | 0.045-0.064 | 0.076-0.147 | -0.01..+0.05 |
| C conditional on the three observed cells, new cells flat at the old cells' mean 50.31 | **0.34-0.38** | 0.34-0.38 | 0.57-0.63 | +1.16 |
| C ... flat at 53M's level 47.99 | **0.16-0.20** | same | 0.53-0.58 | +1.12 |
| C ... flat at 505M's level 52.98 | **0.18-0.23** | same | 0.61-0.68 | +1.20 |
| L reference, NOT a null: the new cells on the old cells' line | 0.53-0.69 (power) | | 0.81-0.89 | +1.42 |
Grid (no noise beyond a fixed seed-residual shape): with the old cells fixed, the primary reads SHOWN (increase) only for
93M gain 46-51 and 294M gain 49-56 (35 of 441 combinations of 40..60 x 40..60).
Reading: as an operating characteristic the rule is conservative (2.5-3.6% false SHOWN one-sided; F98's own rule 8-15%).
**Given the three cells already observed, a no-growth outcome at the two new sizes still reads "SHOWN (increase)" 16-38% of
the time** (the old cells carry 69% of the |weights| and already slope +1.44), and the rule's power when the new points sit
on the old line is only 0.53-0.69: the new points can confirm only weakly and refute only weakly. F98's own rule (no
residual criterion) is far more permissive (53-68% under the conditional null). The FINDINGS entry quotes these numbers next
to whatever verdict is read; the "new vs old line" block is the new points' own evidence.
Limitations: the conditional draws ignore the correlation between a new cell's seed residuals and the old cells' (rho
+0.25 on average); tau is estimated from three points.

## Expected ranges (outside -> check the run before using it; the read-out flags +-15)
Between the neighbours' F98 8-seed means (Torch): 93M none 20.6-28.0, prior 68.6-77.9, gain 48.0-49.9; 294M none 24.4-28.0,
prior 77.4-77.9, gain 49.9-53.0 (F98: 2% gains +48 to +53). Every run: code_sha 93e22bf4c4c0, 18310 steps, 75 evals,
prior_events [[916, 1], [7324, 0]] / on_frac 0.349973 (win-ctx), none never on; step-0 lm = pt2sz_refs.json within 1e-3.
Divergence risk: none at 1.92e-2 (and an X at 3.84e-2): 505M none at 1.92e-2 fell to 15.5, no divergence.

## Metrics
area (above); in-training bpb = val_bits_per_byte at the last eval (8 x 16 fixed val windows); offline bpb = offline_lm.py
full val (check.json info.bpb_full; descriptive here). Train vs eval conditions: identical to pt2same (runs/pt2same/AUDIT.md
"Train vs eval" table: the arithmetic line at the end of a 512-token row, teacher forcing, format 0, held-out pairs by
rng(seed)); only the shape and lr differ.

## Cost and wall time (make_jobs_pt2sz.py --plan)
Per run, from run JSONs (curve[-1]['sec'], medians) + start-up / dry run / offline_lm / check (sacct elapsed - training:
53M 150 s, 190M 170 s, tpp2-294m 185 s, 505M 200 s):
- 93M (grad ckpt off): log-log in n_params through pt2same 53M A100 (2300 s training) and 190M A100 (6740 s): 3702 s
  (+160 s) = **1.07 A100-h**; the other route (53M A100 x the 7% L40S 93M / 53M ratio at 3e8, 4964 / 3465 = 1.4325) gives
  3295 s (+160 s) = 0.96 h. The larger is used.
- 294M (grad ckpt on): pttpp tpp2-294m on A100-80GB (9570 steps, 75 evals, 5708 s median) x 18310 / 9570 = 10921 s (+190 s)
  = **3.09 A100-h**; the eval-split route (190M on A100 at two step counts: 0.353 s/step + 3.49 s/eval, the eval share
  scaled to 294M) gives 2.98 h. The larger is used. (For scale: pt2same 53M 0.68 h, 190M 1.92 h wall; pt2s505 505M 4.36 h.)
| stage | 93M runs | 294M runs | A100-h |
|---|---|---|---|
| A | 32 x 1.07 = 34.2 | 32 x 3.09 = 98.9 | 133.1 |
| C | 8 x 1.07 = 8.6 | 8 x 3.09 = 24.7 | 33.3 |
| A + C | 42.8 | 123.6 | **166.4** |
| worst case + X on all four arms | +8.6 | +24.7 | **199.7** |
Wall time of running (queue waits and the selection by hand come on top): stage A at 8 concurrent A100s ~16.6 h, at 16
~8.3 h (294M first: its 32 runs are 4 waves of 3.1 h at 8 concurrent); X then C, each one 294M wave ~3.1 h -> ~23 h at 8
concurrent, ~15 h at 16. The user's gpu48 per-user cap (16 GPUs across partitions, shared with ptdom on B200 and any other
GPU job) and nice 1000 on a busy a100_tandon decide the concurrency; at writing a100_tandon had ~600 pending jobs.
Time limits reserve at most 32 x 3 + 32 x 8 = 352 h for stage A.
**Disk**: stage A / X models removed after their checks; kept: 16 stage-C models, 8 x 0.37 GB (93M) + 8 x 1.18 GB (294M) =
**~12.4 GB**; transient per running job: the model file (<= 1.18 GB) until its check, the 2000-step ckpt (294M ~3.5 GB,
deleted by the trainer at the end); run JSONs small. Keeping every model would be ~60 GB (48 x 0.37 + 48 x 1.18 at the
worst case). /scratch at writing 4.69 / 5.00 TB (93.9%); submit_pt2sz.sh refuses at >= 97%.

## Submission (stage A only; the coordinator)
`bash runs/pt2sz/submit_pt2sz.sh` checks (no HOLD; `make_jobs_pt2sz.py --check --stage A`: every stage-A job file = its
rendering, the snapshot, the frozen shas, the known answers; 64 runs; none submitted yet; myquota /scratch < 97%) and prints
the 64 sbatch lines, **294M first (its 32 stage-A runs, the longer ones), then 93M**, each seed-major; `--go` submits them
in that order and writes SUBMITTED_<run> flags (the sbatch output). X / C: by hand after `select_pt2sz.py --write`. Kill
switch: `touch runs/pt2sz/HOLD` (a pending or requeued job exits at start; a job that ends on HOLD exits 0 = COMPLETED
without a run JSON: the selection shows it as missing).

## Tests before submission (runs/pt2sz/tests; outputs *.out next to the scripts)
Light tests: `tests/run_all.sh` on the final files (run_all.out: ALL TESTS PASS, 2026-11-05T22:25:58Z); CPU-heavy tests on a
Torch CPU node via cpu_run.sh (`tests/cpu_tests.sh`, job 22003014 COMPLETED); references: scale_work/pt2sz/refs_cpu.sh
(job 21999483 COMPLETED).
| test | result |
|---|---|
| render + `bash -n` (render.out, check.out, bash_n.out) | 64 stage-A job files written by `make_jobs_pt2sz.py` (the generator runs known_answers() first); `--check --stage A` OK; `bash -n` clean 64/64; 32 x 294M (time 08:00:00) + 32 x 93M (03:00:00), nice 1000, a100_tandon |
| render_check.py | RENDER CHECK PASS: a second rendering byte-identical; names / tags unique and unused by any other batch; job names = llmadd_p2z-<run>; headers; every required check, in order; the model-removal file name; CHECK_PY block = make_pt2main_tasks.CHECK_PY = pt2same's, 64/64; body diff vs pt2same's 190M job of the same arm / seed: 45 changed lines each, all in the allowed classes (job name, nice, time, comments, R/OUT/TAG, INTENT, ARGS, model-line grep, DRY hash + echo, CHECK_PY run file / intent + RC capture, RC=0, model removal, done line, exit $RC), 64/64 |
| args_diff.py (argparse emulation) | ARGS DIFF PASS: K1 the emulation reproduces 7 submitted run JSONs' args exactly (pt2same 53M / 190M, pt2s505, pttpp tpp2-294m); differing keys exactly as intended: vs pt2same 190M / pt2s505 505M {d_model, n_layers, n_heads, d_ff, out, tag}; vs pt2same 53M + lr; vs pttpp tpp2-294m (same shape) {tokens 3e8 vs 156.8M, eval_every 250 vs 131, ckpt_every 2000 vs 0, out, tag}; the 64 jobs' CHECK_PY intent = their emulated args on every key (only the tokenizer is outside the intent, as in pt2same; covered by the DRY rows); INTENT lines = the intent |
| generator known answers (every rendering) | this batch's ARGS / intent builders reproduce make_jobs_pt2s505.render's submitted job files (4 cells), its ARGS and intent, and make_jobs_pt2same.intent + the ARGS lines of its submitted 53M / 190M jobs (8 cells) |
| check_ka.py (CHECK_PY known answers) | CHECK KA PASS 48/48: CHECK_PY with this batch's intent form passes on real pt2same 190M none s0, 53M win-ctx s5, pt2s505 win-ctx s6, none@9.6e-3 s1 (each with its own offline_lm.json / gpu.csv), and that intent = the one embedded in each submitted job; planted lr x2, eval_every 26, tokens 156.8M, steps 18309, shifted window, step-0 ref +0.002, arith_lines 4, d_model + 64, code_sha a3bce3ab63c1, 74 evals -> rc 4 each |
| dry_digest.py (CPU, the rendered jobs' ARGS) | DRY DIGEST PASS 4/4 (93M / 294M x none / win-ctx, seed 0): DRY sha = the job file's = pt2sz_refs = make_jobs_pttpp.dry2; 294M dry file = pttpp tpp2-294m's line for line (model line included; digests 15fde897d34702d9, 5a4d2b79c3c6907b); 93M = pt2same 190M's except the model line (7e0fbd72aa941d93, fe5bc5a5ffbad255); model line, 3 rows x 2 operand sites, target line only for win-ctx |
| e2e_job.py (job files run by bash on CPU, fake nvidia-smi) | E2E JOB PASS 10/10: A / B (93M none s0, 294M win-ctx s1, A100-SXM4-80GB) rc 0 up to the DRY line; C A100-40GB / D L40S rc 3 before any input check; E HOLD rc 0; F seed-1 rows in a seed-0 job rc 3; G changed trainer rc 3; tail: T1 stage A passing -> rc 0, model removed; T2 stage A, lr != intent -> rc 4, model kept; T3 stage C passing -> rc 0, model kept |
| CPU references (refs_pt2sz.py) | ALL IDENTITIES HOLD (above, "References") |
| select_pt2sz.py --selftest | SELECT SELFTEST PASS 24/24 (synthetic roots, real CHECK_PY, the real generator): empty -> wait; interior pick -> C; top two within 2 -> the lower IN-TRAINING bpb (the offline bpb, opposite order, not used; pt2main rule 1 printed and flagged); not every-lr-within-2; open upper -> X 3.84e-2, open lower 294M win-ctx -> X 7.5e-5 (rendered: nice 1000, model removal); X worse -> C (seeds 4-7, model kept); X better -> STOP, nothing after; CHECK_PY fail, double-path mismatch, step-0 lm +0.002, tokens of the tpp line -> WITHHELD; missing / unfinished -> wait; own failure scored 0, a misspelt name -> WARNING; 6 malformed decision logs refused |
| readout_pt2sz.py --selftest | READOUT SELFTEST PASS 22/22: designed +3 -> SHOWN (increase) with b, halves, arms (-3 / 0), LOSO, no-b11, residuals on the old line 0, per-seed slopes, sign-flip p, LM share 0.700 exact; zero -> NOT SHOWN (a); -3 -> SHOWN (decrease); halves disagree -> NOT SHOWN (b); size scatter +-6 -> NOT SHOWN (c) with (a), (b) passing while F98's rule reads SHOWN; new points +4 / -2 off the old line read exactly; a shared seed effect collapses the joint CI only; reused run with another code_sha, a counted run on an L40S (node.txt), a reused 505M NaN area, a failed C check, a missing C decision / C run / reused run, a C decision != the recomputed pick -> NOT DETERMINABLE; gains ~20 -> INVESTIGATE BEFORE READING; designed reused cells != F98 -> NOT DETERMINABLE |
| readout_pt2sz.py --ka (real reused runs) | KA PASS 20/20: the 48 F98 runs pass every re-check (integrity, double path, node.txt A100-SXM4-80GB); F98 reproduced: gains 47.99 / 49.95 / 52.98, none 20.59 / 27.98 / 24.45, prior 68.58 / 77.93 / 77.42, every per-seed gain = readout.json; slope +1.4354, joint CI [+0.2201, +2.6442] (= readout.json to 1e-9), halves +1.837 / +1.034, pt2same's rule SHOWN (+); arms none +1.388 [+0.155, +2.668], prior +2.823 [+2.557, +3.137]; per-seed slopes (+0.92 .. -2.21 .. +4.93), t df 7 [-0.05, +2.92], p .109, sign-flip p .109; median slope +1.08; weights -0.3167 / +0.0473 / +0.2693; LM share pooled 0.676 / 0.740 / 0.757 (pttpp's per-seed-median definition 0.675 / 0.742 / 0.750) |
| readout / select on the real root, before any run | NOT DETERMINABLE (no C decision); every arm INCOMPLETE (missing runs); nothing written |
| regress_frozen.py | REGRESSION PASS: runs/pt2same/FROZEN.sha256 and runs/pttpp/FROZEN.txt all OK before and after, the three frozen refs files unchanged; in their own processes make_jobs_pt2same re-renders its 32 job files (+ JOBS.sha256 / a1 manifest verify), make_jobs_pt2s505 its 32 (+ tasks()), make_jobs_pttpp all 302 submitted job files byte for byte; nothing in runs/pt2same, pt2s505, pttpp, pt300x newer than this build's start |
| sbatch --test-only (2 jobs; testonly.out) | both accepted, nothing queued: p2z-294m-lr4.8e-3-none-s0 and p2z-93m-lr6e-4-win-ctx-s3 on a100_tandon, estimated start 2026-11-07 ~00:17 (~31 h queue at nice 1000 at writing) |
| submit_pt2sz.sh without --go (submit_dry.out) | checks pass (/scratch 93.94%), 64 sbatch lines, the 32 294M first, nothing submitted |
| submit_test.sh | SUBMIT TEST PASS: print mode (64 lines, 294M first, no flag); bad argument rc 2; on an isolated scratch copy with a fake sbatch: HOLD -> rc 1, --go -> 64 flags in 294M-first order, a second --go refuses; no flag / out dir in the real batch dir |

## Audit status (experiment-audit)
i double path: area recomputed from the run JSON vs CHECK_PY's area (<= 0.006) for every new and reused run; the reused
cells vs F98 (readout.json, every per-seed gain); offline vs in-training bpb both in check.json. ii asserts: generator
(frozen shas, import without main, the ARGS / intent builders = the submitted pt2s505 / pt2same jobs', grids = ladder rungs
with the interpolation inside, shapes, steps / evals / window, refs = dry2), every job (HOLD, trainer and module shas,
COMMIT, data sha256, GPU name, model line, operand sites, target table, DRY rows, CHECK_PY), selection and read-out
(integrity re-check of every counted run, node.txt GPU, decision log validation). iii known answers: CHECK_PY on real
pt2same / pt2s505 runs + planted failures, dry digests = submitted Torch jobs' dry_inputs.txt, F98's numbers reproduced by
the read-out, designed selftests. iv: - (no result yet). v: **pending** — the fresh general-purpose adversarial build audit
is started by the coordinator (this build agent did not run it). vi: trap list checked (position / template / range = pt2same;
the prior's activity asserted per run; no silent filtering: own failures scored, missing runs block; same-name-different-
meaning: N = N_nonemb, the lr rule differs from the reused cells', LM-share definitions). vii: every job prints the decoded
training and eval rows (--dry_inputs) before training.

## Files at writing (sha256[:12]; 2026-11-05 ~17:30 EST, before any submission)
make_jobs_pt2sz.py af2ba3425158; pt2sz_refs.json bb92a90edc26; runs/pt2sz: select_pt2sz.py ab95a029492c, readout_pt2sz.py
37ddcba5105a, submit_pt2sz.sh 2a0ee7915e6f; code/COMMIT 28a1f51774e5, code/tools/pretrain_range.py 93e22bf4c4c0; the 64
stage-A job files = the generator's rendering (`make_jobs_pt2sz.py --check --stage A`; concatenated sha256[:12]
4c5266f2ea5b); tests/: render_check.py f24db5821607, args_diff.py fc3f30658877, check_ka.py a8db0afecef0, dry_digest.py
af17d942fdee, e2e_job.py 1ec8cd2321c8, regress_frozen.py 6fbad6b3727c, sim_null.py 14a0e39e4cd1, submit_test.sh
16482d3003ca, run_all.sh 6af1381f39bc, cpu_tests.sh d72f44051394; scale_work/pt2sz/refs_pt2sz.py a6189ccea43f, refs_cpu.sh
a1fad5d57761. Read-only inputs asserted by the generator: make_jobs_pt2same.py 84b1f1103231, make_jobs_pttpp.py
b4bfa6eb248d, make_jobs_pt2s505.py a5a7fb60eedc, make_jobs_rb6scale.py 19bc055c2d09, make_jobs_ptlr1b.py 8dd7b3407459,
pttpp_refs.json d10d720c1242, pt2s505_refs.json 479e394771e9, rb6scale_refs.json b65472eb3d29, runs/pt2s505/picks_confirm.json
713ee399cce4, runs/pt2same/code (trainer 93e22bf4c4c0, make_pt2main_tasks a2e8a9621383, readout_rules e91ee8a8a846,
offline_lm 226f4cac70ea); the read-out asserts runs/pt2same/readout_pt2same.py bf7cced43238. A change to the selection or the
read-out after this point is an amendment written here before any pt2sz result is read.

## Open points for the coordinator (decided by the build agent; not in the instruction)
1. Selection rule: F87 rule 1 with the IN-TRAINING bpb, as instructed (pt300x's protocol), although F98's three reused
   picks came from pt2main rule 1 (within 2, OFFLINE bpb). The pt2main pick is printed next to each decision, never used.
2. The ladder gets one rung below pttpp's (7.5e-5), so that an open lower edge of 294M win-ctx (grid from 1.5e-4) can get
   its X. The 294M grids are asymmetric by the stated argmax argument (none 1 below / 2 above, win-ctx 2 below / 1 above).
3. Primary (a) uses pt300x's within-size bootstrap (F98 used the joint one): F98's joint-bootstrap rule is printed next to
   it as "F98's own statistics", no second verdict. The within-size bootstrap ignores the shared seed stream (conservative
   when seeds correlate positively across sizes; rho +0.25 on the old cells).
4. 294M step-0 references = the A100 GPU values of pttpp's tpp2-294m runs (identical step 0 at 3e8: the CPU value equals the
   tpp-line CPU value bitwise; CPU - GPU <= 1.0e-4); 93M = CPU values (no 2% 93M GPU run exists).
5. Stage A / X models are removed after a passing check (pttpp job2); this changes the job's tail vs pt2same (RC capture,
   `exit $RC`; a failed check still exits 4 and keeps its model). ~12.4 GB kept instead of ~50-60 GB.
6. The expected-range flag also covers the gain (pt300x: none / prior only); node.txt's GPU name is re-checked by the
   read-out for every counted run (new and reused).
7. Queue: the --test-only estimate is ~31 h before the first start at nice 1000; the user's gpu48 cap and ptdom share it.
8. Experiment-audit v (the fresh adversarial build audit) is pending: the coordinator runs it before submission.

## Fresh build audit, submission and the user's decision (coordinator, 2026-11-05 ~20:00 EST; no result read)
- Fresh adversarial build audit (experiment-audit v; a new general-purpose agent, blind): tests/audit_build.md. Blocking:
  none. Important: I1, the lr-pick offset (a one-rung pick change moves a cell's gain by 0.9-1.4 points, as large as the
  ~2.3-point 294M - 93M difference the new points must resolve; with no true size effect and a plausible 294M win-ctx pick
  one rung low, P(SHOWN) 0.56-0.69 in its simulation; mechanism checked by the coordinator at make_jobs_pt2sz.py:98-99,
  select_pt2sz.py:157-160, readout_pt2sz.py:221-234); I2, the verdict is mostly fixed by the three cells seen before
  registration (independent re-simulation: false SHOWN 0.18-0.40 given them, power 0.68 on the old line, likelihood ratio
  of SHOWN ~1.7-3.8; (b) passes 98% under the conditional null). 12 minor items in the report.
- The user, asked whether to add the audit's suggested amendment (descriptive pick-sensitivity output, or a robustness
  condition on SHOWN) before any result: 「不改」. The registered rule (a)(b)(c) and read-out stand unchanged; the audit's
  I1 / I2 numbers are carried into the FINDINGS entry whatever the outcome (with the likelihood ratios beside the verdict).
- Submission: all 64 stage-A jobs (SUBMITTED_ flags; 294M first). The first submit_pt2sz.sh --go was cut by a 600 s
  wrapper timeout after 33 jobs (flags = queue, 33 = 33, no orphan); the other 31 were submitted with
  `make_jobs_pt2sz.py --stage A --submit` (skips flagged runs; --check passed before). 64 flags, 64 jobs queued, nice 1000.

## Deadline (the user, 2026-11-10)
"给 pt2sz 定一个截止日：比如 11-13 还没开跑，论文的 2% 尺寸部分就只用已有的 3 个尺寸（F98），不再等它。" If no pt2sz job has started by
2026-11-13, the paper's 2% size part uses F98's three sizes only. The batch may still run and be reported later.
