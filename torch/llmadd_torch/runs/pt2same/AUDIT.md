# pt2same — the 2% size comparison on one cluster (53M / 190M re-run on Torch A100-80GB; 2026-11-02)

User decision 2026-11-02 (relayed by the coordinator): re-run the 53M and 190M 2% cells at 3e8 tokens on Torch A100-80GB
with the same trainer, job body and learning rates as pt2s505 and F86's picks, seeds 0-7. Written BEFORE any job of this
batch was submitted (only CPU tests ran: tests/). Generator llmadd_torch/make_jobs_pt2same.py; nothing is submitted by the
generator (runs/pt2same/submit_pt2same.sh --go, run by the coordinator).

## Question
F86's 2% (pt2main) cells at 3e8 tokens: 53M (pt2b) and 190M (pt2s190) ran on Burst (GCP A100-SXM4-40GB, trainer
a3bce3ab63c1 via burst/pin_run.sh, Python env llmadd_env311), 505M (pt2s505, F86 505M addendum) on Torch A100-SXM4-80GB
(trainer 93e22bf4c4c0, env open_r1). The pre-registered cross-site check FAILED for 2% win-ctx (Burst-ro
tools/xsite_summary.py, P1b-win-ctx: Burst - Torch L40S per seed +2.06 / -4.51 / -4.11 / +0.37, mean -1.55 > 1.50, max 4.51 >
3.00; P1b-none passed), so 2% cells are not compared across sites and the 53M / 190M / 505M size comparison is confounded
with the site. **Question:** with all three sizes on one cluster, GPU type, trainer and job body, does the prior's area gain
(win-ctx minus none, each at its F86 pick) change with model size at a fixed 3e8 tokens and 2% arithmetic?
Secondary (descriptive): the size of the Burst -> Torch site effect on the same cells and seeds.

## Design
| cell | lr (F86 pick) | Burst twin tasks (seeds 0-3 / 4-7) | n_params | N_nonemb | model |
|---|---|---|---|---|---|
| pt2same-53m-lr8e-3-none | 8e-3 | 350-353 / 370-373 (pt2b) | 52,672,128 | 14,165,376 | d384 x 6, 6 heads, d_ff 1536 (trainer default), grad ckpt off |
| pt2same-53m-lr2e-3-win-ctx | 2e-3 | 350-353 / 370-373 (pt2b) | 52,672,128 | 14,165,376 | same |
| pt2same-190m-lr4.8e-3-none | 4.8e-3 | 366-369 / 382-385 (pt2s190) | 190,297,344 | 113,283,840 | d768 x 12, 12 heads, d_ff 3072, grad ckpt on |
| pt2same-190m-lr6e-4-win-ctx | 6e-4 | 358-361 / 382-385 (pt2s190) | 190,297,344 | 113,283,840 | same |
| reused: pt2s505-lr4.8e-3-none / -lr6e-4-win-ctx | 4.8e-3 / 6e-4 | (Torch, pt2s505 seeds 0-7) | 505,437,184 | 402,752,512 | d1024 x 24, 16 heads, d_ff 4096, grad ckpt on |

- Seeds 0-7 per cell: 4 cells x 8 seeds = **32 jobs**. Seed s = the same init, data stream (rng 2000+s / 1000+s) and held-out
  split (rng s) as Burst seed s and as pt2s505 seed s (the held-out split and the DRY rows are model-independent).
- Arms (make_pt2main_tasks.ARM): none = `--target none`; win-ctx = `--target helix3 --schedule window --window 0.05,0.4
  --site ctx` (lambda 5 = trainer default). `--arith_lines 1` (2%), 3e8 tokens = 18,310 steps of 32 x 512 on pretrain_data
  (trainer default), eval every 250 steps (75 evals), `--save_model --ckpt_every 2000`; every other flag = trainer default.
- **lrs fixed in advance, no new selection.** none@8e-3 / win-ctx@2e-3 (53M) and none@4.8e-3 / win-ctx@6e-4 (190M) are F86's
  picks (FINDINGS F86 table + R02 addendum; tools/pt2main_summary.py on the Burst results prints the same best lrs today:
  53M none 8e-3, win-ctx 2e-3 (1e-3 within 2); 190M none 4.8e-3 (9.6e-3 within 2, tie broken on offline bpb), win-ctx 6e-4
  (1.2e-3 within 2)). 505M = pt2s505's picks by the same rule (picks_confirm.json: none 4.8e-3, win-ctx 6e-4). Why fixed: (1)
  the purpose is to remove the site confound only; re-selecting would change site and lr together; (2) every size keeps "its
  own best lr by pt2main rule 1"; (3) a grid would cost 3-5x. Known risk, stated before any result: a site effect could move
  an optimum; the ties above (within 2 area points) say the area is flat near each pick, so a moved optimum costs little
  area. Selection effects: the 53M / 190M picks were made on Burst seeds 0-3, which share init and data with Torch seeds 0-3
  here, so seeds 0-3 may carry some winner's curse (F86: 53M none@8e-3 23.1 on seeds 0-3 vs 13.8 on 4-7); seeds 4-7 are
  selection-free at every size (505M: picked on its own seeds 0-3). Hence the halves rule below. R27 caveat (FINDINGS
  erratum 2026-11-02): seeds 4-7 are independent training runs, but each seed's held-out split is drawn independently, so
  their held-out eval items overlap partly with seeds 0-3 (~1/4 of the 6-bucket area weight, mostly b11 / b12): the halves
  and the seed bootstrap treat seeds as independent units, which slightly understates the seed-to-seed correlation.
- Trainer and job body = pt2s505's: code = runs/pt2same/code = `git archive 7f1d06b tools/`, asserted equal to
  runs/pt2s505/code file by file (COMMIT copied verbatim; pretrain_range.py 93e22bf4c4c0 + every imported module's sha
  re-checked in every job). The 93e22bf4c4c0 default path was regression-tested against a3bce3ab63c1 (the Burst trainer) on
  CPU in pt2s505's tests (run JSON values identical, 36/36 model tensors bitwise, none / win-ctx / win-ctx grad ckpt on).
- Hardware: Torch a100_tandon (every node NVIDIA A100-SXM4-80GB), account <slurm-account>, qos gpu48, 1 GPU,
  8 CPUs, 64G, `--requeue`, **`--nice=20`** (as pt2s505 and the user's instruction; note: the batches queued now use 2000 /
  5000, so these jobs go ahead of them on the same partition). Time limits 53M 02:00:00, 190M 05:00:00 (2.5 x plan rounded
  up to the half hour; TIMEOUT is not requeued, preemption is and resumes from the 2000-step ckpt).

## Differences from the Burst twins (all checked, tests/args_diff.log, tests/dry_cpu.log)
| what | Burst (F86) | Torch (this batch) | why acceptable |
|---|---|---|---|
| run args `data`, `tokenizer`, `out` | /scratch/yh6210/llmadd_data/..., llmadd_runs/<task>/out/<tag> | /scratch/yh6210/llmadd_torch/pretrain_data, .../models/OLMo-2-1124-7B/stage1-step2000-tokens9B, runs/pt2same/out/<run>/<tag> | site paths only: data sha256 identical (both jobs check the same two hashes); tokenizer = the same OLMo-2 tokenizer (the 9 DRY lines incl. decoded rows are identical); out is a path |
| run arg `tag` | _pt2b-lr8e-3-none etc. | _pt2same-53m-lr8e-3-none etc. | a file-name suffix only (the trainer uses it in output file names, nowhere else); unique among all batches (asserted) |
| code_sha | a3bce3ab63c1 (64abc19, pin_run) | 93e22bf4c4c0 (7f1d06b) | = pt2s505's trainer (the point: one trainer for all sizes); default path regression-tested identical (above) |
| GPU / env | A100-SXM4-40GB, Python 3.11.13 env311 | A100-SXM4-80GB, Python 3.11.13, torch 2.10.0+cu128 (open_r1) | this IS the site change; the A100-80GB guard keeps one GPU type |
| job body | Burst task (tick, pin_run, PINNED.txt) | pt2s505's sbatch body + an A100-80GB guard (pt2s505: any A100) + an INTENT line | same checks; CHECK_PY verbatim |
| step-0 lm reference | 53M Torch pt999 2pct-none; 190M seeds 0-3 ptsize999, 4-7 none (recorded only) | 53M the same values; 190M pttpp_refs '2|190m' (seeds 0-3 identical to Burst's, 4-7 = the Burst runs) | every reference equals the Burst run's own step-0 lm exactly (32/32, args_diff.log) |
Every other run arg (31 of the 35 keys the run JSON records) is identical, as are steps 18310, n_params, gradient
checkpointing, prior / read-out layers, eval spacing, lm_chunk 0, prior_events [[916, 1], [7324, 0]] (win-ctx) / [] (none)
and the 75 evals (checked for all 32 jobs against the Burst run JSONs). The trainer ARGS string itself equals the Burst
task's ARGS with the paths / tag substituted (asserted at render time); 53M passes no model flags, as on Burst.

## Metric (unique definition; same as F86 and pt2s505)
- **area** = readout_rules.area(r) = 100 x mean over ALL 75 evals of the mean teacher-forced exact match over the 6 held-out
  buckets b11, b12, b13, b22, b23, b33 (code/tools/readout_rules.py, sha e91ee8a8a846; the same function F86's
  pt2main_summary and the R02 addendum used). Double path: CHECK_PY's own `info.area` in check.json must agree within 0.01.
- High = the held-out sums are answered correctly early and throughout the 3e8 tokens; it is a curve area, not a final
  accuracy (the R02 caveat).
- **Delta** = area(win-ctx, seed s) - area(none, seed s), paired by seed within a size.
- Token position: the scored span = the answer tokens + the newline of an arithmetic line at the END of a 512-token row
  (the training position; F63's fix), teacher forcing; padding none (fixed-length rows).

Training vs eval (the held-out buckets), per experiment-audit:
| dimension | training rows | eval rows (held_b*) | same? |
|---|---|---|---|
| sequence position / prefix | one arithmetic line at the end of each 512-token row after ~500 tokens of text (dry: `2 prior sites at [504, 507]`) | the line at the end of a 512-token row after val text (evalset: seq - len(line) text tokens, then the line) | yes (F63) |
| template | 4 formats drawn at random (FORMATS) | format 0 `Q: a + b = ` only | eval format is one of the trained ones |
| number range | operands 0..999 by digit-length bucket; held-out pairs excluded; 5% rehearsal pairs | the held-out pairs of b11..b33 (150 rows each, b11 37) | held out by design: measures generalisation to unseen pairs (b11's pairs also occur in natural text: area without b11 printed by CHECK_PY) |
| data source / split | train.bin windows (rng 2000+seed), lines rng 1000+seed | val.bin contexts (rng 11), pairs held out by rng(seed) as sets | disjoint by content (pair set; train vs val text) |
| model state | train mode, bf16 autocast, grad ckpt (190M on, 53M off), 1 GPU, no sharding | eval mode, no_grad, bf16 autocast, same GPU | standard; identical across sizes and sites |
| prior | win-ctx: helix3 aux loss on the operand sites, steps 916..7323, lambda 5 | no aux term at eval | n/a |
Expected magnitudes: chance ~0; F86 (Burst) 8-seed means 53M none 18.43 / prior 69.91, 190M none 29.06 / prior 78.45;
505M (Torch) none 24.4 / prior 77.4.

## Pre-registered read-out
`python runs/pt2same/readout_pt2same.py [--json <f>] [--investigated "<note>"]` (self-tested on synthetic data before any
result: tests/readout_selftest.log). It implements exactly:
1. **Integrity (every counted run: 53M, 190M of this batch and pt2s505's 505M, 2 arms x 8 seeds = 48 runs).** finished (last
   eval at step 18309), check.json present and passing (CHECK_PY: every run arg = the intent, code_sha, run facts, 75
   evals, prior events / on fraction / use, step-0 lm within 1e-3 of the reference, offline full-val double path), and the
   recomputed area = check.json's within 0.01. Missing / unfinished -> incomplete; failed -> WITHHELD; either -> no verdict.
   An infrastructure loss (TIMEOUT, node failure, a preemption that was not requeued) is resubmitted as the same job file
   (remove its SUBMITTED_ flag, then submit_pt2same.sh --go: the trainer resumes from its last 2000-step ckpt, run_guard
   prevents a duplicate). A job that fails its own checks (exit 3 / 4) is not re-run without the user: the cell stays
   WITHHELD and the failure is listed.
2. **Expected range (investigate before reading).** Each new Torch cell's 8-seed mean area vs F86's Burst 8-seed mean of
   the same cell: 53M none 18.43, 53M prior 69.91, 190M none 29.06, 190M prior 78.45. Outside +-15 -> OUT OF RANGE: the
   read-out stops after the range table (no gain, slope or verdict printed) until the cell is investigated (check.json,
   dry rows, step-0 lm, node / GPU, curves, per-seed outliers) and the finding is passed with `--investigated`; it is
   printed with the result and goes into the FINDINGS entry.
3. **Per size (Torch only).** Delta per seed; means over seeds 0-3, 4-7, 0-7; wins x/8; 90% CI of the 8-seed mean (paired
   seed bootstrap, 10000 resamples, numpy default_rng(0), percentile).
4. **The 2% size comparison on one cluster (primary).** y = the 8-seed mean Delta of 53M, 190M (this batch) and 505M
   (pt2s505); x = ln N_nonemb (14,165,376 / 113,283,840 / 402,752,512). OLS slope (area points per ln N). 90% CI: joint
   paired seed bootstrap (one resample of the 8 seed indices per replicate, used for every size and both arms; seed s is the
   same data stream and held-out split at every size), 10000 resamples, default_rng(0), percentile. Plus the slopes of
   seeds 0-3 alone and 4-7 alone (point estimates).
5. **Decision rule.** "SHOWN (+)" / "SHOWN (-)" iff the 8-seed 90% CI excludes 0 AND the seeds 0-3 and seeds 4-7 slopes both
   have the 8-seed slope's sign. Otherwise "no size trend shown", with the point estimate, the CI and the reason (CI
   includes 0 / halves disagree). "SHOWN" is a direction only (no minimum effect size); the slope and per-size Deltas are
   always printed. **Never "constant" / "no size effect"**: no equivalence test is pre-registered. With three sizes a
   non-monotone pattern gives a small slope: the per-size table is read next to the slope, never instead of it.
6. **Descriptive (no verdict).** none and prior areas per size and their slopes (same bootstrap); Burst vs Torch for the same
   cells and seeds (none, prior, Delta: mean Torch - Burst, per-seed range, 90% CI, wins; step-0 lm agreement): the site
   effect; the mixed-site slope (Burst 53M / 190M + Torch 505M) next to the one-cluster slope.
Not read here: t90 / speed-up ratios, LM cost (offline bpb is in check.json and may be tabulated descriptively), any lr
question. No comparison with 7% (F87 / pttpp).

## Checks inside every job (exit 3 before training, exit 4 after)
1. HOLD (runs/pt2same/HOLD -> exit 0); sha256[:12] of the trainer and every module it imports (= WANT_SHA); COMMIT names
   93e22bf4c4c0; GPU name must match `*A100*80GB*` (exit 3 on A100-40GB, L40S, H100, ...); data sha256 (train.bin, val.bin).
2. INTENT line (size, dims, n_params, N_nonemb, grad ckpt, tokens / steps, 75 evals, warmup 183, window steps 916..7323 or
   none, arm, lr, seed, step-0 reference, Burst twin). `--dry_inputs` with the job's own ARGS: the model line (`model: 52.7M
   parameters; gradient checkpointing off` / `model: 190.3M parameters; gradient checkpointing on`), 3 printed rows with
   exactly 2 operand sites, the helix3 table line only for win-ctx, the sha256 of the 9 DRY lines = make_pt2main_tasks.DRY =
   the Burst twin's own dry_inputs.txt (asserted by the generator for all 32; recomputed on CPU in tests).
3. Training, offline_lm.py (full val), CHECK_PY verbatim with this run's intent -> check.json (exit 4 on a fail); gpu.csv.

## Cost and disk
- Planned: 53M 0.75 A100-h (Burst A100-40GB xsite53 measured 0.722 h, GPU util 97%), 190M 1.9 h (xsite rescaled 1.92 h;
  Burst pt2s190 elapsed ~2.1 h; Torch A100-80GB tpp2-190m 0.40 s/step at eval every 70 -> ~1.85 h at eval every 250; pt2s505
  505M 0.84 s/step -> 4.3 h, for scale) -> 16 x 0.75 + 16 x 1.9 = **~42 A100-h**; the limits reserve at most 112 h.
- Disk: models kept (offline_lm needs them; pt2s505 kept every model): 16 x 0.21 GB + 16 x 0.76 GB = ~15.5 GB; transient
  ckpts <= 2.3 GB per running 190M job. /scratch 4.71 / 5.00 TB (94.2%, myquota 2026-11-02).

## Tests before submission (CPU; scripts and logs runs/pt2same/tests/, all re-run by tests/run_all.sh on the frozen files)
| test | result |
|---|---|
| frozen inputs (FROZEN.sha256: this generator / read-out / submit script, pt2s505 / pttpp / rb6scale / ptlr1b generators, both refs files, the snapshot's COMMIT, trainer, make_pt2main_tasks, readout_rules, offline_lm) | all OK before and after the tests (frozen_check.log) |
| render_check.py | 32 files = the generator's rendering (JOBS.sha256, a second in-memory rendering byte-identical); names / tags unique and unused by any other batch; bash -n 32/32; #SBATCH header and every required check in all 32 (GPU guard before the input checks, dry run before training); the CHECK_PY block = make_pt2main_tasks.CHECK_PY = the Burst twin task's, 32/32; body diff vs pt2s505's job of the same arm and seed: 35 changed lines per job, all in the allowed classes (job name, time, comments, R/OUT/TAG line, GPU case, INTENT, ARGS, model-line grep, DRY hash + echo, CHECK_PY run file + intent) |
| generator render-time asserts (every job) | ARGS = the Burst task file's ARGS with paths / tag substituted; every flag = make_pt2main_tasks.render's want-table; the intent with Burst's code_sha / tag / lm0 = the intent embedded in the Burst task file that ran; make_pt2main_tasks.DRY = the hash of the Burst run's own dry_inputs.txt |
| args_diff.py | known answers: the argparse emulation reproduces 2 pt2s505 run JSON args and all 32 Burst twin run JSON args exactly; every job vs its Burst twin: exactly {data, out, tag, tokenizer} differ (32/32); run facts and 75 evals = the intent; code_sha a3bce3ab63c1 -> 93e22bf4c4c0; step-0 lm reference = the Burst run's step-0 lm exactly (32/32) and = all 32 of pttpp's Torch A100-80GB 190M 2% runs exactly; CHECK_PY's arg fields = the emulated args; read-out task table / picks / N_nonemb = the generator's |
| dry_cpu.py (seeds 0 and 4, both arms, both sizes = 8 jobs; the job's own ARGS, --out into tests/dry) | 8/8: rc 0, the model line, 3 rows x 2 operand sites, target line only for win-ctx; DRY sha = the job file's = make_pt2main_tasks.DRY = recomputed from the Burst twin's dry_inputs.txt; the whole dry_inputs.txt = the Burst one line by line (pin_run lines aside) |
| e2e_job.py (the job files run by bash up to the end of step 1 on CPU, fake nvidia-smi, scratch R) | A / B (53M none s0, 190M win-ctx s4, A100-SXM4-80GB): rc 0, data sha256 OK, INTENT, DRY = reference; C A100-SXM4-40GB / D L40S: rc 3 before any input check; E HOLD: rc 0, nothing run; F seed 1 rows in a seed-0 job: rc 3 'dry_inputs differ'; G a changed trainer file: rc 3 'sha mismatch' |
| sbatch --test-only (2 jobs: 190M win-ctx s0, 53M none s4) | both accepted on a100_tandon (estimated start 2026-11-03 ~10:00-12:00); nothing queued |
| readout selftest (synthetic only) | 11/11: known +6 / ln N -> SHOWN (+); known -4 -> SHOWN (-); exact zero slope -> no size trend shown; flat noisy -> no size trend shown; halves disagree although the 8-seed CI excludes 0 -> no size trend shown; a seed effect shared by all sizes cancels (joint resampling); file layer: complete fake tree -> verdict; a failed check / a missing run / an area double-path mismatch -> no verdict; a cell out of range -> PENDING, with --investigated -> verdict |
| readout_known.py (published runs only; no pt2same result exists) | the loader reproduces F86 R02's Burst 8-seed means and halves for all 4 cells (18.43 / 69.91 / 29.06 / 78.45) and F86's 505M addendum (none 24.4, prior 77.4, Delta +53.0, halves +52.0 / +53.9, 8/8); read-out paths = each job's run file and check.json, 32/32 |
| pt2s505_regress.py | make_jobs_pt2s505.render() still produces all 32 of pt2s505's job files byte for byte; tasks() = those 32; nothing in runs/pt2s505 newer than this batch's snapshot |
| submit_test.sh | print mode: 32 sbatch lines in JOBS.sha256 order, no flag written; bad argument: exit 2; a changed job file / a wrong frozen hash (scratch copy): exit 3 before any sbatch line; --go with a fake sbatch on a scratch copy: 32 flags, a second --go submits nothing |

Experiment-audit status before submission: i ✓ (area double path in the read-out; DRY rows by three routes; args by
emulation vs the run JSONs); ii ✓ (render-time asserts; in-job GPU / sha / data / model line / sites / target / DRY
checks; CHECK_PY); iii ✓ (known answers: emulation K1 / K2, published areas, synthetic slopes; negative controls C / D / F /
G); iv – (no result yet); v **pending**: the fresh adversarial reviewer (general-purpose sub-agent, given only this file,
the code paths and tests/*.log) is to be started by the coordinator right after submission (this session has no agent
tool); vi ✓ (trap list: position = training position; fresh data each step; the prior's gradient checked by CHECK_PY;
8 seeds; scoring position asserted in the dry rows; one area definition; n = 8 enforced by the integrity rule; pt2s505
regression + e2e for side effects); vii ✓ (each job log prints the decoded train rows with the prior sites marked and the
held-out / ood eval rows with their scored answer tokens, before training).

## Open points for the coordinator
1. `--nice=20` (as instructed and as pt2s505) puts these 32 jobs ahead of the batches queued now at 2000 / 5000 on the same
   a100_tandon partition (e.g. rb6_190m_dim at 5000). Confirm that this priority order is intended.
2. Experiment-audit v: start the fresh adversarial review sub-agent in the background right after `submit_pt2same.sh --go`;
   also the task panel entry + Monitor (the experiment-tracking rule).
3. Cost ~42 A100-h (limits reserve <= 112 h); disk +~15.5 GB of kept models with /scratch at 94.2%.
4. After the jobs: `python runs/pt2same/readout_pt2same.py --json runs/pt2same/readout.json`; the FINDINGS entry quotes
   the verdict exactly as printed (SHOWN (direction) / no size trend shown) and the descriptive site effect.

## Submission
(to be filled by the coordinator: date, job ids = SUBMITTED_<run> flags)

## Amendment A1 (coordinator, 2026-11-02, before any submission and before any result of this batch)
Source: the fresh adversarial build audit (tests/audit_build.md: no blocking item; I1-I3 important, M1-M9 minor).
Answers to the open points: (1) nice 20 ahead of rb6_190m_dim (5000) and ptshare (2000) is intended (the user: as fast as
possible; this batch feeds the paper). (2) The build audit above counts as review v for the build; a result-level audit
follows the read-out. (3) Cost and disk accepted.
Added to the pre-registration (nothing above is changed; the primary rule stays as written):
1. Slope weights (I1): with ln N_nonemb of 53M / 190M / 505M the OLS weights of the three cell gains are -0.317 / +0.047 /
   +0.269, so the verdict rests almost entirely on Delta(505M) - Delta(53M); 190M carries 7.5%. The read-out prints the
   three pairwise contrasts Delta505 - Delta53, Delta190 - Delta53, Delta505 - Delta190 (8 seeds, 90% seed bootstrap,
   descriptive) next to the verdict, and the FINDINGS entry states the weights.
2. Known risk (I2), written before any result: 53M none@8e-3 is a sharp, untied peak on Burst (seeds 0-3 23.10 vs
   neighbours 14.75 @4e-3 / 15.20 @1.6e-2; seeds 4-7 at 8e-3 only 13.8: a winner's curse). The lrs are F86's Burst picks
   (53M, 190M) but 505M's were chosen on Torch on its own seeds 0-3. If the site moves the optimum, the cost falls mainly on
   none (steep lr curve) at the small sizes, which inflates their Delta and biases the slope NEGATIVE. Three cells pick at a
   grid edge (190M none rising to the top; 190M and 505M win-ctx at the bottom), inherited from F86 / pt2s505.
3. Descriptive sensitivity for the most fragile cell (I2): 53M none at 4e-3 and 1.6e-2 on Torch, seeds 0-3 (8 extra runs,
   ~6 A100-h, same generator render() with only the lr overridden, job files in runs/pt2same/a1/). Read-out: 53M none
   best-of-{4e-3, 8e-3, 1.6e-2} on Torch seeds 0-3 by pt2main rule 1, and the slope recomputed with that cell
   (descriptive; it never changes the primary verdict).
4. Interpretation (I3): Delta ~ prior ceiling - none (seed SD: prior 0.9-2.0, none 5.4-7.9); any wording about the gain
   changing with size names the none arm as its source, at fixed lambda 5 (M3).
5. Site effect (M1): the read-out prints Torch - Burst; the cross-site check P1b was Burst - Torch (opposite sign); with no
   same-site replicate the site effect is not separated from run-to-run noise (descriptive only).
6. Re-runs (M5): a submitted job is never resubmitted (SUBMITTED_ flag); a re-run gets a new run name. The sentence in
   section 'Re-runs' that says to delete the flag is withdrawn. M6: 'same init across sizes' is withdrawn (shapes differ);
   the same seed shares the data stream and the held-out split only.
7. Seed overlap (FINDINGS R27): seed halves are independent training runs, but about 1/4 of the area weight comes from
   held-out items shared with seeds 0-3; the halves rule is read with this caveat.

## Submission (2026-11-02 18:21 EST, coordinator)
32 main jobs via submit_pt2same.sh --go (job ids in the SUBMITTED_<run> flags, 21696726-21696760 range) and the 8 A1 jobs
(a1/SUBMITTED_*, 21696761-21696768), all nice 20, account 520, a100_tandon. A1 job headers inherit the comment "F86's pick,
fixed in advance" from render(); for the A1 cells the lr is the sensitivity lr of Amendment A1 item 3, not a pick.

## Amendment A2 (A1 read-out definition, written before any A1 result)
Written 2026-11-02 ~18:50 EST. State then: the 8 A1 jobs (53M none lr 4e-3 / 1.6e-2, seeds 0-3) PENDING, no A1 out dir;
4 main seed-0 jobs running (nothing of this batch read). Source: tests/audit_a1.md, item 5a (three traps: 4-seed vs 8-seed
slope, mixing lrs to make 8 seeds, selection bias of choosing and evaluating on the same 4 seeds; recommended design
followed). This defines Amendment A1 item 3's 'best-of-{4e-3, 8e-3, 1.6e-2} ... and the slope recomputed with that cell';
nothing above is changed and the primary rule stays as written.
Implementation: runs/pt2same/readout_pt2same_a1.py (sha256 a358df929e82..., written before any A1 result). It imports
readout_pt2same as a module (sha asserted = FROZEN.sha256's bf7cced43238...; never edited; its globals asserted unchanged
after the run), runs the primary read-out path first (RO.load + RO.read, the same code as `readout_pt2same.py`), and
asserts it equals runs/pt2same/readout.json when that file exists (canonical JSON, every key; a difference -> 'STOP', exit 3,
no A1 section). Pass the same `--investigated` note as the primary run.
- **(a) Seeds.** The A1 analysis uses seeds 0-3 only, on both sides (the 8e-3 side too). Lrs are never mixed to make 8 seeds
  (no 'seeds 0-3 at the new lr + seeds 4-7 at 8e-3'). A 4-seed slope is never compared with, or printed next to, the
  8-seed primary slope: the A1 section prints only seeds 0-3 slopes.
- **(b) Torch best lr of 53M none** among {4e-3, 8e-3, 1.6e-2} by pt2main rule 1 = readout_rules.best_lr (near_rule
  'within', TIE 2): the best mean area over seeds 0-3; every lr within 2 points of the top is tied and the tie goes to the
  lower mean offline (full-val) bpb, taken exactly as tools/pt2main_summary.py:102-104 of the snapshot (runs/pt2same/code:
  the run dir's offline_lm.json entry with tag '_<tag>', the seed, max_windows 0, through readout_rules.offline_bpb with the
  run's divergence state); an exact bpb tie -> the lower lr. readout_rules.bracket on the three means: 'open' (best at a
  grid edge and > 2 above its neighbour) is printed 'open bracket at the <edge> edge, not extended' (no further lr is run);
  'note' (area argmax at an edge by <= 2) is printed as a note. 8e-3 is the pick's own 4 runs (the primary's files).
- **(c) Descriptive comparison.** S0-3(lr) = OLS slope on ln N_nonemb of the seeds 0-3 mean Delta of 53M / 190M / 505M,
  Delta per seed = win-ctx - none, every cell at its pre-registered pick (53M win-ctx 2e-3, 190M 4.8e-3 / 6e-4, 505M
  4.8e-3 / 6e-4; seeds 0-3), and 53M none at lr. Printed: S0-3(8e-3) (= the primary's seeds 0-3 slope, asserted) and
  S0-3(best). Identity, asserted to 1e-9 and printed: S0-3(best) - S0-3(8e-3) = w53 x (mean none@8e-3 - mean none@best)
  (seeds 0-3 means), w53 = (x53 - xbar) / sum (xi - xbar)^2 with x = ln N_nonemb (14,165,376 / 113,283,840 / 402,752,512)
  = **-0.316653** (190M +0.047304, 505M +0.269349). A better none at the Torch-best lr therefore raises the slope by 0.3167
  per area point. The slope is also recomputed by numpy polyfit (double path).
- **(d) Selection-bias guard.** Leave-one-seed-out: for each held-out seed h in 0-3, rule 1 (as in (b), incl. the bpb
  tie-break on those 3 seeds' bpb) on the other 3 seeds picks lr_h; the value is seed h's area at lr_h; LOSO none = the mean
  of the 4 values. Printed next to the plain best-of-3 (chosen and evaluated on the same seeds 0-3) and @8e-3, with the fold
  picks, and the slope S0-3 with 53M none per seed = the LOSO values (the same identity holds and is asserted). No bracket
  per fold.
- **(e) Table.** 53M none area per seed (0-3) and mean at 4e-3 / 8e-3 / 1.6e-2: Torch (this batch + A1) next to Burst (F86
  pt2b runs 350-353 lr4e-3 / lr8e-3, 362-365 lr1.6e-2; area via readout_pt2same.load_run, i.e. finished, check.json
  passing, area double path), and Torch - Burst per seed and mean; a Burst gap prints n/a with its reason (descriptive,
  never blocks). Burst's own pick (8e-3, F86) is printed under it.
- **(f) Integrity.** Each of the 12 Torch runs (3 lrs x seeds 0-3) must pass readout_pt2same.load_run's checks (finished,
  check.json present and passing, recomputed area = check.json's within 0.01), have a finite area, and have a valid
  full-val offline_lm entry (pt2main rule 6) whose bpb_full = check.json's (double path). In addition the primary read-out
  must have a verdict (not 'no verdict' / 'PENDING'; tests/audit_a1.md 5a step 2). Otherwise every A1 line, (b) to (e),
  prints 'not computed' and the reasons are listed; there is no fallback to 8e-3 or to fewer runs. A diverged run is data
  (its area counts against its lr, its bpb counts as worst), as in pt2main.
- **(g) Separation.** A1 never changes the primary verdict. It is printed in its own section after the primary output
  ('== Amendment A1 item 3 / A2 ... does NOT change the VERDICT above'); its JSON goes to a separate file
  (`--json runs/pt2same/readout_a1.json`; readout.json and the primary JSON are refused). No CI, no verdict, no
  equivalence wording: 4-seed point estimates only. The FINDINGS entry calls it 'seeds 0-3, descriptive, with selection
  bias (see LOSO)'.
Run after the primary read-out: `python runs/pt2same/readout_pt2same_a1.py --json runs/pt2same/readout_a1.json`.
Tests (before any A1 result): `--selftest` on synthetic trees, 14/14 PASS (tests/selftest_a1.log): best = 8e-3 (A1 slope =
primary seeds 0-3 slope to 1e-12, LOSO folds all 8e-3); best at the upper / lower edge -> 'open bracket ..., not extended';
tie within 2 broken by the lower offline bpb (the in-training bpb would pick the other lr); one failed / one missing A1 run,
a missing offline entry, an area double-path mismatch, a primary without a verdict -> every A1 line 'not computed' (the
primary verdict = that of the clean tree); identity vs a brute-force polyfit; LOSO known answer (areas 4e-3 [44,10,10,10],
8e-3 [14,16,16,16], 1.6e-2 [5]x4: in-sample best 4e-3 18.5, folds 8e-3/4e-3/4e-3/4e-3 -> LOSO 11.0); a readout.json
mismatch -> STOP; the readout.json guard; RO globals unchanged. `--known` on the published Burst pt2b runs only (no
pt2same file): per-seed areas = tests/audit_a1.md Q2 (4e-3 14.75, 8e-3 23.10, 1.6e-2 15.20), the offline-bpb extraction
and its check.json double path work on real files, rule 1 picks 8e-3 untied, each Burst dir unique (tests/known_a1.log,
PASS). The script is not in FROZEN.sha256 (that file is not edited); its sha is recorded here.

## Amendment A3 (coordinator, 2026-11-02 evening, after the fresh A2 audit tests/audit_a2.md; no result read)
1. NaN guard (audit_a2 item 7, out of A2's scope): the frozen readout_pt2same.load_run lets a NaN area through. Before the
   primary read-out is run, the coordinator checks that every area of the 32 main and 12 A1 runs is finite (a separate
   check, the frozen read-out unchanged); any NaN withholds the verdict and is investigated first.
2. Re-runs of A1 jobs (m4): a re-run gets a new name; the A1 read-out then needs an amendment mapping it before reading.
3. Wording (m5): LOSO corrects only the Torch re-selection; the 8e-3 side is itself a Burst pick with a winner's curse.
4. Reading order (m6): the Torch - Burst table of the 53M none areas is read before any A1 slope line.
