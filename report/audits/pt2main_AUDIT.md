# pt2main — the 2% main setting on Burst: lr brackets, long budget, 190M (2026-10-16)

User decisions (2026-10-16): the 2% arithmetic share (`--arith_lines 1`) becomes the MAIN setting; the 5% window
(`--schedule window --window 0.05,0.4`) is the main recipe; every arm, the prior too, gets its own best lr; 2%
long-budget runs (so none can reach 90% and a clean speed-up ratio exists) and 2% at 190M; maximise parallelism (Burst
holds 12 GPU tasks at once).

Files: generator `tools/make_pt2main_tasks.py` -> `tasks/350-*.sh .. tasks/361-*.sh` (72 tasks; its `EXTRA` list, empty,
renders re-runs and conditional cells as 362-399 the same way); read-out
`tools/pt2main_summary.py` (thin wrapper around `tools/pt999_summary.py`, which now also knows the `win-shuf-ctx` arm);
this file. Trainer `tools/pretrain_range.py` pinned at 64abc19 (code_sha a3bce3ab63c1, = the ptgw / ptlr / xsite53
snapshot) through `burst/pin_run.sh`; env `llmadd_env311`; data `/scratch/$USER/llmadd_data/*` (task 332, sha256 =
Torch's).

## Questions
1. **pt2b (53M, 3e8 tokens).** At 2%, does the window prior on the operands (win-ctx) still beat none when each arm
   runs at its own best lr (grid 1e-3 / 2e-3 / 4e-3 for both)?
2. **pt2s190 (190M, 3e8 tokens).** The same at 190M: none at 6e-4 / 1.2e-3 / 2.4e-3, win-ctx at 6e-4 / 1.2e-3,
   win-text (prior on numbers in the FineWeb text) at 6e-4.
3. **pt2long (53M, 1.2e9 tokens = 4x).** With a budget long enough for none to reach 90%: the t90 ratio best-vs-best
   (the paper's "N x faster" at 2%); geometry control (helix3_shuf in the window) and the text site at lr 1e-3.

Self-contained: every comparison is between cells of this batch (same hardware, same pinned code, same seeds = same
init, data stream and held-out split). No seed pairing with Torch cells; Torch numbers below are only expectations.

## Cells (helix3, lambda 5, prior layers 1/3 and 2/3 of the depth, bs 32 x 512, seeds 0-3, every model saved)
| block | model | tokens (steps) | data | cells (arm @ lr) | tasks | `# time:` | planning A100-h |
|---|---|---|---|---|---|---|---|
| pt2b | 53M d384x6 | 3e8 (18310) | pretrain_data | none, win-ctx @ 1e-3, 2e-3, 4e-3 | 350-353 | 02:00:00 | 24 x 0.75 = 18 |
| pt2s190 | 190M d768x12 (grad. ckpt on) | 3e8 (18310) | pretrain_data | none @ 6e-4, 1.2e-3, 2.4e-3; win-ctx @ 6e-4, 1.2e-3; win-text @ 6e-4 | 354-357 | 05:30:00 | 24 x 2.0 = 48 |
| pt2long | 53M d384x6 | 1.2e9 (73242) | pretrain_data_3B | none, win-ctx @ 1e-3, 2e-3; win-text, win-shuf-ctx @ 1e-3 | 358-361 | 08:00:00 | 24 x 3.0 = 72 |

`# time:` = 2 x the Torch L40S time of the same configuration (training loop; pt999 2pct-none / 2pct-gate-ctx /
2pct-gate-text, ptgw lr5e-4-W: 0.952-0.969 h; ptsize999 190m-2pct-none / -gate-ctx: 2.598-2.616 h; pt2long = 4 x 0.97
h), rounded up to the half hour, asserted in the generator. Reason (audit 2026-10-16, finding 2): the A100-40GB is
untimed, a 53M run may be bound by the Burst node's CPUs rather than its GPU, and the tick never retries a TIMEOUT, so
a run that hits the limit loses all its hours; billing is by elapsed time, so a longer limit costs only the guard's
reservation. This is 2.7x the planning rate, above the shared "1.5-2x the expected A100-40GB time": the expected time
is taken as the slowest GPU this code has been timed on. The previous pt2long limit (05:06 = 1.7 x 3.0 h) left 1.3x over
L40S speed. xsite53's measured sec/step may lower the limits before 354+ / 358+ are submitted (regenerate).

Arms: none = `--target none`; win-ctx = `--target helix3 --schedule window --window 0.05,0.4 --site ctx`; win-text =
the same with `--site text`; win-shuf-ctx = `--target helix3_shuf` (rows of the helix3 table permuted per seed) with
the window on the operands. Tags `<block>-lr<lr>-<arm>` (unique; `pt2b-`, `pt2s190-`, `pt2long-` are used nowhere
else); task names `<NNN>-<tag>-s<seed>` with NNN = block base + seed. Submission order (the tick submits in name order,
at most 12 GPU tasks in flight; the xsite53 tasks 340-347 go first): pt2b, then pt2s190, then pt2long, and within a
block all cells of seed 0 before seed 1, so the first wave shows every cell. Planned A100-h: 138 (limit-hours the guard
reserves while in flight: 2 / 6 / 8 h per task, at most 12 x 8 = 96 h). 362-399: the generator's EXTRA list (re-runs
of a failed task and the conditional cells below), rendered like the 72 and read by the read-out.

## Metrics (as in tools/pt999_summary.py; one cell per tag)
- **area** (primary): mean over all evals (every 250 steps + the last step) of the 6-bucket mean held-out exact match
  (`held_b11 .. held_b33`, answer tokens AND the newline, teacher forced, greedy; `evaluate()` in pretrain_range.py),
  x100. 150 items per bucket (b11: 35-39, all its held pairs). Unordered held-out pairs per seed; neither order is
  ever an arithmetic line.
- **t90** (M tokens): first crossing of 0.9 by the 6-bucket mean, linear interpolation between evals. Long budget:
  a run that never crosses is counted at the budget (1200M), so a ratio with a censored none is a lower bound.
- **area5 / b11 area**: area without b11 / of b11 alone (see natural occurrences below).
- **LM cost**: offline bits per byte on ALL of the run's val.bin (`tools/offline_lm.py` on the saved model; 2e6 tokens,
  3906 windows of 512); the in-training 65k-token value only where the offline one is missing (marked).
- Diagnostics: prior_used_frac (window: 0.350 of the steps), cka_ops_helix3 (fixed read-out layers, every arm),
  final per-bucket accuracy, b33 split by sum < / >= 1000.
- pt2long's val.bin (3B build) differs from pretrain_data's and the eval rows take their context text from val.bin:
  LM and accuracy numbers compare only within pt2long.

## Pre-registered rules (implemented in tools/pt2main_summary.py; tested on fake results, below)
1. **Best lr per arm** (per block) = argmax of the mean area over seeds 0-3, over the arm's grid (the cells above plus
   any EXTRA lr); every lr within 2 points of the top -> the one with the lower mean bpb: offline (full val) when every
   compared run has it, else the in-training bpb for all of them (marked; never a mix). A diverged run keeps its
   (near-zero) area, so it counts against its lr; its NaN bpb counts as worst. **Open bracket**: the best lr is at
   the edge of the arm's grid and beats its neighbour by more than 2 points (plan T1's rule). An area argmax at the
   edge by 2 or less is printed as a note (a flat optimum may lie beyond it) but triggers nothing.
2. **"The 2% speed-up survives tuning"** (per block) iff area(win-ctx@best) - area(none@best) > 0 with >= 3/4 seed
   wins and paired t p < .05 on seeds 0-3, AND, when the confirm seeds 4-7 of both cells exist (conditional cells,
   run when the gap is below 10), the same sign with >= 3/4 wins on seeds 4-7 (plan T1's confirm rule; seeds 4-7
   are selection-free). A gap below 10 without them reads "survival pending". Same test for win-text (single lr)
   vs none@best. Repeated on area5 (reported; the decision uses area).
3. **Long-budget headline** = t90 ratio none@best / win-ctx@best = ratio of the seed medians over seeds 0-3 (4 seeds:
   mean of the middle two), censored runs counted at the budget; 90% CI from 10000 paired seed bootstraps (numpy
   default_rng(0)). The median of n values contains a censored run as soon as (n + 1) // 2 of them are censored (4
   seeds: **2**), so: written ">= R" when at least 2 of 4 none runs are censored (the median is then a lower bound;
   with all censoring at the budget a Kaplan-Meier median equals this median while fewer than half are censored, and
   at exactly half its curve sits at 0.5 up to the budget, where conventions differ), no ratio when at least 2 of 4
   win-ctx runs are; the share of bootstrap resamples whose none median is censored is printed next to the CI (those
   resamples make the CI's upper end a lower bound too). (The first version marked ">=" only at 3 of 4: audit finding 1.) The ratio
   belongs to this budget's schedule: the window covers 5-40% of 1.2e9 tokens (6e7-4.8e8, vs 1.5e7-1.2e8 in pt2b) and
   the cosine decays over 1.2e9, so it is printed with the window in tokens and written as "N x at a 1.2e9-token
   budget (window 0.06-0.48e9 tokens)", not as the main recipe's ratio. Confirm seeds 4-7, if run, give a second,
   selection-free ratio line.
4. **Control**: at pt2long lr 1e-3, win-shuf-ctx must be below win-ctx (mean area difference > 0, >= 3/4 seed wins).
   If not, the pt2long gain is not attributed to the helix geometry.
5. **b11 separately**: natural "a + b = c" occurrences of held-out pairs in the training text are not excluded in this
   batch (symmetric across arms within a seed: same held-out split and same text rows). Report b11 and area5 next to
   area; count the exposures afterwards on Torch (CPU), e.g.
   `python tools/count_natural_add.py --data /scratch/yh6210/llmadd_torch/pretrain_data_3B --tokenizer <olmo2> --seeds 0-3 --out <prefix>`
   (and on pretrain_data for pt2b / pt2s190). pt2long samples 1.2e9 of the 3e9 tokens (random windows), so a run
   expects ~0.4 exposures per corpus occurrence.
6. **Integrity** (a failure means the comparison is not read): one finished run per (tag, seed); within a block and
   seed the step-0 LM loss agrees across cells within 1e-3 (same init and first batch); every counted task's
   check.json passed. **A diverged run is data, not an integrity failure** (rule 1): its check looks at the prior only
   before the first non-finite eval and skips the offline |orig - logged| test, so it passes when nothing else is
   wrong; the read-out lists it with its divergence step. (The first version failed a diverged prior run's check and
   so voided the comparison it should count against: audit finding 3.) **Re-runs**: a task runs once, so a failed
   task (TIMEOUT, OOM, a lost pin, a failed check) is re-run as the same cell and seed under a new number in EXTRA;
   the read-out takes, per (tag, seed), the first finished run in task order whose check passed (else the first
   finished one, reported as a failed check) and reports the others.
- Conditional cells, NOT generated (budget: the user decides; added to the generator's EXTRA, 362-399, rendered and read
  like the 72): an open bracket -> the next lr beyond the edge for that arm (x2 or x0.5, seeds 0-3); a best-vs-best gap
  below 10 area points -> seeds 4-7 for none@best and win-ctx@best (winner's curse: the lr is selected on the seeds
  that test it). Seeds 4-7 first need their DRY row hash computed on Torch (as for 0-3; the generator refuses without).

## Expected (written before results; Torch L40S, other hardware, so magnitudes only)
- pt2b: Torch 2% at lr 1e-3: none 9.4 (8 seeds, 7.3-13.1), gate-ctx 59.1, win-text 65 (seeds 6-7). 7% none rises
  with lr up to 2e-3 (35.0 / 56.2 / 58.8 at 5e-4 / 1e-3 / 2e-3), so none@4e-3 may be best (open bracket possible).
  win-ctx 55-70; best-vs-best gap +30-50 if the 7% pattern holds.
- pt2s190: Torch 190M 2% none at 6e-4: 12.2 (4 seeds); gate-ctx 67 / 56 (seeds 0-1). none should gain from a higher lr.
- pt2long: if the number of arithmetic lines governs (7% none t90 194M tokens = 1.5M lines), 2% none needs ~0.8e9
  tokens (some seeds censored); win-ctx 0.2-0.4e9 -> a ratio of roughly 2-4. win-shuf-ctx well below win-ctx (at 7%
  the shuffled gate was below none). win-text between none and win-ctx.
- Per task: prior_on_frac 0.349973 (pt2b / pt2s190: steps 916-7323 of 18310) and 0.349990 (pt2long: 3663-29296 of
  73242); prior_used_frac = prior_on_frac for ctx, 0.9-1.0 of it for text (Torch 2pct-win-text: equal).

## Train vs eval conditions
| dimension | train | eval | same? |
|---|---|---|---|
| sequence position | 1 arithmetic line at the end of a 512-token row after FineWeb text | one line at the end of a 512-token row after val text | yes (at 2% the only training line is the last one; the 7% caveat does not apply) |
| template | 4 formats, random | format 0 ("Q: a + b = ") | subset |
| values | operands 0..999, 6 length buckets balanced, held pairs rejected | held pairs per bucket; seen = rehearsal pairs; ood_b4x = 4-digit, never trained | held disjoint by construction (both orders) |
| text | train.bin windows | val.bin windows | disjoint files (pt2long: 3B val; pretrain_data's val lies inside 3B's train, irrelevant here) |
| model state | train mode, bf16 autocast (190M: gradient checkpointing, same math) | eval mode, bf16 autocast | yes |

## Checks inside every task (exit 3 before training, exit 4 after)
1. data sha256 of train.bin / val.bin = Torch's (tasks/332-data.sh values);
2. `--dry_inputs` through pin_run (the trainer of 64abc19): the model line (`model: 52.7M parameters; gradient
   checkpointing off` / `model: 190.3M parameters; gradient checkpointing on`, i.e. the state the trainer decides),
   ctx/none: 3 printed rows with exactly 2 operand sites (1 line per row), text: the text-site statistics line, the
   target table (`target features (1000, 10)` for helix3 / helix3_shuf, none for none), and the sha256 of the 9 selected
   decoded lines = the value computed on Torch with the same pinned code and sha-identical data (generator table DRY);
   PINNED.txt names a3bce3ab63c1;
3. after training + offline_lm (`check.json`): every run arg = intent (34 fields incl. data and out path, tag, lr, tokens,
   arith_lines, model dims, schedule / window / site / target); code_sha; steps, n_params, grad_ckpt_on, geo /
   read-out layers, lm_chunk 0, eval_every_steps 250; number of evals (75 / 294), strictly increasing, last at steps-1
   (a resumed run has no duplicates or gaps); window arms: prior_events = [[916, 1], [7324, 0]] / [[3663, 1], [29297, 0]]
   and prior_on_frac exact, geo > 0 and 0 < grad_prior < inf at every eval where the prior acted (before a divergence),
   used = on (ctx) or within [0.9, 1] x on (text);
   none: never on; step-0 lm within 1e-3 of the same-seed Torch run where one exists (pt2b: pt999 2pct-none; pt2s190:
   ptsize999 190m-2pct-none; the loss before the first update depends on neither lr nor arm); offline_lm on the full
   val with |orig_minus_logged| < 0.002 (skipped only if the run diverged). Divergence onset = the first eval with a
   non-finite lm, val_loss, geo or grad_prior. Info (not checked): area, area5, b11, t90, bpb, diverged and its step,
   sec/step, projected hours, GPU utilisation median and peak memory from gpu.csv.

## Verification on Torch before commit (2026-10-16)
Fake Burst root on Torch (scratch): `$LLMADD_ROOT/llmadd` = this repo (for `pin_run.sh`'s git archive of 64abc19),
`llmadd_env311/bin/python` = open_r1's python (3.11.13, torch 2.10.0+cu128, transformers 4.57.6), `llmadd_data/*` =
Torch's pretrain_data, pretrain_data_3B and tokenizer; the task scripts run from a copy of `burst/` + the rendered
tasks, exactly as run_task.sbatch runs them (cwd = code snapshot, `OUT`, `LLMADD_ROOT` set).
- **Generator.** `python tools/make_pt2main_tasks.py --check --show`: 72 tasks, 18 tags, numbers 350-361 in block
  order; intent asserts on every rendered ARGS (each flag once, exact values per arm / size / budget, nothing else),
  steps 18310 / 73242, window events as the trainer's float rule ([[916, 1], [7324, 0]] / [[3663, 1], [29297, 0]]).
  Written once (72 written), second run 72 unchanged. Scratch repo copy: a hand-edited task file is reported and kept;
  a task with status/ but no file -> "refused ... nothing written", exit 2; a foreign file in 35x -> warning.
- **Headers / syntax.** `bash -n` on all 72; headers parsed with the tick's own `hdr()` / `tl_hours()`: kind gpu, cpus 8,
  after `331-env311 332-data`, no gpus line; time 02:00:00 / 05:30:00 / 08:00:00 -> 2 / 6 / 8 h reserved (384 h if all
  were in flight; at most 12 are, <= 96 h). Submission order of 340-361 identical under `LC_ALL=C` and `en_US.UTF-8`
  sort. (Re-checked after the audit fixes.)
- **Rendered trainer arguments, seed 0 of every cell** (`--show`; `$DATA` = `$ROOT/llmadd_data/<data>`), e.g.
  `--data $DATA --tokenizer $TOK --tokenization whole --out $OUT/pt2b-lr4e-3-win-ctx --target helix3 --schedule window
  --window 0.05,0.4 --site ctx --arith_lines 1 --lr 4e-3 --seed 0 --tag _pt2b-lr4e-3-win-ctx --save_model --ckpt_every 2000`;
  pt2s190 adds `--d_model 768 --n_layers 12 --n_heads 12 --d_ff 3072`; pt2long adds `--tokens 1.2e9` (full list in the
  generator's `--show` output).
- **Step 1 of all 72 tasks** (inputs + `--dry_inputs` through `pin_run.sh`, placeholder row hash): data sha256 OK,
  PINNED a3bce3ab63c1, model line `52.7M ... off` / `190.3M ... on`, 2 operand sites per row (ctx / none), text-site
  statistics (text, 8 keys: median 105-124 numbers per batch, prior applied on 50/50 batches), `target features (1000, 10)`
  for helix3 / helix3_shuf and none for none. One row hash per (data, size, site, seed) across all cells (20 keys), now
  the generator's DRY table. **Known answers:** the 4 keys 53M / pretrain_data / ctx = the Torch logs of pt999
  2pct-none s0-3; the 4 keys 190M / pretrain_data / ctx = ptsize999 190m-2pct-none s0-3 (and 190m-2pct-gate-ctx);
  the text path (53M / pretrain_data / text, same pinned code) = the pt999 2pct-gate-text logs s0-3. The 190M-text and
  3B keys have no Torch log (computed here with the same code and sha-identical data). With the real table, step 1 of
  350 win-ctx s0, 357 win-text s3, 359 win-text s1, 361 win-shuf-ctx s3 reaches "step 1 OK" (rc 0).
  Negative: a pt2long task with pretrain_data in place of pretrain_data_3B -> `train.bin: FAILED`, exit 3, no training.
  (A first render asserted `target features (1000, 11)`; the trainer's helix3 table has 10 columns -- numlib drops sin
  for period 2 -- so every helix3 task failed its step-1 assert; fixed to 10 and re-run.)
- **Intent check (step 4) on real full-size Torch runs** (the rendered check script and intent, only the Torch cell's
  arith_lines / lr / tag swapped in): ptgw lr5e-4-W seed 1 (the pinned code a3bce3ab63c1, window on the operands):
  every check passes (34 args, steps 18310, n_params 52672128, layers [2, 4], lm_chunk 0, 75 evals, events
  [[916, 1], [7324, 0]], on 0.349973 = used, geo > 0 at 26 evals) except offline_lm (none computed there). pt999
  2pct-none seed 0 (older snapshot 79e9b410dcc3): step-0 lm = the reference (diff 0.0), facts / evals pass; FAIL as
  expected on code_sha, on 3 args that snapshot lacks (gate_start, gate_hyst, n_prefix) and on prior_events (not logged).
- **End-to-end CPU smoke of 3 real task scripts** (copies differing only in: tiny overrides appended to the training
  call `--d_model 64 --n_layers 3 --n_heads 2 --d_ff 128 --bs 16 --seq 256 --tokens 327680 --eval_every 10 --n_eval 20
  --ckpt_every 20` (argparse keeps the last value), `offline_lm --max_windows 64`, and step 4's intent recomputed for
  80 steps / 5 evals / events [[4, 1], [32, 0]] / 6.5M parameters; step 1 runs the real arguments and passes):
  - 350-pt2b-lr2e-3-win-ctx-s0, attempt 1 with `--stop_after 45` (a preemption after the step-39 checkpoint): offline_lm
    skips the unfinished run, check exit 4 (unfinished, stop_after, on_frac at step 40). Attempt 2, same `$OUT` (= a
    tick retry): "resumed ... at step 40 of 80", finished, checkpoint removed, model saved; every check passes except
    the full-val one (64 windows by construction; offline double path 1.8e-7). Resumed vs uninterrupted run: 198
    curve values, 0 differ (bit-identical), same prior_events.
  - 359-pt2long-lr1e-3-win-text-s1 (3B data, text site): all checks pass except the full-val one; used 0.325 / on 0.35
    (small batches skip the text prior more often), double path 1.1e-7.
  - 354-pt2s190-lr6e-4-none-s0: step 1 with the 190M model (checkpointing on) passes; check FAILs `grad_ckpt_on`
    by construction (the smoke kept the 190M expectation, the 6.5M model is below 150M) -- the check sees the
    trainer's decision -- and the full-val one; everything else passes.
  - Files that return to git per task: PINNED.txt, check.json / check.txt, dry_inputs.txt, gpu.csv, node.txt,
    offline_lm.json, <tag>/run_*.json (checkpoints and models are .pt: they stay on Burst).
- **1.2e9 tokens on 3e9 data** (trainer arithmetic, pinned code): 73242 steps (1,199,996,928 tokens), warmup 732,
  cosine over all steps, window 3663-29296 (25634 steps, 0.349990), 294 evals (every 250 steps + the last), 36
  checkpoints (every 2000 steps, overwritten, deleted at the end). Rows are random 512-token windows of train.bin
  (`rng.integers(0, len(train) - n)`, no epoch logic), so a 3e9-token file only changes coverage (~33% of tokens seen at
  least once). Run JSON: Torch 2% runs take 1340-1460 bytes per eval -> 294 evals ~ 0.43 MB (< 5 MB, returns through
  git). Checkpoint ~ 0.63 GB (53M: weights + 2 Adam moments), model file 211 MB; 190M: ~2.3 GB / 761 MB.
- **Read-out.** `tools/pt999_summary.py` after the edit (ARMS gains `win-shuf-ctx` before `shuf-ctx`, one new pair
  win-ctx - win-shuf-ctx, docstring): output byte-identical to the original on a frozen copy of 149 Torch run JSONs
  (pt999 + ptsize999: 87 lines; pt999b; ptlr). No existing tag ends in `-win-shuf-ctx`. `tools/pt2main_summary.py` on
  72 fake task dirs with designed answers: every cell area equals the independently computed value and
  pt999_summary's (double path, |diff| < 0.05); best lr by the bpb tie-break (pt2b none 2e-3 over 4e-3 at +1.0
  area; pt2s190 win-ctx 6e-4 over 1.2e-3); open brackets flagged (pt2s190 none upper; pt2long none upper, win-ctx
  lower); survives / does not survive (win-text 2/4 wins) as designed; t90 ratio 5.42 (designed 5.42) with none 1/4
  censored; control passes at 3/4; b11 split (72.2 vs area5 66.7); a step-0 lm mismatch, a failed check.json, a
  missing offline file ('*') and missing tasks ('incomplete', no decision) are reported.

### Audit fixes (2026-10-16, after the independent adversarial review; scratch `fix_v2/`)
The review found no blocking issue and six important ones (dispositions in "Audit status" v). Changed: the generator
(time limits, the check's divergence handling and grad_prior, EXTRA), the read-out (censoring, re-runs / extra cells /
confirm seeds, bpb tie-break, edge note, window in tokens), this file. The 72 task files (never committed, no status/
or results/) were deleted and regenerated: same 72 names; outside the embedded check script the only changed lines are
`# time:`, the limit in the planning comment and, for pt2long, "no same-seed Torch run" (diff against a backup); the
check script is identical in all 72 (one md5). `--check`: 72 unchanged.
- **Check, known answers + faults** (the rendered check on edited copies of the real pinned-code run ptgw lr5e-4-W s1,
  offline file supplied): real run passes every check (new: 0 < grad_prior < inf at the 26 evals where the prior
  acted); NaN from eval 20, NaN prior one eval before the LM, NaN from eval 4 (before the window): pass, `diverged`
  with onset steps 5000 / 4750 / 1000; grad_prior never logged, grad_prior = 0 once, geo = 0 once: FAIL (exit 4). The
  first version's check on the same copies: the three diverged runs FAIL (finding 3 reproduced), the two grad_prior
  faults pass (the weak check).
- **End-to-end CPU smoke of the regenerated 350-pt2b-lr4e-3-win-ctx-s0** (the smoke recipe above, current `burst/`
  copy, pinned trainer through `pin_run.sh`): step 1 OK with the real arguments and DRY hash; tiny training finished;
  every check passes (incl. grad_prior at the 1 eval where the prior acted) except the full-val one (64 windows by
  construction; double path 3.5e-8).
- **Read-out on fake results with designed answers** (tools copies; `fake2.py`, five variants): base = the first
  design: identical except the intended changes (bpb of the cell with one missing offline file now 1.2000* = in-training
  for all instead of a 1.0875 mix; edge notes; LM-cost label; headline with the censored-resample share and the window
  "60-480M tokens of 1200M"). cens2 (none@best censored in 2 of 4 seeds): ">=7.45x", median ">=927M" (designed
  >=7.45x); the first version printed "7.45x" without ">=" (finding 1 reproduced). censw (win-ctx 2 of 4 censored at
  both lrs): "no ratio". div (a win-ctx@4e-3 seed diverges at eval 10, NaN losses and offline bpb): listed as diverged
  data with its step, its area 38.3 (designed 38.33) counts against its lr, bpb worst; nothing breaks. extra (a tools
  copy whose EXTRA holds 27 tasks, test-only DRY placeholders for seeds 4-7): a re-run (374) supersedes the unfinished
  352 and the cell's area equals the design (10.0); a second finished run (375) of 350-...-4e-3-none-s0 is reported and
  ignored (21.7 = design); a re-run (376) of a finished run whose check failed (350-...-1e-3-win-ctx-s0) counts
  instead of it (24.3 = design 24.33); the extra lr pt2s190 none@4.8e-3 joins the grid (62.3 = design, new best, open bracket) and
  the gap to win-ctx falls to +7.3 -> "survival pending: confirm seeds 4-7 needed"; pt2b confirm seeds 4-7 (+2.3, 3/4)
  -> "agrees" -> SURVIVES; pt2long confirm seeds give a second headline line 5.09x (designed 5.09x). Double path with
  pt999_summary holds in every variant (19 cells in extra).

## Audit status (experiment-audit, minimal mode + known answers)
i double path: rows = pinned `--dry_inputs` vs Torch logs (12 keys incl. the text path); offline_lm recomputes the
in-training bpb in every task; summary area = pt999_summary's. ii asserts: intent asserts in the generator; exit 3
before training (data sha, model line, sites, target table, row hash, PINNED); exit 4 after (34 args, facts, evals,
window, prior use, step-0 lm vs Torch, offline). iii known answers: Torch logs (rows), Torch step-0 lm, a real
pinned-code window run through the check, fake results through the read-out, resume = uninterrupted. iv: n/a (no
results). v: an independent adversarial review ran after the build (2026-10-16): no blocking finding; six important
ones, all checked and handled: (1) the ">=" mark needed 3 of 4 censored none runs instead of 2 -> fixed (rule 3);
(2) pt2long's 05:06 limit left 1.3x over L40S speed and a TIMEOUT is never retried -> limits = 2x the L40S time
(02:00 / 05:30 / 08:00); (3) a diverged prior run failed its check and so voided its comparison -> divergence is data
(rule 6, check fixed); (4) re-runs and conditional cells were invisible to the read-out -> EXTRA in the generator, read
by the read-out, confirm seeds applied (rules 2, 6); (5) the lr grids are narrower than "every arm gets its own best
lr" -> recorded as a deviation below, for the user; (6) the headline ratio depends on the 1.2e9 schedule -> printed
and written with the window in tokens (rule 3); a token-anchored window arm is for the user. Minor ones fixed where
trivial: the tie-break no longer mixes bpb definitions, an edge argmax within 2 is noted, the check asserts
0 < grad_prior < inf. vi: 1 (2%: the only training line is at the eval position), 3 (geo > 0 while on, used = on), 7 (n per eval
set in `--dry_inputs`), 8 (pt999_summary output unchanged; every task script executed) covered. vii: every task prints
decoded training rows with the prior sites in ⟦ ⟧ and decoded eval rows (`dry_inputs.txt`, returned).

## Known limits and open items
- **Time limits** = 2x the Torch L40S time (A100-40GB never timed): 02:00 / 05:30 / 08:00. xsite53 (340-347)
  measures sec/step first; if Burst is more than 2x slower than L40S, raise them before 354+ / 358+ is submitted
  (delete the untouched task files and regenerate; the generator never overwrites); if it is much faster, lowering
  them only frees guard reservation. The tick does not retry TIMEOUT; a continuation would need a new task that copies
  the checkpoint from `llmadd_runs/<task>/out/<tag>/` (not implemented).
- **190M on a 40 GB A100**: peak memory unmeasured (no LM-loss chunking at bs 32: ~20 GB of logits work + 3 GB of
  weights / Adam + checkpointed activations). An OOM fails in the first steps (cheap, visible in the log); the remedy
  `--lm_chunk_tokens 8192` changes the summation order, so all pt2s190 cells would move to it (new task names).
- **Budget**: ~138 A100-h planned (18 + 48 + 72) on top of xsite53; the guard reserves the full limit while in flight
  (at most 12 x 8 h = 96 h). If only ds_ga_1011 (~142 h) is usable, this batch alone nearly exhausts it, and pt2long
  (the headline) is submitted last, so it is the block that would starve.
- **Burst disk**: 72 saved models (48 x 211 MB + 24 x 761 MB ~ 28 GB) plus up to 12 checkpoints in flight (0.6-2.3 GB);
  scratch quota on Burst unknown (332-data prints `df`).
- **pt2long**: eval contexts and val bpb come from the 3B val.bin -> not comparable with pt2b; the 3B train text also
  contains natural additions with held-out pairs (rule 5; count them afterwards). Window and evals scale with the
  budget (window = 5-40% of 1.2e9 tokens), so pt2long is a different schedule, not a continuation of pt2b.
- **Winner's curse**: best lr and the test use the same 4 seeds; the confirm seeds (4-7) are conditional (rule 2).
- **Pinned trainer, kill window**: 64abc19 deletes the checkpoint before saving the model; a kill in those seconds leaves
  a finished run JSON and no model -> offline_lm skips it, the check fails (exit 4) and the read-out lists the check.
  The run's curve is still returned; rule 6 says the comparison is not read until a re-run (EXTRA) with a passing
  check replaces it, which the read-out then takes.
- **pin_run.sh** (infra piece) has a narrow race when two first-wave tasks find a damaged pinned tree at once; a task
  hit by it fails before training and needs a re-run (EXTRA).
- **Shared numbers**: the 6 cells of a block-seed share a task number (72 tasks in 12 numbers); names stay unique.
- `pt999_summary.py` is now also edited here (2 lines + docstring); pieces that change it concurrently must merge.

## For the user to decide (before the push)
1. **lr grids vs "every arm (the prior too) gets its own best lr"** (audit finding 5; a deviation, not decided here).
   pt2b gives both arms 3 lrs. pt2s190: none 3 (6e-4 / 1.2e-3 / 2.4e-3), win-ctx 2 (6e-4 / 1.2e-3), win-text 1 (6e-4).
   pt2long: none 2 and win-ctx 2 (1e-3 / 2e-3), win-text 1, win-shuf-ctx 1 (a control at win-ctx's lr 1e-3). A 2-point
   grid never brackets: the read-out calls it "closed" only when both lrs are within 2 points, otherwise "open", which
   means a serial second wave of conditional cells. The optimum lr falls with the token budget (plan, lowshare note),
   yet pt2long has nothing below 1e-3. Options (planning A100-h, 4 seeds each):
   a. add now, in BLOCKS (first wave; the generator's 72 / 18 asserts become 84 / 21): pt2long none@5e-4 and
      win-ctx@5e-4 (24 h), pt2s190 win-ctx@2.4e-3 (8 h): +32 h;
   b. also give win-text a second lr (pt2s190 @1.2e-3, 8 h; pt2long @2e-3, 12 h): +20 h more;
   c. keep the grids and accept a serial second wave through EXTRA for every open bracket; write the deviation in
      the paper's methods ("win-text and win-shuf-ctx at a single lr").
2. **Budget-anchored headline** (finding 6). pt2long's window is 5-40% of 1.2e9 tokens = 6e7-4.8e8, 4x later in
   tokens than pt2b's 1.5e7-1.2e8, and the cosine decays over 1.2e9, so the ratio is a property of this budget.
   Option: a token-anchored arm, win-ctx with `--window 0.0125,0.1` at 1.2e9 (the pt2b window in tokens), 4 seeds,
   12 A100-h (needs a new arm in the generator: window parameter in WANT / the check's events; its DRY rows are the
   existing 3B ctx ones). Without it the ratio is written "at a 1.2e9-token budget (window 0.06-0.48e9 tokens)".
3. **Order** (budget): pt2long, the headline, is submitted last. If only ~142 h are usable, renumber so pt2long runs
   before pt2s190 (cheap now: nothing is pushed).
4. **Conditional cells** (open-bracket lrs, confirm seeds 4-7): go into EXTRA when the results call for them; seeds 4-7
   first need their DRY row hashes computed on Torch.

## 2026-10-16 amendment (before any submission or result)
- Order: pt2b (350-353), then pt2long (354-357), then pt2s190 (358-361). The longest runs start earlier, so the batch
  ends sooner (user: maximise parallelism).
- EXTRA, decided up front (user: every arm, the prior too, at its own best lr):
  - pt2long none and win-ctx at 5e-4 (362-365). The optimum lr falls with the training horizon, so the long budget's
    bracket is extended downwards.
  - pt2s190 win-ctx at 2.4e-3 (366-369).
- The best-lr rules are unchanged and run over the wider grids. 84 tasks in total.

## 2026-10-16 amendment 2 (fresh re-audit of HEAD dbd42e6; before any GPU task was submitted)
- The added lr cells now sit inside their blocks, so no budget stop can leave an lr grid half-run, and the long runs go
  before pt2s190:
  - pt2b (350-353): none at 1e-3, 2e-3, 4e-3, 8e-3; win-ctx at 1e-3, 2e-3, 4e-3. none@8e-3 was added: the bracket is
    expected to be open at 4e-3.
  - pt2long (354-357): none and win-ctx at 5e-4, 1e-3, 2e-3; win-text and win-shuf-ctx at 1e-3.
  - pt2s190 (358-361): none at 6e-4, 1.2e-3, 2.4e-3; win-ctx at 6e-4, 1.2e-3, 2.4e-3; win-text at 6e-4.
  - 88 tasks in total; EXTRA is empty again.
- pt2long time limit 10 h, i.e. 2.5 x the L40S time. A TIMEOUT is never retried, and the A100-40GB is untimed until
  xsite53 reports.
- Accounts: ds_ga_1011 now goes first, because it is known to run GPU jobs. Task 334 is a 15-min GPU job pinned to
  ds_ga_1006 (new '# account:' header in tick.sh) and shows early whether that account starts jobs at all.
- Still open, to be fixed before any result is read (tools written and checked on fake data):
  - Selection criterion: the pt2long headline is a t90 ratio, but the best lr is chosen by area. The headline will
    pick each arm's lr by median t90 (censored at the budget) and report the area-best lr next to it.
  - Shared definitions with runs/ptlr/AUDIT.md: diverged, survives tuning, N x, tie-break.
  - Rule 6 (integrity failures withhold verdicts) enforced in code.
  - Area and t90 also reported without b11, because b11 pairs occur in the natural text.

## 2026-10-16 amendment 3: the read-out's open items closed in code (before any result)
Written and tested before any pt2main result existed (no `results/` or `status/` entry for 350-361; origin at 7672427).
`tools/readout_rules.py` is now the single definition set of the Batch 1 read-outs, imported by
`tools/pt2main_summary.py` and `tools/ptscale_summary.py` (Torch ptlr; runs/ptlr/AUDIT.md points here too): area,
t_theta, diverged, unstable, best lr + tie-break, open bracket, "survives tuning", "N x", the seed bootstrap and the Burst
re-run rule. `pt999_summary.py` keeps its own area / t90 on purpose (double path); its output is byte-identical
(checked on pt999 + ptsize999, frozen copies and live).
Changes against rules 1-6 above (the read-out's own audit, same day, before any result, split them into what the code
applies and what waits for the user; the code applies every rule as pre-registered):
- Rule 2 ("survives tuning"), as pre-registered: seeds 0-3 (mean > 0, >= 3/4 wins, paired t p < .05) and the confirm
  seeds 4-7 when both cells have them; without them a gap >= 10 prints 'SURVIVES tuning', a gap < 10 'survival pending'
  with the EXTRA trigger. **PROPOSED, pending the user's sign-off** (printed next to the verdict as '[PROPOSED shared
  rule ...]'): the ptlr reading, where a paper claim needs the confirm seeds 4-7 whatever the gap ('survives
  (unconfirmed)'); with it, a gap >= 10 would need conditional cells that rule 2 does not trigger.
- Rule 3 (headline): each arm's lr = the lowest median t90 over its grid, censored runs at the budget (ties -> the higher
  mean area, then the lower lr); the area-best lr is printed next to it (amendment 2's open item). "N x" = median t90
  none / median t90 win-ctx; a run that never reaches 0.9 counts as +inf; a median is censored when >= (n + 1) // 2 runs
  are (4 seeds: 2): '>=' when the none median is censored, no ratio when the win-ctx median is (as pre-registered) or
  both are. **PROPOSED, pending the user's sign-off** (printed as '[PROPOSED shared rule ...]'): '<=' instead of no
  ratio when only the win-ctx median is censored (the ratio is then an upper bound). With every censored run at the
  budget this median equals the Kaplan-Meier median (midpoint convention) whenever it is not censored; the code asserts
  it on every call. The 90% CI (10000 paired seed bootstraps, default_rng(0)) is printed with 'n=4: indicative only'
  and the shares of resamples with a censored median.
- Rule 5: t90 is also printed without b11 (5-bucket mean): a t90_5 column and a descriptive headline line. The table's
  t90* columns are pt999_summary's convention (median over the seeds that reach 0.9, k/4), labelled as such: not the
  headline's median t90 (censored runs at the budget).
- Rule 6 enforced: a counted run with a missing / failed check.json, a step-0 lm more than 1e-3 off its block-seed median
  (or the same-seed Torch value), or no valid full-val offline_lm entry (|orig_minus_logged| < 0.002 unless diverged)
  makes every verdict using its cell print 'WITHHELD: <reason>' (best lr of its arm, best-vs-best, headline, control);
  the cell's numbers are still printed. A diverged run (first non-finite lm / val_loss / geo / grad_prior) stays data.
- A results dir of a pt2 block that the generator does not make (a re-run not yet in EXTRA) is printed as IGNORED.
- A best-vs-best verdict whose none or prior arm has an open bracket (rule 1) also prints 'NOT FINAL' (the conditional
  cell beyond the edge decides); added by the read-out's audit, the verdict itself is unchanged (base test: pt2s190).
Tests on fake results (scratch `readout/fake_pt2main.py`: all 88 tasks from real Torch run JSONs, pt999 2pct-none /
ptsize999 190m-2pct-none seeds, held-out curves replaced by designed piecewise-linear curves; designed -> printed):
- base: pt2b none best 2e-3 by the offline tie-break over 4e-3 (0.7 points higher), win-ctx 2e-3, +56.5 -> 'SURVIVES
  tuning (gap >= 10)' with the proposed 'survives (unconfirmed)' next to it; pt2long area-best none 1e-3, t90-best 2e-3
  (medians 1200 / 493 / 444M), win-ctx 1e-3; headline 3.34x (designed 3.345; the old read-out's area-best lrs gave
  3.72x = designed 493 / 132.66), without b11 3.27x (3.270); control +19.9 4/4; win-ctx - none +8.4 -> 'survival
  pending' + EXTRA note; win-text +1.1 3/4 p .11 -> does not survive; pt2s190 none open bracket at the upper edge,
  win-ctx 6e-4 by the tie-break, win-text 2/4 -> does not survive. Diff against the previous read-out: only the
  intended lines (header, t90_5 column, headline lrs / bound marks / CI text, the PROPOSED brackets).
- cens (none censored in 2 of 4 at every lr): '>=6.23x' (designed 826.2 / 132.66 = 6.228), without b11 '>=6.09x';
  censw (win-ctx censored in 2 of 4): 'no ratio' as pre-registered, with the proposed '<=0.66x' (0.663) next to it;
  censboth: 'no ratio'.
- lm0torch (every pt2b cell of seed 1 starts +0.005 off, so the block-seed median moves along): only the same-seed Torch
  check catches it -> 7 INTEGRITY lines, pt2b best lrs and best-vs-best WITHHELD, pt2long / pt2s190 unchanged. A
  results dir 362-pt2b-lr2e-3-none-s0 outside EXTRA -> 'IGNORED' note (was silent).
- withheld (a failed check in pt2b none@4e-3 s2, step-0 lm +0.01 in pt2long win-ctx@5e-4 s2, no offline entry in
  pt2s190 win-text s1): best lr pt2b none, pt2long win-ctx, pt2s190 win-text and every verdict using them WITHHELD with
  the reason; the control and the other arms unchanged. The old read-out printed all verdicts (a NOTE / WARNING only).
- extra (tools copy with EXTRA = confirm seeds 4-7 of pt2b none@2e-3 / win-ctx@2e-3 and a re-run of an unfinished
  task; test-only DRY placeholders): the re-run counts (area 5.0 = design), confirm +56.6 4/4 -> 'SURVIVES tuning
  (confirmed on seeds 4-7)'.

## 2026-10-17 conditional cells (the pre-registered open-bracket rule, applied before any best-lr verdict)
- pt2b none: the area still rises at the top of the grid (8e-3: 23.7, n=3, vs 4e-3: 14.7), so none@1.6e-2 is added for
  seeds 0-3 (tasks 362-365).
- pt2b win-ctx: best at the bottom of the grid (1e-3: 70.2 vs 2e-3: 69.8), so win-ctx@5e-4 is added for seeds 0-3
  (tasks 362-365).
- The read-out marks both arms NOT FINAL until these runs are in.

## 2026-10-17 conditional cells 2 (open-bracket rule, before any best-lr verdict on these blocks was recorded)
- pt2long none: the area is highest at the top of the grid (2e-3: 69.2 > 1e-3: 66.9), so none@4e-3 is added for seeds 0-3
  (tasks 366-369).
- pt2s190 none: the same holds (2.4e-3: 22.2 > 1.2e-3: 19.9), so none@4.8e-3 is added for seeds 0-3 (tasks 366-369).
- pt2b none's bracket is now closed (8e-3: 23.1 > 1.6e-2: 15.2).

## 2026-10-17 confirm seeds (user: confirm the headline cells now)
- pt2b none@8e-3 and win-ctx@2e-3, seeds 4-7 (tasks 370-373): the arms' best lrs; none's bracket is closed.
- Their DRY row hashes were computed with the same method; that method reproduced the known seed-0 value of each key
  (53M pretrain_data, 190M pretrain_data, 53M pretrain_data_3B).
- The step-0 lm references for 53M seeds 4-7 come from pt999 2pct-none.
- pt2long and pt2s190 confirm seeds follow once their open brackets close (tasks 366-369).

## 2026-10-17 lambda sweep at the 2% main setting (user)
- Before this, lambda had been swept only at 7% with the gate (F82: lambda 1 >= 5 > 20, and lambda 1 had the lower LM
  cost).
- Added: win-ctx at its best lr (2e-3) with lambda 0.3, 1 and 2, seeds 0-3 (tasks 374-377). Each is paired by seed with
  pt2b-lr2e-3-win-ctx (lambda 5) and pt2b-lr8e-3-none on the same hardware.
- Read-out: area, t90 and offline bpb per lambda; the best lambda by area, with ties within 2 broken by lower offline bpb
  (the same rule as the best lr).
- The grid is log-spaced (0.3 / 1 / 2 / 5). 0.5 follows only if the optimum falls between 0.3 and 1.
- 2026-10-18: pt2s190 none still rises at the top of its grid (4.8e-3: 26.8 > 2.4e-3: 22.2), so none@9.6e-3 is added for seeds 0-3 (tasks 378-381).
