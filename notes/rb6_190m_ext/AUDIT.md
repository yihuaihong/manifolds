# rb6_190m_ext — 190M rainbow follow-up: one more lr, the clock prior's attribution, none at high lr (2026-10-22)

User approval (relayed by the coordinator): "我觉得可以跑一下" for the three cheapest follow-ups of the rb6_190m results
audit (notes/batch1_build/audit_rb6_190m_results.md, "最便宜的补跑"; FINDINGS F88 190M addendum). Written to this file
BEFORE any job of this batch was submitted and before any of its results exists.

## Questions (descriptive, seeds 0-3)
All readings below are descriptive on seeds 0-3 (the main batch's seeds, on which its lrs were selected): no
confirmatory claim, no size-trend claim (only 190M runs are read; F87), no "best lr" claim.
- **Q1 (lr)**: does the clock gap all6 - only-numbers (main batch at 6e-4: -34.2 [-40.2, -40.1, -25.7, -30.7], 0/4)
  persist at lr 1.2e-3? Both lrs are reported side by side for every domain.
- **Q2 (attribution)**: at the main batch's lr 6e-4, is the clock gap from the clock prior? Clock of only-clock vs
  only-numbers vs none@6e-4; all6-minus-clock vs all6 vs only-numbers.
- **Q3 (none at high lr)**: for S4 (fresh non-id final3; s4_hard area beside it) and dim (strict final3), does any none
  config over {6e-4, 4.8e-3, 9.6e-3, 1.92e-2} leave the floor? The best tested none over these four is then the 190M
  comparison for S4 / dim (labelled "best tested").

## Cells (190M, the rb6_190m design except the listed change; seeds 0-3; 24 jobs)
| cell | trainer flags (= make_rb6m_tasks.arm_flags of the arm) | lr | question |
|---|---|---|---|
| all6-lam1@1.2e-3 | `--target helix3 --schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains numbers,months,weekdays,clock,s4,dim --lam 1` | 1.2e-3 | Q1 |
| only-numbers@1.2e-3 | `--target helix3 --schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains numbers --lam 1` | 1.2e-3 | Q1 |
| only-clock@6e-4 | `--target none --schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains clock --lam 1` (= the 53M only-clock arm, Burst tasks 441 / 447 / 453 / 459) | 6e-4 | Q2 |
| all6-minus-clock@6e-4 | `--target helix3 --schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains numbers,months,weekdays,s4,dim --lam 1` | 6e-4 | Q2 |
| none@9.6e-3 | `--target none` | 9.6e-3 | Q3 |
| none@1.92e-2 | `--target none` | 1.92e-2 | Q3 |

Every cell: `--arith_lines 1 --domains numbers,months,weekdays,clock,s4,dim --extra_lines 2` (so the clock LINES are in
the data of every cell, also of all6-minus-clock and the none cells), `--d_model 768 --n_layers 12 --n_heads 12 --d_ff
3072` (190,297,344 parameters, gradient checkpointing on, prior / read-out layers 4, 8), 3e8 tokens = 18,310 steps of
32 x 512, eval every 250 steps (75 evals), `--save_model --ckpt_every 2000`; every other flag = the trainer default
(asserted by the generator: the rendered flag set is exactly this). Tags `rb6tb-190m-2pct-k2-<arm>-lr<lr>`, job dirs
`out/<tag>-s<seed>/`; job names `llmadd_rb6_190m_ext-<arm>-lr<lr>-s<seed>`; seed-major submission order.

**How all6-minus-clock is expressed (no trainer change).** The trainer (pretrain_range.py 93e22bf4c4c0) takes any subset
of `--domains` as `--prior_domains` (asserts: a subset, no duplicate, numbers in it iff `--target` != none) and with
`--lam_rule each` gives every active prior `--lam` (`LAMD = {d: lam if d in prior_domains else 0}`). So
all6-minus-clock = all6-lam1's flags with `clock` removed from `--prior_domains`: the clock prior has lambda 0, is never
on / applied / logged / weighted; the other five keep lambda 1 each (total prior weight 5 instead of 6, the design's
intent: the same per-prior lambda). The generator adds the arm to the snapshot module's SPEC table in memory as
`("helix3", ("numbers", "months", "weekdays", "s4", "dim"), (), "1", ...)` (the tuple form of the other arms), so
arm_flags / intent / the dry-input block / CHECK_PY use the same code paths as every other arm; the files are unchanged.

## lr provenance (pre-registered caveats)
- 6e-4 / 4.8e-3 are rb6_190m's lrs reused from pt2s190 (numbers area, seeds 0-3). 1.2e-3 (prior arms) and 9.6e-3 /
  1.92e-2 (none) are the next grid steps (x2; the 505M sweep's grid). Two lrs per prior arm and four none lrs are still
  not a tuned bracket: no cell is "at its best lr".
- Q3's "best tested" none is the best of four lrs ON THE SAME seeds 0-3 it is compared on (optimistic for none; a
  winner's curse in none's favour); it is not "none at its best lr" and not comparable with A8.3 (53M, the S4-tuned
  none@1.6e-2 on the fresh seeds 8-15). At 53M the S4-tuned none was 1.6e-2; 1.92e-2 may diverge: a diverged run is data.
- Seeds 0-3 share the init and the numbers lines with the pt2s190 lr-selection runs (as rb6_190m): every row is
  "descriptive, no confirmatory claim".

## Read-out (pre-registered; tools/rb6scale_ext_summary.py on branch scale-ladder, written and tested before any result)
A thin second wrapper (rb6scale_summary.py and rainbow6_summary.py are NOT changed: the rb6_190m read-out stays byte-
identical, tested). Command (after all 24 jobs; mid-batch `--allow_partial` -> every line that misses a seed is
PROVISIONAL n=k of 4; strict mode refuses):
```
python tools/rb6scale_ext_summary.py /scratch/yh6210/llmadd_torch/runs/rb6_190m/out \
  /scratch/yh6210/llmadd_torch/runs/rb6_190m_ext/out --json rb6_190m_ext.json
```
**Metrics** (rainbow6_summary's A5 READ, unchanged): numbers = 6-bucket held-out area; months / weekdays = tier-E final3;
clock = held_wrap final3; s4 = held_fresh_nonid final3; dim = the STRICT set final3; s4_hard area (A8.3's metric) with
A8.2's "learned" (last eval vs the exact Latin-filler chance, one-sided binomial alpha .01) k/4; offline bpb x1e3; seen.
Per cell: mean over seeds 0-3, per-seed values, the seed-level 90% t interval and gate verdict (FLOOR / BORDERLINE /
MEASURABLE / CEILING vs [chance + 5, 90]); A5's INVALID rule for none cells (a domain's seen final3 < 80 in any seed);
the gradient split of every prior cell (LM share, prior / LM, clip, share prior > LM; FLAG at a median prior / LM > 1).
Paired rows: per-seed differences, paired t (df 3), exact sign-flip p, 95% CI, BH q .05 within each family; all
two-sided except as stated.

**Q1** (family Q1: all6 - only-numbers per domain at 6e-4 and at 1.2e-3, 12 rows; family Q1-lr: 1.2e-3 - 6e-4 within
each prior arm, 12 rows, descriptive). Reading on clock, with d(lr) = mean of all6 - only-numbers and r = d(1.2e-3) /
d(6e-4) (d(6e-4) = -34.2):
- r >= 2/3 and at least 3 of 4 seeds negative at 1.2e-3 -> **PERSISTS at 1.2e-3**;
- r >= 2/3 but fewer than 3 seeds negative -> UNDETERMINED (seed-inconsistent);
- r <= 1/3 (incl. a sign flip) -> **LARGELY GONE at 1.2e-3** (the 6e-4 gap then depends on the lr);
- otherwise -> PARTIAL / UNDETERMINED.
Either way: "two lrs, not all6's best lr"; the other domains are reported side by side, no reading.

**Q2** (family Q2, clock only, 6 rows: only-clock - none@6e-4, only-clock - only-numbers, only-numbers - none@6e-4,
all6-minus-clock - all6, all6-minus-clock - only-numbers, all6 - only-numbers). Reading, with seed means mc, a6, on of
all6-minus-clock, all6-lam1, only-numbers (all @6e-4) and f = (mc - on) / (a6 - on) (the share of the gap that removing
the clock prior leaves):
- f >= 2/3 -> **mc ~ all6: the gap is NOT from the clock prior** (removing it leaves the gap);
- f <= 1/3 -> **mc ~ only-numbers: the clock prior (together with the other four domain priors) is involved**;
- 1/3 < f < 2/3 -> **UNDETERMINED** (in between);
- |a6 - on| < 10 points -> UNDETERMINED (no gap to attribute).
Beside it the per-seed f_s and how many seeds give the same reading (descriptive). The f <= 1/3 reading does NOT mean
"the clock prior alone harms clock": that is only-clock's row (only-clock - only-numbers, only-clock - none@6e-4),
reported descriptively without a reading (53M context only: only-clock 49.9 vs only-numbers 89.7; no size comparison).
Family Q2-other (descriptive): all6-minus-clock - all6 on the other domains + s4_hard + bpb; only-clock - none@6e-4 on
the other domains + bpb.

**Q3**. For s4 and dim, per none config over {6e-4, 4.8e-3, 9.6e-3, 1.92e-2} (seed-level gate, seeds 0-3):
- verdict FLOOR (upper bound < chance + 5) -> **AT THE FLOOR**;
- seed-level lower bound >= chance + 5 (MEASURABLE, CEILING, or BORDERLINE above the floor) -> **LEFT THE FLOOR**;
- otherwise (BORDERLINE straddling the floor) -> **UNDETERMINED**;
- INVALID (seen < 80) -> not counted.
Answer per metric: YES if any valid config LEFT THE FLOOR; NO if every valid config is AT THE FLOOR; else
UNDETERMINED. s4_hard: area and A8.2 learned k/4 per config, beside the s4 answer (no separate gate).
**Best tested** none per metric (s4 final3, s4_hard area, dim strict final3): the highest mean over the four configs,
INVALID configs excluded, a tie (1e-9) -> the lower lr; labelled "best tested over {6e-4, 4.8e-3, 9.6e-3, 1.92e-2},
chosen on these seeds 0-3: optimistic for none". Family Q3 (6 rows): all6-lam1@6e-4 - best tested none and
all6-lam1@1.2e-3 - best tested none, on s4, s4_hard, dim; s4 / s4_hard one-sided (greater) iff that none's s4 gate is
FLOOR (A5's F1b rule), dim two-sided. PROVISIONAL whenever any of the four none cells misses a seed.

**Run identity and integrity** (before any table): every run's JSON against its intent by rb6scale_summary.torch_identity
(every trainer argument incl. lr, model dims, target, prior_domains, lam; code_sha 93e22bf4c4c0, toy_struct d182ce548bcc,
n_params, gradient checkpointing, layers; lam_domains; the active / shuffled prior sets from domains.<d>.prior / shuf;
the logged multiplier gnorm_prior / gnorm_true = 1 for every active prior) and check.json PASS; a failing run is
EXCLUDED with the reason. rainbow6_summary.check over all 40 runs (hits = accuracies double path, layout, window, the
trainer's metrics = the summary's; per seed the same data_sha at every eval and the same step-0 lm across all 10 cells,
main batch included). Missing / excluded / unfinished -> PROVISIONAL (n=k of 4); strict refuses. A diverged run is data
(A6: its values over all evals count, its bpb NOT TESTED).

## Code
- Trainer and job code = runs/rb6_190m's exactly: runs/rb6_190m_ext/code = git archive of tools/ at c6dcb6e (branch
  scale-ladder) with rb6_190m's COMMIT file; the generator asserts COMMIT equal and every file byte-identical to
  runs/rb6_190m/code (full sha256 over the tree), plus WANT_SHA (pretrain_range 93e22bf4c4c0, toy_struct d182ce548bcc,
  pretrain_helix 25d5a35e6f17, numlib d09bb47b04c2, digit_split_data 742b922c8125, run_guard b944dc859c7a, offline_lm
  226f4cac70ea); every job re-checks the 7 shas (exit 3).
- Generator: llmadd_torch/make_jobs_rb6_190m_ext.py = a copy of make_jobs_rb6scale.py (which, with rb6scale_refs.json and
  the 505M files, stays read-only and unchanged: sha256 recorded before and after). Changes: the batch table; the snapshot
  rule (above); `register()` (all6-minus-clock in memory; only-clock / only-numbers / all6-lam1 / none asserted = the 53M
  tables); ARM_FLAGS (the literal flag strings, a second path) and ACTIVE (the intended active priors) asserted against
  arm_flags / intent (active, lam_domains, prior_domains, target, lam, lam_rule, schedule); for all6-minus-clock clock
  off and five priors on, for only-clock clock on and numbers off; a gate on the batch's CPU references
  (scale_work/rb190ext_refs.json); --submit refuses without this AUDIT.md.
- References: the per-seed DRY row hash and step-0 lm are rb6_190m's (rb6scale_refs.json, read-only): the rows and the
  step-0 forward are arm- and lr-independent (the prior is off at step 0; lr acts after it) - verified for this batch
  below, not assumed.
- Read-out: tools/rb6scale_ext_summary.py (new file, branch scale-ladder).

## Checks inside every job (exit 3 before training, exit 4 after) — rb6_190m's, unchanged
HOLD kill switch (`touch runs/rb6_190m_ext/HOLD`: a pending job exits at start); the 7 snapshot shas; COMMIT names the
trainer; A100 guard (`*A100*`); data sha256 (train.bin 4de2ed19..., val.bin d6f744a9...); `--dry_inputs` with the job's
own ARGS: the model line, the numbers target line (helix3 1000 x 10 / absent), for each of the 6 domains prior / shuf /
lam in the `domains:` line (all6-minus-clock: `clock:false:false:0.0`, the other five `true:false:1.0`; only-clock:
`clock:true:false:1.0`, the other five `false:false:0.0`), the numbers line first, the content scan, the 88 DRY_GREP
lines' sha256 = the seed's reference; training; offline_lm.py (full val); make_rb6m_tasks.CHECK_PY with the cell's intent
-> check.json (CHECK_PY is written for any set of active priors: every active prior on exactly in the window, applied on
>= 95% of on-steps, geo logged, weighted, multiplier = lambda; every inactive prior never on / applied / logged /
weighted; numbers off -> geo == 0 at every eval; data_sha after step 0 and step-0 lm = the references; the rest as
rb6_190m).

## Tests before submission
Scripts in the session scratchpad rb190ext/ (copies run from llmadd_torch/scale_work/rb190ext_scripts/, since the
scratchpad is node-local); outputs llmadd_torch/scale_work/rb190ext* (CPU jobs 20654128 on cs655, local runs on cl014).
| test | result |
|---|---|
| snapshot | runs/rb6_190m_ext/code = git archive c6dcb6e + rb6_190m's COMMIT; tree sha256 = runs/rb6_190m/code file for file (asserted at every generator call); WANT_SHA 7/7 |
| CPU --dry_inputs, all 24 (arm, lr, seed) (refs_ext.py dry, the jobs' own ARGS; table: rb190ext_table.txt) | 24/24 rc 0; DRY_GREP rows (88 lines) sha256 = rb6_190m's of the seed (= the Burst 53M rows + the 190M model line) 24/24; the eval lines = the main batch's dry log 24/24; the WHOLE dry log = the main batch's dry log of the seed except the per-domain prior / shuf / lam / target keys and the target-features line 24/24 (decoded training and eval rows identical; rows_side_by_side.txt); DOMS = make_rb6m_tasks.DOMS 24/24; the 'domains:' line and the '[<d>]' lines = the intent for every domain 24/24: all6-minus-clock clock prior False lam 0.0, numbers / months / weekdays / s4 / dim prior True lam 1.0, numbers target helix3; only-clock clock True 1.0, the other five False 0.0, numbers target none; target-features line present iff helix3 |
| step-0 lm (refs_ext.py step0; CPU bf16, no_grad) | CPU bf16 kernels differ across hosts (the stored LM0 host vs cs655 / cl014: 2.5e-5 - 8.2e-5 for every arm, the main batch's too), so arm-independence is tested within a host: cs655 s0 all6-minus-clock = only-clock, s1 all6-lam1@1.2e-3 = none@1.92e-2, s3 none@9.6e-3 = only-numbers@1.2e-3; cl014 s0 none@6e-4 = all6-minus-clock = only-clock, s2 none@6e-4 = all6-minus-clock (all bitwise); every value within 8.2e-5 of the stored LM0 (the jobs' check: 1e-3); data_sha after step 0 = DSHA0 every record. (A further same-host job, 20655443, may still run later; it writes its own directory and is not needed by the gate.) |
| e2e_ext.py: the rendered job script end to end on CPU (tiny overrides, --grad_ckpt on, fake A100 nvidia-smi; e2e.py's method) | 6/6 job rc 0 and every CHECK_PY check PASS: all6-minus-clock s1, s2; only-clock s2; all6-lam1@1.2e-3 s2; only-numbers@1.2e-3 s1; none@1.92e-2 s2. Run JSON per domain (second path): all6-minus-clock clock prior False, lam 0, used 0, geo_clock never logged, no gnorm_prior_clock; the other five prior True, lam 1, applied 0.35 (= the window), gnorm_prior / gnorm_true = 1.000000; only-clock: clock on (multiplier 1.000000), the other five off (incl. numbers: geo 0); none: all off. Tiny step-0 lm bitwise equal across arms of a seed (s2: all6, all6-minus-clock, only-clock, none 11.52961254119873) |
| jobcheck_ext.py (the 24 rendered jobs vs the submitted rb6_190m jobs) | bash -n 24/24; the line diff to the main-batch job of the same seed and nearest arm only in the allowed lines (name, header, R/OUT/TAG, ARGS, the target assert, the per-domain loop, the CHECK_PY call / intent) 24/24; the CHECK_PY heredoc byte-identical 24/24; ARGS = the template's except the arm flags / lr / out / tag 24/24; intent keys changed only active / lam_domains / events / on_frac / lam_rule / args (lr, tag, target, prior_domains, lam, schedule, lam_rule) |
| read-out tests (ro_ext_test.py; relabelled real runs) | RO EXT TEST PASS (48): unit tests of the Q1 / Q2 / Q3 rules (boundaries, sign flip, no gap, seed-inconsistent, INVALID, ties); base (16 real + 24 relabelled: all6 / only-numbers @6e-4 as @1.2e-3, all6 edited to only-clock and to all6-minus-clock, 53M none@8e-3 / @1.6e-2 with the 190M size facts as none@9.6e-3 / @1.92e-2): the 4 main cells = the stored rb6_190m read-out (values, means, gates), Q1 6e-4 rows = the stored S3, Q1 1.2e-3 rows = 6e-4 rows -> PERSISTS, only-clock rows = S3 / S1 clock, all6-minus-clock = all6 -> f 1 '~ all6' 4/4, none@9.6e-3 = the stored 53M none@8e-3 cell, none@1.92e-2 = an independent recomputation, Q3 labels = the CI rule, best tested = argmax; scipy t p; missing -> strict refusal / PROVISIONAL; wrong lr, clock prior on in all6-minus-clock, clock prior off in only-clock, failed check -> EXCLUDED; diverged -> data, bpb NOT TESTED; data_sha mismatch -> AssertionError; the rb6_190m read-out (AUDIT command) byte-identical to the stored output (text and JSON) |
| read-only files | make_jobs_rb6scale.py, rb6scale_refs.json, make_jobs_pt2s505.py, pt2s505_refs.json, runs/pt2s505/pick_confirm.py: sha256 before = after (scratchpad rb190ext/readonly_sha_*.txt); rb6scale_summary.py e807e43be36b unchanged |
| sbatch --test-only | 24/24 accepted (a100_tandon; start estimates 2026-10-23 19:18 - 21:36: the queue holds the 505M batches) |

## Audit status (experiment-audit, minimal mode)
ii asserts (generator + every job, above); CPU dry runs of every (arm, lr, seed) with the decoded rows compared to the main
batch's (vii); known answers (DRY rows / step-0 lm = the main batch's; read-out on relabelled runs); the fresh adversarial
review (v) is launched by the coordinator after submission. i / iii / iv: – (minimal mode; the metrics and the job
pipeline are rb6_190m's, already audited).

## Submitted (2026-10-22 ~20:10 EDT)
24 jobs 20665778-20665837 (SUBMITTED_<name> flags hold sbatch's output), seed-major; account torch_pr_520_tandon_priority,
a100_tandon, QoS gpu48, `--nice=20`, 1 GPU, 8 CPUs, 64G, `--time=05:30:00` (rb6_190m's; its runs took 2.4 h), `--requeue`.
Planned ~24 x 2.5 = 60 A100-h; ~18 GB of saved models (/scratch had ~250 GB free). Code c6dcb6e (= runs/rb6_190m/code),
trainer 93e22bf4c4c0; read-out tools/rb6scale_ext_summary.py 00f76a1fc62d (scale-ladder 9ac63ee, not pushed); logs
llmadd_torch/logs/llmadd_rb6_190m_ext-<arm>-lr<lr>-s<seed>_<jobid>.out. Kill switch: touch runs/rb6_190m_ext/HOLD
(pending jobs exit at start).

## Amendment E1 (2026-10-22, after the fresh build audit notes/batch1_build/audit_rb6_190m_ext_build.md, BEFORE any result of this batch)
- M1 (Q1 scope): the prior arms' main lr 6e-4 is the bottom of pt2s190's grid; this batch adds only 1.2e-3 above it.
  A PERSISTS / LARGELY GONE reading answers "at 1.2e-3", not "the lr was tuned". Lower lrs (3e-4) are untested unless
  added later; the read-out states this beside Q1.
- M2 (Q2 meaning): f >= 2/3 means only "with the other four domain priors on, the clock prior is not necessary for the
  gap". It does NOT separate redundancy, dose (six -> five priors) or global-clip effects, and there is no "minus another
  prior" control. f <= 1/3 means "the clock prior is involved (with the others)", never "the clock prior alone causes it".
- M3 (Q2 consistency gate, a tightening): a Q2 verdict other than UNDETERMINED additionally needs >= 3/4 seeds on the
  same side of the threshold (per-seed f), as Q1 already requires; otherwise UNDETERMINED.
- M4 (Q3 wording): "leaves the floor / stays at the floor" is about the tested none configs at 3e8 tokens; it is never
  "learnable / not learnable without a prior". The Q3 answer line must print this. The 'YES' of Q3 is "any of four
  configs" with no multiplicity correction: say so.
- none@1.92e-2 may diverge (warmup 183 steps). NaN runs count as data (INVALID if seen < 80); a soft blow-up with
  finite loss is checked on the lm curve before reading.

### E1 implemented (2026-10-22, before any result; jobs still pending)
tools/rb6scale_ext_summary.py 00f76a1fc62d -> b5f3b5e6156e (the E1 version, branch scale-ladder):
- M3: q2_reading(): a "~ all6" / "~ only-numbers" verdict needs >= 3/4 paired seeds with f_s on the same side of the
  threshold (>= 2/3 resp. <= 1/3; a seed without a 10-point gap does not count), else "UNDETERMINED (seed-inconsistent)".
- M1 / M2 / M4: the Q1 scope note (only 1.2e-3 added; 3e-4 untested), the Q2 meaning note (not necessary / involved with
  the other four on; redundancy, dose, clip not separated; no minus-another-prior control) and the gradient split of
  all6-lam1@6e-4 / all6-minus-clock@6e-4 beside Q2, the Q3 meaning note on both answer lines (tested none configs at 3e8
  tokens, never learnable / not learnable; YES = any of four, no multiplicity correction; INVALID configs not counted).
- Build audit L3: r, f and every per-seed r_s / f_s outside [0, 1] are flagged REVERSED (< 0) / OVERSHOOT (> 1) beside
  the unchanged pre-registered label.
- Build audit L6: rainbow6_summary.check's finiteness assert is relaxed only for none@1.92e-2 (arm none-lr1.92e-2); a
  non-finite value in any other cell stops the read-out. none@1.92e-2's lm curve is summarised per seed (lm at steps 0 /
  1000 / 5000 / 9000 / 13000 / last, final val_loss, minimum, the largest rise above the running minimum, flagged
  "inspect" above 0.5 nats, DIVERGED step).
- Tests: ro_ext_test.py 58/58 PASS: + unit tests of q2_reading (2/4 seeds past the line with the mean past it ->
  UNDETERMINED; 3/4 -> the verdict; in between unchanged) and range_flag; + fake scenario q2inconsistent (all6-minus-clock's
  clock grafted from none@6e-4 at s0, s1 and only-numbers@6e-4 at s2, s3: mean f 0.90 but 2/4 seeds -> UNDETERMINED
  (seed-inconsistent), s0 / s1 OVERSHOOT); + nanother (a NaN in all6-lam1@1.2e-3 -> stops); the diverged none@1.92e-2
  scenario prints "DIVERGED at step 9000" in the lm summary; the rb6_190m read-out still byte-identical (text and JSON).
- Copies of this AUDIT.md, the generator, the CPU references and the test scripts: branch scale-ladder notes/rb6_190m_ext/
  (pushed to origin scale-ladder; not merged into main), so the pre-registration survives a /scratch flush.
