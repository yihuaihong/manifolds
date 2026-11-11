# ptlr_1b — a 1B point on F87's 7% single-prior size ladder, on Torch B200 (ladder-1b; 2026-10-23)

User approval: 「我想用上这个 B200，没问题，我们暂时用这个 setup，先排着」. The coordinator gave the go for stage 2 on 2026-10-23
after the stage-1 smoke, approving this design (ckpt_every 500, --time 5 h, 4 singleton slots, models only at the picks,
the lr grids below, both accounts, the 1B definition). Written BEFORE any job of this batch was submitted (the smoke ran
in scale_work/b200/smoke only).

## Question
F87 (FINDINGS.md, and addenda 1-2) tested whether the 7% speed-up of the helix3 prior survives tuning best-vs-best: win-ctx@best −
none@best by area and R_0.5, each arm at its own best lr, at 53M / 190M / 505M. This batch asks whether it holds at 1B, and
whether a 1B point changes rule 8's size trend (still "undetermined").

## Design = F87's ptlr cells (runs/ptlr/AUDIT.md, make_jobs_ptlr.py), at 1B
- Trainer: F87's snapshot. The batch's code/ comes from `git archive` of llmadd-burst commit 64abc19 (tools/pretrain_range.py
  code_sha a3bce3ab63c1). The generator asserts that the extracted tools/ equal runs/ptlr/code/tools file by file, and every
  sbatch re-checks the sha before starting (exit 3).
- Args: F87's, with only the model and the lr changed, plus --ckpt_every 500 and --save_model at seeds 2-7 (see
  "Deviations"). That means `--data pretrain_data --tokenizer <OLMo-2> --tokenization whole --arith_lines 4`, and everything
  else at the trainer default:
  - 3e8 tokens = 18,310 steps of 32 x 512; eval every 250 steps (75 evals, 150 items per bucket);
  - 1% warmup (183 steps), cosine to 0.1x, AdamW (0.9, 0.95), wd 0.1, clip 1;
  - lambda 5; prior and read-out layers at 1/3 and 2/3 of the depth (12, 24);
  - gradient checkpointing on (auto above 150M); LM loss in one piece (lm_chunk 0, as at 505M); bf16 autocast.

  make_jobs_ptlr1b.py asserts every intended flag, no other flag and no repeated flag (fault-injected).
- Arms: none (`--target none`) and win-ctx (`--target helix3 --schedule window --window 0.05,0.4 --site ctx`). gate-ctx is
  not run at 1B: it is F87's secondary arm and was untuned at 190M / 505M.
- **1B model.** The ladder's sizing rule is the GPT-2 shapes (head dim 64, d_ff = 4d, depth growing with width): 190M = d768 x 12
  (GPT-2 small), 505M = d1024 x 24 (GPT-2 medium). So 1B = **GPT-2 large: `--d_model 1280 --n_layers 36 --n_heads 20 --d_ff
  5120`**.
  - 1,072,259,840 parameters (OLMo-2 blocks, SwiGLU, tied 100,278-row embedding counted once). "1B" is the total, like
    "505M".
  - **N_nonemb 943,904,000** (rule 8's x). The smoke printed `model: 1072.3M parameters; gradient checkpointing on`.
- GPU: B200 only (`--constraint=b200 --comment="preemption=yes" --gres=gpu:1`). A job that lands on anything else exits
  with code 4 before training. So every 1B arm runs on the same GPU type (criterion (f)); the figure states L40S for
  53M-505M and B200 for 1B.

## lr grids (why)
F87's best lrs (area, seeds 0-3):
- none: 53M 8e-3, 190M 4.8e-3, 505M 9.6e-3. At 505M, 4.8e-3 is equal on 8 seeds (57.6 vs 57.5) and 1.92e-2 lost 13 points.
  At 190M, 9.6e-3 lost to 4.8e-3 (56.7 vs 64.3).
- win-ctx: 53M 2e-3 (1e-3 tied), 190M 1.2e-3 (6e-4 tied), 505M 1.2e-3 (3e-4 tied). The prior is lr-flat (spread <= 1.9).

Width grows x1.25 from 505M to 1B, and the usual lr ∝ 1/width expectation moves the optimum down, not up. Grids, seeds 0-1:
- **none {2.4e-3, 4.8e-3, 9.6e-3}**: 505M's two tied lrs plus one halving below. 1.92e-2 is left out (it lost at 505M).
- **win-ctx {3e-4, 6e-4, 1.2e-3}**: 505M's two tied lrs plus the point between (= pt2s505's prior grid).

Extending a grid needs the user.

## Selection and stages
- **A: both arms x 3 lrs x seeds 0-1 (12 runs, about 36 B200-h).** Each arm's lr is picked by F87 rule 1, applied to
  seeds 0-1 (rule 1b below).
- **B: seeds 2-3 at each arm's pick (4 runs).** Submitted automatically:
  - maybe_select.sh runs at the end of every 1B job. The first job to find all 12 stage-A runs complete runs the read-out
    snapshot runs/ptlr_1b/readout (= ladder-1b 4b2d214) with `--json selection_A.json`, then pick_1b.py, then
    make_jobs_ptlr1b.py `--test-only` and `--submit`.
  - It runs once: select.lock, select.done, and a stale-lock takeover.
  - pick_1b.py writes picks.json only if both 1B arms have a best lr (complete, not WITHHELD) and neither is an F87 open
    bracket (an edge pick beating its neighbour by > 2).
  - **An F87 open bracket stops the automation**: select.log names it, nothing is submitted, and extending the grid goes
    to the user.
  - **An edge pick within 2 points** (F87: "not an open bracket") is reported in select.log and the report as **"OPEN
    (edge pick)"**. Stage B still runs at it: it extends no grid.
- **C: confirm seeds 4-7 at the picks (8 runs).** Only if runs/ptlr_1b/APPROVE_C exists (the coordinator's approval of the
  cost). If the file is there when stage A ends, C is submitted with B. Otherwise touch it later and run
  `make_jobs_ptlr1b.py --submit` (it skips SUBMITTED_*).
- **Two copies of every run**, one per account: torch_pr_520_tandon_priority (b200_tandon_prem or b200; can preempt on
  b200) and torch_pr_625_general (b200). tools/run_guard.py lets the first copy compute the run and makes the other exit.
  A requeued job keeps its id and its claim; a copy that starts while the claimant is pending takes the claim over and
  resumes from the ckpt (F87 practice).
- **At most 4 runs at a time.** Job names are llmadd_ptlr1b-slot<k> with `--dependency=singleton`, where k = run index
  mod 4 and both copies of a run share its slot. Logs: logs/llmadd_ptlr1b-<run>_<jobid>.out (`--open-mode=append`, so a
  requeue appends).

## Deviations from F87 (all decided before any result)
1. **Lr selection on seeds 0-1, not 0-3 (cost).** Seeds 2-3 run only at the picks, and the best-vs-best test is on seeds
   0-3 at the picks.
   - The winner's curse is larger than F87's: two of the four test seeds chose the lr. The confirm seeds 4-7 are the guard.
   - F87 rule 1's top-two tie (within 2 points → lower in-training bpb) is applied to the seed-0-1 means.
2. **--ckpt_every 500 (F87: 2000).** This changes only how much a preemption loses (at most about 4.5 min instead of
   18 min). Each ckpt write costs about 8 s, about 3% of the run.
   - The smoke showed the resumed segment's first eval identical to the uninterrupted segment:
     - --stop_after resume: lm, val_loss, geo, cka, ema and prior_on_frac equal to all digits;
     - kill -9 resume: the same except val_loss, which differs by 2e-6.
3. **--save_model only for seeds 2-7 (at the picks).** Stage A saves no model. Its read-out needs only the in-training bpb
   (F87's ptlr cells had no offline bpb either). This keeps /scratch safe: 94% full, about 270 GB free.
4. **GPU B200, preemptible.** On a B200 a run is not bitwise reproducible: two uninterrupted 300-step runs differ in
   every tensor, val_loss differs by 0.010 at step 299, and lm differs from about step 50 on. A resumed run therefore
   matches an uninterrupted one within that run-to-run spread, not bitwise. L40S runs are not compared bitwise with these
   either.
5. Time limit 5 h per segment (a run is about 3.0 B200-h). TIMEOUT does not requeue.

## Pre-registered rules (F87's, with the changes marked)
1b. **Best lr per arm** = argmax of the mean area over seeds 0-1.
    - If the top two are within 2 points, the lower mean in-training bpb wins (8 x 16 val windows at the last eval;
      marked '*').
    - readout_rules.bracket as F87: an open bracket is an edge pick beating its neighbour by > 2. The edge-pick reporting
      is in "Selection and stages" above.
2.  **Diverged and unstable runs stay in the mean with their area.**
    - Diverged = the first non-finite lm / val_loss / geo / grad_prior.
    - Unstable = final val_loss > the same seed's none@2.4e-3 + 0.5 nats. That reference exists only for seeds 0-1, and
      for seeds 2-7 only if 2.4e-3 is none's pick; otherwise there is no unstable check.
    - A run that stops with an error of its own is listed in runs/ptlr_1b/OWN_FAILURES.txt: its area runs over all 75
      slots with the missing evals scored 0.
    - A preemption or node loss is requeued and resumes; it is not a failure.
7.  **"Survives tuning at 1B"** iff, on seeds 0-3 at the picks, win-ctx@pick − none@pick area > 0 with paired t p < .05
    and >= 3/4 wins, AND the R_0.5 bootstrap 90% CI is above 1. It stays PROVISIONAL until confirm seeds 4-7 agree (same
    sign, >= 3/4 wins).
4.  R_0.9 / R_0.5 at the picks as F87: median t_theta, censored at 300M, seed-bootstrap 90% CI. The lrs picked by median
    t90 on seeds 0-1 are printed as a descriptive line, without a ratio.
8.  **Size trend**: ptscale_summary's slope of log R_0.5 vs log N_nonemb over 53M / 190M / 505M / 1B.
    - It is called "grows" / "shrinks" only if its 90% CI excludes 0.
    - A size with an F87 open bracket is left out (NOT FINAL). 1B is "incomplete" until seeds 2-3 exist.
    - The CI does not include the lr-selection uncertainty (F87 addendum 1). With 1B's lrs chosen on 2 seeds, that
      uncertainty grows, so any written trend claim needs the coordinator's / user's decision. Default wording:
      "undetermined".

## Controls (the read-out withholds every verdict that uses a failed cell)
Checked in every 1B run:
- code_sha a3bce3ab63c1 (SNAP ptlr_1b);
- 18,310 steps and 75 evals; grad_ckpt_on True;
- n_params − V·d_model = 943,904,000 (asserted);
- win-ctx: prior_on_frac 6408/18310 and prior_events [[916, 1], [7324, 0]]; none: prior_used_frac 0;
- step-0 lm equal across the 1B cells of a seed (within 1e-3 of the median), and for seed 0 within 1e-3 of the B200 smoke
  value 11.73940. The smoke's A, A2, B, C and real runs all printed 11.739395141601562; the lr does not enter step 0.

Ceiling (seen_bXY < 0.9 at the end) is printed, descriptive, as F87. The cell keying, the train-vs-eval table, the negative
control and the metrics are F87's (runs/ptlr/AUDIT.md), unchanged.

## Expected ranges (outside → check the run before using it)
- none@pick area 50-75: 505M none 57.5-62.1, bimodal. Seeds can sit in the low mode (505M seed 5: 34.2).
- win-ctx area 87-92: 505M 88.1-89.6.
- Final in-training bpb about 1.05-1.12: below 505M's.
- win-ctx t50 about 15-25M tokens.
- Divergence risk is highest for none@9.6e-3. Runs are about 3.0 h each.

## Stage-1 smoke (2026-10-23, job 20675216, gb015, B200; scale_work/b200/smoke)
- **Environment and memory.** torch 2.10.0+cu128 and sm_100 (capability 10.0) work; bf16 autocast and SDPA forward and
  backward are fine. Peak memory 60.5 GB of 183 GB at about 100% utilisation.
- **Speed.** 0.58 s/step including evals and two ckpt writes (steps 0-1000 with the real args; prior on from 916).
- **Size on disk.** model 4.29 GB; ckpt 12.87 GB (model + 2 Adam moments).
- **Resume tests** (300-step runs, window on at 15..119, ckpt every 100): the first resumed eval is identical as stated in
  Deviations 2. The end-of-run difference to an uninterrupted run (val_loss 0.020 / 0.059) is of the same order as
  between two uninterrupted runs (0.010). Prior events are identical, and there are no duplicated or missing evals.
- **A real preemption and requeue happened, and the requeued job resumed correctly.**
  - The job was preempted at 09:56 (Restarts=1). It was requeued under the same id, restarted at 11:29 on gb007, re-entered
    the script and resumed the real-args run from its step-999 ckpt ("resumed from ... at step 1000 of 18310").
  - check_smoke_resume.py (cpu job 20729197, afterany) compared its step-1000 eval with the pre-preemption one: **PASS**.
    lm, geo, cka_ops_helix3, ema_first_line, prior_on, prior_on_frac, held_b11 and held_b33 are equal to all digits;
    val_loss differs by 1.25e-5 and val_bits_per_byte by 3.7e-6 (eval noise).
  - The ckpt was then deleted (smoke dir now 109 KB).
  - With the prior on: 0.57 s/step including evals, so a run is about 2.9 h.

## Cost and disk
- **Cost.** Stage A about 36 B200-h, B about 12, C about 24.
- **Disk.**
  - Concurrent ckpts at most 4 x 12.9 GB (plus one transient copy per writing run), at most about 103 GB at peak, freed at
    the end of each run.
  - Models: B 4 x 4.3 GB, C 8 x 4.3 GB, about 51 GB in total.
  - /scratch had about 270 GB free on 2026-10-23 (myquota 4.72 / 5.00 TB). The smoke's models are deleted; its 12.9 GB
    ckpt is deleted after the resume check.

## Known limits
- Two selection seeds, while none is bimodal (SD 7.8 / 12.2 / 9.8 at 53M / 190M / 505M). A pick within a few points is
  noise-driven; the confirm seeds are the guard.
- B200 vs L40S numerics (bf16 autocast on both).
- Preemptible queue: the smoke waited about 11 h to start and about 3 days on 10-14.
- L2 window (F87 / pt2s505): a preemption between the final run JSON and the model save leaves a complete run without a
  model (seeds 2-7). The read-out does not need the model.
- 4-slot singleton: about 36 h of stage-A wall time at best.

## Audit (minimal mode, before submission)
- **ii asserts.** The generator asserts the snapshot sha, COMMIT, the file-by-file equality with F87's snapshot and every
  flag. Fault injection (an off-grid lr, the wrong arm's lr, seed 8, 24 layers, a repeated --lr) stopped it.
  pick_1b.py / cells() reject an off-grid pick.
- **iii known answers.** CPU `--dry_inputs` of none and win-ctx x seeds 0-7 vs the dry sections of F87's 505M job logs
  of the same arm and seed: identical except the model line (scale_work/b200/dry_1b_cmp.txt). The step-0 lm of seed 0 is
  the smoke's.
- **Job files.** jobcheck_1b.py checks bash -n; ARGS token by token vs F87's 505M job of the same arm and seed (equal
  except out / tag / lr / model / ckpt_every / save_model); body lines; the #SBATCH header. Result: PASS, 12/12. Planted
  errors (ckpt_every 2000, an extra --lam, constraint h200) are caught. sbatch --test-only passes for every job under both
  accounts.
- **Read-out.** ladder-1b 4b2d214: byte-identical on the existing results with and without --json. 8 fake 1B scenarios
  print their designed values. pick_1b.py on the fake read-outs decides as designed: picks, OPEN stop, withheld (lm0 off
  the median or the reference), incomplete, edge-pick tie.
- **Selection plumbing** (job 20728768, logs/llmadd_ptlr1b-plumb-cpu_20728768.out): maybe_select.sh end to end on a
  cpu_short node, with the fake root and a stand-in generator that only runs `sbatch --test-only`. It picked as designed,
  wrote picks.json and select.done, and the second call stopped at select.done. With the SLURM_* / SBATCH_* variables
  removed, `sbatch --test-only` from the compute node succeeded for both accounts.
- **Submission tests.** `make_jobs_ptlr1b.py --test-only`: 24/24 copies OK (scale_work/b200/testonly_A.txt). 520 copies
  are estimated to start at once (b200_tandon_prem), 625 copies on 10-24.
- **Frozen before submission (sha256[:12]).** make_jobs_ptlr1b.py 8dd7b3407459, maybe_select.sh ba787f4ac54e, pick_1b.py
  10a2c3d8cc5f (chmod a-w); readout/ = ladder-1b 4b2d214 (ptscale_summary.py 61ea4e2e68d2).
- **v.** A fresh adversarial audit runs after submission (coordinator).

Files:
- llmadd_torch/make_jobs_ptlr1b.py;
- runs/ptlr_1b/{maybe_select.sh, pick_1b.py, readout/, code/};
- scale_work/b200/{job_smoke1b.sbatch, cmp_runs.py, env_check.py, dry_1b.py, jobcheck_1b.py};
- branch ladder-1b (tools/ptscale_summary.py).

## Submitted (2026-10-23 ~12:05; submit_A.log)
24 copies of the 12 stage-A runs, jobs 20735258-20735315 (per run the 520 copy first, then the 625 copy; submit_A.log). Under singleton,
Slurm keeps one job per slot eligible: the 520 copy of each slot's first run (20735258, 20735261, 20735265, 20735267); the
others wait on "Dependency". Smoke copy 20675217 (520) is still queued; it exits at start (owner.lock = 20675216). Stage B is
automatic (maybe_select.sh); stage C needs runs/ptlr_1b/APPROVE_C.

## Amendment B1 (build audit, 2026-10-23, before any 1B result)
Fresh audit of the build after submission (notes/batch1_build/audit_ptlr1b_build.md): no blocker, no HOLD. The points
below were checked by the coordinator against the code before they were written here.

1. **Read-out fix: rule 8 did not check seeds 2-3 at the 1B picks.** The 1B best is picked on seeds 0-1, so its "ok"
   says nothing about seeds 2-3. Rule 7 withheld on them (ptscale_summary.py, `withheld([kn, ka], R.MAIN)`), but the
   rule-8 slope took the 1B point anyway (synthetic scenario ABbad: rule 7 WITHHELD, slope with 1B 3.91x). Now a SEL
   size with a withheld seed 0-3 at a pick gives "withheld (seeds 0-3 at the picks)" and no slope.
2. **Rule 8 with 1B: labels and a companion slope (pre-registered here).**
   - The 1B point enters the trend with two differences from the F87 sizes: its lrs were picked on 2 seeds (F87: 4), and
     it ran on B200 (F87: L40S). It is also a high-leverage end point. The rule-8 lines now say both whenever 1B is in.
   - Companion slope on **confirm seeds 4-7 only** (no seed used to pick an lr, at any size), same method, printed
     whenever 1B is in the trend and every size has its confirm seeds at the picks. It is descriptive; the rule-8 call is
     still the seeds-0-3 R_0.5 line.
   - **Interpretation rule.** If the confirm-only R_0.5 call differs from the seeds-0-3 call, the size trend is written
     as "depends on the seeds used: not a finding". Without stage C (no APPROVE_C) there is no confirm-only slope, and a
     4-size trend is written as "PROVISIONAL: 1B lrs picked on 2 of its 4 seeds".
   - Code: ladder-1b ac2e03c, then 2ce723a (diff audit, minor items: the read-out itself prints the two phrasings of
     the interpretation rule, and the confirm-only R_0.5 line is labelled descriptive); readout/tools/ptscale_summary.py
     61ea4e2e68d2 → 404f4466e3bb, readout/COMMIT updated (the old file is kept as
     runs/ptlr_1b/readout_pre_B1_ptscale_summary.py). Nothing else in readout/ changed, and no script pins this sha.
   - Regression: on the real runs root (no ptlr_1b/out yet) the text and --json outputs of the old and new read-out are
     byte-identical. On the synthetic scenarios (the build audit's fake1b.py, plus ABC / ABCbad with 1B confirm seeds)
     the --json (pick_1b's input) is identical in all 9, and only rule-8 lines change: A / Adup / Aedge / Aopen / Apart
     none; AB labels + "confirm-only slope not available"; ABbad "1B withheld (seeds 0-3 at the picks)", no slope; ABC
     the confirm-only lines; ABCbad confirm-only WITHHELD on the bad seed 5.
3. **Corrections to "Selection and stages" and "Known limits" (singleton).** Slurm's singleton is per job name and user,
   and the two copies of a run share the slot name, so a slot's six jobs run strictly in submission order (squeue at
   submission: 4 eligible, 20 "Dependency"). Hence:
   - "a copy that starts while the claimant is pending takes the claim over" cannot happen: the 625 copy can start only
     after the 520 copy has ended. The 625 copy is a retry after FAILED / TIMEOUT only.
   - After each run, its 625 copy must still get a B200, start and exit before the slot's next run can start: up to
     three extra 625 queue waits per slot in stage A (the smoke's 625 copy waited 1.5-15 h). "36 h at best" does not
     hold with the 625 copies in the queue.
   - Removing them needs scancel (the user's call; the coordinator cannot scancel). HOLD does not help: the copy must
     start to read it.
4. **Silent stops (no job fails, nothing is submitted).** None of them can submit wrongly.
   - maybe_select.sh writes select.done even if some `--submit` calls failed (it logs that in select.log).
   - A failed assert in `pick_1b.py --check-complete` is swallowed by `|| exit 0`, without a line in select.log.
   - `--check-complete` does not read OWN_FAILURES: one run that fails on its own stops the automation; it then needs
     a manual run.
   - Guard (the coordinator's watcher): any change of select.log / select.done / readout_A.txt / HOLD / APPROVE_C is an
     event, and so is the slot group leaving the queue. Before stage B starts training, the coordinator reads
     readout_A.txt against "Expected ranges" and puts HOLD if a cell is outside them.
5. **F87 reproduction.** With runs/ptlr_1b/out present, the default command no longer prints F87's 3-size slope (it
   gives "no slope", then the 4-size slope). F87's numbers are reproduced with the pre-extension read-out sha or a root
   without ptlr_1b.
6. **Wording.** pick_1b's "OPEN (edge pick)" is, in F87's terms, *not* an open bracket (an edge pick within 2 points).
   FINDINGS will write "edge pick within 2 points (not open by F87 rule 1)", never "OPEN".
7. **Final model save is not atomic (stages B / C).** A preemption during the save can leave a truncated model that
   looks present. The read-out does not use models; any later use of a 1B model first checks that it loads and that its
   size is about 4.29 GB.
8. **Disk.** About 290 GB free on 2026-10-23 (shared quota); 1B worst case 155-170 GB. myquota is checked before
   stage B / C are submitted.
9. **Diff audit of the read-out fix** (fresh, 2026-10-23): no blocker or important item. It added 8 synthetic
   scenarios (a bad prior_on_frac / step-0 lm / code_sha on seed 2 or 3 → 1B withheld; a bad seed only at an unpicked lr
   or only on a confirm seed → 1B kept) and recomputed the slope and the bootstrap CI from the run JSONs by an
   independent path: equal to all digits on ABC and on a variant. Its three minor items are fixed in 2ce723a. Round 2:
   the real runs root is byte-identical again; PROVISIONAL appears only without a confirm-only slope; in ABCdiff (the two
   calls differ) the read-out prints "depends on the seeds used: not a finding".
10. **625 copies of stage A cancelled (2026-10-23, the user's decision).** Jobs 20735259 / 262 / 266 / 268 / 272 / 274 /
    276 / 279 / 282 / 311 / 313 / 315 (all pending). Each slot now runs its three 520 jobs back to back. A run that
    FAILs or TIMEOUTs on its own is no longer retried automatically: the coordinator resubmits it by hand (preemption
    still requeues the same job). Their SUBMITTED_*_625 markers stay, so `--submit` does not resubmit them. Stage B / C
    (make_jobs_ptlr1b.py, read-only) will again submit both copies; cancelling those 625 copies is again the user's call.

## Amendment B2 (2026-10-25, after stage A's read-out, before any none@1.2e-3 job or result)
Trigger: stage A finished; maybe_select.sh wrote "NO PICKS: none: OPEN bracket at the lower edge" (readout_A.txt, seeds
0-1: none 2.4e-3 57.1 > 4.8e-3 50.2 > 9.6e-3 48.7, 2.4e-3 ahead by 6.9 > 2). Every stage-A cell is inside "Expected
ranges" except none@9.6e-3's in-training bpb 1.1457 (slightly above 1.12; the coordinator checked, B1 item 4). User,
2026-10-25: 「你觉得有必要的实验我觉得就可以跑」; the coordinator judged one step down necessary for a 1B verdict.

1. **New cell: none@1.2e-3 (= 2.4e-3 / 2, the grid's factor-2 step), seeds 0-1, 2 jobs, account 520 only**
   (b200_tandon_prem or b200, preemptible, --requeue). No 625 copy (B1 item 10: a 625 copy only adds queue waits). A
   run that FAILs / TIMEOUTs on its own is resubmitted by hand.
   - Rendering: llmadd_torch/make_jobs_ptlr1b_b2.py imports the read-only make_jobs_ptlr1b.py (sha 8dd7b3407459, asserted;
     an ast check asserts that importing it runs no code but its __main__ guard) and reuses its snapshot() check and its
     job_text() / render_args() with every assert. The only in-process change is none's lr grid, widened to
     {1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3} so that render_args accepts 1.2e-3.
   - A B2 job differs from the stage-A none job of the same seed only in the lr, the tag, the out dir and the job name
     (llmadd_ptlr1b-b2-<run>, no slot: 2 jobs), plus one added line at the end: the B2 selection trigger
     (maybe_select_b2.sh, item 4). jobcheck_b2.py checks this line by line and token by token.
   - Same trainer snapshot, data, seeds, 3e8 tokens, ckpt_every 500, no --save_model (seeds 0-1), B200 guard, HOLD,
     --time 5 h.
2. **Read-out (pre-registered).** F87 rule 1 re-applied to none {1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3}, seeds 0-1, by the
   unchanged read-out:
   - 1.2e-3 beats 2.4e-3 by > 2 points → still OPEN at the lower edge: stop, no stage B, the user decides; no further
     automatic step.
   - The two within 2 points → F87's top-two tie-break (lower mean in-training bpb). If that picks 1.2e-3 it is an edge
     pick within 2 points (not open by F87 rule 1, B1 item 6): closed.
   - 2.4e-3 best by > 2 → closed.
   - win-ctx: its pick stays 6e-4 (readout_A.txt: 6e-4 88.8 and 1.2e-3 89.6 within 2, tie broken on the in-training
     bpb 1.0850 < 1.0918; the win-ctx data do not change). pick_1b_b2.py asserts the read-out still gives 6e-4.
   - The none pick's seeds 0-1 come from the selection, as in stage A (winner's curse as in Deviation 1).
3. **Stage B after B2 (automatic, 520 only):** seeds 2-3 at none@pick and win-ctx@6e-4 (4 runs), rendered by
   make_jobs_ptlr1b_b2.py --stage B (job_text unchanged, slot names slot0-3 as make_jobs_ptlr1b.cells() would give
   them, so at most 4 at a time), --test-only first, then --submit under 520 only. Stage C (seeds 4-7) still needs
   runs/ptlr_1b/APPROVE_C: then `make_jobs_ptlr1b_b2.py --stage C --submit` (520 only).
4. **Trigger.** Every B2 job ends with maybe_select_b2.sh (after the unchanged maybe_select.sh call, which stops at
   select.done). It does nothing unless all 14 runs (stage A + B2) are complete. The first job to find them complete runs
   the read-out snapshot (readout/, unchanged: ladder-1b 2ce723a, ptscale_summary.py 404f4466e3bb), then
   pick_1b_b2.py, then the wrapper's --stage B (--test-only, then --submit).
   - Files: select_b2.lock (stale-lock takeover as maybe_select.sh), select_b2.log, select_b2.done, selection_B2.json,
     readout_B2.txt, picks_b2.json (not picks.json, so make_jobs_ptlr1b.cells() never sees it).
   - The coordinator's watcher reads readout_B2.txt against "Expected ranges" before stage B trains (B1 item 4); HOLD
     stops stage B jobs at start.
5. **Read-out change: none needed.** ptscale_summary.py builds each (size, arm)'s grid from the run args it finds (cells
   keyed on args), not from a list, and SEL only names the selection seeds. A none@1.2e-3 1B cell joins 1B's none grid
   as it appears. Tested before submission on synthetic roots (the real 12 stage-A runs plus two designed none@1.2e-3
   runs): the 4-lr grid is printed and the designed best / open edge come out in every scenario (scale_work/b200/b2/).
   Default output on the real root is byte-identical to readout_A.txt and selection_A.json (unchanged code).
6. **Cost and disk.** 2 runs x about 3 B200-h; no models, so the only disk use is the transient ckpts (at most 2 x 12.9 GB).
7. **Tests before submission (scale_work/b200/b2/).**
   - jobcheck_b2.py: PASS 2/2. Each B2 job, with the job name, the run name and the lr mapped back, equals the stage-A
     none@2.4e-3 job of its seed byte for byte, plus the two hook lines. ARGS change only in --lr / --tag / --out.
     Planted errors are caught: ckpt_every 2000, seed swap, --time 6 h, maybe_select_b2 → maybe_select.
   - Read-out on synthetic roots (run_tests.sh, tests.out):
     - ctrl (no 1.2e-3 cell): --json identical to selection_A.json; the text differs only in the padding of the
       sources-path column (longer root path). On the real root the text is byte-identical to readout_A.txt.
     - open: 1.2e-3 62.0 → best 1.2e-3, OPEN lower → NO PICKS.
     - tie_lo: 57.6, lower bpb → 1.2e-3, edge pick within 2 → picks 1.2e-3.
     - tie_hi: 57.6, higher bpb → 2.4e-3.
     - worse: 52.0 → 2.4e-3.
     - unfinished: none incomplete, check-complete 13/14.
     - lm0bad: none WITHHELD → NO PICKS.
     - win-ctx stays 6e-4 throughout.
   - test_wrapper.py: PASS.
     - Importing the wrapper renders nothing.
     - Stage B = job_text(12-15) for a none pick of 2.4e-3 or 1.2e-3 (slot0-3, --save_model, no B2 hook), so for 2.4e-3
       it equals make_jobs_ptlr1b's own rendering.
     - Stage C is refused without APPROVE_C, then = job_text(16-23).
     - An off-grid none pick or a win-ctx pick other than 6e-4 is refused; the ast guard rejects a module with a
       top-level call.
   - Plumbing on a compute node (cpu_short job 20904050; synthetic root tie_hi; stand-in generator rendering stage B
     with the real wrapper and running only `sbatch --test-only --account=520`): 14/14 complete → read-out → picks none
     2.4e-3 / win-ctx 6e-4 → 4 stage-B test-only OK (no SLURM_* left) → select_b2.done; the second call stops at
     select_b2.done.
   - `make_jobs_ptlr1b_b2.py --stage b2 --test-only`: 2/2 OK (b200_tandon_prem; estimated start 10-28).
   - myquota before submission: /scratch 4.62 / 5.00 TB (92.5%).
8. **Frozen (sha256[:12], chmod a-w):**
   - make_jobs_ptlr1b_b2.py 1ecf7a3f7a57, pick_1b_b2.py ca52f2199970, maybe_select_b2.sh e5ec45ee0b19;
   - job files b1f30a60dfab (s0) / 00f605f09dae (s1);
   - read-out unchanged: readout/tools/ptscale_summary.py 404f4466e3bb, readout/COMMIT unchanged.
9. **Submitted 2026-10-25 ~14:56: jobs 20904240 (s0) and 20904241 (s1)**, account 520 (b200_tandon_prem,b200), pending.
   The job files are unchanged since jobcheck (sha256[:12] as in item 8). The fresh minimal-mode audit runs after
   submission.
10. **Fresh post-submission audit (minimal mode, read-only, 2026-10-25): no blocker.** It verified independently:
    - the job diff and the Slurm-stored scripts;
    - account 520 / b200_tandon_prem,b200;
    - the wrapper (frozen shas, ast guard, the grid widened in-process only, 520 only, the stage-C gate);
    - that the original chain cannot submit B2-inconsistent jobs;
    - the read-out's arg-keyed grid, byte-identical on the real root;
    - the stage-A areas recomputed from the run JSONs (none 57.06 / 50.25 / 48.72; win-ctx 6e-4 88.85, 1.2e-3 89.59);
    - unit tests of the 4-lr top-two / bracket (+2.01 → OPEN, +2.00 → tie);
    - every pick_1b_b2 branch.

    Acted on:
    - **IMPORTANT, fixed:** the read-out snapshot was not pinned at run time.
      - readout/tools/{ptscale_summary.py, readout_rules.py, pt999_summary.py} and readout/COMMIT are now chmod a-w.
      - maybe_select_b2.sh checks their sha256[:12] (404f4466e3bb e91ee8a8a846 01e2fec07499) before the read-out. On a
        mismatch it logs "READ-OUT SNAPSHOT CHANGED", writes select_b2.done and selects / submits nothing (tested with
        a tampered copy). On a match it logs the shas.
      - maybe_select_b2.sh e5ec45ee0b19 → 6e7ce43362f6 (chmod a-w). The submitted jobs call it at run time; nothing else changed.
    - **Corrections to the text above:**
      - "Expected ranges": none@4.8e-3's bpb 1.1240 is also above 1.12, not only 9.6e-3's 1.1457.
      - Item 1 says "one added line": two lines are appended (a comment and the call).
    - **Notes:**
      - The B2 job files say `#SBATCH --account=torch_pr_625_general` (as every stage-A file; the account comes from the
        command line). A manual resubmission must pass `--account=torch_pr_520_tandon_priority`.
      - Stage B now has no 625 copies. B1 item 10 left that to the user; it matches the user's choice for stage A.
      - The automatic chain has no myquota step: /scratch is at 4.62 / 5.00 TB, and stage B adds at most about 75 GB.
      - sbatch from a B200 node is untested (the plumbing ran on cpu_short). Read stage B's first log for the GPU name
        and s/step: a job that silently lost its GPU would show as slow, not as wrong numbers.
      - jobcheck_b2 maps NEW → OLD before comparing, so a line that wrongly kept OLD would pass it; the audit's direct
        diff found none.

## Amendment B3 (2026-10-25 ~22:30, stage C submitted)
- **Approval.** runs/ptlr_1b/APPROVE_C was created 2026-10-25 22:28 EDT on the coordinator's instruction.
  - Basis: the user's 2026-10-25 「你觉得有必要的实验我觉得就可以跑」, and this file's "APPROVE_C = the coordinator's approval of
    the cost".
  - Reason: without confirm seeds the 1B point stays provisional (B1 item 2), while every other F87 size has confirm
    seeds 4-7.
- **Context.** B2 closed: none@1.2e-3 scored 42.8, below 2.4e-3's 57.1. picks_b2.json = none@2.4e-3 / win-ctx@6e-4.
  Stage B: jobs 20937726 / 28 / 29 / 31 (520).
- **Submission.** `make_jobs_ptlr1b_b2.py --stage C --test-only` (8/8 OK), then `--stage C --submit` (submit_C.log).
  - The wrapper is the already-audited one (sha256[:12] 1ecf7a3f7a57, unchanged), account 520 only, no 625 copies, as
    stage B.
  - Jobs: 20938082 none-s4, 20938083 win-ctx-s4, 20938084 none-s5, 20938085 win-ctx-s5, 20938086 none-s6,
    20938087 win-ctx-s6, 20938088 none-s7, 20938089 win-ctx-s7.
  - Slots slot0-3 with singleton: each slot runs its stage-B job first, then its two stage-C jobs (at most 4 at a time).
- **jobcheck_c.py: PASS 8/8** (scale_work/b200/b2/jobcheck_c.out). Each stage-C job equals the stage-B job of its arm
  (seed 2) byte for byte once the slot name and the seed are mapped back. The run name in --output, the comment and the
  banners carries the seed; tag and out dir are the same, as they should be. ARGS differ only in --seed; --save_model is
  present.
- **Disk.** myquota at submission: /scratch 4.62 / 5.00 TB (92.5%), about 380 GB free.
  - Stage B + C save 12 models x 4.29 GB, about 52 GB.
  - At most 4 transient ckpts x 12.9 GB (plus one copy per write) at a time.
- **No new audit.** Only parameters changed (seeds 4-7, approved by APPROVE_C). The code is the audited B2 wrapper and
  job_text(), unchanged.
