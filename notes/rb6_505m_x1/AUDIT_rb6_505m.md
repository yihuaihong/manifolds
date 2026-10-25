# rb6_505m — the six-prior rainbow pretraining at 505M: lr sweep, then main (scale ladder, batch 2; 2026-10-22)

User approval (relayed by the coordinator): "可以的，没问题" for the model-size scaling batches, and the standing
"需要现在该提交的都可以提交"; the main stage is submitted automatically after the sweep's selection (no further approval).
Written to this file BEFORE any job of this batch was submitted. Everything not stated here is as in
runs/rb6_190m/AUDIT.md (design, checks, read-out), with the 505M model.

## Model
`--d_model 1024 --n_layers 24 --n_heads 16 --d_ff 4096` (= runs/ptlr / runs/ptsize999's 500m flags; 505,437,184
parameters; gradient checkpointing on; prior / read-out layers 8, 16); the 53M rb6 main design otherwise unchanged (2%
numbers first, K = 2 domain lines, six domains, window (0.05, 0.4), ctx, lam_rule each, lambda 1 per active prior, 3e8
tokens = 18,310 steps of 32 x 512, 75 evals). Tags `rb6tb-505m-2pct-k2-<arm>-lr<lr>`.

## Stage 1: lr sweep (12 jobs, seeds 0-1)
| arm | lrs |
|---|---|
| none (`--target none`) | 4.8e-3, 9.6e-3, 1.92e-2 |
| all6-lam1 (the six priors, lambda 1 each) | 3e-4, 6e-4, 1.2e-3 |
Why these grids: none's 505M bracket at 7% single-prior (runs/ptlr: 4.8e-3 57.6, 9.6e-3 62.1, 1.92e-2 added by the open
bracket rule) and at 190M 2% none's best 4.8e-3; the prior at 505M 7% ran 1.2e-3 (89.6), at 190M 2% its best was the
lower edge 6e-4. 1.92e-2 may diverge: a diverged run is data.

## Selection rule (PRE-REGISTERED; tools/rb6scale_select.py, tested on fake curves before any sweep result)
Per arm over its grid, seeds 0-1:
1. the lr with the highest mean numbers-domain area (the rainbow headline numbers metric: rainbow6_summary READ['numbers']
   = 100 x the mean over all 75 evals of the 6-bucket held-out exact match);
2. every lr within 2 area points of the top is tied -> the lower mean in-training bpb (the last eval's
   val_bits_per_byte, mean over seeds 0-1; a non-finite value = the worst) -> then the lower lr;
3. bracket state reported: an edge pick = OPEN ("one more step outward only with the user's approval"); a middle pick =
   closed. An OPEN bracket does NOT stop the main stage (the picks are the picks); the extra outward step is the user's
   call and is not pre-registered here.
A diverged run (first non-finite lm / val / prior value) is data: its area over all evals counts against its lr, its bpb
is the worst. Completeness: all 12 runs present, finished, identity clean (rb6scale_summary.torch_identity: every
trainer argument = the cell's intent, run facts, code / toy shas, lambda, the logged multiplier, check.json PASS) and the
integrity of rainbow6_summary.check across the 12 (same data and step-0 lm per seed); otherwise status "incomplete", NO
picks, and the main stage is NOT submitted (the selection job's log names the problem; the user decides).
Selection seeds = 0-1 only; selection effects on the main read-out are labelled (below).

## Stage 2: main (submitted automatically by the selection job)
Cells, seeds 0-3: none@pick_none ("best"), none@pick_all6 ("low": the lr-matched baseline), all6-lam1@pick_all6,
only-numbers@pick_all6 (= F88 / rb6_190m's four roles). The sweep's seeds 0-1 at the picked lrs are REUSED (same tag,
same job dir); new jobs: none@pick_none s2-3, all6-lam1@pick_all6 s2-3, none@pick_all6 s0-3, only-numbers@pick_all6 s0-3
= 12 jobs (the none and all6 grids are disjoint, so none@pick_all6 is always a new cell).
Mechanism: a CPU job (`llmadd_rb6_505m-select`, cpu_short) with `--dependency=afterany:<the 12 sweep job ids>` runs, from
the batch's code snapshot, `tools/rb6scale_select.py runs/rb6_505m/out --json runs/rb6_505m/picks.json`; only if its
status is "picked" it runs `make_jobs_rb6scale.py --batch rb6_505m --stage main --picks runs/rb6_505m/picks.json
--test-only` and then `--submit` (SUBMITTED_<name> flags; the reused seeds 0-1 already have theirs). Its log:
llmadd_torch/logs/llmadd_rb6_505m-select_<jobid>.out; picks and the generator output in runs/rb6_505m/select.log.

## Read-out (pre-registered; tools/rb6scale_summary.py with `--picks runs/rb6_505m/picks.json`)
As runs/rb6_190m/AUDIT.md (the same metrics, cells per size side by side, families S1 / S1b / SE / S2 / S3 / SH / SL,
identity, integrity, cross-size data identity, PROVISIONAL / strict refusal, no size-trend claim), with these labels:
- the 505M lrs were selected on the numbers area of THIS batch's seeds 0-1 (which are part of the read-out's seeds 0-3):
  every paired line is "on the lr-selection seeds 0-3: descriptive, no confirmatory claim"; the numbers rows carry a
  selection effect in favour of the selected cells (none@best on numbers in particular: the winner's curse);
- none's lr was chosen on numbers only: not tuned for S4 or any other domain;
- an OPEN bracket of either arm is printed next to its cells.

## Jobs
Sweep 12 + main 12 = 24 jobs; account torch_pr_520_tandon_priority, a100_tandon, gpu48, `--nice=20`, 1 GPU, 8 CPUs, 64G,
`--time=13:00:00` (2 x 6.3 h, the rainbow estimate at 505M; rounded up), `--requeue`. Planned ~24 x 6.3 = 151 A100-h.
Checks inside every job: as rb6_190m (model line `model: 505.4M parameters; gradient checkpointing on`; DRY rows = the
Torch CPU 505M dry run = the Burst 53M rows with this model line; step-0 lm vs the 505M CPU reference; ...).

## Tests before submission (CPU; scripts / logs: the session scratchpad's scale/ and llmadd_torch/scale_work/)
| test | result |
|---|---|
| CPU regression | as runs/rb6_190m/AUDIT.md (the same trainer and cell types; rb6-all6 / none / only-numbers / all6 + --grad_ckpt on: IDENTICAL to c434d29's) |
| references (refs.py, the job's own ARGS) | 505M --dry_inputs of 7 (arm, seed) cells: per seed the 88 DRY rows = the Burst 53M seed's rows with the model line `model: 505.4M parameters; gradient checkpointing on` (seeds 0-3), DOMS = make_rb6m_tasks.DOMS, priors as intended; step-0 lm seeds 0-3 = 11.713464 / 11.727497 / 11.707520 / 11.718043 (all6 s0 = none s0 bitwise); data_sha after step 0 = the 53M DSHA0; CPU method calibrated at 505M on the default path (ptlr 500m-lr9.6e-3-7pct-none s0): CPU 11.711900 vs GPU 11.711954 (-5.4e-5) |
| jobcheck.py (the 12 sweep jobs) | JOBCHECK PASS (133): bash -n, CHECK_PY = the Burst task's byte for byte, intent differs only in the size facts / lr / tag / seed, input-check block = the Burst task's except model / DRY / pin / ARGS lines; on the 505M CPU dry logs 12 own cases exit 0, 72 cross cases exit 3 with the right reason |
| sel_test.py (rb6scale_select.py on fake curves: real 53M runs relabelled as the 12 sweep runs, planted numbers areas / bpb) | SELECT TEST PASS (9): ties within 2 -> lower in-training bpb (A: none 4.8e-3 over 9.6e-3, OPEN lower; all6 6e-4 closed), exact bpb tie -> lower lr and an upper-edge pick OPEN (B), a diverged run's area counts and its bpb is the worst (C), a missing run / a failed check.json / a mislabelled lr -> incomplete, no picks (D-F), a stray 53M run ignored (G) |
| the selection job's plumbing (select_job.sbatch on fake R dirs, the generator calls replaced by echo) | scenario A: snapshot sha checks pass, status picked, then the main stage's --test-only and --submit calls; scenario D: incomplete -> "NOT PICKED", no generator call |
| main-stage rendering with fake picks (none 9.6e-3, all6 6e-4; --print only; the 12 extra job files deleted again) | 16 jobs: none@9.6e-3, none@6e-4, all6-lam1@6e-4, only-numbers@6e-4 x seeds 0-3; the sweep cells' files re-rendered identically |
| sbatch --test-only (sweep) | 12/12 accepted |

## Submitted (2026-10-22 ~04:36 EDT)
Sweep: 12 jobs 20599789-20599804 (SUBMITTED_<name>); code 7f1d06b (runs/rb6_505m/code/COMMIT), trainer 93e22bf4c4c0.
Selection + automatic main stage: job 20599806 (select_job.sbatch, cpu_short, --dependency=afterany on the 12; SUBMITTED_select).
Its outputs: runs/rb6_505m/picks.json, runs/rb6_505m/select.log, logs/llmadd_rb6_505m-select_20599806.out.
Kill switch: touch runs/rb6_505m/HOLD (pending GPU jobs and the selection job exit at start).

## Freeze note (2026-10-22, build audit notes/batch1_build/audit_rb6_505m_build.md M1 / M2 / M3; no result read)
- The selection job renders the main stage with the live generator, which is not under version control. These files are
  now read-only (chmod a-w) and must not change until the main stage has been submitted. sha256[:12]:
  make_jobs_rb6scale.py 19bc055c2d09, rb6scale_refs.json b65472eb3d29. For pt2s505: make_jobs_pt2s505.py a5a7fb60eedc,
  pt2s505_refs.json 479e394771e9, runs/pt2s505/pick_confirm.py (read-only too).
- The worktree /scratch/yh6210/llmadd-burst-scale (branch scale-ladder, commits c6dcb6e / 7f1d06b) is read by both
  automatic stages via git show. Do not remove it until they have run.
- M3: 505M six-prior peak memory is estimated at 62-66 GB of 80 GB, untested on GPU. Check the first all6 job's log past
  step 1000 for OOM.
- Before reading 505M results: the summary's lr-source line is wrong for 505M (it says pt2s190); fix it with an amendment
  first.

## Amendment X1 (2026-10-25): none@2.4e-3, seeds 0-1, one step below none's grid (PRE-REGISTERED before any 2.4e-3 result)
Written before any X1 job was rendered or submitted. No 2.4e-3 run exists. The only results read so far are the sweep
values the selection job printed (select.log, job 20599806, 2026-10-23T14:55Z).
- Why: none's pick 4.8e-3 sits at the LOWER edge of {4.8e-3, 9.6e-3, 1.92e-2}, so the bracket is OPEN (Selection rule 3).
  Recorded values: 4.8e-3 mean area 36.5, mean in-training bpb 1.1346; 9.6e-3 35.9 / 1.1573; 1.92e-2 16.2 / 1.2026.
  4.8e-3 and 9.6e-3 tied within 2, and the lower bpb won. Rule 3 makes the step outward the user's call; it is not
  pre-registered above.
- User approval (relayed by the coordinator, 2026-10-25), given for exactly this step: "你觉得有必要的实验我觉得就可以跑".
- Added cell: none @ 2.4e-3 (= 4.8e-3 / 2, the grid's factor-2 step), seeds 0, 1 = 2 jobs, about 2 x 6.3 = 13 A100-h.
  It is outside the pre-registered sweep and is labelled X1 wherever it is used. Each job differs from the sweep's
  none@4.8e-3 job of the same seed only in the lr, the tag, the out dir and the job name (jobcheck below). Model,
  data, checks, account and time limit are unchanged. Out dirs: runs/rb6_505m/out/rb6tb-505m-2pct-k2-none-lr2.4e-3-s{0,1}.
- Generator: llmadd_torch/make_jobs_rb6_505m_x1.py. It imports the read-only make_jobs_rb6scale.py (19bc055c2d09) and
  calls its snapshot() (verify only, code 7f1d06b) and render() with all their asserts. It renders nothing unless this
  section exists. The job files say "generated by make_jobs_rb6scale.py" because render() is that file's, unchanged.

### X1 reading (pre-registered; tightened by Addendum X1.1 at the end of this section)
1. Re-apply Selection rules 1-2 to none over {2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2}, seeds 0-1: the highest mean numbers area
   wins; every lr within 2 area points of the top is tied; ties go to the lower mean in-training bpb, then the lower lr.
   Rule 3 (an edge pick = OPEN) applies to the 4-lr grid. A diverged run is data: its area counts and its bpb is the worst.
   all6-lam1 is untouched: same runs, same grid, pick 6e-4, closed.
2. Tool: tools/rb6scale_select.py at scale-ladder 37d545a (sha256[:12] 0b1b6cb88ebd), frozen in runs/rb6_505m/x1_code/
   (git archive of tools/ at 37d545a, COMMIT lists the shas). After BOTH X1 jobs have ended, the coordinator runs
   `cd runs/rb6_505m/x1_code/tools && python rb6scale_select.py ../../out --grid_none 2.4e-3,4.8e-3,9.6e-3,1.92e-2
   --json ../../picks_x1.json`, with the env of select_job.sbatch. Nothing is submitted automatically.
   runs/rb6_505m/picks.json, the pre-registered selection, is never overwritten: the tool refuses to write over a JSON
   made without the flag. Completeness is as in the Selection rule, over 14 runs (the 12 sweep runs + the 2 X1 runs):
   present, finished, identity clean (torch_identity), check.json PASS, and rainbow6_summary.check integrity.
3. This follows from the recorded values, not from a new result: 9.6e-3 and 1.92e-2 cannot become the pick. 9.6e-3 is
   tied only when 4.8e-3 is, and 4.8e-3 has the lower bpb; 1.92e-2 is 20 points below. So the pick is 2.4e-3 or 4.8e-3.
   Any other pick is a bug: stop and report it.

### X1 outcomes
a. Pick stays 4.8e-3: none's bracket is closed (2.4e-3 below it, 9.6e-3 above it). The main read-out is unchanged:
   none_nlr = none@4.8e-3, read with picks.json as pre-registered. The 2.4e-3 runs appear in the X1 selection table
   only, as descriptive values. The main read-out lists them as IGNORED.
b. Pick becomes 2.4e-3: none@best (role none_nlr, the numbers-selected none cell) becomes none@2.4e-3.
   - The main read-out's best-vs-best rows (S1b; the none_nlr rows of SE and SH) need 2.4e-3 at seeds 2-3. The
     coordinator submits these 2 jobs (none@2.4e-3 s2, s3) as part of X1. They use the same render() and checks; the
     505M step-0 lm and DRY references exist for seeds 0-3. Until both finish, the read-out refuses in strict mode; with
     --allow_partial it is PROVISIONAL.
   - The lower edge stays OPEN, because 2.4e-3 is now the edge. No further step down runs automatically; that is the
     user's call again. The main stage's none@6e-4 sits 4x below 2.4e-3, not a factor-2 neighbour, so it does not close
     the bracket. Its seeds 0-1 numbers area may be quoted as a descriptive reference only, never as a grid point.
   - none@4.8e-3 seeds 0-3 (already run) is not dropped. The read-out with picks.json is reported next to it, labelled
     "pre-X1 pick (none@4.8e-3)".
   - Selection effect: 4 lrs are now compared on seeds 0-1, so the winner's curse on none@2.4e-3's numbers rows is
     larger than for a 3-lr pick. Every none_nlr row carries "X1: none's lr picked over {2.4e-3, 4.8e-3, 9.6e-3,
     1.92e-2} on seeds 0-1 (Amendment X1); lower edge OPEN".
c. Data incomplete or invalid: picks_x1.json has status "incomplete" (an X1 run missing or unfinished, an identity or
   check.json failure, or an integrity failure). X1 is then incomplete: no X1 pick. The main read-out stays with
   picks.json (none@4.8e-3), labelled "none's bracket OPEN at the lower edge; X1 incomplete". The problems go to the user.

### Effect on the main read-out
tools/rb6scale_summary.py (sha256[:12] e807e43be36b) is unchanged by X1. It takes none@best only from --picks
(design["505m"]["none_nlr"] = pick["none"]), so no new flag is needed.
- (a) and (c): `--picks runs/rb6_505m/picks.json`, as pre-registered.
- (b): `--picks runs/rb6_505m/picks_x1.json` gives none_nlr = none@2.4e-3, labelled by its actual lr; the none@4.8e-3
  runs are listed IGNORED. Also run `--picks runs/rb6_505m/picks.json`, the pre-X1 read-out, and report it alongside.
Tested on fake 505M runs (see tests). Still open and separate from X1: the summary does not print the bracket state,
although the Read-out section says an OPEN bracket is printed next to its cells. It also keeps the wrong lr-source line
(Freeze note). Both belong to the pending read-out amendment, which must also name X1's grid.

### X1 tests before submission (scripts / outputs: scale-ladder notes/rb6_505m_x1/tests/, the session scratchpad x1/)
| test | result |
|---|---|
| select regression, no flag | old (f9c1435) vs new (37d545a) rb6scale_select.py: stdout and JSON byte-identical on the 12 sweep dirs and on runs/rb6_505m/out. On the 12 dirs the new output = select.log lines 2-11 and = picks.json byte for byte |
| select --grid_none (fake none@2.4e-3 = real none@4.8e-3 s0/s1 relabelled, planted area / bpb) | SELECT X1 TEST 28/28 PASS. Clear win: 2.4e-3, OPEN lower. Lower area: 4.8e-3, closed. Tie with lower bpb: 2.4e-3. Tie with higher bpb: 4.8e-3. Exact bpb tie: the lower lr, 2.4e-3. Diverged s1: bpb worst, 4.8e-3. Without the flag on the same data: 4.8e-3 OPEN, so the flag takes effect. GRID line printed and the grid stored in the JSON. Missing, failed check.json, unfinished or wrong lr: incomplete. Five bad flag values refused. picks.json not overwritten |
| summary accepts the X1 pick (53M F88 runs with the 505M size facts as fake 505M cells) | RO X1 TEST 7/7 PASS. The X1 picks JSON gives none_nlr = none@2.4e-3, the 4.8e-3 runs IGNORED, strict complete. Every value equals the 4.8e-3 picks case; only labels differ. Without 2.4e-3 s2/s3: strict REFUSED, --allow_partial PROVISIONAL (n=2 of 4) |
| jobcheck_x1.py (the 2 X1 jobs vs the submitted sweep none@4.8e-3 job of the same seed) | JOBCHECK X1 PASS (16). bash -n passes. render() reproduces the submitted 4.8e-3 files and the X1 files byte for byte. Token diff: 2689 tokens each, 9 differ, all 4.8e-3 -> 2.4e-3: job name 1, out dir 2, tag 3, lr 2, intent JSON 1; nothing inserted or deleted; the whole file = the template with the lr substituted. ARGS differ only in the --out / --lr / --tag values; the intent only in args.lr / args.tag. lm0_ref = the 505M CPU reference. The input-check block on real GPU dry logs: own seed's none log exits 0; the other seed's exits 3 (DRY rows); an all6 log exits 3 (prior target) |
| sbatch --test-only | 2/2 accepted |

### X1 submitted (2026-10-25T09:30Z)
- Jobs 20876003 (none@2.4e-3 s0) and 20876004 (s1), via make_jobs_rb6_505m_x1.py --submit. SUBMITTED_<name> flags written.
  Code 7f1d06b, trainer 93e22bf4c4c0. /scratch before submission: 4.89 of 5.00 TB used. Each run peaks at about 5.7 GB
  while training (model + checkpoint) and keeps about 1.9 GB at the end.
- Kill switch: as for the batch (runs/rb6_505m/HOLD). Audit: minimal mode, i.e. the asserts above plus a fresh
  general-purpose review agent started right after submission (generator, X1, read-out diff). Its result is recorded
  below.

### X1 build review (2026-10-25, fresh general-purpose agent, read-only; minimal mode)
No blocking issue. The two jobs and the select diff are correct:
- Slurm's queued scripts = the job files, byte for byte; account, partition, QOS, nice, time and requeue as the template.
- The ckpt / claim paths sit in the lr-specific out dir.
- step-0 lm and the DRY row hash do not depend on lr: equal across the sweep's lrs per seed.
- Importing the generator runs nothing; the frozen files are unchanged.
- The grid is used everywhere; the default output is byte-identical, re-checked from x1_code/.
- The "9.6e-3 / 1.92e-2 cannot be picked" argument holds on picks.json's exact values.
Important: R1 (the summary's labels / bracket for none_nlr) and R2 (no resubmission rule). Minor: m3-m10. All are
handled by Addendum X1.1. The reviewer's scripts and outputs are in the session scratchpad, x1/audit_scratch/.

### Addendum X1.1 (2026-10-25, after the build review; both X1 jobs still PENDING, no 2.4e-3 result exists)
These rules replace or extend the X1 text above where they differ.
1. Running the selection (replaces the command in reading 2; review m3 / m5). runs/rb6_505m/x1_code/ is now
   chmod -R a-w. Run once, after BOTH X1 jobs have ended and while no 505M job of this batch is running (the tool
   loads every run_*.json under out/, and a running job rewrites its JSON):
   `cd runs/rb6_505m/x1_code/tools && for f in rb6scale_select.py rb6scale_summary.py rainbow6_summary.py
   make_rb6m_tasks.py make_rb6_tasks.py; do grep -q "$f $(sha256sum $f | cut -c1-12)" ../COMMIT || exit 3; done &&
   test ! -e ../../picks_x1.json && /scratch/yh6210/miniconda3/envs/open_r1/bin/python rb6scale_select.py ../../out
   --grid_none 2.4e-3,4.8e-3,9.6e-3,1.92e-2 --json ../../picks_x1.json`
   Use the env of select_job.sbatch. An integrity failure (rainbow6_summary.check asserts) stops the tool with a nonzero
   exit and writes no JSON. A nonzero exit therefore = status "incomplete" = outcome c.
2. Resubmission (new; review R2), decided before any result. A job may end with no finished run for a non-data reason:
   sacct state NODE_FAIL, PREEMPTED, TIMEOUT, or CANCELLED by the system; a full /scratch; or a requeue that never
   resumed. The coordinator then resubmits the same job file (`sbatch runs/rb6_505m/job_<name>.sbatch`; it resumes from
   the checkpoint). At most 2 resubmissions per job, each job id recorded here.
   - The decision uses only sacct's state / exit code and the job log's "## " step and error lines. It never opens
     run_*.json, check.txt / check.json or any metric line.
   - Exit 3 means an input / sha check failed: a build bug, so stop and report, with no resubmission.
   - Exit 4 means CHECK_PY failed: invalid, outcome c, with no resubmission.
   - A diverged run is data, not a failure.
   - Only once the resubmissions are used up does the cell count as missing (c).
   This rule also covers (b)'s s2 / s3.
3. Labels in (a) (review m6). Every none_nlr row carries "X1: picked over {2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2} on seeds 0-1;
   bracket closed by X1", because the winner's curse of a 4-lr pick replaces picks.json's "OPEN".
4. Primary and secondary in (b) (review m7).
   - The read-out with picks_x1.json is the PRIMARY one.
   - The read-out with picks.json ("pre-X1 pick (none@4.8e-3)") is secondary: never quoted on its own and never used to
     choose between the two. Its BH q values differ, because SE / SH mix none_plr and none_nlr rows in one family.
   - Seeds 2-3 of either none lr never take part in choosing none's lr.
5. The s2 / s3 jobs in (b) (review m8). make_jobs_rb6_505m_x1.py fixes SEEDS = (0, 1). For (b) its seeds change to
   (2, 3). jobcheck_x1.py then runs again on those 2 files, against the submitted none@4.8e-3 s2 / s3 jobs, before they
   are submitted. The worktree /scratch/yh6210/llmadd-burst-scale must still exist, since snapshot() verifies via git show.
6. Wording, n = 2 (review m9). 4.8e-3's two seeds already differ by 3.3 area points, more than the 2-point tie band. So (b)
   says "2.4e-3 was picked by the pre-registered rule", never "none's best lr moved down". The per-seed areas and bpb are
   always printed next to the pick.
7. Main read-out condition (review R1 / m4; confirmed at rb6scale_summary.py:47-49, 85-87, 216-220). The unchanged summary
   accepts the X1 pick, but its labels are wrong:
   - ROLE_DESC and the lr_provenance NOTE call none_nlr "selected on ... pt2 single-prior 2%, seeds 0-3".
   - It prints no bracket, grid or amendment.
   - It cannot tell picks.json from picks_x1.json.
   Neither picks file is used for a 505M read-out until the pending read-out amendment (Freeze note) is done. That
   amendment must:
   (i) fix ROLE_DESC / lr_provenance for 505M (the rb6_505m sweep, seeds 0-1; under (b) X1's 4-lr grid);
   (ii) print the picks file's grid / amendment / bracket next to the none cells, with the X1 labels;
   (iii) when the file has a "grid", assert grid["none"] == ["2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"] and pick["none"]
         in {"2.4e-3", "4.8e-3"} (X1 reading 3 in code);
   (iv) refuse a picks file without "grid" for the X1 read-out.
   It must also keep the 190M read-out byte-identical. "No new flag is needed" above holds only for selecting the cell.
8. Order (review m10). The 505M main read-out, and that amendment's first look at 505M values, come only after X1 has
   closed (a; b with s2-3; or c). So S1b is never reported and then changed.
