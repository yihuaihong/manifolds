# rb6_190m_dim — 190M dim closing batch: none one lr higher, then confirm seeds (2026-10-25)

User approval (2026-10-25, relayed by the coordinator): "可以放在队伍的最后面，如果你感觉这些任务有必要的话". The
coordinator judged the batch necessary. "Queue last" is implemented as `--nice=5000` on every job (below).

Context (FINDINGS F88 补记二, rb6_190m_ext): at 190M, dim strict final3 = all6-lam1@6e-4 89.4, @1.2e-3 76.0; none@6e-4
14.3, @4.8e-3 12.8, @9.6e-3 22.1, @1.92e-2 40.1. none's lr bracket is open at the top (still rising), all6@6e-4 - the
best tested none = +49.3 (4/4, t-p .011, BH .065), no confirm seed. On S4, none@1.92e-2 (60.5) is already above all6
(44.5). At 53M (F88 A6) the dim best-vs-best was only +12.8, n.s.

## Pre-registration (written before any job of this batch was submitted)
Written before any job of this batch was submitted and before any of its results exists. The only results used are the
already reported rb6_190m / rb6_190m_ext values above.

### Question
On dim (strict final3) at 190M: does all6-lam1@6e-4 stay above none at none's best dim lr, once none's lr is pushed one
step higher and the comparison is repeated on fresh seeds? The answer is used ONLY to decide whether the dim cell may go
back into Table 1; the user has decided not to put it there for now. No size comparison (only 190M runs are read).

### Design (190M, rb6_190m's design, data and code; only the listed things change)
Everything as runs/rb6_190m/AUDIT.md: 2% numbers first, K = 2 domain lines over the six domains, window (0.05, 0.4),
ctx site, lam_rule each, lambda 1 per active prior, d 768 x 12 layers, 12 heads, d_ff 3072 (190,297,344 parameters,
gradient checkpointing on, prior / read-out layers 4, 8), 3e8 tokens = 18,310 steps of 32 x 512, eval every 250 steps,
`--save_model --ckpt_every 2000`. Arms = make_rb6m_tasks.arm_flags: none = `--target none`; all6-lam1 = `--target helix3
--schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains numbers,months,weekdays,clock,s4,dim
--lam 1`. Tags `rb6tb-190m-2pct-k2-<arm>-lr<lr>`, job dirs `runs/rb6_190m_dim/out/<tag>-s<seed>/`.
1. **Stage D1** (4 jobs): none@3.84e-2 (= 1.92e-2 x 2), seeds 0-3. Purpose: push none's dim lr bracket one step up.
   3.84e-2 may diverge: a diverged run is data (its values over all evals count; its bpb is the worst / NOT TESTED).
2. **Selection** (after D1, automatic): see the rule below; a CPU job with `--dependency=afterany:<the 4 D1 job ids>`.
3. **Stage D2** (8 jobs, submitted by the selection job only if its status is "picked"): none@best_dim seeds 4-7 and
   all6-lam1@6e-4 seeds 4-7. D2 is submitted in every "picked" case, OPEN included (OPEN changes the reading, not the run
   plan; no further lr is added automatically). Reuse: a cell / seed that already exists in runs/rb6_190m or
   runs/rb6_190m_ext (the same code snapshot c6dcb6e) is REUSED, not re-run (the generator checks it). Verified
   2026-10-25: no 190M run at seeds 4-7 exists in llmadd_torch/runs/*/out (only seeds 0-3 of the rb6 cells; the
   ptlr / ptsize999 190M runs are other designs), so all 8 D2 jobs are new.

### Selection rule (tools/rb6scale_dim_summary.py select; tested before submission)
Over none@{6e-4, 4.8e-3, 9.6e-3, 1.92e-2, 3.84e-2}, seeds 0-3 (the first four from runs/rb6_190m, runs/rb6_190m_ext):
1. the lr with the highest mean dim strict final3 (rainbow6_summary READ['dim'] = the STRICT set final3);
2. every lr within 2 points of the top (>= top - 2) is tied -> the lower mean in-training ("online") bpb (the last eval's
   val_bits_per_byte, mean over seeds 0-3; a non-finite value is the worst) -> then the lower lr
   (= rb6scale_select.select; second path readout_rules.best_lr with the in-training bpb, asserted equal);
3. bracket: **OPEN (upper edge)** iff the pick is 3.84e-2 AND its mean is more than 2 points above 1.92e-2's
   (= readout_rules.bracket 'open upper', asserted). Then the reading is "not best-vs-best" and no further lr is added
   automatically. (A lower-edge pick 6e-4 beating 4.8e-3 by > 2 would also be OPEN; impossible here: 14.3 < 40.1.)
   A pick at 3.84e-2 that is within 2 of 1.92e-2 is closed (with a note that the argmax is at the edge).
4. A diverged run is data: its dim counts, its bpb is the worst. A5's INVALID flag (dim seen final3 < 80 in a seed) is
   printed beside each lr but does NOT exclude it (the rule above has no such exclusion).
5. Complete = all 20 runs present, finished (last eval at step steps - 1), identity clean (rb6scale_summary
   .torch_identity: every trainer argument = the cell's intent incl. lr, the run facts, code / toy shas, lambda, logged
   multiplier, check.json PASS) and rainbow6_summary.check passes (same data_sha at every eval and the same step-0 lm per
   seed across cells; the trainer's metrics = the summary's; a non-finite value is allowed only in none@1.92e-2 /
   none@3.84e-2). Otherwise status "incomplete", NO pick, D2 is NOT submitted; the problems go to the user.
6. The selection runs once: picks.json is never overwritten.

### Reading (pre-registered; tools/rb6scale_dim_summary.py readout; pt2main rule 2 = readout_rules.survives)
Primary row: **dim strict final3, all6-lam1@6e-4 - none@best_dim**, paired by seed.
- **SURVIVES** needs, on seeds 0-3: mean > 0, >= 3/4 seed wins, two-sided paired t p < .05; AND on the confirm seeds 4-7:
  mean of the same sign (> 0) and >= 3/4 wins. One pre-registered primary test: no multiplicity correction. The
  selection is on seeds 0-3 (a winner's curse in none's favour = against all6); seeds 4-7 are selection-free.
- Otherwise, in this order:
  - **INCOMPLETE** (数据不全): no selection (D1 incomplete), or seeds 0-3 pass but the confirm seeds are missing (PROVISIONAL);
  - **OPEN** (区间开着): the bracket is open -> "all6 - the best TESTED none, not best-vs-best"; the SURVIVES criteria
    against that none are printed beside it, but the label is never SURVIVES;
  - **REVERSED**: mean <= 0 on seeds 0-3 with p < .05;
  - **NOT SIGNIFICANT** (不显著): mean <= 0 on seeds 0-3 (p >= .05), or mean > 0 with >= 3/4 wins but p >= .05;
  - **DIRECTION UNSTABLE** (方向不稳): mean > 0 but < 3/4 wins on seeds 0-3, or seeds 0-3 pass and the confirm seeds 4-7 do
    not (mean <= 0 or < 3/4 wins).
- Descriptive, no reading: the same contrast (all6-lam1@6e-4 - none@best_dim, none's lr chosen on dim) on s4 (fresh non-id
  final3) and numbers (6-bucket area), seeds 0-3 and 4-7 separately; the read-out prints that on S4 none@1.92e-2 is
  already known to be above all6 (60.5 vs 44.5).
- Per cell: dim, s4, numbers, offline bpb x1e3, dim seen, per seed; the dim seed-level gate; the lm curve of none@3.84e-2
  (divergence onset, the largest rise above the running minimum, flagged "inspect" above 0.5 nats).
- Gradient: every all6-lam1@6e-4 seed (0-7): LM share, prior / LM, share of on-evals with prior > LM, clip factor
  1 / max(1, T), share of on-evals clipped (T > 1) (rainbow6_summary.grad_split / clip_stats); FLAG at a median prior / LM > 1.
- Run identity and integrity as the selection; strict mode refuses on any missing / excluded / unfinished run;
  `--allow_partial` prints PROVISIONAL; `--picks <picks.json>` must equal the recomputed selection (else it refuses).
- Not allowed in the write-up: a size comparison; "best-vs-best" when OPEN; "confirmed" without the confirm seeds; using
  the S4 / numbers rows as tests.
Command (after all 12 jobs):
```
python tools/rb6scale_dim_summary.py readout /scratch/yh6210/llmadd_torch/runs/rb6_190m/out \
  /scratch/yh6210/llmadd_torch/runs/rb6_190m_ext/out /scratch/yh6210/llmadd_torch/runs/rb6_190m_dim/out \
  --picks /scratch/yh6210/llmadd_torch/runs/rb6_190m_dim/picks.json --json rb6_190m_dim.json
```

### Queue position ("排在最后", the user's request)
Every job of this batch: `#SBATCH --nice=5000` (all other llmadd jobs: nice 20). Slurm priority weights here: age <= 1000,
partition 10000, fairshare and job size equal for the account's own jobs, so a nice of 5000 keeps these jobs behind every
other job of this account in the same partition / QoS, however long they wait. Same account / partition / QoS as
rb6_190m_ext: torch_pr_520_tandon_priority, a100_tandon, gpu48, 1 GPU, 8 CPUs, 64G, `--time=05:30:00`, `--requeue`.
The selection job: cpu_short, 2 CPUs, 24G, 2 h.

### Cost
12 x ~2.5 A100-h (rb6_190m_ext's 190M runs took ~2.4 h) = ~30 A100-h (D1 ~10, D2 ~20). Disk: ~0.76 GB saved model per run
= ~9 GB (+ temporary checkpoints during a run); /scratch at ~93% of 5 TB (myquota before submission).

### Code
- Trainer and job code = runs/rb6_190m's exactly: runs/rb6_190m_dim/code = git archive of tools/ at c6dcb6e with
  rb6_190m's COMMIT file (make_jobs_rb6_190m_ext.snapshot: tree sha256 = runs/rb6_190m/code file for file, asserted at
  every generator call); WANT_SHA 7/7; every job re-checks them (exit 3).
- Generator: llmadd_torch/make_jobs_rb6_190m_dim.py, NOT a copy. It imports make_jobs_rb6scale.py (read-only,
  19bc055c2d09; rb6scale_refs.json b65472eb3d29) and reuses its render() with all its asserts, and
  make_jobs_rb6_190m_ext.py (ba2eb39015af) for snapshot() only; ast asserts that neither runs code on import. The job files
  say "generated by make_jobs_rb6scale.py" (render() is that file's). The generator changes exactly one line of render()'s
  output: `#SBATCH --nice=20` -> `#SBATCH --nice=5000`. Seeds 4-7: the batch's CPU references
  (scale_work/rb190dim_refs.json) are added in memory to make_jobs_rb6scale's LM0 / DRY / BURST_DRY (asserted absent
  before); render() then checks the DRY hash against the Burst 53M seed's rows (tasks 470 / 472 / 474 / 476) with the 190M
  model line. Every rendered (arm, lr, seed) needs its own CPU --dry_inputs reference (refs gate). Submitted job files are
  never rewritten (SUBMITTED_<name> flags). --submit refuses without this section.
- Selection job: runs/rb6_190m_dim/select_job.sbatch (rendered by the generator at D1): HOLD; the sha of every file of the
  frozen read-out tool (runs/rb6_190m_dim/select_code = git archive of tools/ at the read-out commit, COMMIT file); the
  generator's and the CPU references' sha at D1 time; refuses if picks.json exists; then `rb6scale_dim_summary.py select`
  and, only if "picked", the generator's `--stage D2 --picks picks.json --test-only` and `--submit`. Log runs/rb6_190m_dim/select.log.
- Kill switch: `touch runs/rb6_190m_dim/HOLD` (pending GPU jobs and the selection job exit at start).
- The worktree /scratch/yh6210/llmadd-burst-scale (branch scale-ladder) is read by the selection job's generator call
  (git show c6dcb6e in snapshot()): do not remove it before D2 is submitted.

## Tests before submission (2026-10-25; scripts llmadd_torch/scale_work/rb190dim_scripts/, outputs scale_work/rb190dim*)
| test | result |
|---|---|
| snapshot | runs/rb6_190m_dim/code = git archive c6dcb6e + rb6_190m's COMMIT; tree sha256 = runs/rb6_190m/code file for file (asserted at every generator call); WANT_SHA 7/7 |
| CPU references (refs_dim.py in CPU job 20933476 on cs628, the jobs' own ARGS; rb190dim_table.txt) | --dry_inputs 28/28 rc 0: none@3.84e-2 s0-3 rows = rb6_190m's DRY of the seed, eval lines and the whole dry log except the arm lines = the main batch's dry log; seeds 4-7 (none@5 lrs, all6-lam1@6e-4): one DRY hash per seed = the Burst 53M seed's rows (tasks 470 / 472 / 474 / 476) with the 190M model line, DOMS = make_rb6m_tasks.DOMS, the whole log except the arm lines identical across the 6 cells of a seed, priors / target line = intent. Step 0 (one host): s0 none@6e-4 = none@3.84e-2 bitwise, +2.5e-5 vs the stored LM0[0] (host calibration < 1e-4); s4 none@6e-4 = none@3.84e-2 = all6-lam1 bitwise, s7 none@6e-4 = none@1.92e-2 = all6-lam1, s5 / s6 none = all6-lam1; data_sha after step 0 = DSHA0 every record. LM0 s4-7 = 11.640874 / 11.667198 / 11.668338 / 11.678402 -> scale_work/rb190dim_refs.json (sha256[:12] 55f901755cde), ALL IDENTITIES HOLD |
| e2e_dim.py (the rendered job end to end on CPU, tiny overrides, fake A100; e2e_ext.py's method) | 3/3 job rc 0 and every CHECK_PY check PASS: none@3.84e-2 s1 (D1), all6-lam1@6e-4 s5 and none@1.92e-2 s6 (D2, seed-4-7 references in memory); run JSON priors as intended (all6: six on, multiplier 1; none: all off) |
| jobcheck_dim.py | JOBCHECK DIM PASS (105): the 4 D1 files vs the submitted rb6_190m none@4.8e-3 job of the same seed: after the allowed substitutions (nice, out / R, job name, tag, lr) token-identical, intent differs only in args.lr; vs the submitted rb6_190m_ext none@1.92e-2 jobs: identical except comment lines; D2 rendered in memory through build() for each of the 5 possible picks (40 jobs) vs rb6_190m none@4.8e-3 / all6-lam1@6e-4 at seed s - 4: token-identical after nice / out / name / tag / lr / seed / DRY-hash substitutions, intent differs only in args.seed / lm0_ref / data_sha0 / dom (+ args.lr for none); none REUSED (no 190M seed-4-7 run exists); bash -n and exactly one `--nice=5000` line in every job; a picks file with status incomplete / an lr outside the grid / a wrong tool sha is refused |
| ro_dim_test.py (tools/rb6scale_dim_summary.py on unit cases + fake trees of copied real runs) | RO DIM TEST PASS (45): choose() ties within 2 -> lower bpb, exactly 2.0 = tied, 2.1 -> OPEN, exact tie -> lower lr, NaN bpb worst, middle pick closed; verdict() every label (SURVIVES, NOT SIGNIFICANT x2, REVERSED, DIRECTION UNSTABLE x2, INCOMPLETE x2, OPEN over SURVIVES). Trees: base (real 190M runs, none@3.84e-2 = relabelled none@1.92e-2, seeds 4-7 = the 53M confirm runs 470-477 with the 190M size facts) -> pick 1.92e-2 (exact tie, lower lr), means = FINDINGS, SURVIVES, the seeds 0-3 row = the stored rb6_190m_ext Q3 dim row exactly (+44.3 / 42.6 / 74.7 / 35.5, p .011), the confirm row = the source runs' values; ns -> NOT SIGNIFICANT; unstable03, confirm_disagree -> DIRECTION UNSTABLE; open -> pick 3.84e-2, OPEN, not best-vs-best; diverged (s1 non-finite) -> data, worst bpb, DIVERGED step printed; nanother -> stops; incomplete_d1 -> no pick, strict refuses, --allow_partial INCOMPLETE; incomplete_d2 -> strict refuses, partial INCOMPLETE (PROVISIONAL n=3); unfinished / failcheck / wronglr -> select incomplete (failcheck EXCLUDED); datasha -> AssertionError; flags: --picks agreeing printed, --picks disagreeing refused, without --picks the same reading, select never overwrites its JSON, the flags used printed |
| regression (reg_readouts.sh; rb190dim_reg/) | the rb6_190m read-out (its AUDIT command) and the rb6_190m_ext read-out: text and JSON byte-identical before (scale-ladder b828949) and after the read-out commit 8251667; the ext output = the read-out FINDINGS F88 补记二 used (e.g. dim all6@6e-4 - none@1.92e-2 +49.3, t-p .011) |
| select_plumbing_test.py (select_job_text() on fake batch dirs, the generator call replaced by echo) | SELECT PLUMBING TEST PASS (7): the real select_job.sbatch = the rendering; picked -> D2 --test-only then --submit with that picks.json; incomplete -> NOT PICKED, no call; HOLD -> exits; picks.json present -> exit 3 untouched; a changed select_code file -> exit 3 |
| generator CLI | --stage D2 --picks <the plumbing test's picked file> --print renders none@1.92e-2 / all6-lam1@6e-4 x seeds 4-7 (the 8 files deleted again); --stage D1 --test-only: 4/4 accepted (start estimate 2026-10-29 03:11, queued last), select job accepted (cpu_short) |
| read-only files | make_jobs_rb6scale.py 19bc055c2d09, rb6scale_refs.json b65472eb3d29, make_jobs_rb6_190m_ext.py ba2eb39015af: sha256 before = after; tools/rb6scale_summary.py f856bd596db4, rb6scale_ext_summary.py b5f3b5e6156e, rainbow6_summary.py f41d8b85a97d, rb6scale_select.py 0b1b6cb88ebd, readout_rules.py e91ee8a8a846 unchanged |

## Audit status (experiment-audit, minimal mode)
ii asserts (generator, every job, the read-out); CPU dry runs of every renderable (arm, lr, seed) and the e2e runs (vii); known
answers (DRY rows / step-0 lm; the read-out on relabelled real runs reproduces the stored ext row); a fresh adversarial
review of the generator, this pre-registration and the read-out after submission (v). i / iii / iv: - (minimal mode; the
metrics and the job pipeline are rb6_190m's, already audited).

## Submitted (2026-10-25 ~22:35 EDT)
D1: 4 jobs 20936532-20936535 (none@3.84e-2 s0-3; SUBMITTED_<name> flags hold sbatch's output); priority 6644 at
submission vs ~11,700 for this account's other pending jobs (nice 5000 = the user's "queue last"); account
torch_pr_520_tandon_priority, a100_tandon, gpu48, 1 GPU, 8 CPUs, 64G, `--time=05:30:00`, `--requeue`. Code c6dcb6e
(= runs/rb6_190m/code), trainer 93e22bf4c4c0. Selection + automatic D2: job 20936536 (select_job.sbatch, cpu_short,
`--dependency=afterany:20936532:20936533:20936534:20936535`; SUBMITTED_select); read-out tool frozen in
runs/rb6_190m_dim/select_code (scale-ladder 8251667, rb6scale_dim_summary.py 70954a2d0c56). Frozen (chmod a-w) until D2 is
submitted: make_jobs_rb6_190m_dim.py (sha256[:12] e113d287fa37, checked by the select job), scale_work/rb190dim_refs.json
(55f901755cde, checked), runs/rb6_190m_dim/select_code. Logs: llmadd_torch/logs/llmadd_rb6_190m_dim-<arm>-lr<lr>-s<seed>_<jobid>.out,
the selection in runs/rb6_190m_dim/select.log. Planned ~12 x 2.5 = 30 A100-h, ~9 GB of saved models (/scratch at 93% of
5 TB before submission). Kill switch: touch runs/rb6_190m_dim/HOLD.

## Amendment M1 (2026-10-25, after the fresh build review; BEFORE any result: the 5 jobs are pending, out/ is empty)
Review (fresh general-purpose agent, read-only): no blocker. Changes to the pre-registration, from its non-blocking points:
- a (pinned read-out): the read-out runs the FROZEN copy, not the live branch:
  `cd /scratch/yh6210/llmadd_torch/runs/rb6_190m_dim/select_code/tools && python rb6scale_dim_summary.py readout <the three outs> --picks ../../picks.json --json <out>`
  (rb6scale_dim_summary.py 70954a2d0c56 = scale-ladder 8251667; the reader also checks that picks.json's tool_sha equals it).
- b (recovery if D1 is incomplete, decided now): a D1 job that fails for an infrastructure reason (node, preemption,
  OOM, timeout; no check.json) is re-run with its identical, unchanged job file; then picks.json (status incomplete) is
  moved to picks_incomplete_<n>.json and the same select_job.sbatch is resubmitted with afterany on the re-run. A run that
  finished with a FAILED check is reported to the user, not re-run.
- c (a failed D2 submission is not a failed job): the selection job exits 0 either way; after it runs, select.log must
  show 8 "Submitted batch job" lines and runs/rb6_190m_dim must have 8 new SUBMITTED_ flags (seeds 4-7), else the user
  is told.
- d (inputs read at D2 time, do not remove until D2 is submitted): the worktree /scratch/yh6210/llmadd-burst-scale, the
  Burst results /scratch/yh6210/llmadd-burst-ro/results/{470,472,474,476}-*/dry_inputs.txt, make_jobs_rb6_190m_ext.py
  (ba2eb39015af, asserted on import).
- e (named deviation): rb6_190m_ext's "best tested" excluded A5-INVALID configs; this selection does not (rule 4 above).
  It only matters if a 3.84e-2 seed diverges while the others score high; it is the direction that can only make the
  comparison harder for all6 (an OPEN, not a dropped cell).
- g (wording): the OPEN label quotes readout_rules.survives' text against the best tested none (e.g. "...; vs it:
  SURVIVES tuning ..."); that inner text is NOT a SURVIVES verdict. Only a label that starts with "SURVIVES" is one.
- f (cost, for the user): D2 (~20 A100-h) also runs when the bracket is OPEN, as designed; to skip it in that case,
  touch runs/rb6_190m_dim/HOLD after D1 and before the selection job starts.
