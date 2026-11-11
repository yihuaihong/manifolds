# rb6_190m — the six-prior rainbow pretraining at 190M (scale ladder, batch 1; 2026-10-22)

User approval (relayed by the coordinator): "可以的，没问题" for the model-size scaling batches, and the standing
"需要现在该提交的都可以提交". Written to this file BEFORE any job of this batch was submitted.

## Question (descriptive)
F88 (53M, Burst; report/audits/rb6_A5_main_design.md A5-A8.3): with six manifold priors as an auxiliary loss (lambda 1
each) the numbers and S4 domains passed every pre-registered rule at 53M. This batch runs the same design at 190M so the
paper can show the per-domain pattern per model size. It is **descriptive per size**: no confirmatory claim, no
headline, **no size-trend claim** (F87 rule: sizes are not tested against each other).

## Cells (190M, the 53M main design otherwise unchanged; seeds 0-3; 16 jobs)
| cell | trainer flags (= make_rb6m_tasks.arm_flags of the 53M arm) | lr | role |
|---|---|---|---|
| none@6e-4 | `--target none` | 6e-4 | "low": none at the prior arms' lr (lr-matched contrast, the F1 analog) |
| none@4.8e-3 | `--target none` | 4.8e-3 | "best": none's selected lr at pt2s190 (the F1b analog) |
| all6-lam1@6e-4 | `--target helix3 --schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains numbers,months,weekdays,clock,s4,dim --lam 1` | 6e-4 | the six priors, lambda 1 each, at the prior's pt2s190 lr |
| only-numbers@6e-4 | `--target helix3 --schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains numbers --lam 1` | 6e-4 | the numbers prior alone (the 53M only-numbers arm: helix3 on the operands, the five domain priors off) |

Every cell: `--arith_lines 1 --domains numbers,months,weekdays,clock,s4,dim --extra_lines 2` (2% numbers, the numbers line
first after the text, K = 2 domain lines per row cycling months / weekdays / clock / s4 / dim), the 190M model flags of
pt2s190 `--d_model 768 --n_layers 12 --n_heads 12 --d_ff 3072` (190,297,344 parameters; gradient checkpointing on by the
trainer's > 150M rule; prior and read-out layers 4, 8), 3e8 tokens = 18,310 steps of 32 x 512, eval every 250 steps (75
evals), `--save_model --ckpt_every 2000`; every other flag = the trainer default (asserted by the generator).
Tags `rb6tb-190m-2pct-k2-<arm>-lr<lr>` (tb = Torch batch; the summary's TAG pattern), job dirs `out/<tag>-s<seed>/`
(= a Burst results/<task> dir: dry_inputs.txt, gpu.csv, node.txt, <tag>/run_*.json + model, offline_lm.json, check.json).

## lr provenance (pre-registered caveat; the user's instruction)
The lrs are REUSED from pt2s190 (F86: 190M, 2% numbers, single numbers prior at lambda 5, NO domain lines, chosen on the
numbers area over seeds 0-3): none's best 4.8e-3 (26.8 vs 9.6e-3 28.8, tie within 2 -> lower offline bpb) and win-ctx's
best 6e-4 (78.6 vs 1.2e-3 76.9: the lower edge of its grid, an edge NOTE). Therefore:
- none is NOT tuned for S4 or any other non-number domain, nor for the rainbow mixture; "none@4.8e-3" means "none at its
  numbers-best lr of the single-prior 2% setting", never "none at its best lr" for a domain.
- all6-lam1 / only-numbers use the lr chosen for ONE numbers prior at lambda 5; six priors at lambda 1 were never tuned at
  190M.
- Seeds 0-3 here share the init and the numbers lines with pt2s190's seeds 0-3, on which both lrs were selected (the A5
  winner's-curse note): every numbers row involving these cells is "on the lr-selection seeds".

## Read-out (pre-registered; `tools/rb6scale_summary.py` on branch scale-ladder, a thin wrapper around rainbow6_summary.py)
Written and tested on fake / relabelled runs before any result of this batch is read. Command (after all 16 jobs):
```
python tools/rb6scale_summary.py /scratch/yh6210/llmadd_torch/runs/rb6_190m/out \
  results/401-rb6pb-* results/404-rb6pb-* results/430-rb6pb-* results/431-rb6pb-* \
  results/419-rb6pb-* results/415-rb6pb-* results/432-rb6pb-* results/433-rb6pb-* \
  results/438-rb6mb-* results/444-rb6mb-* results/450-rb6mb-* results/456-rb6mb-* results/46[6-9]-rb6mb-* \
  --json rb6scale_190m.json
```
(Burst paths relative to the llmadd-burst-ro root: the 53M cells none@1e-3, all6-lam1@1e-3, only-numbers@1e-3,
none@8e-3 at seeds 0-3, the F88 tasks.) Mid-batch: add `--allow_partial` (every line that misses a seed: PROVISIONAL n=k of 4).

**Metrics** (rainbow6_summary's per-domain headline read-outs, A5 READ, unchanged): numbers = 6-bucket held-out area;
months / weekdays = tier-E final3 (heldE); clock = held_wrap final3; s4 = held_fresh_nonid final3; dim = the STRICT set
final3; plus s4_hard area (A8.3's primary S4 metric: cells not solvable by Latin-square propagation), offline bpb x1e3
(full val, offline_lm.py), seen (mean seen final3 of the 5 domains) and the gradient split (LM share, prior / LM).

**Descriptive per size** (the main output): per cell and domain the mean over seeds 0-3, the per-seed values in seed
order, the seed-level 90% t interval and the A5 seed-level gate verdict (FLOOR / BORDERLINE / MEASURABLE / CEILING vs
[chance + 5, 90]; the heuristic line beside it); the 53M analog cells (none@1e-3 = low, none@8e-3 = best,
all6-lam1@1e-3, only-numbers@1e-3; Burst seeds 0-3) printed side by side. No statistic across sizes.

**Paired tests** (A5 style: per-seed difference paired by seed, paired t df 3, exact sign-flip p, 95% CI, BH q .05 within
each family), every line labelled **"on the lr-selection seeds 0-3: descriptive, no confirmatory claim"**:
| family | contrast | rows |
|---|---|---|
| S1 (F1 analog) | all6 - none@low (same lr) | numbers, clock, s4, dim |
| S1b (F1b analog) | all6 - none@best | numbers, clock, s4, dim |
| SE (FE analog) | all6 - none@low, all6 - none@best | months E, weekdays E |
| S2 (F2 / F7 numbers row) | only-numbers - none@low | the 6 domains |
| S3 (F3 analog) | all6 - only-numbers | the 6 domains |
| SH (A8.3 metric) | all6 - none@low, all6 - none@best | s4_hard area |
| SL (F8 analog) | all6 - none@low, only-numbers - none@low | offline bpb x1e3 (LM COST FLAG at a mean > +4) |
Sidedness: the s4 and s4_hard rows are one-sided (greater) iff the baseline cell's seed-level s4 gate (seeds 0-3) is FLOOR
(A5's F1b rule), else two-sided; S3 and every other row two-sided. The same families are printed for the 53M cells
(a double path: they must reproduce F88's F1 / F1b / F2 / F3 / F8 values on seeds 0-3).

**Run identity and integrity** (before any table): every 190M run's JSON against its intent (every trainer argument incl.
lr, model dims, target, prior_domains, lam, extra_lines, tag, seed; code_sha 93e22bf4c4c0, toy_struct d182ce548bcc,
n_params 190297344, gradient checkpointing on, layers [4, 8]; lam_domains; the active / shuffled prior sets; the logged
multiplier gnorm_prior / gnorm_true = lambda) and check.json PASS next to it; a run failing any of them is EXCLUDED with
the reason. rainbow6_summary.check per size (hits = accuracies double path, layout, window, the trainer's metrics = the
summary's); per seed every arm has the same data_sha at every eval and the same step-0 lm (1e-3); NEW: per seed the 190M
runs' data_sha series = the 53M Burst runs' (the data are size-independent). A missing / excluded / unfinished run makes
every line that needs it PROVISIONAL (n=k of 4); strict mode refuses. A diverged run is data (A6 rule: its values over
all evals count, its bpb is the worst / NOT TESTED).

## Code
- Trainer: llmadd-burst origin/main 4908225 `tools/pretrain_range.py` 93e22bf4c4c0 (Rev D), `toy_struct.py` d182ce548bcc;
  imports pretrain_helix 25d5a35e6f17, numlib d09bb47b04c2, digit_split_data 742b922c8125, run_guard b944dc859c7a;
  offline_lm 226f4cac70ea (= c434d29's and 64abc19's). The 53M rainbow ran pretrain_range cc984148b15a / toy_struct
  6248613ac6d2 (c434d29); pt2main a3bce3ab63c1 (64abc19). The chain cc984148b15a -> 3c99f6f8cb17 -> 93e22bf4c4c0 was
  regression-tested only with `--domains numbers,months` (runs/ptheld, runs/ptheld2), so a direct CPU regression of the
  cell types of this batch was run (below).
- Job code: runs/rb6_190m/code = git archive of tools/ at the committed HEAD of branch scale-ladder (worktree
  /scratch/yh6210/llmadd-burst-scale; not pushed); code/COMMIT records HEAD and the sha256[:12] of every file above.
- Generator: llmadd_torch/make_jobs_rb6scale.py (asserts the snapshot = HEAD and = the regression-tested shas; every
  cell's flags present / absent; intent = make_rb6m_tasks.intent of the 53M arm with the size facts; CHECK_PY =
  make_rb6m_tasks.CHECK_PY verbatim; refuses without a step-0 lm or DRY reference; refuses an out dir of another batch).
- References: llmadd_torch/rb6scale_refs.json (written by the session scratchpad's scale/refs.py; copy and logs in
  llmadd_torch/scale_work/).

## Checks inside every job (exit 3 before training, exit 4 after)
1. HOLD kill switch; the sha256[:12] of the 7 snapshot files; COMMIT names the trainer; an A100 guard (`*A100*`); the data
   sha256 (train.bin 4de2ed19..., val.bin d6f744a9...).
2. `--dry_inputs` (the job's own ARGS): the model line `model: 190.3M parameters; gradient checkpointing on`; the numbers
   target line (helix3: (1000, 10); none: absent); for each of the 6 domains prior / shuf / lam in the `domains:` line;
   the numbers line first in every row; the content scan; the 88 DRY_GREP lines' sha256 = the Torch CPU dry run of the
   seed = the Burst 53M seed's rows with the model line replaced (asserted by the generator and by refs.py).
3. Training, offline_lm.py (full val), then make_rb6m_tasks.CHECK_PY with the cell's intent -> check.json (every
   trainer argument; code / toy shas; steps, n_params, gradient checkpointing, layers, lm_chunk 0, eval spacing; lambda
   per domain; domains.<d> prior / shuf / lam; finished 75 evals; data_sha after step 0 = the 53M seed's (size-independent,
   reproduced on CPU at 190M); step-0 lm within 1e-3 of the 190M CPU reference; layout, token share 1.9-2.4%, slots,
   splits = the 53M seed's, held-out formats; the prior window, every active prior on / applied / logged / weighted,
   every inactive one never; the logged multiplier = lambda; geo > -1e-3; the CKA-rise bar 0.1; offline double path
   < 0.002) -> exit 4 on a fail. The info line: gradient split, versions, sec/step, GPU util / memory.

## Tests before submission (CPU)
Scripts and logs: the session scratchpad's scale/ (reg.py, ro_test.py, sel_test.py, jobcheck.py) and
llmadd_torch/scale_work/ (refs.py, step0_ng.py, e2e.py and their logs / outputs).
| test | result |
|---|---|
| CPU regression (reg.py; tiny d64 x 3, seq 64, 40 steps, 5 evals, one thread, old and new trainer side by side on one host) | origin/main trainer vs c434d29 (cc984148b15a / toy 6248613ac6d2): rb6-all6 (six priors, lambda 1, lam_rule each, K 2), rb6-none, rb6-onlynum: IDENTICAL (run JSON: every value equal except code_sha, dep_sha toy_struct, args.out, the per-eval wall-clock 'sec'; saved model 36/36 tensors bitwise; --dry_inputs stdout byte-identical; run stdout equal after removing 'sec' and the warning's file path); rb6-all6-ckpt (six priors, --grad_ckpt on = the 190M / 505M state, lr 4.8e-3, seed 2): IDENTICAL the same way (36/36 tensors bitwise). The pt2 / none-ckpt configs belong to batch 3 (runs/pt2s505/AUDIT.md) |
| CPU references (refs.py on the snapshot trainer, the job's own ARGS) | --dry_inputs of 8 (arm, seed) cells at 190M: per seed the 88 DRY rows = the Burst 53M seed's rows with the model line `model: 190.3M parameters; gradient checkpointing on` (4/4 seeds), DOMS = make_rb6m_tasks.DOMS, eval rows identical across arms, every domain's prior / shuf / lam = the arm's intent, target line as intended; step-0 lm (step0_ng.py, CPU bf16 autocast) seeds 0-3 = 11.646783 / 11.681499 / 11.659709 / 11.640708 (all6 s0 = none s0 bitwise); data_sha after step 0 = the 53M DSHA0 of every seed (size-independent); calibration of the CPU method at 190M on the default path (pt2s190 none@4.8e-3 s0): CPU 11.645783 vs the Burst A100 run 11.645764 (+1.9e-5, inside the check's 1e-3) |
| jobcheck.py (the 16 rendered jobs) | bash -n 16/16; the CHECK_PY heredoc = the Burst task's (430 / 432 / 450) byte for byte 16/16; the embedded intent differs from the Burst seed-2 task's only in the size facts (code / toy sha, n_params, gradient checkpointing, layers, step-0 lm; data_sha0 / splits for seeds != 2) and args (d_model, n_layers, n_heads, d_ff, lr, tag, seed) 16/16; the input-check block = the Burst task's except the model line / DRY / pin / ARGS lines 16/16; the block executed on the 8 Torch CPU dry logs: 13 own cases exit 0, 115 cross cases exit 3 with the right reason (another arm: '## not a helix3 table' / '## domain d not prior ...'; another seed: '## dry_inputs differ') |
| e2e.py (the rendered 190M job script run end to end with bash on CPU: tiny overrides appended to ARGS, --grad_ckpt on, the tiny dry's DRY, the tiny facts in the intent, a fake nvidia-smi reporting an A100) | all6-lam1 s1 and only-numbers s2: job rc 0, every CHECK_PY check PASS (25); none s0: the pipeline ran through (dry checks, training, offline_lm, CHECK_PY, exit 4 on the failed check as designed), 15/16 checks PASS; the one FAIL is the slot balance at 40 steps (weekdays 0.44 / 0.56 vs 1/2 +- 0.05; the seed-0 tiny data, the same values in both trainers' regression runs): a small-sample statistic at 40 x 32 rows; the production runs see exactly the 53M Burst runs' data (data_sha checked at step 0 by CHECK_PY and at every eval by the read-out), on which this check passed (tasks 401, 404, 430, 431). The tiny runs' step-0 lm = the CPU reference bitwise (3/3) |
| read-out tests (ro_test.py on relabelled real runs) | RO TEST PASS (18): the 53M S1 / S1b rows reproduce F88's F1 / F1b (numbers +53.2 / +32.1, clock +61.6 / -3.3, s4 +39.8 / +37.0, dim +47.3 / +28.1); the 53M runs relabelled as 190M give every cell value, gate verdict and family row identically; t p = scipy; a mislabelled lr / a failed check.json / a wrong logged multiplier -> EXCLUDED; a missing run -> strict refusal, --allow_partial PROVISIONAL n=3 of 4; a diverged run -> data with a DIVERGED warning; a data_sha mismatch within or across sizes -> AssertionError |
| sbatch --test-only | 16/16 accepted (a100_tandon, start estimates 06:02-09:47) |

## Jobs
16 jobs, account <slurm-account>, a100_tandon, QoS gpu48, `--nice=20`, 1 GPU, 8 CPUs, 64G,
`--time=05:30:00` (2 x 2.1 h x 1.2 = 5.04 h for the rainbow, rounded up to the half hour), `--requeue` (a requeued job resumes from its checkpoint;
run_guard keeps one writer). Planned ~16 x 2.6 = 42 A100-h.

## Audit status (experiment-audit, minimal mode)
asserts in the generator and in every job (above); decoded training / eval rows: dry_inputs.txt of every job and the CPU
dry logs (item vii); known answers: the DRY rows = Burst's, data_sha / splits = the 53M seeds', the regression; the
fresh adversarial review is launched by the coordinator after submission.

## Submitted (2026-10-22 ~04:31 EDT)
16 jobs 20599674-20599692 (SUBMITTED_<name> flags hold sbatch's output); code c6dcb6e (runs/rb6_190m/code/COMMIT), trainer
93e22bf4c4c0; logs llmadd_torch/logs/llmadd_rb6_190m-<arm>-lr<lr>-s<seed>_<jobid>.out. Kill switch: touch runs/rb6_190m/HOLD
(pending jobs exit at start).

## Amendment R1 (2026-10-22, after the fresh build audit, BEFORE any 190M result was read; read-out wording only)
Audit: notes/batch1_build/audit_rb6_190m_build.md (no blocker; no HOLD). Jobs, trainer, generator and references unchanged.
tools/rb6scale_summary.py 6359b38b245d (c6dcb6e) -> e807e43be36b (b417a3e, branch scale-ladder):
- Q4: A5's INVALID rule for the none cells (a domain whose seen final3 < 80 in any seed is INVALID; per-domain seen
  printed); a test is one-sided on s4 / s4_hard only against a FLOOR baseline, never against an INVALID one (noted).
- Q5: cells are labelled by their actual lr (none@6e-4, none@4.8e-3, ...), never "none@best"; "none's lr not tuned for
  <domain> (chosen on numbers)" stands next to every none verdict line and every non-numbers row against a none cell, and
  in the JSON (caveats, per cell, per row).
- Q7: SH reads "s4_hard area vs none@<lr>, seeds 0-3, descriptive (A8.3's METRIC, NOT the A8.3 comparison: A8.3 =
  all6@1e-3 vs the S4-tuned none@1.6e-2 on the fresh seeds 8-15)"; the 190M S4 / s4_hard rows say "190m has no none whose
  lr was tuned on S4: not comparable with A8.3"; dim / clock rows carry the 53M findings (dim: none@8e-3 MEASURABLE;
  clock: the S1 gain was an lr effect) and the 190M "not tuned" note.
- Q8: every prior cell and seed reports the window medians of the LM share, prior / LM, the clip factor 1 / max(1, T) of
  the logged decomposition and the share of on-evals with prior > LM; A5's FLAG when a seed's median prior / LM > 1; NOT
  LOGGED when a run lacks the decomposition (the trainer's own grad_total / clip_coef is not logged on this path: printed
  as NOT LOGGED).
- Minor: the 53M SE rows are labelled "seeds 0-3 incl. the discovery seeds: NOT F88's FE (seeds 2-7); descriptive".
Tests: ro_test.py 32/32 PASS (+ an INVALID scenario: none@6e-4 s1 s4 seen final3 lowered to 25 -> INVALID, its S1 / SH s4 rows
two-sided with the note, the other cells unchanged; + a gradient scenario: no decomposition -> NOT LOGGED, prior = 2 x LM
-> FLAG); the 53M S1 / S1b rows still = F88's F1 / F1b; sel_test PASS.
Not changed here (the coordinator's / user's call, audit H1(b)): no S4-tuned none cell at 190M.
