# pt2s505 — F86's 2% single-prior main setting at 505M (scale ladder, batch 3; 2026-10-22)

User approval (relayed by the coordinator): "可以的，没问题" for the model-size scaling batches, and the standing
"需要现在该提交的都可以提交"; the confirm seeds are submitted automatically after the selection (no further approval).
Written to this file BEFORE any job of this batch was submitted.

## Design = pt2main (report/audits/pt2main_AUDIT.md), at 505M
- Arms: none (`--target none`); win-ctx (`--target helix3 --schedule window --window 0.05,0.4 --site ctx`, lambda 5 = the
  trainer default, as pt2main); `--arith_lines 1` (2%: one arithmetic line per row, the only line, at the eval position),
  3e8 tokens = 18,310 steps of 32 x 512 on pretrain_data, eval every 250 steps (75 evals), `--save_model --ckpt_every
  2000`; every other flag = the trainer default (asserted).
- Model = runs/ptlr's 500m: `--d_model 1024 --n_layers 24 --n_heads 16 --d_ff 4096` (505,437,184 parameters, gradient
  checkpointing on, prior / read-out layers 8, 16).
- Grids, seeds 0-3 (24 jobs): none 4.8e-3 / 9.6e-3 / 1.92e-2; win-ctx 3e-4 / 6e-4 / 1.2e-3. Why: at 190M 2% none's best was
  4.8e-3 (9.6e-3 tied) and win-ctx's 6e-4 (the lower edge); at 505M 7% none rose to 9.6e-3 (62.1) with 1.92e-2 queued and
  the prior ran at 1.2e-3. 1.92e-2 may diverge: a diverged run is data (pt2main rule 1 / 6).
- Tags `pt2s505-lr<lr>-<arm>` (unique: used by no other batch), job dirs `out/<tag>-s<seed>/` (= a Burst results/<task>
  dir).

## Read-out and selection (PRE-REGISTERED: pt2main's own rules, by its own code)
`python tools/pt2main_summary.py --gen /scratch/yh6210/llmadd_torch/make_jobs_pt2s505.py --res runs/pt2s505/out
[--json <file>]` (branch scale-ladder; the new flags only point the unchanged read-out at this batch; without them its
output on the Burst results is byte-identical, tested). pt2main's rules apply unchanged (readout_rules.py):
1. best lr per arm = the argmax of the mean area over seeds 0-3 over the arm's grid; every lr within 2 points of the top
   -> the lower mean bpb (offline full val if every compared run has it, else in-training for all, marked); a diverged
   run keeps its area, its bpb is the worst. Open bracket = the best at an edge beating its neighbour by > 2; an area
   argmax at the edge by <= 2 = a note.
2. "the 2% speed-up survives tuning" = survives(win-ctx@best, none@best): mean > 0, >= 3/4 wins, paired t p < .05 on
   seeds 0-3, and the confirm seeds 4-7 (same sign, >= 3/4) when both cells have them.
6. integrity: a missing / failed check.json, a step-0 lm off its block-seed median or the same-seed CPU reference by
   > 1e-3, or no valid full-val offline entry -> every verdict using the cell is WITHHELD; unfinished / missing ->
   incomplete.
No size-trend claim (F87); the 505M rows are read next to pt2b / pt2s190 descriptively.

## Confirm seeds (automatic; 8 jobs)
A CPU job (`llmadd_pt2s505-select`, cpu_short, `--dependency=afterany:<the 24 main job ids>`) runs the read-out above with
`--json runs/pt2s505/selection.json` from the batch's code snapshot. Pre-registered trigger (pt2main's own practice:
"confirm seeds follow once their open brackets close"; rule 1's verdict is NOT FINAL while a bracket is open):
- if BOTH arms have a best lr (complete, not WITHHELD) and NEITHER bracket is open -> it writes
  runs/pt2s505/picks_confirm.json (status "final") and submits seeds 4-7 of none@best and win-ctx@best
  (`make_jobs_pt2s505.py --stage confirm --picks ... --test-only`, then `--submit`): 8 jobs;
- otherwise nothing is submitted, and runs/pt2s505/select.log names the reason (incomplete, WITHHELD, or the arm whose
  bracket is OPEN at which edge): the next lr beyond an open edge (x2 or x0.5, seeds 0-3) is pt2main's conditional cell
  and needs the user's approval.
The confirm seeds then enter rule 2 (best-vs-best with confirm). Seeds 4-7 are selection-free (the lrs were chosen on
seeds 0-3).

## Checks inside every job (exit 3 before training, exit 4 after)
1. HOLD; the sha256[:12] of the snapshot's trainer and imports; COMMIT names the trainer; an A100 guard; the data
   sha256.
2. `--dry_inputs` (the job's own ARGS): the model line `model: 505.4M parameters; gradient checkpointing on`; 3 printed rows
   with exactly 2 operand sites; the target table (helix3: (1000, 10); none: absent); the sha256 of the 9 DRY rows = the
   Torch CPU 505M dry run of the seed = the Burst pt2s190 seed's rows with this model line (seeds 0-3: tasks 358-361;
   seeds 4-7: 382-385).
3. Training, offline_lm.py, then make_pt2main_tasks.CHECK_PY verbatim with the 505M intent (every trainer argument;
   code_sha 93e22bf4c4c0; steps, n_params, gradient checkpointing, layers, lm_chunk 0, eval spacing; 75 evals; the window
   events and prior use; step-0 lm within 1e-3 of the 505M CPU reference; offline double path) -> check.json.

## Code
Trainer = origin/main's (93e22bf4c4c0), whose default path was regression-tested against 64abc19's a3bce3ab63c1 (the
pt2main trainer) on tiny none / win-ctx / win-ctx --grad_ckpt on configs (scale/reg.py; results below), on top of the
recorded chain a3bce3ab63c1 -> 3c99f6f8cb17 (runs/ptheld) -> 93e22bf4c4c0 (runs/ptheld2). Job code = runs/pt2s505/code
(git archive of tools/ at the committed HEAD of branch scale-ladder). Generator llmadd_torch/make_jobs_pt2s505.py.

## Jobs
24 + 8 jobs, account <slurm-account>, a100_tandon, gpu48, `--nice=20`, 1 GPU, 8 CPUs, 64G,
`--time=11:00:00` (2 x 5.3 h, the A100 estimate from L40S 6.4 h, rounded up), `--requeue`. Planned ~32 x 5.3 = 170 A100-h.

## Tests before submission (CPU; scripts / logs: the session scratchpad's scale/ and llmadd_torch/scale_work/)
| test | result |
|---|---|
| CPU regression (reg.py; tiny d64 x 3, seq 64, 40 steps, one thread, old and new side by side on one host) | origin/main trainer vs 64abc19's a3bce3ab63c1: pt2-none (lr 4.8e-3): IDENTICAL (run JSON every value except code_sha / args.out / per-eval sec; models 36/36 bitwise; dry and run stdout equal); pt2-winctx (lambda 5, lr 6e-4) and pt2-winctx-ckpt (--grad_ckpt on): IDENTICAL the same way (36/36 tensors bitwise; prior on at 1 eval) |
| CPU references (refs_pt2.py, the job's own ARGS) | --dry_inputs of none and win-ctx at seeds 0-7: per seed the 9 DRY rows identical across the arms and = the Burst pt2s190 seed's rows (tasks 358-361, 382-385) with the 505M model line; 3 rows with 2 operand sites; target line only for win-ctx. Step-0 lm (step0_ng.py) seeds 0-7: 11.721513 / 11.722209 / 11.718904 / 11.727621 / 11.739694 / 11.731402 / 11.727548 / 11.722637 (win-ctx s0 = none s0 bitwise); the CPU method calibrated at 505M against a GPU value: -5.4e-5 (runs/rb6_505m/AUDIT.md) |
| jobcheck_pt2.py (the 24 main jobs) | JOBCHECK PT2 PASS (496): make_jobs_pt2s505.intent with the 190M facts and a3bce3ab63c1 reproduces the intent embedded in 16 Burst pt2s190 tasks exactly (none / win-ctx, seeds 0-7); per job bash -n, CHECK_PY = the Burst pt2s190 task's byte for byte, the intent differs only in code_sha / n_params / layers / lm0_ref and args dims / lr / tag, the input-check block = the Burst task's except model / DRY / pin / ARGS lines; on the 16 Torch CPU 505M dry logs 24 own cases exit 0, 360 cross cases exit 3 with the right reason |
| e2e_pt2.py (the rendered job run end to end with bash on CPU; tiny overrides + --grad_ckpt on; fake nvidia-smi) | none@9.6e-3 s1 and win-ctx@6e-4 s2: job rc 0, every CHECK_PY check PASS (none 7, win-ctx 10; the tiny step-0 lm = its CPU reference) |
| pt2gen_test.py (pt2main_summary --gen on the 24 runs relabelled from Burst pt2s190 + pick_confirm.py) | PT2GEN TEST PASS (7): closed brackets -> best none 4.8e-3 / win-ctx 6e-4 (pt2s190's own picks), the pt999_summary double path holds, no integrity line -> FINAL; an open lower bracket of win-ctx -> NOT FINAL; a failed check.json -> none WITHHELD -> NOT FINAL; a missing run -> incomplete -> NOT FINAL. pt2main_summary without the new flags: byte-identical output on the Burst results (143 lines) |
| select_job.sbatch plumbing (fake R dirs, the generator calls replaced by echo; a stale picks file planted) | closed: final -> the confirm stage's --test-only and --submit calls; open: NOT FINAL, no call, the stale picks file removed first |
| confirm-stage rendering with fake picks (--print only; the picks file and the 8 job files deleted again) | 8 jobs: none@pick and win-ctx@pick x seeds 4-7 |
| sbatch --test-only | 24/24 accepted |

## Submitted (2026-10-22 ~05:10 EDT)
24 jobs 20602377-20602426 (SUBMITTED_<name>); code 7f1d06b (runs/pt2s505/code/COMMIT), trainer 93e22bf4c4c0.
Selection + automatic confirm stage: job 20602448 (select_job.sbatch, cpu_short, --dependency=afterany on the 24;
SUBMITTED_select). Outputs: runs/pt2s505/selection.json, picks_confirm.json (only when final), select.log.
Kill switch: touch runs/pt2s505/HOLD.
