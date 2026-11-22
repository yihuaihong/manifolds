

## Amendment V7 (post-hoc, 2026-11-22): all6@2e-3 with the torus / box5 lines removed after the prior window

**Status.** Post-hoc and exploratory. The user approved it on 2026-11-22 (「可以的」, relayed by the main session). There is no verdict rule and no gate. Nothing in the registered analysis (readout_pt6v2.py, CELLS / ALLOC / COMPOSITION / PICK.json, the V3 / V4 / V5 outputs) changes, and readout_pt6v2.py does not read `out_v7/`.

**Why.**
- At lr 2e-3, all6 beats only-dim on strict dim by +25 pts at the end of training (8/8 seeds).
- The gain appears after the window, and it needs the torus / box5 priors: all4 is about equal to only-dim (F106).
- The V5 probes (F107) show that only-dim still encodes the answer at the answer position but fails to output it.

**Hypothesis.** Torus / box5 share a lattice answer code with dim. Post-window LM training on their lines keeps dim's state→token readout working.

**Prediction (this is the hypothesis, not a rule).** Without torus / box5 lines after the window, dim's end-of-training advantage over only-dim largely disappears.

**Cell.** `all6drop@2e-3`, seeds 16..23, 8 runs, one job per run.
- The run is the registered all6@2e-3 job of each seed. Its ARGS are byte-identical apart from the out dir, plus `--drop_domains_after torus,box5@7324`.
- Step 7324 is the first step after the prior window (prior_on(s) = 916 ≤ s < 7324).
- From that step, the torus and box5 lines are still drawn: every RNG stream consumes exactly what it did before. They are then left out of the row, and the row's text chunk extends to the left, to the same chunk end, to fill their space.
- Tokens per step, the numbers line, and the dim / cities / countries lines (content, rate and order) are unchanged. Rows before step 7324 are byte-identical to the registered run.
- The torus / box5 held-out evals still run at every eval. geo_torus / geo_box5 have no sites after the drop.

**Comparison.** The registered all6@2e-3 and only-dim@2e-3 runs of the same seeds (`runs/pt6v2/out/`). No new control arm.

**Trainer.** `runs/pt6v2/out_v7/code/tools/pretrain_range.py` 8da22c6ea208 = a398fccbfe93 + `--drop_domains_after`.
- Patch: `out_v7/patch_trainer_v7.py`. Diff: `out_v7/trainer_v7.diff`. The other files in `out_v7/code/` are byte-identical.
- At its default the flag is a no-op and is left out of the run JSON's args.
- When on, it is printed, asserted at step 0, logged per eval (`dropped_lines`), and recorded in the run JSON (`drop_domains_after`).

**Checks before submission (CPU smoke, `out_v7/smoke/`, job 24063503: 33/33 PASS, `compare2.txt`).**
- Flag off: run JSON and stdout equal the registered trainer's, except code_sha, out and timing.
- Flag on, real s16 config: batches 0..7323 are byte-identical to flag off, and the four RNG states after step 7329 are identical.
- Steps 7324..7329: no torus / box5 line, checked three ways (segments, sites, and a token-sequence search). Kept lines are identical and in the same order, the tokens per step are equal, and the dropped count equals the cycling rule's count.
- dry_inputs gives the 63-line reference sha b71458d3a12d. Resume preserves the drop.
- Negative control: check_v7.py fails on a flag-off run.
- In `compare.txt`, the only fail was an un-normalised `done in Ns` timing line. The normaliser was fixed and the comparison re-run, giving `compare2.txt`.

**Job-level checks.**
- `check_v7.py` (cf44ea38c274) runs before the registered CHECK_PY and exits 5 on a fail. Against the registered run of the same seed, it checks:
  - data_sha is equal before 7324 and differs after it;
  - the token share of the kept types is equal at every eval;
  - dropped_lines matches the rule;
  - dom_sites, the slot share, and geo_<dropped> is absent.
- The registered CHECK_PY gets two counted edits:
  - the slot rule is read at the last eval before the drop;
  - geo_<dropped> is counted over the evals before the drop.
- A step-0 lm difference from the registered run is flag-only (device numerics).

**Device.** A100 / H100 (a100_tandon, h100_tandon). The registered seeds 16..22 ran on B200 and seed 23 on A100. The device difference is a flag in the read, not a gate.

**Planned descriptive read (`out_v7/v7read.py`, no verdict rule).**
1. Dim strict final3 and area of all6drop vs all6 and vs only-dim, paired by seed, with mean and sign counts.
2. Dim's drop from the window's close (the eval at 7250) to the end, per run and per arm.
3. Torus / box5 held-out accuracy after the drop (all6drop vs all6): this shows the dropped domains decaying.
4. numbers / cities / countries final3, as a damage check.
5. The device flag per run.

**Canary.** `llmadd_pt6v2v7-canary-s16`, 800 steps, window 40..319, drop at 320.
- check_v7 is fatal. The registered CHECK_PY is non-fatal, because a short run fails its length checks by design.
- dry_inputs uses the production ARGS.
