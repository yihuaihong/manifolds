# pt6v2 phase 2b notes (2026-11-19, exp-builder; the coordinator merges into AUDIT.md; AUDIT.md itself untouched)

## Coordinator decisions applied (2026-11-19)
1 nice 20, 3 TIMEOUT restarts from scratch, 4 canary on L40S, 5 ckpt 500, 7 toy3 sha checks in every job, 8 A100 account /
partition as pt2sz: already in the phase-1 rendering, kept.
6 A100 guard: make_jobs_pt6v2.py GPU_CASE a100 `*A100*80GB*` -> `*A100*` (40GB or 80GB; 53M fits, same architecture);
CHECK_PY's in-job GPU-name check uses "A100" (substring), so it accepts both. Memory note: the pt6comb B200 smoke's
nvidia-smi peak was 32.6 GiB (caching allocator on a 180 GiB card; not the requirement); the A100 smoke's gpu_mem_max_MiB tells
whether a 40GB card is tight. If a 40GB A100 OOMs, the job dies at the first steps (cheap) and its seed is restarted on 80GB.
2 time limits: the smokes will not have finished when this phase returns, so the rendered limits stay at the estimates (L40S
4 h, A100 3 h). When the smokes finish: `tests/smoke_check_pt6v2.py <dev> <log>` prints the RATE line; set the limit to 2.5 x
the full-run estimate. The naive estimate (median s/step at eval-every-50 x 18310) is an upper bound: on B200 the ptdom pilot
(400 steps, eval 50) gives naive 1.12 h vs 0.31 h measured for the pt6comb B200 full runs (factor ~0.28; different designs,
indicative only). Suggested: limit = 2.5 x (naive x 0.3), not below 2 h.

## refs (job 23687462; fix + post job 23688586)
Build phase rc 0 for every dry / step-0 row; `TABLE placeholder PASS` (pt6v2_refs_dim-cities-countries-compass-hcube.json,
bd40e65690e9). `domrefs` crashed: KeyError 'languages' (a source mix holds a non-candidate domain, ptdom-B2's languages).
Fix tests/refs_pt6v2.py:225 `V.SOURCE[d]` -> `V.SOURCE.get(d)`. Re-run (CPU, 23688586): `DOMREFS PASS (18 domains)`
(pt6v2_domrefs.json 7727cd4f4ece), `SPLIT-INDEPENDENCE placeholder PASS (65 domain x cell comparisons)`, table re-derivation.

## Test 3: tiny CPU training (tests/tiny_pt6v2.sh, tiny_args_pt6v2.py, tiny_check_pt6v2.py; job 23689529; the first try 23688409 was cancelled at 15 min: too slow for its 30-min limit at eval every 10)
PLACEHOLDER composition (numbers, dim, cities, countries, compass, hcube), the jobs' own command line (args_raw, seed 16, lr 2e-3)
+ d64 x 3 layers, seq 128, bs 16, 200 steps, eval every 20, window (0.05, 0.4) = steps 10-79 inside the run; arms all6, none,
only-compass, only-hcube, Rev H trainer in runs/pt6v2/code with the job env (PYTHONPATH llmadd_torch/pylib). Checks T3.1 rc 0 +
'done in' + step-0 toy3 line (trainer asserts toy3 gnorm_true > 0); T3.2 data_sha equal at every eval, step-0 lm equal (1e-4);
T3.3 each active prior on exactly in the window, others off; T3.4 gradient / LM-share fields for each active prior at every
on-window eval; T3.5 no prior field for an inactive prior. Result: see HANDOFF_phase2b.md.

## GPU smokes (written before submission; not part of any result)
Rendering: `make_jobs_pt6v2.py --smoke l40s,a100` = the canary's job (all6 @ 2e-3, seed 16, 400 steps, eval 50, ckpt 100,
PLACEHOLDER composition, window events [[20,1],[160,0]]) on the given device's header and guard, out runs/pt6v2/pilot/smoke_<dev>,
job name llmadd_pt6v2-smoke-<dev>, kill switch HOLD / HOLD_smoke, no read-out. Purpose: the device guard and in-job checks on
L40S and A100, and the step rate for the time limits.
Pass (tests/smoke_check_pt6v2.py <dev> <log>, all four): S1 `DEVICE <dev>` in the log; S2 the step-0 eval JSON line and the
step-0 toy3 gradient line; S3 the trainer's final `done in` line; S4 `CHECKS PASS` (or the A5.1 flag-only FAIL: GPU-numerics
tolerance / CKA-rise checks only; at 400 steps the CKA-rise bar may legitimately flag). Recorded, not gated: s/step, naive
full-run h, GPU memory max, util median.
Fail -> do not submit that device's seeds; a guard exit (`## not an A100` etc.) means the partition gave another card: fix the
header / constraint, not the guard.

## submit_pt6v2.sh
Wraps `make_jobs_pt6v2.py --submit`: dry by default (checks only), `--go` submits. Refuses (via make_jobs) without
COMPOSITION.json, refs, readout_pt6v2.py, or with HOLD / HOLD_<dev> / HOLD_canary; also /scratch quota >= 97%, and job files not
written (`--check --write` by the coordinator first; --submit now asserts every unsubmitted job file equals the rendering).
Prints the alloc and run count per device; canary first; B200 via the singleton slot chains.
