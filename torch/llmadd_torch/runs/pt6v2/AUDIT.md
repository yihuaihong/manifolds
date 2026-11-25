# pt6v2 — intent spec and pre-registration (confirmatory tier; written 2026-11-19 before any pt6v2 result)

The second six-manifold combination: manifold priors as an auxiliary CKA loss in 53M pretraining. Goal: a "perfect six".
pt6comb (composition B) gave 5 of 6 HELPS with interference not detected. s4 came out NOT SHOWN and people sat at FLOOR.
The design is the user's decision of 2026-11-18: single stage, mixed GPU types.

## Design
- **Composition.** numbers + 5 domains from ALLOWED, in the trainer's order (make_jobs_pt6v2.py ALLOWED):
  - s4, dim, cities, countries, people, months;
  - Part R: stars, landmarks, compass, prefixes, ranks, scalar_adj;
  - toy3: cyc12, torus, hcube, d6, box5, caesar.
- **Placeholder** until the pick around 11-20: dim, cities, countries, compass, hcube. The pick is fixed once in COMPOSITION.json (`--set_comp`).
- **Body.** Everything else is pt6comb composition B's C body on Rev H, i.e. llmadd-burst-ptdom3 ef7b8854; the snapshot runs/pt6v2/code equals runs/ptdom3/code byte for byte.
  - 53M, 3e8 tokens; numbers line at 2% and first; K=2 lines cycling over the 5 domains;
  - window (0.05, 0.4), site ctx, lam_rule each, lambda 1, numbers prior helix3.
- **Cells.** 12 per seed, seeds 16-23 (96 runs), no lr selection:
  - all6@1e-3 (P) and all6@2e-3;
  - none@{2e-3, 8e-3, 1.6e-2, 3.2e-2};
  - only-m@2e-3 for each of the six m.
- **Exposure.** 234,368 domain lines per run. That is 1.00× pt6comb B's and 1.20× the 6-domain ptdom / ptdom2 / ptdom3 sources. Descriptive: `--plan` flags anything outside 0.67-1.5×.

## Verdict rule, per manifold m, on m's PRIMARY area
**HELPS**
- N_m = the none lr whose 8-seed mean of m's primary area is highest; a tie goes to the higher lr.
- g_s = all6@1e-3 − none@N_m.
- HELPS iff the one-sided paired t has p < 0.05 (df 7) AND g_s > 0 in ≥ 6 of 8 seeds.
- Picking N_m on the same seeds favours none, so the rule is conservative. The read-out prints this.
- Flags (never conditions):
  - "none at its top lr 3.2e-2" when N_m = 3.2e-2;
  - "prior at its lowest lr" (always).
- Descriptive: g against each of the 4 none lrs, and against the mean6-picked none (F102's rule).

**Interference** (pt6comb A2, unchanged)
- h = all6@2e-3 − only-m@2e-3, u = only-m@2e-3 − none@2e-3.
- "Interference not detected" iff (mean u > 0 and mean h ≥ −0.5·mean u) OR the one-sided t p (H1: mean h < 0) ≥ 0.05.
- When u ≤ 0, only the t clause applies.
- A3: "AUX GATE FAILED at all6@2e-3". An incomplete pair is reported as "WITHHELD".

**Headline and gates**
- Headline: "k of 6 HELPS with interference not detected (at the same lr 2e-3)". Holm is a reference column.
- GATE (A2.1) is checked per half, seeds 16-19 and 20-23.
- Refusal and `--allow_partial` work as in pt6comb.

**Primary read-outs** (each from its source read-out)
- pt6comb's for numbers, s4 (s4_hard), dim (the strict set), cities, countries, people. months as in ptdom.
- readout_ptdom2.py for the Part R domains.
- readout_ptdom3.py for toy3:
  - the twin-free set is PRIMARY for hcube and d6; the full set for cyc12, torus, box5; d6 keeps its AAE flag;
  - caesar is read out separately.
- FLOOR / CEILING tags.

## GPU types (per seed; all 12 cells of a seed on one type)
- Allocation: `--alloc b200:16,17,18 l40s:19,20,21 a100:22,23`.
- Each job asserts at start that it is on its seed's type (exit 3 otherwise) and prints `DEVICE <type>`. intent / check.json carry `"device"`.
- The read-out prints the device per seed plus a "mixed devices" note. That note is never a condition: the user said 「不同卡型可以，误差很小」 ("different card types are fine; the error is small").
- Plan:

| device | per run | time limit | measured or estimated |
|---|---|---|---|
| B200 | 0.45 h | 3 h | measured, 1,091-1,120 s training (tests/walltime_scan.out) |
| L40S | 1.35 h | 4 h | estimate, ≈3.0× margin |
| A100 | 1.05 h | 3 h | estimate, ≈2.9× margin |

- Totals: 16.2 B200-h + 48.6 L40S-h + 25.2 A100-h = 90 GPU-h. Disk ≤ 0.2 GB.
- Resume:
  - a preempted or requeued job re-runs the input checks and resumes from its ckpt (every 500 steps, on every device);
  - a TIMEOUT is not requeued. `--rerun` starts from scratch as -r<k>.

## Pipeline canary (pass rules fixed now)
- `pt6v2-canary-all6-lr2e-3-s16`: 400 steps, out runs/pt6v2/pilot/, default L40S. It is submitted first, together with the batch.
- It passes iff all three hold:
  1. the log has `DEVICE`, `step 0` and the trainer's final line, and the in-job CHECK_PY prints PASS;
  2. `canary_readout.txt` gives a number for each of the 6 manifolds;
  3. no check that blocks a read-out fails for a reason tied to the hypothesis.
- If 1 or 2 fails, the coordinator cancels the batch.

## Tests (runs/pt6v2/tests/)

| test | what | status |
|---|---|---|
| 1 | regress_pt6comb.py / .log: 80 pt6comb C job files (comp B, seeds 8-15, B200), re-rendered | PASS. Every residual line falls in a registered category: Rev H shas / COMMIT + the toy3 dep checks, guard + DEVICE, intent batch / code_sha / device, comments, HOLD names |
| 2 | split independence: `refs_pt6v2.py cmp placeholder`, every seed 16-23 vs each domain's source mix (pt6comb B; ptdom B2 for compass; ptdom3 T3 for hcube) | in CPU job 23687462 |
| 3 | tiny CPU training run | NOT DONE (phase 2) |
| 4 | read-out self-tests | NOT DONE (phase 2) |
| 5 | known answers | NOT DONE (phase 2) |
| 6 | `--plan` | done, numbers above |

- `--check` renders all 97 texts (bash -n) for the placeholder and for comp B, with fake refs.

## Pending decisions (待协调者定)
1. nice 20 on every device (pt2sz used 1000).
2. The L40S / A100 time limits are estimates; no ptdom-style 53M run exists on either.
3. A TIMEOUT restarts from scratch.
4. The canary runs on L40S (`--canary_dev`).
5. ckpt 500 on every device.
6. The A100 guard requires 80GB.
7. The toy3.py / toy3_nonce.json sha checks sit in every job (Rev H snapshot), including compositions without toy3.
8. The A100 header uses account 520 / a100_tandon, as in pt2sz.
9. readout_pt6v2.py and submit_pt6v2.sh are not yet written. The canary renders a "READ-OUT NOT RENDERED" line until readout_pt6v2.py exists, and `--submit` refuses until then.

## Coordinator (2026-11-19 ~00:40 EST): decisions on the pending items; phase 2 delegated
- Pending 1-5, 7, 8 accepted as built. Pending 6 changed: the A100 guard accepts any A100 (40GB or 80GB); 53M fits, same architecture.
- Pending 2: two GPU smokes (L40S, A100; 400 steps, placeholder, not part of any result) measure the step rate; time limits become 2.5x the measured full-run estimate when available.
- Phase 2 runs as two builders in parallel: 2a = readout_pt6v2.py + tests 4 / 5 (notes in AUDIT_phase2a.md); 2b = submit_pt6v2.sh, test 3, the refs job 23687462 check, the smokes (notes in AUDIT_phase2b.md). The coordinator re-renders (`--write`) after both, then one fresh code audit (exp-auditor) of the whole build.

## Phase 2 merged (coordinator, 2026-11-19 ~01:05 EST)
- 2a: readout_pt6v2.py e4beaf64417a; self-test 25/25, mutants 8/8 killed (tests/selftest_pt6v2_20261119T0053.log); known answers 14/14 (tests/known_pt6v2_20261119T0054.log: pt6comb readout.txt reproduced, 455 ptdom2 pairs, 21 ptdom3 numbers, canary path). Notes: AUDIT_phase2a.md c9a36dd063bb.
- 2b: make_jobs_pt6v2.py 961ef81fe230, submit_pt6v2.sh 23ff3c308576, tests/refs_pt6v2.py d2ec97b41a71 (refs rerun 23688586: DOMREFS PASS 18, SPLIT-INDEPENDENCE PASS 65, TABLE PASS); test 3 = CPU job 23689529, smokes 23689162 (L40S) / 23689163 (A100), pending. Notes: AUDIT_phase2b.md f2e4c7f47b73.
- Decisions on 2a's pending items: 1 caesar counts in k (it is one of the six when chosen; its block stays separate); 2-3, 5 accepted as built (tags / flags only, strict refusal as pt6comb); 4 make_jobs_pt6v2.py is frozen from 961ef81fe230 on; any later change is listed here and readout_pt6v2.py's SRC sha is updated in the same step.
- GPU allocation at submit: `sbatch --test-only` puts L40S starts at ~11-24 (1,732 pending), A100 the same day. Unless that changes, the batch runs on B200 + A100 only and the canary on A100 (`--canary_dev a100`); decided at submit.
- Next: one fresh code audit (exp-auditor) of the whole build; then pick time.
- A100 smoke 23689163 (coordinator, 2026-11-19 ~01:15): SMOKE a100 PASS 4/4 (pilot/smoke_check_a100.out); 0.236 s/step at eval-every-50, naive full run 1.20 h (upper bound), limit 3 h kept; GPU mem max 32.3 GB (fits a 40GB A100). L40S smoke still pending.
- Test 3 (CPU tiny, 23689529; coordinator 2026-11-19 ~01:35): 18/19. The one FAIL is the test's own cadence (6 evals, only 1 inside the window, the check wanted >= 3), not the code: at that eval every arm had exactly its priors on and the rest off, gradient / LM-share fields present, no field for an inactive prior. The same on-window behaviour at 3 on-window evals is covered by the A100 smoke on the real 53M body (CHECKS PASS). Not re-run. (grad_lm_share 0.12-0.28 is the d64 toy model; pt6comb's 53M all6 had 0.76.)

## Code audit (exp-auditor, fresh, 2026-11-19; tests/audit_code_pt6v2.md): no blocker
## Amendment V1 (coordinator, 2026-11-19 ~02:10 EST; before any pt6v2 result, nothing of pt6v2 submitted except the two smokes)
1. **none grid += 4e-3** (audit important 1, checked by the coordinator on runs/pt6comb/picks_after_L1y.json: pt6comb's none is best at 4e-3 for cities 86.8 / countries 90.2 vs 2e-3 86.0 / 89.6; without 4e-3 N_m would be forced to 2e-3 and g overstated by ~0.7). NONE_LRS = 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2; 13 cells x 8 seeds = 104 runs + the canary. N_m rule unchanged otherwise. New flag (label only): "none at its lowest lr 2e-3" when N_m = 2e-3.
2. P stays fixed at 1e-3 (registered; picking P on the same seeds would bias toward HELPS). Where a manifold is NOT SHOWN, its line prints the same-lr gain all6@2e-3 - none@2e-3 and u first (audit important 2: a domain that prefers a higher lr can read NOT SHOWN at P = 1e-3, cf. pt6comb s4). At pick time, prefer candidates whose gain holds at 1e-3-2e-3.
3. GATE preview (audit important 3): after the first full-length all6@1e-3 run, read its LM share with tmp_audit/pt6v2_code/grad_dp.py; flag only.
4. The report prints the two qualifier lines (audit important 4): N_m chosen on the same seeds favours none; pt6comb readout.txt's line (Tier L: the comparison / toy targets carry the answers; NOT SHOWN != no effect).
5. Minor: readout_pt6v2.py asserts the sha of make_jobs_pt6v2.py (frozen after V1).

## V1 build (exp-builder, 2026-11-19 ~02:00 EST; nothing written with --write, nothing submitted, no job touched)
- Files (sha256[:12]):
  - make_jobs_pt6v2.py 961ef81fe230 -> **2f10290f95b2** (frozen from here on). NONE_LRS += 4e-3; counts 12 -> 13 cells, 96 -> 104 runs, 97 -> 105 jobs (docstring, cells() assert, plan print, `len(jobs) == 105`). No job-body change.
  - runs/pt6v2/readout_pt6v2.py e4beaf64417a -> **f4bbc4de583f**. Changes:
    - the NONE_LRS assert and the docstring;
    - BOT_FLAG "none at its lowest lr 2e-3" (label only) when N_m = 2e-3;
    - a NOT SHOWN line starts with `[NOT SHOWN at P: same-lr gain all6@2e-3 - none@2e-3 <g> (wins/n, p), u <u>]`;
    - the two qualifier lines before HEADLINE: Q_SAME_SEEDS (wording ours) and Q_READING (= runs/pt6comb/readout.txt's reading line, verbatim; asserted in test 5);
    - asserts make_jobs_pt6v2.py = 2f10290f95b2 and that MJ2 is imported from that file, next to the SRC asserts.
  - runs/pt6v2/submit_pt6v2.sh 23ff3c308576 -> **3f07090c34da** (header comment only; bash -n ok).
  - New test files:
    - tests/v1_body_diff.py e325e5faac3f, with tests/make_jobs_pt6v2.preV1_961ef81fe230 (the pre-V1 copy, 444).
    - Pre-edit copies of the three files are in this session's scratchpad.
- Tests (new logs in tests/, 20261119T06xxZ; the old logs are kept):
  - **Test 1** regress_pt6comb_…0653Z.log: PASS. 80/80 files, unregistered 0; categories identical to before: comments 720, trainer 880, hold 160, guard 240.
  - **V1 body diff** v1_body_diff_…0652Z.log: PASS.
    - Placeholder with real refs, alloc b200:16-19 a100:20-23, canary a100.
    - All 97 old names re-rendered. 52 are identical. The other 45 B200 jobs differ only in the slot job-name and its "singleton slot N" comment (90 + 90 lines): the slot cycle shifts with 13 cells per seed.
    - The only new names are pt6v2-none-lr4e-3-s16..23. Their trainer args = none@8e-3's except lr / out dir / run name (checked for s16 and s20).
  - **--check**: 105 texts pass bash -n (rc 0) in each of three logs:
    - check_placeholder_refs_…0653Z.log: real refs, with --print; 52 b200 + 53 a100;
    - check_placeholder_fake_…0653Z.log;
    - check_compB_fake_…0653Z.log.
  - **--plan** plan_…0653Z.log, `--alloc b200:16,17,18,19 a100:20,21,22,23`:
    - 13 cells x 8 = 104 + canary;
    - B200 4 x 13 = 52 runs = 23.4 B200-h; A100 52 runs x 1.05 h = 54.6 A100-h; total 78.0 GPU-h. With the A100 smoke's 1.20 h upper bound: 62.4 A100-h.
    - Exposure unchanged (234,368 lines; compass / hcube 1.20x).
  - **Test 4** selftest_pt6v2_…0655Z.log: self-test 30/30 (25 old + 5 V1); mutants 10/10 killed (8 old + 2 V1). The V1 cases:
    - N_m can be 4e-3;
    - the low-edge flag iff N_m = 2e-3;
    - a NOT SHOWN line starts with the same-lr gain and u, and HELPS lines do not;
    - the incomplete-pair variant;
    - the qualifier lines before HEADLINE.
    - The 2 new mutants: the low-edge flag moved to the top lr; the same-lr-first prefix moved to HELPS lines.
  - **Test 5** known_pt6v2_…0655Z.log: 15/15 (14 old + Q_READING verbatim).
- Notes for the coordinator:
  - Q_READING names only "comparisons and s4 / dim" as Tier L (pt6comb verbatim); toy3 domains such as hcube are not named. Whether to extend it at pick time is undecided.
  - The job files still have to be rendered and written (`--check --write`). The canary text embeds the readout sha f4bbc4de583f.
  - The pending L40S smoke 23689162 does not depend on either changed file.
  - Lines 16 / 52 ("12 per seed", "96 runs", "12 cells") and line 88 ("97 texts") of the registered body describe the pre-V1 design; Amendment V1 supersedes them (left unedited: registered text).
- Coordinator on the V1 build's pending items (2026-11-19 ~02:30): (a) the copied Tier L reading line names only s4 / dim; at pick time, if a toy3 or ptdom2 domain is chosen, the line is extended to name every chosen domain whose target carries the answers (a printed-text change, made together with set_comp, logged here); (b) the counts in the registered text above (lines 16 / 52 / 88) stay as written; V1 supersedes them. Build complete; waiting for the candidate read-outs (ptdom2 R2, ptdom3) to pick the composition.

- Coordinator 2026-11-19 (L40S smoke 23689162, gl063): SMOKE l40s PASS 4/4 (pilot/smoke_check_l40s.out); 0.375 s/step -> full run naive 1.91 h, 2.5x = 4.77 h vs the 4 h L40S limit in HW6 (2.1x naive; naive is an upper bound) -- flag only; L40S not in the planned alloc (queue ~5 days), raise the limit in place if it is used.

## Composition pick rule (coordinator, registered 2026-11-19 ~05:55 EST; the user 11-19: 「你可以自动执行完这个完整过程」 = pick + submit without a further OK)
Seen before writing this (stated so the rule's independence is clear): Part R2 seeds 8-14 partial means (tests of runs/ptdom2/tests/partR_partial_20261119T0524.*: vs the picked none, area: compass +1.5 3/7, prefixes -0.4, ranks +2.1 6/7, scalar_adj +1.8 5/7, landmarks -1.2 3/8) and the ptdom3 seed-0 screen of 8 of 22 runs (torus only@1e-3 82.9 vs none@1.6e-2 74.4; cyc12 FLOOR). Not seen: R2 seed 15, ptdom3 hcube / d6 / box5 / caesar prior runs, any replication.
- Candidates: Part R domains compass, prefixes, ranks, scalar_adj, landmarks (stars is NOT FINAL: out); toy3 cyc12, torus, hcube, d6, box5, caesar. S4 out (F102 NOT SHOWN; the user: swap it). Fallbacks in order: people (F102 HELPS 8/8, FLOOR), months (F101 PASS, ends below chance).
- Contrast per candidate c, paired by seed on c's PRIMARY area (Part R: ptdom2's area; toy3: readout_ptdom3's primary = twin-free for hcube / d6, full set otherwise): g = only-c@p - none@n, where p = the prior lr in {1e-3, 2e-3} with the highest mean (pt6v2 runs all6 at 1e-3, only-m at 2e-3) and n = the none lr with the highest mean among every none lr present at the evidence seeds. Evidence seeds: Part R 8-15 (whatever exists at pick time), toy3 0-2 (screen + replication).
- Eligible iff all of: E1 no FLOOR / CEILING tag (the source read-out's definitions) on p vs n; E2 >= 3 paired seeds; E3 n is not the highest none lr present unless that lr is >= 3.2e-2 (pt6v2's top); E4 mean g > 0 and >= 2/3 of the seeds positive; E5 d = mean g / sd(g, ddof 1) >= 0.5 (d 0.5 ~ 40% power for HELPS at n 8; lower is a likely NOT SHOWN).
- Pick: the two eligible candidates with the highest d (tie: larger mean g); fewer than two eligible -> fill from the fallbacks in order. The composition is numbers + dim + cities + countries + the two picks.
- toy3 replication (B200, approved for ptdom3; the standing <= 2 replication seeds): every toy3 domain whose seed-0 prior pick at 1e-3 / 2e-3 beats its seed-0 best none and is not FLOOR gets only-d@p seeds 1, 2; shared none cells (every none run reads every toy3 domain): none@1.6e-2 seeds 1, 2 and none@3.2e-2 seeds 0, 1, 2 (+ a lower none lr at seeds 1, 2 if it is some qualifying domain's seed-0 best).
- Conditional Amendment V2 (fixed now): if a picked candidate's best none over its evidence seeds is 3.2e-2, pt6v2 adds none@6.4e-2 (8 runs) before submission.
- Pick time: Part R2 complete (seed 15 in, or its runs failed twice) and every toy3 replication run finished. The pick script prints every candidate's g, wins, d, n, p, E1-E5 and the reason for each in / out; its output is logged here before submission.
- Allocation (resource, not a number-changing field): B200-heavy, `--alloc b200:16,17,18,19,20,21,22 a100:23` (B200 0.45 h per run at our 3-GPU cap ~6.7 runs/h; A100 ~1 completion/h in practice). H100 / H200 not used: their per-user share is taken by the cia window's jobs.


## V2 build (2026-11-19, exp-builder; build and test only, no training job submitted)
Implements the conditional Amendment V2 (opt-in none@6.4e-2) and the one-command pick → submit chain. No verdict rule changed.

### New sha256[:12] values
The pre-V2 copies are in tests/*.preV2_<old sha>, and the exact patch is tests/patch_v2.py.

| file | old → new | what changed |
|---|---|---|
| make_jobs_pt6v2.py | 2f10290f95b2 → **bb15d248afd8** | `--extra_none 6.4e-2` and CELLS.json (details below) |
| readout_pt6v2.py | f4bbc4de583f → **dbd3654b6f28** | none grid from CELLS.json; top-lr flag from the actual top; Tier L reading line; MJ2 sha → bb15d248afd8 |
| submit_pt6v2.sh | 3f07090c34da → **a1a06ded3a9d** | `--extra_none` passed through to --check and --submit; a new refusal unless `cells manifest: same`. Every earlier refusal is kept. |
| tests/run_refs_pt6v2.sh | c82e56ddfd7d → 355f0067bf62 | new `placeholder` mode (the comp job list run on the placeholder) for the poll-and-grep test; build and comp unchanged |
| auto_pt6v2.sh | new, 95b1135cc607 | the chain |
| tests/auto_read_pick.py | new, 92726049c7f2 | validates PICK.json |
| tests/v2_body_diff.py | new, 23d65ba329bc | the byte-identity proof |

### make_jobs_pt6v2.py: `--extra_none 6.4e-2`
- The only value accepted is 6.4e-2. It is asserted to be in MJ.LADDER and above the grid's top.
- `cells(doms, extra)` puts none@6.4e-2 right after the none grid. That gives 14 cells per seed and 113 jobs (14 × 8 + the canary). `--plan` counts follow.
- The trainer takes it: `--lr` is an argparse float, used only for the AdamW lr and the schedule. render_v2 asserts `lr in LADDER`.

### CELLS.json (the cell-set manifest)
- `--write` writes runs/pt6v2/CELLS.json, chmod 444. It holds design, domains_arg, none_lrs, extra_none, cells, n_runs, seeds and canary.
- It can be rewritten until a SUBMITTED_pt6v2-* file exists. After that, a different cell set is refused.
- `--disk` prints `cells manifest: same|missing|differ`. `--submit` asserts `same`.

### readout_pt6v2.py
- **None grid.** readout() reads the grid from CELLS.json. Without CELLS.json it uses the default 5-lr grid. Only the default grid or the default grid + 6.4e-2 is accepted.
- **Out dirs.** A none out dir whose lr is outside the grid becomes a problem, so the strict read-out refuses.
- **N_m** is the argmax over every lr in the grid; a tie goes to the higher lr, as before.
- **Edge flags.** The top and low-edge flags use the grid's own max and min. Under V2 the top flag reads "none at its top lr 6.4e-2". A 1-lr grid (the pt6comb known answer) gets no edge flag, as before.
- **Reading line.** `reading_line(mani)` replaces the fixed pt6comb line in the report. It builds the line from:
  - a Tier L table that covers all 18 ALLOWED domains;
  - the single-nonce-token tied-row caveat for s4, dim, cyc12, torus, hcube, d6 and box5;
  - caesar's real-letter-token note;
  - pt6comb's tail, verbatim (asserted).
- **Q_READING** (pt6comb's line) is kept for the known-answer check.

### Tier L
Tier L means the prior's target carries the answers, so a HELPS reads as "given the structure, the model can use it".

| domains | Tier L | source |
|---|---|---|
| cities, countries, people | yes | ptdom AUDIT 145-156 (anchors); ptdom2 AUDIT 81 |
| months | **no** | ptdom AUDIT 145-156 |
| stars, landmarks, compass, prefixes, ranks, scalar_adj | yes | ptdom2 AUDIT 81 |
| cyc12, torus, hcube, d6, box5, caesar | yes | ptdom3 AUDIT 14-24 (all yes); caesar also tests/byteid_23439548.log |
| s4, dim | yes | pt6comb AUDIT 232-235 |

### Tests (logs in tests/ with a _V2 suffix)
| test | result |
|---|---|
| selftest | 35/35. Mutations 13/13 killed, including 3 new V2 mutants: top flag fixed at 3.2e-2; N_m ignores 6.4e-2; months read as Tier L. |
| known | 15/15, including the two canary-path checks |
| regress_pt6comb | PASS, 80/80, 0 unregistered |
| v2_body_diff, default | 105/105 byte-identical: names, order, texts, trainer args. Placeholder, real refs, alloc b200:16-22 a100:23, canary on a100. |
| v2_body_diff, `--extra_none` | exactly the 8 new pt6v2-none-lr6.4e-2-s16..23; 0 unregistered diff lines (only the B200 slot lines shift) |
| sbatch --test-only | 6/6 accepted (canary A100 01:00:00; B200 03:00:00; A100 03:00:00; none@6.4e-2 on B200 and on A100). No time limit was changed. |
| check_V2 | 105 and 113 texts pass bash -n; `--extra_none 1.28e-1` is refused |
| auto_pt6v2.sh dry, stub pick (placeholder + extra) | every step ran; step 5's dry submit refused with "no COMPOSITION.json", as expected (log auto_20261119T110042Z.log) |
| auto_pt6v2.sh `--test_refs` | CPU job 23707298 COMPLETED. The log has "TABLE dim-cities-countries-compass-hcube PASS" (the re-derivation is identical to the written refs) and "SPLIT-INDEPENDENCE ... PASS (65 comparisons)". The poll survived a timeout kill and resumed from the recorded job id. |

### Deviation from the task text
Step 4 of the chain adds `--canary_dev a100`. Without it, `--write` renders the canary on L40S (the default), and submit's `--canary_dev a100` disk check then refuses with "differ".

### Not tested
- No none@6.4e-2 run has trained on a GPU. Its stability at this lr is unknown, and the canary (all6@2e-3) does not cover it.
- submit_pt6v2.sh's new CELLS.json grep is not exercised end to end, because the dry submit stops earlier at "no COMPOSITION.json". make_jobs' `--disk` line it greps is tested.

## Pick rule readings (coordinator, 2026-11-19 ~07:10 EST; before R2 seed 15 and before any toy3 replication result; pick_pt6v2.py 698562b437f1 + tests/pick_partial_20261119T1104.txt)
- E3 is read on the batch's whole none grid at any seed (the source grid; Part R's R1 grid reaches 1.6e-2), not only on the lrs present at the evidence seeds: the point of E3 is "the none optimum is bracketed". The literal reading failed every Part R row on E3 alone; under either reading every Part R row also fails E1, E4 or E5 today, so the choice changes no pick now.
- E1 uses each source read-out's FLOOR definition unchanged (Part R: ptdom2's shortcut-baseline line, which marks 4 of 5 Part R rows FLOOR).
- The builder's interpretations I1-I4 (lr "present" = run at more than half the seeds of the best-covered lr of that arm; ties -> higher lr; toy3 replication cells derived from the screen's none grid; failed twice = >= 2 terminated attempts by the SUBMITTED markers + sacct) are accepted as written in pick_pt6v2.py's docstring.
- Partial standing (not a pick): no candidate eligible; the fallbacks would give numbers + months + dim + cities + countries + people. torus needs its replication seeds; hcube / d6 / box5 / caesar await the screen.
- Implemented (coordinator edit of pick_pt6v2.py 698562b437f1 -> 9cf7c6b3763f): E3 on the source grid + present lrs; selftest planted world gives prefixes a source grid topping at 4e-3 so the E3-fail case stays tested; SELFTEST PASS 38/38 (tests/pick_selftest_E3src_*.txt). Partial run tests/pick_partial_E3src_*.txt: still no eligible candidate (Part R rows fail E1 FLOOR and/or E4/E5; torus k 1).
- Code audit of pick_pt6v2.py (exp-auditor, fresh; tests/audit_pick_pt6v2.md): no blocker; 31 Part R seed pairs recomputed independently, all equal. Fixed by the coordinator (-> sha 7a2e73faa8c8; SELFTEST PASS 39/39, tests/pick_selftest_audfix_*.txt): (i) a run JSON without check.json is UNFINISHED, not INVALID (the job writes check.json after the run JSON); (ii) a required toy3 replication cell that is INVALID on its first attempt refuses strict mode (re-run once with -r1; a second failure is excused, the toy3 analogue of I4); (iii) toy3 E3 counts none@3.2e-2 in the grid only where it has a valid run. Noted, not changed (registered rule): n is the argmax on the same seeds (g biased low); ranks / scalar_adj are held out by the shortcut FLOOR line and d >= 0.5; Part R is already OUT whatever seed 15 shows, so the pick depends on toy3 only.

## Incident (coordinator, 2026-11-19 06:41-06:58 EST): an accidental --go with a partial pick; nothing ran
- While fixing diff-review item I1 my Edit calls failed (file not read first) and the test command I had sent with them ran `auto_pt6v2.sh --go --pick_json tests/pick_partial_E3src_20261119T1110.json` on the unfixed script. It wrote COMPOSITION.json (the fallback standing months, dim, cities, countries, people), the CPU refs for it (TABLE / SPLIT PASS), ALLOC.json, CELLS.json, 105 job files, and submitted 7 jobs (canary + 6 B200 slot heads, 23708791-23708847) before I stopped it.
- All 7 cancelled while PENDING (sacct: CANCELLED, elapsed 00:00:00, start None): no pt6v2 run trained, no number exists. The written files are moved to tests/incident_20261119_accidental_go/ (COMPOSITION.json, CELLS.json, ALLOC.json, 7 SUBMITTED flags, the auto log, jobs/, the refs job id + done marker). The refs file pt6v2_refs_months-dim-cities-countries-people.json stays (deterministic CPU references, valid if that composition is the pick).
- Since nothing ran, resetting COMPOSITION.json is not a change of a decided composition: the pick rule decides it at pick time as registered.
- Guards added: runs/pt6v2/HOLD (auto --go and submit_pt6v2.sh refuse while it exists; removed by hand right before the real --go); auto --go refuses --pick_json; tests/auto_read_pick.py refuses a PICK.json that is not complete + strict (checked: the partial JSON is refused); tests/auto_count_submitted.py after submit (I3).
- I2 (none@6.4e-2 is a core cell): only relevant if Amendment V2 triggers; then the coordinator first makes a crashed none@6.4e-2 run drop only from that seed's N_m candidates (not the whole seed) before submitting.

## toy3 screen done -> replication cells (coordinator, 2026-11-19 ~09:27 EST)
- expwatch ptdom3 DONE 23/23 (all VALID). readout_ptdom3.py -> runs/ptdom3/tests/readout_screen_final_20261119T1425.txt.
- Registered toy3 replication rule (seed-0 prior pick at 1e-3 / 2e-3 beats the seed-0 best none, not FLOOR): torus (already in), hcube (pick 2e-3, twin-free primary +14.8 vs none@1.6e-2), box5 (pick 1e-3, +33.4 vs none@1.6e-2) qualify; cyc12, d6, caesar are FLOOR. Every qualifier's seed-0 best none is 1.6e-2, so no lower none lr is added.
- Submitted with make_jobs_ptdom3r.py --stage rep: only-hcube@2e-3 s1,2 and only-box5@1e-3 s1,2 (23722036-23722039, B200 slot chains). expwatch ptdom3rep re-armed with all 11 replication jobs. Pick time = these 11 finished (R2 is complete).

## toy3 readiness (prep, 2026-11-19, exp-builder; build and test only, nothing submitted, HOLD untouched)
Outputs in tests/prep_toy3_20261119T1450Z/; details and commands in HANDOFF_prep_toy3.md.
- Dry chain, {box5,hcube} / {box5,torus} / {hcube,torus} (each with dim, cities, countries), Amendment V2 off / on: `auto_pt6v2.sh --pick_json <stub>` 6/6 exit 0 (step 4: 105 / 113 texts parse; step 5 refused at HOLD, as designed); `make_jobs_pt6v2.py ... --check --fake_refs` 6/6 exit 0. No forbidden file written. The 6 dry auto_*.log files moved into the prep dir.
- CPU refs pre-check (cpu_short 23724463; the refs job's computation via tests/refs_pt6v2.py, output redirected into the prep dir, `table` without --write): TABLE PASS and SPLIT-INDEPENDENCE PASS (55 comparisons) for dim-cities-countries-hcube-box5, -torus-box5 and -torus-hcube; 66/66 processes rc 0.
- I2 implemented in readout_pt6v2.py (dbd3654b6f28 -> be0cc0ddcb49). Under Amendment V2, none@6.4e-2 is not a core cell. A started but invalid run of it drops that lr only from that seed's N_m candidates; the seed stays in every verdict and uses its best valid none lr by the same seed-mean ranking, with the flag 'I2: N_m not valid at a seed: ...'. A missing run (no out dir) is still refused in strict mode; --allow_partial treats it like an invalid run. The lr stays in the N_m grid while it is valid at more than half of the read seeds (pick_pt6v2's I1 'present'), and its mean is over its valid seeds. The report prints an 'Amendment V2 (I2)' line. Side fix: the report table no longer raises a KeyError on a crashed record in a non-core cell (it prints '-'). SELFTEST 40/40 (35 old + 5 I2), MUTATIONS 16/16 (13 old + 3 I2), KNOWN 15/15. A diff review of this change is pending (the coordinator starts it).
- Planted toy3 worlds (readout_pt6v2 world() / readout_from() for the 3 compositions x {V1, V2, V2 with an invalid none@6.4e-2 at s17}): primaries hcube = held_nonid twin-free, torus = held_nonid, box5 = held_no_negtwin_nonid (the full set). render_findings_pt6v2.py rc 0 on all 9; `grep -c MISSING` = 1 on each, the renderer's fixed 'MISSING min lm_share / clip_coef' line (the same as the placeholder default sample).
- Canary milestone check: tests/canary_check_pt6v2.py (b5e1aa803f31) covers (a) ran and logged, (b) CANARY READ-OUT 6 of 6 PASS, (c) every intent check PASS. On the real L40S smoke log, A and C pass and B fails, as expected because a smoke has no read-out. Synthetic logs give PASS (exit 0), 5 of 6 (exit 1), flag-only CHECKS FAIL (exit 2) and a truncated log (exit 1). The new readout's `--canary` on pilot/smoke_l40s gives 'CANARY READ-OUT: 6 of 6 manifolds read: PASS'.
- LM share: tmp_audit/pt6v2_code/grad_dp.py runs on the smoke run JSONs: share 0.36 (A100) / 0.32 (L40S), dp_maxdiff ~1e-8, 3 on-window evals each. These are 400-step smokes (prior on at steps 20-160, all6@2e-3), not the full window, so they are a flag only. They are below AUX_MIN 0.5; pt6comb's full-length 53M all6 had 0.76. Read the first full all6@1e-3 run with grad_dp.py as planned (V2 build item 3).
- I2 review fixes (tests/audit_diff_I2.md; readout be0cc0ddcb49 -> bddf63acdf27, render 2f2c021aa1bd -> 20c106615baa; details with file:line in HANDOFF_prep_toy3.md):
  (1) N_m is ranked on the seeds where every grid lr is valid (paired). This is the same as the registered rule when every run is valid. The auditor's counter-example now picks 3.2e-2.
  (2) Strict mode: a none@6.4e-2 run that is unstarted, unfinished, requeued, has no run JSON, or has no check.json yet is MISSING (REFUSED; PROVISIONAL under --allow_partial). Only a finished run with failed checks, or a run that failed twice (pick_pt6v2 I4: >= 2 SUBMITTED attempts, all terminal), is invalid and falls back.
  (3) The renderer prints '6.4e-2 not in the N_m grid (valid at <= half the seeds: k of n)' once, and 'n/a (k valid seeds)' for a null vs_lr.
  Tests:
  - selftest 48/48, mutants 22/22, known 15/15.
  - The auditor's cmp_old_new.py, re-run: V2-off and all-valid V2 worlds give the same res as pre-I2 (60/60 each).
  - Renders 12/12 rc 0; MISSING only on the fixed line.

## The pick (coordinator, 2026-11-19 ~11:40 EST; R2 complete, all 11 toy3 replication runs COMPLETED)
- pick_pt6v2.py 7a2e73faa8c8 strict, exit 0 (inside auto_pt6v2.sh; the same table as runs/ptdom3/tests/readout_rep_final_20261119T1638.pick.txt):
  eligible ranking (d, then mean g): box5 d 9.09 g +34.50; torus d 3.50 g +12.60; hcube d 3.48 g +20.92. Every Part R row OUT; cyc12 / d6 / caesar OUT (FLOOR).
  PICKS box5, torus; FALLBACKS none; COMPOSITION numbers + dim, cities, countries, torus, box5; Amendment V2 not triggered.
- Replication read-out runs/ptdom3/tests/readout_rep_final_20261119T1638.{txt,json} (readout_ptdom3r.py, code-audited; indep.py TOTAL diffs 0; cross-check vs pick EQUAL, 84 fields).
- Noted to the user before the pick (not a rule change): torus's area gain is all in the prior window (in +44.0, out -4.0, final3 -12.2, 0/3); hcube holds after the window (out +6.75 3/3, final3 +4.6 2/3); torus edges hcube by d 0.02. The registered rule picks by d; the user was asked whether to add an end-of-training condition, with the registered rule as the stated default; no reply by pick time.
- Dry chain auto_pt6v2.sh: steps 1-5 pass (105 jobs render; refusal only at HOLD). HOLD removed by hand, then `auto_pt6v2.sh --go`.
- 2026-11-19 ~12:20 EST: canary 23734574 COMPLETED; tests/canary_check_pt6v2.py 23734574 --jobs <all pt6v2 ids> -> CANARY CHECK PASS 15/15 (tests/canary_check_real.txt; CANARY READ-OUT 6 of 6 PASS, CHECKS PASS); expwatch rearm pt6v2. (Note: --jobs takes space-separated ids.)
- 2026-11-19 ~19:30 EST: 103/105 done (left: only-torus@2e-3 s22, only-box5@2e-3 s22, B200 slot chain pending). LM share for all6@1e-3 (grad_dp.py, 8 seeds, 26 on-steps each): 0.69-0.73, dp_maxdiff <= 1.4e-7 -> passes the >= 0.5 aux-loss rule (tests/grad_dp_all6_1e-3_full.out). Provisional read-out (--allow_partial, 7 read seeds, s22 out): tests/readout_partial_20261119T1930.{txt,json} -> 6 of 6 HELPS, interference not detected [PROVISIONAL]. Flags: dim and box5 N_m = none@3.2e-2 (grid top; box5 none still rising 31.2 -> 42.4); final3 g torus -4.7 (0/7), box5 -2.0 (4/7); s23 ran on a100 (note only). Strict read-out waits for the 2 runs.
- 2026-11-19 ~20:01 EST: batch DONE (105/105; expwatch pt6v2 finished). Strict read-out (readout_pt6v2.py bddf63acdf27, all 8 seeds): tests/readout_final_20261119T2001.{txt,json}; render tests/render_final_20261119T2001.md (MISSING count 1 = normal). HEADLINE 6 of 6 HELPS, interference not detected; every manifold 8/8 wins. Frozen code copies for the result audit: tests/frozen_final/. Result audit (exp-auditor) started -> tests/audit_result_final.md. Flags unchanged: dim and box5 N_m = grid top 3.2e-2 (post-hoc none@6.4e-2 being added, Amendment V3); final3 g torus -3.0 (1/8), box5 -1.6 (5/8).

## Amendment V3 (post-hoc, 2026-11-19)

User-approved 2026-11-19 ~20:00 EST (relayed by the main session; B200 approved). Post-hoc: Amendment V2's own condition was not triggered, and this amendment is written after the registered read-out exists.

- What was added: one more no-prior baseline lr, none@6.4e-2, at seeds 16-23 (8 runs `pt6v2-none-lr6.4e-2-s{16..23}`), recorded in `runs/pt6v2/POSTHOC_V3.json` (job ids, devices, slots, job-text sha). Driver `runs/pt6v2/posthoc_v3_jobs.py`: each job text = `make_jobs_pt6v2.render_v2(none, 6.4e-2, seed)` on the seed's ALLOC.json device, with the slot `make_jobs_pt6v2.py --extra_none 6.4e-2` would give (byte-identical to that rendering); s16-22 on the existing B200 singleton slots (3-GPU cap unchanged), s23 on a100. CELLS.json (ea9a9b01f081), make_jobs_pt6v2.py (bb15d248afd8) and the 105 registered jobs are not touched.
- Why: in the registered read-out dim and box5 pick N_m = 3.2e-2, the top of the registered grid, and box5's none is still rising there (1.6e-2 -> 3.2e-2: 31.2 -> 42.4). One lr above the grid says whether g = all6@1e-3 - none@N_m shrinks once the baseline gets a better lr.
- Read-out: `readout_pt6v2.py` without a flag = the registered read-out; the post-hoc runs are left out of every number, problem and NOT VALID line even when their directories exist (tested: output identical to the pre-V3 read-out except the sha lines, with 2 of the 6.4e-2 directories already on disk). `--posthoc_none 6.4e-2` prints blocks 1-2 unchanged, then block 3 "extended grid (post-hoc V3, user-approved 2026-11-19)": the same verdict rule on the grid + 6.4e-2 with the I2 rules for the extra lr (an invalid run -> that seed's other none lrs, ranked paired; a missing run -> block 3 PROVISIONAL; 6.4e-2 in the N_m grid only if valid at more than half of the seeds; failed twice -> invalid). Per manifold: N_m, g, 90% boot, wins, p (Holm), final3 g, and the change vs registered. `--json` puts it under the key `posthoc_V3`; `render_findings_pt6v2.py` adds a compact extended table after the registered table when that key is present (otherwise byte-identical output).
- Reading rule: the registered verdict stays the registered one; the extended verdict is reported beside it. If a manifold's extended g changes sign or loses HELPS, report both, and the paper wording follows the extended one (accurate, not defensive). If nothing changes, the paper says the gain holds with the baseline's grid extended to 6.4e-2.
- 2026-11-19 ~21:10 EST (coordinator): V3 jobs 23783618-23783625 submitted by the builder (HANDOFF_V3_none64.md); 3 running at 21:05. Verified flag-off read-out (readout_pt6v2.py cae52696322d) vs the audited final output: text identical except the 2 sha lines, JSON byte-identical. expwatch pt6v2-v3 armed (batch end / failure). Diff review (exp-auditor) of the V3 readout/render diff started -> tests/audit_diff_V3.md. F105 written from the registered read-out (manifolds main 00c0794a); dim / box5 re-read after V3.
- 2026-11-19 ~21:40 EST: V3 diff review (tests/audit_diff_V3.md): no blocking, no important items; flag-off unchanged (JSON cmp identical, render byte-identical); extended numbers match an independent implementation (79/79 over 6 planted worlds); job texts equal the registered none@3.2e-2 texts except lr and slot. Minor (not fixed): render_findings_pt6v2.py:128-142 does not show how many 6.4e-2 runs are valid / whether 6.4e-2 entered the grid -> when reading V3, take that from the text read-out (block 3), not the render table.
- 2026-11-19 ~21:40 EST: V3 DONE (8/8 COMPLETED, checks pass). Extended read-out tests/readout_V3_20261119T2130.{txt,json}: 6.4e-2 valid 8/8, in the grid; no N_m changes (none@6.4e-2: dim 35.0, box5 24.1 < 3.2e-2's 47.3 / 41.0) -> extended = registered, 6/6 HELPS. Per-seed best over 6 lrs (auditor's indep.py + 6.4e-2, scratch copy): dim +32.9 (8/8), box5 +18.2 (7/8). FINDINGS F105 addendum + markers on F105 qualifier 1 / 不得写 3 (manifolds main 2eade500); Notion row updated; numbers sent to the writing session.

- 2026-11-20 ~00:30 (coordinator): post-hoc descriptive read, no training (user asked "how much faster", "what does torus final3 −3.0 mean", "why is all6 above only-m"). Scripts = the auditor's indep.py loader (lines 1–172, NONE_LRS incl. 6.4e-2) + new tails; outputs in tests/posthoc_speed_20261120/. Time to 90% of none@N_m final3 above chance (8-seed mean curve, eval every 250 steps): all6@1e-3 is 3.9 / 4.4 / 2.9 / 3.1 / 4.1 / 5.0× sooner (numbers/dim/cities/countries/torus/box5); fastest none lr at that threshold = N_m in all six. torus: all6 peaks 98.2 at window close, best none mean curve at any lr 94.6; final3 90.4 vs none@1.6e-2 93.3. h (all6@2e-3 − only-m@2e-3) arises mostly after the window: dim in +4.4 / after +19.9, torus −0.2 / +7.5, box5 +5.1 / +9.3; final3 h dim +25.0 (8/8), torus +11.7 (8/8). Result audit of these numbers requested (tests/audit_speed_20261120.md).
- 2026-11-20 ~01:00 (coordinator): read-only representation check of the "lattice priors share something" explanation (tests/posthoc_speed_20261120/repr_check.py, repr_out.txt; logged cka_<d> and gcos_<a>|<b>). Cross-domain CKA, only-X@2e-3 − none@2e-3 at window close: only-box5 → dim +0.159 (8/8), only-dim → box5 +0.183 (8/8), only-torus → dim +0.084 (7/8) / box5 +0.090 (7/8), dim/box5 → torus ≈ 0; only-cities → countries +0.222 (8/8), only-countries → cities +0.237 (8/8); cross-family ≤ 0.06. Own-target CKA after the window is retained equally in all6@2e-3 and only-m@2e-3 (dim 0.975 vs 0.971, torus 0.977 vs 0.975, box5 0.968 vs 0.969), so the extra accuracy drop of only-m is not a loss of geometry at the CKA site. In-window gradient cosines between priors are all ≤ 0.05 (top: cities|countries, dim|torus, torus|box5, dim|box5). User approved V4 (all6full ×8, all4 ×8) 2026-11-20; builder delegated.

## Amendment V4 (post-hoc, 2026-11-20)

- Approval: user 2026-11-20 ("ok，怎么快怎么来", relayed by the main session; B200 approved). Post-hoc, exploratory, **no verdict rule, no gate**; the registered read-out and the V3 read-out are untouched (readout_pt6v2.py and its outputs unchanged, see below).
- Cells (8 seeds each, s16-23; record runs/pt6v2/POSTHOC_V4.json; driver runs/pt6v2/posthoc_v4_jobs.py):
  - **all6full@1e-3** = the registered all6@1e-3 job with `--window 0.05,1.0` (was 0.05,0.4). Question: does keeping the priors on prevent the post-window drop (all6@1e-3 means torus 98->90, box5 81->53, dim 93->84 from window close to final)?
  - **all4@2e-3** = the registered all6@2e-3 job with `--prior_domains numbers,dim,cities,countries` (torus, box5 priors off). `--domains` stays all six. Question: does dim's +13.5 lift of all6@2e-3 over only-dim@2e-3 need the torus / box5 priors?
- Reading: descriptive, each run against the existing pt6v2 runs of the same seed (all6full vs all6@1e-3; all4 vs all6@2e-3 and only-<m>@2e-3). The first finished run of each cell is its canary (no separate canary job).
- Verified before submit:
  - window 1.0: pretrain_range.py (Rev H a398fccbfe93) `prior_on(s) = w0*steps <= s < w1*steps`, loop `range(start, steps)`, steps 18310 -> on for s = 916..18309 (the last step), no validation of w1; prior_events records toggles only, so the trainer's events = [[916, 1]] (make_rb6_tasks.window_events gives [[916,1],[18310,0]], the off event never reached). The job intent sets events [[916,1]], on_frac 17394/18310 = 0.94997; CHECK_PY_6 (unchanged) asserts both at the end of the run, so the first finished all6full run proves the prior stayed on to the end.
  - all4: `lam_rule each` -> LAMD = lam (1) for each of the 4 prior domains, 0 for torus / box5 (pretrain_range.py 684-687); intent lam_domains / active and the job's dry block (`torus:false:false:0.0`, `box5:false:false:0.0`) say the same. Data: --domains unchanged and the domain / order RNGs are seeded by the seed only, so the in-job data_sha0 / dry_inputs checks against the same per-seed CPU references the only-<m> arms pass apply as is.
  - Each V4 job text = the same seed's unpatched render_v2 base job with only the expected substitutions (tag, out -> out_v4, the window or prior_domains argument, the two dry-block specs) plus the header description, one provenance line and the intent line (intent differs from the base in exactly: all6full args.tag / args.window / events / on_frac; all4 args.tag / args.prior_domains / active / lam_domains.torus / lam_domains.box5). bash -n and sbatch --test-only pass for all 16.
  - readout_pt6v2.py (flag off) and `--posthoc_none 6.4e-2`: text and JSON outputs byte-identical before vs after the V4 files exist.
- Placement: job files, SUBMITTED_ flags and run dirs in runs/pt6v2/out_v4/ (readout_pt6v2.py reads out/ and its run-name regex has no all6full / all4 arm; a6_guard_v2's glob of runs/pt6v2/job_*.sbatch does not see them). Nothing registered touched: CELLS.json ea9a9b01f081, make_jobs_pt6v2.py bb15d248afd8, readout_pt6v2.py cae52696322d, ALLOC.json 2d96730f431d, POSTHOC_V3.json 75bed7fe30ee.
- Device: all 16 on B200, 3 singleton chains llmadd_pt6v2-b200-slot0/1/2, order s16 all6full, s16 all4, s17 all6full, ... round-robin. **s23 device change**: ALLOC.json gives a100; a100_tandon was blocked (group GPU quota QOSGrpGRES) at build time, so s23 runs on B200 (flag only).
- Backup of this file before the section: tests/AUDIT.preV4.md.

## Amendment V5 (post-hoc, 2026-11-20)

User-approved 2026-11-20 ("可以的，没问题", relayed by the main session; B200 approved). Post-hoc, exploratory: no verdict rule,
no gate; readout_pt6v2.py, CELLS.json, ALLOC.json, V3 / V4 and every registered job / out dir are untouched.

- Why: at lr 2e-3, dim accuracy at the end of training is 77.3 with all six priors (all6) vs 52.3 with only the dim prior
  (only-dim), while the logged cka_dim@L2 / @L4 (item positions) is retained equally. Question: is the representation in later
  layers, or at the answer position, washed out less in the joint model?
- Runs (4): all6@2e-3 and only-dim@2e-3, seeds 16 and 17 (both B200 in ALLOC.json). Training arguments = the registered job's
  ARGS byte for byte, plus `--save_model_at 7323,12000,18309`; out dir runs/pt6v2/out_v5/<registered name>/.
- Saved weights (model.state_dict(), no optimizer), after step s's update = the weights the eval of step s logs:
  7323 = the LAST step with the prior on (prior_on(s) = 916 <= s < 7324; no eval at 7323, nearest evals 7250 / 7500),
  12000 (an eval step), 18309 (the final step and last eval). File `<tag>/wts_whole_helix3_seed<s>_<tag>_step<k>.pt`.
- Trainer: runs/pt6v2/out_v5/code/tools/pretrain_range.py 041b2f4241b9 = a398fccbfe93 + one flag `--save_model_at`
  (patch out_v5/patch_trainer_v5.py, diff out_v5/trainer_v5.diff). No RNG draw, no forward pass; at its default the flag is
  kept out of the run JSON's args. CPU smoke (out_v5/smoke/, job 23858908): flag off = registered trainer (run JSON identical
  except code_sha / out / timing; stdout identical except the sha, the code path and one warning's line number); flag on =
  identical curve, wts at the final step == the --save_model model, stop-after-10 + resume gives bit-identical snapshots;
  real-config --dry_inputs rows = the CPU reference (63 lines, b71458d3a12d).
- Devices: each run has a B200 job (singleton chains slot3 = all6, slot4 = only-dim) and an A100 / H100 racing copy
  (<slurm-account>, a100_tandon,h100_tandon, in-job GPU check accepts A100 or H100) on the same out dir. An early
  two-phase run-guard claim before the first write to $OUT lets the first claimant compute; the other exits 0. A device that is
  not the original B200 is a device change vs the registered run: flagged in the read, not a gate. A takeover after a
  preemption (resume on the other device) shows in node.txt and is flagged the same way.
- Read: runs/pt6v2/probe_v5.py (CPU) per run: drift CKA (7323 vs 18309), CKA to dim's exponent vectors, ridge decodability,
  per layer (embedding output + every block), at the item positions and the answer-prediction position. Descriptive.
- Driver runs/pt6v2/posthoc_v5_jobs.py; record runs/pt6v2/POSTHOC_V5.json (job ids); handoff runs/pt6v2/HANDOFF_V5.md.
- 2026-11-20 ~13:20 (coordinator): V4 canary all6full s16 (B200 23857429) COMPLETED, CHECKS PASS (prior_events [[916,1]], on-frac 0.95, prior on at the last eval; data_sha = pt6v2 s16). Preliminary 1-seed read (tests/posthoc_speed_20261120/v4read.py): final3 torus 99.9 (window version 84.0; best none any lr 97.5), box5 97.4 (64.5; 62.5), dim 98.3 (90.8; 63.2); bpb 1.2260 (all6 window 1.2362, none@2e-3 1.2058). Race copy 23859535 exited (twin was running).
- 2026-11-20 (coordinator): probe_v5 code review (tests/audit_probe_v5.md, saved verbatim by the main session): no blockers; important 1-4,6 (per-row OOF scoring by strict/slot/hit subsets, answer-position carry-over control, slot split, item-mean CKA + frozen decoder, DiD summary) being applied by a builder (analysis only); item 5: V5 seeds are effectively redrawn (copies on A100/H100) -> check each pair's own behavioural strict gap first; a pair with < 10 pt cannot answer the question -> up to 2 extra seeds under the standing authorization.
- 2026-11-20 (coordinator): probe_v5 review fixes applied (probe_v5.py 803c3cf6c8e5 -> dfaa756340ef; selftest_v2 PASS incl. a synthetic strict-only effect that the DiD catches on strict (+1.05 / +0.83) and not pooled (-0.00 / -0.03)); carry-over control = residual R2 (pure carry -29/-26, computed answer +0.30/+0.33). No blockers in the review, so no diff re-review.
- 2026-11-20 ~14:45 (coordinator): all4 s16 done on race copy 23859539 (A100, device change vs registered B200; flag only), CHECKS PASS. Preliminary 1-seed read: dim area all4 69.1 vs only-dim@2e-3 82.0 (h4 -12.9; all6 h6 +4.3 for this seed), final3 dim 63.2 vs all6 92.5 / only 77.0; torus/box5 without their priors still 47.5 / 31.7 vs none@2e-3 11.5 / 19.8; bpb 1.2391 (all6@2e-3 and none@2e-3 1.2058). Cancelled B200 twins 23857430 23857431 23857432 (their copies hold the claim).
- 2026-11-20 ~14:55 (coordinator): all6full s17 (copy 23859542, A100) CHECKS PASS: final3 torus 100.0 (window 93.0; best none 95.5), box5 94.9 (39.4; 62.6), dim 96.5 (86.6; 61.9); bpb 1.2083 (window 1.2187, none@2e-3 1.2046). Cancelled B200 twin 23857433.
- 2026-11-20 ~15:25 (coordinator): all4 s17 (copy 23859545, A100) CHECKS PASS: dim area all4 68.5 vs only-dim 67.7 (h4 +0.8; all6 h6 +12.9 for this seed); final3 dim 64.5 / all6 81.8 / only 58.9; bpb 1.2099 (normal; s16's 1.239 not repeated). Two seeds: h4 -12.9 / +0.8 vs h6 +4.3 / +12.9 (all6 - all4 on dim area: +17.2 / +12.1).
- 2026-11-20 ~15:35 (coordinator): all6full s18 (copy, A100) CHECKS PASS: final3 torus 100.0 (window 95.5; best none 95.9), box5 94.1 (56.2; 68.6), dim 100.0 (88.9; 47.3); bpb 1.2077 (window 1.2084, none@2e-3 1.2061). Cancelled B200 twins 23857438 23857439.
- 2026-11-20 ~16:00 (coordinator): all4 s18 (copy, A100) CHECKS PASS: dim area all4 69.4 vs only-dim 70.3 (h4 -0.9; h6 +17.8); final3 dim 58.0 / all6 90.5 / only 55.6; bpb 1.2067. Three seeds h4 -12.9 / +0.8 / -0.9 vs h6 +4.3 / +12.9 / +17.8.
- 2026-11-20 ~16:25 (coordinator): all6full s19 (copy, A100) CHECKS PASS: final3 torus 99.5 (window 88.3; best none 95.1), box5 94.9 (57.8; 51.9), dim 100.0 (85.3; 61.4); bpb 1.2045 (window 1.2029, none@2e-3 1.1991).
- 2026-11-20 ~17:50 (coordinator): V4 B200 twin 23857453 (all4 s19) started while its copy 23859614 held the claim: the frozen body exited the trainer at the guard, then ran the intent checks on the copy's partial curve (56 evals) and wrote check.json FAIL + B200 rows into gpu.csv (expected per the "Race copies" note; the copy's own end-of-run check rewrites check.json; its GPU-type line may FAIL on the mixed gpu.csv rows = device flag only). Cancelled twins 23857467 (all4 s20, RUNNING 1:27 on B200, cancelled before any write: out dir holds only the copy's claim), 23857468 (all6full s21), 23857470 (all4 s21); copies hold the claims. all6full s20 is being computed on B200 by twin 23857454 (its copy 23859617 declined at the guard).
- 2026-11-20 ~17:50 (coordinator): V5 guard path bug (found while reading): every V5 sbatch guards RUNJ=run_whole_helix3_*, but only-dim runs write run_whole_none_* (no numbers prior -> target none). For only-dim the early guard never sees the result as complete, so a copy that starts after its B200 twin finished would pass the early guard; the trainer's own guard (correct path) then returns, and the body would rewrite check.json / gpu.csv / dry_inputs.txt (weights and run JSON safe). Mutual exclusion while both RUN works. Handling: cancel the losing partner by hand. Cancelled 23859742 (only-dim s16 copy; B200 23859741 RUNNING with the claim) and 23859740 (all6 s16 copy; B200 23859739 done, CHECKS PASS, 3 weight files). only-dim s17 (23859745 B200 / 23859746 copy): cancel the other as soon as one starts. compare_curve_v5.py had the same hard-coded name: fixed to run_whole_none_* for only-dim (1408fd37ed82 -> f50c92bfb9e7).
- 2026-11-20 ~17:50 (coordinator): V5 all6 s16 rerun (B200, same device as registered) CHECKS PASS, bpb 1.2069; curve not bit-identical to the registered run (11364 differing fields over 75 evals, e.g. held_b23_ge1000 up to 0.75, lpmargin up to 5.3 nats): GPU nondeterminism diverges the trajectory, so V5 pairs are re-draws as the probe audit (item 5) expected -> read each pair's own behavioural strict gap before the probe.
- 2026-11-20 ~18:00 (coordinator): all 8 remaining V4 copies now RUNNING on A100/H100 and hold their claims; cancelled the last B200 twins 23857483 (all4 s22), 23857485 (all6full s23), 23857487 (all4 s23). Twin 23857471 (all6full s22) had started first and, like 23857453, wrote check.json FAIL from the copy's partial curve (9 evals) before ending; the copy rewrites it at its end. Only all6full s20 runs on B200 (twin 23857454 holds the claim).
- 2026-11-20 ~18:05 (coordinator): all4 s19 (copy 23859614, A100) CHECKS PASS (the copy's end check rewrote the twin's partial FAIL; GPU line PASS): dim area h4 = all4 - only-dim +5.1 (h6 +25.6), final3 dim all4 50.5 / all6 88.1 / only 44.2; torus / box5 (no own prior) area 54.3 / 18.4 vs none@2e-3 19.9 / 3.1; bpb 1.2150 (all6@2e-3 1.2012, none@2e-3 1.1991). 4/4 all4 seeds: h6 - h4 on dim = 17.2 / 12.1 / 18.7 / 20.5. V5: B200 23859743 (all6 s17) declined at the V5 guard (copy 23859744 RUNNING on H100 holds the claim), nothing written.
- 2026-11-20 ~18:20 (coordinator): all6full s20 (B200 twin 23857454) CHECKS PASS: final3 torus 100.0 (window 88.8; best none 95.0), box5 84.3 (24.3; 44.7), dim 93.3 (64.6; 66.2); bpb 1.2071 (window 1.2162, none@2e-3 1.2083). 5/5 all6full seeds: final3 above window and above the best none lr on torus / box5 / dim.
- 2026-11-20 ~18:20 (coordinator): V5 only-dim s16 (B200 23859741) CHECKS PASS, 3 weight files; cancelled its pending B200 partner 23859745 for only-dim s17 (copy 23859746 RUNNING on H100 holds the claim; the guard-path bug above). V5 pair s16 behavioural dim strict (58 q): gap all6 - only-dim @7250 +0.0, @12000 +15.5, @18309 +24.1 (DiD +24.1; the registered pair: +5.2 / +1.7 / +17.2, DiD +12.1) -> pair s16 answers the question (gap >= 10 pt). s17 pair: both copies on H100 (same device within the pair; device change vs registered B200, flag only). Note: speed_base.device() reads gpu.csv column 0 (timestamp), so the "dev" field of v4read / v5pair prints a time, not a GPU name; GPU names come from check.txt / compare_curve_v5.py (cosmetic).
- 2026-11-20 ~18:15 (coordinator): V4 s20-s22 CHECKS PASS (all4 s20/s21/s22, all6full s21/s22; copies on H100 except where noted by check.txt). all6full final3 torus / box5 / dim: s21 99.5 / 95.5 / 98.9 (window 83.9 / 65.0 / 89.8; best none 93.6 / 60.2 / 69.9), s22 99.2 / 70.6 / 85.5 (93.2 / 41.1 / 71.4; 93.7 / 44.6 / 55.7) -> 7/7 seeds above window and above the best none lr on all three. all4 dim area h4 / h6: s20 +0.8 / +13.6, s21 +5.7 / +16.8, s22 -0.3 / +14.3 -> h6 > h4 in 7/7 (h6 - h4 = 17.2 12.1 18.7 20.5 12.8 11.1 14.6). bpb note: none@2e-3 s22 = 1.2387 (the known s22 outlier), so the final bpb read must not rest on none@2e-3 alone.
- 2026-11-20 ~18:35 (coordinator): V4 complete, 16/16 CHECKS PASS, data_sha = pt6v2 in all. Devices: all4 A100 x3 / H100 x5; all6full B200 x2 / A100 x3 / H100 x3 (registered pt6v2 runs all B200: device change, flag only). 8-seed read tests/posthoc_speed_20261120/v4final.py -> v4final_out.txt: all6full - all6win@1e-3 final3 dim +12.3 [+5.9,+18.6] torus +9.4 [+5.4,+13.4] box5 +38.5 [+27.9,+49.1] (8/8 each); all6full beats the per-seed oracle best none lr on all six manifolds 8/8 (numbers +31.4, dim +35.2, cities +2.5, countries +2.1, torus +4.1, box5 +33.2); bpb 1.2112 vs none@2e-3 1.2098 (+0.11%), per-seed best none 1.2065 (+0.38%), window 1.2133. all4: dim area h4 +1.5 [-4.8,+7.8] (5/8) vs h6 +13.5 [+7.4,+19.6] (8/8), all6 - all4 area +12.0 [+3.8,+20.2] / final3 +22.0 [+8.6,+35.5] (7/8); numbers/cities/countries all6 - all4 within 0.4. Side read (v4tor.py): at lr 2e-3 every single prior lifts torus behaviour over none (area 19-37 vs 8.7) and all4 lifts it to 47.9 (8/8) - torus at 2e-3 is far below its N_m, so not a headline. Result audit (exp-auditor) started in the background.
- 2026-11-20 ~18:55 (coordinator): V5 all 4 runs done, CHECKS PASS, 3 weight files each (s16 pair B200, s17 pair H100 copies; device change vs registered B200 for s17, flag only). Behavioural dim strict gap all6 - only-dim (out_v5/v5pair.py): s16 @7250 +0.0 / @12000 +15.5 / @18309 +24.1 (DiD +24.1); s17 +6.5 / +33.8 / +22.1 (DiD +15.6); registered pairs DiD +12.1 / +10.4. Both pairs have a gap >= 10 pt -> both answer the question, no extra seed. probe_v5.py --v5 launched on cpu_short (cpu_run.sh, 16 cpu, 48G, 1 h) -> out_v5/probe_v5.log.
- 2026-11-20 ~19:20 (coordinator): code release step (a) (writer relayed the user's release request): pushed branch ptdom3 (ef7b8854 = the pt6v2 trainer) and refreshed the public mirror torch/llmadd_torch for pt6v2 / ptdom3 / ptdom2 / pt6comb / ptdom (manifolds main e85d205; 98 files, Slurm account names redacted, secret scan clean). LICENSE (MIT) and the README move wait for the user's own confirmation.
- 2026-11-20 ~19:45 (coordinator): probe_v5 done (0 warnings; out_v5/probe_v5.log). Descriptive read: item positions (slot A, strict): lattice R2 0.96-0.99 at 18309 in all four runs, all6 - only-dim DiD_R2 within -0.004..+0.012 at L2-L6 (both seeds); answer position L5-L6 strict: R2 0.97-0.99 and nearest-lattice acc 1.000 in all four runs at all three checkpoints, while only-dim's own strict accuracy falls to 0.707 / 0.597 at 18309 (behaviour DiD +0.241 / +0.156 logged). Answer-position L2-L4 and transport-residual R2: mixed signs across seeds. Reading (preliminary, first result audit running): the post-window loss is not a loss of the item or answer-position representation; it is in the readout to the output token. Result audit (exp-auditor) started in the background.
- 2026-11-20 ~20:10 (coordinator): V5 result audit back (tests/audit_v5_probe_result_20261120.md): S1 two fixes (s17 behaviour DiD was from eval 7250 -> window-mean baseline +21.6 / +18.2, verified out_v5/v5win.py; 'every layer' -> layers 2-6), S2 'wrong answers' -> strict ones, S3 holds and is strengthened by the transport residual on miss rows. FINDINGS F107 (manifolds main a5ef915), Notion V5 row green, writer informed.
- 2026-11-20 ~20:50 (coordinator): release code gap fix: tools/manifold_domains.py data dir configurable (MANIFOLD_DATA) + missing-CSV domains skipped; pushed ptdom3 9d930c9 and main b313231; tag pt6v2-run = ef7b8854 (the commit the pt6v2 / V3 / V4 jobs ran). Checked (/scratch/yh6210/tmp_audit/relcheck): --dry_inputs of all6@1e-3 s16 byte-identical old vs patched without CSVs (table shas unchanged); all 36 domain tables identical with CSVs. The frozen run snapshots runs/*/code are untouched.
- 2026-11-20 ~21:40 (coordinator): public release pushed, manifolds main 85f21ff (MIT LICENSE; reader-facing README from the writing session; old Burst README -> burst/README.md; 2288 per-seed JSON/CSV outputs under torch/llmadd_torch/runs/<batch>/out*/, redacted, numbers checked against sources). User confirmed in the coordinator session ('ok 按你推荐的来'); recipe confirmed as Overleaf 051d2e2 (several priors on to the end).

## Amendment V6 (post-hoc interpretability, exploratory, 2026-11-22)
User asked 2026-11-22 for interpretability verification of why the torus/box5 priors help dim; no-training read-outs are standing-authorized.
Exploratory tier: no verdict rule, no gate; every number is reported. Code (exp-builder 2026-11-22): probe_v6.py (driver), probe_v6_lib/make_probe_copy6.py (probe copy of the V5 trainer sha 041b2f4241b9, counted edits E1-E4 as V5), probe_v6_lib/probe6_inner.py (the read-out, runs inside the trainer's main() with its own eval prompts), probe_v6_aggregate.py; out_v6/job_probe_v6.sbatch (cpu_short array over the 12 V5 checkpoints: all6 / only-dim x s16 / s17 x steps 7323 / 12000 / 18309; 7323 has no logged eval). Outputs only under out_v6/.
Position: the answer-prediction position ap = seq - len(span) - 1 (the ' is' token, asserted, whose logits give the answer's first token; spans ' <item>' + '\n' = 2 tokens, asserted). Layers: 0 = embedding output, l = block l output. Forwards = the trainer's evaluate() (bf16 autocast, 250-row chunks, logits_to_keep = dom_max_ans + 1, exact match over the span). Sets: <d>_held / <d>_seen of dim, box5, torus (training template), seen rows of cities / countries; dim strict = rainbow6_summary.strict_mask (V5).
- A1 decoders: grouped (by answer item) 9-fold ridge, alpha by inner grouped CV (V5's procedure, any width), state at ap -> dim 3-D exponents / box5 2-D integers / torus DTAB = the prior's cos/sin of both Z6 coordinates (4-D); OOF R2 and nearest-lattice accuracy on held + seen pooled, held, dim strict. Null: the same CV on 10 answer-label permutations (shuffled-label decoders, expect R2 <= 0). Layers 2-6.
- A1 overlap: mean cos^2 of the principal angles between (a) the ridge decoder column spaces (rank = coordinate width, fit on seen rows) and (b) the top-k (k 3, 6) PCs of per-answer class means of seen rows, for dim-box5, dim-torus, box5-torus, dim-cities, dim-countries. Nulls: random subspaces of the same ranks (200), covariance-matched random subspaces (random combinations of each domain's own centred states, 200), shuffled-label decoders (10 per side); for (b) class means of row-shuffled answer labels (10).
- A1 transfer: decoder fit on the source domain's rows (all), applied to the target's states, then an OLS map (intercept) from those 2-4 outputs to the target's coordinates, grouped 9-fold OOF R2 (and dim strict); pairs box5<->dim, torus<->dim, box5<->torus. Nulls: source decoder fit on shuffled labels (10); random projections of the same width (10). Own-decoder R2 printed beside it.
- A2 ablation: at ap only, h <- h - ((h - mu_E) U) U^T (mean ablation: mu_E = the evaluated set's clean mean at that layer, fp32), on the state fed to block h + 1 for h in {3, 4, 5} or to the final norm (h = 6); later positions (the '\n' prediction) see the edit only through attention. U = top-k PCs of a source domain's SEEN-row class means at that layer; k = c_E and 2 c_E (c_E = the evaluated domain's coordinate width: dim 3, box5 2, torus 4). Evaluated on dim_held (exact match held + strict, first-token and '\n'-token accuracy, answer margin) with U from box5 (i), torus (ii), dim itself (iii, positive control: fit on seen, evaluated on held), 20 random k-subspaces (iv, negative control), cities / countries (v, specificity); reverse: box5_held and torus_held with U from dim, themselves, the other lattice domain, cities / countries, random. Re-run of blocks h+1..6 from the stored clean block inputs; k = 0 must give logits equal (torch.equal) to the clean forward (asserted, else the full-forward hook path) and the clean hits are compared with the logged hits (cross-device: flag only) and the V5 CPU probe.
- A3 logit lens: final RMSNorm (fp32) + the tied unembedding on each layer's ap state, restricted to the domain's answer tokens; margin = correct-token logit - max over the other answer tokens, top-1 among them; dim strict / strict misses / all, box5 and torus held. DLA: every block's attn and MLP output (post-sublayer norms = what is added to the residual) through the final norm with its scale frozen at the real final residual, on the margin direction (correct - the best other token at the last layer); the last two blocks are tabled. Asserted / printed: the last-layer lens (bf16, the model's own path) equals the logits; components sum to the final residual.
- A4 weights: linear CKA between the answer tokens' (tied) embedding rows and the lattice coordinates per domain (null: 200 row permutations); CKA between per-answer class means of the final-norm L6 state at ap and those tokens' unembedding rows (null: permutations).
Hypothesis (a prediction, not a rule): box5 and torus share an answer-coordinate subspace with dim at L4-L6 (overlap and transfer above their nulls, in both arms), and in only-dim the break is in the readout: dim-strict lens margin collapses only at the last layer(s) / in the unembedding alignment (A3, A4 lower in only-dim than all6 while A1 R2 is equal); removing box5's or torus's answer subspace lowers dim strict accuracy more than random subspaces do, less than dim's own (iii), and more in all6 than in only-dim; cities / countries subspaces act like random.


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

## Amendment V6 addendum: fixes after the code review (exploratory, 2026-11-22)

The code review is `tests/audit_v6_code_20261122.md`. These fixes were made before the full run. Every number is still reported, and there is still no verdict rule.

**Torus coordinates (B1).** The prior's DTAB torus table holds the frequency-1 characters (1,0) and (0,1) of Z6×Z6. The trained models' answer states encode other characters.

How the pair is chosen:
- For each checkpoint, the read takes the generating pair of characters (36 pairs up to sign; det of the pair a unit mod 6) with the highest mean grouped-ridge OOF R² on the SEEN torus rows, over layers 4–6.
- Held rows are not used to choose, so held R² stays clean.

Where the chosen pair is used:
- A1 R² and nearest;
- the decoder-space overlap;
- the transfer;
- the A4 coordinate CKAs.

The DTAB pair is reported beside it ('torus_dtab'). Class-mean PC overlap and A2 do not use coordinates and are unchanged.

**A2 dose (B2).**
- **Dose.** Every ablation condition records its dose: the fraction of the evaluated rows' centred answer-position variance at that layer which the subspace removes.
- **Two dose-matched nulls, 5 draws each.**
  - Random subspaces drawn from the evaluated set's own state covariance.
  - Class-mean PCs of each other lattice source, with the answer labels shuffled.
- **Read.**
  - The drop is clean minus ablated.
  - The drop is also given per unit dose.
  - The read also gives the excess over a line, drop = a + b·dose, fitted through all null draws (isotropic random, covariance-matched, shuffled-label) at the same dose.
  - This is computed on held accuracy and on dim strict.

**Reporting.**
- A2 held (all held rows, n=342 for dim) is the primary row, with strict (n=58) next to it.
- Class-mean PC overlap is the primary A1 overlap; decoder-space overlap is secondary.
- Shuffle nulls are reported as z-scores and as the max, not as a p95 at n=10.
- The aggregate reports A2 as the drop from each arm's own clean value, plus the all6 − only-dim DiD. It prints the per-seed differences and the number of seeds in each mean.
- 2026-11-22 (coordinator): V7 canary 24064071 read: every check_v7 drop check PASS (flag consumed; torus/box5 sites 0 from the drop step, every other domain > 0; dropped_lines = cycling rule; data_sha = registered before the drop, different after; kept line types' token share = registered). Only FAIL = step-0 lm vs the registered B200 run, rel 1.15e-6 (11.597748 vs 11.597734, canary on A100) = execution-environment difference, flag only (check_v7 exits 0 on flag-only). The registered CHECK_PY's prior_events / prior_on_frac FAILs are by design for the 800-step canary. Main runs continue; watcher re-armed.
- 2026-11-22 (coordinator): V6 review fixes diff-checked (B1 torus pair chosen on SEEN rows only, probe6_inner.py:426-450; B2 dose + cov / shuffled-label nulls, :699-746; B3 aggregate drops + DiD, probe_v6_aggregate.py:37-142); post-fix smoke 24064752 rc=0, k=0 no-op, own-subspace control hurts (0.792 -> 0.058), torus chosen-pair R2 0.70-0.87 vs shuffled max -0.05. Full run submitted: sbatch --array=0-11 out_v6/job_probe_v6.sbatch -> 24066228 (shas probe6_inner 1c49096ba7f4, probe_v6 b607d40e4f70, aggregate b4b9cdecabf4).

## Amendment V7b (post-hoc, 2026-11-22): V7 rerun with the dropped lines replaced in their slots (slot-matched)

**Status / why.** Post-hoc, exploratory, no gate. The user approved it on 2026-11-22: 8 seeds, about 4 A100 h. Audit I1 (`tests/audit_v7_result_20261122.md`): V7 leaves the torus / box5 lines out and the text grows left, so dim's slot-1 share goes 0.50 → 0.748 over steps 7251..end (v7bread). The domain eval question is always in slot 1, so F109's "77% retained" may be inflated. **Hypothesis:** part of V7's retained dim advantage comes from that slot shift.
**Change.** The arm is `all6dropS@2e-3`, seeds 16..23. Each is the V7 job of its seed + `--drop_fill inplace`; out dir `out_v7b/`; Slurm names `llmadd_pt6v2v7b-*`; driver `posthoc_v7b_jobs.py` (8d9ff67c972a); manifest `POSTHOC_V7B.json`. The default `left` is V7, and it is kept out of the run JSON's args.
- From step 7324, each torus / box5 line is drawn as before. It is then replaced in its slot by a training-text filler of its token length: own stream `default_rng(7100 + seed)`, last token = its newline. Every kept line keeps its flag-off position and slot.
- Trainer: `out_v7b/code/tools/pretrain_range.py` bb0b55eb5e34 = 8da22c6ea208 + `--drop_fill` (`out_v7b/patch_trainer_v7b.py`, `trainer_v7b.diff`).
- `out_v7b/check_v7b.py` (12171d6c8db7) replaces check_v7.py and is log-and-flag only. The registered CHECK_PY and V7's `exit 5` no longer fail a job.
**Checks before submission.** CPU smoke `out_v7b/smoke/`, jobs 24140394 + 24140573: 55/55 PASS (`compare.txt`).
- Flag off: run JSON and stdout = the registered trainer's. Default and explicit `left`: = V7 (data_sha, run JSON, stdout). The real-config dumps at flag off and left equal the V7 smoke dumps byte for byte.
- Inplace, real s16 config, steps 7324..7329:
  - numbers / dim / cities / countries sites, slots and dslot counts = flag off's exactly, and no torus / box5 site remains;
  - every filler sits at the dropped line's position and length (154 = cycling rule), is the replayed 7100 + seed text and ends in a newline;
  - all other tokens = flag off's, and the left text is unchanged;
  - token counts per type = V7's;
  - batches 0..7323 and the four existing RNG states are unchanged.
- Other checks pass: resume (curve = uninterrupted), the guard (inplace without a drop exits 1), dry_inputs 63 lines b71458d3a12d, and the step-0 assert / log lines. check_v7b passes on the tiny inplace run and flags the V7 run.
- `out_v7b/v7bread.py` (50bdd048bc40) was smoke-tested on V7 outputs: S(V7) = 0.765 [0.500, 0.960]. With `--test-alias` (V7b := V7), S_b = S and the slot effect is 0.
**Decision rule (verbatim, written before submission).** Report S_b with its CI and the paired slot effect; if the slot effect's CI includes 0 and S_b's CI includes 0.77, F109's 77% stands and its qualifier 2 is replaced by the measured slot effect; otherwise the paper's 77% is replaced by S_b. No check gates the read-out.
**Jobs (submitted 2026-11-23 01:12 UTC, A100 / H100, 3 h).** Canary `llmadd_pt6v2v7b-canary-s16` 24140718 (drop@320, 800 steps); mains `llmadd_pt6v2v7b-all6dropS-s16..s23` 24140719, 24140721, 24140723, 24140725, 24140727, 24140729, 24140730, 24140731. Read-out: `out_v7b/v7bread.py --json out_v7b/v7bread.json | tee out_v7b/v7bread.txt` after all 8 mains finish.

## Amendment V8 (post-hoc, 2026-11-23 UTC, user-approved): only-dim and none trained 2x / 4x longer
**Question.** Does a run without the joint priors catch up with all6@2e-3 at 1x (18,310 steps; 8-seed strict dim final3 77.3 [57.5, 92.5]) when it trains longer? At 1x, all6 leads only-dim by 25.0 pts (8-seed means), and only-dim loses 32.9 pts from window end to final3 (F105 补记二).
**Cells (8 runs, seeds 16 / 17).** only-dim@2e-3 and none@3.2e-2 (= N_m, the area-best none lr for dims, tests/posthoc_speed_20261120/speed_out.txt) at x2 (`--tokens 599982080` = 36620 steps) and x4 (`--tokens 1199964160` = 73240 steps). only-dim keeps its window at absolute steps (`--window 0.025,0.2` / `0.0125,0.1`: prior on 916..7323, events [[916,1],[7324,0]], = 1x); none has no prior (events []). The cosine LR and the warmup (steps // 100 = 366 / 732 vs 183) stretch with the run, as intended.
**Jobs as registered otherwise.** Trainer a398fccbfe93 unchanged, `--ckpt_every 500`, eval every 250, `--requeue`, A100 / H100 (the registered s16 / s17 ran on B200; the device is a flag in the read), 4 h (x2) / 7 h (x4). Rendered by `posthoc_v8_jobs.py` (7aa1516ba1de) into `out_v8/job_*.sbatch` + `POSTHOC_V8.json`; out dirs `out_v8/<arm>-x<k>-s<seed>/`; every post-hoc check (CHECK_PY) is log-and-flag only, as in V7b.
**Checks before submission.** CPU dry (cpu_short jobs 24152103, 24153545): dry_inputs gives 63 lines b71458d3a12d for all four configs (= the in-job assert), with dim the only prior for only-dim and none for none. The only-dim x2 / x4 step-0 run JSONs confirm steps, window, events and lr. The none x2 / x4 step-0 checks run in 24153545, and all 8 GPU jobs wait on it (afterok; it exits 1 on V8 DRY FAIL). `out_v8/v8read.py` (b623468c678f) was smoke-tested on the 1x runs, both with the arms absent and with `--test-alias`.
**Reading rule (verbatim).** Report each arm's final3 strict dim at ×2 / ×4 against all6 1×; 'catches up' = mean ≥ all6 1× mean with both seeds within all6's 8-seed range; no gate.
**Jobs (submitted 2026-11-23 04:14 UTC, dependency afterok:24153545).** `llmadd_pt6v2v8-onlydim-x2-s16` 24153622, `-onlydim-x4-s16` 24153624, `-onlydim-x2-s17` 24153625, `-onlydim-x4-s17` 24153626, `-none-x2-s16` 24153627, `-none-x4-s16` 24153629, `-none-x2-s17` 24153630, `-none-x4-s17` 24153632. Read-out after all 8 finish: `cd out_v8 && python -B v8read.py --json v8read.json | tee v8read.txt`.

## Amendment V9 (post-hoc, 2026-11-25 UTC, user-approved)
**Approval.** User, 2026-11-25 (relayed by the main session): "可以的，我觉得都可以跑 可以多用一些gpu" ("fine, run all of it, use more GPUs"); B200 approved. Exploratory, no gate.
**Question.** At lr 2e-3 the joint lift on dim is h6 = all6 − only-dim = +13.5 strict-dim area (8/8). Removing the torus and box5 priors (all4) leaves h4 = +1.5. But all4 also has fewer priors and less total prior gradient: the median grad_prior over prior-on evals, averaged over seeds, is 0.095 for all4 and 0.143 for all6 (reproduced by `out_v9/v9read.py`). V9 asks whether the lift comes from the torus / box5 geometry or from the prior dose.
**Cells.** all6-shufTB@2e-3, seeds 16–23 (8 runs). This is the registered all6@2e-3 run plus `--prior_shuf torus,box5`; data, domains, the six priors, λ 1 each, window 0.05–0.4, 18,310 steps and trainer a398fccbfe93 are all unchanged. The trainer's own mechanism replaces each shuffled target with F[q]. q is far_perm on the domain's own stream; it is drawn in every arm and logged as shuf_cka_to_true, so the data RNG is unchanged. The CKA to the true target is below 0.15. The target Gram is P G Pᵀ (same spectrum and Frobenius norm), so the dose is matched.
**Comparators.** all6@2e-3 and only-dim@2e-3 in `out/`, and all4@2e-3 in `out_v4/`, all on the same seeds.
**Jobs.**
* Rendered by `posthoc_v9_jobs.py` into `out_v9/job_*.sbatch` and `POSTHOC_V9.json`. Each job is the registered job's A100 render, with byte-identity of the registered render asserted for every seed, plus counted edits:
  * ARGS gain `--prior_shuf torus,box5`;
  * out dir `out_v9/pt6v2-all6shufTB-lr2e-3-s<seed>`;
  * the dry spec gives torus / box5 shuf true;
  * intent: `prior_shuf` and `shuf`;
  * CHECK_PY: `dw` expects shuf for torus / box5, and the λ ratio check skips them;
  * CHECK_PY is log-and-flag only;
  * V9 early run guard;
  * A100 / H100, 03:00:00, `--open-mode=append`.
* Each primary has a B200 race copy (`-b200r`):
  * <slurm-account>, `--comment="preemption=yes"`, no qos, `--constraint=b200`, 12:00:00;
  * it shares the primary's out dir, and the run guard leaves one trainer per run.
**Checks before submission.** These are CPU dry runs.
* 24530352 (rows):
  * dry_inputs for all 8 seeds gives the rows sha each job asserts (unchanged by the flag);
  * the domains line has six priors, λ 1, and shuf on torus / box5 only.
* 24530353 (step0):
  * the s16 job's ARGS at step 0 log `prior_shuf`, λ, prior and shuf for all six domains;
  * with window 0,0.4, the shuffled and unshuffled runs are compared on the same batch at step 0, checking that the torus / box5 geo and prior-gradient terms differ and that the post-update parameters differ;
  * the trainer's Gram functions are checked to be permutation-equivariant.
* All 16 GPU jobs wait on 24530353 (afterok; it exits 1 on FAIL).
* `out_v9/v9read.py` reproduces the known answer with `--test-alias`: h6 +13.5 [+7.4, +19.6] 8/8, h4 +1.5 [−4.8, +7.8] 5/8, final3 all6 77.3 / all4 55.3 / only 52.3, dose 0.1428 / 0.0953.
**Reading rule (verbatim).** Exploratory, no gate. Lift from the torus / box5 geometry if h_shuf (all6-shufTB − only-dim, dim strict area) is close to h4 and all6 − all6-shufTB > 0 in most seeds; lift from the prior dose if h_shuf is close to h6.
**Expected flags (not failures).**
* CHECK_PY "item-state CKA to the target rises" may fail for torus / box5, because the target is shuffled.
* The device differs from the registered run: s16–s22 ran on B200, s23 on A100.
**Jobs (submitted 2026-11-25 ~07:50 UTC, dependency afterok:24530353).**
* Primary `llmadd_pt6v2v9-shufTB-s<seed>`: s16 24530845, s17 24530847, s18 24530853, s19 24530879, s20 24530881, s21 24530901, s22 24530912, s23 24530924.
* B200 race `-b200r`: s16 24530846, s17 24530852, s18 24530858, s19 24530880, s20 24530900, s21 24530902, s22 24530913, s23 24530933.
**Read-out.** `cd out_v9 && python -B v9read.py --json v9read.json | tee v9read.txt`.
