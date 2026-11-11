# ptdom2 — shuffled-target controls of F100's four PASS domains, and a re-screen of six near-miss domains on fresh seeds (53M, 2026-11-08)

Status: **BUILT, NOT SUBMITTED** except ONE GPU pilot job (section "GPU pilot"; not part of any result). Written before any
ptdom2 GPU result existed. **Amendment A1 (end of this file; the coordinator's decisions of 2026-11-08 on "Pending decisions",
before any ptdom2 result) holds where it differs from the text above; Amendment A2 (after A1, the fresh build audit
tests/audit_build.md, before any ptdom2 result) holds where it differs from both; Amendment A3 (after the A2 audit
tests/audit_a2.md, before any ptdom2 result) holds where it differs from all three.** Every ptdom file is frozen and read only (runs/ptdom/*, make_jobs_ptdom.py 6061b299129c,
readout_ptdom.py 1e2f17522e47); the re-used ptdom runs are frozen by sha in runs/ptdom2/reused_ptdom.json.

## Intent
F100 (FINDINGS, ptdom) left two gaps that its results audit (runs/ptdom/tests/audit_results.md, items 3 and 6) named:
1. **No dose-matched shuffled-target control**: a PASS says "the prior with this target helps", never "the geometry helps".
   Part S runs, for each of F100's four PASS domains (countries, people, cities, months), the same prior with the item ->
   coordinate correspondence permuted (`--prior_shuf <d>`), everything else identical, and pairs it with ptdom's own
   true-target runs.
2. **Near-miss domains**: several NOT ADVANCE domains had positive same-lr gains (stars, compass, scalar_adj, prefixes,
   landmarks; A5.3: a NOT ADVANCE is not evidence of no effect), and ranks was NOT SHOWN by one seed. Part R re-selects the lr
   of six of them with a wider prior grid (+4e-3, conditionally +8e-3) and confirms on eight FRESH seeds (8-15) that no
   selection step reads, with the shuffled control alongside.

Design: the user's decision of 2026-11-08, relayed by the coordinator, implemented as given; every point where the build had
to interpret or saw a problem is in "Pending decisions (待协调者定)" at the end, not changed silently.

## Design
Everything that is not stated here is ptdom's (AUDIT.md A3-A6.1 of runs/ptdom): 53M (52.7M params, d384 x 6), 3e8 tokens =
18,310 steps of 32 x 512, 75 evals (every 250 + the last), the rb6 data stack per batch (FineWeb-Edu + one numbers line (2%,
first after the text) + K = 2 domain lines of the batch's six domains), the numbers prior off, the prior = mean over layers 2,
4 of 1 - linear CKA (states at the last token of each item name vs the target Gram) with lambda 1, window (0.05, 0.4) = steps
916..7323, site ctx, lam_rule each; one batch's `--domains` per run (A: months, stars, landmarks, languages, colours, cities;
B1: countries, animals, country_size, animal_size, people, units; B2: zodiac, compass, planets, elements, prefixes, ranks; B3:
chinese_zodiac, kinship, life_stages, geo_periods, greek, scalar_adj); the trainer Rev G (commit 11f61b9, pretrain_range.py
d87a1502cee8); B200 only.

### Arms
- `none`, `only-<d>` (ptdom's), and new **`only-<d>-shuf`** = `only-<d>` + `--prior_shuf <d>`. The shuffled arm's prior uses
  the trainer's far permutation of d's target rows (pretrain_range.py:1091-1097: `DTAB_ACT[d] = F[qperm]`, qperm from d's own
  registered stream: SeedSequence(4300 + seed).spawn(54)[3 i + 1] for the Rev F domains, SeedSequence(4200 + seed).spawn(16)[6 +
  i] for ANCHOR, SeedSequence(4100 + seed).spawn(16)[6 + i] for months; the first of 3,000 permutations with CKA to the true
  target < 0.15, asserted at start for a shuffled domain). It permutes ONLY which item gets which coordinate: the same items,
  sites, window, lambda, the 4-site / 3-item minimum, lr and data. The eval-prompt item-state CKA (`cka_<d>`) is always on the
  TRUE target; `geo_<d>` is the prior's own value (the shuffled target in a shuffled arm). Verified on CPU (tests/shuf_trace.out,
  below) and by the trainer code read (pretrain_range.py:2208, 2224, 2244, 2366: the training term and its weighted gradient
  use DTAB_ACT; dom_grads' "true" term and evaluate() use DTAB).

### Part S (shuffled controls of F100's PASS domains)
- Cells: `only-<d>-shuf@2e-3`, seeds 0-7, for countries, people (batch B1) and cities, months (batch A) = 32 runs. 2e-3 is
  F100's prior pick of all four.
- Paired with ptdom's `only-<d>@2e-3` seeds 0-7 (same batch data, seed, snapshot; read only, sha-frozen).

### Part R (re-screen of six near-miss domains)
Domains: stars, landmarks (A); compass, prefixes, ranks (B2); scalar_adj (B3).
- **R1 (lr selection; seeds 0-3)**: new cells `only-<d>@4e-3` seeds 0-3 (24 runs). Prior grid = ptdom's lrs + 4e-3:
  stars / landmarks / compass / prefixes / scalar_adj {1e-3, 2e-3, 4e-3}, ranks {5e-4, 1e-3, 2e-3, 4e-3}. None grid = the
  batch's ptdom lrs: A {1e-3, 2e-3, 4e-3, 8e-3}, B2 {1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2}, B3 {2e-3, 4e-3, 8e-3} (select_ptdom2.py
  asserts that exactly these exist on disk). Selection per unit = F87 rule 1 'top2' exactly as ptdom (readout_ptdom.select_unit
  -> readout_rules.best_lr: the argmax of the mean area over seeds 0-3; the top two within 2 points -> the lower mean bpb,
  offline only when every run of both has it, else in-training; then the lower lr). The A2.1 gate first: per prior (arm, lr)
  cell the median over its runs of the run's on-window median a / (a + b) < 0.5 (or an unverifiable run) -> the cell cannot be
  the pick; every cell failing -> GATE.
- **R1x (conditional extensions, once; seeds 0-3)**: prior `only-<d>@8e-3` where 4e-3 is the prior's area argmax; none one
  step (x2) above the batch's top ptdom lr (A 1.6e-2, B2 3.2e-2, B3 1.6e-2) where an R domain's none area argmax is that top
  lr, once per batch. After R1x every unit is re-selected over its lrs + the extensions (the batch's none extension is in the
  none grid of every R domain of that batch: "none grid = every lr of the batch", Pending 3).
- **NOT FINAL** (no verdict, not in the Holm family, no R2 runs): after the unit's allowed step (taken or not due), its final
  pick or area argmax is the OUTERMOST (highest) lr it ran (Pending 2 for a tie-break pick).
- **R2 (confirmation; seeds 8-15, never read by a selection)**: per FINAL domain d, `only-<d>@P`, `only-<d>-shuf@P`,
  `none@N` at seeds 8-15; a none cell is shared by the domains of a batch with the same N (run once). 16 runs per domain + 8 per
  distinct (batch, N): 120-144 runs when all six are FINAL.

### Seeds and names
- Seeds 0-3: ptdom's selection seeds (R1 / R1x cells); 0-7: Part S (ptdom's verdict seeds of the paired true arm); 8-15: R2
  only (fresh; CPU references runs/ptdom2/ptdom2_refs_<A|B2|B3>.json, seeds 8-15 only; ptdom's references serve seeds 0-7).
- Run / out-dir name `ptdom2-<batch>-<arm>-lr<lr>-s<seed>[-r<k>]` in runs/ptdom2/out (no ptdom name can collide: ptdom's start
  `ptdom-`; the generator also refuses a name that exists in another batch's out dir); trainer tag `_<the same without -s>`.
- Slurm: 8 singleton B200 chains `llmadd_ptdom2-b200-slot0..7` shared by every part (slot = submission index mod 8);
  logs llmadd_torch/logs/llmadd_<run>_<jobid>.out.

## Metrics (unique definitions; = ptdom's)
- **Primary: area** = 100 x the mean over the 75 evals of the held-out read-out (`readout_ptdom.READ`): comparison domains
  (all six R domains, countries, people, cities) `<d>_readout` = the 2-way choice over every held-out question in both orders
  (chance 50; Tier L: the target carries the answers); months `months_readout_kspan` = the full-span 12-way choice on tier S
  (chance 8.3). Read from the run JSON's metrics (re-derived from the per-question hits by readout_ptdom.run_subsets, which must
  agree, in the real read-out).
- final3 = the mean of the last 3 evals; hard-subset final3 = `<d>_readout_hard` final3 (the questions both heuristics get wrong
  or tie; comparisons only).
- Shortcut baseline (descriptive): readout_ptdom.split_info's training-frequency rule on the pooled hard subset; FLOOR line =
  max(chance, answer-frequency line, shorter-option line, the shortcut baseline on every held-out question) + 5.

## Pre-registered verdicts (readout_ptdom2.py; statistics = readout_ptdom's own functions)
Four conditions on 8 paired per-seed values g (seed order), each required:
(a) the paired bootstrap 90% CI lower bound > 0 (readout_ptdom.boot_ci: 10,000 resamples, default_rng(0), percentile);
(b) both halves' means > 0;
(c) the one-sided one-sample t p (H1: mean > 0, df 7; readout_ptdom.sf_test), **Holm-adjusted** within the family, < 0.10;
(d) at least 6 of 8 values > 0.

- **Part R** (seeds 8-15; halves = seeds 8-11 / 12-15; Holm family = the R domains with a verdict, i.e. status FINAL after
  R1 / R1x, at most 6):
  - **PASS** iff (a)-(d) on g_s = area(only-d@P, s) - area(none@N, s); else **NOT SHOWN**.
  - **GEOMETRY-SPECIFIC** iff (a)-(d) on g'_s = area(only-d@P, s) - area(only-d-shuf@P, s), Holm over the same family
    (adjusted separately); else **NOT SHOWN**. Computed and reported for every FINAL domain, whatever its PASS verdict.
  - NOT FINAL / GATE domains: that word, no verdict.
- **Part S** (seeds 0-7; halves 0-3 / 4-7; Holm family = the four S domains): **GEOMETRY-SPECIFIC** iff (a)-(d) on g'_s =
  area(only-d@2e-3, s) [ptdom] - area(only-d-shuf@2e-3, s) [ptdom2]; else **NOT SHOWN**.
- **Refusal**: a verdict is refused (exit) while any needed run is missing or invalid, the data of a seed differ between the
  arms (readout_ptdom.integrity: data_sha at every eval, step-0 lm), a re-used ptdom run differs from the manifest, a
  comparison sub-read-out disagrees with the logged series, or a picks file differs from its re-derivation. `--allow_partial`
  prints PROVISIONAL (no verdict word) instead.
- **Printed beside every verdict (descriptive, never a condition)**: final3 gains and hard-subset final3 gains of both
  contrasts; the 2e-3 same-lr gain on ptdom's seeds 0-3 (only-d@2e-3 - none@2e-3; and on seeds 8-15 when P = N); FLOOR /
  CEILING (ptdom's definitions; when the FLOOR line > 100 the note "the training-frequency shortcut is above both arms; the tag
  holds unconditionally", audit_results item 5); the shortcut baseline against EACH arm separately (ptdom's condition (iii) is
  not a rule here, audit_results item 2); the aux gate per cell (the selection cell on seeds 0-3, the pick and the shuffled cell
  on seeds 8-15); the lr-edge notes (Pending 1); the shared-site note for stars / landmarks (>= 20% shared site tokens: a
  NOT SHOWN reads "with this site").
- What the words mean: PASS = the prior helps the held-out read-out vs none on fresh seeds (Tier L). GEOMETRY-SPECIFIC =
  the true target beats the dose-matched shuffled target; NOT SHOWN = no reliable difference at 8 seeds under this rule, never
  "no effect". The tied-embedding caveat (a single-token item's site is its read-out candidate: compass 100%, scalar_adj 94%,
  months 12/12) applies to both arms equally, so a geometry contrast is not affected by it the way a vs-none contrast is.

## Validity (inherited)
- ptdom's in-job checks (CHECK_PY, byte for byte, for none and true-target arms) and validity rules: A5.1 (a run failing only
  the GPU-numerics double-path tolerances and / or the CKA-rise bar is VALID data, flagged), A6 / A6.1 (a geo overshoot at a
  perfect fit, verified from the run JSON, is valid data, flagged), A2.3 (a finished run is never re-run; a re-run is a new
  name `-r<k>`). load2 (readout_ptdom2.py) = readout_ptdom.load generalised to the ptdom2 names: identical records on all 384
  ptdom cells (tests/real_known.out 1), plus an identity check (args.seed / lr / tag / prior_domains / prior_shuf and
  domains.<d>.prior / shuf = the directory name).
- Shuffled arms run **CHECK_PY_SHUF** = ptdom's CHECK_PY with exact, counted edits (make_jobs_ptdom2._check_py_shuf; the full
  diff is printed in tests/render_diff.out):
  1. domains.<d>.shuf must be true exactly for the shuffled domain (ptdom: always false);
  2. new: `--prior_shuf` of the job's ARGS = the intent's shuf, a subset of the active priors, and domains.<d>.shuf_cka_to_true
     < 0.15;
  3. the lambda ratio gnorm_prior / gnorm_true is checked for true-target priors only: in a shuffled arm gnorm_true is the TRUE
     target's term, so the ratio is not lambda (it would fail every shuffled run); new: the shuffled prior's weighted gradient
     finite > 0 at every on-eval where logged; lambda stays tied by the ARGS / lam_domains check and the on-window
     decomposition double path (Pending 6; the CPU test shows the exact 2x scaling at lambda 2);
  4. the CKA-rise sanity bar (same name prefix, so A5.1's flagged-valid rule applies unchanged) is judged on the SHUFFLED
     target = 1 - geo_<d> on the training batch, from its first logged value to the window's last eval; the true-target eval
     CKA rise is logged beside it (info).
  The A6 / A5.1 name guard (readout_ptdom2.guard2) asserts in every ptdom2 job file that the CHECK_PY is ptdom's or CHECK_PY_SHUF
  exactly (by the job's ARGS), plus readout_ptdom.a6_guard's conditions.
- A6 / A6.1 for a shuffled arm: applied **literally** (readout_ptdom.geo_a6 verbatim, `A6_SHUF = "literal"`): its perfect-fit
  condition reads cka_<active>, which in a shuffled arm is the eval CKA to the TRUE target, so a shuffled run with a geo
  overshoot stays INVALID and its domain's verdict is refused (Pending 4; an alternative is implemented, not active).

## Code and files (sha256[:12] at the build; the generator asserts make_jobs_ptdom.py's, the read-out readout_ptdom.py's)
| file | sha | role |
|---|---|---|
| llmadd_torch/make_jobs_ptdom2.py | 70142319baba (render_diff ran on 1bd9a57a590d; later edits: the --stage alias of --part, a submitted pilot's bodies never rewritten; every job file re-rendered byte-identical) | generator: renders through make_jobs_ptdom.render (read-only import, 6061b299129c asserted) |
| runs/ptdom2/readout_ptdom2.py | a286e1c9e038 | load2, the R1 decision (decide_R), the R / S verdicts, the self-test (readout_ptdom.py 1e2f17522e47 asserted) |
| runs/ptdom2/select_ptdom2.py | 4f39b4676c09 | R1 / R1x selection CLI, --verify |
| runs/ptdom2/submit_ptdom2.sh | 34e1bd984979 | submission helper (checks; --go submits) |
| runs/ptdom2/pilot/pilot_check.py | 3e74e4969d9c | the pilot's pass rules |
| CHECK_PY (ptdom's) / CHECK_PY_SHUF | 12dff9487798 / 9c6dd473b416 (sha256[:12] of the heredoc text) | in-job checks |
| runs/ptdom2/code/ | = runs/ptdom/code/tools byte for byte (every file), git archive of 11f61b9 | the trainer snapshot |
| runs/ptdom2/ptdom2_refs_A / _B2 / _B3.json | 59f66875173d / 4a21f51e5594 / a66050805e0e (chmod 444) | CPU references, seeds 8-15 |
| runs/ptdom2/reused_ptdom.json | cf8781e2778e (sha256[:16] cf8781e2778ebda3) | the 132 re-used ptdom cells' run JSON / check.json shas |

The job body: make_jobs_ptdom.render is called with a few of its module names re-pointed for the duration of one call (R,
tag_of, job_name, slurm_name, args_of, intent, want_flags, dry_block, arms, saves, STAGES, SEL / CONF (seeds 8-15), SLOT0 /
SLOTS, CHECK_PY), then exact, counted text edits of the comment lines naming runs/ptdom and of the kill-switch line
(HOLD_<part>). The rendered texts equal their ptdom twins after the registered substitutions (tests/render_diff.out).
Selection cells (R1, R1x) use ptdom's stage-1 body (no saved model, no offline LM); confirmation cells (S, R2) ptdom's
stage-2 body (`--save_model`, offline_lm) (Pending 9).

### Re-used ptdom runs (read only; runs/ptdom2/reused_ptdom.json freezes each run JSON's and check.json's sha256[:16])
- R1 selection: only-<d>@{its PRIOR_LRS0} and none@{the batch's NONE_LRS0} at seeds 0-3 (stars, landmarks, compass, prefixes,
  scalar_adj: 1e-3 / 2e-3; ranks 5e-4 / 1e-3 / 2e-3; none A 1e-3..8e-3, B2 1e-3..1.6e-2, B3 2e-3..8e-3).
- Part S: only-countries / only-people (B1), only-cities / only-months (A) @2e-3, seeds 0-7.
- 132 cells, all valid under the A6.1 loader; select_ptdom2.py and the read-out refuse if any differs from the manifest.

## Cost, wall clock, disk (make_jobs_ptdom2.py --plan)
Per-run B200 hours = ptdom's measured median job wall time (dry inputs to done; tests/walltime.out): A 0.46-0.47, B1 0.34-0.35,
B2 0.38-0.39, B3 0.36; planned A 0.47, B1 0.36, B2 0.40, B3 0.37 (ptdom's own plan basis 0.6 in brackets).
| part | runs | B200-h |
|---|---|---|
| S | 32 | 13.3 (19.2) |
| R1 | 24 | 10.0 (14.4) |
| R1x | 0-36 | 0-15.0; ptdom's data alone already ask none@1.6e-2 for batch A (stars' / landmarks' none argmax is 8e-3: 4 runs, 1.9 h; tests/real_known.out) |
| R2 | 120-144 with six FINAL domains (16 fewer per NOT FINAL / GATE domain, minus its unshared none cell) | 50.1-60.2 |
| total | min 56 (23 h) / nominal ~188 (78 h) / max 236 (99 h) | |
Wall clock with 8 chains running: S + R1 ~3 h, R1x <= 2 h, R2 6-8 h, plus queue waits (ptdom: 384 runs in ~3.5 days on 16
chains). Disk: saved models 0.21 GB per S / R2 run = 7 + 25-30 GB; R1 / R1x none; /scratch quota 4.79 / 5.00 TB (95.8%) at the
build; submit_ptdom2.sh refuses at >= 97%.

## Submission (submit_ptdom2.sh; never --go in the build)
1. `bash runs/ptdom2/submit_ptdom2.sh --part S --go` and `... --part R1 --go` (independent; may run together on the 8 chains).
2. After every R1 job has finished with check.json: `python runs/ptdom2/select_ptdom2.py --after R1` -> picks_R_after1.json
   (written once, chmod 444): stage R1x (extensions asked) or R2 (final). A GATE domain appends to runs/ptdom2/HOLD_R2 and
   exits 2 (the user decides).
3. If asked: `submit_ptdom2.sh --part R1x --go`; then `select_ptdom2.py --after R1x` -> picks_R_after2.json (final).
4. `submit_ptdom2.sh --part R2 --go` (it takes picks_R_after2.json if present, else picks_R_after1.json).
5. `PYTHONPATH=llmadd_torch/pylib python runs/ptdom2/readout_ptdom2.py --json runs/ptdom2/readout.json > runs/ptdom2/readout.txt`
   (Part S can be read as soon as S is complete: `--part S`).
submit_ptdom2.sh refuses unless: no HOLD / HOLD_<part>; /scratch < 97%; the snapshot and the frozen shas verify; the read-out
self-test passes (dated log in tests/); the GPU pilot passed (pilot/pilot_check.out "PILOT PASS"); `--check` renders the part
(bash -n, every render assertion); no job of the part was submitted before; R1x / R2: the picks file is read-only and
`select_ptdom2.py --verify` re-derives it. **Kill switches**: `touch runs/ptdom2/HOLD` (every part) or `runs/ptdom2/HOLD_<S|R1|R1x|R2>`;
a pending or requeued job exits 0 at start; the generator refuses --submit while one exists. Re-runs (a job that exited before
its run finished): `make_jobs_ptdom2.py --part X [--picks f] --rerun --print`, then `--rerun --submit` (ptdom's A2.3 rules).

## GPU pilot (one job; not part of any result; pass rules registered here before it ran)
- Job `llmadd_ptdom2-pilot` (runs/ptdom2/pilot/job_ptdom2-pilot.sbatch; B200, account 520 priority, preemption=yes, 1 h): two
  bodies run in turn with bash, each = the ptdom2 job body of its cell (the real schedule, `--stop_after 2500`: evals 0..2500,
  on-window evals 1000..2500; save_model off): `only-compass-shuf@2e-3` s0 (B2) and `only-countries-shuf@2e-3` s0 (B1); then
  pilot_check.py. A truncated run fails by design exactly three CHECK_PY checks ('finished: 75 evals', 'prior_events',
  'prior_on_frac').
- Pass rules (pilot_check.py): P1 the failed checks are exactly those three; P2 (a) 1 - geo_<d> (fit to the SHUFFLED target)
  rises by >= 0.1 from step 0 to the last on-eval, (b) the eval CKA to the TRUE target at step 2500 <= ptdom's none@2e-3 s0's +
  0.10, (c) ptdom's only-<d>@2e-3 s0 (true target) is >= the pilot + 0.2 there; P3 data_sha = ptdom's twin at every eval and
  |lm - twin| <= 0.05 before the window; P4 sec / step <= 1.5 x the twin's (memory, utilisation, projected hours printed);
  P5 (info) the LM share a / (a + b) at the on-evals.
- Submitted: see "Pilot submission" at the end. Read it with:
  `cat runs/ptdom2/pilot/pilot_check.out; tail -n 40 llmadd_torch/logs/llmadd_ptdom2-pilot_<jobid>.out;`
  `sacct -j <jobid> -o JobID,State,Elapsed,NodeList`; or by hand `PYTHONPATH=llmadd_torch/pylib python runs/ptdom2/pilot/pilot_check.py
  ptdom2-pilot-B2-only-compass-shuf-lr2e-3-s0 ptdom2-pilot-B1-only-countries-shuf-lr2e-3-s0`.

## Tests (runs/ptdom2/tests/; CPU: cpu_short via cpu_run.sh, or this node at nice 19; every log is the script's own output)
| test | script / log | result |
|---|---|---|
| --prior_shuf on all 10 shuffled domains (instrumented trainer, tiny CPU model, true vs shuffled arm, same seed / data; + --lam 2 for compass, countries) | shuf_trace.py, run_shuf_trace.sh / shuf_trace.out | **SHUF TRACE PASS 92/92**. T1: DTAB_ACT = DTAB[q] exactly, q re-derived independently on the registered stream (moves 7/8 .. 30/31 items), logged shuf_cka_to_true = the re-derivation (0.047-0.146 at seed 0), DTAB_ACT is DTAB in the true arm. T2: every training-forward prior call is d's only, on DTAB_ACT (rows = table[ids]), inside the window only (17 / 17 calls), at the same steps with the same site item ids in both arms; no other domain trained / logged / shuffled. T3: lambda 1 in both; at the first on-step (identical states) lm, grad_lm, gnorm_true equal, the true arm's gnorm_prior = gnorm_true; --lam 2: gnorm_prior ratio 2.0000000. T4: at every on-window eval the logged geo_<d> = the trainer's bf16 arithmetic on the recorded states with the independently derived OWN target (<= 1e-6), fp64 within 0.02, the other target differs at >= 1 eval (landmarks: 4 of 5; at step 8 the two coincide in bf16). T5: data_sha equal at every eval; every eval-prompt CKA uses the TRUE target |
| CPU references, seeds 8-15 (A, B2, B3) + shuffled-arm arm independence | refs_ptdom2.py, run_refs_ptdom2.sh / run_refs_ptdom2_<A,B1,B2,B3>.out, refs_B1_ptdomcmp.out, refs_B2_s8_rerun.out | TABLE A / B2 / B3 PASS (seeds 8-15; DRY rows, splits, table shas one value per seed over none and the shuffled arms; no DRY hash repeats over seeds 0-15). PTDOMCMP A / B1 / B2 PASS: the S / pilot shuffled arms at seed 0 (months, cities, countries, people, compass) = ptdom's references (DRY rows, splits, tables, lm0, data_sha). One CPU-noise event: B2 s8's step-0 lm of only-ranks-shuf differed from none's by 3.8e-6 in the cpu_short run (same data_sha / DRY rows); re-run here twice both arms are bit-identical -> the table takes none's value and accepts another arm within 1e-5 (the in-job tolerance is 1e-3). The B1 job's last step aborted on a script edit made while it ran (its cells had finished): its ptdomcmp was run by hand (refs_B1_ptdomcmp.out) |
| shuffled target far enough at every planned cell | refs_ptdom2.py shufcka / shufcka.out | SHUFCKA PASS: 82 planned shuffled cells (S seeds 0-7, R2 seeds 8-15, the pilot), shuf_cka_to_true max 0.136 (< 0.15; a value >= 0.15 would crash the job at start) |
| job bodies | render_diff.py / render_diff.out | **RENDER DIFF PASS 12/12**: make_jobs_ptdom.render (imported, unpatched) re-renders all 386 submitted ptdom job files byte for byte; every ptdom2 job (S 32, R1 24, R1x 36 and R2 144 from synthetic picks covering every extension lr, the 2 pilot bodies) = its ptdom twin after the registered substitutions, a shuffled arm differing only in CHECK_PY_SHUF and the dry spec; bash -n, the B200 slot-chain header, seeds, names / HOLD under ptdom2, --prior_shuf exactly on the shuffled arms; the raw diffs per arm kind and the full CHECK_PY_SHUF diff are printed |
| the jobs' input checks on real CPU dry logs | dryblock_test.py / dryblock_test.out | DRYBLOCK PASS 19/19: own cell exit 0 (S x 4, R2 x 7 incl. every shuffled R domain, the 2 pilot bodies); a shuffled job with the true arm's real GPU dry log, the true-arm job with a shuffled log, another seed, another batch, none's log -> exit 3 |
| CHECK_PY_SHUF on real (tiny CPU) run JSONs | check_py_shuf_test.py / check_py_shuf_test.out | CHECK_PY_SHUF TEST PASS 24/24 (compass, countries, scalar_adj, months): only the tiny / truncation checks fail; the shuffled-target, domains, lambda, args checks pass; tampers fail their check (the true run under the shuffled intent; the shuffled run under ptdom's CHECK_PY; --lam 2 in ARGS; shuf_cka_to_true 0.2; a flat geo -> the CKA-rise bar) |
| read-out / selection self-test (planted known answers) | readout_ptdom2.py --selftest / selftest_readout.out | **SELFTEST PASS 30/30**: Holm, sf_test = scipy, the four conditions each biting alone (halves, 5/8 wins, bootstrap <= 0), Holm over 6 applied inside the verdict (raw p 0.024 -> Holm 0.143 -> NOT SHOWN); R1: 4e-3 argmax -> 8e-3 asked; none's argmax at the top -> one extension per batch; NOT FINAL at the extension (prior and none); a missing cell refused; the gate excludes a failing cell and GATEs an all-failing arm; R2: +6 -> PASS + GEOMETRY-SPECIFIC; true = shuffled -> PASS + NOT SHOWN; zero -> NOT SHOWN; a negative half -> NOT SHOWN; a missing run -> refused / PROVISIONAL; a NOT FINAL domain leaves the family; S: planted +4 / 0 -> GEOMETRY-SPECIFIC / NOT SHOWN, sources checked, a missing run refused; loader identity / A5.1 / unfinished |
| the self-test bites | mutation_readout.py / mutation_readout.out | MUTATION READOUT PASS: 13 mutations (wins 5, no halves, no bootstrap, no Holm, geometry vs none, R on seeds 0-3, no NOT FINAL, no prior / none extension, no gate exclusion, S sign flipped, no refusal, no identity check) each fail the self-test; the control passes |
| real ptdom data, planted ptdom2 cells | real_known.py / real_known.out | **REAL KNOWN PASS 14/14**: load2 = readout_ptdom.load on all 384 ptdom cells; Part S reads exactly ptdom's only-<d>@2e-3 seeds 0-7 (independent json read), per-seed gains = the planted deltas (area, final3, hard final3), verdicts = planted (x 3 scenarios), integrity holds; R1 on real data: decide_R = an independent F87 rule 1 (picks, argmaxes, extensions). The pre-registered rule on ptdom's data alone asks R1x none@1.6e-2 for batch A (stars' and landmarks' none argmax 8e-3) |
| select -> picks -> generator (CLIs) | plumbing.py / plumbing.out | PLUMBING PASS 11/11: --after R1 writes picks_R_after1.json once (chmod 444) with the expected E; --verify agrees; --part R1x renders exactly E x seeds 0-3; --after R1x -> stage R2, --part R2 renders exactly C x seeds 8-15 (120); a tampered picks file -> --verify exit 1 and the generator refuses; a GATE domain -> HOLD path written, exit 2; nothing written in runs/ptdom2 |
| pilot check (negative / positive controls on real ptdom runs) | pilot_check_test.py / pilot_check_test.out | PILOT CHECK TEST PASS 3/3: a true-target prior disguised as the pilot fails P2b / P2c (its true-target CKA 0.999 vs none 0.683); a shuffled-looking run passes every rule; a fourth failed check fails P1 |
| re-used ptdom runs frozen | reused_manifest.py / reused_manifest.out | 132 cells, all valid; runs/ptdom2/reused_ptdom.json (chmod 444); select / read-out refuse on a mismatch (checked on the real data: no problem) |
| sbatch --test-only | make_jobs_ptdom2.py --test-only / sbatch_test_only.out | S x 2, R1 x 2, the pilot: accepted (b200_tandon_prem); nothing submitted by the test |
| submission script (print only) | submit_ptdom2.sh --part S (no --go), --part R2 / submit_print.out | S: quota 95.8%, plan, snapshot, self-test 30/30, the pilot NOTE (not passed yet: --go refuses), 32 jobs, none submitted before, the misalign B200 note; R2 without a picks file: REFUSED (rc 3). --go was never run |
| per-run wall time basis | walltime.py / walltime.out | ptdom's medians per batch and kind (the plan hours) |

## Audit status (experiment-audit, minimal mode for the build)
ii (asserts): the trainer's own (the far permutation's CKA < 0.15 asserted for a shuffled domain at start, the site / split /
content asserts, the CPU double paths), every job's input checks (DRY rows = the CPU reference; the dry block now checks
shuf true for a shuffled arm) and CHECK_PY / CHECK_PY_SHUF; the generator's render assertions (every flag, every other flag at
its default, seeds, slots), load2's identity check, the manifest, guard2. iii (known answers): the self-test, the shuffle trace,
the real-data known answers, the CHECK_PY_SHUF known answers, the mutation test. vii: every job's dry_inputs.txt (decoded
training / eval rows) as ptdom. i: the S pairing re-read independently from the run files (real_known); the logged geo vs an
independent fp64 recomputation (shuf_trace T4). **v (fresh adversarial review): NOT DONE in the build: for the coordinator to
launch (minimal mode: right after the pilot submission, in parallel with it).** iv: not applicable before results.

## Pilot submission
- 2026-11-08 ~21:24 EST: `make_jobs_ptdom2.py --pilot --submit` -> **job 22355337** (llmadd_ptdom2-pilot; runs/ptdom2/pilot/SUBMITTED_ptdom2-pilot).
  No other ptdom2 job was submitted; no job was cancelled. The pass rules above were written before it ran.
- Read it: `sacct -j 22355337 -o JobID,State,Elapsed,NodeList`; `tail -n 40 llmadd_torch/logs/llmadd_ptdom2-pilot_22355337.out`;
  `cat runs/ptdom2/pilot/pilot_check.out` ('PILOT PASS n/n' is what submit_ptdom2.sh --go requires); the bodies' logs
  runs/ptdom2/pilot/body_<name>.log and check.json / check.txt in runs/ptdom2/pilot/<name>/.

## Pending decisions (待协调者定; nothing here changes the design without a decision)
1. **None's lower edge and a prior pick below none's range are not covered** (ptdom's fixB case). The design extends none only
   upward and calls NOT FINAL only at the top. On ptdom's own data (tests/real_known.out): prefixes' none argmax is 1e-3 = B2's
   LOWEST none lr; scalar_adj's prior pick is 1e-3 (by the bpb tie-break; argmax 2e-3) while B3's none grid starts at 2e-3;
   ranks' prior grid reaches 5e-4 < B2's 1e-3. Both directions favour a false PASS (none's best not found at the prior's lr).
   Implemented: descriptive notes only ('NONE LOWER EDGE', 'PRIOR PICK BELOW NONE'S RANGE'). Options: none one step lower
   (B2 5e-4, B3 1e-3) in R1x where that happens (+4 runs each), or NOT FINAL.
2. **Extension trigger vs NOT FINAL for a tie-break pick**: the design triggers the 8e-3 / none extensions on the AREA ARGMAX;
   when the pick (not the argmax) sits at the outermost lr by the 2-point bpb tie-break, no extension runs and the
   implementation calls the domain NOT FINAL (pick or argmax at the outermost lr). Alternative: trigger on pick or argmax.
3. **The batch's none extension joins every R domain's none grid in that batch** (reading "none grid = every lr of the batch",
   as the design lists ptdom's extensions); ptdom's A2 rule added an extension only to the unit that asked. On ptdom's data A's
   1.6e-2 is asked by stars / landmarks; with the shared reading both use it.
4. **A6 / A6.1 for a shuffled arm** (implemented literally): its perfect-fit test reads cka_<active>, which in a shuffled arm is
   the eval CKA to the TRUE target, so a shuffled run with a bf16 geo overshoot (ptdom had 3 such runs on 8-12-item domains;
   compass, ranks, prefixes are 8-11 items) stays INVALID, and as a finished run is never re-run, that domain's verdict is
   refused. An alternative is implemented but OFF (`readout_ptdom2.A6_SHUF = "adjacent"`: the perfect fit verified on the two
   adjacent on-evals' geo <= 0.005, the floor and every other A6 condition unchanged).
5. **The CKA-rise sanity bar of a shuffled arm** is judged on the SHUFFLED target (1 - geo_<d> on the training batch); A5.1's
   flagged-valid rule unchanged. Confirm.
6. **Lambda of a shuffled arm** is verified more weakly in-job (the trainer logs no unweighted shuffled-term gradient): ARGS ->
   lam_domains / domains.<d>.lam, and the decomposition double path; the CPU trace shows the exact 2x scaling at --lam 2.
7. **The aux gate on seeds 8-15** (the pick and the shuffled cell) is descriptive, as the design says. ptdom (A2.1 (ii), A4.5)
   turned a failing pick into "AUX GATE FAILED at the pick" instead of a verdict. Keep descriptive or adopt ptdom's rule?
8. **A GATE domain writes runs/ptdom2/HOLD_R2** (ptdom's habit: the user decides) and so blocks every R2 job; alternative:
   drop that domain and continue.
9. **Saved models in S / R2** (ptdom's stage-2 body: ~32 GB at a /scratch quota of 95.8%) vs ptdom's stage-1 body (no model,
   no offline LM; nothing in the verdicts needs them).
10. **R1x's none extension can be decided now** from ptdom's data alone (A none@1.6e-2, 4 runs): submitting it with R1 would
    save one queue round (the design orders R1 -> R1x).
11. **Part S's first half (seeds 0-3) are F100's selection seeds of the TRUE arm** (2e-3 was picked on them): a winner's curse in
    favour of GEOMETRY-SPECIFIC on that half; seeds 4-7 are unselected and the halves condition needs both. Options: keep (the
    design), or report seeds 4-7 alone beside it (descriptive), or add seeds 8-15 for both arms (+32 runs).
12. **B200 sharing**: runs/ptdom/NOTIFY_misalign_when_b200_free.txt records the promise to the misalign session; tell it
    before S / R1 occupy B200. 8 chains (ptdom: 16) is a constant (make_jobs_ptdom2.N_SLOTS).
13. **The 2e-3 same-lr gain** printed for an R domain comes from ptdom's seeds 0-3 (selection seeds); on seeds 8-15 a same-lr
    pair exists only when P = N (also printed then).
14. Shared none cells in R2 (one none@N per batch and lr) make the domains of a batch correlated; Holm stays valid.
15. **v (fresh adversarial review)**: not run by this build (minimal mode): for the coordinator, in parallel with the pilot.


## Amendment A1 (coordinator, 2026-11-08, before any ptdom2 GPU result; the pilot 22355337 was still pending, no result read)
The coordinator's decisions on "Pending decisions" 1-14, implemented exactly; where an instruction needed an interpretation it
is named. **Where anything above disagrees, A1 holds.** Nothing was submitted or cancelled by this amendment; the pilot job and
its bodies are unchanged (they already were stage-1 bodies; a submitted pilot's files are never rewritten).

**B200 sharing:** the coordinator notified the misalign session on 2026-11-08 that ptdom2 will occupy B200.

### A1.1 Lower edge and fixB (decision 1)
- **(a) none's lower edge:** where an R domain's none pick or area argmax (A1.2) is its batch's LOWEST none lr, the batch's none
  gets ONE step down (x1/2: A 5e-4, B2 5e-4, B3 1e-3) in R1x, seeds 0-3.
- **(b) fixB (ptdom A5.2):** where the prior's pick (after R1) lies outside the none range of its batch (below the lowest or above
  the highest, counting this round's none asks), none runs at that lr in R1x, seeds 0-3 (ptdom ran fixB at 0-7; here the
  selection seeds only, as decided). A fixB lr equal to an asked step is one cell.
- Both once; the new none lrs join the grid of every R domain of the batch (A1.3); then every unit is re-selected.
- **NOT FINAL** at the final decision: a pick or area argmax still at an outermost lr of its unit (prior: the upper edge;
  none: either edge), or the prior's pick still outside none's range.
- Interpretations (named): (i) the prior has no lower extension (the design's): its lower edge is no edge rule (a prior pick at
  its lowest lr underrates the prior, the conservative direction; fixB covers none's side). (ii) At the R1x decision any edge is
  final: there is no third round, also for a direction that was never used (e.g. a new none lowest made by a fixB lr).
  (iii) An edge whose next lr was run and failed the gate is closed (ptdom A2.1), as before.
- On ptdom's data with the R1pre cells, scalar_adj's prior pick is 1e-3 (bpb tie-break; argmax 2e-3) below B3's 2e-3 unless its
  4e-3 wins: then R1x asks B3 none@1e-3 (fixB = B3's lower step; tests/real_known.out). That depends on R1 and cannot be pre-run.

### A1.2 Trigger = pick OR area argmax at the outermost lr (decision 2)
Every extension (prior 8e-3; none up / down) fires when the unit's pick or area argmax is at the outermost lr; NOT FINAL only
when it is still there after the one allowed step (A0's argmax-only trigger and its NOT FINAL for an un-extended tie-break pick
are superseded).

### A1.3 Shared none grid (decision 3): confirmed as built.

### A1.4 A6 / A6.1 for a shuffled arm (decision 4)
`readout_ptdom2.A6_SHUF = "adjacent"`: for a SHUFFLED arm only, readout_ptdom.geo_a6 with its cka condition replaced by: the
geo_<active> (1 - CKA to the shuffled target on a training batch) at the previous AND the next eval with prior_on <= 0.005; the
floor -0.005, grad_prior finite > 0, the inactive-domain conditions and the A6_KEEP checks unchanged. True-target and none arms:
readout_ptdom.geo_a6 unchanged.

### A1.5, A1.6 (decisions 5, 6): confirmed as built (CKA-rise bar on the shuffled target; lambda via ARGS / lam_domains and the
decomposition double path).

### A1.7 The aux gate at the pick on seeds 8-15 (decision 7)
As ptdom (A2.1 (ii), A4.5): the A2.1 cell gate of the prior's pick cell on seeds 8-11 and on seeds 12-15 separately; either
failing -> **"AUX GATE FAILED at the pick"** instead of a verdict word; the domain stays in the Holm family with its p.
Interpretation (named): the pick cell is the true-target arm of BOTH contrasts, so neither PASS nor GEOMETRY-SPECIFIC is given
(both read "AUX GATE FAILED at the pick"; the words it would have had are printed). The shuffled cell's gate is descriptive.

### A1.8 GATE (decision 8)
A domain whose every prior cell fails the gate is GATE: no verdict, out of the Holm family, no R2 cells; **no HOLD is written**;
select_ptdom2.py exits 0 and the other domains go on (A0's HOLD_R2 / exit 2 superseded).

### A1.9 Every part is ptdom's stage-1 body (decision 9)
No `--save_model`, no offline LM in S, R1, R1pre, R1x, R2 (the pilot already was). CHECK_PY asserts "no saved model and no
offline LM, by design". The F87 bpb tie-break uses the in-training bpb (readout_rules.best_lr uses offline only when every run of
both lrs has it). Every job = its ptdom STAGE-1 twin after the registered substitutions (tests/render_diff.out). Disk: none.

### A1.10 R1pre: the extensions ptdom's data already ask (decision 10)
Verified: applying the A1 edge rules to ptdom's none cells ALONE (the none units do not read any ptdom2 run) gives exactly
- **A none@1.6e-2**: stars' none pick and argmax 8e-3, landmarks' too = A's top (means stars 54.8 / 53.7 / 61.2 / 63.3,
  landmarks 71.9 / 74.2 / 77.3 / 80.9 at 1e-3 / 2e-3 / 4e-3 / 8e-3);
- **B2 none@5e-4**: prefixes' none pick and argmax 1e-3 = B2's lowest (84.2 / 82.2 / 83.3 / 81.6 / 80.1 at 1e-3 .. 1.6e-2);
- nothing else (compass pick 2e-3 / argmax 8e-3, ranks 4e-3 / 8e-3: inside B2's 1e-3..1.6e-2; scalar_adj 4e-3 / 4e-3 inside
  B3's 2e-3..8e-3).
(readout_ptdom2.none_asks_from_ptdom; tests/real_known.out 3.) They form part **R1pre** (make_jobs_ptdom2.R1PRE; 8 runs, seeds
0-3, HOLD_R1pre), submitted with R1. decide_R counts them as the batch's used step in their direction (A up, B2 down) and never
asks them again; select_ptdom2.py --after R1 and the read-out refuse unless R1PRE equals what the rule derives from ptdom's data
(check_r1pre). After R1, A can still get its down step, B2 its up step, B3 either (and fixB).

### A1.11 Part S, seeds 4-7 (decision 11)
The verdict rule is unchanged. Beside it (descriptive): the true - shuffled contrast on seeds 4-7 only (not used by F100's lr
selection): mean, wins of 4, one-sided t p (raw, not Holm-adjusted).

### A1.12-14 (decisions 12-14): unchanged (8 chains; the 2e-3 same-lr line as printed; shared none cells under Holm).

### A1 cost (make_jobs_ptdom2.py --plan; per-run hours = ptdom's stage-1 medians A 0.46, B1 0.35, B2 0.39, B3 0.37)
| part | runs | B200-h |
|---|---|---|
| S | 32 | 13.0 |
| R1 + R1pre | 24 + 8 | 9.8 + 3.4 |
| R1x | 0-40 (6 priors at 8e-3; A down, B2 up, B3 up / down incl. fixB) | 0-16.2 |
| R2 | 120-144 with six FINAL domains | 49.1-59.0 |
| total | min 64 / nominal ~188 / max 248 | 26.2 / 76.8 / 101.4 |
Disk: no saved models.

### A1 files (sha256[:12])
| file | A0 build | A1 |
|---|---|---|
| llmadd_torch/make_jobs_ptdom2.py | 70142319baba (the build, row "Code and files") | 6692cde52800 (R1pre / R1PRE / NONE_LOW, SAVE all False, the R1x cell validation, plan) |
| runs/ptdom2/readout_ptdom2.py | a286e1c9e038 | c0ff2f7ba811 (decide_R rewritten: A1.1-A1.3, A1.8, A1.10; none_asks_from_ptdom, check_r1pre; A1.4 A6_SHUF; A1.7 gate; A1.11; self-test 43 cases) |
| runs/ptdom2/select_ptdom2.py | 4f39b4676c09 | bcee95740b85 (check_r1pre; no HOLD at GATE; --pre for tests) |
| runs/ptdom2/submit_ptdom2.sh | 34e1bd984979 | cf5db17cf681 (part R1pre) |
| CHECK_PY / CHECK_PY_SHUF | 12dff9487798 / 9c6dd473b416 | unchanged |
| pilot job / bodies | 2fe8d0c44578 / 469443e3b64e, 9255c61ea4ca | unchanged (submitted) |
| tests: render_diff / real_known / plumbing / mutation_readout / dryblock_test | | 5882458ee99d / c9190e0ebbf5 / 42879eec5509 / f19a3434d8bb / aeef6f5b4fcc |

### A1 tests (re-run; logs overwritten in place, the A0 results are quoted in the test table above)
| test | log | result |
|---|---|---|
| read-out / selection self-test | selftest_readout.out | **SELFTEST PASS 43/43**: A0's statistics / verdict cases + A1: the PICK at 4e-3 by the bpb tie-break asks 8e-3; none rising -> one up step per batch, fixB for ranks (5e-4) and scalar_adj (1e-3), none_asks_from_ptdom = the up steps only; none falling -> one down step per batch, fixB coinciding with it; still at the edge after the step -> NOT FINAL (up and down); the step worse -> FINAL; with R1PRE run: FINAL at R1 without R1x, and A's argmax at 1.6e-2 -> NOT FINAL at once; missing R1pre cells refused; a gate-failed top lr closes the edge, every cell failing -> GATE without HOLD; the gate at the pick failing on seeds 12-15 -> "AUX GATE FAILED at the pick" for both words, in the family, while one low seed in a half passes; the S seeds 4-7 line; A6 adjacent: a shuffled overshoot with fitting neighbours valid (eval CKA 0.7), not fitting / no neighbour invalid, true-target arms unchanged |
| the self-test bites | mutation_readout.out | **MUTATION READOUT PASS**: 19 mutations caught (A0's 13 + no NOT FINAL, no down step, no fixB, argmax-only trigger, gate at the pick ignored, A6 literal for shuffled arms, S seeds 4-7 from 0-3) |
| job bodies | render_diff.out | **RENDER DIFF PASS 14/14**: the 386 ptdom files re-rendered byte for byte; S 32, R1 24, R1pre 8, R1x 40 (synthetic: every prior 8e-3 + A down, B2 up, B3 up / down), R2 144, pilot 2 = their ptdom STAGE-1 twins after the substitutions; --save_model absent everywhere |
| real data | real_known.out | **REAL KNOWN PASS 15/15**: load2 = load (384); the S pairing; A1.10: the rule on ptdom's data alone asks exactly R1PRE; R1 with planted 4e-3 and R1pre cells = an independent F87 rule 1 + the A1 edge rules (the -20 case asks B3 none@1e-3 by fixB) |
| CLIs | plumbing.out | **PLUMBING PASS 12/12** (the default R1pre checked against the world's ptdom cells; R1pre / R1x / R2 rendered exactly; GATE: rc 0, no HOLD) |
| input checks | dryblock_test.out | **DRYBLOCK PASS 22/22** (+ R1pre none@1.6e-2 / 5e-4 against ptdom's real none dry logs: exit 0; a prior arm's log: exit 3) |
| CHECK_PY_SHUF, pilot check | check_py_shuf_test.out, pilot_check_test.out | 24/24, 3/3 (unchanged code; re-run) |
| sbatch --test-only, submit print | sbatch_test_only_a1.out, submit_print_a1.out | R1pre x 2, S x 1 accepted; submit_ptdom2.sh --part R1pre (no --go): 8 jobs, none submitted before, rc 0 |
| real-data guards | (command line) | guard2 on the 64 rendered job files; manifest, check_r1pre, integrity: no problem |
Not re-run (code unchanged by A1): shuf_trace, the CPU references, shufcka, reused_manifest.

### A1 submission order (supersedes "Submission" step 1-2)
`submit_ptdom2.sh --part S --go`, `--part R1 --go` and `--part R1pre --go` (together, after the pilot passes); after every R1
and R1pre job: `select_ptdom2.py --after R1`; then as before (R1x if asked, `--after R1x`, R2, the read-out).

### A1 audit status
Minimal mode as the build (ii asserts, iii known answers incl. the mutation test). v (a fresh adversarial review) is for the
coordinator, covering the build and A1.

### Coordinator on A1's two interpretations (2026-11-08, before any ptdom2 result)
- A gate failure at the pick withholds both PASS and GEOMETRY-SPECIFIC (the pick cell is the true arm of both contrasts): confirmed.
- The prior's grid is not extended downward, so a prior pick at its lowest lr is not an edge: confirmed. This is
  conservative for PASS (the prior may be below its best), and it does not affect the same-lr geometry contrast. The read-out
  should print "PRIOR AT ITS LOWEST LR" beside such a domain (descriptive).
- A fresh adversarial audit of the build + A1 was started 2026-11-08 (tests/audit_build.md), in parallel with the pilot 22355337.


## Amendment A2 (coordinator, 2026-11-08, after the fresh audit of the build + A1, tests/audit_build.md; before any ptdom2 GPU result)
No ptdom2 result existed (the pilot 22355337 was still pending, unassigned). The coordinator's decisions, implemented exactly;
interpretations named. **Where anything above disagrees, A2 holds.** Nothing was submitted or cancelled.

### A2.1 Part S on fresh seeds 8-15 (audit blocking 1)
- **Why:** F100's PASS needed seeds 4-7 all positive with a Holm p < 0.10, and seeds 0-3 chose the lr: the TRUE arm's 8 seeds
  0-7 were all selected by F100, while the shuffled arm would be new, so true - shuffled is biased in favour of true even when
  the target does not matter (the audit's simulation: up to 2-5x the nominal false-GEOMETRY-SPECIFIC rate). The A0 / A1 claim
  that seeds 4-7 were "never used by F100's lr selection" (readout_ptdom2.py A1.11 line, AUDIT A1.11 / Pending 11) was wrong
  and is withdrawn.
- **Cells (part S):** at seeds 8-15, lr 2e-3: only-<d> and only-<d>-shuf for countries, people (B1) and cities, months (A), and
  none@2e-3 once per batch (B1, A) = 80 runs. CPU references of B1 at seeds 8-15: runs/ptdom2/ptdom2_refs_B1.json
  (08963dd58475; TABLE B1 PASS, tests/run_refs_ptdom2_B1s8.out); A's existed.
- **Verdicts (seeds 8-15 only; halves 8-11 / 12-15; the four conditions of R; Holm family = the four S domains, per contrast):**
  **PASS** on only-d@2e-3 - none@2e-3 (= F100's PASS replicated on fresh seeds) and **GEOMETRY-SPECIFIC** on only-d@2e-3 -
  only-d-shuf@2e-3. Interpretation (named): A1.7's gate at the pick applies to S as to R (the pick cell only-d@2e-3 on seeds 8-11
  and 12-15; failing -> "AUX GATE FAILED at the pick"). All three arms must be ptdom2 runs (checked).
- **The old seeds 0-7 pairing** (ptdom's true arm vs a shuffled arm at seeds 0-7) is descriptive only, labelled "the true arm's
  seeds were selected by F100 (0-3 chose the lr, 4-7 decided the PASS): biased in favour of true". Its shuffled runs are the
  OPTIONAL part **S0** (32 runs, 13.0 B200-h; not in the totals); the line is printed only when S0 has run (Pending A2-a).
- R2 never renders a cell that part S runs (A none@2e-3 at seeds 8-15 when A's none pick is 2e-3): the cell is shared.

### A2.2 Descriptive lines beside every R and S verdict (audit important 2); the pre-registered reading
- shuffled - none (S: none@2e-3 of part S; R: none@N, marked "lr-confounded" when P != N), area and final3;
- the window split of each area gain (true - none, true - shuffled, shuffled - none): the contribution of the evals at steps
  916-7323 (26 of 75) and of the others, in + out = the gain (from the per-eval read-out series; area = 100 x its mean is a new
  double path, series_problems);
- the three arms' LM share a / (a + b) (the cell median on seeds 8-15; none = 1).
- **Reading (pre-registered):** only PASS and GEOMETRY-SPECIFIC together may be written as "the gain depends on the correct
  geometry". GEOMETRY-SPECIFIC without PASS is printed as "true > shuffled, but no gain vs none shown (the shuffled prior may
  hurt)"; PASS without GEOMETRY-SPECIFIC as "the prior helps vs none; not shown to need the correct geometry".

### A2.3 FLOOR / CEILING in Part S (audit important 3)
ptdom's definitions for both contrasts, the k-step line included (months: max(chance, the shortest-span line) + 5 = 13.3; a
line > 100 noted).

### A2.4 The audit's minor items
- `--allow_partial`: integrity / manifest / R1pre / series / sub-read-out problems are printed ("PROBLEMS ...") and every verdict
  is PROVISIONAL (no verdict word) while one stands; the strict read-out still refuses.
- report() no longer fails on a missing run (the --allow_partial read-out on the real, empty ptdom2 tree runs:
  tests/readout_partial_noruns_a2.out).
- "PRIOR AT ITS LOWEST LR" is printed as a note (every status; descriptive).
- per domain: the range of shuf_cka_to_true over the 8 shuffled runs; DIVERGED runs listed.
- the tied-embedding sentence: "for a single-token item the prior's site is the read-out candidate's own output row; both arms
  move those rows, but only the true arm arranges them in the answer geometry, so a GEOMETRY-SPECIFIC there cannot tell 'the
  model used the geometry' from 'the answer geometry was written into the output rows'" (supersedes "applies to both arms
  equally, so a geometry contrast is not affected", Design / verdicts section).
- double path (descriptive): the shuffled arm's eval CKA to the TRUE target at the window's last eval <= the paired none's +
  0.10 and <= the true arm's - 0.2, counted over the 8 seeds.
- **The pilot's bodies were rewritten after submission**: runs/ptdom2/pilot/body_*.sh have mtime 2026-11-08 21:29:56 EST, the
  pilot was submitted at 21:20:31 (SUBMITTED_ptdom2-pilot), by a `make_jobs_ptdom2.py --pilot --check` run before the guard
  against rewriting a submitted pilot existed. Their content is byte-identical to the current rendering (sha256[:12]
  469443e3b64e / 9255c61ea4ca, the A1 table; the audit's pilot_body_cmp.py), and the job had not started, so it will run
  exactly these texts.
- select_ptdom2.py prints the next step as `bash runs/ptdom2/submit_ptdom2.sh --part ... --go` (through the gates), not the
  generator's --submit.
- Not acted on (no decision): the pilot covers the shuffled arms of B1 / B2 only (A and B3 by the CPU trace); the pilot gate
  is not tied to the generator's sha (audit minor 12).

### A2.5 Cost (make_jobs_ptdom2.py --plan; per-run hours = ptdom's stage-1 medians)
| part | runs | B200-h |
|---|---|---|
| S (A2: seeds 8-15, three arms) | 80 | 32.4 |
| S0 (optional, descriptive) | 32 | 13.0 (not in the totals) |
| R1 + R1pre | 24 + 8 | 9.8 + 3.4 |
| R1x | 0-40 | 0-16.2 |
| R2 | 120-144 (8 fewer when A's none pick is 2e-3: part S's cell) | 49.1-59.0 |
| total | min 112 / nominal ~236 / max 296 | 45.6 / 96.2 / 120.9 |
Wall clock with 8 chains running: S + R1 + R1pre ~5.7 h, R1x <= 2 h, R2 6-7.5 h, plus queue waits. No saved models.

### A2 files (sha256[:12])
| file | A1 | A2 |
|---|---|---|
| llmadd_torch/make_jobs_ptdom2.py | 6692cde52800 | 34e7fd02cbe6 (S at seeds 8-15 with none; S0; R2 drops part S's cells; plan) |
| runs/ptdom2/readout_ptdom2.py | c0ff2f7ba811 | 4b29de5bcac3 (contrast_domain / apply_rules for R and S; verdict_S on seeds 8-15; A2.2-A2.4; load2 keeps the per-eval series, eval steps and the window's last CKA; self-test 49 cases) |
| runs/ptdom2/select_ptdom2.py | bcee95740b85 | 1c2dfc6adda0 (next step through submit_ptdom2.sh) |
| runs/ptdom2/submit_ptdom2.sh | cf5db17cf681 | adcb18e10422 (part S0; S 80 runs) |
| runs/ptdom2/ptdom2_refs_B1.json | - | 08963dd58475 (seeds 8-15) |
| tests: render_diff / real_known / mutation_readout / dryblock_test / refs_ptdom2 / run_refs_ptdom2.sh | | e66c6e37fc36 / 3ea82d57d14f / bdbd73386eab / 4d0fec954bce / ab5685dba69a / 9e46cc9ade43 |
| CHECK_PY / CHECK_PY_SHUF, the pilot | unchanged | unchanged |

### A2 tests (re-run; logs overwritten in place)
| test | log | result |
|---|---|---|
| CPU references B1 seeds 8-15 | run_refs_ptdom2_B1s8.out | TABLE B1 PASS (seeds 8-15; DRY rows / splits / tables one value per seed over none and both shuffled B1 arms; no DRY hash repeats over 0-15) |
| shuffled target far enough | shufcka.out | SHUFCKA PASS, 114 planned shuffled cells (S at 8-15 and S0 at 0-7, R2, the pilot), max 0.136 |
| read-out self-test | selftest_readout.out | **SELFTEST PASS 49/49**; new A2 cases: S planted (true - none, shuffled - none) = countries (+4, 0) -> PASS + GEOMETRY-SPECIFIC, people (+4, +4) -> PASS + NOT SHOWN, cities (0, -4) -> NOT SHOWN + GEOMETRY-SPECIFIC with the 'may hurt' reading, months (0, 0) -> neither; shuffled - none and the window split (in + out = the gain; a planted in-window shift lands in the window part); the cka double path 8/8; the seeds 0-7 line labelled; months FLOOR (line 13.3); a missing run refused / PROVISIONAL with the report printing and the problems shown; an outside problem -> PROVISIONAL; series_problems |
| the self-test bites | mutation_readout.out | **MUTATION READOUT PASS**, 22 mutations (+ S on seeds 0-7, outside problems ignored, the reading lines swapped, the window split over every eval, no FLOOR rows) |
| real data | real_known.out | **REAL KNOWN PASS 15/15**: S's contrasts read the seeds 8-15 cells (copies of real seed 0-7 runs relabelled): true - none = the real values read independently, true - shuffled = the planted deltas, the planted verdicts, the window split sums to the gain, the seeds 0-7 line = part S0's planted deltas, labelled; R1 / R1pre as A1 |
| job bodies | render_diff.out | **RENDER DIFF PASS 17/17** (S 80 at seeds 8-15 with none cells, S0 32, R1 24, R1pre 8, R1x 40, R2 136 (A none@2e-3 dropped: part S's cell), the pilot = their ptdom stage-1 twins) |
| input checks | dryblock_test.out | **DRYBLOCK PASS 28/28** (+ S at seed 8 / 12: both B1 shuffled arms and B1 / A none on their seed-8 / 12 references -> exit 0; a true arm on a none log, another arm -> exit 3) |
| CLIs, CHECK_PY_SHUF, pilot check | plumbing.out, check_py_shuf_test.out, pilot_check_test.out | 12/12, 24/24, 3/3 |
| real tree, --allow_partial | readout_partial_noruns_a2.out | runs (no TypeError): every S domain PROVISIONAL (missing runs) |
| guards | (command line) | guard2 on the 144 rendered job files |

### A2 submission order (supersedes A1's)
`submit_ptdom2.sh --part S --go` (80 runs), `--part R1 --go`, `--part R1pre --go` after the pilot passes; `--part S0 --go` only if
the coordinator wants the descriptive seeds 0-7 line; then R as before.

### Pending (A2)
- a. Run part S0 (+32 runs, 13 B200-h) for the descriptive, selection-biased seeds 0-7 line, or drop that line.
- b. The audit's not-acted minors: the pilot does not cover A / B3 shuffled arms on GPU; the pilot gate is not tied to a sha.
- c. A fresh review of the A2 diff (the coordinator's next step).

### Coordinator on A2 (2026-11-08, before any ptdom2 result)
- The A1.7 pick gate applies to Part S too (only-d@2e-3, judged separately on seeds 8-11 and 12-15): confirmed.
- `--part S0` (the shuffled arm at seeds 0-7, for the biased descriptive line) is NOT run. The line is not printed.
- Not acted on (minor, recorded):
  - The GPU pilot does not cover the shuffled arms of batches A and B3. The CPU trace covers all 10 shuffled domains, and a
    GPU-only failure there would fail CHECK_PY_SHUF loudly in the first jobs.
  - The pilot gate is not bound to the generator sha.
- A fresh audit of the A2 diff is started now (tests/audit_a2.md).


## Amendment A3 (coordinator, 2026-11-08, after the fresh A2 audit tests/audit_a2.md; before any ptdom2 GPU result)
No ptdom2 result existed (the pilot 22355337 still pending). The coordinator's decisions, implemented exactly; interpretations
named. **Where anything above disagrees, A3 holds.** Nothing was submitted or cancelled.

### A3.1 Part S's PASS against F100's none lr (audit_a2 blocking 1; the coordinator's A2 instruction had none@2e-3)
- F100's PASS contrasts (runs/ptdom/readout.txt:139, :148; readout.json): countries / people vs none@2e-3, **cities / months vs
  none@4e-3**. `make_jobs_ptdom2.S_NONE` = these picks; the read-out checks it against runs/ptdom/readout.json (check_s_none, a
  refusal reason) and that F100's prior pick is 2e-3 for all four.
- **Part S adds A's none@4e-3 at seeds 8-15 (8 runs): 88 runs.** A's none@2e-3 still runs.
- PASS = only-d@2e-3 - none@<F100's pick> (countries / people 2e-3, cities / months 4e-3): F100's PASS on fresh seeds.
  GEOMETRY-SPECIFIC unchanged (only-d@2e-3 - only-d-shuf@2e-3).
- shuffled - none is at the SAME lr (none@2e-3) for all four; cities / months also print a descriptive "same-lr PASS" line
  (only-d@2e-3 - none@2e-3: the four conditions on the raw one-sided p, no Holm), labelled "not F100's contrast".
- R2 never renders a cell that part S runs (now also A none@4e-3 at seeds 8-15 when A's R pick is 4e-3).

### A3.2 When "the gain depends on the correct geometry" may be printed (audit_a2 important 2)
Only when ALL hold: PASS and GEOMETRY-SPECIFIC; no FLOOR tag (true vs none, true vs shuffled); the final3 gains true - none and
true - shuffled both > 0 (mean); shuffled - none area mean >= -1 (the printed shuffled - none line: S at the same lr 2e-3; R at
none@N, lr-confounded when P != N, named); |LM share(true) - LM share(shuffled)| <= 0.1 (the cell medians on seeds 8-15). Else
the line reads "PASS and GEOMETRY-SPECIFIC, but the geometry-dependence sentence is withheld (A3.2): " and names every failing
condition with its numbers: FLOOR (the tag), "only learns faster" (both final3 gains), "the shuffled arm got worse" (shuffled -
none), "the dose differs" (both LM shares). The verdict words do not change. The audit's five planted worlds (floor_m, floor_p,
fast, harm_small, gate_shuf) are in the self-test and none gets the sentence; a positive control does.

### A3.3 Minor items
- Mutation tests for the description lines: the audit's seven (x1 WIN one eval wider, x6 the shuffled LM share from the true
  cell, x7 the FLOOR > 100 note, x8 CEILING, x9 the true-vs-shuffled FLOOR, x11 the S ptdom2-source check, x13 the cka double
  path), plus one per A3.2 condition and S's PASS against none@2e-3: all caught (34 mutations).
- WIN = the trainer's window, asserted per run in load2: a prior run's prior_on evals = the evals inside WIN, and a finished
  prior run's prior_events = [[916, 1], [7324, 0]] (AssertionError otherwise). The window split's own test is now exact (1e-9).
- The FLOOR > 100 note now reads "the best non-geometric line, the <name>, is <x> > 95: the FLOOR tag holds whenever both arms
  are <= <line>" (it names the line: chance, answer-frequency, shorter-option, the shortcut baseline or the shortest span).
- --allow_partial: the would-be word reads "PROVISIONAL (<word> if final; Holm over the <n> complete of the <N> domains (A3.3:
  the family is reduced))".
- Module docstrings of readout_ptdom2.py and make_jobs_ptdom2.py: Part S = seeds 8-15 (88 runs); S0 optional.
- The B1 references re-run with the current refs_ptdom2.py (bb75694271bf) into tests/refs_B1_a3rerun (rerun_refs_B1_a3.sh,
  cpu_short job 22369676; tests/rerun_refs_B1_a3.out): DRY rows, splits, table shas and step-0 data_sha bit-identical at all 8
  seeds; step-0 lm identical at 7 of 8, seed 10 differs by 1.7e-5 (CPU float noise of the multi-threaded step-0 forward, as B2 s8
  in the build; the in-job tolerance is 1e-3). ptdom2_refs_B1.json (08963dd58475) is kept.
- submit_ptdom2.sh (fb10e0897d31) in print mode for S, R1, R1pre: tests/submit_print_a3.out (88 / 24 / 8 jobs, none submitted
  before, self-test 61/61, the pilot NOTE, rc 0 each).
- Part S0 is refused without the coordinator's explicit flag: make_jobs_ptdom2.py --part S0 --submit needs --allow_s0, and so
  does submit_ptdom2.sh --part S0 --go (tests/s0_gate_a3.out: both refuse without it; nothing was submitted).

### A3 cost (make_jobs_ptdom2.py --plan)
| part | runs | B200-h |
|---|---|---|
| S | 88 | 36.1 |
| S0 (optional) | 32 | 13.0 (not in the totals) |
| R1 + R1pre | 24 + 8 | 9.8 + 3.4 |
| R1x | 0-40 | 0-16.2 |
| R2 | 120-144 (8 fewer per A none cell part S already runs) | 49.1-59.0 |
| total | min 120 / nominal ~244 / max 304 | 49.3 / 99.9 / 124.6 |

### A3 files (sha256[:12])
| file | A2 | A3 |
|---|---|---|
| llmadd_torch/make_jobs_ptdom2.py | 34e7fd02cbe6 | 62d680144854 (S_NONE, S 88, the S0 --allow_s0 gate, docstring) |
| runs/ptdom2/readout_ptdom2.py | 4b29de5bcac3 | 1584aed5dd92 (S PASS vs S_NONE + check_s_none, N_same, the same-lr PASS line, reading_conditions, the WIN asserts, the FLOOR note, the reduced-Holm label, docstring, self-test 61) |
| runs/ptdom2/select_ptdom2.py | 1c2dfc6adda0 | unchanged |
| runs/ptdom2/submit_ptdom2.sh | adcb18e10422 | fb10e0897d31 (--allow_s0; the order text) |
| tests: refs_ptdom2 / mutation_readout / real_known / plumbing / dryblock_test / rerun_refs_B1_a3.sh | | bb75694271bf / 4b2c6cc13821 / 0e2e3c2fe9a6 / d989a6042b5f / 9f3d87552bc6 / 0303856c9d73 |

### A3 tests (re-run; logs in place)
| test | log | result |
|---|---|---|
| read-out self-test | selftest_readout.out | **SELFTEST PASS 61/61** (+ the A3.2 positive control and the five planted worlds without the sentence, each with its named reason; the FLOOR > 100 note naming the shortcut baseline 95.4 and CEILING; floor_geo; cities' PASS against none@4e-3 with the same-lr line against none@2e-3; the cka double path catching a fitting shuffled arm; the reduced-Holm label; an S arm from the ptdom tree refused; check_s_none) |
| the self-test bites | mutation_readout.out | **MUTATION READOUT PASS**, 34 mutations caught |
| real data | real_known.out | **REAL KNOWN PASS 15/15** (S's PASS = the real only-<d> minus none at F100's pick, cities / months' same-lr line = minus none@2e-3, read independently) |
| job bodies | render_diff.out | **RENDER DIFF PASS 17/17** (S 88 incl. A none@4e-3 = ptdom stage-1 twins) |
| input checks | dryblock_test.out | **DRYBLOCK PASS 30/30** (+ A none@4e-3 s9: own reference exit 0, B1's log exit 3) |
| CLIs, CHECK_PY_SHUF, pilot check | plumbing.out, check_py_shuf_test.out, pilot_check_test.out | 12/12 (R2 renders C minus part S's A none@4e-3 cells), 24/24, 3/3 |
| B1 references re-run | rerun_refs_B1_a3.out | as A3.3 above |
| submit print mode, S0 gates | submit_print_a3.out, s0_gate_a3.out | rc 0 x 3; S0 refused without --allow_s0 (both paths) |
| real tree, --allow_partial | readout_partial_noruns_a3.out | runs; every S domain PROVISIONAL (missing runs) |
| guards on the real data | (command line) | guard2 on 152 job files; manifest, R1pre, S_NONE, series, integrity: no problem |

### GPU pilot result (2026-11-09 00:09-00:17 EST, gb008)
Job 22355337 COMPLETED. pilot/pilot_check.out: **PILOT PASS 12/12**.
- P1: only the 3 truncation checks fail.
- P2: the fit to the shuffled target rises (compass 0.159 -> 0.991; countries 0.180 -> 0.996). The CKA to the TRUE target at
  step 2500 stays low: pilot 0.046 / 0.124, against none 0.683 / 0.394 and the true arm 0.999 / 0.990.
- P3: data_sha equals the twin's at every eval; lm before the window is within 0.019.
- P4: speed 1.01x the twin's.
- P5 (info): LM-share medians 0.90 / 0.85.

### Submission (coordinator, 2026-11-09 00:26-00:41 EST)
`submit_ptdom2.sh --part S --go`, `--part R1 --go` and `--part R1pre --go`: every one rc 0. Jobs submitted: S 88, R1 24,
R1pre 8, i.e. 120 SUBMITTED_ flags and 120 llmadd_ptdom2- jobs queued. Each part's gate passed: pilot PASS, read-out
self-test 61/61, /scratch 95.89% < 97%, --check. Logs: submit_go_<part>_20261109.log.

The A3 audit (tests/audit_a3.md) found no blocking item. Its three important items concern the read-out only (the "gain
depends on the correct geometry" reading and a crash path when A none@2e-3 is missing). They are fixed as Amendment A4
before any ptdom2 read-out; no job file changes. misalign was told before B200 use (2026-11-08).

## Amendment A4: read-out fixes from tests/audit_a3.md (coordinator's instruction, 2026-11-09 ~00:45 EST; before any ptdom2 read-out was used)
The A3 audit found no blocking item and three important ones, all in the read-out. Job files and the generator are unchanged
(make_jobs_ptdom2.py 62d680144854, submit_ptdom2.sh fb10e0897d31). The 120 S / R1 / R1pre jobs were already submitted. The
build agent made the edits on a copy (wip_a4/) and swapped them in atomically at 01:02 EST, after every submission had
finished. It then stopped on an API limit before writing this section, so the coordinator wrote it from the code and from
tests/rerun_a4.log.
- **A4.1 'only learns faster' with noise:** "the gain depends on the correct geometry" needs the paired bootstrap 90% lower
  bound > 0 for both final3 gains (true - none@N and true - shuffled). It no longer only requires positive means. The known
  answers hold for 10 rng seeds (tests/selftest_rngs.py b72e231da332: SELFTEST RNGS PASS 10/10).
- **A4.2 the shuffled arm:**
  - (a) shuffled - none (S: none@2e-3) must have a bootstrap 90% UPPER bound >= 0, i.e. the shuffled arm is not shown to
    hurt. A missing or invalid none cell fails this condition.
  - (b) mean(true - shuffled) >= 0.5 x mean(true - none@N).
  - These replace A3's -1 bar. Planted cases: true +4 / shuffled +3 -> no sentence, reason (b); true +1.5 / shuffled -0.9
    (significant harm) -> no sentence, reason (a); a +5 / ~0 positive control -> the sentence.
- **A4.3 descriptive cells:** when A's none@2e-3 is missing or invalid, the PASS of cities / months (against none@4e-3) and
  GEOMETRY-SPECIFIC are judged as usual. Only the lines that need none@2e-3 print "missing / invalid". There is no TypeError
  in strict or --allow_partial mode. B1's none@2e-3 is the PASS arm of countries / people, so a gap there is a refusal reason.
- **A4.4 minors:**
  - a run whose prior window differs from WIN is INVALID with the reason (no crash);
  - check_s_none refuses a planted F100 read-out whose picks differ;
  - the A2.2 closing sentence is removed;
  - R1x docstrings are updated;
  - mutation coverage extended.
- **A4.5:** the dose condition (|LM share true - shuffled| <= 0.1) is unchanged.
**Tests on the final files** (tests/rerun_a4.log; readout_ptdom2.py 21797ec55b8f, select_ptdom2.py 78c4a3a757ba;
2026-11-09 06:04-06:09 UTC):
| test | result |
|---|---|
| readout self-test | SELFTEST PASS 82/82 |
| selftest_rngs | 10/10 rng seeds |
| mutation_readout | 58 mutations each caught |
| real_known | 15/15 |
| plumbing | 12/12 |
| CHECK_PY_SHUF | 24/24 |
| pilot_check | 3/3 |
| render_diff | 17/17 |
| dryblock | 30/30 |
**Read-out hygiene:** the agent also ran `readout_ptdom2.py --allow_partial` on the real root at 06:07 UTC
(tests/readout_partial_a4.out), after the code was final. The read-out prints PROVISIONAL with no verdict word (most cells
were missing then). The coordinator has not read its numbers. A fresh audit of A4 runs before the first verdict read-out.

## Amendment A5 (coordinator, 2026-11-09 ~06:30 EST): interval conditions for the reading sentence (tests/audit_a4.md important 1-2; before any verdict read-out)
The A4 audit found no blocking item. The verdict words are unchanged by A4 (4,000 random families + 32 full-pipeline domains
identical; zero-effect P(PASS) 0.023-0.028, P(GS) 0.018-0.029). Important 1-2: at ptdom-like noise and n = 8, the A4.2 point
rules still printed "the gain depends on the correct geometry" in 44-47% of worlds where the shuffled target hurts (true +1.5
/ shuffled -0.9), and in 17-34% where the shuffled arm takes 75% of the gain.
**A5 (reading layer only; the verdict words, Holm, the gate and NOT FINAL do not change):** besides every A4 condition, the
sentence needs both of these excluded by the paired bootstrap 90% interval of shuffled - none (the none of A4.2a; Part S
none@2e-3, R none@N, lr-confounded when P != N):
- (c) a harm of more than 1 area point: lower bound >= -1 (HARM_MAX);
- (d) the shuffled arm reaching half of the gain: upper bound <= 0.5 x mean(true - the same none).
Otherwise the reading names "UNDETERMINED at n = 8" with the interval. A missing seed in these series also gives
UNDETERMINED. This lowers the sentence's power by design: at n = 8 the data cannot separate "the geometry is needed" from "the
shuffled target hurts or helps a little" unless the shuffled arm sits close to none.
**Code:** readout_ptdom2.py 21797ec55b8f -> 3f78507d5d47 (copy of A4: superseded_a4/readout_ptdom2.py.a4); the self-test helper
takes per-seed shuffled offsets; 4 new self-test cases (A5c alone, A5d alone, positive controls); the two A4.2 cases now also
expect the A5 lines; tests/mutation_readout.py +4 mutations (each rule disabled, each rule on the mean instead of the interval).
**Tests (tests/rerun_a5.sh -> rerun_a5.log, cpu_short; no read-out on the real root):**
| test | result |
|---|---|
| readout self-test | SELFTEST PASS 86/86 |
| selftest_rngs | 10/10 |
| mutation_readout | 62 mutations each caught |
| real_known | 15/15 |
| plumbing | 12/12 |
| CHECK_PY_SHUF | 24/24 |
| pilot_check | 3/3 |
| render_diff | 17/17 |
| dryblock | 30/30 |
**Not acted on (A4 audit minors):**
- The global sub-read-out check also refuses on an invalid descriptive run (recorded; strict mode, rare).
- wip_a4's --allow_partial run on the real root at 00:58 EST also printed PROVISIONAL only (recorded here).
- The auditor's five cpu_short helper jobs (22406463, 22406472, 22406483, 22406560, 22407417): the coordinator's scancel was
  refused by the auto-permission check. They write only into the audit's scratch dir.

### Coordinator after tests/audit_a5.md (2026-11-09 ~07:00 EST; before any verdict read-out)
The A5 audit found no blocking item. The verdict words are bit-identical to A4 (16,000 rows; 18 scenarios x 3,000 sims), and
A5 blocks the cases it targets. Measured on the audit's noise model, P(sentence printed) from A4 to A5:
| scenario | countries | cities |
|---|---|---|
| true +1.5 / shuffled -0.9 | 0.270 -> 0.034 | 0.132 -> 0.005 |
| shuffled -1.5 | 0.158 -> 0.006 | 0.114 -> 0.003 |
| shuffled takes 75% of the gain | 0.046 -> 0.001 | 0.072 -> 0.003 |
| zero effect | <= 0.001 | <= 0.001 |
**Important (power):** the sentence has low power whatever the effect size. With the shuffled arm equal to none, P(printed)
for countries / cities / months / people is:
| true effect | countries | cities | months | people |
|---|---|---|---|---|
| +3 | 0.28 | 0.06 | 0.53 | 0.87 |
| +5 | 0.47 | 0.22 | 0.63 | 0.89 |
| +8 | 0.47 | 0.32 | - | - |
This is because the n = 8 bootstrap half-width of shuffled - none (countries ~1.0, cities ~1.5) exceeds HARM_MAX = 1.

**Decision (no code change):**
- The sentence stays descriptive with this power table. "UNDETERMINED" is the expected reading for countries / cities and
  is not evidence against geometry dependence.
- The FINDINGS wording is fixed now:
  - (i) "the gain needs the correct geometry" may be written only for PASS + GEOMETRY-SPECIFIC + no FLOOR + A4.1's final3
    condition, and always together with the shuffled - none 90% interval.
  - (ii) Where A5 reads UNDETERMINED, the wording is: "the true target beats a dose-matched shuffled target, but at 8 seeds
    a harm or a partial benefit of the shuffled target cannot be excluded: shuffled - none [lo, hi]".
  - (iii) GEOMETRY-SPECIFIC without PASS keeps READING's wording.
- No per-domain delta is introduced; the auditor's alternatives (delta = 1.5, or intervals only) are not taken.

**Corrections to the A5 text above:**
- A missing seed in the A5 series gives the "missing / invalid" reason in the pipeline (the sentence is withheld). The
  UNDETERMINED branch for a missing seed is not reachable.
- "62 mutations each caught" is from the separate run after rerun_a5.log (that log ran the older file with 58 mutations;
  tests/mutation_readout.out holds the 62 / 62 run, file 818032ad0788).

**Minors recorded, not acted on:**
- 11 of the auditor's 20 extra mutations are not caught by the self-test (A5 skipped for cities / months, the tn5 sign, A5d
  against none@N, other thresholds, an 80% interval).
- The report header still lists A1-A4.
- The P != N lr-confounded label is not repeated on the A5 lines.

## Coordinator log 2026-11-10 ~04:00-04:30 EST
- Every S / R1 / R1pre job (120) is COMPLETED, 120 check.json pass.
- `select_ptdom2.py --after R1` (rc 0; select_after_R1.txt) -> picks_R_after1.json 9c90d33523a9.
  - FINAL: landmarks, compass, prefixes, ranks.
  - PENDING on R1x: stars (prior argmax 4e-3 -> 8e-3 asked); scalar_adj (prior pick 1e-3 below B3's none range -> fixB
    none@1e-3).
  - Submitted: `submit_ptdom2.sh --part R1x --picks runs/ptdom2/picks_R_after1.json --go` (8 jobs; /scratch 94.93%).
- Part S is complete. Strict read-out `readout_ptdom2.py --part S --json runs/ptdom2/readout_S.json` -> readout_S.txt.
  All four domains are PASS (against F100's none pick) and GEOMETRY-SPECIFIC; the reading sentence is withheld for all four.
  A fresh results audit has started (tests/audit_results_S.md), before any FINDINGS entry.
- 2026-11-10 ~14:50 EST: the 8 R1x jobs are COMPLETED (128 ptdom2 runs, all check.json pass). `select_ptdom2.py --after R1x`
  (rc 0; select_after_R1x.txt) -> picks_R_after2.json e4284d1ea1c4.
  - FINAL: landmarks, compass, prefixes, ranks, scalar_adj.
  - NOT FINAL: stars (see select_after_R1x.txt).
  - R2 = 15 cells x seeds 8-15. Submitted with `submit_ptdom2.sh --part R2 --picks runs/ptdom2/picks_R_after2.json --go`:
    120 jobs, /scratch 94.98%.
  - The Part S results audit was restarted (the first audit agent stopped on an API limit; its scratch scripts are reused).
