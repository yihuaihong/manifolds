# pt6comb — six manifold priors trained together at 53M: does each help, and does none interfere? (2026-11-10)

Status: **BUILT, NOT SUBMITTED** except the ONE GPU smoke job (section "GPU smoke"; not part of any result). Written before any
pt6comb GPU result existed. Nothing of ptdom / ptdom2 / rb6 was modified (they are imported read only, sha-asserted).
Design: the user's goal ("six manifolds added together, each helps, none interferes") and the LLM_addition coordinator's design of
2026-11-10, implemented as given; the coordinator's two follow-ups of 2026-11-10 (composition as a parameter; then **composition B
decided: numbers + S4 + dim + countries + cities + people**; months dropped because F76 / F88 record a tier-E drop of months that
comes with the numbers prior; composition A kept as a parameter only; the "X from ptdom2 Part R" logic removed). Every point where
the build had to interpret or saw a problem is in "Pending decisions (待协调者定)" at the end.

## 1. Intent
- Background: F100 / F101 (ptdom / ptdom2 Part S): the single priors of countries, cities, people (and months) help their own
  held-out read-out at 53M on fresh seeds; F86 / F88: the numbers prior (helix3 on the operands) helps strongly; F88 (rb6): S4 passed
  every pre-registered rule (A8.3: s4_hard area +26.3, 7/8, fresh seeds), dim narrowly missed at 53M (p .06) and was above every
  tested none at 505M (F88 补记三).
- Question, per manifold m of the six: in 53M pretraining on identical data, does the six-prior arm all6 raise m's held-out read-out
  above none (HELPS), and is all6 on m not much below the single prior only-m (NO INTERFERENCE)? Headline: the number of manifolds
  with HELPS and NO INTERFERENCE, each named.

## 2. Feasibility: can trainer Rev G mix the Rev B toy domains with the ptdom domains? (coordinator item 2) — YES, without a trainer change
Read from the code and shown on CPU (tests/feas/: feas.sh -> feas.out, feas_check.py -> feas_check.out; job 22591972, cpu_short):
- Domain list: `--domains` accepts any subset of ALL_DOMAINS incl. numbers (pretrain_range.py:536-545); EXT is built in the fixed
  order EXTRA + ANCHOR + NEWDOM (:666-668), so --domains numbers,s4,dim,cities,countries,people gives EXT = s4, dim, cities, countries,
  people; each domain's split / shuffle / eval-context streams are indexed by its position in its OWN fixed list (s4 / dim: the 4100
  stream children 1/6/11 + EXTRA index; cities: 4200 + ANCHOR index; countries / people: 4300 + NEWDOM index, :870-879), so every
  domain's split equals its split in rb6 / ptdom at the same seed whatever the mix (verified, below).
- Lines / sites / read-outs: per-domain code paths (dom_q for s4 / dim: single-token nonce sites at both inputs, asserted to decode
  to the items at step 0; anc_line for the comparisons: the last token of each item name, asserted on all 9,000 / 6,000 sites by
  --dry_inputs); the read-outs are per domain (Rev B exact match + per-question hits for s4 / dim; Rev E 2-way for the comparisons).
- Numbers prior with ptdom domains: `--target helix3` + numbers in --prior_domains (asserted iff, :541); lam_rule each gives numbers
  lambda 1 (:677-680); dom_grads logs gnorm_true / gnorm_prior for numbers and sums every active prior into grad_prior (:2195-2234),
  so the A2.1 gate's b = |sum lambda_d g_d| includes the numbers prior.
- CPU evidence: 53M --dry_inputs of none and all6 at seed 0 run (model 52.7M, 'target features (1000, 10)' for all6, 3 site-rule
  lines, content scan clean, token shares text 92.34 / numbers 2.124 / s4 0.469 / dim 0.469 / cities 1.55 / countries 1.41 / people
  1.64 %, slot shares 0.48-0.53); a tiny d64 x 3 smoke of all6 / none / only-numbers / only-s4 (200 steps, window 10..79): **FEASIBILITY
  PASS 68/68** (same data_sha at every eval and the same step-0 lm in all four arms; each active prior on exactly at the on-evals,
  gnorm_prior / gnorm_true = 1 for all six incl. numbers, applied 100% of the on-steps, no field for an inactive prior; every metric
  (numbers, s4_readout, dim_readout, the 2-way read-outs) and the per-question hits present; rb6's s4_hard / strict-dim code runs on
  the new run JSONs and re-derives the trainer's s4_readout area from the hits).
- What did need adapting (generator / checks only, no trainer change): ptdom's CHECK_PY assumed the numbers prior off, no toy domain,
  and a run file named run_whole_none_...; ptdom's render counted a site-rule line for every non-months domain (s4 / dim print none).
  All handled by counted edits (section 6) and caught by tests (the run-file name would otherwise have made every numbers-prior run
  fail its CHECK with no check.json).

## 3. Design
Everything not stated here is ptdom's (runs/ptdom/AUDIT.md A3-A6.1) / ptdom2's (A1-A5): 53M (52.7M params, d384 x 6), 3e8 tokens =
18,310 steps of 32 x 512, 75 evals (every 250 + the last), B200 only, ptdom's stage-1 job body (no saved model, no offline LM),
trainer Rev G (commit 11f61b9, pretrain_range.py d87a1502cee8; runs/pt6comb/code = git archive of its tools/, byte-identical to
runs/ptdom/code).

### 3.1 Data (composition B; identical in every arm of a seed)
Row = FineWeb-Edu text + one numbers line (2%, always first after the text; F86 / F88's main setting) + K = 2 lines of the five domains
s4, dim, cities, countries, people (the trainer's cycling over EXT; K = 2 = rb6's pre-registered K*). Domain lines per run = 234,368 per
domain (5 domains; ptdom had 195,307 with 6): s4 / dim ~679 exposures per training question (= rb6's ~680: rb6 also had five
non-number domains at K 2), comparisons ~130 per training prompt (ptdom ~108).

### 3.2 The six manifolds and their settings (source of each item)
| manifold | lines / site | split | target (prior) | primary read-out (area) | source |
|---|---|---|---|---|---|
| numbers | `Q: 643 + 287 = 930` (4 formats, operands 0..999 by length bucket, 5% rehearsal), the first line after the text; site ctx = the operands' last tokens | per-seed held-out pairs (20 / 150 per bucket) | helix3 (1000 x 10), `--target helix3`, lambda 1 | the 6-bucket held-out exact match (mean of held_b11..b33) = metrics['numbers'] | F86 / F88 (pt2b's 2% setting; lambda 1 = F88's auxiliary-loss rule, not F86's 5) |
| s4 | `⟦ gez⟧ then⟦ cuk⟧ is nok` (24 nonce single tokens, the cube rotation group, "a then b"); sites = both inputs | question split, `--s4_train 0.6` (rb6 default), 345 / 231 | R: the 3x3 rotation matrices (9-dim), the trainer's centred Gram | **s4_hard**: the held_fresh_nonid cells NOT solvable by Latin-square propagation (naked + hidden singles) from the seed's training table, from the per-question hits (rainbow6_summary.s4_hard_series; 179 cells at seed 0, 173-193 at seeds 8-15 = rb6 A8.3's precomputed sets, the splits being identical); chance = the pure table-filler's expected accuracy (~24-27%) | F88 A7 / A8.2 / A8.3 (the confirmatory metric of F88 补记 A8.3); toy_struct d182ce548bcc |
| dim | `⟦ jel⟧ times⟦ zag⟧ is vak` (27 nonce single tokens, {-1,0,1}^3 exponent vectors, times / per); sites = both inputs | relation-class split, `--dim_train 0.5` (rb6 default), 344 / 342 | expo: the exponent vectors (3-dim) | **the strict dim set**: no_negtwin_nonid AND no same-answer token-transform twin trained (rainbow6_summary.strict_mask / strict_series; 58 cells at seed 0) | rb6 A5 STRICT (F88's dim read-out "严格集") |
| cities | `Q: Which city is closer to⟦ Shanghai⟧,⟦ Istanbul⟧ or H⟦anoi⟧? Answer: Hanoi` (60 items; sites = the last token of anchor and both options) | ft36 option-pair split 40% | sphere | the 2-way choice over every held-out question, both orders (chance 50) = metrics['cities_readout'] | F100 / F101 (ptdom batch A) |
| countries | idem, 70 countries | idem | sphere | idem | F100 / F101 (B1) |
| people | `Q: Who was born earlier, Confuc⟦ius⟧ or Spart⟦acus⟧? Answer: Confucius` (106; both options) | idem | rank (1-D) | idem | F100 / F101 (B1) |
Every split above equals rb6's / ptdom's / ptdom2's recorded split of the same seed (tests/run_refs_pt6comb_B.out: **PTDOMCMP PASS**,
60 domain x seed comparisons over seeds 0-3, 8-15: cities = ptdom A / ptdom2 A references, countries / people = B1's, s4 / dim =
make_rb6m_tasks.DOMS). The held-out-format templates of s4 / dim (3 each) are scored at every eval (descriptive).
- `mean6` = the unweighted mean of the six primary areas: the lr-selection objective only.
- Window (0.05, 0.4) = steps 916..7323 for every prior (one shared schedule), lam_rule each, lambda 1 each (F88).

### 3.3 Arms
`none` (`--target none`); `all6` (`--target helix3 --schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains
numbers,s4,dim,cities,countries,people --lam 1`); `only-<m>` for each of the six (only-numbers = `--target helix3 ... --prior_domains
numbers`; only-<d> = `--target none ... --prior_domains <d>`).

### 3.4 lr selection (seeds 0-3; part L1, then L1x once)
- L1: none @ {2e-3, 4e-3, 8e-3}, all6 @ {1e-3, 2e-3, 4e-3}, seeds 0-3 = 24 runs. The six only-<m> arms run at **2e-3 only** (F100's
  prior pick; F86's 53M 3e8 numbers-prior pick was also 2e-3); they are not selected and have no seeds 0-3.
- The A2.1 auxiliary-loss gate first (ptdom A2.1): per all6 (lr) cell the median over its finished runs of the run's on-window median
  LM share by norm a / (a + b) (a = grad_lm, b = grad_prior = |sum lambda_d g_d| over the six, pre-clip) < 0.5, or an unverifiable run
  (a missing gradient field at an on-window eval, no on-window eval, the logged grad_lm_share off its re-derivation by > 1e-3) -> the
  cell cannot be picked; every all6 cell failing -> **GATE** (no confirmation).
- Each unit (all6, none) by **F87 rule 1** (readout_rules.best_lr 'top2') on mean6: the argmax of the 4-seed mean; the top two within
  2 points -> the lower mean in-training val bpb (stage-1 bodies have no offline LM); then the lower lr.
- Extensions, once (ptdom2 A1 / A1.2): all6 @8e-3 when all6's pick OR area argmax is 4e-3 (and 4e-3 passed the gate); none @1.6e-2 /
  @1e-3 when none's pick or argmax is 8e-3 / 2e-3; **fixB** (ptdom A5.2, ptdom2 A1.1 (b)): none at all6's pick when it lies outside
  none's range (counting this round's asks; with these grids only 1e-3, = none's down step: one cell). all6 has no lower extension
  (ptdom2 A1.1 (i); the read-out prints "PRIOR AT ITS LOWEST LR"). A gate-failed next lr closes that edge.
- After L1x both units are re-selected over all their lrs; **NOT FINAL** when a pick or argmax is still at an outermost lr it ran
  (all6: its top; none: either edge) or all6's pick lies outside none's range -> no confirmation; the coordinator decides.
- select_pt6comb.py writes picks_after_L1.json / picks_after_L1x.json once (chmod 444); the read-out and submit_pt6comb.sh re-derive
  them (--verify).

### 3.5 Confirmation (part C; fresh seeds 8-15, never read by any selection)
none@N, all6@P, only-<m>@2e-3 for the six m, seeds 8-15 = 64 runs (CPU references for seeds 8-15 exist).

## 4. Pre-registered verdicts (readout_pt6comb.py; seeds 8-15; written before any pt6comb GPU result)
Per manifold m, per seed s, on m's primary area: g_s = all6@P - none@N; h_s = all6@P - only-m@2e-3; u_s = only-m@2e-3 - none@N.
- **HELPS** iff the one-sided paired t p (H1: mean g > 0; df 7) < 0.05 AND g_s > 0 in >= 6 of the 8 seeds; else **NOT SHOWN**.
- **NO INTERFERENCE** iff mean h >= -0.5 x mean u, OR the one-sided paired t p (H1: mean h < 0) >= 0.05; else **INTERFERENCE**.
- **Headline**: "k of 6 manifolds HELPS and NO INTERFERENCE: <names>". The six tests are NOT corrected for multiplicity (the user's
  choice of a direct rule, 2026-11-10); the report states this and prints the Holm-adjusted HELPS p (family of the six) as a
  reference column ("HELPS also under Holm" / "not under Holm").
- Also reported: each manifold's mean g with its 95% t interval and bootstrap 90% interval (readout_ptdom.boot_ci, 10,000 resamples,
  default_rng(0)), and the per-seed mean of the six g (mean, 95% t interval, bootstrap 90%).
- **all6 GATE**: the A2.1 cell gate at all6@P on seeds 8-11 and on seeds 12-15 separately (ptdom A4.5 / ptdom2 A1.7: the inherited
  per-half rule); either failing -> no verdict word for any manifold ("all6 GATE ..."), the numbers printed.
- **Refusal**: a needed run missing or invalid, data differing between the arms of a seed (data_sha at every eval, step-0 lm within
  1e-3), a metric double path disagreeing, a picks file differing from its re-derivation -> the strict read-out refuses;
  `--allow_partial` prints "PROVISIONAL (<word> if final)" with the reasons.
- What the words mean: HELPS = all6 raises m's held-out read-out above none@N on fresh seeds at this rule (Tier L for the
  comparisons and S4 / dim: the targets carry the answers). INTERFERENCE = all6 is shown below only-m by more than half of only-m's
  gain. NOT SHOWN / NO INTERFERENCE at 8 seeds are statements about this rule, never "no effect" / "no harm" in general.

### Descriptive lines (never a condition)
final3 of g / h / u; the window split of g (the evals at steps 916..7323 = 26 of 75, and the others; in + out = the area gain); FLOOR
(both all6 and none mean areas <= the mean over the seeds of the FLOOR line) / CEILING (none >= 95) with ptdom's line per read-out:
numbers 0 + 5; s4_hard max(1/24, the table-filler's chance) + 5; dim strict max(chance = max(uniform, majority training answer), the
answer-elimination heuristic) + 5; comparisons max(50, the answer-frequency and shorter-option lines, the training-frequency shortcut
baseline on every held-out question) + 5. Per arm (seeds 8-15): the LM share a / (a + b) (cell median, the lowest run), the share of
on-window evals with b > a and with clip < 1, the clip coefficient; the LM cost vs none@N (final in-training val loss and val bpb x 1e3;
"lr-confounded" when the lrs differ) and the same-lr rows all6 - none on seeds 0-3 at every lr both ran. Secondary read-outs: s4
held_fresh_nonid area, dim no_negtwin_nonid area, the comparisons' hard-subset final3; the eval-prompt CKA to each target at the
window's last eval (all6 / only-m / none). Flagged-valid runs (A5.1 / A6) are listed.

## 5. Validity (inherited, generalised)
- In-job checks (exit 3 before training, 4 after): ptdom's job body: HOLD, the 12 snapshot shas, the B200 guard, the data sha256,
  --dry_inputs with the input checks (model line; the numbers target — 'target features (1000, 10)' for numbers-prior arms, none
  otherwise; per domain prior / shuf / lam; numbers first; the content scan; 3 site-rule lines; the DRY rows = the CPU reference of the
  seed), training, CHECK_PY_6 with the cell's intent and its $ARGS.
- VALID run (readout_pt6comb.load): finished (75 evals) and check.json passes, or its only failed checks are the CKA-rise bar (a weak
  prior: data), the GPU-numerics double-path tolerances (ptdom A5.1: flagged data) and / or the geo check at a geo overshoot of a
  perfect fit, **verified for EVERY active prior** (ptdom A6 / A6.1 generalised, geo_a6_multi: grad_prior finite > 0 at every eval
  with prior_on; at every such eval where an active prior's value is <= -0.001 — numbers: its value geo with cka_ops_helix3 — that
  value >= -0.005 and its eval CKA in [0.995, 1.01]; no inactive prior has a value / gradient field; the separate checks present once
  and passed). The run's identity (seed, lr, tag, target, prior_domains, domains, prior flags) must equal its directory name
  (ptdom2's identity check). A finished run is never re-run (ptdom A2.3); re-runs are new names -r<k>; integrity() refuses a cell
  with a finished run in an earlier dir.
- Metric double paths (refusal reasons): the numbers area re-derived from the six buckets = metrics['numbers']; the comparisons'
  2-way series and sub-read-outs re-derived from the per-question hits (readout_ptdom.run_subsets: table sha, question order, masks,
  heuristic lines, series, areas); the s4 / dim held_fresh_nonid / no_negtwin_nonid areas re-derived from the hits = the metrics.
- a6_guard6: every rendered job file's CHECK_PY is CHECK_PY_6 exactly, with the geo check's template once and the separate checks.

## 6. Code and files (sha256[:12] at the build)
| file | sha | role |
|---|---|---|
| llmadd_torch/make_jobs_pt6comb.py | 344ab0352dee | generator: renders through make_jobs_ptdom.render (6061b299129c asserted) with module names re-pointed for one call (ptdom2's technique; make_jobs_ptdom2.py 62d680144854 asserted) + counted text edits; CHECK_PY_6; --plan / --snapshot / --part L1, L1x, C / --smoke / --rerun |
| runs/pt6comb/readout_pt6comb.py | c2e4a855f677 | metrics, loader (A6 generalised, identity), integrity, gate, decide (selection), verdicts, report, --selftest (readout_ptdom.py 1e2f17522e47 asserted; rainbow6_summary.py of the snapshot for s4_hard / strict dim) |
| runs/pt6comb/select_pt6comb.py | 75b4f0c00915 | --after L1 / L1x, --verify |
| runs/pt6comb/submit_pt6comb.sh | 1abd11924fcf | submission helper (checks; --go submits) |
| runs/pt6comb/smoke/smoke_check.py | 92fa2b4f7bd4 | the GPU smoke's pass rules |
| CHECK_PY_6 / ptdom's CHECK_PY | f71e3b35f3fb / 12dff9487798 (sha256[:12] of the heredoc text) | in-job checks |
| runs/pt6comb/code/ | = runs/ptdom/code/tools byte for byte (git archive of 11f61b9) | the trainer snapshot |
| runs/pt6comb/pt6comb_refs_B.json | 9bbeca432ac0 (chmod 444) | CPU references, seeds 0-3, 8-15 |
CHECK_PY_6 = ptdom's CHECK_PY + 7 counted edits (the full unified diff is printed in tests/render_diff.out): (1) the numbers target is
the arm's (helix3 / none); (2) a toy read-out check (s4 / dim held / seen at every eval and the per-question hits hits_<d>_held, one
digit per held question); (3) the GPU-tolerance text lists only the domains with a scoring double path; (4) the numbers prior: rb6's
"applied on every on-step, its value geo logged" when active, ptdom's "off" check otherwise; (5) the geo check covers every active
prior (numbers: geo); (6) the CKA-rise bar: numbers = cka_ops_helix3 (rb6's); (7) info: the operands' final CKA. Render edits (counted):
the comments naming ptdom, the HOLD line (HOLD_<part>), the slot text, the site-rule count (3 = the comparisons; ptdom counted every
non-months domain), the run-file name (run_whole_helix3_... for numbers-prior arms), and the smoke bodies' out dir.

## 7. Tests (runs/pt6comb/tests/; CPU: cpu_short via cpu_run.sh, or this node at nice 19; every log is the script's own output)
| test | script / log | result |
|---|---|---|
| feasibility (53M dry inputs + tiny CPU smoke of four arms) | feas/feas.sh, feas_check.py / feas/feas.out, feas_check.out | **FEASIBILITY PASS 68/68** (section 2) |
| CPU references (dry + step 0 of none at seeds 0-3, 8-15; arm independence: dry of all6 / only-numbers / only-s4 at seed 0 and all6 at seed 8, step 0 of all6 at 0 and 8, only-numbers at 0) | refs_pt6comb.py, run_refs_pt6comb.sh / run_refs_pt6comb_B.out, refs_B/ | **TABLE B PASS** (one DRY hash / split / table / lm0 / data_sha per seed over the arms; no DRY hash repeats; step-0 lm arm-independent within 1e-5) -> pt6comb_refs_B.json; **PTDOMCMP PASS 60/60** (the splits = ptdom's / ptdom2's / rb6's) |
| the jobs' input checks on real CPU dry logs | dryblock_test.py / dryblock_test.out | **DRYBLOCK PASS 29/29**: own cell exit 0 (every reference seed; all6, only-numbers, only-s4, the smoke bodies; lr-independent); exit 3 for an all6 job on none's log, none on all6's, only-numbers on all6's, only-s4 on only-numbers', all6 on only-numbers', another seed (2), ptdom A / B1's real GPU dry logs |
| CHECK_PY_6 on real (tiny CPU) run JSONs | check_py6_test.py / check_py6_test.out | **CHECK_PY_6 TEST PASS 13/13**: each arm's own run fails exactly the two checks a tiny CPU run cannot pass (the 2% numbers share at seq 128, the B200 name); tampers fail their own check (numbers geo -0.002; hits_s4_held missing; the numbers prior never applied; no weighted numbers gradient; a flat cka_ops_helix3; domains.numbers.target none; --lam 2 in ARGS); cross-arm runs fail the arm checks |
| job bodies | render_diff.py / render_diff.out | **RENDER DIFF PASS 4/4**: 102 bodies (L1 24, C 64, L1x 12, smoke 2): bash -n, every line differing from the ptdom stage-1 twin in a registered category, the CHECK_PY body = CHECK_PY_6 exactly (diff printed); every ARGS as its arm wants; after all patched renders, make_jobs_ptdom.render re-renders the 240 submitted ptdom stage-1 files and make_jobs_ptdom2.render2 the 88 ptdom2 part-S files byte for byte |
| metric definitions on real published runs | real_known.py / real_known.out | **REAL KNOWN PASS 21/21**: s4_hard area all6 - none@1.6e-2 at rb6 A8.3 seed 8 = +22.9 (F88 A8.3's per-seed list); F88 confirm seeds 4-7: numbers area +57.6 (vs none@1e-3) / +33.8 (vs none@8e-3), S4 held_fresh_nonid final3 +44.1 / +42.9, dim strict final3 +55.6 (vs none@8e-3); F101: countries +4.8, people +1.2, cities +4.1 (ptdom2 Part S); every run's numbers / comparison series area = its metric |
| loader on real (stretched) run JSONs | loader_test.py / loader_test.out | **LOADER TEST PASS 11/11**: the four tiny runs stretched to 75 evals load VALID with every manifold's area / FLOOR line and no double-path problem; identity mismatch -> INVALID; a failed non-flag check -> INVALID; CKA-only -> valid; A5.1 -> valid flagged; unfinished -> INVALID; A6 generalised (an s4 overshoot at cka 1.0002 in all6 -> valid flagged; at 0.9 -> INVALID); tampered numbers metric / flipped cities hit / s4_readout metric -> problems; an earlier finished dir and data differing between arms -> problems |
| read-out / selection self-test (planted known answers) | readout_pt6comb.py --selftest / selftest_readout.out | **SELFTEST PASS 46/46**: statistics = scipy; Holm; selection S1-S15 (interior picks -> final; edges -> extensions; none's lower edge + fixB as one cell; the bpb tie-break; the gate excluding a cell and closing its edge; GATE; missing cell; after L1x FINAL / NOT FINAL (all6 top, none either edge); the objective is mean6 (numbers alone would pick otherwise); the PICK at an edge triggers); verdicts V1-V20 (HELPS / NOT SHOWN incl. 5/8 and p in (.05, .10); INTERFERENCE / NO INTERFERENCE by margin and by p; the u <= 0 case; the headline; Holm reference; the mean of six; the gate per half; window split; FLOOR / CEILING; a missing run; data identity); A6 generalised (8 cases) |
| the tests bite | mutation_readout.py / mutation_readout.out | **MUTATION READOUT PASS 25/25 caught** (the control passes both): thresholds (wins 5, alpha .10, margin 0), two-sided HELPS, interference p direction, 'and' for the two clauses, margin sign, no wins condition, Holm applied, headline = HELPS only, gate pooled / ignored, window boundary, objective = numbers only, argmax-only trigger, no fixB, NOT FINAL ignored, gate exclusion ignored, no none lower edge, A6 floor / inactive-numbers check dropped, s4 = held_fresh_nonid, dim = no_negtwin_nonid, numbers = b33 only (caught by real_known), data identity dropped |
| select -> picks -> generator | plumbing.py / plumbing.out | **PLUMBING PASS 16/16** (planted worlds: C renders exactly 64; L1x renders exactly the asked cells; --verify; written once; writable / tampered / GATE / NOT FINAL files refused by the generator; data mismatch refused; nothing written in runs/pt6comb) |
| the smoke's pass rules | smoke_check_test.py / smoke_check_test.out | **SMOKE CHECK TEST PASS 7/7**: a positive control built from the real tiny runs passes; each negative control fails exactly its rule (a prior on after the window, a CKA not above none, LM share 0.3, data mismatch, an extra failed check, a metric off its series) |
| sbatch --test-only | make_jobs_pt6comb.py --smoke / --part L1 --test-only / sbatch_test_only.out | accepted (b200_tandon_prem, gb002); nothing submitted by the test |
| composition A stays a parameter | comp_a_render.py / comp_a_render.out | **COMP A RENDER PASS 33/33** (X = landmarks / colours / compass / scalar_adj: every arm renders with a supplied placeholder reference, --domains in the trainer's order, months' 4 formats; --submit refused for A) |
| submission script (print) | submit_pt6comb.sh --part smoke / L1 / C (no --go) / submit_print.out | smoke rc 0 (1 job), L1 rc 0 (24 jobs, 'the GPU smoke has not passed yet: --go refuses'), C refused (no picks file, rc 3); quota 95.00%; self-test 46/46 each time; the misalign B200 note printed |

## 8. Audit status (experiment-audit; minimal mode for the build)
ii (asserts): the trainer's own, every job's input checks and CHECK_PY_6, the generator's render assertions (every flag, every
other flag at its default), the loader's identity / A6 / integrity / double paths, a6_guard6. iii (known answers): the self-test,
real_known (published numbers reproduced), loader_test, check_py6_test, the mutation test, smoke_check_test. vii: every job's
dry_inputs.txt (decoded training / eval rows with the sites marked; refs_B/dry_*.log). i: the metric double paths (section 5).
**v (fresh adversarial review): launched right after the smoke submission, in parallel (section "GPU smoke").** iv: not applicable
before results.

## 9. Cost, wall clock, disk (make_jobs_pt6comb.py --plan)
Planned 0.45 B200-h per run (ptdom's stage-1 medians A 0.46 / B1 0.35; the smoke calibrates).
| part | runs | B200-h |
|---|---|---|
| L1 | 24 | 10.8 |
| L1x | 0-12 (all6 8e-3, none 1.6e-2 / 1e-3 = fixB) | 0-5.4 |
| C | 64 | 28.8 |
| smoke | 1 job, 2 bodies x 8,000 steps | <= 1 |
| total | 88-100 runs | 39.6-45.0 (+ the smoke) |
Wall clock with 8 chains: L1 ~1.4 h, L1x <= 0.7 h, C ~3.6 h, plus queue waits. Disk: no saved models; run dirs ~1-2 MB each; the
smoke job deletes its two resume checkpoints (~0.6 GB each) after its check. /scratch quota 95.00% at the build (myquota);
submit_pt6comb.sh refuses at >= 97%.

## 10. Submission (submit_pt6comb.sh; the build ran only the smoke)
1. `bash runs/pt6comb/submit_pt6comb.sh --part smoke --go` (done by the build, below); read runs/pt6comb/smoke/smoke_check.out.
2. `bash runs/pt6comb/submit_pt6comb.sh --part L1 --go` (24 runs; refuses unless smoke_check.out says SMOKE PASS).
3. After every L1 job has finished with check.json: `python runs/pt6comb/select_pt6comb.py --after L1` -> picks_after_L1.json (once,
   chmod 444): stage L1x or C; GATE -> rc 2 (the coordinator decides).
4. If asked: `submit_pt6comb.sh --part L1x --go`; then `select_pt6comb.py --after L1x` -> picks_after_L1x.json (NOT FINAL -> rc 2).
5. `submit_pt6comb.sh --part C --go` (64 runs, seeds 8-15).
6. `PYTHONPATH=llmadd_torch/pylib python runs/pt6comb/readout_pt6comb.py --json runs/pt6comb/readout.json > runs/pt6comb/readout.txt`.
submit_pt6comb.sh refuses unless: no HOLD / HOLD_<part>; /scratch < 97%; the snapshot and frozen shas verify; the read-out self-test
passes (dated log); the smoke passed (L1 / L1x / C); `--check` renders the part; no job of the part was submitted before; L1x / C:
the picks file is read-only and `select_pt6comb.py --verify` re-derives it. Kill switches: `touch runs/pt6comb/HOLD` or
`runs/pt6comb/HOLD_<smoke|L1|L1x|C>`. Re-runs: `make_jobs_pt6comb.py --part X [--picks f] --rerun --print`, then `--rerun --submit`.

## 11. GPU smoke (one job; not part of any result; pass rules registered here before it ran)
- Job `llmadd_pt6comb-smoke` (runs/pt6comb/smoke/job_pt6comb-smoke.sbatch 5ffed0b2da39; B200, account 520 priority,
  preemption=yes, 1 h): two bodies run in turn with bash, each = the pt6comb job body of its cell (the real schedule, `--stop_after
  8000`: 33 evals 0..8000, on-window 1000..7250 (26), after the window 7500 / 7750 / 8000; no saved model): all6@2e-3 s0
  (body f663232a54c8) and none@2e-3 s0 (body f357a0861aad); then smoke/smoke_check.py; then its two resume checkpoints are deleted.
- Pass rules (smoke_check.py, all required): SM1 each body's CHECK_PY_6 failed exactly the truncation checks (all6: 'finished: 75
  evals' and 'prior_on_frac =='; none: 'finished') and the schedule is 33 evals with 26 on-window; SM2 each of the six priors in all6
  is on exactly at the 26 on-window evals and never after (prior_on_domains, gnorm_prior_<d> at exactly those evals, its applied-step
  count unchanged from 7500 to 8000), applied on >= 95% of the on-steps; SM3 each prior fits its own target at its own sites: the
  eval-prompt CKA to the target at step 7250 (numbers: cka_ops_helix3) minus none's >= 0.1; SM4 all6's LM share a / (a + b) median
  over the on-window evals >= 0.5 (the gate's quantity, clip coefficient printed); SM5 data_sha equal at every eval, step-0 lm = the CPU
  reference in both, |lm(all6) - lm(none)| <= 0.05 before the window; SM6 projected full run <= 2.5 h and GPU memory < 170 GB; SM7 the
  read-out's six series run on the real GPU JSONs and the numbers / comparison series equal the trainer's metrics.
- Submission: see "Smoke submission" below.

## 12. Caveats (stated; no rule reads them)
- **Tier L**: the comparison targets carry the answers (F100); the S4 rotation matrices determine the whole group table and the dim
  exponent vectors the answers (F88: "rescue_standard (c) not met"). HELPS reads "given the structure, the model uses it".
- **Tied embeddings**: s4 / dim items are single nonce tokens; the prior's site token is the read-out candidate's own (tied) output
  row (as ptdom A4.1 for single-token items), so the prior can move the answer rows directly.
- **S4 without a prior is learnable at high lr** (F88 A7 / A8.3: none@1.6e-2 learns S4 on some seeds): HELPS on s4 is against none@N
  chosen by mean6, not none's best S4 lr.
- **Shared comparator**: one none@N cell serves all six HELPS tests and the six u's: the six verdicts are correlated.
- **u and the lr**: u = only-m@2e-3 - none@N is lr-confounded when N != 2e-3 (Pending 2).
- **The numbers prior and months**: months was dropped (F76 / F88 tier-E drop with the numbers prior); clock / weekdays are absent.
- The LM-share gate uses norms (ptdom A2.1), not AdamW's update shares (F100's caveat).

## 13. Pending decisions (待协调者定; nothing here changes the design without a decision)
1. **NO INTERFERENCE when only-m does not beat none (mean u <= 0)**: the literal margin -0.5 x mean u is then >= 0, so the first clause
   asks all6 to be ABOVE only-m; the p clause still applies (self-test V13: a significant h of -0.46 with u = -1 reads INTERFERENCE).
   Implemented literally (the report flags such a manifold). Alternative: margin = -0.5 x max(mean u, 0).
2. **u's comparator**: none@N (implemented; lr-confounded when N != 2e-3). Option: add none@2e-3 at seeds 8-15 (+8 runs, 3.6 B200-h)
   when N != 2e-3, for a same-lr u (and the same-lr all6 - none row when P = 2e-3).
3. **all6's lower edge**: no lower extension (ptdom2 A1.1 (i); conservative for HELPS and for NO INTERFERENCE). rb6's all6 best was
   1e-3 with 5e-4 below it lower (F88 A7). Option: an all6@5e-4 extension when its pick / argmax is 1e-3 (+4 runs).
4. **The s4 / dim primary metrics**: s4_hard area (F88 A8.3's confirmatory metric) and the strict dim set (rb6's main dim read-out),
   both areas. Alternatives: the trainer's s4_readout (held_fresh_nonid, which a Latin table-filler partly solves) / dim_readout
   (no_negtwin_nonid); both are printed as secondary lines.
5. **The gate at the confirmation per half** (the inherited ptdom A4.5 / ptdom2 A1.7 rule) vs a single 8-seed median (the literal
   "LM share >= 0.5"). Implemented: per half (stricter).
6. **GATE / NOT FINAL**: no confirmation is rendered (select exits 2); the coordinator decides (run C anyway, labelled
   lr-confounded; or one more lr step). Implemented as the stop.
7. **The only-<m> arms' gate on seeds 8-15** is descriptive (printed per arm); a failing only-m would make its h / u comparisons
   "prior-dominated": no rule reads it.
8. **B200 sharing**: runs/ptdom/NOTIFY_misalign_when_b200_free.txt records the promise to tell the misalign session before
   manifolds use B200; the build's smoke is one <= 1 h job; L1 / C would occupy 8 chains (the coordinator tells misalign).
9. **Composition A** (numbers + months + cities + countries + people + X) stays a parameter (renders and passes the generator's
   assertions with a supplied reference; no CPU references, not submittable: the generator refuses --submit for A).
10. **Holm**: reference column only (the user's direct rule); the headline says "uncorrected".
11. Same-lr all6 vs none on the confirmation seeds exists only when P = N (otherwise the LM rows are lr-confounded; seeds 0-3 carry
    the same-lr LM rows).

## Smoke submission (the build, 2026-11-10 ~17:08 EST; the pass rules of section 11 were written before it ran)
- `bash runs/pt6comb/submit_pt6comb.sh --part smoke --go` (rc 0; tests/submit_go_smoke_20261110.log): **job 22595628**
  (llmadd_pt6comb-smoke; runs/pt6comb/smoke/SUBMITTED_pt6comb-smoke). Gates passed: quota 95.00% < 97%, snapshot, self-test 46/46,
  --check, nothing of the part submitted before. No other pt6comb job was submitted; no job was cancelled.
- Job file / bodies as registered: job_pt6comb-smoke.sbatch 5ffed0b2da39, body all6 f663232a54c8, body none f357a0861aad (a
  submitted smoke's bodies are never rewritten by the generator).
- Read it: `sacct -j 22595628 -o JobID,State,Elapsed,NodeList`; `tail -n 40 llmadd_torch/logs/llmadd_pt6comb-smoke_22595628.out`;
  `cat runs/pt6comb/smoke/smoke_check.out` ('SMOKE PASS n/n' is what submit_pt6comb.sh --go requires for L1 / L1x / C); the bodies'
  logs runs/pt6comb/smoke/body_<name>.log and check.json / check.txt in runs/pt6comb/smoke/<name>/.
- experiment-audit v: a fresh adversarial review (general-purpose sub-agent, read only) was started right after the submission; its
  report: /scratch/yh6210/tmp_audit/pt6comb_audit/audit_build.md.

## Amendment A1 (the build, 2026-11-10, after the fresh build audit; before any pt6comb result; reporting layer only)
The fresh adversarial build audit (general-purpose sub-agent, read only; /scratch/yh6210/tmp_audit/pt6comb_audit/audit_build.md, its
checks chk1.py / chk2.py / sim_noint.py there) found **no blocking item**. It verified: the four metric definitions, the series-eval
alignment, final3, the window split and FLOOR = this AUDIT; the gate's b includes the numbers prior's gradient (pretrain_range.py:2221);
the verdict formulas, Holm, the per-half gate, the refusals and every selection rule = section 3.4 / 4, with no path by which seeds
8-15 reach a selection; rainbow6_summary.toy_tables = the snapshot's toy_struct tables; no cross-domain leakage (no nonce token in an
item name, no digit in an item name, one shared site token 'ama' across domains); the frozen files' shas / mtimes unchanged; CHECK_PY_6's
seven edits open no hole (tampers caught). Its important items, verified by the build at the cited lines, and what was done:
- **I1 (h is lr-confounded when P != 2e-3; the 2e-3 of s4 / dim's single priors has no prior support: rb6 ran them at 1e-3)** —
  readout_pt6comb now labels every contrast that carries an lr difference ("lr-confounded contrasts: g / h / u"); the design is
  unchanged: **Pending 12** (all6@2e-3 at seeds 8-15 when P != 2e-3, +8 runs) and **Pending 13** (the single priors' lr).
- **I2 (NO INTERFERENCE has low power at 8 seeds; the report did not say which clause met it)** — the report now prints per manifold
  "NO INTERFERENCE by margin / p / margin+p", h's 95% t interval, and "by p alone = 'interference not shown at 8 seeds'". The auditor's
  simulation with the real verdict function (cities-like noise): P(NO INTERFERENCE) 0.28 when all6 erases the single prior's whole gain,
  0.51 when it erases 75% (sim_noint.py; recorded, not re-run by the build). **Pending 14**.
- **I3 (the secondary read-outs were computed but not printed, although Pending 4 relied on them)** — now printed per manifold: g / h
  (and wins) on s4 held_fresh_nonid area, dim no_negtwin_nonid area, the comparisons' hard-subset final3. Self-test V21 / V22.
- **I4 (one none@N chosen by mean6 serves all six HELPS tests: each g carries none's own lr effect; P may be poor for s4 / dim)** —
  stated (section 12 extended here), labelled by the lr line; **Pending 15**.
Minors acted on: m3 the loader's divergence onset now = CHECK_PY_6's (lm, val_loss, geo, grad_prior, every geo_<d>); m7 the s4_hard set
of a seed in rb6's S4_HARD_PRE (seeds 8-15) must equal it (a double-path problem otherwise); m1 a FLOOR line > 95 is annotated
("holds almost unconditionally": people's line exceeds 100, as F100 item 5); m2 the numbers CKA column is labelled (training-batch
operands, not eval prompts). Recorded, not acted on: m4 a finished but invalid L1 run stalls the selection (a finished run is never
re-run; the coordinator decides, as ptdom A6); m5 b is the norm of the vector SUM of the six weighted gradients, which can cancel (the
gate is looser than per-prior norms; ptdom's quantity kept); m9 none's mean6 may keep rising with lr, so NOT FINAL is a likely outcome
(Pending 6); m10 --allow_partial only (a missing hits eval is a refusal reason in the strict read-out).
Files (sha256[:12]): readout_pt6comb.py c2e4a855f677 -> see the table below; tests re-run below. The GPU smoke (submitted before A1)
imports readout_pt6comb only for SM7's series_of, which A1 does not change.

### Pending decisions added by A1
12. **h's lr**: when P != 2e-3, add all6@2e-3 at seeds 8-15 (+8 runs, 3.6 B200-h) so that h (and u) are same-lr contrasts; or keep and
    read NO INTERFERENCE as lr-confounded.
13. **The single priors' lr**: 2e-3 for all six (the design); rb6's single priors (incl. only-s4 / only-dim) ran at 1e-3 and all6's best
    there was 1e-3. Option: only-s4 / only-dim at 1e-3 instead (or both lrs).
14. **NO INTERFERENCE's power**: keep the rule (the user's direct rule) and print the clause + interval (done), or require the margin
    clause (drop "or p >= 0.05"), or add seeds.
15. **HELPS against one none@N**: keep (the design), or add a descriptive same-lr g row per manifold where P equals a none lr that ran
    on seeds 8-15.

### A1 files and tests (sha256[:12]; re-run on the A1 code; logs overwritten in place)
| file | build | A1 |
|---|---|---|
| runs/pt6comb/readout_pt6comb.py | c2e4a855f677 | fad2b11fe4ae |
| make_jobs_pt6comb.py, select_pt6comb.py, submit_pt6comb.sh, smoke/smoke_check.py, CHECK_PY_6, job files | as section 6 | unchanged (344ab0352dee, 75b4f0c00915, 1abd11924fcf, 92fa2b4f7bd4) |
Self-test **48/48** (+ V21 the report's clause / interval / lr-label / secondary lines, V22 the secondary g); mutation **25/25 caught**;
real_known 21/21; loader_test 11/11; plumbing 16/16; smoke_check_test 7/7; check_py6_test 13/13 (dead code removed from the test).
Smoke job 22595628: still PENDING at the time of writing (b200_tandon_prem busy: ~215 pending jobs, both nodes occupied).

## Amendment A2 (coordinator's decisions on §13, 2026-11-10, before any pt6comb result; the smoke 22595628 is unaffected)
**Where anything above disagrees, A2 holds.** Nothing was submitted or cancelled by A2.
1. **The same-lr interference contrast.** Part C always also runs **all6@2e-3 and none@2e-3** at seeds 8-15; when P = 2e-3 or N = 2e-3
   that cell IS the existing one (merged, never run twice): C = none@N, all6@P, none@2e-3, all6@2e-3, six only-<m>@2e-3 minus duplicates
   = 64 (P = N = 2e-3) / 72 (one of them 2e-3) / 80 runs (make_jobs_pt6comb.c_cells = readout_pt6comb.c_cells; the generator asserts the
   picks file's C list equals it). The interference contrast is now **h = all6@2e-3 - only-m@2e-3** with **u = only-m@2e-3 -
   none@2e-3**: all six single priors at 2e-3, so h and u carry no lr difference. all6@P - only-m@2e-3 is printed only, labelled
   "lr-confounded" when P != 2e-3. **s4 / dim's single priors also run at 2e-3**: rb6 / F88 ran its single priors (only-s4, only-dim)
   at 1e-3; 2e-3 here makes every interference contrast same-lr (supersedes Pending 12 / 13).
2. **u <= 0**: the margin clause (-0.5 x mean u) does not apply; the one-sided t clause alone decides (H1: mean h < 0; p >= 0.05 ->
   not detected), and the line prints "the single prior shows no gain itself (mean u <= 0)" (supersedes Pending 1). Note: with
   u <= 0 the literal margin was >= 0, and mean h >= 0 already implies p(h < 0) >= 0.5, so the verdict WORD is the same under both
   readings; what changes is the clause reported (always 't' here) and the note. The self-test checks the attribution (A2-4, A2-6).
3. **HELPS unchanged**: all6@P - none@N, N by mean6 (section 3.4); none NOT FINAL -> ptdom2 A1's extension handling (unchanged).
   Printed beside each manifold (descriptive): the same-lr gain all6@2e-3 - none@2e-3, its mean, bootstrap 90% interval and wins
   (supersedes Pending 15).
4. **Wording**: the conclusion never says "proven not to interfere". The verdict word is **"interference not detected"** (written
   「没有测出干扰」 in Chinese text), vs **INTERFERENCE**; the headline reads "k of 6 manifolds HELPS with no interference detected (at the
   same lr 2e-3)". h and u are printed with their means and **bootstrap 90% intervals** (readout_ptdom.boot_ci), h also with its 95% t
   interval and the clause that met the rule ("by margin / p / margin+p"; "by p alone = not detected at 8 seeds").
5. Every other pending item keeps the build's default (Pending 2-11, 14); the coordinator tells misalign about B200.

### A2 verdict rule (replaces section 4's NO INTERFERENCE line)
Per manifold m, seeds 8-15: h_s = all6@2e-3 - only-m@2e-3, u_s = only-m@2e-3 - none@2e-3. **Interference not detected** iff (mean u > 0
and mean h >= -0.5 x mean u) OR the one-sided paired t p (H1: mean h < 0; df 7) >= 0.05; when mean u <= 0 only the t clause. Else
**INTERFERENCE**. HELPS and the headline count as before (HELPS and interference not detected), uncorrected; Holm as the reference.

### A2 cost (make_jobs_pt6comb.py --plan)
C 64-80 runs = 28.8-36.0 B200-h (72 when one of P / N is 2e-3); total 88-116 runs = 39.6-52.2 B200-h (+ the smoke).

### A2 tests (re-run; logs in place)
New self-test cases: C1 c_cells for (P, N) = (2e-3, 4e-3) / (4e-3, 8e-3) / (2e-3, 2e-3) / (4e-3, 2e-3) -> 9 / 10 / 8 / 9 cells, none
twice; C2 decide's C list; A2-1 h is all6@2e-3 - only-m (a planted world where all6@P - only-m is +1.4 but the same-lr h is -3.6 ->
INTERFERENCE); A2-2 u is against none@2e-3 (u +2.3, margin -1.15, h -1.3 -> INTERFERENCE; against none@N it would have been u +5.3 and
not detected); A2-3 the same-lr gain row; A2-4 / A2-6 u <= 0 -> the t clause only, the note; A2-5 the report's 90% intervals, the note,
the same-lr rows; V13 / V21 / S1 updated. render_diff T0: part C renders the pair once (72 / 80 / 64 / 72 runs, every name unique).
Mutation test: + 6 A2 mutations (h against all6@P; u against none@N; the margin clause when u <= 0; no same-lr pair in C; the pair not
merged; HELPS on the same-lr gain).
Results: **SELFTEST PASS 56/56**; **MUTATION READOUT PASS 31/31** (25 + 6 A2; two old mutation strings re-pointed to the rewritten
lines); RENDER DIFF PASS 5/5 (T0 new); PLUMBING 16/16 (P3 / P4: 72 runs); LOADER 11/11; REAL KNOWN 21/21; SMOKE CHECK TEST 7/7;
CHECK_PY_6 TEST 13/13; DRYBLOCK 29/29. The 24 L1 job files are byte-identical to the build's (CHECK_PY_6 and the job bodies unchanged
by A2). Files (sha256[:12]): readout_pt6comb.py 42d7b6158b04, make_jobs_pt6comb.py 08e197092b2c,
select_pt6comb.py 75b4f0c00915, submit_pt6comb.sh 1abd11924fcf, smoke_check.py 92fa2b4f7bd4.

## Amendment A3 (coordinator's decisions on the A2 audit tests/audit_a2.md, 2026-11-10; before any pt6comb result and before any select)
The A2 audit found no blocking item (its two paths: 4,800 verdicts 0 disagreements; L1 / smoke files byte-identical; no selection
reads seeds 8-15). **Where anything above disagrees, A3 holds.** Nothing was submitted or cancelled; no picks file exists yet.
1. **(I1) all6@2e-3 is gated too.** The A2.1 cell gate (median over runs of the run's on-window median a / (a + b) < 0.5, or an
   unverifiable run) at all6@2e-3, on seeds 8-11 and 12-15 separately (the same rule as all6@P; one cell when P = 2e-3). Failing ->
   every manifold's interference word reads **"AUX GATE FAILED at all6@2e-3"** (no "interference not detected" printed); HELPS is
   judged as usual; the headline withholds the interference words and lists the HELPS manifolds. all6@P failing -> all6 GATE as before.
2. **(m1)** The selection note "PRIOR AT ITS LOWEST LR" now reads "conservative for HELPS; the interference contrast is at 2e-3
   whatever P is, A2" (it is written into the picks file: changed before any select; select --verify re-derives it, tests/plumbing.out).
   Also the module docstring's headline line and submit_pt6comb.sh's C hint ("64-80 runs").
3. **(m3)** A missing / invalid run of the same-lr pair (none@2e-3 / all6@2e-3 where not merged with none@N / all6@P) withholds only the
   interference words and the same-lr rows ("WITHHELD (the same-lr pair is incomplete)", res['missing_pair']); the strict read-out
   needs none@N, all6@P and the six only-m@2e-3 cells (a missing one still refuses).
4. **(m2, m5, m6)** The table prints the all6@2e-3 and none@2e-3 levels beside all6@P / none@N / only-m; the header says "HELPS is
   judged at all6@P vs none@N; interference at the same lr 2e-3" and the headline names both lrs; a same-lr LM cost line (all6@2e-3 -
   none@2e-3: in-training val loss and bpb x 1e3, seeds 8-15) and the all6@2e-3 gate line are printed.
5. **(m4)** Self-test cases added for the verdict-relevant mutations the A2 auditor's own set showed uncaught: A3-1 the all6@2e-3
   gate (P 4e-3) -> 'AUX GATE FAILED at all6@2e-3' in every table row; A3-2 the HELPS gate checks all6@P, not all6@2e-3; A3-3 / A3-4 the
   pair incomplete -> withheld, HELPS unchanged and not refused, while a missing all6@P still refuses; A3-5 no 'NO INTERFERENCE' anywhere
   in a report or a selection note, the judged-at line, the new levels / LM / gate lines; A3-6 the lr labels; A3-7 / A3-8 / A3-9 the
   three boundaries (mean h exactly at the margin -> by margin; mean u exactly 0 -> no margin; p exactly 0.05 -> not HELPS and not
   detected by p). Not added (description rows only): final3 / secondary h / u at P / N. Mutation test: + 9 A3 mutations (the three
   boundaries, the pair dropped from the missing list, the pair refusing HELPS, the table word, the gate's cell, the all6@2e-3 gate
   ignored, the stale note); the 3 A2 h / u mutation strings re-pointed to the rewritten line.
### A3 tests (re-run; logs in place)
**SELFTEST PASS 65/65** (+ A3-1..A3-9); **MUTATION READOUT PASS 40/40** (31 + 9 A3; the control passes); PLUMBING 16/16 (incl. --verify
agreeing with a freshly written picks file under the A3 note text); RENDER DIFF 5/5; LOADER 11/11; REAL KNOWN 21/21; SMOKE CHECK TEST
7/7. Job files unchanged (CHECK_PY_6 and the bodies untouched by A3). Files (sha256[:12]): readout_pt6comb.py 0015aa97e8c7,
submit_pt6comb.sh 3b23f4a63174, make_jobs_pt6comb.py 08e197092b2c, select_pt6comb.py 75b4f0c00915.

## Smoke result (job 22595628, gb010, 00:20:32; read 2026-11-10 after A3)
smoke/smoke_check.out: **SMOKE PASS 21/21** (pre-registered rules of section 11). Both bodies exited 4 by design (the truncation checks
only, SM1); the six priors in all6 were on exactly at the 26 on-window evals and never after, applied on 100% of the on-steps (SM2);
every prior fit its own target at step 7250 well above none (SM3); all6's LM share a / (a + b) median 0.752 over the on-window evals
(min 0.394; b > a at 1 of 26; clip < 1 at 0) (SM4); identical data, step-0 lm = the CPU reference, |dlm| <= 0.013 before the window
(SM5); projected full run 0.32 h (all6) / 0.30 h (none), GPU memory max 32.6 GB (SM6); the read-out's six series run on the GPU JSONs
(SM7). The resume checkpoints were deleted (smoke/ 3.2 MB). The smoke's read-out values are not results (seed 0, 8,000 steps).

## Coordinator (2026-11-10 ~21:00 EST): A3 not re-audited; L1 submission
A3 applies the A2 audit's own findings (I1, m1-m5) to the read-out and selection code, and adds 9 self-test cases and 9
mutations (self-test 65/65, mutations 40/40). The L1 job files are unchanged. A3 is therefore not sent to another fresh audit.
The GPU smoke 22595628 passed (SMOKE PASS 21/21). The misalign session was told about the B200 use on 2026-11-10.
L1 (24 runs, seeds 0-3) is submitted with `submit_pt6comb.sh --part L1 --go`.
