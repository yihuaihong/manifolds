# toystruct: toy grokking with a structure prior, S4 (cube rotation group) and dimensional analysis (2026-10-16)

User decision 2026-10-16: run BOTH toy experiments now (S4 = the cube rotation group, the structure behind mental-
rotation tasks; dimensional analysis), small and decisive, before any 53M pretraining version. Design / critique:
~/Research_Projects/LLM_addition_review/notes/abstract_priors_design_priors.md (D3, D5) and
abstract_priors_critique_final.md ("建议的第一步实验", ranked priors 1 and 3). Builds on F45 / F59 (toy_grok: single
circle early = harmful; multi-frequency target early = best, ~4x faster grokking).

Question: when a one-layer transformer learns a closed algebra over nonce items from a fraction of its question
table, does a prior that puts the items' embeddings into the algebra's geometry (the group's faithful irreps; the
exponent lattice) make it learn faster or at all, and is the gain specific to that structure (vs a random geometry of
the same rank and spectrum, a non-faithful quotient, a same-information non-additive code)?

Revision 2 (2026-10-16, after the independent audit, before any main-run result; main seeds 0-4 never trained beyond
the 300-step smoke): (1) a continuous CO-PRIMARY (AUC = mean held-out accuracy over the budget) next to t_grok, each at
p < 0.025, plus explicit "floor: uninformative" and "not shown at n=5" verdicts; (2) dim split by RELATION CLASS
(project rule: symmetry-clean splits), so the dim none pilot was rerun; (3) per-arm windows (`R@always`) and a rank-6
control `specrand_oh` exist as OPTIONS (off by default: user's call); (4) rule (d) reading narrowed; (5) atomic JSON
writes, no per-process summary. Audit table and open decisions: section "Audit response" at the end.

## Code and provenance
- Trainer: llmadd-burst `tools/toy_struct.py` (new; `tools/toy_grok.py` untouched). Tests: `tools/test_toy_struct.py`.
  Both are UNCOMMITTED working-tree files (repo is public; no commit / push by the agent): the job code is the snapshot
  `runs/toystruct/code/tools/` with `code/COMMIT` = HEAD + sha256[:12] of every copied file; the generator refuses
  to write jobs if the snapshot differs from the repo file. Snapshot revision 2 (2026-10-16, refreshed after the
  audit fixes; nothing had been submitted): HEAD 259e2a3 (another session's commit; tools/toy_grok.py unchanged),
  toy_struct.py a099a1edaece, test_toy_struct.py 1533ad98f6c0, toy_grok.py bec2858158d3 (unchanged since dabaad0).
  Revision 1 was HEAD 4a29104, toy_struct.py 1d39b2e91473. The revision-2 pilots ran from the frozen copy
  preflight/pilot2_code (063e3257407b), identical to a099a1edaece in every line before the summary section.
  Every job first runs `--check` (CKA matrix, closure, all build asserts) into out/<cell>/check.txt.
- Generator: `llmadd_torch/make_jobs_toystruct.py` (writes `job_<task>-f<frac><option>.sbatch`, does not submit;
  `--submit`; options `--always`, `--s4fracs`, `--oh`, `--posctl` change the job tags, and a rerun deletes unsubmitted
  job files of another option set).
- Model / optimiser = toy_grok exactly: one layer, d 128, 4 heads, MLP 512, no LayerNorm, causal SDPA, readout at the
  last token; untied W_U; AdamW lr 1e-3 (10-step linear warmup), betas (0.9, 0.98), wd 1.0, full batch, fp32.
- Same seed = same init (torch.manual_seed(seed) right before the model is built) and same split for every arm, so
  arms are paired by seed.

## Tasks
| task | items (tokens) | question | N | train (0.3 / 0.6) | held out (0.3 / 0.6) | closure (0.3 / 0.6) |
|---|---|---|---|---|---|---|
| mod (known answer) | 0..112, '=' 113 | (a, b, =) -> a+b mod 113 | 12769 | 3830 / – | 8939 / – | – |
| s4 (question split) | 24 rotations 0..23 (fixed enumeration order, no names), '=' 24 | (a, b, =) -> c, R_c = R_b R_a ("a then b") | 576 | 172 / 345 | 404 / 231 (fresh 379-391 / 205-210) | 1.00 / 1.00 (seeds 0-4) |
| dim (class split) | 27 exponent vectors in {-1,0,1}^3, mul 27, div 28, '=' 29 | (a, op, b, =) -> item e_a +- e_b, in-box only | 686 (343 + 343) = 172 relation classes | 202-204 / 410-412 | 482-484 / 274-276 (all fresh) | 1.00 / 1.00, nullity 3 (seeds 0-4) |
- S4 convention: R acts on column vectors; "a then b" = R_b R_a (test: a = 90 deg about z, b = 90 deg about x sends x
  -> y -> z). Asserted at build: 24 distinct det +1 signed permutation matrices, closure, associativity on the whole
  table, identity, unique inverses, Latin square, non-abelian (456 / 576 cells with gh != hg; 12 odd elements), the
  action on the 4 body diagonals is a faithful homomorphism onto S4, its sign = det|R|, V4 = {I, 3 face half-turns} is
  the kernel of S4 -> S3 with 6 cosets of 4 and the quotient is a homomorphism.
- Split (`--split`, written explicitly into every job): s4 = question split, numpy default_rng(seed).permutation(N),
  first int(frac N) train (toy_grok's split). dim = CLASS split: the same draw over the 172 relation classes (e_x + e_y
  = e_z is stated by x*y, y*x, z/y, z/x; 171 classes of 4 + {0*0, 0/0}), first int(frac 172) classes train, so no
  phrasing of a held-out relation is ever trained. Why (audit 2026-10-16, re-measured): with the question split, 64-72 %
  (0.3) and 87-99 % (0.6) of dim's held-out questions had a class-mate in training, so commutativity + the mul/div
  inverse alone could reach ~0.95 held-out accuracy; the old pilot's none (dim 0.6) "grokked" at or just above that
  ceiling (held-out at the stop 0.953 / 0.953 / 0.960 / 0.971 vs class-mate ceiling 0.956 / 0.956 / 0.942 / 0.964,
  seeds 100 / 101 / 103 / 104; the censored seed 102 0.920 vs 0.884). s4 keeps the question split: the
  operation is not commutative, the only twins are 13-26 commuting cells per held-out set, and using one requires
  knowing that the pair commutes (group structure, not a generic symmetry); they are reported as test_fresh / test_comm.
  Asserted: no duplicate question, train and held-out disjoint by content, no class straddles train / held-out (class
  mode), every item is an input and an answer somewhere in training. test_fresh = held-out accuracy on the questions
  with no class-mate in training (fresh mask checked against an independent enumeration of each relation's phrasings).
- Closure = share of held-out answers derivable from the training questions by the algebra. s4: fixpoint of
  associativity ((gh)k = g(hk), either side derives the other; every derived cell asserted equal to the true table);
  secondary with Latin-square cancellation (same 1.00). dim: the held-out relation row e_a +- e_b - e_c lies in the
  row space of the training relations (matrix rank test; the least-squares residual test must agree, asserted);
  nullity of the training relation matrix = 3 means the exponents are identified up to GL(3). Known answers (tests):
  one missing s4 cell is derived; the identity row alone derives nothing; one dim relation derives exactly its
  triangle (b*a, c/a, c/b). Closure 1.00 at both fractions = the held-out set is answerable by the algebra and no
  target is needed to answer it (injection rate 0: the targets hold item coordinates only, never answers).

## Arms (prior = lam * (1 - linear CKA(W_E[items], target)), lam 1, on for steps [0, 2000); centred raw features)
| task | arm | target | centred rank | what it tests |
|---|---|---|---|---|
| both | none | – | – | baseline |
| s4 | R | 3x3 rotation matrix flattened (9) | 9 | faithful 3-d irrep (= standard x sign of S4) |
| s4 | R_sign | R x sign(body-diagonal permutation) | 9 | the other faithful 3-d irrep; CKA(R, R_sign) = 0 exactly (Schur) |
| s4 | absR | abs(R) = axis permutation (9 cols) | 4 | S4 -> S3 quotient (kernel V4), non-faithful: constant on V4 cosets; CKA 0 with R and R_sign |
| s4 | specrand | random 9-d, centred singular values of R (all sqrt 8), Haar singular vectors orthogonal to 1, draw rng 1000+seed | 9 | same rank and spectrum, no structure |
| s4 | shuf | R with the items permuted: rng 124+seed, the permutation minimising max CKA to {R, R_sign} of 3000 | 9 | same Gram values, wrong assignment |
| dim | expo | e (3 cols, raw) | 3 | the additive lattice |
| dim | quot | e without exponent 3 (2 cols) | 2 | quotient: items differing only in e_3 coincide |
| dim | onehot | one-hot of each coordinate's value (9 cols) | 6 | same information, not additive (contains e linearly plus abs(e_i)) |
| dim | specrand | random 3-d with the singular values of e (all sqrt 18), rng 1000+seed | 3 | same rank and spectrum |
| dim | shuf | e with the items permuted: rng 124+seed, first with CKA to e < 0.15 | 3 | wrong assignment |
| dim (OPTION `--oh`) | specrand_oh | random 6-d with onehot's centred singular values, rng 2000+seed | 6 | rank control for onehot (rule (d)) |
| s4 (OPTION `--always`) | R@always, R_sign@always, specrand@always | as R / R_sign / specrand, prior on for the whole budget | 9 | rescue capacity of a persistent prior (positive control for the window) |

Arm name `<target>@always` / `<target>@<t0>-<t1>` overrides the cell window for that arm only (tested: prior_on per eval).

### CKA matrices between targets (seed 0; last row / column = the seed-0 initial W_E[items], the chance level of every
"CKA to target over time" curve)
S4:
```
                 R    R_sign      absR  specrand      shuf  W_E_init
       R     1.000     0.000     0.000     0.389     0.330     0.566
  R_sign     0.000     1.000     0.000     0.395     0.344     0.562
    absR     0.000     0.000     1.000     0.245     0.420     0.368
specrand     0.389     0.395     0.245     1.000     0.404     0.571
    shuf     0.330     0.344     0.420     0.404     1.000     0.574
W_E_init     0.566     0.562     0.368     0.571     0.574     1.000
```
seeds 0-4: CKA(specrand, R) 0.389 / 0.457 / 0.352 / 0.322 / 0.418; shuf max CKA to {R, R_sign} 0.344 / 0.344 / 0.344 /
0.340 / 0.340 (CKA to R 0.330 / 0.344 / 0.344 / 0.326 / 0.340).
dim (specrand_oh is computed and logged in every arm, trained on only with `--oh`):
```
                 expo      quot    onehot  specrand specrand_oh      shuf  W_E_init
       expo     1.000     0.816     0.707     0.081     0.253     0.131     0.349
       quot     0.816     1.000     0.577     0.034     0.195     0.014     0.280
     onehot     0.707     0.577     1.000     0.150     0.248     0.204     0.457
   specrand     0.081     0.034     0.150     1.000     0.241     0.108     0.329
specrand_oh     0.253     0.195     0.248     0.241     1.000     0.205     0.438
       shuf     0.131     0.014     0.204     0.108     0.205     1.000     0.325
   W_E_init     0.349     0.280     0.457     0.329     0.438     0.325     1.000
```
onehot's centred column span CONTAINS expo's (e_i = [e_i = 1] - [e_i = -1]; asserted), rank 6 vs 3.
seeds 0-4: CKA(specrand, expo) 0.081 / 0.139 / 0.149 / 0.095 / 0.127; CKA(shuf, expo) 0.131 / 0.121 / 0.072 / 0.059 / 0.149.
Asserted analytically: CKA(expo, quot) = 2/sqrt 6 = 0.816; CKA(R, R_sign) = CKA(R, absR) = CKA(R_sign, absR) = 0.

Design notes forced by the numbers (not in the original design):
- S4 shuf "far from R" alone lands in R_sign (first version: CKA(shuf, R) 0.24 but CKA(shuf, R_sign) 0.52): a 9-d
  target in the 23-d centred space (= R 9 + R_sign 9 + 2-d irrep 4 + sign 1) cannot avoid both faithful irreps (the
  minimum of max(CKA_R, CKA_Rsign) is >= ~0.22). shuf therefore minimises the max over {R, R_sign}; best of 3000 =
  0.34. shuf is a "same values, wrong assignment" control, never a zero-structure null. Side effect (audit): pushed
  away from both faithful irreps, shuf leans toward the quotient (CKA to absR 0.36-0.42 vs specrand's 0.22-0.30):
  describe shuf as "partly quotient-like".
- specrand is a generic random 9-d subspace (expected CKA to R = 9/23 = 0.39), not filtered: rule (a) asks "the
  group's irreps vs a generic subspace of the same rank and spectrum"; specrand shares ~0.39 of R's Gram, so the test
  is conservative.
- An isotropic random init already has CKA ~0.57 with any rank-9 S4 target (0.35 with expo): CKA-over-time is read
  against the W_E_init row, never against 0.

## Train vs eval
| dimension | train | eval | same? |
|---|---|---|---|
| template / position | (a, b, =) or (a, op, b, =), answer read at the last token | same | yes |
| questions | s4: int(frac N) questions; dim: int(frac 172) whole relation classes | the rest (disjoint by content and, dim, by relation class: asserted) | no, by design: generalisation to held-out cells answerable by the algebra (closure 1.00), not by a twin |
| items / answers | every item as input and as answer (asserted) | same item set | yes |
| model state | train mode (no dropout, no norm layers: train == eval forward) | no_grad | yes |
| batch | full batch every step | all held-out questions every 100 steps | yes |
| precision / device | fp32, one GPU (L40S) or CPU | same process | yes |

## Metrics (one definition each; implementation tools/toy_struct.py:run_arm unless noted)
- test (held-out accuracy): argmax over the n_out item logits at the last position == answer, over ALL held-out
  questions (n 404 / 231 s4, 482-484 / 274-276 dim), every 100 steps and at the last step. Uniform chance 1/24 = 0.042
  (s4), 1/27 = 0.037 (dim); answers are not uniform, so every run also records chance_majority = held-out accuracy of
  always giving the most frequent training answer (main seeds: s4 0.022-0.032, below 1/24 because of the Latin square;
  dim class split 0.021-0.088, question split was 0.071-0.095). Floor reference = max(1/n_out, median chance_majority).
- test_fresh: the same on held-out questions with no class-mate in training (dim: all held-out; s4: minus the
  commuting cells whose swapped twin was trained).
- train: the same on all training questions. ce = training CE of the step's full batch BEFORE that step's update
  (toy_grok's logging); test_ce = held-out CE after the update.
- t_grok (CO-PRIMARY 1): first eval step with test >= 0.95; never reached = censored at the budget (+inf in the tests,
  ties with other censored runs; never averaged). t_mem: first eval step with train >= 0.99.
- AUC (CO-PRIMARY 2, revision 2): mean of test over every eval of the budget (B / 100 + 1 evals) = the normalised area
  under the held-out learning curve. It integrates speed and level, so it registers a partial rescue that never
  reaches 0.95 (which t_grok scores as a censored tie). toy_struct.py:auc; a run stopped at its grok (pilots only)
  carries the stop value forward; an incomplete main-run curve is an assert, not an AUC.
- t_0.5 (descriptive): first eval with test >= 0.5.
- final_test: test at the last step.
- cka_<target> for every target of the task, every eval: linear CKA(W_E[items], target) (same function as the loss);
  cka_init before step 0; geo = 1 - CKA to the arm's own target.
- coset_share (s4, every arm): among wrong held-out answers, share whose predicted item is in the correct V4 coset
  (same |R|; chance 3/23 = 0.130). drop_share (dim, every arm): among wrong held-out answers, share that differ from the
  truth only in exponent 3 (chance 2/26 = 0.077 under uniform errors; only approximate, since answers are not uniform).
  n_wrong reported next to both.
- test_comm / test_noncomm (s4: held-out cells with gh = hg vs not), test_mul / test_div (dim).
- summary (toy_struct.py:summarize, written before any result): censored medians, paired wins, the exact test below,
  the pooled shares, rule verdicts.

## Expected ranges (anything outside -> look before reading)
- none t_mem: a few hundred steps (F45 mod: 260); train acc 1.0 by ~1k steps in every arm except possibly absR /
  quot while their prior is on (coset-mates are pushed together, so their training answers conflict).
- CKA to own target at step 2000 >= 0.8 for every prior arm (smoke at 300 steps: see Tests); none stays near W_E_init.
- held-out test before grokking ~ chance to ~0.3; after grokking 1.0. A run with final train < 0.99 = optimisation
  failure: report it, do not drop it. S4 held-out accuracy of a pure memoriser is BELOW chance (pilot none: 0.00-0.04
  vs chance 0.042): the table is a Latin square, so a row's training answers are wrong for every held-out cell of
  that row. Below chance is the memorisation signature, not a bug.
- low-rank priors can slow memorisation while on (smoke, dim 0.6: expo train 0.55 at step 300 when every other arm is
  at 1.00; F45's rank-2 circle did the same).
- dim none lands near the onehot geometry, not the additive one (first pilot, QUESTION split, dim 0.6 at grok: CKA to
  onehot 0.77-0.94, to expo 0.43-0.63); that pilot's "grok" was mostly class-mate lookup (see Split), so this is a
  statement about the twin solution, to be re-read on the class split.
- absR / quot shares during the window: >> chance if the prior is read; none's shares ~ chance unless the network
  learns the quotient first.

## Controls
- Known answer (positive control for the trainer): --task mod == toy_grok.py for 200 steps (all logged values, 4 arms
  incl. a window that switches on at step 100 and a shuffled target); the same loop serves s4 and dim.
- Negative controls: specrand (same rank / spectrum), shuf (same values, wrong assignment), none.
- Specificity controls: R_sign (must match R if group structure, not the particular irrep, is what helps), absR
  (non-faithful quotient: predicts coset-only errors), quot, onehot.
- "Prior is read": CKA to own target rises above none's (asserted in the smoke); absR / quot error signatures.

## Budget (PRE-REGISTERED 2026-10-16, before the pilot ran)
- B(0.6) = 25,000 steps, B(0.3) = 50,000 steps, every arm, eval every 100.
- None pilot: arm none only, seeds 100-104 (disjoint from the main seeds 0-4), CPU 1 thread, both tasks x both
  fractions at those budgets, output preflight/pilot_none (not part of any result).
- Rule: if >= 1 pilot seed of none at a fraction groks in (B/2, B] (late, close to the budget), that fraction's budget
  doubles, once (so censoring does not fall inside none's grok-time distribution). Otherwise B stays (also when none
  groks in no pilot seed: that is the (c) regime; a longer budget is a separate decision for the user).
- If none groks by step 2000 in all pilot seeds at both fractions of a task, rule (a) cannot qualify for that task;
  this is reported to the user (a lower fraction would be the user's call), the fractions are not changed here.
- Pilot result (2026-10-16, --stop_at_grok, CPU 1 thread; preflight/pilot_none). The pilot ran across three
  revisions of toy_struct.py that differ only in log printing and the summary (training path identical). Its JSONs'
  code_sha (77bfeaf0f9f9 x 6, 04d854755e04 x 15) is the file at the END of each run: a provenance bug found here and
  fixed before the snapshot (code_sha is now taken at import):
  | cell | none groks | t_grok per seed 100-104 | held-out acc at the budget (ungrokked seeds) |
  |---|---|---|---|
  | s4 0.3 (B 50k) | 0/5 | – | 0.025, 0.015, 0.010, 0.025, 0.017 (flat from step 1000) |
  | s4 0.6 (B 25k) | 0/5 | – | 0.02, 0.04, 0.04, 0.03, 0.00 (flat) |
  | dim 0.3 (B 50k) | 0/5 | – | 0.28, 0.12, 0.06, 0.06, 0.10 (seed 100 still rising: 0.14 @10k, 0.20 @20k, 0.28 @50k) |
  | dim 0.6 (B 25k) | 4/5 | 7300, 2000, censored (0.92, plateau 0.89-0.93 from 2k), 1100, 6200 | median 6200 |
  Rule applied: no pilot seed grokked in (B/2, B] (dim 0.6 grok steps are all <= 7300 < 12,500) -> every budget
  stays: 25,000 (0.6) and 50,000 (0.3). Not every cell grokked by 2000 -> rule (a) can qualify.
  Reading for the user (not a rule change): (1) S4: the unaided toy never generalises at 0.3 or 0.6, so both S4
  fractions should qualify for (a) and (c) is live at 0.3; the risk is a FLOOR (every arm censored, nothing
  separable). A feasibility run of none at frac 0.9 (seed 100, 25k, not pre-registered, not a result) is in
  preflight/feasibility_s4_f0.9: see "Feasibility" below. (2) dim 0.3: none fails 5/5 but one seed is still rising at
  50k, so a (c) "cannot be learned" for dim is "not within 50k steps", not "never". (3) dim 0.6: none's median 6200
  qualifies (>= 4000) but 2/5 seeds grok by 2000: a narrow window for a speed-up.
  SUPERSEDED for dim (revision 2): those dim rows were measured on the question split, whose held-out set was mostly
  class-mate lookups (see Split). The s4 rows stand (s4 split unchanged).
- Dim none pilot, rerun on the CLASS split (revision 2; same budgets, same rule, --stop_at_grok, CPU 1 thread,
  frozen code preflight/pilot2_code, output preflight/pilot_none_dimclass). Pilot seeds = the first five >= 100 whose
  class split passes every build assert with closure 1.00 at 0.3 and 0.6: 100, 102, 103, 105, 106 (101: closure 0.917
  at 0.3, nullity 4; 104: item 2 never an input in training at 0.3). Main seeds 0-4 all pass (closure 1.00, nullity 3).
  | cell | none groks | t_grok per seed 100 / 102 / 103 / 105 / 106 | held-out at the budget (ungrokked) |
  |---|---|---|---|
  | dim 0.3 (B 50k) | 0/5 | – | 0.048 / 0.033 / 0.035 / 0.064 / 0.039 (max over the run 0.048-0.093: flat at chance) |
  | dim 0.6 (B 25k) | 5/5 | 2900 / 2500 / 6800 / 1100 / 4000 (median 2900) | – |
  Rule applied: no pilot seed grokked in (B/2, B] -> both dim budgets stay (25,000 / 50,000).
  Reading for the user (not a rule change): (1) on the clean split, dim 0.3 is the (c) regime (none flat at chance to
  50k in 5/5) and the feasibility run has expo rising to 0.73 by 8k vs specrand 0.04 (seed 100). (2) dim 0.6: none
  genuinely learns the lattice from 103 relations (fresh held-out 0.95 by 1100-6800 steps); the pilot median 2900 is
  BELOW the qualifying line 4000, so if the main seeds behave alike, (a) at dim 0.6 will read "untestable (none groks
  early)" and dim (a) rests on 0.3. Feasibility at dim 0.6 (seed 100): expo groks at 400, specrand at 5200.

## PRE-REGISTERED rules (2026-10-16, before any main-run result; implemented in toy_struct.py:summarize)
Comparison test T(a < b) at one fraction, on t_grok: PASS iff (i) censored median t_grok(a) < that of b, (ii) a groks
strictly earlier than b on >= 4 of the 5 seeds (paired by seed; both censored = no win), and (iii) the exact one-sided
two-sample permutation test of the Gehan / Mann-Whitney U (censored runs tied at the budget; all 252 relabelings of
5 + 5) gives p < .05. (A paired sign test reaches p < .05 only at 5/5; it is not used.)
Qualifying fraction: none's censored median t_grok (main seeds 0-4) >= 4000 = 2 x the window end ("none groks late or
not at all").
- (a) "Group structure helps" (s4) iff T(R < specrand) AND T(R_sign < specrand) both pass at the same qualifying
  fraction. "Linear structure helps" (dim) iff T(expo < specrand) passes at a qualifying fraction. Only one of R /
  R_sign passing = "irrep-specific, not group structure as such" (no claim). No qualifying fraction = (a) untestable
  here (not a negative). Reported with it, no claim alone: T(R < none), T(R_sign < none), T(specrand < none) (the
  "any consistent encoding" effect, F77 T4), T(shuf < none), T(absR < specrand).
- (b) "The prior was read" (s4) iff absR's coset_share pooled over the evals at steps 1000..2000 (11 evals, the
  second half of the window; weights n_wrong) is > 0.5 (chance 0.13) in >= 4 of 5 seeds, a seed counting only with
  >= 20 pooled wrong answers (fewer than 4 such seeds = undetermined); at each fraction separately. dim analogue:
  quot's drop_share > 0.5 (chance 0.077). Reported with it: none's, R's / expo's and specrand's shares at the same
  evals; if none's share is also > 0.5 in >= 4/5 seeds, (b) is read as "coset-structured errors are generic", not
  as evidence that the prior was read.
- (c) "Without the prior it cannot be learned" (within 50k steps) only if at frac 0.3 none fails to grok within the
  budget in >= 4/5 seeds while R (s4) / expo (dim) groks in >= 4/5. Reported with it: specrand and shuf in the same
  cell; if specrand also groks in >= 4/5, the claim narrows to "any consistent prior of this rank rescues it".
- (d) onehot vs expo (dim), each fraction: T(expo < onehot) passes = "additive geometry matters beyond the
  information"; T(onehot < expo) passes = "the information (or onehot's richer geometry) suffices, additivity is not
  what helps"; neither = "not separated at n = 5" (medians reported). Read only if expo or onehot passes T(. < none)
  or T(. < specrand) at that fraction; otherwise there is nothing to separate.
- Secondary, descriptive only: final_test, t_mem, CKA to own target at step 2000 and at the end, none's final CKA to
  R / R_sign / absR (which irrep the unaided network lands on), test_noncomm vs test_comm, test_mul vs test_div.
- No other arm, window, lambda, fraction or seed is added after results without being labelled post hoc.

## AMENDMENT to the rules (revision 2, 2026-10-16, after the audit; before any main-run result) -- these REPLACE v1 above
What was seen before amending: only pilot seeds (100-102) in feasibility runs, never a main seed beyond the 300-step
smoke. Trigger: at S4 0.6, R with the [0, 2000) window plateaus / rises slowly (held-out 0.32 / 0.71 / 0.72 at 8k,
seeds 101 / 102 / 100) while specrand stays at 0.01-0.03; t_grok@0.95 scores that as a censored tie (p = 1).
- T(a, b) now has two CO-PRIMARIES, both on the same paired seeds; a comparison PASSES if EITHER passes:
  - t_grok: as v1 (censored median a < b, >= ceil(0.8 n) = 4/5 strict paired wins, exact one-sided Gehan / MWU) with
    p < 0.025;
  - AUC: median AUC(a) - median AUC(b) >= 0.05 (no claim from sub-chance shifts, e.g. a memoriser moving from 0.01
    to chance), AUC(a) > AUC(b) on >= 4/5 seeds, exact one-sided MWU p < 0.025.
  0.025 = Bonferroni over the two co-primaries (family alpha 0.05 per comparison). What still passes: 4/5 grok vs 0/5
  (p 0.0238), 5/5 vs 5/5 separated (0.0040) or with 1 of 25 pairs inverted (0.0079); what no longer passes at
  t_grok: 5/5 vs 5/5 with 4 of 25 pairs inverted (0.0476; U = 22 gives 0.111 and failed already).
  Power cost (log-normal grok times, audit model, 400 draws): t_grok test at a 3x speed-up 0.77 (log sd 0.5) / 0.41
  (0.75), vs 0.92 / 0.65 at p < 0.05 -- see "Open decisions".
- Verdict of a failed comparison: "floor: uninformative" if both arms are censored on >= 4/5 seeds AND both median AUCs
  <= floor reference + 0.05 (nothing generalises: says nothing about the prior); otherwise "not shown at n=5" (low
  power: never written as "no effect").
- (a) as v1 with the new T; additionally: both (a) comparisons "floor" = "(a) uninformative (floor)" at that fraction,
  not a negative. With `--always`, (a) is also read on the @always arms as a SECONDARY ("does a persistent prior help"),
  never replacing the windowed (a).
- (b) unchanged.
- (c) at the LOWEST fraction of the task's design (0.3 by default; 0.45 for S4 if the user picks `--s4fracs 0.45,0.6`):
  none fails to grok in >= 4/5 AND R / expo groks in >= 4/5 = RESCUE; none fails in >= 4/5 AND the AUC test
  T(R / expo > none) passes = PARTIAL RESCUE ("the prior raises held-out accuracy where none stays at the memoriser
  level", no grok claim); none fails AND T(R / expo, none) is "floor" = uninformative; specrand also rescuing (groks
  >= 4/5 or passes the AUC test vs none) narrows either to "any consistent prior of this rank".
- (d) narrowed (audit IMPORTANT 4): onehot's centred span CONTAINS expo (rank 6 vs 3), and F59 already shows rank alone
  speeds grokking, so T(onehot < expo) = "the rank-6 superset of expo is faster", NOT "additivity is not what helps";
  T(expo < onehot) = "the rank-3 additive code beats its rank-6 superset". Only with `--oh`: T(onehot < specrand_oh)
  separates onehot's structure from its rank. Read only if expo or onehot passes vs none or vs specrand.
- Everything else (qualifying fraction, budget rule, seeds, arms, window, lambda) is unchanged; implemented in
  toy_struct.py:compare / rule_a / rule_c / summarize, known answers in test_toy_struct.py:summary_known_answers.

## Cost
- Measured, this code, CPU 1 thread on the (loaded) session host, eval every 100: s4 0.3 7.8-9.5 ms/step, s4 0.6
  15-17, dim 0.3 10.8-13.5, dim 0.6 23-27 (smoke and pilot JSONs, ms_per_step). Revision 2: dim 0.3 class split
  14-20 ms/step (pilot, 2 processes on 2 cores); the r2 test log's 25 / 39 / 47 / 83 ms/step ran with 4 processes on
  2 cores and is not a rate estimate.
- GPU not measurable here (the session host has no GPU). Reference: toy-multifreq (toy_grok, same model, 5 seed
  processes sharing one A100/H100, eval every 100) = 573 s per 25k-step run = 22.9 ms/step per process; launch-bound,
  so roughly independent of the batch size at these sizes. Generator: MS_PER_STEP = 23.
- Per job (5 processes in parallel, 6 arms each, sequential): 0.3 = 6 x 50,000 x 23 ms = 1.92 h; 0.6 = 6 x 25,000 x
  23 ms = 0.96 h. --time = 2x rounded up: 04:00:00 (0.3), 02:00:00 (0.6).
- Total: 4 jobs (s4-f0.3, s4-f0.6, dim-f0.3, dim-f0.6), estimate 2 x 1.92 + 2 x 0.96 = 5.75 L40S-h, limit 12 h. Every
  run JSON records ms_per_step and device, so the first finished job recalibrates the estimate.
- CPU alternative (not generated, no CPU-partition convention in this project): with 5 cores per job, the measured
  1-thread rates give ~0.75 h (s4 0.3), ~0.7 h (s4 0.6), ~1.1 h (dim 0.3 and 0.6) per job, i.e. not slower than the
  5-way-shared GPU estimate. The user's call.
- Caveat (audit): the 23 ms/step reference ran on gh015 (A100/H100, not L40S) and without --threads 1. Its sacct
  gpuutil (103 over 57 min) suggests high utilisation, so the 40 % preemption rule is unlikely to bite; the 2x --time
  margin covers an L40S being up to 2x slower. A TIMEOUT is not requeued but loses only the arm in progress (finished
  arms are on disk; resubmitting the same job resumes).
- Options: `--always` makes S4 cells 9 arms (s4 0.3 2.88 h, --time 6 h; s4 0.6 1.44 h, 3 h: +2.2 L40S-h);
  `--oh` makes dim cells 7 arms (+0.5 L40S-h); dropping s4-f0.3 saves 1.92 h (2.88 h with `--always`).

## Feasibility (NOT pre-registered, not a result; preflight/feasibility_s4_f0.9)
- s4 none at train fraction 0.9 (518 train / 58 held out), seed 100, CPU: t_mem 200, t_grok 9900, test 0.966 at the
  stop. So this model / optimiser CAN learn S4 from enough of the table; its failure at 0.3 / 0.6 is a data-size
  effect, not a broken pipeline. In the same run none's coset_share over steps 1000..2000 was 0.40 (chance 0.13): the
  unaided network makes coset-structured errors on its own while learning, so rule (b)'s none control is needed.
- A 5-seed version of this cell (none, 0.9, 25k, ~0.2 L40S-h) is in the generator behind --posctl, OFF by default
  (descriptive only; the user's call).
- Rescue capacity (audit seed 100 + revision-2 seeds 101-102; CPU 1 thread, --stop_at_grok, 8k steps unless noted;
  preflight/feas2, audit copy in the session scratch toy_struct/audit/feas). Held-out accuracy at 8k (AUC over 8k):
  | S4 cell | none | R, window [0, 2000) | R@always | specrand, window | specrand@always |
  |---|---|---|---|---|---|
  | frac 0.6, seed 100 | 0.02 (pilot) | 0.72, plateau 0.72-0.77 from 3k (0.63) | groks at 4700 (0.76) | 0.01 (0.007) | 0.00 (0.003) |
  | frac 0.6, seed 101 | 0.04 (pilot) | 0.32, slow rise (0.24) | 0.81, rising (0.42) | 0.03 (0.025) | – |
  | frac 0.6, seed 102 | 0.04 (pilot) | 0.71, rising (0.48) | 0.91, rising (0.59) | 0.03 (0.025) | – |
  | frac 0.45, seed 100 | 0.01 (0.015) | 0.09, flat 0.07-0.12 (0.08) | 0.26, rising (0.17) | – | – |
  | frac 0.45, seed 101 | 0.02 (0.019) | 0.10, flat (0.09) | 0.32, rising (0.17) | – | – |
  | frac 0.3, seeds 100 / 101 / 102 | 0.01-0.03 (pilot) | – | 0.03 (25k: 0.06) / 0.03 / 0.06 | – | – |
  Reading (feasibility, not a result): at 0.6 the R prior lifts held-out accuracy from the memoriser floor in 3/3 seeds
  and specrand does not, but the windowed R reaches 0.95 by 8k in 0/3 seeds (always-on: 1/3); the v1 t_grok rule
  would have scored this as a censored tie, which is what the AUC co-primary fixes. At 0.3 even an always-on R prior
  stays at chance (3/3 seeds, one to 25k): the s4-f0.3 cell is very likely a floor in every arm, so its (a) and (c)
  will most likely read "uninformative", not "no effect". 0.45 sits in between (window ~0.1, always-on ~0.3 at 8k).
- dim on the class split (seed 100, window arms, 8k): frac 0.3 expo 0.73 and rising (AUC 0.34) vs specrand 0.04
  (0.03) vs none 0.02-0.05 (pilot); frac 0.6 expo groks at 400 (AUC 0.97), specrand at 5200 (0.69), none at 2900.
- The s4 0.45 none pilot (group F of preflight/pilot2.py) was deliberately skipped (killed at start, no data): it is
  only needed if the user picks 0.45.

## Tests (tools/test_toy_struct.py; revision-1 run: preflight/test_toy_struct_final.log; revision-2 run on the files of the current snapshot, a099a1edaece / 1533ad98f6c0: preflight/test_toy_struct_r2.log, ALL TESTS PASSED, known answer 4800 / 4800 bitwise)
- Known answer: --task mod vs toy_grok.py, 200 steps, eval every step, arms none / w0 / w100 / always_shuf, 1 CPU
  thread: 4800 / 4800 logged values (ce, train, test, cka_circle, fourier_top5) bitwise equal, max |diff| 0; t_mem,
  t_grok, final_test, seed, frac, lam, width, p, freqs equal; the four arms differ from each other at step 150; w100
  equals none for steps 0-99 and differs from step 101. Found on the way: with 2 CPU threads toy_grok.py itself is
  NOT bitwise reproducible run to run (two identical runs differ from step 1, max |diff| 7.8e-4 by step 60); with 1
  thread both scripts are deterministic (two full smoke runs gave identical numbers). GPU runs are likewise not
  expected to be bitwise reproducible; arms stay paired by seed (same init, same split).
- Structure: convention check (a = 90 deg about z, b = 90 deg about x: "a then b" = R_b R_a, x -> y -> z); 456 / 576
  non-commuting cells; 12 odd elements; V4 cosets 6 x 4; targets / ranks / CKA matrix / spectra / shuf-is-a-permutation
  / splits / closure 1.00 for seeds 0-4 x fractions 0.3, 0.6; closure known answers (one missing cell derived; the
  identity row alone derives nothing; one dim relation derives exactly its triangle); asserts that must fire do
  (duplicate question; a corrupted, non-associative table); exact MWU / Gehan p: separated 5 vs 5 = 1/252 = 0.0040,
  4/5 grok vs 0/5 = 0.0238, equal to scipy's exact MWU on 20 random untied cases.
- Smoke, 300 steps, every arm, both tasks, both fractions, CPU: CKA to the own target 0.28-0.57 at init -> 0.99-1.00
  at step 299, above none's CKA to the same target (0.22-0.83); prior_on True throughout, False for none (geo None);
  a --win 0,150 run has prior_on [True, True, False, False] at evals 0/100/200/299; all arms of a seed share the init;
  absR coset_share at 299 = 0.50 (0.3) / 0.79 (0.6) vs 0.06-0.32 for the other arms (chance 0.13).
- Timing: see Cost. The job command line was also run from the snapshot directory (dim 0.6, 200 steps, 6 arms).
- Revision 2 additions (final run: preflight/test_toy_struct_r2.log):
  - splits: s4 question split = toy_grok's draw (perm equal to default_rng(seed).permutation(576)); dim class split
    has no class across train / held-out; the relation classes equal an independent enumeration of each relation's
    phrasings (s4 528 classes = 480 singletons + 48 commuting pairs; dim 172 = 171 x 4 + 1 x 2); the fresh mask
    equals a second-path computation for seeds 0-4 x 0.3 / 0.6 (s4 fresh 379-391 / 205-210 = the audit's twin counts);
    dim --split question still reproduces the old draw; closure 1.00 for main seeds on the class split.
  - decision rules on synthetic 5-seed cells: partial rescue (R 0.70 vs specrand 0.01, all censored) -> PASS (AUC),
    t_grok p 1.000; rescue (4/5 grok) -> PASS (t_grok + AUC), p 0.024; floor (R 0.04 vs 0.01 on every seed, p 0.004
    but < AUC_MIN) -> "floor: uninformative", (a) uninformative; overlap -> "not shown at n=5"; rule (c) RESCUE /
    PARTIAL RESCUE / floor as expected; budget < 4000 -> (a) "untestable"; a stopped pilot run's AUC carries forward,
    an incomplete main-run curve raises.
  - smoke: n_train / n_fresh / split recorded per run and equal to the split; dim test_fresh == test at every eval;
    no .tmp files left; --summarize runs on every smoke cell; per-arm windows: R@always prior_on at all evals,
    R@200-250 only at step 200, target R.

## Seven checks (experiment-audit)
| check | status | evidence |
|---|---|---|
| i double path | pass | closure by matrix rank and by least-squares residual (asserted equal); MWU vs scipy exact; CKA values vs analytic (2/sqrt 6, Schur zeros, sqrt 8 spectrum); toy_struct mod vs toy_grok bitwise |
| ii intent asserts | pass | group axioms + S4 isomorphism + V4 quotient; split disjoint by content and (dim) by relation class, every item an input and an answer; expo inside onehot's span; specrand_oh spectrum; ranks, spectra, orthogonality, shuf is a permutation; derived closure cells equal the table; arm names valid per task |
| iii known answer | pass | mod == toy_grok (4800/4800); decision rules on synthetic cells with known verdicts (revision 2); negative asserts fire; closure known answers; the prior is read (smoke); upper check: none groks S4 at 0.9 (9900) and dim at 0.6 (pilot 4/5) |
| iv anomaly repro | n/a | no results yet |
| v fresh adversarial reviewer | DONE once (revision 1) -> revision 2 | independent audit 2026-10-16 (verdict: hold; 1 blocker, 4 important, 4 minor; response below). Revision 2 itself (new split, co-primary, rules, generator options) has NOT had a fresh review: launch one on the diff (session scratch toy_struct/fix/*_pre_fix.* vs the current files) right after submission, or before it if the user wants |
| vi trap list | pass | 0 (revision 2) symmetry-clean dim split (project rule), atomic writes, one summary per cell after all PIDs; 1 same template / position for train and held out; 2 held-out disjoint, closure 1.00, below-chance memoriser signature documented; 3 prior gradient live (CKA -> 1.00, window switch-off tested); 4 5 seeds, paired, exact test; 5 readout = last token, printed; 6 one CKA function for loss and metric, t_grok defined once; 7 every held-out question scored, n_wrong next to every share; 8 toy_grok.py untouched (last change dabaad0), job script waits on each PID and appends logs across requeues |
| vii decoded inputs side by side | pass | every seed's log prints 3 training and 3 held-out questions as ids + names with the scored position marked, e.g. `ids [5, 22, 24] len 3 \| g05 g22 ⟦=⟧ (pos 2) -> g15` |

## Audit response (revision 2, 2026-10-16)
| audit finding | real? | fix |
|---|---|---|
| BLOCKER: t_grok@0.95 cannot register a partial rescue | yes: replicated on pilot seeds 100-102 at S4 0.6 (window R 0.32-0.72 at 8k vs specrand 0.01-0.03, all censored) | AUC co-primary + "floor" / "not shown" verdicts (Amendment); no training-code change; known-answer cells in the tests |
| IMP 1: window short for S4, 0.3 a floor | yes: always-on R at 0.3 stays at 0.03-0.06 (3 seeds, one to 25k); at 0.6 window R reaches 0.95 by 8k in 0/3, always-on in 1/3 | not changed by default (design = "window as F45 / F59", fractions 0.3 / 0.6 approved): `--always` (positive control arms) and `--s4fracs` ready; see Open decisions |
| IMP 2: power at n = 5 | yes (and the Bonferroni split lowers t_grok power further: 3x speed-up 0.41-0.77) | every failed test reads "not shown at n=5" or "floor: uninformative", never "no effect"; alpha choice and n = 10 are Open decisions |
| IMP 3: dim split not symmetry-clean | yes, worse than reported: 87-99 % of held-out at 0.6 had a class-mate in training and the pilot's none "grok" sat at that ceiling, so it hit t_grok too | dim split by relation class (default, `--split class`); test_fresh logged for both tasks; dim none pilot rerun on the class split (Budget); s4 keeps the question split (reason in Split) |
| IMP 4: rule (d) misread | yes: onehot's span contains expo (now asserted), rank 6 vs 3 | (d) reading narrowed (Amendment); rank-6 control `specrand_oh` behind `--oh` |
| MINOR dim chance not 1/27 | yes | chance_majority logged per run; floor reference = max(1/n_out, median chance_majority); drop_share chance marked approximate |
| MINOR non-atomic JSON + per-process summary | yes | tmp file + os.replace; training processes no longer summarise; the job summarises once after all PIDs (tested: no tmp files left) |
| MINOR GPU estimate from A100/H100 | yes | caveat in Cost; 2x margin kept |
| MINOR shuf leans to the quotient | yes | wording "partly quotient-like" (Arms notes) |

## Open decisions (the user's; defaults = the approved design + the fixes above)
1. S4 0.3 is very likely a floor in every arm (always-on R at chance in 3/3 seeds, one to 25k). Options: keep it as
   approved (~1.9 L40S-h, expected verdicts "uninformative"); drop it (`--s4fracs 0.6`); or replace it with 0.45
   (`--s4fracs 0.45,0.6`; window R ~0.1, always-on ~0.3 at 8k vs none 0.02; its none pilot at the pre-set 50k budget
   has NOT been run, the budget rule needs it first, ~45 CPU-min on 2 cores).
2. `--always`: add R@always / R_sign@always / specrand@always to the S4 cells (+2.2 L40S-h). At 0.6 always-on R
   reaches 0.81-0.95 by 8k vs window R 0.32-0.72, so it is the likelier place for t_grok rescues, and it tells "the
   window is too short" apart from "the prior does not help". Read as secondary (a) and (c), never replacing them.
3. alpha per co-primary: 0.025 (Bonferroni, default) or 0.05 (family error <= 0.10 by the bound, raises the t_grok
   test's power at a 3x speed-up from 0.41-0.77 to 0.65-0.92). Must be fixed before submission.
4. n = 5 seeds (default) or 10 (twice the GPU-h; a 3/5-vs-0/5 pattern, p 0.083, is the typical near miss at n = 5).
5. `--oh` (specrand_oh arm, +0.5 L40S-h) to make rule (d) interpretable beyond "rank-6 superset vs rank-3 additive".
6. `--posctl` (S4 none at 0.9, ~0.2 L40S-h), GPU vs CPU, and a fresh reviewer on the revision-2 diff (before or
   right after submission).
7. Expectation, no action needed: on the clean split dim 0.6 none groks early (pilot median 2900 < 4000), so dim (a)
   most likely rests on dim 0.3 (none flat at chance to 50k in 5/5 pilot seeds).

## 2026-10-17 submission choices (before any result; Claude, under the user's go-ahead for both toy experiments)
- S4 fractions 0.45 and 0.6 (--s4fracs 0.45,0.6). The audit's feasibility runs put every S4 arm at the floor at 0.3.
  0.45 uses its pre-registered 50k budget; its none pilot was not run, so the doubling rule is checked on the main runs'
  none arm and reported.
- --always: R / R_sign / specrand also always on, which separates "the 2000-step window is too short" from "the prior
  does not help".
- --oh: specrand_oh, the rank-6 control for onehot, rule (d).
- --posctl: S4 none at 0.9, descriptive.
- alpha 0.025 per co-primary (as pre-registered); 5 seeds.
- Fresh review of revision 2 runs in the background after submission (experiment-audit gate).

## 2026-10-17 amendment 2 (fresh re-audit of revision 2; before any job ran; HOLD set while fixing)
Evaluation and read-out only. The training path is unchanged, and so are the losses, splits, seeds and arms.
1. **AUC floor line.** An AUC PASS now also needs the faster arm's median AUC to be above chance + AUC_MIN (the cell's
   floor line). Before this, a lift from 0.013 to 0.069 at S4 (floor 0.092) read "PASS (AUC)" / "PARTIAL RESCUE".
   Checked on the auditor's synthetic case: 0.069 vs 0.013 now reads "floor: uninformative"; 0.20 vs 0.013 still passes.
2. **dim negation twins.** x / y and x * (-y) state the same answer through different relation classes. Under the class
   split, 27-32% (0.3) and 55-64% (0.6) of held-out questions have that twin in training (seeds 0-4; recomputed, equal to
   the audit). Every eval now logs test_negtwin / test_nonegtwin / n_negtwin.
   - Rule added (pre-registered now): a "learned the lattice" reading of any dim pass also needs the AUC test on the
     no-twin subset (same thresholds, floor line included).
   - It does not bias expo vs specrand: every arm sees the same held-out set.
   - test_fresh is identical to test for dim; it is kept but is not a freshness measure.
3. **Summary text.** The header prints the windows of every run, and rule texts use the actual n.
- S4 inverse twins ((b^-1, a^-1) -> c^-1 in training for 40-59%) have a different answer, so they are not a lookup; noted only.
- Not changed: the time limits. The auditor's check is the first finished JSON's ms/step x 1.2 x steps x arms < limit,
  read when the first run ends.
