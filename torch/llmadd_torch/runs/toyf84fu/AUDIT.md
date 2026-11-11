# toyf84fu: follow-ups that make the toy conclusions of F84 补充（定稿） stronger (2026-10-19, BUILT, NOT SUBMITTED)

Object: FINDINGS "F84 补充（定稿）" (LLM_addition_review/FINDINGS.md:3429) and its results audit
notes/batch1_build/audit_toy_optimiser_sweeps.md, section 6 "cheapest next steps", plus the coordinator's Latin
addition (a)-(c) from the writing session. Everything below was written before any run of this batch.

## 0. Headline of the build (read first)
- **The Latin premise is not right as stated (needs the user's / writing session's attention).** The toy's
  `closure_latin = 1.0` is associativity (the group law) plus a one-missing-cell Latin rule. Associativity alone
  (`closure`) is already 1.000 on every seed at every fraction 0.2 / 0.3 / 0.45 / 0.6, in the toy and in the rainbow.
  Latin-square propagation alone (naked + hidden singles, no group law, A7 / A8's definition) derives **0 held-out cells
  at 0.2 / 0.3 / 0.45** and **2.6-10.8 % at 0.6** (rainbow 0.6: 3.5-11.7 %). See section 6. So "at 60 % the table is solvable
  by Latin table completion without group structure" does not hold, and it does not explain why the tuned none learns
  S4 0.6: none reaches 1.000 on the non-implied cells (5/5 seeds, lr 1e-2 wd 1).
- 17 jobs in 5 groups, 5 seeds each, 5.4 L40S-h estimated (g1-g4: 1.6 h; g5: 3.8 h); every job asks for 2 h.

## 1. Code and provenance
- Trainer: `tools/toy_struct.py` at the pin. It comes from `git archive` of llmadd-burst 398935a. Its sha256[:12] is
  b74a78cb5472, the same file as toynone2 / toynone3 / toyprior_lr / Burst 480-482. It is unmodified and lives in
  `code/tools/` (`code/COMMIT`). tools/ in the repo is untouched.
- Runner: `code/hits_wrap.py` (sha256[:12] **c2384fbe7fb4**). It runs toy_struct's own `main()`. It only observes:
  - OneLayer is subclassed with the inherited `__init__`, so the init is the same. `forward()` returns the original
    logits. Under no_grad, on exactly the held-out batch, it records the argmax.
  - run_arm is wrapped. Each eval gets `hits_te` (hex of packbits(pred == y), in toy_struct's te order) and
    `hits_pass_mismatch`. Each run gets `hits_wrap` = {sha, n_te, encoding}.
  - In-run asserts: two held-out passes per eval; `hits.sum() == n_test - n_wrong`; `|hits.mean() - test| <= 1e-6`.
  - code_sha in every JSON stays b74a78cb5472.
- Generator: `/scratch/yh6210/llmadd_torch/make_jobs_toyf84fu.py`. It follows the conventions of make_jobs_toynone3 /
  make_jobs_toyprior_lr: account torch_pr_625_general, partition l40s_public, qos gpu48, 1 GPU, 8 CPU, 32G, --requeue,
  5 seed processes per job, `--threads 1 --eval_every 100`, `toy_struct.py --summarize <out>` at the end, the HOLD
  switch, and SUBMITTED_<name> flags.
  - The generator asserts both shas.
  - Every job re-checks both shas before any GPU work (exit 3). It refuses a non-L40S GPU (exit 4). It appends
    date / job / restart / host / GPU name + uuid to `out/<cell>/gpu.txt`.
  - Options: `--print [--groups ...]` lists what would be submitted; `--submit [--groups g1,...]`. Any other option, and
    `--groups` without `--submit` / `--print`, is refused. preflight/generator_options.log shows this, and that the
    option patch left the 17 rendered files byte-identical.
- Read-out: `readout.py` (uses the pinned toy_struct for mwu_p / auc / compare conventions) and `latin.py`.
- Nothing submitted. No HOLD file. 0 SUBMITTED_ flags.

## 2. Cells (budgets / splits = the existing cells)
| group | job (cell dir under out/) | task, frac, split, steps | lr | wd | arms | seeds |
|---|---|---|---|---|---|---|
| g1 same device | g1-s4-f0.6-lr1e-2-wd1 | s4 0.6 question, 100k | 1e-2 | 1 | none, R@always, R_sign@always (one job = one L40S) | 0-4 |
| g2 upper lr edge | g2-s4-f0.6-lr3e-2-wd1 | s4 0.6, 100k | 3e-2 | 1 | none, R@always | 0-4 |
| | g2-s4-f0.6-lr3e-2-wd3 | s4 0.6, 100k | 3e-2 | 3 | none | 0-4 |
| g3 dim none | g3-dim-f0.3-lr1e-2-wd0.03 | dim 0.3 class, 200k | 1e-2 | 0.03 | none | 0-4 |
| | g3-dim-f0.3-lr2e-2-wd0.1 | dim 0.3, 200k | 2e-2 | 0.1 | none | 0-4 |
| | g3-dim-f0.3-lr2e-2-wd1 | dim 0.3, 200k | 2e-2 | 1 | none | 0-4 |
| g4 dim prior lr | g4-dim-f0.3-lr3e-3-wd1 / -lr1e-2-wd1 | dim 0.3, 200k | 3e-3 / 1e-2 | 1 | expo, specrand | 0-4 |
| g5 S4 prior lr (coordinator (c)) | g5-s4-f{0.2,0.3,0.45}-lr{3e-3,1e-2,3e-2}-wd1 (9 jobs) | s4 question, 200k (= toyshape none budget) | 3e-3 / 1e-2 / 3e-2 | 1 | R@always, specrand@always | 0.2: 0,1,2,4,5; else 0-4 |

Notes on the cells:
- Every arm passes `--win 0,2000 --lam 1` (F84's schedule; the defaults).
- dim prior arms: `expo` and `specrand` take that [0, 2000) window, the exact F84 arm strings (runs/toystruct
  job_dim-f0.3-oh). F84's default wd is 1 (the argparse default; toystruct passed no --wd), so "wd 1" is the only wd.
- g5: toyshape (runs/toyshape, d182ce548bcc; its default path is bitwise b74a78cb5472) runs R@always at lr 1e-3 / 1e-2
  only, at B = 50k, and has no specrand.
  - g5 adds 3e-3 / 3e-2 and specrand.
  - R@always@1e-2 is kept at 200k because the coordinator listed 1e-2. It overlaps toyshape's 50k run; its first 50k
    double as a cross-batch determinism check. Drop it if a strict no-duplicate rule is wanted (open decision 2).
- S4 0.2 seeds 0, 1, 2, 4, 5: toy_struct.split refuses seed 3 (an item is never a training answer). This is
  toyshape's rule, checked by latin.toy_seeds.
- Cost: measured L40S rates (toynone2 / toyprior_lr / toystruct JSONs): s4 none 3.05 ms/step/process, s4 prior 3.7,
  dim none 2.61, dim prior 2.69. Planning uses 3.1 / 3.8 / 2.7 / 2.75. Jobs take 0.09-0.42 h each, 5.4 L40S-h in total.
  The previous comparable jobs ran 6-20 min.

## 3. Metrics (one definition each; readout.py docstring)
- t_grok, AUC, final: as toy_struct / F84.
  - t_grok = the first eval with held-out >= 0.95 (censored = inf).
  - AUC = mean held-out accuracy over every eval of the budget.
  - final = the last eval.
- last50 = mean held-out over the last 50 evals (audit §6). It is printed beside `final` for every cell, dim included.
- @cut = only the evals at steps 0, 100, ..., cut - 100. That is the same eval set for a cut-step run and a longer run;
  the cut-step run's own eval at cut - 1 is dropped. This is toyshape's `upto()` convention.
- Unstable run: from step 1000, train < 0.9 on > 25 % of evals, or final train < 0.9. OPTIMISER FAILURE = >= 3/5
  unstable runs.
  - Checked on the known cells: dim lr 3e-2 wd 1 and wd 3 have 5/5 unstable runs (the audit's collapse), and dim lr
    1e-2 wd 1 has 0/5.
  - A failure config is never "best". It closes a bracket only "by instability".
- Late gain = mean of the last 5 evals minus the mean of the first 5 evals at or after 0.75 x budget (toyshape rule 1c).
- rule-2 best = most groks, then the highest median AUC.
- S4 0.6 speed rule = among the configs with the most groks, the lowest censored-median t_grok; ties go to the higher
  AUC, then the lower lr.
  - Why it is needed: every S4 0.6 candidate groks 5/5, and their 100k AUCs differ by <= 0.005 (post-grok wobble), so
    rule 2 would select on noise.
  - Existing data: rule 2 picks R@always lr 3e-3 (AUC .984 vs .981 at 1e-2), while the speed rule picks 1e-2.
  - Both picks are printed.
- Latin split (S4):
  - implied = held-out cells derived by naked + hidden singles from the seed's own training cells (latin.py, the
    algorithm of rainbow6_summary.s4_latin_level / A8 s4_hard); non-implied = the rest.
  - With hits_te (this batch), the per-eval accuracy on each part is exact.
  - Without hits, bounds come from n_wrong: with k correct, non-implied lies in [max(0, k - n_imp), min(k, n_non)] / n_non.
  - The masks are asserted to come from the run's own split (split_sha).
- Train vs eval: unchanged toy_struct.
  - Template (a, b, =) / (a, op, b, =), scored at '='.
  - Full batch, fp32, no dropout or norm. Eval on all held-out every 100 steps.
  - Decoded inputs side by side in every seed log (preflight/smoke_jobs.log).

## 4. PRE-REGISTERED reading (written before any run)
**g1 (same device).** Primary: g1 none vs g1 R@always (one L40S job, lr 1e-2, wd 1, 100k). Report:
- the ratio of censored-median t_grok, the per-seed ratios (a censored none gives the lower bound 100k / t_R), the
  median per-seed ratio, the minimum, the number of seeds where R is earlier, and the exact one-sided Gehan p;
- paired T(R@always, none) on the F84 co-primaries; AUC over the first 5k.

Wording:
- If R is earlier on 5/5 seeds with p < .025, write "R@always vs none on one L40S: about a-b x", with a and b = the two
  estimators (ratio of medians, median per-seed ratio). ">4x" only if both are >= 4.
- Otherwise write "speed-up not shown on one device at n = 5".
- The mixed-device 4.6x of F84 is replaced, not averaged. R_sign@always is the same, as a secondary.

Side readings:
- (i) g1 none vs toynone2 none, same sha / seed / split / init / GPU type.
  - Bitwise-equal curves (hits keys ignored) -> "an L40S re-run reproduces".
  - Otherwise -> "run-to-run nondeterminism on one GPU type". The per-seed ratio is then noisy at that level, and only
    g1's none is paired with g1's R. Never pooled.
- (ii) g1 R (L40S) vs Burst 482 R (A100): a median difference > 1.5x is reported as a device effect on R.
- (iii) gpu.txt: more than one start means requeued, so "same GPU type, maybe not the same card".

**g2 (upper lr edge).**
- Grids (all L40S, 100k):
  - none = toynone lr 1e-3 + toynone2 + g1 at (1e-2, 1) (it replaces toynone2's copy there) + g2 at (3e-2, 1 / 3);
  - R@always = toyprior_lr 3e-3 + g1 1e-2 + g2 3e-2. F84's lr 1e-3 cell is shown at 25k (median 10.2k) and is used
    only as R's lower bracket neighbour.
- Selection: the speed rule.
- "An arm's optimum moves past 1e-2" iff its speed-rule best is the lr 3e-2 config. A failure config cannot be best.
- If none@3e-2 is an OPTIMISER FAILURE, write "above 1e-2 the optimiser fails (bracket closed by instability)", never
  "a bracketed optimum".
- If an arm's best is at 3e-2, its bracket is OPEN at the top. Report "best tested". No further lr without the user.
- Best-vs-best (L40S only) is reported with g1's statistics and wording rule. Rule 2's pick is printed beside it.

**g3 (dim none).** The grid = toynone lr 1e-3 + toynone2 + toynone3 / Burst 480 / 481 (lr 3e-2) + g3.
- (a) toynone criterion: if any config has >= 2/5 groks, "at most 1/5 in every config" FAILS. That config is then a
  "none learns" config: the dim reading moves to best-vs-best like S4, and needs g4.
- (b) toyshape rule 1(b) with cap 0.20: on the median final AND on the median last-50 mean. A config above either is
  listed. dim 0.3 is already "not a wall cell" (lr 1e-2 wd 1: 0.214 / 0.238). Report the new maximum.
- (c) Rule 1(c): median late gain >= 0.02 means still rising.
- (d) Bracket of the rule-2 best (now lr 1e-2 wd 0.1):
  - wd 0.03 tests its bottom wd edge; lr 2e-2 tests the gap below the 3e-2 failure.
  - Best moves to wd 0.03 -> OPEN at the bottom again; reported, no further extension without the user.
  - 2e-2 is a failure -> "closed by instability".
- (e) Every grokked none run: t_grok, post-grok mean (sustained iff >= 0.8), share >= 0.9, final and last-50.

**g4 (dim prior lr).**
- Common budget 50k, because F84's expo / specrand@1e-3 cell ran 50k.
- expo's rule-2 best lr is chosen at @50k over {1e-3, 3e-3, 1e-2}. Its lr bracket is closed iff 3e-3 wins; wd is not
  swept for expo (1, as F84).
- none's rule-2 best is chosen at @50k over every none config, truncated.
- T(best expo, best none) @50k on the t_grok + AUC co-primaries. The no-twin AUC (F84 amendment) is secondary.
- At 200k: expo at each new lr vs the rule-2 best none at 200k.
- Specificity at the tuned lr: T(expo, specrand) at each lr (200k and @50k). "Holds at the tuned lr" iff it passes at
  expo's best lr.
- "specrand blocks learning that none achieves" iff all hold:
  - at the same (lr, wd 1), specrand's AUC is below none's on >= 4/5 seeds;
  - the median difference is <= -0.05;
  - none's AUC is above chance + 0.05.
- Wording:
  - A dim best-vs-best ratio is written only if expo's lr bracket and none's lr and wd brackets are all closed.
    Otherwise write "best tested (bracket open on ...)".
  - A censored none median gives "> X x".
  - expo seed 0 (failed at 1e-3) is reported per lr.

**Latin split and rescue (coordinator (b), the writing session's rule, operationalised here).**
- Every S4 arm in every cell read (toystruct, toynone / toynone2, toyprior_lr, Burst 482, g1, g2, g5, toyshape) is
  reported on implied and NON-implied cells separately: n implied per seed, final / last-50 / grokked / t_grok / AUC.
- At a fraction f in {0.2, 0.3, 0.45}, **RESCUE on non-implied cells** iff all of (i)-(iii) hold:
  - (i) every none config of toyshape's sweep (extensions included; 4B = 200k) fails on non-implied cells:
    - <= 1/5 seeds reach 0.95;
    - median final and median last-50 are <= 0.20;
    - median late gain is < 0.02.
    The upper bounds are used if only bounds exist.
  - (ii) R@always reaches >= 0.95 on non-implied cells within B = 50k on >= 4/5 seeds (lower bound). Its lr is the one
    with the highest median AUC_B on non-implied cells, from toyshape's 1e-3 / 1e-2 at 50k and g5's 3e-3 / 1e-2 / 3e-2
    truncated to 50k.
  - (iii) its paired AUC_B vs the none config with the highest median AUC_B passes compare(), on the non-implied cells
    AND on test_fresh (toyshape rule 4).
- The verdict carries "SPECIFICITY NOT SHOWN vs specrand" unless R@always also beats specrand@always at the same lr
  (paired AUC_B, non-implied).
- Otherwise the verdict is "not a rescue (reason)".
- S4 0.6 is never a rescue fraction: tuned none is 1.000 on the non-implied cells.
- Because the implied share is 0 at 0.2-0.45, there the reading equals toyshape's own rule-1 + rule-4 reading of the
  assign arm, at a tuned prior lr. toyshape uses assign (R@always) only as its upper reference.

## 5. What the toy JSONs log (coordinator (b)) and the cheapest way to the split
- Logged per eval (b74a78cb5472):
  - step, ce, train, test, test_ce, lr_now, prior_on, cka_<target>, geo, n_wrong;
  - test_fresh;
  - S4: coset_share, test_noncomm, test_comm;
  - dim: drop_share, test_mul / test_div, test_negtwin / test_nonegtwin, n_negtwin.
- Not logged: per-question hits and model weights (toy_struct has no torch.save). So an eval-only pass on saved
  models is impossible for toystruct / toynone2 / toyprior_lr / toyshape.
- The cheapest way:
  - (1) At 0.2 / 0.3 / 0.45, nothing is needed: 0 implied cells on every seed, so non-implied = all held-out.
  - (2) At 0.6, n_wrong bounds the non-implied accuracy to a band of width n_imp / n_non <= 0.12, and the band is exact
    at 0 and 1. Every existing 0.6 cell is read this way, e.g. toynone2 none lr 1e-2 wd 1: final non-implied 1.000
    exact, t_grok non-implied in [4600, 4700].
  - (3) Exact values need hits_wrap (this batch: g1, g2, g5). For an old cell they need a re-run through the wrapper
    (5 seeds x ~6-20 min per cell). Re-runs are comparable only if an L40S re-run is deterministic, which g1's none
    tells.
  - toyshape is already queued without hits. Its S4 cells are at 0.2-0.45, where (1) makes hits unnecessary.
- A/B for the wrapper (preflight/ab_hits.py, CPU, 1 thread; 8 run pairs: s4 0.6 none / R / R_sign at lr 1e-2, s4 0.2
  seed 5 R / specrand at 3e-2, dim none / expo / specrand at 3e-3):
  - 1684 values are bitwise equal on every shared key. The wrapper adds only hits_*.
  - The hits reproduce test, n_wrong and test_fresh, plus s4 test_comm / test_noncomm and dim test_mul / test_div /
    test_negtwin / test_nonegtwin, from masks rebuilt with toy_struct's own split code, at every eval.
  - The hit vectors change over training. PASSED.

## 6. Latin table (coordinator (a); latin.py -> latin_table.md, per seed there)
| split | frac | seeds | assoc (toy `closure`) | assoc+latin (toy `closure_latin`) | Latin-only share | Latin level (A7) | Latin-only share on rainbow read-out | level on read-out |
|---|---|---|---|---|---|---|---|---|
| toy | 0.2 | 0,1,2,4,5 | 1.000 | 1.000 | 0.000 | 0.065 | 0.000 | 0.064-0.066 |
| toy | 0.3 | 0-4 | 1.000 | 1.000 | 0.000 | 0.083-0.085 | 0.000 | 0.084-0.086 |
| toy | 0.45 | 0-4 | 1.000 | 1.000 | 0.000 | 0.133-0.137 | 0.000 | 0.134-0.138 |
| toy | 0.6 | 0-4 | 1.000 | 1.000 | 0.026-0.108 | 0.285-0.328 | 0.025-0.111 | 0.285-0.335 |
| rainbow S4 | 0.6 | 0-7 | 1.000 | 1.000 | 0.035-0.117 | 0.269-0.353 | 0.034-0.118 | **0.268-0.352 (= A7's 26.8-35.2)** |

Known answers, asserted:
- The rainbow levels on the read-out equal A7's printed 35.24 34.78 29.60 30.44 | 32.24 30.18 26.82 33.06.
- Every rb6 run's recorded closure / closure_latin / n_readout is reproduced.
- The rainbow's held set = toy_struct.split(0.6, split_seed).

Why the toy and the rainbow look different (100 % vs 27-35 %):
- It is not the split. Both use toy_struct.split at 0.6 (the rainbow with its own split_seed). The Latin-only
  statistics of the two are the same range.
- It is two different deduction systems and two metrics.
  - The toy's closure_latin uses associativity, i.e. the group law. At every fraction >= 0.2, associativity already
    derives the whole held-out table from the training cells.
  - A7's level uses only the Latin-square property, and gives partial credit 1/|candidates| to the cells it cannot
    derive.
- A7's "26.8-35.2 %" is that partial-credit level on the read-out (no identity input, no trained commuting twin). Only
  3-12 % of the read-out is actually derived.
- The A7 audit (M4): full constraint propagation (GAC) is 0-2.2 points above singles. Not computed here.
- A random 60 % partial S4 table has unfilled intercalates, so it does not have a unique Latin completion.

Consequences:
- "solvable by table completion without group structure" is false at 0.6 (<= 11 % of held-out cells).
- closure (with the group law) is 1.0 at 0.2 too, where none presumably fails (toyshape will tell). So it does not
  separate the fractions.
- The tuned none's 5/5 at 0.6 is learning of non-implied cells: final 1.000 there, exact.

## 7. Tests (all CPU, <= 8 GB; logs in preflight/)
| test | file | result |
|---|---|---|
| wrapper A/B + hits second path | ab_hits.py / ab_hits.log | PASSED (8 pairs, 1684 values) |
| rendered-job smoke, one job per group (g1, g2 wd 1, g3 wd 0.03, g4 3e-3, g5 0.2 3e-2): the job's own command line, first seed, 300 steps, --device cpu, through optprobe.py | smoke_jobs.py / smoke_jobs.log | PASSED |
| generator options | generator_options.log | PASSED (refusals; rendered files unchanged) |
| read-out known answers on existing data | readout.py --selftest / selftest.log, readout_selftest.md | PASSED |
| read-out end-to-end on fake roots with designed outcomes | fake_e2e.py / fake_e2e.log, fake_e2e_readout.md | PASSED (17 lines, 3 rescue verdicts) |
| Latin levels vs A7 / rb6 records | latin.py (asserts) | PASSED |

What the smoke test checked, per run:
- code_sha b74a78cb5472; hits_wrap c2384fbe7fb4;
- task / frac / split / steps / lr / wd / lam as rendered; every arm present; windows as intended;
- `lr_now` == lr at every eval from step 10;
- the optimiser's own param-group lr == lr x min(1, (s+1)/10) and weight_decay == wd at every one of the 300 steps,
  for every optimiser (optprobe). These are the consumed values, not the CLI args;
- hits at every eval; the decoded inputs in the log; `--summarize` runs.

Paired wd A/B: dim lr 1e-2 seed 0, wd 0.03 vs 0.1, same split and init. test_ce differs at every eval after step 0
(45.96 / 43.34 at step 100).

Read-out self-test:
- `compare_vals` equals `toy_struct.compare` on 6 real comparisons.
- It reproduces the audit's numbers:
  - S4: none [22700, 3100, 58100, 4600, 3800] vs R (A100) [1000, 1000, 800, 1200, 1100]; 4.60x / 3.83x; Gehan .0040;
    AUC .956 / .877 (<5k);
  - dim lr 1e-2 wd 1: 0/5 .224 / .214 / last-50 .238; wd 0.1 last-50 .181;
  - expo@1e-3 vs none lr 1e-2 wd 0.1 @50k: t_grok p .0238, AUC p .0754;
  - the Latin bounds are exact at final 1.000.
- Fake end-to-end: g1-g4 are relabelled real cells, and the g1 none gets synthetic hits (the exact path). g5 / toyshape
  are synthetic.
  - Expected and found: 3.8-4.6x "NOT '>4x'", the bitwise-identical determinism line, the speed-rule picks, "closed by
    instability".
  - The rescue verdicts are RESCUE / "some none config succeeds" / "prior does not meet" at 0.2 / 0.3 / 0.45, as
    designed.

## 8. Seven checks (experiment-audit)
| check | status |
|---|---|
| i double path | ✓: hits vs every logged subset accuracy; Latin levels vs A7 / rb6 records; compare_vals vs toy_struct.compare; read-out vs the audit's independent numbers |
| ii intent asserts | ✓: in-run lr_now (toy_struct); hits consistency (wrapper); sha + L40S checks in every job; split_sha alignment of Latin masks; complete-curve and seed-set asserts in the read-out; consumed lr / wd by optprobe |
| iii known answer | ✓: fake end-to-end with designed verdicts; self-test numbers; A7 levels |
| iv anomaly repro | – (no results); g1's none is itself a same-device re-run of toynone2 |
| v fresh adversarial reviewer | see section 10 |
| vi trap list | 1 train / eval unchanged ✓; 2 split asserts, test_fresh, Latin-implied cells separated ✓; 3 consumed lr / wd shown, wrapper A/B ✓; 4 5 seeds, single-seed dim grok flagged in (e) ✓; 5 scored at '=' (decoded) ✓; 6 one definition per metric, final vs last-50 side by side, speed rule vs rule 2 printed together ✓; 7 seed sets asserted, missing cells listed, never filled ✓; 8 tools/ untouched, toy_struct default path unchanged (same file) ✓ |
| vii decoded inputs | ✓ (smoke_jobs.log) |

## 9. Open decisions (the user's)
1. **g5 rests on the Latin premise, which section 0 / 6 corrects.**
   - Literally ("fractions where closure_latin < 1") no fraction qualifies, and g5 would be empty.
   - Under the Latin-only reading, every fraction 0.2-0.45 qualifies (0 implied cells).
   - g5 then gives the assign prior (R@always) a tuned lr, plus specrand, at toyshape's fractions.
   - It is worth it if toyshape's S4 0.2 / 0.3 turn out to be wall cells. It is 3.8 of the 5.4 L40S-h.
   - Options: submit with g1-g4; wait for toyshape's none cells; or drop it.
2. g5's R@always@1e-2 at 200k duplicates toyshape's 50k run up to 50k. It was kept because of the coordinator's lr
   list.
3. The S4 0.6 "best" uses the speed rule. Under rule 2, R@always's best is lr 3e-3, which is interior, not the top edge.
   F84's "R optimum on the top lr edge" holds only under the speed criterion.
4. The hits wrapper is new runner code. It is A/B bitwise-equal and does not touch tools/. The alternative is to call
   toy_struct.py directly: no hits, bounds only at 0.6.
5. Submission: `python /scratch/yh6210/llmadd_torch/make_jobs_toyf84fu.py --submit`. For the audit-§6 cells only, add
   `--groups g1,g2,g3,g4`. To preview: `--print [--groups ...]`.
   - After the jobs finish: `python runs/toyf84fu/readout.py --selftest && python runs/toyf84fu/readout.py`.
   - Per-cell: `toy_struct.py --summarize` runs in each job.

## 10. Independent review (check v)
Pending at the time of writing; result appended below when it returns.

### Review result (2026-10-20, fresh agent; notes/batch1_build/audit_toyf84fu_build.md)
- No blocker; g1-g4 can be submitted. The Latin correction is confirmed by an independent implementation (singles and
  GAC both derive 0 held-out cells at 0.2 / 0.3 / 0.45; 2.6-10.8 % at 0.6; 35-48 % of held-out cells at 0.6 lie in fully
  held-out intercalates, undeterminable by any pure Latin reasoning).
- To fix in readout.py BEFORE any result is read: (5) the rescue verdict silently drops a toyshape none config with a
  missing seed (readout.py:66-91, 594-626) -> must make that config / verdict PROVISIONAL; (6) the best-vs-best guard
  prints "ratio > X x" before checking the bracket (readout.py:535-537) and checks only expo's bracket; g2's best-vs-best
  line must use g1's wording rule. Minor: g5's 200k steps vs a 50k decision; hard-coded chance 1/27, 1/24; identity
  cells inside non-implied.
- Submitted 2026-10-20: g1-g4 only. g5 (the assignment prior R@always at 0.2-0.45) is NOT submitted: R's matrix target
  encodes the whole table, so by report/audits/rescue_standard.md criterion (c) it cannot support a rescue claim.

### readout.py fixes for review items I1 / I2 (2026-10-21, before any g1-g4 result was read)
- I1: `s4_low` checks the pre-registered toyshape none grid (`S4_LOW_NONE`: 4 lr x 3 wd, plus lr 1e-1 x 3 wd and wd 10 x
  4 lr at S4 0.2 / 0.3 = 19 / 19 / 12 configs) and R@always at lr 1e-3 / 1e-2. If a config is absent or incomplete,
  `read_rescue` prints "NO READING (incomplete: ...)" for that fraction. An incomplete non-R arm no longer drops
  R@always: it falls back to `arms={"R@always"}`. The Latin table still shows every arm when the cell is complete.
- I2: g4 checks the brackets first: expo lr, none lr, none wd. A ratio is printed only if all three are closed.
  Otherwise it prints "NO RATIO: best tested (bracket open on ...)", and the closing wording line says the same.
  g2's best-vs-best line uses g1's wording rule: "about a-b x" only if R is earlier on every seed with Gehan
  p < ALPHA, else "not shown". It adds "(best tested: bracket open on ...)" when the none lr / wd or R lr bracket is open.
- Known-answer tests (scratch only; the g1-g4 output root was empty, so no new result was read):
  - one seed removed from toyshape s4-f0.2-none-lr1e-2-wd1 -> "S4 0.2: NO READING (incomplete: ...)";
  - full data -> 19 / 19 / 12 none configs read, verdicts as in F89;
  - g4 on the pre-existing cells -> "NO RATIO: best tested (bracket open on expo lr ...; none wd ...)";
  - `--selftest` output unchanged.
- Not changed (minor items M1-M9 stay as listed): hard-coded chance, M4 median-of-differences, M6 tie rule.

## Amendment A100 (2026-10-21, user: 「需要现在该提交的都可以提交」; written before any g5 result exists)
- **What**: the rescue test at the S4 wall cells 0.2 / 0.3 (wall cells per the F89 addendum), run entirely on A100
  (a100_tandon, torch_pr_520_tandon_priority). 26 jobs, about 12 GPU-h:
  - the toyshape none grid: 19 configs x 5 seeds x 200k per fraction (runs/toyshape_a100);
  - the toyshape R@always cells: lr 1e-3 / 1e-2, 50k, with their shape / ctl arms;
  - g5: R@always + specrand@always at lr 3e-3 / 1e-2 / 3e-2, 200k (runs/toyf84fu_a100).
- **What is the same**: every job file is a sed copy of the L40S one; only the job name, account, partition, out root
  and HOLD path change (diff-checked, 0 unexpected differences). The code dirs are the same (toyshape d182ce548bcc,
  toyf84fu b74a78cb5472) and are read-only. The L40S results are never mixed in (criterion (f)). The L40S copy of g5 is
  not submitted, because the A100 grid supersedes it.
- **Reading** (unchanged rule, `read_rescue`):
  `python runs/toyf84fu/readout.py --toyshape_root runs/toyshape_a100/out --new_root runs/toyf84fu_a100/out`, at 0.2 / 0.3.
  - "Every none config fails": every one of the 19 configs <= 1/5 groks, median final and last-50 <= 0.20, and late
    gain < 0.02, on non-implied cells.
  - R@always's lr is the best median AUC_B over the five A100 R cells.
  - "Meets": >= 4/5 grok within B = 50k on non-implied cells, and the paired test passes against none's best-AUC_B config
    on both the non-implied cells and test_fresh.
  - Specificity: vs specrand@always at that lr. If it fails there, the verdict carries "SPECIFICITY NOT SHOWN" and
    criterion (d) is not met.
  - Before reading, every gpu line must show an A100; otherwise WITHHELD.
- **Tier**: R@always's matrix target encodes the whole table, so a positive reading is at most a Tier L rescue
  (rescue_standard Amendment 1: "the model learns to use a given structure"). It must never be called an S-tier or
  unqualified rescue.
- **(e), conditional, submitted without further approval if triggered**: if a fraction reads RESCUE (and specificity
  holds), the whole A100 grid at that fraction is re-run on 5 fresh seeds: the first 5 seeds >= 100 that pass
  toy_struct's split asserts, listed before submission. The grid is the none 19 configs, R@always at the chosen lr, and
  specrand@always. The claim requires the same verdict there. Otherwise the reading is reported as "not a rescue".
- 2026-10-21 23:55 ET, amendment A100 fix: the 6 g5 copies exited 4 at start ("not an L40S"). Their job file guards the
  GPU type (`case "$gpu" in *L40S*`); the sed clone kept that guard, and the clone's diff check only compared the
  copies with their originals, so it could not catch a site-specific line. No run trained and no result was written
  (out/ holds gpu.txt only). Fix: the guard now requires an A100, plus one comment line; every other line still equals
  the L40S job. All 6 resubmitted. The toyshape_a100 jobs have no such guard; 17 had completed by then.
