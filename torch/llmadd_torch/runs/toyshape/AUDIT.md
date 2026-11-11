# toyshape: wall 2. Does a SHAPE-ONLY prior break the inductive-bias (sample-complexity) wall? (2026-10-19)

Context: WRITING_HANDOFF.md, 2026-10-19 section "先验能破哪些墙". Wall ② (inductive bias / sample-complexity threshold) is
still to be tested: "只给形状、不给分配" (shape given, assignment not), with a permutation-invariant spectral prior, on mod / S4
at train fractions so small that none cannot learn.
The handoff's rule stands: no "the prior makes the unlearnable learnable" until ① or ② has an audited result.
Builds on:
- F45 / F59: mod 113 at 0.3. The single circle always-on hurts; the 1-5 torus always-on is about 4x faster.
- F84 + the 10-19 addendum: S4 "not learned without the prior" was an lr artefact. At lr 1e-2, none groks S4 0.6 5/5.
- toynone / toynone2 / toynone3 / toyprior_lr: none lr x wd sweeps. The --lr bug of 10-18.

Question: at train fractions where the best-tuned none fails, does a prior that fixes only the geometry of the item
embeddings let the one-layer toy learn the held-out table? That geometry is the centred-Gram spectrum plus the norm
profile (kind `shape`), optionally plus every item's sorted similarity row (kind `shapeprof`). The prior never says which
item sits where. The checks:
- is this specific to the right shape? Two controls under the same loss: a wrong shape of the same rank (`line`,
  anisotropic) and a flat constant-norm sphere of another rank (`sphere`, same isotropy and norms as the true shape);
- what does spec + norm actually encode? For the torus and S4 it is only "a flat constant-norm r-sphere", so a pass of
  `shape` is read as a rank-r isotropic prior; "geometry beyond rank" needs `shapeprof` to beat `shape`;
- how much of the assign prior's gain does it keep? (assign = shape + assignment, the F84-style CKA prior)
- what data multiplier is it worth?

**Revision 2026-10-19 (before any run):** an independent adversarial audit (experiment-audit check v) found that the
first version of the reading could report a non-specific or non-wall effect as "BREAKS THE WALL", and that the GPU
eigendecomposition was untested. Everything below is the revised, still pre-run, pre-registration; the findings and fixes
are listed in "Independent audit 2026-10-19". The first version (toy_struct.py e1bae1b77663) was never submitted.

## Binding lessons and how each is met
| lesson | how |
|---|---|
| (a) a new flag needs an A/B showing that a non-default value changes behaviour, plus an in-run log / assert of the consumed value | shape arms (sphere included): A/B vs none, vs assign, sphere vs the true shape (trajectories differ; own objective falls; assignment not imposed); `prior_kind`, `shape_terms`, `shape_check` (permutation / target asserts on the device), `shape_target_rank` and `shape_eig` are in every shape-arm JSON; `--shape_log` / `--mod_twins`: A/B logging-only (curves bitwise equal) plus the consumed flag in the JSON; `--lr` reaches the shape arms (A/B, lr_now logged and asserted, the existing assert untouched); generator options `--ext` / `--noprof` / `--only`: rendered job sets differ as intended (84 / 94 / no shapeprof arms), `--only` refuses unknown names and use without `--submit` |
| (b) the default path stays bitwise identical to b74a78cb5472 | preflight/regress_default.py (CPU, 1 thread, old vs new), rerun on the final file d182ce548bcc: 7 training cases (s4 / dim / mod / torus / non-default lr, wd / s4 0.2 seed 5), every JSON key except code_sha / sec / ms_per_step equal; `--check` stdout equal; `--summarize` byte-equal on 7 real result dirs (toystruct, toynone2, toyprior_lr) |
| (c) "cannot learn" survives none's own lr x wd sweep with the optimum bracketed, and a 4x budget | none at lr {1e-3, 3e-3, 1e-2, 3e-2} x wd {0.1, 1, 3} at 4 B at every fraction; a wall cell additionally needs every none config at or below the 0.20 level and not still rising (rule 1); a FLAT cell is not a closed bracket until the one-step extension (lr 1e-1, wd 10) has run (rule 2: the F84 trap); the wall reading uses the best of all none configs |
| (d) intent spec with PRE-REGISTERED rules | this file, written before any run and revised before any run after the independent audit; implemented in toy_struct.py:summarize_wall and tested on 14 synthetic cells with known verdicts |
| project rule: symmetry-clean held-out (audit_port_fixes) | S4: non-commutative, and test_fresh is logged. mod: an unordered-pair CLASS split is not viable at 0.05: 2-9 answers and, for seeds 3 / 4, one item never occur in training, so the held-out ceiling is ~0.92 < 0.95. So mod keeps toy_grok's question split, logs test_fresh (held-out questions whose commuted twin is not in training; `--mod_twins`), and every AUC pass must also hold on test_fresh; a run without test_fresh gives the cell no verdict |

## Code and provenance
- Trainer: llmadd-burst `tools/toy_struct.py`. The additions are opt-in; every pre-existing arm / flag / output is unchanged
  (regression above).
  - Arms: `shape:<target>` / `shapeprof:<target>`, with the usual `@always` / `@t0-t1`. Targets: the task's own, plus the
    controls `line` and `sphere`.
  - Flags: `--shape_log` (logging), `--mod_twins` (logging), `--summarize_wall` (reading).
  - Refactor: `compare()` -> `compare_values()`, same decisions and same key order.
  - The file is UNCOMMITTED (public repo, no agent commits). It is copied once into `runs/toyshape/code/tools/` with
    `code/COMMIT` (HEAD + sha256[:12] of each copied file).
  - Snapshot: HEAD 4ada6e6 + toy_struct.py **d182ce548bcc**, the code_sha every result must carry. Also copied:
    test_toy_struct.py 1533ad98f6c0, toy_grok.py bec2858158d3, test_toyshape.py, regress_default.py, double_path.py.
    The first snapshot (e1bae1b77663, pre-audit) was moved out of the run dir before any submission.
  - The generator pins its sha (PIN_SHA). It refuses another repo file or snapshot, and every job re-checks the
    snapshot's sha before any GPU work (exit 3).
- Tests: `preflight/test_toyshape.py` (wall-2 additions), `preflight/regress_default.py` (default path),
  `preflight/double_path.py` (check i), `preflight/smoke_jobs.py` (rendered jobs). Logs are in preflight/; the snapshot
  carries copies of the scripts.
- Generator: `llmadd_torch/make_jobs_toyshape.py [--ext] [--noprof] [--submit [--only a,b]]`. It writes
  `job_<name>.sbatch`, asserts each rendered flag set, and asserts that every (cell dir, arm) is written by exactly one
  job. It does not submit (`--submit` is the user's call); HOLD switch and SUBMITTED_ flags as in ptlr / toynone2.
- Model / optimiser = toy_struct (= toy_grok): one layer, d 128, 4 heads, MLP 512, no LN, full batch, AdamW betas
  (0.9, 0.98), 10-step warmup to --lr. Same seed = same init and same split for every arm, so arms are paired by seed.

## The shape-only prior (exact)
H = W_E[items] (n x 128; mod n 113, s4 n 24); G = centred Gram of H; T = the target configuration, G_T its centred Gram.
- spec = 1 - <lam(G), lam(G_T)> / (||G||_F ||G_T||_F), lam = eigenvalues sorted descending.
  - By von Neumann this is 1 - max over item-side orthogonal maps of CKA(H, T): the assign prior with the assignment
    maximised out.
  - spec <= 1 - CKA always; tested on 150 random draws, min gap +0.21.
  - The eigendecomposition runs on the CPU in float64 (LAPACK dsyevd) whatever the arm's device; the rest of the loss
    stays on the arm's device. Reason: cuSOLVER float32 on a Gram with ~111 near-zero eigenvalues was never tested and
    could fail to converge (audit finding 5). Cost: one host round trip per step (the old NaN check already synced).
- norm = 1 - cos(sort(diag G), sort(diag G_T)): the profile of squared centred norms. For circle / torus / R / sphere it
  is constant.
- prof (kind `shapeprof` only) = 1 - sum_i <sort(G[i, :]), s_T> / (||rows||_F sqrt(n) ||s_T||): every item's sorted
  similarity profile. Only for vertex-transitive targets (all rows share one profile s_T; asserted).
- Losses: `shape` = spec + norm; `shapeprof` = spec + norm + prof. Each term is in [0, 1]. The prior added to the CE is
  lam * loss with lam 1, like the assign prior's lam * (1 - CKA).
- Invariance: P G P^T has the same spectrum, the same sorted diagonal, and the same set of sorted rows. So all terms are
  invariant under any item permutation, and 0 for T and every permutation of it.
- Targets:
  - mod `circle` = cos/sin(2 pi k a / 113) for --freqs: freq 1 (circle, rank 2) or 1-5 (torus, rank 10, F59).
  - s4 `R` = the 3x3 rotation matrices flattened (9 equal eigenvalues, ||R||_F^2 = 3 for every item).
  - `line` = the wrong shape of the same rank r: path-graph modes cos(pi k (i + 1/2) / n) / k, k = 1..r. This is an open
    chain with spectrum ~ 1/k^2 and non-constant norms (anisotropic; participation ratio 1.5-2.2).
  - `sphere` = the matched control: a flat (r' equal eigenvalues), constant-norm, centred configuration of rank r'
    (an equal-norm tight frame, built deterministically by alternating projections, flat and constant to 1e-9, asserted).
    r' = mod r + 1 (3 for the circle, 11 for the torus: odd, so no union of Fourier circles, which come in pairs);
    s4 12 (not a subset sum of S4's centred isotypic dimensions 1, 4, 9, 9; 10 = 1 + 9 would be). It has the true
    shape's isotropy and norm profile at another rank, so "shape beats sphere" means the rank matters, and "sphere breaks
    the wall too" means a generic isotropic constant-norm prior does it. Note: sphere(r + 1) can hold the true shape plus
    one extra isotropic dimension (e.g. a Fourier circle x a sign bit), so it is a strict control.
- What the minimum set contains. The prior never fixes the assignment, so these are expected, not bugs:
  - mod circle (spec + norm): 113 points on a circle (constant norm) in any plane, with isotropic second moment, and ANY
    assignment. They need not be equally spaced; `prof` forces the regular 113-gon. Because 113 is prime, all 56
    Fourier circles (a -> k a) lie in this set, and so do the shuffled circles.
  - mod torus (spec + norm): any 113 points with constant norm on a flat 10-d sphere. This does not say "five circles".
  - s4 R (spec + norm): any 24 points with constant norm spanning a flat 9-d sphere.
    - R_sign is included (same spec, norm and profile, asserted 0).
    - So is a Z24 9-d orbit (asserted 0). Only `prof` separates it (0.063).
  - Hence a `shape` pass is worded as "a rank-r isotropic constant-norm prior" (rule 5), never "the right geometry".
- No answer injection (known answer, tests section 1): minimising the shape loss alone from a random init reaches
  ~1e-5, while the CKA to the assigned target stays at chance. Circle 0.027, torus 0.091, R 0.360 (chance for a random
  9-d subspace 9/23 = 0.39).

Shape distance (spec + norm; [prof]) between targets and the seed-0 init, from `--check --shape_log` (sphere rows are
the known answer 1 - sqrt(r / r'), asserted to 1e-5):
```
s4 (n 24)       ->  R              R_sign         absR           line     sphere (r 12)
  R                 0.000 [0.000]  0.000 [0.000]  0.333 [0.167]  0.604    0.134
  absR              0.333 [0.167]  0.333 [0.167]  0.000 [0.000]  0.413
  line (r 9)        0.604 [0.160]  0.604 [0.160]  0.413 [0.197]  0.000
  sphere (r 12)     0.134
  W_E_init seed 0   0.167 [0.161]  0.167 [0.161]  0.336 [0.324]  0.510
mod (n 113)     ->  circle (freq 1)   line (r 2)   sphere (r 3) | torus (1-5)    line (r 10)  sphere (r 11)
  target            0.000 [0.000]     0.279        0.184        | 0.000 [0.000]  0.625        0.047
  W_E_init seed 0   0.659             0.785                     | 0.341 [0.222]  0.690
```
The torus and sphere(11) are close under spec + norm (0.047): for the torus the sphere control is strict.
Participation ratio (effective dimension):
| target | value |
|---|---|
| circle / line r 2 / sphere r 3 | 2.0 / 1.47 / 3 |
| torus / line r 10 / sphere r 11 | 10 / 2.2 / 11 |
| R / line r 9 / sphere r 12 | 9 / 2.2 / 12 |

## Design
| | mod (p 113, question split, toy_grok's) | s4 (24 rotations, question split) |
|---|---|---|
| train fractions | 0.05, 0.1, 0.15, 0.2, 0.3 | 0.2, 0.3, 0.45 |
| seeds | 0-4 | 0-4; at 0.2: 0, 1, 2, 4, 5 (seed 3's split leaves an item that is never an answer in training, the existing split assert; rule = the first five seeds whose split passes) |
| prior budget B | 25,000 (toy_grok / F45 / F59) | 50,000 (F84 at 0.3 / 0.45) |
| none budget | 4 B = 100,000 | 4 B = 200,000 |
| none optimiser grid | lr {1e-3, 3e-3, 1e-2, 3e-2} x wd {0.1, 1, 3} | same |
| flat-cell extension (rule 2) | lr 1e-1 x wd {0.1, 1, 3} and wd 10 x lr {1e-3, 3e-3, 1e-2, 3e-2} (not the corner: lr x wd = 1 zeroes AdamW's weights each step) | same |
| prior arms optimiser | lr {1e-3, 1e-2}, wd 1, lam 1 | same |
| variants | circle (--freqs 1), torus (--freqs 1,2,3,4,5) | R |

Prior arms per (variant, fraction, lr), two jobs (main / ctl):
| role | job | always-on | [0, 2000) window |
|---|---|---|---|
| assign (CKA to the fixed target: upper reference) | main | mod `always`; s4 `R@always` | mod `w0`; s4 `R` |
| shape (spec + norm): TESTED | main | `shape:circle@always` / `shape:R@always` | `shape:circle` / `shape:R` |
| wrongshape (line, same rank, anisotropic): CONTROL | main | `shape:line@always` | `shape:line` |
| rankctl (sphere, rank r', isotropic, constant norm): CONTROL | ctl | `shape:sphere@always` | `shape:sphere` |
| shapeprof (spec + norm + profile): TESTED (default since the audit; `--noprof` drops it) | ctl | `shapeprof:circle@always` / `shapeprof:R@always` | `shapeprof:circle` / `shapeprof:R` |

Every run carries `--shape_log`, so each arm's shape distance to each target (and for mod to both the freq-1 circle and
the 1-5 torus, `shapeL_fourier1` / `shapeL_fourier1-5`) is logged every 100 steps. Every mod run carries `--mod_twins`.
Splits per seed (checked): s4 closure 1.00 at 0.2 / 0.3 / 0.45 for all main seeds; majority-chance 0.019-0.035.

mod twin share (held-out with the commuted twin trained) at seeds 0-4:
| frac | share |
|---|---|
| 0.05 | 0.049-0.050 |
| 0.1 | 0.097-0.101 |
| 0.15 | 0.145-0.151 |
| 0.2 | 0.195-0.200 |
| 0.3 | 0.291-0.301 |

mod 0.05: seeds 0 and 1 miss one answer class in training (held-out ceiling ~0.99); the other seeds miss none.

## Train vs eval
| dimension | train | eval | same? |
|---|---|---|---|
| template / position | (a, b, =), answer read at the last token | same | yes |
| questions | int(frac N) questions (toy_grok's draw) | the rest; test_fresh = those whose (b, a) / commuting twin is not trained | no, by design |
| items | every item an input; s4 every item an answer (asserted); mod 0.05: <= 1 answer class unseen (seeds 0, 1) | same items | yes |
| model state / batch / precision | no dropout or norm layers; full batch; fp32 (the shape loss's eigendecomposition in float64 on the CPU) | no_grad, all held-out every 100 steps | yes |
| prior | on the item rows of W_E only, window or always | none at eval | n/a |

Decoded inputs are printed side by side for 3 train / 3 held-out questions in every seed's log; see the smoke output in
Tests.

## Metrics
- t_grok, AUC, final test: as in toy_struct / F84.
- test_fresh, every eval: s4 commuting twins; mod twins via `--mod_twins`.
- The wall reading uses the first B steps of every run, `upto()`:
  - AUC_B = mean held-out test over the evals at steps 0, 100, ..., B - 100 (B / 100 evals). The same evals for a prior
    arm run B steps and a none run 4 B; the prior's extra last eval at B - 1 is left out.
  - t_grok_B = the first of these evals with test >= 0.95.
  - Exact, because the lr schedule does not depend on the total steps.
- Late change of a none run (`late_change()`, rule 1): over the last 25 % of 4 B, the mean of the last 5 evals minus the
  mean of the first 5 evals at or after 0.75 x 4 B, for held-out accuracy (gain) and test CE (drop, nats). A diverged
  (non-finite) run counts as no change.
- geo for a shape arm = its own shape loss (spec + norm [+ prof]), with the terms per eval. The cka_<target> keys still
  show how close the embedding comes to the assigned target.
- shapeL_<target> (`--shape_log`) = spec + norm distance of any arm (none included) to each shape target.

## PRE-REGISTERED reading (2026-10-19, before any run, revised after the independent audit; toy_struct.py:summarize_wall, tested on synthetic cells)
`--summarize_wall runs/toyshape/out` reads the cell dirs directly under out/. It refuses runs from more than one
code_sha (NO VERDICT) and gives a cell no verdict if any of its runs lacks test_fresh.
1. **Wall cell.** A (task, fraction) is a wall cell iff EVERY none config of the sweep that ran (12, plus any extension
   cells), at 4 B:
   - (a) groks on <= 1/5 seeds;
   - (b) has a median final held-out test <= 0.20 (critique D1's level);
   - (c) is not still rising: median last-25 % accuracy gain < 0.02, and for the best config (rule 2) also a median
     last-25 % test-CE drop < 0.01 nats (D1).
   Otherwise "not a wall cell (reason)": speed-up / lift reading only, never "breaks the wall".
2. **Bracket.** The best none config = most groks within 4 B, then the highest median AUC over 4 B.
   - Closed iff it is not on an edge of the grid that ran (lr bottom / top, wd bottom / top).
   - If it sits on an edge, the cell's none is extended once, one x3 step beyond that edge (lr 1e-1 / 3e-4; wd 10 / 0.03);
     the generator's NONE_EXT block takes these conditional cells. The cell is then re-read with every config.
   - "Flat" = every config at 0 groks with AUC at or below the floor line: no optimum was found, which is not a bracket
     (the F84 trap: S4 none sat at chance at lr 1e-3 and learned at 1e-2). A flat cell's bracket stays OPEN until the
     one-step extension lr 1e-1 x wd {0.1, 1, 3} and wd 10 x the 4 lrs has run (generator `--ext` queues it up front for
     the likely wall cells); flat over the extended grid then counts as closed.
   - A wall verdict with an open bracket is prefixed "BRACKET OPEN" and cannot be a headline.
3. **Choices.** Each prior role and schedule takes its lr (1e-3 or 1e-2) with the highest median AUC_B. The paired tests
   use the none config with the highest median AUC_B. That is 12+ choices vs 2, which favours none.
4. **Tests.** compare_values(a, b) on AUC_B and t_grok_B, paired by seed, unchanged from F84:
   - wins >= 4/5; exact one-sided MWU p < 0.025; AUC median difference >= 0.05;
   - the role's median AUC above the floor line (chance + 0.05; mod chance 1/113, s4 max(1/24, median majority)).
   - A role MEETS the rule iff it groks within B on >= 4/5 seeds AND its AUC test vs best none passes on held-out test
     AND on test_fresh.
   - "a beats b" iff the paired AUC test a vs b passes on held-out test AND on test_fresh.
5. **Verdict at a wall cell, per tested role (shape, shapeprof) and schedule (always = PRIMARY, window = secondary).**
   - A control (line or sphere) meets the rule too -> "breaks it, but so does <control>: NOT SHAPE-SPECIFIC".
   - Else the role meets it and beats BOTH controls -> "BREAKS THE WALL", worded by what the loss encodes:
     - shape: "the true shape's spectrum + norms: a rank-r isotropic constant-norm prior";
     - shapeprof: "spectrum + norms + row profile", plus "geometry beyond rank SHOWN" iff shapeprof also beats shape,
       else "geometry beyond rank not shown".
   - Meets it without beating both controls -> "breaks it; SPECIFICITY NOT SHOWN (vs each control: pass / n.s.)".
   - AUC pass (test and test_fresh) without the groks -> "partial".
   - "floor: uninformative" iff compare says floor; otherwise "not shown at n=5", never "no effect".
   - At a non-wall cell: "not a wall cell (reason): speed-up reading only".
   - The circle and torus variants and the S4 cells are read separately; shape and shapeprof are read separately (no
     pooling). Torus = a second, pre-specified mod target: if only one variant breaks the wall, name that variant.
   - **Headline** for a task: a PRIMARY verdict "BREAKS THE WALL" (no prefix: wall cell, closed bracket, complete grid)
     at >= 1 wall cell, reported with the full verdict pattern across fractions. "The right geometry" (beyond an
     isotropic rank-r prior) can only come from shapeprof with "geometry beyond rank SHOWN".
6. **Data multiplier.** For each role and schedule:
   - A = median final test at B (the role's chosen lr).
   - f* = the smallest swept fraction at which the best none config's median final test (at 4 B) >= A.
   - Multiplier = f* / f, given as a bracket between neighbouring grid fractions; "> 0.3 / f" if none never reaches A.
     The mod grid is coarse (0.05 steps), so the multiplier is a bracket, not a number.
7. Reported with it, no claim alone:
   - assign vs none: the upper reference; a wall that assign cannot break either says the cell is too hard for any
     embedding prior;
   - shape vs line, shape vs sphere, shapeprof vs shape (paired AUC_B tables);
   - the share of assign's lift, (AUC shape - AUC none) / (AUC assign - AUC none);
   - none's own distance to the variant's shape at B and 4 B (shapeL_fourier1 / shapeL_fourier1-5 / shapeL_R): does
     the unaided net reach the shape when it learns?
   - CKA to the assigned target in shape arms at the end (does data + shape find a valid assignment?).
8. Nothing (arm, fraction, lr, seed, budget, lam) is added after results without the label "post hoc", except the
   pre-registered bracket extensions of rule 2. The window arms are secondary throughout.

What the result can and cannot say:
- "BREAKS THE WALL" for shape = in this toy, a prior that fixes only the true shape's rank, isotropy and norm profile,
  without the assignment, lets the net learn the held-out table from a fraction at which none, at its best of 12+
  optimisers and 4x the steps, neither learns nor is still rising; and neither an anisotropic same-rank shape nor an
  isotropic other-rank sphere does so.
- It is still the toy (one layer, nonce items, closed table, full batch), not an LLM.
- A failure means "a spectral-shape prior of this strength does not". shapeprof is the stronger form, now in the default
  design; a negative for both is still specific to these loss forms and lam 1.

## Expected ranges (anything outside -> look before reading)
- Every shape arm: own shape loss <= 0.05 after a few hundred steps while on.
  - A/B, lr 1e-3: s4 0.6 shape:R 0.166 -> 0.0009 in 300 steps. mod 0.05 by step 149: circle 0.656 -> 0.047, torus
    0.339 -> 0.036, line 0.78 -> 0.046. Sphere: see Tests.
  - After a window closes it may drift back.
  - CKA to the assigned target stays near its init / chance unless the data pick the assignment. mod circle init ~0.03;
    s4 R init ~0.57 (isotropic 24 points in 128-d), and shape-only optimum ~0.36.
- assign mod circle @always: may HURT memorisation and grokking (F45: rank-2 circle always-on at 0.3 grokked 1/5).
  The torus @always helped there (F59, 4/5). Neither is a bug.
- none: memorises (train 1.00) in a few hundred steps at every lr except perhaps 3e-2 and the extension lr 1e-1.
  - Held-out at chance or below for s4: Latin-square memoriser signature.
  - Possible divergence at lr 3e-2 / 1e-1 / wd 10: a non-finite ce counts as not learning and is reported.
- s4 0.45: none probably learns at lr 1e-2 within 200k (0.6 groks 5/5 at 4.6k), so 0.45 is likely not a wall cell.
  s4 0.2 / 0.3 and mod <= 0.15 are the likely wall cells; each could also be a floor for every arm (F84: R@always at
  0.3, lr 1e-3, stayed at chance).
- Shape arms' ms/step above the CKA arms (a host round trip plus a float64 113 x 113 / 24 x 24 dsyevd per step):
  ~1-2 ms more on an idle CPU core; unmeasured on a GPU node, see Cost.

## Tests (preflight/; outputs in preflight/*.log; final runs on d182ce548bcc, CPU, 1 thread)
Logs of the pre-audit file e1bae1b77663 are kept in preflight/pre_audit_e1bae1b77663/.

**Regression, default path (regress_default.log; lesson b).** Old snapshot b74a78cb5472 (git 398935a) vs d182ce548bcc,
same args, separate dirs.
- Training: 26 runs, 1756 logged curve values; every JSON key except code_sha / sec / ms_per_step equal (floats exact),
  stdout equal:
  - s4 0.6 (none, R, R_sign, absR, specrand, shuf, R@always);
  - s4 0.45 at lr 1e-2 wd 0.1 (none, R@always);
  - s4 0.2 seed 5 (none, R);
  - dim 0.3 (7 arms incl. specrand_oh);
  - mod 0.3 (none, always, w0, always_shuf; eval every 50);
  - mod torus (none, always);
  - mod 0.05 at lr 3e-2 wd 3 (none, w0).
- `--check` stdout identical for s4, dim, mod, mod torus (seeds 0, 1).
- `--summarize` gives byte-identical summary.md / summary.json / stdout on copies of 7 real result dirs:
  - toystruct s4-f0.6-always, s4-f0.45-always, dim-f0.3-oh, s4-f0.9-posctl;
  - toynone2 s4-f0.6-lr1e-2-wd1, dim-f0.3-lr1e-2-wd0.1;
  - toyprior_lr s4-f0.6-lr3e-3-wd1.
- Transitively, the mod path still equals toy_grok. test_toy_struct's known answer (4800/4800 bitwise) ran on revision
  a099a1edaece; each later change kept the lr-1e-3 mod path bitwise, by its own regression and by this one.
- tools/test_toy_struct.py --skip_known on the new file (structure, decision rules, smoke): ALL TESTS PASSED
  (test_toy_struct.log).
- Not mine: tools/pretrain_range.py shows as modified in the repo (another piece). This experiment's snapshot copies
  only toy_struct.py, test_toy_struct.py and toy_grok.py.

**Wall-2 tests (test_toyshape.log): ALL WALL-2 TESTS PASSED.**
1. Shape loss known answers:
   - sphere controls: rank 3 / 11 / 12, spectrum max/min - 1 <= 2.7e-15, norm max/min - 1 <= 3.1e-12, deterministic;
   - sphere vs the true shape = 1 - sqrt(r / r') to 1e-5: circle 0.1835, torus 0.0465, R 0.134 (Z24 too);
   - float32 input: terms come back in float32; float32 vs float64 path 1.1e-7;
   - target and 25 permutations of it, every target and kind: |loss| <= 2.0e-6 (float32);
   - permutation invariance on random embeddings: max |dL| 3.0e-7 (float32) / 1.0e-15 (float64);
   - random embeddings (min of 10 draws): circle 0.650, torus 0.342, R 0.160, line 0.52-0.76, sphere 0.11-0.59;
   - wrong shapes: line2 -> circle 0.279, line10 -> torus 0.625, line9 -> R 0.604, absR -> R 0.333;
   - spec <= 1 - CKA on 150 draws (min gap 0.187);
   - R_sign = 0 under shape and shapeprof; Z24 orbit 0 under shape, 0.063 under shapeprof;
   - shape-only Adam reaches ~0 (shapeprof 0.020) with CKA to the assigned target at chance: circle 0.027, torus 0.091,
     R 0.360, sphere3 0.005; every item row gets a finite nonzero gradient;
   - NaN / inf embeddings give NaN terms, no exception;
   - invalid arms are refused before training (shape:specrand, shapeX:R, shape:circle on s4, shapeprof:line,
     shapeprof:sphere, shape:fourier1).
2. A/B on training (s4 0.6, 300 steps, seed 0; mod 0.05, 150 steps):
   - Every shape arm's ce differs from none's and from assign's at every eval after step 0; sphere differs from the
     true shape.
   - Own loss: shape:R@always 0.166 -> 0.0009 (none drifts to 0.079); shape:line@always 0.508 -> 0.0025 (none 0.486);
     shapeprof:R@always 0.326 -> 0.026; shape:sphere@always 0.123 -> 0.0005 (none 0.089).
   - CKA to R: 0.338 under shape, 0.433 under sphere, vs 0.999 under assign (assignment not imposed).
   - Window arm (--win 0,150): prior_on [T, T, F, F]; equals @always at step 100 and differs from step 200.
   - In the JSON: prior_kind / shape_terms / shape_check / target / shape_target_rank (R 9, line 9, sphere 12; mod
     2 / 2 / 3 and torus 10 / 10 / 11) / shape_eig; lr_now = 1e-3 after warmup.
   - `--lr 1e-2` on a shape arm: lr_now 0.01, a different trajectory.
   - mod 0.05: circle shape 0.656 -> 0.047 (none 0.529), cka_circle 0.061 vs assign 0.953; torus 0.339 -> 0.036
     (none 0.157), cka 0.140 vs 0.930; line 0.78 / 0.69 -> 0.047 / 0.033; sphere 0.59 / 0.32 -> 0.046 / 0.034.
   - mod readouts: none logs shapeL_fourier1 0.529 and shapeL_fourier1-5 0.157 in both variants; the run's own
     variant equals shapeL_circle (second path).
3. `--shape_log`: logging only. none and R@always are bitwise equal on every shared key; the shapeL_* keys (sphere
   included) appear only with the flag.
   `--mod_twins`: logging only. mod 0.05 seed 0 n_fresh 11535 / 12131 (twin share 0.049), equal to a second-path count;
   curves bitwise equal; refused for s4.
4. summarize_wall on 14 synthetic cells, verdicts as pre-registered (shape@always unless noted):
   - 0.05 "BREAKS THE WALL (the true shape's spectrum + norms: a rank-2 isotropic constant-norm prior; beats line and
     sphere, paired)", bracket "flat over the extended grid"; shapeprof there "BREAKS ...; geometry beyond rank not
     shown"; a test-CE drop on a non-best none config does not block the wall;
   - 0.07 shapeprof "BREAKS ...; geometry beyond rank SHOWN (shapeprof > shape, paired)";
   - 0.1 "NOT SHAPE-SPECIFIC" via line; 0.12 via sphere;
   - 0.13, the auditor's case (line groks 3/5 at the same steps): "SPECIFICITY NOT SHOWN (vs wrongshape: n.s., vs
     rankctl: pass)";
   - 0.15 not a wall cell (none groks 3/5); 0.16 above the 0.20 cap (none at 0.50); 0.17 still rising (gain +0.023);
     0.18 the best config's test CE still drops (+0.050 nats);
   - 0.2 "BRACKET OPEN (OPEN at lr top ...): partial"; 0.22 flat without the extension: "BRACKET OPEN (... flat ...)"
     and no headline;
   - 0.25 a lift on test but not on test_fresh is refused ("not shown at n=5");
   - 0.3 "floor: uninformative"; 0.35 a run without test_fresh: "no verdict (test_fresh missing)";
   - headline = exactly the three prefix-free PRIMARY BREAKS (0.05 shape, 0.05 shapeprof, 0.07 shapeprof);
   - the best lr is picked by AUC_B; multiplier at 0.05 = 3.00 (f* 0.15);
   - mixed code_sha: NO VERDICT; a root above out/ reads nothing;
   - late_change refuses a full-length curve without late evals, gives (0, 0) for a smoke-length or diverged run;
     upto() aligns B and 4 B runs and refuses an incomplete curve; compare() keeps its key order.

**Double path (double_path.log).** toy_struct's shape loss vs an independent numpy implementation: max |diff| 5.6e-15
(8 targets incl. line and sphere x 20 draws x kinds, float64).

**Generator option A/B.** Default: 84 jobs, 1780 runs. `--ext`: 94 jobs (10 extension jobs, wd 10 and lr 1e-1).
`--noprof`: 84 jobs, the 26 ctl jobs carry no shapeprof arm. `--only` without `--submit`, `--only` with an unknown name and
an unknown option are refused before any sbatch call. Every (cell dir, arm) is written by exactly one job (asserted).

**Smoke of rendered jobs (smoke_jobs.log).** Ten jobs, rendered with `--ext`: mod none 0.05 lr 1e-2; mod circle and
torus prior 0.05 lr 1e-2, main and ctl; s4 none 0.2 lr 1e-2; s4 prior 0.2 lr 1e-2, main and ctl; mod 0.05 wd-10
extension; s4 0.2 lr-1e-1 extension. The script takes the job files' own command lines, steps / 250, first seed, CPU,
from the snapshot code:
- 19 commands exit 0;
- every JSON code_sha d182ce548bcc, lr / wd / steps as rendered, lr_now at the end = the job's lr (0.001-0.1);
- every prior cell carries all 10 arms (assign, shape, line, sphere, shapeprof; always and window);
- per-cell --summarize runs; --summarize_wall runs over the smoke root (PROVISIONAL, grid incomplete, as expected;
  preflight/smoke_wall_summary.md).
Decoded inputs from the first command's log (mod 0.05, seed 0; the scored position is the last token):
```
[mod frac 0.05 seed 0] n_train 638 n_test 12131 mod_twins True
  train ids [89, 3, 113] len 3 | 89 3 ⟦=⟧ (pos 2) -> 92
  train ids [94, 78, 113] len 3 | 94 78 ⟦=⟧ (pos 2) -> 59
  train ids [103, 104, 113] len 3 | 103 104 ⟦=⟧ (pos 2) -> 94
  eval  ids [2, 25, 113] len 3 | 2 25 ⟦=⟧ (pos 2) -> 27
  eval  ids [32, 80, 113] len 3 | 32 80 ⟦=⟧ (pos 2) -> 112
  eval  ids [68, 49, 113] len 3 | 68 49 ⟦=⟧ (pos 2) -> 4
```

## Cost (L40S, 5 seed processes per GPU; generator output)
Each job holds one L40S for its wall time = per-process steps x per-process ms/step. Planning rates (ms/step/process,
5 processes sharing the GPU), measured on L40S in runs/toynone2 and toyprior_lr:
- s4 none: 3.0;
- s4 CKA prior: 3.7.
Not measured on a GPU node yet (no GPU on the session host):
- mod: planning 6, worst 25. The only mod GPU number is toy_grok on A100 / H100 without --threads 1: 22.9 ms/step.
- shape arms' CPU eigendecomposition plus a host round trip per step: planning +1.5 (mod) / +1 (s4), worst s4 10.

| group | jobs | runs | steps per process | planning L40S-h | worst case (mod 25 / s4 10 ms) | --time |
|---|---|---|---|---|---|---|
| none mod (5 fracs x 4 lr; 3 wd in sequence) | 20 | 300 | 3 x 100k | 10.0 | 41.7 | 4 h |
| none s4 (3 fracs x 4 lr) | 12 | 180 | 3 x 200k | 6.0 | 20.0 | 3 h |
| prior mod main (2 variants x 5 fracs x 2 lr; 6 arms) | 20 | 600 | 6 x 25k | 6.2 | 20.8 | 2 h |
| prior mod ctl (sphere x 2 + shapeprof x 2) | 20 | 400 | 4 x 25k | 4.2 | 13.9 | 2 h |
| prior s4 main (3 fracs x 2 lr; 6 arms) | 6 | 180 | 6 x 50k | 2.2 | 5.0 | 2 h |
| prior s4 ctl | 6 | 120 | 4 x 50k | 1.5 | 3.3 | 1 h |
| **total (default)** | **84** | **1780** | | **30.2** | **~105** | |
| option --ext (flat extension up front: mod 0.05 / 0.1 / 0.15, s4 0.2 / 0.3; per cell an lr-1e-1 job with 3 wd and a wd-10 job with 4 lrs) | +10 | +175 | 3 or 4 x 4 B | +5.8 | +22.5 | 3-5 h |
| option --noprof (drop shapeprof) | 0 | -260 | | -2.9 | -8.6 | |

- --time = 1.5 x the worst case, rounded up to the hour.
- A TIMEOUT loses only the run in progress: finished JSONs are skipped on resubmission. --requeue covers preemption.
- **Calibrate first** (audit finding 5): submit `mod-torus-f0.05-prior-lr1e-2` (main) and `-ctl`, `mod-f0.05-none-lr1e-2`
  and `s4-R-f0.2-prior-lr1e-2` with `--only`; read ms_per_step / device from the first finished JSONs and the logs (any
  eigendecomposition error), then submit the rest. A crash in a shape arm kills only that seed process's later arms;
  the shape arms are first in both prior jobs, so the calibrator shows it at once.
- Conditional bracket extensions (rule 2, non-flat edge): 0.5 (mod) / 0.5 (s4) L40S-h per extra lr value per fraction
  (3 wd x 5 seeds).

## Seven checks (experiment-audit)
| check | status | evidence |
|---|---|---|
| i double path | pass | shape loss vs an independent numpy implementation (spectrum from the SVD of the centred embedding, norms by einsum, profile by per-row sorted products), 8 targets (line and sphere controls included) x 20 draws x kinds: max diff 5.6e-15 (preflight/double_path.py); the mod readout of the run's own variant equals shapeL_circle (fourier1 in a circle run, fourier1-5 in a torus run); n_fresh (mod --mod_twins) vs a second-path count of untrained commuted twins: equal; spec <= 1 - CKA (von Neumann) on 150 draws |
| ii intent asserts | pass | in-run shape self-check on the device (permutation invariance, target and permuted target ~0, init > 0); line target rank / spectrum; sphere rank / flatness / constant norm (1e-9); vertex-transitivity for prof; shape arms' kind / target validated before training (readouts are not targets); --mod_twins only for mod + question split; upto() / late_change() refuse incomplete curves; summarize_wall refuses mixed code_sha and missing test_fresh; generator asserts every rendered flag set, one job per (cell dir, arm), the pinned sha; jobs re-check the snapshot sha; the existing lr_now assert and split asserts (S4 0.2 seed 3 refused by them) untouched |
| iii known answer | pass | shape-only optimisation reaches ~0 with CKA to the assigned target at chance (no assignment injected); R_sign = same shape (0), Z24 = same spec + norm (0) but prof 0.063; sphere vs true shape = 1 - sqrt(r / r') exactly; synthetic wall cells give the pre-registered verdicts (14 cells, see Tests); default path bitwise equal to b74a78cb5472, which itself equals toy_grok on mod (test_toy_struct known answer) |
| iv anomaly repro | n/a | no results |
| v fresh adversarial reviewer | RUN (2026-10-19, independent auditor, before any submission) | 5 important + 4 minor findings, all addressed (next section). A re-review of the fix diff (toy_struct.py e1bae1b77663 -> d182ce548bcc, the generator, this file) is recommended in minimum mode, in the background after submission |
| vi trap list | pass (pre-run) | 1 same template / position train vs eval (table); 2 held-out disjoint, test_fresh for twins (required), none gets 4x steps and must not be rising; 3 the shape loss provably acts (A/B) and is NaN-safe; 4 5 seeds, paired, exact test; 5 readout = last token, decoded in every log; 6 one shape-loss function for training, logging, the check table, the self-check; AUC_B defined once (upto); 7 every held-out question scored, n_fresh logged; 8 default path regression-identical, summaries of 7 real result dirs byte-identical; job lines rendered and smoke-run from the job files |
| vii decoded inputs side by side | pass | every seed log prints 3 train / 3 held-out questions with ids, names, scored position (smoke output in Tests) |

## Independent audit 2026-10-19 (check v) and fixes (all before any run)
| # | finding (auditor) | real? | fix |
|---|---|---|---|
| 1 | shape-specificity never tested: "shape meets the rule, wrongshape does not" read as BREAKS even with no shape vs wrongshape gap (4/5 vs 3/5) | yes (verified in code: the shape > wrongshape test was computed but unused) | rule 5: BREAKS needs the role to BEAT each control paired (AUC on test and test_fresh); otherwise "SPECIFICITY NOT SHOWN"; synthetic cell 0.13 is the auditor's case and now reads so |
| 2 | spec + norm fixes only "a flat constant-norm r-sphere"; `line` differs in effective rank AND norms, so it is a confounded control | yes (the builder's own known answer showed it) | (a) new matched control `sphere` (rank r', flat, constant norm; mod r + 1, s4 12), in every cell; (b) `shapeprof` arms in the default design; (c) verdict wording: shape = "rank-r isotropic constant-norm prior"; "geometry beyond rank" only via shapeprof > shape |
| 3 | wall-cell rule lenient: only "<= 1/5 grok", so a none at 0.85 and rising counted as a wall | yes | rule 1: every none config also at median final <= 0.20 and not rising (last-25 % gain < 0.02; best config's test-CE drop < 0.01 nats, critique D1); else "not a wall cell (reason)" |
| 4 | "flat" exempted from bracketing: the F84 trap | yes | rule 2: flat is an OPEN bracket until the one-step extension (lr 1e-1, wd 10) has run; generator `--ext` queues it up front for the likely wall cells (open decision 3) |
| 5 | GPU eigvalsh untested (cuSOLVER float32, ~111 near-zero eigenvalues; a crash repeats on requeue), per-step syncs untimed | plausible (no GPU on this host; not reproducible here) | the eigendecomposition runs on the CPU in float64 (LAPACK) whatever the device; calibrator jobs first (`--only`); timing still unmeasured |
| m1 | missing test_fresh silently counted as a pass | yes | a cell with any run lacking test_fresh gets no verdict; passes() requires both tests |
| m2 | mod none logs shapeL only for the freq-1 targets | yes | `--shape_log` on mod also logs shapeL_fourier1 and shapeL_fourier1-5; summarize_wall reports none's distance to the variant's shape |
| m3 | summarize_wall globbed `**/run_*.json` (smoke + main would mix) | yes | reads `<root>/*/run_*.json` only and refuses more than one code_sha |
| m4 | power at 5 vs 5 | noted, no change | |

## Caveats (known before any run)
1. **What the shape loss encodes.** spec + norm = "a flat constant-norm sphere of the true rank" for the torus and S4,
   "points on a circle" for mod's circle. A `shape` pass is therefore worded as an isotropic rank-r prior. shapeprof pins
   every item's similarity profile; it is non-convex (from a random init it stalls at ~0.02).
   - S4: the random init is already 0.167 from the R shape; none drifts to 0.079 in a 300-step smoke.
   - Torus vs sphere(11): only 0.047 apart under spec + norm. A torus "NOT SHAPE-SPECIFIC (sphere)" is a likely and
     informative outcome: then the effect is a generic isotropic prior, not the five circles.
2. **The controls.** `line` differs from the true shape in effective rank and norms (participation ratio 1.5-2.2 vs
   2-10); `sphere` matches isotropy and norms at rank r' and can hold the true shape plus one dimension. Beating both
   is required for BREAKS.
3. **Dose.** Every shape loss is in [0, 2] (spec + norm) or [0, 3] (shapeprof), the assign loss in [0, 1]; all lam 1.
   Not dose-matched. The prior acts on the item embeddings only (W_E rows), like F84.
4. **Selection and multiplicity.**
   - Best lr of 2 for each prior role and best of 12+ none configs, both chosen on the held-out curve. There is no
     separate validation split: the tables are closed. It favours none.
   - Two tested roles (shape, shapeprof) x 13 primary cell readings at alpha 0.025 per co-primary. A single
     "BREAKS THE WALL" is not a robust claim by itself; the write-up gives the verdict pattern across fractions.
5. n = 5 seeds. "not shown at n=5" is never "no effect". The data multiplier is bracketed by a coarse grid.
6. mod 0.05: seeds 0 / 1 miss one answer class in training (held-out ceiling ~0.99). S4 0.2 uses seeds 0, 1, 2, 4, 5.
7. Toy only: one layer, d 128, nonce items, closed table, full batch. Nothing here transfers to LLMs without the 53M step.

## Open decisions (the user's; defaults as generated)
1. `--noprof`: shapeprof is now in the default design (auditor's fix 2b, +2.9 L40S-h). Drop it only if cost matters.
2. PRIMARY schedule = always-on, chosen here because an inductive bias is a persistent constraint. F84 had the window
   as primary; flip before submission if wanted.
3. `--ext`: queue the flat-cell extension up front for mod 0.05 / 0.1 / 0.15 and s4 0.2 / 0.3 (+10 jobs, +5.8 planning /
   +22.5 worst-case L40S-h; recommended by the auditor, saves one ~2-day queue round), or only when rule 2 fires.
4. S4 0.2 with seeds 0, 1, 2, 4, 5 (rule above), or drop S4 0.2.
5. Prior arms at B only (default, conservative: none gets 4x the steps), or at 4 B (+~42 L40S-h planning, 3x the prior cost) so that a
   late shape grok is not censored.
6. Commit: tools/toy_struct.py (d182ce548bcc) is uncommitted in a PUBLIC repo; the snapshot carries it. Commit or not
   is the user's call. tools/pretrain_range.py is also modified in the working tree (another piece): do not stage it with
   this one.
7. Re-review of the fix diff (check v, minimum mode) after submission.
