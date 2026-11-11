# newmani Stage 1: new-manifold toy screen, scout (2026-10-20; revised 2026-10-21 after the build audit; built, NOT submitted)

Design: `~/Research_Projects/LLM_addition_review/notes/new_manifolds_screen_design.md` (2026-10-19; §3 the screen, §4 the
decision rules). Rescue standard: `llmadd-burst/report/audits/rescue_standard.md` incl. Amendment 1 (tiers S / L).
Builds on runs/toyshape (wall 2: shape-only prior, `summarize_wall` rules 1-2), runs/toyf84fu (Latin split, latin.py),
F84 + addendum (S4 "unlearnable" was an lr artefact), F88 A7.

## Revision 2026-10-21 (after the fresh build audit, before any submission or result)
Audit: `~/Research_Projects/LLM_addition_review/notes/batch1_build/audit_newmani_build.md` (trainer trustworthy; the reading
layer could mis-read). Every finding, its fix and its test (file:line = the current files):
| audit item | fix | where | test |
|---|---|---|---|
| B1: the best-config test-CE-drop component of the wall rule measures the CE noise of confident memorisers (s4 0.2 / 0.3 in toyshape were vetoed by it alone) | **Amendment A1** below: the CE component counts only while the best config's late held-out CE <= log(n_out) + 1 nat, else n/a; every wall candidate prints whether the CE component alone vetoes it (and whether the ungated rule would have) | readout.py `late_ce`, `wall_reasons` (CE_GATE_NATS), read_cell's "CE component" line | preflight/b1_toyshape.py: the gate-off rule reproduces all 13 published toyshape verdicts; gate on turns s4 0.2 / 0.3 into walls, mod unchanged (table in A1); readout self-test B / F |
| I1: missing prior seeds / lrs / arms gave a clean CANDIDATE or a false floor; the positive control accepted < 3 seeds | a tier verdict needs its role AND its controls at all 3 lrs x the cell's 3 seeds, else "INCOMPLETE (prior arms: ...)"; an incomplete none grid gives "PROVISIONAL (...)", never CANDIDATE / floor; the task-level selection needs every level complete and the positive control complete; positive controls need every seed | readout.py read_cell (seeds_ok, tier block), select | readout self-test E (the audit's 4 scenarios) and the audit's own chk_readout.py rerun on the fixed reader (preflight/audit_chk_readout_after_fix.log) |
| I2: Stage 2's "seeds 10-14" fail the split / closure asserts at low m | **pre-registered Stage 2 seed rule** (section below): per m cell, the first 5 seeds >= 100 passing the same checks at the same frac, never a Stage-1 seed; listed now | preflight/seed_scan.py, stage2_seeds.json | seed_scan.log |
| I3: box5 had no positive control | none at f 0.4 and 0.6, lr 1e-3 / wd 1 (the design's config) and lr 1e-2 / wd 0.1 (dim's best), 3 seeds, 200k; OK iff one (frac, config) groks >= 2/3 with every seed present | make_jobs_newmani.py TASKS["box5"]["pos"]; readout.py POS_CFG | readout self-test G; jobs box5-pos-j0 / j1 rendered |
| I4: m is not information-matched across tasks | wording (cells section: training classes per item per cell; cross-task statements descriptive only) | this file | - |
| I5: --ext re-packed default jobs and could rewrite submitted job files | append-only generator: extension runs in their own '-ext-' jobs (--ext or --ext_cells), defaults never re-packed; any change to (or drop of) a submitted job's file is REFUSED before writing; --submit / --redo skip a job already in squeue | make_jobs_newmani.py (EXT_GROUPS, the append-only guard, queued_names) | preflight/test_generator.py: the audit's gen_copy.py scenario reproduced on the old generator (45 of 90 runs moved), the new one keeps every default job byte-identical and refuses changes |
| item 6 (deadline exit 0 hides missing prior arms) | `--status` reads the Slurm state (sacct) of every submitted job and lists jobs that ENDED with runs missing separately, with the --redo command; requeue behaviour unchanged | make_jobs_newmani.py --status | test_generator.py 1h (a real COMPLETED job id) |
| M1 box whole-negation twin (-a, op, -b) -> -c not excluded | excluded from test_strict (as the same-answer negation twin) | toy_struct.py nm_masks (box branch) | test_newmani §4 (second path, cell by cell) |
| M2 group division rewrites (c, b^-1) -> a, (a^-1, c) -> b not in the twin orbits | report-only subset test_norewrite (neither rewrite trained); main metric unchanged (one-step associativity = legitimate generalisation) | toy_struct.py nm_masks (grp branch) | test_newmani §4 |
| M3 2I's central -1 not excluded from strict | test_strict excludes every central input / answer (the centre contains e; only 2I has a second central element) | toy_struct.py nm_masks | test_newmani §4 |
| M4 labels | sphere renamed "rank r' != r" everywhere; GROUP_IRREPS comment = complex irreducible dimensions (d^2 = the real isotypic block) | this file; toy_struct.py GROUP_IRREPS comment | - |
| M6 flat prior arms labelled "OPEN at bottom" | lr bracket "flat (no optimum ...)" when every lr ties or sits at the floor | readout.py prior block | readout self-test A |
| M10 shuf far only tested, not asserted | asserted in every run (max CKA to the faithful refs < 0.15; s4 = the known-answer path is exempt, ~0.34 as in F84); part of the seed rule | toy_struct.py check_targets | seed_scan (Stage-1 seeds unchanged; max 0.035 / 0.048 / 0.053 / 0.150 / 0.136) |
| M5, M7, M8, M9 | known limitations (section "Known limitations") | | |

Trainer sha **d0aed90927fd -> ea797e5321d3** (new tasks' read-out masks and the shuf assert only; the default path still
bitwise = d182ce548bcc, regression rerun). The d0aed90927fd snapshot and job files are kept in code.d0aed90927fd/ and
preflight/superseded_d0aed90927fd/ (never submitted). Stage-1 cells (fractions, seeds, split_sha, n_train) are unchanged.

## Binding decisions (user, 2026-10-20) and how each is met
| decision | how |
|---|---|
| goal = genuine rescue cases: tuned none fails under every optimiser setting, the prior succeeds | Stage 1 finds the (task, m) cells where that pattern appears; Stage 2 (fresh seeds 10-14, n = 5) confirms by the design's G1-G4. Stage 1 never states a rescue |
| per candidate an assignment prior (Tier L) and a shape-only prior (Tier S), with the F84 / wall-2 controls | every (task, m) cell runs assign `<true>@always` (L), `specrand@always` + `shuf@always` (L controls), `shape:<true>@always` (S, spectrum + norms, permutation-invariant), `shape:sphere@always` (rank-matched constant-norm sphere) + `shape:line@always` (S controls) |
| criterion (a): none full lr x wd sweep with a 4x budget cell | none at lr {1e-3, 3e-3, 1e-2, 3e-2} x wd {0.1, 1, 3} (toyshape's grid), every config at 4 B = 200k steps; rule-2 flat extension available up front (`--ext`) or conditional |
| criterion (b): the priors get their own lr sweep | every prior arm at lr {1e-3, 3e-3, 1e-2} (wd 1, lam 1), best lr per arm by median AUC_B, its bracket state reported |
| criterion (f): one device type for every arm | every run on an L40S (l40s_public, gpu48); each job checks `nvidia-smi` and exits 4 on another GPU; the JSON records device, gpu/<job>.txt the GPU name per (re)start |
| Latin-only and associativity closure of held-out cells per split, as runs/toyf84fu/latin.py | every run JSON carries `closure` (groups: associativity fixpoint, toy_struct.s4_closure; box: linear row space, dim's definition), `closure_latin` (groups: assoc + Latin), `latin_share` / `latin_level` / `chance_nonimp` (Latin-only naked + hidden singles = latin.py's algorithm, vectorised, equal to latin.py on 12 S4 splits); readout.py prints them per seed |
| read-out splits held-out accuracy into implied vs non-implied | every eval logs `test_implied` / `test_nonimplied` (Latin-only implied cells), plus `test_strict`; readout.py reports both |
| top 5 at the design's training-cell budgets; top 2 first if short | box5^3, 2I, D57, A5, S5 at m ~ 6 / 12 / 24 cells per item; wave 1 = box5 + 2I, wave 2 = D57 / A5 / S5 (`--wave`) |
| code: opt-in only, default path bitwise vs d182ce548bcc; fresh worktree; snapshot + COMMIT; no commit / push / sbatch | worktree `/scratch/yh6210/llmadd-burst-newmani` (branch newmani-screen from origin/main 1f6aa80), file uncommitted; preflight/regress_default.py; generator pins the sha, jobs re-check it |
| every new flag: A/B + in-run assert of the consumed value | preflight/test_newmani.py section 7; JSON keys `group` / `box` / `n_out` / `true_target` / `ckpt` |

## Intent spec
**Question.** On five new closed operation tables, is there a training-set size m (cells per item) at which the tuned
one-layer toy without a prior does not learn the held-out table, while an embedding prior does? Ask it separately for a
prior that gives the structure (Tier L: CKA to a faithful matrix representation / the exponent coordinates) and for one
that gives only the shape (Tier S: centred-Gram spectrum + norm profile, no assignment).

**What the target encodes (rescue_standard (c), Amendment 1).** Tier L targets determine the whole table once the operation
is learned (a faithful matrix rep: T(a then b) = T(b) T(a); expo: e_a +- e_b), so a Tier-L success reads only "given the
structure, the model learns to use it". They place no question's answer directly, so they are not anchor targets. Tier S
targets give rank, isotropy and norms only. For a flat constant-norm target (every group target here) spec + norm =
"a flat constant-norm r-sphere": a Tier-S pass is worded as an isotropic rank-r prior (toyshape caveat 1).

### Tasks (tools/toy_struct.py, opt-in `--task box` / `--task grp`)
| task | items (tokens) | question -> answer | true target (rank r) | other targets built (asserted, not trained in Stage 1) | sphere r' | split |
|---|---|---|---|---|---|---|
| box5 = `--task box --box=-2:2,-2:2,-2:2` | 125 exponent vectors in {-2..2}^3 (0..124; 125 mul, 126 div, 127 '=') | (a, op, b, =) -> e_a + e_b / e_a - e_b, in the box only (6,859 per op, 13,718) | expo (3, flat) | quot (2), onehot (12) | 4 | relation classes {x*y, y*x, z/y, z/x} (3,443 classes) |
| 2i = `--task grp --group 2i` | 120 unit icosians (600-cell; 0..119, 120 '=') | (a, b, =) -> a then b = q_b q_a (14,400) | quat4 (4, flat) | quat4_galois (4), A5rot9 (9, kernel +-1) | 5 | twin orbits (1,774) |
| d57 | 114 = R_k F^f (57 rotations, 57 reflections) | a then b (12,996) | rho1 (4, flat) | rho2, rho3 (kernel 3), rho19 (kernel 19), abel (Z57 x Z2 law), torusD (rho1..5, 20) | 6 | twin orbits (5,743) |
| a5 | 60 icosahedral rotations | a then b (3,600) | rot3 (9, flat) | rot3_galois (9), std4 (16), sym5 (25) | 10 | twin orbits (1,774) |
| s5 | 120 permutation matrices | a then b (14,400) | std4 = P - J/5 (16, flat) | std4_sign (16), sign (1) | 18 | twin orbits (7,099) |

- **Group tables** are built from the faithful rep (`group_table`, products keyed to 1e-6). They are asserted equal to an
  independent combinatorial table (quaternion product / D57 formula / permutation composition / cube_group), and asserted
  for: associativity (all n^3), a unique identity, inverses, the Latin property, non-commutativity, |G| = sum d^2 over
  the irreps, the element-order histogram, and the number of conjugacy classes.
- **Targets.** Every matrix target is asserted a homomorphism with the stated kernel size. Inequivalent irreps are
  asserted CKA < 1e-9 (Schur), and irreducible targets asserted flat.
- **Sphere rank r'** = the smallest r' > r that is not a subset sum of the non-trivial isotypic dimensions d^2 (computed,
  asserted = the design's 5 / 6 / 10 / 18; box r + 1 = 4).
- **Orbit split (grp).** The twin moves are:
  - commute (b, a) when ab = ba;
  - the inverse twin (b^-1, a^-1);
  - for each central z, (a z, z^-1 b) and (a z, b).

  Whole orbits go to train or held-out. The box uses dim's relation classes; its negation twin (a, other op, -b) is
  reported, and excluded from the strict set.
- **Known answers.**
  - `--task grp --group s4 --split question` reproduces `--task s4`.
  - `--task box --box=-1:1,-1:1,-1:1` reproduces `--task dim`.
  - In both, every key of the old run JSON is equal, curves exact; test_newmani §6.

### Cells (preflight/seed_scan.json; rule: the first 3 seeds whose split passes coverage + closure 1.0 [+ box nullity 3]; the fraction moves up the design's fallback when < 3 of seeds 0-39 pass)
| task | m level | frac | seeds | n_train | m (cells / item) | assoc closure | Latin-only implied share | chance (1/n, majority) | elim. heuristic on strict |
|---|---|---|---|---|---|---|---|---|---|
| box5 | 6 | 0.067 (0.055 / 0.06 refused: an item never an input, or closure < 1) | 4, 10, 25 | 910-920 | 7.3-7.4 | 1.0 (nullity 3) | 0 | 0.008 / 0.011-0.014 | 0.0085 |
| box5 | 12 | 0.11 | 0, 1, 2 | 1506-1510 | 12.0-12.1 | 1.0 | 0 | 0.008 / 0.010-0.014 | 0.0089 |
| box5 | 24 | 0.22 | 0, 1, 2 | 3012-3020 | 24.1-24.2 | 1.0 | 0 | 0.008 / 0.014-0.019 | 0.010 |
| 2i | 6 | 0.067 (0.05 / 0.06 refused) | 6, 14, 22 | 940-956 | 7.8-8.0 | 1.0 (assoc+latin 1.0) | 0 | 0.0083 / 0.0077 | 0.0095 |
| 2i | 12 | 0.1 | 0, 3, 4 | 1412-1460 | 11.8-12.2 | 1.0 | 0 | 0.0083 / 0.0074 | 0.0103 |
| 2i | 24 | 0.2 | 0, 1, 2 | 2840-2884 | 23.7-24.0 | 1.0 | 0 | 0.0083 / 0.006-0.007 | 0.013 |
| 2i | pos | 0.4 | 0, 1, 2 | 5724-5780 | 47.7-48.2 | 1.0 | 0 | | 0.023 |
| d57 | 6 | 0.053 | 7, 14, 31 | 665-697 | 5.8-6.1 | 1.0 | 0 | 0.0088 / 0.008 | 0.0098 |
| d57 | 12 | 0.105 | 1, 3, 4 | 1338-1355 | 11.7-11.9 | 1.0 | 0 | 0.0088 / 0.008 | 0.0109 |
| d57 | 24 | 0.21 | 0, 1, 2 | 2715-2730 | 23.8-23.9 | 1.0 | 0 | | 0.014 |
| d57 | pos | 0.4 | 0, 1, 2 | 5162-5220 | 45.3-45.8 | 1.0 | 0 | | 0.024 |
| a5 | 6 | 0.1 | 1, 3, 5 | 355-367 | 5.9-6.1 | 1.0 | 0 | 0.017 / 0.014-0.015 | 0.021 |
| a5 | 12 | 0.2 | 0, 1, 2 | 717-721 | 12.0 | 1.0 | 0 | 0.017 / 0.014 | 0.026 |
| a5 | 24 (= f 0.4, also A5's positive control) | 0.4 | 0, 1, 2 | 1430-1446 | 23.8-24.1 | 1.0 | 0 | 0.017 / 0.012-0.014 | 0.046 |
| s5 | 6 | 0.05 | 3, 7, 9 | 714-718 | 6.0 | 1.0 | 0 | 0.0083 / 0.008 | 0.0092 |
| s5 | 12 | 0.1 | 0, 2, 3 | 1433-1445 | 11.9-12.0 | 1.0 | 0 | 0.0083 / 0.0075 | 0.0103 |
| s5 | 24 | 0.2 | 0, 1, 2 | 2875-2884 | 24.0 | 1.0 | 0 | | 0.013 |
| s5 | pos | 0.4 | 0, 1, 2 | 5747-5768 | 47.9-48.1 | 1.0 | 0 | | 0.023 |
| box5 | pos (positive control, audit I3) | 0.4 | 0, 1, 2 | 5487 | 43.9 | 1.0 | 0 | | |
| box5 | pos6 (positive control) | 0.6 | 0, 1, 2 | 8230 | 65.8-65.9 | 1.0 | 0 | | |

- **Latin-only implied share is 0 at every Stage 1 split.** A pure table-filler derives no held-out cell, so non-implied =
  all held-out and the implied / non-implied split is degenerate here. It is still logged; it matters only at denser
  splits (S4 0.6: 0.026-0.069).
- **Associativity closure is 1.0 by construction** (the seed rule): every held-out answer is determined by the training
  cells plus the group law, i.e. the table is identifiable.
- The m levels at 6 for box5 and 2I land at m 7.3-8.0 (the design's own fallback, "覆盖不满足就改 0.067"): **"m6" of box5
  and 2i is m 7.3-8.0**, not 6.
- **m is not information-matched across tasks (audit I4).** m counts training CELLS per item; the split draws whole twin
  classes, and their size differs (2I orbits ~8 cells; A5 / S5 ~2; D57 ~2.3; box relation classes ~4). Training classes
  per item (from seed_scan.json):

  | level | box5 | 2i | d57 | a5 | s5 |
  |---|---|---|---|---|---|
  | m6 | 1.84 | 0.98 | 2.67 | 2.95 | 2.95 |
  | m12 | 3.02 | 1.48 | 5.29 | 5.90 | 5.91 |
  | m24 | 6.06 | 2.95 | 10.58 | 11.82 | 11.82 |

  So at the same m label 2I has ~4x fewer independent relations than A5 / S5 (2I / {+-1} = A5: both have 1,774 orbits).
  Every verdict is per task; **any cross-task statement (e.g. design 3.4 "2I rescues, S5 does not -> compactness") is
  descriptive only**, confounded by this, and never a finding.

### Arms, tiers, budgets (per (task, m), 3 seeds; B = 50,000)
| arm | role | tier | lr | wd | steps |
|---|---|---|---|---|---|
| none | baseline | - | 1e-3, 3e-3, 1e-2, 3e-2 | 0.1, 1, 3 | 4 B = 200k |
| `<true>@always` | assign | **L** | 1e-3, 3e-3, 1e-2 | 1 | B |
| `specrand@always` | same centred spectrum, random singular vectors (rng 1000 + seed) | L control (d) | same | 1 | B |
| `shuf@always` | the true rows under a far permutation (CKA < 0.15 to every faithful ref) | L control (d) | same | 1 | B |
| `shape:<true>@always` | spec + norm of the true target, no assignment | **S** | same | 1 | B |
| `shape:sphere@always` | flat constant-norm sphere of rank r' != r (r' = the smallest rank above r that is not an isotypic sum: 5 / 6 / 10 / 18 / 4 vs r 4 / 4 / 9 / 16 / 3) | S control (same isotropy and norms, another rank; NOT rank-matched) | same | 1 | B |
| `shape:line@always` | open chain, rank r, spectrum ~ 1/k^2 | S control | same | 1 | B |
| positive control | none at f 0.4 (2i / d57 / s5), lr 1e-3 wd 1; box5 at f 0.4 and 0.6, lr 1e-3 wd 1 and lr 1e-2 wd 0.1; a5 = its m24 cell (f 0.4), lr 1e-3 wd 1 | known answer | | | 200k |
| `--ext` / `--ext_cells` (optional, append-only) | none lr 1e-1 x wd {0.1, 1, 3}, wd 10 x the 4 lrs, in separate '-ext-' jobs | rule-2 extension | | | 200k |

All arms lam 1, `@always` (the design's primary schedule; the [0, 2000) window is Stage 2's secondary reading). Model /
optimiser = toy_grok (one layer, d 128, 4 heads, MLP 512, no LN, full batch, AdamW (0.9, 0.98), 10-step warmup to
--lr; lr_now logged + asserted every eval).

## Metrics (single definitions; implementations)
- **held-out test**: accuracy over every held-out question at every eval (every 100 steps + the last step);
  toy_struct.run_arm. Answer read at the last token '='; decoded inputs printed per seed (vii).
- **test_implied / test_nonimplied**: the same over the held-out cells a pure Latin-square filler derives / does not
  (toy_struct.nm_latin; = runs/toyf84fu/latin.py on S4, test_newmani §5).
  - grp: one full Latin square, naked + hidden singles to a fixpoint.
  - box: one partial table per operation, naked singles only, because a row's value set is not known without the law.
- **test_strict**: non-implied and no shortcut cell (toy_struct.nm_masks).
  - grp: no central input (e; and 2I's -1), answer not central (audit M3).
  - box: no copy cell (x*0, 0*x, x/0), answer != 0, neither negation twin trained: (a, other op, -b) (same answer) nor the
    whole negation (-a, op, -b) -> -c (audit M1).
- **test_norewrite** (grp, report only; audit M2): held-out cells (a, b) -> c with neither division rewrite (c, b^-1) -> a
  nor (a^-1, c) -> b trained (their share with a trained rewrite: ~0.11-0.35 by m, up to 0.6 at f 0.4). The box's relation
  classes already contain its rewrites.
- Also logged: d57 test_RR / RF / FR / FF; box test_multi (b moves >= 2 coordinates) and test_edge (answer on the box
  surface).
- **Wrong-answer label shares** (descriptive): 2i neg_share (-c, chance 1/119); d57 coset3 / coset19 / refl shares; s5
  parity_share; box drop_share.
- **AUC_B, t_grok_B** = toy_struct.upto (the evals 0..B - 100 of any run); **late gain / CE drop** =
  toy_struct.late_change over the last 25 % of 4 B; AUC 4B = toy_struct.auc.
- **chance** = max(1/n, median majority-of-train); **floor line** = max(chance, median one-step elimination heuristic on
  the strict set) + 0.05 (design 3.1).
- **closure / closure_latin / nullity**: toy_struct.closure. Groups use s4_closure (the associativity fixpoint, both
  directions) and the version with Latin cancellation. Box uses the linear row-space test, two methods asserted equal.

### Train vs eval
| dimension | train | eval | same? |
|---|---|---|---|
| template / position | (a, b, =) or (a, op, b, =), scored at the last token | same | yes |
| questions | whole twin orbits / relation classes drawn by frac | the other orbits / classes | no, by design; no twin of a held-out cell is trained (asserted cell by cell) |
| items | every item an input and an answer (asserted) | same items | yes |
| model state / precision | full batch, fp32, no dropout; shape loss's eigendecomposition in float64 on the CPU | no_grad, every held-out question every 100 steps | yes |
| prior | item rows of W_E only, always on | off at eval | n/a |

## PRE-REGISTERED reading (2026-10-20, before any run; implemented in readout.py, tested on synthetic cells)
### From the design (§3.2 Stage 1, §4 G1-G4; quoted, binding for Stage 2)
- **Rule-1 wall definition** (toyshape rule 1 + design G1): a (task, m) is a wall cell iff every none config of the sweep
  meets all four:
  - groks on <= floor(0.2 n) seeds within 4 B;
  - median final held-out test <= 0.20;
  - not still rising: median last-25 % gain < 0.02, and the best config's median test-CE drop < 0.01 nats (**the CE part
    is gated by Amendment A1**: it counts only while that config's late held-out CE <= log(n_out) + 1 nat);
  - median final test_strict <= the floor line.

  Bracket (rule 2):
  - Best config = most groks, then the highest median AUC over 4 B. Closed iff it is not on a grid edge.
  - "Flat" (0 groks and AUC <= chance + 0.05 everywhere) stays OPEN until the extension (lr 1e-1 x 3 wd, wd 10 x 4 lr)
    has run.
- **G1 墙**: the selected cell is a wall cell and its bracket is closed (extension included).
- **G2 挽救**: the true target @always at its best of 3 lrs (by AUC_B) must meet both:
  - >= 4/5 seeds grok within B (held-out >= 0.95);
  - the paired AUC_B test vs the best none (compare_values: >= 4/5 wins, exact one-sided MWU p < 0.025, median
    difference >= 0.05, above the floor line) passes on all held-out AND on test_strict.
- **G3 特异**: the true target beats specrand AND shuf by the G2 test on both read-out sets; D57 also beats abel on the
  blocks with a reflection.
- **G4 能迁移**: median t_grok_B <= 20k (0.4 B), and the selected cell has <= 1,800 training cells.
- **Verdict.** G1-G4 all pass -> next rainbow round.
- **"加速档"** = G1 fails, G2-G4 pass, and t_grok is >= 3x faster than the best none at its best lr. This is the user's
  call.
- **Wording.**
  - "Not learned" only within "this model family, these optimisers, the 4 B budget".
  - "Geometry-specific" only from G3.
  - "The target encodes the table" is written every time (Tier L).
  - Each task is judged separately, with a multiplicity note (5 tasks x 2 read-out sets).
- **Tier S in Stage 2 (Amendment 1).** The same G1-G4 with shape:<true> in place of the assign arm, and with sphere and
  line in place of specrand / shuf in G3, read by toyshape rule 5:
  - a control meeting the rule -> "NOT SHAPE-SPECIFIC";
  - otherwise beats both controls -> "BREAKS THE WALL" (as a rank-r isotropic constant-norm prior);
  - otherwise "SPECIFICITY NOT SHOWN".

  A result is reported with the highest tier it meets.

### Stage 1 (this run): the scout reading (readout.py)
1. **Split facts per seed**: n_train, m, closure, closure_latin / nullity, the Latin-only implied share, n implied /
   non-implied / strict. Also chance and floor line.
2. **None sweep** per config: groks in 4 B, median and max final test, final strict, final non-implied, AUC 4B, AUC_B,
   late gain / CE drop.
   - **Scout wall** = rule-1 wall at n = 3: every config 0/3 groks; EVERY seed's final <= 0.20 (design 3.2 "所有配置、
     所有 seed 终值 <= 0.20，而且 0 grok", stricter than the median); not rising (CE part gated, Amendment A1); strict
     median <= floor line.
   - Printed per cell: the CE component's state (late CE vs gate, on / n/a), whether it ALONE vetoes the cell, and whether
     the ungated pre-A1 rule would have vetoed it on CE alone.
   - A missing config or seed (vs the scan's 3 seeds) -> PROVISIONAL (never CANDIDATE / floor).
   - Bracket state reported (best config, edges, flat -> OPEN).
3. **Prior arms**: each arm's lr = the highest median AUC_B of {1e-3, 3e-3, 1e-2}. Its lr bracket is "closed" iff
   3e-3 wins, "OPEN at bottom / top" at an edge, "flat (no optimum)" when every lr ties or sits at the floor (audit M6),
   "INCOMPLETE" when an lr or a seed is missing.
4. **Tier verdict per cell** (L: assign with controls specrand / shuf; S: shape with controls sphere / line):
   - none grid incomplete -> "PROVISIONAL (...)";
   - not a scout wall -> "not a wall cell: acceleration reading only";
   - scout wall but the role or one of its controls lacks an lr or a seed (all 3 lrs x the cell's 3 seeds required), or is
     absent -> "INCOMPLETE (prior arms: ...)" (audit I1);
   - scout wall, all complete, the role groks within B on >= 2/3 seeds -> **CANDIDATE**. If a control of that tier also
     groks on >= 2/3, it is flagged "specificity doubtful";
   - scout wall, all complete, the role fails -> "floor for tier X".
5. **Selection per task and tier** (design 3.2):
   - The CANDIDATE m closest to 12 (ties -> the smaller m). The design's "smallest m" picks among candidates, and
     "closest to 12" breaks a multiple; the two are read in that order.
   - No candidate and no wall cell at any m -> "no wall at any m: acceleration reading only".
   - Wall cells but no candidate -> "floor".
   - The selection is made only when all three levels have a complete verdict (CANDIDATE / floor / not a wall) and the
     positive control is complete; otherwise "INCOMPLETE" (candidates so far listed): nothing is read for that task.
   - Selected cells go to Stage 2 on the pre-registered fresh seeds (section "Stage 2 seeds") under the G rules. The
     user picks the Stage 2 cells from this table.
6. **Positive control**: none at f 0.4 (lr 1e-3 wd 1; box5 also f 0.6 and lr 1e-2 wd 0.1) must grok on >= 2/3 seeds in
   at least one (frac, config) cell, every seed present. A5 uses its m 24 cell. Failed -> "NOT READ: check the pipeline"
   for that task (design 3.1 "学不会就先查管线，不读结果"); incomplete -> INCOMPLETE.
7. **Paired AUC_B tests vs the best none** are printed as descriptive only: at n = 3 the exact one-sided p is >= 0.05, so
   none can pass alpha 0.025.
8. **Nothing is added after results without the label "post hoc"**: no arm, lr, seed, fraction or budget. The
   exceptions are the pre-registered rule-2 extensions (`--ext`, or the conditional extension of a flat or edge cell).

### Amendment A1 (2026-10-21; before any newmani result exists; basis: build audit B1 and the F89 defect record "规则 1(c) 在 S4 上被 CE 锯齿触发 ... 下次复用前修")
- **Rule.** In rule 1's "not still rising", the best none config's last-25 % test-CE drop (>= 0.01 nats) counts only while
  that config's late held-out CE (median over seeds of the mean of the last 5 evals before 4 B) is <= log(n_out) + 1
  nat. Above that the component is **n/a**. The accuracy part (gain >= 0.02 in any config) is unchanged.
- **Why.** A memoriser that is confidently wrong on held-out sits at 13-55 nats. Its CE drifts by nats with no change in
  accuracy, so 0.01 nats there is noise:
  - audit: 43 % of 527 non-learning toyshape none runs "fire";
  - F89: S4 0.2 / 0.3 were vetoed by it alone, with gain -0.014 / -0.008.

  Near log(n_out) the drop still means probability mass moving onto right answers, so the component keeps its job there.
- **Known answer** (preflight/b1_toyshape.py, b1_toyshape.log): toyshape's rule 1 recomputed from the raw runs with
  readout.py's own `wall_reasons` (toyshape's median cap, no strict term).
  - Gate OFF reproduces all 13 published verdicts.
  - Gate ON changes exactly s4 0.2 and s4 0.3 to walls; both brackets are "flat over the extended grid".
  - Every mod verdict is unchanged, and s4 0.45 is still vetoed by the cap.

  | toyshape cell | none configs x seeds | best config | best late CE / gate (log n + 1) | best CE drop | published (pre-A1) | recomputed gate OFF | gate ON (A1) | CE alone vetoed (pre-A1) | bracket (published) |
  |---|---|---|---|---|---|---|---|---|---|
  | mod-circle-f0.05 | 19 x 5 | lr 0.03 wd 1 | 6.24 / 5.73 (n/a) | -0.001 | True | True | **True** | False | flat over the extended grid (lr to 1e-1, wd to 10) |
  | mod-circle-f0.1 | 19 x 5 | lr 0.01 wd 10 | 8.45 / 5.73 (n/a) | -0.090 | True | True | **True** | False | flat over the extended grid (lr to 1e-1, wd to 10) |
  | mod-circle-f0.15 | 19 x 5 | lr 0.01 wd 1 | 0.48 / 5.73 (on) | -0.459 | False | False | **False** | False | closed |
  | mod-circle-f0.2 | 12 x 5 | lr 0.01 wd 1 | 0.03 / 5.73 (on) | -0.004 | False | False | **False** | False | closed |
  | mod-circle-f0.3 | 12 x 5 | lr 0.003 wd 3 | 0.02 / 5.73 (on) | +0.000 | False | False | **False** | False | OPEN at wd top |
  | mod-torus-f0.05 | 19 x 5 | lr 0.03 wd 1 | 6.24 / 5.73 (n/a) | -0.001 | True | True | **True** | False | flat over the extended grid (lr to 1e-1, wd to 10) |
  | mod-torus-f0.1 | 19 x 5 | lr 0.01 wd 10 | 8.45 / 5.73 (n/a) | -0.090 | True | True | **True** | False | flat over the extended grid (lr to 1e-1, wd to 10) |
  | mod-torus-f0.15 | 19 x 5 | lr 0.01 wd 1 | 0.48 / 5.73 (on) | -0.459 | False | False | **False** | False | closed |
  | mod-torus-f0.2 | 12 x 5 | lr 0.01 wd 1 | 0.03 / 5.73 (on) | -0.004 | False | False | **False** | False | closed |
  | mod-torus-f0.3 | 12 x 5 | lr 0.003 wd 3 | 0.02 / 5.73 (on) | +0.000 | False | False | **False** | False | OPEN at wd top |
  | s4-R-f0.2 | 19 x 5 | lr 0.01 wd 3 | 20.76 / 4.18 (n/a) | +2.439 | False | False | **True** | True | flat over the extended grid (lr to 1e-1, wd to 10) |
  | s4-R-f0.3 | 19 x 5 | lr 0.01 wd 3 | 15.35 / 4.18 (n/a) | +1.305 | False | False | **True** | True | flat over the extended grid (lr to 1e-1, wd to 10) |
  | s4-R-f0.45 | 12 x 5 | lr 0.01 wd 0.1 | 1.29 / 4.18 (on) | +1.311 | False | False | **False** | False | OPEN at wd bottom |
- Not changed: the newmani reading's other rule-1 parts, Stage 2's G rules (they inherit A1 for the wall test), and
  toyshape's own published reading (F89 stands as written; this table is a re-reading for the record, not a revision of
  F89).

### Amendment A2 (registered 2026-10-25 ~05:15 EDT: before any Stage 2 run exists, and before any result of the box5-m6 extension job 20875155 is read; basis: the results audit's F1, notes/batch1_build/audit_newmani_results.md; user approval 2026-10-25 "你觉得有必要的实验我觉得就可以跑")
- **Defect (F1).** Rule 1/2's reference config ("best none" = most groks, then the highest median AUC 4B) picks configs
  that cannot fit their own training set. lr 3e-2 wd >= 1 has a median final train of 0.012-0.034 in 29 of its 30
  Stage-1 configs, i.e. chance level on the training set. In a flat cell a crashed model sits at chance, while every
  memoriser sits below chance (the Latin property), so the crashed config wins the AUC ranking. Its late CE is about
  log n_out, so A1's gate is open, and its collapse / re-memorise cycles read as "CE still falling". In Stage 1 this alone
  vetoed the walls of 2i / a5 / d57 m6, and box5 m6 kept its wall by 0.009 nats.
- **Rule.**
  1. A none config is **trainable** iff the median over its seeds of `final_train` (training accuracy at the last eval
     of its 4 B run) is >= **0.99** (`readout.TRAINABLE_THR`, fixed now).
  2. The **reference config** = the trainable config with the most groks, then the highest median AUC 4B, then the lower
     lr. It replaces "best none" in two places:
     - rule 1's CE component: A1's gate is unchanged (on iff its late CE <= log(n_out) + 1, else n/a), and the drop
       >= 0.01 nats is read on this config;
     - rule 2's edge test: the bracket is closed iff this config is not on an edge of the grid that ran.

     Rule 2's flat test is unchanged (it is read over every config).
  3. Every other part of rule 1 is still read over EVERY config, crashed ones included:
     - groks <= floor(0.2 n) (Stage-1 scout: 0 / 3);
     - final held-out <= 0.20 (scout: every seed; G1: the median);
     - median accuracy gain < 0.02;
     - median final strict <= the floor line.
  4. **No trainable config** -> the reference config, the CE component and the edge test are n/a. The wall verdict is
     then "n/a: no trainable none config", because such a sweep shows only that none cannot be trained, not that it
     does not generalise. n/a never counts as a wall: G1 fails as n/a, and a Stage-1 tier reads "not a wall cell (A2 n/a)".
  5. **The G2 / G3 comparator is NOT restricted to trainable configs.** The coordinator asked for this decision and
     recommended restricting; that is not adopted, for three reasons:
     - A maximum over a subset can only be lower, so restricting can only make G2 easier.
     - In a flat cell the F1 mechanism (chance above below-chance) raises the all-config comparator by <= 0.006 AUC_B
       (box5 m6: 0.012 vs 0.008). That is the conservative direction, and negligible against G2's 0.05 margin.
     - A config that learns early and crashes later is a real "none within B" signal, which the restriction would
       drop. Example: box5 m24 lr 1e-2 wd 0.1 groks 3/3 at 1.3k-2.9k steps, but its median final train is 0.06.

     So the paired AUC_B tests use, per read-out set (all held-out; test_strict), the none config with the highest
     median AUC_B on that set over every config (ties -> lower lr). This is Stage 1's `nb`, applied per set. The
     comparator restricted to trainable configs is printed beside it, descriptive only. The 加速档's none t_grok is also
     read over every config (the earliest median t_grok in 4 B).
- **Why 0.99** (preflight/a2_threshold.py -> a2_threshold.log). The data are the 187 Stage-1 none configs (561 runs);
  the extension dirs are excluded.

  | band (median final train) | configs | which configs |
  |---|---|---|
  | >= 0.999 | 138 | memorisers |
  | [0.99, 0.999) | 5 | lr 1e-2 wd 3 x 3, lr 3e-2 wd 0.1 x 2 |
  | [0.9, 0.99) | 7 | lr 3e-2 wd 0.1 x 6, lr 1e-2 wd 3 x 1 |
  | [0.5, 0.9) | 7 | lr 1e-2 wd 3 x 4 (2i m24 at exactly 0.50: the +-c quotient only), lr 3e-2 wd 0.1 x 2, lr 3e-2 wd 3 x 1 |
  | < 0.5 | 30 | lr 3e-2 wd 1 x 15, lr 3e-2 wd 3 x 14, box5 m24 lr 1e-2 wd 0.1 x 1 (groks, then crashes) |

  - Below 0.99 there is no memoriser that fits its training set: those configs are crashed, oscillating (lr 3e-2
    wd 0.1) or held below a full fit by wd 3.
  - Healthy memorisers' single-seed final values dip to 0.991-0.998 at the last eval, so 0.999 would drop some of them.
  - AUDIT "Ranges to check" already registered "none memorises (train ~1.00)".
  - 0.99 is a point measure: an oscillating lr 3e-2 wd 0.1 config can pass it on 2 of 3 lucky last evals (the lowest
    last-25 % mean train among passing configs is 0.754).
  - The Stage-1 sensitivity of the A2 verdicts to the threshold (0.5 / 0.9 / 0.95 / 0.99 / 0.999) and to a last-25 %
    mean measure is reported below. It is a check only; the rule is 0.99 on final_train.
- **Where A2 applies.**
  - (i) Stage 2 (G1 via the reference config; G2 / G3 per item 5).
  - (ii) The reading of the box5-m6 extension: the Stage-1 cell box5 m6 plus its 7 extension configs, read with
    `readout.py --a2`.
  - (iii) A **post hoc** re-reading of Stage 1 (`readout.py --a2`). F94's pre-registered verdicts stand as written;
    the re-reading changes none of them.
- **Output.**
  - `readout.py` without --a2 is the pre-registered A1 reading, byte-identical to c1ce6fd75bd3's output (regression
    below).
  - `readout.py --a2` writes readout_a2.md / readout_a2.json. They hold the A1 text plus, per cell, an A1 | A2 line
    (reference config, trainable k / N, CE state, wall, bracket, tier verdicts) and a selection table with both
    columns.
- **Known answer.** The results audit's chk4 (post hoc, threshold 0.99) found:
  - the CE component is n/a in every flat cell;
  - 2i / a5 / d57 m6 become walls with floor tiers;
  - the selection is unchanged.

  `readout.py --a2` on the 1371 Stage-1 runs must reproduce this (result below).
- **Result on the Stage-1 data (POST HOC; preflight/a2_stage1_posthoc.py -> a2_stage1_posthoc.log,
  readout_a2_stage1_posthoc.md / .json; the extension dirs are excluded).**
  - Regression: the default call is byte-identical to c1ce6fd75bd3 (md + json) and equals the published readout.md
    beyond its root line. The --a2 output minus its A2 parts is the A1 output.
  - Known answer reproduced: the 10 flat cells all have the A2 CE component n/a (their reference configs are memorisers
    at late CE 15-51). The wall changes exactly at 2i / a5 / d57 m6, each now a wall with floor tiers. No task's
    selection changes: the box5 Tier L pick is still m 6; d57's label becomes "wall cells at m 6, 12".
  - box5 m6: reference config lr 1e-2 wd 3 (median train 0.999), CE n/a, wall True; A1 had lr 3e-2 wd 1 (train 0.022).
  - Sensitivity: thresholds 0.5 / 0.9 / 0.95 / 0.99 / 0.999 on final_train, and 0.95 / 0.99 on the last-25 % mean
    train, give identical A2 wall / tier verdicts and selections in every cell.
  - The reference config itself does move: 0.5 changes 1 cell, 0.999 changes 2, and the last-25 % 0.99 changes 6. In
    s5 m24 (not flat) the last-25 % reference takes the bracket from closed to "OPEN at wd top".
  - So the verdicts are not on a knife edge here, but the reference / bracket in a non-flat cell can depend on the
    measure. This is the reviewer's item 1, and is why Stage 2 prints the last-25 % G1 beside the registered one.

### Stage 2 seeds (PRE-REGISTERED 2026-10-21, audit I2; preflight/stage2_seeds.json)
Rule: for each Stage-1 m cell (same task, same frac), take the first 5 seeds scanning upward from seed 100 that pass the
same checks as Stage 1 (split asserts, closure 1.0 [+ box nullity 3], targets incl. shuf far < 0.15) and are not Stage-1
seeds. This is fixed now, before any Stage-1 result; a Stage-2 cell runs on exactly these seeds.

| task | m level | frac | Stage-1 seeds | Stage-2 seeds | seeds refused before them (from 100) |
|---|---|---|---|---|---|
| box5 | 6 | 0.067 | 4, 10, 25 | 100, 103, 108, 122, 126 | 22 |
| box5 | 12 | 0.11 | 0, 1, 2 | 100, 101, 102, 103, 104 | 0 |
| box5 | 24 | 0.22 | 0, 1, 2 | 100, 101, 102, 103, 104 | 0 |
| 2i | 6 | 0.067 | 6, 14, 22 | 121, 144, 165, 178, 186 | 82 |
| 2i | 12 | 0.1 | 0, 3, 4 | 100, 103, 104, 105, 107 | 3 |
| 2i | 24 | 0.2 | 0, 1, 2 | 100, 102, 103, 104, 105 | 1 |
| d57 | 6 | 0.053 | 7, 14, 31 | 114, 127, 138, 148, 175 | 71 |
| d57 | 12 | 0.105 | 1, 3, 4 | 101, 102, 103, 104, 105 | 1 |
| d57 | 24 | 0.21 | 0, 1, 2 | 100, 101, 102, 103, 104 | 0 |
| a5 | 6 | 0.1 | 1, 3, 5 | 100, 103, 106, 107, 108 | 4 |
| a5 | 12 | 0.2 | 0, 1, 2 | 100, 101, 102, 103, 104 | 0 |
| a5 | 24 | 0.4 | 0, 1, 2 | 100, 101, 102, 103, 104 | 0 |
| s5 | 6 | 0.05 | 3, 7, 9 | 101, 102, 107, 109, 111 | 7 |
| s5 | 12 | 0.1 | 0, 2, 3 | 100, 101, 102, 103, 104 | 0 |
| s5 | 24 | 0.2 | 0, 1, 2 | 100, 101, 102, 103, 104 | 0 |

**Stage 1 cannot claim a rescue**, of either tier. It has 3 seeds and no confirm seeds, the prior lr bracket may be
open, and none's bracket may be open. "CANDIDATE" means only "worth Stage 2".

### Stage 2 (box5 m6, Tier L) — registered 2026-10-25, before any Stage-2 run exists (user approval 2026-10-25)
**Amendment A2.1** (coordinator's decision, 2026-10-25). Implemented and self-tested 06:46-06:49 EDT, written here 06:50
EDT.
- **State of the data at that time.** It was checked twice (06:46:27, 06:49:39), and Stage 2 had no results:
  - 20877109 started at 06:48:08, but out_stage2/ had 0 run_*.json;
  - 20877110 was PENDING;
  - every lane runs its none runs first, about 0.36 h each, so no JSON can exist before ~07:10.
- **Rule.**
  - A2's registered measure stays: trainable = median final_train >= 0.99.
  - The second measure becomes a **verdict gate** (it was descriptive): trainable = median last-25 % mean train >= 0.99.
  - The whole Stage-2 verdict is computed with each measure. If the two G1s (pass / fail) differ, or the two verdict
    categories (PASS / 加速档 / other) differ, the conservative reading applies. The verdict is then **"UNDETERMINED
    (the reference config depends on the training-fit measure)"**, with both results listed; it is never PASS and never
    加速档.
  - Read literally, a G1 disagreement gives UNDETERMINED even when both verdicts are NOT PASSED.
  - When the two agree, the verdict is the registered one.
- **Where.** readout_stage2.py.
  - It was sha d89b62f65244 at registration. After the fresh review of A2.1 (no blocking item) it is **a76a41c56204**:
    the review's presentation fixes plus 2 more self-tests, with no change to the rule. Earlier versions are kept as
    preflight/superseded_readout_stage2_*.py.
  - **Timing, recorded as it happened.**
    - The first Stage-2 run JSON was written at 07:10:45. That is after A2.1's registration (06:50, d89b62f65244)
      but before the last presentation edit (07:11:37, a76a41c56204; self-test passed 07:11:44).
    - That edit does not touch the rule: the agree / UNDETERMINED condition, the two measures, and G1-G4 are
      identical to d89b62f65244. It changes only the text, a JSON key name, asserts and tests.
    - No Stage-2 JSON was opened or read by the builder at any point.
  - **J["verdict"] is THE verdict** (A2 + A2.1).
  - J["verdict_A2_final_train_only"] and J["verdict_late"] are each measure read alone. They are never reported as the
    verdict (the review's important item: the old key name "verdict_reg" invited that).
  - The UNDETERMINED text names only categories and G1 states, never a verbatim PASS / 加速档 verdict.
- **Known limitation, kept as registered.** The last-25 % mean (readout.train_level(r, "late")) includes the run's final
  off-grid eval (step 4 B - 1): 501 evals against late_change's 500, an effect of <= 0.002 on the mean.
  - It is not changed, because Stage-2 JSONs could appear any minute (first none runs due ~07:10).
  - readout.py's train_level docstring still says "a sensitivity check only". It predates A2.1, and is left untouched
    to keep readout.py at its regression-tested sha 2182373026af.
- **Self-tests** (readout_stage2_selftest.log, STAGE 2 READOUT SELFTEST PASSED):
  - 1-7 are unchanged and pass. The two measures agree there; scenario 1 asserts it explicitly.
  - 8-10b (INCOMPLETE / NO VERDICT) return before A2.1, as intended.
  - 11: 加速档 vs PASS -> UNDETERMINED.
  - 12: PASS vs 加速档 -> UNDETERMINED. A config fits its training set except at the last eval (0.98), so it is the
    last-25 % reference only, and its CE fires.
  - 13a: G1 differs while both are NOT PASSED -> UNDETERMINED.
  - 13b: 加速档 vs NOT PASSED (G1 n/a) -> UNDETERMINED, not 加速档.
  - 14 (the bracket path, added after the review): a non-flat cell. A2's reference is interior (closed, PASS); the
    last-25 % reference is a grid corner (OPEN, 加速档) -> UNDETERMINED.
  - 15: PASS vs G1 n/a -> UNDETERMINED.
- **Coordinator's confirmation, 2026-10-25, of A2 item 5.** The G2 / G3 comparator is NOT restricted to trainable
  configs (all configs, the maximum), which is stricter for the prior.
- **P2.** It is submitted only after P1 has results, readout_diag reports the trigger met, and the coordinator decides.

- **Cell.** F94's pre-registered selection: box5 Tier L "select m 6"; A2's re-reading leaves it unchanged.
  - `--task box --box=-2:2,-2:2,-2:2`, frac 0.067, class split.
  - Seeds 100, 103, 108, 122, 126 (the "Stage 2 seeds" table). They are asserted in three places: the generator (=
    stage2_seeds.json, none of them a Stage-1 seed), the reader (every run's split_sha = stage2_seeds.json's), and the
    smoke (the trainer's split_sha / n_train / shuf-far CKA = the json's).
  - n_train 912-918 (m 7.3), at most 1,800 (G4).
- **Arms** (140 runs, B = 50k):
  - none: the 12 Stage-1 configs plus the 7 rule-2 extension configs (lr 1e-1 x wd {0.1, 1, 3}; wd 10 x the 4 lrs),
    5 seeds, 4 B = 200k. 95 runs. The extension runs up front because rule 2 needs it for a flat cell.
  - expo@always (assign, Tier L), specrand@always, shuf@always: lr {1e-3, 3e-3, 1e-2}, wd 1, lam 1, 5 seeds, B. 45 runs.
- **Not in Stage 2.**
  - **Tier S.** Stage 1 read Tier S at box5 m6 as a floor: shape 0/3 at every lr, AUC_B <= 0.013, and both controls
    0/3. The selection is per task and tier, and Tier S selected nothing for box5 (F94 table). Stage 2 confirms selected
    cells only, so re-running a floor would be an unregistered second look.
  - The [0, 2000) window schedule ("Stage 2's secondary reading", Arms section) and the design's table-3.3 extras
    (quot / onehot targets). They are not in the coordinator's arm list; Open decision 4 stays the user's. The reading
    is the primary @always schedule only.
  - Prior lr 3e-2 (Open decision 3). Stage 1's assign lr bracket was OPEN at the bottom (1e-3 best). G2 takes the best
    of the 3 registered lrs; the bracket state is printed, not gated.
- **Reading** (readout_stage2.py; G1-G4 of the design + A1 + A2; implemented and self-tested before any run):
  - **G1 wall.** Every one of the 19 none configs, crashed ones included, must meet all four:
    - groks on <= floor(0.2 x 5) = 1 seed within 4 B;
    - median final held-out <= 0.20;
    - median last-25 % gain < 0.02;
    - median final strict <= the floor line.

    In addition, the A2 reference config's CE drop must be < 0.01 nats while A1's gate is on. The bracket must also be
    closed: flat over the extended grid, or the reference config not on a grid edge. With no trainable config, G1 is
    n/a and fails.
  - **G2 rescue.** Assign at its best lr (median AUC_B over all held-out) must meet both:
    - it groks within B on >= 4/5 seeds;
    - the paired AUC_B test passes on all held-out AND on test_strict: compare_values' AUC part, i.e. median
      difference >= 0.05, wins >= 4/5, exact one-sided MWU p < 0.025, and the assign median above the floor line.

    The comparator follows A2 item 5: per set, the none config with the highest median AUC_B over every config.
  - **G3 specific.** Assign beats specrand AND shuf, each at its own best lr, by the same test on both sets.
  - **G4.** Assign's median t_grok_B <= 0.4 B = 20k, and n_train <= 1,800.
  - **Verdict.**
    - G1-G4 pass -> PASS (next rainbow round).
    - G1 fails, G2-G4 pass, and none's earliest median t_grok in 4 B (over every config; inf = never) divided by
      assign's median t_grok_B is >= 3 -> 加速档 (the user's call).
    - G1 n/a (no trainable none config) -> NOT PASSED with no 加速档 reading, because the baseline did not train.
    - Otherwise NOT PASSED, with the failing Gs named.
    - A1's G1, and the verdict it would give, are printed beside A2's for the record.
  - **Second measure: a verdict GATE since Amendment A2.1** (added as descriptive after the review, made a gate by
    A2.1, both before any Stage-2 result):
    - a G1 with trainable = median last-25 % mean train >= 0.99, and the verdict it gives;
    - if it differs from the registered measure in G1 or in category, the verdict is UNDETERMINED (see A2.1 above);
    - the per-seed final / last-25 % mean / min train of every reference config and of the G2 comparators is printed.
  - **Data gaps.** Any missing config / lr / seed, an extra config or lr, or B != 50k -> INCOMPLETE: nothing is read.
    A foreign seed, a wrong split_sha / frac / split, a second code_sha or device, a code_sha other than the pin
    ea797e5321d3, or a prior run off wd 1 / lam 1 -> NO VERDICT.
  - **Wording** (from the design): "not learned" only within this one-layer toy, these optimisers and the 4 B budget;
    "the target encodes the table" (Tier L); "geometry-specific" only from G3; one task, one tier, two read-out sets.
- **Self-test** (`readout_stage2.py --selftest` -> preflight/readout_stage2_selftest.log), synthetic, before any data:
  1. PASS.
  2. The F1 situation: A1's G1 fails on a crashed config's CE, A2 passes.
  3. G1 fails, ratio 15 -> 加速档.
  3b. G1 fails, ratio 2 -> NOT PASSED.
  3c. G1 fails on a trainable reference's CE, none never groks -> 加速档.
  4. G2: 3/5 groks.
  4b. G2: strict at the floor.
  5. G3: shuf groks too.
  6. G4: t_grok > 0.4 B.
  7. No trainable none config -> NOT PASSED (G1 n/a).
  8. INCOMPLETE x 3: an extension config, a prior seed, an assign lr.
  9. NO VERDICT x 3: a Stage-1 seed, a second code_sha, a wrong split_sha.
  10. B != 50k -> INCOMPLETE.
  10b. code_sha != the pin -> NO VERDICT.
  11. An oscillating reference: the registered G1 fails on its CE (加速档, "none never groks"); the last-25 % G1 passes.
      Since A2.1 the verdict is UNDETERMINED.
  12-15. The A2.1 cases (see A2.1 above).
- **Jobs and cost.**
  - `make_jobs_newmani.py --stage2` renders 2 jobs s2-box5-m6-j0 / j1 (wave 3, never selected by --wave), each 5 lanes,
    out_stage2/.
  - Rate model (Device section): 47.1 process-h = 9.5 planned L40S-h (2 x 4.76 h); --time 10 h each.
  - Stage 1 measured box5 m6 at 6.5 ms/step for none and 3.4-4.8 for assign / controls, against the model's 7.96, so
    the actual cost is likely ~15-20 % below plan.
  - Account torch_pr_625_general, l40s_public, gpu48, --nice 500; the same snapshot (toy_struct ea797e5321d3), sha and
    L40S checks as Stage 1.
  - **Submitted 2026-10-25 05:59:53 EDT** (after --print and the CPU smoke of the rendered Stage-2 and P1 commands; the P2 smoke finished later): s2-box5-m6-j0 =
    **20877109**, s2-box5-m6-j1 = **20877110** (`--stage2 --diag --diag_q --submit --only ... --nice 500`).

### Amendment A3 — none's low-wd direction on box5 m6 (POST HOC; registered 2026-10-25 11:37 EDT, before any low-wd run exists: out_a3/ does not exist, nothing is submitted)
- **Basis.**
  - The Stage-2 reading is PASS (G1-G4). The results audit (notes/batch1_build/audit_newmani_stage2_results.md) found
    no blocking item, but its important item I1 is that none's wd bottom edge (0.1) was never extended.
  - Rule 2's extension only goes up (lr 1e-1, wd 10). The same task family learned at the wd bottom:
    - the {-1,0,1}^3 dimension lattice (= box -1:1) at lr 1e-2 wd 0.03, 2/5 grokking at 181k / 190k of 200k (F84
      addendum g3, FINDINGS line ~3535);
    - Stage 1 box5 m12, whose only none grok was at wd 0.1 (bracket "OPEN at wd bottom").
  - Coordinator's decision with the user's approval, 2026-10-25. Everything in A3 is post hoc: it is added after the
    Stage-2 result, and it can only remove the "rescue" label, never add one.
- **Runs, A3 main** (job group a3-box5-m6, out_a3/): 45 runs.
  - box5 m6, the Stage-2 seeds 100 / 103 / 108 / 122 / 126.
  - none at lr {3e-3, 1e-2, 3e-2} x wd {0.03, 0.01, 0}: 9 configs x 5 seeds, at 4 B = 200k.
  - Every other flag is identical to the Stage-2 none runs: class split, frac 0.067, shape_log, eval every 100,
    checkpoints, the same snapshot ea797e5321d3, L40S.
- **Runs, A3-long** (my addition; group a3long-box5-m6, out_a3/, cells suffixed "-16B"): 10 runs.
  - none at lr 1e-2 x wd {0.03, 0.01}, the same 5 seeds, at **16 B = 800k steps**.
  - Why: the only precedent grokked at 0.9-0.95 x 4 B at wd 0.03. Grokking time grows as wd falls, so at wd 0.01 a
    late grok could sit beyond 4 B.
  - The 4 B rule's "still rising" test does not catch a step-like grok after a flat plateau.
  - It is bounded: the precedent's lr, and the two lowest non-zero wds.
  - wd 0 is not extended. Grokking in this family needs wd, and wd 0 is covered at 4 B with the rising test.
  - lr below 1e-3 is not added. In this family a lower lr only slows learning within a fixed budget: F84's S4 "wall"
    was an lr that was too low, cured by a higher lr.
- **Pre-registered reading** (readout_a3.py; the 4 B part decides, A3-long can only qualify):
  1. **Union G1.** G1 is recomputed exactly as in Stage 2 over the 28 none configs: the 19 Stage-2 configs plus the 9
     new ones.
     - Every config must meet rule 1's wall conditions: groks <= floor(0.2 x 5) = 1, median final <= 0.20, median
       last-25 % gain < 0.02, and median final strict <= the floor line.
     - The A2 reference config is chosen over the union. Its CE drop must stay < 0.01 nats while A1's gate is on.
     - The bracket must be closed over the union grid: flat over the extended grid, or the reference not on an edge.
       wd 0 is now the wd bottom edge.
     - This is done under both training-fit measures (A2 final_train, A2.1 last-25 % mean). A disagreement in G1 or in
       the category gives **UNDETERMINED** (A2.1).
     - Each new config's four wall conditions are also printed one by one, with whether it is the reference and
       whether its CE fires.
  2. **Verdict** (4 B). Stage 2's 19 configs alone passed G1, so any union G1 failure comes from the new configs.
     - Union G1 passes under both measures -> **"低 wd 方向也是墙：Tier L 挽救在 wd ∈ {0, 0.01, 0.03, 0.1, 1, 3, 10} ×
       lr {1e-3..1e-1} 的范围内成立"**, within the 4 B budget, subject to A3-long (item 4).
     - Union G1 fails, the A2 reference config is at wd 0 and rising (its gain >= 0.02 or its CE component fires),
       and no new config groks > 1/5 or has a median final > 0.20 -> **"区间仍开着（wd 下沿）"**: not a rescue.
     - Any other union G1 failure -> **"G1 在低 wd 方向不成立 → 不是挽救（墙依赖于没测的优化器方向）"**.
     - Missing runs, foreign data, or Stage 2 not complete / not PASS -> INCOMPLETE / NO VERDICT. The checks are as in
       Stage 2: seeds, split_sha, frac, split, code_sha = the pin, a single device, and wd / steps exactly the
       registered ones.
  3. **G2, descriptive only.** It is the same test as Stage 2's G2 (compare_values' AUC part, all held-out and strict).
     It compares assign (Stage 2's pick: lr 1e-3, its 5 seeds) against each new config, and against the union
     comparator (the highest median AUC_B over all 28 configs per set).
  4. **A3-long** (16 B), per config:
     - groks within 16 B (t_grok of the run), median t_grok, median final at 16 B, last-25 %-of-16 B gain, and median
       final train.
     - If a config groks on >= 2/5 seeds within 16 B, a 4 B "wall" verdict may NOT be worded "rescue" without the
       qualifier "within the 4 B budget; none learns box5 m6 by ~N B at lr 1e-2 wd X". Whether that is still called a
       rescue (4 B) or a 加速档 is the user's call.
     - Otherwise the verdict carries "none 0-1/5 per config within 16 B at lr 1e-2 wd {0.03, 0.01}".
     - A3-long never upgrades a 4 B failure. While it is incomplete, a 4 B wall verdict reads "PROVISIONAL (A3-long not
       complete)".
- **Cost** (rate model 1.2 + 5.54e-3 x 1220 rows = 7.96 ms/step):
  - A3 main: 45 x 0.44 h = 19.9 process-h, about 4.0 L40S-h.
  - A3-long: 10 x 1.77 h = 17.7 process-h, about 3.5 L40S-h.
  - Total about 7.5 planned L40S-h. Stage 2 measured 6.5 ms/step, so the actual cost is likely ~20 % lower.
- **Code, tests, submission** (2026-10-25):
  - **Code.**
    - readout_a3.py sha 0b7c26dbd680 at submission, 9a6a061f706a after A3.1 (below). readout_stage2.py (a76a41c56204) and
      readout.py (2182373026af) are unchanged.
    - make_jobs_newmani.py b2adb640e352: `--a3`, append-only and sticky. The previous version is kept as
      preflight/superseded_make_jobs_newmani_fd37f308abe8.py.
  - **Tests.**
    - `readout_a3.py --selftest` -> readout_a3_selftest.log: **A3 READOUT SELFTEST PASSED**. It covers 12 cases: all
      wall; one new config groks; wd 0 rising (OPEN); wd 0 at the cap (FAIL); A3-long groks (QUALIFIER); A3-long
      incomplete (PROVISIONAL); A3 incomplete; an A2.1 disagreement (UNDETERMINED); a foreign wd; foreign steps;
      Stage 2 not PASS (n/a); code_sha != pin.
    - test_generator_a3.py -> test_generator_a3.log: **GENERATOR A3 TESTS PASSED**.
      - Old (fd37f308abe8) vs new render byte-identical job files and stdout for the default, --ext_cells and
        --stage2 --diag --diag_q calls. The default equals the 32 real job files.
      - --a3 adds exactly 45 + 10 runs with the Stage-2 none flags, and they are sticky.
      - A tampered submitted file is REFUSED.
    - a3_preflight.py -> a3_preflight.log: **A3 PREFLIGHT PASSED**.
      - readout_stage2.py on out_stage2 equals the published readout_stage2.md / .json.
      - readout_a3.py on the real Stage 2 plus an empty A3 root: Stage 2 PASS, 0 A3 configs read, INCOMPLETE.
      - CPU smoke of 3 rendered commands: wd 0 / 0.01 / 0.03 consumed, split_sha equals stage2_seeds.json, decoded
        inputs printed.
  - **Submitted 2026-10-25 11:45 EDT**, before that out_a3/ did not exist: a3-box5-m6-j0 = **20892854**, j1 =
    **20892856**; a3long-box5-m6-j0 = **20892858**, j1 = **20892860** (`--a3 --submit --only ... --nice 500`; L40S,
    torch_pr_625_general).
  - **Planned cost:** 37.6 process-h = 8.0 L40S-h; --time 5 + 5 + 4 + 4 h.
- **Amendment A3.1** (from the A3 review's important item I1). Registered 12:08:11 EDT, with 0 run JSONs in out_a3/ (none
  read).
  - **The rule.** The union grid is L-shaped: wd < 0.1 was run only at lr 3e-3-3e-2. A3's bracket edge test therefore
    becomes LOCAL. The reference config is "on an edge" when any of its four axis neighbours is missing from the grid
    that ran:
    - the previous / next lr of the union's lr list at the same wd;
    - the previous / next wd of the union's wd list at the same lr;
    - a global extreme also counts as an edge.
  - The flat rule is unchanged. Stage 2's own reading is unchanged (its cell was flat).
  - The A3 reader also prints, descriptively and not as gates, per new config:
    - whether it is the reference, and whether its CE fires;
    - the median over seeds of its maximum held-out at any eval (flagged when > 0.20);
    - the number of non-finite test_ce evals;
    - the 加速档 ratio on a FAIL / OPEN, so the user can judge.
  - A3-long's qualifier reports the 2nd-earliest t_grok, the time at which >= 2/5 is reached. It is triggered as soon
    as >= 2 seeds grok, even while A3-long is incomplete.
  - **Timing, recorded as it happened.**
    - The A3.1 text was saved at 12:08:11, when out_a3/ held 0 JSONs.
    - The first A3 JSON was written at 12:08:59. The A3.1 code was saved at 12:09:06 and finished (one self-test string,
      one test isolating the bracket) by ~12:20.
    - No A3 JSON was opened by the builder at any point.
    - readout_a3.py is sha **9a6a061f706a** (the pre-A3.1 version 0b7c26dbd680 is kept in preflight/).
  - **Self-tests now** (readout_a3_selftest.log, A3 READOUT SELFTEST PASSED): the 12 earlier cases plus:
    - A3-long at exactly 2/5 -> qualifier with the 2nd t_grok;
    - A3-long incomplete with 2 groks -> qualifier, not PROVISIONAL;
    - an L-shaped grid edge: the reference is at lr 3e-3 wd 0.03 with held-out 0.10 and strict at the floor. G1 fails on
      the bracket alone ("neighbour lr 0.001 wd 0.03 not run"), where the global rule would say closed.
  - **A3 review** (fresh general-purpose agent, minimum mode, after submission, read-only). **No blocking item.**
    - Verified:
      - the jobs are exactly 45 + 10 runs, flag-identical to Stage 2's none runs;
      - wd 0 reaches AdamW;
      - the 16 B runs fit their --time;
      - the generator is append-only and never submits by accident;
      - the reader matches the text, with no mixing of 4 B and 16 B quantities;
      - the G2 inputs are correct.
    - **Important I1** (L-shaped edges): fixed by A3.1.
    - **Minor items:**
      - M2 (2/5 gives a median of inf), M4 (an incomplete A3-long masks the qualifier), M8 (no 加速档 ratio printed),
        M9 (is-reference / CE not printed per config), M10 (no finiteness count): fixed by A3.1.
      - M1 (a config that rises to ~0.9 and falls back passes all four conditions), M3 (the "×" range wording), M5 (the
        16 B qualifier ignores final / gain), M6 (the wd smoke only echoes the value), M7 (wd 0 is a physical edge: one
        lucky grok gives "OPEN at wd bottom", the conservative direction): kept as registered and noted here.
    - The reader now prints each config's max held-out at any eval (M1) and the list of configs that ran. Any
      conclusion must name the configs actually run and trainable per wd (M3), not the full "×" product.
- **Not changed.**
  - F94 and the Stage-2 pre-registered verdict: the Stage-2 readout files are not rewritten.
  - readout_stage2.py and readout.py are unchanged (the A3 reader is a separate file).
- **To-do before any rainbow design (results audit M4 = M9; coordinator 2026-10-25).** Log the prior-vs-CE
  gradient share (one autograd.grad at eval steps). It needs a trainer change plus a regression, and it is not done
  now.

### Diagnostics (POST HOC, 2026-10-25; results audit 2d P1 / P2; not part of any pre-registered reading)
- **Question: pipeline vs configuration only** (audit F2). The pre-registered positive control used the slowest
  config, lr 1e-3 wd 1. F94's verdicts for 2i / a5 / s5 (NOT READ) do not change, whatever these runs show.
- **P1** (jobs diag-pos-j0 / j1, 2 x 6 lanes x 1 run; rate model 2 x 2.2 h = 4.4 planned L40S-h, --time 5 h each;
  out_diag/). **Submitted 2026-10-25: diag-pos-j0 = 20877111, diag-pos-j1 = 20877112.**
  - Runs: 2i and s5 at frac 0.4 on the Stage-1 positive-control splits (seeds 0, 1, 2; the smoke checks that split_sha
    equals the Stage-1 runs'). none lr 3e-3 wd 3 and lr 1e-2 wd 1, 200k. 12 runs.
  - Why these two configs: in a5 f 0.4, lr 3e-3 wd 3 grokked 3/3 (5.5k-67.5k steps) and lr 1e-2 wd 1 grokked 3/3
    (3.3k-30.9k). In s5 f 0.2, lr 1e-2 wd 1 seed 2 grokked at 64.5k.
- **P1 reading** (readout_diag.py, self-tested), per task:
  - a config groks (held-out >= 0.95 at any eval of the 200k run) on >= 2/3 seeds, with every seed present -> "pipeline
    OK (post hoc): none learns <task> at f 0.4 with lr X wd Y". Then the Stage-1 failure is attributed to the
    pre-registered control's configuration, not to the pipeline.
  - both configs complete and <= 1/3 -> "not shown: pipeline vs configuration undecided";
  - otherwise INCOMPLETE.
- **P2, CONDITIONAL — the coordinator decides; rendered with `--diag_q`, NOT submitted.**
  - **Trigger:** both 2i P1 configs complete with 0/3 groks. readout_diag.py prints "P2 trigger: met / not met /
    pending".
  - **Runs:** 2i, frac 0.4, `--split question` (a random question split: twins of held-out cells can be trained), none
    lr 1e-2 wd 1, seeds 0, 1, 2, 200k. 3 runs, 1 job diag-q-2i-j0 (1.84 h planned, --time 4 h).
  - **Choices.** The config is the audit's P2: the fastest-grokking a5 f 0.4 config, and the one s5 grokked with. Seeds
    0-2: the question split has no seed rule; the trainer's run-time asserts (coverage, closure 1.0) are the check, and
    seeds 0-2 pass them in the smoke (n_train 5760).
  - **Command:** `make_jobs_newmani.py --diag_q --submit --only diag-q-2i-j0 --nice 500`.
  - **Reading** (reworded after the review: ~97-98 % of the question split's held-out cells have a trained twin):
    - >= 2/3 groks -> "none completes the 2I table at f 0.4 when the twins are trained: what the orbit split removes is
      what none needs". This does NOT show that none learns 2I's group law.
    - 0/3 -> "none does not learn the 2I table at f 0.4 even with ~98 % of the held-out cells' twins trained: the limit
      is this model family / budget / config, not the split". Then 2i's m cells are not "a wall that moves with m", and
      that is reported as such.
    - 1/3 -> not shown.

    It is a diagnostic only. It is never read as "2I is learnable" or as "the Stage-1 wall is a split artefact".

## Expected outcomes (registered before any run; design 3.4 + mine)
- **Design 3.4 predictions.**
  - box5^3: none fails at m 12 (uncertain); expo > specrand ~ shuf ~ none.
  - 2I: quat4 rescues; specrand, shuf ~ none.
  - D57: the RR block is learned by every arm (it is addition mod 57).
  - A5: rot3 ~ rot3'.
  - S5: if 2I rescues and S5 does not, that is support for compactness (inference).
- **Design's overall guess (an estimate, not data).** Of the 5 tasks, 1-2 pass G1, most likely the box. Groups more likely
  land at "acceleration" or "floor" (Chughtai et al. learn S5 / A5 / D59 / D61 at 40 %; S4 0.6 is learned by tuned none).
- **Mine, for Stage 1.**
  - Positive controls grok (Chughtai's setting).
  - At m 24 none probably learns the groups at lr 1e-2 (S4 0.6, m 14.4, learns), so m 24 is likely "not a wall".
  - m 6 is likely a floor for every arm (F84: R@always at S4 0.3, m 7.2, lr 1e-3 stayed at chance).
  - So m 12 is the likely informative level.
  - Tier S: no shape arm broke a wall in toyshape on the same toy so far (to be read from wall 2); expected weaker than L.
- **Ranges to check before reading** (anything outside -> look first):
  - none memorises (train ~1.00) within a few thousand steps at lr >= 3e-3;
  - held-out at or below chance (~0.008) for the Latin-memoriser signature;
  - every shape arm's own loss <= 0.05 after a few hundred steps;
  - the shape arms' CKA to the true target near the random-subspace level (2I 0.03, box 0.02; test_newmani §3) unless
    the data pick the assignment;
  - assign arms' CKA to the true target -> ~1;
  - specrand's CKA to the true target 0.02-0.18;
  - shuf's < 0.15 (asserted);
  - possible divergence at lr 3e-2 (a non-finite CE counts as not learning).

## Known limitations (audit minor items not fixed in code)
- **M5, selection reading.** The design says "the smallest m; with several, the one closest to 12". readout.py picks the
  closest to 12 (ties -> smaller). With candidates {6, 12} that is 12, where the first sentence alone would give 6. The
  user picks the Stage-2 cells anyway; the readout lists every candidate m.
- **M7, the grp / box GPU path is unexercised.** Every known answer ran on CPU, and the smoke used a stub nvidia-smi. A
  device audit of run_arm found no CPU / GPU mixing, and shape arms already ran on L40S in toyshape.
  - CUDA's embedding backward is non-deterministic, so "resume = straight run bitwise" holds on CPU only; on GPU it is
    statistical.
  - **Submit one calibrator first** (2i-pos-j0, 3 runs, ~2 h). Check `device == cuda`, ms_per_step vs the rate model
    (planned 1.2 + 5.54e-3 x rows), and the log for errors, then submit the rest.
- **M8, GPU utilisation / preemption.** 5 single-thread small models per L40S may run at low utilisation (the gpu48 note:
  < 40 % risks preemption).
  - --requeue + 10k-step checkpoints make a preemption cost time only.
  - A cancellation without requeue shows up in `--status` as ENDED with runs missing (item 6) and in the reader as
    INCOMPLETE (I1).
- **M9, no prior-vs-CE gradient share is logged** (the project principle "the prior is an auxiliary loss" is reported for
  the 53M runs, not here). In the toy the prior dominates the W_E update once training CE ~ 0, which is the intended
  mechanism.
  - Before a Stage-1 cell feeds a rainbow design, its gradient share must be measured (cheap: one extra autograd.grad at
    eval steps).
  - Not added now, to keep the audited training path unchanged.
- **Token-level relabelling twins of the box** (coordinate permutations, sign flips other than the global one): neither
  excluded nor reported (design: "only reported"). Their effect is bounded like M1's (<= frac); a report-only subset
  needs a relabelling orbit computation and was left out.

## Device and cost (criterion f: L40S for every arm)
- **Why L40S, not cpu_short.**
  - toyshape's 1,955 finished L40S runs (5 processes per GPU, 1 thread each) give ms/step/process linear in the training
    rows: mod n_train 638 / 1276 / 1915 / 2553 / 3830 -> 4.7 / 7.7 / 11.2 / 14.6 / 22.4. That is 1.2 + 5.54e-3 x rows,
    with the eval over a 12.8k-question table included. Shape arms add 0.6-1.5.
  - On CPU (1 thread) the same 2I m 12 run takes 52 ms/step (bare loop) to 144 ms/step (full run on this host, load
    146 / 128 cores).
  - CPU per task type, measured on this host (1 thread, bare train step; full-table eval forward 0.36-0.47 s for the
    120-item groups and the box, 0.05-0.07 s for A5):

    | task type (rows, T, outputs) | ms/step |
    |---|---|
    | grp 120 items (720 / 1440 / 2880 rows, T 3) | 30 / 52 / 128 |
    | box 125 items (1500 / 3000 rows, T 4) | 82 / 193 |
    | a5 60 items (720 / 1440 rows, T 3) | 22 / 46 |
    | dim (206 rows, T 4) | 10 |
  - So one L40S (5 processes) is about 1-2.5x the whole 32-CPU cpu_short QoS cap.
  - gpu48 allows 16 GPUs per user and 48 h per job, so no run needs splitting under cpu_short's 6 h MaxWall.
  - The F84 / wall-2 toy data are also on L40S.
- **Rate model.** rows = n_train x T / 3 (box T 4); planned ms/step = 1.2 + 5.54e-3 x rows (+1.0 for shape arms).
  `--time` = 2 x the job's planned time (min 2 h).
- **Planned L40S-h** (sum over processes / 5):
  - box5: m6 4.4, m12 6.8, m24 13.0, positive controls 6.9 (f 0.4 + 0.6, 2 configs, 6 lanes per GPU);
  - 2i: m6 3.6, m12 5.1, m24 9.4, pos 1.1;
  - d57: m6 2.8, m12 4.8, m24 9.0, pos 1.0;
  - a5: m6 1.8, m12 2.9, m24 5.1;
  - s5: m6 2.9, m12 5.1, m24 9.5, pos 1.1.
  - The m 24 cells are 46.0 of it. `--ext` (every cell) adds ~37 L40S-h in 15 separate '-ext-' jobs.
- **Jobs.** 26 jobs, 1,371 runs (generator output, 2026-10-21).
  - Wave 1 (box5 + 2i): 13 jobs, 555 runs, 51.8 planned L40S-h, --time sum 110 h.
  - Wave 2: 13 jobs, 816 runs, 48.2 planned L40S-h, --time sum 102 h.
  - **Total 100.0 planned L40S-h.**
  - Per job: 5 lanes (6 for box5's positive controls, 3 for the other positive controls), 1.7-5.2 h planned, --time
    4-11 h.
  - Per task (L40S-h): box5 31.5, 2i 20.3, d57 18.6, a5 10.0, s5 19.6.
  - These are job times (the longest lane).
- **Wall clock.**
  - At the 16-GPU cap: wave 1 ~3 h, everything ~6 h, if nothing else of ours runs.
  - Our own ~29 queued llmadd GPU jobs (ptlr 500m, ptheld, toyf84fu; ~170 GPU-h of requested time) share that cap, and
    l40s_public had ~1,500 pending jobs at build time. A realistic range is therefore 0.5-1.5 days for both waves.
  - The same work on cpu_short (32-CPU cap) would be ~3,300-7,000 CPU-h = 4-9 days.
- **Requeue / time limit.**
  - Every run saves a checkpoint every 10k steps.
  - It stops at `--deadline` (job end - 15 min) with a checkpoint.
  - A requeued or resubmitted job skips finished runs (JSON present) and resumes checkpointed ones.
  - `--status` lists done / partial per job, the Slurm state (sacct), and separately every job that ENDED (e.g. COMPLETED
    after a deadline exit 0) with runs missing; `--submit --only <jobs> --redo` resubmits those (never one still in
    squeue).

## Code and provenance
- **Trainer**: `tools/toy_struct.py` in the worktree, uncommitted (PUBLIC repo: no secrets / emails; nothing but code).
  sha256[:12] **ea797e5321d3** (2026-10-21; was d0aed90927fd before the audit fixes).
  - New, all opt-in:
    - tasks `--task box --box`, `--task grp --group`;
    - read-outs (test_implied / nonimplied / strict / norewrite / blocks, label shares; JSON split facts);
    - a run-time assert that shuf is far (max CKA < 0.15 to the faithful targets; not for the s4 known-answer path);
    - `--ckpt_dir` / `--seg_end` / `--deadline` / `--ckpt_every`.
  - Existing code touched only behind task checks (`in ("dim", "box")`, `in ("s4", "grp")`, `D.get(...)` for keys only
    new tasks set).
- **Snapshot**: `runs/newmani/code/tools/` holds toy_struct.py, test_toy_struct.py, toy_grok.py and the preflight
  scripts, plus `code/COMMIT`.
- **Generator**: `llmadd_torch/make_jobs_newmani.py [--ext | --ext_cells a,b] [--print|--submit [--wave 1|2] [--only a,b]
  [--nice N] [--redo]] [--status]`.
  - It pins the sha, refuses another worktree file / snapshot / seed_scan.json, and asserts every rendered flag set.
  - It asserts each (out dir, arm, seed) is in exactly one job, and the run count of the design.
  - Append-only: extension runs only in new '-ext-' jobs. A change to, or drop of, a submitted job's file is REFUSED before
    any write. --submit skips jobs already in squeue (and refuses to submit if squeue fails).
  - HOLD switch; SUBMITTED_<job> flags (the job ids, read back by --status).
- **Jobs** re-check the snapshot sha (exit 3) and the GPU type (exit 4), and log the GPU per (re)start to gpu/<job>.txt.
- **Reader**: `runs/newmani/readout.py` (pinned toy_struct; `--selftest`).

## Tests (preflight/; logs next to the scripts; final runs on ea797e5321d3, 2026-10-21; the d0aed90927fd logs are in preflight/superseded_d0aed90927fd/)
**Unit tests, `preflight/test_newmani.py` (test_newmani.log): ALL NEWMANI TESTS PASSED** (CPU, 1 thread).
1. **Group tables.** For 2i, d57, a5 and s5:
   - |G| = 120 / 114 / 60 / 120; associativity over all n^3; a unique identity; inverses; Latin property.
   - Commuting shares 0.0750 / 0.2632 / 0.0833 / 0.0583, equal to the design's 7.5 / 26.3 / 8.3 / 5.8 %.
   - Centres 2 / 1 / 1 / 1; orbits 1,774 / 5,743 / 1,774 / 7,099.
   - At build time: element-order histograms, conjugacy-class counts (9 / 30 / 5 / 7), and the matrix table = an
     independent combinatorial table.
   - 2I: the explicit 600-cell list = the closure of three generators.
2. **Target faithfulness.**
   - Every matrix target is a homomorphism with the stated kernel; the faithful ones are injective.
   - quat4 = the quaternion product q_b q_a.
   - Inequivalent irreps: CKA 0. d57 abel ~ rho1 0.408 (design 0.41).
   - Per-seed check_targets, incl. the new shuf-far assert.
3. **Shape-only targets.**
   - True target: rank r, flat. Sphere: rank r' != r, flat and constant-norm. Line: rank r.
   - The loss is 0 on targets and on permutations of them, and permutation-invariant.
   - Sphere -> true = 1 - sqrt(r / r').
   - Shape-only optimisation leaves the CKA to the assigned target at the random-subspace level (Tier S, no assignment).
4. **Split / twin exclusion**, at all 20 cells (60 splits, box5 positive controls included):
   - every twin move maps train to train and held-out to held-out, cell by cell;
   - closure 1.0;
   - the strict masks exclude central inputs / answers, copy cells, answer 0, both negation twins;
   - box whole-negation-twin share now out of strict: 0.065 / 0.109 / 0.221 (m6 / m12 / m24);
   - test_norewrite equals a by-content second path; the held-out share with a trained division rewrite is 0.09-0.37 at
     m6-m24 (0.6 at f 0.4).
5. **Latin-only deduction.** Equal to runs/toyf84fu/latin.py on 12 S4 toy splits (incl. the non-zero 0.6 shares
   0.026 / 0.061 / 0.069); sound on every new task.
6. **Known answers on training.** grp s4 (question split) = s4: 9 arms, every s4 key, curves exact. box -1:1 = dim:
   10 arms, every dim key. Extra keys are only the documented read-outs (now including test_norewrite).
7. **Flag A/B.**
   - --group / --box change the task.
   - --ckpt_dir: a 3-segment run (seg_end 100, 200, end) = the straight run, bitwise on CPU; --ckpt_every saves; --deadline
     stops with a checkpoint.
   - A foreign checkpoint is refused; the flags without --ckpt_dir are refused.

**Default-path regression, `preflight/regress_default.py` (regress_default.log): REGRESSION PASSED.** Old = d182ce548bcc,
new = ea797e5321d3; CPU, 1 thread.
- 11 training cases, 46 runs: toyshape's 7 plus the wall-2 shape / shapeprof / sphere / line arms (s4, mod torus +
  circle with --mod_twins --shape_log, dim). Every JSON key except code_sha / sec / ms_per_step is equal (key order
  included); stdout equal.
- --check: 7 variants, stdout identical. --summarize: 11 real result dirs byte-identical.
- --summarize_wall: the full real runs/toyshape/out (1,955 runs), byte-identical. Both sides now run on one symlink farm;
  the first d0aed90927fd run had differed only by the printed root path.

**Existing unit tests, `tools/test_toy_struct.py` (full, incl. the toy_grok known answer; test_toy_struct.log):
ALL TESTS PASSED.** The known answer mod = toy_grok.py: 4800 / 4800 values bitwise equal; structure, decision rules, smoke, timing.

**Amendment A1 known answer, `preflight/b1_toyshape.py` (b1_toyshape.log): B1 KNOWN-ANSWER TEST PASSED.**
- Gate off reproduces all 13 published toyshape verdicts.
- Gate on: s4 0.2 / 0.3 become walls; mod unchanged (table in A1).

**Reader, `readout.py --selftest` (readout_selftest.log): READOUT SELFTEST PASSED.**
- A: selection; the specificity flag; a non-wall; a flat bracket; a closed lr bracket; a flat prior lr label (M6).
- B: each way out of the wall; the gated CE component on (late CE 3.9 <= 5.79) alone vetoes, printed.
- C: an incomplete none grid -> PROVISIONAL, never CANDIDATE; a failed positive control -> NOT READ; mixed code_sha ->
  NO VERDICT.
- D: a missing level -> INCOMPLETE.
- E (audit I1): assign at one lr / one seed, assign missing, specrand 1/3 seeds -> every level "INCOMPLETE (prior
  arms ...)" and the selection INCOMPLETE (never select / floor); the positive control with 1/3 seeds -> INCOMPLETE.
- F (A1): late CE 22.3 > gate 5.79 -> the CE component n/a, the cell is a wall; "the ungated pre-A1 rule would have vetoed
  it on CE alone" printed.
- G (box5 positive control): one (frac, config) groks -> OK; none -> FAILED; a config with 1 seed -> INCOMPLETE.
- Second path: the audit's own chk_readout.py on the fixed reader (audit_chk_readout_after_fix.log). All three
  missing-arm cases now read tier L INCOMPLETE with an INCOMPLETE selection; tier S stays "floor" (its arms are complete);
  the 1/3-seed positive control reads INCOMPLETE.

**Generator, `preflight/test_generator.py` (test_generator.log): GENERATOR TESTS PASSED.** Render and --print only (never
sbatch).
- 0: the audit's gen_copy.py scenario on the OLD generator is reproduced: --ext moves 45 of box5-m6-j0's 90 runs out of it.
- 1c: with box5-m6-j0 SUBMITTED, --ext leaves all 26 default job files byte-identical and puts the 315 extension runs in
  15 new '-ext-' jobs.
- 1d: --ext_cells box5-m6 renders only box5-m6-ext-j0.
- 1e: a submitted extension job stays rendered and identical.
- 1f / 1g: a submitted job whose file would change, or which the render would drop -> REFUSED before any write.
- 1h: --status lists a submitted job whose Slurm job (real id 20314607) is COMPLETED with 0/90 runs under "ENDED WITH RUNS
  MISSING".
- 1i: --print --wave 1 --nice 500 skips a job already in the queue, does not re-offer a submitted one, and passes
  --nice=500.

**Seed scan, `preflight/seed_scan.py` (seed_scan.log, seed_scan.json + stage2_seeds.json, sha = the pin).** Stage-1
cells are unchanged vs d0aed90927fd (fractions, seeds, split_sha, n_train; checked). Two box5 positive-control cells were
added, and Stage-2 seeds are listed for all 15 m cells.

**Latin reference, `preflight/latin_ref_dump.py` (latin_ref.json):** unchanged (latin.py's masks for the 12 S4 splits).

**Job smoke, `preflight/smoke_jobs.py` (smoke_jobs.log): SMOKE PASSED.**
- A: the whole job a5-m6-j0 (90 runs, 5 lanes) under bash, steps / 250, a stub nvidia-smi reporting an L40S:
  - HOLD -> exit 0 before any run;
  - run 1 with a 90-s deadline stopped at 5/90 runs with 5 checkpoints;
  - run 2 (= requeue) skipped finished runs, resumed the 5 checkpointed ones (e.g. segment_starts [0, 100]) and finished
    all 90;
  - code_sha ea797e5321d3, lr_now = lr, eval counts checked; no checkpoint left; gpu.txt 2 lines (the HOLD run exits
    before the GPU line).
- B: one run each of box5-m12 expo@always, box5-m24 none, 2i-m12 shape:quat4@always, box5-pos (none lr 1e-2 wd 0.1,
  f 0.6, m 65.9), d57-m12 shape:sphere@always, s5-m6 shuf@always, a5-m24 shape:line@always; decoded inputs printed.
- C: readout.py over the partial smoke root reads every task INCOMPLETE (positive controls and levels incomplete),
  never select / floor / no wall. It prints the A1 CE line, e.g. a5 m6: late CE 5.37 > gate 5.09, n/a; the ungated rule
  would have vetoed it on CE alone.

## Seven checks (experiment-audit)
| check | status | evidence |
|---|---|---|
| i double path | pass | group tables: matrix product = an independent combinatorial table (all 5 groups); 2I list = generator closure; Latin-only deduction = latin.py's sequential algorithm (12 S4 splits, masks + candidate counts); box closure: projection = least squares (asserted per run); box -1:1 = dim and grp s4 = s4 on every key |
| ii intent asserts | pass | build-time group axioms / homomorphisms / kernels / Schur / flatness / sphere rank; split: coverage, no orbit or class across the split, closure 1.0 (+ nullity 3) asserted in every run; Latin soundness asserted per split; checkpoint key (arm, seed, steps, lr, wd, lam, win, split_sha, code_sha) asserted on resume; generator flag sets + one job per run; jobs check sha + GPU |
| iii known answer | pass | grp s4 = s4, box -1:1 = dim (training, every JSON key); latin.py known answer; shape loss 0 on targets / permutations, sphere = 1 - sqrt(r / r'); positive controls in the design (f 0.4) to be read before any task |
| iv anomaly repro | n/a | no results |
| v fresh adversarial reviewer | RUN on d0aed90927fd (2026-10-21, notes/batch1_build/audit_newmani_build.md): 1 blocking (B1) + 5 important + 10 minor, all addressed (revision table at the top). A minimum-mode re-review of the diff (d0aed90927fd -> ea797e5321d3, readout.py, the generator) is recommended in the background after submission | |
| vi trap list | pass (pre-run) | 1 same template / position (decoded); 2 orbit split, no trained twin, strict set, 4x none budget, rising rule; 3 every new flag A/B'd, checkpoint resume bitwise; 4 3 seeds = scout only, Stage 2 fresh seeds; 5 scored at '=' (decoded); 6 one function per metric (upto / late_change / auc / nm_latin), Latin share defined once; 7 every held-out question scored, n per mask logged; 8 default path bitwise (regress), summaries of real dirs and the full toyshape wall reading byte-identical |
| vii decoded inputs side by side | pass | every seed log prints 3 train / 3 held-out questions with ids, names, scored position (smoke_jobs.log) |

## 2026-10-25 additions: code, tests, seven checks (A2, Stage 2, diagnostics)
**Code** (none of it in the repo; the trainer is unchanged at ea797e5321d3, snapshot code/tools; no commit / push):
| file | sha256[:12] | change |
|---|---|---|
| runs/newmani/readout.py | 2182373026af (was c1ce6fd75bd3, kept as preflight/superseded_readout_c1ce6fd75bd3.py) | A2: `--a2`, `--trainable_thr` (sensitivity only), trainable / reference_config / bracket_of / tier_verdict / none_row factored out; the default reading unchanged |
| runs/newmani/readout_stage2.py | a76a41c56204 (A2.1 at registration d89b62f65244; before: b8d23c2135e0, first 275fa8e2e48b; all kept in preflight/superseded_readout_stage2_*.py) | new: the Stage-2 reading (G1-G4 + A1 + A2); after the review: the pin check, a descriptive last-25 % G1, per-seed training fit of the reference / comparator configs, 加速档 text naming the G1 failure |
| runs/newmani/readout_diag.py | 7fdf06c30e21 (first version 4a61d737dfa9, kept as preflight/superseded_readout_diag_4a61d737dfa9.py) | new: the P1 / P2 reading (post hoc); after the review: the P2 wording (twin exposure), the P1 split_sha and device checks |
| llmadd_torch/make_jobs_newmani.py | fd37f308abe8 (was bedf99d75c28, kept as preflight/superseded_make_jobs_newmani_pre_stage2.py; the submitting version 2808cf371997 as preflight/superseded_make_jobs_newmani_2808cf371997.py) | `--stage2` / `--diag` / `--diag_q`, append-only and sticky; out_stage2/, out_diag/; after the review: `--diag_q --submit` needs `--only` (the submitted job files re-render byte-identical, checked) |

**Tests** (preflight/; all CPU; no Stage-2, diagnostic or extension result was read):
- `readout.py --selftest` -> readout_selftest_a2.log: **READOUT SELFTEST PASSED**.
  - The original A-G pass unchanged.
  - New H, A2:
    - a crashed config as A1's best: A1 not a wall on CE alone, A2 a wall;
    - all configs crashed: A1 CANDIDATE, A2 n/a;
    - mixed, with the trainable reference still falling: A2 vetoes on CE where A1 does not;
    - mixed, with a crashed config above the cap: both veto.
  - In every case the A1 text and JSON inside the --a2 output equal the default output.
  - H5: the threshold is consumed by both the function and the CLI (0.5 vs 0.99 change the reference config;
    `--trainable_thr` without `--a2` is refused; the value is printed and the SENSITIVITY label written).
- a2_stage1_posthoc.py -> a2_stage1_posthoc.log: regression, known answer and sensitivity (A2 "Result on the Stage-1
  data").
- a2_threshold.py -> a2_threshold.log: the threshold data.
- `readout_stage2.py --selftest` -> readout_stage2_selftest.log: **STAGE 2 READOUT SELFTEST PASSED** (scenarios 1-15 in
  the Stage 2 section).
- `readout_diag.py --selftest` -> readout_diag_selftest.log: **DIAG READOUT SELFTEST PASSED**:
  - s5 OK while 2i is not shown, with the trigger met;
  - 2i at 1/3 -> trigger not met;
  - every P2 outcome, including INCOMPLETE;
  - a run at another frac -> NO VERDICT.
- test_generator_stage2.py -> test_generator_stage2.log: **GENERATOR STAGE-2 TESTS PASSED**.
  - A: old vs new generator are byte-identical in job files and stdout for the default, --ext and --ext_cells renders
    (27 / 41 / 28 job files), and the default render equals the real job files.
  - B: --stage2 / --diag / --diag_q leave every earlier file byte-identical. The 140 / 12 / 3 runs carry exactly the
    registered flags, seeds and roots.
  - C: sticky once submitted; a submitted file that would change -> REFUSED; --wave 1 never offers them; --only ...
    --nice 500 does; --status lists them.
  - The pre-2026-10-21 test_generator.py cannot be rerun: its part 0 needs the 10-21 auditor's scratch copy, which is
    gone. Its parts 1c-1i are covered by A, since the old generator passed them and the new one renders identically.
- smoke_stage2.py -> smoke_stage2.log: the rendered commands themselves, run on CPU for 200 steps (only --steps / --out
  / --ckpt_dir changed): 9 Stage-2 runs (shuf at all 5 seeds, expo, specrand, two extension configs) plus P1 x 2 and
  P2 x 3. **SMOKE STAGE 2 / DIAG PASSED**:
  - every run exits 0 on code_sha ea797e5321d3;
  - Stage-2 split_sha / n_train / shuf-far CKA equal stage2_seeds.json for all 5 seeds;
  - P1's split_sha equals the Stage-1 positive-control runs';
  - P2 has a question split with n_train 5760 and closure 1.0 for seeds 0-2;
  - lr / wd / lam / lr_now are as rendered, and the decoded inputs are printed.

**Seven checks (minimum mode: ii + v):**
| check | status |
|---|---|
| i double path | partly: A2's Stage-1 re-reading reproduces the results audit's independent chk4 (known answer) |
| ii intent asserts | pass: seeds / split_sha / frac / roots / flags asserted in the generator, the readers and the smoke; A2 threshold consumed and printed |
| iii known answer | pass: chk4 reproduced; Stage-2 and diagnostic readers on synthetic cells with known verdicts |
| iv anomaly repro | – (minimum mode; no results) |
| v fresh adversarial reviewer | run after submission (minimum mode, fresh general-purpose agent): **no blocking item**; 2 important + 4 minor, all addressed or registered ("Review 2026-10-25" below) |
| vi trap list | – (minimum mode) |
| vii decoded inputs | pass: every run logs 3 train / 3 held-out decoded questions; the smoke prints them |

**Review 2026-10-25** (fresh general-purpose agent; minimum mode, after submission, read-only). It found **no blocking
item**: the submitted jobs are exactly the registered runs, it re-derived the Stage-2 seed rule independently, and the
Stage-2 reader matches G1-G4 line by line. Its items and what was done:
| item | severity | done |
|---|---|---|
| A2's reference config can be an oscillating config. "Trainable" reads one point (the last eval), and among the trainable configs of a flat cell the AUC 4B ranking is noise-level (box5 m6 Stage 1: lr 1e-2 wd 3, seeds' final train 0.999 / 0.368 / 1.000, last-25 % mean 0.955-0.980). Harmless while the cell is flat (CE n/a at late CE 15-31; bracket = flat). If the Stage-2 cell is not flat, the edge test / CE could read on such a config and swap PASS and 加速档 by noise. | important | The registered rule is unchanged (0.99 on final_train; changing it is the user's call, possible only before any Stage-2 result is read). The reader now prints, before any data: a descriptive G1 with trainable = last-25 % mean train >= 0.99, the verdict it gives, a "THE REGISTERED AND THE LAST-25 % G1 DIFFER: report both" flag, and the per-seed final / last-25 % mean / min train of the A2, A1, last-25 % reference configs and of the G2 comparators. Self-test 11 covers it. a2_stage1_posthoc.py now also prints reference / bracket changes per threshold. |
| P2's question split gives ~97-98 % of held-out cells a trained twin (reviewer's computation with the trainer's split, seeds 0-2; 52 % have a trained same-answer orbit cell), so a P2 grok is close to twin lookup. | important | The P2 reading is reworded (Diagnostics section, readout_diag.py): >= 2/3 -> "none completes the table when the twins are trained; does NOT show that none learns 2I's group law"; 0/3 -> "the limit is the model family / budget / config, not the split". Never read as "2I is learnable" or "the Stage-1 wall is a split artefact". An orbit-fresh subset would need a trainer change (the user's call). |
| 加速档 with ratio inf when none never groks reads as "none learns, only slower" | minor | The verdict now names the G1 failure and says "none never groks" (self-tests 2, 3c, 11). |
| a blanket `--submit` with `--diag_q` would submit the conditional P2 | minor | Asserted: `--diag_q --submit` needs `--only` (generator test C5). |
| a later default `readout.py` call would overwrite F94's readout.md, now with the extension runs | minor | Frozen read-only copies readout_stage1_prereg_F94.md / .json (sha 5bc67a7f6aca = the published readout.md). |
| readout_stage2.py did not require code_sha = the pin; readout_diag.py did not check the P1 split_sha / device | minor | Both added (self-tests 10b and diag 5). |

**Review of A2.1, 2026-10-25** (fresh general-purpose agent, minimum mode, read-only).
- **No blocking item.** The A2.1 logic matches the text and has no gap: G1 FAIL vs n/a is caught by the category
  comparison. NO VERDICT / INCOMPLETE still return first. Diffed against the old reader on 22 self-test scenarios plus 5
  probes, there is no side effect.
- **Important item, fixed:** the old key "verdict_reg" and the "descriptive" wording in this file could lead someone to
  report a single-measure PASS / 加速档 where the verdict is UNDETERMINED.
- **Minor items:** the verbatim PASS inside the UNDETERMINED text, n/a printed as FAIL, a vacuous assert, and the
  ref_train name are all fixed. The missing bracket-path / n/a tests are added (14, 15). The one-eval window
  difference is kept as registered (known limitation above).

## Caveats (known before any run)
1. **Tier L targets encode the table** (Amendment 1); a Tier-L candidate is at most "given the structure, the model
   learns to use it".
2. **The Latin-only implied share is 0 at every Stage 1 split.** The implied / non-implied split is degenerate here, and
   the strict set differs from all held-out only by the central / zero / copy / negation-twin cells.
3. **n = 3 per arm.** Stage 1 is a selection step, and its paired tests are descriptive. Selection on held-out curves with
   no validation split favours none (12 configs vs 3 lrs).
4. **Multiplicity.** 5 tasks x 3 m x 2 tiers. A single CANDIDATE is not evidence; Stage 2 uses fresh seeds.
5. **Not dose-matched.** Shape loss in [0, 2] vs assign in [0, 1], both lam 1, as toyshape.
6. **Toy only**: one layer, nonce items, closed table, full batch.
7. **Rate model.** It is fitted on mod (T 3, 12.8k-question eval). Box (T 4) is scaled by 4/3; A5's eval table is smaller
   (conservative). --time = 2 x planned.

## Open decisions (the user's)
0. **(2026-10-25, to-do before any rainbow design)** M9 / results audit M4: log the prior-vs-CE gradient share (Amendment A3 section).
1. **Which waves to submit.** Default: the calibrator 2i-pos-j0 first (M7), then wave 1 (box5 + 2i, 52 L40S-h), then wave
   2 (48 L40S-h); or both waves at once.
   Alternatively m 24 only after m 6 / 12 are read (saves 46 L40S-h if m 12 already shows no wall or a candidate).
2. **Rule-2 extension.** Up front for every cell (`--ext`, +~37 L40S-h), or per cell when it reads flat / at an edge
   (`--ext_cells <task>-m<m>`, append-only).
3. **Prior lr grid {1e-3, 3e-3, 1e-2}** (Stage 2's) vs adding 3e-2 for a two-sided bracket at the top (+~30 %).
4. **Stage 2's window arms and the other irreps / quotients / abel** (design table 3.3) are not in Stage 1. Confirm that is
   fine.
5. **Commit.** The worktree file is uncommitted in a PUBLIC repo; the snapshot carries it. Committing is the user's call.
6. **The adversarial re-review (check v)** of the 2026-10-21 diff after submission, minimum mode, in the background.
7. **Amendment A1's margin** (1 nat above log n_out) is fixed now; it is a judgement call (the audit offered alternatives:
   a bounded rank / probability metric, or a drop above the cross-seed SD). Changing it later is post hoc.
8. **M9**: add prior-vs-CE gradient-share logging before Stage 2 (needs a trainer change and a new regression).
   2026-10-25: Stage 2 runs on the unchanged trainer (ea797e5321d3), without M9. M9 stays a precondition before a
   newmani cell feeds a rainbow design (Known limitations).
9. **(2026-10-25) A2 item 5, the G2 / G3 comparator.** It is NOT restricted to trainable configs (the conservative
   choice). The coordinator first recommended restricting, then **confirmed not restricting (2026-10-25)**: closed.
10. **(2026-10-25) P2 trigger.** The coordinator decides once readout_diag.py prints "P2 trigger: met".
11. **(2026-10-25) Commit** of the new / changed files is still the user's (Open decision 5):
    - readout.py, readout_stage2.py, readout_diag.py and the preflight scripts live in runs/ (not the repo);
    - make_jobs_newmani.py lives in llmadd_torch (not the repo);
    - the trainer is unchanged.

### Coordinator note on A3 / A3.1 (2026-10-25, after the fresh review of readout_a3.py 9a6a061f706a; no A3 result read by anyone)
Review: notes/batch1_build/audit_newmani_a31_reader.md. Verdict: usable for the read-out; no blocking or important item.
- m1 (wording rule, text only; the reader is NOT changed, because A3 JSONs already exist): every A3 verdict other than
  WALL, including UNDETERMINED with one side OPEN and the other FAIL, is written as "not a rescue" in FINDINGS and the
  paper. Only WALL (+ A3-long qualifiers if any) lets F95's PASS be called a Tier-L rescue.
- m2 (timeline correction): the A3.1 code was saved at 12:09:06 and the A3.1 work finished at 12:10:32, not "~12:20".
  At 12:09:06, out_a3 already had 5 JSONs. They were not opened. readout_a3.md did not exist until the coordinator's
  first read-out.
- m3-m6 (tie-break not registered, key names, reporting the reference config when only the bracket fails, wording):
  listed in the review. The coordinator reports the reference config and the per-config table beside the verdict.
