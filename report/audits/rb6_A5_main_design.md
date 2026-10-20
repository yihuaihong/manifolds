## Amendment A5 (2026-10-19, the MAIN design, after the pilot results and BEFORE any main run)

USER DECISIONS (binding, 2026-10-19, after the pilot 400-419): K = 2 domain lines per row (the pre-registered gate's
K*), lr 1e-3, window (0.05, 0.4), ctx site, lam_rule each, **lambda 1 per active prior** (lambda 5 rejected: the priors
made ~2/3 of every update; "the manifold loss is only an auxiliary loss"), seeds 0-3; one all6 arm at **lambda 0.3**
per prior; every arm at its own best lr: none's best lr at the 2% 53M setting is 8e-3 (Burst pt2b bracket) -> a
**none@8e-3** arm; the prior arms' bracket is flat at 1e-3 vs 2e-3 (70.2 vs 69.9), so they stay at 1e-3. Coordinator
additions (same day): a confirm block (seeds 4-7 of the two headline cells; first optional, ON by the decision below), the strict dim read-out, the
heuristic floor line, the per-seed ordering fix of the summary (below).
USER DECISIONS after the fresh build audit (2026-10-19, binding, pre-registered before any main run): (a) a headline claim
must hold in BOTH F1 (all6-lam1 vs none@1e-3) and F1b (all6-lam1 vs none@8e-3); (b) the confirm block is ON
(make_rb6m_tasks.CONFIRM = True: tasks 470-477 are generated); (c) "robust" (all seeds agree) is reported, not a decision
rule (as the earlier F-rules). Build-audit fixes (report notes/batch1_build/audit_rb6m_build.md; no blocker found): the
pilot's hypotheses are tested on the fresh seeds 2-7 only (FE); every verdict is seed-level (the item bootstrap is
descriptive); the winner's curse of none@8e-3 is recorded; F8's +4e-3 bpb flag is implemented; tier E has no heuristic
line; the read-out command and the refusal of a pilot mix; dropped seeds fail loudly; re-run numbers reserved; library
versions recorded per run; the lambda actually used is asserted per run.
DECISIONS on the three remaining questions (2026-10-19, binding): (1) none-lr8e-3 gets seeds 4-7 as tasks 490-493 (written
with the confirm block; 478-489 stay reserved / toyb), so F1b and FE have the fresh seeds 2-7 (n = 6) against none@8e-3
too and F1b has confirm seeds; (2) no lambda-5 arm: lambda 1 was chosen on the user's principle that the manifold prior is
an auxiliary loss (lambda 5's priors made ~2/3 of every update), NOT on domain gains, so lambda 1 vs 5 is not a pilot
hypothesis and is not tested; (3) a CKA-only FAIL of a shuffled prior is a finding and the run stays valid (A5 and the
check text say so).

### Code
| file | sha256[:12] | status |
|---|---|---|
| tools/pretrain_range.py | cc984148b15a (pinned c434d29) | unchanged |
| tools/toy_struct.py | 6248613ac6d2 (pinned c434d29) | unchanged (HEAD's copy is b74a78cb5472: the summary re-implements the two tables it needs and asserts them against every run) |
| tools/make_rb6_tasks.py | 7f689a95f000 | unchanged; its 18 renderings are asserted byte-identical to tasks/400-417 after undoing its one later edit (9cbfca7, geo tolerance, "future renders only"), and 418 / 419 = 408 / 409 byte for byte |
| tools/make_rb6m_tasks.py | 941a387a41a3 | NEW: the 40 main tasks 430-469 + the confirm blocks 470-477 and 490-493 (ON) = 52 tasks (written to tasks/) |
| tools/rainbow6_summary.py | c32891cace9a -> 70758fe86995 | the A5 read-out (below); pilot output unchanged except the per-seed ordering fix |

### Arms (K = 2; every arm sees the same data per seed; 53M, 3e8 tokens, bs 32 x 512, 18,310 steps)
| arm | trainer flags ($ARM; plus --lr) | active priors (lambda) | tag | seeds | tasks |
|---|---|---|---|---|---|
| none | --target none (lr 1e-3) | - | rb6pb-53m-2pct-k2-none (REUSED pilot cell) | 0, 1 = pilot 401, 404; 2, 3 new | 430, 431 |
| all6-lam1 ("all6") | --target helix3 W --prior_domains numbers,months,weekdays,clock,s4,dim --lam 1 | all 6 (1.0) | rb6pb-53m-2pct-k2-all6-lam1 (REUSED) | 0, 1 = pilot 419, 415; 2, 3 new | 432, 433 |
| all6-shuf | --target helix3_shuf W --prior_domains (all 6) --prior_shuf (all 6) --lam 1 | all 6 (1.0), shuffled targets | rb6mb-53m-2pct-k2-all6-shuf | 0-3 | 434-437 |
| only-numbers | --target helix3 W --prior_domains numbers --lam 1 | numbers (1.0) | rb6mb-53m-2pct-k2-only-numbers | 0-3 | 438, 444, 450, 456 |
| only-months / weekdays / clock / s4 / dim | --target none W --prior_domains <d> --lam 1 | d (1.0) | rb6mb-53m-2pct-k2-only-<d> | 0-3 | 439-443, 445-449, 451-455, 457-461 |
| all6-lam0.3 | as all6-lam1 with --lam 0.3 | all 6 (0.3) | rb6mb-53m-2pct-k2-all6-lam0.3 | 0-3 | 462-465 |
| none-lr8e-3 | --target none, --lr 8e-3 | - | rb6mb-53m-2pct-k2-none-lr8e-3 | 0-3; 4-7 (decision 1) | 466-469; 490-493 |
| confirm block (ON) | none / all6-lam1 as above | as above | the pilot tags | 4-7 | 470-477 |
W = --schedule window --window 0.05,0.4 --site ctx --lam_rule each (the pilot's). none / none-lr8e-3 pass no prior flag
(lam_rule norm, lam 5.0: inert, lam_domains all 0.0 asserted). Single-prior arms use lam_rule each with one prior (=
norm). Shuffled targets (read in the pinned trainer): numbers --target helix3_shuf = helix3 of default_rng(124 + seed)
.permutation(1000) (a uniform permutation, not a far one: CKA to helix3 0.008-0.011 at seeds 0-3); the five domains --prior_shuf = the
table rows permuted by far_perm on stream 4100 children 6-10 (first permutation with max CKA to the references < 0.15,
else the farthest of 3000; references: the true table, and for s4 also R_sign; asserted < 0.15 except s4). The shuffled
arm's DOSE = the same window (steps 916-7323) and the same lambda 1 per prior as all6-lam1. Per seed, the CKA of the
shuffled table to the true one (dry logs, 'domains:' line; the same in every arm of a seed):
| seed | numbers | months | weekdays | clock | s4 (vs R) | dim |
|---|---|---|---|---|---|---|
| 0 | 0.008 | 0.132 | 0.117 | 0.010 | 0.344 | 0.102 |
| 1 | 0.008 | 0.074 | -0.000 | 0.126 | 0.340 | 0.041 |
| 2 | 0.011 | 0.131 | 0.143 | 0.050 | 0.344 | 0.082 |
| 3 | 0.009 | 0.083 | 0.143 | 0.133 | 0.347 | 0.123 |
s4's shuffle stays at CKA ~0.34 to R (no permutation of the 24 rotations gets below 0.15 against R and R_sign): all6-shuf
is a partial control for s4 (A1 finding 8).

Reuse: the pilot's K2 none and all6-lam1 runs of seeds 0, 1 ran at the same PIN with exactly these arguments (the
generator asserts the reused cells' $ARM = the pilot's ARM string and their tag = the pilot's tag); only seeds 2, 3 are
new, under the pilot's tags, so seeds 0-3 form one cell in the summary (cells are keyed by (K, arm); the tag phase pb / mb
is not part of the key; check() still refuses two runs of one (arm, K, seed)).

Numbering = submission order (tick submits ready tasks in file-name order, MAX_GPU_JOBS 12; asserted at render):
430-433 none / all6-lam1 seeds 2, 3 (complete the core all6 - none contrast at n = 4: every other contrast pairs against
these cells) -> 434-437 all6-shuf s0-s3 (the specificity control of any all6 effect) -> 438-461 the six single-prior
arms SEED-MAJOR (438-443 = seed 0 of numbers, months, weekdays, clock, s4, dim, then seeds 1, 2, 3: after every 6 tasks
each single arm has the same seeds, so a partial read-out stays balanced and paired) -> 462-465 all6-lam0.3 -> 466-469
none-lr8e-3 (the user's order). First wave (12) = 430-441 if no other GPU task is ready.

Confirm blocks (ON, decisions 2026-10-19 (b) and (1); make_rb6m_tasks.CONFIRM = True, --no_confirm leaves both out):
470-477 = (none, all6-lam1) x seeds 4-7, seed-major (every finished pair is paired), and 490-493 = none-lr8e-3 seeds 4-7.
Seeds 0-3 decide, seeds 4-7 confirm (readout_rules.MAIN / CONFIRM); seeds 2-7 are the fresh seeds that decide the pilot's
tier-E hypothesis (FE, below), now with n = 6 against none and against none@8e-3.
Re-run numbers (reserved in the generator, accepted by --check): 478, 479, 483-489 (480-482 = the toyb tasks pushed
2026-10-19). A re-run is the failed task's rendering under a new number (same tag, seed, args); in the read-out command its
directory replaces the failed one.
Tick order: the tick submits ready tasks in file-name order (MAX_GPU_JOBS 12). origin/main (2026-10-19, after this
branch's base) has 8 unsubmitted GPU tasks below 430 (386-393, pt2main text-site confirm seeds, limits 10:00 / 05:30) and
the toyb tasks 480-482 (limits 01:00); nothing was in flight at the 07:12Z heartbeat and SENTINEL is off, so after the
push and one tick kick: 386-393 + 430-433 (wave 1), then 434-... as slots free, 480-482 after 477, 490-493 last.

### Intent checks of every task (make_rb6m_tasks.py; the pilot's, generalised)
Before training (--dry_inputs at PIN, exit 3): data sha256; the model line; the numbers target (none: no 'target
features' line; helix3 / helix3_shuf: (1000, 10)); for EACH of the six domains in the 'domains:' line its "prior" / "shuf"
/ "lam" (active: true / lambda, inactive: false / 0.0, shuf true exactly in all6-shuf) and the numbers "target"; numbers
first; the content scan; the DRY_GREP sha256 + count = Torch's for (K 2, seed); the pinned trainer sha.
After training (CHECK_PY on the run JSON, exit 4): all 45 args = intent (incl. lr, lam, target, prior_domains, prior_shuf);
code / dep shas; run facts; lam_domains = intent; domains.<d>.prior / shuf / lam = intent; finished; data_sha after step 0
and step-0 lm (1e-3) = Torch's; layout (numbers first, token share 1.9-2.4%, slots 1/2 +- 0.05); splits = Torch's; the
held-out-format sets. Prior arms: prior_events [[916, 1], [7324, 0]], on_frac; each ACTIVE prior on exactly at the
on-evals, applied on >= 95% of the on-steps, geo_<d> logged at >= 95% of the evals, a weighted gradient at >= 90% of the
on-evals; each INACTIVE prior never on / never applied / never logged / no weighted gradient (numbers inactive: geo == 0 at
every eval, no gnorm_prior_numbers); geo > GEO_TOL = -1e-3 (pilot: geo_weekdays -3.2e-5 in 411 and -2.4e-4 in 416 made
both exit rc 4 under the strict "> 0": valid runs; the pilot generator's later -1e-4 would still fail 416) and 0 <
grad_prior < inf at the on-evals; the CKA to the target ACTUALLY USED rises by >= 0.1 from step 0 to the window's last
eval for every active prior: true target -> the item-state cka_<d> / cka_ops_helix3 (as the pilot), shuffled target ->
1 - geo_<d> (resp. 1 - geo) on the training batch (the prior's own value and the only logged CKA to the shuffled table;
logged at step 0 too), with the rise of the CKA to the true target in the info. Decision (3): a shuffled prior whose ONLY
FAIL is this rise (< 0.1) is a FINDING (how fittable a permuted target is) and the run stays VALID: its results are
published (the tick publishes on any exit code) and read like any other; the check's own text says so for the shuffled
arm. Expected: the shuffled NUMBERS target, a uniform permutation of 0..999, was fitted far more slowly than the true helix
in the tiny smoke (+0.047 vs the five shuffled domain targets +0.20..+0.57). none arms: no prior ever on, geo == 0, no
geo_<d> / gnorm_prior_<d> keys. none-lr8e-3 = none except args.lr. The lambda actually used (build audit): for every
unshuffled active prior gnorm_prior_<d> / gnorm_true_<d> = lambda (rel. 1e-3) at every on-eval (the trainer scales the same
gradient vector; pilot 419 / 415: 1.0000 at all 26 on-evals for all 6 priors, 418: 5.0000). The info line adds the
gradient split and the library versions of the env311 python (python, torch, transformers, numpy, tokenizers).

### Torch reference values (K 2; pinned tools = git archive c434d29 tools; Torch data = DATA by sha256)
Method (scratchpad rb6m/refs/: run_refs.py, step0.py, queue*.sh, out/*.log + *.json): --dry_inputs of the pinned trainer
with the task's own arguments and tag, DRY = sha256 + count of the DRY_GREP lines through the task's own bash pipeline;
step 0 = the pinned trainer's own main() up to its step-0 training forward (CPU, 2 threads, bf16 autocast), the HF
causal-LM loss of that forward computed with the same operations but the vocabulary log-softmax in row chunks under
no_grad (the full fp32 logits + autograd need ~17 GB; this host has < 14 GB free) and reduced by the same nll_loss kernel,
then stopped; data_sha after step 0 from the trainer's own dyn. FIRST reproduced the recorded pilot values exactly
(make_rb6_tasks.DRY / LM0 / DSHA0 / DOMS, task tag rb6pb-53m-2pct-k2-none):
| (K 2, seed) | DRY (sha256, lines) | step-0 lm | data_sha | DOMS |
|---|---|---|---|---|
| seed 0 | fe6a8bba...b632512, 88 = recorded | 11.614818572998047 = recorded, bitwise (Burst GPU run 401: 11.614827, +8.6e-6) | 28c9fff8ef37e82f = recorded | = recorded |
| seed 1 | b4aabe5c...5921800, 88 = recorded | 11.603224754333496 = recorded, bitwise | dfac64fb1b4a4f36 = recorded | = recorded |
New values (all 88 DRY lines; the DOMS split facts are in make_rb6m_tasks.DOMS):
| (K 2, seed) | DRY sha256 | step-0 lm | data_sha after step 0 |
|---|---|---|---|
| 2 | 964d1b8b61a15305ed080f7bf994f7792f929018dd0bb539d8543130e43cdfdf | 11.570906639099121 | 36b63d307358a0a4 |
| 3 | ea21c901acad5eca30cf8477ec32e533475804ee59f58024c2a59e92b2fd015e | 11.604287147521973 | df68c5608eedaa9a |
| 4 (confirm) | 1f07da985f7d7acf2186a6e5ff45d37d955105a7abd870f94c9a019121ddb471 | 11.58791446685791 | 9451e7743f0607ed |
| 5 (confirm) | b0fe6b14623ddf6345562f17d7e9348902f1f5a1968e9e991fea6ef72b80bd04 | 11.569442749023438 | 99801f9e457357aa |
| 6 (confirm) | 7db5d83aefd58e6303eff4207106c81eedc25f9e5a6c7ffab0b76c9e1ae83e18 | 11.584881782531738 | 20b49004bf55264c |
| 7 (confirm) | 4465f4f1ae6a72290cab8ac0c414cdd6df0e4788445b5c15600a984f6a492e95 | 11.593488693237305 | 83ce379a1c8bd6cd |
Arm independence (the same values in every arm of a seed, so one reference per seed): DRY and DOMS identical over all 10
computed arms at seeds 0, 1 (every arm but all6-lam1, whose pilot runs exist) and all 11 arms at seeds 2, 3, over none /
all6-lam1 / none-lr8e-3 at seeds 4-7; step-0 lm and data_sha bitwise identical over none / only-s4 / all6-shuf / none-lr8e-3 (seed 2),
none / all6-shuf / only-numbers (seed 3), none / all6-lam1 / none-lr8e-3 (seed 4). The '  eval ' lines (every domain eval row) are
identical across arms too. Clock / s4 / dim split sizes differ by seed (e.g. clock n_train 475-482, dim n_train 342 at
seeds 5, 7), months / weekdays are the same at every seed.

### Read-out (pre-registered; tools/rainbow6_summary.py A5 sections; printed when a main-design arm is present)
Read-out command (Burst repo root, after the 52 tasks; a re-run 478-479 / 483-489 REPLACES its failed task's directory in
the list, because the summary refuses two runs of one (arm, K, seed)):
```
python tools/rainbow6_summary.py results/401-rb6pb-* results/404-rb6pb-* results/415-rb6pb-* results/419-rb6pb-* \
  results/4[3-6][0-9]-rb6*-* results/47[0-7]-rb6pb-* results/49[0-3]-rb6mb-* \
  --ref 'results/35[0-3]-pt2b-lr1e-3-none-s?' --ref 'results/35[0-3]-pt2b-lr8e-3-none-s?' \
  --ref 'results/35[0-3]-pt2b-lr1e-3-win-ctx-s?' --json rb6m_summary.json
```
Only these directories: the summary REFUSES main-design runs mixed with the pilot's K 1 / 5 cells or with its lambda-5
arm 'all6' (tasks 414 / 418; the main design's all6 is all6-lam1), and with --main_readout it excludes them from the A5
sections with a printed note. Before every task has finished, add --allow_partial (verdicts PROVISIONAL; an incomplete
family is then a warning instead of a failure). The reused cells' seeds 2-7 carry the pilot tag prefix rb6pb- (A5 reuse):
the summary prints which seeds of none / all6-lam1 are the pilot's (0, 1) and which are main-design runs (2-7).

Unit and pairing: K = 2; seeds are the independent units; paired by seed (the same data per seed in every arm: data_sha
at every eval and step-0 lm asserted by check()). Seed groups: 0-3 = DECIDE (every arm), 4-7 = CONFIRM (none /
all6-lam1 only), 2-7 = FRESH (never seen by the pilot; seeds 0, 1 are the pilot's discovery seeds), 0-7 = pooled.
Main read-out per domain (from the pilot gate at K* = 2):
| domain | read-out | statistic | why |
|---|---|---|---|
| numbers | 6-bucket held-out mean | area (100 x mean over all 75 evals) | the main experiment's metric (pt999 / pt2b); final3 is at ceiling for prior arms |
| months | tier E (months_heldE) | final3 | MEASURABLE on tier E only; tier S is at the floor in every pilot arm |
| weekdays | tier E (weekdays_heldE) | final3 | as months |
| clock | tier S with wrap (clock_held_wrap) | final3 | MEASURABLE |
| s4 | held, fresh, non-copy (s4_held_fresh_nonid) | final3 | FLOOR for none -> one-sided against a floor baseline |
| dim | **STRICT** (dim_held_strict, new) | final3 | MEASURABLE; the strict set replaces no_negtwin_nonid (below) |
Descriptive only (never a claim): months / weekdays tier S, clock tier E, dim no_negtwin_nonid (the pilot's set, beside
the strict one), numbers final3 and bucket groups (A3), all areas of the domains, seen sets, held-out formats.

Strict dim set (audit 2026-10-19: toy_struct.negtwin_mask removes only (a, op', -b); 62 / 120 (s0) and 69 / 130 (s1)
read-out questions have another same-answer twin in training, used by none: K2 none final3 28.9 on that half vs 17.3 on
the clean half): no_negtwin_nonid minus every question with ANY trained same-answer twin among the 15 token-level
transforms {inputs in order or swapped} x {negate a or not} x {negate b or not} x {times or per} (the question itself
excluded). The algebraic identities of the exponent group are asserted to be among them with the same answer for every
held question (times a*b: b*a, a/(-b), b/(-a); per a/b: a*(-b), (-b)*a, (-b)/(-a)); twins of the question's own relation
class are asserted untrained (class split). Computed in the summary from the per-question hits (hits_dim_held; the
trainer is unchanged), with the s4 / dim tables re-implemented and asserted against every run's classes and masks. Sizes
(the trainer's split re-run with the pinned toy_struct, tests/strict_sizes.py; seeds 0, 1 = the pilot runs' values):
| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| no_negtwin_nonid | 120 | 130 | 142 | 132 | 128 | 156 | 150 | 130 |
| strict | 58 | 61 | 79 | 46 | 64 | 94 | 73 | 65 |
| chance on strict (%) | 3.7 | 9.8 | 3.7 | 8.7 | 6.2 | 3.7 | 3.7 | 12.3 |
| heuristic on strict (%) | 15.7 | 14.5 | 15.7 | 15.4 | 14.2 | 14.4 | 15.0 | 15.3 |
Every dropped twin is a swap+negate form (b/(-a) of a times question, (-b)/(-a) of a per question). Seed 3's
strict set has 46 questions (the trainer's own minimum for a read-out set was 50): its item noise is larger, nothing
else changes. Pilot gate on both sets (none, 2 seeds; the pilot's 90% item bootstrap): K1 no_negtwin_nonid 28.0 [23.2,
33.5] MEASURABLE, strict 19.0 [12.4, 26.0] MEASURABLE (floor 11.8); K2 23.4 [19.9, 27.1] vs strict 17.3 [12.9, 21.3], both
MEASURABLE; K5 3.7 / 0.8, both FLOOR: the verdicts are unchanged (the audit's own bootstrap: K1 [12.5, 26.0], K2 [12.9,
21.3]). At the seed level (A5's rule, n = 2) K2's strict and no_negtwin_nonid are both BORDERLINE (seeds 1.7 / 32.8 and
7.2 / 39.5): the dim read-out must be re-read at n = 4.

Seed-level inference everywhere (build audit item 2): the A1 item bootstrap resamples questions WITHIN seeds and never
seeds, so its CI has no seed-to-seed variance (pilot K2 none: dim strict seeds 1.7 / 32.8 but CI [12.9, 21.3]; clock wrap
14.7 / 29.3 vs [19.3, 24.9]; every CI excludes both seeds). In the A5 sections every verdict is SEED-LEVEL:
- gate (none, none@8e-3; seeds 0-3, and with confirm seeds also 2-7, 4-7, 0-7 for none): the mean over seeds of final3
  and its 90% Student t interval across seeds (df n - 1) vs [chance + 5, 90] (A1 rule 1's bounds): MEASURABLE / FLOOR /
  CEILING / BORDERLINE as A1. The item-bootstrap CI and its verdict are printed beside, labelled DESCRIPTIVE. (Pilot K2
  none at the seed level, n = 2: clock wrap, dim strict and dim no_negtwin_nonid become BORDERLINE; the pilot gate itself
  is unchanged, it was pre-registered.)
- heuristic line: per seed final3 - heuristic, its 90% t interval: "above the heuristic: yes" iff the lower bound > 0,
  "no" iff the upper bound < 0, else undecided; plus how many seeds are above. Tier E: "n/a" (copying the trained
  equivalent shift (i, k -+ n) answers every tier-E question by construction; the elimination heuristic scores 0, so a
  comparison with it says nothing).
- matrix: the per-arm mean over seeds 0-3 with its 90% seed-level t interval; F = its upper bound < floor, H = above
  the heuristic at the seed level.
- families: paired tests across seeds (below), as before.
With n = 4 (df 3) the t intervals are wide (t_0.95 = 2.35); with n = 2 (df 1, t = 6.31) almost nothing is decided: that
is the honest resolution of 2-4 seeds.

Heuristic line (audit 2026-10-19; descriptive, next to every floor verdict): the floor chance + 5 lies below the
answer-elimination heuristic (answer uniformly among the answers not given by a trained question with the same start item
/ row or column (s4, a Latin square) / first input and operation (dim)); on the pilot seeds 0 / 1: months S 18.1 / 18.2,
weekdays S 27.7 / 28.2, clock S wrap 9.8 / 9.7, s4 26.8 / 26.9, dim no_negtwin_nonid 15.4 / 14.9, dim strict 15.7 / 14.5;
tier E: no heuristic line ("n/a"; copying the trained equivalent shift answers it by construction). Reproduced exactly by
the summary's own implementation (the audit's elim.py values). Every gate line prints "above the heuristic: yes / no /
undecided" at the SEED level (below) and the number of seeds above. A floor-baseline domain (s4) that a prior arm lifts
above chance + 5 but not above the heuristic has learned at most the Latin-square constraint, not the group.

Inference (every test): per-seed difference d_s = arm A - arm B of the metric on the family's decision seeds; paired t
(df = n - 1), two-sided unless stated; exact sign-flip p over all 2^n sign patterns (floor 2 / 2^n two-sided, 1 / 2^n
one-sided: .125 / .0625 at n = 4, so it can never reach .05 there: reported, not a decision rule); Benjamini-Hochberg at
q = 0.05 within each pre-declared family (and seed group). SIGNIFICANT = BH-adjusted t p < 0.05 on the decision seeds.
"robust" = all seeds in the direction of the mean: REPORTED, NOT a decision rule (user decision 2026-10-19 (c), as the
earlier F-rules). Identical differences in every seed (t undefined): p = its limit (0 for a nonzero difference in the
tested direction), flagged. Mean, per-seed values (seed order) and the 95% CI are printed for every test; an n.s. result
is "not shown", never "no effect". A seed whose metric is missing (offline bpb: no offline_lm.json record) is printed as
DROPPED, and a dropped seed or a pre-declared test without its arms / seeds FAILS the read-out (the BH m and n would
change silently) unless --allow_partial (or --main_readout), where it is a printed warning.

Discovery seeds (build audit item 1, binding): seeds 0, 1 of none / all6-lam1 are the pilot's runs, on which the tier-E
harm hypothesis was formed and K* = 2 was chosen (none's K2 tier E was high there: months E 26.7 / 50.6 / 27.2 at K 1 / 2 /
5). Any hypothesis formed on the pilot is tested on the FRESH seeds 2-7 only:
- the tier-E harm of all6 (months / weekdays): family FE, decided on seeds 2-7 (n = 6 against none and against
  none@8e-3), with seeds 0-3 and 0-7 printed beside (descriptive);
- lambda 1 is NOT a pilot hypothesis (decision 2): it was chosen on the user's principle that the manifold loss is an
  auxiliary loss (at lambda 5 the priors made ~2/3 of every update, the global clip cut the LM step to ~0.6), not on the
  domain read-outs; lambda 1 vs 5 is not tested. The in-design dose contrast F5 (lambda 0.3 vs 1) is also printed on
  seeds 2-3 (its all6-lam1 seeds 0, 1 are pilot runs; n = 2, descriptive);
- K* = 2 is a design choice, not a claim.
For F1 / F1b the fresh seeds 2-7 and the pooled 0-7 are printed beside the decision on seeds 0-3 (descriptive).

Families (pre-declared; metric per domain as in the table; s4 one-sided only against a floor baseline):
| family | contrast | tests | decision seeds | sided |
|---|---|---|---|---|
| F1 joint own-domain effect | all6-lam1 - none | numbers, clock S wrap, s4, dim strict (4) | 0-3 (+ 2-7, 0-7 descriptive; confirm 4-7) | s4 one-sided (greater), the rest two-sided |
| F1b each arm at its own best lr | all6-lam1 - none@8e-3 | the same 4 | 0-3 (+ 2-7, 0-7 descriptive; confirm 4-7) | s4 one-sided iff none@8e-3's SEED-LEVEL s4 gate (seeds 0-3) is FLOOR, else two-sided |
| FE the pilot's tier-E hypothesis | all6-lam1 - none, all6-lam1 - none@8e-3 | months E, weekdays E (4) | **2-7** (+ 0-3, 0-7 descriptive) | two-sided |
| F2 single vs none | only-d - none on d | 6 | 0-3 | s4 one-sided |
| F3 stacking | all6-lam1 - only-d on d | 6 | 0-3 | two-sided (> 0 synergy, < 0 interference) |
| F4 geometry specificity | all6-lam1 - all6-shuf | the 6 | 0-3 | two-sided |
| F5 dose | all6-lam0.3 - all6-lam1 | offline bpb, the 6, seen (mean of the 5 domains' seen final3) (8) | 0-3 (+ 2-3 descriptive) | two-sided |
| F6 baseline lr | none@8e-3 - none | the 6 + offline bpb (7) | 0-3 (confirm 4-7: the seeds that did not select 8e-3) | two-sided |
| F7 unrelated-task harm | only-d - none on every OTHER domain e | 30 | 0-3 | two-sided |
| F8 LM cost | prior arm - none, offline bpb (full val, offline_lm.py) x 1000 | all6-lam1, all6-lam0.3, 6 singles, all6-shuf (9) | 0-3 | two-sided; LM COST FLAG printed when the mean > +4 (x1e-3, A1 R5) whatever p says |
(Before the audit F1 / F1b had 6 tests incl. months / weekdays tier E; those moved to FE so that each hypothesis is tested
once, on the right seeds, and the pilot-biased tier-E p-values do not enter F1's BH.)

HEADLINE claims (user decision 2026-10-19 (a), binding): a headline claim about domain d must be SIGNIFICANT in BOTH F1
(all6-lam1 - none@1e-3) and F1b (all6-lam1 - none@8e-3), with the same sign:
- numbers / clock / s4 / dim: decided on seeds 0-3 (F1 and F1b); then the confirm seeds 4-7 of BOTH contrasts: each is
  "confirmed" iff its seeds-4-7 mean has the same sign and >= 3/4 of them point that way (readout_rules.survives); a
  headline is confirmed iff both are, WITHDRAWN ("not a headline") if either disagrees, "unconfirmed" while one is missing.
- months / weekdays tier E: decided on the fresh seeds 2-7 (FE), both contrasts (n = 6 each); no separate confirmation.
The summary prints one HEADLINE line per domain (headline gain / harm, confirmed / unconfirmed / withdrawn, or not a
headline with the F1 / F1b status and n).

Winner's curse of none@8e-3 (build audit item 4): lr 8e-3 was selected on the pt2b none arm's seeds 0-3 (area 20.1, 31.0,
20.0, 21.2: mean 23.1, vs 13.8 on its seeds 4-7: 13.9, 21.9, 8.5, 10.8). rb6m's none-lr8e-3 seeds 0-3 share the init and
the numbers lines with those runs (A2 identity: the numbers eval forwards are byte-identical to the default path's), so
part of that luck carries over: F1b's numbers difference is biased TOWARD none@8e-3 (conservative for an all6 gain
claim), and F6's numbers difference (none@8e-3 - none) is biased upward. The summary prints this note next to every
numbers row that involves none@8e-3; the best lr was also chosen without domain lines and on numbers only. The none@8e-3
seeds 4-7 (490-493, decision 1) did not take part in the selection: F6's confirm seeds 4-7 and the F1b / FE fresh seeds
4-7 are free of it (seeds 2, 3 are not).

Non-inferiority (F5, "the gain is kept at lambda 0.3"): per domain, the one-sided 90% lower bound (t, df 3) of
all6-lam0.3 - all6-lam1 > -3 area points (numbers) / -5 final3 points (domains); read only where F1 (FE for tier E) shows a
significant gain.
Gradient split (descriptive, every prior arm and seed): window medians at the on-evals of the LM share grad_lm / T (T =
the pre-clip total), the clip factor 1 / max(1, T), grad_prior / grad_lm and the share of on-evals with grad_prior >
grad_lm; FLAG if the median grad_prior / grad_lm > 1 in any seed (user: the prior is auxiliary). Pilot K2: lambda 5 2.61
(FLAG, c 0.61-0.67), lambda 1 0.53-0.55 (c 1.00, LM share 0.87-0.88).
Library versions: every task's check records python / torch / transformers / numpy / tokenizers of the env311 python in
check.json info.versions; the summary prints the distinct sets and WARNS if runs differ (the pilot's checks predate this:
"not recorded"; env311 at 331: torch 2.10.0+cu128, transformers 4.57.6).

Pre-registered readings:
- F1 > 0 = "the joint prior helps d (at lr 1e-3)"; F1 < 0 = "it harms d"; a HEADLINE needs F1b too (above).
- Tier-E harm (the pilot's: all6 lambda 5 months about -28 / weekdays about -29; lambda 1 about -8; both seeds the same
  sign) is decided by FE on seeds 2-7. WHOSE harm: the own prior iff F2 (only-d - none) is significantly negative on d;
  the other priors iff F3 (all6-lam1 - only-d) is significantly negative and / or F7 shows a single prior e != d harming d
  (F2 / F3 / F7 decide on seeds 0-3 and contain the reused none seeds 0, 1: an attribution is read as exploratory);
  neither -> not attributable at n = 4. Tier E measures transfer from the trained equivalent shift k +- n of the same item
  (a memory transfer), not circle geometry (tier S is at the floor in every arm): no claim about the circle.
- s4 (none at the floor): F1 / F2 significant = "the prior makes s4 learnable at lr 1e-3" only if the arm's SEED-LEVEL 90%
  lower bound also clears the floor (9.2); the seed-level heuristic line (26.8) says whether it is above the elimination
  heuristic (above the floor but not above the heuristic = at most the Latin-square constraint, not the group). If
  none@8e-3 is not FLOOR on s4 at the seed level, F1b's s4 test is two-sided and the claim is stated "at the prior arms'
  lr; the best-lr baseline also learns s4".
- F1b: a gain significant in F1 but not in F1b is "lr-dependent (not shown against the best-lr baseline)" and not a
  headline; mind the winner's curse (it favours none@8e-3 on numbers).
- F3: > 0 synergy, < 0 interference, n.s. not shown. F4: > 0 = the true geometry matters beyond an auxiliary CKA loss of
  the same dose on the same tokens; < 0 = the shuffled target does better. Caveat: s4's shuffled table keeps CKA ~0.34 to
  the true R (partial control, A1 finding 8); numbers' shuffle is a uniform permutation.
- F5: "less harm at lambda 0.3" iff bpb significantly lower or tier E significantly higher; "gain kept" per domain by the
  non-inferiority bound; the choice of lambda is the user's.
- F6 + the seed-level gate table: a FLOOR -> MEASURABLE change at 8e-3 qualifies the corresponding prior claim (above);
  the F6 tests describe what lr 8e-3 does to every read-out and to bpb (numbers inflated by the winner's curse).
- F7: significant negative = unrelated-task harm of prior d on domain e (the 6 x 5 matrix). F8: significant positive or
  the LM COST FLAG (mean > +4e-3 bpb) = an LM cost (pilot K2 lambda 1: +7.0e-3, both seeds).

What each comparison can / cannot show:
| comparison | can show | cannot show |
|---|---|---|
| F1 all6-lam1 - none | the joint effect of six priors (lambda 1 each, one window, one clip) on numbers / clock / s4 / dim at lr 1e-3 | which prior causes it (F2 / F3 / F7); that the effect survives the baseline's best lr (F1b) |
| F1b all6-lam1 - none@8e-3 | whether the gain holds against the baseline at its own best lr (user principle; a headline needs it) | lr-matched attribution (the arms differ in lr AND prior); an unbiased numbers contrast (winner's curse favours none@8e-3) |
| FE tier E on seeds 2-7 | whether the pilot's tier-E harm replicates on seeds the pilot never saw | a harm of circle geometry (tier E is memory transfer); against none@8e-3 only n = 2 |
| F2 only-d - none | the own-domain effect of one prior (alone in the clip) | the effect inside all6 (other priors share the clip and the item tokens: clock / numbers share 0..23) |
| F3 all6 - only-d | stacking: interference or synergy of the five other priors on d | which other prior (F7 hints) or whether it is the shared clip / the extra gradient mix |
| F4 all6 - all6-shuf | geometry specificity: the true targets vs the same CKA loss on permuted targets, same dose | that the shuffled loss is inert (it reshapes the same states; s4's shuffle is partial) |
| F5 lambda 0.3 - 1 | the dose response between two lambdas (harm and gain) | an optimum (two points); effects inside the seed noise (n = 4) |
| F6 none@8e-3 - none | what the baseline's best numbers lr does to every domain / bpb; whether a floor verdict is an lr artefact | the domains' own best lr (8e-3 was chosen on numbers) |
| F7 only-d - none on e | unrelated-task harm (or help) of a single prior | the same inside all6 |
| F8 bpb | the LM cost of each prior arm (full val, offline) | downstream LM quality |
| gradient split | whether a prior arm's update is prior-dominated (the user's 'auxiliary' principle) | the per-parameter (AdamW-normalised) mix |

Coordinator notes recorded for the read-out (2026-10-19 pilot audit, report notes/batch1_build/audit_rb6_pilot_results.md):
- months / weekdays claims are tier E only; tier S is at the floor in every pilot arm.
- K5's bpb is unusable (none seeds 1.353 vs 1.255; arms scattered 1.25-1.38): irrelevant at K2 (all K1 / K2 runs
  1.197-1.235), stated here because the pilot table prints it.
- The floor (chance + 5) is below the elimination heuristic (s4 26.8, weekdays S 28, months S 18, dim 15): the heuristic
  line above.
- Held-out-format evals are NOT an eval bug: free generation after an unfamiliar prompt ending fails (seen sets ~0 except
  where the paraphrase ends like a training prompt: clock ' ', dim ' =', weekdays '='). The in-run format numbers are
  descriptive only; format robustness will be measured OFFLINE on the saved models with candidate-restricted scoring
  (argmax over the domain's answer set) by a separate tool (not built yet).
- Task 416 (K5 all6 s1) also exited rc 4 on the geo float check (geo_weekdays -2.4e-4; valid run), like 411 (the
  2026-10-19 re-runs note recorded only 411).
- rainbow6_summary per-seed lists were in load order (the pilot's K2 prior arms printed [s1, s0]; e.g. K2 clock lambda 5
  is s0 = 25.2, s1 = 70.5): fixed (seed order everywhere; regression below).

### Budget
52 runs (40 + the confirm blocks 470-477, 490-493) x ~0.92 A100-h (pilot K2 tasks: 52-55 min wall incl. dry run, offline
LM and checks) = ~48 A100-h; `# time:` 02:00:00 each (104 h of limits; the tick commits the full limit of queued / running tasks: <= 12 x 2 h
= 24 h in flight).
Accounts (heartbeat 2026-10-19T07:12Z; budgets raised to 290 per account by origin/main 7573c9f): ds_ga_1011 268 / 290,
ds_ga_1006_001 197 / 290 -> ~115 h of room for: 386-393 (8 tasks, 62 h of limits, real use lower), these 52 runs (~48
A100-h, 104 h of limits) and toyb 480-482 (~3 h). Fits if 386-393 use <= ~60 h; the tick holds a task back (logs 'no GPU
account has ... left') rather than over-committing, since queued tasks count at their full limit.

### Tests (CPU; scratchpad rb6m/tests/, rb6m/reg/, rb6m/refs/)
| test | result |
|---|---|
| generator --check / --show (gen_check.out, gen_show.out) | "pilot generator unchanged" (its 18 renderings = tasks/400-417 with 9cbfca7's geo line undone; 418 / 419 = 408 / 409 byte for byte); 52 tasks rendered (40 + the confirm blocks 470-477, 490-493), every flag of every task asserted (every other flag = its default), the waves, the seed-major singles, both confirm blocks and the reserved re-run numbers asserted; written: "52 tasks: 52 written", then --check "0 to write, 52 unchanged"; bash -n 52/52 on the written files (--no_confirm: 40). The reused cells' ARGS / ROOT / pin_run lines = the pilot tasks' (404 / 415) byte for byte except the seed |
| render test (render_test.py) | bash -n 52/52; each task's own input-check block (model line, target, per-domain prior / shuf / lam, numbers first, content scan, DRY count + sha256, pinned trainer) on the Torch dry log of its own (arm, seed): 52/52 exit 0 (none-lr8e-3 seeds 4-7: own dry logs, DRY = none's); 630 cross cases with the expected outcome AND reason (another arm, same seed: exit 3 '## domain ...' / '## numbers target' / target line; none vs none-lr8e-3: exit 0, identical dry inputs, the run JSON check catches the lr; the same arm at another seed: exit 3 DRY); the reused cells against the pilot's real Burst dry logs (401 / 404 / 419 / 415): 24/24 exit 3 DRY; the embedded CHECK_PY = make_rb6m_tasks.CHECK_PY, the embedded intent = intent(arm, seed), the embedded ARGS = the generator's, 52/52 |
| CHECK_PY on the pilot's real run JSONs (check_test.py; relocated copies of 401 / 404 / 419 / 415) | 4/4 PASS with the main-design intent (none 16 checks, all6-lam1 25 incl. the lambda ratio: 1.0000 at all 26 on-evals for all 6 priors; versions recorded), and the pilot's own CHECK_PY PASSES on the same inputs (15 / 21); 20 single tampers FAIL exactly the intended check (+ the weighted months gradient 1.01 x lambda at one on-eval) (prior on at one eval, a lambda, domains.prior / shuf, data hash, step-0 lm + 2e-3, lr, a geo_<d> key in none, numbers not first, a domain off in the window, s4 applied on half, months / numbers CKA flat, geo -2e-3, lambda 0.3, no weighted weekdays gradient, geo_s4 never logged, window one step early) and two allowed cases (geo -5e-4; a lambda ratio 1.0005, inside 1e-3) pass; 6/6 whole runs against another arm's intent FAIL (none vs none-lr8e-3: only args; all6-lam1 vs lam0.3: args + lambda + domains + the lambda ratio; vs shuf: args + domains; vs only-months / none: many) |
| tiny CPU smoke (tiny_smoke.py; pinned trainer, d64 x 3, seq 64, 100 steps, evals every 10, window steps 5-39) | A/B TEST PASS (tests/tiny_ab.out; 7 runs, seed 0: none, only-months, only-numbers, all6-lam1, all6-shuf, none-lr8e-3, all6-lam0.3; consumed values printed from each run JSON: args.lr, target, prior_domains, prior_shuf, lam_domains, prior_events, domains.<d>.prior / shuf / lam): every arm has none's data_sha at all 11 evals and the bitwise same step-0 lm 11.521324; only-months: months alone on at the 3 on-evals, prior_used_frac months 0.35 / all others 0, only gnorm_prior_months and geo_months, numbers geo 0, weighted / unweighted = 1.0, cka_months 0.131 -> 0.898 (none 0.134), lm differs from none after the window opens; only-numbers: numbers alone (geo nonzero, no geo_<d>), cka_ops_helix3 0.195 -> 0.500 (none 0.172); weighted / unweighted gradient = lambda for all 6 priors at every on-eval (1.0 in all6-lam1, 0.300 +- 2e-6 in all6-lam0.3: the consumed lambda); all6-shuf: at step 0 (same model, same batch) 1 - CKA to the table used differs from all6-lam1's for every prior (the table differs), domains.<d>.shuf true for all 6 (false in all6-lam1), cos(LM grad, active term) = cos(LM grad, true term) exactly in all6-lam1 but not in all6-shuf (another direction at every on-eval, |diff| 0.002-0.11); window end: CKA to the table USED rises +0.05 (numbers) / +0.45..+0.57 (circles, dim) / +0.20 (s4) while the CKA to the TRUE target stays at -0.05..+0.02 (all6-lam1: +0.24..+0.82 on both); none-lr8e-3: args.lr 0.008, lm at step 10 7.59 vs 10.77 (none), no prior. (A first version of the shuffled-term test compared gradient NORMS, which can coincide for different terms; replaced by the direction test.) |
| CHECK_PY on the tiny runs (tiny facts substituted: steps, n_params, layers, eval spacing, token share 13-20%, CKA bar 0.04, the tiny none run's step-0 lm / data hash) | TINY CHECK TEST PASS (tests/tiny_check.out): all 7 tiny runs PASS every check of their own arm's intent (16 checks for the none arms, 25 for the prior arms); 13 tampers FAIL exactly the intended check (+ lambda 1 used in all6-lam0.3: the lambda ratio; and a shuffled prior's ratio is not checked: passes) (weekdays on / applied / a geo_clock value / a numbers loss term / a weighted dim gradient in only-months, numbers applied on half, the used CKA flat for months and for numbers in all6-shuf, domains.clock.shuf false, lambda 1 on dim in all6-lam0.3, lr 1e-3 in none-lr8e-3), and the allowed case (the TRUE-target CKA flat in all6-shuf) passes; 8/8 runs against another arm's intent FAIL. Tiny CKA bar 0.04 (the shuffled numbers target rises only +0.047 in 35 on-steps at this scale; the 53M bar stays 0.1, and a shuffled prior's CKA-only FAIL is a finding, decision 3); the info line gives the used / true rises and the gradient split (tiny scale: every prior arm prior-dominated, prior / LM 1.7-14: not predictive of 53M, A3) |
| summary regression on the pilot (reg/: the pilot dirs + the pt999 2pct refs, diffed against rb6_pilot.txt / .json) | 520 lines, 25 changed, all 25 exactly the per-seed swap of a K2 all6 / all6-lam1 line ([s1, s0] -> [s0, s1]); JSON: only cells/k2\|all6* differ (36 cells, per-seed lists reversed, seeds label [0, 1]); every other key identical; with --main_readout the first 520 lines are the default run's |
| A5 read-out on planted data (fake_main_test.py: the pilot's real K2 runs relabelled as the 11 arms x seeds 0-3 + confirm seeds 4-7 of none / all6-lam1, planted per-question hits on the six read-out sets, planted offline bpb and gradients) | FAKE MAIN TEST PASS (after the audit fixes): 84 tests in 10 families (F1 4, F1b 4, FE 4, F2-F8 6/6/6/8/7/30/9): the decision seeds per family (FE: 2-7, n 6 vs none / 2 vs none@8e-3), diffs = the plants per seed in seed order, p vs scipy, BH vs an independent loop per family, the extra seed groups (2-7, 0-7, 0-3), sidedness (s4 one-sided vs none; F1b by none@8e-3's seed-level s4 gate); the seed-level gate (8 lines vs scipy t intervals and verdicts), the seed-level heuristic line (yes / no / undecided, seeds above), tier E "n/a"; the per-arm matrix with seed-level intervals; confirm status on the 16 eligible tests (F1, F1b, F6, F8's all6-lam1; planted clock flip of all6-lam1 and s4 flip of none@8e-3 on seeds 4-7 -> NOT confirmed); FE on seeds 2-7 with n 6 against none and none@8e-3; the HEADLINE rule re-derived from the rows for all 6 domains (a clock headline withdrawn on both, an s4 headline withdrawn on F1b alone, a tier-E harm headline on FE); the F8 LM flag; the winner's-curse notes; the gradient FLAG; strict sizes 58 / 61; a missing offline bpb: the strict read-out fails naming the drop, --allow_partial prints it and runs with n 3 |
| mixed-lr end to end (mixed_lr_test.py: real 401 / 404 / 419 / 415 + copies of 401 / 404 relabelled none-lr8e-3, refs pt2b 350-353 at lr 1e-3 none, 8e-3 none, 1e-3 win-ctx) | runs through check() and every section (--allow_partial: an incomplete batch); the A3 numbers table pairs none with the lr-1e-3 none ref, none-lr8e-3 with the lr-8e-3 none ref, all6-lam1 with win-ctx; no ref - ref yardstick across lrs; the same batch without --allow_partial FAILS (incomplete families); + a K1 pilot dir or the lambda-5 'all6' (414) is REFUSED; with --main_readout the 'all6' is excluded with a note |
| statistics (stats_test.py) | Student t CDF vs scipy for df 1-7 (max error 5.9e-11), quantiles, paired() t p and sign-flip p for n 2-8 (1400 random cases), BH on 300 random families |
| strict set / heuristic second path | the summary's tables (re-implemented cube group and exponent table) reproduce every run's s4 / dim classes and masks (asserted per run); strict sizes 58 / 61 and the heuristic levels of seeds 0 / 1 = the audit's (62 / 69 dropped; months S 18.1 / 18.2, weekdays S 27.7 / 28.2, clock 9.8 / 9.7, s4 26.8 / 26.9, dim 15.4 / 14.9); the A5 gate's CIs = the pilot gate's on every shared set |
| Torch references | seeds 0, 1 reproduced bitwise (DRY, step-0 lm, data_sha, DOMS) before seeds 2-7; arm independence as above, incl. none-lr8e-3 at seeds 4-7 (dry 4/4, step 0 at seed 4: bitwise = none's) |

Bugs found by these tests and fixed before this amendment: the summary's per-run caches were keyed by id(run) (an id can
be reused once a run object is freed: a test that discards runs got seed 0 / 1 values for seeds 6 / 7); the cache now
holds the run, so an id cannot be reused while cached (inside the summary every run stays alive anyway). Identical
per-seed differences (t undefined) first gave p = None (a constant +50-point difference "not significant"); now p = its
limit, flagged.

After the fresh build audit (no blocker): the changes above were applied, every test re-run (all PASS), and the pilot
regression is unchanged (still exactly the 25 per-seed swaps); with --main_readout on the pilot data the A5 sections
exclude the K1 / K5 cells and the lambda-5 'all6' with a note, and the pilot's K2 none dim / clock verdicts become
BORDERLINE at the seed level (n = 2).

### Deviations from the build request and why
- geo check tolerance -1e-3 (the pilot generator's current -1e-4 would still have failed task 416: geo_weekdays -2.4e-4).
- Two summary behaviours change for the NEW arms only (the pilot output is unchanged): the A3 numbers table pairs a
  reference by numbers-prior kind AND lr (only-<domain> arms with the none reference, all6-shuf with none, none-lr8e-3 with
  a reference of its own lr), and its ref - ref yardstick only within one lr.
- F1b (all6-lam1 vs none@8e-3) is an extra family next to F1; by the user's decision (a) a headline needs both.
- F1 / F1b now have 4 tests (numbers, clock, s4, dim); months / weekdays tier E moved to FE (decided on seeds 2-7), so a
  pilot-biased p-value enters no BH.
- A headline whose confirm seeds 4-7 (of F1 or of F1b) disagree is withdrawn (readout_rules.survives' rule); the
  coordinator asked only that confirm seeds be reported beside.
- The strict dim set is the main dim read-out (coordinator); its seed-3 size is 46 (< the trainer's 50 for a read-out set).
- Tiny smoke CKA bar 0.04 (pilot A2 / A4 used 0.05): the shuffled numbers target rose only +0.047 at the tiny scale.

### Open questions
None. Decided 2026-10-19: F1 AND F1b for a headline (a); confirm block ON (b); "robust" reported only (c); none-lr8e-3
seeds 4-7 added as 490-493 (1); no lambda-5 arm, lambda 1 is the user's principle (2); a shuffled prior's CKA-only FAIL is
a finding, the run stays valid (3).

### Audit status (experiment-audit)
i double path: strict set / heuristic re-implemented from the group tables and asserted against the runs' own masks and
classes; the A5 gate reproduces the pilot gate's CIs exactly on the shared sets; p-values / CIs / BH against scipy and an
independent loop on planted data. ii asserts: per-domain prior / shuf / lam at the input stage and in the run JSON; the
pilot generator's renderings; waves; flags. iii known answers: the reproduced pilot references (bitwise), planted read-out
(84 tests), the audit's heuristic and strict-set numbers. vi pitfalls: shuffled-target CKA measured against the target
used; seed order; confirm seeds kept out of the decision. vii decoded rows: the Torch dry logs of every (arm, seed)
(refs/out/dry_*.log). v DONE 2026-10-19 (fresh adversarial review of make_rb6m_tasks.py fe21eacd9acb and
rainbow6_summary.py 1750f8f7e528: no blocker; report notes/batch1_build/audit_rb6m_build.md, scripts scratchpad/rb6m_audit/);
its findings are fixed as listed at the top (941a387a41a3 / 70758fe86995). A re-review of these fixes is not done.

## Amendment A5.1 (2026-10-19, before any main result)

Trigger: the fresh re-review of the post-audit fixes (report notes/batch1_build/audit_rb6m_fixes.md, fake-data tests in
scratchpad/rb6m_rereview/) found no blocker in the 52 submitted tasks (430 is running) and two important READ-OUT issues.
Written and fixed before any main-design result was read. Only tools/rainbow6_summary.py changes (70758fe86995 at 2451f30
-> **410620f8b268**); the tasks 430-477 / 490-493, tools/make_rb6m_tasks.py (941a387a41a3), the pinned trainer
(c434d29: cc984148b15a) and toy_struct are unchanged.

### Binding read-out rules added
1. **Seed completeness.** Every main-design arm has a pre-registered seed set: seeds 0-3, and 0-7 for none, all6-lam1 and
   none-lr8e-3 (the confirm blocks 470-477, 490-493). Every test uses the pre-registered seeds of its group and arms:
   F1 / F1b / F2-F8 seeds 0-3, FE seeds 2-7 (n = 6 against both baselines), the confirm block seeds 4-7 (n = 4).
   - If any of them is missing (a task not finished, not published - Burst publishes only ended tasks -, failed before
     training, a re-run directory left out of the command, or a run EXCLUDED by rule 2), the test, gate line, matrix row,
     confirm block and headline that need it print **PROVISIONAL (n=k of K)**: not decided, and never "confirmed" (a
     confirm block with fewer than 4 seeds is "PROVISIONAL (n=k of 4 confirm seeds): unconfirmed, so far agrees /
     disagrees"; a headline with an incomplete F1 / F1b / FE test is "PROVISIONAL ...: not decided").
   - Without --allow_partial the read-out REFUSES (AssertionError naming the missing seeds); --main_readout no longer
     relaxes this. A read-out during the batch therefore needs --allow_partial and is PROVISIONAL throughout. A last line
     states "A5 read-out status: complete (every pre-registered seed present)" or "PROVISIONAL (missing seeds ...)".
   - Only a complete read-out (no PROVISIONAL, no exclusion) can produce a claim.
2. **Run identity.** Before any table (main(), whenever a main-design run is given or --main_readout), every K = 2
   main-design run is checked against its tag's intent, `make_rb6m_tasks.intent(arm, seed)` (the table the task's own
   check used):
   - every trainer argument of the intent (lr, lam, target, prior_domains, prior_shuf, extra_lines = K, tag, seed, model
     and eval settings ...), lam_domains, the active and the shuffled prior sets (domains.<d>.prior / shuf), code_sha and
     toy_struct's dep_sha;
   - the logged multiplier of every unshuffled active prior: gnorm_prior_<d> / gnorm_true_<d> = lambda (rel. 1e-3) at
     every on-eval; an on-eval where it cannot be computed (gnorm_true missing) prints "lambda ratio NOT CHECKED at k of
     n on-evals" (never a silent pass);
   - results/<task>/check.json must exist and PASS. Exception (A5 decision 3): a shuffled arm (all6-shuf) whose ONLY
     failing check is the CKA rise is kept, with a printed note ("a finding, the run stays valid").
   A run failing any of these is **EXCLUDED** from every table with the reason printed ("EXCLUDED run <arm> s<seed>
   (<task>): ..."); its seed is then missing (rule 1).
3. **Marks and labels.** Tier-E rows outside FE (F2-F7 months / weekdays, the F5 non-inferiority lines) that contain the
   discovery seeds 0, 1 of none / all6-lam1 are marked "[tier E incl. the DISCOVERY seeds 0, 1 ...: exploratory; the
   tier-E test is FE on seeds 2-7]". F5's descriptive group is labelled by its real seeds (2-3). FE's headline labels say
   "FE vs none" / "FE vs none@8e-3". In the main design the pilot's PILOT GATE section (item bootstrap) is labelled
   descriptive. The recorded library versions are compared with env311 at task 331 (torch 2.10.0+cu128, transformers
   4.57.6); the pilot's seeds 0, 1 have none recorded (their checks predate A5).

### Known limits (recorded, not changed)
- The submitted tasks' own CHECK_PY still passes its lambda-ratio check vacuously when gnorm_true is missing (the tasks
  are submitted and stay unchanged); rule 2's NOT CHECKED line covers it in the read-out. A5's wording "the lambda
  actually used is asserted" over-claims: it is the multiplier applied in the logged gradient decomposition (for the
  domain priors the same LAMD value as lam_domains; the independent part is the numbers prior's lam_num).
- Re-review item 5: a single-prior arm whose one prior is skipped at an on-eval (< 4 sites or < 3 items) fails its
  check's "0 < grad_prior" and is now EXCLUDED by rule 2. Measured on the pilot's K2 runs: 0 skips (minimum 12-13 sites
  per eval), so negligible; if it happens the exclusion line names the check and the user decides (re-run or accept).
- The confirm rule (same sign and >= 3/4 of the confirm seeds) passes a null effect with probability ~5/16 per test: it
  is the pre-registered readout_rules.survives rule, reported as such.
- Provenance (re-review item 7): the recorded pilot read-out rb6_pilot.txt was produced by an uncommitted summary that
  differs from c434d29's (05599d7f59a7) only in the density line's wording ("tie -> the smaller K" vs "(tie -> 1)");
  numbers and JSON byte-identical.

### Read-out command
Unchanged (A5); mid-batch add --allow_partial (PROVISIONAL). A re-run 478-479 / 483-489 replaces its failed task's
directory.

### Tests (CPU; scratchpad rb6m/a51/)
| test | result |
|---|---|
| pilot regression (the 18 pilot runs + pt999 2pct refs) | byte-identical to the 2451f30 summary's output (TXT and JSON), i.e. still exactly the 25 per-seed swaps vs rb6_pilot.txt; with --main_readout --allow_partial the first 520 lines are the default run's; --main_readout without --allow_partial now refuses (the pilot has seeds 0, 1 only) |
| the re-reviewer's d_fake.py (its harness and planted data, unchanged, on the new summary) | scenarios A, B, D, E as before (scipy / own-BH agreement); scenario C (a missing all6-lam1 run) and C2 (a 1-seed confirm block) now FAIL the strict read-out naming the missing seeds; 1 expected text mismatch: months' headline label now reads "FE vs none n.s." instead of "F1 n.s." (the re-review's own item 1.8) |
| seed_completeness_test.py (the re-reviewer's plant / fake I/O, scenarios C, C2, C3 = none@8e-3 seeds 4-7 not yet published, C4 = only-dim seed 1 missing; strict and --allow_partial) | SEED COMPLETENESS TEST PASS (62 checks): strict fails naming the missing seeds; with --allow_partial a test is PROVISIONAL exactly when one of its pre-registered decision seeds is missing (C 33 / C2 4 / C3 2 / C4 8 of 84), no confirm block with < 4 seeds says CONFIRMED (C2: "PROVISIONAL (n=1 of 4 confirm seeds): unconfirmed, so far agrees"), no headline says confirmed on an incomplete block, every affected headline is PROVISIONAL; the complete design A has no PROVISIONAL verdict |
| identity_test.py (real pilot run JSONs 401 / 404 / 415 / 419 with their published offline_lm.json / check.json, copies changed in one way each, through the CLI) | IDENTITY TEST PASS: kept - none-lr8e-3 s0 with lr 0.008 and a passing check; all6-shuf s0 whose check failed only the CKA rise (note: A5 decision 3); all6-lam0.3 s1 with gnorm_true missing at 3 on-evals (note: NOT CHECKED at 3 of 26); EXCLUDED with the reason - the re-reviewer's mislabelled lr (tag none-lr8e-3, args.lr 1e-3: "'lr': [0.001, 0.008]"), a failed non-CKA check, a shuffled arm failing the CKA rise AND another check, a missing check.json, a logged multiplier 1.0 under lam 0.3; the tables contain only the kept runs; the read-out is PROVISIONAL; strict with the mislabelled run: excluded, then refused for the missing seed |
| the re-reviewer's g_mix / g_mix2 (real dirs) | a relabelled none-lr8e-3 run (lr 1e-3 or 8e-3, no check.json) is now EXCLUDED with the reason; the K1 / lambda-5 mix still refused; with --main_readout excluded with a note |
| fake_main_test.py (planted full design; + A5.1: discovery marks on exactly the tier-E rows outside FE that involve none / all6-lam1, the complete design not PROVISIONAL, F5 group 2-3) | FAKE MAIN TEST PASS |
| mixed_lr_test.py (real runs + relabelled none-lr8e-3 copies with their check.json) | MIXED LR TEST PASS (pairs by lr unchanged; the incomplete batch refused without --allow_partial with the new message) |
| stats_test.py | STATS TEST PASS |

## Amendment A6 (2026-10-19, after the main read-out, BEFORE any A6 run)

Trigger: the user's decisions of 2026-10-19 after reading the main read-out (rb6m_main.txt; FINDINGS F88). F88 left open
three questions:
- every arm at its own best lr: all6's lr was never tuned, and F1b compares all6@1e-3 with none@8e-3;
- whether the prior hurts clock at the same lr: F1b clock is -3.3 n.s., with lr and prior changing together;
- whether "S4 / dim not learnable without the prior" survives a closed lr bracket for none, which so far was tried only at
  1e-3 and 8e-3, and 8e-3 was chosen on numbers.

The hypotheses below come from the main results. They are tested only through the new runs: every A6 comparison involves
at least one A6 arm. The A5 rules, the A5 families and the A5 headline (F1 & F1b) are unchanged.

### Code
- tools/make_rb6m_tasks.py, 941a387a41a3 -> **10b60268e7b1**: the A6 block, with the 52 existing renderings unchanged.
- tools/rainbow6_summary.py, 410620f8b268 -> **89818e712ac3**: the A6 section. The pilot and main outputs are
  byte-identical.
- Unchanged:
  - tools/readout_rules.py (e91ee8a8a846);
  - CHECK_PY (embedded verbatim in every task);
  - the pinned trainer (c434d29: pretrain_range cc984148b15a) and toy_struct (6248613ac6d2).
- Base: origin/main bbd8714. origin/main is now caad91c (report/FINDINGS.md F88 only), which does not touch these files.

### Arms and tasks (K = 2, seeds 0-3, 53M, 3e8 tokens, 18,310 steps of 32 x 512)
| arm (tag rb6mb-53m-2pct-k2-<arm>) | = | tasks |
|---|---|---|
| all6-lam1-lr4e-3 | all6-lam1 with --lr 4e-3 | 494, 496, 498, 500 (s0-s3) |
| all6-lam1-lr8e-3 | all6-lam1 with --lr 8e-3 | 495, 497, 499, 501 |
| none-lr3e-3 | none with --lr 3e-3 | 502, 504, 506, 508 |
| none-lr1.6e-2 | none with --lr 1.6e-2 | 503, 505, 507, 509 |

- Order: seed-major within each block, so every finished pair of lrs is paired.
- Numbering: 494-509 are free in tasks/, status/ and results/ on origin/main. 478 is the 440 re-run, and 483-489 stay
  reserved for re-runs.
- The generator asserts that each A6 task's trainer flags equal its 1e-3 base arm's except --lr, --tag and --out.
- Header: kind gpu, time 02:00:00, cpus 8, after 331-env311 332-data. The pin is c434d29 via burst/pin_run.sh.
- Same task body as A5:
  - the input checks (model line, prior target, per-domain prior / shuf / lam, numbers first, content scan, DRY count +
    sha256, pinned trainer);
  - training;
  - offline_lm.py;
  - CHECK_PY with `make_rb6m_tasks.intent(arm, seed)` (args.lr = 0.004 / 0.008 / 0.003 / 0.016) -> check.json, exit 4
    on a failed check.

### Torch references (reused per seed; lr acts only after the step-0 forward)
- Dry runs of all 16 (arm, seed) cells on the pinned tools (scratchpad rb6m/refs/out_a6/):
  - each gives the seed's DRY sha256 with 88 lines and the seed's domain splits (16/16);
  - 12 of them are byte-identical to the base arm's Torch dry log at the same seed;
  - the other 4 (all6 s0 / s1) are checked by the A6 task's own check block, which accepts the pilot's real Burst dry
    log of 419 / 415 (exit 0).
- Step 0 on CPU: the step-0 lm and data_sha after step 0 are bitwise the seed's reference.
  - all6-lam1-lr8e-3 s0: 11.614818572998047 / 28c9fff8ef37e82f
  - none-lr1.6e-2 s1: 11.603224754333496 / dfac64fb1b4a4f36

| seed | DRY sha256 (88 lines) | step-0 lm | data_sha after step 0 |
|---|---|---|---|
| 0 | fe6a8bba65f7c326... | 11.614818572998047 | 28c9fff8ef37e82f |
| 1 | b4aabe5cd8c829e6... | 11.603224754333496 | dfac64fb1b4a4f36 |
| 2 | 964d1b8b61a15305... | 11.570906639099121 | 36b63d307358a0a4 |
| 3 | ea21c901acad5eca... | 11.604287147521973 | df68c5608eedaa9a |

### Read-out (pre-registered; tools/rainbow6_summary.py `a6_readout`, printed after the A5 sections when an A6 arm is given)
The A5 / pilot sections are computed WITHOUT the A6 runs, so everything up to and including "A5 read-out status" equals
rb6m_main.txt byte for byte. The A6 section then uses the A6 arms plus the A5 cells none, none-lr8e-3 and all6-lam1,
seeds 0-3 only.

Read-out values are the A5 ones (READ):
- numbers: the 6-bucket area;
- months / weekdays: heldE final3;
- clock: held_wrap final3;
- s4: held_fresh_nonid final3;
- dim: the STRICT set final3;
- bpb: offline bpb_full x 1e3.

Tests are paired across seeds: paired t, the exact sign-flip p and a t interval. BH (q 0.05) is applied within each
family.

1. **all6's best lr** over {1e-3, 4e-3, 8e-3}, by ONE criterion fixed now: `readout_rules.best_lr` on the numbers area
   (seeds 0-3).
   - Every lr within 2 points of the top is tied, and the tie goes to the lower mean offline bpb (in-training bpb if any
     tied run lacks it). A non-finite bpb counts as the worst; an exact tie goes to the lower lr. This is the pt2b rule
     that picked none@8e-3.
   - No per-domain selection.
   - The bracket uses `readout_rules.bracket`. It is OPEN when the best is at an edge and beats its neighbour by > 2
     (all6's best may lie beyond the grid; reported, no extra runs pre-registered), and gets a NOTE when the area argmax
     is at an edge within 2.
2. **F1c** = all6@best - none@8e-3 on F1b's four domains (numbers, clock, s4, dim), seeds 0-3.
   - F1b's tests: s4 is one-sided (greater) because none@8e-3's seed-level s4 gate is FLOOR (A5: 3.1 [0.0, 6.2]); the
     rest are two-sided. BH is within F1c (m = 4).
   - Printed next to the A5 F1 and F1b rows of that domain.
   - Each domain states its **reading**:
     - "no change: all6's best lr is 1e-3, so F1c = F1b by construction";
     - "no change (F1b X, F1c Y)";
     - "CHANGES the best-vs-best reading: F1b X -> F1c Y (the A5 headline, F1 & F1b, is not changed by it)".
     - Here X / Y is gain or harm (significant) or n.s.; "PROVISIONAL: cannot compare" when F1b or F1c is incomplete.
   - **The headline rule stays F1 & F1b.** F1c neither creates nor removes a headline; it qualifies the best-vs-best
     wording.
3. **F9** = all6@8e-3 - none@8e-3, the same lr, on the six domains + offline bpb, seeds 0-3. BH within F9 (m = 7); the s4
   side as F1b. The clock row is marked "<- the question".
4. **LM cost of the new prior arms.**
   - F8c = offline bpb x1e3, all6@4e-3 / 8e-3 - none@1e-3, as F8. BH within F8c (m = 2).
   - **LM COST FLAG** when the mean > +4 (x1e-3 bpb, A1 R5) whatever p says, or when the prior arm diverged.
   - Gradient split per seed (window medians of the on-evals before the global clip): LM share of T, clip factor
     1/max(1, T), prior/LM and the share of on-evals with prior > LM. Shown for all6@1e-3 (A5 reference), 4e-3 and 8e-3;
     FLAG = a seed's median prior/LM > 1.
   - F8c is against none@1e-3, so it mixes lr and prior. The same-lr LM cost at 8e-3 is F9's bpb row.
5. **"S4 / dim not learnable without the prior."** Per domain (all six; s4 and dim carry the claim), none's best lr over
   {1e-3, 3e-3, 8e-3, 1.6e-2} = `readout_rules.best_lr` on that domain's read-out (seeds 0-3, the same tie rule).
   - The bracket is closed or OPEN at an edge (the best at an edge by > 2), and each lr's seed-level A5 gate verdict is
     printed (FLOOR / BORDERLINE / MEASURABLE / CEILING).
   - The claim **holds only if every none lr is FLOOR** by the seed-level gate (complete seeds). Otherwise it reads
     "'<d> not learnable without the prior' does NOT hold: none's best lr X reaches v [seed CI] -> verdict (floor f,
     heuristic h)", plus "none's bracket is OPEN at the X edge: its best on <d> may lie beyond the grid" when open.
   - Already known before A6: none@8e-3's strict dim is 31.6 [18.0, 45.2] = MEASURABLE (A5 gate). So the dim claim
     cannot hold in A6. A6 only shows where none's dim peaks and whether the bracket closes.
   - s4 so far: none@1e-3 0.2, none@8e-3 3.1, both FLOOR.
   - **F1d** = all6@best - none@best-on-that-domain, per domain, DESCRIPTIVE (no BH, no verdict). It is printed next to
     F1c. Months / weekdays rows that involve none@1e-3 or all6@1e-3 carry the A5.1 tier-E discovery-seed mark.
6. **Winner's curse.** A5's note (none@8e-3 was selected on pt2b seeds 0-3, which share the init and the numbers data with
   these seeds) stays on every numbers row that involves none@8e-3: F1c and F9, and F1d when none's best numbers lr is
   8e-3.
   - When all6's best lr is not 1e-3, the F1c header also says that all6's lr was chosen on numbers on the same seeds 0-3.
     F1c numbers then carries a selection effect in all6's favour as well.

**A diverged run is data, not an exclusion** (pt2main's rule). A 53M run at 1.6e-2 may diverge. In pt2b, none@1.6e-2
(362-365, the same 53M 2% setting with the older pt2 data) finished 4/4 with no non-finite value, final bpb 1.253-1.268
against 1.218-1.241 at 8e-3.
- Onset: the first eval with a non-finite lm, val_loss, geo, grad_prior or geo_<d>. This is CHECK_PY's onset (and its
  info.diverged_at_step).
- The run's task check PASSES when nothing else fails:
  - its lambda-ratio and grad checks use the on-evals before the onset;
  - the CKA-rise and offline-delta checks are skipped;
  - a diverged run with a wrong lr is still caught (tested).
- Read-out:
  - listed ("diverged runs ... {arm sN: step}" plus a WARNING line), never excluded;
  - its area / final3 count over ALL evals of the run as trained. Accuracies are argmax exact matches, so they stay
    finite: the broken model's chance-level answers;
  - its offline bpb (NaN) is the worst in every best-lr tie-break, so it never wins a tie;
  - a paired row on its bpb (F9 bpb, F8c) prints "NOT TESTED: DIVERGED <arm> at seeds [...] (a non-finite value = the
    worst: that run's LM is broken; no t test)", is never significant, and raises F8c's LM COST FLAG when the prior arm
    diverged;
  - its gradient split uses its on-evals before the onset (the row says "[before the divergence at step S: n on-evals]").
- A run that crashes instead of finishing is not "diverged data" but an unfinished task (rule below).
  - This is not expected: the trainer has no NaN guard and keeps stepping, the evals are argmax / norm ratios, and
    offline_lm.py writes NaN without an assert.
  - The strict read-out then refuses; --allow_partial lists it as unfinished and PROVISIONAL; the user decides.

**Seed completeness and run identity** (as A5.1):
- Every A6 run is checked against `make_rb6m_tasks.intent(arm, seed)`: args incl. lr, lam_domains, priors, code_sha, the
  logged lambda multiplier, and check.json PASS. A failing run is EXCLUDED with the reason.
- Seeds 0-3 of every arm used are required. A missing seed makes every verdict that needs it PROVISIONAL, and the read-out
  REFUSES without --allow_partial.
- The last line reads "== A6 read-out status: complete (seeds 0-3 of every arm used)" or "PROVISIONAL (...)".

### What each comparison can and cannot show
- (1) / F1c can show whether all6's best-vs-best gain over none@8e-3 holds, per domain, once all6 also gets an lr search.
  - It cannot remove the selection effects (both lrs chosen on numbers on seeds 0-3).
  - It cannot replace the confirm seeds: the A6 arms have none.
  - It cannot say "each at its own best lr per domain": the rule is numbers-only on purpose.
- F9 removes the lr confound for clock at one lr (8e-3).
  - It cannot say which lr is best for either arm.
  - A clock harm at 8e-3 would be a same-lr effect of the priors at that lr, not a general one.
- (5) can refute "not learnable without the prior" for a domain, given a non-FLOOR none lr.
  - It can support the claim only within the grid {1e-3 ... 1.6e-2} and at 53M / 3e8 tokens: an OPEN bracket says the
    grid may be too narrow.
  - F1d picks none's lr per domain on the same seeds, so it is biased in none's favour: it is descriptive and
    conservative for the prior.
- The new information is the A6 runs only. Their baselines (none@8e-3, all6@1e-3, none@1e-3 seeds 0-3) are the main runs
  that suggested these hypotheses, and they reappear unchanged.

### Read-out command
The command that produced rb6m_main.txt, plus the A6 directories (494-509 run last):
```
python tools/rainbow6_summary.py results/401-rb6pb-* results/404-rb6pb-* results/415-rb6pb-* results/419-rb6pb-* \
  results/4[3-6][0-9]-rb6*-* results/47[0-8]-rb6*-* results/49[0-3]-rb6mb-* \
  results/49[4-9]-rb6mb-* results/50[0-9]-rb6mb-* \
  --ref 'results/35[0-3]-pt2b-lr1e-3-none-s?' --ref 'results/35[0-3]-pt2b-lr8e-3-none-s?' \
  --ref 'results/35[0-3]-pt2b-lr1e-3-win-ctx-s?' --json rb6m_a6_summary.json
```
- 440 holds only a failed log_tail.txt; 478 is its re-run.
- Mid-batch, add --allow_partial (A6 PROVISIONAL).

### Tick order and budget
- 494-509 are the highest task numbers in tasks/, so Burst runs them after everything else (file-name order, MAX_GPU_JOBS
  12): 12 start and then 4 more, about 2 waves of about 1 h each with Burst idle.
- Training is 0.85 A100-h per run (median of the 40 main rb6mb runs with a check.json; GPU util 97%, 32.3 GB peak).
  With offline_lm and the check, about 16 x 0.92 = 15 A100-h; 16 x 2 h = 32 h of limits.

### Tests (CPU; scratchpad rb6m/a6/)
| test | result |
|---|---|
| generator --check | 68 tasks: 0 to write, 68 unchanged (430-477, 490-493, 494-509), 52 already ran; bash -n clean on all 16 new files |
| regress.sh: pilot (18 pilot runs + pt999 refs) and main (the rb6m_main.txt command) | TXT and JSON byte-identical to the committed summary's (main TXT = rb6m_main.txt) |
| render_test_a6.py (all 68 tasks) | RENDER TEST PASS. bash -n; each task's input-check block exits 0 on its own Torch dry log, 1054 cross cases exit 3 with the right reason, 56 runs on the pilot's Burst logs (8 A6 tasks accept 419 / 415 / 401 / 404 at their own seed); embedded CHECK_PY = the generator's, intent = intent(arm, seed), ARGS = trainer_args |
| check_test_a6.py (the A5 cases on the rb6m3 generator + A6) | CHECK SCRIPT TEST PASS (51). Pilot JSONs relabelled to each A6 intent (seeds 0 / 1) PASS; tag with the lr left at 1e-3, or an A6 run vs its base intent, fails only "run args" ({'lr': [0.001, 0.004]} ...); DIVERGED copies (NaN from step 9000 in none-lr1.6e-2 / step 3000 = in-window in all6-lam1-lr8e-3, offline bpb NaN) PASS with info.diverged and the onset step; diverged + wrong lr still caught |
| tiny_a6.py (pinned trainer, tiny model, --stop_after 20) | TINY A6 A/B PASS (26). Controls (all6-lam1 / none + stop_after) = the full tiny runs bitwise at steps 0 / 10 / 20; each A6 arm: args.lr consumed (0.004 / 0.008 / 0.003 / 0.016), every other argument = the base's, same data_sha and step-0 lm, same prior schedule, lm at steps 10 / 20 differs from the base (e.g. all6 4e-3 8.427, 8e-3 7.322, base 10.602 at step 20) and between the two lrs |
| a6_plumbing_test.py (real main dirs + 16 relabelled copies of the same seeds' runs, CLI) | A6 PLUMBING TEST PASS (16). The output = rb6m_main.txt byte for byte, then the A6 section; JSON = main JSON + 'a6'; identity 16 checked 0 excluded; identical copies -> best 1e-3 -> F1c = F1b by construction; a copy whose lr was not changed is EXCLUDED, strict refuses, --allow_partial PROVISIONAL; two DIVERGED copies: listed, warned, not excluded, F9 / F8c bpb rows NOT TESTED naming the arm, LM COST FLAG, gradient split from the 8 pre-onset on-evals, 8e-3 loses the tie-break (bpb inf) |
| a6_planted_test.py (planted per-(arm, seed) values; the summary's own rules) | A6 PLANTED TEST PASS (20). Best 4e-3 by the bpb tie-break (not the area argmax), bracket note; F1c clock p = scipy, "CHANGES ... F1b n.s. -> F1c gain"; s4 one-sided; F9 clock harm; F8c flag 8e-3 only; gradient FLAG 8e-3 only; none's s4 at 3e-3 MEASURABLE -> "does NOT hold", dim bracket OPEN at the upper edge, the diverged none@1.6e-2 s2 kept with area 0; scenario 2 "HOLDS"; scenario 3 missing seed: strict refuses, partial PROVISIONAL; winner's-curse notes on exactly the none@8e-3 numbers rows; F1c header names all6's own lr selection |
| A5.1 tests on the A6 summary: seed_completeness (62), identity (11), fake_main, mixed_lr (4), stats | all PASS (the pre-A5.1 copies in rb6m/tests fail the same way on the committed summary: stale, superseded by rb6m/a51/tests) |

### Decisions taken in the build (for the reviewer)
- The A6 references in the code and the task headers point to "amendment A6 of report/audits/rb6_A5_main_design.md".
  If A6 is filed elsewhere, change those 7 strings and regenerate 494-509 (not yet run).
- The summary's grad_split now ignores on-evals at or after a divergence onset. Finite runs are unchanged (byte-identical
  outputs).
- CHECK_PY is unchanged (it is embedded in the 52 submitted tasks). Its info.grad_window_median is NaN for a diverged
  prior run. That field is information only; the read-out uses its own pre-onset split.

### Audit status (experiment-audit)
New experiment code and tasks: a fresh audit is due before submission (gate rule), not yet run.

## Amendment A6.1 (2026-10-19, before any A6 result)

Trigger: the fresh audit of A6 (report notes/batch1_build/audit_rb6_a6.md; its scripts in scratchpad a6_audit/).
- It found no blocker in the 16 tasks 494-509. They were submitted and are running.
- It found read-out issues: the wording of claim (5), BH's m, the gradient FLAG of a diverged run, the finiteness assert
  in check(), the order of the strict refusal, and what --allow_partial does to the A5 section.

These fixes are written before any A6 result was read. Only tools/rainbow6_summary.py changes: 89818e712ac3 (414ecd1)
-> **f2d6470b054e**. Unchanged:
- tasks 494-509;
- tools/make_rb6m_tasks.py (10b60268e7b1);
- tools/readout_rules.py (e91ee8a8a846);
- CHECK_PY;
- the pinned trainer (c434d29: cc984148b15a) and toy_struct.

The strict pilot, A5 and A5.1 outputs are byte-identical, TXT and JSON (main = rb6m_main.txt).

### Binding read-out rules (replacing the corresponding A6 text)
1. **Claim (5), "<d> not learnable without the prior" (s4, dim).** It is decided by the seed-level A5 gate of EVERY none lr
   in {1e-3, 3e-3, 8e-3, 1.6e-2} (seeds 0-3), not by none's best lr.
   - An lr is **above the floor** when its seed-level CI lower bound is >= the floor. That covers MEASURABLE, CEILING, and
     BORDERLINE only by crossing the ceiling: verdict_of calls a CI of [80, 99] BORDERLINE, but it is above the floor.
   - **HOLDS**: every none lr is FLOOR.
   - **DOES NOT HOLD**: at least one lr is above the floor.
     - Every such lr is named with its level: "<lr> final3 [seed CI] vs floor f -> verdict".
     - BORDERLINE lrs follow as "also BORDERLINE across the floor: ...".
   - **NOT ESTABLISHED**: no lr is above the floor, but at least one is BORDERLINE across the floor. Each is named with its
     level: "not shown to be at the floor, not shown to be learned; every other lr FLOOR".
     - This is never written "does not hold". In a FINDINGS entry it must not become "learnable without the prior".
   - **PROVISIONAL**: a none run is missing (not published, EXCLUDED, or unfinished).
     - The line reads "PROVISIONAL (n=k of 16 none runs; missing <arm> [seeds]): not decided; so far <every lr's level,
       with n k of 4 where short>".
     - It starts "every available lr FLOOR:" when that is so.
     - It carries no HOLDS / DOES NOT HOLD / NOT ESTABLISHED. A missing seed lowers certainty; it never flips the direction.
   - The reason names the lrs that break FLOOR, not none's best lr. A6's template sentence ("none's best lr X reaches ...")
     is withdrawn. The best lr and the bracket stay in the line above the claim.
   - The heuristic level and an OPEN bracket are appended as before.
   - The JSON records the sentence in a6.claims and the status in a6.claim_status.
2. **dim: none@8e-3's confirm seeds, printed beside the claim (descriptive, not part of it).**
   - none@8e-3's dim gate on seeds 4-7 (the A5 gate k2|none-lr8e-3[4-7]|dim_held_strict) is shown next to its seeds 0-3
     verdict, with "replicates" / "does NOT replicate".
   - Today: seeds 4-7 give 10.0 [0.1, 20.0] vs floor 11.5 -> BORDERLINE, against MEASURABLE 31.6 on seeds 0-3. That does
     NOT replicate, and 8e-3 was chosen on pt2b seeds 0-3, which share the init and data.
   - So a dim "DOES NOT HOLD" that rests on none@8e-3 rests on seeds that do not replicate. It must be reported with this
     line.
3. **BH's m is printed on every family line** (F1c, F9, F8c): "BH m = k of n rows".
   - m counts only rows with a p, as A5's bh() does. When a row is NOT TESTED (a diverged run's bpb) or n < 2, the line
     says so: "..., j without a p (NOT TESTED / n < 2): the others' BH milder".
   - F9's pre-registered m = 7 then shows as 6 of 7. The computation is unchanged; the deviation is visible.
4. **Gradient split of a diverged run.**
   - Each such seed is labelled "[DIVERGED at step S: pre-onset evals only (k of K)]". K counts every on-eval with both
     gradients logged.
   - A seed whose k < K/2 raises no "FLAG: prior > LM". The early window has a larger prior / LM ratio: audit case,
     all6-lam1 s1 has 1.16 on the first 8 on-evals against 0.53 on all 26.
   - Its prior/LM > 1 is noted as "(prior/LM > 1 at s[..] only from < half the on-evals: not a FLAG)".
5. **check()**: the A5 finiteness assert (lm, val_loss, geo_<d> at every eval) is back for every non-A6 run. Only an A6
   arm's non-finite values give the DIVERGED warning (data).
6. **Strict mode refuses BEFORE any A6 conclusion.** A missing or excluded A6-section run, or a DROPPED value, prints only:
   - the A6 header and mode;
   - the run identity and EXCLUDED lines;
   - the warnings, the missing seeds and the diverged list;
   - the problems and "A6 read-out REFUSED (strict)".
   The process then exits with AssertionError and writes no JSON. No best lr, family row, reading or claim is printed.
7. **--a6_partial (new) and --allow_partial.**
   - `--a6_partial` lets the A6 section be incomplete. Every verdict that needs a missing run is PROVISIONAL, and the status
     line reads PROVISIONAL. The A5 section stays strict and byte-identical to rb6m_main.txt (TXT; JSON without "a6"). An
     unfinished A5-design run is still refused.
   - `--allow_partial` keeps its A5.1 meaning: any --allow_partial read-out is PROVISIONAL. With every A5 seed present, the
     A5 section then differs from the strict output in exactly 2 lines (tested):
     - the added line "  PROVISIONAL: unfinished runs or an incomplete batch (--allow_partial)" under the A5 MAIN-DESIGN
       header;
     - the A5 status line, which reads "PROVISIONAL (--allow_partial)" instead of "complete (...)".
     No A5 number changes. The A6 section prints a NOTE saying exactly this.
   - The mode is printed on the A6 section's second line (the consumed value).
8. **An unfinished A6 run counts as missing** (A5.1 rule 1). It is listed as "UNFINISHED run(s) counted as missing". This is
   reachable only with --a6_partial / --allow_partial, because check() refuses it otherwise.

### Recorded, not changed
- Audit item 9: the F1c numbers row's inline note names only none@8e-3's winner's curse. all6's own selection of its lr
  on numbers on the same seeds (bias in all6's favour) is stated in the F1c header when the best lr is not 1e-3.
- Audit item 10: a re-run of an A6 task can only take 479 / 483-489, which are shared with A5 re-runs. Its directory must be
  added to the read-out command; if it is missing, the read-out refuses (strict) or reads PROVISIONAL, never silently wrong.
- Audit Q5-3 (real NaN run, tiny model, lr 30):
  - the trainer, offline_lm and CHECK_PY run through;
  - the accuracies after divergence are 0.0 (argmax of NaN logits = token 0), not "chance level" as A6 said; FLOOR is
    unaffected;
  - a finite but exploded run (lm about 6e9) is ordinary data, shown only by bpb and the LM COST FLAG.
- Audit Q4: F1c's p-values are not corrected for choosing all6's lr among three. The selection bias on numbers is small
  (simulated 0-3.5 points against F1b's +32.1). The A6 arms have no confirm seeds.

### Read-out command
A6's command, unchanged. Mid-batch, add **--a6_partial**: A5 strict and byte-identical, A6 PROVISIONAL. Not
--allow_partial.

### Tests (CPU; scratchpad rb6m/a61/)
| test | result |
|---|---|
| regress.sh (pilot + main, strict) | TXT and JSON byte-identical to the 89818e712ac3 / 4090d62 outputs; main = rb6m_main.txt |
| verify61.py on run_all61.sh: the auditor's plant.py scenarios (main, holds, all6div, missing_none, missing_all6) + A6.1 scenarios (meas3, missing_holds, unfin_none, ceil_border), real JSONs through the CLI, each strict / --allow_partial / --a6_partial | A6.1 SCENARIO TEST PASS (48). Strict and --a6_partial: the A5 part = rb6m_main.txt byte for byte + JSON = main JSON + 'a6' (all 9 scenarios); --allow_partial: exactly the 2 A5 lines named in the NOTE. main (auditor's): s4 NOT ESTABLISHED naming 3e-3 6.0 [-8.1, 20.1] BORDERLINE, not the best lr 1.6e-2; dim DOES NOT HOLD naming 8e-3 31.6 [18.0, 45.2] MEASURABLE, beside it none@8e-3 seeds 4-7 10.0 [0.1, 20.0] BORDERLINE 'does NOT replicate'; BH m 4 / 7 / 2. holds: HOLDS. meas3: 3e-3 MEASURABLE while the best lr is 1.6e-2 (BORDERLINE) -> DOES NOT HOLD naming 3e-3, 1.6e-2 only as 'also BORDERLINE'. ceil_border: 1.6e-2 89.7 [79.3, 100.0] BORDERLINE by the ceiling -> DOES NOT HOLD. all6div: F9 'BH m = 6 of 7 rows, 1 without a p', F8c 1 of 2; s1 '[DIVERGED at step 3000: pre-onset evals only (8 of 26); fewer than half: no FLAG from this seed]', no FLAG (the A6 build printed FLAG here). missing_none / missing_holds (all FLOOR, a none seed missing): strict = header + identity + missing + REFUSED only, rc 1, no JSON; partial modes 'PROVISIONAL (n=15 of 16 none runs; missing ...): not decided; so far every available lr FLOOR' (was 'does NOT hold (PROVISIONAL)'). missing_all6: strict refused before conclusions; --a6_partial F1c rows PROVISIONAL (n=3 of 4). unfin_none (an A6 run cut at step 12000): strict refused by check(); partial modes 'UNFINISHED run(s) counted as missing', s4 PROVISIONAL |
| check_finite_test.py | CHECK FINITE TEST PASS: an A5-design run with NaN lm / val_loss / geo_months asserts in every mode (strict, --allow_partial, --a6_partial); the same run as an A6 arm gives one DIVERGED warning |
| a6_plumbing_test.py (A6's, on the A6.1 summary) | A6 PLUMBING TEST PASS (16): unchanged expectations (the A5 part = rb6m_main.txt, identity, the mislabelled lr EXCLUDED and refused, two diverged copies) |
| a6_planted_test.py (A6's, updated for the A6.1 wording + the strict refusal before conclusions) | A6 PLANTED TEST PASS (21): s4 'DOES NOT HOLD: none is above the floor at 3e-3 15.8 [15.0, 16.5] ... MEASURABLE', claim_status; scenario 2 HOLDS; scenario 3 strict prints no A6 conclusion |
| A5.1 tests: seed_completeness, identity, fake_main, mixed_lr, stats | all PASS (seed_completeness 62, identity 11, fake_main, mixed_lr 4, stats); mixed_lr's pt2b refs now read from llmadd-burst-ro (the old rb6m worktree is gone) |

## Amendment A7 (2026-10-19, after the A6 read-out, BEFORE any A7 run)

Trigger: the user's decisions of 2026-10-19 after the A6 read-out (rb6m_a6.txt). Two things in A6 prompt it:
- none@1.6e-2 learns S4 in some seeds. Its s4 read-out final3 per seed is 68.3 / 16.9 / 36.3 / 8.7, and it first
  reaches 25% at step 3750 / never / 9250 / never.
- all6@1e-3 gives 36.9 / 41.5 / 52.7 / 29.0 on seeds 0-3 and 48.8 / 41.1 / 51.0 / 36.5 on seeds 4-7. It reaches 25%
  at steps 1750-3000 in 8/8 seeds.
- Also, all6's lr bracket is open at the lower edge (numbers area: 1e-3 68.8 > 4e-3 61.8 > 8e-3 55.7).

**Hypothesis**, formed on these A5 / A6 numbers and tested only with the new runs: "the prior makes S4 learned
reliably and early, while without it S4 is learned only at a high lr, in some seeds and late".
- The none side of every A7 test is new.
- all6@1e-3 has no new seeds. Its seeds 0-7 were all seen when the hypothesis was formed.
- The A5 / A6 / A6.1 rules and headlines are unchanged. A7's claims are new, and each is stated as tested on new runs only.

### Code
- tools/make_rb6m_tasks.py: 10b60268e7b1 -> **2b629d98a87c**. Adds the A7 block; the 68 existing renderings are unchanged.
- tools/rainbow6_summary.py: f2d6470b054e (f1731e4) -> **490bfe8877f9**. Adds the A7 section.
  - The A7 runs are split off before every other section, so the pilot, A5 and A6 outputs are byte-identical with and
    without them.
  - New flag --a7_partial.
- Unchanged: readout_rules (e91ee8a8a846), CHECK_PY, the pinned trainer (c434d29: cc984148b15a) and toy_struct.
- Base: origin/main de73af2.

### Runs (K = 2; each identical to its base arm except --lr and the tag; tasks 510-521, seed-major)
| arm (tag rb6mb-53m-2pct-k2-<arm>) | = | seeds | tasks |
|---|---|---|---|
| none-lr3.2e-2 | none with --lr 3.2e-2 | 0-3 | 510, 513, 516, 519 |
| all6-lam1-lr5e-4 | all6-lam1 with --lr 5e-4 | 0-3 | 511, 514, 517, 520 |
| none-lr1.6e-2 | the A6 cell (none with --lr 1.6e-2), fresh seeds | 4-7 | 512, 515, 518, 521 |

- 483-489 stay reserved for re-runs; 479 is the re-run of 508.
- The generator asserts each A7 task's flags = its base arm's except --lr / --tag / --out.
- Header: gpu, 02:00:00, cpus 8, after 331-env311 332-data.

### Torch references
- The seeds' references are reused: A5 for seeds 0-3, and the confirm block (470-477, 490-493) for seeds 4-7.
- Dry runs of all 12 cells give the seed's DRY sha (88 lines). They are byte-identical to the base arm's Torch dry log at
  that seed (12/12; for all6 s0 / s1 the base is all6-lam1-lr4e-3's log, which equals the Burst inputs of 419 / 415).
- Step 0 of none-lr1.6e-2 s5 on CPU: lm 11.569442749023438 and data_sha 99801f9e457357aa, bitwise the seed-5 references.

### Read-out (pre-registered; tools/rainbow6_summary.py `a7_readout`, printed after the A6 section when an A7 run is given)
The S4 read-out is READ["s4"] = s4_held_fresh_nonid (as A5 / A6).

**The per-seed level (decision threshold of (1)).** This is the full Latin-square propagation level of that seed's
read-out set (`s4_latin_level`).
- Method: the trained cells' answers are givens. Naked singles and hidden singles per row / column are iterated to a
  fixpoint. A derived cell scores 1; any other held cell scores 1 / (remaining candidates) when its answer is among them.
- It uses only the Latin-square property, not the group law. It is the F88 result audit's computation
  (notes/batch1_build/audit_rb6m_results.md), re-implemented in the summary.
- It reproduces the audit: mean 32.5 over seeds 0-3 and 31.5 over seeds 0-7. Per seed: 35.2 34.8 29.6 30.4 | 32.2 30.2
  26.8 33.1.
- It is the same for every arm of a seed (same split).
- The summary's one-step elimination heuristic (26.2 / 25.7) is printed beside it, as a descriptive line only.

1. **Reliability.** Per arm and per seed: is s4 final3 >= that seed's level? The count is k of n.
   - R1 (primary): all6@1e-3 vs none@1.6e-2 at the fresh seeds 4-7.
   - R2 (second none): all6@1e-3 vs none@3.2e-2 at seeds 0-3, the same seeds, so the pairs share data and init.
   - Decision: Fisher exact on the 2x2 counts, one-sided in the hypothesis' direction (all6 has the larger share), BH
     within R (m = 2). A significant comparison prints "MORE RELIABLE with the prior (SIGNIFICANT)".
   - Reported beside: Barnard exact (pooled z, one-sided), and the paired seed-level t on final3 (one-sided, BH within R
     over the t's, sign-flip p).
   - With n = 4 against 4, Fisher can reach p < 0.05 only at 4/4 vs 0/4 (p = 1/70 = .014). That is a limit of the design.
2. **Speed.** Per seed: the first eval step with s4 read-out >= 25%, or "never" (censored at the budget, step 18309).
   - The same pairs as (1): S1 and S2.
   - Decision: an exact permutation Gehan test, one-sided (all6 earlier), BH within S (m = 2).
     - U = sum over pairs of +1 when all6's seed reached it first (the earlier one observed), -1 when none's did, 0 for
       ties or both never.
     - p = the share of the C(8, 4) = 70 relabellings with U >= U_obs; the minimum is 1/70.
   - Beside: the exact sign test over the paired seeds (both never = a tie, dropped; with 4 pairs its minimum is
     0.0625), and medians (the step by which half the seeds reached it; "never" if not reached).
3. **Best vs best on s4.**
   - all6@best: over {5e-4, 1e-3, 4e-3, 8e-3} by the numbers area, A6 (1)'s rule (readout_rules.best_lr, ties within 2 ->
     lower offline bpb).
   - none@best on s4: over {1e-3, 3e-3, 8e-3, 1.6e-2, 3.2e-2} by the s4 AREA, the same tie rule.
   - Tests: all6@best - none@best on s4 area and s4 final3, two-sided, paired by seed, BH within the family (m = 2, or 4
     when the fresh seeds exist).
     - Seeds 0-3 rows are marked "[both lrs chosen on these seeds]".
     - Seeds 4-7 rows are fresh and are added where both cells have them: all6@1e-3, and none@{1e-3, 8e-3, 1.6e-2}.
       Otherwise the section says "not available".
4. **none@3.2e-2.**
   - Divergence is handled as in A6 / A6.1. A diverged run is data: its accuracies over all evals count (0 after the
     onset), its offline bpb is the worst in the tie-break, and it is listed, never excluded.
   - none's s4 bracket over the five lrs is printed by area (the A7 criterion, used by (3)) and by final3 (A6 (5)'s), each
     with readout_rules.bracket, plus each lr's seed-level gate verdict.
   - If none's best is at the 3.2e-2 edge by > 2, the bracket stays OPEN and is reported as such ("none's s4 may peak
     beyond the grid"). No further runs are pre-registered.
5. **all6@5e-4.**
   - A6 (1) is re-run over the four lrs; the bracket is printed, and whether the best lr is UNCHANGED or CHANGED against
     A6's (1e-3).
   - **F1c updated** = all6@best - none@8e-3 on numbers / clock / s4 / dim, with F1b's tests (s4 one-sided), BH m = 4, and
     each domain's reading against F1b ("F1c unchanged: ... = F1b by construction" while the best stays 1e-3).

**Robustness** (A5.1 / A6.1):
- Required runs:
  - all6 at the four lrs and none at the five lrs, seeds 0-3;
  - all6-lam1 and none-lr1.6e-2 at seeds 4-7.
- A missing run counts as missing whether it is not published, EXCLUDED by run identity (tag's intent = run JSON, and
  check.json PASS) or unfinished. It makes every verdict that needs it PROVISIONAL: counts n=k of 4, never SIGNIFICANT.
- Strict mode refuses BEFORE any A7 conclusion: it prints the header, mode, identity, missing seeds and "A7 read-out
  REFUSED (strict)", then raises AssertionError and writes no JSON.
- `--a7_partial` lets the A7 section be incomplete while the A5 / A6 sections stay strict (byte-identical).
- The last line reads "== A7 read-out status: complete (every A7 seed present)" or "PROVISIONAL (...)".

### What A7 can and cannot show
- (1) / (2) on R1 test the hypothesis on fresh none runs. The all6 side is old: seen when the hypothesis was formed, and
  never at a new seed.
- With 4 seeds per arm, only an extreme split decides (4/4 vs 0/4; Gehan U = 16).
- R2 / S2 use none@3.2e-2, which may diverge. A diverged run counts as not learned and never reached.
- The level is the Latin propagation, so "above the level" means beyond what Latin-square constraints alone give. It does
  not mean "learned the group".
- (3) on seeds 0-3 carries the selection of both lrs on those seeds. Only the seeds 4-7 rows are free of it, and they exist
  only for some best-lr pairs.
- p-values are not corrected for the choice of lr.

### Read-out command
The A6 command (incl. the re-run 479) plus the A7 directories:
```
python tools/rainbow6_summary.py results/401-rb6pb-* results/404-rb6pb-* results/415-rb6pb-* results/419-rb6pb-* \
  results/4[3-6][0-9]-rb6*-* results/47[0-8]-rb6*-* results/49[0-3]-rb6mb-* \
  results/49[4-9]-rb6mb-* results/50[0-9]-rb6mb-* results/479-rb6mb-* results/51[0-9]-rb6mb-* results/52[01]-rb6mb-* \
  --ref 'results/35[0-3]-pt2b-lr1e-3-none-s?' --ref 'results/35[0-3]-pt2b-lr8e-3-none-s?' \
  --ref 'results/35[0-3]-pt2b-lr1e-3-win-ctx-s?' --json rb6m_a7_summary.json
```
- Mid-batch: add --a7_partial.
- A re-run of an A7 task takes 483-489, and its directory is added to the command.

### Tick order and budget
- 510-521 are the highest task numbers, so they run last: one wave of 12 (MAX_GPU_JOBS 12), about 1 h with Burst idle.
- About 12 x 0.92 = 11 A100-h; 24 h of limits.

### Tests (CPU; scratchpad rb6m/a7/)
| test | result |
|---|---|
| generator --check | 80 tasks: 0 to write, 80 unchanged (68 existing + 12 written), 68 already ran; bash -n clean |
| regress7.sh | pilot, A5 main (= rb6m_main.txt) and A6 (= rb6m_a6.txt, the A6 command incl. 479) TXT + JSON byte-identical to the committed summary's |
| render_test_a7.py (80 tasks) | RENDER TEST PASS: the 12 A7 dry logs = their base arm's; each task's input-check block exits 0 on its own log; 1382 cross cases with the right reason; 80 runs on the pilot's Burst logs (12 at their own seed exit 0); embedded CHECK_PY / intent / ARGS = the generator's |
| check_test_a7.py | CHECK SCRIPT TEST PASS (70 = the A5 / A6 cases + 19 A7). Pilot JSONs relabelled to none-lr3.2e-2 / all6-lam1-lr5e-4 (s0 / s1) and A5-confirm none runs relabelled to none-lr1.6e-2 s4 / s5 PASS; the lr left at 1e-3 or the base intent -> only 'run args'; a diverged none-lr3.2e-2 copy PASSES |
| tiny_a7.py (pinned trainer, --stop_after 20) | TINY A7 A/B PASS: args.lr 0.032 / 0.0005 consumed; other args = the base's; same data_sha, step-0 lm and prior schedule; lm at steps 10 / 20 differs from the base and from another lr of the same base (none 3.2e-2 6.821, 1.6e-2 7.057, 1e-3 9.960 at step 20; all6 5e-4 11.043, 1e-3 10.602) |
| a7_planted_test.py (planted values; the summary's own rules) | A7 PLANTED TEST PASS (17): Fisher = scipy, Barnard ~ scipy (< 2e-3), paired t = scipy, BH = scipy; Gehan U / p = an independent computation (incl. ties and censoring), sign test, medians; best lr by the tie-break (UNCHANGED / CHANGED, OPEN lower), none's s4 OPEN upper reported; (3) at seeds 0-3 and, when both exist, 4-7 (m = 4); a diverged run counted; a missing seed: strict refuses before any conclusion, --a7_partial PROVISIONAL |
| a7_plumbing_test.py (real dirs + 12 relabelled copies, CLI) | A7 PLUMBING TEST PASS (10): output = rb6m_a6.txt byte for byte + the A7 section, JSON = A6 JSON + 'a7'; identity 12 / 0; levels reproduce 32.5 / 31.5; a copy with its lr unchanged EXCLUDED, strict refused before any A7 conclusion; --a7_partial PROVISIONAL with A5 / A6 unchanged |
| the A5.1 / A6 / A6.1 tests on the A7 summary | all PASS: seed_completeness, identity, fake_main, mixed_lr, stats, a6_planted, check_finite (the A6 / A6.1 scenario outputs are covered by the byte-identical A6 regression on the real A6 data) |

## Amendment A7.1 (2026-10-19, interpretation only, BEFORE any A7 result; from the fresh audit notes/batch1_build/audit_rb6_a7.md)
- The prior side of every A7 test (all6@1e-3, seeds 0-3 and 4-7) was SEEN when the hypothesis was formed; only the none side (none@1.6e-2 seeds 4-7, none@3.2e-2 seeds 0-3) is new. So a SIGNIFICANT R1 / S1 is read as "none is unreliable / late on fresh seeds, against the prior arm's already-known 8/8 runs", not as a fresh-data confirmation of the prior's reliability. The printed "MORE RELIABLE / EARLIER with the prior (SIGNIFICANT)" lines are to be quoted with this reading.
- R2 (all6@1e-3 seeds 0-3 vs none@3.2e-2 seeds 0-3) cannot reach significance by construction: all6's seeds 0-3 are known to be 3/4 above their Latin level (seed 3: 29.0 < 30.44), so even 0/4 for none gives Fisher p = 5/70 = .071. An R2 n.s. is not evidence against the hypothesis; R2 is reported descriptively (counts, levels, steps to 25%).
- (3) best-in-grid vs best-in-grid stays descriptive on seeds 0-3 (both lrs chosen on those seeds). The Latin level is naked + hidden singles to a fixpoint; full constraint propagation (GAC) would be 0-2.2 points higher per seed (audit M4): "above the level" is read with that margin.
- No change to tasks, code or the A5 / A6 rules.

## Amendment A8 (2026-10-20, BEFORE any A7 result and before any A8 run; user: "make the conclusions stronger"; the user asked whether the S4 test set is good enough)
- **Strict S4 read-out (s4_hard)**: the A5 S4 read-out set (s4_held_fresh_nonid) restricted to the cells NOT solvable by Latin-square propagation (naked + hidden singles to a fixpoint) from that seed's own training table, the same per-seed computation as A7 (levels 35.2 34.8 29.6 30.4 | 32.2 30.2 26.8 33.1 for seeds 0-7). On s4_hard a pure Latin-square table-filling strategy scores at chance, so its accuracy cannot come from that heuristic. It is computed from the per-question hits already logged in every run (no re-run), for every existing and future rb6 run, and printed with n per seed.
- s4_hard is the PRIMARY S4 metric for A7's reliability (1) and speed (2) read-outs and for A8; the A5 / A6 / A7 rows on s4_held_fresh_nonid stay as they are (secondary). The threshold of "learned" on s4_hard: final3 >= 25% (chance on s4_hard is ~1/24 + the Latin-row/column elimination share; the exact chance per seed is printed).
- **A8 runs**: all6-lam1@1e-3 and none@1.6e-2 at fresh seeds 8-11 (8 runs, same args except seed; Torch references to be computed for seeds 8-11). Here BOTH sides are fresh, so A8 is the confirmatory test of "the prior makes S4 learned reliably and early" that A7 cannot be (A7.1). Tests: the count of seeds learned on s4_hard (Fisher exact, one-sided, prior > none), steps to 25% on s4_hard (Gehan, one-sided), paired t on s4_hard final3 and area; seeds 8-11 decide, all other seeds descriptive.
- No change to the A5 / A6 / A7 rules or headlines.
