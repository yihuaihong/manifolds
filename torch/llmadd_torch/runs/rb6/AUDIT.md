# rb6 — rainbow6 PILOT: six manifolds pretrained at once at 53M (2026-10-17)

Status: code + job files built, independent audit done and its findings fixed (Amendment A1 below, before any run);
user decisions of 2026-10-17 after A1 implemented as Amendment A2 (numbers line first, domain evals in that position,
held-out-format evals, lambda 5 per prior; Burst tasks 400-407), BEFORE any run; NOT submitted. Rev B is uncommitted in the working tree of llmadd-burst (public repo:
code only, no keys); make_jobs_rb6.py refuses --submit until Rev B is committed and PIN is set.
Update 2026-10-18 (Amendment A4, BEFORE any run; NOT submitted): Rev B + A1-A3 are committed (c434d29, trainer
cc984148b15a); the user's pilot extension = K 1, 2, 5 x none / all6 lambda 5 / all6 lambda 1 x seeds 0, 1 = 18 Burst
tasks 400-417 (supersedes "Burst tasks 400-407"; generator 1bbb2244206a, PIN to be set at its commit). Independent audit
of A4 done (no blocking finding; findings and fixes at the end of A4; summary c32891cace9a = a print-only fix). OPEN for
the user BEFORE results: the K\* gate scores every K on the K 1 eval layout (A4 audit finding 1: what that does and does
not bias; accept, or ask for a K-matched eval = a trainer amendment).

Background: notes/next_phase_synthesis.md (rainbow v2 = RB2, Rev B, R0 pilot, R1 rules), notes/next_phase_designs.json
(RB2-P0 / RB2-53M), FINDINGS F70 (fine-tune rainbow: lambda / n_priors restores the single-prior gains), F76 + addendum
(rainbow v1 pretraining at 10..99: weekdays / months held-out at the floor for every arm; circle priors pull 3-4x harder
than the helix at equal lambda; +0.064 nats/token LM cost with three priors), F80 / F81 (class splits; the target must give
structure, not answers), F84 (S4 / {-1,0,1}^3 toy: window too short for S4, always-on much stronger; negation twins),
F82 / F85 (main recipe: 2% numbers, window (0.05, 0.4), ctx site, helix3, lambda 5, lr 1e-3).

USER DECISIONS (2026-10-17), the constraints of this design: pretrain 6 manifolds at once at 53M — numbers 0..999
(helix3), months (circle Z12), weekdays (Z7), clock hours (Z24), S4 cube rotations (faithful 3-d matrix rep R, nonce
single-token names), dimension lattice {-1,0,1}^3 (exponent vectors, nonce single-token names). Every arm sees the SAME
data mixture; arms differ only in which priors are on. Data option 1: numbers stay at the 2% main setting
(--arith_lines 1); the other 5 domains are ADDED on top. First a PILOT: none vs all-6-on with the current best setup,
2 seeds each, to check each domain is "learnable but not at ceiling"; the pilot also compares 1 vs 2 extra domain lines
per row.

## Code
- Trainer (A2, binding): sha256[:12] **cc984148b15a** (= code_sha of every result; see Amendment A2). Before A2:
- Trainer: llmadd-burst tools/pretrain_range.py Rev B, sha256[:12] **4a657953dfcc** (= code_sha of every result; =
  e231a219007f + the A1 read-out bookkeeping, training path unchanged), commit
  TBD (PIN in make_jobs_rb6.py). All Rev B behaviour is behind `--domains`; without it the trainer is the pinned
  a3bce3ab63c1 (regression below). Imports tools/toy_struct.py (S4 group, dim table, splits, closure, negation twins),
  sha256[:12] **6248613ac6d2** (recorded in every result as dep_sha; re-checked by every sbatch).
- Generator: llmadd_torch/make_jobs_rb6.py (8 pilot job files rendered in this directory; asserts every flag of every
  job; `--print-main K` lists the main design). Conventions as make_jobs_ptlr.py: HOLD kill switch, SUBMITTED_ flags,
  code snapshot by `git archive PIN tools` + code/COMMIT, the sbatch re-checks both shas, `--dry_inputs` then the run,
  --save_model, --ckpt_every 2000, L40S, 03:00:00 (53M numbers-only 0.95-0.97 L40S-h + domain evals / gradient logs).
- Summary: llmadd-burst tools/rainbow6_summary.py (read-outs computed from the curves, checked against the run JSON's
  own `metrics` = double path; integrity asserts; the pilot gate and density rule below).

## Data mixture (identical in every arm of a (seed, K); asserted)
Row (512 tokens) = FineWeb-Edu text (train.bin, 4e8 tokens; stream 2000 + seed) + 1 numbers line (exactly the 2% main
setting's line: 4 formats, operands 0..999 by length bucket, per-seed held-out pairs, 5% rehearsal pool; stream 1000) +
K domain lines; the domain of the j-th extra line of row r at step s is EXTRA[((s * 32 + r) * K + j) mod 5]
(months, weekdays, clock, s4, dim: 6-7 (K = 1) or 12-13 (K = 2) lines per domain per batch), the question drawn
uniformly from that domain's training pool (stream 4100: SeedSequence(4100 + seed), child 0); the 1 + K lines in random
order (stream 5000 + seed); text fills the rest. RNG registry: 1000 pairs, 2000 rows, 3000 prefix, 4000 share (Rev A),
4100 domains (children 0 training lines, 1-5 splits, 6-10 shuffled targets, 11-15 eval contexts; index = position in
EXTRA, so a domain's split never depends on which other domains are on), 5000 line order, 10**6 + step logging.
No prior touches a data stream: `data_sha` (running sha256 of every batch) and step-0 lm are identical across arms
(CPU smoke: none vs all6, 11/11 evals, step-0 lm 11.525801658630371 both).

| (measured, --dry_inputs, 50 batches) | text | numbers | months | weekdays | clock | s4 | dim |
|---|---|---|---|---|---|---|---|
| token share %, K = 1 | 96.35 | 2.12 | 0.273 | 0.273 | 0.508 | 0.234 | 0.234 |
| token share %, K = 2 | 94.82 | 2.13 | 0.547 | 0.547 | 1.016 | 0.469 | 0.469 |
| first line after the text, K = 1 | — | 0.51 | 0.10 | 0.10 | 0.09 | 0.10 | 0.10 |
| first line after the text, K = 2 | — | 0.33 | 0.14 | 0.13 | 0.13 | 0.14 | 0.14 |
Per run (18,310 steps): 585,920 numbers lines (= pt999 2%); 117k (K = 1) / 234k (K = 2) lines per domain, i.e. each
training question ~990 / 1,990 times (months), ~3,100 / 6,200 (weekdays), ~245 / 490 (clock), ~340 / 680 (s4, dim).
Deliberate differences from pt999 2% (so numbers are NOT comparable to pt999's 2pct cells, only within rb6): the numbers
line is first after the text in 1/(1+K) of the rows (pt999: always), and the text is 6-26 tokens shorter.

## Domains (split per seed; sizes for seed 0)
| domain | line (prior sites ⟦ ⟧, scored span « ») | items | split | train / held / heldE | chance (uniform, majority) | closure |
|---|---|---|---|---|---|---|
| months | `5 months after⟦ March⟧ is« August\n»`, k = 1..23, k != 12 | 12 names (single tokens) | 40% of the (item, k mod 12) classes held (tier S, both k); of the trained classes 50% have one k held (tier E) | 118 / 106 / 40 | 8.3, 5.7 | 2-hop 1.00, chain 1.00, inverse trained 0.81 |
| weekdays | `3 days after⟦ Monday⟧ is« Thursday\n»`, k = 1..13, k != 7 | 7 names | as months | 38 / 34 / 12 | 14.3, 5.9 | 0.94, 1.00, 0.53 |
| clock | `9 hours after ⟦14⟧:00 is «23:00\n»`, k = 1..47, k != 24 | hours 0..23 = number tokens | as months; training lines without wrap whose pair is a held-out numbers pair are dropped (29 for seed 0, 36 for seed 1) and those classes never in tier E | 482 / 442 (337 wrap) / 151 | 4.2, 1.8 | 1.00, 1.00, 0.60 |
| s4 | `⟦ gez⟧ then⟦ cuk⟧ is« nok\n»` (R_c = R_b R_a) | 24 nonce tokens | question split, train fraction 0.6 (F84) | 345 / 231 | 4.2, 2.6 | associativity 1.00, + Latin 1.00; 22 held cells have their commuting twin trained |
| dim | `⟦ zaj⟧ times⟦ roz⟧ is« zaj\n»` / `… per …` (e_a +- e_b in the box) | 27 nonce tokens | relation-class split, train fraction 0.5 (86 of 172 classes; F84) | 344 / 342 | 3.7, 7.9 | row space 1.00, nullity 3; 182 held have their negation twin trained |
Nonce tokens: OLMo-2 single tokens " xyz" (CVC, lowercase, not numbers), the 51 rarest such tokens in the 4e8 train bin
after removing words / abbreviations / names / vulgar ones (≤ 85 occurrences each, i.e. ≤ 2.1e-7 per token; months
26k-50k, weekdays 4.8k-8.7k), S4 and dim disjoint, fixed across seeds (S4_NAMES / DIM_NAMES in the trainer, counts
re-verified by every --dry_inputs).
Clock and numbers share token ids: the hours 0..23 (and every k) are the numbers' operand tokens. The numbers prior
(helix3) acts on operand states inside arithmetic lines, the clock prior (C24) on hour states inside clock lines; with
tied embeddings both move the same embedding rows. A no-wrap clock line states h + k = an addition fact, so: (1) clock
lines never state a held-out numbers pair (dropped above: no leak into the numbers evals); (2) the clock read-out is
tier S **with wrap** (h + k >= 24); no-wrap held-out clock questions are reported split by whether the addition pair is
a numbers training pair (nowrap_numtrain: answerable as plain addition) or held out (nowrap_numheld).

Asserts in the code (fail = crash): piecewise tokenisation = whole-string tokenisation for every question line; every
site and answer span decodes to its item / answer (build time, and step 0 on the real batch); tier-S classes not
trained; tier-E questions not trained but their class is; every non-zero shift trained on >= 2 items; every item a
trained start and a trained answer (s4 / dim via toy_struct.split); dim: no relation class on both sides; no held-out
prompt (token ids up to the answer) equals a training prompt, within and across domains; no clock training line states a
held-out numbers pair; far-shuffled targets CKA < 0.15 to the true one (s4: max over R and R_sign, best of 3000 =
0.34 / 0.34 for seeds 0 / 1, as F84); CKA(target, target) = 1 (numpy and the trainer's torch path); --dry_inputs:
content scan of 640 training rows (no held-out prompt inside any row's lines; also counted in the text: 0), nonce
counts = recorded.

## Priors
- On/off per domain (--prior_domains; numbers iff --target helix3 / helix3_shuf); one shared schedule: window
  (0.05, 0.4) = steps 916..7323 (6,408 of 18,310).
- Targets: numbers helix3 (periods 2, 5, 10, 100, 1000 + x / 1000) on the operands; months / weekdays / clock the circle
  of their period (cos, sin of 2 pi i / n) on the start item; s4 R (9-d) on a and b; dim the exponent vector (3-d) on the
  two inputs. Never on an answer token.
- Site = ctx: the hooked state (layers 2, 4 = 1/3, 2/3 of the depth) at the item token inside that domain's lines of the
  batch; loss term = mean over the layers of 1 - linear CKA(states, centred Gram of the column-standardised target rows of
  the batch's items) (pretrain_helix.cka / centred_gram, the numbers prior's code); a domain's term is skipped on a batch
  with < 4 sites or < 3 distinct items (logged as prior_used_frac_<d>).
- lambda rule (--lam_rule norm, used here): every active prior gets lam / (number of active priors), so the total prior
  weight is lam = 5 in every arm (F70 '@norm'): all6 = 5/6 per prior; a single-prior arm = 5 (= the numbers recipe).
  --lam_rule each (5 per prior) exists, not used.
- Shuffled variant (--prior_shuf): numbers = helix3_shuf (the existing per-seed permutation, CKA 0.008 to helix3); the
  others = a far permutation of the items (CKA to the true target < 0.15, s4 see above).
- Logged at every eval, every arm: geo_<d> (the active prior's value; its fit is still logged when off),
  cka_<d>@L (item states of the domain's eval prompts vs the TRUE target, read layers 2, 4), prior_on_domains,
  prior_used_frac_<d>, dom_sites, gradient norms of the LM loss and of every TRUE prior (unweighted) and all their
  cosines, the weighted active terms (gnorm_prior_<d>) and their sum (grad_prior / grad_cos as the default path),
  token_share, first_line_frac, data_sha.

## Metrics (unique definitions)
- `<d>_held` / `<d>_heldE` / `<d>_seen`: exact match of the WHOLE scored span (answer tokens through the newline; clock
  "23" ":" "00" "\n"), teacher forced, argmax per position, of the domain's held-out (tier S) / tier-E / trained
  questions, each placed at the END of a 512-token row right after a val-text chunk = the training position of a first
  line (pretrain_range.py:evaluate, Rev B block, the numbers evals' scoring code). Contexts per question so every set has
  >= 200 rows (max 8): rows above. Sub-splits: `clock_held_wrap / _nowrap_numtrain / _nowrap_numheld`,
  `<cyc>_held_wrap / _k_ge_n`, `s4_held_fresh / _comm_twin_trained`, `dim_held_no_negtwin / _negtwin_trained /
  _times / _per`.
- numbers: unchanged (held_bXY, 6-bucket mean).
- area = 100 x mean over ALL evals (75); final3 = 100 x mean of the last 3 evals (steps 18000, 18250, 18309: the last ~5M tokens, lr at its
  floor); t50 / t90 = M tokens to the first eval >= 0.5 / 0.9, linear interpolation, None = never (censored at 300M).
  pretrain_range.py:curve_metrics -> run JSON `metrics`; tools/rainbow6_summary.py:metrics recomputes them (|diff| <
  1e-9 asserted). Same definitions as readout_rules.area / t_theta.
- Read-out set per domain (A1; run JSON domains.<d>.readout): numbers 6-bucket mean; months / weekdays tier S; clock
  tier S with wrap; s4 `s4_held_fresh_nonid` (no commuting twin trained, no identity input); dim
  `dim_held_no_negtwin_nonid` (no negation twin trained, not a {x, 0} copy class). Superseded wording: s4 held; dim held
  (claims additionally require the same sign on dim_held_no_negtwin, F84 amendment 2).
- bpb: in-training val_bits_per_byte (65k tokens, noisy); offline_lm.py on the saved models for any LM claim.

| dimension | training | eval | same? |
|---|---|---|---|
| position | domain line right after the text in 1/(1+K) of the rows, else after 1-2 other lines | right after the text, row end | yes for the first-line slot (logged share) |
| template | one per domain (dim: times / per) | same | yes |
| items / k range | train pool | held-out pool, same ranges | generalisation by design (class / question split) |
| split overlap | — | none by content (asserted) | — |
| model state | train, bf16 autocast | eval, bf16 autocast | as the numbers evals |

## Expected magnitudes / predictions (written before any rb6 result)
- Chance (A1: max(uniform, majority training answer) ON THE READ-OUT SET; seed 0 / 1): months 8.3 / 8.3, weekdays
  14.3 / 14.3, clock (wrap) 4.2 / 4.2, s4 (fresh, nonid) 4.2 / 4.2, dim (no negtwin, nonid) 3.7 / 5.4. Before A1:
  months 8.3, weekdays 14.3, clock 4.2 (wrap), s4 4.2, dim 7.9-8.5 on the whole held sets; floor line =
  chance + 5.
- none, numbers: pt999 2pct-none area 9.4, final held 17.2 (8 seeds) — expect similar or lower (numbers first-line share
  0.51 / 0.33). Seen sets (all domains): high (memorised; each training question seen hundreds of times) — the upper-bound
  check of the eval pipeline.
- none, held-out: v1 (F76) had weekdays / months at the floor for every arm under a class split (with minus lines);
  F84's toy none never learned S4 at 0.6 nor dim at 0.3 (dim 0.6 yes). Prediction: weekdays / months tier S and s4 at or
  near the floor, tier E off the floor (the equivalent shift of the same item is trained), clock wrap near the floor,
  clock no-wrap / numtrain well above it (addition), dim uncertain. Several domains may fail the gate: that is a result
  of the pilot, not a bug — unless the seen sets are also low.
- all6: numbers area well above none (2% priors 47-66 in F82; here 5/6 lambda per prior), item-state CKA >= 0.9 at the
  window's end for the circles (CPU smoke: months 0.94, clock 0.93 at step 40), lower for s4 / dim (smoke 0.48 / 0.67).
  The circle priors' gradients are 5-8x the numbers prior's at equal weight (CPU smoke; v1: 3-4x): grad_prior / grad_lm
  is the thing to watch; bpb within +0.02 of none expected, not guaranteed.

## Controls / known answers
- Default path = a3bce3ab63c1 (CPU regression, results below).
- CKA(target, target) = 1; far-shuffled CKA < 0.15 (asserted); all6-shuf (main) keeps the true-target read-out low
  (CPU smoke: cka_months 0.10-0.16 while its geo falls).
- Same data in every arm (data_sha, step-0 lm); resume from a checkpoint bit-identical (CPU smoke); window steps exact.
- Upper bound: the seen sets must rise in every arm (if a domain's seen set stays low, the eval / data path is broken,
  not the model).

## PILOT (8 runs, ~9 L40S-h)
{none, all6} x {K = 1, 2} x seeds 0, 1; 53M d384 x 6, seq 512, bs 32, 3e8 tokens (18,310 steps), lr 1e-3 (1% warmup,
cosine to 0.1x), AdamW (0.9, 0.95) wd 0.1, clip 1.0, bf16; all6 = --target helix3 --schedule window --window 0.05,0.4
--site ctx --lam_rule norm --prior_domains numbers,months,weekdays,clock,s4,dim. Tags rb6p-53m-2pct-k<K>-<arm>.

### Pre-registered pilot rule (decided on the none arm only; the all6 arm is never used to select)
Binding version: rules 1, 4 and 5 as amended by A1 (below); the text here is the original.
1. MEASURABLE at K iff the mean over the 2 none seeds of final3 on the domain's read-out set lies in
   [chance + 5, 90] points (chance = max(uniform, majority-answer baseline) of each seed's split, averaged).
   Alternative (pre-registered): a cyclic domain failing on tier S is MEASURABLE (tier E) iff its tier E (clock: tier E
   with wrap) passes the same rule; its own-domain claims are then about the equivalent-shift tier only.
2. Density: K* = the K with more measurable non-number domains (tier E counts); tie -> K = 1 (half the synthetic
   tokens, less text dilution). Reported next to it, not part of the rule: numbers none area at K = 1 vs 2.
3. Main design at K*: every domain stays in the mixture (same data whatever the gate says). MEASURABLE domains get
   two-sided own-domain claims. A FLOOR domain (none < chance + 5 at K*) gets a one-sided reading only: all6 - none > 0
   is "the prior makes it learnable" (as numbers at 2%), a null is uninterpretable. A CEILING domain (none > 90) gets
   speed claims (area, t90) only.
4. Sanity of all6 (descriptive, not a gate; any failure goes to the user before the main design): no non-finite value;
   every active prior applied on >= 95% of the on-steps; item-state CKA at the window's end; in-training bpb within
   +0.02 of the same seed's none; seen sets high.
5. Integrity (asserted by rainbow6_summary.py): one code_sha, finished runs, identical data across arms, exact window.

## MAIN design sketch (for approval after the pilot; not generated)
At K*: none, all6, only-numbers, only-months, only-weekdays, only-clock, only-s4, only-dim, all6-shuf x seeds 0-3 =
36 runs, of which none / all6 seeds 0, 1 are the pilot runs (identical config, reused) -> 32 new runs, ~35 L40S-h
(`make_jobs_rb6.py --print-main K`). Proposed rules (n = 4: paired t p < .05 and 4/4 wins):
- R1 joint benefit (own domain): all6 - none on area; MEASURABLE two-sided, FLOOR one-sided (rule 3).
- R2 geometry specificity: all6 - all6-shuf > 0 on the same read-outs.
- R3 single vs joint: only-d - none (own domain); all6 - only-d with a no-interference margin (90% CI lower bound >
  -3 numbers / -5 others). Caveat: only-d has lambda 5 on its prior, all6 5/6: the comparison confounds "other priors"
  with "lambda / 6" (option: an all6 lambda-each arm, +4 runs).
- R4 caveat (A1): 29-42 trained clock lines per seed with k > 24 and no wrap after the period shift (e.g. '29 hours
  after 3:00 is 8:00') are the shifted twin of a HELD-OUT numbers pair (3 + 5); recorded as
  domains.clock.n_numbers_held_shift_twin. Same data in every arm, so arm contrasts are unaffected; an only-clock
  effect on numbers b11 / b12 / b22 may go through this route.
- R4 transfer (descriptive): the 6 x 6 matrix of only-d effects on every domain (e.g. only-numbers on clock wrap /
  nowrap; only-clock on months).
- R5 LM cost: offline bpb (offline_lm.py, full val) all6 - none and only-d - none; flag > +0.004.
- R6 coexistence on shared tokens: in all6 at the window's end, cka_clock (hour states in clock lines vs C24) and
  cka_ops_helix3 (operand states in arithmetic lines vs helix3) both >= 0.9.

## Amendment A1 (2026-10-17, after the independent audit, BEFORE any run): binding changes

Independent adversarial review of e231a219007f (auditor scripts: session scratchpad rainbow6/audit/). Each finding was
re-checked by the builder (rainbow6/fix/); fixes are read-out bookkeeping only: the training path, the data, the
splits, the priors and the RNG streams are unchanged (Rev B smoke runs give the same curves on every old key and
bit-identical models; see the test table at the end of this section). Trainer 4a657953dfcc.

| # | finding (auditor) | real? (builder's check) | fix |
|---|---|---|---|
| 1 HIGH | s4 / dim read-outs contain copy cells a trivial rule answers (s4 identity input; dim x*0, 0*x, x/0, x/x) | yes: seeds 0-3 s4 18-22 identity cells of 231 (copy + swap heuristic 13-17% vs floor 9.2); dim 54-70 copy cells of 342 (15.8-20.5%), on dim_held_no_negtwin 23-28% (floor 8.7-13.5) | new masks `nonid`, `fresh_nonid` (s4), `nonid`, `no_negtwin_nonid` (dim); the read-out (gate AND claims) is s4_held_fresh_nonid (203 / 191 questions, seed 0 / 1) and dim_held_no_negtwin_nonid (120 / 130); chance recomputed on the read-out set; copy cells = "answer is an input, or dim x/x" asserted equal to the identity cells (s4) and to the relation classes {x, 0} and the auditor's four rules (dim) on the whole table |
| 2 HIGH | lambda rule norm departs from "the current best setup" | yes, a design choice: notes/next_phase_designs.json RB2 had "lam 5 per prior" with @norm as an extra arm; F70 (fine-tune) found @norm better | not changed (user decision): make_jobs_rb6.py `LAM_RULE` (one line); the none arm does not depend on it (`--submit --arms none`) |
| 3 MED | 5-point margin inside the noise (weekdays 17 classes: SE ~6.8 points for 2 seeds) | yes (class is the independence unit; both k of a class held together) | the trainer stores per-question hit counts at every eval (`hits_<set>`, units / masks in domains.<d>.eval_<set>); the gate uses a 90% cluster-bootstrap CI (units resampled within each seed, seeds averaged, 4000 draws): MEASURABLE iff CI within [chance + 5, 90]; FLOOR iff upper < floor; CEILING iff lower > 90; else BORDERLINE (to the user); per-seed values printed |
| 4 MED | the summary decides on unfinished runs, on one none seed, without the seen check | yes (warning only; any number of seeds; no seen check) | unfinished runs assert (`--allow_partial` to look; verdicts then PROVISIONAL); >= 2 none seeds per K, same seeds at every K, asserted; none seen final3 < 80 -> INVALID (overrides the verdict) |
| 5 MED | the pilot rule never looks at all6 | yes | descriptive all6 verdict on the same CI / bounds ("in range" / "above 90: no room" / "at the floor" / "borderline") + per-seed Δ vs none; never a gate |
| 6 LOW | clock wrap set mixes k >= 24 "subtract 24 then add" (31-34%) with true mod wraps; trained k > 24 clock lines are period-shifted twins of held numbers pairs | yes: seed 0 105 of 337 (true mod-wrap 232); shift twins 29 / 36 / 42 / 38 (seeds 0-3) | `modwrap` mask (i + k mod n >= n) reported for every cyclic held set; `n_numbers_held_shift_twin` recorded; R4 caveat added; read-out unchanged (wrap) |
| 7 LOW | step-0 lm exact equality across GPU jobs | plausible (GPU kernels need not be bitwise reproducible); data_sha is the real check | tolerance 1e-3 |
| 8 LOW | months inverse_trained 0.81 (seed 0) vs 0.55-0.62 (seeds 1-3); s4 shuffled target CKA 0.34 | yes (recorded in domains.<d>) | documented: all6-shuf is a partial control for s4 (as F84); per-seed values are printed |

Binding pilot rule after A1 (replaces rules 1, 4, 5 above; 2 and 3 unchanged except BORDERLINE / INVALID):
1. Per domain the READ-OUT set (domains.<d>.readout): months / weekdays tier S, clock tier S with wrap, s4 fresh and
   non-copy, dim no-negation-twin and non-copy. chance = max(uniform, majority training answer) on that set, seeds
   averaged; floor = chance + 5. Statistic: mean over the none seeds of final3; 90% cluster-bootstrap CI as above.
   MEASURABLE / FLOOR / CEILING / BORDERLINE as above; INVALID iff a none seed's seen-set final3 < 80. A cyclic domain
   not MEASURABLE on tier S is MEASURABLE (tier E) iff tier E (clock: with wrap) is MEASURABLE by the same rule.
2. K* = the K with more MEASURABLE (incl. tier E) non-number domains; tie -> K = 1 (unchanged); BORDERLINE counted as
   not measurable and listed.
3. BORDERLINE / INVALID domains go to the user before the main design (no default reading).
4. all6: descriptive range verdict (finding 5) besides the sanity list.
5. Integrity: as before + finished runs, >= 2 none seeds per K, per-question hits = curve accuracies at every eval
   (second double path), read-out metrics in the run JSON = the summary's.

Not fixed here, to the user: lambda rule (finding 2), the S4 schedule (F84: window too short for S4, always-on much
stronger; the pilot uses the shared window), and a re-review of the A1 diff by a fresh subagent (none was available in
this session; the diff is bookkeeping only).

A1 tests (trainer 4a657953dfcc, summary eb8630899e53; scripts and logs in the session scratchpad rainbow6/fix/):
| test | result |
|---|---|
| default-path regression vs pinned a3bce3ab63c1 (CPU, 1 thread, 50 steps; builder c1-c5 + auditor a1-a2) | 7/7 configs: per-step loss + pre-clip grad norm identical 50/50, curve, run JSON keys / args, saved model (36 tensors) and --dry_inputs bytes identical; step 0 / 49 losses as before (c1 11.526635 / 10.184761 ... a2 11.518012 / 9.525920) |
| Rev B smoke vs e231a219007f (A all6 100 steps, B resume, C none, D all6-shuf, E only-months) | every old curve key identical, only the allowed new keys (hits_*, *_modwrap, *_nonid), old domain info / metrics unchanged, models bit-identical to e231a; resume B == A (curve + model); data_sha A = C = D = E on common steps, step-0 lm 11.525801658630371 |
| oracle / off-by-one fake model | 65 eval sets and sub-splits (55 + the 10 new) score 1.000 / 0.000; every per-question hit string = contexts / 0 |
| copy-cell second path (auditor's rules, toy_struct only; seeds 0, 1) | run masks == the auditor's rules; shortcut heuristic on the whole held set s4 13.0 / 17.3, dim 65.5 / 64.0; on the read-out s4 1.0 / 0.0, dim 1.7 / 5.4 vs chance 4.2 / 4.2, 3.7 / 5.4 |
| summary known answers (planted per-question hits on 8 tiny pilot-shaped runs) | 10/10 verdicts as planted (incl. CEILING-on-S -> tier E, BORDERLINE s4 CI [7.9, 13.0] vs floor 9.2, dim copy-only = whole held 20.5 / 15.8 but read-out 0 -> FLOOR); K* = 2 on 4 vs 3; tampering caught: curve value, hits string, data_sha, window step, step-0 lm +1e-2, unfinished run, one none seed; passes: step-0 lm +1e-5, seen 50 -> INVALID; --allow_partial -> PROVISIONAL |
| 53M --dry_inputs, all6 K = 1, seeds 0 / 1 | no Traceback; read-out sizes s4 203 / 191, dim 120 / 130, clock wrap 337 / 329; token shares and first-line shares as before; content scan clean; nonce counts as recorded |

## Regression and CPU tests (Rev B e231a219007f vs pinned a3bce3ab63c1; tiny model d64 x 3) (before A1; A1 re-run above)
Pinned = `git archive HEAD tools` (pretrain_range.py sha256 a3bce3ab63c1); Rev B = a frozen copy of the working tree
(e231a219007f). Same args, CPU, OMP / MKL 1 thread, 50 steps, bs 32; per-step total loss (the tensor passed to
backward) and pre-clip gradient norm recorded by a wrapper (steprec.py: patches Tensor.backward / clip_grad_norm_, the
trainer is unchanged); evals every 10 steps (6), --save_model, --dry_inputs run separately.
| config | per-step loss + grad norm (50) | curve (6 evals, all keys but sec) | run JSON (args / keys) | saved model (36 tensors) | --dry_inputs stdout |
|---|---|---|---|---|---|
| c1 win-ctx 2% (helix3, window 0.05,0.4, arith_lines 1, seq 64) | identical | identical | identical | bit-identical | identical (1585 B) |
| c2 none 2% | identical | identical | identical | bit-identical | identical (1558 B) |
| c3 gate-ctx 7% (gate 0.5, arith_lines 4, seq 96) | identical | identical | identical | bit-identical | identical (1703 B) |
| c4 win-text (site text, text_min 4, seq 128) | identical | identical | identical | bit-identical | identical (1846 B) |
| c5 prefix whole helix3_shuf (n_prefix 32) | identical | identical | identical | bit-identical | identical (1664 B) |
Only code_sha (and the out dir) differ. The default path's run JSON omits the 11 Rev B flags (DOM_ARGS), so its args
are key-for-key those of a3bce3ab63c1. Scripts: session scratchpad rainbow6/regtools/ (run_reg.sh, steprec.py,
compare.py), outputs rainbow6/reg/.

Rev B path, CPU (tiny d64 x 3, seq 64, bs 32, K = 2, every domain prior on, window 0.05,0.4, 100 steps):
- runs end to end; geo falls for every active prior while on (e.g. months 0.60 -> 0.21, dim 0.66 -> 0.38 over steps
  10-30); item-state CKA to the true target rises in all6 (months 0.15 -> 0.94, clock 0.08 -> 0.93, s4 0.16 -> 0.48,
  dim 0.07 -> 0.67 by step 40) and stays at 0.01-0.16 in none; all6-shuf keeps cka_months at 0.10-0.16 (its geo falls
  on the shuffled target); only-months: lambda 5 on months only, cka_months 0.15 -> 0.68.
- data identical across arms: data_sha equal at 11/11 evals (none vs all6), step-0 lm 11.525801658630371 in both.
- resume: --ckpt_every 20 --stop_after 29, then restart (resumed at step 20, inside the window): curve (11 evals) and
  saved model bit-identical to the uninterrupted run; no checkpoint left behind.
- eval pipeline known answer (oracle_test.py: build_model patched to a fake model whose logits put all mass on the
  true next token): all 55 eval sets / sub-splits (numbers held / seen / ood, every domain's held / heldE / seen and
  sub-splits) score exactly 1.000; the off-by-one fake (mass on the current token) scores 0.000 on all 55.
- rainbow6_summary.py known answer (fake_test.py: planted none curves on 8 tiny pilot-shaped runs, metrics rewritten
  with the trainer's curve_metrics): all 10 gate verdicts as planted (MEASURABLE / MEASURABLE (tier E) / FLOOR), K* = 1
  on a 4 vs 3 count; tampering is caught (a curve value changed without its metrics -> double-path assert; one arm's
  data_sha changed -> 'data mixture differs'; one window step off -> 'window steps').

## Audit status (experiment-audit, full mode, pre-submission)
i double path: metrics in the trainer vs rainbow6_summary (asserted); tier-E / sub-split read-outs beside the primary.
ii asserts: listed above. iii known answers: CKA = 1, far shuffle, data identity, resume identity, oracle / off-by-one
eval (1.000 / 0.000 on 55 sets), summary on planted results, default-path regression; seen sets high = upper bound on
the real pilot (to be confirmed). iv –. v fresh adversarial review: DONE 2026-10-17 on e231a219007f (findings and
fixes: Amendment A1); a re-review of the small A1 diff is still open. Original note: (to be launched by the
orchestrating session before submission; materials: this file, tools/pretrain_range.py (Rev B block, dom_row, evaluate,
training loop), tools/rainbow6_summary.py, make_jobs_rb6.py, the --dry_inputs logs). vi pitfalls: position (first-line
share logged), leakage (content asserts, clock / numbers overlap handled), silent prior failure (prior_used_frac,
gradient norms), tied embeddings (clock / numbers share rows: caveat). vii --dry_inputs: example lines of every domain
with ⟦ ⟧ sites and « » scored span, held-out eval rows in the training position next to training rows whose first line
is the same domain, token shares, first-line shares (logs in the session scratchpad rainbow6/dry/).

## Amendment A2 (2026-10-17, user decisions after A1, BEFORE any run): binding changes

USER DECISIONS (2026-10-17, after A1): data option 1 = numbers stay at the 2% main setting so the numbers results compare
DIRECTLY with the main experiment (pt999 / pt2b 2% cells); train with fixed simple templates; ADD evaluation-only
held-out-format templates (3-4 paraphrases per domain never seen in training) to measure format robustness; pilot = none
vs all6 with the current best setup (lambda 5 per active prior). Everything below is inside `--domains`; the default path
is still the pinned a3bce3ab63c1 (regression in the A2 test table). Trainer **cc984148b15a** (was 4a657953dfcc; toy_struct
6248613ac6d2 unchanged), summary **f23554a964ee**, Burst generator tools/make_rb6_tasks.py **f3b02b40e4c5** (new).

What changed and why:
| # | change | why | supersedes |
|---|---|---|---|
| 1 | Row layout: text, then the numbers line, ALWAYS first, then the K domain lines in random order (stream 5000 + seed: the order of the domain lines only). The text offset is the default path's draw (stream 2000, the same bound seq - len(numbers line)) and the text is the END of the default path's chunk (its first len(domain lines) tokens dropped). | Causal attention: the numbers line then sees exactly the default path's context (same pair and format from streams 1000 / 2000, which are consumed identically, after the same text, shortened only at the far left), so numbers results compare directly with pt999 / pt2b 2% of the same seed. A1's shuffled order put numbers first in only 51% (K = 1) / 33% (K = 2) of the rows (open question 4 of the build report). | "the 1 + K lines in random order"; "Deliberate differences from pt999 2% ... numbers are NOT comparable" (Data mixture section); first-line shares table |
| 2 | Domain evals (every set: held, heldE, seen) in the training position of slot 1: val text + ONE numbers training line (pair and format drawn as draw_pair does, from the domain's eval-context stream 4100 child 11 + i; for clock a pair stating the question's own addition i + k or i + k mod 24 is redrawn, n_ctx_pairs_redrawn) + the question at the row end. | A domain line now always follows a numbers line; the eval must match. K = 2: a domain line sits in slot 1 or 2 about half the time each (slot_frac logged; the eval is slot 1). | "each placed at the END of a 512-token row right after a val-text chunk = the training position of a first line" (Metrics) |
| 3 | Held-out-FORMAT evals (`PARA` in the trainer): months / weekdays / clock 4 templates, s4 / dim 3 (dim: a times / per pair each), e.g. 'What month is 5 months after March? August', 'Date math: March + 5 months = August', 'Starting from March, count forward 5 months: August', 'If it is March now, in 5 months it will be August'; clock '5 hours after 9:00 the time is 14:00', ...; s4 ' gez followed by cuk gives nok', 'Apply gez and then cuk to get nok', 'Combining gez with cuk yields nok'; dim ' zaj multiplied by / divided by roz gives ...', 'Multiply / Divide zaj by roz: ...', ' zaj x / roz = ...'. Scored on the SAME questions (held = tier S with every sub-split mask, and seen) and the SAME contexts (text end + numbers line) as the training template, same exact-match rule (answer through the newline, teacher forced, argmax), same position. Keys `<d>_p<t>_<set>` (+ sub-splits, hits_), metrics `<d>_p<t>_readout` / `<d>_p<t>_seen`, DINFO formats / eval_p<t>_<set>. | User decision: measure format robustness. Item tokens and the scored span are the training line's tokens (asserted per question), so a drop is the template, not the tokens. | — (new, descriptive; no gate) |
| 4 | Leak asserts for 3 (build time): no training line (every domain's pool, the 4 numbers formats) is an instance of any template (slots as wildcards); no rendered paraphrase prompt occurs in a training line (string and token level); every paraphrase prompt differs from the training prompt; --dry_inputs scans 640 training rows for every paraphrase prompt (lines: assert; text: counted). | experiment-audit vi (leakage) | — |
| 5 | lambda rule: the pilot's all6 = `--lam_rule each` (lambda 5 on every active prior = the best single-domain setup); `norm` stays selectable (make_jobs_rb6.py LAM_RULE / make_rb6_tasks.py W). | User decision (A1 finding 2 resolved). Consequences: the numbers prior gets lambda 5 = the main recipe (pt2b / pt999 win-ctx), so the all6 numbers cell compares with win-ctx@1e-3; the total prior weight in all6 is 30 (circle priors pull 5-8x the helix at equal weight, CPU smoke) - the LM cost (bpb) and grad_prior / grad_lm are the things to watch; R3's caveat (only-d lambda 5 vs all6 5/6) is gone: every arm has lambda 5 per active prior. | "lambda rule (--lam_rule norm, used here)"; R3 caveat |
| 6 | Burst: tools/make_rb6_tasks.py, tasks 400-407 (408-409 for re-runs), tag rb6pb-, pinned at PIN (to be set after the commit), DRY row hash per (K, seed) computed on Torch, step-0 lm (CPU) and data_sha after step 0 and the split sizes per seed as references, intent check script (prior on / applied / CKA rising for all6, never on for none, layout, format sets), `# time:` 02:00:00 = 2 x 0.72 A100-h x 1.2. Torch: make_jobs_rb6.py (8 job files re-rendered with the A2 sha). | User: the pilot can run on either site; one site per summary (the tags differ). | — |

Binding pilot rule: unchanged from A1 (rules 1-5 as amended there); the gate still reads only the none arm's TRAINING-
template read-out sets (now in the slot-1 position). The held-out-format evals are descriptive: no gate, no pre-registered
claim in the pilot (a main-design rule would be added before the main design, after the user sees the pilot).

Numbers comparability (A2, measured on CPU, see the test table): for the same seed, every Rev B training row = the default
path's row with its first L text tokens removed and the K domain lines (L tokens) appended; the numbers eval inputs are
byte-identical to the default path's. What still differs from pt999 / pt2b 2%: (a) the text before the numbers line is
shorter by L (K = 1: 6-13, mean 7.8; K = 2: 12-20, mean 15.6 tokens), so the numbers line sits L tokens earlier (absolute position; RoPE
attention is relative); (b) the row continues after the numbers line with domain lines (the numbers line cannot see
them; they add LM loss on their own tokens); (c) all6: five more priors on (lambda 5 each); (d) the domain training
lines share the number tokens (clock hours and every shift k, months / weekdays k) with the numbers operands (A1 R4
caveat). The comparable main-setting cells: none <-> pt999 / pt2b 2pct none@1e-3; all6's numbers <-> win-ctx@1e-3.

Expected magnitudes, revised before any run: numbers none area ~ pt999 2pct-none (9.4) if the domain lines do not interfere
(the build report expected 'similar or lower' because numbers came first in only 1/(1+K) rows: that cause is gone); all6
numbers vs pt2b win-ctx@1e-3 (area 70.2 in make_pt2main_tasks.py's open-bracket note of 2026-10-17; take the final
pt2main_summary value): lower would mean interference from the other priors / lines. Held-out formats: below the training template on held-out questions (typical format
brittleness of small LMs); the seen sets in a paraphrase show whether memorised facts transfer to a new template at all.

A2 tests (trainer cc984148b15a, summary f23554a964ee, generator f3b02b40e4c5; CPU, 1 thread per run; scripts and logs in the session
scratchpad rainbow6/fix2/: run_ident.sh + numbers_identity.py, run_reg.sh, run_smoke.sh + smoke_check.py, oracle/oracle_test3.py,
run_sumtest.sh + fake_test3.py, check_test.py, dry_hash_rb6.py, step0.py, run_all.out):
| test | result |
|---|---|
| numbers-line identity (capture.py: every model forward of the default path at HEAD a3bce3ab63c1 vs Rev B none / all6, tiny model, seq 128, 20 steps, seeds 0 / 1, K = 1 / 2) | PASS 6/6: the step-0 numbers eval forwards (15 sets + 8 val-loss batches, 743 rows) byte-identical; 640/640 training rows per run = the default row minus its first L text tokens + the K domain lines (L tokens); L = 6-13 (mean 7.8) at K = 1, 12-20 (mean 15.6) at K = 2 (line lengths are fixed per domain: months / weekdays 7, clock 13, s4 / dim 6 tokens); all6 training inputs = none's |
| default-path regression vs pinned a3bce3ab63c1 (builder c1-c5 + auditor a1-a2; per-step loss + pre-clip grad norm, curve, JSON keys / args, model, --dry_inputs bytes) | ALL IDENTICAL 7/7 (c1 11.526635 -> 10.184761, c3 16.167389 -> 11.918018, a1 11.518012 -> 10.217460, a2 11.518012 -> 9.525920; as A1) |
| multi-domain smoke (tiny, K = 2, lam_rule each: A all6 100 steps, B = A stopped at 29 and resumed at 20, C none, D all6-shuf, E only-months) | resume B == A (11 evals + model bit-identical, no checkpoint left); data_sha A = C = D = E on the common evals, step-0 lm 11.53638744354248 in all; every prior applied on 100% of the on-steps; lambda 5 per active prior; numbers first in every row, slot shares 0.48-0.52; 36 held-out-format sets at every eval; all6-shuf months true-target CKA <= 0.17. Domain priors take hold (CKA step 0 -> window end, step 30): months 0.17 -> 0.77, weekdays 0.25 -> 0.79, clock 0.09 -> 0.71, s4 0.16 -> 0.38, dim 0.08 -> 0.57 (none: max 0.20 / 0.56 / 0.16 / 0.16 / 0.08). The numbers prior is applied (weighted grad 11-13 at every on-eval, geo 0.84 -> 0.79 while on) but its operand CKA rises only 0.16 -> 0.21 in the window (0.41 at step 99 vs none 0.28; A1 norm smoke 0.19 -> 0.35; numbers-only window at this scale +0.06 in 8 on-steps): the smoke's own >= +0.1 criterion FAILS for numbers only. Reading: the five domain terms carry ~4/5 of the prior gradient (total 58-70 vs LM 1.1 before clipping) - the lambda-balance risk of A1 finding 2, now with total prior weight 30; not a code fault (to the user; the 53M Burst check asserts a +0.1 rise for numbers too) |
| oracle / off-by-one fake model (all sets incl. the held-out formats) | ORACLE TEST PASS x 2: 175 eval sets / sub-splits (111 of them held-out-format) score 1.000 / 0.000; 49 per-question hit strings = contexts / 0 |
| summary known answers (fake_test3.py: 8 tiny pilot-shaped runs, planted per-question hits on the none training-template sets and on every held-out-format set of both arms; metrics recomputed with the trainer's curve_metrics) | FAKE TEST PASS: 10/10 gate verdicts as planted (as A1), K* = 2; format table = planted round(p x contexts) / contexts for every (K, arm, seed, domain, template); all6 - none per template as planted (K2 months 37.5 / 37.5 / 50.0 / 25.0); mean drop 6.25 as planted; seen paraphrases as planted; 8/8 tampers fail: curve, data, window, unfinished, one none seed (A1) and paraphrase hit string, paraphrase metric, numbers not first (A2) |
| Burst intent check (make_rb6_tasks.CHECK_PY run on the smoke runs A / C with the tiny run's facts substituted: model size, steps, window, its own step-0 lm / data hash / splits, token-share bound 15-19% at seq 64, CKA-rise bar 0.05 because of the tiny-scale numbers prior above; 53M tasks keep 0.1) | CHECK SCRIPT TEST PASS: all6 21/21 and none 15/15 checks pass; 18/18 tampers fail exactly the intended check (data hash, step-0 lm + 2e-3, numbers not first, a format set missing, split size, lambda, an arg; all6: a prior never applied, CKA not rising, a domain off in the window; none: a prior on) |
| 53M Torch references for the Burst tasks (CPU, Torch data = Burst's by sha256) | method check: the pinned 64abc19 dry run reproduces pt2main's recorded DRY (pretrain_data, 53m, ctx, 0) = 379354e2... MATCH; CPU step-0 lm of the default path seed 0 11.616469 vs the known GPU value 11.616473 (pt999 2pct-none; diff 3.8e-6 << the 1e-3 tolerance). rb6 DRY (sha256 of the DRY_GREP lines; 83 lines at K = 1, 88 at K = 2) identical for none and all6 at every (K, seed); the task's bash pipeline (grep -E | sha256sum) reproduces them; split facts identical across K and arms; step-0 lm / data_sha after step 0: K1 s0 11.615037 / 5d408328037e094e (none = all6), K2 s0 11.614819 / 28c9fff8ef37e82f, K1 s1 11.602010 / 3f8eefe500da5bec, K2 s1 11.603225 / dfac64fb1b4a4f36. 53M --dry_inputs (all 8): no traceback; token share numbers 2.124-2.126% (K = 1 text 96.35%, K = 2 94.83%); numbers first line 1.0; K = 2 slot shares 0.47-0.53; content scan of 640 rows: 0 held-out prompts and 0 of 6171 held-out-format prompts in lines or text; clock eval contexts redrawn 1 (seed 0) / 2 (seed 1); nonce counts as recorded. Run JSON size at 75 evals ~1.4 MB (< the 5 MB push limit) |
Audit status (A2, experiment-audit): i double path (trainer metrics vs summary, per-question hits vs accuracies) extended
to every held-out-format set; ii asserts as listed in the A2 table; iii known answers above (identity to the default path,
regression, oracle / off-by-one, planted summary, planted check script, the known pt2main DRY value and step-0 lm); vii
decoded rows in the 53M --dry_inputs logs (session scratchpad rainbow6/fix2/refs/k*_s*_*.log). OPEN: v - a fresh
adversarial review of the A2 diff (pretrain_range.py cc984148b15a vs 4a657953dfcc, rainbow6_summary.py, make_rb6_tasks.py)
before submission; and the user's call on the tiny-scale numbers-prior weakness under lam_rule each (A2 test table, smoke).

## Amendment A3 (2026-10-18, fixes after the fresh adversarial review of A2, BEFORE any run)

Audit item v (fresh adversarial review) is now DONE on A2 (trainer cc984148b15a, summary f23554a964ee, generator
f3b02b40e4c5): no blocking code bug; the default path re-checked bit-identical (9/9 configs incl. two new ones), numbers
identity 6/6, a semantic answer check of every eval row (0 problems), oracle / off-by-one 1.000 / 0.000 on 175 sets, the
DRY hashes and the Burst plumbing reproduced. It found two over-claims in the write-up and two labelling gaps. A3 fixes
them in the SUMMARY only (rainbow6_summary.py f23554a964ee -> **05599d7f59a7**); the trainer (cc984148b15a), toy_struct
(6248613ac6d2) and the Burst generator (f3b02b40e4c5, its DRY / step-0 / split references) are unchanged, so every
trainer test of A2 (identity, default-path regression, smoke, oracle, 53M references) stands as recorded. The pilot
rule and gate are unchanged (they read only the none arm's training-template sets).

| # | finding (severity) | real? | fix |
|---|---|---|---|
| 1 | global clip (MEDIUM-HIGH): `clip_grad_norm_(.., 1.0)` acts on the TOTAL gradient (LM + every active prior), so all6's numbers prior does not get win-ctx's step; "all6 numbers <-> win-ctx@1e-3" is not a direct comparison and "lower would mean interference" (A2 'Expected magnitudes') over-reads it | yes. Main setting (real run JSONs, the summary's own clip_stats): pt2b / pt999 win-ctx@1e-3 median pre-clip T 0.61-0.68 per seed, T > 1 on 2/26 on-evals (one seed 1/26), clip factor 1.000, numbers prior after the clip 0.21-0.30 (0.34-0.44 of T), LM after the clip 0.56-0.60. CPU tiny all6 (each): T 69-93, c 0.011-0.015, numbers prior 0.10-0.15 of T; tiny win-ctx alone also T 10.4, so the tiny scale does not predict 53M. Mechanism note: AdamW (betas 0.9 / 0.95) normalises per parameter, so a slowly varying global factor largely cancels; what the clip does not undo is the MIX, the numbers prior's share of the update, for which numbers / T is a rough proxy | numbers table per prior arm and reference, per seed: T, c = 1/max(1, T), share of on-evals with T > 1, numbers prior after the clip and / T, LM after the clip (double path: the reconstruction from grad_lm / grad_prior / grad_cos vs the value clip_grad_norm_ returned, every eval step of 4 tiny runs, max relative difference 1.8e-3). Wording: an all6 - win-ctx difference is the joint effect of the domain lines, five more priors and the shared clip, NOT by itself interference (supersedes the A2 sentence). lam_rule each kept (user decision); options are for the user, not changed here |
| 2 | numbers buckets b11 / b12 / b22 not clean vs pt2b / pt999 (MEDIUM) | yes: clock k = 1..47, hours 0..23; lines with h + k < 24 ('7 hours after 9:00 is 16:00') are extra addition statements of b11 / b12 / b22 (builder's captures: +14% / +13% b11 / b12 at K = 1, +19% / +32% / +3% at K = 2), and clock / months / weekdays lines contain the number tokens 0..47 (the clock prior acts on 0..23) | the numbers table gives the area over all 6 buckets, b11/12/22, b13/23/33 (no domain line states an addition with a 3-digit operand) and b33 (no domain line or prior touches either operand), for rb6 and every --ref cell, paired by seed. Measured on the references: none@1e-3 b13/23/33 2.5-3.4, b33 0.6-1.1 = at the floor, so the clean groups separate only the PRIOR arms (win-ctx 51-60 / 28-49); the none-arm comparison rests on the contaminated small buckets. Yardstick (one setting, two runs: pt999 - pt2b at seeds 0-3): none all6 area +3.5 / -1.0 / +0.4 / -2.4, win-ctx -2.1 / +4.5 / +4.1 / -0.4, win-ctx b33 -6.5 / +19.2 / +5.1 / +2.2: the summary prints this ref - ref line whenever two references of one kind are given; b33 is clean but noisy |
| 3 | p1 templates are near-paraphrases (LOW-MED): months / weekdays / clock p1 contain the training prompt up to the last input slot ('k months after X', clock differs only by ' the time') | yes (template check on PARA and on the 53M decoded training lines: p1 of the three cyclic domains, nothing else). The leak asserts' direction (no paraphrase inside a training line) is right; the reverse is the definition of near | NEAR computed in the summary (TRAIN_PROMPT, the training prompt to the last input slot, contained verbatim), marked * in the format tables, JSON near flags; drop over all templates AND over the far ones |
| 4 | lr 1e-3 is the prior arm's best, not none's (LOW, user) | yes: pt2b win-ctx 70.2 @1e-3 / 69.9 @2e-3 / 64.2 @4e-3; none 9.9 @1e-3 / 14.2 @2e-3 / 19.8 @8e-3 | not changed (comparability with pt2b / pt999 and one lr per paired comparison). When the gate is read: a FLOOR at 1e-3 may partly reflect an lr that is low for the none arm |
| 5 | numbers eval at the row end vs training rows with 6-20 domain-line tokens after the numbers line (LOW) | yes, negligible (causal attention; RoPE; the immediate context identical); already A2 (a) / (b) | none |
| 6 | the pilot does not start on the push (LOW) | yes: burst/config.env SENTINEL=off; a running task's tick (every 90 s and at its end) submits new tasks, otherwise one kick on Burst (`bash /scratch/$USER/llmadd/burst/tick_locked.sh`); MAX_GPU_JOBS 12, earlier pt2main tasks still queued | the launch instructions say so (no code change) |

A3 tests (summary 05599d7f59a7; CPU; scripts and logs in the session scratchpad rainbow6/fix3/):
| test | result |
|---|---|
| summary known answers (fake_test4.py = fake_test3.py + A3: the 8 tiny pilot-shaped runs + 4 tiny default-path reference runs with the same args and a second none reference under another tag; planted bucket accuracies with a seed offset, planted gradients at the on-evals) | FAKE TEST PASS (fake_test4_final.out): A1 / A2 checks as before (10/10 gate verdicts, format table, all6 - none per template, drop 6.25), and A3: bucket-group areas of 7 cells x 2 seeds x 4 groups, 7 paired-by-seed lines (no none-vs-prior cross pairs), the ref - ref yardstick, clip stats recomputed with plain loops, near flags, far drop 4.1667 as planted; 13/13 tampers fail (A1 / A2's 8 + a reference at another lr, an rb6 run as reference, an unfinished reference, another window, two references of one (tag, seed)) |
| clip double path (clipcmp.py: steprec.py records what clip_grad_norm_ returns at every step of the frozen cc984148b15a; tiny all6 K1 / K2, none K2, default-path win-ctx) | PASS: 30 eval steps, reconstructed T within 1.8e-3 (relative) of the true pre-clip norm, on and off the window |
| real references (real_refs.py: num_area / clip_stats on pt2b 350-353 and pt999 2pct, vs an independent per-bucket / plain-loop recomputation) | match; the values in the table above |
| TRAIN_PROMPT / near (tpl_check.py: 80 decoded 53M training lines, 16 per domain; is_near on pretrain_range.PARA) | PASS |

Read-out command (Burst, after the 8 tasks): `python tools/rainbow6_summary.py results/40[0-7]-rb6pb-* --ref
'results/35[0-3]-pt2b-lr1e-3-none-s?' --ref 'results/35[0-3]-pt2b-lr1e-3-win-ctx-s?' --json rb6pb_summary.json`
(add the Torch pt999 2pct-none / 2pct-win-ctx run JSONs as further --ref for the yardstick line).
Audit status (A3 diff, experiment-audit): i double path (reconstructed T vs clip_grad_norm_; group areas vs a per-bucket
recomputation) PASS; ii comparability / finished / one-run-per-seed / no --domains asserts on every reference; iii planted
known answers + 13 tampers PASS; iv -; v OPEN: a fresh general-purpose review of the A3 diff (no subagent tool in the
fixing session; descriptive tables only, so it may run in the background after the push); vi trap 3 (clip) is what A3
reports, trap 8: the new summary on the A2 test input leaves every existing JSON section and table identical; vii - (no
input change). v on A2 DONE (findings above). OPEN for the
user: the numbers prior's share under lam_rule each (finding 1; the 53M value is unmeasured until the pilot runs; the
Burst check fails a task whose numbers CKA rises < 0.1), and the lr caveat of finding 4.

## Amendment A4 (2026-10-18, the user's pilot extension of 2026-10-17, BEFORE any run)

USER DECISIONS (2026-10-17): the pilot is K in {1, 2, 5} domain lines per row (K = 5 = one line of every domain in every
row, i.e. as many lines of each domain as numbers lines) x arms {none, all6 at lambda 5 per prior, all6 at lambda 1 per
prior} x seeds {0, 1} = 18 Burst tasks. The trainer (cc984148b15a, committed at c434d29) and toy_struct (6248613ac6d2)
are unchanged; the summary changes only in its printed tie rule (05599d7f59a7 -> **c32891cace9a**, audit finding 2: its
JSON output is byte-identical); the Burst generator changes: tools/make_rb6_tasks.py f3b02b40e4c5 -> cfda13a74a83 (built)
-> **1bbb2244206a** (after the audit: comments only, the 18 rendered task files byte-identical to cfda13a74a83's; PIN
still "", to be set to c434d29 at the commit of the generator). Torch's make_jobs_rb6.py (the 8 L40S jobs of K 1, 2) is
not extended: the pilot runs on Burst.

What changed (generator only):
| # | change | why / check |
|---|---|---|
| 1 | KS = (1, 2, 5); ARMS = none, all6, **all6-lam1** = all6 + `--lam 1` (lam_rule each, so lambda 1.0 on each of the 6 priors; total prior weight 6 vs 30) | user decision. Intent per arm: lam_domains = 0.0 / 5.0 / 1.0 for every domain, args.lam 5.0 / 5.0 / 1.0, lam_rule norm / each / each; the rendered trainer arguments are asserted flag by flag (every other flag = its default) |
| 2 | numbering 400-417, spares 418-429 (re-runs): **400-405 none** (NNN = 400 + 3 seed + i_K), **406-411 seed 0** of the two prior arms, **412-417 seed 1** (NNN = 406 + 6 seed + 2 i_K + (0 all6, 1 all6-lam1)); tag rb6pb-53m-2pct-k<K>-<arm> | cheap-and-decisive first: the tick submits ready tasks in name order with MAX_GPU_JOBS = 12, so the first wave of 12 is the whole none arm (the binding gate needs both none seeds at every K) + one seed of every prior cell (a first lambda 5 vs 1 look at every K); the second wave (6) completes the paired lambda read-out. The waves are asserted in main(). Nothing was rendered under the A2 numbers (PIN was never set), so no task name is reused |
| 3 | input check before training (new, every task): the 'domains:' line of --dry_inputs (the trainer's DINFO) must show exactly 6 x `"lam": <lambda>` and 6 x `"prior": true` (false for none), else exit 3 | the DRY hash does not contain lambda (the DRY lines are identical for the three arms: checked), so a wrong lambda would otherwise surface only in the run JSON check after ~0.9 GPU-h |
| 4 | CHECK_PY slot check: every slot within 1/K +- 0.05 of every domain's lines (K = 2: 45-55% as before; K = 5: 15-25%; K = 1: 1.0); check info adds lam_domains | K = 5 has 5 slots after the numbers line |
| 5 | CKA-rise criterion for all6-lam1: the **same +0.1** (item-state CKA to the true target, step 0 -> the window's last eval, every prior incl. the numbers operands) | a sanity bar, not a prior-strength test (corrected after the audit, finding 3). 53M references of the numbers prior ALONE (one prior, its own clip; cka_ops_helix3 at step 7250 = the window's last eval, minus step 0, seeds 0-3; extend/fixA4/cka_ref.py): lambda 5 at lr 1e-3 (results 350-353 win-ctx) +0.75..+0.79; lambda 0.3 / 1 / 2 at lr 2e-3 (374-377) and 5 at lr 2e-3 (350-353) +0.75..+0.79 alike (numbers area 68.4 at lambda 1 vs 69.9 at 5); the **none arm at lr 1e-3 also rises +0.24..+0.44** by step 7250. So for numbers the bar sits below what no prior gives (it catches a broken prior path only; the other prior checks: applied >= 95%, geo > 0, finite grad_prior). No 53M reference exists for six priors sharing one clip (at any lambda) or for the five domain priors at lambda 1: a CKA-only FAIL (every other check PASS) is a finding about that prior's strength, read against the same-seed none run's cka_<d> at the same eval (every run JSON's curve has it; the summary's sanity table prints the prior arms' window-end CKA), not a broken run (the tick publishes results/ on any exit code). Tiny CPU smoke (check_test5 run A5, lambda 1, K 5): rises months +0.77, weekdays +0.68, clock +0.80, s4 +0.35, dim +0.73, numbers +0.29 (tiny run A, lambda 5, K 2, 35-step window: numbers +0.06; tiny scale only) |
| 6 | `# time:` 02:00:00 for every task; PLAN_H per K (K = 5: 0.867 A100-h = 0.864 x (1 + 0.5 ms / 0.1416 s); 2 x 0.867 = 1.73 h < 02:00:00) | the eval sets are the same rows at every K (below), so the eval overhead is unchanged; the only K-dependent per-step cost is the CPU loop between two steps, measured below |

K = 5 Torch references (same method as A2: CPU --dry_inputs / step-0 runs of the PINNED tools = git archive c434d29
tools, on the Torch data = Burst's by sha256; the task's own trainer arguments and tag; scripts and logs in the session
scratchpad rainbow6/extend/: dry_ref.py, step0.py, run_repro.sh, run_k5a.sh, run_k5b.sh, fill_k5.py):
| item | value |
|---|---|
| reproduction first (K 1, seed 0, none, task tag rb6pb-53m-2pct-k1-none) | DRY 2106164b...82e39, 83 lines = recorded; step-0 lm 11.615036964416504 = recorded (bitwise); data_sha after step 0 5d408328037e094e = recorded; data sha256 train / val = DATA |
| DRY (K 5) | seed 0: 5563b8a6dad867de82dbdedac115eef8698da02ac4bde86e20740044964f342c, 103 lines; seed 1: 9828d29a689ed9a2f0964f3e23a9a654055b01538ee4fb76ba214295e5c2525f, 103 lines (83 + 20: 25 vs 5 slot rows); identical for none, all6 and all6-lam1 (and all6-lam1 at K 1 seed 0 = the recorded K 1 hash) |
| LM0 / DSHA0 (K 5) | seed 0: 11.61912727355957 / b595b5e7f178a592; seed 1: 11.60159683227539 / 420fa2e9a9c1c248; all6 and all6-lam1 at K 5 seed 0 = none's (bitwise) |
| DOMS | K-independent: the split facts of every K 5 dry run = DOMS[seed] (A2 values); no K entry needed |
| eval contexts | the 36 '  eval ' lines of --dry_inputs (every domain set and held-out format, with their text + numbers-line contexts) are byte-identical at K 1, 2, 5 for each seed: the domain evals do not depend on K (eval stream 4100 children 11-15) |
| token share (50 batches) | text 90.26 / 90.26, **numbers 2.124 / 2.126** (K 1: 2.124, K 2: 2.124), months 1.367, weekdays 1.367, clock 2.539, s4 1.172, dim 1.172 % (seeds 0 / 1); domain lines 7.62% of all tokens (K 1: 1.52%, K 2: 3.05%) |
| first line / slots | numbers first in every row (1.0); each domain once per row, slot shares 0.181-0.224 (seed 0) / 0.182-0.214 (seed 1) over 50 batches (the run's cumulative value over 585,920 rows will be ~0.200 +- 0.001) |
| layout | the 5 domain lines are 39 tokens (months / weekdays 7, clock 13, s4 / dim 6), so the numbers line sits 39 tokens earlier than in the default path (K 1: 6-13, K 2: 12-20) after a text 39 tokens shorter (e.g. 461 vs 486 tokens at K 2); the numbers line itself = the default path's (same pair, format and text end: train row 0 is '643 + 287 = 930' at every K) |
| no-leak scans | content scan of 640 training rows, both seeds: 0 held-out prompts and 0 of 6171 held-out-format prompts in the lines or the text; nonce counts as recorded |
| exposure per run (seed 0) | 585,920 lines per domain (= numbers lines): each training question ~4,965 x (months), 15,419 x (weekdays), 1,216 x (clock), 1,698 x (s4), 1,703 x (dim); slot 1 (the eval position) gets 1/5 of them = 117,184 per domain = K 1's total |
| run JSON size | K-dependent part measured on tiny CPU runs with the same eval settings (none, K 2 vs K 5): +216 B per eval (slot_frac 5 entries, token shares); 53M: A2's ~1.4 MB at 75 evals + 75 x 216 B = ~1.42 MB (limit 5 MB, effective 4 MiB) |
| time | eval sets K-independent (eval rows above), so the A2 estimate holds: 0.72 A100-h x 1.2 = 0.86 h. The K-dependent CPU work between two steps, measured (capture_time.py: the unchanged trainer, tiny model, CPU, the gap from the end of one optimizer step to the next training forward, 39 steps each): median 7.38 ms (K 1), 6.99 (K 2), 7.58 / 7.79 (K 5, two runs), so K 5 - K 1 <= 0.5 ms per step vs 0.1416 s per A100 step (0.72 h / 18,310 steps): +0.35% at most -> PLAN_H 0.867 h, limit 02:00:00 (x 2.3). Not measured on a GPU (no GPU here) |
| clock additions in the numbers buckets (A3 finding 2 at K 5) | measured in the captured training rows (1,280 rows per K, the same numbers lines at every K): clock lines that are plain additions (k < 24, h + k < 24) / numbers lines of the bucket = b11 +10% / +20% / **+47%**, b12 +18% / +36% / **+81%**, b22 +1% / +3% / +7% at K 1 / 2 / 5 (A3 measured +14 / +13% at K 1, +19 / +32 / +3% at K 2 on 640 rows): linear in K |

Pilot decision rule, extended (binding; replaces A1 rule 2, everything else of A1 rules 1-5 unchanged; written before any
result):
1. Gate per K in {1, 2, 5}: A1 rule 1 on the none arm only (2 seeds per K, the same seeds at every K): MEASURABLE /
   FLOOR / CEILING / BORDERLINE / INVALID per domain on its read-out set, tier-E fallback for the cyclic domains.
2. **K\*** = the K with the most MEASURABLE (incl. tier E) non-number domains on the none arm; **ties -> the smaller
   synthetic share** (= the smaller K: domain lines 1.52 / 3.05 / 7.62% of the tokens at K 1 / 2 / 5); BORDERLINE counts
   as not measurable and is listed. (rainbow6_summary.py: max over K of (count, -K); planted test below.) The all6 arms
   never select K.
3. **lambda\*** (descriptive; the final choice is the user's): at K\*, for each lambda in {1, 5}, per domain (numbers and
   the five) the paired-by-seed all6-lambda - none difference of the read-out final3 and area (summary 'paired' and
   'all6 vs the same bounds' tables, room verdict per A1 rule 4), and the LM cost: offline bpb (offline_lm.py, full val;
   check info bpb_full) of the prior arm - none per seed, plus the in-training Delta bpb; next to it the sanity list
   (every prior applied >= 95% of the on-steps, CKA rise >= 0.1, no non-finite value, seen sets >= 80) and the A3 clip
   numbers (T, clip factor, numbers prior / T) of both lambdas. Reported per lambda: the domains with a gain on both
   seeds, the mean gain per domain, Delta bpb (A1 R5 flag: > +0.004). No pre-registered winner; with 2 seeds a lambda
   difference inside the seed-to-seed spread (numbers yardstick +-4.5 area points, A3) is reported as no difference. The
   other K are shown, not used. The main design is then generated at (K\*, lambda\*) after the user's choice.
4. Integrity (A1 rule 5) per K and arm: one code_sha, finished runs, identical data across the three arms of a (K, seed)
   (data_sha at every eval, step-0 lm), exact window.

Caveats added by A4 (to read with the results):
- Clock lines with h + k < 24 are extra small additions (A3 finding 2); at K 5 every row has one clock line, so the
  numbers buckets b11 / b12 / b22 get about 5x the K 1 and 2.5x the K 2 extra statements (measured above). The numbers
  read-out vs the main setting at K 5 rests even more on b13 / b23 / b33 and b33 (A3 table); the none-arm numbers
  comparison across K mixes this with the shorter text.
- Eval layout x K (rewritten after the audit, finding 1; checked in the K 1 / K 5 dry logs). The domain evals use ONE
  layout at every K (the eval rows are byte-identical at K 1, 2, 5): text 487-497 tokens (held-out formats 479-495) +
  one numbers line + the question ending at the row end (scored 508-511; K 1's training rows: text 487-495, the domain
  line ending at 511). That context (text + numbers line + the domain line) is slot 1 of
  training at EVERY K, with the same count per domain (117,184 = K 1's total; at K 2 / 5 it is 1/2 / 1/5 of the domain's
  lines, the other lines sit in slots 2..K after other domain lines, extra exposure). The model is OLMo-2 (RoPE, no
  absolute position embedding), so what still differs from K 5's slot-1 training rows is (a) the text before the numbers
  line (K 5: 461-462 tokens vs the eval's 487-497, i.e. 25-35 more tokens at the far left) and (b) the absolute index of
  the question (K 5's slot-1 line ends ~30 tokens earlier). The eval is therefore not off-distribution at K 5 by
  construction, and a K difference on the none arm is the joint effect of the K - 1 extra lines per row (more exposure
  in slots 2..K, more interference, more synthetic share) and the shorter text; the none seen sets per K (printed in the
  gate lines; trained questions, the same slot-1 exposure at every K and K x the total) are the in-layout check: a
  K 5 seen drop is interference or layout, not held-out generalisation. Residual (a)/(b) effects are not measured: no eval places
  a question in slots 2..K, and adding one is a trainer change (the trainer is pinned). K\* follows rule 2 exactly as
  written (binding); whether that is acceptable is put to the user before results (Status line).
- The text share falls to 90.3% at K 5 (96.4% at K 1): Delta bpb is always read against the same K's none.
- all6-lam1's numbers rows in the summary are paired with the lambda-5 win-ctx@1e-3 reference (no lambda-1 reference at
  lr 1e-3 exists; the summary's --ref asserts lambda 5): read them as all6-lam1 vs the lambda-5 main recipe; numbers-only
  at lr 2e-3 differs by ~1.5 area points between lambda 1 and 5 (above).
- GPU budget: 18 tasks x 02:00:00 = 36 GPU-h of limits (the tick commits the full limit of queued tasks: 24 h for the
  first wave of 12); expected use 18 x ~0.87 = ~16 A100-h.

A4 tests (CPU; scratchpad rainbow6/extend/):
| test | result |
|---|---|
| generator `--check` | "18 tasks: 18 to write, 0 unchanged, 0 differ (kept), 0 already ran"; every flag of every task asserted; waves asserted |
| render (render_test4.py; PIN placeholder, files only in the scratchpad) | bash -n 18/18; the task's own input-check block (model line, target, lambda / prior of the 6 domains, numbers first, content scan, DRY count + sha256, pinned trainer sha) against the Torch dry log of its (K, seed, arm): 18/18 exit 0 (15 real logs; all6-lam1 at K 1 seed 1 and K 2 seeds 0, 1 = the all6 log with lambda 5.0 -> 1.0, synthetic); 90/90 cross cases (another arm, K or seed) exit 3 for the intended reason; the 8 A2 cells change only in their numbers, header comment, the new lambda check, the slot check and one info field (intent JSON identical) |
| CHECK_PY self-test (check_test5.py) | CHECK SCRIPT TEST PASS: real tiny runs A (all6 lambda 5, K 2), C (none, K 2), A5 (all6-lam1, K 5), C5 (none, K 5) and SYNTH (A's JSON with lambda 1 in args and lam_domains = a synthetic all6-lam1 run at K 2) pass every check (21 / 15 / 21 / 15 / 21); 62 tampers fail exactly the intended check (incl. new: slot share off by 0.08, a slot list of another K, the other lambda's lam_domains, the other lambda's args.lam); a whole run of the other lambda arm fails exactly the args and lambda checks (3/3) |
| summary on 3 K x 3 arms (sum3k_test.py; summary 05599d7f59a7 unchanged) | SUMMARY 3-K TEST PASS: 12 tiny runs (fix2 sumtest K 1, 2 none / all6 + K 5 none / all6-lam1, seeds 0, 1), planted measurable counts -> K\*: (2, 3, 4) -> 5, (3, 4, 4) -> 2 (tie -> the smaller K), (3, 3, 3) -> 1; the summary reports K [1, 2, 5], arms all6 / all6-lam1 / none, the K 5 all6-lam1 rows of the paired, range, format, sanity and numbers tables, numbers lambda [1.0] for all6-lam1 and [5.0] for all6, the K 5 slot shares |
| re-run after the audit fixes (extend/fixA4/; generator 1bbb2244206a, summary c32891cace9a) | `--check` "18 tasks: 18 to write, 0 unchanged, 0 differ (kept), 0 already ran"; RENDER TEST PASS (18/18 input blocks exit 0, 90/90 cross cases exit 3, bash -n 18/18; the 18 rendered files byte-identical to the cfda13a74a83 render); CHECK SCRIPT TEST PASS (A / C / A5 / C5 / SYNTH 21 / 15 / 21 / 15 / 21 checks pass, 62/62 tampers, 3/3 whole runs of the other lambda); SUMMARY 3-K TEST PASS (sum3k_test_fix.py: K\* 5 / 2 / 1 as planted, the printed line now says 'tie -> the smaller K' and no 'tie -> 1' is printed; the three summary JSONs byte-identical to the 05599d7f59a7 run, one stdout line differs per case) |

Audit status (A4, experiment-audit): i double path: unchanged (summary); the K 5 references by two routes (DRY of three
arms; step-0 of three arms; eval rows of three K); ii asserts: lambda / prior at the input stage, the waves, the flags;
iii known answers: the reproduced A2 references, planted K\* cases, tampers; vi pitfalls: clock contamination x K,
eval-position share x K, the lambda-5 reference for the lambda-1 numbers rows; vii decoded K 5 rows in the dry logs
(rainbow6/extend/refs/k5/*.log); v: DONE - an independent auditor (2026-10-18, its own dry runs of tasks 402 / 417 on
git archive c434d29 tools, its own step-0 route by a forward hook on task 417, its own rendering of the input blocks;
files in rainbow6/extend/audit/) found no blocking issue; its findings and what was done:
| # | finding (auditor's grade) | verified? | fix |
|---|---|---|---|
| 1 | the binding K\* gate favours small K by construction: one eval layout (K 1's) at every K, only 1/5 of a domain's lines in slot 1 at K 5; A4 called the rule binding but said to read K\* "with that in mind" (MEDIUM, design) | partly. The inconsistency is real. The direction is not supported: the eval context (text + numbers line + question) is slot 1 of training at every K with the same 117,184 lines per domain (trainer dom_row: slot order permuted per row, domains cycled), the model is RoPE (OLMo-2), and the remaining differences are 25-35 far-left text tokens and the question's absolute index (~30 tokens) at K 5 | caveat rewritten with the measured layout (Caveats, 'Eval layout x K'); K\* stays rule 2 as written (binding, no direction change); put to the user before results (Status line); no K-matched eval without a trainer change (pinned) |
| 2 | stale summary text 'tie -> K = 1' / '(tie -> 1)' while the code picks the smaller K (LOW) | real (rainbow6_summary.py:28, :454; the code at :447 is right) | text fixed (summary c32891cace9a); JSON output byte-identical; the 3-K test now asserts the printed rule |
| 3 | CKA-rise bar 0.1 at lambda 1 justified from numbers-only runs at lr 2e-3, not six priors sharing one clip at lr 1e-3; tiny lambda-5 K 2 numbers +0.06 (LOW) | partly: the lr 1e-3 lambda-5 reference exists (+0.75..+0.79), but none at lr 1e-3 itself rises +0.24..+0.44, so the bar is a sanity floor only; no six-prior reference exists; the +0.06 is the tiny model | row 5 above and the generator comment corrected: a CKA-only FAIL is read against the same-seed none CKA, not as a broken run; no threshold change (nothing to calibrate it on before the pilot) |
| 4 | no GPU timing for K 5; a TIMEOUT is not retried (burst/tick.sh retries only CANCELLED / NODE_FAIL / PREEMPTED / BOOT_FAIL / LOST) (LOW) | real (tick.sh checked) | no change: 2 x 0.867 h < 02:00:00 (x 2.3 margin); wave 1 contains the K 5 tasks 402, 405, 410, 411, whose check info gives sec_per_step_median / projected_run_h; a TIMEOUT is re-run as 418-429 from step 0 |
| 5 | the push loop `until git push ...; do git pull --rebase ...; done` never ends on a non-race failure (LOW) | real | the commit commands now retry at most 5 times and stop on a failed rebase |
| 6 | the lambda shell check rejects other printed forms (e.g. 1.00); the first wave is 400-411 only if no other ready GPU task is pushed first (INFO) | real, fail-safe (exit 3 before training; the trainer prints json floats: 5.0 / 1.0 / 0.0, as in the Torch dry logs) | the commit commands check that only the 18 new tasks lack a status before the push |

Read-out (Burst; supersedes A3's 40[0-7] glob): after the first wave (the binding gate + one seed per prior cell)
`python tools/rainbow6_summary.py results/40[0-9]-rb6pb-* results/41[01]-rb6pb-* --ref 'results/35[0-3]-pt2b-lr1e-3-none-s?'
--ref 'results/35[0-3]-pt2b-lr1e-3-win-ctx-s?' --json rb6pb_wave1.json`; after all 18: the same with
`results/41[0-7]-rb6pb-*` and `--json rb6pb_summary.json`. A re-run (418-429) replaces its failed task's directory in the
call (the summary refuses two runs of one (arm, K, seed)).

## 2026-10-19 re-runs
Tasks 408 and 409 (K2 all6 / all6-lam1, seed 0) ended CANCELLED+ after 0 minutes: Burst nodes failed to boot and the tick's
3 retries ran out, so neither started training. They re-run unchanged, byte-identical task files, as 418 and 419 (spare
numbers). The read-out glob results/4[01][0-9]-rb6pb-* covers them.
- 2026-10-19: task 411 (K5 all6-lam1 s0) ended FAILED, rc 4 from its own check.
  - The only failing check was "geo > 0": geo_weekdays = -3.16e-5 at step 5500. Weekdays (7 items, rank-2 circle) reached
    a perfect fit, and float CKA exceeded 1 by about 3e-5.
  - Every other check passed (numbers area 69.69; CKA rises; offline LM double path 3.5e-8). The run is valid, and the
    read-out uses it: rainbow6_summary reads run JSONs, not check results.
  - make_rb6_tasks.py now tests geo > -1e-4. This applies to future renders only; the queued pilot tasks keep the strict
    check, so a perfect fit there may also show FAILED and is read the same way.

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

## Amendment A8.1 (2026-10-20, implementation of A8, BEFORE any A7 result and any A8 run)

What A8 (commit ecc367c) decided is implemented as written. Nothing in A8 is changed. One finding needs a decision by the
user BEFORE results; see "For the user to decide".

### Code
- tools/rainbow6_summary.py: 490bfe8877f9 (A7) -> **5edd08a68d96** (A8 + A8.2).
- tools/make_rb6m_tasks.py: 2b629d98a87c -> **a1b3d3486605**: the A8 block, the seed 8-11 references, and the --no_* write logic
  by membership. The 80 existing renderings are unchanged.
- Unchanged: readout_rules, CHECK_PY, the pinned trainer (c434d29: cc984148b15a) and toy_struct.
- Base: origin/main ecc367c.

### s4_hard (the A8 read-out)
**Definition.** s4_hard (`s4_hard(r)`) is the s4_held_fresh_nonid cells NOT solved by A7's Latin propagation of the seed's
own training table (naked + hidden singles to a fixpoint; the same code as `s4_latin_level`).
- The function asserts that the propagation's implied level of the whole read-out set equals A7's level, and that no true
  answer is ever eliminated.
- An independent re-implementation in the test gives the same cells.
- The set is the same for every arm of a seed.

**Accuracy.** Each eval's accuracy is the mean over the s4_hard cells of the logged per-question hits (hits_s4_held, one
context). No re-run is needed; it is computed for every run.
- final3 and area use the A5 definitions on this series.
- Learned / speed: A8.2 (the exact binomial test against the seed's chance at the last eval / the first eval from which it passes at every later eval).

**Exact chance.** The chance is the pure Latin table-filler's expected accuracy on s4_hard: the mean over the cells of
1 / (remaining candidates).

**Per seed (seeds 0-7):**

| seed | n of the read-out set | chance | Latin level (A7) |
|---|---|---|---|
| 0 | 179 of 203 | 26.6 | 35.2 |
| 1 | 171 of 191 | 27.2 | 34.8 |
| 2 | 182 of 194 | 25.0 | 29.6 |
| 3 | 188 of 200 | 26.0 | 30.4 |
| 4 | 184 of 202 | 25.6 | 32.2 |
| 5 | 182 of 197 | 24.4 | 30.2 |
| 6 | 197 of 204 | 24.2 | 26.8 |
| 7 | 190 of 209 | 26.4 | 33.1 |

**Where it is printed** (A8: "printed with n per seed"; the primary S4 metric of A7 (1) / (2) and of A8):
- **A7 (1h)**: printed right after A7's (1) rows. R1h = all6@1e-3 vs none@1.6e-2 at seeds 4-7; R2h = vs none@3.2e-2 at seeds
  0-3.
  - Counts: seeds learned on s4_hard (A8.2).
  - Decision: one-sided Fisher, BH m = 2.
  - Beside: Barnard, and the paired t on s4_hard final3.
  - Its header says that the (1) rows above, on the old set, are SECONDARY.
- **A7 (2h)**: printed right after A7's (2) rows. It covers the first eval from which s4_hard stays significant (A8.2, sustained), with a one-sided exact Gehan test (BH
  m = 2), the sign test and medians. Its header says the (2) rows are SECONDARY.
- **The A8 s4_hard table**: every K = 2 run given, per seed n / chance / level, and per arm x seed final3 / area / step.
  - It is printed after the A7 section when an A7 or A8 run is given.
  - A new flag, --s4_hard, prints it for any command, e.g. the A5 or pilot one (default off, so those outputs are unchanged).
- **The A8 section**: printed last when an A8 run is given.

### Output lines that change (tested; everything else byte-identical)
- **Pilot, A5 (rb6m_main.txt) and A6 (rb6m_a6.txt) commands**: no change, TXT and JSON byte for byte (regress8.sh). This
  also holds when A8 runs are added: everything up to and including the A6 status line is unchanged (A8 plumbing).
- **A7 command (with A7 runs)**: the A8 summary's output = the A7 summary's output with exactly 3 insertions and no
  changed or removed line (a7diff.py, difflib on the A7 plumbing copies):
  1. after the last (1) row: a blank line, the "== A7 (1h) PRIMARY S4 reliability (A8) ..." header, and the R1h and R2h
     rows (4 lines);
  2. after the last (2) row: a blank line, the "== A7 (2h) PRIMARY S4 speed (A8) ..." header, and the S1h and S2h rows
     (4 lines);
  3. after the "== A7 read-out status" line: the A8 s4_hard table, which is a blank line, the header, the per-seed line and
     one line per arm.
  - JSON: a7.families gains "Rh" / "Sh", and a new top-level "s4_hard" appears; every other key is equal.
  - A7's own (1) header still reads "Fisher exact on the counts = the decision". That line is unchanged, as required; the
    (1h) header states that it is now secondary.
- **With A8 runs**: the A8 section is appended at the end. Its lines are the header, mode, identity, warnings, missing and
  diverged lines, the decision lines H1-H4, the conjunction line, the descriptive other seeds, A8.1 and the A8 status line.

### A8 runs and references
**Tasks.** 522-529 = all6-lam1@1e-3 and none@1.6e-2 at seeds 8, 9, 10, 11; per seed the pair: 522 / 523 = s8, 524 / 525
= s9, and so on.
- Tags: rb6pb-53m-2pct-k2-all6-lam1 (the reused A5 cell's tag) and rb6mb-53m-2pct-k2-none-lr1.6e-2.
- The arguments are the A5 / A6 cells' arguments with the seed (asserted).
- Header: gpu, 02:00:00, cpus 8, after 331-env311 332-data.
- 483-489 stay reserved.
- They are the highest numbers, so they run after A7's 510-521. That is about 8 x 0.92 = 7.4 A100-h.

**Torch references** (refs/run_refs_a8.py on the pinned tools; a8/refs8.py checks them):
- Seed 4 was recomputed first and equals its recorded values exactly: DRY 1f07da985f7d7acf.. (88 lines), DOMS, LM0
  11.58791446685791, DSHA0 9451e7743f0607ed.
- Seeds 8-11 then used the same method: --dry_inputs of none and all6-lam1 give equal DRY sha / DOMS; step 0 of none at every
  seed, and of all6-lam1 at seed 8, give the same value.
- none-lr1.6e-2's own dry logs at seeds 8 / 9 equal none's byte for byte.

| seed | DRY sha256 (88 lines) | LM0 | DSHA0 |
|---|---|---|---|
| 8 | 9cc7c5f0e911d73b.. | 11.61306095123291 | a3181fb4799d4806 |
| 9 | 61ae59aeb59e9d5e.. | 11.606510162353516 | c38dac2d365eb3ae |
| 10 | 6bb4ed0e31c43f6f.. | 11.57604694366455 | 0c4d582a9d024701 |
| 11 | 43d9194d521dacc2.. | 11.581033706665039 | fd9ea9b83ff3522d |

### The A8 read-out (a8_readout; A8's tests, made explicit)
The A8 runs are split off before every other section. Seeds 8-11 decide, and every test is one-sided (prior > none):
- **H1 'reliably'**: the count of seeds learned on s4_hard (A8.2), compared by Fisher exact. Barnard is printed beside.
- **H2 'early'**: the first eval from which s4_hard stays significant (A8.2, sustained), compared by an exact permutation Gehan test. The sign test and medians are printed
  beside.
- **H3 / H4 'level'**: the paired t on s4_hard final3 and on area, BH over the two.

**The hypothesis line** reads CONFIRMED only if H1 AND H2 are significant at 0.05. A8 names one test per part of the
hypothesis, and the conjunction is an intersection-union test, so it needs no multiplicity correction. Otherwise the line
reads "NOT CONFIRMED (<part> n.s.)". With a missing seed it reads "PROVISIONAL: not decided".

**Other parts of the section:**
- The other seeds of the two arms are printed descriptively (s4_hard final3 / step).
- **A8.1 (descriptive)**: numbers area, clock held_wrap and dim strict final3 on seeds 8-11, as all6@1e-3 - none@1.6e-2
  paired by seed with a two-sided t. These are labelled descriptive. none@1e-3 and none@8e-3 have no seeds 8-11. all6's
  seeds 0-7 mean is printed beside.
- Robustness as A7: a missing, excluded or unfinished run gives PROVISIONAL. Strict mode refuses before any conclusion.
  --a8_partial relaxes only the A8 section.

### Read-out command
The A7 command plus `results/52[2-9]-rb6*-*`; mid-batch add --a8_partial.

### Threshold
A8's 25% sat at chance; replaced by A8.2 (below).

### Tests (CPU; scratchpad rb6m/a8/)
| test | result |
|---|---|
| regress8.sh | pilot, A5 (= rb6m_main.txt) and A6 (= rb6m_a6.txt) TXT + JSON byte-identical |
| a7diff.py | the A7 output with the A8 summary = the A7 summary's + exactly 3 inserted blocks ((1h), (2h), the s4_hard table); JSON + Rh / Sh / s4_hard only |
| s4hard_planted_test.py (real JSONs, planted hits) | S4_HARD PLANTED TEST PASS: the set = an independent propagation's unsolved read-out cells; the table-filler scores the chance on s4_hard (26.0 / 25.5 / 26.7 vs 26.6 / 25.6 / 26.4) and the Latin level on the old set (34.8 / 32.2 / 33.4 vs 35.2 / 32.2 / 33.1); a perfect learner 100 / step 0; a hard-only learner from step 5000 has t25 = 5000 and old final3 = n_hard / n_readout; the same set across arms; a level differing from A7's -> AssertionError |
| a8_plumbing_test.py (real dirs + 8 relabelled copies, CLI) | A8 PLUMBING TEST PASS: output = rb6m_a6.txt + the s4_hard table + A8; JSON = A6's + s4_hard + a8; identity 8 / 0; H1 4/4 vs 0/4 (Fisher 1/70), H2 Gehan U 16 p 1/70 -> CONFIRMED; a copy with lr 1e-3 under the none-lr1.6e-2 tag EXCLUDED, strict refused before any A8 conclusion; --a8_partial PROVISIONAL, never CONFIRMED, A5 / A6 unchanged; with the A7 copies as well, everything above the s4_hard table = the A7-only output |
| render_test_a8.py (88 tasks) | RENDER TEST PASS: bash -n + own-log input checks 88, 1558 cross cases, 96 runs on the pilot's Burst logs; none-lr1.6e-2 dry logs at seeds 8 / 9 = none's byte for byte |
| check_test_a8.py | CHECK SCRIPT TEST PASS (76): real seed-4 / 5 runs relabelled to seed 8 / 9 with the new reference facts PASS; by args only they fail exactly data_sha / step-0 lm / domain splits (the references are seed-specific); intent(arm, 8) differs from intent(arm, 4) only in seed / lm0_ref / data_sha0 / dom; the A5-A7 cases as before |
| refs8.py | seed 4 recomputed = recorded; seeds 8-11 new and equal across both arms; all6-lam1 step 0 at seed 8 = none's |
| earlier suites on the A8 summary | all PASS: seed_completeness, identity, fake_main, mixed_lr, stats, a6_planted, check_finite, a7_plumbing (JSON check allows the new s4_hard key), a7_planted (+ an s4_hard lookup in its harness and a check of the (1h) / (2h) blocks, 18) |

## Amendment A8.2 (2026-10-20, before any A7 / A8 result; the 25% threshold of A8 sat at chance 24.2-27.2%, discovered while implementing)
- **Learned on s4_hard** means the run's LAST eval's s4_hard hits are significantly above its seed's exact chance.
  - Test: a one-sided exact binomial test at alpha .01, with n = the seed's s4_hard size and p = the exact chance (the
    Latin table-filler's expected accuracy).
  - The 3 evals of final3 are not pooled, because they are not independent.
  - The per-seed critical accuracy is printed. Seeds 0-7: 35.2 35.7 33.0 34.0 | 33.7 32.4 32.0 34.7, i.e. chance + 7.8 to
    8.6 points.
  - The binomial is conservative for a table-filler, whose per-cell chances differ (the count is Poisson-binomial with a
    smaller variance). Simulated, a pure filler passes the last-eval rule 0.6-0.7% of the time.
- **Speed (sustained)** is the first eval FROM WHICH that same per-eval test passes at that eval and at every later eval,
  censored at the budget. It is "never" whenever the last eval fails.
  - Why the sustained form: the any-eval form (the first eval passing the test at all) looks at 75 evals. A pure Latin
    table-filler passes it in 33-42% of simulated runs (seeds 0 / 4 / 7: 0.342 / 0.424 / 0.334; 4000 Poisson-binomial runs
    each).
  - The sustained form needs the last eval to pass, so its false-pass rate is the last-eval rate: 0.0060 / 0.0065 / 0.0070,
    within alpha .01.
  - The fixed-seed planted filler at seed 0 passes the any-eval form at step 5000 and gets "never" under the sustained one.
  - A run significant only at one early eval gets "never". On existing runs this applies to all6@1e-3 seeds 0 and 3, which
    pass once at step 2750 and are not learned at the last eval.
  - A learned run whose evals dip below the critical count late gets a late time (in the A8 plumbing copies, the seed-7
    copy's time is the last eval, 18309).
- **Where it applies**: A7 (1h) / (2h) and A8 H1 / H2. H3 / H4 (the paired t on s4_hard final3 / area) stay as written.
  A8's 25% is withdrawn everywhere.
- **Reading of A8 confirmed by the user**: CONFIRMED only if H1 AND H2 are significant (an intersection-union test).
- **The rule is data-independent.** It fixes alpha (.01) and the reference (the exact chance) without looking at any
  result. The critical accuracies follow from n and the chance alone.

## Amendment A8.3 (2026-10-20, before any A8 result was read; user decision)
- Why: A7 showed that A8's binary decision (seeds "learned" on s4_hard, Fisher exact, 4 vs 4) has little power: at the
  rates seen in A7 (prior ~6/8, none@1.6e-2 ~2/8), even 8 vs 8 gives Fisher p ~ .07. This amendment is made after reading
  A7 and BEFORE any A8 (522-529) result was read by anyone in this session.
- Confirmatory set: seeds 8-15 for BOTH all6-lam1@1e-3 and none@1.6e-2 (8 pairs, all fresh: never seen, never used to
  select an lr or to form the hypothesis). Seeds 8-11 = A8 (tasks 522-529); seeds 12-15 = new tasks (same args except
  seed). Seeds 0-7 of both arms are the exploratory set: reported, never part of the decision.
- PRIMARY (the only decision): the s4_hard AREA (100 x mean over all evals of s4_hard accuracy), all6 - none paired by
  seed over seeds 8-15, one-sided paired t at alpha .05 (prior > none). n = 8. Reported with the 95% CI, per-seed
  differences, and the exact sign-flip p beside (descriptive).
- SECONDARY (reported, not deciding): s4_hard final3 paired t; the count of seeds learned on s4_hard (A8.2 binomial
  rule) with Fisher exact; the sustained speed (A8.2) with Gehan; the A8 H1 & H2 "CONFIRMED" rule is demoted to this
  secondary block. Also the numbers / clock / dim headline domains on seeds 8-15, descriptive.
- Missing / excluded / unfinished run in seeds 8-15: PROVISIONAL, no decision (as A5.1); a re-run replaces it.
- The user's request "look at which metric is better afterwards" was declined: choosing the metric after the results
  would invalidate the test; all metrics are reported, the area alone decides.

### Erratum to A8.2 / A8.1 (2026-10-20, before any A8 result was read; fresh audit notes/batch1_build/audit_rb6_a8.md, I3)
- The A8.2 note says all6@1e-3 seeds 0 and 3 "pass once at step 2750 but are not learned at the last eval". This is wrong
  for seed 0: it is significant on s4_hard at 55 of its 75 evals and misses only the last eval (61 < 63 correct). It shows
  the fragility of the last-eval "learned" rule for borderline seeds (audit I1), which is one more reason the A8.3
  primary is the s4_hard AREA (integrated over all evals) and the learned count is secondary.
- Audit I2: a significant H1 implies a significant H2 in this design, so "early" is not tested separately by the
  demoted A8 rule; under A8.3 neither decides.
- The audit precomputed seeds 8-11 n / chance / critical for s4_hard; the read-out must reproduce them (assert).
