# ptdom — do the F81 fine-tuning positives also help in pretraining? Six single-prior arms vs none at 53M (2026-11-03)

Status: BUILT, NOT SUBMITTED (only `sbatch --test-only`, and A4's four GPU pilots). **Amendment A3 (2026-11-04) extends the
screen to 24 domains in four batches with a two-stage design and supersedes the lr design below (A / X / C) and its cost;
Amendment A4 (2026-11-04, end of this file) amends A3 before any stage-1 submission (trainer Rev G: the full-span k-way
read-out; none's lr extensions at both edges; the selection-free verdict condition; every batch on B200; per-batch HOLD; GPU
pilots); Amendment A5 (2026-11-04, after A4, before any stage-1 submission) adopts ptheld's double-path tolerances with
a run failing only them kept as flagged data, and fixB (none run at a prior's pick outside none's lr range); read A3 + A4 +
A5 for the current design.** Written before any run of this batch. **Amended before any
submission by Amendment A1 (end of this file), then by Amendment A2 (after A1; A2 holds where they differ; superseded lines carry
an [A2] marker)**: none's grid 1e-3 .. 1.6e-2 (stage A 92 runs), an area argmax at a grid
edge also triggers the extension [A2: only none's argmax at the LOWER edge at stage A; otherwise the 2-point margin, A2.2],
the auxiliary-loss gate in code [A2: per (arm, lr) cell with a / (a + b), A2.1], the shared-site and colours-shortcut rules, two CHECK
fixes, the cost; lines it supersedes carry an [A1] marker. USER DECISION (2026-11-02,
relayed by the coordinator): screen in 53M pretraining the domains that helped in the 36-domain fine-tuning screen (F81
robust positives), each prior ALONE vs none; the domains that pass count as "shown effective in pretraining" and later form
a new six-manifold combination with numbers and dimensions. Candidates: months (F81 210B +21.6; a stepping domain, the target
does not carry the answer) and the anchor-comparison domains stars, landmarks, languages, colours, cities (F81 42B / 210B
robust positives; their targets carry the answers, so a PASS reads "given the structure, the model uses it" = Tier L,
never "the model discovered the structure"). Background: FINDINGS F81 (the screen and its R2 injection table), F88 + A6-A8.3
(the rb6 stack at 53M), F84 / F94 / F95 (Tier L), F87 rule 1 (selection), F98 (the marginal-bootstrap lesson), R27 (seed
overlap of held-out questions).

## Question (one per domain, pre-registered)
For d in {months, stars, landmarks, languages, colours, cities}: in 53M pretraining on identical data, does the prior on d
alone (lambda 1, the rb6 window) raise d's held-out read-out above none, each at its own best lr (best vs best)?
Answer per domain: **PASS (shown effective in pretraining)** or **NOT SHOWN** (rule below). No pooled claim, no size claim.

## Design
### Data (the rb6 stack; identical in every arm of a seed, only which prior is on differs)
Row (512 tokens) = FineWeb-Edu text (train.bin; stream 2000 + seed) + **one numbers line (2%, the rb6 / pt2 main setting:
4 formats, operands 0..999 by length bucket, per-seed held-out pairs, 5% rehearsal; always the first line after the text)**
+ **K = 2 domain lines** in random order (stream 5000 + seed); the domain of the j-th line of row r at step s is
EXT[((s x 32 + r) x 2 + j) mod 6], EXT = months, stars, landmarks, languages, colours, cities (equal shares by the trainer's
cycling rule); the question uniform over that domain's training pool (stream 4100, child 0). 3e8 tokens = 18,310 steps of
32 x 512; 75 evals (every 250 steps + the last). The numbers prior is OFF in every arm (`--target none`).

| domain | training line (fixed template; prior sites ⟦ ⟧, scored span « ») | items / target | split key (held 40%) | train / held (seed 0) | chance on the read-out | heuristic lines (seed 0): answer-frequency / shorter option |
|---|---|---|---|---|---|---|
| months | `4 months after⟦ August⟧ is« December\n»` (rb6's, k = 1..23, k != 12) | 12 (single tokens) / circle Z12 | the (item, k mod 12) class = tier S (rb6 / F81; + tier E: one k of 50% of the trained classes) | 118 / 106 (+40 tier E) | 8.3 (1/12; majority 5.7) | — |
| stars | `Q: Which star is closer to Algie⟦ba⟧ in the night sky, Algen⟦ib⟧ or Ar⟦ided⟧? Answer:« Algenib\n»` | 89 (85 multi-token) / sphere (unit vector of RA, dec) | the option pair {b, c} with every anchor and both orders (ft36 / F81) | 1788 / 1212 | 50 | 49.3 / 49.8 |
| landmarks | `Q: Which landmark is closer to Eiffel⟦ Tower⟧, Big⟦ Ben⟧ or Great Pyramid of G⟦iza⟧? Answer: ...` | 76 (76) / sphere | option pair | 1778 / 1222 | 50 | 61.6 / 51.1 |
| languages | `Q: Which language is more closely related to⟦ English⟧,⟦ German⟧ or⟦ Uzbek⟧? Answer: ...` | 62 (20) / tree (family, branch; raw Gram) | option pair | 1816 / 1184 | 50 | **87.2** / 62.2 |
| colours | `Q: Which colour is closer in hue to⟦ red⟧,⟦ orange⟧ or⟦ teal⟧? Answer: ...` | 24 (6) / hue circle | option pair | 1828 / 1172 | 50 | 55.6 / 50.3 |
| cities | `Q: Which city is closer to⟦ Shanghai⟧,⟦ Istanbul⟧ or H⟦anoi⟧? Answer: ...` | 60 (27) / sphere | option pair | 1806 / 1194 | 50 | 67.7 / 54.9 |

- Anchor questions = ft36.pair_questions (every valid comparison of manifold_domains._pair_answer with the atlas margins,
  at most 1,500 = ft36's subset seed 7, each in both orders) on manifold_domains.domains() (geonamescache 3.0.2 from
  llmadd_torch/pylib, matplotlib CSS4, atlas CSVs); the table (items, target, every question) is hashed into
  `domains.<d>.table.sha` and checked per run. Split per seed on the domain's own registered stream
  (SeedSequence(4200 + seed).spawn(16), child 1 + i: split; 6 + i: shuffled target, unused; 11 + i: eval contexts), so
  the rb6 streams (4100) are untouched and months' split = rb6's months split of the same seed.
- **K = 2 and the shares (why):** K = 2 is rb6's pre-registered pilot gate K* (K in {1, 2, 5}: the most MEASURABLE
  non-number domains on the none arm, ties -> the smaller synthetic share; rb6 AUDIT A4 / A5). With six domains each gets 1/3 line per row
  (rb6: 2/5) = 195,307 lines per run: months' 118 training questions ~1,650 times each (rb6 ~1,990, its seen set 100%), an
  anchor training prompt ~108 times (1,780-1,830 prompts per table; rb6's s4 / dim ~680, clock ~490). The anchor tables are
  ~15x months', so equal line shares already give them the most distinct material; a larger K would push the domain lines
  from 7.8% to ~12% of the tokens. Measured (seed 0, 50 batches): text 90.09%, numbers 2.124%, months 0.456, stars 1.777,
  landmarks 1.633, languages 1.324, colours 1.303, cities 1.297%; slot shares 0.47-0.53. Risk recorded: ~108 exposures may
  leave an anchor seen set < 80 (flagged per domain, below).
- Months is rb6's months domain unchanged (template, split, eval contexts, held-out formats); its rb6 53M values are the
  prior expectation: tier S (exact match) at the floor in every arm (none@1e-3 0.3, only-months@1e-3 1.7, none@8e-3 12.0 with
  one seed 46.6), tier E measurable (34-61).

### Arms (all arms see identical data per seed)
`none`; `only-months`, `only-stars`, `only-landmarks`, `only-languages`, `only-colours`, `only-cities`. A prior arm =
`--target none --schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains <d> --lam 1` (lambda 1,
F88 / A5's auxiliary-loss rule; window steps 916..7323). Prior = mean over the hooked layers 2, 4 of 1 - linear CKA(states
at d's item sites in the batch, the Gram of d's target rows): months as rb6 (the start item); **anchors: three sites per line
(anchor a, options b and c), each at the LAST token of the item name** (ft36's site: the token holding the name's last
character; asserted at build time on every line that it ends exactly at the name's end, that the next token starts after it
and that the name's own tokens ' ' + name end there; `--dry_inputs` prints the rule with examples, e.g. 'Achernar' =
[' Ach', 'ern', 'ar'] -> site 'ar', 'Ho Chi Minh City' -> ' City', and re-asserts it on all 9,000 sites of each domain; the
first training step re-asserts it on the real batch). Target = the atlas primary (manifold_atlas.primary): sphere for
stars / landmarks / cities, hue for colours, tree for languages (raw centred Gram, ft36's RAW_GRAM rule: its column weights
encode family > branch); the others the trainer's centred_gram (column-standardised, as every rb6 prior). Never on the scored
answer span (the options are sites where they are named in the question, before "Answer:"; the answer after it is not). A
batch's term is skipped with < 4 sites or < 3 distinct items (logged as prior_used_frac).

### lr selection (stages; select_ptdom.py = readout_ptdom.decide)
- **Stage A (84 runs) [A1: 92 runs; none's grid 1e-3 / 2e-3 / 4e-3 / 8e-3 / 1.6e-2, overlapping the prior arms' at 1e-3 /
  2e-3]:** every arm x its 3-point factor-2 grid x seeds 0-3. Prior arms 5e-4 / 1e-3 / 2e-3 (around rb6's
  all6 pick 1e-3, A6 / A7: 5e-4 < 1e-3 > 4e-3); none 4e-3 / 8e-3 / 1.6e-2 (around rb6's 53M none pick 8e-3; none's best for
  months tier E / S4 / numbers was 8e-3-1.6e-2; none@1e-3 was below none@8e-3 on numbers, months tier E, clock, S4 and dim). The two grids are
  disjoint by design (best vs best, F87); consequence for the LM row below.
- **Units:** (only-d, d) is selected on **d's own read-out**; none is selected **separately for every domain on that
  domain's read-out** (unit (none, d)), so each domain's comparison is best none for d vs best prior for d. One shared none
  grid serves all six units (no extra run); it implies **1 to 3 distinct none picks after stage A (up to 5 with the two
  possible extension lrs)** [A1: 1 to 5 after stage A, at most 6], each confirmed once (a none lr picked by several domains is one confirm cell).
- **Rule (F87 rule 1 as pre-registered, readout_rules.best_lr near 'top2'):** argmax of the mean area over seeds 0-3; top two
  within 2 points -> the lower mean bpb (offline full-val when every run of both has it, else in-training; marked). Open
  bracket = readout_rules.bracket: the pick at a grid edge beating its neighbour by > 2 points [A1: OR the area argmax at
  a grid edge (readout_ptdom.extend_rule); A2: superseded: the 2-point margin for every unit, plus none's argmax at the
  LOWER edge of its grid at stage A; STOP by the margin only (A2.2)] -> **stage X: one lr one
  factor-2 step beyond that edge, seeds 0-3** (none's X lrs shared by the domains asking for the same one); re-selected over
  grid + X; still open -> **STOP** for that unit (no confirm; the domain is NOT FINAL; the user decides). Units final after A
  keep their pick (X lrs of other units are not added to their grid).
- **Stage C:** the confirm seeds 4-7 at every final pick (6 prior cells + the distinct none picks). Every comparison then
  has 8 seeds; seeds 0-3 are the selection seeds of both sides (winner's curse on both; the halves rule below).

### Metrics (unique definitions; trainer pretrain_range.py Rev E, curve_metrics; readout_ptdom.load)
- Every domain eval places the question at the row end in slot 1 = text + one numbers TRAINING line + the question (rb6 A2;
  the training position of a slot-1 line), teacher forced, eval mode, bf16 autocast.
- **2-way (anchors):** for each held-out question and its context, the summed log-prob of ' <answer>\n' vs ' <other
  option>\n' after the same prompt and context (two rows; ft36's scoring); hit iff strictly larger (ties logged as ties2_*;
  a tie counts as a miss). `<d>_held_2way` over **every held-out question (both orders, ~1,200)**; chance **50** exactly for an
  option-order bias (both orders present). Known biases of the scorer recorded as heuristic lines per seed: answer frequency
  (answer the option more often right in training) and shorter option (fewer tokens = higher summed log-prob at init); the
  step-0 value of every read-out is printed (tiny CPU run, languages 61.9 at init). Secondary sub-split `<d>_held_2way_hard`
  = the questions both heuristics get wrong or tie (masks in domains.<d>.eval_held.masks).
- [A4: the primary k-way read-out is the full-span choice (A4.1); this one is the secondary line; the known answer below is tie-aware] **k-way (months):** argmax of the answer-token logit over the 12 month tokens (single tokens, asserted), on tier S
  (`months_held_kway`), tier E and seen. ft36 also summed the newline's log-prob; with single-token candidates whose training
  continuation is always '\n' this differs only when the first-token log-probs are within the newline term (recorded).
  Exact match (rb6's months metric) is logged unchanged; a known-answer check asserts that every exact-match hit is a k-way hit.
- **Primary metric = area** (100 x mean over the 75 evals) of: months `months_readout_kway` (tier S k-way); anchors
  `<d>_readout` (= held_2way). **Secondary: final3** (mean of the last 3 evals). Selection and decision use the same area.
- Seen sets (memorisation check): `months_seen_kway`, `<d>_seen_2way` (150 training questions x 2 contexts).
- double paths: the 2-way span log-prob vs a direct per-position recomputation on the first 4 rows of every 2-way set at
  every eval (lp_dp_maxdiff_*; CPU < 2e-2 asserted, GPU check < 0.25 nats [A5: eval 0 <= 0.05, later <= 0.5 nat per scored
  token; a run failing only this is flagged data, A5.1]); grad_T (decomposition) vs the backward's
  pre-clip norm; the read-out re-derives the picks from the run JSONs and compares them with picks_A / picks_AX.

| dimension | training | eval | same? |
|---|---|---|---|
| position | slot 1 (after the numbers line) in ~1/2 of a domain's lines, slot 2 otherwise | slot 1, row end | yes for slot 1 (logged shares) |
| template | one per domain | the same (months also 4 held-out formats, descriptive) | yes |
| questions | training pool | held-out pool (split by key, asserted disjoint; both orders) | generalisation by design |
| scoring | LM loss on every token | 2-way / k-way choice of the answer span | the read-out is a choice, stated as such |
| model state | train, bf16 autocast | eval, bf16 autocast | as rb6 |

### Pre-registered decision (readout_ptdom.py; per domain; gain_s = area(only-d@pick, s) - area(none@pick_d, s), seeds 0-7)
**PASS (shown effective in pretraining)** iff all of
- (a) the 90% paired seed-bootstrap CI of the mean gain (10,000 resamples of the 8 seeds, default_rng(0), percentile)
  excludes 0 (lower bound > 0);
- (b) both seed halves positive: mean gain over seeds 0-3 > 0 AND over seeds 4-7 > 0;
- (c) the two-sided one-sample t-test of the 8 per-seed gains, **Holm-adjusted over the six domains, p_Holm < 0.05** (so the
  per-seed 95% t interval excludes 0 even before the correction; F98: a bootstrap CI excluded 0 while the t interval did
  not);

otherwise **NOT SHOWN** [A1: colours must also beat the anchor-conditional baseline on its hard subset (A2: by its hard-subset FINAL3, A2.3), else NOT
SEPARABLE; a NOT SHOWN on stars / landmarks reads 'not shown with this site (many shared site tokens)'; anchor verdict
lines carry Tier L]. A unit that STOPped (open bracket after X) -> **NOT FINAL** (no verdict); the Holm family is
always the six domains (a NOT FINAL domain enters with the t p of the seeds it has, 1.0 below 2 seeds: conservative). Descriptive tags (never
change the verdict): FLOOR (both arms' mean area <= max(chance, heuristic lines) + 5: not measurable, never "no effect"),
CEILING (none >= 95), SEEN < 80 (a seed of either arm below 80 on its seen set: under-trained), HARM (the three conditions
hold for -gain). Every verdict line prints the picks, gain, per-seed gains, wins, bootstrap CI, halves, t95, p, p_Holm,
final3 gain, chance, heuristic lines, step-0 value and the heuristic-hard subset gain.
- **What a PASS means:** "the prior with this target helps this domain's held-out read-out in 53M pretraining". There is
  **no dose-matched shuffled-target control in this screen** (cost), so a PASS is NOT geometry-specific. Anchor domains:
  **Tier L** (the target encodes every held-out answer: F81 R2 99-100%) — "given the structure, the model uses it", never
  "the model discovered the structure"; the heuristic-hard subset shows whether the gain survives where answer frequency and
  length do not solve the question. Months: the target does not carry the answer (tier S needs composition); its F81
  analog is the 210B +21.6.
- **What NOT SHOWN means:** no reliable gain at 8 seeds under this rule; with a FLOOR tag the screen cannot tell.
- Expected (written before any run): months FLOOR / NOT SHOWN (rb6 tier S at the floor in every arm); languages near the
  answer-frequency line (87) in both arms (little room); stars, the domain where F81 none sat at chance, the clearest test.

### Side effects (descriptive, readout_ptdom.side_effects; never a verdict)
- LM: offline full-val bpb x 1e3 (offline_lm.py, every run) and the in-training final bpb: only-d@pick - none@pick_d and -
  none@(its LM-best lr in the grid), per seed. **lr-confounded** (the grids are disjoint; rb6: lr alone costs +25e-3 at 8e-3):
  never "the prior improves / costs LM" without the lr pair. [A1: plus the lr-matched row only-d@lr - none@lr at 1e-3 and
  2e-3 (and any lr both arms ran)]
- numbers-line area (6 buckets) under each prior vs none@pick_d.
- the other five domains' read-outs under each single prior vs none at THEIR picks (cross-domain matrix).
- gradient split per prior arm at its pick (window medians over the on-evals): LM share |g_lm| / |g_lm + lambda g_d|
  (grad_lm_share), the clip coefficient min(1, 1 / |g|) of the backward (clip_coef; global clip 1.0), |lambda g_d| / |g_lm|;
  FLAG when a seed's median LM share < 0.5 (the prior dominates the update: F88's auxiliary-loss rule). Logged at every eval.
  [A2: the LM share is a / (a + b), a = |g_lm|, b = |lambda g_d| (pre-clip); the gate is per (arm, lr) cell (A2.1)]

## Code
- Trainer: tools/pretrain_range.py **Rev E**, branch `ptdom` of /scratch/yh6210/llmadd-burst-ptdom (worktree of
  llmadd-burst-tpp's repo, from origin/main 75c7e1e; commits 55c96be + 21cdc14; NOT pushed): sha256[:12] **bb34927f16d8**
  (= code_sha of every run). Opt-in: only an anchor domain in `--domains` enables Rev E (no new production flag;
  `--anc_eval_max` is testing-only and kept out of the run JSON at its default). New: ANCHOR domains (ft36 questions on
  manifold_domains tables, the split, anc_line with the site asserts, the other-option rows), the raw-Gram path for the
  tree target, the 2-way / k-way choice read-outs and their double path, the heuristic lines and hard masks, the gradient
  split (grad_T, grad_lm_share, grad_total, clip_coef, grad_T_reldiff), dep_sha of ft36 / manifold_domains /
  manifold_atlas. Imports unchanged: toy_struct d182ce548bcc, pretrain_helix 25d5a35e6f17, numlib d09bb47b04c2,
  digit_split_data 742b922c8125, run_guard b944dc859c7a; ft36 2f84907b4725, manifold_domains 0607a9d693da,
  manifold_atlas 43efc295ecc0; offline_lm 226f4cac70ea.
- Snapshot: runs/ptdom/code = git archive of tools/ at 21cdc14; code/COMMIT lists every sha; the generator asserts snapshot =
  commit = WANT_SHA and re-snapshots only while no SUBMITTED_ flag exists.
- Generator: llmadd_torch/make_jobs_ptdom.py (`--plan`, `--snapshot`, `--stage A|X|C [--picks]`, `--print`, `--test-only N`,
  `--submit`; asserts every flag of every job, every other flag = its default; refuses a seed without CPU references or an
  out dir of another batch; never rewrites a submitted job file).
- Selection: runs/ptdom/select_ptdom.py; read-out: runs/ptdom/readout_ptdom.py (`--selftest`); submission:
  runs/ptdom/submit_ptdom.sh (prints; `--go` submits stage A only).
- CPU references: runs/ptdom/ptdom_refs.json (tests/refs_ptdom.py): per seed 0-7 the DRY row hash, the splits, the anchor
  table shas (arm-independent: none@8e-3, only-stars@1e-3, only-months@5e-4 at seed 0), the step-0 lm and data_sha
  (scale_work/step0_ng.py, the rb6 method; none@8e-3 and only-stars@1e-3 at seed 0 identical).

## Checks inside every job (exit 3 before training, exit 4 after)
1. HOLD kill switch; the sha256[:12] of the 12 snapshot files; COMMIT names the trainer; **an L40S guard**; the data sha256.
2. `--dry_inputs` with the job's own ARGS: the model line (52.7M, no grad ckpt); no numbers target; per domain prior / shuf /
   lam in the `domains:` line; the numbers line first in every row; the content scan (640 rows: no held-out / held-out-format
   prompt in any training line); the anchor site rule asserted on 5 x 9,000 sites; the DRY row hash (3 decoded training
   rows with the sites, the eval rows of every domain incl. the other-option rows, slot rows, shares) = the CPU reference of
   the seed.
3. [A4: stage 1 without --save_model and without offline_lm (A4.5); B200 ckpt every 500 (A3)] Training (`--save_model --ckpt_every 2000`; a requeued job resumes from its checkpoint), offline_lm.py (full val), then
   CHECK_PY with the run's intent -> check.json: every trainer argument (and no other), code / dep shas, run facts, lambda
   per domain, domains.<d> prior / shuf / lam, 75 evals, data_sha after step 0 and step-0 lm = the CPU reference, numbers
   first / share / slots, splits and table shas = the reference, the held-out formats and every Rev E read-out at every
   eval, the 2-way double path, the k-way known answer, the gradient split and its double path [A1: on-window evals only; A2: their MAX, A2.3],
   prior window / on-domains /
   applied >= 95% / logged / weighted gradient only for the active prior, **gnorm_prior / gnorm_true = lambda 1** [A1: the
   lambda parsed from the job's $ARGS; every ARGS flag = the run's args], geo >
   -1e-3 [A6: failing it at a geo overshoot of a perfect fit verified by the loader = valid data, flagged, A6], item-state CKA rise >= 0.1 (a CKA-only failure is a weak prior: the run stays valid data [A5: so does a run whose
   only other failed checks are the GPU-numerics double-path tolerances, flagged, A5.1]), none: no prior ever and
   LM share 1, offline double path < 0.002, the GPU name; info line: read-outs, heuristics, gradient medians, versions,
   sec/step, GPU util / memory.

## Hardware, jobs, cost
- [A3 / A4: every batch on B200, A4.6] L40S only: account torch_pr_625_general, partition l40s_public, QoS gpu48, `--nice=20`, 1 GPU, 8 CPUs, 64G,
  `--time=03:00:00`, `--requeue`; logs llmadd_torch/logs/llmadd_ptdom-<arm>-lr<lr>-s<seed>_<jobid>.out.
- Timing basis: L40S / A100-40GB = 1.31 at 53M 2% (pt999-2pct 3,429-3,459 s vs pt2b 2,635 s); rb6 53M K 2 = 3,075 s on
  A100 -> 1.12 h on L40S; ptdom's eval rows ~1.1x rb6's plus the other-option forwards -> plan **1.3 L40S-h per run**
  (incl. dry run, offline LM, checks).
- Runs: stage A 84 (109 L40S-h); X 0-32 (0-42); C 28 (one none pick) - 44 (five) (36-57). **Nominal (no X, one none pick)
  112 runs = 146 L40S-h; maximum 160 runs = 208 L40S-h > 150.** [A1: stage A 92 (119.6); C 28-48; nominal 120 runs =
  156.0 L40S-h; maximum 172 runs = 223.6 L40S-h] [A2: expected 127-140 runs = 166-181 L40S-h by simulation, A2.2] Cheaper variant (proposed, NOT implemented): prior arms on
  2 lrs (1e-3 / 2e-3; 5e-4 lost to 1e-3 in rb6 A7) and none on 2 lrs (8e-3 / 1.6e-2) -> stage A 56 runs (73 h), nominal 84
  runs = 109 L40S-h, max ~140 runs; the X rule then fires more often at the dropped edges.
- Disk: saved models 0.21 GB per run (offline LM, later probes): 24-34 GB on /scratch; checkpoints transient.

## Tests before submission (CPU on the claude_host node; scripts and logs in runs/ptdom/tests/)
| test | script / log | result |
|---|---|---|
| read-out + selection self-test (planted known answers) | readout_ptdom.py --selftest / tests/selftest_readout.out | SELFTEST PASS 31/31: Holm = the step-down rule (2000 random vectors) and a hand-computed case; known positive -> PASS; zero -> NOT SHOWN; halves disagree (boot90 low +1.1) -> NOT SHOWN; t95 includes 0 while boot90 excludes it (+0.31; p .080, the F98 case) -> NOT SHOWN; raw p .018 but Holm .091 -> fails (passes without Holm); floor -> NOT SHOWN + FLOOR; selection: closed pick, top-two tie by offline bpb, open edge -> X -> closed, open -> X -> still open -> STOP (NOT FINAL), none's per-domain picks and shared X / C cells, units final after A unchanged; picks double path agrees; loader: failed check excluded, CKA-only fail valid, unfinished excluded; data_sha mismatch reported; strict refusal of a missing seed, --allow_partial PROVISIONAL |
| 200-step CPU smoke (tiny d64 x 3, seq 128, bs 8; window steps 10..79; 11 evals; reduced eval sets incl. --anc_eval_max 40) | tests/smoke_200.sh, smoke_check.py / tests/smoke_200.out, smoke/ | SMOKE PASS 13/13: only-stars vs none on identical data (data_sha at all 11 evals, step-0 lm 11.532384 both); the stars prior loss on the fixed eval prompts (1 - cka_stars) 0.909 -> 0.849 (step 60) -> 0.656 (80), below none's by 0.11-0.30 from step 40 on; training-batch geo_stars noisy (0.67, 0.69, 0.33, 0.66 at steps 0-60; on-eval mean 0.56 < 0.67); gnorm_prior / gnorm_true = 1 at the 3 on-evals; applied on 70/200 steps exactly, no other prior; none: geo 0 at every eval, no geo_* / gnorm_prior_* key, LM share 1; 2-way double path < 2e-2. Tiny-model LM share in the window 0.08-0.10 (unweighted prior gradient 8-10 vs LM 0.8): see open question 5 |
| CPU references (53M, the jobs' own ARGS on runs/ptdom/code) | tests/refs_ptdom.py, run_refs.sh / tests/run_refs.log, refs/ | TABLE PASS (seeds 0-3: run_refs.log; seeds 4-7: run_refs2.log, step-0 lm 11.590243 / 11.576292 / 11.578110 / 11.595399, data_sha 67b43c555076bbc2 / 4e09b941c6d22734 / f521a1e9dcc96c1b / 19e8f82fea42054f): DRY 84 lines per seed, 8 distinct hashes, seed 0 322bf3c5911b = only-stars@1e-3 = only-months@5e-4 (arm-independent); 5 site-rule lines per dry log; step-0 lm s0-3 11.615026 / 11.596675 / 11.573246 / 11.600012, data_sha 160aa036eddf3f35 / b9a2e3e86a5d6cf7 / 54ed0bebaeb0d352 / c8752b7f06dc4810; only-stars@1e-3 s0 = none@8e-3 s0 bitwise; splits per seed (e.g. seed 0 months 118 / 106, stars 1788 / 1212, cities 1806 / 1194) and table shas recorded -> ptdom_refs.json; dry 225-285 s each on 2 shared CPUs |
| generator unit tests (splits, leakage by key, site decoding) | tests/test_generators.py / tests/test_generators.out | GENERATORS PASS 52/52: per anchor domain the split re-derived independently (ft36.pair_questions + ft36.split on SeedSequence(4200 + seed) child 1 + i) = the trainer's split_seed / n_held / n_train / n_held_keys and its full held question list (fixture: a full-eval-set tiny run) and the smoke's first 40; no held option pair in training in either order with any anchor; both orders of every held comparison held; no held prompt string in training; seen questions are training questions; seed 1's split differs; months: rb6's tier-S class split re-derived (cyclic_split, stream 4100), no held class trained, 118 / 106 / 40 = rb6's; site decoding on the 53M dry logs of seeds 0 and 1: all 191 marked sites of 76 decoded lines end an item name and are not followed by a letter / digit, 3 sites per anchor line and 1 per months line, 3 training rows (len 512), >= 3 eval rows per domain + the other-option rows; the site checker rejects two planted bad lines |
| per-run CHECK known answers | tests/check_py_test.py / tests/check_py_test.out | CHECKPY PASS 9/9: CHECK_PY (the job text) on the smoke's real run JSONs with the arm's intent (tiny facts substituted; step-0 lm / data_sha from the OTHER arm, splits / table shas from the 53M refs): only the 2 GPU-job checks fail (+ for only-stars the 53M-calibrated CKA-rise bar: tiny model +0.06 at the window's last eval), and the loader keeps such a CKA-only run valid; tampers fail their check: lambda 2, a wrong data_sha0, a wrong table sha, lr 2e-3 in the intent, a geo_stars value in the none run, landmarks marked on in the stars run |
| job files | tests/jobcheck.py / tests/jobcheck.out | JOBCHECK PASS 4/4: 84 stage-A files rendered; 84/84 bash -n, embedded intent = make_jobs_ptdom.intent, CHECK_PY byte-identical and compiling, ARGS = args_of, header (625 / l40s_public / gpu48 / nice 20 / 03:00:00 / requeue) + HOLD + L40S guard; the input-check block executed on the CPU dry logs: own cells exit 0 (6/6: none@8e-3 s0-3, only-stars@1e-3 s0, only-months@5e-4 s0), another seed's rows exit 3 (4/4), another arm's domains line exit 3 (6/6) |
| select -> generator plumbing (synthetic runs) | tests/plumbing.py / tests/plumbing.out | PLUMBING PASS 9/9: select_ptdom --stage A on 84 synthetic runs -> X = only-stars@4e-3 + none@3.2e-2, C = 5 prior picks + none@8e-3; make_jobs_ptdom --stage X / C --picks renders exactly 8 / 24 jobs (seeds 0-3 / 4-7) and writes nothing with --no-write; refuses an X lr two steps beyond the edge and a picks file of another design; select refuses to overwrite a decision and a missing stage-A cell |
| sbatch --test-only | make_jobs_ptdom.py --stage A --test-only 2 / tests/sbatch_test_only.out | 2/2 accepted (none@4e-3 s0, none@8e-3 s0: l40s_public, estimated start 2026-11-05 03:48 on gl018); nothing submitted (no SUBMITTED_ flag, no ptdom job in the queue) |
| trainer regression (old = origin/main 93e22bf4c4c0, the trainer of rb6_190m / rb6_505m; new = Rev E bb34927f16d8) | tests/reg_trainer.sh, reg_compare.py / tests/reg_trainer.out | REGRESSION PASS: tiny d64 x 3, seq 128, bs 8, 40 steps, 3 evals, one thread, both trainers on this host: the default path (no --domains; helix3 window, lam 5), rb6 all6-lam1 (the six rb6 domains, K 2, lam_rule each) and rb6 none: IDENTICAL (run JSON equal except code_sha / args.out / per-eval 'sec' incl. domains, metrics, dep_sha and every curve field; saved model 36/36 tensors bitwise; --dry_inputs stdout equal, 11 / 96 / 95 lines; run stdout equal after removing 'sec' and the warning's file path) |
| frozen rb6 generators / read-outs | tests/reg_frozen.py / tests/reg_frozen.out | FROZEN PASS: make_rb6m_tasks.py --check (96 tasks unchanged, 0 differ); the 16 rb6_190m job files re-rendered byte-identical; the rb6_505m main read-out x1a re-run with its frozen code = readout_505m_x1a.txt (320 lines) and .json |

## Audit status (experiment-audit, minimal mode for the build)
ii asserts in the trainer (sites, splits by key, both orders, prompt tokenisation, no held prompt in training, the k-way
known answer, the 2-way double path on CPU) and in every job (input checks, CHECK_PY); iii known answers: the read-out
self-test (planted PASS / zero / halves / t-vs-bootstrap / Holm / floor / STOP), the heuristic lines and step-0 values, the
reproduced rb6 months split; vii decoded training and eval rows in every job's dry_inputs.txt and the CPU dry logs;
i double paths listed under Metrics. v (fresh adversarial review): NOT DONE in the build — to be launched by the
coordinator before submission (or right after it, the gate's minimal mode).

## Open questions (for the user / coordinator; nothing here changes the design without a decision)
1. Cost: the maximum (208 L40S-h) exceeds 150; the 2-lr variant above, or accept.
2. LM cost has no lr-matched none (disjoint grids); a none@1e-3 cell at seeds 0-3 (4 runs, ~5 L40S-h) would give an
   lr-matched LM row for the prior arms at 1e-3 (descriptive).
3. Languages: the answer-frequency heuristic alone scores 87.2 (seed 0), so little room above it; read its hard subset.
4. Months tier S is expected at the floor (rb6); tier E k-way is logged but not the pre-registered metric.
5. The LM share at 53M is unknown until stage A: the tiny CPU smoke (d64 x 3) had a stars-prior LM share of ~0.1 in the
   window (the prior dominated the tiny model's update), rb6's single priors at 53M had >= 0.98 (window median). rb6 53M
   seed 0 (Burst 401 / 439 / 442): an UNFITTED prior's unweighted gradient is 1.9-4.3x the LM's (months 4.3, s4 1.9, dim
   2.5 in the none run), a fitted one 0.15x (months in only-months, s4 in only-s4; window medians), so the share depends
   on how fast the prior fits; the anchor priors (89 / 76 / 62 / 60 items, multi-token names) may fit slower than months. Each run's check info
   and the read-out print it (FLAG < 0.5); if a prior arm is flagged at stage A, that is reported to the user before
   stage C (the design's "auxiliary loss" premise would not hold for that arm), not silently continued. [A1: gated in code;
   A2: per (arm, lr) cell with a / (a + b), A2.1]

## Amendment A1 (coordinator, before any submission)

2026-11-03, written before any run of this batch (nothing submitted: no SUBMITTED_ flag, no ptdom job in the queue; only
`sbatch --test-only`). Source: the fresh adversarial build audit tests/audit_build.md (important items 1-4, minor 5-13).
The coordinator decided items 1-5 below; they are implemented exactly and are part of the pre-registration. Where the text
above disagrees with A1, A1 holds (the lines above are kept as the build's record; the affected ones carry an [A1] marker).
Code: make_jobs_ptdom.py (grid, cost, CHECK_PY), runs/ptdom/readout_ptdom.py (extend_rule, aux_gate, anchor sub-read-outs,
colours rule, same-lr LM row, wording), select_ptdom.py (gate before any decision), new runs/ptdom/aux_gate_check.py; the
trainer and runs/ptdom/code are UNCHANGED (21cdc14, pretrain_range.py bb34927f16d8; the 12 snapshot shas re-verified).

### A1.1 The lr confound (audit item 1)
- **Change:** none's stage-A grid is {1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2} (seeds 0-3; + 8 runs: none@1e-3, none@2e-3), so it
  overlaps the prior arms' {5e-4, 1e-3, 2e-3} at 1e-3 and 2e-3. **Reason:** the build's grids were disjoint (every prior
  lr 2-32x below none's lowest) and the bracket extended only when the pick AFTER the bpb tie-break sat at an edge and beat
  its neighbour by > 2; a none read-out that rises slowly toward low lr (< 2 points per factor 2) was then never measured
  at the prior arms' lrs, so an lr effect could pass all three conditions as a prior effect (a bias toward a false PASS
  only).
- **Selection otherwise unchanged:** F87 rule 1 (best_lr 'top2') per unit; none selected separately on every domain's
  read-out (unit (none, d)); one extension per unit; STOP if still open. [A2: the trigger below is SUPERSEDED by A2.2: the 2-point margin for every
  unit; of (ii) only none's argmax at the LOWER grid edge at stage A is kept; STOP by the margin only] **New pre-registered trigger
  (readout_ptdom.extend_rule, every unit, prior arms included):** a unit is OPEN when (i) the pick is at a grid edge and
  beats its neighbour by > 2 (readout_rules.bracket 'open', as before) OR (ii) the area argmax over the grid (the first
  maximum) is at a grid edge, also when the tie-break moved the pick inside or the margin is <= 2 (bracket's former
  'note'). The extension is one factor-2 step beyond that GRID edge (the pick's open edge first when both fire on different
  sides); after X the same rule over grid + X -> STOP (NOT FINAL). none's possible X lrs: 5e-4 and 3.2e-2 (shared by the
  domains that ask for them). readout_rules.py (frozen) is not modified: extend_rule calls bracket and asserts that 'note'
  is returned exactly when the argmax is at an edge and the pick is not open.
- **Same-lr LM row:** the read-out prints per prior arm the offline full-val bpb x1e3 and the in-training final bpb x1e3 of
  only-d minus none at the SAME lr, for every lr both arms ran (1e-3 and 2e-3; an X lr too), paired over the seeds both
  have (seeds 0-3 at stage A, 8 where both cells were confirmed). The rows vs none@pick_d and vs none@its LM-best lr stay
  and remain lr-confounded.
- Consequences: none has 1-6 distinct picks (7 possible lrs, at most one per domain); stage A = 92 runs.

### A1.2 The auxiliary-loss gate (audit item 2; the user's principle: the prior is only an auxiliary loss, the LM gradient must dominate)
- [A2: Quantity and Rule SUPERSEDED by A2.1: a / (a + b) per (arm, lr) cell, a missing field fails; exclusion, GATE units]
  **Quantity** (the one F88 / rb6 report as the window-median LM share; rainbow6_summary.grad_split): per run, the median
  over its on-window evals (prior_on = 1 with grad_prior and grad_lm logged, before a divergence) of the trainer's logged
  **`grad_lm_share`** = `grad_lm` / `grad_T`, `grad_T` = |g_lm + sum_d lambda_d g_d| of the autograd decomposition at the
  eval step. Double path: rb6's re-derivation `grad_lm` / sqrt(`grad_lm`^2 + `grad_prior`^2 + 2 `grad_lm` `grad_prior`
  `grad_cos`); a disagreement > 1e-3 counts as unverifiable (= fail). Clip: the share of on-window evals with
  **`clip_coef`** < 1 (`clip_coef` = min(1, 1 / (`grad_total` + 1e-6)), `grad_total` = the backward's pre-clip norm, global
  clip 1.0), printed beside it.
- **Rule** (readout_ptdom.aux_gate, run by select_ptdom.py BEFORE any decision, at --stage A and AX): per prior arm the
  median over its FINISHED runs (75 evals, valid or not: the gradient log is a fact about the update) of the run's
  window-median LM share; any arm < 0.5 (or with finished runs but no verifiable split) -> prints **AUX GATE FAILED**,
  appends the reason to **runs/ptdom/HOLD** (the kill switch every pending job checks at start; make_jobs_ptdom.py
  refuses --submit while it exists), writes no picks file, exit 2: no X / C decision; the user decides. The read-out
  prints the same gate.
- [A2: per (arm, lr) cell, A2.1 (iii)] **Early watcher:** runs/ptdom/aux_gate_check.py (read-only, never writes HOLD) on the PARTIAL run JSONs (the trainer
  rewrites its JSON at every eval; a file caught mid-write is retried, then skipped): per prior arm the pooled median LM
  share over the on-window evals logged so far, the per-run medians, the clipped share and the double path; exit 1 if any
  arm < 0.5 (or unverifiable) -> the coordinator touches HOLD and reports to the user; exit 2 before any on-window eval;
  exit 0 otherwise. Run it as soon as the first prior runs pass step 1500 (window from step 916: on-window evals at 1000,
  1250, 1500) and again later.
- A precondition of the screen, not a change of the verdict rule.

### A1.3 Shared prior-site tokens (audit item 3)
- **The site stays the last token of each item name** (no trainer change). Items whose site token is also another item's
  (OLMo-2 tokenizer on the snapshot's tables; = the audit's lasttok.py): **stars 30 of 89** (12 tokens: 'ux' Acrux / Becrux
  / Gacrux / Pollux, 'ab' Girtab / Kochab / Marchab / Markab, 'ar' Achernar / Izar / Schedar, 'a' Agena / Ankaa / Shaula,
  'or', 'an', 'ent', 'am', 'ak', 'air', 'ba', 'amin'), **landmarks 16 of 76** (6: ' Bridge' x4, ' Tower' x3, ' Falls' x3,
  ' Castle' x2, ' Palace' x2, ' House' x2), **languages 2 of 62** ('ese': Burmese / Javanese), **colours 0 of 24**,
  **cities 5 of 60** (' City': Ho Chi Minh / Mexico / New York / New Taipei / Quezon City).
- Held-out questions whose three items (anchor, both options) all have unique site tokens, seeds 0-7 mean [min, max]:
  stars 354 of 1,204 (29%; [324, 406]), landmarks 597 of 1,199 (50%), languages 1,108 of 1,199 (92%), colours 1,192 of
  1,192 (100%), cities 899 of 1,202 (75%).
- **Read-out:** per anchor domain the count line, and a descriptive line: the gain on the unique-site questions only
  (area recomputed from the per-question hits `hits2_<d>_held` of every eval; the same recomputation of the full set and of
  the hard subset must equal the logged `<d>_held_2way` / `<d>_held_2way_hard` at every eval and their metrics, and the
  re-derived split, masks, heuristic lines and table sha must equal the run's, else the strict read-out refuses).
- **Pre-registered:** a NOT SHOWN on stars or landmarks is reported as **'NOT SHOWN with this site (many shared site
  tokens)'**: it says nothing about the target in pretraining with another site.

### A1.4 The colours shortcut (audit item 4)
- **Baseline for every anchor domain** (seed-specific, from that seed's training split; the audit's heur2.py rule): for
  a held-out 'a: b or c', answer the option with the higher win rate (times it was the answer / times it was an option)
  among the training questions with the same anchor a; never seen with a, or equal rates -> the answer-frequency line
  (0.5 on its tie). Seeds 0-7 mean, on every held-out question / on the build's heuristic-hard subset [min, max]: stars
  49.8 / 9.7, landmarks 65.5 / 10.3, languages 86.8 / 12.0, **colours 84.3 [82.9, 86.0] / 69.6 [68.0, 73.1]**, cities
  69.7 / 13.8 (= heur2.py on seeds 0 and 1 to the printed digit, incl. the subset sizes). Printed as the third heuristic
  line next to each anchor domain's read-out and next to its hard-subset line.
- **Pre-registered (colours)** [A2: FINAL3 instead of AREA, A2.3]: a PASS (the three conditions) must also beat the baseline on the hard subset: the mean
  over the 8 seeds of only-colours@pick's `colours_readout_hard` AREA > the mean over the same seeds of the baseline's
  accuracy on that subset; otherwise the verdict is **NOT SEPARABLE (from a training-frequency shortcut)**, not "shown
  effective in pretraining". The area (the primary metric's form, early chance-level evals included) makes this
  conservative against a PASS. Other domains: the baseline is descriptive. The FLOOR tag is unchanged.

### A1.5 Minor items
- **Decomposition check** (audit 7) [A2: the MAX, A2.3]: CHECK_PY's `grad_T` vs `grad_total` check reads the on-window evals only (logged at
  >= 90% of them; median relative difference < 0.05); none (no window): every eval (grad_T = grad_lm vs the backward). The
  pre-A1 median over all 75 evals could not fail (off-window grad_T = grad_lm).
- **Lambda check** (audit 7): CHECK_PY receives the job's `$ARGS` (its 8th argument; the trainer's own command line):
  every flag = the run's args, and the INTENDED lambda per domain is parsed from it (lam_rule each: lambda_d = --lam for
  each --prior_domains entry, 0 otherwise) and compared with the logged lam_domains, domains.<d>.lam and
  gnorm_prior_<d> / gnorm_true_<d> at every on-eval; with the on-window decomposition check this ties the lambda of the
  backward to the job's command line. The intent-table check stays as a second path.
- **Tier L** on every anchor domain's verdict line (and its sub-lines) of the read-out.
- **Months stays in the Holm family of six**: it is an expected floor (rb6 tier S at the floor in every arm), so it costs
  the other five alpha / 6 instead of alpha / 5 (the user's design; stated on the months line of the read-out).
- Wording: the picks "double path" is a consistency check (same code run twice; audit 9): renamed picks_consistency.
- Not changed by A1 (no decision): audit 5 (the fixed domain pairing in a row: months with stars, landmarks with
  languages, colours with cities; the cross-domain matrix is confounded with row co-occurrence), 6 (per-batch Gram
  standardisation; target entailment 98.3-100 per batch, not exactly F81's 99-100), 8 (FLOOR wording with heuristic
  lines), 11 (atlas CSVs / package versions not checked before training), 12 (step-0 lm tolerance 1e-3 verified on A100
  only: read the first finished job's check.json), 13 (months' cost).

### A1 cost (1.3 L40S-h per run, unchanged basis) [A2: the expected cost under A2 is in A2.2]
| | runs | L40S-h |
|---|---|---|
| stage A (none 5 lrs + 6 prior arms x 3 lrs, seeds 0-3) | 92 | 119.6 |
| stage X (max: one per prior arm + none at both edges) | 0-32 | 0-41.6 |
| stage C (one none pick - six none picks) | 28-48 | 36.4-62.4 |
| **nominal (no X, one none pick)** | **120** (was 112) | **156.0** (was 146) |
| **worst case** | **172** (was 160) | **223.6** (was 208) |

Both now exceed the 150 L40S-h mark (the nominal by 6 h); open question 1 stands with these numbers. Open question 2
(lr-matched LM row) is answered by A1.1; open question 5 is now gated in code (A1.2).

### A1 tests (re-run after the change; logs in runs/ptdom/tests/)
| test | script / log | result |
|---|---|---|
| read-out + selection self-test (planted known answers, A1 grids) | readout_ptdom.py --selftest / tests/selftest_readout.out | SELFTEST PASS 47/47: the build's 31 cases on the A1 grids, plus extend_rule on 6 hand-made mean vectors (pick open; the audit's case = argmax at the lower edge within 2 while the tie-break picks inside -> open; an edge pick by the tie-break with the argmax inside -> closed; pick open lower + argmax upper -> the pick's edge; ...); in the chain: only-landmarks (argmax 5e-4 at the edge, pick 1e-3 by offline bpb) -> X 2.5e-4 -> C 1e-3; only-languages (edge pick by tie-break, argmax inside) -> closed; none|landmarks (the audit's case) -> X 5e-4 shared with none\|stars -> C 2e-3; X / C cells; the colours rule (three conditions hold, hard-subset area 70.5 <= planted baseline 71 -> NOT SEPARABLE, vs 69 -> PASS; stars untouched); the stars / landmarks wording; Tier L tags and the months line; the unique-site line read back; the same-lr LM row (-5.00 on 8 paired seeds at 1e-3, +10.00 on 4 at 2e-3); the gate (0.3 -> FAILED; an unfinished run not counted; a planted logged-vs-rb6 disagreement fails); the read-out refuses without the sub-read-outs |
| anchor sub-read-outs on real data | tests/a1_anchor_test.py / tests/a1_anchor_test.out | A1 ANCHOR PASS 14/14: shared-site counts = the audit's; table sha = the CPU reference of all 8 seeds; the re-derived split = the references (seeds 0-7); seed-0 heuristic lines = the table above and = the smoke JSON's exactly; the anchor-conditional baseline = heur2.py on seeds 0 / 1 (held, hard, subset sizes); run_subsets on the REAL smoke JSONs (both arms, 5 domains): question order, masks, heuristic lines, full and hard series at every eval = the logged ones, areas = the metrics; tampers (a flipped hit, a wrong table sha, a wrong split seed) reported; the seeds 0-7 table of A1.3 / A1.4 |
| auxiliary-loss gate | tests/aux_gate_test.py / tests/aux_gate_test.out | A1 AUX GATE PASS 8/8: select_ptdom --stage A at only-stars share 0.3 -> exit 2, AUX GATE FAILED, the (temp) HOLD names only-stars, no picks file; at 0.9 -> a decision with amendment A1 and the gate summary; aux_gate_check on partial curves: 0.4 -> exit 1, 0.9 -> exit 0, a truncated JSON skipped and reported, before the window -> exit 2, a logged share 0.05 off the rb6 re-derivation -> exit 1; on the REAL smoke JSONs the stars window-median LM share 0.098 (the audit's 0.083-0.099), re-derivation max diff 1.4e-6, clipped 3/3; runs/ptdom/HOLD never created |
| per-run CHECK known answers | tests/check_py_test.py / tests/check_py_test.out | CHECKPY PASS 12/12: the smoke runs with the smoke's own command line as $ARGS: only the 2 GPU-job checks fail (+ the tiny model's CKA bar for only-stars); "the job's ARGS (29 / 23 flags) = the run's args"; lambda from ARGS = logged; on-window decomposition median 4.8e-4 over 3 / 3 evals; new tampers fail their check: --lam 2 in ARGS, the none arm's ARGS on the stars run, grad_T_reldiff 0.3 at the 3 on-window evals (the pre-A1 all-eval median 1.5e-5 would have passed); the build's tampers still fail theirs (lambda 2 in the intent now fails the intent-table checks only) |
| job files | tests/jobcheck.py / tests/jobcheck.out | JOBCHECK PASS 5/5: 92 stage-A files (23 cells x seeds 0-3, seed-major) rendered; 92/92 bash -n, intent, CHECK_PY byte-identical, ARGS and "$ARGS" passed to CHECK_PY, header + HOLD + L40S guard; no stale job file; the input-check block on the CPU dry logs: own cells exit 0 (7/7, incl. the new none@1e-3 s0), another seed exit 3 (4/4), another arm exit 3 (10/10) |
| select -> generator plumbing | tests/plumbing.py / tests/plumbing.out | PLUMBING PASS 13/13: 92 synthetic runs -> gate passed, X = only-stars@4e-3 + none@3.2e-2, C = 5 prior picks + none@8e-3; X / C render 8 / 24 jobs; refusals (overwrite, missing cell, X two steps out, another design, a picks file without amendment A1, none@2e-3 or none@2.5e-4 as X); none@5e-4 accepted as X |
| CPU references of the new none lrs | tests/a1_refs_check.py (a1_cpu_chain.sh) / tests/a1_cpu_chain.out | A1 REFS PASS 8/8: dry none@1e-3 s0: DRY rows 322bf3c5911b (84 lines), splits, table shas = the seed-0 reference; step 0 none@2e-3 s0: lm 11.615025520324707 and data sha 160aa036eddf3f35 = the reference (exact); refs_ptdom.py table (no --write): TABLE PASS |
| CPU references, full re-run | tests/a1_refs_full.sh, a1_refs_compare.py / tests/a1_refs_full.out | PARTIAL: dry none@8e-3 s0 and s1 re-run (= the references; s1 every field but the wall clock); the other 16 [A2: 17 of the script's 19] cells NOT re-run (stopped: 485-644 s per cell on the shared 2-CPU node); a1_refs_compare: every stored cell (21) = ptdom_refs.json, which is not rewritten |
| 200-step CPU smoke re-run | smoke_200.sh tests/smoke_a1, a1_smoke_compare.py / tests/smoke_200_a1.out | SMOKE PASS 13/13; the re-run = the original bitwise (run JSON except the wall clock and the out path; 36 / 36 tensors) in both arms |
| generator unit tests | tests/test_generators.py / tests/test_generators.out | GENERATORS PASS 52/52 (unchanged) |
| trainer regression | tests/reg_trainer.sh / tests/reg_trainer_a1.out | PARTIAL: default and rb6all6 IDENTICAL again (run JSON except code_sha / out / wall clock, 36 / 36 tensors bitwise, dry and run stdout; tests/reg_compare_a1_partial.out); rb6none NOT re-run (stopped: ~25 min per tiny run on the shared node; the build's reg_trainer.out has it IDENTICAL and the trainer file is byte-identical) |
| frozen rb6 generators / read-outs | tests/reg_frozen.py / tests/reg_frozen.out | FROZEN PASS (unchanged) |
| sbatch --test-only | make_jobs_ptdom.py --stage A --test-only 3 / tests/sbatch_test_only.out | 3/3 accepted (none@1e-3 s0, none@2e-3 s0, none@4e-3 s0: l40s_public, estimated start 2026-11-04 11:45 on gl033); nothing submitted (no SUBMITTED_ flag, no ptdom job in the queue) |
| submission script (print) | submit_ptdom.sh / tests/submit_print.out | plan 92 / 120 runs = 156.0 / 172 runs = 223.6 L40S-h; the snapshot verified; self-test pass; 92 jobs listed; the aux_gate_check step printed; submitted: no |


Not done in A1: a fresh adversarial review of the A1 diff (experiment-audit item v) is for the coordinator to launch; the
16 [A2: 17, see A2.3] remaining reference cells were not re-run (the trainer and runs/ptdom/code are byte-identical to the build's; the new
cells, two old cells and the smoke reproduce exactly).

## Amendment A2 (coordinator, before any submission)

2026-11-04, written before any run of this batch (nothing submitted: no SUBMITTED_ flag, no ptdom job in the queue, no out
dir; only `sbatch --test-only`). Source: the fresh adversarial review of A1, tests/audit_a1.md (important 1-2, minor 3-9).
The coordinator decided A2.1-A2.3; they are implemented exactly and are part of the pre-registration. Where A1 or the build
text disagrees, A2 holds (the superseded lines carry an [A2] marker). Code: runs/ptdom/readout_ptdom.py (window_split /
window_grad / cell_gate / aux_gate; extend_rule / select_unit / decide; the gate at the pick and the GATE verdicts;
shortcut_rule; pair_diff; cell_dirs / load / integrity for re-runs), select_ptdom.py, aux_gate_check.py,
llmadd_torch/make_jobs_ptdom.py (CHECK_PY's decomposition check and info line; job_name / render(rr) / rerun_plan /
`--rerun`; the `--submit` refusal; picks files must carry amendment A2), submit_ptdom.sh (the re-run path). The trainer and
runs/ptdom/code are UNCHANGED (21cdc14, pretrain_range.py bb34927f16d8; the 12 snapshot shas re-verified by `--snapshot`).

### A2.1 The auxiliary-loss gate, per (arm, lr) cell (audit_a1 important 1, minor 3, minor 4)
- **Rule** (the user's: "the prior is only an auxiliary loss; the LM gradient must dominate"), judged per (arm, lr) CELL,
  not per arm. **Statistic: the LM share by norm a / (a + b)**, a = `grad_lm`, b = `grad_prior` (the pre-clip norms of the
  trainer's autograd decomposition; b = |sum_d lambda_d g_d|), at the on-window evals only (prior_on = 1, before a
  divergence). It is < 0.5 exactly when b > a, whatever the cosine of the two gradients. A1's grad_lm / |g_lm + g_prior|
  passed a prior gradient up to 1.73 x the LM's at cos 0, and any opposing pair (share > 1).
- **Per run** the median over its on-window evals; **per cell** the median over its runs of those run medians. **A cell
  FAILS** if that median is < 0.5, or if one of its runs is unverifiable:
  - a gradient field (grad_lm, grad_prior, grad_cos, grad_lm_share, clip_coef) is missing or not finite at an on-window
    eval (A1 skipped such evals: audit_a1 minor 3);
  - the run has no on-window eval;
  - the logged grad_lm_share disagrees with its re-derivation a / sqrt(a^2 + b^2 + 2ab cos) by > 1e-3 (A1's double path
    of the logged fields, kept).

  Exactly 0.5 (b = a) passes. Every cell's line prints the share of its on-window evals with b > a and with clip_coef < 1.
- **(i) Selection.** select_ptdom.py applies the gate before any decision. The gate lives in readout_ptdom.decide, so the
  read-out re-derives it. It covers every finished stage-A / X prior cell on seeds 0-3.
  - A failing cell is reported and **excluded** from its arm's selection: it cannot be picked, and the selection runs over
    the admissible lrs.
  - Decided here: an open side whose next lr is a gate-failed cell is **closed** ("closed by the gate"). The next lr is one
    factor-2 step beyond the admissible edge; that lr was run and is not an auxiliary-loss setting. So no X and no STOP
    follows from that side.
  - **Every cell of an arm fails** → "AUX GATE FAILED" and no pick. The prior unit becomes GATE. The domain's none unit
    becomes GATE too (no comparison, so no X / C is spent on it). The reason is appended to runs/ptdom/HOLD and the
    selection exits 2.
  - The picks file is still written, with the other units' decisions. A1 wrote none at a gate failure; one failing arm no
    longer blocks the other units' decisions. HOLD still blocks every submission until the user decides.
- **(ii) Read-out.** readout_ptdom.py runs the gate before printing any verdict, on the picked prior cell with ALL its
  finished seeds 0-7, so the confirm seeds are gated too (audit_a1 minor 4).
  - A failing pick prints **"AUX GATE FAILED at the pick"** instead of a verdict (no PASS and no NOT SHOWN). The domain
    stays in the Holm family of six with its t p.
  - A GATE domain prints "AUX GATE FAILED (every cell of the prior arm fails: no pick)" and enters Holm with p 1.0.
- **(iii) Watcher.** aux_gate_check.py judges per cell on the partial runs (the latest dir of each cell).
  - Exit 1 if a cell's median so far is < 0.5, or a run with on-window evals is unverifiable.
  - Runs before the window are listed, not judged. Exit 2 before any on-window eval.
- **Names.** rb6's grad_split and F88's "0.87-0.92" are grad_lm / |g_T|. a / (a + b) is the by-norm share of FINDINGS
  (e.g. the 1B gradient line, 0.91). This batch uses a / (a + b) by the coordinator's decision. CHECK_PY's info line now
  also logs it (lm_share_norm).
- **Real-data check.** On the real smoke JSONs (tiny d64 x 3) the stars cell's on-window median a / (a + b) is 0.0896 (A1's
  a / |g_T|: 0.098); an independent recomputation from the JSON fields agrees. That tiny cell FAILS, as expected for the
  tiny model (the build's open question 5).

### A2.2 The extension trigger (audit_a1 important 2)
- **Rule.** F87 rule 1's 2-point margin for every unit: readout_rules.bracket 'open', the pick at a grid edge beating its
  neighbour by > 2. The pick's open edge comes first.
- **One exception, kept from A1 (the lr-confound fix).** At stage A, a none unit whose area argmax (the first maximum) is at
  the LOWER edge of none's grid (1e-3) takes its one extension to 5e-4, even without the margin.
- **After an extension,** STOP needs the 2-point margin; the exception does not apply.
- **Dropped.** A1's other trigger cases are gone: a prior arm's argmax at an edge within 2, none's argmax at the upper edge
  within 2, and a STOP by an argmax at the new edge.
- **Cost method.** tests/a2_cost_mc.py simulates the expected cost (log: a2_cost_mc.out). It calls the real
  readout_ptdom.decide on synthetic stage-A / X runs, 1,000 iterations per scenario. The noise is independent across
  domains, arms and seeds, with seed-mean sd sigma. The gate passes everywhere. Cost = 92 + 4 |X| + 4 |C| runs x 1.3 L40S-h.
- **Cross-check.** The same simulator run with the superseded A1 rule (tests/a1_rule_ref.py, the A1 functions verbatim; log
  a2_cost_mc_a1rule.out) reproduces the A1 audit's stop_mc.py. For flat, sigma 0.5 it gives prior P(X) / P(STOP) 0.666 /
  0.174, none 0.398 / 0.065, NOT FINAL 0.230; the audit had .66 / .16 / .41 / .07 / .22.

| scenario | prior unit P(X) / P(STOP) | none unit P(X) / P(STOP) | P(a domain NOT FINAL) | E[runs] | E[L40S-h] (p95) | A1 rule: E[L40S-h], NOT FINAL |
|---|---|---|---|---|---|---|
| flat, sigma 0.5 | 0.003 / 0 | 0.206 / 0 | 0 | 131.8 | 171.4 (176.8) | 190.0, 0.230 |
| flat, sigma 1.5 | 0.231 / 0.008 | 0.305 / 0.006 | 0.013 | 139.5 | 181.4 (192.4) | 189.9, 0.257 |
| weak peak 0.25 pt/step^2, sigma 1 | 0.072 / 0 | 0.093 / 0 | 0 | 131.1 | 170.4 (182.0) | 186.0, 0.059 |
| peak 0.5 pt/step^2, sigma 1 | 0.047 / 0 | 0.025 / 0 | 0 | 127.4 | 165.6 (176.8) | 180.6, 0.017 |
| confound: none +1 pt per halving toward low lr, prior weak peak, sigma 1 | 0.072 / 0 | 0.717 / 0.107 | 0.107 | 129.5 | 168.3 (176.8) | 181.3, 0.536 |

- **Reading the table.**
  - Expected X runs fall from 14-22 (A1) to 2-10 (A2).
  - The NOT FINAL rate falls from 2-26% to 0-1.3%. The exception is the confound scenario: 10.7%, against 54% under A1.
    There none keeps rising toward low lr after the extension, and the margin rule then STOPs on purpose.
  - The expected total is still above the nominal 120 runs / 156 h in every scenario. The reason is stage C, not X:
    none's per-domain picks scatter under independent noise. E[C] is 32-37 runs, about 2-3 distinct none picks besides the
    six prior picks. Read-outs that move together across domains would give fewer.

**A2 cost (1.3 L40S-h per run, unchanged basis)**

| | runs | L40S-h |
|---|---|---|
| stage A (none 5 lrs + 6 prior arms x 3 lrs, seeds 0-3) | 92 | 119.6 |
| stage X, expected (A2 simulation) | 1.7-10.3 | 2.2-13.4 |
| stage C, expected | 31.7-37.2 | 41.2-48.4 |
| **expected total (A2 simulation, five scenarios)** | **127-140** | **166-181** (p95 177-192) |
| nominal (no X, one none pick) | 120 | 156.0 |
| worst case (unchanged) | 172 | 223.6 |

### A2.3 Minor items (audit_a1 minor 5-9)
- **Same-lr LM row (minor 7).** A diverged run counts as the worst (bpb +inf) and is never dropped. This is
  readout_ptdom.pair_diff, used by every LM pair row.
  - Prior diverged → +inf; none diverged → -inf; both diverged → 0 (a tie at the worst). n counts these runs.
  - The line prints the diverged counts and the median. A missing value (no offline file) is still skipped.
- **HOLD exits and re-runs (minor 6).** The path is documented and implemented.
  - **A job name is submitted once.** make_jobs_ptdom.py `--submit` refuses every name that has a SUBMITTED_ flag and
    prints the refusal.
  - **Who gets a re-run.** A cell whose latest submission exited before its run finished is re-run under a NEW name
    <name>-r1 (then -r2, ...) with `--rerun`, at stage A, X or C. Typical cases: a pending job that started while HOLD
    existed (it exits 0 before training and its flag stays), or a crash.
  - **Eligibility.** All three must hold:
    - the latest name has a flag;
    - its job is not in the queue (checked by the flag's job id and by the job name, via squeue; if squeue fails, the
      re-run is refused);
    - its out dir holds no finished run JSON (75 evals). A finished run is data and is never re-run, even when its check
      failed.
  - **What a re-run is.** A fresh run in its own out dir. Its intent and args equal the original's except `--out`. The
    trainer `--tag` is unchanged, so the run JSON keeps its name. The loader reads a cell's latest dir, and integrity()
    refuses a cell with a finished run in an earlier dir.
  - **When.** Only after the USER removes HOLD: run `--stage A --rerun --print`, then `--rerun --submit`.
    submit_ptdom.sh prints this path.
- **Colours rule (minor 8), pre-registered now, before any result.** A colours PASS also needs the mean over the 8 seeds
  of only-colours@pick's `colours_readout_hard` FINAL3 accuracy to exceed the mean anchor-conditional baseline accuracy
  on the same hard subset (69.6 [68.0, 73.1] over seeds 0-7). Otherwise the verdict is NOT SEPARABLE. A1 compared the
  AREA, whose early near-chance evals could have made a PASS impossible.
- **Mislabelled logs (minor 9).**
  - tests/a1_refs_full.out: its last line is relabelled "A1 REFS PARTIAL RE-RUN". Only 2 of the 19 cells of
    a1_refs_full.sh were re-run; A1's "the other 16" should read 17. The compare read all 21 stored cells.
  - tests/reg_compare_a1_partial.out: "REGRESSION FAIL" is relabelled "REGRESSION PARTIAL" (rb6none was not run; nothing
    differed). tests/reg_trainer_a1.out gets a pointer line. The original labels are quoted in place.
  - The scripts now print the right labels themselves: a1_refs_compare.py prints "STORED CELLS", and reg_compare.py prints
    PARTIAL for a cell that was not run.
- **Decomposition check (minor 5).** CHECK_PY's grad_T vs grad_total check now takes the MAX over the on-window evals
  (< 0.05; none: every eval). A1 took the median.
  - GPU evidence: 72 Rev D runs on Torch (runs/ptheld2t, 53M) have a maximum grad_T_reldiff of 7.1e-4 over their
    prior-applied evals.
- **Job files.** The CHECK_PY block is part of every job's text, so the 92 stage-A files were re-rendered (`--stage A`, no
  submission).
  - All 92 differ from the A1 files by the same 9 diff lines inside that block: the max check with its comment, and the
    info field.
  - The header, ARGS, intent and input checks are unchanged.

### A2 tests (re-run after the change; logs in runs/ptdom/tests/)
| test | script / log | result |
|---|---|---|
| read-out + selection self-test (planted known answers, A2 rules) | readout_ptdom.py --selftest / tests/selftest_readout_a2.out (= selftest_readout.out, rewritten by submit_ptdom.sh) | SELFTEST PASS 61/61. Covers: the A2 extend rule on 11 hand-made mean vectors (the none exception at the lower edge with and without the margin; none's upper-edge argmax within 2 closed; a prior arm's edge argmax within 2 closed; after X, an argmax at the new edge within 2 closed); the planted chain under A2 (only-landmarks closed → C, where A1 had X; none\|languages upper-edge argmax closed, where A1 had X; none\|stars after X has its argmax at the new lower edge within 2 → C, where A1 would STOP; none\|landmarks takes the none exception → X 5e-4; none\|cities STOPs by the margin); every A1 verdict known answer; the colours final3 rule (final3 72.5 vs baseline 73 → NOT SEPARABLE; vs 71 → PASS, where A1's area 70.5 gives NOT SEPARABLE); the gate per cell (0.3 fails; 0.5 passes; a logged-share disagreement, a missing grad_cos, b = 1.5 a at cos 0 and at cos -0.9 all fail; an arm failing in every cell; an unfinished run not counted); exclusion and closing; (ii) an 8-seed pick median of 0.45 → "AUX GATE FAILED at the pick" instead of its PASS; GATE units in decide and the read-out; re-run dirs in the loader and integrity(); a diverged run counted as the worst in the LM rows |
| auxiliary-loss gate incl. the A1 auditor's counter-examples | tests/aux_gate_test.py / tests/aux_gate_test_a2.out | A2 AUX GATE PASS 19/19. Through select_ptdom: CX1 (pick at 0.3, others 0.9) → the cell fails and is excluded, pick 5e-4, no HOLD; CX1b (5 of 12 at 0.3) → 1e-3 fails, 2e-3 (median 0.6) passes; CX2 (b = 1.5 a) → 0.400 fails; CX3 opposing (b = 1.2 a, cos -0.9) → 0.455 fails, while the b = a boundary passes; every cell failing → AUX GATE FAILED, GATE units, HOLD, exit 2, the other units decided; a failing argmax cell closes the open side (control without it: X 4e-3); a missing grad_prior fails. Through aux_gate_check: CX6 → exit 1 (A1: 0); CX7 (no grad_cos) → exit 1 (A1: 0); missing grad_prior, opposing gradients and a logged-share mismatch → exit 1; a truncated JSON is skipped; a pre-window run is listed; only pre-window runs → exit 2; a -r1 dir supersedes its first dir. Real smoke JSONs: a / (a + b) 0.0896 = an independent recomputation; the tiny cell fails. The real HOLD was never created |
| re-run path | tests/a2_rerun_test.py / tests/a2_rerun_test.out | A2 RERUN PASS 12/12. rerun_plan: HOLD exit → -r1; unfinished → -r1; -r1 HOLD-exited → -r2; queued (by id or by name) → refused; finished → refused; never submitted → not listed; squeue failing → refused. render(rr=1) differs from rr=0 in exactly 7 diff lines (names / OUT / --out / comment) with an identical intent; bash -n passes and **sbatch --test-only accepts the -r1 file**. main() on a temp copy with sbatch recorded (never sent): 91 flagged names refused by --submit and 1 submitted; --rerun lists and submits exactly the 92 -r1 jobs; HOLD blocks --submit. On the real runs/ptdom, --rerun gives 0 jobs and no file is created |
| expected cost | tests/a2_cost_mc.py / tests/a2_cost_mc.out, tests/a2_cost_mc_a1rule.out (with tests/a1_rule_ref.py) | the A2.2 table; the A1-rule run reproduces the A1 audit's probabilities |
| per-run CHECK known answers | tests/check_py_test.py / tests/check_py_test_a2.out | CHECKPY PASS 14/14: A1's 12 checks, plus grad_T_reldiff 0.3 at ONE of 3 on-window evals failing the max check (A1's on-window median 4.8e-4 would pass), plus check.json info lm_share_norm 0.0896 = readout_ptdom's |
| select → generator plumbing | tests/plumbing.py / tests/plumbing_a2.out | PLUMBING PASS 15/15: A1's 13, plus an A1 picks file refused and an A2 decision carrying the 18-cell gate summary |
| job files | tests/jobcheck.py / tests/jobcheck_a2.out | JOBCHECK PASS 5/5 on the 92 re-rendered files (bash -n, intent, CHECK_PY byte-identical, ARGS, header + guards, no stale file, input-check block 7/7 own, 4/4 seed, 10/10 arm). All 92 differ from the A1 files by the same 9-line diff (the CHECK_PY max and its info field) |
| anchor sub-read-outs (the loader changed) | tests/a1_anchor_test.py / tests/a1_anchor_test_a2.out | A1 ANCHOR PASS 14/14 (unchanged) |
| sbatch --test-only | make_jobs_ptdom.py --stage A --test-only 3 / tests/sbatch_test_only_a2.out | 3/3 accepted (none@1e-3 / 2e-3 / 4e-3 s0, l40s_public, gl027), plus the -r1 file above. Nothing submitted: no SUBMITTED_ flag, no ptdom job in the queue |
| submission script (print) | submit_ptdom.sh / tests/submit_print_a2.out | plan 92 / 120 / 172 runs; the snapshot verified; self-test 61/61; 92 jobs listed; the A2 watcher and re-run path printed; submitted: no |

Not re-run in A2 (their code is unchanged): the generator unit tests, the trainer regression, the frozen rb6 read-outs, the CPU references and the smoke. The trainer, runs/ptdom/code, the split / site code and the frozen files are untouched.

The A1 versions of the changed tests are kept as records: tests/aux_gate_test_a1.py and tests/plumbing_a1.py. They test the superseded A1 semantics and are not expected to pass on the A2 code.

Not done in A2: a fresh adversarial review of the A2 diff (experiment-audit item v), for the coordinator to launch.

## Amendment A3 (user decision 2026-11-04: 24 candidate domains, four batches, two stages)

2026-11-04, written before any run (nothing of ptdom was ever submitted: no SUBMITTED_ flag, no ptdom job in the queue, no
out dir; only `sbatch --test-only`). Source: the USER DECISION of 2026-11-04 relayed by the coordinator: screen 24
candidate domains in 53M pretraining as fast as possible, in two stages, then pick the six that pass (plus numbers and
dimensions) for a later six-manifold run. **A3 supersedes A1 / A2 where stated** (the lr design: stage A / X / C, the
grids, the extension and STOP rules, months in a Holm family of six, the colours-only shortcut rule, the A1 / A2 cost); it
**keeps** the A2.1 per-cell auxiliary-loss gate (the same code), the A1.4 / A2.3 shortcut baseline (now for every
comparison domain, final3), the A1.3 shared-site statistics (every domain; a wording threshold), the Tier L tags, the A2.3
re-run path and diverged-run rule, and the per-run CHECK (generalised). The A2 build's files are archived unchanged in
runs/ptdom/superseded_A2/ (its 92 job files, the A2 scripts, the A2 snapshot code_revE). The A2 review (tests/audit_a2.md,
another session, 04:21-04:34) read the A2 versions; its items that carry over are handled in A3.9.

### A3.1 Batches and domains
Each batch is self-contained: identical data for every arm of a seed, its own none arm, ONE GPU type, `--domains
numbers,<its six>`, K = 2 domain lines per row cycling the six (fixed trainer order = the order below), the numbers line
(2%) first after the text, 3e8 tokens. letters, years and number_words are not used (harmful in fine-tuning, F81).

| batch | GPU | domains (trainer order) | lines in a row (fixed pairing, 64 mod 6 = 4) | token share % (seed 0) |
|---|---|---|---|---|
| A | L40S [A4: B200] | months, stars, landmarks, languages, colours, cities (the A2 build's data and templates, unchanged) | months+stars, landmarks+languages, colours+cities | text 90.1, numbers 2.124, domains 0.46-1.78 |
| B1 | B200 | countries, animals, country_size, animal_size, people, units (PAIR2) | countries+animals, country_size+animal_size, people+units | text 90.8, numbers 2.124, domains 1.04-1.34 |
| B2 | B200 | zodiac, compass, planets, elements, prefixes, ranks (OFFSET: zodiac / planets / elements) | zodiac+compass, planets+elements, prefixes+ranks | text 91.4, numbers 2.124, domains 0.96-1.38 |
| B3 | L40S [A4: B200] | chinese_zodiac, kinship, life_stages, geo_periods, greek, scalar_adj (OFFSET: chinese_zodiac / geo_periods / greek) | chinese_zodiac+kinship, life_stages+geo_periods, greek+scalar_adj | text 90.7, numbers 2.124, domains 0.93-1.52 |

Per domain (seeds 0-7 mean [min, max]; tests/a3/test_domains_a3.out, a3_sizes.out). k-step = the k-way read-out over the
n items (chance 1/n; months: rb6's tier S, 1/12); comparison = the 2-way read-out (chance 50), its heuristic lines (answer
frequency / shorter option), the hard subset (both wrong or tied) and the training-frequency SHORTCUT baseline (A1.4's rule;
without an anchor the option's training win rate) on held / on the hard subset; elimination = the k-step Latin-square
shortcut (A3.6); exposures = 195,307 domain lines per run / training questions; shared = items whose site token another item
also has. Batch A: the A2 tables above (A1.3 / A1.4, unchanged).

| domain | kind, target | n items | train / held | exposures | chance | answer freq. / shorter | hard n | shortcut held / hard | elim. | shared |
|---|---|---|---|---|---|---|---|---|---|---|
| countries | comparison + anchor, sphere | 70 | 1812 / 1188 | 108 | 50 | 71.5 / 48.6 | 283 [216, 310] | 73.3 / 11.4 | - | 2 |
| animals | comparison + anchor, tree (raw Gram) | 65 | 1788 / 1212 | 109 | 50 | 74.9 / 49.8 | 261 [242, 288] | 77.5 / 15.0 | - | 0 |
| country_size | comparison 'more', log | 70 | 1006 / 672 | 194 | 50 | **100.0** / 55.5 | **0.2 [0, 2]** (7 seeds empty) | 99.9 / 100 | - | 2 |
| animal_size | comparison 'more', log | 65 | 1800 / 1200 | 109 | 50 | **98.8** / 45.8 | 16.5 [8, 22] | 99.6 / 73.5 | - | 0 |
| people | comparison 'less', rank | 106 | 1800 / 1200 | 109 | 50 | **92.0** / 45.4 | 73.5 [58, 100] | 95.4 / 68.1 | - | 10 |
| units | comparison 'within', within (raw) | 24 | 100 / 66 | 1,953 | 50 | 77.5 / 55.9 | 17.2 [10, 24] | 82.0 / 51.2 | - | 2 |
| zodiac | k-step cyclic (k 1..24), circle | 12 | 180 / 108 | 1,082 | 8.3 | - | - | - | 47.3 [40.2, 54.7] | 2 |
| compass | comparison + anchor 'angle', circle | 8 | 176 / 112 | 1,113 | 50 | 50.0 / 50.0 | 60 [52, 70] | 82.3 / 69.3 | - | 0 |
| planets | k-step linear (k 1..3), line | 8 | 12 / **6 [5, 7]** | 16,276 | 12.5 | - | - | - | 28.0 | 0 |
| elements | k-step linear (k 1..3), line | 36 | 64 / 38 | 3,070 | 2.8 | - | - | - | 7.0 | 8 |
| prefixes | comparison 'more', line | 11 | 64 / 42 | 3,052 | 50 | 83.0 / 39.6 | 8.8 [4, 12] | 78.0 / 41.4 | - | 3 |
| ranks | comparison 'more', line | 8 | 26 / **16** | 7,512 | 50 | **90.6** / 50.0 | **2.5 [0, 6]** (1 seed empty) | 93.8 / 65.5 | - | 0 |
| chinese_zodiac | k-step cyclic (k 1..24), circle | 12 | 182 / 106 | 1,076 | 8.3 | - | - | - | 50.6 [45.2, 52.8] | 0 |
| kinship | comparison 'more', generation | 21 | 220 / 148 | 888 | 50 | **91.9** / 50.1 | 14.0 [10, 26] | 93.6 / 50.3 | - | 0 |
| life_stages | comparison 'more', line | 7 | 18 / **12** | 10,850 | 50 | 85.4 / 50.0 | **3.0 [0, 8]** (1 seed empty) | 90.6 / 66.7 | - | 0 |
| geo_periods | k-step linear (k 1..3), line | 12 | 20 / **10 [8, 12]** | 10,016 | 8.3 | - | - | - | 18.9 | **5** |
| greek | k-step linear (k 1..4), line | 24 | 53 / 33 | 3,702 | 4.2 | - | - | - | 9.9 | 0 |
| scalar_adj | comparison 'within', within (raw) | 31 | 68 / 44 | 2,872 | 50 | 82.4 / 45.7 | 13.2 [8, 22] | 84.4 / 47.6 | - | 0 |

(The k-step chance printed is 1/n; the trainer's chance_readout = max(1/n, the majority training answer's share of the
held questions), logged per seed. The comparison splits hold 40% of the option-pair keys: the held count is the same at
every seed for templates without an anchor, the held SET differs (tested).)

### A3.2 Trainer Rev F (tools/pretrain_range.py, branch ptdom, commit 1208bad, sha256[:12] e4f1c92482d2; not pushed)
- Opt-in: only a Rev F domain in `--domains` enables it; no new flag. NEWDOM = the 18 new domains (B1, B2, B3 order).
- Comparison domains (PAIR2_NEW) take the Rev E anchor path unchanged: ft36.pair_questions on manifold_domains tables, the
  option-pair split (ft36.split, 40%), sites at the LAST token of each item name in the prompt (anchor if the template has
  one, both options), the 2-way read-out with its heuristic lines and hard mask, the raw Gram for tree / within targets.
  Logged per domain: has_anchor, its stream.
- k-step domains (OFFSET_NEW) = ft36.offset_questions (manifold_domains.OFFSET's template, "Q: " prefix, k in words up to
  twenty; cyclic: k = 1..2n; linear: the template's k range inside the table), training line 'Q: <prompt> <answer>\n';
  split = ft36.main's make_split re-implemented (off_split: 40% of the keys; cyclic: the (x, k mod n) class, a held class
  with k mod n = 0 given back; linear: (x, k), held keys given back until every start item and every k is trained), with
  the asserts (no key on both sides; cyclic: held shifts trained; linear: held x and k trained; every answer = the k-step
  item); site = the last token of the start item's name (anc_line's asserts); target = the atlas primary (cyclic circle,
  linear line; asserted); read-out = the k-way choice [A4: the full-span choice, Rev G, A4.1; this first-token one is the secondary line]: argmax of the answer position's logit over the n items' FIRST tokens
  (distinct, asserted; every line's answer token = its item's candidate), the months path generalised to multi-token items
  (an exact-match hit is a k-way hit: the known answer kept); bootstrap unit = the split key; descriptive mask 'wrap'
  (cyclic).
- Streams: every new domain has its own SeedSequence(4300 + seed).spawn(54) children 3 i / 3 i + 1 / 3 i + 2 (split,
  shuffled target, eval contexts of NEWDOM[i]); the Rev B 4100 and Rev E 4200 streams are untouched (batch A's data and
  splits bit-identical). Training lines: dom_rng (4100 child 0) and order_rng (5000) as before; the checkpoint saves them
  (B200 resume).
- `--dry_inputs` prints per domain the site rule asserted on every line ("site rule: ... (the start item)" / "(both options;
  no anchor in the template)"), the k-way candidates, 2 train lines, 2 held + 1 seen eval rows (+ the other-option rows of
  the comparisons), the training rows per slot with the sites marked (experiment-audit vii); the DRY grep adds "  k-way
  candidates " (batch A has none: its DRY rows and references are unchanged).
- Regression (tests/a3/reg_a3.sh, reg_compare_a3.py -> reg_a3.out; cpu_short, one thread per process): the default path,
  rb6 all6-lam1 and rb6 none vs origin/main 93e22bf4c4c0, and the batch-A path (only-stars, only-months, none) vs Rev E
  bb34927f16d8: IDENTICAL (run JSON except code_sha / out / wall clock, 36 / 36 model tensors bitwise, dry and run stdout).
  Batch A's CPU references hold under Rev F with the A3 ARGS (tests/a3/run_refs_a3_A.out: DRY rows, splits, table shas,
  step-0 lm and data_sha at seeds 0 and 4 = ptdom_refs.json).

### A3.3 Stage 1 (every batch; pre-registered)
- Cells: seeds 0-3; prior arms (`only-<d>`, lambda 1, window (0.05, 0.4), ctx, lam_rule each) at lr 1e-3 / 2e-3; none at
  2e-3 / 4e-3 / 8e-3 (overlapping the prior at 2e-3: the same-lr LM row uses it) = 12 + 48 = 60 runs per batch, 240 in all.
- Picks per domain d (unit (only-d, d) and unit (none, d), seeds 0-3, d's held-out read-out AREA): F87 rule 1
  (readout_rules.best_lr 'top2': the argmax of the mean area; the top two within 2 points -> the lower mean bpb, offline
  full-val when every run of both has it). A2.1 gate first, per (arm, lr) cell over seeds 0-3 (a / (a + b), median of run
  medians < 0.5 or an unverifiable run fails); a failing cell is excluded from the selection; an arm with no admissible
  stage-1 cell -> GATE (no pick, the domain cannot advance), select writes HOLD and exits 2 after writing the picks file.
- **Screen (lenient):** d ADVANCES iff (i) gain_s = area(only-d@P, s) - area(none@N_d, s) > 0 at all of seeds 0-3, (ii) the
  gate passes at the picked cell (seeds 0-3), and (iii) comparison domains: the mean over seeds 0-3 of only-d@P's
  hard-subset FINAL3 > the mean over the same seeds of the shortcut baseline's accuracy on that subset; seeds with an empty
  hard subset are skipped; every seed empty -> (iii) fails ("the answer-frequency line solves every held-out question").
  [A4: (iii) on the pooled questions, >= 30 or NOT TESTABLE (not blocking), A4.5; and (iv) the same-lr gain at 2e-3 > 0 in
  4/4 seeds, A4.2]
  k-step domains (months included) have no (iii). Every domain is reported in the screen table (select_ptdom.py --batch X
  --after 1; readout_ptdom.py --screen X), with its gains, picks, argmaxes, gate median, shortcut line, elimination
  baseline (k-step), shared sites and the stage-2 cells.
- "Anchor domains" of the decision = every comparison domain (the 2-way family whose target carries the answers = Tier L),
  with or without an anchor in the template; templates without an anchor use the option's training win rate as the
  shortcut baseline (A1.4's rule with the anchor dropped).
- [A4: superseded by A4.3 (the noise scale was 1 point; audit_a3 important 4)] Under no effect the screen passes a domain with probability 0.036 (tests/a3/a3_null_mc.out, the real best_lr; below 1/16
  because none's pick is the best of 3 lrs and the prior's of 2), the stage-2 rule then with <= 0.0037 per domain (raw p;
  Holm only lowers it): about 0.09 false PASS expected over 24 null domains.

### A3.4 Stage 2 (advancing domains only; pre-registered)
- C cells: seeds 4-7 at the stage-1 picks (only-d@P_d, and none@N_d; a none lr once per batch).
- [A4: E cells at seeds 0-7; none also at its upper edge (1.6e-2) and a second lower step (5e-4); NOT FINAL; A4.2] E cells (seeds 0-3): ONE extension per unit whose stage-1 area argmax (the first maximum over the admissible lrs) is at a
  grid edge: none's lower edge 2e-3 -> 1e-3; the prior's lower edge 1e-3 -> 5e-4, upper edge 2e-3 -> 4e-3 (with two prior
  lrs the argmax is always at an edge: every advancing domain gets a prior extension); none's upper edge 8e-3: no extension
  (the user's list; A3.10 open 2). A none extension is shared by the batch's domains that need it.
- After the E cells (select_ptdom.py --batch X --after 2): every unit that ran an extension is re-selected over grid +
  extension (F87 rule 1, seeds 0-3; a gate-failing extension cell excluded); a moved pick -> stage 2b = seeds 4-7 there.
  No second extension; a final argmax at the extension lr -> tag EDGE (descriptive). Units of other domains do not add the
  extension to their grid (A2's rule).

### A3.5 Verdict (pre-registered; readout_ptdom.py, once every batch's stage 2 / 2b has finished)
Per advancing domain, seeds 0-7 paired at the final picks (gain_s = area(only-d@P*, s) - area(none@N*, s)):
- [A4: plus (d) the selection-free condition on seeds 4-7 (A4.3); the gate per half (A4.5); PROVISIONAL unless all four chains are complete (A4.4)] **PASS (shown effective in pretraining)** iff (a) the 8-seed paired bootstrap 90% CI of the mean gain (10,000 resamples,
  default_rng(0), percentile) excludes 0; (b) both seed halves positive (seeds 0-3 and 4-7); (c) the two-sided one-sample
  t-test of the 8 gains, **Holm-adjusted over ALL advancing domains of ALL batches together**, p < 0.05 (so the per-seed
  t interval excludes 0); and the mean > 0.
- The A2.1 gate at the picked cell over all its finished seeds (0-7) before the verdict: a failing pick -> "AUX GATE FAILED
  at the pick" (no verdict; it stays in the Holm family with its t p).
- Comparison domains: a PASS must also meet the shortcut condition on the 8 seeds (A3.3 (iii) at the final picks), else
  **NOT SEPARABLE** (from a training-frequency shortcut); their verdict lines carry [Tier L] ("given the structure, the model
  uses it"; never "discovered"). A NOT SHOWN on a domain where >= 20% of the items share their site token reads "NOT SHOWN
  with this site (many shared site tokens)" (A1.3's two: stars 34%, landmarks 21%; new: geo_periods 42%, prefixes 27%,
  elements 22%).
- Tags (never change a verdict): FLOOR (both arms <= max(chance, heuristic lines) + 5 [A4: + the shortcut, elimination and shortest-span lines, A4.5]), CEILING (none >= 95), SEEN < 80, HARM
  (the three conditions hold for -gain), EDGE. Every line prints the picks, gain, per-seed gains, wins, boot90, halves, t95,
  p, p_Holm, final3 gain, **seeds 4-7 alone** (the only half not used for selection or screening), chance, heuristics.
- Domains that did not advance keep their screen line ("not advanced at stage 1", with the reason); they are not in the
  Holm family.

### A3.6 What the results can and cannot say (written before any run)
- PASS = "the prior with this target helps this domain's held-out read-out in 53M pretraining"; no shuffled-target control
  (not geometry-specific); comparison domains Tier L. NOT SHOWN = no reliable gain under the rule; NOT SEPARABLE = a gain
  that the training-frequency shortcut already explains.
- **Order comparisons are dominated by the frequency shortcut.** For the 1-D 'more / less' domains the answer-frequency line
  alone solves 92-100% of the held-out questions (country_size 100.0, animal_size 98.8, people 92.0, kinship 91.9, ranks
  90.6, life_stages 85.4); the hard subset is empty or tiny (country_size empty at 7 of 8 seeds, ranks / life_stages
  0-8 questions). Under (iii) country_size is judged on at most 2 questions of a single seed (or fails outright when that
  seed is not among 0-3); ranks and life_stages on a handful. Expected: these screen out (NOT ADVANCE / NOT SEPARABLE)
  whatever the prior does.
- Tiny held-out sets: planets 6 [5, 7] questions, geo_periods 10, life_stages 12, ranks 16 (x 8 contexts): read-outs in
  steps of 2-17 points; their training questions are seen 7,500-16,000 times per run (memorisation is not the question).
- k-step domains: the Latin-square elimination shortcut (choose among items never the training answer after the same x nor
  for the same shift) reaches 47-51% on zodiac / chinese_zodiac (chance 8.3) and 28% on planets: an above-chance k-way
  read-out is not by itself evidence of the geometry (descriptive line, not a rule; the same holds for months in rb6).
- Selection and verdict share seeds 0-3 (picks, screen); the halves condition and the printed seeds 4-7 line are the
  unselected evidence.

### A3.7 Cost (runs x plan hours; make_jobs_ptdom.py --plan)
- Basis: L40S 1.3 h per run (A2's). B200: pt300x's 294M jobs took 4,007 s of training on B200 (13 runs, tests/a3_b200_timing.py)
  vs 14,645 s on L40S (pttpp: 0.7551 s/step + 10.92 s/eval): B200 / L40S = 0.274 at 294M -> 0.33 h of training at 53M + 0.1 h
  (dry inputs, offline LM, checks) = 0.43 h; **planned 0.6 B200-h** (a 53M step is launch-bound on a B200; the first finished
  job calibrates).
- [A4: every batch on B200: 240 runs = 144 B200-h; A4.6] Stage 1: 4 x (12 + 48) = **240 runs = 156 L40S-h (A, B3) + 72 B200-h (B1, B2)**.
- Stage 2 (+ 2b), per batch with m advancing domains, flat lr response: 8 m (confirm + the always-run prior extension) +
  2 m (the prior extension wins, p 1/2) + none (4 per distinct pick, 1-3; the 1e-3 extension when some argmax is 2e-3; its
  confirm when 1e-3 wins). min / expected / max runs: 30% advancing 74 / 110 / 147 (L40S-h 48 / 71 / 96, B200-h 22 / 33 /
  44); 50%: 112 / 172 / 224 (73 / 112 / 146; 34 / 52 / 67); 70%: 150 / 229 / 282 (98 / 149 / 183; 45 / 69 / 84). Given
  A3.6 (order comparisons screen out), 30-50% is the likelier range.
- [A4: 4 slots per batch, 16 chains in all, A4.6] Wall clock: B200 8 singleton slots per batch (B1 slots 0-7, B2 8-15) -> 60 runs in ~8 waves x ~0.6 h ~ 5 h per batch
  once running; `sbatch --test-only` (2026-11-04 10:47 UTC): B200 jobs start immediately, L40S jobs at ~19:15 UTC (an
  ~8.5 h queue).

### A3.8 Jobs, code, files
- Generator llmadd_torch/make_jobs_ptdom.py: `--batch {A,B1,B2,B3} --stage {1,2,2b} [--picks f] [--rerun] [--print]
  [--test-only N] [--submit]`, `--plan`, `--snapshot [--resnapshot]`. Job / run / out-dir name ptdom-<X>-<arm>-lr<lr>-s<s>
  [-r<k>]. L40S jobs: the A2 header (625, l40s_public, gpu48, nice 20, 3 h, requeue, ckpt every 2000 steps). B200 jobs:
  pt300x's 294M header line for line (account 520 priority, --constraint b200, preemption=yes, --dependency=singleton on
  llmadd_ptdom-b200-slot<k>, nice 20, 16 CPUs, 128G, 3 h, requeue, open-mode append, log llmadd_<run>_<id>.out), ckpt every
  500 steps (a preempted job re-runs its input checks and resumes). Every job: HOLD, the 12 snapshot shas, the batch's
  GPU-type guard, data sha256, --dry_inputs with the input checks and the DRY row hash of the batch and seed (exit 3),
  training, offline_lm, CHECK_PY with the cell's intent and $ARGS (exit 4). CPU references: ptdom_refs.json (A, the A2
  file) and ptdom_refs_<B1|B2|B3>.json (tests/a3/refs_a3.py, seeds 0-7, arm-independent at seed 0).
- CHECK_PY generalised per batch: the comparisons' 2-way held / seen / hard (hard only when the subset is not empty), the
  k-step domains' k-way held / seen, every non-months table sha, the batch's domain set exactly, the GPU type, ckpt_every.
  Fixed during the A3 build (the smoke caught it): the none arm's "no prior key" check tested the prefix 'geo_', which
  every geo_periods read-out key has; it now tests the keys geo_<d> exactly (would have invalidated all 12 B3 none runs).
- runs/ptdom/select_ptdom.py `--batch X --after 1|2` (picks_<X>_after1.json / _after2.json, written once, read-only);
  readout_ptdom.py (all batches; `--screen X`; `--selftest`); aux_gate_check.py `[--batch X]`; submit_ptdom.sh `--batch X
  --stage 1 [--go]` (prints the later commands). runs/ptdom/code = git archive of 1208bad (re-snapshotted while nothing
  was submitted). 240 stage-1 job files in runs/ptdom; nothing submitted.
- Re-run dead end (audit_a2 minor 7): if training finished but offline_lm / CHECK_PY did not (no check.json), the run is
  data and is not re-run: run the job's steps 3-4 by hand on the same OUT (`offline_lm.py ... "$OUT/$TAG"`, then the CHECK_PY
  heredoc with the job's intent and ARGS), then select.

### A3.9 Items of the A2 review (tests/audit_a2.md) under A3
- Important 1 (the cost simulation's noise too small) concerned A2's X / STOP rules, which A3 drops; A3's stage-2 estimate
  assumes a flat response (worst case for extensions), not a noise level.
- Minor 4 (the watcher called a finished prior run without on-window evals "before the window"): fixed; a run past step
  1000 without an on-window eval now fails its cell (tests/a3/aux_check_a3.out).
- Minor 7 (re-run dead end): documented in A3.8.
- Minor 2 (the gate at the pick pools seeds 0-7, so prior-dominated confirm seeds can hide behind seeds 0-3), minor 3
  (median of medians), minor 5 (watcher exit 1 -> HOLD is stricter than select), minor 6 (bracket across excluded cells: A3
  has no bracket): unchanged by design; minor 2 and 5 are open questions for the user (A3.10).

### A3.10 Open questions (for the user; nothing here changes A3 without a decision) [A4: 2 (none's upper edge: 1.6e-2), 3 (seeds 4-7 with the extension), 4 (every batch on B200) and 5 (the gate per half; the watcher names HOLD_<X>) decided by A4; 1 stays open (A4.5: NOT TESTABLE is recorded, not blocking)]
1. Order comparisons (country_size, animal_size, people, kinship, ranks, life_stages; partly prefixes / scalar_adj / units)
   are answered by a training-frequency shortcut at 85-100%: under (iii) they essentially cannot advance. Accept (screened
   out as not separable), or replace their read-out / the domains?
2. none's upper edge (8e-3) gets no extension (the user's list). A none argmax at 8e-3 leaves none possibly better above
   it, a bias toward a false PASS; add 1.6e-2 for that case?
3. Two prior lrs -> the prior extension always runs; when it wins, stage 2b adds a round (~one more queue wait). Option: run
   seeds 4-7 at the extension speculatively in stage 2 (+4 runs per advancing domain, no 2b round).
4. B200 concurrency: 8 singleton slots per batch (16 in all; pt300x used 8). L40S queue ~8.5 h at the test: move A / B3
   to B200 as well (one GPU type per batch would still hold)?
5. audit_a2 minor 2 / 5: gate seeds 4-7 separately at the pick; watcher exit 1 -> report first, HOLD by the user?

### A3.11 Tests (logs in runs/ptdom/tests/a3/)
| test | script / log | result |
|---|---|---|
| question builders, splits, leakage, answers, sites (18 new domains, seeds 0-7) | test_domains_a3.py / test_domains_a3.out | A3 DOMAINS PASS 542/542: split_seed / n_train / n_held / n_held_keys = the trainer's at every seed (comparisons ft36.split; k-step a third make_split), the seed-0 held lists = the fixture run's; leakage by key (comparisons: no held pair either order / anchor, both orders held; cyclic: both surface forms held, no identity class, shifts trained; linear: x and k trained; no held prompt trained); a different split at seed 1; k-step answers = items[(x + k) (mod n)], prompts = the atlas template and = manifold_domains._offset_questions; every comparison answer re-derived from the atlas coordinates by an independent rule; the trainer's heuristic lines and hard sizes = the read-out's re-derivation at seeds 0-7; site decoding on the 53M dry logs (3 / 2 / 1 sites per anchor / no-anchor / k-step line, every site ends an item name), 3 training rows and >= 3 eval rows per domain; batch A anchors reproduce A1.3 (30 / 16 / 2 / 0 / 5) and A1.4 (colours 86.0 / 84.3, hard 71.0 / 69.4); the read-out's run_subsets on the smoke's real run JSONs of all four batches (34 arm x comparison domains): recomputed full / hard 2-way series = the logged ones at every eval, areas = the metrics, masks / heuristics / table sha = the re-derivation, the shortcut baseline equal in both arms |
| trainer regression | reg_a3.sh, reg_compare_a3.py / reg_a3.out | A3 REGRESSION PASS: default, rb6all6, rb6none vs 93e22bf4c4c0; Astars, Amonths, Anone vs bb34927f16d8: IDENTICAL |
| CPU references | refs_a3.py, run_refs_a3.sh / run_refs_a3_<X>.out, ptdom_refs_<X>.json | TABLE PASS for B1 / B2 / B3 (seeds 0-7; seed 0 arm-independent: none@4e-3 = only-<first>@1e-3); A: Rev F with the A3 ARGS = ptdom_refs.json at seeds 0 and 4 (DRY rows, splits, tables, step-0 lm, data_sha) |
| 200-step CPU smoke per batch | smoke_a3.sh, smoke_check_a3.py / smoke_a3.out, smoke_check_a3.out | SMOKE A3 PASS 60/60 (after the CHECK_PY / smoke 'geo_' fix; the first pass failed exactly that check on B3 none): A stars, B1 animals, B2 compass, B3 kinship (bs 8) and B2 zodiac (bs 24: one site per k-step line): the prior loss on the fixed eval prompts falls over the window and sits below none's; lambda 1; applied on >= 95% of the on-steps; no other prior; none: no prior term ever; same data; read-outs logged |
| per-run CHECK known answers | check_py_a3.py / check_py_a3.out | CHECKPY A3 PASS 18/18: on the smoke JSONs only the 2 GPU-job checks fail (+ the tiny CKA bar for only-stars); the batch's GPU name passes, the other type fails; tampers fail their check (a missing k-way read-out, n_hard 3 without a hard read-out, a wrong table sha, an extra domain, ckpt_every); ranks' empty hard subset is not required |
| read-out / selection self-test (known answers) | readout_ptdom.py --selftest / selftest_readout_a3.out | SELFTEST PASS 32/32: screen (4/4 vs 3/4, the gate at the pick, the shortcut incl. an empty hard subset, GATE + HOLD semantics), the extensions (prior lower / upper, none lower, none upper / inside: none), the re-selection (moved pick, bpb tie-break keeps the old pick, EDGE), stage 2b, Holm over 7 domains of two batches and over 9 (raw p .018 -> Holm .145 fails), the F98 case, halves, NOT SEPARABLE at 8 seeds after a met screen, the gate failing at the pick over 8 seeds (median 0.475), audit_a1's CX1 / CX2 / CX3 / CX7, loader / integrity per batch |
| select -> generator plumbing | plumbing_a3.py / plumbing_a3.out | PLUMBING A3 PASS 10/10: B2 --after 1 -> the picks file (read-only) -> stage 2 renders E x seeds 0-3 + C x seeds 4-7 (44); overwrite refused; --after 2 -> compass moved -> stage 2b renders its 4 runs; the read-out agrees with both files; B1 with every only-units cell at 0.3 -> exit 2, (temp) HOLD, units GATE; a missing cell -> refused |
| early watcher | aux_check_a3.py / aux_check_a3.out | AUX CHECK A3 PASS 8/8 (per batch / cell; -r1 supersedes; pre-window -> exit 2; missing grad_cos -> exit 1; past step 1000 without an on-window eval -> exit 1; truncated JSON skipped) |
| job files | jobcheck_a3.py / jobcheck_a3.out | JOBCHECK A3 PASS 17/17: 240 stage-1 files (60 per batch): bash -n, intent, CHECK_PY byte-identical, ARGS + "$ARGS", ckpt, the batch's header and GPU guard, B200 slots 0-7 / 8-15; no stale file; B200 header = pt300x's 294M header line for line; the input-check block on the CPU dry logs: own 17/17 exit 0, another seed 13/13, another arm 8/8, another batch 30/30 exit 3 |
| sbatch --test-only | make_jobs_ptdom.py --test-only 1 per batch / sbatch_test_only_a3.out | 4/4 accepted (A / B3 l40s_public; B1 / B2 b200_tandon_prem, start immediately); nothing submitted (no SUBMITTED_ flag, no ptdom job in the queue) |
| submission script (print) | submit_ptdom.sh --batch B2 --stage 1 / submit_print_a3.out | the plan, the snapshot, self-test 32/32, the 60 jobs, the later commands; submitted: no |

Not done in A3: a fresh adversarial review of the A3 build (experiment-audit item v), for the coordinator to launch before
(or, in the gate's minimal mode, right after) the first submission.

## Amendment A4 (coordinator, before any stage-1 submission)

2026-11-04, written before any stage-1 run. Nothing of the screen was submitted: no SUBMITTED_ flag in runs/ptdom, no out dir;
the only jobs that ran are the four short GPU pilots of A4.7 (runs/ptdom/pilot/, not part of any result). Source: the fresh
adversarial review of A3 (tests/audit_a3.md: blocking 1, important 2-4, minor 5-14) and A3.10's open questions. The
coordinator decided items 1-7 below; they are implemented exactly (where an instruction needed an interpretation, it is
named) and are part of the pre-registration. **Where A3 (or anything above) disagrees, A4 holds; the superseded lines carry
an [A4] marker.** Code: trainer **Rev G** (branch ptdom, commit 11f61b9, pretrain_range.py sha256[:12] d87a1502cee8, not
pushed; evaluation only), runs/ptdom/code re-snapshotted (the A3 snapshot is superseded_A3/code_revF); make_jobs_ptdom.py,
readout_ptdom.py, select_ptdom.py, aux_gate_check.py, submit_ptdom.sh changed (their A3 versions in superseded_A3/scripts/;
A3's 240 job files in superseded_A3/jobs/). Tests and logs: runs/ptdom/tests/a4/.

### A4.1 The k-way read-out on the full item span; the tie-aware known answer (decision 1; audit_a3 blocking 1, minor 5)
- **Primary k-way metric = the full-span choice** `<d>_readout_kspan` (eval sets `<d>_<set>_kspan`, every k-way domain:
  months in batch A, zodiac / planets / elements in B2, chinese_zodiac / geo_periods / greek in B3; held, heldE (months),
  seen and months' held-out formats). Per eval row, every item c's summed teacher-forced log-prob of its span after the same
  prompt and context; hit iff the answer's is strictly larger than every other item's (a tie is a miss, counted in
  `tiesk_<set>`); per-question hits `hitsks_<set>`, mean margin `lpkmargin_<set>`. The span is ' <item>' **plus the closing
  newline**, exactly as ft36 (and so F81) scores its candidates (' <answer>\n'; the newline also keeps an item from winning as
  another item's prefix: prefix-free asserted) and as every eval row's scored span already is; the decision text reads
  "' <item>' after the prompt, as ft36 / F81 did": the ft36 form is implemented. Months: ' <month>' + '\n'.
- Candidate rows keep the 2-way's convention: the same text END and numbers line, the text start moved so that every row has
  512 tokens. Known answers / double paths (trainer asserts at build time; CHECK_PY at every eval, < 0.25 nats on GPU [A5:
  superseded by A5.1's tolerances, failing them alone = flagged data]; CPU
  asserts < 2e-2): the answer's own candidate row IS the eval row (token for token) and is scored again in the candidate pass
  (`lpk_own_maxdiff`, against the eval forward's span log-prob in another logits window); the direct per-position path on
  the first 4 candidate rows (`lpk_dp_maxdiff`); every candidate's span = the item's span in the training lines (asserted on
  every eval line); log-probs <= 0.
- The **first-token k-way choice** (A3's `<d>_readout_kway`) stays logged and is printed as the secondary line with every
  k-step verdict (`gain_first_token_kway`).
- **Tie-aware known answer**: `kway_em_bad_<set>` counts an exact-match hit whose first-token k-way choice is another item
  only when the two logits differ by more than one bf16 step at their magnitude (2^-7 x max |logit|); within it the case is
  a bf16 tie, logged as `kway_em_tie_<set>`. With an exact-match hit the answer token is the vocabulary argmax, so a
  violation can only be an index error. CHECK_PY: 0 violations, and the field present at every eval [A4: A3's "<= 5 over the
  run" is superseded]. Known answers on the trainer's own source lines: tests/a4/kway_tie_test.py (planted bf16 logits: 2
  ties, 2 violations, A3's count 4).
- **GPU evidence (A4.7)**: in 9 evals the pilots logged 25 / 0 / 14 / 35 such ties (A / B1 / B2 / B3) and 0 violations: A3's
  rule would have invalidated three of the four pilots. The audit's blocking item is confirmed and fixed.
- New descriptive line `heuristic_shortest_kspan` (domains.<d>): the summed log-prob favours fewer tokens (an untrained model
  scores ~ -ln V per token), the k-way analog of the 2-way's shorter-option line: the expected accuracy of choosing the
  item(s) with the fewest span tokens; it enters FLOOR (A4.5). Single-token items (span = 2 tokens): months 12/12, zodiac
  3/12, planets 7/8, elements 23/36, chinese_zodiac 11/12, geo_periods 1/12, greek 21/24.
- **Tied-embedding caveat (stated, no rule):** the model ties input and output embeddings. For a single-token item the
  token that carries the prior's site (the last token of the item name) is also the item's whole candidate token in the
  read-out (k-way: the answer position's logit row; 2-way: the option's first token). The prior moves those embedding rows,
  i.e. it can move the read-out's output rows directly, not only the computation that reads them; for a multi-token item the
  site token and the candidate's first token differ. A PASS on a domain of mostly single-token items (months, planets,
  chinese_zodiac, greek, elements) may partly reflect this direct route; it is printed under every verdict table.

### A4.2 The lr confound (decision 2; audit_a3 important 2; A3.10 open 2) [A5: plus fixB, A5.2]
- **none's extensions at both edges** [A4: supersedes A3.4's "none's upper edge: no extension" and "no second extension"]:
  stage-1 area argmax at 2e-3 -> 1e-3 (stage 2), and **5e-4 if 1e-3 is still none's argmax** after stage 2 (stage 2b);
  argmax at 8e-3 -> 1.6e-2 (stage 2), no further step. The prior's extensions are A3's (1e-3 -> 5e-4, 2e-3 -> 4e-3; one
  step; EDGE tag when its final argmax is the extension, the conservative direction).
- **NOT FINAL**: a none unit whose final pick OR area argmax sits at an outermost lr it ran (5e-4 after both lower steps,
  1.6e-2) -> no verdict ("NOT FINAL ... lr-confounded"); it stays in the Holm families with its p (conservative).
  Interpretation: the decision says "none pick still at an edge"; the extensions are triggered by the argmax, and a pick that
  the 2-point bpb tie-break moved inside while the argmax sits at the edge leaves the same open bracket, so both count. The
  literal 'pick only' reading is simulated beside it (A4.3: it lowers P(NOT FINAL) by ~0.05).
- **Same-lr condition (iv) of the screen**: the paired gain only-d@2e-3 - none@2e-3 (both arms run 2e-3 at seeds 0-3) > 0 in
  4/4 seeds, or the domain does not advance. The runs are data whatever the gate says about only-d@2e-3 (its gate state is
  printed). Every stage-2 verdict line prints it, with the same-lr gains at every lr both arms ran (and their seeds).
- **Rounds** (A4.5 'seeds 4-7 with the extensions'): stage 2 = C (seeds 4-7 at the stage-1 picks) + E (the first-step
  extensions at seeds 0-7); after 2 the units are re-selected over grid + their extensions (seeds 0-3); stage 2b = none's
  5e-4 at seeds 0-7 plus seeds 4-7 at any pick that moved to an lr without them (a top-two bpb tie towards another grid lr;
  a shared none cell often already has them); stage 2c = the same after 2b (no extension; never needed in the simulator).
  select_ptdom.py --after 1 | 2 | 3 | 4; make_jobs_ptdom.py --stage 1 | 2 | 2b | 2c; the chain is complete when a decision
  asks for nothing. Units of other domains never add an extension to their grid (A2's rule).

### A4.3 Calibration of the stage-2 rule (decision 3; audit_a3 important 4) [A5: the shared-trend rows are superseded by A5.2's simulation with fixB]
- **Noise level.** No ptdom run exists. The rb6 53M runs (Burst results rb6mb-* / rb6pb-*: the same trainer family, data and
  eval protocol, 2%, K 2, 75 evals; tests/a4/a4_noise.py -> a4_noise.out): per-run sd over seeds of the area, numbers line
  median 4.5 (17 cells); the domains' exact-match read-outs median 10.2 (IQR 6.1-15.1; 110 cell x read-out values); the
  paired between-arm sd / sqrt 2 ~ the within-cell sd (all 8.8, domains 9.4): the seed effect is not shared between arms, so
  the simulation draws independent per-run noise. ptdom's choice read-outs (2-way over ~1,200 held-out questions, k-way over
  every item) should be smoother than exact match; simulated at sigma 3 and 5 (the decision's range) and 10 (the rb6 domain
  level).
- **Simulator** tests/a4/a4_null_mc.py (a4_null_mc.sh, cpu_short; log a4_null_mc.out): the REAL readout_ptdom screen,
  chain_step, rule, sf_test, t_test (readout_rules.best_lr inside) on synthetic in-memory runs of one six-domain batch; the
  full A4 pipeline (stage 1, the screen with (iv), stage 2 / 2b / 2c, the verdict, NOT FINAL); the aux gate and the shortcut
  condition taken as passed (upper bounds on PASS); per domain, nominal (Holm family of 1).
- **The rule chosen (pre-registered)** [A4: adds to A3.5]: PASS iff A3's (a) 8-seed bootstrap 90% CI > 0, (b) both halves
  positive, (c) the 8-seed two-sided t p, Holm over every stage-2 domain of all batches, < 0.05, mean > 0, **and (d) the
  selection-free condition: seeds 4-7 (used by no selection or screen step) all positive and their one-sided one-sample t p
  (df 3), Holm over every stage-2 domain of all batches, < 0.10** (SF_ALPHA). An exact sign-flip test on 4 seeds has a
  minimum p of 1/16 = 0.0625, which Holm over two or more domains can never bring below 0.10, so the t form is used.

| scenario (nominal per domain) | P(advance) | P(PASS) | P(PASS \| advanced): A4 rule, SF 0.10 | SF 0.05 | A3 rule | P(NOT FINAL \| adv.): pick or argmax (pick only) |
|---|---|---|---|---|---|---|
| null, flat, sigma 3 | 0.012 | 0.0004 | **0.035** | 0.028 | 0.110 | 0.27 (0.20) |
| null, flat, sigma 5 | 0.011 | 0.0004 | **0.036** | 0.030 | 0.109 | 0.26 (0.21) |
| null, flat, sigma 10 | 0.010 | 0.0003 | **0.035** | 0.027 | 0.108 | 0.24 (0.21) |
| null, flat, sigma 5, t3 tails | 0.011 | 0.0004 | 0.034 | 0.022 | 0.101 | 0.29 (0.21) |
| null + shared lr trend +0.5 / halving, sigma 3 / 5 | 0.017 / 0.015 | 0.0010 / 0.0007 | 0.059 / 0.050 | 0.045 / 0.039 | 0.17 / 0.14 | 0.22 / 0.24 |
| trend +1, sigma 3 / 5 | 0.025 / 0.019 | 0.0022 / 0.0012 | 0.088 / 0.060 | 0.076 / 0.047 | 0.23 / 0.17 | 0.27 / 0.23 |
| trend +2, sigma 3 / 5 | 0.037 / 0.027 | 0.0057 / 0.0028 | 0.151 / 0.105 | 0.134 / 0.082 | 0.27 / 0.24 | 0.54 / 0.34 |
| trend +3, sigma 3 / 5 | 0.045 / 0.036 | 0.0070 / 0.0052 | 0.154 / 0.147 | 0.144 / 0.127 | 0.21 / 0.26 | 0.74 / 0.48 |
| trend -1 (higher lr better: rb6's direction), sigma 3 / 5 | 0.003 / 0.005 | 0.0001 / 0.0001 | 0.015 / 0.031 | 0.010 / 0.024 | 0.05 / 0.07 | 0.53 / 0.40 |

| power: a constant prior effect delta at every prior lr, flat lr response | P(advance) | P(PASS), A4 rule SF 0.10 | P(PASS \| advanced) | A3 rule P(PASS) | P(NOT FINAL \| adv.) |
|---|---|---|---|---|---|
| sigma 3: delta +3 / +6 / +10 | 0.16 / 0.56 / 0.93 | **0.038 / 0.33 / 0.77** | 0.25 / 0.60 / 0.82 | 0.08 / 0.45 / 0.80 | 0.20 / 0.17 / 0.15 |
| sigma 5: delta +3 / +6 / +10 | 0.06 / 0.22 / 0.56 | **0.008 / 0.070 / 0.34** | 0.14 / 0.33 / 0.60 | 0.02 / 0.13 / 0.45 | 0.21 / 0.19 / 0.16 |
| sigma 10: delta +3 / +6 / +10 | 0.03 / 0.06 / 0.15 | **0.002 / 0.006 / 0.038** | 0.08 / 0.11 / 0.26 | 0.006 / 0.018 / 0.079 | 0.21 / 0.22 / 0.19 |

- **Reading.** (1) Under no prior effect and a flat lr response the A4 rule's false-PASS rate among advancing domains is
  0.035 (sigma 3-10, also with heavy tails), within the decision's <= 0.05 per domain; A3's rule had 0.11 (the auditor's
  0.08-0.20 with A3's screen). (2) The new same-lr condition cuts P(advance | no effect) from A3's 0.036 to 0.010-0.012. (3)
  Under a SHARED lr trend towards low lr (both arms better at lower lr) P(PASS | advanced) exceeds 0.05: 0.05-0.09 at
  +0.5-1 point per halving, 0.10-0.15 at +2-3 (unconditional P(PASS) <= 0.007 per domain), while NOT FINAL takes 0.22-0.74
  of the advancing domains; SF 0.05 would hold 0.05 up to +0.5 per halving at about 10% relative power. The trend in the
  conservative direction (rb6: none better at higher lr) gives 0.015-0.031. SF_ALPHA 0.10 is kept (the decision's minimum,
  the nominal criterion met); 0.05 is an option for the user. (4) Power is modest: at sigma 5 a +6-point effect passes with
  P 0.07 and +10 with 0.34 (Holm over m domains lowers it further); about one advancing domain in five ends NOT FINAL even
  under a flat response (the none extensions reach an outermost lr by noise), 0.04-0.06 less under the literal 'pick' reading.
- [A4: A3.3's last bullet ("0.036 ... <= 0.0037 ... about 0.09 false PASS") and tests/a3/a3_null_mc.out are superseded:
  run noise 1 point, rho without effect; audit_a3 important 4.]

### A4.4 The read-out refuses a final verdict while anything is missing (decision 4; audit_a3 important 3, minor 11)
- readout_ptdom.readout re-derives every batch's decision chain (screen, after 2, after 3, ...). A **final** verdict needs
  all four batches read (the Holm family's batches), every chain complete (the last decision asks for nothing and every asked
  cell has a valid run), and every decision's picks file present and equal to the re-derivation (a differing file is no
  longer only printed as DIFFER). Otherwise the header prints PROVISIONAL with every reason, and every verdict reads
  "PROVISIONAL (<what it would be> if final ...)" — no PASS; the Holm families are printed with what is missing, never
  shrunk silently. A test may name its own family (readout(..., family=...)), printed as "A TEST FAMILY, not the design's";
  main() always uses the four batches.

### A4.5 Minor items (decision 5)
- **Per-batch HOLD** (audit_a3 minor 8): select_ptdom.py --after 1 writes runs/ptdom/HOLD_<X> (default) when an arm fails the
  gate in every stage-1 cell; every job checks runs/ptdom/HOLD (all batches) and HOLD_<its batch>; make_jobs_ptdom.py
  --submit refuses while either exists; aux_gate_check.py names HOLD_<X> of the failing cell's batch. [A4: supersedes A2.1 /
  A3.3's single HOLD.]
- **FLOOR** (minor 9): both arms' areas <= max(chance, answer-frequency line, shorter-option line, the training-frequency
  shortcut baseline on every held-out question (comparisons), the elimination baseline and the shortest-span line (k-step))
  + 5. Months has no elimination line (the rb6 path); its shortest-span line is chance (every span 2 tokens).
- **Shortcut condition (iii)** (minor 7) [A4: supersedes A3.3 (iii) / A3.5's per-seed mean and the "every seed empty ->
  fails" rule]: decided on the POOLED hard-subset questions of the seeds (seeds 0-3 at the screen, 0-7 at the verdict), both
  accuracies weighted by n_hard; **testable iff >= 30 pooled questions** (SC_MIN); else 'NOT TESTABLE': recorded and
  reported (screen line; verdict tag SHORTCUT NOT TESTABLE), it never blocks advancing and never turns a PASS into NOT
  SEPARABLE. From A3.1's hard sizes, likely not testable at the screen: country_size (~1 pooled), ranks (~10), life_stages
  (~12); prefixes borderline (~35).
- **No saved model in stage 1** (minor 12: /scratch 95.5%): stage-1 ARGS have no --save_model and the job has no offline LM
  step; CHECK_PY asserts no model file and no offline file ("stage 1 (A4): no saved model ..."). The bpb tie-break of F87
  rule 1 then uses the in-training val bpb for any pair involving a stage-1 cell (readout_rules.best_lr: offline only when
  every run of both has it), and the LM side-effect rows of stage-1 cells are in-training. Stage 2 / 2b / 2c jobs save the
  model and run offline_lm as before.
- **Seeds 4-7 at the extension lrs run with stage 2** (A3.10 open 3): every E cell at seeds 0-7 (A4.2).
- **The gate at the pick per half** (audit_a2 minor 2, A3.10 open 5): the A2.1 cell gate at the picked prior cell on seeds
  0-3 and on seeds 4-7 separately; either failing -> "AUX GATE FAILED at the pick" (the 8-seed median is printed beside).
- Unchanged by A4 (no decision): audit_a3 minor 10 (the preemption window between the checkpoint unlink and the model save;
  stage 1 saves no model now), minor 14 (batch A references re-verified at seed 0 under Rev G, A4.8); minor 13's GEOK test is
  now recorded (tests/a4/check_py_a4.py); minor 6's GPU-only tolerances ran on GPU (A4.7).

### A4.6 Hardware and cost (decision 6)
- **Every batch on B200** [A4: supersedes A3.1 / A3.7 / A3.8's L40S for A and B3]: one GPU type per batch still holds (the
  job's guard, CHECK_PY's GPU name). pt300x's 294M protocol (account 520 priority, --constraint b200, preemption=yes,
  --requeue, --open-mode append, ckpt every 500 steps, resume after preemption, 3 h). **16 singleton slot chains: 4 per
  batch** (A 0-3, B1 4-7, B2 8-11, B3 12-15) [A4: supersedes B1 0-7 / B2 8-15].
- Pilot timing (A4.7): 50 steps + one eval take 5.5 (B1) to 11 s (A, whose months k-span adds ~33k candidate rows per
  eval); the pilots' loops (400 steps, 9 evals) 50-98 s. The split between step and eval time is not identifiable from a
  pilot (every interval has the same mix): a full run lies between 75 evals alone and 366 such intervals, 0.2-1.1 h on B200;
  the plan keeps 0.6 B200-h per run, the 3 h limit holds either way, and the first finished stage-1 job calibrates.
- Cost (make_jobs_ptdom.py --plan): stage 1 = 240 runs = 144 B200-h; stage 2 / 2b with a flat lr response at 30 / 50 / 70%
  advancing: 153 / 236 / 310 runs expected (92 / 142 / 186 B200-h); A4.3's P(advance) suggests the low end.

### A4.7 GPU pilots (decision 7; the GPU-only tolerance checks had never run on GPU: audit_a3 minor 6)
One short job per batch (`make_jobs_ptdom.py --batch X --pilot`; the batch's real data, domains, eval sets and a prior arm;
400 steps via --tokens 6553600, evals every 50 steps (9; 3 on the window, steps 20..159), ckpt every 100, --save_model so
offline_lm runs; output runs/ptdom/pilot/, job names llmadd_ptdom-pilot-<X>, B200, 1 h; the same job text as stage 2 with the
pilot's intent). All four ran once and passed every in-job check (CHECKS PASS; nothing had to be fixed or re-run); the
confirmations runs/ptdom/pilot/check_ok_<X> are required by submit_ptdom.sh --go. Not part of any result.

| batch | job | cell | checks | DRY rows = ref | step-0 lm - ref | 2-way double path | k-span double paths | k-way ties / violations | decomposition max reldiff | CKA rise | LM share a / (a + b), window median | offline LM |orig - logged| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 21848184 | only-stars@1e-3 s0 | 35 / 35 pass | 322bf3c5911b, 84 | +6.2e-5 | 0.080 | 0.084 | 25 / 0 | 1.7e-3 | 0.88 | 0.42 | 3.5e-8 |
| B1 | 21848188 | only-animals@1e-3 s0 | 34 / 34 (no k-way domain) | ac60a4fe9755, 79 | +8.2e-5 | 0.065 | - | 0 / 0 | 5.1e-5 | 0.97 | 0.86 | 3.5e-8 |
| B2 | 21848189 | only-elements@1e-3 s0 | 35 / 35 | fef991315336, 73 | +5.3e-5 | 0.066 | 0.058 | 14 / 0 | 4.1e-4 | 0.88 | 0.32 | 3.5e-8 |
| B3 | 21848212 | only-chinese_zodiac@1e-3 s0 | 35 / 35 | 8dcfbef71e9d, 73 | -3.1e-5 | 0.062 | 0.059 | 35 / 0 | 1.6e-4 | 0.91 | 0.57 | 3.5e-8 |

Also every pilot: data_sha after step 0 = the CPU reference, lambda 1 (gnorm_prior / gnorm_true 1.0 at the 3 on-evals), the
gradient-split fields at every eval (the aux gate's inputs), GPU = NVIDIA B200 in every sample; torch 2.10.0, transformers
4.57.6. **Note for the user (not a gate result):** on the pilots' compressed window (steps 20-159 of 400, a 400-step lr
schedule) the prior's gradient exceeded the LM's for stars (a / (a + b) 0.42) and elements (0.32); the 53M window is steps
916-7323 of 18,310, so this does not predict the stage-1 gate, but the A2.1 gate (< 0.5 fails the cell) may bite there.

### A4.8 Tests (runs/ptdom/tests/a4/; the A3 tests they replace stay in tests/a3/ as records of the A3 code)
| test | script / log | result |
|---|---|---|
| read-out / selection self-test (A4 known answers) | readout_ptdom.py --selftest / selftest_readout_a4.out | SELFTEST PASS 38/38: Holm; extension / next_ext / not_final known answers (none 8e-3 -> 1.6e-2, 1e-3 -> 5e-4, NOT FINAL at 5e-4 / 1.6e-2 by pick or argmax); pooled shortcut (met / not met / NOT TESTABLE incl. every subset empty); sf_test = scipy's one-sided t; the rule: A3's cases, the A3 conditions with a negative confirm seed -> fails, noisy seeds 4-7 -> fails, audit_a3's cx_halves case passes alone and fails under Holm x 9; the planted two-batch chain (A + B2): the same-lr condition stops languages, cities advances with the shortcut NOT TESTABLE, stage 2 = E x seeds 0-7 + C x seeds 4-7, after 2 (stars' prior moves to 4e-3 with EDGE and no C; none 1e-3 -> 5e-4; cities' none pick moves to 8e-3 -> C in 2b; compass NOT FINAL at 1.6e-2; ranks' moved none pick needs no C, its seeds 4-7 shared), after 3 (zodiac NOT FINAL at 5e-4, complete), verdicts (PASS x 5 incl. stars at the moved pick, planets NOT SHOWN by the seeds 4-7 condition, prefixes NOT SEPARABLE on the pooled 8 seeds, cities PASS + SHORTCUT NOT TESTABLE), the read-out PROVISIONAL for missing batches, a missing / differing picks file and missing 2b cells; the gate per half (seeds 4-7 at 0.45 fail while the 8-seed median 0.675 passes); audit_a1's gate counter-examples; loader / integrity |
| pipeline simulation | a4_null_mc.py, a4_null_mc.sh / a4_null_mc.out; noise a4_noise.py / a4_noise.out | the A4.3 tables |
| trainer regression | reg_a4.sh, reg_compare_a4.py / reg_a4.out | A4 REGRESSION PASS: default, rb6 all6-lam1, rb6 none vs origin/main 93e22bf4c4c0 IDENTICAL; batch A (only-stars, only-months, none) and B2 (only-elements, none) vs Rev F 1208bad IDENTICAL UP TO REV G (every training curve field and the 36 model tensors bitwise; dry / run stdout equal after removing exactly the Rev G keys and lines; the additions present for every k-way domain: 95 eval keys in A, 43 in B2) |
| CPU references under Rev G | refs_check_a4.py, refs_a4.sh / refs_a4.out | A4 REFS PASS 8/8: per batch the stage-1 reference cell (seed 0, A4 stage-1 ARGS, no --save_model) and the pilot cell (pilot ARGS): DRY rows, splits, table shas = ptdom_refs[_X].json; domains line = intent; the Rev G k-span lines present (not in the DRY rows, so the references hold) |
| GPU pilots | runs/ptdom/pilot/, logs llmadd_ptdom-pilot-<X>_<id>.out | A4.7: 4 / 4 CHECKS PASS |
| per-run CHECK on real B200 run JSONs | check_py_a4.py / check_py_a4.out | CHECKPY A4 PASS 24/24: the four pilots' CHECK re-derived (exit 0); tampers fail exactly their check (kway_em_bad 1, k-span direct path 0.5, the own row 0.3, a missing <d>_held_kspan, no kway_em_bad key); the stage-1 intent on a saved-model run fails exactly 'stage 1 (A4): no saved model'; the GEOK key test on the smoke's B3 none run |
| tie-aware known answer | kway_tie_test.py / kway_tie_test.out | KWAY TIE TEST PASS (the trainer's own lines on planted bf16 logits: bad 2, tie 2; A3's count 4) |
| 200-step CPU smoke per batch | smoke_a4.sh, smoke_check_a4.py / smoke_a4.out | SMOKE A4 PASS 70/70: A3's 60 checks on the Rev G snapshot, plus per pair the full-span read-out of every k-way set at every eval in both arms (direct double path < 2e-2, the own row = the eval forward exactly, ties counted, the *_kspan metrics, hitsks per question) and the tie-aware known answer (0 violations, kway_em_tie logged; no k-span key on a non-k-way domain) |
| the read-out on real B200 JSONs | pilot_readout_check.py / pilot_readout_check.out | PILOT READOUT PASS 5/5: the loader reads every A4 field of the four pilots (k-span and first-token areas, the shortest-span line, the gradient split, the offline bpb); run_subsets re-derives every comparison domain's questions, masks, heuristics, table sha and full / hard 2-way series at every eval: 0 problems |
| question builders, splits, sub-read-outs | test_domains_a4.py (the A3 test on the Rev G snapshot and the A4 smoke JSONs) / test_domains_a4.out | A4 DOMAINS PASS 542/542 (A3's checks unchanged: splits, leakage by key, answers, sites, heuristics, run_subsets on the A4 smoke JSONs) |
| select -> generator plumbing | plumbing_a4.py / plumbing_a4.out | PLUMBING A4 PASS 16/16: B2 after 1 -> stage 2 (E x 8 seeds + C x 4) -> after 2 (E none 5e-4; compass moved, no C) -> stage 2b -> after 3 complete; after 4 refused; a tampered after-1 file refused; the read-out final on the test family, PROVISIONAL with the design's; B1's gate failure writes HOLD_B1 (per batch); with a real HOLD_B1 (and a fake sbatch on PATH) --submit of B1 is refused while B2 renders; a missing cell refused |
| early watcher | aux_check_a4.py / aux_check_a4.out | AUX CHECK A4 PASS 2/2 (A3's 8 checks on the A4 watcher; a failing B2 cell names HOLD_B2 only) |
| job files | jobcheck_a4.py / jobcheck_a4.out | JOBCHECK A4 PASS 20/20: 240 stage-1 files (bash -n, intent with save_model false, CHECK_PY byte-identical, ARGS without --save_model, no offline LM step, B200 header = pt300x's, HOLD + HOLD_<X>, slots 4 per batch = 16 chains, 15 runs each); the 4 pilot files; no stale file; the input-check block on the CPU dry logs (Rev G logs at the seed-0 reference cells): own 18/18 exit 0, another seed 14/14, another arm 10/10, another batch 36/36 exit 3 |
| sbatch --test-only / submission print | sbatch_test_only_a4.out / submit_print_a4.out | 4/4 accepted (none@2e-3 s0 of each batch: b200_tandon_prem, gb001, start immediately); submit_ptdom.sh --batch B2 --stage 1 (print): the plan, the snapshot 11f61b9, self-test 38/38, the 60 jobs, the later commands; submitted: no (no SUBMITTED_ flag in runs/ptdom; no ptdom job in the queue; --go now also requires runs/ptdom/pilot/check_ok_<X>) |

### A4.9 Not done, and open for the user
- [A5: done: tests/audit_a4.md; its items are handled by A5] Not done in A4: a fresh adversarial review of the A4 changes (experiment-audit item v; this build had no sub-agent): for
  the coordinator to launch before (or, in the gate's minimal mode, right after) the first stage-1 submission. Stage 1 was
  NOT submitted.
- For the user (nothing here changes A4 without a decision): (1) SF_ALPHA 0.10 (kept) vs 0.05 (A4.3: <= 0.05 also under a
  +0.5 / halving shared lr trend, ~10% relative power); (2) NOT FINAL by 'pick or argmax' (implemented) vs the literal 'pick'
  (P(NOT FINAL | adv.) ~0.20 instead of ~0.26 under a flat null); (3) power is low at sigma 5-10 (A4.3); (4) the span with
  ft36's closing newline (implemented) vs ' <item>' without it; (5) the pilots' early-window LM share below 0.5 for stars and
  elements (A4.7).

## Amendment A5 (coordinator, before any stage-1 submission; no ptdom result read)

2026-11-04, written before any stage-1 run. Nothing of the screen was submitted (no SUBMITTED_ flag in runs/ptdom, no out dir,
no ptdom job in the queue); the only ptdom jobs that ever ran are A4.7's four GPU pilots (not results). No ptdom result was
read: A5 uses only the pilots' double-path / span-token fields (the in-job checks' own inputs). Source: the fresh adversarial
review of A4 (tests/audit_a4.md: blocking 1, important 2, minors 3, 4, 6) and the coordinator's decisions 1-3, implemented
exactly; where an instruction needed an interpretation it is named. **Where A4 (or anything above) disagrees, A5 holds;**
the superseded lines carry an [A5] marker. The trainer is unchanged (Rev G, 11f61b9, d87a1502cee8; runs/ptdom/code not
touched). Changed: make_jobs_ptdom.py (CHECK_PY, the intent, cells() F, plan), readout_ptdom.py, select_ptdom.py,
submit_ptdom.sh; their A4 versions and A4's 240 job files are in runs/ptdom/superseded_A4/. Tests and logs: tests/a5/.

### A5.1 The double-path tolerances (decision 1; audit_a4 blocking 1)
- **Rule (= runs/ptheld/AUDIT.md Amendment DP, adopted for both the 2-way and the k-span double paths):** at eval 0 (step 0,
  a near-uniform model where an index error costs ~11 nats) every value <= 0.05 nats; at every later eval (before a
  divergence) <= 0.5 nat per scored token. Applies to lp_dp_maxdiff_<d>_<held|seen> (2-way) and lpk_dp_maxdiff_<set> /
  lpk_own_maxdiff_<set> (k-span, every k-way set incl. months' heldE and formats). A4's '< 0.25 at every eval' is superseded.
- **Scored tokens (interpretation):** the trainer logs only the maximum over a set's first 4 rows, not their span lengths, so
  the token count is bounded by the domain's LONGEST scored span ' <item>\n' (incl. the newline; make_jobs_ptdom.SPAN_TOK,
  tests/a5/span_tok.py: from the snapshot's tables and the jobs' tokenizer exactly as the trainer builds the spans, = every
  k-step domain's logged kspan_tokens item by item and every domain's item_tokens histogram in the four pilots, 55/55). An
  upper bound per row: a run meeting the per-row rule never fails; a short row may pass up to the long-row bound. Later-eval
  tolerances (nats): months 1.0, stars 2.5, landmarks 3.5, languages 2.0, colours 1.5, cities 2.5; countries 3.0, animals
  2.5, country_size 3.0, animal_size 2.5, people 4.0, units 1.5; zodiac 2.0, compass 1.0, planets 1.5, elements 2.5, prefixes
  1.5, ranks 1.5; chinese_zodiac 1.5, kinship 2.0, life_stages 1.0, geo_periods 2.0, greek 2.0, scalar_adj 2.0 (carried in
  every job's intent as dp_tol0 / dp_tol).
- **The fields are a separate check** ("double-path fields logged (finite) at every eval before a divergence", every k-way
  domain's held / seen): a missing or non-finite value invalidates the run (it is not a tolerance). A4 failed a missing field
  through inf, but a NaN could pass A4's max().
- **Pre-registered (before any stage-1 run): a run whose only failed checks are these two GPU-numerics tolerance checks
  (and / or A1's CKA-rise bar) is VALID data, FLAGGED, never dropped.** CHECK_PY writes check.json "valid_data_flagged" and
  prints "CHECKS FAIL (only GPU-numerics tolerance / CKA-rise checks failed: the run is VALID data, flagged; AUDIT.md A5.1)";
  the exit code stays 4 (the job shows FAILED in Slurm; the loader, not the exit code, decides validity). readout_ptdom.load
  keeps such a run (rec dp_flag); select_ptdom.py, readout_ptdom.py --screen and the read-out list every flagged run. The
  flag is descriptive (no rule reads it). Consequence, stated: these two double paths no longer invalidate a run by
  themselves; an index error in them would have to be GPU-only (the same code asserts < 2e-2 on CPU, the smoke ran it, the
  pilots logged 0.010-0.017 at eval 0) and would show as a flag on every run; the other in-job checks (the k-way known
  answer kway_em_bad == 0, the read-out fields, step-0 lm, data_sha, splits, tables, the decomposition double path, ...) still
  invalidate.
- **Validation (tests/a5/ptheld_dp_a5.py -> ptheld_dp_a5.out):**
  - the 184 finished 53M ptheld GPU runs the auditor used (75 evals; H_one / H_both / H_ans = 2 / 3 / 3 scored tokens ->
    1.0 / 1.5 / 1.5): after step 1250, 38,640 (eval, set) values: old rule (> 0.25) 1.75% of the values, 130 / 184 runs (70.7%)
    with an exceed; **new rule 0 values, 0 / 184 runs** (largest value / tolerance 0.93: H_ans 1.39 vs 1.5); eval 0: 552
    values, max 0.0195 <= 0.05;
  - the four ptdom pilots: 720 values (A 288, B1 108, B2 162, B3 162): eval-0 max 0.017 (<= 0.05), later largest value /
    tolerance 0.084, 0 over.
  - Caveat: ptheld scores 8 rows per set and 3 sets, ptdom 4 rows and 6-12 sets x 2-3 paths (450-900 values per check and
    run); with ptheld's largest ratio 0.93, an occasional late exceed on a 2-token domain (tolerance 1.0: months, compass,
    life_stages) is possible; such a run is flagged data.
- **Is the job-check change a pure tolerance loosening? No.** Later evals loosen (0.25 -> 1.0-4.0), but eval 0 TIGHTENS
  (< 0.25 -> <= 0.05, the decision's rule) and a NaN double-path value now fails (the fields check); the exit-code semantics
  are unchanged. So by the coordinator's criterion the pilot confirmations (runs/ptdom/pilot/check_ok_<X>, written by the A4
  CHECK_PY) do not carry over automatically. Evidence for the coordinator: the A5 CHECK_PY re-derived on the four pilots' real
  B200 run JSONs with the job's own inputs and the A5 pilot intent (= the submitted A4 intent + exactly dp_tol0 / dp_tol)
  passes every check (exit 0) in all four (tests/a5/check_py_a5.out). check_ok_<X> and the submitted pilot job files are
  untouched (A4 records); submit_ptdom.sh --go still requires check_ok_<X> (unchanged). Whether that re-derivation stands in
  for an A5 confirmation, or new pilots are run, is the coordinator's decision.

### A5.2 fixB: none at a prior's pick outside none's lr range (decision 2; audit_a4 important 2)
- **Rule (pre-registered; the auditor's simulated 'fixB'):** for an advancing domain d, once none's own extension chain for
  d is done (no next step) and the prior's final pick P is fixed (from after 2 on: the prior's one extension has run), if P
  lies outside the lr range none has run FOR d (none's stage-1 grid + d's own none extensions; other domains' extensions
  are not part of d's range, A2's rule), none is run at P at seeds 0-7 (same batch, same GPU type: the decision's list F,
  rendered as stage 2b / 2c), and once those runs exist none is re-selected for d by F87 rule 1 (seeds 0-3) over its range +
  P; that pick is none's final pick for d (seeds 4-7 at it, C, if it lacks them). The verdict rule is A4's (incl. the
  selection-free seeds 4-7 test, one-sided t, Holm, p < 0.10: unchanged).
- Interpretations (named): (a) a none cell at P already asked for another domain (e.g. none@1e-3 as another domain's
  extension) is the same cell: it is reused at once, no new runs; (b) NOT FINAL also when P is an outermost lr (5e-4) and
  none's re-selected pick or argmax is P (as simulated; A4's NOT FINAL stands too); (c) the fixB lr triggers no further none
  extension (as simulated); (d) with stage-1 grids none 2e-3..8e-3 and prior picks in {5e-4, 1e-3, 2e-3, 4e-3}, F is only ever
  none@1e-3 or none@5e-4 (make_jobs_ptdom.FIXB_LR, asserted).
- **No verdict until they exist:** while a domain's F cells are missing, the read-out prints "NO VERDICT (A5 fixB: none not
  yet run at the prior's pick P ...)" for it (not even a provisional PASS / NOT SHOWN) and lists it among the PROVISIONAL
  reasons; select_ptdom.py --after k lists the extra none jobs by name and the domains they are for. Picks files carry
  amendment "A5" and the list F (the generator refuses an A4 file, F in a stage-2 file, F at a prior arm or at an lr outside
  FIXB_LR, F duplicating E); the read-out's consistency check compares F.
- **Known answers:** the self-test (46/46): the planted A + B2 design (none@1e-3 reused at once for months / cities /
  compass / planets / prefixes, picks unchanged) and a planted B3 chain with NEW none cells (chinese_zodiac with no prior
  effect, a shared trend and none's lucky 4e-3: A4 would PASS it, A5 gives NO VERDICT, then NOT FINAL at none's 5e-4;
  kinship with +4: NO VERDICT, then PASS with the FIXB tag); plumbing_a5 runs the same through the CLI (select lists the 16
  jobs, the generator renders exactly them, a picks file without F is refused); cx_a5.py: the auditor's CX1 made
  deterministic -> A4 code PASS for zodiac and compass, A5 code NOT FINAL (none re-selected at 5e-4), CX2 unchanged.
- **Simulation (tests/a5/a5_null_mc.py + .sh -> a5_null_mc.out; the A4.3 model and scenario list on the A5 code, fixB inside
  the real chain_step, other RNG seeds; per domain, nominal Holm family of 1; SF 0.10 pre-registered, 0.05 for reference):**

| scenario | P(PASS \| adv): A4 (A4.3) | **A5 fixB, SF 0.10** (se) | A5, SF 0.05 | P(NOT FINAL \| adv) A5 | P(fixB \| adv) | fixB extra none runs / batch |
|---|---|---|---|---|---|---|
| null flat, sigma 3 / 5 / 10 | 0.035 / 0.036 / 0.035 | **0.041 / 0.040 / 0.034** (0.003) | 0.033 / 0.032 / 0.026 | 0.27 / 0.27 / 0.24 | 0.08-0.09 | 0.04-0.05 |
| null flat, sigma 5, t3 tails | 0.034 | **0.024** (0.004) | 0.015 | 0.26 | 0.10 | 0.05 |
| shared trend +0.5 / halving, sigma 3 / 5 | 0.059 / 0.050 | **0.039 / 0.036** (0.004) | 0.028 / 0.026 | 0.29 / 0.25 | 0.16 / 0.12 | 0.13 / 0.09 |
| trend +1, sigma 3 / 5 | 0.088 / 0.060 | **0.054 / 0.060** (0.004 / 0.005) | 0.041 / 0.049 | 0.37 / 0.27 | 0.23 / 0.15 | 0.25 / 0.13 |
| trend +2, sigma 3 / 5 | 0.151 / 0.105 | **0.029 / 0.041** (0.003) | 0.022 / 0.033 | 0.69 / 0.43 | 0.27 / 0.21 | 0.43 / 0.26 |
| trend +3, sigma 3 / 5 | 0.154 / 0.147 | **0.010 / 0.028** (0.001 / 0.003) | 0.008 / 0.020 | 0.89 / 0.63 | 0.20 / 0.26 | 0.36 / 0.40 |
| trend -1 (higher lr better), sigma 3 / 5 | 0.015 / 0.031 | **0.013 / 0.024** (0.006) | 0.011 / 0.019 | 0.50 / 0.42 | 0.02 / 0.05 | 0.00 / 0.01 |

| power (flat lr response, a constant effect delta) | P(PASS) A4 (A4.3) | **P(PASS) A5** | P(PASS \| adv) A5 | fixB extra none runs / batch |
|---|---|---|---|---|
| sigma 3: delta +3 / +6 / +10 | 0.038 / 0.33 / 0.77 | **0.037 / 0.315 / 0.737** | 0.25 / 0.57 / 0.79 | 1.4 / 3.6 / 3.8 |
| sigma 5: delta +3 / +6 / +10 | 0.008 / 0.070 / 0.34 | **0.009 / 0.068 / 0.318** | 0.15 / 0.31 / 0.57 | 0.5 / 1.9 / 3.5 |
| sigma 10: delta +3 / +6 / +10 | 0.002 / 0.006 / 0.038 | **0.002 / 0.008 / 0.038** | 0.08 / 0.13 / 0.25 | 0.2 / 0.5 / 1.3 |

- **More iterations near the target** (a5_null_mc_more.py / .sh -> a5_null_mc_more.out, other seeds; pooled with the table
  by tests/a5/pool_mc.py -> pool_mc.out, weights = advancing domains), P(PASS | adv) SF 0.10 (se) / SF 0.05:

| shared trend / halving | +0.5 | +0.75 | +1 | +1.25 | +1.5 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| sigma 3 | 0.045 (0.003) / 0.034 | 0.051 (0.003) / 0.039 | **0.052 (0.002)** / 0.041 | 0.044 (0.003) / 0.034 | 0.041 (0.002) / 0.033 | 0.029 / 0.022 | 0.010 / 0.008 |
| sigma 5 | 0.045 (0.003) / 0.033 | 0.050 (0.003) / 0.038 | **0.051 (0.002)** / 0.040 | 0.050 (0.003) / 0.039 | 0.046 (0.003) / 0.034 | 0.041 / 0.033 | 0.028 / 0.020 |

- **Reading.** (1) fixB removes the auditor's path: A4's shared-trend excess (0.088-0.154 at +1..+3) is gone; under a flat
  null 0.034-0.041 (t3 tails 0.024). (2) The worst case is a mild trend (+0.75..+1.25 / halving): 0.050-0.052 (se 0.002-0.003),
  i.e. at the 0.05 target within simulation error, not clearly below it. Likely source (not isolated by a test): none
  is re-selected on the same seeds 0-3 whose noise made its pick (a lucky inside cell can keep the pick over the fixB lr), and
  the 2-point bpb tie-break; at stronger trends the NOT FINAL rule absorbs it (NOT FINAL | adv 0.43-0.89). With SF 0.05 every
  cell is <= 0.041 (reference only; SF 0.10 kept as decided). Holm over several advancing domains lowers the family-wise
  rates further. (3) Power: unconditional P(PASS) falls by ~5% relative (sigma 3, delta +6: 0.332 -> 0.315; sigma 5, delta +10:
  0.337 -> 0.318), as the auditor's fixB estimate. (4) Cost: 0.04-0.43 extra none runs per batch under the nulls, 1.3-3.8 under
  real effects (stage-2+ 20-99 runs per batch there).

### A5.3 Minor items (decision 3; audit_a4 minors 3, 4, 6)
- **--allow_partial (minor 3):** readout() now receives the comparison sub-read-out problems (sprobs); non-empty -> a
  PROVISIONAL reason (no final verdict, no PASS) and the report header prints them ("COMPARISON SUB-READ-OUT PROBLEMS").
  Without --allow_partial the read-out still refuses (unchanged).
- **The tie tolerance (minor 4): NOT changed.** The decision's condition ("if the pilots confirm ties are exact") cannot be
  met: the pilots log only the counts kway_em_tie / kway_em_bad (25 / 0 / 14 / 35 ties, 0 violations), never the gap, so
  they cannot show that the ties are gap == 0. Making gap == 0 the rule needs a trainer change (Rev H: log or test the gap),
  a new snapshot, regression, references and GPU pilots: not done; Rev G and its 1-2 ulp band stay (by construction an
  exact-match hit whose first-token choice differs has gap 0 under correct indexing, so the band can only absorb an index
  error of exactly 1-2 ulp; the auditor: practically no effect).
- **NOT ADVANCE where none prefers a higher lr (minor 6), documented (descriptive, not a rule):** in stage 1 the prior runs
  only 1e-3 / 2e-3 and none 2e-3..8e-3; the prior's 4e-3 extension comes only after advancing. Under a shared trend towards
  high lr (rb6's none direction) a real effect can screen out (the auditor's CX5: +3 true effect -> NOT ADVANCE). **Such a
  NOT ADVANCE is not evidence of no effect.** The screen marks every NOT ADVANCE whose none stage-1 argmax lies above the
  prior's stage-1 grid ("NOTE (A5.3)", picks field note_none_higher_lr) and prints the sentence under every screen table.

### A5.4 Tests (runs/ptdom/tests/a5/; the A4 tests stay in tests/a4/ as records of the A4 code)
| test | script / log | result |
|---|---|---|
| span tokens | span_tok.py / span_tok.out | SPAN TOK PASS 55/55: every domain's longest ' <item>\n' span (newline its own token) = make_jobs_ptdom.SPAN_TOK; = the pilots' kspan_tokens and item_tokens |
| new tolerances on real GPU data | ptheld_dp_a5.py / ptheld_dp_a5.out | 184 ptheld 53M runs: after step 1250 0 / 38,640 values and 0 / 184 runs exceed (old rule: 1.75%, 130 / 184); eval 0 max 0.0195; the four ptdom pilots: 0 / 720 over, eval-0 max 0.017, later max ratio 0.084 |
| per-run CHECK on the pilots' real B200 JSONs | check_py_a5.py / check_py_a5.out | CHECKPY A5 PASS 62/62: the A5 pilot intent = A4's + dp_tol0 / dp_tol; the A5 CHECK_PY passes all four pilots (exit 0); eval-0 0.06 and tolerance + 0.01 fail exactly the tolerance check, flagged; tolerance - 0.01 passes (A4's CHECK_PY fails it); A4's k-span tampers (0.5 / 0.3 later) now pass; a missing / NaN field fails the fields check, not flagged; a tolerance + another failure not flagged; A4's other tampers, the stage-1 intent and GEOK unchanged; the flag names = readout_ptdom.DP_FLAG_CHECKS |
| read-out / selection self-test | readout_ptdom.py --selftest / selftest_readout_a5.out | SELFTEST PASS 46/46 (A4's 38 unchanged + 8 A5: fixB with reused and new none cells, NO VERDICT while pending, NOT FINAL / PASS after, the generator's F validation, a differing F, --allow_partial sub-read-out problems, the A5.3 note, the A5.1 loader rule) |
| select -> generator plumbing | plumbing_a5.py / plumbing_a5.out | PLUMBING A5 PASS 24/24 (A4's 16 on the A5 build + fixB through the CLI: select --after 2 lists the 16 none jobs by name, --stage 2b renders exactly them with --save_model, NO VERDICT before them, select --after 3 refused before them, complete after, NOT FINAL / PASS + FIXB, a picks file without F refused; a run failing only the 2-way tolerance -> the screen decided, the run listed FLAGGED) |
| the auditor's counter-examples | cx_a5.py / cx_a5.out | CX A5 PASS 6/6: CX1 (deterministic jitter): A4 code PASS for zodiac and compass, A5 code NOT FINAL (F none@5e-4, none re-selected at 5e-4); CX2 unchanged (NOT FINAL by none's own chain); CX3 valid + flagged, screen decided; CX4 PROVISIONAL; CX5 NOT ADVANCE with the A5.3 note |
| pipeline simulation | a5_null_mc.py, .sh / a5_null_mc.out; a5_null_mc_more.py, .sh / a5_null_mc_more.out; pool_mc.py / pool_mc.out | the A5.2 tables (run on the claude_host, 2 processes at nice 19: cpu_short jobs were pending > 1 h) |
| job files | jobcheck_a5.py / jobcheck_a5.out | JOBCHECK A5 PASS 24/24: the 240 re-rendered stage-1 files (A4's 20 checks) and each differs from its A4 version ONLY in the intent (+ dp_tol0 / dp_tol) and the CHECK_PY heredoc (sha256[:12] 12dff9487798, byte-identical in all 240); the 4 submitted pilot files are the A4 records (A4 intent, A4 CHECK_PY, SUBMITTED flags) |
| loader on the pilots, early watcher | pilot_readout_check_a5.out, aux_check_a4_on_a5.out | A4's tests re-run on the A5 code: PILOT READOUT PASS 5/5, AUX CHECK A4 PASS 2/2 |
| submission print | submit_print_a5.out | submit_ptdom.sh --batch B2 --stage 1 (print only): the plan (A5 cost), the snapshot 11f61b9, self-test 46/46, the 60 jobs; submitted: no |

Cost (make_jobs_ptdom.py --plan): stage 1 unchanged (240 runs, 144 B200-h); stage 2+ expected 169 / 252 / 325 runs at 30 / 50 /
70% advancing (A4: 153 / 236 / 310; +~10 B200-h for fixB's none cells), maximum +16 runs per batch.

### A5.5 Not done, and open (nothing here changes A5 without a decision)
- Stage 1 was NOT submitted; no HOLD, picks or SUBMITTED_ file was written in runs/ptdom (the plumbing test's temporary real
  HOLD_B1 is removed by the test, checked).
- A fresh adversarial review of the A5 changes (experiment-audit item v) was not run in this build.
- The pilot confirmations (A5.1): the job-check change is not a pure loosening; the coordinator decides whether the CPU
  re-derivation (check_py_a5.out) stands in for an A5 confirmation or new pilots are run.
- fixB's false-PASS rate among advancing domains is <= 0.05 except at a mild shared trend (+0.75..+1.25 / halving: 0.050-0.052,
  se 0.002-0.003, A5.2), i.e. at the target within simulation error; SF_ALPHA 0.10 is kept as
  decided; the SF 0.05 column is the reference for the user.

## Submission of stage 1 (coordinator, 2026-11-04 11:57 EST; no ptdom result read)
- The A5 audit (tests/audit_a5.md) cleared stage 1 on two conditions: (i) the CPU re-run of CHECK_PY on the four pilots'
  real B200 run files stands in for new pilots - accepted by the coordinator (CHECK_PY only reads the run files, so the CPU
  re-run is equivalent to the in-job check); (ii) its important item 1 (inherited lr-selection gap: P(PASS | advanced)
  0.07-0.08 with an interior lr optimum and identical arms) must be decided and pre-registered before
  select_ptdom.py --after 1 is run. Put to the user with option A (also run none at all three of its stage-1 lrs on seeds
  4-7, and pick none's lr for the selection-free verdict on seeds 4-7) and option B (keep, state the rate).
- Submitted with submit_ptdom.sh --batch {A,B1,B2,B3} --stage 1 --go: 240 jobs, B200, 4 singleton slot chains per batch
  (SUBMITTED_ flags). Early aux-gate check: aux_gate_check.py --batch X once the first prior runs pass step 1500.
- 2026-11-04 14:43 EST: job 21863318 (ptdom-B3-none-lr2e-3-s0) FAILED after 48 s with "CUDA-capable device(s) is/are busy or unavailable" on node gb005 (an infrastructure fault before any training); re-run as ptdom-B3-none-lr2e-3-s0-r1 by the A2 re-run path (make_jobs_ptdom.py --rerun).
- 2026-11-04 14:59 EST: job 21863321 (ptdom-B3-none-lr4e-3-s0) failed the same way on gb005 (CUDA device unavailable, before training); re-run as -r1 with SBATCH_EXCLUDE=gb005; every pending ptdom job is updated with ExcNodeList=gb005 (a faulty node; scheduling only, no training change).
- Correction (2026-11-04 15:08 EST): the ExcNodeList=gb005 update of the pending jobs did NOT take effect (scontrol update returns "Unspecified error" for user jobs on Torch); only the -r1 re-run of 21863321 excludes gb005 (SBATCH_EXCLUDE). Further gb005 failures (CUDA device unavailable, exit before training) are re-run the same way.

## User decision on audit_a5 important item 1 (2026-11-04 16:13 EST, before select_ptdom.py --after 1 and before any stage-1 result was read)
The user chose option B: the stage-2 verdict rule stays as registered in A3-A5 (no extra none runs at all its lrs on seeds
4-7). The user asked that the paper not state the simulated false-PASS rate. The internal record keeps it: the A5 audit's
simulation gives P(PASS | advanced) 0.07-0.08 with identical arms and an interior lr optimum (0.059-0.093 across the
simulated cases); the FINDINGS entry of this screen will carry that number.

## A6 — geo overshoot at a perfect fit (written before any B2 result was read)

2026-11-06, during stage 1 (B2 and B3 still running; picks_A_after1.json and picks_B1_after1.json were written by the A5 code
at 00:49).

**Pre-registration hygiene** [A6.1: replaces A6's "No B2 result was read"; the heading's "before any B2 result was read" means
before any B2 result was seen by the build or entered the rule]:
- **The trigger run's areas were exposed in two places.**
  - Its check.txt ends with the in-job info JSON: read_area / read_final3 / seen_final3 / hard_final3 of the run's six B2
    domains.
  - Its check.json carries the same block under "info".
- **What the build opened.** The build (both the A6 and the A6.1 builds) did NOT open check.txt. It opened the trigger's
  check.json through scripts that print only:
  - check names, their ok flags and the failed check's value (26);
  - the run JSON's geo_planets / cka_planets / grad_prior fields: the overshoot eval, plus the step-0 cka_planets 0.071 quoted
    in A6.4;
  - for every other run, its validity and failed-check names under the A5 / A6 / A6.1 loaders.
- **What it held without printing.** The loader holds every run's areas in memory; real_a6.py compares them for equality
  between loaders and never prints them.
- **The one exception printed** is an A4 test re-run here (pilot_readout_check). It prints the four A4.7 GPU pilots' areas,
  exactly as its A5 record does; the pilots are not results.
- **What the audit opened.** The A6 audit (tests/a6/audit_a6.md) opened the trigger's check.txt per its task list. Its last
  line, with that run's six B2 areas, was visible to the auditor, who did not record or use it.
- **What the rule depends on.**
  - The rule (A6.2 with A6.1.1's bar and cap) depends only on: the geo_<active>, cka_<active> and grad_prior fields, the
    inactive domains' geo / prior-gradient fields, and check names.
  - Its thresholds come from the trainer's numerics: Cauchy-Schwarz, the bf16 path, and the geo / cka fields of the runs
    (A6.4, A6.1.1).
  - **No area, gain or read-out entered the rule or any threshold.**

The coordinator's rule is implemented exactly; where it needed an interpretation, the interpretation is named
(A6.3). **Where A5 (or anything above) disagrees, A6 holds.** Changed: readout_ptdom.py, select_ptdom.py (their A5 versions in
runs/ptdom/superseded_A5/scripts/). **Unchanged:** CHECK_PY, make_jobs_ptdom.py and every rendered job file, the trainer (Rev G,
11f61b9) and runs/ptdom/code, aux_gate_check.py, submit_ptdom.sh. Tests and logs: runs/ptdom/tests/a6/.

### A6.1 The trigger run
- **The run.** out/ptdom-B2-only-planets-lr1e-3-s2 (job 21863151, B200) finished all 75 evals. It failed exactly one of its 36
  job checks: "geo > -0.001 for the active prior and 0 < grad_prior < inf at every on-eval (before a divergence)" (value 26 =
  its on-evals).
- **The values.** At step 6250, geo_planets = −0.0011473 (the training-batch prior value 1 − CKA) and grad_prior = 0.00968.
  At the same eval, cka_planets = **1.000142** unrounded. This is the eval-prompt item-state CKA, the mean of L2 1.000421 and
  L4 0.999862; it overshoots 1 too.
- **Every other on-eval passes.** geo_planets > −0.001 at the other 25. Five of them are slightly negative: −1.8e-4 to
  −4.8e-5, inside the tolerance. grad_prior is finite and > 0 at all 26 (minimum 0.00312).
- **Inactive domains.** No inactive domain has a geo_<d> or gnorm_prior_<d> field, and the separate checks for that passed.
- **Cause.** Both values use the same CKA, (G·K) / (|G| |K|) (pretrain_helix.cka). Cauchy-Schwarz bounds it at 1 only in
  exact arithmetic. At a perfect fit, float error pushes it above 1: planets has 8 items. make_rb6_tasks.py:205 ("float CKA
  can exceed 1 by ~3e-5 at a perfect fit") and make_rb6m_tasks.py:371 ("float CKA exceeds 1 at a perfect fit: pilot 411 /
  416") document this. Here the prior value is computed under bf16 autocast (dom_eval_log) and reached −1.15e-3, just beyond
  this batch's −1e-3. [A6.1, audit_a6 minor 6: corrected below]
  - **Where the value comes from.** At an on-eval it is the training forward's dgeo, the value added to the loss
    (pretrain_range.py:2359-2368, bf16 autocast). dom_eval_log only records it, unrounded.
  - **The mechanism.** cka() computes G = H @ H.T inside autocast (bf16), and centred_gram's t @ t.T is also called inside
    autocast, so K is bf16 too (checked in pretrain_helix.py:51-63, cka and centred_gram, imported by pretrain_range.py:161).
    G * K is therefore a bf16
    product; the sum and the norms run in fp32.
  - **The bound.** Per the audit's derivation and simulation (its bf16_sim.out), at a perfect fit the overshoot is bounded by
    about 2^-9 ≈ 2e-3 per layer.
  - **The rb6 citation.** make_rb6_tasks.py's "~3e-5" dates from rb6's −1e-4 tolerance and is an order of magnitude too
    small for this path. It documents only that the overshoot exists.
- **Consequence under A5.** The run is invalid, so select_ptdom.py --batch B2 --after 1 would refuse: the cell
  only-planets@1e-3 s2 would lack a valid run. A finished run is never re-run (A2.3). The small domains (8 items: planets,
  ranks, compass) can hit the same overshoot again, in stage 1 or in stage 2.

### A6.2 The rule (pre-registered)
- **Which runs it covers.** The run's failed checks are A5.1's set (the GPU-numerics double-path tolerances and / or the
  CKA-rise bar) plus this geo / grad_prior check.
- **When it is valid.** Such a run is **VALID data, FLAGGED "A6", if and only if** the loader verifies all of the following
  from the run JSON itself:
  1. At every eval with prior_on, grad_prior of the active prior is finite and > 0.
  2. At every such eval where geo_<active> <= −0.001:
     - cka_<active> >= 0.999 (a perfect fit; the logged field, unrounded) [A6.1: 0.995 <= cka_<active> <= 1.01, A6.1.1];
     - geo_<active> >= −0.005, a hard floor, so a real sign bug still fails.
  3. No inactive domain has any geo or prior gradient, and the separate job checks for that are present and passed.

  Otherwise the run stays invalid, as under A5, and its reason ends in "(A6 does not apply: ...)".
- **How the check is matched.** By its exact rendered name prefix, as A5.1 matches DP_FLAG_CHECKS:
  readout_ptdom.GEO_FLAG_CHECK = 'geo > -0.001 for the active prior and 0 < grad_prior < inf at every on-eval'.
  - In CHECK_PY the name is an f-string, chk(f"geo > {TOL} ..."), with TOL = the intent's geo_tol.
  - readout_ptdom.a6_guard() runs at the start of every load(). In the CHECK_PY of **every** rendered job file
    (runs/ptdom/job_ptdom-*.sbatch) it asserts: the template exactly once, the TOL line, geo_tol == −0.001 (so the rendered
    name starts with the prefix), and each of the four separate checks named exactly once.
  - So a renamed check fails loudly; it cannot silently stop matching.
- **Where flagged runs are listed.**
  - readout_ptdom.load: rec['geo_flag'] holds the active prior, the number of evals with prior_on, the smallest grad_prior
    and each overshoot eval (step, geo, cka).
  - geo_flags() lists them like dp_flags.
  - The read-out lists them in its text ("FLAGGED (A6, valid data): ...") and in its json (geo_flags).
  - So do readout_ptdom.py --screen and select_ptdom.py ("FLAGGED (A6, valid data): ...").
- **The flag is descriptive.** No rule reads it: the screen, the decision chain, the aux gate and the verdicts treat an A6
  run exactly like any other valid run. Shown by the self-test: the planted A + B2 design with 60 runs flagged A6 gives
  byte-identical screens, chains, gates, verdicts and report.

### A6.3 Interpretations (named)
- **(a) "Every eval with prior_on" is read literally.** It means every curve entry with prior_on, including any after a
  divergence. That is stricter than the check's "before a divergence"; for a run that did not diverge the two sets are the
  same. A diverged run with a non-finite grad_prior after the divergence stays invalid.
- **(b) "grad_prior of the active prior".** This is the logged grad_prior = |Σ λ_d g_d|. With one active prior it equals
  |λ g_d| (= gnorm_prior_<d>, the trainer's dom_grads).
  - A6 requires exactly one active prior, the same in domains.<d>.prior, args.prior_domains and the arm name.
  - none never runs this check.
- **(c) The cka field.** The logged cka_<d> is the mean over read layers 2 and 4 of the item-state CKA on the fixed eval prompts,
  a float written unrounded. It is a second measurement of the same fit, on other states than geo_<d>, which is computed on
  the training batch. The per-layer values (cka_<d>@L2 / @L4) are not tested.
- **(d) A non-finite geo_<active>** at an eval with prior_on cannot be verified (NaN <= −0.001 is false), so A6 does not apply.
- **(e) The failure must be explained.** The JSON must show at least one eval with geo_<active> <= −0.001, or a bad grad_prior
  (which fails 1). If the check failed with nothing in the JSON to explain it, the loader and CHECK_PY disagree: the run
  stays invalid.
- **(f) Condition 3 from the run JSON.** The loader requires all of:
  - no non-None geo_<x> for any inactive domain at any eval;
  - no gnorm_prior_<x> key for any inactive domain at any eval;
  - the numbers prior's geo == 0 at every eval;
  - the four separate checks present exactly once in check.json and passed: "the active prior applied ...; every inactive
    one never", "geo_<d> logged ... never for an inactive one", "weighted prior gradients ... from the active prior only",
    "numbers prior off ...". A failure of any of them already lies outside the allowed set.
- **(g) Boundaries.** geo = −0.005 and cka = 0.999 are inside (>=) [A6.1: cka = 0.995 and cka = 1.01 are inside]. geo = −0.001 is an overshoot eval: the check fails at
  geo <= −0.001.

### A6.4 Why the floor is −0.005 and the CKA bar 0.999 [A6.1: the bar is 0.995 with a cap 1.01, A6.1.1; the floor is unchanged]
- **The floor −0.005 admits only an overshoot.** 1 − CKA >= −0.005 means the batch CKA is at most 1.005. A value between −0.005
  and −0.001 is 1 − CKA of a fit within 0.5% of perfect that float error pushed just above 1. Any other source is not
  admitted.
- **What it leaves room for.** It is about 4× the observed overshoot (−0.00115) and 5× the check's own tolerance [A6.1,
  audit_a6 minor 6: an empirical multiple, not a derivation; the bf16 bound, about 2e-3 per layer, would justify −0.0025]. That leaves
  room for the same bf16 artefact on the other 8-item domains without widening the class of admitted values.
- **A real sign bug fails.** A prior computed as CKA − 1, or a target sign flipped, gives geo far below −0.005 away from a
  perfect fit: the trigger's cka_planets at step 0 is 0.071, which would give geo ≈ −0.93.
- **The CKA bar 0.999 asks for a perfect fit on independent states.** The eval-prompt CKA at the same eval must be within
  0.1% of 1. This is the condition under which the documented float overshoot occurs. The trigger shows 1.000142. [A6.1:
  within 0.5% (0.995) and at most 1.01, A6.1.1: 0.999 lay inside the eval CKA's own bf16 noise]
- **An overshoot without a perfect fit fails.** A negative training-batch geo while the eval CKA is, e.g., 0.95 is not this
  artefact (coordinator's case: −0.0012 at cka 0.95), and the run stays invalid.

### A6.5 Files (sha256[:12] before -> after)
| file | A5 | A6 |
|---|---|---|
| runs/ptdom/readout_ptdom.py | bd1cf6261ff7 | **b44568ce401d** (GEO_FLAG_CHECK / A6_KEEP / GEO_FLOOR / GEO_CKA_MIN, a6_guard, geo_a6, the load() branch, geo_flags, A6_LINE, the read-out's geo_flags + text line, --screen line, patch_a6 / fake_a6, 3 self-test cases) |
| runs/ptdom/select_ptdom.py | 3a1d706de434 | **d851f9822efc** (the FLAGGED (A6, valid data) line, docstring) |
| llmadd_torch/make_jobs_ptdom.py (CHECK_PY) | 6061b299129c | 6061b299129c (unchanged) |
| runs/ptdom/aux_gate_check.py | 52937ad7e656 | 52937ad7e656 (unchanged) |
| job_ptdom-*.sbatch | – | unchanged: CHECK_PY byte-identical in all 346 rendered files (= make_jobs_ptdom.CHECK_PY) |

New: tests/a6/real_a6.py (3aadf9bdf620), plumbing_a6.py (1d6c35d6217c), mutation_a6.py (e3702dea4a5f); superseded_A5/
(README.txt + scripts/).

### A6.6 Tests (runs/ptdom/tests/a6/; the A5 tests stay in tests/a5/ as records of the A5 code)
| test | script / log | result |
|---|---|---|
| read-out / selection self-test | readout_ptdom.py --selftest / selftest_readout_a6.out | **SELFTEST PASS 49/49**: A5's 46 unchanged (every PASS line and the planted report text identical to selftest_readout_a5.out) + 3 A6. (1) **Loader on synthetic run JSONs (15 cases).** Valid + FLAGGED A6: −0.00115 at cka 1.0 (the trigger); −0.005 at cka 0.999 (boundaries); two overshoot evals (both listed). A6 + A5.1 (2-way tolerance + CKA rise) → valid, listed in both dp_flags and geo_flags. Invalid with the reason: −0.0012 at cka 0.95, −0.006 at cka 1.0, grad_prior 0 at one on-eval, an A6-like failure + another failed check (data_sha), a failed check nothing explains, a geo_zodiac in an only-elements run, a separate check missing from check.json, an overshoot + grad_prior 0, a NaN geo, cka 0.9989, geo −0.0050001. (2) **Planted A + B2 design** (screen → stage 2 → 2b → read-out) with every run of only-months / only-planets / only-elements (60 runs) flagged A6: all valid; screens, chains, gates, verdicts and report identical to the clean runs, only the FLAGGED (A6) line added. (3) **The check name.** CHECK_PY's f-string rendered with geo_tol −0.001 = 'geo > -0.001 ... (before a divergence)' = the loader's prefix; a6_guard passes on the rendered job files and rejects a renamed check, another geo_tol, a renamed separate check, the template twice and an empty dir |
| the tests bite | mutation_a6.py / mutation_a6.out | **MUTATION A6 PASS**: unmutated self-test PASS. Five mutations of the rule each make an A6 case FAIL: floor −0.01, CKA bar 0.9, A6 never applies, guard disabled, the inactive-domain verification dropped |
| the trigger run + every real ptdom run | real_a6.py / real_a6.out | **REAL A6 PASS 6/6**. Trigger: its only failed check = the prefix exactly; the A5 loader marks it invalid, the A6 loader VALID, flagged A6 (planets, 26 evals with prior_on, overshoot at step 6250 only; in geo_flags, not dp_flags). Every ptdom run dir (187 cells, 185 finished; both loaders back to back, none finished in between): **valid 184 (A5 loader) → 185 (A6 loader)**; per batch A 60 → 60, B1 60 → 60, B2 38 → 39, B3 26 → 26. **The only run whose validity changed is ptdom-B2-only-planets-lr1e-3-s2** (invalid → valid, failed only the geo check). Every other record is identical under both loaders, all fields but why / geo_flag, so A's and B1's written picks are unaffected. It is the only finished run that failed the geo check; no finished run is invalid under A6; FLAGGED A5.1: none. a6_guard passes on all 346 rendered job files; CHECK_PY is byte-identical in all and = make_jobs_ptdom.CHECK_PY |
| select → generator plumbing | plumbing_a6.py / plumbing_a6.out | **PLUMBING A6 PASS 26/26**: plumbing_a5's checks on the A6 build (every PASS line as in plumbing_a5.out, except two replaced and two added). Replaced: (1) the real-HOLD_B1 --submit step is NOT run: 72 ptdom jobs were pending, a real HOLD_B1 would stop any B1 job starting meanwhile, and make_jobs_ptdom.py is unchanged by A6 (its record: plumbing_a5.out); a read-only check of the B2 job file's HOLD / HOLD_B2 line is kept. (2) The last check is now: no real HOLD / HOLD_<X>, and the picks files are unchanged (name + sha256). Added (section 8): select --after 1 on B2 with two picked-cell runs flagged A6 (one also A5.1) → exit 0, both listed under FLAGGED (A6, valid data), the second also under FLAGGED (A5.1), and the picks file = the clean design's (all fields but inputs); the same with cka 0.95 → that run is 'not valid' ("A6 does not apply: (2) step 10: cka_zodiac 0.95 < 0.999") and select refuses (lacks valid runs). Meanwhile another session submitted B1 stage-2 jobs (SUBMITTED_ flags 319 → 332, job files 346 → 346); the test writes no flag or job file (generator --no-write, never --submit) |
| A5 tests re-run on the A6 code (loader changed) | cx_a5_on_a6.out, check_py_a5_on_a6.out, pilot_readout_check_a6.out, aux_check_a4_on_a6.out | CX A5 PASS 6/6, CHECKPY A5 PASS 62/62 (CHECK_PY unchanged; the flag names = DP_FLAG_CHECKS), PILOT READOUT PASS 5/5, AUX CHECK A3 8/8 + A4 2/2: **each log identical to its A5 record** (diff after normalising temp paths: 0 lines) |

Not re-run (their code is unchanged by A6): jobcheck_a5.py (it calls make_jobs_ptdom.py --stage 1 without --no-write; the job
files are covered read-only by real_a6.py and a6_guard), span_tok, ptheld_dp_a5, the simulations, the trainer regression, the
references and the smokes.

### A6.7 Audit status
experiment-audit v pending (the coordinator runs a fresh audit). Nothing was submitted, cancelled or re-run by this build, and
no picks / HOLD file was written in runs/ptdom. [A6.1: done: tests/a6/audit_a6.md, 0 blocking, 0 important, 9 minor; handled by
Amendment A6.1 below]

## Amendment A6.1 — the A6 audit's follow-ups (tests/a6/audit_a6.md; before any B2 result is used)

2026-11-06. Four follow-ups decided by the coordinator after the fresh A6 audit (0 blocking, 0 important, 9 minor),
implemented exactly. "A6.1.x" numbers this amendment's items; they are distinct from subsection "A6.1 The trigger run" above.
**Where A6 disagrees, A6.1 holds** (the superseded A6 lines carry an [A6.1] marker).
- **Status at the time.** B2 has not been selected: no picks_B2_*.json exists. No B2 result entered a decision or this
  amendment (hygiene: the statement at the top of A6).
- **Changed.** readout_ptdom.py and submit_ptdom.sh; their A6 versions are in runs/ptdom/superseded_A6/scripts/.
- **Unchanged.** select_ptdom.py, CHECK_PY / make_jobs_ptdom.py and every job file, aux_gate_check.py, the trainer.
- **No job state was touched.** No job, job file, HOLD or picks file.

### A6.1.1 The cka bar 0.999 → 0.995 and an upper cap 1.01 (audit_a6 minors 1, 2)
- **Rule (replaces A6.2 (2)'s cka condition).** At every eval with prior_on where geo_<active> <= −0.001,
  **0.995 <= cka_<active> <= 1.01**. The geo floor −0.005 is kept. Everything else in A6.2 / A6.3 is unchanged. Below the
  bar: "(not a perfect fit)". Above the cap: "(above any float overshoot: an anomaly)". Both → A6 does not apply, the run
  stays invalid.
- **Why 0.995: 0.999 sat inside the eval CKA's own noise.** By the audit's numerics over the 185 finished runs (geo / cka
  fields only; numerics.out):
  - the eval-prompt cka_<d> overshoots 1 by up to +1.04e-3 as a 2-layer mean (+1.48e-3 per layer);
  - the bf16 error is symmetric, so it reaches about 1e-3 below 1 at a perfect fit too;
  - at the real geo < 0 on-evals, 4 / 187 have cka < 0.999, the minimum 0.998003, and none is < 0.99.

  A finished run is not re-runnable (A2.3). One with geo <= −0.001 and an eval cka in [0.998, 0.999) would stall the selection,
  the problem A6 exists to solve.
- **Why 0.995 still means a perfect fit.** It lies about 5× the eval CKA's noise (~1e-3) below 1. It admits every real geo < 0
  on-eval: their minimum is 0.998. The other conditions still bind: geo in [−0.005, −0.001] on the training batch, every other eval
  > −0.001, grad_prior finite > 0, no inactive field. With them, only a near-perfect fit is admitted: by the audit's simulation
  (bf16_sim.out), training-batch geo <= −1e-3 from an exact geo >= 1e-3 occurred 0 times.
- **The cap 1.01.** CKA <= 1 in exact arithmetic (Cauchy-Schwarz). The largest real overshoot is +1.5e-3 per layer, so a cka
  above 1.01 is a computation anomaly, not rounding. The audit showed A6 accepted cka 1.30 (adversarial.out).
- **Not acted on (no decision).** The count of overshoot evals per run stays uncapped (the audit's second point under minor 2).
  The real run has 1 / 26; geo_flags prints every overshoot eval, so a systematic shift would be visible in the FLAGGED list.

### A6.1.2 submit_ptdom.sh no longer overwrites the A5 record (audit_a6 minor 4)
- **The change.** submit_ptdom.sh:29-31 wrote its self-test log to tests/a5/selftest_readout_a5.out, which A6.6 and A6.1.6
  use as the A5 record. It now writes a new dated file, tests/a6/selftest_readout_submit_<UTC time>_<pid>.out, and prints
  its path.
- **The A5 record was not overwritten.** tests/a5/selftest_readout_a5.out: 11-04 11:51, sha256[:12] 7edc08331b6f, "SELFTEST
  PASS (46/46)". Nothing had to be restored.
- **Tested without running the script.** The full script calls make_jobs_ptdom.py --snapshot and --stage 1 --print, so
  tests/a6/submit_fragment_a61.sh tests it in parts:
  - bash -n on the script;
  - its diff against the A6 copy is exactly the header line and this block;
  - the block itself is extracted from the script and run;
  - result: one new dated log, SELFTEST PASS 49/49; the A5 record unchanged; 0 lines writing or reading tests/a5.

### A6.1.3 Pre-registration hygiene (audit_a6 minor 7)
The statement at the top of A6 replaces "No B2 result was read". The build did not open check.txt. It read the trigger's
check.json, which holds the same areas under "info", printing only check names, the geo / cka / grad_prior fields and
validity. The audit opened check.txt and saw its last line with the six B2 areas. The rule and its thresholds depend only on
geo / cka / grad_prior fields, inactive-domain fields and check names; no area entered them.
- **Correction to the coordinator's wording.** The request said "the build and the audit opened check.txt". The build's
  scripts (scratch inspect_planets.py, tests/a6/real_a6.py, the loader) open only check.json.
- **What the stated fact rests on.** The "check.txt" string in readout_ptdom.a6_guard and real_a6.py is the heredoc
  separator inside the job files, not a file read. /scratch does not update access times, so atime cannot show either way.
  The statement therefore rests on the scripts' source.

### A6.1.4 The audit's minor items (one line each)
1. CKA bar inside the eval CKA's bf16 noise (liveness): **fixed**, A6.1.1 (0.995).
2. No upper limit: cka cap **fixed** (1.01, A6.1.1); the number of overshoot evals per run **not acted on** (no decision; real 1 / 26; every one is listed).
3. a6_guard failure modes (a half-written or empty job file, or no job file next to a copied module: loud AssertionError; the self-test depends on runs/ptdom's job files; lru_cache does not re-read new files within a process): **carried**. All are loud and fixed by re-running; real_a6.py points the A6 copy's guard at runs/ptdom; A7 must do the same for a superseded A6.1 copy.
4. submit_ptdom.sh:30 overwrites the A5 record: **fixed**, A6.1.2 (it had not been overwritten).
5. An A6 run still shows FAILED in check.json (valid_data_flagged false), check.txt and Slurm (exit 4): **not acted on**. CHECK_PY must not change; the loader decides validity, as for A5.1; nothing reads valid_data_flagged; rerun_plan never re-runs a finished run.
6. A6.1 / A6.4 argument not rigorous (the ~3e-5 citation, the bf16 G·K mechanism, the 2^-9 bound, dgeo computed in the training forward and only recorded by dom_eval_log, the floor an empirical multiple): **fixed** (text, A6.1 Cause and A6.4, [A6.1] markers); the floor −0.005 is kept.
7. Pre-registration hygiene (check.txt shows the areas): **fixed**, A6.1.3 and the statement at the top of A6.
8. mutation_a6 / the self-test do not cover one-active-prior, unexplained failure, 'present once': **fixed**. Self-test cases added: two active priors, an arm-name mismatch, a separate check present twice. Mutations added: m7 / m8 / m9, plus m6 (no cap).
9. The trigger now enters B2's stage-1 screen; integrity refuses a -r dir over a finished run: **not acted on** (record only, as the audit says).

### A6.1.5 Files (sha256[:12] A6 → A6.1)
| file | A6 | A6.1 |
|---|---|---|
| runs/ptdom/readout_ptdom.py | b44568ce401d | **1e2f17522e47** (GEO_CKA_MIN 0.995, GEO_CKA_MAX 1.01 and the cap check in geo_a6; the why / A6_LINE text; patch_a6 active / extra_active; the self-test cases of A6.1.1 / minor 8) |
| runs/ptdom/submit_ptdom.sh | d2295528c00a | **85c1250c4c5f** (the self-test log path, A6.1.2) |
| runs/ptdom/select_ptdom.py | d851f9822efc | d851f9822efc (unchanged) |
| llmadd_torch/make_jobs_ptdom.py (CHECK_PY), aux_gate_check.py, job files | 6061b299129c, 52937ad7e656, CHECK_PY e36e77be4530 x 346 | unchanged |
| tests/a6/real_a6.py, plumbing_a6.py, mutation_a6.py | 3aadf9bdf620, 1d6c35d6217c, e3702dea4a5f | ff7ad0314328, cace37dbf8f8, 802c13450e0b (edited in place: their A6 versions are not kept, their A6 logs *_a6.out / real_a6.out etc. are) |
| tests/a6/submit_fragment_a61.sh | – | e58bbc5716f9 (new) |

### A6.1.6 Tests (runs/ptdom/tests/a6/, logs *_a61.out; the A6 logs are kept unchanged)
| test | log | result |
|---|---|---|
| read-out / selection self-test | selftest_readout_a61.out | **SELFTEST PASS 49/49**. The A6 loader case now has 22 synthetic runs. New (the coordinator's): cka 0.9975 → valid, 0.994 → invalid "< 0.995 (not a perfect fit)", 1.02 → invalid "> 1.01 (above any float overshoot ...)". Also new: cka 1.01 (the cap) → valid; two active priors, an arm-name mismatch → invalid "not one active prior"; a separate check present twice → invalid. Changed for the new bar: the boundary case is geo −0.005 at cka 0.995 (valid), and "just below the bar" is cka 0.9949 (invalid; A6 had 0.9989, now valid). Every other line as in selftest_readout_a6.out, except the guard line's job-file count (294 at the A6 run, 346 now: B1's stage-2 files, rendered by another session in between) |
| the tests bite | mutation_a61.out | **MUTATION A6 PASS**: 9 mutations each fail an A6 case (A6's 5 + m6 no cap, m7 one-active-prior dropped, m8 unexplained failure accepted, m9 'present once' relaxed) |
| the real runs | real_a61.out | **REAL A6 PASS 9/9**. A5 → A6.1 loader: **valid 184 → 185; the only run whose validity changed is ptdom-B2-only-planets-lr1e-3-s2** (A 60 / 60, B1 60 / 60, B2 38 → 39, B3 26 / 26). Every other record identical; FLAGGED A6: the trigger only (cka_planets 1.000142 in [0.995, 1.01]). A6 → A6.1 loader: the same validity and geo_flag in all 187 cells; only the trigger's why text differs (the printed range). **picks_A_after1.json (d2974639f124) and picks_B1_after1.json (6aa1f96d9dde) = their re-derivation**: select --after 1 rebuilt in memory on each batch's 60 stage-1 cells, every field, with the A6.1 code and with the A5 code (control), no integrity / sub-read-out problem, nothing written. a6_guard on the 346 job files, CHECK_PY byte-identical |
| select → generator plumbing | plumbing_a61.out | **PLUMBING A6 PASS 26/26** (the cka 0.95 refusal now reads "< 0.995"); every check name as in plumbing_a6.out; picks files unchanged by the test |
| submit_ptdom.sh self-test step | submit_fragment_a61.out | A6.1.2: bash -n; diff = header + block; one new dated log 49/49; the A5 record unchanged |
| A5 tests on the A6.1 code | cx_a5_on_a61.out, check_py_a5_on_a61.out, pilot_readout_check_a61.out, aux_check_a4_on_a61.out | CX 6/6, CHECKPY 62/62, PILOT READOUT 5/5, AUX CHECK 8/8 + 2/2: each identical to its A6 log (0 differing lines after normalising temp paths) |

### A6.1.7 Audit status
experiment-audit v for the A6.1 diff: the coordinator decides (small diff: the bar / cap constants, one check, test cases and
the log path).
Coordinator decision (2026-11-06): no separate fresh audit for A6.1. A6.1 applies the fresh A6 audit's own findings
(tests/a6/audit_a6.md minor 1-2, 4, 6-8); the diff is the cka bar 0.999 -> 0.995, the new cap 1.01, one rejection message, the
selftest log path and test cases. Evidence: 9/9 mutations caught, selftest 49/49, A5 -> A6.1 validity changes only the trigger
run (184 -> 185), A6 -> A6.1 identical in all 187 cells, picks_A_after1 / picks_B1_after1 equal to a fresh rebuild. The
coordinator checked the constants (readout_ptdom.py:131 GEO_TOL, GEO_FLOOR, GEO_CKA_MIN, GEO_CKA_MAX = -1e-3, -5e-3, 0.995,
1.01) and their use (:332-333, :442) before writing this line.


## Coordinator log, 2026-11-06 ~15:10-15:40 EST (restarted session; selections by the pre-registered scripts, unchanged code)
Code at run time (sha256[:12]): readout_ptdom.py 1e2f17522e47, select_ptdom.py d851f9822efc, make_jobs_ptdom.py 6061b299129c
(= A6.1). No HOLD file.
- **Stage 1 complete in all four batches** (A / B1 earlier; B2 60/60 and B3 60/60 now). Two runs flagged A6, both valid data:
  B2 only-planets@1e-3 s2 (geo -0.001147, cka 1.000142) and B3 only-geo_periods@2e-3 s3 (geo -0.001129, cka 0.999589).
  Aux gate passed in every cell of B2 and B3 (lowest median LM share: chinese_zodiac 0.860).
- `select_ptdom.py --batch B2 --after 1` (rc 0; select_B2_after1.txt 0b728a6e584f) -> picks_B2_after1.json 9b1f309a5ce9:
  ADVANCE ranks only. `--batch B3 --after 1` (rc 0; select_B3_after1.txt 3fecb41a1e24) -> picks_B3_after1.json
  35fc75a56d6f: ADVANCE none (B3's chain ends here).
- B2 stage 2 (16 extension runs at seeds 0-7 + 8 confirm runs at seeds 4-7):
  `make_jobs_ptdom.py --batch B2 --stage 2 --picks runs/ptdom/picks_B2_after1.json --submit`, started 15:11 EST. Slurm
  was slow and the controller was briefly DOWN, so sbatch calls hung for minutes. All 24 jobs were submitted (~15:11-15:50, last
  job 22110746; 84 B2 SUBMITTED_ flags = 60 + 24; 24 B2 jobs queued).
- A's stage 2 is complete: `select_ptdom.py --batch A --after 2` (rc 0; select_A_after2.txt f04bfcee2aac) ->
  picks_A_after2.json e8f53426de49: "complete (nothing more to run)"; no pick moved.
- Advancing domains for the stage-2 Holm family (7): months, colours, cities (A); countries, animal_size, people (B1);
  ranks (B2).
- A partial read-out was run for status only: `readout_ptdom.py --allow_partial` -> readout_partial_20261106.txt
  46f158b0b29b. It prints PROVISIONAL because B1 and B2 are incomplete. It is not a verdict, and no rule or decision
  depends on it.
- 2026-11-06 ~17:00 EST: B1 stage 2 is complete. `select_ptdom.py --batch B1 --after 2` (rc 0; select_B1_after2.txt
  7e62f2628a12) wrote picks_B1_after2.json (a46195a9d615): stage 2b = E none@5e-4 (seeds 0-7; animal_size's none argmax
  moved to 1e-3, the edge); no C, no fixB. Submitted with `make_jobs_ptdom.py --batch B1 --stage 2b --picks
  runs/ptdom/picks_B1_after2.json --submit` (8 jobs).
- 2026-11-06 21:30 EST: job 22106293 (ptdom-B2-only-ranks-lr5e-4-s5, a stage-2 extension run, gb008) finished training and
  exited 4 on one failed check: 'geo > -0.001 ... at every on-eval'. geo_a6 (A6.1 loader) on its run JSON: applies (why []).
  The only overshoot eval is step 5000, geo_ranks -0.001083 with cka_ranks 0.999498; grad_prior min 0.00873 over 26 on-evals.
  The run is VALID data, flagged A6. It is the third A6 case, the second on ranks' family of small comparison domains (after
  B2 planets s2 and B3 geo_periods s3). Not re-run (a finished run is not re-runnable, A2.3).
- 2026-11-07 ~13:00 EST: every ptdom job had left the queue. `select_ptdom.py --batch B1 --after 3` (rc 0) -> picks_B1_after3.json
  f000ac20f179: B1 complete. `--batch B2 --after 2` (rc 0) -> picks_B2_after2.json 72dfee364876: stage 2b = A5 fixB none@1e-3
  seeds 0-7 for ranks (the prior's pick 1e-3 is outside none's range 2e-3..1.6e-2). Submitted with `make_jobs_ptdom.py --batch B2
  --stage 2b --picks runs/ptdom/picks_B2_after2.json --submit` (8 jobs). Status read-out readout_partial_20261107.txt: still
  PROVISIONAL (B2 waits for fixB). It is not a verdict.
- 2026-11-07 ~21:45 EST: B2 fixB done. `select_ptdom.py --batch B2 --after 3` (rc 0; select_B2_after3.txt 9506643e6022) ->
  picks_B2_after3.json e1c5fd68adf4: B2 complete (none re-selected including 1e-3: pick 4e-3, argmax 8e-3). All four decision
  chains are complete. **Strict read-out** `readout_ptdom.py --json runs/ptdom/readout.json` -> readout.txt 6aeb5f6f56c0 /
  readout.json 3caeac06fe8d (no PROVISIONAL). Results:
  - PASS: months [FLOOR], cities, countries, people [FLOOR].
  - NOT SHOWN: colours, animal_size [FLOOR, CEILING], ranks [FLOOR, FIXB, SHORTCUT NOT TESTABLE].
  - The other 17 domains did not advance at stage 1.
  A fresh results audit (experiment-audit v) was started before any FINDINGS entry. Its report goes to tests/audit_results.md.
  Sent misalign the "nothing left on B200" message, with the caveat that a follow-up combination batch may use B200
  pending the user's decision; NOTIFY_misalign_when_b200_free.txt is kept until then.
