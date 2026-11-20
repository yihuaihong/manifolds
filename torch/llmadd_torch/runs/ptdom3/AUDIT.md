# ptdom3: intent, necessity, cost (build phase 1 of 3, 2026-11-17, exploratory tier)

Question: which new domains could replace S4 or people in the "perfect six"? Each domain's manifold prior is an auxiliary
CKA loss in 53M pretraining. Protocol = ptdom / pt6comb, unchanged:
- 53M model, 3e8 tokens; numbers lines at 2%, first after the text; K=2.
- Prior on the item's last token, window (0.05, 0.4), lambda 1; numbers prior off.
- Same eval schedule, 75 evals. Seed 0 only.
- Prior lr {1e-3, 2e-3, 4e-3}; none lr {2e-3, 4e-3, 8e-3, 1.6e-2}; one shared none.

Exploratory tier: no pre-registered decision rule and no read-out gate. Stamps and provenance work as usual.

## Domains (tools/toy3.py, commit 6c51dc14 on branch ptdom3)

| domain | items | line | target (prior) | Tier L | split | read-out set (seed-0 stand-in size) |
|---|---|---|---|---|---|---|
| cyc12 | 12 nonce | `a with b is c` (Z12) | circle, 2-D | yes | class 0.5 | held_nonid (62) |
| torus | 36 nonce | `a and b is c` (Z6xZ6) | two circles, 4-D | yes | class 0.6 | held_nonid (506) |
| hcube | 32 nonce | `a xor b is c` (Z2^5) | {+1,-1}^5 | yes | class 0.6 | held_nonid (372) |
| d6 | 12 nonce | `a then b is c` (D6, non-abelian, "a then b" = R_b R_a) | faithful 2x2 rep, flattened 4-D | yes | question 0.5 | held_fresh_nonid (51) |
| box5 | 25 nonce | `a plus/minus b is c` on {-2..2}^2 | grid, 2-D | yes | class 0.5 | held_no_negtwin_nonid (144) |
| notes | dropped | Gb, Bb and every sharp are 2 OLMo-2 tokens, so the last token is shared across pitch classes | | | | |
| chess | dropped | no square a1..h8 is 1 token (' e'+'4'); the last token is the rank digit, shared by 8 squares and by the numbers operands | | | | |

"Tier L": the answer is an exact function of the inputs' target rows. tests/toy3_test.py checks this for every question:
- cyc12 and torus: angle sum;
- hcube: elementwise product;
- d6: matrix product;
- box5: vector sum or difference.

Nonce names, in tools/toy3_nonce.json:
- the rarest 117 OLMo-2 single tokens ' CVC' following the S4_NAMES rule;
- train.bin counts run from 7 to 260;
- the 51 s4/dim names are excluded;
- 69 words, names, abbreviations and vulgar tokens were skipped, each with its reason listed.

## Intent spec (one block per number the read-out prints)

**{d}_readout (exact match)**
- Teacher-forced exact match over the answer plus newline, with argmax over the full vocabulary, on the domain's read-out set.
- It measures whether held-out relational facts are produced, i.e. generalisation through the structure.
- Planted perfect answerer = 100; a shuffled-item answerer is about 1/n_items (tests/toy3_test.out).

**area**
- 100 × the mean of {d}_readout over the 75 evals: learning speed and level combined.
- It is the primary number per run.

**pick, gain, same-lr gain**
- pick: the arm's best area over its lrs. The prior arm has 3 lrs; none has 4.
- gain = prior pick − none pick, in area points.
- Same-lr gain = prior − none at lr 2e-3 and at lr 4e-3, the lrs both arms share. It separates the prior's effect from the lr grid.

**final3**
- The mean of the last 3 evals of {d}_readout: the end level, which area can hide.

**floor and ceiling flags**
- The none pick near chance (floor) or near 100 (ceiling) means the gain is not interpretable.
- These are printed marks, not gates. Thresholds are inherited from readout_ptdom2.py.

**LM gradient share and clip coefficient (per prior run)**
- The LM loss must dominate the update (manifold_aux_loss_principle). The read-out prints both from the trainer log.
- A share below 1/2 is marked, not blocked.

## Necessity (which decision each cell changes)

| cells | count | decision it changes |
|---|---|---|
| none × lr {2e-3, 4e-3, 8e-3, 1.6e-2} | 4 | the reference for every gain in the batch. Its data mix contains the five new domains' lines, so ptdom/ptdom2's none runs cannot stand in |
| {cyc12, torus, hcube, d6, box5} prior × lr {1e-3, 2e-3, 4e-3} | 15 | "is domain X a candidate to replace S4 or people in the perfect six?" One domain per 3 cells; the pick needs the lr scan |
| canary (a few hundred steps, all 5 domains with the prior on, through the read-out) | 1 | whether to release the 19 runs: lines, read-out, a non-zero prior loss and the step-0 asserts all work |

At most 6 domains per batch, so the 5 domains form one batch with one shared none. No other seeds or controls are added.

## Cost

- 19 runs × ~28 min ≈ 8.9 B200-hours, plus about 0.1 for the canary.
- The B200 is preemptible, scheduled as singleton slot chains as in ptdom2.
- Wall time ≈ ceil(19 / slots) × 28 min once the jobs start. The B200 queue has taken 1–2 days before a start.
- With no model saving, the /scratch footprint is only logs and JSON.

## Inherited fixes (to port in phases 2 and 3)

- runs/ptdom/AUDIT.md and tests/audit_{a1..a5,build,results}.md
- runs/ptdom2/AUDIT.md and tests/audit_{a2..a5,build,results_S}.md
- runs/pt6comb/AUDIT.md: CHECK_PY_6, the a6 guard, geo_a6_multi

Phase 1 changed no file that those fixes touch: pretrain_range.py and toy_struct.py are byte-identical to 11f61b97.

## Phase 2 additions (2026-11-17, commit ef7b8854 on branch ptdom3)

caesar (the coordinator's addition):
- Z26 on the capital letters, A = 0: line ` D shifted by C is F`, target the circle (2-D), split class 0.5, read-out held_nonid (310 / 314 / 310 cells for seeds 0-2).
- Items are REAL tokens ' A'..' Z', not nonce names. Letter tokens are very frequent in ordinary text, so the prior site is a common token with a strong pre-existing representation, unlike every nonce domain (train.bin counts 7-260). A caesar gain or loss is therefore not comparable one-to-one with the nonce domains. tests/toy3_test.py prints the letter counts next to the nonce maximum.
- Single-token check, with and without a leading space: tests/toy3_test.py (job 23439548). The trainer also asserts one distinct token per item after a space at load (Rev H). If ' A'..' Z' fail, caesar is dropped (remove it from toy3.TOY3 and the trainer's TOY3 literal together).

d6 split changed: train fraction 0.5 -> 0.4.
- Under the trainer's own stream (SeedSequence(4400 + seed), child 9), the 0.5 split leaves 46 / 46 / 52 held_fresh_nonid cells for seeds 0 / 1 / 2. That is below the trainer's n_readout >= 50 assert, which is kept.
- 0.4 gives 58 / 62 / 63 (0.45 gives 53 / 53 / 57: too thin). Source: tests/d6_split.py.
- The phase-1 stand-in sizes in the table above came from a different stream. The trainer-stream sizes for seed 0 are: cyc12 59, torus 489, hcube 368, d6 58 (at 0.4), box5 150, caesar 310.

Rev H of tools/pretrain_range.py (sha a398fccbfe93), made by tests/patch_trainer.py (12 exact, counted replacements):
- Every new path is guarded by `d in TOY3` or REVT (a toy3 domain in --domains).
- REVT is OR-ed into REVE, so a toy3-only batch logs grad_lm_share and clip_coef, which the read-out needs.
- Own stream SeedSequence(4400 + seed); the 4100 / 4200 / 4300 streams are untouched.
- The trainer's asserts:
  - at load: names are single, distinct tokens, disjoint from s4 / dim, and decode back;
  - toy3.split: every item is a trained input and a trained answer;
  - held-out rows are disjoint from train rows (plus the common DLINE held-prompt-not-in-train assert);
  - read-out >= 50;
  - the toy3 nonce counts are re-counted against toy3_nonce.json;
  - at step 0 (all compute still unspent): every toy3 domain's TRUE prior has a non-zero gradient norm, for every arm including none, and the norms are printed. This is the "prior loss component non-zero" check. The prior is first applied at 5% of the steps, and dom_grads computes the true prior at every eval step whether it is on or off.
- hard / hard_ro (the closure analogue of s4_hard) go into info as closure_latin and n_hard_closure. They are never an eval mask.
- The c_far (shuffled-target CKA < 0.15) assert stays strict for toy3. If one fails in the dry run, report it; do not loosen it.
- Byte identity against 11f61b97 (--dry_inputs stdout, each config with its job's own ARGS) is tested in tests/byteid.sh (job 23439548). The 12 configs:
  - default 53M;
  - ptsz 136M none;
  - rb6 190M all6 (s4 / dim);
  - ptdom A / B1 / B2 / B3;
  - ptdom2 A / B1 / B2 / B3 (shuf and only arms);
  - pt6comb B all6.

Cost with caesar: 4 none + 6 domains x 3 lrs = 22 runs x ~28 min ≈ 10.3 B200-hours, plus about 0.1 for the canary.

## Code / design audit (exp-auditor, 2026-11-17; coordinator's summary)
- Blockers: none (no wrong-direction risk, no crash risk, no cell that cannot answer its question). d6 target faithful (mul[a,b] = R_b R_a matches "a then b"); prior sites correct incl. caesar; the shared none comparable (step-0 data sha, lm0, dry inputs equal across arms).
- Important 1: twin leakage in the held read-out set. A held cell whose token triple is a training fact (XOR / involutions) can be answered by recall: seed 0 hcube 302/368 (82%), d6 30/58 (52%), cyc12 10%, torus 8%, caesar 4%, box5 0% (auditor's /scratch/yh6210/tmp_audit/ptdom3/twin_share.py; toy3.py:68-70, :211-212). Direction: compresses the prior's gain (false negative). Fix (coordinator, read-out only, no rerun): readout_ptdom3.py gains a twin-free area / final3 from the per-question hits, primary for hcube (n 66) and d6 (n 28); d6 also without its 3 (a,a)->e cells. For any later combination run, the trainer-side split should exclude twins.
- Important 2: no shuffled-target arm -> a positive gain reads "this prior helps", not "this geometry helps" (rare nonce tokens + tied embeddings). Wording only; a shuffled control belongs in the confirmation of any domain that is chosen.
- Important 3: the pick grid is asymmetric (none max over 4 lrs up to 1.6e-2, prior over 3 up to 4e-3): the pick gain is biased low; read the same-lr gains (2e-3, 4e-3) first.
- Minor: a missing grad_prior at an on-eval would mark a run INVALID (0/422 earlier single-prior runs miss it; the canary covers it); cyc12 repeats each fact ~10x torus; box5's copy mask also drops a = 2b minus cells.

## Submission (coordinator, 2026-11-17; the user: 「ptdom3 可以用 B200」)
read-out 9abd89ad75a7 (twin-free columns; selftest 37/37, cpu job 23445742). `make_jobs_ptdom3.py --check` re-rendered the canary's read-out sha (9abd89ad75a7 now in its job file), then `--submit canary` + `--submit screen`: 23 jobs (runs/ptdom3/submitted.tsv; log tests/submit_go_20261117.log). expwatch batch ptdom3 armed, milestones = the canary and the slot chains. If the canary fails: HOLD_S (or scancel the ids in submitted.tsv).
- 2026-11-18 ~14:00 EST (coordinator): canary 23454812 COMPLETED in 2 min 16 s on B200: step-0 line present (all six toy3 prior gradients non-zero), `CANARY READ-OUT OK`, every check PASS (args = intent, code_sha a398fccbfe93, data_sha = CPU reference, step-0 lm within 1e-3, splits = CPU reference). The first 8 screen runs (4 none + cyc12 x3 + torus 1e-3) COMPLETED (~18 min each); 14 pending. expwatch re-armed without milestones (wakes on failure or at the batch end).
