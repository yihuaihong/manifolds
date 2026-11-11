# ft36 — fine-tuning prior screen over all 36 atlas domains (2026-10-14)

User request (2026-10-14): "我想要36个都测试一下" — test every atlas domain with a manifold prior. This is the cheap
screen (LoRA fine-tuning of OLMo-2-7B); pretraining versions come later for the domains where geometry helps.
Code: tools/ft36.py (snapshot runs/ft36/code, commit in code/COMMIT); domains / targets / templates from
tools/manifold_domains.py and the primary target per domain from tools/manifold_atlas.py (PRIMARY / primary()).
Generator: llmadd_torch/make_jobs_ft36.py (2 checkpoints x 6 jobs x 6 domains).

## Question
For each domain: does pulling the item representations toward the domain's primary target geometry (prior) improve
held-out accuracy on the domain's own task more than training on the task alone (none), and more than pulling toward
the same target with items permuted (prior_shuf)? At 42B (early; number-helix gains largest there, F10-F14) and 210B.

## Arms (per domain, seeds 0-2, same split for every arm and seed)
- none: LoRA r 8 (all attention + MLP projections), task CE on the answer tokens only, AdamW lr 1e-4, bs 16, 280 steps.
- prior: + 20 * mean over layers 4, 8, 12, 16 of (1 - CKA(last-token states of "Q: <item>" for ALL items of the
  domain, centred Gram of the primary target)). Every step uses the full item set (not a sampled batch).
- prior_shuf: same, target rows permuted (a new far permutation per seed: CKA(shuffled, true) < 0.15 when possible).

## Tasks and splits (zero-shot question formats = the atlas test templates)
- offset domains "k <unit> after x" (candidates = all items, chance 1/n):
  - cyclic (days, months, hours, pitch, keys, zodiac, chinese_zodiac): k = 1..2n; hold out 40% of the classes
    (item, k mod n) with k mod n != 0 -> a held-out question's equivalent (same item, same shift mod n) is never
    trained (fixes the 100% equivalent leakage of the old combo split).
  - other (numbers "x + k =", number_words, ordinals, years, letters, greek, planets, elements, geo_periods): hold
    out 40% of (item, k); combos are given back to training when their item or k would otherwise never be trained.
- comparison domains (candidates = the two options, chance 1/2): every valid comparison of the domain's rule
  (manifold_domains._pair_answer, atlas margins), at most 1500 comparisons, both orders; hold out 40% of the
  unordered option pairs {b, c} (both orders and every anchor of a held pair are held).

## Metric: test accuracy (main) and train accuracy
- Definition: the right answer has the highest summed log-prob of " <answer>" after the prompt among the candidates;
  mean over held-out questions (train: <= 400 training questions). Implementation ft36.accuracy / seq_logprobs
  (no padding: grouped by length).
- Measures: generalisation of the fine-tuned task to held-out item / shift (or option-pair) combinations.
- Token position: scores are the log-probs of the candidate tokens after the prompt; the prior reads the last token
  of "Q: <item>" (position = item's last token, no context).
- Expected: base ~ chance for offset domains at 42B; comparisons often above chance (F74: knowledge domains decodable).
  Prior gains of the size seen before (+5..+30 points) only where the model is still learning the task.

## Metric: geometry
- CKA (true target) of the "Q: <item>" states at layers 4/8/12/16 before and after training (all arms).

## Train vs eval
| dimension | train | eval | same? |
|---|---|---|---|
| template | domain template, zero-shot | same | yes |
| items / shifts | training classes / pairs | held-out classes / pairs | disjoint by construction (asserts) |
| scoring | CE on answer tokens (teacher forced) | candidate ranking by summed log-prob | different by design |
| model state | train (LoRA dropout 0.05) | eval | standard |

## Controls
- Negative: prior_shuf (same strength, wrong geometry); positive: numbers at 42B should reproduce the known helix gain.
- Ceiling: train accuracy should approach 1 for none after 280 steps on most domains.

## Known limits
- Small domains have tiny test sets (planets 7, geo_periods / life_stages 12, ranks 16, greek / ordinals 33,
  days 34): report n and treat them as underpowered; family-level pooling for claims.
- Steps are fixed (280) while training-set sizes range 11..1816 -> epochs differ by domain (same for all arms).
- Summed log-prob favours shorter multi-token candidates (same for all arms).

## Audit 2026-10-14 (fresh general-purpose subagent, read-only; scratchpad/audit_ft36) — jobs held (runs/ft36/HOLD), code rewritten
Verified correct by the audit: candidate scoring positions (max diff 4e-6 vs naive), right padding + labels (CE equal to
naive), item tokens are suffixes of "Q: item", regrouping order, LoRA reset (domain B after A bit-identical to B alone),
gradients reach the LoRA below the hooked layers, cyclic class split has 0 equivalent leakage, comparison pairs one-sided.
Problems and what the rewrite does (I verified B1, B3, I1 myself before changing anything):
- B1 BLOCKING (base): base.json of a job's 2nd..6th domain was measured with the previous domain's trained adapter
  (|lora_B| 8.5 when days followed planets). NOW: base per seed with model.disable_adapter().
- B2 BLOCKING (design): the prior shaped "Q: <item>" probe states, which no task prompt contains (items sit after
  "five days after", mid-sentence, or at position 0 for numbers); F10 / F40 worked because probe and task shared the
  state. NOW: every task prompt starts with "Q: " and the prior acts on the item tokens INSIDE a fresh batch of 64
  training prompts per step (the states the task reads; F71's computation site). The probe geometry is still logged.
- B3 BLOCKING (pitch / keys): " C#" = [" C", "#"], so " C" (also a candidate) always won; pitch test ceiling 0.47.
  NOW: answers and candidates end with "\n"; an assert forbids any candidate token sequence being a prefix of another.
- I1 IMPORTANT: numbers test had 375/968 (39%) problems whose swap was trained (number_words 11/45). NOW: unordered
  key {x, k} for numbers and number_words; an assert forbids a held-out problem with its swap in training.
- I2 IMPORTANT: tree targets (animals, languages) were z-scored per column, which drops the hierarchy weights (18% / 12%
  of comparisons contradicted the task). NOW: raw centred features for "tree" / "within" targets.
- I3 IMPORTANT: one split for all seeds and tiny test sets. NOW: the split follows the seed; per-question test hits are
  saved for item-level tests; report n and n_test_answer_trained; family-level pooling for claims.
- I4 IMPORTANT: snapshot taken once from the working tree. NOW: the ft36 snapshot is refreshed from a clean commit
  after the rewrite (and the generator refuses a dirty tree).
- I5 IMPORTANT (interpretation): the target carries the answers of held-out comparisons (sphere / hue / compass >= 99%
  consistent), so prior > none on knowledge domains may be extra supervision rather than an inductive bias; prior_shuf
  controls strength, not information. Report it that way; a prior-blind-item control is a possible follow-up.
- minor: us_cities plane vs great-circle answers 94% consistent; emotions plane includes arousal (task = valence);
  elements 9/36, geo_periods 3/12, planets 1/7 test answers never appear as a training answer (n_test_answer_trained
  logged); hooks now on layers[L] as in F10 / F40 (was L-1); ft36 CKA is not the atlas CKA (no cross-domain z-scoring);
  far_perm ordinals seed 0 = 0.152; identical target rows (animals 42, languages 45, kinship 14 ...) mean CKA -> 1 asks
  to merge items; CKA baselines are high at small n; result files now written atomically and re-run when unreadable.
