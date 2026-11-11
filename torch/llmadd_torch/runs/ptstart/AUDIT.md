# AUDIT: pretraining with the in-context eval (F65) and the window start-time sweep (ptstart)

Written 2026-10-13 before submitting ptstart. Code: llmadd-burst tools/pretrain_helix.py.

## Metric: test_ctx (held-out addition accuracy in training position)
- Single definition: fraction of the 1000 held-out pairs (pairs[:1000] of a fixed permutation of all (a, b), a, b in 10..99)
  for which argmax of the logits at the last position equals the first token of str(a + b). Implementation:
  pretrain_helix.py evaluate() over evalsets["test_ctx"]. (The sum 20..198 is one token in the OLMo-2 tokenizer.)
- Measures: whether the model produces the right sum for a pair it was never trained on, in the same place as
  training puts arithmetic. High = generalises (or recalls the swapped pair, see below); low = does not.
- Token position: logits at the last token of "<held-out FineWeb chunk>Q: a + b = " (the trailing " " token); in training
  the same " " token sits at index seq-3 and predicts the answer at seq-2. Eval sequence length = seq - 2, so the
  scored position has the same absolute index (seq-3) as in training.
- Input format and context:
  | dimension | training | eval (test_ctx) | same? |
  | position of the scored token | seq-3 | seq-3 | yes |
  | what precedes the prompt | 1 line: FineWeb text; 4 lines: line 1 after text, lines 2-4 after other arithmetic lines (with answers) | FineWeb text (val.bin) | partly: matches line 1's context and the last line's position; no in-context solved examples before it |
  | template | 4 formats at random | format 0 ("Q: a + b = "); test_plus_ctx = format 1 | yes (format 0 is one of the 4) |
  | operand range | 10..99 | 10..99 | yes |
  | data split | trainp = pairs[1000:] | pairs[:1000] | ordered pairs disjoint; BUT 870/1000 test pairs have the swapped pair (b, a) in trainp -> see strict subset |
  | text | train.bin (first 400M tokens) | val.bin (next 2M tokens) | disjoint |
  | model state | train(), bf16 autocast | eval(), bf16 autocast | dropout none in this config |
- Expected magnitude: chance ~ 1/180 (0.6%); step 0 = 0; a model that has learned addition in training position ~ 100%
  (train_ctx and in-batch EMA should agree within ~5 points once learned).
- Controls: positive = train_ctx (training pairs, same position) and in-batch EMA; negative = step 0 / random init;
  shuffled-helix arm = wrong-geometry control; upper bound = train_ctx at the end ~ 100%.
- Known caveat (added by this audit): commutativity. 87% of "held-out" pairs have their swap in training, so test_ctx mixes
  generalisation with recall of a + b from b + a. Report test_ctx_strict = the 130 pairs whose swap is also unseen
  (118 with a != b, 12 with a == b) next to it.

## Metric: tokens to 50% / 90% (t50, t90)
- First eval step at which test_ctx >= 0.5 / 0.9, in tokens (step x bs x seq); evals every 250 steps (8.2M tokens) ->
  resolution 8.2M tokens. Reported as the median over seeds; ratio = none median / arm median.

## ptstart sweep (what is varied)
- helix_window with windows (start, end) = (0, .4), (.05, .4), (.2, .4) [vary start], (.1, .2), (.1, 1.0) [vary end];
  (.1, .4) and none are in pt7c (seeds 0-3); replication seeds 4, 5 of none and (.1, .4) added (check iv).
- Everything else identical to pt7c: 4 arithmetic lines, 3e8 tokens, lam 5, eval_every 250, --ctx_eval --save_model.

## Audit results so far (2026-10-13)
- ii (asserts): added to arithmetic_sets(): ordered test / train pairs disjoint; every training line ends "<sum> \n" with
  the sum one token; numbers 0..999 are single tokens; every eval prompt ends with the same token that precedes the
  answer in training; *_ctx sequences have length seq-2. Operand / answer indices of every training line are asserted
  against the token ids (lpos).
- vii (decoded inputs side by side): inputs_sample.txt / inputs_sample_v2.txt. Scored index 509 in training and in every
  *_ctx set. Difference noted: with 4 lines, the in-batch EMA is measured on the last line, which follows 3 solved
  lines (few-shot-like); test_ctx follows text only (= the condition of a row's first line). ema_first_line is now
  logged and drives helix_gate.
- i (two routes): offline re-evaluation of the saved pt7c weights (tools/pt_eval_saved.py, job 19688817) reproduces the
  logged final test_ctx exactly (e.g. none s0 99.3 / 99.3, window s3 99.9 / 99.9).
- vi #2 (memorisation vs generalisation): strict subset (130 pairs unseen in both orders) at the end of pt7c:
  window 100 / 100 / 100 / 100, helix 100 / 99.2 / 98.5 / 100, none 96.9 / 96.9 / 96.2 / 95.4, shuffled 95.4 / 96.2 / 94.6 / 96.2.
  So none's test_ctx (~99.3) is inflated by ~3 points by commutative recall; the helix arms are not. The speed result
  (t50 / t90) is on test_ctx; strict_ctx is logged at every eval in the new runs, so speed on strict can be checked.
- iii: negative control step 0 = 0; upper bound train_ctx ~ 100 at the end.

## New arms / metrics (ptfinal)
- --geo_site ctx: CKA on operand tokens inside the training lines (marked in inputs_sample_v2.txt), target = helix
  features of the operand values in the batch (with repeats; the Gram is centred over the batch).
- --geo_site ans: CKA on the token before the answer (the one that predicts it), target = helix of the sum (20..198).
- helix_gate: prior on while ema_first_line < gate_thr.
- Harder end-point tests, all GENERALISATION (never trained): add3 (operands 100..499), add1 (second operand 1..9),
  sub (a - b), test_f2 / test_f3 (other training formats, held-out pairs). Reported as such, not as "accuracy".

## Second adversarial review (new sites / gate / hard tests), 2026-10-13 — and what changed for ptu (commit after 2becdba)
- Positions (ctx operands, answer predictor), gradients through store, gate signal: verified correct by the reviewer
  (28,400 lines and 6,400 batch rows checked; store == hidden_states, requires_grad True).
- Important 1 (accepted, partly verified): in the ORDERED split, none's strict / test_ctx errors at the end concentrate
  on sums with little training support (192: 3 of 7 pairs trained; 21-25), whether or not the swap was trained; strict
  happens to contain three sum-192 pairs. So "commutative recall" is not the explanation of the end-point gap; "rare
  answers" is. Re-check running (check_edge_rerun.txt). In the unordered split only 4 of 985 held-out items have
  support < 5 (none 0); uheld_ctx_common / _rare are logged.
- Important 2 (fixed): the answer-site helix with T = 100 made s and s + 100 similar (after z-scoring cos 0.77 > 0.52
  for s, s + 1), conflicting with reading out the answer; now periods (2, 5, 10, 200): cos(s, s + 100) 0.28.
- Important 3 (fixed in pt_summary): same condition + seed in two batches = same init and data stream; deduplicated.
- Important 4 (resolved by the unordered split): test_plus / f2 / f3 shared the commutative contamination.
- Minor: shuffle permutation now drawn per seed; first-batch asserts on meta positions; floor tests (add3 answers never
  trained as outputs, sub never trained, add1 duplicates) are exploratory only.
