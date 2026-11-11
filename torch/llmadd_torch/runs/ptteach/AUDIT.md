# ptteach — automatic (teacher-derived) target geometry for the pretraining prior

Setup = ptu (runs/ptstart/AUDIT.md for the base intent spec: unordered split, ctx eval, strata). Only the TARGET changes.

## Target
- Source: tools/teacher_targets.py reads the atlas states (tools/manifold_atlas.py, states_<tag>.npz) of the prompt
  "Q: x" (neutral template), x = 10..99, last token, OLMo-2-7B at layer 16 (main checkpoint, or the 9B-token one).
  The atlas z-scores each hidden dimension over all atlas items; teacher_targets then centres over the 90 numbers.
- feat_k8 = top-8 PC coordinates (U_k S_k), 56% of variance; feat_full = all 4096 centred dims (its Gram = the teacher Gram).
- CKA to the hand helix: main L16 full 0.617, k8 0.574; 9B L16 full 0.464.
- Loss: same as the helix arms, λ·(1 − CKA(student operand states, target Gram)) on the operands inside the training
  lines (--geo_site ctx), rows of the target indexed by operand value (T90[x − 10]).
- Shuffled control: target rows permuted by a value permutation; first draw with CKA(shuffled, true) < 0.15, else the
  farthest of 3000 (feat_full cannot go below ~0.15: high-rank part survives any shuffle). Recorded as shuf_target_cka.
- --geo_site ans with --target_file is refused (the target covers operands only).

## Metrics (unchanged from ptu)
- uheld_ctx: exact answer at training position on ~985 unordered held-out problems; t_reach 90% = speed.
- cka_helix: CKA of "Q: x" states to the HAND helix (for teacher runs too); cka_target: to the teacher target.
- cka_ops_in_context (hand helix) and cka_ops_in_context_target (teacher), for operands inside lines.

## Expected
- If teacher geometry carries what helps: t16k8 / t16full reach 90% faster than none, comparable to ctx-helix (7.4×).
- Shuffled teacher: no speedup (as ctx-shuf). If shuffled ≈ true, the effect is regularisation, not geometry.
- t9Bk8 (early teacher, addition not yet learned): if it helps as much, the useful part is present before competence.
- Upper check: train_ctx → ~100% at end for all arms; negative: shuffled arms ≈ none.

## Train vs eval
Identical to ptu (training position, same formats, unordered held-out pairs never in training in either order).

## Audit 2026-10-13 (fresh adversarial subagent; checks in scratchpad/audit_ptteach) — fixed at commit c2dbfce
- BLOCKING (fixed): pt_summary.label / plot_pt_curves had no target in the label, so teacher arms merged with ptu's
  ctx-helix / ctx-shuf and with each other, and the per-seed dedupe kept only t16full. Label now carries
  target=<file>:<key>; unfinished (requeued) runs are skipped; a paired (same-seed) area test is printed.
- Correction to "Target" above: centred_gram z-scores every target column, so feat_k8 enters as 8 EQUAL-weight PCs
  (whitened). What training sees, CKA to the hand helix: main k8 0.625, main full 0.731, 9B k8 0.423 (not 0.574 /
  0.617); whitened k8 vs the raw teacher Gram 0.794; full is unaffected (0.997). t9Bk8 vs t16k8 therefore differs in
  helix-likeness too (0.42 vs 0.63), not only in teacher competence.
- Added arms (19736229–19736241): none-t16k8 (none with cka_target logged; training identical, verified 0 diff),
  t16full-nohelix (teacher with the helix span projected out of every column: CKA to helix 0.000, to the teacher
  0.460), 2pct-t16k8-shuf (2% shuffled control at the ctx site).
- Target files made read-only; sha256: main_L16 ddaa9164…, 9B_L16 b8c4f6d4…; runs now record target_sha256.
- Known limits: 4 seeds, shuffled arms very variable (ptu ctx-shuf t90 = 287/152/never/78 M) → use paired area
  tests; t90 resolution 4.1M tokens (ctx-helix t90 ≈ 25–29M, so ±15%); feat_full's shuffle cannot go below CKA ~0.18.
