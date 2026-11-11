# pt999b — where and when the prior acts, in the unified 0..999 setup (2026-10-14)

Shared setup, metrics, train-vs-eval table and known limits: runs/pt999/AUDIT.md (same script, data, split, evals,
L40S; analyse together with pt999: tools/pt999_summary.py runs/pt999/out runs/pt999b/out). 7% arithmetic, 4 seeds.

## Questions
1. WHERE (same schedule): ctx window (0.05, 0.4) here vs text window (0.05, 0.4) in pt999 vs prefix window here; and
   prefix whole / gate vs ctx whole / gate (pt999). F71 at 10..99: ctx 7.4x, prefix "Q: x" 2.9x (whole); F78: text
   between them, closer to ctx.
2. WHEN (text prior onset): windows (0, 0.4), (0.05, 0.4) [pt999], (0.1, 0.4), (0.2, 0.4). F68 (prefix, 10..99): onset
   ~5% best, step 0 worst; F78 (text, 10..99): step-0 onset costs +0.12..0.15 nats of val loss, the 5% window ~0.

## New code: --site prefix (tools/pretrain_range.py)
- Each step with the prior on: n_prefix = 256 prompts "Q: x" (4 tokens: 'Q', ':', ' ', x) at position 0, x drawn with
  lengths 1-3 equally often (the operand distribution) from its OWN random stream (3000 + seed; saved in checkpoints),
  so the training batches are identical to every other arm of the same seed. Prior = 1 - CKA(last-token states at
  geo_layers, helix3 Gram of the drawn values), as for ctx.
- The prefix forward overwrites the hooked states; the training forward's states are restored for the read-outs
  (cka_ops_helix3 is always measured on the operands of the training batch, fixed layers).
- When the prior is off, the logged geo uses a step-seeded generator (does not touch the prior's stream).

## Expected (written before results)
- Site: ctx >= text > prefix in speed (area, t90); prefix's cka_ops_helix3 below ctx / text.
- Onset: (0.05, 0.4) and (0.1, 0.4) fastest; (0, 0.4) slower and with a val-loss cost; (0.2, 0.4) slower (late).

## Controls
- Negative / references: pt999 none and shuffled gates (8 seeds). Regression: ctx / text paths of this snapshot equal
  to pt999's (CPU, scratchpad/test_prefix.sh).

## Audit 2026-10-14 (fresh subagent, read-only; scratchpad/audit_pt999b) — no blocking issue
Verified: 'Q: x' = 4 tokens for all 1000 x, prior on position 3 at layers 2 / 4, Gram matches the drawn values (<1e-5),
training-forward states restored before every read-out, prior gradient only through the prefix states, training data
streams identical across arms (md5 per step), schedules as intended, ctx / text paths unchanged (111/111 + 117/117).
- IMPORTANT (added arms): prefix prior gradient ~3.9x LM at init vs ctx 1.3x (clip 1.0) -> site is confounded with
  strength: 7pct-win-prefix-lam1.7 (lambda matched); no shuffled prefix control: 7pct-win-prefix-shuf. 4 seeds, saved.
- Correction: x is drawn with lengths 1/2/3 equally often; the operands in the lines are 0.28 / 0.35 / 0.37 (held-out
  rejection thins 1+1), so "drawn like the operands" is approximate. Also unlike F69 / F71 (all 90 values once per
  step): 256 draws with replacement. 'Q: x' is the first 4 tokens of eval format 0 (without the text before it).
- The 28 first jobs do not save models (the 8 added ones do). --site prefix with gradient checkpointing now raises.
