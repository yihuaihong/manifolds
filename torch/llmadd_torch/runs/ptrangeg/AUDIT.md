# ptrangeg — experiment B with gated / windowed priors (P1, 2026-10-14)

Question: does switching the prior off once the model has learned keep the 0..999 gains and remove the language-
modelling cost (whole-run priors in ptrange: val bits/byte +1-2%)?

## Setup (as ptrange; only the schedule changes)
- Operands 0..999 (one-, two-, three-digit buckets equally often), whole-number tokens, held-out unordered pairs per
  bucket, 3e8 tokens, 4 arithmetic lines per row, seeds 0-3, prior on the operands inside the training lines.
- --schedule gate: prior on while the EMA (0.98) of in-batch exact match (all answer tokens, teacher forced) on the
  first arithmetic line of each row is < 0.5; window: steps in [0.05, 0.4) of training.
- Arms: helix3 gate, helix3 window, digitcirc gate. References (ptrange, whole run): none, helix3, digitcirc.

## Metrics
held_b11..b33 / seen_* / ood_b4x exact match; val_bits_per_byte (LM cost, comparable across tokenisations);
prior_on_frac; ema_first_line; grad_lm / grad_prior / grad_cos when on.

## Expected (written before results)
- Gate / window keep most of the speed-up of the whole-run prior (t90 within ~1.5× of whole-run) and bits/byte
  within noise of none (whole run: +0.012 to +0.022).
- prior_on_frac for the gate: the share of training before first-line accuracy reaches 0.5.

## Audit 2026-10-14 (fresh subagent; scratchpad/audit_p1) — fixed at a0ea2e5 before any job ran
- IMPORTANT (fixed): the gate used the mixed-bucket first-line accuracy; in the whole-run reference runs that average
  crosses 0.5 at 7-8% of training while held_b33 is 0.05-0.09 (verified myself on ptrange/whole-helix3, 4 seeds;
  b33 reaches 0.9 only at steps 6750-10250). The gate now stays on until the LOWEST per-bucket EMA (6 operand-length
  buckets, first lines) reaches 0.5, and in-batch correctness requires the newline after the answer (as the evals).
  Logged: ema_bucket (per bucket), ema_bucket_min, prior_toggles.
- Known limits: grad_* only on steps where the prior is on; geo when the prior is off = true 1 - CKA (pretrain_range)
  but 0.0 in pretrain_helix; 4 seeds and a fixed ~65k-token val set for bits/byte (offline re-evaluation on >= 2M
  tokens with the saved models before claiming "no LM cost"); references ran on A100/H100, these on L40S.

## Results audit 2026-10-14 (fresh subagent; scratchpad/audit_ptrangeg_results) -> FINDINGS F79
- Speed kept on average (area gate 86.9, window 88.9, whole 88.2), but the gate did NOT keep b33: final 94.8 vs whole
  98.3 (p .035), b33 t90 1.56x slower -- it switches off on in-batch training accuracy while held-out b33 is ~50-60%.
- LM cost: gate +0.0218 = whole +0.0223 bits/byte; window +0.0037 (n.s., CI includes 0); gate - window +0.018 (p .003).
  The expectation above ("bits/byte within noise of none") holds for the window only.
- Interpretation narrowed: onset (0 vs 5%) and offset (10-20% vs 40%) are confounded; the prior is satisfied within
  ~250 steps of onset. Still to do: offline re-evaluation of bits/byte on >= 2M tokens with the saved models.
