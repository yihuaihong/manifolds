# pttext — natural-text prior (P1, 2026-10-14)

Question: can the number prior work without acting on the arithmetic lines, by shaping the two-digit numbers that
occur naturally in the web text? (A real pretraining run has no synthetic lines to aim at.)

## Setup (as ptu, F71; only the site changes)
- 53M OLMo-2-style model, FineWeb-Edu, 3e8 tokens, unordered held-out split (985 problems never trained in either
  order), 7% (4 lines) and 2% (1 line) arithmetic appended to each 512-token row, seeds 0-3.
- --geo_site text: positions = tokens "10".."99" in the TEXT part of each row (before the arithmetic lines) that are
  whole numbers (no digit token or "."/"," + digit on either side; OLMo-2 splits digits in chunks of 3 without the
  leading space). ~54 per batch of 32 rows. Target Gram = helix features of their values (with repeats), as for ctx.
  A batch with < 16 numbers (or < 4 distinct values) skips the prior; text_used_frac logs how often it acted.
- Arms: text (whole run), text window (0.05, 0.4), text + gate (EMA of first-line in-batch accuracy < 0.5),
  text shuffled (whole run; same strength, wrong geometry). 2%: the same + gate-ctx (reference not in ptu).

## Metrics
- uheld_ctx (exact next-token answer at the training position), t90 / curve mean / final, as in F71 (pt_summary.py).
- cka_ops_in_context: CKA of the operands INSIDE the arithmetic lines to the helix (transfer from text to the
  computation site — the mechanism F71 identified); cka_text: CKA at the text numbers; val loss.

## Expected (written before results)
- If shaping number tokens anywhere transfers: text arms faster than none and cka_ops_in_context rises above none's
  (0.38 / 0.30 at L2 / L4 in F71); text-shuf no faster than none.
- If only the computation site matters (F71 mechanism): text ≈ prefix-site result (2.9× at 7%, weak at 2%), well
  below ctx (7.4×).
- LM loss: whole-run text prior may cost more than ctx (it touches ordinary text); gate / window should not.

## Train vs eval
Identical to ptu: same evals at the training position; only where the prior acts changes. Text numbers are never
the evaluated problems (evals use separate prompts after a held-out FineWeb chunk).

## Audit 2026-10-14 (fresh subagent; scratchpad/audit_p1) — fixed at a0ea2e5 before any job ran
- IMPORTANT (fixed): cka_text had no baseline; text numbers are skewed (10-19 = 38%, ~35 distinct values per batch)
  and repeats alone give CKA 0.41-0.43 with no geometry. Added arm text-none (none + text-site logging; training
  identical to none) for 7% and 2%. Transfer is read from cka_ops_in_context (comparable across arms), not cka_text.
- IMPORTANT (fixed): 4 seeds cannot separate shuffled / 2% arms from none (ptu 2pct-helix is bimodal); all pttext
  conditions now have 8 seeds (88 jobs); the ptu references have seeds 4-7 queued (ptu8).
- Minor (fixed): the text filter also excludes decimals (" .45", " $.99"), ranges / dates / ratios / times / exponents
  ("1939-45", "10/19/13", "1/10", "10:30", "1e10"), list indices ("#45") and digit groups ("12 34"); tested on
  constructed cases. Actual numbers per batch: 72 (4 lines) / 80 (1 line) on average, never below text_min.
- Known limits: cka_text of text-shuf is the fit to the shuffled target (different meaning); text-gate may flip on
  and off (toggles not logged in pretrain_helix; geo_on_frac per eval); run_guard cannot recover a claimant that dies
  without requeue after the other copy has exited (check for unfinished runs before analysis).

## Results audit 2026-10-14 (fresh subagent; scratchpad/audit_pttext_results) -> FINDINGS F78
- All numbers recomputed from 153 run JSONs and checked against the job logs (0 mismatches).
- Wording fixes: cka_ops_in_context is computed on the operands of the HELD-OUT EVAL prompts at the training position,
  not on the training lines; text numbers per batch were 55.8 (7%) / 61.0 (2%), min 25, text_used_frac ~0.9998.
- Deviation from this design: seeds 4-7, text-none and 2pct text-shuf were cancelled (not started) after the 7% results,
  when the project moved to the unified 0..999 setup (pt999). Claims are 4-seed; shuffled-vs-none is undecided at n=4.
- Not foreseen above: a text prior switched on at step 0 (whole run or gate) costs +0.12..0.15 nats of final val loss;
  the window (0.05, 0.4) costs +0.01..0.03.
