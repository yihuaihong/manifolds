# downstream — general-language benchmarks of the from-scratch pretrained models (evaluation only; 2026-10-22)

User approval (relayed by the coordinator): 「可以的，没问题，只要跟对应的模型比较是公平的就行」. Evaluation only, no
retraining. Written and committed (git branch `downstream`, report/audits/downstream_AUDIT.md = a copy of this file)
BEFORE any score of a project model was computed. The known-answer checks below (random-init models, public GPT-2 small)
run before the real run; their results are appended under "Known-answer results" without changing anything above it.

## Question
What does the manifold prior cost (or not) on general language benchmarks? Fair comparison = prior vs none at the SAME lr
and the SAME seeds (same init seed and data order; the only difference is the prior). Plus a descriptive gap to the
reference = none at its LM-optimal lr.

## Models (read-only inputs; nothing is written into a source run dir)
| group | path | cells |
|---|---|---|
| 505M 7% | runs/ptlr/out/500m-lr{1.2e-3,4.8e-3,9.6e-3,1.92e-2}-7pct-none, 500m-lr1.2e-3-7pct-win-ctx; runs/ptsize999/out/500m-7pct-{none,win-ctx,gate-ctx,gate-text} (lr 3e-4, from run args) | none@{3e-4, 1.2e-3, 4.8e-3, 9.6e-3, 1.92e-2}, win-ctx@{3e-4, 1.2e-3}, gate-ctx@3e-4, gate-text@3e-4; seeds 0-3 (none@9.6e-3 and win-ctx@1.2e-3 also 4-7) |
| 190M rainbow | runs/rb6_190m/out/*/*/model_*.pt | none@6e-4, none@4.8e-3, all6-lam1@6e-4, only-numbers@6e-4; seeds 0-3 |
| 53M / 190M 7% (descriptive) | runs/ptlr/out/{53m,190m}-*, runs/pt999/out/7pct-*, runs/ptsize999/out/190m-7pct-* | seed 0 only per cell (CLEANUP_2026-10-22.md); pt999 7pct-none also seeds 1-7 |
| later (same command) | runs/rb6_505m, runs/pt2s505, runs/rb6_190m_ext | comparisons pre-registered by amendment before their scores are read |

A model is evaluated only if model_*.pt exists and its run_*.json is finished (last curve step = steps - 1). The weights
are loaded with `pretrain_helix.build_model(run args, tokenizer)` + `load_state_dict(strict=True)` (as
tools/pt_eval_saved.py / offline_lm.py), eval mode.

## Benchmarks and metrics (unique definitions; implementation tools/downstream_eval.py)
All scores are percentages over the FULL split (no subsampling); n is printed next to every score.
| benchmark | split, n | input (context + ⟦scored continuation⟧) | primary metric | chance |
|---|---|---|---|---|
| BLiMP | nyu-mll/blimp, 67 paradigms x 1000 pairs = 67,000 | `<eos>` + ⟦sentence⟧ (whole sentence scored, no leading space) | acc = share of pairs with sum log p(good) > sum log p(bad) (strict; a tie counts wrong) = mean of the 67 paradigm accuracies | 50 |
| LAMBADA (OpenAI) | lambada_openai lambada_test.jsonl, 5,153 | `<eos>` + text without its last word + ⟦" " + last word⟧ (text.rsplit(" ", 1)) | acc = every token of the last word is the greedy argmax; also ppl = exp(mean over items of the summed NLL of the word's tokens) (lm-eval-harness definitions) | 0 |
| HellaSwag | validation, 10,042 | `<eos>` + harness preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize()) + ⟦" " + preprocess(ending)⟧, 4 endings | acc_norm = argmax of sum loglik / len(ending in characters) (harness) | 25 |
| ARC-Easy | test, 2,376 | `<eos>` + "Question: {q}\nAnswer:" + ⟦" " + choice⟧ | acc = argmax of the summed loglik | mean of 1/n_choices (~25) |
| PIQA | validation (test labels are hidden), 1,838 | `<eos>` + "Question: {goal}\nAnswer:" + ⟦" " + sol⟧, 2 solutions | acc = argmax of the summed loglik | 50 |
acc_norm / acc of the other kind are stored too (descriptive). LAMBADA ppl, BLiMP per field / per paradigm: descriptive.
Data: data/general/{blimp (downloaded 2026-10-22, HF nyu-mll/blimp), lambada_openai, hellaswag, ai2_arc, piqa_parquet
(HF ybisk/piqa refs/convert/parquet, downloaded 2026-10-22)}.

## Train vs eval conditions
| dimension | training | evaluation | same? |
|---|---|---|---|
| tokenizer | OLMo-2 (stage1-step2000-tokens9B dir) | the run's own args.tokenizer | yes |
| document start | FineWeb-Edu docs are `text + <eos>` in one stream (pretrain_data.py); a doc starts after `<|endoftext|>` (id 100257; the tokenizer's bos = eos) | every eval sequence starts with `<eos>` | yes (doc-start convention) |
| context length | windows of 512 input positions | <= 512 input positions; longer sequences cut from the LEFT (context side; the continuation is never cut); n_trunc printed per benchmark (expected 0) | yes |
| position | random 512-token windows of the stream (+ arithmetic lines) | sequence starts at position 0 after `<eos>` | doc start in training occurs at any offset; position-0 start differs (RoPE, relative) — accepted |
| text distribution | FineWeb-Edu + 7% (505M / 53M / 190M ptlr) or 2% + domain lines (rb6) arithmetic | benchmark text (zero-shot, no few-shot) | generalisation test by design (zero-shot benchmark), the same for both arms |
| precision | bf16 autocast (training and the in-training eval) | fp32, TF32 off (more precise; the same for both arms) | no (deliberate) |
| model state | train mode | eval mode (no dropout in this config) | n/a |

Tokenisation of context + continuation (harness `_encode_pair`): trailing spaces of the context move to the continuation;
whole = enc(context + continuation), context_enc = enc(context), continuation_enc = whole[len(context_enc):].
Asserted: continuation_enc is non-empty and decodes back to the continuation text.

## Pre-registered comparisons
PRIMARY (verdicts), each benchmark, seeds 0-3:
- P1 505M 7%: win-ctx@1.2e-3 vs none@1.2e-3 (ptlr; both code a3bce3ab63c1).
- P2 505M 7%: win-ctx@3e-4 vs none@3e-4 (ptsize999; both code 79e9b410dcc3).
- P3 190M rainbow: all6-lam1@6e-4 vs none@6e-4 (rb6_190m).
DESCRIPTIVE (same statistic printed, labelled descriptive, no verdict used in any claim):
- D1 505M gate-ctx@3e-4 vs none@3e-4; D2 505M gate-text@3e-4 vs none@3e-4; D3 190M only-numbers@6e-4 vs none@6e-4.
- REF: every prior arm above minus the reference none of its group (paired by seed 0-3; lr differs, so descriptive).
- Reference = the group's none cell with the lowest mean bpb over seeds 0-3: offline full-val `bpb_full` (offline_lm.json)
  when it exists for every seed of every none cell of the group, otherwise the run's final logged `val_bits_per_byte`
  (in-training, 65k tokens); the source used is printed.
- none lr sweep tables (505M: 5 lrs; 190M: 2 lrs), extra seeds 4-7 (means only), 53M / 190M 7% seed-0 tables (prior -
  none at the same lr where a same-lr none exists; single models, no CI).

## Statistic and verdicts
- d_s = score(prior, seed s) - score(none, seed s), in percentage points, s = 0..3 (the common seeds; a verdict needs n = 4,
  else INCOMPLETE). Mean, SD, 95% t-CI = mean +- t(0.975, n-1) * SD / sqrt(n) (t = 3.182 for n = 4), wins = #(d_s > 0)
  (ties printed). n = 4 -> descriptive power only; no multiplicity correction and no multiplicity-corrected claims.
- Non-inferiority margins (fixed now): BLiMP 1.0, LAMBADA acc 1.0, HellaSwag acc_norm 1.0, ARC-E acc 1.0, PIQA acc 1.0
  points.
- Verdict per (comparison, benchmark), in this order:
  1. "at chance: uninformative" if both arms' means are within 2 points of chance (no verdict);
  2. "non-inferior" if CI lower bound >= -margin;
  3. "worse" if CI upper bound < 0 and mean < -margin;
  4. else "undetermined".
  (A CI entirely below 0 but above -margin is "non-inferior"; the read-out flags it as "small cost within margin".)
- LAMBADA ppl: descriptive per-seed ratio prior / none, geometric mean and 95% t-CI of the log ratio; no verdict.
- Read-out: tools/downstream_readout.py (selftest on fake inputs must pass first) -> runs/downstream/readout.{md,json}.

## Expected magnitudes (sanity, not hypotheses)
Models are 53M-505M trained on 3e8 tokens: HellaSwag / ARC-E / PIQA may be close to chance (then "uninformative" is the
expected outcome, not a failure); BLiMP well above 50 (~65-75); LAMBADA acc a few to ~20%. Outside [chance - 3, 100] on
any metric -> stop and inspect.

## Known-answer checks (before the real run; pass criteria fixed now)
(i) random-init models (torch seed 0; the 53M / 190M / 505M architectures from the run args): HellaSwag acc_norm, ARC-E
    acc, PIQA acc within +-3 points of chance; BLiMP on pairs with equal token counts within +-3 of 50 (all-pairs BLiMP
    printed; a random model prefers fewer tokens, so all-pairs may deviate by length: printed next to the shorter-sentence
    heuristic); LAMBADA acc <= 1%.
(ii) public GPT-2 small (HF `gpt2`, 124M), scorer in lm-eval-harness prefix mode (EOT only when the context is empty,
    1024 positions): LAMBADA acc within +-1.0 point of 32.56 and ppl within +-5% of 40.06 (lm-eval-harness, PR #2056
    results table: acc 0.3256 +- 0.0065, perplexity 40.0554); BLiMP within [81, 87] (Warstadt et al. 2020 §6.3: GPT-2
    117M-1558M "overall BLiMP accuracy of 0.84 +- .01"; Table 3 GPT-2-large 83.0). HellaSwag / ARC-E / PIQA of GPT-2 and
    the EOS-prefix mode: printed, descriptive (no verified reference used).
(iii) 2 decoded examples per benchmark (tokenised then decoded, the scored continuation in ⟦ ⟧, token indices, length)
    and 2 decoded val.bin document starts (the `<eos>` convention) printed at the start of every job (audit item vii).
Double path (audit i, every model): 24 random items per benchmark rescored one sequence at a time with full logits; the
summed log-likelihoods must match the batched path within 1e-2 nats and the greedy flags exactly (assert; crash on fail).
Load check (every model): nats/token on the first 16 non-overlapping 512-token val.bin windows, stored; assert < the
run's final logged val_loss + 0.5 (a wrongly loaded model would be ~11).

## Audit status
Seven items: i double path (in code), ii asserts (in code), iii known answers (above), iv only for anomalous results,
v fresh general-purpose sub-agent after submission (background), vi trap list in the review, vii decoded inputs in logs.

## Amendment A1 (2026-10-22 ~22:00 EDT; BEFORE any project-model benchmark score was computed)
Trigger: the pre-submission adversarial audit (fresh general-purpose sub-agent; CPU checks on random-init models and GPT-2
only, no project model scored). It found no blocking bug in the scoring code. These changes come from that audit:
1. **PIQA random-init criterion**: the audit measured (full PIQA, torch seed 0, CPU) 53M 52.34 / **190M 52.99** / 505M 52.61
   against the ±3 limit of 53.00. With summed logliks a near-uniform model prefers the choice with fewer tokens (the
   fewest-tokens heuristic alone scores 53.92). That is length bias, not label leakage. The criterion is now: PIQA acc on
   the pairs whose two solutions have the SAME number of scored tokens (audit: 781 of 1838) within ±tol of 50. Every
   random-init criterion uses tol = max(3 points, 3 binomial SE at chance on its n); for the other benchmarks this equals
   the original ±3 except BLiMP equal-length pairs if n < 2,500. All-pairs PIQA is printed next to the fewest-tokens
   heuristic (descriptive). The change was made after seeing the auditor's random-init PIQA values, so it is labelled
   post-hoc for this known-answer check. No project score had been seen.
2. **BLiMP definition**: follows the task's definition (whole sentence, no leading space, strict >; a tie counts wrong).
   lm-eval-harness scores " " + sentence and counts a tie correct, so "harness definitions" does not apply to BLiMP.
   Audit, GPT-2, 1/20 subset: 83.34 without the space vs 82.93 with it; both inside [81, 87]. A paired difference is
   unaffected (same definition for both arms).
3. **Double path**: the greedy flags may differ on ≤ 1 of 24 items (an argmax near-tie under different kernel shapes);
   the loglik tolerance stays 1e-2 nats. The decode-back check (continuation tokens decode to the continuation) is now an
   assert (≤ 0.1%; audit: 0 on the full sets).
4. **Load check** (replaces "nats/token < logged + 0.5"): an exact replica of the trainer's in-training
   val_bits_per_byte (np.random.default_rng(7), 8 × 16 windows of seq + 1, eval mode, bf16 autocast), asserted
   |replica − logged final| < 5e-3 bpb per model and flagged in the read-out if > 2e-3. The KA job runs it first on one
   model per training-code version (ptlr a3bce3ab63c1 505M and 53M, ptsize999 79e9b410dcc3, pt999 79e9b410dcc3, rb6_190m
   93e22bf4c4c0). It computes the LM metric only, no benchmark, and the KA gate requires < 2e-3 (CPU smoke: 53M ptlr
   3.4e-7).
5. **KA gate bound to the code**: the gate counts only full-run KA files (limit 0) whose eval_code_sha equals the
   snapshot's downstream_eval.py. Group jobs run the gate first (exit 5 on a fail) and are submitted with afterok on the KA
   job. KA job 20665519 was generated by the old generator; its pinned shas no longer match the snapshot, so it exits at
   start (exit 3, by design). It is replaced by a new KA job.
6. **Precision of the 4-seed verdicts**: n = 4 with a 1-point margin will usually give "undetermined" on PIQA / ARC-E (and
   maybe BLiMP). The read-out prints that "undetermined" ≠ "no cost". Added, descriptive only: a seed × item bootstrap
   95% percentile interval of the mean paired difference (resample the common seeds and the items with replacement,
   2000 draws, numpy rng 0), from the per-item hit strings. It is cross-checked against the stored scores (assert) and
   is not a verdict.
7. **53M rows** of the read-out show BLiMP and LAMBADA only, as the task specifies (HellaSwag / ARC-E / PIQA for 190M /
   505M). The JSON keeps every score. Seed-0 rows pair prior and none only at the same lr.
8. **Anomaly flags** (automated): any primary metric < chance − 3, any double-path greedy mismatch, any load-check
   |Δ| > 2e-3 is listed in the read-out.
9. **Position wording**: the models do encode absolute position. In the 7% models, positions 464–511 always hold
   arithmetic lines in training, and the audit measured text NLL there +0.33 nats. Eval sequences are ≤ 245 tokens, so
   they never reach those positions. The position-0 / `<eos>` start costs +0.06–0.09 nats on a document's first 5 tokens
   and ≈ 0 from token 16 on (audit, val documents), the same for both arms.
10. **Ops**: the generator checks for a dirty tree before touching the snapshot; `--open-mode=append` (a requeue keeps
    the first log); scoring batches 65,536 tokens (GPU utilisation).
Also from the audit, for the record: 13-gram overlap of the benchmarks with train.bin is negligible (LAMBADA 0.04%,
HellaSwag 0.13%, ARC-E 0.36%, PIQA 0.07%). Seeds 0–3 of each primary pair share the step-0 LM loss bit for bit (same
init, same first batch), and each pair shares its training code sha.
Nothing above "Amendment A1" was changed.
