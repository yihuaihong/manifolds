# ptheld2 — wall 1 follow-up: H stratified by value x text frequency, the prior on to the end (2026-10-21)

Context: F90 (runs/ptheld, 53M, 7%, H = 50 three-digit numbers by text-frequency quintile): wall 1 does not hold at this
scale (none answers ~21% of the H_one rows); the prior arms reach 0.60 (text) / 0.94 (emb) while the window is on and
fall to ~0.39 after it closes (win-ctx, which cannot reach H, 0.31). POST HOC (after R4 was read; motivates this batch,
confirms nothing): none's failure tracks h's VALUE (1xx 0.57 -> 9xx 0.04); frequency and value are confounded (rho -0.83);
on the rarest stratum every none config is <= 0.05. F90's own recommendation: stratify H by value, draw by frequency
inside the value layer, make the rare layer's final value the primary metric, run none at the same budget and a window
that stays on to the end / fades out. Written before any run (runs/ptheld2/out did not exist); the 52 main jobs were
submitted on 2026-10-21 by the coordinator after the interim report; amendments SUPP and AUD were added after the
submission and before any result existed.

Question. Same model, data and arithmetic as ptheld, but H = 54 numbers, 6 per hundreds block, half rare / half common
inside the block. On the layers where none is expected at the floor (large values: blocks 7-9; rare numbers inside every
block), does a helix3 prior that reaches H and stays on until the end of training (lambda 1; or fading linearly to 0)
make h + y answerable, beyond none and beyond a prior that never gives h a value label (win-ctx-always)?

## Code and provenance
- Trainer: llmadd-burst `tools/pretrain_range.py` Rev D, commit f6e9a22 on branch `ptheld2-build` (worktree
  /scratch/yh6210/llmadd-burst-p2, from origin/main 6e827c6; not pushed), code_sha 93e22bf4c4c0. Rev D = two opt-in flags
  (`--held_strat value_freq`, `--lam_decay`) and, with either flag, extra logs (lam_now, grad_total, clip_coef,
  grad_lm_share, grad_T; docstring block "Rev D"). Without them the trainer is Rev C1 3c99f6f8cb17 (= runs/ptheld/code):
  CPU regression in Tests.
- Job code = snapshot `runs/ptheld2/code/tools/` (git archive of tools/ at the committed HEAD f6e9a22); `code/COMMIT` =
  HEAD + sha256[:12] of the trainer and every module it imports (pretrain_helix 25d5a35e6f17, numlib d09bb47b04c2,
  digit_split_data 742b922c8125, run_guard b944dc859c7a = the ptheld snapshot's). Generator
  `llmadd_torch/make_jobs_ptheld2.py`: refuses to render when the worktree is not on ptheld2-build, tools/ has uncommitted
  changes, or the snapshot's trainer / any import differs from HEAD's; asserts every cell's flags (intended values present,
  no other flag, no flag twice, lambda = 1 for the prior arms); every sbatch re-checks the trainer's sha256 at start (exit 3)
  and honours runs/ptheld2/HOLD; SUBMITTED_<job> markers; `--print` lists every job's ARGS; nothing is submitted without
  `--submit`.
- Read-out: llmadd-burst `tools/ptheld2_summary.py` (same branch), written and tested on FAKE results before any result
  (`--selftest`). The rules below are its rules; where this file and the code disagree, that is a bug to fix before reading.

## Setting (= ptheld, except H, the window, lambda and the decay)
53M (d384 x 6), seq 512, bs 32, 3e8 tokens = 18,310 steps (4x: 73,242), 1% warmup, cosine to 0.1x, AdamW (0.9, 0.95), wd
0.1, clip 1.0, bf16 autocast, hidden-state sites at layers 2 and 4, --arith_lines 4 (7%), eval every 250 steps + the last
(75 evals; 4x 294), L40S (rescue criterion (f)). Prior arms: helix3, `--schedule window --window 0.05,1.0` = prior on at
steps 916..18309 (from 5% to the end; prior_on_frac 17394 / 18310, prior_events [[916, 1]]), lambda 1 (`--lam 1`; F82 M7:
lambda 1 better than 5 at 7%; the user's manifold-as-auxiliary-loss rule), lr 2e-3. No `--lam_rule` (a --domains flag;
not used on this path). Same seed = same init, same H, same remap / far permutation (logged, unused), same data stream for
every arm (paired by seed; checked by held_data_sha).
- `--lam_decay` (win-emb-decay only): lambda_t = 1 x (1 - (s - 916) / 17394) at the on steps: 1.0 at step 916, 0.636 at
  7250 (ptheld's window end), 0.0178 / 0.0035 / 0.00006 at the final3 steps 18000 / 18250 / 18309 (amendment AUD M2). Logged as lam_now at every
  eval and asserted in-run against the formula.

## H (--held_operands 54 --held_strat value_freq)
- 6 per hundreds block (100-199 = block 1 .. 900-999 = block 9). Inside a block the 100 numbers are ranked by their
  text-site count over all of train.bin (the Rev C count, the text site's own rule; ties by value), split into the lower 50
  ("rare") and upper 50 ("common"); 3 are drawn from each half without replacement, held stream 6000 + seed, blocks
  ascending, rare then common. Asserted in-run: 54 h, 3 + 3 per block, h // 100 = block, max count of a rare half <= min
  of its common half. Logged per h: `held_operands.block`, `.half`, `.count`; per block half the count range
  (`half_count_edges`); rows per layer (`layer_rows`).
- Count ranges per block half (seed-independent; rare / common): b1 506-893 / 918-50,548; b2 291-486 / 490-18,269; b3
  172-321 / 327-13,032; b4 118-210 / 213-9,974; b5 75-143 / 145-13,771; b6 53-107 / 108-6,129; b7 41-96 / 96-4,157; b8
  36-80 / 81-5,401; b9 39-73 / 74-2,652. So "rare" is a within-block notion: a rare 1xx (>= 506) is commoner than a common
  9xx (>= 74). Frequency is still partly confounded with value ACROSS blocks; inside a block the halves differ by count only.
- Seeds 0-7 (independent re-implementation of the draw from the logged counts, scratch hcheck.py): every seed 54 h, 3 + 3
  per block; median count of the rare h 101-134, of the common h 182-273; H overlap between two seeds 1-6 of 54 (mean 3.8).
- Every other Rev C guarantee is unchanged (H never an operand or an answer token of a training line; clean pairs redrawn
  within the bucket; content scan of every training row every step; dry scan of 50 batches; the held / seen eval sets keep
  clean pairs only; held_data_sha). With 6 h in 100..199 per seed, 60 of the sums 1000..1998 never occur in training (their
  first token is an h) vs 40-60 in ptheld.

## Arms (52 runs; all with --arith_lines 4 --held_operands 54 --held_strat value_freq --save_model --ckpt_every 2000)
| tag (53m-lr<lr>-7pct-h54vf-...) | args beyond the base | seeds | reaches H? | role |
|---|---|---|---|---|
| none @ 2e-3 | --target none --lr 2e-3 | 0-7 | - | baseline (prior arms' lr) |
| none @ 8e-3 | --target none --lr 8e-3 | 0-7 | - | baseline at its best 7% lr (ptlr) |
| none @ 8e-3, 4x | --target none --lr 8e-3 --tokens 1.2e9 | 0-3 | - | budget cell (rescue (a)) |
| win-emb-always | --target helix3 --schedule window --window 0.05,1.0 --lam 1 --site emb --lr 2e-3 | 0-7 | all 1000 rows, every step | test |
| win-emb-decay | the same + --lam_decay | 0-7 | the same, weight 1 -> 0 | test (does a faded prior leave H usable?) |
| win-text-always | the same, --site text (no decay) | 0-7 | through the text, when h occurs | test |
| win-ctx-always | the same, --site ctx | 0-7 | no value label for h (target: seen operands; its gradient reaches E[h] through attention) | non-specific control |
- emb site: lambda (1 - CKA(W_E[number tokens 0..999], K_helix3)); tied embeddings, so it also shapes the output rows of
  H (relevant to the answer classifier; F90 wording caveat "injecting h's value coordinates substitutes for task exposure").

## Metrics (run-JSON keys)
- H_one: 648 rows per seed = 12 per h (partner 1 / 2 / 3 digits x 2 draws x h first / second), every answer token seen;
  exact match of all answer tokens and the newline, teacher forced, format 0 at the end of a 512-token row (= ptheld).
- Layers (masks of the SAME rows, no extra forward; `held_operands.sets.H_one.masks`): v1..v9 (block; 72 rows each), rare
  / common (324 each), **hi = blocks 7-9 (18 h, 216 rows)**, **lo_rare = the rare halves of all blocks (27 h, 324 rows;
  identical rows to `rare`)**. Per eval: accuracy `H_one_<layer>`, mean answer log-prob `logp_H_one_<layer>`; per row
  `hits_H_one`, `logp_rows_H_one`; plus everything ptheld logs (q1..q5 = the global frequency quintile of h, p1..p3,
  hfirst, H_both, H_ans, held / seen sets, the logp double path).
- acc final3 = mean of the last three evals (1x: 18000 / 18250 / 18309; 4x: 72750 / 73000 / 73241). held6 = mean of the six
  filtered held_bXY (seen operands, held-out pairs). bpb = val_bits_per_byte final3.
- Gradient split (Rev D; manifold_aux_loss_principle): at every eval with the prior term in the loss, before the clip,
  grad_lm = |g_LM|, grad_prior = |lambda_t g_prior|, grad_cos, grad_T = |g_LM + g_prior|, grad_lm_share = grad_lm / grad_T (GLOBAL norms, descriptive only: amendment AUD I1)
  (the rb6 convention; > 1 when the prior opposes the LM); at every eval grad_total = the pre-clip norm clip_grad_norm_
  returned for that step's update (double path of grad_T: grad_T_reldiff), clip_coef = min(1, 1 / (grad_total + 1e-6)),
  lam_now (0 when no prior term), grad_lm_share = 1.0 when the loss is the LM alone. (grad_lm / grad_prior / grad_cos
  already existed at the on-evals, pretrain_range.py Rev C; reused.)

## PRE-REGISTERED rules (written 2026-10-21 before any run; the user fixed the co-primary layers the same day, before any run)
Units: seed = the unit of replication; "3/4" = at least ceil(3/4 n); selection seeds 0-3, confirm seeds 4-7.
**Co-primary layers** (user decision 2026-10-21): A = H_one_hi (blocks 7-9), B = H_one_lo_rare (the rare halves). Every
rule below runs separately on each layer; the two verdicts are printed separately and never merged.
**Validity** (a failure withholds every verdict that uses the cell): as ptheld_summary.check_run with amendment DP (logp
double path: step 0 <= 0.05 for H_one / H_both / H_ans; later evals <= 1.0 / 1.5 / 1.5), plus: code_sha 93e22bf4c4c0;
base args (above) with held_operands 54, held_strat value_freq, lambda 1, window 0.05,1.0 for the prior arms; steps
18,310 (4x 73,242); 75 evals (4x 294); held_lines_scanned = 128 x (step + 1); held_rej > 0; prior arms prior_on_frac
17394 / 18310 and prior_events [[916, 1]]; none never on; emb arms grad_prior_emb_rows = 1000 wherever logged; H: 54, 3 + 3
per block, block = h // 100; layer_rows hi 216, lo_rare 324; H_one_<layer> and logp_H_one_<layer> present at every eval;
lam_now at every eval (prior arms with the prior on: 1.0, decay: the formula, both within 1e-9, text site also 0 when the
batch had < 16 numbers; prior off and none: 0); H_one_hi / H_one_lo_rare of the final3 evals = the mean of hits_H_one on
the layer's mask (double path); grad_total, clip_coef,
grad_lm_share at every eval; per seed across arms: H, counts, H_one pairs equal, step-0 lm within 1e-3, held_data_sha
equal at every common eval step (also 1x vs 4x).
**Informative seed** (as ptheld): held6 final3 >= 0.80, not diverged (readout_rules.diverged_at), not unstable (final
val_loss <= the same seed's none@2e-3 + 0.5 nats). A control seed that is not informative counts neither way; a prior-arm
seed that fails the held6 / divergence gate counts as a failure of (b) and keeps its measured value in the paired tests.
**Selection** (per layer L, seeds 0-3 only): among win-emb-always, win-emb-decay, win-text-always the arm with the highest
mean acc_L final3 over seeds 0-3; arms within 0.02 of the top -> the lower mean bpb over seeds 0-3. Confirmation runs on
seeds 4-7 of that arm only (the selection seeds are never used for the verdict).
**Rescue reading** (per layer L, on the confirm seeds; report/audits/rescue_standard.md, tier L):
- (a) floor: for every none config (none@2e-3 and none@8e-3: seeds 4-7; none@8e-3-4x: its seeds 0-3, the only ones it has,
  not used by the selection), >= ceil(3/4 n) of its seeds are informative AND have acc_L final3 <= 0.10. Fewer -> "none
  not at the floor" (or "uninformative") -> no rescue.
- (b) the chosen arm: acc_L final3 >= 0.50 in >= 3/4 of seeds 4-7.
- (c) paired, seeds 4-7, chosen arm minus each of none@2e-3, none@8e-3, win-ctx-always (over the seeds where the control
  is informative, >= 3): wins >= ceil(3/4 m) and the one-sided paired t; p_L = the largest of the three p
  (intersection-union). Specificity: chosen - win-ctx-always paired mean >= +0.30.
- Holm over the two layers: the layer with the smaller p_L needs p_L < .0125, the other < .025; if the first fails, the
  second fails too.
- Verdict per layer (superseded by amendments SUPP and AUD below: RESCUE also needs rescue_standard (a) and (d) met, else
  at most PROVISIONAL): RESCUE (tier L) iff (a), (b), (c) with its Holm threshold and specificity all hold. A rescue claim
  additionally needs the rescue_standard criteria this batch does not run (see Known limits: (a) lr x wd bracket of none,
  (b) the prior arm's own lr sweep, (d) a shuffled-target arm); the read-out prints them as NOT RUN next to the verdict.
  Otherwise the verdict names what failed: "none not at the floor", "prior arm < 0.50", "not specific (vs win-ctx-always
  < +0.30)", "paired tests fail", "withheld (invalid / missing runs)".
- Beside the verdict (no claim): held6 final3 of the chosen arm (seen addition preserved iff >= 0.80 in every confirm
  seed), its bpb minus none@2e-3 (paired), the window-median grad_lm_share and clip_coef of every prior arm (descriptive only, amendment AUD I1; originally: FLAG when a
  seed's median share < 0.5, i.e. the prior gradient exceeds the LM's).
**Descriptive** (no claim): every arm's acc_H1 (full H), every layer (v1..v9, rare, common, hi, lo_rare) final3 per arm
over seeds 0-3 / 4-7 / 0-7; acc at step 7250 (ptheld's window end) vs the final eval vs final3 (for window 0.05,1.0 "the
window end" is the final eval); logp per layer; the selection on seeds 4-7 alone, printed only to show whether it would
pick a different arm (never used).

## Predictions / expected ranges (outside -> check the run before using it)
| cell | acc hi final3 | acc lo_rare final3 | basis |
|---|---|---|---|
| none @ 2e-3 / 8e-3 / 4x | 0-0.10 | 0.05-0.30 (the rare 1xx-3xx are easy: F90 1xx 0.57) | F90 per value; lo_rare spans all blocks, so none may NOT be at the floor on B |
| win-ctx-always | 0.05-0.3 | 0.1-0.4 | F90 win-ctx 0.31 overall (non-specific gain) |
| win-emb-always | 0.3-0.95 | 0.3-0.95 | F90: 0.94 while the window was on (lambda 5); lambda 1 is weaker |
| win-emb-decay | 0.05-0.5 | 0.1-0.5 | lambda ~0.007 at final3: may fall back as after ptheld's window |
| win-text-always | 0.05-0.6 | 0.1-0.6 | rare h are rarely in a batch (hi-rare: 36-96 text-site / 283-481 raw occurrences in train.bin) |
Seen: held6 final3 0.94-1.0 every arm; bpb 1.20-1.26; grad_lm_share at lambda 1: emb ~0.9-1.0 (smoke: 0.98 at lambda 0.7).

## Train vs eval conditions (= ptheld)
| dimension | train | eval | same? |
|---|---|---|---|
| position | 4 lines at the end of a 512-token row after FineWeb text | one line at the row end after val text | yes |
| template | 4 formats, random | format 0 | subset |
| values | operands 0..999 except H; answers without H tokens | H_one: one operand in H, answer tokens seen | intended generalisation test |
| data | train.bin text; H never in a line | val.bin text | disjoint lines by construction |
| model state | train, bf16 autocast | eval mode, bf16 autocast | yes |

## Known limits / deviations from rescue_standard.md
- (a) none: lr 2e-3 / 8e-3 + a 4x cell at 8e-3; no wd axis and no lr above 8e-3 on this H (ptheld's LC bracket was on the
  quintile H: none@1.6e-2 0.12, wd 0.01 0.17). The none optimum on the new layers is not bracketed by the 52 main runs;
  the bracket cells are amendment SUPP (i) (rendered, awaiting the user's approval).
- (b) the prior arms run at one lr (2e-3) and one lambda (1): no own lr sweep (not in SUPP either).
- (d) no shuffled-target arm among the 52 (ptheld's farshuf control broke seen addition: held6 0.59); the remapped-target
  control of the emb arms is amendment SUPP (ii) (rendered, awaiting approval); none exists for the text arm.
- (c) tier L at most (helix gives the values, not the sums).
- lo_rare = rare (same 324 rows): B mixes all value blocks; the rare 1xx-3xx are frequent in absolute terms (>= 172).
- final3 of win-emb-decay is read at lambda ~0.007: it measures what is left after the fade, not the prior at strength.
- The text prior acts on h only in batches where h occurs; hi-rare h occur 36-96 times in all of train.bin as standalone text-site numbers (283-481 raw token occurrences).

## Tests (CPU, 2026-10-21, before submission; scripts / logs in the session scratchpad ptheld2/)
Tiny model for training tests (d64 x 3 layers, seq 128, bs 8, 24 steps, evals at steps 0 / 8 / 16 / 23, n_eval 20);
--dry_inputs also at the production shape (53M, seq 512, bs 32, the rendered ARGS unchanged except --out).
1. **Regression** (reg.sh + compare.py; old = origin/main tools/ = trainer 3c99f6f8cb17 = runs/ptheld/code, new = Rev D
   93e22bf4c4c0; both one thread, same host, side by side): run JSON equal except code_sha / sec / out dir, every eval value
   float-equal, saved model bitwise, stdout equal except the code line, --dry_inputs stdout byte-identical. Configs: r1 none;
   r2 helix3 window ctx; r3 helix3 gate text lambda 1; r4 helix3_shuf prefix; h1 --held_operands 50 none lr 8e-3; h2
   --held_operands 50 helix3 window emb (grad split + emb-rows branch); h4 --held_operands 50 helix3 window text; r5
   --domains numbers,months; h3 --held_operands 50 helix3_hremap emb. For the --held_operands configs the wall-clock fields
   (held_operands.count_sec, the seconds on the "held operands:" / "counted in" lines) are masked (cmp_ho.py, as the Rev C1
   tests did with cmp_common.py). RESULTS: see "Regression results" below.
2. **A/B of --lam_decay** (abrun.sh / ab.py; value_freq, emb site, window 0.05,1.0, lambda 1, seed 0, with vs without the
   flag): see below.
3. **H construction**: in-run asserts (54 h, 3 + 3 per block, h // 100 = block, rare max <= common min, every layer = the rows
   of its h, blocks partition H_one, hi = rows with h >= 700, lo_rare = rare); independent re-implementation of the draw
   from the logged counts (hcheck.py): seeds 0-7 all PASS and seed 0 = the trainer's H / block / half / count.
   Content scan at the production shape (50 batches, 6,400 lines, every cell type): 0 H tokens in the lines; 290 H tokens
   in the text (120 text-site occurrences, 15 / 54 h seen; win-text: 298 / 123 / 21).
4. **Production-shape --dry_inputs** (proddry.sh, seed 0 of all 7 cell types, snapshot trainer): all clean (exit 0). Decoded
   (experiment-audit vii): three training rows with the prior sites in ⟦ ⟧ (ctx: operands; text: numbers in the text; emb:
   no positions), two rows of every H set with H tokens ⟦ ⟧ and the scored span « » in the eval position, H per h with
   block / half / count / u_h, the content scan. The training rows are the same across the arms of a seed (same data
   stream). Emb site 1 - CKA at the 53M init 0.948; text site: median 123 numbers per batch, prior applicable on 50 / 50
   batches. lam_decay prints "lambda 1.0 x (1 - (s - 916) / 17394) at the window steps 916..18309". Eval rows kept (seed
   0): held_b33 113 / 150, b23 125, b13 137; rehearsal pool 525 -> 490; remap candidates 244-319 per h.
5. **Smoke training** of the new path (value_freq + lam_decay, emb, tiny): every Rev D key at every eval; lam_now 0 (step 0,
   prior off) / 0.727 / 0.364 / 0.045 = 1 - (s - 2) / 22; grad_T vs grad_total rel. diff 2e-5; grad_lm_share 0.977 -> 1.000;
   clip_coef 0.57 -> 0.92; grad_prior_emb_rows 1000; layer rows v1..v9 72, rare / common 324, hi 216, lo_rare 324.
   ptheld2_summary.check_run on this real output (tiny constants patched in, keycheck.py): no complaint except the tiny
   shape -> the read-out's key names = the trainer's.
6. **Generator** (scratch copies, gen/): 52 jobs, 70.0 L40S-h, bash -n clean for all 52; refused: win-emb-always rendered
   with --site text, a repeated --lr, --lam 5, win-emb-decay without --lam_decay; no SUBMITTED flag written.
7. **Read-out** (tools/ptheld2_summary.py, `--selftest` on the smoke run JSON; 17 fake scenarios, all PASS): S1 the emb
   prior works on both layers -> RESCUE on A and on B (separate verdicts, the rescue_standard criteria not run listed);
   S2 none@2e-3 at ~0.25 -> "(a) none not at the floor"; S3 win-ctx-always at ~0.65 -> "not specific" (+0.18 < +0.30);
   S4a an invalid run (held_lines_scanned) -> WITHHELD; S4b a missing run -> INCOMPLETE; S4c a wrong consumed lambda ->
   WITHHELD; S4d a logged layer accuracy that does not match its rows' hits -> WITHHELD; S5 seeds 0-3 pick win-text-always,
   seeds 4-7 would pick win-emb-always -> the confirmation uses win-text-always and fails (b) (the 4-7 pick is printed as
   "not used"); S6 layer A rescues, layer B does not (none's rare 1xx-6xx at 0.35 -> B not at the floor), two separate
   verdicts; S7 two arms within 0.02 -> the lower bpb is chosen; S8 two none@8e-3 confirm seeds never learn seen addition ->
   "(a) uninformative" (no vacuous floor); S9 a prior arm with LM share 0.4 -> printed (FLAG removed by amendment AUD); amendment SUPP: S10 LC at the
   floor, bracket closed, hremap fails -> RESCUE with "(a) met, (d) met, (b) NOT RUN"; S11 none@1.6e-2 at 0.3 -> "no
   rescue (ptheld2 criteria hold, rescue_standard not met)"; S12 hremap as good as helix3 -> (d) not met; S13 none@1.6e-2
   better by 1 nat on logp (acc at the floor) -> "lr axis OPEN"; S14 win-text-always chosen -> (d) NOT RUN (no control at
   the text site). Double path on real trainer output: check_run on the two tiny Rev D runs (tiny constants patched in,
   keycheck.py): no complaint beyond the tiny shape; the layer double path (H_one_<L> = mean of the rows' hits on the
   layer mask) holds on real output.
8. **Remap feasibility, seeds 0-7** (remapcheck.py, re-implementation of H, the H_one pairs and the remap from the logged
   counts): seed 0 = the trainer's H / H_one pairs / remap_H; for every seed the smallest unused candidate pool at a draw is
   229-259 (the trainer asserts >= 100 at the start of EVERY H run, so no seed of the submitted batch fails there).

### Regression results (old 3c99f6f8cb17 vs Rev D 93e22bf4c4c0)
All 9 configs IDENTICAL: r1 none, r2 window ctx, r3 gate text, r4 prefix shuf, r5 --domains numbers,months (compare.py:
run JSON every value float-equal, models 36 / 36 tensors bitwise, stdout equal except the code line, --dry_inputs stdout
byte-identical 9 / 10 / 12 / 11 / 31 lines); h1 held none, h2 held window emb, h4 held window text, h3 held hremap emb
(cmp_ho.py: the same with held_operands.count_sec and the count's seconds on two stdout lines masked; models 36 / 36
bitwise; every curve value equal; count_sec e.g. 97.2 vs 98.9 s = wall clock only).

### A/B --lam_decay (tiny, value_freq, emb, window 0.05,1.0, lambda 1, seed 0)
Models 36 / 36 tensors differ; step 0 identical (prior off, lam_now 0 / 0); lam_now 1.0 / 0.727 / 0.364 / 0.045 (always:
1.0 / 1.0 / 1.0); grad_prior 0.299 / 0.161 / 0.142 (always) vs 0.218 / 0.060 / 0.007 (decay); grad_lm_share 0.959 -> 0.992
(always) vs 0.977 -> 1.000 (decay); CKA(embedding rows, helix3) at step 23 0.819 (always) vs 0.793 (decay).

## Jobs (rendered by make_jobs_ptheld2.py; L40S, gpu48; planned 1.1 L40S-h per 1x run, 4.3 per 4x run)
| group | tag | runs | limit | L40S-h | state |
|---|---|---|---|---|---|
| NONE | 53m-lr2e-3-7pct-h54vf-none | 8 | 03:00 | 8.8 | submitted 2026-10-21 by the coordinator (user approval as relayed) |
| NONE | 53m-lr8e-3-7pct-h54vf-none | 8 | 03:00 | 8.8 | submitted |
| NONE | 53m-lr8e-3-7pct-h54vf-none-4x | 4 | 09:00 | 17.2 | submitted |
| PRIOR | 53m-lr2e-3-7pct-h54vf-win-emb-always | 8 | 03:00 | 8.8 | submitted |
| PRIOR | 53m-lr2e-3-7pct-h54vf-win-emb-decay | 8 | 03:00 | 8.8 | submitted |
| PRIOR | 53m-lr2e-3-7pct-h54vf-win-text-always | 8 | 03:00 | 8.8 | submitted |
| CTRL | 53m-lr2e-3-7pct-h54vf-win-ctx-always | 8 | 03:00 | 8.8 | submitted |
| SUPP-A | 53m-lr1.6e-2-7pct-h54vf-none | 4 | 03:00 | 4.4 | rendered, NOT submitted |
| SUPP-A | 53m-lr8e-3-7pct-h54vf-none-wd0.01 | 4 | 03:00 | 4.4 | rendered, NOT submitted |
| SUPP-A | 53m-lr8e-3-7pct-h54vf-none-wd1.0 | 4 | 03:00 | 4.4 | rendered, NOT submitted |
| SUPP-D | 53m-lr2e-3-7pct-h54vf-win-emb-hremap-always | 8 | 03:00 | 8.8 | rendered, NOT submitted |
| SUPP2-B | 53m-lr{1e-3,4e-3}-7pct-h54vf-win-emb-always | 4 + 4 | 03:00 | 8.8 | rendered, NOT submitted |
| SUPP2-B | 53m-lr{1e-3,4e-3}-7pct-h54vf-win-emb-decay | 4 + 4 | 03:00 | 8.8 | rendered, NOT submitted |
| SUPP2-B | 53m-lr{1e-3,4e-3}-7pct-h54vf-win-text-always | 4 + 4 | 03:00 | 8.8 | rendered, NOT submitted |
| SUPP2-D | 53m-lr2e-3-7pct-h54vf-win-emb-hremap-decay | 8 | 03:00 | 8.8 | rendered, NOT submitted |
Main: 52 jobs, 70.0 L40S-h planned (measured in ptheld: 0.99 h per 1x run, 3.96 h per 4x run -> ~64 L40S-h). SUPP: 20 jobs,
22.0 L40S-h planned (~20 measured-equivalent). SUPP-2: 32 jobs, 35.2 L40S-h planned (~32 measured-equivalent). Main
submitted with `python make_jobs_ptheld2.py --submit` (52 SUBMITTED_ flags). The user's call, not run:
`cd /scratch/yh6210/llmadd_torch && python make_jobs_ptheld2.py --supp --submit` (SUPP) and `... python
make_jobs_ptheld2.py --supp2 --submit` (SUPP-2). Without --supp / --supp2 nothing of that group is rendered or submitted;
the 52 main and 20 SUPP sbatch files are byte-identical with and without --supp2 (md5 of all 72 checked).
Post-start checks (first evals): step-0 lm identical across the arms of a seed; held_operands.H and remap_H equal across
arms; held_data_sha equal at step 0; prior arms prior_events [[916, 1]]; lam_now 1.0 (decay at step 1000: 1 - 84 / 17394);
grad_lm_share and clip_coef at the first on-eval (step 1000).

## Amendment SUPP (2026-10-21, before any result of this batch; the coordinator's request, cells awaiting the user's approval)
- Cells (make_jobs_ptheld2.py --supp; same snapshot / code_sha 93e22bf4c4c0; only the listed flags differ from the main
  none / win-emb-always cells): (i) rescue_standard (a), none's lr x wd bracket on this H: none @ 1.6e-2, none @ 8e-3 wd 0.01,
  none @ 8e-3 wd 1.0, seeds 0-3; (ii) rescue_standard (d): win-emb-hremap-always = --target helix3_hremap --site emb, window
  0.05,1.0, lambda 1, lr 2e-3, seeds 0-7 (each h at the helix position of a seen u_h, the Rev C1 remap, drawn on stream
  6400 + seed; logged in every run as remap_H, the same for every arm of a seed).
- Rules (per layer, in ptheld2_summary.std_a / std_d; printed next to every verdict; "NOT RUN" while the cells are absent):
  (a) bracket: each LC cell passes the (a) floor rule on seeds 0-3 (>= 3 of 4 seeds informative and acc_L final3 <= 0.10),
  AND on the lr axis (none@2e-3 / 8e-3 / 1.6e-2) and the wd axis (none@8e-3 wd 0.01 / 0.1 / 1.0), seeds 0-3, no axis is
  OPEN: OPEN = the best cell (mean final3) is at an edge and beats its neighbour by > 0.25 nats on logp_H_one_<L> (or, when
  the best mean acc_L > 0.05, by > 0.02 on acc_L) with >= 3/4 paired wins (= ptheld R1's bracket margins).
  (d) only for an emb arm (win-emb-always / -decay) chosen on the layer: win-emb-hremap-always acc_L final3 < 0.50 in >= 3/4
  of seeds 4-7 AND chosen - hremap paired mean >= +0.20 with >= 3/4 wins (= ptheld R3b's margin); for win-text-always: NOT
  RUN (no shuffled control at the text site). Descriptive: the share of layer rows answered u_h + y (hremap, chosen,
  none@2e-3).
  (b) the prior arm's own lr sweep: not in this batch (NOT RUN).
  Verdict per layer: the ptheld2 criteria fail -> "no rescue: ..."; they hold but (a) or (d) is "not met" -> "no rescue
  (ptheld2 criteria hold, rescue_standard not met)"; otherwise "RESCUE (tier L)" with the state of (a), (b), (d) (met / NOT
  RUN / withheld) -- a rescue claim needs all three met.

## Amendment AUD (2026-10-21, after the fresh-agent build audit notes/batch1_build/audit_ptheld2_build.md, before any result; read-out and wording only -- trainer, snapshot and sbatch unchanged)
- I4 (verdict): RESCUE only when rescue_standard (a) and (d) are both met (the SUPP cells run and valid); SUPP runs invalid
  -> the whole verdict WITHHELD; SUPP not run / incomplete, or no control for the chosen arm (text) -> at most
  "PROVISIONAL (partial)". (b) is reported as its state (one prior lr, not swept), as rescue_standard (b) allows.
- I1 (gradient split): grad_lm_share = |g_LM| / |g_LM + g_prior| over ALL parameters is a global norm ratio. It cannot show
  whether the prior dominates the update of the 1000 number embedding rows (AdamW normalises per coordinate; the emb prior's
  gradient lives on those rows only), and for orthogonal gradients of equal norm it is 0.71, so no share threshold means
  "the prior exceeds the LM". It is a descriptive known limit, not a criterion; the read-out prints share, prior / LM norm
  ratio, the prior's norm on the embedding (grad_prior_emb) and clip_coef without any FLAG. (Not measured: the per-row
  LM vs prior update on W_E[0..999]; possible offline from a saved model.)
- I2 (B layer name): B = "rare by standalone occurrences in the text" (the text-site count), not by token exposure. In
  block 1 the two disagree (Spearman(text-site count, raw token count) = -0.18 over 100..199; 4-digit years are split as
  "17x" + "d"; blocks 2-9: +0.80..+0.94, raw ~3-5x the text-site count: the hi-rare h have 283-481 raw occurrences, 36-96
  text-site ones). Pre-registered descriptive secondary reading: B' = B without block 1 (24 h, 288 rows), printed per arm;
  the primary test is unchanged.
- I3 (A layer): A mixes large values with two-token answers (seed 0: rows with sum >= 1000 are 6% of block 1, 47% of block 9,
  35% of A, 22% of B). Pre-registered descriptive split: every layer (A, B, B', all) per arm by sum < 1000 / >= 1000, next to
  the seen references held_b33 / held_b23 lt1000 / ge1000. Wording: an A-layer result is about "large h AND (for 35% of the
  rows) a two-token answer", not value alone.
- M1: win-ctx-always "never gives h a value label" (its target Gram has seen operands only); its gradient still reaches E[h]
  through attention when h is in the row's text. Wording changed from "cannot reach H".
- M2: final3 = the last three evals = steps 18000 / 18250 / 18309 (1x; 18250 is an eval step), 72750 / 73000 / 73241 (4x).
  win-emb-decay's lambda there: 0.0178 / 0.0035 / 0.00006 (mean 0.007), not "~0.02".
- M3: the natural text contains arithmetic with h ("h + y", "= h"; every arm sees it): per number the count of the token
  next to "+" (optionally across a space) or right after "=" over all of train.bin (tools/ptheld2_textarith.py ->
  runs/ptheld2/textarith.json; total 66,305; per-block medians b1 27, b2 15, b3 9, b4 5, b5-b6 4, b7 3, b8 2, b9 3; seed 0:
  A total 62 / median 3, B total 197 / median 6; = the audit's own count). Printed per seed and layer, descriptive.
- M5: none@2e-3 seeds 0-3 are now required by the confirmation reading (the 4x cell's unstable gate) and a missing reference
  makes a seed uninformative, never a silent pass. M6: check_run re-derives the hi / lo_rare masks from the H_one pairs and
  each h's block / half and withholds the run on any mismatch (besides the hits double path).
- Selftest after the amendment: 21 scenarios PASS (written "22" before; the read-out audit counted 21) (S1 / S6 / S7 / S9 now PROVISIONAL without SUPP; S10 RESCUE with SUPP
  complete and met; S14 text chosen -> PROVISIONAL; S15 SUPP invalid -> WITHHELD; S16 SUPP incomplete -> PROVISIONAL; S17
  none@2e-3 seed missing -> INCOMPLETE; S18 a tampered layer mask -> WITHHELD).

## Amendment AUD2 (2026-10-21, after the fresh-agent read-out audit notes/batch1_build/audit_ptheld2_readout.md, before any result; read-out, AUDIT and SUPP rendering only -- trainer, snapshot and the submitted sbatch unchanged)
Supersedes the verdict rules of the PRE-REGISTERED section, SUPP and AUD where they differ; tools/ptheld2_summary.py
(commit 01c6f7e and later) is the reference. Labels: the ptheld2 core criteria are C1 (none floor), C2 (the 0.50 bar), C3
(paired tests + specificity); rescue_standard's are RS(a) / RS(b) / RS(d) (the old "(a) / (b)" named both).
- B1 (blocking): rescue_standard (b) needs the prior arm's own lr sweep. RS(b) = SUPP-2 cells: the chosen arm at lr 1e-3 /
  2e-3 / 4e-3 on seeds 0-3 (mean acc_L final3); 2e-3 best or within 0.02 of the best -> met (bracket closed); another lr
  better -> "open: best lr X != 2e-3, CONF not run there" (PROVISIONAL); cells absent -> NOT RUN.
- Verdict per layer (in this order): any SUPP / SUPP-2 run the reading uses invalid -> WITHHELD (also when partial AND
  invalid: invalid wins over missing, audit m1); main cells missing / invalid -> INCOMPLETE / WITHHELD; a core criterion
  fails -> "no rescue: ..." (only uninformative controls -> "NOT ESTABLISHED", audit m9); core holds and any RS "not met"
  -> "no rescue (core criteria hold; rescue_standard not met)"; RS(a), RS(b), RS(d) all met -> "RESCUE (tier L)"; RS(a) and
  RS(d) met, RS(b) not run -> "PROVISIONAL rescue (tier L; (b) open: prior at one lr)"; otherwise (not run / incomplete /
  uninformative / open) -> "PROVISIONAL (partial)". RESCUE appears only when all three RS criteria are met. The verdict
  line carries the tier-L wording ("given the structure, the model learns to use it"), the layer wording (A: "large h AND
  (for 35% of its rows) a two-token answer"; B: "rare by standalone occurrences in the text, not by token exposure") and,
  for an emb arm, F90's "injecting h's value coordinates into its embedding substitutes for task exposure". The untrue
  "(bracket state reported)" is gone.
- I1: C1 (and the LC floor inside RS(a)) as ptheld R1(b): >= ceil(3/4 n) seeds informative AND every informative seed
  <= 0.10 (was: >= 3 informative seeds <= 0.10, which ignored a fourth informative seed that learned H).
- I2: RS(d) needs >= 3/4 of the control's seeds 4-7 informative (else "uninformative" -> PROVISIONAL); acc < 0.50 in >= 3/4
  of the informative ones, and the +0.20 margin with >= 3/4 wins over them.
- I3: RS(d)'s control has the chosen arm's schedule: win-emb-hremap-always for win-emb-always, win-emb-hremap-decay
  (SUPP-2) for win-emb-decay, none for win-text-always (NOT RUN). An absent control -> NOT RUN -> at most PROVISIONAL.
- Minor: m2 the cross-arm checks use the fixed reference none@2e-3 (a mismatch is charged to the other cell, never to an
  innocent main cell); m3 a non-finite bpb is the worst in the selection tie (RR.worst, as ptheld B5); m4 the RS(a) bracket
  means and paired differences use informative seeds only (>= 3 per cell), an uninformative LC cell / axis ->
  "uninformative" (PROVISIONAL, not "no rescue"); m5 the text arm: the prior applied iff the batch was usable (geo logged),
  never with < 16 text numbers, prior_used_frac >= 0.9 x prior_on_frac (ptheld's win-text: 1.0); emb / ctx: always applied
  (prior_used_frac = prior_on_frac); m6 the half labels re-derived from count_text_site (rank by count, ties by value) and
  the lo_rare mask from those halves (not from the logged labels), and logp_H_one_<L> of the final3 evals = the mean of
  logp_rows_H_one on the mask (1e-3); m8 the selection seeds' agreement printed (descriptive); m11 RS(b) read from the
  SUPP-2 cells (the cells are recognised, no longer "not a ptheld2 cell").
- Not changed: ptheld_summary.py's own docstring (final3 steps), which this branch does not touch (the other session's
  uncommitted work in llmadd-burst-ro edits that file); rescue_standard (f) (L40S by partition, GPU name in the slurm log).
- Selftest: 36 scenarios + 24 unit checks PASS (Holm known answers incl. the chain and a NaN p; the verdict ladder over 12
  RS state combinations; final3 = the last three evals; a NaN bpb never wins a tie). Mutation test (scratchpad
  ptheld2/mut/: mutate3.py ports the audit's 18 meaningful mutations to the new code, plus 6 of the AUD2 rules): see
  "Mutation test" below.
- SUPP-2 cells (make_jobs_ptheld2.py --supp2, rendered, NOT submitted, the user's call): win-emb-always, win-emb-decay,
  win-text-always at lr 1e-3 and 4e-3, seeds 0-3 (24 jobs); win-emb-hremap-decay (helix3_hremap, emb, --lam_decay, window
  0.05,1.0, lambda 1, lr 2e-3), seeds 0-7 (8 jobs); 32 jobs, 35.2 L40S-h planned. Same snapshot / code_sha; only the
  listed flags differ from the main cells (asserted by the generator).

### Mutation test
tools/ptheld2_summary.py sha 53eb33122fa0 (scratchpad ptheld2/mut/runmut3.sh: each mutant runs the read-out's own
--selftest; no catch is a crash: 0 KeyError / IndexError / TypeError in the logs, every catch is an assertion or a FAIL):
- the audit's 18 meaningful mutations: 18 / 18 caught -- holm_swap and holm_nochain (unit: Holm known answers), conf_off4
  (S5), conf_eq_sel (S4a), supp_missing_rescue and supp_withheld_soft (unit: verdict ladder), field_full / field_v9 /
  field_common / final3_shift (unit: final3 of the layer's own key), floor_030 (S2), floor_all_seeds (S8), spec_vs_none
  (S3), d_text (S14), no_4x_floor (S24), no_xgate (S25), bar_030 (S25: the gate seeds), d_no_margin (S27);
- 6 mutations of the AUD2 rules: 6 / 6 caught -- b_ignored (unit ladder), floor_any3 (S23), d_no_inf (S22),
  d_always_for_decay (S21), needed_missing_first (S28), nan_bpb_not_worst (unit NaN tie);
- the equivalent mutant (layer B read on 'rare', the same 324 rows): not caught, as expected.

## Audit status
i double path ✓ (H draw re-implemented: hcheck.py, remapcheck.py; grad_T vs the backward's grad_total; layer accuracy vs
its rows' hits in the read-out; lam_now asserted in-run against an independent formula); ii asserts ✓ (H construction, layer
masks, lam_now, grad_T on CPU; every Rev C assert unchanged); iii known answers ✓ (default path bit-identical; lam_now =
1 - (s - 2) / 22 on the tiny run; 17 fake scenarios with known verdicts); iv - (no results); v the fresh-agent review was
dispatched by the coordinator after the interim report (not by this agent); vi ✓ (train / eval table; side effects:
regression of 9 configs incl. --domains and every Rev C path; job rendering: bash -n + generator faults); vii ✓
(--dry_inputs at the production shape for all 7 main cell types + hremap-always + none wd0.01, decoded rows in
scratchpad ptheld2/proddry/*.log).
Not run: a resume test of Rev D (the resume path is unchanged; lam_t depends on the step only); a tiny training smoke of
win-text-always / win-ctx-always / hremap-always with value_freq (their code paths are Rev C1's; build covered by the
production dry runs).

## Amendment SITE (2026-10-21, user decision, before any ptheld2 result existed: no run had started)
- Why: on the public L40S partition the 52 main jobs were PENDING with no start estimate for most of them. A new L40S job
  was estimated to start on 10-27, and Burst's GPU budget was exhausted (294/296, 293/296). The user chose the tandon
  account.
- Test-only estimates, single job:
  - torch_pr_520_tandon_priority on a100_tandon: 10-21 19:37Z;
  - priority on h100_tandon: 10-22 05:01Z;
  - advanced: a100 10-24, h200 10-23, h100 10-26.
  - Chosen: priority + a100_tandon. CIA's convention reserves priority for inference; the user asked for the tandon
    account and this is the only option that starts before the deadline.
- What moves: every ptheld2 cell (main 52 + SUPP 20 + SUPP-2 32 = 104) runs on A100 in runs/ptheld2t. Criterion (f)
  then holds on A100: every compared arm is on the same GPU type.
- What is unchanged:
  - the trainer args, except the --out root (runs/ptheld2t/out), checked by a diff over all 104 job pairs;
  - the code: runs/ptheld2t/code is a byte-for-byte copy of runs/ptheld2/code (trainer 93e22bf4c4c0 and its 4 imports);
  - every rule above.
- The L40S copies are stopped with runs/ptheld2/HOLD (all 52 were PENDING; a pending job exits 0 at start) and are never
  read.
- Read-out: `python tools/ptheld2_summary.py --root /scratch/yh6210/llmadd_torch/runs/ptheld2t/out ...`
  (read-out sha 53eb33122fa0).
- Before reading: check that every job log in logs/llmadd_ptheld2t-* prints an A100. A different GPU type in any run
  means WITHHELD.
- Generator: `make_jobs_ptheld2.py --site tandon [--supp] [--supp2] --submit`. Without --site, the 104 L40S job files
  re-render byte-identical (md5 checked).

## Amendment WD3 (2026-10-22, POST HOC: written AFTER the ptheld2t results were read and audited -- in particular the wd 1.0 cell; notes/batch1_build/audit_ptheld2t_results.md (iii), I4-I6 -- and BEFORE any wd 0.3 / none@1e-3 / none@2e-3-4x run; the user authorised the submission)
This completes rescue_standard (a) after the read-out. A FINDINGS entry must say so ("RS(a) completed after the read-out;
the added cells were chosen after the wd 1.0 result was seen, their criteria fixed before their results").
- Why: by the AUD2 rules layer A read PROVISIONAL because RS(a)'s wd axis was uninformative. none@8e-3-wd1 failed the
  unstable gate in 4/4 seeds (final val_loss 0.73-0.77 nats above the same seed's none@2e-3; bpb 1.42-1.43 vs 1.20-1.22;
  held6 0.53 / 0.73 / 0.87 / 0.86): lr x wd = 8e-3 per step breaks the LM itself, so the cell carries no information on H.
- Changes (read-out tools/ptheld2_summary.py sha bc769b424b52; jobs `make_jobs_ptheld2.py --site tandon --supp3`):
  1. The wd axis is {0.01, 0.1, 0.3} at lr 8e-3. wd 1.0 is reported as it is ("uninformative: the LM is broken"; never
     "did not learn seen addition"), is not on the axis and no longer blocks it.
  2. The wd axis' OPEN rule, fixed now (audit I4: on H, wd 1's logp_hi -9.64 is "better" than wd 0.1's -10.41 while its
     seen held_b33 logp is -2.64 vs -0.73 and its median first-token rank 49 vs 4.8: heavy wd softens the distribution).
     Over informative seeds 0-3, the best cell at an edge is OPEN against its neighbour (0.1) iff
     (i) acc: best mean acc_L final3 > 0.05 and paired mean > +0.02 with >= 3/4 paired wins; or
     (ii) rank: the median first-token rank over layer L's rows pooled over the final3 evals (rank1_rows_H_one) is >= 2x
          better, paired mean log2((rank_best + 1) / (rank_nb + 1)) <= -1, with >= 3/4 paired wins; or
     (iii) logp: paired mean > +0.25 nats with >= 3/4 wins, counted only when the best cell's seen held_b33 logp final3 is
          not worse than the neighbour's (paired mean >= -0.05 nats).
     The lr axis keeps the AUD2 rules (acc with the 0.05 gate and 0.02 margin, or logp 0.25 nats).
  3. Holds (layer A): wd 0.3 informative in >= 3/4 seeds and every informative seed A final3 <= 0.10; the wd axis not OPEN;
     the lr axis closed (incl. the lower-edge check in 5). The verdict is then RESCUE (tier L) by the AUD2 ladder, with the
     WD3 wording on the verdict line.
  4. Does not hold: wd 0.3 uninformative -> PROVISIONAL (consider wd 0.3 at lr 2e-3); an axis OPEN -> PROVISIONAL (more
     cells needed, e.g. wd 0.5; this replaces AUD2's "OPEN -> not met -> no rescue" on both axes); any informative wd 0.3
     seed > 0.10 -> layer A "no rescue (core criteria hold; rescue_standard not met)".
  5. Run with it (recommended, audit I5 / I6): none@1e-3 and none@2e-3-4x, seeds 0-3. With none@1e-3 run the lr axis is
     1e-3 / 2e-3 / 8e-3 / 1.6e-2: 2e-3 is no longer an edge, and none@1e-3 best AND OPEN by the AUD2 lr rules -> OPEN
     (PROVISIONAL). none@2e-3-4x: a descriptive budget line (acc and logp vs none@2e-3). Both, when they ran, also pass
     the C1 floor rule (rescue_standard (a): "none fails at every configuration"; stricter than descriptive, so a none
     configuration that learned H can never sit under a RESCUE); not run -> not applied.
  6. Wording on every RESCUE / PROVISIONAL line: tier L, layer A / B, emb, the tested scope ("within the none
     configurations tested ... <= 4x tokens; only while the prior keeps h's coordinates in the embedding") and "RS(a)
     completed after the read-out (amendment WD3)"; the audit's (ii) "do not write" list is printed after the verdicts,
     and the read-out asserts that no verdict head contains any of its assertable phrases (FORBIDDEN).
- The read-out on the current ptheld2t/out (no SUPP-3 run yet): layer A PROVISIONAL with RS(a) INCOMPLETE (none@8e-3-wd0.3
  missing); B no rescue (C1); 104 / 104 runs valid under the WD3 checks.
- SUPP-3 jobs (A100 only: --supp3 needs --site tandon; snapshot runs/ptheld2t/code, trainer 93e22bf4c4c0, a100_tandon,
  torch_pr_520_tandon_priority): none@8e-3 wd 0.3, none@1e-3, none@2e-3 4x, seeds 0-3 = 12 jobs; ~16 A100-h at the
  measured 0.66 h per 1x run and 2.64 h per 4x run in ptheld2t. The 104 tandon and the 104 L40S job files re-render
  byte-identical with and without --supp3 (md5 of all 208). Production-shape --dry_inputs of the three new cells: clean
  (0 H tokens in 6,400 lines). The tandon job files contain no GPU-type check (only nvidia-smi prints the name; Amendment
  SITE's pre-read check: every log prints an A100).
  Submission (the user's call): `cd /scratch/yh6210/llmadd_torch && python make_jobs_ptheld2.py --site tandon --supp --supp2 --supp3 --submit`
  (the 104 already submitted have SUBMITTED_ flags and are skipped; only the 12 SUPP-3 jobs are new).
- Selftest after WD3: 46 scenarios + 24 unit checks PASS (W1 wd 0.3 informative at the floor -> A RESCUE (tier L) with the
  WD3 wording, B no rescue; W2 wd 0.3 uninformative -> PROVISIONAL; W3 / W3b / W3d OPEN on acc / rank / gated logp ->
  PROVISIONAL; W3c better H logp with a worse seen logp -> not counted -> RESCUE; W4 an informative wd 0.3 seed > 0.10 ->
  no rescue; W5 none@1e-3 OPEN on the lr axis -> PROVISIONAL; W6 none@2e-3-4x seed > 0.10 -> no rescue; W7 wd 0.3 not run
  -> PROVISIONAL). Mutation test (read-out sha bc769b424b52, scratchpad ptheld2/mut/runmut3.sh): the audit's 18 mutations
  18 / 18 caught, the AUD2 + WD3 rule mutations 12 / 12 caught (wd1_on_axis, logp_ungated, no_rank_rule, no_lower_edge,
  extra_not_floored, open_as_met among them), 0 catches by a crash; the equivalent mutant (B read on rare) not caught.

## Amendment WD3b (2026-10-22, after the fresh-agent review of the WD3 read-out, notes/batch1_build/audit_ptheld2_wd3.md, and BEFORE any SUPP-3 result was read by anyone; read-out only -- trainer, snapshot and sbatch unchanged)
The 12 SUPP-3 jobs were running; nobody read their outputs. Tests use fakes only, the regression the frozen root of the
original 104 runs (scratchpad r05/frozen104). Read-out: tools/ptheld2_summary.py sha e0580566ffa1 (branch ptheld2-build,
merged with origin/main incl. the R05 fix fdde365 "a missing or non-finite diagnostic is never a pass").
- B1 (blocking): on the wd axis a logp OPEN whose seen held_b33 logp is worse than the neighbour's (paired mean < -0.05
  nats) is now "undetermined" (calibration cannot be excluded) -> PROVISIONAL; it is never counted as closed. (The WD3
  version turned it into "closed", which under the null would drop the logp evidence ~44% of the time and could print
  RESCUE.) No low-noise seen probe was added: with (a) both outcomes of the check are PROVISIONAL, so a second probe could
  only change the label.
- I1: none@1e-3 and none@2e-3-4x are required (submitted): a missing or partial cell -> RS(a) INCOMPLETE -> never RESCUE.
  (S10 / S19 and every scenario of group "a" now include them.)
- I2: the rank rule (median first-token rank >= 2x better, >= 3/4 paired wins) applies on the lr axis too, incl. the
  1e-3 lower edge; the three rules (acc, rank, logp) are the same on both axes, plus the seen-logp check on the wd axis.
- I3 / m2: the [std] block of an evaluated RS(a) notes "wd 0.3 none is a degraded LM" when any wd 0.3 seed's final
  val_loss is > none@2e-3 + 0.2 nats (the informative definition is unchanged), and "wd 1.0 excluded from the wd axis: the
  LM is broken (val_loss ...)".
- m1: "RS(a) completed after the read-out (amendment WD3)" is printed only when RS(a) was evaluated (not NOT RUN /
  INCOMPLETE / WITHHELD).
- m5: the FORBIDDEN phrases are checked on every printed line except the "do not write" list (possessive apostrophes
  dropped, quoted spans removed), besides the verdict head; unit checks show both assertions fire.
- m10: check_run requires rank_hit_bad_H_one = 0, every hit row's rank = 0 and rank1med_H_one = the lower middle of
  rank1_rows_H_one (torch's median) on the final3 evals.
- Not changed: I4 (the wd axis only at lr 8e-3; the scope wording says "cross at wd 0.1 / lr 8e-3") -- not in this
  amendment's brief; m3 (a running SUPP-3 run reads as invalid: read only after sacct shows all 12 COMPLETED).
- Regression (frozen104, the original 104 runs, old = the merged read-out dd27e5efd2bf, new = e0580566ffa1): 190 / 190
  lines; the only difference is the removal of the WD3 phrase from layer A's PROVISIONAL line (m1: RS(a) is INCOMPLETE
  there, wd 0.3 missing); every other byte identical, 104 / 104 runs valid under the new rank double path.
- Selftest: 58 scenarios + 26 unit checks PASS (new V1 / V2 extras missing -> INCOMPLETE; V3 lr-axis rank OPEN; V4
  undetermined; V5 seen per seed with mean 0 -> counted; V6 rank exactly 2x -> OPEN; V7 3.2x; V8 rank only at the last
  eval -> closed; V9 rank on layer A's rows only -> A OPEN; V10 degraded-LM note; V11 a forbidden phrase in the output ->
  refused; V12 rank double path; W3c now PROVISIONAL (undetermined); W1 checks the wd 1.0 note; W7 checks that
  "completed" is absent while RS(a) is INCOMPLETE).
- Mutation test (read-out e0580566ffa1, scratchpad ptheld2/mut/mutate4.py + runmut4.sh): the build / read-out audits' 18 -> 18
  caught; the AUD2 + WD3 rule mutants 12 -> 12; the WD3 read-out audit's 12 (ported; its 2 "stricter" ones are now the code,
  so their inverses = the old WD3 behaviour are tested) -> 12; the WD3b rule mutants 6 -> 6; 0 catches by a crash; the
  equivalent mutant (B read on rare) not caught, as expected.

### Provenance note P1 (2026-10-22, written before the WD3 / WD3b read-out; review notes/batch1_build/audit_ptheld2_wd3b.md, P1)
- What happened: at about 02:46 EDT, as part of the R05 regression (the missing-diagnostic fix), the coordinator ran the
  old and the new ptheld2_summary.py on the LIVE root runs/ptheld2t/out, not on a frozen copy. By then some WD3 cells
  (wd 0.3, none@1e-3, part of the 4x runs) had results, so the outputs (scratchpad/r05/p2_old.txt, p2_new.txt,
  p2_diff.txt) contain WD3 values.
- What was seen: the coordinator displayed only the diff's exit code ("diff rc 1") and its line count (16). No result
  value from those files entered the coordinator's context. On seeing the non-zero diff, the coordinator built
  scratchpad/r05/frozen104 (the original 104 runs only) and re-ran the regression there. The WD3b builder and both WD3
  / WD3b reviewers used fakes or frozen104 only.
- Effect: WD3b only tightens the rules: gated logp -> undetermined -> PROVISIONAL; extras required; rank rule on both
  axes. It cannot move a verdict toward RESCUE. FINDINGS must state this note, and must not say "criteria fixed before
  any result existed" without it.
- Read-out conditions (review): run e0580566ffa1 once; 116/116 runs valid with no INVALID lines; no nan in the RS(a)
  lines. Scope: the wd axis was swept only at lr 8e-3 (a lr x wd cross, not a grid; the (2e-3, wd 0.3) corner is
  untested). The "completed after the read-out" phrase still printed on PROVISIONAL lines is not to be copied.
