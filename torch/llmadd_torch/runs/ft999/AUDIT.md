# ft999 — intent spec (experiment-audit step 0, minimal mode), written 2026-10-22 ~02:00 ET, BEFORE any result

Purpose: LLM_addition paper, Figure 1 bottom row (fine-tuning on three-digit addition with a number-manifold prior).
User approval (relayed by the main session): "可以的，没问题，请你直接开始跑就行". Deadline: results by Fri 10-23 morning ET.
Code: `tools/manifold_train.py --add3` on branch `ft999` (worktree /scratch/yh6210/llmadd-burst-ft999, not pushed),
snapshot `runs/ft999/code` (COMMIT file; sha256[:12] of manifold_train.py asserted in every job).
Generator: `/scratch/yh6210/llmadd_torch/make_jobs_ft999.py`. Read-out: `tools/ft999_summary.py` (written before results,
self-tested on fake curves: `python tools/ft999_summary.py --selftest`).

## Question
Does pulling the "Q: x" representations of x = 0..999 toward the helix3 geometry (lambda 20, CKA, layers 4/8/12/16 as F64)
make LoRA fine-tuning of OLMo-2-7B learn three-digit addition faster / better on held-out pairs than the task loss alone
(none), and than the same prior with the number -> point assignment shuffled (helix3_shuf)?
Prior work: F18 (two-digit training, helix3 over 10..999, three-digit extrapolation: no help); F64 (two-digit, 9B, 7000
pairs, 10 epochs, lambda 20: helix >> none >> helix_shuf). ft999 trains ON three-digit addition (F18 never did).

## Pre-registered design (nothing below is chosen after results)
- Checkpoints (both submitted together): **PRIMARY = 9B = stage1-step2000-tokens9B** (F64 / viz). SECONDARY = 42B =
  stage1-step10000-tokens42B, reported beside the primary and never swapped in for it.
- Arms: none / helix3 / helix3_shuf; seeds 0, 1, 2; one job per (checkpoint, arm, seed) = 18 jobs, all A100 (guarded).
- Data (manifold_train.py, `if args.add3:` data block): operands a, b independent uniform on 0..999; generator
  np.random.default_rng(2026). Test = 2000 pairs drawn first, unique as UNORDERED pairs. Train = 7000 ordered pairs drawn
  next from the same stream, skipping every unordered test pair (b + a of a test pair is never trained; asserted).
  Train and test depend only on that generator: identical for every arm, seed and checkpoint (test_sha in every JSON;
  the read-out refuses runs with different test sets). Seed changes only data order (rng 1000 + seed) and adapter init
  -> seed s of every arm is a pair (as F64).
  - 7000 pairs x 10 epochs (4380 steps = 10 x ceil(7000 / 16), bs 16) = F64's setting, kept so the step budget / curve resolution match F64;
    nothing in the code, F18 or F64 argues otherwise. Note: 7000 pairs are 0.7% of the 10^6 ordered pairs (F64: 86% of
    the two-digit pairs), so held-out accuracy here measures generalisation of the rule, not recall.
  - 2000 test pairs (F64: 1000) because the result is split by sum < 1000 / >= 1000 (~50% each, ~1000 per split).
- Prompt / answer: "Q: a + b = " (F64 template, prompt at sequence start, no context) followed by "<a+b>\n".
  Tokenizer facts (asserted at job start): every 0..999 is one token; the prompt is 9 tokens with a at index 3, b at 6;
  "Q: x" is 4 tokens with x at index 3 for every x in 0..999; sum s < 1000 -> [s][\n]; s >= 1000 -> [s // 10][s % 10][\n]
  (e.g. 1234 -> "123" "4"); the target tokens equal the tokenizer's tokens of the full text for every train and test item.
- Loss: CE = token mean over the answer tokens and the newline (teacher forced, logits of the last 3 positions of the
  11-token input; padding position ignored) + lambda * L_geo. LoRA r 8 on q/k/v/o/gate/up/down, alpha 16, dropout 0.05,
  AdamW lr 1e-4 wd 0, bs 16, no clipping (manifold_train defaults = F64).
- Prior (= F64 site): L_geo = mean over L in {4, 8, 12, 16} (decoder-layer outputs, hook on layers[L]) of
  1 - linear CKA(H_L, K) where H_L = A-slot state (last token of "Q: x") of 180 numbers sampled per step from 0..999
  (geo_rng seed 99: the same sample sequence in every job), K = centred Gram of the target rows. lambda = 20 (F64).
  - helix3 target (F18's, extended from 10..999 to 0..999): cos/sin(2 pi x / T), T in {2, 5, 10, 100, 1000} (sin dropped
    for T = 2), plus x / 1000; columns z-scored over 0..999.
  - helix3_shuf: the same rows permuted by ONE fixed permutation perm3 of 0..999 (np rng 321), the same for all seeds
    (F64 / F18 used one fixed permutation too). FAR check (asserted < 0.15 in every job): CKA(shuffled Gram, true Gram)
    over all 0..999 = 0.0067 (digit3_shuf 0.028, unused); fixed points of perm3 = 0; mean |perm3(x) - x| = 337. It is a full permutation over
    0..999 (mixes 1-, 2-, 3-digit numbers, like F18's helix3_shuf over 10..999); there is no within-digit-length shuffle arm.
  - none: CE only; the helix3 loss is computed under no_grad and only logged (train_log geo_loss).
- Curves (Figure 1 (h)): held-out accuracy (all / sum < 1000 / sum >= 1000) at step 0 (= base) and every 50 steps, for
  every run (last curve point and last state snapshot at step 4350); final evaluation after step 4380 (test + train).
- States: 9B, seed 0, every arm: layer-16 A-slot states of "Q: x" for x = 0..999 at step 0 and every 50 steps ->
  `out/9B/<arm>-s0/states_<arm>_seed0.npz` with states [n_snap, 1000, 4096] float16 (row x = number x), steps, layer (16),
  numbers (0..999). Same format and token position as runs/viz-9B-n7000-*/out/states_*_seed0.npz (those have 90 rows,
  10..99); the new rows need --add3 (default path unchanged, regression below).
- Prior gradient A/B: at step 0 and every 250 steps every job logs ce, geo_loss, lambda, |grad CE| and |grad lambda * L_geo|
  over the trainable parameters on the same batch (`grad_log` in the run JSON, `[grad]` lines in the log); prior arms
  assert |grad L_geo| > 0 (unscaled; grad_lam_geo = lambda x that norm).

## Metric (unique definition)
- held-out accuracy = fraction of the 2000 test pairs whose argmax at every target position (answer tokens + newline,
  teacher forced) equals the target; this equals greedy-decoding exact match of "<sum>\n". manifold_train.py:eval3.
  Splits test_lt1000 (one answer token) / test_ge1000 (two answer tokens). NOT comparable to F64's single-token "test".
- Read-out (tools/ft999_summary.py): area = mean over curve evaluations with step > 0; final = after the last step;
  to50 / to90 = first evaluated step where the seed-mean curve reaches 0.5 / 0.9 (descriptive).
- **Primary statistic (pre-registered): 9B, area (all), helix3 - none, paired by seed, n = 3**: per-seed differences,
  mean, paired t (df 2), k/3 positive. Secondary: final; the two splits; helix3 - helix3_shuf; everything at 42B.
  With n = 3 the smallest two-sided sign-test p is 0.25: results are descriptive ("3/3 seeds, mean +x points").

## Train vs eval
| dimension | train | eval | same? |
|---|---|---|---|
| sequence position / prefix | prompt at position 0, no context | same | yes |
| template | "Q: a + b = " -> "<s>\n" | same | yes |
| number range | operands 0..999 (uniform), sums 0..1998 | same distribution | yes |
| data split | 7000 pairs | 2000 pairs, disjoint as unordered pairs (asserted) | disjoint |
| scored tokens | CE on answer tokens + newline (teacher forced) | exact match on the same tokens (teacher forced) | yes |
| model state | train mode (LoRA dropout 0.05) | eval mode | standard |

## Expected magnitudes (written before results)
- base 9B ~ 0 (F64 two-digit base 0.0); base 42B low (< 10%, guess).
- none 9B: F64's two-digit none ended at 62% with 86% pair coverage; here coverage is 0.7%, so final none could be
  anywhere in 0-60%. sum >= 1000 (two answer tokens, carry into the thousands) is expected to be harder than < 1000.
- train accuracy (memorisation ceiling) should be well above test for none; if train is also ~0 the run did not learn
  (pipeline or lr problem -> check before reading the arm comparison).
- Either direction of helix3 vs none is a reportable outcome: F64 (two-digit) predicts helix3 > none > helix3_shuf;
  F18 (three-digit geometry did not help extrapolation, hurt at 150k) predicts little or negative gain.

## Controls
- Negative: helix3_shuf (same strength, wrong geometry). Positive / ceiling: train accuracy; F64's two-digit curves at 9B.
- Step 0 of every curve = base model (identical across arms by construction; the smoke checked identical step-0 states).

## Checks before submission (minimal mode)
| item | status |
|---|---|
| i double measurement | – (minimal mode) |
| ii asserts (tokens, scored targets, disjointness, far shuffle, prior grad > 0, A slot of Q: x) | in code, ran in the CPU smoke |
| iii known answers | – (minimal mode); read-out self-test on fake curves passes |
| iv anomaly reproduction | – (only if a result is anomalous) |
| v fresh adversarial reviewer | after submission (background) |
| vi trap list | trap 1 (train/eval mismatch) table above; trap 2 unordered split; trap 3 grad_log; trap 8 regression |
| vii decoded inputs | `[vii]` lines at the top of every job log; CPU smoke logs: runs/ft999_dev/smoke/job_9B-{none,helix3,helix3_shuf}-s0.log |

Default-path regression (tiny random 17-layer OLMo-2, CPU, `--geoms none,helix,helix3_shuf --train_pairs 200 --epochs 1
--curve_every 5 --save_states_every 5 --save_preds --fixed_test --lam 20`): origin/main (a740848) vs the submitted snapshot
(33e5cc8): base.json, 3 run JSONs, summary.md byte-identical; 3 states npz array-identical (runs/ft999_dev/compare_reg.py, check_smoke.py, smoke.sh;
scratchpad, outputs runs/ft999_dev/reg_old, reg_snap).

CPU smoke through the real job scripts (job_9B-{none,helix3,helix3_shuf}-s0.sbatch with FT999_SMOKE=1, never active under
slurm; tiny random OLMo-2, 64 train / 200 test pairs, 4 steps): all exit 0; states [3, 1000, 64] fp16, steps 0/2/4,
layer 16, numbers 0..999; curves with the two splits; grad_log at steps 0, 2.
Prior A/B (step 0, same batch): helix3 |grad CE| 4.82, |grad L_geo| 1.46 (x lambda 20 = 29.3; on the tiny model the prior
dominates, share 0.86; the 7B share is logged in every job at steps 0, 250, ...). Lambda consumed: helix3 at lam 0
reproduces none bit for bit (states max diff 0, same CE), helix3 at lam 20 does not (max diff 7.5), and helix3 vs
helix3_shuf differ (2.2).

## Submission (2026-10-22 02:16 ET)
Jobs (sbatch --test-only passed for all 18 first): 42B-helix3-s0=20593096 42B-helix3-s1=20593097 42B-helix3-s2=20593099 42B-helix3_shuf-s0=20593100 42B-helix3_shuf-s1=20593101 42B-helix3_shuf-s2=20593102 42B-none-s0=20593091 42B-none-s1=20593093 42B-none-s2=20593095 9B-helix3-s0=20593072 9B-helix3-s1=20593074 9B-helix3-s2=20593080 9B-helix3_shuf-s0=20593083 9B-helix3_shuf-s1=20593085 9B-helix3_shuf-s2=20593089 9B-none-s0=20593063 9B-none-s1=20593064 9B-none-s2=20593069 
First two started 02:17 / 02:21 on A100-SXM4-80GB (ga003, ga019). 7B step-0 grad A/B (helix3 s0, same batch as none s0): |grad CE| 1.089, |grad L_geo| 0.0472, x lambda 20 = 0.944 -> prior share 0.46 (step 250: 0.49). Same test_sha 420909b9ca714352 in every started job; base 9B accuracy 0.000 (all / <1000 / >=1000).

## Adversarial review (fresh general-purpose sub-agent, 2026-10-22 02:2x-02:5x ET, while the first jobs ran; scripts in runs/ft999_dev/review/)
Verdict: no blocking issue. Checked by the reviewer: [vii] lines, logits_to_keep positions (= full[:, -3:]), teacher-forced
argmax = hand greedy = generate() on the tiny model, unordered-disjoint split on the real base.json, lambda consumed,
pairing (none-s0 and helix3-s0 identical ce / geo_loss / grad_ce at step 0), default path, 9B / 42B tokenizer md5 equal,
states keys / dtype / layer / A slot identical to the viz runs (only 1000 rows instead of 90), all 18 sbatch consistent.
Important (interpretation / read-out), handled before any result was read:
- R1 helix3_shuf is matched to helix3 in lambda only: its geometry is far from the base geometry (base L16 CKA 0.41 to
  helix3 vs 0.022 to shuf), so its prior loss stays high and its gradient may be larger for longer. "helix3 > shuf" can
  also mean "a stronger conflicting auxiliary loss hurts". The read-out now prints geo_share / grad norms per arm;
  compare them before claiming the geometry itself matters.
- R2 the read-out now reports geo_share (median, max) per arm and flags median > 0.5 (project rule: report the LM-gradient
  share of every prior arm), reports base and n per split, and applies a FLOOR rule fixed now, before any result: if the
  seed-mean final held-out accuracy of none at a checkpoint is < 5%, a helix3 - none difference near 0 there is read as
  "nothing learned", not as "no effect" (ft999_summary.py commit c426ba0).
- R3 all seeds share one train/test set, one prior-sample sequence (rng 99) and one permutation (rng 321): the seed spread
  is init / order / dropout only; 3/3 holds for this data set and this permutation (stated in the read-out header).
Minor, recorded: exact match is token-level (canonical tokens of "<sum>\n"); a string-correct answer in non-canonical
tokens ("1" + "029", or "\n\n") counts as wrong, a lower bound of string-level greedy accuracy (small after training,
may matter in the earliest curve points). Steps are 4380 (not 4375; fixed above). "test" here is multi-token EM, not
F64's single-token "test"; viz_helix3d.py / viz_shape.py assume 90 rows (10..99): use rows [10:100] or extend them.
lt1000 / ge1000 also differ in operand size and the carry into the thousands, not only in answer-token count.
The none arm's JSON has geo_sample None although its logged helix3 loss uses 180 numbers.
Read-out command: python /scratch/yh6210/llmadd-burst-ft999/tools/ft999_summary.py (commit c426ba0; jobs run 33e5cc8).
