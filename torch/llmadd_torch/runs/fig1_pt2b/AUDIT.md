# fig1_pt2b — number representations of F86's 2% main setting across training, for the paper's Figure 1 (2026-10-22)

Intent spec (experiment-audit step 0, MINIMAL mode: ii asserts in the code + v a fresh adversarial sub-agent after
submission; i / iii / vii are done as part of the export test, see "Checks"). Written before any run (runs/fig1_pt2b/out
did not exist). The submission is authorised by the user (relayed by the writing session, confirmed by the coordinator).

## What and why
Figure 1 needs the states of the numbers 0..999 at several points of training for three arms of F86's pt2b setting
(53M from scratch, 2% arithmetic, 3e8 tokens): no prior, the helix3 prior on the operands (window 0.05-0.4), and the
dose-matched shuffled-geometry control. F86's Burst runs kept only the final model state off-Burst (only run JSONs are
in llmadd-burst results/), and --ckpt_every overwrites one resumable file, so seed 0 of the three arms is re-run on
Torch with an opt-in flag that keeps the model state at chosen update counts. Descriptive only: no new F-number claim
is planned from these runs; the all-seed CKA curves come from F86's existing logs.

## Code and provenance
- Trainer: llmadd-burst `tools/pretrain_range.py` Rev E (branch `fig1-export`, worktree /scratch/yh6210/llmadd-burst-fig1,
  from origin/main 9bc98f4; not pushed). Rev E = one opt-in flag `--save_at S1,S2,..`: the model state after S updates
  -> `<out>/save_at/model_<tok>_<target>_seed<s><tag>_upd<S>.pt` ({"model", "updates", "steps", "code_sha"}). Without the
  flag the trainer is origin/main (Rev D) bit for bit (CPU regression below). The flag's value is logged (stdout line
  "save_at: the model state after [...] updates", one line per save) and recorded in the run JSON (args.save_at,
  save_at.requested / files); the run asserts at its end that every requested file exists.
- Job code = snapshot `runs/fig1_pt2b/code/tools/` (git archive of tools/ at the committed HEAD; `code/COMMIT` = HEAD +
  sha256[:12] of the trainer and its imports). Generator `llmadd_torch/make_jobs_fig1pt2b.py` (refuses uncommitted
  tools/, asserts every flag; every sbatch re-checks the trainer sha (exit 3), the data sha256, the 53M model line and
  the sha256 of the 9 --dry_inputs lines = pt2main's seed-0 value 379354e2... (the same training rows as F86's Burst
  seed-0 runs; exit 3 otherwise); honours runs/fig1_pt2b/HOLD).
- F86's runs used the pinned trainer 64abc19 (code_sha a3bce3ab63c1); Rev C/D/E's default path is that trainer step for
  step (earlier CPU regressions + this one), so the re-run differs from F86's seed 0 only by hardware (Torch A100 vs
  Burst A100: not bitwise; compared descriptively).

## Setting (= make_pt2main_tasks.py render() for pt2b, seed 0)
53M (d384 x 6 x 6 heads, d_ff 1536, tied embeddings), seq 512, bs 32, 3e8 tokens = 18310 updates, warmup 183, cosine to
0.1x, AdamW (0.9, 0.95) wd 0.1, clip 1.0, bf16 autocast, --arith_lines 1 (2%), eval every 250 steps + the last (75
evals), --save_model --ckpt_every 2000. Torch data /scratch/yh6210/llmadd_torch/pretrain_data (sha256 = Burst's) and the
OLMo-2 tokenizer of the 9B checkpoint (pt999's).

| arm | target | schedule | lr | why |
|---|---|---|---|---|
| none | none | - | 8e-3 | none's area-best lr in F86 pt2b (23.1, seeds 0-3) |
| win-ctx | helix3, site ctx, lambda 5 | window 0.05,0.4 | 2e-3 | the prior's area-best lr in F86 pt2b (69.8) |
| win-shuf-ctx | helix3_shuf, site ctx, lambda 5 | window 0.05,0.4 | 2e-3 | shuffled-geometry control |

Shuffle target: `helix3_shuf` = helix3 of a per-seed whole-table permutation of 0..999 (pretrain_range.py `perm`,
default_rng(124 + seed)), the target of F82's addendum (pt999 2pct-win-shuf-ctx, dose-matched at 2%) and of F86's only
shuffle cell (pt2long win-shuf-ctx@1e-3). `helix3_farshuf` needs --held_operands (asserted by the trainer), so it is not
an option here. F86 has NO pt2b shuffle cell: this one is new, run at the PRIOR's lr (2e-3), not at its own best lr.

## Checkpoints (update counts; "the state after S updates")
steps = int(3e8 // 16384) = 18310; window = prior on iff 0.05 * 18310 <= s < 0.4 * 18310, i.e. steps 916..7323
(prior_events [[916, 1], [7324, 0]]).

| S | meaning | tokens seen |
|---|---|---|
| 0 | initialisation | 0 |
| 916 | window opens (state before the first prior update) | 15.0M |
| 3662 | 20% (0.2 x 18310) | 60.0M |
| 7324 | window closes (state after the last prior update, step 7323) | 120.0M |
| 9324 | window close + 2000 updates | 152.8M |
| 18310 | final (= the --save_model file) | 300.0M |
| 18309 | not a figure point: the state whose forward the last logged eval (step 18309) used; double path only | |

Logged-curve alignment: eval step s logs cka_ops_helix3 from step s's TRAINING forward (the state after s updates) and
evaluates accuracy / cka_emb_helix3 after step s's update (the state after s + 1 updates).

## Read-out (tools/fig1_export.py; per state)
- Sites (number tokens 0..999 are single tokens; the state OF the number token):
  - qx: "Q: x" alone at position 0 = the trainer's --site prefix prompt PRE = enc("Q: x") (4 tokens, state at index 3).
    The trainer's --site text (natural FineWeb numbers) has no fixed prompt; "Q: x" is the brief's text-site prompt.
  - ctx: x as operand a of the trainer's format-0 line "Q: x + 0 = x\n", in the eval-row layout (501 val.bin tokens, then
    the 11-token line; 512 tokens), state at index 504 = the trainer's ia; one fixed text chunk (default_rng(20261022)).
    Causal: the state sees the text and "Q: x" only. Index 504 lies in the training range of format-0 operand a (503 / 504: 2- / 1-token answer).
    Alternatives (CKA only): 3 other text chunks; x as operand b ("Q: 0 + x = x\n", index 507).
- Layers 2, 4 (= the trainer's geo_layers / read_layers for 6 layers) and 6 (the last block's output, pre final norm),
  via forward hooks on model.model.layers[L - 1] as in the trainer; plus the input-embedding rows (tied).
- CKA = pretrain_helix.cka(H, centred_gram(helix3(0..999))) (the trainer's implementation and its K1000), fp32 states.
- Expected: init ~0.05-0.25 (embedding 0.05 at init in pt999); win-ctx at layers 2 / 4 high after the window (pt999
  2pct-win-ctx final cka_ops_helix3 0.97); none lower; shuffle low vs the TRUE helix (F82 add.: 0.18 / 0.22 on batches).

## Checks
- CPU regression (tiny config, 3 arms): origin/main trainer vs Rev E without the flag vs Rev E with it: identical eval
  lines, run JSON (code_sha aside), final model tensors; with the flag exactly the requested files; upd S == the
  resumable ckpt after step S - 1 (independent path); upd 0 == a fresh build_model after torch.manual_seed(seed);
  upd steps == the --save_model state.
- Export double paths (i): (a) replay of the trainer's training batch of a logged step (RNG streams 0 / 1000 / 2000 +
  seed), CKA at its ops positions under bf16 vs the logged cka_ops_helix3 (exact state for S = 0 and 18309); (b) the
  embedding CKA of the state after S updates vs the logged cka_emb_helix3 of eval step S - 1; (c) the KV-cached carrier
  states vs the full 512-token row forward. (vii) decoded token sequences and the taken index for 3 numbers.
- Known answers (iii): the test on pt999 2pct-win-ctx seed 0 (final, logged 0.966) before the new states land.
- Minimal mode: iv – (no anomalous result to replicate), vi via the sub-agent.

## Audit v (fresh general-purpose sub-agent, after submission, 2026-10-22 ~01:40) and what was done
No blocker for the training jobs (Q3 / Q4 "not found": the flag leaves the default path bit-identical, the saved state
is the state after S updates, the replay's RNG order = batch()'s; position 504 = ia, cache vs full row 1.3e-6, hooks and
layer numbering as the trainer). Findings on the READ-OUT, each checked and fixed before any export was used:
1. (important) fig1_logcka's "nearest eval" for S = 916 was 1000, 84 prior updates into the window (win-ctx 0.847 vs
   0.460 at 750). Fixed: at the window edges the primary eval is the latest eval <= S (750 / 7250); both sides are kept.
2. (important) same name, different quantity: the logged cka_ops_helix3 is a 64-operand batch estimate whose values are
   1/3 one-digit (bucket-uniform), the export's CKA is uniform over 0..999 on one carrier; for none / shuffle they
   differ by ~0.1-0.2 (sub-agent, pt999 final: none 0.483 vs 0.615 logged). Fixed: the export also reports cka_opmix
   (the same states at one batch's operand-value mix, 200 draws), side by side; the README states which is which.
3. (important) the replay double path ran under CPU bf16 autocast, which quantises the CKA to bf16 and cannot tell the
   right batch from a wrong one near saturation. Fixed: fp32 CKA besides the bf16 one, and two wrong-batch negative
   controls wherever a logged value exists (the right batch must be the closest); every saved state is replayed on its
   own batch (no logged value at 916 / 3662 / 7324 / 9324: reported with the logged evals around S, descriptive only).
4. (important) the Torch re-run equals F86's Burst seed 0 only at step 0 (lm, cka_ops, cka_emb identical), then
   diverges (GPU non-determinism): the exported states are a 9th sample, to be shown with the re-run's own logged curve
   over the 8-seed band (README).
5. (interpretation) qx ("Q: x" at position 0) is never a training form in --site ctx runs (every training line follows
   ~490 text tokens): a position-transfer probe; a low qx CKA does not mean the prior did not transfer.
6-10. (minor) none at its own lr (8e-3) vs the prior arms at 2e-3 is F86's best-vs-best design (the shuffle compares
   with win-ctx at the same lr); the job checks do not see the target: verify_final.py checks args.target,
   prior_events, 75 evals, the 7 files and upd18310 == --save_model after each run; a preemption between the last JSON
   write and the final save would leave no upd18310 (verify_final would fail; no restarts happened, sacct Restarts 0);
   cka_ops (s updates) vs cka_emb / held6 (s + 1) labelled in the CSV; the export code was committed (0053e3a) before use.

## 2026-10-22: shuffled control win-shuf-ctx@2e-3 at seeds 1-7 (writing session request, under the user's standing approval)
- Why: Figure 1 (d) should show the shuffled control's curve over 8 seeds, like none@8e-3 and win-ctx@2e-3 (F86 pt2b, Burst).
- What: copies of job_win-shuf-ctx-s0.sbatch. Only these change: the seed; a per-seed out dir out/fig1-pt2b-lr2e-3-win-shuf-ctx-s<seed>
  (seed 0's save_at checkpoints feed Figure 1 (b)(c) and must not be shared); no --save_at; the expected dry_inputs hash,
  which is that seed's pt2b Burst task (results/35x / 37x-pt2b-lr2e-3-win-ctx-s<seed>/dry_inputs.txt; seed 0's recomputed
  hash equals the one in seed 0's job).
- Caption must say: the shuffled control uses the prior's lr 2e-3, not its own best lr; it runs on Torch A100-80GB, while
  none / win-ctx are Burst A100-40GB.
