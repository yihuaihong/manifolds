---
description: Reproduce a headline result of "Manifold Pretraining Engineering" (GPU-free check, 53M pretraining, or OLMo-2-7B fine-tuning)
argument-hint: "[check | pretrain | finetune]"
---

Reproduce results of the paper "Manifold Pretraining Engineering: Right Geometry, Right Time, Right Place" with the
code in this repository. Target: $ARGUMENTS. If no target is given, run `check`, then `pretrain` if a GPU is available.

Rules:
- Use the repository's scripts as they are. Do not change the training or evaluation code. If a script hard-codes
  paths of the authors' cluster (the job generators in `torch/llmadd_torch/` do), change only the paths and say so.
- Ask before downloads of more than a few GB and before jobs expected to run longer than an hour.
- Record every command, its wall time and the numbers it gives in `REPRODUCTION.md`, next to the paper's numbers.

1. Environment. Python 3.11 with the versions the runs used (torch 2.10.0 for CUDA 12.8, transformers 4.57.6):
   `python3.11 -m venv .venv && . .venv/bin/activate && pip install -r results/331-env311/requirements311.txt`.
   Run `nvidia-smi` to see whether a GPU is available.

2. `check` (no GPU). `python tools/pt2main_summary.py` reads the stored per-seed results in `results/`. In the 53M
   table it must show a curve average of 18.4 for `pt2b-lr8e-3-none` and 69.9 for `pt2b-lr2e-3-win-ctx` (n=8 each):
   the paper's 18% → 70% (Figure 1d).

3. `pretrain` (one GPU; under an hour per run on an A100). Prepare the data and run the two arms of the Quick start
   in README.md for seed 0 (helix prior at learning rate 2e-3; no prior at 8e-3, its own best rate). From each
   `run_whole_<target>_seed0.json`, compute the curve average: for every entry of `curve`, the mean of `held_b11`,
   `held_b12`, `held_b13`, `held_b22`, `held_b23` and `held_b33`; then the mean over entries, times 100.
   Expected: with the prior about 70 (8-seed mean 69.9, spread about ±1 across seeds); without it about 18 but
   seed-dependent (8-seed mean 18.4, spread about ±7). The paper's claim is that the prior leads in every one of the
   8 paired seeds; repeat with `--seed 1` … `--seed 7` if time allows.

4. `finetune` (one GPU that fits OLMo-2-7B with LoRA). Fetch the 9B-token checkpoint and fine-tune as in Figure 1h:
   `python tools/fetch_olmo.py --revision stage1-step2000-tokens9B --dest models/OLMo-2-7B-9B`, then
   `python tools/manifold_train.py --model models/OLMo-2-7B-9B --out out/ft --mode lora --lam 20
   --geoms none,helix,helix_shuf --seeds 0 --train_pairs 7000 --epochs 10 --fixed_test --curve_every 50`.
   Report, per geometry, the first step at which held-out accuracy reaches 50% and 90%, and the final accuracy.
   Paper (3-seed means): with the helix 50% after 900 steps and 90% after 1,400, ending at 99.9%; without a prior
   50% after 4,150 steps, ending at 61.9%; the shuffled helix does not reach 50% within 10 epochs.

5. Anything else: the table under "Reproducing the paper" in README.md maps every figure and table to its scripts and
   stored runs; Table 1 has its own commands in "Six priors together (Table 1)". The data of the concept domains
   that are not built in come from `python tools/atlas_fetch_data.py`.

At the end, print a table with one row per item you ran: the paper's number, your number, and whether it is within
the seed-to-seed spread given above. List anything that did not run as documented.
