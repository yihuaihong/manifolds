#!/bin/bash
# Default-path regression: origin/main manifold_train.py (llmadd-burst-ro, a740848) vs the ft999 branch, tiny random OLMo-2 on CPU.

source ~/.bashrc >/dev/null 2>&1; conda activate open_r1; set -eu
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
D=/scratch/yh6210/llmadd_torch/runs/ft999_dev
TINY=$D/tiny_olmo2
ARGS="--model $TINY --mode lora --geoms none,helix,helix3_shuf --seeds 0 --lam 20 --train_pairs 200 --epochs 1 --curve_every 5 --save_states_every 5 --save_preds --fixed_test"
rm -rf $D/reg_old $D/reg_new
(cd /scratch/yh6210/llmadd-burst-ro/tools && python manifold_train.py $ARGS --out $D/reg_old > $D/reg_old.log 2>&1) &
P1=$!
(cd /scratch/yh6210/llmadd-burst-ft999/tools && python manifold_train.py $ARGS --out $D/reg_new > $D/reg_new.log 2>&1) &
P2=$!
wait $P1; echo "old exit $?"
wait $P2; echo "new exit $?"
