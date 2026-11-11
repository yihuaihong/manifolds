#!/bin/bash
# CPU smoke of the three 9B seed-0 job scripts (the exact job command, tiny random OLMo-2, 4 steps) + lambda A/B
# (helix3 at lam 0 must reproduce none; lam 20 must not) + default-path regression on the SUBMITTED snapshot code.
source ~/.bashrc >/dev/null 2>&1; conda activate open_r1; set -eu
B=/scratch/yh6210/llmadd_torch; R=$B/runs/ft999; D=$B/runs/ft999_dev
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
rm -rf $D/smoke $D/reg_snap; mkdir -p $D/smoke
# regression: origin/main vs the snapshot that the jobs run (background)
(cd $R/code/tools && HF_HOME=$B/hf_cache PYTHONPATH=$B/pylib python manifold_train.py --model $D/tiny_olmo2 --mode lora \
  --geoms none,helix,helix3_shuf --seeds 0 --lam 20 --train_pairs 200 --epochs 1 --curve_every 5 --save_states_every 5 \
  --save_preds --fixed_test --out $D/reg_snap > $D/reg_snap.log 2>&1; echo "reg_snap exit $?") &
P=$!
for arm in none helix3 helix3_shuf; do
  env -u SLURM_JOB_ID FT999_SMOKE=1 FT999_SMOKE_MODEL=$D/tiny_olmo2 FT999_SMOKE_OUT=$D/smoke \
    bash $R/job_9B-$arm-s0.sbatch > $D/smoke/job_9B-$arm-s0.log 2>&1
  echo "smoke $arm exit $?"
done
# lambda A/B through the same snapshot and the same smoke arguments
(cd $R/code/tools && HF_HOME=$B/hf_cache PYTHONPATH=$B/pylib python manifold_train.py --model $D/tiny_olmo2 --out $D/smoke/lam0-helix3-s0 \
  --mode lora --lam 0 --save_preds --fixed_test --add3 --train_pairs 7000 --test_pairs 2000 --epochs 10 --curve_every 50 --grad_check_every 250 \
  --save_states_every 50 --geoms helix3 --seeds 0 --train_pairs 64 --test_pairs 200 --epochs 1 --curve_every 2 --save_states_every 2 --grad_check_every 2 \
  > $D/smoke/lam0-helix3-s0.log 2>&1)
echo "lam0 exit $?"
wait $P
