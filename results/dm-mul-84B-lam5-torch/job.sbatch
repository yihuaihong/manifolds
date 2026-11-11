#!/bin/bash
#SBATCH --job-name=llmadd_dm-mul-84B-lam5
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# broader-evaluation batch; code = llmadd-burst commit in code/COMMIT
set -eu
cd /scratch/yh6210/llmadd_torch/runs/dm-mul-84B-lam5/code/tools
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache
export PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python manifold_train.py --model /scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step20000-tokens84B --out /scratch/yh6210/llmadd_torch/runs/dm-mul-84B-lam5/out --mode lora --lam 20 --save_preds --task mul --train_pairs 900 --test_pairs 500 --lm_val /scratch/yh6210/llmadd_torch/pretrain_data/val.bin --gpt3_arith /scratch/yh6210/llmadd_torch/data/arithmetic --lam 5 --geoms none,helix,digit_fourier --seeds 0,1,2
