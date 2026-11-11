#!/bin/bash
#SBATCH --job-name=llmadd_s1-curve-21B-n7000-helix-s012
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# broader-evaluation batch; code = llmadd-burst commit in code/COMMIT
set -eu
cd /scratch/yh6210/llmadd_torch/runs/s1-curve-21B-n7000-helix-s012/code/tools
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache
export PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python manifold_train.py --model /scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step5000-tokens21B --out /scratch/yh6210/llmadd_torch/runs/s1-curve-21B-n7000-helix-s012/out --mode lora --lam 20 --save_preds --fixed_test --train_pairs 7000 --epochs 10 --curve_every 50 --gpt3_arith /scratch/yh6210/llmadd_torch/data/arithmetic --geoms helix --seeds 0,1,2
