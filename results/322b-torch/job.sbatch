#!/bin/bash
#SBATCH --job-name=llmadd_322b
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# E1 control: shuffled helix + digit at 9B/7000 (Torch copy of Burst task 322b; code = llmadd-burst commit in code/COMMIT)
set -eu
cd /scratch/yh6210/llmadd_torch/runs/322b-torch/code/tools
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache
export PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/yh6210/miniconda3/envs/open_r1/bin/python manifold_train.py --model /scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step2000-tokens9B --out /scratch/yh6210/llmadd_torch/runs/322b-torch/out --mode lora --lam 20 --fixed_test --train_pairs 7000 --save_preds --epochs 3 --geoms helix_shuf,digit --seeds 0,1,2
