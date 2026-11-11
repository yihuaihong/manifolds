#!/bin/bash
#SBATCH --job-name=llmadd_pt7-helix-s2
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# pretraining with ~7% arithmetic tokens (4 lines per sequence): does the prior make addition emerge earlier? (code = commit in code/COMMIT)
set -eu
cd /scratch/yh6210/llmadd_torch/runs/pt7/code/tools
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py --data /scratch/yh6210/llmadd_torch/pretrain_data --tokenizer /scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step2000-tokens9B --out /scratch/yh6210/llmadd_torch/runs/pt7/out --arm helix --seed 2 --arith_lines 4 --tokens 3e8 --eval_every 250
