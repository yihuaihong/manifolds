#!/bin/bash
#SBATCH --job-name=llmadd_pt-helix_acc-s2
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# dynamic window for the pretraining prior (code = commit in code/COMMIT)
set -eu
cd /scratch/yh6210/llmadd_torch/runs/pt-helix_acc/code/tools
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pretrain_helix.py --data /scratch/yh6210/llmadd_torch/pretrain_data --tokenizer /scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step2000-tokens9B --out /scratch/yh6210/llmadd_torch/runs/pt-helix_acc/out --arm helix_acc --gate_start 0.1 --gate_thr 0.1 --cka_release 0.01 --cka_patience 200 --seed 2 --tokens 3e8 --eval_every 500
