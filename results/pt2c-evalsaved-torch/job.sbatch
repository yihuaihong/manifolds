#!/bin/bash
#SBATCH --job-name=llmadd_pt2c-evalsaved
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# offline re-evaluation of the saved pt2c models incl. the strict (swap-unseen) subset; code = repo HEAD (tools/pt_eval_saved.py)
set -eu
cd /home/yh6210/Research_Projects/llmadd-burst/tools
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
/scratch/yh6210/miniconda3/envs/open_r1/bin/python pt_eval_saved.py --out /scratch/yh6210/llmadd_torch/runs/pt2c-evalsaved/eval.json /scratch/yh6210/llmadd_torch/runs/pt2c/out
