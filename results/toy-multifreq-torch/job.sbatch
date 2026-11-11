#!/bin/bash
#SBATCH --job-name=llmadd_toy-multifreq
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# toy grokking rerun: is early harm caused by the rank-2 circle target? (code = commit in code/COMMIT)
set -eu
cd /scratch/yh6210/llmadd_torch/runs/toy-multifreq/code/tools
for s in 0 1 2 3 4; do /scratch/yh6210/miniconda3/envs/open_r1/bin/python toy_grok.py --out /scratch/yh6210/llmadd_torch/runs/toy-multifreq/out --seeds $s --freqs 1,2,3,4,5 --lam 1 --arms none,always,w0,w1000,w2000,w4000 > /scratch/yh6210/llmadd_torch/runs/toy-multifreq/log_seed$s.txt 2>&1 & done; wait
/scratch/yh6210/miniconda3/envs/open_r1/bin/python -c "import toy_grok, pathlib; toy_grok.summarize(pathlib.Path('/scratch/yh6210/llmadd_torch/runs/toy-multifreq/out'))"
