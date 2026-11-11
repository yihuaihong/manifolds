#!/bin/bash
#SBATCH --job-name=llmadd_dl
#SBATCH --account=<slurm-account>
#SBATCH --partition=cs
#SBATCH --qos=cpu168
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# Torch side: download + bf16-convert OLMo-2 checkpoints for the full-FT runs (CPU only).
#   sbatch torch/dl.sbatch <revision> [<revision> ...]
set -eu
source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh && conda activate open_r1
CODE=/scratch/yh6210/llmadd_torch/code_dl; rm -rf "$CODE"; mkdir -p "$CODE"
git -C /home/yh6210/Research_Projects/llmadd-burst archive HEAD | tar -x -C "$CODE"
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache
for rev in "$@"; do
  python "$CODE/tools/fetch_olmo.py" --revision "$rev" --dest "/scratch/yh6210/llmadd_models/OLMo-2-1124-7B/$rev"
done
du -sh /scratch/yh6210/llmadd_models/OLMo-2-1124-7B/*
