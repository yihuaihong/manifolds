#!/bin/bash
#SBATCH --job-name=llmadd_fullft
#SBATCH --account=<slurm-account>
#SBATCH --partition=h200_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# Torch side: full fine-tuning (fp32 master + AdamW, ~120 GB -> one H200) with the manifold loss.
#   sbatch --job-name=fullft_s1-10000 torch/full_ft.sbatch <revision> <run-name>
# Restartable: finished (geometry, seed) runs are skipped, so a requeue continues where it stopped.
set -eu
REV=$1; NAME=$2; LAM=${3:-1.0}
source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh && conda activate open_r1
export PYTORCH_ALLOC_CONF=expandable_segments:True
RUN=/scratch/yh6210/llmadd_torch/runs/$NAME; mkdir -p "$RUN"
[ -d "$RUN/code" ] || { mkdir -p "$RUN/code"; git -C /home/yh6210/Research_Projects/llmadd-burst archive HEAD | tar -x -C "$RUN/code"; }
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$RUN/code/tools"
python manifold_train.py --model "/scratch/yh6210/llmadd_models/OLMo-2-1124-7B/$REV" --out "$RUN/out" --mode full \
  --geoms ${GEOMS:-none,helix,helix_shuf,digit,digit_shuf} --seeds 0,1,2 --lam $LAM
