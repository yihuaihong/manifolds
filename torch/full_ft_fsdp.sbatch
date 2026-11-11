#!/bin/bash
#SBATCH --job-name=llmadd_fsdp
#SBATCH --account=<slurm-account>
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# Torch side: full fine-tuning on 2 x 80 GB (A100 or H100) with FSDP + standard AdamW (fp32 master sharded).
#   sbatch --job-name=fsdp_s1-150000 torch/full_ft_fsdp.sbatch <revision> <run-name>
set -eu
REV=$1; NAME=$2; LAM=${3:-1.0}
source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh && conda activate open_r1
export PYTORCH_ALLOC_CONF=expandable_segments:True
RUN=/scratch/yh6210/llmadd_torch/runs/$NAME; mkdir -p "$RUN"
[ -d "$RUN/code" ] || { mkdir -p "$RUN/code"; git -C /home/yh6210/Research_Projects/llmadd-burst archive HEAD | tar -x -C "$RUN/code"; }
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$RUN/code/tools"
python -m torch.distributed.run --nproc_per_node=2 manifold_train_fsdp.py --model "/scratch/yh6210/llmadd_models/OLMo-2-1124-7B/$REV" \
  --out "$RUN/out" --optim adamw --geoms ${GEOMS:-none,helix,helix_shuf,digit,digit_shuf} --seeds 0,1,2 --lam $LAM
