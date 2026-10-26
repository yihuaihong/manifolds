#!/bin/bash
#SBATCH --job-name=llmadd_tpp-e2e-cpu
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# end-to-end CPU run of rendered pttpp job scripts (e2e_tpp.py; tiny overrides, fake nvidia-smi, stub hook); nothing submitted
set -u
W=/scratch/yh6210/llmadd_torch/scale_work/tpp; PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
cd $W
for c in l40s-A-none b200-C-win a100-A-win a100-C-none l40s-A-none-bad; do
  $PY e2e_tpp.py $c > e2e_$c.log 2>&1 &
done
wait
grep -h "^E2E " e2e_*.log
