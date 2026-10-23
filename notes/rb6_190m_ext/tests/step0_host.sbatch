#!/bin/bash
#SBATCH --job-name=llmadd_rb190ext-step0host
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# rb6_190m_ext: host-matched step-0 references (CPU bf16 kernels differ across hosts at ~1e-5): none@6e-4 of seeds 0-3
# (the main batch's arm) and every new arm type, all in this one job on one host (refs_ext.py step0; table checks it)
set -u
SP=/scratch/yh6210/llmadd_torch/scale_work/rb190ext_scripts; PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python; W=/scratch/yh6210/llmadd_torch/scale_work
cd $SP
echo "start $(date) $(hostname) cpus $(nproc)"
printf '%s\n' "none 6e-4 0" "none 6e-4 1" "none 6e-4 2" "none 6e-4 3" "only-clock 6e-4 0" "all6-minus-clock 6e-4 0" \
    "all6-minus-clock 6e-4 2" "all6-lam1 1.2e-3 1" "only-numbers 1.2e-3 3" "none 1.92e-2 1" "none 9.6e-3 3" \
  | xargs -P 4 -L 1 bash -c '$0 refs_ext.py step0 $1 $2 $3' $PY > $W/rb190ext_step0_host.log 2>&1
echo "end $(date)"
