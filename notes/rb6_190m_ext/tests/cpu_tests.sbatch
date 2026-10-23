#!/bin/bash
#SBATCH --job-name=llmadd_rb190ext-cputests
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# rb6_190m_ext CPU tests before submission: --dry_inputs of all 24 (arm, lr, seed), step-0 lm of every arm type,
# the end-to-end tiny job runs of every cell type, then the reference table (refs_ext.py, e2e_ext.py)
set -u
SP=/scratch/yh6210/llmadd_torch/scale_work/rb190ext_scripts; PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python; W=/scratch/yh6210/llmadd_torch/scale_work
cd $SP
echo "start $(date) $(hostname) cpus $(nproc)"
( for s in 0 1 2 3; do for c in "only-clock 6e-4" "all6-minus-clock 6e-4" "all6-lam1 1.2e-3" "only-numbers 1.2e-3" "none 9.6e-3" "none 1.92e-2"; do echo "$c $s"; done; done \
  | xargs -P 8 -L 1 bash -c '$0 refs_ext.py dry $1 $2 $3' $PY > $W/rb190ext_dry.log 2>&1 ) &
( printf '%s\n' "only-clock 6e-4 0" "all6-minus-clock 6e-4 0" "all6-minus-clock 6e-4 2" "all6-lam1 1.2e-3 1" \
    "only-numbers 1.2e-3 3" "none 1.92e-2 1" "none 9.6e-3 3" \
  | xargs -P 4 -L 1 bash -c '$0 refs_ext.py step0 $1 $2 $3' $PY > $W/rb190ext_step0.log 2>&1 ) &
wait
echo "refs done $(date)"
$PY e2e_ext.py all6-minus-clock 6e-4 1 > $W/rb190ext_e2e_minusclock_s1.log 2>&1 &
$PY e2e_ext.py only-clock 6e-4 2 > $W/rb190ext_e2e_onlyclock_s2.log 2>&1 &
$PY e2e_ext.py all6-lam1 1.2e-3 2 > $W/rb190ext_e2e_all6_s2.log 2>&1 &
$PY e2e_ext.py only-numbers 1.2e-3 1 > $W/rb190ext_e2e_onlynum_s1.log 2>&1 &
$PY e2e_ext.py none 1.92e-2 2 > $W/rb190ext_e2e_none_s2.log 2>&1 &
$PY e2e_ext.py all6-minus-clock 6e-4 2 > $W/rb190ext_e2e_minusclock_s2.log 2>&1 &
wait
$PY refs_ext.py table > $W/rb190ext_table.txt 2>&1
echo "end $(date)"
