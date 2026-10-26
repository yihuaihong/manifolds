#!/bin/bash
#SBATCH --job-name=llmadd_rb190dim-cputests
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# rb6_190m_dim CPU tests before submission: --dry_inputs of every (arm, lr, seed) the generator can render (D1 none@3.84e-2
# s0-3; D2 none@{5 lrs} and all6-lam1@6e-4 at s4-7), step-0 lm on one host (seed-0 calibration, lr / arm independence,
# seeds 4-7), the reference table (-> scale_work/rb190dim_refs.json), then end-to-end tiny job runs of a D1 and two D2 cells
set -u
SP=/scratch/yh6210/llmadd_torch/scale_work/rb190dim_scripts; PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python; W=/scratch/yh6210/llmadd_torch/scale_work
cd $SP
echo "start $(date) $(hostname) cpus $(nproc)"
( { for s in 0 1 2 3; do echo "none 3.84e-2 $s"; done
    for s in 4 5 6 7; do for lr in 6e-4 4.8e-3 9.6e-3 1.92e-2 3.84e-2; do echo "none $lr $s"; done; echo "all6-lam1 6e-4 $s"; done; } \
  | xargs -P 8 -L 1 bash -c '$0 refs_dim.py dry $1 $2 $3' $PY > $W/rb190dim_dry.log 2>&1 ) &
( printf '%s\n' "none 6e-4 0" "none 3.84e-2 0" "none 6e-4 4" "all6-lam1 6e-4 4" "none 3.84e-2 4" "none 6e-4 5" "all6-lam1 6e-4 5" \
    "none 6e-4 6" "all6-lam1 6e-4 6" "none 6e-4 7" "all6-lam1 6e-4 7" "none 1.92e-2 7" \
  | xargs -P 4 -L 1 bash -c '$0 refs_dim.py step0 $1 $2 $3' $PY > $W/rb190dim_step0.log 2>&1 ) &
wait
echo "refs done $(date)"
$PY refs_dim.py table --write > $W/rb190dim_table.txt 2>&1
tail -n 1 $W/rb190dim_table.txt
$PY e2e_dim.py none 3.84e-2 1 > $W/rb190dim_e2e_none384_s1.log 2>&1 &
$PY e2e_dim.py all6-lam1 6e-4 5 > $W/rb190dim_e2e_all6_s5.log 2>&1 &
$PY e2e_dim.py none 1.92e-2 6 > $W/rb190dim_e2e_none192_s6.log 2>&1 &
wait
grep -h "^E2E\|^RUN PRIORS" $W/rb190dim_e2e_*.log
echo "end $(date)"
