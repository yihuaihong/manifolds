#!/bin/bash
#SBATCH --job-name=llmadd_tpp-refs-cpu
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# CPU references of the pttpp jobs (refs_tpp.py: --dry_inputs with the jobs' own ARGS, step-0 lm, GPU calibration) and the
# new-flag tests (flagtest_tpp.py); no GPU, nothing is submitted
set -u
W=/scratch/yh6210/llmadd_torch/scale_work/tpp; PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
cd $W
echo "start $(date) $(hostname) cpus $(nproc)"
( $PY flagtest_tpp.py > flagtest.log 2>&1 ) &
( { for a in none win-ctx; do for s in 0 1; do for g in tpp7-93m-t0.745 tpp7-294m-t0.745 tpp7-1b-t0.745 tppb-505m-3e8; do echo "$g $a $s"; done; done; echo "tpp7-93m-t0.318 $a 0"; done;
    for g in tpp2-190m-t0.745 tpp2-294m-t0.745; do for a in none win-ctx; do for s in 0 1 2 3 4 5 6 7; do echo "$g $a $s"; done; done; done; } \
  | xargs -P 6 -L 1 bash -c '$0 refs_tpp.py dry $1 $2 $3' $PY > refs_dry.log 2>&1 ) &
( { for g in tpp7-93m-t0.745 tpp7-294m-t0.745 tpp2-294m-t0.745; do for s in 0 1 2 3 4 5 6 7; do echo "$g none $s"; done; echo "$g win-ctx 0"; done;
    for s in 0 1 2 3 4 5 6 7; do echo "tpp2-190m-t0.745 none $s"; done; echo "tpp7-190m-t0.745 none 0"; echo "tpp7-190m-t0.745 none 1";
    echo "tpp7-505m-t0.318 none 0"; } \
  | xargs -P 4 -L 1 bash -c '$0 refs_tpp.py step0 $1 $2 $3' $PY > refs_step0.log 2>&1 ) &
wait
$PY refs_tpp.py table > refs_table.txt 2>&1
echo "end $(date)"
