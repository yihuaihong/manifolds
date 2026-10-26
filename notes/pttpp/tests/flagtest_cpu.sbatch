#!/bin/bash
#SBATCH --job-name=llmadd_tpp-flagtest-cpu
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# new-flag tests of the pttpp batch (flagtest_tpp.py), one process per trainer snapshot; nothing submitted
set -u
W=/scratch/yh6210/llmadd_torch/scale_work/tpp; PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
cd $W
$PY flagtest_tpp.py code7 > flagtest_code7.log 2>&1 &
$PY flagtest_tpp.py code2 > flagtest_code2.log 2>&1 &
wait
grep -h "^FLAGTEST" flagtest_code7.log flagtest_code2.log
