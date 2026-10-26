#!/bin/bash
# regression of the two existing 190M read-outs (rb6_190m: its AUDIT command; rb6_190m_ext: its AUDIT command):
# text (stdout + stderr) and JSON into scale_work/rb190dim_reg/<label>_*; run from the scale-ladder worktree's tools/
set -u
L=$1; W=/scratch/yh6210/llmadd_torch/scale_work/rb190dim_reg; PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
cd /scratch/yh6210/llmadd-burst-scale/tools
echo "HEAD $(git rev-parse --short HEAD) $(git status --porcelain | tr '\n' ' ')" > $W/${L}_head.txt
RO=/scratch/yh6210/llmadd-burst-ro
$PY rb6scale_summary.py /scratch/yh6210/llmadd_torch/runs/rb6_190m/out \
  $RO/results/401-rb6pb-* $RO/results/404-rb6pb-* $RO/results/430-rb6pb-* $RO/results/431-rb6pb-* \
  $RO/results/419-rb6pb-* $RO/results/415-rb6pb-* $RO/results/432-rb6pb-* $RO/results/433-rb6pb-* \
  $RO/results/438-rb6mb-* $RO/results/444-rb6mb-* $RO/results/450-rb6mb-* $RO/results/456-rb6mb-* $RO/results/46[6-9]-rb6mb-* \
  --json $W/${L}_rb6_190m.json > $W/${L}_rb6_190m.txt 2>&1; echo "rb6_190m rc $?"
$PY rb6scale_ext_summary.py /scratch/yh6210/llmadd_torch/runs/rb6_190m/out /scratch/yh6210/llmadd_torch/runs/rb6_190m_ext/out \
  --json $W/${L}_rb6_190m_ext.json > $W/${L}_rb6_190m_ext.txt 2>&1; echo "rb6_190m_ext rc $?"
