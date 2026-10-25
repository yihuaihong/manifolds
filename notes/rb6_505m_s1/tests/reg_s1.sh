#!/bin/bash
# Amendment S1 regression: the pre-S1 tools (scale-ladder 04586fb: rb6scale_summary e807e43be36b) vs the S1 worktree
# file, the same inputs, stdout + stderr and JSON compared byte for byte:
#   1. the rb6_190m read-out (rb6_190m/AUDIT.md's command: rb6_190m/out + the 16 53M F88 Burst dirs)
#   2. the rb6_190m_ext read-out (rb6scale_ext_summary.py b5f3b5e6156e, which imports rb6scale_summary)
set -u
SP=/state/partition1/job-20284046/claude-4550886/-home-yh6210/74cbe2e4-1709-4a6a-b82e-817d33fbdd73/scratchpad/s1
OLD=$SP/old/tools; NEW=/scratch/yh6210/llmadd-burst-scale/tools; PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
W=$SP/reg; rm -rf "$W"; mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
cd /scratch/yh6210/llmadd-burst-ro
B53="results/401-rb6pb-* results/404-rb6pb-* results/430-rb6pb-* results/431-rb6pb-* results/419-rb6pb-* results/415-rb6pb-* results/432-rb6pb-* results/433-rb6pb-* results/438-rb6mb-* results/444-rb6mb-* results/450-rb6mb-* results/456-rb6mb-* results/46[6-9]-rb6mb-*"
ok=1
for v in old new; do
  T=$OLD; [ $v = new ] && T=$NEW
  $PY $T/rb6scale_summary.py /scratch/yh6210/llmadd_torch/runs/rb6_190m/out $B53 --json $W/r190_$v.json > $W/r190_$v.txt 2>&1; echo "rc $?" >> $W/r190_$v.txt
  $PY $T/rb6scale_ext_summary.py /scratch/yh6210/llmadd_torch/runs/rb6_190m/out /scratch/yh6210/llmadd_torch/runs/rb6_190m_ext/out --json $W/ext_$v.json > $W/ext_$v.txt 2>&1; echo "rc $?" >> $W/ext_$v.txt
done
for k in r190 ext; do
  if cmp -s $W/${k}_old.txt $W/${k}_new.txt && cmp -s $W/${k}_old.json $W/${k}_new.json; then
    echo "PASS $k: text ($(wc -l < $W/${k}_new.txt) lines, $(tail -1 $W/${k}_new.txt)) and JSON ($(wc -c < $W/${k}_new.json) bytes) byte-identical old vs new"
  else
    echo "FAIL $k: differ"; ok=0
  fi
done
echo "summary shas: old $(sha256sum $OLD/rb6scale_summary.py | cut -c1-12) new $(sha256sum $NEW/rb6scale_summary.py | cut -c1-12); ext old $(sha256sum $OLD/rb6scale_ext_summary.py | cut -c1-12) new $(sha256sum $NEW/rb6scale_ext_summary.py | cut -c1-12)"
[ $ok = 1 ] && echo "S1 REGRESSION PASS" || echo "S1 REGRESSION FAIL"
