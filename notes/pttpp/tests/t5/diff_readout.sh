#!/bin/bash
# T5: the deployed T4 read-out (runs/pttpp/readout, 0b74254) vs the T5 read-out (ladder-tpp worktree) on the selftest's
# synthetic roots (kept with --keep): the only differences must be the Q1 header, the added 'incl. NONE AT FLOOR' lines
# and, where a NONE AT FLOOR cell flips a SHOWN slope, that verdict (+ the Q1 combination label). Synthetic data only.
#   bash diff_readout.sh <selftest work dir> <T4 tools dir> <T5 tools dir>
set -u
W=$1; T4=$2; T5=$3
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
for sc in size_effect no_effect floor_censored lr_flip none_floor floor_same floor_noeff own_after_train bridge_fail; do
  [ -d "$W/$sc" ] || { echo "$sc: no root"; continue; }
  (cd $T4 && $PY pttpp_summary.py --root "$W/$sc" > "$W/$sc.T4.txt" 2>&1)
  (cd $T5 && $PY pttpp_summary.py --root "$W/$sc" > "$W/$sc.T5.txt" 2>&1)
  d=$(diff "$W/$sc.T4.txt" "$W/$sc.T5.txt")
  other=$(echo "$d" | grep '^[<>]' | grep -v 'incl\. NONE AT FLOOR cells (T5' | grep -v '## Q1 size effect at fixed tpp' \
          | grep -v 'none-at-floor cells' | grep -v -- '-> 7% tpp .*(none-at-floor cells)\|^< .*-> 7% tpp 0.745' | grep -v '=> \[primary = area\] Q1 7%')
  echo "$sc: $(echo "$d" | grep -c '^>') lines T5 only, $(echo "$d" | grep -c '^<') lines T4 only; unexplained: $([ -z "$other" ] && echo none || echo "$other")"
  echo "$d" | grep '^[<>]' | grep -v 'incl\. NONE AT FLOOR cells (T5 sensitivity): no such cell' | grep -v '## Q1 size effect' | cut -c1-230 | sed 's/^/    /'
done
