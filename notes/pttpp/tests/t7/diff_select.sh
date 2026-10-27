#!/bin/bash
# T7: the deployed T6 read-out (runs/pttpp/readout, = fd8901b's tools) vs the T7 read-out (worktree) on the T5 selftest's
# kept synthetic roots: the full read-out and --select (3 groups) must be byte-identical (no root has a REOPEN after a STOP).
#   bash diff_select.sh <kept selftest work dir> <T6 tools> <T7 tools>
set -u
W=$1; T6=$2; T7=$3; O=$(cd "$(dirname "$0")" && pwd)/diff_select
mkdir -p $O
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
n=0; bad=0
for sc in size_effect no_effect floor_censored lr_flip none_floor floor_same floor_noeff own_after_train bridge_fail reopen reopen_c integrity; do
  [ -d "$W/$sc" ] || { echo "$sc: no root"; continue; }
  (cd $T6 && $PY pttpp_summary.py --root "$W/$sc" > $O/$sc.T6.txt 2>&1)
  (cd $T7 && $PY pttpp_summary.py --root "$W/$sc" > $O/$sc.T7.txt 2>&1)
  n=$((n + 1)); cmp -s $O/$sc.T6.txt $O/$sc.T7.txt && echo "IDENTICAL full $sc ($(wc -l < $O/$sc.T6.txt) lines)" || { echo "DIFFERS full $sc"; bad=1; }
  for g in tpp7-93m-t0.745 tpp7-294m-t0.745 tpp7-1b-t0.745; do
    (cd $T6 && $PY pttpp_summary.py --root "$W/$sc" --select $g --json $O/$sc.$g.T6.json > $O/$sc.$g.T6.txt 2>&1)
    (cd $T7 && $PY pttpp_summary.py --root "$W/$sc" --select $g --json $O/$sc.$g.T7.json > $O/$sc.$g.T7.txt 2>&1)
    n=$((n + 1)); { cmp -s $O/$sc.$g.T6.txt $O/$sc.$g.T7.txt && cmp -s $O/$sc.$g.T6.json $O/$sc.$g.T7.json; } || { echo "DIFFERS select $sc $g"; bad=1; }
  done
done
echo "DIFF-SELECT $([ $bad = 0 ] && echo PASS || echo FAIL) ($n comparisons)"
