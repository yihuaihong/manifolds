#!/bin/bash
# Regression of the existing read-outs (runs/pttpp/AUDIT.md "Tests"): every read-out the pttpp branch (ladder-tpp) carries,
# run from the ladder-tpp worktree and from the worktree / snapshot it was frozen in, on the real roots; outputs (text and
# --json) must be byte-identical. ladder-tpp adds tools/pttpp_summary.py and tools/pttpp_selftest.py and changes nothing else.
#   bash regress_tpp.sh <out dir>
set -u
O=${1:?out dir}; mkdir -p "$O"
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
NEW=/scratch/yh6210/llmadd-burst-tpp; OLD=/scratch/yh6210/llmadd-burst-scale; RO=/scratch/yh6210/llmadd-burst-ro
RUNS=/scratch/yh6210/llmadd_torch/runs
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib
fail=0
cmpout() {   # name new old; the worktree path and the side's own --json name are normalised (printed by tracebacks / flags)
  sed -e "s#$NEW#WORKTREE#g; s#$OLD#WORKTREE#g; s#\.NEW\.json#.SIDE.json#g; s#\.OLD\.json#.SIDE.json#g" "$2" > "$2.norm"
  sed -e "s#$NEW#WORKTREE#g; s#$OLD#WORKTREE#g; s#\.NEW\.json#.SIDE.json#g; s#\.OLD\.json#.SIDE.json#g" "$3" > "$3.norm"
  if cmp -s "$2.norm" "$3.norm"; then echo "IDENTICAL $1 ($(wc -l < "$2") lines$(cmp -s "$2" "$3" || echo ', after path normalisation'))"; else echo "DIFFERS $1"; fail=1; fi
}
echo "ladder-tpp vs scale-ladder: changed files under tools/:"
git -C $NEW diff --stat scale-ladder -- tools/ | tail -3
for side in NEW OLD; do
  W=${!side}
  (cd $W && $PY tools/pt2main_summary.py > "$O/pt2main_burst.$side.txt" 2>&1)
  (cd $W && $PY tools/pt2main_summary.py --gen /scratch/yh6210/llmadd_torch/make_jobs_pt2s505.py --res $RUNS/pt2s505/out \
     --json "$O/pt2s505.$side.json" > "$O/pt2s505.$side.txt" 2>&1)
  (cd $W && $PY tools/ptscale_summary.py > "$O/ptscale_scaleladder.$side.txt" 2>&1)
  (cd $RO && $PY $W/tools/rb6scale_summary.py $RUNS/rb6_190m/out \
     results/401-rb6pb-* results/404-rb6pb-* results/430-rb6pb-* results/431-rb6pb-* \
     results/419-rb6pb-* results/415-rb6pb-* results/432-rb6pb-* results/433-rb6pb-* \
     results/438-rb6mb-* results/444-rb6mb-* results/450-rb6mb-* results/456-rb6mb-* results/46[6-9]-rb6mb-* \
     --json "$O/rb6scale_190m.$side.json" > "$O/rb6scale_190m.$side.txt" 2>&1)
  (cd $W && $PY tools/rb6scale_ext_summary.py $RUNS/rb6_190m/out $RUNS/rb6_190m_ext/out --json "$O/rb6ext.$side.json" > "$O/rb6ext.$side.txt" 2>&1)
  (cd $W && $PY tools/rb6scale_dim_summary.py readout $RUNS/rb6_190m/out $RUNS/rb6_190m_ext/out $RUNS/rb6_190m_dim/out \
     --picks $RUNS/rb6_190m_dim/picks.json --json "$O/rb6dim.$side.json" > "$O/rb6dim.$side.txt" 2>&1)
done
for n in pt2main_burst.txt pt2s505.txt pt2s505.json ptscale_scaleladder.txt rb6scale_190m.txt rb6scale_190m.json rb6ext.txt rb6ext.json rb6dim.txt; do
  b=${n%.*}; e=${n##*.}
  cmpout "$n" "$O/$b.NEW.$e" "$O/$b.OLD.$e"
done
if [ -e $RUNS/rb6_190m_dim/picks.json ]; then cmpout rb6dim.json "$O/rb6dim.NEW.json" "$O/rb6dim.OLD.json"
else echo "rb6dim: both sides stop at the missing rb6_190m_dim/picks.json (that batch is not selected yet): compared as text only"; fi
# ptlr_1b: its frozen read-out snapshot (ladder-1b 2ce723a, ptscale_summary.py 404f4466e3bb) vs the ladder-1b worktree
(cd $RUNS/ptlr_1b/readout/tools && $PY ptscale_summary.py --root $RUNS > "$O/ptlr1b.snapshot.txt" 2>&1)
(cd /scratch/yh6210/llmadd-burst-1b && $PY tools/ptscale_summary.py --root $RUNS > "$O/ptlr1b.ladder1b.txt" 2>&1)
cmpout ptlr1b "$O/ptlr1b.snapshot.txt" "$O/ptlr1b.ladder1b.txt"
echo "REGRESSION $([ $fail = 0 ] && echo PASS || echo FAIL)"
