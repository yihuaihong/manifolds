#!/bin/bash
# T5: after the deployment, run the selection hook once by hand for every group (as T2-T4: catch-up for the jobs that
# ended while HOLD was set). Prints the hook rc and the new select-log lines without their DECISION lines (whose basis
# holds selection-seed numbers: the batch stays blind); DECISION lines are only counted.
R=/scratch/yh6210/llmadd_torch/runs/pttpp
for g in tpp7-93m-t0.745 tpp7-190m-t0.745 tpp7-294m-t0.745 tpp7-1b-t0.745 tpp7-93m-t0.318 tpp7-190m-t0.318 tpp7-294m-t0.318 tpp7-505m-t0.318 tpp2-190m-t0.745 tpp2-294m-t0.745; do
  L=$R/select_$g.log
  n0=$( [ -e $L ] && wc -l < $L || echo 0)
  out=$(bash $R/maybe_select.sh $g 2>&1); rc=$?
  new=$( [ -e $L ] && tail -n +$((n0 + 1)) $L || true)
  echo "$g: hook rc $rc; stdout: ${out:-<none>}; new log lines $(echo -n "$new" | grep -c '^\[') (DECISION $(echo -n "$new" | grep -c 'DECISION'))"
  echo "$new" | grep '^\[' | grep -v DECISION | cut -c1-200 | sed 's/^/    /'
  echo "$new" | grep -E 'Submitted batch job' | sed 's/^/    /'
done
