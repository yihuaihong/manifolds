#!/bin/bash
# pt2sz stage A submission (runs/pt2sz/AUDIT.md "Submission"), run by the coordinator. Stage A only: the 32 294M jobs first
# (the longer runs), then the 32 93M jobs, each seed-major. Without --go: the checks, then the sbatch lines (nothing is
# submitted). With --go: the same checks, then sbatch in that order, one SUBMITTED_<run> flag per job (the sbatch output);
# stops at the first failure.
# Checks: no runs/pt2sz/HOLD; the generator's --check (every stage-A job file = its rendering, the trainer snapshot, the
# frozen shas, the known answers vs pt2s505 / pt2same); 64 stage-A runs; none submitted yet; /scratch (myquota) below 97%.
# Stages X / C are submitted by hand after select_pt2sz.py --write (it prints the generator commands).
set -euo pipefail
P=/scratch/yh6210/llmadd_torch/runs/pt2sz
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
GEN=/scratch/yh6210/llmadd_torch/make_jobs_pt2sz.py
SBATCH=sbatch   # (tests/submit_test.sh runs a scratch copy with this line pointing to a fake)
DISK_MAX=97
GO=0
case "${1:-}" in
  "") ;;
  --go) GO=1 ;;
  *) echo "usage: $0 [--go]"; exit 2 ;;
esac
[ $# -le 1 ] || { echo "usage: $0 [--go]"; exit 2; }
[ -e "$P/HOLD" ] && { echo "HOLD present ($P/HOLD): not submitting"; exit 1; }
"$PY" "$GEN" --check --stage A > /dev/null || { echo "## generator --check FAILED: nothing submitted"; exit 1; }
NAMES="$("$PY" "$GEN" --names --stage A --size 294m) $("$PY" "$GEN" --names --stage A --size 93m)"
N=$(echo $NAMES | wc -w)
[ "$N" = 64 ] || { echo "## $N stage-A runs, want 64: nothing submitted"; exit 1; }
for n in $NAMES; do
  [ -e "$P/SUBMITTED_$n" ] && { echo "## $n already submitted ($P/SUBMITTED_$n): nothing submitted"; exit 1; }
  [ -e "$P/job_$n.sbatch" ] || { echo "## job_$n.sbatch missing: nothing submitted"; exit 1; }
done
USE=$(myquota 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | sed -n 's/^\/scratch [^(]*(\([0-9.]*\)%).*/\1/p' | head -n 1)
[ -n "$USE" ] || { echo "## myquota could not be parsed: nothing submitted"; exit 1; }
awk -v u="$USE" -v m="$DISK_MAX" 'BEGIN { exit !(u < m) }' || { echo "## DISK: /scratch at $USE% >= $DISK_MAX%: nothing submitted"; exit 1; }
echo "# checks passed: generator --check OK, 64 stage-A runs (294M first), none submitted, /scratch $USE% < $DISK_MAX%"
for n in $NAMES; do
  if [ "$GO" = 1 ]; then
    out=$("$SBATCH" "$P/job_$n.sbatch") || { echo "## sbatch FAILED for $n: stopping"; exit 1; }
    echo "$out" > "$P/SUBMITTED_$n"
    echo "$n $out"
  else
    echo "sbatch $P/job_$n.sbatch"
  fi
done
[ "$GO" = 1 ] && echo "# submitted 64 stage-A jobs $(date -Is)" || echo "# dry: nothing submitted (--go submits)"
