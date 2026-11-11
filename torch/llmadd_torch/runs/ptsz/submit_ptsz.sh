#!/bin/bash
# ptsz stage A submission (runs/ptsz/AUDIT.md "Submission"). Stage A only, 128 L40S jobs, in this order: 731m (the long
# pole), 391m, 294m, 136m; each size arm-major, lr ascending, seed (the generator's rendering order). Without --go: checks,
# then prints the sbatch lines. With --go: the same checks, then sbatch in that order, one SUBMITTED_<run> flag per job (the
# sbatch output); stops at the first failure.
# Checks: no runs/ptsz/HOLD; the generator's --check (every stage-A job file = its rendering, the trainer snapshot, the
# frozen shas, the step-0 references); the 731m smoke passed (or --no-smoke); 32 stage-A runs per size, every job
# --nice=100 on l40s_public (amendment N1); no stage-A run
# submitted yet; /scratch (myquota) below 97% (transient ckpts while running: up to 8.8 GB per running 731m job).
# Stages X / C are submitted by hand after select_ptsz.py --write (it prints the generator commands).
set -euo pipefail
P=/scratch/yh6210/llmadd_torch/runs/ptsz
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
GEN=/scratch/yh6210/llmadd_torch/make_jobs_ptsz.py
DISK_MAX=97
GO=0; NOSMOKE=0
for a in "$@"; do case "$a" in --go) GO=1;; --no-smoke) NOSMOKE=1;; *) echo "unknown option $a"; exit 64;; esac; done
[ -e "$P/HOLD" ] && { echo "HOLD present ($P/HOLD): not submitting"; exit 1; }
# the 731m smoke (AUDIT.md "Smoke") must have passed (tests/smoke_report.py JOBID writes smoke/SMOKE_VERDICT.json), unless
# --no-smoke (the coordinator explicitly accepts that 731m's L40S memory fit is unverified)
if [ "$NOSMOKE" = 0 ]; then
  "$PY" -c 'import json,sys; v=json.load(open(sys.argv[1])); sys.exit(0 if v.get("pass") is True else 1)' "$P/smoke/SMOKE_VERDICT.json" 2>/dev/null \
    || { echo "## the 731m smoke has not passed ($P/smoke/SMOKE_VERDICT.json missing or pass != true): nothing submitted (--no-smoke overrides)"; exit 1; }
  echo "# smoke: $P/smoke/SMOKE_VERDICT.json pass"
else
  echo "# --no-smoke: the 731m L40S memory fit is NOT verified (the coordinator's explicit choice)"
fi
"$PY" "$GEN" --check --stage A || { echo "## generator --check FAILED: nothing submitted"; exit 1; }
NAMES=""
for z in 731m 391m 294m 136m; do
  NZ="$("$PY" "$GEN" --names --stage A --size $z)"
  [ "$(echo $NZ | wc -w)" = 32 ] || { echo "## $z: $(echo $NZ | wc -w) stage-A runs, want 32: nothing submitted"; exit 1; }
  NAMES="$NAMES $NZ"
done
N=$(echo $NAMES | wc -w)
[ "$N" = 128 ] || { echo "## $N stage-A runs, want 128: nothing submitted"; exit 1; }
for n in $NAMES; do
  [ -e "$P/SUBMITTED_$n" ] && { echo "## $n already submitted ($P/SUBMITTED_$n): nothing submitted"; exit 1; }
  grep -qx '#SBATCH --nice=100' "$P/job_$n.sbatch" && grep -qx '#SBATCH --partition=l40s_public' "$P/job_$n.sbatch" \
    && grep -qx "#SBATCH --job-name=llmadd_$n" "$P/job_$n.sbatch" || { echo "## $n: header not nice 100 / l40s_public / llmadd_psz-: nothing submitted"; exit 1; }
done
USE=$(myquota 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | sed -n 's/^\/scratch [^(]*(\([0-9.]*\)%).*/\1/p' | head -n 1)
[ -n "$USE" ] || { echo "## myquota could not be parsed: nothing submitted"; exit 1; }
awk -v u="$USE" -v m="$DISK_MAX" 'BEGIN { exit !(u < m) }' || { echo "## DISK: /scratch at $USE% >= $DISK_MAX%: nothing submitted"; exit 1; }
echo "# checks passed: generator --check OK, 128 stage-A runs (32 per size), none submitted, nice 100 / l40s_public, /scratch $USE% < $DISK_MAX%"
for n in $NAMES; do
  if [ "$GO" = 1 ]; then
    out=$(sbatch "$P/job_$n.sbatch") || { echo "## sbatch FAILED for $n: stopping"; exit 1; }
    echo "$out" > "$P/SUBMITTED_$n"
    echo "$n $out"
  else
    echo "sbatch $P/job_$n.sbatch"
  fi
done
[ "$GO" = 1 ] && echo "# submitted 128 stage-A jobs $(date -Is)" || echo "# dry: nothing submitted (--go submits)"
