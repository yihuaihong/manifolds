#!/bin/bash
# pt2same submission (intent spec and pre-registered read-out: runs/pt2same/AUDIT.md). Run by the coordinator.
#   bash submit_pt2same.sh          print the sbatch command of every job not yet submitted (nothing is submitted)
#   bash submit_pt2same.sh --go     submit them (sbatch --parsable) and write SUBMITTED_<run> with the job id
# Order = JOBS.sha256 = the generator's order (seed-major, 190M before 53M, none before win-ctx), so the first wave shows
# every cell. Before anything: every job file must match JOBS.sha256 (the generator's rendering) and the generator / read-out
# must match FROZEN.sha256 (the tested versions). Jobs: account torch_pr_520_tandon_priority, a100_tandon, gpu48, nice 20.
# A submitted job is never resubmitted (its SUBMITTED_ flag); a re-run needs a new run name (change the generator).
set -euo pipefail
R=/scratch/yh6210/llmadd_torch/runs/pt2same
GO=0
case "${1:-}" in
  "") ;;
  --go) GO=1 ;;
  *) echo "usage: $0 [--go]"; exit 2 ;;
esac
[ $# -le 1 ] || { echo "usage: $0 [--go]"; exit 2; }
cd "$R"
sha256sum --quiet -c JOBS.sha256 || { echo "## a job file differs from JOBS.sha256: re-render (python make_jobs_pt2same.py) first"; exit 3; }
[ "$(wc -l < JOBS.sha256)" = 32 ] || { echo "## JOBS.sha256 does not list 32 jobs"; exit 3; }
(cd /scratch/yh6210/llmadd_torch && sha256sum --quiet -c "$R/FROZEN.sha256") || { echo "## a frozen file changed (FROZEN.sha256): not submitting"; exit 3; }
if [ -e "$R/HOLD" ]; then echo "WARNING: runs/pt2same/HOLD exists: every job will exit at start"; fi
n=0; skip=0
while read -r _ job; do
  name=${job#job_}; name=${name%.sbatch}
  if [ -e "SUBMITTED_$name" ]; then
    echo "already submitted: $name ($(tr -d '\n' < "SUBMITTED_$name"))"; skip=$((skip + 1)); continue
  fi
  if [ "$GO" = 1 ]; then
    id=$(sbatch --parsable "$R/$job") || { echo "## sbatch FAILED for $name: stopping (submitted so far: $n)"; exit 1; }
    echo "$id" > "SUBMITTED_$name"
    echo "submitted $name: $id"
  else
    echo "sbatch $R/$job"
  fi
  n=$((n + 1))
done < JOBS.sha256
if [ "$GO" = 1 ]; then
  echo "pt2same: $n jobs submitted, $skip already submitted"
else
  echo "pt2same: $n sbatch commands printed, $skip already submitted; nothing submitted (add --go)"
fi
