#!/bin/bash
# pttpp (runs/pttpp/AUDIT.md "Selection and stages"): run at the end of every job of a group: bash maybe_select.sh <group>.
# 1. HOLD -> nothing. 2. Every frozen file (FROZEN.txt: the read-out snapshot, pick_tpp.py, this file, the generator) has
# its recorded sha256[:12], else "FROZEN FILE CHANGED" in the log and nothing happens. 3. pick_tpp.py --check (no lock):
# nothing to do -> exit. 4. The group's lock (a dir; a lock whose job is no longer running is taken over); a job that finds
# the lock held leaves a 'pending' mark and exits, and the holder runs the step again before it releases the lock, so a run
# that finishes during a selection is never missed. 5. pick_tpp.py --decide (decisions, then --test-only / --submit).
set -u
R=${PTTPP_DIR:-/scratch/yh6210/llmadd_torch/runs/pttpp}   # overrides: plumbing tests only
PY=${PTTPP_PY:-/scratch/yh6210/miniconda3/envs/open_r1/bin/python}
G=${1:?group}
LOG=$R/select_$G.log
[ -e "$R/HOLD" ] && { echo "maybe_select: HOLD present"; exit 0; }
while read -r sha f; do
  [ -z "$sha" ] && continue
  case "$sha" in \#*) continue;; esac
  got=$(sha256sum "$f" 2>/dev/null | cut -c1-12)
  if [ "$got" != "$sha" ]; then
    echo "[$(date -Is)] job ${SLURM_JOB_ID:-none}: FROZEN FILE CHANGED: $f ($got, want $sha): nothing selected or submitted" >> "$LOG"
    exit 0
  fi
done < "$R/FROZEN.txt"
PTTPP_DIR=$R $PY "$R/pick_tpp.py" --group "$G" --check || exit 0
me=${SLURM_JOB_ID:-none}
mkdir -p "$R/locks"
if ! mkdir "$R/locks/$G" 2>/dev/null; then
  other=$(cat "$R/locks/$G/job" 2>/dev/null || true)
  if [ -n "$other" ] && [ "$other" != "$me" ] && squeue -h -j "$other" -t R,CG 2>/dev/null | grep -q .; then
    touch "$R/locks/$G.pending"
    echo "maybe_select: the selection of $G is running in job $other (pending mark left)"; exit 0
  fi
  echo "[$(date -Is)] job $me: taking over the lock of job ${other:-?}" >> "$LOG"
fi
echo "$me" > "$R/locks/$G/job"
trap 'rm -rf "$R/locks/$G"' EXIT
for v in $(compgen -e | grep -E "^(SLURM|SBATCH)_"); do [ "$v" = SLURM_JOB_ID ] || unset "$v"; done
for i in 1 2 3; do
  rm -f "$R/locks/$G.pending"
  PTTPP_DIR=$R $PY "$R/pick_tpp.py" --group "$G" --decide >> "$LOG.stdout" 2>&1 || echo "[$(date -Is)] job $me: pick_tpp.py --decide rc $? (see $LOG.stdout)" >> "$LOG"
  [ -e "$R/locks/$G.pending" ] || break
done
