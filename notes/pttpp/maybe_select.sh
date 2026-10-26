#!/bin/bash
# pttpp (runs/pttpp/AUDIT.md "Selection and stages", Amendment T1): run at the end of every job of a group:
# bash maybe_select.sh <group>.
# 1. HOLD -> nothing. 2. Every frozen file (FROZEN.txt: the read-out snapshot, pick_tpp.py, this file, the generator) has
# its recorded sha256[:12], else "FROZEN FILE CHANGED" in the log and nothing happens. 3. pick_tpp.py --check (no lock):
# nothing to do -> exit. 4. The group's lock (a dir). Held by a job that squeue shows running, or whose state squeue cannot
# tell (any squeue error other than "Invalid job id") -> a 'pending' mark and exit; held by a job that is gone -> taken
# over. 5. pick_tpp.py --decide, the lock released, and if a pending mark appeared meanwhile the lock is taken again and
# the step repeated (at most 5 rounds; a mark left after that is logged), so a run finishing during a selection is not
# missed (T1: the earlier version could lose a mark set after its last check).
set -u
R=${PTTPP_DIR:-/scratch/yh6210/llmadd_torch/runs/pttpp}   # overrides: plumbing tests only
PY=${PTTPP_PY:-/scratch/yh6210/miniconda3/envs/open_r1/bin/python}
G=${1:?group}
LOG=$R/select_$G.log
L=$R/locks/$G
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
for v in $(compgen -e | grep -E "^(SLURM|SBATCH)_"); do [ "$v" = SLURM_JOB_ID ] || unset "$v"; done
for round in 1 2 3 4 5; do
  if ! mkdir "$L" 2>/dev/null; then
    other=$(cat "$L/job" 2>/dev/null || true)
    if [ -n "$other" ] && [ "$other" != "$me" ]; then
      q=$(squeue -h -j "$other" -t R,CG,CF -o %i 2>&1); qrc=$?
      if { [ $qrc = 0 ] && [ -n "$q" ]; } || { [ $qrc != 0 ] && ! echo "$q" | grep -q "Invalid job id"; }; then
        touch "$R/locks/$G.pending"
        echo "maybe_select: the selection of $G is held by job $other (squeue rc $qrc): pending mark left"; exit 0
      fi
    fi
    echo "[$(date -Is)] job $me: taking over the lock of job ${other:-?}" >> "$LOG"
  fi
  trap 'rm -rf "$L"' EXIT
  echo "$me" > "$L/job"
  rm -f "$R/locks/$G.pending"
  PTTPP_DIR=$R $PY "$R/pick_tpp.py" --group "$G" --decide >> "$LOG.stdout" 2>&1 || echo "[$(date -Is)] job $me: pick_tpp.py --decide rc $? (see $LOG.stdout)" >> "$LOG"
  rm -rf "$L"; trap - EXIT
  [ -e "$R/locks/$G.pending" ] || exit 0
done
echo "[$(date -Is)] job $me: a pending mark is still set after 5 rounds: run 'bash $R/maybe_select.sh $G' by hand" >> "$LOG"
