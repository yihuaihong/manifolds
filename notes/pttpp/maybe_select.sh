#!/bin/bash
# pttpp (runs/pttpp/AUDIT.md "Selection and stages", Amendments T1 / T3): run at the end of every job of a group:
# bash maybe_select.sh <group>.
# 1. HOLD -> nothing. 2. Every frozen file (FROZEN.txt) has its recorded sha256[:12], else "FROZEN FILE CHANGED" in the log
# and nothing happens. 3. pick_tpp.py --check (no lock): nothing to do -> exit. 4. The group's lock (a dir; owner = the
# Slurm job id, or pid:<host>:<pid> outside Slurm). On a held lock the caller FIRST leaves a 'pending' mark, then tries the
# lock once more (T3: the holder releases the lock before it checks the mark, so either the holder sees the mark or this
# second try succeeds); then the owner is alive -> exit (the holder repeats the step); gone -> take over. Alive: a job id
# that squeue shows running or cannot tell about (any squeue error but "Invalid job id"); a pid on this host that exists.
# A lock older than 2 h is stale whatever its owner (a selection takes minutes; T3: a SIGKILL leaves the lock behind).
# 5. pick_tpp.py --decide, the lock released, and if a pending mark appeared meanwhile the step repeats (<= 5 rounds).
set -u
R=${PTTPP_DIR:-/scratch/yh6210/llmadd_torch/runs/pttpp}   # overrides: plumbing tests only
PY=${PTTPP_PY:-/scratch/yh6210/miniconda3/envs/open_r1/bin/python}
G=${1:?group}
LOG=$R/select_$G.log
L=$R/locks/$G
P=$R/locks/$G.pending
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
me=${SLURM_JOB_ID:-pid:$(hostname):$$}
mkdir -p "$R/locks"
for v in $(compgen -e | grep -E "^(SLURM|SBATCH)_"); do [ "$v" = SLURM_JOB_ID ] || unset "$v"; done
alive() {   # owner -> 0 if it may still be selecting
  local o=$1
  case "$o" in
    pid:*) local h=${o#pid:}; h=${h%%:*}; local pid=${o##*:}
           [ "$h" = "$(hostname)" ] && kill -0 "$pid" 2>/dev/null && return 0
           [ "$h" != "$(hostname)" ] && return 0   # another host's pid: cannot tell -> alive (the 2 h age rule frees it)
           return 1;;
    "") return 1;;
    *) local q qrc; q=$(squeue -h -j "$o" -t R,CG,CF -o %i 2>&1); qrc=$?
       { [ $qrc = 0 ] && [ -n "$q" ]; } && return 0
       { [ $qrc != 0 ] && ! echo "$q" | grep -q "Invalid job id"; } && return 0
       return 1;;
  esac
}
for round in 1 2 3 4 5; do
  if ! mkdir "$L" 2>/dev/null; then
    touch "$P"
    if ! mkdir "$L" 2>/dev/null; then
      other=$(cat "$L/job" 2>/dev/null || true)
      age=$(( $(date +%s) - $(stat -c %Y "$L" 2>/dev/null || date +%s) ))
      if [ "$other" != "$me" ] && [ $age -lt 7200 ] && alive "$other"; then
        echo "maybe_select: the selection of $G is held by $other (age ${age}s): pending mark left"; exit 0
      fi
      echo "[$(date -Is)] job $me: taking over the lock of ${other:-?} (age ${age}s)" >> "$LOG"
    fi
  fi
  trap 'rm -rf "$L"' EXIT
  echo "$me" > "$L/job"
  rm -f "$P"
  PTTPP_DIR=$R $PY "$R/pick_tpp.py" --group "$G" --decide >> "$LOG.stdout" 2>&1 || echo "[$(date -Is)] job $me: pick_tpp.py --decide rc $? (see $LOG.stdout)" >> "$LOG"
  rm -rf "$L"; trap - EXIT
  [ -e "$P" ] || exit 0
done
echo "[$(date -Is)] job $me: a pending mark is still set after 5 rounds: run 'bash $R/maybe_select.sh $G' by hand" >> "$LOG"
