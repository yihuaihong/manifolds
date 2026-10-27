#!/bin/bash
# pttpp (runs/pttpp/AUDIT.md "Selection and stages", Amendments T1 / T3 / T4): run at the end of every job of a group:
# bash maybe_select.sh <group>.
# 1. HOLD -> nothing. 2. Every frozen file (FROZEN.txt) has its recorded sha256[:12], else "FROZEN FILE CHANGED" in the log
# and nothing happens. 3. pick_tpp.py --check (no lock): nothing to do -> exit. 4. The group's lock (a dir). Owner: the
# Slurm job id inside a pttpp training job (SLURM_JOB_NAME llmadd_tpp-*), else pid:<host>:<pid> (a selection run by hand,
# also from inside the claude_host job; T4). On a held lock the caller first leaves a 'pending' mark and tries the lock once
# more (the holder releases before it checks the mark). Then: owner alive (a job id squeue shows running or cannot tell
# about; a pid that exists on this host; a pid of another host) -> exit; owner empty (the lock was just made) -> exit, unless
# the lock is older than 10 min; owner gone -> take over by renaming the lock away (only if it still names that owner) and
# making a fresh one (T4: the old 2 h age rule and the in-place takeover let several callers run at once).
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
case "${SLURM_JOB_NAME:-}" in llmadd_tpp-*) me=${SLURM_JOB_ID:?};; *) me=pid:$(hostname):$$;; esac
mkdir -p "$R/locks"
for v in $(compgen -e | grep -E "^(SLURM|SBATCH)_"); do [ "$v" = SLURM_JOB_ID ] || unset "$v"; done
alive() {   # owner -> 0 if it may still be selecting
  local o=$1
  case "$o" in
    pid:*) local h=${o#pid:}; h=${h%%:*}; local pid=${o##*:}
           [ "$h" != "$(hostname)" ] && return 0   # another host's process: cannot tell -> alive
           kill -0 "$pid" 2>/dev/null && return 0
           return 1;;
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
      if [ -z "$other" ] && [ $age -lt 600 ]; then
        echo "maybe_select: the lock of $G was just made (no owner yet): pending mark left"; exit 0
      fi
      if [ -n "$other" ] && alive "$other"; then
        echo "maybe_select: the selection of $G is held by $other: pending mark left"; exit 0
      fi
      # take over: rename the stale lock away only if it still names that owner, then make a fresh one
      [ "$(cat "$L/job" 2>/dev/null || true)" = "$other" ] && mv "$L" "$L.stale.$$" 2>/dev/null
      if ! mkdir "$L" 2>/dev/null; then
        echo "maybe_select: another caller took the lock of $G: pending mark left"; exit 0
      fi
      echo "[$(date -Is)] job $me: took over the lock of ${other:-?} (age ${age}s; old lock kept as $L.stale.$$)" >> "$LOG"
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
