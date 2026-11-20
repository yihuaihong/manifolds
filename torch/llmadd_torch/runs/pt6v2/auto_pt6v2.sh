#!/bin/bash
# pt6v2 pick -> submit chain (V2 build, AUDIT.md "V2 build"). Dry by default: runs the pick read-out and the generator's --check, prints
# every state-changing command; --go executes them. Stops with "## STOPPED at step N" at the first failure. Log: runs/pt6v2/auto_<UTC ts>.log
#   bash runs/pt6v2/auto_pt6v2.sh [--go] [--pick_json <file>]      (--pick_json: skip step 1 and read <file>; tests use a stub)
#   bash runs/pt6v2/auto_pt6v2.sh --test_refs                      (only step 3, for real, MODE=placeholder: the poll-and-grep test)
# Steps:
#   1 $PY runs/pt6v2/pick_pt6v2.py --json runs/pt6v2/PICK.json   exit 0 complete; exit 2 = inputs incomplete -> stop (exit 2)
#   2 make_jobs_pt6v2.py --set_comp <5> --by "auto-pick <date>"   skipped when COMPOSITION.json = the pick; stops when it differs
#   3 CPU refs job (MODE=comp; tests/refs_pt6v2.sbatch): job id kept in tests/auto_refs_comp.jobid (a re-run polls it, never resubmits);
#     poll sacct until it ends; require COMPLETED + "TABLE <mk> PASS" + "SPLIT-INDEPENDENCE <mk> PASS" in its log
#   4 make_jobs_pt6v2.py --alloc <ALLOC> --canary_dev a100 [--extra_none 6.4e-2] --check --write
#   5 runs/pt6v2/submit_pt6v2.sh --alloc <ALLOC> --canary_dev a100 [--extra_none 6.4e-2] --go   (keeps every refusal of submit_pt6v2.sh)
set -uo pipefail
B=/scratch/yh6210/llmadd_torch
R=$B/runs/pt6v2
T=$R/tests
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
export PYTHONDONTWRITEBYTECODE=1
ALLOC_S="b200:16,17,18,19,20,21,22 a100:23"   # AUDIT.md "Composition pick rule": 7 x B200 + 1 x A100, canary on A100
read -r -a ALLOC <<< "$ALLOC_S"
CDEV=a100
GO=0; PJ=""; TR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift;;
    --pick_json) PJ="$2"; shift 2;;
    --test_refs) TR=1; shift;;
    *) echo "unknown argument $1"; exit 64;;
  esac
done
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$R/auto_$TS.log
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
stop() { say "## STOPPED at step $1: $2"; say "log: $LOG"; exit "${3:-1}"; }
cd "$B" || exit 1
say "auto_pt6v2 $([ "$GO" = 1 ] && echo GO || echo DRY)$([ "$TR" = 1 ] && echo ' (--test_refs)'); alloc $ALLOC_S; canary on $CDEV; log $LOG"

refs_step() {   # $1 MODE (comp | placeholder), $2 refs key mk
  local mode=$1 mk=$2 jf=$T/auto_refs_$1.jobid jid st out n=0
  if [ -s "$jf" ]; then
    jid=$(cat "$jf"); say "step 3: refs job $jid recorded in $jf: polling it (not resubmitted)"
  else
    jid=$(env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_CPUS_PER_TASK -u SLURM_MEM_PER_NODE -u SLURM_NTASKS \
          sbatch --parsable --export=ALL,MODE="$mode" "$T/refs_pt6v2.sbatch") || stop 3 "sbatch failed: $jid"
    jid=${jid%%;*}
    [[ "$jid" =~ ^[0-9]+$ ]] || stop 3 "sbatch returned '$jid'"
    echo "$jid" > "$jf"
    say "step 3: submitted the CPU refs job $jid (MODE=$mode; id kept in $jf)"
  fi
  while :; do
    st=$(sacct -j "$jid" -X -n -P -o State 2>/dev/null | head -n 1)
    case "$st" in
      ""|PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|RESIZING|SUSPENDED)
        n=$((n + 1)); [ "$n" -le 270 ] || stop 3 "refs job $jid still '${st:-not in sacct}' after 90 min"
        sleep 20;;
      *) break;;
    esac
  done
  out=$T/refs_llmadd_pt6v2-refs_$jid.out
  say "step 3: refs job $jid ended: $st; log $out"
  [ "$st" = COMPLETED ] || stop 3 "refs job $jid ended $st (log $out)"
  [ -e "$out" ] || stop 3 "no log $out"
  grep -E "^(TABLE|SPLIT-INDEPENDENCE|DOMREFS) |already done|Traceback|rc=" "$out" | sed 's/^/    /'
  grep -q "already done" "$out" && stop 3 "the refs job was a no-op (tests/refs_$mode.done existed before it; check the earlier refs log)"
  grep -q "^TABLE $mk PASS" "$out" || stop 3 "no 'TABLE $mk PASS' in $out"
  grep -q "^SPLIT-INDEPENDENCE $mk PASS" "$out" || stop 3 "no 'SPLIT-INDEPENDENCE $mk PASS' in $out"
  say "step 3: PASS (TABLE + SPLIT-INDEPENDENCE for $mk)"
}

if [ "$TR" = 1 ]; then
  [ "$GO" = 0 ] && [ -z "$PJ" ] || stop 3 "--test_refs takes no --go / --pick_json"
  MK=$("$PY" "$T/auto_read_pick.py" --placeholder | sed -n 's/^MK=//p')
  refs_step placeholder "$MK"
  say "--test_refs done"; exit 0
fi

# ---- step 1: the pick
PICK=$R/PICK.json
# coordinator 2026-11-19 (after an accidental --go with a partial pick JSON; tests/incident_20261119_accidental_go/):
# --go refuses while runs/pt6v2/HOLD exists (remove it by hand at pick time) and never takes a hand-given pick JSON
if [ "$GO" = 1 ]; then
  [ -e "$R/HOLD" ] && stop 1 "runs/pt6v2/HOLD exists: --go refused (remove HOLD by hand at pick time)"
  [ -n "$PJ" ] && stop 1 "--go with --pick_json refused: --go takes the pick from pick_pt6v2.py (strict) only"
fi
if [ -n "$PJ" ]; then
  PICK=$PJ; say "step 1: --pick_json $PJ (pick_pt6v2.py not run)"
else
  [ -e "$R/pick_pt6v2.py" ] || stop 1 "runs/pt6v2/pick_pt6v2.py does not exist yet"
  say "step 1: + $PY $R/pick_pt6v2.py --json $PICK"
  "$PY" "$R/pick_pt6v2.py" --json "$PICK"; rc=$?
  [ "$rc" = 2 ] && stop 1 "pick_pt6v2.py: inputs incomplete (exit 2); nothing else done" 2
  [ "$rc" = 0 ] || stop 1 "pick_pt6v2.py exited $rc"
fi
[ -e "$PICK" ] || stop 1 "no $PICK"
PA=$("$PY" "$T/auto_read_pick.py" "$PICK" 2>&1) || stop 1 "$PICK invalid: $(echo "$PA" | tail -n 2)"
COMP=$(sed -n 's/^COMP=//p' <<< "$PA"); EXTRA=$(sed -n 's/^EXTRA=//p' <<< "$PA"); STATE=$(sed -n 's/^STATE=//p' <<< "$PA")
REFS=$(sed -n 's/^REFS=//p' <<< "$PA"); MK=$(sed -n 's/^MK=//p' <<< "$PA")
EXN=(); [ "$EXTRA" = 1 ] && EXN=(--extra_none 6.4e-2)
say "step 1: composition $COMP (refs key $MK); Amendment V2 none@6.4e-2: $([ "$EXTRA" = 1 ] && echo yes || echo no); COMPOSITION.json: $STATE; refs file: $REFS"

# ---- step 2: COMPOSITION.json
case "$STATE" in
  same) say "step 2: COMPOSITION.json already = the pick: skipped";;
  differ) stop 2 "COMPOSITION.json exists and differs from the pick ($COMP): the user decides";;
  new) if [ "$GO" = 1 ]; then
         say "step 2: + make_jobs_pt6v2.py --set_comp $COMP --by auto-pick $(date -u +%F)"
         "$PY" make_jobs_pt6v2.py --set_comp "$COMP" --by "auto-pick $(date -u +%F)" || stop 2 "--set_comp failed"
       else say "step 2: (dry) would run: $PY make_jobs_pt6v2.py --set_comp $COMP --by 'auto-pick $(date -u +%F)'"; fi;;
  *) stop 2 "unknown COMPOSITION.json state '$STATE'";;
esac

# ---- step 3: CPU refs
if [ "$GO" = 1 ]; then
  refs_step comp "$MK"
else
  say "step 3: (dry) would run: env -u SLURM_JOB_ID ... sbatch --parsable --export=ALL,MODE=comp $T/refs_pt6v2.sbatch; poll sacct; require 'TABLE $MK PASS' + 'SPLIT-INDEPENDENCE $MK PASS'"
fi

# ---- step 4: render + write
if [ "$GO" = 1 ]; then
  say "step 4: + make_jobs_pt6v2.py --alloc $ALLOC_S --canary_dev $CDEV ${EXN[*]} --check --write"
  "$PY" make_jobs_pt6v2.py --alloc "${ALLOC[@]}" --canary_dev "$CDEV" "${EXN[@]}" --check --write || stop 4 "--check --write failed"
else
  FR=(); [ "$REFS" = 1 ] || FR=(--fake_refs)
  say "step 4: (dry) render check: make_jobs_pt6v2.py --comp $COMP --alloc $ALLOC_S --canary_dev $CDEV ${EXN[*]} --check ${FR[*]} (nothing written)"
  "$PY" make_jobs_pt6v2.py --comp "$COMP" --alloc "${ALLOC[@]}" --canary_dev "$CDEV" "${EXN[@]}" "${FR[@]}" --check | tail -n 2 \
    || stop 4 "--check failed"
  say "step 4: (dry) would run: make_jobs_pt6v2.py --alloc $ALLOC_S --canary_dev $CDEV ${EXN[*]} --check --write"
fi

# ---- step 5: submit
if [ "$GO" = 1 ]; then
  say "step 5: + submit_pt6v2.sh --alloc '$ALLOC_S' --canary_dev $CDEV ${EXN[*]} --go"
  bash "$R/submit_pt6v2.sh" --alloc "$ALLOC_S" --canary_dev "$CDEV" "${EXN[@]}" --go || stop 5 "submit_pt6v2.sh refused or failed"
  # coordinator 2026-11-19 (audit_diff_V2.md I3): make_jobs --submit goes on past a failed sbatch; count the flags
  "$PY" "$T/auto_count_submitted.py" || stop 5 "some jobs were not submitted (a failed sbatch): re-run with --go (a name is submitted once)"
  say "done: submitted (see the SUBMITTED_* flags in runs/pt6v2)"
else
  say "step 5: (dry) submit_pt6v2.sh checks (no --go):"
  bash "$R/submit_pt6v2.sh" --alloc "$ALLOC_S" --canary_dev "$CDEV" "${EXN[@]}"; rc=$?
  say "step 5: (dry) submit_pt6v2.sh exit $rc$([ "$rc" = 3 ] && echo ' (a refusal; expected before step 2-4 ran for real)')"
  say "DRY done: nothing submitted, nothing written"
fi
