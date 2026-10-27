#!/bin/bash
# T5: the fourth audit's F1 reproduction (scratchpad t4_audit/lock/), run against the deployed T4 maybe_select.sh and the T5
# one: a stub batch dir (FROZEN '# none', a pick_tpp.py stub that records a decision), a lock left by this job id while
# the job is running (= a job killed while it held the lock and requeued with the same id), SLURM_JOB_NAME llmadd_tpp-test.
#   bash lock_repro.sh <T4 maybe_select.sh> <T5 maybe_select.sh>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LIVE=${SLURM_JOB_ID:?run inside a Slurm job (its id is the live owner)}
for side in T4 T5; do
  ms=$1; [ $side = T5 ] && ms=$2
  R=$HERE/lock_repro/$side
  rm -rf "$R"; mkdir -p "$R/locks/G"
  echo "# none" > "$R/FROZEN.txt"
  cat > "$R/pick_tpp.py" <<'EOF'
import sys, os, datetime
g = sys.argv[sys.argv.index("--group") + 1]
if "--check" in sys.argv:
    sys.exit(0)
open(os.environ["PTTPP_DIR"] + f"/decided_{g}", "a").write(datetime.datetime.now().isoformat() + "\n")
EOF
  cp "$ms" "$R/maybe_select.sh"
  echo "$LIVE" > "$R/locks/G/job"
  out=$(PTTPP_DIR=$R SLURM_JOB_NAME=llmadd_tpp-test SLURM_JOB_ID=$LIVE bash "$R/maybe_select.sh" G 2>&1); rc=$?
  echo "$side ($(sha256sum "$ms" | cut -c1-12)): rc $rc; stdout: ${out:-<none>}"
  echo "   decided: $([ -e "$R/decided_G" ] && echo yes || echo NO); lock left: $([ -d "$R/locks/G" ] && echo YES || echo no);" \
       "pending mark: $([ -e "$R/locks/G.pending" ] && echo YES || echo no); log: $(cat "$R/select_G.log" 2>/dev/null | sed "s#$R#R#g" || echo -)"
done
