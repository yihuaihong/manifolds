#!/bin/bash
# T8 deployment, part 2 (deploy_t8.sh stopped after step 4: sacct still read PENDING right after the scancel, so the 8
# cancelled B200 jobs were marked KEPT and their flags not archived; then a grep with no match ended the script under set -e).
# State at the start: the 8-slot generator deployed and frozen; the 8 B200 jobs CANCELLED (sacct, restarts 0); the two lost
# runs' files archived. This part: the group locks again, the 8 cancelled jobs' files and flags archived (each must read
# CANCELLED in sacct), then test-only and submit for tpp7-1b-t0.745, the bridge and tpp7-190m-t0.318, the diffs.
set -u
R=/scratch/yh6210/llmadd_torch/runs/pttpp
B=/scratch/yh6210/llmadd_torch
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
CLEAN() { env $(compgen -e | grep -E '^(SLURM_|SBATCH_)' | sed 's/^/-u /') -u TMPDIR -u XDG_RUNTIME_DIR "$@"; }
[ ! -e $R/HOLD ] || { echo "HOLD exists"; exit 1; }
$PY $HERE/wait_quiet.py 30 || exit 1
LOCKS=""
for g in tpp7-1b-t0.745 tpp7-190m-t0.318; do
  mkdir $R/locks/$g || { echo "lock $g held"; exit 1; }
  echo "pid:$(hostname):$$" > $R/locks/$g/job
  LOCKS="$LOCKS $R/locks/$g"
done
trap 'for l in $LOCKS; do rm -rf $l; done' EXIT
echo "[$(date -Is)] locks taken:$LOCKS"
n=0
while read -r name jid; do
  st=$(sacct -X -n -j $jid -o state%30 2>/dev/null | head -1 | sed 's/^ *//')
  rs=$(sacct -X -n -j $jid -o restart 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in
    CANCELLED*) [ "$rs" = 0 ] || { echo "$jid restarted ($rs): stop"; exit 1; }
                [ -e $R/SUBMITTED_$name ] && mv $R/job_$name.sbatch $HERE/cancelled/ && mv $R/SUBMITTED_$name $HERE/cancelled/
                n=$((n + 1)); echo "archived $jid $name ($st, restarts $rs)";;
    *) echo "NOT CANCELLED $jid $name ($st): stop"; exit 1;;
  esac
done < $HERE/b200_resubmit.txt
[ $n = 8 ] || exit 1
for name in $(awk '{print $1}' $HERE/lost.txt); do [ ! -e $R/SUBMITTED_$name ] || { echo "$name was resubmitted meanwhile"; exit 1; }; done
for g in tpp7-1b-t0.745 tppb-505m-3e8 tpp7-190m-t0.318; do
  CLEAN $PY $B/make_jobs_pttpp.py --group $g --test-only > $HERE/testonly_$g.txt 2>&1 || { cat $HERE/testonly_$g.txt; echo "test-only failed: $g"; exit 1; }
  grep -E "^(OK|BAD) |WARNING" $HERE/testonly_$g.txt
done
for g in tpp7-1b-t0.745 tppb-505m-3e8 tpp7-190m-t0.318; do
  CLEAN $PY $B/make_jobs_pttpp.py --group $g --submit > $HERE/submit_$g.txt 2>&1 || { cat $HERE/submit_$g.txt; echo "submit failed: $g"; exit 1; }
  grep -E "Submitted batch job|FAILED|WARNING" $HERE/submit_$g.txt
done
for f in $HERE/cancelled/job_*.sbatch; do
  nm=$(basename $f); echo "diff $nm: $(diff $f $R/$nm | grep '^[<>]' | sed 's/^[<>] //' | tr '\n' ' ')"
done
for f in $HERE/lost/job_*.sbatch; do cmp $f $R/$(basename $f) && echo "identical $(basename $f)"; done
echo "[$(date -Is)] T8 part 2 done; locks released at exit"
