#!/bin/bash
# T8 deployment, WITHOUT HOLD (a job starting while HOLD exists exits without training: two runs were lost that way):
#  1. wait until no selection holds a group lock; take the locks of tpp7-1b-t0.745 and tpp7-190m-t0.318 (their hooks then
#     leave a pending mark and exit; the hooks are run by hand at the end);
#  2. the 8-slot generator (git show <commit>) by an atomic rename, its FROZEN.txt line rewritten at once, every sha checked;
#  3. B200: scancel --state=PENDING (only still-pending jobs) of the 8 jobs whose slot changes; a job that started meanwhile
#     is kept; the cancelled ones' job files and SUBMITTED flags are archived (scale_work/tpp/t8/cancelled);
#  4. the two runs lost to HOLD: their job files and flags archived (scale_work/tpp/t8/lost);
#  5. make_jobs_pttpp.py --group <g> --test-only, and only if all OK --submit, for tpp7-1b-t0.745, the bridge and
#     tpp7-190m-t0.318 (environment as pick_tpp.py passes it: no SLURM_* / SBATCH_* / TMPDIR / XDG_RUNTIME_DIR);
#  6. the new job files vs the archived ones: only the slot line (B200) / nothing (the lost runs); the locks released.
#   bash deploy_t8.sh <commit>
set -eu
C=${1:?commit}
R=/scratch/yh6210/llmadd_torch/runs/pttpp
B=/scratch/yh6210/llmadd_torch
WT=/scratch/yh6210/llmadd-burst-tpp
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
s12() { sha256sum "$1" | cut -c1-12; }
[ -z "$(git -C $WT diff 8f798cd "$C" -- tools/ notes/pttpp/maybe_select.sh notes/pttpp/pick_tpp.py)" ] || { echo "more than the generator changed since T7"; exit 1; }
cmp <(git -C $WT show "$C:notes/pttpp/make_jobs_pttpp.py") $B/make_jobs_pttpp_t8test.py
[ ! -e $R/HOLD ] || { echo "HOLD exists"; exit 1; }
mkdir -p $HERE/cancelled $HERE/lost
CLEAN() { env $(compgen -e | grep -E '^(SLURM_|SBATCH_)' | sed 's/^/-u /') -u TMPDIR -u XDG_RUNTIME_DIR "$@"; }
$PY $HERE/wait_quiet.py 60
LOCKS=""
for g in tpp7-1b-t0.745 tpp7-190m-t0.318; do
  mkdir $R/locks/$g
  echo "pid:$(hostname):$$" > $R/locks/$g/job
  LOCKS="$LOCKS $R/locks/$g"
done
trap 'for l in $LOCKS; do rm -rf $l; done' EXIT
echo "[$(date -Is)] locks taken:$LOCKS"
# 2. the generator
old=$(s12 $B/make_jobs_pttpp.py)
git -C $WT show "$C:notes/pttpp/make_jobs_pttpp.py" > $B/.make_jobs_pttpp.py.t8new
chmod a-w $B/.make_jobs_pttpp.py.t8new
chmod u+w $R/FROZEN.txt
mv -f $B/.make_jobs_pttpp.py.t8new $B/make_jobs_pttpp.py
new=$(s12 $B/make_jobs_pttpp.py)
sed -e "s|^$old $B/make_jobs_pttpp.py\$|$new $B/make_jobs_pttpp.py|" -e "s|Amendment T7)\$|Amendment T8)|" $R/FROZEN.txt > $R/.FROZEN.t8new
[ "$(grep -c "^$new $B/make_jobs_pttpp.py\$" $R/.FROZEN.t8new)" = 1 ]
mv -f $R/.FROZEN.t8new $R/FROZEN.txt
chmod a-w $R/FROZEN.txt
bad=0
while read -r sha f; do case "$sha" in \#*|"") continue;; esac; [ "$(s12 $f)" = "$sha" ] || { echo "MISMATCH $f"; bad=1; }; done < $R/FROZEN.txt
[ $bad = 0 ]
echo "[$(date -Is)] generator $old -> $new; FROZEN.txt rewritten and checked"
# 3. B200: cancel the 8 still-pending jobs whose slot changes
$PY $HERE/t8_lists.py $B/make_jobs_pttpp.py b200 > $HERE/b200_resubmit.txt
cat $HERE/b200_resubmit.txt
[ "$(wc -l < $HERE/b200_resubmit.txt)" = 8 ]
ids=$(awk '{print $2}' $HERE/b200_resubmit.txt | tr '\n' ' ')
scancel --state=PENDING -u yh6210 $ids || true
for t in $(seq 1 36); do
  left=$(squeue -h -j "$(echo $ids | tr ' ' ',')" -o "%i %T" 2>/dev/null || true)
  [ -z "$left" ] && break
  echo "$left" | grep -q RUNNING && break
  sleep 5
done
while read -r name jid; do
  st=$(sacct -X -n -j $jid -o state%20 2>/dev/null | head -1 | tr -d ' ')
  if [ "$st" = CANCELLED ] || [[ "$st" == CANCELLED* ]]; then
    mv $R/job_$name.sbatch $HERE/cancelled/ && mv $R/SUBMITTED_$name $HERE/cancelled/
    echo "cancelled $jid $name"
  else
    echo "KEPT $jid $name (state $st): not resubmitted"
  fi
done < $HERE/b200_resubmit.txt
# 4. the two runs lost to HOLD
$PY $HERE/t8_lists.py $B/make_jobs_pttpp.py lost > $HERE/lost.txt
cat $HERE/lost.txt
[ "$(wc -l < $HERE/lost.txt)" = 2 ]
while read -r name jid; do
  [ -z "$(squeue -h -j $jid 2>/dev/null)" ]
  mv $R/job_$name.sbatch $HERE/lost/ && mv $R/SUBMITTED_$name $HERE/lost/
done < $HERE/lost.txt
# 5. test-only, then submit
for g in tpp7-1b-t0.745 tppb-505m-3e8 tpp7-190m-t0.318; do
  CLEAN $PY $B/make_jobs_pttpp.py --group $g --test-only > $HERE/testonly_$g.txt 2>&1 || { cat $HERE/testonly_$g.txt; echo "test-only failed: $g"; exit 1; }
  grep -E "^(OK|BAD) " $HERE/testonly_$g.txt
done
for g in tpp7-1b-t0.745 tppb-505m-3e8 tpp7-190m-t0.318; do
  CLEAN $PY $B/make_jobs_pttpp.py --group $g --submit > $HERE/submit_$g.txt 2>&1 || { cat $HERE/submit_$g.txt; echo "submit failed: $g"; exit 1; }
  grep -E "Submitted batch job|FAILED|WARNING" $HERE/submit_$g.txt || true
done
# 6. the new files vs the archived ones
for f in $HERE/cancelled/job_*.sbatch; do
  n=$(basename $f); d=$(diff $f $R/$n | grep '^[<>]' | sed 's/^[<>] //' | tr '\n' ' ')
  echo "diff $n: $d"
done
for f in $HERE/lost/job_*.sbatch; do cmp $f $R/$(basename $f) && echo "identical $(basename $f)"; done
echo "[$(date -Is)] T8 deployed ($C); locks released at exit"
