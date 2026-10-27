#!/bin/bash
# T7 deployment: the generator (make_jobs_pttpp.py) and the read-out snapshot change; pick_tpp.py and maybe_select.sh stay.
# HOLD, wait until no selection holds a group lock, swap the read-out snapshot (git archive <commit> tools/; the T6 one kept
# as readout.pre_t7) and the generator (git show <commit>:notes/pttpp/make_jobs_pttpp.py, atomic rename), rewrite their lines
# of FROZEN.txt (and the readout COMMIT line), re-check every frozen sha, remove HOLD.
#   bash deploy_t7.sh <commit>
set -eu
C=${1:?commit}
R=/scratch/yh6210/llmadd_torch/runs/pttpp
B=/scratch/yh6210/llmadd_torch
WT=/scratch/yh6210/llmadd-burst-tpp
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
s12() { sha256sum "$1" | cut -c1-12; }
[ -z "$(git -C $WT diff 61175a6 "$C" -- notes/pttpp/maybe_select.sh notes/pttpp/pick_tpp.py)" ] || { echo "pick_tpp.py / maybe_select.sh changed since T6"; exit 1; }
cmp <(git -C $WT show "$C:notes/pttpp/make_jobs_pttpp.py") $B/make_jobs_pttpp_t7test.py   # the tested copy
[ -e $R/readout.pre_t7 ] && { echo "readout.pre_t7 exists"; exit 1; }
touch $R/HOLD
echo "[$(date -Is)] HOLD set"
$PY $HERE/wait_nolock.py 90
tmp=$R/readout.t7tmp
rm -rf $tmp; mkdir -p $tmp
git -C $WT archive "$C" tools/ | tar -x -C $tmp
echo "$C (ladder-tpp, committed; git archive of tools/; Amendment T7); sha256[:12]: pttpp_summary.py $(s12 $tmp/tools/pttpp_summary.py), readout_rules.py $(s12 $tmp/tools/readout_rules.py), pt999_summary.py $(s12 $tmp/tools/pt999_summary.py), pt2main_summary.py $(s12 $tmp/tools/pt2main_summary.py), " > $tmp/COMMIT
chmod -R a-w $tmp
mv $R/readout $R/readout.pre_t7
mv $tmp $R/readout
git -C $WT show "$C:notes/pttpp/make_jobs_pttpp.py" > $B/.make_jobs_pttpp.py.t7new
chmod a-w $B/.make_jobs_pttpp.py.t7new
mv -f $B/.make_jobs_pttpp.py.t7new $B/make_jobs_pttpp.py
chmod u+w $R/FROZEN.txt
{ echo "# pttpp frozen files (sha256[:12] path), checked by maybe_select.sh before any selection (runs/pttpp/AUDIT.md; Amendment T7)"
  grep -v '^#' $R/FROZEN.txt | while read -r sha f; do echo "$(s12 $f) $f"; done; } > $R/.FROZEN.t7new
diff <(grep -v '^#' $R/FROZEN.txt | awk '{print $2}') <(grep -v '^#' $R/.FROZEN.t7new | awk '{print $2}')
echo "FROZEN lines changed:"; diff <(grep -v '^#' $R/FROZEN.txt) <(grep -v '^#' $R/.FROZEN.t7new) | grep '^>' || true
mv -f $R/.FROZEN.t7new $R/FROZEN.txt
chmod a-w $R/FROZEN.txt
bad=0
while read -r sha f; do case "$sha" in \#*|"") continue;; esac; [ "$(s12 $f)" = "$sha" ] || { echo "MISMATCH $f"; bad=1; }; done < $R/FROZEN.txt
[ $bad = 0 ] || exit 1
cmp <(git -C $WT show "$C:notes/pttpp/make_jobs_pttpp.py") $B/make_jobs_pttpp.py
rm -f $R/HOLD
echo "[$(date -Is)] deployed $C (generator $(s12 $B/make_jobs_pttpp.py), pttpp_summary.py $(s12 $R/readout/tools/pttpp_summary.py)); $(grep -vc '^#' $R/FROZEN.txt) frozen files re-checked; HOLD removed"
