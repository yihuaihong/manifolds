#!/bin/bash
# T5 deployment (as T2-T4): HOLD, wait until no selection holds a group lock, swap the read-out snapshot (git archive of
# the T5 commit's tools/), pick_tpp.py and maybe_select.sh (git show of the commit; atomic renames, so a bash that is
# running the old maybe_select.sh keeps its file), rewrite FROZEN.txt, check every frozen sha, remove HOLD.
#   bash deploy_t5.sh <commit>
set -eu
C=${1:?commit}
R=/scratch/yh6210/llmadd_torch/runs/pttpp
B=/scratch/yh6210/llmadd_torch
WT=/scratch/yh6210/llmadd-burst-tpp
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
git -C $WT cat-file -e "$C:tools/pttpp_summary.py"
s12() { sha256sum "$1" | cut -c1-12; }
touch $R/HOLD
echo "[$(date -Is)] HOLD set"
$PY $HERE/wait_nolock.py 90
# read-out snapshot
tmp=$R/readout.t5tmp
rm -rf $tmp; mkdir -p $tmp
git -C $WT archive "$C" tools/ | tar -x -C $tmp
echo "$C (ladder-tpp, committed; git archive of tools/; Amendment T5); sha256[:12]: pttpp_summary.py $(s12 $tmp/tools/pttpp_summary.py), readout_rules.py $(s12 $tmp/tools/readout_rules.py), pt999_summary.py $(s12 $tmp/tools/pt999_summary.py), pt2main_summary.py $(s12 $tmp/tools/pt2main_summary.py), " > $tmp/COMMIT
chmod -R a-w $tmp
[ -e $R/readout.pre_t5 ] && { echo "readout.pre_t5 exists"; exit 1; }
mv $R/readout $R/readout.pre_t5
mv $tmp $R/readout
# selection plumbing (atomic renames)
for f in pick_tpp.py maybe_select.sh; do
  git -C $WT show "$C:notes/pttpp/$f" > $R/.$f.t5new
  chmod a-w $R/.$f.t5new
  mv -f $R/.$f.t5new $R/$f
done
# FROZEN.txt (T5: + the generator's import-time reference files, audit F4; + the 8 Burst dry logs the 2% selection reads, F5)
RO=/scratch/yh6210/llmadd-burst-ro/results
files="$R/readout/tools/pttpp_summary.py $R/readout/tools/readout_rules.py $R/readout/tools/pt999_summary.py
$R/readout/tools/pt2main_summary.py $R/readout/COMMIT $R/pick_tpp.py $R/maybe_select.sh $B/make_jobs_pttpp.py $B/pttpp_refs.json
$B/make_jobs_ptlr1b.py $B/make_jobs_pt2s505.py $B/make_jobs_rb6scale.py $R/code7/tools/pretrain_range.py $R/code2/tools/pretrain_range.py
$R/code2/tools/make_pt2main_tasks.py $R/code2/tools/offline_lm.py $B/pt2s505_refs.json $B/rb6scale_refs.json
$(ls $RO/{358,359,360,361}-pt2s190-lr6e-4-none-s?/dry_inputs.txt $RO/{382,383,384,385}-pt2s190-lr4.8e-3-none-s?/dry_inputs.txt)"
n=$(echo $files | wc -w); [ $n = 26 ] || { echo "want 26 frozen files, got $n"; exit 1; }
chmod u+w $R/FROZEN.txt
{ echo "# pttpp frozen files (sha256[:12] path), checked by maybe_select.sh before any selection (runs/pttpp/AUDIT.md; Amendment T5)"
  for f in $files; do echo "$(s12 $f) $f"; done; } > $R/.FROZEN.t5new
mv -f $R/.FROZEN.t5new $R/FROZEN.txt
chmod a-w $R/FROZEN.txt
# every frozen sha as maybe_select.sh checks it
bad=0
while read -r sha f; do case "$sha" in \#*|"") continue;; esac; [ "$(s12 $f)" = "$sha" ] || { echo "MISMATCH $f"; bad=1; }; done < $R/FROZEN.txt
[ $bad = 0 ] || exit 1
for f in pick_tpp.py maybe_select.sh; do cmp <(git -C $WT show "$C:notes/pttpp/$f") $R/$f; done
rm -f $R/HOLD
echo "[$(date -Is)] deployed $C; HOLD removed"
cat $R/FROZEN.txt
