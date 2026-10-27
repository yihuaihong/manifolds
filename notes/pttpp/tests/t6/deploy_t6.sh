#!/bin/bash
# T6 deployment: only pick_tpp.py changes (the read-out snapshot, maybe_select.sh and the generator stay). HOLD, wait until
# no selection holds a group lock, put pick_tpp.py (git show <commit>) in place by an atomic rename, rewrite its line of
# FROZEN.txt, re-check every frozen sha, remove HOLD.
#   bash deploy_t6.sh <commit>
set -eu
C=${1:?commit}
R=/scratch/yh6210/llmadd_torch/runs/pttpp
WT=/scratch/yh6210/llmadd-burst-tpp
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
s12() { sha256sum "$1" | cut -c1-12; }
[ -z "$(git -C $WT diff fd8901b "$C" -- tools/ notes/pttpp/maybe_select.sh)" ] || { echo "tools/ or maybe_select.sh changed since T5"; exit 1; }
touch $R/HOLD
echo "[$(date -Is)] HOLD set"
$PY $HERE/wait_nolock.py 90
old=$(s12 $R/pick_tpp.py)
git -C $WT show "$C:notes/pttpp/pick_tpp.py" > $R/.pick_tpp.py.t6new
chmod a-w $R/.pick_tpp.py.t6new
mv -f $R/.pick_tpp.py.t6new $R/pick_tpp.py
new=$(s12 $R/pick_tpp.py)
chmod u+w $R/FROZEN.txt
sed -e "s|^$old $R/pick_tpp.py\$|$new $R/pick_tpp.py|" -e "s|Amendment T5)\$|Amendment T6)|" $R/FROZEN.txt > $R/.FROZEN.t6new
[ "$(grep -c "^$new $R/pick_tpp.py\$" $R/.FROZEN.t6new)" = 1 ] || { echo "FROZEN line not replaced"; exit 1; }
mv -f $R/.FROZEN.t6new $R/FROZEN.txt
chmod a-w $R/FROZEN.txt
bad=0
while read -r sha f; do case "$sha" in \#*|"") continue;; esac; [ "$(s12 $f)" = "$sha" ] || { echo "MISMATCH $f"; bad=1; }; done < $R/FROZEN.txt
[ $bad = 0 ] || exit 1
cmp <(git -C $WT show "$C:notes/pttpp/pick_tpp.py") $R/pick_tpp.py
rm -f $R/HOLD
echo "[$(date -Is)] deployed pick_tpp.py $old -> $new ($C); $(grep -vc '^#' $R/FROZEN.txt) frozen files re-checked; HOLD removed"
