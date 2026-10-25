#!/bin/bash
# Amendment S1.2 (runs/rb6_505m/AUDIT.md): the X1 selection of Addendum X1.1 #1, once, as a script.
# Runs the frozen runs/rb6_505m/x1_code/tools/rb6scale_select.py --grid_none over runs/rb6_505m/out. It writes
# picks_x1.json (the selection's JSON; absent after an integrity failure), picks_x1.exit (the selection's exit code)
# and picks_x1.sha (sha256[:12] of picks_x1.json), and x1_select.log (stdout + stderr). It refuses when any of those
# exist, when a file differs from x1_code/COMMIT, or, for the real batch dir, when an llmadd_rb6_505m job is still
# queued or running. Nothing is submitted.
#     bash runs/rb6_505m/x1_select.sh            (R=<dir> only for tests on a fake batch dir)
set -u
REAL=/scratch/yh6210/llmadd_torch/runs/rb6_505m
R=${R:-$REAL}
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib PYTHONDONTWRITEBYTECODE=1
echo "== X1 selection $(date -u +%FT%TZ) R=$R"
for f in picks_x1.json picks_x1.exit picks_x1.sha x1_select.log; do
  [ ! -e "$R/$f" ] || { echo "## $R/$f exists: the X1 selection runs once"; exit 3; }
done
if [ "$R" = "$REAL" ]; then
  n=$(squeue -h -u "$USER" -o %j | grep -c '^llmadd_rb6_505m-')
  [ "$n" = 0 ] || { echo "## $n llmadd_rb6_505m jobs queued / running: wait (Addendum X1.1 #1)"; exit 3; }
else
  echo "(test batch dir: the squeue check is skipped)"
fi
cd "$R/x1_code/tools" || exit 3
for f in rb6scale_select.py rb6scale_summary.py rainbow6_summary.py make_rb6m_tasks.py make_rb6_tasks.py readout_rules.py; do
  grep -q "$f $(sha256sum "$f" | cut -c1-12)" ../COMMIT || { echo "## $f differs from x1_code/COMMIT"; exit 3; }
done
"$PY" rb6scale_select.py "$R/out" --grid_none 2.4e-3,4.8e-3,9.6e-3,1.92e-2 --json "$R/picks_x1.json" > "$R/x1_select.log" 2>&1
rc=$?
echo "$rc" > "$R/picks_x1.exit"
if [ -e "$R/picks_x1.json" ]; then sha256sum "$R/picks_x1.json" | cut -c1-12 > "$R/picks_x1.sha"; else echo absent > "$R/picks_x1.sha"; fi
chmod a-w "$R/picks_x1.exit" "$R/picks_x1.sha" "$R/x1_select.log"; [ -e "$R/picks_x1.json" ] && chmod a-w "$R/picks_x1.json"
cat "$R/x1_select.log"
echo "== X1 selection exit $rc; picks_x1.json sha256[:12] $(cat "$R/picks_x1.sha") (record both in AUDIT.md)"
