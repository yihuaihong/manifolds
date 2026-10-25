#!/bin/bash
# Amendment S1.2 (runs/rb6_505m/AUDIT.md): the 505M main read-out for one X1 outcome, as a script (Amendment S1
# "Command and picks-file use"). Run only in the order of S1 "Reading order", with no llmadd_rb6_505m job queued or
# running (checked for the real batch dir).
#     bash runs/rb6_505m/readout_505m.sh a|b|b-pre|c [--allow_partial]
# The partial form exists only for outcome b with none@2.4e-3 s2 / s3 missing after the X1.1 #2 resubmissions; the
# read-out code enforces that.
# It checks:
#   - the frozen read-out code against readout_code/COMMIT;
#   - picks.json against its pinned sha (8d285d925c66);
#   - picks_x1.json against picks_x1.sha (written by x1_select.sh).
# It writes runs/rb6_505m/readout_505m_x1<outcome>.{json,txt} (stdout + stderr) and never overwrites them.
# For tests on a fake batch dir: R=<dir> PICKS_SHA=<sha>.
set -u
o=${1:?outcome a / b / b-pre / c}
PARTIAL=${2:-}
REAL=/scratch/yh6210/llmadd_torch/runs/rb6_505m
R=${R:-$REAL}
if [ "$R" = "$REAL" ]; then PICKS_SHA=8d285d925c66; else PICKS_SHA=${PICKS_SHA:?a test batch dir needs PICKS_SHA}; fi
R190=/scratch/yh6210/llmadd_torch/runs/rb6_190m/out
BR=/scratch/yh6210/llmadd-burst-ro/results
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib PYTHONDONTWRITEBYTECODE=1
case "$o" in
  a|b) P=$R/picks_x1.json ;;
  b-pre|c) P=$R/picks.json ;;
  *) echo "## outcome a / b / b-pre / c"; exit 2 ;;
esac
[ -z "$PARTIAL" ] || [ "$PARTIAL" = --allow_partial ] || { echo "## the 2nd argument can only be --allow_partial"; exit 2; }
OUT=$R/readout_505m_x1${o//-/_}
[ ! -e "$OUT.json" ] && [ ! -e "$OUT.txt" ] || { echo "## $OUT.json / .txt exist: not overwritten"; exit 3; }
if [ "$R" = "$REAL" ]; then
  n=$(squeue -h -u "$USER" -o %j | grep -c '^llmadd_rb6_505m-')
  [ "$n" = 0 ] || { echo "## $n llmadd_rb6_505m jobs queued / running: wait (S1 Reading order)"; exit 3; }
fi
[ "$(sha256sum "$R/picks.json" | cut -c1-12)" = "$PICKS_SHA" ] || { echo "## picks.json != its pinned sha $PICKS_SHA"; exit 3; }
if [ -e "$R/picks_x1.json" ]; then
  [ "$(sha256sum "$R/picks_x1.json" | cut -c1-12)" = "$(cat "$R/picks_x1.sha" 2>/dev/null)" ] ||
    { echo "## picks_x1.json != picks_x1.sha (x1_select.sh's record)"; exit 3; }
fi
cd "$R/readout_code/tools" || exit 3
for f in rb6scale_summary.py rainbow6_summary.py make_rb6m_tasks.py make_rb6_tasks.py readout_rules.py rb6scale_select.py; do
  grep -q "$f $(sha256sum "$f" | cut -c1-12)" ../COMMIT || { echo "## $f differs from readout_code/COMMIT"; exit 3; }
done
ROOTS=("$R/out" "$R190" "$BR"/401-rb6pb-* "$BR"/404-rb6pb-* "$BR"/430-rb6pb-* "$BR"/431-rb6pb-* "$BR"/419-rb6pb-*
       "$BR"/415-rb6pb-* "$BR"/432-rb6pb-* "$BR"/433-rb6pb-* "$BR"/438-rb6mb-* "$BR"/444-rb6mb-* "$BR"/450-rb6mb-*
       "$BR"/456-rb6mb-* "$BR"/46[6-9]-rb6mb-*)
[ "${#ROOTS[@]}" = 18 ] || { echo "## ${#ROOTS[@]} roots, not 18 (rb6_505m/out, rb6_190m/out, 16 53M dirs)"; exit 3; }
"$PY" rb6scale_summary.py "${ROOTS[@]}" --picks "$P" --x1 "$o" --x1_json "$R/picks_x1.json" --json "$OUT.json" $PARTIAL \
  > "$OUT.txt" 2>&1
rc=$?
echo "rc $rc" >> "$OUT.txt"
chmod a-w "$OUT.txt"; [ -e "$OUT.json" ] && chmod a-w "$OUT.json"
echo "== 505M read-out, outcome $o: exit $rc -> $OUT.txt"
exit $rc
