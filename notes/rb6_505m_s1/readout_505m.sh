#!/bin/bash
# Amendments S1.2 - S1.4 (runs/rb6_505m/AUDIT.md): the 505M main read-out for one X1 outcome, as a script. Run only in
# the order of S1 "Reading order", with no llmadd_rb6_505m job queued or running (checked for the real batch dir).
#     bash runs/rb6_505m/readout_505m.sh a|b|b-pre|c [--allow_partial]
# --allow_partial exists only for outcome b with none@2.4e-3 s2 / s3 missing after the X1.1 #2 resubmissions; the
# read-out code enforces that.
# It checks:
#   - the frozen read-out code against readout_code/COMMIT;
#   - picks.json against its pinned sha (8d285d925c66);
#   - picks_x1.json against picks_x1.sha (x1_select.sh);
#   - the 18 roots exist.
# A successful read-out goes to runs/rb6_505m/readout_505m_x1<a|b|b_pre|c>[_partial].{json,txt}: stdout + stderr, a
# header with the script / code shas, rc. It is read-only and never overwritten. A failed run is kept as
# <name>.failed-<time>.txt and does not block a later run.
# For tests on a fake batch dir: R=<dir> PICKS_SHA=<sha>; the real dir (compared after realpath) ignores PICKS_SHA.
set -uo pipefail
[ $# -ge 1 ] && [ $# -le 2 ] || { echo "## usage: readout_505m.sh a|b|b-pre|c [--allow_partial]"; exit 2; }
o=$1
PARTIAL=${2:-}
SELF_SHA=$(sha256sum "$(realpath "$0")" | cut -c1-12)   # S1.4: before any cd (a relative $0 breaks after it)
REAL=/scratch/yh6210/llmadd_torch/runs/rb6_505m
R=$(realpath -m "${R:-$REAL}")
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
OUT=$R/readout_505m_x1${o//-/_}${PARTIAL:+_partial}
[ ! -e "$OUT.json" ] && [ ! -e "$OUT.txt" ] || { echo "## $OUT.json / .txt exist: not overwritten"; exit 3; }
if [ "$R" = "$REAL" ]; then
  q=$(squeue -h -u "$(id -un)" -o %j) || { echo "## squeue failed: the queue cannot be checked"; exit 3; }
  n=$(grep -c '^llmadd_rb6_505m-' <<< "$q" || true)
  [ "$n" = 0 ] || { echo "## $n llmadd_rb6_505m jobs queued / running: wait (S1 Reading order)"; exit 3; }
fi
s=$(sha256sum "$R/picks.json" | cut -c1-12) && [ "$s" = "$PICKS_SHA" ] || { echo "## picks.json != its pinned sha $PICKS_SHA"; exit 3; }
if [ -e "$R/picks_x1.json" ]; then
  s=$(sha256sum "$R/picks_x1.json" | cut -c1-12) && [ "$s" = "$(cat "$R/picks_x1.sha" 2>/dev/null)" ] ||
    { echo "## picks_x1.json != picks_x1.sha (x1_select.sh's record)"; exit 3; }
fi
cd "$R/readout_code/tools" || exit 3
for f in rb6scale_summary.py rainbow6_summary.py make_rb6m_tasks.py make_rb6_tasks.py readout_rules.py rb6scale_select.py; do
  [ -f "$f" ] || { echo "## $f missing in readout_code"; exit 3; }
  s=$(sha256sum "$f" | cut -c1-12)
  [ "${#s}" = 12 ] && grep -qF "$f $s" ../COMMIT || { echo "## $f differs from readout_code/COMMIT"; exit 3; }
done
ROOTS=("$R/out" "$R190" "$BR"/401-rb6pb-* "$BR"/404-rb6pb-* "$BR"/430-rb6pb-* "$BR"/431-rb6pb-* "$BR"/419-rb6pb-*
       "$BR"/415-rb6pb-* "$BR"/432-rb6pb-* "$BR"/433-rb6pb-* "$BR"/438-rb6mb-* "$BR"/444-rb6mb-* "$BR"/450-rb6mb-*
       "$BR"/456-rb6mb-* "$BR"/46[6-9]-rb6mb-*)
[ "${#ROOTS[@]}" = 18 ] || { echo "## ${#ROOTS[@]} roots, not 18 (rb6_505m/out, rb6_190m/out, 16 53M dirs)"; exit 3; }
for d in "${ROOTS[@]}"; do [ -d "$d" ] || { echo "## root $d is not a directory"; exit 3; }; done
ts=$(date -u +%Y%m%dT%H%M%SZ)
TMP=$OUT.running-$ts.txt
[ "${#SELF_SHA}" = 12 ] || { echo "## cannot hash the script itself"; exit 3; }
{ echo "== readout_505m.sh (sha256[:12] $SELF_SHA) outcome $o $PARTIAL at $ts; R=$R"
  echo "== readout_code/COMMIT: $(cat ../COMMIT)"; } > "$TMP"
"$PY" rb6scale_summary.py "${ROOTS[@]}" --picks "$P" --x1 "$o" --x1_json "$R/picks_x1.json" --json "$OUT.json" $PARTIAL \
  >> "$TMP" 2>&1
rc=$?
echo "rc $rc" >> "$TMP"
if [ "$rc" = 0 ]; then
  mv "$TMP" "$OUT.txt"
  chmod a-w "$OUT.txt" "$OUT.json"
  echo "== 505M read-out, outcome $o $PARTIAL: exit 0 -> $OUT.txt"
else
  mv "$TMP" "$OUT.failed-$ts.txt"
  [ ! -e "$OUT.json" ] || mv "$OUT.json" "$OUT.failed-$ts.json"
  echo "== 505M read-out, outcome $o $PARTIAL: exit $rc (refused / failed) -> $OUT.failed-$ts.txt"
  tail -n 3 "$OUT.failed-$ts.txt"
fi
exit "$rc"
