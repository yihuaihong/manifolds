#!/bin/bash
# Amendments S1.2 / S1.3 (runs/rb6_505m/AUDIT.md): the X1 selection of Addendum X1.1 #1, once, as a script.
# Runs the frozen runs/rb6_505m/x1_code/tools/rb6scale_select.py --grid_none over runs/rb6_505m/out.
# A DATA outcome writes these records, read-only:
#   - picks_x1.json (absent after an integrity failure),
#   - picks_x1.exit (the selection's exit code),
#   - picks_x1.sha (sha256[:12] of picks_x1.json, or "absent"),
#   - x1_select.log.
# A data outcome is exit 0 with a JSON (picked or incomplete), or the integrity assert of rainbow6_summary.check (the
# AssertionError raised in check(), no JSON). Any other failure records nothing; its log is kept as
# x1_select.failed-<time>.log and the script may be run again once it is fixed. Examples: environment, imports, a
# signal, a missing file.
# Refuses (exit 3, nothing recorded) when:
#   - a record already exists;
#   - a frozen file differs from x1_code/COMMIT or is missing;
#   - the environment does not import the selection's modules;
#   - for the real batch dir: squeue fails, or an llmadd_rb6_505m job is queued or running;
#   - none@2.4e-3 s0 / s1 are not both present and finished. Only once their X1.1 #2 resubmissions are used up may
#     --x1-runs-final run the selection anyway, which then writes an "incomplete" JSON, i.e. outcome c.
#     bash runs/rb6_505m/x1_select.sh [--x1-runs-final]        (R=<dir> only for tests on a fake batch dir)
set -uo pipefail
REAL=/scratch/yh6210/llmadd_torch/runs/rb6_505m
R=$(realpath -m "${R:-$REAL}")
FINAL=${1:-}
{ [ $# -le 1 ] && { [ -z "$FINAL" ] || [ "$FINAL" = --x1-runs-final ]; }; } || { echo "## usage: x1_select.sh [--x1-runs-final]"; exit 2; }
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
export HF_HOME=/scratch/yh6210/llmadd_torch/hf_cache PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib PYTHONDONTWRITEBYTECODE=1
echo "== X1 selection $(date -u +%FT%TZ) R=$R $FINAL; script sha256[:12] $(sha256sum "$0" | cut -c1-12)"
for f in picks_x1.json picks_x1.exit picks_x1.sha x1_select.log; do
  [ ! -e "$R/$f" ] || { echo "## $R/$f exists: the X1 selection has a data outcome already (it runs once)"; exit 3; }
done
if [ "$R" = "$REAL" ]; then
  q=$(squeue -h -u "$(id -un)" -o %j) || { echo "## squeue failed: the queue cannot be checked"; exit 3; }
  n=$(grep -c '^llmadd_rb6_505m-' <<< "$q" || true)
  [ "$n" = 0 ] || { echo "## $n llmadd_rb6_505m jobs queued / running: wait (Addendum X1.1 #1)"; exit 3; }
else
  echo "(test batch dir: the squeue check is skipped)"
fi
cd "$R/x1_code/tools" || exit 3
for f in rb6scale_select.py rb6scale_summary.py rainbow6_summary.py make_rb6m_tasks.py make_rb6_tasks.py readout_rules.py; do
  [ -f "$f" ] || { echo "## $f missing in x1_code"; exit 3; }
  s=$(sha256sum "$f" | cut -c1-12)
  [ "${#s}" = 12 ] && grep -qF "$f $s" ../COMMIT || { echo "## $f differs from x1_code/COMMIT"; exit 3; }
done
"$PY" -c "import numpy, rainbow6_summary, rb6scale_summary, rb6scale_select" ||
  { echo "## the environment does not import the selection's modules: fix it (nothing recorded)"; exit 3; }
"$PY" - "$R/out" <<'PY'
import json, sys
from pathlib import Path
tag, bad = "rb6tb-505m-2pct-k2-none-lr2.4e-3", []
for s in (0, 1):
    ps = list((Path(sys.argv[1]) / f"{tag}-s{s}" / tag).glob("run_*.json"))
    if len(ps) != 1:
        bad.append(f"s{s}: no run JSON")
        continue
    r = json.loads(ps[0].read_text())
    if r["curve"][-1]["step"] != r["steps"] - 1:
        bad.append(f"s{s}: unfinished (last eval {r['curve'][-1]['step']})")
print("X1 runs none@2.4e-3:", "; ".join(bad) if bad else "s0 and s1 present and finished")
sys.exit(1 if bad else 0)
PY
st=$?
if [ "$st" != 0 ] && [ "$FINAL" != --x1-runs-final ]; then
  echo "## the X1 runs are not final: wait / resubmit (Addendum X1.1 #2); once the resubmissions are used up, run with"
  echo "## --x1-runs-final (the selection then records the missing runs: incomplete = outcome c). Nothing recorded."
  exit 3
fi
LOG=$R/x1_select.run-$(date -u +%Y%m%dT%H%M%SZ).log
"$PY" rb6scale_select.py "$R/out" --grid_none 2.4e-3,4.8e-3,9.6e-3,1.92e-2 --json "$R/picks_x1.json" > "$LOG" 2>&1
rc=$?
cat "$LOG"
data=0
if [ "$rc" = 0 ] && [ -e "$R/picks_x1.json" ]; then
  data=1
elif [ "$rc" != 0 ] && [ ! -e "$R/picks_x1.json" ]; then
  lastf=$(grep '^  File ' "$LOG" | tail -n 1); lastl=$(tail -n 1 "$LOG")
  case "$lastf" in *'rainbow6_summary.py", line '*', in check') case "$lastl" in AssertionError*) data=1 ;; esac ;; esac
fi
if [ "$data" != 1 ]; then
  mv "$LOG" "$R/x1_select.failed-${LOG##*run-}"
  [ ! -e "$R/picks_x1.json" ] || mv "$R/picks_x1.json" "$R/picks_x1.failed-${LOG##*run-}.json"
  echo "## the selection failed for a NON-data reason (exit $rc): nothing recorded; log kept as x1_select.failed-${LOG##*run-}"
  exit 3
fi
mv "$LOG" "$R/x1_select.log"
echo "$rc" > "$R/picks_x1.exit"
if [ -e "$R/picks_x1.json" ]; then sha256sum "$R/picks_x1.json" | cut -c1-12 > "$R/picks_x1.sha"; else echo absent > "$R/picks_x1.sha"; fi
chmod a-w "$R/picks_x1.exit" "$R/picks_x1.sha" "$R/x1_select.log"
[ ! -e "$R/picks_x1.json" ] || chmod a-w "$R/picks_x1.json"
echo "== X1 selection: data outcome, exit $rc; picks_x1.json sha256[:12] $(cat "$R/picks_x1.sha") (record both in AUDIT.md)"
exit "$rc"
