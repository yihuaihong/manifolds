#!/bin/bash
# pt6v2 submission wrapper (phase 2b, 2026-11-19): the canary + the 104 main runs (8 seeds x 13 cells; AUDIT.md Amendment V1 added none@4e-3) through
# make_jobs_pt6v2.py --submit (B200 seeds via the singleton slot chains llmadd_pt6v2-b200-slot0..7, pt6comb's / ptdom's protocol;
# every cell of a seed on that seed's device; the canary first). Dry by default: checks only; --go submits.
#   bash runs/pt6v2/submit_pt6v2.sh [--alloc 'b200:16,17,18 l40s:19,20,21 a100:22,23'] [--canary_dev l40s|a100] [--extra_none 6.4e-2] [--go]
# V2 build: --extra_none 6.4e-2 (AUDIT.md Amendment V2) is passed to make_jobs; refuses unless runs/pt6v2/CELLS.json = this cell set.
# Refuses (exit 3): runs/pt6v2/HOLD or HOLD_<dev> / HOLD_canary; /scratch quota >= 97%; and, through make_jobs --submit: no
# COMPOSITION.json composition, no refs for it, no readout_pt6v2.py, job files not written or differing from the rendering
# (the coordinator runs `make_jobs_pt6v2.py --alloc ... --check --write` first). --go also refuses while a non-B200 device in the
# alloc has no passed GPU smoke (runs/pt6v2/pilot/smoke_check_<dev>.out from tests/smoke_check_pt6v2.py, AUDIT_phase2b.md).
set -euo pipefail
B=/scratch/yh6210/llmadd_torch
R=$B/runs/pt6v2
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
export PYTHONDONTWRITEBYTECODE=1
ALLOC='b200:16,17,18 l40s:19,20,21 a100:22,23'
CDEV=()
EXN=()
GO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift;;
    --alloc) ALLOC="$2"; shift 2;;
    --canary_dev) CDEV=(--canary_dev "$2"); shift 2;;
    --extra_none) EXN=(--extra_none "$2"); shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
fail() { echo "## REFUSED: $1"; exit 3; }
for h in "$R/HOLD" "$R/HOLD_canary" "$R"/HOLD_b200 "$R"/HOLD_l40s "$R"/HOLD_a100; do [ ! -e "$h" ] || fail "$h exists (the user decides)"; done
pct=$(myquota 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | awk '$1 == "/scratch" {print $0}' | grep -oE '[0-9.]+%' | head -1 | tr -d '%' || true)
[ -n "$pct" ] || fail "cannot read the /scratch quota (myquota)"
awk -v p="$pct" 'BEGIN { exit !(p < 97) }' || fail "/scratch quota use $pct% >= 97%"
echo "/scratch quota use $pct% (< 97%)"
[ -e "$R/COMPOSITION.json" ] || fail "no runs/pt6v2/COMPOSITION.json (the pick-time procedure, AUDIT.md)"
[ -e "$R/readout_pt6v2.py" ] || fail "no runs/pt6v2/readout_pt6v2.py"
cd "$B"
read -r -a AL <<< "$ALLOC"
"$PY" make_jobs_pt6v2.py --snapshot
mkdir -p "$R/tests"
if grep -q -- '--selftest' "$R/readout_pt6v2.py"; then
  ST="$R/tests/selftest_readout_submit_$(date -u +%Y%m%dT%H%M%SZ)_$$.out"
  PYTHONPATH=$B/pylib "$PY" "$R/readout_pt6v2.py" --selftest > "$ST" 2>&1 || { tail -n 5 "$ST"; fail "the read-out self-test failed ($ST)"; }
  echo "read-out self-test: $(tail -n 1 "$ST") ($ST)"
else
  echo "## NOTE: readout_pt6v2.py has no --selftest; not run"
fi
for a in "${AL[@]}"; do
  dev=${a%%:*}
  [ "$dev" = b200 ] && continue
  if ! grep -q "^SMOKE $dev PASS" "$R/pilot/smoke_check_$dev.out" 2>/dev/null; then
    [ "$GO" = 0 ] && echo "## NOTE: the $dev GPU smoke has not passed yet (runs/pt6v2/pilot/smoke_check_$dev.out); --go refuses" \
                  || fail "the $dev GPU smoke has not passed (runs/pt6v2/pilot/smoke_check_$dev.out)"
  fi
done
CK="$R/tests/submit_check.out"
"$PY" make_jobs_pt6v2.py --alloc "${AL[@]}" "${CDEV[@]}" "${EXN[@]}" --check --print --disk > "$CK" 2>&1 || { tail -n 5 "$CK"; fail "--check ($CK)"; }
tail -n 3 "$CK"
grep -q '^disk: unsubmitted job files {.same.: [0-9]*, .missing.: 0, .differ.: 0}' "$CK" \
  || fail "job files not written or differing from the rendering (make_jobs_pt6v2.py --alloc '$ALLOC' --check --write first; $CK)"
grep -q '^cells manifest: same ' "$CK" \
  || fail "runs/pt6v2/CELLS.json missing or differing from this cell set (--write and this call must agree on --extra_none; $CK)"
first=$(grep -m1 -E '^pt6v2-[^ ]+ \[[a-z0-9]+\]' "$CK" | cut -d' ' -f1)
[[ "$first" == pt6v2-canary-* ]] || fail "the first job is $first, not the canary"
echo "alloc: $ALLOC; extra_none: ${EXN[1]:-none}"
for dev in b200 l40s a100; do
  n=$(grep -cE "^pt6v2-[^ ]+ \[$dev\]" "$CK" || true); s=$(grep -cE "^pt6v2-[^ ]+ \[$dev\] \(SUBMITTED\)" "$CK" || true)
  echo "  $dev: $n jobs ($s submitted before)"
done
n_all=$(grep -cE '^pt6v2-[^ ]+ \[[a-z0-9]+\]' "$CK" || true); n_sub=$(grep -c ' (SUBMITTED): ' "$CK" || true)
echo "total: $n_all jobs (canary first: $first), $n_sub submitted before, $((n_all - n_sub)) to submit"
[ -e "$B/runs/ptdom/NOTIFY_misalign_when_b200_free.txt" ] && echo "## NOTE (B200 sharing): runs/ptdom/NOTIFY_misalign_when_b200_free.txt exists: tell the misalign session before pt6v2 occupies B200 (coordinator)"
if [ "$GO" = 1 ]; then
  "$PY" make_jobs_pt6v2.py --alloc "${AL[@]}" "${CDEV[@]}" "${EXN[@]}" --submit
else
  echo "(checks only; --go submits the $((n_all - n_sub)) unsubmitted jobs listed in $CK)"
fi
