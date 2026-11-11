#!/bin/bash
# pt6comb submission helper (runs/pt6comb/AUDIT.md "Submission"). Without --go it only CHECKS and PRINTS; with --go it submits the part
# through make_jobs_pt6comb.py --part X --submit (B200 singleton slot chains llmadd_pt6comb-b200-slot0..7, ptdom's protocol).
#   bash runs/pt6comb/submit_pt6comb.sh --part smoke [--go]                 (the one GPU smoke; not part of any result)
#   bash runs/pt6comb/submit_pt6comb.sh --part L1 [--go]                    (24 runs, seeds 0-3)
#   bash runs/pt6comb/submit_pt6comb.sh --part L1x|C --picks runs/pt6comb/picks_after_<L1|L1x>.json [--go]
# Refuses unless: no runs/pt6comb/HOLD and no HOLD_<part>; /scratch quota use < 97%; the snapshot (runs/pt6comb/code = 11f61b9) and the
# frozen shas verify (the generator asserts make_jobs_ptdom.py / make_jobs_ptdom2.py, the read-out readout_ptdom.py); the read-out
# self-test passes (dated log in tests/); L1 / L1x / C: the GPU smoke passed (smoke/smoke_check.out 'SMOKE PASS'); --check renders
# every job of the part (bash -n, every render assertion); NO job of the part was submitted before (a re-run is a new name:
# make_jobs_pt6comb.py --part X --rerun); L1x / C: the picks file is read-only and equals its re-derivation (select_pt6comb.py --verify).
# Composition B only (A is a parameter, not submittable). Kill switch: touch runs/pt6comb/HOLD (every part) or runs/pt6comb/HOLD_<part>.
set -euo pipefail
B=/scratch/yh6210/llmadd_torch
R=$B/runs/pt6comb
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
export PYTHONDONTWRITEBYTECODE=1
PART=""; PICKS=""; GO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART=$2; shift 2 ;; --picks) PICKS=$2; shift 2 ;; --go) GO=1; shift ;;
    *) echo "## unknown argument $1"; exit 64 ;;
  esac
done
case "$PART" in smoke|L1|L1x|C) ;; *) echo "## --part smoke | L1 | L1x | C"; exit 64 ;; esac
if [ "$PART" = L1x ] && [ -z "$PICKS" ]; then PICKS=$R/picks_after_L1.json; fi
if [ "$PART" = C ] && [ -z "$PICKS" ]; then
  if [ -e "$R/picks_after_L1x.json" ]; then PICKS=$R/picks_after_L1x.json; else PICKS=$R/picks_after_L1.json; fi
fi
fail() { echo "## REFUSED: $1"; exit 3; }
for h in "$R/HOLD" "$R/HOLD_$PART"; do [ ! -e "$h" ] || fail "$h exists (the user decides)"; done
pct=$(myquota 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | awk '$1 == "/scratch" {print $0}' | grep -oE '[0-9.]+%' | head -1 | tr -d '%' || true)
[ -n "$pct" ] || fail "cannot read the /scratch quota (myquota)"
awk -v p="$pct" 'BEGIN { exit !(p < 97) }' || fail "/scratch quota use $pct% >= 97%"
echo "/scratch quota use $pct% (< 97%)"
cd "$B"
"$PY" make_jobs_pt6comb.py --plan
"$PY" make_jobs_pt6comb.py --snapshot
mkdir -p "$R/tests"
ST="$R/tests/selftest_readout_submit_$(date -u +%Y%m%dT%H%M%SZ)_$$.out"
PYTHONPATH=$B/pylib "$PY" "$R/readout_pt6comb.py" --selftest > "$ST" 2>&1 || { tail -n 5 "$ST"; fail "the read-out self-test failed ($ST)"; }
echo "read-out self-test: $(tail -n 1 "$ST") ($ST)"
if [ "$PART" != smoke ]; then
  grep -q '^SMOKE PASS' "$R/smoke/smoke_check.out" 2>/dev/null || { [ "$GO" = 0 ] && echo "## NOTE: the GPU smoke has not passed yet (runs/pt6comb/smoke/smoke_check.out); --go refuses" || fail "the GPU smoke has not passed (runs/pt6comb/smoke/smoke_check.out)"; }
fi
PK=()
if [ "$PART" = L1x ] || [ "$PART" = C ]; then
  [ -e "$PICKS" ] || fail "no picks file $PICKS (select_pt6comb.py first)"
  [ ! -w "$PICKS" ] || fail "$PICKS is writable (a decision file is chmod 444)"
  PYTHONPATH=$B/pylib "$PY" "$R/select_pt6comb.py" --verify "$PICKS" || fail "$PICKS differs from its re-derivation"
  PK=(--picks "$PICKS")
fi
SF=(--part "$PART"); [ "$PART" = smoke ] && SF=(--smoke)
"$PY" make_jobs_pt6comb.py "${SF[@]}" "${PK[@]}" --check --print > "$R/tests/submit_check_$PART.out" 2>&1 || { tail -n 5 "$R/tests/submit_check_$PART.out"; fail "--check"; }
tail -n 2 "$R/tests/submit_check_$PART.out"
n_sub=$(grep -c ' (SUBMITTED): ' "$R/tests/submit_check_$PART.out" || true)
[ "$n_sub" = 0 ] || fail "$n_sub jobs of part $PART were submitted before (re-runs: make_jobs_pt6comb.py --part $PART ${PK[*]} --rerun)"
n_jobs=$(grep -c '^pt6comb-.*: --data ' "$R/tests/submit_check_$PART.out" || true)
echo "part $PART: $n_jobs jobs, none submitted before"
[ -e "$B/runs/ptdom/NOTIFY_misalign_when_b200_free.txt" ] && echo "## NOTE (B200 sharing): runs/ptdom/NOTIFY_misalign_when_b200_free.txt exists: tell the misalign session before pt6comb occupies B200 (coordinator)"
if [ "$GO" = 1 ]; then
  "$PY" make_jobs_pt6comb.py "${SF[@]}" "${PK[@]}" --submit
else
  echo "(checks only; --go submits the $n_jobs jobs listed in runs/pt6comb/tests/submit_check_$PART.out)"
fi
cat <<EOF
Order (each after every job of the previous step has finished with check.json; AUDIT.md "Submission"):
  bash runs/pt6comb/submit_pt6comb.sh --part smoke --go      # the GPU smoke (<= 1 h); then read runs/pt6comb/smoke/smoke_check.out
  bash runs/pt6comb/submit_pt6comb.sh --part L1 --go         # 24 runs (seeds 0-3)
  python runs/pt6comb/select_pt6comb.py --after L1           # -> picks_after_L1.json (once, chmod 444): L1x asked, or final (C); GATE: rc 2
  bash runs/pt6comb/submit_pt6comb.sh --part L1x --go        # only if picks_after_L1.json asks for extensions
  python runs/pt6comb/select_pt6comb.py --after L1x          # -> picks_after_L1x.json (final; NOT FINAL: rc 2, the coordinator decides)
  bash runs/pt6comb/submit_pt6comb.sh --part C --go          # 64-80 runs (A2: + none@2e-3 / all6@2e-3 unless merged), seeds 8-15
  PYTHONPATH=$B/pylib python runs/pt6comb/readout_pt6comb.py --json runs/pt6comb/readout.json > runs/pt6comb/readout.txt
                                                             # refuses while a needed run is missing (--allow_partial: PROVISIONAL)
Re-runs (a job that exited before its run finished, e.g. at HOLD or a node fault; a finished run is never re-run):
  python make_jobs_pt6comb.py --part X [--picks f] --rerun --print ; then --rerun --submit
EOF
