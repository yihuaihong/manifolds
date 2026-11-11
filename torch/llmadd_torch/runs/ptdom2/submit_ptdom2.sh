#!/bin/bash
# ptdom2 submission helper (runs/ptdom2/AUDIT.md "Submission"). Without --go it only CHECKS and PRINTS; with --go it submits the
# part through make_jobs_ptdom2.py --part X --submit (B200 singleton slot chains llmadd_ptdom2-b200-slot0..7, ptdom's protocol).
#   bash runs/ptdom2/submit_ptdom2.sh --part S|R1|R1pre [--go]        (S0: optional, descriptive only, A2.1; --go needs --allow_s0, A3.3)
#   bash runs/ptdom2/submit_ptdom2.sh --part R1x|R2 [--picks runs/ptdom2/picks_R_after<k>.json] [--go]
# Refuses unless: no runs/ptdom2/HOLD and no HOLD_<part>; /scratch quota use < 97%; the snapshot (runs/ptdom2/code = 11f61b9) and
# make_jobs_ptdom.py / readout_ptdom.py shas verify; the read-out self-test passes; the GPU pilot passed (pilot/pilot_check.out:
# 'PILOT PASS'); --check renders every job of the part (bash -n, every render assertion); NO job of the part was submitted before
# (a re-run is a new name: make_jobs_ptdom2.py --part X --rerun); R1x / R2: the picks file is read-only and equals its
# re-derivation (select_ptdom2.py --verify).
# Kill switch: touch runs/ptdom2/HOLD (every part) or runs/ptdom2/HOLD_<part>; pending / requeued jobs exit at start.
set -euo pipefail
B=/scratch/yh6210/llmadd_torch
R=$B/runs/ptdom2
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
PART=""; PICKS=""; GO=0; ALLOW_S0=0
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART=$2; shift 2 ;; --picks) PICKS=$2; shift 2 ;; --go) GO=1; shift ;; --allow_s0) ALLOW_S0=1; shift ;;
    *) echo "## unknown argument $1"; exit 64 ;;
  esac
done
case "$PART" in S|S0|R1|R1pre|R1x|R2) ;; *) echo "## --part S | S0 | R1 | R1pre | R1x | R2"; exit 64 ;; esac
if [ "$PART" = R1x ] && [ -z "$PICKS" ]; then PICKS=$R/picks_R_after1.json; fi
if [ "$PART" = R2 ] && [ -z "$PICKS" ]; then
  if [ -e "$R/picks_R_after2.json" ]; then PICKS=$R/picks_R_after2.json; else PICKS=$R/picks_R_after1.json; fi
fi
fail() { echo "## REFUSED: $1"; exit 3; }
# A3.3: part S0 is optional and descriptive (A2.1): --go needs the coordinator's explicit --allow_s0
[ "$PART" != S0 ] || [ "$GO" = 0 ] || [ "$ALLOW_S0" = 1 ] || fail "part S0 is optional (A2.1): submit it only on the coordinator's explicit decision (--allow_s0)"
for h in "$R/HOLD" "$R/HOLD_$PART"; do [ ! -e "$h" ] || fail "$h exists (the user decides)"; done
pct=$(myquota 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | awk '$1 == "/scratch" {print $0}' | grep -oE '[0-9.]+%' | head -1 | tr -d '%' || true)
[ -n "$pct" ] || fail "cannot read the /scratch quota (myquota)"
awk -v p="$pct" 'BEGIN { exit !(p < 97) }' || fail "/scratch quota use $pct% >= 97%"
echo "/scratch quota use $pct% (< 97%)"
cd "$B"
"$PY" make_jobs_ptdom2.py --plan
"$PY" make_jobs_ptdom2.py --snapshot
mkdir -p "$R/tests"
ST="$R/tests/selftest_readout_submit_$(date -u +%Y%m%dT%H%M%SZ)_$$.out"
PYTHONPATH=$B/pylib "$PY" "$R/readout_ptdom2.py" --selftest > "$ST" 2>&1 || { tail -n 5 "$ST"; fail "the read-out self-test failed ($ST)"; }
echo "read-out self-test: $(tail -n 1 "$ST") ($ST)"
grep -q '^PILOT PASS' "$R/pilot/pilot_check.out" 2>/dev/null || { [ "$GO" = 0 ] && echo "## NOTE: the GPU pilot has not passed yet (runs/ptdom2/pilot/pilot_check.out); --go refuses" || fail "the GPU pilot has not passed (runs/ptdom2/pilot/pilot_check.out)"; }
PK=()
if [ "$PART" = R1x ] || [ "$PART" = R2 ]; then
  [ -e "$PICKS" ] || fail "no picks file $PICKS (select_ptdom2.py first)"
  [ ! -w "$PICKS" ] || fail "$PICKS is writable (a decision file is chmod 444)"
  PYTHONPATH=$B/pylib "$PY" "$R/select_ptdom2.py" --verify "$PICKS" || fail "$PICKS differs from its re-derivation"
  PK=(--picks "$PICKS")
fi
"$PY" make_jobs_ptdom2.py --part "$PART" "${PK[@]}" --check --print > "$R/tests/submit_check_$PART.out" 2>&1 || { tail -n 5 "$R/tests/submit_check_$PART.out"; fail "--check"; }
tail -n 2 "$R/tests/submit_check_$PART.out"
n_sub=$(grep -c ' (SUBMITTED): ' "$R/tests/submit_check_$PART.out" || true)
[ "$n_sub" = 0 ] || fail "$n_sub jobs of part $PART were submitted before (re-runs: make_jobs_ptdom2.py --part $PART ${PK[*]} --rerun)"
n_jobs=$(grep -c '^ptdom2-.*: --data ' "$R/tests/submit_check_$PART.out" || true)
echo "part $PART: $n_jobs jobs, none submitted before"
[ -e "$B/runs/ptdom/NOTIFY_misalign_when_b200_free.txt" ] && echo "## NOTE (B200 sharing): runs/ptdom/NOTIFY_misalign_when_b200_free.txt exists: the misalign session was told manifolds would leave b200_tandon_prem free; tell it before ptdom2 occupies B200 (coordinator)"
if [ "$GO" = 1 ]; then
  S0F=(); [ "$ALLOW_S0" = 1 ] && S0F=(--allow_s0)
  "$PY" make_jobs_ptdom2.py --part "$PART" "${PK[@]}" "${S0F[@]}" --submit
else
  echo "(checks only; --go submits the $n_jobs jobs listed in runs/ptdom2/tests/submit_check_$PART.out)"
fi
cat <<EOF
Order (each after every job of the previous step has finished with check.json; AUDIT.md "Submission"):
  bash runs/ptdom2/submit_ptdom2.sh --part S --go           # 88 runs (A2 / A3: seeds 8-15; only-d / only-d-shuf at 2e-3; none@2e-3 B1 / A, none@4e-3 A)
  bash runs/ptdom2/submit_ptdom2.sh --part S0 --go --allow_s0   # OPTIONAL (A2.1), only on the coordinator's decision: 32 runs, seeds 0-7
  bash runs/ptdom2/submit_ptdom2.sh --part R1 --go          # 24 runs
  bash runs/ptdom2/submit_ptdom2.sh --part R1pre --go       # 8 runs (A1.10: A none@1.6e-2, B2 none@5e-4, seeds 0-3; with R1)
  python runs/ptdom2/select_ptdom2.py --after R1            # needs R1 + R1pre -> picks_R_after1.json (once, chmod 444): R1x asked, or final (R2)
  bash runs/ptdom2/submit_ptdom2.sh --part R1x --go         # only if picks_R_after1.json asks for extensions
  python runs/ptdom2/select_ptdom2.py --after R1x           # -> picks_R_after2.json (final)
  bash runs/ptdom2/submit_ptdom2.sh --part R2 --go          # seeds 8-15 at the final picks
  PYTHONPATH=$B/pylib python runs/ptdom2/readout_ptdom2.py --json runs/ptdom2/readout.json > runs/ptdom2/readout.txt
                                                            # refuses while a needed run is missing (--allow_partial: PROVISIONAL)
Re-runs (a job that exited before its run finished, e.g. at HOLD or a node fault; a finished run is never re-run):
  python make_jobs_ptdom2.py --part X [--picks f] --rerun --print ; then --rerun --submit
EOF
