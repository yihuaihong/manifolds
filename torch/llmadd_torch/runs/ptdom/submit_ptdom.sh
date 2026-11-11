#!/bin/bash
# ptdom A3 / A4 / A5 submission helper (runs/ptdom/AUDIT.md "Amendment A3", "Amendment A4", "Amendment A5"; A6.1: the self-test
# log path only). Without --go it only PRINTS: the plan, the batch's stage-1
# jobs and the commands of the later stages. With --go it submits the batch's stage 1 (seed-major order) through
# make_jobs_ptdom.py --batch X --stage 1 --submit. A2: a job NAME is submitted once: the generator refuses every name with a
# SUBMITTED_ flag and a re-run is a new name <name>-r<k> (--rerun). Stages 2 / 2b and re-runs are submitted by hand after
# the selection step (the commands below), never by this script.
#   bash runs/ptdom/submit_ptdom.sh --batch X --stage 1          # print
#   bash runs/ptdom/submit_ptdom.sh --batch X --stage 1 --go     # submit batch X's stage 1 (60 runs)
# Kill switch: touch runs/ptdom/HOLD (every batch) or runs/ptdom/HOLD_<X> (A4: batch X only); pending / requeued jobs exit
# at start; the generator refuses to submit while it exists. A4: every batch on B200 (4 singleton slot chains per batch).
set -euo pipefail
B=/scratch/yh6210/llmadd_torch
R=$B/runs/ptdom
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
BATCH=""; STAGE=""; GO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --batch) BATCH=$2; shift 2 ;; --stage) STAGE=$2; shift 2 ;; --go) GO=1; shift ;;
    *) echo "## unknown argument $1"; exit 64 ;;
  esac
done
case "$BATCH" in A|B1|B2|B3) ;; *) echo "## --batch A | B1 | B2 | B3"; exit 64 ;; esac
[ "$STAGE" = 1 ] || { echo "## --stage 1 (stages 2 / 2b: by hand after select_ptdom.py, the commands printed below)"; exit 64; }
if [ "$BATCH" = A ]; then REF=$R/ptdom_refs.json; else REF=$R/ptdom_refs_$BATCH.json; fi
[ -s "$REF" ] || { echo "## no CPU references for batch $BATCH ($REF: runs/ptdom/tests/a3/run_refs_a3.sh $BATCH first)"; exit 3; }
cd "$B"
"$PY" make_jobs_ptdom.py --plan
"$PY" make_jobs_ptdom.py --snapshot
# A6.1 (audit_a6 minor 4): the self-test log goes to a new dated file in tests/a6/; the amendments' test records
# (tests/a5/selftest_readout_a5.out etc.) are never overwritten
mkdir -p "$R/tests/a6"
ST="$R/tests/a6/selftest_readout_submit_$(date -u +%Y%m%dT%H%M%SZ)_$$.out"
"$PY" "$R/readout_ptdom.py" --selftest > "$ST" 2>&1 || { tail -n 5 "$ST"; echo "## read-out self-test failed"; exit 3; }
echo "read-out self-test: $(tail -n 1 "$ST") ($ST)"
ls "$R"/pilot/check_ok_"$BATCH" >/dev/null 2>&1 || echo "## NOTE: no runs/ptdom/pilot/check_ok_$BATCH: the batch's GPU pilot has not been confirmed (AUDIT A4.7)"
if [ "$GO" = 1 ]; then
  [ ! -e "$R/HOLD" ] && [ ! -e "$R/HOLD_$BATCH" ] || { echo "## runs/ptdom/HOLD or HOLD_$BATCH exists"; exit 3; }
  [ -e "$R/pilot/check_ok_$BATCH" ] || { echo "## the batch's GPU pilot is not confirmed (runs/ptdom/pilot/check_ok_$BATCH)"; exit 3; }
  "$PY" make_jobs_ptdom.py --batch "$BATCH" --stage 1 --submit
else
  "$PY" make_jobs_ptdom.py --batch "$BATCH" --stage 1 --print | tail -n 3
  echo "(print only; --go submits the 60 stage-1 jobs of batch $BATCH listed by: python make_jobs_ptdom.py --batch $BATCH --stage 1 --print)"
fi
cat <<EOF
While stage 1 runs: as soon as the first prior runs pass step 1500 (window evals 1000 / 1250 / 1500), and again later:
  python runs/ptdom/aux_gate_check.py --batch $BATCH   # per (arm, lr) cell; exit 1: a cell's median LM share a / (a + b) < 0.5 or
                                                    # a missing gradient field -> touch runs/ptdom/HOLD_$BATCH, report to the user
After a HOLD (A2 re-run path): pending jobs that start while runs/ptdom/HOLD or HOLD_$BATCH exists exit 0 before training and
keep their SUBMITTED_ flag. Only after the USER removes it:
  python make_jobs_ptdom.py --batch $BATCH --stage 1 --rerun --print    # the cells whose latest submission is not queued and has no
                                                                     # finished run, under <name>-r1 (-r2 ...)
  python make_jobs_ptdom.py --batch $BATCH --stage 1 --rerun --submit
Later stages (by hand, in this order; each after every job of the previous stage has finished with check.json):
  python runs/ptdom/select_ptdom.py --batch $BATCH --after 1      # the stage-1 screen table -> runs/ptdom/picks_${BATCH}_after1.json
                                                               #   (written once); an arm failing the gate in every cell -> GATE,
                                                               #   HOLD_$BATCH written, exit 2
  python make_jobs_ptdom.py --batch $BATCH --stage 2 --picks runs/ptdom/picks_${BATCH}_after1.json --submit
                                                               # the advancing domains: seeds 4-7 at the picks + the extensions (seeds 0-7)
  python runs/ptdom/select_ptdom.py --batch $BATCH --after 2      # re-selection -> picks_${BATCH}_after2.json (none 5e-4 / moved picks / A5 fixB: F = none at a prior pick outside none's range, listed by name)
  python make_jobs_ptdom.py --batch $BATCH --stage 2b --picks runs/ptdom/picks_${BATCH}_after2.json --submit   # if it asks for any
  python runs/ptdom/select_ptdom.py --batch $BATCH --after 3      # (after 2b) -> picks_${BATCH}_after3.json; --stage 2c if it asks
  python runs/ptdom/select_ptdom.py --batch $BATCH --after 4      # (after 2c, if one ran) -> complete
  PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib python runs/ptdom/readout_ptdom.py --json runs/ptdom/readout.json > runs/ptdom/readout.txt
                                                               # PROVISIONAL (no PASS) until EVERY batch's decision chain is complete;
                                                               # then the verdicts, Holm over all advancing domains of all four batches
EOF
