#!/bin/bash
# Amendment B2 (runs/ptlr_1b/AUDIT.md): run at the end of each none@1.2e-3 job. Once all 14 runs (stage A + B2) are
# complete, the first job to get here runs the unchanged read-out snapshot (readout/), pick_1b_b2.py, and
# make_jobs_ptlr1b_b2.py --stage B (--test-only, then --submit; account 520 only). Runs once (select_b2.done); a lock
# whose job is no longer running (e.g. preempted mid-selection) is taken over.
set -u
R=${PTLR1B_DIR:-/scratch/yh6210/llmadd_torch/runs/ptlr_1b}   # overrides: plumbing tests only
GEN=${PTLR1B_GEN:-/scratch/yh6210/llmadd_torch/make_jobs_ptlr1b_b2.py}
ROOT=${PTLR1B_ROOT:-/scratch/yh6210/llmadd_torch/runs}
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
[ -e $R/HOLD ] && { echo "maybe_select_b2: HOLD present"; exit 0; }
[ -e $R/select_b2.done ] && { echo "maybe_select_b2: B2 selection already done"; exit 0; }
PTLR1B_DIR=$R $PY $R/pick_1b_b2.py --check-complete || exit 0
me=${SLURM_JOB_ID:-none}
if ! mkdir $R/select_b2.lock 2>/dev/null; then
    other=$(cat $R/select_b2.lock/job 2>/dev/null || true)
    if [ -n "$other" ] && [ "$other" != "$me" ] && squeue -h -j "$other" -t R,CG 2>/dev/null | grep -q .; then
        echo "maybe_select_b2: B2 selection is running in job $other"; exit 0
    fi
    echo "maybe_select_b2: taking over the lock of job ${other:-?}" | tee -a $R/select_b2.log
fi
echo "$me" > $R/select_b2.lock/job
echo "[$(date -Is)] B2 selection by job $me" >> $R/select_b2.log
cd $R/readout/tools || { echo "[$(date -Is)] no readout snapshot" >> $R/select_b2.log; exit 1; }
# the pre-registered read-out must be the frozen snapshot (B2 audit 2026-10-25): stop, without selecting, otherwise
got="$(sha256sum ptscale_summary.py readout_rules.py pt999_summary.py | cut -c1-12 | tr '\n' ' ')"
if [ "$got" != "404f4466e3bb e91ee8a8a846 01e2fec07499 " ]; then
    echo "[$(date -Is)] READ-OUT SNAPSHOT CHANGED ($got): no B2 selection, nothing submitted" | tee -a $R/select_b2.log > $R/select_b2.done; exit 1
fi
echo "[$(date -Is)] read-out snapshot sha256[:12] $got" >> $R/select_b2.log
# sbatch from inside a job: drop the inherited SLURM_* / SBATCH_* variables so they cannot leak into the new jobs
for v in $(compgen -e | grep -E "^(SLURM|SBATCH)_"); do unset "$v"; done
$PY ptscale_summary.py --root $ROOT --json $R/selection_B2.json > $R/readout_B2.txt 2>&1 || { echo "[$(date -Is)] read-out failed (see readout_B2.txt)" >> $R/select_b2.log; exit 1; }
PTLR1B_DIR=$R $PY $R/pick_1b_b2.py --pick $R/selection_B2.json || exit 0
$PY $GEN --stage B --test-only >> $R/select_b2.log 2>&1 || { echo "[$(date -Is)] stage B --test-only failed: nothing submitted" >> $R/select_b2.log; exit 1; }
if $PY $GEN --stage B --submit >> $R/select_b2.log 2>&1; then
    echo "[$(date -Is)] stage B submitted" | tee -a $R/select_b2.log > $R/select_b2.done
else
    echo "[$(date -Is)] SOME STAGE-B SUBMISSIONS FAILED (see above; rerun --stage B --submit, it skips SUBMITTED_*)" | tee -a $R/select_b2.log > $R/select_b2.done
fi
