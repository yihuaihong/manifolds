#!/bin/bash
# ladder-1b (runs/ptlr_1b/AUDIT.md): run at the end of every 1B training job. Once all 12 stage-A runs are complete, the
# first job to get here runs the pre-registered read-out (the readout/ snapshot of branch ladder-1b), pick_1b.py, and
# make_jobs_ptlr1b.py (--test-only, then --submit) for stage B (+ C if APPROVE_C exists). Runs once: select.done; a lock
# whose job is no longer running (e.g. preempted mid-selection) is taken over.
set -u
R=${PTLR1B_DIR:-/scratch/yh6210/llmadd_torch/runs/ptlr_1b}   # overrides: plumbing tests only
GEN=${PTLR1B_GEN:-/scratch/yh6210/llmadd_torch/make_jobs_ptlr1b.py}
ROOT=${PTLR1B_ROOT:-/scratch/yh6210/llmadd_torch/runs}
PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
[ -e $R/HOLD ] && { echo "maybe_select: HOLD present"; exit 0; }
[ -e $R/select.done ] && { echo "maybe_select: selection already done"; exit 0; }
PTLR1B_DIR=$R $PY $R/pick_1b.py --check-complete || exit 0
me=${SLURM_JOB_ID:-none}
if ! mkdir $R/select.lock 2>/dev/null; then
    other=$(cat $R/select.lock/job 2>/dev/null || true)
    if [ -n "$other" ] && [ "$other" != "$me" ] && squeue -h -j "$other" -t R,CG 2>/dev/null | grep -q .; then
        echo "maybe_select: selection is running in job $other"; exit 0
    fi
    echo "maybe_select: taking over the lock of job ${other:-?}"
fi
echo "$me" > $R/select.lock/job
echo "[$(date -Is)] selection by job $me" >> $R/select.log
cd $R/readout/tools || exit 1
# sbatch from inside a job: drop the inherited SLURM_* / SBATCH_* variables so they cannot leak into the new jobs
for v in $(compgen -e | grep -E "^(SLURM|SBATCH)_"); do unset "$v"; done
$PY ptscale_summary.py --root $ROOT --json $R/selection_A.json > $R/readout_A.txt 2>&1 || { echo "[$(date -Is)] read-out failed (see readout_A.txt)" >> $R/select.log; exit 1; }
PTLR1B_DIR=$R $PY $R/pick_1b.py --pick $R/selection_A.json || exit 0
$PY $GEN --test-only >> $R/select.log 2>&1 || { echo "[$(date -Is)] --test-only failed: nothing submitted" >> $R/select.log; exit 1; }
$PY $GEN --submit >> $R/select.log 2>&1 || echo "[$(date -Is)] some submissions failed (see above; rerun --submit, it skips SUBMITTED_*)" >> $R/select.log
echo "[$(date -Is)] stage B submitted" > $R/select.done
