#!/bin/bash
#SBATCH --job-name=llmadd_rb6_190m_dim-select
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out
# rb6_190m_dim: the pre-registered choice of none's dim lr on seeds 0-3 (runs/rb6_190m_dim/AUDIT.md "Selection"), then Stage D2.
# Submitted by make_jobs_rb6_190m_dim.py --stage D1 --submit with --dependency=afterany:<the D1 job ids>. Runs the frozen
# read-out tool (select_code = git archive of tools/ at 8251667); only if its status is "picked": the generator
# (sha256[:12] e113d287fa37, checked) --stage D2 --picks picks.json --test-only, then --submit. Log: runs/rb6_190m_dim/select.log.
set -u
R=/scratch/yh6210/llmadd_torch/runs/rb6_190m_dim; B=/scratch/yh6210/llmadd_torch; PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python
LOG=$R/select.log
if [ -e "$R/HOLD" ]; then echo "HOLD present: exiting" | tee -a "$LOG"; exit 0; fi
echo "== selection $(date -u +%FT%TZ) job ${SLURM_JOB_ID:-} on $(hostname)" | tee -a "$LOG"
cd "$R/select_code/tools" || exit 3
grep -q "rb6scale_dim_summary.py $(sha256sum rb6scale_dim_summary.py | cut -c1-12)" "$R/select_code/COMMIT" || { echo "## rb6scale_dim_summary.py differs from select_code/COMMIT: exiting" | tee -a "$LOG"; exit 3; }
grep -q "rb6scale_summary.py $(sha256sum rb6scale_summary.py | cut -c1-12)" "$R/select_code/COMMIT" || { echo "## rb6scale_summary.py differs from select_code/COMMIT: exiting" | tee -a "$LOG"; exit 3; }
grep -q "rainbow6_summary.py $(sha256sum rainbow6_summary.py | cut -c1-12)" "$R/select_code/COMMIT" || { echo "## rainbow6_summary.py differs from select_code/COMMIT: exiting" | tee -a "$LOG"; exit 3; }
grep -q "rb6scale_select.py $(sha256sum rb6scale_select.py | cut -c1-12)" "$R/select_code/COMMIT" || { echo "## rb6scale_select.py differs from select_code/COMMIT: exiting" | tee -a "$LOG"; exit 3; }
grep -q "readout_rules.py $(sha256sum readout_rules.py | cut -c1-12)" "$R/select_code/COMMIT" || { echo "## readout_rules.py differs from select_code/COMMIT: exiting" | tee -a "$LOG"; exit 3; }
grep -q "make_rb6m_tasks.py $(sha256sum make_rb6m_tasks.py | cut -c1-12)" "$R/select_code/COMMIT" || { echo "## make_rb6m_tasks.py differs from select_code/COMMIT: exiting" | tee -a "$LOG"; exit 3; }
grep -q "make_rb6_tasks.py $(sha256sum make_rb6_tasks.py | cut -c1-12)" "$R/select_code/COMMIT" || { echo "## make_rb6_tasks.py differs from select_code/COMMIT: exiting" | tee -a "$LOG"; exit 3; }
[ "$(sha256sum $B/make_jobs_rb6_190m_dim.py | cut -c1-12)" = "e113d287fa37" ] || { echo "## the generator changed since D1: exiting" | tee -a "$LOG"; exit 3; }
[ "$(sha256sum $B/scale_work/rb190dim_refs.json | cut -c1-12)" = "55f901755cde" ] || { echo "## the CPU references changed since D1: exiting" | tee -a "$LOG"; exit 3; }
[ ! -e "$R/picks.json" ] || { echo "## picks.json exists: the selection runs once; exiting" | tee -a "$LOG"; exit 3; }
export HF_HOME=$B/hf_cache PYTHONPATH=$B/pylib
"$PY" rb6scale_dim_summary.py select "$B/runs/rb6_190m/out" "$B/runs/rb6_190m_ext/out" "$R/out" --json "$R/picks.json" 2>&1 | tee -a "$LOG"
status=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' "$R/picks.json" 2>/dev/null || echo none)
echo "selection status: $status" | tee -a "$LOG"
if [ "$status" != "picked" ]; then
  echo "NOT PICKED: Stage D2 is not submitted (the problems above go to the user)" | tee -a "$LOG"; exit 0
fi
# a clean environment for the child sbatch calls (no SLURM_* / SBATCH_* of this job)
for v in $(env | grep -oE '^(SLURM|SBATCH)_[A-Za-z0-9_]*'); do unset "$v"; done
cd "$B"
"$PY" /scratch/yh6210/llmadd_torch/make_jobs_rb6_190m_dim.py --stage D2 --picks "$R/picks.json" --test-only 2>&1 | tee -a "$LOG"
"$PY" /scratch/yh6210/llmadd_torch/make_jobs_rb6_190m_dim.py --stage D2 --picks "$R/picks.json" --submit 2>&1 | tee -a "$LOG"
echo "== done $(date -u +%FT%TZ)" | tee -a "$LOG"
