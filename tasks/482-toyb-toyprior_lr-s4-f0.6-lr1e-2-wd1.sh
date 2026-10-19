# kind: gpu
# time: 01:00:00
# cpus: 8
# after: 331-env311
# Torch runs/toyprior_lr cell s4-f0.6-lr1e-2-wd1 (S4 prior arms at lr 1e-2, 60% question split), moved to Burst 2026-10-19 (user approved: Torch's per-user GPU cap was held by another project; the
# Torch job is held with the run dir's HOLD file). toy_struct.py pinned at 398935a through burst/pin_run.sh (sha256
# b74a78cb5472... = the Torch snapshot in code/COMMIT), the Torch job's arguments unchanged; 5 seeds as parallel processes
# (L40S on Torch: 10-20 min per cell). Intent spec: runs/toynone/AUDIT.md on Torch (amendment 2026-10-18).
set -euo pipefail
PIN=398935a
pids=""
for s in 0 1 2 3 4; do
  bash burst/pin_run.sh $PIN toy_struct.py --task s4 --frac 0.6 --split question --steps 100000 --seeds $s --arms R@always,R_sign@always,specrand@always --win 0,2000 --lam 1 --lr 1e-2 --wd 1 \
    --eval_every 100 --threads 1 --out "$OUT" > "$OUT/log_seed$s.txt" 2>&1 &
  pids="$pids $!"
done
fail=0
for p in $pids; do wait $p || fail=1; done
grep -q " b74a78cb5472[0-9a-f]* tools/toy_struct.py " "$OUT/PINNED.txt" || { echo "## the pinned toy_struct.py is not b74a78cb5472"; exit 3; }
bash burst/pin_run.sh $PIN toy_struct.py --summarize "$OUT"
[ $fail -eq 0 ] || { echo "a seed process failed"; exit 1; }
