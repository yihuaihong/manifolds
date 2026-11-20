"""auto_pt6v2.sh step 5 check (coordinator 2026-11-19, audit_diff_V2.md I3): make_jobs --submit continues past a failed sbatch and
exits 0, so count the SUBMITTED_* flags against the written cell set (CELLS.json cells x 8 seeds + the canary). Exit 1 on a gap."""
import json
import sys
from pathlib import Path

R = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2")
m = json.loads((R / "CELLS.json").read_text())
want = int(m["n_runs"]) + 1
have = len([f for f in R.glob("SUBMITTED_*") if f.is_file()])
print(f"SUBMITTED flags {have} / expected {want} (CELLS.json n_runs {m['n_runs']} + canary)")
sys.exit(0 if have >= want else 1)
