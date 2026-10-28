"""T8 deployment (no HOLD: a job that starts while HOLD exists exits without training, see AUDIT T8): wait until no group
lock (a directory under runs/pttpp/locks, '.stale.*' renames excepted) has existed for QUIET seconds in a row; exit 0, or 1
after 30 min."""
import sys
import time
from pathlib import Path

L = Path("/scratch/yh6210/llmadd_torch/runs/pttpp/locks")
QUIET = int(sys.argv[1]) if len(sys.argv) > 1 else 60
assert not (L.parent / "HOLD").exists(), "HOLD exists"
t0 = quiet = time.time()
while time.time() - t0 < 1800:
    held = sorted(p.name for p in L.iterdir() if p.is_dir() and ".stale." not in p.name)
    if held:
        print(f"{time.time() - t0:6.0f}s held: {held}", flush=True)
        quiet = time.time()
    elif time.time() - quiet >= QUIET:
        print(f"{time.time() - t0:6.0f}s no group lock for {QUIET}s", flush=True)
        sys.exit(0)
    time.sleep(5)
sys.exit(1)
