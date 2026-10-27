"""T5 deployment: with runs/pttpp/HOLD set, wait until no group lock (a directory under runs/pttpp/locks, '.stale.*'
renames excepted) has existed for QUIET seconds in a row (a hook that passed the HOLD check just before it was set has
then finished); exit 0, or 1 after MAXWAIT seconds. Prints the lock names only."""
import sys
import time
from pathlib import Path

L = Path("/scratch/yh6210/llmadd_torch/runs/pttpp/locks")
QUIET, MAXWAIT = int(sys.argv[1]) if len(sys.argv) > 1 else 90, 1800
assert (L.parent / "HOLD").exists(), "set HOLD first"
t0 = time.time()
quiet_since = time.time()
while time.time() - t0 < MAXWAIT:
    held = sorted(p.name for p in L.iterdir() if p.is_dir() and ".stale." not in p.name)
    if held:
        print(f"{time.time() - t0:6.0f}s held: {held}", flush=True)
        quiet_since = time.time()
    elif time.time() - quiet_since >= QUIET:
        print(f"{time.time() - t0:6.0f}s no group lock for {QUIET}s: safe to swap", flush=True)
        sys.exit(0)
    time.sleep(5)
print("timeout", flush=True)
sys.exit(1)
