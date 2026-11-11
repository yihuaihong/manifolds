"""pt2same Amendment A3 item 1: before the primary read-out, every area of the 32 main and 8 A1 runs must be finite.
Prints only counts and finiteness (no area values), plus each run's check.json pass flag."""
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch/runs/pttpp/readout/tools")
import readout_rules as RR  # noqa: E402

OUT = Path("/scratch/yh6210/llmadd_torch/runs/pt2same/out")
dirs = sorted(p for p in OUT.iterdir() if p.is_dir())
bad, n_ok = [], 0
for d in dirs:
    runs = [f for f in glob.glob(str(d / "*" / "run_*.json")) + glob.glob(str(d / "run_*.json")) if not f.endswith(".claim")]
    chk = d / "check.json"
    if len(runs) != 1 or not chk.exists():
        bad.append(f"{d.name}: {len(runs)} run JSON(s), check.json {'present' if chk.exists() else 'MISSING'}")
        continue
    r = json.load(open(runs[0]))
    a = RR.area(r)
    c = json.load(open(chk))
    passed = c.get("pass", c.get("ok"))
    if not math.isfinite(a) or not RR.finished(r) or passed is not True:
        bad.append(f"{d.name}: finite {math.isfinite(a)}, finished {RR.finished(r)}, check pass {passed}")
    else:
        n_ok += 1
print(f"{len(dirs)} run dirs, {n_ok} ok (finite area, finished, check pass)")
for b in bad:
    print("BAD", b)
sys.exit(1 if bad or len(dirs) != 40 else 0)
