"""A3 cost basis: B200 timing of the pt300x 294M runs (run JSON curve 'sec' and the job logs' start / return stamps), and the
53M L40S basis, scaled to the ptdom 53M run (read-only; prints a table)."""
import datetime as dt
import glob
import json
import re
import statistics as st
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
rows = []
for f in sorted(glob.glob(str(B / "runs/pt300x/out/p3x-294m-*/run_whole_none_seed*_p3x-294m-*.json"))):
    r = json.loads(Path(f).read_text())
    c = r["curve"]
    if len(c) != 75:
        continue
    secs = [e["sec"] for e in c]
    if any(b < a for a, b in zip(secs, secs[1:])):   # a resume resets the clock: skip
        continue
    # per interval: 250 steps + 1 eval; fit sec = a * steps + b * evals by the intervals (all equal) + the total
    dif = [b - a for a, b in zip(secs, secs[1:])]
    tag = Path(f).parent.name
    seed = re.search(r"seed(\d+)", f).group(1)
    log = glob.glob(str(B / f"logs/llmadd_{tag}-s{seed}_*.out"))
    wall = None
    if log:
        t = Path(log[0]).read_text(errors="replace")
        m0 = re.search(r"start (\S+) job", t); m1 = re.search(r"trainer returned (\S+)", t)
        if m0 and m1 and t.count(" start ") == 1:
            wall = (dt.datetime.fromisoformat(m1.group(1)) - dt.datetime.fromisoformat(m0.group(1))).total_seconds()
    rows.append((tag, seed, secs[-1], st.median(dif), wall, "win" in tag))
print(f"pt300x 294M B200 runs with one clock (no resume): {len(rows)}")
for t, s, tot, md, w, pri in rows:
    print(f"  {t} s{s}: trainer sec {tot:.0f}, median sec per 250 steps + eval {md:.1f}, job wall start->trainer return {w}")
tot = [r[2] for r in rows]
walls = [r[4] for r in rows if r[4]]
print(f"294M B200: trainer sec median {st.median(tot):.0f} [min {min(tot):.0f}, max {max(tot):.0f}], wall median {st.median(walls):.0f}")
