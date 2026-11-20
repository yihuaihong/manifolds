"""pt6v2 test helper: measured 53M training wall clock (run JSON curve[-1]['sec']) of finished pt6comb / ptdom3 runs, per GPU
type (out/<run>/node.txt) and arm kind; read only. -> stdout (tests/walltime_scan.out)."""
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
rows = {}
for batch in ("pt6comb", "ptdom3", "ptdom2", "ptdom"):
    for d in sorted((B / "runs" / batch / "out").glob("*")):
        js = list(d.glob("*/run_whole_*_seed*.json"))
        nt = d / "node.txt"
        if len(js) != 1 or not nt.exists():
            continue
        try:
            c = json.loads(js[0].read_text())["curve"]
        except Exception:
            continue
        if not c or c[-1].get("step", 0) < 18000:
            continue
        g = re.findall(r"NVIDIA ([A-Z0-9][^,\n]*),", nt.read_text())
        gpu = g[-1].strip() if g else "?"
        kind = "none" if "-none-" in d.name else ("all" if re.search(r"-all(6|toy3)-", d.name) else "only")
        rows.setdefault((batch, gpu, kind), []).append(c[-1]["sec"])
for k, v in sorted(rows.items()):
    print(f"{k[0]:8s} {k[1]:22s} {k[2]:5s} n={len(v):3d} median {st.median(v):7.0f} s  max {max(v):7.0f} s  ({st.median(v) / 3600:.2f} h)")
