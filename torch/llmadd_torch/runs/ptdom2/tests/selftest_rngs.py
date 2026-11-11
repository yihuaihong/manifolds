"""ptdom2 test (A4.1): the read-out self-test's known answers do not depend on the planted worlds' rng: readout_ptdom2.py
--selftest --rng k must PASS for every k in 1..10 (the readout beside this tests/ directory). -> selftest_rngs.out"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R2D = Path(__file__).resolve().parent.parent
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"


def run(k):
    o = subprocess.run([PY, "-B", str(R2D / "readout_ptdom2.py"), "--selftest", "--rng", str(k)], capture_output=True, text=True, cwd=str(R2D),
                       env=dict(os.environ, PYTHONPATH="/scratch/yh6210/llmadd_torch/pylib", PYTHONDONTWRITEBYTECODE="1"), timeout=1800)
    last = (o.stdout.strip().splitlines() or ["(no output)"])[-1]
    fails = [l[5:100] for l in o.stdout.splitlines() if l.startswith("FAIL ")]
    return o.returncode == 0 and "SELFTEST PASS" in last, f"rng {k}: {last}" + (f"; failing: {fails[:3]}" if fails else "")


print(f"readout under test: {R2D / 'readout_ptdom2.py'}", flush=True)
with ThreadPoolExecutor(max_workers=int(os.environ.get("MUT_WORKERS", "4"))) as ex:
    res = list(ex.map(run, range(1, 11)))
for _, line in res:
    print(line)
print(f"SELFTEST RNGS {'PASS' if all(r for r, _ in res) else 'FAIL'} ({sum(r for r, _ in res)}/10 rng seeds)")
