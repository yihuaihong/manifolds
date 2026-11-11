"""pt2same test: the read-out's file layer on runs whose numbers are already published (known answers), before any pt2same
result exists: readout_pt2same.load_run(run_paths(...)) on
  - the 32 Burst twin runs (site 'burst') -> the F86 R02 addendum's 8-seed means (53M none 18.43 / prior 69.91, 190M none
    29.06 / prior 78.45; 2 decimals) and seed halves (23.10 / 13.77, 69.83 / 69.98, 26.84 / 31.29, 78.56 / 78.35);
  - the 16 pt2s505 runs (site 'torch', 505M) -> the F86 505M addendum (none 24.4 / prior 77.4, Delta +53.0 over 8 seeds;
    seeds 0-3 +52.0, 4-7 +53.9; 1 decimal).
No pt2same run is read (none exists; asserted).
    python readout_known.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/yh6210/llmadd_torch/runs/pt2same")
import readout_pt2same as RO   # noqa: E402

assert not list(RO.ROOTS["torch"].glob("*/check.json")) if RO.ROOTS["torch"].exists() else True, "pt2same results exist"
fails = 0


def say(ok, msg):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + msg)


def cell(site, size, arm):
    v = [RO.load_run(*RO.run_paths(site, size, arm, s, RO.ROOTS)) for s in RO.SEEDS]
    assert all(st == "ok" for st, _, _ in v), [(st, w) for st, _, w in v if st != "ok"]
    return np.array([a for _, a, _ in v])


R02 = {("53m", "none"): (18.43, 23.10, 13.77), ("53m", "win-ctx"): (69.91, 69.83, 69.98),
       ("190m", "none"): (29.06, 26.84, 31.29), ("190m", "win-ctx"): (78.45, 78.56, 78.35)}
for (z, a), (m8, m1, m2) in R02.items():
    x = cell("burst", z, a)
    got = (x.mean(), x[:4].mean(), x[4:].mean())
    say(all(abs(g - w) < 0.006 for g, w in zip(got, (m8, m1, m2))),
        f"Burst {z} {a}: 8-seed {got[0]:.2f} (R02 {m8}), 0-3 {got[1]:.2f} ({m1}), 4-7 {got[2]:.2f} ({m2}); all 8 check.json PASS")
n, p = cell("torch", "505m", "none"), cell("torch", "505m", "win-ctx")
d = p - n
say(abs(n.mean() - 24.4) < 0.05 and abs(p.mean() - 77.4) < 0.05 and abs(d.mean() - 53.0) < 0.05 and abs(d[:4].mean() - 52.0) < 0.05
    and abs(d[4:].mean() - 53.9) < 0.05 and (d > 0).sum() == 8,
    f"Torch 505M (pt2s505): none {n.mean():.2f} (24.4), prior {p.mean():.2f} (77.4), Delta {d.mean():+.2f} (+53.0), 0-3 "
    f"{d[:4].mean():+.2f} (+52.0), 4-7 {d[4:].mean():+.2f} (+53.9), wins {(d > 0).sum()}/8")
# the read-out looks exactly where each job writes: $OUT/$TAG/<run file> and $OUT/check.json of the job file
import re   # noqa: E402
ok = 0
jobs = sorted(Path("/scratch/yh6210/llmadd_torch/runs/pt2same").glob("job_*.sbatch"))
for j in jobs:
    t = j.read_text()
    out = re.findall(r"^R=\S+; OUT=(\S+); TAG=(\S+); SEED=(\d);", t, re.M)[0]
    rf = re.findall(r'^"\$PY" - "\$OUT/\$TAG/(run_whole_\S+\.json)"', t, re.M)[0]
    size, arm, seed = j.name.split("-")[1], "none" if "-none-" in j.name else "win-ctx", int(out[2])
    run, chk = RO.run_paths("torch", size, arm, seed, RO.ROOTS)
    ok += run == Path(out[0]) / out[1] / rf and chk == Path(out[0]) / "check.json"
say(len(jobs) == 32 and ok == 32, f"read-out paths = each job's $OUT/$TAG/<run file> and $OUT/check.json: {ok}/{len(jobs)}")
print(f"READOUT KNOWN ANSWERS {'PASS' if not fails else 'FAIL'} ({fails} failures)")
sys.exit(1 if fails else 0)
