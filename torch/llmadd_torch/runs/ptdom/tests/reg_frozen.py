"""Regression of the frozen rb6 generators and read-outs (nothing of another batch is written; outputs go to tests/frozen/):
  1. make_rb6m_tasks.py --check in the ptdom worktree (its tasks/ = origin/main): the 53M Burst pilot / main renderings
     unchanged (the generator asserts the pilot's 18 tasks byte for byte and reports the 96 main-design tasks)
  2. the 16 rb6_190m job files re-rendered with make_jobs_rb6scale.render on runs/rb6_190m/code = the files, byte for byte
     (in memory; the SUBMITTED files are not touched)
  3. the rb6_505m main read-out (runs/rb6_505m/readout_505m.sh outcome a) re-run with its frozen readout_code on the same
     18 roots -> tests/frozen/x1a.{txt,json}: = readout_505m_x1a.json, and = readout_505m_x1a.txt without its header /
     footer lines (script sha, timestamp, rc)
"""
import json
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
OUT = B / "runs" / "ptdom" / "tests" / "frozen"
OUT.mkdir(parents=True, exist_ok=True)
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
ENV = {"HF_HOME": str(B / "hf_cache"), "PYTHONPATH": str(B / "pylib"), "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}
ok = []

# 1. the 53M Burst generator, in the ptdom worktree (tools/ = origin/main except pretrain_range.py)
o = subprocess.run([PY, "tools/make_rb6m_tasks.py", "--check"], cwd="/scratch/yh6210/llmadd-burst-ptdom", capture_output=True,
                   text=True, env=ENV)
(OUT / "make_rb6m_tasks_check.txt").write_text(o.stdout + o.stderr)
last = o.stdout.strip().splitlines()[-1] if o.stdout.strip() else o.stderr[-300:]
good = o.returncode == 0 and "pilot generator unchanged" in o.stdout and " 0 differ (kept)" in last
ok.append(good)
print(f"1. make_rb6m_tasks.py --check: rc {o.returncode}: {last}  -> {'PASS' if good else 'FAIL'}")

# 2. rb6_190m job files
sys.path.insert(0, str(B))
R190 = B / "runs" / "rb6_190m"
sys.path.insert(0, str(R190 / "code" / "tools"))
import make_jobs_rb6scale as M   # noqa: E402
import make_rb6m_tasks as G      # noqa: E402
import make_rb6_tasks as P       # noqa: E402
assert Path(G.__file__).resolve().parent == (R190 / "code" / "tools").resolve()
n_same, bad = 0, []
for arm, lr in M.BATCHES["rb6_190m"]["stages"]["main"]:
    for s in M.BATCHES["rb6_190m"]["seeds"]:
        name, tag, text, args, I = M.render(G, P, R190, "190m", arm, lr, s, M.WANT_SHA["pretrain_range.py"])
        f = R190 / f"job_{name}.sbatch"
        if f.read_text() == text:
            n_same += 1
        else:
            bad.append(name)
ok.append(n_same == 16 and not bad)
print(f"2. rb6_190m job files re-rendered: {n_same}/16 byte-identical; differ {bad}  -> {'PASS' if ok[-1] else 'FAIL'}")

# 3. the rb6_505m main read-out (outcome a), frozen code, same roots
R5 = B / "runs" / "rb6_505m"
BR = Path("/scratch/yh6210/llmadd-burst-ro/results")
roots = [str(R5 / "out"), str(B / "runs" / "rb6_190m" / "out")]
for pat in ("401-rb6pb-*", "404-rb6pb-*", "430-rb6pb-*", "431-rb6pb-*", "419-rb6pb-*", "415-rb6pb-*", "432-rb6pb-*",
            "433-rb6pb-*", "438-rb6mb-*", "444-rb6mb-*", "450-rb6mb-*", "456-rb6mb-*", "46[6-9]-rb6mb-*"):
    roots += sorted(str(p) for p in BR.glob(pat))
assert len(roots) == 18, len(roots)
o = subprocess.run([PY, "rb6scale_summary.py", *roots, "--picks", str(R5 / "picks_x1.json"), "--x1", "a", "--x1_json",
                    str(R5 / "picks_x1.json"), "--json", str(OUT / "x1a.json")], cwd=R5 / "readout_code" / "tools",
                   capture_output=True, text=True, env=ENV)
(OUT / "x1a.txt").write_text(o.stdout + o.stderr)
frozen = (R5 / "readout_505m_x1a.txt").read_text().splitlines()
body = [l for l in frozen if not l.startswith("== readout_505m.sh") and not l.startswith("== readout_code/COMMIT") and l != "rc 0"]
mine = (o.stdout + o.stderr).splitlines()
jf, jm = json.loads((R5 / "readout_505m_x1a.json").read_text()), json.loads((OUT / "x1a.json").read_text())
good = o.returncode == 0 and mine == body and jf == jm
ok.append(good)
print(f"3. rb6_505m read-out x1a re-run: rc {o.returncode}; text {len(mine)} lines {'=' if mine == body else '!='} the frozen "
      f"{len(body)}; JSON {'=' if jf == jm else '!='} the frozen  -> {'PASS' if good else 'FAIL'}")
print("FROZEN", "PASS" if all(ok) else "FAIL")
