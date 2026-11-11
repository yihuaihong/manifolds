"""ptdom2 test: pilot_check.py on mock pilot dirs built from REAL ptdom run JSONs (temp dir via PILOT_DIR; read only otherwise):
  A  'a true-target prior disguised as the pilot': ptdom's only-compass@2e-3 s0 truncated to its first 11 evals (0..2500), with
     the three truncation failures in check.json -> P1, P2a, P3, P4 pass, P2b / P2c FAIL (its true-target CKA rises like the
     true prior's) -> PILOT FAIL: the check detects a prior that is not shuffled
  B  'a shuffled-looking run': ptdom's none@2e-3 s0 truncated (the same data, no prior: the true-target CKA = none's), with a
     falling geo_compass over the window (a fit to some other target) -> every rule passes -> PILOT PASS
  C  as B with a fourth failed check -> P1 FAIL
-> pilot_check_test.out"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

R0 = Path("/scratch/yh6210/llmadd_torch/runs/ptdom/out")
PC = Path("/scratch/yh6210/llmadd_torch/runs/ptdom2/pilot/pilot_check.py")
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
NAME = "ptdom2-pilot-B2-only-compass-shuf-lr2e-3-s0"
TAG = "ptdom2-B2-only-compass-shuf-lr2e-3"
TRUNC = [{"check": "finished: 75 evals, increasing, the last at step 18309", "ok": False},
         {"check": "prior_events == [[916, 1], [7324, 0]]", "ok": False}, {"check": "prior_on_frac == 0.350082", "ok": False}]
res = []


def src(arm):
    t = f"ptdom-B2-{arm}-lr2e-3"
    return json.loads((R0 / f"{t}-s0" / t / f"run_whole_none_seed0_{t}.json").read_text())


def mock(tmp, run, extra_fail=()):
    d = Path(tmp) / NAME
    (d / TAG).mkdir(parents=True)
    run["curve"] = run["curve"][:11]
    (d / TAG / f"run_whole_none_seed0_{TAG}.json").write_text(json.dumps(run))
    (d / "check.json").write_text(json.dumps({"checks": [{"check": "run args = intent", "ok": True}] + TRUNC + list(extra_fail)}))
    shutil.copy(R0 / "ptdom-B2-only-compass-lr2e-3-s0" / "gpu.csv", d / "gpu.csv")


def run_pc(tmp):
    o = subprocess.run([PY, str(PC), NAME], capture_output=True, text=True, env=dict(os.environ, PILOT_DIR=str(tmp)))
    return o.stdout


with tempfile.TemporaryDirectory() as tmp:
    mock(tmp, src("only-compass"))
    out = run_pc(tmp)
    print("--- A:\n" + out)
    fl = [l.split()[1] for l in out.splitlines() if l.startswith("FAIL ")]
    res.append("PILOT FAIL" in out and "P2b" in fl and "P2c" in fl and "P1" not in fl and "P3" not in fl)
    print(("PASS" if res[-1] else "FAIL") + " A: a true-target prior is caught by P2b / P2c")
for extra in ((), ({"check": "data_sha after step 0 = the CPU reference", "ok": False},)):
    with tempfile.TemporaryDirectory() as tmp:
        r = src("none")
        for e in r["curve"]:
            on = 916 <= e["step"] <= 7323
            e["prior_on"] = int(on)
            e["geo_compass"] = 0.9 - (0.4 * min(1.0, (e["step"] - 916) / 1584) if on else 0.0)
            if on:
                e["grad_prior"] = 0.3 * e["grad_lm"]
        mock(tmp, r, extra)
        out = run_pc(tmp)
        print(f"--- {'C' if extra else 'B'}:\n" + out)
        fl = [l.split()[1] for l in out.splitlines() if l.startswith("FAIL ")]
        res.append(("PILOT FAIL" in out and fl == ["P1"]) if extra else ("PILOT PASS" in out))
        print(("PASS" if res[-1] else "FAIL") + (" C: a fourth failed check is caught by P1" if extra else " B: a shuffled-looking run passes every rule"))
print(f"PILOT CHECK TEST {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
