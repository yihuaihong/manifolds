"""pt6v2 GPU smoke check (phase 2b; pass rules in runs/pt6v2/AUDIT_phase2b.md "GPU smokes", written before submission). Read only.
    python smoke_check_pt6v2.py <dev> <slurm log>          (dev = l40s | a100; out dir runs/pt6v2/pilot/smoke_<dev>)
S1 the log has 'DEVICE <dev>'          S2 the trainer's step-0 eval line ('"step": 0') and the step-0 toy3 gradient line
S3 the trainer's final line 'done in'  S4 CHECK_PY: 'CHECKS PASS' (or the A5.1 flag-only FAIL: GPU-numerics / CKA-rise checks only)
Then the step rate: median sec/step between evals x 18310 = the naive full-run wall clock, an UPPER bound (evals every 50 here vs
250 in the full run; no same-cadence B200 pair exists to calibrate the eval share: the pt6comb B200 'smoke' s0 was a full 18310-step
run). GPU memory max / util from check.json. Prints 'SMOKE <dev> PASS|FAIL' and 'RATE <dev> ...' (coordinator decision 2: time limit
= 2.5 x the full-run estimate; 2.5 x this upper bound is the safe side)."""
import json
import re
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
dev, log = sys.argv[1], Path(sys.argv[2])
R = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2")
FULL_STEPS = 18310
OUT = R / "pilot" / f"smoke_{dev}"
txt = log.read_text(errors="replace") if log.exists() else ""
res = []


def ok(name, cond, info=""):
    res.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'} {name}  {info}")


def sps(curve):
    """median sec per step between consecutive evals after step 0 (the step-0 eval and set-up excluded)"""
    c = [e for e in curve if e["step"] > 0]
    d = [(b["sec"] - a["sec"]) / (b["step"] - a["step"]) for a, b in zip(c, c[1:]) if b["step"] > a["step"]]
    return statistics.median(d) if d else None


ok(f"S1 'DEVICE {dev}' in the log", re.search(rf"^DEVICE {dev} ", txt, re.M), re.findall(r"^DEVICE .*$", txt, re.M)[:1])
ok("S2 the step-0 eval line and the step-0 toy3 gradient line", re.search(r'"step": 0[,}]', txt)
   and "step 0: toy3 true-prior gradient norms" in txt)
ok("S3 the trainer's final line 'done in'", re.search(r"^done in \d", txt, re.M), re.findall(r"^done in .*$", txt, re.M)[:1])
ck = re.findall(r"^CHECKS (PASS|FAIL.*)$", txt, re.M)
ok("S4 CHECK_PY PASS (or the A5.1 flag-only FAIL)", ck and (ck[-1] == "PASS" or "the run is VALID data" in ck[-1]),
   {"CHECKS": ck[-1:], "failed": re.findall(r"^FAIL (.*)$", txt, re.M)[:4]})
js = sorted(OUT.glob("*/run_whole_*_seed16_*.json"))
cj = json.loads((OUT / "check.json").read_text()) if (OUT / "check.json").exists() else {}
info = cj.get("info", {})
if js:
    r = json.loads(js[0].read_text())
    s = sps(r["curve"])
    naive = s * FULL_STEPS if s else None
    print(f"RATE {dev}: median {s:.4f} s/step over {len(r['curve'])} evals of {r['steps']} steps, {1 / s:.2f} steps/s; full run "
          f"{FULL_STEPS} steps: naive {naive / 3600:.2f} h (upper bound), time limit 2.5 x naive = {2.5 * naive / 3600:.2f} h; "
          f"GPU mem max "
          f"{info.get('gpu_mem_max_MiB')} MiB, util median {info.get('gpu_util_median')}, names {info.get('gpu_names')}")
else:
    ok("the smoke's run JSON exists", False, str(OUT))
print(f"SMOKE {dev} {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
sys.exit(0 if all(res) else 1)
