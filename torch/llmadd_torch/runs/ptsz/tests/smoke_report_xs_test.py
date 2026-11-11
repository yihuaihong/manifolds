"""Known-answer test of tests/smoke_report_xs.py (amendments 4 / 4.1) on logs made from the first smoke's real log (22011938):
segments A / B / C renamed XA / XB / XC (the XS job's commands are A / B / C with the out dir changed, XC stopping at 260);
the XC run JSON = 22011938's C run JSON (evals 0 / 250 / 500).
  same      : the first smoke's numbers unchanged + the ALLOC line      -> FAIL (3) and (4) only
  good      : reserved = allocated + 0.5 GiB in every segment + ALLOC   -> PASS
  noalloc   : good without the ALLOC line                               -> FAIL (1) only
  noresume  : good with XB's 'resumed from' line removed                -> FAIL (2) only
  runguard  : good + a 'run guard:' line                                -> FAIL (2) only
  tiny      : good with XB's max_allocated 1 GiB (reserved 1.5 GiB)     -> FAIL (2) only
  gapweak   : reserved = allocated + 0.6 x 22011938's gap               -> FAIL (4) only (the margin (3) still holds)
  lm0shift  : good with XA's step-0 lm + 0.5 in the log                 -> FAIL (5) only (the audit's case for R1)
  resumeoff : good with XB's step-200 lm + 1e-3                         -> FAIL (5) only
  noxc      : good without the XC run JSON                              -> FAIL (5) only
Writes only into temp dirs (XS_SMOKE_DIR / XS_LOGS_DIR); prints PASS / FAIL per case.
    python tests/smoke_report_xs_test.py"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

P = Path("/scratch/yh6210/llmadd_torch/runs/ptsz")
LOG0 = Path("/scratch/yh6210/llmadd_torch/logs/llmadd_psz-smoke731_22011938.out")
PY = sys.executable
GIB = 2 ** 30
C1 = "(1) setting on"
C2 = "(2) XA XB XC ran, no run guard, resumed at step 200"
C3 = "(3) 731m reserved <= total - 2 GiB"
C4 = "(4) takes effect (gap <= 0.5 x without)"
C5 = "(5) numerics unchanged"


def make_log(case):
    out, seg = [], None
    for l in LOG0.read_text().splitlines():
        m = re.match(r"^=== segment ([A-E]) (start|rc)", l)
        if m:
            seg = m.group(1) if m.group(2) == "start" else None
        if l.startswith("SMOKE "):
            j = json.loads(l[6:])
            if j["seg"] not in ("A", "B", "C"):
                continue
            old_gap = j["max_reserved"] - j["max_alloc"]
            j["seg"] = "X" + j["seg"]
            if case == "tiny" and j["seg"] == "XB":
                j["max_alloc"] = GIB
            if case == "gapweak":
                j["max_reserved"] = j["max_alloc"] + int(0.6 * old_gap)
            elif case != "same":
                j["max_reserved"] = j["max_alloc"] + GIB // 2
            l = "SMOKE " + json.dumps(j)
        elif l.startswith("{") and seg in ("A", "B"):
            try:
                j = json.loads(l)
            except json.JSONDecodeError:
                j = None
            if isinstance(j, dict) and "step" in j and "lm" in j:
                if case == "lm0shift" and seg == "A" and j["step"] == 0:
                    j["lm"] += 0.5
                if case == "resumeoff" and seg == "B" and j["step"] == 200:
                    j["lm"] += 1e-3
                l = json.dumps(j)
        l = re.sub(r"^=== segment ([ABC]) ", r"=== segment X\1 ", l)
        if re.match(r"^=== segment [DE] ", l):
            continue
        if case == "noresume" and l.startswith("resumed from"):
            continue
        out.append(l)
        if l.startswith("=== smoke731 start") and case != "noalloc":
            out.insert(len(out) - 1, "ALLOC: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        if l.startswith("=== segment XB start") and case == "runguard":
            out.append("run guard: another job holds the claim; exiting")
    return "\n".join(out) + "\n"


def run(case):
    d = Path(tempfile.mkdtemp(prefix="xs_test_"))
    (d / "logs").mkdir()
    (d / "smoke").mkdir()
    shutil.copy(P / "smoke" / "smoke_report_22011938.json", d / "smoke")
    if case != "noxc":
        (d / "smoke" / "XC").mkdir()
        for f in (P / "smoke" / "C").glob("run_*.json"):
            shutil.copy(f, d / "smoke" / "XC")
    (d / "logs" / f"llmadd_psz-smoke731xs_t{case}.out").write_text(make_log(case))
    env = dict(os.environ, XS_SMOKE_DIR=str(d / "smoke"), XS_LOGS_DIR=str(d / "logs"))
    o = subprocess.run([PY, str(P / "tests" / "smoke_report_xs.py"), f"t{case}"], capture_output=True, text=True, env=env)
    v = json.loads((d / "smoke" / "SMOKE_VERDICT.json").read_text()) if (d / "smoke" / "SMOKE_VERDICT.json").exists() else None
    shutil.rmtree(d)
    return o, v


def main():
    want = {"same": (False, {C3, C4}), "good": (True, set()), "noalloc": (False, {C1}), "noresume": (False, {C2}),
            "runguard": (False, {C2}), "tiny": (False, {C2}), "gapweak": (False, {C4}), "lm0shift": (False, {C5}),
            "resumeoff": (False, {C5}), "noxc": (False, {C5})}
    bad = 0
    for case, (wpass, wfail) in want.items():
        o, v = run(case)
        if v is None:
            print(f"FAIL {case}: no verdict; rc {o.returncode}; {o.stderr[-400:]}")
            bad += 1
            continue
        failed = {k for k, x in v["criteria"].items() if not x}
        ok = v["pass"] is wpass and failed == wfail
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {case}: pass {v['pass']} failed {sorted(failed)}"
              + ("" if ok else f" (want pass {wpass} failed {sorted(wfail)}); numerics {v.get('numerics')}"))
    real = P / "smoke" / "SMOKE_VERDICT.json"
    print(f"real verdict untouched: job {json.loads(real.read_text())['job']}")
    print("XS REPORT TEST " + ("PASS" if not bad else f"FAIL ({bad})"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
