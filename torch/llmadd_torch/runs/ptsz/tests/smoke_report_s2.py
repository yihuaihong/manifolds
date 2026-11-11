"""Read the seed-2 diagnostic smoke (runs/ptsz/smoke, job llmadd_psz-smoke731s2; AUDIT.md "Amendment 6", criteria registered
before the job ran). Segments = job_smoke731xs.sbatch's XA / XB / XC at --seed 2 (expandable_segments on):
  S2A = 731m win-ctx 6e-4 s2, --window 0,0.4 --eval_every 100 --ckpt_every 100 --stop_after 210 (in-window evals, ckpt writes);
  S2B = the same to 310 (resumes S2A's step-199 ckpt); S2C = the real none 2.4e-3 s2 ARGS to 260 (the path that OOMed in
  22346535: the first training backward after the step-0 eval).
PASS (the setting removes the seed-2 OOM, numerics unchanged) iff all of:
  (1) the log shows 'ALLOC: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True';
  (2) S2A, S2B, S2C ran without error (smoke_seg ok), no OutOfMemoryError anywhere, no 'run guard' line;
  (3) S2B resumed at step 200 ('resumed from ... at step 200' in S2B's part) and its step-200 lm = S2A's within 2e-4;
  (4) the step-0 lm of S2A (log) and S2C (run JSON) within 2e-4 of the CPU reference lm0_ref('731m', 2) (the pttpp
      calibration tolerance; CHECK7 allows 1e-3);
  (5) S2C reached step 260 (its run JSON has the evals at 0 and 250).
Printed, not criteria: max_allocated / max_reserved / headroom per segment, the nvidia-smi maximum.
    python tests/smoke_report_s2.py JOBID"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_ptsz as G   # noqa: E402

SM = G.SMOKE
LOGS = Path("/scratch/yh6210/llmadd_torch/logs")
GIB = 2 ** 30
TOL = 2e-4


def part(log, seg):
    return log.split(f"=== segment {seg} start")[1].split(f"=== segment {seg} rc")[0] if f"=== segment {seg} start" in log else ""


def evals(txt):
    out = {}
    for x in txt.splitlines():
        if x.startswith("{"):
            try:
                j = json.loads(x)
            except json.JSONDecodeError:
                continue
            if isinstance(j, dict) and "step" in j and "lm" in j:
                out[j["step"]] = j["lm"]
    return out


def main():
    jid = sys.argv[1]
    log = (LOGS / f"llmadd_psz-smoke731s2_{jid}.out").read_text()
    lines = log.splitlines()
    print("\n".join(l for l in lines if l.startswith(("ALLOC:", "=== smoke731", "NVIDIA", "=== segment", "resumed from", "run guard",
                                                         "Traceback", "torch.OutOfMemory"))))
    sm = {}
    for l in lines:
        if l.startswith("SMOKE "):
            j = json.loads(l[6:])
            sm[j["seg"]] = j
    for k in ("S2A", "S2B", "S2C"):
        j = sm.get(k)
        if j:
            print(f"  {k}: ok {j['ok']}  alloc {j['max_alloc'] / GIB:6.2f}  reserved {j['max_reserved'] / GIB:6.2f}  headroom "
                  f"{(j['total'] - j['max_reserved']) / GIB:5.2f} GiB  wall {j['wall']:.0f} s" + (f"  ERR {j['err'][:200]}" if j["err"] else ""))
        else:
            print(f"  {k}: no SMOKE line")
    g = SM / f"gpu_{jid}.csv"
    if g.exists():
        rows = [r for r in csv.reader(g.read_text().splitlines()) if len(r) >= 5 and r[2].strip()[:1].isdigit()]
        if rows:
            print(f"  nvidia-smi max memory.used {max(float(r[2].split()[0]) for r in rows):.0f} MiB of {float(rows[0][3].split()[0]):.0f}")
    ea, eb = evals(part(log, "S2A")), evals(part(log, "S2B"))
    rc = next(iter((SM / "S2C").glob("run_*.json")), None) if (SM / "S2C").exists() else None
    ec = {e["step"]: e["lm"] for e in json.loads(rc.read_text())["curve"]} if rc else {}
    ref = G.lm0_ref("731m", 2)
    print(f"  step-0 lm: S2A {ea.get(0)!r}, S2C {ec.get(0)!r}, CPU reference {ref!r}")
    print(f"  resume: S2B step 200 {eb.get(200)!r} vs S2A step 200 {ea.get(200)!r}")
    crit = {
        "(1) setting on": "ALLOC: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in lines,
        "(2) ran, no OOM, no run guard": all(sm.get(k, {}).get("ok") is True for k in ("S2A", "S2B", "S2C"))
            and "OutOfMemoryError" not in log and not any(l.startswith("run guard") for l in lines),
        "(3) resumed at 200, lm equal": any(l.startswith("resumed from") and " at step 200 " in l for l in part(log, "S2B").splitlines())
            and eb.get(200) is not None and ea.get(200) is not None and abs(eb[200] - ea[200]) <= TOL,
        "(4) step-0 lm = CPU reference": ref is not None and all(v is not None and abs(v - ref) <= TOL for v in (ea.get(0), ec.get(0))),
        "(5) S2C reached 260": 0 in ec and 250 in ec,
    }
    v = {"job": jid, "amendment": 6, "pass": all(crit.values()), "criteria": crit,
         "smoke": {k: {kk: x for kk, x in j.items() if kk != "args"} for k, j in sm.items()}}
    (SM / f"smoke_report_{jid}.json").write_text(json.dumps(v, indent=1))
    print(f"\n## S2 SMOKE {'PASS' if v['pass'] else 'FAIL'}: {crit}")


if __name__ == "__main__":
    main()
