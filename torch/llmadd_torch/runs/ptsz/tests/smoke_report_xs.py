"""Read the amendment-4 XS smoke (runs/ptsz/smoke, job llmadd_psz-smoke731xs): 731m memory under
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, and write the verdict submit_ptsz.sh reads (smoke/SMOKE_VERDICT.json).
Criteria (runs/ptsz/AUDIT.md "Amendment 4" + "Amendment 4.1", written before the job ran; all must hold):
  (1) the setting is on: the log prints 'ALLOC: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True';
  (2) XA, XB, XC ran (smoke_seg ok), no 'run guard:' line, each segment's max_allocated >= 0.8 x the same segment of
      22011938 (it really ran the 731m computation), and XB resumed at step 200 ('resumed from ... at step 200' in XB's part);
  (3) memory margin: max_reserved <= total - 2 GiB in XA, XB and XC (the criterion 22011938's B failed);
  (4) the setting takes effect: the reserved - allocated gap of XA <= 0.5 x A's gap in 22011938 and XB's <= 0.5 x B's
      (the same commands without the setting) [4.1: 'strictly below' -> at most half];
  (5) numerics [4.1, replaces the constant 'first smoke' criterion]: the step-0 lm of XA (log) and XC (run JSON) within 2e-4
      (the pttpp calibration tolerance) of 22011938's 11.74520206451416 (A = C = D, bit for bit; bitwise equality printed);
      XB's step-200 lm = XA's step-200 lm within 2e-4 (the resume; 22011938: bit for bit); the later evals XA@100 / XA@200 /
      XC@250 within 0.5 nat of A / A / C (ptheld Amendment DP's later-eval tolerance; the differences printed);
  (6) the first smoke's C D E / step-0 / resume criteria (smoke_report_22011938.json) stay true (D and E are not rerun).
Second path: the job's nvidia-smi memory.used maximum vs the card total (printed, not a criterion).
    python tests/smoke_report_xs.py JOBID"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_ptsz as G   # noqa: E402

SM = Path(os.environ.get("XS_SMOKE_DIR", G.SMOKE))            # overridable for the report's own test only
LOGS = Path(os.environ.get("XS_LOGS_DIR", "/scratch/yh6210/llmadd_torch/logs"))
GIB = 2 ** 30
FIRST = "22011938"
LM0_FIRST = 11.74520206451416          # 22011938: A, C and D step-0 lm, identical bit for bit (731m seed 0)
LM0_TOL, LATER_TOL = 2e-4, 0.5
FIRST_LATER = {("XA", 100): 7.655020236968994, ("XA", 200): 6.374513626098633, ("XC", 250): 6.115851402282715}  # A / A / C


def part(log, seg):
    return log.split(f"=== segment {seg} start")[1].split(f"=== segment {seg} rc")[0] if f"=== segment {seg} start" in log else ""


def evals(txt):   # the trainer's per-eval JSON lines of one segment's part of the log
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
    lf = LOGS / f"llmadd_psz-smoke731xs_{jid}.out"
    log = lf.read_text()
    print(f"# XS smoke job {jid}: {lf.name}")
    print("\n".join(l for l in log.splitlines() if l.startswith(("ALLOC:", "=== smoke731", "NVIDIA", "=== segment", "resumed from",
                                                                   "run guard", "Traceback", "torch.OutOfMemory", "RuntimeError"))))
    sm = {}
    for l in log.splitlines():
        if l.startswith("SMOKE "):
            j = json.loads(l[6:])
            sm[j["seg"]] = j
    first = json.loads((SM / f"smoke_report_{FIRST}.json").read_text())
    f_sm, f_crit = first["smoke"], first["criteria"]
    print("\n## memory (torch peak per segment)   [the same command without the setting, job %s]" % FIRST)
    gap, ran = {}, {}
    for k, ref in (("XA", "A"), ("XB", "B"), ("XC", "C")):
        j = sm.get(k)
        if j is None:
            print(f"  {k}: no SMOKE line")
            continue
        r = f_sm[ref]
        gap[k] = (j["max_reserved"] - j["max_alloc"], r["max_reserved"] - r["max_alloc"])
        ran[k] = j["max_alloc"] >= 0.8 * r["max_alloc"]
        print(f"  {k}: ok {j['ok']}  alloc {j['max_alloc'] / GIB:6.2f}  reserved {j['max_reserved'] / GIB:6.2f}  headroom "
              f"{(j['total'] - j['max_reserved']) / GIB:5.2f} GiB  gap {gap[k][0] / GIB:5.2f} GiB  wall {j['wall']:.0f} s"
              f"   [{ref}: alloc {r['max_alloc'] / GIB:6.2f}  reserved {r['max_reserved'] / GIB:6.2f}  gap {gap[k][1] / GIB:5.2f} GiB  "
              f"wall {r['wall']:.0f} s]" + (f"  ERR {j['err'][:160]}" if j["err"] else ""))
    g = SM / f"gpu_{jid}.csv"
    if g.exists():
        rows = [r for r in csv.reader(g.read_text().splitlines()) if len(r) >= 5 and r[2].strip()[:1].isdigit()]
        if rows:
            mem = [float(r[2].strip().split()[0]) for r in rows]
            tot = float(rows[0][3].strip().split()[0])
            print(f"  nvidia-smi (second path): max memory.used {max(mem):.0f} MiB of {tot:.0f} MiB ({max(mem) / tot:.1%}); "
                  f"22011938: 45193 MiB")
    print("\n## numerics (lm)")
    ea, eb = evals(part(log, "XA")), evals(part(log, "XB"))
    xc = next(iter((SM / "XC").glob("run_*.json")), None) if (SM / "XC").exists() else None
    ec = {e["step"]: e["lm"] for e in json.loads(xc.read_text())["curve"]} if xc else {}
    num = {}
    for name, v in (("XA step 0", ea.get(0)), ("XC step 0", ec.get(0))):
        num[name] = v is not None and abs(v - LM0_FIRST) <= LM0_TOL
        print(f"  {name}: {v!r} vs {LM0_FIRST!r}: diff {('%+.2e' % (v - LM0_FIRST)) if v is not None else '-'}; "
              f"bitwise {v == LM0_FIRST}")
    v, w = eb.get(200), ea.get(200)
    num["XB step 200 = XA step 200"] = v is not None and w is not None and abs(v - w) <= LM0_TOL
    print(f"  resume: XB step 200 {v!r} vs XA step 200 {w!r}; bitwise {v == w and v is not None}")
    for (seg, step), ref in FIRST_LATER.items():
        x = (ea if seg == "XA" else ec).get(step)
        num[f"{seg} step {step}"] = x is not None and abs(x - ref) <= LATER_TOL
        print(f"  {seg} step {step}: {x!r} vs 22011938 {ref!r}: diff {('%+.2e' % (x - ref)) if x is not None else '-'} (tolerance {LATER_TOL})")
    xb_part = part(log, "XB")
    crit = {
        "(1) setting on": "ALLOC: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in log.splitlines(),
        "(2) XA XB XC ran, no run guard, resumed at step 200":
            all(sm.get(k, {}).get("ok") is True and ran.get(k) is True for k in ("XA", "XB", "XC"))
            and not any(l.startswith("run guard") for l in log.splitlines())
            and any(l.startswith("resumed from") and " at step 200 " in l for l in xb_part.splitlines()),
        "(3) 731m reserved <= total - 2 GiB": all(k in sm and sm[k]["max_reserved"] <= sm[k]["total"] - 2 * GIB for k in ("XA", "XB", "XC")),
        "(4) takes effect (gap <= 0.5 x without)": all(k in gap and gap[k][0] <= 0.5 * gap[k][1] for k in ("XA", "XB")),
        "(5) numerics unchanged": all(num.values()),
        "(6) first smoke C D E / step-0 / resume": all(f_crit[c] is True for c in ("C D E ok", "resumed", "step-0 lm within 1e-3")),
    }
    verdict = {"job": jid, "amendment": "4.1", "pass": all(crit.values()), "criteria": crit, "numerics": num, "first_smoke": FIRST,
               "est": first["est"], "smoke": {k: {kk: v for kk, v in j.items() if kk != "args"} for k, j in sm.items()}}
    (SM / f"smoke_report_{jid}.json").write_text(json.dumps(verdict, indent=1))
    (SM / "SMOKE_VERDICT.json").write_text(json.dumps(verdict, indent=1))
    print(f"\n## XS SMOKE {'PASS' if verdict['pass'] else 'FAIL'}: {crit} -> {SM / 'SMOKE_VERDICT.json'}")


if __name__ == "__main__":
    main()
