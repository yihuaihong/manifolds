#!/usr/bin/env python3
"""pt6v2 canary milestone check (RUNBOOK_auto.md step 4). Read only: reads the canary's Slurm log, its out dir and sacct; writes nothing.

    python runs/pt6v2/tests/canary_check_pt6v2.py <canary job id | canary log path> [--jobs ID ...] [--out DIR]

(a) the canary ran and logged:  A1 'DEVICE <dev>' line; A2 '## 2. training' and the trainer's step-0 eval line ('"step": 0');
    A3 an eval line after step 0 (training steps ran); A4 the step-0 toy3 gradient line when the run has a toy3 domain;
    A5 the trainer's 'done in'; A6 '## done' (the job body reached its end).
    --jobs: every other pt6v2 job given: a log exists -> its '## 2. training' + '"step": 0' lines must be there (PENDING / no log yet:
    printed, not a failure; a log that ends in a Traceback / 'exit' without a step-0 eval is a failure).
(b) 'CANARY READ-OUT: 6 of 6 manifolds read: PASS' in the log or in <out>/canary_readout.txt (out = runs/pt6v2/pilot/<canary name>),
    and no '## CANARY READ-OUT FAILED' / 'NOT RENDERED'.
(c) the intent checks: the last 'CHECKS' line is 'CHECKS PASS' and no '^FAIL ' line. A flag-only CHECKS FAIL (AUDIT.md A5.1: GPU-numerics
    tolerance / CKA-rise only; the run is VALID data) prints FAIL (flag-only) and exits 2, so the coordinator decides.
Prints 'PASS|FAIL <item>' per item and 'CANARY CHECK PASS|FAIL'; exit 0 = all pass, 1 = a failure, 2 = only a flag-only intent FAIL.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
LOGS, PILOT = B / "logs", B / "runs" / "pt6v2" / "pilot"
TOY3 = ("cyc12", "torus", "hcube", "d6", "box5", "caesar")
res, flag_only = [], [False]


def ok(name, cond, info=""):
    res.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  {info}" if info != "" else ""))


def find_log(x):
    p = Path(x)
    if p.exists():
        return p
    if re.fullmatch(r"\d+", x):
        g = sorted(LOGS.glob(f"*_{x}.out"))
        return g[0] if len(g) == 1 else None
    return None


def sacct(jid):
    try:
        o = subprocess.run(["sacct", "-n", "-X", "-P", "-j", str(jid), "-o", "JobName,State,ExitCode,Elapsed"], capture_output=True,
                           text=True, timeout=30).stdout.strip()
        return o.splitlines()[-1] if o else "?"
    except Exception as ex:   # noqa: BLE001
        return f"sacct failed: {ex!r}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("canary", help="the canary's Slurm job id or its log path")
    ap.add_argument("--jobs", nargs="*", default=[], help="other pt6v2 job ids: started ones must have logged the step-0 eval")
    ap.add_argument("--out", help="the canary's out dir (default runs/pt6v2/pilot/<the canary name in the log file name>)")
    a = ap.parse_args()
    log = find_log(a.canary)
    print(f"canary log: {log}" + (f"; sacct: {sacct(a.canary)}" if re.fullmatch(r"\d+", a.canary) else ""))
    txt = log.read_text(errors="replace") if log else ""
    m = re.search(r"(pt6v2-canary-[a-z0-9_.-]+?-s\d+)_\d+\.out$", log.name) if log else None
    if not m:
        print(f"NOTE: the log name {log.name if log else None} is not a pt6v2 canary log (llmadd_pt6v2-canary-<arm>-lr<lr>-s<seed>_<id>.out)")
    out = Path(a.out) if a.out else (PILOT / m[1] if m else None)
    # (a)
    ok("A1 'DEVICE <dev>' line", re.search(r"^DEVICE \S+", txt, re.M), re.findall(r"^DEVICE .*$", txt, re.M)[:1])
    ok("A2 '## 2. training' and the step-0 eval line", "## 2. training" in txt and re.search(r'^\{"step": 0,', txt, re.M))
    steps = [int(x) for x in re.findall(r'^\{"step": (\d+),', txt, re.M)]
    ok("A3 an eval line after step 0 (training steps ran)", any(s > 0 for s in steps), f"evals at {steps[:3]}..{steps[-1:]}" if steps else "")
    dl = re.search(r'^domains: (\{.*)$', txt, re.M)
    has_toy3 = bool(dl) and any(f'"{d}"' in dl[1] for d in TOY3)
    ok("A4 the step-0 toy3 gradient line" + ("" if has_toy3 else " (no toy3 domain in the 'domains:' line: not required)"),
       ("step 0: toy3 true-prior gradient norms" in txt) if has_toy3 else bool(dl))
    ok("A5 the trainer's 'done in'", re.search(r"^done in \d", txt, re.M), re.findall(r"^done in .*$", txt, re.M)[:1])
    ok("A6 '## done' (the job body reached its end)", re.search(r"^## done ", txt, re.M))
    for j in a.jobs:
        lj = find_log(j)
        if lj is None:
            print(f"PENDING job {j}: no log yet ({sacct(j)})")
            continue
        tj = lj.read_text(errors="replace")
        started = "## 2. training" in tj and re.search(r'^\{"step": 0,', tj, re.M)
        if started:
            print(f"PASS job {j}: step-0 eval logged ({lj.name})")
            res.append(True)
        elif re.search(r"^Traceback|^## REFUSED|exit [1-9]", tj, re.M) or "RUNNING" not in sacct(j):
            ok(f"job {j}: started and logged the step-0 eval", False, f"{lj.name}; {sacct(j)}; tail: {tj.strip().splitlines()[-1:] if tj.strip() else ''}")
        else:
            print(f"NOTE job {j}: running, step-0 eval not logged yet ({lj.name})")
    # (b)
    rt = (out / "canary_readout.txt").read_text(errors="replace") if out and (out / "canary_readout.txt").exists() else ""
    cr = re.findall(r"CANARY READ-OUT: (\d+) of (\d+) manifolds read: (PASS|FAIL)", txt + "\n" + rt)
    bad = [x for x in ("## CANARY READ-OUT FAILED", "## CANARY READ-OUT NOT RENDERED") if x in txt]
    ok("B 'CANARY READ-OUT: 6 of 6 manifolds read: PASS'", cr and all(c == ("6", "6", "PASS") for c in cr) and not bad,
       {"lines": cr[:2], "errors": bad, "canary_readout.txt": str(out / "canary_readout.txt") if out else None, "exists": bool(rt)})
    # (c)
    ck = re.findall(r"^CHECKS (PASS|FAIL.*)$", txt, re.M)
    fails = re.findall(r"^FAIL (.*)$", txt, re.M)
    c_ok = bool(ck) and ck[-1] == "PASS" and not fails
    flag_only[0] = bool(ck) and not c_ok and "the run is VALID data" in ck[-1]
    ok("C every intent check PASS ('CHECKS PASS', no 'FAIL' line)" + (" [flag-only FAIL, AUDIT.md A5.1: VALID data; coordinator decides]"
                                                                      if flag_only[0] else ""), c_ok,
       {"CHECKS": ck[-1:], "n_pass_lines": len(re.findall(r"^PASS ", txt, re.M)), "failed": [f[:120] for f in fails[:4]]})
    n_fail = res.count(False)
    print(f"CANARY CHECK {'PASS' if not n_fail else 'FAIL'} {res.count(True)}/{len(res)}")
    sys.exit(0 if not n_fail else (2 if n_fail == 1 and flag_only[0] and not res[-1] else 1))


if __name__ == "__main__":
    main()
