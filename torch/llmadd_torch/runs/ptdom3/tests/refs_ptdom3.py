"""ptdom3 CPU references (refs_pt6comb.py's method) on runs/ptdom3/code (ef7b8854) with the jobs' own ARGS (make_jobs_ptdom3.args_of3):
  dry   <arm> <lr> <seed> [canary]   --dry_inputs -> tests/refs_T3/dry_<key>.{log,json}: DRY_GREP sha256 + count, the toy3 split
                                     facts ([d] lines), per domain prior / shuf / lam of 'domains:', site-rule lines (0 expected)
  step0 <arm> <lr> <seed> [canary]   the trainer's main() up to the step-0 training forward (scale_work/step0_ng.py) -> lm, data_sha
  table [--write]                    seed 0: DRY / DOM one value over the screen dry cells (arm independence: none, only-cyc12,
                                     only-caesar), LM0 / DSHA0 one value over the screen step-0 cells (none, only-d6; |lm - none's|
                                     <= 1e-5); the canary (alltoy3, 400 steps) kept separately (its own DRY / LM0 / DSHA0; its
                                     split facts must equal the seed's) -> runs/ptdom3/ptdom3_refs_T3.json (written once, chmod 444)
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "ptdom3"
PIN = R / "code" / "tools"
sys.path.insert(0, str(PIN)); sys.path.insert(0, str(B))
import make_rb6_tasks as P          # noqa: E402
import make_jobs_ptdom3 as M3       # noqa: E402
MJ = M3.MJ
assert Path(P.__file__).resolve().parent == PIN.resolve()
for f, v in M3.WANT_SHA.items():
    assert M3.sha(PIN / f) == v, (f, M3.sha(PIN / f), v)
STEP0 = B / "scale_work" / "step0_ng.py"
ENV = dict(os.environ, HF_HOME=str(B / "hf_cache"), PYTHONPATH=str(B / "pylib"), PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
ENV.pop("SLURM_JOB_ID", None)   # run_guard.claim would record a job id
D = R / "tests" / "refs_T3"
D.mkdir(parents=True, exist_ok=True)


def key(arm, lr, seed, canary):
    return f"{'canary_' if canary else ''}{arm}_lr{lr}_s{seed}"


def run(kind, arm, lr, seed, canary, threads=2):
    k = key(arm, lr, seed, canary)
    out = D / f"run_{kind}_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    log = D / f"{kind}_{k}.log"
    a = M3.args_of3(arm, lr, seed, out, canary).split()
    cmd = [MJ.PY, "pretrain_range.py"] + a + ["--dry_inputs"] if kind == "dry" else [MJ.PY, str(STEP0), str(PIN)] + a
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=PIN, env=dict(ENV, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    txt = log.read_text()
    res = {"kind": kind, "arm": arm, "lr": lr, "seed": seed, "canary": canary, "rc": rc, "sec": round(time.time() - t0)}
    if kind == "dry":
        g = MJ.dry_grep(P)
        sel = [l for l in txt.splitlines() if re.match(g, l)]
        doms = {}
        for d in M3.DOMS:
            m = re.search(rf"^\[{d}\] (\{{.*\}})$", txt, re.M)
            J = json.loads(m.group(1)) if m else {}
            doms[d] = {x: J[x] for x in M3.DOM_KEYS3["toy3"] if x in J} if m else None
        m = re.search(r"^domains: (\{.*\})$", txt, re.M)
        dl = json.loads(m.group(1)) if m else {}
        res.update(n=len(sel), sha=hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest() if sel else None, doms=doms,
                   prior={d: {x: v.get(x) for x in ("prior", "shuf", "lam", "target") if x in v} for d, v in dl.items()},
                   model_line=[l for l in txt.splitlines() if l.startswith("model:")],
                   target_line=[l for l in txt.splitlines() if l.startswith("target features")],
                   site_rule_lines=len(re.findall(r"^  site rule: ", txt, re.M)),
                   content_scan=re.findall(r"^content scan: .*$", txt, re.M),
                   readout={d: (re.search(rf'^\[{d}\] .*"readout": "([a-z_]+)"', txt, re.M) or [None, None])[1] for d in M3.DOMS})
    else:
        m = re.search(r"^STEP0 (\{.*\})$", txt, re.M)
        res.update(json.loads(m.group(1)) if m else {"error": txt[-2000:]})
    (D / f"{kind}_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({x: res.get(x) for x in ("kind", "arm", "lr", "canary", "rc", "sec", "n", "sha", "lm", "data_sha", "site_rule_lines")}), flush=True)


def problems_dry(f, j):
    if j["rc"] != 0:
        return [f"{f.name}: rc {j['rc']}"]
    pri = M3.pri_of(j["arm"])
    want = {d: {"prior": d in pri, "shuf": False, "lam": 1.0 if d in pri else 0.0} for d in ("numbers",) + M3.DOMS}
    got = {d: {x: j["prior"].get(d, {}).get(x) for x in ("prior", "shuf", "lam")} for d in want}
    pr = []
    if got != want or j["prior"]["numbers"].get("target") != "none":
        pr.append(f"{f.name}: domains line {got} != intent {want}")
    if j["model_line"] != [P.MODEL_LINE] or j["site_rule_lines"] != 0 or not j["content_scan"] or j["target_line"]:
        pr.append(f"{f.name}: model line / site-rule lines {j['site_rule_lines']} (want 0) / content scan / target line {j['target_line']}")
    if j["readout"] != M3.READSET or any(v is None for v in j["doms"].values()):
        pr.append(f"{f.name}: read-out sets {j['readout']} != {M3.READSET} or a missing [d] line")
    return pr


def table(write=False):
    probs, scr, can = [], {"dry": set(), "dom": set(), "lm0": [], "dsha0": set(), "cells": []}, {}
    for f in sorted(D.glob("dry_*.json")):
        j = json.loads(f.read_text())
        probs += problems_dry(f, j)
        if j["rc"] != 0:
            continue
        if j["canary"]:
            can.update(dry=[j["sha"], j["n"]], dom=j["doms"])
        else:
            scr["dry"].add((j["sha"], j["n"])); scr["dom"].add(json.dumps(j["doms"], sort_keys=True)); scr["cells"].append(f.stem)
    for f in sorted(D.glob("step0_*.json")):
        j = json.loads(f.read_text())
        if j["rc"] != 0 or "lm" not in j:
            probs.append(f"{f.name}: rc {j['rc']} {j.get('error', '')[-300:]}"); continue
        if j["canary"]:
            can.update(lm0=j["lm"], dsha0=j["data_sha"])
        else:
            scr["lm0"].append((j["arm"], j["lm"])); scr["dsha0"].add(j["data_sha"]); scr["cells"].append(f.stem)
    lm_none = [v for a, v in scr["lm0"] if a == "none"]
    if len(lm_none) != 1:
        probs.append(f"step-0 none cells: {len(lm_none)}")
    for a, v in scr["lm0"]:
        if lm_none and abs(v - lm_none[0]) > 1e-5:
            probs.append(f"step-0 lm of {a} {v} differs from none's {lm_none[0]} by > 1e-5")
    for k in ("dry", "dom", "dsha0"):
        if len(scr[k]) != 1:
            probs.append(f"{len(scr[k])} distinct {k} values over {scr['cells']}")
    if set(can) != {"dry", "dom", "lm0", "dsha0"}:
        probs.append(f"canary reference incomplete: {sorted(can)}")
    seed = {}
    if not probs:
        dom = json.loads(next(iter(scr["dom"])))
        seed = {"dry": list(next(iter(scr["dry"]))), "dom": dom, "table_sha": {}, "lm0": lm_none[0], "dsha0": next(iter(scr["dsha0"])),
                "cells": scr["cells"]}
        if can["dom"] != dom:
            probs.append(f"canary split facts differ from the screen's: {can['dom']} vs {dom}")
        canref = {"dry": can["dry"], "dom": can["dom"], "table_sha": {}, "lm0": can["lm0"], "dsha0": can["dsha0"]}
        print(json.dumps({"seed0": {k: seed[k] for k in ("dry", "lm0", "dsha0")}, "canary": {k: canref[k] for k in ("dry", "lm0", "dsha0")}}, indent=1))
        print("n_readout:", {d: v["n_readout"] for d, v in dom.items()})
    print("problems:", probs or "none")
    if write and not probs:
        if M3.REFS.exists():
            same = json.loads(M3.REFS.read_text())["seeds"]["0"] == json.loads(json.dumps(seed))
            print(f"{M3.REFS} exists (written once): the re-derivation is {'IDENTICAL' if same else 'DIFFERENT'}")
            if not same:
                probs.append("the written reference differs")
        else:
            M3.REFS.write_text(json.dumps({"design": "ptdom3", "domains": M3.domains_arg(), "commit": M3.COMMIT,
                                           "method": "runs/ptdom3/tests/refs_ptdom3.py on runs/ptdom3/code (CPU; dry = --dry_inputs DRY_GREP "
                                           "lines; lm0 / dsha0 = scale_work/step0_ng.py)", "seeds": {"0": seed}, "canary": canref}, indent=1))
            M3.REFS.chmod(0o444)
            print("written", M3.REFS, M3.sha(M3.REFS))
    print("TABLE T3", "PASS" if not probs else "FAIL")


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table("--write" in sys.argv)
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), len(sys.argv) > 5 and sys.argv[5] == "canary")
