"""CPU references of the ptdom jobs (the method of scale_work/rb190dim_scripts/refs_dim.py) on the batch's own snapshot
(runs/ptdom/code/tools) with the jobs' own ARGS (make_jobs_ptdom.args_of):
  dry   <arm> <lr> <seed>   --dry_inputs -> refs/dry_<arm>_lr<lr>_s<seed>.{log,json} (DRY_GREP sha256 + count, the splits,
                            the anchor table shas, the per-domain prior / lam of the 'domains:' line)
  step0 <arm> <lr> <seed>   the trainer's own main() up to the step-0 training forward (CPU, bf16 autocast, no_grad:
                            scale_work/step0_ng.py, the rb6 method) -> refs/step0_<...>.json (lm, data_sha)
  table [--write]           per seed: DRY / DOMS / table shas one value over every dry cell of the seed (arm independence),
                            LM0 / DSHA0 one value over every step-0 cell of the seed -> runs/ptdom/ptdom_refs.json
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
R = B / "runs" / "ptdom"
PIN = R / "code" / "tools"
sys.path.insert(0, str(PIN)); sys.path.insert(0, str(B))
import make_rb6_tasks as P      # noqa: E402
import make_jobs_ptdom as M     # noqa: E402

assert Path(P.__file__).resolve().parent == PIN.resolve()
for f, v in M.WANT_SHA.items():
    assert M.sha(PIN / f) == v, (f, M.sha(PIN / f), v)
STEP0 = B / "scale_work" / "step0_ng.py"
ENV = dict(os.environ, HF_HOME=str(B / "hf_cache"), PYTHONPATH=str(B / "pylib"), PYTHONDONTWRITEBYTECODE="1",
           TOKENIZERS_PARALLELISM="false")
ENV.pop("SLURM_JOB_ID", None)   # run_guard.claim would record a job id
OUTD = R / "tests" / "refs"
OUTD.mkdir(parents=True, exist_ok=True)


def key(arm, lr, seed):
    return f"{arm}_lr{lr}_s{seed}"


def dry(arm, lr, seed, threads=2):
    k = key(arm, lr, seed)
    out = OUTD / f"run_dry_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    log = OUTD / f"dry_{k}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([M.PY, "pretrain_range.py"] + M.args_of(arm, lr, seed, out).split() + ["--dry_inputs"], cwd=PIN,
                            env=dict(ENV, OMP_NUM_THREADS=str(threads)), stdout=fh, stderr=subprocess.STDOUT).returncode
    txt = log.read_text()
    g = M.dry_grep(P)
    sel = [l for l in txt.splitlines() if re.match(g, l)]
    h = hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest() if sel else None
    doms, tsha = {}, {}
    for d in M.SIX:
        m = re.search(rf"^\[{d}\] (\{{.*\}})$", txt, re.M)
        J = json.loads(m.group(1)) if m else {}
        kk = M.DOM_KEYS["months" if d == "months" else "anchor"]
        doms[d] = {x: J[x] for x in kk if x in J} if m else None
        if d in M.ANC and m:
            tsha[d] = J["table"]["sha"]
    m = re.search(r"^domains: (\{.*\})$", txt, re.M)
    dl = json.loads(m.group(1)) if m else {}
    res = {"arm": arm, "lr": lr, "seed": seed, "rc": rc, "sec": round(time.time() - t0), "n": len(sel), "sha": h,
           "doms": doms, "table_sha": tsha,
           "prior": {d: {x: v.get(x) for x in ("prior", "shuf", "lam", "target") if x in v} for d, v in dl.items()},
           "model_line": [l for l in txt.splitlines() if l.startswith("model:")],
           "site_rule_lines": len(re.findall(r"^  site rule: .*9000 sites in 3000 lines asserted", txt, re.M)),
           "eval_lines_sha": hashlib.sha256("\n".join(l for l in txt.splitlines() if l.startswith("  eval ")).encode()).hexdigest(),
           "token_share": re.findall(r"^token share .*$", txt, re.M)}
    (OUTD / f"dry_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({x: res[x] for x in ("arm", "lr", "seed", "rc", "sec", "n", "sha", "site_rule_lines")}), flush=True)


def step0(arm, lr, seed, threads=2):
    k = key(arm, lr, seed)
    out = OUTD / f"run_step0_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    log = OUTD / f"step0_{k}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([M.PY, str(STEP0), str(PIN)] + M.args_of(arm, lr, seed, out).split(), cwd=PIN,
                            env=dict(ENV, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    m = re.search(r"^STEP0 (\{.*\})$", log.read_text(), re.M)
    res = {"key": k, "arm": arm, "lr": lr, "seed": seed, "rc": rc, "sec": round(time.time() - t0),
           **(json.loads(m.group(1)) if m else {"error": log.read_text()[-2000:]})}
    (OUTD / f"step0_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res)[:400], flush=True)


def table(write=False):
    seeds, probs = {}, []
    for f in sorted(OUTD.glob("dry_*.json")):
        j = json.loads(f.read_text())
        s = j["seed"]
        if j["rc"] != 0:
            probs.append(f"{f.name}: rc {j['rc']}")
            continue
        pri = () if j["arm"] == "none" else (j["arm"][5:],)
        want = {d: {"prior": d in pri, "shuf": False, "lam": 1.0 if d in pri else 0.0} for d in ("numbers",) + M.SIX}
        got = {d: {x: j["prior"][d][x] for x in ("prior", "shuf", "lam")} for d in want}
        if got != want or j["prior"]["numbers"].get("target") != "none":
            probs.append(f"{f.name}: domains line {got} != the arm's intent")
        if j["model_line"] != [P.MODEL_LINE] or j["site_rule_lines"] != 5:
            probs.append(f"{f.name}: model line {j['model_line']} / site rule lines {j['site_rule_lines']}")
        e = seeds.setdefault(s, {"dry": set(), "dom": set(), "tsha": set(), "eval": set(), "lm0": set(), "dsha0": set(), "cells": []})
        e["dry"].add((j["sha"], j["n"])); e["dom"].add(json.dumps(j["doms"], sort_keys=True))
        e["tsha"].add(json.dumps(j["table_sha"], sort_keys=True)); e["eval"].add(j["eval_lines_sha"]); e["cells"].append(f.stem)
    for f in sorted(OUTD.glob("step0_*.json")):
        j = json.loads(f.read_text())
        if j["rc"] != 0 or "lm" not in j:
            probs.append(f"{f.name}: rc {j['rc']} {j.get('error', '')[:200]}")
            continue
        e = seeds.setdefault(j["seed"], {"dry": set(), "dom": set(), "tsha": set(), "eval": set(), "lm0": set(), "dsha0": set(), "cells": []})
        e["lm0"].add(j["lm"]); e["dsha0"].add(j["data_sha"]); e["cells"].append(f.stem)
    out = {}
    for s, e in sorted(seeds.items()):
        for k in ("dry", "dom", "tsha", "eval", "lm0", "dsha0"):
            if len(e[k]) != 1:
                probs.append(f"seed {s}: {len(e[k])} distinct {k} values over {e['cells']}")
        if all(len(e[k]) == 1 for k in ("dry", "dom", "tsha", "lm0", "dsha0")):
            out[s] = {"dry": list(next(iter(e["dry"]))), "dom": json.loads(next(iter(e["dom"]))),
                      "table_sha": json.loads(next(iter(e["tsha"]))), "lm0": next(iter(e["lm0"])), "dsha0": next(iter(e["dsha0"])),
                      "cells": e["cells"]}
    dsh = [v["dry"][0] for v in out.values()]
    if len(set(dsh)) != len(dsh):
        probs.append("two seeds share a DRY hash")
    print(json.dumps({s: {k: v[k] for k in ("dry", "lm0", "dsha0", "table_sha")} for s, v in out.items()}, indent=1))
    print("problems:", probs or "none")
    if write and not probs:
        M.REFS.write_text(json.dumps({"method": "tests/refs_ptdom.py on runs/ptdom/code (CPU; dry = --dry_inputs DRY_GREP lines; "
                                      "lm0 / dsha0 = scale_work/step0_ng.py)", "seeds": out}, indent=1))
        print("written", M.REFS)
    print("TABLE", "PASS" if not probs else "FAIL")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "table":
        table("--write" in sys.argv)
    else:
        {"dry": dry, "step0": step0}[cmd](sys.argv[2], sys.argv[3], int(sys.argv[4]))
