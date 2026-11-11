"""pt6comb CPU references (the method of runs/ptdom2/tests/refs_ptdom2.py / runs/ptdom/tests/a3/refs_a3.py) on runs/pt6comb/code
(= the ptdom snapshot 11f61b9) with the pt6comb jobs' own ARGS (make_jobs_pt6comb.args_of6):
  dry   <comp> <arm> <lr> <seed>   --dry_inputs -> tests/refs_<comp>/dry_<arm>_lr<lr>_s<seed>.{log,json}: the DRY_GREP sha256 + count,
                                   the splits, the table shas, per domain prior / shuf / lam (+ numbers target) of 'domains:'
  step0 <comp> <arm> <lr> <seed>   the trainer's main() up to the step-0 training forward (scale_work/step0_ng.py) -> lm, data_sha
  table <comp> [--write]           per seed (0-3, 8-15): DRY / DOMS / table shas one value over every dry cell of the seed (arm
                                   independence: none, all6, only-numbers, only-s4 at seed 0; all6 at seed 8), LM0 / DSHA0 one
                                   value over every step-0 cell -> runs/pt6comb/pt6comb_refs_<comp>.json (written once, chmod 444)
  ptdomcmp                         the comparison domains' splits / table shas at seeds 0-3, 8-15 = ptdom's / ptdom2's references
                                   (cities: batch A; countries / people: batch B1) and the s4 / dim splits = rb6's recorded DOMS
                                   (make_rb6m_tasks.DOMS, seeds 0-15): the per-domain streams do not depend on the composition
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
R = B / "runs" / "pt6comb"
PIN = R / "code" / "tools"
sys.path.insert(0, str(PIN)); sys.path.insert(0, str(B))
import make_rb6_tasks as P          # noqa: E402
import make_jobs_pt6comb as M6      # noqa: E402
MJ = M6.MJ
assert Path(P.__file__).resolve().parent == PIN.resolve()
for f, v in MJ.WANT_SHA.items():
    assert MJ.sha(PIN / f) == v, (f, MJ.sha(PIN / f), v)
STEP0 = B / "scale_work" / "step0_ng.py"
ENV = dict(os.environ, HF_HOME=str(B / "hf_cache"), PYTHONPATH=str(B / "pylib"), PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
ENV.pop("SLURM_JOB_ID", None)   # run_guard.claim would record a job id
T = R / "tests"
SEEDS = M6.SEL + M6.CONF


def outd(comp):
    d = T / f"refs_{comp}{os.environ.get('REFS_DIR_SUFFIX', '')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def key(arm, lr, seed):
    return f"{arm}_lr{lr}_s{seed}"


def parse_dry(txt, comp):
    g = MJ.dry_grep(P)
    sel = [l for l in txt.splitlines() if re.match(g, l)]
    h = hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest() if sel else None
    doms, tsha = {}, {}
    for d in M6.comp_doms(comp):
        m = re.search(rf"^\[{d}\] (\{{.*\}})$", txt, re.M)
        J = json.loads(m.group(1)) if m else {}
        doms[d] = {x: J[x] for x in M6.DOM_KEYS6[M6.dom_kind6(d)] if x in J} if m else None
        if d in MJ.ALL24 and d != "months" and m:
            tsha[d] = J["table"]["sha"]
    m = re.search(r"^domains: (\{.*\})$", txt, re.M)
    dl = json.loads(m.group(1)) if m else {}
    return sel, h, doms, tsha, dl


def dry(comp, arm, lr, seed, threads=2):
    k, D = key(arm, lr, seed), outd(comp)
    out = D / f"run_dry_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    log = D / f"dry_{k}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([MJ.PY, "pretrain_range.py"] + M6.args_of6(comp, arm, lr, seed, out).split() + ["--dry_inputs"], cwd=PIN,
                            env=dict(ENV, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    txt = log.read_text()
    sel, h, doms, tsha, dl = parse_dry(txt, comp)
    res = {"comp": comp, "arm": arm, "lr": lr, "seed": seed, "rc": rc, "sec": round(time.time() - t0), "n": len(sel), "sha": h,
           "doms": doms, "table_sha": tsha,
           "prior": {d: {x: v.get(x) for x in ("prior", "shuf", "lam", "target") if x in v} for d, v in dl.items()},
           "model_line": [l for l in txt.splitlines() if l.startswith("model:")],
           "target_line": [l for l in txt.splitlines() if l.startswith("target features")],
           "site_rule_lines": len(re.findall(r"^  site rule: the prior and the CKA read-out use the LAST token of each item name .*: "
                                             r"[0-9]+ sites in [0-9]+ lines asserted; ", txt, re.M)),
           "eval_lines_sha": hashlib.sha256("\n".join(l for l in txt.splitlines() if l.startswith("  eval ")).encode()).hexdigest(),
           "content_scan": re.findall(r"^content scan: .*$", txt, re.M),
           "token_share": re.findall(r"^token share over 50 batches.*$", txt, re.M),
           "slot_share": re.findall(r"^slot share of each domain.*$", txt, re.M)}
    (D / f"dry_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({x: res[x] for x in ("comp", "arm", "lr", "seed", "rc", "sec", "n", "sha", "site_rule_lines")}), flush=True)


def step0(comp, arm, lr, seed, threads=2):
    k, D = key(arm, lr, seed), outd(comp)
    out = D / f"run_step0_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    log = D / f"step0_{k}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([MJ.PY, str(STEP0), str(PIN)] + M6.args_of6(comp, arm, lr, seed, out).split(), cwd=PIN,
                            env=dict(ENV, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    m = re.search(r"^STEP0 (\{.*\})$", log.read_text(), re.M)
    res = {"comp": comp, "key": k, "arm": arm, "lr": lr, "seed": seed, "rc": rc, "sec": round(time.time() - t0),
           **(json.loads(m.group(1)) if m else {"error": log.read_text()[-2000:]})}
    (D / f"step0_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res)[:400], flush=True)


def want_prior(arm, comp):
    pri = M6.pri_of(arm, comp)
    return {d: {"prior": d in pri, "shuf": False, "lam": 1.0 if d in pri else 0.0} for d in M6.manifolds(comp)}


def cell_problems(f, j, comp):
    probs = []
    if j["rc"] != 0:
        return [f"{f.name}: rc {j['rc']}"]
    want = want_prior(j["arm"], comp)
    got = {d: {x: j["prior"].get(d, {}).get(x) for x in ("prior", "shuf", "lam")} for d in want}
    tgt = "helix3" if "numbers" in M6.pri_of(j["arm"], comp) else "none"
    if got != want or j["prior"]["numbers"].get("target") != tgt or set(j["prior"]) != set(want):
        probs.append(f"{f.name}: domains line {got} != the arm's intent {want} (numbers target {tgt})")
    if (j["target_line"] == ["target features (1000, 10)"]) != (tgt == "helix3") or (tgt == "none" and j["target_line"]):
        probs.append(f"{f.name}: target features line {j['target_line']} vs the numbers target {tgt}")
    n_site = len([d for d in M6.comp_doms(comp) if d in MJ.ALL24 and d != "months"])
    if j["model_line"] != [P.MODEL_LINE] or j["site_rule_lines"] != n_site or not j["content_scan"]:
        probs.append(f"{f.name}: model line / site rule lines {j['site_rule_lines']} (want {n_site}) / content scan")
    return probs


def table(comp, write=False):
    seeds, probs, D = {}, [], outd(comp)
    for f in sorted(D.glob("dry_*.json")):
        j = json.loads(f.read_text())
        if j["seed"] not in SEEDS:
            continue
        probs += cell_problems(f, j, comp)
        if j["rc"] != 0:
            continue
        e = seeds.setdefault(j["seed"], {k: set() for k in ("dry", "dom", "tsha", "eval", "lm0", "dsha0")} | {"cells": []})
        e["dry"].add((j["sha"], j["n"])); e["dom"].add(json.dumps(j["doms"], sort_keys=True))
        e["tsha"].add(json.dumps(j["table_sha"], sort_keys=True)); e["eval"].add(j["eval_lines_sha"]); e["cells"].append(f.stem)
    for f in sorted(D.glob("step0_*.json")):
        j = json.loads(f.read_text())
        if j["seed"] not in SEEDS:
            continue
        if j["rc"] != 0 or "lm" not in j:
            probs.append(f"{f.name}: rc {j['rc']} {j.get('error', '')[:200]}")
            continue
        e = seeds.setdefault(j["seed"], {k: set() for k in ("dry", "dom", "tsha", "eval", "lm0", "dsha0")} | {"cells": []})
        e.setdefault("lm0_arms", []).append((j["arm"], j["lm"]))
        e["dsha0"].add(j["data_sha"]); e["cells"].append(f.stem)
        if j["arm"] == "none":
            e["lm0"].add(j["lm"])
    for s, e in seeds.items():   # arm independence of the step-0 lm: every arm within 1e-5 of none's (CPU float noise, ptdom2)
        for arm, v in e.get("lm0_arms", []):
            if e["lm0"] and abs(v - next(iter(e["lm0"]))) > 1e-5:
                probs.append(f"seed {s}: step-0 lm of {arm} {v} differs from none's by > 1e-5")
    out = {}
    for s, e in sorted(seeds.items()):
        for k in ("dry", "dom", "tsha", "eval", "lm0", "dsha0"):
            if len(e[k]) != 1:
                probs.append(f"seed {s}: {len(e[k])} distinct {k} values over {e['cells']}")
        if all(len(e[k]) == 1 for k in ("dry", "dom", "tsha", "lm0", "dsha0")):
            out[s] = {"dry": list(next(iter(e["dry"]))), "dom": json.loads(next(iter(e["dom"]))),
                      "table_sha": json.loads(next(iter(e["tsha"]))), "lm0": next(iter(e["lm0"])), "dsha0": next(iter(e["dsha0"])),
                      "cells": e["cells"]}
    if sorted(out) != list(SEEDS):
        probs.append(f"seeds with a complete reference {sorted(out)} != {list(SEEDS)}")
    dsh = [v["dry"][0] for v in out.values()]
    if len(set(dsh)) != len(dsh):
        probs.append("two seeds share a DRY hash")
    if len({json.dumps(v["table_sha"], sort_keys=True) for v in out.values()}) > 1:
        probs.append("the table shas differ across seeds (the tables are seed-independent)")
    print(json.dumps({s: {k: v[k] for k in ("dry", "lm0", "dsha0")} for s, v in out.items()}, indent=1))
    print("problems:", probs or "none")
    tgt = M6.refs_path(comp)
    if write and not probs and tgt.exists():   # written once: a later identical run only compares
        same = {int(k): v for k, v in json.loads(tgt.read_text())["seeds"].items()} == {s: json.loads(json.dumps(v)) for s, v in out.items()}
        print(f"{tgt} exists (written once): the re-derivation is {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            probs.append("the written reference differs from this re-derivation")
    elif write and not probs:
        tgt.write_text(json.dumps({"design": "pt6comb", "comp": comp, "x": None, "domains": M6.domains_arg(comp),
                                   "method": "runs/pt6comb/tests/refs_pt6comb.py on runs/pt6comb/code (11f61b9, CPU; dry = --dry_inputs "
                                   "DRY_GREP lines; lm0 / dsha0 = scale_work/step0_ng.py)", "seeds": out}, indent=1))
        tgt.chmod(0o444)
        print("written", tgt, MJ.sha(tgt))
    print(f"TABLE {comp}", "PASS" if not probs else "FAIL", f"(seeds {sorted(out)})")


def ptdomcmp(comp="B"):
    """The per-domain splits do not depend on the composition (the streams are indexed by the trainer's fixed domain order): the
    comparison domains' split facts and table shas = ptdom's (seeds 0-3) / ptdom2's (seeds 8-15) references of their batch;
    s4 / dim's = rb6's recorded DOMS (make_rb6m_tasks.DOMS, seeds 0-15)."""
    import make_rb6m_tasks as RB
    D, probs, n = outd(comp), [], 0
    src = {"cities": "A", "countries": "B1", "people": "B1"}
    for f in sorted(D.glob("dry_none_lr*_s*.json")):
        j = json.loads(f.read_text())
        s = j["seed"]
        for d, b in src.items():
            ref = MJ.refs(b) if s < 8 else {int(k): v for k, v in json.loads((B / "runs/ptdom2" / f"ptdom2_refs_{b}.json").read_text())["seeds"].items()}
            if s not in ref:
                probs.append(f"{f.stem}: no {b} reference at seed {s}")
                continue
            same = j["doms"][d] == ref[s]["dom"][d] and j["table_sha"][d] == ref[s]["table_sha"][d]
            n += 1
            print(f"{f.stem} {d}: = batch {b}'s reference at seed {s}: {same}")
            if not same:
                probs.append(f"{f.stem} {d}: {j['doms'][d]} / {j['table_sha'][d]} vs {ref[s]['dom'][d]} / {ref[s]['table_sha'][d]}")
        for d in ("s4", "dim"):
            want = RB.DOMS.get(s, {}).get(d)
            got = {k: j["doms"][d].get(k) for k in (want or {})}
            same = want is not None and got == want
            n += 1
            print(f"{f.stem} {d}: = rb6's recorded split facts at seed {s}: {same}")
            if not same:
                probs.append(f"{f.stem} {d}: {got} vs rb6 {want}")
    print("problems:", probs or "none")
    print("PTDOMCMP", "PASS" if not probs and n else "FAIL", f"({n} domain x seed comparisons)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "table":
        table(sys.argv[2], "--write" in sys.argv)
    elif cmd == "ptdomcmp":
        ptdomcmp(*sys.argv[2:3])
    else:
        {"dry": dry, "step0": step0}[cmd](sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
