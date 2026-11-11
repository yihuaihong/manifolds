"""ptdom2 CPU references (the method of runs/ptdom/tests/a3/refs_a3.py, read-only copy of its logic) on runs/ptdom2/code (= the
ptdom snapshot 11f61b9) with the ptdom2 jobs' own ARGS (make_jobs_ptdom2.args_of2):
  dry   <batch> <arm> <lr> <seed>   --dry_inputs -> tests/refs_<batch>/dry_<arm>_lr<lr>_s<seed>.{log,json}: the DRY_GREP sha256 +
                                    count, the splits, the table shas, per domain prior / shuf / lam / shuf_cka_to_true of 'domains:'
  step0 <batch> <arm> <lr> <seed>   the trainer's main() up to the step-0 training forward (scale_work/step0_ng.py) -> lm, data_sha
  table <batch> [--write]           per seed 8-15: DRY / DOMS / table shas one value over every dry cell of the seed (arm
                                    independence, incl. a shuffled arm), LM0 / DSHA0 one value over every step-0 cell
                                    -> runs/ptdom2/ptdom2_refs_<batch>.json (seeds 8-15 only; written once)
  ptdomcmp <batch>                  the dry / step-0 cells at seeds 0-7 (shuffled arms) against ptdom's own references
                                    (runs/ptdom/ptdom_refs*.json, read only): DRY rows, splits, tables, lm0, data_sha equal
  shufcka                           per batch and seed the trainer's shuf_cka_to_true of every domain (ptdom's job dry logs for
                                    seeds 0-7, these dry logs for 8-15) and the planned shuffled cells' values (< 0.15 asserted)
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
R = B / "runs" / "ptdom2"
PIN = R / "code" / "tools"
sys.path.insert(0, str(PIN)); sys.path.insert(0, str(B))
import make_rb6_tasks as P          # noqa: E402
import make_jobs_ptdom2 as M2       # noqa: E402
MJ = M2.MJ
assert Path(P.__file__).resolve().parent == PIN.resolve()
for f, v in MJ.WANT_SHA.items():
    assert MJ.sha(PIN / f) == v, (f, MJ.sha(PIN / f), v)
STEP0 = B / "scale_work" / "step0_ng.py"
ENV = dict(os.environ, HF_HOME=str(B / "hf_cache"), PYTHONPATH=str(B / "pylib"), PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
ENV.pop("SLURM_JOB_ID", None)   # run_guard.claim would record a job id
T = R / "tests"


def outd(batch):
    d = T / f"refs_{batch}{os.environ.get('REFS_DIR_SUFFIX', '')}"   # A3.3: a re-run into its own dir (compare, never write)
    d.mkdir(parents=True, exist_ok=True)
    return d


def key(arm, lr, seed):
    return f"{arm}_lr{lr}_s{seed}"


def parse_dry(txt, batch):
    g = MJ.dry_grep(P)
    sel = [l for l in txt.splitlines() if re.match(g, l)]
    h = hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest() if sel else None
    doms, tsha, scka = {}, {}, {}
    for d in MJ.BATCHES[batch]:
        m = re.search(rf"^\[{d}\] (\{{.*\}})$", txt, re.M)
        J = json.loads(m.group(1)) if m else {}
        doms[d] = {x: J[x] for x in MJ.DOM_KEYS[MJ.dom_kind(d)] if x in J} if m else None
        if d != "months" and m:
            tsha[d] = J["table"]["sha"]
        scka[d] = J.get("shuf_cka_to_true")
    m = re.search(r"^domains: (\{.*\})$", txt, re.M)
    dl = json.loads(m.group(1)) if m else {}
    return sel, h, doms, tsha, scka, dl


def dry(batch, arm, lr, seed, threads=2):
    k, D = key(arm, lr, seed), outd(batch)
    out = D / f"run_dry_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    log = D / f"dry_{k}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([MJ.PY, "pretrain_range.py"] + M2.args_of2(batch, arm, lr, seed, out).split() + ["--dry_inputs"], cwd=PIN,
                            env=dict(ENV, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    txt = log.read_text()
    sel, h, doms, tsha, scka, dl = parse_dry(txt, batch)
    res = {"batch": batch, "arm": arm, "lr": lr, "seed": seed, "rc": rc, "sec": round(time.time() - t0), "n": len(sel), "sha": h,
           "doms": doms, "table_sha": tsha, "shuf_cka_to_true": scka,
           "prior": {d: {x: v.get(x) for x in ("prior", "shuf", "lam", "target") if x in v} for d, v in dl.items()},
           "model_line": [l for l in txt.splitlines() if l.startswith("model:")],
           "site_rule_lines": len(re.findall(r"^  site rule: the prior and the CKA read-out use the LAST token of each item name .*: "
                                             r"[0-9]+ sites in [0-9]+ lines asserted; ", txt, re.M)),
           "eval_lines_sha": hashlib.sha256("\n".join(l for l in txt.splitlines() if l.startswith("  eval ")).encode()).hexdigest(),
           "content_scan": re.findall(r"^content scan: .*$", txt, re.M)}
    (D / f"dry_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({x: res[x] for x in ("batch", "arm", "lr", "seed", "rc", "sec", "n", "sha", "site_rule_lines")}), flush=True)


def step0(batch, arm, lr, seed, threads=2):
    k, D = key(arm, lr, seed), outd(batch)
    out = D / f"run_step0_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    log = D / f"step0_{k}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([MJ.PY, str(STEP0), str(PIN)] + M2.args_of2(batch, arm, lr, seed, out).split(), cwd=PIN,
                            env=dict(ENV, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    m = re.search(r"^STEP0 (\{.*\})$", log.read_text(), re.M)
    res = {"batch": batch, "key": k, "arm": arm, "lr": lr, "seed": seed, "rc": rc, "sec": round(time.time() - t0),
           **(json.loads(m.group(1)) if m else {"error": log.read_text()[-2000:]})}
    (D / f"step0_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res)[:400], flush=True)


def want_prior(arm, batch):
    pri = () if arm == "none" else (M2.dom(arm),)
    sh = (M2.dom(arm),) if M2.is_shuf(arm) else ()
    return {d: {"prior": d in pri, "shuf": d in sh, "lam": 1.0 if d in pri else 0.0} for d in ("numbers",) + MJ.BATCHES[batch]}


def cell_problems(f, j, batch):
    probs = []
    if j["rc"] != 0:
        return [f"{f.name}: rc {j['rc']}"]
    want = want_prior(j["arm"], batch)
    got = {d: {x: j["prior"].get(d, {}).get(x) for x in ("prior", "shuf", "lam")} for d in want}
    if got != want or j["prior"]["numbers"].get("target") != "none" or set(j["prior"]) != set(want):
        probs.append(f"{f.name}: domains line {got} != the arm's intent {want}")
    n_site = len([d for d in MJ.BATCHES[batch] if d != "months"])
    if j["model_line"] != [P.MODEL_LINE] or j["site_rule_lines"] != n_site or not j["content_scan"]:
        probs.append(f"{f.name}: model line / site rule lines {j['site_rule_lines']} / content scan")
    if M2.is_shuf(j["arm"]) and not (j["shuf_cka_to_true"][M2.dom(j["arm"])] < 0.15):
        probs.append(f"{f.name}: shuf_cka_to_true {j['shuf_cka_to_true'][M2.dom(j['arm'])]} >= 0.15")
    return probs


def table(batch, write=False):
    seeds, probs, D = {}, [], outd(batch)
    for f in sorted(D.glob("dry_*.json")):
        j = json.loads(f.read_text())
        if j["seed"] not in M2.R2_SEEDS:
            continue
        p_ = cell_problems(f, j, batch)
        probs += p_
        if j["rc"] != 0:
            continue
        e = seeds.setdefault(j["seed"], {k: set() for k in ("dry", "dom", "tsha", "eval", "lm0", "dsha0")} | {"cells": []})
        e["dry"].add((j["sha"], j["n"])); e["dom"].add(json.dumps(j["doms"], sort_keys=True))
        e["tsha"].add(json.dumps(j["table_sha"], sort_keys=True)); e["eval"].add(j["eval_lines_sha"]); e["cells"].append(f.stem)
    for f in sorted(D.glob("step0_*.json")):
        j = json.loads(f.read_text())
        if j["seed"] not in M2.R2_SEEDS:
            continue
        if j["rc"] != 0 or "lm" not in j:
            probs.append(f"{f.name}: rc {j['rc']} {j.get('error', '')[:200]}")
            continue
        e = seeds.setdefault(j["seed"], {k: set() for k in ("dry", "dom", "tsha", "eval", "lm0", "dsha0")} | {"cells": []})
        e.setdefault("lm0_arms", []).append((j["arm"], j["lm"]))
        e["dsha0"].add(j["data_sha"]); e["cells"].append(f.stem)
        if j["arm"] == "none":
            e["lm0"].add(j["lm"])
    for s, e in seeds.items():   # arm independence of the step-0 lm: every arm within 1e-5 of none's (CPU float noise;
        for arm, v in e.get("lm0_arms", []):   # tests/refs_B2_s8_rerun.out)
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
    if sorted(out) != list(M2.R2_SEEDS):
        probs.append(f"seeds with a complete reference {sorted(out)} != {list(M2.R2_SEEDS)}")
    old = MJ.refs(batch)   # ptdom's seeds 0-7: no DRY hash may repeat across seeds
    dsh = [v["dry"][0] for v in out.values()] + [v["dry"][0] for v in old.values()]
    if len(set(dsh)) != len(dsh):
        probs.append("two seeds share a DRY hash (incl. ptdom's seeds 0-7)")
    tab_old = {json.dumps(v["table_sha"], sort_keys=True) for v in old.values()}
    if any(json.dumps(v["table_sha"], sort_keys=True) not in tab_old for v in out.values()):
        probs.append("a table sha differs from ptdom's (the tables are seed-independent)")
    print(json.dumps({s: {k: v[k] for k in ("dry", "lm0", "dsha0")} for s, v in out.items()}, indent=1))
    print("problems:", probs or "none")
    tgt = R / f"ptdom2_refs_{batch}.json"
    if "--compare" in sys.argv and tgt.exists():   # A3.3: a re-derivation (e.g. a fresh CPU re-run) against the written reference
        same = {int(k): v for k, v in json.loads(tgt.read_text())["seeds"].items()} == {s: json.loads(json.dumps(v)) for s, v in out.items()}
        diffk = sorted({k for s in out for k in ("dry", "dom", "table_sha", "lm0", "dsha0")
                        if json.loads(json.dumps(out[s][k])) != json.loads(tgt.read_text())["seeds"].get(str(s), {}).get(k)})
        print(f"COMPARE {batch}: this re-derivation vs {tgt.name} ({MJ.sha(tgt)}): {'IDENTICAL' if same else 'DIFFERENT in ' + str(diffk)}"
              " (the 'cells' lists name this run's files and may differ)")
        write = False
    if write and not probs and tgt.exists():   # written once: a later identical run (a duplicate CPU job) only compares
        same = {int(k): v for k, v in json.loads(tgt.read_text())["seeds"].items()} == {s: json.loads(json.dumps(v)) for s, v in out.items()}
        print(f"{tgt} exists (written once): the re-derivation is {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            probs.append("the written reference differs from this re-derivation")
    elif write and not probs:
        tgt.write_text(json.dumps({"design": "ptdom2", "batch": batch, "method": "runs/ptdom2/tests/refs_ptdom2.py on runs/ptdom2/code "
                                   "(11f61b9, CPU; dry = --dry_inputs DRY_GREP lines; lm0 / dsha0 = scale_work/step0_ng.py)", "seeds": out}, indent=1))
        tgt.chmod(0o444)
        print("written", tgt, MJ.sha(tgt))
    print(f"TABLE {batch}", "PASS" if not probs else "FAIL", f"(seeds {sorted(out)})")


def ptdomcmp(batch):
    """Shuffled cells at seeds 0-7 vs ptdom's references (the S part and the pilot use them)."""
    old, probs, n = MJ.refs(batch), [], 0
    for f in sorted(outd(batch).glob("*_s[0-7].json")):
        j = json.loads(f.read_text())
        s = j["seed"]
        if s not in old:
            continue
        n += 1
        if f.name.startswith("dry_"):
            probs += cell_problems(f, j, batch)
            same = {"dry": [j["sha"], j["n"]] == old[s]["dry"], "dom": j["doms"] == old[s]["dom"], "table_sha": j["table_sha"] == old[s]["table_sha"]}
        else:
            same = {"lm0": j.get("lm") == old[s]["lm0"], "dsha0": j.get("data_sha") == old[s]["dsha0"]}
        print(f"{batch} {f.stem}: = ptdom's reference of seed {s}: {same}")
        if not all(same.values()):
            probs.append(f"{f.stem}: differs in {[k for k, v in same.items() if not v]}")
    print("problems:", probs or "none")
    print(f"PTDOMCMP {batch}", "PASS" if not probs and n else "FAIL", f"({n} cells)")


def shufcka():
    """shuf_cka_to_true of every domain per (batch, seed): seeds 0-7 from ptdom's job dirs' dry_inputs.txt (one none run per
    seed; read only), seeds 8-15 from this script's dry logs; the planned shuffled cells must have < 0.15 (the trainer asserts it
    at start; a value >= 0.15 would crash the job)."""
    vals, probs = {}, []
    for b in ("A", "B1", "B2", "B3"):
        for s in range(16):
            if s < 8:
                cand = sorted((MJ.R / "out").glob(f"ptdom-{b}-none-lr*-s{s}*/dry_inputs.txt"))
                txt = cand[0].read_text() if cand else None
            else:
                cand = sorted(outd(b).glob(f"dry_none_lr*_s{s}.log")) if (T / f"refs_{b}").exists() else []
                txt = cand[0].read_text() if cand else None
            if txt is None:
                continue
            vals[b, s] = parse_dry(txt, b)[4]
    plan = [(b, d, s) for d, b in M2.S_DOMS.items() for s in M2.S_SEEDS2 + M2.S_SEEDS] + [(b, d, s) for d, b in M2.R_DOMS.items() for s in M2.R2_SEEDS]   # A2: S at 8-15 (+ S0 0-7)
    plan += [(b, M2.dom(a), s) for b, a, lr, s in M2.PILOT]
    for b, d, s in plan:
        v = vals.get((b, s), {}).get(d)
        if v is None:
            probs.append(f"{b} {d} s{s}: no value")
        elif not v < 0.15:
            probs.append(f"{b} {d} s{s}: shuf_cka_to_true {v} >= 0.15")
    rows = {}
    for b, d, s in plan:
        rows.setdefault(f"{b} {d}", []).append(vals.get((b, s), {}).get(d))
    for k, v in rows.items():
        vv = [x for x in v if x is not None]
        print(f"  {k:20s} shuf_cka_to_true over its {len(v)} planned seeds: max {max(vv):.4f}, min {min(vv):.4f}" if vv else f"  {k}: -")
    print("problems:", probs or "none")
    print("SHUFCKA", "PASS" if not probs else "FAIL", f"({len(plan)} planned shuffled cells)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "table":
        table(sys.argv[2], "--write" in sys.argv)
    elif cmd == "ptdomcmp":
        ptdomcmp(sys.argv[2])
    elif cmd == "shufcka":
        shufcka()
    else:
        {"dry": dry, "step0": step0}[cmd](sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
