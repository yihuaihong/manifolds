"""pt6v2 CPU references (method = runs/pt6comb/tests/refs_pt6comb.py, adapted; on runs/pt6v2/code = Rev H ef7b8854):
  dry  <mix> <arm> <lr> <seed> [canary]   --dry_inputs of one cell -> tests/refs_<mix>/dry_<key>.json
  step0 <mix> <arm> <lr> <seed> [canary]  scale_work/step0_ng.py (step-0 lm, first-batch data sha) -> step0_<key>.json
  domrefs [--write]                       per-domain split facts + table shas of EVERY candidate domain at seeds 16-23, from the
                                          domain's SOURCE mix (pt6comb B / ptdom A, B2, B3 / ptdom3 T3, none@2e-3, on Rev H)
                                          -> runs/pt6v2/pt6v2_domrefs.json (written once, chmod 444)
  table <mix> [--write]                   the per-seed + canary reference of a pt6v2 composition -> make_jobs_pt6v2.refs_path
  cmp <mix>                               test 2 (split independence): the composition's per-domain facts at every seed = domrefs'
<mix> = a pt6v2 composition 'd1-d2-d3-d4-d5' (trainer order; 'comp' = COMPOSITION.json, 'placeholder') or 'src:<source>'."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_pt6v2 as V   # noqa: E402  (asserts the frozen generators' shas, the Rev H snapshot)
sys.argv = _argv
MJ, R = V.MJ, V.R
P = V.load_P()
PIN = R / "code" / "tools"
STEP0 = B / "scale_work" / "step0_ng.py"
ENV = dict(os.environ, HF_HOME=str(B / "hf_cache"), PYTHONPATH=str(B / "pylib"), PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
ENV.pop("SLURM_JOB_ID", None)
T = R / "tests"
SEEDS = V.SEEDS
DOMREFS = R / "pt6v2_domrefs.json"


def mix_doms(mix):
    """-> (key, domains in trainer order, is_source)."""
    if mix.startswith("src:"):
        src = mix[4:]
        return mix.replace(":", "_"), tuple(d for d in V.TRAINER_ORDER if d in V.SOURCE_DOMS[src]), True
    doms = (V.decided_comp() if mix == "comp" else V.PLACEHOLDER if mix == "placeholder" else V.comp_doms(mix.split("-")))
    assert doms, "COMPOSITION.json missing"
    return "-".join(doms), doms, False


def outd(mk):
    d = T / f"refs_{mk}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def key(arm, lr, seed, canary=False):
    return f"{arm}_lr{lr}_s{seed}" + ("_canary" if canary else "")


def args_of(doms, is_src, arm, lr, seed, out, canary=False):
    tag = (f"pt6v2src-{arm}-lr{lr}") if is_src else None
    return V.args_raw(doms, arm, lr, seed, out, canary, tag=tag)


def parse_dry(txt, doms):
    g = MJ.dry_grep(P)
    sel = [l for l in txt.splitlines() if re.match(g, l)]
    h = hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest() if sel else None
    dd, tsha = {}, {}
    for d in doms:
        m = re.search(rf"^\[{d}\] (\{{.*\}})$", txt, re.M)
        J = json.loads(m.group(1)) if m else {}
        dd[d] = {x: J[x] for x in V.DOM_KEYS_V2[V.dom_kind(d)] if x in J} if m else None
        if d in MJ.ALL24 and d != "months" and m:
            tsha[d] = J["table"]["sha"]
    m = re.search(r"^domains: (\{.*\})$", txt, re.M)
    return sel, h, dd, tsha, (json.loads(m.group(1)) if m else {})


def _run(cmd, log, threads):
    with open(log, "w") as fh:
        return subprocess.run(cmd, cwd=PIN, env=dict(ENV, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), stdout=fh,
                              stderr=subprocess.STDOUT).returncode


def dry(mix, arm, lr, seed, canary=False, threads=2):
    mk, doms, is_src = mix_doms(mix)
    k, D = key(arm, lr, seed, canary), outd(mk)
    out = D / f"run_dry_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    t0 = time.time()
    rc = _run([MJ.PY, "pretrain_range.py"] + args_of(doms, is_src, arm, lr, seed, out, canary).split() + ["--dry_inputs"], D / f"dry_{k}.log", threads)
    txt = (D / f"dry_{k}.log").read_text()
    sel, h, dd, tsha, dl = parse_dry(txt, doms)
    res = {"mix": mix, "arm": arm, "lr": lr, "seed": seed, "canary": canary, "rc": rc, "sec": round(time.time() - t0), "n": len(sel), "sha": h,
           "doms": dd, "table_sha": tsha,
           "prior": {d: {x: v.get(x) for x in ("prior", "shuf", "lam", "target") if x in v} for d, v in dl.items()},
           "model_line": [l for l in txt.splitlines() if l.startswith("model:")],
           "target_line": [l for l in txt.splitlines() if l.startswith("target features")],
           "site_rule_lines": len(re.findall(r"^  site rule: the prior and the CKA read-out use the LAST token of each item name .*: "
                                             r"[0-9]+ sites in [0-9]+ lines asserted; ", txt, re.M)),
           "readout_sets": re.findall(r'^\[(\w+)\] \{.*"readout": "(\w+)"', txt, re.M),
           "eval_lines_sha": hashlib.sha256("\n".join(l for l in txt.splitlines() if l.startswith("  eval ")).encode()).hexdigest(),
           "content_scan": re.findall(r"^content scan: .*$", txt, re.M)}
    (D / f"dry_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({x: res[x] for x in ("mix", "arm", "lr", "seed", "canary", "rc", "sec", "n", "sha", "site_rule_lines")}), flush=True)


def step0(mix, arm, lr, seed, canary=False, threads=2):
    mk, doms, is_src = mix_doms(mix)
    k, D = key(arm, lr, seed, canary), outd(mk)
    out = D / f"run_step0_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    t0 = time.time()
    rc = _run([MJ.PY, str(STEP0), str(PIN)] + args_of(doms, is_src, arm, lr, seed, out, canary).split(), D / f"step0_{k}.log", threads)
    lt = (D / f"step0_{k}.log").read_text()
    m = re.search(r"^STEP0 (\{.*\})$", lt, re.M)
    res = {"mix": mix, "arm": arm, "lr": lr, "seed": seed, "canary": canary, "rc": rc, "sec": round(time.time() - t0),
           **(json.loads(m.group(1)) if m else {"error": lt[-2000:]})}
    (D / f"step0_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res)[:400], flush=True)


def cell_problems(f, j, doms):
    if j["rc"] != 0:
        return [f"{f.name}: rc {j['rc']}"]
    pri = V.pri_of(j["arm"], doms)
    want = {d: {"prior": d in pri, "shuf": False, "lam": 1.0 if d in pri else 0.0} for d in V.manifolds(doms)}
    got = {d: {x: j["prior"].get(d, {}).get(x) for x in ("prior", "shuf", "lam")} for d in want}
    tgt = "helix3" if "numbers" in pri else "none"
    probs = []
    if got != want or j["prior"]["numbers"].get("target") != tgt or set(j["prior"]) != set(want):
        probs.append(f"{f.name}: domains line {got} != the arm's intent {want} (numbers target {tgt})")
    if (j["target_line"] == ["target features (1000, 10)"]) != (tgt == "helix3"):
        probs.append(f"{f.name}: target features line {j['target_line']} vs the numbers target {tgt}")
    n_site = len([d for d in doms if d in MJ.ALL24 and d != "months"])
    if j["model_line"] != [P.MODEL_LINE] or j["site_rule_lines"] != n_site or not j["content_scan"]:
        probs.append(f"{f.name}: model line / site rule lines {j['site_rule_lines']} (want {n_site}) / content scan")
    rs = dict(j.get("readout_sets", []))
    for d in doms:
        if d in V.TOY3 and rs.get(d) != V.M3.READSET[d]:
            probs.append(f"{f.name}: {d} read-out set {rs.get(d)} != {V.M3.READSET[d]}")
    return probs


def _collect(D, doms, canary):
    seeds, probs = {}, []
    for f in sorted(D.glob("dry_*.json")):
        j = json.loads(f.read_text())
        if bool(j.get("canary")) != canary:
            continue
        probs += cell_problems(f, j, doms)
        if j["rc"] != 0:
            continue
        e = seeds.setdefault(j["seed"], {k: set() for k in ("dry", "dom", "tsha", "lm0", "dsha0")} | {"cells": [], "lm0_arms": []})
        e["dry"].add((j["sha"], j["n"])); e["dom"].add(json.dumps(j["doms"], sort_keys=True))
        e["tsha"].add(json.dumps(j["table_sha"], sort_keys=True)); e["cells"].append(f.stem)
    for f in sorted(D.glob("step0_*.json")):
        j = json.loads(f.read_text())
        if bool(j.get("canary")) != canary:
            continue
        if j["rc"] != 0 or "lm" not in j:
            probs.append(f"{f.name}: rc {j['rc']} {j.get('error', '')[:200]}")
            continue
        e = seeds.setdefault(j["seed"], {k: set() for k in ("dry", "dom", "tsha", "lm0", "dsha0")} | {"cells": [], "lm0_arms": []})
        e["lm0_arms"].append((j["arm"], j["lm"])); e["dsha0"].add(j["data_sha"]); e["cells"].append(f.stem)
        e["lm0"].add(j["lm"]) if (j["arm"] == "none" or canary) else None
    out = {}
    for s, e in sorted(seeds.items()):
        for arm, v in e["lm0_arms"]:   # arm independence of the step-0 lm (CPU float noise 1e-5, as pt6comb / ptdom2)
            if e["lm0"] and abs(v - next(iter(e["lm0"]))) > 1e-5:
                probs.append(f"seed {s}: step-0 lm of {arm} {v} differs from the reference arm's by > 1e-5")
        ks = ("dry", "dom", "tsha", "lm0", "dsha0")
        for k in ks:
            if len(e[k]) != 1:
                probs.append(f"seed {s}{' canary' if canary else ''}: {len(e[k])} distinct {k} values over {e['cells']}")
        if all(len(e[k]) == 1 for k in ks):
            out[s] = {"dry": list(next(iter(e["dry"]))), "dom": json.loads(next(iter(e["dom"]))),
                      "table_sha": json.loads(next(iter(e["tsha"]))), "lm0": next(iter(e["lm0"])), "dsha0": next(iter(e["dsha0"])),
                      "cells": e["cells"]}
    return out, probs


def table(mix, write=False):
    mk, doms, is_src = mix_doms(mix)
    assert not is_src
    D = outd(mk)
    out, probs = _collect(D, doms, False)
    can, p2 = _collect(D, doms, True)
    probs += p2
    if sorted(out) != list(SEEDS):
        probs.append(f"seeds with a complete reference {sorted(out)} != {list(SEEDS)}")
    if sorted(can) != [V.CANARY[2]]:
        probs.append(f"canary reference seeds {sorted(can)} != [{V.CANARY[2]}]")
    dsh = [v["dry"][0] for v in out.values()]
    if len(set(dsh)) != len(dsh):
        probs.append("two seeds share a DRY hash")
    if len({json.dumps(v["table_sha"], sort_keys=True) for v in out.values()}) > 1:
        probs.append("the table shas differ across seeds (the tables are seed-independent)")
    print(json.dumps({s: {k: v[k] for k in ("dry", "lm0", "dsha0")} for s, v in out.items()}, indent=1))
    print("problems:", probs or "none")
    tgt = V.refs_path(doms)
    body = {"design": "pt6v2", "domains": V.domains_arg(doms),
            "method": "runs/pt6v2/tests/refs_pt6v2.py on runs/pt6v2/code (ef7b8854, CPU; dry = --dry_inputs DRY_GREP lines; lm0 / dsha0 = "
                      "scale_work/step0_ng.py; canary = its own pilot command line)",
            "seeds": {str(s): v for s, v in out.items()}, "canary": can.get(V.CANARY[2], {})}
    if write and not probs and tgt.exists():
        same = json.loads(tgt.read_text())["seeds"] == json.loads(json.dumps(body["seeds"]))
        print(f"{tgt} exists (written once): the re-derivation is {'IDENTICAL' if same else 'DIFFERENT'}")
        probs += [] if same else ["the written reference differs from this re-derivation"]
    elif write and not probs:
        tgt.write_text(json.dumps(body, indent=1))
        tgt.chmod(0o444)
        print("written", tgt, V.sha(tgt))
    print(f"TABLE {mk}", "PASS" if not probs else "FAIL", f"(seeds {sorted(out)}, canary {sorted(can)})")


def domrefs(write=False):
    res, probs = {"design": "pt6v2", "seeds": list(SEEDS), "method": "none@2e-3 --dry_inputs of each domain's source mix on Rev H "
                  "(refs_pt6v2.py)", "domains": {}}, []
    for src in V.SOURCE_DOMS:
        mk, doms, _ = mix_doms(f"src:{src}")
        for f in sorted(outd(mk).glob("dry_*.json")):
            j = json.loads(f.read_text())
            if j["rc"] != 0 or j["arm"] != "none":
                probs.append(f"{mk}/{f.name}: rc {j['rc']}") if j["rc"] else None
                continue
            for d in doms:
                if V.SOURCE.get(d) == src:   # a source mix also holds non-candidate domains (ptdom-B2: languages); 2026-11-19 fix
                    res["domains"].setdefault(d, {})[str(j["seed"])] = {"source": src, "dom": j["doms"][d], "table_sha": j["table_sha"].get(d)}
    for d in V.ALLOWED:
        got = sorted(int(s) for s in res["domains"].get(d, {}))
        if got != list(SEEDS):
            probs.append(f"{d}: domain reference seeds {got}")
    print("problems:", probs or "none")
    if write and not probs:
        if DOMREFS.exists():
            print(f"{DOMREFS} exists: {'IDENTICAL' if json.loads(DOMREFS.read_text())['domains'] == res['domains'] else 'DIFFERENT'}")
        else:
            DOMREFS.write_text(json.dumps(res, indent=1)); DOMREFS.chmod(0o444); print("written", DOMREFS, V.sha(DOMREFS))
    print("DOMREFS", "PASS" if not probs else "FAIL", f"({len(res['domains'])} domains)")


def cmp(mix):
    mk, doms, _ = mix_doms(mix)
    dr = json.loads(DOMREFS.read_text())["domains"]
    probs, n = [], 0
    for f in sorted(outd(mk).glob("dry_*.json")):
        j = json.loads(f.read_text())
        if j["rc"] != 0 or j.get("canary"):
            continue
        for d in doms:
            r = dr[d].get(str(j["seed"]))
            n += 1
            if r is None or j["doms"][d] != r["dom"] or j["table_sha"].get(d) != r["table_sha"]:
                probs.append(f"{f.stem} {d}: {j['doms'][d]} / {j['table_sha'].get(d)} != source {V.SOURCE[d]}: {r}")
    print("\n".join(probs[:20]) or "every split fact and table sha = the source mix's")
    print(f"SPLIT-INDEPENDENCE {mk}", "PASS" if n and not probs else "FAIL", f"({n} domain x cell comparisons)")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "table":
        table(sys.argv[2], "--write" in sys.argv)
    elif c == "domrefs":
        domrefs("--write" in sys.argv)
    elif c == "cmp":
        cmp(sys.argv[2])
    else:
        {"dry": dry, "step0": step0}[c](sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), len(sys.argv) > 6 and sys.argv[6] == "canary")
