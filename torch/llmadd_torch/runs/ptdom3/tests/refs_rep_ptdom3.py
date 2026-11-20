"""ptdom3 replication CPU references (refs_ptdom3.py's method: its run() with the jobs' own ARGS, output dir tests/refs_T3_rep):
  dry / step0 <arm> <lr> <seed>   as refs_ptdom3.py (no canary here)
  table [--write]                 per seed 0 / 1 / 2: DRY / DOM one value over its dry cells, LM0 / DSHA0 one value over its step-0
                                  cells (|lm - none's| <= 1e-5);
                                  seed 0 (none @ 1.6e-2 and @ 3.2e-2) must equal the screen's reference (ptdom3_refs_T3.json):
                                  the 3.2e-2 ARGS pass the trainer's input path and the seed-0 reference applies at that lr;
                                  SEED CHECK: seeds 1 and 2 differ from seed 0 and from each other in DRY sha, LM0, DSHA0 and every
                                  toy3 split_seed (the trainer really reads --seed);
                                  -> runs/ptdom3/ptdom3_refs_T3_rep.json (seeds 1, 2; written once, chmod 444)
"""
import json
import sys
from pathlib import Path

T = Path("/scratch/yh6210/llmadd_torch/runs/ptdom3/tests")
sys.path.insert(0, str(T))
import refs_ptdom3 as RF            # noqa: E402  (its run() / problems_dry(); asserts the snapshot shas)
sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_ptdom3r as MR      # noqa: E402  (REFS_REP)
M3 = RF.M3
RF.D = T / "refs_T3_rep"
RF.D.mkdir(parents=True, exist_ok=True)


def table(write=False):
    probs = []
    ref0, _ = M3.refs3()
    dry = {f.stem: json.loads(f.read_text()) for f in sorted(RF.D.glob("dry_*.json"))}
    st = {f.stem: json.loads(f.read_text()) for f in sorted(RF.D.glob("step0_*.json"))}
    for k, j in dry.items():
        probs += RF.problems_dry(Path(k + ".json"), j)
    seeds = {}
    for s in MR.REP_SEEDS:
        dj = [j for j in dry.values() if j["seed"] == s and j["rc"] == 0]
        sj = [j for j in st.values() if j["seed"] == s and j["rc"] == 0 and "lm" in j]
        bad = [k for k, j in st.items() if j["seed"] == s and (j["rc"] != 0 or "lm" not in j)]
        probs += [f"{k}: rc {st[k]['rc']} {st[k].get('error', '')[-300:]}" for k in bad]
        drys = {(j["sha"], j["n"]) for j in dj}
        doms = {json.dumps(j["doms"], sort_keys=True) for j in dj}
        dsh = {j["data_sha"] for j in sj}
        lmn = [j["lm"] for j in sj if j["arm"] == "none"]
        cells = sorted(k for k, j in {**dry, **st}.items() if j["seed"] == s)
        if len(drys) != 1 or len(doms) != 1 or len(dsh) != 1 or not lmn or any(abs(j["lm"] - lmn[0]) > 1e-5 for j in sj):
            probs.append(f"seed {s}: {len(drys)} dry / {len(doms)} dom / {len(dsh)} dsha0 values, lm {[(j['arm'], j['lr'], j['lm']) for j in sj]} over {cells}")
            continue
        seeds[s] = {"dry": list(next(iter(drys))), "dom": json.loads(next(iter(doms))), "table_sha": {}, "lm0": lmn[0],
                    "dsha0": next(iter(dsh)), "cells": cells}
        print(f"seed {s}: dry {seeds[s]['dry'][0][:12]} ({seeds[s]['dry'][1]} lines), lm0 {seeds[s]['lm0']:.6f}, dsha0 {seeds[s]['dsha0']}, "
              f"n_readout {({d: v['n_readout'] for d, v in seeds[s]['dom'].items()})}, cells {cells}")
    print(f"screen seed 0 reference: dry {ref0['dry'][0][:12]} ({ref0['dry'][1]}), lm0 {ref0['lm0']:.6f}, dsha0 {ref0['dsha0']}")
    if 0 in seeds:
        s0 = seeds[0]
        same0 = (s0["dry"] == ref0["dry"] and json.dumps(s0["dom"], sort_keys=True) == json.dumps(ref0["dom"], sort_keys=True)
                 and abs(s0["lm0"] - ref0["lm0"]) <= 1e-5 and s0["dsha0"] == ref0["dsha0"])
        lrs = sorted({j["lr"] for j in {**dry, **st}.values() if j["seed"] == 0})
        print(f"LR CHECK seed 0 at lr {lrs} = the screen's seed-0 reference:", "PASS" if same0 else "FAIL")
        if not same0:
            probs.append("seed-0 cells at the replication lrs differ from the screen's seed-0 reference")
    for s in (1, 2):
        if s not in seeds:
            continue
        for t, rt in ((0, ref0), (3 - s, seeds.get(3 - s))):
            if rt is None:
                continue
            diff = {"dry": seeds[s]["dry"][0] != rt["dry"][0], "lm0": abs(seeds[s]["lm0"] - rt["lm0"]) > 1e-6,
                    "dsha0": seeds[s]["dsha0"] != rt["dsha0"],
                    "split_seed": all(seeds[s]["dom"][d]["split_seed"] != rt["dom"][d]["split_seed"] for d in M3.DOMS)}
            print(f"SEED CHECK s{s} vs s{t}: " + ", ".join(f"{k} {'differs' if v else 'SAME'}" for k, v in diff.items()))
            if not all(diff.values()):
                probs.append(f"seed {s} vs seed {t}: not every seed-dependent field differs {diff}")
    if set(seeds) != set(MR.REP_SEEDS):
        probs.append(f"seeds with a reference: {sorted(seeds)}")
    print("problems:", probs or "none")
    if write and not probs:
        body = {"1": seeds[1], "2": seeds[2]}
        if MR.REFS_REP.exists():
            same = json.loads(MR.REFS_REP.read_text())["seeds"] == json.loads(json.dumps(body))
            print(f"{MR.REFS_REP} exists (written once): the re-derivation is {'IDENTICAL' if same else 'DIFFERENT'}")
            if not same:
                probs.append("the written reference differs")
        else:
            MR.REFS_REP.write_text(json.dumps({"design": M3.PREFIX, "domains": M3.domains_arg(), "commit": M3.COMMIT,
                                               "method": "runs/ptdom3/tests/refs_rep_ptdom3.py (refs_ptdom3.py's run()) on runs/ptdom3/code "
                                               "(CPU; dry = --dry_inputs DRY_GREP lines; lm0 / dsha0 = scale_work/step0_ng.py)",
                                               "seeds": body, "seed0_check": {k: seeds[0][k] for k in ("dry", "lm0", "dsha0", "cells")}}, indent=1))
            MR.REFS_REP.chmod(0o444)
            print("written", MR.REFS_REP, M3.sha(MR.REFS_REP))
    print("TABLE REP", "PASS" if not probs else "FAIL")


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table("--write" in sys.argv)
    else:
        RF.run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), False)
