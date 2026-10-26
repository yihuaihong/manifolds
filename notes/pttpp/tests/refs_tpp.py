"""Torch CPU references of the pttpp jobs (llmadd_torch/make_jobs_pttpp.py), the pt2s505 / ptlr_1b method on the batch's own
snapshots (runs/pttpp/code7 = F87's trainer a3bce3ab63c1, runs/pttpp/code2 = pt2s505's 93e22bf4c4c0):
  dry   <group> <arm> <seed>    --dry_inputs with the job's own ARGS (stage A, the group's first lr; --out -> refs_out/)
  step0 <group> <arm> <seed>    step 0 of the trainer's own main() (CPU, bf16 autocast, no_grad; scale_work/step0_ng.py)
  table [--write]               the identities below; --write -> llmadd_torch/pttpp_refs.json (only if all hold)
Identities:
  - model line of every dry run = make_jobs_pttpp.MODEL_LINE[size] (n_params and gradient checkpointing as designed);
  - 2%: the 9 DRY rows' sha256 identical across arms and = make_jobs_pttpp.dry2() (the Burst pt2s190 rows with this
    model line; at 190M = make_pt2main_tasks.DRY); 3 rows with 2 operand sites; target line only for win-ctx;
  - 7%: the dry log minus the model line = F87's 505M job log of the same arm and seed (its dry section), seeds 0-1;
  - step-0 lm identical across arms (seed 0) and, where a GPU value exists (7% 190M / 505M from F87's runs, 2% 190M from
    ptsize999 / Burst pt2s190), |CPU - GPU| <= 2e-4 (the calibration; GPU values are the references written for those)."""
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

G2 = T.g2()
ENV = dict(os.environ, HF_HOME="/scratch/yh6210/llmadd_torch/hf_cache", PYTHONPATH="/scratch/yh6210/llmadd_torch/pylib",
           PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
ENV.pop("SLURM_JOB_ID", None)
OUTD = HERE / "refs_out"
OUTD.mkdir(exist_ok=True)
RUNS = Path("/scratch/yh6210/llmadd_torch/runs")
LOGS = Path("/scratch/yh6210/llmadd_torch/logs")


def job_args(gid, arm, seed, out):
    """the job's own ARGS (stage A, the arm's first lr), --out redirected to the scratch dir"""
    lr = T.GRID[gid][arm][0] if gid in T.GRID else dict(T.BRIDGE_CELLS)[arm]
    F = T.facts(gid, arm, lr, seed, "A")
    if F["setting"] == "2":
        OUT, args = T.args2(G2, F)
        args = args.replace(f"--out {OUT}/{F['tag']}", f"--out {out}/{F['tag']}")
    else:
        args = T.args7(F).replace(f"--out {T.R / 'out' / F['tag']}", f"--out {out}/{F['tag']}")
    assert f"--out {out}/" in args
    return F, args.split()


def tools(F):
    return T.R / ("code2" if F["setting"] == "2" else "code7") / "tools"


def dry(gid, arm, seed):
    k = f"{gid}_{arm}_s{seed}"
    out, log = OUTD / f"run_dry_{k}", OUTD / f"dry_{k}.log"
    F, args = job_args(gid, arm, seed, out)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([T.PY, "pretrain_range.py"] + args + ["--dry_inputs"], cwd=tools(F),
                            env=dict(ENV, OMP_NUM_THREADS="1"), stdout=fh, stderr=subprocess.STDOUT).returncode
    g = G2.DRY_GREP.replace('"', '\\"')
    h = subprocess.run(["bash", "-c", f'set +o pipefail; grep -E "{g}" "{log}" | head -n 9 | sha256sum | cut -d" " -f1'],
                       capture_output=True, text=True).stdout.strip()
    txt = log.read_text()
    res = {"gid": gid, "arm": arm, "seed": seed, "setting": F["setting"], "size": F["size"], "rc": rc,
           "sec": round(time.time() - t0), "sha": h, "model_line": [l for l in txt.splitlines() if l.startswith("model:")],
           "rows_sites": re.findall(r"^train row [012] \(len 512, (\d+) prior sites at ", txt, re.M),
           "target_line": re.findall(r"^target features .*$", txt, re.M)}
    (OUTD / f"dry_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def step0(gid, arm, seed, threads=2):
    k = f"{gid}_{arm}_s{seed}"
    out, log = OUTD / f"run_step0_{k}", OUTD / f"step0_{k}.log"
    F, args = job_args(gid, arm, seed, out)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([T.PY, str(HERE.parent / "step0_ng.py"), str(tools(F))] + args, cwd=tools(F),
                            env=dict(ENV, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    m = re.search(r"^STEP0 (\{.*\})$", log.read_text(), re.M)
    res = {"gid": gid, "arm": arm, "seed": seed, "setting": F["setting"], "size": F["size"], "rc": rc,
           "sec": round(time.time() - t0), **(json.loads(m.group(1)) if m else {"error": log.read_text()[-2000:]})}
    (OUTD / f"step0_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res)[:400], flush=True)


def gpu_lm0():
    """step-0 lm of existing GPU runs: (setting, size) -> {seed: value} (identical across the source's cells, asserted)"""
    out = {}
    for (st, size), pats in {("7", "190m"): ["ptlr/out/190m-lr*-7pct-*/run_*.json"],
                             ("7", "505m"): ["ptlr/out/500m-lr*-7pct-*/run_*.json"],
                             ("7", "1b"): ["ptlr_1b/out/1b-lr*-7pct-*/run_*.json"],
                             ("7", "53m"): ["ptlr/out/53m-lr*-7pct-*/run_*.json"],          # read-out only (reused cell)
                             ("2", "505m"): ["pt2s505/out/pt2s505-lr*-s*/*/run_*.json"],     # read-out only (reused cell)
                             ("2", "190m"): []}.items():
        d = {}
        for pat in pats:
            for f in glob.glob(str(RUNS / pat)):
                if f.endswith(".json"):
                    r = json.loads(Path(f).read_text())
                    d.setdefault(r["args"]["seed"], set()).add(r["curve"][0]["lm"])
        out[(st, size)] = d
    d = {s: {v} for s, v in enumerate(G2.LM0[("190m", "pretrain_data")])}   # Torch ptsize999 190m-2pct-none, seeds 0-3
    for f in glob.glob(str(T.J.BURST / "results" / "*-pt2s190-lr*-*-s*" / "*" / "run_*.json")):
        r = json.loads(Path(f).read_text())
        d.setdefault(r["args"]["seed"], set()).add(r["curve"][0]["lm"])   # Burst A100 (other cluster, same pinned trainer)
    out[("2", "190m")] = d
    return out


def table():
    D = [json.loads(p.read_text()) for p in sorted(OUTD.glob("dry_*.json"))]
    S = [json.loads(p.read_text()) for p in sorted(OUTD.glob("step0_*.json"))]
    ok, refs = True, {"lm0": {}, "dry2": {}, "calibration": {}, "gpu_spread": {}}
    for d in D:
        good = d["rc"] == 0 and d["model_line"] == [T.MODEL_LINE[d["size"]]] and d["rows_sites"] == [str(2 if d["setting"] == "2" else 8)] * 3 \
            and d["target_line"] == ([] if d["arm"] == "none" else ["target features (1000, 10)"])
        ok &= good
        if not good:
            print("DRY FAIL", d)
    for size in ("190m", "294m"):
        for s in range(8):
            ds = [d for d in D if d["setting"] == "2" and d["size"] == size and d["seed"] == s]
            exp = T.dry2(G2, size, s)
            good = len(ds) == 2 and all(d["sha"] == exp for d in ds)
            ok &= good
            print(f"2% {size} s{s} dry rows: {sorted({d['sha'][:12] for d in ds})} vs expected {exp[:12]}: {'OK' if good else 'FAIL'}")
            if good:
                refs["dry2"].setdefault(size, {})[str(s)] = exp
    # 7%: the dry log minus the model line = F87's 505M job log of the same arm and seed (dry section)
    for d in D:
        if d["setting"] != "7" or d["seed"] > 1:
            continue
        lr = {"none": "9.6e-3", "win-ctx": "1.2e-3"}[d["arm"]]
        fs = sorted(LOGS.glob(f"llmadd_ptlr-500m-lr{lr}-7pct-{d['arm']}-s{d['seed']}_*.out"))
        ref = fs[0].read_text().splitlines()
        mine = (OUTD / f"dry_{d['gid']}_{d['arm']}_s{d['seed']}.log").read_text().splitlines()
        mine = mine[next(j for j, l in enumerate(mine) if l.startswith("line:")):]   # library warnings before it may differ
        i = next(j for j, l in enumerate(ref) if l.startswith("line:"))
        n = len(mine)
        same = [a == b or (a.startswith("model:") and b.startswith("model:")) for a, b in zip(mine, ref[i:i + n])]
        good = all(same) and len(ref[i:i + n]) == n
        ok &= good
        print(f"7% {d['gid']} {d['arm']} s{d['seed']}: dry log ({n} lines) = F87 505M job log {fs[0].name} except the model line: "
              f"{'OK' if good else 'FAIL at line ' + str(same.index(False) if False in same else n)}")
    G = gpu_lm0()
    cpu = {}
    for x in S:
        if x.get("lm") is None or x["rc"] != 0:
            ok = False
            print("STEP0 FAIL", x)
            continue
        cpu.setdefault((x["setting"], x["size"], x["seed"]), {})[x["arm"]] = x["lm"]
    for (st, size, s), v in sorted(cpu.items()):
        same = len(set(v.values())) == 1
        ok &= same
        val = next(iter(v.values()))
        g = G.get((st, size), {}).get(s)
        line = f"{st}% {size} s{s}: CPU {val:.6f} ({'identical across ' + '/'.join(sorted(v)) if same else 'DIFFER ' + str(v)})"
        if g:
            gv = sorted(g)
            spread = max(gv) - min(gv)
            cal = val - gv[0]
            good = abs(cal) <= 2e-4 and spread <= 1e-3
            ok &= good
            refs["calibration"][f"{st}|{size}|{s}"] = cal
            refs["gpu_spread"][f"{st}|{size}|{s}"] = spread
            line += f"; GPU {gv[0]:.6f} (spread over the source runs {spread:.1e}); CPU - GPU {cal:+.1e} {'OK' if good else 'FAIL'}"
        print(line)
    for (st, size), d in G.items():   # GPU values are the references where they exist (the jobs' runs pair with them)
        for s, g in d.items():
            refs["lm0"].setdefault(f"{st}|{size}", {})[str(s)] = sorted(g)[0]
    for (st, size, s), v in cpu.items():
        if f"{st}|{size}" not in refs["lm0"] or str(s) not in refs["lm0"][f"{st}|{size}"]:
            refs["lm0"].setdefault(f"{st}|{size}", {})[str(s)] = next(iter(v.values()))
    need = [("7", "53m"), ("7", "93m"), ("7", "190m"), ("7", "294m"), ("7", "505m"), ("7", "1b"), ("2", "190m"), ("2", "294m"), ("2", "505m")]
    for st, size in need:
        have = sorted(int(s) for s in refs["lm0"].get(f"{st}|{size}", {}))
        good = have == list(range(8))
        ok &= good
        print(f"lm0 references {st}% {size}: seeds {have} {'OK' if good else 'MISSING'}")
    if "--write" in sys.argv:
        assert ok, "not every identity holds (or something is missing): not written"
        T.REFS.write_text(json.dumps(refs, indent=1, sort_keys=True))
        print("written", T.REFS)
    print("ALL IDENTITIES HOLD" if ok else "SOME IDENTITY FAILS / INCOMPLETE")


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "table":
        table()
    else:
        {"dry": dry, "step0": step0}[m](sys.argv[2], sys.argv[3], int(sys.argv[4]))
