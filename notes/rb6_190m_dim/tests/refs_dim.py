"""CPU references of the rb6_190m_dim jobs (llmadd_torch/make_jobs_rb6_190m_dim.py), the method of refs_ext.py (rb6_190m_ext)
on the batch's own snapshot (runs/rb6_190m_dim/code/tools = runs/rb6_190m/code, asserted by the generator) with the jobs'
own ARGS (make_jobs_rb6scale.args_of, the function render() uses):
  dry   <arm> <lr> <seed>   --dry_inputs -> OUT/dry_<arm>_lr<lr>_s<seed>.{log,json}
  step0 <arm> <lr> <seed>   the trainer's own main() up to the step-0 training forward (CPU, bf16 autocast, no_grad;
                            scale_work/step0_ng.py, the method of rb6scale_refs.json's LM0); one directory per job (= host)
  table [--write]           every result against the references -> llmadd_torch/scale_work/rb190dim_refs.json:
     seeds 0-3 (none@3.84e-2): DRY rows = rb6scale_refs.json's (the main batch), the whole dry log = the main batch's dry log
       of the seed except the arm lines, eval lines = the main batch's;
     seeds 4-7 (none@{6e-4, 4.8e-3, 9.6e-3, 1.92e-2, 3.84e-2}, all6-lam1@6e-4): DRY rows = the Burst 53M seed's rows
       (tasks 470 / 472 / 474 / 476) with the 190M model line (make_jobs_rb6scale.burst_dry_expected), one value over
       every cell of the seed; DOMS = make_rb6m_tasks.DOMS; the whole dry log except the arm lines and eval lines equal
       across the cells of the seed; per-domain prior / shuf / lam / target = the arm's intent;
     step 0: per host group and seed one value bitwise (arms and lrs); data_sha after step 0 = make_rb6m_tasks.DSHA0;
       the host's seed-0 none@6e-4 within 1e-4 of the stored LM0[0] (host calibration of the CPU method); seeds 4-7:
       LM0 = the group's value, none and all6-lam1 bitwise equal on the same host.
"""
import hashlib
import json
import os
import re
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
PIN = B / "runs" / "rb6_190m_dim" / "code" / "tools"
sys.path.insert(0, str(PIN))
sys.path.insert(0, str(B))
import make_rb6m_tasks as G   # noqa: E402
import make_rb6_tasks as P    # noqa: E402
import make_jobs_rb6scale as M   # noqa: E402

assert Path(G.__file__).resolve().parent == PIN.resolve()
for f, v in M.WANT_SHA.items():
    assert M.sha(PIN / f) == v, (f, M.sha(PIN / f), v)
PY = M.PY
STEP0 = B / "scale_work" / "step0_ng.py"
ENV = dict(os.environ, HF_HOME=str(B / "hf_cache"), PYTHONPATH=str(B / "pylib"), PYTHONDONTWRITEBYTECODE="1",
           TOKENIZERS_PARALLELISM="false")
ENV.pop("SLURM_JOB_ID", None)   # run_guard.claim would record a job id
OUTD = B / "scale_work" / "rb190dim"
OUTD.mkdir(exist_ok=True)
MAIN_DRY = B / "scale_work" / "refs_out"   # the rb6_190m CPU dry logs (refs.py)
MAIN_LOG = {0: "dry_190m_all6-lam1_lr6e-4_s0.log", 1: "dry_190m_none_lr6e-4_s1.log", 2: "dry_190m_none_lr6e-4_s2.log",
            3: "dry_190m_all6-lam1_lr6e-4_s3.log"}
SIZE = "190m"
NONE_GRID = ("6e-4", "4.8e-3", "9.6e-3", "1.92e-2", "3.84e-2")
BURST_DRY_NEW = {4: "470-rb6pb-53m-2pct-k2-none-s4", 5: "472-rb6pb-53m-2pct-k2-none-s5",
                 6: "474-rb6pb-53m-2pct-k2-none-s6", 7: "476-rb6pb-53m-2pct-k2-none-s7"}
DRY_CELLS = [("none", "3.84e-2", s) for s in range(4)] + [("none", lr, s) for s in range(4, 8) for lr in NONE_GRID] + \
    [("all6-lam1", "6e-4", s) for s in range(4, 8)]
ACTIVE = {"none": (), "all6-lam1": ("numbers", "months", "weekdays", "clock", "s4", "dim")}


def key(arm, lr, seed):
    return f"{arm}_lr{lr}_s{seed}"


def args_list(arm, lr, seed, out):
    return M.args_of(G, SIZE, arm, lr, seed, out).split()


def dry(arm, lr, seed):
    k = key(arm, lr, seed)
    out = OUTD / f"run_dry_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    log = OUTD / f"dry_{k}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([PY, "pretrain_range.py"] + args_list(arm, lr, seed, out) + ["--dry_inputs"], cwd=PIN,
                            env=dict(ENV, OMP_NUM_THREADS="1"), stdout=fh, stderr=subprocess.STDOUT).returncode
    txt = log.read_text()
    g = P.DRY_GREP.replace('"', '\\"')
    n = subprocess.run(["bash", "-c", f'set +o pipefail; grep -cE "{g}" "{log}"'], capture_output=True, text=True).stdout.strip()
    h = subprocess.run(["bash", "-c", f'set +o pipefail; grep -E "{g}" "{log}" | sha256sum | cut -d" " -f1'],
                       capture_output=True, text=True).stdout.strip()
    doms = {}
    for d in P.EXT:
        m = re.search(rf"^\[{d}\] (\{{.*\}})$", txt, re.M)
        doms[d] = {kk: json.loads(m.group(1))[kk] for kk in P.DOM_KEYS if kk in json.loads(m.group(1))} if m else None
    m = re.search(r"^domains: (\{.*\})$", txt, re.M)
    dl = json.loads(m.group(1)) if m else {}
    res = {"arm": arm, "lr": lr, "seed": seed, "rc": rc, "sec": round(time.time() - t0), "n": int(n or 0), "sha": h,
           "doms": doms, "prior": {d: {kk: v.get(kk) for kk in ("prior", "shuf", "lam", "target") if kk in v} for d, v in dl.items()},
           "model_line": [l for l in txt.splitlines() if l.startswith("model:")],
           "target_line": re.findall(r"^target features .*$", txt, re.M),
           "eval_lines_sha": hashlib.sha256("\n".join(l for l in txt.splitlines() if l.startswith("  eval ")).encode()).hexdigest()}
    (OUTD / f"dry_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({kk: res[kk] for kk in ("arm", "lr", "seed", "rc", "sec", "n", "sha", "model_line")}), flush=True)


def step0_dir():
    j = os.environ.get("SLURM_JOB_ID")
    d = OUTD / (f"step0_job{j}" if j else f"step0_local_{socket.gethostname().split('.')[0]}")
    d.mkdir(exist_ok=True)
    return d


def step0(arm, lr, seed, threads=2):
    k = key(arm, lr, seed)
    SD = step0_dir()
    out = SD / f"run_step0_{k}"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    log = SD / f"step0_{k}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([PY, str(STEP0), str(PIN)] + args_list(arm, lr, seed, out), cwd=PIN,
                            env=dict(ENV, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    m = re.search(r"^STEP0 (\{.*\})$", log.read_text(), re.M)
    res = {"key": k, "arm": arm, "lr": lr, "seed": seed, "rc": rc, "sec": round(time.time() - t0), "host": socket.gethostname(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "maxrss_gb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2),
           **(json.loads(m.group(1)) if m else {"error": log.read_text()[-2000:]})}
    (SD / f"step0_{k}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res)[:500], flush=True)


ARMKEYS = ("prior", "shuf", "lam", "target")


def nondomain_lines(txt):
    """the dry log without what legitimately depends on the arm / the run: the 'target features' line, the keys prior /
    shuf / lam / target of every domain in the 'domains:' and '[<d>]' lines, the lines naming this run's out dir / tag."""
    keep, per = [], {}
    for l in txt.splitlines():
        if l.startswith("target features") or "rb6tb-190m" in l or "/scale_work/" in l:
            continue
        m = re.match(r"^(domains: |\[(months|weekdays|clock|s4|dim)\] )(\{.*\})$", l)
        if m:
            j = json.loads(m.group(3))
            if m.group(1) == "domains: ":
                j = {d: {k: v for k, v in x.items() if k not in ARMKEYS} for d, x in j.items()}
            else:
                per[m.group(2)] = {k: j.get(k) for k in ARMKEYS if k in j}
                j = {k: v for k, v in j.items() if k not in ARMKEYS}
            l = m.group(1) + json.dumps(j, sort_keys=True)
        keep.append(l)
    return keep, per


def table():
    refs = json.loads((B / "rb6scale_refs.json").read_text())[SIZE]
    DRY = {int(k): tuple(v) for k, v in refs["dry"].items()}
    LM0 = {int(k): v for k, v in refs["lm0"].items()}
    assert sorted(DRY) == sorted(LM0) == [0, 1, 2, 3]
    M.BURST_DRY.update({s: v for s, v in BURST_DRY_NEW.items() if s not in M.BURST_DRY})   # in memory (expected rows)
    ok, X = True, {"dry": {}, "dry_seed": {str(s): list(DRY[s]) for s in range(4)}, "lm0": {}, "notes": []}
    per_seed = {}
    for arm, lr, s in DRY_CELLS:
        k = key(arm, lr, s)
        f = OUTD / f"dry_{k}.json"
        if not f.exists():
            print(f"MISSING dry {k}")
            ok = False
            continue
        v = json.loads(f.read_text())
        act = ACTIVE[arm]
        want = {d: (d in act, False, 1.0 if d in act else 0.0) for d in G.DOMS6}
        got = {d: (x.get("prior"), x.get("shuf"), x.get("lam")) for d, x in v["prior"].items()}
        a, per = nondomain_lines((OUTD / f"dry_{k}.log").read_text())
        per_ok = all(per.get(d, {}).get("prior") == (d in act) and per.get(d, {}).get("lam") == (1.0 if d in act else 0.0)
                     for d in P.EXT)
        prior_ok = got == want and v["prior"]["numbers"].get("target") == ("helix3" if "numbers" in act else "none") and per_ok
        target_ok = v["target_line"] == (["target features (1000, 10)"] if "numbers" in act else [])
        doms_ok = v["doms"] == G.DOMS[s]
        exp = DRY[s] if s < 4 else M.burst_dry_expected(G, P, SIZE, s)
        same = (v["sha"], v["n"]) == tuple(exp)
        if s < 4:   # the main batch's dry log of the seed
            main_txt = (MAIN_DRY / MAIN_LOG[s]).read_text()
            ev_ok = v["eval_lines_sha"] == json.loads((MAIN_DRY / MAIN_LOG[s].replace(".log", ".json")).read_text())["eval_lines_sha"]
            rows_ok = a == nondomain_lines(main_txt)[0]
        else:       # across the cells of the seed (checked below)
            per_seed.setdefault(s, []).append((f"{arm}@{lr}@s{s}", a, v["eval_lines_sha"]))
            ev_ok = rows_ok = True
        good = v["rc"] == 0 and same and prior_ok and target_ok and doms_ok and ev_ok and rows_ok and \
            v["model_line"] == [M.SIZES[SIZE]["model_line"]]
        ok &= good
        X["dry"][f"{arm}@{lr}@s{s}"] = dict(rc=v["rc"], sha=v["sha"], n=v["n"], prior_ok=prior_ok, target_ok=target_ok,
                                            doms_ok=doms_ok, eval_ok=ev_ok, rows_ok=rows_ok, model_line=v["model_line"],
                                            prior=got, good=good)
        print(f"s{s} {arm}@{lr}: rc {v['rc']}; DRY {v['sha'][:12]} n {v['n']} "
              f"{'= ' + ('main' if s < 4 else 'Burst 53M rows + 190M model line') if same else '!= expected ' + str(exp)}; "
              f"domains prior/shuf/lam {'= intent' if prior_ok else 'DIFFER ' + str(got)}; target line {v['target_line']} "
              f"{'ok' if target_ok else 'WRONG'}; DOMS {doms_ok}" + (f"; eval lines = main {ev_ok}; whole log except the arm "
                                                                    f"lines = main's {MAIN_LOG[s]}: {rows_ok}" if s < 4 else ""))
    for s, rows in sorted(per_seed.items()):
        same_rows = len({json.dumps(a) for _, a, _ in rows}) == 1
        same_ev = len({e for _, _, e in rows}) == 1
        ok &= same_rows and same_ev and len(rows) == len(NONE_GRID) + 1
        print(f"s{s}: {len(rows)} cells; whole dry log except the arm lines identical across them: {same_rows}; eval lines "
              f"identical: {same_ev}")
        shas = {(X["dry"][c]["sha"], X["dry"][c]["n"]) for c, _, _ in rows}
        ok &= len(shas) == 1
        if len(shas) == 1:
            X["dry_seed"][str(s)] = list(next(iter(shas)))
    # step 0: one value per (host group, seed) bitwise over arms / lrs; data_sha = DSHA0; host calibration on seed 0
    groups = {d.name: sorted(d.glob("step0_*.json")) for d in sorted(p for p in OUTD.glob("step0_*") if p.is_dir())}
    X["step0_groups"] = {}
    for gname, files in groups.items():
        vs = [json.loads(f.read_text()) for f in files]
        X["step0_groups"][gname] = {f"{v['arm']}@{v['lr']}@s{v['seed']}": v.get("lm") for v in vs}
        for v in vs:
            good = v["rc"] == 0 and v.get("data_sha") == G.DSHA0[v["seed"]]
            ok &= good
            if not good:
                print(f"step0 FAIL {gname} {v['arm']}@{v['lr']} s{v['seed']}: rc {v['rc']} data_sha {v.get('data_sha')} "
                      f"vs {G.DSHA0[v['seed']]}")
        cal = [v for v in vs if v["seed"] == 0 and v["rc"] == 0]
        cal_ok = bool(cal) and all(abs(v["lm"] - LM0[0]) < 1e-4 for v in cal)
        ok &= cal_ok
        print(f"step0 group {gname} hosts {sorted({str(v.get('host')) for v in vs})}: host calibration s0 "
              + ", ".join(f"{v['arm']}@{v['lr']} {v['lm']!r} (- stored LM0[0] {v['lm'] - LM0[0]:+.2e})" for v in cal)
              + f" -> {'OK (< 1e-4)' if cal_ok else 'FAIL'}")
        for s in sorted({v["seed"] for v in vs}):
            lms = {v["lm"] for v in vs if v["seed"] == s and v["rc"] == 0}
            arms = {v["arm"] for v in vs if v["seed"] == s and v["rc"] == 0}
            same = len(lms) == 1 and all(v["rc"] == 0 for v in vs if v["seed"] == s)
            ok &= same
            print(f"  s{s}: " + ", ".join(f"{v['arm']}@{v['lr']} {v['lm']!r}" for v in vs if v["seed"] == s)
                  + f" -> {'one value (bitwise)' if same else 'DIFFER'}")
            if s >= 4:
                both = arms >= {"none", "all6-lam1"}
                ok &= both
                if same and both:
                    prev = X["lm0"].get(str(s))
                    if prev is None:
                        X["lm0"][str(s)] = next(iter(lms))
                    else:
                        X["notes"].append(f"s{s}: a second host group {gname} gives {next(iter(lms))!r} (first {prev!r}; "
                                          f"diff {next(iter(lms)) - prev:+.2e})")
                        ok &= abs(next(iter(lms)) - prev) < 1e-4
    for s in range(4, 8):
        ok &= str(s) in X["lm0"] and str(s) in X["dry_seed"]
    print("LM0 seeds 4-7: " + json.dumps(X["lm0"]) + "; DRY seeds 4-7: "
          + json.dumps({s: X["dry_seed"].get(str(s)) for s in range(4, 8)}))
    for n_ in X["notes"]:
        print("NOTE " + n_)
    X["all_hold"] = ok
    out = B / "scale_work" / "rb190dim_refs.json"
    if "--write" in sys.argv:
        assert ok, "not every identity holds: not written"
        out.write_text(json.dumps(X, indent=1, sort_keys=True))
        print("written", out)
    print("ALL IDENTITIES HOLD" if ok else "SOME IDENTITY FAILS")


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "table":
        table()
    else:
        {"dry": dry, "step0": step0}[m](sys.argv[2], sys.argv[3], int(sys.argv[4]))
