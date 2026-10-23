"""CPU references of the rb6_190m_ext jobs (llmadd_torch/make_jobs_rb6_190m_ext.py), the method of scale_work/refs.py
(the rb6_190m references) on the batch's own snapshot (runs/rb6_190m_ext/code/tools = runs/rb6_190m/code, asserted by
the generator) with the jobs' own ARGS:
  dry   <arm> <lr> <seed>   --dry_inputs -> OUT/dry_<arm>_lr<lr>_s<seed>.{log,json}
  step0 <arm> <lr> <seed>   the trainer's own main() up to the step-0 training forward (CPU, bf16 autocast, no_grad;
                            scale_work/step0_ng.py, the method of rb6scale_refs.json's LM0)
  table                     every result against the main batch's references (rb6scale_refs.json DRY / LM0, the 53M
                            DOMS / DSHA0, the per-domain prior / shuf / lam of the arm's intent, the target line), the
                            decoded rows of every new arm vs the main batch's dry log of the seed (only the 'domains:' line
                            may differ), -> llmadd_torch/scale_work/rb190ext_refs.json (with --write, only if all hold)
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
PIN = B / "runs" / "rb6_190m_ext" / "code" / "tools"
sys.path.insert(0, str(PIN))
sys.path.insert(0, str(B))
import make_rb6m_tasks as G   # noqa: E402
import make_rb6_tasks as P    # noqa: E402
import make_jobs_rb6_190m_ext as J   # noqa: E402

assert Path(G.__file__).resolve().parent == PIN.resolve()
J.register(G)
for f, v in J.WANT_SHA.items():
    assert J.sha(PIN / f) == v, (f, J.sha(PIN / f), v)
PY = J.PY
STEP0 = B / "scale_work" / "step0_ng.py"
ENV = dict(os.environ, HF_HOME=str(B / "hf_cache"), PYTHONPATH=str(B / "pylib"), PYTHONDONTWRITEBYTECODE="1",
           TOKENIZERS_PARALLELISM="false")
ENV.pop("SLURM_JOB_ID", None)   # run_guard.claim would record a job id
OUTD = B / "scale_work" / "rb190ext"
OUTD.mkdir(exist_ok=True)
MAIN_DRY = B / "scale_work" / "refs_out"   # the rb6_190m CPU dry logs (refs.py)
MAIN_LOG = {0: "dry_190m_all6-lam1_lr6e-4_s0.log", 1: "dry_190m_none_lr6e-4_s1.log", 2: "dry_190m_none_lr6e-4_s2.log",
            3: "dry_190m_all6-lam1_lr6e-4_s3.log"}
SIZE = "190m"


def key(arm, lr, seed):
    return f"{arm}_lr{lr}_s{seed}"


def args_list(arm, lr, seed, out):
    return J.args_of(G, SIZE, arm, lr, seed, out).split()


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
    """one directory per job (= one host): step0_job<SLURM_JOB_ID>, or step0_local_<host> outside Slurm. The first CPU job
    (20654128, cs655) wrote its records at the top level of OUTD (before this rule)."""
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


ARMKEYS = ("prior", "shuf", "lam", "target")   # the only per-domain keys that differ across arms (main-batch logs: keydiff.py)


def nondomain_lines(txt):
    """the dry log with what legitimately depends on the arm removed: the 'target features' line (numbers prior target),
    the keys prior / shuf / lam / target of every domain in the 'domains:' and '[<d>]' lines (the rest of those lines
    kept), the lines naming this run's out dir / tag. -> (lines, {domain: {key: value}} of the '[<d>]' lines)"""
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
    cells, seeds = J.BATCHES["rb6_190m_ext"]["stages"]["main"], J.BATCHES["rb6_190m_ext"]["seeds"]
    ok, X = True, {"dry": {}, "notes": []}
    for s in seeds:
        main_txt = (MAIN_DRY / MAIN_LOG[s]).read_text()
        main_ev = json.loads((MAIN_DRY / MAIN_LOG[s].replace(".log", ".json")).read_text())["eval_lines_sha"]
        for arm, lr in cells:
            k = key(arm, lr, s)
            f = OUTD / f"dry_{k}.json"
            if not f.exists():
                print(f"MISSING dry {k}")
                ok = False
                continue
            v = json.loads(f.read_text())
            act = J.ACTIVE[arm]
            want = {d: (d in act, False, 1.0 if d in act else 0.0) for d in G.DOMS6}
            got = {d: (x.get("prior"), x.get("shuf"), x.get("lam")) for d, x in v["prior"].items()}
            prior_ok = got == want and v["prior"]["numbers"].get("target") == ("helix3" if "numbers" in act else "none")
            target_ok = v["target_line"] == (["target features (1000, 10)"] if "numbers" in act else [])
            same = (v["sha"], v["n"]) == DRY[s]
            doms_ok = v["doms"] == G.DOMS[s]
            ev_ok = v["eval_lines_sha"] == main_ev
            # decoded rows side by side: the whole dry log = the main batch's dry log of the seed except the arm lines
            (a, per), (b, _) = nondomain_lines((OUTD / f"dry_{k}.log").read_text()), nondomain_lines(main_txt)
            # second path of the prior table: the '[<d>]' lines of the five domains
            per_ok = all(per.get(d, {}).get("prior") == (d in act) and per.get(d, {}).get("lam") == (1.0 if d in act else 0.0)
                         for d in P.EXT)
            prior_ok = prior_ok and per_ok
            rows_ok = a == b
            good = v["rc"] == 0 and same and prior_ok and target_ok and doms_ok and ev_ok and rows_ok and \
                v["model_line"] == [J.SIZES[SIZE]["model_line"]]
            ok &= good
            X["dry"][f"{arm}@{lr}@s{s}"] = dict(rc=v["rc"], sha=v["sha"], n=v["n"], prior_ok=prior_ok, target_ok=target_ok,
                                                doms_ok=doms_ok, eval_ok=ev_ok, rows_equal_main=rows_ok,
                                                model_line=v["model_line"], prior=got, good=good)
            print(f"s{s} {arm}@{lr}: rc {v['rc']}; DRY {v['sha'][:12]} n {v['n']} {'= main' if same else '!= main ' + str(DRY[s])}; "
                  f"domains prior/shuf/lam {'= intent' if prior_ok else 'DIFFER ' + str(got)}; target line {v['target_line']} "
                  f"{'ok' if target_ok else 'WRONG'}; DOMS {doms_ok}; eval lines = main {ev_ok}; whole log except the arm lines = "
                  f"main's {MAIN_LOG[s]}: {rows_ok} ({len(a)} lines)")
            if arm in ("only-clock", "all6-minus-clock"):
                print(f"    {arm} s{s} domains: clock {got['clock']}, numbers {got['numbers']}, months {got['months']}, "
                      f"weekdays {got['weekdays']}, s4 {got['s4']}, dim {got['dim']}")
    # step 0: CPU bf16 kernels differ across hosts at the 1e-5 - 1e-4 level (job 20654128 on cs655: every arm, also the
    # main batch's all6-lam1 / none / only-numbers, 2.5e-5 - 8.2e-5 off the stored LM0 computed on another host), so
    # arm-independence is tested WITHIN a host group (one directory = one job = one host; job 20654128's records are the
    # top level): per (group, seed) every record has the same lm bitwise; every new arm type (only-clock,
    # all6-minus-clock) has, at each of its seeds, a group where a main-batch arm type (none / all6-lam1 / only-numbers)
    # of the seed has its lm bitwise; every record within 1e-4 of the stored LM0 (the jobs' check: 1e-3 vs the GPU run)
    groups = {"job20654128 (cs655, top level)": sorted(OUTD.glob("step0_*.json"))}
    for d in sorted(p for p in OUTD.glob("step0_*") if p.is_dir()):
        groups[d.name] = sorted(d.glob("step0_*.json"))
    MAIN_ARMS = ("none", "all6-lam1", "only-numbers")
    rec, X["step0_groups"] = [], {}
    for gname, files in groups.items():
        vs = [json.loads(f.read_text()) for f in files]
        hosts = {v.get("host") for v in vs}
        X["step0_groups"][gname] = sorted(f"{v['arm']}@{v['lr']}@s{v['seed']}" for v in vs)
        for s in sorted({v["seed"] for v in vs}):
            lms = {v["lm"] for v in vs if v["seed"] == s and v["rc"] == 0}
            same = len(lms) == 1 and all(v["rc"] == 0 for v in vs if v["seed"] == s)
            ok &= same
            print(f"step0 group {gname} hosts {sorted(map(str, hosts))} s{s}: "
                  + ", ".join(f"{v['arm']}@{v['lr']} {v['lm']!r}" for v in vs if v["seed"] == s)
                  + f" -> {'one value (bitwise)' if same else 'DIFFER'}; - stored LM0 {next(iter(lms)) - LM0[s]:+.2e}")
        for v in vs:
            rec.append(dict(v, group=gname))
    for v in rec:
        good = v["rc"] == 0 and v.get("data_sha") == G.DSHA0[v["seed"]] and abs(v["lm"] - LM0[v["seed"]]) < 1e-4
        ok &= good
        if not good:
            print(f"step0 FAIL {v['group']} {v['arm']}@{v['lr']} s{v['seed']}: rc {v['rc']} data_sha {v.get('data_sha')} lm {v.get('lm')}")
    X["step0_arm"] = {}
    for arm in sorted({a for a, _ in cells}):
        mine = [v for v in rec if v["arm"] == arm]
        seeds_v = sorted({v["seed"] for v in mine})
        if arm in MAIN_ARMS:
            verified = seeds_v
        else:
            verified = sorted({v["seed"] for v in mine if any(w["group"] == v["group"] and w["seed"] == v["seed"] and
                                                             w["arm"] in MAIN_ARMS and w["lm"] == v["lm"] for w in rec)})
        good = bool(mine) and verified == seeds_v and all(abs(v["lm"] - LM0[v["seed"]]) < 1e-4 and v["data_sha"] == G.DSHA0[v["seed"]]
                                                           for v in mine)
        ok &= good
        X["step0_arm"][arm] = dict(seeds=seeds_v, seeds_matched_to_a_main_arm_same_host=verified, good=good,
                                   max_abs_minus_lm0=max(abs(v["lm"] - LM0[v["seed"]]) for v in mine) if mine else None)
        print(f"step0 arm type {arm}: seeds {seeds_v}; matched bitwise to a main-batch arm on the same host at seeds {verified}; "
              f"max |lm - stored LM0| {X['step0_arm'][arm]['max_abs_minus_lm0']:.2e} -> {'OK' if good else 'FAIL'}")
    X["all_hold"] = ok
    out = B / "scale_work" / "rb190ext_refs.json"
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
