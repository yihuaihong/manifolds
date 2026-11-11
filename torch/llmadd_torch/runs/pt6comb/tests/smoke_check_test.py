"""pt6comb test: smoke/smoke_check.py (the GPU smoke's pass rules) on smoke-shaped run dirs built from the REAL tiny CPU runs
(tests/feas all6 / none): 33 evals at steps 0..8000; on-window evals (1000..7250) copy the tiny run's on-evals, the others its
off-evals; prior_on / prior_on_domains / prior_used_frac / gnorm keys re-set to the real window; step-0 lm = the CPU reference;
check.json with exactly the truncation failures. Positive control -> SMOKE PASS (SM6 informative in --test mode); each negative control
fails its rule. -> smoke_check_test.out"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0]]
import readout_pt6comb as RP   # noqa: E402
M6 = RP.M6
F = HERE / "tests" / "feas" / "out"
STEPS = list(range(0, 8001, 250))
REF = M6.refs6("B")[0]
res = []


def build(r, arm):
    r = copy.deepcopy(r)
    c = r["curve"]
    on = [e for e in c if e.get("prior_on")] or c
    off = [e for e in c if not e.get("prior_on")]
    pri = M6.pri_of(arm, "B")
    new = []
    for i, s in enumerate(STEPS):
        inw = RP.WIN[0] <= s <= RP.WIN[1] and bool(pri)
        e = copy.deepcopy(on[i % len(on)] if inw else off[i % len(off)])
        e["step"] = s
        e["data_sha"] = c[i % len(c)]["data_sha"]
        cn_ = base["none"]["curve"]   # the same LM values in both bodies before the window (identical data and init)
        e["lm"] = REF["lm0"] if s == 0 else cn_[i % len(cn_)]["lm"]
        e["prior_on"] = int(inw)
        e["prior_on_domains"] = {d: int(inw and d in pri) for d in RP.MANI}
        n_app = max(0, min(s, RP.WIN[1]) - RP.WIN[0] + 1) if pri else 0
        e["prior_on_frac"] = n_app / (s + 1)
        for d in RP.MANI:
            e[f"prior_used_frac_{d}"] = (n_app / (s + 1)) if d in pri else 0.0
        if not inw:
            for k in [k for k in e if k.startswith("gnorm_prior_")]:
                del e[k]
        e["sec"] = 0.05 * s
        new.append(e)
    r["curve"] = new
    m = r["metrics"]
    import numpy as np
    m["numbers"]["area"] = 100 * float(np.mean([np.mean([e[f"held_{b}"] for b in RP.BKT]) for e in new]))
    for d in RP.CMPD:
        m[f"{d}_readout"]["area"] = 100 * float(np.mean([e[f"{d}_held_2way"] for e in new]))
    tag = M6.tag6("B", arm, "2e-3")
    r["args"].update(seed=0, lr=2e-3, tag=f"_{tag}")
    return r


def put(root, arm, r, extra_fail=()):
    name = M6.name6("B", arm, "2e-3", 0, smoke=True)
    tag = M6.tag6("B", arm, "2e-3")
    d = Path(root) / name / tag
    d.mkdir(parents=True)
    tgt = "helix3" if arm == "all6" else "none"
    (d / f"run_whole_{tgt}_seed0_{tag}.json").write_text(json.dumps(r))
    fails = ["finished: 75 evals, increasing, the last at step 18309"] + (["prior_on_frac == 0.349972"] if arm == "all6" else []) + list(extra_fail)
    ck = {"checks": [{"check": f, "ok": False} for f in fails] + [{"check": "run args = intent (47 fields)", "ok": True}]}
    (d.parent / "check.json").write_text(json.dumps(ck))
    return name


def run(root, names):
    o = subprocess.run([sys.executable, str(HERE / "smoke" / "smoke_check.py"), *names, "--root", str(root), "--test"], capture_output=True, text=True,
                       env={"PYTHONPATH": "/scratch/yh6210/llmadd_torch/pylib", "PATH": "/usr/bin:/bin"})
    return o.stdout + o.stderr


base = {a: json.loads(next((F / a).glob("run_*.json")).read_text()) for a in ("all6", "none")}
for a in base:   # the tiny runs' cka values at their on-evals vs off-evals: make the all6 fit visible as in a real run (+0.3)
    pass


def case(desc, mut_a=None, mut_n=None, extra=(), want_fail=None):
    with tempfile.TemporaryDirectory() as root:
        ra, rn = build(base["all6"], "all6"), build(base["none"], "none")
        for e in ra["curve"]:   # a fitted prior in the window (the tiny run fits weakly): CKA well above none's; the 53M-like LM share
            if e["prior_on"]:     # 0.8 (the tiny d64 x 3 model's own share is ~0.06: its prior dominates, feas_check.out)
                e["grad_prior"] = 0.25 * e["grad_lm"]
                e["grad_lm_share"] = e["grad_lm"] / (e["grad_lm"] ** 2 + e["grad_prior"] ** 2 + 2 * e["grad_lm"] * e["grad_prior"] * e["grad_cos"]) ** 0.5
                for d in RP.MANI:
                    k = "cka_ops_helix3" if d == "numbers" else f"cka_{d}"
                    e[k] = min(1.0, rn["curve"][ra["curve"].index(e)][k] + 0.3)
        if mut_a:
            mut_a(ra)
        if mut_n:
            mut_n(rn)
        na, nn = put(root, "all6", ra, extra), put(root, "none", rn)
        out = run(root, [na, nn])
    last = out.strip().splitlines()[-1] if out.strip() else out
    fails = [l for l in out.splitlines() if l.startswith("FAIL")]
    good = (last.startswith("SMOKE PASS") if want_fail is None else (last.startswith("SMOKE FAIL") and any(l.startswith("FAIL " + want_fail) for l in fails)))
    res.append(good)
    print(f"{'PASS' if good else 'FAIL'} {desc}: {last}" + (f"  failing: {[l[:70] for l in fails]}" if fails else ""))
    if not good:
        print(out[-2500:])


case("positive control (the real tiny runs shaped as the smoke)")
case("s4 prior still on at step 7500 (after the window) -> SM2 s4", mut_a=lambda r: r["curve"][30]["prior_on_domains"].update(s4=1), want_fail="SM2 s4")
case("cities' CKA = none's at the window's last eval -> SM3 cities",
     mut_a=lambda r: next(e for e in r["curve"] if e["step"] == 7250).update(cka_cities=0.0), want_fail="SM3 cities")
case("LM share 0.3 at every on-eval -> SM4",
     mut_a=lambda r: [e.update(grad_prior=e["grad_lm"] * 7 / 3) for e in r["curve"] if e["prior_on"]], want_fail="SM4")
case("data differ between the bodies at one eval -> SM5", mut_n=lambda r: r["curve"][9].update(data_sha="x"), want_fail="SM5")
case("an extra failed in-job check -> SM1", extra=("data_sha after step 0 = the CPU reference",), want_fail="SM1 all6")
case("a numbers metric that differs from its series -> SM7", mut_a=lambda r: r["metrics"]["numbers"].update(area=r["metrics"]["numbers"]["area"] + 1),
     want_fail="SM7 all6")
print(f"SMOKE CHECK TEST {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
