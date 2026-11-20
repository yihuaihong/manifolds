"""pt6v2 test 3 checks (runs/pt6v2/AUDIT_phase2b.md "Test 3"): a tiny CPU training run (d64 x 3) of the PLACEHOLDER composition
through the Rev H trainer (runs/pt6v2/code) with the jobs' own command line, four arms: all6, none, only-compass, only-hcube.
    python tiny_check_pt6v2.py <root>      (<root>/out/<arm>/<tag>/run_whole_*_seed16_*.json and <root>/<arm>.log)
T3.1 every arm: rc 0, 'done in' printed, the Rev H step-0 toy3 line printed (hcube's true-prior gradient > 0 is asserted there)
T3.2 the same data in every arm: data_sha equal at every eval; step-0 lm equal (|d| <= 1e-4) across the four arms
T3.3 each arm's ACTIVE priors on exactly in the window (w0 * steps <= s < w1 * steps), every other prior off at every eval
T3.4 all6: at every on-window eval the six priors' gradient fields (gnorm_prior_<d>, gcos_lm|prior_<d>) and the LM-share fields
     (grad_lm_share < 1, grad_T, clip_coef) present; only-m: m's present there
T3.5 no field for an inactive prior: no gnorm_prior_<d> / gcos_lm|prior_<d> / geo_<d> for a d outside the arm's active set, at any eval
Prints 'TEST3 PASS n/n' or 'TEST3 FAIL k/n'."""
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(sys.argv[1])
SIX = ("numbers", "dim", "cities", "countries", "compass", "hcube")
ACTIVE = {"all6": SIX, "none": (), "only-compass": ("compass",), "only-hcube": ("hcube",)}
res = []


def ok(name, cond, info=""):
    res.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'} {name}  {info}", flush=True)


runs, logs = {}, {}
for a in ACTIVE:
    js = sorted((ROOT / "out" / a).glob("*/run_whole_*_seed16_*.json"))
    lg = (ROOT / f"{a}.log").read_text() if (ROOT / f"{a}.log").exists() else ""
    logs[a] = lg
    runs[a] = json.loads(js[0].read_text()) if len(js) == 1 else None
    ok(f"T3.1 {a}: rc 0, 'done in', the step-0 toy3 line",
       runs[a] is not None and re.search(r"^run rc 0$", lg, re.M) and "done in " in lg
       and "step 0: toy3 true-prior gradient norms" in lg,
       {"json": len(js), "rc": re.findall(r"^run rc (\d+)$", lg, re.M), "errors": re.findall(r"^(\w*Error.*)$", lg, re.M)[:2]})
if not all(runs.values()):
    print(f"TEST3 FAIL {sum(res)}/{len(res)} (a run is missing)")
    sys.exit(1)
C = {a: r["curve"] for a, r in runs.items()}
steps = runs["all6"]["steps"]
assert all(r["steps"] == steps for r in runs.values())
w0, w1 = (float(v) for v in runs["all6"]["args"]["window"].split(","))
inwin = lambda s: w0 * steps <= s < w1 * steps
st = [e["step"] for e in C["all6"]]
ok("T3.2 the same evals in every arm", all([e["step"] for e in c] == st for c in C.values()), {"steps": steps, "evals": len(st)})
ok("T3.2 data_sha equal at every eval across the four arms",
   all([e["data_sha"] for e in c] == [e["data_sha"] for e in C["all6"]] for c in C.values()),
   {a: C[a][0]["data_sha"] for a in C})
lm0 = {a: C[a][0]["lm"] for a in C}
ok("T3.2 step-0 lm equal across the four arms (1e-4)", max(lm0.values()) - min(lm0.values()) <= 1e-4, lm0)
n_on = sum(inwin(s) for s in st)
ok("T3.3 the window lies inside the run (>= 3 on-window and >= 3 off evals)", n_on >= 3 and len(st) - n_on >= 3,
   {"window": [w0, w1], "on evals": n_on, "of": len(st)})
for a, c in C.items():
    bad = [(e["step"], d, e["prior_on_domains"].get(d)) for e in c for d in SIX
           if e["prior_on_domains"].get(d) != int(d in ACTIVE[a] and inwin(e["step"]))]
    ok(f"T3.3 {a}: active {list(ACTIVE[a]) or 'none'} on exactly in the window, the rest off at every eval", not bad,
       {"mismatches": bad[:4], "keys": sorted(c[0]["prior_on_domains"])})
for a, c in C.items():
    if not ACTIVE[a]:
        continue
    miss = []
    for e in c:
        if not inwin(e["step"]):
            continue
        for d in ACTIVE[a]:
            for k in (f"gnorm_prior_{d}", f"gcos_lm|prior_{d}"):
                if e.get(k) is None:
                    miss.append((e["step"], k))
        for k in ("grad_T", "clip_coef"):
            if e.get(k) is None:
                miss.append((e["step"], k))
        if not (e.get("grad_lm_share") is not None and e["grad_lm_share"] < 1):
            miss.append((e["step"], "grad_lm_share<1", e.get("grad_lm_share")))
    on_ = [e for e in c if inwin(e["step"])]
    ok(f"T3.4 {a}: the gradient / LM-share fields of {len(ACTIVE[a])} prior(s) at every on-window eval", not miss,
       {"missing": miss[:6], "grad_lm_share": [round(e.get("grad_lm_share") or -1, 3) for e in on_][:8]})
for a, c in C.items():
    extra = sorted({k for e in c for k in e for d in SIX if d not in ACTIVE[a]
                    and k in (f"gnorm_prior_{d}", f"gcos_lm|prior_{d}", f"geo_{d}")})
    ok(f"T3.5 {a}: no prior field for an inactive prior", not extra, {"found": extra[:6]})
print(f"TEST3 {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
sys.exit(0 if all(res) else 1)
