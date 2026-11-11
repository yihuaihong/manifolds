"""pt6comb test: readout_pt6comb.load / integrity / run_metrics on REAL run JSONs of the tiny CPU smoke (tests/feas: all6, none,
only-numbers, only-s4), stretched to the 75-eval schedule (each eval copies a tiny eval of the same prior state: on-window evals
from the tiny run's on-evals, the others from its off-evals; the metrics the trainer would write are recomputed from the stretched
curve) and laid out as pt6comb run dirs with a check.json; then the validity rules and the double paths are exercised by tampers.
-> loader_test.out"""
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0]]
import readout_pt6comb as RP   # noqa: E402
M6 = RP.M6
F = HERE / "tests" / "feas" / "out"
STEPS = list(range(0, 18310, 250)) + [18309]
res = []


def ok(name, cond, val=""):
    res.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val != "" else ""))


def stretch(r, arm, lr, seed):
    r = copy.deepcopy(r)
    c = r["curve"]
    on = [e for e in c if e.get("prior_on")] or c
    off = [e for e in c if not e.get("prior_on")]
    new = []
    for i, s in enumerate(STEPS):
        inw = RP.WIN[0] <= s <= RP.WIN[1] and arm != "none"
        src = on[i % len(on)] if inw else off[i % len(off)]
        e = copy.deepcopy(src)
        e["step"] = s
        e["data_sha"] = c[i % len(c)]["data_sha"]   # the data stream does not depend on the arm (identical tiny series, feas_check)
        new.append(e)
    r["curve"] = new
    m = r["metrics"]
    m["numbers"]["area"] = 100 * float(np.mean([np.mean([e[f"held_{b}"] for b in RP.BKT]) for e in new]))
    for d in RP.CMPD:
        m[f"{d}_readout"]["area"] = 100 * float(np.mean([e[f"{d}_held_2way"] for e in new]))
        if f"{d}_readout_hard" in m:
            m[f"{d}_readout_hard"]["area"] = 100 * float(np.mean([e[f"{d}_held_2way_hard"] for e in new]))
    for d, mk in (("s4", "fresh_nonid"), ("dim", "no_negtwin_nonid")):
        m[f"{d}_readout"]["area"] = RP.hits_area(r, d, mk)
    tag = M6.tag6("B", arm, lr)
    r["args"].update(seed=seed, lr=float(lr), tag=f"_{tag}")
    return r


def checks(arm, fail=()):
    pri = M6.pri_of(arm, "B")
    names = list(RP.KEEP6) + ([M6.NUM_ON_CHECK] if "numbers" in pri else [RP.NUM_OFF])
    if pri:
        names.append(RP.GEO_FLAG6 + " (before a divergence)")
    names += ["run args = intent", "item-state CKA to the target rises by >= 0.1", "2-way scoring double path (GPU-numerics tolerance, A5.1): x",
              "data_sha after step 0 = the CPU reference"]
    return {"pass": not fail, "checks": [{"check": n + " (...)", "ok": not any(n.startswith(f) for f in fail)} for n in names]}


def put(root, arm, lr, seed, r, ck, rr=0):
    name = M6.name6("B", arm, lr, seed, rr)
    d = Path(root) / name
    tag = M6.tag6("B", arm, lr)
    (d / tag).mkdir(parents=True, exist_ok=True)
    (d / tag / RP.run_json(d, arm, lr, seed).name).write_text(json.dumps(r))
    (d / "check.json").write_text(json.dumps(ck))


base = {a: json.loads(next((F / a).glob("run_*.json")).read_text()) for a in ("all6", "none", "only-numbers", "only-s4")}
with tempfile.TemporaryDirectory() as root:
    for a in base:
        put(root, a, "2e-3", 0, stretch(base[a], a, "2e-3", 0), checks(a))
    R_ = RP.load(root)
    ok("L1 the four real (stretched) runs load VALID, finished, with every manifold's area / final3 / FLOOR line and no double-path problem",
       all(R_[a, "2e-3", 0]["valid"] and R_[a, "2e-3", 0]["finished"] and set(R_[a, "2e-3", 0]["area"]) == set(RP.MANI)
           and not R_[a, "2e-3", 0]["problems"] for a in base), {a: (R_[a, "2e-3", 0]["why"], R_[a, "2e-3", 0]["problems"][:2]) for a in base})
    ok("L2 integrity: one data stream at seed 0 over the four arms, no problem", RP.integrity(R_) == [], RP.integrity(R_)[:2])
    r0 = R_["all6", "2e-3", 0]
    print("     all6 areas", {m: round(v, 2) for m, v in r0["area"].items()}, "floor", {m: round(v, 1) for m, v in r0["floor"].items()},
          "mean6", round(RP.mean6(r0), 3), "s4_hard n", r0["sec"]["s4_hard_n"], "dim strict n", r0["sec"]["dim_strict_n"])
with tempfile.TemporaryDirectory() as root:
    t = stretch(base["all6"], "all6", "2e-3", 0); t["args"]["lr"] = 1e-3
    put(root, "all6", "2e-3", 0, t, checks("all6"))
    t = stretch(base["none"], "none", "2e-3", 0)
    put(root, "none", "4e-3", 0, t, checks("none"))   # the run says lr 2e-3 / tag ...-lr2e-3 inside the lr4e-3 dir
    R_ = RP.load(root)
    ok("L3 identity: a run whose args differ from its directory name is INVALID", not R_["all6", "2e-3", 0]["valid"] and "identity" in R_["all6", "2e-3", 0]["why"]
       and not R_["none", "4e-3", 0]["valid"], [R_[k]["why"][:80] for k in R_])
with tempfile.TemporaryDirectory() as root:
    put(root, "all6", "2e-3", 0, stretch(base["all6"], "all6", "2e-3", 0), checks("all6", fail=("data_sha after step 0",)))
    put(root, "none", "2e-3", 0, stretch(base["none"], "none", "2e-3", 0), checks("none", fail=("item-state CKA",)))
    put(root, "only-s4", "2e-3", 0, stretch(base["only-s4"], "only-s4", "2e-3", 0), checks("only-s4", fail=("2-way scoring double path",)))
    t = stretch(base["only-numbers"], "only-numbers", "2e-3", 0); t["curve"] = t["curve"][:74]
    put(root, "only-numbers", "2e-3", 0, t, checks("only-numbers"))
    R_ = RP.load(root)
    ok("L4 a failed non-flag check -> INVALID", not R_["all6", "2e-3", 0]["valid"] and "failed check" in R_["all6", "2e-3", 0]["why"])
    ok("L5 only the CKA-rise bar failed -> VALID (a weak prior, data)", R_["none", "2e-3", 0]["valid"] and "CKA-rise" in R_["none", "2e-3", 0]["why"])
    ok("L6 only a GPU-numerics tolerance failed -> VALID, FLAGGED A5.1", R_["only-s4", "2e-3", 0]["valid"] and R_["only-s4", "2e-3", 0]["dp_flag"])
    ok("L7 74 evals -> INVALID (unfinished)", not R_["only-numbers", "2e-3", 0]["valid"] and "unfinished" in R_["only-numbers", "2e-3", 0]["why"])
with tempfile.TemporaryDirectory() as root:
    t = stretch(base["all6"], "all6", "2e-3", 0)
    i = next(i for i, e in enumerate(t["curve"]) if e.get("prior_on"))
    t["curve"][i]["geo_s4"], t["curve"][i]["cka_s4"] = -0.00115, 1.0002
    put(root, "all6", "2e-3", 0, t, checks("all6", fail=(RP.GEO_FLAG6,)))
    t2 = copy.deepcopy(t); t2["curve"][i]["cka_s4"] = 0.9
    t2["args"]["lr"] = 4e-3; t2["args"]["tag"] = f"_{M6.tag6('B', 'all6', '4e-3')}"
    put(root, "all6", "4e-3", 0, t2, checks("all6", fail=(RP.GEO_FLAG6,)))
    R_ = RP.load(root)
    ok("L8 A6 generalised: an s4 overshoot -0.00115 at cka_s4 1.0002 in an all6 run -> VALID, FLAGGED A6", R_["all6", "2e-3", 0]["valid"]
       and R_["all6", "2e-3", 0]["geo_flag"], R_["all6", "2e-3", 0]["why"][:120])
    ok("L9 the same overshoot at cka_s4 0.9 -> INVALID (A6 does not apply)", not R_["all6", "4e-3", 0]["valid"] and "A6 does not apply" in R_["all6", "4e-3", 0]["why"])
with tempfile.TemporaryDirectory() as root:
    t = stretch(base["all6"], "all6", "2e-3", 0); t["metrics"]["numbers"]["area"] += 1.0
    put(root, "all6", "2e-3", 0, t, checks("all6"))
    t = stretch(base["none"], "none", "2e-3", 0)
    h = t["curve"][7]["hits2_cities_held"]; t["curve"][7]["hits2_cities_held"] = ("1" if h[0] == "0" else "0") + h[1:]
    put(root, "none", "2e-3", 0, t, checks("none"))
    t = stretch(base["only-s4"], "only-s4", "2e-3", 0); t["metrics"]["s4_readout"]["area"] += 0.5
    put(root, "only-s4", "2e-3", 0, t, checks("only-s4"))
    R_ = RP.load(root)
    pr = RP.integrity(R_)
    ok("L10 double paths: a numbers metric off by 1 point, a flipped cities hit, an s4_readout metric off -> integrity problems (strict refuses)",
       any("numbers area" in p for p in pr) and any("cities sub-read-out" in p for p in pr) and any("s4_readout area" in p for p in pr), pr[:4])
with tempfile.TemporaryDirectory() as root:
    put(root, "none", "2e-3", 0, stretch(base["none"], "none", "2e-3", 0), checks("none"))
    put(root, "none", "2e-3", 0, stretch(base["none"], "none", "2e-3", 0), checks("none"), rr=1)
    t = stretch(base["only-s4"], "only-s4", "2e-3", 0); t["curve"][5]["data_sha"] = "different"
    put(root, "only-s4", "2e-3", 0, t, checks("only-s4"))
    R_ = RP.load(root)
    pr = RP.integrity(R_)
    ok("L11 a finished run in an earlier dir under a re-run dir, and data differing between arms of a seed -> integrity problems",
       any("a finished run in" in p for p in pr) and any("data_sha series differ" in p for p in pr) and R_["none", "2e-3", 0]["rerun"] == 1, pr[:3])
print(f"LOADER TEST {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
