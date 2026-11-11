"""runs/newmani POST HOC positive-control diagnostics (2026-10-25; results audit 2d P1 / P2; AUDIT.md "Diagnostics (post
hoc)"). Reads out_diag/*/run_*.json; writes readout_diag.md / .json. Its only question is "pipeline vs configuration" for
2i and s5: it changes no F94 verdict (2i / a5 / s5 stay NOT READ under the pre-registered reading).

    python readout_diag.py [--root DIR]     # DIR default runs/newmani/out_diag; outputs next to this file (or DIR/..)
    python readout_diag.py --selftest

P1: 2i and s5, frac 0.4, class split, seeds 0, 1, 2 (the Stage-1 positive-control splits), none lr 3e-3 wd 3 and lr 1e-2
    wd 1, 200k steps. groks = t_grok within the run (held-out >= 0.95 at any eval). Per task: a config with >= 2/3 groks
    (all 3 seeds present) -> "pipeline OK (post hoc): none learns <task> at f 0.4 with lr X wd Y"; both configs complete
    and <= 1/3 -> "not shown"; else INCOMPLETE.
P2 trigger (the coordinator decides): both 2i P1 configs complete with 0/3 groks.
P2: 2i, frac 0.4, --split question, lr 1e-2 wd 1, seeds 0, 1, 2, 200k. In that split ~97-98 % of the held-out cells have
    a trained twin in their orbit (review 2026-10-25, seeds 0-2), so held-out accuracy is close to a twin-lookup score:
    >= 2/3 -> "none completes the 2I table when the twins are trained (what the orbit split removes is what none needs);
    this does NOT show that none learns 2I's group law at f 0.4"; 0/3 -> "none does not learn the 2I table at f 0.4 even
    with ~98 % of the held-out cells' twins trained: the limit is this model family / budget / config, not the split";
    1/3 -> not shown; missing seeds -> INCOMPLETE.
Checks: one code_sha = the pin, one device; P1 split_sha = preflight/seed_scan.json's <task>|pos split of that seed."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARGS = sys.argv[1:]
assert set(a for a in ARGS if a.startswith("--")) <= {"--root", "--selftest"}, ARGS
PIN = "ea797e5321d3"
SEEDS = (0, 1, 2)
P1 = {(t, lr, wd) for t in ("2i", "s5") for lr, wd in ((3e-3, 3.0), (1e-2, 1.0))}
P2 = ("2i", 1e-2, 1.0)
STEPS = 200000
NEED = math.ceil(2 * len(SEEDS) / 3)


def read_diag(root: Path, steps: int = STEPS, pin: str | None = PIN, scan: dict | None = None) -> tuple[str, dict]:
    """scan = seed_scan.json (the P1 splits); None = selftest only."""
    rs = [json.loads(p.read_text()) for p in sorted(root.glob("*/run_*.json"))]
    ref_sha = None if scan is None else {(t, x["seed"]): x["split_sha"] for t in ("2i", "s5") for x in scan["cells"][f"{t}|pos"]["seeds"]}
    L = [f"# newmani positive-control diagnostics, POST HOC ({root})\n\n"]
    J = {"root": str(root), "n_runs": len(rs), "P1": {}, "P2": None}
    bad = [f"{r.get('group')} {r['arm']} lr {r['lr']:g} wd {r['wd']:g} seed {r['seed']}" for r in rs
           if r["arm"] != "none" or r.get("task") != "grp" or r.get("frac") != 0.4 or r["seed"] not in SEEDS or r["steps"] != steps
           or (pin is not None and r.get("code_sha") != pin)
           or (ref_sha is not None and r.get("split") == "class" and r.get("split_sha") != ref_sha.get((r.get("group"), r["seed"])))
           or not (((r.get("group"), r["lr"], r["wd"]) in P1 and r.get("split") == "class")
                   or ((r.get("group"), r["lr"], r["wd"]) == P2 and r.get("split") == "question"))]
    if bad or len({r.get("code_sha") for r in rs}) > 1 or len({r.get("device") for r in rs}) > 1:
        J["verdict"] = "NO VERDICT"
        return "".join(L) + f"NO VERDICT: unexpected runs {bad[:10]}\n", J
    cells = {}
    for r in rs:
        k = (r["group"], r["lr"], r["wd"], r["split"])
        assert r["seed"] not in cells.setdefault(k, {}), ("duplicate", k, r["seed"])
        cells[k][r["seed"]] = r
    st = {}
    L.append("| task | split | lr | wd | seeds | groks | t_grok per seed | final test median | final train median |\n|---|---|---|---|---|---|---|---|---|\n")
    for (t, lr, wd, sp) in [k + ("class",) for k in sorted(P1)] + [P2 + ("question",)]:
        d = cells.get((t, lr, wd, sp), {})
        g = sum(r["t_grok"] is not None for r in d.values())
        st[(t, lr, wd, sp)] = s = {"n": len(d), "grok": g, "complete": set(d) == set(SEEDS)}
        if d:
            fm = sorted(r["final_test"] for r in d.values())[len(d) // 2]
            ftr = sorted(r["final_train"] for r in d.values())[len(d) // 2]
            L.append(f"| {t} | {sp} | {lr:g} | {wd:g} | {sorted(d)} | {g}/{len(d)} | "
                     + ", ".join(f"s{x}: {d[x]['t_grok']}" for x in sorted(d)) + f" | {fm:.3f} | {ftr:.3f} |\n")
        elif sp == "class":
            L.append(f"| {t} | {sp} | {lr:g} | {wd:g} | none | - | - | - | - |\n")
    L.append("\n")
    for t in ("2i", "s5"):
        cs = {(lr, wd): st[(t, lr, wd, "class")] for (tt, lr, wd) in sorted(P1) if tt == t}
        okc = [k for k, s in cs.items() if s["complete"] and s["grok"] >= NEED]
        if okc:
            v = "pipeline OK (post hoc): none learns " + t + " at f 0.4 with " + ", ".join(f"lr {k[0]:g} wd {k[1]:g}" for k in okc)
        elif all(s["complete"] for s in cs.values()):
            v = "not shown (" + ", ".join(f"lr {k[0]:g} wd {k[1]:g} {s['grok']}/3" for k, s in cs.items()) + "): pipeline vs configuration undecided"
        else:
            v = "INCOMPLETE (" + ", ".join(f"lr {k[0]:g} wd {k[1]:g} {s['n']}/3 seeds" for k, s in cs.items()) + ")"
        J["P1"][t] = v
        L.append(f"**P1 {t}**: {v}\n\n")
    c2 = [st[("2i", lr, wd, "class")] for (tt, lr, wd) in sorted(P1) if tt == "2i"]
    trig = all(s["complete"] and s["grok"] == 0 for s in c2)
    J["P2_trigger"] = "met" if trig else "pending" if not all(s["complete"] for s in c2) else "not met"
    L.append(f"**P2 trigger** (both 2i P1 configs complete with 0/3): {J['P2_trigger']}" + (" -- the coordinator decides whether to submit "
             "diag-q-2i-j0 (make_jobs_newmani.py --diag_q --submit --only diag-q-2i-j0 --nice 500)" if trig else "") + "\n\n")
    s = st[P2 + ("question",)]
    if s["n"]:
        v = ("INCOMPLETE" if not s["complete"] else
             "none completes the 2I table at f 0.4 when the twins are trained (question split, ~98 % of held-out cells with a trained "
             "twin; lr 1e-2 wd 1, 200k): what the orbit split removes is what none needs; this does NOT show that none learns 2I's "
             "group law" if s["grok"] >= NEED else
             "none does not learn the 2I table at f 0.4 even with ~98 % of the held-out cells' twins trained (question split; lr 1e-2 "
             "wd 1, 200k): the limit is this model family / budget / config, not the split" if s["grok"] == 0 else "not shown (1/3)")
        J["P2"] = v
        L.append(f"**P2 2i question split**: {v}" + ("" if trig else " (NOTE: run although the trigger is " + J["P2_trigger"] + ")") + "\n\n")
    L.append("Post hoc, for 'pipeline vs configuration' only: no F94 verdict changes (2i / a5 / s5 stay NOT READ under the "
             "pre-registered reading); in the question split ~97-98 % of held-out cells have a trained twin, so its held-out score is "
             "close to a twin-lookup score: never read it as '2I is learnable' or 'the Stage-1 wall is a split artefact'.\n")
    J["verdict"] = "read"
    return "".join(L), J


def selftest():
    import shutil
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="nm_diag_selftest_")) / "out"

    def w(t, lr, wd, s, tg, split="class", over=None):
        d = root / f"{t}-{lr}-{wd}-{split}"; d.mkdir(parents=True, exist_ok=True)
        r = {"task": "grp", "group": t, "arm": "none", "lr": lr, "wd": wd, "seed": s, "frac": 0.4, "steps": 2000, "split": split,
             "t_grok": tg, "final_test": 1.0 if tg else 0.001, "final_train": 1.0, "code_sha": "syn", **(over or {})}
        (d / f"run_none_seed{s}.json").write_text(json.dumps(r))

    def case(spec, p2=None):
        shutil.rmtree(root, ignore_errors=True)
        for (t, lr, wd), tgs in spec.items():
            for s, tg in zip(SEEDS, tgs):
                if tg != "missing":
                    w(t, lr, wd, s, tg)
        for s, tg in zip(SEEDS, p2 or ()):
            w("2i", 1e-2, 1.0, s, tg, split="question")
        return read_diag(root, steps=2000, pin=None)

    no = (None, None, None)
    t, J = case({("2i", 3e-3, 3.0): no, ("2i", 1e-2, 1.0): no, ("s5", 3e-3, 3.0): (500, None, 900), ("s5", 1e-2, 1.0): no})
    assert J["P1"]["s5"].startswith("pipeline OK") and "lr 0.003 wd 3" in J["P1"]["s5"] and J["P1"]["2i"].startswith("not shown")
    assert J["P2_trigger"] == "met" and J["P2"] is None and "diag-q-2i-j0" in t
    print(f"selftest 1: s5 {J['P1']['s5']}; 2i {J['P1']['2i'][:40]}...; P2 trigger {J['P2_trigger']}")
    t, J = case({("2i", 3e-3, 3.0): (None, 800, None), ("2i", 1e-2, 1.0): no, ("s5", 3e-3, 3.0): no, ("s5", 1e-2, 1.0): (100, 200, "missing")})
    assert J["P2_trigger"] == "not met" and J["P1"]["s5"].startswith("INCOMPLETE") and J["P1"]["2i"].startswith("not shown")
    print(f"selftest 2: 2i 1/3 -> trigger {J['P2_trigger']}; s5 2/2 seeds -> {J['P1']['s5'][:30]}")
    for p2, want in (((300, 400, None), "none completes the 2I table"), (no, "none does not learn the 2I table"), ((300, None, None), "not shown"),
                     ((300, 400), "INCOMPLETE")):
        t, J = case({k: no for k in P1}, p2=p2)
        assert J["P2"].startswith(want), (p2, J["P2"])
        print(f"selftest 3 P2 {p2}: {J['P2'][:60]}")
    shutil.rmtree(root, ignore_errors=True)
    w("2i", 3e-3, 3.0, 0, None, over={"frac": 0.2})
    t, J = read_diag(root, steps=2000, pin=None)
    assert J["verdict"] == "NO VERDICT"
    print("selftest 4: a run at another frac -> NO VERDICT")
    shutil.rmtree(root, ignore_errors=True)
    w("2i", 3e-3, 3.0, 0, None, over={"split_sha": "wrong"})
    t, J = read_diag(root, steps=2000, pin=None, scan={"cells": {f"{x}|pos": {"seeds": [{"seed": s, "split_sha": "right"} for s in SEEDS]}
                                                                for x in ("2i", "s5")}})
    assert J["verdict"] == "NO VERDICT"
    w("2i", 3e-3, 3.0, 1, None, over={"split_sha": "right", "device": "cuda"})
    (root / "2i-0.003-3.0-class" / "run_none_seed0.json").unlink()
    w("2i", 3e-3, 3.0, 0, None, over={"split_sha": "right", "device": "cpu"})
    t, J = read_diag(root, steps=2000, pin=None, scan={"cells": {f"{x}|pos": {"seeds": [{"seed": s, "split_sha": "right"} for s in SEEDS]}
                                                                for x in ("2i", "s5")}})
    assert J["verdict"] == "NO VERDICT"
    print("selftest 5: a P1 split_sha != seed_scan's, or two devices -> NO VERDICT")
    shutil.rmtree(root.parent, ignore_errors=True)
    print("DIAG READOUT SELFTEST PASSED")


if __name__ == "__main__":
    if "--selftest" in ARGS:
        selftest()
    else:
        root = Path(ARGS[ARGS.index("--root") + 1]).resolve() if "--root" in ARGS else HERE / "out_diag"
        txt, J = read_diag(root, scan=json.loads((HERE / "preflight" / "seed_scan.json").read_text()))
        dest = HERE if "--root" not in ARGS else root.parent
        (dest / "readout_diag.md").write_text(txt)
        (dest / "readout_diag.json").write_text(json.dumps(J, indent=1, default=str))
        print(txt)
