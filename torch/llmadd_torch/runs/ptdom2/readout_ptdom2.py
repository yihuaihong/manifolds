"""ptdom2 read-out (PRE-REGISTERED in runs/ptdom2/AUDIT.md before any ptdom2 GPU result): the R1 lr selection (select_ptdom2.py
calls decide_R; the read-out re-derives it), the Part R verdicts on seeds 8-15 and (A2 / A3) the Part S verdicts on fresh
seeds 8-15. Runs: ptdom's (runs/ptdom/out, read only; the R1 selection seeds 0-3) and ptdom2's
(runs/ptdom2/out), loaded by load2 = readout_ptdom.load generalised to the ptdom2 names (verified identical to readout_ptdom.load
on all of ptdom's cells: tests/load2_vs_load.out) with the same validity rules (A5.1 flagged-valid tolerances / CKA-rise bar,
A6 / A6.1 geo overshoot; the A2.1 per-cell auxiliary-loss gate). The statistics are readout_ptdom's own (boot_ci: 10,000
paired resamples, default_rng(0), percentile; sf_test: the one-sided one-sample t; holm).

  python readout_ptdom2.py [--part R|S|all] [--allow_partial] [--json out.json]
  python readout_ptdom2.py --selftest

Primary metric = area (100 x mean over the 75 evals) of the domain's held-out read-out (readout_ptdom.READ: comparisons the
2-way choice over every held-out question in both orders; months the full-span k-way choice, tier S); final3 secondary.
PART R (stars, landmarks | compass, prefixes, ranks | scalar_adj), seeds 8-15 only (no selection step reads them):
  gain_s = area(only-d@P, s) - area(none@N, s); PASS iff the 8-seed bootstrap 90% lower bound > 0, both halves (seeds 8-11, 12-15)
  mean > 0, the one-sided t p (H1: mean > 0, df 7), Holm over the R domains with a verdict (status FINAL), < 0.10, and >= 6/8
  seeds positive; else NOT SHOWN. GEOMETRY-SPECIFIC: the same four conditions on area(only-d@P) - area(only-d-shuf@P) (the
  dose-matched shuffled target, same lr), Holm over the same family (adjusted separately); else NOT SHOWN. NOT FINAL / GATE
  domains: no verdict, not in the family. Printed beside every verdict (descriptive): final3 gains, hard-subset final3 gains,
  the 2e-3 same-lr gain (ptdom's seeds 0-3), FLOOR / CEILING (ptdom's definitions; a FLOOR line > 100 is noted), the training-
  frequency shortcut baseline against each arm, the aux gate per cell, the lr-edge notes.
PART S (countries, people | cities, months; A2 / A3): seeds 8-15, all three arms ptdom2 runs at lr 2e-3 except none, which is
  at F100's none pick (make_jobs_ptdom2.S_NONE: countries / people 2e-3, cities / months 4e-3): PASS = the four conditions on
  only-d@2e-3 - none@N (F100's PASS on fresh seeds), GEOMETRY-SPECIFIC on only-d@2e-3 - only-d-shuf@2e-3, halves 8-11 / 12-15,
  Holm over the four per contrast; shuffled - none at the SAME lr (none@2e-3) and, for cities / months, a descriptive same-lr
  PASS line; the old seeds 0-7 pairing (part S0, optional) only descriptive, labelled as selected by F100.
Beside every R / S verdict: the pre-registered reading (A3.2 as amended by A4): the geometry-dependence sentence only when
PASS and GEOMETRY-SPECIFIC hold (final, gate passed) and: no FLOOR; both final3 gains (true - none, true - shuffled) have a paired
bootstrap 90% lower bound > 0 (A4.1); shuffled - none (S: none@2e-3) has a bootstrap 90% UPPER bound >= 0 (A4.2a: the shuffled
arm not shown to hurt) and mean(true - shuffled) >= 0.5 x mean(true - none) (A4.2b: the shuffled arm takes at most half of the
gain); the two arms' LM shares within 0.1; else each failing condition by name.
A verdict is refused (exit) while a needed run is missing or invalid, or the decision chain differs from the picks files;
--allow_partial prints PROVISIONAL instead (no verdict word). A4.3: a missing / invalid run that only a DESCRIPTIVE line needs
(Part S: A's none@2e-3 for cities / months) refuses nothing; those lines print "missing / invalid" and the sentence is withheld.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B = next(p for p in HERE.parents if (p / "make_jobs_ptdom2.py").is_file())   # A4: the repo root, also from a working copy below runs/ptdom2
R0 = B / "runs" / "ptdom"
sys.path.insert(0, str(R0)); sys.path.insert(0, str(B))
RO_FILE, RO_SHA = R0 / "readout_ptdom.py", "1e2f17522e47"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
assert sha(RO_FILE) == RO_SHA, f"readout_ptdom.py changed: {sha(RO_FILE)} != {RO_SHA}"
import readout_ptdom as RO       # noqa: E402  (read only: load's rules, the gate, select_unit, boot_ci, sf_test, holm, meta)
import make_jobs_ptdom2 as M2    # noqa: E402
assert Path(RO.__file__).resolve() == RO_FILE.resolve()
MJ, RR = RO.MJ, RO.RR
BATCHES, COMP, KSTEP, READ, SEEN = RO.BATCHES, RO.COMP, RO.KSTEP, RO.READ, RO.SEEN
R_DOMS, S_DOMS = M2.R_DOMS, M2.S_DOMS
SEL, S_SEEDS, R2_SEEDS = M2.SEL, M2.S_SEEDS, M2.R2_SEEDS
ALPHA = 0.10           # the one-sided t p (Holm) bar of both verdicts
WINS_MIN = 6           # of 8 seeds
N_EVALS = RO.N_EVALS
A6_SHUF = "adjacent"   # A1.4 (the coordinator, 2026-11-08): A6 / A6.1 for a SHUFFLED arm only = geo_a6_shuf_adjacent (the perfect fit
                       # verified on the two adjacent on-evals' geo <= 0.005; readout_ptdom.geo_a6's cka_<active> is the eval CKA to the
                       # TRUE target there); true-target and none arms keep readout_ptdom.geo_a6 unchanged ('literal' = A0's choice)
NAME2 = re.compile(r"^(ptdom2?)-(A|B1|B2|B3)-(none|only-[a-z_]+?(?:-shuf)?)-lr([0-9.e-]+)-s(\d+)(?:-r(\d+))?$")
med = RO.med


# ------------------------------------------------------------------------------------------------------------ loading
def cell_dirs2(root, prefix):
    out = {}
    for d in sorted(Path(root).glob(f"{prefix}-*-s*")):
        m = NAME2.match(d.name)
        if m and m.group(1) == prefix:
            out.setdefault((m.group(2), m.group(3), m.group(4), int(m.group(5))), []).append((int(m.group(6) or 0), d))
    return {k: sorted(v) for k, v in out.items()}


def tag_of(prefix, batch, arm, lr):
    return f"{prefix}-{batch}-{arm}-lr{lr}"


def run_json2(d, prefix, batch, arm, lr, seed):
    t = tag_of(prefix, batch, arm, lr)
    return Path(d) / t / f"run_whole_none_seed{seed}_{t}.json"


def geo_a6_shuf_adjacent(r, ck, arm):
    """A1.4 (active for SHUFFLED arms, A6_SHUF == "adjacent"): readout_ptdom.geo_a6 with its cka condition
    replaced by: the geo_<active> (1 - CKA to the SHUFFLED target on a training batch) at the previous and the next eval with
    prior_on are both <= 0.005 (a perfect fit measured on other batches); everything else as geo_a6."""
    g = RO.geo_a6(r, ck, arm)
    d = g["flag"]["active"] if g["flag"] else None
    if d is None:
        return g
    on = [e for e in r["curve"] if e.get("prior_on")]
    why = [w for w in g["why"] if "not a perfect fit" not in w and "above any float" not in w]
    for x in g["flag"]["evals"]:
        i = next(j for j, e in enumerate(on) if e["step"] == x["step"])
        nb = [on[j].get(f"geo_{d}") for j in (i - 1, i + 1) if 0 <= j < len(on)]
        if len(nb) < 2 or not all(RO._num(v) and v <= 0.005 for v in nb):
            why.append(f"(2s) step {x['step']}: adjacent on-evals' geo_{d} {nb} not all <= 0.005 (no perfect fit on other batches)")
    return {"why": why, "flag": g["flag"]}


def load2(root, prefix, guard=True):
    """readout_ptdom.load for runs named <prefix>-<batch>-<arm>-lr<lr>-s<seed>[-r<k>] (prefix ptdom or ptdom2; arm none,
    only-<d>, only-<d>-shuf): the same record fields and validity rules. Added (ptdom2): the run's identity (args.seed / lr /
    tag / prior_domains / prior_shuf, domains.<d>.prior / shuf) must match its directory name, else invalid. guard: the A6
    name guard on the job files (ptdom: readout_ptdom.a6_guard; ptdom2: guard2)."""
    if guard:
        RO.a6_guard() if prefix == "ptdom" else guard2()
    out = {}
    for (batch, arm, lr, seed), dirs in cell_dirs2(root, prefix).items():
        doms = BATCHES[batch]
        k_, d = dirs[-1]
        f = run_json2(d, prefix, batch, arm, lr, seed)
        early = []
        for k0, d0 in dirs[:-1]:
            f0 = run_json2(d0, prefix, batch, arm, lr, seed)
            try:
                if f0.exists() and len(json.loads(f0.read_text())["curve"]) == N_EVALS:
                    early.append(d0.name)
            except (json.JSONDecodeError, KeyError):
                pass
        rec = {"batch": batch, "arm": arm, "lr": lr, "seed": seed, "dir": str(d), "rerun": k_, "source": prefix,
               "superseded": [d0.name for _, d0 in dirs[:-1]], "earlier_finished": early, "valid": False, "why": "", "finished": False}
        out[batch, arm, lr, seed] = rec
        if not f.exists():
            rec["why"] = "no run JSON"
            continue
        r = json.loads(f.read_text())
        c = r["curve"]
        ck = json.loads((d / "check.json").read_text()) if (d / "check.json").exists() else None
        off = json.loads((d / "offline_lm.json").read_text()) if (d / "offline_lm.json").exists() else {}
        oe = off.get(str(f.resolve())) or (next(iter(off.values())) if len(off) == 1 else None)
        fails = [k["check"] for k in (ck or {}).get("checks", []) if not k["ok"]]
        dpf = [{"check": k["check"].split(" (")[0], "value": k.get("value")} for k in (ck or {}).get("checks", [])
               if not k["ok"] and k["check"].startswith(RO.DP_FLAG_CHECKS)]
        div = next((e["step"] for e in c if not all(v is None or math.isfinite(v) for v in
                                                      (e.get("lm"), e.get("val_loss"), e.get("geo")))), None)
        cmp_ = [x for x in doms if x in COMP]
        rec.update(code_sha=r.get("code_sha"), args=r["args"], n_evals=len(c), diverged_at=div, finished=len(c) == N_EVALS,
                   area={x: r["metrics"].get(READ[x], {}).get("area") for x in doms},
                   final3={x: r["metrics"].get(READ[x], {}).get("final3") for x in doms},
                   area1={x: r["metrics"].get(RO.READ1[x], {}).get("area") for x in doms if x in KSTEP},
                   kshort={x: (100 * r["domains"][x]["heuristic_shortest_kspan"] if r["domains"][x].get("heuristic_shortest_kspan")
                               is not None else None) for x in doms if x in KSTEP},
                   seen3={x: r["metrics"].get(SEEN[x], {}).get("final3") for x in doms},
                   hard={x: r["metrics"].get(f"{x}_readout_hard", {}).get("area") for x in cmp_},
                   hard3={x: r["metrics"].get(f"{x}_readout_hard", {}).get("final3") for x in cmp_},
                   n_hard={x: r["domains"][x].get("n_hard") for x in cmp_},
                   chance={x: 100 * r["domains"][x]["chance_readout"] for x in doms},
                   heur={x: [100 * r["domains"][x].get("heuristic_ansfreq", 0), 100 * r["domains"][x].get("heuristic_shorter", 0)]
                         for x in cmp_},
                   split_seed={x: r["domains"][x].get("split_seed") for x in doms},
                   step0={x: 100 * c[0][RO.EVAL_KEY[x]] for x in doms},
                   numbers=r["metrics"]["numbers"]["area"], data_sha=[e.get("data_sha") for e in c], lm0=c[0]["lm"],
                   bpb_intr=c[-1]["val_bits_per_byte"] if div is None else math.inf,
                   bpb_off=(RR.offline_bpb(oe, diverged=div is not None) if oe else None),
                   fails=fails, grad=RO.window_grad(c, div), cmp={},
                   cka_final={x: c[-1].get(f"cka_{x}") for x in doms},
                   shuf_cka_to_true={x: r["domains"][x].get("shuf_cka_to_true") for x in doms},
                   steps=[e["step"] for e in c], series={x: [e.get(RO.EVAL_KEY[x]) for e in c] for x in doms},   # A2.2 window split
                   cka_wlast={x: next((e.get(f"cka_{x}") for e in reversed(c) if e["step"] <= WIN[1]), None) for x in doms})
        if div is not None:
            rec["bpb_off"] = math.inf
        for x in cmp_:
            D_ = r["domains"].get(x, {})
            eh = D_.get("eval_held")
            if eh and all(f"hits2_{x}_held" in e for e in c):
                rec["cmp"][x] = {"q": eh["q"], "nctx": eh["contexts"], "masks": eh.get("masks", {}),
                                 "split_seed": D_.get("split_seed"), "table_sha": D_.get("table", {}).get("sha"),
                                 "heur": [D_.get("heuristic_ansfreq"), D_.get("heuristic_shorter")],
                                 "hits": [e[f"hits2_{x}_held"] for e in c], "logged": [e.get(f"{x}_held_2way") for e in c],
                                 "logged_hard": [e.get(f"{x}_held_2way_hard") for e in c],
                                 "area": rec["area"][x], "area_hard": rec["hard"][x]}
        # ptdom2: the run's identity = its directory name (a second path besides CHECK_PY's intent checks)
        a_ = r["args"]
        dm_ = M2.dom(arm)
        want = {"seed": seed, "lr": float(lr), "tag": f"_{tag_of(prefix, batch, arm, lr)}", "prior_domains": dm_ or "",
                "prior_shuf": dm_ if M2.is_shuf(arm) else ""}
        idbad = {k: [a_.get(k), v] for k, v in want.items() if a_.get(k) != v}
        flg = {x: [bool(r["domains"][x].get("prior")), bool(r["domains"][x].get("shuf"))] for x in ("numbers",) + doms}
        wflg = {x: [x == dm_, x == dm_ and M2.is_shuf(arm)] for x in ("numbers",) + doms}
        if flg != wflg:
            idbad["domains.<d>.prior / shuf"] = [flg, wflg]
        winbad = []
        if arm != "none":   # A3.3: WIN (the window split's constant) = the trainer's window in this very run (A4.4: else this run invalid)
            won = [e["step"] for e in c if e.get("prior_on")]
            wwin = [st for st in rec["steps"] if WIN[0] <= st <= WIN[1]]
            if won != wwin:
                winbad.append(f"prior_on evals {won[:2]}..{won[-1:]} ({len(won)}) != the evals inside WIN {list(WIN)} {wwin[:2]}..{wwin[-1:]} ({len(wwin)})")
            if len(c) == N_EVALS and r.get("prior_events") != [[WIN[0], 1], [WIN[1] + 1, 0]]:
                winbad.append(f"prior_events {r.get('prior_events')} != [[{WIN[0]}, 1], [{WIN[1] + 1}, 0]]")
        geo_f = [x for x in fails if x.startswith(RO.GEO_FLAG_CHECK)]
        a6 = None
        if idbad:
            rec["why"] = f"identity: the run's args / domains differ from its directory name {idbad}"
        elif winbad:
            rec["why"] = "window (A4.4): the run's prior window differs from WIN: " + "; ".join(winbad)
        elif ck is None:
            rec["why"] = "no check.json"
        elif len(c) != N_EVALS:
            rec["why"] = f"unfinished ({len(c)} evals)"
        elif fails and not all(x.startswith((RO.CKA_CHECK, RO.GEO_FLAG_CHECK) + RO.DP_FLAG_CHECKS) for x in fails):
            rec["why"] = "failed check: " + "; ".join(fails)
        elif geo_f and (a6 := (geo_a6_shuf_adjacent(r, ck, M2.base(arm)) if (M2.is_shuf(arm) and A6_SHUF == "adjacent")
                               else RO.geo_a6(r, ck, M2.base(arm))))["why"]:
            rec["why"] = "failed check: " + "; ".join(fails) + " (A6 does not apply: " + "; ".join(a6["why"]) + ")"
        else:
            rec["valid"] = True
            rec["dp_flag"] = dpf
            rec["geo_flag"] = a6["flag"] if a6 else None
            rec["why"] = ("valid" + (" (CKA-rise check failed: a weak prior, data)" if any(x.startswith(RO.CKA_CHECK) for x in fails) else "")
                          + (" (FLAG A5.1)" if dpf else "") + (" (FLAG A6)" if rec["geo_flag"] else ""))
    return out


def guard2(job_dir=None):
    """A6 / A5.1 name guard on ptdom2's rendered job files (job_ptdom2-*.sbatch): the CHECK_PY heredoc is make_jobs_ptdom's
    CHECK_PY byte for byte (true-target and none arms) or make_jobs_ptdom2.CHECK_PY_SHUF (shuffled arms: the job's ARGS carry
    --prior_shuf), and readout_ptdom.a6_guard's conditions hold (the geo check's template once, geo_tol, the A6_KEEP names once).
    No job file yet -> 0 (nothing to guard)."""
    jd = Path(job_dir or M2.R)
    files = sorted(jd.glob("job_ptdom2-*.sbatch"))
    sep = "<<'PY' | tee \"$OUT/check.txt\"\n"
    for f in files:
        txt = f.read_text()
        m = re.search(r"\"\$OUT/check.json\" '(\{.*\})' \"\$ARGS\" <<'PY'", txt)
        assert m and txt.count(sep) == 1, f"guard2: {f.name}: no CHECK_PY heredoc / intent"
        cp = txt.split(sep, 1)[1].split("\nPY\n", 1)[0] + "\n"
        shuf = "--prior_shuf " in txt.split('ARGS="', 1)[1].split('"', 1)[0]
        assert cp == (M2.CHECK_PY_SHUF if shuf else MJ.CHECK_PY), f"guard2: {f.name}: CHECK_PY is not the {'shuffled' if shuf else 'ptdom'} one"
        tol = json.loads(m.group(1)).get("geo_tol")
        assert cp.count(RO.GEO_TEMPLATE) == 1 and tol == RO.GEO_TOL, f"guard2: {f.name}: the geo check / geo_tol"
        assert all(cp.count(f'chk("{p}') == 1 for p in RO.A6_KEEP), f"guard2: {f.name}: an A6_KEEP check is not named once"
        ja = txt.split('ARGS="', 1)[1].split('"', 1)[0].split()
        sd = ja[ja.index("--prior_shuf") + 1] if "--prior_shuf" in ja else None
        I_ = json.loads(m.group(1))
        assert (I_.get("shuf") == [sd] and I_["args"].get("prior_shuf") == sd) if sd else ("shuf" not in I_ and I_["args"].get("prior_shuf") == ""), \
            f"guard2: {f.name}: the intent's shuf / prior_shuf != the ARGS' --prior_shuf"
    return len(files)


MANIFEST = M2.R / "reused_ptdom.json"   # tests/reused_manifest.py: sha256[:16] of every re-used ptdom run JSON and check.json


def reused_cells():
    """The ptdom cells ptdom2 re-uses: R1's selection cells (seeds 0-3) and Part S's true-target arm (seeds 0-7)."""
    out = [(b, f"only-{d}", lr, s) for d, b in R_DOMS.items() for lr in M2.PRIOR_LRS0[d] for s in SEL]
    out += [(b, "none", lr, s) for b, lrs in M2.NONE_LRS0.items() for lr in lrs for s in SEL]
    out += [(b, f"only-{d}", M2.S_LR, s) for d, b in S_DOMS.items() for s in S_SEEDS]
    return out


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def check_manifest(R_):
    """Every re-used ptdom cell is read from the run JSON / check.json whose sha the manifest froze (problems -> refuse)."""
    if not MANIFEST.exists():
        return [f"no manifest {MANIFEST} (tests/reused_manifest.py --write)"]
    man, probs = json.loads(MANIFEST.read_text())["cells"], []
    for k in reused_cells():
        key = "|".join(map(str, k))
        r = R_.get(k)
        if key not in man or r is None:
            probs.append(f"{key}: not in the manifest or not loaded")
            continue
        d = Path(r["dir"])
        f = run_json2(d, "ptdom", *k)
        if r["source"] != "ptdom" or file_sha(f) != man[key]["run_sha"] or file_sha(d / "check.json") != man[key]["check_sha"]:
            probs.append(f"{key}: the run JSON / check.json differs from the manifest")
    return probs


def check_s_none(o=None, s_none=None):
    """A3.1: make_jobs_ptdom2.S_NONE = F100's none pick per S domain in runs/ptdom/readout.json (read only). o / s_none: a
    planted read-out dict / S_NONE (A4.4: the self-test's refusal cases); default the real files."""
    o = json.loads((R0 / "readout.json").read_text()) if o is None else o
    sn = M2.S_NONE if s_none is None else s_none
    got = {d: o["domains"][d]["pick_none"] for d in S_DOMS}
    pri = {d: o["domains"][d]["pick_prior"] for d in S_DOMS}
    return ([] if got == sn else [f"S_NONE {sn} != F100's none picks {got}"]) + \
           ([] if all(v == M2.S_LR for v in pri.values()) else [f"F100's prior picks {pri} != S_LR {M2.S_LR}"])


def load_all(root0=None, root2=None, guard=True):
    """ptdom's cells (read only) + ptdom2's; no key may be in both."""
    A = load2(root0 or (R0 / "out"), "ptdom", guard)
    Bx = load2(root2 or (M2.R / "out"), "ptdom2", guard)
    both = sorted(set(A) & set(Bx))
    assert not both, f"cells in both batches: {both[:5]}"
    return {**A, **Bx}


# ------------------------------------------------------------------------------------------------------------ selection (R1)
def gate_cells(R_, b, arm, lrs, seeds=SEL):
    """The A2.1 gate per (arm, lr) cell over the FINISHED runs of the seeds (valid or not)."""
    out = {}
    for lr in lrs:
        rs = [r for k, r in R_.items() if k[:3] == (b, arm, lr) and k[3] in seeds and r.get("finished")]
        if rs:
            out[lr] = RO.cell_gate(rs)
    return out


def unit(R_, b, arm, d, lrs, excluded=()):
    u = RO.select_unit(R_, b, arm, d, lrs, excluded, SEL)
    if "missing" in u:
        raise SystemExit(f"selection refused: {b} {arm}|{d} lacks valid runs {u['missing']}")
    return u


def check_existing(R_):
    """The re-used ptdom cells are exactly PRIOR_LRS0 / NONE_LRS0 with valid runs at seeds 0-3 (and no other ptdom lr of
    these arms exists on disk)."""
    probs = []
    for d, b in R_DOMS.items():
        have = sorted({k[2] for k, r in R_.items() if k[:2] == (b, f"only-{d}") and r["source"] == "ptdom"}, key=float)
        if have != sorted(M2.PRIOR_LRS0[d], key=float):
            probs.append(f"{d}: ptdom prior lrs on disk {have} != {M2.PRIOR_LRS0[d]}")
    for b, lrs in M2.NONE_LRS0.items():
        have = sorted({k[2] for k, r in R_.items() if k[:2] == (b, "none") and r["source"] == "ptdom"}, key=float)
        if have != sorted(lrs, key=float):
            probs.append(f"batch {b}: ptdom none lrs on disk {have} != {lrs}")
    return probs


def none_asks_from_ptdom(R_):
    """A1 (decision 10): the none extensions the rule asks from ptdom's data ALONE (the none units of the R domains over the
    batch's ptdom none grid; no ptdom2 run enters): the upper step (x2) where some R domain's none pick or area argmax is the
    batch's highest ptdom none lr, the lower step (x1/2) where it is the lowest. -> sorted [(batch, 'none', lr)] (= R1PRE)."""
    out = set()
    for d, b in R_DOMS.items():
        g = list(M2.NONE_LRS0[b])
        un = unit(R_, b, "none", d, g)
        if g[-1] in (un["pick"], un["argmax"]):
            out.add((b, "none", M2.NONE_EXT[b]))
        if g[0] in (un["pick"], un["argmax"]):
            out.add((b, "none", M2.NONE_LOW[b]))
    return sorted(out, key=lambda t: (t[0], float(t[2])))


def check_r1pre(R_):
    """A1.10: R1PRE must be exactly what the rule asks from ptdom's data alone (select_ptdom2 and the read-out refuse otherwise)."""
    d_ = none_asks_from_ptdom(R_)
    pre = sorted((tuple(x) for x in M2.R1PRE), key=lambda t: (t[0], float(t[2])))
    return [] if d_ == pre else [f"R1pre {pre} != the extensions ptdom's data ask {d_}"]


def decide_R(R_, after, pre=None):
    """The R decision (AUDIT.md Design + Amendment A1). after 'R1': ptdom's lrs + 4e-3 + R1pre's none cells (make_jobs_ptdom2.R1PRE,
    run with R1: the extensions ptdom's data already ask); 'R1x': + every extension the R1 decision asked. Per R domain d:
      the A2.1 gate per prior cell (seeds 0-3): a failing cell cannot be the pick; every cell failing -> GATE (no verdict, out of
      the Holm family; no HOLD (A1.8));
      F87 rule 1 (readout_ptdom.select_unit, seeds 0-3, in-training bpb tie-break) for (only-d, d) over its prior lrs and for
      (none, d) over its batch's none grid (ptdom's + every none cell ptdom2 ran for that batch: shared, A1.3);
      edges (A1.2: pick OR area argmax at the outermost lr): prior upper edge -> 8e-3 once; none upper edge -> x2 once, none lower
      edge -> x1/2 once (A1.1a); fixB (A1.1b): the prior pick outside none's range (incl. this round's none asks) -> none at the
      prior pick's lr; all asked after R1 only, seeds 0-3 only. An edge whose next lr was run and failed the gate is closed.
      NOT FINAL: at the final decision a pick or argmax still at an outermost lr of its unit (prior: the upper edge; none:
      either edge), or the prior pick outside none's range. The prior has no lower extension (the design's): its lower edge is
      not an edge rule (A1, named).
    A domain whose own prior or whose batch's none has an ask is PENDING until R1x. pre: the none cells run with R1 (default
    make_jobs_ptdom2.R1PRE; tests pass their own); they count as the batch's used extension in their direction. -> the picks dict."""
    assert after in ("R1", "R1x")
    probs = check_existing(R_)
    if probs:
        raise SystemExit("selection refused: " + "; ".join(probs))
    pre = [tuple(x) for x in (M2.R1PRE if pre is None else pre)]
    miss = [(b, a, lr, s) for b, a, lr in pre for s in SEL if not ((b, a, lr, s) in R_ and R_[(b, a, lr, s)]["valid"])]
    if miss:
        raise SystemExit(f"selection refused: the R1pre cells lack valid runs {miss[:6]}")
    prev = None
    run_none = {b: set() for b in M2.NONE_LRS0}
    for b, a, lr in pre:
        run_none[b].add(lr)
    prior_ext = set()
    if after == "R1x":
        prev = decide_R(R_, "R1", pre)
        assert prev["stage_written"] == "R1x", "decide_R after R1x: the R1 decision asked for no extension"
        for b, a, lr in map(tuple, prev["E"]):
            (run_none[b].add(lr) if a == "none" else prior_ext.add(M2.dom(a)))
    doms, E, why_E = {}, [], {}

    def ask(cell, why):
        if cell not in E:
            E.append(cell)
        why_E.setdefault("|".join(cell), []).append(why)

    rows = {}
    for d, b in R_DOMS.items():
        arm = f"only-{d}"
        lrs_p = sorted(set(M2.PRIOR_LRS0[d]) | {M2.R1_LR} | ({M2.R1X_PRIOR} if d in prior_ext else set()), key=float)
        lrs_n = sorted(set(M2.NONE_LRS0[b]) | run_none[b], key=float)
        gate = gate_cells(R_, b, arm, lrs_p)
        miss_g = [lr for lr in lrs_p if lr not in gate]
        if miss_g:
            raise SystemExit(f"selection refused: {b} {arm}@{miss_g} has no finished run at seeds 0-3")
        bad = {lr for lr, g in gate.items() if not g["ok"]}
        up = unit(R_, b, arm, d, lrs_p, bad)
        un = unit(R_, b, "none", d, lrs_n)
        rows[d] = {"batch": b, "prior_lrs": lrs_p, "none_lrs": lrs_n, "gate": {lr: {k: g[k] for k in ("ok", "median", "min", "why")}
                                                                            for lr, g in gate.items()},
                   "gate_excluded": sorted(bad, key=float), "unit_prior": up, "unit_none": un, "pick_prior": up["pick"],
                   "pick_none": un["pick"], "argmax_prior": up["argmax"], "argmax_none": un["argmax"], "ext_prior": None,
                   "ext_none": [], "fixb": None, "why": [], "notes": [], "edges": []}
    for d, x in rows.items():
        b, up, un = x["batch"], x["unit_prior"], x["unit_none"]
        if up["pick"] is None:
            x.update(status="GATE", why=["AUX GATE FAILED: every prior cell fails the gate (no admissible lr): no verdict, not in the Holm family"])
            continue
        at = lambda u, lr: lr in (u["pick"], u["argmax"])
        top_p = x["prior_lrs"][-1]
        if at(up, top_p) and top_p not in x["gate_excluded"]:   # (a gate-failed top lr is never the pick / argmax)
            if after == "R1" and top_p == M2.R1_LR:
                x["ext_prior"] = M2.R1X_PRIOR
                ask((b, f"only-{d}", M2.R1X_PRIOR), f"{d}: prior pick / argmax at its top lr {top_p}")
            else:
                x["edges"].append(f"prior pick {up['pick']} / argmax {up['argmax']} at its outermost lr {top_p}")
        lo_n, hi_n = x["none_lrs"][0], x["none_lrs"][-1]
        if at(un, hi_n):
            if after == "R1" and float(hi_n) <= float(max(M2.NONE_LRS0[b], key=float)):
                x["ext_none"].append(M2.NONE_EXT[b])
                ask((b, "none", M2.NONE_EXT[b]), f"{d}: none pick / argmax at batch {b}'s top lr {hi_n}")
            else:
                x["edges"].append(f"none pick {un['pick']} / argmax {un['argmax']} at its outermost lr {hi_n}")
        if at(un, lo_n):
            if after == "R1" and float(lo_n) >= float(min(M2.NONE_LRS0[b], key=float)):
                x["ext_none"].append(M2.NONE_LOW[b])
                ask((b, "none", M2.NONE_LOW[b]), f"{d}: none pick / argmax at batch {b}'s lowest lr {lo_n}")
            else:
                x["edges"].append(f"none pick {un['pick']} / argmax {un['argmax']} at its outermost (lowest) lr {lo_n}")
    for d, x in rows.items():   # fixB (A1.1b), after this round's none asks of the batch are known
        if x.get("status") == "GATE":
            continue
        b, P = x["batch"], x["pick_prior"]
        rng = [float(v) for v in x["none_lrs"]] + [float(lr) for bb, a, lr in E if bb == b and a == "none"]
        if not (min(rng) <= float(P) <= max(rng)):
            if after == "R1":
                x["fixb"] = P
                ask((b, "none", P), f"{d}: fixB: the prior pick {P} outside none's range [{min(rng):g}, {max(rng):g}]")
            else:
                x["edges"].append(f"the prior pick {P} outside none's range {x['none_lrs'][0]}..{x['none_lrs'][-1]} (fixB used)")
    pend_b = {bb for bb, a, lr in E if a == "none"}
    for d, x in rows.items():
        if x.get("status") == "GATE":
            doms[d] = x
            continue
        b = x["batch"]
        if after == "R1" and (x["ext_prior"] or b in pend_b):
            x["status"] = "PENDING"
            x["why"].append("waits for R1x: " + ", ".join(f"{a}@{lr}" for bb, a, lr in E if (bb == b and a == "none") or a == f"only-{d}"))
        elif x["edges"]:
            x.update(status="NOT FINAL", why=list(x["edges"]))
        else:
            x["status"] = "FINAL"
        if float(x["pick_prior"]) < float(x["none_lrs"][0]):
            x["notes"].append(f"the prior pick {x['pick_prior']} is below none's lowest lr {x['none_lrs'][0]}")
        if x["pick_prior"] == x["prior_lrs"][0]:   # A2.4 (the coordinator's A1 line): descriptive, no rule
            x["notes"].append(f"PRIOR AT ITS LOWEST LR {x['pick_prior']} (the prior has no lower extension: its best lr may lie lower; "
                              "conservative for a PASS)")
        doms[d] = x
    final = after == "R1x" or not E
    C = []
    if final:
        for d, x in doms.items():
            if x["status"] == "FINAL":
                b = x["batch"]
                for c in ((b, f"only-{d}", x["pick_prior"]), (b, f"only-{d}-shuf", x["pick_prior"]), (b, "none", x["pick_none"])):
                    if c not in C:
                        C.append(c)
    srt = lambda xs: [list(t) for t in sorted(xs, key=lambda t: (t[0], t[1], float(t[2])))]
    return {"design": "ptdom2", "part": "R", "amendment": "A1", "status": "decided", "after": after,
            "stage_written": "R2" if final and C else ("R1x" if not final else None), "domains": doms,
            "E": srt(E) if not final else [], "E_why": why_E if not final else {}, "R1pre": srt(pre),
            "C": srt(C), "complete": final, "family": [d for d, v in doms.items() if v["status"] == "FINAL"]}


def summarize_pick(pk):
    """The fields a picks file is compared on (consistency check)."""
    return {"after": pk["after"], "stage_written": pk["stage_written"], "E": pk["E"], "C": pk["C"], "R1pre": pk.get("R1pre"),
            "amendment": pk.get("amendment"),
            "domains": {d: {k: v.get(k) for k in ("status", "pick_prior", "pick_none", "argmax_prior", "argmax_none", "ext_prior",
                                                  "ext_none", "fixb", "prior_lrs", "none_lrs", "gate_excluded")} for d, v in pk["domains"].items()}}


# ------------------------------------------------------------------------------------------------------------ verdict rules
def rule(g, p_holm, halves):
    """The four pre-registered conditions on 8 paired values g (seed order), halves = (first 4, last 4) -> (bool, details)."""
    g = np.asarray(g, float)
    assert len(g) == 8
    lo, hi = RO.boot_ci(g)
    h1, h2 = float(np.mean(g[:4])), float(np.mean(g[4:]))
    wins = int((g > 0).sum())
    c = {"boot90_lo_gt0": lo > 0, "halves_gt0": h1 > 0 and h2 > 0, "p_holm_lt": p_holm < ALPHA, "wins_ge": wins >= WINS_MIN}
    return all(c.values()), {"boot90": [lo, hi], "halves": [h1, h2], "halves_seeds": halves, "wins": wins, "conditions": c,
                             "mean": float(g.mean()), "p_holm": p_holm}


def ttest2(g):
    mu, p, ci = RO.t_test(g)
    return {"mean": mu, "p_two_sided": p, "t95": list(ci)}


def need(R_, cells, problems):
    """The records of the cells [(batch, arm, lr, seed)]; a missing / invalid one is a problem (the read-out refuses)."""
    out = []
    for k in cells:
        r = R_.get(k)
        if not r or not r["valid"]:
            problems.append(f"{k[0]} {k[1]}@{k[2]} s{k[3]}: " + (r["why"] if r else "no run"))
        out.append(r if r and r["valid"] else None)
    return out


def pair(R_, b, a1, a2, lr1, lr2, d, seeds, problems, field="area"):
    x = need(R_, [(b, a1, lr1, s) for s in seeds], problems)
    y = need(R_, [(b, a2, lr2, s) for s in seeds], problems)
    if any(v is None for v in x + y):
        return None, None, None
    v1 = [r[field][d] for r in x]
    v2 = [r[field][d] for r in y]
    if any(v is None for v in v1 + v2):
        problems.append(f"{d}: {field} missing in a run of {a1}@{lr1} / {a2}@{lr2}")
        return None, None, None
    return np.array(v1) - np.array(v2), v1, v2


def hard3_pair(R_, b, a1, a2, lr1, lr2, d, seeds):
    """final3 of the heuristic-hard subset (comparisons), paired; None when a seed's subset is empty."""
    x = [R_.get((b, a1, lr1, s)) for s in seeds]
    y = [R_.get((b, a2, lr2, s)) for s in seeds]
    if d not in COMP or not all(r and r["valid"] and r["hard3"].get(d) is not None for r in x + y):
        return None
    return np.array([r["hard3"][d] for r in x]) - np.array([r["hard3"][d] for r in y])


def shortcut_vs(R_, subs, b, arm, lr, d, seeds):
    """The training-frequency shortcut baseline (readout_ptdom.split_info's h3 on the hard subset) against one arm: the arm's
    pooled hard-subset final3 vs the baseline's accuracy on the same questions (pooled over the seeds, weighted by n_hard);
    descriptive (ptdom's condition (iii) is not a rule here)."""
    rows = []
    for s in seeds:
        r = R_.get((b, arm, lr, s))
        sb = subs.get((b, arm, lr, s), {}).get(d)
        rows.append((r["hard3"].get(d) if r else None, (sb or {}).get("h3_hard"), r["n_hard"].get(d) if r else None))
    x = RO.shortcut(rows)
    return {k: x[k] for k in ("prior", "base", "n_pooled", "testable")} | {"arm_minus_base": (x["prior"] - x["base"]) if x["prior"] is not None and x["base"] is not None else None}


def floor_tags(R_, subs, b, arms_lrs, d, seeds):
    """ptdom's FLOOR (both compared arms' mean area <= max(chance, the heuristic lines, the shortcut baseline on every held-out
    question) + 5) and CEILING (the reference arm >= 95) with audit_results item 5's note (a line > 100)."""
    recs = [R_.get((b, a, lr, s)) for a, lr in arms_lrs for s in seeds]
    recs = [r for r in recs if r and r["valid"]]
    if not recs:
        return None
    lines = [float(np.mean([r["chance"][d] for r in recs]))]
    if d in COMP:
        lines += list(np.mean([r["heur"][d] for r in recs], 0))
        h3 = [subs.get((r["batch"], r["arm"], r["lr"], r["seed"]), {}).get(d, {}).get("h3_held") for r in recs]
        if all(v is not None for v in h3):
            lines.append(float(np.mean(h3)))
    if d in KSTEP:   # ptdom A4.5: the k-span's shortest-span line (months has no elimination line; the other k-step domains are not in ptdom2)
        ks = [r["kshort"].get(d) for r in recs]
        if all(v is not None for v in ks):
            lines.append(float(np.mean(ks)))
    names = ["chance"] + (["answer-frequency line", "shorter-option line"] if d in COMP else [])
    names += (["shortcut baseline (every held-out question)"] if len(lines) > len(names) and d in COMP else [])
    names += (["shortest-span line"] if d in KSTEP and len(lines) > len(names) else [])
    names = (names + ["?"] * len(lines))[:len(lines)]
    best = int(np.argmax(lines))
    line = max(lines) + 5
    means = {f"{a}@{lr}": float(np.mean([R_[(b, a, lr, s)]["area"][d] for s in seeds])) for a, lr in arms_lrs}
    tags = []
    if all(v <= line for v in means.values()):
        tags.append(f"FLOOR (both <= {line:.1f})" + (f" [the best non-geometric line, the {names[best]}, is {lines[best]:.1f} > 95: the FLOOR "
                                                     f"tag holds whenever both arms are <= {line:.1f} (A3.3)]" if line > 100 else ""))
    ref = list(means.values())[-1]
    if ref >= 95:
        tags.append("CEILING (reference arm >= 95)")
    return {"line": line, "lines": lines, "line_names": names, "means": means, "tags": tags}


WIN = (916, 7323)   # the prior window's steps (prior_events [[916, 1], [7324, 0]]): 26 of the 75 evals (1000 .. 7250)


def win_split(R_, b, a1, a2, d, seeds):
    """A2.2 (descriptive): a contrast's area gain split into the contributions of the evals inside the prior window and of the
    others (in + out = the area gain; area = 100 x the mean over the 75 evals). -> per-seed lists or None."""
    xs = [R_.get((b,) + a1 + (s,)) for s in seeds]
    ys = [R_.get((b,) + a2 + (s,)) for s in seeds]
    if not all(r and r["valid"] and r.get("series", {}).get(d) for r in xs + ys):
        return None
    ins, outs = [], []
    for x, y in zip(xs, ys):
        w = [WIN[0] <= st <= WIN[1] for st in x["steps"]]
        assert x["steps"] == y["steps"], "the two arms' eval steps differ"
        dv = [100 * (p - q) for p, q in zip(x["series"][d], y["series"][d])]
        n = len(dv)
        ins.append(sum(v for v, on in zip(dv, w) if on) / n)
        outs.append(sum(v for v, on in zip(dv, w) if not on) / n)
    return {"in": ins, "out": outs, "n_window_evals": sum(WIN[0] <= st <= WIN[1] for st in xs[0]["steps"])}


def series_problems(R_):
    """A2 double path: every valid run's area = 100 x the mean of its per-eval read-out series (readout_ptdom.EVAL_KEY)."""
    out = []
    for k, r in R_.items():
        if r.get("valid") and r.get("series"):
            for d, v in r["series"].items():
                if v and r["area"].get(d) is not None and abs(100 * float(np.mean(v)) - r["area"][d]) > 1e-6:
                    out.append(f"{k}: {d} area {r['area'][d]} != 100 x the mean of its series {100 * float(np.mean(v))}")
    return out


def contrast_domain(R_, subs, b, d, P, N, seeds, problems, notes=(), N_same=None):
    """Everything printed for one domain: the two verdict contrasts (only-d@P - none@N; only-d@P - only-d-shuf@P) on `seeds`
    and the descriptive lines (A2.2: shuffled - none, the window split, the three arms' LM shares; final3, hard final3, FLOOR /
    CEILING, the shortcut baseline per arm, the aux gate per cell, shuf_cka_to_true, the 'shuffled arm does not fit the true
    target' double path, diverged runs). N_same (A3.1, Part S): the none lr of the shuffled - none line (S: 2e-3, the same lr as
    both prior arms); default N (R: marked lr-confounded when P != N). A4.3: when N_same != N its cell is only descriptive: a
    missing / invalid run there refuses nothing (ns_missing names it; the lines that need it print "missing / invalid"); the
    gate needs only the true and shuffled arms."""
    T_, S_, N_ = (f"only-{d}", P), (f"only-{d}-shuf", P), ("none", N)
    NS_ = ("none", N_same or N)
    g, pv, nv = pair(R_, b, T_[0], N_[0], P, N, d, seeds, problems)
    gs, _, sv = pair(R_, b, T_[0], S_[0], P, P, d, seeds, problems)
    gsn = pair(R_, b, S_[0], NS_[0], P, NS_[1], d, seeds, [])[0]
    f3 = lambda a1, a2: pair(R_, b, a1[0], a2[0], a1[1], a2[1], d, seeds, [], "final3")[0]
    cg = lambda arm, lr, ss: RO.cell_gate([r for k, r in R_.items() if k[:3] == (b, arm, lr) and k[3] in ss and r.get("finished")])
    recs = {a: [R_.get((b,) + a + (s,)) for s in seeds] for a in dict.fromkeys((T_, S_, N_, NS_))}
    ok_arm = {a: all(r and r["valid"] for r in v) for a, v in recs.items()}
    have_ts = ok_arm[T_] and ok_arm[S_]   # A4.3: the gate (true arm per half, shuffled descriptive) needs only these two arms
    ns_miss = None if (ok_arm[NS_] or NS_ == N_) else [   # only a DESCRIPTIVE none cell (N_same != N); a verdict arm's gap goes to problems
        f"none@{NS_[1]} s{s}: {r['why'] if r else 'no run'}" for s, r in zip(seeds, recs[NS_]) if not (r and r["valid"])]
    shufc = [r["shuf_cka_to_true"].get(d) for r in recs[S_] if r and r["valid"]]
    ckc = None
    if have_ts and ok_arm[NS_]:   # A2.4 double path (descriptive): the shuffled arm's eval CKA to the TRUE target at the window's last eval
        ok_ = [None not in (rs["cka_wlast"].get(d), rn["cka_wlast"].get(d), rt["cka_wlast"].get(d)) and rs["cka_wlast"][d] <= rn["cka_wlast"][d] + 0.10
               and rs["cka_wlast"][d] <= rt["cka_wlast"][d] - 0.2 for rt, rs, rn in zip(recs[T_], recs[S_], recs[NS_])]
        rd = lambda v: None if v is None else round(v, 3)
        ckc = {"n_ok": int(sum(ok_)), "n": len(ok_), "shuf": [rd(r["cka_wlast"].get(d)) for r in recs[S_]],
               "none": [rd(r["cka_wlast"].get(d)) for r in recs[NS_]], "true": [rd(r["cka_wlast"].get(d)) for r in recs[T_]]}
    div = [f"{a[0]}@{a[1]} s{r['seed']} (step {r['diverged_at']})" for a, v in recs.items() for r in v if r and r.get("diverged_at") is not None]
    row = {"status": "FINAL", "batch": b, "pick_prior": P, "pick_none": N, "gain": g, "geo_gain": gs, "shuf_none": gsn,
           "prior": pv, "none": nv, "shuf": sv, "gain_final3": f3(T_, N_), "geo_final3": f3(T_, S_), "shuf_none_final3": f3(S_, NS_),
           "shuf_none_lr": NS_[1],
           "gain_hard3": hard3_pair(R_, b, T_[0], N_[0], P, N, d, seeds), "geo_hard3": hard3_pair(R_, b, T_[0], S_[0], P, P, d, seeds),
           "split": {"vs none": win_split(R_, b, T_, N_, d, seeds), "vs shuffled": win_split(R_, b, T_, S_, d, seeds),
                     "shuffled vs none": win_split(R_, b, S_, NS_, d, seeds)},
           "lm_share": {f"only-{d}@{P}": cg(T_[0], P, seeds)["median"], f"only-{d}-shuf@{P}": cg(S_[0], P, seeds)["median"],
                        f"none@{N}": 1.0},   # none: no prior term (a placeholder, not measured)
           "floor": floor_tags(R_, subs, b, [T_, N_], d, seeds) if g is not None else None,
           "floor_geo": floor_tags(R_, subs, b, [T_, S_], d, seeds) if gs is not None else None,
           "shortcut": {f"{a[0]}@{a[1]}": (shortcut_vs(R_, subs, b, a[0], a[1], d, seeds) if ok_arm[a] else None)
                        for a in dict.fromkeys((T_, S_, N_, NS_))} if (have_ts and d in COMP) else None,
           "gate": {"prior@P s8-11": cg(T_[0], P, seeds[:4]), "prior@P s12-15": cg(T_[0], P, seeds[4:]), "prior@P s8-15": cg(T_[0], P, seeds),
                    "shuf@P s8-15 (descriptive)": cg(S_[0], P, seeds)} if have_ts else None,
           "shuf_cka_to_true": [min(shufc), max(shufc)] if shufc else None, "cka_check": ckc, "diverged": div,
           "ns_missing": ns_miss, "lr_confounded_shuf_none": P != NS_[1], "notes": list(notes),
           "kind": "k-step" if d in KSTEP else "comparison (Tier L)"}
    return row


GEOM_OK = "PASS and GEOMETRY-SPECIFIC: the gain depends on the correct geometry"
SHARE_MIN, LM_DIFF = 0.5, 0.1   # A4.2b: mean(true - shuffled) >= 0.5 x mean(true - none); A3.2 (kept, A4.5): |LM share true - shuffled| <= 0.1


def reading_conditions(x):
    """A3.2 as amended by A4 (pre-registered): the sentence GEOM_OK is printed only when every condition holds; each failing one
    is named with its numbers. -> [reasons] (empty = all hold). Descriptive: the verdict words do not change.
      FLOOR (true vs none or true vs shuffled);
      A4.1 'only learns faster': the paired bootstrap 90% lower bound (readout_ptdom.boot_ci over the seeds' differences) of
        BOTH final3 gains (true - none@N, true - shuffled) > 0;
      A4.2a 'the shuffled arm got worse': the bootstrap 90% UPPER bound of shuffled - none (S: none@2e-3) >= 0; a missing /
        invalid none cell there fails it (A4.3);
      A4.2b 'the shuffled arm takes much of the gain': mean(true - shuffled) >= 0.5 x mean(true - none@N);
      the dose: |LM share true - shuffled| <= 0.1 (A3.2, unchanged: A4.5)."""
    why = []
    if any(tg.startswith("FLOOR") for fl in (x.get("floor"), x.get("floor_geo")) if fl for tg in fl["tags"]):
        why.append("FLOOR: " + "; ".join(tg for fl in (x.get("floor"), x.get("floor_geo")) if fl for tg in fl["tags"] if tg.startswith("FLOOR")))
    N, lr_sn = x.get("pick_none"), x.get("shuf_none_lr")
    f3s = {k: (None if x.get(k) is None else (float(np.mean(x[k])), float(RO.boot_ci(np.asarray(x[k], float))[0])))
           for k in ("gain_final3", "geo_final3")}
    if any(v is None or not v[1] > 0 for v in f3s.values()):
        dsc = lambda v: "-" if v is None else f"mean {v[0]:+.2f}, boot90 lower {v[1]:+.2f}"
        why.append(f"only learns faster: final3 gain vs none@{N} {dsc(f3s['gain_final3'])}; vs shuffled {dsc(f3s['geo_final3'])} "
                   "(both paired bootstrap 90% lower bounds must be > 0, A4.1)")
    sn = x.get("shuf_none")
    if sn is None:
        why.append(f"shuffled - none@{lr_sn} missing / invalid ({'; '.join((x.get('ns_missing') or ['a run'])[:4])}): whether the "
                   "shuffled arm hurts cannot be checked (A4.2a / A4.3)")
    else:
        lo_, hi_ = RO.boot_ci(np.asarray(sn, float))
        if not hi_ >= 0:
            why.append(f"the shuffled arm got worse: shuffled - none@{lr_sn} mean {mn(sn):+.2f}, boot90 [{lo_:+.2f}, {hi_:+.2f}] "
                       "(the upper bound must be >= 0, A4.2a)")
    g_, gs_ = mn(x.get("gain")), mn(x.get("geo_gain"))
    if g_ is None or gs_ is None or not gs_ >= SHARE_MIN * g_:
        why.append(f"the shuffled arm takes much of the gain: mean(true - shuffled) {s1(gs_)} < {SHARE_MIN:g} x mean(true - none@{N}) "
                   f"{s1(g_)} (= {s1(None if g_ is None else SHARE_MIN * g_)}) (A4.2b)")
    ls = list((x.get("lm_share") or {}).values())
    lt, lsh = (ls[0], ls[1]) if len(ls) >= 2 else (None, None)
    if lt is None or lsh is None or abs(lt - lsh) > LM_DIFF:
        why.append(f"the dose differs: LM share true {f3_(lt)}, shuffled {f3_(lsh)} (|difference| must be <= {LM_DIFF:g})")
    # A5 (the coordinator, 2026-11-09, after tests/audit_a4.md; before any verdict read-out): at n = 8 the point rules of A4.2
    # cannot exclude a harm or an absorption by the shuffled arm, so the sentence also needs both EXCLUDED by the paired
    # bootstrap 90% interval of shuffled - none (the none of A4.2a): (c) lower bound >= -HARM_MAX (a harm of more than 1 area
    # point excluded); (d) upper bound <= SHARE_MIN x mean(true - the same none) (the shuffled arm reaching half of the gain
    # excluded). Otherwise the reading is UNDETERMINED at n = 8 (named with the interval).
    if sn is not None and x.get("geo_gain") is not None and None not in sn and None not in x["geo_gain"]:
        lo5, hi5 = RO.boot_ci(np.asarray(sn, float))
        tn5 = float(np.mean(np.asarray(x["geo_gain"], float) + np.asarray(sn, float)))   # true - none@lr_sn = (true - shuf) + (shuf - none)
        if not lo5 >= -HARM_MAX:
            why.append(f"UNDETERMINED at n = 8: a harm of the shuffled arm is not excluded: shuffled - none@{lr_sn} boot90 "
                       f"[{lo5:+.2f}, {hi5:+.2f}] (the lower bound must be >= -{HARM_MAX:g}, A5c)")
        if not hi5 <= SHARE_MIN * tn5:
            why.append(f"UNDETERMINED at n = 8: the shuffled arm reaching half of the gain is not excluded: shuffled - none@{lr_sn} "
                       f"boot90 upper {hi5:+.2f} > {SHARE_MIN:g} x mean(true - none@{lr_sn}) {tn5:+.2f} (A5d)")
    elif sn is not None:
        why.append("UNDETERMINED: the A5 intervals cannot be computed (a seed is missing)")
    return why


HARM_MAX = 1.0   # A5c: a harm of the shuffled arm larger than this (area points) must be excluded by the boot90 interval


READING = {(True, True): GEOM_OK,
           (False, True): "GEOMETRY-SPECIFIC without PASS: true > shuffled, but no gain vs none shown (the shuffled prior may hurt)",
           (True, False): "PASS without GEOMETRY-SPECIFIC: the prior helps vs none; not shown to need the correct geometry",
           (False, False): "neither shown"}


def apply_rules(rows, fam, problems, ext_probs, halves):
    """The pre-registered conditions with Holm over `fam` per contrast; A1.7 the gate at the pick per half; PROVISIONAL (no
    verdict word) while a run is missing or any outside problem (integrity, manifest, R1pre, sub-read-outs: A2.4) stands."""
    ok_fam = [d for d in fam if rows[d].get("gain") is not None and rows[d].get("geo_gain") is not None and rows[d].get("gate") is not None]
    p1 = {d: RO.sf_test(rows[d]["gain"]) for d in ok_fam}
    p2 = {d: RO.sf_test(rows[d]["geo_gain"]) for d in ok_fam}
    h1 = dict(zip(ok_fam, RO.holm([p1[d] for d in ok_fam]))) if ok_fam else {}
    h2 = dict(zip(ok_fam, RO.holm([p2[d] for d in ok_fam]))) if ok_fam else {}
    final = len(ok_fam) == len(fam) and not problems and not ext_probs
    for d in fam:
        x = rows[d]
        if d not in ok_fam:
            x.update(verdict="PROVISIONAL (missing runs)", geometry="PROVISIONAL (missing runs)", reading=None)
            continue
        ok1, det1 = rule(x["gain"], h1[d], halves)
        ok2, det2 = rule(x["geo_gain"], h2[d], halves)
        det1.update(p_one_sided=p1[d], t_two_sided=ttest2(x["gain"]))
        det2.update(p_one_sided=p2[d], t_two_sided=ttest2(x["geo_gain"]))
        v1, v2 = ("PASS" if ok1 else "NOT SHOWN"), ("GEOMETRY-SPECIFIC" if ok2 else "NOT SHOWN")
        reading = READING[(ok1, ok2)]
        if (ok1, ok2) == (True, True):   # A3.2: the sentence only when every condition holds; else the failing ones by name
            rc = reading_conditions(x)
            x["reading_failed"] = rc
            if rc:
                reading = "PASS and GEOMETRY-SPECIFIC, but the geometry-dependence sentence is withheld (A3.2 / A4): " + " | ".join(rc)
        gt = x["gate"]
        if not (gt["prior@P s8-11"]["ok"] and gt["prior@P s12-15"]["ok"]):   # A1.7
            x["gate_failed"] = [w for k in ("prior@P s8-11", "prior@P s12-15") for w in gt[k]["why"]]
            x["would_be_without_gate"] = [v1, v2]
            v1 = v2 = RO.GATE_PICK_FAIL
            reading = None
        if not final:
            x["would_be"] = [v1, v2]
            x["would_be_holm"] = (f"Holm over the {len(ok_fam)} complete of the {len(fam)} domains (A3.3: the family is reduced)"
                                  if len(ok_fam) < len(fam) else f"Holm over all {len(fam)} domains (the family is complete; an outside problem stands)")
            v1, v2 = f"PROVISIONAL ({v1} if final; {x['would_be_holm']})", f"PROVISIONAL ({v2} if final)"
            reading = None
        x.update(verdict=v1, geometry=v2, rule_gain=det1, rule_geo=det2, reading=reading)
    return final


def verdict_R(R_, subs, picks, allow_partial=False, sites=None, ext_probs=()):
    """Part R from the final picks (status FINAL domains = the Holm family); every needed run at seeds 8-15 must be valid."""
    problems, rows = [], {}
    fam = [d for d, v in picks["domains"].items() if v["status"] == "FINAL"]
    for d in R_DOMS:
        v = picks["domains"][d]
        b = R_DOMS[d]
        if v["status"] != "FINAL":
            rows[d] = {"status": v["status"], "why": v["why"], "verdict": v["status"], "geometry": v["status"], "notes": v.get("notes", [])}
            continue
        sn = ([f"{(sites or {})[d]['n_shared']}/{(sites or {})[d]['n_items']} items share their site token (>= 20%): a NOT SHOWN "
               "reads 'with this site' (ptdom A1.3)"] if (sites or {}).get(d, {}).get("frac", 0) >= RO.SHARED_FRAC else [])
        x = contrast_domain(R_, subs, b, d, v["pick_prior"], v["pick_none"], R2_SEEDS, problems, v.get("notes", []) + sn)
        x.update(same_lr_2e3_seeds0_3=RO.same_lr(R_, b, d, "2e-3", SEL), tier="L",
                 same_lr_seeds8_15=RO.same_lr(R_, b, d, v["pick_prior"], R2_SEEDS) if v["pick_prior"] == v["pick_none"] else None)
        rows[d] = x
    if (problems or ext_probs) and not allow_partial:
        raise SystemExit("Part R verdict refused: " + "; ".join((list(ext_probs) + problems)[:12]))
    final = apply_rules(rows, fam, problems, ext_probs, "seeds 8-11 / 12-15")
    return {"family": fam, "holm_n": len(fam), "rows": rows, "problems": list(ext_probs) + problems, "final": final}


def verdict_S(R_, subs, allow_partial=False, ext_probs=()):
    """A2 Part S: fresh seeds 8-15, all three arms ptdom2 runs at 2e-3: PASS (only-d - none: F100 on fresh seeds) and
    GEOMETRY-SPECIFIC (only-d - only-d-shuf), the R rules, Holm over the four per contrast. Descriptive: the old pairing with
    ptdom's true arm at seeds 0-7 when part S0 has run (biased in favour of the true arm: F100 selected those seeds)."""
    problems, rows = [], {}
    lr = M2.S_LR
    for d, b in S_DOMS.items():
        N = M2.S_NONE[d]   # A3.1: F100's none pick (check_s_none)
        x = contrast_domain(R_, subs, b, d, lr, N, R2_SEEDS, problems, N_same=lr)
        src = {f"{a}@{l_}": [(R_.get((b, a, l_, s)) or {}).get("source") for s in R2_SEEDS]
               for a, l_ in dict.fromkeys(((f"only-{d}", lr), (f"only-{d}-shuf", lr), ("none", N), ("none", lr)))}
        # A4.3: only the runs that exist (a missing verdict-arm run is refused by name through need(); a missing descriptive one
        # prints "missing / invalid")
        nonp2 = {k: [f"s{s}" for s, v_ in zip(R2_SEEDS, v) if v_ is not None and v_ != "ptdom2"] for k, v in src.items()}
        if any(nonp2.values()):
            problems.append(f"{d}: Part S's arms at seeds 8-15 must be ptdom2 runs (not: {({k: v for k, v in nonp2.items() if v})})")
        if N != lr:   # A3.1 (descriptive): the same-lr PASS line (only-d@2e-3 - none@2e-3), the four conditions on the raw p
            gl = pair(R_, b, f"only-{d}", "none", lr, lr, d, R2_SEEDS, [])[0]
            if gl is None:   # A4.3: descriptive only; refuses nothing
                x["same_lr_pass"] = {"missing": x.get("ns_missing") or [f"a run of only-{d}@{lr} is missing / invalid"]}
            else:
                ok_, det_ = rule(gl, RO.sf_test(gl), "seeds 8-11 / 12-15")
                x["same_lr_pass"] = {"would_be": "PASS" if ok_ else "NOT SHOWN", "mean": det_["mean"], "wins": det_["wins"],
                                     "boot90": det_["boot90"], "p_one_sided_raw": RO.sf_test(gl),
                                     "label": f"descriptive: only-{d}@{lr} - none@{lr} (not F100's contrast: F100's none pick is {N}; raw p, no Holm)"}
        old = pair(R_, b, f"only-{d}", f"only-{d}-shuf", lr, lr, d, M2.S_SEEDS, [])[0]
        x["seeds0_7_selected"] = None if old is None else {
            "mean": float(np.mean(old)), "wins": int((old > 0).sum()), "per_seed": [round(float(v), 2) for v in old],
            "p_one_sided_raw": RO.sf_test(old),
            "label": "the true arm's seeds were selected by F100 (0-3 chose the lr, 4-7 decided the PASS): biased in favour of true"}
        rows[d] = x
    if (problems or ext_probs) and not allow_partial:
        raise SystemExit("Part S verdict refused: " + "; ".join((list(ext_probs) + problems)[:12]))
    final = apply_rules(rows, list(S_DOMS), problems, ext_probs, "seeds 8-11 / 12-15")
    return {"family": list(S_DOMS), "holm_n": len(S_DOMS), "rows": rows, "problems": list(ext_probs) + problems, "final": final}


# ------------------------------------------------------------------------------------------------------------ report
f1 = lambda v: "-" if v is None else f"{v:.1f}"
s1 = lambda v: "-" if v is None else f"{v:+.2f}"
mn = lambda a: None if a is None else float(np.mean(a))
wn = lambda a: None if a is None else f"{int((np.asarray(a) > 0).sum())}/{len(a)}"
f3_ = lambda v: "-" if v is None else f"{v:.3f}"


def rule_line(det):
    c = det["conditions"]
    return (f"mean {det['mean']:+.2f}, boot90 [{det['boot90'][0]:+.2f}, {det['boot90'][1]:+.2f}], halves {det['halves'][0]:+.2f} / "
            f"{det['halves'][1]:+.2f} ({det['halves_seeds']}), wins {det['wins']}/8, one-sided t p {det['p_one_sided']:.4f} Holm "
            f"{det['p_holm']:.4f} (two-sided p {det['t_two_sided']['p_two_sided']:.4f}); conditions "
            + ", ".join(f"{k} {'ok' if v else 'NO'}" for k, v in c.items()))


def domain_lines(d, x, R=True):
    """The lines of one FINAL / S domain (never fails on missing pieces: --allow_partial, A2.4)."""
    P, N = x["pick_prior"], x["pick_none"]
    L = [f"  {d:11s} [{'Tier L' if d in COMP else 'k-step'}] batch {x['batch']}: only-{d}@{P} vs none@{N}: {x.get('verdict')}; "
         f"vs only-{d}-shuf@{P}: {x.get('geometry')}"]
    if x.get("rule_gain"):
        L.append(f"      gain vs none:      {rule_line(x['rule_gain'])}")
        L.append(f"      gain vs shuffled:  {rule_line(x['rule_geo'])}")
        L.append(f"      per seed (8..15): vs none {[round(float(v), 2) for v in x['gain']]}; vs shuffled "
                 f"{[round(float(v), 2) for v in x['geo_gain']]}; areas true {f1(mn(x['prior']))}, none {f1(mn(x['none']))}, "
                 f"shuffled {f1(mn(x['shuf']))}")
    if x.get("reading"):
        L.append(f"      reading (pre-registered, A3.2 / A4): {x['reading']}")
    MI = "missing / invalid"   # A4.3: a line whose descriptive none cell (Part S: A's none@2e-3) lacks a valid run
    nsm = x.get("ns_missing")
    if nsm:
        L.append(f"      A4.3: none@{x.get('shuf_none_lr')} (descriptive only; refuses nothing) {MI}: {'; '.join(nsm)}")
    sn = x.get("shuf_none")
    sn_ci = RO.boot_ci(np.asarray(sn, float)) if sn is not None else None
    L.append(f"      shuffled - none@{x.get('shuf_none_lr')} (descriptive, A2.2{'; lr-confounded: P != N' if x.get('lr_confounded_shuf_none') else ''}): "
             + (f"area {s1(mn(sn))} ({wn(sn)}), boot90 [{sn_ci[0]:+.2f}, {sn_ci[1]:+.2f}], final3 {s1(mn(x.get('shuf_none_final3')))}"
                if sn is not None else (MI if nsm else "-")))
    if x.get("same_lr_pass"):
        q = x["same_lr_pass"]
        if "missing" in q:
            L.append(f"      same-lr PASS line (descriptive: only-{d}@{P} - none@{x.get('shuf_none_lr')}): {MI} ({'; '.join(q['missing'][:4])})")
        else:
            L.append(f"      same-lr PASS line ({q['label']}): {q['would_be']}; mean {q['mean']:+.2f}, wins {q['wins']}/8, boot90 "
                     f"[{q['boot90'][0]:+.2f}, {q['boot90'][1]:+.2f}], one-sided t p (raw) {q['p_one_sided_raw']:.4f}")
    sp = x.get("split") or {}
    L.append("      window split of the area gain (evals at steps 916-7323 / the others; in + out = the gain): " + "; ".join(
        f"{k} {s1(mn(v['in']))} / {s1(mn(v['out']))}" if v else f"{k} {MI if (nsm and k == 'shuffled vs none') else '-'}" for k, v in sp.items()))
    L.append("      LM share a / (a + b), median of the cell (seeds 8-15): " + "; ".join(f"{k} {f3_(v)}" for k, v in (x.get("lm_share") or {}).items()))
    L.append(f"      descriptive: final3 gain vs none {s1(mn(x.get('gain_final3')))} ({wn(x.get('gain_final3'))}), vs shuffled "
             f"{s1(mn(x.get('geo_final3')))} ({wn(x.get('geo_final3'))}); hard-subset final3 vs none {s1(mn(x.get('gain_hard3')))}, vs "
             f"shuffled {s1(mn(x.get('geo_hard3')))}"
             + (f"; same-lr gain @2e-3 (ptdom seeds 0-3) {[round(v, 2) if v is not None else None for v in x['same_lr_2e3_seeds0_3']]}"
                if x.get("same_lr_2e3_seeds0_3") is not None else "")
             + (f"; same lr @{P} on seeds 8-15 {s1(mn([v for v in x['same_lr_seeds8_15'] if v is not None]))}" if x.get("same_lr_seeds8_15") else ""))
    fl, fg = x.get("floor"), x.get("floor_geo")
    L.append(f"      FLOOR / CEILING (true vs none): {(fl['tags'] or 'none') if fl else '-'}{(' (line %.1f)' % fl['line']) if fl else ''}; "
             f"(true vs shuffled): {(fg['tags'] or 'none') if fg else '-'}")
    if x.get("shortcut"):
        L.append("      shortcut baseline on the pooled hard subset vs each arm (descriptive; not a condition): " + "; ".join(
            (f"{a}: {f1(v['prior'])} vs baseline {f1(v['base'])} (n {v['n_pooled']}{'' if v['testable'] else ', < 30'})" if v else f"{a}: {MI}")
            for a, v in x["shortcut"].items()))
    if x.get("gate"):
        L.append("      aux gate (a / (a + b) median >= 0.5 per cell): " + "; ".join(
            f"{k} {f3_(g['median'])}{'' if g['ok'] else ' FAIL'}" for k, g in x["gate"].items()))
    if x.get("gate_failed"):
        L.append(f"      {RO.GATE_PICK_FAIL} (A1.7): {'; '.join(x['gate_failed'])} -- without the gate: {x['would_be_without_gate']}")
    sc = x.get("shuf_cka_to_true")
    L.append(f"      shuffled target vs true (shuf_cka_to_true over the 8 seeds): {('%.3f .. %.3f' % tuple(sc)) if sc else '-'}")
    ck = x.get("cka_check")
    if ck:
        L.append(f"      double path (A2.4, descriptive): the shuffled arm's eval CKA to the TRUE target at the window's last eval <= none + 0.10 "
                 f"and <= true - 0.2 in {ck['n_ok']}/{ck['n']} seeds (shuffled {ck['shuf']}; none {ck['none']}; true {ck['true']})")
    elif nsm:
        L.append(f"      double path (A2.4, descriptive; needs none@{x.get('shuf_none_lr')}): {MI}")
    if x.get("diverged"):
        L.append(f"      DIVERGED runs (valid data, the area counts): {', '.join(x['diverged'])}")
    if x.get("seeds0_7_selected"):
        q = x["seeds0_7_selected"]
        L.append(f"      seeds 0-7 (ptdom's true arm vs part S0's shuffled arm; descriptive; {q['label']}): mean {q['mean']:+.2f}, "
                 f"wins {q['wins']}/8, one-sided t p (raw) {q['p_one_sided_raw']:.4f}")
    if x.get("notes"):
        L.append("      notes: " + "; ".join(x["notes"]))
    return L


def report(oR=None, oS=None, picks=None, consistency=None, probs=()):
    L = ["ptdom2 read-out (pre-registered: runs/ptdom2/AUDIT.md + Amendments A1, A2, A3, A4; area = primary, final3 secondary; the rules are "
         "readout_ptdom's statistics; verdict bars: bootstrap 90% lower bound > 0, both halves > 0, one-sided t Holm p < 0.10, >= 6/8 seeds)"]
    if probs:   # A2.4: shown, and no final verdict word while they stand
        L.append(f"PROBLEMS ({len(probs)}; every verdict PROVISIONAL): " + "; ".join(list(probs)[:12]))
    if consistency:
        L.append("picks consistency: " + "; ".join(consistency))
    for nm, o in (("R", oR), ("S", oS)):
        if not o:
            continue
        L.append(f"PART {nm} (seeds 8-15; Holm family = the {o['holm_n']} domains {o['family']})" + ("" if o["final"] else
                 "  PROVISIONAL: " + "; ".join(o["problems"][:6])))
        for d, x in o["rows"].items():
            if x.get("status") != "FINAL":
                L.append(f"  {d:11s} {x['status']} (no verdict, not in the family): {'; '.join(x['why'])}"
                         + (f"  notes: {'; '.join(x['notes'])}" if x.get("notes") else ""))
                continue
            L += domain_lines(d, x)
    L.append("Reading: PASS = the prior helps the held-out read-out vs none on fresh seeds; GEOMETRY-SPECIFIC = the TRUE target beats the "
             "dose-matched shuffled target (same items, sites, window, lambda, lr; only the item -> coordinate correspondence permuted); "
             "a domain's reading line states the geometry dependence only when PASS, GEOMETRY-SPECIFIC and every A3.2 / A4 condition hold "
             "(otherwise it names the failing ones). Comparison domains are Tier L (the target carries the "
             "answers). Tied embeddings: for a single-token item the prior's site is the read-out candidate's own output row; both arms "
             "move those rows, but only the true arm arranges them in the answer geometry, so a GEOMETRY-SPECIFIC there cannot tell "
             "'the model used the geometry' from 'the answer geometry was written into the output rows'.")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------ self-test
def fake2(root, prefix, batch, arm, lr, seed, area, final3=None, hard3=None, share=0.9, chk_fail=(), evals=N_EVALS, dsha=None, lm0=None,
          ident=None, bpb=1.2, win_delta=None, cka=None, pev=None):
    """A synthetic run dir that load2 accepts (area: {domain: value}); ident: overrides of args (identity tamper); pev: the run's
    prior_events (A4.4 window tamper; default the trainer's)."""
    doms = BATCHES[batch]
    t = tag_of(prefix, batch, arm, lr)
    d = Path(root) / f"{t}-s{seed}"
    (d / t).mkdir(parents=True, exist_ok=True)
    gp = 1.0 / share - 1.0
    on = arm != "none"
    win = lambda i: on and 3 < i < 30
    lg = 1.0 / math.sqrt(1.0 + gp * gp)
    wd = win_delta or {}
    ser = lambda x, i: (area[x] + (wd.get(x, 0.0) if 3 < i < 30 else -wd.get(x, 0.0) * 26 / 49)) / 100   # 100 x mean = area exactly
    curve = [{"step": min(250 * i, 18309), "lm": 11.6 + 1e-4 * seed if lm0 is None else lm0, "val_loss": 4.0, "geo": 0.0,
              "data_sha": (dsha[i] if dsha else f"sha{batch}{seed}-{i}"), "val_bits_per_byte": bpb, "prior_on": int(win(i)),
              "grad_lm_share": lg if win(i) else 1.0, "clip_coef": 0.5 if win(i) else 1.0, "grad_lm": 1.0,
              "grad_prior": gp if win(i) else None, "grad_cos": 0.0 if win(i) else None,
              **{RO.EVAL_KEY[x]: ser(x, i) for x in doms}, **{f"cka_{x}": (cka or {}).get(x) for x in doms}} for i in range(evals)]
    f3 = final3 or area
    met = {READ[x]: {"area": area[x], "final3": f3[x]} for x in doms}
    met.update({RO.READ1[x]: {"area": area[x] - 1.0, "final3": f3[x] - 1.0} for x in doms if x in KSTEP})
    met.update({SEEN[x]: {"final3": 99.0} for x in doms}, numbers={"area": 20.0})
    met.update({f"{x}_readout_hard": {"area": area[x] - 5.0, "final3": (hard3 or {}).get(x, f3[x] - 3.0)} for x in doms if x in COMP})
    dm_ = M2.dom(arm)
    dmeta = {x: {"chance_readout": (1 / 12 if x in KSTEP else 0.5), "heuristic_ansfreq": 0.55, "heuristic_shorter": 0.52, "n_hard": 40,
                 "split_seed": 7, "prior": x == dm_, "shuf": x == dm_ and M2.is_shuf(arm), "shuf_cka_to_true": 0.05} for x in doms}
    dmeta["numbers"] = {"prior": False, "shuf": False}
    args = {"lr": float(lr), "seed": seed, "tag": f"_{t}", "prior_domains": dm_ or "", "prior_shuf": dm_ if M2.is_shuf(arm) else ""}
    args.update(ident or {})
    f = d / t / f"run_whole_none_seed{seed}_{t}.json"
    f.write_text(json.dumps({"code_sha": "c0de", "args": args, "curve": curve, "metrics": met, "domains": dmeta,
                             "prior_events": (pev if pev is not None else [[916, 1], [7324, 0]]) if on else []}))
    checks = [{"check": "run args = intent", "ok": True}] + [{"check": c, "ok": False} for c in chk_fail]
    (d / "check.json").write_text(json.dumps({"pass": not chk_fail, "checks": checks}))
    return d


def plant_subs2(R_, h3=60.0, h3_hard=40.0):
    """Planted sub-read-outs; h3: a value or {domain: value} (default 60)."""
    hv = lambda d: h3.get(d, 60.0) if isinstance(h3, dict) else h3
    return {k: {d: {"h3_held": hv(d), "h3_hard": h3_hard} for d in BATCHES[k[0]] if d in COMP} for k, r in R_.items() if "area" in r}


def selftest(rng_seed=1):
    """rng_seed (A4.1): the planted worlds' noise; every known answer must hold for any seed (tests/selftest_rngs.py runs 1-10):
    per-arm noise is zero-mean over the 8 seeds (zm), so a planted effect is the EXACT mean of its contrast, and the final3
    controls of the A3.2 / A4 reading are exact (no noise)."""
    ok = []

    def T(name, cond):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name, flush=True)

    rng = np.random.default_rng(rng_seed)
    zm = lambda sd: (lambda v: v - v.mean())(rng.normal(0.0, sd, 8))   # zero-mean noise over the 8 seeds 8-15
    base_area = {d: 70.0 for b in BATCHES for d in BATCHES[b]}

    def world(tmp, prior_fn, none_fn, shuf_fn, s_shuf_fn=None, skip=(), gate_fail=(), r1=True, r1x=(), r2=True, nfix=None, bpb_fn=None,
              r1x_fn=None):
        """A synthetic ptdom (seeds 0-7 of the re-used cells) + ptdom2 world. prior_fn(d, lr, s) etc. give d's area."""
        r0, r2_ = Path(tmp) / "p0", Path(tmp) / "p2"
        for d, b in R_DOMS.items():
            for lr in M2.PRIOR_LRS0[d]:
                for s in SEL:
                    a = dict(base_area); a[d] = prior_fn(d, lr, s)
                    fake2(r0, "ptdom", b, f"only-{d}", lr, s, a, share=0.3 if (d, lr) in gate_fail else 0.9,
                          bpb=(bpb_fn or (lambda d_, l_: 1.2))(d, lr))
            if r1:
                for s in SEL:
                    a = dict(base_area); a[d] = prior_fn(d, M2.R1_LR, s)
                    if (d, M2.R1_LR, s) not in skip:
                        fake2(r2_, "ptdom2", b, f"only-{d}", M2.R1_LR, s, a, share=0.3 if (d, M2.R1_LR) in gate_fail else 0.9,
                              bpb=(bpb_fn or (lambda d_, l_: 1.2))(d, M2.R1_LR))
        for b, lrs in M2.NONE_LRS0.items():
            for lr in lrs:
                for s in SEL:
                    a = dict(base_area); a.update({d: none_fn(d, lr, s) for d in BATCHES[b] if d in R_DOMS})
                    fake2(r0, "ptdom", b, "none", lr, s, a)
        for b, a_, lr in r1x:
            for s in SEL:
                a = dict(base_area)
                a.update({d: (r1x_fn or (none_fn if a_ == "none" else prior_fn))(d, lr, s) for d in BATCHES[b] if d in R_DOMS})
                fake2(r2_, "ptdom2", b, a_, lr, s, a)
        return r0, r2_

    # ---- 1. Holm / statistics known answers
    T("holm = the step-down rule on a hand case", np.allclose(RO.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
    g = np.array([5.0, 6, 4, 5, 6, 5, 4, 6])
    okr, det = rule(g, 0.001, "")
    T("rule: a known positive effect (+5 +- 1 on 8 seeds) -> all four conditions", okr and det["wins"] == 8)
    okr, det = rule(np.array([1.0, -1, 0.5, -0.5, 1, -1, 0.3, -0.3]), RO.sf_test(np.array([1.0, -1, 0.5, -0.5, 1, -1, 0.3, -0.3])), "")
    T("rule: zero effect -> fails (bootstrap / Holm / wins)", not okr)
    gh = np.array([3.0, 4, 3, 4, -2, -1, -2, 5])
    okr, det = rule(gh, 0.01, "")
    T("rule: second half mean <= 0 -> fails on the halves alone", not okr and not det["conditions"]["halves_gt0"] and det["conditions"]["boot90_lo_gt0"] is not None)
    gw = np.array([10.0, 10, 10, 10, 10, -0.1, -0.1, -0.1])
    okr, det = rule(gw, 0.001, "")
    T("rule: 5/8 wins (big positive values) -> fails on wins alone", not okr and det["wins"] == 5 and det["conditions"]["boot90_lo_gt0"]
      and det["conditions"]["halves_gt0"])
    gb = np.array([0.5, 0.4, -0.3, 0.6, 0.2, -0.4, 0.5, 0.1])
    lo, _ = RO.boot_ci(gb)
    okr, det = rule(gb, 0.001, "")
    T(f"rule: bootstrap lower bound {lo:+.3f} -> fails when <= 0", (lo <= 0) == (not det["conditions"]["boot90_lo_gt0"]))
    p_raw = [0.03, 0.2, 0.5, 0.6, 0.7, 0.8]
    T("Holm over 6: raw p 0.03 -> 0.18 >= 0.10 (fails), while raw 0.03 alone would pass", RO.holm(p_raw)[0] > ALPHA > p_raw[0])
    gg = np.array([4.0, 5, 3, 4, 5, 4, 3, 5])
    T("sf_test = the one-sided t (scipy) on a hand case", abs(RO.sf_test(gg) - __import__("scipy").stats.ttest_1samp(gg, 0, alternative="greater").pvalue) < 1e-12)

    # ---- 2. R1 selection (planted worlds; AUDIT.md Amendment A1)
    flat = lambda d, lr, s: 70.0 + 0.1 * s
    none_mid = lambda d, lr, s: 60.0 + (5.0 if lr == "4e-3" else 0.0)
    E_ = lambda p: sorted(tuple(x) for x in p["E"])
    with tempfile.TemporaryDirectory() as tmp:
        peak4 = lambda d, lr, s: 70.0 + (8.0 if lr == "4e-3" else 0.0) + 0.1 * s
        r0, r2_ = world(tmp, peak4, none_mid, None)
        R_ = load_all(r0, r2_, guard=False)
        p1 = decide_R(R_, "R1", ())
        T("R1: the prior's pick and argmax 4e-3 -> only-<d>@8e-3 asked for every domain (R1x); none inside -> no none ask; all PENDING",
          p1["stage_written"] == "R1x" and E_(p1) == sorted((b, f"only-{d}", "8e-3") for d, b in R_DOMS.items())
          and all(v["status"] == "PENDING" for v in p1["domains"].values()))
        for d, b in R_DOMS.items():
            for s in SEL:
                a = dict(base_area); a[d] = 60.0
                fake2(r2_, "ptdom2", b, f"only-{d}", "8e-3", s, a)
        R_ = load_all(r0, r2_, guard=False)
        p2 = decide_R(R_, "R1x", ())
        T("R1x: 8e-3 below 4e-3 -> every domain FINAL at prior 4e-3 / none 4e-3; C = only-d@4e-3, only-d-shuf@4e-3, none@4e-3 per batch",
          p2["stage_written"] == "R2" and all(v["status"] == "FINAL" and v["pick_prior"] == "4e-3" and v["pick_none"] == "4e-3"
                                              for v in p2["domains"].values())
          and sorted(tuple(x) for x in p2["C"]) == sorted({(b, f"only-{d}", "4e-3") for d, b in R_DOMS.items()} |
                                                          {(b, f"only-{d}-shuf", "4e-3") for d, b in R_DOMS.items()} |
                                                          {(b, "none", "4e-3") for b in set(R_DOMS.values())}))
        for s in SEL:
            a = dict(base_area); a["stars"] = 95.0
            fake2(r2_, "ptdom2", "A", "only-stars", "8e-3", s, a)
        R_ = load_all(r0, r2_, guard=False)
        p3 = decide_R(R_, "R1x", ())
        T("R1x: the prior's argmax at the extension 8e-3 -> NOT FINAL (stars), not in the family; the others FINAL",
          p3["domains"]["stars"]["status"] == "NOT FINAL" and "stars" not in p3["family"] and len(p3["family"]) == 5
          and ["A", "only-stars", "4e-3"] not in p3["C"])
    with tempfile.TemporaryDirectory() as tmp:   # A1.2: the PICK at 4e-3 (bpb tie-break), the argmax at 2e-3
        tie = lambda d, lr, s: 70.0 + {"2e-3": 1.0, "4e-3": 0.0}.get(lr, -10.0)
        r0, r2_ = world(tmp, tie, none_mid, None, bpb_fn=lambda d, lr: 1.1 if lr == "4e-3" else 1.2)
        p1 = decide_R(load_all(r0, r2_, guard=False), "R1", ())
        T("A1.2: the prior's PICK at its top lr 4e-3 by the bpb tie-break (argmax 2e-3) -> only-<d>@8e-3 asked",
          all(v["pick_prior"] == "4e-3" and v["argmax_prior"] == "2e-3" and v["ext_prior"] == "8e-3" for v in p1["domains"].values()))
    with tempfile.TemporaryDirectory() as tmp:   # none rising with lr; flat prior (pick = its lowest lr)
        none_top = lambda d, lr, s: 60.0 + 3.0 * math.log2(float(lr) / 1e-3)
        r0, r2_ = world(tmp, flat, none_top, None)
        R_ = load_all(r0, r2_, guard=False)
        p1 = decide_R(R_, "R1", ())
        want = [("A", "none", "1.6e-2"), ("B2", "none", "3.2e-2"), ("B2", "none", "5e-4"), ("B3", "none", "1e-3"), ("B3", "none", "1.6e-2")]
        T("R1: none's argmax at its batch's top -> one upward step per batch (A 1.6e-2, B2 3.2e-2, B3 1.6e-2); fixB (A1.1b): the prior "
          "pick below none's range -> none at the pick (ranks 5e-4 in B2, scalar_adj 1e-3 in B3); A's prior pick 1e-3 = its lowest: none",
          sorted(x for x in E_(p1) if x[1] == "none") == sorted(want) and p1["domains"]["ranks"]["fixb"] == "5e-4"
          and p1["domains"]["scalar_adj"]["fixb"] == "1e-3" and p1["domains"]["stars"]["fixb"] is None)
        T("A1.10: none_asks_from_ptdom on this world = the none up / down steps from the ptdom cells alone (no fixB, no ptdom2 run)",
          none_asks_from_ptdom(R_) == [("A", "none", "1.6e-2"), ("B2", "none", "3.2e-2"), ("B3", "none", "1.6e-2")])
        r0b, r2b = world(tmp + "/x", flat, none_top, None, r1x=want)
        pb = decide_R(load_all(r0b, r2b, guard=False), "R1x", ())
        T("R1x: none still rising at the extension -> NOT FINAL for every R domain (pick / argmax at the outermost lr after its one step)",
          all(v["status"] == "NOT FINAL" for v in pb["domains"].values()) and pb["stage_written"] is None and pb["family"] == [])
        Ry = load_all(*world(tmp + "/y", flat, none_top, None, skip={("compass", "4e-3", 2)}), guard=False)
        T("R1 refuses a missing selection cell (only-compass@4e-3 s2 absent)", RO._raises(lambda: decide_R(Ry, "R1", ()), "lacks valid runs"))
        T("R1 refuses when the R1pre cells have not run (the default pre)", RO._raises(lambda: decide_R(Ry, "R1"), "R1pre cells lack valid runs"))
    with tempfile.TemporaryDirectory() as tmp:   # A1.1a: none falling with lr (argmax at the lowest)
        none_low = lambda d, lr, s: 60.0 - 3.0 * math.log2(float(lr) / 1e-3)
        r0, r2_ = world(tmp, flat, none_low, None)
        p1 = decide_R(load_all(r0, r2_, guard=False), "R1", ())
        want = [("A", "none", "5e-4"), ("B2", "none", "5e-4"), ("B3", "none", "1e-3")]
        T("A1.1a: none's argmax at its batch's lowest lr -> one downward step (x1/2) per batch; fixB coincides with it (B2 5e-4, B3 1e-3; no duplicate)",
          sorted(x for x in E_(p1) if x[1] == "none") == sorted(want))
        r0b, r2b = world(tmp + "/x", flat, none_low, None, r1x=want)
        pb = decide_R(load_all(r0b, r2b, guard=False), "R1x", ())
        T("R1x: none still falling at the lower step -> NOT FINAL (the lower edge after its one step)",
          all(v["status"] == "NOT FINAL" and any("lowest" in w for w in v["why"]) for v in pb["domains"].values()))
        bump = lambda d, lr, s: none_low(d, lr, s) - (20.0 if lr in ("5e-4",) or (lr == "1e-3" and d == "scalar_adj") else 0.0)
        r0c, r2c = world(tmp + "/y", flat, none_low, None, r1x=want, r1x_fn=bump)
        pc = decide_R(load_all(r0c, r2c, guard=False), "R1x", ())
        T("R1x: the lower step worse than the old lowest -> the none argmax back inside, FINAL (prior pick inside none's range)",
          all(v["status"] == "FINAL" for d, v in pc["domains"].items()))
    with tempfile.TemporaryDirectory() as tmp:   # the default pre = R1PRE run with R1
        r0, r2_ = world(tmp, lambda d, lr, s: 70.0 + (5.0 if lr == "2e-3" else 0.0), none_mid, None, r1x=list(M2.R1PRE))
        p1 = decide_R(load_all(r0, r2_, guard=False), "R1")
        T("R1 with R1pre run (A 1.6e-2, B2 5e-4 in the grids, not asked again): no edge -> FINAL at R1 (stage R2, no R1x)",
          p1["stage_written"] == "R2" and not p1["E"] and all(v["status"] == "FINAL" for v in p1["domains"].values())
          and "1.6e-2" in p1["domains"]["stars"]["none_lrs"] and "5e-4" in p1["domains"]["compass"]["none_lrs"])
        r0b, r2b = world(tmp + "/x", lambda d, lr, s: 70.0 + (5.0 if lr == "2e-3" else 0.0),
                         lambda d, lr, s: 60.0 + (9.0 if lr == "1.6e-2" else 0.0), None, r1x=list(M2.R1PRE))
        pb = decide_R(load_all(r0b, r2b, guard=False), "R1")
        T("R1 with R1pre: A's none argmax at 1.6e-2 (its up step already used by R1pre) -> NOT FINAL at once for stars / landmarks",
          pb["domains"]["stars"]["status"] == "NOT FINAL" and pb["domains"]["landmarks"]["status"] == "NOT FINAL"
          and not any(x[0] == "A" and x[1] == "none" for x in pb["E"]))
    with tempfile.TemporaryDirectory() as tmp:
        peak4 = lambda d, lr, s: 70.0 + (8.0 if lr == "4e-3" else 0.0)
        r0, r2_ = world(tmp, peak4, none_mid, None,
                        gate_fail={("compass", "4e-3")} | {("ranks", lr) for lr in M2.PRIOR_LRS0["ranks"] + ("4e-3",)})
        R_ = load_all(r0, r2_, guard=False)
        p1 = decide_R(R_, "R1", ())
        T("gate: a failing cell is excluded (compass 4e-3 cannot be the pick: its edge is closed by the gate, no 8e-3 ask); every cell "
          "failing -> GATE (ranks), and no HOLD (A1.8)",
          p1["domains"]["compass"]["pick_prior"] != "4e-3" and p1["domains"]["compass"]["ext_prior"] is None
          and p1["domains"]["ranks"]["status"] == "GATE" and "4e-3" in p1["domains"]["compass"]["gate_excluded"])

    # ---- 3. Part R verdicts (FINAL picks planted)
    def r2_world(tmp, eff_none, eff_shuf, missing=(), noise=True, gate_bad=()):
        r0, r2_ = world(tmp, lambda d, lr, s: 70.0 + (8.0 if lr == "4e-3" else 0.0), lambda d, lr, s: 60.0 + (5.0 if lr == "4e-3" else 0.0), None)
        for d, b in R_DOMS.items():
            for s in SEL:
                a = dict(base_area); a[d] = 60.0
                fake2(r2_, "ptdom2", b, f"only-{d}", "8e-3", s, a)
        zv = {d: {k: (zm(sd) if noise else np.zeros(8)) for k, sd in (("p", 1.0), ("n", 0.3), ("s", 0.3))} for d in R_DOMS}
        for d, b in R_DOMS.items():
            for i, s in enumerate(R2_SEEDS):
                a_p = dict(base_area); a_p[d] = 80.0 + zv[d]["p"][i]
                a_s = dict(base_area); a_s[d] = 80.0 - eff_shuf(d, i) + zv[d]["s"][i]   # contrasts: effect + z_p - z_s (sd ~1.04, exact mean)
                if (d, "prior", s) not in missing:
                    fake2(r2_, "ptdom2", b, f"only-{d}", "4e-3", s, a_p, hard3={d: 50.0}, share=0.3 if (d, s) in gate_bad else 0.9)
                fake2(r2_, "ptdom2", b, f"only-{d}-shuf", "4e-3", s, a_s, hard3={d: 48.0})
                if d in ("stars", "compass", "scalar_adj"):   # one none cell per batch (shared): its first domain plants it
                    a_n2 = dict(base_area)
                    a_n2.update({x: 80.0 - eff_none(x, i) + zv[x]["n"][i] for x in BATCHES[b] if x in R_DOMS})
                    fake2(r2_, "ptdom2", b, "none", "4e-3", s, a_n2)
        R_ = load_all(r0, r2_, guard=False)
        return R_, decide_R(R_, "R1x", ())
    with tempfile.TemporaryDirectory() as tmp:
        R_, pk = r2_world(tmp, lambda d, i: 6.0, lambda d, i: 6.0)
        o = verdict_R(R_, plant_subs2(R_), pk)
        T("R verdict: a known +6 effect vs none and vs shuffled on seeds 8-15 -> PASS and GEOMETRY-SPECIFIC for all 6 (Holm over 6)",
          all(o["rows"][d]["verdict"] == "PASS" and o["rows"][d]["geometry"] == "GEOMETRY-SPECIFIC" for d in R_DOMS) and o["holm_n"] == 6)
    with tempfile.TemporaryDirectory() as tmp:
        R_, pk = r2_world(tmp, lambda d, i: 6.0, lambda d, i: 0.0)
        o = verdict_R(R_, plant_subs2(R_), pk)
        T("R verdict: true = shuffled (zero geometry effect), +6 vs none -> PASS but NOT SHOWN (not geometry-specific)",
          all(o["rows"][d]["verdict"] == "PASS" and o["rows"][d]["geometry"] == "NOT SHOWN" for d in R_DOMS))
    with tempfile.TemporaryDirectory() as tmp:
        R_, pk = r2_world(tmp, lambda d, i: 0.0, lambda d, i: 0.0)
        o = verdict_R(R_, plant_subs2(R_), pk)
        T("R verdict: zero effect -> NOT SHOWN / NOT SHOWN", all(o["rows"][d]["verdict"] == "NOT SHOWN" and o["rows"][d]["geometry"] == "NOT SHOWN" for d in R_DOMS))
    with tempfile.TemporaryDirectory() as tmp:
        R_, pk = r2_world(tmp, lambda d, i: 6.0 if i < 4 else -2.0, lambda d, i: 6.0)
        o = verdict_R(R_, plant_subs2(R_), pk)
        T("R verdict: seeds 12-15 negative -> NOT SHOWN (the halves)", all(o["rows"][d]["verdict"] == "NOT SHOWN" for d in R_DOMS)
          and all(not o["rows"][d]["rule_gain"]["conditions"]["halves_gt0"] for d in R_DOMS))
    with tempfile.TemporaryDirectory() as tmp:
        R_, pk = r2_world(tmp, lambda d, i: 6.0, lambda d, i: 6.0, missing={("ranks", "prior", 11)})
        T("R verdict refused while a needed run is missing (ranks only-ranks@4e-3 s11)", RO._raises(lambda: verdict_R(R_, plant_subs2(R_), pk), "refused"))
        o = verdict_R(R_, plant_subs2(R_), pk, allow_partial=True)
        T("R verdict with --allow_partial: PROVISIONAL, no verdict word for ranks; the others PROVISIONAL too (family incomplete)",
          o["rows"]["ranks"]["verdict"].startswith("PROVISIONAL") and all(o["rows"][d]["verdict"].startswith("PROVISIONAL") for d in R_DOMS)
          and not o["final"])
    weak = [2.6, 0.4, 1.8, -0.6, 2.2, 0.3, 1.5, -0.2]   # mean 1.0, one-sided t p ~0.02: passes alone, fails Holm x 6
    with tempfile.TemporaryDirectory() as tmp:
        R_, pk = r2_world(tmp, lambda d, i: weak[i] if d == "stars" else -1.0 + 0.5 * (i % 2), lambda d, i: weak[i] if d == "stars" else 0.0, noise=False)
        o = verdict_R(R_, plant_subs2(R_), pk)
        x = o["rows"]["stars"]
        T(f"R verdict applies Holm over the family: stars raw one-sided p {x['rule_gain']['p_one_sided']:.3f} < 0.10 with the other three "
          f"conditions met, Holm p {x['rule_gain']['p_holm']:.3f} >= 0.10 -> NOT SHOWN (and NOT SHOWN for the geometry gain alike)",
          x["rule_gain"]["p_one_sided"] < ALPHA <= x["rule_gain"]["p_holm"] and x["verdict"] == "NOT SHOWN" and x["geometry"] == "NOT SHOWN"
          and all(v for k, v in x["rule_gain"]["conditions"].items() if k != "p_holm_lt"))
    with tempfile.TemporaryDirectory() as tmp:   # A1.7: the gate at the pick on seeds 8-15, per half
        R_, pk = r2_world(tmp, lambda d, i: 6.0, lambda d, i: 6.0, gate_bad={("stars", s) for s in (12, 13, 14, 15)} | {("compass", 9)})
        o = verdict_R(R_, plant_subs2(R_), pk)
        T("A1.7: the pick cell failing the gate on seeds 12-15 (stars) -> 'AUX GATE FAILED at the pick' instead of PASS and of "
          "GEOMETRY-SPECIFIC, still in the Holm family; one low seed of 4 (compass s9, the half's median passes) -> PASS",
          o["rows"]["stars"]["verdict"] == RO.GATE_PICK_FAIL == o["rows"]["stars"]["geometry"] and o["holm_n"] == 6
          and o["rows"]["stars"]["would_be_without_gate"] == ["PASS", "GEOMETRY-SPECIFIC"] and o["rows"]["compass"]["verdict"] == "PASS"
          and o["rows"]["stars"]["reading"] is None and o["rows"]["compass"]["reading"] == GEOM_OK)
    with tempfile.TemporaryDirectory() as tmp:   # a NOT FINAL domain leaves the family
        R_, pk = r2_world(tmp, lambda d, i: 6.0, lambda d, i: 6.0)
        pk["domains"]["stars"]["status"] = "NOT FINAL"
        pk["domains"]["stars"]["why"] = ["planted"]
        pk["family"] = [d for d in R_DOMS if d != "stars"]
        o = verdict_R(R_, plant_subs2(R_), pk)
        T("R verdict: a NOT FINAL domain gets no verdict and leaves the Holm family (5)", o["rows"]["stars"]["verdict"] == "NOT FINAL" and o["holm_n"] == 5)

    # ---- 4. Part S (A2 / A3: fresh seeds 8-15; true / shuffled at 2e-3; none at F100's pick (B1 2e-3, A 4e-3) + A's none@2e-3)
    def s_world(tmp, spec, s0=False, drop=None, ptdom_arm=None, bad=None):
        """spec[d] = dict(base = none@N's area, true / shuf = area offsets vs none@N (exact means: zero-mean per-arm noise),
        f3 / f3_s = the true / shuffled arm's final3 offset vs none@N's final3, EXACT per seed (a number or 8 values; default:
        final3 = the arm's area), none2 = A's none@2e-3 offset vs none@4e-3, share_t / share_s = the LM shares, cka_s = the
        shuffled arm's eval CKA to the true target). drop / bad: a (batch, 'none', lr, seed) left out / planted invalid."""
        r0, r2_ = Path(tmp) / "p0", Path(tmp) / "p2"
        sp = {d: dict(dict(base=80.0, true=0.0, shuf=0.0, none2=0.0, share_t=0.9, share_s=0.9, cka_s=0.45), **spec.get(d, {})) for d in S_DOMS}
        for s in M2.S_SEEDS:
            for d, b in S_DOMS.items():
                a = dict(base_area); a[d] = sp[d]["base"] + 6.0
                fake2(r0, "ptdom", b, f"only-{d}", "2e-3", s, a)
                if s0:
                    a2 = dict(base_area); a2[d] = sp[d]["base"]
                    fake2(r2_, "ptdom2", b, f"only-{d}-shuf", "2e-3", s, a2)
        za = {(d, k): zm(0.3) for d in S_DOMS for k in ("true", "shuf")}
        ex = lambda v, i: float(v[i] if isinstance(v, (list, tuple)) else v)
        for i, s in enumerate(R2_SEEDS):
            nz = float(rng.normal(0, 0.3))   # shared by every arm of the seed (cancels in every contrast)
            for b, lr in (("B1", "2e-3"), ("A", "4e-3"), ("A", "2e-3")):
                a = dict(base_area)
                a.update({d: sp[d]["base"] + nz + (sp[d]["none2"] if (b, lr) == ("A", "2e-3") else 0.0) for d, bb in S_DOMS.items() if bb == b})
                if (b, "none", lr, s) != drop:
                    fake2(r2_, "ptdom2", b, "none", lr, s, a, cka={d: 0.5 for d in S_DOMS if S_DOMS[d] == b},
                          chk_fail=("data_sha after step 0 = the CPU reference",) if (b, "none", lr, s) == bad else ())
            for d, b in S_DOMS.items():
                q = sp[d]
                for arm, k in ((f"only-{d}", "true"), (f"only-{d}-shuf", "shuf")):
                    a = dict(base_area); a[d] = q["base"] + nz + ex(q[k], i) + float(za[d, k][i])   # A5: an offset may be 8 per-seed values
                    kf = "f3" if k == "true" else "f3_s"
                    f3_d = {**a, d: q["base"] + nz + ex(q[kf], i)} if kf in q else None
                    root = r0 if (ptdom_arm == (arm, s)) else r2_
                    fake2(root, "ptdom" if root == r0 else "ptdom2", b, arm, "2e-3", s, a, final3=f3_d,
                          win_delta={d: 2.0} if k == "true" else None, cka={d: (0.95 if k == "true" else q["cka_s"])},
                          share=q["share_t"] if k == "true" else q["share_s"])
        return load_all(r0, r2_, guard=False)
    hasgeom = lambda x: x.get("reading") is not None and "depends on the correct geometry" in x["reading"]
    with tempfile.TemporaryDirectory() as tmp:   # A2 planted verdicts
        spec = {"countries": dict(true=4.0), "people": dict(true=4.0, shuf=4.0), "cities": dict(shuf=-4.0), "months": {}}
        R_ = s_world(tmp, spec, s0=True)
        o = verdict_S(R_, plant_subs2(R_))
        got = {d: (o["rows"][d]["verdict"], o["rows"][d]["geometry"]) for d in S_DOMS}
        T("A2 S: planted (true - none, shuffled - none) = countries (+4, 0) -> PASS + GEOMETRY-SPECIFIC; people (+4, +4) -> PASS + NOT "
          "SHOWN; cities (0, -4) -> NOT SHOWN + GEOMETRY-SPECIFIC ('may hurt'); months (0, 0) -> neither (seeds 8-15, Holm over 4)",
          got == {"countries": ("PASS", "GEOMETRY-SPECIFIC"), "people": ("PASS", "NOT SHOWN"), "cities": ("NOT SHOWN", "GEOMETRY-SPECIFIC"),
                  "months": ("NOT SHOWN", "NOT SHOWN")} and o["holm_n"] == 4 and not o["problems"] and "may hurt" in o["rows"]["cities"]["reading"])
        x = o["rows"]["countries"]
        T("A3.2 positive control: PASS + GEOMETRY-SPECIFIC with no FLOOR, final3 gains > 0, shuffled - none 0, equal LM shares -> the sentence",
          x["reading"] == GEOM_OK and x["reading_failed"] == [])
        T("A2.2 / A3.3: shuffled - none = the planted 0 / -4; the window split in + out = the gain and the window part EXACTLY "
          "(26 g + 2 x 26) / 75 for the planted +2 per window eval", abs(mn(x["shuf_none"])) < 1e-9 and abs(mn(o["rows"]["cities"]["shuf_none"]) + 4) < 1e-9
          and np.allclose(np.array(x["split"]["vs none"]["in"]) + np.array(x["split"]["vs none"]["out"]), x["gain"], atol=1e-9)
          and np.allclose(x["split"]["vs none"]["in"], (26 * np.asarray(x["gain"]) + 52) / 75, atol=1e-9))
        T("A2.4: the cka double path counts the shuffled runs below none + 0.10 and true - 0.2 (planted 0.45 / 0.5 / 0.95: 8 of 8)",
          x["cka_check"]["n_ok"] == 8)
        q = x["seeds0_7_selected"]
        T("A2.1: the seeds 0-7 pairing (part S0) is descriptive and labelled as selected by F100", q and "selected by F100" in q["label"]
          and abs(q["mean"] - 6.0) < 1e-9)
    with tempfile.TemporaryDirectory() as tmp:   # A3.2: four of the audit's five planted worlds that must NOT get the sentence
        spec = {"countries": dict(true=4.0),                                             # control
                "people": dict(base=95.5, true=3.0),                                     # floor_p: line > 100 (shortcut 95.4), CEILING
                "cities": dict(true=3.0, f3=0.0),                                        # fast: area +3, final3 0
                "months": dict(base=3.0, true=3.0)}                                      # floor_m: line 13.3
        R_ = s_world(tmp, spec)
        o = verdict_S(R_, plant_subs2(R_, h3={"people": 95.4}))
        rw = o["rows"]
        T("A3.2 planted worlds all PASS + GEOMETRY-SPECIFIC (control, floor_p, fast, floor_m)",
          all((rw[d]["verdict"], rw[d]["geometry"]) == ("PASS", "GEOMETRY-SPECIFIC") for d in S_DOMS))
        T("A3.2 control -> the sentence", hasgeom(rw["countries"]))
        T("A3.2 floor_p (people-like FLOOR line 100.4) -> no sentence, reason FLOOR", not hasgeom(rw["people"]) and rw["people"]["reading_failed"]
          and rw["people"]["reading_failed"][0].startswith("FLOOR"))
        T("A3.2 / A4.1 fast (area +3, final3 gain EXACTLY 0 at every seed) -> no sentence, reason 'only learns faster' with the final3 numbers",
          not hasgeom(rw["cities"]) and np.allclose(rw["cities"]["gain_final3"], 0.0, atol=1e-9)
          and any(w.startswith("only learns faster: final3 gain vs none@4e-3 mean +0.00, boot90 lower +0.00") for w in rw["cities"]["reading_failed"]))
        T("A3.2 floor_m (months-like, both arms ~3-6 under the 13.3 line) -> no sentence, reason FLOOR",
          not hasgeom(rw["months"]) and any(w.startswith("FLOOR") for w in rw["months"]["reading_failed"]))
        fl = rw["people"]["floor"]
        T("A3.3: the FLOOR line > 100 note names the line (the shortcut baseline 95.4) and states the condition; CEILING printed (none >= 95)",
          any("the best non-geometric line, the shortcut baseline (every held-out question), is 95.4 > 95" in tg for tg in fl["tags"])
          and any(tg.startswith("CEILING") for tg in fl["tags"]))
        T("A2.3: FLOOR / CEILING also for true vs shuffled (months' floor_geo FLOOR)", any(tg.startswith("FLOOR") for tg in rw["months"]["floor_geo"]["tags"]))
    with tempfile.TemporaryDirectory() as tmp:   # A3.2 harm_small, gate_shuf; A3.1 the F100 none lr for cities
        spec = {"countries": dict(true=1.5, shuf=-4.0),                                  # harm_small
                "people": dict(true=4.0, share_s=0.2, cka_s=0.9),                        # gate_shuf (+ the shuffled arm fits the true target)
                "cities": dict(true=4.0, none2=-1.3),                                    # none@2e-3 1.3 below none@4e-3 (F100's pick)
                "months": {}}
        R_ = s_world(tmp, spec)
        o = verdict_S(R_, plant_subs2(R_))
        rw = o["rows"]
        T("A3.2 harm_small (true +1.5, shuffled -4) -> PASS + GEOMETRY-SPECIFIC but no sentence, reason 'the shuffled arm got worse' (-4)",
          (rw["countries"]["verdict"], rw["countries"]["geometry"]) == ("PASS", "GEOMETRY-SPECIFIC") and not hasgeom(rw["countries"])
          and any(w.startswith("the shuffled arm got worse: shuffled - none@2e-3 mean -4.00, boot90 [") for w in rw["countries"]["reading_failed"])
          and abs(mn(rw["countries"]["shuf_none"]) + 4.0) < 1e-9)
        T("A3.2 gate_shuf (the shuffled arm's LM share 0.2 vs 0.9) -> no sentence, reason 'the dose differs' with both shares",
          not hasgeom(rw["people"]) and any(w.startswith("the dose differs: LM share true 0.9") and "shuffled 0.2" in w for w in rw["people"]["reading_failed"])
          and rw["people"]["lm_share"]["only-people-shuf@2e-3"] < 0.3)
        T("A2.4 cka double path catches a shuffled arm that fits the true target (cka 0.9): 0 of 8", rw["people"]["cka_check"]["n_ok"] == 0)
        T("A3.1: cities' PASS is against none@4e-3 (F100's pick): gain = the planted +4, the same-lr line against none@2e-3 = +5.3; "
          "shuffled - none at the same lr 2e-3", abs(mn(rw["cities"]["gain"]) - 4.0) < 1e-9 and rw["cities"]["pick_none"] == "4e-3"
          and abs(rw["cities"]["same_lr_pass"]["mean"] - 5.3) < 1e-9 and rw["cities"]["shuf_none_lr"] == "2e-3"
          and not rw["cities"]["lr_confounded_shuf_none"] and "same_lr_pass" not in rw["countries"])
        T("A4.4 (y6): cities' shuffled - none VALUE is against none@2e-3 (0 - (-1.3) = +1.3), not against none@4e-3 (0)",
          abs(mn(rw["cities"]["shuf_none"]) - 1.3) < 1e-9)
        T("A4.4 (y11): S's FLOOR (true vs none) is computed against F100's none (cities: none@4e-3), not none@2e-3",
          set(rw["cities"]["floor"]["means"]) == {"only-cities@2e-3", "none@4e-3"} and set(rw["countries"]["floor"]["means"]) == {"only-countries@2e-3", "none@2e-3"})
    with tempfile.TemporaryDirectory() as tmp:
        R_ = s_world(tmp, {"countries": dict(true=4.0)}, drop=("B1", "none", "2e-3", 11))
        T("S verdict refused while a run is missing (B1 none@2e-3 s11 = countries' / people's PASS arm), the reason names the cell",
          RO._raises(lambda: verdict_S(R_, plant_subs2(R_)), "B1 none@2e-3 s11: no run"))
        o = verdict_S(R_, plant_subs2(R_), allow_partial=True)
        txt = report(None, o, probs=["planted integrity problem"])
        T("A2.4 / A3.3: --allow_partial: no verdict word, the would-be word names the reduced Holm family, the report runs and shows the problems",
          not o["final"] and all(o["rows"][d]["verdict"].startswith("PROVISIONAL") for d in S_DOMS) and "PROBLEMS (1" in txt
          and "Holm over the 2 complete of the 4 domains" in o["rows"]["cities"]["verdict"])
        T("A4.3: B1's none@2e-3 is countries' / people's PASS arm (N = N_same): its gap is a refusal reason, never the 'descriptive only' line",
          o["rows"]["countries"]["ns_missing"] is None and "descriptive only" not in txt and any("B1 none@2e-3 s11: no run" in w for w in o["problems"]))
        R2b = s_world(tmp + "/z", {"countries": dict(true=5.0)})
        o2 = verdict_S(R2b, plant_subs2(R2b), allow_partial=True, ext_probs=["batch B1 seed 9: data_sha series differ (planted)"])
        T("A2.4: an outside problem (integrity) with complete runs -> every S verdict PROVISIONAL (no final word)",
          not o2["final"] and all(o2["rows"][d]["verdict"].startswith("PROVISIONAL") for d in S_DOMS))
        T("A4.4 (y7): not final -> no reading line, although countries would be PASS + GEOMETRY-SPECIFIC with every condition met; the "
          "label says the family is complete (not 'reduced': A4.4)", o2["rows"]["countries"]["reading"] is None
          and o2["rows"]["countries"]["would_be"] == ["PASS", "GEOMETRY-SPECIFIC"] and o2["rows"]["countries"]["reading_failed"] == []
          and "complete" in o2["rows"]["countries"]["would_be_holm"] and "reduced" not in o2["rows"]["countries"]["verdict"])
        T("A2 double path: series_problems finds nothing when area = 100 x mean(series); a planted mismatch is reported",
          not series_problems(R2b) and series_problems({("B1", "none", "2e-3", 8): dict(R2b[("B1", "none", "2e-3", 8)],
                                                                                     area={**R2b[("B1", "none", "2e-3", 8)]["area"], "countries": 1.0})}))
        R3 = s_world(tmp + "/y", {}, ptdom_arm=("only-countries", 9))
        T("A2.1: an S arm that is not a ptdom2 run (only-countries@2e-3 s9 from the ptdom tree) -> refused",
          RO._raises(lambda: verdict_S(R3, plant_subs2(R3)), "must be ptdom2 runs"))
    T("A3.1: make_jobs_ptdom2.S_NONE = F100's none picks in runs/ptdom/readout.json (countries / people 2e-3, cities / months 4e-3)",
      check_s_none() == [])
    o_ok = {"domains": {d: {"pick_none": M2.S_NONE[d], "pick_prior": M2.S_LR} for d in S_DOMS}}
    o_n = {"domains": {**o_ok["domains"], "months": {"pick_none": "8e-3", "pick_prior": M2.S_LR}}}
    o_p = {"domains": {**o_ok["domains"], "people": {"pick_none": M2.S_NONE["people"], "pick_prior": "1e-3"}}}
    T("A4.4 (y1): check_s_none refuses a planted F100 read-out whose none pick (months 8e-3) or prior pick (people 1e-3) differs, and a "
      "planted S_NONE (cities 2e-3); agrees with a planted copy of the registered picks",
      check_s_none(o_ok) == [] and len(check_s_none(o_n)) == 1 and "S_NONE" in check_s_none(o_n)[0]
      and len(check_s_none(o_p)) == 1 and "prior picks" in check_s_none(o_p)[0]
      and len(check_s_none(o_ok, {**M2.S_NONE, "cities": "2e-3"})) == 1)
    # ---- 4b. A4 (the coordinator, 2026-11-09): the noise-aware reading conditions; the descriptive none@2e-3 cell
    with tempfile.TemporaryDirectory() as tmp:
        spec = {"countries": dict(true=4.0, shuf=3.0),                     # the shuffled arm takes 3 of the 4 points (A4.2b)
                "people": dict(true=1.5, shuf=-0.9),                       # a significant harm, above A3's -1 bar (A4.2a)
                "cities": dict(true=5.0),                                  # positive control: true +5, shuffled = none
                "months": dict(true=5.0, share_s=0.75)}                    # the dose differs by 0.15 (kept, A4.5)
        R_ = s_world(tmp, spec)
        o = verdict_S(R_, plant_subs2(R_))
        rw = o["rows"]
        T("A4 planted worlds: all four PASS + GEOMETRY-SPECIFIC", all((rw[d]["verdict"], rw[d]["geometry"]) == ("PASS", "GEOMETRY-SPECIFIC") for d in S_DOMS))
        T("A4.2b: true +4 / shuffled +3 (true - shuffled 1 < 0.5 x 4) -> no sentence, first reason 'the shuffled arm takes much of the gain', "
          "and A5d (the upper bound of shuffled - none above half the gain) also named",
          not hasgeom(rw["countries"]) and len(rw["countries"]["reading_failed"]) == 2
          and rw["countries"]["reading_failed"][0].startswith("the shuffled arm takes much of the gain: mean(true - shuffled) +1.00 < 0.5 x mean(true - none@2e-3) +4.00")
          and rw["countries"]["reading_failed"][1].startswith("UNDETERMINED at n = 8: the shuffled arm reaching half of the gain is not excluded"))
        T("A4.2a: true +1.5 / shuffled -0.9 (harm significant: boot90 upper bound < 0) -> no sentence, first reason 'the shuffled arm got worse'; "
          "A5c names whatever lower bound is below -1 (here {})".format(round(RO.boot_ci(rw["people"]["shuf_none"])[0], 2)),
          not hasgeom(rw["people"]) and RO.boot_ci(rw["people"]["shuf_none"])[1] < 0
          and rw["people"]["reading_failed"][0].startswith("the shuffled arm got worse: shuffled - none@2e-3 mean -0.90")
          and all(w.startswith("UNDETERMINED at n = 8: a harm") for w in rw["people"]["reading_failed"][1:])
          and (len(rw["people"]["reading_failed"]) == 2) == (RO.boot_ci(rw["people"]["shuf_none"])[0] < -1))
        T("A4 positive control: true +5, shuffled = none (cities, PASS vs none@4e-3) -> the sentence, no failing condition",
          rw["cities"]["reading"] == GEOM_OK and rw["cities"]["reading_failed"] == [])
        T("A4.5: the dose condition unchanged: LM shares 0.9 vs 0.75 (|difference| 0.15 > 0.1) -> no sentence, the only reason 'the dose differs'",
          not hasgeom(rw["months"]) and len(rw["months"]["reading_failed"]) == 1 and rw["months"]["reading_failed"][0].startswith("the dose differs: LM share true 0.900, shuffled 0.750"))
        txt = report(None, o)
        with tempfile.TemporaryDirectory() as tmp5:   # A5: the interval conditions, each isolated (per-seed shuffled offsets)
            spec5 = {"countries": dict(true=5.0, shuf=[-3.0, 2.5] * 4),   # mean -0.25, wide: upper >= 0 but lower < -1 -> A5c only
                     "people": dict(true=5.0, shuf=[0.0, 4.0] * 4),      # mean +2 (A4.2b holds: 3 >= 2.5), upper > 2.5 -> A5d only
                     "cities": dict(true=5.0), "months": dict(true=5.0)}
            R5 = s_world(tmp5, spec5)
            o5 = verdict_S(R5, plant_subs2(R5))
            r5 = o5["rows"]
            T("A5 planted: countries / people PASS + GEOMETRY-SPECIFIC", all((r5[d]["verdict"], r5[d]["geometry"]) == ("PASS", "GEOMETRY-SPECIFIC")
                                                                          for d in ("countries", "people")))
            T("A5c: shuffled - none mean -0.25 with boot90 lower < -1 (upper >= 0) -> no sentence, the only reason the A5c UNDETERMINED line",
              not hasgeom(r5["countries"]) and len(r5["countries"]["reading_failed"]) == 1
              and r5["countries"]["reading_failed"][0].startswith("UNDETERMINED at n = 8: a harm of the shuffled arm is not excluded"))
            T("A5d: shuffled - none mean +2 with boot90 upper > 0.5 x 5 (A4.2b holds) -> no sentence, the only reason the A5d UNDETERMINED line",
              not hasgeom(r5["people"]) and len(r5["people"]["reading_failed"]) == 1
              and r5["people"]["reading_failed"][0].startswith("UNDETERMINED at n = 8: the shuffled arm reaching half of the gain is not excluded"))
            T("A5 positive controls (true +5, shuffled = none): the sentence for cities and months", hasgeom(r5["cities"]) and hasgeom(r5["months"]))
        T("A4.4: the report states the geometry-dependence sentence exactly once (cities' reading line); the closing paragraph no longer does (A2.2 removed)",
          txt.count("depends on the correct geometry") == 1 and "reading (pre-registered, A3.2 / A4): " + GEOM_OK in txt
          and "only when both hold" not in txt)
    with tempfile.TemporaryDirectory() as tmp:   # A4.1: the final3 condition per gain, exact final3 values (no noise)
        alt = [1.2, -1.0] * 4                                              # mean +0.10, paired bootstrap 90% lower bound < 0
        spec = {"countries": dict(true=3.0, f3=alt),                       # noise-like final3 gain with a positive mean
                "people": dict(true=3.0, f3=0.0, f3_s=-3.0),              # final3 vs none 0 exactly, vs shuffled +3
                "cities": dict(true=3.0, f3=3.0, f3_s=3.0),               # final3 vs none +3, vs shuffled 0 exactly
                "months": dict(true=3.0, f3=3.0)}                          # both final3 gains > 0 -> the sentence
        R_ = s_world(tmp, spec)
        o = verdict_S(R_, plant_subs2(R_))
        rw = o["rows"]
        f3why = lambda d: [w for w in rw[d]["reading_failed"] if w.startswith("only learns faster")]
        T("A4.1 worlds: all four PASS + GEOMETRY-SPECIFIC; the planted final3 gains are exact", all((rw[d]["verdict"], rw[d]["geometry"]) ==
          ("PASS", "GEOMETRY-SPECIFIC") for d in S_DOMS) and np.allclose(rw["countries"]["gain_final3"], alt, atol=1e-9)
          and np.allclose(rw["people"]["geo_final3"], 3.0, atol=1e-9) and np.allclose(rw["cities"]["geo_final3"], 0.0, atol=1e-9))
        T("A4.1: a final3 gain of mean +0.10 whose paired bootstrap 90% lower bound is < 0 -> no sentence (A3's point rule would print it)",
          not hasgeom(rw["countries"]) and mn(rw["countries"]["gain_final3"]) > 0 and RO.boot_ci(rw["countries"]["gain_final3"])[0] < 0
          and len(f3why("countries")) == 1 and rw["countries"]["reading_failed"] == f3why("countries"))
        T("A4.1: final3 vs none 0 but vs shuffled +3 -> no sentence (both gains needed); vs none +3 but vs shuffled 0 -> no sentence",
          not hasgeom(rw["people"]) and rw["people"]["reading_failed"] == f3why("people") and not hasgeom(rw["cities"])
          and rw["cities"]["reading_failed"] == f3why("cities"))
        T("A4.1: both final3 gains +3 exactly (lower bounds > 0) -> the sentence", rw["months"]["reading"] == GEOM_OK)
    for kind in ("missing", "invalid"):   # A4.3: A's none@2e-3 (cities / months: descriptive only) missing / invalid at s11
        with tempfile.TemporaryDirectory() as tmp:
            cell = ("A", "none", "2e-3", 11)
            R_ = s_world(tmp, {"countries": dict(true=5.0), "cities": dict(true=5.0), "months": dict(true=5.0)},
                         **({"drop": cell} if kind == "missing" else {"bad": cell}))
            try:
                o, err = verdict_S(R_, plant_subs2(R_)), None
            except BaseException as e:   # noqa: BLE001 (a TypeError or a refusal is the failure under test)
                o, err = None, f"{type(e).__name__}: {e}"
            try:
                op, errp = verdict_S(R_, plant_subs2(R_), allow_partial=True), None
                txt = report(None, op)
            except BaseException as e:   # noqa: BLE001
                op, errp, txt = None, f"{type(e).__name__}: {e}", ""
            ok_ = err is None and errp is None and o["final"] and not o["problems"]
            rc_ = o["rows"]["cities"] if o else {}
            T(f"A4.3 ({kind}): A's none@2e-3 s11 {kind} -> strict and --allow_partial run (no refusal, no TypeError: {err or errp}); cities / "
              "months judged as usual vs none@4e-3 (PASS + GEOMETRY-SPECIFIC), countries untouched (the sentence)",
              ok_ and all((o["rows"][d]["verdict"], o["rows"][d]["geometry"]) == ("PASS", "GEOMETRY-SPECIFIC") for d in ("cities", "months"))
              and o["rows"]["countries"]["reading"] == GEOM_OK and op["final"])
            T(f"A4.3 ({kind}): only the lines needing none@2e-3 say 'missing / invalid' (shuffled - none, the same-lr PASS line, the window "
              "split's shuffled vs none, the double path, the shortcut entry of none@2e-3); the gate is computed; the sentence withheld naming the cell",
              ok_ and rc_["shuf_none"] is None and "missing" in rc_["same_lr_pass"] and rc_["ns_missing"] and "none@2e-3 s11" in rc_["ns_missing"][0]
              and rc_["gate"] is not None and rc_["cka_check"] is None and rc_["shortcut"]["none@2e-3"] is None and rc_["shortcut"]["none@4e-3"]
              and not hasgeom(rc_) and rc_["reading_failed"] and rc_["reading_failed"][0].startswith("shuffled - none@2e-3 missing / invalid (none@2e-3 s11")
              and txt.count("missing / invalid") >= 2 * 5 and "A4.3: none@2e-3 (descriptive only; refuses nothing) missing / invalid: none@2e-3 s11" in txt
              and o["rows"]["countries"]["ns_missing"] is None)
    with tempfile.TemporaryDirectory() as tmp:   # A4.3: a missing PASS arm (A none@4e-3) is still refused, by name
        R_ = s_world(tmp, {}, drop=("A", "none", "4e-3", 11))
        try:
            verdict_S(R_, plant_subs2(R_)); msg = ""
        except SystemExit as e:
            msg = str(e)
        T("A4.3: A none@4e-3 s11 missing (cities' / months' PASS arm) -> refused, the reason names 'A none@4e-3 s11: no run' and not the "
          "ptdom2-source check", "A none@4e-3 s11: no run" in msg and "must be ptdom2" not in msg)

    # ---- 5. loader: identity, validity rules
    with tempfile.TemporaryDirectory() as tmp:
        a = dict(base_area)
        fake2(tmp, "ptdom2", "B2", "only-compass-shuf", "2e-3", 0, a)
        fake2(tmp, "ptdom2", "B2", "only-prefixes-shuf", "2e-3", 0, a, ident={"prior_shuf": ""})
        fake2(tmp, "ptdom2", "B2", "only-ranks-shuf", "2e-3", 0, a, chk_fail=("item-state CKA to the target rises by >= 0.1 ... (ptdom2: ...)",))
        fake2(tmp, "ptdom2", "B2", "only-ranks", "2e-3", 1, a, chk_fail=("data_sha after step 0 = the CPU reference",))
        fake2(tmp, "ptdom2", "B2", "none", "2e-3", 2, a, evals=40)
        fake2(tmp, "ptdom2", "B2", "only-compass", "2e-3", 3, a, pev=[[916, 1], [7500, 0]])
        L2 = load2(tmp, "ptdom2", guard=False)
        T("loader: a shuffled run with matching identity is valid", L2[("B2", "only-compass-shuf", "2e-3", 0)]["valid"])
        T("loader: a run whose args say not shuffled under a -shuf name is invalid (identity)", not L2[("B2", "only-prefixes-shuf", "2e-3", 0)]["valid"]
          and "identity" in L2[("B2", "only-prefixes-shuf", "2e-3", 0)]["why"])
        T("loader: a failed CKA-rise bar alone = valid data (A5.1)", L2[("B2", "only-ranks-shuf", "2e-3", 0)]["valid"])
        T("loader: another failed check -> invalid; an unfinished run -> invalid", not L2[("B2", "only-ranks", "2e-3", 1)]["valid"]
          and not L2[("B2", "none", "2e-3", 2)]["valid"])
        T("A4.4: a prior run whose prior_events differ from WIN -> that run invalid with a 'window' reason (no AssertionError; the others load)",
          not L2[("B2", "only-compass", "2e-3", 3)]["valid"] and L2[("B2", "only-compass", "2e-3", 3)]["why"].startswith("window (A4.4)")
          and L2[("B2", "only-compass-shuf", "2e-3", 0)]["valid"])
    # ---- 6. A1.4: A6 for a shuffled arm = the 'adjacent' rule; true-target arms keep readout_ptdom.geo_a6
    def a6_case(root, arm, seed, geo_at=10, cka=0.7, adj=0.004):
        d_ = M2.dom(arm)
        dd = fake2(root, "ptdom2", "B2", arm, "2e-3", seed, dict(base_area))
        f = next((dd / tag_of("ptdom2", "B2", arm, "2e-3")).glob("run_*.json"))
        j = json.loads(f.read_text())
        for i, e in enumerate(j["curve"]):
            e[f"geo_{d_}"], e[f"cka_{d_}"] = 0.3, 0.7
            if e.get("prior_on") and abs(i - geo_at) == 1:
                e[f"geo_{d_}"] = adj
            if i == geo_at:
                e[f"geo_{d_}"], e[f"cka_{d_}"] = -0.00115, cka
        f.write_text(json.dumps(j))
        ck = json.loads((dd / "check.json").read_text())
        ck["checks"] += [{"check": x + " (...)", "ok": True} for x in RO.A6_KEEP] + [{"check": RO.GEO_FLAG_CHECK + " (before a divergence)", "ok": False}]
        (dd / "check.json").write_text(json.dumps(ck))
    with tempfile.TemporaryDirectory() as tmp:
        a6_case(tmp, "only-compass-shuf", 0)                 # shuffled, adjacent perfect fit, eval CKA (true target) 0.7
        a6_case(tmp, "only-prefixes-shuf", 0, adj=0.3)       # shuffled, adjacent not a perfect fit
        a6_case(tmp, "only-ranks", 0)                        # true target, cka 0.7 at the overshoot: readout_ptdom.geo_a6 -> invalid
        a6_case(tmp, "only-zodiac", 0, cka=1.0)              # true target, cka 1.0: valid (unchanged)
        a6_case(tmp, "only-planets-shuf", 0, adj=0.004, geo_at=4)   # adjacent missing before the first on-eval (index 4 is the first)
        L6 = load2(tmp, "ptdom2", guard=False)
        k_ = lambda arm: ("B2", arm, "2e-3", 0)
        T("A1.4: a SHUFFLED run with a geo overshoot whose adjacent on-evals fit the shuffled target (geo <= 0.005) -> valid, FLAG A6, "
          "although its eval CKA to the TRUE target is 0.7", L6[k_("only-compass-shuf")]["valid"] and L6[k_("only-compass-shuf")]["geo_flag"])
        T("A1.4: a shuffled run whose adjacent on-evals do not fit (0.3) -> invalid ('adjacent' reason); no adjacent eval on one side -> invalid",
          not L6[k_("only-prefixes-shuf")]["valid"] and "(2s)" in L6[k_("only-prefixes-shuf")]["why"] and not L6[k_("only-planets-shuf")]["valid"])
        T("A1.4: true-target arms keep readout_ptdom.geo_a6: cka 0.7 at the overshoot -> invalid, cka 1.0 -> valid",
          not L6[k_("only-ranks")]["valid"] and "not a perfect fit" in L6[k_("only-ranks")]["why"] and L6[k_("only-zodiac")]["valid"])
    n = sum(ok)
    print(f"SELFTEST {'PASS' if n == len(ok) else 'FAIL'} ({n}/{len(ok)})")
    return n == len(ok)


# ------------------------------------------------------------------------------------------------------------ main
def consistency(R_, files):
    """The picks files (select_ptdom2.py) vs the re-derivation (summarize_pick fields)."""
    out, final = [], None
    for f in files:
        pk = json.loads(Path(f).read_text())
        mine = decide_R(R_, pk["after"], pk.get("R1pre"))
        same = summarize_pick(mine) == summarize_pick(pk)
        out.append(f"{Path(f).name} {'agrees' if same else 'DIFFERS'}")
        if not same:
            raise SystemExit(f"picks consistency: {Path(f).name} differs from the re-derivation")
        if pk["stage_written"] in ("R2", None) and pk["complete"]:
            final = mine
    return out, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("R", "S", "all"), default="all")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rng", type=int, default=1, help="--selftest: the planted worlds' rng seed (A4.1: tests/selftest_rngs.py runs 1-10)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.rng) else 1)
    R_ = load_all()
    probs = RO.integrity(R_) + check_manifest(R_) + check_r1pre(R_) + series_problems(R_) + check_s_none()
    if probs and not a.allow_partial:
        raise SystemExit("integrity: " + "; ".join(probs))
    doms = sorted(set(R_DOMS) | set(S_DOMS))
    meta = RO.domain_meta([d for d in doms if d != "months"])
    Rsub = {k: dict(v, cmp={x: y for x, y in v.get("cmp", {}).items() if x in meta}) for k, v in R_.items() if v.get("cmp")}
    subs, sprobs = RO.subsets_all(Rsub, meta)
    if sprobs and not a.allow_partial:
        raise SystemExit("comparison sub-read-outs: " + "; ".join(sprobs[:10]))
    oR = oS = None
    cons = []
    if a.part in ("R", "all"):
        files = sorted(M2.R.glob("picks_R_after*.json"))
        cons, final = consistency(R_, files) if files else ([], None)
        if final is None:
            if not a.allow_partial:
                raise SystemExit("Part R: no complete decision (picks_R_after*.json with stage R2) yet")
        else:
            oR = verdict_R(R_, subs, final, a.allow_partial, RO.site_summary(meta), probs + sprobs)
    if a.part in ("S", "all"):
        oS = verdict_S(R_, subs, a.allow_partial, probs + sprobs)
    print(report(oR, oS, None, cons, probs + sprobs))   # A2.4: --allow_partial shows them and gives no final verdict word
    if a.json:
        conv = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
        Path(a.json).write_text(json.dumps({"R": oR, "S": oS, "consistency": cons, "integrity": probs, "sub_problems": sprobs}, indent=1, default=conv))


if __name__ == "__main__":
    main()
