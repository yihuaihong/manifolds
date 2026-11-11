"""Pre-registered lr selection of runs/pt2sz (AUDIT.md "lr selection", written before any pt2sz run) = pt300x's protocol
(runs/pt300x/select_pt300x.py) on this batch's 2% runs: F87 rule 1 (runs/ptlr/AUDIT.md rules 1-2): per size and arm,
argmax of the mean area over seeds 0-3; the top two within 2 points -> the lower mean FINAL IN-TRAINING bpb
(readout_rules.best_lr, near_rule 'top2'; the 2% jobs also have the offline full-val bpb, which is printed but not used);
an edge pick that beats its neighbour by more than 2 points = open bracket (readout_rules.bracket) -> X: one ladder rung
beyond that edge, seeds 0-3, once per arm; still open after the X -> STOP (the user decides); closed -> C: confirm seeds
4-7 at the pick.
A diverged run keeps its area (data); an own failure listed in runs/pt2sz/OWN_FAILURES.txt (one run name per line, e.g.
p2z-93m-lr4.8e-3-none-s2; '#' comments; a listed name that is not a run of the arm's grid -> WARNING) is scored over all
75 eval slots with the missing evals 0 (bpb worst); a run whose check.json (make_pt2main_tasks.CHECK_PY) failed, whose
recomputed area differs from the check's by more than 0.006 (double path; the check rounds to 2 decimals), or that fails
the integrity re-check, withholds its arm (nothing is decided; the coordinator investigates). By hand; no hooks.
Descriptive only (never decides): pt2main rule 1 (every lr within 2 of the top, offline full-val bpb tie-break: the rule that
picked the 53M / 190M / 505M cells of F98) on the same runs, with a flag when it would pick another lr.
    python select_pt2sz.py [--root RUNS] [--write]   print the selection; --write appends the new X / C / STOP decisions
                                                     to runs/pt2sz/decisions.json and writes runs/pt2sz/picks.json
    python select_pt2sz.py --selftest [--work DIR]   synthetic roots (temp dir under DIR), real CHECK_PY, real generator
Definitions: tools/readout_rules.py of this batch's snapshot (sha256[:12] e91ee8a8a846), unchanged."""
import datetime
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GEN = Path("/scratch/yh6210/llmadd_torch/make_jobs_pt2sz.py")
ROOT_DEFAULT = Path("/scratch/yh6210/llmadd_torch/runs")
RULES_DIR = Path("/scratch/yh6210/llmadd_torch/runs/pt2sz/code/tools")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
assert sha(RULES_DIR / "readout_rules.py") == "e91ee8a8a846", "readout_rules.py changed"
_spec = importlib.util.spec_from_file_location("make_jobs_pt2sz", GEN)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
G2 = G.g2()   # make_pt2main_tasks of runs/pt2sz/code (CHECK_PY, ...); puts RULES_DIR on sys.path
import readout_rules as R   # noqa: E402
assert Path(R.__file__).resolve().parent == RULES_DIR.resolve()
PS, P2, TP = G.PS, G.P2, G.TP
W_ON = 6408 / 18310
LM0_TOL = 1e-3
AREA_TOL = 0.006


def load_json(f):
    """a run JSON is rewritten at every eval (not atomically): an unparsable file counts as not there yet"""
    try:
        return json.loads(Path(f).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def parse_own(root, out=print):
    f = Path(root) / "pt2sz" / "OWN_FAILURES.txt"
    if not f.exists():
        return set()
    own = {l.split("#")[0].strip() for l in f.read_text().splitlines() if l.split("#")[0].strip()}
    names = {G.tag_of(z, a, lr) + f"-s{s}" for z in G.GRID for a in G.ARMS for lr in G.LADDER for s in G.SEL}
    for n in sorted(own - names):
        out(f"WARNING: OWN_FAILURES.txt lists '{n}', which is not a pt2sz selection run name: ignored")
    return own & names


def ref_lm0(size, seed):
    """the step-0 lm reference of a (size, seed): new sizes = pt2sz_refs.json; 53M / 190M = pt2same's (make_jobs_pt2same.
    lm0_ref); 505M = pt2s505_refs.json (make_jobs_pt2s505.LM0)"""
    if size in G.SIZES:
        return G.lm0_ref(size, seed)
    if size in PS.SIZES:
        return PS.lm0_ref(G2, size, seed)
    assert size == "505m"
    return P2.LM0[("505m", "pretrain_data")][seed]


def integrity(r, size, arm, lr, seed):
    """reasons why a finished 2% 3e8 run is not the intended one (empty = fine): every field that defines the training (the
    intent of this batch's generator for that size / arm / lr / seed; out / tag are not compared), code_sha, the run facts,
    75 evals at the intended steps, the window events and use, the step-0 lm vs the reference. Also used by the read-out on
    the reused 53M / 190M (pt2same) and 505M (pt2s505) runs."""
    a, c, bad = r["args"], r["curve"], []
    F = G.facts(size, arm, lr, seed, "C" if seed >= 4 else "A", G2, lm0=ref_lm0(size, seed))
    I = G.intent_of(G2, F)
    want = {k: v for k, v in I["args"].items() if k not in ("tag",)}
    want["data"] = str(G.DATA)
    wrong = {k: a.get(k, "<absent>") for k, v in want.items() if a.get(k, "<absent>") != v}
    if wrong:
        bad.append(f"args {wrong}")
    if r.get("code_sha") != I["code_sha"]:
        bad.append(f"code_sha {r.get('code_sha')}")
    keys = ("steps", "n_params", "grad_ckpt_on", "geo_layers", "read_layers", "lm_chunk", "eval_every_steps")
    wf = {k: r.get(k) for k in keys if r.get(k) != I[k]}
    if wf:
        bad.append(f"run facts {wf}")
    want_st = list(range(0, 18310, 250)) + [18309]
    if [e["step"] for e in c] != want_st:
        bad.append(f"{len(c)} evals, last at {c[-1]['step'] if c else None} (want 75 at the intended steps)")
    if r.get("prior_events") != I["events"]:
        bad.append(f"prior_events {r.get('prior_events')}")
    on = W_ON if arm != "none" else 0.0
    if c and not (abs(c[-1].get("prior_on_frac", -1) - on) < 1e-9 and abs(c[-1].get("prior_used_frac", -1) - on) < 1e-9):
        bad.append(f"prior_on_frac / used {c[-1].get('prior_on_frac')} / {c[-1].get('prior_used_frac')} (want {on:.6f})")
    ref = I["lm0_ref"]
    if ref is None or not c or not math.isfinite(c[0]["lm"]) or abs(c[0]["lm"] - ref) > LM0_TOL:
        bad.append(f"step-0 lm {c[0]['lm'] if c else None} vs the reference {ref} (tolerance {LM0_TOL})")
    return bad


def new_run(root, size, arm, lr, seed, own):
    """one pt2sz run -> {status: ok | missing | running | nocheck | withheld, area, bpb, off, r, bad, own}"""
    run, chk, _ = G.run_paths(size, arm, lr, seed, root)
    name = f"{G.tag_of(size, arm, lr)}-s{seed}"
    rec = dict(name=name, size=size, arm=arm, lr=lr, seed=seed, r=load_json(run), own=name in own, bad=[],
               area=None, bpb=None, off=None, status=None)
    r = rec["r"]
    if rec["own"]:   # F87 rule 2: an own failure counts against its lr, over all eval slots, missing evals 0
        rec.update(status="ok", bpb=math.inf, off=math.inf, area=(100 * sum(R.held(e) for e in r["curve"]) / 75) if r else 0.0)
        return rec
    if r is None:
        rec["status"] = "missing"
        return rec
    if not R.finished(r):
        rec["status"] = "running"
        return rec
    ck = load_json(chk)
    if ck is None:
        rec["status"] = "nocheck"
        return rec
    if not ck.get("pass"):
        rec["bad"].append("CHECK_PY FAILED: " + "; ".join(k["check"] for k in ck.get("checks", []) if not k["ok"]))
    rec["bad"] += integrity(r, size, arm, lr, seed)
    rec["area"] = float(R.area(r))
    ca = ck.get("info", {}).get("area")
    if ca is None or abs(rec["area"] - ca) > AREA_TOL:
        rec["bad"].append(f"double path: area {rec['area']:.4f} recomputed vs {ca} in {chk.name}")
    v = r["curve"][-1].get("val_bits_per_byte")
    rec["bpb"] = v if v is not None and math.isfinite(v) else math.inf
    o = ck.get("info", {}).get("bpb_full")
    rec["off"] = o if o is not None and math.isfinite(o) and R.diverged_at(r) is None else (math.inf if R.diverged_at(r) is not None else None)
    rec["status"] = "withheld" if rec["bad"] else "ok"
    return rec


def select_arm(root, size, arm, x_lr, own):
    """F87 rule 1 over the arm's grid (stage A + its X lr) on seeds 0-3"""
    lrs = sorted(set(G.GRID[size][arm]) | ({x_lr} if x_lr else set()), key=float)
    recs = {(lr, s): new_run(root, size, arm, lr, s, own) for lr in lrs for s in G.SEL}
    out = dict(size=size, arm=arm, grid=lrs, recs=recs)
    wait = [f"{lr} s{s} {v['status']}" for (lr, s), v in recs.items() if v["status"] in ("missing", "running", "nocheck")]
    if wait:
        return dict(out, status="incomplete", why=wait)
    held = [f"{lr} s{s}: {w}" for (lr, s), v in recs.items() for w in v["bad"]]
    if held:
        return dict(out, status="withheld", why=held)
    inp = {lr: {"area": [recs[(lr, s)]["area"] for s in G.SEL], "off": [None] * len(G.SEL),
                "intr": [recs[(lr, s)]["bpb"] for s in G.SEL]} for lr in lrs}
    s_ = R.best_lr(inp, near_rule="top2")
    assert not s_["offline"]   # F87 rule 1: the in-training bpb
    br = R.bracket(lrs, s_["mean"], s_["best"])
    # descriptive: pt2main rule 1 (every lr within 2, offline full-val bpb; F98's 53M / 190M / 505M picks)
    inp2 = {lr: {"area": inp[lr]["area"], "off": [recs[(lr, s)]["off"] for s in G.SEL], "intr": inp[lr]["intr"]} for lr in lrs}
    p2 = R.best_lr(inp2, near_rule="within")
    # 'unstable' reference: the same seed's none run at the lowest none lr run so far (pt300x review M7: incl. none's X)
    nlo = min(lrs, key=float) if arm == "none" else min(G.GRID[size]["none"], key=float)
    ref = lambda s: new_run(root, size, "none", nlo, s, own)
    unst = [f"{lr} s{s}" for (lr, s), v in recs.items() if v["r"] is not None and not v["own"]
            and (lambda q: q["r"] is not None and not q["own"] and R.unstable(v["r"], q["r"]))(ref(s))]
    div = [f"{lr} s{s}@{R.diverged_at(v['r'])}" for (lr, s), v in recs.items() if v["r"] is not None
           and R.diverged_at(v["r"]) is not None]
    return dict(out, status="ok", best=s_["best"], near=s_["near"], mean=s_["mean"], bpb=s_["bpb"],
                open=br[1] if br and br[0] == "open" else None, note=br[1] if br and br[0] == "note" else None,
                unstable=unst, diverged=div, own=[v["name"] for v in recs.values() if v["own"]],
                pt2main=dict(best=p2["best"], near=p2["near"], offline=p2["offline"], bpb=p2["bpb"]))


def decide(sel, st):
    """-> (kind, lr, text): kind final | stop | wait | withheld | X | C | STOP (the last three are new decisions)"""
    if st["STOP"]:
        return "stop", None, "STOP decided (the user decides)"
    if st["C"]:
        if sel["status"] == "ok" and sel["best"] != st["C"]:
            return "final", st["C"], f"C decided at {st['C']} BUT the recomputed pick is {sel['best']}: MISMATCH (investigate)"
        return "final", st["C"], f"C decided at {st['C']} (seeds 4-7)"
    if sel["status"] == "incomplete":
        return "wait", None, f"waiting for {len(sel['why'])} run(s): {', '.join(sel['why'][:4])}{' ...' if len(sel['why']) > 4 else ''}"
    if sel["status"] == "withheld":
        return "withheld", None, f"WITHHELD (nothing decided; investigate): {sel['why'][0]}{' ...' if len(sel['why']) > 1 else ''}"
    if sel["open"]:
        if st["X"]:
            return "STOP", None, f"bracket still OPEN at the {sel['open']} edge after the X {st['X']}: STOP (the user decides)"
        edge = sel["grid"][-1] if sel["open"] == "upper" else sel["grid"][0]
        x = G.step_out(edge, sel["open"])
        return "X", x, f"OPEN bracket at the {sel['open']} edge ({edge}) -> X at {x}, seeds 0-3"
    return "C", sel["best"], f"closed -> C at {sel['best']}, seeds 4-7"


def say(sel):
    if sel["status"] != "ok":
        return f"{sel['size']} {sel['arm']}: {sel['status'].upper()}"
    g = ", ".join(f"{lr} {sel['mean'][lr]:.1f}" for lr in sel["grid"])
    tb = "" if len(sel["near"]) == 1 else f"; top two {', '.join(sel['near'])} within 2 -> lower in-training bpb " + \
        ", ".join(f"{lr} {sel['bpb'][lr]:.4f}" for lr in sel["near"])
    p2 = sel["pt2main"]
    d2 = (f"  [descriptive: pt2main rule 1 (within 2, {'offline' if p2['offline'] else 'in-training'} bpb) picks {p2['best']}"
          + (" = the same" if p2["best"] == sel["best"] else " = ANOTHER lr (not used)") + "]")
    return (f"{sel['size']} {sel['arm']}: best {sel['best']} (seeds 0-3: {g}{tb})"
            + (f"  OPEN at the {sel['open']} edge" if sel["open"] else "")
            + (f"  (edge pick within 2 at the {sel['note']} edge: not open)" if sel["note"] else "")
            + (f"  [diverged {', '.join(sel['diverged'])}]" if sel["diverged"] else "")
            + (f"  [unstable {', '.join(sel['unstable'])}]" if sel["unstable"] else "")
            + (f"  [own failures {', '.join(sel['own'])}]" if sel["own"] else "") + d2)


def run(root, write=False, out=print):
    root = Path(root)
    dpath = root / "pt2sz" / "decisions.json"
    log, state = G.decisions(dpath)
    own = parse_own(root, out)
    new, cells = [], {}
    for size in G.GRID:
        for arm in G.ARMS:
            st = state[(size, arm)]
            sel = select_arm(root, size, arm, st["X"], own)
            kind, lr, text = decide(sel, st)
            out(say(sel))
            out(f"    -> {text}")
            cells.setdefault(size, {})[arm] = {k: v for k, v in sel.items() if k != "recs"} | {"decision": kind, "lr": lr,
                                                                                               "text": text}
            if kind in ("X", "C", "STOP"):
                new.append(dict(size=size, arm=arm, stage=kind, lr=lr))
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    if not write:
        out(f"(dry: {len(new)} new decision(s) not written; --write appends them)")
        return log, new, cells
    if new:
        log2 = log + [dict(e, seq=len(log) + i, written=stamp, by="select_pt2sz.py") for i, e in enumerate(new)]
        tmp = dpath.with_name(f"decisions.json.tmp{os.getpid()}")
        tmp.write_text(json.dumps({"log": log2}, indent=1))
        G.decisions(tmp)   # the generator's validation of the whole log before it replaces the old one
        assert G.decisions(dpath)[0] == log, "decisions.json changed during the selection"
        os.replace(tmp, dpath)
        log = log2
    (root / "pt2sz" / "picks.json").write_text(json.dumps({"written": stamp, "rule": "F87 rule 1 (top two within 2 -> lower in-training bpb); open bracket -> X once; open after X -> STOP; closed -> C seeds 4-7",
                                                           "cells": cells}, indent=1, default=str))
    for e in new:
        if e["stage"] in ("X", "C"):
            base = f"{G.PY} {GEN} --stage {e['stage']} --size {e['size']} --arm {e['arm']}"
            out(f"NEW {e['stage']} {e['size']} {e['arm']} {e['lr']}: render + test, then submit by hand:\n    {base} --test-only\n    {base} --submit")
        else:
            out(f"NEW STOP {e['size']} {e['arm']}: tell the coordinator / the user (no job)")
    out(f"decisions.json: {len(log)} entries; picks.json written {stamp}")
    return log, new, cells


# ---------------------------------------------------------------- synthetic runs (also used by readout_pt2sz.py --selftest)
def fake_curve(area, lm0, arm, bpb=1.2, steps=18310, evals_at=None, lm_share=0.8):
    """75 evals; every held bucket 0 at step 0 and area x 75 / 74 / 100 after it -> readout_rules.area = area; the prior's
    gradient norms at on-evals with grad_lm / (grad_lm + grad_prior) = lm_share"""
    st = evals_at or (list(range(0, steps, 250)) + [steps - 1])
    v = area * len(st) / (len(st) - 1) / 100
    assert 0 <= v <= 1, area
    c = []
    for k, s in enumerate(st):
        on = arm != "none" and 916 <= s < 7324
        e = {f"held_{b}": (0.0 if k == 0 else v) for b in R.B}
        e.update(step=s, lm=lm0 if k == 0 else 3.0, val_loss=3.0, geo=0.1 if on else None, prior_on=int(on), val_bits_per_byte=bpb,
                 prior_on_frac=(W_ON if arm != "none" else 0.0) if k == len(st) - 1 else 0.1,
                 prior_used_frac=(W_ON if arm != "none" else 0.0) if k == len(st) - 1 else 0.1, sec=float(k))
        if on:
            e.update(grad_prior=0.5, grad_lm=0.5 * lm_share / (1 - lm_share), grad_cos=0.0)
        c.append(e)
    return c


def fake_run(run_json, chk, I, area, bpb=1.2, check="real", mutate=None, lm_share=0.8, off_bpb=None):
    """a finished 2% run JSON with the intended args (I = make_jobs_pt2sz.intent_of) + offline_lm.json + its check.json
    (CHECK_PY itself when check == 'real')"""
    run_json, chk = Path(run_json), Path(chk)
    run_json.parent.mkdir(parents=True, exist_ok=True)
    r = {"args": dict(I["args"], data=str(G.DATA), out=str(run_json.parent)), "steps": I["steps"],
         "geo_layers": I["geo_layers"], "read_layers": I["read_layers"], "n_params": I["n_params"],
         "grad_ckpt_on": I["grad_ckpt_on"], "lm_chunk": 0, "eval_every_steps": 250, "code_sha": I["code_sha"],
         "prior_events": I["events"], "curve": fake_curve(area, I["lm0_ref"], "none" if not I["events"] else "win-ctx", bpb,
                                                          lm_share=lm_share)}
    if mutate:
        mutate(r)
    run_json.write_text(json.dumps(r))
    offl = chk.parent / "offline_lm.json"
    offl.write_text(json.dumps({str(run_json.resolve()): {"max_windows": 0, "orig_minus_logged": 0.0,
                                                          "bpb_full": bpb + 0.001 if off_bpb is None else off_bpb}}))
    if check == "real":
        subprocess.run([sys.executable, "-", str(run_json), str(offl), str(chk.parent / "gpu.csv"), str(G.R / "code" / "COMMIT"),
                        str(G.DATA), str(chk), json.dumps(I, separators=(",", ":"))], input=G2.CHECK_PY, text=True,
                       capture_output=True)
    elif check:
        chk.write_text(json.dumps({"pass": check != "fail", "checks": [], "info": {"area": round(area, 2), "bpb_full": bpb + 0.001}}))
    return run_json


def fake_new_run(root, size, arm, lr, seed, area, bpb=1.2, check="real", mutate=None, lm_share=0.8, off_bpb=None):
    stage = "C" if seed in G.CONF else "A"
    F = G.facts(size, arm, lr, seed, stage, G2)
    I = G.intent_of(G2, F)
    run, chk, _ = G.run_paths(size, arm, lr, seed, root)
    return fake_run(run, chk, I, area, bpb, check, mutate, lm_share, off_bpb)


# ---------------------------------------------------------------- selftest
def selftest(work):
    base = Path(tempfile.mkdtemp(prefix="pt2sz_select_selftest_", dir=work))
    ok, n = 0, 0

    def T(name, cond, info=""):
        nonlocal ok, n
        n += 1
        ok += bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info and not cond else ""))

    def fresh(name):
        r = base / name
        (r / "pt2sz").mkdir(parents=True)
        return r

    def grid(root, size, arm, means, bpbs=None, seeds=G.SEL, check="real", offs=None):
        for lr, m in means.items():
            for s in seeds:
                fake_new_run(root, size, arm, lr, s, m + (0.5, -0.5, 0.25, -0.25)[s % 4],
                             bpb=(bpbs or {}).get(lr, 1.2), check=check, off_bpb=(offs or {}).get(lr))

    def by(cells, size, arm):
        return cells[size][arm]

    quiet = lambda *a: None
    try:
        r0 = fresh("empty")
        _, new, cells = run(r0, out=quiet)
        T("empty root: every arm waits, nothing decided", not new and all(by(cells, s, a)["decision"] == "wait"
                                                                         for s in G.GRID for a in G.ARMS))
        r1 = fresh("closed")
        grid(r1, "93m", "none", {"2.4e-3": 15, "4.8e-3": 20, "9.6e-3": 25, "1.92e-2": 18})
        _, new, cells = run(r1, out=quiet)
        c = by(cells, "93m", "none")
        T("interior pick by > 2 -> C at 9.6e-3 (real CHECK_PY files)", c["decision"] == "C" and c["lr"] == "9.6e-3", c["text"])
        T("its mean = the designed 25.0", abs(c["mean"]["9.6e-3"] - 25.0) < 1e-9, c.get("mean"))
        T("the other three arms wait", sum(by(cells, s, a)["decision"] == "wait" for s in G.GRID for a in G.ARMS) == 3)
        r2 = fresh("tie")
        grid(r2, "93m", "none", {"2.4e-3": 15, "4.8e-3": 20, "9.6e-3": 25, "1.92e-2": 24},
             bpbs={"9.6e-3": 1.25, "1.92e-2": 1.20}, offs={"9.6e-3": 1.10, "1.92e-2": 1.30})
        c = by(run(r2, out=quiet)[2], "93m", "none")
        T("top two within 2 -> the lower IN-TRAINING bpb (1.92e-2, an edge pick within 2: not open) -> C 1.92e-2; the offline "
          "bpb (opposite order) is not used", c["decision"] == "C" and c["lr"] == "1.92e-2" and not c["open"], c["text"])
        T("descriptive pt2main rule 1 (offline bpb) would pick 9.6e-3 and is flagged, not used",
          c["pt2main"]["best"] == "9.6e-3" and c["pt2main"]["offline"], c.get("pt2main"))
        r2b = fresh("tie3")
        grid(r2b, "93m", "none", {"2.4e-3": 15, "4.8e-3": 23.5, "9.6e-3": 25, "1.92e-2": 24},
             bpbs={"4.8e-3": 1.10, "9.6e-3": 1.25, "1.92e-2": 1.20})
        c = by(run(r2b, out=quiet)[2], "93m", "none")
        T("F87's top-two rule (not 'every lr within 2'): the third lr 4.8e-3 is not in the tie", c["lr"] == "1.92e-2"
          and c["near"] == ["9.6e-3", "1.92e-2"], c.get("near"))
        r3 = fresh("open")
        grid(r3, "93m", "none", {"2.4e-3": 15, "4.8e-3": 20, "9.6e-3": 25, "1.92e-2": 30})
        grid(r3, "294m", "win-ctx", {"1.5e-4": 80, "3e-4": 76, "6e-4": 74, "1.2e-3": 70})
        log, new, cells = run(r3, write=True, out=quiet)
        T("open upper edge -> X 3.84e-2; open lower (294M win-ctx at 1.5e-4) -> X 7.5e-5 (the added ladder rung)",
          by(cells, "93m", "none")["lr"] == "3.84e-2" and by(cells, "294m", "win-ctx")["lr"] == "7.5e-5"
          and [e["stage"] for e in log] == ["X", "X"], [(e["size"], e["arm"], e["stage"], e["lr"]) for e in log])
        dp = r3 / "pt2sz" / "decisions.json"
        xs = [x for x in G.runs_of("93m", dp) if x[0] == "X"]
        T("the generator renders the X runs: 93M none 3.84e-2 seeds 0-3", xs == [("X", "none", "3.84e-2", s) for s in G.SEL], xs)
        rend = {nm: (txt, F) for nm, txt, F in G.render("294m", dp, G2)}
        xb = [nm for nm, (txt, F) in rend.items() if F["stage"] == "X"]
        T("294M win-ctx X jobs rendered (lr 7.5e-5, seeds 0-3, nice 1000, the model removed after a passing check)",
          xb == [f"p2z-294m-lr7.5e-5-win-ctx-s{s}" for s in G.SEL]
          and all("--nice=1000" in rend[nm][0] and "## model removed (stage X)" in rend[nm][0] and "--lr 7.5e-5 " in rend[nm][0]
                  for nm in xb), xb)
        _, new2, cells2 = run(r3, write=True, out=quiet)
        T("a second call before the X runs exist: wait, nothing new", not new2 and by(cells2, "93m", "none")["decision"] == "wait")
        grid(r3, "93m", "none", {"3.84e-2": 26})
        log, new, cells = run(r3, write=True, out=quiet)
        T("X worse -> C at 1.92e-2 (interior of the 5 lrs)", by(cells, "93m", "none")["decision"] == "C"
          and by(cells, "93m", "none")["lr"] == "1.92e-2", by(cells, "93m", "none")["text"])
        cs = [x for x in G.runs_of("93m", dp) if x[0] == "C"]
        rendc = {nm: txt for nm, txt, F in G.render("93m", dp, G2) if F["stage"] == "C"}
        T("C runs = seeds 4-7 at 1.92e-2, the model kept (no removal line), a100_tandon", cs == [("C", "none", "1.92e-2", s) for s in G.CONF]
          and all("rm -f" not in t and "--partition=a100_tandon" in t and "--save_model" in t for t in rendc.values())
          and len(rendc) == 4, cs)
        _, new3, cells3 = run(r3, write=True, out=quiet)
        T("after the C: final, nothing new", not new3 and by(cells3, "93m", "none")["decision"] == "final")
        r5 = fresh("open_twice")
        grid(r5, "93m", "none", {"2.4e-3": 15, "4.8e-3": 20, "9.6e-3": 25, "1.92e-2": 30})
        run(r5, write=True, out=quiet)
        grid(r5, "93m", "none", {"3.84e-2": 35})
        log, new, cells = run(r5, write=True, out=quiet)
        T("X better by > 2 -> still open -> STOP (no second X)", by(cells, "93m", "none")["decision"] == "STOP"
          and log[-1]["stage"] == "STOP", by(cells, "93m", "none")["text"])
        _, new4, cells4 = run(r5, write=True, out=quiet)
        T("after a STOP: nothing more", not new4 and by(cells4, "93m", "none")["decision"] == "stop")
        r6 = fresh("checkfail")
        grid(r6, "93m", "win-ctx", {"3e-4": 68, "6e-4": 69, "1.2e-3": 72, "2.4e-3": 66})
        fake_new_run(r6, "93m", "win-ctx", "6e-4", 2, 69, mutate=lambda r: r["args"].update(lam=1.0))
        c = by(run(r6, out=quiet)[2], "93m", "win-ctx")
        T("a run whose args differ (lam 1): CHECK_PY fails -> WITHHELD, nothing decided", c["decision"] == "withheld"
          and "CHECK_PY FAILED" in c["text"], c["text"])
        r7 = fresh("doublepath")
        grid(r7, "93m", "win-ctx", {"3e-4": 68, "6e-4": 69, "1.2e-3": 72, "2.4e-3": 66}, check="synthetic")
        _, chkf, _ = G.run_paths("93m", "win-ctx", "1.2e-3", 1, r7)
        j = json.loads(chkf.read_text()); j["info"]["area"] += 0.05; chkf.write_text(json.dumps(j))
        c = by(run(r7, out=quiet)[2], "93m", "win-ctx")
        T("double path: the check's area off by 0.05 -> WITHHELD", c["decision"] == "withheld" and "double path" in c["text"], c["text"])
        r8 = fresh("own")
        grid(r8, "93m", "none", {"2.4e-3": 15, "4.8e-3": 20, "9.6e-3": 25, "1.92e-2": 12})
        run_f, _, _ = G.run_paths("93m", "none", "9.6e-3", 3, r8)
        run_f.unlink()
        c = by(run(r8, out=quiet)[2], "93m", "none")
        T("a missing run (not listed) -> wait", c["decision"] == "wait", c["text"])
        msgs = []
        (r8 / "pt2sz" / "OWN_FAILURES.txt").write_text("# own failures\np2z-93m-lr9.6e-3-none-s3   # crashed\np2z-93m-lr9.6e-3-none-s9\n")
        c = by(run(r8, out=msgs.append)[2], "93m", "none")
        T("listed as an own failure -> area 0 for that seed: 9.6e-3 mean 18.81 -> C at 4.8e-3; a misspelt name -> WARNING",
          c["decision"] == "C" and c["lr"] == "4.8e-3" and abs(c["mean"]["9.6e-3"] - (25.5 + 24.5 + 25.25) / 4) < 1e-9
          and any(m.startswith("WARNING: OWN_FAILURES.txt lists 'p2z-93m-lr9.6e-3-none-s9'") for m in msgs), (c.get("mean"), msgs[:2]))
        r9 = fresh("running")
        grid(r9, "93m", "none", {"2.4e-3": 15, "4.8e-3": 20, "9.6e-3": 25, "1.92e-2": 22})
        p, chkf, _ = G.run_paths("93m", "none", "4.8e-3", 0, r9)
        j = json.loads(p.read_text()); j["curve"] = j["curve"][:40]; p.write_text(json.dumps(j))
        chkf.unlink()
        c = by(run(r9, out=quiet)[2], "93m", "none")
        T("an unfinished run -> wait", c["decision"] == "wait" and "running" in c["text"], c["text"])
        r10 = fresh("lm0")
        grid(r10, "93m", "none", {"2.4e-3": 15, "4.8e-3": 20, "9.6e-3": 25, "1.92e-2": 22}, check="synthetic")
        p, _, _ = G.run_paths("93m", "none", "2.4e-3", 1, r10)
        j = json.loads(p.read_text()); j["curve"][0]["lm"] += 0.002; p.write_text(json.dumps(j))
        c = by(run(r10, out=quiet)[2], "93m", "none")
        T("integrity re-check: step-0 lm +0.002 -> WITHHELD", c["decision"] == "withheld" and "step-0 lm" in c["text"], c["text"])
        r11 = fresh("tokens")
        grid(r11, "294m", "none", {"2.4e-3": 15, "4.8e-3": 20, "9.6e-3": 25, "1.92e-2": 22}, check="synthetic")
        p, _, _ = G.run_paths("294m", "none", "9.6e-3", 2, r11)
        j = json.loads(p.read_text()); j["args"]["tokens"] = 156794880.0; p.write_text(json.dumps(j))
        c = by(run(r11, out=quiet)[2], "294m", "none")
        T("integrity re-check: tokens of the tpp line (156.8M) -> WITHHELD", c["decision"] == "withheld" and "tokens" in c["text"], c["text"])
        bad_logs = [
            [dict(seq=0, size="93m", arm="none", stage="X", lr="7.68e-2")],                       # two rungs out
            [dict(seq=0, size="93m", arm="none", stage="C", lr="9.6e-3"), dict(seq=1, size="93m", arm="none", stage="X", lr="3.84e-2")],
            [dict(seq=0, size="93m", arm="none", stage="STOP", lr=None), dict(seq=1, size="93m", arm="none", stage="C", lr="9.6e-3")],
            [dict(seq=0, size="93m", arm="none", stage="C", lr="3.84e-2")],                       # C outside grid + X
            [dict(seq=1, size="93m", arm="none", stage="C", lr="9.6e-3")],                        # seq
            [dict(seq=0, size="53m", arm="none", stage="C", lr="9.6e-3")]]                        # not a size of this batch
        refused = 0
        for i, L in enumerate(bad_logs):
            f = base / f"badlog{i}.json"
            f.write_text(json.dumps({"log": L}))
            try:
                G.decisions(f)
            except AssertionError:
                refused += 1
        T(f"the generator refuses malformed decision logs ({refused}/{len(bad_logs)})", refused == len(bad_logs), refused)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"SELECT SELFTEST {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    return ok == n


def main():
    if "--selftest" in sys.argv:
        w = sys.argv[sys.argv.index("--work") + 1] if "--work" in sys.argv else None
        sys.exit(0 if selftest(w) else 1)
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else ROOT_DEFAULT
    print(f"# pt2sz selection {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} root {root}; generator "
          f"{sha(GEN)}; readout_rules {sha(RULES_DIR / 'readout_rules.py')}")
    run(root, write="--write" in sys.argv)


if __name__ == "__main__":
    main()
