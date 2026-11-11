"""Pre-registered lr selection of runs/pt300x (AUDIT.md "lr selection", written before any pt300x run): F87 rule 1 exactly
(runs/ptlr/AUDIT.md rules 1-2): per size and arm, argmax of the mean area over seeds 0-3; the top two within 2 points ->
the lower mean in-training bpb (readout_rules.best_lr, near_rule 'top2', no offline bpb at 7%); an edge pick that beats its
neighbour by more than 2 points = open bracket (readout_rules.bracket) -> X: one ladder step beyond that edge, seeds 0-3,
once per arm; still open after the X -> STOP (the user decides); closed -> C: confirm seeds 4-7 at the pick.
A diverged run keeps its area (data); an own failure listed in runs/pt300x/OWN_FAILURES.txt (one run name per line, e.g.
p3x-93m-lr4.8e-3-none-s2; '#' comments) is scored over all 75 eval slots with the missing evals 0 (bpb worst); a run
whose check (CHECK7) failed, or whose recomputed area differs from the check's (double path), or that fails the integrity
re-check, withholds its arm (nothing is decided; the coordinator investigates). By hand, by the coordinator; no hooks.
    python select_pt300x.py [--root RUNS] [--write]   print the selection; --write appends the new X / C / STOP decisions
                                                      to runs/pt300x/decisions.json and writes runs/pt300x/picks.json
    python select_pt300x.py --selftest [--work DIR]   synthetic roots (temp dir under DIR), real CHECK7, real generator
Definitions are tools/readout_rules.py of the frozen pttpp read-out snapshot (sha256[:12] e91ee8a8a846), unchanged."""
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

GEN = Path("/scratch/yh6210/llmadd_torch/make_jobs_pt300x.py")
ROOT_DEFAULT = Path("/scratch/yh6210/llmadd_torch/runs")
RULES_DIR = Path("/scratch/yh6210/llmadd_torch/runs/pttpp/readout/tools")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
assert sha(RULES_DIR / "readout_rules.py") == "e91ee8a8a846", "readout_rules.py changed"
sys.path.insert(0, str(RULES_DIR))
import readout_rules as R   # noqa: E402

_spec = importlib.util.spec_from_file_location("make_jobs_pt300x", GEN)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
M = G.M
W_ON = 6408 / 18310
LM0_TOL = 1e-3
AREA_TOL = 0.006          # CHECK7 prints the area rounded to 2 decimals


def load_json(f):
    """a run JSON is rewritten at every eval (not atomically): an unparsable file counts as not there yet"""
    try:
        return json.loads(Path(f).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def parse_own(root):
    f = Path(root) / "pt300x" / "OWN_FAILURES.txt"
    if not f.exists():
        return set()
    return {l.split("#")[0].strip() for l in f.read_text().splitlines() if l.split("#")[0].strip()}


def integrity(r, size, arm, lr, seed):
    """reasons why a finished 7% 3e8 run is not the intended one (empty = fine); every field that defines the training,
    for the new runs and the reused F87 / ptlr_1b runs alike (out / tag / save_model / ckpt_every are not compared)"""
    a, c, bad = r["args"], r["curve"], []
    d, nl, nh, dff, n_params, _ = M.SIZES[size]
    want = {"data": str(M.DATA), "tokenizer": str(M.TOK), "tokenization": "whole", "target": G.target_of(arm),
            "schedule": "whole" if arm == "none" else "window", "window": "0.05,0.4", "site": "ctx", "arith_lines": 4,
            "lam": 5.0, "bs": 32, "seq": 512, "tokens": 3e8, "eval_every": 250, "n_eval": 150, "d_model": d, "n_layers": nl,
            "n_heads": nh, "d_ff": dff, "seed": seed, "untie": False, "gate_thr": 0.5, "gate_start": 0.0, "gate_hyst": 0.0,
            "geo_layers": None, "grad_ckpt": "auto", "lm_chunk_tokens": -1, "stop_after": -1, "dry_inputs": False,
            "n_prefix": 256, "text_min": 16}
    wrong = {k: a.get(k, "<absent>") for k, v in want.items() if a.get(k, "<absent>") != v}
    if wrong:
        bad.append(f"args {wrong}")
    if not (isinstance(a.get("lr"), float) and abs(a["lr"] - float(lr)) <= 1e-12 * float(lr)):
        bad.append(f"lr {a.get('lr')} (want {lr})")
    if r.get("code_sha") != M.PIN7_SHA:
        bad.append(f"code_sha {r.get('code_sha')}")
    facts = {"steps": 18310, "n_params": n_params, "grad_ckpt_on": n_params > 150e6, "eval_every_steps": 250,
             "geo_layers": [nl // 3, 2 * nl // 3], "read_layers": [nl // 3, 2 * nl // 3], "lm_chunk": 0}
    wf = {k: r.get(k) for k, v in facts.items() if r.get(k) != v}
    if wf:
        bad.append(f"run facts {wf}")
    want_st = list(range(0, 18310, 250)) + [18309]
    if [e["step"] for e in c] != want_st:
        bad.append(f"{len(c)} evals, last at {c[-1]['step'] if c else None} (want 75 at the intended steps)")
    ev = G.WINDOW_EVENTS if arm != "none" else []
    if r.get("prior_events") != ev:
        bad.append(f"prior_events {r.get('prior_events')}")
    on = W_ON if arm != "none" else 0.0
    if c and not (abs(c[-1].get("prior_on_frac", -1) - on) < 1e-9 and abs(c[-1].get("prior_used_frac", -1) - on) < 1e-9):
        bad.append(f"prior_on_frac / used {c[-1].get('prior_on_frac')} / {c[-1].get('prior_used_frac')} (want {on:.6f})")
    ref = M.lm0_ref("7", size, seed)
    if ref is None or not c or not math.isfinite(c[0]["lm"]) or abs(c[0]["lm"] - ref) > LM0_TOL:
        bad.append(f"step-0 lm {c[0]['lm'] if c else None} vs the reference {ref} (tolerance {LM0_TOL})")
    return bad


def new_run(root, size, arm, lr, seed, own):
    """one pt300x run -> {status: ok | missing | running | nocheck | withheld, area, bpb, r, bad, own}"""
    tag, rf, cf = G.run_file(size, arm, lr, seed)
    d = Path(root) / "pt300x" / "out" / tag
    name = f"{tag}-s{seed}"
    rec = dict(name=name, size=size, arm=arm, lr=lr, seed=seed, r=load_json(d / rf), own=name in own, bad=[],
               area=None, bpb=None, status=None)
    r = rec["r"]
    if rec["own"]:   # F87 rule 2: an own failure counts against its lr, over all eval slots, missing evals 0
        rec.update(status="ok", bpb=math.inf, area=(100 * sum(R.held(e) for e in r["curve"]) / 75) if r else 0.0)
        return rec
    if r is None:
        rec["status"] = "missing"
        return rec
    if not R.finished(r):
        rec["status"] = "running"
        return rec
    ck = load_json(d / cf)
    if ck is None:
        rec["status"] = "nocheck"
        return rec
    if not ck.get("pass"):
        rec["bad"].append("CHECK7 FAILED: " + "; ".join(k["check"] for k in ck.get("checks", []) if not k["ok"]))
    rec["bad"] += integrity(r, size, arm, lr, seed)
    rec["area"] = float(R.area(r))
    ca = ck.get("info", {}).get("area")
    if ca is None or abs(rec["area"] - ca) > AREA_TOL:
        rec["bad"].append(f"double path: area {rec['area']:.4f} recomputed vs {ca} in {cf}")
    v = r["curve"][-1].get("val_bits_per_byte")
    rec["bpb"] = v if v is not None and math.isfinite(v) else math.inf
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
    br = R.bracket(lrs, s_["mean"], s_["best"])
    ref = lambda s: new_run(root, size, "none", G.GRID[size]["none"][0], s, own)
    unst = [f"{lr} s{s}" for (lr, s), v in recs.items() if v["r"] is not None and not v["own"]
            and (lambda q: q["r"] is not None and not q["own"] and R.unstable(v["r"], q["r"]))(ref(s))]
    div = [f"{lr} s{s}@{R.diverged_at(v['r'])}" for (lr, s), v in recs.items() if v["r"] is not None
           and R.diverged_at(v["r"]) is not None]
    return dict(out, status="ok", best=s_["best"], near=s_["near"], mean=s_["mean"], bpb=s_["bpb"],
                open=br[1] if br and br[0] == "open" else None, note=br[1] if br and br[0] == "note" else None,
                unstable=unst, diverged=div, own=[v["name"] for v in recs.values() if v["own"]])


def decide(sel, st):
    """-> (kind, lr, text): kind final | stop | wait | withheld | X | C (the last two are new decisions)"""
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
    return (f"{sel['size']} {sel['arm']}: best {sel['best']} (seeds 0-3: {g}{tb})"
            + (f"  OPEN at the {sel['open']} edge" if sel["open"] else "")
            + (f"  (edge pick within 2 at the {sel['note']} edge: not open)" if sel["note"] else "")
            + (f"  [diverged {', '.join(sel['diverged'])}]" if sel["diverged"] else "")
            + (f"  [unstable {', '.join(sel['unstable'])}]" if sel["unstable"] else "")
            + (f"  [own failures {', '.join(sel['own'])}]" if sel["own"] else ""))


def run(root, write=False, out=print):
    root = Path(root)
    dpath = root / "pt300x" / "decisions.json"
    log, state = G.decisions(dpath)
    own = parse_own(root)
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
        log2 = log + [dict(e, seq=len(log) + i, written=stamp, by="select_pt300x.py") for i, e in enumerate(new)]
        tmp = dpath.with_name(f"decisions.json.tmp{os.getpid()}")
        tmp.write_text(json.dumps({"log": log2}, indent=1))
        G.decisions(tmp)   # the generator's validation of the whole log before it replaces the old one
        assert G.decisions(dpath)[0] == log, "decisions.json changed during the selection"
        os.replace(tmp, dpath)
        log = log2
    (root / "pt300x" / "picks.json").write_text(json.dumps({"written": stamp, "rule": "F87 rule 1 (top two within 2 -> lower in-training bpb); open bracket -> X once; open after X -> STOP; closed -> C seeds 4-7",
                                                            "cells": cells}, indent=1, default=str))
    for e in new:
        if e["stage"] in ("X", "C"):
            base = f"{G.PY} {GEN} --stage {e['stage']} --size {e['size']} --arm {e['arm']}"
            out(f"NEW {e['stage']} {e['size']} {e['arm']} {e['lr']}: render + test, then submit by hand:\n    {base} --test-only\n    {base} --submit")
        else:
            out(f"NEW STOP {e['size']} {e['arm']}: tell the coordinator / the user (no job)")
    out(f"decisions.json: {len(log)} entries; picks.json written {stamp}")
    return log, new, cells


# ---------------------------------------------------------------- synthetic runs (also used by readout_pt300x.py --selftest)
def fake_curve(area, lm0, arm, bpb=1.2, steps=18310, evals_at=None):
    """75 evals; every held bucket 0 at step 0 and area x 75 / 74 / 100 after it -> readout_rules.area = area"""
    st = evals_at or (list(range(0, steps, 250)) + [steps - 1])
    v = area * len(st) / (len(st) - 1) / 100
    assert 0 <= v <= 1, area
    c = []
    for k, s in enumerate(st):
        on = arm != "none" and 916 <= s < 7324
        e = {f"held_{b}": (0.0 if k == 0 else v) for b in R.B}
        e.update(step=s, lm=lm0 if k == 0 else 3.0, val_loss=3.0, geo=0.1, prior_on=int(on), val_bits_per_byte=bpb,
                 prior_on_frac=(W_ON if arm != "none" else 0.0) if k == len(st) - 1 else 0.1,
                 prior_used_frac=(W_ON if arm != "none" else 0.0) if k == len(st) - 1 else 0.1, sec=float(k))
        if on:
            e["grad_prior"] = 0.5
        c.append(e)
    return c


def fake_new_run(root, size, arm, lr, seed, area, bpb=1.2, check="real", stage=None, mutate=None):
    """a finished pt300x run JSON with the intended args (make_jobs_pt300x.intent_x) + its check file (CHECK7 itself)"""
    stage = stage or ("C" if seed in G.CONF else "A")
    F = G.facts(size, arm, lr, seed, stage)
    I = G.intent_x(F)
    r = {"args": dict(I["args"]), "steps": I["steps"], "geo_layers": I["geo_layers"], "read_layers": I["read_layers"],
         "n_params": I["n_params"], "grad_ckpt_on": I["grad_ckpt_on"], "lm_chunk": 0, "eval_every_steps": 250,
         "code_sha": I["code_sha"], "prior_events": I["events"], "curve": fake_curve(area, I["lm0_ref"], arm, bpb)}
    if mutate:
        mutate(r)
    tag, rf, cf = G.run_file(size, arm, lr, seed)
    d = Path(root) / "pt300x" / "out" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / rf).write_text(json.dumps(r))
    if check == "real":
        subprocess.run([sys.executable, "-", str(d / rf), str(d / cf), json.dumps(I, separators=(",", ":"))],
                       input=M.CHECK7, text=True, capture_output=True)
    elif check:
        (d / cf).write_text(json.dumps({"pass": check != "fail", "checks": [], "info": {"area": round(area, 2)}}))
    return d / rf


# ---------------------------------------------------------------- selftest
def selftest(work):
    base = Path(tempfile.mkdtemp(prefix="pt300x_select_selftest_", dir=work))
    ok, n = 0, 0

    def T(name, cond, info=""):
        nonlocal ok, n
        n += 1
        ok += bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info and not cond else ""))

    def fresh(name):
        r = base / name
        (r / "pt300x").mkdir(parents=True)
        return r

    def grid(root, size, arm, means, bpbs=None, seeds=G.SEL, check="real"):
        for lr, m in means.items():
            for s in seeds:
                fake_new_run(root, size, arm, lr, s, m + (0.5, -0.5, 0.25, -0.25)[s % 4],
                             bpb=(bpbs or {}).get(lr, 1.2), check=check)

    def by(cells, size, arm):
        return cells[size][arm]

    quiet = lambda *a: None
    try:
        r0 = fresh("empty")
        _, new, cells = run(r0, out=quiet)
        T("empty root: every arm waits, nothing decided", not new and all(by(cells, s, a)["decision"] == "wait"
                                                                         for s in G.GRID for a in G.ARMS))
        r1 = fresh("closed")
        grid(r1, "93m", "none", {"2.4e-3": 50, "4.8e-3": 60, "9.6e-3": 65, "1.92e-2": 62})
        _, new, cells = run(r1, out=quiet)
        c = by(cells, "93m", "none")
        T("interior pick by > 2 -> C at 9.6e-3 (real CHECK7 files)", c["decision"] == "C" and c["lr"] == "9.6e-3", c["text"])
        T("its mean = the designed 65.0", abs(c["mean"]["9.6e-3"] - 65.0) < 1e-9, c.get("mean"))
        T("the other three arms wait", sum(by(cells, s, a)["decision"] == "wait" for s in G.GRID for a in G.ARMS) == 3)
        r2 = fresh("tie")
        grid(r2, "93m", "none", {"2.4e-3": 50, "4.8e-3": 60, "9.6e-3": 65, "1.92e-2": 64},
             bpbs={"9.6e-3": 1.25, "1.92e-2": 1.20})
        c = by(run(r2, out=quiet)[2], "93m", "none")
        T("top two within 2 -> the lower in-training bpb (1.92e-2, an edge pick within 2: not open) -> C 1.92e-2",
          c["decision"] == "C" and c["lr"] == "1.92e-2" and not c["open"], c["text"])
        r2b = fresh("tie3")
        grid(r2b, "93m", "none", {"2.4e-3": 50, "4.8e-3": 63.5, "9.6e-3": 65, "1.92e-2": 64},
             bpbs={"4.8e-3": 1.10, "9.6e-3": 1.25, "1.92e-2": 1.20})
        c = by(run(r2b, out=quiet)[2], "93m", "none")
        T("F87's top-two rule (not 'every lr within 2'): the third lr 4.8e-3 is not in the tie", c["lr"] == "1.92e-2"
          and c["near"] == ["9.6e-3", "1.92e-2"], c.get("near"))
        r3 = fresh("open")
        grid(r3, "93m", "none", {"2.4e-3": 50, "4.8e-3": 60, "9.6e-3": 65, "1.92e-2": 70})
        grid(r3, "294m", "win-ctx", {"3e-4": 92, "6e-4": 88, "1.2e-3": 86, "2.4e-3": 80})
        log, new, cells = run(r3, write=True, out=quiet)
        T("open upper edge -> X 3.84e-2; open lower (294M win-ctx) -> X 1.5e-4",
          by(cells, "93m", "none")["lr"] == "3.84e-2" and by(cells, "294m", "win-ctx")["lr"] == "1.5e-4"
          and [e["stage"] for e in log] == ["X", "X"], [(e["size"], e["arm"], e["stage"], e["lr"]) for e in log])
        dp = r3 / "pt300x" / "decisions.json"
        xs = [x for x in G.runs_of("93m", dp) if x[0] == "X"]
        T("the generator renders the X runs: 93M none 3.84e-2 seeds 0-3", xs == [("X", "none", "3.84e-2", s) for s in G.SEL], xs)
        rend = {nm: (txt, F) for nm, txt, F in G.render("294m", dp)}
        xb = [nm for nm, (txt, F) in rend.items() if F["stage"] == "X"]
        slots = [next(l for l in rend[nm][0].splitlines() if l.startswith("#SBATCH --job-name")) for nm in xb]
        T("294M win-ctx X jobs on B200 slots 4-7, no --save_model", xb == [f"p3x-294m-lr1.5e-4-win-ctx-s{s}" for s in G.SEL]
          and slots == [f"#SBATCH --job-name=llmadd_p3x-b200-slot{k}" for k in (4, 5, 6, 7)]
          and all("--save_model" not in rend[nm][0] for nm in xb), (xb, slots))
        _, new2, cells2 = run(r3, write=True, out=quiet)
        T("a second call before the X runs exist: wait, nothing new", not new2 and by(cells2, "93m", "none")["decision"] == "wait")
        r4 = r3   # the X runs arrive: 3.84e-2 worse -> C at 1.92e-2 (interior now)
        grid(r4, "93m", "none", {"3.84e-2": 66})
        log, new, cells = run(r4, write=True, out=quiet)
        T("X worse -> C at 1.92e-2 (interior of the 5 lrs)", by(cells, "93m", "none")["decision"] == "C"
          and by(cells, "93m", "none")["lr"] == "1.92e-2", by(cells, "93m", "none")["text"])
        cs = [x for x in G.runs_of("93m", dp) if x[0] == "C"]
        rendc = {nm: txt for nm, txt, F in G.render("93m", dp) if F["stage"] == "C"}
        T("C runs = seeds 4-7 at 1.92e-2, with --save_model, L40S", cs == [("C", "none", "1.92e-2", s) for s in G.CONF]
          and all("--save_model" in t and "--partition=l40s_public" in t for t in rendc.values()) and len(rendc) == 4, cs)
        _, new3, cells3 = run(r4, write=True, out=quiet)
        T("after the C: final, nothing new", not new3 and by(cells3, "93m", "none")["decision"] == "final")
        r5 = fresh("open_twice")
        grid(r5, "93m", "none", {"2.4e-3": 50, "4.8e-3": 60, "9.6e-3": 65, "1.92e-2": 70})
        run(r5, write=True, out=quiet)
        grid(r5, "93m", "none", {"3.84e-2": 75})
        log, new, cells = run(r5, write=True, out=quiet)
        T("X better by > 2 -> still open -> STOP (no second X)", by(cells, "93m", "none")["decision"] == "STOP"
          and log[-1]["stage"] == "STOP", by(cells, "93m", "none")["text"])
        _, new4, cells4 = run(r5, write=True, out=quiet)
        T("after a STOP: nothing more", not new4 and by(cells4, "93m", "none")["decision"] == "stop")
        r6 = fresh("checkfail")
        grid(r6, "93m", "win-ctx", {"6e-4": 88, "1.2e-3": 89, "2.4e-3": 90, "4.8e-3": 86})
        fake_new_run(r6, "93m", "win-ctx", "1.2e-3", 2, 89, mutate=lambda r: r["args"].update(lam=1.0))
        c = by(run(r6, out=quiet)[2], "93m", "win-ctx")
        T("a run whose args differ (lam 1): CHECK7 fails -> WITHHELD, nothing decided", c["decision"] == "withheld"
          and "CHECK7 FAILED" in c["text"], c["text"])
        r7 = fresh("doublepath")
        grid(r7, "93m", "win-ctx", {"6e-4": 88, "1.2e-3": 89, "2.4e-3": 90, "4.8e-3": 86}, check="synthetic")
        tag, rf, cf = G.run_file("93m", "win-ctx", "2.4e-3", 1)
        f = r7 / "pt300x" / "out" / tag / cf
        j = json.loads(f.read_text()); j["info"]["area"] += 0.05; f.write_text(json.dumps(j))
        c = by(run(r7, out=quiet)[2], "93m", "win-ctx")
        T("double path: the check's area off by 0.05 -> WITHHELD", c["decision"] == "withheld" and "double path" in c["text"], c["text"])
        r8 = fresh("own")
        grid(r8, "93m", "none", {"2.4e-3": 50, "4.8e-3": 60, "9.6e-3": 65, "1.92e-2": 55})
        tag, rf, cf = G.run_file("93m", "none", "9.6e-3", 3)
        (r8 / "pt300x" / "out" / tag / rf).unlink()
        c = by(run(r8, out=quiet)[2], "93m", "none")
        T("a missing run (not listed) -> wait", c["decision"] == "wait", c["text"])
        (r8 / "pt300x" / "OWN_FAILURES.txt").write_text("# own failures\np3x-93m-lr9.6e-3-none-s3   # crashed\n")
        c = by(run(r8, out=quiet)[2], "93m", "none")
        T("listed as an own failure -> area 0 for that seed: 9.6e-3 mean 48.81 -> C at 4.8e-3",
          c["decision"] == "C" and c["lr"] == "4.8e-3" and abs(c["mean"]["9.6e-3"] - (65.5 + 64.5 + 65.25) / 4) < 1e-9, c.get("mean"))
        r9 = fresh("running")
        grid(r9, "93m", "none", {"2.4e-3": 50, "4.8e-3": 60, "9.6e-3": 65, "1.92e-2": 62})
        tag, rf, cf = G.run_file("93m", "none", "4.8e-3", 0)
        p = r9 / "pt300x" / "out" / tag / rf
        j = json.loads(p.read_text()); j["curve"] = j["curve"][:40]; p.write_text(json.dumps(j))
        (r9 / "pt300x" / "out" / tag / cf).unlink()
        c = by(run(r9, out=quiet)[2], "93m", "none")
        T("an unfinished run -> wait", c["decision"] == "wait" and "running" in c["text"], c["text"])
        r10 = fresh("lm0")
        grid(r10, "93m", "none", {"2.4e-3": 50, "4.8e-3": 60, "9.6e-3": 65, "1.92e-2": 62}, check="synthetic")
        tag, rf, cf = G.run_file("93m", "none", "2.4e-3", 1)
        p = r10 / "pt300x" / "out" / tag / rf
        j = json.loads(p.read_text()); j["curve"][0]["lm"] += 0.002; p.write_text(json.dumps(j))
        c = by(run(r10, out=quiet)[2], "93m", "none")
        T("integrity re-check: step-0 lm +0.002 -> WITHHELD", c["decision"] == "withheld" and "step-0 lm" in c["text"], c["text"])
        bad_logs = [
            [dict(seq=0, size="93m", arm="none", stage="X", lr="7.68e-2")],                       # two steps out
            [dict(seq=0, size="93m", arm="none", stage="C", lr="9.6e-3"), dict(seq=1, size="93m", arm="none", stage="X", lr="3.84e-2")],
            [dict(seq=0, size="93m", arm="none", stage="STOP", lr=None), dict(seq=1, size="93m", arm="none", stage="C", lr="9.6e-3")],
            [dict(seq=0, size="93m", arm="none", stage="C", lr="3.84e-2")],                       # C outside grid + X
            [dict(seq=1, size="93m", arm="none", stage="C", lr="9.6e-3")]]                        # seq
        refused = 0
        for i, L in enumerate(bad_logs):
            f = base / f"badlog{i}.json"
            f.write_text(json.dumps({"log": L}))
            try:
                G.decisions(f)
            except AssertionError:
                refused += 1
        T("the generator refuses malformed decision logs (5/5)", refused == len(bad_logs), refused)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"SELECT SELFTEST {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    return ok == n


def main():
    if "--selftest" in sys.argv:
        w = sys.argv[sys.argv.index("--work") + 1] if "--work" in sys.argv else None
        sys.exit(0 if selftest(w) else 1)
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else ROOT_DEFAULT
    print(f"# pt300x selection {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} root {root}; generator "
          f"{sha(GEN)}; readout_rules {sha(RULES_DIR / 'readout_rules.py')}")
    run(root, write="--write" in sys.argv)


if __name__ == "__main__":
    main()
