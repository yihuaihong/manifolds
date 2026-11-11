"""Pre-registered lr selection of runs/ptsz (AUDIT.md "lr selection", written before any ptsz run): pt300x's selection
(select_pt300x.py c4f07df1e18c) = F87 rule 1 exactly (runs/ptlr/AUDIT.md rules 1-2): per size and arm, argmax of the mean
area over seeds 0-3; the top two within 2 points -> the lower mean in-training bpb (readout_rules.best_lr, near_rule 'top2');
an edge pick that beats its neighbour by more than 2 points = open bracket (readout_rules.bracket) -> X: one ladder step
beyond that edge, seeds 0-3, once per arm; still open after the X -> STOP (the user decides); closed -> C: seeds 4-7 at the pick.
A diverged run keeps its area (data). Own failures (amendments I5 / I-A / I-B / 3, AUDIT.md, before any result; replace F87
rule 2's "scored 0" for this batch): a run whose own failure (e.g. a deterministic OOM) recurred in a second, different job
is listed in runs/ptsz/OWN_FAILURES.txt, one entry per line: '<run> <job id>:<STATE> <job id>:<STATE> <ISO timestamp with
zone>  # comment' (own_failures() says when an entry is honoured); an honoured entry makes its whole SIZE cell NOT
DETERMINABLE: both arms 'nd', nothing decided for either, never scored. A run whose job ended without a finished run JSON
is shown 'dead' with an ALERT (one sacct call over this user's jobs: the claim's job first, then the latest job of the run
by name), so an OOM is not a silent 'running'.
A run whose check (CHECK7) failed, or whose recomputed area differs from the check's (double path), or that fails the
integrity re-check, withholds its arm (nothing is decided; the coordinator investigates). By hand, by the coordinator.
Changes from select_pt300x.py (fixes of its build audit, tests/audit_build.md of pt300x, before any ptsz run):
  M6: a listed own failure whose name is no run of this batch, or whose run finished with a passing check, prints a WARNING;
  M7: 'unstable' is judged against the same seed's none run at the arm's lowest none lr INCLUDING a decided X (pt300x: grid[0]);
  I5 (amendment after the fresh build audit): own failures -> the size cell NOT DETERMINABLE; DEAD-job alerts;
  I-A / I-B (amendment 2) and amendment 3: which OWN_FAILURES entries are honoured; the claim's job decides 'dead' first.
    python select_ptsz.py [--root RUNS] [--write]   print the selection; --write appends the new X / C / STOP decisions to
                                                    runs/ptsz/decisions.json and writes runs/ptsz/picks.json
    python select_ptsz.py --selftest [--work DIR]   synthetic roots (temp dir under DIR), real CHECK7, real generator
Definitions are tools/readout_rules.py of the frozen pttpp read-out snapshot (sha256[:12] e91ee8a8a846), unchanged."""
import datetime
import getpass
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

GEN = Path("/scratch/yh6210/llmadd_torch/make_jobs_ptsz.py")
ROOT_DEFAULT = Path("/scratch/yh6210/llmadd_torch/runs")
RULES_DIR = Path("/scratch/yh6210/llmadd_torch/runs/pttpp/readout/tools")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
assert sha(RULES_DIR / "readout_rules.py") == "e91ee8a8a846", "readout_rules.py changed"
sys.path.insert(0, str(RULES_DIR))
import readout_rules as R   # noqa: E402

_spec = importlib.util.spec_from_file_location("make_jobs_ptsz", GEN)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
M = G.M
W_ON = 6408 / 18310
LM0_TOL = 1e-3
AREA_TOL = 0.006          # CHECK7 prints the area rounded to 2 decimals
B = "ptsz"                # this batch's directory under the runs root


def load_json(f):
    """a run JSON is rewritten at every eval (not atomically): an unparsable file counts as not there yet"""
    try:
        return json.loads(Path(f).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def all_names():
    """every run name this batch can ever have (stage A, any X, any C) -> (size, arm, lr, seed)"""
    out = {}
    for size in G.GRID:
        for a in G.ARMS:
            g = G.GRID[size][a]
            for lr in set(g) | {G.step_out(g[0], "lower"), G.step_out(g[-1], "upper")}:
                out.update({f"{G.tag_of(size, a, lr)}-s{s}": (size, a, lr, s) for s in G.SEL + G.CONF})
    return out


def integrity(r, size, arm, lr, seed):
    """reasons why a finished 7% 3e8 run is not the intended one (empty = fine); every field that defines the training,
    for the new runs and the reused F87 / ptlr_1b / pt300x runs alike (out / tag / save_model / ckpt_every are not compared)"""
    a, c, bad = r["args"], r["curve"], []
    d, nl, nh, dff, n_params, _ = G.SIZES[size]
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
    ref = G.lm0_ref(size, seed)
    if ref is None or not c or not math.isfinite(c[0]["lm"]) or abs(c[0]["lm"] - ref) > LM0_TOL:
        bad.append(f"step-0 lm {c[0]['lm'] if c else None} vs the reference {ref} (tolerance {LM0_TOL})")
    return bad


# ---------------------------------------------------------------- Slurm states (one sacct call), own failures (amendments I5, I-A, I-B)
STATES = {}              # job id -> sacct state, every job of this user since SACCT_START (load_states)
BYNAME = {}              # run name -> [(job id, state)] of the jobs named llmadd_<run>, ascending job id
SACCT = {"ok": False}    # whether the last load_states got an answer from sacct
SACCT_START = "2026-11-05T00:00:00"
ALIVE = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING", "REQUEUE_HOLD", "REQUEUE_FED",
         "SIGNALING", "STAGE_OUT"}
OWN_STATES = {"FAILED", "OUT_OF_MEMORY"}     # the failure states an own-failure entry may name (amendment I-A)
TRAIN_MARK = f"code {M.PIN7_SHA}; eval every"  # printed by the trainer's TRAINING call only (not by --dry_inputs)
GUARD_MARKS = ("run guard:", "HOLD present", "not a L40S", "code_sha mismatch")   # exits before / instead of training
_warned = set()


def _sacct():
    """-> (returncode, stdout, stderr); every llmadd job of this user since SACCT_START (tests replace this function)"""
    o = subprocess.run(["sacct", "-X", "-n", "-P", "-u", getpass.getuser(), "-S", SACCT_START, "-o", "JobID,JobName,State"],
                       capture_output=True, text=True, timeout=180)
    return o.returncode, o.stdout, o.stderr


def load_states(root=None, out=print):
    """STATES / BYNAME from one sacct call; a timeout, an exception or a non-zero exit prints a WARNING (never silent).
    Called once per select / read-out run: it also resets the once-per-run warning memory"""
    STATES.clear()
    BYNAME.clear()
    _warned.clear()
    SACCT["ok"] = False
    try:
        rc, so, se = _sacct()
    except subprocess.TimeoutExpired:
        out("WARNING: sacct timed out (180 s): no DEAD-job alerts and no sacct check of OWN_FAILURES entries this time")
        return False
    except Exception as e:   # noqa: BLE001
        out(f"WARNING: sacct failed ({e!r}): no DEAD-job alerts and no sacct check of OWN_FAILURES entries this time")
        return False
    if rc != 0:
        out(f"WARNING: sacct exited {rc} ({se.strip()[:200]}): no DEAD-job alerts and no sacct check of OWN_FAILURES entries this time")
        return False
    for l in so.splitlines():
        f = l.split("|")
        if len(f) >= 3 and f[0].strip().isdigit():
            jid, nm, st = f[0].strip(), f[1].strip(), (f[2].split() or [""])[0]
            STATES[jid] = st
            if nm.startswith("llmadd_psz-"):
                BYNAME.setdefault(nm[len("llmadd_"):], []).append((int(jid), st))
    for v in BYNAME.values():
        v.sort()
    SACCT["ok"] = True
    return True


def jobs_of(d, rf, name):
    """(claim, latest): the job id in run_guard's .claim file and its state; the latest job named llmadd_<run> and its state"""
    c = d / (rf + ".claim")
    cj = c.read_text().strip() if c.exists() else ""
    claim = (cj, STATES.get(cj)) if cj else None
    lj = BYNAME.get(name)
    latest = (str(lj[-1][0]), lj[-1][1]) if lj else None
    return claim, latest


def warn_once(key, msg, out):
    if key not in _warned:
        _warned.add(key)
        out(msg)


def own_entries(root):
    f = Path(root) / B / "OWN_FAILURES.txt"
    return [l.split("#")[0].strip() for l in f.read_text().splitlines() if l.split("#")[0].strip()] if f.exists() else []


def first_readout(root, strict=False):
    """runs/ptsz/FIRST_READOUT.json: written by the first readout_ptsz.py call that sees any final ptsz cell (amendment I-A).
    A file that exists but cannot be parsed: strict -> RuntimeError (the read-out refuses); else {'corrupt': True}"""
    f = Path(root) / B / "FIRST_READOUT.json"
    if not f.exists():
        return None
    try:
        fr = json.loads(f.read_text())
        assert isinstance(fr, dict) and isinstance(fr.get("time"), str) and isinstance(fr.get("own_entries"), list)
        datetime.datetime.fromisoformat(fr["time"])
        return fr
    except Exception as e:   # noqa: BLE001
        if strict:
            raise RuntimeError(f"{f} exists but cannot be parsed ({e!r}): refusing (amendment 3; restore it from its recorded sha)")
        return {"corrupt": True, "time": None, "own_entries": []}


def own_failures(root, out=print):
    """the HONOURED entries of runs/ptsz/OWN_FAILURES.txt (amendments I5, I-A, I-B). An entry is
    '<run> <job id>:<STATE> <job id>:<STATE> <ISO 8601 timestamp with zone>  # comment' and is honoured only if: the run is
    a run of this batch; the two job ids differ and both states are FAILED / OUT_OF_MEMORY (= sacct, when sacct answers);
    the run has no finished run JSON with a passing CHECK7; both jobs' logs show the trainer's TRAINING call (its 'code <sha>;
    eval every' line) and no guard / run-guard / HOLD exit; and the entry was in the file at the first read-out of a ptsz
    result (FIRST_READOUT.json) with a timestamp before it. Every other entry prints a WARNING and is ignored."""
    entries = own_entries(root)
    names = all_names()
    fr = first_readout(root)
    logs = Path(root).parent / "logs"
    now = datetime.datetime.now().astimezone()
    honoured = set()
    if fr and fr.get("corrupt"):
        if entries:
            out("WARNING: FIRST_READOUT.json exists but cannot be parsed: no OWN_FAILURES.txt entry is honoured (amendment 3)")
        return honoured
    if fr:
        for e in sorted(set(fr.get("own_entries", [])) - set(entries)):
            out(f"WARNING: OWN_FAILURES.txt entry '{e}' was in the file at the first read-out ({fr['time']}) and is gone: the file is append-only")
    for e in entries:
        bad = lambda why: out(f"WARNING: OWN_FAILURES.txt entry '{e}' ignored: {why}")
        parts = e.split()
        name = parts[0]
        if name not in names:   # M6
            bad("no run name of this batch")
            continue
        size, arm, lr, seed = names[name]
        tag, rf, cf = G.run_file(size, arm, lr, seed)
        d = Path(root) / B / "out" / tag
        r, ck = load_json(d / rf), load_json(d / cf)
        if r is not None and R.finished(r) and ck is not None and ck.get("pass"):   # the pre-amendment readout's WARNING, restored
            out(f"WARNING: {name} is listed as an own failure but finished with a passing check: entry ignored (amendment I-A)")
            continue
        if r is not None and R.finished(r):   # amendment 3: a finished run JSON is not 'an own failure without a finished run'
            out(f"WARNING: {name} is listed as an own failure but has a finished run JSON ({'CHECK7 failed' if ck is not None else 'no check file'}): "
                f"entry ignored (amendment 3; I5 covers only runs without a finished JSON; a failed check withholds the arm, rule 6)")
            continue
        if len(parts) != 4:
            bad("want '<run> <job id>:<STATE> <job id>:<STATE> <ISO timestamp>' (amendment I-A)")
            continue
        jobs = [x.split(":", 1) for x in parts[1:3]]
        if not all(len(x) == 2 and x[0].isdigit() for x in jobs) or jobs[0][0] == jobs[1][0]:
            bad("two DIFFERENT job ids with their states are required (the original job and its resubmission; amendment I-B)")
            continue
        if not all(x[1] in OWN_STATES for x in jobs):
            bad(f"the failure states must be {sorted(OWN_STATES)} (an infrastructure loss is resubmitted, not listed)")
            continue
        try:
            ts = datetime.datetime.fromisoformat(parts[3])
            assert ts.tzinfo is not None
        except Exception:   # noqa: BLE001
            bad("the timestamp is not ISO 8601 with a time zone")
            continue
        if ts > now:   # amendment 3
            bad(f"its timestamp is in the future (now {now.isoformat(timespec='seconds')})")
            continue
        why = []
        for jid, st in jobs:
            log = logs / f"llmadd_{name}_{jid}.out"
            txt = log.read_text(errors="replace") if log.exists() else None
            if txt is None:
                why.append(f"job {jid}: no log {log}")
            elif TRAIN_MARK not in txt:
                why.append(f"job {jid} never reached the trainer's training call (no '{TRAIN_MARK}' line)")
            elif any(m in txt for m in GUARD_MARKS):
                why.append(f"job {jid} exited at a guard / the run guard / HOLD ({[m for m in GUARD_MARKS if m in txt]})")
            if SACCT["ok"]:
                s = STATES.get(jid)
                if s is None:
                    warn_once(("nosacct", jid), f"WARNING: OWN_FAILURES.txt entry '{e}': sacct has no record of job {jid} (state not verified)", out)
                elif s != st:
                    why.append(f"job {jid} is {s} in sacct, the entry says {st}")
        if why:
            bad("; ".join(why) + " (amendment I-B: a recurrence = two different jobs that both trained and failed)")
            continue
        if not SACCT["ok"]:
            out(f"WARNING: OWN_FAILURES.txt entry '{e}': sacct unavailable, the job states are not verified (the logs are)")
        if fr:
            if e not in fr.get("own_entries", []):
                bad(f"added after the first read-out of a ptsz result ({fr['time']})")
                continue
            if ts >= datetime.datetime.fromisoformat(fr["time"]):
                bad(f"its timestamp is not before the first read-out ({fr['time']})")
                continue
        honoured.add(name)
        last = []   # audit amend2 m5: show each job's last error line, so the coordinator can see it is the same failure
        for jid, st in jobs:
            errs = [l.strip() for l in (logs / f"llmadd_{name}_{jid}.out").read_text(errors="replace").splitlines()
                    if "Error" in l or "error" in l or "Killed" in l]
            last.append(f"job {jid} {st}: {errs[-1][:120] if errs else '(no error line)'}")
        warn_once(("honoured", e), f"NOTE: OWN_FAILURES.txt entry honoured for {name} (the {size} cell NOT DETERMINABLE, I5); "
                  + "; ".join(last) + " - check that this is the same failure", out)
    return honoured


def parse_own(root, out=print):   # the pre-I-A name, kept for callers
    return own_failures(root, out)


def new_run(root, size, arm, lr, seed, own, out=print):
    """one ptsz run -> {status: ok | missing | running | dead | nocheck | withheld, area, bpb, r, bad, own}"""
    tag, rf, cf = G.run_file(size, arm, lr, seed)
    d = Path(root) / B / "out" / tag
    name = f"{tag}-s{seed}"
    rec = dict(name=name, size=size, arm=arm, lr=lr, seed=seed, r=load_json(d / rf), own=name in own, bad=[],
               area=None, bpb=None, status=None)
    r = rec["r"]
    if rec["own"]:   # an HONOURED entry (own_failures): never scored; the size cell is NOT DETERMINABLE (amendment I5)
        rec.update(status="withheld", bad=[f"{name}: recurrent own failure listed in OWN_FAILURES.txt (amendment I5): the "
                                           f"{size} cell is NOT DETERMINABLE"])
        return rec
    claim, latest = jobs_of(d, rf, name)
    if claim and SACCT["ok"] and claim[1] is None:   # amendment 3 (audit amend2 m4 Q5): never a silent 'running'
        warn_once(("noclaimstate", name, claim[0]), f"WARNING: {name}: sacct has no record of its claim's job {claim[0]} "
                  f"(state unknown; the run is shown as waiting)", out)
    if r is None and claim is None:
        rec["status"] = "missing"
        if latest and latest[1] and latest[1] not in ALIVE:   # minor m1: a job that exited before the trainer's claim
            warn_once(("missing", name, latest[0]), f"WARNING: {name}: no run JSON and no claim, but its latest job {latest[0]} "
                      f"(by name, sacct) ended {latest[1]}: it exited before the trainer's training call (HOLD / sha / GPU / "
                      f"data guard, or the --dry_inputs call failed; see its log); resubmit by hand", out)
        return rec
    if r is None or not R.finished(r):
        rec["status"] = "missing" if r is None else "running"
        if claim and claim[1] in ALIVE:   # amendment 3: the claim's own job first - still alive -> running, whatever a later
            return rec                   # duplicate job did (e.g. a resubmission that exited at the run guard)
        j = latest or claim
        if j and j[1] and j[1] not in ALIVE:
            rec.update(status="dead", job=j)
            same = latest is None or claim is None or latest[0] == claim[0]
            warn_once(("dead", name, j[0]), f"ALERT: {name}: no finished run JSON (evals {len(r['curve']) if r else 0}/75); the claim's "
                      f"job {claim[0] if claim else '- (no claim file)'} {(claim[1] or '?') if claim else ''}; the latest job of this run by name "
                      f"(sacct) {latest[0] if latest else '-'} {(latest[1] or '?') if latest else ''}: "
                      + (("no claim file to compare" if claim is None else "no job by name to compare" if latest is None
                          else "the same job as the claim") if same else
                         f"NOT the claim's job -> job {latest[0]} did not reach the trainer's training call (run guard / HOLD / "
                         f"guard exit): NOT a recurrence")
                      + "; resubmit once by hand (remove a stale .claim first, audit M8); only a second, different job that trained "
                      f"and failed again is an own failure (OWN_FAILURES.txt -> the {size} cell NOT DETERMINABLE, amendments I5 / I-B)", out)
        return rec
    ck = load_json(d / cf)
    if ck is None:
        rec["status"] = "nocheck"
        if latest and latest[1] and latest[1] not in ALIVE:   # minor m1: died after the trainer, before CHECK7 wrote its file
            warn_once(("nocheck", name, latest[0]), f"WARNING: {name}: finished run JSON but no check file, and its latest job "
                      f"{latest[0]} ended {latest[1]}: run CHECK7 by hand from the job file, or resubmit", out)
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


def unstable_ref_lr(size, none_x):
    """M7: the arm's lowest none lr, including a decided X of the none arm"""
    return min(set(G.GRID[size]["none"]) | ({none_x} if none_x else set()), key=float)


def select_arm(root, size, arm, x_lr, own, none_x=None, out=print):
    """F87 rule 1 over the arm's grid (stage A + its X lr) on seeds 0-3"""
    lrs = sorted(set(G.GRID[size][arm]) | ({x_lr} if x_lr else set()), key=float)
    own_here = sorted(n for n in own if n.startswith(f"psz-{size}-lr"))
    if own_here:   # amendment I5: a recurrent own failure in EITHER arm -> the size cell is NOT DETERMINABLE (both arms)
        return dict(size=size, arm=arm, grid=lrs, recs={}, status="nd",
                    why=[f"recurrent own failure(s) {own_here} (amendment I5): the {size} cell is NOT DETERMINABLE"])
    recs = {(lr, s): new_run(root, size, arm, lr, s, own, out) for lr in lrs for s in G.SEL}
    res = dict(size=size, arm=arm, grid=lrs, recs=recs)
    wait = [f"{lr} s{s} {v['status']}" + (f" (job {v['job'][0]} {v['job'][1]})" if v.get("job") else "")
            for (lr, s), v in recs.items() if v["status"] in ("missing", "running", "dead", "nocheck")]
    if wait:
        return dict(res, status="incomplete", why=wait)
    held = [f"{lr} s{s}: {w}" for (lr, s), v in recs.items() for w in v["bad"]]
    if held:
        return dict(res, status="withheld", why=held)
    inp = {lr: {"area": [recs[(lr, s)]["area"] for s in G.SEL], "off": [None] * len(G.SEL),
                "intr": [recs[(lr, s)]["bpb"] for s in G.SEL]} for lr in lrs}
    s_ = R.best_lr(inp, near_rule="top2")
    br = R.bracket(lrs, s_["mean"], s_["best"])
    rl = unstable_ref_lr(size, none_x)
    ref = lambda s: new_run(root, size, "none", rl, s, own, out)
    unst = [f"{lr} s{s}" for (lr, s), v in recs.items() if v["r"] is not None and not v["own"]
            and (lambda q: q["r"] is not None and not q["own"] and R.unstable(v["r"], q["r"]))(ref(s))]
    div = [f"{lr} s{s}@{R.diverged_at(v['r'])}" for (lr, s), v in recs.items() if v["r"] is not None
           and R.diverged_at(v["r"]) is not None]
    return dict(res, status="ok", best=s_["best"], near=s_["near"], mean=s_["mean"], bpb=s_["bpb"],
                open=br[1] if br and br[0] == "open" else None, note=br[1] if br and br[0] == "note" else None,
                unstable=unst, unstable_ref=rl, diverged=div, own=[v["name"] for v in recs.values() if v["own"]])


def decide(sel, st):
    """-> (kind, lr, text): kind final | stop | wait | withheld | X | C | STOP (the last three are new decisions)"""
    if st["STOP"]:
        return "stop", None, "STOP decided (the user decides)"
    if sel["status"] == "nd":
        return "nd", None, f"NOT DETERMINABLE: {sel['why'][0]} (nothing decided; the user decides)"
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
            + (f"  [unstable vs none@{sel['unstable_ref']} {', '.join(sel['unstable'])}]" if sel["unstable"] else "")
            + (f"  [own failures {', '.join(sel['own'])}]" if sel["own"] else ""))


def run(root, write=False, out=print):
    root = Path(root)
    dpath = root / B / "decisions.json"
    log, state = G.decisions(dpath)
    load_states(root, out)
    own = own_failures(root, out)
    new, cells = [], {}
    for size in G.GRID:
        for arm in G.ARMS:
            st = state[(size, arm)]
            sel = select_arm(root, size, arm, st["X"], own, state[(size, "none")]["X"], out)
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
        log2 = log + [dict(e, seq=len(log) + i, written=stamp, by="select_ptsz.py") for i, e in enumerate(new)]
        tmp = dpath.with_name(f"decisions.json.tmp{os.getpid()}")
        tmp.write_text(json.dumps({"log": log2}, indent=1))
        G.decisions(tmp)   # the generator's validation of the whole log before it replaces the old one
        assert G.decisions(dpath)[0] == log, "decisions.json changed during the selection"
        os.replace(tmp, dpath)
        log = log2
    (root / B / "picks.json").write_text(json.dumps({"written": stamp, "rule": "F87 rule 1 (top two within 2 -> lower in-training bpb); open bracket -> X once; open after X -> STOP; closed -> C seeds 4-7",
                                                     "cells": cells}, indent=1, default=str))
    for e in new:
        if e["stage"] in ("X", "C"):
            base = f"{G.PY} {GEN} --stage {e['stage']} --size {e['size']} --arm {e['arm']}"
            out(f"NEW {e['stage']} {e['size']} {e['arm']} {e['lr']}: render + test, then submit by hand:\n    {base} --test-only\n    {base} --submit")
        else:
            out(f"NEW STOP {e['size']} {e['arm']}: tell the coordinator / the user (no job)")
    out(f"decisions.json: {len(log)} entries; picks.json written {stamp}")
    return log, new, cells


# ---------------------------------------------------------------- synthetic runs (also used by readout_ptsz.py --selftest)
def fake_curve(area, lm0, arm, bpb=1.2, steps=18310, evals_at=None, grad=(1.0, 0.2, 0.1), val_loss=3.0):
    """75 evals; every held bucket 0 at step 0 and area x 75 / 74 / 100 after it -> readout_rules.area = area; in-window
    evals carry grad_lm / grad_prior / grad_cos = grad"""
    st = evals_at or (list(range(0, steps, 250)) + [steps - 1])
    v = area * len(st) / (len(st) - 1) / 100
    assert 0 <= v <= 1, area
    c = []
    for k, s in enumerate(st):
        on = arm != "none" and 916 <= s < 7324
        e = {f"held_{b}": (0.0 if k == 0 else v) for b in R.B}
        e.update(step=s, lm=lm0 if k == 0 else 3.0, val_loss=val_loss, geo=0.1, prior_on=int(on), val_bits_per_byte=bpb,
                 prior_on_frac=(W_ON if arm != "none" else 0.0) if k == len(st) - 1 else 0.1,
                 prior_used_frac=(W_ON if arm != "none" else 0.0) if k == len(st) - 1 else 0.1, sec=float(k))
        if on:
            e.update(grad_lm=grad[0], grad_prior=grad[1], grad_cos=grad[2])
        c.append(e)
    return c


def fake_new_run(root, size, arm, lr, seed, area, bpb=1.2, check="real", stage=None, mutate=None, **kw):
    """a finished ptsz run JSON with the intended args (make_jobs_ptsz.intent_x) + its check file (CHECK7 itself)"""
    stage = stage or ("C" if seed in G.CONF else "A")
    F = G.facts(size, arm, lr, seed, stage)
    I = G.intent_x(F)
    r = {"args": dict(I["args"]), "steps": I["steps"], "geo_layers": I["geo_layers"], "read_layers": I["read_layers"],
         "n_params": I["n_params"], "grad_ckpt_on": I["grad_ckpt_on"], "lm_chunk": 0, "eval_every_steps": 250,
         "code_sha": I["code_sha"], "prior_events": I["events"], "curve": fake_curve(area, I["lm0_ref"], arm, bpb, **kw)}
    if mutate:
        mutate(r)
    tag, rf, cf = G.run_file(size, arm, lr, seed)
    d = Path(root) / B / "out" / tag
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
    base = Path(tempfile.mkdtemp(prefix="ptsz_select_selftest_", dir=work))
    ok, n = 0, 0

    def T(name, cond, info=""):
        nonlocal ok, n
        n += 1
        ok += bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info and not cond else ""))

    def fresh(name):
        r = base / name
        (r / B).mkdir(parents=True)
        return r

    def grid(root, size, arm, means, bpbs=None, seeds=G.SEL, check="real", **kw):
        for lr, m in means.items():
            for s in seeds:
                fake_new_run(root, size, arm, lr, s, m + (0.5, -0.5, 0.25, -0.25)[s % 4],
                             bpb=(bpbs or {}).get(lr, 1.2), check=check, **kw)

    def by(cells, size, arm):
        return cells[size][arm]

    lines = []
    quiet = lambda *a: lines.append(" ".join(str(x) for x in a))
    real_sacct = _sacct

    def use_sacct(rows=(), rc=0, exc=None):
        """replace sacct for the selftest: (job id, job name, state) rows, an exit code, or an exception"""
        def f():
            if exc:
                raise exc
            return rc, "".join(f"{j}|{n}|{s}\n" for j, n, s in rows), ("slurmdbd: connection refused" if rc else "")
        globals()["_sacct"] = f
        _warned.clear()

    use_sacct([])
    try:
        r0 = fresh("empty")
        _, new, cells = run(r0, out=quiet)
        T("empty root: every arm waits, nothing decided", not new and all(by(cells, s, a)["decision"] == "wait"
                                                                         for s in G.GRID for a in G.ARMS))
        r1 = fresh("closed")
        grid(r1, "136m", "none", {"1.2e-3": 50, "2.4e-3": 60, "4.8e-3": 65, "9.6e-3": 62})
        _, new, cells = run(r1, out=quiet)
        c = by(cells, "136m", "none")
        T("interior pick by > 2 -> C at 4.8e-3 (real CHECK7 files)", c["decision"] == "C" and c["lr"] == "4.8e-3", c["text"])
        T("its mean = the designed 65.0", abs(c["mean"]["4.8e-3"] - 65.0) < 1e-9, c.get("mean"))
        T("the other seven arms wait", sum(by(cells, s, a)["decision"] == "wait" for s in G.GRID for a in G.ARMS) == 7)
        r2 = fresh("tie")
        grid(r2, "136m", "none", {"1.2e-3": 50, "2.4e-3": 60, "4.8e-3": 65, "9.6e-3": 64},
             bpbs={"4.8e-3": 1.25, "9.6e-3": 1.20})
        c = by(run(r2, out=quiet)[2], "136m", "none")
        T("top two within 2 -> the lower in-training bpb (9.6e-3, an edge pick within 2: not open) -> C 9.6e-3",
          c["decision"] == "C" and c["lr"] == "9.6e-3" and not c["open"], c["text"])
        r2b = fresh("tie3")
        grid(r2b, "136m", "none", {"1.2e-3": 50, "2.4e-3": 63.5, "4.8e-3": 65, "9.6e-3": 64},
             bpbs={"2.4e-3": 1.10, "4.8e-3": 1.25, "9.6e-3": 1.20})
        c = by(run(r2b, out=quiet)[2], "136m", "none")
        T("F87's top-two rule (not 'every lr within 2'): the third lr 2.4e-3 is not in the tie", c["lr"] == "9.6e-3"
          and c["near"] == ["4.8e-3", "9.6e-3"], c.get("near"))
        r3 = fresh("open")
        grid(r3, "136m", "none", {"1.2e-3": 50, "2.4e-3": 60, "4.8e-3": 65, "9.6e-3": 70})
        grid(r3, "294m", "win-ctx", {"3e-4": 92, "6e-4": 88, "1.2e-3": 86, "2.4e-3": 80})
        log, new, cells = run(r3, write=True, out=quiet)
        T("open upper edge -> X 1.92e-2 (136m none); open lower (294m win-ctx) -> X 1.5e-4",
          by(cells, "136m", "none")["lr"] == "1.92e-2" and by(cells, "294m", "win-ctx")["lr"] == "1.5e-4"
          and [e["stage"] for e in log] == ["X", "X"], [(e["size"], e["arm"], e["stage"], e["lr"]) for e in log])
        dp = r3 / B / "decisions.json"
        xs = [x for x in G.runs_of("136m", dp) if x[0] == "X"]
        T("the generator renders the X runs: 136m none 1.92e-2 seeds 0-3", xs == [("X", "none", "1.92e-2", s) for s in G.SEL], xs)
        rend = {nm: (txt, F) for nm, txt, F in G.render("294m", dp)}
        xb = [nm for nm, (txt, F) in rend.items() if F["stage"] == "X"]
        T("294m win-ctx X jobs: L40S, --nice=100 (amendment N1), --ckpt_every 2000, no --save_model",
          xb == [f"psz-294m-lr1.5e-4-win-ctx-s{s}" for s in G.SEL]
          and all("--partition=l40s_public" in rend[nm][0] and "#SBATCH --nice=100\n" in rend[nm][0]
                  and " --ckpt_every 2000\"" in rend[nm][0] and "--save_model" not in rend[nm][0] for nm in xb), xb)
        _, new2, cells2 = run(r3, write=True, out=quiet)
        T("a second call before the X runs exist: wait, nothing new", not new2 and by(cells2, "136m", "none")["decision"] == "wait")
        r4 = r3   # the X runs arrive: 1.92e-2 worse -> C at 9.6e-3 (interior now)
        grid(r4, "136m", "none", {"1.92e-2": 66})
        log, new, cells = run(r4, write=True, out=quiet)
        T("X worse -> C at 9.6e-3 (interior of the 5 lrs)", by(cells, "136m", "none")["decision"] == "C"
          and by(cells, "136m", "none")["lr"] == "9.6e-3", by(cells, "136m", "none")["text"])
        cs = [x for x in G.runs_of("136m", dp) if x[0] == "C"]
        rendc = {nm: txt for nm, txt, F in G.render("136m", dp) if F["stage"] == "C"}
        T("C runs = seeds 4-7 at 9.6e-3, with --save_model, L40S, 136m without --ckpt_every",
          cs == [("C", "none", "9.6e-3", s) for s in G.CONF] and len(rendc) == 4
          and all("--save_model" in t and "--partition=l40s_public" in t
                  and "--ckpt_every" not in next(l for l in t.splitlines() if l.startswith('ARGS="')) for t in rendc.values()), cs)
        _, new3, cells3 = run(r4, write=True, out=quiet)
        T("after the C: final, nothing new", not new3 and by(cells3, "136m", "none")["decision"] == "final")
        r5 = fresh("open_twice")
        grid(r5, "136m", "none", {"1.2e-3": 50, "2.4e-3": 60, "4.8e-3": 65, "9.6e-3": 70})
        run(r5, write=True, out=quiet)
        grid(r5, "136m", "none", {"1.92e-2": 75})
        log, new, cells = run(r5, write=True, out=quiet)
        T("X better by > 2 -> still open -> STOP (no second X)", by(cells, "136m", "none")["decision"] == "STOP"
          and log[-1]["stage"] == "STOP", by(cells, "136m", "none")["text"])
        _, new4, cells4 = run(r5, write=True, out=quiet)
        T("after a STOP: nothing more", not new4 and by(cells4, "136m", "none")["decision"] == "stop")
        r6 = fresh("checkfail")
        grid(r6, "136m", "win-ctx", {"6e-4": 88, "1.2e-3": 89, "2.4e-3": 90, "4.8e-3": 86})
        fake_new_run(r6, "136m", "win-ctx", "1.2e-3", 2, 89, mutate=lambda r: r["args"].update(lam=1.0))
        c = by(run(r6, out=quiet)[2], "136m", "win-ctx")
        T("a run whose args differ (lam 1): CHECK7 fails -> WITHHELD, nothing decided", c["decision"] == "withheld"
          and "CHECK7 FAILED" in c["text"], c["text"])
        r7 = fresh("doublepath")
        grid(r7, "136m", "win-ctx", {"6e-4": 88, "1.2e-3": 89, "2.4e-3": 90, "4.8e-3": 86}, check="synthetic")
        tag, rf, cf = G.run_file("136m", "win-ctx", "2.4e-3", 1)
        f = r7 / B / "out" / tag / cf
        j = json.loads(f.read_text()); j["info"]["area"] += 0.05; f.write_text(json.dumps(j))
        c = by(run(r7, out=quiet)[2], "136m", "win-ctx")
        T("double path: the check's area off by 0.05 -> WITHHELD", c["decision"] == "withheld" and "double path" in c["text"], c["text"])
        r8 = fresh("own")
        logs = base / "logs"
        logs.mkdir(exist_ok=True)
        grid(r8, "136m", "none", {"1.2e-3": 50, "2.4e-3": 60, "4.8e-3": 65, "9.6e-3": 55})
        tag, rf, cf = G.run_file("136m", "none", "4.8e-3", 3)
        (r8 / B / "out" / tag / rf).unlink()
        own_run = "psz-136m-lr4.8e-3-none-s3"
        trained = f"=== {own_run} start\nmodel: 136.3M parameters; gradient checkpointing off\n{TRAIN_MARK} 250 steps; EMA 0.9800\nTraceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory\n"
        (logs / f"llmadd_{own_run}_5001.out").write_text(trained)
        (logs / f"llmadd_{own_run}_5002.out").write_text(trained)
        use_sacct([("5001", f"llmadd_{own_run}", "OUT_OF_MEMORY"), ("5002", f"llmadd_{own_run}", "FAILED")])
        c = by(run(r8, out=quiet)[2], "136m", "none")
        T("a missing run (not listed) -> wait", c["decision"] == "wait", c["text"])
        ownf = r8 / B / "OWN_FAILURES.txt"
        good = f"{own_run} 5001:OUT_OF_MEMORY 5002:FAILED 2026-11-05T22:00:00-05:00   # OOM twice"
        cases = [("a bare run name (the old format)", f"{own_run}   # crashed", "want '<run>"),
                 ("the same job id twice", f"{own_run} 5001:OUT_OF_MEMORY 5001:OUT_OF_MEMORY 2026-11-05T22:00:00-05:00", "two DIFFERENT job ids"),
                 ("a TIMEOUT state", f"{own_run} 5001:OUT_OF_MEMORY 5002:TIMEOUT 2026-11-05T22:00:00-05:00", "failure states must be"),
                 ("a timestamp without a zone", f"{own_run} 5001:OUT_OF_MEMORY 5002:FAILED 2026-11-05T22:00:00", "ISO 8601 with a time zone"),
                 ("a state that sacct contradicts", f"{own_run} 5001:FAILED 5002:FAILED 2026-11-05T22:00:00-05:00", "5001 is OUT_OF_MEMORY in sacct")]
        for what, entry, msg in cases:
            ownf.write_text(entry + "\n")
            lines.clear()
            c = by(run(r8, out=quiet)[2], "136m", "none")
            T(f"I-A: an OWN_FAILURES entry with {what} -> WARNING, ignored (the arm just waits for the missing run)",
              c["decision"] == "wait" and any(l.startswith("WARNING: OWN_FAILURES.txt entry") and msg in l for l in lines), (c["text"], lines[:2]))
        ownf.write_text("# own failures\n" + good + "\npsz-136m-lr4.8e-3-nnoe-s2 5001:FAILED 5002:FAILED 2026-11-05T22:00:00-05:00\n")
        lines.clear()
        log8, new8, cells8 = run(r8, write=True, out=quiet)
        T("I5 / I-A: a valid entry (two different jobs that both trained and failed, states = sacct, zoned timestamp) -> BOTH "
          "136m arms NOT DETERMINABLE, nothing decided, never scored",
          by(cells8, "136m", "none")["decision"] == "nd" and by(cells8, "136m", "win-ctx")["decision"] == "nd"
          and not [e for e in log8 if e["size"] == "136m"] and "NOT DETERMINABLE" in by(cells8, "136m", "none")["text"]
          and by(cells8, "294m", "none")["decision"] == "wait", (by(cells8, "136m", "none")["text"], lines[:3]))
        T("M6: a misspelled OWN_FAILURES name prints a WARNING", any("WARNING" in l and "nnoe" in l for l in lines), lines[:3])
        (logs / f"llmadd_{own_run}_5002.out").write_text(f"=== {own_run} start\nrun guard: run_whole_none_seed3_x.json is being computed by running job 5001; exiting\n")
        ownf.write_text(good + "\n")
        lines.clear()
        c = by(run(r8, out=quiet)[2], "136m", "none")
        T("I-B: the resubmission exited at the run guard (stale claim, M8) -> NOT a recurrence: WARNING, ignored",
          c["decision"] == "wait" and any("never reached the trainer's training call" in l and "5002" in l for l in lines), lines[:2])
        (logs / f"llmadd_{own_run}_5002.out").write_text(trained)
        ownf.write_text(good + "\n" + "psz-136m-lr2.4e-3-none-s1 5003:FAILED 5004:FAILED 2026-11-05T22:00:00-05:00\n")
        lines.clear()
        c = by(run(r8, out=quiet)[2], "136m", "none")
        T("I-A: an entry for a run that finished with a passing check -> the restored WARNING, that entry ignored (the valid one still applies)",
          c["decision"] == "nd" and any(l.startswith("WARNING: psz-136m-lr2.4e-3-none-s1 is listed as an own failure but finished with a passing check: entry ignored") for l in lines), lines[:3])
        (r8 / B / "FIRST_READOUT.json").write_text(json.dumps({"time": "2026-11-06T00:30:00-05:00", "own_entries": []}))
        ownf.write_text(good + "\n")
        lines.clear()
        c = by(run(r8, out=quiet)[2], "136m", "none")
        T("I-A: a valid-looking entry that was not in the file at the first read-out (FIRST_READOUT.json) -> WARNING, ignored",
          c["decision"] == "wait" and any("added after the first read-out" in l for l in lines), lines[:2])
        (r8 / B / "FIRST_READOUT.json").write_text(json.dumps({"time": "2026-11-06T00:30:00-05:00", "own_entries": [good.split("#")[0].strip()]}))
        lines.clear()
        c = by(run(r8, out=quiet)[2], "136m", "none")
        T("I-A: the same entry recorded at the first read-out, timestamp before it -> honoured (nd)", c["decision"] == "nd", lines[:2])
        T("  the honoured entry prints a NOTE with each job's last error line (audit amend2 m5)",
          any(l.startswith(f"NOTE: OWN_FAILURES.txt entry honoured for {own_run}") and "CUDA out of memory" in l for l in lines), lines[:3])
        (r8 / B / "FIRST_READOUT.json").write_text('{"time": "2026-11-06T00:30:00-05:00", "own_entr')   # truncated write
        lines.clear()
        c = by(run(r8, out=quiet)[2], "136m", "none")
        T("amendment 3: a FIRST_READOUT.json that cannot be parsed -> WARNING, no entry honoured (wait, not nd)",
          c["decision"] == "wait" and any("FIRST_READOUT.json exists but cannot be parsed" in l for l in lines), lines[:2])
        (r8 / B / "FIRST_READOUT.json").unlink()
        ownf.write_text(f"{own_run} 5001:OUT_OF_MEMORY 5002:FAILED 2099-01-01T00:00:00-05:00\n")
        lines.clear()
        c = by(run(r8, out=quiet)[2], "136m", "none")
        T("amendment 3: an entry with a timestamp in the future -> WARNING, ignored", c["decision"] == "wait"
          and any("timestamp is in the future" in l for l in lines), lines[:2])
        tag3, rf3, cf3 = G.run_file("136m", "none", "4.8e-3", 2)   # a finished run whose CHECK7 failed
        ck3 = r8 / B / "out" / tag3 / cf3
        j3 = json.loads(ck3.read_text()); j3["pass"] = False; ck3.write_text(json.dumps(j3))
        n3 = "psz-136m-lr4.8e-3-none-s2"
        for jid in ("5011", "5012"):
            (logs / f"llmadd_{n3}_{jid}.out").write_text(trained)
        use_sacct([("5011", f"llmadd_{n3}", "FAILED"), ("5012", f"llmadd_{n3}", "FAILED")])
        ownf.write_text(f"{n3} 5011:FAILED 5012:FAILED 2026-11-05T22:00:00-05:00\n")
        lines.clear()
        c = by(run(r8, out=quiet)[2], "136m", "none")
        T("amendment 3: an entry for a run with a finished JSON whose CHECK7 failed -> WARNING, not honoured (not nd; rule 6 withholds the arm)",
          c["decision"] != "nd" and any(l.startswith(f"WARNING: {n3} is listed as an own failure but has a finished run JSON (CHECK7 failed)") for l in lines),
          (c["text"], lines[:2]))
        r8b = fresh("dead")   # an OOM'd job: unfinished run JSON, no check, its claim's job ended OUT_OF_MEMORY -> DEAD + ALERT
        grid(r8b, "731m", "win-ctx", {"3e-4": 86, "6e-4": 89, "1.2e-3": 89.5, "2.4e-3": 88})
        nm = "psz-731m-lr6e-4-win-ctx-s2"
        tag, rf, cf = G.run_file("731m", "win-ctx", "6e-4", 2)
        pth = r8b / B / "out" / tag / rf
        j = json.loads(pth.read_text()); j["curve"] = j["curve"][:4]; pth.write_text(json.dumps(j))
        (r8b / B / "out" / tag / cf).unlink()
        (r8b / B / "out" / tag / (rf + ".claim")).write_text("4242\n")
        use_sacct([("4242", f"llmadd_{nm}", "OUT_OF_MEMORY")])
        lines.clear()
        c = by(run(r8b, out=quiet)[2], "731m", "win-ctx")
        T("I5: the claim's job ended OUT_OF_MEMORY -> 'dead' + ALERT naming job 4242 as the claim's and the latest job (the same)",
          c["decision"] == "wait" and "dead (job 4242 OUT_OF_MEMORY)" in c["text"]
          and any(l.startswith(f"ALERT: {nm}") and "the same job as the claim" in l and "4242" in l for l in lines), (c["text"], lines[:2]))
        use_sacct([("4242", f"llmadd_{nm}", "OUT_OF_MEMORY"), ("4300", f"llmadd_{nm}", "FAILED")])
        lines.clear()
        c = by(run(r8b, out=quiet)[2], "731m", "win-ctx")
        T("I-B: a resubmission 4300 FAILED but the claim still says 4242 -> ALERT names both and says NOT the claim's job (no recurrence)",
          "dead (job 4300 FAILED)" in c["text"] and any("NOT the claim's job -> job 4300" in l for l in lines), (c["text"], lines[:2]))
        use_sacct([("4242", f"llmadd_{nm}", "OUT_OF_MEMORY"), ("4301", f"llmadd_{nm}", "PENDING")])
        lines.clear()
        c = by(run(r8b, out=quiet)[2], "731m", "win-ctx")
        T("a resubmission still PENDING -> plain 'running', no ALERT", "running" in c["text"] and "dead" not in c["text"]
          and not any(l.startswith("ALERT") for l in lines), (c["text"], lines[:2]))
        use_sacct([("6001", f"llmadd_{nm}", "RUNNING"), ("6002", f"llmadd_{nm}", "FAILED")])
        (r8b / B / "out" / tag / (rf + ".claim")).write_text("6001\n")
        lines.clear()
        c = by(run(r8b, out=quiet)[2], "731m", "win-ctx")
        T("amendment 3: the claim's job 6001 still RUNNING, a later duplicate 6002 FAILED (exited at the run guard) -> 'running', no ALERT",
          "6e-4 s2 running" in c["text"] and "dead" not in c["text"] and not any(l.startswith("ALERT") for l in lines), (c["text"], lines[:2]))
        use_sacct([("6002", f"llmadd_{nm}", "FAILED")])
        lines.clear()
        c = by(run(r8b, out=quiet)[2], "731m", "win-ctx")
        T("amendment 3: sacct has no record of the claim's job 6001 -> WARNING (not silent)",
          any(l.startswith(f"WARNING: {nm}: sacct has no record of its claim's job 6001") for l in lines), lines[:2])
        (r8b / B / "out" / tag / (rf + ".claim")).write_text("4242\n")
        nm2 = "psz-731m-lr6e-4-win-ctx-s3"
        tag, rf, cf = G.run_file("731m", "win-ctx", "6e-4", 3)
        (r8b / B / "out" / tag / rf).unlink()
        (r8b / B / "out" / tag / cf).unlink()
        use_sacct([("4400", f"llmadd_{nm2}", "COMPLETED")])
        lines.clear()
        c = by(run(r8b, out=quiet)[2], "731m", "win-ctx")
        T("m1: a job that exited before the trainer's claim (e.g. HOLD, exit 0) -> 'missing' + WARNING naming job 4400 COMPLETED",
          "6e-4 s3 missing" in c["text"] and any(l.startswith(f"WARNING: {nm2}: no run JSON and no claim") and "4400" in l and "COMPLETED" in l for l in lines), (c["text"], lines[:3]))
        use_sacct(rc=1)
        lines.clear()
        run(r8b, out=quiet)
        T("m1: sacct exiting non-zero -> a WARNING (never silent)", any(l.startswith("WARNING: sacct exited 1") for l in lines), lines[:2])
        use_sacct(exc=subprocess.TimeoutExpired("sacct", 180))
        lines.clear()
        run(r8b, out=quiet)
        T("m1: a sacct timeout -> a WARNING", any(l.startswith("WARNING: sacct timed out") for l in lines), lines[:2])
        use_sacct([])
        r9 = fresh("running")
        grid(r9, "136m", "none", {"1.2e-3": 50, "2.4e-3": 60, "4.8e-3": 65, "9.6e-3": 62})
        tag, rf, cf = G.run_file("136m", "none", "2.4e-3", 0)
        p = r9 / B / "out" / tag / rf
        j = json.loads(p.read_text()); j["curve"] = j["curve"][:40]; p.write_text(json.dumps(j))
        (r9 / B / "out" / tag / cf).unlink()
        c = by(run(r9, out=quiet)[2], "136m", "none")
        T("an unfinished run -> wait", c["decision"] == "wait" and "running" in c["text"], c["text"])
        r10 = fresh("lm0")
        grid(r10, "731m", "none", {"1.2e-3": 50, "2.4e-3": 60, "4.8e-3": 65, "9.6e-3": 62}, check="synthetic")
        tag, rf, cf = G.run_file("731m", "none", "1.2e-3", 1)
        p = r10 / B / "out" / tag / rf
        j = json.loads(p.read_text()); j["curve"][0]["lm"] += 0.002; p.write_text(json.dumps(j))
        c = by(run(r10, out=quiet)[2], "731m", "none")
        T("integrity re-check (731m, ptsz_refs.json reference): step-0 lm +0.002 -> WITHHELD", c["decision"] == "withheld"
          and "step-0 lm" in c["text"], c["text"])
        r11 = fresh("unstable")   # M7: the reference is the arm's lowest none lr including a decided X
        grid(r11, "294m", "none", {"2.4e-3": 65, "4.8e-3": 60, "9.6e-3": 55, "1.92e-2": 50})
        run(r11, write=True, out=quiet)                                   # open lower -> X 1.2e-3
        grid(r11, "294m", "none", {"1.2e-3": 50}, val_loss=2.0)           # X lower, much better val_loss than 3.0 + 0.5
        c = by(run(r11, out=quiet)[2], "294m", "none")
        T("M7: after a lower X the unstable reference is the X lr (1.2e-3): the grid runs (val_loss 3.0 > 2.0 + 0.5) are flagged",
          c["unstable_ref"] == "1.2e-3" and len(c["unstable"]) == 16 and unstable_ref_lr("294m", None) == "2.4e-3", (c.get("unstable_ref"), c.get("unstable")))
        bad_logs = [
            [dict(seq=0, size="136m", arm="none", stage="X", lr="3.84e-2")],                       # two steps out
            [dict(seq=0, size="136m", arm="none", stage="C", lr="4.8e-3"), dict(seq=1, size="136m", arm="none", stage="X", lr="1.92e-2")],
            [dict(seq=0, size="136m", arm="none", stage="STOP", lr=None), dict(seq=1, size="136m", arm="none", stage="C", lr="4.8e-3")],
            [dict(seq=0, size="136m", arm="none", stage="C", lr="1.92e-2")],                       # C outside grid + X
            [dict(seq=1, size="136m", arm="none", stage="C", lr="4.8e-3")],                        # seq
            [dict(seq=0, size="93m", arm="none", stage="C", lr="2.4e-3")]]                         # not a size of this batch
        refused = 0
        for i, L in enumerate(bad_logs):
            f = base / f"badlog{i}.json"
            f.write_text(json.dumps({"log": L}))
            try:
                G.decisions(f)
            except AssertionError:
                refused += 1
        T(f"the generator refuses malformed decision logs ({len(bad_logs)}/{len(bad_logs)})", refused == len(bad_logs), refused)
    finally:
        globals()["_sacct"] = real_sacct
        shutil.rmtree(base, ignore_errors=True)
    print(f"SELECT SELFTEST {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    return ok == n


def main():
    if "--selftest" in sys.argv:
        w = sys.argv[sys.argv.index("--work") + 1] if "--work" in sys.argv else None
        sys.exit(0 if selftest(w) else 1)
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else ROOT_DEFAULT
    print(f"# ptsz selection {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} root {root}; generator "
          f"{sha(GEN)}; readout_rules {sha(RULES_DIR / 'readout_rules.py')}")
    run(root, write="--write" in sys.argv)


if __name__ == "__main__":
    main()
