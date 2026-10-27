"""pttpp: the pre-registered selection step of one group (runs/pttpp/AUDIT.md "Selection and stages"), run by
maybe_select.sh at the end of every job of the group (and by hand: bash maybe_select.sh <group>).
  --check   (no lock, cheap) exit 0 iff there is something to do: an arm without a final decision (C or STOP) whose
            stage A (+ X) runs are all finished (7%: the run JSON reaches its last step; 2%: + check.json written), or a
            rendered run of a decided stage that is not submitted yet; else exit 1
  --decide  the read-out of the group (7%: <readout>/pttpp_summary.py --select <group>; 2%: <readout>/pt2main_summary.py
            --gen make_jobs_pttpp.py --res out2, pt2main's own rules) -> per arm:
              incomplete                          -> wait (nothing recorded)
              withheld (an integrity failure)     -> STOP: the user decides
              no open bracket                     -> C at the pick (the confirm seeds; 1.07B: seeds 2-7)
              open at an edge, no X decided yet   -> X: one ladder step beyond that edge, on the selection seeds
              open again after the X lr ran       -> STOP: the user decides
            decisions/<group>.json is append-only (seq, arm, stage, lr, time, job, basis); then the disk guard (myquota
            /scratch below 97%), make_jobs_pttpp.py --group <group> --test-only (unsubmitted jobs only) and, only if every
            line is OK, --submit
Overrides (plumbing tests only): PTTPP_DIR, PTTPP_ROOT, PTTPP_GEN, PTTPP_READOUT, PTTPP_PY.
Amendment T2: the state of an arm is the decision log folded in order (make_jobs_pttpp.effective: REOPEN withdraws a C);
before any submission the cost guard: the planned GPU-h (make_jobs_pttpp.planned_hours) must stay within COST_MAX
(B200 180 h, L40S + A100 300 h), else nothing is submitted ("COST GUARD" in the log; the coordinator decides).
Amendment T1 (build audit, before any pttpp result was read): a 7% run counts as finished only once its job wrote
check_s<seed>.json (as the 2% check.json; a run JSON without it read as 'check missing' -> WITHHELD -> a permanent STOP);
the read-out is retried 3 times 30 s apart (a run JSON being rewritten by a running job); TMPDIR / XDG_RUNTIME_DIR of
the submitting job are not passed to the new jobs."""
import datetime
import time
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

R = Path(os.environ.get("PTTPP_DIR", "/scratch/yh6210/llmadd_torch/runs/pttpp"))
ROOT = Path(os.environ.get("PTTPP_ROOT", "/scratch/yh6210/llmadd_torch/runs"))
GEN = Path(os.environ.get("PTTPP_GEN", "/scratch/yh6210/llmadd_torch/make_jobs_pttpp.py"))
RO = Path(os.environ.get("PTTPP_READOUT", str(R / "readout" / "tools")))
PY = os.environ.get("PTTPP_PY", "/scratch/yh6210/miniconda3/envs/open_r1/bin/python")
DISK_MAX = 97.0


def load_gen():
    spec = importlib.util.spec_from_file_location("pttpp_gen", GEN)
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    assert G.R == R, (G.R, R, "the generator's batch dir is not this one")
    return G


def log(gid, msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] job {os.environ.get('SLURM_JOB_ID', 'none')}: {msg}"
    with open(R / f"select_{gid}.log", "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def finished(G, gid, F):
    if F["setting"] == "2":
        t = R / "out2" / F["name"]
        f = list((t / F["tag"]).glob("run_*.json"))
        if not (len(f) == 1 and (t / "check.json").exists()):
            return False
    else:
        target = "none" if F["arm"] == "none" else "helix3"
        f = [R / "out" / F["tag"] / f"run_whole_{target}_seed{F['seed']}_{F['tag']}.json"]
        if not (f[0].exists() and (R / "out" / F["tag"] / f"check_s{F['seed']}.json").exists()):   # T1: the job's check is written
            return False
    try:
        r = json.loads(f[0].read_text())
        return r["curve"][-1]["step"] >= r["steps"] - 1
    except Exception:
        return False


def state(G, gid):
    eff = G.effective(gid)
    dec = {(a, k): {"lr": eff[a][k]} for a in G.ARMS for k in ("X", "C") if eff[a][k]}
    dec.update({(a, "STOP"): {} for a in G.ARMS if eff[a]["STOP"]})
    open_arms = [a for a in G.ARMS if (a, "C") not in dec and (a, "STOP") not in dec]
    runs = G.runs_of(gid)
    ready = [a for a in open_arms
             if all(finished(G, gid, G.facts(gid, a, lr, s, st)) for st, aa, lr, s in runs if aa == a and st in ("A", "X"))]
    unsub = [G.facts(gid, a, lr, s, st)["name"] for st, a, lr, s in runs if not (R / f"SUBMITTED_{G.facts(gid, a, lr, s, st)['name']}").exists()]
    return dec, ready, unsub


def readout(G, gid):
    """-> {arm: dict(status, best, open, ...)} from the frozen read-out"""
    tmp = R / "locks" / f"sel_{gid}.json"
    tmp.unlink(missing_ok=True)
    env = dict(os.environ, HF_HOME="/scratch/yh6210/llmadd_torch/hf_cache", PYTHONPATH="/scratch/yh6210/llmadd_torch/pylib")
    if G.GROUPS[gid]["setting"] == "7":
        cmd = [PY, str(RO / "pttpp_summary.py"), "--select", gid, "--json", str(tmp), "--root", str(ROOT), "--gen", str(GEN)]
    else:
        cmd = [PY, str(RO / "pt2main_summary.py"), "--gen", str(GEN), "--res", str(R / "out2"), "--json", str(tmp)]
    o = subprocess.run(cmd, capture_output=True, text=True, cwd=RO, env=env)
    (R / f"readout_{gid}.txt").write_text(o.stdout + o.stderr)
    if o.returncode or not tmp.exists():
        raise RuntimeError(f"read-out failed (rc {o.returncode}); see readout_{gid}.txt")
    j = json.loads(tmp.read_text())
    if G.GROUPS[gid]["setting"] == "7":
        assert j["gid"] == gid
        return j["arms"]
    b = j[gid]   # pt2main_summary's SELECTION[block]
    out = {}
    for a in G.ARMS:
        if a in b["best"]:
            out[a] = {"status": "ok", "best": b["best"][a], "open": b["open"].get(a), "grid": b["grid"].get(a)}
        else:
            out[a] = {"status": b["nobest"].get(a, "incomplete"), "grid": b["grid"].get(a)}
    return out


def decide(G, gid):
    dec, ready, unsub = state(G, gid)
    new = []
    if ready:
        for attempt in range(3):   # T1: a run JSON of a running job can be caught mid-write
            try:
                sel = readout(G, gid)
                break
            except Exception as e:
                log(gid, f"read-out attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(30)
        for a in ready:
            s = sel[a]
            basis = {k: s.get(k) for k in ("status", "best", "open", "grid", "mean", "bpb", "near", "reasons")}
            if s["status"] == "incomplete":
                log(gid, f"{a}: the read-out says incomplete although the runs look finished; waiting")
                continue
            if s["status"] != "ok":
                new.append(dict(arm=a, stage="STOP", lr=None, why=f"{s['status']}: {(s.get('reasons') or [''])[0]}", basis=basis))
                continue
            assert s["best"] in s["grid"] and s["best"] in G.LADDER, s
            if not s["open"]:
                new.append(dict(arm=a, stage="C", lr=s["best"], basis=basis))
            elif (a, "X") in dec:
                new.append(dict(arm=a, stage="STOP", lr=s["best"], why=f"bracket still OPEN at the {s['open']} edge after one "
                                f"extension (best {s['best']}): the user decides", basis=basis))
            else:
                edge_lr = max(s["grid"], key=float) if s["open"] == "upper" else min(s["grid"], key=float)
                assert edge_lr == s["best"], (edge_lr, s)
                new.append(dict(arm=a, stage="X", lr=G.step_out(s["best"], s["open"]), basis=basis))
    if new:
        f = R / "decisions" / f"{gid}.json"
        f.parent.mkdir(exist_ok=True)
        old = G.decisions(gid)["log"]
        for i, e in enumerate(new):
            e.update(seq=len(old) + i, time=datetime.datetime.now().isoformat(timespec="seconds"),
                     job=os.environ.get("SLURM_JOB_ID", "none"))
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps({"group": gid, "log": old + new}, indent=1))
        tmp.replace(f)
        assert G.decisions(gid)["log"][:len(old)] == old   # append-only
        for e in new:
            log(gid, f"DECISION {e['arm']} {e['stage']} {e['lr']}" + (f" ({e['why']})" if e.get("why") else "")
                + f"; basis {json.dumps(e['basis'], default=str)[:600]}")
    dec, ready, unsub = state(G, gid)
    if not unsub:
        log(gid, "nothing to submit")
        return
    ph = G.planned_hours()
    over = {k: round(v, 1) for k, v in ph.items() if v > G.COST_MAX[k]}
    if over:
        log(gid, f"COST GUARD: planned GPU-h {({k: round(v, 1) for k, v in ph.items()})} exceed {G.COST_MAX} in {over}: "
                 f"{len(unsub)} jobs NOT submitted (the coordinator decides; rerun maybe_select.sh after approval)")
        return
    log(gid, f"planned GPU-h {({k: round(v, 1) for k, v in ph.items()})} within {G.COST_MAX}")
    q = subprocess.run(["myquota"], capture_output=True, text=True)
    m = re.search(r"/scratch\s.*?\(([\d.]+)%\)", q.stdout.replace("\x1b", " "))
    if m and float(m.group(1)) >= DISK_MAX:
        log(gid, f"DISK GUARD: /scratch at {m.group(1)}% >= {DISK_MAX}%: {len(unsub)} jobs NOT submitted (rerun maybe_select.sh later)")
        return
    if not m:
        log(gid, "myquota could not be parsed; submitting anyway")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_")) and k not in ("TMPDIR", "XDG_RUNTIME_DIR")}
    t = subprocess.run([PY, str(GEN), "--group", gid, "--test-only"], capture_output=True, text=True, env=env)
    log(gid, f"--test-only rc {t.returncode}:\n" + t.stdout + t.stderr)
    if t.returncode:
        log(gid, "--test-only failed: nothing submitted")
        sys.exit(1)
    s = subprocess.run([PY, str(GEN), "--group", gid, "--submit"], capture_output=True, text=True, env=env)
    log(gid, f"--submit rc {s.returncode}:\n" + s.stdout + s.stderr)


def main():
    gid = sys.argv[sys.argv.index("--group") + 1]
    G = load_gen()
    if gid not in G.GROUPS:
        print(f"{gid}: no selection (bridge or unknown)")
        sys.exit(1)
    if "--check" in sys.argv:
        dec, ready, unsub = state(G, gid)
        sys.exit(0 if ready or unsub else 1)
    decide(G, gid)


if __name__ == "__main__":
    main()
