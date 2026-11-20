#!/usr/bin/env python3
"""pt6v2 AUDIT.md Amendment V4 (post-hoc, user-approved 2026-11-20): two cells x seeds 16-23 (16 runs), exploratory, no verdict rule.

  all6full@1e-3 = the registered all6@1e-3 job with --window 0.05,1.0 (prior on from step 916 through the last step 18309) instead of
                  0.05,0.4. Question: does keeping the priors on prevent the post-window drop (torus 98->90, box5 81->53, dim 93->84)?
  all4@2e-3     = the registered all6@2e-3 job with --prior_domains numbers,dim,cities,countries (torus and box5 priors off; --domains
                  stays all six, so data and evals are the same). Question: does dim's +13.5 over only-dim@2e-3 need the torus / box5 priors?

Rendering: make_jobs_pt6v2.render_v2 (frozen, sha asserted, read only) with the arm patched in for one call (pri_of / arms / want_flags /
intent_v2 / desc_v2 and, for all6full, make_jobs_ptdom.W), so every guard of the frozen render runs on the new arm: the parsed ARGS
equal want_flags, the intent is the run's, the dry block asserts each domain's prior / lam. Then two counted edits: the out dir
runs/pt6v2/out/<name> -> runs/pt6v2/out_v4/<name> and one provenance comment line. Independent check: the same seed's registered
base job (all6@1e-3 / all6@2e-3, unpatched render_v2 on the same device and slot) with the expected substitutions applied must equal
the V4 text on every line except the header description, the provenance line and the intent line; the intent JSON must equal the base
intent with exactly the expected fields changed.

Nothing of the registered batch is touched: CELLS.json, make_jobs_pt6v2.py, readout_pt6v2.py, ALLOC.json, POSTHOC_V3.json, the 113
job files and SUBMITTED_ flags in runs/pt6v2/. Job files, SUBMITTED_ flags and run dirs of V4 live in runs/pt6v2/out_v4/ (the read-out
reads runs/pt6v2/out/ only, and its run-name regex has no all6full / all4 arm). Record: runs/pt6v2/POSTHOC_V4.json.
Device: every run on B200 (ALLOC.json gives s23 a100; a100_tandon is blocked by the group GPU quota (QOSGrpGRES) at build time, so
s23 moves to B200: a device change vs its registered runs, flagged only). 3 B200 singleton chains (the existing slot names
llmadd_pt6v2-b200-slot0/1/2), order s16 all6full, s16 all4, s17 all6full, ... round-robin over the 3 chains.

Usage:  posthoc_v4_jobs.py [--check] [--print] [--write] [--test-only [N]] [--submit]
  (no flag / --check)  render the 16 jobs + their base jobs, the independent diff check, bash -n each, every guard; nothing is written
  --print              the args, the #SBATCH lines and the intent fields that differ from the base job
  --write              write runs/pt6v2/out_v4/job_<name>.sbatch (never over a submitted one) and POSTHOC_V4.json (without job ids)
  --test-only [N]      sbatch --test-only on the first N on-disk job files (all 16 without N)
  --submit             sbatch each on-disk job file in order (it must equal this rendering), write out_v4/SUBMITTED_<name> with sbatch's
                       stdout and add the job ids to POSTHOC_V4.json; a name with a SUBMITTED_ flag is never submitted again
"""
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pt6v2"
OV4 = R / "out_v4"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]   # noqa: E731
MJ2_SHA, CELLS_SHA, V3_SHA, ALLOC_SHA = "bb15d248afd8", "ea9a9b01f081", "75bed7fe30ee", "2d96730f431d"
assert sha(B / "make_jobs_pt6v2.py") == MJ2_SHA, "make_jobs_pt6v2.py changed (frozen after the V2 build)"
assert sha(R / "CELLS.json") == CELLS_SHA, "CELLS.json changed: Amendment V4 never rewrites the registered cell set"
assert sha(R / "POSTHOC_V3.json") == V3_SHA, "POSTHOC_V3.json changed"
assert sha(R / "ALLOC.json") == ALLOC_SHA, "ALLOC.json changed"
sys.path.insert(0, str(B))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_pt6v2 as MJ2   # noqa: E402  (read only)
sys.argv = _argv
assert Path(MJ2.__file__).resolve() == (B / "make_jobs_pt6v2.py").resolve()
MJ = MJ2.MJ

SEEDS = MJ2.SEEDS
FULL, REG_W = "0.05,1.0", "0.05,0.4"
FOUR = ("numbers", "dim", "cities", "countries")
CELLS = {   # V4 arm -> (lr, the registered base cell it changes one argument of)
    "all6full": ("1e-3", ("all6", "1e-3")),
    "all4": ("2e-3", ("all6", "2e-3")),
}
SLOTS = (0, 1, 2)   # the 3 B200 chains (the user's 3-GPU B200 cap; QOSMaxGRESPerUser enforces it)
ORDER = [(s, c) for s in SEEDS for c in ("all6full", "all4")]
DEV = {s: "b200" for s in SEEDS}
DEV_NOTE = ("ALLOC.json gives s23 a100; a100_tandon was blocked by the group GPU quota (QOSGrpGRES, our p2z jobs pending) at build "
            "time 2026-11-20, so s23 runs on B200 (device change vs its registered a100 runs: flagged only, not a rule)")
ENV_U = ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_STEP_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_NTASKS")
SBATCH = ["env"] + [x for v in ENV_U for x in ("-u", v)] + ["sbatch"]
POSTHOC = R / "POSTHOC_V4.json"
APPROVAL = ('user-approved 2026-11-20 ("ok，怎么快怎么来", relayed by the main session; B200 approved); post-hoc, exploratory, '
            "no verdict rule")
WHY = {"all6full": "does keeping the priors on prevent the post-window drop (all6@1e-3 8-seed means: torus 98->90, box5 81->53, "
                   "dim 93->84 from window close to final)?",
       "all4": "does dim's +13.5 lift of all6@2e-3 over only-dim@2e-3 (same lr) need the torus and box5 priors?"}
RULE = ("descriptive: each V4 run is read against the existing pt6v2 runs of the same seed (all6full vs all6@1e-3; all4 vs all6@2e-3 "
        "and only-<m>@2e-3); no verdict rule, no gate; s23 device change (b200 vs a100) flagged; readout_pt6v2.py does not read these "
        "runs (out_v4/, arm names outside its regex)")
PROV = ("# AUDIT.md Amendment V4 (post-hoc, 2026-11-20): rendered by runs/pt6v2/posthoc_v4_jobs.py = make_jobs_pt6v2.render_v2 with the "
        "arm {arm} patched in (one argument changed vs the registered {base}); out dir runs/pt6v2/out_v4/ (not read by readout_pt6v2.py)\n")


def trainer_on(steps, window):
    """pretrain_range.py prior_on (schedule window): on iff w0 * steps <= s < w1 * steps, for s in range(start, steps)."""
    w0, w1 = (float(v) for v in window.split(","))
    return [s for s in range(steps) if w0 * steps <= s < w1 * steps]


def full_window_facts(steps):
    """-> (events, on_frac) of --window 0.05,1.0 as the trainer records them: prior_events holds toggles only ([s, 1 / 0] when prior_on
    changes between consecutive steps), so a window that runs to the end has one event; on_frac = on steps / steps."""
    on = trainer_on(steps, FULL)
    assert on == list(range(on[0], steps)) and on[-1] == steps - 1, "window 0.05,1.0 does not keep the prior on through the last step"
    ev, frac = MJ2.load_P().window_events(steps, FULL)   # make_rb6_tasks.window_events: the same rule, off event at steps (never reached)
    assert ev == [[on[0], 1], [steps, 0]] and frac == len(on) / steps, (ev, frac)
    return [[on[0], 1]], len(on) / steps


@contextmanager
def v4_arm(arm):
    """Patch make_jobs_pt6v2 (and make_jobs_ptdom.W for all6full) so render_v2 renders the V4 arm; restored on exit."""
    names = ("pri_of", "arms", "want_flags", "intent_v2", "desc_v2")
    o = {k: getattr(MJ2, k) for k in names}
    w_keep = MJ.W

    def pri_of(a, doms):
        if a == "all6full":
            return MJ2.manifolds(doms)
        if a == "all4":
            assert set(FOUR) <= set(MJ2.manifolds(doms)), doms
            return FOUR
        return o["pri_of"](a, doms)

    def arms(doms):
        return o["arms"](doms) + (arm,)

    def want_flags(a, doms):
        w = o["want_flags"](a, doms)
        if a == "all6full":
            assert w["--window"] == REG_W
            w = dict(w, **{"--window": FULL})
        return w

    def intent_v2(P, doms, a, lr, seed, ref, dev, canary=False):
        I = o["intent_v2"](P, doms, a, lr, seed, ref, dev, canary)   # asserts the registered window events for the prior arms
        if a == "all6full":
            assert not canary and I["args"]["window"] == REG_W and I["events"] == [[916, 1], [7324, 0]]
            ev, frac = full_window_facts(P.STEPS)
            I["args"]["window"], I["events"], I["on_frac"] = FULL, ev, frac
        return I

    def desc_v2(a, doms):
        pri = pri_of(a, doms)
        if a == "all6full":
            return ("all6full (Amendment V4, post-hoc): the six priors " + " + ".join(pri) + " (lambda 1 each, window (0.05, 1.0) = on from "
                    "step 916 through the last step, ctx; numbers = helix3 on the operands)")
        if a == "all4":
            off = [m for m in MJ2.manifolds(doms) if m not in pri]
            return ("all4 (Amendment V4, post-hoc): the four priors " + " + ".join(pri) + f" ({' / '.join(off)} priors off; lambda 1 each, "
                    "window (0.05, 0.4), ctx; numbers = helix3 on the operands)")
        return o["desc_v2"](a, doms)

    try:
        MJ2.pri_of, MJ2.arms, MJ2.want_flags, MJ2.intent_v2, MJ2.desc_v2 = pri_of, arms, want_flags, intent_v2, desc_v2
        if arm == "all6full":
            assert MJ.W.count("--window " + REG_W) == 1, MJ.W
            MJ.W = MJ.W.replace("--window " + REG_W, "--window " + FULL)
        yield
    finally:
        for k, v in o.items():
            setattr(MJ2, k, v)
        MJ.W = w_keep


def intent_of(text):
    """The intent JSON a job passes to CHECK_PY (the one line with "$OUT/check.json" '<json>')."""
    ln = [x for x in text.splitlines() if x.startswith('  "$OUT/check.json" \'')]
    assert len(ln) == 1
    return ln[0], json.loads(ln[0].split("'")[1])


def expected_intent(arm, I0, tag):
    """The base intent with exactly the fields this arm changes."""
    I = json.loads(json.dumps(I0))
    I["args"]["tag"] = f"_{tag}"
    if arm == "all6full":
        I["args"]["window"] = FULL
        I["events"], I["on_frac"] = full_window_facts(I["steps"])
    else:
        I["args"]["prior_domains"] = ",".join(FOUR)
        I["active"] = list(FOUR)
        I["lam_domains"] = {d: (1.0 if d in FOUR else 0.0) for d in I["lam_domains"]}
    return I


def diff_keys(a, b, p=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [k_ for k in sorted(set(a) | set(b)) for k_ in diff_keys(a.get(k), b.get(k), f"{p}{k}.")]
    return [] if a == b else [p[:-1]]


def build():
    doms = MJ2.decided_comp()
    assert doms is not None, "COMPOSITION.json missing"
    assert MJ2.manifolds(doms) == ("numbers", "dim", "cities", "countries", "torus", "box5"), doms
    cj = json.loads((R / "CELLS.json").read_text())
    assert cj["domains_arg"] == MJ2.domains_arg(doms) == "numbers,dim,cities,countries,torus,box5"
    alloc = {int(k): v for k, v in json.loads((R / "ALLOC.json").read_text()).items()}
    assert sorted(alloc) == list(SEEDS)
    commit = MJ2.snapshot()
    P = MJ2.load_P()
    RF, _ = MJ2.refs_v2(doms)
    others = {p.name for p in B.glob("runs/*/out*/*")}
    jobs = []
    for i, (s, arm) in enumerate(ORDER):
        assert s in RF, f"no CPU reference for seed {s}"
        lr, (barm, blr) = CELLS[arm]
        dev, slot = DEV[s], SLOTS[i % len(SLOTS)]
        b_name, b_tag, b_text, b_args, b_I = MJ2.render_v2(P, doms, barm, blr, s, RF[s], commit, dev, slot)   # unpatched base
        with v4_arm(arm):
            name, tag, text, args, I = MJ2.render_v2(P, doms, arm, lr, s, RF[s], commit, dev, slot)
        assert MJ.W.count("--window " + REG_W) == 1 and MJ2.pri_of.__module__ == "make_jobs_pt6v2", "patch not restored"
        assert name == f"pt6v2-{arm}-lr{lr}-s{s}" and tag == f"pt6v2-{arm}-lr{lr}", name
        if not (OV4 / f"SUBMITTED_{name}").exists():   # a submitted run's dir appears once it starts
            assert name not in others and not (OV4 / name).exists(), f"{name}: a run dir of this name exists"
        o_out, n_out = f"{R}/out/{name}", f"{OV4}/{name}"
        assert text.count(o_out) == 2 and args.count(o_out) == 1, (text.count(o_out), args.count(o_out))   # OUT= and --out in ARGS
        text, args = text.replace(o_out, n_out), args.replace(o_out, n_out)
        prov = PROV.format(arm=arm, base=f"{barm}@{blr}")
        text = MJ2._edit(text, "set -euo pipefail\n", prov + "set -euo pipefail\n")
        # independent check against the unpatched base job of the same seed / device / slot
        exp = b_text.replace(b_tag, tag).replace(f"{R}/out/", f"{OV4}/")
        exp = exp.replace("set -euo pipefail\n", prov + "set -euo pipefail\n")
        if arm == "all6full":
            assert exp.count("--window " + REG_W) == 1
            exp = exp.replace("--window " + REG_W, "--window " + FULL)
        else:
            six = "--prior_domains " + ",".join(MJ2.manifolds(doms))
            assert exp.count(six) == 1
            exp = exp.replace(six, "--prior_domains " + ",".join(FOUR))
            for m in ("torus", "box5"):
                assert exp.count(f" {m}:true:false:1.0") == 1
                exp = exp.replace(f" {m}:true:false:1.0", f" {m}:false:false:0.0")
        el, tl = exp.splitlines(), text.splitlines()
        assert len(el) == len(tl), (name, len(el), len(tl))
        dl = [k for k, (x, y) in enumerate(zip(el, tl)) if x != y]
        ln_i, I_txt = intent_of(text)
        _, I_base = intent_of(b_text)
        assert I_txt == I, name
        I_exp = expected_intent(arm, I_base, tag)
        assert I_txt == I_exp, (name, diff_keys(I_exp, I_txt))
        allowed = {k for k, x in enumerate(tl) if x.startswith("# pt6v2 main (pt6comb's C job body on Rev H)") or x == ln_i}
        assert set(dl) <= allowed and len(allowed) == 2, (name, [tl[k][:120] for k in dl if k not in allowed])
        assert (FULL in args) == (arm == "all6full") and (REG_W in args) == (arm == "all4"), args
        assert f"#SBATCH --job-name={MJ2.slot_name(slot)}\n#SBATCH --dependency=singleton\n" in text, name
        assert I["device"] == dev and "DEVICE b200" in text
        assert subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0, f"{name}: bash -n"
        jobs.append({"name": name, "arm": arm, "lr": lr, "seed": s, "device": dev, "alloc_device": alloc[s], "slot": slot, "text": text,
                     "args": args, "intent": I, "base": f"{barm}@{blr}", "base_name": b_name,
                     "base_job_sha12": hashlib.sha256(b_text.encode()).hexdigest()[:12],
                     "changed_intent": diff_keys(I_base, I_txt), "changed_lines": len(dl),
                     "job": OV4 / f"job_{name}.sbatch", "flag": OV4 / f"SUBMITTED_{name}"})
    return doms, commit, jobs


def manifest(doms, commit, jobs, ids=None):
    return {"amendment": "V4", "kind": "post-hoc", "approval": APPROVAL, "why": WHY, "reading_rule": RULE,
            "cells": {a: {"lr": lr, "base": f"{b[0]}@{b[1]}",
                          "change": ("--window 0.05,0.4 -> 0.05,1.0 (prior on from step 916 through step 18309)" if a == "all6full" else
                                     "--prior_domains numbers,dim,cities,countries,torus,box5 -> numbers,dim,cities,countries "
                                     "(--domains unchanged: same data and evals; lam_rule each: lambda 1 per remaining prior)")}
                      for a, (lr, b) in CELLS.items()},
            "seeds": list(SEEDS), "domains_arg": MJ2.domains_arg(doms), "code_commit": commit, "out_dir": "runs/pt6v2/out_v4",
            "device_note": DEV_NOTE, "order": [f"s{s} {a}" for s, a in ORDER],
            "cells_json_sha12": CELLS_SHA, "make_jobs_pt6v2_sha12": MJ2_SHA, "driver": "runs/pt6v2/posthoc_v4_jobs.py",
            "runs": {j["name"]: {"seed": j["seed"], "arm": j["arm"], "lr": j["lr"], "base": j["base"], "device": j["device"],
                                 "alloc_device": j["alloc_device"], "device_change": j["device"] != j["alloc_device"],
                                 "slot": MJ2.slot_name(j["slot"]), "job_sha12": hashlib.sha256(j["text"].encode()).hexdigest()[:12],
                                 "base_job_sha12": j["base_job_sha12"], "job_id": (ids or {}).get(j["name"])} for j in jobs},
            "readout": "not read by readout_pt6v2.py (out_v4/; arm names outside its regex); descriptive read by the main session"}


def main():
    doms, commit, jobs = build()
    holds = [h for h in [R / "HOLD", R / "HOLD_b200"] if h.exists()]
    for j in jobs:
        st = ("SUBMITTED" if j["flag"].exists() else
              "on disk (same)" if j["job"].exists() and j["job"].read_text() == j["text"] else
              "on disk (DIFFERS)" if j["job"].exists() else "not written")
        print(f"{j['name']:28s} {j['device']:5s} slot {j['slot']}  base {j['base']:9s} (ALLOC {j['alloc_device']})  "
              f"{j['changed_lines']} lines differ from the base, intent fields {j['changed_intent']}  {st}")
    print(f"check: {len(jobs)} job texts parse (bash -n); each = its base job except the header description, the provenance line and "
          f"the expected substitutions; CELLS.json {CELLS_SHA}, make_jobs_pt6v2.py {MJ2_SHA}, POSTHOC_V3.json {V3_SHA}, ALLOC.json "
          f"{ALLOC_SHA} unchanged; HOLD files: {[h.name for h in holds] or 'none'}")
    if "--print" in sys.argv:
        for j in jobs:
            print(f"\n{j['name']} [{j['device']}]: {j['args']}")
            print("".join(ln + "\n" for ln in j["text"].splitlines() if ln.startswith("#SBATCH")), end="")
            print("intent:", {k: j["intent"][k] for k in ("events", "on_frac", "active", "lam_domains")},
                  {k: j["intent"]["args"][k] for k in ("window", "prior_domains", "tag", "lam_rule", "lam", "domains")})
    if "--write" in sys.argv:
        OV4.mkdir(exist_ok=True)
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"], f"{j['job']} was submitted and differs from this rendering"
                continue
            j["job"].write_text(j["text"])
        old = json.loads(POSTHOC.read_text()) if POSTHOC.exists() else None
        new = manifest(doms, commit, jobs, {n: r["job_id"] for n, r in (old or {}).get("runs", {}).items()})
        if old != new:
            assert old is None or all(r["job_id"] is None for r in old["runs"].values()), "POSTHOC_V4.json holds job ids: not rewritten"
            POSTHOC.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n")
        print(f"written: {len(jobs)} job files in {OV4}; {POSTHOC}")
    if "--test-only" in sys.argv:
        i = sys.argv.index("--test-only")
        n = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit() else len(jobs)
        for j in jobs[:n]:
            o = subprocess.run(SBATCH + ["--test-only", str(j["job"])], capture_output=True, text=True)
            print(j["name"], (o.stdout + o.stderr).strip())
    if "--submit" in sys.argv:
        assert not holds, f"HOLD files exist: {holds}"
        assert POSTHOC.exists(), "run --write first"
        stale = [j["name"] for j in jobs if not j["flag"].exists() and (not j["job"].exists() or j["job"].read_text() != j["text"])]
        assert not stale, f"job files missing or differing from this rendering (run --write first): {stale}"
        ids = {}
        for j in jobs:
            if j["flag"].exists():
                print(f"{j['name']}: refused (SUBMITTED_{j['name']} exists: a name is submitted once)")
                continue
            o = subprocess.run(SBATCH + [str(j["job"])], capture_output=True, text=True)
            print(j["name"], (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                ids[j["name"]] = o.stdout.split()[-1]
        man = json.loads(POSTHOC.read_text())
        for n, i_ in ids.items():
            man["runs"][n]["job_id"] = i_
        POSTHOC.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
        print(f"submitted {len(ids)} of {len(jobs)}; job ids added to {POSTHOC}")


if __name__ == "__main__":
    main()
