#!/usr/bin/env python3
"""pt6v2 AUDIT.md Amendment V3 (post-hoc, user-approved 2026-11-19): none@6.4e-2 for seeds 16-23 (8 runs).

Why: in the registered read-out dim and box5 pick N_m = 3.2e-2, the top of the registered none grid (2e-3 .. 3.2e-2), and box5's
none is still rising there (none@1.6e-2 -> 3.2e-2: 31.2 -> 42.4 at the 7-seed partial read-out of 2026-11-19 19:30). One more none lr
above the grid tells whether g = all6@1e-3 - none@N_m shrinks once the baseline gets its best lr.

The narrowest change: CELLS.json (the registered cell set, sha ea9a9b01f081), make_jobs_pt6v2.py (bb15d248afd8) and the 105 jobs
already submitted are not touched. Every job text here is make_jobs_pt6v2.render_v2 of (none, 6.4e-2, seed) on the seed's ALLOC.json
device, with the B200 slot that make_jobs_pt6v2.py --extra_none 6.4e-2 (Amendment V2's rendering) gives that cell: the text equals
that maker's rendering byte for byte. The B200 jobs reuse the existing singleton slot names llmadd_pt6v2-b200-slot<k> (and the user's
3-GPU B200 cap). The record of the extension is runs/pt6v2/POSTHOC_V3.json (read by readout_pt6v2.py --posthoc_none 6.4e-2).

Usage:  posthoc_v3_jobs.py [--check] [--print] [--write] [--test-only [N]] [--submit]
  (no flag / --check)  render the 8 jobs, bash -n each, every guard; print a table; nothing is written
  --print              the args and the #SBATCH lines of each job
  --write              write runs/pt6v2/job_<name>.sbatch (never over a submitted one) and POSTHOC_V3.json (without job ids)
  --test-only [N]      sbatch --test-only on the first N on-disk job files (all 8 without N)
  --submit             sbatch each on-disk job file (it must equal this rendering), write SUBMITTED_<name> with sbatch's stdout and
                       add the job ids to POSTHOC_V3.json; a name with a SUBMITTED_ flag is never submitted again
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pt6v2"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]   # noqa: E731
MJ2_SHA, CELLS_SHA = "bb15d248afd8", "ea9a9b01f081"
assert sha(B / "make_jobs_pt6v2.py") == MJ2_SHA, "make_jobs_pt6v2.py changed (frozen after the V2 build)"
assert sha(R / "CELLS.json") == CELLS_SHA, "CELLS.json changed: Amendment V3 never rewrites the registered cell set"
sys.path.insert(0, str(B))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_pt6v2 as MJ2   # noqa: E402  (read only)
sys.argv = _argv
assert Path(MJ2.__file__).resolve() == (B / "make_jobs_pt6v2.py").resolve()

ARM, LR = "none", "6.4e-2"
assert LR in MJ2.EXTRA_NONE_OK and LR in MJ2.LADDER
SEEDS = MJ2.SEEDS
POSTHOC = R / "POSTHOC_V3.json"
ENV_U = ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_STEP_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_NTASKS")
SBATCH = ["env"] + [x for v in ENV_U for x in ("-u", v)] + ["sbatch"]
APPROVAL = ("user-approved 2026-11-19 ~20:00 EST (relayed by the main session; B200 approved); post-hoc: Amendment V2's own condition "
            "was not triggered")
WHY = ("registered read-out (tests/readout_partial_20261119T1930.txt, 7 seeds): dim and box5 have N_m = 3.2e-2 = the grid top; box5's "
       "none still rising (1.6e-2 -> 3.2e-2: 31.2 -> 42.4)")
RULE = ("the registered verdict stays the registered one; the extended verdict (grid + 6.4e-2) is reported beside it; if a manifold's "
        "extended g changes sign or loses HELPS, both are reported and the paper wording follows the extended one")


def v2_slots(doms, alloc):
    """The B200 slot make_jobs_pt6v2.main gives each cell with --extra_none 6.4e-2 (its i % N_SLOTS counter over seeds x cells)."""
    i, out = 0, {}
    for s in SEEDS:
        for a, lr in MJ2.cells(doms, LR):
            dev = alloc[s]
            slot = (i % MJ2.N_SLOTS) if dev == "b200" else None
            i += dev == "b200"
            if (a, lr) == (ARM, LR):
                out[s] = slot
    return out


def build():
    doms = MJ2.decided_comp()
    assert doms is not None, "COMPOSITION.json missing"
    cj = json.loads((R / "CELLS.json").read_text())
    assert cj["extra_none"] is None and LR not in cj["none_lrs"] and [ARM, LR] not in cj["cells"], "CELLS.json already holds none@6.4e-2"
    assert cj["domains_arg"] == MJ2.domains_arg(doms)
    alloc = {int(k): v for k, v in json.loads((R / "ALLOC.json").read_text()).items()}
    assert sorted(alloc) == list(SEEDS)
    commit = MJ2.snapshot()
    P = MJ2.load_P()
    RF, _ = MJ2.refs_v2(doms)
    others = {p.name for p in B.glob("runs/*/out/*") if p.parent.parent != R}
    slots = v2_slots(doms, alloc)
    jobs = []
    for s in SEEDS:
        assert s in RF, f"no CPU reference for seed {s}"
        dev = alloc[s]
        name, tag, text, args, I = MJ2.render_v2(P, doms, ARM, LR, s, RF[s], commit, dev, slots[s])
        assert name == MJ2.name_v2(ARM, LR, s) == f"pt6v2-none-lr6.4e-2-s{s}" and name not in others
        assert not (R / "out" / name).exists(), f"{R / 'out' / name} exists"
        if dev == "b200":
            assert f"#SBATCH --job-name={MJ2.slot_name(slots[s])}\n#SBATCH --dependency=singleton\n" in text, name
        assert I["device"] == dev
        assert subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0, f"{name}: bash -n"
        jobs.append({"name": name, "seed": s, "device": dev, "slot": slots[s], "text": text, "args": args,
                     "job": R / f"job_{name}.sbatch", "flag": R / f"SUBMITTED_{name}"})
    return doms, commit, jobs


def manifest(doms, commit, jobs, ids=None):
    return {"amendment": "V3", "kind": "post-hoc", "approval": APPROVAL, "why": WHY, "reading_rule": RULE,
            "arm": ARM, "lr": LR, "seeds": list(SEEDS), "domains_arg": MJ2.domains_arg(doms), "code_commit": commit,
            "cells_json_sha12": CELLS_SHA, "make_jobs_pt6v2_sha12": MJ2_SHA, "driver": "runs/pt6v2/posthoc_v3_jobs.py",
            "runs": {j["name"]: {"seed": j["seed"], "device": j["device"],
                                 "slot": (MJ2.slot_name(j["slot"]) if j["slot"] is not None else None),
                                 "job_sha12": hashlib.sha256(j["text"].encode()).hexdigest()[:12],
                                 "job_id": (ids or {}).get(j["name"])} for j in jobs},
            "readout": "readout_pt6v2.py --posthoc_none 6.4e-2 (flag off: the registered read-out, these runs left out)"}


def main():
    doms, commit, jobs = build()
    holds = [h for h in [R / "HOLD", R / "HOLD_b200", R / "HOLD_a100", R / "HOLD_l40s"] if h.exists()]
    for j in jobs:
        st = ("SUBMITTED" if j["flag"].exists() else
              "on disk (same)" if j["job"].exists() and j["job"].read_text() == j["text"] else
              "on disk (DIFFERS)" if j["job"].exists() else "not written")
        print(f"{j['name']:28s} {j['device']:5s} slot {j['slot'] if j['slot'] is not None else '-'}  {st}")
    print(f"check: {len(jobs)} job texts parse (bash -n); CELLS.json {CELLS_SHA} and make_jobs_pt6v2.py {MJ2_SHA} unchanged; "
          f"HOLD files: {[h.name for h in holds] or 'none'}")
    if "--print" in sys.argv:
        for j in jobs:
            print(f"\n{j['name']} [{j['device']}]: {j['args']}")
            print("".join(ln + "\n" for ln in j["text"].splitlines() if ln.startswith("#SBATCH")), end="")
    if "--write" in sys.argv:
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"], f"{j['job']} was submitted and differs from this rendering"
                continue
            j["job"].write_text(j["text"])
        old = json.loads(POSTHOC.read_text()) if POSTHOC.exists() else None
        new = manifest(doms, commit, jobs, {n: r["job_id"] for n, r in (old or {}).get("runs", {}).items()})
        if old != new:
            assert old is None or all(r["job_id"] is None for r in old["runs"].values()), "POSTHOC_V3.json holds job ids: not rewritten"
            POSTHOC.write_text(json.dumps(new, indent=1) + "\n")
        print(f"written: {len(jobs)} job files; {POSTHOC}")
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
        POSTHOC.write_text(json.dumps(man, indent=1) + "\n")
        print(f"submitted {len(ids)} of {len(jobs)}; job ids added to {POSTHOC}")


if __name__ == "__main__":
    main()
