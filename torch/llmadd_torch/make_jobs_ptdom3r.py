"""ptdom3 replication stage (runs/pt6v2/AUDIT.md, "Composition pick rule", bullet "toy3 replication"): seeds 1, 2 of the toy3
prior picks and of the shared none cells, plus none@3.2e-2 at seeds 0-2. B200 (approved for ptdom3).

make_jobs_ptdom3.py is frozen (make_jobs_pt6v2.py pins its sha 1046e44f3409), so this stage lives here and calls its
render3 read-only: the screen's job body exactly; only --seed, --lr, the run name / out dir (ptdom3-T3-<arm>-lr<lr>-s<seed>,
runs/ptdom3/out/<name>) and these differ:
  - Slurm name llmadd_ptdom3r-b200-slot<0..2> (3 singleton chains = the B200 per-user GPU cap), the header's stage label
    "rep", and the kill switch: runs/ptdom3/HOLD (every ptdom3 job) or runs/ptdom3/HOLD_rep (this stage only);
  - the CPU reference of the cell's seed: seed 0 = runs/ptdom3/ptdom3_refs_T3.json (the screen's), seeds 1 / 2 =
    runs/ptdom3/ptdom3_refs_T3_rep.json (runs/ptdom3/tests/refs_rep_ptdom3.py; written once, chmod 444).
Cells: none @ 2e-3 / 4e-3 / 8e-3 / 1.6e-2 / 3.2e-2, only-<d> @ 1e-3 / 2e-3 / 4e-3, seeds 0-2; a seed-0 cell of the screen grid
is refused (it exists). A cell already in runs/ptdom3/submitted.tsv (or with an out dir) is never rendered over or re-submitted.
Slots: round-robin over the cells not yet submitted, offset by the number of replication rows already in submitted.tsv.
    python make_jobs_ptdom3r.py --stage rep --cells none@1.6e-2:1,2 none@3.2e-2:0,1,2 only-torus@1e-3:1,2 --check [--print NAME]
    python make_jobs_ptdom3r.py --stage rep --cells ... --test-only     bash -n + sbatch --test-only of every rendered job
    python make_jobs_ptdom3r.py --stage rep --cells ... --submit        refuses while runs/ptdom3/HOLD or HOLD_rep exists
"""
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
FROZEN = {"make_jobs_ptdom3.py": "1046e44f3409", "make_jobs_ptdom.py": "6061b299129c"}
for f_, v_ in FROZEN.items():
    assert sha(B / f_) == v_, f"{f_} changed: {sha(B / f_)} != {v_}"
sys.path.insert(0, str(B))
import make_jobs_ptdom3 as M3    # noqa: E402  (read only: render3, refs3, load_P, names, GRID)
assert Path(M3.__file__).resolve().parent == B.resolve()
MJ, R, _edit = M3.MJ, M3.R, M3._edit

STAGE = "rep"
REP_SEEDS = (0, 1, 2)
REP_GRID = {"none": M3.GRID["none"] + ("3.2e-2",), "prior": M3.GRID["prior"]}
N_REP = 3
REFS_REP = R / "ptdom3_refs_T3_rep.json"
SCREEN = {(a, lr) for a in M3.ARMS for lr in (M3.GRID["none"] if a == "none" else M3.GRID["prior"])}
# this session runs inside a Slurm job: sbatch must not inherit its allocation
SBATCH = ["env", "-u", "SLURM_JOB_ID", "-u", "SLURM_JOBID", "-u", "SLURM_STEP_ID", "-u", "SLURM_CPUS_PER_TASK", "-u",
          "SLURM_MEM_PER_NODE", "-u", "SLURM_NTASKS", "sbatch"]
for lr_ in REP_GRID["none"] + REP_GRID["prior"]:
    assert lr_ in MJ.LADDER, lr_


def rep_slot_name(slot, canary=False):
    assert not canary
    return f"llmadd_{M3.PREFIX}r-b200-slot{slot}"


def ref_of(seed):
    if seed == 0:
        ref, _ = M3.refs3()
        assert ref, M3.REFS
        return ref
    assert REFS_REP.exists(), f"no CPU reference {REFS_REP}: run runs/ptdom3/tests/run_refs_rep_ptdom3.sh first"
    j = json.loads(REFS_REP.read_text())
    assert (j.get("design"), j.get("domains"), j.get("commit")) == (M3.PREFIX, M3.domains_arg(), M3.COMMIT), REFS_REP
    assert str(seed) in j["seeds"], f"{REFS_REP} has no seed {seed}"
    return j["seeds"][str(seed)]


def parse_cells(toks):
    """['none@1.6e-2:1,2', ...] -> [(arm, lr, seed)] in the given order (duplicates dropped)."""
    out = []
    for t in toks:
        arm, rest = t.split("@")
        lr, ss = rest.split(":")
        assert arm in M3.ARMS, f"{t}: arm {arm} not in {M3.ARMS}"
        assert lr in REP_GRID["none" if arm == "none" else "prior"], f"{t}: lr {lr} not in {REP_GRID}"
        for s in (int(x) for x in ss.split(",")):
            assert s in REP_SEEDS, f"{t}: seed {s} not in {REP_SEEDS}"
            assert not (s == 0 and (arm, lr) in SCREEN), f"{t}: seed 0 at {arm}@{lr} is a screen run"
            if (arm, lr, s) not in out:
                out.append((arm, lr, s))
    assert out, "no cells"
    return out


@contextmanager
def rep_globals(seed):
    """M3's module globals that render3 / M3.patched read at call time: the seed set, the lr grid, the slot chains."""
    keep = {k: getattr(M3, k) for k in ("SEED", "GRID", "N_SLOTS", "slot_name")}
    try:
        M3.SEED, M3.GRID, M3.N_SLOTS, M3.slot_name = seed, REP_GRID, N_REP, rep_slot_name
        yield
    finally:
        for k, v in keep.items():
            setattr(M3, k, v)


def render_rep(P, arm, lr, seed, commit, slot):
    assert 0 <= slot < N_REP
    with rep_globals(seed):
        name, tag, text, args, I = M3.render3(P, arm, lr, seed, ref_of(seed), commit, 0, slot)
    assert name == M3.name3(arm, lr, seed) and f" --seed {seed} " in f" {args} " and f" --lr {lr} " in f" {args} ", (name, args)
    assert I["args"]["seed"] == seed and I["args"]["lr"] == float(lr)
    text = _edit(text, "# ptdom3 (exploratory screen; ptdom's stage-1 job body), batch T3 S: ",
                 "# ptdom3 replication (runs/pt6v2/AUDIT.md 'toy3 replication'; the screen's job body), batch T3 rep: ")
    text = _edit(text, "# spec runs/ptdom3/AUDIT.md (exploratory tier: no decision rule); rendered by llmadd_torch/make_jobs_ptdom3.py through make_jobs_ptdom.render\n",
                 "# spec runs/ptdom3/AUDIT.md (exploratory tier) + runs/pt6v2/AUDIT.md (toy3 replication); rendered by llmadd_torch/make_jobs_ptdom3r.py\n")
    text = _edit(text, "# runs/ptdom3/HOLD_S (the screen S or the canary only) exists\n"
                       'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_S" ]; then echo "HOLD present: exiting"; exit 0; fi\n',
                 f"# runs/ptdom3/HOLD_{STAGE} (the replication stage only) exists\n"
                 f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{STAGE}" ]; then echo "HOLD present: exiting"; exit 0; fi\n')
    assert f"#SBATCH --job-name={rep_slot_name(slot)}\n" in text and "HOLD_S" not in text and "batch T3 S" not in text
    assert f"SEED={seed};" in text and f"/out/{name};" in text
    return name, tag, text, args, I


def submitted():
    rec = R / "submitted.tsv"
    return {l.split("\t")[0]: l.split("\t")[1] for l in rec.read_text().splitlines() if l.strip()} if rec.exists() else {}


def screen_names():
    return {M3.name3(a, lr, M3.SEED) for a, lr in SCREEN} | {M3.name3(*M3.CANARY, canary=True)}


def build(P, cells):
    """-> (jobs [(slot, name, tag, text, args, I)] of the cells not yet submitted, {name: jid} of those already submitted)."""
    commit = M3.snapshot()
    done = submitted()
    scr = screen_names()
    n_prev = sum(1 for n in done if n not in scr)
    names = [M3.name3(a, lr, s) for a, lr, s in cells]
    others = {p.name for p in B.glob("runs/*/out/*") if p.parent.parent != R} | {p.name for p in B.glob("runs/*/pilot/*") if p.parent.parent != R}
    assert not set(names) & (others | scr), sorted(set(names) & (others | scr))
    jobs, old = [], {}
    for a, lr, s in cells:
        n = M3.name3(a, lr, s)
        if n in done:
            old[n] = done[n]
            continue
        slot = (n_prev + len(jobs)) % N_REP
        jobs.append((slot, *render_rep(P, a, lr, s, commit, slot)))
    return jobs, old


def write_jobs(jobs):
    (R / "jobs").mkdir(exist_ok=True)
    for slot, name, tag, text, args, I in jobs:
        f = M3.jobfile(name)
        if f.exists() and f.read_text() == text:
            continue
        if f.exists():
            f.chmod(0o644)
        f.write_text(text); f.chmod(0o444)
        (R / "jobs" / f"{name}.intent.json").write_text(json.dumps(I, indent=1))


def main():
    a = sys.argv[1:]
    if "--stage" not in a or "--cells" not in a:
        print(__doc__); return
    assert a[a.index("--stage") + 1] == STAGE, "only --stage rep"
    i = a.index("--cells") + 1
    toks = []
    while i < len(a) and not a[i].startswith("--"):
        toks.append(a[i]); i += 1
    cells = parse_cells(toks)
    P = M3.load_P()
    if "--submit" in a:
        assert not (R / "HOLD").exists() and not (R / f"HOLD_{STAGE}").exists(), "HOLD present: not submitting"
    jobs, old = build(P, cells)
    for n, jid in old.items():
        print(f"already submitted (not rendered, not re-submitted): {n} {jid}")
    if "--check" in a or "--print" in a or "--test-only" in a:
        write_jobs(jobs)
        for slot, name, *_ in jobs:
            print(f"{name}\tslot {slot}\t{M3.jobfile(name)}\t{sha(M3.jobfile(name))}")
        if "--print" in a:
            print(M3.jobfile(a[a.index("--print") + 1]).read_text())
        if "--test-only" in a:
            bad = 0
            for slot, name, *_ in jobs:
                f = M3.jobfile(name)
                rn = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
                rt = subprocess.run(SBATCH + ["--test-only", str(f)], capture_output=True, text=True)
                ok = rn.returncode == 0 and rt.returncode == 0 and "to start at" in (rt.stderr + rt.stdout)
                bad += not ok
                print(f"{'OK ' if ok else 'BAD'} {name}: bash -n rc {rn.returncode}; test-only: {(rt.stderr + rt.stdout).strip()[:160]}")
            print("TEST-ONLY", "PASS" if not bad else f"FAIL ({bad})", f"({len(jobs)} jobs)")
        return
    if "--submit" in a:
        rec = R / "submitted.tsv"
        for slot, name, tag, text, args, I in jobs:
            f = M3.jobfile(name)
            assert f.exists() and f.read_text() == text, f"{f}: render --check first (the file differs from the generator)"
            if (R / "out" / name).exists():
                print("SKIP (out dir exists):", name); continue
            o = subprocess.run(SBATCH + ["--parsable", str(f)], capture_output=True, text=True, check=True).stdout.strip()
            jid = o.split(";")[0]
            assert jid.isdigit(), o
            with open(rec, "a") as fh:
                fh.write(f"{name}\t{jid}\t{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{sha(f)}\n")
            print(f"SUBMITTED {name} {jid} slot {slot}")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
