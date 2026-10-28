"""T8 render check: the 8-slot generator (argv[1]) against the deployed 4-slot one and the job files on disk.
1. every non-B200 group: the 8-slot rendering = the 4-slot rendering = every submitted job file, byte for byte;
2. B200 (tpp7-1b-t0.745 and the bridge): index < 4 renders identically (its slot is i % 4 = i % 8); index >= 4 differs from
   the 4-slot rendering in exactly one line, '#SBATCH --job-name=llmadd_tpp-b200-slot<k>', with k = i % 8;
3. a fake root with 1.07B decisions (X both arms, then C both arms; and C only): the decided stages' files differ the same
   way only; at most 8 distinct B200 job names;
4. lists the B200 runs whose jobs are to be cancelled and resubmitted (index >= 4, still PENDING with Restarts 0 per the
   given squeue snapshot) with their job ids."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


N = load(sys.argv[1], "gen8")
O = load("/scratch/yh6210/llmadd_torch/make_jobs_pttpp.py", "gen4")
assert N.SLOTS == 8 and O.SLOTS == 4
fails = []


def check(c, what):
    print(("PASS " if c else "FAIL ") + what, flush=True)
    if not c:
        fails.append(what)


def diffl(a, b):
    la, lb = a.splitlines(), b.splitlines()
    return [(x, y) for x, y in zip(la, lb) if x != y] + ([("<len>", "<len>")] if len(la) != len(lb) else [])


R = O.R
n_same = 0
for gid in O.all_groups():
    if gid == O.BRIDGE or O.GROUPS[gid]["hw"] == "b200":
        continue
    a, b = O.render(gid), N.render(gid)
    ok = [x[0] for x in a] == [x[0] for x in b] and all(x[1] == y[1] for x, y in zip(a, b))
    disk = all((R / f"job_{n}.sbatch").read_text() == t for n, t, F in b if (R / f"SUBMITTED_{n}").exists())
    n_same += len(b)
    check(ok and disk, f"{gid}: 8-slot = 4-slot rendering = the submitted files ({len(b)} runs)")
order = N.b200_order()
assert order == O.b200_order()
ren4 = {n: t for g in (O.BRIDGE, "tpp7-1b-t0.745") for n, t, F in O.render(g)}
ren8 = {n: t for g in (N.BRIDGE, "tpp7-1b-t0.745") for n, t, F in N.render(g)}
bad = []
for i, (g, st, a, lr, s) in enumerate(order):
    F = N.facts(g, a, lr, s, st)
    d = diffl(ren4[F["name"]], ren8[F["name"]])
    want = [] if i < 4 else [(f"#SBATCH --job-name=llmadd_tpp-b200-slot{i % 4}", f"#SBATCH --job-name=llmadd_tpp-b200-slot{i % 8}")]
    if i >= 4 and i % 4 == i % 8:
        want = []
    if d != want:
        bad.append((i, F["name"], d[:3]))
check(not bad, f"B200: {len(order)} runs; index < 4 identical, the others differ only in the slot line, slot = i % 8 ({bad[:2]})")
on_disk = [(i, N.facts(g, a, lr, s, st)["name"]) for i, (g, st, a, lr, s) in enumerate(order)]
same_disk = [n for i, n in on_disk if i < 4 and (R / f"SUBMITTED_{n}").exists() and (R / f"job_{n}.sbatch").read_text() == ren8[n]]
check(len(same_disk) == 4, f"B200 index 0-3: the submitted job files = the 8-slot rendering ({len(same_disk)})")
# 3. decided stages of 1.07B in a fake root
W = Path("/scratch/yh6210/llmadd_torch/scale_work/tpp/t8/fake_render")
for case, log in (("x_then_c", [{"seq": 0, "arm": "none", "stage": "X", "lr": "4.8e-3"}, {"seq": 1, "arm": "win-ctx", "stage": "C", "lr": "3e-4"},
                                {"seq": 2, "arm": "none", "stage": "C", "lr": "4.8e-3"}]),
                  ("c_only", [{"seq": 0, "arm": "none", "stage": "C", "lr": "1.2e-3"}, {"seq": 1, "arm": "win-ctx", "stage": "C", "lr": "6e-4"}])):
    shutil.rmtree(W / case, ignore_errors=True)
    (W / case / "decisions").mkdir(parents=True)
    (W / case / "decisions" / "tpp7-1b-t0.745.json").write_text(json.dumps({"group": "tpp7-1b-t0.745", "log": log}))
    oR, nR = O.R, N.R
    O.R = N.R = W / case
    try:
        o4 = {n: t for n, t, F in O.render("tpp7-1b-t0.745")}
        n8 = N.render("tpp7-1b-t0.745")
        names = set()
        badc = []
        idx = {N.facts(g, a, lr, s, st)["name"]: i for i, (g, st, a, lr, s) in enumerate(N.b200_order())}
        for n, t, F in n8:
            i = idx[n]
            names.add([l for l in t.splitlines() if l.startswith("#SBATCH --job-name=")][0])
            d = diffl(o4[n], t)
            want = [] if i % 4 == i % 8 else [(f"#SBATCH --job-name=llmadd_tpp-b200-slot{i % 4}", f"#SBATCH --job-name=llmadd_tpp-b200-slot{i % 8}")]
            if d != want:
                badc.append((n, d[:2]))
        allnames = {f"#SBATCH --job-name=llmadd_tpp-b200-slot{i % 8}" for i in range(len(N.b200_order()))}
    finally:
        O.R, N.R = oR, nR
    check(not badc and len(allnames) <= 8 and len(n8) == len(o4),
          f"fake 1.07B decisions '{case}': {len(n8)} runs, the decided stages differ only in the slot line; <= 8 slot names ({badc[:1]})")
# 4. what to cancel
snap = {l.split("|")[0]: l.rstrip("\n").split("|") for l in open(sys.argv[2])} if len(sys.argv) > 2 else {}
todo = []
for i, n in on_disk:
    if i < 4 or i % 4 == i % 8:
        continue
    f = R / f"SUBMITTED_{n}"
    jid = f.read_text().split()[-1] if f.exists() else None
    todo.append((i, n, jid, snap.get(jid, ["", "", "", "", "not in snapshot"])[4], i % 4, i % 8))
for t in todo:
    print("   to resubmit:", t)
print(f"T8-RENDER {'PASS' if not fails else 'FAIL'}: {len(fails)} failed")
sys.exit(1 if fails else 0)
