"""A4 byte-identity check: the pre-A4 generator (backup) and the A4 generator render every L1 / L1x / C / smoke body identically;
the on-disk L1 / L1x job files (submitted) equal the A4 generator's render."""
import importlib.util, json, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
sys.argv = [sys.argv[0]]
B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
import make_jobs_pt6comb as NEW
spec = importlib.util.spec_from_file_location("M6old", "/scratch/yh6210/llmadd_torch/runs/pt6comb/tests/pre_a4_20261113/make_jobs_pt6comb.py")
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)
P = NEW.load_P()
RF = NEW.refs6("B")
n, bad = 0, []
def cmp(part, cl, sm=False):
    global n
    for i, (a, lr, s) in enumerate(cl):
        args = (P, part, "B", a, lr, s, RF[s], NEW.COMMIT, 0, 0 if sm else i % 8) + ((None, True) if sm else ())
        t1, t0 = NEW.render6(*args)[2], OLD.render6(*args)[2]
        n += 1
        if t1 != t0:
            bad.append((part, a, lr, s))
cmp("L1", NEW.cells("L1"))
cmp("smoke", NEW.cells("smoke"), True)
with tempfile.TemporaryDirectory() as tmp:
    pk2 = Path(tmp) / "x.json"
    pk2.write_text(json.dumps({"design": "pt6comb", "status": "decided", "stage_written": "L1x", "comp": "B", "x": None,
                               "E": [["all6", "8e-3"], ["none", "1.6e-2"], ["none", "1e-3"]]}))
    pk2.chmod(0o444)
    assert NEW.cells("L1x", "B", str(pk2)) == OLD.cells("L1x", "B", str(pk2))
    cmp("L1x", NEW.cells("L1x", "B", str(pk2)))
    for P_, N_ in (("2e-3", "4e-3"), ("4e-3", "8e-3"), ("2e-3", "2e-3"), ("4e-3", "2e-3"), ("1e-3", "1.6e-2"), ("1e-3", "1e-3")):
        pk = Path(tmp) / f"c_{P_}_{N_}.json"
        pk.write_text(json.dumps({"design": "pt6comb", "status": "decided", "stage_written": "C", "comp": "B", "x": None, "after": "L1x",
                                  "C": [list(c) for c in NEW.c_cells(P_, N_)], "pick_all6": P_, "pick_none": N_}))
        pk.chmod(0o444)
        assert NEW.cells("C", "B", str(pk)) == OLD.cells("C", "B", str(pk))
        cmp("C", NEW.cells("C", "B", str(pk)))
# on disk: the submitted L1 / L1x job files = the A4 generator's render
R = NEW.R
dn, dbad = 0, []
px = R / "picks_after_L1.json"
for part, cl in (("L1", NEW.cells("L1")), ("L1x", NEW.cells("L1x", "B", str(px)))):
    for i, (a, lr, s) in enumerate(cl):
        name, tag, text, args, I = NEW.render6(P, part, "B", a, lr, s, RF[s], NEW.COMMIT, 0, i % 8)
        f = R / f"job_{name}.sbatch"
        if f.exists() and (R / f"SUBMITTED_{name}").exists():
            dn += 1
            if f.read_text() != text:
                dbad.append(name)
print(f"old vs A4 generator: {n} bodies (L1 24, smoke 2, L1x 12, C 6 (P, N) cases), byte-different {len(bad)}: {bad[:4]}")
print(f"submitted L1 / L1x job files on disk vs the A4 render: {dn} compared, byte-different {len(dbad)}: {dbad[:4]}")
print("OLDNEW " + ("PASS" if not bad and not dbad and dn == 32 else "FAIL"))
