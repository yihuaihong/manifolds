"""T2: the amended generator renders every submitted job file of runs/pttpp byte for byte (only decisions() / runs_of()
semantics and the cost helpers changed); prints the planned GPU-h. Imports the generator from argv[1]."""
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gen", sys.argv[1]); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
n = bad = 0
for gid in T.all_groups():
    names = [x[0] for x in T.render(gid)]
    assert len(names) == len(set(names)), (gid, "duplicate run names")
    for name, txt, F in T.render(gid):
        f = T.R / f"job_{name}.sbatch"
        if (T.R / f"SUBMITTED_{name}").exists():
            n += 1
            if f.read_text() != txt:
                bad += 1; print("DIFFERS", name)
subs = {p.name[len("SUBMITTED_"):] for p in T.R.glob("SUBMITTED_*")}
rendered = {x[0] for g in T.all_groups() for x in T.render(g)}
print(f"{n} submitted job files compared, {bad} differ; submitted but not rendered: {sorted(subs - rendered)}")
print("planned GPU-h", {k: round(v, 1) for k, v in T.planned_hours().items()}, "max", T.COST_MAX)
print("RENDER-SAME", "PASS" if not bad and not (subs - rendered) else "FAIL")
