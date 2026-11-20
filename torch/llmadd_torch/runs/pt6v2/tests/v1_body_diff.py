"""Amendment V1 check: render the pt6v2 jobs with make_jobs_pt6v2.py before V1 (961ef81fe230; its copy tests/make_jobs_pt6v2.preV1_961ef81fe230 = argv[1]) and after
V1 (the live file), placeholder composition with its CPU references, the same --alloc and canary device; the 97 old names must
all be re-rendered and every common job text identical except the B200 slot job-name and its 'singleton slot N' comment (the slot cycle
shifts with 13 cells per seed); the new names must be exactly the 8 none@4e-3 runs. Output: stdout (tests/v1_body_diff_<ts>.log)."""
import difflib
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
old_path, ALLOC, CDEV = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_pt6v2 as NEW  # noqa: E402
import importlib.machinery  # noqa: E402
sp = importlib.util.spec_from_file_location("make_jobs_pt6v2_old", old_path, loader=importlib.machinery.SourceFileLoader("make_jobs_pt6v2_old", str(old_path)))
OLD = importlib.util.module_from_spec(sp)
sp.loader.exec_module(OLD)
sys.argv = _argv
assert NEW.sha(old_path) == "961ef81fe230" and NEW.sha(NEW.__file__) != "961ef81fe230"


def render_all(V):
    doms = V.PLACEHOLDER
    alloc = V.parse_alloc(ALLOC.split())
    P, commit = V.load_P(), V.snapshot()
    RF, RC = V.refs_v2(doms)
    assert RF and RC
    a, lr, s = V.CANARY
    jobs = [V.render_v2(P, doms, a, lr, s, RC, commit, CDEV, None, 0, True)]
    i = 0
    for s in V.SEEDS:
        for a, lr in V.cells(doms):
            dev = alloc[s]
            slot = (i % V.N_SLOTS) if dev == "b200" else None
            i += dev == "b200"
            jobs.append(V.render_v2(P, doms, a, lr, s, RF[s], commit, dev, slot))
    return {j[0]: (j[2], j[3]) for j in jobs}


jo, jn = render_all(OLD), render_all(NEW)
print(f"old {len(jo)} jobs, new {len(jn)} jobs; alloc {ALLOC}; canary on {CDEV}")
added = sorted(set(jn) - set(jo))
print("new names:", added)
ok_added = len(added) == 8 and all(re.fullmatch(r"pt6v2-none-lr4e-3-s(1[6-9]|2[0-3])", n) for n in added) and not (set(jo) - set(jn))
cnt, unreg, same = Counter(), Counter(), 0
for n in sorted(set(jo) & set(jn)):
    (to, ao), (tn, an) = jo[n], jn[n]
    if to == tn:
        same += 1
        continue
    if ao != an:
        unreg[f"{n}: trainer args differ"] += 1
    for ln in difflib.unified_diff(to.splitlines(), tn.splitlines(), lineterm="", n=0):
        if ln.startswith(("---", "+++", "@@")):
            continue
        if re.fullmatch(r"[-+]#SBATCH --job-name=llmadd_pt6v2-b200-slot[0-7]", ln):
            cnt["b200 slot job-name"] += 1
        elif re.match(r"[-+]# \(change the generator, not this file\)\. .*; singleton slot [0-7] \(batch ", ln):
            cnt["b200 slot comment"] += 1
        else:
            unreg[ln[:200]] += 1
print(f"common names {len(set(jo) & set(jn))}: identical {same}; residual diff lines {dict(cnt)}")
for k, v in unreg.most_common(20):
    print(f"UNREGISTERED x{v}: {k}")
print("V1 BODY DIFF", "PASS" if ok_added and not unreg else "FAIL",
      f"(new names {len(added)}/8 none@4e-3, unregistered {sum(unreg.values())})")
