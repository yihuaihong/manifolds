"""V2 build check (AUDIT.md "V2 build"): render the pt6v2 jobs with make_jobs_pt6v2.py before V2 (2f10290f95b2; its copy
tests/make_jobs_pt6v2.preV2_2f10290f95b2 = argv[1]) and after V2 (the live file), placeholder composition with its CPU references, the
same --alloc (argv[2]) and canary device (argv[3]).
  (a) default (no --extra_none): the same 105 names, every job text and trainer arg line byte-identical.
  (b) --extra_none 6.4e-2: the new names are exactly the 8 none@6.4e-2 runs; every common text identical except the B200 slot
      job-name and its 'singleton slot N' comment (the slot cycle shifts with 14 cells per seed).
  (c) sbatch --test-only (SLURM_* env stripped) on the canary, the first B200 slot job, the A100 jobs of seed 23 and the 6.4e-2 runs of
      s16 / s23, written to tests/v2_testonly/ (never to runs/pt6v2/).
Output: stdout (tests/v2_body_diff_V2.log)."""
import difflib
import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
old_path, ALLOC, CDEV = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_pt6v2 as NEW  # noqa: E402
sp = importlib.util.spec_from_file_location("make_jobs_pt6v2_old", old_path, loader=importlib.machinery.SourceFileLoader("make_jobs_pt6v2_old", str(old_path)))
OLD = importlib.util.module_from_spec(sp)
sp.loader.exec_module(OLD)
sys.argv = _argv
assert NEW.sha(old_path) == "2f10290f95b2" and NEW.sha(NEW.__file__) != "2f10290f95b2"


def render_all(V, extra=None):
    doms = V.PLACEHOLDER
    alloc = V.parse_alloc(ALLOC.split())
    P, commit = V.load_P(), V.snapshot()
    RF, RC = V.refs_v2(doms)
    assert RF and RC
    a, lr, s = V.CANARY
    jobs = [V.render_v2(P, doms, a, lr, s, RC, commit, CDEV, None, 0, True)]
    i = 0
    for s in V.SEEDS:
        for a, lr in (V.cells(doms) if extra is None else V.cells(doms, extra)):
            dev = alloc[s]
            slot = (i % V.N_SLOTS) if dev == "b200" else None
            i += dev == "b200"
            jobs.append(V.render_v2(P, doms, a, lr, s, RF[s], commit, dev, slot))
    return {j[0]: (j[2], j[3], j[4]["device"]) for j in jobs}, [j[0] for j in jobs]


(jo, oo), (jn, on), (jx, ox) = render_all(OLD), render_all(NEW), render_all(NEW, "6.4e-2")
print(f"alloc {ALLOC}; canary on {CDEV}; old {len(jo)} jobs, new default {len(jn)}, new --extra_none 6.4e-2 {len(jx)}")
same_a = sum(jo[n] == jn[n] for n in jo if n in jn)
ok_a = oo == on and len(jo) == 105 and same_a == 105
print(f"(a) default: names and order identical {oo == on}; identical texts + args {same_a}/{len(jo)}")
print("DEFAULT BYTE-IDENTITY", "PASS" if ok_a else "FAIL")

added = [n for n in ox if n not in jo]
print("(b) new names:", added)
ok_b = len(added) == 8 and all(re.fullmatch(r"pt6v2-none-lr6\.4e-2-s(1[6-9]|2[0-3])", n) for n in added) and not set(jo) - set(jx)
cnt, unreg, same = Counter(), Counter(), 0
for n in sorted(set(jo) & set(jx)):
    (to, ao, _), (tn, an, _) = jo[n], jx[n]
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
print(f"(b) common names {len(set(jo) & set(jx))}: identical {same}; residual diff lines {dict(cnt)}")
for k, v in unreg.most_common(20):
    print(f"UNREGISTERED x{v}: {k}")
for n in added[:1] + added[-1:]:
    print(f"    {n} [{jx[n][2]}]: {jx[n][1]}")
print("EXTRA_NONE DIFF", "PASS" if ok_b and not unreg else "FAIL", f"(new names {len(added)}/8 none@6.4e-2, unregistered {sum(unreg.values())})")

D = B / "runs/pt6v2/tests/v2_testonly"
D.mkdir(exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
pick = [ox[0], next(n for n in ox if jx[n][2] == "b200"), "pt6v2-none-lr6.4e-2-s16", "pt6v2-none-lr6.4e-2-s23"] + [n for n in ox if n.endswith("-s23")][:2]
refused = 0
for n in dict.fromkeys(pick):
    f = D / f"job_{n}.sbatch"
    f.write_text(jx[n][0])
    tl = re.search(r"^#SBATCH --time=(\S+)", jx[n][0], re.M)
    o = subprocess.run(["sbatch", "--test-only", str(f)], capture_output=True, text=True, env=env)
    refused += o.returncode != 0
    print(f"(c) {n} [{jx[n][2]}] --time={tl[1] if tl else '?'}: rc {o.returncode}: {(o.stdout + o.stderr).strip()[:200]}")
print("TEST-ONLY", "PASS" if not refused else "FAIL", f"({len(dict.fromkeys(pick)) - refused} of {len(dict.fromkeys(pick))} accepted)")
