"""pt6comb test: composition A (numbers + months + cities + countries + people + X) stays a parameter (coordinator 2026-11-10): every arm
renders through make_jobs_ptdom.render with a SUPPLIED placeholder reference (no CPU references exist for A), bash -n, the arm's ARGS and
--domains in the trainer's order; --submit is refused for A. -> comp_a_render.out"""
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
sys.argv = [sys.argv[0]]
import make_jobs_pt6comb as M6   # noqa: E402
P = M6.load_P()
res = []
for x in ("landmarks", "colours", "compass", "scalar_adj"):
    doms = M6.comp_doms("A", x)
    dom = {d: {k: 0 for k in M6.DOM_KEYS6[M6.dom_kind6(d)]} for d in doms}
    ref = {"lm0": 11.6, "dsha0": "placeholder", "dom": dom, "table_sha": {d: "x" for d in doms if d != "months"}, "dry": ["0" * 64, 1]}
    n = 0
    for a in M6.arms("A", x):
        name, tag, text, args, I = M6.render6(P, "L1" if a in ("none", "all6") else "C", "A", a, "2e-3", 0, ref, M6.COMMIT, 0, 0, x)
        f = P.flags(args)
        ok = (subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0 and f["--domains"] == "numbers," + ",".join(doms)
              and I["n_fmt"] == {"months": 4} and ("-A-" + x + "-") in name)
        n += ok
        res.append(ok)
    print(f"{'PASS' if n == len(M6.arms('A', x)) else 'FAIL'} composition A X={x}: --domains numbers,{','.join(doms)}; {n}/{len(M6.arms('A', x))} arms render (bash -n, ARGS, n_fmt months 4)")
o = subprocess.run([M6.PY, str(M6.B / "make_jobs_pt6comb.py"), "--part", "L1", "--comp", "A", "--x", "landmarks", "--submit"], capture_output=True, text=True)
refused = o.returncode != 0 and "not submittable" in (o.stdout + o.stderr)
res.append(refused)
print(f"{'PASS' if refused else 'FAIL'} --submit for composition A is refused before anything is rendered or submitted (rc {o.returncode})")
print(f"COMP A RENDER {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
