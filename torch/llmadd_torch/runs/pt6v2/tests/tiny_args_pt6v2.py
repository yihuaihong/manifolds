"""pt6v2 test 3 helper: the jobs' own trainer command line (make_jobs_pt6v2.args_raw, main-run form) of the PLACEHOLDER composition,
seed 16, lr 2e-3, for the arms given; --out = <root>/<arm>. One line per arm: '<arm>\t<args>'. Read only.
    python tiny_args_pt6v2.py <root> all6 none only-compass only-hcube"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
root, arms = sys.argv[1], sys.argv[2:]
sys.argv = sys.argv[:1]
sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt6v2 as V   # noqa: E402  (asserts the frozen generators' shas on import)

doms = V.comp_doms(V.PLACEHOLDER)
for a in arms:
    assert a in V.arms(doms), (a, V.arms(doms))
    print(f"{a}\t{V.args_raw(doms, a, V.SAME_LR, 16, str(Path(root) / a))}")
