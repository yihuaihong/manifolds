"""T8: the runs to act on, from the generator under argv[1] and the batch dir (no metric is read).
  b200: B200 runs whose slot changes with 8 slots (index >= 4, i % 4 != i % 8) and that are submitted: '<name> <job id>'
  lost: the two runs whose jobs exited at start on HOLD (job gone, log 'HOLD present: exiting', no run JSON): '<name> <job id>'"""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
R = T.R
what = sys.argv[2]
if what == "b200":
    for i, (g, st, a, lr, s) in enumerate(T.b200_order()):
        n = T.facts(g, a, lr, s, st)["name"]
        f = R / f"SUBMITTED_{n}"
        if i >= 4 and i % 4 != i % 8 and f.exists():
            print(n, f.read_text().split()[-1])
elif what == "lost":
    want = {"tpp7-190m-t0.318-lr4.8e-3-win-ctx-s5": "21080784", "tpp7-190m-t0.318-lr3.84e-2-none-s2": "21084173"}
    for gid in ("tpp7-190m-t0.318",):
        for st, a, lr, s in T.runs_of(gid):
            F = T.facts(gid, a, lr, s, st)
            if F["name"] not in want:
                continue
            jid = (R / f"SUBMITTED_{F['name']}").read_text().split()[-1]
            assert jid == want[F["name"]], (F["name"], jid)
            tgt = "none" if a == "none" else "helix3"
            assert not (R / "out" / F["tag"] / f"run_whole_{tgt}_seed{s}_{F['tag']}.json").exists(), F["name"]
            log = Path(f"/scratch/yh6210/llmadd_torch/logs/llmadd_tpp-{F['name']}_{jid}.out")
            assert log.read_text().strip() == "HOLD present: exiting", log
            print(F["name"], jid)
