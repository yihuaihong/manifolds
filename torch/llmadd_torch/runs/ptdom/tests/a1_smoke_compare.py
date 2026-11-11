"""Amendment A1: the 200-step CPU smoke re-run (smoke_200.sh into smoke_a1) vs the original smoke/ (the trainer and the
snapshot are unchanged by A1): run JSON equal except args.out and the per-eval wall clock 'sec'; saved models bitwise."""
import glob
import json
import sys
from pathlib import Path

import torch

A, Bn = Path(sys.argv[1]), Path(sys.argv[2])
ok = True
for arm in ("stars", "none"):
    ja, jb = (json.loads(Path(glob.glob(str(W / arm / "run_*.json"))[0]).read_text()) for W in (A, Bn))
    strip = lambda j: {**j, "args": {k: v for k, v in j["args"].items() if k != "out"},
                       "curve": [{k: v for k, v in e.items() if k != "sec"} for e in j["curve"]]}
    sa, sb = strip(ja), strip(jb)
    diff = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    ma, mb = (torch.load(glob.glob(str(W / arm / "model_*.pt"))[0], map_location="cpu") for W in (A, Bn))
    tens = set(ma) == set(mb) and all(torch.equal(ma[k], mb[k]) for k in ma)
    same = not diff and tens
    ok &= same
    print(f"{'PASS' if same else 'FAIL'} smoke {arm}: re-run = original (JSON keys differing {diff}; {len(ma)} tensors "
          f"{'bitwise equal' if tens else 'DIFFER'}; {len(ja['curve'])} evals)")
print("A1 SMOKE RE-RUN", "PASS" if ok else "FAIL")
