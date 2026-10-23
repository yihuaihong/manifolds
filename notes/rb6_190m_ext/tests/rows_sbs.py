"""Item vii: decoded training / eval rows of the new arm types side by side with the main batch's dry log of the same
seed (CPU --dry_inputs, the jobs' own ARGS), plus the per-domain prior / lam of the 'domains:' line of each.
    python rows_sbs.py > scale_work/rb190ext/rows_side_by_side.txt"""
import json
import re
from pathlib import Path

W = Path("/scratch/yh6210/llmadd_torch/scale_work")
MAIN = {0: "dry_190m_all6-lam1_lr6e-4_s0.log", 3: "dry_190m_all6-lam1_lr6e-4_s3.log"}
for s in (0, 3):
    main = (W / "refs_out" / MAIN[s]).read_text().splitlines()
    for arm, lr in (("all6-minus-clock", "6e-4"), ("only-clock", "6e-4"), ("none", "1.92e-2")):
        new = (W / "rb190ext" / f"dry_{arm}_lr{lr}_s{s}.log").read_text().splitlines()
        print(f"===== seed {s}: {arm}@{lr} (left) vs main all6-lam1@6e-4 (right)")
        for nm, txt in ((f"{arm}@{lr}", new), ("main all6-lam1@6e-4", main)):
            dl = json.loads(re.search(r"^domains: (\{.*\})$", "\n".join(txt), re.M).group(1))
            print(f"  domains: line of {nm}: " + "; ".join(f"{d} prior {v.get('prior')} lam {v.get('lam')}"
                                                         + (f" target {v['target']}" if 'target' in v else "") for d, v in dl.items()))
        for pat, k in ((r"^  train row ", 3), (r"^  train line ", 3), (r"^  eval ", 3)):
            a = [l for l in new if re.match(pat, l)][:k]
            b = [l for l in main if re.match(pat, l)][:k]
            for x, y in zip(a, b):
                print(f"  {'SAME' if x == y else 'DIFF'} | {x[:150]}")
                if x != y:
                    print(f"       | {y[:150]}")
