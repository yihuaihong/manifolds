"""Compare the old (origin/main, 93e22bf4c4c0) and new (Rev E, runs/ptdom/code) trainer outputs of reg_trainer.sh:
run JSON equal except code_sha, args.out and the per-eval 'sec'; saved model tensors bitwise equal; --dry_inputs stdout
equal after replacing the out dir; the run stdout equal after removing 'sec' and the code line."""
import glob
import json
import re
import sys
from pathlib import Path

import torch

W = Path(sys.argv[1])
ok_all, missing = True, []   # A2: a cell that was not run is PARTIAL, not FAIL (a FAIL is a difference)
for c in ("default", "rb6all6", "rb6none"):
    res = {}
    for v in ("old", "new"):
        f = glob.glob(str(W / f"{c}-{v}" / "run" / "run_*.json"))
        m = glob.glob(str(W / f"{c}-{v}" / "run" / "model_*.pt"))
        res[v] = (json.loads(Path(f[0]).read_text()) if f else None, torch.load(m[0], map_location="cpu") if m else None)
    (jo, mo), (jn, mn) = res["old"], res["new"]
    if jo is None or jn is None:
        print(f"{c}: MISSING run JSON (old {jo is not None}, new {jn is not None})"); missing.append(c); continue
    strip = lambda j: {**{k: v for k, v in j.items() if k not in ("code_sha",)}, "args": {k: v for k, v in j["args"].items() if k != "out"},
                       "curve": [{k: v for k, v in e.items() if k != "sec"} for e in j["curve"]]}
    so, sn = strip(jo), strip(jn)
    diff = sorted(k for k in set(so) | set(sn) if so.get(k) != sn.get(k))
    cd = [i for i, (a, b) in enumerate(zip(so["curve"], sn["curve"])) if a != b]
    ckeys = sorted({k for i in cd for k in set(so["curve"][i]) | set(sn["curve"][i]) if so["curve"][i].get(k) != sn["curve"][i].get(k)})
    tens = (mo is not None and mn is not None and set(mo) == set(mn) and all(torch.equal(mo[k], mn[k]) for k in mo))
    n_t = len(mo) if mo else 0
    dry_o, dry_n = ((W / f"{c}-{v}.dry").read_text().replace(str(W / f"{c}-{v}"), "<OUT>") for v in ("old", "new"))
    log_s = lambda t: [re.sub(r"^\S+pretrain_range.py:\d+: UserWarning", "UserWarning", re.sub(r"\"sec\": [0-9.e+-]+", "\"sec\": _", l)) for l in t.splitlines()
                       if not l.startswith(("code ", "done in")) and "run_guard" not in l]
    lo, ln = (log_s((W / f"{c}-{v}.log").read_text().replace(str(W / f"{c}-{v}"), "<OUT>")) for v in ("old", "new"))
    ok = not diff and tens and dry_o == dry_n and lo == ln and jo["code_sha"] != jn["code_sha"]
    ok_all &= ok
    print(f"{c}: {'IDENTICAL' if ok else 'DIFFERENT'}: JSON keys differing {diff} (curve evals {cd}, keys {ckeys[:8]}); "
          f"code_sha {jo['code_sha']} -> {jn['code_sha']}; model {n_t} tensors {'bitwise equal' if tens else 'DIFFER'}; "
          f"dry stdout {'equal' if dry_o == dry_n else 'DIFFER'} ({len(dry_o.splitlines())} lines); run stdout "
          f"{'equal' if lo == ln else 'DIFFER'} ({len(lo)} lines); evals {len(jo['curve'])}")
print("REGRESSION", ("PASS" if not missing else f"PARTIAL (identical where run; NOT RUN: {', '.join(missing)})") if ok_all else "FAIL")
