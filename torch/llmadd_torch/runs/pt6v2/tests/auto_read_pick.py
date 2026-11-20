"""auto_pt6v2.sh helper: validate a PICK.json (pick_pt6v2.py --json; keys "composition" = 5 ALLOWED domains, "extra_none_6.4e-2" =
bool) and print shell-readable lines: COMP=<d1,..,d5> EXTRA=0|1 STATE=new|same|differ (vs COMPOSITION.json) REFS=0|1 (the CPU refs
file exists) MK=<refs key = trainer-order domains joined by '-'>. `--placeholder` prints MK for the placeholder composition only."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_pt6v2 as V  # noqa: E402
sys.argv = _argv

if sys.argv[1] == "--placeholder":
    print("MK=" + "-".join(V.PLACEHOLDER))
    sys.exit(0)
j = json.loads(Path(sys.argv[1]).read_text())
# coordinator 2026-11-19 (audit_diff_V2.md I1): a pick_pt6v2.py output must be final (strict, complete); stubs without these keys
# serve dry tests only (auto_pt6v2.sh refuses --go with --pick_json)
assert j.get("complete", True) is True and j.get("mode", "strict") == "strict", \
    f"PICK not final: complete={j.get('complete')!r} mode={j.get('mode')!r} (an --allow_partial standing is never a pick)"
comp, ex = j["composition"], j["extra_none_6.4e-2"]
assert isinstance(comp, list) and len(comp) == 5 and len(set(comp)) == 5, f"composition {comp}: 5 distinct domains"
assert all(d in V.ALLOWED for d in comp), f"composition {comp}: not all in ALLOWED {sorted(V.ALLOWED)}"
assert isinstance(ex, bool), f"extra_none_6.4e-2 {ex!r}: a bool"
doms = V.comp_doms(list(comp))
dec = V.decided_comp()
state = "new" if dec is None else ("same" if tuple(dec) == tuple(doms) else "differ")
print(f"COMP={','.join(comp)}")
print(f"EXTRA={int(ex)}")
print(f"STATE={state}")
print(f"REFS={int(V.refs_path(doms).exists())}")
print("MK=" + "-".join(doms))
