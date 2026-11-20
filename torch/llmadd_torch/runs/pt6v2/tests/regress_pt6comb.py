"""pt6v2 test 1 (regression): make_jobs_pt6v2.render_v2 on pt6comb composition B, seeds 8-15, B200, with pt6comb's slots and
CPU references, against the 80 pt6comb C job files (runs/pt6comb/job_pt6comb-B-*-s{8..15}.sbatch; read only).
Normalised: pt6v2 names / dirs -> pt6comb's. Every residual diff line must fall in a registered category (AUDIT.md test 1):
  trainer      Rev H vs pt6comb's code: COMMIT, sha256 of pretrain_range.py / deps, intent code_sha / dep_sha
  guard        the GPU-type guard + DEVICE line (seed-allocated type)
  intent_dev   the intent's "device" key (and the intent line that carries the shas)
  comments     header / provenance / HOLD comments; the CHECK_PY / site-rule comment text
  hold         HOLD_<batch> -> HOLD_<device>
  slots        '8 chains shared by every pt6v2 B200 job' (pt6comb: 16 chains over the four batches)
-> PASS iff no line is UNREGISTERED. Output: stdout (tests/regress_pt6comb.log)."""
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_pt6v2 as V  # noqa: E402
sys.argv = _argv
import make_jobs_pt6comb as M6  # noqa: E402

P = V.load_P()
commit = V.snapshot()
REF = json.loads((B / "runs/pt6comb/pt6comb_refs_B.json").read_text())["seeds"]
doms = V.comp_doms(M6.COMP_B)
assert doms == tuple(M6.COMP_B), (doms, M6.COMP_B)


def norm(t):
    t = t.replace("runs/pt6v2", "runs/pt6comb").replace("make_jobs_pt6v2.py", "make_jobs_pt6comb.py")
    t = re.sub(r"llmadd_pt6v2-b200-slot", "llmadd_pt6comb-b200-slot", t)
    return re.sub(r"pt6v2-", "pt6comb-B-", t)


CATS = [("trainer", re.compile(r"ef7b8854|a398fccbfe93|11f61b9|COMMIT|sha256sum|code_sha|dep_sha|toy3")),
        ("guard", re.compile(r'case "\$GPU"|^echo "DEVICE ')),
        ("intent_dev", re.compile(r'"device"|^I=|intent')),
        ("comments", re.compile(r"^#")),
        ("hold", re.compile(r"HOLD")),
        ("slots", re.compile(r"chains"))]
cnt, unreg, n = Counter(), Counter(), 0
for f in sorted(B.glob("runs/pt6comb/job_pt6comb-B-*-s*.sbatch")):
    m = re.fullmatch(r"job_pt6comb-B-(.+)-lr([0-9.e-]+)-s(\d+)\.sbatch", f.name)
    if not m or not 8 <= int(m[3]) <= 15:
        continue
    arm, lr, seed = m[1], m[2], int(m[3])
    old = f.read_text()
    slot = int(re.search(r"job-name=llmadd_pt6comb-b200-slot(\d+)", old)[1])
    name, tag, text, args, I = V.render_v2(P, doms, arm, lr, seed, REF[str(seed)], commit, "b200", slot, seeds=tuple(range(8, 16)))
    n += 1
    for ln in difflib.unified_diff(old.splitlines(), norm(text).splitlines(), lineterm="", n=0):
        if ln.startswith(("---", "+++", "@@")):
            continue
        body = ln[1:]
        c = next((k for k, rx in CATS if rx.search(body)), None)
        if c:
            cnt[c] += 1
        else:
            unreg[ln[:220]] += 1
print(f"regression: {n} pt6comb C job files (comp B, seeds 8-15, B200) re-rendered by make_jobs_pt6v2.render_v2")
print("registered residual diff lines:", dict(cnt))
for k, v in unreg.most_common(30):
    print(f"UNREGISTERED x{v}: {k}")
print("REGRESSION", "PASS" if n == 80 and not unreg else "FAIL", f"(files {n}/80, unregistered {sum(unreg.values())})")
