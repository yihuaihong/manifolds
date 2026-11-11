"""pt2same regression: pt2s505's own frozen generator (make_jobs_pt2s505.py, sha256[:12] a5a7fb60eedc, imported, never run
as main: it would re-snapshot / write) still renders every one of its 32 job files byte-identically (24 main + 8 confirm,
render() of each file's cell and seed), and tasks() lists exactly those 32. Also: the other frozen inputs this batch
imports are unchanged (make_jobs_pttpp.py, make_jobs_rb6scale.py, the refs files), and this batch wrote nothing into
runs/pt2s505 (newest mtime there predates this batch). Runs in its own process (pt2s505's G2 lives in runs/pt2s505/code).
    python pt2s505_regress.py"""
import hashlib
import re
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
fails = 0


def say(ok, msg):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + msg)


for f, v in {"make_jobs_pt2s505.py": "a5a7fb60eedc", "make_jobs_pttpp.py": "b4bfa6eb248d", "make_jobs_rb6scale.py": "19bc055c2d09",
             "make_jobs_ptlr1b.py": "8dd7b3407459", "pt2s505_refs.json": "479e394771e9", "pttpp_refs.json": "d10d720c1242"}.items():
    say(sha(B / f) == v, f"frozen {f} sha256[:12] {sha(B / f)} (want {v})")
sys.path.insert(0, str(B))
import make_jobs_pt2s505 as P2   # noqa: E402

G2 = P2._g2()
R = B / "runs" / "pt2s505"
files = sorted(R.glob("job_*.sbatch"))
same = 0
for f in files:
    m = re.fullmatch(r"job_pt2s505-lr(.+)-(none|win-ctx)-s(\d)\.sbatch", f.name)
    lr, arm, seed = m.group(1), m.group(2), int(m.group(3))
    text = P2.render(G2, arm, lr, seed, P2.WANT_SHA["pretrain_range.py"])[2]
    ok = text.encode() == f.read_bytes()
    same += ok
    if not ok:
        print(f"  differs: {f.name}")
say(len(files) == 32 and same == 32, f"pt2s505 render() = its job file byte for byte: {same}/{len(files)}")
T = P2.tasks()
say(sorted(f"job_{t[0]}.sbatch" for t in T) == [f.name for f in files] and all(t[2].encode() == (R / f"job_{t[0]}.sbatch").read_bytes() for t in T),
    f"pt2s505 tasks() (24 main + {len(T) - 24} confirm from picks_confirm.json) = the 32 job files")
newest = max(p.stat().st_mtime for p in R.rglob("*") if "__pycache__" not in p.parts)
mine = (B / "runs" / "pt2same" / "code" / "COMMIT").stat().st_mtime
say(newest < mine, f"nothing in runs/pt2s505 is newer than this batch's snapshot (newest {newest:.0f} < {mine:.0f})")
print(f"PT2S505 REGRESSION {'PASS' if not fails else 'FAIL'} ({fails} failures)")
sys.exit(1 if fails else 0)
