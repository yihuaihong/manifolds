"""Regression: building pt300x changed nothing of the frozen generators and their batches.
1. every file in runs/pttpp/FROZEN.txt still has its recorded sha256[:12];
2. make_jobs_pttpp.render() (imported read-only, as make_jobs_pt300x does) re-renders every submitted pttpp job file byte for
   byte (all groups + the bridge), after make_jobs_pt300x was imported and used in the same process;
3. make_jobs_ptlr1b.job_text() re-renders every ptlr_1b job file byte for byte;
4. make_jobs_ptlr.py (F87; a script that writes its job files at import, so it is not run) is unchanged since this build
   started (sha256[:12] below) and its 3e8 job files are untouched (mtime before 2026-11-02).
    python tests/regress_frozen.py"""
import datetime
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
B = Path("/scratch/yh6210/llmadd_torch")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def frozen():
    bad = []
    for l in (B / "runs/pttpp/FROZEN.txt").read_text().splitlines():
        if l.startswith("#") or not l.strip():
            continue
        v, p = l.split(maxsplit=1)
        if sha(p) != v:
            bad.append((p, sha(p), v))
    return bad


def main():
    ok = True
    b0 = frozen()
    print(f"FROZEN.txt before: {'26/26' if not b0 else b0}")
    import make_jobs_pt300x as X   # the pt300x generator (imports make_jobs_pttpp read-only) and a full rendering
    n300 = sum(len(X.render(s)) for s in X.GRID)
    M = X.M
    n, diff, missing = 0, [], []
    for g in M.all_groups():
        for name, txt, F in M.render(g):
            f = M.R / f"job_{name}.sbatch"
            if not f.exists():
                missing.append(name)
                continue
            n += 1
            if f.read_text() != txt:
                diff.append(name)
    flags = len(list(M.R.glob("SUBMITTED_*")))
    print(f"pttpp: {n} job files re-rendered by the frozen make_jobs_pttpp ({flags} SUBMITTED flags): {len(diff)} differ"
          f"{': ' + str(diff[:5]) if diff else ''}; rendered runs without a file: {len(missing)}")
    ok &= not diff and n == flags and not missing
    M1 = M.M1
    n1, d1 = 0, []
    for i, (lr, a, s) in enumerate(M1.cells()):
        jt, txt = M1.job_text(i, lr, a, s)
        f = M1.R / f"job_{jt}.sbatch"
        n1 += 1
        if not f.exists() or f.read_text() != txt:
            d1.append(jt)
    print(f"ptlr_1b: {n1} job files re-rendered by the frozen make_jobs_ptlr1b: {len(d1)} differ{': ' + str(d1[:5]) if d1 else ''}")
    ok &= not d1
    g = B / "make_jobs_ptlr.py"
    cut = datetime.datetime(2026, 11, 2).timestamp()
    new = [f.name for f in (B / "runs/ptlr").glob("job_*.sbatch") if f.stat().st_mtime >= cut]
    print(f"make_jobs_ptlr.py (F87) sha256[:12] {sha(g)} (mtime {datetime.datetime.fromtimestamp(g.stat().st_mtime):%Y-%m-%d %H:%M}); "
          f"runs/ptlr job files modified since 2026-11-02: {len(new)}")
    ok &= not new and g.stat().st_mtime < cut
    b1 = frozen()
    print(f"FROZEN.txt after: {'26/26' if not b1 else b1}")
    ok &= not b0 and not b1
    print(f"(pt300x rendered {n300} runs in the same process)")
    print(f"REGRESSION {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
