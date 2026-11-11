"""Regression: building ptsz changed nothing of the frozen generators and their batches.
1. every file in runs/pttpp/FROZEN.txt still has its recorded sha256[:12] (before and after, same process);
2. make_jobs_pttpp.render() (imported read-only, as make_jobs_ptsz does) re-renders every submitted pttpp job file byte for
   byte, after make_jobs_ptsz was imported and used (a full rendering) in the same process;
3. make_jobs_ptlr1b.job_text() re-renders every ptlr_1b job file byte for byte;
4. make_jobs_pt300x.render() (the template; e8fd563c335a, asserted by make_jobs_ptsz) re-renders every submitted pt300x job
   file byte for byte, and its decisions.json / picks.json / readout files are unchanged;
5. make_jobs_ptlr.py (F87; writes its job files at import, so it is not run) unchanged and its job files untouched since
   2026-11-05 (this build's start); pttpp_refs.json unchanged (d10d720c1242).
    python tests/regress_frozen.py"""
import datetime
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
B = Path("/scratch/yh6210/llmadd_torch")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
PT300X = {"runs/pt300x/decisions.json": "22532f929c7a", "runs/pt300x/picks.json": "61e1390722ba",
          "runs/pt300x/readout_pt300x.py": "c595c9dbb70c", "runs/pt300x/select_pt300x.py": "c4f07df1e18c",
          "make_jobs_pt300x.py": "e8fd563c335a", "pttpp_refs.json": "d10d720c1242", "make_jobs_ptlr.py": "85c3bf083bc9"}


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
    import make_jobs_ptsz as Z   # the ptsz generator (imports make_jobs_pttpp read-only) and a full rendering
    nz = sum(len(Z.render(s)) for s in Z.GRID)
    import make_jobs_pt300x as X
    assert X.M is Z.M
    M = Z.M
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
    n3, d3 = 0, []
    for size in X.GRID:
        for name, txt, F in X.render(size):
            f = X.P / f"job_{name}.sbatch"
            n3 += 1
            if not f.exists() or f.read_text() != txt:
                d3.append(name)
    f3 = len(list(X.P.glob("SUBMITTED_*")))
    print(f"pt300x: {n3} job files re-rendered by make_jobs_pt300x ({f3} SUBMITTED flags): {len(d3)} differ{': ' + str(d3[:5]) if d3 else ''}")
    ok &= not d3 and n3 == f3
    for p, v in PT300X.items():
        good = sha(B / p) == v
        ok &= good
        print(f"{p} sha256[:12] {sha(B / p)} {'= ' + v if good else '!= ' + v}")
    cut = datetime.datetime(2026, 11, 5).timestamp()
    new = [f.name for f in (B / "runs/ptlr").glob("job_*.sbatch") if f.stat().st_mtime >= cut]
    print(f"runs/ptlr job files modified since 2026-11-05: {len(new)}")
    ok &= not new
    b1 = frozen()
    print(f"FROZEN.txt after: {'26/26' if not b1 else b1}")
    ok &= not b0 and not b1
    print(f"(ptsz rendered {nz} runs in the same process)")
    print(f"REGRESSION {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
