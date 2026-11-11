"""Regression: building pt2sz changed nothing of the frozen generators and their submitted batches.
1. every file of runs/pt2same/FROZEN.sha256 and runs/pttpp/FROZEN.txt still has its recorded sha256, before and after;
   pttpp_refs.json / pt2s505_refs.json / rb6scale_refs.json unchanged (pt2sz writes only its own pt2sz_refs.json);
2. each frozen generator, imported read-only in its OWN process (each g2() imports make_pt2main_tasks from its own snapshot),
   re-renders its submitted job files byte for byte: make_jobs_pt2same.jobs() = the 32 runs/pt2same job files (and
   runs/pt2same/a1/JOBS.sha256 still verifies); make_jobs_pt2s505.render() = its 32 job files and tasks() = them;
   make_jobs_pttpp.render() = every submitted pttpp job file (all groups + the bridge);
3. make_jobs_pt2sz imported and run (--check, in a process of its own) between the before / after checks;
4. nothing in runs/pt2same, runs/pt2s505, runs/pttpp, runs/pt300x is newer than this build's start (runs/pt2sz/code/COMMIT).
    python regress_frozen.py"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def frozen():
    bad = []
    for l in (B / "runs/pt2same/FROZEN.sha256").read_text().splitlines():
        v, p = l.split(maxsplit=1)
        if sha(B / p) != v:
            bad.append(p)
    for l in (B / "runs/pttpp/FROZEN.txt").read_text().splitlines():
        if l.startswith("#") or not l.strip():
            continue
        v, p = l.split(maxsplit=1)
        if sha(p)[:12] != v:
            bad.append(p)
    refs = {"pttpp_refs.json": "d10d720c1242", "pt2s505_refs.json": "479e394771e9", "rb6scale_refs.json": "b65472eb3d29"}
    bad += [f for f, v in refs.items() if sha(B / f)[:12] != v]
    return bad


def child(which):
    sys.path.insert(0, str(B))
    if which == "pt2same":
        import make_jobs_pt2same as PS
        G2 = PS.g2()
        J = PS.jobs(G2)
        files = sorted(PS.R.glob("job_*.sbatch"))
        same = sum((PS.R / f"job_{n}.sbatch").read_text() == t for n, _, t, *_ in J)
        a1 = subprocess.run(["sha256sum", "--quiet", "-c", "JOBS.sha256"], cwd=PS.R / "a1", capture_output=True, text=True)
        main = subprocess.run(["sha256sum", "--quiet", "-c", "JOBS.sha256"], cwd=PS.R, capture_output=True, text=True)
        ok = len(J) == len(files) == 32 and same == 32 and a1.returncode == 0 and main.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'} make_jobs_pt2same.jobs() re-renders {same}/{len(files)} runs/pt2same job files byte for byte; "
              f"runs/pt2same/JOBS.sha256 and a1/JOBS.sha256 verify (rc {main.returncode} / {a1.returncode})")
    elif which == "pt2s505":
        import make_jobs_pt2s505 as P2
        G2 = P2._g2()
        files = sorted(P2.R.glob("job_*.sbatch"))
        same = 0
        for f in files:
            m = re.fullmatch(r"job_pt2s505-lr(.+)-(none|win-ctx)-s(\d)\.sbatch", f.name)
            same += P2.render(G2, m.group(2), m.group(1), int(m.group(3)), P2.WANT_SHA["pretrain_range.py"])[2] == f.read_text()
        T = P2.tasks()
        ok = len(files) == 32 and same == 32 and sorted(f"job_{t[0]}.sbatch" for t in T) == [f.name for f in files]
        print(f"{'PASS' if ok else 'FAIL'} make_jobs_pt2s505.render() re-renders {same}/{len(files)} job files byte for byte; tasks() = them")
    elif which == "pttpp":
        import make_jobs_pttpp as TP
        n, diff, missing = 0, [], []
        for g in TP.all_groups():
            for name, txt, F in TP.render(g):
                f = TP.R / f"job_{name}.sbatch"
                if not f.exists():
                    missing.append(name)
                    continue
                n += 1
                diff += [name] if f.read_text() != txt else []
        flags = len(list(TP.R.glob("SUBMITTED_*")))
        ok = not diff and not missing and n == flags
        print(f"{'PASS' if ok else 'FAIL'} make_jobs_pttpp.render() re-renders {n - len(diff)}/{n} submitted pttpp job files byte for byte "
              f"({flags} SUBMITTED flags; rendered runs without a file: {len(missing)})")
    else:
        raise SystemExit(which)
    sys.exit(0 if ok else 1)


def main():
    ok = True
    b0 = frozen()
    print(f"{'PASS' if not b0 else 'FAIL'} frozen files before: {'all OK' if not b0 else b0}")
    ok &= not b0
    o = subprocess.run([PY, str(B / "make_jobs_pt2sz.py"), "--check"], capture_output=True, text=True, env=ENV)
    print(f"{'PASS' if o.returncode == 0 else 'FAIL'} make_jobs_pt2sz.py --check (imports make_jobs_pt2same / pttpp / pt2s505 / rb6scale, "
          f"runs known_answers): {o.stdout.strip().splitlines()[-1] if o.stdout.strip() else o.stderr[-400:]}")
    ok &= o.returncode == 0
    for w in ("pt2same", "pt2s505", "pttpp"):
        o = subprocess.run([PY, __file__, w], capture_output=True, text=True, env=ENV)
        print(o.stdout.strip() or f"FAIL {w}: {o.stderr[-600:]}")
        ok &= o.returncode == 0
    b1 = frozen()
    print(f"{'PASS' if not b1 else 'FAIL'} frozen files after: {'all OK' if not b1 else b1}")
    ok &= not b1
    start = (B / "runs/pt2sz/code/COMMIT").stat().st_mtime
    newer = [str(p.relative_to(B)) for d in ("pt2same", "pt2s505", "pttpp", "pt300x") for p in (B / "runs" / d).rglob("*")
             if "__pycache__" not in p.parts and p.stat().st_mtime > start]
    print(f"{'PASS' if not newer else 'FAIL'} nothing in runs/pt2same, pt2s505, pttpp, pt300x newer than this build's start: "
          f"{newer[:5] if newer else 'none'}")
    ok &= not newer
    print(f"REGRESSION {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    child(sys.argv[1]) if len(sys.argv) > 1 else main()
