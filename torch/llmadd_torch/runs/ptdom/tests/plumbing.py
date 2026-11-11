"""Plumbing of the stage chain on synthetic runs (readout_ptdom.fake_run; nothing is written into runs/ptdom; A1 grids:
none 1e-3 .. 1.6e-2, prior arms 5e-4 .. 2e-3; A2 rules; picks files carry amendment A2, an A1 picks file is refused; the A1
version of this script is tests/plumbing_a1.py, its log tests/plumbing.out):
select_ptdom.py --stage A on a synthetic stage-A root -> picks_A.json; make_jobs_ptdom.py --stage X / C --picks it
(--no-write --print) renders exactly the X cells (seeds 0-3) and the C cells (seeds 4-7) the decision lists; a picks file
with an X lr two steps beyond the grid edge, or a non-ptdom picks file, is refused by the generator; select refuses an
existing output file and a missing stage-A cell."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "ptdom"
sys.path.insert(0, str(R))
import readout_ptdom as RO   # noqa: E402

PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
ok = []


def T(name, cond, val=None):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val is not None else ""))


def area_of(arm, lr, s):
    """only-stars: 2e-3 open upper (beats 1e-3 by 5); none: cities open upper at 1.6e-2; everything else peaks mid-grid."""
    a = {}
    for d in RO.SIX:
        if arm == "none":
            a[d] = ({"1e-3": 7.0, "2e-3": 8.0, "4e-3": 10.0, "8e-3": 14.0, "1.6e-2": 11.0}[lr] if d != "cities" else
                    {"1e-3": 6.0, "2e-3": 8.0, "4e-3": 10.0, "8e-3": 12.0, "1.6e-2": 16.0}[lr])
        elif arm == f"only-{d}":
            a[d] = {"5e-4": 30.0, "1e-3": 35.0, "2e-3": 31.0}[lr] if d != "stars" else {"5e-4": 30.0, "1e-3": 33.0, "2e-3": 38.0}[lr]
        else:
            a[d] = 20.0
        a[d] += 0.01 * s
    return a


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "out"
    for arm in RO.ARMS:
        for lr in RO.GRID[arm]:
            for s in RO.SEL:
                RO.fake_run(root, arm, lr, s, area_of(arm, lr, s))
    pa = Path(tmp) / "picks_A.json"
    o = subprocess.run([PY, str(R / "select_ptdom.py"), "--stage", "A", "--root", str(root), "--out", str(pa)], capture_output=True, text=True)
    T("select_ptdom --stage A on 92 synthetic runs (A2 per-cell gate passed: a / (a + b) 0.9)", o.returncode == 0 and pa.exists() and "AUX GATE passed (every cell)" in o.stdout
      and len(list(root.iterdir())) == 92, o.stdout[-300:] + o.stderr[-300:])
    pk = json.loads(pa.read_text())
    T("decision: X = only-stars@4e-3, none@3.2e-2; C = 5 prior picks @1e-3 + none@8e-3",
      sorted(map(tuple, pk["X"])) == [("none", "3.2e-2"), ("only-stars", "4e-3")]
      and sorted(map(tuple, pk["C"])) == sorted([(f"only-{d}", "1e-3") for d in RO.SIX if d != "stars"] + [("none", "8e-3")]), (pk["X"], pk["C"]))
    o2 = subprocess.run([PY, str(R / "select_ptdom.py"), "--stage", "A", "--root", str(root), "--out", str(pa)], capture_output=True, text=True)
    T("select refuses to overwrite a decision file", o2.returncode != 0 and "exists" in (o2.stdout + o2.stderr))
    for stage, want_n, seeds in (("X", 8, {0, 1, 2, 3}), ("C", 24, {4, 5, 6, 7})):
        o = subprocess.run([PY, str(B / "make_jobs_ptdom.py"), "--stage", stage, "--picks", str(pa), "--no-write", "--print"],
                           capture_output=True, text=True, cwd=B)
        names = [l.split(":")[0] for l in o.stdout.splitlines() if l.startswith("ptdom-53m-k2-")]
        cells = {tuple(n.rsplit("-s", 1)[0].replace("ptdom-53m-k2-", "").rsplit("-lr", 1)) for n in names}
        want = {tuple(c) for c in pk[stage]}
        T(f"generator --stage {stage} --picks: {want_n} jobs = the decision's cells x seeds {sorted(seeds)}",
          o.returncode == 0 and len(names) == want_n and cells == want and {int(n.rsplit("-s", 1)[1]) for n in names} == seeds,
          (o.returncode, len(names), o.stderr[-300:]))
    T("no job file was written by --no-write", not list(R.glob("job_ptdom-53m-k2-only-stars-lr4e-3-*")))
    bad = json.loads(pa.read_text()); bad["X"] = [["only-stars", "8e-3"]]
    pb = Path(tmp) / "bad.json"; pb.write_text(json.dumps(bad))
    o = subprocess.run([PY, str(B / "make_jobs_ptdom.py"), "--stage", "X", "--picks", str(pb), "--no-write"], capture_output=True, text=True, cwd=B)
    T("generator refuses an X lr two steps beyond the grid edge", o.returncode != 0 and "AssertionError" in o.stderr)
    bad2 = dict(json.loads(pa.read_text()), design="rb6"); pb.write_text(json.dumps(bad2))
    o = subprocess.run([PY, str(B / "make_jobs_ptdom.py"), "--stage", "C", "--picks", str(pb), "--no-write"], capture_output=True, text=True, cwd=B)
    T("generator refuses a picks file of another design", o.returncode != 0)
    pre = json.loads(pa.read_text()); pre.pop("amendment"); pb.write_text(json.dumps(pre))
    o = subprocess.run([PY, str(B / "make_jobs_ptdom.py"), "--stage", "C", "--picks", str(pb), "--no-write"], capture_output=True, text=True, cwd=B)
    T("generator refuses a picks file without an amendment (a decision of the pre-A1 code)", o.returncode != 0 and "AssertionError" in o.stderr)
    pre = dict(json.loads(pa.read_text()), amendment="A1"); pb.write_text(json.dumps(pre))
    o = subprocess.run([PY, str(B / "make_jobs_ptdom.py"), "--stage", "C", "--picks", str(pb), "--no-write"], capture_output=True, text=True, cwd=B)
    T("generator refuses a picks file of amendment A1 (decided by the A1 rules; A2 is required)", o.returncode != 0 and "AssertionError" in o.stderr)
    T("the decision is amendment A2 with the per-cell gate summary (18 cells) and no GATE unit",
      pk["amendment"] == "A2" and len(pk["aux_gate"]) == 18 and pk["GATE"] == [] and pk["hold_written"] is False)
    for xlr, want_ok in (("5e-4", True), ("2e-3", False), ("2.5e-4", False)):
        bx = dict(json.loads(pa.read_text()), X=[["none", xlr]]); pb.write_text(json.dumps(bx))
        o = subprocess.run([PY, str(B / "make_jobs_ptdom.py"), "--stage", "X", "--picks", str(pb), "--no-write", "--print"],
                           capture_output=True, text=True, cwd=B)
        T(f"generator {'accepts' if want_ok else 'refuses'} an X cell none@{xlr} (A1 none grid edges 1e-3 / 1.6e-2: X 5e-4 / 3.2e-2 only)",
          (o.returncode == 0 and sum(l.startswith("ptdom-53m-k2-none-lr5e-4-s") for l in o.stdout.splitlines()) == 4) if want_ok else (o.returncode != 0 and "AssertionError" in o.stderr))
    shutil.rmtree(root / "ptdom-53m-k2-none-lr8e-3-s2")
    o = subprocess.run([PY, str(R / "select_ptdom.py"), "--stage", "A", "--root", str(root), "--out", str(Path(tmp) / "p2.json")],
                       capture_output=True, text=True)
    T("select refuses a missing stage-A cell", o.returncode != 0 and "refused" in (o.stdout + o.stderr), (o.stdout + o.stderr)[-200:])
print("PLUMBING", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)})")
