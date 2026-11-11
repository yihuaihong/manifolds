"""ptdom2 test: select_ptdom2.py -> picks file -> make_jobs_ptdom2.py (the CLIs, on a synthetic world in a temp dir; nothing is
written in runs/ptdom or runs/ptdom2, no job is submitted: the generator runs with --no-write --print).
  1. synthetic ptdom cells (the registered PRIOR_LRS0 / NONE_LRS0, seeds 0-3) whose none data ask exactly R1PRE, the R1pre cells
     and ptdom2 R1 cells (4e-3: stars' and compass' pick and argmax) -> select --after R1 (the default R1pre, checked) writes
     picks_R_after1.json (chmod 444) with E = {only-stars@8e-3, only-compass@8e-3, B3 none@1e-3 (fixB)}; a second --after R1 is
     refused; --verify agrees; --part R1pre renders R1PRE x seeds 0-3, --part R1x exactly E x seeds 0-3
  2. the R1x cells planted (8e-3 below 4e-3; A's none falls at 1.6e-2) -> select --after R1x -> picks_R_after2.json (stage R2,
     every domain FINAL); the generator's --part R2 renders exactly C x seeds 8-15 (C = only-d@P, only-d-shuf@P, none@N per
     batch); --verify agrees; a tampered copy (one C cell changed, chmod 444) -> --verify exits 1 and the generator refuses it
  3. A1.8: a GATE domain (every prior cell of ranks fails the gate) -> GATE in the picks file, no HOLD, rc 0, the others go on
-> plumbing.out
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R2D = Path(__file__).resolve().parent.parent   # A4: the code under test (runs/ptdom2 or a working copy such as runs/ptdom2/wip_a4)
sys.path.insert(0, str(R2D))
sys.argv = [sys.argv[0]]
import readout_ptdom2 as RO2   # noqa: E402
M2 = RO2.M2
print(f"code under test: {R2D}", flush=True)
FLAGS = lambda: sorted(p.name for g in ("HOLD*", "picks_*", "SUBMITTED_*") for p in M2.R.glob(g))   # the real runs/ptdom2
FLAGS0 = FLAGS()
PY = M2.PY
ENV = dict(os.environ, PYTHONPATH=str(B / "pylib"))
res = []


def T(name, ok, info=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info != "" else ""), flush=True)


def sh(*a):
    o = subprocess.run([PY, *map(str, a)], capture_output=True, text=True, env=ENV, cwd=str(B))
    return o.returncode, o.stdout + o.stderr


base = {d: 70.0 for b in M2.BATCHES for d in M2.BATCHES[b]}


def plant(root, prefix, b, arm, lr, seeds, val, share=0.9):
    for s in seeds:
        a = dict(base)
        a.update({d: val(d, lr, s) for d in M2.BATCHES[b] if d in M2.R_DOMS})
        RO2.fake2(root, prefix, b, arm, lr, s, a, share=share)


prior = lambda d, lr, s: 70.0 + {"4e-3": 8.0 if d in ("stars", "compass") else -3.0, "8e-3": -5.0}.get(lr, 0.0)
# none: stars / landmarks pick 4e-3 (tie-break) / argmax 8e-3 = A's top (-> up), prefixes pick and argmax 1e-3 = B2's lowest (-> down):
# exactly R1PRE from the ptdom cells; the R1pre / R1x none cells are 20 points lower (never the argmax)
none_ = lambda d, lr, s: (50.0 if lr in ("1.6e-2", "5e-4") and d in M2.R_DOMS and M2.R_DOMS[d] in ("A", "B2") and (M2.R_DOMS[d], "none", lr) in M2.R1PRE
                          else 50.0 if lr == "1e-3" and d == "scalar_adj"
                          else 60.0 + (6.0 if lr == "8e-3" and d in ("stars", "landmarks") else 6.0 if lr == "1e-3" and d == "prefixes"
                                       else 4.0 if lr == "4e-3" else 0.0) + (-10.0 if lr == "1.6e-2" else 0.0))
gen_names = lambda out: sorted(re.findall(r"^(ptdom2-\S+?)(?: \(SUBMITTED\))?: --data ", out, re.M))   # A4: R1pre's jobs are submitted (flagged) now
with tempfile.TemporaryDirectory() as tmp:
    r0, r2, pdir = Path(tmp) / "p0", Path(tmp) / "p2", Path(tmp) / "picks"
    pdir.mkdir()
    for d, b in M2.R_DOMS.items():
        for lr in M2.PRIOR_LRS0[d]:
            plant(r0, "ptdom", b, f"only-{d}", lr, M2.SEL, prior)
        plant(r2, "ptdom2", b, f"only-{d}", "4e-3", M2.SEL, prior)
    for b, lrs in M2.NONE_LRS0.items():
        for lr in lrs:
            plant(r0, "ptdom", b, "none", lr, M2.SEL, none_)
    for b, a_, lr in M2.R1PRE:   # part R1pre, run with R1
        plant(r2, "ptdom2", b, "none", lr, M2.SEL, none_)
    common = ["--root0", r0, "--root2", r2, "--picks_dir", pdir, "--no_guard"]
    rc, out = sh(R2D / "select_ptdom2.py", "--after", "R1", *common)
    p1 = pdir / "picks_R_after1.json"
    pk1 = json.loads(p1.read_text()) if p1.exists() else {}
    T("1. select --after R1: rc 0, picks_R_after1.json written read-only, stage R1x", rc == 0 and p1.exists() and not os.access(p1, os.W_OK)
      and pk1.get("stage_written") == "R1x", out.strip().splitlines()[-1] if out.strip() else "")
    want_E = sorted([["A", "only-stars", "8e-3"], ["B2", "only-compass", "8e-3"], ["B3", "none", "1e-3"]])
    T("1. the default R1pre equals what this world's ptdom cells ask (check passed: rc 0); E = {only-stars@8e-3, only-compass@8e-3 "
      "(pick and argmax at 4e-3), B3 none@1e-3 (fixB: scalar_adj's pick 1e-3 below B3's 2e-3)}; R1pre not asked again",
      sorted(pk1.get("E", [])) == want_E and pk1.get("R1pre") == [list(x) for x in sorted(M2.R1PRE)], pk1.get("E"))
    rcg, outg = sh(B / "make_jobs_ptdom2.py", "--part", "R1pre", "--no-write", "--print")
    T("1. make_jobs_ptdom2 --part R1pre renders exactly R1PRE x seeds 0-3 (8 jobs)", rcg == 0 and gen_names(outg) ==
      sorted(M2.name2(b, a, lr, s) for b, a, lr in M2.R1PRE for s in M2.SEL))
    rc2, out2 = sh(R2D / "select_ptdom2.py", "--after", "R1", *common)
    T("1. a second --after R1 is refused (written once)", rc2 != 0 and "written once" in out2)
    rcv, outv = sh(R2D / "select_ptdom2.py", "--verify", p1, *common)
    T("1. --verify agrees", rcv == 0 and "agrees" in outv, outv.strip())
    rcg, outg = sh(B / "make_jobs_ptdom2.py", "--part", "R1x", "--picks", p1, "--no-write", "--print")
    want = sorted(M2.name2(b, a, lr, s) for b, a, lr in want_E for s in M2.SEL)
    T(f"1. make_jobs_ptdom2 --part R1x renders exactly E x seeds 0-3 ({len(want)} jobs)", rcg == 0 and gen_names(outg) == want,
      f"rc {rcg}; {outg.strip().splitlines()[-1][:160] if outg.strip() else ''}")
    # 2. R1x planted
    plant(r2, "ptdom2", "A", "only-stars", "8e-3", M2.SEL, prior)
    plant(r2, "ptdom2", "B2", "only-compass", "8e-3", M2.SEL, prior)
    plant(r2, "ptdom2", "B3", "none", "1e-3", M2.SEL, none_)
    rc, out = sh(R2D / "select_ptdom2.py", "--after", "R1x", *common)
    p2 = pdir / "picks_R_after2.json"
    pk2 = json.loads(p2.read_text()) if p2.exists() else {}
    T("2. select --after R1x: rc 0, picks_R_after2.json read-only, stage R2, every domain FINAL",
      rc == 0 and pk2.get("stage_written") == "R2" and all(v["status"] == "FINAL" for v in pk2.get("domains", {}).values()),
      {d: (v["status"], v["pick_prior"], v["pick_none"]) for d, v in pk2.get("domains", {}).items()})
    rcg, outg = sh(B / "make_jobs_ptdom2.py", "--part", "R2", "--picks", p2, "--no-write", "--print")
    s_cells = set(M2.cells("S"))   # A2 / A3: a cell part S already runs (here A none@4e-3) is not rendered again by R2
    want2 = sorted(M2.name2(b, a, lr, s) for b, a, lr in pk2.get("C", []) for s in M2.R2_SEEDS if (b, a, lr, s) not in s_cells)
    T(f"2. make_jobs_ptdom2 --part R2 renders exactly C x seeds 8-15 minus part S's cells ({len(want2)} jobs; C {pk2.get('C')})",
      rcg == 0 and gen_names(outg) == want2 and len(want2) == 8 * len(pk2.get("C", [])) - 8 * sum(1 for c in pk2.get("C", []) if (c[0], c[1], c[2], 8) in s_cells)
      and len(pk2.get("C", [])) >= 15,
      f"rc {rcg}; {outg.strip().splitlines()[-1][:200] if outg.strip() else ''}")
    rcv, outv = sh(R2D / "select_ptdom2.py", "--verify", p2, *common)
    T("2. --verify agrees on picks_R_after2.json", rcv == 0, outv.strip())
    bad = json.loads(p2.read_text())
    bad["C"][0][2] = "2e-3" if bad["C"][0][2] != "2e-3" else "1e-3"
    pb = pdir / "tampered.json"
    pb.write_text(json.dumps(bad)); pb.chmod(0o444)
    rcv, outv = sh(R2D / "select_ptdom2.py", "--verify", pb, *common)
    rcg, outg = sh(B / "make_jobs_ptdom2.py", "--part", "R2", "--picks", pb, "--no-write", "--print")
    T("2. a tampered picks file: --verify exits 1 and the generator refuses it (C != the FINAL domains' cells)", rcv == 1 and rcg != 0,
      f"verify rc {rcv}, generator rc {rcg}")
with tempfile.TemporaryDirectory() as tmp:
    r0, r2, pdir = Path(tmp) / "p0", Path(tmp) / "p2", Path(tmp) / "picks"
    pdir.mkdir()
    for d, b in M2.R_DOMS.items():
        for lr in M2.PRIOR_LRS0[d]:
            plant(r0, "ptdom", b, f"only-{d}", lr, M2.SEL, prior, share=0.3 if d == "ranks" else 0.9)
        plant(r2, "ptdom2", b, f"only-{d}", "4e-3", M2.SEL, prior, share=0.3 if d == "ranks" else 0.9)
    for b, lrs in M2.NONE_LRS0.items():
        for lr in lrs:
            plant(r0, "ptdom", b, "none", lr, M2.SEL, none_)
    for b, a_, lr in M2.R1PRE:
        plant(r2, "ptdom2", b, "none", lr, M2.SEL, none_)
    rc, out = sh(R2D / "select_ptdom2.py", "--after", "R1", "--root0", r0, "--root2", r2, "--picks_dir", pdir, "--no_guard")
    pk = json.loads((pdir / "picks_R_after1.json").read_text()) if (pdir / "picks_R_after1.json").exists() else {}
    T("3. A1.8: every ranks cell fails the gate -> ranks GATE (out of the family), the picks file written, rc 0, NO HOLD file anywhere; "
      "the other domains decided as usual",
      rc == 0 and pk.get("domains", {}).get("ranks", {}).get("status") == "GATE" and "ranks" not in pk.get("family", ["ranks"])
      and not list(Path(tmp).rglob("HOLD*")) and pk.get("domains", {}).get("compass", {}).get("status") == "PENDING", f"rc {rc}")
T("the real runs/ptdom2: no HOLD, no picks file, and its HOLD / picks / SUBMITTED_ files are exactly those before this test (A4: "
  "the parts' 120 SUBMITTED_ flags exist since 2026-11-09)", FLAGS() == FLAGS0 and not list(M2.R.glob("HOLD*")) and not list(M2.R.glob("picks_*")),
  f"{sum(n.startswith('SUBMITTED_') for n in FLAGS0)} SUBMITTED_ flags before and after")
print(f"PLUMBING {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
