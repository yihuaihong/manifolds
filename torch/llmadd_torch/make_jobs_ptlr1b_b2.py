"""Torch B200 jobs (2026-10-25): Amendment B2 of runs/ptlr_1b (runs/ptlr_1b/AUDIT.md "Amendment B2"). Stage A's selection
found none's bracket OPEN at the lower edge (2.4e-3); one step down = none@1.2e-3, seeds 0-1, account 520 only (user
2026-10-25: 「你觉得有必要的实验我觉得就可以跑」). After the B2 runs: stage B (seeds 2-3 at the picks) and, with APPROVE_C,
stage C (seeds 4-7), also 520 only.
NOT a copy: imports make_jobs_ptlr1b (READ-ONLY, sha256[:12] 8dd7b3407459, asserted) and reuses its snapshot() check and
job_text() / render_args() with every assert inside them. make_jobs_ptlr1b's main() runs only under __name__ ==
"__main__" (ast-asserted below: importing it renders and submits nothing). The only in-process change is none's lr
grid, widened by 1.2e-3 so that render_args accepts it. B2 jobs = job_text() with the job name replaced
(llmadd_ptlr1b-b2-<run>) and one line added at the end (maybe_select_b2.sh); stage B / C jobs = job_text() unchanged
(slot index as make_jobs_ptlr1b.cells() would give it). Submitted with --account=torch_pr_520_tandon_priority only; flags
SUBMITTED_<run>_520 (make_jobs_ptlr1b's naming); a submitted job file is never rewritten.
    python make_jobs_ptlr1b_b2.py --stage {b2,B,C} [--print] [--test-only] [--submit]"""
import ast
import hashlib
import json
import sys
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
FROZEN = {"make_jobs_ptlr1b.py": "8dd7b3407459", "runs/ptlr_1b/maybe_select.sh": "ba787f4ac54e",
          "runs/ptlr_1b/pick_1b.py": "10a2c3d8cc5f"}   # runs/ptlr_1b/AUDIT.md "Audit": frozen before stage A
for f_, v_ in FROZEN.items():
    assert sha(HERE / f_) == v_, (f_, sha(HERE / f_), v_, "a frozen ptlr_1b file changed")
_tree = ast.parse((HERE / "make_jobs_ptlr1b.py").read_text())
_calls = [n for n in _tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
_guard = [n for n in _tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
assert not _calls and len(_guard) == 1 and [ast.unparse(x) for x in _guard[0].body] == ["main()"], \
    "make_jobs_ptlr1b runs code on import"
sys.path.insert(0, str(HERE))
import make_jobs_ptlr1b as M   # noqa: E402

assert Path(M.__file__).resolve() == (HERE / "make_jobs_ptlr1b.py").resolve()
B2_LR = "1.2e-3"
assert M.GRID["none"] == ("2.4e-3", "4.8e-3", "9.6e-3") and B2_LR not in M.GRID["none"]
M.GRID = {**M.GRID, "none": (B2_LR,) + M.GRID["none"]}   # in-process only: render_args now accepts none@1.2e-3
ACCT = M.ACCOUNTS[0]
assert ACCT == "torch_pr_520_tandon_priority"
R = M.R
WIN_PICK = "6e-4"   # AUDIT B2 item 2: readout_A's win-ctx pick, unchanged by B2


def picks():
    p = json.loads((R / "picks_b2.json").read_text())
    assert p["status"] == "final" and p["none"] in M.GRID["none"] and p["win-ctx"] == WIN_PICK, p
    return p


def jobs(stage):
    """[(run name, job text)] of a stage"""
    out = []
    if stage == "b2":
        for s in (0, 1):
            jt, txt = M.job_text(0, B2_LR, "none", s)
            old = "#SBATCH --job-name=llmadd_ptlr1b-slot0"
            hook = f"bash {R}/maybe_select.sh || echo \"maybe_select.sh failed (rc $?): selection not run\""
            assert txt.count(old + "\n") == 1 and txt.count(hook + "\n") == 1 and txt.endswith(hook + "\n"), jt
            txt = txt.replace(old + "\n", f"#SBATCH --job-name=llmadd_ptlr1b-b2-{jt}\n")
            txt += (f"# Amendment B2: after both none@{B2_LR} runs, the B2 selection and stage B (once, lock)\n"
                    f"bash {R}/maybe_select_b2.sh || echo \"maybe_select_b2.sh failed (rc $?): B2 selection not run\"\n")
            out.append((jt, txt))
    else:
        p = picks()
        if stage == "B":
            cs = [(p[a], a, s) for s in (2, 3) for a in ("none", "win-ctx")]   # = make_jobs_ptlr1b.cells() order
            first = 12
        else:
            assert stage == "C" and (R / "APPROVE_C").exists(), "stage C needs runs/ptlr_1b/APPROVE_C"
            cs = [(p[a], a, s) for s in range(4, 8) for a in ("none", "win-ctx")]
            first = 16
        for i, (lr, a, s) in enumerate(cs, first):
            out.append(M.job_text(i, lr, a, s))
    return out


def main():
    stage = sys.argv[sys.argv.index("--stage") + 1]
    assert stage in ("b2", "B", "C"), stage
    assert "## Amendment B2" in (R / "AUDIT.md").read_text(), "Amendment B2 must be in runs/ptlr_1b/AUDIT.md first"
    M.snapshot()   # runs/ptlr_1b/code exists: verify only (sha, COMMIT, file-by-file = F87's snapshot)
    bad, rendered = 0, []
    for jt, txt in jobs(stage):
        job, flag = R / f"job_{jt}.sbatch", R / f"SUBMITTED_{jt}_520"
        if flag.exists() or (R / f"SUBMITTED_{jt}_625").exists():
            if job.read_text() != txt:
                print(f"WARNING: {job.name} was submitted and differs from this rendering: kept, not rewritten")
        else:
            job.write_text(txt)
        rendered.append((jt, job, flag))
        if "--print" in sys.argv:
            print(jt, [l for l in txt.splitlines() if l.startswith("ARGS=")][0])
    for jt, job, flag in rendered:
        if "--test-only" in sys.argv:
            o = subprocess.run(["sbatch", "--test-only", f"--account={ACCT}", str(job)], capture_output=True, text=True)
            lines = (o.stdout + o.stderr).strip().splitlines()
            line = next((x for x in lines if "to start at" in x), lines[-1] if lines else "")
            ok = o.returncode == 0 and "to start at" in line
            bad += not ok
            print(f"{'OK ' if ok else 'BAD'} {jt} 520: {line}")
        elif "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", f"--account={ACCT}", str(job)], capture_output=True, text=True)
            if o.returncode == 0:
                flag.write_text(o.stdout)
                print(jt, "520", o.stdout.strip())
            else:
                bad += 1
                print("FAILED", jt, "520", o.stderr.strip())
    print(f"ptlr_1b B2 stage {stage}: {len(rendered)} jobs rendered in {R}; account 520 only; "
          f"{'test-only' if '--test-only' in sys.argv else 'submitted' if '--submit' in sys.argv else 'not submitted'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
