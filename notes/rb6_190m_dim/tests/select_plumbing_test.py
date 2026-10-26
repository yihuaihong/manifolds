"""Plumbing test of runs/rb6_190m_dim/select_job.sbatch: the generator's select_job_text() rendered for a fake batch dir R'
(scale_work/rb190dim_selplumb/<scenario>; its select_code = a copy of the real frozen select_code, its out/ = fake D1 runs
from ro_work/base, i.e. the real none@1.92e-2 runs relabelled as 3.84e-2) with the real B (the real rb6_190m / rb6_190m_ext
outs, the real generator and refs shas) and the generator call replaced by `echo GEN`; run with bash (no Slurm).
Scenarios: picked (status picked -> GEN --test-only then GEN --submit with R'/picks.json), incomplete (one D1 run missing
-> NOT PICKED, no GEN call), hold (HOLD -> exit 0 before anything), picks_exists (exit 3, picks.json untouched), tampered
(a select_code file changed -> exit 3, no selection). Also: the real select_job.sbatch = select_job_text() byte for byte.
    python select_plumbing_test.py"""
import shutil
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
import make_jobs_rb6_190m_dim as J   # noqa: E402

W = B / "scale_work" / "rb190dim_selplumb"
BASE = B / "scale_work" / "rb190dim_scripts" / "ro_work" / "base"
ok_all = True


def say(c, m):
    global ok_all
    ok_all &= bool(c)
    print(("PASS " if c else "FAIL ") + m, flush=True)


def fake(name, seeds=(0, 1, 2, 3)):
    r = W / name
    shutil.rmtree(r, ignore_errors=True)
    (r / "out").mkdir(parents=True)
    shutil.copytree(J.R / "select_code", r / "select_code", symlinks=True)
    for s in seeds:
        d = f"rb6tb-190m-2pct-k2-none-lr3.84e-2-s{s}"
        shutil.copytree(BASE / d, r / "out" / d)
    t = J.select_job_text(r=r, b=J.M.B, gen="echo GEN")
    (r / "job.sh").write_text(t)
    return r


def run(r):
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith(("SLURM", "SBATCH"))}
    o = subprocess.run(["bash", str(r / "job.sh")], capture_output=True, text=True, env=env)
    log = (r / "select.log").read_text() if (r / "select.log").exists() else ""
    return o.returncode, o.stdout + o.stderr, log


say((J.SELECT_JOB).read_text() == J.select_job_text(), "the real select_job.sbatch = select_job_text() (byte for byte)")
r = fake("picked")
rc, out, log = run(r)
gen = [l for l in log.splitlines() if l.startswith("GEN")]
say(rc == 0 and "selection status: picked" in log and gen == [f"GEN --stage D2 --picks {r}/picks.json --test-only",
                                                             f"GEN --stage D2 --picks {r}/picks.json --submit"],
    f"picked: status picked, then the D2 --test-only and --submit calls ({gen})")
say("PICK none_dim 1.92e-2" in log, "picked: the selection on the real 190M runs + the relabelled 3.84e-2 = 1.92e-2 (exact tie)")
r = fake("incomplete", seeds=(0, 1, 3))
rc, out, log = run(r)
say(rc == 0 and "selection status: incomplete" in log and "NOT PICKED" in log and "GEN" not in log,
    "incomplete: NOT PICKED, no generator call")
r = fake("hold")
(r / "HOLD").touch()
rc, out, log = run(r)
say(rc == 0 and "HOLD present" in log and "selection" not in log.replace("HOLD present: exiting", "") and
    not (r / "picks.json").exists(), "hold: exits at start, no selection")
r = fake("picks_exists")
(r / "picks.json").write_text("{}")
rc, out, log = run(r)
say(rc == 3 and "picks.json exists" in log and (r / "picks.json").read_text() == "{}" and "GEN" not in log,
    "picks_exists: exit 3, picks.json untouched")
r = fake("tampered")
p = r / "select_code" / "tools" / "readout_rules.py"
p.write_text(p.read_text() + "\n# tampered\n")
rc, out, log = run(r)
say(rc == 3 and "readout_rules.py differs from select_code/COMMIT" in log and not (r / "picks.json").exists(),
    "tampered: a changed select_code file -> exit 3, no selection")
print("SELECT PLUMBING TEST " + ("PASS" if ok_all else "FAIL"))
