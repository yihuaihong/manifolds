"""pt2sz test (CPU): the rendered job files themselves, run by bash (pt2same's tests/e2e_job.py method), plus the part that
is new against pt2same: the check's exit code and the stage A / X model removal.
Head (each case runs a copy of a job file in which runs/pt2sz is replaced by a scratch dir tests/e2e/<case>/pt2sz whose code/ =
links to the real snapshot, cut after the line '## dry_inputs rows = the reference'; nvidia-smi is a fake reporting the
GPU name of the case):
  A / B positive (93M none s0, 294M win-ctx s1 on 'NVIDIA A100-SXM4-80GB') -> rc 0, data sha256 OK, INTENT, DRY = reference;
  C 'NVIDIA A100-SXM4-40GB', D 'NVIDIA L40S' -> rc 3 before any input check; E HOLD -> rc 0 'HOLD present', nothing run;
  F the seed in ARGS changed 0 -> 1 -> rc 3 'dry_inputs differ'; G a changed trainer file -> rc 3 'sha mismatch'.
Tail (the job's own lines from '## 4. intent checks' to the end, after its R / OUT / TAG line, on a scratch $OUT holding a
synthetic finished run JSON with the job's intent, its offline_lm.json entry and a model file):
  T1 stage A, a passing run -> rc 0, check.json pass, the model file REMOVED;
  T2 stage A, a run whose lr differs from the intent -> rc 4, check.json fail, the model file KEPT;
  T3 stage C (seed 4), a passing run -> rc 0, the model file KEPT.
    python e2e_job.py"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2sz as M   # noqa: E402

sys.path.insert(0, str(M.R))
import select_pt2sz as S   # noqa: E402  (fake_run: the synthetic run JSON + offline_lm.json)

G2 = S.G2
T = Path(__file__).resolve().parent / "e2e"
shutil.rmtree(T, ignore_errors=True)
T.mkdir()
FAKE = T / "bin"
FAKE.mkdir()
(FAKE / "nvidia-smi").write_text("""#!/bin/bash
case "$*" in
  *" -l "*) echo "2026/11/05 00:00:00.000, $FAKE_GPU, 0 %, 0 %, 0 MiB, 81920 MiB"; exec sleep 600;;
  *"--query-gpu=name --format=csv,noheader"*) echo "$FAKE_GPU";;
  *) echo "name, driver_version, memory.total [MiB]"; echo "$FAKE_GPU, fake, 81920 MiB";;
esac
""")
(FAKE / "nvidia-smi").chmod(0o755)
STOP = '"## dry_inputs rows = the reference'
ENV = {k_: v for k_, v in os.environ.items() if not k_.startswith("SLURM")}
ENV.update(PATH=f"{FAKE}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")


def scratch_R(cid, bad_code=False, hold=False):
    R2 = T / cid / "pt2sz"   # named like the batch dir: make_jobs_pt2sz.run_paths(root=R2.parent) finds the scratch runs
    (R2 / "code").mkdir(parents=True)
    (R2 / "code" / "COMMIT").symlink_to(M.R / "code" / "COMMIT")
    if bad_code:
        (R2 / "code" / "tools").mkdir()
        for p in (M.R / "code" / "tools").iterdir():
            if p.name == "pretrain_range.py":
                (R2 / "code" / "tools" / p.name).write_text(p.read_text() + "# changed\n")
            elif p.name != "__pycache__":
                (R2 / "code" / "tools" / p.name).symlink_to(p)
    else:
        (R2 / "code" / "tools").symlink_to(M.R / "code" / "tools")
    if hold:
        (R2 / "HOLD").touch()
    return R2


def head_case(cid, job, gpu, hold=False, edit=None, bad_code=False):
    R2 = scratch_R(cid, bad_code, hold)
    txt = (M.R / f"job_{job}.sbatch").read_text().replace(str(M.R), str(R2))
    lines = txt.splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("echo " + STOP))
    txt = "\n".join(lines[:k + 1]) + '\necho "## E2E: stopped before training"\nexit 0\n'
    if edit:
        txt = edit(txt)
    jf = T / cid / f"job_{job}.sh"
    jf.write_text(txt)
    o = subprocess.run(["bash", str(jf)], capture_output=True, text=True, env=dict(ENV, FAKE_GPU=gpu), timeout=900)
    (T / cid / "log.txt").write_text(o.stdout + o.stderr)
    return o.returncode, o.stdout + o.stderr, R2 / "out" / job


def tail_case(cid, F, mutate=None):
    """the job's own tail (from '## 4. intent checks') on a scratch $OUT with a synthetic finished run"""
    R2 = scratch_R(cid)
    text, args, I = M.render_one(G2, F)
    txt = text.replace(str(M.R), str(R2))
    lines = txt.splitlines()
    rline = next(l for l in lines if l.startswith("R="))
    k = next(i for i, l in enumerate(lines) if l.startswith('echo "## 4. intent checks'))
    script = "#!/bin/bash\nset -euo pipefail\n" + rline + "\n" + "\n".join(lines[k:]) + "\n"
    jf = T / cid / f"tail_{F['name']}.sh"
    jf.write_text(script)
    out = R2 / "out" / F["name"]
    run, chk, model = M.run_paths(F["size"], F["arm"], F["lr"], F["seed"], R2.parent)
    assert run.parent.parent == out, (run, out)
    S.fake_run(run, chk, I, 60.0, check=False, mutate=mutate)
    model.write_bytes(b"fake model")
    o = subprocess.run(["bash", str(jf)], capture_output=True, text=True, env=ENV, timeout=600)
    (T / cid / "log.txt").write_text(o.stdout + o.stderr)
    ck = json.loads(chk.read_text()) if chk.exists() else {}
    return o.returncode, o.stdout + o.stderr, ck.get("pass"), model.exists()


fails = 0


def check(cid, ok, msg):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {cid}: {msg}")


A100 = "NVIDIA A100-SXM4-80GB"
for cid, job in (("A", "p2z-93m-lr2.4e-3-none-s0"), ("B", "p2z-294m-lr6e-4-win-ctx-s1")):
    rc, log, out = head_case(cid, job, A100)
    dry = [l for l in log.splitlines() if l.startswith("## dry_inputs rows")]
    check(cid, rc == 0 and dry and "INTENT " + job in log and "## E2E: stopped before training" in log
          and (out / "dry_inputs.txt").exists() and (out / "node.txt").exists() and "train.bin: OK" in log,
          f"{job} on {A100}: rc {rc}; data sha256 OK; {dry}; {[l for l in log.splitlines() if l.startswith('INTENT')][0][:150]}...")
for cid, gpu in (("C", "NVIDIA A100-SXM4-40GB"), ("D", "NVIDIA L40S")):
    rc, log, out = head_case(cid, "p2z-93m-lr2.4e-3-none-s0", gpu)
    check(cid, rc == 3 and "## not an A100-80GB" in log and not out.exists() and "train.bin" not in log,
          f"GPU '{gpu}': rc {rc}, '{[l for l in log.splitlines() if l.startswith('##')][-1]}', no run dir, no input check")
rc, log, out = head_case("E", "p2z-93m-lr2.4e-3-none-s0", A100, hold=True)
check("E", rc == 0 and log.strip().endswith("HOLD present: exiting") and not out.exists(), f"HOLD: rc {rc}, '{log.strip()}'")
rc, log, out = head_case("F", "p2z-93m-lr2.4e-3-none-s0", A100, edit=lambda t: t.replace(" --seed 0 --tag ", " --seed 1 --tag ", 1))
check("F", rc == 3 and "## dry_inputs differ from the reference rows" in log and "## E2E" not in log,
      f"seed 1 rows in a seed-0 job: rc {rc}, '{[l for l in log.splitlines() if l.startswith('##')][-1]}'")
rc, log, out = head_case("G", "p2z-294m-lr4.8e-3-none-s0", A100, bad_code=True)
check("G", rc == 3 and "## pretrain_range.py sha mismatch: exiting" in log and not out.exists(), f"changed trainer: rc {rc}, '{log.strip().splitlines()[-1]}'")
FA = M.facts("93m", "win-ctx", "6e-4", 2, "A", G2)
rc, log, ok, kept = tail_case("T1", FA)
check("T1", rc == 0 and ok is True and not kept and "## model removed (stage A)" in log and "(check rc 0)" in log,
      f"stage A, passing: rc {rc}, check pass {ok}, model kept {kept}")
rc, log, ok, kept = tail_case("T2", FA, mutate=lambda r: r["args"].update(lr=1.2e-3))
check("T2", rc == 4 and ok is False and kept and "## model removed" not in log and "(check rc 4)" in log,
      f"stage A, lr differs from the intent: rc {rc}, check pass {ok}, model kept {kept}")
FC = M.facts("294m", "none", "9.6e-3", 4, "C", G2)
rc, log, ok, kept = tail_case("T3", FC)
check("T3", rc == 0 and ok is True and kept and "## model removed" not in log, f"stage C, passing: rc {rc}, check pass {ok}, model kept {kept}")
print(f"E2E JOB {'PASS' if not fails else 'FAIL'} ({10 - fails}/10)")
sys.exit(1 if fails else 0)
