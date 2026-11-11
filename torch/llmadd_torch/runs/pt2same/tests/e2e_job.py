"""pt2same test: the rendered job files themselves run by bash on CPU up to the end of step 1 (HOLD, shas, COMMIT, GPU guard,
data sha256, INTENT, --dry_inputs and its checks), then stopped before training. Each case runs a copy of a job file in
which the batch directory runs/pt2same is replaced by a scratch directory tests/e2e/<case>/R (its code/ = links to the
real snapshot), and everything after the line '## dry_inputs rows = the reference' replaced by an exit 0; nvidia-smi is a
fake that reports the GPU name given per case.
Cases: A / B positive (53M none s0, 190M win-ctx s4 on 'NVIDIA A100-SXM4-80GB') -> rc 0 and the DRY line;
       C 'NVIDIA A100-SXM4-40GB', D 'NVIDIA L40S' -> rc 3 before any input check; E HOLD -> rc 0 'HOLD present', nothing
       run; F the seed in ARGS changed 0 -> 1 -> rc 3 'dry_inputs differ'; G a changed trainer file -> rc 3 'sha mismatch'.
    python e2e_job.py"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2same as M   # noqa: E402

T = Path(__file__).resolve().parent / "e2e"
shutil.rmtree(T, ignore_errors=True)
T.mkdir()
FAKE = T / "bin"
FAKE.mkdir()
(FAKE / "nvidia-smi").write_text("""#!/bin/bash
case "$*" in
  *" -l "*) echo "2026/11/02 00:00:00.000, $FAKE_GPU, 0 %, 0 %, 0 MiB, 81920 MiB"; exec sleep 600;;
  *"--query-gpu=name --format=csv,noheader"*) echo "$FAKE_GPU";;
  *) echo "name, driver_version, memory.total [MiB]"; echo "$FAKE_GPU, fake, 81920 MiB";;
esac
""")
(FAKE / "nvidia-smi").chmod(0o755)
STOP = '"## dry_inputs rows = the reference'


def case(cid, job, gpu, hold=False, edit=None, bad_code=False):
    R2 = T / cid / "R"
    (R2 / "code").mkdir(parents=True)
    (R2 / "code" / "COMMIT").symlink_to(M.R / "code" / "COMMIT")
    if bad_code:
        (R2 / "code" / "tools").mkdir()
        for p in (M.R / "code" / "tools").iterdir():
            if p.name == "pretrain_range.py":
                (R2 / "code" / "tools" / p.name).write_text(p.read_text() + "# changed\n")
            else:
                (R2 / "code" / "tools" / p.name).symlink_to(p)
    else:
        (R2 / "code" / "tools").symlink_to(M.R / "code" / "tools")
    if hold:
        (R2 / "HOLD").touch()
    txt = (M.R / f"job_{job}.sbatch").read_text().replace(str(M.R), str(R2))
    lines = txt.splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("echo " + STOP))
    txt = "\n".join(lines[:k + 1]) + '\necho "## E2E: stopped before training"\nexit 0\n'
    if edit:
        txt = edit(txt)
    jf = T / cid / f"job_{job}.sh"
    jf.write_text(txt)
    env = {k_: v for k_, v in os.environ.items() if not k_.startswith("SLURM")}
    env.update(PATH=f"{FAKE}:{os.environ['PATH']}", FAKE_GPU=gpu, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    o = subprocess.run(["bash", str(jf)], capture_output=True, text=True, env=env, timeout=900)
    (T / cid / "log.txt").write_text(o.stdout + o.stderr)
    out = R2 / "out" / job
    return o.returncode, o.stdout + o.stderr, out


fails = 0


def check(cid, ok, msg):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {cid}: {msg}")


A100 = "NVIDIA A100-SXM4-80GB"
for cid, job in (("A", "pt2same-53m-lr8e-3-none-s0"), ("B", "pt2same-190m-lr6e-4-win-ctx-s4")):
    rc, log, out = case(cid, job, A100)
    dry = [l for l in log.splitlines() if l.startswith("## dry_inputs rows")]
    check(cid, rc == 0 and dry and "INTENT " + job in log and "## E2E: stopped before training" in log
          and (out / "dry_inputs.txt").exists() and (out / "node.txt").exists() and "train.bin: OK" in log,
          f"{job} on {A100}: rc {rc}; data sha256 OK; {dry}; {[l for l in log.splitlines() if l.startswith('INTENT')][0][:160]}...")
for cid, gpu in (("C", "NVIDIA A100-SXM4-40GB"), ("D", "NVIDIA L40S")):
    rc, log, out = case(cid, "pt2same-53m-lr8e-3-none-s0", gpu)
    check(cid, rc == 3 and "## not an A100-80GB" in log and not out.exists() and "train.bin" not in log,
          f"GPU '{gpu}': rc {rc}, '{[l for l in log.splitlines() if l.startswith('##')][-1]}', no run dir, no input check")
rc, log, out = case("E", "pt2same-53m-lr8e-3-none-s0", A100, hold=True)
check("E", rc == 0 and log.strip().endswith("HOLD present: exiting") and not out.exists(), f"HOLD: rc {rc}, '{log.strip()}'")
rc, log, out = case("F", "pt2same-53m-lr8e-3-none-s0", A100,
                    edit=lambda t: t.replace(" --seed 0 --tag ", " --seed 1 --tag ", 1))
check("F", rc == 3 and "## dry_inputs differ from the reference rows" in log and "## E2E" not in log,
      f"seed 1 rows in a seed-0 job: rc {rc}, '{[l for l in log.splitlines() if l.startswith('##')][-1]}'")
rc, log, out = case("G", "pt2same-190m-lr4.8e-3-none-s0", A100, bad_code=True)
check("G", rc == 3 and "## pretrain_range.py sha mismatch: exiting" in log and not out.exists(), f"changed trainer: rc {rc}, '{log.strip().splitlines()[-1]}'")
print(f"E2E JOB {'PASS' if not fails else 'FAIL'} ({7 - fails}/7)")
sys.exit(1 if fails else 0)
