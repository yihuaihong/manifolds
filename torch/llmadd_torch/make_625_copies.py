"""Second copy of every PENDING llmadd job under account <slurm-account> on the public L40S partition (user-approved
2026-10-14: the Tandon A100/H100 quota gives this user ~1 GPU at a time). Copies are job_g625_<original>.sbatch next to
the original, identical except for the account / partition lines. Duplicates are safe: the training scripts' run guard
(tools/run_guard.py) and the atlas's skip-if-done make the second copy to start exit at once.
--submit submits the copies not yet submitted (runs/<batch>/SUBMITTED_g625_<jobname>)."""
import glob, re, subprocess, sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
pending = set(subprocess.run(["squeue", "-u", "yh6210", "-h", "-t", "PD", "-o", "%j"], capture_output=True, text=True).stdout.split())
pending = {n for n in pending if n.startswith("llmadd_")}
by_name = {}
for f in glob.glob(str(B / "runs" / "*" / "job_*.sbatch")):
    if Path(f).name.startswith("job_g625_"):
        continue
    m = re.search(r"^#SBATCH --job-name=(\S+)$", Path(f).read_text(), re.M)
    if m and m.group(1) in pending:
        by_name.setdefault(m.group(1), f)   # atlas2 has a Tandon and an L40S file per name: either is fine
missing = sorted(pending - set(by_name))
assert not missing, f"no sbatch file found for {missing[:5]}"
n_sub = 0
for name, f in sorted(by_name.items()):
    src = Path(f)
    txt = src.read_text()
    txt = re.sub(r"^#SBATCH --account=.*$", "#SBATCH --account=<slurm-account>", txt, flags=re.M)
    txt = re.sub(r"^#SBATCH --partition=.*$", "#SBATCH --partition=l40s_public", txt, flags=re.M)
    assert txt.count("<slurm-account>") == 1 and txt.count("l40s_public") == 1, f
    dst = src.with_name("job_g625_" + src.name.removeprefix("job_l40s_").removeprefix("job_"))
    dst.write_text(txt)
    flag = src.parent / f"SUBMITTED_g625_{name}"
    if "--submit" in sys.argv and not flag.exists():
        o = subprocess.run(["sbatch", str(dst)], capture_output=True, text=True)
        if o.returncode == 0:
            flag.write_text(o.stdout); n_sub += 1
        else:
            print(name, o.stderr.strip())
print(f"{len(by_name)} pending job names, copies written; submitted now: {n_sub}")
