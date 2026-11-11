"""Job-file tests of ptdom (after tests/refs_ptdom.py wrote runs/ptdom/ptdom_refs.json):
  1. render stage A (92 job files after Amendment A1, written into runs/ptdom: the deliverable) and check each: bash -n; the embedded intent =
     make_jobs_ptdom.intent(...) for its cell; the CHECK_PY heredoc = make_jobs_ptdom.CHECK_PY byte for byte and compiles;
     ARGS = args_of(...) and (A1) the CHECK_PY call receives "$ARGS"; the SBATCH header (account 625, l40s_public, gpu48,
     nice 20, 03:00:00, requeue); the L40S guard; no stale job file of a cell outside stage A
  2. the input-check block (from '## 1. inputs' to '## 2. training') executed with bash on the CPU dry logs (the dry run
     command replaced by a copy of the log): the job's own (arm, seed) log -> exit 0; another seed's log -> exit 3 (DRY
     rows); another arm's log of the same seed -> exit 3 (the domains: line)
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "ptdom"
sys.path.insert(0, str(B)); sys.path.insert(0, str(R / "code" / "tools"))
import make_jobs_ptdom as M   # noqa: E402
import make_rb6_tasks as P    # noqa: E402

REFD = R / "tests" / "refs"
ok = []


def T(name, cond, val=None):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val is not None else ""))


o = subprocess.run([M.PY, str(B / "make_jobs_ptdom.py"), "--stage", "A"], capture_output=True, text=True, cwd=B)
T("generator --stage A renders", o.returncode == 0 and "92 jobs rendered" in o.stdout, o.stdout.strip()[-200:] + o.stderr[-300:])
RF = M.refs()
commit = (R / "code" / "COMMIT").read_text().split()[0]
jobs = M.cells("A")
n_ok = {"bash": 0, "intent": 0, "check": 0, "args": 0, "header": 0}
for a, lr, s in jobs:
    name = f"{M.tag_of(a, lr)}-s{s}"
    f = R / f"job_{name}.sbatch"
    txt = f.read_text()
    n_ok["bash"] += subprocess.run(["bash", "-n", str(f)]).returncode == 0
    m = re.search(r"\"\$OUT/check.json\" '(\{.*\})' \"\$ARGS\" <<'PY'", txt)
    I = json.loads(m.group(1)) if m else None
    n_ok["intent"] += I == M.intent(P, a, lr, s, RF[s])
    cp = txt.split("<<'PY' | tee \"$OUT/check.txt\"\n", 1)[1].split("PY\necho \"## done", 1)[0]
    n_ok["check"] += cp == M.CHECK_PY and compile(cp, "check", "exec") is not None
    n_ok["args"] += f'ARGS="{M.args_of(a, lr, s, R / "out" / name)}"' in txt and txt.count("'" + json.dumps(I, separators=(",", ":")) + "' \"$ARGS\" <<'PY'") == 1
    hd = ["--account=<slurm-account>", "--partition=l40s_public", "--qos=gpu48", "--nice=20", "--time=03:00:00",
          "--requeue", "--gres=gpu:1"]
    n_ok["header"] += all(f"#SBATCH {h}" in txt for h in hd) and "*L40S*) ;;" in txt and 'if [ -e "$R/HOLD" ]' in txt
T(f"92 stage-A job files: bash -n / intent / CHECK_PY / ARGS (+ passed to CHECK_PY) / header+guards", all(v == 92 for v in n_ok.values()), n_ok)
T("stage A cells (A1): none x 5 lrs (1e-3 .. 1.6e-2) + 6 prior arms x 3 lrs (5e-4 .. 2e-3), x seeds 0-3, seed-major",
  len(jobs) == 92 and [s for _, _, s in jobs] == sorted(s for _, _, s in jobs) and len({(a, lr) for a, lr, _ in jobs}) == 23
  and sorted({lr for a, lr, _ in jobs if a == "none"}, key=float) == ["1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2"])
want_files = {f"job_{M.tag_of(a, lr)}-s{s}.sbatch" for a, lr, s in jobs}
have_files = {p.name for p in R.glob("job_*.sbatch")}
T("the job files in runs/ptdom are exactly the 92 stage-A cells (no stale file)", have_files == want_files, sorted(have_files ^ want_files)[:5])


def block(job_txt, dry_log):
    """The job's input-check block with the dry run replaced by a copy of dry_log (bash script text)."""
    b = job_txt.split('echo "## 1. inputs', 1)[1].split('echo "## 2. training', 1)[0]
    b = 'echo "## 1. inputs' + b
    b = re.sub(r'"\$PY" pretrain_range.py \$ARGS --dry_inputs > "\$OUT/dry_inputs.txt" 2>&1 \|\| \{ cat "\$OUT/dry_inputs.txt"; exit 3; \}',
               f'cp {dry_log} "$OUT/dry_inputs.txt"', b)
    assert f"cp {dry_log}" in b
    return b


def run_block(job, dry_log):
    txt = job.read_text()
    head = txt.split('echo "## 1. inputs', 1)[0]
    R_, OUT_ = re.search(r"^R=(\S+); OUT=(\S+);", head, re.M).groups()
    with tempfile.TemporaryDirectory() as tmp:
        sh = Path(tmp) / "blk.sh"
        sh.write_text("set -euo pipefail\n" + re.search(r"^R=.*$", head, re.M).group(0).replace(f"OUT={OUT_}", f"OUT={tmp}/out")
                      + f"\nmkdir -p {tmp}/out\n" + block(txt, dry_log))
        return subprocess.run(["bash", str(sh)], capture_output=True, text=True)


logs = {(p.stem.split("_lr")[0][4:], p.stem.split("_lr")[1].split("_s")[0], int(p.stem.split("_s")[-1])): p
        for p in REFD.glob("dry_*.log")}
own = cross_seed = cross_arm = 0
tried = {"own": [], "seed": [], "arm": []}
for (a, lr, s), lg in sorted(logs.items()):
    job = R / f"job_{M.tag_of(a, lr)}-s{s}.sbatch"
    if not job.exists():   # e.g. only-months@5e-4 seed 0 (rendered in stage A)
        continue
    rc = run_block(job, lg).returncode
    own += rc == 0; tried["own"].append((a, lr, s, rc))
    oth = logs.get((a, lr, (s + 1) % 8))
    if oth:
        rc2 = run_block(job, oth).returncode
        cross_seed += rc2 == 3; tried["seed"].append((a, lr, s, rc2))
    for (a2, lr2, s2), lg2 in logs.items():
        if s2 == s and a2 != a:
            rc3 = run_block(job, lg2).returncode
            cross_arm += rc3 == 3; tried["arm"].append((a, a2, s, rc3))
T(f"input-check block on the CPU dry logs: own cells exit 0 ({own}/{len(tried['own'])}), another seed exit 3 "
  f"({cross_seed}/{len(tried['seed'])}), another arm exit 3 ({cross_arm}/{len(tried['arm'])})",
  own == len(tried["own"]) >= 6 and cross_seed == len(tried["seed"]) >= 3 and cross_arm == len(tried["arm"]) >= 4, tried)
print("JOBCHECK", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)})")
