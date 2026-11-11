"""pt6comb test: every job's input-check block (the job text from 'cat "$OUT/dry_inputs.txt"' to the DRY-rows echo: the model
line, the numbers target (helix3 table / none), per domain prior / shuf / lam, numbers first, the content scan, the site rule (3
comparison domains), the DRY rows = the CPU reference) executed with bash on REAL CPU --dry_inputs logs (tests/refs_B/dry_*.log of
this build; ptdom's / rb6-free logs for the negatives): its own cell's log -> exit 0; a log of another arm (numbers target or
domain priors), another seed, another batch -> exit 3. -> dryblock_test.out"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
sys.argv = [sys.argv[0]]
import make_jobs_pt6comb as M6   # noqa: E402
P = M6.load_P()
T_ = M6.R / "tests" / "refs_B"
R0 = M6.R0 / "out"
RF = M6.refs6("B")
res = []


def block(text):
    a = text.index('cat "$OUT/dry_inputs.txt"\n')
    b = text.index('echo "## dry_inputs rows = the reference')
    return text[a:b]


def job(part, arm, lr, s, smoke=False):
    return M6.render6(P, part, "B", arm, lr, s, RF[s], M6.COMMIT, 0, 0, None, smoke)[2]


def run(text, log):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "dry_inputs.txt").write_text(Path(log).read_text())
        o = subprocess.run(["bash", "-c", "set -euo pipefail\nOUT=" + tmp + "\n" + block(text)], capture_output=True, text=True)
    return o.returncode, (o.stdout + o.stderr).strip().splitlines()[-1:] if (o.stdout + o.stderr).strip() else []


cases = [  # (description, job text, dry log, expected rc)
    ("L1 none@2e-3 s0, its own log", job("L1", "none", "2e-3", 0), T_ / "dry_none_lr2e-3_s0.log", 0),
    ("L1 none@8e-3 s3 with the none@2e-3 s3 reference log (lr independent)", job("L1", "none", "8e-3", 3), T_ / "dry_none_lr2e-3_s3.log", 0),
    ("L1 all6@2e-3 s0, its own log", job("L1", "all6", "2e-3", 0), T_ / "dry_all6_lr2e-3_s0.log", 0),
    ("L1 all6@1e-3 s0 with all6@2e-3's s0 log (lr independent)", job("L1", "all6", "1e-3", 0), T_ / "dry_all6_lr2e-3_s0.log", 0),
    ("C all6@2e-3 s8, its own log", job("C", "all6", "2e-3", 8), T_ / "dry_all6_lr2e-3_s8.log", 0),
    ("C only-numbers@2e-3 s0, its own log", job("C", "only-numbers", "2e-3", 0), T_ / "dry_only-numbers_lr2e-3_s0.log", 0),
    ("C only-s4@2e-3 s0, its own log", job("C", "only-s4", "2e-3", 0), T_ / "dry_only-s4_lr2e-3_s0.log", 0),
    ("smoke body all6@2e-3 s0, the all6 seed-0 log", job("smoke", "all6", "2e-3", 0, True), T_ / "dry_all6_lr2e-3_s0.log", 0),
    ("smoke body none@2e-3 s0, the none seed-0 log", job("smoke", "none", "2e-3", 0, True), T_ / "dry_none_lr2e-3_s0.log", 0),
    ("all6@2e-3 s0 with none's s0 log (no helix3 table, no domain prior)", job("L1", "all6", "2e-3", 0), T_ / "dry_none_lr2e-3_s0.log", 3),
    ("none@2e-3 s0 with all6's s0 log (a numbers target)", job("L1", "none", "2e-3", 0), T_ / "dry_all6_lr2e-3_s0.log", 3),
    ("only-numbers@2e-3 s0 with all6's s0 log (other domain priors on)", job("C", "only-numbers", "2e-3", 0), T_ / "dry_all6_lr2e-3_s0.log", 3),
    ("only-s4@2e-3 s0 with only-numbers' s0 log (another prior)", job("C", "only-s4", "2e-3", 0), T_ / "dry_only-numbers_lr2e-3_s0.log", 3),
    ("all6@2e-3 s0 with only-numbers' s0 log (numbers target right, domain priors off)", job("L1", "all6", "2e-3", 0), T_ / "dry_only-numbers_lr2e-3_s0.log", 3),
    ("none@2e-3 s1 with seed 0's log (another seed)", job("L1", "none", "2e-3", 1), T_ / "dry_none_lr2e-3_s0.log", 3),
    ("C none@2e-3 s9 with seed 8's log (another seed)", job("C", "none", "2e-3", 9), T_ / "dry_none_lr2e-3_s8.log", 3),
    ("none@2e-3 s0 with ptdom B1's real GPU none@2e-3 s0 dry log (another batch)", job("L1", "none", "2e-3", 0), R0 / "ptdom-B1-none-lr2e-3-s0/dry_inputs.txt", 3),
    ("none@2e-3 s0 with ptdom A's real GPU none@2e-3 s0 dry log (another batch)", job("L1", "none", "2e-3", 0), R0 / "ptdom-A-none-lr2e-3-s0/dry_inputs.txt", 3),
]
for s in (1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15):   # every reference seed: none's own log
    cases.append((f"none@2e-3 s{s}, its own log", job("C" if s >= 8 else "L1", "none", "2e-3", s), T_ / f"dry_none_lr2e-3_s{s}.log", 0))
for desc, text, log, want in cases:
    rc, last = run(text, log)
    res.append(rc == want)
    print(f"{'PASS' if rc == want else 'FAIL'} {desc}: exit {rc} (want {want}) {last}")
print(f"DRYBLOCK {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
