"""ptdom2 test: every job's input-check block (the job text from 'cat "$OUT/dry_inputs.txt"' to the DRY-rows echo: the model
line, the numbers target, per domain prior / shuf / lam, numbers first, the content scan, the site rule, the DRY rows = the CPU
reference) executed with bash on REAL CPU --dry_inputs logs (tests/refs_<b>/dry_*.log of this build; ptdom's job dirs for a
true-target arm): its own cell's log -> exit 0; a log of the other arm (true <-> shuffled) or of another seed -> exit 3.
-> dryblock_test.out"""
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
sys.argv = [sys.argv[0]]
import make_jobs_ptdom2 as M2   # noqa: E402
MJ = M2.MJ
P = M2.load_P()
T_ = M2.R / "tests"
R0 = M2.R0 / "out"
res = []


def block(text):
    a = text.index('cat "$OUT/dry_inputs.txt"\n')
    b = text.index('echo "## dry_inputs rows = the reference')
    return text[a:b]


def job(part, b, arm, lr, s):
    return M2.render2(P, part, b, arm, lr, s, M2.refs2(b)[s], M2.COMMIT, 0, 0, part == "pilot")[2]


def twin(part, b, arm, lr, s):
    return M2.twin_render(P, part, b, arm, lr, s, MJ.refs(b)[s], M2.COMMIT, MJ.SLOT0[b])[2]


def run(text, log):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "dry_inputs.txt").write_text(Path(log).read_text())
        o = subprocess.run(["bash", "-c", "set -euo pipefail\nOUT=" + tmp + "\n" + block(text)], capture_output=True, text=True)
    return o.returncode, (o.stdout + o.stderr).strip().splitlines()[-1:] if (o.stdout + o.stderr).strip() else []


cases = [  # (description, job text, dry log, expected rc)
    ("S0 only-countries-shuf s0, its own log", job("S0", "B1", "only-countries-shuf", "2e-3", 0), T_ / "refs_B1/dry_only-countries-shuf_lr2e-3_s0.log", 0),
    ("S0 only-people-shuf s0, its own log", job("S0", "B1", "only-people-shuf", "2e-3", 0), T_ / "refs_B1/dry_only-people-shuf_lr2e-3_s0.log", 0),
    ("S0 only-months-shuf s0, its own log", job("S0", "A", "only-months-shuf", "2e-3", 0), T_ / "refs_A/dry_only-months-shuf_lr2e-3_s0.log", 0),
    ("S0 only-cities-shuf s0, its own log", job("S0", "A", "only-cities-shuf", "2e-3", 0), T_ / "refs_A/dry_only-cities-shuf_lr2e-3_s0.log", 0),
    ("S0 only-countries-shuf s0 with the TRUE arm's real GPU dry log (ptdom only-countries@2e-3 s0)", job("S0", "B1", "only-countries-shuf", "2e-3", 0),
     R0 / "ptdom-B1-only-countries-lr2e-3-s0/dry_inputs.txt", 3),
    ("ptdom's TRUE-arm job (twin) with the shuffled arm's dry log", twin("S0", "B1", "only-countries-shuf", "2e-3", 0),
     T_ / "refs_B1/dry_only-countries-shuf_lr2e-3_s0.log", 3),
    ("S0 only-countries-shuf s1 with seed 0's log (another seed)", job("S0", "B1", "only-countries-shuf", "2e-3", 1),
     T_ / "refs_B1/dry_only-countries-shuf_lr2e-3_s0.log", 3),
    ("A2 S only-countries-shuf s8, its own log", job("S", "B1", "only-countries-shuf", "2e-3", 8), T_ / "refs_B1/dry_only-countries-shuf_lr2e-3_s8.log", 0),
    ("A2 S only-people-shuf s8, its own log", job("S", "B1", "only-people-shuf", "2e-3", 8), T_ / "refs_B1/dry_only-people-shuf_lr2e-3_s8.log", 0),
    ("A2 S none@2e-3 s8 (B1) with the B1 none@4e-3 s8 reference log (lr independent)", job("S", "B1", "none", "2e-3", 8), T_ / "refs_B1/dry_none_lr4e-3_s8.log", 0),
    ("A2 S only-countries@2e-3 s9 with the B1 none s9 log (a true prior arm vs a none log)", job("S", "B1", "only-countries", "2e-3", 9), T_ / "refs_B1/dry_none_lr4e-3_s9.log", 3),
    ("A2 S only-cities@2e-3 s8 (A) with A's only-stars-shuf s8 log (another arm)", job("S", "A", "only-cities", "2e-3", 8), T_ / "refs_A/dry_only-stars-shuf_lr4e-3_s8.log", 3),
    ("A2 S none@2e-3 s12 (A) with A's none@4e-3 s12 reference log", job("S", "A", "none", "2e-3", 12), T_ / "refs_A/dry_none_lr4e-3_s12.log", 0),
    ("A3 S none@4e-3 s9 (A, F100's none pick for cities / months) with A's none@4e-3 s9 reference log", job("S", "A", "none", "4e-3", 9), T_ / "refs_A/dry_none_lr4e-3_s9.log", 0),
    ("A3 S none@4e-3 s9 (A) with B1's none s9 log (another batch)", job("S", "A", "none", "4e-3", 9), T_ / "refs_B1/dry_none_lr4e-3_s9.log", 3),
    ("R2 none@4e-3 s8 (B2), its own log", job("R2", "B2", "none", "4e-3", 8), T_ / "refs_B2/dry_none_lr4e-3_s8.log", 0),
    ("R2 only-compass-shuf@4e-3 s8, its own log", job("R2", "B2", "only-compass-shuf", "4e-3", 8), T_ / "refs_B2/dry_only-compass-shuf_lr4e-3_s8.log", 0),
    ("R2 only-ranks-shuf@4e-3 s8, its own log", job("R2", "B2", "only-ranks-shuf", "4e-3", 8), T_ / "refs_B2/dry_only-ranks-shuf_lr4e-3_s8.log", 0),
    ("R2 only-prefixes-shuf@4e-3 s8, its own log", job("R2", "B2", "only-prefixes-shuf", "4e-3", 8), T_ / "refs_B2/dry_only-prefixes-shuf_lr4e-3_s8.log", 0),
    ("R2 only-stars-shuf@4e-3 s8, its own log", job("R2", "A", "only-stars-shuf", "4e-3", 8), T_ / "refs_A/dry_only-stars-shuf_lr4e-3_s8.log", 0),
    ("R2 only-landmarks-shuf@4e-3 s8, its own log", job("R2", "A", "only-landmarks-shuf", "4e-3", 8), T_ / "refs_A/dry_only-landmarks-shuf_lr4e-3_s8.log", 0),
    ("R2 only-scalar_adj-shuf@4e-3 s8, its own log", job("R2", "B3", "only-scalar_adj-shuf", "4e-3", 8), T_ / "refs_B3/dry_only-scalar_adj-shuf_lr4e-3_s8.log", 0),
    ("R2 only-compass-shuf@4e-3 s8 with none's s8 log (not shuffled, no prior)", job("R2", "B2", "only-compass-shuf", "4e-3", 8), T_ / "refs_B2/dry_none_lr4e-3_s8.log", 3),
    ("R2 only-compass-shuf@4e-3 s9 with s8's log (another seed)", job("R2", "B2", "only-compass-shuf", "4e-3", 9), T_ / "refs_B2/dry_only-compass-shuf_lr4e-3_s8.log", 3),
    ("R2 none@4e-3 s8 (B3) with B2's s8 log (another batch)", job("R2", "B3", "none", "4e-3", 8), T_ / "refs_B2/dry_none_lr4e-3_s8.log", 3),
    ("R1pre A none@1.6e-2 s1 with ptdom's real GPU dry log of none@8e-3 s1 (arm / lr independent)", job("R1pre", "A", "none", "1.6e-2", 1),
     R0 / "ptdom-A-none-lr8e-3-s1/dry_inputs.txt", 0),
    ("R1pre B2 none@5e-4 s2 with ptdom's real GPU dry log of none@1e-3 s2", job("R1pre", "B2", "none", "5e-4", 2), R0 / "ptdom-B2-none-lr1e-3-s2/dry_inputs.txt", 0),
    ("R1pre B2 none@5e-4 s2 with ptdom's only-compass@1e-3 s2 log (a prior arm)", job("R1pre", "B2", "none", "5e-4", 2), R0 / "ptdom-B2-only-compass-lr1e-3-s2/dry_inputs.txt", 3),
    ("pilot body only-compass-shuf@2e-3 s0, the shuffled arm's seed-0 log", job("pilot", "B2", "only-compass-shuf", "2e-3", 0), T_ / "refs_B2/dry_only-compass-shuf_lr2e-3_s0.log", 0),
    ("pilot body only-countries-shuf@2e-3 s0, the shuffled arm's seed-0 log", job("pilot", "B1", "only-countries-shuf", "2e-3", 0), T_ / "refs_B1/dry_only-countries-shuf_lr2e-3_s0.log", 0),
]
for desc, text, log, want in cases:
    rc, last = run(text, log)
    res.append(rc == want)
    print(f"{'PASS' if rc == want else 'FAIL'} {desc}: exit {rc} (want {want}) {last}")
print(f"DRYBLOCK {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
