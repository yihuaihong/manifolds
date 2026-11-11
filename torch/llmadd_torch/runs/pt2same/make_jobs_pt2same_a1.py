"""pt2same Amendment A1 item 3: the descriptive 53M none lr sensitivity on Torch (lr 4e-3 and 1.6e-2, seeds 0-3).
Uses make_jobs_pt2same.render() unchanged; only PICKS['53m']['none'] is overridden per lr, so every in-job check and the
Burst-twin assertions of render() apply to these cells too (Burst ran 53M none at both lrs: pt2b). Writes job files to
runs/pt2same/a1/ and a1/JOBS.sha256; submits nothing (a1/submit_a1.sh --go does)."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2same as M  # noqa: E402

A1 = M.R / "a1"
A1.mkdir(exist_ok=True)
M.snapshot()
G2 = M.g2()
base = dict(M.PICKS["53m"])


def burst_twins(lr):
    """M.burst_tasks() without its 32-cell completeness assert (the A1 lrs exist on Burst for seeds 0-3 only): exactly
    one Burst task per (53m, none, lr, seed), seeds 0-3 required"""
    out = {}
    for name, tag, text, args, block, arm, lr_, seed in G2.tasks():
        if block == M.BURST_BLOCK["53m"] and arm == "none" and lr_ == lr:
            assert ("53m", "none", lr, seed) not in out, name
            out[("53m", "none", lr, seed)] = name
    assert {k[3] for k in out} >= {0, 1, 2, 3}, (lr, sorted(out))
    return out


out = []
for lr in ("4e-3", "1.6e-2"):
    M.PICKS["53m"]["none"] = lr
    bt = burst_twins(lr)
    for s in (0, 1, 2, 3):
        name, tag, text, args, I, bname = M.render(G2, "53m", "none", s, bt)
        assert f"--lr {lr} " in args and f"lr{lr}" in name, (name, args)
        job = A1 / f"job_{name}.sbatch"
        if (A1 / f"SUBMITTED_{name}").exists():
            assert job.read_text() == text, f"{job.name} was submitted and differs"
        else:
            job.write_text(text)
        out.append((name, job, bname))
M.PICKS["53m"] = base
assert len(out) == 8 and len({n for n, _, _ in out}) == 8
(A1 / "JOBS.sha256").write_text("".join(f"{hashlib.sha256(j.read_bytes()).hexdigest()}  {j.name}\n" for _, j, _ in out))
for name, _, bname in out:
    print(f"{name} (Burst twin {bname})")
print(f"pt2same A1: {len(out)} jobs rendered in {A1}; nothing submitted")
