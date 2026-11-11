"""Known answer for the data stream (pt300x's dry_digest, on the CPU dry runs of scale_work/ptsz/refs_ptsz.py): for every
size x arm, seed 0, the CPU --dry_inputs output, run with the RENDERED stage-A ARGS of this batch (the arm's first grid lr;
only --out redirected; the ARGS recorded by refs_ptsz.py are asserted = the ARGS line of the rendered job file), must equal
line by line the --dry_inputs section of the pttpp 294M job log of the same arm and seed (L40S, the same data stream at fewer
tokens: --dry_inputs prints the held-out eval rows, the model line and the first training batch's 3 decoded rows, none of
which depends on --tokens / --eval_every / --lr): 294m including the model line; 136m / 391m / 731m except the model line,
which must be make_jobs_ptsz.MODEL_LINE of the size. Digest = sha256[:16] of the section ('line:' .. 'train row 2').
    python tests/dry_digest.py"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_ptsz as G   # noqa: E402

LOGS = Path("/scratch/yh6210/llmadd_torch/logs")
REFS_OUT = Path("/scratch/yh6210/llmadd_torch/scale_work/ptsz/refs_out")
PTTPP = {"none": "llmadd_tpp-tpp7-294m-t0.745-lr4.8e-3-none-s0_21032586.out",
         "win-ctx": "llmadd_tpp-tpp7-294m-t0.745-lr1.2e-3-win-ctx-s0_21032594.out"}


def section(lines):
    i = next(k for k, l in enumerate(lines) if l.startswith("line: "))
    j = next(k for k, l in enumerate(lines) if l.startswith("train row 2 "))
    return lines[i:j + 1]


def main():
    h = lambda L: hashlib.sha256(("\n".join(L) + "\n").encode()).hexdigest()[:16]
    ok = n = 0
    for size in G.BATCH:
        for arm in G.ARMS:
            n += 1
            lr = G.GRID[size][arm][0]
            name = f"{G.tag_of(size, arm, lr)}-s0"
            job = (G.P / f"job_{name}.sbatch").read_text()
            rec = json.loads((REFS_OUT / f"dry_{size}_{arm}_s0.json").read_text())
            args_job = re.search(r'^ARGS="(.*)"$', job, re.M).group(1)
            same_args = rec["job_args"] == args_job
            mine = section((REFS_OUT / f"dry_{size}_{arm}_s0.log").read_text().splitlines())
            ref = section((LOGS / PTTPP[arm]).read_text().splitlines())
            ml = [l for l in mine if l.startswith("model:")]
            if size == "294m":
                same = mine == ref
            else:
                same = len(mine) == len(ref) and all(a == b or (a.startswith("model:") and b.startswith("model:"))
                                                     for a, b in zip(mine, ref)) and ml == [G.MODEL_LINE[size]]
            good = same and same_args and rec["rc"] == 0
            ok += good
            nomodel = lambda L: [l for l in L if not l.startswith("model:")]
            print(f"{'SAME' if good else 'DIFF'} {name}: ARGS = the job file's {same_args}; {len(mine)} lines sha {h(mine)} vs pttpp "
                  f"{PTTPP[arm]} {len(ref)} lines sha {h(ref)}; without the model line {h(nomodel(mine))} vs {h(nomodel(ref))}; "
                  f"{ml[0] if ml else 'NO MODEL LINE'}")
    print(f"DRY DIGEST {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
