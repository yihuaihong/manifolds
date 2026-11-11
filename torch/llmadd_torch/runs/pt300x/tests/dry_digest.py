"""Known answer for the data stream: the CPU --dry_inputs output of seed 0 of each new size (both arms), run with the
RENDERED stage-A ARGS of this batch (only --out redirected to a scratch dir), from runs/pt300x/code7, must equal line by line
the --dry_inputs section of the pttpp job log of the same size, arm and seed (pttpp 7% L40S, the same shape at fewer
tokens: --dry_inputs prints the held-out eval rows, the model line and the first training batch's 3 decoded rows, none of
which depends on --tokens / --eval_every / --lr). Digest = sha256 of the section (first 'line:' .. 'train row 2').
    python tests/dry_digest.py OUTDIR     (CPU; ~1-3 min per run)"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt300x as G   # noqa: E402

LOGS = Path("/scratch/yh6210/llmadd_torch/logs")
PTTPP = {("93m", "none"): "llmadd_tpp-tpp7-93m-t0.745-lr9.6e-3-none-s0_21032456.out",
         ("93m", "win-ctx"): "llmadd_tpp-tpp7-93m-t0.745-lr2.4e-3-win-ctx-s0_21032550.out",
         ("294m", "none"): "llmadd_tpp-tpp7-294m-t0.745-lr4.8e-3-none-s0_21032586.out",
         ("294m", "win-ctx"): "llmadd_tpp-tpp7-294m-t0.745-lr1.2e-3-win-ctx-s0_21032594.out"}


def section(lines):
    i = next(k for k, l in enumerate(lines) if l.startswith("line: "))
    j = next(k for k, l in enumerate(lines) if l.startswith("train row 2 "))
    return lines[i:j + 1]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for (size, arm), logname in PTTPP.items():
        name, txt, F = next(x for x in G.render(size) if x[2]["arm"] == arm and x[2]["seed"] == 0 and x[2]["stage"] == "A")
        args = re.search(r'^ARGS="(.*)"$', txt, re.M).group(1)
        scratch = out / f"dryout_{name}"
        args_cpu = args.replace(f"--out {G.P / 'out' / F['tag']} ", f"--out {scratch} ")
        assert args_cpu != args and str(G.P / "out") not in args_cpu
        env = dict(os.environ, HF_HOME="/scratch/yh6210/llmadd_torch/hf_cache", PYTHONPATH="/scratch/yh6210/llmadd_torch/pylib",
                   CUDA_VISIBLE_DEVICES="")
        o = subprocess.run([G.PY, "pretrain_range.py", *args_cpu.split(), "--dry_inputs"], cwd=G.P / "code7" / "tools",
                           env=env, capture_output=True, text=True)
        (out / f"dry_{name}.txt").write_text(o.stdout + o.stderr)
        mine = section(o.stdout.splitlines())
        ref = section((LOGS / logname).read_text().splitlines())
        h = lambda L: hashlib.sha256(("\n".join(L) + "\n").encode()).hexdigest()[:16]
        same = mine == ref
        ok += same
        diff = [(a, b) for a, b in zip(mine, ref) if a != b]
        print(f"{'SAME' if same else 'DIFF'} {name} (rc {o.returncode}): {len(mine)} lines, sha {h(mine)} vs pttpp {logname} "
              f"{len(ref)} lines, sha {h(ref)}" + (f"; first difference {diff[:1]}" if diff else ""))
        print("   model line:", next(l for l in mine if l.startswith("model:")), "| target line:",
              next((l for l in mine if l.startswith("target features")), "(none)"))
    print(f"DRY DIGEST {'PASS' if ok == len(PTTPP) else 'FAIL'} {ok}/{len(PTTPP)}")
    sys.exit(0 if ok == len(PTTPP) else 1)


if __name__ == "__main__":
    main()
