"""Rendered pt300x jobs vs the nearest submitted jobs: every differing trainer argument (ARGS) and every differing job line.
References: pttpp 93M / 294M stage A + C (same shape, fewer tokens, L40S), the pttpp B200 bridge (505M at 3e8 on B200: the
B200 header and ckpt), F87 ptlr 190M (3e8 tokens, L40S) and ptlr_1b 1.07B (3e8, B200). Allowed differences: shape-, steps-,
eval-, lr-, seed-, tag- and path-related args; --save_model / --ckpt_every are listed separately (output / resume only; the
pttpp 7% rule: no ckpt on L40S, ckpt 500 on B200, --save_model only at stage C). CHECK7 must be byte-identical.
    python tests/args_diff.py"""
import re
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt300x as G   # noqa: E402

RUNS = Path("/scratch/yh6210/llmadd_torch/runs")
CAT = {"--d_model": "shape", "--n_layers": "shape", "--n_heads": "shape", "--d_ff": "shape", "--tokens": "steps",
       "--eval_every": "eval", "--lr": "lr", "--seed": "seed", "--tag": "tag", "--out": "path",
       "--save_model": "OUTPUT (model file)", "--ckpt_every": "RESUME (ckpt cadence)"}
ALLOWED = {"shape", "steps", "eval", "lr", "seed", "tag", "path"}


def args_of(txt):
    return G.M.M1.flags(re.search(r'^ARGS="(.*)"$', txt, re.M).group(1))


def mine(size, arm, seed, stage):
    if stage == "C":
        F = G.facts(size, arm, G.GRID[size][arm][1], seed, "C")
        return F["name"], G.job_x(F, slot=G.slot_of(size, 0, "C", arm, seed))
    return next((n, t) for n, t, F in G.render(size) if F["arm"] == arm and F["seed"] == seed and F["stage"] == stage)


PAIRS = [  # (size, arm, seed, stage, reference job file)
    ("93m", "none", 0, "A", "pttpp/job_tpp7-93m-t0.745-lr9.6e-3-none-s0.sbatch"),
    ("93m", "win-ctx", 0, "A", "pttpp/job_tpp7-93m-t0.745-lr2.4e-3-win-ctx-s0.sbatch"),
    ("93m", "none", 4, "C", "pttpp/job_tpp7-93m-t0.745-lr1.92e-2-none-s4.sbatch"),
    ("294m", "none", 0, "A", "pttpp/job_tpp7-294m-t0.745-lr4.8e-3-none-s0.sbatch"),
    ("294m", "win-ctx", 0, "A", "pttpp/job_tpp7-294m-t0.745-lr1.2e-3-win-ctx-s0.sbatch"),
    ("294m", "none", 4, "C", "pttpp/job_tpp7-294m-t0.745-lr1.92e-2-none-s4.sbatch"),
    ("294m", "none", 0, "A", "pttpp/job_tppb-505m-3e8-lr9.6e-3-none-s0.sbatch"),
    ("294m", "win-ctx", 0, "A", "pttpp/job_tppb-505m-3e8-lr1.2e-3-win-ctx-s0.sbatch"),
    ("93m", "none", 0, "A", "ptlr/job_190m-lr4.8e-3-7pct-none-s0.sbatch"),
    ("93m", "win-ctx", 0, "A", "ptlr/job_190m-lr1.2e-3-7pct-win-ctx-s0.sbatch"),
    ("294m", "none", 0, "A", "ptlr/job_190m-lr4.8e-3-7pct-none-s0.sbatch"),
    ("294m", "none", 4, "C", "ptlr_1b/job_1b-lr2.4e-3-7pct-none-s4.sbatch"),
]


def main():
    bad = 0
    for size, arm, seed, stage, ref in PAIRS:
        name, txt = mine(size, arm, seed, stage)
        rt = (RUNS / ref).read_text()
        a, b = args_of(txt), args_of(rt)
        diff = {k: (b.get(k), a.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        cats = {k: CAT.get(k, "UNEXPECTED") for k in diff}
        unexpected = [k for k, c in cats.items() if c == "UNEXPECTED"]
        bad += bool(unexpected)
        print(f"{name} vs {ref}:")
        for k, (old, new) in diff.items():
            print(f"    {k:13s} {str(old):>70s} -> {str(new):<70s} [{cats[k]}]")
        extra = sorted({c for c in cats.values() if c not in ALLOWED})
        print(f"    => {len(diff)} differing args; outside shape/steps/eval/lr/seed/tag/path: {extra or 'none'}")
        if "CHECK7" not in ref and ("pttpp/" in ref):
            c1 = txt.split("<<'PY' || RC=$?\n", 1)[1].split("\nPY\n", 1)[0]
            c2 = rt.split("<<'PY' || RC=$?\n", 1)[1].split("\nPY\n", 1)[0]
            same = c1 == c2 == G.M.CHECK7.rstrip("\n")
            bad += not same
            print(f"    CHECK7 body byte-identical to the reference job's and make_jobs_pttpp.CHECK7: {same}")
    # job lines other than ARGS / the intent JSON: pt300x L40S vs pttpp 93M, pt300x B200 vs the pttpp bridge
    for size, ref in (("93m", "pttpp/job_tpp7-93m-t0.745-lr9.6e-3-none-s0.sbatch"), ("294m", "pttpp/job_tppb-505m-3e8-lr9.6e-3-none-s0.sbatch")):
        name, txt = mine(size, "none", 0, "A")
        strip = lambda t: [l for l in t.split("<<'PY' || RC=$?\n", 1)[0].splitlines() if not l.startswith("ARGS=")]
        A, B = strip(txt), strip((RUNS / ref).read_text())
        print(f"\njob lines (before CHECK7, without ARGS) {name} vs {ref}: {len(A)} vs {len(B)} lines")
        import difflib
        for l in difflib.unified_diff(B, A, "pttpp", "pt300x", n=0, lineterm=""):
            if not l.startswith("@@"):
                print("    " + l[:260])
        tail_a = txt.split("\nPY\n", 1)[1].splitlines()
        tail_b = (RUNS / ref).read_text().split("\nPY\n", 1)[1].splitlines()
        print(f"    after CHECK7: pt300x {tail_a} | pttpp {tail_b}")
    print(f"ARGS DIFF {'PASS' if not bad else 'FAIL'}: no unexpected argument; CHECK7 verbatim")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
