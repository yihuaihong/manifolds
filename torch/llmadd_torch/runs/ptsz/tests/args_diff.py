"""Rendered ptsz jobs vs the nearest submitted jobs; every differing trainer argument and every differing job line.
1. Equivalence with pt300x (the template): ptsz's rendering of 294m (the same shape and grid) and, as a known answer, 93m
   (make_jobs_ptsz.facts(..., ka=True)) vs the SUBMITTED pt300x job file of the same arm, lr, seed and stage, after the
   name / path substitution p3x- -> psz-, runs/pt300x -> runs/ptsz, *_pt300x.py -> *_ptsz.py. Every remaining differing
   line is printed and must be one of the intended differences: the hardware header (B200 -> L40S for 294m), --nice 20 ->
   1000, --time, --open-mode (ckpt runs), the job / log name, comment lines, the GPU-guard string, and --ckpt_every (ARGS
   and the intent JSON's args.ckpt_every). Everything else (the body, ARGS otherwise, the intent JSON otherwise, CHECK7)
   must be identical; for 93m the ARGS and intent are identical outright.
2. ARGS vs pttpp 294M (stage A, same shape, 0.745 line), F87 ptlr 190M / 505M (3e8, L40S, ckpt 2000) and ptlr_1b 1.07B C:
   allowed differences shape / steps / eval / lr / seed / tag / path; --save_model / --ckpt_every listed separately.
3. CHECK7 byte-identical to make_jobs_pttpp.CHECK7 in every rendered job.
    python tests/args_diff.py"""
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_ptsz as G   # noqa: E402

M = G.M
RUNS = Path("/scratch/yh6210/llmadd_torch/runs")
SUB = [("p3x-", "psz-"), ("runs/pt300x", "runs/ptsz"), ("select_pt300x.py", "select_ptsz.py"),
       ("readout_pt300x.py", "readout_ptsz.py"), ("make_jobs_pt300x.py", "make_jobs_ptsz.py")]
CAT = {"--d_model": "shape", "--n_layers": "shape", "--n_heads": "shape", "--d_ff": "shape", "--tokens": "steps",
       "--eval_every": "eval", "--lr": "lr", "--seed": "seed", "--tag": "tag", "--out": "path",
       "--save_model": "OUTPUT (model file)", "--ckpt_every": "RESUME (ckpt cadence)"}
ALLOWED = {"shape", "steps", "eval", "lr", "seed", "tag", "path"}
HEADER_OK = ("#SBATCH --job-name=", "#SBATCH --dependency=", "#SBATCH --account=", "#SBATCH --partition=", "#SBATCH --qos=",
             "#SBATCH --gres=", "#SBATCH --constraint=", "#SBATCH --comment=", "#SBATCH --nice=", "#SBATCH --cpus-per-task=",
             "#SBATCH --mem=", "#SBATCH --time=", "#SBATCH --open-mode=", "#SBATCH --output=")


def args_of(txt):
    return M.M1.flags(re.search(r'^ARGS="(.*)"$', txt, re.M).group(1))


def intent_of(txt):
    return json.loads(re.search(r"^\S+ - \"[^\"]+\" \"[^\"]+\" '(\{.*\})' <<'PY' \|\| RC=\$\?$", txt, re.M).group(1))


def check7_of(txt):
    return txt.split("<<'PY' || RC=$?\n", 1)[1].split("\nPY\n", 1)[0]


def mine(size, arm, lr, seed, stage):
    F = G.facts(size, arm, lr, seed, stage, ka=size not in G.GRID)
    return F, G.job_x(F)


def classify(line):
    if line.startswith(HEADER_OK):
        return "header"
    if line.startswith("# ") or line == "#":
        return "comment"
    if line.startswith('case "$GPU" in'):
        return "GPU guard"
    if line.startswith('ARGS="'):
        return "ARGS"
    if "<<'PY' || RC=$?" in line:
        return "intent"
    return "UNEXPECTED"


EQUIV = [  # (size, arm, lr, seed, stage, pt300x job file)
    ("93m", "none", "2.4e-3", 0, "A", "job_p3x-93m-lr2.4e-3-none-s0.sbatch"),
    ("93m", "win-ctx", "6e-4", 0, "A", "job_p3x-93m-lr6e-4-win-ctx-s0.sbatch"),
    ("93m", "none", "2.4e-3", 4, "C", "job_p3x-93m-lr2.4e-3-none-s4.sbatch"),
    ("93m", "none", "1.2e-3", 1, "X", "job_p3x-93m-lr1.2e-3-none-s1.sbatch"),
    ("294m", "none", "2.4e-3", 0, "A", "job_p3x-294m-lr2.4e-3-none-s0.sbatch"),
    ("294m", "win-ctx", "3e-4", 0, "A", "job_p3x-294m-lr3e-4-win-ctx-s0.sbatch"),
    ("294m", "none", "9.6e-3", 4, "C", "job_p3x-294m-lr9.6e-3-none-s4.sbatch"),
    ("294m", "win-ctx", "6e-4", 7, "C", "job_p3x-294m-lr6e-4-win-ctx-s7.sbatch"),
]
PAIRS = [  # (size, arm, lr, seed, stage, reference job file) for the ARGS diff
    ("294m", "none", "4.8e-3", 0, "A", "pttpp/job_tpp7-294m-t0.745-lr4.8e-3-none-s0.sbatch"),
    ("294m", "win-ctx", "1.2e-3", 0, "A", "pttpp/job_tpp7-294m-t0.745-lr1.2e-3-win-ctx-s0.sbatch"),
    ("391m", "none", "4.8e-3", 0, "A", "ptlr/job_190m-lr4.8e-3-7pct-none-s0.sbatch"),
    ("391m", "win-ctx", "1.2e-3", 0, "A", "ptlr/job_190m-lr1.2e-3-7pct-win-ctx-s0.sbatch"),
    ("731m", "none", "9.6e-3", 0, "A", "ptlr/job_500m-lr9.6e-3-7pct-none-s0.sbatch"),
    ("731m", "win-ctx", "1.2e-3", 1, "A", "ptlr/job_500m-lr1.2e-3-7pct-win-ctx-s1.sbatch"),
    ("731m", "none", "2.4e-3", 4, "C", "ptlr_1b/job_1b-lr2.4e-3-7pct-none-s4.sbatch"),
    ("136m", "none", "2.4e-3", 0, "A", "pt300x/job_p3x-93m-lr2.4e-3-none-s0.sbatch"),
    ("136m", "win-ctx", "1.2e-3", 4, "C", "pt300x/job_p3x-93m-lr1.2e-3-win-ctx-s4.sbatch"),
]


def main():
    bad = 0
    print("## 1. equivalence with pt300x's submitted jobs (after the name / path substitution)")
    for size, arm, lr, seed, stage, ref in EQUIV:
        F, txt = mine(size, arm, lr, seed, stage)
        rt = (RUNS / "pt300x" / ref).read_text()
        for a, b in SUB:
            rt = rt.replace(a, b)
        A, Bl = txt.splitlines(), rt.splitlines()
        diff = [l for l in difflib.unified_diff(Bl, A, "pt300x", "ptsz", n=0, lineterm="") if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        cats = {}
        for l in diff:
            cats.setdefault(classify(l[1:]), []).append(l)
        ua, ub = args_of(txt), args_of(rt)
        dargs = {k: (ub.get(k), ua.get(k)) for k in sorted(set(ua) | set(ub)) if ua.get(k) != ub.get(k)}
        ia, ib = intent_of(txt), intent_of(rt)
        dint = {k: (ib["args"].get(k), ia["args"].get(k)) for k in ia["args"] if ia["args"].get(k) != ib["args"].get(k)}
        dtop = {k: (ib.get(k), ia.get(k)) for k in set(ia) | set(ib) if k != "args" and ia.get(k) != ib.get(k)}
        same_c7 = check7_of(txt) == check7_of(rt) == M.CHECK7.rstrip("\n")
        want_args = {} if size == "93m" else {"--ckpt_every": ("500", "2000")}
        want_int = {} if size == "93m" else {"ckpt_every": (500, 2000)}
        ok = ("UNEXPECTED" not in cats and dargs == want_args and dint == want_int and not dtop and same_c7
              and (size != "93m" or not {"ARGS", "intent", "GPU guard"} & set(cats)))
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {F['name']} vs pt300x {ref}: {len(diff)} differing lines "
              f"({', '.join(f'{k} {len(v)}' for k, v in sorted(cats.items()))}); ARGS diff {dargs or 'none'}; intent diff "
              f"{dint or 'none'}{'; other intent fields ' + str(dtop) if dtop else ''}; CHECK7 identical {same_c7}")
        if ref in ("job_p3x-93m-lr2.4e-3-none-s0.sbatch", "job_p3x-294m-lr2.4e-3-none-s0.sbatch", "job_p3x-294m-lr9.6e-3-none-s4.sbatch"):
            for l in diff:
                if classify(l[1:]) not in ("ARGS", "intent"):
                    print("      " + l[:230])
                else:
                    print("      " + l[0] + f" [{classify(l[1:])} line; differs only in the fields listed]")
    print("\n## 2. ARGS vs the nearest other submitted jobs")
    for size, arm, lr, seed, stage, ref in PAIRS:
        F, txt = mine(size, arm, lr, seed, stage)
        rt = (RUNS / ref).read_text()
        a, b = args_of(txt), args_of(rt)
        diff = {k: (b.get(k), a.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        cats = {k: CAT.get(k, "UNEXPECTED") for k in diff}
        unexpected = [k for k, c in cats.items() if c == "UNEXPECTED"]
        bad += bool(unexpected)
        print(f"{F['name']} vs {ref}:")
        for k, (old, new) in diff.items():
            print(f"    {k:13s} {str(old)[-62:]:>62s} -> {str(new)[-62:]:<62s} [{cats[k]}]")
        extra = sorted({c for c in cats.values() if c not in ALLOWED})
        print(f"    => {len(diff)} differing args; outside shape/steps/eval/lr/seed/tag/path: {extra or 'none'}")
    print("\n## 3. CHECK7 in every rendered stage-A / known-answer job")
    n7 = 0
    for size in G.GRID:
        for a in G.ARMS:
            for lr in G.GRID[size][a]:
                for s in G.SEL:
                    F, txt = mine(size, a, lr, s, "A")
                    n7 += 1
                    if check7_of(txt) != M.CHECK7.rstrip("\n"):
                        bad += 1
                        print("CHECK7 DIFFERS", F["name"])
    print(f"CHECK7 byte-identical to make_jobs_pttpp.CHECK7 in {n7} stage-A renderings")
    print(f"ARGS DIFF {'PASS' if not bad else 'FAIL'}: only the intended differences; CHECK7 verbatim")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
