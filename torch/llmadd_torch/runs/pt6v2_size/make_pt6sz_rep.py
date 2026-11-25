#!/usr/bin/env python3
"""pt6v2-size REP 505M (post-hoc, exploratory, 2026-11-24; exp-builder): replication seeds 17 and 18 at 505M, plus one higher
none lr. Authority: the user's standing authorisation (at most 2 replication seeds, same setup, already-approved cluster A100/H100).
Why: at 505M seed 16, all6 area > none@N_m on all six manifolds and all6full final3 > the best none final3 on all six, but several
gaps are small (cities / countries 1-3 points); the none area optimum on dim sits at the grid edge (1.92e-2), so 3.84e-2 closes
the bracket.
Cells (505M only, 1024x24, 16 heads, d_ff 4096; everything else = the seed-16 505M mains):
  seeds 17, 18: all6@6e-4, all6full@6e-4, only-dim@6e-4, none@{4.8e-3, 9.6e-3, 1.92e-2, 3.84e-2}   (14 runs)
  seed 16:      none@3.84e-2                                                                        (1 run)
Each job = make_pt6sz_jobs.build(dry, seed=s, plan={"505m": [...]}): the same counted edits as the seed-16 505M jobs, applied
to the A100 render of the registered pt6v2 job of seed s (ARGS asserted equal to runs/pt6v2/job_pt6v2-<arm>-lr<lr>-s<s>.sbatch).
The seed-16 defaults of build() render the registered 24 jobs byte-identically (checked: --check-s16).
Dry gate: CPU --dry_inputs of every new job's ARGS (dry_rep/), the job's own row reference in DRY_REF_rep.json; --collect also
checks the consumed values: per-domain prior on/off and lam, split seeds and split sizes = the job's intent (= the registered
seed-s job), rows identical within a seed and different across seeds, s16 rows = the registered s16 505M reference.
Modes: (none) list; --check-s16; --dry-script; --collect; --write; --test-only; --submit (SUBMITTED_<run> flags; manifest
SUBMITTED_rep.json).
REP 190M (2026-11-24, exp-builder; same authority): --size 190m selects the 190M plan (768x12, 12 heads, d_ff 3072; recipe = the
seed-16 190M mains: 06:00:00, 64G, a100_tandon,h100_tandon,l40s_public). Why: at 190M seed 16 all6 beats none@N_m on all six
manifolds in area and final3; the none area optimum sits at the low grid edge for numbers (2.4e-3) and the high edge for torus
(1.92e-2), so 1.2e-3 and 3.84e-2 close both brackets.
  seeds 17, 18: all6@6e-4, all6full@6e-4, only-dim@6e-4, none@{1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2, 3.84e-2}   (18 runs)
  seed 16:      none@1.2e-3, none@3.84e-2                                                                       (2 runs)
Files for 190m: DRY_REF_rep190.json, dry_rep190/, SUBMITTED_rep190.json. Without --size (= --size 505m) every name, file and
render is the REP 505M one (the 15 renders checked byte-identical to the on-disk 505M job files: rep190_checks/).
REP 1B (2026-11-24, exp-builder; same authority): --size 1b selects the 1.07B plan (1280x36, 20 heads, d_ff 5120; recipe = the
finished seed-16 1.07B mains = make_pt6sz_1b_fix.c1b() of the build() render: trainer runs/pt6v2_size/code_1b 4795c5267741,
20:00:00, 128G, a100_tandon,h100_tandon). Why: at 1.07B seed 16 the six priors beat the best no-prior lr on all six manifolds in
area and final3; the none area optimum is at the top of the grid (9.6e-3) for five of six, so 1.92e-2 and 3.84e-2 close the
bracket; 1.2e-3 / 2.4e-3 (best for nothing) and only-dim (at the ceiling on dim) are not repeated.
  seeds 17, 18: all6@6e-4, all6full@6e-4, none@{4.8e-3, 9.6e-3, 1.92e-2, 3.84e-2}   (12 runs)
  seed 16:      none@1.92e-2, none@3.84e-2                                         (2 runs)
Job files 1b/job_<run>.c1b.sbatch (the trainer, as the s16 1.07B mains); flags 1b/SUBMITTED_<run> (the plain name, which
make_pt6sz_jobs.build's squeue / run-JSON checks read); CPU dry run with the code_1b trainer. Files: DRY_REF_rep1b.json,
dry_rep1b/, SUBMITTED_rep1b.json. --size 505m / 190m: every name, file and render unchanged (rep1b_checks/).
SAMELR (2026-11-25, exp-builder; user 2026-11-25 "可以的，我觉得都可以跑 可以多用一些gpu"): --samelr [--size z] selects, at size z,
none@6e-4 for seeds 16, 17, 18, 19 plus all6@6e-4 seed 19 (5 runs per size, 15 over 190M / 505M / 1.07B). Why: the six-prior
arms ran at 6e-4 while every no-prior run used a higher lr, so the bpb comparison is confounded by lr, and the paper lost its
only six-prior downstream comparison (F96 P3, the removed rb6 model); a no-prior model at the priors' lr gives same-lr bpb and
the downstream pairs all6 vs none @6e-4, seeds 16-19 (n = 4, as F96). Recipe per size = that size's REP renders (1.07B via
c1b, the code_1b trainer). Seed 19 = the registered pt6v2 seed-19 job. Files: DRY_REF_samelr<k>.json, dry_samelr<k>/,
SUBMITTED_samelr<k>.json (k = 190 / 505 / 1b). --collect also checks that each seed's dry rows equal that seed's rows in the
REP reference of the size (seeds 16-18); seed 19 is new. Without --samelr every name, file and render is unchanged
(samelr_checks/)."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
S = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2_size")
sys.path.insert(0, str(S))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_pt6sz_jobs as M   # noqa: E402  (imports posthoc_v7_jobs read only; its asserts pin the frozen files)
sys.argv = _argv
P7, R, B = M.P7, M.R, M.B

REP = ("all6", "6e-4"), ("all6full", "6e-4"), ("only-dim", "6e-4"), ("none", "4.8e-3"), ("none", "9.6e-3"), ("none", "1.92e-2"), ("none", "3.84e-2")
REP190 = (("all6", "6e-4"), ("all6full", "6e-4"), ("only-dim", "6e-4"), ("none", "1.2e-3"), ("none", "2.4e-3"), ("none", "4.8e-3"),
          ("none", "9.6e-3"), ("none", "1.92e-2"), ("none", "3.84e-2"))
REP1B = ("all6", "6e-4"), ("all6full", "6e-4"), ("none", "4.8e-3"), ("none", "9.6e-3"), ("none", "1.92e-2"), ("none", "3.84e-2")
CFG = {   # per size: plan, files, and the asserted shape / n_params / header (= the seed-16 mains of that size, make_pt6sz_jobs.SIZES)
    "505m": dict(plan={17: [(a, lr, False) for a, lr in REP], 18: [(a, lr, False) for a, lr in REP], 16: [("none", "3.84e-2", False)]},
                 dry_ref="DRY_REF_rep.json", dryd="dry_rep", mani="SUBMITTED_rep.json", dry_name="llmadd_pt6sz-rep-dry",
                 shape=("1024", "24", "16", "4096"), npar=505437184, time="10:00:00", part="a100_tandon,h100_tandon"),
    "190m": dict(plan={17: [(a, lr, False) for a, lr in REP190], 18: [(a, lr, False) for a, lr in REP190],
                       16: [("none", "1.2e-3", False), ("none", "3.84e-2", False)]},
                 dry_ref="DRY_REF_rep190.json", dryd="dry_rep190", mani="SUBMITTED_rep190.json", dry_name="llmadd_pt6sz-rep190-dry",
                 shape=("768", "12", "12", "3072"), npar=190297344, time="06:00:00", part="a100_tandon,h100_tandon,l40s_public"),
    "1b": dict(plan={17: [(a, lr, False) for a, lr in REP1B], 18: [(a, lr, False) for a, lr in REP1B],
                     16: [("none", "1.92e-2", False), ("none", "3.84e-2", False)]},
               dry_ref="DRY_REF_rep1b.json", dryd="dry_rep1b", mani="SUBMITTED_rep1b.json", dry_name="llmadd_pt6sz-rep1b-dry",
               shape=("1280", "36", "20", "5120"), npar=1072259840, time="20:00:00", part="a100_tandon,h100_tandon"),
}
SIZE = sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "505m"
SAMELR = "--samelr" in sys.argv   # SAMELR (2026-11-25): none@6e-4 s16-19 + all6@6e-4 s19 at SIZE; header / shape / trainer = CFG[SIZE]
SL = [("none", "6e-4", False)]
SAMELR_CFG = {z: dict(plan={16: SL, 17: SL, 18: SL, 19: [("all6", "6e-4", False)] + SL}, dry_ref=f"DRY_REF_samelr{k}.json",
                      dryd=f"dry_samelr{k}", mani=f"SUBMITTED_samelr{k}.json", dry_name=f"llmadd_pt6sz-samelr{k}-dry")
              for z, k in (("505m", "505"), ("190m", "190"), ("1b", "1b"))}
C = dict(CFG[SIZE], **SAMELR_CFG[SIZE]) if SAMELR else CFG[SIZE]
assert (C["time"], C["part"]) == (M.SIZES[SIZE]["time"], M.SIZES[SIZE]["part"]) and \
       C["shape"] == tuple(str(M.SIZES[SIZE][k]) for k in ("d", "L", "H", "ff")) and C["npar"] == M.n_params(*(M.SIZES[SIZE][k] for k in ("d", "L", "ff")))
PLAN = C["plan"]
C1B = SIZE == "1b"   # REP 1B: every job text = make_pt6sz_1b_fix.c1b(build() text), as the finished seed-16 1.07B mains
if C1B:
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import make_pt6sz_1b_fix as F   # noqa: E402  (read only: its import asserts pin code_1b = registered + the dom_grads patch)
    sys.argv = _argv
CODE = F.C1 if C1B else R / "code"   # the trainer the jobs run (and the CPU dry run uses)
NJOBS = sum(len(v) for v in PLAN.values())
DRY_REF = S / C["dry_ref"]
DRYD = S / C["dryd"]
MANI = S / C["mani"]
LOGS = M.B / "logs"
DOMS = ("numbers", "dim", "cities", "countries", "torus", "box5")
PRIOR = {"all6": set(DOMS), "all6full": set(DOMS), "only-dim": {"dim"}, "none": set()}
WIN = {"all6": "0.05,0.4", "all6full": "0.05,1.0", "only-dim": "0.05,0.4", "none": None}
EV = {"all6": [[916, 1], [7324, 0]], "all6full": [[916, 1]], "only-dim": [[916, 1], [7324, 0]], "none": []}


def intent(text):
    return json.loads(P7.intent_line(text).split("'")[1])


def c1b_job(j):
    """REP 1B: the trainer edits of the finished seed-16 1.07B mains (make_pt6sz_1b_fix.c1b) and their job file name; the flag
    keeps the plain SUBMITTED_<run> name (make_pt6sz_jobs.build's squeue and run-JSON checks read it)."""
    j["text"] = F.c1b(j["text"])
    j["job"] = j["job"].with_name(j["job"].name[:-len(".sbatch")] + ".c1b.sbatch")
    return j


def build_all(dry=None):
    jobs = []
    for s, plan in PLAN.items():
        _, js = M.build(dry, seed=s, plan={SIZE: plan})
        for j in js:
            j["seed"] = s
            if C1B:
                c1b_job(j)
        jobs += js
    _, reg = M.build(json.loads(M.DRY_REF.read_text()))   # the registered 24 seed-16 jobs
    keys = ("slurm_name", "out", "job", "flag", "name", "run_json")
    for k in keys:
        vals = [str(j[k]) for j in jobs]
        assert len(set(vals)) == len(vals) == NJOBS, (k, vals)
        assert not set(vals) & {str(j[k]) for j in reg}, (k, set(vals) & {str(j[k]) for j in reg})
    existing_unique(jobs)
    for j in jobs:
        a, lr, s, t = j["arm"], j["lr"], j["seed"], j["text"]
        rn = f"{a}-lr{lr}-s{s}"
        assert j["name"] == f"{SIZE}/{rn}" and j["slurm_name"] == f"llmadd_pt6sz-{SIZE}-{rn}" and j["out"] == str(S / SIZE / rn), j["name"]
        av = j["args"].split()
        g = lambda k: av[av.index(k) + 1] if k in av else None   # noqa: E731
        assert (g("--seed"), g("--lr"), g("--window"), g("--d_model"), g("--n_layers"), g("--n_heads"), g("--d_ff"), g("--ckpt_every")) == \
               (str(s), lr, WIN[a]) + C["shape"] + ("500",), (j["name"], av)
        assert "--save_model" in av and "--tokens" not in av and g("--out") == f"{j['out']}/pt6v2-{a}-lr{lr}"
        pd = set(g("--prior_domains").split(",")) if g("--prior_domains") else set()
        assert pd == PRIOR[a], (j["name"], pd)
        I = intent(t)
        assert I["args"]["seed"] == s and I["args"]["lr"] == float(lr) and I["events"] == EV[a] == j["events"], (j["name"], I["events"])
        assert I["steps"] == 18310 and I["n_params"] == C["npar"] == j["n_params"] and I["save_model"] is True and I["args"]["ckpt_every"] == 500
        for ln in ("#SBATCH --account=<slurm-account>\n", f"#SBATCH --partition={C['part']}\n",
                   f"#SBATCH --time={C['time']}\n", "#SBATCH --requeue\n", "#SBATCH --open-mode=append\n", f"#SBATCH --job-name={j['slurm_name']}\n",
                   f"SEED={s};", f"seed-{s} job (A100 render)"):
            assert t.count(ln) == 1, (j["name"], ln)
        assert f"_seed{s}_pt6v2-{a}-lr{lr}.json" in j["run_json"]
        if C1B:
            hb = t.split("set -euo")[1]
            assert I["code_sha"] == F.NEW_TR and I["gpu"] == "A100|H100|H200|B200" and j["job"].name == f"job_{rn}.c1b.sbatch", j["name"]
            assert hb.count('cd "$C1/tools"\n') == 1 and f'= "{F.NEW_TR}" ]' in hb and f'= "{M.TR}" ]' not in hb and '"$R/code' not in hb
            sb = [x for x in t.splitlines() if x.startswith("#SBATCH")]
            assert "#SBATCH --mem=128G" in sb and not any("b200" in x.lower() or "--constraint" in x or "--comment" in x for x in sb)
            assert j["flag"].name == f"SUBMITTED_{rn}"
    return jobs


def existing_unique(jobs):
    """REP 190M: every not-yet-submitted job's Slurm name, out dir, job file and run JSON is new against everything on disk and in
    Slurm: no job file / flag / out dir at its path, no log named <slurm_name>_*.out, the name in no SUBMITTED*.json manifest, no job
    of this user with that name in sacct since 2026-11-01 (squeue --me is checked by make_pt6sz_jobs.build); an out dir may exist
    with empty dirs only (the CPU dry run creates <out>/<tag>/)."""
    todo = [j for j in jobs if not j["flag"].exists()]
    if not todo:
        return
    man = set()
    for f in S.glob("SUBMITTED*.json"):
        for r in json.loads(f.read_text())["runs"].values():
            man |= {r["slurm_name"], r["out"], r["run_json"]}
    hist = set(subprocess.run(["sacct", "-X", "-n", "-S", "2026-11-01", "--format=JobName%80"], capture_output=True, text=True).stdout.split())
    logs = {p.name for p in LOGS.glob("llmadd_pt6sz-*")}
    bad = []
    for j in todo:
        if j["job"].exists() and j["job"].read_text() != j["text"]:
            bad.append(f"{j['job']} exists with other text")
        rn = j["name"].split("/", 1)[1]   # REP 1B: no other job file / flag variant of this run (plain, .c1b, .b200)
        bad += [f"other file of this run: {p}" for p in j["job"].parent.glob(f"job_{rn}.*") if p != j["job"]]
        bad += [f"flag variant exists: {p}" for p in j["job"].parent.glob(f"SUBMITTED_{rn}*")]
        if Path(j["out"]).exists() and any(p.is_file() for p in Path(j["out"]).rglob("*")):   # the CPU dry run leaves empty dirs only
            bad.append(f"out dir has files: {j['out']}")
        if {j["slurm_name"], j["out"], j["run_json"]} & man:
            bad.append(f"in a SUBMITTED manifest: {j['name']}")
        if j["slurm_name"] in hist:
            bad.append(f"sacct has a job named {j['slurm_name']}")
        if any(x.startswith(j["slurm_name"] + "_") for x in logs):
            bad.append(f"a log exists for {j['slurm_name']}")
    assert not bad, bad


def check_s16():
    """the seed-16 jobs of SIZE rendered through the seed / plan path = the on-disk registered files, byte for byte."""
    dry = json.loads(M.DRY_REF.read_text())
    _, js = M.build(dry, seed=16, plan={SIZE: [(a, lr, False) for a, lr in (("all6", "6e-4"), ("all6full", "6e-4"), ("only-dim", "6e-4"))]
                                       + [("none", lr, False) for lr in M.SIZES[SIZE]["none"]]})
    for j in js:
        if C1B:   # REP 1B: vs the on-disk 1b/job_<run>.c1b.sbatch (the files the finished seed-16 1.07B mains ran)
            c1b_job(j)
        same = j["job"].read_text() == j["text"]
        print(f"s16 seed-path render vs on-disk {j['job'].name}: {'BYTE-IDENTICAL' if same else 'DIFFERS'}")
        assert same
    return True


def dry_script(jobs):
    lines = ["#!/bin/bash", "#SBATCH --account=<slurm-account>", "#SBATCH --partition=cpu_short", "#SBATCH --cpus-per-task=16",
             "#SBATCH --mem=96G", "#SBATCH --time=01:00:00", f"#SBATCH --job-name={C['dry_name']}",
             f"#SBATCH --output={DRYD}/dry_%j.log", "set -u", "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
             f"export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib", f"cd {CODE}/tools", "PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python",
             'echo "start $(date -u +%FT%TZ) $(hostname)"', "i=0"]
    for j in jobs:
        f = DRYD / (j["name"].replace("/", "_") + ".txt")
        lines.append(f'( "$PY" pretrain_range.py {j["args"]} --dry_inputs > {f} 2>&1; echo "rc=$?" >> {f} ) &')
        lines.append("i=$((i+1)); [ $((i % 4)) = 0 ] && wait")
    lines += ["wait", 'echo "all rc=0 done $(date -u +%FT%TZ)"']
    return "\n".join(lines) + "\n"


def collect(jobs):
    ref, bad = {}, []
    for j in jobs:
        f = DRYD / (j["name"].replace("/", "_") + ".txt")
        txt = f.read_text() if f.exists() else ""
        assert txt.rstrip().endswith("rc=0"), f"{f}: dry not finished / failed: {txt[-500:]}"
        pat = re.search(r'n=\$\(set \+o pipefail; grep -cE "([^"]+)" "\$OUT/dry_inputs.txt"\)', j["text"]).group(1)
        body = "\n".join(x for x in txt.splitlines() if x != "rc=0") + "\n"
        tmp = DRYD / "_body.txt"
        tmp.write_text(body)
        gr = subprocess.run(["grep", "-E", pat, str(tmp)], capture_output=True, text=True).stdout
        ref[j["name"]] = {"n": len(gr.splitlines()), "sha": hashlib.sha256(gr.encode()).hexdigest(),
                          "model_line": next((x for x in body.splitlines() if x.startswith("model: ")), None)}
        if ref[j["name"]]["model_line"] != j["model_line"]:
            bad.append(f"{j['name']}: model line {ref[j['name']]['model_line']}")
        # consumed values: per-domain prior flag / lam, split seed and sizes = the job's intent (from the registered seed-s job)
        dl = next(x for x in body.splitlines() if x.startswith("domains: "))
        D = json.loads(dl[len("domains: "):])
        I = intent(j["text"])
        got = {d for d in DOMS if D[d]["prior"]}
        if got != PRIOR[j["arm"]] or any(D[d]["lam"] != (1.0 if d in got else 0.0) for d in DOMS):
            bad.append(f"{j['name']}: prior domains {sorted(got)} lam {[D[d]['lam'] for d in DOMS]}")
        for d in DOMS[1:]:
            for k in ("split_seed", "n_held", "n_train", "n_readout"):
                if k in I["dom"].get(d, {}) and D[d].get(k) != I["dom"][d][k]:
                    bad.append(f"{j['name']}: {d}.{k} dry {D[d].get(k)} != intent {I['dom'][d][k]}")
        j["split_seeds"] = {d: D[d]["split_seed"] for d in DOMS[1:]}
    s16 = json.loads(M.DRY_REF.read_text())[f"{SIZE}/all6-lr6e-4-s16"]["sha"]
    by_seed = {s: {ref[j["name"]]["sha"] for j in jobs if j["seed"] == s} for s in PLAN}
    for s, v in by_seed.items():
        if len(v) != 1:
            bad.append(f"seed {s}: rows differ between arms within the seed: {sorted(x[:12] for x in v)}")
    if by_seed[16] != {s16}:
        bad.append(f"seed 16 rows {sorted(x[:12] for x in by_seed[16])} != the registered s16 {SIZE} reference {s16[:12]}")
    if len(set().union(*by_seed.values())) != len(PLAN):
        bad.append(f"rows not distinct across seeds: {[(s, sorted(x[:12] for x in v)) for s, v in by_seed.items()]}")
    ss = {s: {json.dumps(j["split_seeds"], sort_keys=True) for j in jobs if j["seed"] == s} for s in PLAN}
    if any(len(v) != 1 for v in ss.values()) or len(set().union(*ss.values())) != len(PLAN):
        bad.append(f"split seeds not one per seed / not distinct across seeds: {ss}")
    if SAMELR:   # SAMELR: each seed's rows = that seed's rows in the REP reference of this size (seeds 16-18; seed 19 is new)
        rr = json.loads((S / CFG[SIZE]["dry_ref"]).read_text())
        for s in PLAN:
            rs = {v["sha"] for k, v in rr.items() if k.endswith(f"-s{s}")}
            print(f"seed {s}: rows {sorted(x[:12] for x in by_seed[s])} vs REP {CFG[SIZE]['dry_ref']} {sorted(x[:12] for x in rs) or 'none (new seed)'}")
            if rs and by_seed[s] != rs:
                bad.append(f"seed {s}: rows {sorted(x[:12] for x in by_seed[s])} != REP reference {sorted(x[:12] for x in rs)}")
            if not rs and s != 19:
                bad.append(f"seed {s}: no REP reference rows")
    DRY_REF.write_text(json.dumps(ref, indent=1) + "\n")
    jobs2 = build_all(ref)
    for j in jobs2:
        f = DRYD / (j["name"].replace("/", "_") + ".txt")
        od = DRYD / "_gate" / j["name"].replace("/", "_")
        od.mkdir(parents=True, exist_ok=True)
        (od / "dry_inputs.txt").write_text("\n".join(x for x in f.read_text().splitlines() if x != "rc=0") + "\n")
        o = subprocess.run(["bash", "-c", "set -euo pipefail\nOUT=" + str(od) + "\n" + M.gate_lines(j["text"])], capture_output=True, text=True)
        ok = o.returncode == 0
        if not ok:
            bad.append(f"{j['name']}: own dry gate failed")
        print(f"{'GATE OK ' if ok else 'GATE BAD'} {j['name']:26s} seed {j['seed']} {ref[j['name']]['n']} lines {ref[j['name']]['sha'][:12]} | "
              f"{j['model_line']} | lr {j['lr']} window {j['window']} events {j['events']} | prior {sorted(PRIOR[j['arm']])} | "
              f"{(o.stdout + o.stderr).strip()[-90:]}")
    for s in PLAN:
        print(f"seed {s}: rows {sorted(x[:12] for x in by_seed[s])}, split seeds {sorted(ss[s])}")
    print("collect:", "ALL CHECKS PASS" if not bad else "FAIL\n  " + "\n  ".join(bad))
    return not bad


def main():
    dry = json.loads(DRY_REF.read_text()) if DRY_REF.exists() else None
    if "--check-s16" in sys.argv:
        check_s16()
    jobs = build_all(dry)
    for j in jobs:
        st = "SUBMITTED" if j["flag"].exists() else "on disk" if j["job"].exists() else "not written"
        print(f"{j['name']:26s} {j['slurm_name']:42s} seed {j['seed']} lr {j['lr']:7s} ev {j['events']} dry {'ok' if j['dry_ok'] else 'PENDING'} {st}")
    if "--dry-script" in sys.argv:
        DRYD.mkdir(parents=True, exist_ok=True)
        (DRYD / "dry_rep.sbatch").write_text(dry_script(jobs))
        print("written", DRYD / "dry_rep.sbatch")
    if "--collect" in sys.argv:
        sys.exit(0 if collect(jobs) else 1)
    if "--write" in sys.argv:
        assert all(j["dry_ok"] for j in jobs), "DRY_REF_rep.json incomplete: run --collect first"
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"]
                continue
            j["job"].write_text(j["text"])
        print("written", len(jobs), "job files")
    if "--test-only" in sys.argv:
        for j in jobs:
            o = subprocess.run(P7.SBATCH + ["--test-only", str(j["job"])], capture_output=True, text=True)
            print(j["slurm_name"], (o.stdout + o.stderr).strip().splitlines()[-1:])
    if "--submit" in sys.argv:
        assert all(j["dry_ok"] for j in jobs)
        stale = [j["name"] for j in jobs if not j["flag"].exists() and (not j["job"].exists() or j["job"].read_text() != j["text"])]
        assert not stale, f"job files missing or differing (run --write first): {stale}"
        man = json.loads(MANI.read_text()) if MANI.exists() else {"commit": M.MJ2.COMMIT, "trainer_sha12": F.NEW_TR if C1B else M.TR, "runs": {}}
        for j in jobs:   # PLAN order: seed 17, seed 18, seed 16
            assert f"/{SIZE}/" in str(j["job"]) and j["slurm_name"].startswith(f"llmadd_pt6sz-{SIZE}-")
            if j["flag"].exists():
                print(j["name"], "refused (SUBMITTED flag exists)")
                continue
            o = subprocess.run(P7.SBATCH + [str(j["job"])], capture_output=True, text=True)
            print(j["slurm_name"], (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                man["runs"][j["name"]] = {k: j[k] for k in ("size", "arm", "lr", "seed", "slurm_name", "out", "run_json", "args",
                                                            "events", "steps", "window", "base_job", "n_params")}
                man["runs"][j["name"]].update(job_id=o.stdout.split()[-1], job_sha12=hashlib.sha256(j["text"].encode()).hexdigest()[:12])
                if C1B:
                    man["runs"][j["name"]]["job_file"] = str(j["job"])
                MANI.write_text(json.dumps(man, indent=1) + "\n")


if __name__ == "__main__":
    main()
