"""pt2same test: --dry_inputs of rendered jobs on CPU with the snapshot trainer (the job's own ARGS; only --out points into
tests/dry/ so no run directory of the batch is created) vs the Burst twin run's recorded dry_inputs.txt.
Per job: rc 0; the model line; 3 printed rows with 2 operand sites; the target line (win-ctx only); the sha256 of the 9
DRY lines (the job's grep | head -n 9) = (a) the value the job file checks, (b) make_pt2main_tasks.DRY, (c) the same hash
recomputed from the Burst twin's dry_inputs.txt; and every line of the two dry_inputs files compared (pin_run lines and
the --out path aside).
    python dry_cpu.py [name ...]     (default: seeds 0 and 4 of every cell = 8 jobs; 2 at a time)"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2same as M   # noqa: E402

G2 = M.g2()
T = Path(__file__).resolve().parent / "dry"
T.mkdir(exist_ok=True)
ENV = dict(os.environ, HF_HOME=str(M.B / "hf_cache"), PYTHONPATH=str(M.B / "pylib"), PYTHONDONTWRITEBYTECODE="1",
           TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
ENV.pop("SLURM_JOB_ID", None)
JOBS = {j[0]: j for j in M.jobs(G2)}
want = sys.argv[1:] or [n for n in JOBS if n.endswith(("-s0", "-s4"))]


def one(name):
    _, tag, text, args, I, bname = JOBS[name]
    out = T / f"run_{name}"
    a = args.replace(f"--out {M.R}/out/{name}/{tag}", f"--out {out}/{tag}")
    assert a != args and f"--out {out}/{tag}" in a
    log = T / f"{name}.dry_inputs.txt"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([M.PY, "pretrain_range.py"] + a.split() + ["--dry_inputs"], cwd=M.R / "code" / "tools", env=ENV,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    txt = log.read_text()
    h = subprocess.run(["bash", "-c", f'set +o pipefail; grep -E "{G2.DRY_GREP}" "{log}" | head -n 9 | sha256sum | cut -d" " -f1'],
                       capture_output=True, text=True).stdout.strip()
    job_h = re.findall(r'^\[ "\$got" = ([0-9a-f]{64}) \]', text, re.M)
    size = name.split("-")[1]
    arm = "none" if "-none-" in name else "win-ctx"
    seed = int(name.rsplit("-s", 1)[1])
    btxt = (M.BURST / "results" / bname / "dry_inputs.txt").read_text()
    bh = M.dry_rows_hash(G2, btxt)
    keep = lambda t_: [l for l in t_.splitlines() if not l.startswith("pin_run:")]
    mine, theirs = keep(txt), keep(btxt)
    dl = [(i, x, y) for i, (x, y) in enumerate(zip(mine, theirs)) if x != y]
    res = {"name": name, "burst": bname, "rc": rc, "sec": round(time.time() - t0), "sha": h, "job_checks": job_h,
           "make_pt2main_DRY": G2.DRY[("pretrain_data", size, "ctx", seed)], "burst_recomputed": bh,
           "model_line": [l for l in txt.splitlines() if l.startswith("model:")],
           "rows_2sites": len(re.findall(r"^train row [012] \(len 512, 2 prior sites at ", txt, re.M)),
           "target_line": re.findall(r"^target features .*$", txt, re.M),
           "lines": [len(mine), len(theirs)], "differing_lines": [f"{i}: torch {x[:150]!r} | burst {y[:150]!r}" for i, x, y in dl]}
    res["ok"] = (rc == 0 and job_h == [h] and h == res["make_pt2main_DRY"] == bh
                 and res["model_line"] == [G2.SIZES[size]["model_line"]] and res["rows_2sites"] == 3
                 and res["target_line"] == ([] if arm == "none" else ["target features (1000, 10)"]))
    (T / f"{name}.json").write_text(json.dumps(res, indent=1))
    return res


with ThreadPoolExecutor(2) as ex:
    R_ = list(ex.map(one, want))
bad = 0
for r in R_:
    bad += not r["ok"]
    print(f"{'PASS' if r['ok'] else 'FAIL'} {r['name']} (Burst {r['burst']}): rc {r['rc']}, {r['sec']} s; DRY sha {r['sha'][:12]} "
          f"= job file {[x[:12] for x in r['job_checks']]} = make_pt2main_tasks.DRY {r['make_pt2main_DRY'][:12]} = Burst "
          f"dry_inputs.txt {r['burst_recomputed'][:12]}; {r['model_line']}; rows with 2 sites {r['rows_2sites']}; target "
          f"{r['target_line']}; full file: {r['lines'][0]} vs {r['lines'][1]} lines, {len(r['differing_lines'])} differ")
    for d in r["differing_lines"]:
        print("      " + d)
print(f"DRY CPU {'PASS' if not bad else 'FAIL'} ({len(R_) - bad}/{len(R_)})")
sys.exit(1 if bad else 0)
