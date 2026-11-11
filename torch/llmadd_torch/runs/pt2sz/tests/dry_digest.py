"""pt2sz test (CPU): known answer for the data stream and the job's own DRY check. For seed 0 of each size and arm (4 jobs),
the ARGS line is read from the RENDERED stage-A job file (only --out redirected to a scratch dir), --dry_inputs is run with
the snapshot trainer, and:
  (a) the 9 DRY lines' sha256 (the job's own grep | head -n 9 | sha256sum) = the hash the job file checks = pt2sz_refs.json
      = make_jobs_pttpp.dry2 (the Burst pt2s190 rows of the seed with this model line);
  (b) the whole dry file = a submitted Torch A100-80GB job's dry_inputs.txt of the same seed and arm line for line:
      294M = pttpp tpp2-294m-t0.745 (the same shape at 156.8M tokens; model line included), 93M = pt2same 190M with only
      the model line different;
  (c) the model line, 3 rows with 2 operand sites, the target line only for win-ctx.
Digest = sha256[:16] of the dry file.
    python dry_digest.py OUTDIR"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2sz as M   # noqa: E402

G2 = M.g2()
RUNS = M.B / "runs"
REF = {("294m", "none"): ("pttpp/out2/tpp2-294m-t0.745-lr4.8e-3-none-s0/dry_inputs.txt", True),
       ("294m", "win-ctx"): ("pttpp/out2/tpp2-294m-t0.745-lr6e-4-win-ctx-s0/dry_inputs.txt", True),
       ("93m", "none"): ("pt2same/out/pt2same-190m-lr4.8e-3-none-s0/dry_inputs.txt", False),
       ("93m", "win-ctx"): ("pt2same/out/pt2same-190m-lr6e-4-win-ctx-s0/dry_inputs.txt", False)}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    env = dict(os.environ, HF_HOME=str(M.B / "hf_cache"), PYTHONPATH=str(M.B / "pylib"), CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    for k in [k for k in env if k.startswith("SLURM_")]:
        env.pop(k)
    for (size, arm), (ref, same_model) in REF.items():
        lr = M.GRID[size][arm][1]
        name = f"{M.tag_of(size, arm, lr)}-s0"
        jt = (M.R / f"job_{name}.sbatch").read_text()
        args = re.search(r'^ARGS="(.*)"$', jt, re.M).group(1)
        job_h = re.findall(r'^\[ "\$got" = ([0-9a-f]{64}) \]', jt, re.M)
        scratch = out / f"dryout_{name}"
        tag = M.tag_of(size, arm, lr)
        a = args.replace(f"--out {M.R / 'out' / name}/{tag} ", f"--out {scratch}/{tag} ")
        assert a != args and str(M.R / "out") not in a
        o = subprocess.run([M.PY, "pretrain_range.py", *a.split(), "--dry_inputs"], cwd=M.R / "code" / "tools", env=env,
                           capture_output=True, text=True)
        log = out / f"dry_{name}.txt"
        log.write_text(o.stdout + o.stderr)
        h = subprocess.run(["bash", "-c", f'set +o pipefail; grep -E "{G2.DRY_GREP}" "{log}" | head -n 9 | sha256sum | cut -d" " -f1'],
                           capture_output=True, text=True).stdout.strip()
        exp = M.TP.dry2(G2, size, 0)
        rd = M.refs()["dry2"][size]["0"]
        mine = o.stdout.splitlines()
        mine = mine[next(j for j, l in enumerate(mine) if l.startswith("line:")):]
        theirs = (RUNS / ref).read_text().splitlines()
        if same_model:
            same = mine == theirs
        else:
            same = len(mine) == len(theirs) and all(x == y or (x == M.MODEL_LINE[size] and y.startswith("model:")) for x, y in zip(mine, theirs))
        c = (o.returncode == 0 and [l for l in mine if l.startswith("model:")] == [M.MODEL_LINE[size]]
             and len(re.findall(r"^train row [012] \(len 512, 2 prior sites at ", o.stdout, re.M)) == 3
             and re.findall(r"^target features .*$", o.stdout, re.M) == ([] if arm == "none" else ["target features (1000, 10)"]))
        good = job_h == [h] and h == exp == rd and same and c
        ok += good
        dg = hashlib.sha256(("\n".join(mine) + "\n").encode()).hexdigest()[:16]
        print(f"{'PASS' if good else 'FAIL'} {name} (rc {o.returncode}): DRY sha {h[:12]} = job file {[x[:12] for x in job_h]} = "
              f"pt2sz_refs {rd[:12]} = dry2 {exp[:12]}; dry file ({len(mine)} lines, digest {dg}) = {ref} "
              f"{'line for line' if same_model else 'except the model line'}: {same}; model / sites / target: {c}")
    print(f"DRY DIGEST {'PASS' if ok == len(REF) else 'FAIL'} {ok}/{len(REF)}")
    sys.exit(0 if ok == len(REF) else 1)


if __name__ == "__main__":
    main()
