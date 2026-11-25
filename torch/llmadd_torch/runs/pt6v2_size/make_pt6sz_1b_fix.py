#!/usr/bin/env python3
"""pt6v2-size 1.07B fix (exp-builder, 2026-11-24). The 1.07B canary 24340300 died with CUDA OOM in the trainer's per-domain
gradient diagnostic dom_grads (full-parameter fp32 gradient vectors of LM + 6 true priors kept on the GPU, 4.0 GiB each at
1.07B, plus the weighted active copies). Fix = diagnostic only: runs/pt6v2_size/code_1b = runs/pt6v2/code with pretrain_range.py
patched (code_1b.diff: the vectors live on CPU; same keys, same values up to the fp reduction order; training unchanged).

Jobs = make_pt6sz_jobs.build() texts of the 8 1.07B runs (asserted equal to the submitted 1b/job_<run>.sbatch), with counted
edits only:
  C1. trainer dir runs/pt6v2/code -> runs/pt6v2_size/code_1b (cd, COMMIT path x2); the HOLD kill switch stays runs/pt6v2/HOLD*;
  C2. the trainer sha gate and the COMMIT grep: a398fccbfe93 -> NEW_TR; the intent JSON code_sha -> NEW_TR;
  C3. GPU set += B200 (case, DEVICE echo, intent "gpu"): a run may be resumed across copies (run guard takeover), so both
      copies accept the whole set; mixed devices are a flag in the read (as before);
  C4. one provenance line.
  -> 1b/job_<run>.c1b.sbatch (A100/H100 copy, same account / partitions / time / mem as before).
B200 race copy (user approval relayed by the coordinator 2026-11-24): header only: --account=<slurm-account>,
--comment="preemption=yes", --gres=gpu:1 --constraint=b200, no --partition / --qos lines, --time=12:00:00 (canary keeps 01:30:00);
--requeue, --open-mode=append, --output pattern, ARGS (incl. --ckpt_every) unchanged -> <size>/job_<run>.b200.sbatch. The run
guard lets only the first claimant run. 505M B200 copies (registered trainer, C3 + header only) are written with --b200-505m.
Modes: (none) build + check; --write; --test-only; --submit [--only-c1b | --only-b200] (refuses when SUBMITTED_<run>.c1b / .b200 flags exist)."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

S = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2_size")
sys.path.insert(0, str(S))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_pt6sz_jobs as M   # noqa: E402  (read only: build() re-renders and asserts the registered ARGS)
sys.argv = _argv
R, TR = M.R, M.TR
C1 = S / "code_1b"
NEW_TR = "4795c5267741"
assert M.sha(C1 / "tools/pretrain_range.py") == NEW_TR
assert f"pretrain_range.py {NEW_TR}" in (C1 / "COMMIT").read_text()
for p in sorted((R / "code").rglob("*")):   # every other file byte-identical to the registered snapshot
    if p.is_file() and "__pycache__" not in p.parts and p.name not in ("pretrain_range.py", "COMMIT"):
        q = C1 / p.relative_to(R / "code")
        assert q.read_bytes() == p.read_bytes(), q
assert (C1 / "COMMIT").read_text().startswith((R / "code/COMMIT").read_text())
MANI = S / "SUBMITTED_c1b.json"
PROV1 = ("# pt6v2-size 1.07B fix (2026-11-24, make_pt6sz_1b_fix.py): trainer = runs/pt6v2_size/code_1b (registered a398fccbfe93 + "
         "the diagnostic-only dom_grads CPU patch, code_1b.diff, pretrain_range.py " + NEW_TR + ") after the OOM of canary 24340300; "
         "training path unchanged (smoke_code_1b/); B200 allowed for run-guard takeovers.\n")
PROVB = "# B200 race copy (preemptible; the run guard lets only the first claimant run; a requeue keeps the job id and resumes).\n"


def gpus_b200(t, old):
    alts = "|".join(f"*{g}*" for g in old)
    t = M.edit(t, f'case "$GPU" in {alts}) ;;', f'case "$GPU" in {alts}|*B200*) ;;')
    t = M.edit(t, f'(pt6v2-size: {"/".join(old)} allowed;', f'(pt6v2-size: {"/".join(old)}/B200 allowed;')
    t = M.edit(t, f'"gpu":"{"|".join(old)}"', f'"gpu":"{"|".join(old)}|B200"')
    return t


def c1b(t):
    t = M.edit(t, "R=/scratch/yh6210/llmadd_torch/runs/pt6v2; ", f"R=/scratch/yh6210/llmadd_torch/runs/pt6v2; C1={C1}; ")
    t = M.edit(t, 'cd "$R/code/tools"\n', 'cd "$C1/tools"\n')
    t = M.edit(t, '"$R/code/COMMIT"', '"$C1/COMMIT"', 2)
    t = M.edit(t, f'[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{TR}" ]', f'[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{NEW_TR}" ]')
    t = M.edit(t, f'grep -q "pretrain_range.py {TR}"', f'grep -q "pretrain_range.py {NEW_TR}"')
    t = M.edit(t, f'COMMIT does not name the trainer {TR}"', f'COMMIT does not name the trainer {NEW_TR}"')
    t = M.edit(t, f'"code_sha":"{TR}"', f'"code_sha":"{NEW_TR}"')
    t = gpus_b200(t, M.SIZES["1b"]["gpus"])
    t = M.edit(t, "set -euo pipefail\n", PROV1 + "set -euo pipefail\n")
    assert TR not in t.split("set -euo")[1].replace(f"a398fccbfe93 + the diagnostic", ""), "registered trainer sha left in the body"
    assert '"$R/code' not in t and "runs/pt6v2/code/tools" not in t.split("set -euo")[1]
    return t


def b200(t, can):
    t = M.edit(t, "#SBATCH --account=<slurm-account>\n", "#SBATCH --account=<slurm-account>\n")
    t = re.sub(r"#SBATCH --partition=[^\n]*\n", "", t, count=1)
    t = M.edit(t, "#SBATCH --qos=gpu48\n", "")
    t = M.edit(t, "#SBATCH --gres=gpu:1\n", '#SBATCH --gres=gpu:1\n#SBATCH --constraint=b200\n#SBATCH --comment="preemption=yes"\n')
    if not can:
        t = re.sub(r"#SBATCH --time=[^\n]*\n", "#SBATCH --time=12:00:00\n", t, count=1)
    t = M.edit(t, "set -euo pipefail\n", PROVB + "set -euo pipefail\n")
    hdr = t.split("set -euo")[0]
    assert "--partition" not in hdr and "--qos" not in hdr and "#SBATCH --requeue\n" in hdr and "#SBATCH --open-mode=append\n" in hdr
    assert "#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out\n" in hdr
    return t


def build(b505=False):
    dry = json.loads(M.DRY_REF.read_text())
    _, jobs = M.build(dry)
    out = []
    for j in jobs:
        if j["size"] == "1b":
            assert j["job"].read_text() == j["text"], f"{j['job']}: on-disk job differs from the generator"
            t1 = c1b(j["text"])
            name = j["job"].name[:-len(".sbatch")]
            out.append(dict(j, kind="c1b", text=t1, job=j["job"].with_name(name + ".c1b.sbatch"),
                            flag=j["flag"].with_name(j["flag"].name + ".c1b")))
            out.append(dict(j, kind="b200", text=b200(t1, j["canary"]), job=j["job"].with_name(name + ".b200.sbatch"),
                            flag=j["flag"].with_name(j["flag"].name + ".b200")))
        elif j["size"] == "505m" and b505 and not j["canary"]:
            assert j["job"].read_text() == j["text"], f"{j['job']}: on-disk job differs from the generator"
            name = j["job"].name[:-len(".sbatch")]
            tb = b200(gpus_b200(j["text"], M.SIZES["505m"]["gpus"]), False)
            out.append(dict(j, kind="b200", text=tb, job=j["job"].with_name(name + ".b200.sbatch"),
                            flag=j["flag"].with_name(j["flag"].name + ".b200")))
    for j in out:
        assert subprocess.run(["bash", "-n"], input=j["text"].encode()).returncode == 0, j["job"]
    return out


def main():
    b505 = "--b200-505m" in sys.argv
    jobs = build(b505)
    for j in jobs:
        st = "SUBMITTED" if j["flag"].exists() else "on disk" if (j["job"].exists() and j["job"].read_text() == j["text"]) else "not written"
        print(f"{j['kind']:5s} {j['name']:26s} {j['slurm_name']:42s} {j['job'].name:40s} {st}")
    if "--write" in sys.argv:
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"]
                continue
            j["job"].write_text(j["text"])
        print("written", len(jobs))
    if "--test-only" in sys.argv:
        for j in jobs:
            o = subprocess.run(["sbatch", "--test-only", str(j["job"])], capture_output=True, text=True)
            print(j["kind"], j["slurm_name"], (o.stdout + o.stderr).strip().splitlines()[-1:])
    if "--submit" in sys.argv:
        stale = [j["job"].name for j in jobs if not j["flag"].exists() and (not j["job"].exists() or j["job"].read_text() != j["text"])]
        assert not stale, f"run --write first: {stale}"
        man = json.loads(MANI.read_text()) if MANI.exists() else {"trainer_1b_sha12": NEW_TR, "runs": {}}
        kinds = [k for k in ("c1b", "b200") if f"--only-{k}" in sys.argv] or ["c1b", "b200"]
        order = [j for k in kinds for j in jobs if j["kind"] == k]
        for j in order:
            if j["flag"].exists():
                print(j["job"].name, "refused (flag exists)")
                continue
            o = subprocess.run(["sbatch", str(j["job"])], capture_output=True, text=True)
            print(j["kind"], j["slurm_name"], (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                man["runs"][f"{j['name']}|{j['kind']}"] = {"size": j["size"], "kind": j["kind"], "slurm_name": j["slurm_name"],
                                                           "job_file": str(j["job"]), "out": j["out"], "run_json": j["run_json"],
                                                           "job_id": o.stdout.split()[-1],
                                                           "job_sha12": hashlib.sha256(j["text"].encode()).hexdigest()[:12]}
                MANI.write_text(json.dumps(man, indent=1) + "\n")


if __name__ == "__main__":
    main()
