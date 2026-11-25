#!/usr/bin/env python3
"""pt6v2 Amendment V9 (post-hoc, user-approved 2026-11-25 "可以的，我觉得都可以跑 可以多用一些gpu"; exp-builder): same-count,
dose-matched control for the joint lift on dim. all6@2e-3 with the torus and box5 priors given shuffled targets
(--prior_shuf torus,box5: the trainer's own mechanism, DTAB_ACT[d] = F[qperm] with qperm = far_perm on the domain's own
SeedSequence child, CKA to the true target < 0.15; the target Gram is a row/column permutation of the true one).
Question: is h6 = all6 - only-dim (dim strict area +13.5, 8/8) due to the torus / box5 GEOMETRY (then h_shuf ~ h4 = +1.5)
or to the extra prior gradient (then h_shuf ~ h6)?
Cells: all6-shufTB@2e-3, seeds 16..23 (8 runs). Same data, same 6 priors, lambda 1 each, window 0.05-0.4, 18,310 steps,
REGISTERED TRAINER UNCHANGED (runs/pt6v2/code, a398fccbfe93).

Each primary job = the A100 render of the registered all6@2e-3 job of the seed (make_jobs_pt6v2.render_v2, frozen; the render
with the registered device / slot must equal the registered job file byte for byte), with counted edits only:
  1. ARGS += --prior_shuf torus,box5 (every other training argument byte-identical; --tag kept, the out dir names the arm);
  2. out dir runs/pt6v2/out/<name> -> runs/pt6v2/out_v9/pt6v2-all6shufTB-lr2e-3-s<seed> (OUT= and --out); Slurm name
     llmadd_pt6v2v9-shufTB-s<seed>; --open-mode=append (a requeue appends to the log);
  3. the job's dry_inputs domain spec: torus / box5 shuf false -> true (prior true, lam 1.0 unchanged);
  4. intent JSON (CHECK_PY): args.prior_shuf = "torus,box5", a new field shuf = ["torus", "box5"];
  5. CHECK_PY: the domains check expects shuf = (d in I["shuf"]) (was hard-coded False); the lambda check
     gnorm_prior_<d> / gnorm_true_<d> skips the shuffled domains (gnorm_prior is the SHUFFLED term's gradient, gnorm_true the true
     term's: their ratio is not lambda by design); CHECK_PY is LOG AND FLAG ONLY ('|| echo ## FLAG V9 ...', as V7b / V8);
  6. device A100 / H100 (a100_tandon,h100_tandon, account <slurm-account>, qos gpu48, as V7 / V8), --time 03:00:00
     (the registered A100 render's); --requeue + --ckpt_every 500 (registered); the V7 early run guard (claim before the first
     write to $OUT); one provenance line.
B200 race copy of each (made as the main session's make_b200_race.sh): the primary text with account <slurm-account>,
--comment="preemption=yes" in place of the partition line, no qos line, --constraint=b200 after --gres, --time 12:00:00, the
device case admitting B200 (and the intent's gpu "A100|H100|B200"), Slurm name llmadd_pt6v2v9-shufTB-s<seed>-b200r. Same OUT:
the run guard (early guard + the trainer's run_guard.claim) lets one trainer compute the run.
Modes: (none) = check and list; --print (header + diff vs the registered job, s16 primary and race); --args (training ARGS);
--write (job files + POSTHOC_V9.json); --test-only; --submit; --after <job id> (afterok dependency on the CPU dry check)."""
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pt6v2"
sys.path.insert(0, str(R))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import posthoc_v7_jobs as P7   # noqa: E402  (read only: its import-time asserts pin every frozen file)
sys.argv = _argv
MJ2, sha, edit = P7.MJ2, P7.sha, P7.edit

OV9 = R / "out_v9"
TR = P7.TR_OLD                      # the registered trainer a398fccbfe93, unchanged
assert sha(R / "code/tools/pretrain_range.py") == TR
SEEDS = tuple(range(16, 24))
ARM, LR, SHUF = "all6", "2e-3", ("torus", "box5")
V9ARM = "all6shufTB"
POSTHOC = R / "POSTHOC_V9.json"
APPROVAL = ('user-approved 2026-11-25 ("可以的，我觉得都可以跑 可以多用一些gpu", relayed by the main session; B200 approved); '
            "post-hoc, exploratory, no gate")
WHY = ("h6 = all6 - only-dim on strict dim area = +13.5 (8/8) at lr 2e-3; all4 (no torus / box5 priors) gives h4 = +1.5, but all4 "
       "also has fewer priors and a smaller total prior gradient (0.095 vs 0.143). V9 keeps data, prior count and prior dose and "
       "only breaks the torus / box5 geometry (dose-matched shuffled targets).")
RULE = ("Exploratory, no gate. Lift from the torus / box5 geometry if h_shuf (all6-shufTB - only-dim, dim strict area) is close to "
        "h4 and all6 - all6-shufTB > 0 in most seeds; lift from the prior dose if h_shuf is close to h6.")
PROV = ("# AUDIT.md Amendment V9 (post-hoc, 2026-11-25): rendered by runs/pt6v2/posthoc_v9_jobs.py = the registered all6@2e-3 job "
        "of this seed (A100 render) + --prior_shuf torus,box5; registered trainer runs/pt6v2/code unchanged; CHECK_PY log and flag "
        "only; out dir runs/pt6v2/out_v9/ ({role})\n")
DW_OLD = 'dw = {d: [d in ACT, False, I["lam_domains"][d]] for d in ALLD}\n'
DW_NEW = 'dw = {d: [d in ACT, d in I["shuf"], I["lam_domains"][d]] for d in ALLD}   # V9: torus / box5 use the shuffled target\n'
LR_OLD = '               and e.get(f"gnorm_true_{d}")] for d in ACT}\n'
LR_NEW = ('               and e.get(f"gnorm_true_{d}")] for d in ACT if d not in I["shuf"]}   # V9: a shuffled prior\'s gnorm_prior is '
          "the shuffled term's, gnorm_true the true term's\n")
SPEC_OLD = "torus:true:false:1.0 box5:true:false:1.0;"
SPEC_NEW = "torus:true:true:1.0 box5:true:true:1.0;"


def race(t, s):
    """the B200 race copy (make_b200_race.sh's header edits + the device case and the intent's GPU list)."""
    t = edit(t, f"#SBATCH --job-name=llmadd_pt6v2v9-shufTB-s{s}\n", f"#SBATCH --job-name=llmadd_pt6v2v9-shufTB-s{s}-b200r\n")
    t = edit(t, "#SBATCH --account=<slurm-account>\n", "#SBATCH --account=<slurm-account>\n")
    t = edit(t, "#SBATCH --partition=a100_tandon,h100_tandon\n", '#SBATCH --comment="preemption=yes"\n')
    t = edit(t, "#SBATCH --qos=gpu48\n", "")
    t = edit(t, "#SBATCH --gres=gpu:1\n", "#SBATCH --gres=gpu:1\n#SBATCH --constraint=b200\n")
    t = edit(t, "#SBATCH --time=03:00:00\n", "#SBATCH --time=12:00:00\n")
    t = edit(t, 'case "$GPU" in *A100*|*H100*) ;;', 'case "$GPU" in *A100*|*H100*|*B200*) ;;')
    t = edit(t, '"gpu":"A100|H100"', '"gpu":"A100|H100|B200"')
    t = edit(t, "(primary, A100 / H100)", "(B200 race copy of the primary; one trainer per run via the run guard)")
    return t


def build():
    doms = MJ2.decided_comp()
    assert MJ2.manifolds(doms) == ("numbers", "dim", "cities", "countries", "torus", "box5"), doms
    alloc = {int(k): v for k, v in json.loads((R / "ALLOC.json").read_text()).items()}
    commit = MJ2.snapshot()
    P = MJ2.load_P()
    RF, _ = MJ2.refs_v2(doms)
    jobs = []
    for s in SEEDS:
        reg = R / f"job_pt6v2-{ARM}-lr{LR}-s{s}.sbatch"
        reg_t = reg.read_text()
        if alloc[s] == "b200":
            slot0 = int(reg_t.split("#SBATCH --job-name=llmadd_pt6v2-b200-slot")[1].split("\n")[0])
            name, tag, t0, _, _ = MJ2.render_v2(P, doms, ARM, LR, s, RF[s], commit, "b200", slot0)
        else:
            name, tag, t0, _, _ = MJ2.render_v2(P, doms, ARM, LR, s, RF[s], commit, alloc[s], None)
        assert t0 == reg_t, f"{reg.name}: render_v2 != the registered job file"   # byte-identity of the existing render
        assert name == f"pt6v2-{ARM}-lr{LR}-s{s}" and tag == f"pt6v2-{ARM}-lr{LR}"
        reg_args = P7.args_line(reg_t)
        rj = P7.run_json(reg_args, s)
        ref = R / "out" / name / tag / rj
        assert ref.exists(), ref
        _, _, ta, _, _ = MJ2.render_v2(P, doms, ARM, LR, s, RF[s], commit, "a100", None)
        assert P7.args_line(ta) == reg_args, f"{name}: the A100 render's ARGS != the registered ARGS"
        assert " --prior_shuf" not in reg_args and " --prior_domains numbers,dim,cities,countries,torus,box5 " in reg_args
        v9name = f"pt6v2-{V9ARM}-lr{LR}-s{s}"
        jn = f"llmadd_pt6v2v9-shufTB-s{s}"
        t = edit(ta, f"#SBATCH --job-name=llmadd_{name}\n", f"#SBATCH --job-name={jn}\n")
        t = edit(t, "#SBATCH --partition=a100_tandon\n", "#SBATCH --partition=a100_tandon,h100_tandon\n")
        t = edit(t, "#SBATCH --requeue\n", "#SBATCH --requeue\n#SBATCH --open-mode=append\n")
        t = edit(t, 'case "$GPU" in *A100*) ;;', 'case "$GPU" in *A100*|*H100*) ;;')
        t = edit(t, f'echo "DEVICE a100 (seed {s}\'s allocated GPU type)"\n',
                 'echo "DEVICE $GPU (V9: A100 / H100, or B200 in the race copy; the device vs the registered run is a flag in the read)"\n')
        t = edit(t, '"gpu":"A100"', '"gpu":"A100|H100"')
        t = edit(t, 'all(I["gpu"] in g for g in gname)', 'all(any(x in g for x in I["gpu"].split("|")) for g in gname)')
        t = edit(t, f"{R}/out/{name}", f"{OV9}/{v9name}", 2)                                  # OUT= and --out
        al = P7.args_line(t)
        new = al.replace(' --ckpt_every 500"', f' --ckpt_every 500 --prior_shuf {",".join(SHUF)}"')
        assert new != al and new.count(" --prior_shuf ") == 1
        t = edit(t, al + "\n", new + "\n")
        assert new == reg_args.replace(f"{R}/out/{name}", f"{OV9}/{v9name}").replace(
            ' --ckpt_every 500"', f' --ckpt_every 500 --prior_shuf {",".join(SHUF)}"')
        t = edit(t, SPEC_OLD, SPEC_NEW)
        old_ln = P7.intent_line(t)
        js = old_ln.split("'")[1]
        I = json.loads(js)
        assert json.dumps(I, separators=(",", ":"), ensure_ascii=False) == js, "intent JSON does not round-trip"
        assert I["code_sha"] == TR and I["steps"] == 18310 and I["args"]["prior_shuf"] == "" and "shuf" not in I
        assert I["args"]["prior_domains"] == "numbers,dim,cities,countries,torus,box5" and I["active"] == list(MJ2.manifolds(doms))
        assert all(I["lam_domains"][d] == 1.0 for d in I["active"])
        I["args"]["prior_shuf"] = ",".join(SHUF)
        I["shuf"] = list(SHUF)
        new_ln = old_ln.replace(js, json.dumps(I, separators=(",", ":"), ensure_ascii=False))
        assert new_ln.endswith(""""$ARGS" <<'PY' | tee "$OUT/check.txt\"""")
        new_ln += (' || echo "## FLAG V9: CHECK_PY (the registered intent checks + 2 V9 edits) failed: $OUT/check.json (log and flag '
                   'only, AUDIT.md Amendment V9)"')
        t = edit(t, old_ln, new_ln)
        t = edit(t, DW_OLD, DW_NEW)
        t = edit(t, LR_OLD, LR_NEW)
        t = edit(t, 'mkdir -p "$OUT"\n', P7.guard_block(rj).replace("V7 run guard", "V9 run guard").replace("# V7 early", "# V9 early")
                 + 'mkdir -p "$OUT"\n')
        t = edit(t, "set -euo pipefail\n", PROV.format(role="primary, A100 / H100") + "set -euo pipefail\n")
        for role, tt, suf in (("primary", t, ""), ("b200race", race(t, s), ".b200r")):
            assert subprocess.run(["bash", "-n"], input=tt.encode()).returncode == 0, f"{v9name}{suf}: bash -n"
            assert f"{R}/out/" not in tt and 'cd "$R/code/tools"\n' in tt and f'= "{TR}" ]' in tt
            assert "V9 run guard" in tt and "run_guard.claim(p)" in tt and tt.index("V9 early run guard") < tt.index('mkdir -p "$OUT"')
            assert P7.args_line(tt).count("--prior_shuf torus,box5") == 1 and SPEC_NEW in tt and DW_NEW in tt and LR_NEW in tt
            hdr = tt.split("set -euo")[0]
            if role == "primary":
                assert "#SBATCH --account=<slurm-account>\n" in hdr and "b200" not in hdr.lower()
            else:
                assert "#SBATCH --account=<slurm-account>\n" in hdr and "#SBATCH --constraint=b200\n" in hdr
                assert '#SBATCH --comment="preemption=yes"\n' in hdr and "--partition" not in hdr and "--qos" not in hdr
                assert "#SBATCH --time=12:00:00\n" in hdr and "*B200*) ;;" in tt
            rl, tl = reg_t.splitlines(), tt.splitlines()
            nd = sum(1 for x in difflib.unified_diff(rl, tl, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---"))
            sl = jn + ("-b200r" if suf else "")
            jobs.append({"name": v9name + suf, "run": v9name, "role": role, "seed": s, "text": tt, "slurm_name": sl,
                         "job": OV9 / f"job_{v9name}{suf}.sbatch", "flag": OV9 / f"SUBMITTED_{v9name}{suf}", "out": str(OV9 / v9name),
                         "run_json": f"{OV9}/{v9name}/{tag}/{rj}", "ref": str(ref), "registered_job": reg.name,
                         "registered_sha12": sha(reg), "intent": I, "lines_vs_registered": nd, "args": new.split('"')[1],
                         "dry_sha": reg_t.split('[ "$got" = ')[1][:12]})
    prim = [j for j in jobs if j["role"] == "primary"]
    assert len({j["name"] for j in jobs}) == len({j["slurm_name"] for j in jobs}) == len(jobs) == 16
    assert len({j["out"] for j in prim}) == 8 and {j["out"] for j in jobs} == {j["out"] for j in prim}   # one out dir per run, shared by its twin
    live = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True).stdout.split()
    clash = [j["slurm_name"] for j in jobs if j["slurm_name"] in live and not j["flag"].exists()]
    assert not clash, f"a live job has this name: {clash}"
    for j in prim:
        o = Path(j["out"])
        if not j["flag"].exists():
            assert not (o.exists() and any(o.rglob("run_whole_*.json"))), f"{o}: a run JSON exists before submission"
    return commit, jobs


def manifest(commit, jobs, ids=None):
    ids = ids or {}
    return {"amendment": "V9", "kind": "post-hoc", "approval": APPROVAL, "why": WHY, "reading_rule": RULE,
            "readout": "runs/pt6v2/out_v9/v9read.py (descriptive; no check gates it)",
            "cells": ["all6-shufTB@2e-3 = all6@2e-3 + --prior_shuf torus,box5"], "seeds": list(SEEDS),
            "reference": "registered runs/pt6v2/out/ all6@2e-3 and only-dim@2e-3, runs/pt6v2/out_v4/ all4@2e-3 (same seeds)",
            "trainer": {"path": "runs/pt6v2/code/tools/pretrain_range.py", "sha12": TR, "changed": False},
            "driver": {"path": "runs/pt6v2/posthoc_v9_jobs.py", "sha12": sha(R / "posthoc_v9_jobs.py"),
                       "parent": "runs/pt6v2/posthoc_v7_jobs.py", "parent_sha12": sha(R / "posthoc_v7_jobs.py")},
            "code_commit": commit, "out_dir": "runs/pt6v2/out_v9",
            "device": "primary A100 / H100 (a100_tandon,h100_tandon); B200 race copy (<slurm-account>, preemptible); "
                      "registered s16..s22 ran on B200, s23 on A100: a flag in the read",
            "jobs": {j["name"]: {"run": j["run"], "role": j["role"], "seed": j["seed"], "slurm_name": j["slurm_name"],
                                 "job_file": str(j["job"].relative_to(R)), "out": j["out"], "run_json": j["run_json"],
                                 "ref_run": j["ref"], "registered_job": j["registered_job"],
                                 "registered_sha12": j["registered_sha12"], "dry_rows_sha12": j["dry_sha"],
                                 "lines_vs_registered": j["lines_vs_registered"],
                                 "job_sha12": hashlib.sha256(j["text"].encode()).hexdigest()[:12],
                                 "job_id": ids.get(j["name"])} for j in jobs}}


def main():
    commit, jobs = build()
    holds = [h for h in [R / "HOLD", R / "HOLD_a100", R / "HOLD_b200", OV9 / "HOLD"] if h.exists()]
    for j in jobs:
        st = ("SUBMITTED" if j["flag"].exists() else "on disk (same)" if j["job"].exists() and j["job"].read_text() == j["text"]
              else "on disk (DIFFERS)" if j["job"].exists() else "not written")
        print(f"{j['name']:36s} {j['slurm_name']:34s} {j['lines_vs_registered']:3d} +/- lines vs {j['registered_job']} "
              f"(byte-identical render asserted)  dry rows {j['dry_sha']}  {st}")
    print(f"check: {len(jobs)} job texts parse (bash -n); unique Slurm names; 8 out dirs (primary + its B200 twin share one); no "
          f"live-name clash; no run JSON in the out dirs; trainer {TR} (registered, unchanged); HOLD: {[h.name for h in holds] or 'none'}")
    if "--print" in sys.argv:
        for j in jobs[:2]:
            print(f"\n==== {j['job'].name}\n" + "".join(x + "\n" for x in j["text"].splitlines() if x.startswith("#SBATCH")))
            sys.stdout.writelines(x[:400] + ("\n" if len(x) > 400 else "") for x in
                                  difflib.unified_diff((R / j["registered_job"]).read_text().splitlines(True), j["text"].splitlines(True), n=0))
    if "--args" in sys.argv:   # the exact training ARGS of every run (for the CPU dry check) + the rows sha its job asserts
        for j in jobs:
            if j["role"] == "primary":
                print(f"ARGS\t{j['run']}\t{j['dry_sha']}\t{j['args']}")
    if "--write" in sys.argv:
        OV9.mkdir(exist_ok=True)
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"]
                continue
            j["job"].write_text(j["text"])
        old = json.loads(POSTHOC.read_text()) if POSTHOC.exists() else None
        assert old is None or all(r["job_id"] is None for r in old["jobs"].values()), "POSTHOC_V9.json holds job ids: not rewritten"
        POSTHOC.write_text(json.dumps(manifest(commit, jobs), indent=1, ensure_ascii=False) + "\n")
        print(f"written: {len(jobs)} job files in {OV9}; {POSTHOC}")
    DEP = [f"--dependency=afterok:{':'.join(str(int(x)) for x in sys.argv[sys.argv.index('--after') + 1].split(':'))}"] if "--after" in sys.argv else []
    if "--test-only" in sys.argv:
        for j in jobs:
            o = subprocess.run(P7.SBATCH + DEP + ["--test-only", str(j["job"])], capture_output=True, text=True)
            print(j["job"].name, (o.stdout + o.stderr).strip()[-160:])
    if "--submit" in sys.argv:
        assert not holds, f"HOLD files exist: {holds}"
        stale = [j["job"].name for j in jobs if not j["flag"].exists() and (not j["job"].exists() or j["job"].read_text() != j["text"])]
        assert not stale, f"job files missing or differing (run --write first): {stale}"
        man = json.loads(POSTHOC.read_text())
        man["sbatch_extra"] = DEP
        for j in jobs:
            if j["flag"].exists():
                print(f"{j['job'].name}: refused (SUBMITTED flag exists)")
                continue
            o = subprocess.run(P7.SBATCH + DEP + [str(j["job"])], capture_output=True, text=True)
            print(j["job"].name, DEP, (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                man["jobs"][j["name"]]["job_id"] = o.stdout.split()[-1]
                POSTHOC.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
