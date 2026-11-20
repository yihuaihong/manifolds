#!/usr/bin/env python3
"""pt6v2 AUDIT.md Amendment V5 (post-hoc, user-approved 2026-11-20): checkpoint-saving reruns of all6@2e-3 and only-dim@2e-3,
seeds 16 and 17 (4 runs), exploratory, no verdict rule. Question: at the same lr 2e-3, dim accuracy drops far less after the prior
window with all six priors on than with only the dim prior (end of training 77.3 vs 52.3) while the logged CKA to dim's target at
the prior layers 2 / 4 (item positions) is retained equally: is the representation in LATER layers or at the ANSWER position
washed out less in the joint model? The runs save model weights (no optimizer) at step 7323 (the LAST step with the prior on:
prior_on(s) = 916 <= s < 7324, the weights after its update), 12000 (an eval step) and 18309 (the final step = the last eval);
runs/pt6v2/probe_v5.py reads them on CPU.

Each job = the registered pt6v2 job of the same cell / seed (make_jobs_pt6v2.render_v2, frozen, sha asserted; the render with the
registered slot must equal the registered job file byte for byte) with counted edits only:
  1. out dir runs/pt6v2/out/<name> -> runs/pt6v2/out_v5/<name> (OUT= and --out; the registered dirs hold complete results);
  2. trainer = the V5 copy runs/pt6v2/out_v5/code (pretrain_range.py a398fccbfe93 + --save_model_at = 041b2f4241b9; the other
     files byte-identical): cd, the trainer sha check, the COMMIT path and its grep, the intent's code_sha;
  3. ARGS += --save_model_at 7323,12000,18309 (every other training argument byte-identical), and the intent's args gain the
     same field (CHECK_PY: run args = intent);
  4. an early run-guard block before the first write to $OUT (racing copies: a job that does not own the run exits before it
     touches $OUT, so a losing copy never overwrites the winner's dry_inputs / gpu.csv / node.txt / check.json);
  5. one provenance comment line; the Slurm job name (B200: singleton chains slot3 / slot4; copy: its own name).
Racing copies (A100 / H100, account <slurm-account>, partition a100_tandon,h100_tandon): the a100 render of the same
cell with the same edits, GPU case *A100*|*H100*; same out dir, so whichever copy claims first computes (run_guard.py).
Independent check: the B200 V5 text = the registered job file with the expected substitutions, every line; the copy text = the
B200 V5 text except the device lines (SBATCH head, header comment, HOLD switch, GPU case / DEVICE line, intent device).

Nothing of the registered batch or of V3 / V4 is touched. Record: runs/pt6v2/POSTHOC_V5.json.
Usage:  posthoc_v5_jobs.py [--check] [--print] [--write] [--test-only] [--submit]
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pt6v2"
OV5 = R / "out_v5"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]   # noqa: E731
MJ2_SHA, CELLS_SHA, V3_SHA, ALLOC_SHA = "bb15d248afd8", "ea9a9b01f081", "75bed7fe30ee", "2d96730f431d"
TR_OLD, TR_NEW = "a398fccbfe93", "041b2f4241b9"
assert sha(B / "make_jobs_pt6v2.py") == MJ2_SHA, "make_jobs_pt6v2.py changed"
assert sha(R / "CELLS.json") == CELLS_SHA and sha(R / "POSTHOC_V3.json") == V3_SHA and sha(R / "ALLOC.json") == ALLOC_SHA
assert sha(R / "code/tools/pretrain_range.py") == TR_OLD and sha(OV5 / "code/tools/pretrain_range.py") == TR_NEW
_same = sorted(p.name for p in (R / "code/tools").iterdir() if p.is_file())
assert _same == sorted(p.name for p in (OV5 / "code/tools").iterdir() if p.is_file()), "V5 code dir: file list differs"
assert all(sha(R / "code/tools" / f) == sha(OV5 / "code/tools" / f) for f in _same if f != "pretrain_range.py"), "V5 code: a dep differs"
sys.path.insert(0, str(B))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_pt6v2 as MJ2   # noqa: E402  (read only)
sys.argv = _argv

LR = "2e-3"
SEEDS = (16, 17)
ARMS = ("all6", "only-dim")
STEPS = 18310
SAVE_AT = (7323, 12000, 18309)
SAVE_ARG = ",".join(map(str, SAVE_AT))
B200_SLOT = {"all6": 3, "only-dim": 4}   # two singleton chains (V4 holds 0-2); the per-user 3-GPU B200 cap still applies
ENV_U = ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_STEP_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_NTASKS")
SBATCH = ["env"] + [x for v in ENV_U for x in ("-u", v)] + ["sbatch"]
POSTHOC = R / "POSTHOC_V5.json"
APPROVAL = ('user-approved 2026-11-20 ("可以的，没问题", relayed by the main session; B200 approved); post-hoc, exploratory, '
            "no verdict rule")
WHY = ("at lr 2e-3 dim accuracy drops far less after the prior window with all six priors than with only the dim prior (end of "
       "training 77.3 vs 52.3) while the logged cka_dim@L2 / @L4 (item positions) is retained equally: is the representation in "
       "later layers or at the answer position washed out less in the joint model? (probe_v5.py on the saved weights)")
PROV = ("# AUDIT.md Amendment V5 (post-hoc, 2026-11-20): rendered by runs/pt6v2/posthoc_v5_jobs.py = the registered job of this cell / "
        "seed + trainer runs/pt6v2/out_v5/code (041b2f4241b9 = a398fccbfe93 + --save_model_at) + --save_model_at {save} (weights at "
        "the last prior-on step / 12000 / final) + an early run guard; out dir runs/pt6v2/out_v5/ ({role})\n")


def window_facts():
    w0, w1 = 0.05, 0.4
    on = [s for s in range(STEPS) if w0 * STEPS <= s < w1 * STEPS]
    assert on[0] == 916 and on[-1] == 7323 == SAVE_AT[0], (on[0], on[-1])
    assert all(s % 250 == 0 for s in SAVE_AT[1:2]) and SAVE_AT[2] == STEPS - 1   # 12000 is an eval step; 18309 is the last eval
    return on[0], on[-1]


def guard_block(seed, tag):
    rj = f"$OUT/$TAG/run_whole_helix3_seed{seed}_{tag}.json"
    return (
        "# V5 early run guard (racing copies share $OUT): claim the run BEFORE the first write to $OUT; a job that does not own it\n"
        "# (complete, or claimed by another RUNNING job) exits 0 here, so a losing copy never overwrites the winner's files. Two-phase:\n"
        "# claim, wait 20 s, re-read the claim (two copies starting together: the last writer keeps it, the other exits). The trainer's\n"
        "# own run_guard.claim then sees this job's id and proceeds; a requeued job (same id) keeps its claim and resumes.\n"
        f'RUNJ="{rj}"\n'
        '"$PY" - "$RUNJ" <<\'PYG\' || { echo "## V5 run guard: this job does not own the run: exiting before touching $OUT"; exit 0; }\n'
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, '.')\n"
        "import run_guard\n"
        "p = Path(sys.argv[1]); me = os.environ.get('SLURM_JOB_ID', '')\n"
        "if not me or not run_guard.claim(p):\n"
        "    sys.exit(1)\n"
        "time.sleep(20)\n"
        "c = p.with_name(p.name + '.claim'); got = c.read_text().strip() if c.exists() else ''\n"
        "print(f'V5 run guard: claim of {p.name} = {got} (me {me})', flush=True)\n"
        "sys.exit(0 if got == me else 1)\n"
        "PYG\n")


def edit(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), n, old[:100])
    return text.replace(old, new)


def intent_line(text):
    ln = [x for x in text.splitlines() if x.startswith('  "$OUT/check.json" \'')]
    assert len(ln) == 1
    return ln[0]


def v5_edits(text, name, seed, tag, role):
    o_out, n_out = f"{R}/out/{name}", f"{OV5}/{name}"
    text = edit(text, o_out, n_out, 2)                                              # OUT= and --out in ARGS
    text = edit(text, 'cd "$R/code/tools"\n', 'cd "$R/out_v5/code/tools"\n')
    text = edit(text, f'[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{TR_OLD}" ]', f'[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{TR_NEW}" ]')
    text = edit(text, f'grep -q "pretrain_range.py {TR_OLD}" "$R/code/COMMIT" || {{ echo "## COMMIT does not name the trainer {TR_OLD}"; exit 3; }}',
                f'grep -q "pretrain_range.py {TR_NEW}" "$R/out_v5/code/COMMIT" || {{ echo "## COMMIT does not name the trainer {TR_NEW}"; exit 3; }}')
    text = edit(text, '"$R/code/COMMIT"', '"$R/out_v5/code/COMMIT"')                 # CHECK_PY's pinned COMMIT argument
    text = edit(text, ' --ckpt_every 500"\n', f' --ckpt_every 500 --save_model_at {SAVE_ARG}"\n')
    old_ln = intent_line(text)
    js = old_ln.split("'")[1]
    I = json.loads(js)
    assert json.dumps(I, separators=(",", ":"), ensure_ascii=False) == js, "intent JSON does not round-trip"
    assert I["code_sha"] == TR_OLD and "save_model_at" not in I["args"] and I["save_model"] is False
    I["code_sha"] = TR_NEW
    I["args"]["save_model_at"] = SAVE_ARG
    text = edit(text, old_ln, old_ln.replace(js, json.dumps(I, separators=(",", ":"), ensure_ascii=False)))
    text = edit(text, 'mkdir -p "$OUT"\n', guard_block(seed, tag) + 'mkdir -p "$OUT"\n')
    text = edit(text, "set -euo pipefail\n", PROV.format(save=SAVE_ARG, role=role) + "set -euo pipefail\n")
    body = "".join(x for x in text.splitlines(True) if not x.startswith("# AUDIT.md Amendment V5"))   # PROV names the parent sha
    assert TR_OLD not in body and "$R/code/" not in text and f"{R}/out/" not in text, name
    return text, I


def build():
    window_facts()
    doms = MJ2.decided_comp()
    assert MJ2.manifolds(doms) == ("numbers", "dim", "cities", "countries", "torus", "box5"), doms
    alloc = {int(k): v for k, v in json.loads((R / "ALLOC.json").read_text()).items()}
    commit = MJ2.snapshot()
    P = MJ2.load_P()
    RF, _ = MJ2.refs_v2(doms)
    jobs = []
    for s in SEEDS:
        assert alloc[s] == "b200", (s, alloc[s])
        for arm in ARMS:
            reg = R / f"job_pt6v2-{arm}-lr{LR}-s{s}.sbatch"
            reg_t = reg.read_text()
            slot0 = int(reg_t.split("#SBATCH --job-name=llmadd_pt6v2-b200-slot")[1].split("\n")[0])
            name, tag, t0, args0, I0 = MJ2.render_v2(P, doms, arm, LR, s, RF[s], commit, "b200", slot0)
            assert t0 == reg_t, f"{reg.name}: render_v2 != the registered job file"
            assert name == f"pt6v2-{arm}-lr{LR}-s{s}" and tag == f"pt6v2-{arm}-lr{LR}"
            _, _, tb, _, _ = MJ2.render_v2(P, doms, arm, LR, s, RF[s], commit, "b200", B200_SLOT[arm])
            tb, Ib = v5_edits(tb, name, s, tag, "B200 main copy")
            # independent check 1: V5 B200 text = the registered file + the expected substitutions, line by line
            exp = reg_t.replace(f"llmadd_pt6v2-b200-slot{slot0}\n", f"llmadd_pt6v2-b200-slot{B200_SLOT[arm]}\n")
            exp = exp.replace(f"singleton slot {slot0} ", f"singleton slot {B200_SLOT[arm]} ")
            exp, _ = v5_edits(exp, name, s, tag, "B200 main copy")
            assert exp == tb, f"{name}: V5 B200 text != registered + substitutions"
            el, rl = tb.splitlines(), reg_t.splitlines()
            # the copy (A100 / H100)
            _, _, ta, _, Ia0 = MJ2.render_v2(P, doms, arm, LR, s, RF[s], commit, "a100", None)
            cname = f"llmadd_pt6v2v5-{arm}-s{s}-copy"
            ta = edit(ta, f"#SBATCH --job-name=llmadd_{name}\n", f"#SBATCH --job-name={cname}\n")
            ta = edit(ta, "#SBATCH --partition=a100_tandon\n", "#SBATCH --partition=a100_tandon,h100_tandon\n")
            ta = edit(ta, 'case "$GPU" in *A100*) ;;', 'case "$GPU" in *A100*|*H100*) ;;')
            ta = edit(ta, f'echo "DEVICE a100 (seed {s}\'s allocated GPU type)"\n',
                      f'echo "DEVICE $GPU (V5 racing copy of seed {s}\'s B200 job: device change vs the registered run, flagged only)"\n')
            ta = edit(ta, '"gpu":"A100"', '"gpu":"A100|H100"')   # the in-job GPU check: either type
            ta = edit(ta, 'all(I["gpu"] in g for g in gname)', 'all(any(x in g for x in I["gpu"].split("|")) for g in gname)')
            ta, Ia = v5_edits(ta, name, s, tag, "A100 / H100 racing copy")
            # independent check 2: the copy differs from the B200 V5 text only in device lines and the intent's device
            al = ta.splitlines()
            dev_ok = lambda x: (x.startswith("#SBATCH") or x.startswith("# pt6v2 main") or x.startswith("# (change the generator")   # noqa: E731
                                or x.startswith("# 1. HOLD") or "HOLD" in x or x.startswith('case "$GPU"') or x.startswith('echo "DEVICE')
                                or x.startswith("# AUDIT.md Amendment V5") or x.startswith('  "$OUT/check.json"')
                                or x.startswith('chk(f"GPU = '))
            ia, ib = [x for x in al if not dev_ok(x)], [x for x in el if not dev_ok(x)]
            assert ia == ib, (name, [(x, y) for x, y in zip(ia, ib) if x != y][:3])
            dI = {k for k in set(Ia) | set(Ib) if Ia.get(k) != Ib.get(k)}
            assert dI <= {"device", "gpu"} and Ia["gpu"] == "A100|H100", (name, dI)
            for t in (tb, ta):
                assert f" --save_model_at {SAVE_ARG}\"" in t and t.count(f"{OV5}/{name}") == 2 and "V5 early run guard" in t
                assert subprocess.run(["bash", "-n"], input=t.encode()).returncode == 0, f"{name}: bash -n"
            import difflib
            ndiff = sum(1 for x in difflib.unified_diff(rl, el, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---"))
            for role, t, jn, f in (("b200", tb, MJ2.slot_name(B200_SLOT[arm]), OV5 / f"job_{name}.sbatch"),
                                   ("copy", ta, cname, OV5 / f"job_{name}_copy.sbatch")):
                jobs.append({"name": name, "arm": arm, "seed": s, "role": role, "text": t, "slurm_name": jn, "job": f,
                             "flag": OV5 / f"SUBMITTED_{f.stem[4:]}", "registered_job": reg.name, "registered_sha12": sha(reg),
                             "intent": Ib if role == "b200" else Ia, "lines_vs_registered": ndiff})
    return commit, jobs


def manifest(commit, jobs, ids=None):
    ids = ids or {}
    return {"amendment": "V5", "kind": "post-hoc", "approval": APPROVAL, "why": WHY,
            "reading_rule": "descriptive (probe_v5.py per run: drift / CKA to dim's exponent vectors / ridge decodability per layer, "
                            "item positions and the answer position); no verdict rule, no gate; readout_pt6v2.py does not read out_v5/",
            "cells": {a: {"lr": LR, "registered": f"{a}@{LR}"} for a in ARMS}, "seeds": list(SEEDS),
            "save_model_at": {"steps": list(SAVE_AT), "meaning": "weights only (model.state_dict()), after step s's update = the weights "
                              "the eval of step s logs; 7323 = the LAST step with the prior on (prior_on(s) = 916 <= s < 7324; no eval "
                              "there: nearest evals 7250 / 7500); 12000 = an eval step; 18309 = the final step and last eval",
                              "file": "out_v5/<name>/<tag>/wts_whole_helix3_seed<s>_<tag>_step<k>.pt"},
            "trainer": {"path": "runs/pt6v2/out_v5/code/tools/pretrain_range.py", "sha12": TR_NEW, "parent_sha12": TR_OLD,
                        "patch": "runs/pt6v2/out_v5/patch_trainer_v5.py", "diff": "runs/pt6v2/out_v5/trainer_v5.diff"},
            "code_commit": commit, "out_dir": "runs/pt6v2/out_v5", "driver": "runs/pt6v2/posthoc_v5_jobs.py",
            "racing": "each run has a B200 job (singleton chain) and an A100 / H100 copy (a100_tandon,h100_tandon) on the same out dir; "
                      "the early run guard + run_guard.py let the first to claim compute; the other exits 0 before touching $OUT",
            "runs": {f"{j['name']}:{j['role']}": {"seed": j["seed"], "arm": j["arm"], "role": j["role"], "slurm_name": j["slurm_name"],
                                                    "job_file": str(j["job"].relative_to(R)),
                                                    "job_sha12": hashlib.sha256(j["text"].encode()).hexdigest()[:12],
                                                    "registered_job": j["registered_job"], "registered_sha12": j["registered_sha12"],
                                                    "job_id": ids.get(f"{j['name']}:{j['role']}")} for j in jobs}}


def main():
    commit, jobs = build()
    holds = [h for h in [R / "HOLD", R / "HOLD_b200", R / "HOLD_a100"] if h.exists()]
    for j in jobs:
        st = ("SUBMITTED" if j["flag"].exists() else "on disk (same)" if j["job"].exists() and j["job"].read_text() == j["text"]
              else "on disk (DIFFERS)" if j["job"].exists() else "not written")
        print(f"{j['name']:26s} {j['role']:4s} {j['slurm_name']:34s} {j['lines_vs_registered']} +/- diff lines (B200 V5 vs {j['registered_job']})  {st}")
    print(f"check: {len(jobs)} job texts parse; B200 text = registered job + counted substitutions; copy = B200 text except device "
          f"lines; frozen files unchanged; HOLD files: {[h.name for h in holds] or 'none'}")
    if "--print" in sys.argv:
        for j in jobs[:2]:
            print(f"\n==== {j['job'].name}\n" + "".join(x + "\n" for x in j["text"].splitlines() if x.startswith("#SBATCH")))
    if "--write" in sys.argv:
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"]
                continue
            j["job"].write_text(j["text"])
        old = json.loads(POSTHOC.read_text()) if POSTHOC.exists() else None
        assert old is None or all(r["job_id"] is None for r in old["runs"].values()), "POSTHOC_V5.json holds job ids: not rewritten"
        POSTHOC.write_text(json.dumps(manifest(commit, jobs), indent=1, ensure_ascii=False) + "\n")
        print(f"written: {len(jobs)} job files in {OV5}; {POSTHOC}")
    if "--test-only" in sys.argv:
        for j in jobs:
            o = subprocess.run(SBATCH + ["--test-only", str(j["job"])], capture_output=True, text=True)
            print(j["job"].name, (o.stdout + o.stderr).strip())
    if "--submit" in sys.argv:
        assert not holds, f"HOLD files exist: {holds}"
        stale = [j["job"].name for j in jobs if not j["flag"].exists() and (not j["job"].exists() or j["job"].read_text() != j["text"])]
        assert not stale, f"job files missing or differing (run --write first): {stale}"
        man = json.loads(POSTHOC.read_text())
        for j in jobs:
            k = f"{j['name']}:{j['role']}"
            if j["flag"].exists():
                print(f"{j['job'].name}: refused (SUBMITTED flag exists)")
                continue
            o = subprocess.run(SBATCH + [str(j["job"])], capture_output=True, text=True)
            print(j["job"].name, (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                man["runs"][k]["job_id"] = o.stdout.split()[-1]
                POSTHOC.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
