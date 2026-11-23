#!/usr/bin/env python3
"""pt6v2 Amendment V8 (post-hoc, user-approved 2026-11-23; exp-builder): longer training. Does a model without the joint priors
catch up with all6@2e-3 (1x, 18,310 steps) if it trains 2x / 4x as long?
Cells (seeds 16, 17 = paired with the registered runs): only-dim@2e-3 and none@3.2e-2 (N_m = the area-best none lr for dim,
runs/pt6v2/tests/audit_speed_20261120.md) at x2 (36,620 steps) and x4 (73,240 steps) = 8 runs.

Each job = the A100 render of the registered pt6v2 job of the arm / seed (make_jobs_pt6v2.render_v2, frozen; the render with the
registered device / slot must equal the registered job file byte for byte), REGISTERED TRAINER UNCHANGED (runs/pt6v2/code,
a398fccbfe93), with counted edits only:
  1. ARGS += --tokens <k x 18310 x 16384> (steps = int(tokens // (32 * 512)) = k x 18310 exactly); only-dim: --window 0.05,0.4 ->
     0.05/k,0.4/k (same absolute window: prior on 916..7323, off from 7324; asserted with the trainer's own comparison
     w0 * steps <= s < w1 * steps); none has no --window (no prior). Every other training argument byte-identical. The cosine LR
     schedule and the warmup (steps // 100) stretch with the steps (intended).
  2. out dir runs/pt6v2/out/<name> -> runs/pt6v2/out_v8/<arm>-x<k>-s<seed> (OUT= and --out); Slurm name llmadd_pt6v2v8-<arm>-x<k>-s<seed>;
  3. intent JSON (CHECK_PY): args.tokens / args.window, steps, evals (every 250 + last), on_frac = 6408 / steps; events unchanged;
  4. CHECK_PY is LOG AND FLAG ONLY (its length-dependent checks must not fail a long run): '|| echo ## FLAG V8 ...';
  5. device A100 / H100 (a100_tandon,h100_tandon, as V7 / V7b); --time 04:00:00 (x2) / 07:00:00 (x4); --requeue + --ckpt_every 500
     (registered) so a requeued job resumes; the V7 early run guard (claim before the first write to $OUT); one provenance line.
Modes: (none) = check and list; --print (header + diff vs the registered job for two jobs); --write (job files + POSTHOC_V8.json);
--test-only (sbatch --test-only every job file); --submit (all 8; SUBMITTED_<name> flags)."""
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

OV8 = R / "out_v8"
TR = P7.TR_OLD                      # the registered trainer a398fccbfe93, unchanged
assert sha(R / "code/tools/pretrain_range.py") == TR
SEEDS = (16, 17)
KS = (2, 4)
STEPS1, BS_SEQ = 18310, 32 * 512
ARMS = (("only-dim", "2e-3", "onlydim"), ("none", "3.2e-2", "none"))   # (arm, lr, short); none@3.2e-2 = N_m for dim
WIN = {2: "0.025,0.2", 4: "0.0125,0.1"}
TIME = {2: "04:00:00", 4: "07:00:00"}
EVENTS = [[916, 1], [7324, 0]]
POSTHOC = R / "POSTHOC_V8.json"
APPROVAL = "user-approved 2026-11-23 (relayed by the main session); post-hoc, exploratory, no gate"
WHY = ("at 1x all6@2e-3 ends ~25 pts above only-dim@2e-3 on strict dim (final3), and only-dim loses 32.9 pts from the window's end "
       "to final3 (F105 addendum 2). Question: does a model without the joint priors catch up with all6 1x if it trains 2x / 4x longer?")
RULE = ("Report each arm's final3 strict dim at x2 / x4 against all6 1x; 'catches up' = mean >= all6 1x mean with both seeds within "
        "all6's 8-seed range; no gate.")
PROV = ("# AUDIT.md Amendment V8 (post-hoc, 2026-11-23): rendered by runs/pt6v2/posthoc_v8_jobs.py = the registered {arm}@{lr} job of "
        "this seed (A100 render) + --tokens {tok} ({steps} steps = x{k}){win}; registered trainer runs/pt6v2/code unchanged; CHECK_PY "
        "log and flag only; out dir runs/pt6v2/out_v8/\n")


def consumed(k, arm):
    """the trainer's own expressions: steps = int(tokens // (bs * seq)); prior_on(s) = w0 * steps <= s < w1 * steps."""
    tok = k * STEPS1 * BS_SEQ
    steps = int(float(tok) // BS_SEQ)
    assert steps == k * STEPS1, steps
    if arm == "none":
        return tok, steps, None, []
    w0, w1 = (float(v) for v in WIN[k].split(","))
    on = [s for s in range(steps) if w0 * steps <= s < w1 * steps]
    ev = [[on[0], 1], [on[-1] + 1, 0]]
    assert ev == EVENTS and on == list(range(916, 7324)), ev
    return tok, steps, WIN[k], ev


def build():
    doms = MJ2.decided_comp()
    assert MJ2.manifolds(doms) == ("numbers", "dim", "cities", "countries", "torus", "box5"), doms
    alloc = {int(k): v for k, v in json.loads((R / "ALLOC.json").read_text()).items()}
    commit = MJ2.snapshot()
    P = MJ2.load_P()
    RF, _ = MJ2.refs_v2(doms)
    jobs = []
    for arm, lr, short in ARMS:
        for s in SEEDS:
            reg = R / f"job_pt6v2-{arm}-lr{lr}-s{s}.sbatch"
            reg_t = reg.read_text()
            assert alloc[s] == "b200"
            slot0 = int(reg_t.split("#SBATCH --job-name=llmadd_pt6v2-b200-slot")[1].split("\n")[0])
            name, tag, t0, _, _ = MJ2.render_v2(P, doms, arm, lr, s, RF[s], commit, "b200", slot0)
            assert t0 == reg_t, f"{reg.name}: render_v2 != the registered job file"
            assert name == f"pt6v2-{arm}-lr{lr}-s{s}" and tag == f"pt6v2-{arm}-lr{lr}"
            reg_args = P7.args_line(reg_t)
            rj = P7.run_json(reg_args, s)
            ref = R / "out" / name / tag / rj
            assert ref.exists(), ref
            _, _, ta, _, _ = MJ2.render_v2(P, doms, arm, lr, s, RF[s], commit, "a100", None)
            assert P7.args_line(ta) == reg_args, f"{name}: the A100 render's ARGS != the registered ARGS"
            for k in KS:
                tok, steps, win, ev = consumed(k, arm)
                v8name = f"pt6v2-{arm}-lr{lr}-x{k}-s{s}"
                jn = f"llmadd_pt6v2v8-{short}-x{k}-s{s}"
                t = edit(ta, f"#SBATCH --job-name=llmadd_{name}\n", f"#SBATCH --job-name={jn}\n")
                t = edit(t, "#SBATCH --partition=a100_tandon\n", "#SBATCH --partition=a100_tandon,h100_tandon\n")
                t = edit(t, "#SBATCH --time=03:00:00\n", f"#SBATCH --time={TIME[k]}\n")
                t = edit(t, "#SBATCH --requeue\n", "#SBATCH --requeue\n#SBATCH --open-mode=append\n")   # as the registered job: a requeue appends to the log
                t = edit(t, 'case "$GPU" in *A100*) ;;', 'case "$GPU" in *A100*|*H100*) ;;')
                t = edit(t, f'echo "DEVICE a100 (seed {s}\'s allocated GPU type)"\n',
                         'echo "DEVICE $GPU (V8: A100 / H100 for every run; the device vs the registered B200 run is a flag in the read)"\n')
                t = edit(t, '"gpu":"A100"', '"gpu":"A100|H100"')
                t = edit(t, 'all(I["gpu"] in g for g in gname)', 'all(any(x in g for x in I["gpu"].split("|")) for g in gname)')
                t = edit(t, f"{R}/out/{name}", f"{OV8}/{v8name}", 2)                                  # OUT= and --out
                al = P7.args_line(t)
                new = al.replace(' --ckpt_every 500"', f' --ckpt_every 500 --tokens {tok}"')
                if win:
                    assert new.count(" --window 0.05,0.4 ") == 1
                    new = new.replace(" --window 0.05,0.4 ", f" --window {win} ")
                else:
                    assert " --window" not in new and " --schedule" not in new
                assert new != al and new.count(" --ckpt_every 500 --tokens ") == 1
                t = edit(t, al + "\n", new + "\n")
                assert new == reg_args.replace(f"{R}/out/{name}", f"{OV8}/{v8name}").replace(' --ckpt_every 500"', f' --ckpt_every 500 --tokens {tok}"').replace(
                    " --window 0.05,0.4 ", f" --window {win} " if win else " --window 0.05,0.4 ")
                old_ln = P7.intent_line(t)
                js = old_ln.split("'")[1]
                I = json.loads(js)
                assert json.dumps(I, separators=(",", ":"), ensure_ascii=False) == js, "intent JSON does not round-trip"
                assert I["code_sha"] == TR and I["steps"] == STEPS1 and I["evals"] == 75 and I["events"] == ev
                assert I["args"]["tokens"] == 3e8 and I["args"]["window"] == "0.05,0.4" and I["args"]["eval_every"] == 250
                I["args"]["tokens"] = float(tok)
                if win:
                    I["args"]["window"] = win
                I["steps"] = steps
                I["evals"] = len([x for x in range(steps) if x % 250 == 0 or x == steps - 1])
                if ev:
                    assert abs(I["on_frac"] - 6408 / STEPS1) < 1e-12
                    I["on_frac"] = 6408 / steps
                else:
                    assert I["on_frac"] == 0.0
                new_ln = old_ln.replace(js, json.dumps(I, separators=(",", ":"), ensure_ascii=False))
                assert new_ln.endswith(""""$ARGS" <<'PY' | tee "$OUT/check.txt\"""")
                new_ln += (' || echo "## FLAG V8: CHECK_PY (the registered intent checks) failed: $OUT/check.json (log and flag only, '
                           'AUDIT.md Amendment V8)"')
                t = edit(t, old_ln, new_ln)
                t = edit(t, 'mkdir -p "$OUT"\n', P7.guard_block(rj).replace("V7 run guard", "V8 run guard").replace("# V7 early", "# V8 early")
                         + 'mkdir -p "$OUT"\n')
                t = edit(t, "set -euo pipefail\n", PROV.format(arm=arm, lr=lr, tok=tok, steps=steps, k=k,
                                                                win=f", --window {win} (prior on 916..7323)" if win else ", no prior")
                         + "set -euo pipefail\n")
                assert subprocess.run(["bash", "-n"], input=t.encode()).returncode == 0, f"{v8name}: bash -n"
                assert "#SBATCH --account=<slurm-account>\n" in t and "b200" not in t.split("set -euo")[0].lower()
                assert f"{R}/out/" not in t and 'cd "$R/code/tools"\n' in t and f'= "{TR}" ]' in t
                rl, tl = reg_t.splitlines(), t.splitlines()
                nd = sum(1 for x in difflib.unified_diff(rl, tl, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---"))
                jobs.append({"name": v8name, "arm": arm, "lr": lr, "k": k, "seed": s, "text": t, "slurm_name": jn,
                             "job": OV8 / f"job_{v8name}.sbatch", "flag": OV8 / f"SUBMITTED_{v8name}", "out": str(OV8 / v8name),
                             "run_json": f"{OV8}/{v8name}/{tag}/{rj}", "ref": str(ref), "registered_job": reg.name,
                             "registered_sha12": sha(reg), "intent": I, "lines_vs_registered": nd, "tokens": tok, "steps": steps,
                             "window": win, "events": ev, "args": new.split('"')[1]})
    assert len({j["name"] for j in jobs}) == len({j["slurm_name"] for j in jobs}) == len({j["out"] for j in jobs}) == len(jobs) == 8
    live = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True).stdout.split()
    clash = [j["slurm_name"] for j in jobs if j["slurm_name"] in live and not j["flag"].exists()]
    assert not clash, f"a live job has this name: {clash}"
    for j in jobs:
        o = Path(j["out"])
        if not j["flag"].exists():
            assert not (o.exists() and any(o.rglob("run_whole_*.json"))), f"{o}: a run JSON exists before submission"
    return commit, jobs


def manifest(commit, jobs, ids=None):
    ids = ids or {}
    return {"amendment": "V8", "kind": "post-hoc", "approval": APPROVAL, "why": WHY, "reading_rule": RULE,
            "readout": "runs/pt6v2/out_v8/v8read.py (descriptive; no check gates it)",
            "cells": ["only-dim@2e-3 x2 / x4", "none@3.2e-2 (N_m, area-best none lr for dim) x2 / x4"], "seeds": list(SEEDS),
            "reference": "registered 1x runs runs/pt6v2/out/: all6@2e-3, only-dim@2e-3, none@* (8 seeds each, 18,310 steps)",
            "trainer": {"path": "runs/pt6v2/code/tools/pretrain_range.py", "sha12": TR, "changed": False},
            "schedule_note": "cosine LR and warmup (steps // 100) stretch with the steps; prior window kept at absolute steps 916..7323",
            "driver": {"path": "runs/pt6v2/posthoc_v8_jobs.py", "sha12": sha(R / "posthoc_v8_jobs.py"),
                       "parent": "runs/pt6v2/posthoc_v7_jobs.py", "parent_sha12": sha(R / "posthoc_v7_jobs.py")},
            "code_commit": commit, "out_dir": "runs/pt6v2/out_v8",
            "device": "A100 / H100 (a100_tandon,h100_tandon) as V7 / V7b; registered seeds 16 / 17 ran on B200: a flag in the read",
            "runs": {j["name"]: {"arm": j["arm"], "lr": j["lr"], "x": j["k"], "seed": j["seed"], "slurm_name": j["slurm_name"],
                                 "tokens": j["tokens"], "steps": j["steps"], "window": j["window"], "events": j["events"],
                                 "job_file": str(j["job"].relative_to(R)), "out": j["out"], "run_json": j["run_json"],
                                 "ref_run": j["ref"], "registered_job": j["registered_job"],
                                 "registered_sha12": j["registered_sha12"],
                                 "job_sha12": hashlib.sha256(j["text"].encode()).hexdigest()[:12],
                                 "job_id": ids.get(j["name"])} for j in jobs}}


def main():
    commit, jobs = build()
    holds = [h for h in [R / "HOLD", R / "HOLD_a100", OV8 / "HOLD"] if h.exists()]
    for j in jobs:
        st = ("SUBMITTED" if j["flag"].exists() else "on disk (same)" if j["job"].exists() and j["job"].read_text() == j["text"]
              else "on disk (DIFFERS)" if j["job"].exists() else "not written")
        print(f"{j['name']:32s} {j['slurm_name']:30s} steps {j['steps']:6d} window {j['window']} events {j['events']} "
              f"{j['lines_vs_registered']:3d} +/- lines vs {j['registered_job']}  {st}")
    print(f"check: {len(jobs)} job texts parse (bash -n); unique names / out dirs, no live-name clash, no run JSON in the out dirs; "
          f"trainer {TR} (registered, unchanged); HOLD: {[h.name for h in holds] or 'none'}")
    if "--print" in sys.argv:
        for j in jobs[:1] + jobs[-1:]:
            print(f"\n==== {j['job'].name}\n" + "".join(x + "\n" for x in j["text"].splitlines() if x.startswith("#SBATCH")))
            sys.stdout.writelines(x[:400] + ("\n" if len(x) > 400 else "") for x in
                                  difflib.unified_diff((R / j["registered_job"]).read_text().splitlines(True), j["text"].splitlines(True), n=0))
    if "--args" in sys.argv:   # the exact training ARGS of every job (for the CPU dry check)
        for j in jobs:
            print(f"ARGS\t{j['name']}\t{j['args']}")
    if "--write" in sys.argv:
        OV8.mkdir(exist_ok=True)
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"]
                continue
            j["job"].write_text(j["text"])
        old = json.loads(POSTHOC.read_text()) if POSTHOC.exists() else None
        assert old is None or all(r["job_id"] is None for r in old["runs"].values()), "POSTHOC_V8.json holds job ids: not rewritten"
        POSTHOC.write_text(json.dumps(manifest(commit, jobs), indent=1, ensure_ascii=False) + "\n")
        print(f"written: {len(jobs)} job files in {OV8}; {POSTHOC}")
    # --after <job id>: every job waits for the CPU dry check (afterok; that job exits 1 when V8 DRY FAILs, the GPU jobs then never start)
    DEP = [f"--dependency=afterok:{int(sys.argv[sys.argv.index('--after') + 1])}"] if "--after" in sys.argv else []
    if "--test-only" in sys.argv:
        for j in jobs:
            o = subprocess.run(P7.SBATCH + DEP + ["--test-only", str(j["job"])], capture_output=True, text=True)
            print(j["job"].name, (o.stdout + o.stderr).strip())
    if "--submit" in sys.argv:
        assert not holds, f"HOLD files exist: {holds}"
        stale = [j["job"].name for j in jobs if not j["flag"].exists() and (not j["job"].exists() or j["job"].read_text() != j["text"])]
        assert not stale, f"job files missing or differing (run --write first): {stale}"
        man = json.loads(POSTHOC.read_text())
        for j in jobs:
            if j["flag"].exists():
                print(f"{j['job'].name}: refused (SUBMITTED flag exists)")
                continue
            o = subprocess.run(P7.SBATCH + DEP + [str(j["job"])], capture_output=True, text=True)
            print(j["job"].name, DEP, (o.stdout + o.stderr).strip())
            man["sbatch_extra"] = DEP
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                man["runs"][j["name"]]["job_id"] = o.stdout.split()[-1]
                POSTHOC.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
