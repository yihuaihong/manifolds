#!/usr/bin/env python3
"""pt6v2 AUDIT.md Amendment V7 (post-hoc, user-approved 2026-11-22 "可以的"): all6@2e-3 with the torus and box5 problem lines
removed from the training stream from step 7324 on (the first step after the prior window, prior_on(s) = 916 <= s < 7324),
seeds 16..23 (8 runs), exploratory, no verdict rule. Comparison: the registered all6@2e-3 and only-dim@2e-3 runs, same seeds.

Each job = the A100 render of the registered pt6v2 job of all6@2e-3 / seed (make_jobs_pt6v2.render_v2, frozen, sha asserted; the
render with the registered device / slot must equal the registered job file byte for byte) with counted edits only:
  1. out dir runs/pt6v2/out/<name> -> runs/pt6v2/out_v7/pt6v2-all6drop-lr2e-3-s<seed> (OUT= and --out);
  2. trainer = the V7 copy runs/pt6v2/out_v7/code (a398fccbfe93 + --drop_domains_after = 8da22c6ea208; the other files
     byte-identical): cd, the trainer sha check, the COMMIT path and its grep, the intent's code_sha;
  3. ARGS += --drop_domains_after torus,box5@7324 (every other training argument byte-identical); intent args gain the field;
  4. CHECK_PY: the slot rule is read at the last eval before the drop; geo_<d> of a dropped domain is counted over the evals
     before the drop (both: the registered rule where it still holds; check_v7.py checks the rest); intent gains drop / drop_at;
  5. check_v7.py (sha pinned) after training -> $OUT/check_v7.json, --main --ref <the registered run JSON of the seed>; exit 5 on
     a fail (after CHECK_PY);
  6. an early run-guard block before the first write to $OUT (the run JSON name derived from the ARGS' --tokenization / --target
     / --seed / --tag, asserted = the registered CHECK_PY's run path); one provenance line; job name llmadd_pt6v2v7-all6drop-s<seed>;
  7. device: A100 or H100 (a100_tandon,h100_tandon, account <slurm-account>); the device vs the registered run is a
     flag in the read, not a gate. One job per run (no racing twins).
Canary llmadd_pt6v2v7-canary-s16: the s16 text + --tokens 13107200 (800 steps, window 40..319) and the drop at 320 (the first
step after ITS window); dry_inputs with the production ARGS; CHECK_PY non-fatal (a short run fails its length checks by design);
check_v7.py fatal. Record: runs/pt6v2/POSTHOC_V7.json.
Usage:  posthoc_v7_jobs.py [--check] [--print] [--write] [--test-only] [--submit]
"""
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pt6v2"
OV7 = R / "out_v7"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]   # noqa: E731
MJ2_SHA, CELLS_SHA, V3_SHA, ALLOC_SHA = "bb15d248afd8", "ea9a9b01f081", "75bed7fe30ee", "2d96730f431d"
TR_OLD, TR_NEW = "a398fccbfe93", "8da22c6ea208"
assert sha(B / "make_jobs_pt6v2.py") == MJ2_SHA, "make_jobs_pt6v2.py changed"
assert sha(R / "CELLS.json") == CELLS_SHA and sha(R / "POSTHOC_V3.json") == V3_SHA and sha(R / "ALLOC.json") == ALLOC_SHA
assert sha(R / "code/tools/pretrain_range.py") == TR_OLD and sha(OV7 / "code/tools/pretrain_range.py") == TR_NEW
_same = sorted(p.name for p in (R / "code/tools").iterdir() if p.is_file())
assert _same == sorted(p.name for p in (OV7 / "code/tools").iterdir() if p.is_file()), "V7 code dir: file list differs"
assert all(sha(R / "code/tools" / f) == sha(OV7 / "code/tools" / f) for f in _same if f != "pretrain_range.py"), "V7 code: a dep differs"
CV7 = sha(OV7 / "check_v7.py")
sys.path.insert(0, str(B))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import make_jobs_pt6v2 as MJ2   # noqa: E402  (read only)
sys.argv = _argv

LR, ARM = "2e-3", "all6"
SEEDS = tuple(range(16, 24))
STEPS, DROP_AT, DROP = 18310, 7324, ("torus", "box5")
FLAG = f"{','.join(DROP)}@{DROP_AT}"
CAN_TOK, CAN_STEPS, CAN_AT = 13107200, 800, 320
ENV_U = ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_STEP_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_NTASKS")
SBATCH = ["env"] + [x for v in ENV_U for x in ("-u", v)] + ["sbatch"]
POSTHOC = R / "POSTHOC_V7.json"
APPROVAL = 'user-approved 2026-11-22 ("可以的", relayed by the main session); post-hoc, exploratory, no verdict rule'
WHY = ("at lr 2e-3 all6 beats only-dim on strict dim by +25 pts at the end of training (8/8 seeds), the gain appears after the "
       "window and needs the torus / box5 priors (all4 ~ only-dim, F106); V5 probes (F107): only-dim still encodes the answer at "
       "the answer position but fails to output it. Hypothesis: torus / box5 share a lattice answer code with dim and post-window LM "
       "training on their lines keeps dim's state->token readout working. Prediction (hypothesis): without torus / box5 lines after "
       "the window, dim's end-of-training advantage over only-dim largely disappears.")
PROV = ("# AUDIT.md Amendment V7 (post-hoc, 2026-11-22): rendered by runs/pt6v2/posthoc_v7_jobs.py = the registered all6@2e-3 job of "
        "this seed (A100 render) + trainer runs/pt6v2/out_v7/code (8da22c6ea208 = a398fccbfe93 + --drop_domains_after) + "
        "--drop_domains_after {flag} + check_v7.py + an early run guard; out dir runs/pt6v2/out_v7/ ({role})\n")


def window_facts(steps, at):
    on = [s for s in range(steps) if 0.05 * steps <= s < 0.4 * steps]
    assert on[-1] + 1 == at, (on[0], on[-1], at)
    return on[0], on[-1]


def edit(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), n, old[:100])
    return text.replace(old, new)


def intent_line(text):
    ln = [x for x in text.splitlines() if x.startswith('  "$OUT/check.json" \'')]
    assert len(ln) == 1
    return ln[0]


def args_line(text):
    ln = [x for x in text.splitlines() if x.startswith('ARGS="')]
    assert len(ln) == 1
    return ln[0]


def run_json(args, seed):
    """the trainer's run JSON name, from the ARGS (run_<tokenization>_<target>_seed<seed><tag>.json)."""
    a = args.split('"')[1].split()
    g = lambda k: a[a.index(k) + 1]   # noqa: E731
    assert int(g("--seed")) == seed
    return f"run_{g('--tokenization')}_{g('--target')}_seed{seed}{g('--tag')}.json"


def guard_block(rj):
    return (
        "# V7 early run guard: claim the run BEFORE the first write to $OUT; a job that does not own it (complete, or claimed by\n"
        "# another RUNNING job) exits 0 here. Two-phase: claim, wait 20 s, re-read the claim. The trainer's own run_guard.claim then\n"
        "# sees this job's id and proceeds; a requeued job (same id) keeps its claim and resumes.\n"
        f'RUNJ="$OUT/$TAG/{rj}"\n'
        '"$PY" - "$RUNJ" <<\'PYG\' || { echo "## V7 run guard: this job does not own the run: exiting before touching $OUT"; exit 0; }\n'
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, '.')\n"
        "import run_guard\n"
        "p = Path(sys.argv[1]); me = os.environ.get('SLURM_JOB_ID', '')\n"
        "if not me or not run_guard.claim(p):\n"
        "    sys.exit(1)\n"
        "time.sleep(20)\n"
        "c = p.with_name(p.name + '.claim'); got = c.read_text().strip() if c.exists() else ''\n"
        "print(f'V7 run guard: claim of {p.name} = {got} (me {me})', flush=True)\n"
        "sys.exit(0 if got == me else 1)\n"
        "PYG\n")


SLOT_OLD = 'sl = c[-1]["slot_frac"]\n'
SLOT_NEW = ('sl = [e for e in c if e["step"] < I["drop_at"]][-1]["slot_frac"]   # V7: the registered rule at the last eval before '
            'the drop (check_v7.py: after it)\n')
SLOTN_OLD = 'chk(f"domain lines per slot after the numbers line: K = '
SLOTN_NEW = 'chk(f"(V7: last eval before the drop) domain lines per slot after the numbers line: K = '
GEO_OLD = ('    ng = {d: sum(e.get(f"geo_{d}") is not None for e in c) for d in EXT}\n'
           '    chk("geo_<d> logged at >= 95% of the evals for the active prior, never for an inactive one",\n'
           '        all((n >= 0.95 * len(c)) if d in ACT else (n == 0) for d, n in ng.items()), ng)\n')
GEO_NEW = ('    cd_ = lambda d: [e for e in c if e["step"] < I["drop_at"]] if d in I["drop"] else c   # V7: a dropped domain has no sites after the drop\n'
           '    ng = {d: sum(e.get(f"geo_{d}") is not None for e in cd_(d)) for d in EXT}\n'
           '    chk("geo_<d> logged at >= 95% of the evals for the active prior (V7: a dropped domain, of the evals before the drop), never for an inactive one",\n'
           '        all((n >= 0.95 * len(cd_(d))) if d in ACT else (n == 0) for d, n in ng.items()), ng)\n')


def v7_edits(text, name, v7name, seed, role, flag, ref, can=False):
    reg_args = args_line(text)
    rj = run_json(reg_args, seed)
    assert f'"$PY" - "$OUT/$TAG/{rj}" ' in text, "the derived run JSON name != the registered CHECK_PY's run path"
    text = edit(text, f"{R}/out/{name}", f"{OV7}/{v7name}", 2)                       # OUT= and --out in ARGS
    text = edit(text, 'cd "$R/code/tools"\n', 'cd "$R/out_v7/code/tools"\n')
    text = edit(text, f'[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{TR_OLD}" ]', f'[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{TR_NEW}" ]')
    text = edit(text, f'grep -q "pretrain_range.py {TR_OLD}" "$R/code/COMMIT" || {{ echo "## COMMIT does not name the trainer {TR_OLD}"; exit 3; }}\n',
                f'grep -q "pretrain_range.py {TR_NEW}" "$R/out_v7/code/COMMIT" || {{ echo "## COMMIT does not name the trainer {TR_NEW}"; exit 3; }}\n'
                f'[ "$(sha256sum "$R/out_v7/check_v7.py" | cut -c1-12)" = "{CV7}" ] || {{ echo "## check_v7.py sha mismatch: exiting"; exit 3; }}\n')
    text = edit(text, '"$R/code/COMMIT"', '"$R/out_v7/code/COMMIT"')                 # CHECK_PY's pinned COMMIT argument
    extra = f" --drop_domains_after {flag}" + (f" --tokens {CAN_TOK}" if can else "")
    text = edit(text, ' --ckpt_every 500"\n', f' --ckpt_every 500{extra}"\n')
    new_args = args_line(text)
    assert new_args == reg_args.replace(f"{R}/out/{name}", f"{OV7}/{v7name}").replace(' --ckpt_every 500"', f' --ckpt_every 500{extra}"')
    if can:   # dry_inputs with the production ARGS (the 63-line reference); training with the canary's
        prod = reg_args.replace(f"{R}/out/{name}", f"{OV7}/{v7name}").replace(' --ckpt_every 500"', f' --ckpt_every 500 --drop_domains_after {FLAG}"')
        text = edit(text, new_args + "\n", new_args + "\n" + "ARGS_DRY=" + prod[len("ARGS="):] + "   # V7 canary: dry_inputs = production ARGS\n")
        text = edit(text, '"$PY" pretrain_range.py $ARGS --dry_inputs', '"$PY" pretrain_range.py $ARGS_DRY --dry_inputs')
    old_ln = intent_line(text)
    js = old_ln.split("'")[1]
    I = json.loads(js)
    assert json.dumps(I, separators=(",", ":"), ensure_ascii=False) == js, "intent JSON does not round-trip"
    assert I["code_sha"] == TR_OLD and "drop_domains_after" not in I["args"] and I["steps"] == STEPS
    I["code_sha"] = TR_NEW
    I["args"]["drop_domains_after"] = flag
    I["drop"], I["drop_at"] = list(DROP), int(flag.split("@")[1])
    if can:
        I["args"]["tokens"] = float(CAN_TOK) if isinstance(I["args"].get("tokens"), float) else CAN_TOK
        I["steps"] = CAN_STEPS
        I["evals"] = len([s for s in range(CAN_STEPS) if s % 250 == 0 or s == CAN_STEPS - 1])
    new_ln = old_ln.replace(js, json.dumps(I, separators=(",", ":"), ensure_ascii=False))
    assert new_ln.endswith(""""$ARGS" <<'PY' | tee "$OUT/check.txt\"""")
    if can:
        new_ln += ' || echo "## V7 canary: CHECK_PY failed (a short run fails its length-dependent checks by design; see check.json)"'
    text = edit(text, old_ln, new_ln)
    text = edit(text, SLOT_OLD, SLOT_NEW)
    text = edit(text, SLOTN_OLD, SLOTN_NEW)
    text = edit(text, GEO_OLD, GEO_NEW)
    text = edit(text, 'echo "## 4. intent checks $(date -u +%FT%TZ)"\n',
                'echo "## 4a. V7 drop checks $(date -u +%FT%TZ)"\n'
                f'V7RC=0; "$PY" "$R/out_v7/check_v7.py" "$RUNJ" "$OUT/check_v7.json" {flag} --main --ref "{ref}" | tee "$OUT/check_v7.txt" || V7RC=$?\n'
                'echo "## 4. intent checks $(date -u +%FT%TZ)"\n')
    text = edit(text, 'echo "## done $(date -u +%FT%TZ)"\n',
                '[ "$V7RC" = 0 ] || { echo "## V7 drop checks FAIL (exit $V7RC): $OUT/check_v7.json"; exit 5; }\n'
                'echo "## done $(date -u +%FT%TZ)"\n')
    text = edit(text, 'mkdir -p "$OUT"\n', guard_block(rj) + 'mkdir -p "$OUT"\n')
    text = edit(text, "set -euo pipefail\n", PROV.format(flag=flag, role=role) + "set -euo pipefail\n")
    body = "".join(x for x in text.splitlines(True) if not x.startswith("# AUDIT.md Amendment V7"))
    left = [x[:160] for x in body.splitlines() if "check_v7.py\" \"$RUNJ" not in x and (TR_OLD in x or "$R/code/" in x or f"{R}/out/{name}" in x)]
    assert not left, (name, left)
    return text, I


def build():
    window_facts(STEPS, DROP_AT)
    window_facts(CAN_STEPS, CAN_AT)
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
        assert t0 == reg_t, f"{reg.name}: render_v2 != the registered job file"
        assert name == f"pt6v2-{ARM}-lr{LR}-s{s}" and tag == f"pt6v2-{ARM}-lr{LR}"
        ref = R / "out" / name / tag / run_json(args_line(reg_t), s)
        assert ref.exists(), ref
        _, _, ta, _, _ = MJ2.render_v2(P, doms, ARM, LR, s, RF[s], commit, "a100", None)
        assert args_line(ta) == args_line(reg_t), f"{name}: the A100 render's ARGS != the registered ARGS"
        variants = [("main", f"pt6v2-all6drop-lr{LR}-s{s}", f"llmadd_pt6v2v7-all6drop-s{s}", FLAG, False)]
        if s == 16:
            variants.append(("canary", f"canary-all6drop-s{s}", f"llmadd_pt6v2v7-canary-s{s}", f"{','.join(DROP)}@{CAN_AT}", True))
        for role, v7name, jn, flag, can in variants:
            t = edit(ta, f"#SBATCH --job-name=llmadd_{name}\n", f"#SBATCH --job-name={jn}\n")
            t = edit(t, "#SBATCH --partition=a100_tandon\n", "#SBATCH --partition=a100_tandon,h100_tandon\n")
            t = edit(t, 'case "$GPU" in *A100*) ;;', 'case "$GPU" in *A100*|*H100*) ;;')
            t = edit(t, f'echo "DEVICE a100 (seed {s}\'s allocated GPU type)"\n',
                     f'echo "DEVICE $GPU (V7: A100 / H100 for every seed; the device vs the registered run is a flag in the read)"\n')
            t = edit(t, '"gpu":"A100"', '"gpu":"A100|H100"')
            t = edit(t, 'all(I["gpu"] in g for g in gname)', 'all(any(x in g for x in I["gpu"].split("|")) for g in gname)')
            t, I = v7_edits(t, name, v7name, s, f"{role}, A100 / H100", flag, ref, can)
            assert "#SBATCH --account=<slurm-account>\n" in t and "b200" not in t.split("set -euo")[0].lower()
            assert subprocess.run(["bash", "-n"], input=t.encode()).returncode == 0, f"{v7name}: bash -n"
            rl, tl = reg_t.splitlines(), t.splitlines()
            nd = sum(1 for x in difflib.unified_diff(rl, tl, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---"))
            jf = OV7 / f"job_{v7name}.sbatch"
            jobs.append({"name": v7name, "seed": s, "role": role, "text": t, "slurm_name": jn, "job": jf, "ref": str(ref),
                         "flag": OV7 / f"SUBMITTED_{v7name}", "registered_job": reg.name, "registered_sha12": sha(reg),
                         "intent": I, "lines_vs_registered": nd, "out": str(OV7 / v7name), "drop": flag})
    return commit, jobs


def manifest(commit, jobs, ids=None):
    ids = ids or {}
    return {"amendment": "V7", "kind": "post-hoc", "approval": APPROVAL, "why": WHY,
            "reading_rule": "descriptive, no verdict rule, no gate: dim strict final3 / area of all6drop vs the registered all6@2e-3 "
                            "and only-dim@2e-3, paired by seed; dim's drop from the window's close to the end; torus / box5 "
                            "accuracy after the drop (out_v7/v7read.py); readout_pt6v2.py does not read out_v7/",
            "cell": f"all6drop@{LR}", "seeds": list(SEEDS), "drop": {"domains": list(DROP), "step": DROP_AT,
                                                                       "meaning": "from step 7324 (the first step after the prior "
                                                                                  "window 916..7323) the torus / box5 lines are drawn and "
                                                                                  "left out; the row's text fills their space"},
            "trainer": {"path": "runs/pt6v2/out_v7/code/tools/pretrain_range.py", "sha12": TR_NEW, "parent_sha12": TR_OLD,
                        "patch": "runs/pt6v2/out_v7/patch_trainer_v7.py", "diff": "runs/pt6v2/out_v7/trainer_v7.diff"},
            "check_v7": {"path": "runs/pt6v2/out_v7/check_v7.py", "sha12": CV7},
            "code_commit": commit, "out_dir": "runs/pt6v2/out_v7", "driver": "runs/pt6v2/posthoc_v7_jobs.py",
            "device": "A100 / H100 (a100_tandon,h100_tandon), one job per run; registered seeds 16-22 ran on B200: a flag in the read",
            "runs": {j["name"]: {"seed": j["seed"], "role": j["role"], "slurm_name": j["slurm_name"], "drop": j["drop"],
                                 "job_file": str(j["job"].relative_to(R)), "out": j["out"], "ref_run": j["ref"],
                                 "job_sha12": hashlib.sha256(j["text"].encode()).hexdigest()[:12],
                                 "registered_job": j["registered_job"], "registered_sha12": j["registered_sha12"],
                                 "job_id": ids.get(j["name"])} for j in jobs}}


def main():
    commit, jobs = build()
    holds = [h for h in [R / "HOLD", R / "HOLD_a100", OV7 / "HOLD"] if h.exists()]
    for j in jobs:
        st = ("SUBMITTED" if j["flag"].exists() else "on disk (same)" if j["job"].exists() and j["job"].read_text() == j["text"]
              else "on disk (DIFFERS)" if j["job"].exists() else "not written")
        print(f"{j['name']:28s} {j['role']:6s} {j['slurm_name']:32s} {j['lines_vs_registered']:3d} +/- lines vs {j['registered_job']}  {st}")
    print(f"check: {len(jobs)} job texts parse (bash -n); ARGS = registered + the flag; frozen files unchanged; check_v7.py {CV7}; "
          f"HOLD files: {[h.name for h in holds] or 'none'}")
    if "--print" in sys.argv:
        for j in jobs[:1] + jobs[-1:]:
            print(f"\n==== {j['job'].name}\n" + "".join(x + "\n" for x in j["text"].splitlines() if x.startswith("#SBATCH")))
            sys.stdout.writelines(difflib.unified_diff((R / j["registered_job"]).read_text().splitlines(True),
                                                       j["text"].splitlines(True), n=0))
    if "--write" in sys.argv:
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"]
                continue
            j["job"].write_text(j["text"])
        old = json.loads(POSTHOC.read_text()) if POSTHOC.exists() else None
        assert old is None or all(r["job_id"] is None for r in old["runs"].values()), "POSTHOC_V7.json holds job ids: not rewritten"
        POSTHOC.write_text(json.dumps(manifest(commit, jobs), indent=1, ensure_ascii=False) + "\n")
        print(f"written: {len(jobs)} job files in {OV7}; {POSTHOC}")
    if "--test-only" in sys.argv:
        for j in jobs:
            o = subprocess.run(SBATCH + ["--test-only", str(j["job"])], capture_output=True, text=True)
            print(j["job"].name, (o.stdout + o.stderr).strip())
    if "--submit" in sys.argv:
        assert not holds, f"HOLD files exist: {holds}"
        stale = [j["job"].name for j in jobs if not j["flag"].exists() and (not j["job"].exists() or j["job"].read_text() != j["text"])]
        assert not stale, f"job files missing or differing (run --write first): {stale}"
        man = json.loads(POSTHOC.read_text())
        for j in sorted(jobs, key=lambda j: j["role"] != "canary"):
            if j["flag"].exists():
                print(f"{j['job'].name}: refused (SUBMITTED flag exists)")
                continue
            o = subprocess.run(SBATCH + [str(j["job"])], capture_output=True, text=True)
            print(j["job"].name, (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                man["runs"][j["name"]]["job_id"] = o.stdout.split()[-1]
                POSTHOC.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
