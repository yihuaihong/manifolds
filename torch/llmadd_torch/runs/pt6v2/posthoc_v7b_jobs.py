#!/usr/bin/env python3
"""pt6v2 Amendment V7b (post-hoc, 2026-11-22; exp-builder): the slot-matched rerun of V7 (all6drop@2e-3). Job files = the V7 job
of the seed, rendered by posthoc_v7_jobs.build() (= the registered all6@2e-3 job, A100 render, + V7's edits; imported read only),
with these edits only:
  - trainer runs/pt6v2/out_v7b/code (bb0b55eb5e34 = the V7 trainer 8da22c6ea208 + --drop_fill, patch_trainer_v7b.py), COMMIT there;
  - ARGS (and the canary's ARGS_DRY) + ' --drop_fill inplace'; intent code_sha / args.drop_fill;
  - out dir runs/pt6v2/out_v7b/<pt6v2-all6dropS-lr2e-3-s<seed> | canary-all6dropS-s16>; Slurm names llmadd_pt6v2v7b-*;
  - check_v7.py -> out_v7b/check_v7b.py (sha pinned; --ref = the registered all6 run, --v7 = the V7 run of the seed): LOG AND FLAG
    ONLY; the registered intent checks (CHECK_PY) and V7's 'exit 5' no longer fail the job (AUDIT.md Amendment V7b).
Modes: (none) = check and list; --print (header + diff vs the V7 job of two jobs); --write (job files + POSTHOC_V7B.json);
--test-only (sbatch --test-only every job file); --submit (canary first, then the 8 mains; SUBMITTED_<name> flags)."""
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

OV7, OV7B = P7.OV7, R / "out_v7b"
sha = P7.sha
TR_V7, TR_V7B = P7.TR_NEW, "bb0b55eb5e34"
assert sha(OV7B / "code/tools/pretrain_range.py") == TR_V7B, "not the V7b trainer"
_same = sorted(p.name for p in (OV7 / "code/tools").iterdir() if p.is_file())
assert _same == sorted(p.name for p in (OV7B / "code/tools").iterdir() if p.is_file()), "V7b code dir: file list differs"
assert all(sha(OV7 / "code/tools" / f) == sha(OV7B / "code/tools" / f) for f in _same if f != "pretrain_range.py"), "V7b code: a dep differs"
assert f"pretrain_range.py {TR_V7B} " in (OV7B / "code/COMMIT").read_text()
CV7B = sha(OV7B / "check_v7b.py")
POSTHOC = R / "POSTHOC_V7B.json"
APPROVAL = 'user-approved 2026-11-22 (8 seeds, ~4 A100 h; relayed by the main session); post-hoc, exploratory, no gate'
WHY = ("audit I1 (runs/pt6v2/tests/audit_v7_result_20261122.md): V7 leaves the torus / box5 lines out and the row's text grows to the "
       "left, so dim's slot-1 share goes 0.50 -> ~0.75 after the drop; the domain eval question is always in slot 1, so F109's '77% "
       "retained' may be inflated by the slot shift. V7b replaces each dropped line in its slot by a same-length text filler: every "
       "kept line keeps the flag-off (registered all6) positions and slots; total text tokens = V7's.")
RULE = ("Report S_b with its CI and the paired slot effect; if the slot effect's CI includes 0 and S_b's CI includes 0.77, F109's 77% "
        "stands and its qualifier 2 is replaced by the measured slot effect; otherwise the paper's 77% is replaced by S_b. No check "
        "gates the read-out.")
PROV = ("# AUDIT.md Amendment V7b (post-hoc, 2026-11-22): rendered by runs/pt6v2/posthoc_v7b_jobs.py = the V7 job of this seed "
        "(posthoc_v7_jobs.py, below) + trainer runs/pt6v2/out_v7b/code ({tr} = {tv7} + --drop_fill) + --drop_fill inplace + "
        "check_v7b.py (log and flag only; CHECK_PY no longer fails the job); out dir runs/pt6v2/out_v7b/ ({role}); the V7 line below "
        "describes the parent job\n")


def v7b_name(v7name):
    return v7name.replace("pt6v2-all6drop-", "pt6v2-all6dropS-").replace("canary-all6drop-", "canary-all6dropS-")


def v7b_edits(j):
    t, v7name, s, can = j["text"], j["name"], j["seed"], j["role"] == "canary"
    name = v7b_name(v7name)
    assert name != v7name
    jn_old = j["slurm_name"]
    jn = jn_old.replace("llmadd_pt6v2v7-all6drop-", "llmadd_pt6v2v7b-all6dropS-").replace("llmadd_pt6v2v7-canary-", "llmadd_pt6v2v7b-canary-")
    assert jn != jn_old
    tag = t.split("TAG=")[1].split(";")[0]
    rj = P7.run_json(P7.args_line(t), s)
    v7run = OV7 / v7name / tag / rj
    t = P7.edit(t, f"#SBATCH --job-name={jn_old}\n", f"#SBATCH --job-name={jn}\n")
    t = P7.edit(t, f"{OV7}/{v7name}", f"{OV7B}/{name}", 3 if can else 2)          # OUT=, --out in ARGS (and ARGS_DRY)
    t = P7.edit(t, '"$R/out_v7/code/', '"$R/out_v7b/code/', 3)                    # cd, COMMIT grep, CHECK_PY's COMMIT argument
    t = P7.edit(t, f'[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{TR_V7}" ]', f'[ "$(sha256sum pretrain_range.py | cut -c1-12)" = "{TR_V7B}" ]')
    t = P7.edit(t, f'grep -q "pretrain_range.py {TR_V7}" ', f'grep -q "pretrain_range.py {TR_V7B}" ')
    t = P7.edit(t, f'|| {{ echo "## COMMIT does not name the trainer {TR_V7}"; exit 3; }}', f'|| {{ echo "## COMMIT does not name the trainer {TR_V7B}"; exit 3; }}')
    t = P7.edit(t, f'[ "$(sha256sum "$R/out_v7/check_v7.py" | cut -c1-12)" = "{P7.CV7}" ] || {{ echo "## check_v7.py sha mismatch: exiting"; exit 3; }}\n',
                f'[ "$(sha256sum "$R/out_v7b/check_v7b.py" | cut -c1-12)" = "{CV7B}" ] || {{ echo "## check_v7b.py sha mismatch: exiting"; exit 3; }}\n')
    al = P7.args_line(t)
    assert al.count(f" --drop_domains_after {j['drop']}") == 1
    t = P7.edit(t, al + "\n", al.replace(f" --drop_domains_after {j['drop']}", f" --drop_domains_after {j['drop']} --drop_fill inplace") + "\n")
    if can:
        dl = [x for x in t.splitlines() if x.startswith("ARGS_DRY=")]
        assert len(dl) == 1 and dl[0].count(f" --drop_domains_after {P7.FLAG}\"") == 1
        t = P7.edit(t, dl[0] + "\n", dl[0].replace(f" --drop_domains_after {P7.FLAG}\"", f" --drop_domains_after {P7.FLAG} --drop_fill inplace\"") + "\n")
    old_ln = P7.intent_line(t)
    js = old_ln.split("'")[1]
    I = json.loads(js)
    assert json.dumps(I, separators=(",", ":"), ensure_ascii=False) == js and I["code_sha"] == TR_V7 and "drop_fill" not in I["args"]
    I["code_sha"] = TR_V7B
    I["args"]["drop_fill"] = "inplace"
    new_ln = old_ln.replace(js, json.dumps(I, separators=(",", ":"), ensure_ascii=False))
    if not can:   # log and flag only (the canary's CHECK_PY already ends with '|| echo')
        assert new_ln.endswith(""""$ARGS" <<'PY' | tee "$OUT/check.txt\"""")
        new_ln += (' || echo "## FLAG V7b: CHECK_PY (the registered intent checks) failed: $OUT/check.json (log and flag only, '
                   'AUDIT.md Amendment V7b)"')
    t = P7.edit(t, old_ln, new_ln)
    old_c = (f'V7RC=0; "$PY" "$R/out_v7/check_v7.py" "$RUNJ" "$OUT/check_v7.json" {j["drop"]} --main --ref "{j["ref"]}" '
             f'| tee "$OUT/check_v7.txt" || V7RC=$?\n')
    t = P7.edit(t, old_c, f'V7RC=0; "$PY" "$R/out_v7b/check_v7b.py" "$RUNJ" "$OUT/check_v7b.json" {j["drop"]} --main --ref "{j["ref"]}" '
                          f'--v7 "{v7run}" | tee "$OUT/check_v7b.txt" || V7RC=$?\n')
    t = P7.edit(t, 'echo "## 4a. V7 drop checks $(date -u +%FT%TZ)"\n', 'echo "## 4a. V7b drop checks (log and flag only) $(date -u +%FT%TZ)"\n')
    t = P7.edit(t, '[ "$V7RC" = 0 ] || { echo "## V7 drop checks FAIL (exit $V7RC): $OUT/check_v7.json"; exit 5; }\n',
                '[ "$V7RC" = 0 ] || echo "## FLAG V7b: check_v7b.py exited $V7RC (crash): $OUT/check_v7b.txt (log and flag only)"\n'
                'echo "## V7b drop checks verdict: $(tail -1 "$OUT/check_v7b.txt" 2>/dev/null)"\n')
    t = P7.edit(t, "# AUDIT.md Amendment V7 (post-hoc", PROV.format(tr=TR_V7B, tv7=TR_V7, role=j["role"] + ", A100 / H100")
                + "# AUDIT.md Amendment V7 (post-hoc")
    assert subprocess.run(["bash", "-n"], input=t.encode()).returncode == 0, f"{name}: bash -n"
    body = [x for x in t.splitlines() if not x.startswith("# AUDIT.md Amendment V7 (")]
    left = [x[:160] for x in body if (TR_V7 in x and not x.startswith("# AUDIT.md Amendment V7b")) or "out_v7/check_v7.py" in x
            or ("out_v7/" in x and f'--v7 "{v7run}"' not in x and not x.startswith("# AUDIT.md Amendment V7b"))]
    assert not left, (name, left)
    assert t.count("--drop_fill inplace") == (3 if can else 2), (name, t.count("--drop_fill inplace"))   # ARGS (+ARGS_DRY) + PROV
    return {"name": name, "seed": s, "role": j["role"], "text": t, "slurm_name": jn, "job": OV7B / f"job_{name}.sbatch",
            "flag": OV7B / f"SUBMITTED_{name}", "out": str(OV7B / name), "ref": j["ref"], "v7_run": str(v7run), "drop": j["drop"],
            "v7_name": v7name, "v7_text": j["text"], "intent": I, "run_json": f"{OV7B}/{name}/{tag}/{rj}"}


def build():
    commit, j7 = P7.build()
    jobs = [v7b_edits(j) for j in j7]
    # uniqueness / collisions: names, Slurm names, out dirs, job files; no out dir of V7b exists with a run JSON; no live job of mine
    # with these names; no V7 / registered path reused
    assert len({j["name"] for j in jobs}) == len({j["slurm_name"] for j in jobs}) == len({j["out"] for j in jobs}) == len(jobs) == 9
    assert not {j["slurm_name"] for j in jobs} & {j["slurm_name"] for j in j7}
    assert all(not Path(j["out"]).is_relative_to(OV7) and not Path(j["out"]).is_relative_to(R / "out") for j in jobs)
    live = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True).stdout.split()
    clash = [j["slurm_name"] for j in jobs if j["slurm_name"] in live]
    assert not clash, f"a live job has this name: {clash}"
    for j in jobs:
        o = Path(j["out"])
        if not j["flag"].exists():
            assert not (o.exists() and any(o.rglob("run_whole_*.json"))), f"{o}: a run JSON exists before submission"
    return commit, jobs


def manifest(commit, jobs, ids=None):
    ids = ids or {}
    return {"amendment": "V7b", "kind": "post-hoc", "approval": APPROVAL, "why": WHY, "reading_rule": RULE,
            "readout": "runs/pt6v2/out_v7b/v7bread.py (descriptive; no check gates it)",
            "cell": "all6dropS@2e-3", "seeds": list(P7.SEEDS),
            "drop": {"domains": list(P7.DROP), "step": P7.DROP_AT, "fill": "inplace", "fill_stream": "default_rng(7100 + seed)",
                     "meaning": "from step 7324 each torus / box5 line is drawn as before and replaced in its slot by a training-text "
                                "filler of its token length (last token = its newline); every kept line keeps its flag-off positions "
                                "and slot; the row's text is the flag-off text"},
            "trainer": {"path": "runs/pt6v2/out_v7b/code/tools/pretrain_range.py", "sha12": TR_V7B, "parent_sha12": TR_V7,
                        "patch": "runs/pt6v2/out_v7b/patch_trainer_v7b.py", "diff": "runs/pt6v2/out_v7b/trainer_v7b.diff"},
            "check_v7b": {"path": "runs/pt6v2/out_v7b/check_v7b.py", "sha12": CV7B, "gates": False},
            "driver": {"path": "runs/pt6v2/posthoc_v7b_jobs.py", "parent": "runs/pt6v2/posthoc_v7_jobs.py",
                       "parent_sha12": sha(R / "posthoc_v7_jobs.py")},
            "code_commit": commit, "out_dir": "runs/pt6v2/out_v7b",
            "device": "A100 / H100 (a100_tandon,h100_tandon) as V7; registered seeds 16-22 ran on B200: a flag in the read",
            "runs": {j["name"]: {"seed": j["seed"], "role": j["role"], "slurm_name": j["slurm_name"], "drop": j["drop"],
                                 "job_file": str(j["job"].relative_to(R)), "out": j["out"], "run_json": j["run_json"],
                                 "ref_run": j["ref"], "v7_run": j["v7_run"], "v7_name": j["v7_name"],
                                 "job_sha12": hashlib.sha256(j["text"].encode()).hexdigest()[:12],
                                 "job_id": ids.get(j["name"])} for j in jobs}}


def main():
    commit, jobs = build()
    holds = [h for h in [R / "HOLD", R / "HOLD_a100", OV7B / "HOLD"] if h.exists()]
    for j in jobs:
        nd = sum(1 for x in difflib.unified_diff(j["v7_text"].splitlines(), j["text"].splitlines(), lineterm="", n=0)
                 if x[:1] in "+-" and x[:3] not in ("+++", "---"))
        st = ("SUBMITTED" if j["flag"].exists() else "on disk (same)" if j["job"].exists() and j["job"].read_text() == j["text"]
              else "on disk (DIFFERS)" if j["job"].exists() else "not written")
        print(f"{j['name']:30s} {j['role']:6s} {j['slurm_name']:34s} {nd:3d} +/- lines vs V7 {j['v7_name']}  {st}")
    print(f"check: {len(jobs)} job texts parse (bash -n); unique names / out dirs, no live-name clash, no run JSON in the out dirs; "
          f"trainer {TR_V7B}; check_v7b.py {CV7B}; posthoc_v7_jobs.py {sha(R / 'posthoc_v7_jobs.py')}; HOLD: {[h.name for h in holds] or 'none'}")
    if "--print" in sys.argv:
        for j in jobs[:1] + jobs[-1:]:
            print(f"\n==== {j['job'].name}\n" + "".join(x + "\n" for x in j["text"].splitlines() if x.startswith("#SBATCH")))
            sys.stdout.writelines(x[:400] + ("\n" if len(x) > 400 else "") for x in
                                  difflib.unified_diff(j["v7_text"].splitlines(True), j["text"].splitlines(True), n=0))
    if "--write" in sys.argv:
        OV7B.mkdir(exist_ok=True)
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"]
                continue
            j["job"].write_text(j["text"])
        old = json.loads(POSTHOC.read_text()) if POSTHOC.exists() else None
        assert old is None or all(r["job_id"] is None for r in old["runs"].values()), "POSTHOC_V7B.json holds job ids: not rewritten"
        POSTHOC.write_text(json.dumps(manifest(commit, jobs), indent=1, ensure_ascii=False) + "\n")
        print(f"written: {len(jobs)} job files in {OV7B}; {POSTHOC}")
    if "--test-only" in sys.argv:
        for j in jobs:
            o = subprocess.run(P7.SBATCH + ["--test-only", str(j["job"])], capture_output=True, text=True)
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
            o = subprocess.run(P7.SBATCH + [str(j["job"])], capture_output=True, text=True)
            print(j["job"].name, (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                man["runs"][j["name"]]["job_id"] = o.stdout.split()[-1]
                POSTHOC.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
