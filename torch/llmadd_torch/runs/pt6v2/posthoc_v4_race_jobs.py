#!/usr/bin/env python3
"""pt6v2 Amendment V4 race copies (2026-11-20): one A100/H100 copy of each of the 16 B200 V4 jobs; whichever of a pair starts first
computes the run, the other one exits without training (run_guard, runs/pt6v2/code/tools/run_guard.py).

Why: the 16 B200 V4 jobs (posthoc_v4_jobs.py, 3 singleton chains under the per-user 3-GPU B200 cap, shared with another session's
pending jobs) had a test-only start of 2026-11-22 01:27; priority a100_tandon,h100_tandon estimated 2026-11-20 16:14. The user wants
V4 as fast as possible and endorsed A100 beside B200 (relayed by the main session). The B200 jobs stay queued, untouched.

Rendering: make_jobs_pt6v2.render_v2 on device "a100" (the path that rendered V3's s23 a100_tandon job), with the V4 arm patched in by
posthoc_v4_jobs.v4_arm, then posthoc_v4_jobs' own two counted edits (out dir runs/pt6v2/out -> out_v4, the V4 provenance line). So
each copy writes the SAME out dir and result file as its B200 twin. Job name = llmadd_<run name> (the a100 rendering's), no singleton
dependency. Then counted device / race edits only:
  D1 partition a100_tandon -> a100_tandon,h100_tandon (account <slurm-account>, qos gpu48, gres gpu:1, 8 cpus, 64G,
     03:00:00 = the a100 rendering's, as in the registered a100 jobs)
  D2 the GPU guard accepts A100 or H100 and sets GPUT / DEVT; the intent passed to CHECK_PY takes "gpu" / "device" from them (the
     check "GPU = <type> in every nvidia-smi sample" then asserts one GPU type for the whole run, whichever it is)
  D3 two comment lines name the device (A100 or H100)
  S1 race pre-check, before anything in $OUT is written (gpu.csv / node.txt / dry_inputs.txt are appended / rewritten in the body
     before the trainer runs): exit 0 if the B200 twin job is running, or if run_guard.claim(result file) declines (run complete, or a
     RUNNING job holds the claim); otherwise run_guard.claim writes this job's id into <result>.claim, and the trainer's own claim()
     call (pretrain_range.py:581-584) then passes (same id). Same function, same file: no new guard logic.
  S2 after the trainer: if <result>.claim does not hold this job's id (the trainer's guard declined), exit 0 before CHECK_PY, so
     check.json / check.txt are never rewritten by a copy that did not compute the run.
The training ARGS line is byte-identical to the twin's; the intent differs only in "gpu" / "device"; the twin diff of every pair is
asserted line by line (allowed: the device lines above and the race block).

Not fixable here: the B200 twin bodies are frozen (submitted). A B200 twin that starts while or after its copy computes appends B200
rows to gpu.csv (-> the copy's GPU check fails) and, its trainer guard declining, still runs CHECK_PY and rewrites check.json (FAIL).
So once a copy is RUNNING, its B200 twin should be cancelled by the main session (--status prints those ids; nothing is cancelled here).

Usage:  posthoc_v4_race_jobs.py [--check] [--diff] [--write] [--test-only [N]] [--submit] [--status]
  (no flag / --check)  render the 16 copies, assert the twin diffs, bash -n, every guard; nothing is written
  --diff               write runs/pt6v2/tests/race_v4_twin_diff.txt (unified diff of every pair)
  --write              write runs/pt6v2/out_v4/job_<name>.a100.sbatch (never over a submitted one)
  --test-only [N]      sbatch --test-only on the first N on-disk copy files
  --submit             sbatch each on-disk copy (must equal this rendering), write out_v4/SUBMITTED_<name>.a100, add the ids to
                       POSTHOC_V4.json["race_copies"]; a name with a SUBMITTED_<name>.a100 flag is never submitted again
  --status             read only: per pair the squeue state of copy and twin, the claim holder, check.json pass; prints the twin ids
                       whose copy holds the claim (cancel candidates) - it never cancels anything
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
OV4 = R / "out_v4"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]   # noqa: E731
PV4_SHA = "ed5910122924"
assert sha(R / "posthoc_v4_jobs.py") == PV4_SHA, "posthoc_v4_jobs.py changed (the B200 V4 driver)"
sys.path.insert(0, str(R))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import posthoc_v4_jobs as PV4   # noqa: E402  (read only; asserts make_jobs_pt6v2 / CELLS.json / POSTHOC_V3 / ALLOC shas on import)
sys.argv = _argv
MJ2 = PV4.MJ2
POSTHOC = PV4.POSTHOC
PART_OLD, PART_NEW = "#SBATCH --partition=a100_tandon\n", "#SBATCH --partition=a100_tandon,h100_tandon\n"
GUARD_OLD = ('case "$GPU" in *A100*) ;; *) echo "## not an A100 (40GB or 80GB): seed {s} is allocated to A100 (one GPU type for every '
             'cell of a seed, AUDIT.md GPU types): exiting"; exit 3;; esac\necho "DEVICE a100 (seed {s}\'s allocated GPU type)"\n')
GUARD_NEW = ('case "$GPU" in *A100*) GPUT=A100; DEVT=a100;; *H100*) GPUT=H100; DEVT=h100;; *) echo "## not an A100 or H100 (the GPU '
             'types of this V4 race copy): exiting"; exit 3;; esac\necho "DEVICE $DEVT (V4 race copy of B200 job {twin}; seed {s}\'s '
             'ALLOC.json type {alloc})"\n')
RACE = r'''# V4 race copy (2026-11-20, runs/pt6v2/posthoc_v4_race_jobs.py): B200 job @TWIN@ computes the same run (same OUT, same result file);
# whichever starts first computes. Decided before anything in $OUT is written: exit 0 untouched if B200 job @TWIN@ is running, or if
# run_guard.claim (the trainer's own guard, same result file) declines (run complete, or a RUNNING job holds the claim); otherwise this
# job's id is written into the claim and the trainer's own claim() call passes.
RES="$OUT/$TAG/@RESNAME@"
rc=0; "$PY" - "$RES" @TWIN@ <<'RACE' || rc=$?
import sys, run_guard
res, twin = sys.argv[1], sys.argv[2]
if run_guard._running(twin):
    print(f"run guard (race copy): B200 twin job {twin} is running; exiting", flush=True)
    sys.exit(10)
sys.exit(0 if run_guard.claim(res) else 10)
RACE
[ "$rc" = 0 ] || { [ "$rc" = 10 ] && { echo "## V4 race copy: not computing this run; exiting, nothing in \$OUT written"; exit 0; }; echo "## V4 race copy: guard pre-check failed (rc $rc)"; exit 3; }
'''
POST = ('[ "$(cat "$RES.claim" 2>/dev/null)" = "${SLURM_JOB_ID:-none}" ] || { echo "## V4 race copy: the trainer\'s run guard declined '
        '(claim $(cat "$RES.claim" 2>/dev/null)); exiting before the checks: check.json / check.txt not rewritten"; exit 0; }\n')
TRAIN = '"$PY" pretrain_range.py $ARGS\n'
ALLOWED_TWIN = ("#SBATCH --job-name=", "#SBATCH --dependency=singleton", "#SBATCH --constraint=b200", '#SBATCH --comment="preemption=yes"',
                "#SBATCH --cpus-per-task=16", "#SBATCH --mem=128G", "#SBATCH --open-mode=append", "#SBATCH --output=",
                "# pt6v2 main (pt6comb's C job body on Rev H), composition ", "# (change the generator, not this file). Preemptible:",
                "# 1. HOLD, snapshot shas, the GPU type (B200)", "# runs/pt6v2/HOLD_b200 ", 'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_b200" ]',
                'case "$GPU" in *B200*)', 'echo "DEVICE b200 ', '  "$OUT/check.json" \'')
ALLOWED_COPY_HEAD = ("#SBATCH --job-name=llmadd_pt6v2-", PART_NEW.strip(), "#SBATCH --qos=gpu48", "#SBATCH --cpus-per-task=8",
                     "#SBATCH --mem=64G", "#SBATCH --output=/scratch/yh6210/llmadd_torch/logs/%x_%j.out",
                     "# pt6v2 main (pt6comb's C job body on Rev H), composition ", "# (change the generator, not this file). A requeued job",
                     "# 1. HOLD, snapshot shas, the GPU type (A100 or H100)", "# runs/pt6v2/HOLD_a100 ",
                     'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_a100" ]', 'case "$GPU" in *A100*) GPUT=A100', 'echo "DEVICE $DEVT ',
                     '  "$OUT/check.json" \'')


def edit(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:120])
    return text.replace(old, new)


def intent_json(line, gput=None, devt=None):
    """The intent JSON of a CHECK_PY call line; a copy's spliced "gpu" / "device" are resolved as bash would with GPUT / DEVT."""
    if gput is not None:
        line = line.replace('"gpu":"\'"$GPUT"\'"', f'"gpu":"{gput}"').replace('"device":"\'"$DEVT"\'"', f'"device":"{devt}"')
    return json.loads(line.split("'")[1])


def render_copy(P, doms, commit, RF, tw, twin_id, alloc):
    s, arm, lr = tw["seed"], tw["arm"], tw["lr"]
    with PV4.v4_arm(arm):
        name, tag, text, args, I = MJ2.render_v2(P, doms, arm, lr, s, RF[s], commit, "a100", None)
    assert PV4.MJ.W.count("--window " + PV4.REG_W) == 1 and MJ2.pri_of.__module__ == "make_jobs_pt6v2", "patch not restored"
    assert name == tw["name"] and I["device"] == "a100" and I["gpu"] == "A100", (name, I["device"], I["gpu"])
    # posthoc_v4_jobs.build's two counted edits (same out dir / result file as the twin; the V4 provenance line)
    o_out, n_out = f"{R}/out/{name}", f"{OV4}/{name}"
    assert text.count(o_out) == 2 and args.count(o_out) == 1
    text, args = text.replace(o_out, n_out), args.replace(o_out, n_out)
    assert args == tw["args"], (name, "training ARGS differ from the twin")
    text = edit(text, "set -euo pipefail\n", PV4.PROV.format(arm=arm, base=tw["base"]) + "set -euo pipefail\n")
    # D1-D3 device lines
    text = edit(text, PART_OLD, PART_NEW)
    text = edit(text, GUARD_OLD.format(s=s), GUARD_NEW.format(s=s, twin=twin_id, alloc=alloc))
    ln_i, _ = PV4.intent_of(text)
    ln_n = edit(edit(ln_i, '"gpu":"A100"', '"gpu":"\'"$GPUT"\'"'), '"device":"a100"}\'', '"device":"\'"$DEVT"\'"}\'')
    text = edit(text, ln_i, ln_n)
    text = edit(text, ", A100: ", ", A100 or H100 (V4 race copy): ")
    text = edit(text, "# 1. HOLD, snapshot shas, the GPU type (A100)", "# 1. HOLD, snapshot shas, the GPU type (A100 or H100)")
    # S1 / S2 race block (the result file = the trainer's path = the one CHECK_PY reads)
    resname = f"run_whole_helix3_seed{s}_{tag}.json"
    assert text.count(f'"$OUT/$TAG/{resname}"') == 1 and I["args"]["target"] == "helix3"
    text = edit(text, 'mkdir -p "$OUT"\n', RACE.replace("@TWIN@", str(twin_id)).replace("@RESNAME@", resname) + 'mkdir -p "$OUT"\n')
    text = edit(text, TRAIN, TRAIN + POST)
    # order: GPU guard < race pre-check < first write into $OUT; trainer < S2 < CHECK_PY
    k = {t: text.index(t) for t in ('case "$GPU" in', 'rc=0; "$PY" - "$RES"', 'mkdir -p "$OUT"\n', '>> "$OUT/gpu.csv"', '>> "$OUT/node.txt"',
                                    '> "$OUT/dry_inputs.txt"', TRAIN + POST, '"$OUT/check.json" \'')}
    v = list(k.values())
    assert v == sorted(v), k
    head = text[:k['rc=0; "$PY" - "$RES"']]
    head = "\n".join(x for x in head.splitlines() if not x.lstrip().startswith("#"))
    assert not any(w in head for w in ('> "$OUT', 'mkdir -p "$OUT', 'tee "$OUT', 'cp ', 'mv ', 'rm ')), "a write to $OUT before S1"
    race = (RACE.replace("@TWIN@", str(twin_id)).replace("@RESNAME@", resname) + POST).splitlines()
    return name, tag, text, args, race


def twin_diff(tw_text, cp_text, name, race):
    """Assert the copy differs from its twin only in allowed device / race lines; -> unified diff text."""
    a, b = tw_text.splitlines(), cp_text.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    race_lines = set(race)
    added = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        for x in a[i1:i2]:
            assert x.startswith(ALLOWED_TWIN), (name, "twin line", x[:150])
        for y in b[j1:j2]:
            assert y.startswith(ALLOWED_COPY_HEAD) or y in race_lines, (name, "copy line", y[:150])
            added += [y] if y in race_lines else []
    assert sorted(added) == sorted(race), (name, "race block not inserted whole")
    # the training ARGS line identical; the intent differs from the twin's in gpu / device only
    ta = [x for x in a if x.startswith('ARGS="')]
    assert ta == [x for x in b if x.startswith('ARGS="')] and len(ta) == 1, name
    it = intent_json(PV4.intent_of(tw_text)[0])
    for g, d in (("A100", "a100"), ("H100", "h100")):
        ic = intent_json([x for x in b if x.startswith('  "$OUT/check.json" \'')][0], g, d)
        assert PV4.diff_keys(it, ic) == ["device", "gpu"], (name, PV4.diff_keys(it, ic))
    return "".join(difflib.unified_diff([x + "\n" for x in a], [y + "\n" for y in b], f"B200 twin job_{name}.sbatch",
                                        f"copy job_{name}.a100.sbatch", n=0))


def bash_splice_check(text):
    """Run the copy's intent argument through bash with GPUT / DEVT set: it must parse to JSON with those values."""
    ln = [x for x in text.splitlines() if x.startswith('  "$OUT/check.json" \'')][0]
    tok = ln.split('"$OUT/check.json" ', 1)[1].rsplit(' "$ARGS"', 1)[0]
    for g, d in (("A100", "a100"), ("H100", "h100")):
        o = subprocess.run(["bash", "-c", f"GPUT={g}; DEVT={d}; printf '%s' {tok}"], capture_output=True, text=True, check=True)
        J = json.loads(o.stdout)
        assert J["gpu"] == g and J["device"] == d, (J["gpu"], J["device"])


def build():
    doms, commit, twins = PV4.build()
    man = json.loads(POSTHOC.read_text())
    alloc = {int(k): v for k, v in json.loads((R / "ALLOC.json").read_text()).items()}
    P = MJ2.load_P()
    RF, _ = MJ2.refs_v2(doms)
    jobs = []
    for tw in twins:
        assert tw["flag"].exists() and tw["job"].read_text() == tw["text"], f"{tw['name']}: B200 twin not submitted or differs on disk"
        twin_id = man["runs"][tw["name"]]["job_id"]
        assert twin_id and tw["flag"].read_text().split()[-1] == twin_id, tw["name"]
        name, tag, text, args, race = render_copy(P, doms, commit, RF, tw, twin_id, alloc[tw["seed"]])
        assert f"#SBATCH --job-name=llmadd_{name}\n" in text and "--dependency" not in text and "--constraint" not in text
        assert "#SBATCH --account=<slurm-account>\n" in text and "#SBATCH --gres=gpu:1\n" in text
        d = twin_diff(tw["text"], text, name, race)
        bash_splice_check(text)
        assert subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0, f"{name}: bash -n"
        jobs.append({"name": name, "seed": tw["seed"], "arm": tw["arm"], "twin_id": twin_id, "twin_slot": MJ2.slot_name(tw["slot"]),
                     "text": text, "diff": d, "job": OV4 / f"job_{name}.a100.sbatch", "flag": OV4 / f"SUBMITTED_{name}.a100",
                     "res": OV4 / name / tag / f"run_whole_helix3_seed{tw['seed']}_{tag}.json"})
    return jobs


def squeue_states(ids):
    o = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T"], capture_output=True, text=True, timeout=60)
    return dict(x.split() for x in o.stdout.splitlines() if x.strip())


def main():
    jobs = build()
    holds = [h for h in [R / "HOLD", R / "HOLD_a100"] if h.exists()]
    for j in jobs:
        st = ("SUBMITTED" if j["flag"].exists() else
              "on disk (same)" if j["job"].exists() and j["job"].read_text() == j["text"] else
              "on disk (DIFFERS)" if j["job"].exists() else "not written")
        print(f"{j['name']:28s} copy of B200 {j['twin_id']} ({j['twin_slot']})  {len(j['diff'].splitlines())} diff lines  {st}")
    print(f"check: {len(jobs)} copies parse (bash -n); each = its B200 twin except device lines + the race block (asserted); ARGS identical; "
          f"intent differs in gpu/device only; HOLD files: {[h.name for h in holds] or 'none'}")
    if "--diff" in sys.argv:
        p = R / "tests" / "race_v4_twin_diff.txt"
        p.write_text("".join(f"===== {j['name']}\n{j['diff']}" for j in jobs))
        print(f"written {p}")
    if "--write" in sys.argv:
        for j in jobs:
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"], f"{j['job']} was submitted and differs from this rendering"
                continue
            j["job"].write_text(j["text"])
        print(f"written: {len(jobs)} copy files in {OV4}")
    if "--test-only" in sys.argv:
        i = sys.argv.index("--test-only")
        n = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit() else len(jobs)
        for j in jobs[:n]:
            o = subprocess.run(PV4.SBATCH + ["--test-only", str(j["job"])], capture_output=True, text=True)
            print(j["name"], (o.stdout + o.stderr).strip())
    if "--status" in sys.argv:
        man = json.loads(POSTHOC.read_text())
        rc = man.get("race_copies", {})
        ids = [j["twin_id"] for j in jobs] + [rc[j["name"]]["job_id"] for j in jobs if rc.get(j["name"], {}).get("job_id")]
        st = squeue_states(ids)
        cancel = []
        for j in jobs:
            cid = rc.get(j["name"], {}).get("job_id")
            cl = j["res"].with_name(j["res"].name + ".claim")
            holder = cl.read_text().strip() if cl.exists() else "-"
            ck = j["res"].parent.parent / "check.json"
            ok = json.loads(ck.read_text()).get("pass") if ck.exists() else "-"
            done = j["res"].exists()
            print(f"{j['name']:28s} copy {cid} {st.get(cid, 'ended')}  twin {j['twin_id']} {st.get(j['twin_id'], 'ended')}  "
                  f"claim {holder}  check.json pass {ok}  result file {'yes' if done else 'no'}")
            if cid and holder == cid and st.get(j["twin_id"]) == "PENDING":
                cancel.append(j["twin_id"])
        print("B200 twins whose copy holds the claim (cancel candidates, main session decides):", " ".join(cancel) or "none")
    if "--submit" in sys.argv:
        assert not holds, f"HOLD files exist: {holds}"
        stale = [j["name"] for j in jobs if not j["flag"].exists() and (not j["job"].exists() or j["job"].read_text() != j["text"])]
        assert not stale, f"copy files missing or differing from this rendering (run --write first): {stale}"
        man = json.loads(POSTHOC.read_text())
        keep = {k: v for k, v in man.items() if k != "race_copies"}
        rc = man.get("race_copies", {})
        for j in jobs:
            if j["flag"].exists():
                print(f"{j['name']}: refused (SUBMITTED_{j['name']}.a100 exists: a copy is submitted once)")
                continue
            o = subprocess.run(PV4.SBATCH + [str(j["job"])], capture_output=True, text=True)
            print(j["name"], (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                rc[j["name"]] = {"job_id": o.stdout.split()[-1], "slurm_name": f"llmadd_{j['name']}", "twin_job_id": j["twin_id"],
                                 "twin_slot": j["twin_slot"], "file": f"runs/pt6v2/out_v4/{j['job'].name}",
                                 "job_sha12": hashlib.sha256(j["text"].encode()).hexdigest()[:12],
                                 "partition": "a100_tandon,h100_tandon", "account": "<slurm-account>"}
        man2 = dict(keep, race_copies=rc, race_note=(
            "A100/H100 race copies (posthoc_v4_race_jobs.py): same out dir and result file as the B200 twin; whichever starts first "
            "computes (run_guard); a copy that does not compute exits 0 (COMPLETED) without writing in the out dir. Device of each run = "
            "the gpu field of its check.json / gpu.csv (A100, H100 or B200): flagged only."))
        assert {k: v for k, v in man2.items() if k not in ("race_copies", "race_note")} == keep
        POSTHOC.write_text(json.dumps(man2, indent=1, ensure_ascii=False) + "\n")
        print(f"submitted {sum(1 for j in jobs if j['name'] in rc)} copies; ids in {POSTHOC} race_copies")


if __name__ == "__main__":
    main()
