"""pt2sz test: the 64 rendered stage-A job files.
R1 64 files = the generator's rendering (a second rendering in memory is byte-identical; make_jobs_pt2sz --check passes),
   names / tags unique and used by no other batch's run directory; Slurm job names unique and = llmadd_p2z-<run>; bash -n.
R2 the #SBATCH header of every job: account <slurm-account>, a100_tandon, gpu48, 1 GPU, nice 1000, 8 CPUs,
   64G, --requeue, the size's time limit, the log path; every job holds HOLD, the 7 trainer-module sha checks, COMMIT, the
   A100-80GB case, the data sha256, INTENT, --dry_inputs + its 4 checks, training, offline_lm, CHECK_PY with RC capture,
   the model removal (stage A: only after a passing check) and exit $RC; order: GPU guard before the inputs, the dry run
   before training, the removal after the check.
R3 the CHECK_PY block of every job = make_pt2main_tasks.CHECK_PY = the pt2same job's block, byte for byte.
R4 body diff vs pt2same's job of the same arm and seed (190M: the shape with model flags): every differing line falls in an
   allowed class (job name, nice, time limit, comments, the R / OUT / TAG line, the INTENT line, ARGS, the model-line grep,
   the DRY hash and its echo, the CHECK_PY call lines (run file; RC capture), RC=0, the model removal, the done line,
   exit $RC); anything else fails. The diffs of two jobs are printed.
    python render_check.py"""
import difflib
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2sz as M   # noqa: E402

G2 = M.g2()
R = M.R
fails = 0


def say(ok, msg):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + msg)


J = [x for z in M.SUBMIT_ORDER for x in M.render(z, G2=G2) if x[2]["stage"] == "A"]
files = sorted(R.glob("job_*.sbatch"))
say(len(files) == 64 and len(J) == 64, f"R1 {len(files)} job files, {len(J)} rendered stage-A jobs")
say(all((R / f"job_{n}.sbatch").read_text() == t for n, t, F in J), "R1 a second rendering (in memory) is byte-identical to every file")
o = subprocess.run([M.PY, str(M.HERE / "make_jobs_pt2sz.py"), "--check", "--stage", "A"], capture_output=True, text=True)
say(o.returncode == 0 and "checked" in o.stdout, f"R1 make_jobs_pt2sz.py --check --stage A: rc {o.returncode}: {o.stdout.strip().splitlines()[-1] if o.stdout.strip() else o.stderr[-300:]}")
other = {p.name for p in M.B.glob("runs/*/out*/*") if p.parent.parent != R}
names, tags = [n for n, *_ in J], {F["tag"] for *_, F in J}
say(len(set(names)) == 64 and len(tags) == 16 and not (set(names) | tags) & other,
    f"R1 names / tags unique (64 / {len(tags)}) and used by no other batch's run dir")
jn = {f.name: re.findall(r"^#SBATCH --job-name=(.*)$", f.read_text(), re.M)[0] for f in files}
say(len(set(jn.values())) == 64 and all(v == "llmadd_" + k[4:-7] and v.startswith("llmadd_p2z-") for k, v in jn.items()),
    "R1 Slurm job names unique and = llmadd_p2z-<run>")
bn = [subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True) for f in files]
say(all(x.returncode == 0 and not x.stderr for x in bn), f"R1 bash -n: {sum(x.returncode == 0 for x in bn)}/64 clean")
HEAD = {"--account": "<slurm-account>", "--partition": "a100_tandon", "--qos": "gpu48", "--gres": "gpu:1",
        "--nice": "1000", "--cpus-per-task": "8", "--mem": "64G", "--requeue": None, "--output": f"{M.B}/logs/%x_%j.out"}
MUST = ['if [ -e "$R/HOLD" ]; then echo "HOLD present: exiting"; exit 0; fi', 'cd "$R/code/tools"',
        'grep -q "pretrain_range.py 93e22bf4c4c0" "$R/code/COMMIT"', 'case "$GPU" in *A100*80GB*) ;;',
        '(cd "$DATA" && sha256sum -c -) <<\'SHA\' || exit 3', 'echo "INTENT ', '"$PY" pretrain_range.py $ARGS --dry_inputs >',
        '"$PY" pretrain_range.py $ARGS\n', '"$PY" offline_lm.py --data "$DATA" --tokenizer "$TOK" --out "$OUT/offline_lm.json" "$OUT/$TAG"',
        '<<\'PY\' | tee "$OUT/check.txt" || RC=$?', "set -euo pipefail", "\nRC=0\n", "\nexit $RC\n",
        'if [ "$RC" = 0 ]; then rm -f "$OUT/$TAG/model_whole_', "--save_model --ckpt_every 2000"]
bad = []
for f in files:
    t = f.read_text()
    size = f.name.split("-")[1]
    hd = dict((m[0], m[1] or None) for m in re.findall(r"^#SBATCH (--[a-z-]+)(?:=(.*))?$", t, re.M))
    want = dict(HEAD, **{"--time": M.TLIM[size]})
    if {k: hd.get(k, "<absent>") for k in want} != want or set(hd) - set(want) - {"--job-name"}:
        bad.append((f.name, hd))
    if not all(m in t for m in MUST) or len(re.findall(r'^\[ "\$\(sha256sum \S+ \| cut -c1-12\)" = "[0-9a-f]{12}" \]', t, re.M)) != 7:
        bad.append((f.name, "a required check is missing", [m for m in MUST if m not in t]))
    if (t.index('case "$GPU"') > t.index("sha256sum -c -") or t.index("--dry_inputs >") > t.index('## 2. training')
            or t.index("|| RC=$?") > t.index('rm -f "$OUT/$TAG/model_') or t.index('rm -f "$OUT/$TAG/model_') > t.index("\nexit $RC\n")):
        bad.append((f.name, "order"))
    rm = re.findall(r'rm -f "\$OUT/\$TAG/(model_\S+\.pt)"', t)
    nm = f.name[4:-7]
    tag = nm.rsplit("-s", 1)[0]
    if rm != [f"model_whole_{'none' if '-none-' in nm else 'helix3'}_seed{nm.rsplit('-s', 1)[1]}_{tag}.pt"]:
        bad.append((f.name, "model file name", rm))
say(not bad, f"R2 headers, required checks, order and the model file name in all 64 jobs {bad[:2] if bad else ''}")
blk = lambda t: t.split("<<'PY' | tee \"$OUT/check.txt\"", 1)[1].split("\n", 1)[1].split("PY\n", 1)[0]
S2 = M.PS.R
ok3 = []
for n, text, F in J:
    ref = (S2 / "job_pt2same-190m-lr4.8e-3-none-s0.sbatch").read_text()
    ok3.append(blk(text) == G2.CHECK_PY == blk(ref))
say(all(ok3), f"R3 CHECK_PY block = make_pt2main_tasks.CHECK_PY = pt2same's job block: {sum(ok3)}/64")
CLASSES = [("job name", r"^#SBATCH --job-name="), ("nice", r"^#SBATCH --nice=(20|1000)$"), ("time limit", r"^#SBATCH --time="),
           ("comment", r"^# "), ("R / OUT / TAG line", r"^R=\S+; OUT=\S+; TAG=\S+; SEED=\d; DATA="),
           ("GPU case", r'^case "\$GPU" in '), ("INTENT line", r'^echo "INTENT '), ("ARGS", r'^ARGS="'),
           ("model-line grep", r"^grep -qxF 'model: "), ("DRY hash", r'^\[ "\$got" = [0-9a-f]{64} \]'),
           ("DRY echo", r'^echo "## dry_inputs rows = the reference \('), ("CHECK_PY run file", r'^"\$PY" - "\$OUT/\$TAG/run_whole_'),
           ("CHECK_PY intent + RC", r"""^  "\$OUT/check\.json" '\{.*\}' <<'PY' \| tee "\$OUT/check\.txt"( \|\| RC=\$\?)?$"""),
           ("RC=0", r"^RC=0$"), ("model removal", r'^if \[ "\$RC" = 0 \]; then rm -f "\$OUT/\$TAG/model_'),
           ("done line", r'^echo "## done \$\(date -u \+%FT%TZ\)( \(check rc \$RC\))?"$'), ("exit RC", r"^exit \$RC$")]
SHOW = ("p2z-294m-lr9.6e-3-none-s0", "p2z-93m-lr6e-4-win-ctx-s3")
for mine, text, F in J:
    theirs = f"pt2same-190m-lr{'4.8e-3-none' if F['arm'] == 'none' else '6e-4-win-ctx'}-s{F['seed']}"
    a = (S2 / f"job_{theirs}.sbatch").read_text().splitlines()
    b = text.splitlines()
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    cls, unk = {}, []
    for l in d:
        c = next((c for c, rx in CLASSES if re.match(rx, l[1:])), None)
        if c is None:
            unk.append(l)
        else:
            cls[c] = cls.get(c, 0) + 1
    if mine in SHOW:
        print(f"---- diff job_{theirs}.sbatch -> job_{mine}.sbatch ({len(d)} lines; long lines cut at 220)")
        for l in d:
            print("    " + l[:220])
    say(not unk, f"R4 {mine} vs {theirs}: {len(d)} changed lines, classes {cls}" + (f" UNCLASSIFIED {unk}" if unk else ""))
print(f"RENDER CHECK {'PASS' if not fails else 'FAIL'} ({fails} failures)")
sys.exit(1 if fails else 0)
