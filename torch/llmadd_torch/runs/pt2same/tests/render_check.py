"""pt2same test: the 32 rendered job files.
R1 32 files = the generator's rendering (JOBS.sha256 verifies; a second rendering in memory is byte-identical), names / tags
   unique and used by no other batch's run directory; bash -n on every file.
R2 the #SBATCH header of every job: account torch_pr_520_tandon_priority, a100_tandon, gpu48, 1 GPU, nice 20, 8 CPUs, 64G,
   --requeue, the size's time limit; every job holds HOLD, the 7 trainer-module sha checks, COMMIT, the A100-80GB case, the
   data sha256, INTENT, --dry_inputs + its 4 checks, training, offline_lm, CHECK_PY.
R3 CHECK_PY block of every job = make_pt2main_tasks.CHECK_PY = the Burst twin task's block, byte for byte.
R4 body diff vs pt2s505 (the same arm and seed): every differing line falls in an allowed class (job name, time limit,
   comments, the R / OUT / TAG line, the GPU case line, the INTENT line, ARGS, the model-line grep, the DRY hash and its
   echo, the run-file / intent lines of the CHECK_PY call); anything else fails. The diff of two pairs is printed.
    python render_check.py"""
import difflib
import hashlib
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2same as M   # noqa: E402

G2 = M.g2()
R = M.R
fails = 0


def say(ok, msg):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + msg)


J = M.jobs(G2)
files = sorted(R.glob("job_*.sbatch"))
say(len(files) == 32 and len(J) == 32, f"R1 {len(files)} job files, {len(J)} rendered jobs")
man = {l.split()[1]: l.split()[0] for l in (R / "JOBS.sha256").read_text().splitlines()}
say(sorted(man) == [f.name for f in files] and all(hashlib.sha256(f.read_bytes()).hexdigest() == man[f.name] for f in files),
    "R1 JOBS.sha256 lists exactly the 32 files and verifies")
say(all((R / f"job_{n}.sbatch").read_text() == t for n, _, t, *_ in J), "R1 a second rendering (in memory) is byte-identical to every file")
say(list(man) == [f"job_{n}.sbatch" for n, *_ in J], "R1 manifest order = the submission order (seed-major, 190M first)")
other = {p.name for p in M.B.glob("runs/*/out*/*") if p.parent.parent != R}
names, tags = [n for n, *_ in J], {t for _, t, *_ in J}
say(len(set(names)) == 32 and not (set(names) | tags) & other, f"R1 names / tags unique (32 / {len(tags)}) and used by no other batch")
jn = [re.findall(r"^#SBATCH --job-name=(.*)$", f.read_text(), re.M)[0] for f in files]
say(len(set(jn)) == 32, "R1 Slurm job names unique")
bn = [subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True) for f in files]
say(all(o.returncode == 0 and not o.stderr for o in bn), f"R1 bash -n: {sum(o.returncode == 0 for o in bn)}/32 clean")
HEAD = {"--account": "torch_pr_520_tandon_priority", "--partition": "a100_tandon", "--qos": "gpu48", "--gres": "gpu:1", "--nice": "20",
        "--cpus-per-task": "8", "--mem": "64G", "--requeue": None, "--output": f"{M.B}/logs/%x_%j.out"}
MUST = ['if [ -e "$R/HOLD" ]; then echo "HOLD present: exiting"; exit 0; fi', 'cd "$R/code/tools"',
        'grep -q "pretrain_range.py 93e22bf4c4c0" "$R/code/COMMIT"', 'case "$GPU" in *A100*80GB*) ;;',
        '(cd "$DATA" && sha256sum -c -) <<\'SHA\' || exit 3', 'echo "INTENT ', '"$PY" pretrain_range.py $ARGS --dry_inputs >',
        '"$PY" pretrain_range.py $ARGS\n', '"$PY" offline_lm.py --data "$DATA" --tokenizer "$TOK" --out "$OUT/offline_lm.json" "$OUT/$TAG"',
        '<<\'PY\' | tee "$OUT/check.txt"', "set -euo pipefail"]
bad = []
for f in files:
    t = f.read_text()
    size = f.name.split("-")[1]
    hd = dict((m[0], m[1] or None) for m in re.findall(r"^#SBATCH (--[a-z-]+)(?:=(.*))?$", t, re.M))
    want = dict(HEAD, **{"--time": M.TLIM[size]})
    if {k: hd.get(k, "<absent>") for k in want} != want or set(hd) - set(want) - {"--job-name"}:
        bad.append((f.name, hd))
    if not all(m in t for m in MUST) or len(re.findall(r'^\[ "\$\(sha256sum \S+ \| cut -c1-12\)" = "[0-9a-f]{12}" \]', t, re.M)) != 7:
        bad.append((f.name, "a required check is missing"))
    if t.index('case "$GPU"') > t.index("sha256sum -c -") or t.index("--dry_inputs >") > t.index('## 2. training'):
        bad.append((f.name, "order: GPU guard before inputs, dry run before training"))
say(not bad, f"R2 headers and required checks in all 32 jobs {bad[:2] if bad else ''}")
blk = lambda t: t.split("<<'PY' | tee \"$OUT/check.txt\"\n", 1)[1].split("\nPY\n", 1)[0] + "\n"
bt = lambda t: t.split("<<'PY' | tee \"$OUT/check.txt\"\n", 1)[1].split("PY\n", 1)[0]
ok3 = []
for n, tag, text, args, I, bname in J:
    btext = (M.BURST / "tasks" / f"{bname}.sh").read_text()
    ok3.append(bt(text) == G2.CHECK_PY == bt(btext))
say(all(ok3), f"R3 CHECK_PY block = make_pt2main_tasks.CHECK_PY = the Burst twin's block: {sum(ok3)}/32")
CLASSES = [("job name", r"^#SBATCH --job-name="), ("time limit", r"^#SBATCH --time="), ("comment", r"^# "),
           ("R / OUT / TAG line", r"^R=\S+; OUT=\S+; TAG=\S+; SEED=\d; DATA="), ("GPU case", r'^case "\$GPU" in '),
           ("INTENT line", r'^echo "INTENT '), ("ARGS", r'^ARGS="'), ("model-line grep", r"^grep -qxF 'model: "),
           ("DRY hash", r'^\[ "\$got" = [0-9a-f]{64} \]'), ("DRY echo", r'^echo "## dry_inputs rows = the reference \('),
           ("CHECK_PY run file", r'^"\$PY" - "\$OUT/\$TAG/run_whole_'), ("CHECK_PY intent", r"""^  "\$OUT/check\.json" '\{""")]
S505 = M.B / "runs" / "pt2s505"
SHOW = ("pt2same-190m-lr6e-4-win-ctx-s4", "pt2same-53m-lr8e-3-none-s0")
for mine in names:   # pt2s505's job of the same arm (its pick) and seed
    theirs = f"pt2s505-lr{'4.8e-3-none' if '-none-' in mine else '6e-4-win-ctx'}-s{mine.rsplit('-s', 1)[1]}"
    a = (S505 / f"job_{theirs}.sbatch").read_text().splitlines()
    b = (R / f"job_{mine}.sbatch").read_text().splitlines()
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    cls, unk = {}, []
    for l in d:
        c = next((c for c, rx in CLASSES if re.match(rx, l[1:])), None)
        if c is None:
            unk.append(l)
        else:
            cls[c] = cls.get(c, 0) + 1
    if mine in SHOW:
        print(f"---- diff job_{theirs}.sbatch -> job_{mine}.sbatch ({len(d)} lines; long lines cut at 200)")
        for l in d:
            print("    " + l[:200])
    say(not unk, f"R4 {mine} vs {theirs}: {len(d)} changed lines, classes {cls}" + (f" UNCLASSIFIED {unk}" if unk else ""))
print(f"RENDER CHECK {'PASS' if not fails else 'FAIL'} ({fails} failures)")
sys.exit(1 if fails else 0)
