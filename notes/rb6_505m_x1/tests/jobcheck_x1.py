"""Amendment X1 job check: the 2 rendered none@2.4e-3 job files against the SUBMITTED sweep none@4.8e-3 job of the same
seed (the audited, GPU-run template):
  1. bash -n;
  2. make_jobs_rb6scale.render() (via the X1 generator's imports) reproduces the submitted none@4.8e-3 files byte for byte
     and the X1 files byte for byte (the snapshot / refs / generator have not drifted since the sweep);
  3. token by token (whitespace tokens, difflib, no autojunk): every differing token is a same-position replacement whose
     new token = the old one with 4.8e-3 -> 2.4e-3 (and the intent's lr 0.0048 -> 0.0024), each classified as job name /
     out dir / tag / lr / intent (lr + tag); no inserted or deleted token;
  4. ARGS differ only in the --out / --lr / --tag values; the intent JSON only in args.lr and args.tag;
  5. the job's input-check block (model line .. DRY rows) run on real GPU dry_inputs.txt logs: the sweep's none@4.8e-3 log
     of the same seed -> exit 0, the other seed's -> exit 3 (DRY rows), an all6-lam1@6e-4 log -> exit 3 (prior target).
    python jobcheck_x1.py"""
import difflib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "rb6_505m"
ok = True


def say(cond, msg):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


sys.argv = [sys.argv[0]]   # the X1 generator module: import only (its main() is not called)
sys.path.insert(0, str(B))
import make_jobs_rb6_505m_x1 as X   # noqa: E402  (asserts the frozen shas and that make_jobs_rb6scale runs nothing on import)
M = X.M
sys.path.insert(0, str(R / "code" / "tools"))
import make_rb6m_tasks as G   # noqa: E402
import make_rb6_tasks as P    # noqa: E402
assert Path(G.__file__).resolve().parent == (R / "code" / "tools").resolve()
pin = M.WANT_SHA["pretrain_range.py"]
assert M.snapshot(R) == "7f1d06b"

OLD, NEW = "4.8e-3", "2.4e-3"
sub = lambda t: t.replace(OLD, NEW).replace('"lr":0.0048', '"lr":0.0024')


def kind(a):
    if a.startswith("--job-name="):
        return "job name"
    if a.startswith("'{\"code_sha\""):
        return "intent (lr + tag)"
    if "/out/" in a:
        return "out dir"
    if "rb6tb-505m-2pct-k2-none-lr" in a:
        return "tag"
    if a.rstrip(",") == OLD:
        return "lr"
    return "UNEXPLAINED"


for s in X.SEEDS:
    newf = R / f"job_rb6tb-505m-2pct-k2-none-lr{NEW}-s{s}.sbatch"
    reff = R / f"job_rb6tb-505m-2pct-k2-none-lr{OLD}-s{s}.sbatch"
    assert (R / f"SUBMITTED_rb6tb-505m-2pct-k2-none-lr{OLD}-s{s}").exists()
    t, ref = newf.read_text(), reff.read_text()
    say(subprocess.run(["bash", "-n", str(newf)]).returncode == 0, f"s{s}: bash -n {newf.name}")
    r_old = M.render(G, P, R, "505m", "none", OLD, s, pin)[2]
    r_new = M.render(G, P, R, "505m", "none", NEW, s, pin)[2]
    say(r_old == ref and r_new == t, f"s{s}: render() = the submitted sweep file {reff.name} and = the X1 file, byte for byte")
    a, b = ref.split(), t.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pairs, bad = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op != "replace" or i2 - i1 != j2 - j1:
            bad.append((op, a[i1:i2], b[j1:j2]))
            continue
        pairs += list(zip(a[i1:i2], b[j1:j2]))
    expl = [(x, y, kind(x)) for x, y in pairs]
    bad += [(x[:60], y[:60]) for x, y, k in expl if k == "UNEXPLAINED" or sub(x) != y]
    cnt = Counter(k for _, _, k in expl)
    say(not bad and len(a) == len(b) and sub(ref) == t,
        f"s{s}: token diff vs {reff.name}: {len(a)} tokens each, {len(pairs)} differ, all = 4.8e-3 -> 2.4e-3 in "
        f"{dict(sorted(cnt.items()))}; inserted / deleted / unexplained {len(bad)}; whole file = template with the lr "
        f"substituted: {sub(ref) == t}")
    for x, y, k in expl:
        print(f"    [{k}] {x[:110]} -> {y[:110]}" if k != "intent (lr + tag)" else f"    [{k}] (intent JSON, {len(x)} chars)")
    ld = [l for l in difflib.unified_diff(ref.splitlines(), t.splitlines(), lineterm="", n=0)
          if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    print(f"    line diff: {len(ld) // 2} lines changed: " + "; ".join(sorted({l[1:28] for l in ld if l[0] == '-'})))
    args = lambda u: re.search(r'^ARGS="(.*)"$', u, re.M).group(1).split()
    A, Ar = args(t), args(ref)
    diff_flags = sorted({Ar[i - 1] for i in range(len(Ar)) if Ar[i] != A[i]}) if len(A) == len(Ar) else None
    say(diff_flags == ["--lr", "--out", "--tag"] and A[A.index("--lr") + 1] == NEW and A[A.index("--seed") + 1] == str(s),
        f"s{s}: ARGS differ only in the values of {diff_flags}; --lr {A[A.index('--lr') + 1]} --seed {A[A.index('--seed') + 1]}")
    intent = lambda u: json.loads(re.search(r"^  \"\$OUT/check.json\" '(\{.*\})' <<'PY'", u, re.M).group(1))
    I, Ir = intent(t), intent(ref)
    top = sorted(k for k in set(I) | set(Ir) if I.get(k) != Ir.get(k))
    ak = sorted(k for k in set(I["args"]) | set(Ir["args"]) if I["args"].get(k) != Ir["args"].get(k))
    say(top == ["args"] and ak == ["lr", "tag"] and I["args"]["lr"] == 0.0024 and I["args"]["tag"] == f"_rb6tb-505m-2pct-k2-none-lr{NEW}"
        and I["lm0_ref"] == M.LM0["505m"][s] and I["args"]["seed"] == s and I["active"] == [] and I["n_params"] == 505437184,
        f"s{s}: intent differs only in {top} / args {ak} (lr {I['args']['lr']}, tag {I['args']['tag']}); lm0_ref {I['lm0_ref']:.6f} "
        f"= the 505M CPU reference of seed {s}; no active prior; n_params {I['n_params']}")
    # the input-check block on real GPU dry logs
    lines = t.splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("grep -qxF 'model:"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith('echo "## dry_inputs rows = the reference'))
    block = "\n".join(lines[i0:i1 + 1])
    for src, want in ((f"none-lr{OLD}-s{s}", 0), (f"none-lr{OLD}-s{1 - s}", 3), (f"all6-lam1-lr6e-4-s{s}", 3)):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "dry_inputs.txt").write_text((R / "out" / f"rb6tb-505m-2pct-k2-{src}" / "dry_inputs.txt").read_text())
            o = subprocess.run(["bash", "-c", f"set -euo pipefail\nOUT={d}\n{block}"], capture_output=True, text=True)
        say(o.returncode == want, f"s{s}: input-check block on the GPU dry log of {src}: exit {o.returncode} (want {want}) "
            f"{(o.stdout + o.stderr).strip().splitlines()[-1][:90] if (o.stdout + o.stderr).strip() else ''}")
print("JOBCHECK X1", "PASS" if ok else "FAIL")
