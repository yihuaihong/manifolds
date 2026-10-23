"""Checks of the 24 rendered rb6_190m_ext job files against the 16 submitted rb6_190m job files (the audited template):
bash -n; per job, the line-level diff to the main-batch job of the same seed and the nearest arm (none -> none@6e-4,
all6-lam1 / all6-minus-clock -> all6-lam1@6e-4, only-numbers -> only-numbers@6e-4, only-clock -> none@6e-4) is confined to
the allowed lines (job name, header comments, the R / OUT / TAG line, the kill-switch comment, ARGS, the numbers-target
assert, the per-domain prior / shuf / lam loop, the CHECK_PY call / intent lines); the CHECK_PY heredoc is byte-identical;
the ARGS differ only by the arm flags / lr / out / tag; the intent JSON differs only in the keys the arm / lr determine.
    python jobcheck_ext.py"""
import difflib
import json
import re
import subprocess
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch/runs")
NEAR = {"none": "none-lr6e-4", "all6-lam1": "all6-lam1-lr6e-4", "all6-minus-clock": "all6-lam1-lr6e-4",
        "only-numbers": "only-numbers-lr6e-4", "only-clock": "none-lr6e-4"}
ALLOWED = (r"^#SBATCH --job-name=", r"^# rb6_190m", r"^# code = ", r"^# llmadd_torch/make_jobs", r"^# except this cell",
           r"^R=", r"^# kill switch", r"^ARGS=", r"^! grep -q '\^target features'", r"^grep -qxF 'target features",
           r"^for sp in ", r'^"\$PY" - "\$OUT/\$TAG/run_whole_', r"^  \"\$OUT/check.json\" '")
ok = True
jobs = sorted((B / "rb6_190m_ext").glob("job_*.sbatch"))
assert len(jobs) == 24, len(jobs)


def heredoc(t):
    return t[t.index("<<'PY' | tee"):t.index("\nPY\necho")]


def intent_of(t):
    return json.loads(re.search(r"^  \"\$OUT/check.json\" '(\{.*\})' <<'PY'", t, re.M).group(1))


def args_of(t):
    return re.search(r'^ARGS="(.*)"$', t, re.M).group(1).split()


for j in jobs:
    m = re.match(r"job_rb6tb-190m-2pct-k2-(.+)-lr(.+)-s(\d)\.sbatch", j.name)
    arm, lr, s = m[1], m[2], m[3]
    t = j.read_text()
    rc = subprocess.run(["bash", "-n", str(j)]).returncode
    ref = (B / "rb6_190m" / f"job_rb6tb-190m-2pct-k2-{NEAR[arm]}-s{s}.sbatch").read_text()
    d = [l for l in difflib.unified_diff(ref.splitlines(), t.splitlines(), lineterm="", n=0)
         if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    bad = [l for l in d if not any(re.match(a, l[1:]) for a in ALLOWED)]
    hd = heredoc(t) == heredoc(ref)
    # ARGS: the reference's tokens minus its arm flags / lr / out / tag = this job's minus the same
    strip = lambda a: [w for i, w in enumerate(a) if not (w.startswith(("--target", "--schedule", "--window", "--site",
                       "--lam_rule", "--prior_domains", "--lam", "--lr", "--out", "--tag")) or (i and a[i - 1] in (
                       "--target", "--schedule", "--window", "--site", "--lam_rule", "--prior_domains", "--lam", "--lr",
                       "--out", "--tag")))]
    args_ok = strip(args_of(t)) == strip(args_of(ref)) and f"--lr {lr}" in " ".join(args_of(t))
    I, Ir = intent_of(t), intent_of(ref)
    keys = {k for k in set(I) | set(Ir) if I.get(k) != Ir.get(k)}
    akeys = {k for k in set(I["args"]) | set(Ir["args"]) if I["args"].get(k) != Ir["args"].get(k)}
    keys_ok = keys <= {"args", "lam_domains", "active", "events", "on_frac", "lam_rule"} and akeys <= {
        "lr", "tag", "target", "prior_domains", "lam", "schedule", "lam_rule"}
    good = rc == 0 and not bad and hd and args_ok and keys_ok
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {j.name}: bash -n {rc}; diff to {NEAR[arm]} s{s}: {len(d)} lines, outside the allowed "
          f"set {len(bad)}; CHECK_PY heredoc identical {hd}; ARGS = template except arm / lr / out / tag {args_ok}; intent "
          f"keys changed {sorted(keys)} / args {sorted(akeys)}" + (f"\n    {bad[:3]}" if bad else ""))
    print(f"    intent active {I['active']} lam_domains {I['lam_domains']} lr {I['args']['lr']} prior_domains "
          f"'{I['args']['prior_domains']}' target {I['args']['target']}")
print("JOBCHECK", "PASS" if ok else "FAIL")
