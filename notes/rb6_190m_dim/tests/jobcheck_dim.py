"""Job-file check of rb6_190m_dim (before submission): every job this batch can render against the SUBMITTED rb6_190m job of
the same arm (none -> rb6_190m none@4.8e-3, all6-lam1 -> rb6_190m all6-lam1@6e-4; render() of the same read-only
make_jobs_rb6scale.py) at the same seed (D1, seeds 0-3) or at seed s - 4 (D2, seeds 4-7). Allowed differences, and only
these: lr, seed (incl. what the generator derives from the seed: the DRY row hash, the intent's lm0_ref / data_sha0 / dom),
tag, out dir (R / OUT incl. the batch name in comments and paths), job name, nice. Method: the allowed substitutions are
applied to the submitted job's text (old -> new value, each counted), then the result must equal the new job's text
token for token except the intent JSON, which is compared decoded (differing keys must be in the allowed set). D1 is
read from the rendered job files; D2 is rendered in memory for every possible pick (the five none lrs) through the
generator's build() (nothing is written). Also: bash -n of every job and of the select job; the D1 jobs vs the submitted
rb6_190m_ext none@1.92e-2 jobs (another generator: only comment lines may differ after the substitutions).
    python jobcheck_dim.py"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
import make_jobs_rb6_190m_dim as J   # noqa: E402

R, MAIN, EXT = J.R, B / "runs" / "rb6_190m", B / "runs" / "rb6_190m_ext"
REF_LR = {"none": "4.8e-3", "all6-lam1": "6e-4"}
ALLOWED_INTENT = {"args.lr", "args.tag", "args.seed", "lm0_ref", "data_sha0", "dom"}
ok_all = True


def say(c, m):
    global ok_all
    ok_all &= bool(c)
    print(("PASS " if c else "FAIL ") + m, flush=True)


def intent_of(t):
    m = re.search(r"\"\$OUT/check\.json\" '(\{.*\})' <<'PY'", t)
    return m.group(1), json.loads(m.group(1))


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k == "args":
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def compare(new, ref, arm, lr, s, s_ref, refbatch, comments_only=False):
    """-> (ok, report)"""
    rlr = REF_LR[arm] if refbatch == "rb6_190m" else "1.92e-2"
    rtag, ntag = f"rb6tb-190m-2pct-k2-{arm}-lr{rlr}", f"rb6tb-190m-2pct-k2-{arm}-lr{lr}"
    dry_r, dry_n = J.M.DRY["190m"][s_ref][0], J.M.DRY["190m"][s][0]
    subs = [  # (old, new, category); applied in this order
        (f"#SBATCH --nice=20\n", f"#SBATCH --nice=5000\n", "nice"),
        (f"/runs/{refbatch}/", f"/runs/{J.BATCH}/", "out"),
        (f"/runs/{refbatch};", f"/runs/{J.BATCH};", "out"),
        (f"llmadd_{refbatch}-{arm}-lr{rlr}-s{s_ref}\n", f"llmadd_{J.BATCH}-{arm}-lr{lr}-s{s}\n", "job name"),
        (f"# {refbatch}: {arm} @ lr {rlr}, 190m, seed {s_ref};", f"# {J.BATCH}: {arm} @ lr {lr}, 190m, seed {s};", "comment lr/seed/out"),
        (f"runs/{refbatch}/", f"runs/{J.BATCH}/", "out"),
        (f"seed-{s_ref} rows", f"seed-{s} rows", "seed"),
        (f"{rtag}-s{s_ref}", f"{ntag}-s{s}", "tag/out"),
        (f"_seed{s_ref}_{rtag}", f"_seed{s}_{ntag}", "seed/tag"),
        (rtag, ntag, "tag"),
        (f"SEED={s_ref};", f"SEED={s};", "seed"),
        (f" --lr {rlr} --seed {s_ref} ", f" --lr {lr} --seed {s} ", "lr/seed"),
        (dry_r, dry_n, "seed (DRY row hash)"),
        (f"({dry_r[:12]}, 88 lines)", f"({dry_n[:12]}, 88 lines)", "seed (DRY row hash)"),
    ]
    t, counts = ref, {}
    for old, nw, cat in subs:
        if old == nw:
            continue
        k = t.count(old)
        t = t.replace(old, nw)
        counts[cat] = counts.get(cat, 0) + k
    ij_t, it = intent_of(t)
    ij_n, inn = intent_of(new)
    t = t.replace(ij_t, "<INTENT>")
    nn = new.replace(ij_n, "<INTENT>")
    fa, fb = flat(it), flat(inn)
    dk = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    tok_a, tok_b = t.split("\n"), nn.split("\n")
    if comments_only:   # another generator: its comment lines ('# ...', not '#SBATCH') are dropped before comparing
        keep = lambda ls: [l for l in ls if not (l.startswith("# ") or l == "#")]
        tok_a, tok_b = keep(tok_a), keep(tok_b)
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(tok_a, tok_b)) if a.split() != b.split()]
    lines_ok = len(tok_a) == len(tok_b) and not bad
    int_ok = set(dk) <= ALLOWED_INTENT and (fb.get("args.lr") == float(lr) and fb.get("args.seed") == s
                                            and fb.get("lm0_ref") == J.M.LM0["190m"][s])
    return lines_ok and int_ok, (f"lines {len(tok_b)} vs {len(tok_a)}, token-different lines after the allowed substitutions: "
                                 f"{[(i, a[:60], b[:60]) for i, a, b in bad][:3]}; substitutions {counts}; intent keys "
                                 f"differing {dk}")


# ---- D1: the rendered files vs rb6_190m none@4.8e-3 of the same seed (and vs rb6_190m_ext none@1.92e-2: comments only)
J.load_refs()
for s in J.D1_SEEDS:
    name = f"rb6tb-190m-2pct-k2-none-lr3.84e-2-s{s}"
    new = (R / f"job_{name}.sbatch").read_text()
    ref = (MAIN / f"job_rb6tb-190m-2pct-k2-none-lr4.8e-3-s{s}.sbatch").read_text()
    ok, rep = compare(new, ref, "none", "3.84e-2", s, s, "rb6_190m")
    say(ok, f"D1 {name} vs rb6_190m none@4.8e-3 s{s}: {rep}")
    ext = (EXT / f"job_rb6tb-190m-2pct-k2-none-lr1.92e-2-s{s}.sbatch").read_text()
    ok, rep = compare(new, ext, "none", "3.84e-2", s, s, "rb6_190m_ext", comments_only=True)
    say(ok, f"D1 {name} vs rb6_190m_ext none@1.92e-2 s{s} (another generator: comment lines only): {rep}")
    say(subprocess.run(["bash", "-n", str(R / f"job_{name}.sbatch")]).returncode == 0, f"bash -n {name}")
    say(new.count("#SBATCH --nice=5000\n") == 1 and "--nice=20" not in new, f"{name}: nice 5000 (one line)")
say(subprocess.run(["bash", "-n", str(J.SELECT_JOB)]).returncode == 0, "bash -n select_job.sbatch")

# ---- D2: rendered in memory for every possible pick, vs rb6_190m at seed s - 4
cm = (R / "select_code" / "COMMIT").read_text()
tool_sha = re.search(r"rb6scale_dim_summary\.py (\w{12})", cm).group(1)
for best in J.NONE_GRID:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"batch": J.BATCH, "status": "picked", "grid": list(J.NONE_GRID), "pick": {"none_dim": best},
                   "tool_sha": tool_sha}, fh)
    jobs, head, pin, b_ = J.build("D2", fh.name)
    names = [n for n, *_ in jobs]
    want = [f"rb6tb-190m-2pct-k2-{a}-lr{lr}-s{s}" for s in J.D2_SEEDS for a, lr in (("none", best), ("all6-lam1", "6e-4"))]
    say(b_ == best and names == want and all(t is not None for _, t, _, _ in jobs),
        f"D2 pick {best}: 8 jobs none@{best} / all6-lam1@6e-4 x seeds 4-7, none REUSED (no 190M seed 4-7 run exists)")
    for name, text, args, I in jobs:
        arm, lr = ("none", best) if name.startswith("rb6tb-190m-2pct-k2-none-") else ("all6-lam1", "6e-4")
        s = int(name.rsplit("-s", 1)[1])
        assert I["args"]["seed"] == s and name == f"rb6tb-190m-2pct-k2-{arm}-lr{lr}-s{s}", name
        ref = (MAIN / f"job_rb6tb-190m-2pct-k2-{arm}-lr{REF_LR[arm]}-s{s - 4}.sbatch").read_text()
        ok, rep = compare(text, ref, arm, lr, s, s - 4, "rb6_190m")
        say(ok, f"D2 {name} vs rb6_190m {arm}@{REF_LR[arm]} s{s - 4}: {rep}")
        with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as fj:
            fj.write(text)
        say(subprocess.run(["bash", "-n", fj.name]).returncode == 0 and text.count("#SBATCH --nice=5000\n") == 1,
            f"bash -n + nice 5000 {name}")
        Path(fj.name).unlink()
    Path(fh.name).unlink()
# a bad picks file is refused
for bad in ({"status": "incomplete"}, {"pick": {"none_dim": "7.68e-2"}}, {"tool_sha": "000000000000"}):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({**{"batch": J.BATCH, "status": "picked", "grid": list(J.NONE_GRID), "pick": {"none_dim": "1.92e-2"},
                      "tool_sha": tool_sha}, **bad}, fh)
    try:
        J.build("D2", fh.name)
        say(False, f"D2 with a bad picks file {bad}: not refused")
    except AssertionError as e:
        say(True, f"D2 with a bad picks file {bad}: refused ({str(e)[:70]})")
print("JOBCHECK DIM " + ("PASS" if ok_all else "FAIL"))
