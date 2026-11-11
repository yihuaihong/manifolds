"""pt6comb test: every pt6comb job body = its ptdom STAGE-1 twin (make_jobs_ptdom.render, UNPATCHED: batch B1, none -> none@lr,
a prior arm -> only-countries@lr, a ptdom seed) after the registered substitutions only, and the patching leaves make_jobs_ptdom /
make_jobs_ptdom2 intact:
  T1  L1 (24), C (A2: 80, from a synthetic decided picks file at P 4e-3 / N 8e-3 incl. the same-lr pair; the cell lists of four (P, N) cases), L1x (12: every extension lr) and the two smoke
      bodies render; bash -n; every differing line falls in a registered category (names / slot / log, comments, the R / OUT / TAG /
      SEED line, the HOLD line, the GPU guard text, ARGS, the arm's dry block, the site-rule count, the DRY reference, the CHECK call
      (run file name, intent), the CHECK_PY body); the CHECK_PY body differs from ptdom's by exactly the counted CHECK_PY_6 edits
      (the full unified diff is printed)
  T2  after all those patched renders, make_jobs_ptdom.render re-renders all 240 submitted ptdom stage-1 job files byte for byte, and
      make_jobs_ptdom2.render2 the 88 submitted ptdom2 part-S job files byte for byte
  T3  the ARGS of every rendered job: --target / --prior_domains / --lam / --domains / --lr / --seed / no --save_model as the arm wants
-> render_diff.out"""
import difflib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
sys.argv = [sys.argv[0]]
import make_jobs_pt6comb as M6   # noqa: E402
MJ, M2 = M6.MJ, M6.M2
P = M6.load_P()
RF = M6.refs6("B")
RF0 = MJ.refs("B1")
res = []


def ok(name, cond, val=""):
    res.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val != "" else ""))


CATS = [  # (category, regex a ptdom twin line and its pt6comb line must both match)
    ("sbatch name / log", r"^#SBATCH --(job-name|output)="),
    ("comment", r"^#"),
    ("R / OUT / TAG / SEED", r"^R=.*; OUT=.*; TAG=.*; SEED="),
    ("HOLD", r'^if \[ -e "\$R/HOLD" \]'),
    ("GPU guard text", r'^case "\$GPU" in'),
    ("ARGS", r'^ARGS="'),
    ("dry block: numbers target", r"^(!\s*)?grep -q(xF)? '(\^target features|target features \(1000, 10\))'"),
    ("dry block: domain specs", r"^for sp in "),
    ("dry block: numbers target line", r"""^grep -qF '"numbers": \{"target": """),
    ("site-rule count", r'^\[ "\$\(grep -cE .\^  site rule: '),
    ("DRY reference", r'^\[ "\$n" = \d+ \] && \[ "\$got" = [0-9a-f]+ \]|^echo "## dry_inputs rows = the reference'),
    ("CHECK call: run file", r'^"\$PY" - "\$OUT/\$TAG/run_whole_'),
    ("CHECK call: intent", r"""^  "\$OUT/check.json" '\{"""),
]


def cat(a, b):
    for nm, rx in CATS:
        if re.match(rx, a) and re.match(rx, b):
            return nm
    return None


def classify(twin, mine):
    """Differences outside the CHECK_PY block, by category; the CHECK_PY block compared separately."""
    sep = "<<'PY' | tee \"$OUT/check.txt\"\n"
    tw_pre, tw_cp = twin.split(sep, 1)
    mi_pre, mi_cp = mine.split(sep, 1)
    a, b = tw_pre.splitlines(), mi_pre.splitlines()
    cats, bad = {}, []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op != "replace" or (i2 - i1) != (j2 - j1):
            bad.append((op, a[i1:i2], b[j1:j2]))
            continue
        for x, y in zip(a[i1:i2], b[j1:j2]):
            c = cat(x, y)
            if c is None:
                bad.append(("line", x[:120], y[:120]))
            else:
                cats[c] = cats.get(c, 0) + 1
    cp_ok = tw_cp.split("\nPY\n", 1)[0] + "\n" == MJ.CHECK_PY and mi_cp.split("\nPY\n", 1)[0] + "\n" == M6.CHECK_PY_6 \
        and tw_cp.split("\nPY\n", 1)[1] == mi_cp.split("\nPY\n", 1)[1]
    return cats, bad, cp_ok


def twin(arm, lr, seed):
    a0 = "none" if arm == "none" else "only-countries"
    s0 = seed if seed < 8 else seed - 8
    return MJ.render(P, "B1", a0, lr, s0, RF0[s0], M6.COMMIT, 0, MJ.SLOT0["B1"], "1", False)[2]


# T1
jobs = []
for i, (a, lr, s) in enumerate(M6.cells("L1")):
    jobs.append(("L1", a, lr, s, False, M6.render6(P, "L1", "B", a, lr, s, RF[s], M6.COMMIT, 0, i % 8)))
with tempfile.TemporaryDirectory() as tmp:
    nC = {}
    for P_, N_ in (("2e-3", "4e-3"), ("4e-3", "8e-3"), ("2e-3", "2e-3"), ("4e-3", "2e-3")):   # A2: the same-lr pair, merged when equal
        pk = Path(tmp) / f"picks_{P_}_{N_}.json"
        C = [list(c) for c in M6.c_cells(P_, N_)]
        pk.write_text(json.dumps({"design": "pt6comb", "status": "decided", "stage_written": "C", "comp": "B", "x": None, "C": C,
                                  "pick_all6": P_, "pick_none": N_}))
        pk.chmod(0o444)
        cl = M6.cells("C", "B", str(pk))
        names_ = [M6.name6("B", a, lr, s) for a, lr, s in cl]
        nC[(P_, N_)] = (len(cl), len(set(names_)), sorted({(a, lr) for a, lr, _ in cl if a in ("none", "all6")}))
        if (P_, N_) == ("4e-3", "8e-3"):
            for i, (a, lr, s) in enumerate(cl):
                jobs.append(("C", a, lr, s, False, M6.render6(P, "C", "B", a, lr, s, RF[s], M6.COMMIT, 0, i % 8)))
    pk2 = Path(tmp) / "picks2.json"
    pk2.write_text(json.dumps({"design": "pt6comb", "status": "decided", "stage_written": "L1x", "comp": "B", "x": None,
                               "E": [["all6", "8e-3"], ["none", "1.6e-2"], ["none", "1e-3"]]}))
    pk2.chmod(0o444)
    for i, (a, lr, s) in enumerate(M6.cells("L1x", "B", str(pk2))):
        jobs.append(("L1x", a, lr, s, False, M6.render6(P, "L1x", "B", a, lr, s, RF[s], M6.COMMIT, 0, i % 8)))
for a, lr, s in M6.SMOKE:
    jobs.append(("smoke", a, lr, s, True, M6.render6(P, "smoke", "B", a, lr, s, RF[s], M6.COMMIT, 0, 0, None, True)))
allcats, allbad, cp_bad, nbash = {}, [], 0, 0
for part, a, lr, s, sm, (name, tag, text, args, I) in jobs:
    lr_t = lr if lr in ("1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2") else "2e-3"
    tw = twin(a, lr_t, s)
    cats, bad, cp_ok = classify(tw, text)
    for k, v in cats.items():
        allcats[k] = allcats.get(k, 0) + v
    allbad += [(name,) + b for b in bad]
    cp_bad += not cp_ok
    nbash += subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0
ok("T0 (A2) part C renders the same-lr pair none@2e-3 / all6@2e-3 once: (P, N) = (2e-3, 4e-3) 72 runs, (4e-3, 8e-3) 80, (2e-3, 2e-3) 64, "
   "(4e-3, 2e-3) 72; every name unique; the none / all6 cells as expected",
   nC[("2e-3", "4e-3")][:2] == (72, 72) and nC[("4e-3", "8e-3")][:2] == (80, 80) and nC[("2e-3", "2e-3")][:2] == (64, 64)
   and nC[("4e-3", "2e-3")][:2] == (72, 72) and nC[("4e-3", "8e-3")][2] == [("all6", "2e-3"), ("all6", "4e-3"), ("none", "2e-3"), ("none", "8e-3")]
   and nC[("2e-3", "2e-3")][2] == [("all6", "2e-3"), ("none", "2e-3")], {k: v[:2] for k, v in nC.items()})
ok(f"T1 {len(jobs)} job bodies (L1 24, C 80 at P 4e-3 / N 8e-3, L1x 12, smoke 2) render, bash -n {nbash}/{len(jobs)}; every differing line from the ptdom twin "
   "falls in a registered category; the CHECK_PY body = ptdom's CHECK_PY (twin) / CHECK_PY_6 (pt6comb) exactly",
   nbash == len(jobs) and not allbad and not cp_bad, {"categories (lines over all jobs)": allcats, "unregistered": allbad[:4], "CHECK_PY mismatches": cp_bad})
print("     the registered CHECK_PY_6 edits (unified diff of ptdom's CHECK_PY -> CHECK_PY_6):")
for l in difflib.unified_diff(MJ.CHECK_PY.splitlines(), M6.CHECK_PY_6.splitlines(), "CHECK_PY (ptdom)", "CHECK_PY_6 (pt6comb)", n=0, lineterm=""):
    print("     " + l[:230])
# T3
bad3 = []
for part, a, lr, s, sm, (name, tag, text, args, I) in jobs:
    f = P.flags(args)
    pri = M6.pri_of(a, "B")
    want = {"--target": "helix3" if "numbers" in pri else "none", "--prior_domains": ",".join(pri) or None, "--lam": "1" if pri else None,
            "--domains": "numbers,s4,dim,cities,countries,people", "--lr": lr, "--seed": str(s), "--save_model": None,
            "--stop_after": "8000" if sm else None, "--extra_lines": "2", "--lam_rule": "each" if pri else None}
    d = {k: (f.get(k), v) for k, v in want.items() if f.get(k) != v}
    if d:
        bad3.append((name, d))
ok("T3 every job's ARGS: target / prior_domains / lam / domains / lr / seed / no --save_model / stop_after (smoke only) as the arm wants",
   not bad3, bad3[:3])
# T2
n0, diff0 = 0, []
for b in MJ.BATCHES:
    if b.startswith("C6"):
        continue
    for i, (a, lr, s) in enumerate(MJ.cells(b, "1")):
        f = MJ.R / f"job_{MJ.job_name(b, a, lr, s)}.sbatch"
        if not f.exists():
            continue
        slot = MJ.SLOT0[b] + i % MJ.SLOTS
        t = MJ.render(P, b, a, lr, s, MJ.refs(b)[s], M6.COMMIT, 0, slot, "1", False)[2]
        n0 += 1
        if t != f.read_text():
            diff0.append(f.name)
ok("T2 make_jobs_ptdom.render (unpatched, after every pt6comb render) re-renders the submitted ptdom stage-1 job files byte for byte",
   n0 == 240 and not diff0 and not any(k.startswith("C6") for k in MJ.BATCHES), (n0, diff0[:3]))
n2, diff2 = 0, []
for i, (b, a, lr, s, rr) in enumerate([(b, a, lr, s, 0) for b, a, lr, s in M2.cells("S")]):
    f = M2.R / f"job_{M2.name2(b, a, lr, s)}.sbatch"
    t = M2.render2(P, "S", b, a, lr, s, M2.refs2(b)[s], M2.COMMIT, 0, i % M2.N_SLOTS)[2]
    n2 += 1
    if t != f.read_text():
        diff2.append(f.name)
ok("T2 make_jobs_ptdom2.render2 re-renders the 88 submitted ptdom2 part-S job files byte for byte", n2 == 88 and not diff2, (n2, diff2[:3]))
print(f"RENDER DIFF {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
