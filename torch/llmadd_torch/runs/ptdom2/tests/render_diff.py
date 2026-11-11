"""ptdom2 test: the job bodies.
  1. make_jobs_ptdom.render, imported read-only by make_jobs_ptdom2 (UNPATCHED), re-renders every submitted ptdom job file
     (runs/ptdom/job_ptdom-*.sbatch; its slot, stage and re-run index parsed from the file) byte for byte: the function
     ptdom2 renders through IS ptdom's job body.
  2. every ptdom2 job (parts S, R1, the pilot bodies, and R1x / R2 from synthetic picks files covering every extension lr and
     every R domain FINAL) against its ptdom twin (the same function unpatched: the true-target twin's arm, the same lr, a ptdom
     seed (s < 8: s; else s - 8), ptdom's stage-1 body (A1.9: every ptdom2 part is a stage-1 body): after the registered substitutions (run name, tag, out
     dir, slot, seed, the HOLD_<part> line, the edited comment lines, --prior_shuf / --stop_after, the seed-dependent intent
     fields and DRY hash) the texts must be identical, except a shuffled arm's CHECK_PY (= make_jobs_ptdom2.CHECK_PY_SHUF; its
     exact diff to ptdom's CHECK_PY is printed) and its dry-block spec (prior true / shuf true). One raw unified diff per arm
     kind is printed.
  3. job-file facts: bash -n; the B200 header (account 520 priority, --constraint b200, preemption=yes, singleton slot chains
     llmadd_ptdom2-b200-slot0..7, 3 h, requeue, append); the parts' seeds; names / out dirs only under ptdom2; HOLD / HOLD_<part>.
-> render_diff.out
"""
import difflib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
sys.argv = [sys.argv[0]]
import make_jobs_ptdom2 as M2   # noqa: E402
MJ = M2.MJ
P = M2.load_P()
COMMIT = M2.COMMIT
res = []


def T(name, ok, info=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info != "" else ""), flush=True)


# ---------------------------------------------------------------- 1. ptdom's own job files
files = sorted((M2.R0).glob("job_ptdom-*.sbatch"))
same, differ = 0, []
RF0 = {b: MJ.refs(b) for b in MJ.BATCHES}
for f in files:
    txt = f.read_text()
    m = re.match(r"job_(ptdom-(A|B1|B2|B3)-(none|only-[a-z_]+)-lr([0-9.e-]+)-s(\d+)(?:-r(\d+))?)\.sbatch", f.name)
    b, arm, lr, s, rr = m.group(2), m.group(3), m.group(4), int(m.group(5)), int(m.group(6) or 0)
    slot = int(re.search(r"#SBATCH --job-name=llmadd_ptdom-b200-slot(\d+)", txt).group(1))
    stage = re.search(r"^# ptdom A3 / A4 batch \S+ stage (\S+):", txt, re.M).group(1)
    _, _, t2, _, _ = MJ.render(P, b, arm, lr, s, RF0[b][s], COMMIT, rr, slot, stage, False)
    (same := same + 1) if t2 == txt else differ.append(f.name)
T(f"1. make_jobs_ptdom.render (imported, unpatched) re-renders all {len(files)} submitted ptdom job files byte for byte", not differ and same == len(files),
  f"{same} identical; differ {differ[:5]}")

# ---------------------------------------------------------------- 2. ptdom2 jobs vs their ptdom twins
RF2 = {b: M2.refs2(b) for b in MJ.BATCHES}


def synth_picks(tmp, stage):
    """Synthetic picks files (design ptdom2, chmod 444) for the R1x and R2 cell lists."""
    if stage == "R1x":
        # A1: every prior 8e-3 + every none step R1pre did not use (A down 5e-4, B2 up 3.2e-2, B3 up 1.6e-2 and down = fixB 1e-3)
        E = [[b, f"only-{d}", "8e-3"] for d, b in M2.R_DOMS.items()] + [["A", "none", "5e-4"], ["B2", "none", "3.2e-2"],
                                                                         ["B3", "none", "1.6e-2"], ["B3", "none", "1e-3"]]
        pk = {"design": "ptdom2", "amendment": "A1", "status": "decided", "stage_written": "R1x", "E": E, "C": [], "domains": {}}
    else:
        picks = {"stars": ("8e-3", "1.6e-2"), "landmarks": ("1e-3", "2e-3"), "compass": ("4e-3", "3.2e-2"), "prefixes": ("2e-3", "1e-3"),
                 "ranks": ("5e-4", "8e-3"), "scalar_adj": ("1e-3", "1.6e-2")}
        dom = {d: {"status": "FINAL", "pick_prior": p, "pick_none": n} for d, (p, n) in picks.items()}
        C = []
        for d, (p, n) in picks.items():
            b = M2.R_DOMS[d]
            for x in ([b, f"only-{d}", p], [b, f"only-{d}-shuf", p], [b, "none", n]):
                if x not in C:
                    C.append(x)
        pk = {"design": "ptdom2", "status": "decided", "stage_written": "R2", "E": [], "C": C, "domains": dom}
    f = Path(tmp) / f"picks_{stage}.json"
    f.write_text(json.dumps(pk))
    f.chmod(0o444)
    return f


def norm(text, name, tag, seed, part, batch):
    """Replace the registered per-cell tokens by placeholders (both texts)."""
    t = text.replace(f"{M2.R}/out/{name}", "<OUT>").replace(f"{M2.R}/pilot/{name}", "<OUT>").replace(f"{M2.R0}/out/{name}", "<OUT>")
    t = t.replace(str(M2.R), "<R>").replace(str(M2.R0), "<R>").replace("runs/ptdom2/", "runs/ptdom/")
    t = t.replace(name, "<NAME>").replace(tag, "<TAG>")
    t = re.sub(r"llmadd_ptdom2?-b200-slot\d+", "<SLOT>", t)
    t = re.sub(r"singleton slot \d+ \(batch \S+: slots \d+\.\.\d+; [^)]*\)\.", "singleton slot <SLOT>.", t)
    t = re.sub(r"--seed \d+ ", "--seed <SEED> ", t)
    t = re.sub(r"SEED=\d+;", "SEED=<SEED>;", t)
    t = re.sub(r"run_whole_none_seed\d+_", "run_whole_none_seed<SEED>_", t)
    t = re.sub(r"seed \d+; data = numbers", "seed <SEED>; data = numbers", t)
    t = re.sub(r'\[ "\$n" = \d+ \] && \[ "\$got" = [0-9a-f]+ \]', '[ "$n" = <N> ] && [ "$got" = <DRY> ]', t)
    t = re.sub(r"## dry_inputs rows = the reference \([0-9a-f]+, \d+ lines\)", "## dry_inputs rows = the reference (<DRY>)", t)
    t = re.sub(r" --prior_shuf [a-z_]+", "", t)
    t = t.replace(f" --stop_after {M2.PILOT_STOP}", "")
    t = re.sub(r"only the ([a-z_]+)-shuf prior", r"only the \1 prior", t)
    t = re.sub(r'\[ -e "\$R/HOLD_[A-Za-z0-9]+" \]', '[ -e "$R/HOLD_<X>" ]', t)
    t = re.sub(r"# runs/ptdom/HOLD_\S+ \((A4: this batch|this part) only\) exists", "# runs/ptdom/HOLD_<X> (one part / batch only) exists", t)
    t = t.replace("runs/ptdom/HOLD (every part) or", "runs/ptdom/HOLD (every batch) or")
    t = t.replace("# ptdom2 part " + part + " (the ptdom A3-A6.1 job body), batch " + batch + ":", "# <HEAD> batch " + batch + ":")
    t = re.sub(r"# ptdom A3 / A4 batch " + batch + r" stage \S+:", "# <HEAD> batch " + batch + ":", t)
    t = t.replace("# spec and pre-registered verdict runs/ptdom/AUDIT.md; rendered by llmadd_torch/make_jobs_ptdom2.py through make_jobs_ptdom.render",
                  "# spec and pre-registered screen / verdict runs/ptdom/AUDIT.md (Amendments A3, A4); generated by llmadd_torch/make_jobs_ptdom.py")
    return t


def intent_of(text):
    return json.loads(re.search(r"\"\$OUT/check.json\" '(\{.*\})' \"\$ARGS\" <<'PY'", text).group(1))


def strip_intent(text):
    return re.sub(r"\"\$OUT/check.json\" '(\{.*\})' \"\$ARGS\" <<'PY'", "\"$OUT/check.json\" '<INTENT>' \"$ARGS\" <<'PY'", text)


def cp_of(text):
    return text.split("<<'PY' | tee \"$OUT/check.txt\"\n", 1)[1].split("\nPY\n", 1)[0] + "\n"


SEEDDEP = ("lm0_ref", "data_sha0", "dom", "table_sha")
shown = set()
with tempfile.TemporaryDirectory() as tmp:
    parts = {"S": M2.cells("S"), "S0": M2.cells("S0"), "R1": M2.cells("R1"), "R1pre": M2.cells("R1pre"), "R1x": M2.cells("R1x", synth_picks(tmp, "R1x")),
             "R2": M2.cells("R2", synth_picks(tmp, "R2")),
             "pilot": list(M2.PILOT)}
    for part, cl in parts.items():
        n_ok, bad = 0, []
        for i, (b, arm, lr, s) in enumerate(cl):
            pil = part == "pilot"
            name, tag, txt, args, I = M2.render2(P, part, b, arm, lr, s, RF2[b][s], COMMIT, 0, i % M2.N_SLOTS, pil)
            s0 = s if s < 8 else s - 8
            n0, t0, txt0, a0, I0 = M2.twin_render(P, part, b, arm, lr, s0, RF0[b][s0], COMMIT, MJ.SLOT0[b])
            # the intent: equal except the registered per-cell fields
            Ic, I0c = json.loads(json.dumps(I)), json.loads(json.dumps(I0))
            for k in SEEDDEP:
                Ic.pop(k), I0c.pop(k)
            for k in ("seed", "tag"):
                Ic["args"].pop(k), I0c["args"].pop(k)
            if M2.is_shuf(arm):
                ok_sh = Ic.pop("shuf") == [M2.dom(arm)] and Ic["args"].pop("prior_shuf") == M2.dom(arm) and I0c["args"].pop("prior_shuf") == ""
            else:
                ok_sh = "shuf" not in Ic
            if pil:
                ok_sh = ok_sh and Ic.pop("pilot") is True and I0c.pop("pilot") is False and Ic["args"].pop("stop_after") == M2.PILOT_STOP and I0c["args"].pop("stop_after") == -1
            ok_seed = I["args"]["seed"] == s and all(I[k] == RF2[b][s][{"lm0_ref": "lm0", "data_sha0": "dsha0", "dom": "dom", "table_sha": "table_sha"}[k]] for k in SEEDDEP)
            cpx, cp0 = cp_of(txt), cp_of(txt0)
            cp_ok = cpx == (M2.CHECK_PY_SHUF if M2.is_shuf(arm) else MJ.CHECK_PY) and cp0 == MJ.CHECK_PY
            txs = txt
            if M2.is_shuf(arm):   # the two registered differences of a shuffled arm, swapped back BEFORE the normalisation
                assert txs.count(cpx) == 1 and txs.count(f" {M2.dom(arm)}:true:true:1.0") == 1
                txs = txs.replace(cpx, cp0).replace(f" {M2.dom(arm)}:true:true:1.0", f" {M2.dom(arm)}:true:false:1.0")
            n1 = norm(strip_intent(txs), name, tag, s, part, b)
            n2 = norm(strip_intent(txt0), n0, t0, s0, part, b)
            dl = [l for l in difflib.unified_diff(n2.splitlines(), n1.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
            good = not dl and Ic == I0c and ok_sh and ok_seed and cp_ok
            n_ok += good
            if not good:
                bad.append((name, dl[:6], Ic == I0c, ok_sh, ok_seed, cp_ok))
            kind = ("pilot-" if pil else "") + ("none" if arm == "none" else "shuf" if M2.is_shuf(arm) else "true")
            if kind not in shown:
                shown.add(kind)
                raw = [l for l in difflib.unified_diff(txt0.splitlines(), txt.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++"))]
                print(f"--- raw diff, {kind} arm: ptdom twin {n0} (seed {s0}) -> ptdom2 {name} ({len([l for l in raw if l[:1] in '+-'])} changed lines; "
                      "long lines cut at 200 characters)")
                for l in raw:
                    print("    " + l[:200])
        if part == "R2":   # A2: a none cell Part S runs (A none@2e-3 at seeds 8-15; landmarks' synthetic none pick) is not rendered again
            s_names = {M2.name2(*c) for c in parts["S"]}
            T("A2: R2 never renders a cell of Part S (the synthetic C holds A none@2e-3: its 8 seeds are dropped, 136 of 144)",
              len(cl) == 136 and not {M2.name2(*c) for c in cl} & s_names)
        T(f"2. part {part}: {len(cl)} jobs = their ptdom twins after the registered substitutions (names, out dir, tag, slot, seed, HOLD_<part>, "
          f"comment lines, --prior_shuf, --stop_after, seed-dependent intent / DRY fields); shuffled arms: only CHECK_PY_SHUF and the dry spec",
          n_ok == len(cl), f"{n_ok}/{len(cl)}" + (f"; first bad {bad[0]}" if bad else ""))
        # 3. job-file facts
        f_ok = []
        for i, (b, arm, lr, s) in enumerate(cl):
            pil = part == "pilot"
            name, tag, txt, args, I = M2.render2(P, part, b, arm, lr, s, RF2[b][s], COMMIT, 0, i % M2.N_SLOTS, pil)
            hdr = txt.split("\n# ", 1)[0]
            seeds_ok = s in {"S": M2.S_SEEDS2, "S0": M2.S_SEEDS, "R1": M2.SEL, "R1pre": M2.SEL, "R1x": M2.SEL, "R2": M2.R2_SEEDS, "pilot": (0,)}[part]
            f_ok.append(subprocess.run(["bash", "-n"], input=txt.encode()).returncode == 0 and seeds_ok
                        and f"#SBATCH --job-name=llmadd_ptdom2-b200-slot{i % M2.N_SLOTS}\n#SBATCH --dependency=singleton\n" in hdr
                        and "#SBATCH --account=<slurm-account>" in hdr and "#SBATCH --constraint=b200" in hdr
                        and '#SBATCH --comment="preemption=yes"' in hdr and "#SBATCH --time=03:00:00" in hdr and "#SBATCH --requeue" in hdr
                        and "#SBATCH --open-mode=append" in hdr and f"#SBATCH --output={B}/logs/llmadd_{name}_%j.out" in hdr
                        and name.startswith("ptdom2-") and f'[ -e "$R/HOLD_{part}" ]' in txt and f"R={M2.R};" in txt
                        and ("/runs/ptdom/" not in txt.replace("/runs/ptdom/code/", "")) and (" --prior_shuf " in args) == M2.is_shuf(arm)
                        and ((" --save_model" in args) == M2.SAVE[part]))
        T(f"3. part {part}: bash -n, the B200 slot-chain header, seeds {sorted({s for *_, s in cl})}, names / out dirs / HOLD under ptdom2, "
          f"--prior_shuf exactly on the shuffled arms, --save_model = {M2.SAVE[part]}", all(f_ok), f"{sum(f_ok)}/{len(f_ok)}")
cpd = [l for l in difflib.unified_diff(MJ.CHECK_PY.splitlines(), M2.CHECK_PY_SHUF.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++"))]
print(f"--- CHECK_PY_SHUF vs ptdom's CHECK_PY ({len([l for l in cpd if l[:1] in '+-'])} changed lines):")
for l in cpd:
    print("    " + l[:220])
T("CHECK_PY_SHUF compiles; ptdom's CHECK_PY is byte-identical to the one in every ptdom job file", compile(M2.CHECK_PY_SHUF, "x", "exec") is not None
  and all(cp_of(f.read_text()) == MJ.CHECK_PY for f in files))
print(f"RENDER DIFF {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
