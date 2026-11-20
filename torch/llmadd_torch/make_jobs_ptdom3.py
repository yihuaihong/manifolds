"""Torch jobs of runs/ptdom3 (exploratory screen, runs/ptdom3/AUDIT.md): which toy3 domain's manifold prior helps its own
exact-match read-out in 53M pretraining? Six toy3 domains (cyc12, torus, hcube, d6, box5, caesar; trainer Rev H, branch
ptdom3, commit ef7b8854, worktree /scratch/yh6210/llmadd-burst-ptdom3), one batch T3 = --domains numbers,<the six>, K = 2.

Cells (seed 0 only, no saved model; ptdom's stage-1 job body):
  none  @ 2e-3 / 4e-3 / 8e-3 / 1.6e-2                       4 runs (the shared reference of every gain)
  only-<d> @ 1e-3 / 2e-3 / 4e-3 for the six d               18 runs
  canary: alltoy3 (the six toy3 priors at once) @ 2e-3, 400 steps (--tokens 32 x 512 x 400, evals every 50, ckpt every 100),
          then readout_ptdom3.py --canary on its run JSON, then CHECK_PY_3 (one B200 job, <= 1 h; not part of any result)
Every job is rendered by make_jobs_ptdom.render (the frozen ptdom generator, sha asserted; pt6comb's technique: module names
re-pointed for one call, then exact counted text edits). CHECK_PY_3 = pt6comb's CHECK_PY_6 (ptdom's CHECK_PY + 7 counted edits,
incl. the toy read-outs + per-question hits check) + one counted edit: each toy3 domain's read-out set is logged at every eval
and is the metrics' read-out set. After training each job greps its own log for the trainer's 'step 0: toy3 true-prior gradient
norms' line (a printed mark only; the trainer itself asserts the norms are non-zero before any compute).
Run / out-dir name ptdom3-T3-<arm>-lr<lr>-s0[-r<k>]; Slurm llmadd_ptdom3-b200-slot<0..7> (singleton chains); canary
llmadd_ptdom3-canary, out dir runs/ptdom3/pilot/.
    python make_jobs_ptdom3.py --snapshot
    python make_jobs_ptdom3.py --check [--print NAME]       render all 23 jobs into runs/ptdom3/jobs/ (no submission)
    python make_jobs_ptdom3.py --test-only                   bash -n + sbatch --test-only of every rendered job
    python make_jobs_ptdom3.py --submit canary|screen        refuses while runs/ptdom3/HOLD exists; never re-submits a name
"""
import hashlib
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
FROZEN = {"make_jobs_ptdom.py": "6061b299129c", "make_jobs_ptdom2.py": "62d680144854", "make_jobs_pt6comb.py": "01d1ad3d2867"}
for f_, v_ in FROZEN.items():
    assert sha(B / f_) == v_, f"{f_} changed: {sha(B / f_)} != {v_}"
sys.path.insert(0, str(B))
import make_jobs_ptdom as MJ     # noqa: E402  (read only: render, CHECK_PY, constants)
import make_jobs_ptdom2 as M2    # noqa: E402  (read only: _edit)
import make_jobs_pt6comb as M6   # noqa: E402  (read only: CHECK_PY_6, dom keys of the toy kind)
for m_ in (MJ, M2, M6):
    assert Path(m_.__file__).resolve().parent == B.resolve()

R = B / "runs" / "ptdom3"
R0 = B / "runs" / "ptdom"                         # the frozen ptdom snapshot (11f61b9): every unchanged file is compared with it
REPO = Path("/scratch/yh6210/llmadd-burst-ptdom3")
COMMIT = "ef7b8854"                               # Rev H (branch ptdom3, not pushed)
NEW = {"pretrain_range.py": "a398fccbfe93", "toy3.py": "ddf037ed29fd", "toy3_nonce.json": "9172a4c7c116"}
WANT_SHA = {**MJ.WANT_SHA, **NEW}
DEP = tuple(MJ.DEP) + ("toy3.py", "toy3_nonce.json")   # Rev H's run-JSON dep_sha under REVT
PY, LADDER, HW, K = MJ.PY, MJ.LADDER, MJ.HW, 2
_edit = M2._edit
PREFIX, KEY = "ptdom3", "T3"
TOY3 = ("cyc12", "torus", "hcube", "d6", "box5", "caesar")
DOMS = TOY3
READ3 = {d: f"{d}_readout" for d in TOY3}         # the run JSON metrics: area / final3 over the domain's read-out set
SEEN3 = {d: f"{d}_seen" for d in TOY3}
READSET = {"cyc12": "held_nonid", "torus": "held_nonid", "hcube": "held_nonid", "d6": "held_fresh_nonid",
           "box5": "held_no_negtwin_nonid", "caesar": "held_nonid"}
GRID = {"none": ("2e-3", "4e-3", "8e-3", "1.6e-2"), "prior": ("1e-3", "2e-3", "4e-3")}
ARMS = ("none",) + tuple(f"only-{d}" for d in TOY3)
CANARY = ("alltoy3", "2e-3", 0)
SEED = 0
N_SLOTS = 8
DOM_KEYS3 = {**MJ.DOM_KEYS, "toy3": M6.DOM_KEYS6["toy"]}
STAGES = ("S", "canary")


def pri_of(arm):
    return () if arm == "none" else (TOY3 if arm == "alltoy3" else (arm[5:],))


def domains_arg():
    return "numbers," + ",".join(DOMS)


def tag3(arm, lr):
    return f"{PREFIX}-{KEY}-{arm}-lr{lr}"


def name3(arm, lr, seed, rr=0, canary=False):
    return (f"{PREFIX}-canary-" if canary else f"{PREFIX}-") + f"{KEY}-{arm}-lr{lr}-s{seed}" + (f"-r{rr}" if rr else "")


def slot_name(slot, canary=False):
    return f"llmadd_{PREFIX}-canary" if canary else f"llmadd_{PREFIX}-b200-slot{slot}"


def arm_flags(arm):
    pri = pri_of(arm)
    return "--target none" if not pri else f"--target none {MJ.W} --prior_domains {','.join(pri)} --lam 1"


def want_flags(arm):
    pri = pri_of(arm)
    if not pri:
        return {"--target": "none", "--schedule": None, "--window": None, "--site": None, "--lam_rule": None,
                "--prior_domains": None, "--prior_shuf": None, "--lam": None}
    return {"--target": "none", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx", "--lam_rule": "each",
            "--prior_domains": ",".join(pri), "--prior_shuf": None, "--lam": "1"}


def args_of3(arm, lr, seed, out, canary=False):
    tag = tag3(arm, lr)
    ck = MJ.PILOT_CKPT if canary else HW["b200"]["ckpt"]
    return (f"--data {MJ.DATA} --tokenizer {MJ.TOK} --tokenization whole --out {out}/{tag} {arm_flags(arm)} --arith_lines 1 "
            f"--domains {domains_arg()} --extra_lines {K} --lr {lr} --seed {seed} --tag _{tag} --ckpt_every {ck}"
            + (f" --tokens {32 * 512 * MJ.PILOT_STEPS} --eval_every {MJ.PILOT_EVAL}" if canary else ""))


def intent3(P, arm, lr, seed, ref, canary=False):
    pri = pri_of(arm)
    steps, ev_ = (MJ.PILOT_STEPS, MJ.PILOT_EVAL) if canary else (P.STEPS, P.EVAL_EVERY)
    events, on_frac = P.window_events(steps) if pri else ([], 0.0)
    if pri:
        assert events == ([[20, 1], [160, 0]] if canary else [[916, 1], [7324, 0]]), events
    lam_rule = "each" if pri else "norm"
    hw = HW["b200"]
    return {"batch": KEY, "code_sha": WANT_SHA["pretrain_range.py"], "dep_sha": {f: WANT_SHA[f] for f in DEP}, "steps": steps,
            "n_params": P.N_PARAMS, "grad_ckpt_on": False, "geo_layers": [2, 4], "read_layers": [2, 4], "lm_chunk": 0,
            "eval_every_steps": ev_, "lam_rule": lam_rule, "save_model": False, "pilot": bool(canary),
            "lam_domains": {d: (1.0 if d in pri else 0.0) for d in ("numbers",) + DOMS}, "K": K, "ext": list(DOMS),
            "comp": [], "kstep": [], "toy": list(DOMS), "readset": dict(READSET), "tables": [], "active": list(pri),
            "geo_tol": MJ.GEO_TOL, "evals": len(range(0, steps, ev_)) + ((steps - 1) % ev_ != 0), "events": events,
            "on_frac": on_frac, "lm0_ref": ref.get("lm0"), "data_sha0": ref.get("dsha0"), "dom": ref.get("dom"),
            "table_sha": ref.get("table_sha"), "n_fmt": {}, "cka_rise": MJ.CKA_RISE, "dp_tol0": MJ.DP_TOL0, "dp_tol": {},
            "numbers_share": [0.019, 0.024], "read": dict(READ3), "seen": dict(SEEN3), "gpu": hw["gpu"],
            "args": {"data": None, "tokenizer": str(MJ.TOK), "out": None, "tokenization": "whole", "target": "none", "site": "ctx",
                     "schedule": "window" if pri else "whole", "window": "0.05,0.4", "arith_lines": 1, "lr": float(lr),
                     "lam": 1.0 if pri else 5.0, "tokens": float(32 * 512 * MJ.PILOT_STEPS) if canary else 3e8, "bs": P.BS,
                     "seq": P.SEQ, "d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536, "seed": seed,
                     "tag": f"_{tag3(arm, lr)}", "save_model": False, "ckpt_every": MJ.PILOT_CKPT if canary else hw["ckpt"],
                     "geo_layers": None, "eval_every": ev_, "n_eval": 150, "grad_ckpt": "auto", "lm_chunk_tokens": -1,
                     "untie": False, "gate_thr": 0.5, "gate_start": 0.0, "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256,
                     "stop_after": -1, "dry_inputs": False, "domains": domains_arg(), "extra_lines": K,
                     "prior_domains": ",".join(pri), "prior_shuf": "", "lam_rule": lam_rule, "cyc_held": 0.4, "cyc_heldE": 0.5,
                     "s4_train": 0.6, "dim_train": 0.5, "dom_eval_rows": 200, "dom_seen": 150}}


def dry_block3(arm):
    pri = pri_of(arm)
    specs = " ".join(f"{d}:{str(d in pri).lower()}:false:{json.dumps(1.0 if d in pri else 0.0)}" for d in ("numbers",) + DOMS)
    return ("""! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a numbers prior target (every ptdom3 arm has none)"; exit 3; }
dl=$(grep -m1 '^domains: ' "$OUT/dry_inputs.txt" || true)
has() { grep -qF -e "$1," -e "$1}" <<< "$blk"; }
for sp in """ + specs + """; do
  IFS=: read -r d p s l <<< "$sp"
  blk=$(grep -oE "\\"$d\\": \\{[^}]*\\}" <<< "$dl" || true)
  has "\\"prior\\": $p" && has "\\"shuf\\": $s" && has "\\"lam\\": $l" ||
    { echo "## domain $d not prior $p / shuf $s / lam $l in the domains: line"; exit 3; }
done
grep -qF '"numbers": {"target": "none", ' <<< "$dl" || { echo "## numbers target not none"; exit 3; }
for d in """ + " ".join(f"{d}:{READSET[d]}" for d in DOMS) + """; do
  IFS=: read -r d s <<< "$d"
  grep -qE "^\\[$d\\] \\{.*\\"readout\\": \\"$s\\"" "$OUT/dry_inputs.txt" || { echo "## $d read-out set is not $s"; exit 3; }
done""")


READSET_CHECK = "toy3 read-out set <d>_<set> logged (float) at every eval and = the metrics' read-out set"


def _check_py_3(cp):
    return _edit(cp, '# The double paths of the 2-way and the k-span scoring (Rev E / G): 2-way = the span log-prob vs the direct per-position path on\n',
                 'RS_ = I.get("readset", {})   # ptdom3: each toy3 domain\'s read-out set (held_nonid / held_fresh_nonid / held_no_negtwin_nonid)\n'
                 'if RS_:\n'
                 '    rb_ = [f"{d}_{s_}" for d, s_ in RS_.items() if any(not isinstance(e.get(f"{d}_{s_}"), float) for e in c)]\n'
                 '    mb_ = [d for d, s_ in RS_.items() if (r.get("metrics", {}).get(I["read"][d]) or {}).get("set") != f"{d}_{s_}"]\n'
                 '    chk("' + READSET_CHECK + ' " + json.dumps(RS_), not rb_ and not mb_, [rb_, mb_])\n'
                 '# The double paths of the 2-way and the k-span scoring (Rev E / G): 2-way = the span log-prob vs the direct per-position path on\n')


CHECK_PY_3 = _check_py_3(M6.CHECK_PY_6)


def dom_kind3(d):
    return "toy3" if d in TOY3 else MJ.dom_kind(d)


@contextmanager
def patched(canary=False):
    names = ("R", "WANT_SHA", "tag_of", "job_name", "slurm_name", "args_of", "intent", "want_flags", "dry_block", "arms", "saves",
             "STAGES", "SEL", "CONF", "SLOT0", "SLOTS", "CHECK_PY", "dom_kind", "DOM_KEYS")
    keep = {k: getattr(MJ, k) for k in names}
    kind0 = MJ.dom_kind
    assert KEY not in MJ.BATCHES and KEY not in MJ.HWOF
    try:
        MJ.BATCHES[KEY] = DOMS
        MJ.HWOF[KEY] = "b200"
        MJ.R, MJ.WANT_SHA = R, WANT_SHA
        MJ.tag_of = lambda b_, a_, l_: tag3(a_, l_)
        MJ.job_name = lambda b_, a_, l_, s_, rr=0, pilot=False: name3(a_, l_, s_, rr, canary)
        MJ.slurm_name = lambda b_, a_, l_, s_, rr=0, slot=None, pilot=False: slot_name(slot, canary)
        MJ.args_of = lambda b_, a_, l_, s_, out, save=True, pilot=False: args_of3(a_, l_, s_, out, canary)
        MJ.intent = lambda P_, b_, a_, l_, s_, ref, save=True, pilot=False: intent3(P_, a_, l_, s_, ref, canary)
        MJ.want_flags = want_flags
        MJ.dry_block = lambda b_, a_: dry_block3(a_)
        MJ.arms = lambda b_: ARMS + ((CANARY[0],) if canary else ())
        MJ.saves = lambda stage, pilot=False: False
        MJ.STAGES = STAGES
        MJ.SEL, MJ.CONF = (SEED,), ()
        MJ.SLOT0, MJ.SLOTS = {KEY: 0}, N_SLOTS
        MJ.CHECK_PY = CHECK_PY_3
        MJ.dom_kind = lambda d: "toy3" if d in TOY3 else kind0(d)
        MJ.DOM_KEYS = DOM_KEYS3
        yield KEY
    finally:
        for k, v in keep.items():
            setattr(MJ, k, v)
        MJ.BATCHES.pop(KEY, None)
        MJ.HWOF.pop(KEY, None)


def desc3(arm):
    pri = pri_of(arm)
    if not pri:
        return "none (no prior)"
    if arm == "alltoy3":
        return "CANARY alltoy3: the six toy3 priors " + " + ".join(pri) + " (lambda 1 each, window (0.05, 0.4), ctx)"
    return f"only the {pri[0]} prior (lambda 1, window (0.05, 0.4), ctx)"


RO_FILE = B / "readout_ptdom3.py"
STEP0_MARK = ('grep -q "step 0: toy3 true-prior gradient norms" "/proc/$$/fd/1" 2>/dev/null && echo "## step-0 toy3 true-prior gradient '
              'line present (the trainer asserted every norm > 0)" || echo "## FLAG: no step-0 toy3 true-prior gradient line found in this log"\n')


def render3(P, arm, lr, seed, ref, commit, rr=0, slot=None, canary=False):
    stage = "canary" if canary else "S"
    assert (arm == CANARY[0]) == canary and lr in LADDER and lr in (GRID["none"] if arm == "none" else GRID["prior"])
    with patched(canary) as key:
        desc0 = (f"only the {arm[5:]} prior (lambda 1, window (0.05, 0.4), ctx)" if arm != "none" else "none (no prior)")
        name, tag, text, args, I = MJ.render(P, key, arm, lr, seed, ref, commit, rr, None if canary else slot, stage, canary)
    assert name == name3(arm, lr, seed, rr, canary) and tag == tag3(arm, lr)
    n5 = len(DOMS)
    text = _edit(text, f"# ptdom A3 / A4 batch {key} stage {'pilot' if canary else stage}: {desc0} @ lr",
                 f"# ptdom3 (exploratory screen; ptdom's stage-1 job body), batch {key} {stage}: {desc3(arm)} @ lr")
    text = _edit(text, "; code = runs/ptdom/code (COMMIT", "; code = runs/ptdom3/code (COMMIT")
    text = _edit(text, "# spec and pre-registered screen / verdict runs/ptdom/AUDIT.md (Amendments A3, A4); generated by llmadd_torch/make_jobs_ptdom.py\n",
                 "# spec runs/ptdom3/AUDIT.md (exploratory tier: no decision rule); rendered by llmadd_torch/make_jobs_ptdom3.py through make_jobs_ptdom.render\n")
    if canary:
        text = _edit(text, "(not part of any result; output runs/ptdom/pilot/)", "(ptdom3 pipeline canary, not part of any result; output runs/ptdom3/pilot/)")
    else:
        text = _edit(text, "; 16 chains over the four batches).", f"; {N_SLOTS} chains).")
    text = _edit(text, "# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/ptdom/HOLD (every batch) or\n"
                       f"# runs/ptdom/HOLD_{key} (A4: this batch only) exists\n"
                       f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{key}" ]; then echo "HOLD present: exiting"; exit 0; fi\n',
                 "# kill switch (no scancel from the agent): a pending or requeued job exits at start when runs/ptdom3/HOLD (every job) or\n"
                 f"# runs/ptdom3/HOLD_{stage} (the screen S or the canary only) exists\n"
                 f'if [ -e "$R/HOLD" ] || [ -e "$R/HOLD_{stage}" ]; then echo "HOLD present: exiting"; exit 0; fi\n')
    text = _edit(text, "# 4. CHECK_PY (make_jobs_ptdom.py) with this run's intent",
                 "# 4. CHECK_PY_3 (make_jobs_ptdom3.py: ptdom's CHECK_PY + pt6comb's 7 + ptdom3's 1 counted edits) with this run's intent")
    text = _edit(text, f"the site rule ({n5} domains asserted)", "the site rule (0 lines: toy3 items are single tokens, asserted by the trainer at load)")
    text = _edit(text, f'"$OUT/dry_inputs.txt")" = {n5} ] || {{ echo "## the site rule is not asserted for the {n5} domains with item-name sites"; exit 3; }}',
                 '"$OUT/dry_inputs.txt")" = 0 ] || { echo "## site-rule lines for toy3 domains (none expected)"; exit 3; }')
    post = STEP0_MARK
    if canary:
        post += (f'[ "$(sha256sum {RO_FILE} | cut -c1-12)" = "{sha(RO_FILE)}" ] || {{ echo "## readout_ptdom3.py changed since rendering"; exit 3; }}\n'
                 f'echo "## 2b. canary read-out $(date -u +%FT%TZ)"\n'
                 f'"$PY" {RO_FILE} --canary "$OUT" | tee "$OUT/canary_readout.txt" || echo "## CANARY READ-OUT FAILED"\n')
    text = _edit(text, '"$PY" pretrain_range.py $ARGS\n', '"$PY" pretrain_range.py $ARGS\n' + post)
    for bad in ("runs/ptdom/", "make_jobs_ptdom.py)", "HOLD_T3"):
        assert bad not in text, bad
    return name, tag, text, args, I


# ---------------------------------------------------------------- snapshot, references
def snapshot():
    """runs/ptdom3/code = git archive of tools/ at ef7b8854; asserted: the shas of WANT_SHA (= the commit's files); every other
    file byte-identical to runs/ptdom/code/tools (11f61b9); only toy3.py / toy3_nonce.json added."""
    R.mkdir(parents=True, exist_ok=True)
    if not (R / "code").exists():
        tmp = R / "code.tmp"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        arc = subprocess.run(["git", "-C", str(REPO), "archive", COMMIT, "tools"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
        (tmp / "COMMIT").write_text(f"{COMMIT} (ptdom3 Rev H, committed, not pushed; git archive of tools/); sha256[:12]: "
                                    + ", ".join(f"{f} {sha(tmp / 'tools' / f)}" for f in WANT_SHA) + "\n")
        tmp.rename(R / "code")
    cm = (R / "code" / "COMMIT").read_text()
    assert cm.split()[0] == COMMIT
    for f, v in WANT_SHA.items():
        s_ = sha(R / "code" / "tools" / f)
        assert f"{f} {s_}" in cm and s_ == v, (f, s_, v)
        assert hashlib.sha256(subprocess.run(["git", "-C", str(REPO), "show", f"{COMMIT}:tools/{f}"], capture_output=True,
                                             check=True).stdout).hexdigest()[:12] == s_, (f, COMMIT)
    T_ = R / "code" / "tools"
    mine = sorted(p.relative_to(T_) for p in T_.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    theirs = sorted(p.relative_to(R0 / "code" / "tools") for p in (R0 / "code" / "tools").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert sorted(set(mine) - set(theirs)) == [Path("toy3.py"), Path("toy3_nonce.json")] and not set(theirs) - set(mine), (set(mine) ^ set(theirs))
    for p in theirs:
        if str(p) != "pretrain_range.py":
            assert (T_ / p).read_bytes() == (R0 / "code" / "tools" / p).read_bytes(), p
    return COMMIT


REFS = R / "ptdom3_refs_T3.json"


def refs3():
    """CPU references (runs/ptdom3/tests/refs_ptdom3.py; written once, chmod 444): {"seed": {...}, "canary": {...}}."""
    if not REFS.exists():
        return {}, {}
    j = json.loads(REFS.read_text())
    assert j.get("design") == PREFIX and j.get("domains") == domains_arg(), REFS
    return j["seeds"]["0"], j["canary"]


def load_P():
    snapshot()
    sys.path.insert(0, str(R / "code" / "tools"))
    import make_rb6_tasks as P
    assert Path(P.__file__).resolve().parent == (R / "code" / "tools").resolve()
    return P


def cells():
    """-> [(arm, lr, seed, slot)] of the 22 screen runs (none first: the reference), slots round-robin."""
    c = [("none", lr) for lr in GRID["none"]] + [(f"only-{d}", lr) for d in DOMS for lr in GRID["prior"]]
    assert len(c) == 22
    return [(a, lr, SEED, i % N_SLOTS) for i, (a, lr) in enumerate(c)]


def build(P):
    ref, rc = refs3()
    assert ref and rc, f"no CPU reference {REFS}: run runs/ptdom3/tests/run_refs_ptdom3.sh first"
    commit = snapshot()
    jobs = []
    for arm, lr, seed, slot in cells():
        jobs.append((slot, *render3(P, arm, lr, seed, ref, commit, 0, slot)))
    a, lr, s = CANARY
    jobs.append((None, *render3(P, a, lr, s, rc, commit, 0, None, True)))
    names = [j[1] for j in jobs]
    assert len(set(names)) == len(names) == 23
    others = {p.name for p in B.glob("runs/*/out/*") if p.parent.parent != R} | {p.name for p in B.glob("runs/*/pilot/*") if p.parent.parent != R}
    assert not set(names) & others, sorted(set(names) & others)
    return jobs


def jobfile(name):
    return R / "jobs" / f"{name}.sbatch"


def main():
    a = sys.argv[1:]
    if "--snapshot" in a:
        print("snapshot", snapshot(), R / "code"); return
    P = load_P()
    if "--check" in a or "--print" in a or "--test-only" in a:
        jobs = build(P)
        (R / "jobs").mkdir(exist_ok=True)
        for slot, name, tag, text, args, I in jobs:
            f = jobfile(name)
            if f.exists() and f.read_text() == text:
                continue
            if f.exists():
                f.chmod(0o644)
            f.write_text(text); f.chmod(0o444)
            (R / "jobs" / f"{name}.intent.json").write_text(json.dumps(I, indent=1))
        for slot, name, tag, text, args, I in jobs:
            print(f"{name}\tslot {slot}\t{jobfile(name)}\t{sha(jobfile(name))}")
        if "--print" in a:
            print(jobfile(a[a.index("--print") + 1]).read_text())
        if "--test-only" in a:
            bad = 0
            for slot, name, *_ in jobs:
                f = jobfile(name)
                rn = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
                rt = subprocess.run(["sbatch", "--test-only", str(f)], capture_output=True, text=True)
                ok = rn.returncode == 0 and rt.returncode == 0 and "to start at" in (rt.stderr + rt.stdout)
                bad += not ok
                print(f"{'OK ' if ok else 'BAD'} {name}: bash -n rc {rn.returncode}; test-only: {(rt.stderr + rt.stdout).strip()[:160]}")
            print("TEST-ONLY", "PASS" if not bad else f"FAIL ({bad})", f"({len(jobs)} jobs)")
        return
    if "--submit" in a:
        which = a[a.index("--submit") + 1]
        assert which in ("canary", "screen"), which
        assert not (R / "HOLD").exists() and not (R / f"HOLD_{'canary' if which == 'canary' else 'S'}").exists(), "HOLD present: not submitting"
        jobs = build(P)
        rec = R / "submitted.tsv"
        done = {l.split("\t")[0] for l in rec.read_text().splitlines()} if rec.exists() else set()
        for slot, name, tag, text, args, I in jobs:
            if (name.startswith(f"{PREFIX}-canary-")) != (which == "canary"):
                continue
            f = jobfile(name)
            assert f.exists() and f.read_text() == text, f"{f}: render --check first (the file differs from the generator)"
            if name in done or (R / "out" / name).exists() or (R / "pilot" / name).exists():
                print("SKIP (submitted before / out dir exists):", name); continue
            o = subprocess.run(["sbatch", "--parsable", str(f)], capture_output=True, text=True, check=True).stdout.strip()
            jid = o.split(";")[0]
            with open(rec, "a") as fh:
                fh.write(f"{name}\t{jid}\t{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{sha(f)}\n")
            print(f"SUBMITTED {name} {jid}")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
