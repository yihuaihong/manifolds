"""Torch jobs (2026-10-26): the arithmetic-share dose at 53M (intent spec and pre-registered rules runs/ptshare/AUDIT.md =
llmadd-burst report/audits/ptshare_AUDIT.md). User approval relayed by the coordinator 2026-10-26, queued "after this batch"
(the fixed-tokens-per-parameter batch, nice 20) -> --nice=2000; rb6_190m_dim (nice 5000) stays last.
Question: at one size (53M), one trainer, one GPU type, does the helix3 window prior's gain depend on the arithmetic share?
Only the share changes: the trainer's existing --arith_lines k (k lines per 512-token row; 2% of F82 / F86 = k 1, 7% of F82 /
F87 = k 4: the same mechanism), k = 1, 2, 8 here and k = 4 reused from runs/ptlr (none@8e-3, win-ctx@2e-3, seeds 0-7).
  Trainer = runs/ptlr's: tools/ at commit PIN (git archive -> runs/ptshare/code; every file asserted byte-identical to
  runs/ptlr/code/tools), code_sha a3bce3ab63c1; same base args, account, L40S partition, qos and time limit as ptlr 53M.
  Design constants (shares, lr grids, expansions, tags) = tools/ptshare_design.py of the read-out snapshot runs/ptshare/readout
  (git archive of llmadd-burst branch arith-dose at RO_COMMIT; COMMIT lists the sha256[:12] of every read-out file).
  Job text = make_jobs_ptlr.py's own template (READ-ONLY, sha256[:12] asserted): the f-string of its job.write_text(...) and
  its G / ARM / WANT / flags are taken from its source with ast and evaluated here, not copied; the job then gets exactly the
  edits in EDITS (job name, --nice, comments, HOLD path text, an L40S guard, and the dry-run block: log to a file, 2k operand
  sites in each printed row, the target line, the share replay tools/arith_share.py which re-derives rows 0-2 of the dry log).
Stages
  --stage main            CELLS (k 1 / 2 / 8 x none / win-ctx x 3 lrs x seeds 0-3 = 72 jobs) + the round-1 selection job
                          (cpu_short, --dependency=afterany on the 72)
  --stage auto --picks f  the cells of a picks file written by a selection job (tools/ptshare_pick.py): one-step lr
                          expansions (seeds 0-3, round 1 only) and confirm seeds 4-7; after a round-1 expansion, the round-2
                          selection job (afterany on the expansion jobs)
A submitted job file is never rewritten (SUBMITTED_<name> flags); --submit submits what is not yet submitted.
    python make_jobs_ptshare.py [--stage main | --stage auto --picks <file>] [--print] [--test-only] [--submit]"""
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
REPO = Path("/scratch/yh6210/llmadd-burst")          # git object store (the branch's worktree shares it)
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
R = B / "runs" / "ptshare"
PTLR = B / "runs" / "ptlr"
PIN, PIN_SHA = "64abc19", "a3bce3ab63c1"             # = runs/ptlr/code/COMMIT and its trainer's code_sha
RO_BRANCH, RO_COMMIT = "arith-dose", "1842e3c"       # the read-out snapshot (branch commit of 2026-10-26, before any job)
RO_FILES = ("ptshare_design.py", "arith_share.py", "ptshare_summary.py", "ptshare_rules.py", "ptshare_pick.py", "readout_rules.py",
            "ptshare_refs.json")
PTLR_GEN = B / "make_jobs_ptlr.py"           # read-only; its file sha was 85c3bf083bc9 on 2026-10-26
PTLR_TEMPLATE_SHA = "c9b834f3679a"            # sha256[:12] of the unparsed G, ARM, WANT, flags and job f-string
SEL_ACCOUNT, SEL_PARTITION = "<slurm-account>", "cpu_short"   # = the pt2s505 selection job's
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# ---------------------------------------------------------------- snapshots (created once, verified on every call)
def snapshot():
    snapshot_code()
    return snapshot_readout()


def snapshot_code():
    if not (R / "code").exists():
        tmp = R / "code.tmp"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        arc = subprocess.run(["git", "-C", str(REPO), "archive", PIN, "tools"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
        (tmp / "COMMIT").write_text(PIN + "\n")
        tmp.rename(R / "code")
    assert (R / "code" / "COMMIT").read_text().strip() == PIN and sha(R / "code/tools/pretrain_range.py") == PIN_SHA
    mine = sorted(p.relative_to(R / "code") for p in (R / "code").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    theirs = sorted(p.relative_to(PTLR / "code") for p in (PTLR / "code").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert mine == theirs, "runs/ptshare/code and runs/ptlr/code differ in their file sets"
    for p in mine:
        assert (R / "code" / p).read_bytes() == (PTLR / "code" / p).read_bytes(), f"{p} differs from runs/ptlr/code"


def snapshot_readout():
    if not (R / "readout").exists():
        assert RO_COMMIT != "PENDING", "set RO_COMMIT (the committed read-out) first"
        tmp = R / "readout.tmp"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        arc = subprocess.run(["git", "-C", str(REPO), "archive", RO_COMMIT, "tools"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arc, check=True)
        (tmp / "COMMIT").write_text(f"{RO_COMMIT} {RO_BRANCH}\n" + "".join(f"{f} {sha(tmp / 'tools' / f)}\n" for f in RO_FILES))
        tmp.rename(R / "readout")
    lines = (R / "readout" / "COMMIT").read_text().splitlines()
    assert lines[0] == f"{RO_COMMIT} {RO_BRANCH}", (lines[0], RO_COMMIT)
    for f, l in zip(RO_FILES, lines[1:]):
        assert l == f"{f} {sha(R / 'readout' / 'tools' / f)}", (f, l)
    return lines[0]


def design():
    sys.path.insert(0, str(R / "readout" / "tools"))
    import ptshare_design as D
    assert Path(D.__file__).resolve().parent == (R / "readout" / "tools").resolve(), D.__file__
    return D


# ---------------------------------------------------------------- ptlr's template, taken from its source
def ptlr_parts(gen=None):
    """make_jobs_ptlr.py's G / ARM / WANT / flags() and its job template, pinned by the sha256[:12] of their source
    (ast.unparse), not of the whole file: ptlr's generator is append-only (amendments add CELLS), and an amendment
    between now and a selection job must not stop the confirm stage; a change of the template itself still does"""
    tree = ast.parse(Path(gen or PTLR_GEN).read_text())
    ns, parts = {}, []
    for node in tree.body:   # G, ARM, WANT (literals / f-strings) and flags(): no other top-level code is run
        names = [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
        if (isinstance(node, ast.Assign) and names in (["G"], ["ARM"], ["WANT"])) or (
                isinstance(node, ast.FunctionDef) and node.name == "flags"):
            exec(compile(ast.Module([node], type_ignores=[]), str(PTLR_GEN), "exec"), ns)
            parts.append(ast.unparse(node))
    assert {"G", "ARM", "WANT", "flags"} <= set(ns) and len(parts) == 4, sorted(ns)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "write_text" and isinstance(n.func.value, ast.Name) and n.func.value.id == "job"]
    assert len(calls) == 1 and isinstance(calls[0].args[0], ast.JoinedStr), "make_jobs_ptlr.py: one job.write_text(f-string)"
    parts.append(ast.unparse(calls[0].args[0]))
    got = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]
    assert got == PTLR_TEMPLATE_SHA, (got, PTLR_TEMPLATE_SHA, "make_jobs_ptlr.py's template / ARM / WANT / flags changed")
    tmpl = compile(ast.Expression(calls[0].args[0]), f"{PTLR_GEN}:template", "eval")
    names = {n.id for n in ast.walk(calls[0].args[0]) if isinstance(n, ast.Name)}
    assert names == {"jt", "hours", "B", "R", "PIN_SHA", "args"}, names
    return ns, tmpl


def ptlr_text(tmpl, jt, hours, Rdir, args):
    return eval(tmpl, {"jt": jt, "hours": hours, "B": B, "R": Rdir, "PIN_SHA": PIN_SHA, "args": args})


DRY_LINE = f"{PY} pretrain_range.py $ARGS --dry_inputs\n"
GPU_LINE = "nvidia-smi --query-gpu=name --format=csv,noheader\n"
OLD_COMMENT = ("# lr bracket (best lr per arm), unified 0..999 setup; code = llmadd-burst commit in code/COMMIT; intent spec "
               "runs/ptlr/AUDIT.md\n")


def edits(k, arm, tag, seed, D):
    """[(old, new)]: every old occurs exactly once in ptlr's rendering; the only differences from a ptlr job"""
    out = f"{R}/out/{tag}"
    tgt = ("""! grep -q '^target features' "$DRY" || { echo "## a prior target in a none run: exiting"; exit 3; }"""
           if arm == "none" else
           """grep -qxF 'target features (1000, 10)' "$DRY" || { echo "## not the helix3 table (1000 x 10): exiting"; exit 3; }""")
    dry = (f'OUTD={out}; DRY="$OUTD/dry_inputs_s{seed}.txt"; mkdir -p "$OUTD"\n'
           f"# the share replay must be the read-out snapshot's (runs/ptshare/readout/COMMIT)\n"
           f'grep -qxF "arith_share.py $(sha256sum {R}/readout/tools/arith_share.py | cut -c1-12)" {R}/readout/COMMIT '
           f'|| {{ echo "## arith_share.py differs from the read-out snapshot: exiting"; exit 3; }}\n'
           f'{PY} pretrain_range.py $ARGS --dry_inputs > "$DRY" 2>&1 || {{ cat "$DRY"; exit 3; }}\n'
           f'cat "$DRY"\n'
           f"# {k} arithmetic line(s) per row = {2 * k} operand sites in each printed row; the target table only in win-ctx\n"
           f"""[ "$(grep -c '^train row [012] (len 512, {2 * k} prior sites at ' "$DRY")" = 3 ] """
           f'|| {{ echo "## not {k} arithmetic line(s) per row: exiting"; exit 3; }}\n'
           f"{tgt}\n"
           f"# the actual arithmetic share of this run's rows (and its replay = the dry rows 0-2)\n"
           f'{PY} {R}/readout/tools/arith_share.py --code {R}/code/tools --dry "$DRY" $ARGS '
           f'|| {{ echo "## the share replay disagrees with the dry rows: exiting"; exit 3; }}\n')
    gpu = ('GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1); echo "$GPU"\n'
           'case "$GPU" in *L40S*) ;; *) echo "## not an L40S (the GPU of the reused 7% cells): exiting"; exit 3;; esac\n')
    return [("#SBATCH --job-name=llmadd_ptlr-", "#SBATCH --job-name=llmadd_ptshare-"),
            ("#SBATCH --requeue\n", f"#SBATCH --requeue\n#SBATCH --nice={D.NICE}\n"),
            (OLD_COMMENT, f"# arithmetic-share dose, 53M, share {D.LABEL[k]} = --arith_lines {k}; trainer = the ptlr snapshot's (code/COMMIT); "
                          f"intent spec runs/ptshare/AUDIT.md; generated by llmadd_torch/make_jobs_ptshare.py (template = "
                          f"make_jobs_ptlr.py's)\n"),
            ("when runs/ptlr/HOLD exists", "when runs/ptshare/HOLD exists"),
            (GPU_LINE, gpu),
            (DRY_LINE, dry)]


def render(k, arm, lr, seed, ns, tmpl, D):
    assert k in D.NEW and arm in D.ARMS and lr in D.grid(k, arm, expanded=True) and seed in D.MAIN + D.CONFIRM, (k, arm, lr, seed)
    tag = D.tag(k, arm, lr)
    jt = f"{tag}-s{seed}"
    extra = " ".join(x for x in (ns["ARM"][arm], f"--arith_lines {k}", "", f"--lr {lr}") if x)   # = ptlr's assembly (53m: no model flags)
    args = (f"--data {B}/pretrain_data --tokenizer {TOK} --tokenization whole --out {R}/out/{tag} {extra} --seed {seed} "
            f"--tag _{tag} --save_model --ckpt_every 2000")
    f = ns["flags"](args)
    want = {"--lr": lr, "--arith_lines": str(k), "--seed": str(seed), "--tag": f"_{tag}", "--out": f"{R}/out/{tag}",
            "--save_model": True, "--ckpt_every": "2000", "--tokenization": "whole", "--data": f"{B}/pretrain_data",
            "--tokenizer": str(TOK), "--lam": None, "--geo_layers": None, "--bs": None, "--tokens": None, "--seq": None,
            "--gate_start": None, "--gate_hyst": None, "--gate_thr": None, "--untie": None, "--d_model": None,
            "--n_layers": None, "--n_heads": None, "--d_ff": None, "--eval_every": None, "--n_eval": None,
            "--grad_ckpt": None, "--lm_chunk_tokens": None, "--stop_after": None, "--dry_inputs": None, **ns["WANT"][arm]}
    for kk, v in want.items():
        assert f.get(kk) == v, (jt, kk, f.get(kk), v)
    assert set(f) == {kk for kk, v in want.items() if v is not None}, (jt, sorted(f))
    if arm == "none":
        assert not ({"--schedule", "--window", "--site", "--gate_thr"} & set(f)), (jt, sorted(f))
    text = ptlr_text(tmpl, jt, D.HOURS, R, args)
    for old, new in edits(k, arm, tag, seed, D):
        assert text.count(old) == 1, (jt, old)
        text = text.replace(old, new)
    assert "runs/ptlr" not in text and text.count(f"--arith_lines {k} ") == 1, jt
    return jt, tag, text, args


def select_text(n):
    return f"""#!/bin/bash
#SBATCH --job-name=llmadd_ptshare-select-r{n}
#SBATCH --account={SEL_ACCOUNT}
#SBATCH --partition={SEL_PARTITION}
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output={B}/logs/%x_%j.out
# ptshare selection round {n} (runs/ptshare/AUDIT.md "Selection, expansion, confirm seeds"): rule 1 per (share, arm) by
# tools/ptshare_summary.py, then tools/ptshare_pick.py -> picks_r{n}.json -> make_jobs_ptshare.py --stage auto (--test-only,
# then --submit only if every test passed). Submitted with --dependency=afterany:<this round's training jobs>.
set -u
R={R}; PY={PY}; LOG=$R/select.log
if [ -e "$R/HOLD" ]; then echo "HOLD present: exiting" | tee -a "$LOG"; exit 0; fi
echo "== selection round {n} $(date -u +%FT%TZ) job ${{SLURM_JOB_ID:-}} on $(hostname)" | tee -a "$LOG"
rm -f "$R/selection_r{n}.json" "$R/picks_r{n}.json"   # never act on a stale file
cd "$R/readout/tools"
for f in {' '.join(RO_FILES)}; do
  grep -qxF "$f $(sha256sum $f | cut -c1-12)" "$R/readout/COMMIT" || {{ echo "## $f differs from the read-out snapshot: exiting" | tee -a "$LOG"; exit 3; }}
done
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
"$PY" ptshare_summary.py --root {B}/runs --json "$R/selection_r{n}.json" > "$R/summary_r{n}.txt" 2>&1; rc=$?
cat "$R/summary_r{n}.txt" | tee -a "$LOG"
[ $rc = 0 ] || {{ echo "## the read-out failed (rc $rc): nothing submitted" | tee -a "$LOG"; exit 3; }}
"$PY" ptshare_pick.py --round {n} "$R/selection_r{n}.json" "$R/picks_r{n}.json" 2>&1 | tee -a "$LOG"
if [ ! -s "$R/picks_r{n}.json" ]; then echo "round {n}: nothing to submit (the reasons above go to the user)" | tee -a "$LOG"; exit 0; fi
for v in $(env | grep -oE '^(SLURM|SBATCH)_[A-Za-z0-9_]*'); do unset "$v"; done
cd {B}
"$PY" make_jobs_ptshare.py --stage auto --picks "$R/picks_r{n}.json" --test-only > "$R/test_r{n}.txt" 2>&1; rc=$?
cat "$R/test_r{n}.txt" | tee -a "$LOG"
[ $rc = 0 ] || {{ echo "## --test-only failed (rc $rc): nothing submitted" | tee -a "$LOG"; exit 3; }}
"$PY" make_jobs_ptshare.py --stage auto --picks "$R/picks_r{n}.json" --submit 2>&1 | tee -a "$LOG"
echo "== done $(date -u +%FT%TZ)" | tee -a "$LOG"
"""


# ---------------------------------------------------------------- cells (append-only)
def main_cells(D):
    return [(k, arm, lr, s) for k in D.NEW for arm in D.ARMS for lr in D.GRID[arm] for s in D.MAIN]


CELLS_NOTE = "main stage = main_cells(D): k 1 / 2 / 8 x none / win-ctx x ptshare_design.GRID x seeds 0-3 (72 jobs)"
# later cells come only from picks files written by the selection jobs (runs/ptshare/picks_r<n>.json), listed in the log;
# anything else (e.g. an lr the user approves after a STOP) is an amendment: AUDIT.md first, then the design file of a new
# read-out commit (ptshare_design.EXPAND / GRID are append-only), then a stage added below; existing cells never change


def picks_cells(path, D):
    pk = json.loads(Path(path).read_text())
    assert Path(path).resolve().parent == R.resolve() and re.fullmatch(r"picks_r[12]\.json", Path(path).name), path
    n = pk["round"]
    assert n in (1, 2) and Path(path).name == f"picks_r{n}.json", (n, path)
    out = []
    for c in pk["cells"]:
        k, arm, lr, seeds, why = c["k"], c["arm"], c["lr"], tuple(c["seeds"]), c["why"]
        assert k in D.NEW and arm in D.ARMS and why in ("expand", "confirm"), c
        if why == "expand":
            assert n == 1 and seeds == D.MAIN and lr in D.EXPAND[arm].values(), c
        else:
            assert seeds == D.CONFIRM and lr in D.grid(k, arm, expanded=True), c
        out += [(k, arm, lr, s, why) for s in seeds]
    return n, out


def sbatch(args):
    o = subprocess.run(["sbatch"] + args, capture_output=True, text=True)
    return o.returncode, (o.stdout + o.stderr).strip()


def main():
    head = snapshot()
    D = design()
    ns, tmpl = ptlr_parts()
    assert ns["ARM"]["win-ctx"] == "--target helix3 --schedule window --window 0.05,0.4 --site ctx" and ns["ARM"]["none"] == "--target none"
    stage = opt("--stage", "main")
    if stage == "main":
        rnd, cells = 0, [c + ("main",) for c in main_cells(D)]
        assert len(cells) == 72, len(cells)
    else:
        assert stage == "auto" and opt("--picks"), "--stage auto needs --picks <runs/ptshare/picks_r<n>.json>"
        rnd, cells = picks_cells(opt("--picks"), D)
    other = {p.name for p in B.glob("runs/*/out/*") if p.parent.parent != R}
    R.mkdir(parents=True, exist_ok=True)
    jobs, seen = [], set()
    for k, arm, lr, s, why in cells:
        jt, tag, text, args = render(k, arm, lr, s, ns, tmpl, D)
        assert jt not in seen and tag not in other, (jt, tag)
        seen.add(jt)
        job, flag = R / f"job_{jt}.sbatch", R / f"SUBMITTED_{jt}"
        if flag.exists():   # a submitted job file is never rewritten
            if job.read_text() != text:
                print(f"WARNING: {job.name} was submitted and differs from this rendering: kept, not rewritten")
        else:
            job.write_text(text)
        jobs.append((jt, job, flag, args, why))
    sel_n = 1 if stage == "main" else (2 if rnd == 1 and any(w == "expand" for *_, w in jobs) else None)
    if sel_n:
        sj = R / f"select_r{sel_n}.sbatch"
        st = select_text(sel_n)
        if (R / f"SUBMITTED_select_r{sel_n}").exists():
            if sj.read_text() != st:
                print(f"WARNING: {sj.name} was submitted and differs from this rendering: kept, not rewritten")
        else:
            sj.write_text(st)
    # cost cap (pre-submission amendment 2026-10-26: about 95 L40S-h planned, at most ~118 with every expansion; never more)
    done = {p.name[len("SUBMITTED_"):] for p in R.glob("SUBMITTED_53m-*")}
    total = len(done | {jt for jt, *_ in jobs})
    print(f"training jobs of the batch after this stage: {total} of at most {D.MAX_JOBS} (~{0.98 * total:.0f} L40S-h)")
    if total > D.MAX_JOBS:
        print(f"STOP: {total} training jobs would exceed the batch cap {D.MAX_JOBS} (~118 L40S-h): nothing is tested or submitted")
        sys.exit(1)
    for k in sorted({c[0] for c in cells}):
        print(f"share {D.share_label(k)} (tag name {D.LABEL[k]}): --arith_lines {k} ({2 * k} operand sites per row); measured share "
              + (", ".join(f"s{s} {100 * v['share']:.2f}%" for s, v in sorted(D.refs()['share'][k].items())) if D.refs() else "(refs missing)"))
    if "--print" in sys.argv:
        for jt, _, _, args, why in jobs:
            print(f"{jt} [{why}]: {args}")
    bad = 0
    if "--test-only" in sys.argv:
        for jt, job, _, _, _ in jobs:
            rc, msg = sbatch(["--test-only", str(job)])
            bad += rc != 0
            print(jt, msg)
        if sel_n:
            rc, msg = sbatch(["--test-only", str(R / f"select_r{sel_n}.sbatch")])
            bad += rc != 0
            print(f"select_r{sel_n}", msg)
    if "--submit" in sys.argv:
        assert "--test-only" not in sys.argv
        ids = []
        for jt, job, flag, _, why in jobs:
            if not flag.exists():
                rc, msg = sbatch([str(job)])
                print(jt, msg)
                if rc == 0:
                    flag.write_text(msg + "\n")
                else:
                    bad += 1
            m = re.search(r"Submitted batch job (\d+)", flag.read_text()) if flag.exists() else None
            if m and (stage == "main" or why == "expand"):
                ids.append(m.group(1))
        sflag = R / f"SUBMITTED_select_r{sel_n}" if sel_n else None
        want = len(jobs) if stage == "main" else sum(w == "expand" for *_, w in jobs)
        if sel_n and not sflag.exists():
            if len(ids) != want:
                print(f"NOT submitting select_r{sel_n}: {len(ids)} of {want} training jobs have a job id")
                bad += 1
            else:
                rc, msg = sbatch([f"--dependency=afterany:{':'.join(ids)}", str(R / f"select_r{sel_n}.sbatch")])
                print(f"select_r{sel_n}", msg)
                if rc == 0:
                    sflag.write_text(msg + "\n")
                else:
                    bad += 1
    print(f"ptshare {stage}{f' (round {rnd})' if rnd else ''}: {len(jobs)} jobs rendered in {R}; read-out {head}, trainer "
          f"{PIN_SHA}; submitted: {'yes' if '--submit' in sys.argv else 'no'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
