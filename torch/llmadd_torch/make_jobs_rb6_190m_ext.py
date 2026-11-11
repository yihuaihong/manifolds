"""Torch jobs (2026-10-22): rb6_190m_ext, the 190M follow-up of the six-prior rainbow batch runs/rb6_190m (F88 190M
addendum; user approval "我觉得可以跑一下"). A COPY of make_jobs_rb6scale.py (that file, rb6scale_refs.json and the 505M
files stay READ-ONLY: the pending 505M selection jobs call them); only the batch table, the arm table, the snapshot rule,
the CPU-reference gate and the names differ. The design is rb6_190m's (tools/pretrain_range.py --domains, the 53M rb6
main design A5: 2% numbers = one arithmetic line per row first after the text, K = 2 other-domain lines per row cycling
months / weekdays / clock / s4 / dim, window (0.05, 0.4), ctx site, lam_rule each, lambda 1 per active prior, 3e8 tokens =
18310 steps of 32 x 512; 190M = d 768 x 12 layers, 12 heads, d_ff 3072, gradient checkpointing on) except the listed change:
  all6-lam1@1.2e-3, only-numbers@1.2e-3     Q1: one more lr for the two prior arms (the main batch ran them at 6e-4)
  only-clock@6e-4                           Q2: the clock prior alone (= the 53M only-clock arm: --target none, prior clock)
  all6-minus-clock@6e-4                     Q2: numbers (helix3), months, weekdays, s4, dim priors on, the clock prior OFF
                                            (--prior_domains numbers,months,weekdays,s4,dim; the clock LINES stay in the
                                            data: --domains is the six); lambda 1 per active prior (lam_rule each)
  none@9.6e-3, none@1.92e-2                 Q3: none at higher lrs (S4 / dim at 190M)
seeds 0-3 each, 24 jobs (intent spec and PRE-REGISTERED read-out: runs/rb6_190m_ext/AUDIT.md).
Every arm's trainer flags = make_rb6m_tasks.arm_flags(<arm>) of the snapshot + the model flags + --lr (all6-minus-clock
is added to the snapshot module's SPEC table IN MEMORY, the tuple form of its other arms; the files are not changed), and
the literal flag strings are asserted (ARM_FLAGS, a second path); the intent = make_rb6m_tasks.intent of the arm with the
size facts (rb6_190m's); CHECK_PY = make_rb6m_tasks.CHECK_PY verbatim (it is written for any set of active priors).
Code: runs/rb6_190m_ext/code = git archive of tools/ at c6dcb6e = the commit of runs/rb6_190m/code (asserted: COMMIT equal
and every file byte-identical to runs/rb6_190m/code), so the jobs run exactly the main batch's files. The generator refuses
to render without the CPU references of this batch (scale_work/rb190ext_refs.json: --dry_inputs of every (arm, lr, seed)
= the main batch's DRY rows of the seed, the per-domain prior / shuf / lam = the intent; step-0 lm of every arm type =
none@6e-4's on the same host bitwise, that within 1e-4 of the main batch's LM0). Every sbatch as rb6_190m's: HOLD kill switch, the shas of the snapshot (exit 3), an A100 guard,
the data sha256, --dry_inputs with the Burst task's input checks + the DRY row hash (exit 3 before training), training
(requeue resumes from the checkpoint), offline_lm.py, CHECK_PY -> check.json (exit 4 on a failed check), gpu.csv. Account
torch_pr_520_tandon_priority, a100_tandon, gpu48, nice 20.
    python make_jobs_rb6_190m_ext.py --batch rb6_190m_ext [--print] [--test-only] [--submit] [--no_refs (render only)]"""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch"); REPO = Path("/scratch/yh6210/llmadd-burst-scale"); BRANCH = "scale-ladder"
BURST = Path("/scratch/yh6210/llmadd-burst-ro")   # origin/main of yihuaihong/manifolds (the Burst results, read only)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
TOK = B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
DATA = B / "pretrain_data"
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
# the trainer, what it imports, offline_lm and the generator modules whose CHECK_PY / intent the jobs embed; the values
# are the files the CPU regression tested (runs/rb6_190m/AUDIT.md "Code"): origin/main 4908225
WANT_SHA = {"pretrain_range.py": "93e22bf4c4c0", "toy_struct.py": "d182ce548bcc", "pretrain_helix.py": "25d5a35e6f17",
            "numlib.py": "d09bb47b04c2", "digit_split_data.py": "742b922c8125", "run_guard.py": "b944dc859c7a",
            "offline_lm.py": "226f4cac70ea"}
GEN = ("make_rb6m_tasks.py", "make_rb6_tasks.py")
# the read-out / selection tools the batch reads with (in the snapshot from the start, so the automatic 505M main stage
# selects with the committed, tested code): tools/rb6scale_summary.py, tools/rb6scale_select.py
EXTRA_FILES = ("rainbow6_summary.py", "rb6scale_summary.py", "rb6scale_select.py", "readout_rules.py")
DATA_SHA = ("4de2ed19093c37b302b6a1ced56136e6a8b805e6d6b48057a77316811704ff11",   # make_rb6_tasks.DATA (train.bin, val.bin)
            "d6f744a99b988ed6928fb588356cc8fe3d61bcf348b8d06fc0eece25c624a4c8")
ACCOUNT, PARTITION, QOS, NICE = "torch_pr_520_tandon_priority", "a100_tandon", "gpu48", 20
SIZES = {
    # model flags, n_params (190M = pt2s190's 190297344; 505M = ptlr 500m's 505437184; the domains add no parameter: the
    # nonce names are existing tokens), the dry run's model line, prior / read-out layers (n_layers // 3, 2 n_layers // 3),
    # --time = 2 x (measured A100 3e8-token time x 1.2 for the rainbow), rounded up
    "190m": dict(dims="--d_model 768 --n_layers 12 --n_heads 12 --d_ff 3072", d=(768, 12, 12, 3072), n_params=190297344,
                 model_line="model: 190.3M parameters; gradient checkpointing on", layers=[4, 8], time="05:30:00",
                 plan_h=2.1 * 1.2),
    "505m": dict(dims="--d_model 1024 --n_layers 24 --n_heads 16 --d_ff 4096", d=(1024, 24, 16, 4096), n_params=505437184,
                 model_line="model: 505.4M parameters; gradient checkpointing on", layers=[8, 16], time="13:00:00",
                 plan_h=6.3),
}
# Step-0 LM loss references per (size, seed): the trainer's own main() up to the step-0 training forward on CPU (bf16
# autocast, under no_grad; rb6m/a83/step0_ng.py on the snapshot trainer), arm-independent (checked: AUDIT.md); filled by
# the session scratchpad's scale/refs.py. data_sha after step 0 and the domain splits are size-independent (asserted =
# make_rb6m_tasks.DSHA0 / DOMS by refs.py); the DRY row hash differs from 53M's only by the model line (asserted below).
LM0 = {"190m": {}, "505m": {}}
DRY = {"190m": {}, "505m": {}}
REFS = Path(__file__).with_name("rb6scale_refs.json")   # written by scale/refs.py table (CPU), read here
if REFS.exists():
    _r = json.loads(REFS.read_text())
    for _sz in _r:
        LM0[_sz] = {int(k): v for k, v in _r[_sz]["lm0"].items()}
        DRY[_sz] = {int(k): tuple(v) for k, v in _r[_sz]["dry"].items()}
BATCHES = {
    "rb6_190m_ext": dict(size="190m", stages={"main": [("only-clock", "6e-4"), ("all6-minus-clock", "6e-4"),
                                                       ("all6-lam1", "1.2e-3"), ("only-numbers", "1.2e-3"),
                                                       ("none", "9.6e-3"), ("none", "1.92e-2")]}, seeds=(0, 1, 2, 3)),
}
# the main batch whose code snapshot this batch runs (byte-identical, asserted) and its commit on branch scale-ladder
SRC_BATCH, SNAP_COMMIT = B / "runs" / "rb6_190m", "c6dcb6e"
# the arm added to the snapshot module's SPEC table in memory (the tuple form of make_rb6m_tasks.SPEC: numbers --target,
# active priors, shuffled priors, --lam, the 53M lr placeholder that intent() overrides with the cell's lr)
NEW_ARMS = {"all6-minus-clock": ("helix3", ("numbers", "months", "weekdays", "s4", "dim"), (), "1", "1e-3")}
NEW_DESC = {"all6-minus-clock": "all6-lam1 without the clock prior: helix3 on the operands + the months / weekdays / s4 / dim "
                                "priors on their item tokens (ctx), window (0.05, 0.4), lambda 1 per active prior (lam_rule "
                                "each); the clock prior off, the clock lines still in the data"}
# second path: the literal trainer flags of every arm of this batch (the rb6_190m AUDIT table's strings for the first three)
_W = "--schedule window --window 0.05,0.4 --site ctx --lam_rule each"
ARM_FLAGS = {"none": "--target none",
             "all6-lam1": f"--target helix3 {_W} --prior_domains numbers,months,weekdays,clock,s4,dim --lam 1",
             "only-numbers": f"--target helix3 {_W} --prior_domains numbers --lam 1",
             "only-clock": f"--target none {_W} --prior_domains clock --lam 1",
             "all6-minus-clock": f"--target helix3 {_W} --prior_domains numbers,months,weekdays,s4,dim --lam 1"}
# the intended active priors per arm (second path for intent(): lam_domains / active / CHECK_PY's per-domain checks)
ACTIVE = {"none": (), "all6-lam1": ("numbers", "months", "weekdays", "clock", "s4", "dim"), "only-numbers": ("numbers",),
          "only-clock": ("clock",), "all6-minus-clock": ("numbers", "months", "weekdays", "s4", "dim")}
EXT_REFS = B / "scale_work" / "rb190ext_refs.json"   # the CPU references of this batch (scratchpad rb190ext/refs_ext.py)
BURST_DRY = {0: "401-rb6pb-53m-2pct-k2-none-s0", 1: "404-rb6pb-53m-2pct-k2-none-s1", 2: "430-rb6pb-53m-2pct-k2-none-s2",
             3: "431-rb6pb-53m-2pct-k2-none-s3"}   # a Burst 53M dry log per seed (DRY arm-independent: A5 / A6 / A7)


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def tree(root):
    """{relative path: full sha256} of every file under root (__pycache__ excluded: written by the jobs' imports)."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def snapshot(R):
    """runs/rb6_190m_ext/code = git archive of tools/ at SNAP_COMMIT (= the commit runs/rb6_190m/code/COMMIT names) with
    runs/rb6_190m's COMMIT file verbatim (once; temp dir + rename). Afterwards, at every call: the COMMIT files are equal,
    every file of the snapshot = runs/rb6_190m/code's (same file set, full sha256; the jobs run exactly the main batch's
    code), every listed file = the same file in that commit (git show <commit>:tools/<f>) = COMMIT's sha, and the trainer
    files = WANT_SHA (the regression-tested ones). -> the snapshot's commit."""
    R.mkdir(parents=True, exist_ok=True)
    src_cm = (SRC_BATCH / "code" / "COMMIT").read_text()
    assert src_cm.split()[0] == SNAP_COMMIT, ("runs/rb6_190m/code/COMMIT does not name", SNAP_COMMIT, src_cm[:40])
    if not (R / "code").exists():   # once; temp dir + rename (a failed extraction leaves no snapshot)
        tmp = R / "code.tmp"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=git("archive", SNAP_COMMIT, "tools"), check=True)
        (tmp / "COMMIT").write_text(src_cm)
        tmp.rename(R / "code")
    cm = (R / "code" / "COMMIT").read_text()
    assert cm == src_cm, "COMMIT differs from runs/rb6_190m/code/COMMIT"
    mine, theirs = tree(R / "code"), tree(SRC_BATCH / "code")
    assert mine == theirs, ("the snapshot differs from runs/rb6_190m/code",
                            sorted(set(mine) ^ set(theirs))[:5], [f for f in mine if f in theirs and mine[f] != theirs[f]][:5])
    commit = cm.split()[0]
    for f in (*WANT_SHA, *GEN, *EXTRA_FILES):   # the snapshot = what COMMIT names = the committed file of that commit
        s_ = sha(R / "code" / "tools" / f)
        assert f"{f} {s_}" in cm, f
        assert hashlib.sha256(git("show", f"{commit}:tools/{f}")).hexdigest()[:12] == s_, (f, commit)
    for f, v in WANT_SHA.items():
        assert sha(R / "code" / "tools" / f) == v, (f, sha(R / "code" / "tools" / f), v)
    return commit


def register(G):
    """Add NEW_ARMS to the snapshot module's SPEC / DESC tables in memory (the files are not changed: snapshot() asserts
    them byte-identical to runs/rb6_190m/code); assert the existing arms this batch uses are the 53M tables' ones."""
    for a_, spec in NEW_ARMS.items():
        assert a_ not in G.SPEC or G.SPEC[a_] == spec, (a_, G.SPEC.get(a_))
        G.SPEC[a_], G.DESC[a_] = spec, NEW_DESC[a_]
    assert G.SPEC["only-clock"] == ("none", ("clock",), (), "1", "1e-3"), G.SPEC["only-clock"]   # the 53M only-clock arm
    assert G.SPEC["only-numbers"] == ("helix3", ("numbers",), (), "1", "1e-3") and G.SPEC["none"][:4] == ("none", (), (), None)
    assert G.SPEC["all6-lam1"] == ("helix3", G.DOMS6, (), "1", "1e-3"), G.SPEC["all6-lam1"]
    for a_ in ARM_FLAGS:
        assert G.arm_flags(a_) == ARM_FLAGS[a_], (a_, G.arm_flags(a_), ARM_FLAGS[a_])
    return G


def tag_of(size, arm, lr):
    return f"rb6tb-{size}-2pct-k2-{arm}-lr{lr}"


def args_of(G, size, arm, lr, seed, out):
    """The trainer ARGS (out = the task dir; the trainer writes to out/<tag>, as the Burst tasks' $OUT/$TAG)."""
    tag = tag_of(size, arm, lr)
    return (f"--data {DATA} --tokenizer {TOK} --tokenization whole --out {out}/{tag} {G.arm_flags(arm)} --arith_lines 1 "
            f"--domains {G.ALL6} --extra_lines {G.K} {SIZES[size]['dims']} --lr {lr} --seed {seed} --tag _{tag} "
            "--save_model --ckpt_every 2000")


def intent(G, size, arm, lr, seed):
    """CHECK_PY's intent: the 53M main-design intent of the arm (make_rb6m_tasks.intent) with the size facts."""
    S = SIZES[size]
    I = copy.deepcopy(G.intent(arm, seed))
    assert I["args"]["lr"] == 1e-3 and I["args"]["d_model"] == 384 and I["n_params"] == 52672128   # the 53M base
    I.update(code_sha=WANT_SHA["pretrain_range.py"], toy_sha=WANT_SHA["toy_struct.py"], n_params=S["n_params"],
             grad_ckpt_on=True, geo_layers=S["layers"], read_layers=S["layers"], lm0_ref=LM0[size].get(seed))
    d_model, n_layers, n_heads, d_ff = S["d"]
    I["args"].update(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, lr=float(lr),
                     tag=f"_{tag_of(size, arm, lr)}")
    return I


def burst_dry_expected(G, P, size, seed):
    """The DRY row hash expected at this size from the Burst 53M dry log of the seed: the same DRY_GREP lines with the
    model line replaced by this size's (every other selected line is model-independent: rows, eval rows, splits)."""
    txt = (BURST / "results" / BURST_DRY[seed] / "dry_inputs.txt").read_text().splitlines()
    import re
    g = re.compile(P.DRY_GREP)
    sel = [l for l in txt if g.match(l)]
    assert sum(l == P.MODEL_LINE for l in sel) == 1 and sum(l.startswith("model:") for l in sel) == 1, BURST_DRY[seed]
    h53 = hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest()
    assert (h53, len(sel)) == G.DRY[seed], (BURST_DRY[seed], "this emulation of the task's grep | sha256sum != the recorded DRY")
    sel = [SIZES[size]["model_line"] if l == P.MODEL_LINE else l for l in sel]
    return hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest(), len(sel)


def render(G, P, R, size, arm, lr, seed, pin_sha):
    S = SIZES[size]
    tag = tag_of(size, arm, lr)
    name = f"{tag}-s{seed}"
    OUT = R / "out" / name
    args = args_of(G, size, arm, lr, seed, OUT)
    f = P.flags(args)
    d_model, n_layers, n_heads, d_ff = S["d"]
    want = {"--data": str(DATA), "--tokenizer": str(TOK), "--tokenization": "whole", "--out": f"{OUT}/{tag}",
            "--arith_lines": "1", "--domains": G.ALL6, "--extra_lines": str(G.K), "--lr": lr, "--seed": str(seed),
            "--tag": f"_{tag}", "--save_model": True, "--ckpt_every": "2000", "--d_model": str(d_model),
            "--n_layers": str(n_layers), "--n_heads": str(n_heads), "--d_ff": str(d_ff), **G.want_flags(arm)}
    for k, v in want.items():
        assert f.get(k) == v, (name, k, f.get(k), v)
    assert set(f) == {k for k, v in want.items() if v is not None}, (name, sorted(f))   # every other flag = its default
    assert arm in ARM_FLAGS and float(lr) > 0
    assert G.arm_flags(arm) == ARM_FLAGS[arm] and f" {ARM_FLAGS[arm]} " in f" {args} ", (name, G.arm_flags(arm))
    I = intent(G, size, arm, lr, seed)
    # the intent's prior table (CHECK_PY's per-domain checks) against the second path ACTIVE
    act = ACTIVE[arm]
    assert tuple(I["active"]) == act and I["shuf"] == [], (name, I["active"], I["shuf"])
    assert I["lam_domains"] == {d: (1.0 if d in act else 0.0) for d in G.DOMS6}, (name, I["lam_domains"])
    assert I["args"]["prior_domains"] == ",".join(act) and I["args"]["lr"] == float(lr), (name, I["args"]["prior_domains"])
    assert I["args"]["target"] == ("helix3" if "numbers" in act else "none"), (name, I["args"]["target"])
    assert (I["args"]["lam"] == 1.0 and I["lam_rule"] == "each" and I["args"]["schedule"] == "window") if act else \
        (I["lam_rule"] == "norm" and I["args"]["schedule"] == "whole" and I["events"] == []), (name, I["lam_rule"])
    if arm == "all6-minus-clock":
        assert "clock" not in I["active"] and I["lam_domains"]["clock"] == 0.0 and len(I["active"]) == 5
    if arm == "only-clock":
        assert I["active"] == ["clock"] and I["lam_domains"]["clock"] == 1.0 and I["lam_domains"]["numbers"] == 0.0
    # the batch's own CPU references (refs_ext.py): this (arm, lr, seed)'s --dry_inputs rows = the main batch's DRY of the
    # seed, its 'domains:' line = the intent; the step-0 lm of the arm type = a main-batch arm type's of the seed computed
    # on the same host (bitwise; CPU bf16 kernels differ across hosts at 1e-5 - 1e-4), every value within 1e-4 of the
    # main batch's LM0 (the jobs' check: 1e-3 vs the GPU run)
    if "--no_refs" not in sys.argv:
        X = json.loads(EXT_REFS.read_text())
        assert X.get("all_hold") is True, "rb190ext_refs.json: not every identity held"
        dr = X["dry"].get(f"{arm}@{lr}@s{seed}")
        assert dr and dr["good"] and dr["rc"] == 0 and (dr["sha"], dr["n"]) == tuple(DRY[size][seed]) and dr["prior_ok"] \
            and dr["model_line"] == [S["model_line"]] and dr["target_ok"] and dr["rows_equal_main"], (name, "CPU dry ref", dr)
        st = X["step0_arm"].get(arm)
        assert st and st["good"] and st["seeds"] and st["seeds_matched_to_a_main_arm_same_host"] == st["seeds"] and \
            st["max_abs_minus_lm0"] < 1e-4, (name, "no / failed CPU step-0 reference of the arm type", st)
    assert I["lm0_ref"] is not None, f"{name}: no step-0 lm reference for ({size}, seed {seed}); run scale/refs.py first"
    assert I["data_sha0"] == G.DSHA0[seed] and I["dom"] == G.DOMS[seed]
    assert DRY[size].get(seed) is not None, f"{name}: no DRY row hash for ({size}, seed {seed}); run scale/refs.py first"
    dry, n_dry = DRY[size][seed]
    assert (dry, n_dry) == burst_dry_expected(G, P, size, seed), (name, "Torch CPU dry rows != the Burst 53M rows + model line")
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    target_assert, lam_loop, specs = G.dry_block(arm)
    tgt = G.SPEC[arm][0]
    runfile = f"run_whole_{tgt}_seed{seed}_{tag}.json"
    h = P.hours(S["time"])
    assert 2 * S["plan_h"] <= h < 2 * S["plan_h"] + 1, (S["time"], S["plan_h"])
    dep_chk = "\n".join(f'[ "$(sha256sum {fn} | cut -c1-12)" = "{v}" ] || {{ echo "## {fn} sha mismatch: exiting"; exit 3; }}'
                        for fn, v in WANT_SHA.items())
    tl = (f"""#!/bin/bash
#SBATCH --job-name=llmadd_{R.name}-{arm}-lr{lr}-s{seed}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --qos={QOS}
#SBATCH --gres=gpu:1
#SBATCH --nice={NICE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={S['time']}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# {R.name}: {arm} @ lr {lr}, {size}, seed {seed}; the 53M rb6 main design's arm ({G.DESC[arm][:80]}...) at {size};
# code = runs/{R.name}/code (COMMIT); intent spec and pre-registered read-out runs/{R.name}/AUDIT.md; generated by
# llmadd_torch/make_jobs_rb6_190m_ext.py (change the generator, not this file); the rb6_190m design and code (c6dcb6e)
# except this cell's arm / lr (AUDIT.md: Q1 lr 1.2e-3 for the prior arms, Q2 only-clock / all6-minus-clock, Q3 none high lr).
# 1. HOLD, snapshot shas, A100, data sha256; --dry_inputs: model line, prior target, per domain prior / shuf / lam, numbers
#    first, content scan, DRY rows = the Torch CPU dry run = the Burst 53M seed-{seed} rows with this size's model line (exit 3)
# 2. training (--save_model --ckpt_every 2000; a requeued job resumes from the checkpoint); 3. offline_lm.py (full val);
# 4. make_rb6m_tasks.CHECK_PY with this cell's intent -> $OUT/check.json (exit 4 on a fail); gpu.csv every 60 s
set -euo pipefail
R={R}; OUT={OUT}; TAG={tag}; SEED={seed}; DATA={DATA}; TOK={TOK}; PY={PY}
# kill switch (no scancel from the agent): a pending job exits at start when runs/{R.name}/HOLD exists
if [ -e "$R/HOLD" ]; then echo "HOLD present: exiting"; exit 0; fi
cd "$R/code/tools"
{dep_chk}
grep -q "pretrain_range.py {pin_sha}" "$R/code/COMMIT" || {{ echo "## COMMIT does not name the trainer {pin_sha}"; exit 3; }}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU: $GPU"
case "$GPU" in *A100*) ;; *) echo "## not an A100 (the design's GPU; time limits are A100-based): exiting"; exit 3;; esac
mkdir -p "$OUT"
export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib
[ -s "$OUT/gpu.csv" ] || echo "timestamp, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]" > "$OUT/gpu.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader -l 60 >> "$OUT/gpu.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
{{ date -u +%FT%TZ; hostname; echo "SLURM_JOB_ID ${{SLURM_JOB_ID:-}} restart ${{SLURM_RESTART_COUNT:-0}}"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "cpus $(nproc)"; }} >> "$OUT/node.txt" 2>&1 || true

echo "## 1. inputs $(date -u +%FT%TZ)"
(cd "$DATA" && sha256sum -c -) <<'SHA' || exit 3
{DATA_SHA[0]}  train.bin
{DATA_SHA[1]}  val.bin
SHA
ARGS="{args}"
"$PY" pretrain_range.py $ARGS --dry_inputs > "$OUT/dry_inputs.txt" 2>&1 || {{ cat "$OUT/dry_inputs.txt"; exit 3; }}
cat "$OUT/dry_inputs.txt"
grep -qxF '{S['model_line']}' "$OUT/dry_inputs.txt" || {{ echo "## not the intended model size / gradient checkpointing state"; exit 3; }}
""" + target_assert + "\n" + lam_loop + "\n" + f"""grep -q '^first line after the text (share of rows): {{"numbers": 1.0, ' "$OUT/dry_inputs.txt" || {{ echo "## the numbers line is not first in every row"; exit 3; }}
grep -q '^content scan: 640 training rows, no held-out prompt of ' "$OUT/dry_inputs.txt" || {{ echo "## no content scan"; exit 3; }}
n=$(set +o pipefail; grep -cE "{P.DRY_GREP}" "$OUT/dry_inputs.txt")
got=$(set +o pipefail; grep -E "{P.DRY_GREP}" "$OUT/dry_inputs.txt" | sha256sum | cut -d' ' -f1)
[ "$n" = {n_dry} ] && [ "$got" = {dry} ] || {{ echo "## dry_inputs differ from the reference rows ($n lines, sha256 $got)"; exit 3; }}
echo "## dry_inputs rows = the reference ({dry[:12]}, {n_dry} lines)"

echo "## 2. training $(date -u +%FT%TZ)"
"$PY" pretrain_range.py $ARGS
echo "## 3. offline LM $(date -u +%FT%TZ)"
"$PY" offline_lm.py --data "$DATA" --tokenizer "$TOK" --out "$OUT/offline_lm.json" "$OUT/$TAG"
echo "## 4. intent checks $(date -u +%FT%TZ)"
"$PY" - "$OUT/$TAG/{runfile}" "$OUT/offline_lm.json" "$OUT/gpu.csv" "$R/code/COMMIT" "$DATA" \\
  "$OUT/check.json" '{ij}' <<'PY' | tee "$OUT/check.txt"
{G.CHECK_PY}PY
echo "## done $(date -u +%FT%TZ)"
""")
    return name, tag, tl, args, I


def main():
    bname = opt("--batch")
    assert bname in BATCHES, f"--batch one of {sorted(BATCHES)}"
    cfg = BATCHES[bname]
    size = cfg["size"]
    R = B / "runs" / bname
    head = snapshot(R)
    pin_sha = WANT_SHA["pretrain_range.py"]
    sys.path.insert(0, str(R / "code" / "tools"))
    import make_rb6m_tasks as G   # the snapshot's generator: CHECK_PY, intent, arm flags (the Burst tasks' own tables)
    import make_rb6_tasks as P
    assert Path(G.__file__).resolve().parent == (R / "code" / "tools").resolve()
    stage = opt("--stage", "main")
    assert stage == "main", stage
    register(G)
    cells, seeds = cfg["stages"][stage], cfg["seeds"]
    assert len(cells) == len(set(cells)) == 6 and seeds == (0, 1, 2, 3)
    if "--submit" in sys.argv:   # pre-registration first; never without the CPU references
        assert (R / "AUDIT.md").exists() and "--no_refs" not in sys.argv, "write runs/<batch>/AUDIT.md first; no --no_refs"
    existing = {p.name for p in B.glob("runs/*/out/*")}
    jobs = []
    for s in seeds:   # seed-major: every finished seed has every cell (a partial read-out stays paired)
        for arm, lr in cells:
            name, tag, text, args, I = render(G, P, R, size, arm, lr, s, pin_sha)
            assert name not in existing or (R / "out" / name).exists(), name   # never another batch's out dir
            job = R / f"job_{name}.sbatch"
            flag = R / f"SUBMITTED_{name}"
            if flag.exists():   # a submitted job file is never rewritten (the automatic stage re-renders reused cells)
                if job.read_text() != text:
                    print(f"WARNING: {job.name} was submitted and differs from this rendering: kept, not rewritten")
            else:
                job.write_text(text)
            jobs.append((name, job, flag, args))
    if "--print" in sys.argv:
        for name, job, flag, args in jobs:
            print(f"{name}: {args}")
    for name, job, flag, args in jobs:
        if "--test-only" in sys.argv:
            o = subprocess.run(["sbatch", "--test-only", str(job)], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
    print(f"{bname} {stage}: {len(jobs)} jobs rendered in {R}; code {head}, trainer {pin_sha}; "
          f"submitted: {'yes' if '--submit' in sys.argv else 'no'}")


if __name__ == "__main__":
    main()
