"""Torch jobs (2026-10-22, the LLM_addition model-size ladder, batch 3: pt2s505) — F86's 2% single-prior main setting
(the pt2main design, report/audits/pt2main_AUDIT.md: 2% arithmetic = one line per row, helix3 window prior (0.05, 0.4)
on the operands = win-ctx at lambda 5 (the trainer default, as pt2main), every model saved) at 505M:
  none     lr 4.8e-3, 9.6e-3, 1.92e-2   seeds 0-3
  win-ctx  lr 3e-4, 6e-4, 1.2e-3        seeds 0-3          (24 jobs)
  confirm  seeds 4-7 of none@best and win-ctx@best (pt2main rule 1's best lr, read by tools/pt2main_summary.py); submitted
           automatically by the selection job when both arms' selections are final (runs/pt2s505/AUDIT.md)  (8 jobs)
Model = runs/ptlr's 500m: --d_model 1024 --n_layers 24 --n_heads 16 --d_ff 4096 (505,437,184 parameters, gradient
checkpointing on, prior / read-out layers 8, 16); 3e8 tokens = 18310 steps of 32 x 512 on pretrain_data.
This module also has make_pt2main_tasks.py's interface (BLOCKS, SEEDS, LM0, DRY, tasks()), so
    python tools/pt2main_summary.py --gen /scratch/yh6210/llmadd_torch/make_jobs_pt2s505.py --res runs/pt2s505/out
reads the batch with pt2main's own pre-registered rules. Every job = the pt2main Burst task's body on Torch: HOLD, the
snapshot shas, an A100 guard, the data sha256, --dry_inputs (model line, 3 rows with 2 operand sites, the target table,
the 9 DRY rows' sha256 = the Torch CPU 505M dry run = the Burst pt2s190 rows with this model line), training, offline_lm,
make_pt2main_tasks.CHECK_PY verbatim with the 505M intent -> check.json (exit 4 on a fail). Account
<slurm-account>, a100_tandon, gpu48, nice 20.
    python make_jobs_pt2s505.py [--stage main | --stage confirm --picks <json>] [--print] [--test-only] [--submit]"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_jobs_rb6scale as J   # noqa: E402  (the snapshot helper and the shared constants)

B, TOK, DATA, PY = J.B, J.TOK, J.DATA, J.PY
R = B / "runs" / "pt2s505"
BLOCK, SIZE = "pt2s505", "505m"
WANT_SHA = J.WANT_SHA
GEN2 = ("make_pt2main_tasks.py",)
EXTRA2 = ("pt2main_summary.py", "pt999_summary.py", "readout_rules.py")
ACCOUNT, PARTITION, QOS, NICE = J.ACCOUNT, J.PARTITION, J.QOS, J.NICE
DIMS, N_PARAMS, LAYERS = (1024, 24, 16, 4096), 505437184, [8, 16]
MODEL_LINE = "model: 505.4M parameters; gradient checkpointing on"
FLAGS = "--d_model 1024 --n_layers 24 --n_heads 16 --d_ff 4096"
TLIM, PLAN_H = "11:00:00", 5.3   # 2 x 5.3 h (505M single prior on A100, estimated from L40S 6.4 h), rounded up
CELLS = [("none", "4.8e-3"), ("none", "9.6e-3"), ("none", "1.92e-2"), ("win-ctx", "3e-4"), ("win-ctx", "6e-4"),
         ("win-ctx", "1.2e-3")]
SEEDS = range(4)
CONFIRM = (4, 5, 6, 7)
# make_pt2main_tasks' interface: (base number, block, size, data, tokens, planning A100-h, limit, cells)
BLOCKS = [(0, BLOCK, SIZE, "pretrain_data", "3e8", PLAN_H, TLIM, CELLS)]
# references (scale/refs_pt2.py on CPU, the snapshot trainer; llmadd_torch/pt2s505_refs.json): the step-0 lm of seeds 0-7
# (arm- and lr-independent; the CPU method calibrated at 505M against a GPU value: -5.4e-5) and the 9 DRY rows' sha256
LM0, DRY = {}, {}
REFS = Path(__file__).with_name("pt2s505_refs.json")
if REFS.exists():
    _r = json.loads(REFS.read_text())
    LM0[(SIZE, "pretrain_data")] = tuple(_r["lm0"][str(s)] for s in range(8))
    DRY.update({("pretrain_data", SIZE, "ctx", int(s)): v for s, v in _r["dry"].items()})
BURST_DRY = {s: f"3{58 + s}-pt2s190-lr6e-4-none-s{s}" for s in range(4)}
BURST_DRY.update({s: f"3{78 + s}-pt2s190-lr4.8e-3-none-s{s}" for s in CONFIRM})   # 382-385: pt2s190 none@4.8e-3 s4-7
PICKS = R / "picks_confirm.json"   # written by the selection job (status 'final'); the confirm cells come from it


def confirm_cells():
    """[(arm, lr, seed)] of the confirm stage, from the selection's picks (empty while there are none)."""
    if not PICKS.exists():
        return []
    pk = json.loads(PICKS.read_text())
    assert pk["status"] == "final" and pk["block"] == BLOCK, pk
    out = [(arm, pk["best"][arm], s) for s in CONFIRM for arm in ("none", "win-ctx")]
    assert all((a, lr) in CELLS for a, lr, _ in out), out
    return out


def intent(G2, arm, lr, seed, size_facts=None, code_sha=None):
    """CHECK_PY's intent, built as make_pt2main_tasks.render builds it (asserted against its pt2s190 rendering in the
    tests: with the 190M facts this function returns render()'s intent exactly)."""
    dims, n_params, ckpt = size_facts or (DIMS, N_PARAMS, True)
    d, nl, nh, dff = dims
    steps = int(3e8 // (G2.BS * G2.SEQ))
    events, on_frac = G2.window_events(steps) if arm != "none" else ([], 0.0)
    site = G2.WANT[arm]["--site"] or "ctx"
    lm0 = LM0.get((SIZE, "pretrain_data")) if size_facts is None else G2.LM0.get(("190m", "pretrain_data"))
    return {"code_sha": code_sha or WANT_SHA["pretrain_range.py"], "steps": steps, "n_params": n_params, "grad_ckpt_on": ckpt,
            "geo_layers": [nl // 3, 2 * nl // 3], "read_layers": [nl // 3, 2 * nl // 3], "lm_chunk": 0,
            "eval_every_steps": G2.EVAL_EVERY, "evals": len(range(0, steps, G2.EVAL_EVERY)) + ((steps - 1) % G2.EVAL_EVERY != 0),
            "events": events, "on_frac": on_frac, "site": site, "lm0_ref": lm0[seed] if lm0 and seed < len(lm0) else None,
            "args": {"tokenization": "whole", "target": G2.WANT[arm]["--target"], "site": site,
                     "schedule": G2.WANT[arm]["--schedule"] or "whole", "window": "0.05,0.4", "arith_lines": 1, "lr": float(lr),
                     "lam": float(G2.WANT[arm].get("--lam") or 5.0), "tokens": 3e8, "bs": G2.BS, "seq": G2.SEQ, "d_model": d,
                     "n_layers": nl, "n_heads": nh, "d_ff": dff, "seed": seed, "tag": f"_{BLOCK if size_facts is None else 'pt2s190'}-lr{lr}-{arm}",
                     "save_model": True, "ckpt_every": 2000, "geo_layers": None, "eval_every": G2.EVAL_EVERY, "n_eval": 150,
                     "grad_ckpt": "auto", "lm_chunk_tokens": -1, "untie": False, "gate_thr": 0.5, "gate_start": 0.0,
                     "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256, "stop_after": -1, "dry_inputs": False}}


def burst_dry_expected(G2, seed):
    """The 9 DRY rows' sha256 expected at 505M from the Burst pt2s190 dry log of the seed (none = ctx rows): the same rows
    with the model line replaced (every other selected line is model-independent)."""
    txt = (J.BURST / "results" / BURST_DRY[seed] / "dry_inputs.txt").read_text().splitlines()
    g = re.compile(G2.DRY_GREP)
    sel = [l for l in txt if g.match(l)][:9]
    h190 = hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest()
    assert h190 == G2.DRY[("pretrain_data", "190m", "ctx", seed)], (BURST_DRY[seed], "emulation != the recorded DRY")
    m190 = G2.SIZES["190m"]["model_line"]
    assert sum(l == m190 for l in sel) == 1, BURST_DRY[seed]
    return hashlib.sha256(("\n".join(MODEL_LINE if l == m190 else l for l in sel) + "\n").encode()).hexdigest()


def render(G2, arm, lr, seed, pin_sha):
    tag = f"{BLOCK}-lr{lr}-{arm}"
    name = f"{tag}-s{seed}"
    OUT = R / "out" / name
    args = (f"--data {DATA} --tokenizer {TOK} --tokenization whole --out {OUT}/{tag} {G2.ARM[arm]} --arith_lines 1 {FLAGS} "
            f"--lr {lr} --seed {seed} --tag _{tag} --save_model --ckpt_every 2000")
    f = G2.flags(args)
    d, nl, nh, dff = DIMS
    want = {"--data": str(DATA), "--tokenizer": str(TOK), "--tokenization": "whole", "--out": f"{OUT}/{tag}", "--arith_lines": "1",
            "--lr": lr, "--seed": str(seed), "--tag": f"_{tag}", "--save_model": True, "--ckpt_every": "2000", "--tokens": None,
            "--lam": None, "--geo_layers": None, "--bs": None, "--seq": None, "--eval_every": None, "--n_eval": None,
            "--gate_thr": None, "--gate_start": None, "--gate_hyst": None, "--untie": None, "--grad_ckpt": None,
            "--lm_chunk_tokens": None, "--stop_after": None, "--dry_inputs": None, **G2.WANT[arm],
            "--d_model": str(d), "--n_layers": str(nl), "--n_heads": str(nh), "--d_ff": str(dff)}
    for k, v in want.items():
        assert f.get(k) == v, (name, k, f.get(k), v)
    assert set(f) <= {k for k, v in want.items() if v is not None}, (name, sorted(f))
    assert arm in ("none", "win-ctx") and (arm, lr) in CELLS, (arm, lr)
    assert 2 * PLAN_H <= G2.hours(TLIM) < 2 * PLAN_H + 1
    I = intent(G2, arm, lr, seed)
    assert I["lm0_ref"] is not None, f"{name}: no step-0 lm reference (seed {seed}); run scale/refs_pt2.py first"
    assert ("pretrain_data", SIZE, "ctx", seed) in DRY, f"{name}: no DRY rows hash (seed {seed}); run scale/refs_pt2.py first"
    dry = DRY[("pretrain_data", SIZE, "ctx", seed)]
    assert dry == burst_dry_expected(G2, seed), (name, "Torch CPU 505M dry rows != the Burst pt2s190 rows + model line")
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    site_assert = ("""[ "$(grep -c '^train row [012] (len 512, 2 prior sites at ' "$OUT/dry_inputs.txt")" = 3 ] """
                   """|| { echo "## not 1 arithmetic line (2 operand sites) per row"; exit 3; }""")
    target_assert = ("""! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a prior target in a none run"; exit 3; }"""
                     if arm == "none" else
                     """grep -qxF 'target features (1000, 10)' "$OUT/dry_inputs.txt" || { echo "## not the helix3 table (1000 x 10: cos 2, cos/sin 5 10 100 1000, x / 1000)"; exit 3; }""")
    runfile = f"run_whole_{G2.WANT[arm]['--target']}_seed{seed}_{tag}.json"
    dep_chk = "\n".join(f'[ "$(sha256sum {fn} | cut -c1-12)" = "{v}" ] || {{ echo "## {fn} sha mismatch: exiting"; exit 3; }}'
                        for fn, v in WANT_SHA.items())
    text = f"""#!/bin/bash
#SBATCH --job-name=llmadd_pt2s505-{arm}-lr{lr}-s{seed}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --qos={QOS}
#SBATCH --gres=gpu:1
#SBATCH --nice={NICE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={TLIM}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
# pt2s505 (scale ladder batch 3): F86's 2% single-prior main setting (pt2main) at 505M; cell {tag}: {G2.ARM_DESC[arm]},
# lr {lr}, seed {seed}; 2% arithmetic (1 line per row), lambda 5. Intent spec and pre-registered rules runs/pt2s505/AUDIT.md
# (= pt2main's, report/audits/pt2main_AUDIT.md); generated by llmadd_torch/make_jobs_pt2s505.py (change the generator).
# 1. HOLD, snapshot shas, A100, data sha256; --dry_inputs: model line, 2 operand sites per row, target table, the 9 rows'
#    sha256 = the Torch CPU 505M dry run = the Burst pt2s190 seed-{seed} rows with this model line (exit 3 before training)
# 2. training (--save_model --ckpt_every 2000; a requeued job resumes); 3. offline_lm.py (full val);
# 4. make_pt2main_tasks.CHECK_PY with this cell's intent -> $OUT/check.json (exit 4 on a fail); gpu.csv every 60 s
set -euo pipefail
R={R}; OUT={OUT}; TAG={tag}; SEED={seed}; DATA={DATA}; TOK={TOK}; PY={PY}
# kill switch (no scancel from the agent): a pending job exits at start when runs/pt2s505/HOLD exists
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
{G2.DATA['pretrain_data'][0]}  train.bin
{G2.DATA['pretrain_data'][1]}  val.bin
SHA
ARGS="{args}"
"$PY" pretrain_range.py $ARGS --dry_inputs > "$OUT/dry_inputs.txt" 2>&1 || {{ cat "$OUT/dry_inputs.txt"; exit 3; }}
cat "$OUT/dry_inputs.txt"
grep -qxF '{MODEL_LINE}' "$OUT/dry_inputs.txt" || {{ echo "## not the intended model size / gradient checkpointing state"; exit 3; }}
{site_assert}
{target_assert}
got=$(set +o pipefail; grep -E "{G2.DRY_GREP}" "$OUT/dry_inputs.txt" | head -n 9 | sha256sum | cut -d' ' -f1)
[ "$got" = {dry} ] || {{ echo "## dry_inputs differ from the reference rows (sha256 $got)"; exit 3; }}
echo "## dry_inputs rows = the reference ({dry[:12]})"

echo "## 2. training $(date -u +%FT%TZ)"
"$PY" pretrain_range.py $ARGS
echo "## 3. offline LM $(date -u +%FT%TZ)"
"$PY" offline_lm.py --data "$DATA" --tokenizer "$TOK" --out "$OUT/offline_lm.json" "$OUT/$TAG"
echo "## 4. intent checks $(date -u +%FT%TZ)"
"$PY" - "$OUT/$TAG/{runfile}" "$OUT/offline_lm.json" "$OUT/gpu.csv" "$R/code/COMMIT" "$DATA" \\
  "$OUT/check.json" '{ij}' <<'PY' | tee "$OUT/check.txt"
{G2.CHECK_PY}PY
echo "## done $(date -u +%FT%TZ)"
"""
    return name, tag, text, args, I


def _g2():
    sys.path.insert(0, str(R / "code" / "tools"))
    import make_pt2main_tasks as G2
    assert Path(G2.__file__).resolve().parent == (R / "code" / "tools").resolve(), G2.__file__
    return G2


def tasks():
    """make_pt2main_tasks.tasks()'s interface for tools/pt2main_summary.py --gen: [(name, tag, text, args, block, arm, lr,
    seed)], the 24 main tasks seed-major, then the confirm tasks (seeds 4-7) once the picks exist."""
    G2 = _g2()
    out = [render(G2, a, lr, s, WANT_SHA["pretrain_range.py"])[:4] + (BLOCK, a, lr, s) for s in SEEDS for a, lr in CELLS]
    out += [render(G2, a, lr, s, WANT_SHA["pretrain_range.py"])[:4] + (BLOCK, a, lr, s) for a, lr, s in confirm_cells()]
    return out


def main():
    _orig = (J.GEN, J.EXTRA_FILES)
    J.GEN, J.EXTRA_FILES = GEN2, EXTRA2   # this batch's snapshot holds the pt2main generator and read-out
    try:
        head = J.snapshot(R)
    finally:
        J.GEN, J.EXTRA_FILES = _orig
    G2 = _g2()
    stage = J.opt("--stage", "main")
    if stage == "confirm":
        assert J.opt("--picks") and Path(J.opt("--picks")).resolve() == PICKS.resolve(), "--picks must be runs/pt2s505/picks_confirm.json"
        cells = confirm_cells()
        assert len(cells) == 8, cells
    else:
        cells = [(a, lr, s) for s in SEEDS for a, lr in CELLS]
    existing = {p.name for p in B.glob("runs/*/out/*")}
    jobs = []
    for a, lr, s in cells:
        name, tag, text, args, I = render(G2, a, lr, s, WANT_SHA["pretrain_range.py"])
        assert name not in existing or (R / "out" / name).exists(), name
        job, flag = R / f"job_{name}.sbatch", R / f"SUBMITTED_{name}"
        if flag.exists():   # a submitted job file is never rewritten (the automatic stage re-renders reused cells)
            if job.read_text() != text:
                print(f"WARNING: {job.name} was submitted and differs from this rendering: kept, not rewritten")
        else:
            job.write_text(text)
        jobs.append((name, job, flag, args))
    if "--print" in sys.argv:
        for name, _, _, args in jobs:
            print(f"{name}: {args}")
    for name, job, flag, _ in jobs:
        if "--test-only" in sys.argv:
            o = subprocess.run(["sbatch", "--test-only", str(job)], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
        if "--submit" in sys.argv and not flag.exists():
            o = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
            print(name, (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                flag.write_text(o.stdout)
    print(f"pt2s505 {stage}: {len(jobs)} jobs rendered in {R}; code {head}, trainer {WANT_SHA['pretrain_range.py']}; "
          f"submitted: {'yes' if '--submit' in sys.argv else 'no'}")


if __name__ == "__main__":
    main()
