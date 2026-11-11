"""Torch jobs (2026-11-02, user decision; intent spec and PRE-REGISTERED read-out: runs/pt2same/AUDIT.md): pt2same — the
2% arithmetic size comparison at a fixed 3e8 tokens on ONE cluster. F86's 53M (pt2b) and 190M (pt2s190) cells ran on Burst
(GCP A100-40GB, trainer a3bce3ab63c1), the 505M cells (pt2s505) on Torch A100-80GB (trainer 93e22bf4c4c0); the
pre-registered cross-site check failed for 2% win-ctx (Burst-ro tools/xsite_summary.py: P1b-win-ctx mean -1.55, |.| > 1.50),
so 2% cells are not compared across sites. This batch re-runs the 53M and 190M cells on Torch A100-80GB with pt2s505's
trainer snapshot (git archive 7f1d06b tools/, = runs/pt2s505/code file by file) and pt2s505's job body, at F86's picks
(no new selection), seeds 0-7:
  53M   none@8e-3   win-ctx@2e-3      (pt2b:    Burst tasks 350-353 seeds 0-3, 370-373 seeds 4-7)
  190M  none@4.8e-3 win-ctx@6e-4      (pt2s190: none 366-369 / win-ctx 358-361 seeds 0-3, 382-385 seeds 4-7)
4 cells x 8 seeds = 32 jobs. 505M = pt2s505 none@4.8e-3 / win-ctx@6e-4 (8 seeds, Torch A100-80GB), reused read-only.
Trainer ARGS = the Burst task's ARGS (make_pt2main_tasks.render) with the Torch data / tokenizer / out paths and this
batch's tag, asserted string-equal at render time; CHECK_PY = make_pt2main_tasks.CHECK_PY verbatim; the intent = the
Burst task's intent except code_sha, tag and the step-0 lm reference (asserted against the Burst task files that ran).
Every job: HOLD, the snapshot's shas, an A100-80GB guard (exit 3 on anything else), the data sha256, an INTENT line,
--dry_inputs (model line, 2 operand sites per row, target table, the 9 DRY rows' sha256 = make_pt2main_tasks.DRY = the
Burst run's own dry_inputs.txt of this cell and seed, asserted here), training (--save_model --ckpt_every 2000, --requeue
resumes), offline_lm.py, CHECK_PY -> check.json (exit 4 on a fail), gpu.csv. Account torch_pr_520_tandon_priority,
a100_tandon, gpu48, nice 20.
Reused read-only (sha256[:12] asserted; importing them runs no main(), asserted by ast): make_jobs_pttpp.py (_extract, SIZES /
N_nonemb, the 190M step-0 references), make_jobs_pt2s505.py, make_jobs_rb6scale.py (WANT_SHA, accounts, BURST).
This generator never submits: runs/pt2same/submit_pt2same.sh does (prints the sbatch commands unless --go).
    python make_jobs_pt2same.py [--plan] [--print] [--test-only <name> ...]"""
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
FROZEN = {"make_jobs_pt2s505.py": "a5a7fb60eedc", "make_jobs_pttpp.py": "b4bfa6eb248d", "make_jobs_rb6scale.py": "19bc055c2d09",
          "make_jobs_ptlr1b.py": "8dd7b3407459", "pt2s505_refs.json": "479e394771e9", "pttpp_refs.json": "d10d720c1242"}
for _f, _v in FROZEN.items():
    assert sha(HERE / _f) == _v, (_f, sha(HERE / _f), _v, "a read-only generator / reference file changed")
for _f, _ok in (("make_jobs_pttpp.py", ("sys.path.insert(0, str(HERE))",)),):   # importing it runs no main()
    _t = ast.parse((HERE / _f).read_text())
    _calls = [ast.unparse(n) for n in _t.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    _g = [n for n in _t.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
    assert tuple(_calls) == _ok and len(_g) == 1 and [ast.unparse(x) for x in _g[0].body] == ["main()"], (_f, _calls)
sys.path.insert(0, str(HERE))
import make_jobs_pttpp as TP    # noqa: E402  (its import asserts make_jobs_pt2s505 / rb6scale / ptlr1b / refs by sha + ast)
import make_jobs_pt2s505 as P2  # noqa: E402  (pt2s505: the job body this batch copies; WANT_SHA)
import make_jobs_rb6scale as J  # noqa: E402  (WANT_SHA, DATA_SHA, the A100 account, BURST)

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pt2same"
BLOCK = "pt2same"
TOK, PY, DATA = J.TOK, J.PY, J.DATA
assert B == J.B == HERE and DATA == B / "pretrain_data"
WANT_SHA = J.WANT_SHA                      # the 2% trainer 93e22bf4c4c0 and every module it imports (= pt2s505's)
assert WANT_SHA is P2.WANT_SHA and WANT_SHA is TP.WANT_SHA2 and WANT_SHA["pretrain_range.py"] == "93e22bf4c4c0"
PIN2, PIN2_CODE = TP.PIN2, TP.PIN2_CODE   # 7f1d06b, runs/pt2s505/code
assert PIN2 == "7f1d06b" and PIN2_CODE == B / "runs" / "pt2s505" / "code"
EXTRA_SHA = {"make_pt2main_tasks.py": "a2e8a9621383", "readout_rules.py": "e91ee8a8a846", "pt2main_summary.py": "68c3b7c26de2"}
ACCOUNT, PARTITION, QOS, NICE = J.ACCOUNT, J.PARTITION, J.QOS, J.NICE
assert (ACCOUNT, PARTITION, QOS, NICE) == ("torch_pr_520_tandon_priority", "a100_tandon", "gpu48", 20)
GPU_CASE = "*A100*80GB*"   # Torch a100_tandon: 'NVIDIA A100-SXM4-80GB' (every pt2s505 / pttpp 2% node.txt)
BURST = J.BURST            # /scratch/yh6210/llmadd-burst-ro (origin/main of the Burst repo; read only)

# ---------------------------------------------------------------- design (pre-registered: runs/pt2same/AUDIT.md)
# F86's picks (FINDINGS F86 + its R02 addendum; tools/pt2main_summary.py on the Burst results prints the same best lrs):
# fixed in advance, no selection in this batch
PICKS = {"53m": {"none": "8e-3", "win-ctx": "2e-3"}, "190m": {"none": "4.8e-3", "win-ctx": "6e-4"}}
SIZES = ("53m", "190m")
ARMS = ("none", "win-ctx")
SEEDS = tuple(range(8))
BURST_BLOCK = {"53m": "pt2b", "190m": "pt2s190"}
# planning A100-h per run: 53M = the Burst A100-40GB xsite53 measurement (0.722 h, xsite_summary.py); 190M = 1.9 h
# (xsite's rescaled 1.92 h; Burst pt2s190 elapsed ~2.1 h; Torch A100-80GB tpp2-190m 0.40 s/step at eval every 70 steps
# -> ~1.85 h at 3e8 tokens / eval every 250). Limit = 2.5 x plan rounded up to the half hour (pttpp's rule); TIMEOUT is
# not requeued, preemption is (--requeue + ckpt every 2000 steps)
PLAN_H = {"53m": 0.75, "190m": 1.9}
TLIM = {"53m": "02:00:00", "190m": "05:00:00"}
STEPS, TOK_STEP, EVAL_EVERY = 18310, 32 * 512, 250
for _s in SIZES:
    _h = math.ceil(2.5 * PLAN_H[_s] * 2) / 2
    assert TLIM[_s] == f"{int(_h):02d}:{int(round(60 * (_h % 1))):02d}:00", (_s, _h)
# order of submission (submit_pt2same.sh): seed-major, the longer 190M first, so the first wave shows every cell
ORDER = [(s, z, a) for s in SEEDS for z in ("190m", "53m") for a in ARMS]


def g2():
    """make_pt2main_tasks of this batch's snapshot (CHECK_PY, DRY_GREP, DATA, ARM, WANT, SIZES, LM0, DRY, flags, render)"""
    d = str(R / "code" / "tools")
    if d not in sys.path:
        sys.path.insert(0, d)
    import make_pt2main_tasks as G2
    assert Path(G2.__file__).resolve().parent == (R / "code" / "tools").resolve(), G2.__file__
    return G2


def snapshot():
    """runs/pt2same/code = git archive of 7f1d06b tools/ (once), asserted = runs/pt2s505/code file by file, COMMIT copied
    verbatim (pttpp's _extract); every trainer module = WANT_SHA, the read-out modules = EXTRA_SHA"""
    R.mkdir(parents=True, exist_ok=True)
    TP._extract(R / "code", PIN2, PIN2_CODE, (PIN2_CODE / "COMMIT").read_text())
    cm = (R / "code" / "COMMIT").read_text()
    for f, v in {**WANT_SHA, **EXTRA_SHA}.items():
        assert sha(R / "code" / "tools" / f) == v and f"{f} {v}" in cm, (f, sha(R / "code" / "tools" / f), v)
    return cm.split()[0]


def lm0_ref(G2, size, seed):
    """the same-seed step-0 lm reference (arm- and lr-independent: same init and first batch): 53M = make_pt2main_tasks.LM0
    (Torch pt999 2pct-none, seeds 0-7, what the Burst pt2b tasks checked against); 190M = pttpp's '2|190m' (GPU values:
    seeds 0-3 = make_pt2main_tasks.LM0 = ptsize999, 4-7 = the Burst pt2s190 runs), asserted equal on seeds 0-3"""
    if size == "53m":
        return G2.LM0[("53m", "pretrain_data")][seed]
    v = TP.lm0_ref("2", "190m", seed)
    if seed < len(G2.LM0[("190m", "pretrain_data")]):
        assert v == G2.LM0[("190m", "pretrain_data")][seed], (seed, v)
    return v


def burst_tasks(G2):
    """{(size, arm, lr, seed): Burst task name} for this batch's cells, from make_pt2main_tasks.tasks() (base + EXTRA);
    exactly one task per cell and seed"""
    out = {}
    for name, tag, text, args, block, arm, lr, seed in G2.tasks():
        for size in SIZES:
            if block == BURST_BLOCK[size] and arm in ARMS and PICKS[size][arm] == lr:
                assert (size, arm, lr, seed) not in out, (name, out[(size, arm, lr, seed)])
                out[(size, arm, lr, seed)] = name
    assert sorted(out) == sorted((z, a, PICKS[z][a], s) for z in SIZES for a in ARMS for s in SEEDS), sorted(out)
    return out


def burst_intent(text):
    """the CHECK_PY intent JSON embedded in a Burst task / job file (the argument before <<'PY')"""
    m = re.findall(r"""^  "\$OUT/check\.json" '(\{.*\})' <<'PY'""", text, re.M)
    assert len(m) == 1, "no unique intent line"
    return json.loads(m[0])


def dry_rows_hash(G2, dry_txt):
    """the job's DRY check on a dry_inputs.txt: grep -E DRY_GREP | head -n 9 | sha256sum"""
    g = re.compile(G2.DRY_GREP)
    sel = [l for l in dry_txt.splitlines() if g.match(l)][:9]
    return hashlib.sha256(("\n".join(sel) + "\n").encode()).hexdigest()


def intent(G2, size, arm, lr, seed, code_sha=None, tag=None, lm0="ref"):
    """CHECK_PY's intent (make_jobs_pt2s505.intent with the size facts of make_pt2main_tasks.SIZES); code_sha / tag / lm0
    default to this batch's (the Burst task's values are passed in the known-answer assert of render())"""
    sz = G2.SIZES[size]
    d, nl, nh, dff = sz["dims"]
    steps = int(3e8 // (G2.BS * G2.SEQ))
    events, on_frac = G2.window_events(steps) if arm != "none" else ([], 0.0)
    site = G2.WANT[arm]["--site"] or "ctx"
    return {"code_sha": code_sha or WANT_SHA["pretrain_range.py"], "steps": steps, "n_params": sz["n_params"],
            "grad_ckpt_on": sz["grad_ckpt"], "geo_layers": [nl // 3, 2 * nl // 3], "read_layers": [nl // 3, 2 * nl // 3],
            "lm_chunk": 0, "eval_every_steps": G2.EVAL_EVERY,
            "evals": len(range(0, steps, G2.EVAL_EVERY)) + ((steps - 1) % G2.EVAL_EVERY != 0),
            "events": events, "on_frac": on_frac, "site": site,
            "lm0_ref": lm0_ref(G2, size, seed) if lm0 == "ref" else lm0,
            "args": {"tokenization": "whole", "target": G2.WANT[arm]["--target"], "site": site,
                     "schedule": G2.WANT[arm]["--schedule"] or "whole", "window": "0.05,0.4", "arith_lines": 1, "lr": float(lr),
                     "lam": float(G2.WANT[arm].get("--lam") or 5.0), "tokens": 3e8, "bs": G2.BS, "seq": G2.SEQ, "d_model": d,
                     "n_layers": nl, "n_heads": nh, "d_ff": dff, "seed": seed, "tag": tag or f"_{tag_of(size, arm, lr)}",
                     "save_model": True, "ckpt_every": 2000, "geo_layers": None, "eval_every": G2.EVAL_EVERY, "n_eval": 150,
                     "grad_ckpt": "auto", "lm_chunk_tokens": -1, "untie": False, "gate_thr": 0.5, "gate_start": 0.0,
                     "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256, "stop_after": -1, "dry_inputs": False}}


def tag_of(size, arm, lr):
    return f"{BLOCK}-{size}-lr{lr}-{arm}"


def render(G2, size, arm, seed, btasks):
    """(name, tag, job text, ARGS, intent, Burst task name) of one cell and seed, every intended fact asserted"""
    assert size in SIZES and arm in ARMS and seed in SEEDS
    lr = PICKS[size][arm]
    tag = tag_of(size, arm, lr)
    name = f"{tag}-s{seed}"
    OUT = R / "out" / name
    sz = G2.SIZES[size]
    d, nl, nh, dff = sz["dims"]
    size_flag = f" {sz['flags']}" if sz["flags"] else ""
    args = (f"--data {DATA} --tokenizer {TOK} --tokenization whole --out {OUT}/{tag} {G2.ARM[arm]} --arith_lines 1{size_flag} "
            f"--lr {lr} --seed {seed} --tag _{tag} --save_model --ckpt_every 2000")
    # 1. ARGS = the Burst task's ARGS (make_pt2main_tasks.render) with the Torch paths and this batch's tag, as a string
    bname = btasks[(size, arm, lr, seed)]
    btag = f"{BURST_BLOCK[size]}-lr{lr}-{arm}"
    assert bname == f"{bname[:3]}-{btag}-s{seed}", (bname, btag)
    btext = (BURST / "tasks" / f"{bname}.sh").read_text()          # the task file that ran on Burst
    bargs = re.findall(r'^ARGS="(.*)"$', btext, re.M)
    assert len(bargs) == 1, bname
    assert f'SEED={seed}; TAG={btag}; ARM="{G2.ARM[arm]}"' in btext, bname
    emul = (bargs[0].replace("$DATA", str(DATA)).replace("$TOK", str(TOK)).replace("$OUT", str(OUT)).replace("$TAG", tag)
            .replace("$ARM", G2.ARM[arm]).replace("$SEED", str(seed)))
    assert emul == args, (name, emul, args)
    # 2. every flag (make_pt2main_tasks.render's own asserts)
    f = G2.flags(args)
    want = {"--data": str(DATA), "--tokenizer": str(TOK), "--tokenization": "whole", "--out": f"{OUT}/{tag}", "--arith_lines": "1",
            "--lr": lr, "--seed": str(seed), "--tag": f"_{tag}", "--save_model": True, "--ckpt_every": "2000", "--tokens": None,
            "--lam": None, "--geo_layers": None, "--bs": None, "--seq": None, "--eval_every": None, "--n_eval": None,
            "--gate_thr": None, "--gate_start": None, "--gate_hyst": None, "--untie": None, "--grad_ckpt": None,
            "--lm_chunk_tokens": None, "--stop_after": None, "--dry_inputs": None, **G2.WANT[arm],
            **({"--d_model": str(d), "--n_layers": str(nl), "--n_heads": str(nh), "--d_ff": str(dff)} if sz["flags"] else
               {"--d_model": None, "--n_layers": None, "--n_heads": None, "--d_ff": None})}
    for k, v in want.items():
        assert f.get(k) == v, (name, k, f.get(k), v)
    assert set(f) <= {k for k, v in want.items() if v is not None}, (name, sorted(f))
    # 3. the intent = the Burst task's intent except code_sha, tag, lm0_ref (known answer: the task file that ran)
    I = intent(G2, size, arm, lr, seed)
    bI = burst_intent(btext)
    mine_as_burst = intent(G2, size, arm, lr, seed, code_sha=G2.PIN_SHA, tag=f"_{btag}", lm0=bI["lm0_ref"])
    assert mine_as_burst == bI, (name, "intent != the Burst task's", {k: (mine_as_burst.get(k), bI.get(k)) for k in bI if mine_as_burst.get(k) != bI.get(k)})
    assert I["lm0_ref"] is not None and (bI["lm0_ref"] is None or bI["lm0_ref"] == I["lm0_ref"]), (name, I["lm0_ref"], bI["lm0_ref"])
    assert I["steps"] == STEPS and I["evals"] == 75 and (I["events"] == [[916, 1], [7324, 0]] if arm != "none" else I["events"] == [])
    # 4. the DRY rows: make_pt2main_tasks.DRY = the Burst run's own dry_inputs.txt (this cell, this seed) = the task's value
    dry = G2.DRY[("pretrain_data", size, "ctx", seed)]
    assert dry in btext, (bname, "the Burst task checked another DRY value")
    assert dry_rows_hash(G2, (BURST / "results" / bname / "dry_inputs.txt").read_text()) == dry, (bname, "Burst dry_inputs.txt != DRY")
    assert PLAN_H[size] * 2 <= G2.hours(TLIM[size]) <= PLAN_H[size] * 3 + 0.5, (size, TLIM[size])
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    site_assert = ("""[ "$(grep -c '^train row [012] (len 512, 2 prior sites at ' "$OUT/dry_inputs.txt")" = 3 ] """
                   """|| { echo "## not 1 arithmetic line (2 operand sites) per row"; exit 3; }""")
    target_assert = ("""! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a prior target in a none run"; exit 3; }"""
                     if arm == "none" else
                     """grep -qxF 'target features (1000, 10)' "$OUT/dry_inputs.txt" || { echo "## not the helix3 table (1000 x 10: cos 2, cos/sin 5 10 100 1000, x / 1000)"; exit 3; }""")
    runfile = f"run_whole_{G2.WANT[arm]['--target']}_seed{seed}_{tag}.json"
    pin_sha = WANT_SHA["pretrain_range.py"]
    dep_chk = "\n".join(f'[ "$(sha256sum {fn} | cut -c1-12)" = "{v}" ] || {{ echo "## {fn} sha mismatch: exiting"; exit 3; }}'
                        for fn, v in WANT_SHA.items())
    nne = TP.SIZES[size][5]
    assert TP.SIZES[size][4] == sz["n_params"] and TP.SIZES[size][:4] == sz["dims"], size
    events = I["events"]
    win = f"steps {events[0][0]}..{events[1][0] - 1}" if events else "none (no prior)"
    about = textwrap.fill(f"pt2same (2% size comparison on one cluster): F86's cell {btag} (Burst task {bname}, A100-40GB) "
                          f"re-run on Torch A100-80GB with pt2s505's trainer and job body; cell {tag}: {G2.ARM_DESC[arm]}, lr "
                          f"{lr} (F86's pick, fixed in advance), seed {seed}; {size}, 2% arithmetic (1 line per row), lambda 5, "
                          "3e8 tokens. Intent spec and pre-registered read-out runs/pt2same/AUDIT.md; generated by "
                          "llmadd_torch/make_jobs_pt2same.py (change the generator, not this file).", 120,
                          initial_indent="# ", subsequent_indent="# ", break_on_hyphens=False, break_long_words=False)
    text = f"""#!/bin/bash
#SBATCH --job-name=llmadd_pt2same-{size}-{arm}-lr{lr}-s{seed}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --qos={QOS}
#SBATCH --gres=gpu:1
#SBATCH --nice={NICE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={TLIM[size]}
#SBATCH --requeue
#SBATCH --output={B}/logs/%x_%j.out
{about}
# 1. HOLD, snapshot shas, A100-80GB, data sha256; --dry_inputs: model line, 2 operand sites per row, target table, the 9 rows'
#    sha256 = make_pt2main_tasks.DRY = the Burst run's own rows of this cell and seed (exit 3 before training)
# 2. training (--save_model --ckpt_every 2000; a requeued job resumes); 3. offline_lm.py (full val);
# 4. make_pt2main_tasks.CHECK_PY with this cell's intent -> $OUT/check.json (exit 4 on a fail); gpu.csv every 60 s
set -euo pipefail
R={R}; OUT={OUT}; TAG={tag}; SEED={seed}; DATA={DATA}; TOK={TOK}; PY={PY}
# kill switch (no scancel from the agent): a pending job exits at start when runs/pt2same/HOLD exists
if [ -e "$R/HOLD" ]; then echo "HOLD present: exiting"; exit 0; fi
cd "$R/code/tools"
{dep_chk}
grep -q "pretrain_range.py {pin_sha}" "$R/code/COMMIT" || {{ echo "## COMMIT does not name the trainer {pin_sha}"; exit 3; }}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU: $GPU"
case "$GPU" in {GPU_CASE}) ;; *) echo "## not an A100-80GB (this batch's one GPU type: Torch a100_tandon): exiting"; exit 3;; esac
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
echo "INTENT {name}: {size} d{d} x {nl} ({nh} heads, d_ff {dff}), {sz['n_params']} params, N_nonemb {nne}, grad ckpt {'on' if sz['grad_ckpt'] else 'off'}; tokens 3e8 = {STEPS} steps x {TOK_STEP}, eval every {EVAL_EVERY} steps -> {I['evals']} evals, warmup {max(1, STEPS // 100)} steps, prior window {win}; {arm} lr {lr} seed {seed}; step-0 lm ref {I['lm0_ref']:.6f}; Burst twin {bname}"
ARGS="{args}"
"$PY" pretrain_range.py $ARGS --dry_inputs > "$OUT/dry_inputs.txt" 2>&1 || {{ cat "$OUT/dry_inputs.txt"; exit 3; }}
cat "$OUT/dry_inputs.txt"
grep -qxF '{sz["model_line"]}' "$OUT/dry_inputs.txt" || {{ echo "## not the intended model size / gradient checkpointing state"; exit 3; }}
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
    return name, tag, text, args, I, bname


def jobs(G2):
    """every job of the batch in submission order: [(name, tag, text, args, intent, Burst task)]"""
    bt = burst_tasks(G2)
    out = [render(G2, z, a, s, bt) for s, z, a in ORDER]
    assert len(out) == 32 and len({o[0] for o in out}) == 32 and len({o[1] for o in out}) == 4
    return out


def plan():
    print(f"{'cell':32s} {'Burst twin tasks (s0-3 / s4-7)':40s} {'n_params':>11s} {'N_nonemb':>11s} {'h/run':>6s} {'limit':>9s}")
    G2 = g2()
    bt = burst_tasks(G2)
    tot = 0.0
    for z in SIZES:
        for a in ARMS:
            lr = PICKS[z][a]
            tw = [bt[(z, a, lr, s)][:3] for s in SEEDS]
            print(f"{tag_of(z, a, lr):32s} {tw[0] + '-' + tw[3] + ' / ' + tw[4] + '-' + tw[7] + ' (' + BURST_BLOCK[z] + ')':40s} "
                  f"{TP.SIZES[z][4]:11,d} {TP.SIZES[z][5]:11,d} {PLAN_H[z]:6.2f} {TLIM[z]:>9s}")
            tot += 8 * PLAN_H[z]
    print(f"32 runs, planned {tot:.1f} A100-h (limits reserve {sum(8 * 2 * G2.hours(TLIM[z]) for z in SIZES):.0f} h at most)")


def main():
    snapshot()
    G2 = g2()
    if "--plan" in sys.argv:
        plan()
        return
    au = R / "AUDIT.md"
    assert au.exists() and "## Pre-registered read-out" in au.read_text(), "runs/pt2same/AUDIT.md (the pre-registration) must exist first"
    existing = {p.name for p in B.glob("runs/*/out*/*") if p.parent.parent != R}   # other batches' run dirs
    rendered = []
    for name, tag, text, args, I, bname in jobs(G2):
        assert name not in existing and tag not in existing, name
        job, flag = R / f"job_{name}.sbatch", R / f"SUBMITTED_{name}"
        if flag.exists():   # a submitted job file is never rewritten
            if not job.exists() or job.read_text() != text:
                print(f"WARNING: {job.name} was submitted and differs from this rendering (or is missing): kept, not rewritten")
        elif not job.exists() or job.read_text() != text:
            job.write_text(text)
        rendered.append((name, job, args, bname))
    names = {f"job_{n}.sbatch" for n, *_ in rendered}
    stray = sorted(p.name for p in R.glob("job_*.sbatch") if p.name not in names)
    assert not stray, ("job files not made by this rendering", stray)
    (R / "JOBS.sha256").write_text("".join(f"{hashlib.sha256(j.read_bytes()).hexdigest()}  {j.name}\n" for _, j, _, _ in rendered))
    if "--print" in sys.argv:
        for name, _, args, bname in rendered:
            print(f"{name} (Burst twin {bname}): {args}")
    if "--test-only" in sys.argv:
        want = sys.argv[sys.argv.index("--test-only") + 1:]
        assert want and set(want) <= {n for n, *_ in rendered}, want
        for name, job, _, _ in rendered:
            if name in want:
                o = subprocess.run(["sbatch", "--test-only", str(job)], capture_output=True, text=True)
                print(f"{'OK ' if o.returncode == 0 and 'to start at' in o.stdout + o.stderr else 'BAD'} {name}: "
                      f"{(o.stdout + o.stderr).strip()}")
    print(f"pt2same: {len(rendered)} jobs rendered in {R} (manifest JOBS.sha256); code {PIN2}, trainer "
          f"{WANT_SHA['pretrain_range.py']}; nothing submitted (runs/pt2same/submit_pt2same.sh --go does)")


if __name__ == "__main__":
    main()
