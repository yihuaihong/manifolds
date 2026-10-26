"""New-flag tests of the pttpp batch on CPU (memory new_flag_takes_effect: a non-default value must change the behaviour, and
the consumed value is logged / asserted). The batch is the first to pass non-default --tokens / --eval_every to F87's and
pt2s505's trainers (runs/pttpp/code7, code2). Tiny model (d64 x 2, seq 128, bs 32, 4 eval items),
4 threads, the real data and tokenizer (bs 32 as every job: the trainer multiplies --eval_every by 32 / bs; seq 128 -> 4096 tokens per step):
  1. --tokens 163840 --eval_every 5  -> steps 40, evals at 0,5,...,35,39 (9), eval_every_steps 5, window events [[2,1],[16,0]]
     --tokens 163840 (eval_every default 250) -> 2 evals (0, 39): --eval_every takes effect
     --tokens 327680 --eval_every 10 -> steps 80, evals 0,10,...,70,79 (9), events [[4,1],[32,0]]: --tokens takes effect,
     and the window scales with the steps (the trainer's w0 * steps <= s < w1 * steps)
  2. data stream is a prefix: lr 0 (the model never moves, so the logged lm of step s is the loss of batch s under the
     initial model): --tokens 163840 vs 327680, both --eval_every 5 -> lm identical at steps 0, 5, ..., 35 (none and win-ctx)
  3. static: the warmup / window / cosine lines of both snapshots are the fractions of steps the design relies on
Both trainers (code7 a3bce3ab63c1, code2 93e22bf4c4c0). Prints FLAGTEST PASS / FAIL."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

OUT = HERE / "flag_out"
ENV = dict(os.environ, HF_HOME="/scratch/yh6210/llmadd_torch/hf_cache", PYTHONPATH="/scratch/yh6210/llmadd_torch/pylib",
           PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")   # the same thread count in every run (CPU determinism)
ENV.pop("SLURM_JOB_ID", None)
TINY = "--d_model 64 --n_layers 2 --n_heads 1 --d_ff 256 --seq 128 --bs 32 --n_eval 4"   # bs 32: the trainer scales --eval_every by 32 / bs
ARM = {"none": "--target none", "win-ctx": "--target helix3 --schedule window --window 0.05,0.4 --site ctx"}
fails = []


def run(code, name, arm, extra):
    out = OUT / code / name
    args = (f"--data {T.DATA} --tokenizer {T.TOK} --tokenization whole --out {out} {ARM[arm]} --arith_lines 4 {TINY} "
            f"--seed 0 --tag _{name} {extra}").split()
    log = OUT / code / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        rc = subprocess.run([T.PY, "pretrain_range.py"] + args, cwd=T.R / code / "tools", env=ENV, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    f = sorted(out.glob("run_*.json"))
    assert rc == 0 and len(f) == 1, (code, name, rc, log.read_text()[-1500:])
    return json.loads(f[0].read_text())


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        fails.append(what)


for code in (sys.argv[1:] or ["code7", "code2"]):
    src = (T.R / code / "tools" / "pretrain_range.py").read_text()
    for line in ("steps = int(args.tokens // (args.bs * args.seq))", "warm = max(1, steps // 100)",
                 "return w0 * steps <= s < w1 * steps", "eval_every = max(1, round(args.eval_every * 32 / args.bs))",
                 "math.cos(math.pi * min(1.0, s / steps))", "if s % eval_every == 0 or s == steps - 1:"):
        check(line in src, f"{code}: trainer line present: {line}")
    a = run(code, "w-t20480-e5", "win-ctx", "--tokens 163840 --eval_every 5 --lr 1e-3")
    b = run(code, "w-t20480-edef", "win-ctx", "--tokens 163840 --lr 1e-3")
    c = run(code, "w-t40960-e10", "win-ctx", "--tokens 327680 --eval_every 10 --lr 1e-3")
    for r, S, E, ev in ((a, 40, 5, [[2, 1], [16, 0]]), (b, 40, 250, [[2, 1], [16, 0]]), (c, 80, 10, [[4, 1], [32, 0]])):
        st = [e["step"] for e in r["curve"]]
        want = sorted(set(range(0, S, E)) | {S - 1})
        check(r["steps"] == S and r["eval_every_steps"] == E and st == want and r["prior_events"] == ev
              and r["args"]["tokens"] == float(S * 4096) and st == [e["tokens"] // 4096 for e in r["curve"]],
              f"{code}: --tokens {S * 4096} --eval_every {E}: steps {r['steps']}, eval_every_steps {r['eval_every_steps']}, "
              f"{len(st)} evals {st[:3]}...{st[-2:]}, prior_events {r['prior_events']} (want {S}, {E}, {len(want)}, {ev})")
        # the design's own functions give the same numbers
        check(T.window_events(S)[0] == ev and T.n_evals(S, E) == len(want), f"{code}: make_jobs_pttpp.window_events / n_evals agree ({S}, {E})")
    check(len(a["curve"]) != len(b["curve"]), f"{code}: --eval_every 5 vs the default changes the number of evals ({len(a['curve'])} vs {len(b['curve'])})")
    for arm in ("none", "win-ctx"):
        p1 = run(code, f"{arm}-lr0-t20480", arm, "--tokens 163840 --eval_every 5 --lr 0")
        p2 = run(code, f"{arm}-lr0-t40960", arm, "--tokens 327680 --eval_every 5 --lr 0")
        l1 = {e["step"]: e["lm"] for e in p1["curve"]}
        l2 = {e["step"]: e["lm"] for e in p2["curve"]}
        common = [s for s in range(0, 40, 5)]
        same = all(l1[s] == l2[s] for s in common)
        distinct = len({l1[s] for s in common}) == len(common)   # the batches do change from step to step
        check(same and distinct, f"{code} {arm}: lr 0, 40 vs 80 steps: lm at steps {common} identical ({same}) and distinct "
                                 f"across steps ({distinct}): the data stream of a shorter run is the prefix of the longer one")
print("FLAGTEST", "PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
