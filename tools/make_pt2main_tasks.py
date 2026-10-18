"""Burst task files for the 2% main setting (pt2main). User decisions 2026-10-16: the 2% arithmetic share (--arith_lines 1)
is the MAIN setting and the 5% window (--schedule window --window 0.05,0.4) the main recipe; every arm, the prior too, gets
its own best lr; 2% long-budget runs (so none can reach 90% and a clean speed-up ratio exists) and 2% at 190M.
Intent spec + pre-registered rules: report/audits/pt2main_AUDIT.md; read-out: tools/pt2main_summary.py.
Self-contained on Burst: every comparison has its own none cell on the same hardware and pinned code; no seed pairing
with Torch cells. helix3 target, lambda 5, prior layers 1/3 and 2/3 of the depth, seeds 0-3, every model saved:
  pt2b     53M, 3e8 tokens (pretrain_data):     none at lr 1e-3, 2e-3, 4e-3, 8e-3; win-ctx at 1e-3, 2e-3, 4e-3
                                                                                                              tasks 350-353
  pt2long  53M, 1.2e9 tokens (pretrain_data_3B): none and win-ctx at 5e-4, 1e-3, 2e-3; win-text and win-shuf-ctx
                                                (helix3_shuf) at 1e-3                                         tasks 354-357
  pt2s190  190M, 3e8 tokens (pretrain_data):    none at 6e-4, 1.2e-3, 2.4e-3; win-ctx at 6e-4, 1.2e-3, 2.4e-3;
                                                win-text at 6e-4                                             tasks 358-361
  (2026-10-16, user: every arm at its own best lr, maximise parallelism; re-audit: the added cells belong inside their
  block so the long runs start before pt2s190 and a budget stop cannot strand an lr grid half-run)
Task name = <NNN>-<tag>-s<seed>, NNN = block base + seed (the 6 cells of a block share it). The tick submits in name order
and holds at most 12 GPU tasks: the cheap, decisive block comes first (pt2b), then the longest runs (pt2long, so the batch
ends sooner), then pt2s190; inside a block every cell's seed 0 comes before any seed 1 (the first wave shows every cell).
362-399: EXTRA below (re-runs of a failed task and conditional cells), rendered like the base tasks.
`# time:` = 2 x the Torch L40S time of the same configuration, rounded up to the half hour (L40S_H; the A100-40GB is
untimed, and the tick never retries a TIMEOUT).
Every task (header '# after: 331-env311 332-data'): data sha256 = Torch's; --dry_inputs through burst/pin_run.sh at PIN
with the model line (size, gradient checkpointing as the trainer decides), the prior sites and the sha256 of the decoded
rows = the value computed on Torch with the same pinned code and sha-identical data (DRY below; exit 3 otherwise, before
any training); training with --save_model --ckpt_every 2000 (a tick retry keeps out/ and resumes); offline_lm.py on the
saved model; intent checks of the run JSON -> $OUT/check.{json,txt} (exit 4 on a fail); nvidia-smi every 60 s -> gpu.csv.
Idempotent: an existing task file is never overwritten (a changed rendering is reported and kept); a task name that has
status/<task>.json but no task file is refused (exit 2, nothing written): it ran already, a re-run needs a new name.
    python tools/make_pt2main_tasks.py [--check] [--show]   (--check: render + assert, write nothing; --show: print the
                                                             trainer arguments of seed 0 of every cell)
"""
import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIN, PIN_SHA = "64abc19", "a3bce3ab63c1"   # commit; sha256[:12] of its tools/pretrain_range.py (= code_sha of every run)
SEEDS = range(4)
BS, SEQ, EVAL_EVERY = 32, 512, 250   # trainer defaults, not passed (asserted on the run JSON)
# sha256 of train.bin / val.bin = the Torch files (tasks/332-data.sh builds them on Burst and checks the same values)
DATA = {"pretrain_data": ("4de2ed19093c37b302b6a1ced56136e6a8b805e6d6b48057a77316811704ff11",
                          "d6f744a99b988ed6928fb588356cc8fe3d61bcf348b8d06fc0eece25c624a4c8"),
        "pretrain_data_3B": ("5e456d775670562dbdd96680eed93d36de2ef0d299bff32a5cc975943828d485",
                             "4b7c27cfd8613cac30ecec1f634cce023bb895611af0b14ec4c4cb02866ca609")}
W = "--schedule window --window 0.05,0.4"
ARM = {"none": "--target none",
       "win-ctx": f"--target helix3 {W} --site ctx",
       "win-text": f"--target helix3 {W} --site text",
       "win-shuf-ctx": f"--target helix3_shuf {W} --site ctx"}
# 2026-10-17 (user): lambda sweep at the 2% main setting (lambda was only swept at 7% with the gate, F82: 1 >= 5 > 20)
LAMS = ("0.3", "1", "2")
for _l in LAMS:
    ARM[f"win-ctx-lam{_l}"] = f"--target helix3 {W} --site ctx --lam {_l}"
ARM_DESC = {"none": "no prior",
            "win-ctx": "helix3 prior on the operands in the lines (ctx), window (0.05, 0.4)",
            "win-text": "helix3 prior on numbers 0..999 in the row's FineWeb text (text), window (0.05, 0.4)",
            "win-shuf-ctx": "helix3_shuf prior (rows of the target table permuted) on the operands (ctx), window (0.05, 0.4)"}
for _l in LAMS:
    ARM_DESC[f"win-ctx-lam{_l}"] = f"helix3 prior on the operands in the lines (ctx), window (0.05, 0.4), lambda {_l}"
# intent of every arm, asserted on the rendered ARGS (argparse keeps the LAST of a repeated flag: none allowed)
WANT = {"none": {"--target": "none", "--schedule": None, "--window": None, "--site": None},
        "win-ctx": {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx"},
        "win-text": {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "text"},
        "win-shuf-ctx": {"--target": "helix3_shuf", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx"}}
for _l in LAMS:
    WANT[f"win-ctx-lam{_l}"] = {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx", "--lam": _l}
# size: model flags, (d_model, n_layers, n_heads, d_ff), parameters and gradient checkpointing (auto: on above 150M) as
# the trainer reports them (Torch runs pt999 2pct-none / ptsize999 190m-2pct-none), the --dry_inputs model line
SIZES = {"53m": {"flags": "", "dims": (384, 6, 6, 1536), "n_params": 52672128, "grad_ckpt": False},
         "190m": {"flags": "--d_model 768 --n_layers 12 --n_heads 12 --d_ff 3072", "dims": (768, 12, 12, 3072),
                  "n_params": 190297344, "grad_ckpt": True}}
for _s in SIZES.values():
    _s["model_line"] = f"model: {_s['n_params'] / 1e6:.1f}M parameters; gradient checkpointing {'on' if _s['grad_ckpt'] else 'off'}"
# blocks in submission order: (base number, block, size, data, tokens, planning A100-40GB hours per run, time limit, cells)
# planning rates (Batch 1 plan): 53M 2.5 A100-h per 1e9 tokens, 190M 2.0 A100-h per 3e8 tokens
BLOCKS = [(350, "pt2b", "53m", "pretrain_data", "3e8", 0.75, "02:00:00",
           [("none", "1e-3"), ("none", "2e-3"), ("none", "4e-3"), ("none", "8e-3"), ("win-ctx", "1e-3"), ("win-ctx", "2e-3"),
            ("win-ctx", "4e-3")]),
          (354, "pt2long", "53m", "pretrain_data_3B", "1.2e9", 2.5 * 1.2, "10:00:00",
           [("none", "5e-4"), ("none", "1e-3"), ("none", "2e-3"), ("win-ctx", "5e-4"), ("win-ctx", "1e-3"), ("win-ctx", "2e-3"),
            ("win-text", "1e-3"), ("win-shuf-ctx", "1e-3")]),
          (358, "pt2s190", "190m", "pretrain_data", "3e8", 2.0, "05:30:00",
           [("none", "6e-4"), ("none", "1.2e-3"), ("none", "2.4e-3"), ("win-ctx", "6e-4"), ("win-ctx", "1.2e-3"),
            ("win-ctx", "2.4e-3"), ("win-text", "6e-4")])]
# Torch L40S hours per run (training loop, the last eval's 'sec'): pt999 2pct-none s0-3 0.952-0.961, 2pct-gate-ctx s0-1
# 0.962-0.965, 2pct-gate-text s0 0.969, ptgw lr5e-4-W s1 0.963; ptsize999 190m-2pct-none s0-3 2.598-2.615, 190m-2pct-gate-ctx
# s0 2.616; pt2long = 4 x pt2b (no 1.2e9 run exists). The limit is 2x this, rounded up to the half hour (asserted in
# tasks()): the A100-40GB is untimed (a 53M run may be bound by the Burst node's CPUs rather than the GPU), and a TIMEOUT
# is never retried (tick.sh), so a run that hits it loses all its hours. Billing is by elapsed time; the guard reserves
# the limit while a task is in flight (2 / 6 / 8 h). xsite53's measured sec/step may lower the limits before submission.
L40S_H = {"pt2b": 0.97, "pt2s190": 2.62, "pt2long": 4 * 0.97}
# limit factor over L40S_H: 2, except pt2long 2.5 (re-audit 2026-10-16: a TIMEOUT is never retried and the A100-40GB is
# untimed; 32 x 10 h is the most at stake, and the guard reserves the limit only while a task is in flight)
LIM_X = {"pt2b": 2, "pt2s190": 2, "pt2long": 2.5}
# Re-runs and conditional cells: (task number 362-399, block, arm, lr, seed), rendered exactly like the 72 with that
# block's constants, intent and checks. A task runs once (tick.sh), so a failed task (TIMEOUT, OOM, a lost pin) is re-run
# as the same cell and seed under a new number; the conditional cells pre-registered in the AUDIT (an lr beyond an open
# bracket; confirm seeds 4-7 of none@best and win-ctx@best) go here too. tools/pt2main_summary.py reads every task of
# tasks(): a re-run supersedes an unfinished run of its cell and seed, an extra lr joins its arm's grid, seeds 4-7 are
# the confirm seeds. Seeds >= 4 need their DRY row hash first (computed on Torch as for seeds 0-3).
# 2026-10-16 (user: every arm, the prior too, at its own best lr; maximise parallelism), decided before any result: the
# long budget's best lr is expected lower than at 3e8 (the optimum falls with the horizon), so pt2long gets 5e-4 for none
# and win-ctx; pt2s190 win-ctx gets 2.4e-3 to bracket the prior like none.
# 2026-10-17, pre-registered open-bracket rule (report/audits/pt2main_AUDIT.md), fired on the returned pt2b cells before
# any best-lr verdict: none's area still rises at the top of its grid (8e-3: 23.7, n=3, vs 4e-3: 14.7) -> none@1.6e-2;
# win-ctx is best at the bottom of its grid (1e-3: 70.2 vs 2e-3: 69.8) -> win-ctx@5e-4.
EXTRA = ([(362 + s, "pt2b", "none", "1.6e-2", s) for s in range(4)]
         + [(362 + s, "pt2b", "win-ctx", "5e-4", s) for s in range(4)]
         # 2026-10-17, the same rule fired on the returned pt2long / pt2s190 cells (before any best-lr verdict was
         # written): none's area is highest at the top of its grid (pt2long 2e-3: 69.2 > 1e-3: 66.9; pt2s190 2.4e-3:
         # 22.2 > 1.2e-3: 19.9) -> one more doubling for none, seeds 0-3
         + [(366 + s, "pt2long", "none", "4e-3", s) for s in range(4)]
         + [(366 + s, "pt2s190", "none", "4.8e-3", s) for s in range(4)]
         # confirm seeds 4-7 (user 2026-10-17: confirm the headline cells now, not after the rainbow), pt2b: none's bracket is
         # closed (best 8e-3) and win-ctx's best is 2e-3 (tie with 1e-3 broken on offline bpb) -> both at their best lr
         + [(370 + s - 4, "pt2b", arm, lr, s) for s in range(4, 8) for arm, lr in (("none", "8e-3"), ("win-ctx", "2e-3"))]
         # lambda sweep (user 2026-10-17): win-ctx at its best lr 2e-3 with lambda 0.3 / 1 / 2, seeds 0-3, paired by seed with
         # pt2b-lr2e-3-win-ctx (lambda 5) and pt2b-lr8e-3-none on the same hardware
         + [(374 + s, "pt2b", f"win-ctx-lam{l}", "2e-3", s) for s in range(4) for l in LAMS])
BLOCK_DESC = {"pt2b": "53M lr bracket at the main budget: 3e8 tokens = 18310 steps of 32 x 512 on pretrain_data (4e8 tokens)",
              "pt2s190": "190M (d768 x 12, gradient checkpointing on) at 3e8 tokens = 18310 steps of 32 x 512 on pretrain_data",
              "pt2long": "53M long budget: 1.2e9 tokens = 73242 steps of 32 x 512 on pretrain_data_3B (3e9 tokens; its val.bin "
                         "differs from pretrain_data's, so LM and eval-context numbers compare only within pt2long)"}
# step-0 LM loss of the Torch runs with the same seed and the same first batch (2% rows on pretrain_data; the loss before
# the first update depends on neither lr nor arm): pt999 2pct-none (53M) and ptsize999 190m-2pct-none (190M), seeds 0-3
LM0 = {("53m", "pretrain_data"): (11.616473197937012, 11.587233543395996, 11.565471649169922, 11.602105140686035,
                                  # seeds 4-7: pt999 2pct-none (Torch), added 2026-10-17 for the confirm seeds
                                  11.594945907592773, 11.568119049072266, 11.584515571594238, 11.593491554260254),
       ("190m", "pretrain_data"): (11.645764350891113, 11.675423622131348, 11.666858673095703, 11.6434907913208)}
# sha256 of the 9 selected --dry_inputs lines (the grep below) per (data, size, rows, seed); rows = the prior-site marking
# (ctx: operands; text: numbers in the text; 'none' marks the operands like ctx). Computed on Torch 2026-10-16 by running
# each task's step 1 through burst/pin_run.sh at PIN with LLMADD_ROOT = a fake root over the Torch data (sha-identical);
# the 53M / 190M pretrain_data ctx values and the 53M text values equal the Torch logs of pt999 2pct-none / 2pct-gate-text
# and ptsize999 190m-2pct-none (report/audits/pt2main_AUDIT.md, verification).
DRY = {("pretrain_data", "190m", "ctx", 0): "033a837b81848bca94cb3f2892a804f0e4e9fdfcc9aec87593492de5cbf54101",
       ("pretrain_data", "190m", "ctx", 1): "6eb60f99e5648524be6df2def2247f13c68742ae22852548e86fce953264da4f",
       ("pretrain_data", "190m", "ctx", 2): "af9f0b40f7334548cdb13dc8fbd0c6f198edd1bdec4cccc1f0e505a7a4b3e137",
       ("pretrain_data", "190m", "ctx", 3): "99f86683e5b961aca659f0474bff212033c1c9162bcf9ec206a5de3c9806d805",
       ("pretrain_data", "190m", "text", 0): "871c717bcdc7d2b321ed3f4695ea6dd224060d4e6c8232eec0e08a03b98e81b3",
       ("pretrain_data", "190m", "text", 1): "f9c46d49a543681701047b3c004655282069b6ed5c1658fe45b6a26628fdb59d",
       ("pretrain_data", "190m", "text", 2): "f96c4980ca7bb7e7f247fba0cf87a44df72c53961db09bd7913b189016379869",
       ("pretrain_data", "190m", "text", 3): "a40f17fbfda0dc0ea2265f863e2cd7c763a31c35361f30b774b28c706a87937c",
       ("pretrain_data", "53m", "ctx", 0): "379354e22b42e615e89aee1398d72e768837c900d1db089c60f88e8649007346",
       ("pretrain_data", "53m", "ctx", 1): "67e619ccb5dc8449e4b2c19d3a79070d529fae68fe61cd663ab478c4974c8504",
       ("pretrain_data", "53m", "ctx", 2): "01db7786312a143adfae91ce1d9854683bdd7905cbcc500ca57ee765a8cb48c7",
       ("pretrain_data", "53m", "ctx", 3): "df75d83482726619cc70c114aae0c20b5b6b1de61c7772074801ff485865e1c6",
       ("pretrain_data_3B", "53m", "ctx", 0): "58a44ccbc47ab0fdd818a1daf439fd97e6a4197ca815b1fc4856b6a4f8ece4e8",
       ("pretrain_data_3B", "53m", "ctx", 1): "7a4e101e6000bbe02debca27d792fd2c8974e9d842eaddd08d635caf38dfd0ed",
       ("pretrain_data_3B", "53m", "ctx", 2): "197c75b23aeea72587e258d1c10bc522f657f79eca36381dcca67dc35d5afbae",
       ("pretrain_data_3B", "53m", "ctx", 3): "7371c51c376e08f5857026e28979fe1bcd839588a9d4f5fece44a3b592d25c40",
       ("pretrain_data_3B", "53m", "text", 0): "ecf98276f6bb4621ad1d224f931482385c78d621a57a600b2268db3bfa584436",
       ("pretrain_data_3B", "53m", "text", 1): "c3a447c62757ecdb3eb8cc66f03ca3d16c481b629172166dd0d7055b3a961232",
       ("pretrain_data_3B", "53m", "text", 2): "34277b6603e9f4ba58281429b81d3b99ccafe83f2c0ec86017944f674286727a",
       ("pretrain_data_3B", "53m", "text", 3): "c43aabe01bc12b8ec13ad2dff1fee596bbb4e4d09358c80a2555dd3e9e5bbcfe",
       # confirm seeds 4-7 (2026-10-17): computed the same way, the method reproducing the known seed-0 value of each key
       ("pretrain_data", "53m", "ctx", 4): "9df8674d52970374eb4305ca588ec7231419c56738b10543a7aa724f1461687f",
       ("pretrain_data", "53m", "ctx", 5): "96caf9f331745f38c3f635afc57c5b686f8eadd89bdc7f329efe34dc32af56c2",
       ("pretrain_data", "53m", "ctx", 6): "65a746548b21f9dbd0a570149468b75a8e07026f75393d341c75e8d8c91321cf",
       ("pretrain_data", "53m", "ctx", 7): "3fdd52cd798ca08c817105451892472339729cb2438d1b421cd09bd3a8287669",
       ("pretrain_data", "190m", "ctx", 4): "30845923a6629ece8c5876fc11f9d0032b452f706e1d0bf4bcb578a0493d9b80",
       ("pretrain_data", "190m", "ctx", 5): "872c729dcf5660773ffc81a88f309a94b38691c99440ee147830d23c53966cf9",
       ("pretrain_data", "190m", "ctx", 6): "e4f02d7ff355b3016801f3dda7c8bb54bd72e1823f7bf14a060a8ab473e98660",
       ("pretrain_data", "190m", "ctx", 7): "567debef56131fa65c68dd1393eb2b6e6b773a4752c5c112cda0919a319d87cd",
       ("pretrain_data_3B", "53m", "ctx", 4): "1bbf848398a0184ad72a216c73a0b9567d2d9453e077b5b9696bc1db7beb9457",
       ("pretrain_data_3B", "53m", "ctx", 5): "3aa3ca8357fa2bee57d626d3ea58e1e47bfa397feb94b0bd49c63fee2ad7fbdd",
       ("pretrain_data_3B", "53m", "ctx", 6): "bce4b0e5177a51ca10e4c266e9dd5617a83d3ffe0f10780adf0438c84caebee2",
       ("pretrain_data_3B", "53m", "ctx", 7): "1b33b8d3f4d4378d242e15fb27658a034c49fcd809a34f4e21d67ed98da10c59"}
DRY_GREP = "^(line:|held_b|ood_b|\\{'held_b11'|model:|train row)"


def flags(args):
    """{--flag: value or True} of an ARGS string; asserts that no flag is given twice."""
    t, d = args.split(), {}
    for i, w in enumerate(t):
        if w.startswith("--"):
            assert w not in d, f"{w} given twice: {args}"
            d[w] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True
    return d


def hours(tl):
    """HH:MM:SS -> hours"""
    h, m, sec = (int(v) for v in tl.split(":"))
    return h + m / 60 + sec / 3600


def window_events(steps, window="0.05,0.4"):
    """The trainer's window rule (prior on iff w0 * steps <= s < w1 * steps, floats as in pretrain_range.prior_on):
    prior_events [[first on step, 1], [first off step, 0]] and the fraction of steps with the prior on."""
    w0, w1 = (float(v) for v in window.split(","))
    on = [s for s in range(steps) if w0 * steps <= s < w1 * steps]
    return [[on[0], 1], [on[-1] + 1, 0]], len(on) / steps


CHECK_PY = r'''import csv, json, math, statistics, sys
from pathlib import Path
run, offl, gpu, pinned, data, out_json = sys.argv[1:7]
I = json.loads(sys.argv[7])
B = ("b11", "b12", "b13", "b22", "b23", "b33")
mean = statistics.fmean
checks = []
def chk(name, ok, value):
    checks.append({"check": name, "ok": bool(ok), "value": value})
r = json.loads(Path(run).read_text()); c = r["curve"]; a = r["args"]
want = dict(I["args"], data=data, out=str(Path(run).parent))
diff = {k: [a.get(k), v] for k, v in want.items() if a.get(k) != v}
chk(f"run args = intent ({len(want)} fields)", not diff, diff)
chk("code_sha " + I["code_sha"], r.get("code_sha") == I["code_sha"], r.get("code_sha"))
keys = ("steps", "n_params", "grad_ckpt_on", "geo_layers", "read_layers", "lm_chunk", "eval_every_steps")
got = {k: r.get(k) for k in keys}
chk("run facts = intent " + json.dumps({k: I[k] for k in keys}), all(got[k] == I[k] for k in keys), got)
st = [e["step"] for e in c]
chk(f"finished: {I['evals']} evals, increasing, the last at step {I['steps'] - 1}",
    len(c) == I["evals"] and st == sorted(set(st)) and st[-1] == I["steps"] - 1, [len(c), st[-1]])
# a diverged run (lr 4e-3 may spike) is data, not a failed check: its area counts against its lr (AUDIT rules 1 and 6).
# Onset = the first eval with a non-finite LM loss, val loss, prior loss or prior gradient; the prior checks look only at
# the evals before it, the offline |orig - logged| check is skipped (NaN weights).
fin = lambda v: v is None or math.isfinite(v)
bad = [e["step"] for e in c if not (fin(e["lm"]) and fin(e["val_loss"]) and fin(e["geo"]) and fin(e.get("grad_prior")))]
diverged = bool(bad)
if I["events"]:
    used, on_frac = c[-1]["prior_used_frac"], c[-1]["prior_on_frac"]
    chk(f"prior_events == {I['events']}", r.get("prior_events") == I["events"], r.get("prior_events"))
    chk(f"prior_on_frac == {I['on_frac']:.6f}", abs(on_frac - I["on_frac"]) < 1e-9, on_frac)
    # evals where the prior acted (on and usable): the trainer logs geo and the prior's gradient norm (grad_prior) there
    on = [e for e in c if e["prior_on"] and e["geo"] is not None and (not bad or e["step"] < bad[0])]
    chk("geo > 0 and 0 < grad_prior < inf at every eval where the prior acted (before a divergence)",
        (bool(on) or diverged) and all(e["geo"] > 0 and 0 < e.get("grad_prior", 0) < math.inf for e in on), len(on))
    if I["site"] == "ctx":   # the operands are in every batch: the prior acts on every step it is on
        chk("prior_used_frac == prior_on_frac (ctx)", abs(used - on_frac) < 1e-9, [used, on_frac])
    else:                    # text: skipped on batches with < 16 numbers (Torch 2pct-win-text: never)
        chk("0.9 x prior_on_frac <= prior_used_frac <= prior_on_frac (text)", 0.9 * on_frac <= used <= on_frac + 1e-9, [used, on_frac])
else:
    chk("prior never on", c[-1]["prior_on_frac"] == 0 and r.get("prior_events") == [], c[-1]["prior_on_frac"])
if I["lm0_ref"] is not None:
    chk("step-0 lm within 1e-3 of the same-seed Torch run (same init and first batch)",
        abs(c[0]["lm"] - I["lm0_ref"]) <= 1e-3, c[0]["lm"] - I["lm0_ref"])
o = json.loads(Path(offl).read_text()).get(str(Path(run).resolve())) if Path(offl).exists() else None
chk("offline_lm on the full val, |orig_minus_logged| < 0.002 (skipped if the run diverged)",
    o is not None and o["max_windows"] == 0 and (diverged or abs(o["orig_minus_logged"]) < 0.002), o and o["orig_minus_logged"])
held = lambda e, bs=B: mean([e[f"held_{b}"] for b in bs])
per = r["args"]["bs"] * r["args"]["seq"]
t90 = None
for p, e in zip([None] + c[:-1], c):
    if held(e) >= 0.9:
        t90 = e["step"] * per / 1e6 if p is None else (p["step"] + (0.9 - held(p)) / (held(e) - held(p)) * (e["step"] - p["step"])) * per / 1e6
        break
rate = [(q["sec"] - p["sec"]) / (q["step"] - p["step"]) for p, q in zip(c, c[1:]) if q["sec"] > p["sec"] and q["step"] > p["step"]]
util, mem = [], []
for row in csv.reader(open(gpu)) if Path(gpu).exists() else []:
    try:
        util.append(int(row[2].split()[0])); mem.append(int(row[4].split()[0]))
    except (IndexError, ValueError):
        pass
info = {"tag": a["tag"], "seed": a["seed"], "lr": a["lr"], "area": round(100 * mean([held(e) for e in c]), 2),
        "area_no_b11": round(100 * mean([held(e, B[1:]) for e in c]), 2), "b11_area": round(100 * mean([e["held_b11"] for e in c]), 2),
        "b33_area": round(100 * mean([e["held_b33"] for e in c]), 2), "t90_M": t90, "budget_M": r["steps"] * per / 1e6,
        "lm0": c[0]["lm"], "final_val_bpb": c[-1]["val_bits_per_byte"], "bpb_full": o and o["bpb_full"], "diverged": diverged,
        "diverged_at_step": bad[0] if bad else None, "prior_used_frac": c[-1]["prior_used_frac"], "cka_ops_helix3": c[-1].get("cka_ops_helix3"),
        "sec_per_step_median": statistics.median(rate) if rate else None,
        "projected_run_h": statistics.median(rate) * r["steps"] / 3600 if rate else None,
        "gpu_util_median": statistics.median(util) if util else None, "gpu_mem_max_MiB": max(mem) if mem else None,
        "gpu_samples": len(util), "pinned_calls": len(Path(pinned).read_text().splitlines()) if Path(pinned).exists() else 0}
ok = all(k["ok"] for k in checks)
Path(out_json).write_text(json.dumps({"pass": ok, "checks": checks, "info": info}, indent=1))
for k in checks:
    print(("PASS " if k["ok"] else "FAIL ") + k["check"] + f"  ({k['value']})")
print(json.dumps(info))
print("CHECKS", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 4)
'''


def render(num, block, size, data, tokens, plan_h, tlim, arm, lr, seed):
    """(task name, tag, file text, trainer ARGS with the shell variables) of one (cell, seed) as task number num."""
    tag = f"{block}-lr{lr}-{arm}"
    name = f"{num}-{tag}-s{seed}"
    sz = SIZES[size]
    T = float(tokens)
    steps = int(T // (BS * SEQ))   # as the trainer: int(args.tokens // (args.bs * args.seq))
    target = WANT[arm]["--target"]
    site = WANT[arm]["--site"] or "ctx"   # the trainer's default; 'none' marks the operands in --dry_inputs
    tok_flag = "" if T == 3e8 else f" --tokens {tokens}"   # 3e8 = the trainer default
    size_flag = f" {sz['flags']}" if sz["flags"] else ""
    args = (f"--data $DATA --tokenizer $TOK --tokenization whole --out $OUT/$TAG $ARM --arith_lines 1{size_flag}{tok_flag} "
            f"--lr {lr} --seed $SEED --tag _$TAG --save_model --ckpt_every 2000")
    # intent asserts on the rendered arguments
    f = flags(args.replace("$ARM", ARM[arm]).replace("$SEED", str(seed)).replace("$TAG", tag))
    d, nl, nh, dff = sz["dims"]
    want = {"--data": "$DATA", "--tokenizer": "$TOK", "--tokenization": "whole", "--out": f"$OUT/{tag}", "--arith_lines": "1",
            "--lr": lr, "--seed": str(seed), "--tag": f"_{tag}", "--save_model": True, "--ckpt_every": "2000",
            "--tokens": None if T == 3e8 else tokens, "--lam": None, "--geo_layers": None, "--bs": None,
            "--seq": None, "--eval_every": None, "--n_eval": None, "--gate_thr": None, "--gate_start": None,
            "--gate_hyst": None, "--untie": None, "--grad_ckpt": None, "--lm_chunk_tokens": None, "--stop_after": None,
            "--dry_inputs": None, **WANT[arm],
            **({"--d_model": str(d), "--n_layers": str(nl), "--n_heads": str(nh), "--d_ff": str(dff)} if sz["flags"] else
               {"--d_model": None, "--n_layers": None, "--n_heads": None, "--d_ff": None})}
    for k, v in want.items():
        assert f.get(k) == v, (name, k, f.get(k), v)
    assert set(f) <= {k for k, v in want.items() if v is not None}, (name, sorted(f))
    assert float(f.get("--tokens", 3e8)) == T and steps == (18310 if T == 3e8 else 73242), (name, steps)
    assert LIM_X[block] * L40S_H[block] <= hours(tlim) < LIM_X[block] * L40S_H[block] + 0.5 and hours(tlim) >= 1.7 * plan_h, (name, tlim)
    events, on_frac = window_events(steps) if arm != "none" else ([], 0.0)
    if arm != "none":
        assert events == ([[916, 1], [7324, 0]] if steps == 18310 else [[3663, 1], [29297, 0]]), events
    lm0 = LM0.get((size, data))
    intent = {"code_sha": PIN_SHA, "steps": steps, "n_params": sz["n_params"], "grad_ckpt_on": sz["grad_ckpt"],
              "geo_layers": [nl // 3, 2 * nl // 3], "read_layers": [nl // 3, 2 * nl // 3], "lm_chunk": 0,
              "eval_every_steps": EVAL_EVERY, "evals": len(range(0, steps, EVAL_EVERY)) + ((steps - 1) % EVAL_EVERY != 0),
              "events": events, "on_frac": on_frac, "site": site, "lm0_ref": lm0[seed] if lm0 and seed < len(lm0) else None,
              "args": {"tokenization": "whole", "target": target, "site": site, "schedule": WANT[arm]["--schedule"] or "whole",
                       "window": "0.05,0.4", "arith_lines": 1, "lr": float(lr), "lam": float(WANT[arm].get("--lam") or 5.0), "tokens": T, "bs": BS,
                       "seq": SEQ, "d_model": d, "n_layers": nl, "n_heads": nh, "d_ff": dff, "seed": seed, "tag": f"_{tag}",
                       "save_model": True, "ckpt_every": 2000, "geo_layers": None, "eval_every": EVAL_EVERY, "n_eval": 150,
                       "grad_ckpt": "auto", "lm_chunk_tokens": -1, "untie": False, "gate_thr": 0.5, "gate_start": 0.0,
                       "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256, "stop_after": -1, "dry_inputs": False}}
    ij = json.dumps(intent, separators=(",", ":"))
    assert "'" not in ij
    assert (data, size, site, seed) in DRY, f"{name}: no DRY row hash for {(data, size, site, seed)}; compute it on Torch first"
    dry = DRY[(data, size, site, seed)]
    if site == "ctx":
        site_assert = ("""[ "$(grep -c '^train row [012] (len 512, 2 prior sites at ' "$OUT/dry_inputs.txt")" = 3 ] """
                       """|| { echo "## not 1 arithmetic line (2 operand sites) per row"; exit 3; }""")
    else:
        site_assert = ("""grep -q '^text numbers per batch over 50 batches: ' "$OUT/dry_inputs.txt" """
                       """|| { echo "## no text-site statistics in --dry_inputs"; exit 3; }""")
    if target == "none":
        target_assert = """! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a prior target in a none run"; exit 3; }"""
    else:
        target_assert = """grep -qxF 'target features (1000, 10)' "$OUT/dry_inputs.txt" || { echo "## not the helix3 table (1000 x 10: cos 2, cos/sin 5 10 100 1000, x / 1000)"; exit 3; }"""
    train_sha, val_sha = DATA[data]
    runfile = f"run_whole_{target}_seed{seed}_{tag}.json"
    about = textwrap.fill(f"{BLOCK_DESC[block]}. Cell {tag}: {ARM_DESC[arm]}, lr {lr}, seed {seed}; 2% arithmetic (1 line "
                          f"per row), lambda 5.", 120, initial_indent="# ", subsequent_indent="# ")
    lm0_note = (f"step-0 lm vs the same-seed Torch run ({'pt999 2pct-none' if size == '53m' else 'ptsize999 190m-2pct-none'}, "
                f"same init and first batch)" if intent["lm0_ref"] is not None else
                "no same-seed Torch run on this data: step-0 lm only recorded")
    text = f"""# kind: gpu
# time: {tlim}
# cpus: 8
# after: 331-env311 332-data
# pt2main {block} (2026-10-16 user decisions: 2% arithmetic = the main setting, the 5% window = the main recipe, every arm
# gets its own best lr; intent spec and pre-registered rules report/audits/pt2main_AUDIT.md). Generated by
# tools/make_pt2main_tasks.py: change the generator, not this file.
{about}
# Self-contained on Burst: compared only with this batch's cells (same hardware, same pinned code), never paired with a
# Torch cell. The tag is unique so a summary never merges sites.
# Trainer pinned at {PIN} (tools/pretrain_range.py sha256 {PIN_SHA}...) through burst/pin_run.sh; env llmadd_env311.
# 1. inputs: data sha256 = Torch's; --dry_inputs: model line, prior sites, target table, and the rows' sha256 = the value
#    computed on Torch with the same pinned code (exit 3 otherwise, before any training)
# 2. training with --save_model --ckpt_every 2000 (a retry keeps $OUT and resumes from the checkpoint)
# 3. offline_lm.py on the saved model (full val.bin of {data})
# 4. intent checks of the run JSON -> $OUT/check.{{json,txt}} (exit 4 on a fail;
#    {lm0_note})
# nvidia-smi utilisation / memory every 60 s -> $OUT/gpu.csv (planning: {plan_h:.2f} A100-h; limit {tlim}).
set -euo pipefail
SEED={seed}; TAG={tag}; ARM="{ARM[arm]}"
ROOT=${{LLMADD_ROOT:-/scratch/$USER}}; DATA=$ROOT/llmadd_data/{data}; TOK=$ROOT/llmadd_data/olmo2_tok; PIN={PIN}
[ -s "$OUT/gpu.csv" ] || echo "timestamp, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]" > "$OUT/gpu.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader -l 60 >> "$OUT/gpu.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
{{ date -u +%FT%TZ; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "cpus $(nproc)"; }} >> "$OUT/node.txt" 2>&1 || true

echo "## 1. inputs $(date -u +%FT%TZ)"
(cd "$DATA" && sha256sum -c -) <<'SHA' || exit 3
{train_sha}  train.bin
{val_sha}  val.bin
SHA
ARGS="{args}"
bash burst/pin_run.sh $PIN pretrain_range.py $ARGS --dry_inputs > "$OUT/dry_inputs.txt" 2>&1 || {{ cat "$OUT/dry_inputs.txt"; exit 3; }}
cat "$OUT/dry_inputs.txt"
grep -qxF '{sz["model_line"]}' "$OUT/dry_inputs.txt" || {{ echo "## not the intended model size / gradient checkpointing state"; exit 3; }}
{site_assert}
{target_assert}
got=$(set +o pipefail; grep -E "{DRY_GREP}" "$OUT/dry_inputs.txt" | head -n 9 | sha256sum | cut -d' ' -f1)
[ "$got" = {dry} ] || {{ echo "## dry_inputs differ from the rows computed on Torch (sha256 $got)"; exit 3; }}
grep -q " {PIN_SHA}[0-9a-f]* tools/pretrain_range.py " "$OUT/PINNED.txt" || {{ echo "## the pinned trainer is not {PIN_SHA}"; exit 3; }}

echo "## 2. training $(date -u +%FT%TZ)"
bash burst/pin_run.sh $PIN pretrain_range.py $ARGS
echo "## 3. offline LM $(date -u +%FT%TZ)"
bash burst/pin_run.sh $PIN offline_lm.py --data "$DATA" --tokenizer "$TOK" --out "$OUT/offline_lm.json" "$OUT/$TAG"
echo "## 4. intent checks $(date -u +%FT%TZ)"
"$ROOT/llmadd_env311/bin/python" - "$OUT/$TAG/{runfile}" "$OUT/offline_lm.json" "$OUT/gpu.csv" "$OUT/PINNED.txt" "$DATA" \\
  "$OUT/check.json" '{ij}' <<'PY' | tee "$OUT/check.txt"
{CHECK_PY}PY
"""
    return name, tag, text, args


def tasks():
    """Every (cell, seed) of BLOCKS in submission order (the base tasks), then EXTRA: [(name, tag, text, args, block, arm, lr, seed)]."""
    out, blocks = [], {b[1]: b for b in BLOCKS}
    for base, block, size, data, tokens, plan_h, tlim, cells in BLOCKS:
        assert len({(a, lr) for a, lr in cells}) == len(cells)
        for seed in SEEDS:
            for arm, lr in cells:
                out.append(render(base + seed, block, size, data, tokens, plan_h, tlim, arm, lr, seed) + (block, arm, lr, seed))
    for num, block, arm, lr, seed in EXTRA:
        assert 362 <= num <= 399 and block in blocks and arm in ARM and float(lr) > 0 and seed in range(8), (num, block, arm, lr, seed)
        _, _, size, data, tokens, plan_h, tlim, _ = blocks[block]
        out.append(render(num, block, size, data, tokens, plan_h, tlim, arm, lr, seed) + (block, arm, lr, seed))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="render and assert only; write nothing")
    ap.add_argument("--show", action="store_true", help="print the trainer arguments of seed 0 of every cell")
    ap.add_argument("--repo", default=str(REPO))
    a = ap.parse_args()
    repo = Path(a.repo)
    T = tasks()
    n_cells = sum(len(b[-1]) for b in BLOCKS)
    n_base = n_cells * len(SEEDS)
    names, base = [t[0] for t in T], T[:n_base]
    assert len(set(names)) == len(names), "task names not unique"
    assert len(base) == n_base == 88 and len({t[1] for t in base}) == n_cells == 22, (len(base), len({t[1] for t in base}))
    nums = [int(t[0][:3]) for t in base]
    assert nums == sorted(nums) and set(nums) == set(range(350, 362)), "task numbers / order"
    assert [t[4] for t in base] == sorted([t[4] for t in base], key=["pt2b", "pt2long", "pt2s190"].index), "block order"
    assert all(362 <= int(t[0][:3]) <= 399 for t in T[n_base:]), "EXTRA numbers"
    if a.show:
        for name, tag, text, args, *_ in T:
            if name.endswith("-s0"):
                arm = text.split('ARM="', 1)[1].split('"', 1)[0]
                print(f"{name}\n    {args.replace('$ARM', arm).replace('$SEED', '0').replace('$TAG', tag)}")
    tdir, sdir = repo / "tasks", repo / "status"
    foreign = sorted(p.name for p in tdir.glob("3[5-9][0-9]*.sh") if p.stem not in names)
    if foreign:
        print("WARNING: task files in 350-399 not made by this generator:", foreign)
    new, same, differ, ran = [], [], [], []
    for name, tag, text, args, *_ in T:
        p = tdir / f"{name}.sh"
        has_status = (sdir / f"{name}.json").exists()
        if p.exists():
            (same if p.read_text() == text else differ).append(name)
            if has_status:
                ran.append(name)
        elif has_status:
            print(f"refused: {name} has status/{name}.json (it ran) but no task file; a re-run needs a new task name; "
                  "nothing written", file=sys.stderr)
            sys.exit(2)
        else:
            new.append((p, text))
    for name in differ:
        print(f"DIFFERS from the current rendering (kept, not overwritten): tasks/{name}.sh")
    if not a.check:
        for p, text in new:
            p.write_text(text)
    print(f"{len(T)} tasks ({len(T) - n_base} EXTRA): {len(new)} {'to write' if a.check else 'written'}, {len(same)} unchanged, {len(differ)} differ "
          f"(kept), {len(ran)} already ran")


if __name__ == "__main__":
    main()
