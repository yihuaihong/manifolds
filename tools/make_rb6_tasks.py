"""Burst task files for the rainbow6 PILOT (rb6: tools/pretrain_range.py Rev B --domains, amendments A1 + A2).
User decisions 2026-10-17: six manifolds pretrained at once at 53M (numbers 0..999 helix3, months C12, weekdays C7, clock
hours C24, S4 cube rotations, dimension lattice {-1,0,1}^3); every arm sees the SAME data, arms differ only in which
priors are on; data option 1 = numbers at the 2% main setting (--arith_lines 1; the numbers line is the first line after
the text, so the numbers cells compare directly with the pt999 / pt2b 2pct cells) + K domain lines per row after it;
fixed simple training templates, held-out-format templates in evaluation only; pilot = none vs all6 with the current best
setup (helix3 + the five domain priors on their item tokens, ctx site, window (0.05, 0.4), lambda 5 PER ACTIVE PRIOR =
--lam_rule each, lr 1e-3), K = 1, 2 domain lines per row, seeds 0, 1. Intent spec + pre-registered pilot rule:
runs/rb6/AUDIT.md (Torch; amendments A1, A2); read-out: tools/rainbow6_summary.py.
Tasks 400-407 (408-409 reserved for re-runs): NNN = 400 + 4 * seed + 2 * (K - 1) + arm (none 0, all6 1); name
NNN-rb6pb-53m-2pct-k<K>-<arm>-s<seed>. Tag rb6pb-... (pb = pilot on Burst): never merged with Torch's rb6p-... runs.
`# time:` = 2 x the planned A100-40GB hours, rounded up to the half hour: 0.72 A100-h measured for a 53M 3e8-token 2%
run on Burst (pt2b, sec/step x steps, 12 runs 0.714-0.723 h) x 1.2 for the domain + held-out-format evals and the
per-eval gradient logging (FLOP estimate: +13% format evals, +4% domain evals, +2% gradients) = 0.86 h -> 02:00:00.
Every task (header '# after: 331-env311 332-data'): data sha256 = Torch's; --dry_inputs through burst/pin_run.sh at PIN
with the model line, the prior target table (all6) / none, the numbers line first in every row, and the sha256 of the
selected decoded lines (DRY_GREP: numbers and domain lines, eval rows incl. every held-out format, training rows per slot,
token / first-line / slot shares, content scan, nonce counts) = the value computed on Torch for this (K, seed) with the
same code and sha-identical data (exit 3 otherwise, before any training); training with --save_model --ckpt_every 2000
(a tick retry keeps $OUT and resumes); offline_lm.py on the saved model; intent checks of the run JSON ->
$OUT/check.{json,txt} (exit 4 on a fail; incl. per-domain prior on / applied / item-state CKA rising for all6, prior
never on for none, step-0 lm and data hash = Torch's, the domain splits = Torch's); nvidia-smi every 60 s -> gpu.csv.
PIN is empty until Rev B (+ A2) is committed: then nothing is written (--check / --show still render and assert).
Idempotent as make_pt2main_tasks.py: an existing task file is never overwritten; a task name with status/ but no file is
refused (exit 2).
    python tools/make_rb6_tasks.py [--check] [--show]
"""
import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIN = ""                      # commit of Rev B + A2 (set after the commit; the tasks run tools/ at this commit)
PIN_SHA = "cc984148b15a"      # sha256[:12] of its tools/pretrain_range.py (= code_sha of every run)
TOY_SHA = "6248613ac6d2"      # sha256[:12] of its tools/toy_struct.py (imported by the trainer; dep_sha of every run)
SEEDS, KS, ARMS = (0, 1), (1, 2), ("none", "all6")
BS, SEQ, EVAL_EVERY, STEPS = 32, 512, 250, 18310   # trainer defaults (53M, 3e8 tokens), not passed
DATA = ("pretrain_data", "4de2ed19093c37b302b6a1ced56136e6a8b805e6d6b48057a77316811704ff11",
        "d6f744a99b988ed6928fb588356cc8fe3d61bcf348b8d06fc0eece25c624a4c8")   # = make_pt2main_tasks.DATA["pretrain_data"]
ALL6 = "numbers,months,weekdays,clock,s4,dim"
EXT = ("months", "weekdays", "clock", "s4", "dim")
W = "--schedule window --window 0.05,0.4 --site ctx --lam_rule each"
ARM = {"none": "--target none", "all6": f"--target helix3 {W} --prior_domains {ALL6}"}
ARM_DESC = {"none": "no prior (the same data)",
            "all6": "helix3 on the operands + the five domain priors on their item tokens (ctx), one window (0.05, 0.4), "
                    "lambda 5 per active prior (lam_rule each)"}
WANT = {"none": {"--target": "none", "--schedule": None, "--window": None, "--site": None, "--lam_rule": None,
                 "--prior_domains": None, "--prior_shuf": None},
        "all6": {"--target": "helix3", "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx", "--lam_rule": "each",
                 "--prior_domains": ALL6, "--prior_shuf": None}}
N_PARAMS, MODEL_LINE = 52672128, "model: 52.7M parameters; gradient checkpointing off"
PLAN_H, LIM_X, TLIM = 0.72 * 1.2, 2, "02:00:00"
N_FMT = {"months": 4, "weekdays": 4, "clock": 4, "s4": 3, "dim": 3}   # held-out-format templates (pretrain_range.PARA)
CKA_RISE = 0.1   # all6: item-state CKA to the true target at the window's last eval - at step 0, every domain (and the
#                  numbers operands' cka_ops_helix3); CPU smoke (tiny model, 40 steps): +0.32..+0.79
# Torch reference values (filled 2026-10-17 from CPU runs of the final code at PIN_SHA on the Torch data, sha-identical to
# Burst's; script and logs in the session scratchpad rainbow6/fix2/): per (K, seed) the sha256 and count of the DRY_GREP
# lines of --dry_inputs (identical for none and all6: checked), the step-0 LM loss (CPU bf16 autocast; calibration: the
# default path's CPU value vs the known GPU value of pt999 2pct-none seed 0) and the data hash after step 0 (data_sha);
# per seed the domain split sizes.
DRY = {(1, 0): ('2106164b29d20df3068596101e6b0ead8ddacdd98674029d6d59ab45d9b82e39', 83),
       (1, 1): ('30058444ee687ae274ec2e35d940911612e5030b8523f40cae9e2f739d842ee1', 83),
       (2, 0): ('fe6a8bba65f7c326e52ac3d5a060c2fabe821489ad935ec7e8db10eb2b632512', 88),
       (2, 1): ('b4aabe5cd8c829e698feea8e5b55024f2baac5799c0813b92c5a680725921800', 88)}
LM0 = {(1, 0): 11.615036964416504, (1, 1): 11.602009773254395, (2, 0): 11.614818572998047, (2, 1): 11.603224754333496}
DSHA0 = {(1, 0): '5d408328037e094e', (1, 1): '3f8eefe500da5bec', (2, 0): '28c9fff8ef37e82f', (2, 1): 'dfac64fb1b4a4f36'}
DOMS = {0: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 482, "n_held": 442, "n_heldE": 151, "n_readout": 337, "n_readoutE": 113, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 203, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 344, "n_held": 342, "n_readout": 120, "readout": "held_no_negtwin_nonid", "n_items": 27}}, 1: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 478, "n_held": 442, "n_heldE": 148, "n_readout": 329, "n_readoutE": 116, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 191, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 344, "n_held": 342, "n_readout": 130, "readout": "held_no_negtwin_nonid", "n_items": 27}}}
DRY_GREP = ("^(line:|held_b|ood_b|\\{'held_b11'|model:|train row |  train line |  fmt p|  eval |  train row, slot |"
            "token share |first line after |slot share |content scan:|nonce tokens:)")
DOM_KEYS = ("n_train", "n_held", "n_heldE", "n_readout", "n_readoutE", "readout", "n_items")


def flags(args):
    """{--flag: value or True} of an ARGS string; asserts that no flag is given twice."""
    t, d = args.split(), {}
    for i, w in enumerate(t):
        if w.startswith("--"):
            assert w not in d, f"{w} given twice: {args}"
            d[w] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True
    return d


def hours(tl):
    h, m, sec = (int(v) for v in tl.split(":"))
    return h + m / 60 + sec / 3600


def window_events(steps, window="0.05,0.4"):
    """The trainer's window rule (prior on iff w0 * steps <= s < w1 * steps): prior_events and the on fraction."""
    w0, w1 = (float(v) for v in window.split(","))
    on = [s for s in range(steps) if w0 * steps <= s < w1 * steps]
    return [[on[0], 1], [on[-1] + 1, 0]], len(on) / steps


CHECK_PY = r'''import csv, json, math, statistics, sys
from pathlib import Path
run, offl, gpu, pinned, data, out_json = sys.argv[1:7]
I = json.loads(sys.argv[7])
B = ("b11", "b12", "b13", "b22", "b23", "b33")
EXT = ("months", "weekdays", "clock", "s4", "dim")
mean = statistics.fmean
checks = []
def chk(name, ok, value):
    checks.append({"check": name, "ok": bool(ok), "value": value})
r = json.loads(Path(run).read_text()); c = r["curve"]; a = r["args"]
want = dict(I["args"], data=data, out=str(Path(run).parent))
diff = {k: [a.get(k), v] for k, v in want.items() if a.get(k) != v}
chk(f"run args = intent ({len(want)} fields)", not diff, diff)
chk("code_sha " + I["code_sha"], r.get("code_sha") == I["code_sha"], r.get("code_sha"))
chk("dep_sha toy_struct.py " + I["toy_sha"], r.get("dep_sha", {}).get("toy_struct.py") == I["toy_sha"], r.get("dep_sha"))
keys = ("steps", "n_params", "grad_ckpt_on", "geo_layers", "read_layers", "lm_chunk", "eval_every_steps", "lam_rule")
got = {k: r.get(k) for k in keys}
chk("run facts = intent " + json.dumps({k: I[k] for k in keys}), all(got[k] == I[k] for k in keys), got)
chk("lambda per domain = intent " + json.dumps(I["lam_domains"]), r.get("lam_domains") == I["lam_domains"], r.get("lam_domains"))
st = [e["step"] for e in c]
chk(f"finished: {I['evals']} evals, increasing, the last at step {I['steps'] - 1}",
    len(c) == I["evals"] and st == sorted(set(st)) and st[-1] == I["steps"] - 1, [len(c), st[-1]])
# a diverged run is data, not a failed check (as pt2main): onset = the first eval with a non-finite LM / val / prior value
fin = lambda v: v is None or math.isfinite(v)
bad = [e["step"] for e in c if not (fin(e["lm"]) and fin(e["val_loss"]) and fin(e["geo"]) and fin(e.get("grad_prior"))
                                     and all(fin(e.get(f"geo_{d}")) for d in EXT))]
diverged = bool(bad)
pre = [e for e in c if not bad or e["step"] < bad[0]]
# data: the same mixture as Torch's dry run (step 0) and as every other arm of this (K, seed)
chk(f"data_sha after step 0 = Torch's {I['data_sha0']}", c[0].get("data_sha") == I["data_sha0"], c[0].get("data_sha"))
if I["lm0_ref"] is not None:
    chk("step-0 lm within 1e-3 of the Torch CPU reference (same init and first batch; the prior is off at step 0)",
        abs(c[0]["lm"] - I["lm0_ref"]) <= 1e-3, c[0]["lm"] - I["lm0_ref"])
chk("the numbers line is the first line after the text in every row (A2)", abs(c[-1]["first_line_frac"]["numbers"] - 1) < 1e-12,
    c[-1]["first_line_frac"])
ts = c[-1]["token_share"]["numbers"]
chk(f"numbers token share in {I['numbers_share']} (the 2% setting; Torch dry 2.12-2.13%)",
    I["numbers_share"][0] <= ts <= I["numbers_share"][1], ts)
sl = c[-1]["slot_frac"]
chk(f"domain lines per slot after the numbers line: K = {I['K']} (K = 2: each slot 45-55% of every domain's lines)",
    all(len(v) == I["K"] and (I["K"] == 1 or min(v) >= 0.45) for v in sl.values()), sl)
dm = {d: {k: r["domains"][d].get(k) for k in I["dom"][d]} for d in EXT}
chk("domain splits = Torch's dry run " + json.dumps(I["dom"]), dm == I["dom"], dm)
nf = {d: len(r["domains"][d].get("formats", {})) for d in EXT}
fk = [f"{d}_p{t}_{s}" for d in EXT for t in range(1, I["n_fmt"][d] + 1) for s in ("held", "seen")]
chk(f"held-out-format templates {I['n_fmt']}, scored at every eval ({len(fk)} sets)",
    nf == I["n_fmt"] and all(e.get(k) is not None for e in c for k in fk), nf)
# priors
if I["events"]:
    on_frac = c[-1]["prior_on_frac"]
    chk(f"prior_events == {I['events']}", r.get("prior_events") == I["events"], r.get("prior_events"))
    chk(f"prior_on_frac == {I['on_frac']:.6f}", abs(on_frac - I["on_frac"]) < 1e-9, on_frac)
    chk("prior_on_domains: every domain on exactly at the evals with prior_on", all(
        all(v == e["prior_on"] for v in e["prior_on_domains"].values()) and len(e["prior_on_domains"]) == 6 for e in c),
        sum(e["prior_on"] for e in c))
    chk("numbers prior applied on every on-step (ctx: prior_used_frac == prior_on_frac)",
        abs(c[-1]["prior_used_frac"] - on_frac) < 1e-9 and abs(c[-1]["prior_used_frac_numbers"] - on_frac) < 1e-9,
        [c[-1]["prior_used_frac"], c[-1]["prior_used_frac_numbers"], on_frac])
    used = {d: c[-1][f"prior_used_frac_{d}"] for d in EXT}
    chk("every domain prior applied on >= 95% of the on-steps", all(u >= 0.95 * on_frac - 1e-12 for u in used.values()), used)
    on = [e for e in pre if e["prior_on"]]
    # geo_<d> is None on a batch with < 4 sites or < 3 distinct items of d (the prior skipped there; counted above)
    chk("geo > 0 for every prior and 0 < grad_prior < inf at every eval where the priors acted (before a divergence)",
        (bool(on) or diverged) and all(e["geo"] > 0 and all(e.get(f"geo_{d}") is None or e[f"geo_{d}"] > 0 for d in EXT)
                                      and 0 < e.get("grad_prior", 0) < math.inf for e in on), len(on))
    if on:
        we = on[-1]
        rise = {d: we[f"cka_{d}"] - c[0][f"cka_{d}"] for d in EXT}
        rise["numbers"] = we["cka_ops_helix3"] - c[0]["cka_ops_helix3"]
        chk(f"item-state CKA to the true target rises by >= {I['cka_rise']} from step 0 to the window's last eval (step "
            f"{we['step']}), every prior", diverged or all(v >= I["cka_rise"] for v in rise.values()), {k: round(v, 3) for k, v in rise.items()})
else:
    chk("no prior ever on (prior_on_frac 0, no events, no domain prior on or applied)",
        c[-1]["prior_on_frac"] == 0 and r.get("prior_events") == [] and all(
            not any(e["prior_on_domains"].values()) for e in c) and all(c[-1][f"prior_used_frac_{d}"] == 0 for d in ("numbers",) + EXT),
        c[-1]["prior_on_frac"])
o = json.loads(Path(offl).read_text()).get(str(Path(run).resolve())) if Path(offl).exists() else None
chk("offline_lm on the full val, |orig_minus_logged| < 0.002 (skipped if the run diverged)",
    o is not None and o["max_windows"] == 0 and (diverged or abs(o["orig_minus_logged"]) < 0.002), o and o["orig_minus_logged"])
held = lambda e: mean([e[f"held_{b}"] for b in B])
m = r.get("metrics", {})
rate = [(q["sec"] - p["sec"]) / (q["step"] - p["step"]) for p, q in zip(c, c[1:]) if q["sec"] > p["sec"] and q["step"] > p["step"]]
util, mem = [], []
for row in csv.reader(open(gpu)) if Path(gpu).exists() else []:
    try:
        util.append(int(row[2].split()[0])); mem.append(int(row[4].split()[0]))
    except (IndexError, ValueError):
        pass
f3 = lambda k: m.get(k, {}).get("final3")
info = {"tag": a["tag"], "seed": a["seed"], "K": a["extra_lines"], "numbers_area": round(100 * mean([held(e) for e in c]), 2),
        "numbers_final3": f3("numbers"), "readout_final3": {d: f3(f"{d}_readout") for d in EXT},
        "readoutE_final3": {d: f3(f"{d}_readoutE") for d in EXT}, "seen_final3": {d: f3(f"{d}_seen") for d in EXT},
        "format_readout_final3": {d: [f3(f"{d}_p{t}_readout") for t in range(1, I["n_fmt"][d] + 1)] for d in EXT},
        "final_cka": {d: c[-1].get(f"cka_{d}") for d in EXT}, "cka_ops_helix3": c[-1].get("cka_ops_helix3"),
        "seen_all_ge_80": all((f3(f"{d}_seen") or 0) >= 80 for d in EXT),
        "lm0": c[0]["lm"], "final_val_bpb": c[-1]["val_bits_per_byte"], "bpb_full": o and o["bpb_full"], "diverged": diverged,
        "diverged_at_step": bad[0] if bad else None, "sec_per_step_median": statistics.median(rate) if rate else None,
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


def intent(k, arm, seed):
    """The intent of one task: everything the check asserts on the run JSON (also used by the CPU test of CHECK_PY)."""
    events, on_frac = window_events(STEPS) if arm != "none" else ([], 0.0)
    if arm != "none":
        assert events == [[916, 1], [7324, 0]], events
    lam_d = {d: (5.0 if arm == "all6" else 0.0) for d in ("numbers",) + EXT}
    tag = f"rb6pb-53m-2pct-k{k}-{arm}"
    return {"code_sha": PIN_SHA, "toy_sha": TOY_SHA, "steps": STEPS, "n_params": N_PARAMS, "grad_ckpt_on": False,
            "geo_layers": [2, 4], "read_layers": [2, 4], "lm_chunk": 0, "eval_every_steps": EVAL_EVERY,
            "lam_rule": "each" if arm == "all6" else "norm", "lam_domains": lam_d, "K": k,
            "evals": len(range(0, STEPS, EVAL_EVERY)) + ((STEPS - 1) % EVAL_EVERY != 0), "events": events, "on_frac": on_frac,
            "lm0_ref": LM0.get((k, seed)), "data_sha0": DSHA0.get((k, seed)), "dom": DOMS.get(seed), "n_fmt": N_FMT,
            "cka_rise": CKA_RISE, "numbers_share": [0.019, 0.024],
            "args": {"tokenization": "whole", "target": WANT[arm]["--target"], "site": "ctx",
                     "schedule": WANT[arm]["--schedule"] or "whole", "window": "0.05,0.4", "arith_lines": 1, "lr": 1e-3,
                     "lam": 5.0, "tokens": 3e8, "bs": BS, "seq": SEQ, "d_model": 384, "n_layers": 6, "n_heads": 6,
                     "d_ff": 1536, "seed": seed, "tag": f"_{tag}", "save_model": True, "ckpt_every": 2000,
                     "geo_layers": None, "eval_every": EVAL_EVERY, "n_eval": 150, "grad_ckpt": "auto", "lm_chunk_tokens": -1,
                     "untie": False, "gate_thr": 0.5, "gate_start": 0.0, "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256,
                     "stop_after": -1, "dry_inputs": False, "domains": ALL6, "extra_lines": k,
                     "prior_domains": ALL6 if arm == "all6" else "", "prior_shuf": "",
                     "lam_rule": "each" if arm == "all6" else "norm", "cyc_held": 0.4, "cyc_heldE": 0.5, "s4_train": 0.6,
                     "dim_train": 0.5, "dom_eval_rows": 200, "dom_seen": 150}}


def render(k, arm, seed):
    """(task name, tag, file text, trainer ARGS with the shell variables) of one (K, arm, seed)."""
    num = 400 + 4 * seed + 2 * (k - 1) + ARMS.index(arm)
    tag = f"rb6pb-53m-2pct-k{k}-{arm}"
    name = f"{num}-{tag}-s{seed}"
    args = (f"--data $DATA --tokenizer $TOK --tokenization whole --out $OUT/$TAG $ARM --arith_lines 1 --domains {ALL6} "
            f"--extra_lines {k} --lr 1e-3 --seed $SEED --tag _$TAG --save_model --ckpt_every 2000")
    f = flags(args.replace("$ARM", ARM[arm]).replace("$SEED", str(seed)).replace("$TAG", tag))
    want = {"--data": "$DATA", "--tokenizer": "$TOK", "--tokenization": "whole", "--out": f"$OUT/{tag}", "--arith_lines": "1",
            "--domains": ALL6, "--extra_lines": str(k), "--lr": "1e-3", "--seed": str(seed), "--tag": f"_{tag}",
            "--save_model": True, "--ckpt_every": "2000", **WANT[arm]}
    for kk, v in want.items():
        assert f.get(kk) == v, (name, kk, f.get(kk), v)
    assert set(f) == {kk for kk, v in want.items() if v is not None}, (name, sorted(f))   # every other flag = its default
    assert LIM_X * PLAN_H <= hours(TLIM) < LIM_X * PLAN_H + 0.5, (name, TLIM)
    I = intent(k, arm, seed)
    for key in ("lm0_ref", "data_sha0", "dom"):
        assert I[key] is not None, f"{name}: no Torch reference {key} for (K {k}, seed {seed}); compute it first"
    assert (k, seed) in DRY, f"{name}: no DRY hash for (K {k}, seed {seed}); compute it on Torch first"
    dry, n_dry = DRY[(k, seed)]
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    if arm == "none":
        target_assert = """! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a prior target in a none run"; exit 3; }"""
    else:
        target_assert = """grep -qxF 'target features (1000, 10)' "$OUT/dry_inputs.txt" || { echo "## not the helix3 table (1000 x 10)"; exit 3; }"""
    runfile = f"run_whole_{WANT[arm]['--target']}_seed{seed}_{tag}.json"
    about = textwrap.fill(f"Cell {tag}: {ARM_DESC[arm]}; 53M, 3e8 tokens = {STEPS} steps of {BS} x {SEQ} on {DATA[0]}, lr "
                          f"1e-3; 2% numbers (1 line per row, first after the text) + {k} domain line(s) per row (months, "
                          f"weekdays, clock, s4, dim cycled); seed {seed}.", 120, initial_indent="# ", subsequent_indent="# ")
    text = f"""# kind: gpu
# time: {TLIM}
# cpus: 8
# after: 331-env311 332-data
# rb6 rainbow6 PILOT (2026-10-17 user decisions: six manifolds at once, the same data in every arm, numbers at the 2% main
# setting first in every row, fixed training templates + held-out-format evals, none vs all6 with lambda 5 per prior; intent
# spec and pre-registered pilot rule runs/rb6/AUDIT.md amendments A1, A2). Generated by tools/make_rb6_tasks.py: change the
# generator, not this file.
{about}
# Read-out: tools/rainbow6_summary.py over this batch's run JSONs only (tag rb6pb-: never merged with Torch's rb6p- runs).
# Trainer pinned at {PIN} (tools/pretrain_range.py sha256 {PIN_SHA}..., toy_struct.py {TOY_SHA}...) through burst/pin_run.sh.
# 1. inputs: data sha256 = Torch's; --dry_inputs: model line, prior target, numbers line first, and the sha256 of the
#    selected decoded lines = the value computed on Torch for (K {k}, seed {seed}) (exit 3 otherwise, before any training)
# 2. training with --save_model --ckpt_every 2000 (a retry keeps $OUT and resumes from the checkpoint)
# 3. offline_lm.py on the saved model (full val.bin of {DATA[0]})
# 4. intent checks of the run JSON -> $OUT/check.{{json,txt}} (exit 4 on a fail)
# nvidia-smi utilisation / memory every 60 s -> $OUT/gpu.csv (planning: {PLAN_H:.2f} A100-h; limit {TLIM}).
set -euo pipefail
SEED={seed}; TAG={tag}; ARM="{ARM[arm]}"
ROOT=${{LLMADD_ROOT:-/scratch/$USER}}; DATA=$ROOT/llmadd_data/{DATA[0]}; TOK=$ROOT/llmadd_data/olmo2_tok; PIN={PIN}
[ -s "$OUT/gpu.csv" ] || echo "timestamp, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]" > "$OUT/gpu.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader -l 60 >> "$OUT/gpu.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
{{ date -u +%FT%TZ; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "cpus $(nproc)"; }} >> "$OUT/node.txt" 2>&1 || true

echo "## 1. inputs $(date -u +%FT%TZ)"
(cd "$DATA" && sha256sum -c -) <<'SHA' || exit 3
{DATA[1]}  train.bin
{DATA[2]}  val.bin
SHA
ARGS="{args}"
bash burst/pin_run.sh $PIN pretrain_range.py $ARGS --dry_inputs > "$OUT/dry_inputs.txt" 2>&1 || {{ cat "$OUT/dry_inputs.txt"; exit 3; }}
cat "$OUT/dry_inputs.txt"
grep -qxF '{MODEL_LINE}' "$OUT/dry_inputs.txt" || {{ echo "## not the intended model size / gradient checkpointing state"; exit 3; }}
{target_assert}
grep -q '^first line after the text (share of rows): {{"numbers": 1.0, ' "$OUT/dry_inputs.txt" || {{ echo "## the numbers line is not first in every row"; exit 3; }}
grep -q '^content scan: 640 training rows, no held-out prompt of ' "$OUT/dry_inputs.txt" || {{ echo "## no content scan"; exit 3; }}
n=$(set +o pipefail; grep -cE "{DRY_GREP}" "$OUT/dry_inputs.txt")
got=$(set +o pipefail; grep -E "{DRY_GREP}" "$OUT/dry_inputs.txt" | sha256sum | cut -d' ' -f1)
[ "$n" = {n_dry} ] && [ "$got" = {dry} ] || {{ echo "## dry_inputs differ from the rows computed on Torch ($n lines, sha256 $got)"; exit 3; }}
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
    """Every (K, arm, seed) of the pilot in task-number order: [(name, tag, text, args, K, arm, seed)]."""
    out = [render(k, a, s) + (k, a, s) for s in SEEDS for k in KS for a in ARMS]
    return sorted(out, key=lambda t: t[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="render and assert only; write nothing")
    ap.add_argument("--show", action="store_true", help="print the trainer arguments of every task")
    ap.add_argument("--repo", default=str(REPO))
    a = ap.parse_args()
    repo = Path(a.repo)
    T = tasks()
    names = [t[0] for t in T]
    assert len(set(names)) == len(names) == 8 and [int(n[:3]) for n in names] == list(range(400, 408)), names
    assert len({t[1] for t in T}) == 4, "4 cells (K x arm), 2 seeds each"
    if a.show:
        for name, tag, text, args, k, arm, seed in T:
            print(f"{name}\n    {args.replace('$ARM', ARM[arm]).replace('$SEED', str(seed)).replace('$TAG', tag)}")
    tdir, sdir = repo / "tasks", repo / "status"
    foreign = sorted(p.name for p in tdir.glob("40[0-9]*.sh") if p.stem not in names)
    if foreign:
        print("WARNING: task files in 400-409 not made by this generator:", foreign)
    new, same, differ, ran = [], [], [], []
    for name, tag, text, *_ in T:
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
        if not PIN:
            print("PIN not set (commit Rev B + A2 first, then set PIN): nothing written", file=sys.stderr)
            sys.exit(2)
        for p, text in new:
            p.write_text(text)
    print(f"{len(T)} tasks: {len(new)} {'to write' if a.check or not PIN else 'written'}, {len(same)} unchanged, "
          f"{len(differ)} differ (kept), {len(ran)} already ran")


if __name__ == "__main__":
    main()
