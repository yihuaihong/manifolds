"""pt2sz test: the trainer arguments of rendered jobs vs the run JSON 'args' of the nearest submitted 2% runs. The full
argument dict a job's ARGS produce is emulated with the snapshot trainer's own argparse (pretrain_range.main up to
parse_args, then the run JSON's own filter: DOM_ARGS / REVC_DEFAULTS / REVD_DEFAULTS; pt2same's tests/args_diff.py method).
Known answers first: K1 the ARGS line of submitted pt2same / pt2s505 / pttpp-2% job files -> = their finished run JSON's
args exactly. Then every pairing below -> the differing keys must be exactly the expected set:
  vs pt2same 190M (same lr, seed): shape (d_model, n_layers, n_heads, d_ff), out, tag
  vs pt2s505 505M (same lr, seed): shape, out, tag
  vs pt2same 53M (other lr): shape, lr, out, tag
  vs pttpp tpp2-294m-t0.745 (the same shape at 156.8M tokens, same lr, seed): tokens, eval_every (3e8 defaults here),
     ckpt_every (2000 here, as pt2same / pt2s505; pttpp's 2% jobs had none), out, tag
Also: every rendered stage-A job's CHECK_PY intent args = its emulated run args on every key (data / out set as CHECK_PY
does; the only run arg outside the intent is the tokenizer, as in pt2same / pt2s505); the INTENT line's numbers = the intent.
    python args_diff.py   (CPU only, writes nothing)"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2sz as M   # noqa: E402

G2 = M.g2()
sys.path.insert(0, str(M.R / "code" / "tools"))
import pretrain_range as P   # noqa: E402  (the snapshot trainer: 93e22bf4c4c0)
assert Path(P.__file__).resolve().parent == (M.R / "code" / "tools").resolve()
RUNS = M.B / "runs"


class _Got(Exception):
    pass


_orig = argparse.ArgumentParser.parse_args


def _pa(self, *a, **k):
    raise _Got(_orig(self, *a, **k))


def emulate(args):
    argparse.ArgumentParser.parse_args = _pa
    sys.argv = ["pretrain_range.py"] + args.split()
    try:
        P.main()
    except _Got as g:
        ns = g.args[0]
    else:
        raise AssertionError("parse_args not reached")
    finally:
        argparse.ArgumentParser.parse_args = _orig
    dom = bool(ns.domains)
    return {k: v for k, v in vars(ns).items() if (dom or k not in P.DOM_ARGS)
            and not (k in P.REVC_DEFAULTS and v == P.REVC_DEFAULTS[k]) and not (k in P.REVD_DEFAULTS and v == P.REVD_DEFAULTS[k])}


def diff(a, b):
    return {k: (a.get(k, "<absent>"), b.get(k, "<absent>")) for k in sorted(set(a) | set(b)) if a.get(k, "<absent>") != b.get(k, "<absent>")}


fails = 0


def say(ok, msg):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + msg)


def job_args(f):
    return re.findall(r'^ARGS="(.*)"$', Path(f).read_text(), re.M)[0]


def run_of(d, tag, arm, seed):
    t = "none" if arm == "none" else "helix3"
    return json.loads((RUNS / d / tag / f"run_whole_{t}_seed{seed}_{tag}.json").read_text())


REF = {  # name: (job file, run dir, tag, arm, seed)
    "pt2same-190m-none-s0": ("pt2same/job_pt2same-190m-lr4.8e-3-none-s0.sbatch", "pt2same/out/pt2same-190m-lr4.8e-3-none-s0", "pt2same-190m-lr4.8e-3-none", "none", 0),
    "pt2same-190m-win-s5": ("pt2same/job_pt2same-190m-lr6e-4-win-ctx-s5.sbatch", "pt2same/out/pt2same-190m-lr6e-4-win-ctx-s5", "pt2same-190m-lr6e-4-win-ctx", "win-ctx", 5),
    "pt2same-53m-none-s1": ("pt2same/job_pt2same-53m-lr8e-3-none-s1.sbatch", "pt2same/out/pt2same-53m-lr8e-3-none-s1", "pt2same-53m-lr8e-3-none", "none", 1),
    "pt2s505-none-s0": ("pt2s505/job_pt2s505-lr4.8e-3-none-s0.sbatch", "pt2s505/out/pt2s505-lr4.8e-3-none-s0", "pt2s505-lr4.8e-3-none", "none", 0),
    "pt2s505-win-s2": ("pt2s505/job_pt2s505-lr3e-4-win-ctx-s2.sbatch", "pt2s505/out/pt2s505-lr3e-4-win-ctx-s2", "pt2s505-lr3e-4-win-ctx", "win-ctx", 2),
    "tpp2-294m-none-s0": ("pttpp/job_tpp2-294m-t0.745-lr4.8e-3-none-s0.sbatch", "pttpp/out2/tpp2-294m-t0.745-lr4.8e-3-none-s0", "tpp2-294m-t0.745-lr4.8e-3-none", "none", 0),
    "tpp2-294m-win-s4": ("pttpp/job_tpp2-294m-t0.745-lr6e-4-win-ctx-s4.sbatch", "pttpp/out2/tpp2-294m-t0.745-lr6e-4-win-ctx-s4", "tpp2-294m-t0.745-lr6e-4-win-ctx", "win-ctx", 4),
}
for k, (jf, d, tag, arm, seed) in REF.items():
    dd = diff(emulate(job_args(RUNS / jf)), run_of(d, tag, arm, seed)["args"])
    say(not dd, f"K1 emulation of the submitted {jf} ARGS = its run JSON args: {dd or 'identical'}")
SHAPE = {"d_model", "n_layers", "n_heads", "d_ff"}
PAIRS = [  # (mine: size, arm, lr, seed, stage), reference, expected differing keys
    (("93m", "none", "4.8e-3", 0, "A"), "pt2same-190m-none-s0", SHAPE | {"out", "tag"}),
    (("294m", "none", "4.8e-3", 0, "A"), "pt2same-190m-none-s0", SHAPE | {"out", "tag"}),
    (("294m", "win-ctx", "6e-4", 5, "C"), "pt2same-190m-win-s5", SHAPE | {"out", "tag"}),
    (("93m", "none", "4.8e-3", 0, "A"), "pt2s505-none-s0", SHAPE | {"out", "tag"}),
    (("294m", "win-ctx", "3e-4", 2, "A"), "pt2s505-win-s2", SHAPE | {"out", "tag"}),
    (("93m", "none", "2.4e-3", 1, "A"), "pt2same-53m-none-s1", SHAPE | {"lr", "out", "tag"}),
    (("294m", "none", "4.8e-3", 0, "A"), "tpp2-294m-none-s0", {"tokens", "eval_every", "ckpt_every", "out", "tag"}),
    (("294m", "win-ctx", "6e-4", 4, "C"), "tpp2-294m-win-s4", {"tokens", "eval_every", "ckpt_every", "out", "tag"}),
]
for (size, arm, lr, seed, stage), ref, want in PAIRS:
    F = M.facts(size, arm, lr, seed, stage, G2)
    text, args, I = M.render_one(G2, F)
    jf, d, tag, rarm, rseed = REF[ref]
    ra = run_of(d, tag, rarm, rseed)["args"]
    mine = emulate(args)
    dd = diff(mine, ra)
    say(set(dd) == want, f"ARGS {F['name']} vs {ref}: differing keys {sorted(dd)} (expected {sorted(want)})")
    for k_ in sorted(dd):
        print(f"      {k_}: pt2sz {dd[k_][0]!r} | {ref} {dd[k_][1]!r}")
# every rendered stage-A job: CHECK_PY's intent args = the emulated args (all keys), the INTENT line = the intent
n_ok, n = 0, 0
for z in M.SUBMIT_ORDER:
    for name, text, F in M.render(z, G2=G2):
        n += 1
        args = re.findall(r'^ARGS="(.*)"$', text, re.M)[0]
        I = json.loads(re.findall(r"""^  "\$OUT/check\.json" '(\{.*\})' <<'PY'""", text, re.M)[0])
        mine = emulate(args)
        want = dict(I["args"], data=str(M.DATA), out=f"{M.R}/out/{name}/{F['tag']}")
        di = {k: (mine.get(k), v) for k, v in want.items() if mine.get(k) != v}
        extra = set(mine) - set(want)
        il = re.findall(r'^echo "INTENT (.*)"$', text, re.M)[0]
        d_, nl, nh, dff = F["dims"]
        il_ok = (f"{z} d{d_} x {nl} ({nh} heads, d_ff {dff}), {F['n_params']} params, N_nonemb {F['nne']}" in il
                 and "tokens 3e8 = 18310 steps x 16384, eval every 250 steps -> 75 evals, warmup 183 steps" in il
                 and (("prior window steps 916..7323" in il) if F["arm"] != "none" else ("prior window none" in il))
                 and f"{F['arm']} lr {F['lr']} seed {F['seed']}; stage A; step-0 lm ref {I['lm0_ref']:.6f}" in il)
        # CHECK_PY's intent has no 'tokenizer' field (pt2same's / pt2s505's neither): the tokenizer is covered by the DRY
        # rows' hash (decoded rows) and the ARGS assert; every other run arg is an intent field
        good = not di and extra == {"tokenizer"} and mine["tokenizer"] == str(M.TOK) and il_ok and I["steps"] == 18310 and I["evals"] == 75
        n_ok += good
        if not good:
            print(f"FAIL {name}: intent vs args {di}, extra {extra}, INTENT line ok {il_ok}")
say(n_ok == n == 64, f"INTENT 64 stage-A jobs: CHECK_PY's arg fields = the emulated run args (every key; the only run arg not in the intent is tokenizer = the OLMo-2 path, covered by the DRY rows); INTENT line = the intent: {n_ok}/{n}")
print(f"ARGS DIFF {'PASS' if not fails else 'FAIL'} ({fails} failures)")
sys.exit(1 if fails else 0)
