"""pt2same test: the trainer arguments of every rendered job vs the Burst run JSON 'args' of the same cell and seed.
The full argument dict a job's ARGS produce is emulated with the snapshot trainer's own argparse (pretrain_range.main up
to parse_args, then the run JSON's own filter: DOM_ARGS / REVC_DEFAULTS / REVD_DEFAULTS). Known answers first:
  K1 pt2s505's job ARGS -> = its finished run JSON's args exactly (none s0, win-ctx s4)
  K2 each Burst twin task's ARGS (tasks/<name>.sh, $-variables = that run's recorded data / tokenizer / out) -> = the Burst
     run JSON's args exactly (the newer trainer's parser reproduces what a3bce3ab63c1 recorded)
Then: every job's emulated args vs its Burst twin's run JSON args -> the differing keys (expected: data, tokenizer, out,
tag), the run facts (steps, n_params, grad ckpt, layers, eval spacing, lm_chunk, prior events) vs the job's intent, the
step-0 lm reference vs the Burst runs and pttpp's Torch A100-80GB 190M runs, and the read-out's Burst task table.
    python args_diff.py   (from anywhere; CPU only, writes nothing)"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2same as M   # noqa: E402

G2 = M.g2()
sys.path.insert(0, str(M.R / "code" / "tools"))
import pretrain_range as P   # noqa: E402  (the snapshot trainer: 93e22bf4c4c0)
assert Path(P.__file__).resolve().parent == (M.R / "code" / "tools").resolve()
sys.path.insert(0, str(M.R))
import readout_pt2same as RO   # noqa: E402


class _Got(Exception):
    pass


_orig = argparse.ArgumentParser.parse_args


def _pa(self, *a, **k):
    raise _Got(_orig(self, *a, **k))


def emulate(args):
    """the run JSON 'args' the trainer would record for this ARGS string"""
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


# K1: pt2s505 jobs
S505 = M.B / "runs" / "pt2s505"
for job, run in (("job_pt2s505-lr4.8e-3-none-s0.sbatch", "out/pt2s505-lr4.8e-3-none-s0/pt2s505-lr4.8e-3-none/run_whole_none_seed0_pt2s505-lr4.8e-3-none.json"),
                 ("job_pt2s505-lr6e-4-win-ctx-s4.sbatch", "out/pt2s505-lr6e-4-win-ctx-s4/pt2s505-lr6e-4-win-ctx/run_whole_helix3_seed4_pt2s505-lr6e-4-win-ctx.json")):
    a = re.findall(r'^ARGS="(.*)"$', (S505 / job).read_text(), re.M)[0]
    d = diff(emulate(a), json.loads((S505 / run).read_text())["args"])
    say(not d, f"K1 emulation = the pt2s505 run JSON args ({job}): {d or 'identical'}")

BURST = M.BURST
bt = M.burst_tasks(G2)
rows = []
keysets = {}
for name, tag, text, args, I, bname in M.jobs(G2):
    size = name.split("-")[1]
    arm = "none" if "-none-" in name else "win-ctx"
    seed = int(name.rsplit("-s", 1)[1])
    lr = M.PICKS[size][arm]
    btag = f"{M.BURST_BLOCK[size]}-lr{lr}-{arm}"
    target = "none" if arm == "none" else "helix3"
    br = json.loads((BURST / "results" / bname / btag / f"run_whole_{target}_seed{seed}_{btag}.json").read_text())
    ba = br["args"]
    # K2: the Burst task's own ARGS through the emulation = what the Burst run recorded
    targs = re.findall(r'^ARGS="(.*)"$', (BURST / "tasks" / f"{bname}.sh").read_text(), re.M)[0]
    sub = (targs.replace("$DATA", ba["data"]).replace("$TOK", ba["tokenizer"]).replace("$OUT/$TAG", ba["out"])
           .replace("$TAG", btag).replace("$ARM", G2.ARM[arm]).replace("$SEED", str(seed)))
    k2 = diff(emulate(sub), ba)
    say(not k2, f"K2 {bname}: emulation of the Burst task ARGS = its run JSON args: {k2 or 'identical'}")
    # the job vs the Burst twin
    mine = emulate(args)
    d = diff(mine, ba)
    keysets.setdefault(tuple(sorted(d)), []).append(name)
    unexpected = set(d) - {"data", "tokenizer", "out", "tag"}
    say(not unexpected, f"ARGS {name} vs {bname}: differing keys {sorted(d)}"
        + (f"  UNEXPECTED {({k: d[k] for k in unexpected})}" if unexpected else ""))
    if name.endswith(("-s0", "-s4")):
        for k in sorted(d):
            print(f"      {k}: Torch {d[k][0]!r} | Burst {d[k][1]!r}")
    # run facts = the job's intent (code_sha differs by design)
    facts = ("steps", "n_params", "grad_ckpt_on", "geo_layers", "read_layers", "lm_chunk", "eval_every_steps")
    fd = {k: (br.get(k), I[k]) for k in facts if br.get(k) != I[k]}
    ev = br.get("prior_events") == I["events"] and len(br["curve"]) == I["evals"]
    say(not fd and ev and br["code_sha"] == G2.PIN_SHA and I["code_sha"] == "93e22bf4c4c0",
        f"FACTS {name}: Burst run steps / n_params / grad ckpt / layers / lm_chunk / eval spacing / prior events / "
        f"{I['evals']} evals = the job's intent; code_sha Burst {br['code_sha']} -> Torch {I['code_sha']}")
    # step-0 lm: the job's reference vs the Burst run of this seed (arm-independent: same init and first batch)
    dl = br["curve"][0]["lm"] - I["lm0_ref"]
    say(abs(dl) <= 1e-3, f"LM0 {name}: Burst step-0 lm - the job's reference = {dl:+.1e} (CHECK_PY tolerance 1e-3)")
    # the job's intent args = the emulated args on every intent key (data / out are added by CHECK_PY itself)
    want = dict(I["args"], data=str(M.DATA), out=mine["out"])
    di = {k: (mine.get(k), v) for k, v in want.items() if mine.get(k) != v}
    say(not di, f"INTENT {name}: CHECK_PY's {len(want)} arg fields = the emulated run args: {di or 'all equal'}")
    rows.append((name, bname, sorted(d)))
print("== differing-key sets over the 32 jobs:")
for k, v in keysets.items():
    print(f"  {list(k)}: {len(v)} jobs")
# pttpp's Torch A100-80GB 190M runs (trainer 93e22bf4c4c0, same model, same seed): step-0 lm vs this batch's reference
for p in sorted((M.B / "runs" / "pttpp" / "out2").glob("tpp2-190m-*/check.json")):
    c = json.loads(p.read_text())
    s = c["info"]["seed"]
    dl = c["info"]["lm0"] - M.lm0_ref(G2, "190m", s)
    say(abs(dl) <= 1e-3, f"LM0 pttpp {p.parent.name}: Torch A100-80GB step-0 lm - the 190M reference (seed {s}) = {dl:+.1e}")
# the read-out's hard-coded Burst task table = the generator's
rt = {(z, a, s): bt[(z, a, M.PICKS[z][a], s)] for z in M.SIZES for a in M.ARMS for s in M.SEEDS}
say(rt == RO.BURST_TASK, "readout_pt2same.BURST_TASK = make_jobs_pt2same.burst_tasks (32 Burst twins)")
say(RO.PICKS["53m"] == M.PICKS["53m"] and RO.PICKS["190m"] == M.PICKS["190m"]
    and json.loads((S505 / "picks_confirm.json").read_text())["best"] == RO.PICKS["505m"],
    "read-out picks = the generator's (53M, 190M) and pt2s505's picks_confirm.json (505M)")
say(all(RO.N_NONEMB[z] == M.TP.SIZES[z][5] for z in RO.SIZES), "read-out N_nonemb = make_jobs_pttpp.SIZES")
print(f"ARGS DIFF {'PASS' if not fails else 'FAIL'} ({fails} failures)")
sys.exit(1 if fails else 0)
