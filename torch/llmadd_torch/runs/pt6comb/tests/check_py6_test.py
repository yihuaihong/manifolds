"""pt6comb test: CHECK_PY_6 (the job text's in-job checks) on REAL run JSONs of the tiny CPU smoke (tests/feas: all6, none,
only-numbers, only-s4 on composition B's data; d64 x 3, seq 128, bs 16, 200 steps), each with its arm's intent (make_jobs_pt6comb
.intent6) and its own command line as $ARGS. Tiny facts substituted in the intent (model size, steps, evals, the window events,
the reduced eval sets, the step-0 lm / data sha / split facts / table shas of the run itself: the CPU references hold for 53M);
the numbers token share (2% at seq 512) and the GPU name cannot hold on a tiny CPU run: exactly those two checks must fail.
Tampers must fail their own check. -> check_py6_test.out"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
B = Path("/scratch/yh6210/llmadd_torch")
sys.path.insert(0, str(B))
sys.argv = [sys.argv[0]]
import make_jobs_pt6comb as M6   # noqa: E402
P = M6.load_P()
F = M6.R / "tests" / "feas" / "out"
DATA, TOK = str(M6.MJ.DATA), str(M6.MJ.TOK)
COMMON = f"--data {DATA} --tokenizer {TOK} --tokenization whole --arith_lines 1 --domains numbers,s4,dim,cities,countries,people --extra_lines 2"
WIN = "--schedule window --window 0.05,0.4 --site ctx --lam_rule each --lam 1"
TINY = "--d_model 64 --n_layers 3 --n_heads 4 --d_ff 128 --seq 128 --n_eval 20 --dom_eval_rows 20 --dom_seen 20 --seed 0 --bs 16 --tokens 409600 --eval_every 10 --anc_eval_max 40 --lr 1e-3"
ARMS = {"all6": ("helix3", "numbers,s4,dim,cities,countries,people"), "none": ("none", None),
        "only-numbers": ("helix3", "numbers"), "only-s4": ("none", "s4")}
EXPECTED_FAIL = ("numbers token share", "GPU = B200")
res = []


def jargs(arm):
    tgt, pd = ARMS[arm]
    return f"{COMMON} --out {F / arm} --target {tgt}" + (f" {WIN} --prior_domains {pd}" if pd else "") + f" {TINY}"


def run_json(arm):
    return next((F / arm).glob("run_*.json"))


def tiny_intent(arm, r):
    c = r["curve"]
    D = r["domains"]
    ref = {"lm0": c[0]["lm"], "dsha0": c[0]["data_sha"],
           "dom": {d: {k: D[d].get(k) for k in M6.DOM_KEYS6[M6.dom_kind6(d)]} for d in M6.comp_doms("B")},
           "table_sha": {d: D[d]["table"]["sha"] for d in ("cities", "countries", "people")}}
    I = M6.intent6(P, "B", arm, "1e-3", 0, ref)
    I.update(steps=r["steps"], n_params=r["n_params"], geo_layers=[1, 2], read_layers=[1, 2], eval_every_steps=r["eval_every_steps"],
             evals=len(c))
    if I["events"]:
        I["events"] = r["prior_events"]
        I["on_frac"] = c[-1]["prior_on_frac"]
    I["args"].update(d_model=64, n_layers=3, n_heads=4, d_ff=128, seq=128, bs=16, tokens=409600.0, eval_every=10, n_eval=20,
                     dom_eval_rows=20, dom_seen=20, ckpt_every=0, tag="", anc_eval_max=40)
    return I


def expect(desc, fails, must=(), allowed=EXPECTED_FAIL, cross=False):
    if fails is None:
        res.append(False)
        print(f"FAIL {desc}: CHECK_PY_6 crashed")
        return
    bad = [] if cross else [f for f in fails if not f.startswith(allowed) and not any(f.startswith(m) for m in must)]   # cross-arm: more may fail
    miss = [m for m in must if not any(f.startswith(m) for f in fails)]
    ok = not bad and not miss and all(any(f.startswith(a) for f in fails) for a in allowed)
    res.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {desc}: failed checks = {fails}" + (f"  UNEXPECTED {bad}" if bad else "") + (f"  MISSING {miss}" if miss else ""))


for arm in ARMS:
    r = json.loads(run_json(arm).read_text())
    I = tiny_intent(arm, r)
    # args.out must equal the run file's parent: the run is written into <tmp>/t/ with args.out / --out pointed there
    def check_r(rr, I_, a_):
        import tempfile as tf
        with tf.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "t"; rd.mkdir()
            rr = copy.deepcopy(rr); rr["args"]["out"] = str(rd)
            a2 = a_.replace(f"--out {F / arm}", f"--out {rd}")
            rf = rd / "run.json"; rf.write_text(json.dumps(rr))
            out = Path(tmp) / "check.json"
            o = subprocess.run([M6.PY, "-", str(rf), str(Path(tmp) / "offline_lm.json"), str(Path(tmp) / "gpu.csv"),
                                str(M6.R / "code" / "COMMIT"), DATA, str(out), json.dumps(I_), a2], input=M6.CHECK_PY_6,
                               capture_output=True, text=True)
            if not out.exists():
                print(o.stderr[-1500:])
                return None, None
            ck = json.loads(out.read_text())
        return [k["check"] for k in ck["checks"] if not k["ok"]], ck
    fails, ck = check_r(r, I, jargs(arm))
    expect(f"{arm}: its own run, intent and ARGS", fails)
    if ck:
        names = [k["check"] for k in ck["checks"]]
        print(f"     {len(names)} checks; numbers-prior check: {[n[:60] for n in names if n.startswith(('numbers prior', M6.NUM_ON_CHECK))]}; "
              f"toy check: {sum(n.startswith(M6.TOY_CHECK) for n in names)}; geo check: {[n[:50] for n in names if n.startswith('geo > ')]}")
    if arm == "all6":
        on = [i for i, e in enumerate(r["curve"]) if e.get("prior_on")]
        t = copy.deepcopy(r); t["curve"][on[1]]["geo"] = -0.002
        expect("all6 tamper: the numbers prior value geo = -0.002 at an on-eval", check_r(t, I, jargs(arm))[0], must=("geo > -0.001 for every active prior",))
        t = copy.deepcopy(r); del t["curve"][4]["hits_s4_held"]
        expect("all6 tamper: hits_s4_held missing at one eval", check_r(t, I, jargs(arm))[0], must=(M6.TOY_CHECK,))
        t = copy.deepcopy(r); t["curve"][-1]["prior_used_frac_numbers"] = 0.0
        expect("all6 tamper: the numbers prior never applied (prior_used_frac_numbers 0)", check_r(t, I, jargs(arm))[0], must=(M6.NUM_ON_CHECK,))
        t = copy.deepcopy(r)
        for i in on:
            del t["curve"][i]["gnorm_prior_numbers"]
        expect("all6 tamper: no weighted numbers gradient at the on-evals", check_r(t, I, jargs(arm))[0],
               must=("weighted prior gradients at the on-evals from the active prior only", "gnorm_prior_<d> / gnorm_true_<d> = the lambda"))
        t = copy.deepcopy(r)
        for e in t["curve"]:
            e["cka_ops_helix3"] = r["curve"][0]["cka_ops_helix3"]
        expect("all6 tamper: the operands' CKA to helix3 flat (the numbers prior did not fit)", check_r(t, I, jargs(arm))[0],
               must=("item-state CKA to the target rises",))
        t = copy.deepcopy(r); t["domains"]["numbers"]["target"] = "none"
        expect("all6 tamper: domains.numbers.target none", check_r(t, I, jargs(arm))[0], must=("domains.<d> prior / shuf / lam = intent",))
        expect("all6 tamper: --lam 2 in the job's ARGS", check_r(r, I, jargs(arm).replace("--lam 1", "--lam 2"))[0],
               must=("the job's ARGS", "lambda per domain from the job's ARGS", "gnorm_prior_<d> / gnorm_true_<d> = the lambda"))
        expect("all6's run under none's intent and ARGS", check_r(r, tiny_intent("none", r), jargs("none"))[0],
               must=("run args = intent", "the job's ARGS", "lambda per domain from the job's ARGS", "lambda per domain = intent",
                     "domains.<d> prior / shuf / lam", "no prior ever on"), cross=True)
    if arm == "only-numbers":
        expect("only-numbers' run under only-s4's intent and ARGS", check_r(r, tiny_intent("only-s4", r), jargs("only-s4"))[0],
               must=("run args = intent", "the job's ARGS", "lambda per domain", "domains.<d> prior / shuf / lam", "prior_on_domains",
                     "numbers prior off", "the active prior applied", "geo_<d> logged", "weighted prior gradients"), cross=True)
print(f"CHECK_PY_6 TEST {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
