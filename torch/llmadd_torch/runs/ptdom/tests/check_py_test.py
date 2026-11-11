"""Known-answer test of the per-run CHECK (make_jobs_ptdom.CHECK_PY, the jobs' text) on the real Rev E run JSONs of the
200-step CPU smoke (tests/smoke/stars, tests/smoke/none), with the intent of the arm (make_jobs_ptdom.intent) and the tiny
run's facts substituted (steps, model, layers, evals, window, tokens, eval sets, --anc_eval_max). The references are
cross-checks, not copies: step-0 lm / data_sha from the OTHER arm's run (identical data), the splits and anchor table shas
from the 53M CPU references of seed 0 (size-independent). Expected: every check PASS except the two that need a GPU job
(offline_lm full-val file, the L40S name in gpu.csv). Tampers must fail exactly their check.
A1: CHECK_PY takes the job's ARGS as its 8th argument (here the smoke's own command line, smoke_200.sh): every flag = the
run's args, and the INTENDED lambda per domain comes from it (the gnorm ratio is compared with that, not with the intent's
table); the decomposition double path reads the on-window evals only (none: every eval). New tampers: --lam 2 in the
ARGS, another arm's ARGS, a grad_T_reldiff of 0.3 at the 3 on-window evals (the pre-A1 median over all 11 evals passes it).
A2: the decomposition check takes the MAX over the on-window evals: a grad_T_reldiff of 0.3 at ONE of the 3 on-window evals
fails (A1's on-window median passes it); the check's info carries the A2 LM share by norm a / (a + b) (= readout_ptdom's)."""
import copy
import glob
import json
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "ptdom"
sys.path.insert(0, str(B)); sys.path.insert(0, str(R / "code" / "tools"))
import make_jobs_ptdom as M   # noqa: E402
import make_rb6_tasks as P    # noqa: E402

S = R / "tests" / "smoke"
RUN = {a: Path(glob.glob(str(S / d / "run_*.json"))[0]) for a, d in (("only-stars", "stars"), ("none", "none"))}
J = {a: json.loads(p.read_text()) for a, p in RUN.items()}
RF = M.refs()[0]
ENV_FAIL = {"offline_lm on the full val", "GPU = L40S"}
ok = []


def T(name, cond, val=None):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val is not None else ""))


def tiny_intent(arm):
    other = J["none" if arm != "none" else "only-stars"]
    ref = {"lm0": other["curve"][0]["lm"], "dsha0": other["curve"][0]["data_sha"],
           "dom": copy.deepcopy(RF["dom"]), "table_sha": RF["table_sha"]}
    for d in M.ANC:   # --anc_eval_max 40: the read-out set is the first 40 held questions
        ref["dom"][d]["n_readout"] = 40
    I = M.intent(P, arm, "1e-3", 0, ref)
    pri = arm != "none"
    I.update(steps=200, n_params=J[arm]["n_params"], geo_layers=[1, 2], read_layers=[1, 2], eval_every_steps=20, evals=11,
             events=[[10, 1], [80, 0]] if pri else [], on_frac=0.35 if pri else 0.0, numbers_share=[0.0, 1.0])
    I["args"].update(d_model=64, n_layers=3, n_heads=4, d_ff=128, seq=128, bs=8, tokens=204800.0, eval_every=5, n_eval=20,
                     dom_eval_rows=20, dom_seen=20, anc_eval_max=40, tag="", ckpt_every=0)
    return I


TINY = ("--d_model 64 --n_layers 3 --n_heads 4 --d_ff 128 --seq 128 --bs 8 --tokens 204800 --eval_every 5 --n_eval 20 "
        "--dom_eval_rows 20 --dom_seen 20 --anc_eval_max 40 --save_model --seed 0")
COMMON = (f"--data {M.DATA} --tokenizer {M.TOK} --tokenization whole --arith_lines 1 "
          "--domains numbers,months,stars,landmarks,languages,colours,cities --extra_lines 2 --lr 1e-3")


def tiny_args(arm):
    """The smoke's own command line (tests/smoke_200.sh) = the $ARGS a job passes to CHECK_PY."""
    if arm == "none":
        return f"{COMMON} --out {S}/none --target none {TINY}"
    return (f"{COMMON} --out {S}/stars --target none --schedule window --window 0.05,0.4 --site ctx --lam_rule each "
            f"--prior_domains stars --lam 1 {TINY}")


def run_check(run_json, I, tmp, args=None):
    args = args if args is not None else tiny_args("none" if not I["active"] else "only-stars")
    o = subprocess.run([M.PY, "-", str(run_json), f"{tmp}/none.json", f"{tmp}/none.csv", str(R / "code" / "COMMIT"),
                        str(M.DATA), f"{tmp}/check.json", json.dumps(I), args], input=M.CHECK_PY, capture_output=True, text=True)
    ck = json.loads(Path(f"{tmp}/check.json").read_text()) if Path(f"{tmp}/check.json").exists() else {"checks": []}
    return o.returncode, {c["check"] for c in ck["checks"] if not c["ok"]}, o


CKAF = set()
with tempfile.TemporaryDirectory() as tmp:
    for arm in ("only-stars", "none"):
        rc, fails, o = run_check(RUN[arm], tiny_intent(arm), tmp)
        envf = {f for f in fails if any(f.startswith(e) for e in ENV_FAIL)}
        # the CKA-rise sanity bar (0.1, set for 53M) is not reached by the tiny model in 50 on-steps (cka_stars 0.091 ->
        # 0.151 at the window's last eval, 0.344 at step 80): expected here; a CKA-only failure keeps a run valid (read-out)
        ckaf = {f for f in fails if f.startswith("item-state CKA to the target rises")}
        CKAF = ckaf if arm == "only-stars" else CKAF
        T(f"{arm}: every check PASS except the GPU-job ones ({len(envf)})" + (" and the tiny model's CKA-rise bar" if ckaf else "")
          + "; rc 4", rc == 4 and fails == envf | ckaf and len(envf) == 2 and (arm != "none" or not ckaf),
          sorted(fails - envf - ckaf) or o.stderr[-400:])
    sys.path.insert(0, str(R))
    import readout_ptdom as RO
    ck = Path(tmp) / "ro" / "ptdom-53m-k2-only-stars-lr1e-3-s0"
    (ck / "ptdom-53m-k2-only-stars-lr1e-3").mkdir(parents=True)
    j = copy.deepcopy(J["only-stars"]); j["curve"] = j["curve"] + [j["curve"][-1]] * (RO.N_EVALS - len(j["curve"]))
    (ck / "ptdom-53m-k2-only-stars-lr1e-3" / "run_whole_none_seed0_ptdom-53m-k2-only-stars-lr1e-3.json").write_text(json.dumps(j))
    (ck / "check.json").write_text(json.dumps({"pass": False, "checks": [{"check": f, "ok": False} for f in CKAF]}))
    rec = RO.load(Path(tmp) / "ro")[("only-stars", "1e-3", 0)]
    T("the read-out loader keeps a run whose only failed check is the CKA-rise bar (valid data, noted)",
      rec["valid"] and "CKA-rise" in rec["why"], rec["why"])
    I = tiny_intent("only-stars"); I["lam_domains"]["stars"] = 2.0
    rc, fails, _ = run_check(RUN["only-stars"], I, tmp)
    T("tamper: lambda 2 in the intent -> the lambda-per-domain = intent and domains-line checks fail; the gnorm ratio (vs the "
      "job's ARGS, lambda 1) passes", any(f.startswith("lambda per domain = intent") for f in fails)
      and any(f.startswith("domains.<d> prior / shuf / lam") for f in fails) and not any(f.startswith("gnorm_prior_<d>") for f in fails),
      sorted(f[:50] for f in fails))
    rc, fails, _ = run_check(RUN["only-stars"], tiny_intent("only-stars"), tmp, tiny_args("only-stars").replace("--lam 1 ", "--lam 2 "))
    T("A1 tamper: --lam 2 in the job's ARGS (the run logged lambda 1) -> the ARGS = run args, lambda-from-ARGS and gnorm-ratio "
      "checks fail", any(f.startswith("the job's ARGS") for f in fails) and any(f.startswith("lambda per domain from the job's ARGS") for f in fails)
      and any(f.startswith("gnorm_prior_<d> / gnorm_true_<d> = the lambda of the job's ARGS") for f in fails), sorted(f[:60] for f in fails))
    rc, fails, _ = run_check(RUN["only-stars"], tiny_intent("only-stars"), tmp, tiny_args("none"))
    T("A1 tamper: the none arm's ARGS with the stars run -> the ARGS = run args and lambda-from-ARGS checks fail",
      any(f.startswith("the job's ARGS") for f in fails) and any(f.startswith("lambda per domain from the job's ARGS") for f in fails),
      sorted(f[:60] for f in fails))
    j = copy.deepcopy(J["only-stars"])
    for e in j["curve"]:
        if e["prior_on"]:
            e["grad_T_reldiff"] = 0.3
    p = Path(tmp) / "run_whole_none_seed0.json"; p.write_text(json.dumps(j))
    import statistics
    old_med = statistics.median(e["grad_T_reldiff"] for e in j["curve"])
    rc, fails, _ = run_check(p, tiny_intent("only-stars"), tmp)
    T(f"A1 tamper: grad_T_reldiff 0.3 at the 3 on-window evals -> the on-window decomposition check fails (the pre-A1 median "
      f"over all 11 evals, {old_med:.1e}, would pass)", any(f.startswith("decomposition double path") for f in fails) and old_med < 0.05,
      sorted(f[:50] for f in fails))
    on_i = [i for i, e in enumerate(j["curve"]) if e["prior_on"]]
    j = copy.deepcopy(J["only-stars"]); j["curve"][on_i[0]]["grad_T_reldiff"] = 0.3
    p.write_text(json.dumps(j))
    med_on = statistics.median(e["grad_T_reldiff"] for e in j["curve"] if e["prior_on"])
    rc, fails, _ = run_check(p, tiny_intent("only-stars"), tmp)
    T(f"A2 tamper: grad_T_reldiff 0.3 at ONE of the {len(on_i)} on-window evals -> the max-based decomposition check fails (A1's "
      f"on-window median {med_on:.1e} would pass)", any(f.startswith("decomposition double path: max") for f in fails) and med_on < 0.05
      and len(on_i) == 3, sorted(f[:50] for f in fails))
    rc, fails, _ = run_check(RUN["only-stars"], tiny_intent("only-stars"), tmp)
    info = json.loads(Path(f"{tmp}/check.json").read_text())["info"]["grad_window_median"]
    ref = RO.window_grad(J["only-stars"]["curve"])["share"]
    T(f"A2: check.json info carries the LM share by norm a / (a + b) {info['lm_share_norm']:.4f} = readout_ptdom.window_grad's {ref:.4f}; "
      f"the untampered run passes the max-based decomposition check", abs(info["lm_share_norm"] - ref) < 1e-12
      and not any(f.startswith("decomposition double path") for f in fails))
    I = tiny_intent("none"); I["data_sha0"] = "0" * 16
    rc, fails, _ = run_check(RUN["none"], I, tmp)
    T("tamper: a wrong data_sha0 -> the data check fails", any(f.startswith("data_sha after step 0") for f in fails))
    I = tiny_intent("none"); I["table_sha"] = dict(I["table_sha"], stars="000000000000")
    rc, fails, _ = run_check(RUN["none"], I, tmp)
    T("tamper: a wrong anchor table sha -> the table check fails", any(f.startswith("anchor tables") for f in fails))
    I = tiny_intent("none"); I["args"]["lr"] = 2e-3
    rc, fails, _ = run_check(RUN["none"], I, tmp)
    T("tamper: lr 2e-3 in the intent -> the args check fails", any(f.startswith("run args = intent") for f in fails))
    j = copy.deepcopy(J["none"]); j["curve"][3]["geo_stars"] = 0.5
    p = Path(tmp) / "run_whole_none_seed0.json"; p.write_text(json.dumps(j))
    rc, fails, _ = run_check(p, tiny_intent("none"), tmp)
    T("tamper: a geo_stars value in the none run -> 'no prior ever' fails", any(f.startswith("no prior ever") for f in fails))
    j = copy.deepcopy(J["only-stars"]); j["curve"][2]["prior_on_domains"]["landmarks"] = 1
    p.write_text(json.dumps(j))
    rc, fails, _ = run_check(p, tiny_intent("only-stars"), tmp)
    T("tamper: landmarks marked on in the stars run -> the prior_on_domains check fails", any(f.startswith("prior_on_domains") for f in fails))
print("CHECKPY", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)})")
