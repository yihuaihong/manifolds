"""ptdom2 test: CHECK_PY_SHUF (the job's in-run checks of a shuffled arm) on REAL run JSONs of the trainer: the CPU tiny runs of
shuf_trace (tests/shuf/<batch>-only-<d>[-shuf]: --stop_after 24, so a truncated run) with the cell's own ptdom2 intent
(make_jobs_ptdom2.intent2) where the tiny facts are substituted (model size, steps, evals, window events, bs / seq / tokens / eval
flags, geo / read layers [1, 2] of the 3-layer model, the step-0 lm / data_sha / splits of the OTHER arm's tiny run of the same
seed: the data are arm-independent), and the
job's ARGS = the tiny command line. Expected: the shuffled run passes every check except the ones a tiny truncated CPU run must
fail (EXPECT_TINY); then tampers fail exactly their checks:
  a. the TRUE run with the shuffled intent and ARGS -> 'domains.<d> prior / shuf / lam' and 'ptdom2 shuffled target' fail
  b. the shuffled run with ptdom's CHECK_PY and the true-target intent / ARGS -> the domains check and the args checks fail
  c. the shuffled run with --lam 2 in its ARGS -> the lambda-from-ARGS check fails
  d. shuf_cka_to_true 0.2 written into the shuffled run -> 'ptdom2 shuffled target' fails
  e. the shuffled run's geo_<d> made flat (no fit to the shuffled target) -> the CKA-rise bar fails (a flagged-valid check, A5.1)
-> check_py_shuf_test.out"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
T_ = B / "runs" / "ptdom2" / "tests"
sys.path.insert(0, str(B)); sys.path.insert(0, str(T_))
sys.argv = [sys.argv[0]]
import make_jobs_ptdom2 as M2   # noqa: E402
import shuf_trace as ST         # noqa: E402
MJ = M2.MJ
P = M2.load_P()
res = []
EXPECT_TINY = ("finished: ", "prior_events == ", "prior_on_frac == ", "GPU = ", "numbers token share", "domain lines per slot",
               "item-state CKA to the target rises", "decomposition double path")


def T(name, ok, info=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info != "" else ""), flush=True)


def run_check(cp, run_file, intent, args):
    with tempfile.TemporaryDirectory() as tmp:
        o = subprocess.run([M2.PY, "-", str(run_file), f"{tmp}/offline_lm.json", f"{tmp}/gpu.csv", str(M2.R / "code" / "COMMIT"),
                            str(M2.MJ.DATA), f"{tmp}/check.json", json.dumps(intent), args], input=cp, capture_output=True, text=True)
        ck = json.loads(Path(f"{tmp}/check.json").read_text()) if Path(f"{tmp}/check.json").exists() else None
    if ck is None:
        print(o.stdout[-2000:], o.stderr[-2000:])
    return ck


def tiny_intent(batch, arm, run, other):
    """The cell's ptdom2 intent with the tiny run's facts substituted (step-0 lm / data_sha / splits from the OTHER arm's run)."""
    ref0 = MJ.refs(batch)[0]
    I = M2.intent2(P, batch, arm, "1e-3", 0, ref0, save=False)
    c, a = run["curve"], run["args"]
    on = [e["step"] for e in c]
    I.update(steps=run["steps"], n_params=run["n_params"], eval_every_steps=run["eval_every_steps"], evals=75,
             geo_layers=run["geo_layers"], read_layers=run["read_layers"],
             events=[[8, 1], [64, 0]], on_frac=56 / 160, lm0_ref=other["curve"][0]["lm"], data_sha0=other["curve"][0]["data_sha"],
             dom={d: {k: other["domains"][d].get(k) for k in I["dom"][d]} for d in I["dom"]})
    for k in ("d_model", "n_layers", "n_heads", "d_ff", "seq", "bs", "tokens", "eval_every", "n_eval", "dom_eval_rows", "dom_seen",
              "anc_eval_max", "stop_after", "save_model", "ckpt_every", "lr", "tag"):
        if k in a:
            I["args"][k] = a[k]
        else:
            I["args"].pop(k, None)
    return I


for batch, d in (("B2", "compass"), ("B1", "countries"), ("B3", "scalar_adj"), ("A", "months")):
    dr = T_ / "shuf"
    rs_f = next((dr / f"{batch}-only-{d}-shuf").glob("run_*.json"))
    rt_f = next((dr / f"{batch}-only-{d}").glob("run_*.json"))
    rs, rt = json.loads(rs_f.read_text()), json.loads(rt_f.read_text())
    arm_s, arm_t = f"only-{d}-shuf", f"only-{d}"
    I_s, I_t = tiny_intent(batch, arm_s, rs, rt), tiny_intent(batch, arm_t, rt, rs)
    I_s["args"]["tag"], I_t["args"]["tag"] = rs["args"]["tag"], rt["args"]["tag"]
    A_s = ST.tiny_args(batch, arm_s, 1, rs["args"]["out"])
    A_t = ST.tiny_args(batch, arm_t, 1, rt["args"]["out"])
    ck = run_check(M2.CHECK_PY_SHUF, rs_f, I_s, A_s)
    fails = [k["check"] for k in ck["checks"] if not k["ok"]]
    # the testing-only --anc_eval_max 40 can leave a heuristic-hard subset empty among the first 40 held questions: allowed only
    # when every missing read-out of that check is a *_2way_hard key
    hard_only = lambda k: k["check"].startswith("choice read-outs at every eval") and k["value"] and all(v.endswith("_2way_hard") for v in k["value"])
    unexp = [x for x in fails if not x.startswith(EXPECT_TINY) and not any(hard_only(k) for k in ck["checks"] if k["check"] == x)]
    names = [k["check"] for k in ck["checks"]]
    shuf_named = [x for x in names if x.startswith(("ptdom2 shuffled", "domains.<d> prior / shuf / lam", "lambda per domain from the job's ARGS",
                                                   "run args = intent", "the job's ARGS"))]
    T(f"{d}: CHECK_PY_SHUF on the real tiny shuffled run: only the tiny / truncation checks fail; the shuffled-target, domains, "
      f"lambda and args checks ({len(shuf_named)}) pass", not unexp and len(shuf_named) == 6 and all(x not in fails for x in shuf_named),
      f"{len(names)} checks; failed {[x[:40] for x in fails]}; unexpected {unexp}")
    ck = run_check(M2.CHECK_PY_SHUF, rt_f, I_s, A_t)
    f2 = [k["check"] for k in ck["checks"] if not k["ok"]]
    T(f"{d} a: the TRUE run with the shuffled intent -> the domains and the shuffled-target checks fail",
      any(x.startswith("domains.<d> prior / shuf / lam") for x in f2) and any(x.startswith("ptdom2 shuffled target") for x in f2))
    ck = run_check(MJ.CHECK_PY, rs_f, I_t, A_s)
    f3 = [k["check"] for k in ck["checks"] if not k["ok"]]
    T(f"{d} b: the shuffled run under ptdom's CHECK_PY and the true intent -> the domains check and 'run args = intent' fail",
      any(x.startswith("domains.<d> prior / shuf / lam") for x in f3) and any(x.startswith("run args = intent") for x in f3))
    ck = run_check(M2.CHECK_PY_SHUF, rs_f, I_s, A_s.replace(" --lam 1", " --lam 2"))
    f4 = [k["check"] for k in ck["checks"] if not k["ok"]]
    T(f"{d} c: --lam 2 in the job's ARGS -> 'lambda per domain from the job's ARGS' fails", any(x.startswith("lambda per domain from the job's ARGS") for x in f4))
    with tempfile.TemporaryDirectory() as tmp:
        r2 = json.loads(rs_f.read_text())
        r2["domains"][d]["shuf_cka_to_true"] = 0.2
        f_ = Path(tmp) / rs_f.name
        f_.write_text(json.dumps(r2))
        ck = run_check(M2.CHECK_PY_SHUF, f_, I_s, A_s)
        f5 = [k["check"] for k in ck["checks"] if not k["ok"]]
        r3 = json.loads(rs_f.read_text())
        for e in r3["curve"]:
            if e.get(f"geo_{d}") is not None:
                e[f"geo_{d}"] = 0.7
        f_.write_text(json.dumps(r3))
        ck = run_check(M2.CHECK_PY_SHUF, f_, I_s, A_s)
        f6 = [k["check"] for k in ck["checks"] if not k["ok"]]
        rise = next(k["value"] for k in ck["checks"] if k["check"].startswith("item-state CKA to the target rises"))
    T(f"{d} d: shuf_cka_to_true 0.2 -> 'ptdom2 shuffled target' fails", any(x.startswith("ptdom2 shuffled target") for x in f5))
    T(f"{d} e: a flat geo_{d} (no fit to the shuffled target) -> the CKA-rise bar fails (rise {rise['rise']}); the true-target eval CKA "
      f"rise is reported beside it", any(x.startswith("item-state CKA to the target rises") for x in f6) and "true_target_eval_cka_rise" in rise)
print(f"CHECK_PY_SHUF TEST {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
