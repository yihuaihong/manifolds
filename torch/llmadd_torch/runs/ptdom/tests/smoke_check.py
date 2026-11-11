"""Checks of the 200-step CPU smoke (smoke_200.sh): the prior acts (its loss falls over the window), is exactly off for
the none arm, and both arms see the same data."""
import glob
import json
import sys
from pathlib import Path

W = Path(sys.argv[1])
L = lambda a: json.loads(Path(glob.glob(str(W / a / "run_*.json"))[0]).read_text())
s, n = L("stars"), L("none")
cs, cn = s["curve"], n["curve"]
res = []


def T(name, cond, val=None):
    res.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val is not None else ""))


T("both finished: 11 evals at steps 0, 20, .., 180, 199", [e["step"] for e in cs] == [e["step"] for e in cn] == list(range(0, 200, 20)) + [199])
T("the same data in both arms: data_sha at every eval and the step-0 lm", [e["data_sha"] for e in cs] == [e["data_sha"] for e in cn]
  and cs[0]["lm"] == cn[0]["lm"], cs[0]["lm"])
on = [e for e in cs if e["prior_on"]]
T("prior arm: window steps 10..79 (events [[10, 1], [80, 0]]), evals 20 / 40 / 60 in it", s["prior_events"] == [[10, 1], [80, 0]]
  and [e["step"] for e in on] == [20, 40, 60], s["prior_events"])
g = [e["geo_stars"] for e in cs]
# the prior loss = 1 - CKA(item states, target Gram). On the training batch (geo_stars, ~8 sites at bs 8) it is noisy, so the
# decrease is read on the FIXED eval prompts (the same quantity, 3 sites x 60 questions, the same rows at every eval)
pl = [1 - e["cka_stars"] for e in cs]
pn = [1 - e["cka_stars"] for e in cn]
T("prior arm: the prior loss on the fixed eval prompts (1 - cka_stars) falls over the window: at the window's last eval "
  "(step 60) and at step 80 below step 0 by >= 0.05", pl[3] <= pl[0] - 0.05 and pl[4] <= pl[0] - 0.05, [round(v, 4) for v in pl[:6]])
T("prior arm vs none (same data): the prior loss on the eval prompts lower than none's at every eval from step 40 on, by >= 0.05",
  all(pl[i] <= pn[i] - 0.05 for i in range(2, len(pl))), [round(b - a, 3) for a, b in zip(pl, pn)])
T("prior arm: the training-batch prior loss geo_stars, mean over the on-evals < its step-0 value (noisy per batch)",
  sum(e["geo_stars"] for e in on) / len(on) < g[0], [round(v, 4) for v in g[:5]])
T("prior arm: gnorm_prior_stars / gnorm_true_stars = lambda 1 at every on-eval",
  all(abs(e["gnorm_prior_stars"] / e["gnorm_true_stars"] - 1) < 1e-3 for e in on), [e["gnorm_prior_stars"] for e in on])
T("prior arm: LM share < 1 at the on-evals, = 1 off; decomposition = backward (reldiff < 0.05)",
  all(e["grad_lm_share"] < 1 for e in on) and all(e["grad_lm_share"] == 1.0 for e in cs if not e["prior_on"])
  and all(e["grad_T_reldiff"] < 0.05 for e in cs), [round(e["grad_lm_share"], 4) for e in on])
T("prior arm: applied on every on-step, no other prior ever", abs(cs[-1]["prior_used_frac_stars"] - 70 / 200) < 1e-9
  and all(cs[-1][f"prior_used_frac_{d}"] == 0 for d in ("numbers", "months", "landmarks", "languages", "colours", "cities"))
  and all(not any(k.startswith("gnorm_prior_") and k != "gnorm_prior_stars" for k in e) for e in cs), cs[-1]["prior_used_frac_stars"])
T("none arm: no prior ever (prior_on 0, geo == 0 exactly, no geo_* / gnorm_prior_* keys, LM share 1, events [])",
  n["prior_events"] == [] and all(e["prior_on"] == 0 and e["geo"] == 0 and e["grad_lm_share"] == 1.0
                                  and not any(k.startswith(("geo_", "gnorm_prior_")) for k in e) for e in cn)
  and all(cn[-1][f"prior_used_frac_{d}"] == 0 for d in ("numbers", "months", "stars", "landmarks", "languages", "colours", "cities")))
T("none arm: lam_domains all 0; prior arm: stars 1, the rest 0", set(n["lam_domains"].values()) == {0.0}
  and s["lam_domains"]["stars"] == 1.0 and sum(s["lam_domains"].values()) == 1.0)
T("Rev E read-outs logged in both arms (2-way / k-way, ties, double path < 2e-2 on CPU)",
  all(isinstance(e.get("stars_held_2way"), float) and isinstance(e.get("months_held_kway"), float)
      and e["lp_dp_maxdiff_stars_held"] < 2e-2 for e in cs + cn))
T("anc_eval_max 40: 40 held questions per anchor domain evaluated, both orders", all(
  s["domains"][d]["eval_held"]["questions"] == 40 and s["domains"][d]["n_readout"] == 40 for d in ("stars", "landmarks", "languages", "colours", "cities")))
print("prior arm geo_stars by eval:", [round(v, 4) for v in g])
print("SMOKE", "PASS" if all(res) else "FAIL", f"({sum(res)}/{len(res)})")
