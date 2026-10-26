"""Known answers of the generator's intents (runs/pttpp/AUDIT.md "Tests"):
  intent7 with F87's 505M facts (3e8 tokens, eval every 250, --save_model, --ckpt_every 2000, F87's tag / out) = the args and
          facts of F87's real run JSONs (ptlr 500m none@9.6e-3 seed 0, win-ctx@1.2e-3 seed 1), field by field
  intent2 with pt2s505's facts = make_jobs_pt2s505.intent() exactly (none@4.8e-3 seed 4, win-ctx@6e-4 seed 2), and
  dry2() with 505M's model line = make_jobs_pt2s505.burst_dry_expected() for seeds 0-7
Prints INTENT-KA PASS / FAIL."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

G2 = T.g2()
fails = []
for arm, lr, s in (("none", "9.6e-3", 0), ("win-ctx", "1.2e-3", 1)):
    tag = f"500m-lr{lr}-7pct-{arm}"
    f = T.B / "runs/ptlr/out" / tag / f"run_whole_{'none' if arm == 'none' else 'helix3'}_seed{s}_{tag}.json"
    r = json.loads(f.read_text())
    F = T.facts(T.BRIDGE, arm, lr, s, "A")
    F.update(save=True, ckpt_every=2000, tag=tag)
    I = T.intent7(F)
    I["args"]["out"] = str(T.B / "runs/ptlr/out" / tag)
    diff = {k: (r["args"].get(k), v) for k, v in I["args"].items() if r["args"].get(k) != v}
    extra = sorted(set(r["args"]) - set(I["args"]))
    facts = {k: (r.get(k), I[k]) for k in ("steps", "n_params", "grad_ckpt_on", "geo_layers", "read_layers", "lm_chunk",
                                           "eval_every_steps", "code_sha") if r.get(k) != I[k]}
    ev = (r["prior_events"], I["events"]) if r["prior_events"] != I["events"] else None
    ok = not diff and not extra and not facts and ev is None and len(r["curve"]) == I["evals"]
    print(f"{'PASS' if ok else 'FAIL'} intent7 (F87 505M facts) = F87 run {f.name}: args diff {diff}, extra {extra}, facts {facts}, events {ev}")
    fails += [] if ok else [f.name]
for arm, lr, s in (("none", "4.8e-3", 4), ("win-ctx", "6e-4", 2)):
    F = T.facts("tpp2-190m-t0.745", arm, lr, s, "A")
    d, nl, nh, dff, n, nne = T.SIZES["505m"]
    F.update(dims=(d, nl, nh, dff), n_params=n, layers=[nl // 3, 2 * nl // 3], ckpt_on=True, S=18310, E=250, evals=75,
             T=3e8, events=T.window_events(18310)[0] if arm != "none" else [], on_frac=T.window_events(18310)[1] if arm != "none" else 0.0,
             ckpt_every=2000, tag=f"pt2s505-lr{lr}-{arm}", lm0=T.P2.LM0[("505m", "pretrain_data")][s])
    mine, theirs = T.intent2(G2, F), T.P2.intent(G2, arm, lr, s)
    ok = mine == theirs
    print(f"{'PASS' if ok else 'FAIL'} intent2 (pt2s505 facts) = make_jobs_pt2s505.intent({arm}, {lr}, {s})"
          + ("" if ok else f": {[k for k in theirs if mine.get(k) != theirs[k]]} / {[k for k in theirs['args'] if mine['args'].get(k) != theirs['args'][k]]}"))
    fails += [] if ok else [f"intent2 {arm}"]
saved = T.MODEL_LINE["294m"]
T.MODEL_LINE["294m"] = T.P2.MODEL_LINE   # dry2's emulation with 505M's model line must be pt2s505's
for s in range(8):
    ok = T.dry2(G2, "294m", s) == T.P2.burst_dry_expected(G2, s)
    fails += [] if ok else [f"dry2 s{s}"]
T.MODEL_LINE["294m"] = saved
print(f"{'PASS' if not [x for x in fails if x.startswith('dry2')] else 'FAIL'} dry2 with 505M's model line = make_jobs_pt2s505.burst_dry_expected, seeds 0-7")
print("INTENT-KA", "PASS" if not fails else f"FAIL {fails}")
