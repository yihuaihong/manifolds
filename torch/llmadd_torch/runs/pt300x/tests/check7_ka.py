"""Known answer for the per-run check: CHECK7 (verbatim from make_jobs_pttpp) with this batch's 3e8 intent form
(make_jobs_pt300x.facts -> make_jobs_pttpp.intent7: tokens 3e8, eval_every 250, 18310 steps, 75 evals, window events
[[916, 1], [7324, 0]]) PASSES on real F87 / ptlr_1b 3e8 run JSONs when only the size / lr / seed / out / tag / save / ckpt facts
are set to those runs' (L40S 190M none, L40S 505M win-ctx, B200 1.07B win-ctx), and FAILS on planted intent errors.
The CONSUMED line (it prints the run's area) is not shown.
    python tests/check7_ka.py"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt300x as G   # noqa: E402

M = G.M
RUNS = Path("/scratch/yh6210/llmadd_torch/runs")
CASES = [("190m", "none", "4.8e-3", 0, "ptlr/out/190m-lr4.8e-3-7pct-none", True, 2000),
         ("505m", "win-ctx", "1.2e-3", 1, "ptlr/out/500m-lr1.2e-3-7pct-win-ctx", True, 2000),
         ("1b", "win-ctx", "6e-4", 5, "ptlr_1b/out/1b-lr6e-4-7pct-win-ctx", True, 500)]


def intent_for(size, arm, lr, seed, d, save, ck):
    F = dict(G.facts("294m", arm, G.GRID["294m"][arm][0], 0, "A"))   # this batch's facts, then the run's size / lr / ...
    dm, nl, nh, dff, n_params, nne = M.SIZES[size]
    F.update(lr=lr, seed=seed, dims=(dm, nl, nh, dff), n_params=n_params, nne=nne, layers=[nl // 3, 2 * nl // 3],
             ckpt_on=n_params > 150e6, lm0=M.lm0_ref("7", size, seed), tag=Path(d).name, save=save, ckpt_every=ck,
             tpp=F["T"] / nne)
    I = M.intent7(F)
    I["args"]["out"] = str(RUNS / d)
    return I


def check(run, I):
    o = subprocess.run([G.PY, "-", str(run), "/dev/null", json.dumps(I, separators=(",", ":"))], input=M.CHECK7,
                       capture_output=True, text=True)
    return o.returncode, [l for l in o.stdout.splitlines() if not l.startswith("CONSUMED")]


def main():
    ok = n = 0
    for size, arm, lr, seed, d, save, ck in CASES:
        run = next((RUNS / d).glob(f"run_*_seed{seed}_*.json"))
        I = intent_for(size, arm, lr, seed, d, save, ck)
        rc, lines = check(run, I)
        n += 1; ok += rc == 0
        print(f"{'PASS' if rc == 0 else 'FAIL'} real {d} seed {seed} with the pt300x intent form: rc {rc}")
        for l in lines:
            print("    " + l[:200])
        for what, mut in (("lr one rung up", lambda J: J["args"].update(lr=J["args"]["lr"] * 2)),
                          ("eval_every 26", lambda J: J["args"].update(eval_every=26)),
                          ("tokens 31260672", lambda J: J["args"].update(tokens=31260672.0)),
                          ("steps 18309", lambda J: J.update(steps=18309)),
                          ("window events shifted", lambda J: J.update(events=[[915, 1], [7324, 0]]) if J["events"] else J.update(events=[[916, 1], [7324, 0]])),
                          ("step-0 lm reference +0.002", lambda J: J.update(lm0_ref=J["lm0_ref"] + 0.002))):
            J = json.loads(json.dumps(I))
            mut(J)
            rc2, _ = check(run, J)
            n += 1; ok += rc2 == 4
            print(f"  {'PASS' if rc2 == 4 else 'FAIL'} planted '{what}' -> CHECK7 rc {rc2} (want 4)")
    print(f"CHECK7 KA {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
