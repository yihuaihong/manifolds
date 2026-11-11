"""Known answer for the per-run check: CHECK7 (verbatim from make_jobs_pttpp) with this batch's intent form
(make_jobs_ptsz.facts -> intent_x -> make_jobs_pttpp.intent7: tokens 3e8, eval_every 250, 18310 steps, 75 evals, window
events [[916, 1], [7324, 0]], ckpt_every by size) PASSES on real 3e8-token 7% run JSONs when only the size / lr / seed / out /
tag / save / ckpt / lm0 facts are set to those runs' (L40S: F87 190M none (ckpt 2000), F87 505M win-ctx (ckpt 2000), pt300x
93M none stage C (ckpt 0); B200: pt300x 294M win-ctx C (ckpt 500)), and FAILS (rc 4) on planted intent errors: lr one rung
up, eval_every 26, tokens 31.3M, steps 18309, shifted window events, step-0 reference +0.002, ckpt_every 2000 -> 0, and a
d_model of the wrong ladder size. The CONSUMED line (it prints the run's area) is not shown.
    python tests/check7_ka.py"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_ptsz as G   # noqa: E402

M = G.M
RUNS = Path("/scratch/yh6210/llmadd_torch/runs")
CASES = [("190m", "none", "4.8e-3", 0, "ptlr/out/190m-lr4.8e-3-7pct-none", True, 2000),
         ("505m", "win-ctx", "1.2e-3", 1, "ptlr/out/500m-lr1.2e-3-7pct-win-ctx", True, 2000),
         ("93m", "none", "2.4e-3", 5, "pt300x/out/p3x-93m-lr2.4e-3-none", True, 0),
         ("294m", "win-ctx", "6e-4", 6, "pt300x/out/p3x-294m-lr6e-4-win-ctx", True, 500)]


def intent_for(size, arm, lr, seed, d, save, ck):
    """this batch's facts of a registered size and arm (stage C form), then the run's size / lr / seed / out / tag / save /
    ckpt / lm0 (no other field touched)"""
    F = dict(G.facts("391m", arm, G.GRID["391m"][arm][0], 4, "C"))
    dm, nl, nh, dff, n_params, nne = G.SIZES[size]
    F.update(lr=lr, seed=seed, dims=(dm, nl, nh, dff), n_params=n_params, nne=nne, layers=[nl // 3, 2 * nl // 3],
             ckpt_on=n_params > 150e6, lm0=G.lm0_ref(size, seed), tag=Path(d).name, save=save, ckpt_every=ck,
             tpp=F["T"] / nne, size=size)
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
        print(f"{'PASS' if rc == 0 else 'FAIL'} real {d} seed {seed} with the ptsz intent form: rc {rc}")
        for l in lines:
            print("    " + l[:200])
        for what, mut in (("lr one rung up", lambda J: J["args"].update(lr=J["args"]["lr"] * 2)),
                          ("eval_every 26", lambda J: J["args"].update(eval_every=26)),
                          ("tokens 31260672", lambda J: J["args"].update(tokens=31260672.0)),
                          ("steps 18309", lambda J: J.update(steps=18309)),
                          ("window events shifted", lambda J: J.update(events=[[915, 1], [7324, 0]]) if J["events"] else J.update(events=[[916, 1], [7324, 0]])),
                          ("step-0 lm reference +0.002", lambda J: J.update(lm0_ref=J["lm0_ref"] + 0.002)),
                          ("ckpt_every flipped (2000 <-> 0)", lambda J: J["args"].update(ckpt_every=0 if J["args"]["ckpt_every"] else 2000)),
                          ("d_model / n_params of another ladder size (731m)", lambda J: (J["args"].update(d_model=1152), J.update(n_params=G.SIZES["731m"][4])))):
            J = json.loads(json.dumps(I))
            mut(J)
            rc2, _ = check(run, J)
            n += 1; ok += rc2 == 4
            print(f"  {'PASS' if rc2 == 4 else 'FAIL'} planted '{what}' -> CHECK7 rc {rc2} (want 4)")
    print(f"CHECK7 KA {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
