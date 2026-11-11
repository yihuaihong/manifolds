"""pt2sz test: known answers for the per-run check. make_pt2main_tasks.CHECK_PY (verbatim, as in every job) with an intent
built by THIS batch's builder (make_jobs_pt2sz.facts -> intent_of = make_jobs_pttpp.intent2 with the 3e8 facts) PASSES on
real finished 2% 3e8 runs of pt2same (53M, 190M) and pt2s505 (505M) when only the size / lr / seed / tag / step-0 reference
are set to those runs' (each run's own offline_lm.json, gpu.csv, the snapshot COMMIT and the data dir, as the job passes
them), and FAILS (rc 4) on planted intent errors: lr x2, eval_every 26, tokens 156.8M (the tpp line), steps 18309, the
window shifted by one step, the step-0 reference +0.002, arith_lines 4 (7%), d_model + 64, another code_sha, 74 evals.
    python check_ka.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pt2sz as M   # noqa: E402

G2 = M.g2()
RUNS = M.B / "runs"
sys.path.insert(0, str(RUNS / "pt2sz"))
import select_pt2sz as S   # noqa: E402  (ref_lm0: the step-0 references of the reused sizes)

CASES = [("190m", "none", "4.8e-3", 0, "pt2same/out/pt2same-190m-lr4.8e-3-none-s0", "pt2same-190m-lr4.8e-3-none"),
         ("53m", "win-ctx", "2e-3", 5, "pt2same/out/pt2same-53m-lr2e-3-win-ctx-s5", "pt2same-53m-lr2e-3-win-ctx"),
         ("505m", "win-ctx", "6e-4", 6, "pt2s505/out/pt2s505-lr6e-4-win-ctx-s6", "pt2s505-lr6e-4-win-ctx"),
         ("505m", "none", "9.6e-3", 1, "pt2s505/out/pt2s505-lr9.6e-3-none-s1", "pt2s505-lr9.6e-3-none")]
PLANT = [("lr one rung up", lambda J: J["args"].update(lr=J["args"]["lr"] * 2)),
         ("eval_every 26", lambda J: J["args"].update(eval_every=26)),
         ("tokens 156794880 (the tpp line)", lambda J: J["args"].update(tokens=156794880.0)),
         ("steps 18309", lambda J: J.update(steps=18309)),
         ("window events shifted", lambda J: J.update(events=[[915, 1], [7324, 0]]) if J["events"] else J.update(events=[[916, 1], [7324, 0]])),
         ("step-0 lm reference +0.002", lambda J: J.update(lm0_ref=J["lm0_ref"] + 0.002)),
         ("arith_lines 4 (7%)", lambda J: J["args"].update(arith_lines=4)),
         ("d_model + 64", lambda J: J["args"].update(d_model=J["args"]["d_model"] + 64)),
         ("code_sha a3bce3ab63c1", lambda J: J.update(code_sha="a3bce3ab63c1")),
         ("74 evals", lambda J: J.update(evals=74))]


def check(d, tag, arm, seed, I, out):
    t = "none" if arm == "none" else "helix3"
    run = RUNS / d / tag / f"run_whole_{t}_seed{seed}_{tag}.json"
    o = subprocess.run([M.PY, "-", str(run), str(RUNS / d / "offline_lm.json"), str(RUNS / d / "gpu.csv"), str(M.R / "code" / "COMMIT"),
                        str(M.DATA), str(out), json.dumps(I, separators=(",", ":"))], input=G2.CHECK_PY, capture_output=True, text=True)
    return o.returncode, [l for l in o.stdout.splitlines() if l.startswith(("PASS", "FAIL", "CHECKS"))]


def main():
    ok = n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for size, arm, lr, seed, d, tag in CASES:
            F = M.facts(size, arm, lr, seed, "C" if seed >= 4 else "A", G2, tag=tag, lm0=S.ref_lm0(size, seed))
            I = M.intent_of(G2, F)
            rc, lines = check(d, tag, arm, seed, I, Path(tmp) / "c.json")
            n += 1; ok += rc == 0
            print(f"{'PASS' if rc == 0 else 'FAIL'} real {d} with the pt2sz intent form: CHECK_PY rc {rc}")
            for l in lines:
                print("    " + l[:200])
            # the intent the submitted job embedded = this builder's (the known answer of the builder itself)
            job = (RUNS / d.split("/")[0] / f"job_{Path(d).name}.sbatch").read_text()
            import re
            emb = json.loads(re.findall(r"""^  "\$OUT/check\.json" '(\{.*\})' <<'PY'""", job, re.M)[0])
            n += 1; ok += emb == I
            print(f"  {'PASS' if emb == I else 'FAIL'} the intent embedded in the submitted job_{Path(d).name}.sbatch = this batch's builder's")
            for what, mut in PLANT:
                J = json.loads(json.dumps(I))
                mut(J)
                rc2, lines2 = check(d, tag, arm, seed, J, Path(tmp) / "c.json")
                n += 1; ok += rc2 == 4
                print(f"  {'PASS' if rc2 == 4 else 'FAIL'} planted '{what}' -> CHECK_PY rc {rc2} (want 4): "
                      f"{[l[5:90] for l in lines2 if l.startswith('FAIL')][:2]}")
    print(f"CHECK KA {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
