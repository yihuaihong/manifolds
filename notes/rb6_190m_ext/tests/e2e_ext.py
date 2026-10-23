"""rb6_190m_ext copy of scale_work/e2e.py (the rb6_190m end-to-end smoke): the rendered job script of
make_jobs_rb6_190m_ext.render (the batch's snapshot runs/rb6_190m_ext/code/tools; the arm table registered in memory as
the generator does), per cell type (only-clock, all6-minus-clock, all6-lam1@1.2e-3, only-numbers@1.2e-3, none@1.92e-2),
run end to end on CPU. Otherwise as e2e.py: the 190M rendering of (arm, lr, seed) with ONLY these substitutions, each asserted to
occur exactly once in the job text:
  - R / OUT -> a test dir (its code/ = the snapshot snap/new/tools + its COMMIT line), no #SBATCH effect (run with bash);
  - the trainer ARGS + tiny overrides (argparse keeps the last value): d64 x 3 layers, seq 64, 40 steps, evals every 10,
    smaller eval sets, --grad_ckpt on (the 190M / 505M state);
  - the model line, the DRY row count / hash -> the tiny dry run's (computed here first, by the job's own grep pipeline);
  - CHECK_PY's intent -> the tiny facts: steps, n_params, layers, eval spacing, evals, window events, token share, the
    step-0 lm reference (step0_ng.py on the tiny args: the CPU reference method itself), data_sha0, the domain splits,
    and the CKA-rise bar (a 40-step run cannot show the 53M bar; disabled: -1).
A fake nvidia-smi on PATH reports an A100 (this host has no GPU). Expected: rc 0 and every check PASS.
    python e2e.py <arm> <lr> <seed>"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path("/scratch/yh6210/llmadd_torch/scale_work")
PIN = Path("/scratch/yh6210/llmadd_torch/runs/rb6_190m_ext/code/tools")
sys.path.insert(0, str(PIN))
sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_rb6m_tasks as G   # noqa: E402
import make_rb6_tasks as P    # noqa: E402
import make_jobs_rb6_190m_ext as J   # noqa: E402
assert Path(G.__file__).resolve().parent == PIN.resolve()
J.register(G)
sys.argv.append("--no_refs")   # the CPU-reference gate of render() is tested separately (refs_ext.py); here the job body

TINY = ("--d_model 64 --n_layers 3 --n_heads 2 --d_ff 256 --seq 64 --tokens 81920 --eval_every 10 --n_eval 30 "
        "--dom_eval_rows 50 --dom_seen 30 --grad_ckpt on")
STEPS, EVERY = 40, 10
arm, lr, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
T = HERE / "rb190ext_e2e" / f"{arm}-lr{lr}-s{seed}"
subprocess.run(["rm", "-rf", str(T)], check=True)
(T / "code").mkdir(parents=True)
os.symlink(PIN, T / "code" / "tools")
(T / "code" / "COMMIT").write_text("e2e (runs/rb6_190m_ext/code/tools = runs/rb6_190m/code); sha256[:12]: " + ", ".join(
    f"{f} {J.sha(PIN / f)}" for f in (*J.WANT_SHA, *J.GEN)) + "\n")
fake = T / "bin"
fake.mkdir()
(fake / "nvidia-smi").write_text("""#!/bin/bash
case "$*" in
  *"-l "*) while true; do echo "2026/10/22 00:00:00.000, NVIDIA A100-SXM4-80GB, 97 %, 40 %, 30000 MiB, 81920 MiB"; sleep 60; done;;
  *noheader*) echo "NVIDIA A100-SXM4-80GB";;
  *) echo "name, driver_version, memory.total [MiB]"; echo "NVIDIA A100-SXM4-80GB, 0.0, 81920 MiB";;
esac
""")
(fake / "nvidia-smi").chmod(0o755)
ENV = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
ENV.pop("SLURM_JOB_ID", None)

# placeholders so render() runs (replaced below by the tiny values)
assert J.LM0["190m"][seed] and J.DRY["190m"][seed] == J.burst_dry_expected(G, P, "190m", seed)   # the production refs
name, tag, text, args, I = J.render(G, P, T, "190m", arm, lr, seed, J.WANT_SHA["pretrain_range.py"])


def sub(t, old, new):
    assert t.count(old) == 1, (t.count(old), old[:120])
    return t.replace(old, new)


# 1. the tiny dry run (the job's own trainer call + TINY) and the tiny step-0 reference
targs = args + " " + TINY
dlog = T / "tiny_dry.txt"
with open(dlog, "w") as fh:
    rc = subprocess.run([J.PY, "pretrain_range.py"] + (targs + " --dry_inputs").split(), cwd=PIN, env=ENV, stdout=fh,
                        stderr=subprocess.STDOUT).returncode
assert rc == 0, dlog.read_text()[-2000:]
g = P.DRY_GREP.replace('"', '\\"')
n = subprocess.run(["bash", "-c", f'set +o pipefail; grep -cE "{g}" "{dlog}"'], capture_output=True, text=True).stdout.strip()
h = subprocess.run(["bash", "-c", f'set +o pipefail; grep -E "{g}" "{dlog}" | sha256sum | cut -d" " -f1'],
                   capture_output=True, text=True).stdout.strip()
txt = dlog.read_text()
mline = [l for l in txt.splitlines() if l.startswith("model:")]
assert len(mline) == 1 and mline[0].endswith("gradient checkpointing on"), mline
doms = {}
for d in P.EXT:
    m = re.search(rf"^\[{d}\] (\{{.*\}})$", txt, re.M)
    doms[d] = {k: json.loads(m.group(1))[k] for k in P.DOM_KEYS if k in json.loads(m.group(1))}
slog = T / "tiny_step0.txt"
with open(slog, "w") as fh:
    subprocess.run([J.PY, str(HERE / "step0_ng.py"), str(PIN)] + (targs.replace(f"--out {T}/out/{name}/{tag}",
                   f"--out {T}/step0run")).split(), cwd=PIN, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
s0 = json.loads(re.search(r"^STEP0 (\{.*\})$", slog.read_text(), re.M).group(1))
print("tiny dry", n, h, mline[0], "| step0", s0, flush=True)

# 2. the job text with the tiny substitutions
I2 = json.loads(json.dumps(I))
ev, onf = P.window_events(STEPS)
I2.update(steps=STEPS, eval_every_steps=EVERY, evals=len(range(0, STEPS, EVERY)) + ((STEPS - 1) % EVERY != 0),
          events=ev if I["events"] else [], on_frac=onf if I["events"] else 0.0, geo_layers=[1, 2], read_layers=[1, 2],
          lm0_ref=s0["lm"], data_sha0=s0["data_sha"], dom=doms, numbers_share=[0.10, 0.25], cka_rise=-1,
          n_params=6615232)   # d64 x 3 layers, d_ff 256 (the regression's tiny runs; the run JSON is compared below)
I2["args"].update(d_model=64, n_layers=3, n_heads=2, d_ff=256, seq=64, tokens=81920.0, eval_every=EVERY, n_eval=30,
                  dom_eval_rows=50, dom_seen=30, grad_ckpt="on")
t = text
t = sub(t, f'ARGS="{args}"', f'ARGS="{targs}"')
t = sub(t, f"grep -qxF '{J.SIZES['190m']['model_line']}'", f"grep -qxF '{mline[0]}'")
dry, nd = J.DRY["190m"][seed]
t = sub(t, f'[ "$n" = {nd} ] && [ "$got" = {dry} ]', f'[ "$n" = {n} ] && [ "$got" = {h} ]')
t = sub(t, "'" + json.dumps(I, separators=(",", ":")) + "'", "'" + json.dumps(I2, separators=(",", ":")) + "'")
assert "#SBATCH --partition=a100_tandon" in t and f"R={T}" in t
(T / "job.sh").write_text(t)
# 3. run the job body
with open(T / "job.log", "w") as fh:
    rc = subprocess.run(["bash", str(T / "job.sh")], env=ENV, stdout=fh, stderr=subprocess.STDOUT).returncode
cj = json.loads((T / "out" / name / "check.json").read_text()) if (T / "out" / name / "check.json").exists() else None
run = next((T / "out" / name / tag).glob("run_*.json"), None)
rn = json.loads(run.read_text()) if run else {}
print(f"E2E {arm}@{lr} s{seed}: job rc {rc}; check {'PASS' if cj and cj['pass'] else 'FAIL'}; n_params {rn.get('n_params')}; "
      f"code_sha {rn.get('code_sha')} dep {rn.get('dep_sha')}; step-0 lm run {rn.get('curve', [{}])[0].get('lm')!r} vs "
      f"CPU reference {s0['lm']!r}")
for c in (cj or {}).get("checks", []):
    print(("  PASS " if c["ok"] else "  FAIL ") + c["check"][:150] + f"  ({str(c['value'])[:150]})")
print("E2E", "PASS" if rc == 0 and cj and cj["pass"] else "FAIL")
# second path on the run JSON itself: which priors were on / applied / logged / weighted, and the multiplier = 1
if rn:
    c_ = rn["curve"]
    act = J.ACTIVE[arm]
    on_ = [e for e in c_ if e.get("prior_on")]
    rep = {}
    for d in G.DOMS6:
        used = c_[-1].get(f"prior_used_frac_{d}", 0)
        geo_n = sum(e.get(f"geo_{d}") is not None for e in c_) if d != "numbers" else sum(e["geo"] != 0 for e in c_)
        gk = sum(f"gnorm_prior_{d}" in e for e in on_)
        rat = [e[f"gnorm_prior_{d}"] / e[f"gnorm_true_{d}"] for e in on_ if e.get(f"gnorm_prior_{d}") is not None
               and e.get(f"gnorm_true_{d}")]
        rep[d] = dict(prior=rn["domains"][d].get("prior"), lam=rn["lam_domains"][d], used=round(used, 4), geo_evals=geo_n,
                      gnorm_evals=gk, mult=[round(min(rat), 6), round(max(rat), 6)] if rat else None)
        want_on = d in act
        ok_d = (rep[d]["prior"] is want_on and rep[d]["lam"] == (1.0 if want_on else 0.0)
                and ((used > 0 and geo_n > 0 and gk > 0 and rat and all(abs(v - 1) < 1e-3 for v in rat)) if want_on
                     else (used == 0 and geo_n == 0 and gk == 0)))
        print(f"  RUN {d:8s} intended {'ON ' if want_on else 'OFF'}: {rep[d]} -> {'OK' if ok_d else 'WRONG'}")
        assert ok_d, (arm, d, rep[d])
    print(f"RUN PRIORS OK ({arm}: on = {','.join(act) or 'none'}; lambda per active prior 1; n on-evals {len(on_)})")
