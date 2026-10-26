"""End-to-end CPU run of rendered pttpp job scripts (make_jobs_pttpp.job7 / job2), the e2e_pt2.py method: the generator
renders into a test dir (T.R patched; code7 / code2 symlinked to the batch's snapshots), then ONLY these substitutions
(each asserted exactly once): ARGS + tiny overrides (d64 x 3, seq 64, 40 steps, evals every 10, 30 eval items; argparse
keeps the last flag); the intent JSON -> the tiny facts (steps, n_params, N_nonemb, layers, eval spacing, evals, window
events; no step-0 reference); 2% also the model line, '(len 512,' of the operand-site check and the DRY hash -> the tiny dry
run's. A fake nvidia-smi reports the job's GPU; maybe_select.sh is a stub that records its argument.
Cases: 7% L40S stage A (none), 7% B200 stage C (win-ctx, --save_model), 2% A100 stage A (win-ctx: model removed after the
check) and stage C (none: model kept); a planted intent error (steps 41) must give rc 4 with the hook still called.
    python e2e_tpp.py <case>        -> E2E <case> PASS / FAIL"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

REAL_R = T.R
G2 = T.g2()
case = sys.argv[1]
CASES = {"l40s-A-none": ("tpp7-190m-t0.745", "none", "4.8e-3", 1, "A", "NVIDIA L40S", False),
         "b200-C-win": ("tpp7-1b-t0.745", "win-ctx", "6e-4", 2, "C", "NVIDIA B200", False),
         "a100-A-win": ("tpp2-294m-t0.745", "win-ctx", "6e-4", 0, "A", "NVIDIA A100-SXM4-80GB", False),
         "a100-C-none": ("tpp2-190m-t0.745", "none", "4.8e-3", 5, "C", "NVIDIA A100-SXM4-80GB", False),
         "l40s-A-none-bad": ("tpp7-93m-t0.745", "none", "9.6e-3", 0, "A", "NVIDIA L40S", True)}
gid, arm, lr, seed, stage, gpu, plant = CASES[case]
X = HERE / "e2e_out" / case
subprocess.run(["rm", "-rf", str(X)], check=True)
X.mkdir(parents=True)
for c in ("code7", "code2"):
    os.symlink(REAL_R / c, X / c)
(X / "maybe_select.sh").write_text(f'#!/bin/bash\necho "$@" >> {X}/hook_called.txt\n')
fake = X / "bin"
fake.mkdir()
(fake / "nvidia-smi").write_text(f"""#!/bin/bash
case "$*" in
  *"-l "*) while true; do echo "2026/10/26 00:00:00.000, {gpu}, 97 %, 40 %, 30000 MiB, 81920 MiB"; sleep 60; done;;
  *noheader*) echo "{gpu}";;
  *) echo "name, driver_version, memory.total [MiB]"; echo "{gpu}, 0.0, 81920 MiB";;
esac
""")
(fake / "nvidia-smi").chmod(0o755)
ENV = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
ENV.pop("SLURM_JOB_ID", None)
T.R = X
F = T.facts(gid, arm, lr, seed, stage)
if F["setting"] == "2":
    OUT, text = T.job2(G2, F)
else:
    text = T.job7(F, slot=0 if F["hw"] == "b200" else None)
assert str(REAL_R) not in text, "the rendering still points at the real batch dir"


def sub(t, old, new):
    assert t.count(old) == 1, (t.count(old), old[:120])
    return t.replace(old, new)


TINY = "--d_model 64 --n_layers 3 --n_heads 2 --d_ff 256 --seq 64 --tokens 81920 --eval_every 10 --n_eval 30"
S, E = 40, 10
args = re.search(r'^ARGS="(.*)"$', text, re.M).group(1)
text = sub(text, f'ARGS="{args}"', f'ARGS="{args} {TINY}"')
tdir = X / "tiny_dry"
targs = (args + " " + TINY).replace(str(X / "out"), str(tdir)).replace(str(X / "out2"), str(tdir))
dlog = X / "tiny_dry.txt"
code = X / ("code2" if F["setting"] == "2" else "code7") / "tools"
with open(dlog, "w") as fh:
    assert subprocess.run([T.PY, "pretrain_range.py"] + (targs + " --dry_inputs").split(), cwd=code, env=ENV, stdout=fh,
                          stderr=subprocess.STDOUT).returncode == 0, dlog.read_text()[-2000:]
mline = [l for l in dlog.read_text().splitlines() if l.startswith("model:")]
assert len(mline) == 1, mline
n_params = int(round(float(mline[0].split()[1][:-1]) * 1e6))   # model: 0.1M parameters -> approximate; exact below
import torch  # noqa: E402
from transformers import Olmo2Config, Olmo2ForCausalLM  # noqa: E402
with torch.device("meta"):
    m = Olmo2ForCausalLM(Olmo2Config(vocab_size=T.V, hidden_size=64, intermediate_size=256, num_hidden_layers=3,
                                     num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True))
n_params = sum(p.numel() for p in m.parameters())
ev, on = T.window_events(S) if arm != "none" else ([], 0.0)
evals = T.n_evals(S, E)
if F["setting"] == "7":
    I = T.intent7(F)
    I2 = dict(I, steps=41 if plant else S, n_params=n_params, n_nonemb=n_params - T.V * 64, grad_ckpt_on=False, geo_layers=[1, 2],
              read_layers=[1, 2], eval_every_steps=E, evals=evals, events=ev, on_frac=on, lm0_ref=None,
              args=dict(I["args"], d_model=64, n_layers=3, n_heads=2, d_ff=256, seq=64, tokens=81920.0, eval_every=E, n_eval=30))
    ij, ij2 = (json.dumps(v, separators=(",", ":")) for v in (I, I2))
    text = sub(text, f"'{ij}'", f"'{ij2}'")
else:
    I = T.intent2(G2, F)
    I2 = dict(I, steps=S, n_params=n_params, grad_ckpt_on=False, geo_layers=[1, 2], read_layers=[1, 2], eval_every_steps=E,
              evals=evals, events=ev, on_frac=on, lm0_ref=None,
              args=dict(I["args"], d_model=64, n_layers=3, n_heads=2, d_ff=256, seq=64, tokens=81920.0, eval_every=E, n_eval=30))
    ij, ij2 = (json.dumps(v, separators=(",", ":")) for v in (I, I2))
    text = sub(text, f"'{ij}'", f"'{ij2}'")
    text = sub(text, f"grep -qxF '{T.MODEL_LINE[F['size']]}'", f"grep -qxF '{mline[0]}'")
    text = sub(text, "'^train row [012] (len 512, 2 prior sites at '", "'^train row [012] (len 64, 2 prior sites at '")
    g = G2.DRY_GREP.replace('"', '\\"')
    h = subprocess.run(["bash", "-c", f'set +o pipefail; grep -E "{g}" "{dlog}" | head -n 9 | sha256sum | cut -d" " -f1'],
                       capture_output=True, text=True).stdout.strip()
    text = sub(text, f'[ "$got" = {T.dry2(G2, F["size"], seed)} ]', f'[ "$got" = {h} ]')
job = X / "job.sh"
job.write_text(text)
log = X / "job.log"
with open(log, "w") as fh:
    rc = subprocess.run(["bash", str(job)], env=ENV, stdout=fh, stderr=subprocess.STDOUT).returncode
txt = log.read_text()
hook = (X / "hook_called.txt").read_text().split() if (X / "hook_called.txt").exists() else []
fails = []
want_rc = 4 if plant else 0
if rc != want_rc:
    fails.append(f"rc {rc} (want {want_rc})")
if hook != [gid]:
    fails.append(f"hook called with {hook} (want [{gid}])")
if F["setting"] == "7":
    ck = json.loads((X / "out" / F["tag"] / f"check_s{seed}.json").read_text())
    if ck["pass"] == plant:
        fails.append(f"check pass {ck['pass']}")
    if "CONSUMED" not in txt or "INTENT" not in txt:
        fails.append("no INTENT / CONSUMED line")
    cons = json.loads(re.search(r"^CONSUMED (\{.*\})$", txt, re.M).group(1))
    if not plant and not (cons["steps"] == S and cons["eval_every_steps"] == E and cons["evals"] == evals and cons["tokens_consumed"] == S * 2048):
        fails.append(f"consumed values {cons}")
    model = list((X / "out" / F["tag"]).glob("model_*.pt"))
    if bool(model) != (stage == "C"):
        fails.append(f"model saved {bool(model)} at stage {stage}")
else:
    ck = json.loads((OUT / "check.json").read_text())
    if not ck["pass"]:
        fails.append("2% check FAILED: " + str([c["check"] for c in ck["checks"] if not c["ok"]]))
    model = list((OUT / F["tag"]).glob("model_*.pt"))
    if bool(model) != (seed >= 4):
        fails.append(f"model present {bool(model)} at seed {seed} (stage A / X removes it after the check)")
    if "## model removed" in txt and seed >= 4:
        fails.append("model removed at a confirm seed")
print(txt[-3000:])
print(f"E2E {case} {'PASS' if not fails else 'FAIL'}" + (f": {fails}" if fails else "") + f" (rc {rc}, hook {hook})")
sys.exit(1 if fails else 0)
