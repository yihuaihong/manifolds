"""Progress of the ptsize999 runs: last eval step, fraction done, seconds per step (from the curve), ETA of the running runs."""
import glob, json, os, time

R = "/scratch/yh6210/llmadd_torch/runs/ptsize999/out"
rows = []
for f in sorted(glob.glob(f"{R}/*/run_*.json")):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    c = r["curve"]; last = c[-1]
    sps = (c[-1]["sec"] - c[0]["sec"]) / max(1, c[-1]["step"] - c[0]["step"]) if len(c) > 1 else float("nan")
    left_h = (r["steps"] - 1 - last["step"]) * sps / 3600
    age_min = (time.time() - os.path.getmtime(f)) / 60
    rows.append((os.path.basename(os.path.dirname(f)), r["args"]["seed"], last["step"], r["steps"], sps, left_h, age_min,
                 r.get("grad_ckpt_on"), r.get("n_params")))
for name, seed, st, steps, sps, left, age, gc, npar in rows:
    print(f"{name:22s} s{seed} step {st:5d}/{steps} ({100 * st / (steps - 1):5.1f}%) {sps:5.2f} s/step, ~{left:5.1f} h left, "
          f"file updated {age:5.0f} min ago, grad_ckpt {gc}, {npar / 1e6 if npar else 0:.0f}M")
