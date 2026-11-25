#!/usr/bin/env python3
"""pt6v2-size (post-hoc, exploratory, user-approved 2026-11-24 "可以的 没问题 尽快就行"; exp-builder): the Table 1 six-prior
set (pt6v2: numbers, dim, cities, countries, torus, box5) at 190M / 505M / 1.07B with the registered 53M recipe; seed 16 only.

Each job = the A100 render of a registered pt6v2 job of seed 16 (make_jobs_pt6v2.render_v2, frozen; V8's method,
runs/pt6v2/posthoc_v8_jobs.py), REGISTERED TRAINER UNCHANGED (runs/pt6v2/code, a398fccbfe93), with counted edits only:
  1. architecture (the ladder shapes: ptlr 190m / 500m, ptlr_1b): ARGS += --d_model D --n_layers L --n_heads H --d_ff F;
     grad_ckpt / lm_chunk left at the trainer's auto (as the ladders: ckpt on above 150M, lm chunk off at bs 32 x 512);
  2. lr: --lr and the tag pt6v2-<arm>-lr<lr>; base renders all6@1e-3, only-dim@2e-3, none@2e-3 (only the lr / tag differ);
  3. all6full = all6 with --window 0.05,1.0 (as V4: prior on from step 916 through step 18309);
  4. --save_model on every main (HF release); --ckpt_every 500 + --requeue + --open-mode=append (registered / V8);
  5. dry gate: the model line and the DRY-rows reference (n, sha256) from the CPU dry run of THIS job's ARGS (DRY_REF.json);
  6. intent JSON (CHECK_PY): n_params, grad ckpt, geo / read layers, lr, window events, save_model, GPU set; CHECK_PY is
     LOG AND FLAG ONLY (size-dependent expectations, e.g. lm0_ref, must not fail a run), as V7b / V8;
  7. device A100 / H100 / H200 (+ L40S for 190M); out dir runs/pt6v2_size/<size>/<arm>-lr<lr>-s16; Slurm name
     llmadd_pt6sz-<size>-<arm>-lr<lr>-s16; V8's early run guard; one provenance line.
Canary per size: all6@6e-4, 300 steps (--tokens 4915200; window events 15 on / 120 off = the early drop), eval every 50,
ckpt every 100, --save_model (exercises the save path); step time = run JSON curve sec, peak memory = gpu.csv.
Modes: (none) = build and list; --dry-script (write the CPU dry runner); --collect (CPU dry outputs -> DRY_REF.json, then
run every job's own dry gate lines on its CPU output); --write; --test-only; --submit (1.07B first; SUBMITTED_<name> flags)."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs" / "pt6v2"
S = B / "runs" / "pt6v2_size"
sys.path.insert(0, str(R))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import posthoc_v7_jobs as P7   # noqa: E402  (read only: its import-time asserts pin every frozen file)
sys.argv = _argv
MJ2, sha, edit = P7.MJ2, P7.sha, P7.edit

TR = P7.TR_OLD
assert sha(R / "code/tools/pretrain_range.py") == TR
SEED, STEPS, BS_SEQ, V = 16, 18310, 32 * 512, 100278
NONE_A = ("2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2")
NONE_B = ("1.2e-3", "2.4e-3", "4.8e-3", "9.6e-3")
PART_MAIN = "a100_tandon,h100_tandon"   # h200_tandon: --test-only refuses <slurm-account> (accepts only _advanced + gpu48)
SIZES = {   # ladder shapes (make_jobs_ptlr.py 190m / 500m, make_jobs_ptlr1b.py MODEL)
    "1b": dict(d=1280, L=36, H=20, ff=5120, time="20:00:00", mem="128G", none=NONE_B, part=PART_MAIN, gpus=("A100", "H100", "H200")),
    "505m": dict(d=1024, L=24, H=16, ff=4096, time="10:00:00", mem="96G", none=NONE_A, part=PART_MAIN, gpus=("A100", "H100", "H200")),
    "190m": dict(d=768, L=12, H=12, ff=3072, time="06:00:00", mem="64G", none=NONE_A, part=PART_MAIN + ",l40s_public",
                 gpus=("A100", "H100", "H200", "L40S")),
}
BASE = {"all6": ("all6", "1e-3"), "all6full": ("all6", "1e-3"), "only-dim": ("only-dim", "2e-3"), "none": ("none", "2e-3")}
P_LR = "6e-4"
CAN_STEPS = 300
DRY_REF = S / "DRY_REF.json"
MANI = S / "SUBMITTED.json"
PROV = ("# pt6v2-size (post-hoc, exploratory, 2026-11-24): rendered by runs/pt6v2_size/make_pt6sz_jobs.py = the registered {barm}@{blr} "
        "seed-{seed} job (A100 render) + {size} ({d}x{L}, {H} heads, d_ff {ff}) + --lr {lr}{extra} + --save_model; registered trainer "
        "runs/pt6v2/code unchanged; CHECK_PY log and flag only; out dir runs/pt6v2_size/{size}/. The 53M wording in the header "
        "above describes the parent job.\n")


def n_params(d, L, ff):
    """the trainer's count (tied embedding once): V d + L (4 d^2 + 3 d ff + 4 d) + d; = 52,672,128 at 53M, 1,072,259,840 at 1.07B."""
    return V * d + L * (4 * d * d + 3 * d * ff + 4 * d) + d


assert n_params(384, 6, 1536) == 52672128 and n_params(1280, 36, 5120) == 1072259840


def events(steps, win):
    """the trainer's prior_on(s) = w0 * steps <= s < w1 * steps -> toggle events, on_frac."""
    w0, w1 = (float(v) for v in win.split(","))
    on = [s for s in range(steps) if w0 * steps <= s < w1 * steps]
    ev = [[on[0], 1]] + ([[on[-1] + 1, 0]] if on[-1] + 1 < steps else [])
    return ev, len(on) / steps


def cells(size):
    z = SIZES[size]
    out = [("all6", P_LR, False), ("all6full", P_LR, False), ("only-dim", P_LR, False)] + [("none", lr, False) for lr in z["none"]]
    return [("all6", P_LR, True)] + out   # canary first within a size


def build(dry=None, seed=SEED, plan=None):
    """seed / plan (2026-11-24 REP 505M, make_pt6sz_rep.py): plan = {size: [(arm, lr, canary), ...]} replaces cells(size) and
    skips sizes not in it; the defaults (seed 16, plan None) render the registered 24 jobs byte-identically."""
    doms = MJ2.decided_comp()
    assert MJ2.manifolds(doms) == ("numbers", "dim", "cities", "countries", "torus", "box5"), doms
    commit = MJ2.snapshot()
    P = MJ2.load_P()
    RF, _ = MJ2.refs_v2(doms)
    base = {}
    for barm, blr in set(BASE.values()):
        name, tag, ta, _, _ = MJ2.render_v2(P, doms, barm, blr, seed, RF[seed], commit, "a100", None)
        reg = R / f"job_pt6v2-{barm}-lr{blr}-s{seed}.sbatch"
        assert P7.args_line(ta) == P7.args_line(reg.read_text()), f"{name}: A100 render ARGS != registered ARGS"
        base[(barm, blr)] = (name, tag, ta, reg)
    jobs = []
    for size in SIZES:
        z = SIZES[size]
        npar = n_params(z["d"], z["L"], z["ff"])
        mline = f"model: {npar / 1e6:.1f}M parameters; gradient checkpointing {'on' if npar > 150e6 else 'off'}"
        gl = [z["L"] // 3, 2 * z["L"] // 3]
        for arm, lr, can in (cells(size) if plan is None else plan.get(size, ())):
            barm, blr = BASE[arm]
            bname, btag, ta, reg = base[(barm, blr)]
            tag = f"pt6v2-{arm}-lr{lr}"
            rname = f"canary-s{seed}" if can else f"{arm}-lr{lr}-s{seed}"
            out = S / size / rname
            jn = f"llmadd_pt6sz-{size}-canary-s{seed}" if can else f"llmadd_pt6sz-{size}-{arm}-lr{lr}-s{seed}"
            win = "0.05,1.0" if arm == "all6full" else (None if arm == "none" else "0.05,0.4")
            steps = CAN_STEPS if can else STEPS
            t = edit(ta, f"#SBATCH --job-name=llmadd_{bname}\n", f"#SBATCH --job-name={jn}\n")
            t = edit(t, "#SBATCH --partition=a100_tandon\n", f"#SBATCH --partition={z['part']}\n")
            t = edit(t, "#SBATCH --time=03:00:00\n", f"#SBATCH --time={'01:30:00' if can else z['time']}\n")
            t = edit(t, "#SBATCH --mem=64G\n", f"#SBATCH --mem={z['mem']}\n")
            t = edit(t, "#SBATCH --requeue\n", "#SBATCH --requeue\n#SBATCH --open-mode=append\n")
            alts = "|".join(f"*{g}*" for g in z["gpus"])
            t = edit(t, 'case "$GPU" in *A100*) ;;', f'case "$GPU" in {alts}) ;;')
            t = edit(t, f'echo "DEVICE a100 (seed {seed}\'s allocated GPU type)"\n',
                     f'echo "DEVICE $GPU (pt6v2-size: {"/".join(z["gpus"])} allowed; mixed devices are a flag in the read)"\n')
            t = edit(t, '"gpu":"A100"', f'"gpu":"{"|".join(z["gpus"])}"')
            t = edit(t, 'all(I["gpu"] in g for g in gname)', 'all(any(x in g for x in I["gpu"].split("|")) for g in gname)')
            t = edit(t, f"{R}/out/{bname}", str(out), 2)                                   # OUT= and --out
            n_tag = t.count(btag)
            assert n_tag >= 4, (bname, n_tag)
            t = t.replace(btag, tag)
            al = P7.args_line(t)
            new = edit(al, f" --lr {blr} ", f" --lr {lr} ")
            if arm == "all6full":
                new = edit(new, " --window 0.05,0.4 ", " --window 0.05,1.0 ")
            ck = "100" if can else "500"
            tail = (f" --ckpt_every {ck}" + (f" --tokens {CAN_STEPS * BS_SEQ} --eval_every 50" if can else "")
                    + f" --d_model {z['d']} --n_layers {z['L']} --n_heads {z['H']} --d_ff {z['ff']} --save_model\"")
            new = edit(new, ' --ckpt_every 500"', tail)
            t = edit(t, al + "\n", new + "\n")
            assert "ARGS_DRY=" not in t
            t = edit(t, "grep -qxF 'model: 52.7M parameters; gradient checkpointing off'", f"grep -qxF '{mline}'")
            m = re.search(r'\[ "\$n" = (\d+) \] && \[ "\$got" = ([0-9a-f]{64}) \]', t)
            assert m, bname
            m2 = re.search(r'echo "## dry_inputs rows = the reference \(([0-9a-f]{12}), (\d+) lines\)"', t)
            assert m2 and m2.group(1) == m.group(2)[:12] and m2.group(2) == m.group(1), bname
            dr = (dry or {}).get(f"{size}/{rname}")
            dn, dsha = (str(dr["n"]), dr["sha"]) if dr else ("PENDING", "0" * 64)
            t = edit(t, m.group(0), f'[ "$n" = {dn} ] && [ "$got" = {dsha} ]')
            t = edit(t, m2.group(0), f'echo "## dry_inputs rows = the reference ({dsha[:12]}, {dn} lines)"')
            old_ln = P7.intent_line(t)
            js = old_ln.split("'")[1]
            I = json.loads(js)
            assert json.dumps(I, separators=(",", ":"), ensure_ascii=False) == js
            assert I["code_sha"] == TR and I["steps"] == STEPS and I["n_params"] == 52672128 and I["geo_layers"] == [2, 4]
            assert I["args"]["lr"] == float(blr) and I["args"]["d_model"] == 384 and I["save_model"] is False
            I["steps"] = steps
            I["evals"] = len([x for x in range(steps) if x % (50 if can else 250) == 0 or x == steps - 1])
            I["eval_every_steps"] = 50 if can else 250
            I["n_params"], I["grad_ckpt_on"], I["geo_layers"], I["read_layers"] = npar, npar > 150e6, gl, gl
            I["lm_chunk"] = 8192 if BS_SEQ * V * 4 > 8e9 else 0
            I["save_model"] = True
            I["args"].update(lr=float(lr), d_model=z["d"], n_layers=z["L"], n_heads=z["H"], d_ff=z["ff"], save_model=True,
                             ckpt_every=int(ck))
            if can:
                I["args"].update(tokens=float(CAN_STEPS * BS_SEQ), eval_every=50)
            if win:
                ev, frac = events(steps, win)
                if not can and arm != "all6full":
                    assert ev == I["events"] == [[916, 1], [7324, 0]], (ev, I["events"])
                I["args"]["window"], I["events"], I["on_frac"] = win, ev, frac
            else:
                assert I["events"] == [] and I["on_frac"] == 0.0
                ev = []
            new_ln = old_ln.replace(js, json.dumps(I, separators=(",", ":"), ensure_ascii=False))
            assert new_ln.endswith(""""$ARGS" <<'PY' | tee "$OUT/check.txt\"""")
            new_ln += (' || echo "## FLAG pt6v2-size: CHECK_PY (the registered 53M intent checks, size-adapted) failed: '
                       '$OUT/check.json (log and flag only)"')
            t = edit(t, old_ln, new_ln)
            rj = P7.run_json(new, seed)
            t = edit(t, 'mkdir -p "$OUT"\n', P7.guard_block(rj).replace("V7 run guard", "pt6sz run guard").replace("# V7 early", "# pt6sz early")
                     + 'mkdir -p "$OUT"\n')
            extra = (", --window 0.05,1.0" if arm == "all6full" else "") + (f", CANARY {CAN_STEPS} steps" if can else "")
            t = edit(t, "set -euo pipefail\n", PROV.format(barm=barm, blr=blr, size=size, lr=lr, extra=extra, seed=seed, **z) + "set -euo pipefail\n")
            assert subprocess.run(["bash", "-n"], input=t.encode()).returncode == 0, f"{jn}: bash -n"
            assert "#SBATCH --account=<slurm-account>\n" in t and "b200" not in t.split("set -euo")[0].lower()
            assert f"{R}/out/" not in t and 'cd "$R/code/tools"\n' in t and f'= "{TR}" ]' in t
            jobs.append({"size": size, "arm": arm, "lr": lr, "canary": can, "name": f"{size}/{rname}", "slurm_name": jn,
                         "text": t, "job": S / size / f"job_{rname}.sbatch", "flag": S / size / f"SUBMITTED_{rname}",
                         "out": str(out), "tag": tag, "run_json": f"{out}/{tag}/{rj}", "args": new.split('"')[1],
                         "model_line": mline, "n_params": npar, "events": ev, "steps": steps, "window": win,
                         "base_job": reg.name, "dry_ok": dr is not None})
    assert len({j["slurm_name"] for j in jobs}) == len({j["out"] for j in jobs}) == len(jobs) == (24 if plan is None else sum(len(v) for v in plan.values()))
    live = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True).stdout.split()
    clash = [j["slurm_name"] for j in jobs if j["slurm_name"] in live and not j["flag"].exists()]
    assert not clash, f"a live job has this name: {clash}"
    for j in jobs:
        o = Path(j["out"])
        if not j["flag"].exists():
            assert not (o.exists() and any(o.rglob("run_whole_*.json"))), f"{o}: a run JSON exists before submission"
    return commit, jobs


DRYD = S / "dry"


def dry_script(jobs):
    """one CPU job: --dry_inputs of every job's exact ARGS (4 at a time), outputs in runs/pt6v2_size/dry/<size>_<run>.txt."""
    lines = ["#!/bin/bash", "#SBATCH --account=<slurm-account>", "#SBATCH --partition=cpu_short", "#SBATCH --cpus-per-task=16",
             "#SBATCH --mem=96G", "#SBATCH --time=01:00:00", "#SBATCH --job-name=llmadd_pt6sz-dry",
             f"#SBATCH --output={DRYD}/dry_%j.log", "set -u", "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
             f"export HF_HOME={B}/hf_cache PYTHONPATH={B}/pylib", f"cd {R}/code/tools", "PY=/scratch/yh6210/miniconda3/envs/open_r1/bin/python",
             'echo "start $(date -u +%FT%TZ) $(hostname)"', "i=0"]
    for j in jobs:
        f = DRYD / (j["name"].replace("/", "_") + ".txt")
        lines.append(f'( "$PY" pretrain_range.py {j["args"]} --dry_inputs > {f} 2>&1; echo "rc=$?" >> {f} ) &')
        lines.append("i=$((i+1)); [ $((i % 4)) = 0 ] && wait")
    lines += ["wait", 'echo "all rc=0 done $(date -u +%FT%TZ)"']
    return "\n".join(lines) + "\n"


def gate_lines(t):
    """the job's own dry gate: from the line after 'cat "$OUT/dry_inputs.txt"' to the reference echo (inclusive)."""
    a = t.index('cat "$OUT/dry_inputs.txt"\n') + len('cat "$OUT/dry_inputs.txt"\n')
    b = t.index('echo "## dry_inputs rows = the reference')
    b = t.index("\n", b) + 1
    return t[a:b]


def collect(jobs):
    ref = {}
    for j in jobs:
        f = DRYD / (j["name"].replace("/", "_") + ".txt")
        txt = f.read_text() if f.exists() else ""
        assert txt.rstrip().endswith("rc=0"), f"{f}: dry not finished / failed: {txt[-500:]}"
        pat = re.search(r'n=\$\(set \+o pipefail; grep -cE "([^"]+)" "\$OUT/dry_inputs.txt"\)', j["text"]).group(1)
        body = "\n".join(x for x in txt.splitlines() if x != "rc=0") + "\n"
        tmp = DRYD / "_body.txt"
        tmp.write_text(body)
        g = subprocess.run(["grep", "-E", pat, str(tmp)], capture_output=True, text=True).stdout
        ref[j["name"]] = {"n": len(g.splitlines()), "sha": hashlib.sha256(g.encode()).hexdigest(),
                          "model_line": next((x for x in body.splitlines() if x.startswith("model: ")), None)}
        assert ref[j["name"]]["model_line"] == j["model_line"], (j["name"], ref[j["name"]]["model_line"], j["model_line"])
    DRY_REF.write_text(json.dumps(ref, indent=1) + "\n")
    _, jobs2 = build(ref)
    bad = 0
    for j in jobs2:
        f = DRYD / (j["name"].replace("/", "_") + ".txt")
        od = DRYD / "_gate" / j["name"].replace("/", "_")
        od.mkdir(parents=True, exist_ok=True)
        (od / "dry_inputs.txt").write_text("\n".join(x for x in f.read_text().splitlines() if x != "rc=0") + "\n")
        o = subprocess.run(["bash", "-c", "set -euo pipefail\nOUT=" + str(od) + "\n" + gate_lines(j["text"])], capture_output=True, text=True)
        ok = o.returncode == 0
        bad += not ok
        print(f"{'GATE OK ' if ok else 'GATE BAD'} {j['name']:28s} {ref[j['name']]['n']} lines {ref[j['name']]['sha'][:12]} | "
              f"{j['model_line']} | lr {j['lr']} window {j['window']} events {j['events']} | {(o.stdout + o.stderr).strip()[-160:]}")
    print("collect:", "ALL GATES PASS" if not bad else f"{bad} GATES FAIL")
    return bad == 0


def main():
    dry = json.loads(DRY_REF.read_text()) if DRY_REF.exists() else None
    commit, jobs = build(dry)
    for j in jobs:
        st = "SUBMITTED" if j["flag"].exists() else "on disk" if j["job"].exists() else "not written"
        print(f"{j['name']:26s} {j['slurm_name']:40s} steps {j['steps']:5d} ev {j['events']} dry {'ok' if j['dry_ok'] else 'PENDING'} {st}")
    if "--dry-script" in sys.argv:
        DRYD.mkdir(parents=True, exist_ok=True)
        (DRYD / "dry_all.sbatch").write_text(dry_script(jobs))
        print("written", DRYD / "dry_all.sbatch")
    if "--collect" in sys.argv:
        sys.exit(0 if collect(jobs) else 1)
    if "--write" in sys.argv:
        assert all(j["dry_ok"] for j in jobs), "DRY_REF.json incomplete: run --collect first"
        for j in jobs:
            j["job"].parent.mkdir(parents=True, exist_ok=True)
            if j["flag"].exists():
                assert j["job"].read_text() == j["text"]
                continue
            j["job"].write_text(j["text"])
        print("written", len(jobs), "job files")
    if "--test-only" in sys.argv:
        for j in jobs:
            o = subprocess.run(P7.SBATCH + ["--test-only", str(j["job"])], capture_output=True, text=True)
            print(j["slurm_name"], (o.stdout + o.stderr).strip().splitlines()[-1:])
    if "--submit" in sys.argv:
        assert all(j["dry_ok"] for j in jobs)
        stale = [j["name"] for j in jobs if not j["flag"].exists() and (not j["job"].exists() or j["job"].read_text() != j["text"])]
        assert not stale, f"job files missing or differing (run --write first): {stale}"
        man = json.loads(MANI.read_text()) if MANI.exists() else {"commit": commit, "trainer_sha12": TR, "runs": {}}
        for j in jobs:   # SIZES order: 1b first
            if j["flag"].exists():
                print(j["name"], "refused (SUBMITTED flag exists)")
                continue
            o = subprocess.run(P7.SBATCH + [str(j["job"])], capture_output=True, text=True)
            print(j["slurm_name"], (o.stdout + o.stderr).strip())
            if o.returncode == 0:
                j["flag"].write_text(o.stdout)
                man["runs"][j["name"]] = {k: j[k] for k in ("size", "arm", "lr", "canary", "slurm_name", "out", "run_json", "args",
                                                            "events", "steps", "window", "base_job", "n_params")}
                man["runs"][j["name"]].update(job_id=o.stdout.split()[-1], job_sha12=hashlib.sha256(j["text"].encode()).hexdigest()[:12])
                MANI.write_text(json.dumps(man, indent=1) + "\n")


if __name__ == "__main__":
    main()
