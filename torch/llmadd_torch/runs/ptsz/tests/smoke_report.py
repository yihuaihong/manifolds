"""Read the 731m smoke (runs/ptsz/smoke, job llmadd_psz-smoke731): memory, speed, resume, step-0 calibration.
  memory: per segment torch.cuda max_memory_allocated / max_memory_reserved (the SMOKE line) and, second path, the
          nvidia-smi memory.used maximum of the job (1 s sampling) vs the L40S total;
  speed:  C (the real none ARGS, evals 0 / 250 / 500) and D (eval every 25, 11 evals to 250):
          C: sec(500) - sec(250) = 250 t + e;  D: sec(250) - sec(0) = 250 t + 10 e  ->  e, t;
          run estimate at 3e8 = 18310 t + 75 e + 9 ckpt writes (the timed torch.save of A / B) [+ win-ctx overhead: F87's
          505M L40S win-ctx / none median ratio]; 136m from E is only a memory check (prior on from step 0, 210 steps);
  resume: B's log says 'resumed from ... at step 200'; B's restored evals 0 / 100 = A's; B's step-200 lm = A's step-200 lm
          (the same state after the step-199 ckpt; the ptlr_1b smoke's test); B reaches the in-window eval at 300;
  step 0: the GPU step-0 lm of 731m seed 0 (C, A) and 136m seed 0 (E) vs the CPU references (ptsz_refs.json) - the pttpp
          calibration tolerance 2e-4 (CHECK7 tolerance 1e-3).
    python tests/smoke_report.py JOBID"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_ptsz as G   # noqa: E402

SM = G.SMOKE
LOGS = Path("/scratch/yh6210/llmadd_torch/logs")
GIB = 2 ** 30


def run_json(seg):
    fs = [f for f in (SM / seg).glob("run_*.json")]
    return json.loads(fs[0].read_text()) if fs else None


def main():
    jid = sys.argv[1]
    log = (LOGS / f"llmadd_psz-smoke731_{jid}.out").read_text()
    print(f"# smoke job {jid}: {(LOGS / f'llmadd_psz-smoke731_{jid}.out').name}")
    print("\n".join(l for l in log.splitlines() if l.startswith(("=== smoke731", "NVIDIA", "=== segment", "resumed from", "stop_after",
                                                                   "model:", "Traceback", "torch.OutOfMemory", "RuntimeError"))))
    sm = {}
    for l in log.splitlines():
        if l.startswith("SMOKE "):
            j = json.loads(l[6:])
            sm[j["seg"]] = j
    print("\n## memory (torch peak per segment; L40S total %.2f GiB)" % (next(iter(sm.values()))["total"] / GIB if sm else float("nan")))
    for k, j in sm.items():
        print(f"  {k}: ok {j['ok']}  max_allocated {j['max_alloc'] / GIB:6.2f} GiB  max_reserved {j['max_reserved'] / GIB:6.2f} GiB  "
              f"headroom (total - reserved) {(j['total'] - j['max_reserved']) / GIB:5.2f} GiB  wall {j['wall']:.0f} s  ckpt saves {j['ckpt_save_s']}"
              + (f"  ERR {j['err'][:160]}" if j["err"] else ""))
    g = SM / f"gpu_{jid}.csv"
    if g.exists():
        mem = [float(r[2].strip().split()[0]) for r in csv.reader(g.read_text().splitlines()) if len(r) >= 5 and r[2].strip()[:1].isdigit()]
        tot = [float(r[3].strip().split()[0]) for r in csv.reader(g.read_text().splitlines()) if len(r) >= 5 and r[3].strip()[:1].isdigit()]
        util = [float(r[4].strip().split()[0]) for r in csv.reader(g.read_text().splitlines()) if len(r) >= 5 and r[4].strip()[:1].isdigit()]
        if mem:
            print(f"  nvidia-smi (second path): max memory.used {max(mem):.0f} MiB of {tot[0]:.0f} MiB ({max(mem) / tot[0]:.1%}); "
                  f"mean utilisation {sum(util) / len(util):.0f}% over {len(util)} samples")
    print("\n## speed (731m, L40S)")
    C, D = run_json("C"), run_json("D")
    est = None
    if C and D:
        sc = {e["step"]: e["sec"] for e in C["curve"]}
        sd = {e["step"]: e["sec"] for e in D["curve"]}
        Cd, Dd = sc[500] - sc[250], sd[250] - sd[0]
        e = (Dd - Cd) / 9
        t = (Cd - e) / 250
        saves = [s for k in ("A", "B", "A2", "B2") if k in sm for s in sm[k]["ckpt_save_s"]]
        ck = sum(saves) / len(saves) if saves else 0.0
        none = 18310 * t + 75 * e
        win = none * 23170 / 22995
        est = dict(t=t, e=e, ckpt_s=ck, none_s=none, win_s=win, total_win_s=win + 9 * ck + G.START_S)
        print(f"  C: sec(500) - sec(250) = {Cd:.1f} s; D: sec(250) - sec(0) = {Dd:.1f} s -> {t:.4f} s/step, {e:.2f} s/eval")
        print(f"  ckpt write (torch.save of model + optimiser, {G.CKPT_GB['731m']:.1f} GB): {saves} -> mean {ck:.1f} s")
        print(f"  3e8 run: none {none:.0f} s; win-ctx x {23170 / 22995:.4f} (F87 505M L40S ratio) {win:.0f} s; + 9 ckpt writes "
              f"{9 * ck:.0f} s + start-up {G.START_S} s -> {est['total_win_s']:.0f} s = {est['total_win_s'] / 3600:.2f} h "
              f"(generator before the smoke: TRAIN_S {G.TRAIN_S['731m']:.0f} s)")
    else:
        print("  C / D run JSON missing")
    print("\n## resume (A -> B)")
    A, Bj = run_json("A"), None
    if "B" in sm:
        Bj = run_json("A")   # B continues A's run JSON in the same out dir
    if A is not None:
        res = [l for l in log.splitlines() if l.startswith("resumed from")]
        print(f"  resume lines: {res}")
        seg_lines = log.split("=== segment B start")[1] if "=== segment B start" in log else ""
        a_curve = log.split("=== segment A start")[1].split("=== segment A rc")[0] if "=== segment A start" in log else ""
        def ev(txt):
            out = {}
            for x in txt.splitlines():
                if x.startswith("{"):
                    try:
                        j = json.loads(x)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(j, dict) and "step" in j and "lm" in j:
                        out[j["step"]] = j
            return out
        ea, eb = ev(a_curve), ev(seg_lines.split("=== segment B rc")[0]) if seg_lines else {}
        if 200 in ea and 200 in eb:
            print(f"  step-200 eval: A lm {ea[200]['lm']!r} val_loss {ea[200]['val_loss']!r} | B (resumed) lm {eb[200]['lm']!r} "
                  f"val_loss {eb[200]['val_loss']!r}; lm identical: {ea[200]['lm'] == eb[200]['lm']}")
        if Bj is not None:
            print(f"  final run JSON evals at {[e['step'] for e in Bj['curve']]}")
    print("\n## step-0 lm: GPU (L40S) vs the CPU references")
    for seg, size in (("C", "731m"), ("A", "731m"), ("E", "136m")):
        r = run_json(seg)
        if r is None:
            continue
        ref = G.lm0_ref(size, r["args"]["seed"])
        d = r["curve"][0]["lm"] - ref
        print(f"  {seg} {size} seed {r['args']['seed']}: GPU {r['curve'][0]['lm']:.6f} CPU {ref:.6f} diff {d:+.2e} "
              f"({'within' if abs(d) <= 2e-4 else 'OUTSIDE'} the pttpp calibration 2e-4; CHECK7 tolerance 1e-3: {'ok' if abs(d) <= 1e-3 else 'FAIL'})")
    # the verdict submit_ptsz.sh reads: 731m fits (A and B, i.e. the in-window grad_info evals before and after a resume,
    # ran without error and below the card's memory), C / D / E ran, the resume happened, step-0 lm within CHECK7's 1e-3
    lm_ok = []
    for seg, size in (("C", "731m"), ("E", "136m")):
        r = run_json(seg)
        lm_ok.append(r is not None and abs(r["curve"][0]["lm"] - G.lm0_ref(size, r["args"]["seed"])) <= 1e-3)
    crit = {"A ok": sm.get("A", {}).get("ok") is True, "B ok": sm.get("B", {}).get("ok") is True,
            "C D E ok": all(sm.get(k, {}).get("ok") is True for k in "CDE"),
            "731m reserved <= total - 2 GiB": all(sm[k]["max_reserved"] <= sm[k]["total"] - 2 * 2**30 for k in ("A", "B", "C", "D") if k in sm),
            "resumed": any(l.startswith("resumed from") for l in log.splitlines()), "step-0 lm within 1e-3": all(lm_ok)}
    verdict = {"job": jid, "pass": all(crit.values()), "criteria": crit, "est": est,
               "smoke": {k: {kk: v for kk, v in j.items() if kk != "args"} for k, j in sm.items()}}
    (SM / f"smoke_report_{jid}.json").write_text(json.dumps(verdict, indent=1))
    (SM / "SMOKE_VERDICT.json").write_text(json.dumps(verdict, indent=1))
    print(f"\n## SMOKE {'PASS' if verdict['pass'] else 'FAIL'}: {crit} -> {SM / 'SMOKE_VERDICT.json'}")


if __name__ == "__main__":
    main()
