"""T8: remaining work of pttpp by hardware (structural fields only: a run is done when its run JSON's last step is the
last step and its check file exists; no metric is read). Counts decided-but-unfinished runs (stage A, X, C) and the runs
still to be decided (C for every arm without C / STOP; the T6 X for a none arm whose stage A is not decided yet is a
possibility, listed separately), with est_h per run (the generator's planning rates)."""
import json
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

R = T.R
q = {}
for l in open("/scratch/yh6210/llmadd_torch/scale_work/tpp/t8/squeue_tpp.txt"):
    f = l.rstrip("\n").split("|")
    q[f[0]] = f[4]


def done(F):
    if F["setting"] == "2":
        t = R / "out2" / F["name"]
        js = list((t / F["tag"]).glob("run_*.json"))
        if not (len(js) == 1 and (t / "check.json").exists()):
            return False
        j = js[0]
    else:
        tgt = "none" if F["arm"] == "none" else "helix3"
        j = R / "out" / F["tag"] / f"run_whole_{tgt}_seed{F['seed']}_{F['tag']}.json"
        if not (j.exists() and (R / "out" / F["tag"] / f"check_s{F['seed']}.json").exists()):
            return False
    try:
        r = json.loads(j.read_text())
        return r["curve"][-1]["step"] >= r["steps"] - 1
    except Exception:
        return False


def qstate(F):
    s = R / f"SUBMITTED_{F['name']}"
    if not s.exists():
        return "unsubmitted"
    jid = s.read_text().split()[-1]
    return q.get(jid, "gone")


rows = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
todo = defaultdict(lambda: [0, 0.0])
for gid in list(T.GROUPS) + [T.BRIDGE]:
    rs = T.bridge_runs() if gid == T.BRIDGE else T.runs_of(gid)
    for st, a, lr, s in rs:
        F = T.facts(gid, a, lr, s, st)
        k = "done" if done(F) else qstate(F)
        rows[F["hw"]][(gid, st, k)][0] += 1
        rows[F["hw"]][(gid, st, k)][1] += F["est_h"]
    if gid != T.BRIDGE:
        eff = T.effective(gid)
        for a in T.ARMS:
            if eff[a]["C"] is None and not eff[a]["STOP"]:
                F = T.facts(gid, a, T.GRID[gid][a][0], T.GROUPS[gid]["conf"][0], "C")
                n = len(T.GROUPS[gid]["conf"])
                todo[(F["hw"], gid, a, "C to decide")][0] += n
                todo[(F["hw"], gid, a, "C to decide")][1] += n * F["est_h"]
for hw, d in rows.items():
    print(f"== {hw}")
    tot = defaultdict(lambda: [0, 0.0])
    for (gid, st, k), (n, h) in sorted(d.items()):
        if k != "done":
            print(f"   {gid:20s} {st} {k:12s} {n:3d} runs {h:6.1f} h")
        tot[k][0] += n
        tot[k][1] += h
    print("   totals:", {k: (v[0], round(v[1], 1)) for k, v in tot.items()})
print("== C still to decide (counted at the lower grid lr's est_h)")
for (hw, gid, a, w), (n, h) in sorted(todo.items()):
    print(f"   {hw:5s} {gid:20s} {a:8s} {n} runs {h:5.1f} h")
print("planned_hours", {k: round(v, 1) for k, v in T.planned_hours().items()})
