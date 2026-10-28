"""T8: runs/pttpp/T8_remaining.md — per group (size x tpp x 2% / 7%, and the bridge): the designed hardware, the runs done /
running / pending / decided-but-unsubmitted / lost, the planning GPU-h per run on that hardware, the open lr decisions (X / C).
Structural fields only (run JSON last step, check files, SUBMITTED flags, squeue states): no metric.
    python remaining_table.py <squeue snapshot: id|name|account|partition|state|...> <out .md>"""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

R = T.R
q = {}
for l in open(sys.argv[1]):
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


HW = {"l40s": "L40S (625, l40s_public)", "b200": "B200 (520, preemptible)", "a100": "A100-80GB (520, a100_tandon)"}
rows = []
for gid in list(T.GROUPS) + [T.BRIDGE]:
    rs = T.bridge_runs() if gid == T.BRIDGE else T.runs_of(gid)
    cnt = {"done": 0, "RUNNING": 0, "PENDING": 0, "unsubmitted": 0, "other": 0}
    for st, a, lr, s in rs:
        F = T.facts(gid, a, lr, s, st)
        if done(F):
            cnt["done"] += 1
            continue
        sub = R / f"SUBMITTED_{F['name']}"
        if not sub.exists():
            cnt["unsubmitted"] += 1
            continue
        k = q.get(sub.read_text().split()[-1], "gone")
        cnt[k if k in cnt else "other"] += 1
    F0 = T.facts(gid, rs[0][1], rs[0][2], rs[0][3], "A")
    hw = F0["hw"]
    eh = sorted({round(T.facts(gid, a, lr, s, st)["est_h"], 2) for st, a, lr, s in rs})
    if gid == T.BRIDGE:
        dec, size, line, setting = "none (no selection)", "505M", "3e8 tokens", "7%"
        todo = 0
    else:
        c = T.GROUPS[gid]
        size, line, setting = c["size"].upper().replace("1B", "1.07B"), c["line"], c["setting"] + "%"
        eff = T.effective(gid)
        parts, todo = [], 0
        for a in T.ARMS:
            e = eff[a]
            if e["STOP"]:
                parts.append(f"{a}: STOP (the user)")
            elif e["C"]:
                parts.append(f"{a}: C {e['C']} (closed)")
            else:
                todo += len(c["conf"])
                parts.append(f"{a}: open" + (f" (X {e['X']} running / to finish, then C)" if e["X"] else
                                              " (stage A not complete: C, or an X first if the pick is at an edge)"))
        dec = "; ".join(parts)
    rows.append((setting, size, line, gid, HW[hw], cnt, todo, eh, dec))
now = datetime.datetime.now().astimezone().isoformat(timespec="minutes")
out = [f"# pttpp: remaining work per group (Amendment T8; {now}; structural fields only, no result read)", "",
       "Run counts are the runs designed so far (stage A, the decided X / C stages); 'C to decide' = the confirm runs of the",
       "arms without a C yet (8-seed cells: seeds 4-7; 1.07B: 2-7); an X may come first (open bracket, or T6: none's pick at",
       "the upper grid edge). GPU-h per run = the generator's planning rate on the designed hardware (est_h).", "",
       "| setting | size | tpp | group | hardware | done | running | pending | unsubmitted | lost / other | C to decide | GPU-h per run | lr decisions |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for st, size, line, gid, hw, cnt, todo, eh, dec in sorted(rows, key=lambda r: (r[0], r[2], r[1])):
    out.append(f"| {st} | {size} | {line} | {gid} | {hw} | {cnt['done']} | {cnt['RUNNING']} | {cnt['PENDING']} | {cnt['unsubmitted']} | "
               f"{cnt['other']} | {todo} | {' / '.join(f'{x:g}' for x in eh)} | {dec} |")
Path(sys.argv[2]).write_text("\n".join(out) + "\n")
print("\n".join(out))
