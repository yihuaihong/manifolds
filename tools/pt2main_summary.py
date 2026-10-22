"""Pre-registered read-out of the pt2main Burst batch (tasks 350-361 and EXTRA 362-399 from tools/make_pt2main_tasks.py;
intent spec and rules report/audits/pt2main_AUDIT.md, incl. both 2026-10-16 amendments and the read-out amendment).
Definitions (area, t90, diverged, best lr, open bracket, survives tuning, "N x", bootstrap, Burst re-runs):
tools/readout_rules.py, the one definition set shared with tools/ptscale_summary.py. Thin wrapper around
tools/pt999_summary.py:
  1. collects the returned runs results/<task>/<tag>/run_*.json of every task the generator makes (its tasks(): the 88
     and EXTRA) and prints pt999_summary.py's tables for those task dirs, unchanged (cell lines, per-bucket areas,
     same-lr paired comparisons vs none, win-ctx - win-shuf-ctx). One run per (tag, seed) (readout_rules.pick_burst_runs):
     a re-run (EXTRA, same cell and seed, new number) supersedes an unfinished or missing run, or a finished run whose
     check failed; of several finished runs the first (in task order) whose check passed counts, else the first, and the
     others are reported;
  2. applies the batch's rules, with area / t90 by readout_rules (pt999_summary's formulas; its printed cell means are
     parsed and must agree within 0.05, exit 3 otherwise):
     - best lr per (block, arm) = readout_rules.best_lr over the arm's lr grid (BLOCKS + EXTRA cells), seeds 0-3; open
       bracket / edge note = readout_rules.bracket;
     - "2% speed-up survives tuning" (per block) = readout_rules.survives(win-ctx@best, none@best), with the confirm
       seeds 4-7 when both cells have them; without them rule 2 as pre-registered (gap >= 10 'SURVIVES', below 'survival
       pending'), the shared reading ('survives (unconfirmed)') printed next to it as PROPOSED, pending the user's
       sign-off; the same for win-text; repeated without b11; an open bracket of either arm adds 'NOT FINAL' (rule
       1's conditional cell beyond the edge decides);
     - pt2long headline: each arm's lr = the lowest median t90 over its grid (censored runs at the budget; ties -> the
       higher area, then the lower lr), the area-best lr printed next to it; ratio = readout_rules.speedup of
       none@that lr over win-ctx@that lr ('>=' when the none median is censored; no ratio when the win-ctx median is,
       rule 3 as pre-registered, with the shared '<=' bound printed as PROPOSED, pending the user's sign-off) with
       readout_rules.boot_ci; printed with the window in tokens (the ratio belongs to this budget's schedule), and
       again on the 5-bucket mean without b11 (descriptive);
     - control (pt2long, lr 1e-3): win-shuf-ctx must be below win-ctx (mean area difference > 0, >= 3/4 seed wins);
     - b11 separately: per cell the b11 area, the 5-bucket area (area5) and the 5-bucket t90 (t90_5); the table's t90
       columns are pt999_summary's convention (median over the seeds that reach 0.9, k/4 of them), NOT the headline's
       median t90 (a run that never reaches 0.9 counted at the budget);
     - rule 6, integrity, enforced: a counted run whose check.json is missing or failed, whose step-0 lm is off its
       block-seed median (or the same-seed Torch value) by more than 1e-3, or that has no valid full-val offline_lm entry
       makes every verdict that uses its cell print 'WITHHELD: <reason>' instead (best lr of its arm, best-vs-best,
       headline, control); the cell's numbers are still printed. Unfinished / missing runs make a verdict 'incomplete'.
       A diverged run is data, not an integrity failure (listed with its divergence step; its check passes when nothing
       else is wrong, and its offline entry may hold NaN). A results dir of a pt2 block that the generator does not make
       (not in tasks(), e.g. a re-run not yet in EXTRA) is listed as IGNORED.
    python tools/pt2main_summary.py [<results dir>]   (default: results/ of this repo)
Scale ladder (2026-10-22, llmadd_torch/runs/pt2s505/AUDIT.md): the same read-out of another batch generated with
make_pt2main_tasks.py's interface (BLOCKS, SEEDS, LM0, tasks(); a job dir = <res>/<task name>/<tag>/run_*.json):
    python tools/pt2main_summary.py --gen <generator .py> --res <dir of the task dirs> [--json <best lrs per block>]
Without these flags the output is byte-identical to before (regression on the Burst pt2main results).
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "--gen" in sys.argv:   # scale ladder: another generator with make_pt2main_tasks.py's interface (and its own results root)
    import importlib.util
    _spec = importlib.util.spec_from_file_location("pt2gen", sys.argv[sys.argv.index("--gen") + 1])
    G = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(G)
else:
    import make_pt2main_tasks as G   # noqa: E402  (blocks, cells, seeds, EXTRA: one source of truth with the task files)
import readout_rules as R        # noqa: E402  (the shared definitions)

B, B5 = R.B, R.B5
MAIN, CONFIRM = tuple(G.SEEDS), R.CONFIRM
assert MAIN == R.MAIN, MAIN
if "--gen" in sys.argv:
    assert "--res" in sys.argv, "--gen needs --res <dir of the task dirs>"
    RES = Path(sys.argv[sys.argv.index("--res") + 1])
    JSON_OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
else:
    RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results"
    JSON_OUT = None
SELECTION = {}   # --json: per block the best lr per arm (rule 1), the arms without one and why, the open brackets
LM0_TOL = 1e-3

# ---------------------------------------------------------------- collect
T = G.tasks()   # [(name, tag, text, args, block, arm, lr, seed)]: the 88 in submission order, then EXTRA
grid, block_of = {}, {}   # block -> {arm: [lr, ...]} (BLOCKS cells, then EXTRA lrs); tag -> block
size_data = {b[1]: (b[2], b[3]) for b in G.BLOCKS}
for name, tag, _, _, block, arm, lr, s in T:
    lrs = grid.setdefault(block, {}).setdefault(arm, [])
    if lr not in lrs:
        lrs.append(lr)
    block_of[tag] = block
picked, problems = R.pick_burst_runs(RES, [(t[0], t[1], t[7]) for t in T])
known = {t[0] for t in T}
for d in sorted(RES.glob("[0-9]*-pt2*")):
    if d.is_dir() and d.name not in known and re.match(r"^\d+-(%s)-" % "|".join(b[1] for b in G.BLOCKS), d.name):
        problems.append(f"IGNORED {d.name}: not a task of tools/make_pt2main_tasks.py (a re-run counts only once it is in "
                        f"EXTRA)")
runs = {k: r for k, (_, r) in picked.items()}
bpb, dirs, notes, bad = {}, [], [], {}   # bad: (tag, seed) -> [integrity reasons]
for (tag, s), (task, r) in picked.items():
    dirs.append(str(task))
    ck = task / "check.json"
    if not R.check_passed(task):
        failed = [c["check"] for c in json.loads(ck.read_text())["checks"] if not c["ok"]] if ck.exists() else []
        bad.setdefault((tag, s), []).append(f"check.json {'missing' if not ck.exists() else 'FAILED'} in {task.name}"
                                            + (f" ({'; '.join(failed)})" if failed else ""))
    div = R.diverged_at(r)
    off = json.loads((task / "offline_lm.json").read_text()) if (task / "offline_lm.json").exists() else {}
    o = [v for v in off.values() if v.get("tag") == f"_{tag}" and v.get("seed") == s and v.get("max_windows") == 0]
    ob = R.offline_bpb(o[0], div is not None) if o else None
    if ob is None:
        bad.setdefault((tag, s), []).append(f"no valid full-val offline_lm entry in {task.name}"
                                            + (" (|orig_minus_logged| >= 0.002)" if o else ""))
    bpb[(tag, s)] = (ob, r["curve"][-1]["val_bits_per_byte"])   # (offline, in-training)
    if div is not None:
        notes.append(f"diverged (data: its area counts against its lr): {task.name}, first non-finite eval at step {div}")
# step-0 lm: within a block and seed every cell starts from the same init and first batch (the pairing is real)
for block in grid:
    ref = G.LM0.get(size_data[block])
    for s in MAIN + CONFIRM:
        lm0 = {k: r["curve"][0]["lm"] for k, r in runs.items() if block_of[k[0]] == block and k[1] == s}
        if not lm0:
            continue
        med = float(np.median(list(lm0.values())))
        for k, v in sorted(lm0.items()):
            if not (math.isfinite(v) and abs(v - med) <= LM0_TOL):
                bad.setdefault(k, []).append(f"step-0 lm {v:.6f} is {v - med:+.4f} off the block-seed median (pairing broken)")
            if ref and s < len(ref) and not abs(v - ref[s]) <= LM0_TOL:
                bad.setdefault(k, []).append(f"step-0 lm {v:.6f} is {v - ref[s]:+.4f} off the same-seed Torch run")
for p in problems:
    print("NOTE", p)
for p in notes:
    print("NOTE", p)
for k in sorted(bad):
    for why in bad[k]:
        print(f"INTEGRITY {k[0]} seed {k[1]}: {why}")
if not runs:
    sys.exit(f"no finished pt2main runs under {RES}")

# ---------------------------------------------------------------- 1. pt999_summary.py tables, unchanged
print("=" * 30, "tools/pt999_summary.py on the returned pt2main task dirs (one per tag and seed)")
o = subprocess.run([sys.executable, str(HERE / "pt999_summary.py"), *dirs], capture_output=True, text=True)
print(o.stdout, end="")
if o.returncode:
    print(o.stderr); sys.exit("pt999_summary.py failed")
theirs = {m.group(1): float(m.group(2)) for m in re.finditer(r"^  (\S+)\s+n=\d+ area\s+(-?[\d.]+)±", o.stdout, re.M)}
for tag in {t for t, _ in runs}:   # double path: the same area by both scripts
    mine = np.mean([R.area(r) for (t, _), r in runs.items() if t == tag])
    if abs(theirs.get(tag, np.inf) - mine) > 0.05:
        sys.exit(f"area of {tag}: pt999_summary {theirs.get(tag)} vs here {mine:.2f}")
print(f"(double path: the mean area of all {len(theirs)} cells agrees with pt999_summary.py within 0.05)")


# ---------------------------------------------------------------- 2. pre-registered rules
def cell(tag, f=R.area, seeds=MAIN):
    """values over the seeds, or None if any seed is missing / unfinished"""
    v = [f(runs[(tag, s)]) if (tag, s) in runs else None for s in seeds]
    return None if None in v else np.array(v)


def area5(r):
    return R.area(r, B5)


def withheld(tags, seeds=MAIN):
    """integrity reasons of the counted runs of these cells (rule 6)"""
    return [f"{t} seed {s}: {why}" for t in tags for s in seeds for why in bad.get((t, s), [])]


def say_withheld(what, reasons):
    print(f"  {what}: WITHHELD: {reasons[0]}" + (f" (+{len(reasons) - 1} more)" if len(reasons) > 1 else ""))


def bpb_of(tag, seeds=MAIN):
    """{"off": [offline bpb or None], "intr": [in-training bpb]} over the seeds (readout_rules.best_lr's input)"""
    return {"off": [bpb[(tag, s)][0] for s in seeds], "intr": [bpb[(tag, s)][1] for s in seeds]}


def t90s(tag, seeds=MAIN, bs=B):
    return np.array([R.t_theta(runs[(tag, s)], 0.9, bs) for s in seeds])


def med_txt(t):
    return f"{np.nanmedian(t):6.0f}" if np.isfinite(t).any() else "     -"


print("\n" + "=" * 30, "pt2main pre-registered read-out (report/audits/pt2main_AUDIT.md; definitions tools/readout_rules.py)")
for base, block, size, data, tokens, plan_h, tlim, _ in G.BLOCKS:
    tags = {(a, lr): f"{block}-lr{lr}-{a}" for a, lrs in grid[block].items() for lr in lrs}
    print(f"\n## {block} ({size}, {tokens} tokens, {data})")
    print(f"  {'cell':28s} {'area':>12s} {'area5':>6s} {'b11':>6s} {'t90* (M)':>9s} {'t90_5*':>6s} {'bpb':>8s}"
          f"   (t90*: median over the seeds that reach 0.9, k/4; not the headline's median t90)")
    for (a, lr), t in tags.items():
        ar, a5, b1 = cell(t), cell(t, area5), cell(t, lambda r: R.area(r, ("b11",)))
        if ar is None:
            print(f"  {t:28s} incomplete ({sum((t, s) in runs for s in MAIN)}/4)")
            continue
        tt, t5 = t90s(t), t90s(t, bs=B5)
        bc = bpb_of(t)
        off = all(v is not None for v in bc["off"])
        bp = np.mean([R.worst(v) for v in bc["off" if off else "intr"]])
        conf = cell(t, seeds=CONFIRM)
        wh = withheld([t])
        print(f"  {t:28s} {ar.mean():5.1f}±{ar.std():4.1f} {a5.mean():6.1f} {b1.mean():6.1f} {med_txt(tt)} "
              f"{int(np.isfinite(tt).sum())}/4 {med_txt(t5)} {bp:.4f}{'' if off else '*'}"
              + (f"   seeds 4-7: {conf.mean():5.1f}" if conf is not None else "")
              + (f"   [integrity: {len(wh)} problem(s); verdicts on this cell are withheld]" if wh else ""))
    best, nobest, openbr = {}, {}, {}   # arm -> best lr; arm -> 'incomplete' / 'withheld'; arm -> open edge
    for arm, lrs in grid[block].items():
        lrs = sorted(lrs, key=float)
        m = {lr: cell(tags[(arm, lr)]) for lr in lrs}
        if any(v is None for v in m.values()):
            print(f"  best lr {arm}: incomplete")
            nobest[arm] = "incomplete"
            continue
        wh = withheld([tags[(arm, lr)] for lr in lrs])
        if wh:
            say_withheld(f"best lr {arm}", wh)
            nobest[arm] = "withheld"
            continue
        sel = R.best_lr({lr: {"area": m[lr], **bpb_of(tags[(arm, lr)])} for lr in lrs})
        best[arm] = sel["best"]
        br = R.bracket(lrs, sel["mean"], best[arm])
        if br and br[0] == "open":
            openbr[arm] = br[1]
        print(f"  best lr {arm}: {best[arm]} (grid {', '.join(lrs)}; within 2 of the top: {', '.join(sel['near'])}"
              f"{'' if len(sel['near']) == 1 or sel['offline'] else '; tie broken on the in-training bpb *'})"
              + (f"  OPEN BRACKET at the {br[1]} edge" if br and br[0] == "open" else "")
              + (f"  (note: area argmax {lrs[-1] if br[1] == 'upper' else lrs[0]} at the {br[1]} edge by <= 2: not an "
                 f"open bracket by rule 1)" if br and br[0] == "note" else ""))
    SELECTION[block] = {"best": dict(best), "nobest": dict(nobest), "open": dict(openbr),
                        "grid": {a: sorted(v, key=float) for a, v in grid[block].items()}}
    for arm in ("win-ctx", "win-text"):
        if arm not in grid[block]:
            continue
        miss = [a for a in ("none", arm) if a not in best]
        if miss:
            w = [a for a in miss if nobest[a] == "withheld"]
            print(f"  {arm}@best - none@best: " + (f"WITHHELD: the best lr of {' and '.join(w)} is withheld" if w else "incomplete"))
            continue
        nb = tags[("none", best["none"])]
        t = tags[(arm, best[arm])]
        has_conf = cell(t, seeds=CONFIRM) is not None and cell(nb, seeds=CONFIRM) is not None
        wh = withheld([t, nb], MAIN + (CONFIRM if has_conf else ()))
        if wh:
            say_withheld(f"{t} - {nb}", wh)
            continue
        sv = R.survives(cell(t), cell(nb), *((cell(t, seeds=CONFIRM), cell(nb, seeds=CONFIRM)) if has_conf else ()),
                        confirm_free_gap=10)
        sv5 = R.survives(cell(t, area5), cell(nb, area5), confirm_free_gap=10)
        bt, bn = bpb_of(t), bpb_of(nb)
        off = all(v is not None for v in bt["off"] + bn["off"])   # offline if every run of both cells has it
        k = "off" if off else "intr"
        cost = R.fmt_paired(R.paired(1000 * np.array(bt[k], float), 1000 * np.array(bn[k], float)))
        print(f"  {t} - {nb}: area {R.fmt_paired(sv['main'])}; without b11 {R.fmt_paired(sv5['main'])}; LM cost "
              f"({'offline' if off else 'in-training*'} bpb x1e-3, arm - none; wins = seeds where the arm costs) {cost}"
              + (f"\n    confirm seeds 4-7: area {R.fmt_paired(sv['confirm'])} -> "
                 f"{'agrees' if sv['confirm_ok'] else 'does NOT agree'}" if sv["confirm"] else "")
              + f"\n    -> 2% speed-up {sv['verdict']}"
              + ("; its confirm seeds 4-7 are the pre-registered conditional cells (EXTRA)"
                 if sv["main_ok"] and not sv["confirm"] and sv["main"][0] < 10 else "")
              + (f" [PROPOSED shared rule, pending the user's sign-off: {sv['proposed']}]" if sv["proposed"] else "")
              + f"{'' if arm == 'win-ctx' else ' (text site)'}"
              + "".join(f"; NOT FINAL: {a}'s bracket is open at the {openbr[a]} edge (rule 1: the next lr beyond it is the "
                        f"pre-registered conditional cell)" for a in ("none", arm) if a in openbr)
              + f"{'' if sv['main_ok'] == sv5['main_ok'] else ' (differs without b11)'}")
    if block != "pt2long":
        continue

    def lr_by_t90(arm):
        """(lr with the lowest median t90, {lr: median at the budget}, integrity reasons), or None while incomplete"""
        lrs = sorted(grid[block][arm], key=float)
        if any(cell(tags[(arm, lr)]) is None for lr in lrs):
            return None
        bud = R.budget_M(runs[(tags[(arm, lrs[0])], MAIN[0])])
        med = {lr: R.median_at_budget(t90s(tags[(arm, lr)]), bud) for lr in lrs}
        mean_area = {lr: float(cell(tags[(arm, lr)]).mean()) for lr in lrs}
        return R.best_lr_t90(med, mean_area), med, withheld([tags[(arm, lr)] for lr in lrs])

    sel = {a: lr_by_t90(a) for a in ("none", "win-ctx")}
    if any(v is None for v in sel.values()):
        print("  HEADLINE: incomplete")
    elif sel["none"][2] or sel["win-ctx"][2]:
        say_withheld("HEADLINE", sel["none"][2] + sel["win-ctx"][2])
    else:
        lr_n, lr_w = sel["none"][0], sel["win-ctx"][0]
        for a in ("none", "win-ctx"):
            print(f"  headline lr {a}: {sel[a][0]} by median t90 (censored runs at the budget: "
                  + ", ".join(f"{lr} {v:.0f}M" for lr, v in sel[a][1].items()) + f"); area-best {best.get(a, '?')}")

        def headline(seeds, label, bs=B):
            n_ = [runs[(tags[("none", lr_n)], s)] for s in seeds]
            w_ = [runs[(tags[("win-ctx", lr_w)], s)] for s in seeds]
            bud = R.budget_M(n_[0])
            tn = np.array([R.t_theta(r, 0.9, bs) for r in n_])
            tw = np.array([R.t_theta(r, 0.9, bs) for r in w_])
            sp, ci = R.speedup(tn, tw, bud, arm_censored="none"), R.boot_ci(tn, tw, bud)
            per = n_[0]["args"]["bs"] * n_[0]["args"]["seq"] / 1e6
            ev = w_[0].get("prior_events") or []
            win = f"window {ev[0][0] * per:.0f}-{ev[1][0] * per:.0f}M tokens" if len(ev) == 2 else "window ?"
            ge = lambda t: ">=" if R.censored_median(t) else ""
            print(f"  {label} t90 none@{lr_n} / win-ctx@{lr_w} (lr by median t90; area-best none@{best.get('none', '?')}, "
                  f"win-ctx@{best.get('win-ctx', '?')}): {sp['text']}" + (f" ({ci['text']})" if sp["ratio"] is not None else "")
                  + f"; medians {ge(tn)}{sp['med_ref']:.0f}M / {ge(tw)}{sp['med_arm']:.0f}M; censored at {bud:.0f}M: none "
                  f"{sp['cen_ref']}/{len(seeds)}, win-ctx {sp['cen_arm']}/{len(seeds)}; {win} of {bud:.0f}M (this budget's "
                  f"schedule)" + (f" [PROPOSED shared rule, pending the user's sign-off: {sp['proposed']}]" if sp["proposed"] else ""))

        headline(MAIN, "HEADLINE (seeds 0-3)")
        headline(MAIN, "  without b11 (5-bucket t90, descriptive)", B5)
        conf = [tags[("none", lr_n)], tags[("win-ctx", lr_w)]]
        if all((t, s) in runs for t in conf for s in CONFIRM):
            if withheld(conf, CONFIRM):
                say_withheld("confirm (seeds 4-7)", withheld(conf, CONFIRM))
            else:
                headline(CONFIRM, "confirm (seeds 4-7)")
    if ("win-shuf-ctx", "1e-3") in tags and ("win-ctx", "1e-3") in tags:
        ct = [tags[("win-ctx", "1e-3")], tags[("win-shuf-ctx", "1e-3")]]
        x, y = cell(ct[0]), cell(ct[1])
        if x is None or y is None:
            print("  CONTROL win-ctx@1e-3 - win-shuf-ctx@1e-3: incomplete")
        elif withheld(ct):
            say_withheld("CONTROL win-ctx@1e-3 - win-shuf-ctx@1e-3", withheld(ct))
        else:
            d = x - y
            print(f"  CONTROL win-ctx@1e-3 - win-shuf-ctx@1e-3: {d.mean():+.1f} (wins {int((d > 0).sum())}/4) -> "
                  f"{'passes (shuffled geometry below the helix)' if d.mean() > 0 and (d > 0).sum() >= 3 else 'FAILS: geometry specificity not shown'}")
if JSON_OUT:
    Path(JSON_OUT).write_text(json.dumps(SELECTION, indent=1))
