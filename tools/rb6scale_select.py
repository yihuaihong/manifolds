"""lr selection of the 505M rainbow sweep (rb6 at 505M, llmadd_torch/runs/rb6_505m; rule PRE-REGISTERED in
runs/rb6_505m/AUDIT.md before the sweep was submitted). Its picks decide the 505M main stage, which the selection job
submits automatically (llmadd_torch/make_jobs_rb6scale.py --batch rb6_505m --stage main --picks <this JSON>).
Rule, per arm (none over 4.8e-3 / 9.6e-3 / 1.92e-2; all6-lam1 over 3e-4 / 6e-4 / 1.2e-3; seeds 0-1):
  1. the lr with the highest mean numbers-domain area over seeds 0-1 (the rainbow headline numbers metric:
     rainbow6_summary READ['numbers'] = 100 x the mean over all evals of the 6-bucket held-out exact match);
  2. every lr within 2 area points of the top is tied -> the lower mean in-training bpb (the last eval's val_bits_per_byte,
     mean over seeds 0-1; a non-finite value is the worst) -> then the lower lr;
  3. bracket: OPEN when the pick is at an edge of the arm's grid (reported; one more step outward only with the user's
     approval); else closed.
A diverged run is data (A6 rule): its area over all evals counts against its lr, its non-finite bpb is the worst.
Completeness (else status 'incomplete', NO picks, nothing is submitted): all 12 runs present (one per arm, lr, seed),
finished (last eval at step steps - 1), their identity clean (rb6scale_summary.torch_identity: args = the cell's intent,
run facts, shas, lambda, logged multiplier, check.json PASS) and the integrity of rainbow6_summary.check (same data and
step-0 lm per seed across the 12, the trainer's metrics = the summary's, ...).
Amendment X1 (runs/rb6_505m/AUDIT.md, 2026-10-25, written before any 2.4e-3 result): --grid_none <lrs> re-applies rules
1-3 to none over an extended grid (X1: 2.4e-3,4.8e-3,9.6e-3,1.92e-2; seeds 0-1); the lrs as spelled in the tags,
ascending, a superset of the pre-registered none grid; all6-lam1's grid unchanged. The grid used is printed and stored
in the JSON ('grid', 'amendment'); a JSON written without the flag (the pre-registered picks.json) is never overwritten
with it. Without the flag the output (text and JSON) is byte-identical to the pre-registered selection's.
    python tools/rb6scale_select.py <dirs with the sweep's run_*.json> --json picks.json
    python tools/rb6scale_select.py <dirs> --grid_none 2.4e-3,4.8e-3,9.6e-3,1.92e-2 --json picks_x1.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rainbow6_summary as R6   # noqa: E402
import rb6scale_summary as S    # noqa: E402

SIZE = "505m"
GRID = {"none": ("4.8e-3", "9.6e-3", "1.92e-2"), "all6-lam1": ("3e-4", "6e-4", "1.2e-3")}
SEEDS = (0, 1)
TIE = 2.0


def worst(v):
    return v if v is not None and math.isfinite(v) else math.inf


def select(area, bpb, grid):
    """area / bpb: {lr: [per-seed values]} over the grid -> (pick, tied lrs, means, bracket)."""
    m = {lr: float(np.mean(area[lr])) for lr in grid}
    top = max(m.values())
    tied = [lr for lr in grid if m[lr] >= top - TIE]
    b = {lr: float(np.mean([worst(v) for v in bpb[lr]])) for lr in grid}
    pick = min(tied, key=lambda lr: (b[lr], float(lr)))
    i = sorted(grid, key=float).index(pick)
    br = ("OPEN at the " + ("lower" if i == 0 else "upper") + " edge (one more step outward only with the user's approval)"
          if i in (0, len(grid) - 1) else "closed")
    return pick, tied, m, b, br


def main():
    ap = argparse.ArgumentParser(description="505M rainbow sweep lr selection; see the module docstring")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", required=True)
    ap.add_argument("--grid_none", default=None, help="Amendment X1: none's grid, comma-separated lrs as in the tags, "
                    "ascending, containing the pre-registered 4.8e-3,9.6e-3,1.92e-2 (default: the pre-registered grid)")
    opt = ap.parse_args()
    out = {"size": SIZE, "status": "incomplete", "rule": "max mean numbers area over seeds 0-1; ties within 2 -> lower mean "
           "in-training bpb -> lower lr; an edge pick = OPEN", "problems": [], "pick": {}, "bracket": {}, "table": {}}
    grid = GRID
    if opt.grid_none is not None:
        gn = tuple(opt.grid_none.split(","))
        assert all(lr and float(lr) > 0 and math.isfinite(float(lr)) for lr in gn), ("--grid_none: positive lrs", gn)
        assert len({float(lr) for lr in gn}) == len(gn), ("--grid_none: an lr twice", gn)
        assert list(gn) == sorted(gn, key=float), ("--grid_none: ascending order", gn)
        assert set(GRID["none"]) <= set(gn), ("--grid_none must contain the pre-registered none grid (spelled as in the "
                                              "tags)", GRID["none"], gn)
        grid = dict(GRID, none=gn)
        if Path(opt.json).exists():
            try:
                prev = json.loads(Path(opt.json).read_text())
            except ValueError:
                prev = None
            assert isinstance(prev, dict) and "grid" in prev, (f"{opt.json} was written without --grid_none (the "
                                                               "pre-registered selection): not overwritten")
        out["grid"] = {a: list(g_) for a, g_ in grid.items()}
        out["amendment"] = "X1 (runs/rb6_505m/AUDIT.md): none's grid extended by --grid_none"
        print(f"GRID (--grid_none, Amendment X1): none {' '.join(grid['none'])}; all6-lam1 {' '.join(grid['all6-lam1'])} "
              f"(pre-registered: none {' '.join(GRID['none'])})")
    runs = R6.load(opt.roots)
    cells, problems = {}, []
    for x in runs:
        m = S.TORCH_ARM.match(x["arm"])
        sz = R6.TAG.match(x["tag"])["size"]
        if sz != SIZE or not m or m[1] not in grid or m[2] not in grid[m[1]] or x["seed"] not in SEEDS:
            problems.append(f"not a sweep run (ignored): {x['tag']} s{x['seed']}")
            continue
        key = (m[1], m[2], x["seed"])
        assert key not in cells, ("two runs of one sweep cell", key)
        cells[key] = x
    for (arm, lr, s), x in sorted(cells.items()):
        bad = S.torch_identity(x, SIZE, arm, lr)
        if x["r"]["curve"][-1]["step"] != x["r"]["steps"] - 1:
            bad.append(f"unfinished (last eval {x['r']['curve'][-1]['step']})")
        for b_ in bad:
            problems.append(f"{arm}@{lr} s{s}: {b_}")
    want = [(a, lr, s) for a in grid for lr in grid[a] for s in SEEDS]
    miss = [k for k in want if k not in cells]
    for k in miss:
        problems.append(f"missing run {k[0]}@{k[1]} s{k[2]}")
    blocking = [p for p in problems if not p.startswith("not a sweep run")]
    if not blocking:   # integrity across the 12 (a diverged run is data: the A6 rule)
        R6.A6_ARMS = R6.A6_ARMS + tuple(sorted({x["arm"] for x in cells.values()}))
        for w in R6.check(list(cells.values()), False):
            print("WARNING", w)
            out.setdefault("warnings", []).append(w)
    for p in problems:
        print(("PROBLEM " if p in blocking else "NOTE ") + p)
    out["problems"] = problems
    print(f"505M rainbow sweep selection: {len(cells)} sweep runs, {len(blocking)} blocking problems")
    for arm, ag in grid.items():
        area = {lr: [R6.metrics(cells[(arm, lr, s)]["r"], "numbers")["area"] for s in SEEDS if (arm, lr, s) in cells]
                for lr in ag}
        bpb = {lr: [R6.final_bpb(cells[(arm, lr, s)]["r"]) for s in SEEDS if (arm, lr, s) in cells] for lr in ag}
        div = {lr: [s for s in SEEDS if (arm, lr, s) in cells and R6.diverged_at(cells[(arm, lr, s)]["r"]) is not None]
               for lr in ag}
        for lr in ag:
            print(f"  {arm:10s} lr {lr:7s} numbers area {' '.join(S.fmt(v, 5) for v in area[lr])} mean "
                  f"{S.fmt(float(np.mean(area[lr])) if area[lr] else None, 5)} | in-training bpb "
                  f"{' '.join(S.fmt(v, 6, 4) for v in bpb[lr])}" + (f" | DIVERGED seeds {div[lr]} (data)" if div[lr] else ""))
        out["table"][arm] = {"area": area, "bpb": bpb, "diverged": div}
        if blocking:
            continue
        pick, tied, m, b, br = select(area, bpb, ag)
        out["pick"][arm], out["bracket"][arm] = pick, br
        out["table"][arm].update(mean_area=m, mean_bpb=b, tied=tied)
        print(f"  -> {arm}: pick {pick} (top mean area {max(m.values()):.1f}; tied within {TIE:g}: {tied}; mean in-training bpb "
              f"{ {k: round(v, 4) for k, v in b.items() if k in tied} }); bracket {br}")
    if not blocking:
        out["status"] = "picked"
        print(f"PICKS none {out['pick']['none']} all6-lam1 {out['pick']['all6-lam1']}; brackets: none {out['bracket']['none']}; "
              f"all6-lam1 {out['bracket']['all6-lam1']}")
    else:
        print("INCOMPLETE: no picks (the main stage is not submitted; the problems above go to the user)")
    Path(opt.json).write_text(json.dumps(out, indent=1, default=lambda o: None))


if __name__ == "__main__":
    main()
