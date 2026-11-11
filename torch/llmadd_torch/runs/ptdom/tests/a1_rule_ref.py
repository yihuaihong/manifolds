"""The SUPERSEDED A1 selection rule, verbatim from readout_ptdom.py as it stood after Amendment A1 (unit_grid, extend_rule,
select_unit, edge_step, decide; no auxiliary-loss gate in decide), kept ONLY so tests/a2_cost_mc.py --rule A1 can compare
the cost of the A1 and A2 extension rules on the same simulated responses. Not used by any job, selection or read-out."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from readout_ptdom import GRID, LADDER, SEL, SIX, RR   # noqa: E402,F401


def unit_grid(arm, ext=()):
    return sorted(set(GRID[arm]) | set(ext), key=float)


def extend_rule(lrs, mean, pick):
    """A1 (pre-registered before any run): the unit is OPEN (one factor-2 extension beyond a grid edge) when the pick is at
    a grid edge and beats its neighbour by > TIE (readout_rules.bracket 'open') OR the area argmax is at a grid edge
    (bracket's 'note': argmax = the first maximum, the lower lr on an exact tie). Side: the pick's open edge first, else the
    argmax's edge. -> {"open", "side", "reason", "argmax", "bracket"}"""
    br = RR.bracket(lrs, mean, pick)
    k = max(range(len(lrs)), key=lambda j: mean[lrs[j]])
    am_edge = None if len(lrs) < 2 else ("lower" if k == 0 else "upper" if k == len(lrs) - 1 else None)
    pick_open = bool(br and br[0] == "open")
    # consistency with readout_rules.bracket: 'note' exactly when the argmax is at an edge and the pick is not open
    assert (br is not None and br[0] == "note") == (am_edge is not None and not pick_open), (lrs, mean, pick, br)
    if pick_open:
        side, reason = br[1], "pick at the edge beats its neighbour by > 2" + (f"; argmax at the {am_edge} edge" if am_edge else "")
    elif am_edge:
        side, reason = am_edge, f"area argmax at the {am_edge} edge"
    else:
        side, reason = None, None
    return {"open": side is not None, "side": side, "reason": reason, "argmax": lrs[k], "bracket": list(br) if br else None}


def select_unit(R_, arm, dom, ext=()):
    """F87 rule 1 on seeds 0-3 for unit (arm, dom) over its grid (+ its extension lrs ext). -> dict or None (missing)."""
    lrs = unit_grid(arm, ext)
    cells, miss = {}, []
    for lr in lrs:
        rs = [R_.get((arm, lr, s)) for s in SEL]
        if not all(r and r["valid"] for r in rs):
            miss += [f"{arm}@{lr} s{s}" for s, r in zip(SEL, rs) if not (r and r["valid"])]
            continue
        cells[lr] = {"area": [r["area"][dom] for r in rs], "off": [r["bpb_off"] for r in rs],
                     "intr": [r["bpb_intr"] for r in rs]}
    if miss:
        return {"missing": miss}
    b = RR.best_lr(cells, near_rule="top2")
    ex = extend_rule(lrs, b["mean"], b["best"])
    return {"grid": lrs, "mean": {k: round(v, 4) for k, v in b["mean"].items()}, "pick": b["best"], "near": b["near"],
            "tie_break": ("offline" if b["offline"] else "in-training") if len(b["near"]) > 1 else None,
            "bracket": ex["bracket"], "open": ex["open"], "side": ex["side"], "open_reason": ex["reason"], "argmax": ex["argmax"]}


def edge_step(lr, side):
    return LADDER[LADDER.index(lr) + (1 if side == "upper" else -1)]


def decide(R_, prev=None):
    """Stage decisions. prev = None: after stage A; prev = the picks_A decision: after stage X (only the X units are
    re-selected over grid + their extension; units final after A keep their pick). -> picks dict."""
    units = {}
    for d in SIX:
        for arm in (f"only-{d}", "none"):
            key = f"{arm}|{d}"
            if prev is not None and prev["units"][key]["decision"] != "X":
                units[key] = dict(prev["units"][key], stage_final="A")
                continue
            ext = [prev["units"][key]["x_lr"]] if prev is not None else []
            u = select_unit(R_, arm, d, ext)
            if "missing" in u:
                raise SystemExit(f"selection refused: unit {key} lacks valid runs {u['missing']}")
            if prev is None and u["open"]:   # one factor-2 step beyond the GRID edge of that side (the pick may be inside)
                u.update(decision="X", x_lr=edge_step(u["grid"][0] if u["side"] == "lower" else u["grid"][-1], u["side"]))
            elif u["open"]:
                u.update(decision="STOP", x_lr=ext[0])
            else:
                u.update(decision="C", x_lr=ext[0] if ext else None)
            u["stage_final"] = None if u["decision"] == "X" else ("A" if prev is None else "AX")
            units[key] = u
    newC = sorted({(k.split("|")[0], u["pick"]) for k, u in units.items() if u["decision"] == "C"
                   and (prev is None or prev["units"][k]["decision"] == "X")}, key=lambda t: (t[0], float(t[1])))
    if prev is not None:   # a none cell confirmed after A is not confirmed again
        newC = [c for c in newC if list(c) not in prev["C"]]
    X = sorted({(k.split("|")[0], u["x_lr"]) for k, u in units.items() if u["decision"] == "X"}, key=lambda t: (t[0], float(t[1])))
    return {"design": "ptdom", "amendment": "A1", "status": "decided", "stage_read": "A" if prev is None else "AX",
            "units": units, "X": [list(x) for x in X], "C": [list(c) for c in newC],
            "STOP": sorted(k for k, u in units.items() if u["decision"] == "STOP")}

