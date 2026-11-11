"""pt6comb lr selection (PRE-REGISTERED in runs/pt6comb/AUDIT.md; the function is readout_pt6comb.decide, so the read-out re-derives
the same decision = a consistency check).

  python select_pt6comb.py --after L1  [--root runs/pt6comb/out] [--out runs/pt6comb/picks_after_L1.json]
  python select_pt6comb.py --after L1x [--root ...] [--out runs/pt6comb/picks_after_L1x.json]
  python select_pt6comb.py --verify runs/pt6comb/picks_after_<L1|L1x>.json      (re-derive and compare; submit_pt6comb.sh runs it)

--after L1 (every L1 run at seeds 0-3 finished and valid): the A2.1 gate per all6 cell (seeds 0-3; a failing cell cannot be picked;
  every cell failing -> GATE); F87 rule 1 on mean6 (the mean of the six manifolds' areas) for all6 over 1e-3 / 2e-3 / 4e-3 and for
  none over 2e-3 / 4e-3 / 8e-3; the extensions (ptdom2 A1: all6 8e-3 when its pick or area argmax is 4e-3 and 4e-3 passed the gate;
  none 1.6e-2 / 1e-3 when its pick or argmax is 8e-3 / 2e-3; fixB none at all6's pick when it lies outside none's range) ->
  picks_after_L1.json (written once, chmod 444): stage L1x (extensions asked) or C (final).
--after L1x (every asked cell finished and valid): picks_after_L1.json must equal its re-derivation; both units re-selected over all
  their lrs; NOT FINAL when a pick or argmax is still at an outermost lr (all6: the top; none: either edge) or all6's pick lies outside
  none's range -> picks_after_L1x.json.
GATE / NOT FINAL: the file is written with that status (no stage) and the script exits 2: no confirmation renders from it (the
coordinator decides; AUDIT.md Pending). Refuses (exit 1) while a needed cell lacks a valid run or the data / metrics of seeds 0-3 fail
the integrity checks; an existing picks file is never overwritten.
"""
import sys
sys.dont_write_bytecode = True

import argparse          # noqa: E402
import json              # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_pt6comb as RP   # noqa: E402


def derive(root, after, picks_dir):
    R_ = RP.load(root)
    probs = RP.integrity(R_, seeds=RP.SEL)
    prev = None
    if after == "L1x":
        f1 = Path(picks_dir) / "picks_after_L1.json"
        if not f1.exists():
            raise SystemExit("REFUSED: no picks_after_L1.json")
        prev = json.loads(f1.read_text())
        if RP.strip(RP.decide(R_, "L1")) != RP.strip(prev):
            raise SystemExit("REFUSED: picks_after_L1.json differs from its re-derivation")
        if prev.get("stage_written") != "L1x":
            raise SystemExit(f"REFUSED: picks_after_L1.json asks for stage {prev.get('stage_written')}, not L1x")
    return R_, RP.decide(R_, after, prev), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", choices=("L1", "L1x"))
    ap.add_argument("--verify")
    ap.add_argument("--root", default=str(HERE / "out"))
    ap.add_argument("--out")
    ap.add_argument("--picks_dir", default=str(HERE))
    a = ap.parse_args()
    if a.verify:
        pk = json.loads(Path(a.verify).read_text())
        _, dec, probs = derive(a.root, pk.get("after"), Path(a.verify).parent)
        same = RP.strip(dec) == RP.strip(pk)
        print(f"verify {a.verify}: the re-derivation is {'IDENTICAL' if same else 'DIFFERENT'}" + (f"; problems {probs[:5]}" if probs else ""))
        sys.exit(0 if same and not probs else 1)
    assert a.after, "--after L1 | L1x (or --verify FILE)"
    out = Path(a.out) if a.out else Path(a.picks_dir) / f"picks_after_{a.after}.json"
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists (a decision is written once)")
    R_, dec, probs = derive(a.root, a.after, a.picks_dir)
    if dec["status"] == "missing":
        raise SystemExit("REFUSED: cells lack a valid run: " + "; ".join(dec["missing"][:12]))
    if probs:
        raise SystemExit("REFUSED: integrity / double-path problems at seeds 0-3: " + "; ".join(probs[:8]))
    print(RP.selection_text(R_, dec))
    out.write_text(json.dumps(RP.strip(dec), indent=1))
    out.chmod(0o444)
    print(f"written {out} ({RP.sha(out)}): status {dec['status']}, stage {dec['stage_written']}, E {dec['E']}, picks all6@{dec['pick_all6']} "
          f"none@{dec['pick_none']}")
    if dec["status"] != "decided":
        print(f"## {dec['status']}: no confirmation is rendered from this file; the coordinator decides (AUDIT.md Pending). " + "; ".join(dec["notes"]))
        sys.exit(2)
    nxt = "L1x" if dec["stage_written"] == "L1x" else "C"
    print(f"next: bash runs/pt6comb/submit_pt6comb.sh --part {nxt} --picks {out} --go")


if __name__ == "__main__":
    main()
