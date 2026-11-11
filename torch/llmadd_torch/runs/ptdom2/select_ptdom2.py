"""ptdom2 R1 selection (PRE-REGISTERED in runs/ptdom2/AUDIT.md; the function is readout_ptdom2.decide_R, so the read-out
re-derives the same decision = a consistency check).

  python select_ptdom2.py --after R1  [--root0 runs/ptdom/out] [--root2 runs/ptdom2/out] [--out runs/ptdom2/picks_R_after1.json]
  python select_ptdom2.py --after R1x [...]                                                  [--out runs/ptdom2/picks_R_after2.json]

--after R1 (every only-<d>@4e-3 run at seeds 0-3 finished and valid; ptdom's re-used cells read only):
  1. the A2.1 auxiliary-loss gate per (arm, lr) prior cell (seeds 0-3, every finished run, valid or not): a failing cell
     cannot be the pick; an arm failing in every cell -> GATE (no pick, no verdict, out of the Holm family, no R2 for that
     domain; A1.8: no HOLD, the other domains go on);
  A1: the R1pre cells (make_jobs_ptdom2.R1PRE, run with R1) must be valid and equal what the rule asks from ptdom's data;
  the extensions follow AUDIT.md Amendment A1 (pick OR argmax at an outermost lr; none up x2 / down x1/2 once; fixB);
  2. integrity of every batch read (the same data per seed in every arm; one code sha) and the re-used ptdom cells exactly
     as registered (make_jobs_ptdom2.PRIOR_LRS0 / NONE_LRS0);
  3. per R domain F87 rule 1 (top2 + the bpb tie-break) for the prior over its lrs + 4e-3 and for none over its batch's
     lrs (ptdom's + R1pre's + every none cell ptdom2 ran for the batch: shared, A1.3); E (Amendment A1; asked after R1 only,
     seeds 0-3, each step once): only-d@8e-3 where the prior's PICK OR area argmax is 4e-3 (A1.2; a gate-failed 4e-3 is
     closed); none x2 above the batch's top ptdom lr where an R domain's none pick or argmax is that top lr, and x1/2 below
     its lowest where it is the lowest (A1.1a; R1pre's cells count as the used step in their direction); fixB (A1.1b): none
     at the prior pick's lr where that pick lies outside none's range (this round's none asks included); at most 40 runs
     (make_jobs_ptdom2.py --plan; the generator's own docstring still states A0's trigger, superseded by A1) ->
     picks_R_after1.json (written once, chmod 444) ->
     make_jobs_ptdom2.py --part R1x --picks runs/ptdom2/picks_R_after1.json --submit; with no E the decision is final:
     stage_written R2 (-> --part R2 --picks picks_R_after1.json).
--after R1x (every asked extension run finished and valid): picks_R_after1.json must equal its re-derivation; every unit
  re-selected over its lrs + the extensions; NOT FINAL where a final pick or area argmax is the outermost lr the unit ran
  -> picks_R_after2.json (stage R2) -> make_jobs_ptdom2.py --part R2 --picks runs/ptdom2/picks_R_after2.json --submit.
Refuses (exit) while a needed cell lacks a valid run; a picks file is written once (an existing one is never overwritten).
  python select_ptdom2.py --verify runs/ptdom2/picks_R_after<k>.json   (re-derive and compare; submit_ptdom2.sh runs it)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_ptdom2 as RO2   # noqa: E402
RO, M2 = RO2.RO, RO2.M2


def write_once(out, obj):
    out.write_text(json.dumps(obj, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    out.chmod(0o444)
    print(f"written {out} (sha256[:12] {hashlib.sha256(out.read_bytes()).hexdigest()[:12]})")


def table(pk):
    L = [f"ptdom2 R selection after {pk['after']} (F87 rule 1 on seeds 0-3; the A2.1 gate per prior cell):"]
    for d, v in pk["domains"].items():
        up, un = v["unit_prior"], v["unit_none"]
        L.append(f"  {d:11s} {v['status']:9s} batch {v['batch']}: prior lrs {v['prior_lrs']} means {up.get('mean')} -> pick "
                 f"{v['pick_prior']} (argmax {v['argmax_prior']}{', tie-break ' + up['tie_break'] if up.get('tie_break') else ''}"
                 f"{', gate-excluded ' + str(v['gate_excluded']) if v['gate_excluded'] else ''}); none lrs {v['none_lrs']} means "
                 f"{un.get('mean')} -> pick {v['pick_none']} (argmax {v['argmax_none']})"
                 + (f"; extension prior {v['ext_prior']}" if v["ext_prior"] else "") + (f"; extension none {v['ext_none']}" if v["ext_none"] else "")
                 + (f"  [{'; '.join(v['why'])}]" if v["why"] else "") + (f"  notes: {'; '.join(v['notes'])}" if v["notes"] else ""))
        L.append("               gate (a / (a + b) median per prior cell, seeds 0-3): " + "; ".join(
            f"@{lr} {('%.3f' % g['median']) if g['median'] is not None else '-'}{'' if g['ok'] else ' FAIL'}" for lr, g in v["gate"].items()))
    L.append(f"  -> E (seeds 0-3, part R1x) {pk['E']}; C (seeds 8-15, part R2) {pk['C']}; R2's Holm family {pk['family']}"
             + ("" if pk["complete"] else " (pending R1x)"))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", choices=("R1", "R1x"))
    ap.add_argument("--verify", default=None, help="a written picks file: re-derive its decision and compare (exit 0 agree / 1 differ)")
    ap.add_argument("--root0", default=str(RO2.R0 / "out"))
    ap.add_argument("--root2", default=str(M2.R / "out"))
    ap.add_argument("--picks_dir", default=str(HERE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--hold", default=None, help="unused since A1.8 (a GATE domain writes no HOLD); kept so older test calls parse")
    ap.add_argument("--no_guard", action="store_true", help="TESTS ONLY: skip the job-file name guard and the manifest (synthetic roots)")
    ap.add_argument("--pre", default=None, help="TESTS ONLY: the R1pre cells as b:none:lr,... (default make_jobs_ptdom2.R1PRE, checked "
                                                "against ptdom's data)")
    a = ap.parse_args()
    if a.verify:
        pk = json.loads(Path(a.verify).read_text())
        mine = RO2.decide_R(RO2.load_all(a.root0, a.root2, guard=not a.no_guard), pk["after"], pk.get("R1pre"))
        same = RO2.summarize_pick(mine) == RO2.summarize_pick(pk)
        print(f"{a.verify}: {'agrees with' if same else 'DIFFERS from'} its re-derivation (after {pk['after']}, stage {pk['stage_written']})")
        raise SystemExit(0 if same else 1)
    assert a.after, "--after R1 | R1x (or --verify FILE)"
    k = {"R1": 1, "R1x": 2}[a.after]
    out = Path(a.out or Path(a.picks_dir) / f"picks_R_after{k}.json")
    if out.exists():
        raise SystemExit(f"{out} exists: a decision is written once")
    R_ = RO2.load_all(a.root0, a.root2, guard=not a.no_guard)
    need_b = sorted(set(M2.R_DOMS.values()))
    Rb = {key: r for key, r in R_.items() if key[0] in need_b}
    for key, r in sorted(Rb.items()):
        if not r["valid"] and (key[3] in M2.SEL):
            print(f"not valid: {key}: {r['why']}")
    probs = RO.integrity(Rb) + ([] if a.no_guard else RO2.check_manifest(R_))   # the re-used ptdom runs = the frozen manifest
    pre = RO2.M2.R1PRE if a.pre is None else [tuple(x.split(":")) for x in a.pre.split(",") if x]
    if a.pre is None:
        probs += RO2.check_r1pre(R_)   # A1.10: R1pre = what the rule asks from ptdom's data alone
    if probs:
        raise SystemExit("integrity: " + "; ".join(probs))
    fl, g6 = RO.dp_flags(Rb), RO.geo_flags(Rb)
    print(f"FLAGGED (A5.1, valid data): {len(fl)}; FLAGGED (A6, valid data): {len(g6)}"
          + ("".join(f"; {x[0]}" for x in fl + g6) if fl or g6 else ""))
    if k == 2:
        p1f = Path(a.picks_dir) / "picks_R_after1.json"
        p1 = json.loads(p1f.read_text())
        mine = RO2.decide_R(R_, "R1", p1.get("R1pre"))
        assert RO2.summarize_pick(mine) == RO2.summarize_pick(p1), "picks_R_after1.json differs from its re-derivation"
        assert p1["stage_written"] == "R1x", "picks_R_after1.json asked for no extension: it is final (no --after R1x)"
        miss = [(b, a_, lr, s) for b, a_, lr in map(tuple, p1["E"]) for s in M2.SEL
                if not ((b, a_, lr, s) in R_ and R_[(b, a_, lr, s)]["valid"])]
        if miss:
            raise SystemExit(f"selection refused: the extension cells lack valid runs: {miss[:8]} (re-run first)")
    pk = RO2.decide_R(R_, a.after, pre)
    pk["inputs"] = {"root0": str(Path(a.root0).resolve()), "root2": str(Path(a.root2).resolve()),
                    "n_valid": sum(r["valid"] for r in Rb.values()), "readout_ptdom2_sha": RO2.sha(RO2.__file__)}
    print(table(pk))
    gate_dom = [d for d, v in pk["domains"].items() if v["status"] == "GATE"]
    write_once(out, pk)
    if gate_dom:   # A1.8: no HOLD; the domain is GATE (no verdict, out of the Holm family), the others go on
        print(f"GATE (A1.8): {gate_dom}: every prior cell fails the gate: no verdict, not in the Holm family; the other domains go on")
    if pk["stage_written"] == "R1x":
        print(f"next (A2.4: through the submission gates): bash runs/ptdom2/submit_ptdom2.sh --part R1x --picks {out} --go   ({4 * len(pk['E'])} runs, seeds 0-3)")
    elif pk["stage_written"] == "R2":
        print(f"next (A2.4: through the submission gates): bash runs/ptdom2/submit_ptdom2.sh --part R2 --picks {out} --go   (C = {len(pk['C'])} cells "
              f"x seeds 8-15; a none cell Part S already runs is not rendered again)")
    else:
        print("no R domain is FINAL: no R2 runs (every domain NOT FINAL / GATE)")


if __name__ == "__main__":
    main()
