"""ptdom A3 / A4 selection and screen (PRE-REGISTERED in runs/ptdom/AUDIT.md "Amendment A3" and "Amendment A4"; the functions
are readout_ptdom.screen / chain_step, so the read-out re-derives the same decisions = a consistency check).

  python select_ptdom.py --batch X --after 1 [--root runs/ptdom/out] [--out runs/ptdom/picks_X_after1.json] [--hold runs/ptdom/HOLD_X]
  python select_ptdom.py --batch X --after 2|3|4 [--root ...] [--out runs/ptdom/picks_X_after<k>.json]
      (A4: after stage 2 / 2b / 2c; the earlier picks files picks_X_after1..k-1 are read and must equal the re-derivation)

--after 1 (stage 1 of batch X finished: every cell at seeds 0-3 with a valid run):
  1. the A2 auxiliary-loss gate per (arm, lr) cell FIRST (readout_ptdom.aux_gate, seeds 0-3, every finished prior cell,
     valid or not): a cell fails at a median a / (a + b) < 0.5 or an unverifiable run; a failing cell is excluded from its
     arm's selection; an arm whose every stage-1 cell fails -> GATE (no pick, the domain cannot advance), the reason is
     appended to --hold (A4 default runs/ptdom/HOLD_X: every pending job OF THIS BATCH exits at start, the generator refuses
     to submit this batch; the other batches go on) and the selection exits 2 after the picks file is written (the user
     decides).
  2. integrity of the batch (the same data per seed in every arm; one code sha; no finished run superseded by a re-run);
     the comparison domains' sub-read-outs re-derived from the per-question hits and tied to the trainer's logs (strict).
  3. the screen of every domain (picks by F87 rule 1, the paired gains on seeds 0-3, the gate at the pick, the shortcut
     condition for the comparisons) and the stage-2 cells: C = seeds 4-7 at the advancing domains' picks (prior and none;
     a none lr once), E = the extension lrs at seeds 0-3 (prior: always, its 2-point grid's argmax is at an edge; none: at
     an argmax 2e-3 -> 1e-3) -> picks_X_after1.json (written once, read-only) -> make_jobs_ptdom.py --batch X --stage 2.
--after k = 2, 3, 4 (A4; stage 2 / 2b / 2c finished): every advancing unit re-selected over grid + the extensions it ran
  (seeds 0-3); E = none's second lower step 5e-4 where 1e-3 is still its argmax (after 2 only; seeds 0-7), C = a pick
  without seeds 4-7 so far -> picks_X_after<k>.json -> make_jobs_ptdom.py --batch X --stage 2b (after 2) / 2c (after 3)
  when E or C is not empty; 'complete' when nothing is asked. The verdicts: readout_ptdom.py once every batch's chain is
  complete (Holm over all advancing domains of the four batches; otherwise PROVISIONAL).
  A5 fixB (AUDIT.md A5.2): F = the extra NONE jobs (seeds 0-7, same batch, same GPU type) at a prior's final pick lr that
  lies outside the lr range none ran for that domain (once none's own extensions are done); they are listed by name and
  rendered with the stage the file writes (2b / 2c); none is re-selected including them at the next decision; the domain
  has no verdict until they have run.
Refuses (exit) when a needed cell lacks a valid run; an unfinished run (e.g. a HOLD exit) is re-run first
(make_jobs_ptdom.py --rerun, A2). A5.1: a run whose only failed checks are the GPU-numerics double-path tolerances (and / or
the CKA-rise bar) is valid data: it is listed as FLAGGED, never refused. A6 (AUDIT.md A6): so is a run whose only failed checks
are these and the geo / grad_prior check at a geo overshoot of a perfect fit verified by readout_ptdom.geo_a6 (listed as
FLAGGED (A6, valid data)); otherwise such a run stays invalid.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_ptdom as RO   # noqa: E402


def write_once(out, obj):
    out.write_text(json.dumps(obj, indent=1))
    out.chmod(0o444)
    print(f"written {out} (sha256[:12] {hashlib.sha256(out.read_bytes()).hexdigest()[:12]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=tuple(RO.BATCHES), required=True)
    ap.add_argument("--after", choices=("1", "2", "3", "4"), required=True)
    ap.add_argument("--root", default=str(HERE / "out"))
    ap.add_argument("--picks_dir", default=str(HERE), help="where picks_X_after<k>.json are read (k < --after) and written")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hold", default=None, help="the kill switch written when an arm fails the gate in every stage-1 cell "
                                                 "(A4 default runs/ptdom/HOLD_<batch>; tests: a temp path)")
    ap.add_argument("--subs_json", default=None, help="TESTS ONLY: planted comparison sub-read-outs instead of the per-question re-derivation")
    a = ap.parse_args()
    b = a.batch
    a.hold = a.hold or str(HERE / f"HOLD_{b}")
    out = Path(a.out or Path(a.picks_dir) / f"picks_{b}_after{a.after}.json")
    if out.exists():
        raise SystemExit(f"{out} exists: a decision is written once (move it away only if the AUDIT says so)")
    R_ = {k: v for k, v in RO.load(a.root).items() if k[0] == b}
    for k, r in sorted(R_.items()):
        if not r["valid"]:
            print(f"not valid: {k}: {r['why']}")
    fl = RO.dp_flags(R_, b)   # A5.1: valid data, listed
    print(f"FLAGGED (A5.1, valid data): {len(fl)} runs exceed a GPU-numerics double-path tolerance"
          + (": " + "; ".join(f"{k[1]}@{k[2]} s{k[3]} ({', '.join(v)})" for k, v in fl) if fl else ""))
    g6 = RO.geo_flags(R_, b)   # A6: valid data, listed
    print(RO.A6_LINE.format(n=len(g6)) + (": " + "; ".join(f"{k[1]}@{k[2]} s{k[3]} ({', '.join(v)})" for k, v in g6) if g6 else ""))
    gate = RO.aux_gate(R_, b, RO.SEL)   # A2 (i): before any decision, per cell
    print(RO.gate_report(gate))
    probs = RO.integrity(R_)
    if probs:
        raise SystemExit("integrity: " + "; ".join(probs))
    if a.subs_json:
        subs = {tuple(json.loads(k)): v for k, v in json.loads(Path(a.subs_json).read_text()).items()}
    else:
        meta = RO.domain_meta(RO.BATCHES[b])
        subs, sprobs = RO.subsets_all(R_, meta)
        if sprobs:
            raise SystemExit("comparison sub-read-outs: " + "; ".join(sprobs[:20]))
    if a.after == "1":
        held = False
        if gate["arms_all_failed"]:
            msg = (f"AUX GATE FAILED (select_ptdom --batch {b} --after 1): prior arm(s) {', '.join(gate['arms_all_failed'])}: every "
                   "stage-1 cell fails: " + "; ".join(f"{k}: {', '.join(x['why'])}" for k, x in gate["cells"].items()
                                                     if k.split("@")[0] in gate["arms_all_failed"])
                   + f". The prior is not an auxiliary loss there at any stage-1 lr: no pick, the domain cannot advance; pending "
                     f"jobs of batch {b} exit at start while this file exists (A4: per batch); the user decides.\n")
            with open(a.hold, "a") as fh:
                fh.write(msg)
            print(msg.strip())
            print(f"written {a.hold}")
            held = True
        pk = RO.screen(R_, b, subs)
        assert pk["aux_gate_failed_cells"] == gate["failed"], (pk["aux_gate_failed_cells"], gate["failed"])
        pk["aux_gate"] = {k: {f: x[f] for f in ("ok", "n_runs", "median", "min", "frac_b_gt_a", "frac_clip_lt1", "why")}
                          for k, x in gate["cells"].items()}
        pk["hold_written"] = held
        pk["inputs"] = {"root": str(Path(a.root).resolve()), "n_runs_valid": sum(r["valid"] for r in R_.values())}
        print(RO.screen_report(pk))
        write_once(out, pk)
        print(f"stage 2 of batch {b}: {8 * len(pk['E'])} extension runs (seeds 0-7, A4) + {4 * len(pk['C'])} confirm runs (seeds 4-7): "
              f"python make_jobs_ptdom.py --batch {b} --stage 2 --picks {out} --submit")
        if held:
            print(f"exit 2: {a.hold} written (an arm failed the auxiliary-loss gate in every stage-1 cell); batch {b} cannot be submitted until the user decides")
            raise SystemExit(2)
        return
    k = int(a.after)   # A4: after stage 2 / 2b / 2c
    prev, shas = [], []
    for j in range(1, k):
        pj = Path(a.picks_dir) / f"picks_{b}_after{j}.json"
        x = json.loads(pj.read_text())
        assert x.get("amendment") == "A5" and x["batch"] == b and x["after"] == str(j), (pj, x.get("amendment"), x.get("after"))
        mine = RO.screen(R_, b, subs) if j == 1 else RO.chain_step(R_, b, prev)
        assert (mine["C"], mine["E"], mine.get("F", [])) == (x["C"], x["E"], x.get("F", [])) and (j > 1 or mine["ADVANCE"] == x["ADVANCE"]), \
            f"the decision after {j} re-derived differs from {pj.name}"
        assert j == 1 or x["E"] or x["C"] or x.get("F") or j == k - 1, f"{pj.name} was complete: there is no decision after {j + 1}"
        prev.append(x)
        shas.append(hashlib.sha256(pj.read_bytes()).hexdigest()[:12])
    assert prev[-1]["E"] or prev[-1]["C"] or prev[-1].get("F"), \
        f"picks_{b}_after{k - 1}.json asked for nothing: the chain is complete (no --after {k})"
    miss = [c for c in RO.asks(b, prev[-1]) if not ((b,) + c in R_ and R_[(b,) + c]["valid"])]
    if miss:
        raise SystemExit(f"selection refused: the cells asked by picks_{b}_after{k - 1}.json lack valid runs: {miss[:8]} (re-run first)")
    pk = RO.chain_step(R_, b, prev)
    pk["inputs"] = {"root": str(Path(a.root).resolve()), "previous_sha": shas}
    print(RO.chain_report(b, [pk]))
    write_once(out, pk)
    if pk["F"]:   # A5.2: list the extra none jobs by name
        done = RO.asked(b, prev)
        names = [RO.MJ.job_name(b, a_, lr_, s_) for a_, lr_ in map(tuple, pk["F"]) for s_ in RO.SEL + RO.CONF
                 if (a_, lr_, s_) not in done]
        print(f"{RO.FIXB} (AUDIT.md A5.2): {len(names)} extra none jobs (seeds 0-7, batch {b}, {RO.MJ.HW[RO.MJ.HWOF[b]]['gpu']}): "
              + ", ".join(names) + " -- for "
              + "; ".join(f"{d} (prior pick {f_['none']['fixb_lr']} outside none's range {f_['none']['range']})"
                          for d, f_ in pk["final"].items() if f_["none"].get("fixb_pending"))
              + "; those domains get no verdict until these jobs have run")
    if pk["complete"]:
        print(f"batch {b}: the decision chain is complete (nothing more to run); the read-out gives verdicts once all four batches are")
    else:
        print(f"stage {pk['stage_written']} of batch {b}: {8 * len(pk['E'])} extension runs (seeds 0-7) + {8 * len(pk['F'])} {RO.FIXB} none "
              f"runs (seeds 0-7) + {4 * len(pk['C'])} confirm runs (seeds 4-7): python make_jobs_ptdom.py --batch {b} --stage "
              f"{pk['stage_written']} --picks {out} --submit")


if __name__ == "__main__":
    main()
