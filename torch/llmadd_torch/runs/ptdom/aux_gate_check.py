"""ptdom A3 (A2's early watcher, judged per CELL, now per batch): the auxiliary-loss check for the coordinator / a watcher,
on PARTIAL runs (the trainer rewrites its run JSON at every eval). Run it as soon as the first prior runs pass step 1500 (the
window starts at step 916: the evals at 1000 / 1250 / 1500 are on-window) and again later. Read-only: it never writes HOLD
(the coordinator decides; select_ptdom.py writes HOLD_<batch> itself when an arm fails in every stage-1 cell; A4: per-batch
HOLD_<batch> files stop only their batch, runs/ptdom/HOLD stops every batch).

  python runs/ptdom/aux_gate_check.py [--batch X] [--root runs/ptdom/out]     (default: every batch with runs)

Per (batch, arm, lr) cell of a prior arm (every seed so far; a cell's latest run dir, A2 re-runs): per run the median over
the on-window evals logged so far (prior_on, before a divergence: readout_ptdom.window_split) of the LM share by norm
a / (a + b) (a = grad_lm, b = grad_prior, pre-clip); the cell's value = the median of its run medians; the shares of its
on-window evals with b > a and with clip_coef < 1; the largest |logged grad_lm_share - a / sqrt(a^2 + b^2 + 2ab cos)|.
A cell FAILS (readout_ptdom.cell_gate, the rule of select / the read-out) if its median is < 0.5 or a run with on-window evals
is unverifiable: a missing gradient field at an on-window eval, a logged share off its re-derivation by > 1e-3.
Exit 0: no cell fails; 1: a cell fails -> touch runs/ptdom/HOLD_<batch> of the failing cell's batch and report to the user;
2: no on-window eval in any run yet.
A run JSON caught mid-write is retried 3 times (2 s apart), then reported unreadable and skipped; a run before the window
(no on-window eval yet, last eval before step 1000) is listed and not judged; A3 (audit_a2 minor 4): a prior run PAST step
1000 without an on-window eval is judged and fails its cell (unverifiable: 'no on-window eval').
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_ptdom as RO   # noqa: E402

FIRST_ON_EVAL = 1000   # the first eval inside the window (window steps 916..7323, evals every 250 steps)


def read_json(f, tries=3):
    for k in range(tries):
        try:
            return json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            if k + 1 < tries:
                time.sleep(2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / "out"))
    ap.add_argument("--batch", choices=tuple(RO.BATCHES), default=None)
    a = ap.parse_args()
    cells, unread, early = {}, [], []
    for (b, arm, lr, seed), dirs in sorted(RO.cell_dirs(a.root).items(), key=lambda kv: (kv[0][0], kv[0][1], float(kv[0][2]), kv[0][3])):
        if arm == "none" or (a.batch and b != a.batch):
            continue
        d = dirs[-1][1]
        f = RO.run_json(d, b, arm, lr, seed)
        if not f.exists():
            continue
        r = read_json(f)
        if r is None:
            unread.append(d.name)
            continue
        c = r["curve"]
        div = next((e["step"] for e in c if not all(v is None or math.isfinite(v) for v in
                                                      (e.get("lm"), e.get("val_loss"), e.get("geo")))), None)
        g = RO.window_grad(c, div)
        if not g["n_on"] and (g["last_step"] is None or g["last_step"] < FIRST_ON_EVAL):
            early.append(f"{b} {arm}@{lr} s{seed} (last eval step {g['last_step']})")
            continue
        # A3 (audit_a2 minor 4): a run past the window's first eval (step 1000) without an on-window eval is judged:
        # cell_gate fails it as unverifiable ('no on-window eval'), it is not 'before the window'
        cells.setdefault((b, arm, lr), []).append({"seed": seed, "grad": g})
    lines, worst = [], 0
    for (b, arm, lr), rs in cells.items():
        x = RO.cell_gate(rs)
        worst = max(worst, 0 if x["ok"] else 1)
        pc = lambda v: "-" if v is None else f"{100 * v:.0f}%"
        lines.append(f"{b:2s} {arm + '@' + lr:28s} runs {x['n_runs']} (seeds {','.join(map(str, x['seeds']))}), on-window evals {x['n_on']:3d}: "
                     f"median LM share a / (a + b) {'-' if x['median'] is None else '%.3f' % x['median']} "
                     f"[min {'-' if x['min'] is None else '%.3f' % x['min']}]; b > a {pc(x['frac_b_gt_a'])}, clip < 1 {pc(x['frac_clip_lt1'])}; "
                     f"double path {'-' if x['dp_maxdiff'] is None else '%.1e' % x['dp_maxdiff']}"
                     + ("" if x["ok"] else "  FAIL: " + "; ".join(x["why"])))
        for r_ in sorted(rs, key=lambda r_: r_["seed"]):
            gg = r_["grad"]
            lines.append(f"    s{r_['seed']} at step {gg['last_step']}: {gg['n_on']} on-window evals, median LM share "
                         f"{'-' if gg['share'] is None else '%.3f' % gg['share']}, b > a on {gg['n_b_gt_a']}, clip < 1 on {gg['n_clip_lt1']}"
                         + (f", missing fields at {gg['n_missing']} ({', '.join(gg['missing'])})" if gg["n_missing"] else ""))
    print("\n".join(lines) if lines else "no prior-arm run with an on-window eval yet")
    for e in early:
        print(f"before the window (not judged yet): {e}")
    for u in unread:
        print(f"unreadable (being written?): {u}")
    if not cells:
        print("AUX CHECK: no on-window eval yet (exit 2)")
        sys.exit(2)
    failed_b = sorted({b for (b, arm, lr), rs in cells.items() if not RO.cell_gate(rs)["ok"]})
    print("AUX CHECK " + (f"FAILED: a cell fails the auxiliary-loss gate in batch(es) {failed_b}; touch "
                          + " ".join(f"runs/ptdom/HOLD_{b}" for b in failed_b) + " and report to the user (exit 1)"
                          if worst else "passed so far: every cell (exit 0)"))
    sys.exit(worst)


if __name__ == "__main__":
    main()
