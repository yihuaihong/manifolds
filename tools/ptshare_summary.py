"""Pre-registered read-out of the arithmetic-share dose (Torch runs/ptshare, 53M, L40S) with the reused 7% cells of
runs/ptlr (+ runs/ptgw main-W). Intent spec and rules: report/audits/ptshare_AUDIT.md (= llmadd_torch/runs/ptshare/AUDIT.md).
Every definition is tools/readout_rules.py's (area, t_theta, budget_M, finished, diverged_at, unstable, best_lr, bracket,
paired, speedup, boot_ci, censored_median, median_at_budget, km double path); the design (shares = --arith_lines 1 / 2 / 4 /
8, lr grids, one-step expansions, tags) is tools/ptshare_design.py; the references (step-0 lm and measured share per share and
seed, CPU) are tools/ptshare_refs.json. Written and tested on fake results (tools/ptshare_selftest.py) before any ptshare run.

Cells are keyed on r["args"]: (arith_lines k, arm, lr) at the 53M ptlr base (every other argument = ptlr's; runs that differ
are counted and left out). k = 4 comes only from ptlr / ptgw, k = 1 / 2 / 8 only from ptshare. A cell's seeds 0-3 come from
the first source with all four finished (order: the --dirs order), its seeds 4-7 likewise.
  rule 1  best lr per (k, arm) on seeds 0-3 = readout_rules.best_lr(near_rule 'top2') (runs/ptlr/AUDIT.md rule 1: the top
          two if within 2 area points -> the lower mean in-training bpb, marked '*'; the jobs run no offline LM);
          readout_rules.bracket: 'open' = best at a grid edge beating its neighbour by > 2 -> one factor-2 step outward
          (seeds 0-3, once per (k, arm)); still open after it -> STOP (the user's call); k = 4: the same rule on its 3-lr
          sub-grid of ptlr / ptgw cells (must reproduce F87's none 8e-3 / win-ctx 2e-3, else a WARNING)
  rule 2  diverged runs keep their area (counted against their lr); unstable (final val_loss > the same seed's none@4e-3 of
          the same share + 0.5 nats) is listed; own failures (OWN_FAILURES.txt) count with the missing evals scored 0
  rule 3  per share, at the selected lrs, seeds 0-3 / 4-7 / all 8, side by side: Delta area (win-ctx - none; paired mean,
          paired seed bootstrap 90% CI, wins, paired t p), R_0.5 and R_0.9 (readout_rules.speedup, '>=' / '<=' bounds,
          censored counts none / prior, readout_rules.boot_ci 90% CI); both arms' absolute areas; floor / ceiling flags
  rule 4  slopes vs ln(share) = ln(k) + const (OLS over the informative shares) of Delta area, log R_0.5, log R_0.9 and of
          each arm's absolute area, for seeds 0-3 / 4-7 / all 8; 90% CI = paired seed bootstrap (seeds resampled within
          each share, the arms paired, shares independent; 10000 resamples, numpy default_rng(0)); censored runs at the
          budget; a bounded R makes the slope a bound in the direction its OLS weight gives
  rule 5  verdict, reading A (pre-submission amendment 2026-10-26, coordinator, user-authorised; ptshare_design.RULE5_READING):
          P = Delta area for the whole slope; S = log R_0.5 with a censored R used as its bound (direction agreement only;
          reading B, P = log R_0.5, is computed only with --show_both, for tests). A mandatory attribution line (both arms'
          absolute-area slopes) is printed with the verdict; every output names a share by its measured token share.
          main test = P's 8-seed slope, 90% CI excludes 0 -> shown; guardrail = P's slope on the confirm seeds 4-7 only,
          same sign; S's 8-seed slope, same sign (no significance). All three -> 'shown (direction)'; P's CI includes 0 ->
          'not shown' (point estimate and CI printed); guardrail or S of the other sign (or of an undeterminable sign) ->
          'direction unstable'. A share where both arms sit at the floor (8-seed mean area < 5 and median final held6 <
          0.10) or both at the ceiling (mean area >= 90 and median final held6 >= 0.95) is uninformative and left out of
          every slope; fewer than 3 informative shares -> no slope ('not shown'). A share whose prior alone is at the
          ceiling: its area gain is compressed, read it by R (noted). R_0.9 is printed, never decides.
Validity (before any number is used): code_sha a3bce3ab63c1; 18310 steps / 75 evals; 52,672,128 parameters, no gradient
checkpointing; win-ctx prior_on_frac 6408 / 18310 and prior_events [[916, 1], [7324, 0]]; none never on; step-0 lm within
1e-3 of the CPU reference of its (k, seed) and of the (k, seed) median over the counted cells. A failure withholds every
verdict that uses the cell. LM-gradient share and clip coefficient of every prior cell from the logged pre-clip norms.
    python tools/ptshare_summary.py [--root /scratch/yh6210/llmadd_torch/runs] [--dirs d1 d2 ...] [--json f]
                                    [--own_failures f] [--refs f] [--nboot 10000]
"""
import argparse
import collections
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_rules as R   # noqa: E402
import ptshare_design as D   # noqa: E402
import ptshare_rules as Q   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/scratch/yh6210/llmadd_torch/runs")
ap.add_argument("--dirs", nargs="*", help="run dirs in priority order (default <root>/{ptshare,ptlr,ptgw}/out)")
ap.add_argument("--json", help="write the selection and every number printed here")
ap.add_argument("--own_failures", help="'<tag> <seed>' lines (default <root>/ptshare/OWN_FAILURES.txt)")
ap.add_argument("--refs", help="CPU references (default tools/ptshare_refs.json)")
ap.add_argument("--nboot", type=int, default=R.N_BOOT)
ap.add_argument("--show_both", action="store_true", help="tests only: compute and print both readings of rule 5 while "
                "ptshare_design.RULE5_READING is unset (the reading is chosen before any ptshare result)")
A = ap.parse_args()
root = Path(A.root)
DIRS = [Path(d) for d in (A.dirs if A.dirs is not None else [root / f"{x}/out" for x in ("ptshare", "ptlr", "ptgw")])]
OWNF = Path(A.own_failures) if A.own_failures else root / "ptshare/OWN_FAILURES.txt"
REFS = json.loads(Path(A.refs).read_text()) if A.refs else (json.loads(D.REFS.read_text()) if D.REFS.exists() else None)
LM0 = {int(k): {int(s): v for s, v in d.items()} for k, d in REFS["lm0"].items()} if REFS else {}
SHARE = {int(k): {int(s): v["share"] for s, v in d.items()} for k, d in REFS["share"].items()} if REFS else {}
LAB = {k: D.share_label(k, SHARE) for k in D.LINES}   # every output names a share by its measured token share (amendment 2026-10-26)

PIN_SHA, STEPS, EVALS, N_PARAMS = "a3bce3ab63c1", 18310, 75, 52672128
SRC_OK = {k: ({"ptlr", "ptgw"} if k == D.REUSED else {"ptshare"}) for k in D.LINES}
BASE = {"tokenization": "whole", "lam": 5.0, "bs": 32, "seq": 512, "tokens": 3e8, "geo_layers": None, "eval_every": 250,
        "n_eval": 150, "untie": False, "text_min": 16, "n_prefix": 256, "d_model": 384, "n_layers": 6, "n_heads": 6,
        "d_ff": 1536, "site": "ctx", "gate_start": 0.0, "gate_hyst": 0.0, "gate_thr": 0.5}
INTEGRITY_ARGS = {"save_model": True, "ckpt_every": 2000, "data": "/scratch/yh6210/llmadd_torch/pretrain_data",
                  "tokenizer": "/scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step2000-tokens9B",
                  "grad_ckpt": "auto", "lm_chunk_tokens": -1, "stop_after": -1, "dry_inputs": False}
GROUPS = (("0-3", D.MAIN), ("4-7", D.CONFIRM), ("8", D.MAIN + D.CONFIRM))
OUT = {"select": {}, "shares": {}, "slopes": {}, "verdict": None, "notes": []}


def fmt_lr(lr):
    m, e = f"{lr:.1e}".split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def arm_of(a):
    if a.get("target") == "none":
        return "none" if a.get("schedule", "whole") == "whole" else None
    if (a.get("target"), a.get("schedule"), a.get("window"), a.get("site", "ctx")) == ("helix3", "window", "0.05,0.4", "ctx"):
        return "win-ctx"
    return None


def why_out(a, src):
    arm = arm_of(a)
    if arm is None:
        return "arm"
    for k, v in BASE.items():   # every source is the a3bce3ab63c1 trainer: every key is logged (strict)
        if a.get(k, "<absent>") != v:
            return f"{k}={a.get(k, '<absent>')}"
    k = a.get("arith_lines")
    if k not in D.LINES:
        return f"arith_lines={k}"
    if src not in SRC_OK[k]:
        return f"share {LAB[k]} not read from {src}"
    lr = fmt_lr(a["lr"])
    if lr not in D.grid(k, arm, expanded=True):
        return "lr outside the ptshare grid"
    want = D.REUSED_TAGS.get((src, arm, lr)) if k == D.REUSED else D.tag(k, arm, lr)
    if a["tag"].lstrip("_") != want:   # the pre-registered tag of the cell (the reused 7% cells: pinned per source)
        return f"tag {a['tag'].lstrip('_')} is not the cell's pre-registered tag"
    return None


# ---------------------------------------------------------------- collect
own = set()
if OWNF.exists():
    for ln in OWNF.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            t, s = ln.split()
            own.add((t, int(s)))
cells = collections.defaultdict(dict)   # (k, arm, lr) -> {source: {seed: (file, run, own_failure)}}
left_out, notes, src_tag, unfin, own_used = collections.Counter(), [], collections.defaultdict(set), [], set()
for d in DIRS:
    src = d.parent.name if d.name == "out" else d.name
    for f in sorted(glob.glob(f"{d}/*/run_*.json")):
        r = json.loads(Path(f).read_text())
        a = r["args"]
        w = why_out(a, src)
        if w:
            left_out[(src, w)] += 1
            continue
        key, tag, seed = (a["arith_lines"], arm_of(a), fmt_lr(a["lr"])), a["tag"].lstrip("_"), a["seed"]
        of = False
        if not R.finished(r):
            if (tag, seed) in own:
                of = True
                own_used.add((tag, seed))
                notes.append(f"own failure (counts, missing evals scored 0): {f} (last eval {r['curve'][-1]['step']})")
            else:
                unfin.append((key, src, seed, f, r))
                continue
        elif (tag, seed) in own:
            own_used.add((tag, seed))
            notes.append(f"WARNING: {OWNF} lists {tag} {seed}, but {f} is finished: the entry is ignored")
        assert seed not in cells[key].get(src, {}), f"two runs of {key} seed {seed} in {d}: {f}"
        cells[key].setdefault(src, {})[seed] = (f, r, of)
        src_tag[(key, src)].add(tag)
for (key, src), tags in src_tag.items():
    assert len(tags) == 1, f"{key} in {src} under several tags {tags}"
for tag, seed in sorted(own - own_used):
    hit = [(key, src) for (key, src), tags in src_tag.items() if tag in tags]
    if seed not in D.MAIN + D.CONFIRM or len(hit) != 1:
        notes.append(f"WARNING: {OWNF} lists {tag} {seed}: no matching cell / seed outside 0-7, ignored")
        continue
    key, src = hit[0]
    sib = next(iter(cells[key][src].values()))[1]
    cells[key][src][seed] = (f"<no run JSON: {tag} seed {seed}>", {"args": dict(sib["args"], seed=seed), "steps": sib["steps"],
                                                                   "curve": [], "code_sha": sib.get("code_sha")}, True)
    notes.append(f"own failure without a run JSON (counts as area 0, bpb worst): {tag} seed {seed}")


def pick(key, seeds):
    for src, runs in cells.get(key, {}).items():
        if all(s in runs for s in seeds):
            return src, {s: runs[s] for s in seeds}
    return None


main = {k: pick(k, D.MAIN) for k in cells}
conf = {k: pick(k, D.CONFIRM) for k in cells}
for key, src, seed, f, r in unfin:
    notes.append(f"unfinished: {f} (last eval {r['curve'][-1]['step']} of {r['steps']}): not counted until it finishes (or is "
                 f"listed in {OWNF.name})")


def area_of(fr):
    f, r, of = fr
    return 100 * sum(R.held(e) for e in r["curve"]) / EVALS if of else R.area(r)


def t_of(fr, th):
    return R.t_theta(fr[1], th) if fr[1]["curve"] else np.nan


def final_held(fr):
    return R.held(fr[1]["curve"][-1]) if fr[1]["curve"] else 0.0


def bpb_of(fr):
    return math.inf if fr[2] or not fr[1]["curve"] else R.worst(fr[1]["curve"][-1]["val_bits_per_byte"])


# ---------------------------------------------------------------- validity
bad = collections.defaultdict(list)
lm0 = collections.defaultdict(dict)
for k_ in cells:
    for pk in (main[k_], conf[k_]):
        if not pk:
            continue
        src, rs = pk
        for s, (f, r, of) in rs.items():
            c = r["curve"]
            if r.get("code_sha") != PIN_SHA:
                bad[(k_, s)].append(f"code_sha {r.get('code_sha')} (want {PIN_SHA}) in {f}")
            if not c:
                continue
            lm0[(k_[0], s)][(k_, s)] = c[0]["lm"]
            if not of and not (r["steps"] == STEPS and len(c) == EVALS and c[-1]["step"] == STEPS - 1):
                bad[(k_, s)].append(f"steps {r['steps']} / {len(c)} evals (want {STEPS} / {EVALS}) in {f}")
            if r.get("n_params") != N_PARAMS or r.get("grad_ckpt_on") is not False:
                bad[(k_, s)].append(f"n_params {r.get('n_params')} / grad_ckpt_on {r.get('grad_ckpt_on')} in {f}")
            wrong = {kk: r["args"].get(kk) for kk, v in INTEGRITY_ARGS.items() if r["args"].get(kk) != v}
            if wrong:
                bad[(k_, s)].append(f"args {wrong} in {f}")
            if k_[1] == "win-ctx" and not of:
                if abs(c[-1].get("prior_on_frac", -1) - R.W_ON) > 1e-6:
                    bad[(k_, s)].append(f"prior_on_frac {c[-1].get('prior_on_frac')} (want {R.W_ON:.6f}) in {f}")
                if r.get("prior_events") != R.W_EVENTS:
                    bad[(k_, s)].append(f"prior_events {r.get('prior_events')} in {f}")
                # the prior acts: a gradient at every on-eval and the fit inside the window far below the last pre-window
                # eval (53 real 53M window-ctx runs: >= 40x); a diverged run is skipped (rule 2: it counts with its area)
                on = [e for e in c if e.get("prior_on")]
                pre = [e for e in c if not e.get("prior_on") and e["step"] < 916]
                g_in = float(np.median([e["geo"] for e in on])) if on and all(e.get("geo") is not None for e in on) else None
                if R.diverged_at(r) is None and not (
                        len(on) == 26 and all((e.get("grad_prior") or 0) > 0 for e in on) and pre and pre[-1].get("geo")
                        and g_in is not None and g_in < 0.25 * pre[-1]["geo"]):
                    bad[(k_, s)].append(f"prior not acting ({len(on)} on-evals, grad_prior {[e.get('grad_prior') for e in on][:3]}..., "
                                        f"geo before the window {pre[-1].get('geo') if pre else None}, median inside {g_in}) in {f}")
            if k_[1] == "none" and (c[-1].get("prior_used_frac", 0) != 0 or c[-1].get("prior_on_frac", 0) != 0):
                bad[(k_, s)].append(f"none run with the prior on ({c[-1].get('prior_on_frac')}) in {f}")
for (k, s), dd in lm0.items():
    med = float(np.median(list(dd.values())))
    ref = LM0.get(k, {}).get(s)
    for ks, v in dd.items():
        if not (math.isfinite(v) and abs(v - med) <= 1e-3):
            bad[ks].append(f"step-0 lm {v:.6f} is {v - med:+.4f} off the {LAB[k]} seed-{s} median (pairing broken)")
        if ref is None or not abs(v - ref) <= 1e-3:
            bad[ks].append(f"step-0 lm {v:.6f} vs the CPU reference {ref} of {LAB[k]} seed {s}")

for n in notes:
    print("NOTE", n)
OUT["notes"] += notes
for ks in sorted(bad, key=str):
    for why in bad[ks]:
        print(f"INTEGRITY {LAB[ks[0][0]]} {ks[0][1]}@{ks[0][2]} seed {ks[1]}: {why}")
print("== sources (priority order): runs in the ptshare grid / left out (reason: count)")
for d in DIRS:
    src = d.parent.name if d.name == "out" else d.name
    n_in = sum(len(v.get(src, {})) for v in cells.values())
    lo = {w: c for (s_, w), c in left_out.items() if s_ == src}
    print(f"  {src:8s} {str(d):55s} {n_in:3d} in grid; left out {sum(lo.values())}"
          + (f" ({', '.join(f'{w}: {c}' for w, c in sorted(lo.items()))})" if lo else "") + ("" if d.exists() else "  (missing)"))
if SHARE:
    print("== measured arithmetic share of the rows (tools/arith_share.py on the CPU dry runs; seeds 0-7): "
          + ", ".join(f"{k} line(s) per row = {LAB[k]} (seeds {100 * min(SHARE[k].values()):.2f}-{100 * max(SHARE[k].values()):.2f}%)" for k in sorted(SHARE)))


def withheld(keys, seeds):
    return [f"{LAB[k[0]]} {k[1]}@{k[2]} seed {s}: {why}" for k in keys for s in seeds for why in bad.get((k, s), [])]


def vals(key, fn, seeds):
    pk = main[key] if seeds == D.MAIN else conf[key] if seeds == D.CONFIRM else None
    if seeds == D.MAIN + D.CONFIRM:
        return None if main[key] is None or conf[key] is None else np.concatenate([vals(key, fn, D.MAIN), vals(key, fn, D.CONFIRM)])
    return None if pk is None else np.array([fn(pk[1][s]) for s in seeds])


# ---------------------------------------------------------------- rule 1: selection per (k, arm)
print("\n== rule 1: best lr per (share, arm) on seeds 0-3 (mean area; tie = the top two within 2 -> lower in-training bpb *)")
best = {}
for k in D.LINES:
    OUT["select"][str(k)] = {}
    for arm in D.ARMS:
        g = {lr: (k, arm, lr) for lr in (D.grid(k, arm, expanded=True)) if (k, arm, lr) in cells}
        pre = [lr for lr in D.grid(k, arm) if lr in g]
        lrs = sorted(g, key=float)
        expanded = any(lr not in D.GRID[arm] for lr in lrs)
        info = {"grid": lrs, "expanded": expanded, "best": None, "open": None, "status": None, "why": ""}
        OUT["select"][str(k)][arm] = info
        if len(pre) < len(D.GRID[arm]):
            info.update(status="missing" if not g else "incomplete",
                        why=f"pre-registered lrs present {pre} of {list(D.GRID[arm])}")
        elif any(main[g[lr]] is None for lr in lrs):
            have = {lr: sorted({s for rs in cells[g[lr]].values() for s in rs}) for lr in lrs}
            info.update(status="incomplete", why=f"seeds 0-3 not all finished: {have}")
        elif withheld(list(g.values()), D.MAIN):
            wh = withheld(list(g.values()), D.MAIN)
            info.update(status="withheld", why=wh[0] + (f" (+{len(wh) - 1} more)" if len(wh) > 1 else ""))
        else:
            inp = {lr: {"area": vals(g[lr], area_of, D.MAIN), "off": [None] * 4, "intr": list(vals(g[lr], bpb_of, D.MAIN))}
                   for lr in lrs}
            sel = R.best_lr(inp, near_rule="top2")
            br = R.bracket(lrs, sel["mean"], sel["best"])
            ref = {s: fr[1] for s, fr in main[(k, "none", D.GRID["none"][0])][1].items()} if main.get((k, "none", D.GRID["none"][0])) else {}
            dv = {lr: [f"s{s}@{R.diverged_at(fr[1])}" for s, fr in main[g[lr]][1].items() if fr[1]["curve"] and R.diverged_at(fr[1]) is not None]
                  for lr in lrs}
            un = {lr: [f"s{s}" for s, fr in main[g[lr]][1].items() if not fr[2] and R.unstable(fr[1], ref.get(s))] for lr in lrs}
            info.update(diverged={lr: v for lr, v in dv.items() if v}, unstable={lr: v for lr, v in un.items() if v})
            info.update(best=sel["best"], mean={lr: round(v, 3) for lr, v in sel["mean"].items()}, near=sel["near"],
                        bpb={lr: round(v, 5) for lr, v in sel["bpb"].items()},
                        open=br[1] if br and br[0] == "open" else None, note=br[1] if br and br[0] == "note" else None)
            if info["open"] is None:
                info["status"] = "final"
            elif k == D.REUSED:
                info.update(status="stop", why=f"{LAB[4]} bracket open on its sub-grid (the reused cells have no expansion)")
            elif expanded:
                info.update(status="stop", why=f"open at the {info['open']} edge after the one-step expansion: the user's call")
            else:
                info.update(status="open", why=f"open at the {info['open']} edge: one step outward "
                                               f"({D.EXPAND[arm][info['open']]}, seeds 0-3)")
            best[(k, arm)] = sel["best"] if info["status"] == "final" else None
        gtxt = ", ".join(f"{lr} {info['mean'][lr]:.1f}" for lr in lrs) if info.get("mean") else ", ".join(lrs) or "-"
        print(f"  {LAB[k]:6s} {arm:7s} grid [{gtxt}] -> {info['status']}"
              + (f": best {info['best']}" if info["best"] else "") + (f"; {info['why']}" if info["why"] else "")
              + (f" (tied {', '.join(info['near'])}; in-training bpb *)" if info.get("near") and len(info["near"]) > 1 else "")
              + (f" (note: area argmax at the {info['note']} edge by <= 2: not open)" if info.get("note") else "")
              + (f" | diverged (counted, rule 2) {info['diverged']}" if info.get("diverged") else "")
              + (f" | unstable (final val_loss > none@{D.GRID['none'][0]} + 0.5) {info['unstable']}" if info.get("unstable") else ""))
if all(OUT["select"][str(D.REUSED)][a]["status"] == "final" for a in D.ARMS):
    got = (OUT["select"][str(D.REUSED)]["none"]["best"], OUT["select"][str(D.REUSED)]["win-ctx"]["best"])
    if got != ("8e-3", "2e-3"):
        print(f"  WARNING: the {LAB[4]} sub-grid selection {got} differs from F87's (8e-3, 2e-3); the rule's output is used")


# ---------------------------------------------------------------- rule 3: per share at the selected lrs
def boot_mean(d, nb):
    idx = np.random.default_rng(0).integers(0, len(d), size=(nb, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def grad_stats(key, seeds):
    """prior on-evals of the cell: LM share of the pre-clip gradient norm |g_lm| / (|g_lm| + |g_prior|), the derived
    pre-clip total norm |g_lm + g_prior| (from the logged norms and cosine) and clip coefficient min(1, 1 / total)"""
    sh, tot, first, ratio = [], [], [], []
    for s in seeds:
        fr = (main[key] if s in D.MAIN else conf[key])[1][s]
        on = [e for e in fr[1]["curve"] if e.get("grad_lm") is not None and e.get("grad_prior") is not None]
        for i, e in enumerate(on):
            a_, b_, c_ = e["grad_lm"], e["grad_prior"], e.get("grad_cos", 0.0)
            sh.append(a_ / (a_ + b_))
            ratio.append(b_ / a_)
            tot.append(math.sqrt(max(a_ * a_ + b_ * b_ + 2 * a_ * b_ * c_, 0.0)))
            if i == 0:
                first.append(b_ / a_)
    if not sh:
        return None
    sh, tot = np.array(sh), np.array(tot)
    clip = np.minimum(1.0, 1.0 / tot)
    return {"n_evals": len(sh), "lm_share_median": float(np.median(sh)), "lm_share_min": float(sh.min()),
            "prior_gt_lm_frac": float(np.mean(sh < 0.5)), "total_norm_median": float(np.median(tot)),
            "clip_coef_median": float(np.median(clip)), "clip_coef_min": float(clip.min()),
            "clipped_frac": float(np.mean(tot > 1.0)), "first_on_prior_over_lm": [float(min(first)), float(max(first))],
            "first_on_prior_over_lm_median": float(np.median(first)), "prior_over_lm_median": float(np.median(ratio))}


print("\n== rule 3: per share at the selected lrs (win-ctx - none; 90% CIs = paired seed bootstrap, 10000, default_rng(0); "
      "censored = never reached within the budget)")
share_data = {}
for k in D.LINES:
    if best.get((k, "none")) is None or best.get((k, "win-ctx")) is None:
        print(f"  {LAB[k]}: no selected lr for both arms yet")
        continue
    kn, kw = (k, "none", best[(k, "none")]), (k, "win-ctx", best[(k, "win-ctx")])
    sd = {"none_lr": kn[2], "prior_lr": kw[2], "groups": {}}
    OUT["shares"][str(k)] = sd
    for gname, seeds in GROUPS:
        an, aw = vals(kn, area_of, seeds), vals(kw, area_of, seeds)
        if an is None or aw is None:
            sd["groups"][gname] = None
            print(f"  {LAB[k]:6s} seeds {gname:3s}: none@{kn[2]} / win-ctx@{kw[2]}: incomplete")
            continue
        wh = withheld([kn, kw], seeds)
        if wh:
            sd["groups"][gname] = {"withheld": wh}
            print(f"  {LAB[k]:6s} seeds {gname:3s}: WITHHELD: {wh[0]}" + (f" (+{len(wh) - 1} more)" if len(wh) > 1 else ""))
            continue
        bud = R.budget_M(next(fr for fr in (main[kn][1].values()) if fr[1]["curve"])[1])
        dA = aw - an
        g = {"n": len(seeds), "area_none": float(an.mean()), "area_prior": float(aw.mean()), "dA": float(dA.mean()),
             "dA_ci": boot_mean(dA, A.nboot), "dA_wins": int((dA > 0).sum()), "dA_p": R.paired(aw, an)[1],
             "final_none": float(np.median(vals(kn, final_held, seeds))), "final_prior": float(np.median(vals(kw, final_held, seeds))),
             "bpb_none": float(np.mean(vals(kn, bpb_of, seeds))), "bpb_prior": float(np.mean(vals(kw, bpb_of, seeds))), "bud": bud}
        for th in (0.5, 0.9):
            tn, tw = vals(kn, lambda fr: t_of(fr, th), seeds), vals(kw, lambda fr: t_of(fr, th), seeds)
            sp, ci = R.speedup(tn, tw, bud), R.boot_ci(tn, tw, bud, A.nboot)
            g[f"R{int(100 * th)}"] = {"ratio": sp["ratio"], "mark": sp["mark"], "text": sp["text"], "med_none": sp["med_ref"],
                                      "med_prior": sp["med_arm"], "cen_none": sp["cen_ref"], "cen_prior": sp["cen_arm"],
                                      "ci": [ci["lo"], ci["hi"]], "ci_cen_none": ci["cen_ref"], "ci_cen_prior": ci["cen_arm"],
                                      "t_none": [None if not np.isfinite(v) else float(v) for v in tn],
                                      "t_prior": [None if not np.isfinite(v) else float(v) for v in tw]}
        sd["groups"][gname] = g
        r5, r9 = g["R50"], g["R90"]
        print(f"  {LAB[k]:6s} seeds {gname:3s} none@{kn[2]} {g['area_none']:5.1f} | win-ctx@{kw[2]} {g['area_prior']:5.1f} | "
              f"dArea {g['dA']:+6.1f} [{g['dA_ci'][0]:+.1f}, {g['dA_ci'][1]:+.1f}] wins {g['dA_wins']}/{len(seeds)} p {g['dA_p']:.3g} | "
              f"R_0.5 {r5['text'].split(' (')[0]} [{r5['ci'][0]:.2f}, {r5['ci'][1]:.2f}] cen {r5['cen_none']}/{len(seeds)} none, "
              f"{r5['cen_prior']}/{len(seeds)} prior | R_0.9 {r9['text'].split(' (')[0]} [{r9['ci'][0]:.2f}, {r9['ci'][1]:.2f}] "
              f"cen {r9['cen_none']}/{len(seeds)} none, {r9['cen_prior']}/{len(seeds)} prior")
    g8 = sd["groups"].get("8")
    if g8 and "withheld" not in g8:
        fc = {"none": Q.floor_ceiling(g8["area_none"], g8["final_none"]), "prior": Q.floor_ceiling(g8["area_prior"], g8["final_prior"])}
        fl = {a_: fc[a_] == "floor" for a_ in fc}
        ce = {a_: fc[a_] == "ceiling" for a_ in fc}
        sd.update(floor=fl, ceiling=ce, uninformative=Q.uninformative(fc["none"], fc["prior"]),
                  prior_ceiling_only=ce["prior"] and not ce["none"], none_t50_censored_8=g8["R50"]["cen_none"])
        sd["grad"] = grad_stats(kw, D.MAIN + D.CONFIRM)
        share_data[k] = (kn, kw)
        print(f"        8 seeds: final held6 (median) none {g8['final_none']:.3f} / prior {g8['final_prior']:.3f}; bpb (in-training) "
              f"none {g8['bpb_none']:.4f} / prior {g8['bpb_prior']:.4f}; floor none/prior {fl['none']}/{fl['prior']}, ceiling "
              f"{ce['none']}/{ce['prior']}" + (" -> UNINFORMATIVE (both at the floor or both at the ceiling): left out of every slope"
                                                if sd["uninformative"] else "")
              + (" -> the prior alone is at the ceiling: the area gain is compressed, read this share by R" if sd["prior_ceiling_only"] else ""))
        r5 = g8["R50"]
        sd["prior_t50_after_window"] = r5["med_prior"] - D.WINDOW_START_M
        sd["area_headroom_prior"] = 100 * (EVALS - sum(e["step"] < D.WINDOW_START_M / (32 * 512 / 1e6) for e in
                                                        next(fr for fr in main[kw][1].values() if fr[1]["curve"])[1]["curve"])) / EVALS - g8["area_prior"]
        print(f"        8 seeds: median t50 none {'>=' if r5['cen_none'] >= 4 else ''}{r5['med_none']:.1f}M / prior "
              f"{'>=' if r5['cen_prior'] >= 4 else ''}{r5['med_prior']:.1f}M; the prior's t50 is {sd['prior_t50_after_window']:.1f}M after "
              f"the window opens ({D.WINDOW_START_M:.1f}M; one eval = 4.1M); the prior's area is {sd['area_headroom_prior']:.1f} below "
              f"the most a window prior can reach (every eval after the window's start at 1.0) (descriptive: near 0 = R and Delta "
              f"area at this share mostly measure none's deficit)")
        gs = sd["grad"]
        if gs:
            print(f"        win-ctx@{kw[2]} gradients on {gs['n_evals']} prior-on evals: LM share of the pre-clip norm median "
                  f"{gs['lm_share_median']:.3f} (min {gs['lm_share_min']:.3f}; prior > LM on {100 * gs['prior_gt_lm_frac']:.0f}%), "
                  f"pre-clip total norm median {gs['total_norm_median']:.2f}, clip coefficient median {gs['clip_coef_median']:.3f} "
                  f"(min {gs['clip_coef_min']:.3f}; clipped on {100 * gs['clipped_frac']:.0f}%); first on-eval prior / LM "
                  f"{gs['first_on_prior_over_lm'][0]:.2f}-{gs['first_on_prior_over_lm'][1]:.2f} (derived from the logged norms)")


# ---------------------------------------------------------------- rule 4: slopes vs ln(share)
def share_arrays(k, seeds):
    kn, kw = share_data[k]
    bud = OUT["shares"][str(k)]["groups"]["8"]["bud"]
    return {"an": vals(kn, area_of, seeds), "aw": vals(kw, area_of, seeds), "bud": bud,
            "t50n": vals(kn, lambda fr: t_of(fr, 0.5), seeds), "t50w": vals(kw, lambda fr: t_of(fr, 0.5), seeds),
            "t90n": vals(kn, lambda fr: t_of(fr, 0.9), seeds), "t90w": vals(kw, lambda fr: t_of(fr, 0.9), seeds)}


METRICS = (("dA", "Delta area"), ("lR50", "log R_0.5"), ("lR90", "log R_0.9"), ("area_none", "area none"),
           ("area_prior", "area win-ctx"), ("lt50n", "log t50 none"), ("lt50w", "log t50 prior"))
TIMES = {"lR50": ("t50n", "t50w"), "lR90": ("t90n", "t90w"), "lt50n": ("t50n", None), "lt50w": (None, "t50w")}


def point(m, d):
    """-> (y, bound) with bound +1 = the true value is >= y, -1 = <= y, 0 = exact, None = no value"""
    if m == "dA":
        return float((d["aw"] - d["an"]).mean()), 0
    if m in ("area_none", "area_prior"):
        return float(d["an" if m == "area_none" else "aw"].mean()), 0
    tn, tw = TIMES[m]
    if tn and tw:
        sp = R.speedup(d[tn], d[tw], d["bud"])
        if sp["ratio"] is None:
            return None, None
        return math.log(sp["ratio"]), {"": 0, ">=": 1, "<=": -1}[sp["mark"]]
    t = d[tn or tw]
    return math.log(R.median_at_budget(t, d["bud"])), (1 if R.censored_median(t) else 0)


def boot_vals(m, d, idx):
    """per resample: the metric and its bound (+1 true >= value: a censored none median in a ratio or a censored time;
    -1: a censored prior median in a ratio; 0 none)"""
    if m in ("dA", "area_none", "area_prior"):
        x = {"dA": d["aw"] - d["an"], "area_none": d["an"], "area_prior": d["aw"]}[m]
        return x[idx].mean(1), np.zeros(len(idx), int)
    n = idx.shape[1]
    cen = lambda key: (~np.isfinite(d[key]))[idx].sum(1) >= (n + 1) // 2
    sub = lambda key: np.median(np.where(np.isfinite(d[key]), d[key], d["bud"])[idx], 1)
    tn, tw = TIMES[m]
    if tn and tw:
        b = np.where(cen(tn), 1, 0) - np.where(cen(tw), 1, 0)   # both censored: no ratio; counted as both ways below
        both = cen(tn) & cen(tw)
        return np.log(sub(tn) / sub(tw)), np.where(both, 9, b)
    key = tn or tw
    return np.log(sub(key)), np.where(cen(key), 1, 0)


def slope_table(ks, seeds):
    data = {k: share_arrays(k, seeds) for k in ks}
    rng = np.random.default_rng(0)
    idx = {k: rng.integers(0, len(data[k]["an"]), size=(A.nboot, len(data[k]["an"]))) for k in ks}   # one draw for all metrics
    res = {}
    for m, _ in METRICS:
        pts = [point(m, data[k]) for k in ks]
        if any(p[0] is None for p in pts):
            res[m] = {"slope": None, "why": "a share has no ratio (both medians censored)"}
            continue
        b, w = Q.ols(ks, [p[0] for p in pts])
        bound = Q.slope_bound(w, [p[1] for p in pts])   # +1: true slope >= b; -1: <= b; None: both ways
        YB = [boot_vals(m, data[k], idx[k]) for k in ks]
        bs = np.stack([y for y, _ in YB], 1) @ w
        lo, hi = float(np.percentile(bs, 5)), float(np.percentile(bs, 95))
        up = np.zeros(A.nboot, bool)
        down = np.zeros(A.nboot, bool)
        for wk, (_, bb) in zip(w, YB):   # a resample's slope is bounded from below (up) / above (down) by its censored medians
            sg = int(np.sign(wk))
            up |= (bb == 9) | (sg * bb == 1)
            down |= (bb == 9) | (sg * bb == -1)
        ci_cons = [float(np.percentile(np.where(down, -np.inf, bs), 5)), float(np.percentile(np.where(up, np.inf, bs), 95))]
        res[m] = {"slope": b, "ci": [lo, hi], "bound": bound, "per_share": {LAB[k]: p[0] for k, p in zip(ks, pts)},
                  "bounds": {LAB[k]: p[1] for k, p in zip(ks, pts) if p[1]}, "per_doubling": b * math.log(2),
                  "push": {"1": float(up.mean()), "-1": float(down.mean())}, "ci_cons": ci_cons}
    return res


def fmt_line(s):
    extra = ""
    if s.get("slope") is not None and (s["bounds"] or s["push"]["1"] or s["push"]["-1"]):
        extra = (f" | censored medians: {', '.join(f'{lab} {chr(8805) if v > 0 else chr(8804)}' for lab, v in s['bounds'].items()) or 'none'}"
                 f" at the point; resamples bounded from below {100 * s['push']['1']:.0f}% / above {100 * s['push']['-1']:.0f}%;"
                 f" CI with those set against a claim [{s['ci_cons'][0]:+.3f}, {s['ci_cons'][1]:+.3f}]")
    return Q.fmt(s) + extra


print("\n== rule 4: slopes vs ln(share) (OLS over the informative shares; 90% CI = paired seed bootstrap within each share, "
      "10000 resamples; negative = the less arithmetic, the larger)")
complete = [k for k in D.LINES if k in share_data and all(OUT["shares"][str(k)]["groups"].get(g) and
                                                           "withheld" not in OUT["shares"][str(k)]["groups"][g] for g, _ in GROUPS)]
missing = [LAB[k] for k in D.LINES if k not in complete]
inf = [k for k in complete if not OUT["shares"][str(k)]["uninformative"]]
if missing:
    print(f"  incomplete: shares {', '.join(missing)} lack a selected lr for both arms, all 8 seeds, or valid runs -> no slope, no verdict")
    OUT["verdict"] = {"label": "incomplete", "zh": "未完成", "why": f"shares {missing} incomplete"}
else:
    unf = [LAB[k] for k in complete if k not in inf]
    if unf:
        print(f"  uninformative shares left out of every slope: {', '.join(unf)}")
    if len(inf) < 3:
        print(f"  only {len(inf)} informative shares (< 3): no slope")
    else:
        for gname, seeds in GROUPS:
            st = slope_table(inf, seeds)
            OUT["slopes"][gname] = st
            for m, lab in METRICS:
                print(f"  seeds {gname:3s} {lab:13s} {fmt_line(st[m])}")
        # descriptive: leave one share out (point estimates, 8 seeds), for the two candidate primary metrics
        OUT["loso"] = {}
        for m in ("dA", "lR50"):
            pts = {k: point(m, share_arrays(k, D.MAIN + D.CONFIRM)) for k in inf}
            OUT["loso"][m] = {LAB[k]: (Q.ols([j for j in inf if j != k], [pts[j][0] for j in inf if j != k])[0]
                                           if len(inf) > 3 and all(pts[j][0] is not None for j in inf if j != k) else None) for k in inf}
            print(f"  descriptive, 8 seeds, {dict(METRICS)[m]} slope leaving one share out: "
                  + ", ".join(f"without {lab} {v:+.3f}" if v is not None else f"without {lab} -" for lab, v in OUT["loso"][m].items()))
    # ---------------------------------------------------------------- covariate: the prior's strength at fixed lambda
    cov = {k: OUT["shares"][str(k)].get("grad") for k in inf}
    if all(cov.values()) and len(inf) >= 3:
        y = [math.log(cov[k]["first_on_prior_over_lm_median"]) for k in inf]
        OUT["covariate"] = {"first_on_prior_over_lm_median": {LAB[k]: cov[k]["first_on_prior_over_lm_median"] for k in inf},
                            "prior_over_lm_median": {LAB[k]: cov[k]["prior_over_lm_median"] for k in inf},
                            "lm_share_median": {LAB[k]: cov[k]["lm_share_median"] for k in inf},
                            "slope_log_first_ratio": Q.ols(inf, y)[0]}
        print("  covariate (descriptive; lambda fixed, and the ratio also changes with lr: at 8.50% 1.58 / 1.18 / 0.95 at 1e-3 / 2e-3 / "
              "4e-3, so across shares with different selected lrs it mixes lr and share): |g_prior| / |g_lm| at the selected lr, first "
              "on-eval, median over 8 seeds: " + ", ".join(f"{LAB[k]} (win-ctx@{share_data[k][1][2]}) {cov[k]['first_on_prior_over_lm_median']:.2f}" for k in inf)
              + "; over every on-eval: " + ", ".join(f"{LAB[k]} {cov[k]['prior_over_lm_median']:.2f}" for k in inf)
              + f"; slope of log(first-on ratio) per ln(share) {OUT['covariate']['slope_log_first_ratio']:+.3f}")
    # descriptive: the same lr at every share (the 8.50% selected lrs: win-ctx 2e-3 / none 8e-3; none has no prior gradient)
    same = {k: (k, "win-ctx", "2e-3") for k in D.LINES}
    if all(main.get(kk) is not None for kk in same.values()):
        cs = {k: grad_stats(kk, D.MAIN) for k, kk in same.items()}
        if all(cs.values()):
            OUT["covariate_same_lr"] = {"lr": "2e-3", "seeds": "0-3",
                                        "first_on_prior_over_lm_median": {LAB[k]: cs[k]["first_on_prior_over_lm_median"] for k in D.LINES},
                                        "prior_over_lm_median": {LAB[k]: cs[k]["prior_over_lm_median"] for k in D.LINES}}
            print("  covariate at one lr (descriptive): win-ctx@2e-3 (the 8.50% selected lr; none at its 8e-3 has no prior gradient), seeds "
                  "0-3, |g_prior| / |g_lm| first on-eval median: " + ", ".join(f"{LAB[k]} {cs[k]['first_on_prior_over_lm_median']:.2f}" for k in D.LINES)
                  + "; over every on-eval: " + ", ".join(f"{LAB[k]} {cs[k]['prior_over_lm_median']:.2f}" for k in D.LINES))
    # ---------------------------------------------------------------- rule 5: verdict
    cen = {k: OUT["shares"][str(k)]["none_t50_censored_8"] for k in inf}
    name = {"dA": "Delta area", "lR50": "log R_0.5"}
    sl = OUT["slopes"]
    print(f"\n== rule 5 (none's t50 censored over 8 seeds: " + ", ".join(f"{LAB[k]} {v}/8" for k, v in cen.items())
          + "; a censored none median makes that share's R a lower bound)")
    readings = {}
    pick = D.RULE5_READING
    # A (amendment 2026-10-26, chosen before any result): P = Delta area for the whole slope (defined at every share); S = log
    # R_0.5 with a censored R used as its bound, direction agreement only. B (not chosen; --show_both for tests): P = log R_0.5
    for rd, P, allow in (("A", "dA", False), ("B", "lR50", True)):
        S = "lR50" if P == "dA" else "dA"
        what = ("P = Delta area for the whole slope; S = log R_0.5 (a censored R as its bound; direction only)" if rd == "A" else
                "P = log R_0.5 for every share, a censored R used as its bound; S = Delta area")
        if pick not in ("A", "B") and not A.show_both:
            print(f"  reading {rd}: {what} (no verdict printed: the reading is not chosen yet)")
            continue
        if rd != pick and not A.show_both:
            continue
        v = Q.decide(sl.get("8", {}).get(P), sl.get("4-7", {}).get(P), sl.get("8", {}).get(S), len(inf), allow_bound=allow)
        v.update(P=P, S=S)
        readings[rd] = v
        print(f"  reading {rd} ({what}): {v['zh']} / {v['label']} -- {v['why']}")
    ceil_only = [LAB[k] for k in inf if OUT["shares"][str(k)]["prior_ceiling_only"]]
    if ceil_only:
        print(f"  note: the prior alone is at the ceiling at {', '.join(ceil_only)}: the area gain there is compressed (read by R); "
              f"with P = Delta area the verdict also rests on S = log R_0.5 having the same direction")
    near = [LAB[k] for k in inf if OUT["shares"][str(k)]["prior_t50_after_window"] < 2 * 4.1]
    if near:
        print(f"  note: the prior's t50 is within 2 evals of the window's opening at {', '.join(near)}: there R and Delta area are "
              f"set mostly by none (the prior cannot be faster); read the gain's change with the share as a change of none's deficit")
    # mandatory with any verdict (amendments 2026-10-26): the high-share end (prior near its ceiling, Delta area compressed)
    # enlarges "the lower the share, the larger the gain"; the low-share end (none's R censored, a lower bound) makes the
    # secondary R's negative slope look smaller (conservative); P = Delta area does not use R. Which arm moves the gain is
    # read from the two arms' absolute areas
    s8 = sl.get("8", {})
    if s8.get("area_none", {}).get("slope") is not None:
        an_, aw_ = s8["area_none"], s8["area_prior"]
        attr = (f"attribution (mandatory): area none slope {an_['slope']:+.2f} [{an_['ci'][0]:+.2f}, {an_['ci'][1]:+.2f}], area win-ctx "
                f"slope {aw_['slope']:+.2f} [{aw_['ci'][0]:+.2f}, {aw_['ci'][1]:+.2f}] per ln(share); Delta area slope = their "
                f"difference {aw_['slope'] - an_['slope']:+.2f}; absolute areas (8 seeds) none / win-ctx: "
                + ", ".join(f"{LAB[k]} {OUT['shares'][str(k)]['groups']['8']['area_none']:.1f} / "
                            f"{OUT['shares'][str(k)]['groups']['8']['area_prior']:.1f}" for k in inf))
        print("  " + attr)
        OUT["attribution"] = attr
    if pick in readings:
        v = readings[pick]
        wording = None
        if v["zh"].startswith("显示了") and "算术越少" in v["zh"]:
            wording = ("写法：增益随占比下降而增大，其中多少来自 none 的缺口、多少来自先验，看两组 arm 的绝对面积（上面 attribution 行，必须同时报）。"
                       f"高占比端（{LAB[4]} / {LAB[8]}）先验接近上限，面积差被压缩，这会放大『占比越低增益越大』。"
                       f"低占比端（{LAB[1]} / {LAB[2]}）R 被删失，只是下界，这会让次指标 R 的负斜率显得更小，是保守的。主指标面积差不用 R。")
        elif v["zh"].startswith("显示了"):
            wording = "写法：增益随占比上升而增大；必须同时报 attribution 行（两组 arm 的绝对面积）"
        OUT["verdict"] = dict(v, reading=pick, readings=readings, wording=wording)
        print(f"  VERDICT (reading {pick}, chosen before any result): {v['zh']} / {v['label']}" + (f"\n  {wording}" if wording else ""))
    else:
        OUT["verdict"] = {"label": "pending: the reading is not chosen", "zh": "待定", "why": "RULE5_READING unset", "readings": readings}
        print("  VERDICT: pending -- RULE5_READING unset")
    print(f"  (no confirmatory comparison with F86 / F87 cross-batch numbers; the {LAB[4]} row is the ptlr cells themselves)")
if A.json:
    Path(A.json).write_text(json.dumps(OUT, indent=1, default=float))
