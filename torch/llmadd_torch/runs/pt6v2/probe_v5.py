#!/usr/bin/env python
"""pt6v2 Amendment V5 probe (exp-builder 2026-11-20; exploratory, descriptive, no gate -- AUDIT.md Amendment V5).
Review fixes 2026-11-20 (tests/audit_probe_v5.md items 1, 2, 3, 4, 6 + the L0-NaN minor item): see HANDOFF_V5_probe.md.

Question: after the prior window (steps 916-7323) dim accuracy falls much less with all six priors than with only the
dim prior, while the logged cka_dim@L2/@L4 (item positions) is retained equally. Are later layers, or the answer
position, washed out less in the joint model?

Per run (out_v5/<run>/<arm>/ with wts_*_step{7323,12000,18309}.pt) and checkpoint, at every layer (0 = embedding output,
l = block l output) and two positions -- item = both input-item sites of every dim_held + dim_seen eval row (the logged
cka_dim's states; slot A = the first item (pos 506, the clean item representation), slot B = the second item (pos 508,
after the op token)), ans = the answer-prediction position (' is', its logits predict the answer's first token):
 (a) drift: linear CKA between the states of checkpoint 7323 and 18309 / 12000 of the same run, same prompts; raw rows
     and item-mean (states averaged over contexts per item -> 27 x D; ans: per answer item)
 (b) CKA to the target geometry: the trainer's recipe cka(H, centred_gram(exponent vectors)); item -> the item's
     exponent vector, ans -> the ANSWER item's exponent vector; raw rows and item-mean (27 x D)
 (c) ridge decodability of the same exponent vectors, fitted on ALL rows: item-grouped 9-fold CV (3 of the 27 items held
     out per fold; ans: grouped by the answer item); ridge alpha = r * mean squared singular value of the centred
     training states, r in 10^[-5..2] (15 values) chosen by an inner item-grouped 8-fold CV on the training items only.
     The per-row out-of-fold (OOF) predictions and errors are saved (oof_<label>.npz) and scored separately on row
     subsets: strict held (the trainer's strict readout set), non-strict held, seen, answered correctly / incorrectly
     (ans_hit at that checkpoint), slot A / slot B, strict-held slot A; pooled is kept as an extra column. Scores: R2
     (pooled over the 3 coordinates, against the subset's own mean) and nearest-lattice-point accuracy (round + clip to
     {-1,0,1}^3 = nearest of the 27 items).
     Frozen decoder: the fold fits at 7323 (alpha chosen at 7323) predict the same held-out rows at 12000 / 18309
     without refitting; scored on the same subsets.
 (c') answer position only, the input-transport control: E(a) and E(b) are decoded at the answer position too (the same
     ridge, all rows, grouped by a / by b). Per op (times: E(c) = E(a) + E(b), per: E(c) = E(a) - E(b), asserted) the
     transport formula Ea_hat +- Eb_hat (no refit) predicts E(c); reported: R2 of E(a), E(b), E(c), of the transport
     formula, and the RESIDUAL R2 = R2 of the transport residual rho = E(c) - (Ea_hat +- Eb_hat) explained by the direct
     decoder's correction Ec_hat - (Ea_hat +- Eb_hat), i.e. 1 - SSE(Ec_hat) / SS(rho). > 0: the direct E(c) readout
     removes part of the transport formula's error (answer information beyond transported inputs); < 0: the transport
     formula is the better account. (The refit incremental R2 of Ec_hat over an OLS on [Ea_hat, Eb_hat] is ~0 by
     linearity whether the state transports the inputs or carries the answer -- shown in the selftest -- so it is kept in
     the JSON only, as incr_refit.)
 (d) the model's own dim accuracy at the answer position: dim_held (exact match, the trainer's code) and the strict set
     (rainbow6_summary.strict_series, the readout's dim primary) from this probe's hits_dim_held
Every (a)/(b)/(c) table prints the shuffle baseline beside it (20 item-level shuffles of the 27-row exponent table, the
same shuffles everywhere; drift: 20 row / item permutations): b_mean, r2_p95, the lattice chance (the constant predictor
on that subset) and the shuffled lattice p95. Degenerate cells (zero-variance states, e.g. the answer position at L0)
print NaN.
Built-in checks (flag only, never fail): positive (b@read layers vs the logged cka_dim@L at the nearest eval step),
answer (d vs the logged dim_held / strict / hits string), negative (shuffled targets: (b), (c)).

Usage:
  probe_v5.py --run <run dir> [...]       one or more run dirs -> <out>/<label>.json + oof_<label>.npz + tables
  probe_v5.py --v5                        the 4 V5 runs (all6 / only-dim x seeds 16, 17), then --summary
  probe_v5.py --summary <json> ...        all6 - only-dim per layer and checkpoint, the 7323 -> 18309 DiDs, behaviour
  probe_v5.py --selftest                  smoke C/D tiny checkpoints (5/10/19 as A/B/C), summary on relabelled copies,
                                          synthetic pooling-false-negative guard + transport control (real row layout)
  probe_v5.py --timing                    real-config random-init model, registered all6 s16 args (CPU time estimate)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent                      # runs/pt6v2
LIB = R / "probe_v5_lib"
TOOLS = R / "out_v5" / "code" / "tools"
B = Path("/scratch/yh6210/llmadd_torch")
sys.path[:0] = [str(TOOLS), str(LIB)]
V5_STEPS = (7323, 12000, 18309)
V5_RUNS = [R / "out_v5" / f"pt6v2-{a}-lr2e-3-s{s}" / f"pt6v2-{a}-lr2e-3" for s in (16, 17) for a in ("all6", "only-dim")]
SIX = {"numbers", "dim", "cities", "countries", "torus", "box5"}
N_SHUF, ALPHAS = 20, 10.0 ** np.linspace(-5, 2, 15)
NOT_ARGS = {"out", "probe_wts", "probe_npz"}
NAN = float("nan")
OPS = {27: "times", 28: "per"}                           # dom_q op tokens; E(c) = E(a) + E(b) / E(a) - E(b) (asserted)
SUBS = {"item": ["strict_A", "A", "pooled", "strict", "B", "nonstrict_held", "seen", "hit", "miss"],
        "ans": ["strict", "pooled", "nonstrict_held", "seen", "hit", "miss"]}
MAIN = {"item": ["strict_A", "A", "pooled"], "ans": ["strict", "pooled"]}   # main tables (first = the primary readout)
TRANS_SUBS = ["strict", "pooled", "nonstrict_held", "seen"]
GAP_MIN = 0.10                                           # behavioural strict gap (all6 - only-dim) needed at the end
WARN = []


def warn(msg):
    WARN.append(msg)
    print(f"  !!! WARNING: {msg}", flush=True)


# ------------------------------------------------------------------ dump (the trainer's own prompt construction)
def argv_from(a: dict) -> list:
    argv = []
    for k, v in a.items():
        if k in NOT_ARGS or v is None:
            continue
        if isinstance(v, bool):
            argv += [f"--{k}"] if v else []
        else:
            argv += [f"--{k}", str(v)]
    return argv


def dump(run_json: dict, wts: list, ddir: Path) -> dict:
    """Runs the probe copy of the V5 trainer with the run's own args; returns meta.json (asserted against the run)."""
    subprocess.run([sys.executable, str(LIB / "make_probe_copy.py")], check=True)
    ddir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", str(LIB / "pretrain_range_probe.py")] + argv_from(run_json["args"]) + \
          ["--out", str(ddir / "trainer_out"), "--probe_wts", ",".join(map(str, wts)), "--probe_npz", str(ddir)]
    env = dict(os.environ, HF_HOME=str(B / "hf_cache"), PYTHONPATH=str(B / "pylib"), PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    with open(ddir / "dump.log", "w") as f:
        rc = subprocess.run(cmd, cwd=TOOLS, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        print(open(ddir / "dump.log").read()[-4000:])
        sys.exit(f"probe dump failed rc={rc}: {ddir / 'dump.log'}")
    meta = json.loads((ddir / "meta.json").read_text())
    meta["dump_wall_s"] = round(time.time() - t0, 1)
    meta["setup_s"] = round(meta["dump_wall_s"] - sum(c["forward_s"] for c in meta["ckpts"]), 1)   # trainer setup + load
    # the probe's args and dim eval sets are the run's (crash: before any analysis is spent)
    bad = {k: (v, meta["args"].get(k)) for k, v in run_json["args"].items() if k not in NOT_ARGS and meta["args"].get(k) != v}
    assert not bad, ("probe args != run args", bad)
    for nm, rec in meta["dinfo"].items():
        lg = run_json["domains"]["dim"][nm]
        for k in ("questions", "contexts", "rows", "q", "cls", "masks"):
            assert rec[k] == lg[k], ("probe eval set != logged", nm, k)
    return meta


# ------------------------------------------------------------------ measures
def centred_gram(F):   # pretrain_helix.centred_gram in float64
    F = np.asarray(F, np.float64)
    F = F[:, F.std(0) > 1e-9]
    F = (F - F.mean(0)) / F.std(0)
    F = F - F.mean(0)
    return F @ F.T


def fstd(F):   # centred_gram's feature matrix (centred_gram(F) = fstd(F) @ fstd(F).T)
    F = np.asarray(F, np.float64)
    F = F[:, F.std(0) > 1e-9]
    F = (F - F.mean(0)) / F.std(0)
    return F - F.mean(0)


def is_degen(H):   # zero-variance states (e.g. ' is' at the embedding output): every measure on them is NaN
    H = np.asarray(H, np.float64)
    if len(H) < 2:
        return True
    return not np.any(np.abs(H - H.mean(0)) > 1e-9 * (np.abs(H).max() + 1e-30))


def cka_feat(Hc, nHH, Fs):   # = the trainer's cka(H, Fs Fs^T) in feature form: tr(G K) / (|G| |K| + 1e-8), Hc centred
    if Fs.shape[1] == 0 or not nHH > 0:
        return NAN
    return float(np.linalg.norm(Hc.T @ Fs) ** 2 / (nHH * np.linalg.norm(Fs.T @ Fs) + 1e-8))


def cka_subset(H, F_list):   # (b) on a row subset, true target + shuffles
    if is_degen(H):
        return [NAN] * len(F_list)
    Hc = H - H.mean(0)
    nHH = float(np.linalg.norm(Hc.T @ Hc))
    return [cka_feat(Hc, nHH, fstd(F)) for F in F_list]


def cka_lin(A, Bm):   # linear CKA between two state sets of the same rows (feature form)
    if is_degen(A) or is_degen(Bm):
        return NAN
    A = A.astype(np.float64); A = A - A.mean(0)
    Bm = Bm.astype(np.float64); Bm = Bm - Bm.mean(0)
    num = np.linalg.norm(A.T @ Bm) ** 2
    den = np.linalg.norm(A.T @ A) * np.linalg.norm(Bm.T @ Bm)
    return float(num / den) if den > 0 else NAN


def cka_lin_null(A, Bm, perms):   # drift null: the same CKA with the rows of Bm permuted (mean, p95)
    if is_degen(A) or is_degen(Bm):
        return NAN, NAN
    A = A.astype(np.float64); A = A - A.mean(0)
    Bm = Bm.astype(np.float64); Bm = Bm - Bm.mean(0)
    den = np.linalg.norm(A.T @ A) * np.linalg.norm(Bm.T @ Bm)
    v = [float(np.linalg.norm(A.T @ Bm[p]) ** 2 / den) for p in perms]
    return float(np.mean(v)), float(np.percentile(v, 95))


def item_mean(H, v, m):   # states averaged over contexts per item (rows in m) -> (items, n_items x D)
    items = np.unique(v[m])
    return items, np.stack([H[m & (v == i)].mean(0) for i in items]) if len(items) else np.zeros((0, H.shape[1]))


def item_folds(groups, k, seed=0):
    u = np.random.default_rng(seed).permutation(np.unique(groups))
    return [u[i::k] for i in range(k)]


def _fit(X, Y, tr):
    mu, ym = X[tr].mean(0), Y[tr].mean(0)
    U, S, Vt = np.linalg.svd(X[tr] - mu, full_matrices=False)
    keep = S > S.max() * 1e-10 if len(S) else S > 0
    return mu, ym, U[:, keep], S[keep], Vt[keep], (U[:, keep].T @ (Y[tr] - ym))


def _pred(fit, X, te, rel):
    mu, ym, U, S, Vt, UtY = fit
    a = rel * float(np.mean(S ** 2)) if len(S) else 1.0
    P = (X[te] - mu) @ Vt.T
    return P @ ((S / (S ** 2 + a))[:, None] * UtY) + ym


def ridge_cv(X, Ys, groups, X_extra=()):
    """Item-grouped CV ridge; Ys = list of [n, 3] targets (true + shuffles). Returns out-of-fold preds, chosen r's, and
    (frozen decoder) each fold's fit applied, without refitting, to the same held-out rows of every X_extra (same rows,
    later checkpoints)."""
    X = X.astype(np.float64)
    Xe = [x.astype(np.float64) for x in X_extra]
    Yall = np.concatenate(Ys, 1); C = len(Ys)
    oof = np.zeros_like(Yall); ext = [np.zeros_like(Yall) for _ in Xe]; chosen = [[] for _ in range(C)]
    for f in item_folds(groups, 9):
        te = np.isin(groups, f); tr = ~te
        if not te.any():
            continue
        inner = item_folds(groups[tr], 8, seed=1)
        err = np.zeros((len(ALPHAS), C))
        for g in inner:
            ite = tr & np.isin(groups, g); itr = tr & ~np.isin(groups, g)
            if not ite.any():
                continue
            ft = _fit(X, Yall, itr)
            for ai, rel in enumerate(ALPHAS):
                e = (_pred(ft, X, ite, rel) - Yall[ite]) ** 2
                err[ai] += e.reshape(len(e), C, 3).sum((0, 2))
        ft = _fit(X, Yall, tr)
        best = err.argmin(0)
        for ai in np.unique(best):
            cols = np.concatenate([np.arange(3 * c, 3 * c + 3) for c in np.where(best == ai)[0]])
            oof[np.ix_(te, cols)] = _pred(ft, X, te, ALPHAS[ai])[:, cols]
            for e_, x_ in zip(ext, Xe):
                e_[np.ix_(te, cols)] = _pred(ft, x_, te, ALPHAS[ai])[:, cols]
        for c in range(C):
            chosen[c].append(float(ALPHAS[best[c]]))
    split = lambda M: [M[:, 3 * c:3 * c + 3] for c in range(C)]
    return split(oof), chosen, [split(e_) for e_ in ext]


def scores(y, yh):   # pooled (all rows), as before
    if not np.all(np.isfinite(yh)):
        return dict(r2=NAN, r2_dims=[NAN] * 3, lattice_acc=NAN)
    sst = ((y - y.mean(0)) ** 2).sum(0)
    sse = ((y - yh) ** 2).sum(0)
    lat = np.clip(np.rint(yh), -1, 1)
    return dict(r2=float(1 - sse.sum() / sst.sum()), r2_dims=[float(v) for v in 1 - sse / np.maximum(sst, 1e-12)],
                lattice_acc=float((lat == y).all(1).mean()))


def sscore(y, yh, m):   # a row subset: R2 against the subset's own mean, lattice accuracy, mean squared error per row
    n = int(m.sum())
    if n < 3 or not np.all(np.isfinite(yh[m])):
        return dict(n=n, r2=NAN, lattice_acc=NAN, mse=NAN)
    yy, pp = y[m], yh[m]
    sst = float(((yy - yy.mean(0)) ** 2).sum()); sse = float(((yy - pp) ** 2).sum())
    return dict(n=n, r2=(1 - sse / sst) if sst > 1e-12 else NAN,
                lattice_acc=float((np.clip(np.rint(pp), -1, 1) == yy).all(1).mean()), mse=sse / n)


def const_pred(y, groups):   # constant predictor: the train-fold mean (folds as the ridge)
    yh = np.zeros_like(y, dtype=np.float64)
    for f in item_folds(groups, 9):
        te = np.isin(groups, f)
        if te.any():
            yh[te] = y[~te].mean(0)
    return yh


def const_baseline(y, groups):
    return scores(y, const_pred(y, groups))


def res_r2(yc, pc, pt, m):   # R2 of the transport residual rho = yc - pt explained by the direct correction pc - pt
    n = int(m.sum())
    if n < 3 or not (np.all(np.isfinite(pc[m])) and np.all(np.isfinite(pt[m]))):
        return NAN
    rho = yc[m] - pt[m]
    ss = float(((rho - rho.mean(0)) ** 2).sum()); sse = float(((yc[m] - pc[m]) ** 2).sum())
    return 1 - sse / ss if ss > 1e-12 else NAN


def stage2_oof(F, y, groups):   # OLS with intercept, grouped 9-fold CV (by answer item) -> OOF predictions
    Fi = np.c_[np.ones(len(F)), F]
    yh = np.zeros_like(y, dtype=np.float64)
    for f in item_folds(groups, 9):
        te = np.isin(groups, f); tr = ~te
        if not te.any():
            continue
        yh[te] = Fi[te] @ np.linalg.lstsq(Fi[tr], y[tr], rcond=None)[0]
    return yh


def transport_stats(Yc, oc, Ya, oa, Yb, ob, ans_op, ans_c, masks, refit=True):
    """(c') per op and subset. Yc/Ya/Yb: lists (true + shuffles) of targets; oc/oa/ob: their OOF predictions."""
    out = {}
    for op in sorted(OPS):
        om = ans_op == op
        if not om.any():
            continue
        sg = 1.0 if OPS[op] == "times" else -1.0
        pt = [a_ + sg * b_ for a_, b_ in zip(oa, ob)]
        inc = None
        if refit and np.all(np.isfinite(oa[0])):   # the reviewer's refit incremental R2 (JSON only; ~0 by linearity)
            ii = np.where(om)[0]
            base = stage2_oof(np.c_[oa[0][ii], ob[0][ii]], Yc[0][ii], ans_c[ii])
            full = stage2_oof(np.c_[oa[0][ii], ob[0][ii], oc[0][ii]], Yc[0][ii], ans_c[ii])
            inc = (np.full_like(Yc[0], np.nan), np.full_like(Yc[0], np.nan)); inc[0][ii] = base; inc[1][ii] = full
        for sn, m0 in masks.items():
            m = m0 & om
            rs = [res_r2(Yc[j], oc[j], pt[j], m) for j in range(len(Yc))]
            d = dict(n=int(m.sum()), r2_a=sscore(Ya[0], oa[0], m)["r2"], r2_b=sscore(Yb[0], ob[0], m)["r2"],
                     r2_c=sscore(Yc[0], oc[0], m)["r2"], r2_transport=sscore(Yc[0], pt[0], m)["r2"], res_r2=rs[0],
                     res_shuf_mean=float(np.nanmean(rs[1:])) if len(rs) > 1 and np.isfinite(rs[1:]).any() else NAN,
                     res_p95=float(np.nanpercentile(rs[1:], 95)) if len(rs) > 1 and np.isfinite(rs[1:]).any() else NAN)
            if inc is not None:
                d["incr_refit"] = sscore(Yc[0], inc[1], m)["r2"] - sscore(Yc[0], inc[0], m)["r2"]
            out.setdefault(OPS[op], {})[sn] = d
    return out


def setv(d, path, l_, val, n):   # d[path...][l_] = val, lists of length n pre-filled with NaN
    for p in path[:-1]:
        d = d.setdefault(p, {})
    d.setdefault(path[-1], [NAN] * n)[l_] = val


def nanstat(x, f):
    x = [v for v in x if np.isfinite(v)]
    return float(f(x)) if x else NAN


# ------------------------------------------------------------------ logged values, row attributes
def logged_at(run_json, step):
    """The logged eval entries to compare a checkpoint with: the exact step, else the two neighbouring evals (nearest
    first)."""
    cur = {e["step"]: e for e in run_json["curve"]}
    if step in cur:
        return [(step, cur[step], True)]
    lo = max([s for s in cur if s < step], default=None); hi = min([s for s in cur if s > step], default=None)
    nb = sorted([s for s in (lo, hi) if s is not None], key=lambda s: abs(s - step))
    return [(s, cur[s], False) for s in nb]


def strict_acc(run_json, step, hits):
    import rainbow6_summary as RS
    r = dict(run_json); r["curve"] = [{"step": step, "hits_dim_held": hits}]
    return RS.strict_series(r)[0][1]


def strict_logged(run_json):
    import rainbow6_summary as RS
    return dict(RS.strict_series(run_json))


def row_attrs(st, meta, rj):
    """Per answer row: strict-held membership (rainbow6_summary.strict_mask over eval_held, in eval_held q order);
    item site -> its row's answer index."""
    import rainbow6_summary as RS
    held_q = [int(q) for q in meta["dinfo"]["eval_held"]["q"]]
    pos_of = {q: i for i, q in enumerate(held_q)}
    assert len(pos_of) == len(held_q), "duplicate held questions"
    m, _ = RS.strict_mask(rj)
    sq = np.asarray(m, bool)
    assert len(sq) == len(held_q), (len(sq), len(held_q))
    held = st["ans_set"] == 0
    a_strict = np.zeros(len(st["ans_c"]), bool)
    a_strict[held] = sq[[pos_of[int(q)] for q in st["ans_q"][held]]]
    idx = {(int(s), int(r)): i for i, (s, r) in enumerate(zip(st["ans_set"], st["ans_row"]))}
    i2a = np.array([idx[(int(s), int(r))] for s, r in zip(st["item_set"], st["item_row"])])
    return a_strict, i2a, sq


def subset_masks(ps, at, hit):
    held = at["set"] == 0
    m = {"pooled": np.ones(len(held), bool), "strict": at["strict"], "nonstrict_held": held & ~at["strict"],
         "seen": at["set"] == 1, "hit": hit, "miss": ~hit}
    if ps == "item":
        m.update(A=at["slot"] == 0, B=at["slot"] == 1, strict_A=at["strict"] & (at["slot"] == 0))
    return {k: m[k] for k in SUBS[ps]}


# ------------------------------------------------------------------ one run
def analyse(run_dir: Path, steps, out: Path, label=None, run_json_path=None, wts_override=None) -> dict:
    run_dir = Path(run_dir)
    jp = Path(run_json_path) if run_json_path else None
    if jp is None:
        js = sorted(run_dir.glob("run_*.json"))
        assert len(js) == 1, ("expected one run JSON", js)
        jp = js[0]
    rj = json.loads(jp.read_text())
    a = rj["args"]
    arm = ("all6" if set(a.get("prior_domains", "").split(",")) == SIX else
           "only-dim" if a.get("prior_domains") == "dim" else a.get("prior_domains", "?"))
    label = label or f"{arm}_s{a['seed']}"
    if wts_override is not None:
        wts = wts_override
    else:
        wts = []
        for s in steps:
            w = sorted(run_dir.glob(f"wts_*_step{s}.pt"))
            assert len(w) == 1, ("expected one weight file", s, w)
            wts.append(w[0])
    print(f"\n=== {label}: {run_dir}\n    run JSON {jp.name} (code_sha {rj.get('code_sha')}), steps {list(steps)}", flush=True)
    t0 = time.time()
    ddir = out / f"dump_{label}"
    meta = dump(rj, wts, ddir)
    st = dict(np.load(ddir / "static.npz"))
    print(f"    dump {meta['dump_wall_s']} s (trainer setup {meta['setup_s']} s; forward {[c['forward_s'] for c in meta['ckpts']]} s); "
          f"{meta['n_layers']} blocks + embedding; read layers {meta['read_layers']}; seq {meta['seq']}; source trainer sha {meta['src_sha']}")
    print(f"    probe args == run JSON args; probe dim_held / dim_seen questions, contexts, cls, masks == run JSON domains.dim")
    print("    decoded prompts (probed positions in ⟦ ⟧ = the two item sites and the answer-prediction token; scored span « »):")
    for p in meta["prompts"]:
        print(f"      {p['set']} row {p['row']}: seq_len {p['seq_len']}, item sites {p['item_sites']}, answer-pred pos "
              f"{p['answer_pred_pos']}, answer tokens {p['answer_tokens']}, question {p['question']} -> {p['answer']}; "
              f"tokens {p['shown_from']}..: {p['text']!r}")
    S = [dict(np.load(ddir / c["file"])) for c in meta["ckpts"]]
    res = analyse_states(rj, jp, meta, st, S, steps, out, label, arm, run_dir, t_start=t0)
    return res


def analyse_states(rj, jp, meta, st, S, steps, out, label, arm, run_dir, logged_strict=None, run_checks=True, t_start=None):
    """(a)-(d) from dumped states S[k] = {item: [L+1, n_item, D], ans: [L+1, n_ans, D], ans_hit: [n_ans]}."""
    t0 = time.time() if t_start is None else t_start
    T = st["target"]; nL = meta["n_layers"]; rl = meta["read_layers"]; NL = nL + 1; K = len(S)
    a_strict, i2a, sq = row_attrs(st, meta, rj)
    ops = sorted(int(o) for o in np.unique(st["ans_op"]))
    assert set(ops) <= set(OPS), ops
    for op in ops:   # the exponent algebra the transport formula uses (crash before analysis)
        om = st["ans_op"] == op; sg = 1 if OPS[op] == "times" else -1
        assert np.array_equal(T[st["ans_c"][om]], T[st["ans_a"][om]] + sg * T[st["ans_b"][om]]), ("E(c) != E(a) +- E(b)", op)
    for p in meta["prompts"]:
        assert f" {OPS[int(p['question'][1])]}" in p["text"], ("op token / name mismatch", p["question"], p["text"])
    at = {"item": dict(set=st["item_set"], strict=a_strict[i2a], slot=st["item_slot"]),
          "ans": dict(set=st["ans_set"], strict=a_strict)}
    hits = [np.asarray(sd["ans_hit"]).astype(bool) for sd in S]
    masks = [{"item": subset_masks("item", at["item"], h[i2a]), "ans": subset_masks("ans", at["ans"], h)} for h in hits]
    print(f"    rows: item sites {len(st['item_v'])} (dim_held + dim_seen, 2 per row), answer positions {len(st['ans_c'])}; "
          f"distinct items at sites {len(np.unique(st['item_v']))}, distinct answers {len(np.unique(st['ans_c']))}; strict "
          f"held questions {int(sq.sum())}/{len(sq)}; ops {[(OPS[o], int((st['ans_op'] == o).sum())) for o in ops]}")
    for ps in ("item", "ans"):
        print(f"    subset n ({ps}): " + ", ".join(f"{sn} {'/'.join(str(int(masks[k][ps][sn].sum())) for k in range(K))}"
                                              for sn in SUBS[ps]) + "  (per checkpoint)")
    # targets: true + item-level shuffles of the 27-row exponent table (same shuffles at every layer / checkpoint)
    rng = np.random.default_rng(12345)
    perms = [rng.permutation(len(T)) for _ in range(N_SHUF)]
    tabs = [T] + [T[p] for p in perms]
    pos = {"item": (st["item_v"], st["item_v"]), "ans": (st["ans_c"], st["ans_c"])}
    Ks = {ps: [centred_gram(tb[v]) for tb in tabs] for ps, (v, _) in pos.items()}
    Ys = {ps: [tb[v] for tb in tabs] for ps, (v, _) in pos.items()}
    Ya, Yb = [tb[st["ans_a"]] for tb in tabs], [tb[st["ans_b"]] for tb in tabs]
    base = {ps: const_baseline(Ys[ps][0], g) for ps, (_, g) in pos.items()}
    cpred = {ps: const_pred(Ys[ps][0], g) for ps, (_, g) in pos.items()}
    drng = np.random.default_rng(777)
    res = dict(label=label, arm=arm, seed=rj["args"]["seed"], run_dir=str(run_dir), run_json=str(jp), steps=list(steps),
               n_layers=nL, read_layers=rl, src_sha=meta["src_sha"], code_sha=rj.get("code_sha"), positions=list(pos),
               n_shuffles=N_SHUF, subsets=SUBS, main_subsets=MAIN,
               strict=dict(n_held_q=len(sq), n_strict_q=int(sq.sum()), mask="".join("1" if x else "0" for x in sq)),
               ridge=dict(cv="9-fold grouped by item (item sites) / by answer item (answer position; E(a), E(b): by a, b), "
                             "fixed permutation seed 0; fitted on all rows, scored per subset from the per-row OOF predictions",
                          alpha="r * mean(S^2) of the centred training states; r chosen by inner 8-fold grouped CV on the "
                                "training items only, grid 10^linspace(-5, 2, 15)", target="exponent vector in {-1,0,1}^3",
                          frozen=f"the {steps[0]} fold fits (alpha chosen at {steps[0]}) applied to the same held-out rows of "
                                 f"the later checkpoints, no refit",
                          transport="residual R2 = 1 - SSE(Ec_hat) / SS(E(c) - (Ea_hat +- Eb_hat)), per op, per subset"),
               const_baseline=base, ckpts=[], drift={}, drift_sub={}, drift_imean={},
               checks=dict(positive=[], answer=[], negative=[]), prompts=meta["prompts"])
    cks = [dict(step=s, wts=str(c["wts"]), forward_s=c["forward_s"], b={}, c={}, neg={}, sub={}, frozen={}, imean={},
                transport={}) for s, c in zip(steps, meta["ckpts"])]
    f32 = lambda *sh: np.full(sh, np.nan, np.float32)
    oo = {}
    for ps, (v, _) in pos.items():
        oo[f"{ps}_pred"] = f32(K, NL, len(v), 3); oo[f"{ps}_err"] = f32(K, NL, len(v))
        oo[f"{ps}_frozen_pred"] = f32(K, NL, len(v), 3)
    oo["ans_a_pred"] = f32(K, NL, len(st["ans_c"]), 3); oo["ans_b_pred"] = f32(K, NL, len(st["ans_c"]), 3)
    for ps, (v, g) in pos.items():
        Y0 = Ys[ps][0]
        for l_ in range(NL):
            Hs = [S[k][ps][l_].astype(np.float64) for k in range(K)]
            dg = [is_degen(H) for H in Hs]
            ext = None
            for k in range(K):
                H, ck, mk = Hs[k], cks[k], masks[k][ps]
                # (b) pooled: the trainer recipe, Gram form (the positive check's number)
                if dg[k]:
                    cs = [NAN] * len(tabs)
                else:
                    Hc = H - H.mean(0); G = Hc @ Hc.T; nG = np.linalg.norm(G)
                    cs = [float((G * K_).sum() / (nG * np.linalg.norm(K_) + 1e-8)) for K_ in Ks[ps]]
                setv(ck, ["b", ps], l_, cs[0], NL)
                # (c) ridge on all rows; at the first checkpoint the fold fits also predict the later ones (frozen)
                if dg[k]:
                    oof = [np.full_like(Y0, np.nan) for _ in tabs]; ch = [[NAN]]
                else:
                    oof, ch, ex = ridge_cv(H, Ys[ps], g, X_extra=[Hs[j] for j in range(1, K)] if k == 0 else ())
                    if k == 0:
                        ext = ex
                sc = [scores(Ys[ps][j], oof[j]) for j in range(len(oof))]
                for kk, val in (("r2", sc[0]["r2"]), ("r2_dims", sc[0]["r2_dims"]), ("lattice_acc", sc[0]["lattice_acc"]),
                                ("alpha_rel_median", nanstat(ch[0], np.median))):
                    setv(ck, ["c", ps, kk], l_, val, NL)
                for kk, val in (("b_mean", nanstat(cs[1:], np.mean)), ("b_p95", nanstat(cs[1:], lambda x: np.percentile(x, 95))),
                                ("r2_mean", nanstat([x["r2"] for x in sc[1:]], np.mean)),
                                ("r2_p95", nanstat([x["r2"] for x in sc[1:]], lambda x: np.percentile(x, 95))),
                                ("lat_mean", nanstat([x["lattice_acc"] for x in sc[1:]], np.mean)),
                                ("lat_p95", nanstat([x["lattice_acc"] for x in sc[1:]], lambda x: np.percentile(x, 95)))):
                    setv(ck, ["neg", ps, kk], l_, val, NL)
                oo[f"{ps}_pred"][k, l_] = oof[0]; oo[f"{ps}_err"][k, l_] = ((oof[0] - Y0) ** 2).sum(1)
                # per-subset scores of the same OOF predictions, (b) on the subset, item-mean (b)
                for sn, m in mk.items():
                    tr_ = sscore(Y0, oof[0], m); sh = [sscore(Ys[ps][j], oof[j], m) for j in range(1, len(oof))]
                    cb = sscore(Y0, cpred[ps], m)
                    bs = [NAN] * len(tabs) if (dg[k] or m.sum() < 3) else cka_subset(H[m], [tb[v][m] for tb in tabs])
                    if sn == "pooled" and np.isfinite(bs[0]) and abs(bs[0] - cs[0]) > 1e-6:
                        warn(f"{label} step {steps[k]} {ps} L{l_}: feature-form CKA {bs[0]:.6f} != Gram-form {cs[0]:.6f}")
                    for kk, val in (("n", tr_["n"]), ("r2", tr_["r2"]), ("lattice_acc", tr_["lattice_acc"]), ("mse", tr_["mse"]),
                                    ("r2_shuf_mean", nanstat([x["r2"] for x in sh], np.mean)),
                                    ("r2_p95", nanstat([x["r2"] for x in sh], lambda x: np.percentile(x, 95))),
                                    ("lat_shuf_mean", nanstat([x["lattice_acc"] for x in sh], np.mean)),
                                    ("lat_p95", nanstat([x["lattice_acc"] for x in sh], lambda x: np.percentile(x, 95))),
                                    ("const_r2", cb["r2"]), ("const_lat", cb["lattice_acc"]),
                                    ("b", bs[0]), ("b_mean", nanstat(bs[1:], np.mean)),
                                    ("b_p95", nanstat(bs[1:], lambda x: np.percentile(x, 95)))):
                        setv(ck, ["sub", ps, sn, kk], l_, val, NL)
                for sn in MAIN[ps]:
                    items, M = item_mean(H, v, mk[sn])
                    bim = ([NAN] * len(tabs) if (dg[k] or len(items) < 4) else cka_subset(M, [tb[items] for tb in tabs]))
                    for kk, val in (("n_items", len(items)), ("b", bim[0]), ("b_mean", nanstat(bim[1:], np.mean)),
                                    ("b_p95", nanstat(bim[1:], lambda x: np.percentile(x, 95)))):
                        setv(ck, ["imean", ps, sn, kk], l_, val, NL)
                # (c') answer position: E(a), E(b) decoders and the transport residual
                if ps == "ans":
                    if dg[k]:
                        oa = ob = [np.full_like(Y0, np.nan) for _ in tabs]
                    else:
                        oa = ridge_cv(H, Ya, st["ans_a"])[0]; ob = ridge_cv(H, Yb, st["ans_b"])[0]
                    oo["ans_a_pred"][k, l_] = oa[0]; oo["ans_b_pred"][k, l_] = ob[0]
                    tsx = transport_stats(Ys[ps], oof, Ya, oa, Yb, ob, st["ans_op"], st["ans_c"],
                                          {sn: mk[sn] for sn in TRANS_SUBS})
                    for opn, dd in tsx.items():
                        for sn, d in dd.items():
                            for kk, val in d.items():
                                setv(ck, ["transport", opn, sn, kk], l_, val, NL)
            # frozen decoder: the first checkpoint's fold fits on the later checkpoints' states
            for j in range(1, K):
                P = ext[j - 1] if (ext is not None and not dg[j]) else [np.full_like(Y0, np.nan) for _ in tabs]
                oo[f"{ps}_frozen_pred"][j, l_] = P[0]
                for sn, m in masks[j][ps].items():
                    tr_ = sscore(Y0, P[0], m); sh = [sscore(Ys[ps][jj], P[jj], m) for jj in range(1, len(P))]
                    for kk, val in (("n", tr_["n"]), ("r2", tr_["r2"]), ("lattice_acc", tr_["lattice_acc"]), ("mse", tr_["mse"]),
                                    ("r2_p95", nanstat([x["r2"] for x in sh], lambda x: np.percentile(x, 95))),
                                    ("lat_p95", nanstat([x["lattice_acc"] for x in sh], lambda x: np.percentile(x, 95)))):
                        setv(cks[j], ["frozen", ps, sn, kk], l_, val, NL)
            # (a) drift between checkpoints of this run (same prompts): raw rows and item-mean, with permutation nulls
            for j in range(1, K)[::-1]:
                key = f"{steps[0]}->{steps[j]}"
                setv(res["drift"], [key, ps], l_, cka_lin(Hs[0], Hs[j]), NL)
                for sn in MAIN[ps]:
                    m = masks[0][ps][sn]
                    A_, B_ = Hs[0][m], Hs[j][m]
                    nm, n95 = cka_lin_null(A_, B_, [drng.permutation(len(A_)) for _ in range(N_SHUF)])
                    for kk, val in (("a", cka_lin(A_, B_)), ("null_mean", nm), ("null_p95", n95)):
                        setv(res["drift_sub"], [key, ps, sn, kk], l_, val, NL)
                    it0, M0 = item_mean(Hs[0], v, m); it1, M1 = item_mean(Hs[j], v, m)
                    assert np.array_equal(it0, it1)
                    nm, n95 = ((NAN, NAN) if len(it0) < 4 else
                               cka_lin_null(M0, M1, [drng.permutation(len(it0)) for _ in range(N_SHUF)]))
                    for kk, val in (("a", cka_lin(M0, M1) if len(it0) >= 4 else NAN), ("null_mean", nm), ("null_p95", n95),
                                    ("n_items", len(it0))):
                        setv(res["drift_imean"], [key, ps, sn, kk], l_, val, NL)
    for k, (s, c) in enumerate(zip(steps, meta["ckpts"])):
        # (d) the model's own accuracy (this probe's hits; the trainer's exact-match code)
        hstr = c["hits"]["dim_held"]
        cks[k]["d"] = dict(dim_held=c["acc"]["dim_held"], dim_held_masks=c["acc_masks"]["dim_held"], dim_seen=c["acc"]["dim_seen"],
                           strict=strict_acc(rj, s, hstr), hits_dim_held=hstr, trainer_recipe_cka=c["trainer_cka"],
                           trainer_recipe_cka_fp32=c["trainer_cka_fp32"])
        res["ckpts"].append(cks[k])
    res["logged_strict"] = (sorted(strict_logged(rj).items()) if (logged_strict is None and rj.get("curve")) else
                            (logged_strict or []))
    if run_checks:
        checks(res, rj, meta)
    np.savez_compressed(out / f"oof_{label}.npz", steps=np.asarray(steps), target=T, item_v=st["item_v"],
                        item_slot=st["item_slot"], item_set=st["item_set"], item_strict=at["item"]["strict"], item_to_ans=i2a,
                        ans_c=st["ans_c"], ans_a=st["ans_a"], ans_b=st["ans_b"], ans_op=st["ans_op"], ans_set=st["ans_set"],
                        ans_strict=a_strict, ans_hit=np.stack(hits).astype(np.int8), **oo)
    res["oof_file"] = str(out / f"oof_{label}.npz")
    res["analysis_s"] = round(time.time() - t0 - meta.get("dump_wall_s", 0), 1); res["dump_s"] = meta.get("dump_wall_s", 0)
    res["setup_s"] = meta.get("setup_s")
    table(res)
    (out / f"{label}.json").write_text(json.dumps(res, indent=1))
    print(f"    wrote {out / (label + '.json')} + oof_{label}.npz (dump {res['dump_s']} s, analysis {res['analysis_s']} s)", flush=True)
    return res


def checks(res, rj, meta):
    sl = strict_logged(rj) if rj.get("curve") else {}
    print("  checks (flag only):")
    for ck in res["ckpts"]:
        s = ck["step"]
        for es, e, exact in logged_at(rj, s):
            tag = "exact step" if exact else f"nearest eval {es} (no eval at {s})"
            for L_ in res["read_layers"]:
                lk = f"cka_dim@L{L_}"
                if lk not in e:
                    continue
                mine, rep, lg = ck["b"]["item"][L_], ck["d"]["trainer_recipe_cka"][lk], e[lk]
                r32 = ck["d"]["trainer_recipe_cka_fp32"][lk]
                dlt = mine - lg
                res["checks"]["positive"].append(dict(step=s, logged_step=es, exact=exact, layer=L_, probe=mine,
                                                      trainer_recipe_bf16=rep, trainer_recipe_fp32=r32, logged=lg, diff=dlt,
                                                      diff_recipe_bf16=rep - lg))
                print(f"    positive  step {s} L{L_}: probe (b, float64) {mine:.5f} | trainer recipe on the same states: fp32 "
                      f"{r32:.5f}, bf16 autocast as logged {rep:.5f} | logged {lk} @{es} {lg:.5f} -> diff (b) {dlt:+.5f}, "
                      f"diff bf16 recipe {rep - lg:+.5f} [{tag}]")
                if abs(mine - r32) > 1e-3:
                    warn(f"step {s} L{L_}: probe CKA != the trainer recipe in fp32 on the same states ({mine:.5f} vs {r32:.5f})")
                if abs(dlt) > 0.01 and es == logged_at(rj, s)[0][0]:
                    warn(f"step {s} L{L_}: probe CKA {mine:.4f} vs logged {lk}@{es} {lg:.4f}: |diff| {abs(dlt):.4f} > 0.01 ({tag})")
            d = ck["d"]
            lh = e.get("dim_held"); lhs = e.get("hits_dim_held"); lst = sl.get(es)
            nd = sum(x != y for x, y in zip(d["hits_dim_held"], lhs)) if lhs else None
            res["checks"]["answer"].append(dict(step=s, logged_step=es, exact=exact, probe_dim_held=d["dim_held"], logged_dim_held=lh,
                                                probe_strict=d["strict"], logged_strict=lst, questions_differing=nd))
            print(f"    answer    step {s}: probe dim_held {d['dim_held']:.4f} vs logged @{es} {lh if lh is None else round(lh, 4)}; "
                  f"strict {d['strict']:.4f} vs {lst if lst is None else round(lst, 4)}; hits_dim_held questions differing "
                  f"{nd}/{len(d['hits_dim_held'])} [{tag}]")
            if lh is not None and abs(d["dim_held"] - lh) > 0.02 and es == logged_at(rj, s)[0][0]:
                warn(f"step {s}: probe dim_held {d['dim_held']:.4f} vs logged @{es} {lh:.4f} (|diff| > 0.02, {tag})")
    for ck in res["ckpts"]:
        for ps in res["positions"]:
            n_, c_, b_ = ck["neg"][ps], ck["c"][ps], res["const_baseline"][ps]
            if not np.isfinite(n_["r2_mean"]).any():
                continue
            i_r = int(np.nanargmax(n_["r2_mean"])); i_l = int(np.nanargmax(n_["lat_mean"])); i_b = int(np.nanargmax(n_["b_mean"]))
            res["checks"]["negative"].append(dict(step=ck["step"], pos=ps, max_shuf_r2_mean=n_["r2_mean"][i_r],
                                                  max_shuf_lat_mean=n_["lat_mean"][i_l], max_shuf_b_mean=n_["b_mean"][i_b],
                                                  const_lat=b_["lattice_acc"], const_r2=b_["r2"]))
            print(f"    negative  step {ck['step']} {ps:4s}: shuffled target, max over layers of the mean over {N_SHUF}: "
                  f"(b) {n_['b_mean'][i_b]:.3f} (L{i_b}; true {ck['b'][ps][i_b]:.3f}) | (c) R2 {n_['r2_mean'][i_r]:+.3f} "
                  f"(L{i_r}; true {c_['r2'][i_r]:+.3f}) | lattice {n_['lat_mean'][i_l]:.3f} (L{i_l}; true {c_['lattice_acc'][i_l]:.3f}; "
                  f"constant predictor {b_['lattice_acc']:.3f}, chance 1/27 = 0.037)")
            if n_["r2_mean"][i_r] > 0.05 or n_["lat_mean"][i_l] > b_["lattice_acc"] + 0.05:
                warn(f"step {ck['step']} {ps}: shuffled-target probe above baseline (R2 {n_['r2_mean'][i_r]:.3f}, "
                     f"lattice {n_['lat_mean'][i_l]:.3f} vs constant {b_['lattice_acc']:.3f})")


# ------------------------------------------------------------------ tables
def f_(x, w=9, p=3, sg=True):
    if x is None or not np.isfinite(x):
        return f"{'nan':>{w}s}"
    return f"{x:+{w}.{p}f}" if sg else f"{x:{w}.{p}f}"


def ptab(title, cols, nrows, w=9):
    print(title)
    print("     L | " + " ".join(f"{n:>{w}s}" for n, _ in cols))
    for l_ in range(nrows):
        print(f"    {l_:>2d} | " + " ".join(f_(v[l_], w) for _, v in cols))


def mx(lists):   # elementwise max over checkpoints (shuffle baselines: the largest one is printed)
    return [nanstat([x[l_] for x in lists], np.max) for l_ in range(len(lists[0]))]


def table(res):
    st_, NL, lab, C = res["steps"], res["n_layers"] + 1, res["label"], res["ckpts"]
    print(f"  tables [{lab}]: R2 against the subset's own mean; shp95 = max over checkpoints of the shuffled-target R2 p95; "
          f"lat_ch = constant-predictor lattice accuracy on the subset (chance 1/27 = 0.037); latp95 = shuffled lattice p95; "
          f"fz@s = the {st_[0]} decoder frozen and applied at s; b_shm / bim_shm = max over checkpoints of the shuffled (b) "
          f"mean; a / aim null = permutation null mean (max over pairs); NaN = degenerate cell")
    for ps in res["positions"]:
        for sn in MAIN[ps]:
            sub = [c["sub"][ps][sn] for c in C]
            ns = "/".join(str(int(np.nanmax(np.asarray(x["n"], float)))) for x in sub)
            ni = C[0]["imean"][ps][sn]["n_items"][0]
            cols = [(f"R2@{s}", x["r2"]) for s, x in zip(st_, sub)] + [("shp95", mx([x["r2_p95"] for x in sub]))] + \
                   [(f"fz@{s}", c["frozen"][ps][sn]["r2"]) for s, c in zip(st_[1:], C[1:])] + \
                   [(f"lat@{s}", x["lattice_acc"]) for s, x in zip(st_, sub)] + [("lat_ch", sub[0]["const_lat"]),
                                                                                 ("latp95", mx([x["lat_p95"] for x in sub]))]
            ptab(f"  [{lab}] {ps} subset={sn} (n rows {ns}; items {ni}) -- (c) ridge R2 | frozen-decoder R2 | lattice acc", cols, NL)
            dk = [f"{st_[0]}->{s}" for s in st_[1:]]
            dsub = [res["drift_sub"][k][ps][sn] for k in dk]; dim_ = [res["drift_imean"][k][ps][sn] for k in dk]
            cols = [(f"a@{k.split('->')[1]}", d["a"]) for k, d in zip(dk, dsub)] + [("a_null", mx([d["null_mean"] for d in dsub]))] + \
                   [(f"aim@{k.split('->')[1]}", d["a"]) for k, d in zip(dk, dim_)] + [("aim_null", mx([d["null_mean"] for d in dim_]))] + \
                   [(f"b@{s}", x["b"]) for s, x in zip(st_, sub)] + [("b_shm", mx([x["b_mean"] for x in sub]))] + \
                   [(f"bim@{s}", c["imean"][ps][sn]["b"]) for s, c in zip(st_, C)] + \
                   [("bim_shm", mx([c["imean"][ps][sn]["b_mean"] for c in C]))]
            ptab(f"  [{lab}] {ps} subset={sn} -- (a) drift CKA from {st_[0]} (raw rows, item-mean) | (b) CKA to the exponent "
                 f"geometry (raw rows, item-mean {ni} x D)", cols, NL)
        for sn in [x for x in SUBS[ps] if x not in MAIN[ps]]:
            sub = [c["sub"][ps][sn] for c in C]
            ns = "/".join(str(int(np.nanmax(np.asarray(x["n"], float)))) for x in sub)
            cols = [(f"R2@{s}", x["r2"]) for s, x in zip(st_, sub)] + [("shp95", mx([x["r2_p95"] for x in sub]))] + \
                   [(f"lat@{s}", x["lattice_acc"]) for s, x in zip(st_, sub)] + [("lat_ch", mx([x["const_lat"] for x in sub])),
                                                                                 ("latp95", mx([x["lat_p95"] for x in sub]))] + \
                   [(f"b@{s}", x["b"]) for s, x in zip(st_, sub)] + [("b_shm", mx([x["b_mean"] for x in sub]))]
            ptab(f"  [{lab}] {ps} subset={sn} (n rows per checkpoint {ns}) -- (c) R2 | lattice | (b)", cols, NL)
    for c in C:
        for opn, dd in c["transport"].items():
            cols = []
            for sn, tg in (("strict", "s"), ("pooled", "p")):
                d = dd[sn]
                cols += [(f"{tg}:R2a", d["r2_a"]), (f"{tg}:R2b", d["r2_b"]), (f"{tg}:R2c", d["r2_c"]), (f"{tg}:R2tr", d["r2_transport"]),
                         (f"{tg}:res", d["res_r2"]), (f"{tg}:resp95", d["res_p95"])]
            ns = ", ".join(f"{sn} {int(np.nanmax(np.asarray(dd[sn]['n'], float)))}" for sn in TRANS_SUBS)
            ptab(f"  [{lab}] ans transport control step {c['step']} op={opn} (n {ns}; s: strict, p: pooled) -- R2 of E(a), E(b), "
                 f"E(c), of the formula Ea_hat {'+' if opn == 'times' else '-'} Eb_hat; res = residual R2 (> 0: answer beyond "
                 f"transported inputs); resp95 = shuffled p95", cols, NL)
    print("  (d) " + "; ".join(f"step {c['step']}: dim_held {c['d']['dim_held']:.4f}, strict {c['d']['strict']:.4f}, "
                               f"dim_seen {c['d']['dim_seen']:.4f}" for c in C))


# ------------------------------------------------------------------ summary
def _d(a, b):
    return [x - y for x, y in zip(a, b)]


def _did(a0, a1, o0, o1):
    return [(w - x) - (y - z) for w, x, y, z in zip(a1, a0, o1, o0)]


def nearest(ls, step):
    ls = [(int(s), v) for s, v in ls]
    return min(ls, key=lambda sv: abs(sv[0] - step)) if ls else (None, NAN)


def summary(paths):
    rs = [json.loads(Path(p).read_text()) for p in paths]
    by = {}
    for r in rs:
        by.setdefault(r["seed"], {})[r["arm"]] = r
    print("\n=== summary: all6 - only-dim per layer (positive = all6 higher). DiD = (all6@last - all6@first) - "
          "(only-dim@last - only-dim@first), first/last = the first/last checkpoint; DiD_fz uses the frozen first-checkpoint "
          "decoder at the last checkpoint minus the refitted R2 at the first; daim@s = item-mean drift CKA first->s (= its "
          "DiD, drift is 1 at the first checkpoint); da = raw-row drift; NaN = degenerate cell")
    out = {}
    for seed, arms in sorted(by.items()):
        if not {"all6", "only-dim"} <= set(arms):
            print(f"  seed {seed}: arms {sorted(arms)} (needs all6 and only-dim), skipped"); continue
        A, O = arms["all6"], arms["only-dim"]
        assert A["n_layers"] == O["n_layers"] and len(A["ckpts"]) == len(O["ckpts"])
        st_ = A["steps"]; f, l = 0, len(st_) - 1; NL = A["n_layers"] + 1
        print(f"  seed {seed}: all6 steps {A['steps']} | only-dim steps {O['steps']}")
        if A["strict"]["mask"] != O["strict"]["mask"]:
            warn(f"seed {seed}: the strict question sets of all6 and only-dim differ (subset differences mix row sets)")
        out[seed] = {}
        for ps in A["positions"]:
            out[seed][ps] = {}
            for sn in A["main_subsets"][ps]:
                sa = [c["sub"][ps][sn] for c in A["ckpts"]]; so = [c["sub"][ps][sn] for c in O["ckpts"]]
                cols = {f"dR2@{s}": _d(x["r2"], y["r2"]) for s, x, y in zip(st_, sa, so)}
                cols["DiD_R2"] = _did(sa[f]["r2"], sa[l]["r2"], so[f]["r2"], so[l]["r2"])
                cols["DiD_lat"] = _did(sa[f]["lattice_acc"], sa[l]["lattice_acc"], so[f]["lattice_acc"], so[l]["lattice_acc"])
                for s, ca, co in zip(st_[1:], A["ckpts"][1:], O["ckpts"][1:]):
                    cols[f"dfz@{s}"] = _d(ca["frozen"][ps][sn]["r2"], co["frozen"][ps][sn]["r2"])
                cols["DiD_fz"] = _did(sa[f]["r2"], A["ckpts"][l]["frozen"][ps][sn]["r2"], so[f]["r2"],
                                      O["ckpts"][l]["frozen"][ps][sn]["r2"])
                for s in st_[1:]:
                    k = f"{st_[0]}->{s}"
                    cols[f"daim@{s}"] = _d(A["drift_imean"][k][ps][sn]["a"], O["drift_imean"][k][ps][sn]["a"])
                k = f"{st_[0]}->{st_[l]}"
                cols[f"da@{st_[l]}"] = _d(A["drift_sub"][k][ps][sn]["a"], O["drift_sub"][k][ps][sn]["a"])
                ia = [c["imean"][ps][sn]["b"] for c in A["ckpts"]]; io = [c["imean"][ps][sn]["b"] for c in O["ckpts"]]
                cols["DiD_bim"] = _did(ia[f], ia[l], io[f], io[l])
                cols["DiD_b"] = _did(sa[f]["b"], sa[l]["b"], so[f]["b"], so[l]["b"])
                out[seed][ps][sn] = cols
                ns = "/".join(str(int(np.nanmax(np.asarray(x["n"], float)))) for x in sa)
                ptab(f"    seed {seed} position={ps} subset={sn} (n rows {ns})", list(cols.items()), NL)
        tr = {}
        for opn in A["ckpts"][0]["transport"]:
            tr[opn] = {}
            for sn in ("strict", "pooled"):
                ta = [c["transport"][opn][sn]["res_r2"] for c in A["ckpts"]]; to = [c["transport"][opn][sn]["res_r2"] for c in O["ckpts"]]
                cols = {f"dres@{s}": _d(x, y) for s, x, y in zip(st_, ta, to)}
                cols["DiD_res"] = _did(ta[f], ta[l], to[f], to[l])
                tr[opn][sn] = cols
            ptab(f"    seed {seed} ans transport op={opn}: residual R2, all6 - only-dim (s: strict, p: pooled)",
                 [(f"{'s' if sn == 'strict' else 'p'}:{k}", v) for sn in ("strict", "pooled") for k, v in tr[opn][sn].items()], NL)
        out[seed]["transport"] = tr
        # behaviour: the probe's own forward (exact checkpoints) and the V5 runs' own logged curves (nearest evals)
        pa = [c["d"]["strict"] for c in A["ckpts"]]; po = [c["d"]["strict"] for c in O["ckpts"]]
        (la0s, la0), (la1s, la1) = nearest(A["logged_strict"], st_[f]), nearest(A["logged_strict"], st_[l])
        (lo0s, lo0), (lo1s, lo1) = nearest(O["logged_strict"], st_[f]), nearest(O["logged_strict"], st_[l])
        gap_end = la1 - lo1 if np.isfinite(la1 - lo1) else pa[l] - po[l]
        flag = not gap_end >= GAP_MIN
        beh = dict(probe_strict_all6=pa, probe_strict_only=po, probe_diff=_d(pa, po), probe_DiD=(pa[l] - pa[f]) - (po[l] - po[f]),
                   logged_steps=[la0s, la1s, lo0s, lo1s], logged_all6=[la0, la1], logged_only=[lo0, lo1],
                   logged_diff=[la0 - lo0, la1 - lo1], logged_DiD=(la1 - la0) - (lo1 - lo0), gap_end=gap_end, cannot_answer=flag)
        out[seed]["behaviour"] = beh
        print(f"    seed {seed} behaviour, strict accuracy all6 vs only-dim -- probe forward: " + "; ".join(
            f"step {s}: {x:.3f} vs {y:.3f} (diff {x - y:+.3f})" for s, x, y in zip(st_, pa, po)) + f"; DiD {beh['probe_DiD']:+.3f}")
        print(f"    seed {seed} behaviour, logged curves: all6 @{la0s} {la0:.3f} -> @{la1s} {la1:.3f}; only-dim @{lo0s} {lo0:.3f} -> "
              f"@{lo1s} {lo1:.3f}; diff {la0 - lo0:+.3f} -> {la1 - lo1:+.3f}; DiD {beh['logged_DiD']:+.3f}")
        print("    (d) dim_held / strict, all6 vs only-dim: " + "; ".join(
            f"step {ca['step']}: {ca['d']['dim_held']:.3f}/{ca['d']['strict']:.3f} vs {co['d']['dim_held']:.3f}/{co['d']['strict']:.3f}"
            for ca, co in zip(A["ckpts"], O["ckpts"])))
        if flag:
            print(f"    seed {seed}: CANNOT ANSWER THE QUESTION (behavioural gap absent): end-of-training strict difference "
                  f"{gap_end:+.3f} < {GAP_MIN:.2f}")
    return out


# ------------------------------------------------------------------ selftest / timing
def synth(out: Path, fails: list):
    """Synthetic guard on the REAL row layout (timing dump: 1284 item sites, 642 answer rows; strict set from the
    registered all6 s16 run JSON). 3 layers, the same noise draws for every arm / checkpoint (unchanged cells -> DiD 0):
      L0: item = identity only; ans = one constant state (degenerate, like ' is' at the embedding) -> NaN
      L1: unchanged; item = planted geometry; ans = TRANSPORT only (E(a) Wa + E(b) Wb)      -> residual R2 < 0
      L2: item = planted geometry; ans = COMPUTED answer (E(c) W2) -> residual R2 > 0. only-dim at the last checkpoint
          loses the geometry on the strict rows only (identity states) and gets a sharper geometry on the non-strict
          rows (noise calibrated here so the pooled R2 change is ~0) -> the DiD must be large on strict, ~0 pooled."""
    tdir = R / "tests" / "probe_v5_timing" / "dump_timing"
    st = dict(np.load(tdir / "static.npz")); meta0 = json.loads((tdir / "meta.json").read_text())
    jp = R / "out" / "pt6v2-all6-lr2e-3-s16" / "pt6v2-all6-lr2e-3" / "run_whole_helix3_seed16_pt6v2-all6-lr2e-3.json"
    rj = json.loads(jp.read_text())
    a_strict, i2a, sq = row_attrs(st, meta0, rj)
    T = st["target"]; v, c, a, b = st["item_v"], st["ans_c"], st["ans_a"], st["ans_b"]
    nI, nA, D, nL, steps = len(v), len(c), 48, 2, V5_STEPS
    rng = np.random.default_rng(2026)
    W, W2, Wa, Wb = (rng.standard_normal((3, D)) for _ in range(4))
    Eid = rng.standard_normal((27, D)); ci = rng.standard_normal((nI, D)); ca = rng.standard_normal((nA, D))
    i_str = a_strict[i2a]
    planted = {"item": (T[v] @ W, Eid[v], ci, i_str), "ans": (T[c] @ W2, Eid[c], ca, a_strict)}

    def l2(ps, sig, sig_ns, lose):
        g, e, cx, sm = planted[ps]
        nz = np.where(sm, sig, sig_ns if lose else sig)[:, None]
        X = g + nz * (e + cx)
        if lose:
            X[sm] = 3.0 * e[sm] + sig * cx[sm]   # identity only on the strict rows
        return X

    def r2_pooled(X, ps):
        y = T[v] if ps == "item" else T[c]
        return scores(y, ridge_cv(X, [y], v if ps == "item" else c)[0][0])["r2"]

    cal = {}
    for ps in ("item", "ans"):   # baseline R2 ~0.6 (room for the compensation), then the pooled-neutral non-strict noise
        sig = min((0.5, 0.75, 1.0, 1.5, 2.0, 3.0), key=lambda s_: abs(r2_pooled(l2(ps, s_, s_, False), ps) - 0.6))
        r0 = r2_pooled(l2(ps, sig, sig, False), ps)
        sig_ns = min((sig * f_ for f_ in (1.0, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)),
                     key=lambda s_: abs(r2_pooled(l2(ps, sig, s_, True), ps) - r0))
        cal[ps] = (sig, sig_ns)
        print(f"  synth calibration {ps}: sig {sig} (pooled R2 {r0:+.3f}), non-strict sig at the last checkpoint {sig_ns:.3f} "
              f"(pooled R2 {r2_pooled(l2(ps, sig, sig_ns, True), ps):+.3f})")
    hstr_of = lambda h: "".join(str(int(x)) for x in h[st["ans_set"] == 0][np.argsort(
        [meta0["dinfo"]["eval_held"]["q"].index(int(q)) for q in st["ans_q"][st["ans_set"] == 0]])])
    paths = []
    for arm in ("all6", "only-dim"):
        S, cks = [], []
        for k, s in enumerate(steps):
            lose = arm == "only-dim" and s == steps[-1]
            item = np.zeros((nL + 1, nI, D)); ans = np.zeros((nL + 1, nA, D))
            item[0] = Eid[v]; ans[0] = 1.0
            item[1] = T[v] @ W + 0.5 * (Eid[v] + ci)
            ans[1] = T[a] @ Wa + T[b] @ Wb + 0.5 * (0.5 * (Eid[a] + Eid[b]) + ca)
            item[2] = l2("item", *cal["item"], lose); ans[2] = l2("ans", *cal["ans"], lose)
            hit = np.ones(nA, np.int8); hit[np.random.default_rng(5).random(nA) < 0.2] = 0
            if lose:
                hit[a_strict] = 0
            S.append(dict(item=item.astype(np.float32), ans=ans.astype(np.float32), ans_hit=hit))
            hs = hstr_of(hit)
            cks.append(dict(wts="synthetic", forward_s=0.0, hits={"dim_held": hs}, acc={"dim_held": float(np.mean([int(x) for x in hs])),
                            "dim_seen": float(hit[st["ans_set"] == 1].mean())}, acc_masks={"dim_held": {}}, trainer_cka={}, trainer_cka_fp32={}))
        meta = dict(meta0, n_layers=nL, read_layers=[1, 2], ckpts=cks, src_sha="synthetic")
        rjs = dict(rj, args=dict(rj["args"], seed=99))
        ls = [(7250, 0.5), (18309, 0.8)] if arm == "all6" else [(7250, 0.5), (18309, 0.5)]
        analyse_states(rjs, jp, meta, st, S, steps, out, f"synth_{arm}", arm, "synthetic", logged_strict=ls, run_checks=False)
        paths.append(out / f"synth_{arm}.json")
    sm = summary(paths)[99]
    A_ = json.loads(paths[0].read_text()); O_ = json.loads(paths[1].read_text())
    for ps, sn in (("ans", "strict"), ("item", "strict_A")):
        ds, dp = sm[ps][sn]["DiD_R2"][2], sm[ps]["pooled"]["DiD_R2"][2]
        print(f"  synth {ps}: L2 DiD R2 {sn} {ds:+.3f} (must be > 0.30), pooled {dp:+.3f} (must be |.| < 0.10); "
              f"DiD_fz {sn} {sm[ps][sn]['DiD_fz'][2]:+.3f}, pooled {sm[ps]['pooled']['DiD_fz'][2]:+.3f}")
        if not (ds > 0.30):
            fails.append(f"synth {ps}: strict DiD {ds:+.3f} did not detect the planted strict-only loss")
        if not (abs(dp) < 0.10):
            fails.append(f"synth {ps}: pooled DiD {dp:+.3f} not ~0 (the calibrated pooled-neutral case)")
        if not (sm[ps][sn]["DiD_fz"][2] > 0.30):
            fails.append(f"synth {ps}: frozen-decoder DiD on {sn} did not detect the loss")
        z = [abs(x) for sn_ in MAIN[ps] for col in sm[ps][sn_].values() for x in [col[1]] if np.isfinite(x)]
        if max(z) > 1e-9:
            fails.append(f"synth {ps}: L1 (unchanged) summary not 0: {max(z):.2e}")
    c0 = A_["ckpts"][0]
    nan_ok = all(not np.isfinite(x) for x in (c0["b"]["ans"][0], c0["c"]["ans"]["r2"][0], c0["c"]["ans"]["lattice_acc"][0],
                                              c0["sub"]["ans"]["strict"]["r2"][0], c0["sub"]["ans"]["pooled"]["b"][0],
                                              c0["imean"]["ans"]["pooled"]["b"][0], A_["drift"]["7323->18309"]["ans"][0]))
    print(f"  synth L0 answer position (constant state): b / R2 / lattice / item-mean b / drift all NaN: {nan_ok}")
    if not nan_ok:
        fails.append("synth: degenerate L0 answer cells not NaN")
    for opn in ("times", "per"):
        t1, t2 = c0["transport"][opn]["pooled"], c0["transport"][opn]["strict"]
        print(f"  synth transport {opn}: L1 (transport only) residual R2 pooled {t1['res_r2'][1]:+.3f} strict {t2['res_r2'][1]:+.3f}, "
              f"R2 E(a) {t1['r2_a'][1]:+.3f} E(b) {t1['r2_b'][1]:+.3f} E(c) {t1['r2_c'][1]:+.3f} formula {t1['r2_transport'][1]:+.3f}, "
              f"refit incr {t1['incr_refit'][1]:+.3f} | L2 (computed answer) residual R2 pooled {t1['res_r2'][2]:+.3f}, "
              f"R2 E(c) {t1['r2_c'][2]:+.3f} formula {t1['r2_transport'][2]:+.3f}, refit incr {t1['incr_refit'][2]:+.3f} | "
              f"shuffled p95 L1 {t1['res_p95'][1]:+.3f} L2 {t1['res_p95'][2]:+.3f}")
        if not (t1["res_r2"][1] < 0 and t1["res_r2"][2] > 0.2 and t1["res_r2"][2] > t1["res_p95"][2]):
            fails.append(f"synth transport {opn}: residual R2 does not separate transport (L1 {t1['res_r2'][1]:+.3f}) from "
                         f"computed (L2 {t1['res_r2'][2]:+.3f})")
    beh = sm["behaviour"]
    print(f"  synth behaviour: logged gap at the end {beh['gap_end']:+.3f}, DiD {beh['logged_DiD']:+.3f}, flagged {beh['cannot_answer']}")
    if beh["cannot_answer"]:
        fails.append("synth: a 30-pt behavioural gap was flagged as absent")
    oo = np.load(out / "oof_synth_only-dim.npz")
    print(f"  synth oof file: {sorted(oo.files)[:6]}... item_pred {oo['item_pred'].shape}, ans_frozen_pred {oo['ans_frozen_pred'].shape}")


def selftest(out: Path):
    sm = R / "out_v5" / "smoke"
    steps = (5, 10, 19)   # smoke C/D weights, in the roles of 7323 / 12000 / 18309; the curve has evals at 0 and 19 only
    fails = []
    rc = analyse(sm / "C", steps, out, label="smokeC")
    rd = analyse(sm / "D", steps, out, label="smokeD")   # D = C's config with a preemption at 10 + resume
    # the exactness expectations on CPU (the smoke ran on CPU too): step 19 has an exact eval
    for r in (rc, rd):
        p19 = [c for c in r["checks"]["positive"] if c["step"] == 19]
        a19 = [c for c in r["checks"]["answer"] if c["step"] == 19]
        print(f"  selftest {r['label']}: step-19 positive |diff| max {max(abs(c['diff']) for c in p19):.2e}; answer: dim_held "
              f"{a19[0]['probe_dim_held']} vs {a19[0]['logged_dim_held']}, questions differing {a19[0]['questions_differing']}")
        print(f"  selftest {r['label']}: step-19 bf16 trainer recipe vs logged |diff| max "
              f"{max(abs(c['diff_recipe_bf16']) for c in p19):.2e} (CPU smoke vs CPU probe: expect ~0)")
        # the CPU smoke's logged CKA is itself a bf16 number (its reduction runs under autocast, thread count 3 vs 16 here):
        # tolerance = 3 bf16 ulps of the logged value (ulp = 2^(floor(log2 x) - 7)); (b) itself must be within the 0.01 flag
        for c in p19:
            ulp = 2.0 ** (np.floor(np.log2(max(abs(c["logged"]), 1e-30))) - 7)
            print(f"  selftest {r['label']} L{c['layer']}: logged {c['logged']:.6f} (bf16 ulp {ulp:.1e}); bf16 recipe diff "
                  f"{c['diff_recipe_bf16']:+.2e} = {abs(c['diff_recipe_bf16']) / ulp:.1f} ulp; (b) diff {c['diff']:+.2e}")
            if abs(c["diff_recipe_bf16"]) > max(1e-3, 3 * ulp) or abs(c["diff"]) > 0.01:
                fails.append(f"{r['label']} L{c['layer']}: CPU positive check beyond tolerance at the eval step 19")
        if a19[0]["questions_differing"] != 0:
            fails.append(f"{r['label']}: CPU hits_dim_held differs from the logged string at the eval step 19")
        sa_ = r["ckpts"][0]["sub"]["item"]["strict_A"]
        print(f"  selftest {r['label']}: strict held questions {r['strict']['n_strict_q']}/{r['strict']['n_held_q']}, strict_A "
              f"item sites {sa_['n'][0]}, R2 per layer {[round(x, 3) for x in sa_['r2']]}")
        if sa_["n"][0] >= 3 and not all(np.isfinite(x) for x in sa_["r2"][1:]):
            fails.append(f"{r['label']}: strict_A subset R2 not finite at L>=1")
    # summary mode on relabelled copies: C as all6, D as only-dim, seeds 0 and 1 (C == D weights -> differences ~0)
    fake = []
    for src, arm in ((rc, "all6"), (rd, "only-dim")):
        for sd in (0, 1):
            r = dict(src, arm=arm, seed=sd, label=f"fake_{arm}_s{sd}")
            p = out / f"fake_{arm}_s{sd}.json"; p.write_text(json.dumps(r)); fake.append(p)
    sm_ = summary(fake)
    vals = [x for s in sm_.values() for ps, p in s.items() if ps != "behaviour"
            for blk in (p.values() if ps != "transport" else [c for o in p.values() for c in o.values()])
            for col in blk.values() for x in col if np.isfinite(x)]
    mxv = max(abs(x) for x in vals)
    print(f"  selftest summary: max |all6 - only-dim| over the relabelled C/D copies = {mxv:.2e} over {len(vals)} finite cells "
          f"(C, D = same weights up to resume); behavioural flag: {[sm_[s]['behaviour']['cannot_answer'] for s in sorted(sm_)]}")
    if mxv > 1e-3:
        fails.append(f"summary of C vs D (same weights) not ~0: {mxv:.2e}")
    if not all(sm_[s]["behaviour"]["cannot_answer"] for s in sm_):
        fails.append("summary: a zero behavioural gap (C vs D) was not flagged")
    # synthetic: the pooling false-negative guard, the transport control, NaN cells, frozen decoder (real row layout)
    synth(out, fails)
    # the run modes' wiring on V5 paths (existence only)
    for d in V5_RUNS:
        print(f"  V5 run dir {d}: exists {d.exists()}, wts {[p.name for p in sorted(d.glob('wts_*_step*.pt'))] if d.exists() else []}")
    print(f"SELFTEST {'PASS' if not fails else 'FAIL'}" + "".join(f"\n  - {f}" for f in fails))


def timing(out: Path):
    reg = R / "out" / "pt6v2-all6-lr2e-3-s16" / "pt6v2-all6-lr2e-3"
    jp = reg / "run_whole_helix3_seed16_pt6v2-all6-lr2e-3.json"
    rj = json.loads(jp.read_text())
    rj = dict(rj, args=dict(rj["args"], save_model_at="7323,12000,18309"))   # the V5 runs' extra arg (no effect on prompts)
    t0 = time.time()
    meta = dump(rj, ["init"], out / "dump_timing")
    t_dump = time.time() - t0
    print(f"timing: real config ({meta['n_layers']} blocks, seq {meta['seq']}), random init, registered all6 s16 args: "
          f"dump {meta['dump_wall_s']} s = trainer setup {meta['setup_s']} s + forward {meta['ckpts'][0]['forward_s']} s per checkpoint")
    print("timing: probe args == registered run args; dim eval sets (q, cls, masks, contexts) == registered run's DINFO")
    for p in meta["prompts"]:
        print(f"  {p['set']} row {p['row']}: seq_len {p['seq_len']}, item sites {p['item_sites']}, answer-pred pos "
              f"{p['answer_pred_pos']}: {p['text']!r}")
    st = dict(np.load(out / "dump_timing" / "static.npz")); sd = dict(np.load(out / "dump_timing" / meta["ckpts"][0]["file"]))
    t1 = time.time()
    T = st["target"]; rng = np.random.default_rng(12345)
    tabs = [T] + [T[rng.permutation(len(T))] for _ in range(N_SHUF)]
    for ps, v in (("item", st["item_v"]), ("ans", st["ans_c"])):
        Ks = [centred_gram(tb[v]) for tb in tabs]
        for l_ in range(meta["n_layers"] + 1):
            H = sd[ps][l_].astype(np.float64); Hc = H - H.mean(0); G = Hc @ Hc.T
            _ = [float((G * K).sum()) for K in Ks]
            ridge_cv(H, [tb[v] for tb in tabs], v)
            if ps == "ans":   # (c') E(a), E(b) decoders
                ridge_cv(H, [tb[st["ans_a"]] for tb in tabs], st["ans_a"]); ridge_cv(H, [tb[st["ans_b"]] for tb in tabs], st["ans_b"])
    t_an = time.time() - t1
    per_run = meta["setup_s"] + 3 * meta["ckpts"][0]["forward_s"] + 3 * t_an * 1.3 + 15
    print(f"timing: analysis (b)+(c)+(c')+negatives per checkpoint {t_an:.1f} s (+30% for subsets, item-mean, drift nulls)")
    print(f"timing: estimate per run (setup + 3 x (forward + 1.3 x analysis) + 15 s overhead) = {per_run:.0f} s; 4 runs x 3 "
          f"checkpoints = {4 * per_run / 60:.1f} min on this node ({os.cpu_count()} cpus visible, OMP {os.environ.get('OMP_NUM_THREADS')})")
    print(f"timing: item rows {len(st['item_v'])}, answer rows {len(st['ans_c'])}, wall {time.time() - t0:.0f} s (dump {t_dump:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", nargs="+", default=[])
    ap.add_argument("--v5", action="store_true")
    ap.add_argument("--summary", nargs="+", default=[])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--steps", default=",".join(map(str, V5_STEPS)))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    steps = tuple(int(v) for v in a.steps.split(","))
    if a.selftest or a.timing:
        out = Path(a.out or R / "tests" / ("probe_v5_selftest" if a.selftest else "probe_v5_timing"))
    else:
        out = Path(a.out or R / "out_v5" / "probe_v5")
    out = out.resolve()   # the dump subprocess runs in the trainer directory: relative --out paths must not break it
    out.mkdir(parents=True, exist_ok=True)
    print(f"probe_v5 {time.strftime('%Y-%m-%d %H:%M:%S')} out {out}; python {sys.version.split()[0]}; cpus {os.cpu_count()}", flush=True)
    if a.selftest:
        selftest(out)
    if a.timing:
        timing(out)
    runs = list(a.run) + ([str(d) for d in V5_RUNS] if a.v5 else [])
    res = [analyse(Path(d), steps, out) for d in runs]
    if a.summary or a.v5:
        summary(a.summary or [out / f"{r['label']}.json" for r in res])
    print(f"\nprobe_v5 done; {len(WARN)} warning(s)" + "".join(f"\n  - {w}" for w in WARN))


if __name__ == "__main__":
    main()
