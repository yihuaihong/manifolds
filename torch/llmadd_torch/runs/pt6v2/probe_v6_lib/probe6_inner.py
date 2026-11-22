"""pt6v2 Amendment V6 read-out (exp-builder 2026-11-22; exploratory, no gate): runs INSIDE the probe copy of the V5 trainer.

pretrain_range_probe6.py (make_probe_copy6.py, edit E4) calls run(locals of main()) right after the model, the hooks and
every eval set are built, i.e. with the trainer's own eval prompts: dom_evalsets["<d>_held"|"<d>_seen"] = (ids [rows, seq],
lens = scored span length, sites_ev, masks, d, qi, nq) (V5 trainer sha 041b2f4241b9, lines 1169-1200).
One weight file per process (--probe_wts). Answer position ap = seq - len(span) - 1 (the ' is' token whose logits predict
the answer's first token; dim / box5 / torus spans are ' <item>' + '\\n' = 2 tokens, asserted). Layers: 0 = embed_tokens
output, l = block l output (= the trainer's store[l]); the state "h" fed to block h + 1 (h = n_layers: fed to the final norm).
Forwards are evaluate()'s (bf16 autocast, chunks of 250 rows per eval set, logits_to_keep = dom_max_ans + 1, exact match
over the whole span).
  A1  answer-position ridge decoders (dim 3-D exponents, box5 2-D integers, torus = DTAB['torus'] = the prior's cos/sin
      of both Z6 coordinates), out-of-fold R2 (grouped by answer item); subspace overlap (mean cos^2 of principal angles) of
      decoder column spaces and of top-k PCs of per-answer class means, with random / covariance-matched / shuffled-label
      nulls; cross-task transfer (src decoder on tgt states -> OLS map to tgt coordinates, OOF R2) with nulls.
  A2  mean-ablation of a k-dim subspace U at ap: h <- h - ((h - mu_E) U) U^T (mu_E = the evaluated set's mean state at
      that layer and position), on the state fed to block h + 1 (h in hook_states) or to the final norm, ONLY at ap; the
      later positions see the change through attention. Exact match (the trainer's), first-token and '\\n'-token accuracy
      and the answer margin. Subspaces = top-k PCs of per-answer class means of a source domain's SEEN rows at that layer
      (own = positive control on held rows), n_rand random k-subspaces (negative control), cities / countries
      (specificity). k = 0 must reproduce the clean logits bit for bit.
  A3  logit lens at ap: final norm (fp32) + the tied unembedding on every layer's state; margin = logit(correct answer
      token) - max over the domain's other answer tokens; top-1 among them. Direct logit attribution of attn / MLP
      outputs (post_attention_layernorm / post_feedforward_layernorm outputs = what is added to the residual) of every
      block, through the final norm with its scale frozen at the real final residual.
  A4  weights: CKA(answer-token unembedding rows, lattice coordinates); CKA(per-answer class means of the final-norm
      state at ap, those rows).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import numlib as nl

LAT = ("dim", "box5", "torus")
ANCH = ("cities", "countries")
ALPHAS = 10.0 ** np.linspace(-5, 2, 15)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------ linear algebra (float64)
def folds(groups, k, seed):
    u = np.random.default_rng(seed).permutation(np.unique(groups))
    return [u[i::k] for i in range(k)]


def fit_svd(X, Y, tr):
    mu, ym = X[tr].mean(0), Y[tr].mean(0)
    U, S, Vt = np.linalg.svd(X[tr] - mu, full_matrices=False)
    keep = S > S.max() * 1e-10
    return mu, ym, U[:, keep], S[keep], Vt[keep], U[:, keep].T @ (Y[tr] - ym)


def coef(ft, rel):   # D x C ridge weights, alpha = rel * mean squared singular value (V5's rule)
    _, _, _, S, Vt, UtY = ft
    a = rel * float(np.mean(S ** 2))
    return Vt.T @ ((S / (S ** 2 + a))[:, None] * UtY)


def predict(ft, X, rel):
    return (X - ft[0]) @ coef(ft, rel) + ft[1]


def ridge_cv(X, Ys, groups, nf=9):
    """Grouped (by answer item) nf-fold CV ridge, alpha per target by an inner grouped (nf-1)-fold CV on the training
    items only (V5's procedure, any target width). Ys = list of [n, c] targets. Returns OOF predictions, chosen rel."""
    X = X.astype(np.float64)
    Yall = np.concatenate(Ys, 1); off = np.cumsum([0] + [y.shape[1] for y in Ys])
    oof = np.zeros_like(Yall); chosen = [[] for _ in Ys]
    for f in folds(groups, nf, 0):
        te = np.isin(groups, f); tr = ~te
        if not te.any():
            continue
        err = np.zeros((len(ALPHAS), len(Ys)))
        for g in folds(groups[tr], nf - 1, 1):
            ite = tr & np.isin(groups, g); itr = tr & ~np.isin(groups, g)
            if not ite.any():
                continue
            ft = fit_svd(X, Yall, itr)
            P = (X[ite] - ft[0]) @ ft[4].T
            for ai, rel in enumerate(ALPHAS):
                a = rel * float(np.mean(ft[3] ** 2))
                e = ((P @ ((ft[3] / (ft[3] ** 2 + a))[:, None] * ft[5]) + ft[1] - Yall[ite]) ** 2).sum(0)
                err[ai] += [e[off[c]:off[c + 1]].sum() for c in range(len(Ys))]
        ft = fit_svd(X, Yall, tr)
        best = err.argmin(0)
        for ai in np.unique(best):
            pr = predict(ft, X[te], ALPHAS[ai])
            for c in np.where(best == ai)[0]:
                oof[te, off[c]:off[c + 1]] = pr[:, off[c]:off[c + 1]]
        for c in range(len(Ys)):
            chosen[c].append(float(ALPHAS[best[c]]))
    return [oof[:, off[c]:off[c + 1]] for c in range(len(Ys))], chosen


def r2(y, yh, m=None):   # pooled over coordinates, against the subset's own mean
    if m is not None:
        y, yh = y[m], yh[m]
    if len(y) < 2:
        return float("nan")
    sst = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ((y - yh) ** 2).sum() / sst) if sst > 0 else float("nan")


def nearest_acc(yh, T, c, m=None):
    d2 = ((yh[:, None, :] - T[None]) ** 2).sum(-1)
    ok = d2.argmin(1) == c
    return float(ok[m].mean() if m is not None else ok.mean())


def orth(M):
    if M.shape[1] == 0:
        return M
    Q, R = np.linalg.qr(M)
    return Q[:, np.abs(np.diag(R)) > 1e-10 * max(1e-30, np.abs(np.diag(R)).max())]


def overlap(U1, U2):   # mean cos^2 of the principal angles = |U1^T U2|_F^2 / min rank
    k = min(U1.shape[1], U2.shape[1])
    return float(((U1.T @ U2) ** 2).sum() / k) if k else float("nan")


def class_mean_pcs(H, c, k):
    """Top-k right singular vectors of the centred per-answer class means (classes = answer items present)."""
    cls = np.unique(c)
    M = np.stack([H[c == i].mean(0) for i in cls]).astype(np.float64)
    M -= M.mean(0)
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:k].T, S


def ols_oof(Z, Y, groups, nf=9):   # OLS with intercept, grouped nf-fold CV -> OOF predictions
    Zi = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    out = np.zeros_like(Y, dtype=np.float64)
    for f in folds(groups, nf, 0):
        te = np.isin(groups, f)
        if not te.any():
            continue
        B_, *_ = np.linalg.lstsq(Zi[~te], Y[~te], rcond=None)
        out[te] = Zi[te] @ B_
    return out


def lin_cka(A, B):   # linear CKA, feature form, centred
    A = A - A.mean(0); B = B - B.mean(0)
    num = np.linalg.norm(A.T @ B) ** 2
    den = np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)
    return float(num / den) if den > 0 else float("nan")


def pstat(x):
    x = np.asarray(x, float)
    return {"mean": float(np.mean(x)), "p95": float(np.quantile(x, 0.95)), "sd": float(np.std(x)), "max": float(np.max(x)),
            "n": int(len(x))}


def zsc(v, st):   # z of a value against a null summary (pstat)
    return float((v - st["mean"]) / st["sd"]) if st["sd"] > 0 else float("nan")


def show(tok, row, ap, a0, lo):
    s = ""
    for j in range(lo, len(row)):
        t = tok.decode([int(row[j])])
        s += ("«" if j == a0 else "") + (f"⟦{t}⟧" if j == ap else t)
    return s + "»"


# ------------------------------------------------------------------ main
def run(L, src_sha):
    t_all = time.time()
    args, model, dev, tok = L["args"], L["model"], L["dev"], L["tok"]
    DES, DINFO, DTAB, DTASK = L["dom_evalsets"], L["DINFO"], L["DTAB"], L["DTASK"]
    K = int(L["dom_max_ans"])
    cfg = json.loads(args.probe6_cfg)
    outd = Path(args.probe6_out); outd.mkdir(parents=True, exist_ok=True)
    n_rand, n_shuf, n_null = int(cfg.get("n_rand", 20)), int(cfg.get("n_shuf", 10)), int(cfg.get("n_null", 200))
    n_dose = int(cfg.get("n_dose", 5))   # B2: draws of each dose-matched A2 null
    nL = len(model.model.layers); D = model.config.hidden_size
    hook_states = [int(h) for h in cfg.get("hook_states", [nL - 3, nL - 2, nL - 1, nL])]
    a1_layers = [int(h) for h in cfg.get("a1_layers", list(range(max(1, nL - 4), nL + 1)))]
    abl_sets = list(cfg.get("abl_sets", ["dim_held", "box5_held", "torus_held"]))
    abl_srcs = list(cfg.get("abl_srcs", ["dim", "box5", "torus", "cities", "countries"]))
    timing = {}
    say(f"probe_v6: trainer sha {src_sha}; weights {args.probe_wts}; n_layers {nL}, D {D}, dom_max_ans {K}; cfg {json.dumps(cfg)}")
    say(f"probe_v6: hook states {hook_states} (state h feeds block h+1; h = {nL} feeds the final norm); A1 layers {a1_layers}")

    # ---- the run JSON: args and the eval sets of every domain read here equal the logged run's (crash before compute)
    rj = json.loads(Path(cfg["run_json"]).read_text())
    NOT = {"out", "probe_wts", "probe6_out", "probe6_cfg"}
    bad = {k: (v, getattr(args, k, None)) for k, v in rj["args"].items() if k not in NOT and getattr(args, k, None) != v}
    assert not bad, ("probe args != run args", bad)
    for d in LAT + ANCH:
        for st_ in ("held", "seen"):
            got, lg = DINFO[d][f"eval_{st_}"], rj["domains"][d][f"eval_{st_}"]
            assert "q" in got and "q" in lg, (d, st_)
            for k in sorted(set(got) & set(lg)):
                assert json.loads(json.dumps(got[k], default=_js)) == lg[k], ("probe eval set != logged", d, st_, k)
    say("probe_v6: probe args == run JSON args; eval_held / eval_seen of dim, box5, torus, cities, countries == run JSON")

    # ---- static: rows, answer items, answer position, answer tokens (asserted against the prompt tokens)
    import rainbow6_summary as RS
    sqm, _ = RS.strict_mask(rj)
    sqm = np.asarray(sqm, bool)
    DIMN, T3N, AQ, AIT = L["DIMN"], L.get("T3N", {}), L.get("AQ", {}), L.get("AIT", {})
    enc = L["enc"]
    T = {d: np.asarray(DTAB[d], np.float64) for d in LAT + ANCH}
    # B1 (code review): the torus items' (x, y) in Z6 x Z6 from the prior's table (cos, sin of 2 pi x / 6, of 2 pi y / 6;
    # asserted); characters u . (x, y) up to sign; generating pairs = det unit mod 6 (bijective encodings of the 36 items)
    Tt = T["torus"]
    assert Tt.shape == (36, 4) and np.allclose(Tt[:, 0] ** 2 + Tt[:, 1] ** 2, 1) and np.allclose(Tt[:, 2] ** 2 + Tt[:, 3] ** 2, 1)
    XY = np.stack([np.round(np.arctan2(Tt[:, 1], Tt[:, 0]) * 3 / np.pi) % 6, np.round(np.arctan2(Tt[:, 3], Tt[:, 2]) * 3 / np.pi) % 6], 1).astype(int)
    assert len({tuple(v) for v in XY}) == 36

    def char_feat(u):   # [36, 2] cos / sin of the character u at every item
        ph = 2 * np.pi * (XY @ np.asarray(u)) / 6
        return np.stack([np.cos(ph), np.sin(ph)], 1)
    CHARS = []
    for a_ in range(6):
        for b_ in range(6):
            if (a_, b_) != (0, 0) and ((-a_) % 6, (-b_) % 6) not in CHARS:
                CHARS.append((a_, b_))
    GEN_PAIRS = [(u, v) for i, u in enumerate(CHARS) for v in CHARS[i + 1:] if (u[0] * v[1] - u[1] * v[0]) % 6 in (1, 5)]
    DTAB_PAIR = ((0, 1), (1, 0))
    assert DTAB_PAIR in GEN_PAIRS and np.allclose(np.concatenate([char_feat((1, 0)), char_feat((0, 1))], 1), Tt), "DTAB torus != characters (1,0),(0,1)"
    say(f"probe_v6: torus characters {len(CHARS)} (up to sign), generating pairs {len(GEN_PAIRS)}; DTAB = (1,0),(0,1) asserted")
    sets = {}
    prompts = []
    use_sets = [f"{d}_{s}" for d in LAT for s in ("held", "seen")] + [f"{d}_seen" for d in ANCH]
    for name in use_sets:
        d, st_ = name.split("_")
        ids, lens, sites_ev, mk, d_, qi_, nq_ = DES[name]
        assert d_ == d
        qlist = DINFO[d][f"eval_{st_}"]["q"]
        idn = ids.cpu().numpy(); lnn = lens.cpu().numpy(); seq = idn.shape[1]
        c = np.zeros(len(idn), np.int64); ap = seq - lnn - 1
        for r in range(len(idn)):
            q = qlist[qi_[r]]
            if d == "dim":
                c[r] = int(DTASK[d]["Y"][q]); nm = DIMN[c[r]]
            elif d in LAT:
                c[r] = int(DTASK[d]["Y"][q]); nm = T3N[d][c[r]]
            else:
                Q = AQ[d][q]; c[r] = int(Q["opts"][Q["ans"]]); nm = AIT[d][c[r]]
            span = tok.decode(idn[r, seq - lnn[r]:].tolist())
            assert span == f" {nm}\n", (name, r, span, nm)
            if d in LAT:
                assert lnn[r] == 2 and tok.decode([int(idn[r, ap[r]])]) == " is", (name, r, lnn[r], tok.decode([int(idn[r, ap[r]])]))
        strict = np.zeros(len(idn), bool)
        if name == "dim_held":
            strict = sqm[qi_]
        cand = None
        if d in LAT:   # the domain's answer tokens (one per item), from enc and from the rows
            cand = np.array([enc(f" {DIMN[i] if d == 'dim' else T3N[d][i]}") for i in range(len(T[d]))], dtype=object)
            assert all(len(t) == 1 for t in cand), (d, "multi-token item")
            cand = np.array([t[0] for t in cand], np.int64)
            assert len(set(cand.tolist())) == len(cand)
            assert (idn[np.arange(len(idn)), seq - lnn] == cand[c]).all(), (name, "answer token != enc(item)")
        sets[name] = dict(d=d, ids=ids, lens=lens, ln=lnn, ap=ap, c=c, strict=strict, mk=mk, qi=qi_, nq=nq_, cand=cand, n=len(idn))
        for r in ([0, 1] if st_ == "held" else [0]) if d in LAT else ([0] if d == "cities" else []):
            lo = max(0, int(ap[r]) - 14)
            prompts.append(dict(set=name, row=int(r), seq_len=int(seq), answer_pred_pos=int(ap[r]),
                                answer_tokens=list(range(seq - int(lnn[r]), seq)), answer_item=int(c[r]),
                                text=show(tok, idn[r], int(ap[r]), seq - int(lnn[r]), lo), shown_from=lo))
    say("probe_v6: decoded eval inputs (answer-prediction position in ⟦ ⟧, scored span « »):")
    for p in prompts:
        say(f"  {p['set']} row {p['row']}: seq_len {p['seq_len']}, answer-pred pos {p['answer_pred_pos']}, answer tokens "
            f"{p['answer_tokens']}, answer item {p['answer_item']}; tokens {p['shown_from']}..: {p['text']!r}")
    say("probe_v6: rows " + ", ".join(f"{n} {s['n']}" for n, s in sets.items()) + f"; dim strict rows {int(sets['dim_held']['strict'].sum())}")

    # ---- weights
    w = args.probe_wts
    sd = torch.load(w, map_location=dev, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    WE = model.model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr() == WE.data_ptr(), "embeddings not tied"
    Wnp = WE.detach().float().cpu().numpy().astype(np.float64)
    g_fin = model.model.norm.weight.detach().float().cpu().numpy().astype(np.float64)
    eps_fin = float(model.model.norm.variance_epsilon)

    # ---- hooks: every layer's output, every block's attn / MLP contribution, the layer kwargs (for the partial re-run)
    hs, comp, kw_box = {}, {}, {}
    hooks = [model.model.embed_tokens.register_forward_hook(lambda _m, _i, o: hs.__setitem__(0, o))]
    for l_, blk in enumerate(model.model.layers, 1):
        hooks.append(blk.register_forward_hook(lambda _m, _i, o, l_=l_: hs.__setitem__(l_, nl._hidden(o))))
        hooks.append(blk.post_attention_layernorm.register_forward_hook(lambda _m, _i, o, l_=l_: comp.__setitem__(("attn", l_), o)))
        hooks.append(blk.post_feedforward_layernorm.register_forward_hook(lambda _m, _i, o, l_=l_: comp.__setitem__(("mlp", l_), o)))
    hooks.append(model.model.layers[0].register_forward_pre_hook(
        lambda _m, a, kw: kw_box.__setitem__("kw", dict(kw)), with_kwargs=True))

    def score(lg, x, ln):   # evaluate()'s exact match + the per-token parts and the answer margin
        pred = lg[:, :-1].argmax(-1)
        tgt = x[:, -K:]
        ok = torch.ones(len(x), dtype=torch.bool, device=dev)
        for k in range(K):
            need = ln > (K - 1 - k)
            ok &= ~need | (pred[:, k] == tgt[:, k])
        return ok, pred, tgt

    # ---- clean forwards
    t0 = time.time()
    clean = {}
    lens_eq = {}
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for name, S_ in sets.items():
            ids, lens = S_["ids"], S_["lens"]
            st = np.zeros((nL + 1, S_["n"], D), np.float32)
            cp = np.zeros((2, nL + 1, S_["n"], D), np.float32)   # [attn|mlp, block (1..nL), row]
            hit, t1ok, t2ok, lgc, chunks = [], [], [], [], []
            for i in range(0, len(ids), 250):
                x, ln = ids[i:i + 250], lens[i:i + 250]
                hs.clear(); comp.clear(); kw_box.clear()
                lg = model(input_ids=x, logits_to_keep=K + 1).logits
                assert sorted(hs) == list(range(nL + 1))
                ok, pred, tgt = score(lg, x, ln)
                ar = torch.arange(len(x), device=dev); apos = x.shape[1] - ln - 1
                for l_ in range(nL + 1):
                    st[l_, i:i + len(x)] = hs[l_][ar, apos].float().cpu().numpy()
                for l_ in range(1, nL + 1):
                    cp[0, l_, i:i + len(x)] = comp[("attn", l_)][ar, apos].float().cpu().numpy()
                    cp[1, l_, i:i + len(x)] = comp[("mlp", l_)][ar, apos].float().cpu().numpy()
                # logit lens at the last layer, the model's own way (norm on the full sequence, slice, lm_head): == logits
                lz = model.lm_head(model.model.norm(hs[nL])[:, -(K + 1):, :])
                lens_eq[name] = lens_eq.get(name, True) and bool(torch.equal(lz, lg))
                hit.append(ok.cpu().numpy())
                if S_["cand"] is not None:
                    j1 = (K - ln).cpu().numpy()
                    p_np, t_np = pred.cpu().numpy(), tgt.cpu().numpy()
                    t1ok.append(p_np[np.arange(len(x)), j1] == t_np[np.arange(len(x)), j1])
                    t2ok.append(p_np[np.arange(len(x)), j1 + 1] == t_np[np.arange(len(x)), j1 + 1])
                    cand_t = torch.tensor(S_["cand"], device=dev)
                    lgc.append(lg[ar, torch.tensor(j1, device=dev)][:, cand_t].float().cpu().numpy())
                if name in abl_sets:   # the clean per-chunk tensors of the partial re-run (references, not copies)
                    kw = dict(kw_box["kw"]); kw["past_key_values"] = None
                    chunks.append(dict(lo=i, x=x, ln=ln, kw=kw, H={h: hs[h] for h in hook_states}, lg=lg.clone()))
            S_["states"] = st; S_["comp"] = cp
            S_["hit"] = np.concatenate(hit)
            if t1ok:
                S_["tok1"], S_["tok2"], S_["lgc"] = np.concatenate(t1ok), np.concatenate(t2ok), np.concatenate(lgc)
            S_["chunks"] = chunks
            S_["hits_str"] = "".join(map(str, np.bincount(S_["qi"][S_["hit"].astype(bool)], minlength=S_["nq"]).tolist()))
    timing["clean_forward_s"] = round(time.time() - t0, 1)
    acc = {n: float(S_["hit"].mean()) for n, S_ in sets.items() if S_["d"] in LAT}
    strict_acc = float(sets["dim_held"]["hit"][sets["dim_held"]["strict"]].mean())
    say(f"probe_v6: clean forward {timing['clean_forward_s']} s; exact-match acc {json.dumps({k: round(v, 4) for k, v in acc.items()})}; "
        f"dim strict {strict_acc:.4f}; logit lens @L{nL} (bf16, the model's own path) == logits: {lens_eq}")
    checks = dict(lens_last_equals_logits=lens_eq)

    # ---- check: unablated hits vs the logged eval at this step (cross-device: flag only) and vs the V5 CPU probe dump
    step = int(cfg.get("step", -1))
    cur = {e["step"]: e for e in rj["curve"]}
    hits_cmp = {}
    for name, S_ in sets.items():
        if S_["d"] not in LAT:
            continue
        rec = {}
        if step in cur and f"hits_{name}" in cur[step]:
            lg_ = cur[step][f"hits_{name}"]
            rec["logged_step"] = step
            rec["logged_acc"] = cur[step].get(name)
            rec["n_q_differ_logged"] = int(sum(a != b for a, b in zip(lg_, S_["hits_str"])))
        else:
            nb = sorted(cur, key=lambda s: abs(s - step))[:2]
            rec["no_eval_at_step"] = True
            rec["nearest"] = {int(s): dict(acc=cur[s].get(name), n_q_differ=int(sum(a != b for a, b in zip(cur[s].get(f"hits_{name}", ""), S_["hits_str"]))))
                              for s in nb}
        hits_cmp[name] = rec
    if cfg.get("v5_meta"):
        m5 = json.loads(Path(cfg["v5_meta"]).read_text())
        for ck in m5["ckpts"]:
            if Path(ck["wts"]).name == Path(w).name:
                for name in ("dim_held", "dim_seen"):
                    hits_cmp[name]["n_q_differ_v5_cpu_probe"] = int(sum(a != b for a, b in zip(ck["hits"][name], sets[name]["hits_str"])))
                    hits_cmp[name]["v5_cpu_probe_acc"] = ck["acc"][name]
    if "dim_held" in hits_cmp and "logged_step" in hits_cmp["dim_held"]:
        hits_cmp["dim_held"]["logged_strict"] = float(dict(RS.strict_series(rj))[step])
        hits_cmp["dim_held"]["probe_strict"] = strict_acc
    checks["hits_vs_logged"] = hits_cmp
    say("probe_v6: unablated hits vs logged / V5 CPU probe: " + json.dumps(hits_cmp))

    res = dict(src_sha=src_sha, wts=w, step=step, cfg=cfg, n_layers=nL, D=D, dom_max_ans=K, hook_states=hook_states,
               a1_layers=a1_layers, prompts=prompts, rows={n: S_["n"] for n, S_ in sets.items()},
               n_strict=int(sets["dim_held"]["strict"].sum()), acc=acc, dim_strict_acc=strict_acc,
               hits={n: S_["hits_str"] for n, S_ in sets.items() if S_["d"] in LAT}, checks=checks)
    for n in ("box5_held", "torus_held", "dim_held"):
        res.setdefault("acc_masks", {})[n] = {k: (float(sets[n]["hit"][m].mean()) if m.any() else None) for k, m in sets[n]["mk"].items()}

    def H_of(d, st_, l_):
        return sets[f"{d}_{st_}"]["states"][l_].astype(np.float64)

    def rows_of(d, l_):   # held + seen pooled (held first)
        Hh, Hs = H_of(d, "held", l_), H_of(d, "seen", l_)
        c = np.concatenate([sets[f"{d}_held"]["c"], sets[f"{d}_seen"]["c"]])
        strict = np.concatenate([sets[f"{d}_held"]["strict"], np.zeros(sets[f"{d}_seen"]["n"], bool)])
        return np.concatenate([Hh, Hs]), c, strict

    # ================================================================== A1
    t0 = time.time()
    rng = np.random.default_rng(6060)
    A1 = {"r2": {}, "overlap": {}, "transfer": {}}
    shuf = {d: [rng.permutation(len(T[d])) for _ in range(n_shuf)] for d in LAT + ANCH}
    alpha_med = {}
    # ---- B1 (code review 2026-11-22): any generating pair of Z6 x Z6 characters encodes the torus answer; the prior's DTAB
    # is the pair (1,0), (0,1). Pick the pair per checkpoint by grouped-ridge OOF R2 on the SEEN rows only (held stays
    # clean), mean over torus_sel_layers; A1 R2 / nearest / dec overlap / transfer and A4 coordinate CKAs use the chosen
    # pair ('torus'), and the DTAB pair is reported beside it ('torus_dtab'). pc overlap and A2 do not use coordinates.
    sel_layers = [int(h) for h in cfg.get("torus_sel_layers", [nL - 2, nL - 1, nL])]
    cs_ = sets["torus_seen"]["c"]
    sc_ = np.zeros(len(GEN_PAIRS)); per_layer = {}
    i_dt = GEN_PAIRS.index(DTAB_PAIR)
    for l_ in sel_layers:
        oof, _ = ridge_cv(H_of("torus", "seen", l_), [char_feat(u)[cs_] for u in CHARS], cs_)
        P_ = dict(zip(CHARS, oof))
        r2p = np.array([r2(np.concatenate([char_feat(u)[cs_], char_feat(v)[cs_]], 1), np.concatenate([P_[u], P_[v]], 1))
                        for u, v in GEN_PAIRS])
        sc_ += r2p / len(sel_layers)
        per_layer[l_] = dict(best=[list(u) for u in GEN_PAIRS[int(r2p.argmax())]], best_seen_r2=float(r2p.max()),
                             dtab_seen_r2=float(r2p[i_dt]))
    ib = int(sc_.argmax()); CH_SEL = GEN_PAIRS[ib]
    TV = dict(T); TV["torus"] = np.concatenate([char_feat(CH_SEL[0]), char_feat(CH_SEL[1])], 1); TV["torus_dtab"] = T["torus"]
    top = np.argsort(-sc_)[:5]
    A1["torus_pair"] = dict(chosen=[list(u) for u in CH_SEL], mean_seen_r2=float(sc_[ib]), dtab_pair=[list(u) for u in DTAB_PAIR],
                            dtab_mean_seen_r2=float(sc_[i_dt]), sel_layers=sel_layers, per_layer=per_layer, n_pairs=len(GEN_PAIRS),
                            top5=[([list(u) for u in GEN_PAIRS[i]], float(sc_[i])) for i in top])
    say(f"probe_v6: torus pair (B1) chosen {A1['torus_pair']['chosen']} mean seen R2 {sc_[ib]:.3f} over L{sel_layers}; "
        f"DTAB {A1['torus_pair']['dtab_pair']} {sc_[i_dt]:.3f}; per layer {json.dumps(per_layer)}")
    BASE = lambda v: "torus" if v == "torus_dtab" else v   # noqa: E731  (variant -> its domain's rows)

    for l_ in a1_layers:
        for d in LAT + ("torus_dtab",):
            X, c, strict = rows_of(BASE(d), l_)
            Ys = [TV[d][c]] + [TV[d][p][c] for p in shuf[BASE(d)]]
            oof, ch = ridge_cv(X, Ys, c)
            alpha_med[d, l_] = float(np.median(ch[0]))
            rs_ = [r2(y, o) for y, o in zip(Ys[1:], oof[1:])]
            rec = dict(r2=r2(Ys[0], oof[0]), nearest=nearest_acc(oof[0], TV[d], c), r2_shuf=pstat(rs_),
                       held_r2=r2(Ys[0], oof[0], np.arange(len(c)) < sets[f"{BASE(d)}_held"]["n"]),
                       alpha_rel_median=alpha_med[d, l_])
            rec["r2_z_shuf"] = zsc(rec["r2"], rec["r2_shuf"])
            if d == "dim":
                rec["strict_r2"] = r2(Ys[0], oof[0], strict)
                rec["strict_nearest"] = nearest_acc(oof[0], TV[d], c, strict)
            A1["r2"][f"{d}@L{l_}"] = rec
    timing["a1_r2_s"] = round(time.time() - t0, 1)
    say("probe_v6: A1 R2 (all | held | shuffled-label max, z): " + "; ".join(
        f"{k_} {v_['r2']:.3f} | {v_['held_r2']:.3f} | {v_['r2_shuf']['max']:.3f}, {v_['r2_z_shuf']:.1f}" for k_, v_ in A1["r2"].items()))

    def dec_space(d, l_, st_="seen", targ=None):   # ridge decoder column space, fit on d's rows of set st_
        H = H_of(BASE(d), st_, l_); c = sets[f"{BASE(d)}_{st_}"]["c"]
        Y = (TV[d] if targ is None else targ)[c]
        ft = fit_svd(H, Y, np.ones(len(H), bool))
        return orth(coef(ft, alpha_med.get((d, l_), 1e-2)))

    t0 = time.time()
    pairs = [("dim", "box5"), ("dim", "torus"), ("box5", "torus"), ("dim", "cities"), ("dim", "countries")]
    for l_ in a1_layers:
        for a, b in pairs:
            if (b, l_) not in alpha_med:   # anchors: the alpha of an all-rows grouped CV on their seen rows
                Hb, cb = H_of(b, "seen", l_), sets[f"{b}_seen"]["c"]
                _, chb = ridge_cv(Hb, [T[b][cb]], cb)
                alpha_med[b, l_] = float(np.median(chb[0]))
            rec = {}
            # primary (code review): top-k PCs of per-answer class means (coordinate-free), row-shuffle null as z / max
            Xa = H_of(a, "seen", l_) - H_of(a, "seen", l_).mean(0); Xb = H_of(b, "seen", l_) - H_of(b, "seen", l_).mean(0)
            for k in (3, 6):
                Pa, _ = class_mean_pcs(H_of(a, "seen", l_), sets[f"{a}_seen"]["c"], k)
                Pb, _ = class_mean_pcs(H_of(b, "seen", l_), sets[f"{b}_seen"]["c"], k)
                rec[f"pc{k}"] = overlap(Pa, Pb)
                # null: class means of a ROW-level shuffle of the answer labels (class sizes kept), both domains
                nul = []
                for _ in range(n_shuf):
                    ca = rng.permutation(sets[f"{a}_seen"]["c"]); cb_ = rng.permutation(sets[f"{b}_seen"]["c"])
                    nul.append(overlap(class_mean_pcs(H_of(a, "seen", l_), ca, k)[0], class_mean_pcs(H_of(b, "seen", l_), cb_, k)[0]))
                rec[f"pc{k}_null_rowshuffle"] = pstat(nul)
                rec[f"pc{k}_z_rowshuffle"] = zsc(rec[f"pc{k}"], rec[f"pc{k}_null_rowshuffle"])
                rec[f"pc{k}_null_covmatched"] = pstat([overlap(orth(Xa.T @ rng.standard_normal((len(Xa), k))),
                                                               orth(Xb.T @ rng.standard_normal((len(Xb), k)))) for _ in range(n_null)])
            # secondary: ridge decoder column spaces (read directions; can be low even when write directions are shared)
            Ua, Ub = dec_space(a, l_), dec_space(b, l_)
            ka, kb = Ua.shape[1], Ub.shape[1]
            rec |= dict(dec=overlap(Ua, Ub), ranks=[ka, kb])
            rec["dec_null_random"] = pstat([overlap(orth(rng.standard_normal((D, ka))), orth(rng.standard_normal((D, kb))))
                                            for _ in range(n_null)])
            rec["dec_null_covmatched"] = pstat([overlap(orth(Xa.T @ rng.standard_normal((len(Xa), ka))),
                                                        orth(Xb.T @ rng.standard_normal((len(Xb), kb)))) for _ in range(n_null)])
            sh = [overlap(Ua, dec_space(b, l_, targ=TV[b][p])) for p in shuf[b]] + \
                 [overlap(dec_space(a, l_, targ=TV[a][p]), Ub) for p in shuf[a]]
            rec["dec_null_shuffled_labels"] = pstat(sh)
            rec["dec_z_shuffled"] = zsc(rec["dec"], rec["dec_null_shuffled_labels"])
            if "torus" in (a, b):   # the prior's DTAB pair beside the chosen one
                rec["dec_dtab"] = overlap(*(dec_space("torus_dtab" if x == "torus" else x, l_) for x in (a, b)))
            A1["overlap"][f"{a}-{b}@L{l_}"] = rec
    timing["a1_overlap_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    for l_ in a1_layers:
        for src, tgt in [("box5", "dim"), ("dim", "box5"), ("torus", "dim"), ("dim", "torus"), ("box5", "torus"), ("torus", "box5"),
                         ("torus_dtab", "dim"), ("dim", "torus_dtab"), ("box5", "torus_dtab"), ("torus_dtab", "box5")]:
            Xs, cs, _ = rows_of(BASE(src), l_); Xt, ct, strict_t = rows_of(BASE(tgt), l_)
            Yt = TV[tgt][ct]

            def transfer(targ_src):
                ft = fit_svd(Xs, targ_src[cs], np.ones(len(Xs), bool))
                Z = predict(ft, Xt, alpha_med[src, l_])
                o = ols_oof(Z, Yt, ct)
                return r2(Yt, o), (r2(Yt, o, strict_t) if tgt == "dim" else float("nan"))

            tr_, trs = transfer(TV[src])
            nul = [transfer(TV[src][p]) for p in shuf[BASE(src)]]
            rp = []
            for _ in range(n_shuf):
                Z = Xt @ rng.standard_normal((D, TV[src].shape[1]))
                o = ols_oof(Z, Yt, ct); rp.append(r2(Yt, o))
            ns = pstat([a for a, _ in nul])
            A1["transfer"][f"{src}->{tgt}@L{l_}"] = dict(
                r2=tr_, strict_r2=trs, null_shuffled_src=ns, z_shuffled_src=zsc(tr_, ns),
                null_shuffled_src_strict=pstat([b for _, b in nul]) if tgt == "dim" else None,
                null_random_proj=pstat(rp), own_decoder_r2=A1["r2"][f"{tgt}@L{l_}"]["r2"], bottleneck=int(TV[src].shape[1]))
    timing["a1_transfer_s"] = round(time.time() - t0, 1)
    res["A1"] = A1

    # ================================================================== A3 (logit lens + DLA; fp32 from the saved states)
    t0 = time.time()
    A3 = {}
    for name in ("dim_held", "dim_seen", "box5_held", "torus_held"):
        S_ = sets[name]; Wc = Wnp[S_["cand"]]              # [n_items, D]
        ar = np.arange(S_["n"]); c = S_["c"]
        miss = ~S_["hit"].astype(bool)
        subs = {"all": np.ones(S_["n"], bool), "miss": miss}
        if name == "dim_held":
            subs.update(strict=S_["strict"], strict_miss=S_["strict"] & miss)
        rec = {"n": {k: int(v.sum()) for k, v in subs.items()}, "layers": {}}
        for l_ in range(nL + 1):
            h = S_["states"][l_].astype(np.float64)
            r_ = np.sqrt((h ** 2).mean(1, keepdims=True) + eps_fin)
            lz = (h / r_ * g_fin) @ Wc.T
            oth = lz.copy(); oth[ar, c] = -np.inf
            mg = lz[ar, c] - oth.max(1); top = lz.argmax(1) == c
            rec["layers"][l_] = {k: dict(margin=float(mg[m].mean()) if m.any() else None, top1=float(top[m].mean()) if m.any() else None)
                                 for k, m in subs.items()}
            if l_ == nL:
                rec["lens_fp32_vs_model_bf16_maxabs"] = float(np.abs(lz - S_["lgc"]).max())
                rec["lens_fp32_argmax_eq_model"] = float((lz.argmax(1) == S_["lgc"].argmax(1)).mean())
                tstar = oth.argmax(1)
        # DLA through the final norm, scale frozen at the real final residual
        hN = S_["states"][nL].astype(np.float64)
        r_ = np.sqrt((hN ** 2).mean(1, keepdims=True) + eps_fin)
        dW = Wnp[S_["cand"][c]] - Wnp[S_["cand"][tstar]]       # [n, D] correct - best other (fp32 final lens)
        parts = {"emb": S_["states"][0].astype(np.float64)}
        for l_ in range(1, nL + 1):
            parts[f"attn{l_}"] = S_["comp"][0, l_].astype(np.float64)
            parts[f"mlp{l_}"] = S_["comp"][1, l_].astype(np.float64)
        resid = sum(parts.values())
        rec["dla_sum_vs_final_resid_maxabs"] = float(np.abs(resid - hN).max())
        dla = {k: ((v / r_ * g_fin) * dW).sum(1) for k, v in parts.items()}
        tot = sum(dla.values())
        h_ = hN / r_ * g_fin
        rec["dla_total_vs_lens_margin_maxabs"] = float(np.abs(tot - (h_ * dW).sum(1)).max())
        last2 = [f"attn{nL - 1}", f"mlp{nL - 1}", f"attn{nL}", f"mlp{nL}"]
        rec["dla"] = {k: {p: float(dla[p][m].mean()) for p in last2} | {"rest": float(sum(dla[p] for p in dla if p not in last2)[m].mean()),
                                                                        "total": float(tot[m].mean())}
                      for k, m in subs.items() if m.any()}
        rec["dla_all_blocks"] = {p: float(dla[p][subs["strict" if name == "dim_held" else "all"]].mean()) for p in dla}
        A3[name] = rec
    timing["a3_s"] = round(time.time() - t0, 1)
    res["A3"] = A3

    # ================================================================== A4 (weights)
    t0 = time.time()
    A4 = {}
    for d in LAT:
        Wd = Wnp[sets[f"{d}_held"]["cand"]]
        rec = dict(cka_unembed_vs_coords=lin_cka(Wd, TV[d]),
                   null_perm=pstat([lin_cka(Wd, TV[d][rng.permutation(len(TV[d]))]) for _ in range(n_null)]))
        Hh, c, _ = rows_of(d, nL)
        r_ = np.sqrt((Hh ** 2).mean(1, keepdims=True) + eps_fin)
        Hn = Hh / r_ * g_fin
        cls = np.unique(c)
        M = np.stack([Hn[c == i].mean(0) for i in cls])
        rec["n_classes"] = int(len(cls))
        rec["cka_classmean_finalnorm_vs_unembed"] = lin_cka(M, Wd[cls])
        rec["null_perm_classmean"] = pstat([lin_cka(M, Wd[cls][rng.permutation(len(cls))]) for _ in range(n_null)])
        rec["cka_classmean_finalnorm_vs_coords"] = lin_cka(M, TV[d][cls])
        rec["null_perm_classmean_coords"] = pstat([lin_cka(M, TV[d][cls][rng.permutation(len(cls))]) for _ in range(n_null)])
        if d == "torus":   # B1: the prior's DTAB pair beside the chosen pair
            rec["cka_unembed_vs_coords_dtab"] = lin_cka(Wd, T[d])
            rec["cka_classmean_finalnorm_vs_coords_dtab"] = lin_cka(M, T[d][cls])
        A4[d] = rec
    timing["a4_s"] = round(time.time() - t0, 1)
    res["A4"] = A4

    # ================================================================== A2 (ablation at ap)
    t0 = time.time()
    rngr = np.random.default_rng(7070)
    kmax = max(2 * T[sets[n]["d"]].shape[1] for n in abl_sets)
    RND = [orth(rngr.standard_normal((D, kmax))) for _ in range(n_rand)]
    mode = {}

    def abl_logits(chunk, h, U, mu, how):
        x, ln, H = chunk["x"], chunk["ln"], chunk["H"][h]
        ar = torch.arange(len(x), device=dev); apos = x.shape[1] - ln - 1

        def edit(Hin):   # the projection in fp32 (outside autocast); only the answer-prediction position changes
            Hm = Hin.clone()
            with torch.autocast(dev, enabled=False):
                v = Hm[ar, apos].float()
                Hm[ar, apos] = (v - ((v - mu) @ U) @ U.T).to(Hm.dtype)
            return Hm
        if how == "partial":
            z = edit(H)
            for blk in model.model.layers[h:]:
                z = nl._hidden(blk(z, **chunk["kw"]))
            return model.lm_head(model.model.norm(z)[:, -(K + 1):, :])
        mod = model.model.layers[h] if h < nL else model.model.norm

        def pre(_m, a, kw):
            if a:
                return (edit(a[0]),) + tuple(a[1:]), kw
            kw = dict(kw); kw["hidden_states"] = edit(kw["hidden_states"]); return a, kw
        hd = mod.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            return model(input_ids=x, logits_to_keep=K + 1).logits
        finally:
            hd.remove()

    def evaluate_cond(name, h, U, mu, how, keep_rows=False, check_equal=False):
        S_ = sets[name]
        hit, t1, t2, mg, eq = [], [], [], [], True
        cand_t = torch.tensor(S_["cand"], device=dev)
        for ch in S_["chunks"]:
            lg = abl_logits(ch, h, U, mu, how)
            if check_equal:
                eq &= bool(torch.equal(lg, ch["lg"]))
            ok, pred, tgt = score(lg, ch["x"], ch["ln"])
            j1 = (K - ch["ln"]).cpu().numpy(); n_ = len(j1); p_np, t_np = pred.cpu().numpy(), tgt.cpu().numpy()
            hit.append(ok.cpu().numpy()); t1.append(p_np[np.arange(n_), j1] == t_np[np.arange(n_), j1])
            t2.append(p_np[np.arange(n_), j1 + 1] == t_np[np.arange(n_), j1 + 1])
            lc = lg[torch.arange(n_, device=dev), torch.tensor(j1, device=dev)][:, cand_t].float().cpu().numpy()
            c = S_["c"][ch["lo"]:ch["lo"] + n_]
            o = lc.copy(); o[np.arange(n_), c] = -np.inf
            mg.append(lc[np.arange(n_), c] - o.max(1))
        hit, t1, t2, mg = map(np.concatenate, (hit, t1, t2, mg))
        rec = dict(acc=float(hit.mean()), tok1=float(t1.mean()), tok2=float(t2.mean()), margin=float(mg.mean()))
        if name == "dim_held":
            s = S_["strict"]
            rec.update(strict=float(hit[s].mean()), strict_margin=float(mg[s].mean()), strict_tok1=float(t1[s].mean()))
        for k_, m_ in S_["mk"].items():
            if k_ in ("nonid", "no_negtwin_nonid"):
                rec[f"acc_{k_}"] = float(hit[m_].mean()) if m_.any() else None
        if keep_rows:
            rec["hits"] = "".join("1" if b else "0" for b in hit)
        return rec, eq

    A2 = {}
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for name in abl_sets:
            S_ = sets[name]; dE = S_["d"]; cE = T[dE].shape[1]; ks = [cE, 2 * cE]
            clean_rec = dict(acc=float(S_["hit"].mean()), tok1=float(S_["tok1"].mean()), tok2=float(S_["tok2"].mean()))
            if name == "dim_held":
                clean_rec["strict"] = strict_acc
            A2[name] = {"clean": clean_rec, "k": ks, "conds": {}}
            for h in hook_states:
                mu = torch.tensor(S_["states"][h].mean(0), device=dev)
                U0 = torch.zeros((D, 0), device=dev)
                # k = 0: the edit with an empty subspace must reproduce the clean logits bit for bit (partial path), else full
                t_ = time.time()
                r0, eq = evaluate_cond(name, h, U0, mu, "partial", check_equal=True)
                how = "partial"
                if not eq:
                    r0, eq = evaluate_cond(name, h, U0, mu, "full", check_equal=True)
                    how = "full"
                assert eq, (name, h, "k=0 edit does not reproduce the clean logits in either path")
                assert r0["acc"] == clean_rec["acc"], (name, h, r0["acc"], clean_rec["acc"])
                mode[f"{name}@h{h}"] = dict(path=how, k0_s=round(time.time() - t_, 2))
                A2[name]["conds"][f"h{h}"] = {"k0": r0}
                # B2 (code review 2026-11-22): dose = the fraction of the evaluated rows' centred ap-state variance (layer h)
                # that a subspace removes (= what the mean ablation takes out); dose-matched nulls (a) random subspaces
                # drawn from the evaluated set's own state covariance, (b) class-mean PCs of each OTHER lattice source with
                # row-shuffled answer labels; n_dose draws each. Matched-dose read: drop = clean - ablated, drop per unit
                # dose, and the excess drop over a line (drop = a + b dose) fit through all null draws (isotropic random,
                # cov-matched, shuffled-label).
                Hc = S_["states"][h].astype(np.float64); Hc = Hc - Hc.mean(0); vtot = float((Hc ** 2).sum())
                dose = lambda U_: float(((Hc @ U_) ** 2).sum() / vtot)   # noqa: E731
                mets = ["acc", "strict"] if name == "dim_held" else ["acc"]
                c0 = {m_: (strict_acc if m_ == "strict" else clean_rec["acc"]) for m_ in mets}

                def run_U(P_):
                    return evaluate_cond(name, h, torch.tensor(P_, dtype=torch.float32, device=dev), mu, how)[0] | {"dose": dose(P_)}

                def summ(rr):
                    return ({m_: pstat([r[m_] for r in rr]) | {"min": float(min(r[m_] for r in rr))}
                             for m_ in rr[0] if isinstance(rr[0][m_], float)} | {"draws": [{m_: r[m_] for m_ in mets + ["dose"]} for r in rr]}) if rr else {}
                for k in ks:
                    cr = {}
                    for src in abl_srcs:
                        P_, _ = class_mean_pcs(H_of(src, "seen", h), sets[f"{src}_seen"]["c"], k)
                        U = torch.tensor(P_, dtype=torch.float32, device=dev)
                        cr[src], _ = evaluate_cond(name, h, U, mu, how, keep_rows=True)
                        cr[src]["overlap_with_own_pcs"] = overlap(P_, class_mean_pcs(H_of(dE, "seen", h), sets[f"{dE}_seen"]["c"], k)[0])
                        cr[src]["dose"] = dose(P_)
                    nulls = {"random": [run_U(Q[:, :k]) for Q in RND],
                             "null_cov": [run_U(orth(Hc.T @ rngr.standard_normal((len(Hc), k)))) for _ in range(n_dose)]}
                    for src in [s for s in LAT if s != dE]:
                        Hs_, cs__ = H_of(src, "seen", h), sets[f"{src}_seen"]["c"]
                        nulls[f"null_shuf_{src}"] = [run_U(class_mean_pcs(Hs_, rngr.permutation(cs__), k)[0]) for _ in range(n_dose)]
                    for nn, rr in nulls.items():
                        cr[nn] = summ(rr)
                    allnull = [r for rr in nulls.values() for r in rr]
                    dd = np.array([r["dose"] for r in allnull])
                    cmp_ = {"null_dose_range": [float(dd.min()), float(dd.max())], "n_null_draws": len(allnull)}
                    for m_ in mets:
                        yd = np.array([c0[m_] - r[m_] for r in allnull])
                        A_ = np.stack([np.ones_like(dd), dd], 1)
                        cf, *_ = np.linalg.lstsq(A_, yd, rcond=None)
                        sd_ = float(np.std(yd - A_ @ cf, ddof=2)) if len(yd) > 2 else float("nan")
                        cmp_[m_] = {"null_fit": [float(cf[0]), float(cf[1])], "null_fit_resid_sd": sd_,
                                    "null_drop_per_dose": pstat(yd / np.maximum(dd, 1e-9))}
                        for src in abl_srcs:
                            dr = c0[m_] - cr[src][m_]; ds = cr[src]["dose"]; ex = dr - float(cf[0] + cf[1] * ds)
                            cmp_[m_][src] = dict(drop=dr, dose=ds, drop_per_dose=dr / max(ds, 1e-9), excess_vs_null_fit=ex,
                                                 excess_z=ex / sd_ if sd_ and sd_ == sd_ else float("nan"),
                                                 dose_in_null_range=bool(dd.min() <= ds <= dd.max()))
                    cr["dose_cmp"] = cmp_
                    A2[name]["conds"][f"h{h}"][f"k{k}"] = cr
                k_ = ks[0]; cr = A2[name]["conds"][f"h{h}"][f"k{k_}"]
                say(f"  A2 {name} h{h}: path {how}, k=0 reproduces clean (acc {r0['acc']:.4f}); done k {ks} in {time.time() - t_:.1f} s")
                for m_ in mets:
                    say(f"    k{k_} {m_} clean {c0[m_]:.3f} | " + " ".join(f"{s} {cr[s][m_]:.3f}@{cr[s]['dose']:.3f}" for s in abl_srcs) +
                        " | " + " ".join(f"{nn} {cr[nn][m_]['mean']:.3f}@{cr[nn]['dose']['mean']:.3f}" for nn in cr if nn == "random" or nn.startswith("null_")) +
                        " | excess vs null fit " + " ".join(f"{s} {cr['dose_cmp'][m_][s]['excess_vs_null_fit']:+.3f}" for s in abl_srcs))
    timing["a2_s"] = round(time.time() - t0, 1)
    timing["a2_paths"] = mode
    res["A2"] = A2
    for hd in hooks:
        hd.remove()

    # ---- states for later re-reads (answer position, all layers; lattice domains)
    np.savez_compressed(outd / "states_ans.npz", **{f"{n}_states": S_["states"].astype(np.float16) for n, S_ in sets.items()},
                        **{f"{n}_c": S_["c"] for n, S_ in sets.items()}, **{f"{n}_hit": S_["hit"].astype(np.int8) for n, S_ in sets.items()},
                        dim_held_strict=sets["dim_held"]["strict"])
    timing["total_inner_s"] = round(time.time() - t_all, 1)
    res["timing"] = timing
    (outd / "result.json").write_text(json.dumps(res, indent=1, default=_js))
    (outd / "inner_log.txt").write_text("\n".join(LOG) + "\n")
    say(f"probe_v6: wrote {outd / 'result.json'}; timing {json.dumps(timing)}")
