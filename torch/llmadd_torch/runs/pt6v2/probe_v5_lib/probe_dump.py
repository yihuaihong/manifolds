"""pt6v2 Amendment V5 probe (exp-builder 2026-11-20): the state dump, run INSIDE the probe copy of the V5 trainer.

pretrain_range_probe.py (make_probe_copy.py, edit E4) calls run(locals of main()) right after the model, the item-site
hooks (store) and every eval set are built, i.e. with the trainer's own dim eval prompts:
  dom_evalsets["dim_held"|"dim_seen"] = (ids [rows, seq], lens = scored span length, sites_ev [(row, pos, item)], masks,
  "dim", qi = question index per row, n questions)   -- trainer lines 1169-1200 (V5 sha 041b2f4241b9)
For each weight file (or "init" = the freshly built model, timing only) it runs the dim CKA sets exactly as evaluate()
does (lines 1847-1948: bf16 autocast, chunks of 250 rows, logits_to_keep = dom_max_ans + 1, exact match over the span)
and saves, at every layer (0 = embed_tokens output, l = block l output = the trainer's store[l]):
  item states at sites_ev (both input items of every dim_held + dim_seen row = the logged cka_dim's states), and
  answer-prediction states at position seq - len(span) - 1 (the ' is' token whose logits predict the answer's first token).
It also recomputes cka_dim@L for the read layers with the trainer's own cka / centred_gram over the trainer's own store
(the replica of the logged number) and the per-question hit string hits_<set> exactly as logged.
Outputs (directory --probe_npz): static.npz, meta.json, states_<k>.npz per weight file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import numlib as nl
from pretrain_helix import centred_gram, cka


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def show(tok, row, marks, a0, lo):
    """Decoded tokens from index lo, probed positions in ⟦ ⟧, the scored span (answer through the newline) in « »."""
    s = ""
    for j in range(lo, len(row)):
        t = tok.decode([int(row[j])])
        s += ("«" if j == a0 else "") + (f"⟦{t}⟧" if j in marks else t)
    return s + "»"


def run(L, src_sha):
    t0 = time.time()
    args, model, dev, tok = L["args"], L["model"], L["dev"], L["tok"]
    DES, DINFO, DTAB, DTASK, DIMN = L["dom_evalsets"], L["DINFO"], L["DTAB"], L["DTASK"], L["DIMN"]
    store, read_layers, dom_max_ans = L["store"], L["read_layers"], L["dom_max_ans"]
    d = "dim"
    names = [n for n, v in DES.items() if v[4] == d and len(v[2])]   # the sets whose item sites feed cka_dim (trainer order)
    assert names == ["dim_held", "dim_seen"], names
    outd = Path(args.probe_npz); outd.mkdir(parents=True, exist_ok=True)
    nL = len(model.model.layers)
    hs = {}
    model.model.embed_tokens.register_forward_hook(lambda _m, _i, o: hs.__setitem__(0, o))
    for l_, blk in enumerate(model.model.layers, 1):
        blk.register_forward_hook(lambda _m, _i, o, l_=l_: hs.__setitem__(l_, nl._hidden(o)))
    X, Y = DTASK[d]["X"], DTASK[d]["Y"]
    T = np.asarray(DTAB[d], dtype=np.float64)   # the 27 items' exponent vectors (the prior's target table)
    # ---- static: rows, sites, answer positions, asserted against the prompt tokens
    st = {k: [] for k in ("item_set", "item_row", "item_pos", "item_v", "item_q", "item_slot",
                          "ans_set", "ans_row", "ans_pos", "ans_c", "ans_a", "ans_b", "ans_op", "ans_q")}
    prompts = []
    for si, name in enumerate(names):
        ids, lens, sites_ev, mk, _d, qi_, nq_ = DES[name]
        qlist = DINFO[d][f"eval_{name[len(d) + 1:]}"]["q"]
        assert len(qlist) == nq_
        idn = ids.cpu().numpy(); lnn = lens.cpu().numpy(); seq = idn.shape[1]
        for r in range(len(idn)):
            q = int(qlist[qi_[r]]); a, op, b = int(X[q][0]), int(X[q][1]), int(X[q][2]); c = int(Y[q])   # dom_q (trainer 836)
            ln = int(lnn[r]); ap = seq - ln - 1
            assert tok.decode([int(idn[r, ap])]) == " is", (name, r, tok.decode([int(idn[r, ap])]))
            assert tok.decode(idn[r, seq - ln:].tolist()) == f" {DIMN[c]}\n", (name, r)
            sr = sites_ev[sites_ev[:, 0] == r]
            assert [int(v) for v in sr[:, 2]] == [a, b] and (sr[:, 1] < ap).all(), (name, r, sr.tolist(), a, b)
            for k_, (_, p, v) in enumerate(sr.tolist()):
                tt = tok.decode([int(idn[r, p])]).strip()
                assert tt and (DIMN[v].startswith(tt) or DIMN[v].endswith(tt)), (name, r, p, tt, DIMN[v])
                for kk, vv in zip(("item_set", "item_row", "item_pos", "item_v", "item_q", "item_slot"), (si, r, p, v, q, k_)):
                    st[kk].append(vv)
            for kk, vv in zip(("ans_set", "ans_row", "ans_pos", "ans_c", "ans_a", "ans_b", "ans_op", "ans_q"),
                              (si, r, ap, c, a, b, op, q)):
                st[kk].append(vv)
            if (si == 0 and r < 2) or (si == 1 and r == 0):
                marks = {int(p) for p in sr[:, 1]} | {ap}
                lo = max(0, int(sr[:, 1].min()) - 14)
                prompts.append(dict(set=name, row=r, seq_len=seq, item_sites=[int(p) for p in sr[:, 1]], answer_pred_pos=ap,
                                    answer_tokens=list(range(seq - ln, seq)), question=[a, op, b], answer=c,
                                    text=show(tok, idn[r], marks, seq - ln, lo), shown_from=lo))
    st = {k: np.asarray(v, dtype=np.int64) for k, v in st.items()}
    assert np.array_equal(st["item_v"], np.concatenate([DES[n][2][:, 2] for n in names]))
    np.savez(outd / "static.npz", target=T, **st)
    meta = dict(src_sha=src_sha, args=vars(args), names=names, n_layers=nL, read_layers=read_layers, dom_max_ans=int(dom_max_ans),
                seq=int(DES[names[0]][0].shape[1]), dev=dev, torch=torch.__version__, prompts=prompts,
                dinfo={f"eval_{n[len(d) + 1:]}": {k: DINFO[d][f"eval_{n[len(d) + 1:]}"][k] for k in ("questions", "contexts", "rows", "q", "cls", "masks")}
                       for n in names},
                setup_s=None, ckpts=[])
    meta["setup_s"] = round(time.time() - t0, 1)
    wts = [w for w in args.probe_wts.split(",") if w]
    assert wts
    for w in wts:
        t1 = time.time()
        if w != "init":
            sd = torch.load(w, map_location=dev, weights_only=True)
            model.load_state_dict(sd, strict=True)
        model.eval()
        S, R = len(st["item_v"]), len(st["ans_c"])
        D = model.config.hidden_size
        item = np.zeros((nL + 1, S, D), np.float32); ans = np.zeros((nL + 1, R, D), np.float32)
        rec = dict(wts=w, hits={}, acc={}, acc_masks={}, trainer_cka={})
        row_hit = []
        tst = {L_: [] for L_ in read_layers}; tv = []
        oi = oa = 0
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            for name in names:
                ids, lens, sites_ev, mk, _d, qi_, nq_ = DES[name]
                hits = []
                for i in range(0, len(ids), 250):   # evaluate(): the same chunks, so the same bf16 batch composition
                    x, ln = ids[i:i + 250], lens[i:i + 250]
                    hs.clear()
                    lg = model(input_ids=x, logits_to_keep=dom_max_ans + 1).logits
                    pred = lg[:, :-1].argmax(-1)
                    tgt = x[:, -dom_max_ans:]
                    ok = torch.ones(len(x), dtype=torch.bool, device=dev)
                    for k in range(dom_max_ans):
                        need = ln > (dom_max_ans - 1 - k)
                        ok &= ~need | (pred[:, k] == tgt[:, k])
                    hits.append(ok)
                    assert sorted(hs) == list(range(nL + 1)), sorted(hs)
                    m = sites_ev[(sites_ev[:, 0] >= i) & (sites_ev[:, 0] < i + 250)]
                    ri, pi = torch.tensor(m[:, 0] - i, device=dev), torch.tensor(m[:, 1], device=dev)
                    for L_ in read_layers:   # the trainer's own hook output = this probe's hook at block L_
                        assert torch.equal(store[L_], hs[L_]), L_
                        tst[L_].append(store[L_][ri, pi].float())
                    tv.append(m[:, 2])
                    ar = torch.arange(len(x), device=dev); apos = x.shape[1] - ln - 1
                    assert np.array_equal(apos.cpu().numpy(), st["ans_pos"][oa:oa + len(x)])
                    for l_ in range(nL + 1):
                        item[l_, oi:oi + len(m)] = hs[l_][ri, pi].float().cpu().numpy()
                        ans[l_, oa:oa + len(x)] = hs[l_][ar, apos].float().cpu().numpy()
                    oi += len(m); oa += len(x)
                hit = torch.cat(hits).cpu().numpy()
                row_hit.append(hit)
                rec["acc"][name] = float(hit.mean())
                rec["acc_masks"][name] = {k: (float(hit[msk].mean()) if msk.any() else None) for k, msk in mk.items()}
                rec["hits"][name] = "".join(map(str, np.bincount(qi_[hit.astype(bool)], minlength=nq_).tolist()))
        assert oi == S and oa == R
        v = np.concatenate(tv)
        assert np.array_equal(v, st["item_v"])
        for L_ in read_layers:   # the logged number's recipe: trainer cka(H, centred_gram(DTAB[d][v])) on its store,
            # INSIDE the bf16 autocast block as evaluate() runs it (lines 1941-1948: its Gram matmuls run in bf16)
            with torch.autocast(dev, dtype=torch.bfloat16):
                rec["trainer_cka"][f"cka_dim@L{L_}"] = float(cka(torch.cat(tst[L_]), centred_gram(DTAB[d][v], dev)))
            rec["trainer_cka_fp32"] = rec.get("trainer_cka_fp32", {})
            rec["trainer_cka_fp32"][f"cka_dim@L{L_}"] = float(cka(torch.cat(tst[L_]), centred_gram(DTAB[d][v], dev)))
        k_ = len(meta["ckpts"])
        np.savez(outd / f"states_{k_}.npz", item=item, ans=ans, ans_hit=np.concatenate(row_hit).astype(np.int8))
        rec["file"] = f"states_{k_}.npz"; rec["forward_s"] = round(time.time() - t1, 1)
        meta["ckpts"].append(rec)
        print(f"probe_dump: {w}: {rec['forward_s']} s, acc {rec['acc']}, trainer-recipe {rec['trainer_cka']}", flush=True)
    (outd / "meta.json").write_text(json.dumps(meta, indent=1, default=_js))
    print(f"probe_dump: wrote {outd} (setup {meta['setup_s']} s)", flush=True)
