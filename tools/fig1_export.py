"""Figure 1 representation export (2026-10-22): the states of the numbers 0..999 in a pretrain_range.py run, at the model
states its --save_at flag saved (or a --save_model final state), with linear CKA to the helix3 target.

Sites (both single-token numbers, whole tokenisation; the state of the number token itself):
  qx   the prompt "Q: x" alone at position 0 (no text before it): the trainer's --site prefix prompt (PRE = enc("Q: x"),
       F69 / F71's prefix site), its last token. Not the trainer's --site text (numbers that occur naturally in FineWeb
       text: that site has no fixed prompt).
  ctx  x as the FIRST operand of the trainer's format 0 line "Q: {a} + {b} = {a + b}\\n" with b = 0 ("Q: x + 0 = x\\n"),
       placed as in the trainer's eval rows (evalset): seq - len(line) tokens of val.bin text, then the line, 512 tokens
       in all; the state at a's token (the trainer's ia: the operand's last token). Causal attention: that state sees the
       text and "Q: x" only (b and the answer come after it), so one fixed text chunk (CARRIER_SEED) serves all 1000
       numbers and the forward is the shared prefix (text + "Q: ", cached once) plus the number token. Alternative
       carriers (CKA only): three other text chunks, and x as the second operand ("Q: 0 + x = x\\n").
Layers: the output of decoder block L (1-based; the trainer's forward hooks on model.model.layers[L - 1]; L = n_layers is
the last block's output, before the final norm). Also the input-embedding rows of the number tokens (tied = output rows).
CKA: pretrain_helix.cka(H, centred_gram(helix3(0..999))), the trainer's implementation and its K1000 (cka_emb_helix3);
states in fp32 (no autocast), stored as float16 (the CKA of the float16 states is asserted equal within 1e-3).
Double paths (--replay): rebuild a logged step's TRAINING batch by replaying the trainer's RNG streams (held-out split
and rehearsal on seed, pairs on 1000 + seed, formats and text offsets on 2000 + seed; default path, no --domains /
--held_operands), take the operand states at the ops positions under bf16 autocast as the trainer does, and compare
the mean CKA over its read_layers with the logged cka_ops_helix3 of that step (the state after S updates is the one eval
step S's cka_ops_helix3 used; a final --save_model state is one update after the last logged one); the embedding CKA
of the state after S updates against the logged cka_emb_helix3 of eval step S - 1 (evaluated after its update).

    python tools/fig1_export.py --run <run_*.json> --out <npz> [--replay] [--layers 2,4,6] STATE [STATE ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

import numlib as nl
from pretrain_helix import build_model, centred_gram, cka
from pretrain_range import BUCKETS, FORMATS, RANGE, features, held_out

CARRIER_SEED = 20261022   # text chunk k of the ctx carrier: default_rng(CARRIER_SEED + k).integers(0, len(val) - n)
N_ALT_CTX = 3             # alternative text chunks k = 1..3 (CKA only)


def load_state(p):
    """(state_dict, updates or None): a --save_at file ({"model", "updates", ...}) or a --save_model state_dict."""
    st = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(st, dict) and "model" in st and "updates" in st:
        return st["model"], int(st["updates"])
    return st, None


class Hooked:
    """The model with forward hooks on the outputs of the decoder blocks in `layers` (1-based)."""

    def __init__(self, model, layers):
        self.model, self.got = model, {}
        for L in layers:
            model.model.layers[L - 1].register_forward_hook(lambda _m, _i, o, L=L: self.got.__setitem__(L, nl._hidden(o)))


def replay_batches(a, enc, train, want):
    """{step: (rows [bs, seq] int64, ops [(row, position, value)])} for the steps in `want`: the trainer's batch() on the
    default path (pretrain_range.py main(): held_out / rehearsal on default_rng(seed), draw_pair on default_rng(1000 +
    seed), the line format and the text offset on default_rng(2000 + seed), consumed in the trainer's order)."""
    rng_split = np.random.default_rng(a.seed)
    held = held_out(rng_split)
    held_set = {(min(x, y), max(x, y)) for v in held.values() for x, y in v}
    rehearsal = {}
    for la, lb in BUCKETS:
        prs = set()
        while len(prs) < min(100, 25 if (la, lb) == (1, 1) else 100):
            x, y = int(rng_split.integers(*RANGE[la])), int(rng_split.integers(*RANGE[lb]))
            if (min(x, y), max(x, y)) not in held_set:
                prs.add((x, y))
        rehearsal[(la, lb)] = sorted(prs)
    reh = [q for v in rehearsal.values() for q in v]
    data_rng, rng = np.random.default_rng(1000 + a.seed), np.random.default_rng(2000 + a.seed)

    def draw_pair():
        if data_rng.random() < 0.05:
            return reh[int(data_rng.integers(len(reh)))]
        while True:
            la, lb = BUCKETS[int(data_rng.integers(len(BUCKETS)))]
            if data_rng.random() < 0.5:
                la, lb = lb, la
            x, y = int(data_rng.integers(*RANGE[la])), int(data_rng.integers(*RANGE[lb]))
            if (min(x, y), max(x, y)) not in held_set:
                return x, y
    pieces = []
    for f in FORMATS:
        pre, rest = f.split("{a}"); mid, post = rest.split("{b}")
        pieces.append((enc(pre) if pre else [], enc(mid), enc(post)))
    num = [enc(str(v)) for v in range(1999)]
    nl_ids = enc("\n")

    def line(x, y, f):
        pre, mid, post = pieces[f]
        ids = pre + num[x] + mid + num[y] + post
        return ids + num[x + y] + nl_ids, len(pre) + len(num[x]) - 1, len(pre) + len(num[x]) + len(mid) + len(num[y]) - 1
    out, want = {}, set(want)
    for s in range(max(want) + 1):
        rows, ops = [], []
        for r in range(a.bs):
            parts = []
            for _ in range(a.arith_lines):
                x, y = draw_pair()
                parts.append((x, y, int(rng.integers(len(FORMATS)))))
            L_ = [line(*p) for p in parts]
            n = a.seq - sum(len(q[0]) for q in L_)
            o = int(rng.integers(0, len(train) - n))
            if s in want:
                row = list(np.asarray(train[o:o + n], dtype=np.int64)) + sum((q[0] for q in L_), [])
                off = n
                for (ids, ia, ib), (x, y, _) in zip(L_, parts):
                    ops += [(r, off + ia, x), (r, off + ib, y)]
                    off += len(ids)
                assert len(row) == a.seq
                rows.append(row)
        if s in want:
            out[s] = (np.array(rows, dtype=np.int64), np.array(ops, dtype=np.int64))
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the run's run_*.json (args, read_layers, the logged curve)")
    ap.add_argument("--out", required=True, help="output .npz (a .json summary next to it)")
    ap.add_argument("--layers", default="", help="decoder blocks (1-based); default: the trainer's read_layers + the last")
    ap.add_argument("--replay", action="store_true", help="double paths on the replayed training batches (see docstring)")
    ap.add_argument("--labels", default="", help="comma list: a label per state (default: upd<S> / final)")
    ap.add_argument("states", nargs="+")
    o = ap.parse_args()
    t0 = time.time()
    R = json.loads(Path(o.run).read_text())
    a = Namespace(**R["args"])
    assert a.tokenization == "whole" and not getattr(a, "domains", "") and not getattr(a, "held_operands", 0), \
        "whole tokens, default path only"
    steps, read_layers = R["steps"], R["read_layers"]
    layers = [int(v) for v in o.layers.split(",")] if o.layers else sorted(set(read_layers) | {a.n_layers})
    assert all(1 <= L <= a.n_layers for L in layers), layers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    val = np.memmap(Path(a.data) / "val.bin", dtype=np.uint32, mode="r")
    train = np.memmap(Path(a.data) / "train.bin", dtype=np.uint32, mode="r")
    xs = np.arange(1000)
    num_tok = [enc(str(v)) for v in xs]
    assert all(len(t) == 1 for t in num_tok) and len({t[0] for t in num_tok}) == 1000, "0..999 must be single tokens"
    num_tok = np.array([t[0] for t in num_tok], dtype=np.int64)
    K1000 = centred_gram(features("helix3", xs), "cpu")
    dec = lambda ids: [tok.decode([int(t)]) for t in ids]

    # ------------------------------------------------------------------ site qx: "Q: x" at position 0 (the trainer's PRE)
    QX = np.array([enc(f"Q: {v}") for v in xs], dtype=np.int64)   # ragged would fail here: one length for every x
    assert QX.ndim == 2 and (QX[:, -1] == num_tok).all(), "Q: x must end in x's own token"
    p_qx = QX.shape[1] - 1
    # ------------------------------------------------------------------ site ctx: operand a of "Q: x + 0 = x\n" after text
    pre0, rest0 = FORMATS[0].split("{a}"); mid0, post0 = rest0.split("{b}")
    pre0, mid0, post0 = enc(pre0), enc(mid0), enc(post0)
    nl_ids = enc("\n")

    def carrier(x, k=0, second=False):
        """(row ids, position of x): the eval-row layout (text chunk k of val.bin, then the line, seq tokens in all)."""
        ln = (pre0 + [int(num_tok[x])] + mid0 + [int(num_tok[0])] + post0 if not second else
              pre0 + [int(num_tok[0])] + mid0 + [int(num_tok[x])] + post0) + [int(num_tok[x])] + nl_ids
        n = a.seq - len(ln)
        off = int(np.random.default_rng(CARRIER_SEED + k).integers(0, len(val) - n))
        row = list(np.asarray(val[off:off + n], dtype=np.int64)) + ln
        p = n + len(pre0) + (0 if not second else 1 + len(mid0))
        assert len(row) == a.seq and row[p] == num_tok[x] and enc(tok.decode(ln)) == ln, (x, k, second)
        return row, p
    lens = {len(carrier(x)[0]) for x in (0, 7, 42, 999)} | {carrier(x)[1] for x in (0, 7, 42, 999)}
    assert len(lens) == 2, "the carrier row length and x's position must not depend on x"
    # experiment-audit vii: the exact token sequences and the positions whose state is taken
    show = []
    for x in (7, 42, 538):
        row, p = carrier(x)
        assert tok.decode([row[p]]) == str(x) and tok.decode([int(QX[x, p_qx])]) == str(x)
        show.append(f"x={x}: qx ids {QX[x].tolist()} tokens {dec(QX[x])} -> state at index {p_qx} ({tok.decode([int(QX[x, p_qx])])!r}); "
                    f"ctx row len {len(row)}, text {p - len(pre0)} tokens (val.bin offset chunk k=0), tail tokens "
                    f"{dec(row[p - 8:])} -> state at index {p} ({tok.decode([row[p]])!r}), i.e. the "
                    f"{len(pre0) + 1}th token of the line {tok.decode(row[p - len(pre0):])!r}")
    print("\n".join(show), flush=True)
    # the training range of operand a's position (format 0, any operand / answer lengths): the carrier is a training form
    p_rng = []
    for la in (1, 2, 3):
        for lb in (1, 2, 3):
            for ns in (1, 2):
                L_ = len(pre0) + 1 + len(mid0) + 1 + len(post0) + ns + len(nl_ids)
                p_rng.append(a.seq - L_ + len(pre0))
    p_ctx = carrier(0)[1]
    assert min(p_rng) <= p_ctx <= max(p_rng), (p_ctx, p_rng)

    model = build_model(a, tok, "cpu")
    H = Hooked(model, sorted(set(layers) | set(read_layers)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    num_t = torch.tensor(num_tok, device=dev)

    def states_qx():
        model.model(input_ids=torch.tensor(QX, device=dev), use_cache=False)
        return {L: H.got[L][:, p_qx].float().cpu() for L in layers}

    def states_carrier(k=0, second=False, chunk=125):
        """Shared prefix (text + the line up to x) once, then the number token with the prefix's KV cache."""
        from transformers import DynamicCache
        row0, p = carrier(0, k, second)
        prefix = torch.tensor([row0[:p]], device=dev)
        legacy = model.model(input_ids=prefix, use_cache=True).past_key_values.to_legacy_cache()
        assert len(legacy) == a.n_layers and legacy[0][0].shape[0] == 1 and legacy[0][0].shape[2] == p, legacy[0][0].shape
        out = {L: [] for L in layers}
        for i in range(0, 1000, chunk):
            xb = num_t[i:i + chunk]
            cache = DynamicCache.from_legacy_cache(tuple(   # a fresh copy of the prefix cache for this chunk
                (kk.expand(len(xb), -1, -1, -1).contiguous(), vv.expand(len(xb), -1, -1, -1).contiguous()) for kk, vv in legacy))
            model.model(input_ids=xb[:, None], past_key_values=cache, use_cache=True)
            for L in layers:
                out[L].append(H.got[L][:, -1].float().cpu())
        return {L: torch.cat(v) for L, v in out.items()}, p

    def check_carrier(S_ctx, k=0, second=False, xs_chk=(0, 7, 42, 538, 999)):
        """double path of the cache: full forward of the whole 512-token carrier row (the line through its newline)."""
        rows = [carrier(x, k, second) for x in xs_chk]
        model.model(input_ids=torch.tensor([r for r, _ in rows], device=dev), use_cache=False)
        d = max(float((H.got[L][j, p].float().cpu() - S_ctx[L][x]).abs().max() / (S_ctx[L][x].abs().max() + 1e-12))
                for L in layers for j, ((_, p), x) in enumerate(zip(rows, xs_chk)))
        assert d < 1e-3, ("cached carrier states differ from the full-row forward", d)
        return d

    curve = {e["step"]: e for e in R["curve"]}
    labels = o.labels.split(",") if o.labels else None
    res = {"run": str(Path(o.run).resolve()), "code_sha_run": R.get("code_sha"), "steps": steps, "read_layers": read_layers,
           "layers": layers, "export_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12],
           "trainer_sha": hashlib.sha256((Path(__file__).parent / "pretrain_range.py").read_bytes()).hexdigest()[:12],
           "carrier": {"ctx": f"operand a of 'Q: x + 0 = x\\n' (format 0, {a.seq - p_ctx + len(pre0)} tokens) after "
                       f"{p_ctx - len(pre0)} val.bin tokens (one chunk: default_rng({CARRIER_SEED}).integers(0, len(val) - n)); "
                       f"state at index {p_ctx}; alternatives: chunks k = 1..{N_ALT_CTX} (seed {CARRIER_SEED} + k), and "
                       f"operand b of 'Q: 0 + x = x\\n' (index {carrier(0, 0, True)[1]})",
                       "qx": f"'Q: x' at position 0, state at index {p_qx}", "audit_vii": show},
           "device": dev, "ckpt": []}
    arrays = {f"{site}__L{L}": [] for site in ("ctx", "qx") for L in layers}
    emb, ups = [], []
    loaded, rep_of = [], {}   # rep_of[S]: the logged step whose training batch is replayed for the state after S updates
    for i, sp in enumerate(o.states):
        sd, S = load_state(sp)
        if S is None:   # a --save_model file: the final state
            S = steps
        loaded.append((sp, S))
        if o.replay and S in curve:          # exact: eval step S's cka_ops_helix3 used the state after S updates
            rep_of[S] = S
        elif o.replay and S == steps:        # the final state: one update after the last logged forward
            rep_of[S] = steps - 1
    rep_steps = {}
    if rep_of:
        print(f"replaying the trainer's batches up to step {max(rep_of.values())} ...", flush=True)
        t_r = time.time()
        rep_steps = replay_batches(a, enc, train, sorted(set(rep_of.values())))
        print(f"replayed in {time.time() - t_r:.0f}s", flush=True)
    for i, (sp, S) in enumerate(loaded):
        sd, _ = load_state(sp)
        model.load_state_dict(sd)
        model.to(dev).eval()
        ent = {"file": str(Path(sp).resolve()), "updates": S, "tokens": S * a.bs * a.seq,
               "label": labels[i] if labels else ("final" if S == steps else f"upd{S}"), "cka": {}, "checks": {}}
        Sq = states_qx()
        Sc, _ = states_carrier()
        ent["checks"]["carrier_cache_vs_full_rel_maxdiff"] = check_carrier(Sc)
        for site, St in (("ctx", Sc), ("qx", Sq)):
            for L in layers:
                c32 = float(cka(St[L], K1000))
                c16 = float(cka(St[L].half().float(), K1000))
                assert abs(c32 - c16) < 1e-3, (site, L, c32, c16)
                ent["cka"][f"{site}@L{L}"] = c32
                arrays[f"{site}__L{L}"].append(St[L].half().numpy())
        for site in ("ctx", "qx"):
            ent["cka"][f"{site}@mean_read_layers"] = float(np.mean([ent["cka"][f"{site}@L{L}"] for L in read_layers if L in layers]))
        E = model.get_input_embeddings().weight[num_t].float().cpu()
        ent["cka"]["emb"] = float(cka(E, K1000))
        emb.append(E.half().numpy()); ups.append(S)
        # alternative carriers (CKA only): other text chunks; x as the second operand
        alt = {}
        for k in range(1, N_ALT_CTX + 1):
            Sa, _ = states_carrier(k)
            alt.update({f"ctx_k{k}@L{L}": float(cka(Sa[L], K1000)) for L in layers})
        Sb, _ = states_carrier(0, second=True)
        ent["checks"]["carrier_b_cache_vs_full_rel_maxdiff"] = check_carrier(Sb, 0, True)
        alt.update({f"ctx_b@L{L}": float(cka(Sb[L], K1000)) for L in layers})
        ent["cka_alt"] = alt
        # double paths against the run's own logs
        if S - 1 in curve and "cka_emb_helix3" in curve[S - 1]:
            ent["checks"]["emb_vs_logged(step S-1)"] = [ent["cka"]["emb"], curve[S - 1]["cka_emb_helix3"]]
        if S in rep_of:
            Sr = rep_of[S]
            rows, ops = rep_steps[Sr]
            assert all(tok.decode([int(rows[r, p])]) == str(v) for r, p, v in ops.tolist()), "replayed ops decode"
            x = torch.tensor(rows, device=dev)
            # bf16 autocast forward of the decoder only (the hooked block outputs; no LM head: its 32 x 512 x 100k logits do
            # not fit a 24 GB CPU job). Tested on the CPU regression runs: the trainer's own sequence (train mode, grad on,
            # labels) gives the same CKA bit for bit; the read-out then as the trainer's cka_ops_h3 (no_grad + autocast)
            with torch.autocast(dev, dtype=torch.bfloat16):
                model.model(input_ids=x, use_cache=False)
            with torch.autocast(dev, dtype=torch.bfloat16):
                Kb = centred_gram(features("helix3", ops[:, 2]), dev)
                cs = {L: float(cka(H.got[L][torch.tensor(ops[:, 0], device=dev), torch.tensor(ops[:, 1], device=dev)], Kb))
                      for L in read_layers}
            H.got.clear()
            ent["checks"]["replay"] = {"batch_step": Sr, "state_updates": S, "n_ops": int(len(ops)),
                                       "cka_per_layer": cs, "mine_mean": float(np.mean(list(cs.values()))),
                                       "logged_cka_ops_helix3": curve[Sr]["cka_ops_helix3"],
                                       "exact_state": Sr == S}
        res["ckpt"].append(ent)
        print(json.dumps({k: v for k, v in ent.items() if k != "file"}), flush=True)
    outp = Path(o.out); outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, **{f"states__{k}": np.stack(v) for k, v in arrays.items()}, emb=np.stack(emb),
             updates=np.array(ups), tokens=np.array(ups) * a.bs * a.seq, layers=np.array(layers),
             sites=np.array(["ctx", "qx"]), values=xs,
             cka=np.array([[[e["cka"][f"{s}@L{L}"] for L in layers] for s in ("ctx", "qx")] for e in res["ckpt"]]),
             cka_emb=np.array([e["cka"]["emb"] for e in res["ckpt"]]),
             labels=np.array([e["label"] for e in res["ckpt"]]))
    res["sec"] = round(time.time() - t0, 1)
    outp.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(f"wrote {outp} ({outp.stat().st_size / 1e6:.1f} MB) in {res['sec']}s")


if __name__ == "__main__":
    main()
