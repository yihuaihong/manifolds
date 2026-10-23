"""Downstream (general-language) evaluation of saved from-scratch models (pretrain_helix.py / pretrain_range.py
--save_model): BLiMP, LAMBADA (OpenAI), HellaSwag, ARC-Easy, PIQA, zero-shot, full splits; lm-eval-harness definitions except BLiMP
(task definition: no leading space, strict >; the harness scores " " + sentence and counts a tie correct).
Intent spec and pre-registered read-out: llmadd_torch/runs/downstream/AUDIT.md (copy: report/audits/downstream_AUDIT.md).

  BLiMP      67 paradigms x 1000 pairs; acc = sum log p(good) > sum log p(bad) (strict), whole sentence after <eos>
  LAMBADA    5153; context = text without its last word, continuation = " " + last word; acc = every token greedy,
             ppl = exp(mean over items of the summed NLL of the word)
  HellaSwag  validation 10042; harness preprocess; acc_norm = argmax loglik / len(ending chars) (primary), acc
  ARC-Easy   test 2376; "Question: q\\nAnswer:" + " " + choice; acc = argmax summed loglik (primary), acc_norm
  PIQA       validation 1838; "Question: goal\\nAnswer:" + " " + sol; acc (primary), acc_norm
Every sequence starts with the document separator <eos> (pretrain_data.py: documents are text + <eos>) for the project
models (--prefix eos); --prefix harness reproduces lm-eval-harness (a prefix token only for an empty context) for the
GPT-2 known-answer check. At most --max_len input positions (512 = the models' training windows); longer sequences are
cut from the left (n_trunc is reported). fp32, TF32 off (the load-check replica of the in-training bpb: bf16 autocast,
as in training). Batched right-padded scoring (causal: padding after a sequence
cannot change its logits); double path: 24 random items per benchmark rescored one at a time with full logits.

    python tools/downstream_eval.py --out_dir <dir> <model_*.pt | glob> [...]          # saved project models
    python tools/downstream_eval.py --out_dir <dir> --random_init <run_*.json>        # known answer: random init
    python tools/downstream_eval.py --out_dir <dir> --hf gpt2 --prefix harness --max_len 1024   # known answer: GPT-2
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

DATA = "/scratch/yh6210/llmadd_torch/data/general"
BENCH = ("blimp", "lambada", "hellaswag", "arc_easy", "piqa")
EXPECT_N = {"blimp": 67000, "lambada": 5153, "hellaswag": 10042, "arc_easy": 2376, "piqa": 1838}
PRIMARY = {"blimp": "acc", "lambada": "acc", "hellaswag": "acc_norm", "arc_easy": "acc", "piqa": "acc"}
CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ----------------------------------------------------------------------------------------------------------- data
def hs_pre(text: str) -> str:
    """lm-eval-harness hellaswag preprocess (WikiHow bracket artifacts)."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def load_items(root: str, limit: int = 0) -> dict:
    """benchmark -> list of dict(ctx, conts, choice_text, gold, ...); full splits unless --limit (smoke tests only)."""
    import pandas as pd
    root = Path(root)
    it = {}
    rows, ident = [], 0
    files = sorted(glob.glob(str(root / "blimp/*/*.parquet")))
    assert len(files) == 67, f"BLiMP: {len(files)} paradigm files, expected 67"
    for f in files:
        df = pd.read_parquet(f)
        assert len(df) == 1000 and df["UID"].nunique() == 1, f
        for r in df.itertuples():
            ident += r.sentence_good == r.sentence_bad
            rows.append(dict(ctx="", conts=[r.sentence_good, r.sentence_bad], choice_text=[r.sentence_good, r.sentence_bad],
                             gold=0, uid=r.UID, field=r.field))
    # the HF copy has 7 pairs with good == bad (passive_1 324, 810; principle_A_case_2 105, 287, 372, 816, 967): kept, as
    # in lm-eval-harness; a tie counts wrong for every model (-0.01 points on the level, 0 on a paired difference)
    assert ident == 7, ("BLiMP pairs with good == bad", ident)
    it["blimp"] = rows
    lam = [json.loads(x)["text"] for x in open(root / "lambada_openai/data/lambada_test.jsonl")]
    it["lambada"] = [dict(ctx=t.rsplit(" ", 1)[0], conts=[" " + t.rsplit(" ", 1)[1]], choice_text=[t.rsplit(" ", 1)[1]], gold=0)
                     for t in lam]
    hs = pd.read_parquet(root / "hellaswag/data/validation-00000-of-00001.parquet")
    it["hellaswag"] = []
    for r in hs.itertuples():
        ends = [hs_pre(e) for e in r.endings]
        it["hellaswag"].append(dict(ctx=hs_pre(r.activity_label + ": " + r.ctx_a + " " + r.ctx_b.capitalize()),
                                    conts=[" " + e for e in ends], choice_text=ends, gold=int(r.label)))
    arc = pd.read_parquet(root / "ai2_arc/ARC-Easy/test-00000-of-00001.parquet")
    it["arc_easy"] = []
    for r in arc.itertuples():
        labels, texts = list(r.choices["label"]), [str(t) for t in r.choices["text"]]
        assert r.answerKey in labels, (r.id, r.answerKey, labels)
        it["arc_easy"].append(dict(ctx=f"Question: {r.question}\nAnswer:", conts=[" " + t for t in texts], choice_text=texts,
                                   gold=labels.index(r.answerKey)))
    pq = pd.read_parquet(root / "piqa_parquet/plain_text/validation/0000.parquet")
    it["piqa"] = [dict(ctx=f"Question: {r.goal}\nAnswer:", conts=[" " + r.sol1, " " + r.sol2], choice_text=[r.sol1, r.sol2],
                       gold=int(r.label)) for r in pq.itertuples()]
    for b in BENCH:
        assert len(it[b]) == EXPECT_N[b], (b, len(it[b]))
        assert all(0 <= x["gold"] < len(x["conts"]) for x in it[b]), b
        if limit:
            it[b] = it[b][:limit] if b != "blimp" else it[b][::max(1, len(it[b]) // limit)][:limit]
    return it


# ---------------------------------------------------------------------------------------------------- tokenising
def encode_pair(tok, ctx: str, cont: str):
    """lm-eval-harness _encode_pair: trailing context spaces move to the continuation; the continuation's tokens are the
    whole string's tokens after the context's."""
    n = len(ctx) - len(ctx.rstrip())
    if n:
        cont, ctx = ctx[-n:] + cont, ctx[:-n]
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    whole = enc(ctx + cont)
    c = enc(ctx) if ctx else []
    return c, whole[len(c):], whole[:len(c)] == c


def build_requests(tok, items: dict, prefix: str, max_len: int, doc_tok: int):
    """-> per benchmark: list of (seq int64 array, start) per (item, choice); targets seq[start:] are scored from the
    logits at positions start-1 .. len-2; seq has at most max_len + 1 tokens (cut from the left)."""
    reqs, stats = {}, {}
    for b, rows in items.items():
        R, ntr, nmis, nsplit = [], 0, 0, 0
        for x in rows:
            for cont in x["conts"]:
                c, w, ok = encode_pair(tok, x["ctx"], cont)
                nsplit += not ok
                assert w, (b, x["ctx"][-50:], cont)
                if tok.decode(w).strip() != cont.strip():
                    nmis += 1
                if prefix == "eos" or not c:
                    c = [doc_tok] + c
                seq = c + w
                if len(seq) > max_len + 1:
                    seq, ntr = seq[-(max_len + 1):], ntr + 1
                start = len(seq) - len(w)
                assert start >= 1, (b, len(seq), len(w))
                R.append((np.asarray(seq, dtype=np.int64), start))
        reqs[b] = R
        stats[b] = {"n_requests": len(R), "n_trunc": ntr, "n_cont_decode_mismatch": nmis, "n_ctx_not_prefix": nsplit,
                    "max_seq": max(len(s) for s, _ in R), "mean_seq": float(np.mean([len(s) for s, _ in R]))}
        assert nsplit <= 0.001 * len(R), (b, "context tokens are not a prefix of the joint tokens", nsplit)
        assert nmis <= 0.001 * len(R), (b, "scored continuation tokens do not decode back to the continuation", nmis)
    return reqs, stats


def show(tok, seq, start) -> str:
    return (tok.decode(seq[:start].tolist()) + "⟦" + tok.decode(seq[start:].tolist()) + "⟧"
            + f"   [scored token idx {start}..{len(seq) - 1}, seq len {len(seq)}]")


def print_examples(tok, items, reqs, val=None, eos=None):
    print("#### decoded inputs (tokenised -> decoded; scored continuation in ⟦ ⟧; * = gold)", flush=True)
    for b in BENCH:
        k = 0
        offs = np.cumsum([0] + [len(x["conts"]) for x in items[b]])
        picks = (0, len(items[b]) // 2) if b == "blimp" else (0, 1)
        for i in picks:
            for j in range(len(items[b][i]["conts"])):
                s, st = reqs[b][offs[i] + j]
                print(f"[{b} item {i} choice {j}{' *' if j == items[b][i]['gold'] else ''}] {show(tok, s, st)!r}", flush=True)
            k += 1
    if val is not None:
        pos = np.flatnonzero(np.asarray(val[:200000]) == eos)[:2]
        for p in pos:
            w = val[max(0, p - 12):p + 24].tolist()
            print(f"[training stream val.bin, doc boundary at token {p}] {tok.decode(w)!r}", flush=True)


# ------------------------------------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_batched(model, reqs, dev, tok_budget=65536, head_chunk=4096):
    """Summed log-likelihood and all-greedy flag of every request (right-padded batches sorted by length)."""
    base, head = model.base_model, model.get_output_embeddings()
    ll = np.zeros(len(reqs)); gr = np.zeros(len(reqs), dtype=bool)
    order = sorted(range(len(reqs)), key=lambda i: -len(reqs[i][0]))
    i = 0
    while i < len(order):
        L = len(reqs[order[i]][0]) - 1
        idx = order[i:i + max(1, tok_budget // L)]
        i += len(idx)
        ids = torch.zeros((len(idx), L), dtype=torch.long)
        rows, cols, tg, own = [], [], [], []
        for r, k in enumerate(idx):
            s, st = reqs[k]
            ids[r, :len(s) - 1] = torch.from_numpy(s[:-1])
            p = np.arange(st - 1, len(s) - 1)
            rows.append(np.full(len(p), r)); cols.append(p); tg.append(s[st:]); own.append(np.full(len(p), k))
        h = base(input_ids=ids.to(dev)).last_hidden_state
        rows, cols = torch.from_numpy(np.concatenate(rows)).to(dev), torch.from_numpy(np.concatenate(cols)).to(dev)
        tg, own = torch.from_numpy(np.concatenate(tg)).to(dev), np.concatenate(own)
        tl, hit = [], []
        for a in range(0, len(tg), head_chunk):
            lg = head(h[rows[a:a + head_chunk], cols[a:a + head_chunk]]).float()
            t = tg[a:a + head_chunk]
            tl.append(Fn.log_softmax(lg, -1).gather(1, t[:, None])[:, 0]); hit.append(lg.argmax(-1) == t)
        tl, hit = torch.cat(tl).double().cpu().numpy(), torch.cat(hit).cpu().numpy()
        np.add.at(ll, own, tl)
        bad = np.zeros(len(reqs), dtype=bool); np.logical_or.at(bad, own, ~hit)
        for k in idx:
            gr[k] = not bad[k]
    return ll, gr


@torch.no_grad()
def score_single(model, seq, st, dev):
    """Independent path: one unpadded sequence through the full HF forward (all logits)."""
    lg = model(input_ids=torch.from_numpy(seq[:-1])[None].to(dev)).logits[0].float()
    t = torch.from_numpy(seq[st:]).to(dev)
    p = torch.arange(st - 1, len(seq) - 1, device=dev)
    return float(Fn.log_softmax(lg[p], -1).gather(1, t[:, None]).sum()), bool((lg[p].argmax(-1) == t).all())


def metrics(b, items, ll, gr):
    offs = np.cumsum([0] + [len(x["conts"]) for x in items])
    out = {"n": len(items)}
    if b == "blimp":
        g, w = ll[offs[:-1]], ll[offs[:-1] + 1]
        hit = g > w
        out["acc"] = 100 * float(hit.mean())
        out["n_ties"] = int((g == w).sum())
        uid = np.array([x["uid"] for x in items]); fld = np.array([x["field"] for x in items])
        out["per_paradigm"] = {u: 100 * float(hit[uid == u].mean()) for u in sorted(set(uid))}
        out["per_field"] = {f: 100 * float(hit[fld == f].mean()) for f in sorted(set(fld))}
        out["acc_paradigm_mean"] = float(np.mean(list(out["per_paradigm"].values())))
        out["hits"] = "".join("1" if h else "0" for h in hit)
        return out
    if b == "lambada":
        out["acc"] = 100 * float(gr.mean())
        out["ppl"] = float(math.exp(-ll.mean()))
        out["mean_nll_word"] = float(-ll.mean())
        out["hits"] = "".join("1" if h else "0" for h in gr)
        return out
    pred, predn, chance = [], [], []
    for i, x in enumerate(items):
        l = ll[offs[i]:offs[i + 1]]
        pred.append(int(np.argmax(l)))
        predn.append(int(np.argmax(l / np.array([float(len(c)) for c in x["choice_text"]]))))
        chance.append(1 / len(x["conts"]))
    gold = np.array([x["gold"] for x in items])
    h, hn = np.array(pred) == gold, np.array(predn) == gold
    out.update(acc=100 * float(h.mean()), acc_norm=100 * float(hn.mean()), chance=100 * float(np.mean(chance)),
               hits_acc="".join("1" if v else "0" for v in h), hits_acc_norm="".join("1" if v else "0" for v in hn),
               gold_share=[100 * float((gold == k).mean()) for k in range(max(len(x["conts"]) for x in items))])
    return out


def length_heuristics(b, items, reqs):
    """Scores of 'fewest scored tokens wins' (what a near-uniform random model does with summed logliks)."""
    offs = np.cumsum([0] + [len(x["conts"]) for x in items])
    n = np.array([len(s) - st for s, st in reqs])
    if b == "blimp":
        g, w = n[offs[:-1]], n[offs[:-1] + 1]
        return {"shorter_good_share": 100 * float((g < w).mean()), "equal_len_share": 100 * float((g == w).mean()),
                "eq_mask": (g == w)}
    if b == "lambada":
        return {}
    hit = [int(np.argmin(n[offs[i]:offs[i + 1]]) == x["gold"]) for i, x in enumerate(items)]
    eq = np.array([len(set(n[offs[i]:offs[i + 1]].tolist())) == 1 for i in range(len(items))])
    return {"fewest_tokens_acc": 100 * float(np.mean(hit)), "equal_len_share": 100 * float(eq.mean()), "eq_mask": eq}


def evaluate(model, tok, items, reqs, dev, rng_seed=0):
    model.eval()
    res = {}
    for b in BENCH:
        t0 = time.time()
        ll, gr = score_batched(model, reqs[b], dev)
        # double path (audit i): random items, one at a time, full logits
        rng = np.random.default_rng(rng_seed)
        pick = rng.choice(len(reqs[b]), size=min(24, len(reqs[b])), replace=False)
        d, gm = [], 0
        for k in pick:
            l1, g1 = score_single(model, *reqs[b][k], dev)
            d.append(abs(l1 - ll[k])); gm += g1 != gr[k]
        assert max(d) < 1e-2, (b, "batched vs single-sequence loglik differ", max(d))
        assert gm <= 1, (b, "greedy flags differ between the two paths", gm)
        r = metrics(b, items[b], ll, gr)
        r["double_path"] = {"n": len(pick), "max_abs_dloglik": float(max(d)), "greedy_mismatch": int(gm)}
        lh = length_heuristics(b, items[b], reqs[b])
        if b != "lambada":   # accuracy on the items whose choices all have the same number of scored tokens (Amendment A1)
            eq = lh.pop("eq_mask")
            hits = np.array([c == "1" for c in r["hits" if b == "blimp" else "hits_acc"]])
            r["acc_equal_len"] = 100 * float(hits[eq].mean()) if eq.any() else None
            r["n_equal_len"] = int(eq.sum())
            r["chance_equal_len"] = 50.0 if b == "blimp" else (
                100 * float(np.mean([1 / len(x["conts"]) for x, e in zip(items[b], eq) if e])) if eq.any() else None)
        r["length_heuristic"] = lh
        r["sec"] = time.time() - t0
        res[b] = r
        print(f"  {b}: " + ", ".join(f"{k} {r[k]:.2f}" for k in ("acc", "acc_norm", "ppl", "acc_equal_len", "chance")
                                     if r.get(k) is not None) + f"  (n {r['n']}, {r['sec']:.0f}s, dpath {max(d):.1e})",
              flush=True)
    return res


# -------------------------------------------------------------------------------------------------------- models
def run_json_for(mp: Path) -> Path:
    return mp.with_name(mp.name.replace("model_", "run_", 1)[:-3] + ".json")


def load_project_model(mp: Path, dev):
    from pretrain_helix import build_model
    from transformers import AutoTokenizer
    r = json.loads(run_json_for(mp).read_text())
    a = Namespace(**r["args"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    model = build_model(a, tok, dev)
    model.load_state_dict(torch.load(mp, map_location=dev))   # strict
    n = sum(p.numel() for p in model.parameters())
    assert n == r["n_params"], (mp, n, r["n_params"])
    return model.float().eval(), tok, r


LOAD_TOL = 5e-3   # bpb; Amendment A1 (the KA gate requires < 2e-3 on one model per training-code version)


@torch.no_grad()
def load_check(model, tok, r, dev):
    """Replica of the trainer's in-training val_bits_per_byte (pretrain_range.evaluate / offline_lm.py 'orig windows':
    np.random.default_rng(7), 8 batches x 16 offsets of seq + 1 tokens, eval mode, bf16 autocast) vs the run's last logged
    value: a wrongly loaded model differs by >> 1e-2 bpb (cells differ by 0.03-0.1)."""
    a = r["args"]; seq = a["seq"]
    val = np.memmap(Path(a["data"]) / "val.bin", dtype=np.uint32, mode="r")
    vrr = np.random.default_rng(7)
    nats, nbytes = 0.0, 0
    with torch.autocast("cuda" if str(dev).startswith("cuda") else "cpu", dtype=torch.bfloat16):
        for _ in range(8):
            o = vrr.integers(0, len(val) - seq - 1, size=16)
            x = torch.tensor(np.stack([val[i:i + seq + 1].astype(np.int64) for i in o]), device=dev)
            nats += float(Fn.cross_entropy(model(input_ids=x[:, :-1]).logits.float().flatten(0, 1), x[:, 1:].flatten(),
                                           reduction="sum"))
            nbytes += sum(len(tok.decode(row).encode("utf-8")) for row in x[:, 1:].tolist())
    bpb = nats / math.log(2) / nbytes
    logged = r["curve"][-1]["val_bits_per_byte"]
    assert abs(bpb - logged) < LOAD_TOL, ("in-training bpb replica differs from the logged final value", bpb, logged)
    return {"bpb_replica": bpb, "bpb_logged_final": logged, "replica_minus_logged": bpb - logged}


def finished(rj: Path) -> bool:
    try:
        r = json.loads(rj.read_text())
        return r["curve"][-1]["step"] >= r["steps"] - 1
    except (OSError, KeyError, IndexError, ValueError):
        return False


def out_path(out_dir: Path, mp: Path) -> Path:
    parts = mp.resolve().parts
    fam = parts[parts.index("runs") + 1] if "runs" in parts else "other"
    return out_dir / fam / (mp.stem + ".json")


# ---------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="model_*.pt files or globs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--prefix", choices=("eos", "harness"), default="eos")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--random_init", nargs="*", default=[], help="run_*.json files: evaluate an untrained model of that architecture")
    ap.add_argument("--hf", nargs="*", default=[], help="HF model ids (known-answer checks)")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only: first N items per benchmark")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--load_check_only", nargs="*", default=[],
                    help="model_*.pt: only the in-training bpb replica (known answer for loading; no benchmark is scored)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"downstream_eval {CODE_SHA} dev {dev} prefix {a.prefix} max_len {a.max_len} limit {a.limit}"
          f" {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}", flush=True)
    items = load_items(a.data, a.limit)
    cache = {}   # tokenizer path -> (tok, reqs, stats)

    def prep(key, tok):
        if key not in cache:
            eos = tok.eos_token_id
            reqs, st = build_requests(tok, items, a.prefix, a.max_len, eos)
            print(f"#### tokenised with {key}: " + json.dumps(st), flush=True)
            cache[key] = (reqs, st)
            return reqs, st, True
        return (*cache[key], False)

    def save(path, rec):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(rec, indent=1)); tmp.replace(path)
        print("  ->", path, flush=True)

    common = {"eval_code_sha": CODE_SHA, "prefix": a.prefix, "max_len": a.max_len, "limit": a.limit, "dtype": "fp32",
              "data": a.data, "slurm_job": os.environ.get("SLURM_JOB_ID")}
    for hid in a.hf:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hid)
        model = AutoModelForCausalLM.from_pretrained(hid, torch_dtype=torch.float32).to(dev).eval()
        reqs, st, new = prep(hid, tok)
        print_examples(tok, items, reqs) if new else None
        p = out_dir / "known_answer" / f"hf_{hid.replace('/', '_')}_{a.prefix}{'_lim' + str(a.limit) if a.limit else ''}.json"
        print(f"### HF {hid}", flush=True)
        save(p, dict(common, kind="hf", model=hid, tok_stats=st, scores=evaluate(model, tok, items, reqs, dev)))
        del model
    for rjs in a.random_init:
        from pretrain_helix import build_model
        from transformers import AutoTokenizer
        r = json.loads(Path(rjs).read_text()); ar = Namespace(**r["args"])
        tok = AutoTokenizer.from_pretrained(ar.tokenizer)
        torch.manual_seed(0)
        model = build_model(ar, tok, dev).float().eval()
        reqs, st, new = prep(ar.tokenizer, tok)
        print_examples(tok, items, reqs) if new else None
        name = f"randinit_d{ar.d_model}_L{ar.n_layers}"
        print(f"### random init {name} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M)", flush=True)
        p = out_dir / "known_answer" / f"{name}_{a.prefix}{'_lim' + str(a.limit) if a.limit else ''}.json"
        save(p, dict(common, kind="random_init", arch_from=rjs, tok_stats=st, scores=evaluate(model, tok, items, reqs, dev)))
        del model
    for mp in map(Path, a.load_check_only):   # known answer for the loading path: LM metric only, NO benchmark score
        model, tok, r = load_project_model(mp, dev)
        lc = load_check(model, tok, r, dev)
        print(f"### load check only {mp}: " + json.dumps(lc), flush=True)
        save(out_dir / "known_answer" / f"loadcheck_{mp.parent.name}_s{r['args']['seed']}.json",
             dict(common, kind="load_check", model=str(mp), train_code_sha=r.get("code_sha"), load_check=lc))
        del model
    paths = sorted({Path(p) for g in a.models for p in (glob.glob(g) or [g])})
    print(f"### {len(paths)} model files", flush=True)
    for mp in paths:
        rj = run_json_for(mp)
        if not mp.exists() or not finished(rj):
            print("skip (no model or unfinished run):", mp, flush=True); continue
        if time.time() - mp.stat().st_mtime < 120:
            print("skip (model file modified < 2 min ago):", mp, flush=True); continue
        p = out_path(out_dir, mp) if not a.limit else out_path(out_dir / "smoke", mp)
        st_m = mp.stat()
        if p.exists() and not a.force:
            old = json.loads(p.read_text())
            if old.get("eval_code_sha") == CODE_SHA and old.get("model_size") == st_m.st_size and old.get("model_mtime") == st_m.st_mtime:
                print("done already:", p, flush=True); continue
        t0 = time.time()
        model, tok, r = load_project_model(mp, dev)
        ar = r["args"]
        reqs, st, new = prep(ar["tokenizer"], tok)
        if new:
            val = np.memmap(Path(ar["data"]) / "val.bin", dtype=np.uint32, mode="r")
            print_examples(tok, items, reqs, val, tok.eos_token_id)
        lc = load_check(model, tok, r, dev)
        print(f"### {mp}  (lr {ar['lr']}, seed {ar['seed']}, load check: in-training bpb replica {lc['bpb_replica']:.5f} vs "
              f"logged final {lc['bpb_logged_final']:.5f})", flush=True)
        rec = dict(common, kind="project", model=str(mp), run_json=str(rj), model_size=st_m.st_size, model_mtime=st_m.st_mtime,
                   tag=ar["tag"], seed=ar["seed"], lr=ar["lr"], target=ar["target"], schedule=ar.get("schedule"),
                   site=ar.get("site"), d_model=ar["d_model"], n_layers=ar["n_layers"], n_params=r["n_params"],
                   train_code_sha=r.get("code_sha"), val_loss_logged_final=r["curve"][-1]["val_loss"],
                   val_bpb_logged_final=r["curve"][-1]["val_bits_per_byte"], load_check=lc, tok_stats=st,
                   scores=evaluate(model, tok, items, reqs, dev))
        rec["sec_total"] = time.time() - t0
        save(p, rec)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    print("### all done", flush=True)


if __name__ == "__main__":
    main()
