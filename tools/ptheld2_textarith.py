"""Descriptive count for runs/ptheld2 (fresh build audit 2026-10-21, M3): how often does each number 0..999 occur in an
addition / equals context of the natural FineWeb text in train.bin -- the number token right next to a '+' token
(optionally across one space token), on either side, or right after a '=' token (optionally across one space). These are
arithmetic statements every arm sees in the text (the per-step content scan covers the synthetic lines only). A coarse
rule (over-counts e.g. '+100' as a sign); per number, seed-independent. Each position is counted once (chunks overlap by
2 tokens for the neighbours only).
    python tools/ptheld2_textarith.py --out /scratch/yh6210/llmadd_torch/runs/ptheld2/textarith.json"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

B = "/scratch/yh6210/llmadd_torch"
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=f"{B}/pretrain_data")
ap.add_argument("--tokenizer", default=f"{B}/models/OLMo-2-1124-7B/stage1-step2000-tokens9B")
ap.add_argument("--out", required=True)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.tokenizer)
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
num = np.array([enc(str(v))[0] for v in range(1000)])
assert all(len(enc(str(v))) == 1 for v in range(1000))
lut = np.full(len(tok), -1, dtype=np.int64); lut[num] = np.arange(1000)
P = np.array(sorted({t for s in ("+", " +", " + ", "+ ") for t in enc(s) if tok.decode([t]).strip() == "+"}))
E = np.array(sorted({t for s in ("=", " =") for t in enc(s) if tok.decode([t]).strip() == "="}))
S = np.array(sorted(set(enc(" "))))
train = np.memmap(Path(args.data) / "train.bin", dtype=np.uint32, mode="r")
n, C = len(train), 50_000_000
cnt = np.zeros(1000, dtype=np.int64)
for i in range(0, n, C):
    lo, hi = max(0, i - 2), min(n, i + C + 2)
    t = np.asarray(train[lo:hi], dtype=np.int64)
    v = lut[t]
    isP, isE, isS = np.isin(t, P), np.isin(t, E), np.isin(t, S)
    nxt = lambda a, k: np.concatenate([a[k:], np.zeros(k, bool)])
    prv = lambda a, k: np.concatenate([np.zeros(k, bool), a[:-k]])
    hit = (nxt(isP, 1) | (nxt(isS, 1) & nxt(isP, 2)) | prv(isP, 1) | (prv(isS, 1) & prv(isP, 2))
           | prv(isE, 1) | (prv(isS, 1) & prv(isE, 2)))
    pos = np.arange(lo, hi)
    m = (v >= 0) & hit & (pos >= i) & (pos < min(n, i + C))   # each position once
    cnt += np.bincount(v[m], minlength=1000)
Path(args.out).write_text(json.dumps({"rule": "number token next to '+' (optionally across one space token, either side) or "
                                              "right after '=' (optionally across one space)", "plus_ids": P.tolist(),
                                      "eq_ids": E.tolist(), "space_ids": S.tolist(), "train_bin_tokens": int(n),
                                      "count": cnt.tolist()}))
print(f"wrote {args.out}: total {int(cnt.sum())}; per-block medians " +
      ", ".join(f"b{b} {int(np.median(cnt[100 * b:100 * b + 100]))}" for b in range(1, 10)))
