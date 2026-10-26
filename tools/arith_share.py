"""Arithmetic token share of the pretraining rows of tools/pretrain_range.py (the ptlr snapshot, code_sha a3bce3ab63c1)
for --arith_lines k, with a known-answer check against the trainer's own --dry_inputs log (runs/ptshare/AUDIT.md).

The trainer is not changed and not run here. This script replays its row construction for the given seed: the held-out
split and the rehearsal pool from rng_split = default_rng(seed) (pretrain_range.held_out + the same rehearsal loop as
main()), draw_pair on data_rng = default_rng(1000 + seed), the format of each line and the text offset on
rng = default_rng(2000 + seed), the lines tokenised piecewise exactly as main()'s line(). In --dry_inputs mode the trainer
builds its first batch from the same, untouched streams, so the operand positions of rows 0-2 replayed here must equal
the 'train row r (len 512, 2k prior sites at [...])' lines it prints (first 12 positions; exit 3 otherwise).
Share = mean over the replayed rows of (arithmetic tokens in the row) / seq; the rows are the run's own first batches.
    python arith_share.py --code <snapshot tools dir> [--dry <dry_inputs log>] [--batches 20] [--json f] <trainer args>
prints 'ARITH_SHARE {...}' (arith_lines, seed, rows, mean arithmetic tokens per row and per line, share, range)."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True, help="the trainer snapshot's tools dir (pretrain_range.py is imported from it)")
ap.add_argument("--dry", help="the trainer's --dry_inputs log of the same ARGS: rows 0-2 must match the replay")
ap.add_argument("--batches", type=int, default=20, help="batches replayed for the share (default 20 = 640 rows at bs 32)")
ap.add_argument("--json", help="also write the result here")
A, rest = ap.parse_known_args()
tp = argparse.ArgumentParser()   # the trainer flags this replay depends on (the trainer's own defaults)
for k, typ, d in (("--data", str, None), ("--tokenizer", str, None), ("--tokenization", str, None), ("--seed", int, 0),
                  ("--arith_lines", int, 4), ("--seq", int, 512), ("--bs", int, 32), ("--site", str, "ctx")):
    tp.add_argument(k, type=typ, default=d)
T, _ = tp.parse_known_args(rest)
assert T.data and T.tokenizer and T.tokenization == "whole", "whole-number tokenization only (the dose batch)"
assert T.site == "ctx", "the dry rows mark the operand sites only for --site ctx"
assert T.arith_lines >= 1, T.arith_lines
sys.path.insert(0, A.code)
import pretrain_range as pr   # noqa: E402  (main() runs only under __main__)
from transformers import AutoTokenizer   # noqa: E402

assert Path(pr.__file__).resolve().parent == Path(A.code).resolve(), pr.__file__
tok = AutoTokenizer.from_pretrained(T.tokenizer)
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
pieces = []
for f in pr.FORMATS:   # = main(): pieces of a line, so the operand positions are known
    pre, rest_ = f.split("{a}"); mid, post = rest_.split("{b}")
    pieces.append((enc(pre) if pre else [], enc(mid), enc(post)))


def line(a, b, f):
    pre, mid, post = pieces[f]
    A_, B_ = enc(str(a)), enc(str(b))
    ids = pre + A_ + mid + B_ + post
    ia, ib = len(pre) + len(A_) - 1, len(pre) + len(A_) + len(mid) + len(B_) - 1
    return ids + enc(str(a + b)) + enc("\n"), ia, ib


rng_split = np.random.default_rng(T.seed)
held = pr.held_out(rng_split)
held_set = {(min(a, b), max(a, b)) for v in held.values() for a, b in v}
rehearsal = {}
for la, lb in pr.BUCKETS:   # = main()'s rehearsal loop (the next draws of rng_split)
    prs = set()
    while len(prs) < min(100, 25 if (la, lb) == (1, 1) else 100):
        a, b = int(rng_split.integers(*pr.RANGE[la])), int(rng_split.integers(*pr.RANGE[lb]))
        if (min(a, b), max(a, b)) not in held_set:
            prs.add((a, b))
    rehearsal[(la, lb)] = sorted(prs)
rehearsal_all = [q for v in rehearsal.values() for q in v]
data_rng = np.random.default_rng(1000 + T.seed)
rng = np.random.default_rng(2000 + T.seed)
n_train = len(np.memmap(Path(T.data) / "train.bin", dtype=np.uint32, mode="r"))


def draw_pair():   # = main()'s draw_pair
    if data_rng.random() < 0.05:
        return rehearsal_all[int(data_rng.integers(len(rehearsal_all)))]
    while True:
        la, lb = pr.BUCKETS[int(data_rng.integers(len(pr.BUCKETS)))]
        if data_rng.random() < 0.5:
            la, lb = lb, la
        a, b = int(data_rng.integers(*pr.RANGE[la])), int(data_rng.integers(*pr.RANGE[lb]))
        if (min(a, b), max(a, b)) not in held_set:
            return a, b


rows = []   # per row: (arithmetic tokens L, operand positions)
for _ in range(A.batches):
    for _r in range(T.bs):   # = main()'s batch(): k lines, then the text offset
        parts = []
        for _l in range(T.arith_lines):
            a, b = draw_pair()
            parts.append(line(a, b, int(rng.integers(len(pr.FORMATS)))))
        L = sum(len(p[0]) for p in parts)
        n = T.seq - L
        assert n > 0, (L, T.seq)
        int(rng.integers(0, n_train - n))   # the text offset (consumes the same draw)
        sites, off = [], n
        for ids, ia, ib in parts:
            sites += [off + ia, off + ib]
            off += len(ids)
        assert off == T.seq
        rows.append((L, sites))
Ls = np.array([L for L, _ in rows])
res = {"arith_lines": T.arith_lines, "seed": T.seed, "rows": len(rows), "seq": T.seq,
       "arith_tokens_per_row": float(Ls.mean()), "arith_tokens_per_line": float(Ls.mean() / T.arith_lines),
       "share": float(Ls.mean() / T.seq), "share_min": float(Ls.min() / T.seq), "share_max": float(Ls.max() / T.seq),
       "dry_rows_checked": 0}
if A.dry:
    txt = Path(A.dry).read_text()
    got = {int(m.group(1)): (int(m.group(2)), int(m.group(3)), [int(v) for v in m.group(4).split(",") if v.strip()])
           for m in re.finditer(r"^train row (\d+) \(len (\d+), (\d+) prior sites at \[([0-9, ]*)\]\)", txt, re.M)}
    if sorted(got) != [0, 1, 2]:
        print(f"ARITH_SHARE FAIL: the dry log has train rows {sorted(got)} (want 0, 1, 2)")
        sys.exit(3)
    for r in (0, 1, 2):
        ln, ns, shown = got[r]
        want = rows[r][1]
        if not (ln == T.seq and ns == 2 * T.arith_lines == len(want) and shown == want[:12]):
            print(f"ARITH_SHARE FAIL: row {r}: dry log (len {ln}, {ns} sites, {shown}) != replay "
                  f"(len {T.seq}, {len(want)} sites, {want[:12]})")
            sys.exit(3)
    res["dry_rows_checked"] = 3
print("ARITH_SHARE " + json.dumps(res), flush=True)
print(f"arith share: {T.arith_lines} line(s) per row, {res['arith_tokens_per_row']:.2f} arithmetic tokens per row "
      f"({res['arith_tokens_per_line']:.2f} per line) = {100 * res['share']:.2f}% of the {T.seq}-token row over the run's "
      f"first {len(rows)} rows (seed {T.seed}; range {100 * res['share_min']:.1f}-{100 * res['share_max']:.1f}%)"
      + ("; replay = the dry log's rows 0-2" if res["dry_rows_checked"] else ""), flush=True)
if A.json:
    Path(A.json).write_text(json.dumps(res, indent=1))
