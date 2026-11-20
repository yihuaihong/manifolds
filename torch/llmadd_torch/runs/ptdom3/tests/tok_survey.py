"""ptdom3 build (2026-11-17): token survey for the new domains (CPU, read only).
(1) nonce candidates: OLMo-2 single tokens ' xyz' (lowercase consonant-vowel-consonant, the S4_NAMES / DIM_NAMES rule of
    2026-10-17), round-trip single token, not one of the 51 s4 / dim names, counted in train.bin; sorted rarest first.
(2) notes: every spelling of the 12 pitch classes, the number words one..eleven, ' semitone(s)': single token? count?
(3) chess: ' a1'..' h8' and ' A1'..' H8': single token? count?
Writes tok_survey.json next to this file and prints a summary."""
import json, re, sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

TOK = "/scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
DATA = Path("/scratch/yh6210/llmadd_torch/pretrain_data")
src = Path("/scratch/yh6210/llmadd-burst-ptdom3/tools/pretrain_range.py").read_text()
old = set(re.findall(r'"([a-z]{3})": \d+', src[src.index("S4_NAMES = {"):src.index("DOM_ARGS =")]))
assert len(old) == 51, len(old)
tok = AutoTokenizer.from_pretrained(TOK)
enc = lambda s: tok.encode(s, add_special_tokens=False)
C, V = "bcdfghjklmnpqrstvwxyz", "aeiou"
pat = re.compile(f"^ [{C}][{V}][{C}]$")
cand = {}
for i in range(len(tok)):
    s = tok.decode([i])
    if pat.match(s) and enc(s) == [i] and s[1:] not in old:
        cand[s[1:]] = i
notes = {}
for nm in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B", "Cb", "E#", "Fb", "B#"]:
    notes[nm] = enc(" " + nm)
words = {w: enc(" " + w) for w in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                                   "semitone", "semitones", "plus", "minus", "with", "and", "xor", "then", "is", "file",
                                   "files", "rank", "ranks", "right", "left", "up", "down"]}
chess = {f"{f}{r}": enc(f" {f}{r}") for f in "abcdefgh" for r in range(1, 9)}
chessU = {f"{f}{r}": enc(f" {f}{r}") for f in "ABCDEFGH" for r in range(1, 9)}
ids = sorted(set(cand.values()) | {t for v in list(notes.values()) + list(words.values()) + list(chess.values())
                                   + list(chessU.values()) for t in v})
train = np.memmap(DATA / "train.bin", dtype=np.uint32, mode="r")
cnt = np.zeros(len(tok), dtype=np.int64)
for i in range(0, len(train), 50_000_000):
    cnt += np.bincount(np.asarray(train[i:i + 50_000_000]), minlength=len(tok))[:len(tok)]
# the 51 old names re-counted (must equal the trainer's recorded values: the counting is the trainer's)
oldc = {nm: int(cnt[enc(" " + nm)[0]]) for nm in old}
recorded = {k: int(v) for k, v in re.findall(r'"([a-z]{3})": (\d+)', src[src.index("S4_NAMES = {"):src.index("DOM_ARGS =")])}
assert oldc == recorded, {k: (oldc[k], recorded[k]) for k in old if oldc[k] != recorded[k]}
rank = sorted(cand, key=lambda s: (int(cnt[cand[s]]), s))
out = {"n_train_tokens": int(len(train)), "n_cvc_single_new": len(cand),
       "nonce_ranked": [[s, cand[s], int(cnt[cand[s]])] for s in rank],
       "notes": {k: [v, [int(cnt[t]) for t in v]] for k, v in notes.items()},
       "words": {k: [v, [int(cnt[t]) for t in v]] for k, v in words.items()},
       "chess_lower": {k: [v, [int(cnt[t]) for t in v]] for k, v in chess.items()},
       "chess_upper": {k: [v, [int(cnt[t]) for t in v]] for k, v in chessU.items()}}
Path(__file__).with_name("tok_survey.json").write_text(json.dumps(out, indent=0))
print("old 51 nonce counts re-derived = recorded: OK")
print(f"new CVC single tokens: {len(cand)}; rarest 200 with counts:")
print(" ".join(f"{s}:{int(cnt[cand[s]])}" for s in rank[:200]))
print("count at rank 120 / 160 / 200:", [int(cnt[cand[rank[k]]]) for k in (119, 159, 199) if k < len(rank)])
print("notes:", {k: (len(v), [int(cnt[t]) for t in v]) for k, v in notes.items()})
print("words:", {k: len(v) for k, v in words.items()})
print("chess lower single-token:", sum(len(v) == 1 for v in chess.values()), "/ 64;",
      "upper:", sum(len(v) == 1 for v in chessU.values()), "/ 64")
print("chess lower multi:", [k for k, v in chess.items() if len(v) != 1])
print("chess lower counts:", {k: int(cnt[v[0]]) for k, v in chess.items() if len(v) == 1})
