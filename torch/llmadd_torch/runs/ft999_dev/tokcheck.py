"""Tokenizer facts for ft999: number tokens 0..999, prompt layout for 0..999 operands, answer tokens for sums 0..1998."""
import sys
from transformers import AutoTokenizer

M = "/scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
tok = AutoTokenizer.from_pretrained(M)
print("bos", tok.bos_token, tok.bos_token_id, "eos", tok.eos_token, tok.eos_token_id, "pad", tok.pad_token, tok.pad_token_id)
print("padding_side", tok.padding_side)
print("add_special_tokens example:", tok("Q: 23 + 45 = ")["input_ids"], tok.convert_ids_to_tokens(tok("Q: 23 + 45 = ")["input_ids"]))
# 1. every n in 0..999 is a single token
num = []
for n in range(1000):
    e = tok(str(n), add_special_tokens=False)["input_ids"]
    assert len(e) == 1, (n, e)
    num.append(e[0])
print("0..999 single tokens: OK; distinct:", len(set(num)))
# 2. prompt "Q: a + b = " for all a, b in 0..999 : token layout
pre = tok("Q: ")["input_ids"]
for a in list(range(0, 1000, 7)) + [0, 5, 9, 10, 99, 100, 999]:
    for b in (0, 3, 9, 10, 57, 99, 100, 512, 999):
        ids = tok(f"Q: {a} + {b} = ")["input_ids"]
        t = tok.convert_ids_to_tokens(ids)
        assert len(ids) == 9, (a, b, t)
        assert ids[3] == num[a] and ids[6] == num[b], (a, b, t)
        assert t[4] == "Ġ+" and t[5] == "Ġ" and t[7] == "Ġ=" and t[8] == "Ġ", t
for x in range(1000):
    ids = tok(f"Q: {x}")["input_ids"]
    assert len(ids) == 4 and ids[3] == num[x], (x, ids)
print("prompt layout OK: 9 tokens, A at 3, B at 6; 'Q: x' 4 tokens with x at 3 for all x in 0..999")
# 3. answers: s < 1000 -> [num[s]]; s >= 1000 -> [num[s // 10], num[s % 10]]; then '\n' separate
nl_ids = tok("\n", add_special_tokens=False)["input_ids"]
print("newline ids", nl_ids, tok.convert_ids_to_tokens(nl_ids))
assert len(nl_ids) == 1
NL = nl_ids[0]
for s in range(1999):
    exp = [num[s]] if s < 1000 else [num[s // 10], num[s % 10]]
    a = min(s, 999); b = s - a
    p = tok(f"Q: {a} + {b} = ")["input_ids"]
    full = tok(f"Q: {a} + {b} = {s}\n")["input_ids"]
    assert full == p + exp + [NL], (s, tok.convert_ids_to_tokens(full))
print("answers OK: s<1000 one token, s>=1000 = [s//10][s%10], then newline", NL)
for s in (0, 7, 99, 999, 1000, 1009, 1234, 1998):
    a = min(s, 999); b = s - a
    print(s, tok.convert_ids_to_tokens(tok(f"Q: {a} + {b} = {s}\n")["input_ids"]))
