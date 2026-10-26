"""Candidate shapes for the two new rungs (head dim 64, d_ff = 4d, width and depth strictly between the neighbours):
exact parameter counts of the trainer's model (pretrain_helix.build_model: Olmo2ForCausalLM, tied 100,278-row embedding)
built on the meta device, N_nonemb = n_params - V * d, and the log distance to the geometric midpoint of the neighbours'
N_nonemb (53M 14,165,376 / 190M 113,283,840 / 505M 402,752,512)."""
import math
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

V = 100278
REF = {"53M": (384, 6, 14165376), "190M": (768, 12, 113283840), "505M": (1024, 24, 402752512), "1B": (1280, 36, 943904000)}


def count(d, L):
    cfg = Olmo2Config(vocab_size=V, hidden_size=d, intermediate_size=4 * d, num_hidden_layers=L, num_attention_heads=d // 64,
                      num_key_value_heads=d // 64, max_position_embeddings=512, tie_word_embeddings=True)
    with torch.device("meta"):
        m = Olmo2ForCausalLM(cfg)
    return sum(p.numel() for p in m.parameters())


for name, (d, L, nne) in REF.items():
    n = count(d, L)
    assert n - V * d == nne, (name, n - V * d, nne)
    print(f"check {name}: d{d} x {L}: n_params {n:,} N_nonemb {n - V * d:,} (= the ladder's value)")
for lo, hi in (("53M", "190M"), ("190M", "505M")):
    (d0, L0, n0), (d1, L1, n1) = REF[lo], REF[hi]
    mid = math.sqrt(n0 * n1)
    print(f"\nbetween {lo} and {hi}: geometric midpoint N_nonemb {mid / 1e6:.2f}M")
    rows = []
    for d in range(d0 + 64, d1, 64):
        for L in range(L0 + 1, L1):
            n = count(d, L)
            nne = n - V * d
            rows.append((abs(math.log(nne / mid)), d, L, n, nne))
    for dist, d, L, n, nne in sorted(rows)[:8]:
        print(f"  d{d} x {L} ({d // 64} heads, d_ff {4 * d}): n_params {n:,} N_nonemb {nne:,} ({nne / 1e6:.2f}M) "
              f"log distance {math.log(nne / mid):+.3f}; d/L {d / L:.1f}")
