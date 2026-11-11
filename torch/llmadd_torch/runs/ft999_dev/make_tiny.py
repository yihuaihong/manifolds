"""A tiny random OLMo-2 (17 layers, d 64) with the real OLMo-2-7B tokenizer, for CPU regression / smoke tests."""
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Olmo2Config, Olmo2ForCausalLM

SRC = "/scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
DST = Path(sys.argv[1])
torch.manual_seed(0)
big = AutoConfig.from_pretrained(SRC)
cfg = Olmo2Config(vocab_size=big.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=17,
                  num_attention_heads=4, num_key_value_heads=4, pad_token_id=big.pad_token_id, bos_token_id=big.bos_token_id,
                  eos_token_id=big.eos_token_id, max_position_embeddings=512)
m = Olmo2ForCausalLM(cfg)
m.save_pretrained(DST, safe_serialization=True)
AutoTokenizer.from_pretrained(SRC).save_pretrained(DST)
print("saved", DST, "vocab", cfg.vocab_size, "params", sum(p.numel() for p in m.parameters()))
