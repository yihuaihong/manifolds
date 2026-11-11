"""Download the models for the cross-model / cross-trajectory atlas (atlas v2) into models/xmodel/<tag>/ with a
.complete marker each. Non-gated models only (Llama / Gemma need an HF token). Order: small first.
    python download_xmodels.py"""
from pathlib import Path

from huggingface_hub import snapshot_download

X = Path("/scratch/yh6210/llmadd_torch/models/xmodel")
O1 = "allenai/OLMo-2-0425-1B"
P = "EleutherAI/pythia-6.9b"
MODELS = [  # (tag, repo, revision)
    *[(f"olmo2-1b-{t}", O1, r) for t, r in (
        ("0B", "stage1-step0-tokens0B"), ("1B", "stage1-step300-tokens1B"), ("21B", "stage1-step10000-tokens21B"),
        ("42B", "stage1-step20000-tokens42B"), ("84B", "stage1-step40000-tokens84B"), ("210B", "stage1-step100000-tokens210B"),
        ("630B", "stage1-step300000-tokens630B"), ("4001B", "stage1-step1907359-tokens4001B"), ("main", "main"))],
    ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B", "main"),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B", "main"),
    ("qwen3-8b-base", "Qwen/Qwen3-8B-Base", "main"),
    ("olmo3-7b", "allenai/Olmo-3-1025-7B", "main"),
    ("mistral-7b", "mistralai/Mistral-7B-v0.1", "main"),
    # Pythia-6.9B: 2.1M tokens per step
    *[(f"pythia-6.9b-{t}", P, r) for t, r in (("4B", "step2000"), ("21B", "step10000"), ("84B", "step40000"),
                                               ("210B", "step100000"), ("300B", "step143000"))],
]

for tag, repo, rev in MODELS:
    d = X / tag
    if (d / ".complete").exists():
        continue
    snapshot_download(repo, revision=rev, local_dir=d,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json", "*.tiktoken"])
    if not list(d.glob("*.safetensors")):   # old revisions (Pythia steps) only have pytorch_model*.bin
        snapshot_download(repo, revision=rev, local_dir=d, allow_patterns=["pytorch_model*.bin", "pytorch_model*.json"])
        if not list(d.glob("pytorch_model*.bin")):
            print(tag, "has no weights -- not marked complete", flush=True)
            continue
    (d / ".complete").write_text(f"{repo}@{rev}\n")
    print("done", tag, flush=True)
