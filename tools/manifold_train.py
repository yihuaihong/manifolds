"""Manifold-regularised fine-tuning of an OLMo-2 checkpoint on two-digit addition.

Question: if fine-tuning also pulls the operand representations onto an ideal number manifold, does the
model generalise better (held-out pairs, three-term, three-digit) than with the task loss alone?

Loss:  CE(answer token) + lam * L_geo
L_geo: the GEOMETRY of the operand representation, not a readout of it. Every step the 90 prefixes "Q: x"
       (x = 10..99) are run through the model; at each layer in `--layers` the states of x form H [90, d],
       and L_geo = mean_L (1 - CKA(H, F_G)), with F_G [90, k] the ideal coordinates of 10..99 and linear
       CKA on centred Gram matrices. CKA is invariant only to rotations and isotropic scaling, so it is
       satisfied only if the representation itself has the shape of the ideal manifold.
       (v1 used ||(h - mu) R - F||^2 with a readout R fitted in 4096-d; 90 points can be mapped linearly
       onto ANY target there - shuffled included - so v1 constrained nothing. Found on s1-10k, 2026-09-27.)
Geometries G: none (task only) | helix | helix_shuf | digit | digit_shuf. "_shuf" uses the same basis on a
       fixed permutation of 10..99: same dimensionality and strength, wrong numbers.
       Learned targets (TEACHER): main | main_shuf | self - the target Gram is a MODEL's own states of the same
       numbers at the same layer (or its embedding rows with --geo_site emb): --main_model's (a later, more
       mature checkpoint) or this checkpoint's own before training ("self": an anchor that only resists drift).
       "3" variants (helix3, digit3, main3, ...) run over 10..999, sampled per step.
       Spectrum vs separability (Fu et al., arXiv 2604.20817: a Fourier spike at 1/T does not make n mod T
       linearly separable):
       helix_nonsep  the helix plus fixed noise in its T = 2, 5, 10 columns that has no power at any helix
                     frequency: same spike at 1/T, but n mod T is no longer linearly separable in the target
                     (B = 3: kappa mod 2/5/10 1.0 -> ~0, power at 1/T unchanged, CKA to the helix 0.51);
       helix_coarse  only the T = 100 pair and x/100 of the helix (no spike at 1/2, 1/5, 1/10, no separability):
                     the part of helix_nonsep that is still clean, i.e. its control;
       sep_ce        no shape target: leave-one-out nearest-class-mean cross-entropy of n mod 2, 5, 10 and the
                     tens digit on the same states (separability only; parameter-free, so unlike a trainable
                     linear head on 90 points in d dimensions it cannot be satisfied without moving the states).
Tasks (--task add|sub) and splits (--split random | holdout_operand: 15 operand values never seen in training |
       carry: train without a units carry/borrow, test only with one); --save_preds keeps every eval answer.
Methods (--mode): lora (r=8, all linear layers) | reft (low-rank edit of the operand states at the
       manifold layers, base frozen) | full (all weights, fp32 master + AdamW; needs ~120 GB, e.g. one H200).

Every (geometry, seed) run starts from the same base weights; for a given seed all geometries share the
data order and the initialisation of the trainable parameters, so comparisons are paired. One JSON per
run under --out; existing runs are skipped (restartable).

    python tools/manifold_train.py --model <dir> --out <dir> --mode lora --geoms none,helix,helix_shuf,digit,digit_shuf --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

import numlib as nl

GEOMS = ("none", "helix", "helix_shuf", "digit", "digit_shuf", "helix_nonsep", "helix_coarse", "sep_ce")
MULT_GEOMS = ("log", "factor", "factor_shuf", "helix_factor")   # coordinates in which multiplication is a translation
NOISY_GEOMS = ("helix_n25", "helix_n50", "helix_n100", "helix_n200")   # the helix with fixed per-number noise (std k/100)
DIGIT_GEOMS = ("digit_fourier", "digit_fourier_shuf", "fone", "fone_shuf", "helix_digitf")   # lookup-friendly per-place codes / FoNE circles
# three-digit geometries over 10..999 (sampled per step, see --geo_sample)
GEOMS3 = ("helix3", "helix3_shuf", "digit3", "digit3_shuf")
# learned targets: name -> (source model, shuffled, 2 = 10..99 / 3 = 10..999)
TEACHER = {"main": ("main", False, 2), "main_shuf": ("main", True, 2), "self": ("self", False, 2),
           "main3": ("main", False, 3), "main3_shuf": ("main", True, 3), "self3": ("self", False, 3)}
STAGED = ("sep_then_helix", "helix_gated")   # sep_ce then helix (kappa gate); helix while train-batch accuracy (EMA) < --gate_thr
ALL_GEOMS = GEOMS + GEOMS3 + tuple(TEACHER) + STAGED + MULT_GEOMS + NOISY_GEOMS + DIGIT_GEOMS


def geom_range(G: str, lo3: int = 10) -> np.ndarray:
    """Numbers of geometry G; lo3 = 0 (--add3) extends the three-digit geometries to 0..999."""
    return np.arange(lo3, 1000) if G.startswith(("helix3", "digit3")) else np.arange(10, 100)


def is_three(G: str) -> bool:
    return G in GEOMS3 or (G in TEACHER and TEACHER[G][2] == 3)


def two_prompt(a, b, op="+"): return f"Q: {a} {op} {b} = "
def three_prompt(a, b, c, op="+"): return f"Q: {a} {op} {b} {op} {c} = "
def terse_prompt(a, b, op="+"): return f"{nl.PREFIX}{a}{op}{b}="
def gpt3_prompt(a, b, word): return f"Q: What is {a} {word} {b}?\n\nA: "   # EleutherAI/arithmetic context + the answer's space


# natural-language templates (test pairs only); each template is its own eval set (prompts must share one length)
WORD = {
    "add": ("Tom has {a} apples and buys {b} more. How many apples does he have now? Answer: ",
            "A shelf holds {a} books and {b} more books are added. How many books are on the shelf? Answer: ",
            "There are {a} people on a bus and {b} more get on. How many people are on the bus? Answer: "),
    "sub": ("Tom has {a} apples and gives away {b}. How many apples does he have left? Answer: ",
            "A shelf holds {a} books and {b} books are removed. How many books are on the shelf? Answer: ",
            "There are {a} people on a bus and {b} get off. How many people are on the bus? Answer: "),
    "mul": ("There are {a} boxes with {b} pencils in each box. How many pencils are there? Answer: ",
            "A farmer plants {a} rows of {b} trees. How many trees does he plant? Answer: ",
            "Each of {a} students reads {b} pages. How many pages do they read in total? Answer: "),
    "max": ("Tom has {a} apples and Anna has {b} apples. How many apples does the one with more apples have? Answer: ",
            "Which number is larger, {a} or {b}? Answer: ",
            "Shelf A holds {a} books and shelf B holds {b} books. How many books are on the fuller shelf? Answer: "),
}
FORMATS = {   # training surface formats for --mixed_formats (none of them is the GPT-3 or a word-problem format)
    "add": ("Q: {a} + {b} = ", "Compute {a} + {b}. Result: ", "The sum of {a} and {b} is ", "{a} plus {b} equals "),
    "sub": ("Q: {a} - {b} = ", "Compute {a} - {b}. Result: ", "The difference between {a} and {b} is ", "{a} minus {b} equals "),
}
GPT3_FILES = {"add": ("two_digit_addition", "plus"), "sub": ("two_digit_subtraction", "minus"),
              "mul": ("two_digit_multiplication", "times")}
OPS = {"add": "+", "sub": "-", "mul": "*", "mod": "#", "max": "max", "mixed": "+"}   # mixed: + is the probe / slot op


def answers(task: str, P: np.ndarray) -> np.ndarray:
    """Answer of each operand row (two or more operands for add / sub)."""
    if task == "add":
        return P.sum(1)
    if task == "sub":
        return P[:, 0] - P[:, 1:].sum(1)
    if task == "mul":
        return P[:, 0] * P[:, 1]
    if task == "mod":
        return (P[:, 0] + P[:, 1]) % 100
    return P.max(1)


def gpt3_items(path: str, task: str) -> np.ndarray:
    """(a, b) rows of the GPT-3 two-digit arithmetic file for `task` with operands 10..99 and answers 0..999."""
    import re
    fname, word = GPT3_FILES[task]
    rows = []
    for line in open(f"{path}/data/{fname}.jsonl"):
        m = re.search(rf"What is (\d+) {word} (\d+)\?", json.loads(line)["context"])
        a, b = int(m.group(1)), int(m.group(2))
        if 10 <= a <= 99 and 10 <= b <= 99 and 0 <= answers(task, np.array([[a, b]]))[0] <= 999:
            rows.append((a, b))
    return np.array(rows)


NONSEP_B = 3.0


def nonsep_noise(B: float = NONSEP_B) -> np.ndarray:
    """[90, 5] fixed noise for the T = 2, 5, 10 columns of the helix of 10..99 (rows x - 10): projected off the
    constant and every helix column, so it has no power at 1/2, 1/5, 1/10 (exact DFT bins of 90 points)."""
    Hb = np.concatenate([np.ones((90, 1)), nl.features(np.arange(10, 100))[0]], 1)
    g = np.random.default_rng(77).standard_normal((90, 5))
    Q = np.linalg.qr(Hb)[0]
    g = g - Q @ (Q.T @ g)
    return B * g / g.std(0)


def sep_ce_loss(H: torch.Tensor, xs: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    """Mean over labels (n mod 2, 5, 10, tens digit) of the leave-one-out nearest-class-mean cross-entropy:
    logit_c(i) = -|h_i - mean_{j in c, j != i} h_j|^2 / (scale * mean_i |h_i - mean h|^2)."""
    H = H.float()
    H = H - H.mean(0)
    tau = scale * (H ** 2).sum(1).mean()
    loss = 0.0
    labels = (xs % 2, xs % 5, xs % 10, (xs // 10) % 10)
    for y in labels:
        C = int(y.max()) + 1
        Y = Fn.one_hot(y, C).float()                       # [n, C]
        S, cnt = Y.T @ H, Y.sum(0)                          # class sums, counts
        mu = (S[None] - Y[:, :, None] * H[:, None]) / (cnt[None] - Y).clamp_min(1)[:, :, None]   # LOO means [n, C, d]
        d2 = ((H[:, None] - mu) ** 2).sum(-1)
        logits = (-d2 / tau).masked_fill((cnt == 0)[None], float("-inf"))   # classes absent from xs (tens digit 0)
        loss = loss + Fn.cross_entropy(logits, y)
    return loss / len(labels)


def geom_features(G: str, x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if G.endswith("_shuf"):
        x = perm[x]; G = G[: -len("_shuf")]
    if G == "helix":
        return nl.features(x)[0]
    if G == "helix_coarse":
        return nl.features(x, periods=(100,))[0]
    if G == "helix_nonsep":
        F = nl.features(x)[0]
        F[:, :5] += nonsep_noise()[x - 10]
        return F
    if G == "digit":
        return np.concatenate([np.eye(10)[x // 10], np.eye(10)[x % 10]], 1)
    if G.startswith("helix_n"):
        F = nl.features(np.arange(1000))[0]
        F = (F - F[10:100].mean(0)) / F[10:100].std(0)
        F = F + int(G[len("helix_n"):]) / 100 * np.random.default_rng(99).normal(size=F.shape)
        return F[x]
    if G == "digit_fourier":
        code = lambda d: np.stack([np.cos(2 * np.pi * d / 10), np.sin(2 * np.pi * d / 10), np.cos(2 * np.pi * d / 5),
                                   np.sin(2 * np.pi * d / 5), (-1.0) ** d], 1)
        return np.concatenate([code(x // 10), code(x % 10)], 1)
    if G == "helix_digitf":
        return np.concatenate([nl.features(x)[0], geom_features("digit_fourier", x, perm)], 1)
    if G == "fone":
        return np.stack([np.cos(2 * np.pi * x / 10), np.sin(2 * np.pi * x / 10), np.cos(2 * np.pi * x / 100),
                         np.sin(2 * np.pi * x / 100)], 1)
    if G == "log":
        return np.log(x)[:, None]
    if G == "factor":
        val = lambda n, q: next(k for k in range(8) if n % q ** (k + 1))   # exponent of the prime q in n
        V = np.array([[val(int(n), q) for q in (2, 3, 5, 7)] for n in x], dtype=float)
        return np.concatenate([np.log(x)[:, None], V, np.eye(10)[x % 10]], 1)
    if G == "helix_factor":
        return np.concatenate([nl.features(x)[0], geom_features("factor", x, perm)], 1)
    if G == "helix3":
        return np.concatenate([nl.features(x, periods=(2, 5, 10, 100, 1000), linear=False)[0], (x / 1000.0)[:, None]], 1)
    if G == "digit3":
        return np.concatenate([np.eye(10)[x // 100], np.eye(10)[(x // 10) % 10], np.eye(10)[x % 10]], 1)
    raise ValueError(G)


def cka(H: torch.Tensor, K_F: torch.Tensor) -> torch.Tensor:
    """Linear CKA between states H [n, d] and a centred target Gram K_F [n, n] (differentiable in H)."""
    H = H.float() - H.float().mean(0, keepdim=True)
    K_H = H @ H.T
    return (K_H * K_F).sum() / (K_H.norm() * K_F.norm() + 1e-12)


def target_grams(perm: np.ndarray, dev) -> dict:
    """Centred Gram matrices of the ideal coordinates of 10..99 for every two-digit geometry."""
    out = {}
    for G in ("helix", "helix_shuf", "digit", "digit_shuf", "helix_nonsep", "helix_coarse") + MULT_GEOMS + NOISY_GEOMS + DIGIT_GEOMS:
        F = geom_features(G, np.arange(10, 100), perm)
        F = F[:, F.std(0) > 1e-9]
        F = (F - F.mean(0)) / F.std(0)
        Ft = torch.tensor(F, dtype=torch.float32, device=dev)
        out[G] = Ft @ Ft.T
    return out


def target_features(G: str, perm: np.ndarray, dev, lo3: int = 10) -> torch.Tensor:
    """Standardised ideal coordinates, row x for x in 0..999 (rows outside geom_range(G, lo3) are unused)."""
    xs = geom_range(G, lo3)
    F = geom_features(G, xs, perm)
    F = F[:, F.std(0) > 1e-9]
    F = (F - F.mean(0)) / F.std(0)
    out = torch.zeros(1000, F.shape[1], device=dev)
    out[torch.tensor(xs, device=dev)] = torch.tensor(F, dtype=torch.float32, device=dev)
    return out


def gram_of(F_rows: torch.Tensor) -> torch.Tensor:
    Fc = F_rows - F_rows.mean(0, keepdim=True)
    return Fc @ Fc.T


@torch.no_grad()
def transplant_number_embeddings(emb_mod, num_all: torch.Tensor, main_path: str, how: str, dev) -> dict:
    """Overwrite the input-embedding rows of the number tokens 0..999 with main's rows mapped into this
    checkpoint's embedding space by scaled orthogonal Procrustes fitted on all NON-number tokens.
    how = main | main_shuf (rows permuted within 10..99 and within 100..999). Returns fit diagnostics.
    how = main_shape | main_shape_shuf: fit the Procrustes map on the NUMBER rows themselves, i.e. keep this
    checkpoint's location/scale of the number cloud and adopt main's relative geometry (the shuffled variant
    permutes main's rows before the fit: same procedure, wrong correspondence)."""
    from safetensors import safe_open
    index = Path(main_path) / "model.safetensors.index.json"
    shard = (json.loads(index.read_text())["weight_map"]["model.embed_tokens.weight"] if index.exists() else "model.safetensors")
    with safe_open(str(Path(main_path) / shard), framework="pt") as f:
        Em = f.get_tensor("model.embed_tokens.weight").float().to(dev)
    Ec = emb_mod.weight.detach().float()
    rng = np.random.default_rng(0)
    if how.startswith("main_shape"):
        Mn = Em[num_all]
        if how == "main_shape_shuf":
            p = np.arange(1000); p[10:100] = 10 + rng.permutation(90); p[100:] = 100 + rng.permutation(900)
            Mn = Mn[torch.tensor(p, device=dev)]
        Cn = Ec[num_all]
        A, B = Mn - Mn.mean(0), Cn - Cn.mean(0)
        U, S, Vt = torch.linalg.svd(A.T @ B, full_matrices=False)
        R = U @ Vt
        scale = S.sum() / (A ** 2).sum()
        rows = A @ R * scale + Cn.mean(0)
        r2 = 1 - ((rows - Cn) ** 2).sum() / (B ** 2).sum()
        emb_mod.weight[num_all] = rows.to(emb_mod.weight.dtype)
        return {"shape_fit_r2_number_rows": float(r2), "scale": float(scale),
                "mean_row_change_norm": float((rows - Cn).norm(dim=1).mean()), "mean_row_norm": float(Cn.norm(dim=1).mean())}
    V = Ec.shape[0]
    keep = torch.ones(V, dtype=torch.bool, device=dev); keep[num_all] = False
    perm_ids = torch.randperm(int(keep.sum()), generator=torch.Generator().manual_seed(0)).to(dev)
    others = torch.nonzero(keep).squeeze(1)[perm_ids]
    fit, held = others[: len(others) * 9 // 10], others[len(others) * 9 // 10:]
    A, B = Em[fit] - Em[fit].mean(0), Ec[fit] - Ec[fit].mean(0)
    U, S, Vt = torch.linalg.svd(A.T @ B)
    R = U @ Vt
    scale = S.sum() / (A ** 2).sum()
    mapped = lambda X: (X - Em[fit].mean(0)) @ R * scale + Ec[fit].mean(0)
    r2 = 1 - ((mapped(Em[held]) - Ec[held]) ** 2).sum() / ((Ec[held] - Ec[held].mean(0)) ** 2).sum()
    rows = mapped(Em[num_all])
    if how == "main_shuf":
        p = np.arange(1000); p[10:100] = 10 + rng.permutation(90); p[100:] = 100 + rng.permutation(900)
        rows = rows[torch.tensor(p, device=dev)]
    before = Ec[num_all].clone()
    emb_mod.weight[num_all] = rows.to(emb_mod.weight.dtype)
    return {"procrustes_heldout_r2_non_number_tokens": float(r2), "scale": float(scale),
            "mean_row_change_norm": float((rows - before).norm(dim=1).mean()), "mean_row_norm": float(before.norm(dim=1).mean())}


def decoder_layers(model) -> list:
    return [m for m in model.modules() if type(m).__name__ == "Olmo2DecoderLayer"]


@torch.no_grad()
def number_states(model, layers: list, prefix_all: torch.Tensor, pos: int, bs: int) -> dict:
    """{L: [1000, d] float32}: the state at `pos` after decoder layer L for the prefixes "Q: x" (rows 10..999)."""
    dls = decoder_layers(model)
    got: dict = {}
    hooks = [dls[L].register_forward_hook(lambda _m, _i, o, L=L: got.__setitem__(L, nl._hidden(o)[:, pos, :].float()))
             for L in layers]
    out = {L: torch.zeros(1000, model.config.hidden_size, device=prefix_all.device) for L in layers}
    try:
        for i in range(10, 1000, bs):
            model(input_ids=prefix_all[i:i + bs], logits_to_keep=1)
            for L in layers:
                out[L][i:i + bs] = got[L]
    finally:
        for h in hooks:
            h.remove()
    return out


class Reft(torch.nn.Module):
    """Low-rank additive edit of selected token states at one layer: h + (h Wd^T + b) Wu^T (Wu starts at 0)."""

    def __init__(self, d: int, r: int):
        super().__init__()
        self.down = torch.nn.Linear(d, r)
        self.up = torch.nn.Linear(r, d, bias=False)

    def reset(self) -> None:
        torch.nn.init.normal_(self.down.weight, std=self.down.in_features ** -0.5)
        torch.nn.init.zeros_(self.down.bias)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(self.down(h.float())).to(h.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("lora", "reft", "full"), required=True)
    ap.add_argument("--geoms", default=",".join(GEOMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--train_pairs", type=int, default=1500)
    ap.add_argument("--fixed_test", action="store_true", help="split random: always the default test set (pairs[1500:2500]); extra train pairs come from pairs[2500:]")
    ap.add_argument("--test_pairs", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--geo_site", choices=("layers", "emb"), default="layers",
                    help="layers: CKA on the A-slot states at --layers; emb: CKA on the input-embedding rows of 10..99 (needs --emb_delta unless mode=full)")
    ap.add_argument("--emb_delta", action="store_true", help="make the input-embedding rows of the number tokens 0..999 trainable (additive delta)")
    ap.add_argument("--emb_init", choices=("orig", "main", "main_shuf", "main_shape", "main_shape_shuf"), default="orig",
                    help="replace the number-token rows 0..999 of the input embedding before training: main = Procrustes-aligned rows of --main_model, main_shuf = same rows shuffled among the numbers")
    ap.add_argument("--main_model", default=None)
    ap.add_argument("--sep_lam", type=float, default=1.0, help="weight of the sep_ce loss (replaces --lam for that arm)")
    ap.add_argument("--geo_sample", type=int, default=180, help="numbers per step for the 10..999 geometries (helix3/digit3)")
    ap.add_argument("--task", choices=("add", "sub", "mul", "mod", "max", "mixed"), default="add",
                    help="sub: a - b with a >= b; mul: a * b <= 999; mod: (a + b) mod 100 written with a new operator #; max: the larger operand")
    ap.add_argument("--gpt3_arith", default=None, help="EleutherAI/arithmetic data dir: also evaluate on its two-digit items")
    ap.add_argument("--curve_every", type=int, default=0, help="test accuracy every K optimizer steps (0 = off)")
    ap.add_argument("--general_eval", default=None, help="dir with HellaSwag / ARC / LAMBADA files: general-ability suite")
    ap.add_argument("--general_n", type=int, default=500)
    ap.add_argument("--gate_thr", type=float, default=0.5, help="helix_gated: prior on while the EMA of training-batch accuracy is below this")
    ap.add_argument("--lm_val", default=None, help="uint32 token file (OLMo-2 tokenizer): report LM loss on 32 x 512 tokens")
    ap.add_argument("--save_states_every", type=int, default=0, help="snapshot the number states (last layer of --layers) every K steps")
    ap.add_argument("--mixed_formats", action="store_true", help="add / sub: training batches rotate over FORMATS (GPT-3 and word formats held out)")
    ap.add_argument("--split", choices=("random", "holdout_operand", "carry"), default="random")
    ap.add_argument("--geo_numbers", choices=("all", "seen"), default="all",
                    help="two-digit geometries over all of 10..99, or only over the operand values seen in training")
    ap.add_argument("--save_preds", action="store_true", help="store every predicted answer of the eval sets (not train)")
    ap.add_argument("--device_map", default=None, help="'auto': split the model over all visible GPUs (e.g. 32B on 2 x 40 GB); inputs and losses live on the first GPU")
    ap.add_argument("--add3", action="store_true",
                    help="three-digit addition (ft999): operands 0..999, held-out test pairs disjoint as UNORDERED pairs, answer "
                         "= the exact tokens of the sum + newline (s >= 1000 is two tokens [s // 10][s % 10]), teacher-forced exact "
                         "match; the 3-digit geometries (helix3 / helix3_shuf / digit3...) and --save_states_every cover 0..999")
    ap.add_argument("--grad_check_every", type=int, default=250, help="--add3: log |grad CE| and |grad lam * L_geo| every K steps (and at step 0)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lr = args.lr or {"lora": 1e-4, "reft": 1e-3, "full": 1e-5}[args.mode]
    assert not (args.geo_site == "emb" and args.mode != "full" and not args.emb_delta), "--geo_site emb needs trainable embeddings (--emb_delta)"
    assert args.emb_init == "orig" or args.main_model, "--emb_init main* needs --main_model"
    geoms = args.geoms.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    assert all(G in ALL_GEOMS for G in geoms), geoms
    assert not any(TEACHER.get(G, ("",))[0] == "main" for G in geoms) or args.main_model, "main* targets need --main_model"
    assert args.geo_numbers == "all" or not any(is_three(G) for G in geoms), "--geo_numbers seen is for the two-digit geometries"
    man_layers = [int(v) for v in args.layers.split(",")]
    op = OPS[args.task]
    t0 = time.time()
    lo3 = 10   # first number of the three-digit geometries / prefixes (0 with --add3)
    if args.add3:
        assert args.task == "add" and args.split == "random" and args.fixed_test and not args.mixed_formats, "--add3: add, random, --fixed_test"
        assert args.geo_site == "layers" and args.geo_numbers == "all" and not args.emb_delta and args.emb_init == "orig", "--add3: prior on layer states"
        assert not args.gpt3_arith and not args.main_model and not args.general_eval and args.mode == "lora", "--add3: lora, no gpt3 / general / teacher"
        assert all(G == "none" or G in GEOMS3 for G in geoms), ("--add3 arms: none or a 0..999 geometry", geoms)
        lo3 = 0
        print(f"[add3] operands 0..999, geometries over {lo3}..999, lam {args.lam}, layers {man_layers}, geo_sample {args.geo_sample}", flush=True)

    # ---------------------------------------------------------------- tokenizer, prefixes, teacher states
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = ("cuda:0" if args.device_map else "cuda") if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    toks = tok.convert_ids_to_tokens(tok(two_prompt(23, 45, op))["input_ids"])
    assert toks[-6:] == ["23", "Ġ" + op, "Ġ", "45", "Ġ=", "Ġ"], toks
    slot_pos = {"A": len(toks) - 6, "B": len(toks) - 3}
    enc = lambda ps: nl.encode(tok, ps, dev)
    prefix_all = torch.zeros(1000, slot_pos["A"] + 1, dtype=torch.long, device=dev)
    prefix_all[lo3:] = enc([f"Q: {x}" for x in range(lo3, 1000)])      # A-slot state of x depends only on this prefix
    if args.add3:   # every "Q: x" (x = 0..999) has the same length and x at the A slot
        assert all(int(tok.decode(prefix_all[x, slot_pos["A"]])) == x for x in range(1000)), "add3: A slot of Q: x is not x"
    num_prefix = prefix_all[10:100]
    teach: dict = {}    # {"main" | "self": {L | "emb": [1000, d] float32}}, rows 0..9 unused
    if args.main_model:
        tm = AutoModelForCausalLM.from_pretrained(args.main_model, torch_dtype=torch.bfloat16).to(dev)
        tm.config.use_cache = False
        teach["main"] = number_states(tm, man_layers, prefix_all, slot_pos["A"], args.eval_bs)
        tn = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
        teach["main"]["emb"] = tm.get_input_embeddings().weight[tn].detach().float()
        del tm
        torch.cuda.empty_cache() if dev == "cuda" else None
        print(f"teacher states from {args.main_model} ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- model
    base_dtype = torch.float32 if (args.mode == "full" or dev == "cpu") else torch.bfloat16
    if args.device_map:
        # GPU 0 also holds the inputs, the embedding and the geometry prompts' activations: give it less of the model
        mem = ({i: ("30GiB" if i == 0 else "38GiB") for i in range(torch.cuda.device_count())}
               if torch.cuda.is_available() else None)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=base_dtype, device_map=args.device_map, max_memory=mem)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=base_dtype).to(dev)
    model.config.use_cache = False
    layers = decoder_layers(model)
    d = model.config.hidden_size
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (args.mode == "full" and dev == "cuda") else torch.autocast("cpu", enabled=False)
    if any(TEACHER.get(G, ("",))[0] == "self" for G in geoms):
        teach["self"] = number_states(model, man_layers, prefix_all, slot_pos["A"], args.eval_bs)
        teach["self"]["emb"] = model.get_input_embeddings().weight[num_ids[:1000]].detach().float()

    reft_mods = None
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        if args.device_map:   # 32B on 2 x 40 GB: recompute activations in backward (same gradients, less memory)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        layers = decoder_layers(model)
    elif args.mode == "reft":
        for p in model.parameters():
            p.requires_grad_(False)
        reft_mods = torch.nn.ModuleDict({str(L): Reft(d, args.rank) for L in man_layers}).to(dev)
    else:
        init_state = None   # taken after the optional embedding transplant (below)

    def trainable():
        ps = list(reft_mods.parameters()) if args.mode == "reft" else [p for p in model.parameters() if p.requires_grad]
        return ps + ([emb_delta] if emb_delta is not None else [])

    def reset(seed: int) -> None:
        torch.manual_seed(seed)
        if emb_delta is not None:
            with torch.no_grad():
                emb_delta.zero_()
        if args.mode == "lora":
            from peft.tuners.lora import LoraLayer
            for m in model.modules():
                if isinstance(m, LoraLayer):
                    m.reset_lora_parameters("default", True)
        elif args.mode == "reft":
            for m in reft_mods.values():
                m.reset()
        else:
            model.load_state_dict({k: v.to(torch.float32) for k, v in init_state.items()})

    # ---------------------------------------------------------------- hooks: capture (+ ReFT edit) at manifold layers
    store: dict = {}
    reft_on = {"v": False}

    def make_hook(L: int):
        def hook(_m, _i, o):
            h = nl._hidden(o)
            if reft_mods is not None and reft_on["v"]:
                h0, h = h, h.clone()          # read from h0, write into the copy (autograd-safe)
                for p in slot_pos.values():
                    if p < h.shape[1]:
                        h[:, p, :] = reft_mods[str(L)](h0[:, p, :])
                o = nl._with_hidden(o, h)
            store[L] = h
            return o
        return hook

    for L in man_layers:
        layers[L].register_forward_hook(make_hook(L))

    # ---------------------------------------------------------------- number-token embeddings
    emb_mod = model.get_input_embeddings()
    num_all = num_ids[:1000]
    emb_info = {}
    if args.emb_init != "orig":
        emb_info = transplant_number_embeddings(emb_mod, num_all, args.main_model, args.emb_init, dev)
        print(f"embedding init {args.emb_init}: {emb_info}")
    emb_delta = None
    if args.emb_delta:
        emb_delta = torch.nn.Parameter(torch.zeros(1000, d, device=dev))
        row_of = torch.full((emb_mod.weight.shape[0],), -1, dtype=torch.long, device=dev)
        row_of[num_all] = torch.arange(1000, device=dev)

        def add_delta(_m, inp, out):
            r = row_of[inp[0]]
            m = r >= 0
            if not m.any():
                return out
            out = out.clone()
            out[m] = out[m] + emb_delta[r[m]].to(out.dtype)
            return out
        emb_mod.register_forward_hook(add_delta)

    if args.mode == "full":
        init_state = {k: v.detach().to("cpu", torch.bfloat16).clone() for k, v in model.state_dict().items()}

    def number_embedding_rows(xs=None) -> torch.Tensor:
        xs = torch.arange(10, 100, device=dev) if xs is None else xs
        E = emb_mod.weight[num_all[xs]].float()
        return E + emb_delta[xs] if emb_delta is not None else E

    def forward(ids: torch.Tensor, reft: bool) -> torch.Tensor:
        reft_on["v"] = reft
        with autocast:
            return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float().to(dev)

    def forward_k(ids: torch.Tensor, reft: bool, k: int) -> torch.Tensor:
        """Logits of the last k positions [n, k, V] (--add3: teacher-forced multi-token answers)."""
        reft_on["v"] = reft
        with autocast:
            return model(input_ids=ids, logits_to_keep=k).logits[:, -k:].float().to(dev)

    # ---------------------------------------------------------------- data
    A3: dict = {}
    if args.add3:
        # ft999: operands uniform over 0..999. The test set (--test_pairs) is drawn first, the training set after it from
        # the same stream with every UNORDERED test pair {a, b} excluded (b + a of a test pair is never trained). Both
        # depend only on this fixed generator: the same for every arm, seed and checkpoint (a smaller --train_pairs is a prefix).
        r3 = np.random.default_rng(2026)

        def draw(n: int, banned: set, unordered: bool) -> np.ndarray:
            rows, keys = [], set()
            while len(rows) < n:
                a, b = (int(v) for v in r3.integers(0, 1000, 2))
                u, k = (min(a, b), max(a, b)), ((min(a, b), max(a, b)) if unordered else (a, b))
                if u in banned or k in keys:
                    continue
                keys.add(k); rows.append((a, b))
            return np.array(rows)
        test = draw(args.test_pairs, set(), unordered=True)
        test_keys = {(min(a, b), max(a, b)) for a, b in test.tolist()}
        train = draw(args.train_pairs, test_keys, unordered=False)
        assert not test_keys & {(min(a, b), max(a, b)) for a, b in train.tolist()}, "add3: a test pair (either order) is in training"
        assert len(train) == args.train_pairs and len(test) == args.test_pairs and train.min() >= 0 and train.max() <= 999
        nid = nl.number_token_ids(tok, 999)
        NL = tok("\n", add_special_tokens=False)["input_ids"]
        assert len(NL) == 1, NL
        NL = NL[0]

        def seq3(P: np.ndarray):
            """Teacher-forced inputs [n, len(prompt) + 2] and targets [n, 3] (-100 = not scored) of "Q: a + b = " -> "<a + b>\\n"."""
            s = P.sum(1)
            prompts = enc([two_prompt(a, b, "+") for a, b in P])
            sh = s < 1000
            tgt = np.full((len(P), 3), -100, dtype=np.int64)
            tgt[sh, 0], tgt[sh, 1] = nid[s[sh]], NL
            tgt[~sh, 0], tgt[~sh, 1], tgt[~sh, 2] = nid[s[~sh] // 10], nid[s[~sh] % 10], NL
            pl = prompts.tolist()
            for i, (a, b) in enumerate(P.tolist()):   # intent: the targets are exactly the tokenizer's tokens of "<sum>\n"
                full = tok(f"{two_prompt(a, b, '+')}{a + b}\n")["input_ids"]
                assert full == pl[i] + [t for t in tgt[i].tolist() if t != -100], (a, b, tok.convert_ids_to_tokens(full))
            T = torch.tensor(tgt, device=dev)
            return torch.cat([prompts, T[:, :2]], 1), T, s
        for name, P in (("test", test), ("train", train)):
            A3[name] = seq3(P)
        plen = A3["test"][0].shape[1] - 2
        assert plen == slot_pos["A"] + 6, plen

        def show(name: str, i: int) -> str:   # vii: the decoded model input, scored target tokens in brackets
            inp, T, s = A3[name]
            n_t = int((T[i] != -100).sum())
            seq = inp[i, :plen].tolist() + T[i, :n_t].tolist()
            cells = [repr(tok.decode([t]))[1:-1] for t in seq]
            cells = [f"⟦{c}⟧" if j >= plen else c for j, c in enumerate(cells)]
            return (f"{name:5s} {A3[name + '_P'][i, 0]:3d} + {A3[name + '_P'][i, 1]:3d} = {s[i]:4d} | " + "|".join(cells)
                    + f" | scored token idx {list(range(plen, plen + n_t))} (logits at {list(range(plen - 1, plen - 1 + n_t))}), "
                      f"input len {inp.shape[1]}, full len {plen + n_t}")
        A3["test_P"], A3["train_P"] = test, train
        for i in range(3):
            print("[vii]", show("train", i), flush=True)
            print("[vii]", show("test", i), flush=True)
        jl, jg = int(np.argmax(A3["test"][2] < 1000)), int(np.argmax(A3["test"][2] >= 1000))
        print("[vii]", show("test", jl), "\n[vii]", show("test", jg), flush=True)
        print(f"[add3] train {len(train)} pairs ({(A3['train'][2] >= 1000).mean():.3f} with sum >= 1000), test {len(test)} "
              f"({(A3['test'][2] >= 1000).mean():.3f}); unordered train/test overlap 0 (asserted)", flush=True)
        heldout, seen, items, evalsets = np.array([], dtype=int), np.unique(train), {"test": test}, {}
        train_ans = None
    else:
        ans_of = lambda P: answers("add" if args.task == "mixed" else args.task, P)
        rng = np.random.default_rng(0)
        pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100)
                          if (args.task != "sub" or a >= b) and (args.task != "mul" or a * b <= 999)])
        pairs = pairs[rng.permutation(len(pairs))]
        heldout = np.array([], dtype=int)
        if args.split == "random":
            if args.fixed_test:
                test = pairs[1500: 1500 + args.test_pairs]
                train = np.concatenate([pairs[: min(args.train_pairs, 1500)], pairs[1500 + args.test_pairs:][: max(0, args.train_pairs - 1500)]])
            else:
                train, test = pairs[: args.train_pairs], pairs[args.train_pairs: args.train_pairs + args.test_pairs]
        else:
            if args.split == "holdout_operand":
                heldout = np.sort(np.random.default_rng(7).choice(np.arange(10, 100), 15, replace=False))
                in_train = ~np.isin(pairs, heldout).any(1)
            else:   # units carry (add) / borrow (sub) only in the test set
                u = pairs % 10
                assert args.task in ("add", "sub"), "--split carry: add / sub only"
                in_train = (u[:, 0] + u[:, 1] < 10) if args.task == "add" else (u[:, 0] >= u[:, 1])
            train, test = pairs[in_train][: args.train_pairs], pairs[~in_train][: args.test_pairs]
        mixed = {}
        if args.task == "mixed":   # per-operation splits; "test" stays the addition test set
            mrng = np.random.default_rng(0)
            for o, ntr, nte in (("add", 1500, 1000), ("sub", 1500, 1000), ("mul", 900, 500)):
                P = np.array([(a, b) for a in range(10, 100) for b in range(10, 100) if (o != "sub" or a >= b) and (o != "mul" or a * b <= 999)])
                P = P[mrng.permutation(len(P))]
                mixed[o] = (P[:ntr], P[ntr:ntr + nte])
            train = np.concatenate([mixed[o][0] for o in mixed])
            train_op = sum(([OPS[o]] * len(mixed[o][0]) for o in mixed), [])
            train_ans = np.concatenate([answers(o, mixed[o][0]) for o in mixed])
            test = mixed["add"][1]
        else:
            assert len(train) == args.train_pairs and len(test) == args.test_pairs, (len(train), len(test))
            train_op, train_ans = [op] * len(train), ans_of(train)
        # per-number means are taken per slot, so every value must occur in every slot of the probe prompts
        # a small --fixed_test train set (E1) misses some values: probe on the default 1500 train pairs (never test pairs)
        probe_is_train = args.task == "add" and args.split == "random" and len(train) >= 1500
        probe = train if probe_is_train else (pairs[:1500] if args.fixed_test else pairs)
        for j in range(2):
            assert set(np.unique(probe[:, j])) >= set(range(10, 100)), f"slot {j}: some operand value never occurs in the probe prompts"
        seen = np.unique(train)
        if args.task == "add":
            trip = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 199])
            big = np.array([(a, b) for a in range(100, 500) for b in range(100, 500)])
        else:
            trip = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a - b - c >= 0])
            big = np.array([(a, b) for a in range(100, 500) for b in range(100, 500) if a >= b])
        three = trip[rng.permutation(len(trip))][:400]
        big = big[rng.permutation(len(big))][:400]
        evalsets = {
            "test": (enc([two_prompt(*x, op) for x in test]), ans_of(test)),
            "train": (enc([two_prompt(a, b, o) for (a, b), o in zip(train, train_op)]), train_ans),
            "test_terse": (enc([terse_prompt(*x, op) for x in test]), ans_of(test)),
        }
        items = {"test": test, "test_terse": test}
        for o in ("sub", "mul") if mixed else ():
            evalsets[f"test_{o}"] = (enc([two_prompt(a, b, OPS[o]) for a, b in mixed[o][1]]), answers(o, mixed[o][1]))
            items[f"test_{o}"] = mixed[o][1]
        if mixed and args.gpt3_arith:   # GPT-3 arithmetic, pairs unseen in that operation's training set
            for o in mixed:
                seen_o = {tuple(x) for x in mixed[o][0].tolist()}
                P = np.array([x for x in gpt3_items(args.gpt3_arith, o) if tuple(x) not in seen_o])
                evalsets[f"gpt3_unseen_{o}"] = (enc([gpt3_prompt(a, b, GPT3_FILES[o][1]) for a, b in P]), answers(o, P))
                items[f"gpt3_unseen_{o}"] = P
        if args.task in ("add", "sub"):
            evalsets["three_term"] = (enc([three_prompt(*x, op) for x in three]), ans_of(three))
            evalsets["three_digit"] = (enc([two_prompt(*x, op) for x in big]), ans_of(big))
            items.update(three_term=three, three_digit=big)
        for k, tpl in enumerate(WORD.get(args.task, ())):   # natural-language templates on the test pairs
            evalsets[f"word{k}"] = (enc([tpl.format(a=a, b=b) for a, b in test]), ans_of(test))
            items[f"word{k}"] = test
        if args.gpt3_arith and args.task in GPT3_FILES:    # GPT-3 arithmetic benchmark (two-digit items)
            g3 = gpt3_items(args.gpt3_arith, args.task)
            seen_pairs = {tuple(x) for x in train.tolist()}
            g3u = np.array([x for x in g3 if tuple(x) not in seen_pairs])
            for name, P in (("gpt3", g3), ("gpt3_unseen", g3u)):
                evalsets[name] = (enc([gpt3_prompt(a, b, GPT3_FILES[args.task][1]) for a, b in P]), ans_of(P))
                items[name] = P
        train_ids = evalsets["train"][0]
        fmt_ids = [train_ids]
        if args.mixed_formats:
            assert args.task in FORMATS and FORMATS[args.task][0] == two_prompt("{a}", "{b}", op), "--mixed_formats: add / sub"
            fmt_ids = [enc([f.format(a=a, b=b) for a, b in train]) for f in FORMATS[args.task]]
            for k, f in enumerate(FORMATS[args.task][1:], 1):
                evalsets[f"fmt{k}"] = (enc([f.format(a=a, b=b) for a, b in test]), ans_of(test))
                items[f"fmt{k}"] = test
        probe_ids = train_ids if probe_is_train else enc([two_prompt(*x, op) for x in probe])
    perm = np.arange(100); perm[10:] = 10 + np.random.default_rng(123).permutation(90)
    num_of = torch.full((model.config.vocab_size,), -1, dtype=torch.long, device=dev)
    num_of[num_ids[:1000]] = torch.arange(1000, device=dev)

    lm_val_ids = None
    if args.lm_val:
        vt = np.memmap(args.lm_val, dtype=np.uint32, mode="r")
        offs = np.random.default_rng(7).integers(0, len(vt) - 513, size=32)
        lm_val_ids = torch.tensor(np.stack([vt[o:o + 513].astype(np.int64) for o in offs]), device=dev)

    @torch.no_grad()
    def lm_val_loss() -> float | None:
        """Mean next-token loss on held-out FineWeb-Edu text (general ability)."""
        if lm_val_ids is None:
            return None
        model.eval()
        tot = 0.0
        for i in range(0, len(lm_val_ids), 8):
            x = lm_val_ids[i:i + 8]
            with torch.autocast("cuda" if dev.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                lg = model(input_ids=x[:, :-1]).logits
            tot += float(Fn.cross_entropy(lg.float().flatten(0, 1), x[:, 1:].flatten())) * len(x)
        return tot / len(lm_val_ids)

    @torch.no_grad()
    def test_acc(reft: bool) -> float:
        model.eval()
        ids, ans = evalsets["test"]
        am = torch.cat([forward(ids[i:i + args.eval_bs], reft).argmax(-1) for i in range(0, len(ids), args.eval_bs)])
        return float((am == num_ids[torch.tensor(ans, device=dev)]).float().mean())

    @torch.no_grad()
    def evaluate(reft: bool, preds: dict | None = None) -> dict:
        """Accuracy per eval set; with `preds` also fills {set: predicted number or -1} (not for train)."""
        model.eval()
        res = {}
        for name, (ids, ans) in evalsets.items():
            am = torch.cat([forward(ids[i:i + args.eval_bs], reft).argmax(-1) for i in range(0, len(ids), args.eval_bs)])
            res[name] = float((am == num_ids[torch.tensor(ans, device=dev)]).float().mean())
            if preds is not None and name in items:
                preds[name] = num_of[am].tolist()
        return res

    @torch.no_grad()
    def eval3(name: str, reft: bool, preds: dict | None = None) -> dict:
        """--add3: teacher-forced exact match of "<sum>\\n" (= greedy decoding of the answer and its newline), overall
        and split by sum < 1000 (one answer token) / >= 1000 (two); with `preds`: per-item hits + first-token argmax."""
        model.eval()
        inp, T, s = A3[name]
        am = torch.cat([forward_k(inp[i:i + args.eval_bs], reft, 3).argmax(-1) for i in range(0, len(inp), args.eval_bs)])
        hit = ((am == T) | (T == -100)).all(1).cpu().numpy()
        if preds is not None:
            preds[name] = {"hit": hit.astype(int).tolist(), "first_token_number": num_of[am[:, 0]].tolist()}
        return {name: float(hit.mean()), f"{name}_lt1000": float(hit[s < 1000].mean()), f"{name}_ge1000": float(hit[s >= 1000].mean())}

    @torch.no_grad()
    def number_means(reft: bool) -> dict:
        """{(L, slot): [90, d]} per-number mean state over the probe prompts (the training prompts by default)."""
        model.eval()
        sums = {(L, s): torch.zeros(90, d, device=dev) for L in man_layers for s in slot_pos}
        cnt = {s: torch.zeros(90, device=dev) for s in slot_pos}
        for i in range(0, len(probe_ids), args.eval_bs):
            forward(probe_ids[i:i + args.eval_bs], reft)
            chunk = probe[i:i + args.eval_bs]
            for j, s in enumerate(slot_pos):
                idx = torch.tensor(chunk[:, j] - 10, device=dev)
                for L in man_layers:
                    sums[(L, s)].index_add_(0, idx, store[L][:, slot_pos[s], :].float().to(dev))
                cnt[s].index_add_(0, idx, torch.ones(len(idx), device=dev))
        return {k: (v / cnt[k[1]][:, None]).cpu().numpy() for k, v in sums.items()}

    grams = target_grams(perm, dev)
    feats2 = {G: target_features(G, perm, dev) for G in grams}
    perm3 = np.arange(1000); perm3[lo3:] = lo3 + np.random.default_rng(321).permutation(1000 - lo3)
    feats3 = {G: target_features(G, perm3, dev, lo3) for G in GEOMS3}
    far_check = {}
    if args.add3:   # the shuffled control must be FAR from the true geometry (project rule: CKA(shuffled, true) < 0.15)
        all3 = torch.arange(1000, device=dev)
        for G in GEOMS3:
            if G.endswith("_shuf"):
                Kt, Ks = gram_of(feats3[G[:-5]][all3]), gram_of(feats3[G][all3])
                far_check[G] = float((Kt * Ks).sum() / (Kt.norm() * Ks.norm()))
                assert G not in geoms or far_check[G] < 0.15, ("add3: shuffled target not far from the true one", G, far_check[G])
        assert int((perm3 == np.arange(1000)).sum()) < 10, "add3: perm3 leaves many numbers in place"
        print(f"[add3] CKA(shuffled target, true target) over 0..999: {far_check}; fixed points of perm3: "
              f"{int((perm3 == np.arange(1000)).sum())}; mean |perm3(x) - x| = {np.abs(perm3 - np.arange(1000)).mean():.0f}", flush=True)
    perm_t, perm3_t = torch.tensor(perm, device=dev), torch.tensor(perm3, device=dev)
    geo_xs2 = None if args.geo_numbers == "all" else torch.tensor(seen, device=dev)   # None = all of 10..99
    geo_rng = np.random.default_rng(99)
    report_xs3 = torch.tensor(np.sort(np.random.default_rng(5).choice(np.arange(10, 1000), 300, replace=False)), device=dev)

    def geo_states(reft: bool, xs=None) -> dict:
        """{L: H [n, d]} A-slot states of the numbers xs (default 10..99), with grad; with --geo_site emb the
        single entry "emb" holds the input-embedding rows instead (no forward needed)."""
        if args.geo_site == "emb":
            return {"emb": number_embedding_rows(xs)}
        forward(num_prefix if xs is None else prefix_all[xs], reft)
        return {L: store[L][:, slot_pos["A"], :].to(dev) for L in man_layers}

    def target_gram(G: str, key, xs) -> torch.Tensor:
        """Centred target Gram of geometry G for the numbers xs (None = 10..99) at site `key` (a layer or "emb")."""
        if G in TEACHER:
            src, shuf, n = TEACHER[G]
            rows = torch.arange(10, 100, device=dev) if xs is None else xs
            if shuf:
                rows = (perm3_t if n == 3 else perm_t)[rows]
            return gram_of(teach[src][key][rows])
        if G in GEOMS3:
            return gram_of(feats3[G][xs])
        return grams[G] if xs is None else gram_of(feats2[G][xs])

    def geo_loss(G: str, reft: bool) -> torch.Tensor:
        if is_three(G):
            xs = torch.tensor(np.sort(geo_rng.choice(np.arange(lo3, 1000), args.geo_sample, replace=False)), device=dev)
        else:
            xs = geo_xs2
        Hs = geo_states(reft, xs)
        if G == "sep_ce":
            ys = torch.arange(10, 100, device=dev) if xs is None else xs
            return sum(sep_ce_loss(H, ys) for H in Hs.values()) / len(Hs)
        return sum(1 - cka(Hs[L], target_gram(G, L, xs)) for L in Hs) / len(Hs)

    @torch.no_grad()
    def geometry_report(reft: bool) -> dict:
        model.eval()
        key = lambda L: f"L{L}" if L != "emb" else "emb"
        forward(num_prefix, reft)
        Hs = {L: store[L][:, slot_pos["A"], :].to(dev) for L in man_layers}
        Hs["emb"] = number_embedding_rows()
        rep = {key(L): {"cka_" + G: float(cka(H, grams[G])) for G in grams} for L, H in Hs.items()}
        from geom_diag import kappa   # linear separability of n mod T (CV ridge classifier), see sep_ce
        x90 = np.arange(10, 100)
        for L, H in Hs.items():
            Hn = H.float().cpu().numpy()
            rep[key(L)].update({f"kappa_mod{T}": kappa(Hn, x90 % T) for T in (2, 5, 10)})
            rep[key(L)]["kappa_tens"] = kappa(Hn, x90 // 10)
            rep[key(L)]["sep_ce"] = float(sep_ce_loss(H, torch.tensor(x90, device=H.device)))
        for src in teach:
            for L, H in Hs.items():
                rep[key(L)]["cka_" + src] = float(cka(H, gram_of(teach[src][L][10:100])))
        forward(prefix_all[report_xs3], reft)
        H3 = {L: store[L][:, slot_pos["A"], :].to(dev) for L in man_layers}
        H3["emb"] = number_embedding_rows(report_xs3)
        for L, H in H3.items():
            rep[key(L)].update({"cka_" + G: float(cka(H, gram_of(feats3[G][report_xs3]))) for G in GEOMS3})
            for src in teach:
                rep[key(L)]["cka_" + src + "3"] = float(cka(H, gram_of(teach[src][L][report_xs3])))
        if args.add3:   # every number 0..999 (the prior's own range), all layers
            all3 = torch.arange(1000, device=dev)
            forward(prefix_all, reft)
            for L in man_layers:
                rep[key(L)].update({f"cka_{G}_all1000": float(cka(store[L][:, slot_pos["A"], :].to(dev), gram_of(feats3[G][all3])))
                                    for G in GEOMS3})
        return rep

    gsuite = None
    add3_info = {}
    if args.add3:
        import hashlib
        sha = lambda P: hashlib.sha256(np.ascontiguousarray(P, dtype=np.int64).tobytes()).hexdigest()[:16]
        add3_info = {"test_sha": sha(test), "train_sha": sha(train), "n_test": len(test), "n_train": len(train),
                     "test_frac_ge1000": float((A3["test"][2] >= 1000).mean()), "far_check_cka": far_check, "lo3": lo3,
                     "perm3_fixed_points": int((perm3 == np.arange(1000)).sum()), "scoring": "teacher-forced exact match of <sum>\\n"}
        base_preds = {} if args.save_preds else None
        base_eval = {**eval3("test", False, base_preds), **eval3("train", False)}
        (out / "base.json").write_text(json.dumps({"eval": base_eval, "geometry": geometry_report(reft=False), "task": "add3",
                                                   "add3": add3_info, "items": {"test": test.tolist(), "train": train.tolist()},
                                                   "preds": base_preds}, indent=1))
        print(f"base: {base_eval} add3 {add3_info} ({time.time() - t0:.0f}s)", flush=True)
    else:
        base_preds = {} if args.save_preds else None
        base_eval = evaluate(reft=False, preds=base_preds)
        if lm_val_ids is not None:
            base_eval["lm_val_loss"] = lm_val_loss()
        if args.general_eval:
            from general_eval import load_suite, run_suite
            gsuite = load_suite(args.general_eval, args.general_n)
            base_eval["general"] = run_suite(model, tok, gsuite, dev)
            print("base general:", base_eval["general"], flush=True)
        base_means = number_means(reft=False)
        (out / "base.json").write_text(json.dumps({"eval": base_eval, "manifold": diagnostics(base_means, None, slot_pos),
                                                   "geometry": geometry_report(reft=False), "task": args.task, "split": args.split,
                                                   "heldout": heldout.tolist(), "items": {k: v.tolist() for k, v in items.items()},
                                                   "preds": base_preds}, indent=1))
        print(f"base: {base_eval} ({time.time() - t0:.0f}s)")

    for seed in seeds:
        for G in geoms:
            path = out / f"run_{G}_seed{seed}.json"
            if path.exists():
                continue
            reset(seed)
            params = trainable()
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, **({"fused": True} if (args.mode == "full" and dev == "cuda") else {}))
            order_rng = np.random.default_rng(1000 + seed)
            log = []
            model.train()
            step, half = 0, args.epochs * math.ceil(len(train) / args.bs) // 2
            ema_acc, gate_on_steps = 0.0, 0
            curve = []
            snaps, snap_steps = [], []
            grad_log = []

            def snapshot():
                with torch.no_grad():
                    model.eval()
                    if args.add3:   # every number 0..999, row x = number x
                        H = geo_states(reft=True, xs=torch.arange(1000, device=dev))[man_layers[-1]]
                        assert H.shape[0] == 1000, H.shape
                    else:
                        H = geo_states(reft=True)[man_layers[-1]]
                    snaps.append(H.float().cpu().numpy().astype(np.float16)); snap_steps.append(step)
                    model.train()
            if args.save_states_every:
                snapshot()
            if args.add3 and args.curve_every:   # step 0 = the untrained adapter (= base)
                curve.append({"step": 0, **eval3("test", reft=True)})
                model.train()
            for ep in range(args.epochs):
                order = order_rng.permutation(len(train))
                ce_s = man_s = 0.0; nb = 0
                for i in range(0, len(order), args.bs):
                    bi = order[i:i + args.bs]
                    chunk = train[bi]
                    # free the previous gradients before the forwards, and run both forwards inside ONE autocast
                    # region so they share a single bf16 copy of the weights (full FT: two copies + stale grads
                    # overflowed 140 GB)
                    opt.zero_grad(set_to_none=True)
                    with autocast:
                        if args.add3:   # CE = token mean over the answer tokens and the newline (teacher forced)
                            bt = torch.tensor(bi, device=dev)
                            logits, tb = forward_k(A3["train"][0][bt], True, 3), A3["train"][1][bt]
                            ce = Fn.cross_entropy(logits.flatten(0, 1), tb.flatten(), ignore_index=-100)
                        else:
                            logits = forward(fmt_ids[nb % len(fmt_ids)][torch.tensor(bi, device=dev)], reft=True)   # format rotates per batch
                            ce = Fn.cross_entropy(logits, num_ids[torch.tensor(train_ans[bi], device=dev)])
                        if G == "none":
                            with torch.no_grad():   # logged only (add3: the helix3 loss over 0..999)
                                ml = geo_loss("helix3" if args.add3 else "helix", reft=True)
                            loss = ce
                        elif G == "helix_gated":
                            if ema_acc < args.gate_thr:
                                ml = geo_loss("helix", reft=True); loss = ce + args.lam * ml; gate_on_steps += 1
                            else:
                                ml = torch.zeros((), device=dev); loss = ce
                        else:
                            Gs = ("sep_ce" if step < half else "helix") if G == "sep_then_helix" else G
                            ml = geo_loss(Gs, reft=True)
                            loss = ce + (args.sep_lam if Gs == "sep_ce" else args.lam) * ml
                    with torch.no_grad():
                        if args.add3:
                            bacc = float(((logits.argmax(-1) == tb) | (tb == -100)).all(1).float().mean())
                        else:
                            bacc = float((logits.argmax(-1) == num_ids[torch.tensor(train_ans[bi], device=dev)]).float().mean())
                        ema_acc = 0.95 * ema_acc + 0.05 * bacc
                    if args.add3 and args.grad_check_every and step % args.grad_check_every == 0:
                        # A/B of the two loss terms on this batch: |grad CE| and |grad lam * L_geo| over the trainable params
                        def gnorm(t):
                            gs = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
                            return math.sqrt(sum(float((g.float() ** 2).sum()) for g in gs if g is not None))
                        rec = {"step": step, "ce": float(ce), "geo_loss": float(ml), "lam": args.lam, "grad_ce": gnorm(ce)}
                        if G != "none":
                            rec["grad_geo"] = gnorm(ml)   # unscaled; the loss adds args.lam * ml (lambda A/B: smoke lam 0 == none)
                            assert float(ml) > 0 and math.isfinite(rec["grad_geo"]) and rec["grad_geo"] > 0, ("add3: prior has no gradient", rec)
                            rec["grad_lam_geo"] = args.lam * rec["grad_geo"]
                            rec["geo_share"] = rec["grad_lam_geo"] / (rec["grad_lam_geo"] + rec["grad_ce"] + 1e-30)
                        grad_log.append(rec)
                        print(f"[grad] {G} seed{seed} {rec}", flush=True)
                    loss.backward()
                    opt.step(); step += 1
                    if args.save_states_every and step % args.save_states_every == 0:
                        snapshot()
                    if args.curve_every and step % args.curve_every == 0:
                        curve.append({"step": step, **eval3("test", reft=True)} if args.add3 else {"step": step, "test": test_acc(reft=True)})
                        model.train()
                    ce_s += float(ce.detach()); man_s += float(ml.detach()); nb += 1
                    del logits, loss
                log.append({"epoch": ep, "ce": ce_s / nb, "geo_loss": man_s / nb, "ema_train_acc": ema_acc, "gate_on_frac": gate_on_steps / max(step, 1)})
            del opt
            if snaps:
                np.savez_compressed(out / f"states_{G}_seed{seed}.npz", states=np.stack(snaps), steps=np.array(snap_steps),
                                    layer=man_layers[-1], numbers=np.arange(1000) if args.add3 else np.arange(10, 100))
            preds = {} if args.save_preds else None
            if args.add3:
                ev, diag = {**eval3("test", True, preds), **eval3("train", True)}, {}
            else:
                ev = evaluate(reft=True, preds=preds)
                if lm_val_ids is not None:
                    ev["lm_val_loss"] = lm_val_loss()
                if gsuite is not None:
                    ev["general"] = run_suite(model, tok, gsuite, dev)
                diag = diagnostics(number_means(reft=True), None, slot_pos)
            extra = {"add3": add3_info, "grad_log": grad_log, "steps": step, "epochs": args.epochs, "bs": args.bs,
                     "train_pairs": args.train_pairs, "model": args.model} if args.add3 else {}
            path.write_text(json.dumps({"mode": args.mode, "loss": "v2_cka", "geo_site": args.geo_site, "emb_delta": args.emb_delta,
                                        "task": args.task, "split": args.split, "geo_numbers": args.geo_numbers, "preds": preds,
                                        "emb_init": args.emb_init, "emb_init_info": emb_info, "geom": G, "seed": seed, "lam": args.lam, "lr": lr,
                                        "geo_sample": args.geo_sample if is_three(G) else None,
                                        "layers": man_layers, "train_log": log, "curve": curve, "eval": ev, "manifold": diag,
                                        "geometry": geometry_report(reft=True), **extra}, indent=1))
            print(f"{args.mode} {G} seed{seed}: {ev} ({time.time() - t0:.0f}s)")
    if not args.add3:   # add3 read-out: tools/ft999_summary.py
        summarize(out)


def diagnostics(means: dict, targets: dict, slot_pos: dict) -> dict:
    """Held-out-number R^2 of the helix and digit geometries on the per-number mean states (slot A)."""
    out = {}
    x = np.arange(10, 100)
    for (L, s), X in means.items():
        if s != "A":
            continue
        out[f"L{L}"] = {"r2_helix": nl.cv_r2(nl.features(x)[0], X),
                        "r2_digit": nl.cv_r2(np.concatenate([np.eye(10)[x // 10], np.eye(10)[x % 10]], 1), X)}
    return out


def summarize(out: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    if not runs:
        return
    base = json.loads((out / "base.json").read_text())
    keys = [k for k, v in runs[0]["eval"].items() if isinstance(v, (int, float))]   # nested dicts (general suite) are reported elsewhere
    ts = f", task {runs[0].get('task', 'add')}, split {runs[0].get('split', 'random')}" + (f", geo_numbers {runs[0]['geo_numbers']}" if runs[0].get("geo_numbers", "all") != "all" else "")
    lines = [f"# manifold fine-tuning ({runs[0]['mode']}, lam {runs[0]['lam']}, lr {runs[0]['lr']}, layers {runs[0]['layers']}{ts})\n\n",
             "| geom | n seeds | " + " | ".join(keys) + " | r2 helix / digit (first layer) |\n", "|---|---|" + "---|" * (len(keys) + 1) + "\n",
             "| base | - | " + " | ".join(f"{base['eval'][k]:.3f}" for k in keys) + " | - |\n"]
    for G in ALL_GEOMS:
        rs = [r for r in runs if r["geom"] == G]
        if not rs:
            continue
        cells = []
        for k in keys:
            v = np.array([r["eval"][k] for r in rs])
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}")
        first = next(iter(rs[0]["manifold"]))
        r2 = np.mean([[r["manifold"][first]["r2_helix"], r["manifold"][first]["r2_digit"]] for r in rs], 0)
        ck = ""
        if "geometry" in rs[0]:
            gl = [k for k in rs[0]["geometry"] if k.startswith("L")][-1]
            c = np.mean([[r["geometry"][gl]["cka_helix"], r["geometry"][gl]["cka_digit"]] for r in rs], 0)
            ck = f" CKA@{gl} helix {c[0]:.2f} digit {c[1]:.2f}"
            if "cka_main" in rs[0]["geometry"][gl]:
                ck += f" main {np.mean([r['geometry'][gl]['cka_main'] for r in rs]):.2f} main3 {np.mean([r['geometry'][gl]['cka_main3'] for r in rs]):.2f}"
            if "emb" in rs[0]["geometry"]:
                e = np.mean([[r["geometry"]["emb"]["cka_helix"], r["geometry"]["emb"]["cka_digit"]] for r in rs], 0)
                ck += f"; emb helix {e[0]:.2f} digit {e[1]:.2f}"
        lines.append(f"| {G} | {len(rs)} | " + " | ".join(cells) + f" | {r2[0]:.2f} / {r2[1]:.2f}{ck} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
