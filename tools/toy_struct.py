"""Toy grokking with a structure prior on the item embeddings, three tasks (generalises tools/toy_grok.py, which stays
untouched; --task mod reproduces it step for step and is the known-answer path, tools/test_toy_struct.py).

Model and optimiser are toy_grok's: one-layer transformer (d 128, 4 heads, MLP 512, no LayerNorm, causal attention,
readout at the last token '='), AdamW lr 1e-3 (10-step warmup), betas (0.9, 0.98), weight decay 1, full batch.
Prior: lam * (1 - linear CKA(W_E[items], target)) added to the CE while the schedule is on (default window [0, 2000)).

--task mod  (a, b, '=') -> (a + b) mod p, p 113; arms and targets exactly as toy_grok (none, always, w<t0>, *_shuf;
            circle of --freqs).
--task s4   the 24 proper rotations of the cube = 3x3 signed permutation matrices with det +1 (the group S4); tokens
            0..23 are the rotations in a fixed enumeration order (nonce: no names), token 24 = '='.
            Question (a, b, '=') -> c with R_c = R_b R_a  ("a then b": a acts first on column vectors).
            576 questions. Asserted at build time: closure, associativity of the table, identity, inverses,
            non-abelian, the action on the 4 body diagonals is an isomorphism onto S4 (and a homomorphism), its sign
            equals det|R|, V4 = kernel of S4 -> S3 is normal with 6 cosets of 4.
            Targets: R (9 cols, faithful 3-d irrep); R_sign = R * sign of the body-diagonal permutation (the other
            faithful 3-d irrep); absR = |R| (the S4 -> S3 quotient, kernel V4, non-faithful); specrand (random 9-d,
            same centred singular values as R, random singular vectors orthogonal to the constant, a new draw per
            seed); shuf (R with the items permuted: a far permutation per seed, CKA to R < 0.15 or the farthest of 3000).
--task dim  items = exponent vectors e in {-1,0,1}^3 (27 "quantities"), tokens 0..26, 27 = mul, 28 = div, 29 = '='.
            Question (a, op, b, '=') -> the item with e_a + e_b (mul) or e_a - e_b (div), only when it is in the box:
            7^3 = 343 per op, 686 questions (asserted).
            Targets: expo (e raw, 3 cols); quot (drop exponent 3: e[:, :2]); onehot (one-hot of each coordinate value,
            9 cols: same information, not additive); specrand (spectrum matched to expo, per seed); shuf (far perm).
            Optional (not in the default arms): specrand_oh (spectrum matched to onehot, rank 6, rng 2000 + seed): the
            rank control for onehot (onehot's centred span contains expo; rank 6 vs 3).
All targets are raw features, centred over the items inside the Gram (as toy_grok: K = centred Gram of the target).
Arm names: a target name uses the cell's --win; '<target>@always' or '<target>@<t0>-<t1>' overrides it for that arm.

Split (--split, default auto = class for dim, question for s4 and mod):
  question: numpy default_rng(seed).permutation(N), the first int(frac * N) questions train (toy_grok's split).
  class:    the same over RELATION CLASSES (all phrasings of one relation go to the same side): dim e_x + e_y = e_z is
            x*y, y*x, z/y, z/x (171 classes of 4 + the identity class {0*0, 0/0}: 172); s4 (a, b) and (b, a) when
            ab = ba. Project rule (symmetry-clean splits): with the question split, 87-99 % of dim's held-out
            questions at frac 0.6 had a class-mate in training, i.e. commutativity + the mul/div inverse alone
            reached ~0.95 held-out accuracy (audit 2026-10-16). s4 keeps the question split: its operation is not
            commutative and the 13-26 commuting twins per held-out set are reported as test_fresh / test_comm.
  test_fresh = held-out accuracy on the questions none of whose class-mates is in training (all of them in class mode).
Asserted: no held-out question in training (by content), no relation class split across train / held-out (class
mode), every item seen as an input and as an answer in training. Closure fraction (share of held-out answers derivable
from the training questions by the algebra):
s4 = associativity fixpoint ((ab)c = a(bc), both directions), plus a secondary one with Latin-square cancellation;
dim = the held-out relation row (e_a +- e_b - e_c) lies in the row space of the training relations (rank test and
least-squares residual, the two must agree); nullity of the training relation matrix (3 = identifiable up to GL(3)).

    python tools/toy_struct.py --task s4 --frac 0.3 --steps 50000 --seeds 0 --out <dir>
    python tools/toy_struct.py --task s4 --check            # targets, CKA matrix, closure, asserts; no training
    python tools/toy_struct.py --summarize <dir>             # pre-registered tests (runs/toystruct/AUDIT.md)
Run JSONs are written atomically (tmp file + os.replace); a training run never summarises (run --summarize once all
processes of a cell are done).

Wall 2 (2026-10-19, OPT-IN; every arm and flag above behaves exactly as before, runs/toyshape/AUDIT.md): a SHAPE-ONLY
prior that constrains the geometry of the item embeddings without saying which item sits where. Arm '<kind>:<target>'
(+ the usual '@always' / '@t0-t1' override, else the cell --win; works for every task, also mod):
  kind shape     = spec + norm;  kind shapeprof = spec + norm + prof (vertex-transitive targets only), each term in [0, 1]:
  spec = 1 - <lam(G_H), lam(G_T)> / (|G_H|_F |G_T|_F), lam = eigenvalues of the centred Gram, sorted (= 1 - the largest
         linear CKA over item-side orthogonal maps, von Neumann; <= 1 - CKA, i.e. <= the assign prior's loss);
  norm = 1 - cos(sorted diag G_H, sorted diag G_T)  (the profile of squared centred norms; constant for circle / R);
  prof = 1 - sum_i <sort(G_H[i]), s_T> / (|sortrows G_H|_F sqrt(n) |s_T|)  (every target row has the same sorted profile s_T).
  All three are invariant under any permutation of the items (P G P^T) and 0 for the target and every permutation of it.
  Targets: the task's own (mod 'circle' = cos/sin of --freqs, i.e. the torus for --freqs 1,2,3,4,5; s4 R / R_sign / absR;
  dim expo / quot / onehot) and two controls built from the centred rank r of the true shape (mod circle, s4 R, dim expo):
  'line'   = an open chain of the same rank r: path-graph modes cos(pi k (i + 1/2) / n) / k, k = 1..r (wrong spectrum
             ~ 1/k^2, non-constant norms; anisotropic);
  'sphere' = a flat, constant-norm, centred configuration of rank r' = sphere_rank(task, r) (mod r + 1, odd, so no union
             of Fourier circles; s4 12 for r 9, not a sum of S4's centred isotypic dimensions 1, 4, 9, 9): the same
             isotropy and norm profile as the true shape at another rank (audit 2026-10-19: spec + norm of the torus / R
             is "a flat constant-norm r-sphere", so this is the matched control).
  The eigendecomposition of the centred Gram runs on the CPU in float64 (LAPACK; audit 2026-10-19: cuSOLVER float32 on
  a Gram with ~111 near-zero eigenvalues was untested), the rest on the arm's device.
  --shape_log: every arm also logs shapeL_<target> (spec + norm) for every shape target, every eval, and for mod also
  shapeL_fourier1 / shapeL_fourier1-5 (circle / torus, whatever --freqs is): logging only.
  --summarize_wall <root>: the pre-registered wall-2 reading over the cell dirs <root>/<cell>/ (runs/toyshape/AUDIT.md).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

WIN_DEFAULT = "0,2000"
ARMS = {"mod": "none,always,w0,w1000,w2000,w4000,w6000,w8000,w12000,always_shuf,w2000_shuf,w6000_shuf",
        "s4": "none,R,R_sign,absR,specrand,shuf",
        "dim": "none,expo,quot,onehot,specrand,shuf"}
TRUE = {"s4": "R", "dim": "expo"}   # the target that shuf permutes and specrand matches
FAR_REFS = {"s4": ("R", "R_sign"), "dim": ("expo",)}   # shuf must be far from all of these
DROP = 2                            # quot drops exponent 3 (index 2)
B_WIN = (1000, 2000)                # pre-registered evals for the coset / dropped-exponent error shares
LATE = 4000                         # rule (a): a fraction qualifies iff none's censored median t_grok >= LATE (2 x window end)
ALPHA = 0.025                       # per co-primary (t_grok, AUC): Bonferroni over the two, family alpha 0.05 per comparison
AUC_MIN = 0.05                      # AUC test: median AUC(a) - median AUC(b) >= AUC_MIN (no claim from sub-chance shifts)
SPLIT_AUTO = {"mod": "question", "s4": "question", "dim": "class"}
# wall 2 (opt-in): shape-only prior
SHAPE_KINDS = {"shape": ("spec", "norm"), "shapeprof": ("spec", "norm", "prof")}
SHAPE_TRUE = {"mod": "circle", "s4": "R", "dim": "expo"}   # the true shape; 'line' has its centred rank
SHAPE_OK = {"mod": ("circle", "line", "sphere"), "s4": ("R", "R_sign", "absR", "line", "sphere"),
            "dim": ("expo", "quot", "onehot", "line", "sphere")}


class OneLayer(nn.Module):
    """toy_grok.OneLayer with the vocabulary, output classes and sequence length as arguments
    (parameters are created in the same order, so the same seed gives the same init)."""

    def __init__(self, n_vocab: int, n_out: int, T: int, d: int = 128, heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.W_E = nn.Parameter(torch.randn(n_vocab, d) / d ** 0.5)
        self.W_pos = nn.Parameter(torch.randn(T, d) / d ** 0.5)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp_in, self.mlp_out = nn.Linear(d, d_mlp), nn.Linear(d_mlp, d)
        self.W_U = nn.Linear(d, n_out, bias=False)

    def forward(self, x):
        h = self.W_E[x] + self.W_pos
        B, T, D = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.o(att.transpose(1, 2).reshape(B, T, D))
        h = h + self.mlp_out(Fn.relu(self.mlp_in(h)))
        return self.W_U(h[:, -1])


def centred_gram(F: torch.Tensor) -> torch.Tensor:
    F = F - F.mean(0)
    return F @ F.T


def cka(H: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    G = centred_gram(H)
    return (G * K).sum() / (G.norm() * K.norm() + 1e-12)


def fourier_top(W: torch.Tensor, k: int = 5) -> float:
    P = torch.fft.rfft(W.float(), dim=0).abs().pow(2).sum(1)[1:]
    return float(P.topk(k).values.sum() / P.sum())


# ---------------------------------------------------------------------------------------------------- numpy helpers
def np_gram(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, np.float64)
    F = F - F.mean(0)
    return F @ F.T


def np_cka(A: np.ndarray, B: np.ndarray) -> float:
    G, K = np_gram(A), np_gram(B)
    return float((G * K).sum() / (np.linalg.norm(G) * np.linalg.norm(K)))


def centred_rank(F: np.ndarray) -> int:
    F = np.asarray(F, np.float64)
    return int(np.linalg.matrix_rank(F - F.mean(0), tol=1e-8))


def spec_matched(F: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random features with the centred singular values of F; left singular vectors Haar-random and orthogonal to 1."""
    Fc = np.asarray(F, np.float64) - np.mean(F, 0)
    s = np.linalg.svd(Fc, compute_uv=False)
    k = int((s > 1e-9 * s[0]).sum())
    Z = rng.standard_normal((len(F), k))
    U, _ = np.linalg.qr(Z - Z.mean(0))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    out = U @ np.diag(s[:k]) @ V.T
    assert np.allclose(out.mean(0), 0, atol=1e-10)
    assert np.allclose(np.linalg.svd(out - out.mean(0), compute_uv=False)[:k], s[:k], atol=1e-8)
    return out


def far_perm(F: np.ndarray, seed: int, refs: list | None = None, n_try: int = 3000, thr: float = 0.15):
    """Permutation q of the items such that F[q] is far from every reference target (default: F itself): the first q
    with max_ref CKA(F[q], ref) < thr, else the one with the smallest max of n_try (pretrain_helix / ft36 convention,
    rng 124 + seed). For s4 the references are R and R_sign: far from R alone lands in R_sign (CKA ~0.5, both are
    faithful 3-d irreps and a 9-d target cannot avoid both, see AUDIT.md)."""
    r = np.random.default_rng(124 + seed)
    Gs = [np_gram(x) for x in (refs if refs is not None else [F])]
    best = (2.0, None)
    for _ in range(n_try):
        q = r.permutation(len(F))
        G1 = np_gram(F[q])
        c = max(float((G0 * G1).sum() / (np.linalg.norm(G0) * np.linalg.norm(G1))) for G0 in Gs)
        if c < best[0]:
            best = (c, q)
        if c < thr:
            break
    return best


# ---------------------------------------------------------------------------------------------------- the S4 group
def cube_group():
    mats = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), dtype=np.int64)
            for r, (c, s) in enumerate(zip(perm, sg)):
                M[r, c] = s
            if round(np.linalg.det(M)) == 1:
                mats.append(M)
    mats = np.array(mats)
    n = len(mats)
    assert n == 24, n
    for M in mats:
        assert (M @ M.T == np.eye(3, dtype=np.int64)).all() and round(np.linalg.det(M)) == 1
    key = {tuple(M.ravel()): i for i, M in enumerate(mats)}
    assert len(key) == 24
    mul = np.zeros((n, n), dtype=np.int64)          # mul[a, b] = "a then b" = index of R_b @ R_a
    for a in range(n):
        for b in range(n):
            P = mats[b] @ mats[a]
            assert tuple(P.ravel()) in key, "closure"
            mul[a, b] = key[tuple(P.ravel())]
    A, B_, C = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    assert (mul[mul[A, B_], C] == mul[A, mul[B_, C]]).all(), "associativity"
    e = key[tuple(np.eye(3, dtype=np.int64).ravel())]
    assert (mul[e] == np.arange(n)).all() and (mul[:, e] == np.arange(n)).all(), "identity"
    assert all((mul[a] == e).sum() == 1 and (mul[:, a] == e).sum() == 1 for a in range(n)), "inverses"
    for a in range(n):   # Latin square
        assert len(set(mul[a])) == n and len(set(mul[:, a])) == n
    assert (mul != mul.T).any(), "non-abelian"
    diag = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]])
    pi = np.zeros((n, 4), dtype=np.int64)            # pi[g, i] = j  with  R_g d_i = +- d_j
    for g in range(n):
        for i in range(4):
            v = mats[g] @ diag[i]
            js = [j for j in range(4) if (v == diag[j]).all() or (v == -diag[j]).all()]
            assert len(js) == 1
            pi[g, i] = js[0]
        assert sorted(pi[g]) == [0, 1, 2, 3]
    assert len({tuple(p) for p in pi}) == 24, "action on the body diagonals is faithful (onto S4)"
    for a in range(n):
        for b in range(n):   # a then b acts as pi_b o pi_a
            assert (pi[mul[a, b]] == pi[b][pi[a]]).all(), "homomorphism to S4"

    def perm_sign(p):
        p, s = list(p), 1
        for i in range(len(p)):
            while p[i] != i:
                j = p[i]
                p[i], p[j] = p[j], p[i]
                s = -s
        return s
    sgn = np.array([perm_sign(p) for p in pi])
    absR = np.abs(mats)
    assert (sgn == np.array([round(np.linalg.det(P)) for P in absR])).all(), "sign(diagonal perm) = det |R|"
    pkey = {}
    coset = np.array([pkey.setdefault(tuple(P.ravel()), len(pkey)) for P in absR])
    assert len(pkey) == 6 and all((coset == c).sum() == 4 for c in range(6))
    V4 = np.where(coset == coset[e])[0]
    assert len(V4) == 4 and all(np.diag(mats[v]).tolist() in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]) for v in V4)
    for a in range(n):
        for b in range(n):   # quotient is a homomorphism: coset(a then b) depends only on (coset a, coset b)
            assert coset[mul[a, b]] == coset[mul[np.where(coset == coset[a])[0][0], np.where(coset == coset[b])[0][0]]]
    return mats, mul, sgn, coset


def s4_closure(mul: np.ndarray, tr_cells: list, latin: bool = False) -> set:
    """Cells (a, b) derivable from the training cells by associativity (both directions) [and Latin cancellation]."""
    n = len(mul)
    known = {(a, b): int(mul[a, b]) for a, b in tr_cells}
    changed = True
    while changed:
        changed = False
        byleft = {}
        for (g, h), v in known.items():
            byleft.setdefault(g, {})[h] = v
        for (g, h), a in list(known.items()):   # g h = a ; h k = b  =>  a k = g b
            for k, b in list(byleft.get(h, {}).items()):
                ak, gb = known.get((a, k)), known.get((g, b))
                if ak is not None and gb is None:
                    known[(g, b)] = ak; changed = True
                elif gb is not None and ak is None:
                    known[(a, k)] = gb; changed = True
        if latin:   # a row (column) with one unknown cell: it takes the one unused value
            for a in range(n):
                miss = [b for b in range(n) if (a, b) not in known]
                if len(miss) == 1:
                    known[(a, miss[0])] = (set(range(n)) - {known[(a, b)] for b in range(n) if b != miss[0]}).pop(); changed = True
                miss = [b for b in range(n) if (b, a) not in known]
                if len(miss) == 1:
                    known[(miss[0], a)] = (set(range(n)) - {known[(b, a)] for b in range(n) if b != miss[0]}).pop(); changed = True
    for (a, b), v in known.items():
        assert v == mul[a, b], "derived cell wrong"
    return set(known)


# ---------------------------------------------------------------------------------------------------- tasks
def build_task(task: str, p: int = 113, freqs=(1,)) -> dict:
    """Questions, answers and fixed targets (numpy). Per-seed targets come from seed_targets()."""
    if task == "mod":
        a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        X = np.stack([a.ravel(), b.ravel(), np.full(p * p, p)], 1)
        Y = (X[:, 0] + X[:, 1]) % p
        ang = 2 * np.pi * np.arange(p) / p
        circ = np.concatenate([np.stack([np.cos(k * ang), np.sin(k * ang)], 1) for k in freqs], 1)
        return dict(task=task, X=X, Y=Y, n_items=p, n_vocab=p + 1, n_out=p, T=3, targets={"circle": circ}, p=p, freqs=list(freqs))
    if task == "s4":
        mats, mul, sgn, coset = cube_group()
        a, b = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
        X = np.stack([a.ravel(), b.ravel(), np.full(576, 24)], 1)
        Y = mul[X[:, 0], X[:, 1]]
        R = mats.reshape(24, 9).astype(np.float64)
        T = {"R": R, "R_sign": R * sgn[:, None], "absR": np.abs(R)}
        noncomm = mul[X[:, 0], X[:, 1]] != mul[X[:, 1], X[:, 0]]
        ckey = [(a, b) if nc else (min(a, b), max(a, b)) for (a, b), nc in zip(X[:, :2].tolist(), noncomm)]
        return dict(task=task, X=X, Y=Y, n_items=24, n_vocab=25, n_out=24, T=3, targets=T, mul=mul, coset=coset, sgn=sgn,
                    noncomm=noncomm, cls=class_ids(ckey))
    if task == "dim":
        E = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)
        key = {tuple(v): i for i, v in enumerate(E)}
        rows, ys = [], []
        for op, s in ((27, 1), (28, -1)):
            for i in range(27):
                for j in range(27):
                    c = E[i] + s * E[j]
                    if np.abs(c).max() <= 1:
                        rows.append((i, op, j, 29)); ys.append(key[tuple(c)])
        X, Y = np.array(rows), np.array(ys)
        assert (X[:, 1] == 27).sum() == 343 and (X[:, 1] == 28).sum() == 343 and len(X) == 686
        onehot = np.concatenate([(E[:, i:i + 1] == np.array([[-1, 0, 1]])).astype(np.float64) for i in range(3)], 1)
        T = {"expo": E.astype(np.float64), "quot": np.delete(E, DROP, 1).astype(np.float64), "onehot": onehot}
        # relation class: e_x + e_y = e_z is stated by x*y, y*x (mul) and z/y = x, z/x = y (div); key = (min, max of x, y)
        ckey = [(min(a, b), max(a, b)) if op == 27 else (min(c, b), max(c, b)) for (a, op, b, _), c in zip(X.tolist(), Y.tolist())]
        return dict(task=task, X=X, Y=Y, n_items=27, n_vocab=30, n_out=27, T=4, targets=T, E=E, cls=class_ids(ckey))
    raise ValueError(task)


def class_ids(keys: list) -> np.ndarray:
    """Integer id per question from its relation-class key (ids in order of first appearance)."""
    ids = {}
    return np.array([ids.setdefault(k, len(ids)) for k in keys], dtype=np.int64)


def seed_targets(D: dict, seed: int) -> tuple[dict, dict]:
    """All targets for this seed (fixed ones + the per-seed specrand / shuf) and their provenance."""
    T = dict(D["targets"])
    info = {}
    if D["task"] == "mod":
        T["circle_shuf"] = T["circle"][np.random.default_rng(5).permutation(D["p"])]   # toy_grok: a fixed permutation
        return T, info
    F = T[TRUE[D["task"]]]
    T["specrand"] = spec_matched(F, np.random.default_rng(1000 + seed))
    if D["task"] == "dim":   # rank control for onehot (own rng: the other targets' draws are unchanged)
        T["specrand_oh"] = spec_matched(T["onehot"], np.random.default_rng(2000 + seed))
    refs = [T[k] for k in FAR_REFS[D["task"]]]
    c, q = far_perm(F, seed, refs)
    T["shuf"] = F[q]
    info = {"shuf_far_max_cka": c, "shuf_cka_to_true": np_cka(T["shuf"], F), "specrand_cka_to_true": np_cka(T["specrand"], F),
            "shuf_perm": q.tolist()}
    return T, info


def split_mode(task: str, mode: str = "auto") -> str:
    m = SPLIT_AUTO[task] if mode == "auto" else mode
    assert m in ("question", "class") and not (task == "mod" and m == "class"), (task, mode)
    return m


def split(D: dict, frac: float, seed: int, mode: str = "auto"):
    """(perm, tr, te): perm[:len(tr)] = tr. question mode = toy_grok's split; class mode = the same draw over relation
    classes (D['cls']), questions of a class in table order."""
    mode = split_mode(D["task"], mode)
    N = len(D["X"])
    if mode == "question":
        perm = np.random.default_rng(seed).permutation(N)
        ntr = int(frac * N)
        tr, te = perm[:ntr], perm[ntr:]
    else:
        cls = D["cls"]
        pc = np.random.default_rng(seed).permutation(int(cls.max()) + 1)
        is_tr = np.isin(cls, pc[: int(frac * len(pc))])
        tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
        perm = np.concatenate([tr, te])
        assert not (set(cls[tr]) & set(cls[te])), "relation class split across train / held-out"
    rows = [tuple(r) for r in D["X"]]
    assert len(set(rows)) == N, "duplicate questions"
    s_tr, s_te = {rows[i] for i in tr}, {rows[i] for i in te}
    assert not (s_tr & s_te) and len(s_tr) + len(s_te) == N, "held-out question in training"
    if D["task"] != "mod":
        n = D["n_items"]
        Xtr = D["X"][tr]
        seen_in = set(Xtr[:, 0]) | set(Xtr[:, 2 if D["task"] == "dim" else 1])
        assert seen_in == set(range(n)), f"item never an input in training: {set(range(n)) - seen_in}"
        assert set(D["Y"][tr]) == set(range(n)), "item never an answer in training"
    return perm, tr, te


def fresh_mask(D: dict, tr: np.ndarray, te: np.ndarray) -> np.ndarray | None:
    """For each held-out question: True iff no other question of its relation class is in training."""
    if "cls" not in D:
        return None
    return ~np.isin(D["cls"][te], D["cls"][tr])


def negtwin_mask(D: dict, tr: np.ndarray, te: np.ndarray) -> np.ndarray | None:
    """dim: for each held-out question (a, op, b), True iff its negation twin (a, the other op, -b) is in training. The
    twin has the same answer (e_a + e_b = e_a - (-e_b)) but another relation class (re-audit 2026-10-17), so the class split
    leaves a lookup shortcut through a token-level negation map, like the commutativity the split removes."""
    if D["task"] != "dim":
        return None
    E = D["E"]
    key = {tuple(v): i for i, v in enumerate(E)}
    neg = np.array([key[tuple(-v)] for v in E])
    row = {tuple(r): i for i, r in enumerate(D["X"].tolist())}
    trs = set(tr.tolist())
    out = []
    for i in te:
        a, op, b, eq = D["X"][i].tolist()
        j = row.get((a, 55 - op, int(neg[b]), eq))   # 27 (mul) <-> 28 (div)
        assert j is not None and D["Y"][j] == D["Y"][i], ("negation twin missing or different answer", i)
        out.append(j in trs)
    return np.array(out, dtype=bool)


def majority_chance(D: dict, tr: np.ndarray, te: np.ndarray) -> float:
    """Held-out accuracy of always answering the most frequent training answer (>= 1 / n_out up to ties)."""
    maj = np.bincount(D["Y"][tr], minlength=D["n_out"]).argmax()
    return float((D["Y"][te] == maj).mean())


def decode(D: dict, i: int) -> str:
    """Question i as the model sees it (token ids and names), the scored position (the last token) in brackets."""
    x, y = D["X"][i], int(D["Y"][i])
    if D["task"] == "mod":
        names = [str(t) if t < D["p"] else "=" for t in x]; ans = str(y)
    elif D["task"] == "s4":
        names = [f"g{t:02d}" if t < 24 else "=" for t in x]; ans = f"g{y:02d}"
    else:
        nm = lambda t: "e(" + ",".join(f"{v:+d}" for v in D["E"][t]) + ")"
        names = [nm(t) if t < 27 else {27: "mul", 28: "div", 29: "="}[int(t)] for t in x]; ans = nm(y)
    return f"ids {x.tolist()} len {len(x)} | " + " ".join(names[:-1]) + f" ⟦{names[-1]}⟧ (pos {len(x) - 1}) -> {ans}"


def show_inputs(D: dict, tr: np.ndarray, te: np.ndarray, k: int = 3) -> str:
    return "\n".join([f"  train {decode(D, int(i))}" for i in tr[:k]] + [f"  eval  {decode(D, int(i))}" for i in te[:k]])


def closure(D: dict, tr: np.ndarray, te: np.ndarray) -> dict:
    if D["task"] == "s4":
        cells = [tuple(D["X"][i, :2]) for i in tr]
        held = [tuple(D["X"][i, :2]) for i in te]
        k1 = s4_closure(D["mul"], cells)
        k2 = s4_closure(D["mul"], cells, latin=True)
        return {"closure": float(np.mean([c in k1 for c in held])), "closure_latin": float(np.mean([c in k2 for c in held]))}
    if D["task"] == "dim":
        def row(i):
            a, op, b, _ = D["X"][i]
            r = np.zeros(27)
            r[a] += 1; r[b] += 1 if op == 27 else -1; r[D["Y"][i]] -= 1
            return r
        A = np.array([row(i) for i in tr])
        rk = np.linalg.matrix_rank(A, tol=1e-8)
        der_rank, der_lsq = [], []
        for i in te:
            r = row(i)
            der_rank.append(np.linalg.matrix_rank(np.vstack([A, r]), tol=1e-8) == rk)
            x = np.linalg.lstsq(A.T, r, rcond=None)[0]
            der_lsq.append(np.linalg.norm(A.T @ x - r) < 1e-6)
        assert der_rank == der_lsq, "rank test and least-squares residual disagree"
        Etrue = D["E"].astype(np.float64)
        assert np.abs(A @ Etrue).max() < 1e-9, "true exponents violate a training relation"
        return {"closure": float(np.mean(der_rank)), "nullity": int(27 - rk)}
    return {}


# ---------------------------------------------------------------------------------------------------- arms
def arm_spec(task: str, arm: str, win: str, width: int):
    """(target name or None, on(step) -> bool, window description)."""
    if ":" in arm:   # wall 2 (opt-in) shape arm '<kind>:<target>[@always|@t0-t1]', any task; else the cell window --win
        arm_kind(arm)
        target, _, w = arm.split(":", 1)[1].partition("@")
        if w:
            win = w if w == "always" else w.replace("-", ",")
        if win == "always":
            return target, (lambda s: True), "always"
        t0, t1 = (int(x) for x in win.split(","))
        assert 0 <= t0 < t1, win
        return target, (lambda s, t0=t0, t1=t1: t0 <= s < t1), [t0, t1]
    if task == "mod":   # toy_grok semantics
        target = "circle_shuf" if arm.endswith("_shuf") else "circle"
        base = arm[: -len("_shuf")] if arm.endswith("_shuf") else arm
        if base == "none":
            return None, (lambda s: False), None
        if base == "always":
            return target, (lambda s: True), "always"
        t0 = int(base[1:])
        return target, (lambda s, t0=t0: t0 <= s < t0 + width), [t0, t0 + width]
    if arm == "none":
        return None, (lambda s: False), None
    if "@" in arm:   # per-arm window override: R@always, R@0-8000
        arm, w = arm.split("@")
        win = w if w == "always" else w.replace("-", ",")
    if win == "always":
        return arm, (lambda s: True), "always"
    t0, t1 = (int(x) for x in win.split(","))
    assert 0 <= t0 < t1, win
    return arm, (lambda s, t0=t0, t1=t1: t0 <= s < t1), [t0, t1]


def arm_target(arm: str) -> str:
    return arm.split("@")[0].split(":")[-1]


CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]   # at import: the code that is running


def code_sha() -> str:
    return CODE_SHA


def target_table(D: dict, T: dict, W_init: np.ndarray | None = None) -> tuple[list, np.ndarray]:
    names = list(T)
    F = [T[k] for k in names]
    if W_init is not None:
        names = names + ["W_E_init"]; F = F + [W_init]
    M = np.array([[np_cka(A, B) for B in F] for A in F])
    return names, M


def check_targets(D: dict, T: dict, info: dict) -> None:
    """Intent asserts on the targets (ranks, exact orthogonalities, spectra)."""
    task = D["task"]
    rk = {k: centred_rank(v) for k, v in T.items()}
    if task == "s4":
        assert rk == {"R": 9, "R_sign": 9, "absR": 4, "specrand": 9, "shuf": 9}, rk
        for a, b in (("R", "R_sign"), ("R", "absR"), ("R_sign", "absR")):
            assert np_cka(T[a], T[b]) < 1e-9, (a, b)   # distinct irreps: orthogonal matrix coefficients (Schur)
        s = np.linalg.svd(T["R"] - T["R"].mean(0), compute_uv=False)
        assert np.allclose(s, np.sqrt(8)), s           # Schur: sum_g R_ij R_kl = (24 / 3) d_ik d_jl
    if task == "dim":
        assert rk == {"expo": 3, "quot": 2, "onehot": 6, "specrand": 3, "specrand_oh": 6, "shuf": 3}, rk
        assert abs(np_cka(T["expo"], T["quot"]) - 2 / math.sqrt(6)) < 1e-9   # 2 of 3 equal coordinate grams
        # onehot's centred column span contains expo's (e_i = [e_i = 1] - [e_i = -1]): the rank-6 superset of expo
        P = np.linalg.svd(T["onehot"] - T["onehot"].mean(0), full_matrices=False)[0][:, :6]
        Ec = T["expo"] - T["expo"].mean(0)
        assert np.allclose(P @ (P.T @ Ec), Ec), "expo not inside onehot's span"
        so = lambda F: np.linalg.svd(F - F.mean(0), compute_uv=False)[:6]
        assert np.allclose(so(T["specrand_oh"]), so(T["onehot"])), "specrand_oh spectrum"
    if task != "mod":
        F = T[TRUE[task]]
        assert np.allclose(np.linalg.svd(T["specrand"] - T["specrand"].mean(0), compute_uv=False)[:rk[TRUE[task]]],
                           np.linalg.svd(F - F.mean(0), compute_uv=False)[:rk[TRUE[task]]])
        assert sorted(map(tuple, T["shuf"].tolist())) == sorted(map(tuple, F.tolist())), "shuf is a permutation of the true rows"


# ---------------------------------------------------------------------------------------------------- wall 2: shape-only prior (opt-in)
def line_target(n: int, r: int) -> np.ndarray:
    """'line' = an open chain of n items in r dims: path-graph modes cos(pi k (i + 1/2) / n) / k, k = 1..r (centred,
    orthogonal; centred spectrum (n / 2) / k^2, non-constant norms). The wrong-shape control of the same rank."""
    i = np.arange(n)
    F = np.stack([np.cos(np.pi * k * (i + 0.5) / n) / k for k in range(1, r + 1)], 1)
    assert centred_rank(F) == r and np.allclose(F.mean(0), 0, atol=1e-12)
    assert np.allclose(np.linalg.svd(F, compute_uv=False) ** 2, (n / 2) / np.arange(1, r + 1) ** 2), "line spectrum"
    return F


def sphere_rank(task: str, r: int) -> int:
    """Rank of the 'sphere' control for a true shape of centred rank r: mod r + 1 (odd: no union of Fourier circles,
    which come in pairs); s4 12 for R's 9 (achievable isotypic sums of the centred regular representation are the
    subset sums of 1, 4, 9, 9: 12 is not one; 10 = 1 + 9 would be); otherwise r + 1."""
    if task == "s4" and r == 9:
        return 12
    return r + 1


def sphere_target(n: int, r: int) -> np.ndarray:
    """'sphere' = a centred configuration of n items in rank r with a flat centred spectrum (r equal eigenvalues) and
    constant norms (1): an equal-norm tight frame. Alternating projections (centre + polar = flat; row-normalise =
    constant norm) from a fixed numpy draw; deterministic; converges in < 100 iterations for the sizes used here."""
    F = np.random.default_rng(31_000 + 100 * n + r).standard_normal((n, r))
    for _ in range(5000):
        F = F - F.mean(0)
        U, _s, Vt = np.linalg.svd(F, full_matrices=False)
        F = U @ Vt * np.sqrt(n / r)                    # flat centred spectrum (U is orthogonal to the constant)
        nr = np.linalg.norm(F, axis=1)
        if np.abs(nr - 1).max() < 1e-12:
            break
        F = F / nr[:, None]
    lam = np.linalg.svd(F - F.mean(0), compute_uv=False) ** 2
    nu = ((F - F.mean(0)) ** 2).sum(1)
    assert centred_rank(F) == r and lam.max() / lam.min() - 1 < 1e-9 and nu.max() / nu.min() - 1 < 1e-9, (n, r, lam, nu)
    return F


def shape_feats(D: dict, T: dict) -> dict:
    """The task's shape targets (raw features): its own targets allowed by SHAPE_OK, 'line' of the true shape's rank r,
    'sphere' of rank sphere_rank(task, r)."""
    F = {k: T[k] for k in SHAPE_OK[D["task"]] if k in T}
    r = centred_rank(T[SHAPE_TRUE[D["task"]]])
    F["line"] = line_target(D["n_items"], r)
    F["sphere"] = sphere_target(D["n_items"], sphere_rank(D["task"], r))
    return F


def shape_readout_feats(D: dict, T: dict) -> dict:
    """--shape_log readouts: the shape targets, plus for mod the freq-1 circle and the freq 1-5 torus whatever --freqs is
    (so a none run logs its distance to both mod variants; audit 2026-10-19)."""
    F = shape_feats(D, T)
    if D["task"] == "mod":
        ang = 2 * np.pi * np.arange(D["p"]) / D["p"]
        circ = lambda ks: np.concatenate([np.stack([np.cos(k * ang), np.sin(k * ang)], 1) for k in ks], 1)
        F.update({"fourier1": circ((1,)), "fourier1-5": circ((1, 2, 3, 4, 5))})
    return F


def shape_desc(F: np.ndarray) -> dict:
    """Permutation-invariant descriptor of a target configuration (float64): sorted centred-Gram spectrum (descending,
    clipped at 0), sorted squared centred norms, and the common sorted Gram row if every row has the same one."""
    G = np_gram(F)
    lam = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None).copy()
    rows = np.sort(G, 1)
    vt = bool(np.abs(rows - rows[0]).max() <= 1e-9 * np.trace(G) / len(G))
    return {"lam": lam, "nu": np.sort(np.diag(G)), "prof": rows[0].copy() if vt else None}


def shape_tensors(F: np.ndarray, dev) -> dict:
    d = shape_desc(F)
    return {k: (None if v is None else torch.tensor(v, dtype=torch.float32, device=dev)) for k, v in d.items()}


def shape_terms(H: torch.Tensor, S: dict, terms) -> dict:
    """The shape loss terms of item embeddings H (n x d) toward descriptor S; each in [0, 1] (module docstring)."""
    G = centred_gram(H)
    # the eigendecomposition runs on the CPU in float64 (LAPACK dsyevd), whatever the arm's device (audit 2026-10-19:
    # cuSOLVER float32 on a Gram with ~111 near-zero eigenvalues was untested on the GPU); a no-op for float64 CPU input
    G64 = G.to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(G64).all()):   # a diverged run (e.g. lr 3e-2): eigvalsh would raise; NaN terms keep it running
        return {k: G.sum() * float("nan") for k in terms}   # (pre-registered: divergence = not learning, reported)
    out = {}
    if "spec" in terms:
        lam = torch.linalg.eigvalsh(G64).flip(0).to(device=G.device, dtype=G.dtype)
        out["spec"] = 1 - (lam * S["lam"]).sum() / (lam.norm() * S["lam"].norm() + 1e-12)
    if "norm" in terms:
        nu = torch.sort(torch.diagonal(G)).values
        out["norm"] = 1 - (nu * S["nu"]).sum() / (nu.norm() * S["nu"].norm() + 1e-12)
    if "prof" in terms:
        assert S["prof"] is not None, "prof needs a vertex-transitive target"
        rows = torch.sort(G, dim=1).values
        out["prof"] = 1 - (rows @ S["prof"]).sum() / (rows.norm() * S["prof"].norm() * len(G) ** 0.5 + 1e-12)
    return out


def shape_loss(H: torch.Tensor, S: dict, terms) -> torch.Tensor:
    return sum(shape_terms(H, S, terms).values())


def shape_selfcheck(W0: torch.Tensor, F: np.ndarray, S: dict, terms, seed: int, dev) -> dict:
    """Asserted at the start of every shape arm, on its device: the loss is invariant under an item permutation of the
    initial embeddings, ~0 for the target and for a permutation of it, and > 0 at the init (numpy rng: torch's untouched)."""
    q = torch.tensor(np.random.default_rng(10_000 + seed).permutation(len(F)), device=dev)
    Ft = torch.tensor(F, dtype=torch.float32, device=dev)
    with torch.no_grad():
        v = {"init": float(shape_loss(W0, S, terms)), "init_perm": float(shape_loss(W0[q], S, terms)),
             "target": float(shape_loss(Ft, S, terms)), "target_perm": float(shape_loss(Ft[q], S, terms))}
    assert abs(v["init"] - v["init_perm"]) < 1e-4, ("shape loss not permutation-invariant", v)
    assert abs(v["target"]) < 1e-4 and abs(v["target_perm"]) < 1e-4, ("shape loss of the target not ~0", v)
    assert v["init"] > 1e-3, ("shape loss already ~0 at init", v)
    return v


def arm_kind(arm: str) -> str:
    """'cka' for every arm that existed before wall 2; 'shape' / 'shapeprof' for '<kind>:<target>[@...]'."""
    if ":" not in arm:
        return "cka"
    k = arm.split(":", 1)[0]
    assert k in SHAPE_KINDS, arm
    return k


# ---------------------------------------------------------------------------------------------------- training
def run_arm(D, arm, seed, tr_np, te_np, T, args, dev, extra):
    target, on, window = arm_spec(D["task"], arm, args.win, args.width)
    kind = arm_kind(arm)                                   # 'cka' = every pre-wall-2 arm (its path is unchanged)
    shape_log = bool(getattr(args, "shape_log", False))
    n_items = D["n_items"]
    X_all = torch.tensor(D["X"], device=dev)
    Y_all = torch.tensor(D["Y"], device=dev)
    tr, te = torch.tensor(tr_np, device=dev), torch.tensor(te_np, device=dev)
    K = {k: centred_gram(torch.tensor(v, dtype=torch.float32, device=dev)) for k, v in T.items()}
    torch.manual_seed(seed)
    model = OneLayer(D["n_vocab"], D["n_out"], D["T"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    with torch.no_grad():
        cka_init = {k: float(cka(model.W_E[:n_items], K[k])) for k in K}
        W_init = model.W_E[:n_items].detach().cpu().double().numpy()
    SH = shape_chk = None
    if kind != "cka" or shape_log:   # wall 2 (opt-in): permutation-invariant shape descriptors of the shape targets
        SF = shape_readout_feats(D, T) if shape_log else shape_feats(D, T)
        SH = {k: shape_tensors(v, dev) for k, v in SF.items()}
    if kind != "cka":
        terms = SHAPE_KINDS[kind]
        assert target in SH and target in SHAPE_OK[D["task"]], (arm, target, sorted(SH))
        shape_chk = shape_selfcheck(model.W_E[:n_items].detach(), SF[target], SH[target], terms, seed, dev)
    if D["task"] == "s4":
        coset = torch.tensor(D["coset"], device=dev)
        noncomm_te = torch.tensor(D["noncomm"][te_np], device=dev)
    if D["task"] == "dim":
        E = torch.tensor(D["E"], device=dev)
        op_te = X_all[te, 1]
        negtwin_te = torch.tensor(negtwin_mask(D, tr_np, te_np), device=dev)
    fm = fresh_mask(D, tr_np, te_np)
    fresh_te = None if fm is None else torch.tensor(fm, device=dev)
    curve = []
    t_start = time.time()
    stopped = None
    for s in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (s + 1) / 10)   # 2026-10-18: was a hard-coded 1e-3, which made --lr inert
        ce = Fn.cross_entropy(model(X_all[tr]), Y_all[tr])
        if kind == "cka":
            loss = ce + (args.lam * (1 - cka(model.W_E[:n_items], K[target])) if on(s) else 0.0)
        else:   # wall 2: shape-only prior (no assignment)
            loss = ce + (args.lam * shape_loss(model.W_E[:n_items], SH[target], terms) if on(s) else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % args.eval_every == 0 or s == args.steps - 1:
            with torch.no_grad():
                acc = lambda idx: float((model(X_all[idx]).argmax(-1) == Y_all[idx]).float().mean())
                if D["task"] == "mod":   # toy_grok's record, in its order
                    c = {"step": s, "ce": float(ce), "train": acc(tr), "test": acc(te),
                         "cka_circle": float(cka(model.W_E[:n_items], K["circle"])), "fourier_top5": fourier_top(model.W_E[:n_items])}
                else:
                    c = {"step": s, "ce": float(ce), "train": acc(tr), "test": acc(te)}
                lt = model(X_all[te])
                pred, y = lt.argmax(-1), Y_all[te]
                wrong = pred != y
                c["test_ce"] = float(Fn.cross_entropy(lt, y))
                c["lr_now"] = float(opt.param_groups[0]["lr"])   # the lr actually in use (2026-10-18 check)
                assert s < 10 or abs(c["lr_now"] - args.lr) <= 1e-12 * max(1.0, args.lr), (c["lr_now"], args.lr)
                c["prior_on"] = bool(on(s))
                ck = {k: float(cka(model.W_E[:n_items], K[k])) for k in K}
                c.update({f"cka_{k}": v for k, v in ck.items() if f"cka_{k}" not in c})
                if kind == "cka":
                    c["geo"] = None if target is None else 1 - ck[target]
                else:   # a shape arm's geo = its own objective; the cka_* keys still show the assignment
                    tv = shape_terms(model.W_E[:n_items], SH[target], terms)
                    c["geo"] = float(sum(tv.values()))
                    c["shape_terms"] = {k: float(v) for k, v in tv.items()}
                c["n_wrong"] = int(wrong.sum())
                if fresh_te is not None:   # held-out questions with no class-mate in training
                    c["test_fresh"] = float((pred[fresh_te] == y[fresh_te]).float().mean()) if fresh_te.any() else None
                if D["task"] == "s4":   # wrong answers inside the correct V4 coset (chance 3/23)
                    c["coset_share"] = float((coset[pred[wrong]] == coset[y[wrong]]).float().mean()) if wrong.any() else None
                    c["test_noncomm"] = float((pred[noncomm_te] == y[noncomm_te]).float().mean())
                    c["test_comm"] = float((pred[~noncomm_te] == y[~noncomm_te]).float().mean())
                if D["task"] == "dim":  # wrong answers that differ from the truth only in the dropped exponent (chance 2/26)
                    keep = [i for i in range(3) if i != DROP]
                    Ep, Ey = E[pred[wrong]], E[y[wrong]]
                    c["drop_share"] = float((Ep[:, keep] == Ey[:, keep]).all(1).float().mean()) if wrong.any() else None
                    c["test_mul"] = float((pred[op_te == 27] == y[op_te == 27]).float().mean())
                    c["test_div"] = float((pred[op_te == 28] == y[op_te == 28]).float().mean())
                    nt = negtwin_te
                    c["n_negtwin"] = int(nt.sum())
                    c["test_negtwin"] = float((pred[nt] == y[nt]).float().mean()) if nt.any() else None
                    c["test_nonegtwin"] = float((pred[~nt] == y[~nt]).float().mean()) if (~nt).any() else None
                if shape_log:   # wall 2 readout (logging only): spec + norm distance to every shape target
                    c.update({f"shapeL_{k}": float(shape_loss(model.W_E[:n_items], v, SHAPE_KINDS["shape"])) for k, v in SH.items()})
                curve.append(c)
            if args.stop_at_grok and c["test"] >= 0.95:   # pilots only (budget rule): the curve ends here
                stopped = s + 1
                break
    sec = time.time() - t_start
    first = lambda key, thr: next((c["step"] for c in curve if c[key] >= thr), None)
    res = {"task": D["task"], "arm": arm, "target": target, "window": window, "seed": seed, "lam": args.lam, "lr": args.lr, "wd": args.wd,
           "frac": args.frac, "steps": args.steps, "n_train": len(tr_np), "n_test": len(te_np),
           "t_mem": first("train", 0.99), "t_grok": first("test", 0.95), "censored_at": args.steps,
           "final_test": curve[-1]["test"], "final_train": curve[-1]["train"], "cka_init": cka_init,
           "sec": sec, "ms_per_step": 1000 * sec / (stopped or args.steps), "stopped_at": stopped, "eval_every": args.eval_every,
           "n_fresh": None if fm is None else int(fm.sum()),
           "device": str(dev), "torch": torch.__version__, "threads": torch.get_num_threads(),
           "code_sha": code_sha(), **extra, "curve": curve}
    if D["task"] == "mod":
        res.update({"width": args.width, "p": D["p"], "freqs": D["freqs"]})
    if kind != "cka":   # wall 2: what the loop consumed
        res.update({"prior_kind": kind, "shape_terms": list(terms), "shape_check": shape_chk,
                    "shape_target_rank": centred_rank(SF[target]), "shape_eig": "cpu float64"})
    if shape_log:
        res.update({"shape_log": True, "shape_readouts": list(SH)})
    return res, W_init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("mod", "s4", "dim"), default="mod")
    ap.add_argument("--out")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--lam", type=float, default=1.0)
    # 2026-10-17 (user: make "not learned without the prior" harder): the optimiser of toy_grok by default
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--win", default=WIN_DEFAULT, help="s4/dim: prior window 't0,t1' (steps [t0, t1)) or 'always'")
    ap.add_argument("--width", type=int, default=2000, help="mod: toy_grok window width for arms w<t0>")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--freqs", default="1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = leave the default)")
    ap.add_argument("--check", action="store_true", help="print targets, CKA matrix, closure, asserts; no training")
    ap.add_argument("--stop_at_grok", action="store_true", help="pilot only: stop a run at its first eval with test >= 0.95")
    ap.add_argument("--summarize", default=None, help="summarise the run JSONs in this directory and exit")
    ap.add_argument("--split", choices=("auto", "question", "class"), default="auto",
                    help="auto = class for dim, question for s4 / mod (see the module docstring)")
    # wall 2 (2026-10-19, opt-in; off = the path above, bitwise)
    ap.add_argument("--shape_log", action="store_true", help="every arm logs shapeL_<target> (spec + norm) each eval; logging only")
    ap.add_argument("--summarize_wall", default=None, help="the pre-registered wall-2 reading over every cell under this root")
    ap.add_argument("--mod_twins", action="store_true",
                    help="mod: relation classes = unordered pairs, so test_fresh / n_fresh (held-out questions whose commuted "
                         "twin (b, a) is not in training) are logged; the split stays toy_grok's question split (logging only)")
    args = ap.parse_args()
    if args.summarize:
        summarize(Path(args.summarize)); return
    if args.summarize_wall:
        summarize_wall(Path(args.summarize_wall)); return
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    D = build_task(args.task, args.p, tuple(int(k) for k in args.freqs.split(",")))
    if args.mod_twins:   # wall 2 (opt-in, logging only): commutativity classes for mod's test_fresh
        assert args.task == "mod" and split_mode(args.task, args.split) == "question", "--mod_twins: mod, question split only"
        D["cls"] = class_ids([(min(a, b), max(a, b)) for a, b in D["X"][:, :2].tolist()])
        assert D["cls"].max() + 1 == D["p"] * (D["p"] + 1) // 2
    arms = (args.arms or ARMS[args.task]).split(",")
    if args.task != "mod":
        names = {"none"} | set(D["targets"]) | {"specrand", "shuf"} | ({"specrand_oh"} if args.task == "dim" else set())
        assert {arm_target(a) for a in arms if ":" not in a} <= names and not any(a.startswith("none@") for a in arms), arms
        for a in arms:
            arm_spec(args.task, a, args.win, args.width)   # parses every window before any training
    for a in arms:   # wall 2 shape arms (any task): kind, target and window checked before any training
        if ":" in a:
            assert arm_kind(a) in SHAPE_KINDS and arm_spec(args.task, a, args.win, args.width)[0] in SHAPE_OK[args.task], a
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.check:
        check(D, args, seeds); return
    smode = split_mode(args.task, args.split)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for seed in seeds:
        perm, tr, te = split(D, args.frac, seed, smode)
        T, info = seed_targets(D, seed)
        if args.task != "mod":
            check_targets(D, T, info)
        extra = {"split_sha": hashlib.sha256(perm[: len(tr)].tobytes()).hexdigest()[:12], **closure(D, tr, te),
                 **{k: v for k, v in info.items() if k != "shuf_perm"}}
        if args.task != "mod":
            extra.update({"split": smode, "chance_majority": majority_chance(D, tr, te)})
        if args.mod_twins:
            extra["mod_twins"] = True
        if args.task != "mod":
            extra["shuf_perm"] = info["shuf_perm"]
            names, M = target_table(D, T)
            extra["target_cka"] = {"names": names, "matrix": M.round(6).tolist()}
        print(f"[{args.task} frac {args.frac} seed {seed}] n_train {len(tr)} n_test {len(te)} "
              + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in extra.items()
                         if k not in ("shuf_perm", "target_cka", "split_sha")), flush=True)
        print(show_inputs(D, tr, te), flush=True)
        for arm in arms:
            path = out / f"run_{arm}_seed{seed}.json"
            if path.exists():
                continue
            res, _ = run_arm(D, arm, seed, tr, te, T, args, dev, extra)
            tmp = out / f".tmp_{path.name}.{os.getpid()}"   # atomic: a reader / a preemption never sees a partial JSON
            tmp.write_text(json.dumps(res))
            os.replace(tmp, path)
            print(f"{arm} seed{seed}: t_mem {res['t_mem']} t_grok {res['t_grok']} final test {res['final_test']:.3f} "
                  f"({res['ms_per_step']:.2f} ms/step, {time.time() - t0:.0f}s)", flush=True)
    print(f"done; summary: python {Path(__file__).name} --summarize {out}  (after every process of the cell)", flush=True)


def check(D: dict, args, seeds: list) -> None:
    """Targets, the CKA matrix (targets + the seed's initial W_E), closure fractions; every assert runs."""
    print(f"task {D['task']}: {len(D['X'])} questions, {D['n_items']} items, vocab {D['n_vocab']}, T {D['T']}")
    for seed in seeds:
        T, info = seed_targets(D, seed)
        if D["task"] != "mod":
            check_targets(D, T, info)
        torch.manual_seed(seed)
        W0 = OneLayer(D["n_vocab"], D["n_out"], D["T"]).W_E[: D["n_items"]].detach().double().numpy()
        names, M = target_table(D, T, W0)
        print(f"seed {seed}: centred ranks " + ", ".join(f"{k} {centred_rank(v)}" for k, v in T.items())
              + "; " + ", ".join(f"{k} {v:.3f}" for k, v in info.items() if k != "shuf_perm"))
        if seed == seeds[0]:
            w = max(len(n) for n in names)
            print(" " * (w + 1) + " ".join(f"{n:>9}" for n in names))
            for n, row in zip(names, M):
                print(f"{n:>{w}} " + " ".join(f"{x:9.3f}" for x in row))
            if args.shape_log:   # wall 2: shape loss (spec + norm [prof]) of each configuration (row) toward each target (col)
                SF = shape_feats(D, T)
                SF_rows = {**SF, "W_E_init": W0}
                print("shape loss spec + norm [+ prof where the target is vertex-transitive], row = configuration, col = target")
                for rn, Fr in SF_rows.items():
                    H = torch.tensor(Fr, dtype=torch.float32)
                    cells = []
                    for cn, Fc in SF.items():
                        S = shape_tensors(Fc, "cpu")
                        t = shape_terms(H, S, SHAPE_KINDS["shapeprof" if S["prof"] is not None else "shape"])
                        cells.append(f"{cn} {float(t['spec'] + t['norm']):.3f}" + (f" [{float(t['prof']):.3f}]" if "prof" in t else ""))
                    print(f"  {rn:>9}: " + ", ".join(cells))
        smode = split_mode(D["task"], args.split)
        for frac in sorted({0.3, 0.6, args.frac}):
            perm, tr, te = split(D, frac, seed, smode)
            fm = fresh_mask(D, tr, te)
            print(f"  frac {frac} ({smode} split): n_train {len(tr)} n_test {len(te)} {closure(D, tr, te)}"
                  + ("" if fm is None else f" n_fresh {int(fm.sum())}")
                  + ("" if D["task"] == "mod" else f" majority-chance {majority_chance(D, tr, te):.3f}"))


# ---------------------------------------------------------------------------------------------------- summary
def cens(r) -> float:
    return math.inf if r["t_grok"] is None else float(r["t_grok"])


def cmedian(rs) -> float:
    v = sorted(cens(r) for r in rs)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def mwu_p(x: list, y: list) -> float:
    """Exact one-sided permutation p of the Gehan / Mann-Whitney U that x is SMALLER (earlier grok) than y;
    censored values (inf) tie with each other (common censoring time = budget)."""
    def U(a, b):
        return sum((ai < bj) + 0.5 * (ai == bj) for ai in a for bj in b)
    pool, n = list(x) + list(y), len(x)
    u0 = U(x, y)
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        a = [pool[i] for i in idx]; b = [pool[i] for i in range(len(pool)) if i not in idx]
        hits += U(a, b) >= u0 - 1e-9; tot += 1
    return hits / tot


def median_vals(v: list) -> float:
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def n_evals(steps: int, every: int) -> int:
    """Number of evals in a full run: every `every` steps from step 0, plus the last step."""
    return len(range(0, steps, every)) + (0 if (steps - 1) % every == 0 else 1)


def auc(r, key: str = "test") -> float:
    """CO-PRIMARY: mean held-out accuracy over every eval of the budget (the normalised area under the held-out
    learning curve). It registers a partial rescue that never reaches 0.95 (audit 2026-10-16: S4 0.6, R window plateaus
    at 0.72-0.77 vs specrand 0.01, which t_grok scores as a censored tie). A run stopped at its grok (pilot
    --stop_at_grok) carries its last value forward to the full eval count; a main run must be complete."""
    v = [c[key] for c in r["curve"]]
    n = n_evals(r["steps"], r.get("eval_every", 100))
    if r.get("stopped_at"):
        v = v + [v[-1]] * (n - len(v))
    assert len(v) == n, f"incomplete curve: {len(v)} of {n} evals ({r['arm']} seed {r['seed']})"
    return float(np.mean(v))


def t_half(r):
    return next((c["step"] for c in r["curve"] if c["test"] >= 0.5), None)


N_OUT = {"s4": 24, "dim": 27}


def chance_of(rs: list, task: str) -> float:
    """Floor reference: the larger of 1 / n_out and the median majority-of-train baseline on held-out."""
    base = 1 / N_OUT.get(task, rs[0].get("p", 113))
    cm = [r["chance_majority"] for r in rs if r.get("chance_majority") is not None]
    return max(base, float(np.median(cm))) if cm else base


def compare(runs: dict, a: str, b: str, chance: float) -> dict | None:
    """Arm a vs arm b, paired by seed, on the two CO-PRIMARIES (PRE-REGISTERED, runs/toystruct/AUDIT.md):
    t_grok: censored median a < b, a groks strictly earlier on >= ceil(0.8 n) seeds, exact one-sided Gehan / MWU p < ALPHA;
    AUC:    median a - median b >= AUC_MIN, AUC a > AUC b on >= ceil(0.8 n) seeds, exact one-sided MWU p < ALPHA.
    ALPHA = 0.025 each (Bonferroni over the two). Verdict: PASS (t_grok / AUC / both); else 'floor' when both arms are
    censored on >= ceil(0.8 n) seeds and both median AUCs <= chance + AUC_MIN (nothing generalises: uninformative);
    else 'not shown at n' (low power: never read as 'no effect')."""
    if a not in runs or b not in runs:
        return None
    sa, sb = {r["seed"]: r for r in runs[a]}, {r["seed"]: r for r in runs[b]}
    seeds = sorted(set(sa) & set(sb))
    ga, gb = [cens(sa[s]) for s in seeds], [cens(sb[s]) for s in seeds]
    ua, ub = [auc(sa[s]) for s in seeds], [auc(sb[s]) for s in seeds]
    t = compare_values(ga, gb, ua, ub, chance)
    return {"a": a, "b": b, "n": t["n"], "seeds": seeds, **{k: t[k] for k in ("grok", "auc", "pass", "floor", "verdict")}}


def compare_values(ga: list, gb: list, ua: list, ub: list, chance: float) -> dict:
    """compare()'s decision on per-seed values (paired lists): censored t_grok (inf = never) and AUC. Shared by the
    F84 summary and the wall-2 reading (which feeds it t_grok / AUC over the first B steps, summarize_wall)."""
    n = len(ga); need = math.ceil(0.8 * n)
    assert len(gb) == len(ua) == len(ub) == n, (len(ga), len(gb), len(ua), len(ub))
    g = {"med_a": median_vals(ga), "med_b": median_vals(gb), "wins": sum(x < y for x, y in zip(ga, gb)), "p": mwu_p(ga, gb)}
    g["pass"] = bool(g["med_a"] < g["med_b"] and g["wins"] >= need and g["p"] < ALPHA)
    u = {"med_a": median_vals(ua), "med_b": median_vals(ub), "wins": sum(x > y for x, y in zip(ua, ub)),
         "p": mwu_p([-x for x in ua], [-x for x in ub]), "per_seed_a": ua, "per_seed_b": ub}
    # re-audit 2026-10-17: a lift that stays at or below the cell's own floor line (chance + AUC_MIN) is not a pass
    u["pass"] = bool(u["med_a"] - u["med_b"] >= AUC_MIN and u["med_a"] > chance + AUC_MIN and u["wins"] >= need
                     and u["p"] < ALPHA)
    floor = bool(not g["pass"] and not u["pass"] and sum(map(math.isinf, ga)) >= need and sum(map(math.isinf, gb)) >= need
                 and max(u["med_a"], u["med_b"]) <= chance + AUC_MIN)
    verdict = ("PASS (t_grok + AUC)" if g["pass"] and u["pass"] else "PASS (t_grok)" if g["pass"] else
               "PASS (AUC)" if u["pass"] else "floor: uninformative" if floor else f"not shown at n={n}")
    return {"n": n, "grok": g, "auc": u, "pass": bool(g["pass"] or u["pass"]), "floor": floor, "verdict": verdict}


def share_in(r, key: str, lo: int, hi: int):
    """Pooled share over the evals with lo <= step <= hi of the wrong test answers (weights = n_wrong)."""
    num = den = 0.0
    for c in r["curve"]:
        if lo <= c["step"] <= hi and c.get(key) is not None:
            num += c[key] * c["n_wrong"]; den += c["n_wrong"]
    return (num / den if den else None), int(den)


def rule_a(tests: dict, task: str, qualifies: bool, suffix: str = "") -> str:
    keys = [f"R{suffix}<specrand{suffix}", f"R_sign{suffix}<specrand{suffix}"] if task == "s4" else [f"expo{suffix}<specrand{suffix}"]
    T = [tests.get(k) for k in keys]
    if any(t is None for t in T):
        return "arms missing"
    if not qualifies:
        return "untestable at this fraction (none's censored median t_grok < 4000: none groks early); not a negative"
    if all(t["pass"] for t in T):
        return "PASS: " + ("group structure helps (R and R_sign both beat specrand)" if task == "s4" else "linear structure helps")
    if task == "s4" and sum(t["pass"] for t in T) == 1:
        return "irrep-specific (only one of R / R_sign beats specrand): no claim"
    if all(t["floor"] for t in T):
        return "uninformative (floor: neither the prior arms nor specrand generalise)"
    return f"not shown at n={min(t['n'] for t in T)} (low power; not evidence against)"


def rule_c(runs: dict, tests: dict, task: str, suffix: str = "") -> str:
    true, need = TRUE[task] + suffix, None
    if "none" not in runs or true not in runs:
        return "arms missing"
    n = len(runs["none"]); need = math.ceil(0.8 * n)
    nf = sum(r["t_grok"] is None for r in runs["none"]); tg = sum(r["t_grok"] is not None for r in runs[true])
    t = tests.get(f"{true}<none")
    sr = runs.get("specrand" + suffix)
    sr_rescues = sr is not None and (sum(r["t_grok"] is not None for r in sr) >= need
                                     or bool(tests.get(f"specrand{suffix}<none", {}).get("auc", {}).get("pass")))
    narrow = "; specrand also rescues -> narrows to 'any consistent prior of this rank'" if sr_rescues else ""
    head = f"none failed {nf}/{n}, {true} grokked {tg}/{len(runs[true])}: "
    if nf < need:
        return head + "none learns here: not the (c) regime"
    if tg >= need:
        return head + "RESCUE (without the prior it is not learned within the budget)" + narrow
    if t and t["auc"]["pass"]:
        return head + f"PARTIAL RESCUE (held-out AUC above none's, AUC median {t['auc']['med_a']:.2f} vs {t['auc']['med_b']:.2f}; no grok)" + narrow
    if t and t["floor"]:
        return head + "floor: uninformative (the prior arm does not generalise either)"
    return head + f"not shown at n={n}"


def summarize(out: Path) -> None:
    rs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    if not rs:
        print("no runs in", out); return
    task = rs[0].get("task", "mod")
    for k in ("task", "frac", "steps", "lam"):   # one cell per directory
        assert len({r.get(k) for r in rs}) == 1, f"mixed {k} in {out}: {sorted({str(r.get(k)) for r in rs})}"
    assert len({r.get("eval_every", 100) for r in rs}) == 1 and len({r.get("split") for r in rs}) == 1
    runs = {}
    for r in rs:
        runs.setdefault(r["arm"], []).append(r)
    base_order = ARMS.get(task, "").split(",")
    order = [a for a in base_order if a in runs] + sorted(set(runs) - set(base_order))
    budget = rs[0]["steps"]
    pilot = any(r.get("stopped_at") for r in rs)
    chance = chance_of(rs, task) if task in TRUE else None
    L = [f"# toy_struct {task}, frac {rs[0]['frac']}, {rs[0].get('split', 'question')} split, budget {budget} steps, lam {rs[0]['lam']}, "
         f"cell windows {sorted({str(r.get('window')) for r in rs})}\n\n"]
    if pilot:
        L.append("PILOT (--stop_at_grok): AUC carries the value at the stop forward; not a result\n\n")
    if chance is not None:
        L.append(f"chance (max of 1/n_out and the majority-of-train baseline): {chance:.3f}; floor line = chance + {AUC_MIN} = {chance + AUC_MIN:.3f}\n\n")
    L += ["| arm | seeds | t_mem median | t_grok censored median | grokked | AUC median (co-primary) | t_0.5 median | final test mean "
          "| final test_fresh mean | CKA own target @2000 / final |\n", "|---|---|---|---|---|---|---|---|---|---|\n"]
    summ = {"task": task, "frac": rs[0]["frac"], "budget": budget, "split": rs[0].get("split"), "pilot": pilot, "chance": chance,
            "alpha_per_coprimary": ALPHA, "auc_min": AUC_MIN, "arms": {}, "tests": {}}
    for A in order:
        R_ = runs[A]
        tm = sorted(r["t_mem"] if r["t_mem"] is not None else math.inf for r in R_)
        cm = cmedian(R_)
        own = R_[0].get("target")
        ck = ""
        if own:   # wall 2 shape arms: their own objective (shape loss, 0 = on target) instead of the CKA
            key = f"cka_{own}" if R_[0].get("prior_kind", "cka") == "cka" else "geo"
            at = [next((c[key] for c in r["curve"] if c["step"] >= 1999), float("nan")) for r in R_]
            fi = [r["curve"][-1][key] for r in R_]
            ck = f"{np.mean(at):.2f} / {np.mean(fi):.2f}" + ("" if key != "geo" else " (shape loss)")
        ng = sum(r["t_grok"] is not None for r in R_)
        au = [auc(r) for r in R_]
        th = median_vals([math.inf if t_half(r) is None else t_half(r) for r in R_])
        fr = [r["curve"][-1].get("test_fresh") for r in R_]
        fr = [x for x in fr if x is not None]
        L.append(f"| {A} | {len(R_)} | {tm[len(tm) // 2]} | {'> ' + str(budget) if cm == math.inf else int(cm)} | {ng}/{len(R_)} "
                 f"| {median_vals(au):.3f} | {'> ' + str(budget) if th == math.inf else int(th)} "
                 f"| {np.mean([r['final_test'] for r in R_]):.3f} | {f'{np.mean(fr):.3f}' if fr else '-'} | {ck} |\n")
        summ["arms"][A] = {"n": len(R_), "t_grok_cmedian": None if cm == math.inf else cm, "grokked": ng,
                           "t_grok": {r["seed"]: r["t_grok"] for r in R_}, "auc": {r["seed"]: a for r, a in zip(R_, au)},
                           "final_test": {r["seed"]: r["final_test"] for r in R_},
                           "final_test_fresh": {r["seed"]: r["curve"][-1].get("test_fresh") for r in R_}}
    if task in TRUE:
        late = "none" in runs and budget >= LATE and cmedian(runs["none"]) >= LATE   # undefined below a LATE budget
        summ["qualifies_a"] = bool(late)
        L.append(f"\nqualifying fraction for rule (a) (none's censored median t_grok >= {LATE}): {late}\n\n")
        L.append(f"comparison T(a, b), paired by seed; PASS on either co-primary, each at p < {ALPHA} (t_grok: a earlier; "
                 f"AUC: a higher by >= {AUC_MIN} in the medians); wins need >= ceil(0.8 n)\n\n")
        L.append("| test | n | t_grok median a / b | wins | p | AUC median a / b | wins | p | verdict |\n"
                 "|---|---|---|---|---|---|---|---|---|\n")
        sufs = [""] + sorted({"@" + a.split("@")[1] for a in runs if "@" in a})
        if task == "s4":
            core = [("R", "specrand"), ("R_sign", "specrand"), ("absR", "specrand"), ("R", "none"), ("R_sign", "none"),
                    ("specrand", "none"), ("shuf", "none"), ("absR", "none")]
        else:
            core = [("expo", "specrand"), ("onehot", "specrand"), ("quot", "specrand"), ("expo", "none"), ("onehot", "none"),
                    ("specrand", "none"), ("shuf", "none"), ("expo", "onehot"), ("onehot", "expo"),
                    ("onehot", "specrand_oh"), ("specrand_oh", "none"), ("expo", "specrand_oh")]
        pairs = []
        for s in sufs:
            pairs += [(a + s, b if b == "none" else b + s) for a, b in core]
            if s:
                pairs += [(TRUE[task] + s, TRUE[task]), (TRUE[task], TRUE[task] + s)]
        fm = lambda m: "> budget" if m == math.inf else f"{m:.0f}"
        for a, b in dict.fromkeys(pairs):
            t = compare(runs, a, b, chance)
            if t:
                summ["tests"][f"{a}<{b}"] = t
                g, u = t["grok"], t["auc"]
                L.append(f"| {a} < {b} | {t['n']} | {fm(g['med_a'])} / {fm(g['med_b'])} | {g['wins']}/{t['n']} | {g['p']:.3f} "
                         f"| {u['med_a']:.3f} / {u['med_b']:.3f} | {u['wins']}/{t['n']} | {u['p']:.3f} | {t['verdict']} |\n")
        for s in sufs:
            v = rule_a(summ["tests"], task, late, s)
            summ[f"rule_a{s}"] = v
            L.append(f"\nrule (a){' [' + s + ' arms, secondary]' if s else ''}: {v}\n")
        key, arm, ch = ("coset_share", "absR", 3 / 23) if task == "s4" else ("drop_share", "quot", 2 / 26)
        L.append(f"\nrule (b): {key} of wrong test answers pooled over evals in [{B_WIN[0]}, {B_WIN[1]}] (uniform-error chance "
                 f"{ch:.3f}, approximate for dim: answers are not uniform); per seed (share, n_wrong); a seed counts only with n_wrong >= 20\n\n")
        summ["share"] = {}
        for A in order:
            vals = [(r["seed"],) + share_in(r, key, *B_WIN) for r in runs[A]]
            summ["share"][A] = {s: [v, n] for s, v, n in vals}
            L.append(f"- {A}: " + ", ".join(f"s{s} {'-' if v is None else f'{v:.2f}'} ({n})" for s, v, n in sorted(vals)) + "\n")
        if arm in runs:
            det = [v for v, n in summ["share"][arm].values() if v is not None and n >= 20]
            summ["rule_b"] = bool(sum(v > 0.5 for v in det) >= 4) if len(det) >= 4 else None   # None = undetermined
            L.append(f"\nrule (b) {arm}: share > 0.5 in {sum(v > 0.5 for v in det)}/{len(det)} determined seeds -> "
                     f"{'undetermined (< 4 seeds with >= 20 wrong answers)' if summ['rule_b'] is None else summ['rule_b']}\n")
            if "none" in runs:
                dn = [v for v, n in summ["share"]["none"].values() if v is not None and n >= 20]
                L.append(f"control: none's share > 0.5 in {sum(v > 0.5 for v in dn)}/{len(dn)} determined seeds"
                         " (>= 4 -> the error signature is generic, not evidence that the prior was read)\n")
        for s in sufs:
            v = rule_c(runs, summ["tests"], task, s)
            summ[f"rule_c{s}"] = v
            L.append(f"rule (c){' [' + s + ']' if s else ''} (read at the lowest fraction of the design only): {v}\n")
        if task == "dim":
            L.append("rule (d) (read only if expo or onehot passes vs none or vs specrand): expo < onehot = the rank-3 additive "
                     "code beats its rank-6 superset (onehot's centred span contains expo); onehot < expo = the rank-6 superset "
                     "is faster, which rank alone can produce (F59) -- with specrand_oh, onehot < specrand_oh separates onehot's "
                     "structure from its rank; neither = not separated at n=5\n")
            # re-audit 2026-10-17 (amendment before any result): held-out questions whose negation twin (a, other op, -b;
            # same answer) is in training can be answered by a lookup; a "learned the lattice" reading of any dim pass also
            # needs the AUC on the no-twin subset (same rule as the AUC co-primary, floor line included)
            if all(c.get("test_nonegtwin") is not None for r in rs for c in r["curve"]):
                L.append("\nno-twin subset (held-out questions whose negation twin is not in training; twin counts per seed "
                         + ", ".join(f"s{r['seed']} {r['curve'][0]['n_negtwin']}" for r in sorted(runs.get("none", rs[:1]), key=lambda r: r["seed"]))
                         + "): median AUC per arm: "
                         + ", ".join(f"{A} {median_vals([auc(r, 'test_nonegtwin') for r in runs[A]]):.3f}" for A in order) + "\n")
                for a, b in (("expo", "specrand"), ("expo", "none"), ("onehot", "specrand")):
                    if a in runs and b in runs:
                        sa, sb = {r["seed"]: r for r in runs[a]}, {r["seed"]: r for r in runs[b]}
                        sd = sorted(set(sa) & set(sb)); n = len(sd)
                        ua, ub = [auc(sa[x], "test_nonegtwin") for x in sd], [auc(sb[x], "test_nonegtwin") for x in sd]
                        ma, mb = median_vals(ua), median_vals(ub)
                        w, pv = sum(x > y for x, y in zip(ua, ub)), mwu_p([-x for x in ua], [-x for x in ub])
                        ok = ma - mb >= AUC_MIN and ma > chance + AUC_MIN and w >= math.ceil(0.8 * n) and pv < ALPHA
                        summ["tests"][f"{a}<{b}@nonegtwin"] = {"n": n, "med_a": ma, "med_b": mb, "wins": w, "p": pv, "pass": bool(ok)}
                        L.append(f"- AUC {a} > {b} on the no-twin subset: {ma:.3f} / {mb:.3f}, wins {w}/{n}, p {pv:.3f} -> "
                                 f"{'PASS' if ok else 'not shown'}\n")
            else:
                L.append("\nno-twin subset: not logged in these runs (code before the 2026-10-17 amendment)\n")
    (out / "summary.md").write_text("".join(L))
    clean = lambda x: ({k: clean(v) for k, v in x.items()} if isinstance(x, dict) else [clean(v) for v in x] if isinstance(x, list)
                       else "censored" if isinstance(x, float) and math.isinf(x) else x)
    (out / "summary.json").write_text(json.dumps(clean(summ), indent=1))
    print("".join(L))


# ---------------------------------------------------------------------------------------------------- wall 2 reading
# PRE-REGISTERED (runs/toyshape/AUDIT.md, 2026-10-19, before any run; revised the same day after the independent audit,
# still before any run): role -> (always-on arm, [0, 2000)-window arm)
WALL_ROLES = {"mod": {"assign": ("always", "w0"), "shape": ("shape:circle@always", "shape:circle"),
                      "shapeprof": ("shapeprof:circle@always", "shapeprof:circle"),
                      "wrongshape": ("shape:line@always", "shape:line"), "rankctl": ("shape:sphere@always", "shape:sphere")},
              "s4": {"assign": ("R@always", "R"), "shape": ("shape:R@always", "shape:R"),
                     "shapeprof": ("shapeprof:R@always", "shapeprof:R"),
                     "wrongshape": ("shape:line@always", "shape:line"), "rankctl": ("shape:sphere@always", "shape:sphere")}}
WALL_TESTED = ("shape", "shapeprof")   # the two shape roles; each read on its own, no pooling
WALL_CONTROLS = {"wrongshape": "line (same rank, anisotropic)", "rankctl": "sphere (other rank, isotropic, constant norm)"}
WALL_RANK = {"circle": 2, "torus": 10, "R": 9}   # centred rank of each variant's true shape
WALL_SCHED = ("always", "window")      # PRIMARY = always; window = secondary
WALL_NONE_LRS, WALL_NONE_WDS = (1e-3, 3e-3, 1e-2, 3e-2), (0.1, 1.0, 3.0)
WALL_NONE_GRID = [(lr, wd) for lr in WALL_NONE_LRS for wd in WALL_NONE_WDS]
# rule 2: a flat cell is extended once, one x3 step beyond the top of each axis (not the corner lr 1e-1 x wd 10, where
# AdamW's decoupled decay lr * wd = 1 zeroes the weights every step)
WALL_FLAT_EXT = [(1e-1, wd) for wd in WALL_NONE_WDS] + [(lr, 10.0) for lr in WALL_NONE_LRS]
WALL_PRIOR_LRS, WALL_PRIOR_WD = (1e-3, 1e-2), 1.0
WALL_CAP = 0.20                            # rule 1: every none config's median final test <= this (critique D1's level)
WALL_RISE_ACC, WALL_RISE_CE = 0.02, 0.01   # rule 1: last-25 % median held-out accuracy gain / test-CE drop (nats, D1)
WALL_FRESH = ("mod", "s4")                 # every comparison must also pass on test_fresh; its presence is required


def upto(r: dict, B: int, key: str = "test") -> tuple:
    """(t_grok, AUC) of a run read on its first B steps only: the evals at steps 0, e, ..., B - e (B / e of them; a prior
    arm's extra last eval at B - 1 is left out), so a prior arm run for B steps and none run for 4 B are read on the same
    evals. t_grok = first of these evals with held-out test >= 0.95 (inf = never); AUC = mean of `key` over them
    (key test_fresh: the held-out questions whose twin is not in training, mod --mod_twins / s4)."""
    e = r.get("eval_every", 100)
    v = [c for c in r["curve"] if c["step"] < B and c["step"] % e == 0]
    assert B % e == 0 and r["steps"] >= B and len(v) == B // e and v[-1]["step"] == B - e, (r["arm"], r["seed"], len(v), B)
    assert all(c.get(key) is not None for c in v), (r["arm"], r["seed"], key)
    tg = next((c["step"] for c in v if c["test"] >= 0.95), None)
    return (math.inf if tg is None else float(tg)), float(np.mean([c[key] for c in v]))


def late_change(r: dict, B4: int) -> tuple:
    """(held-out accuracy gain, test-CE drop in nats) over the last 25 % of the budget B4: the mean of the last 5 evals
    before B4 minus the mean of the first 5 evals at or after 0.75 B4 (CE: the reverse). A non-finite value (a diverged
    run) counts as no change: divergence = not learning (pre-registered). Only a smoke-length curve (< 20 evals) may
    have too few late evals (then 0, 0)."""
    e = r.get("eval_every", 100)
    v = [c for c in r["curve"] if c["step"] < B4 and c["step"] % e == 0]
    late = [c for c in v if c["step"] >= 0.75 * B4]
    k = min(5, len(late) // 2)
    assert k >= 1 or len(v) < 20, (r["arm"], r["seed"], B4, len(v))
    if k == 0:
        return 0.0, 0.0
    a, z = late[:k], late[-k:]
    assert a[-1]["step"] < z[0]["step"], (r["arm"], r["seed"], B4)
    assert all("test_ce" in c for c in a + z), (r["arm"], r["seed"], "test_ce missing")
    m = lambda cs, k: float(np.mean([c[k] for c in cs]))
    acc, ce = m(z, "test") - m(a, "test"), m(a, "test_ce") - m(z, "test_ce")
    return (acc if math.isfinite(acc) else 0.0), (ce if math.isfinite(ce) else 0.0)


def has_fresh(r: dict) -> bool:
    return all(c.get("test_fresh") is not None for c in r["curve"])


def wall_variant(r: dict) -> str:
    if r["task"] == "mod":
        f = r["freqs"]
        return "circle" if f == [1] else "torus" if f == [1, 2, 3, 4, 5] else "freqs" + "-".join(map(str, f))
    return SHAPE_TRUE[r["task"]]


def _write_wall(root: Path, text: str, summ: dict) -> None:
    clean = lambda x: ({str(k): clean(v) for k, v in x.items()} if isinstance(x, dict) else [clean(v) for v in x]
                       if isinstance(x, (list, tuple)) else "censored" if isinstance(x, float) and math.isinf(x) else x)
    for name, body in (("wall_summary.md", text), ("wall_summary.json", json.dumps(clean(summ), indent=1))):
        tmp = root / f".tmp_{name}.{os.getpid()}"
        tmp.write_text(body)
        os.replace(tmp, root / name)
    print(text)


def summarize_wall(root: Path) -> None:
    """Wall-2 reading over the cell dirs root/<cell>/run_*.json of ONE experiment (one code_sha; PRE-REGISTERED rules,
    runs/toyshape/AUDIT.md; a smoke or older run mixed in gives no verdict).
    1 wall cell (task, frac): EVERY none config of the lr x wd sweep that ran (a) groks on <= floor(0.2 n) seeds within
      its 4 B budget, (b) has a median final held-out test <= WALL_CAP, (c) is not still rising: median last-25 % gain
      < WALL_RISE_ACC (every config) and, for the best config, median last-25 % test-CE drop < WALL_RISE_CE nats.
      Otherwise 'not a wall cell (reason)': speed-up reading only.
    2 bracket: best none config = most groks, then median AUC over 4 B. Closed iff it is not on an edge of the grid that
      ran. 'Flat' (every config 0 groks and AUC at or below the floor line) is NOT closed until the one-step extension
      WALL_FLAT_EXT has run and the cell is still flat. A wall verdict with an open bracket is prefixed 'BRACKET OPEN'.
    3 per prior role and schedule, its lr (of 1e-3 / 1e-2, wd 1) with the highest median AUC_B; best none for the paired
      tests = the config with the highest median AUC_B (12+ choices vs 2: favours none). AUC_B / t_grok_B = upto().
    4 a role MEETS the rule iff it groks within B on >= ceil(0.8 n) seeds AND compare_values(role, best none).auc passes
      on held-out test AND on test_fresh (floor line included). 'a beats b' = the paired AUC test passes on both.
    5 verdict per tested role (shape, shapeprof) and schedule (always = PRIMARY), at a wall cell:
      a control (line, sphere) meets the rule too -> 'NOT SHAPE-SPECIFIC'; else the role meets it and beats BOTH controls
      -> 'BREAKS THE WALL' (shape: as a rank-r isotropic constant-norm prior; shapeprof: with the row profile, plus
      'geometry beyond rank SHOWN' iff shapeprof also beats shape); meets it without beating both -> 'SPECIFICITY NOT
      SHOWN'; AUC pass without the groks -> 'partial'; 'floor: uninformative'; else 'not shown at n'.
    6 data multiplier: f* / f, f* = the smallest swept fraction at which the best none config's median final test (at 4 B)
      >= the role's median final test (at B)."""
    rs = [json.loads(p.read_text()) for p in sorted(root.glob("*/run_*.json"))]
    if not rs:
        print("no runs under", root, "(expects <root>/<cell>/run_*.json, e.g. runs/toyshape/out)"); return
    shas = sorted({str(r.get("code_sha")) for r in rs})
    if len(shas) != 1:
        _write_wall(root, f"# wall 2 (shape-only prior) reading over {root}\n\nNO VERDICT: {len(rs)} runs from {len(shas)} "
                          f"code_sha values {shas}; read one experiment's out/ dir\n", {"root": str(root), "code_sha": shas})
        return
    none, prior, notes = {}, {}, []
    for r in rs:
        if r["task"] not in WALL_ROLES:
            continue
        if r["arm"] == "none":
            none.setdefault((r["task"], r["frac"]), {}).setdefault((r["lr"], r["wd"]), {})[r["seed"]] = r
        else:
            if r["wd"] != WALL_PRIOR_WD or r["lam"] != 1.0:
                notes.append(f"excluded (not the pre-registered wd 1 / lam 1): {r['task']} {r['arm']} lr {r['lr']} wd {r['wd']} lam {r['lam']}")
                continue
            prior.setdefault((r["task"], wall_variant(r), r["frac"]), {}).setdefault((r["arm"], r["lr"]), {})[r["seed"]] = r
    L = [f"# wall 2 (shape-only prior) reading over {root}\n\n", f"{len(rs)} runs; code_sha {shas}\n\n"]
    L += [f"- {x}\n" for x in notes[:20]]
    summ = {"root": str(root), "n_runs": len(rs), "code_sha": shas, "cells": {}}
    fm = lambda v: "never" if v == math.inf else f"{v:.0f}"
    cf = lambda k: f"lr {k[0]:g} wd {k[1]:g}"
    over = ["| task | variant | frac | none: best groks within 4B | wall | bracket | shape@always | shapeprof@always "
            "| shape@window | shapeprof@window | assign@always groks / AUC_B | multiplier (shape@always) |\n",
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    heads = []
    for (task, var, frac) in sorted(prior):
        P, Nc = prior[(task, var, frac)], none.get((task, frac), {})
        cell = {"task": task, "variant": var, "frac": frac}
        summ["cells"][f"{task}-{var}-f{frac}"] = cell
        L.append(f"\n## {task} {var}, frac {frac}\n\n")
        Bs = sorted({r["steps"] for d in P.values() for r in d.values()})
        B4s = sorted({r["steps"] for d in Nc.values() for r in d.values()})
        if len(Bs) != 1 or not Nc or len(B4s) != 1 or B4s[0] != 4 * Bs[0]:
            L.append(f"NO VERDICT: prior budgets {Bs}, none budgets {B4s} (needs one B and none at 4 B)\n")
            cell["verdict"] = "no verdict (budgets)"; continue
        B, B4 = Bs[0], B4s[0]
        allr = [r for d in list(P.values()) + list(Nc.values()) for r in d.values()]
        nofresh = sum(not has_fresh(r) for r in allr) if task in WALL_FRESH else 0
        if nofresh:   # audit 2026-10-19: a missing test_fresh must not count as a pass
            L.append(f"NO VERDICT: {nofresh} of {len(allr)} runs without test_fresh (mod needs --mod_twins)\n")
            cell["verdict"] = "no verdict (test_fresh missing)"; continue
        chance = chance_of(allr, task)
        cell.update({"B": B, "B4": B4, "chance": chance, "floor_line": chance + AUC_MIN})
        missing = [g for g in WALL_NONE_GRID if g not in Nc]
        prov = bool(missing) or any(len(d) < 5 for d in Nc.values())
        L.append(f"prior budget B {B}, none budget {B4} (= 4 B); chance {chance:.4f}, floor line {chance + AUC_MIN:.4f}"
                 + (f"; PROVISIONAL: none grid incomplete (missing {missing})" if missing else "") + "\n\n")
        # ---- the none sweep
        L += ["| none lr | wd | n | groks within 4B | t_grok cmedian | AUC 4B median | AUC_B median | final test median "
              "| last 25%: gain / test-CE drop (medians) |\n", "|---|---|---|---|---|---|---|---|---|\n"]
        nrow = {}
        for (lr, wd) in sorted(Nc):
            d = Nc[(lr, wd)]
            rr = [d[s] for s in sorted(d)]
            ub = [upto(r, B) for r in rr]
            lc = [late_change(r, B4) for r in rr]
            nrow[(lr, wd)] = {"n": len(rr), "grok4": sum(r["t_grok"] is not None for r in rr), "auc4": median_vals([auc(r) for r in rr]),
                              "aucB": median_vals([u for _, u in ub]), "final": median_vals([r["final_test"] for r in rr]),
                              "gain": median_vals([g for g, _ in lc]), "ce_drop": median_vals([c for _, c in lc]), "seeds": sorted(d)}
            x = nrow[(lr, wd)]
            L.append(f"| {lr:g} | {wd:g} | {x['n']} | {x['grok4']}/{x['n']} | {fm(cmedian(rr))} | {x['auc4']:.3f} | {x['aucB']:.3f} "
                     f"| {x['final']:.3f} | {x['gain']:+.3f} / {x['ce_drop']:+.3f} |\n")
        n = min(x["n"] for x in nrow.values())
        best_grok = max(x["grok4"] for x in nrow.values())
        bk = max(nrow, key=lambda k: (nrow[k]["grok4"], nrow[k]["auc4"], -k[0]))
        cap = [k for k in nrow if nrow[k]["final"] > WALL_CAP]
        rising = [k for k in nrow if nrow[k]["gain"] >= WALL_RISE_ACC or (k == bk and nrow[k]["ce_drop"] >= WALL_RISE_CE)]
        why = ([f"best none groks {best_grok}/{n} within 4B"] if best_grok > math.floor(0.2 * n) else []) \
            + ([f"none above the {WALL_CAP:.2f} cap: " + ", ".join(f"{cf(k)} {nrow[k]['final']:.2f}" for k in cap)] if cap else []) \
            + (["none still rising in the last 25% of 4B: " + ", ".join(f"{cf(k)} (gain {nrow[k]['gain']:+.3f}, CE drop "
                                                                        f"{nrow[k]['ce_drop']:+.3f})" for k in rising)] if rising else [])
        wall = not why
        lrs, wds = sorted({k[0] for k in nrow}), sorted({k[1] for k in nrow})
        flat = all(x["grok4"] == 0 and x["auc4"] <= chance + AUC_MIN for x in nrow.values())
        if flat:
            ext_missing = [g for g in WALL_FLAT_EXT if g not in nrow]
            bracket = ("flat over the extended grid (lr to 1e-1, wd to 10)" if not ext_missing else
                       "OPEN (flat on the grid that ran: extend once to lr 1e-1 and wd 10)")
        else:
            edges = [e for e, hit in (("lr top", bk[0] == lrs[-1]), ("lr bottom", bk[0] == lrs[0]),
                                      ("wd top", bk[1] == wds[-1]), ("wd bottom", bk[1] == wds[0])) if hit]
            bracket = "closed" if not edges else "OPEN at " + ", ".join(edges)
        closed = not bracket.startswith("OPEN")
        nb = max(nrow, key=lambda k: (nrow[k]["aucB"], -k[0]))
        L.append(f"\nwall cell (every none config: groks <= {math.floor(0.2 * n)}/{n} within 4B, median final <= {WALL_CAP:.2f}, "
                 f"not rising): **{wall}**" + ("" if wall else " (" + "; ".join(why) + ")")
                 + f"; best config {cf(bk)}, bracket {bracket}; best none on AUC_B (paired tests) {cf(nb)}\n")
        sk = {"circle": "shapeL_fourier1", "torus": "shapeL_fourier1-5"}.get(var, f"shapeL_{var}")
        nbr = [Nc[nb][s] for s in sorted(Nc[nb])]
        if all(sk in c for r in nbr for c in r["curve"]):   # reported only (rule 7)
            atB = median_vals([[c for c in r["curve"] if c["step"] < B][-1][sk] for r in nbr])
            at4 = median_vals([r["curve"][-1][sk] for r in nbr])
            L.append(f"none ({cf(nb)}) distance to the {var} shape (spec + norm, reported only): {atB:.3f} at B, {at4:.3f} at 4 B\n")
            cell["none_shapeL"] = {"key": sk, "at_B": atB, "at_4B": at4}
        L.append("\n")
        cell.update({"none": {f"lr{k[0]:g}-wd{k[1]:g}": v for k, v in nrow.items()}, "wall": wall, "wall_why": why,
                     "none_best_grok": best_grok, "bracket": bracket, "bracket_closed": closed, "none_best": [bk[0], bk[1]],
                     "none_best_aucB": [nb[0], nb[1]], "provisional": prov})
        # ---- prior roles: best lr per role and schedule
        L += ["| role | schedule | arm | lr chosen (AUC_B per lr) | n | groks within B | AUC_B median | AUC_B fresh median "
              "| final test median | own objective at end | CKA to the assigned target at end |\n",
              "|---|---|---|---|---|---|---|---|---|---|---|\n"]
        pick = {}
        ck_key = "cka_circle" if task == "mod" else f"cka_{SHAPE_TRUE[task]}"
        for role, arms in WALL_ROLES[task].items():
            for si, arm in enumerate(arms):
                cand = {lr: d for (a, lr), d in P.items() if a == arm}
                if not cand:
                    continue
                stats = {}
                for lr, d in cand.items():
                    rr = [d[s] for s in sorted(d)]
                    ub = [upto(r, B) for r in rr]
                    uf = [upto(r, B, "test_fresh")[1] if has_fresh(r) else None for r in rr]
                    stats[lr] = {"runs": {r["seed"]: (r, t, u, f) for r, (t, u), f in zip(rr, ub, uf)}, "aucB": median_vals([u for _, u in ub])}
                lr = max(stats, key=lambda k: (stats[k]["aucB"], -k))
                S = stats[lr]["runs"]
                gB = sum(x[1] < math.inf for x in S.values())
                fin = median_vals([x[0]["final_test"] for x in S.values()])
                geo = [g for g in (x[0]["curve"][-1].get("geo") for x in S.values()) if g is not None]
                cka_end = [g for g in (x[0]["curve"][-1].get(ck_key) for x in S.values()) if g is not None]
                ufr = [x[3] for x in S.values() if x[3] is not None]
                pick[(role, WALL_SCHED[si])] = {"arm": arm, "lr": lr, "runs": S, "groksB": gB, "n": len(S), "aucB": stats[lr]["aucB"], "final": fin}
                per_lr = ", ".join("%g: %.3f" % (k, v["aucB"]) for k, v in sorted(stats.items()))
                L.append(f"| {role} | {WALL_SCHED[si]} | {arm} | {lr:g} ({per_lr}"
                         f"{'' if set(stats) == set(WALL_PRIOR_LRS) else '; INCOMPLETE'}) | {len(S)} | {gB}/{len(S)} | {stats[lr]['aucB']:.3f} "
                         f"| {f'{median_vals(ufr):.3f}' if ufr else '-'} | {fin:.3f} | {f'{np.median(geo):.3f}' if geo else '-'} "
                         f"| {f'{np.median(cka_end):.3f}' if cka_end else '-'} |\n")
        # ---- paired tests (AUC_B / t_grok_B), verdicts, data multiplier
        NB = Nc[nb]

        def test(pa, Nd, fresh=False):
            sd = sorted(set(pa["runs"]) & set(Nd))
            if not sd or (fresh and not all(pa["runs"][s][3] is not None and has_fresh(Nd[s]) for s in sd)):
                return None
            ga, ua = [pa["runs"][s][1] for s in sd], [pa["runs"][s][3 if fresh else 2] for s in sd]
            gb, ub = zip(*[upto(Nd[s], B, "test_fresh" if fresh else "test") for s in sd])
            return compare_values(ga, list(gb), ua, list(ub), chance)

        def test_pp(pa, pb, fresh=False):
            sd = sorted(set(pa["runs"]) & set(pb["runs"]))
            i = 3 if fresh else 2
            if not sd or any(pa["runs"][s][i] is None or pb["runs"][s][i] is None for s in sd):
                return None
            return compare_values([pa["runs"][s][1] for s in sd], [pb["runs"][s][1] for s in sd],
                                  [pa["runs"][s][i] for s in sd], [pb["runs"][s][i] for s in sd], chance)

        cell["tests"], cell["verdict"], cell["multiplier"], cell["meets"] = {}, {}, {}, {}
        for sch in WALL_SCHED:
            L += [f"\n### schedule {sch} ({'PRIMARY' if sch == 'always' else 'secondary'})\n\n"
                  "| test (read on the first B steps, paired by seed) | n | t_grok median a / b | wins | p | AUC_B median a / b | wins | p | verdict |\n",
                  "|---|---|---|---|---|---|---|---|---|\n"]
            kn = lambda role: f"{role}@{sch} > none (lr {nb[0]:g} wd {nb[1]:g})"
            rows = [(kn(role), pick[(role, sch)], None) for role in WALL_ROLES[task] if (role, sch) in pick]
            for a_, b_ in (("shape", "wrongshape"), ("shape", "rankctl"), ("shapeprof", "wrongshape"), ("shapeprof", "rankctl"),
                           ("shapeprof", "shape")):
                if (a_, sch) in pick and (b_, sch) in pick:
                    rows.append((f"{a_}@{sch} > {b_}@{sch}", pick[(a_, sch)], pick[(b_, sch)]))
            for name, pa, pb in rows:
                for fresh in (False, True):
                    t = test(pa, NB, fresh) if pb is None else test_pp(pa, pb, fresh)
                    if t is None:
                        continue
                    nm_ = name + (" [test_fresh AUC]" if fresh else "")
                    cell["tests"][nm_] = t
                    g, u = t["grok"], t["auc"]
                    L.append(f"| {nm_} | {t['n']} | {fm(g['med_a'])} / {fm(g['med_b'])} | {g['wins']}/{t['n']} | {g['p']:.3f} "
                             f"| {u['med_a']:.3f} / {u['med_b']:.3f} | {u['wins']}/{t['n']} | {u['p']:.3f} | {t['verdict']} |\n")

            def passes(name):   # the AUC test passes on held-out test AND on test_fresh (both required, rule 4)
                t, tf = cell["tests"].get(name), cell["tests"].get(name + " [test_fresh AUC]")
                return bool(t is not None and tf is not None and t["auc"]["pass"] and tf["auc"]["pass"])

            def meets(role):
                pa = pick.get((role, sch))
                return None if pa is None else bool(pa["groksB"] >= math.ceil(0.8 * pa["n"]) and passes(kn(role)))

            beats = lambda a_, b_: passes(f"{a_}@{sch} > {b_}@{sch}")

            def verdict(role):
                if (role, sch) not in pick or any((c, sch) not in pick for c in WALL_CONTROLS):
                    return "arms missing"
                pa = pick[(role, sch)]
                if not wall:
                    return "not a wall cell (" + "; ".join(why) + "): speed-up reading only"
                cm = [c for c in WALL_CONTROLS if meets(c)]
                ctl = " and ".join(WALL_CONTROLS[c] for c in cm)
                if meets(role):
                    if cm:
                        return f"breaks it, but so does {ctl}: NOT SHAPE-SPECIFIC"
                    sep = {c: beats(role, c) for c in WALL_CONTROLS}
                    if all(sep.values()):
                        if role == "shape":
                            return (f"BREAKS THE WALL (the true shape's spectrum + norms: a rank-{WALL_RANK.get(var, '?')} isotropic "
                                    "constant-norm prior; beats line and sphere, paired)")
                        return ("BREAKS THE WALL (spectrum + norms + row profile; beats line and sphere, paired); "
                                + ("geometry beyond rank SHOWN (shapeprof > shape, paired)" if beats("shapeprof", "shape")
                                   else "geometry beyond rank not shown (shapeprof vs shape)"))
                    return "breaks it; SPECIFICITY NOT SHOWN (" + ", ".join(
                        f"vs {c}: {'pass' if sep[c] else 'n.s.'}" for c in WALL_CONTROLS) + ")"
                if passes(kn(role)):
                    return (f"partial: AUC_B above none (test and test_fresh), groks {pa['groksB']}/{pa['n']} within B (< 4/5)"
                            + (f"; {ctl} meets the rule" if cm else ""))
                t = cell["tests"].get(kn(role))
                if t is not None and t["floor"]:
                    return "floor: uninformative"
                return f"not shown at n={pa['n']}"

            for role in WALL_TESTED:
                v = verdict(role)
                if v != "arms missing":
                    if wall and not closed:   # pre-registered: extend none once beyond the edge / the flat grid
                        v = f"BRACKET OPEN ({bracket}; extend none, AUDIT.md rule 2): " + v
                    if prov:
                        v = "PROVISIONAL (none grid incomplete): " + v
                cell["verdict"][f"{role}@{sch}"] = v
                L.append(f"\n**verdict {role}@{sch}{' (PRIMARY)' if sch == 'always' else ' (secondary)'}**: {v}\n")
                if sch == "always" and v.startswith("BREAKS THE WALL"):
                    heads.append(f"{task} {var} frac {frac}: {role}: {v}")
            cell["meets"][sch] = {role: meets(role) for role in WALL_ROLES[task]}
            L.append(f"\nmeets the rule ({sch}): " + ", ".join(f"{k} {v}" for k, v in cell["meets"][sch].items())
                     + " (assign = the upper reference)\n")
            # data multiplier: smallest swept fraction at which the best none config's median final test reaches the role's
            fr = sorted(f for (t_, f) in none if t_ == task)
            best_final = {f: max(median_vals([r["final_test"] for r in d.values()]) for d in none[(task, f)].values()) for f in fr}
            for role in WALL_ROLES[task]:
                pa = pick.get((role, sch))
                if pa is None:
                    continue
                A = pa["final"]
                hit = next((f for f in fr if best_final[f] >= A), None)
                below = [f for f in fr if hit is None or f < hit]
                m = (f"<= 1 (none reaches {A:.3f} at frac {hit} <= {frac})" if hit is not None and hit <= frac else
                     f"{hit / frac:.2f} (none first reaches {A:.3f} at frac {hit}; not at {max(below) if below else '-'})" if hit is not None else
                     f"> {fr[-1] / frac:.2f} (none never reaches {A:.3f} up to frac {fr[-1]})")
                cell["multiplier"][f"{role}@{sch}"] = {"A": A, "f_star": hit, "text": m}
                L.append(f"- data multiplier {role}@{sch}: {m}\n")
        sh = lambda k: cell["verdict"].get(k, "-")
        pa = pick.get(("assign", "always"))
        asg = "-" if pa is None else "%d/%d / %.3f" % (pa["groksB"], pa["n"], pa["aucB"])
        over.append(f"| {task} | {var} | {frac} | {best_grok}/{n} | {wall} | {bracket} | {sh('shape@always')} | {sh('shapeprof@always')} "
                    f"| {sh('shape@window')} | {sh('shapeprof@window')} | {asg} | {cell['multiplier'].get('shape@always', {}).get('text', '-')} |\n")
    summ["headline"] = heads
    hl = ["## headline (rule 5: a PRIMARY 'BREAKS THE WALL' at >= 1 wall cell, bracket closed, grid complete)\n\n"] + \
         ([f"- {h}\n" for h in heads] or ["- none: no cell reads 'BREAKS THE WALL' on the PRIMARY schedule\n"]) + ["\n"]
    _write_wall(root, "".join(L[:2] + hl + ["## overview\n\n"] + over + L[2:]), summ)

if __name__ == "__main__":
    main()
