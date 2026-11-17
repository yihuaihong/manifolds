"""ptdom3 (2026-11-17): the seven new toy-structure domains of the ptdom3 screen, for the trainer's TOY3 path.

A NEW module so that toy_struct.py (sha d182ce548bcc, recorded by every earlier batch) stays byte-identical. Pure numpy;
no state; build(d) is deterministic (no RNG). Every domain is Tier L: its target (the prior's geometry) together with
the relation word determines the answer.

  domain  items                          line (training template)                       target (prior)            split
  cyc12   Z_12, 12 nonce tokens          "⟦a⟧ with⟦b⟧ is⟦c⟧"   c = a + b mod 12         circle (2)                class {a, b}, 0.5
  torus   Z_6 x Z_6, 36 nonce tokens     "⟦a⟧ and⟦b⟧ is⟦c⟧"    c = a + b (mod 6 each)   two circles (4)           class {a, b}, 0.6
  hcube   Z_2^5, 32 nonce tokens         "⟦a⟧ xor⟦b⟧ is⟦c⟧"    c = a XOR b              +-1^5 coordinates (5)     class {a, b}, 0.6
  d6      dihedral D_6 (order 12),       "⟦a⟧ then⟦b⟧ is⟦c⟧"   R_c = R_b R_a            the faithful 2x2 real     question, 0.5 (s4's
          12 nonce tokens                                                              rep, flattened (4)        rule; read-out fresh)
  box5    {-2..2}^2 exponent vectors,    "⟦a⟧ plus⟦b⟧ is⟦c⟧" / "⟦a⟧ minus⟦b⟧ is⟦c⟧"     coordinates (2)           relation class, 0.5
          25 nonce tokens                c = a +- b, only inside the box (dim's rule)                              (dim's rule)
Dropped at the build (runs/ptdom3/tests/tok_survey.out): notes (12 pitch classes) - Gb / Bb and every sharp are two
OLMo-2 tokens (' G' + 'b', ' C' + '#'), so 'the item's last token' is shared by several pitch classes; chess (a1..h8) - no
square is a single token (' e' + '4'), the last token is the rank digit, shared by 8 squares and by the numbers operands.

Sites (the prior's rows) = both input items, each a single nonce token (asserted by the trainer at build time and by
tests/toy3_test.py). The answer item is never a site.
Read-out mask (held-out questions; the trainer's DINFO readout): held AND nonid (no copy cell: an input is the identity /
zero, or box5's x minus x, or hcube's x xor x) AND (d6: fresh = commuting twin not trained; box5: no trained negation
twin, dim's rule). 'hard' (descriptive mask): held cells NOT derivable from the training table by associativity + Latin
cancellation (toy_struct.s4_closure(latin=True); the binary groups only) = s4_hard's analogue.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

TOY3 = ("cyc12", "torus", "hcube", "d6", "box5")   # fixed order: SeedSequence(4400 + seed) children 3 i, 3 i + 1, 3 i + 2
TOY3_STREAM = 4400
GROUPS = ("cyc12", "torus", "hcube", "d6")   # binary group tables (X = a, b, eq; mul defined)
SPLIT = {"cyc12": ("class", 0.5), "torus": ("class", 0.6), "hcube": ("class", 0.6), "d6": ("question", 0.5),
         "box5": ("class", 0.5)}
WORD = {"cyc12": " with", "torus": " and", "hcube": " xor", "d6": " then"}
# nonce names (toy3_nonce.json, written once by runs/ptdom3/tests/toy3_test.py from tok_survey.json: the rarest OLMo-2
# single tokens ' xyz' over S4_NAMES' letters (consonants bcdfgjklmnprstvz, vowels aeiou), the 51 s4 / dim names excluded,
# English words / slang, abbreviations, given names, common foreign function words and vulgar ones skipped (the skip list
# and its reasons are in the json); value = occurrences in train.bin)
_NJ = Path(__file__).with_name("toy3_nonce.json")
NONCE = json.loads(_NJ.read_text())["names"] if _NJ.exists() else {}   # domain -> {name: count}


def _class_ids(keys):
    ids = {}
    return np.array([ids.setdefault(k, len(ids)) for k in keys], dtype=np.int64)


def _group(task, mul, F):
    n = len(mul)
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    X = np.stack([a.ravel(), b.ravel(), np.full(n * n, n)], 1)
    Y = mul[X[:, 0], X[:, 1]]
    e = int(np.flatnonzero((mul == np.arange(n)).all(1))[0])
    assert (mul[:, e] == np.arange(n)).all() and (mul[e] == np.arange(n)).all(), (task, "identity")
    noncomm = mul[X[:, 0], X[:, 1]] != mul[X[:, 1], X[:, 0]]
    ckey = [(p, q) if nc else (min(p, q), max(p, q)) for (p, q), nc in zip(X[:, :2].tolist(), noncomm)]
    copy = (X[:, 0] == e) | (X[:, 1] == e) | (Y == X[:, 0]) | (Y == X[:, 1])
    if task == "hcube":
        copy |= X[:, 0] == X[:, 1]       # x xor x = 0: a constant answer from a token-equality rule
    return dict(task=task, X=X, Y=Y, n_items=n, n_out=n, inputs=(0, 1), mul=mul, e=e, noncomm=noncomm,
                cls=_class_ids(ckey), copy=copy, targets={"true": np.asarray(F, np.float64)}, tier="L")


def build(task: str) -> dict:
    """Questions X, answers Y (item indices), inputs (the item columns of X), cls (split unit), copy (bool), targets
    {'true': F} (n_items x dim), mul (groups) and the per-question line pieces (render)."""
    if task == "cyc12":
        n = 12
        mul = (np.arange(n)[:, None] + np.arange(n)[None]) % n
        ang = 2 * np.pi * np.arange(n) / n
        return _group(task, mul, np.stack([np.cos(ang), np.sin(ang)], 1))
    if task == "torus":
        xy = np.array([(x, y) for x in range(6) for y in range(6)])
        key = {tuple(v): i for i, v in enumerate(xy)}
        mul = np.array([[key[tuple((xy[i] + xy[j]) % 6)] for j in range(36)] for i in range(36)])
        ang = 2 * np.pi * xy / 6
        return _group(task, mul, np.concatenate([np.stack([np.cos(ang[:, 0]), np.sin(ang[:, 0])], 1),
                                                 np.stack([np.cos(ang[:, 1]), np.sin(ang[:, 1])], 1)], 1))
    if task == "hcube":
        mul = np.arange(32)[:, None] ^ np.arange(32)[None]
        bits = (np.arange(32)[:, None] >> np.arange(5)[None]) & 1
        return _group(task, mul, 1.0 - 2.0 * bits)
    if task == "d6":
        mats = []
        for f in (0, 1):
            for k in range(6):
                c, s = np.cos(2 * np.pi * k / 6), np.sin(2 * np.pi * k / 6)
                Rk = np.array([[c, -s], [s, c]])
                mats.append(Rk @ (np.array([[1.0, 0.0], [0.0, -1.0]]) if f else np.eye(2)))
        mats = np.array(mats)
        flat = np.round(mats.reshape(12, 4), 9)
        key = {tuple(v): i for i, v in enumerate(flat)}
        assert len(key) == 12, "faithful"
        # "a then b": R_c = R_b R_a (toy_struct.cube_group's convention for s4)
        mul = np.array([[key[tuple(np.round((mats[b] @ mats[a]).reshape(4), 9))] for b in range(12)] for a in range(12)])
        D = _group(task, mul, mats.reshape(12, 4))
        assert D["noncomm"].any(), "d6 is non-abelian"
        return D
    if task == "box5":
        E = np.array(list(itertools.product(range(-2, 3), repeat=2)), dtype=np.int64)
        key = {tuple(v): i for i, v in enumerate(E)}
        rows, ys = [], []
        for op, s in ((25, 1), (26, -1)):    # 25 = plus, 26 = minus (dim: 27 times, 28 per)
            for i in range(25):
                for j in range(25):
                    c = E[i] + s * E[j]
                    if np.abs(c).max() <= 2:
                        rows.append((i, op, j, 27)); ys.append(key[tuple(c)])
        X, Y = np.array(rows), np.array(ys)
        assert (X[:, 1] == 25).sum() == 361 and (X[:, 1] == 26).sum() == 361
        # relation class (dim's rule): e_x + e_y = e_z is stated by x plus y, y plus x, z minus y, z minus x
        ckey = [(min(a, b), max(a, b)) if op == 25 else (min(c, b), max(c, b)) for (a, op, b, _), c in zip(X.tolist(), Y.tolist())]
        z = key[(0, 0)]
        copy = (Y == X[:, 0]) | (Y == X[:, 2]) | ((X[:, 1] == 26) & (X[:, 0] == X[:, 2]))
        neg = np.array([key[tuple(-v)] for v in E])
        return dict(task=task, X=X, Y=Y, n_items=25, n_out=25, inputs=(0, 2), E=E, zero=z, neg=neg, cls=_class_ids(ckey),
                    copy=copy, targets={"true": E.astype(np.float64)}, tier="L", mul=None)
    raise ValueError(task)


def item_names(task: str) -> list:
    """The item strings as they appear in a line (with the leading space), item i = the i-th name of NONCE[task]."""
    return [" " + n for n in NONCE[task]]


def pieces(D: dict, q: int, names: list) -> tuple[list, str, list]:
    """The training line of question q as (text, role) pieces (role ('site', item) / 'ans' / None), its answer string
    and its input item strings (stripped), for the trainer's pieces_line and decode asserts."""
    t, x, c = D["task"], D["X"][q], int(D["Y"][q])
    if t in GROUPS:
        a, b = int(x[0]), int(x[1])
        pcs = [(names[a], ("site", a)), (WORD[t], None), (names[b], ("site", b)), (" is", None), (names[c], "ans"), ("\n", None)]
        ins = [a, b]
    else:   # box5
        a, op, b = int(x[0]), int(x[1]), int(x[2])
        pcs = [(names[a], ("site", a)), (" plus" if op == 25 else " minus", None), (names[b], ("site", b)), (" is", None),
               (names[c], "ans"), ("\n", None)]
        ins = [a, b]
    return pcs, names[c].strip(), [names[i].strip() for i in ins]


def split(D: dict, frac: float, seed: int, mode: str):
    """toy_struct.split's two modes on D (question: a permutation of the questions; class: of the relation classes),
    generalised to D['inputs']; asserts every item is a trained input and a trained answer."""
    N = len(D["X"])
    if mode == "question":
        perm = np.random.default_rng(seed).permutation(N)
        ntr = int(frac * N)
        tr, te = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    else:
        cls = D["cls"]
        pc = np.random.default_rng(seed).permutation(int(cls.max()) + 1)
        is_tr = np.isin(cls, pc[: int(frac * len(pc))])
        tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
        assert not (set(cls[tr]) & set(cls[te])), "relation class split across train / held-out"
    rows = [tuple(r) for r in D["X"].tolist()]
    assert len(set(rows)) == N, "duplicate questions"
    assert not ({rows[i] for i in tr} & {rows[i] for i in te}), "held-out question in training"
    n = D["n_items"]
    seen_in = set()
    for c in D["inputs"]:
        seen_in |= set(D["X"][tr, c].tolist())
    assert seen_in == set(range(n)), f"item never an input in training: {set(range(n)) - seen_in}"
    assert set(D["Y"][tr].tolist()) == set(range(n)), "item never an answer in training"
    return tr, te


def negtwin_mask(D, tr, te):
    """box5 (dim's rule): held (a, op, b) whose negation twin (a, other op, -b) (same answer, other class) is trained."""
    row = {tuple(r): i for i, r in enumerate(D["X"].tolist())}
    trs = set(tr.tolist())
    out = []
    for i in te:
        a, op, b, eq = D["X"][i].tolist()
        j = row.get((a, 51 - op, int(D["neg"][b]), eq))   # 25 <-> 26
        assert j is not None and D["Y"][j] == D["Y"][i], ("negation twin", i)
        out.append(j in trs)
    return np.array(out, dtype=bool)


def masks(D: dict, tr: np.ndarray, te: np.ndarray, closure=None) -> tuple[dict, str, np.ndarray]:
    """Held-out sub-split masks (over te), the read-out mask name and the read-out mask. closure = toy_struct.s4_closure."""
    nonid = ~D["copy"][te]
    m = {"nonid": nonid}
    if D["task"] == "d6":
        fr = ~np.isin(D["cls"][te], D["cls"][tr])
        m.update(fresh=fr, fresh_nonid=fr & nonid)
        name, ro = "held_fresh_nonid", fr & nonid
    elif D["task"] == "box5":
        tw = negtwin_mask(D, tr, te)
        m.update(no_negtwin=~tw, no_negtwin_nonid=~tw & nonid)
        name, ro = "held_no_negtwin_nonid", ~tw & nonid
    else:
        name, ro = "held_nonid", nonid
    if D["task"] in GROUPS and closure is not None:
        k = closure(D["mul"], [tuple(D["X"][i, :2].tolist()) for i in tr], latin=True)
        hard = np.array([tuple(D["X"][i, :2].tolist()) not in k for i in te])
        m.update(hard=hard, hard_ro=hard & ro)
    return m, name, ro
