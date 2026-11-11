"""S4 table-completion levels of a training split (runs/toyf84fu, 2026-10-19; intent spec AUDIT.md section "Latin split").

Three deduction systems, from the training cells of a split (the held-out cells = the rest of the 24 x 24 table):
  assoc        toy_struct.closure()["closure"]: associativity, both directions, to a fixpoint. It uses the GROUP LAW.
  assoc+latin  toy_struct.closure()["closure_latin"]: the same plus "a row / column with one unknown cell" to a fixpoint.
  latin        Latin-square property ONLY (no group law): naked singles and hidden singles per row / column to a
               fixpoint, the computation of rainbow6_summary.s4_latin_level (A7) and of A8's s4_hard. A derived cell
               is a held-out cell that a pure table-filler gets right for sure; every other held-out cell keeps a
               candidate set (the values its row and column do not already use).
Per split: the share of held-out cells each system derives; for latin also the "Latin level" (A7: a derived cell scores
1, any other 1 / |candidates|), and the chance of a pure table-filler on the NON-implied cells (mean 1 / |candidates|).
Read-out subsets: all held-out (toy_struct's `test`) and fresh_nonid (the rainbow's S4 read-out: no trained
commuting twin, no identity input).

Splits: the toy's own split code (toy_struct.split, question mode, pinned b74a78cb5472) at frac 0.2 / 0.3 / 0.45 / 0.6;
the rainbow's S4 splits rebuilt from the rb6 run JSONs (domains.s4.eval_held.q; train = the rest), checked against
the run's recorded closure / closure_latin / n_readout and against A7's printed levels (known answer).

    python latin.py            # tables -> latin_table.md (+ stdout)
Imported by readout.py: latin_split(frac, seed) -> per held-out cell (in toy_struct's te order) implied / candidates.
"""
from __future__ import annotations

import glob
import json
import sys
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code" / "tools"))
import numpy as np  # noqa: E402
import toy_struct as ts  # noqa: E402

PIN_SHA = "b74a78cb5472"
assert ts.code_sha() == PIN_SHA, ts.code_sha()
RB6 = "/scratch/yh6210/llmadd-burst-ro/results/4[0-9][0-9]-rb6*/*/run_*.json"
A7_LEVELS = {0: 35.24, 1: 34.78, 2: 29.60, 3: 30.44, 4: 32.24, 5: 30.18, 6: 26.82, 7: 33.06}   # audit_rb6_a7.md, 2 dp
FRACS = (0.2, 0.3, 0.45, 0.6)


@lru_cache(maxsize=1)
def s4_task() -> dict:
    return ts.build_task("s4")


def latin_singles(mul: np.ndarray, train_cells: list, held_cells: list) -> tuple[dict, dict]:
    """Naked + hidden singles per row / column to a fixpoint (Latin-square property only). Returns (derived {cell: value}
    for held cells, candidates {cell: set} for every held cell). Same algorithm as rainbow6_summary.s4_latin_level."""
    n = len(mul)
    K = {c: int(mul[c]) for c in train_cells}
    cand = {c: set(range(n)) - {v for (a, b), v in K.items() if a == c[0] or b == c[1]} for c in held_cells}

    def place(c, v):
        K[c] = v
        cand[c] = {v}
        for c2 in cand:
            if c2 != c and (c2[0] == c[0] or c2[1] == c[1]):
                cand[c2].discard(v)
    changed = True
    while changed:
        changed = False
        for c, cs in list(cand.items()):   # naked singles
            if len(cs) == 1 and c not in K:
                place(c, next(iter(cs)))
                changed = True
        for line in range(n):   # hidden singles per row / column
            for axis in (0, 1):
                cells = [c for c in cand if c[axis] == line and c not in K]
                for v in range(n):
                    if any(K.get((line, j) if axis == 0 else (j, line)) == v for j in range(n)):
                        continue
                    where = [c for c in cells if v in cand[c]]
                    if len(where) == 1:
                        place(where[0], v)
                        changed = True
    derived = {c: K[c] for c in held_cells if c in K}
    for c, v in derived.items():
        assert v == mul[c], ("Latin propagation derived a wrong answer", c)
    for c in held_cells:
        assert int(mul[c]) in cand[c], ("true answer eliminated", c)
    return derived, cand


def split_cells(D: dict, tr: np.ndarray, te: np.ndarray) -> tuple[list, list]:
    return [tuple(int(v) for v in D["X"][i, :2]) for i in tr], [tuple(int(v) for v in D["X"][i, :2]) for i in te]


def readout_mask(D: dict, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """The rainbow's S4 read-out over te: fresh (no class-mate, i.e. commuting twin, trained) AND no identity input."""
    e = int(np.flatnonzero((D["mul"] == np.arange(24)).all(1))[0])
    nonid = np.array([D["X"][q, 0] != e and D["X"][q, 1] != e for q in te])
    return ts.fresh_mask(D, tr, te) & nonid


def levels(D: dict, tr: np.ndarray, te: np.ndarray) -> dict:
    tc, hc = split_cells(D, tr, te)
    der, cand = latin_singles(D["mul"], tc, hc)
    imp = np.array([c in der for c in hc])
    inv = np.array([1.0 / len(cand[c]) for c in hc])
    credit = np.where(imp, 1.0, inv)
    ro = readout_mask(D, tr, te)
    cl = ts.closure(D, tr, te)
    out = {"n_train": len(tr), "n_test": len(te), "assoc": cl["closure"], "assoc_latin": cl["closure_latin"],
           "latin_share": float(imp.mean()), "latin_level": float(credit.mean()), "n_nonimp": int((~imp).sum()),
           "chance_nonimp": float(inv[~imp].mean()) if (~imp).any() else None,
           "n_readout": int(ro.sum()), "latin_share_ro": float(imp[ro].mean()), "latin_level_ro": float(credit[ro].mean()),
           "imp_mask": imp, "inv_cand": inv}
    return out


def toy_seeds(frac: float, k: int = 5) -> tuple[list, list]:
    """The first k seeds whose toy split passes toy_struct.split's asserts (toyshape's rule; S4 0.2 -> 0, 1, 2, 4, 5)."""
    D, ok, bad, s = s4_task(), [], [], 0
    while len(ok) < k:
        try:
            ts.split(D, frac, s, "question")
            ok.append(s)
        except AssertionError:
            bad.append(s)
        s += 1
    return ok, bad


@lru_cache(maxsize=None)
def latin_split(frac: float, seed: int) -> dict:
    """For readout.py: Latin-only implied mask and 1/|candidates| over the held-out cells of the toy split, in te order."""
    D = s4_task()
    perm, tr, te = ts.split(D, frac, seed, "question")
    out = levels(D, tr, te)
    out["split_sha"] = __import__("hashlib").sha256(perm[: len(tr)].tobytes()).hexdigest()[:12]   # = toy_struct's split_sha
    return out


def rb6_splits() -> dict:
    """{seed: (split_seed, tr, te, recorded s4 record)} from the rb6 run JSONs (one per seed; all runs of a seed agree)."""
    D, out = s4_task(), {}
    for f in sorted(glob.glob(RB6)):
        r = json.load(open(f))
        S = r.get("domains", {}).get("s4")
        if S is None:
            continue
        seed = int(r["args"]["seed"])
        te = np.array([int(q) for q in S["eval_held"]["q"]])
        if seed in out:
            assert out[seed][0] == S["split_seed"] and np.array_equal(np.sort(out[seed][2]), np.sort(te)), ("rb6 split differs within a seed", f)
            continue
        tr = np.array(sorted(set(range(576)) - set(te.tolist())))
        assert len(tr) == S["n_train"] and len(te) == S["n_held"]
        _, tr2, te2 = ts.split(D, S["train_frac"], S["split_seed"], "question")   # the rainbow calls toy_struct.split
        assert np.array_equal(te2, te), ("rb6 held set != toy_struct.split(split_seed)", seed)
        out[seed] = (S["split_seed"], tr2, te2, S)
    return out


def main() -> None:
    D = s4_task()
    L = ["# S4 table-completion levels per seed (runs/toyf84fu/latin.py; toy_struct b74a78cb5472)\n\n",
         "assoc = associativity closure (toy `closure`); assoc+latin = toy `closure_latin`; latin = Latin-square property only "
         "(naked + hidden singles, A7 / A8). Shares are of the held-out cells; level = A7's partial-credit level; "
         "chance_nonimp = a pure table-filler's expected accuracy on the non-implied cells; ro = the rainbow read-out "
         "(fresh_nonid).\n\n",
         "| split | frac | seed | n_train / n_test | assoc | assoc+latin | latin share | latin level | n non-implied | chance non-implied | n_ro | latin share ro | latin level ro |\n",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    summ = {}
    for frac in FRACS:
        seeds, bad = toy_seeds(frac)
        for s in seeds:
            _, tr, te = ts.split(D, frac, s, "question")
            v = levels(D, tr, te)
            summ.setdefault(("toy", frac), []).append(v)
            L.append(f"| toy | {frac} | {s} | {v['n_train']} / {v['n_test']} | {v['assoc']:.3f} | {v['assoc_latin']:.3f} | "
                     f"{v['latin_share']:.3f} | {v['latin_level']:.3f} | {v['n_nonimp']} | {fmt(v['chance_nonimp'])} | "
                     f"{v['n_readout']} | {v['latin_share_ro']:.3f} | {v['latin_level_ro']:.3f} |\n")
        if bad:
            L.append(f"| toy | {frac} | {','.join(map(str, bad))} | split refused by toy_struct.split (an item never trained as input / answer) | | | | | | | | | |\n")
    rb = rb6_splits()
    for s in sorted(rb):
        sseed, tr, te, S = rb[s]
        v = levels(D, tr, te)
        assert abs(v["assoc"] - S["closure"]) < 1e-12 and abs(v["assoc_latin"] - S["closure_latin"]) < 1e-12, ("recorded closure", s)
        assert v["n_readout"] == S["n_readout"], ("n_readout", s, v["n_readout"], S["n_readout"])
        if s in A7_LEVELS:
            assert abs(100 * v["latin_level_ro"] - A7_LEVELS[s]) < 0.006, ("A7 level", s, 100 * v["latin_level_ro"], A7_LEVELS[s])
        summ.setdefault(("rainbow", S["train_frac"]), []).append(v)
        L.append(f"| rainbow (split_seed {sseed}) | {S['train_frac']} | {s} | {v['n_train']} / {v['n_test']} | {v['assoc']:.3f} | "
                 f"{v['assoc_latin']:.3f} | {v['latin_share']:.3f} | {v['latin_level']:.3f} | {v['n_nonimp']} | {fmt(v['chance_nonimp'])} | "
                 f"{v['n_readout']} | {v['latin_share_ro']:.3f} | {v['latin_level_ro']:.3f} |\n")
    L.append("\nKnown answers checked (asserts): every rainbow seed's assoc / assoc+latin equal its recorded closure / "
             "closure_latin, n_readout equals the recorded n_readout, the held set equals toy_struct.split(train_frac, "
             f"split_seed); the latin level on the read-out equals A7's printed levels for seeds {sorted(A7_LEVELS)} (to 0.006 points).\n\n")
    L.append("| split | frac | seeds | assoc min | assoc+latin min | latin share range | latin level range | latin share ro range | latin level ro range |\n|---|---|---|---|---|---|---|---|---|\n")
    for (k, frac), vs in summ.items():
        rng = lambda key: f"{min(v[key] for v in vs):.3f}-{max(v[key] for v in vs):.3f}"
        L.append(f"| {k} | {frac} | {len(vs)} | {min(v['assoc'] for v in vs):.3f} | {min(v['assoc_latin'] for v in vs):.3f} | "
                 f"{rng('latin_share')} | {rng('latin_level')} | {rng('latin_share_ro')} | {rng('latin_level_ro')} |\n")
    (HERE / "latin_table.md").write_text("".join(L))
    print("".join(L))


if __name__ == "__main__":
    main()
