"""ptdom3 (2026-11-17): the toy3 read-out sizes under the trainer's own split stream (Rev H): SeedSequence(4400 + seed).spawn(18),
child 3 i -> sr = default_rng(child); base = int(sr.integers(2**31)) is sr's first draw (nothing draws from sr before the toy3
branch); attempts base + 0, 1, ... until toy3.split accepts. Seeds 0-2 (seed 0 = the screen, 1-2 = the replication seeds the
standing rule allows); d6 at train fractions 0.5 / 0.45 / 0.4. The trainer asserts n_readout >= 50 (kept)."""
import sys
import numpy as np
sys.path.insert(0, "/scratch/yh6210/llmadd-burst-ptdom3/tools")
import toy3
import toy_struct as ts


def readout(d, seed, frac=None):
    D = toy3.build(d)
    mode, f0 = toy3.SPLIT[d]
    frac = f0 if frac is None else frac
    ch = np.random.SeedSequence(toy3.TOY3_STREAM + seed).spawn(3 * len(toy3.TOY3))[3 * toy3.TOY3.index(d)]
    sr = np.random.default_rng(ch)
    base = int(sr.integers(2 ** 31))
    for att in range(200):
        try:
            tr, te = toy3.split(D, frac, base + att, mode)
            break
        except AssertionError:
            continue
    else:
        return None
    m, name, ro = toy3.masks(D, tr, te, ts.s4_closure)
    return len(tr), len(te), int(ro.sum()), name, att


for d in toy3.TOY3:
    for seed in (0, 1, 2):
        print(d, "seed", seed, "frac", toy3.SPLIT[d][1], "-> train, held, read-out, set, attempt:", readout(d, seed))
for frac in (0.45, 0.4):
    for seed in (0, 1, 2):
        print("d6 seed", seed, "frac", frac, "->", readout("d6", seed, frac))
