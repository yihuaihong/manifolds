"""Compare default-path outputs of origin/main vs ft999 (bitwise for JSON/md, array-exact for npz)."""
import sys
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
fa = sorted(p.name for p in a.iterdir()); fb = sorted(p.name for p in b.iterdir())
assert fa == fb, (fa, fb)
assert fa, "no outputs"
for n in fa:
    if n.endswith(".npz"):
        x, y = np.load(a / n), np.load(b / n)
        assert sorted(x.files) == sorted(y.files), n
        for k in x.files:
            assert x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and np.array_equal(x[k], y[k]), (n, k)
        print(f"{n}: arrays identical {[(k, x[k].shape) for k in x.files]}")
    else:
        same = (a / n).read_bytes() == (b / n).read_bytes()
        assert same, f"{n} differs"
        print(f"{n}: bytes identical ({(a / n).stat().st_size} B)")
print("REGRESSION OK:", len(fa), "files")
