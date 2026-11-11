"""pt6comb test: the read-out's metric definitions (readout_pt6comb.series_of) reproduce PUBLISHED numbers on real run JSONs
(read only):
  K1  F88 A8.3 (rb6, Burst results 522 / 523, seed 8): s4_hard area all6-lam1@1e-3 - none@1.6e-2 = +22.9 (FINDINGS per-seed list)
  K2  F88 confirm seeds 4-7 (rb6 470-477 all6-lam1@1e-3 / none@1e-3, 490-493 none@8e-3): numbers area all6 - none@1e-3 +57.6 and
      - none@8e-3 +33.8; S4 (held_fresh_nonid) final3 +44.1 / +42.9; dim strict final3 - none@8e-3 +55.6 (FINDINGS F88 table and
      its 'confirm seed' notes; the dim +55.6 is the F1b contrast, against none@8e-3)
  K3  F101 (ptdom2 Part S, seeds 8-15): area only-d@2e-3 - none: countries +4.8, people +1.2 (none@2e-3), cities +4.1 (none@4e-3)
  and on every run read: the numbers area = metrics['numbers'], the comparisons' series area = metrics['<d>_readout'].
-> real_known.out"""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0]]
import readout_pt6comb as RP   # noqa: E402

BU = Path("/scratch/yh6210/llmadd-burst/results")
P2 = Path("/scratch/yh6210/llmadd_torch/runs/ptdom2/out")
res = []


def ok(name, cond, val=""):
    res.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val != "" else ""))


def rj(d):
    f = sorted(Path(d).glob("*/run_*.json"))
    assert len(f) == 1, (d, f)
    return json.loads(f[0].read_text())


def area(r, m):
    return 100 * float(np.mean(RP.series_of(r, m)))


def final3(r, m):
    return 100 * float(np.mean(RP.series_of(r, m)[-3:]))


def task(n):
    d = sorted(BU.glob(f"{n}-*"))
    assert len(d) == 1, (n, d)
    return rj(d[0])


# K1
a, b = task(522), task(523)
assert a["args"]["seed"] == b["args"]["seed"] == 8
d1 = area(a, "s4") - area(b, "s4")
ok("K1 F88 A8.3 seed 8: s4_hard area all6 - none@1.6e-2 = +22.9", round(d1, 1) == 22.9, round(d1, 4))
# K2
num, num8, s4f, s4f8, dimf8 = [], [], [], [], []
sf = lambda r: 100 * float(np.mean([e["s4_held_fresh_nonid"] for e in r["curve"][-3:]]))
for s, (nn, aa, n8) in zip((4, 5, 6, 7), ((470, 471, 490), (472, 473, 491), (474, 475, 492), (476, 477, 493))):
    rn, ra, r8 = task(nn), task(aa), task(n8)
    assert rn["args"]["seed"] == ra["args"]["seed"] == r8["args"]["seed"] == s and rn["args"]["lr"] == ra["args"]["lr"] == 1e-3 and r8["args"]["lr"] == 8e-3
    for r in (rn, ra, r8):
        ok(f"K2 seed {s} {r['args']['tag']}: numbers area (6-bucket series) = metrics", abs(area(r, "numbers") - r["metrics"]["numbers"]["area"]) < 1e-6)
    num.append(area(ra, "numbers") - area(rn, "numbers")); num8.append(area(ra, "numbers") - area(r8, "numbers"))
    s4f.append(sf(ra) - sf(rn)); s4f8.append(sf(ra) - sf(r8))
    dimf8.append(final3(ra, "dim") - final3(r8, "dim"))
ok("K2 F88 confirm (seeds 4-7): numbers area all6 - none@1e-3 = +57.6", round(float(np.mean(num)), 1) == 57.6, [round(x, 2) for x in num])
ok("K2 F88 confirm: numbers area all6 - none@8e-3 = +33.8", round(float(np.mean(num8)), 1) == 33.8, [round(x, 2) for x in num8])
ok("K2 F88 confirm: S4 held_fresh_nonid final3 all6 - none@1e-3 = +44.1", round(float(np.mean(s4f)), 1) == 44.1, [round(x, 2) for x in s4f])
ok("K2 F88 confirm: S4 held_fresh_nonid final3 all6 - none@8e-3 = +42.9", round(float(np.mean(s4f8)), 1) == 42.9, [round(x, 2) for x in s4f8])
ok("K2 F88 confirm: dim strict final3 all6 - none@8e-3 = +55.6 (the read-out's dim series)", round(float(np.mean(dimf8)), 1) == 55.6, [round(x, 2) for x in dimf8])
# K3
want = {"countries": ("B1", "2e-3", 4.8), "people": ("B1", "2e-3", 1.2), "cities": ("A", "4e-3", 4.1)}
for d, (bt, nlr, val) in want.items():
    g = []
    for s in range(8, 16):
        ra, rn = rj(P2 / f"ptdom2-{bt}-only-{d}-lr2e-3-s{s}"), rj(P2 / f"ptdom2-{bt}-none-lr{nlr}-s{s}")
        for r in (ra, rn):
            assert abs(area(r, d) - r["metrics"][f"{d}_readout"]["area"]) < 1e-6, (d, s)
        g.append(area(ra, d) - area(rn, d))
    ok(f"K3 F101 {d}: area only-{d}@2e-3 - none@{nlr} over seeds 8-15 = {val:+.1f} (and every run's series area = its metric)",
       round(float(np.mean(g)), 1) == val, [round(x, 2) for x in g])
print(f"REAL KNOWN {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
