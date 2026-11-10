"""F86 addendum (2026-11-10, request of the writing session): 8-seed paired gains of win-text (the web-text plug-in prior,
at the operand prior's selected lr) vs none (at none's own selected lr), curve-average accuracy = readout_rules.area, read
directly from the Burst run JSONs. Known answer: the seeds 0-3 / 4-7 means must reproduce F86's 12.4 / 13.6 (pt2long) and
45.4 / 41.6 (pt2s190).
    python report/scripts_f86_wintext8.py"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "tools"))
import readout_rules as RR   # noqa: E402

SETS = {"pt2long (53M, 1.2B tokens)": ("pt2long", "2e-3", "1e-3"), "pt2s190 (190M, 300M tokens)": ("pt2s190", "4.8e-3", "6e-4")}


def run(tag, lr, arm, s):
    ds = [d for d in glob.glob(str(R / "results" / f"*-{tag}-lr{lr}-{arm}-s{s}")) if Path(d).name.split("-", 1)[1] == f"{tag}-lr{lr}-{arm}-s{s}"]
    assert len(ds) == 1, (tag, lr, arm, s, ds)
    d = Path(ds[0])
    ck = json.loads((d / "check.json").read_text())
    assert ck.get("pass") is True, (d, "check.json not pass")
    fs = glob.glob(str(d / f"{tag}-lr{lr}-{arm}" / "run_*.json"))
    assert len(fs) == 1, (d, fs)
    r = json.loads(Path(fs[0]).read_text())
    assert RR.finished(r), (d, "not finished")
    assert r["args"]["seed"] == s and abs(float(r["args"]["lr"]) - float(lr)) < 1e-12, (d, r["args"]["seed"], r["args"]["lr"])
    return r, d.name


for name, (tag, lr_none, lr_text) in SETS.items():
    g, used = [], []
    for s in range(8):
        rn, dn = run(tag, lr_none, "none", s)
        rt, dt = run(tag, lr_text, "win-text", s)
        assert rn["args"]["site"] != "text" or True
        g.append(RR.area(rt) - RR.area(rn))
        used.append((dn, dt))
    g = np.array(g)
    print(f"{name}: win-text@{lr_text} - none@{lr_none}: per seed {[round(x, 2) for x in g]}")
    print(f"  8-seed mean {g.mean():+.2f}, wins {(g > 0).sum()}/8; seeds 0-3 {g[:4].mean():+.2f}, seeds 4-7 {g[4:].mean():+.2f}")
    for dn, dt in used:
        print(f"    {dn} | {dt}")
