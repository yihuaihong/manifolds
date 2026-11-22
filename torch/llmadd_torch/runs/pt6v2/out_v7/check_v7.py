#!/usr/bin/env python3
"""pt6v2 Amendment V7: the drop's own checks on one run JSON (independent of the trainer's bookkeeping where possible).
usage: check_v7.py <run JSON> <out JSON> <domains@step> [--main] [--ref <registered run JSON of the same seed>]
Expectations are recomputed here from the line-cycling rule (domain of line j of row r at step t =
EXT[((t*bs + r)*K + j) % len(EXT)]) and the drop step; --ref compares the data stream with the registered run of the same seed
(data_sha identical at every eval before the drop and different after it; the token share of every kept line type identical at
every eval: the same lines in the same steps, the same total tokens). Exit 0 = all pass, 5 = a check failed."""
import json
import math
import sys
from pathlib import Path

import numpy as np

run, out_json, spec = sys.argv[1], sys.argv[2], sys.argv[3]
MAIN = "--main" in sys.argv
REF = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
checks = []


def chk(name, ok, val=""):
    checks.append({"check": name, "ok": bool(ok), "value": val if isinstance(val, (int, float, str, bool, type(None))) else str(val)[:600]})


r = json.loads(Path(run).read_text())
a, c, steps = r["args"], r["curve"], r["steps"]
dd, ds = spec.split("@")
DROP, D = [x for x in dd.split(",") if x], int(ds)
EXT = [x for x in a["domains"].split(",") if x != "numbers"]
bs, K = a["bs"], a["extra_lines"]
st = [e["step"] for e in c]
chk(f"flag consumed: args.drop_domains_after = {spec!r} and run JSON drop_domains_after = domains {DROP}, step {D}",
    a.get("drop_domains_after") == spec and r.get("drop_domains_after", {}).get("domains") == DROP
    and r["drop_domains_after"].get("step") == D, {"arg": a.get("drop_domains_after"), "json": r.get("drop_domains_after")})
chk("eval site keys = EXT in order", list(c[0]["dom_sites"]) == EXT, list(c[0]["dom_sites"]))
if MAIN:
    w0, w1 = (float(v) for v in a["window"].split(","))
    on = [s for s in range(steps) if w0 * steps <= s < w1 * steps]
    chk(f"drop step = the first step after the prior window (window {on[0]}..{on[-1]})", D == on[-1] + 1, [on[0], on[-1], D])


def line_doms(t0, t1):
    """domain index of every line of steps t0..t1-1: array (steps, bs, K)."""
    t = np.arange(t0, t1)[:, None, None]; rr = np.arange(bs)[None, :, None]; j = np.arange(K)[None, None, :]
    return ((t * bs + rr) * K + j) % len(EXT)


di = [EXT.index(d) for d in DROP]
bad_sites, bad_drop = [], []
for e in c:
    s = e["step"]
    ss = e["dom_sites"]
    if s < D:
        ok_ = all(ss[d] > 0 for d in EXT)
    else:
        ok_ = all(ss[d] == 0 for d in DROP) and all(ss[d] > 0 for d in EXT if d not in DROP)
    if not ok_:
        bad_sites.append((s, ss))
    if s >= D:
        L = line_doms(D, s + 1)
        want = {d: int((L == EXT.index(d)).sum()) for d in DROP}
    else:
        want = {d: 0 for d in DROP}
    if e.get("dropped_lines") != want:
        bad_drop.append((s, e.get("dropped_lines"), want))
chk(f"dom_sites: every domain > 0 at every eval before step {D}; {DROP} = 0 and every other domain > 0 at every eval from it",
    not bad_sites, bad_sites[:3])
chk("dropped_lines at every eval = the count recomputed from the cycling rule and the drop step", not bad_drop and len(c) > 0,
    bad_drop[:3] or (c[-1].get("dropped_lines") if c else None))
chk("run JSON drop_domains_after.dropped_lines = the last eval's dropped_lines",
    r.get("drop_domains_after", {}).get("dropped_lines") == c[-1].get("dropped_lines"), r.get("drop_domains_after"))
# slot share after the numbers line at the last eval: lines before the drop and rows that keep two lines -> 1/K each (random
# order); a row that keeps one line puts it in slot 1
S = st[-1]
L = line_doms(0, S + 1)
post = np.arange(0, S + 1) >= D
keep = ~np.isin(L, di) | ~post[:, None, None]
nkeep = keep.sum(2)
bad_slot, info_slot = [], {}
for d in EXT:
    k = EXT.index(d)
    m = (L == k) & keep
    alone = int((m & (nkeep[:, :, None] == 1)).sum()); multi = int(m.sum()) - alone
    tot = alone + multi
    exp1 = (multi / K + alone) / tot
    sd = math.sqrt(multi * (1 / K) * (1 - 1 / K)) / tot
    tol = max(0.05, 4 * sd)
    got = e_ = c[-1]["slot_frac"][d]
    info_slot[d] = [round(got[0], 4), round(exp1, 4), alone, multi]
    if len(got) != K or abs(sum(got) - 1) > 1e-9 or abs(got[0] - exp1) > tol:
        bad_slot.append((d, got, exp1, tol))
chk("slot share at the last eval: slot 1 = (lines in two-line rows / K + lines alone in their row) / lines, +- max(0.05, 4 sd)",
    not bad_slot, bad_slot or info_slot)
gbad = [(e["step"], d) for e in c if e["step"] >= D for d in DROP if e.get(f"geo_{d}") is not None]
chk(f"geo_<d> of {DROP} not logged from step {D} on (no sites)", not gbad, gbad[:4])
evk = [k for k in c[0] if any(k == f"{d}_held" for d in DROP)]
ebad = [(e["step"], k) for e in c for k in evk if not (isinstance(e.get(k), (int, float)) and math.isfinite(e[k]))]
chk(f"the dropped domains' held-out evals ({evk}) logged and finite at every eval", len(evk) == len(DROP) and not ebad, ebad[:4])
if REF:
    rf = json.loads(Path(REF).read_text())
    rc = {e["step"]: e for e in rf["curve"]}
    common = [e for e in c if e["step"] in rc]
    pre_ = [e for e in common if e["step"] < D]; post_ = [e for e in common if e["step"] >= D]
    chk(f"data_sha = the registered run's at every common eval before step {D} ({len(pre_)}) and different at every one from it "
        f"({len(post_)})", bool(pre_) and all(e["data_sha"] == rc[e["step"]]["data_sha"] for e in pre_)
        and all(e["data_sha"] != rc[e["step"]]["data_sha"] for e in post_), [(e["step"], e["data_sha"], rc[e["step"]]["data_sha"])
                                                                           for e in common][:2])
    kept = [t for t in c[0]["token_share"] if t not in DROP and t != "text"]
    tb = [(e["step"], t, e["token_share"][t], rc[e["step"]]["token_share"][t]) for e in common for t in kept
          if abs(e["token_share"][t] - rc[e["step"]]["token_share"][t]) > 1e-12]
    chk(f"token share of every kept line type {kept} = the registered run's at every common eval (same lines, same total tokens)",
        bool(common) and not tb, tb[:4])
    for e in c:
        if e["step"] == 0 and 0 in rc:
            chk("step-0 lm = the registered run's (rel. 1e-6; same data, same init; device numerics may differ: flag only)",
                abs(e["lm"] - rc[0]["lm"]) <= 1e-6 * abs(rc[0]["lm"]), [e["lm"], rc[0]["lm"]])
FLAGC = ("step-0 lm = the registered run's",)
ok = all(k["ok"] for k in checks)
flag_only = not ok and all(k["ok"] or k["check"].startswith(FLAGC) for k in checks)
Path(out_json).write_text(json.dumps({"pass": ok, "valid_data_flagged": flag_only, "checks": checks}, indent=1))
for k in checks:
    print(("PASS " if k["ok"] else "FAIL ") + k["check"] + f"  ({k['value']})")
print("V7 CHECKS", "PASS" if ok else ("FAIL (flag-only)" if flag_only else "FAIL"))
sys.exit(0 if ok or flag_only else 5)
