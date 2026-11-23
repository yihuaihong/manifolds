#!/usr/bin/env python3
"""pt6v2 Amendment V7b: the in-place drop's own checks on one run JSON. LOG AND FLAG ONLY: always exits 0 (unless it crashes);
the verdict is the last line ("V7B CHECKS PASS" / "V7B CHECKS FAIL") and "pass" in the out JSON.
usage: check_v7b.py <run JSON> <out JSON> <domains@step> [--main] [--ref <registered all6 run JSON, same seed>]
                    [--v7 <V7 (all6drop, drop_fill left) run JSON, same seed>]
Recomputed here: the line-cycling rule (domain of line j of row r at step t = EXT[((t*bs + r)*K + j) % len(EXT)]) gives the
dropped-line counts; with --drop_fill inplace every dropped line becomes one filler, so the filler count = the dropped count.
--ref (the registered all6 run of the seed): every kept line keeps its row, positions and slot, so at every common eval
  - the slot share (slot_frac) of every kept domain = the registered run's EXACTLY (the dim slot-1 share is printed);
  - dom_sites and the token share of every kept line type = the registered run's;
  - data_sha = the registered run's before the drop and different from it after.
--v7 (the V7 run of the seed): the token share of EVERY line type, text included (fillers count as text) = V7's at every
  common eval (the same total text tokens); data_sha = V7's before the drop and different after."""
import json
import math
import sys
from pathlib import Path

import numpy as np

run, out_json, spec = sys.argv[1], sys.argv[2], sys.argv[3]
MAIN = "--main" in sys.argv
opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None   # noqa: E731
REF, V7 = opt("--ref"), opt("--v7")
checks, info = [], {}


def chk(name, ok, val=""):
    checks.append({"check": name, "ok": bool(ok), "value": val if isinstance(val, (int, float, str, bool, type(None))) else str(val)[:600]})


r = json.loads(Path(run).read_text())
a, c, steps = r["args"], r["curve"], r["steps"]
dd, ds = spec.split("@")
DROP, D = [x for x in dd.split(",") if x], int(ds)
EXT = [x for x in a["domains"].split(",") if x != "numbers"]
KEPT = [d for d in EXT if d not in DROP]
bs, K = a["bs"], a["extra_lines"]
st = [e["step"] for e in c]
dj = r.get("drop_domains_after", {})
chk(f"flag consumed: args drop_domains_after = {spec!r}, drop_fill = 'inplace'; run JSON drop_domains_after = {DROP}@{D}, "
    f"fill inplace, fill_stream 7100 + seed",
    a.get("drop_domains_after") == spec and a.get("drop_fill") == "inplace" and dj.get("domains") == DROP and dj.get("step") == D
    and dj.get("fill") == "inplace" and dj.get("fill_stream") == 7100 + a["seed"],
    {"arg": [a.get("drop_domains_after"), a.get("drop_fill")], "json": dj})
chk("eval site keys = EXT in order", list(c[0]["dom_sites"]) == EXT, list(c[0]["dom_sites"]))
if MAIN:
    w0, w1 = (float(v) for v in a["window"].split(","))
    on = [s for s in range(steps) if w0 * steps <= s < w1 * steps]
    chk(f"drop step = the first step after the prior window (window {on[0]}..{on[-1]})", D == on[-1] + 1, [on[0], on[-1], D])


def line_doms(t0, t1):
    t = np.arange(t0, t1)[:, None, None]; rr = np.arange(bs)[None, :, None]; j = np.arange(K)[None, None, :]
    return ((t * bs + rr) * K + j) % len(EXT)


bad_sites, bad_drop, bad_fill = [], [], []
for e in c:
    s = e["step"]
    ss = e["dom_sites"]
    ok_ = all(ss[d] > 0 for d in EXT) if s < D else (all(ss[d] == 0 for d in DROP) and all(ss[d] > 0 for d in KEPT))
    if not ok_:
        bad_sites.append((s, ss))
    if s >= D:
        L = line_doms(D, s + 1)
        want = {d: int((L == EXT.index(d)).sum()) for d in DROP}
    else:
        want = {d: 0 for d in DROP}
    if e.get("dropped_lines") != want:
        bad_drop.append((s, e.get("dropped_lines"), want))
    f = e.get("fill") or {}
    if f.get("lines") != sum(want.values()) or (f.get("tokens", -1) > 0) != (s >= D):
        bad_fill.append((s, f, sum(want.values())))
chk(f"dom_sites: every domain > 0 at every eval before step {D}; {DROP} = 0 and every kept domain > 0 at every eval from it",
    not bad_sites, bad_sites[:3])
chk("dropped_lines at every eval = the count recomputed from the cycling rule and the drop step", not bad_drop and len(c) > 0,
    bad_drop[:3] or (c[-1].get("dropped_lines") if c else None))
chk("fill.lines at every eval = the dropped lines so far (one filler per dropped line); fill.tokens > 0 only from the drop",
    not bad_fill and len(c) > 0, bad_fill[:3] or (c[-1].get("fill") if c else None))
chk("run JSON fill_counts / dropped_lines = the last eval's", dj.get("fill_counts") == c[-1].get("fill")
    and dj.get("dropped_lines") == c[-1].get("dropped_lines"), dj)
# without --ref: a filler keeps the slot, so every kept domain's lines sit in two-line rows -> slot 1 share = 1/K (+- 4 sd)
bad_slot = []
L = line_doms(0, st[-1] + 1)
for d in KEPT:
    m = int((L == EXT.index(d)).sum())
    got = c[-1]["slot_frac"][d]
    tol = max(0.02, 4 * math.sqrt((1 / K) * (1 - 1 / K) / m))
    info[f"slot1_{d}"] = round(got[0], 5)
    if len(got) != K or abs(sum(got) - 1) > 1e-9 or abs(got[0] - 1 / K) > tol:
        bad_slot.append((d, got, 1 / K, tol))
chk("slot share of every kept domain at the last eval: slot 1 = 1/K +- max(0.02, 4 sd) (no slot shift: fillers keep the slots)",
    not bad_slot, bad_slot or {d: info[f"slot1_{d}"] for d in KEPT})
gbad = [(e["step"], d) for e in c if e["step"] >= D for d in DROP if e.get(f"geo_{d}") is not None]
chk(f"geo_<d> of {DROP} not logged from step {D} on (no sites)", not gbad, gbad[:4])
evk = [k for k in c[0] if any(k == f"{d}_held" for d in DROP)]
ebad = [(e["step"], k) for e in c for k in evk if not (isinstance(e.get(k), (int, float)) and math.isfinite(e[k]))]
chk(f"the dropped domains' held-out evals ({evk}) logged and finite at every eval", len(evk) == len(DROP) and not ebad, ebad[:4])


def common_with(path, tag):
    if not path or not Path(path).exists():
        chk(f"{tag} run JSON present ({path})", False, path)
        return None, []
    rf = json.loads(Path(path).read_text())
    rc = {e["step"]: e for e in rf["curve"]}
    return rc, [e for e in c if e["step"] in rc]


rc, common = common_with(REF, "--ref (registered all6)")
if rc is not None:
    pre_ = [e for e in common if e["step"] < D]; post_ = [e for e in common if e["step"] >= D]
    chk(f"data_sha = the registered run's at every common eval before step {D} ({len(pre_)}) and different at every one from it "
        f"({len(post_)})", bool(pre_) and all(e["data_sha"] == rc[e["step"]]["data_sha"] for e in pre_)
        and all(e["data_sha"] != rc[e["step"]]["data_sha"] for e in post_), [(e["step"], e["data_sha"], rc[e["step"]]["data_sha"])
                                                                           for e in common][:2])
    sb = [(e["step"], d, e["slot_frac"][d], rc[e["step"]]["slot_frac"][d]) for e in common for d in KEPT
          if any(abs(x - y) > 1e-12 for x, y in zip(e["slot_frac"][d], rc[e["step"]]["slot_frac"][d]))
          or len(e["slot_frac"][d]) != len(rc[e["step"]]["slot_frac"][d])]
    last = common[-1]["step"] if common else None
    if common:
        info["dim_slot1_last_common"] = {"step": last, "v7b": c[st.index(last)]["slot_frac"]["dim"][0],
                                         "registered_all6": rc[last]["slot_frac"]["dim"][0]}
    chk(f"slot share of every kept domain {KEPT} = the registered all6 run's at every common eval ({len(common)}; dim slot 1 at "
        f"step {last}: {info.get('dim_slot1_last_common')})", bool(common) and not sb, sb[:4])
    sib = [(e["step"], d, e["dom_sites"][d], rc[e["step"]]["dom_sites"][d]) for e in common for d in KEPT
           if e["dom_sites"][d] != rc[e["step"]]["dom_sites"][d]]
    chk(f"dom_sites of every kept domain = the registered run's at every common eval", bool(common) and not sib, sib[:4])
    kept_t = [t for t in c[0]["token_share"] if t not in DROP and t != "text"]
    tb = [(e["step"], t, e["token_share"][t], rc[e["step"]]["token_share"][t]) for e in common for t in kept_t
          if abs(e["token_share"][t] - rc[e["step"]]["token_share"][t]) > 1e-12]
    chk(f"token share of every kept line type {kept_t} = the registered run's at every common eval", bool(common) and not tb, tb[:4])
    for e in c:
        if e["step"] == 0 and 0 in rc:
            chk("step-0 lm = the registered run's (rel. 1e-6; same data, same init; device numerics may differ: flag only)",
                abs(e["lm"] - rc[0]["lm"]) <= 1e-6 * abs(rc[0]["lm"]), [e["lm"], rc[0]["lm"]])
vc, vcommon = common_with(V7, "--v7 (V7 all6drop)")
if vc is not None:
    tb = [(e["step"], t, e["token_share"][t], vc[e["step"]]["token_share"].get(t)) for e in vcommon for t in e["token_share"]
          if abs(e["token_share"][t] - vc[e["step"]]["token_share"].get(t, -1)) > 1e-12]
    chk(f"token share of EVERY line type, text included (fillers = text), = the V7 run's at every common eval ({len(vcommon)})",
        bool(vcommon) and not tb, tb[:4])
    pre_ = [e for e in vcommon if e["step"] < D]; post_ = [e for e in vcommon if e["step"] >= D]
    chk(f"data_sha = the V7 run's before step {D} ({len(pre_)}) and different from it after ({len(post_)})",
        bool(pre_) and all(e["data_sha"] == vc[e["step"]]["data_sha"] for e in pre_)
        and all(e["data_sha"] != vc[e["step"]]["data_sha"] for e in post_), "")
FLAGC = ("step-0 lm = the registered run's",)
ok = all(k["ok"] for k in checks)
flag_only = not ok and all(k["ok"] or k["check"].startswith(FLAGC) for k in checks)
Path(out_json).write_text(json.dumps({"pass": ok, "valid_data_flagged": flag_only, "checks": checks, "info": info}, indent=1))
for k in checks:
    print(("PASS " if k["ok"] else "FAIL ") + k["check"] + f"  ({k['value']})")
print(json.dumps(info))
print("V7B CHECKS", "PASS" if ok else ("FAIL (flag-only: step-0 lm numerics)" if flag_only else "FAIL"))
sys.exit(0)   # log and flag only (AUDIT.md Amendment V7b): never fails the job
