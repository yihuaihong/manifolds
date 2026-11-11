"""pt2same Amendment A1 item 3 read-out, exactly as defined in AUDIT.md 'Amendment A2' (written and self-tested BEFORE any
A1 result existed). Descriptive only: it never changes the primary verdict, which it prints first, unchanged, by running
readout_pt2same.py's own path (imported as a module, never edited, its globals never touched; its sha is asserted).
  (a) seeds 0-3 only, on both sides; lrs are never mixed to make 8 seeds; no 4-seed slope is compared with the 8-seed one
  (b) Torch best lr of 53M none over {4e-3, 8e-3, 1.6e-2} by pt2main rule 1 (readout_rules.best_lr, near 'within' 2,
      tie -> the lower offline bpb taken as tools/pt2main_summary.py:102-104), readout_rules.bracket; an open bracket is
      reported 'open bracket, not extended'
  (c) the three-size slope of Delta on ln N_nonemb on seeds 0-3 with 53M none at 8e-3 vs at the Torch best lr (every other
      cell at its pre-registered pick, seeds 0-3), with the identity slope difference = w53 x (none@8e-3 - none@best)
  (d) leave-one-seed-out lr choice (rule 1 on 3 seeds, the held-out seed's area at that lr, averaged) next to (c)
  (e) 53M none areas at the three lrs, Torch seeds 0-3 next to Burst seeds 0-3 (F86 pt2b runs), Torch - Burst by seed
  (f) the 12 Torch runs must pass the main read-out's checks (+ a valid full-val offline_lm entry, pt2main rule 6), and the
      primary read-out must have a verdict; otherwise every A1 line prints 'not computed' with the reasons
  (g) printed in its own section after the primary verdict; output JSON separate (never readout.json)
    python readout_pt2same_a1.py [--json readout_a1.json] [--investigated "<note>"] [--primary-json readout.json]
    python readout_pt2same_a1.py --selftest     (synthetic trees only; touches no result)
    python readout_pt2same_a1.py --known        (published Burst pt2b runs only: loader known answers)"""
import contextlib
import copy
import hashlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent              # runs/pt2same
B = HERE.parent.parent                              # /scratch/yh6210/llmadd_torch
RO_SHA = "bf7cced4323878b2bd0d13db83c451a1f2e2eb3b4019fd747711a9813590c88c"
_ro = hashlib.sha256((HERE / "readout_pt2same.py").read_bytes()).hexdigest()
assert _ro == RO_SHA, f"readout_pt2same.py changed: {_ro}"
_frozen = {ln.split()[1]: ln.split()[0] for ln in (HERE / "FROZEN.sha256").read_text().splitlines() if ln.strip()}
assert _frozen["runs/pt2same/readout_pt2same.py"] == RO_SHA, "FROZEN.sha256 disagrees on readout_pt2same.py"
sys.path.insert(0, str(HERE))
import readout_pt2same as RO   # noqa: E402  (loader, integrity, slope helpers; used read-only)
RR = RO.RR

LRS = ("4e-3", "8e-3", "1.6e-2")                    # sorted by value
PICK = RO.PICKS["53m"]["none"]
assert PICK == "8e-3" and list(LRS) == sorted(LRS, key=float)
S4 = tuple(RO.H1)                                   # seeds 0-3
assert S4 == (0, 1, 2, 3)
X = [math.log(RO.N_NONEMB[z]) for z in RO.SIZES]
W = {z: (x - np.mean(X)) / sum((v - np.mean(X)) ** 2 for v in X) for z, x in zip(RO.SIZES, X)}   # OLS weights
W53 = float(W["53m"])                               # -0.316653 (AUDIT A1 item 1: -0.317)
assert abs(W53 + 0.317) < 5e-4, W53
BURST_NAME = {}                                     # (lr, seed) -> Burst pt2b task (F86; one task per lr and seed)
for _s in S4:
    BURST_NAME[("4e-3", _s)] = f"{350 + _s}-pt2b-lr4e-3-none-s{_s}"
    BURST_NAME[("8e-3", _s)] = RO.BURST_TASK[("53m", "none", _s)]
    BURST_NAME[("1.6e-2", _s)] = f"{362 + _s}-pt2b-lr1.6e-2-none-s{_s}"
NC = "not computed"


def a1_paths(site, lr, s, roots):
    """(run JSON, check.json, tag) of 53M none at lr, seed s; at lr = the pick it must equal RO.run_paths (double path)"""
    if site == "torch":
        tag = f"pt2same-53m-lr{lr}-none"
        d = roots["torch"] / f"{tag}-s{s}"
    else:
        name = BURST_NAME[(lr, s)]
        tag, d = name[4:name.rindex("-s")], roots["burst"] / name
    run, chk = d / tag / f"run_whole_none_seed{s}_{tag}.json", d / "check.json"
    if lr == PICK:
        assert (run, chk) == RO.run_paths(site, "53m", "none", s, roots), (site, lr, s)
    return run, chk, tag


def load_a1_run(run, chk, tag, s):
    """RO.load_run's checks, then the offline full-val bpb exactly as tools/pt2main_summary.py:102-104 (missing -> WITHHELD,
    pt2main rule 6) with check.json's bpb_full as its double path; the in-training bpb = the last eval's val_bits_per_byte.
    -> (status, why, {"area", "off", "intr"})"""
    st, a, info = RO.load_run(run, chk)
    if st != "ok":
        return st, info, None
    if not math.isfinite(a):
        return "WITHHELD", f"non-finite area {a}", None
    r = json.loads(run.read_text())
    div = RR.diverged_at(r)
    offf = chk.parent / "offline_lm.json"
    off = json.loads(offf.read_text()) if offf.exists() else {}
    o = [v for v in off.values() if v.get("tag") == f"_{tag}" and v.get("seed") == s and v.get("max_windows") == 0]
    ob = RR.offline_bpb(o[0], div is not None) if o else None
    if ob is None:
        return "WITHHELD", "no valid full-val offline_lm entry (pt2main rule 6)" + (" (|orig_minus_logged| >= 0.002)" if o else ""), None
    cb = json.loads(chk.read_text())["info"].get("bpb_full")
    if not (cb == ob or (isinstance(cb, float) and isinstance(ob, float) and math.isnan(cb) and math.isnan(ob))):
        return "WITHHELD", f"double path: offline bpb {ob} != check.json bpb_full {cb}", None
    return "ok", "", {"area": a, "off": ob, "intr": r["curve"][-1].get("val_bits_per_byte")}


def load_a1(roots, site="torch"):
    """{lr: {"area": [4], "off": [4], "intr": [4], "status": [4], "why": [4]}} over seeds 0-3"""
    out = {}
    for lr in LRS:
        c = {"area": [], "off": [], "intr": [], "status": [], "why": []}
        for s in S4:
            run, chk, tag = a1_paths(site, lr, s, roots)
            st, why, v = load_a1_run(run, chk, tag, s)
            c["status"].append(st)
            c["why"].append(why)
            for k in ("area", "off", "intr"):
                c[k].append(v[k] if v else None)
        out[lr] = c
    return out


def slope_h1(G):
    """OLS slope of the seeds 0-3 mean Delta on ln N_nonemb (53M, 190M, 505M); G: {size: per-seed Delta, >= seeds 0-3}"""
    y = [float(np.mean([G[z][s] for s in S4])) for z in RO.SIZES]
    sl = RO.ols_slope(X, y)
    assert abs(sl - np.polyfit(X, y, 1)[0]) < 1e-9, (sl, y)   # double path
    return sl


def rule1(T, seeds):
    """pt2main rule 1 over the three lrs on these seeds"""
    i = [S4.index(s) for s in seeds]
    return RR.best_lr({lr: {k: [T[lr][k][j] for j in i] for k in ("area", "off", "intr")} for lr in LRS})


def compute(T, A, G):
    """T: load_a1 (all ok); A: {(size, arm): 8 Torch areas} of the primary; G: {size: 8 Deltas} of the primary"""
    sel = rule1(T, S4)
    best = sel["best"]
    br = RR.bracket(list(LRS), sel["mean"], best)
    prior53 = np.asarray(A[("53m", "win-ctx")], float)
    n8 = np.asarray(T[PICK]["area"], float)
    nb = np.asarray(T[best]["area"], float)
    s_prim = slope_h1(G)
    s_best = slope_h1({**G, "53m": prior53[:4] - nb})
    folds = []
    for h in S4:
        f = rule1(T, [s for s in S4 if s != h])
        folds.append({"held_out": h, "lr": f["best"], "near": f["near"], "area": float(T[f["best"]]["area"][h])})
    nl = np.asarray([f["area"] for f in folds])
    s_loso = slope_h1({**G, "53m": prior53[:4] - nl})
    ident = {}
    for k, s_, nv in (("best", s_best, nb), ("loso", s_loso, nl)):
        lhs, rhs = s_ - s_prim, W53 * (n8.mean() - nv.mean())
        assert abs(lhs - rhs) < 1e-9, (k, lhs, rhs)   # identity: only the 53M none mean moved
        ident[k] = {"slope_diff": float(lhs), "w53_x_none_diff": float(rhs)}
    return {"best": best, "near": sel["near"], "offline": sel["offline"], "mean": sel["mean"], "bpb": sel["bpb"],
            "bracket": list(br) if br else None, "none_pick_mean": float(n8.mean()), "none_best_mean": float(nb.mean()),
            "none_loso_mean": float(nl.mean()), "folds": folds, "slope_h1_pick": s_prim, "slope_h1_best": s_best,
            "slope_h1_loso": s_loso, "identity": ident, "w53": W53,
            "delta53_h1": {"pick": float((prior53[:4] - n8).mean()), "best": float((prior53[:4] - nb).mean()),
                           "loso": float((prior53[:4] - nl).mean())}}


def fmt4(v, sign=""):
    return " ".join(f"{x:{sign}6.2f}" if x is not None else "   n/a" for x in v)


def section(res, T, Bu, A, G):
    """the A1 section (prints; returns the JSON dict)"""
    out = {"definition": "AUDIT.md Amendment A2", "seeds": list(S4), "lrs": list(LRS)}
    print("\n== Amendment A1 item 3 / A2: 53M none lr sensitivity on Torch (descriptive; does NOT change the VERDICT above)")
    print("  (a) seeds 0-3 only on both sides; 4-seed point estimates; no CI, no verdict; never compared with the 8-seed slope")
    lines = ("(b) Torch best lr of 53M none (pt2main rule 1)", "(c) seeds 0-3 slope, 53M none @8e-3 vs @Torch best",
             "(d) leave-one-seed-out lr choice", "(e) 53M none area by lr, Torch vs Burst")
    reasons = []
    if res.get("verdict") is None or res["verdict"].startswith(("no verdict", "PENDING")):
        reasons.append(f"the primary read-out has no verdict ({res.get('verdict')})")
    if T is not None:
        for lr in LRS:
            for s, (st, w) in enumerate(zip(T[lr]["status"], T[lr]["why"])):
                if st != "ok":
                    reasons.append(f"{st} torch 53m none lr {lr} seed {s}: {w}")
    if reasons:
        for r_ in reasons:
            print(f"  (f) {r_}")
        for ln in lines:
            print(f"  {ln}: {NC}")
        out.update(status=NC, reasons=reasons)
        return out
    a = compute(T, A, G)
    out.update(status="computed", **a)
    print("  (f) integrity: the 12 Torch runs (3 lrs x seeds 0-3) present, finished, passing, area and offline bpb double paths")
    bt = a["bracket"]
    brt = ("open bracket at the %s edge, not extended" % bt[1] if bt and bt[0] == "open" else
           "note: area argmax at the %s edge by <= 2 (not an open bracket by rule 1)" % bt[1] if bt else "bracket closed")
    print(f"  (b) Torch best lr: {a['best']} (means " + ", ".join(f"{lr} {a['mean'][lr]:.2f}" for lr in LRS)
          + f"; within 2 of the top: {', '.join(a['near'])}"
          + (f"; tie broken on the {'offline' if a['offline'] else 'in-training *'} bpb "
             + ", ".join(f"{lr} {a['bpb'][lr]:.4f}" for lr in a["near"]) if len(a["near"]) > 1 else "")
          + f"); {brt}")
    print(f"  (c) slope of the seeds 0-3 mean Delta on ln N_nonemb (53M / 190M / 505M, every other cell at its pick):")
    print(f"      53M none @{PICK} (pre-registered pick): none 0-3 {a['none_pick_mean']:6.2f}  Delta53 0-3 "
          f"{a['delta53_h1']['pick']:+7.2f}  slope 0-3 {a['slope_h1_pick']:+.3f}")
    print(f"      53M none @{a['best']} (Torch best-of-3, chosen and evaluated on seeds 0-3): none 0-3 {a['none_best_mean']:6.2f}"
          f"  Delta53 0-3 {a['delta53_h1']['best']:+7.2f}  slope 0-3 {a['slope_h1_best']:+.3f}  "
          f"(difference {a['identity']['best']['slope_diff']:+.3f})")
    print(f"      identity: difference = w53 x (none@{PICK} - none@best) = {W53:+.6f} x "
          f"({a['none_pick_mean']:.3f} - {a['none_best_mean']:.3f}) = {a['identity']['best']['w53_x_none_diff']:+.3f}  OK")
    print(f"  (d) leave-one-seed-out: " + "; ".join(f"held-out {f['held_out']}: {f['lr']} -> {f['area']:.2f}" for f in a["folds"]))
    print(f"      none 0-3: LOSO {a['none_loso_mean']:6.2f}  vs best-of-3 in-sample {a['none_best_mean']:6.2f}  vs @{PICK} "
          f"{a['none_pick_mean']:6.2f};  slope 0-3 with LOSO {a['slope_h1_loso']:+.3f} (difference vs @{PICK} "
          f"{a['identity']['loso']['slope_diff']:+.3f} = w53 x ({a['none_pick_mean']:.3f} - {a['none_loso_mean']:.3f})  OK)")
    print(f"  (e) 53M none area by lr, seeds 0-3 (Torch A100-80GB vs Burst A100-40GB, F86 pt2b; Torch - Burst by seed):")
    print(f"      {'lr':7s} {'site':6s} {'s0':>6s} {'s1':>6s} {'s2':>6s} {'s3':>6s}   {'mean':>6s}")
    out["table"] = {}
    for lr in LRS:
        t = T[lr]["area"]
        bu = [v if st == "ok" else None for v, st in zip(Bu[lr]["area"], Bu[lr]["status"])]
        d = [x - y if y is not None else None for x, y in zip(t, bu)]
        bm = float(np.mean(bu)) if None not in bu else None
        out["table"][lr] = {"torch": t, "burst": bu, "torch_minus_burst": d, "torch_mean": float(np.mean(t)), "burst_mean": bm}
        print(f"      {lr:7s} {'Torch':6s} {fmt4(t)}   {np.mean(t):6.2f}")
        print(f"      {'':7s} {'Burst':6s} {fmt4(bu)}   " + (f"{bm:6.2f}" if bm is not None else "   n/a"))
        print(f"      {'':7s} {'T - B':6s} {fmt4(d, '+')}   " + (f"{np.mean(t) - bm:+6.2f}" if bm is not None else "   n/a"))
        for s, (st, w) in enumerate(zip(Bu[lr]["status"], Bu[lr]["why"])):
            if st != "ok":
                print(f"      Burst {lr} seed {s}: {st} ({w})")
    print(f"      Burst pick (F86, pt2b none grid 1e-3..1.6e-2): {PICK}")
    return out


def canon(x):
    return json.loads(json.dumps(x, default=float))


def check_js(js, primary_json):
    """the A1 output never goes into readout.json (nor into the primary JSON given)"""
    if js is not None:
        bad = {HERE / "readout.json"} | ({Path(primary_json).resolve()} if primary_json else set())
        assert Path(js).resolve() not in bad, "the A1 output never goes into readout.json"


def run(roots, investigated=None, primary_json=None, js=None):
    """the primary read-out (unchanged path), its check against readout.json, then the A1 section. -> (rc, primary, a1)"""
    frozen = copy.deepcopy((RO.PICKS, RO.ROOTS, RO.F86, RO.BURST_TASK, RO.N_NONEMB, RO.SIZES, RO.SEEDS, RO.H1, RO.H2))
    check_js(js, primary_json)
    D = RO.load(roots)
    res = RO.read(roots, investigated=investigated, js=None, data=D)
    if primary_json and Path(primary_json).exists():
        saved = json.loads(Path(primary_json).read_text())
        if canon(res) != saved:
            diff = sorted(k for k in set(saved) | set(canon(res)) if saved.get(k) != canon(res).get(k))
            print(f"\nSTOP: the primary read-out recomputed here differs from {primary_json} in {diff}; A1 not computed")
            return 3, res, None
        print(f"\n(primary read-out recomputed = {primary_json}: identical)")
    elif primary_json:
        print(f"\n({primary_json} not found: the primary read-out above was computed here, not compared)")
    A = {(z, a): np.asarray(D[("torch", z, a)]["area"], float) if all(st == "ok" for st in D[("torch", z, a)]["status"])
         else None for z in RO.SIZES for a in RO.ARMS}
    G = None
    if "sizes" in res:
        G = {z: A[(z, "win-ctx")] - A[(z, "none")] for z in RO.SIZES}
        for z in RO.SIZES:
            assert G[z].tolist() == res["sizes"][z]["delta_per_seed"], z
        assert abs(slope_h1(G) - res["slope"]["h1"]) < 1e-9, (slope_h1(G), res["slope"]["h1"])
    T = load_a1(roots, "torch")
    for s in S4:   # double path: the pick's runs read by both loaders
        if T[PICK]["status"][s] == "ok":
            assert T[PICK]["area"][s] == D[("torch", "53m", "none")]["area"][s], s
    Bu = {lr: {"area": [], "status": [], "why": []} for lr in LRS}   # Burst (descriptive): RO.load_run's checks only
    for lr in LRS:
        for s in S4:
            run_, chk, _ = a1_paths("burst", lr, s, roots)
            st, a, w = RO.load_run(run_, chk)
            Bu[lr]["status"].append(st)
            Bu[lr]["area"].append(a)
            Bu[lr]["why"].append(w if st != "ok" else "")
    out = section(res, T, Bu, A, G)
    assert copy.deepcopy((RO.PICKS, RO.ROOTS, RO.F86, RO.BURST_TASK, RO.N_NONEMB, RO.SIZES, RO.SEEDS, RO.H1, RO.H2)) == frozen, \
        "readout_pt2same globals changed"
    if js:
        Path(js).write_text(json.dumps({"primary_verdict": res.get("verdict"), "a1": out}, indent=1, default=float))
    return 0, res, out


# ---------------------------------------------------------------- self-test (synthetic trees only)
def _write(run, chk, tag, s, area, off, intr, fail=False, offline=True):
    run.parent.mkdir(parents=True, exist_ok=True)
    e = lambda st: {"step": st, "lm": 11.6, "val_loss": 3.0, "val_bits_per_byte": intr, **{f"held_{b}": area / 100 for b in RR.B}}
    run.write_text(json.dumps({"steps": 18310, "args": {"bs": 32, "seq": 512}, "curve": [e(st) for st in (0, 250, 18309)]}))
    chk.write_text(json.dumps({"pass": not fail, "checks": [{"check": "fake", "ok": not fail, "value": 0}],
                               "info": {"area": round(area, 2), "bpb_full": off}}))
    o = {str(run.resolve()): {"tag": f"_{tag}", "seed": s, "max_windows": 0, "bpb_full": off, "orig_minus_logged": 0.0}}
    of = chk.parent / "offline_lm.json"
    if offline:
        of.write_text(json.dumps(o))
    elif of.exists():
        of.unlink()


def _tree(root, none4, off=None, intr=None, burst4=None, fail=(), missing=(), no_offline=(), primary_missing=()):
    """a full fake tree: the primary's 48 Torch + 32 Burst runs (RO._fake_tree, RO.selftest's values) with 53M none seeds
    0-3 at 8e-3 = none4['8e-3'], plus the 8 A1 Torch runs and the 8 Burst runs at 4e-3 / 1.6e-2 (seeds 0-3)"""
    base = {}
    for z in RO.SIZES:
        for a in RO.ARMS:
            base[("torch", z, a)] = [RO.F86.get((z, a), 25.0 if a == "none" else 77.0) + 0.5 * k for k in range(8)]
            if z in RO.NEW:
                base[("burst", z, a)] = [RO.F86[(z, a)] - 0.25 * k for k in range(8)]
    base[("torch", "53m", "none")] = list(none4[PICK]) + [18.0, 19.0, 20.0, 21.0]
    roots = RO._fake_tree(root, base, missing=set(primary_missing))
    off = off or {lr: [1.0 + 0.01 * i] * 4 for i, lr in enumerate(LRS)}
    intr = intr or off
    burst4 = burst4 or {"4e-3": [26.12, 8.48, 9.92, 14.47], "1.6e-2": [18.33, 12.57, 20.02, 9.89]}
    for lr in LRS:
        for s in S4:
            run_, chk, tag = a1_paths("torch", lr, s, roots)
            if (lr, s) in missing:
                continue
            _write(run_, chk, tag, s, none4[lr][s], off[lr][s], intr[lr][s], fail=(lr, s) in fail, offline=(lr, s) not in no_offline)
            if lr != PICK:
                run_, chk, tag = a1_paths("burst", lr, s, roots)
                _write(run_, chk, tag, s, burst4[lr][s], 1.2, 1.2)
    return roots


def selftest():
    n_ok, n = 0, 0

    def go(name, roots, check, show=False, **kw):
        nonlocal n_ok, n
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc, res, a1 = run(roots, **kw)
        txt = buf.getvalue()
        sec = txt[txt.find("== Amendment A1"):] if "== Amendment A1" in txt else txt[txt.rfind("STOP"):]
        try:
            good, note = check(rc, res, a1, txt)
        except Exception as ex:   # noqa: BLE001
            good, note = False, f"check raised {ex!r}"
        n += 1
        n_ok += bool(good)
        print(f"{'PASS' if good else 'FAIL'} {name}: {note}")
        print("    " + (txt if show else sec).rstrip().replace("\n", "\n    "))
        return rc, res, a1

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        # 1. Torch best = 8e-3 (untied): the A1 slope equals the primary seeds 0-3 slope; every LOSO fold picks 8e-3
        nA = {"4e-3": [10, 12, 11, 9], "8e-3": [24, 22, 20, 26], "1.6e-2": [15, 14, 16, 15]}
        rA = _tree(t / "A", nA)
        pj = t / "A_readout.json"
        with contextlib.redirect_stdout(io.StringIO()):
            RO.read(rA, js=str(pj))
        pj_bytes = pj.read_bytes()
        go("Torch best = 8e-3", rA, lambda rc, res, a, txt: (
            rc == 0 and a["status"] == "computed" and a["best"] == "8e-3" and a["near"] == ["8e-3"]
            and abs(a["slope_h1_best"] - a["slope_h1_pick"]) < 1e-12 and abs(a["slope_h1_pick"] - res["slope"]["h1"]) < 1e-12
            and [f["lr"] for f in a["folds"]] == ["8e-3"] * 4 and abs(a["none_loso_mean"] - 23.0) < 1e-9
            and a["bracket"] is None and "bracket closed" in txt
            and abs(a["table"]["4e-3"]["torch_minus_burst"][0] - (10 - 26.12)) < 1e-9
            and abs(a["table"]["8e-3"]["burst_mean"] - np.mean([RO.F86[("53m", "none")] - 0.25 * k for k in range(4)])) < 1e-9
            and pj.read_bytes() == pj_bytes,
            f"best {a['best']}, slope 0-3 pick {a['slope_h1_pick']:+.6f} = best {a['slope_h1_best']:+.6f} = primary h1 "
            f"{res['slope']['h1']:+.6f}; LOSO folds {[f['lr'] for f in a['folds']]}"), show=True, primary_json=str(pj))
        # 2. best at an edge with an open bracket: 1.6e-2 beats 8e-3 by 10 -> 'open bracket at the upper edge, not extended'
        nB = {"4e-3": [12] * 4, "8e-3": [19, 21, 20, 20], "1.6e-2": [30, 31, 29, 30]}
        go("best at an edge, open bracket", _tree(t / "B", nB), lambda rc, res, a, txt: (
            a["best"] == "1.6e-2" and a["bracket"] == ["open", "upper"] and "open bracket at the upper edge, not extended" in txt,
            f"best {a['best']}, bracket {a['bracket']}"))
        # 2b. lower edge open as well
        nB2 = {"4e-3": [30] * 4, "8e-3": [20] * 4, "1.6e-2": [12] * 4}
        go("best at the lower edge, open bracket", _tree(t / "B2", nB2), lambda rc, res, a, txt: (
            a["best"] == "4e-3" and a["bracket"] == ["open", "lower"] and "open bracket at the lower edge, not extended" in txt,
            f"best {a['best']}, bracket {a['bracket']}"))
        # 3. tie within 2 broken by the lower OFFLINE bpb (the in-training bpb would pick the other lr)
        nC = {"4e-3": [21] * 4, "8e-3": [22] * 4, "1.6e-2": [10] * 4}
        offC = {"4e-3": [1.00] * 4, "8e-3": [1.05] * 4, "1.6e-2": [1.20] * 4}
        intrC = {"4e-3": [1.10] * 4, "8e-3": [1.00] * 4, "1.6e-2": [1.20] * 4}
        go("tie within 2 broken by offline bpb", _tree(t / "C", nC, off=offC, intr=intrC), lambda rc, res, a, txt: (
            a["best"] == "4e-3" and a["near"] == ["4e-3", "8e-3"] and a["offline"] and a["bracket"] is None
            and "tie broken on the offline bpb" in txt
            and abs(a["identity"]["best"]["slope_diff"] - W53 * (22 - 21)) < 1e-9,
            f"best {a['best']} (area 21 < 22, bpb 1.00 < 1.05), near {a['near']}, offline {a['offline']}"))
        # 4. one failed A1 run -> every A1 line 'not computed'; the primary verdict is that of the clean tree
        rD = _tree(t / "D", nA, fail={("4e-3", 2)})
        go("one failed A1 run -> not computed", rD, lambda rc, res, a, txt: (
            a["status"] == NC and txt.count(f": {NC}") == 4 and "WITHHELD torch 53m none lr 4e-3 seed 2" in txt
            and "Torch best lr:" not in txt and res["verdict"] == json.loads(pj_bytes)["verdict"]
            and canon(res) == json.loads(pj_bytes),
            f"status {a['status']}, primary verdict unchanged: {res['verdict']}"), primary_json=str(pj))
        go("one missing A1 run -> not computed", _tree(t / "D2", nA, missing={("1.6e-2", 1)}), lambda rc, res, a, txt: (
            a["status"] == NC and txt.count(f": {NC}") == 4 and "incomplete torch 53m none lr 1.6e-2 seed 1" in txt,
            f"status {a['status']}"))
        go("A1 run without a full-val offline entry -> not computed", _tree(t / "D3", nA, no_offline={("8e-3", 3)}),
           lambda rc, res, a, txt: (a["status"] == NC and "no valid full-val offline_lm entry" in txt, f"status {a['status']}"))
        rD4 = _tree(t / "D4", nA)
        run_, chk, _ = a1_paths("torch", "1.6e-2", 0, rD4)
        c = json.loads(chk.read_text()); c["info"]["area"] += 1.0; chk.write_text(json.dumps(c))
        go("A1 area double-path mismatch -> not computed", rD4, lambda rc, res, a, txt: (
            a["status"] == NC and "double path: area" in txt, f"status {a['status']}"))
        go("primary without a verdict (a missing 505M run) -> not computed",
           _tree(t / "D5", nA, primary_missing={("torch", "505m", "win-ctx", 6)}), lambda rc, res, a, txt: (
            res["verdict"].startswith("no verdict") and a["status"] == NC and "primary read-out has no verdict" in txt,
            f"primary {res['verdict']}; A1 {a['status']}"))
        # 5. identity: 4e-3 higher by ~6 -> slope difference = w53 x (none@8e-3 - none@4e-3), checked against a
        #    brute-force OLS (numpy polyfit on the three seeds 0-3 means built by hand)
        nE = {"4e-3": [29, 27, 30, 26], "8e-3": [22, 21, 23, 20], "1.6e-2": [12, 13, 11, 14]}
        rE = _tree(t / "E", nE)

        def chkE(rc, res, a, txt):
            pr = {z: np.mean([res["sizes"][z]["delta_per_seed"][s] for s in S4]) for z in RO.SIZES}
            prior53 = np.mean([RO.F86[("53m", "win-ctx")] + 0.5 * k for k in range(4)])
            y = [prior53 - np.mean(nE["4e-3"]), pr["190m"], pr["505m"]]
            bf = np.polyfit(X, y, 1)[0]
            want = W53 * (np.mean(nE["8e-3"]) - np.mean(nE["4e-3"]))
            return (a["best"] == "4e-3" and abs(a["slope_h1_best"] - bf) < 1e-9
                    and abs(a["identity"]["best"]["slope_diff"] - want) < 1e-9 and abs(W53 + 0.316653) < 1e-6,
                    f"slope diff {a['identity']['best']['slope_diff']:+.6f} = w53 {W53:+.6f} x ({np.mean(nE['8e-3']):.2f} - "
                    f"{np.mean(nE['4e-3']):.2f}) = {want:+.6f}; brute-force slope {bf:+.6f} = {a['slope_h1_best']:+.6f}")
        go("identity check", rE, chkE)
        # 6. LOSO known answer: in-sample best 4e-3 (mean 18.5, carried by seed 0); folds pick 8e-3 / 4e-3 / 4e-3 / 4e-3,
        #    held-out areas 14, 10, 10, 10 -> LOSO mean 11.0 (selection bias of the in-sample pick 7.5)
        nF = {"4e-3": [44, 10, 10, 10], "8e-3": [14, 16, 16, 16], "1.6e-2": [5] * 4}
        go("LOSO known answer", _tree(t / "F", nF), lambda rc, res, a, txt: (
            a["best"] == "4e-3" and abs(a["none_best_mean"] - 18.5) < 1e-9
            and [f["lr"] for f in a["folds"]] == ["8e-3", "4e-3", "4e-3", "4e-3"]
            and np.allclose([f["area"] for f in a["folds"]], [14, 10, 10, 10]) and abs(a["none_loso_mean"] - 11.0) < 1e-9
            and abs(a["identity"]["loso"]["slope_diff"] - W53 * (15.5 - 11.0)) < 1e-9,
            f"folds {[(f['lr'], f['area']) for f in a['folds']]}, LOSO {a['none_loso_mean']:.2f} vs in-sample "
            f"{a['none_best_mean']:.2f}"))
        # 7. readout.json mismatch -> STOP before the A1 section (rc 3); the A1 JSON never goes into readout.json
        bad = json.loads(pj_bytes); bad["verdict"] = "tampered"
        pj2 = t / "A_tampered.json"; pj2.write_text(json.dumps(bad))
        go("primary differs from readout.json -> STOP", rA, lambda rc, res, a, txt: (
            rc == 3 and a is None and "== Amendment A1" not in txt and "differs from" in txt, f"rc {rc}"), primary_json=str(pj2))
        n += 1
        ok = True
        for js_, pj_ in ((str(HERE / "readout.json"), None), (str(pj), str(pj))):   # the guard alone: nothing is run
            try:
                check_js(js_, pj_)
                ok = False
            except AssertionError as ex:
                ok &= "never goes into readout.json" in str(ex)
        check_js(str(t / "a1.json"), str(pj))
        n_ok += ok
        print(f"{'PASS' if ok else 'FAIL'} --json readout.json (or the primary JSON) is refused; another path is accepted")
        n += 1
        ok = RO.PICKS == {"53m": {"none": "8e-3", "win-ctx": "2e-3"}, "190m": {"none": "4.8e-3", "win-ctx": "6e-4"},
                          "505m": {"none": "4.8e-3", "win-ctx": "6e-4"}} and RO.ROOTS["torch"] == B / "runs" / "pt2same" / "out"
        n_ok += ok
        print(f"{'PASS' if ok else 'FAIL'} readout_pt2same globals unchanged after every case")
    print(f"SELFTEST {'PASS' if n_ok == n else 'FAIL'} ({n_ok}/{n})")
    return n_ok == n


def known():
    """published Burst pt2b runs only (no pt2same result): the Burst paths and the offline-bpb extraction on real files;
    known answers = tests/audit_a1.md Q2 / AUDIT A1 item 2 (per-seed areas, means 14.75 / 23.10 / 15.20); rule 1 on the
    three lrs picks 8e-3 untied (F86's pick); every Burst task name is the only results dir of its lr and seed"""
    roots = {"burst": RO.ROOTS["burst"]}
    Bu = load_a1(roots, "burst")
    want = {"4e-3": [26.12, 8.48, 9.92, 14.47], "1.6e-2": [18.33, 12.57, 20.02, 9.89]}
    ok = True
    for lr in LRS:
        st = Bu[lr]["status"]
        dirs = [sorted(p.name for p in roots["burst"].glob(f"[0-9]*-pt2b-lr{lr}-none-s{s}")) for s in S4]
        uniq = all(d == [BURST_NAME[(lr, s)]] for s, d in zip(S4, dirs))
        good = all(x == "ok" for x in st) and uniq and all(v is not None and math.isfinite(v) for v in Bu[lr]["off"])
        if lr in want:
            good &= all(abs(a - w) < 0.005 for a, w in zip(Bu[lr]["area"], want[lr]))
        m = float(np.mean(Bu[lr]["area"])) if all(x == "ok" for x in st) else float("nan")
        good &= abs(m - {"4e-3": 14.75, "8e-3": 23.10, "1.6e-2": 15.20}[lr]) < 0.01
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} Burst 53M none {lr}: {fmt4(Bu[lr]['area'])} mean {m:.2f}; offline bpb "
              f"{' '.join(f'{v:.4f}' for v in Bu[lr]['off'] if v is not None)}; dirs unique {uniq}")
    sel = RR.best_lr({lr: {k: Bu[lr][k] for k in ("area", "off", "intr")} for lr in LRS})
    good = sel["best"] == "8e-3" and sel["near"] == ["8e-3"] and RR.bracket(list(LRS), sel["mean"], "8e-3") is None
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} rule 1 on Burst seeds 0-3 over the three lrs: {sel['best']} (near {sel['near']})")
    print(f"KNOWN {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--known" in sys.argv:
        sys.exit(0 if known() else 1)
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    rc, _, _ = run(RO.ROOTS, investigated=arg("--investigated"), primary_json=arg("--primary-json", str(HERE / "readout.json")),
                   js=arg("--json"))
    sys.exit(rc)
