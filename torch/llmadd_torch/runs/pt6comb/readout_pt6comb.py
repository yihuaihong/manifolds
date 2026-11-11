"""pt6comb read-out (PRE-REGISTERED in runs/pt6comb/AUDIT.md before any pt6comb GPU result): six manifold priors trained
together at 53M (composition B: numbers, s4, dim, cities, countries, people), the user's goal 'each helps, none interferes'.

Metrics (one definition per manifold m; area = 100 x the mean over the 75 evals of the per-eval read-out; final3 = the last 3):
  numbers    the 6-bucket held-out exact match (mean of held_b11..b33 per eval) = metrics['numbers'] (F86 / F88; double path)
  s4         s4_hard: the held_fresh_nonid cells NOT solvable by the Latin-square propagation from the seed's training table
             (rb6 A8.2 / A8.3, F88 A8.3's confirmatory metric), from the logged per-question hits (rainbow6_summary.s4_hard_series)
  dim        the strict dim set (rb6 A5 STRICT: no_negtwin_nonid AND no same-answer token-transform twin trained; F88's dim
             read-out), from the per-question hits (rainbow6_summary.strict_series)
  cities / countries / people   the 2-way choice over every held-out question, both orders = metrics['<d>_readout'] (F100 / F101;
             Tier L: the target carries the answers); per-question sub-read-outs re-derived (readout_ptdom.run_subsets)
  mean6      the unweighted mean of the six areas (the lr-selection objective only)
Selection (seeds 0-3; select_pt6comb.py = decide()): all6 over 1e-3 / 2e-3 / 4e-3, none over 2e-3 / 4e-3 / 8e-3, each by F87 rule 1
  (readout_rules.best_lr 'top2' on mean6: the argmax of the 4-seed mean; the top two within 2 points -> the lower mean in-training
  val bpb) after the A2.1 auxiliary-loss gate per all6 cell (median over runs of the run's on-window median a / (a + b) < 0.5, or
  an unverifiable run -> the cell cannot be picked; every cell failing -> GATE). Extensions once (ptdom2 A1): all6 x2 above when
  its pick OR area argmax is its top lr; none x2 above / x1/2 below likewise; fixB: none at all6's pick when it lies outside none's
  range; a gate-failed next lr closes that edge. After the extensions: NOT FINAL when a pick or argmax is still at an outermost lr
  it ran (all6: the top; none: either edge) or all6's pick lies outside none's range -> no confirmation (the coordinator decides).
Verdicts (fresh seeds 8-15; none@N, all6@P, none@2e-3, all6@2e-3 (A2: merged with N / P when equal), only-<m>@2e-3; per manifold m;
  g = all6@P - none@N; A2 (the SAME-lr interference contrast): h = all6@2e-3 - only-m@2e-3, u = only-m@2e-3 - none@2e-3; per seed, on
  m's area):
  HELPS            iff the one-sided paired t p (H1: mean g > 0; df 7) < 0.05 AND g > 0 in >= 6 of 8 seeds; else NOT SHOWN
  interference not detected (the word 'NO INTERFERENCE' of A0; A2: always written 'interference not detected', never 'shown absent')
                   iff (mean u > 0 and mean h >= -0.5 x mean u) OR the one-sided paired t p (H1: mean h < 0) >= 0.05; mean u <= 0:
                   the margin clause does not apply (the single prior shows no gain itself), the t clause alone; else INTERFERENCE
  descriptive      the same-lr gain all6@2e-3 - none@2e-3 and all6@P - only-m@2e-3 (lr-confounded when P != 2e-3), mean + 90%
  headline         the number of manifolds with HELPS and interference not detected, each named; uncorrected for six tests; the
                   Holm-adjusted HELPS p (family of six) printed as the reference; the six gains' means with intervals and their
                   per-seed mean over the six
  all6 GATE        the A2.1 cell gate at all6@P on seeds 8-11 and on 12-15 separately (ptdom A4.5 / ptdom2 A1.7): either failing ->
                   no verdict word for any manifold ('all6 GATE'), the numbers printed. A3: the same gate at all6@2e-3 (the arm of h);
                   failing -> every manifold's interference word reads 'AUX GATE FAILED at all6@2e-3' (HELPS unaffected)
  A3 (missing)     a missing / invalid run of the same-lr pair none@2e-3 / all6@2e-3 (when not merged with none@N / all6@P) withholds only
                   the interference words and the same-lr rows; the strict HELPS read-out needs none@N, all6@P and the only-m cells
Descriptive (never a condition): final3 of g / h / u; the window split of each area gain (evals at steps 916..7323 vs the others);
  FLOOR (both arms' mean area <= the mean over seeds of max(chance and the non-geometric baselines) + 5; ptdom's definition per
  read-out) / CEILING (none >= 95); per arm the LM share a / (a + b), the clip coefficient, b > a and clip < 1 shares; the LM cost
  (final in-training val loss and val bpb, all6 / only-m - none; lr-confounded when the lrs differ; the same-lr rows on seeds 0-3);
  secondary read-outs (s4 held_fresh_nonid, dim no_negtwin_nonid, the comparisons' hard-subset final3); the eval-prompt CKA to each
  target at the window's last eval per arm.
Validity (inherited): VALID run = finished (75 evals), its check.json passes, or its only failed checks are the CKA-rise sanity bar /
  the GPU-numerics double-path tolerances (ptdom A5.1: flagged data) / the geo check at a geo overshoot of a perfect fit verified
  from the run JSON for EVERY active prior (ptdom A6 / A6.1 generalised: geo_a6_multi). The run's identity (seed, lr, tag, target,
  prior_domains, domains) must equal its directory name. A finished run is never re-run; the loader reads a cell's latest dir and
  integrity() refuses a cell with a finished run in an earlier dir. Refusal: a needed run missing or invalid, data differing between
  the arms of a seed, a metric double path disagreeing, a picks file differing from its re-derivation; --allow_partial prints
  PROVISIONAL with the reasons.

  python readout_pt6comb.py [--root runs/pt6comb/out] [--allow_partial] [--json out.json]      (the verdicts, seeds 8-15)
  python readout_pt6comb.py --selection [--root ...]                                            (the selection table, seeds 0-3)
  python readout_pt6comb.py --selftest
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True   # never write a .pyc next to the frozen ptdom / ptdom2 modules imported below

import argparse          # noqa: E402
import copy              # noqa: E402
import hashlib           # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import re                # noqa: E402
import statistics        # noqa: E402
import tempfile          # noqa: E402
from functools import lru_cache   # noqa: E402
from pathlib import Path           # noqa: E402

import numpy as np       # noqa: E402

HERE = Path(__file__).resolve().parent
B = Path("/scratch/yh6210/llmadd_torch")   # absolute (a test copy of this module lives elsewhere: tests/mutation_readout.py)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
RO_FILE, RO_SHA = B / "runs" / "ptdom" / "readout_ptdom.py", "1e2f17522e47"
assert sha(RO_FILE) == RO_SHA, f"readout_ptdom.py changed: {sha(RO_FILE)} != {RO_SHA}"
sys.path.insert(0, str(RO_FILE.parent))
import readout_ptdom as RO   # noqa: E402  (read only: window_grad, cell_gate, run_subsets, domain_meta, boot_ci, t_test, sf_test, holm)
TOOLS = HERE / "code" / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(B))
import rainbow6_summary as RS   # noqa: E402  (rb6's s4 / dim read-out definitions, the snapshot's copy)
import make_jobs_pt6comb as M6  # noqa: E402
RR = RO.RR
assert Path(RS.__file__).resolve().parent == TOOLS.resolve() and sha(TOOLS / "rainbow6_summary.py") == sha(B / "runs/ptdom/code/tools/rainbow6_summary.py")
assert Path(M6.__file__).resolve() == (B / "make_jobs_pt6comb.py").resolve()

COMP = "B"
MANI = M6.manifolds(COMP)                        # numbers, s4, dim, cities, countries, people
CMPD = ("cities", "countries", "people")        # the 2-way comparison domains
BKT = ("b11", "b12", "b13", "b22", "b23", "b33")
SEL, CONF, SINGLE_LR, GRID_L1 = M6.SEL, M6.CONF, M6.SINGLE_LR, M6.GRID_L1
N_EVALS, ALPHA, MIN_WINS, INT_FRAC = 75, 0.05, 6, 0.5
WIN = (916, 7323)                               # the prior window's steps (0.05, 0.4 of 18,310): 26 of the 75 evals inside
AUX_MIN = RO.AUX_MIN
CKA_CHECK, DP_FLAG_CHECKS = RO.CKA_CHECK, RO.DP_FLAG_CHECKS
GEO_TOL, GEO_FLOOR, GEO_CKA_MIN, GEO_CKA_MAX = RO.GEO_TOL, RO.GEO_FLOOR, RO.GEO_CKA_MIN, RO.GEO_CKA_MAX
GEO_FLAG6 = M6.GEO_NAME6.format(TOL=GEO_TOL)    # 'geo > -0.001 for every active prior (numbers: geo) and 0 < grad_prior < inf at every on-eval'
KEEP6 = ("the active prior applied on >= 95% of the on-steps; every inactive one never",
         "geo_<d> logged at >= 95% of the evals for the active prior, never for an inactive one",
         "weighted prior gradients at the on-evals from the active prior only")
NUM_OFF = "numbers prior off: never applied, no numbers loss term"
NAME = re.compile(r"^pt6comb-(B)-(none|all6|only-[a-z0-9_]+)-lr([0-9.e-]+)-s(\d+)(?:-r(\d+))?$")
GATE_TXT = "all6 GATE (the LM share a / (a + b) of all6@P < 0.5 on seeds 8-11 or 12-15: the prior is not an auxiliary loss there)"
med = lambda v: float(np.median(v)) if len(v) else None
mean = lambda v: float(np.mean(v)) if len(v) else None
assert GEO_FLAG6 == "geo > -0.001 for every active prior (numbers: geo) and 0 < grad_prior < inf at every on-eval"
assert GRID_L1 == {"none": ("2e-3", "4e-3", "8e-3"), "all6": ("1e-3", "2e-3", "4e-3")} and SINGLE_LR == "2e-3"


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ------------------------------------------------------------------------------------------------------------ metrics
def series_of(r, m):
    """Per-eval accuracy (fraction) of manifold m's primary read-out (module docstring)."""
    c = r["curve"]
    if m == "numbers":
        return [float(np.mean([e[f"held_{b}"] for b in BKT])) for e in c]
    if m == "s4":
        return [v for _, v in RS.s4_hard_series(r)]
    if m == "dim":
        return [v for _, v in RS.strict_series(r)]
    return [e[f"{m}_held_2way"] for e in c]


def hits_area(r, d, mask_name):
    """100 x the mean over evals of the accuracy on eval_held restricted to a logged mask, from the per-question hits."""
    D = r["domains"][d]["eval_held"]
    mk = [ch == "1" for ch in D["masks"][mask_name]]
    n, nc = sum(mk), D["contexts"]
    return 100 * float(np.mean([sum(int(e[f"hits_{d}_held"][i]) for i, k in enumerate(mk) if k) / (nc * n) for e in r["curve"]]))


def floor_line(r, m, sub=None):
    """The FLOOR line of manifold m on this run's seed (ptdom's definition per read-out): max(chance, the non-geometric baselines)
    + 5. numbers: chance 0 (exact match of a sum); s4_hard: max(uniform 1/24, the pure Latin table-filler's expected accuracy on
    the hard cells); dim strict: max(chance = max(uniform, majority training answer), the answer-elimination heuristic);
    comparisons: max(50, the answer-frequency and shorter-option lines, the training-frequency shortcut baseline on every held-out
    question (readout_ptdom.run_subsets 'h3_held'))."""
    if m == "numbers":
        return 100 * float(r["domains"]["numbers"].get("chance", 0.0)) + 5
    if m == "s4":
        return max(100 / 24, RS.s4_hard(r)["chance"]) + 5
    if m == "dim":
        return max(100 * RS.set_chance(r, "dim", RS.STRICT), RS.heuristic(r, "dim", RS.STRICT)) + 5
    D = r["domains"][m]
    lines = [50.0, 100 * D.get("heuristic_ansfreq", 0), 100 * D.get("heuristic_shorter", 0)]
    if sub is not None:
        lines.append(sub["h3_held"])
    return max(lines) + 5


def window_series_split(sa, sb, steps):
    """The area gain 100 x mean(sa - sb) split into the evals inside the prior window (steps 916..7323) and the others (in + out =
    the gain; each part = 100 x sum over its evals / the number of evals)."""
    n = len(steps)
    ins = [i for i, s in enumerate(steps) if WIN[0] <= s <= WIN[1]]
    d = [a - b for a, b in zip(sa, sb)]
    tin = 100 * sum(d[i] for i in ins) / n
    return {"in": tin, "out": 100 * sum(d) / n - tin, "n_in": len(ins), "n": n}


@lru_cache(maxsize=None)
def meta_cmp():
    return RO.domain_meta(list(CMPD))


def run_metrics(r):
    """-> (metrics dict, problems): every manifold's series / area / final3, the double paths, the FLOOR lines, the secondary
    read-outs, the eval CKA at the window's last on-eval."""
    c, m_, probs = r["curve"], r["metrics"], []
    out = {"series": {}, "area": {}, "final3": {}, "floor": {}, "sec": {}, "cka_win": {}, "sub": {}}
    for m in MANI:
        s = series_of(r, m)
        if len(s) != len(c):
            probs.append(f"{m}: {len(s)} series values for {len(c)} evals")
        out["series"][m] = s
        out["area"][m] = 100 * float(np.mean(s)) if s else None
        out["final3"][m] = 100 * float(np.mean(s[-3:])) if s else None
    # double paths: the metric the trainer wrote vs the series re-derived here
    if abs(out["area"]["numbers"] - m_["numbers"]["area"]) > 1e-6:
        probs.append(f"numbers area {out['area']['numbers']} != metrics {m_['numbers']['area']}")
    for d in CMPD:
        if abs(out["area"][d] - m_[f"{d}_readout"]["area"]) > 1e-6:
            probs.append(f"{d} area {out['area'][d]} != metrics {m_[f'{d}_readout']['area']}")
    for d, mk in (("s4", "fresh_nonid"), ("dim", "no_negtwin_nonid")):
        a_ = hits_area(r, d, mk)
        if abs(a_ - m_[f"{d}_readout"]["area"]) > 1e-6:
            probs.append(f"{d}_readout area re-derived from the hits {a_} != metrics {m_[f'{d}_readout']['area']}")
        out["sec"][f"{d}_readout"] = {"area": m_[f"{d}_readout"]["area"], "final3": m_[f"{d}_readout"]["final3"]}
    h = RS.s4_hard(r)
    out["sec"]["s4_hard_n"], out["sec"]["s4_latin_level"] = h["n"], h["level"]
    pre_ = RS.S4_HARD_PRE.get(r["args"]["seed"])   # A1 (audit m7): rb6 A8.3's precomputed s4_hard sets (seeds 8-15; the splits are identical)
    if pre_ is not None and ((h["n"], h["n_readout"]) != pre_[:2] or abs(h["chance"] - pre_[2]) > 1e-9):
        probs.append(f"s4_hard ({h['n']}, {h['n_readout']}, {h['chance']}) != rb6's precomputation {pre_} at seed {r['args']['seed']}")
    out["sec"]["dim_strict_n"] = int(RS.strict_mask(r)[0].sum())
    for d in CMPD:
        out["sec"][f"{d}_hard"] = {"area": m_.get(f"{d}_readout_hard", {}).get("area"), "final3": m_.get(f"{d}_readout_hard", {}).get("final3")}
        D_ = r["domains"][d]
        eh = D_.get("eval_held")
        x = {"q": eh["q"], "nctx": eh["contexts"], "masks": eh.get("masks", {}), "split_seed": D_.get("split_seed"),
             "table_sha": D_.get("table", {}).get("sha"), "heur": [D_.get("heuristic_ansfreq"), D_.get("heuristic_shorter")],
             "hits": [e[f"hits2_{d}_held"] for e in c], "logged": [e.get(f"{d}_held_2way") for e in c],
             "logged_hard": [e.get(f"{d}_held_2way_hard") for e in c], "area": m_[f"{d}_readout"]["area"],
             "area_hard": m_.get(f"{d}_readout_hard", {}).get("area")}
        sub, p_ = RO.run_subsets(x, meta_cmp()[d])
        probs += [f"{d} sub-read-out: {v}" for v in p_]
        out["sub"][d] = sub
    for m in MANI:
        out["floor"][m] = floor_line(r, m, out["sub"].get(m))
    on = [e for e in c if e.get("prior_on")]
    we = on[-1] if on else next((e for e in c if e["step"] >= 7250), c[-1])   # the window's last eval (step 7250)
    for m in MANI:
        out["cka_win"][m] = we.get("cka_ops_helix3") if m == "numbers" else we.get(f"cka_{m}")
    out["steps"] = [e["step"] for e in c]
    return out, probs


# ------------------------------------------------------------------------------------------------------------ validity (A6 generalised)
def geo_a6_multi(r, ck, arm):
    """ptdom A6 / A6.1 for EVERY active prior of the arm (numbers: its value 'geo' and its CKA cka_ops_helix3): a failed geo check
    is valid data (flagged) iff (1) grad_prior finite > 0 at every eval with prior_on; (2) at every such eval where an active
    prior's value is <= -0.001: that value >= -0.005 and its CKA (the eval-prompt CKA, unrounded) in [0.995, 1.01]; (3) no inactive
    domain has a geo / prior-gradient field (numbers inactive: geo 0 and no gnorm_prior_numbers), and the separate checks are
    present once and passed (KEEP6 + the numbers-prior check of the arm). The failure must be explained (some low value or a bad
    grad_prior). -> {'why': [...], 'flag': ...}"""
    c, why = r["curve"], []
    doms = r.get("domains", {})
    pri = M6.pri_of(arm, COMP)
    act = tuple(x for x in MANI if doms.get(x, {}).get("prior"))
    jpri = [x for x in str(r.get("args", {}).get("prior_domains") or "").split(",") if x]
    if not pri or act != pri or tuple(jpri) != pri:
        return {"why": [f"the active priors {act} / args {jpri} != the arm's {pri}"], "flag": None}
    on = [e for e in c if e.get("prior_on")]
    if not on:
        why.append("no eval with prior_on")
    gp_bad = [e["step"] for e in on if not (_num(e.get("grad_prior")) and e["grad_prior"] > 0)]
    if gp_bad:
        why.append(f"(1) grad_prior not finite > 0 at {len(gp_bad)} of {len(on)} evals with prior_on (steps {gp_bad[:5]})")
    low = []
    for d in pri:
        gk, kk = ("geo", "cka_ops_helix3") if d == "numbers" else (f"geo_{d}", f"cka_{d}")
        nf = [e["step"] for e in on if e.get(gk) is not None and not _num(e[gk])]
        if nf:
            why.append(f"{gk} not finite at {len(nf)} evals with prior_on (steps {nf[:5]}): unverifiable")
        for e in on:
            if _num(e.get(gk)) and e[gk] <= GEO_TOL:
                x = {"prior": d, "step": e["step"], "geo": e[gk], "cka": e.get(kk)}
                low.append(x)
                if x["geo"] < GEO_FLOOR:
                    why.append(f"(2) step {x['step']}: {gk} {x['geo']:.6f} < the floor {GEO_FLOOR}")
                if not (_num(x["cka"]) and x["cka"] >= GEO_CKA_MIN):
                    why.append(f"(2) step {x['step']}: {kk} {x['cka']} < {GEO_CKA_MIN} (not a perfect fit)")
                elif x["cka"] > GEO_CKA_MAX:
                    why.append(f"(2) step {x['step']}: {kk} {x['cka']} > {GEO_CKA_MAX} (above any float overshoot: an anomaly)")
    if on and not low and not gp_bad:
        why.append(f"the check failed, but no active prior's value is <= {GEO_TOL} at an eval with prior_on and grad_prior is finite > 0: "
                   "the failure is not the A6 case")
    ina = [x for x in MANI if x not in pri and x != "numbers"]
    bad3 = sorted({f"geo_{x}" for e in c for x in ina if e.get(f"geo_{x}") is not None}
                  | {f"gnorm_prior_{x}" for e in c for x in ina if f"gnorm_prior_{x}" in e})
    if "numbers" not in pri and (any(e.get("geo") not in (None, 0) for e in c) or any("gnorm_prior_numbers" in e for e in c)):
        bad3.append("the inactive numbers prior has a value / gradient")
    if bad3:
        why.append(f"(3) an inactive prior has a geo / prior-gradient field: {bad3[:6]}")
    ckn = [(k_["check"], k_["ok"]) for k_ in ck.get("checks", [])]
    for p in KEEP6 + ((M6.NUM_ON_CHECK,) if "numbers" in pri else (NUM_OFF,)):
        m = [o for n, o in ckn if n.startswith(p)]
        if m != [True]:
            why.append(f"(3) the separate check '{p} ...' is not present once and passed ({m})")
    gp = [e["grad_prior"] for e in on if _num(e.get("grad_prior"))]
    return {"why": why, "flag": {"active": list(pri), "n_on": len(on), "grad_prior_min": min(gp) if gp else None, "evals": low}}


@lru_cache(maxsize=None)
def a6_guard6(job_dir=str(HERE)):
    """The check names the loader matches by prefix occur in the CHECK_PY_6 of every rendered pt6comb job file (job_pt6comb-*.sbatch
    in job_dir): the geo check's template once, geo_tol == -0.001, KEEP6 once each, the numbers-prior check of the arm once."""
    files = sorted(Path(job_dir).glob("job_pt6comb-*.sbatch"))
    assert files, f"A6 guard: no rendered job file job_pt6comb-*.sbatch in {job_dir}"
    sep = "<<'PY' | tee \"$OUT/check.txt\"\n"
    tmpl = 'chk(f"' + M6.GEO_NAME6
    for f in files:
        txt = f.read_text()
        m = re.search(r"\"\$OUT/check.json\" '(\{.*\})' \"\$ARGS\" <<'PY'", txt)
        assert m and txt.count(sep) == 1, f"A6 guard: {f.name}: no CHECK_PY heredoc / intent found"
        cp = txt.split(sep, 1)[1].split("\nPY\n", 1)[0]
        assert cp + "\n" == M6.CHECK_PY_6, f"A6 guard: {f.name}: its CHECK_PY is not CHECK_PY_6"
        I = json.loads(m.group(1))
        assert cp.count(tmpl) == 1 and I.get("geo_tol") == GEO_TOL, f"A6 guard: {f.name}: the geo check / geo_tol"
        miss = [p for p in KEEP6 + (M6.NUM_ON_CHECK, NUM_OFF) if cp.count(f'chk("{p}') != 1]
        assert not miss, f"A6 guard: {f.name}: the separate checks {miss} are not named so (once)"
    return len(files)


# ------------------------------------------------------------------------------------------------------------ loading
def cell_dirs(root):
    out = {}
    for d in sorted(Path(root).glob("pt6comb-*-s*")):
        m = NAME.match(d.name)
        if m:
            out.setdefault((m.group(2), m.group(3), int(m.group(4))), []).append((int(m.group(5) or 0), d))
    return {k: sorted(v) for k, v in out.items()}


def run_json(d, arm, lr, seed):
    tag = M6.tag6(COMP, arm, lr)
    tgt = "helix3" if "numbers" in M6.pri_of(arm, COMP) else "none"
    return Path(d) / tag / f"run_whole_{tgt}_seed{seed}_{tag}.json"


def identity(r, arm, lr, seed):
    """The run's own record = its directory name (ptdom2's identity check)."""
    a = r.get("args", {})
    pri = M6.pri_of(arm, COMP)
    want = {"seed": seed, "lr": float(lr), "tag": f"_{M6.tag6(COMP, arm, lr)}", "target": "helix3" if "numbers" in pri else "none",
            "prior_domains": ",".join(pri), "domains": M6.domains_arg(COMP), "prior_shuf": ""}
    bad = {k: [a.get(k), v] for k, v in want.items() if a.get(k) != v}
    bad.update({f"domains.{d}.prior": [r["domains"].get(d, {}).get("prior"), d in pri] for d in MANI
                if r["domains"].get(d, {}).get("prior") != (d in pri)})
    return bad


def load(root, guard=True):
    """{(arm, lr, seed): rec} of every pt6comb cell under root (the latest run dir of the cell)."""
    if guard:
        a6_guard6()
    out = {}
    for (arm, lr, seed), dirs in cell_dirs(root).items():
        k_, d = dirs[-1]
        f = run_json(d, arm, lr, seed)
        early = []
        for k0, d0 in dirs[:-1]:
            f0 = run_json(d0, arm, lr, seed)
            try:
                if f0.exists() and len(json.loads(f0.read_text())["curve"]) == N_EVALS:
                    early.append(d0.name)
            except (json.JSONDecodeError, KeyError):
                pass
        rec = {"arm": arm, "lr": lr, "seed": seed, "dir": str(d), "rerun": k_, "earlier_finished": early, "valid": False, "why": "",
               "finished": False, "problems": []}
        out[arm, lr, seed] = rec
        if arm not in M6.arms(COMP) or lr not in M6.LADDER:
            rec["why"] = f"not a pt6comb arm / lr ({arm}, {lr})"
            continue
        if not f.exists():
            rec["why"] = "no run JSON"
            continue
        r = json.loads(f.read_text())
        c = r["curve"]
        ck = json.loads((d / "check.json").read_text()) if (d / "check.json").exists() else None
        fails = [k["check"] for k in (ck or {}).get("checks", []) if not k["ok"]]
        dpf = [k["check"].split(" (")[0] for k in (ck or {}).get("checks", []) if not k["ok"] and k["check"].startswith(DP_FLAG_CHECKS)]
        div = next((e["step"] for e in c if not all(v is None or math.isfinite(v) for v in (e.get("lm"), e.get("val_loss"), e.get("geo"),
                    e.get("grad_prior"), *(e.get(f"geo_{d}") for d in MANI if d != "numbers")))), None)   # A1 (audit m3): = CHECK_PY_6's onset
        rec.update(code_sha=r.get("code_sha"), n_evals=len(c), finished=len(c) == N_EVALS, diverged_at=div, fails=fails,
                   data_sha=[e.get("data_sha") for e in c], lm0=c[0]["lm"],
                   bpb_intr=c[-1]["val_bits_per_byte"] if div is None else math.inf,
                   val_loss=c[-1]["val_loss"] if div is None else math.inf, grad=RO.window_grad(c, div))
        idb = identity(r, arm, lr, seed)
        a6 = None
        geo_f = [x for x in fails if x.startswith(GEO_FLAG6)]
        if idb:
            rec["why"] = f"identity: the run's record differs from its directory name {idb}"
        elif ck is None:
            rec["why"] = "no check.json"
        elif len(c) != N_EVALS:
            rec["why"] = f"unfinished ({len(c)} evals)"
        elif fails and not all(x.startswith((CKA_CHECK, GEO_FLAG6) + DP_FLAG_CHECKS) for x in fails):
            rec["why"] = "failed check: " + "; ".join(fails)
        elif geo_f and (a6 := geo_a6_multi(r, ck, arm))["why"]:
            rec["why"] = "failed check: " + "; ".join(fails) + " (A6 does not apply: " + "; ".join(a6["why"]) + ")"
        else:
            rec["valid"] = True
            rec["dp_flag"] = dpf
            rec["geo_flag"] = a6["flag"] if a6 else None
            rec["why"] = ("valid" + (" (CKA-rise check failed: a weak prior, data)" if any(x.startswith(CKA_CHECK) for x in fails) else "")
                          + (f" (FLAG A5.1: {'; '.join(dpf)}; valid data)" if dpf else "")
                          + (f" (FLAG A6: geo overshoot at a perfect fit {rec['geo_flag']['evals']}; valid data)" if a6 else ""))
        if rec["finished"]:
            mt, probs = run_metrics(r)
            rec.update(mt)
            rec["problems"] = probs
    return out


def integrity(R_, seeds=None):
    """The same data per seed across every valid run (data_sha at every eval; step-0 lm within 1e-3); one code_sha; no cell with
    a FINISHED run in an earlier dir; every valid run's metric double paths agree."""
    probs, V = [], [r for k, r in R_.items() if r["valid"] and (seeds is None or k[2] in seeds)]
    for k, r in sorted(R_.items()):
        if r.get("earlier_finished"):
            probs.append(f"{k[0]}@{k[1]} s{k[2]}: a finished run in {r['earlier_finished']} and a later re-run dir {Path(r['dir']).name}")
        if r["valid"] and r.get("problems"):
            probs += [f"{k[0]}@{k[1]} s{k[2]}: {p}" for p in r["problems"]]
    if len({r["code_sha"] for r in V}) > 1:
        probs.append(f"code_sha not unique: {sorted({r['code_sha'] for r in V})}")
    for s in sorted({r["seed"] for r in V}):
        rs = [r for r in V if r["seed"] == s]
        if len({tuple(r["data_sha"]) for r in rs}) != 1:
            probs.append(f"seed {s}: data_sha series differ across runs (the data must be identical in every arm)")
        if max(r["lm0"] for r in rs) - min(r["lm0"] for r in rs) > 1e-3:
            probs.append(f"seed {s}: step-0 lm differs by > 1e-3 across runs")
    return probs


# ------------------------------------------------------------------------------------------------------------ selection
def mean6(r):
    return float(np.mean([r["area"][m] for m in MANI]))


def lrs_ran(R_, arm, seeds=SEL):
    return sorted({k[1] for k in R_ if k[0] == arm and k[2] in seeds}, key=float)


def gate_cells(R_, arm, lrs, seeds):
    """A2.1 per (arm, lr) cell over the FINISHED runs at the seeds (valid or not)."""
    out = {}
    for lr in lrs:
        rs = [R_[arm, lr, s] for s in seeds if (arm, lr, s) in R_ and R_[arm, lr, s].get("finished")]
        if rs:
            out[lr] = RO.cell_gate(rs)
    return out


def select_unit(R_, arm, lrs, excluded=(), seeds=SEL):
    """F87 rule 1 on mean6 over lrs minus the gate-failed ones; refuses ({'missing'}) when a cell lacks a valid run at a seed."""
    cells, miss = {}, []
    for lr in lrs:
        rs = [R_.get((arm, lr, s)) for s in seeds]
        if not all(r and r["valid"] for r in rs):
            miss += [f"{arm}@{lr} s{s}" for s, r in zip(seeds, rs) if not (r and r["valid"])]
            continue
        if lr not in excluded:
            cells[lr] = {"area": [mean6(r) for r in rs], "off": [None] * len(rs), "intr": [r["bpb_intr"] for r in rs],
                         "per_m": {m: float(np.mean([r["area"][m] for r in rs])) for m in MANI}}
    if miss:
        return {"missing": miss}
    adm = sorted(cells, key=float)
    if not adm:
        return {"grid": [], "excluded": sorted(excluded, key=float), "pick": None, "argmax": None, "mean": {}, "near": []}
    b = RR.best_lr({lr: {k: v for k, v in x.items() if k != "per_m"} for lr, x in cells.items()}, near_rule="top2")
    k = max(range(len(adm)), key=lambda j: b["mean"][adm[j]])
    return {"grid": adm, "excluded": sorted(set(excluded) & set(lrs), key=float), "mean": {x: round(v, 4) for x, v in b["mean"].items()},
            "per_m": {lr: {m: round(v, 3) for m, v in cells[lr]["per_m"].items()} for lr in adm},
            "bpb": {x: round(v, 6) for x, v in b["bpb"].items()}, "pick": b["best"], "near": b["near"],
            "tie_break": "in-training bpb" if len(b["near"]) > 1 else None, "argmax": adm[k]}


def decide(R_, after, prev=None):
    """The pre-registered selection (select_pt6comb.py; the read-out re-derives it). after 'L1': the L1 cells; 'L1x': the L1 decision
    (prev, which must be re-derived exactly) + its asked cells. -> the decision dict (status decided / GATE / NOT FINAL / missing)."""
    assert after in ("L1", "L1x")
    lrs = {"all6": list(GRID_L1["all6"]), "none": list(GRID_L1["none"])}
    if after == "L1x":
        assert prev is not None and prev.get("status") == "decided" and prev.get("stage_written") == "L1x", prev
        for a, lr in prev["E"]:
            if lr not in lrs[a]:
                lrs[a].append(lr)
        lrs = {a: sorted(v, key=float) for a, v in lrs.items()}
    gate = gate_cells(R_, "all6", lrs["all6"], SEL)
    bad = sorted(lr for lr, g in gate.items() if not g["ok"])
    miss = [f"all6@{lr}: no finished run at seeds 0-3" for lr in lrs["all6"] if lr not in gate]
    ua = select_unit(R_, "all6", lrs["all6"], bad)
    un = select_unit(R_, "none", lrs["none"])
    if miss or "missing" in ua or "missing" in un:
        return {"status": "missing", "missing": miss + ua.get("missing", []) + un.get("missing", [])}
    base = {"design": M6.PREFIX, "comp": COMP, "x": None, "after": after, "units": {"all6": ua, "none": un},
            "gate": {lr: {k: g[k] for k in ("ok", "why", "median", "min", "max", "per_run", "frac_b_gt_a", "frac_clip_lt1")} for lr, g in gate.items()},
            "lrs": lrs, "notes": []}
    if ua["pick"] is None:
        return dict(base, status="GATE", stage_written=None, E=[], C=[], pick_all6=None, pick_none=un["pick"],
                    notes=["every all6 cell fails the auxiliary-loss gate: no pick, no confirmation (the coordinator decides)"])
    P_, N_ = ua["pick"], un["pick"]
    if P_ == min(lrs["all6"], key=float):
        base["notes"].append(f"PRIOR AT ITS LOWEST LR: all6's pick {P_} is its lowest lr (no lower extension, ptdom2 A1.1 (i): "
                             "conservative for HELPS; the interference contrast is at 2e-3 whatever P is, A2)")
    top_a, lo_n, hi_n = max(lrs["all6"], key=float), min(lrs["none"], key=float), max(lrs["none"], key=float)
    if after == "L1":
        E = []
        if top_a in (P_, ua["argmax"]) and top_a not in bad:
            E.append(("all6", M6.EXT_UP["all6"]))
        if hi_n in (N_, un["argmax"]):
            E.append(("none", M6.EXT_UP["none"]))
        if lo_n in (N_, un["argmax"]):
            E.append(("none", M6.EXT_DOWN["none"]))
        rng = [float(lr) for a, lr in E if a == "none"] + [float(x) for x in lrs["none"]]
        if not (min(rng) <= float(P_) <= max(rng)):   # fixB (ptdom A5.2 / ptdom2 A1.1 (b)): none at all6's pick
            assert P_ in M6.FIXB_LR, P_
            E.append(("none", P_))
        E = list(dict.fromkeys(E))
        if E:
            return dict(base, status="decided", stage_written="L1x", E=[list(x) for x in E], C=[], pick_all6=P_, pick_none=N_)
    else:
        nf = []
        if top_a in (P_, ua["argmax"]):
            nf.append(f"all6's pick {P_} / argmax {ua['argmax']} at its top lr {top_a}")
        for edge in (lo_n, hi_n):
            if edge in (N_, un["argmax"]):
                nf.append(f"none's pick {N_} / argmax {un['argmax']} at an outermost lr it ran ({edge})")
        if not (float(lo_n) <= float(P_) <= float(hi_n)):
            nf.append(f"all6's pick {P_} outside none's range {lo_n}..{hi_n}")
        if nf:
            return dict(base, status="NOT FINAL", stage_written=None, E=[], C=[], pick_all6=P_, pick_none=N_, notes=base["notes"] + nf)
    C = [list(c) for c in c_cells(P_, N_)]   # A2: + the same-lr pair none@2e-3 / all6@2e-3 (merged when P / N = 2e-3)
    return dict(base, status="decided", stage_written="C", E=[], C=C, pick_all6=P_, pick_none=N_)


def strip(dec):
    """The fields a picks file must reproduce exactly (re-derivation check)."""
    return json.loads(json.dumps({k: dec.get(k) for k in ("design", "comp", "x", "after", "status", "stage_written", "E", "C",
                                                         "pick_all6", "pick_none", "lrs", "units", "gate", "notes")}))


# ------------------------------------------------------------------------------------------------------------ verdicts
def t_one(g):
    """One-sided one-sample t p of H1 mean > 0 (df n - 1) = readout_ptdom.sf_test."""
    return RO.sf_test(g)


def manifold_verdict(g, h, u):
    """HELPS / NO INTERFERENCE of one manifold (A2): g = all6@P - none@N (HELPS, the original rule); h = all6@2e-3 - only-m@2e-3 and
    u = only-m@2e-3 - none@2e-3 (the SAME-lr interference contrast). INTERFERENCE NOT DETECTED iff (mean u > 0 and mean h >= -0.5 x mean
    u) or the one-sided t p (H1: mean h < 0) >= 0.05; when mean u <= 0 the margin clause does not apply (the single prior shows no gain
    itself) and only the t clause decides."""
    g = np.asarray(g, float)
    p_help, wins = t_one(g), int((g > 0).sum())
    helps = bool(p_help < ALPHA and wins >= MIN_WINS)
    if h is None:   # A3: the interference contrast withheld (the same-lr pair missing / invalid)
        mu, p2, ci = RO.t_test(g)
        return {"g": g.tolist(), "h": None, "u": None, "mean_g": float(g.mean()), "wins_g": wins, "p_help": p_help, "ci95_g": list(ci),
                "boot90_g": list(RO.boot_ci(g)), "helps": helps, "mean_h": None, "mean_u": None, "margin": None, "p_int": None, "noint": None,
                "noint_by": "", "neg_u_note": False, "ci95_h": None, "boot90_h": None, "boot90_u": None}
    h, u = np.asarray(h, float), np.asarray(u, float)
    p_int = t_one(-h)                       # H1: mean h < 0 (all6 below only-m)
    u_pos = float(u.mean()) > 0
    margin = -INT_FRAC * float(u.mean()) if u_pos else None
    by_m = u_pos and float(h.mean()) >= margin
    by_p = p_int >= ALPHA
    noint = bool(by_m or by_p)
    mu, p2, ci = RO.t_test(g)
    return {"ci95_h": list(RO.t_test(h)[2]), "boot90_h": list(RO.boot_ci(h)), "boot90_u": list(RO.boot_ci(u)), "g": g.tolist(), "h": h.tolist(),
            "u": u.tolist(), "mean_g": float(g.mean()), "wins_g": wins, "p_help": p_help, "ci95_g": list(ci), "boot90_g": list(RO.boot_ci(g)),
            "helps": helps, "mean_h": float(h.mean()), "mean_u": float(u.mean()), "margin": margin, "p_int": p_int, "noint": noint,
            "noint_by": ("margin" if by_m else "") + ("+p" if (by_m and by_p) else ("p" if by_p else "")), "neg_u_note": not u_pos}


def c_cells(P_, N_):
    """A2: the confirmation cells: none@N, all6@P, the same-lr pair none@2e-3 / all6@2e-3 (merged with N / P when equal) and the six
    only-<m>@2e-3, in this order, no cell twice."""
    out = []
    for c in [("none", N_), ("all6", P_), ("none", SINGLE_LR), ("all6", SINGLE_LR)] + [(f"only-{m}", SINGLE_LR) for m in MANI]:
        if c not in out:
            out.append(c)
    return out


def verdict(R_, dec, allow_partial=False):
    """The confirmation verdicts on seeds 8-15 at the final picks (dec = the final decision). -> result dict. A3: the same-lr pair
    (none@2e-3 / all6@2e-3 where not merged with none@N / all6@P) is checked separately: a missing / invalid run there withholds only
    the interference words and the same-lr rows (res['missing_pair']); all6@2e-3 has its own per-half gate."""
    P_, N_ = dec["pick_all6"], dec["pick_none"]
    L2 = SINGLE_LR
    cells = {"none": ("none", N_), "all6": ("all6", P_), "none2": ("none", L2), "all62": ("all6", L2),
             **{f"only-{m}": (f"only-{m}", L2) for m in MANI}}
    core = list(dict.fromkeys([cells["none"], cells["all6"]] + [cells[f"only-{m}"] for m in MANI]))
    pair = [c for c in (cells["none2"], cells["all62"]) if c not in core]
    vok = lambda c, s: bool((R_.get((c[0], c[1], s)) or {}).get("valid"))
    miss = [f"{a}@{lr} s{s}" for a, lr in core for s in CONF if not vok((a, lr), s)]
    get = lambda a, s: R_[(cells[a][0], cells[a][1], s)]
    seeds = [s for s in CONF if all(vok(c, s) for c in core)]
    miss_pair = [f"{a}@{lr} s{s}" for a, lr in pair for s in seeds if not vok((a, lr), s)]
    pair_ok = not miss_pair
    res = {"picks": {"all6": P_, "none": N_, "only": L2, "same_lr": L2}, "missing": miss, "missing_pair": miss_pair, "pair_ok": pair_ok,
           "seeds": seeds, "m": {}}
    if miss and not allow_partial:
        return res
    if len(seeds) < 2:
        res["empty"] = True
        return res
    A = lambda a, m, f="area": [get(a, s)[f][m] for s in seeds]
    dif = lambda a, b, m, f="area": [x - y for x, y in zip(A(a, m, f), A(b, m, f))]
    for m in MANI:
        o = f"only-{m}"
        v = manifold_verdict(dif("all6", "none", m), dif("all62", o, m) if pair_ok else None, dif(o, "none2", m) if pair_ok else None)
        v["final3"] = {"g": mean(dif("all6", "none", m, "final3")), "wins_g": sum(x > 0 for x in dif("all6", "none", m, "final3")),
                       "h": mean(dif("all62", o, m, "final3")) if pair_ok else None, "u": mean(dif(o, "none2", m, "final3")) if pair_ok else None}
        # descriptive: the same-lr HELPS row all6@2e-3 - none@2e-3 (A2; withheld with the pair, A3) and all6@P - only-m (lr-confounded if P != 2e-3)
        hP = dif("all6", o, m)
        if pair_ok:
            g2 = dif("all62", "none2", m)
            v["g_same_lr"] = {"mean": mean(g2), "boot90": list(RO.boot_ci(g2)), "wins": sum(x > 0 for x in g2)}
        else:
            v["g_same_lr"] = None
        v["h_at_P"] = {"mean": mean(hP), "boot90": list(RO.boot_ci(hP)), "lr_confounded": P_ != L2}
        ws = [window_series_split(get("all6", s)["series"][m], get("none", s)["series"][m], get("all6", s)["steps"]) for s in seeds]
        v["window_g"] = {"in": mean([w["in"] for w in ws]), "out": mean([w["out"] for w in ws]), "n_in": ws[0]["n_in"]}
        line = mean([get("none", s)["floor"][m] for s in seeds])
        ma, mn, mo = mean(A("all6", m)), mean(A("none", m)), mean(A(o, m))
        v["levels"] = {"all6": ma, "none": mn, o: mo, "all6@2e-3": mean(A("all62", m)) if pair_ok else None,
                       "none@2e-3": mean(A("none2", m)) if pair_ok else None, "floor_line": line}
        v["tags"] = (["FLOOR"] if ma <= line and mn <= line else []) + (["CEILING"] if mn >= 95 else [])
        v["cka_win"] = {a: mean([get(a, s)["cka_win"][m] for s in seeds if get(a, s)["cka_win"].get(m) is not None])
                        for a in ("all6", "none", o)}
        sk = {"s4": ("s4_readout", "area", "s4 held_fresh_nonid area"), "dim": ("dim_readout", "area", "dim no_negtwin_nonid area")}.get(
            m, (f"{m}_hard", "final3", f"{m} hard-subset final3") if m in CMPD else None)
        if sk:
            sv = lambda a: [get(a, s)["sec"][sk[0]][sk[1]] for s in seeds]
            arms_ = ("all6", "none", "all62", o) if pair_ok else ("all6", "none")
            ok_ = all(x is not None for a in arms_ for x in sv(a))
            v["secondary"] = {"name": sk[2], "g": mean([x - y for x, y in zip(sv("all6"), sv("none"))]) if ok_ else None,
                              "h": mean([x - y for x, y in zip(sv("all62"), sv(o))]) if ok_ and pair_ok else None,
                              "wins_g": sum(x > y for x, y in zip(sv("all6"), sv("none"))) if ok_ else None}
        v["lr_conf"] = {"g": P_ != N_, "h": False, "u": False, "h_at_P": P_ != L2}
        res["m"][m] = v
    holm = RO.holm([res["m"][m]["p_help"] for m in MANI])
    for m, ph in zip(MANI, holm):
        res["m"][m]["p_help_holm"] = ph
        res["m"][m]["helps_holm"] = bool(ph < ALPHA and res["m"][m]["wins_g"] >= MIN_WINS)
    # the gate at all6@P, per half (ptdom A4.5 / ptdom2 A1.7); A3: and at all6@2e-3, the arm of h (the same cell when P = 2e-3)
    halves = {"8-11": [s for s in seeds if s <= 11], "12-15": [s for s in seeds if s >= 12]}
    g_h = {k: RO.cell_gate([get("all6", s) for s in v]) for k, v in halves.items() if v}
    res["gate_all6"] = {k: {"ok": g["ok"], "median": g["median"], "why": g["why"]} for k, g in g_h.items()}
    res["gate_failed"] = any(not g["ok"] for g in g_h.values())
    if pair_ok:
        g_2 = {k: RO.cell_gate([get("all62", s) for s in v]) for k, v in halves.items() if v}
        res["gate_all6_same_lr"] = {k: {"ok": g["ok"], "median": g["median"], "why": g["why"]} for k, g in g_2.items()}
        res["gate_same_lr_failed"] = any(not g["ok"] for g in g_2.values())
    else:
        res["gate_all6_same_lr"], res["gate_same_lr_failed"] = None, None
    res["arms"] = {}
    for a, (arm, lr) in cells.items():
        if a in ("none2", "all62") and ((arm, lr) in (cells["none"], cells["all6"]) or not pair_ok):
            continue   # merged with none@N / all6@P (A2), or the pair is withheld (A3)
        rs = [get(a, s) for s in seeds]
        g8 = RO.cell_gate(rs)
        res["arms"][f"{arm}@{lr}"] = {"lr": lr, "lm_share": g8["median"], "lm_share_min": g8["min"], "gate_ok": g8["ok"],
                                      "frac_b_gt_a": g8["frac_b_gt_a"], "frac_clip_lt1": g8["frac_clip_lt1"],
                                      "clip_coef": med([r["grad"]["clip_coef"] for r in rs if r["grad"].get("clip_coef") is not None]),
                                      "d_val_loss": None if a == "none" else mean([r["val_loss"] - get("none", s)["val_loss"] for r, s in zip(rs, seeds)]),
                                      "d_bpb_1e3": None if a == "none" else mean([1e3 * (r["bpb_intr"] - get("none", s)["bpb_intr"]) for r, s in zip(rs, seeds)]),
                                      "lr_confounded": a != "none" and lr != N_}
    # A3 (m6): the same-lr LM cost all6@2e-3 - none@2e-3 on seeds 8-15
    res["lm_same_lr"] = ({"d_val_loss": mean([get("all62", s)["val_loss"] - get("none2", s)["val_loss"] for s in seeds]),
                          "d_bpb_1e3": mean([1e3 * (get("all62", s)["bpb_intr"] - get("none2", s)["bpb_intr"]) for s in seeds])}
                         if pair_ok else None)
    per_seed = [float(np.mean([res["m"][m]["g"][i] for m in MANI])) for i in range(len(seeds))]
    mu, _, ci = RO.t_test(per_seed)
    res["mean6_gain"] = {"per_seed": per_seed, "mean": mu, "ci95": list(ci), "boot90": list(RO.boot_ci(per_seed))}
    helps_ = [m for m in MANI if res["m"][m]["helps"]]
    if res["gate_failed"]:
        res["headline"] = {"k": None, "list": [], "text": GATE_TXT}
    elif not pair_ok or res["gate_same_lr_failed"]:
        why = ("the same-lr pair is incomplete (" + ", ".join(miss_pair[:4]) + ")" if not pair_ok else
               "AUX GATE FAILED at all6@2e-3 (the arm of the interference contrast)")
        res["headline"] = {"k": None, "list": [], "helps": helps_, "text": f"interference words withheld: {why}; HELPS (all6@{P_} vs none@{N_}): "
                           f"{len(helps_)} of 6: {', '.join(helps_) or 'none'}"}
    else:
        ok = [m for m in MANI if res["m"][m]["helps"] and res["m"][m]["noint"]]
        res["headline"] = {"k": len(ok), "list": ok, "helps": helps_, "text": f"{len(ok)} of 6 manifolds HELPS (judged at all6@{P_} vs none@{N_}) "
                           f"with no interference detected (judged at the same lr {L2}): {', '.join(ok) or 'none'}"}
    return res


def same_lr_lm(R_):
    """Descriptive, seeds 0-3: the LM cost all6 - none at every lr both ran (in-training val loss / bpb)."""
    out = {}
    for lr in sorted(set(lrs_ran(R_, "all6")) & set(lrs_ran(R_, "none")), key=float):
        pr = [(R_.get(("all6", lr, s)), R_.get(("none", lr, s))) for s in SEL]
        pr = [(a, b) for a, b in pr if a and b and a["valid"] and b["valid"]]
        if pr:
            out[lr] = {"n": len(pr), "d_val_loss": mean([a["val_loss"] - b["val_loss"] for a, b in pr]),
                       "d_bpb_1e3": mean([1e3 * (a["bpb_intr"] - b["bpb_intr"]) for a, b in pr])}
    return out


# ------------------------------------------------------------------------------------------------------------ report
f1 = lambda v, p=1: "-" if v is None else f"{v:+.{p}f}"
f0 = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"


def report(sel_txt, res, dec, prov, problems):
    L = ["pt6comb read-out (runs/pt6comb/AUDIT.md; composition B: numbers, s4 (s4_hard), dim (strict), cities, countries, people)",
         f"picks: all6@{dec.get('pick_all6')}, none@{dec.get('pick_none')}, only-<m>@{SINGLE_LR}; decision status {dec.get('status')} "
         f"(after {dec.get('after')})"]
    if prov:
        L.append("PROVISIONAL (no verdict word is final): " + "; ".join(prov))
    if problems:
        L.append("PROBLEMS: " + "; ".join(problems[:20]))
    L += [sel_txt, ""]
    if res.get("empty") or not res.get("m"):
        L.append(f"no complete seed at the confirmation cells (missing {len(res.get('missing', []))} runs)")
        return "\n".join(L)
    P_, N_, L2 = res["picks"]["all6"], res["picks"]["none"], res["picks"]["same_lr"]
    L.append(f"confirmation seeds {res['seeds']} (fresh). HELPS is judged at all6@{P_} vs none@{N_}; interference at the same lr {L2} (A2). "
             f"Per manifold (area points): g = all6@{P_} - none@{N_}; h = all6@{L2} - only-m@{L2}; u = only-m@{L2} - none@{L2}. HELPS iff one-sided "
             f"t p < {ALPHA} and g > 0 in >= {MIN_WINS}/8; interference not detected iff (mean u > 0 and mean h >= -{INT_FRAC} x mean u) or the "
             f"one-sided t p (H1 h < 0) >= {ALPHA} (mean u <= 0: the t clause only). 'Interference not detected' never reads 'shown to be absent'.")
    pair_word = (None if res.get("pair_ok") and not res.get("gate_same_lr_failed") else
                 "AUX GATE FAILED at all6@2e-3" if res.get("pair_ok") else "WITHHELD (the same-lr pair is incomplete)")
    if not res.get("pair_ok"):
        L.append(f"  the same-lr pair is incomplete: {res['missing_pair'][:8]}: every interference word and every same-lr row is withheld; "
                 "HELPS is read as usual (A3)")
    hdr = (f"  {'manifold':10s} {'all6@P':>6s} {'none@N':>6s} {'a6@2e-3':>7s} {'n@2e-3':>6s} {'only':>6s} | {'g':>6s} {'wins':>4s} {'p':>6s} "
           f"{'p Holm':>6s} {'95% CI':>15s} | {'h':>6s} {'h 90% (boot)':>15s} {'u':>6s} {'u 90% (boot)':>15s} {'margin':>6s} {'p(h<0)':>6s} | verdict")
    L.append(hdr)
    bi = lambda x: "-" if x is None else f"[{x[0]:+6.1f},{x[1]:+6.1f}]"
    for m in MANI:
        v = res["m"][m]
        if res["gate_failed"]:
            word = "all6 GATE"
        else:
            word = ("HELPS" if v["helps"] else "NOT SHOWN") + " + " + (pair_word or ("interference not detected" if v["noint"] else "INTERFERENCE"))
            if prov:
                word = f"PROVISIONAL ({word} if final)"
        lv = v["levels"]
        L.append(f"  {m:10s} {f0(lv['all6']):>6s} {f0(lv['none']):>6s} {f0(lv['all6@2e-3']):>7s} {f0(lv['none@2e-3']):>6s} {f0(lv['only-' + m]):>6s} | "
                 f"{f1(v['mean_g']):>6s} {v['wins_g']}/{len(v['g'])} {v['p_help']:6.3f} {v['p_help_holm']:6.3f} {bi(v['ci95_g'])} | "
                 f"{f1(v['mean_h']):>6s} {bi(v['boot90_h'])} {f1(v['mean_u']):>6s} {bi(v['boot90_u'])} {f1(v['margin']):>6s} "
                 f"{'-' if v['p_int'] is None else format(v['p_int'], '6.3f'):>6s} | {word}" + (f" [{', '.join(v['tags'])}]" if v["tags"] else "")
                 + (" (HELPS also under Holm)" if v["helps_holm"] else (" (not under Holm)" if v["helps"] else "")))
        lc = [k for k, x in v["lr_conf"].items() if x]
        if v["noint"] is not None and not pair_word:
            L.append(f"  {'':10s} interference " + (f"not detected by {v['noint_by'] or '-'}" if v["noint"] else "detected") + f"; h 95% t CI "
                     f"[{v['ci95_h'][0]:+.1f}, {v['ci95_h'][1]:+.1f}]"
                     + ("; the single prior shows no gain itself (mean u <= 0): the margin clause does not apply, the t clause alone decides" if v["neg_u_note"] else "")
                     + ("; by p alone = 'interference not detected at 8 seeds' (low power, AUDIT A1)" if v["noint"] and v["noint_by"] == "p" else "")
                     + (f"; lr-confounded contrasts: {', '.join(lc)}" if lc else ""))
        elif v["noint"] is not None:
            L.append(f"  {'':10s} {pair_word}: the interference word is not given (h {f1(v['mean_h'])}, u {f1(v['mean_u'])} printed only)")
        gs = v.get("g_same_lr")
        L.append(f"  {'':10s} descriptive: same-lr gain all6@{L2} - none@{L2} " + (f"{f1(gs['mean'])} {bi(gs['boot90'])} ({gs['wins']}/{len(v['g'])})"
                 if gs else "withheld (the pair is incomplete)") + f"; all6@{P_} - only-m@{L2} {f1(v['h_at_P']['mean'])} {bi(v['h_at_P']['boot90'])}"
                 + (" (lr-confounded)" if v["h_at_P"]["lr_confounded"] else " (= h: P is the same lr)"))
    hl = res["headline"]
    L += ["", f"HEADLINE: {hl['text']}" + ("" if res["gate_failed"] else
          " (uncorrected for the six tests; the Holm-adjusted HELPS p (family of six) is the reference column)"),
          f"mean gain over the six manifolds (per-seed mean of g): {res['mean6_gain']['mean']:+.2f}, 95% t CI [{res['mean6_gain']['ci95'][0]:+.2f}, "
          f"{res['mean6_gain']['ci95'][1]:+.2f}], bootstrap 90% [{res['mean6_gain']['boot90'][0]:+.2f}, {res['mean6_gain']['boot90'][1]:+.2f}]",
          "per-manifold gain g: mean [95% t CI] / bootstrap 90%: " + "; ".join(
              f"{m} {res['m'][m]['mean_g']:+.1f} [{res['m'][m]['ci95_g'][0]:+.1f}, {res['m'][m]['ci95_g'][1]:+.1f}] / [{res['m'][m]['boot90_g'][0]:+.1f}, "
              f"{res['m'][m]['boot90_g'][1]:+.1f}]" for m in MANI), "",
          "descriptive (never a condition):"]
    for m in MANI:
        v = res["m"][m]
        L.append(f"  {m:10s} final3: g {f1(v['final3']['g'])} ({v['final3']['wins_g']}/{len(v['g'])}), h {f1(v['final3']['h'])}, u {f1(v['final3']['u'])}; "
                 f"window split of g: in {f1(v['window_g']['in'])} ({v['window_g']['n_in']} evals) + out {f1(v['window_g']['out'])}; FLOOR line "
                 f"{f0(v['levels']['floor_line'])}" + (" (> 95: a non-geometric line is near the ceiling; the FLOOR tag then holds almost "
                 "unconditionally)" if v["levels"]["floor_line"] > 95 else "") + "; eval CKA at the window's last eval all6 / only / none "
                 + " / ".join(f0(v["cka_win"].get(a), 3) for a in ("all6", f"only-{m}", "none")))
    for m in MANI:
        sc = res["m"][m].get("secondary")
        if sc:
            L.append(f"  secondary {sc['name']}: g {f1(sc['g'])} ({sc['wins_g']}/{len(res['seeds'])}), h {f1(sc['h'])}")
    L.append("  (numbers' CKA column is cka_ops_helix3: the operands of the training batch, read layers; the others are the eval-prompt item CKA)")
    L.append(f"  all6@{res['picks']['all6']} gate per half: " + "; ".join(f"seeds {k}: median LM share {f0(x['median'], 3)} {'ok' if x['ok'] else 'FAIL'}"
                                                                         for k, x in res["gate_all6"].items()))
    if res.get("gate_all6_same_lr") is not None:
        L.append(f"  all6@{res['picks']['same_lr']} gate per half (A3: the arm of h): " + "; ".join(
            f"seeds {k}: median LM share {f0(x['median'], 3)} {'ok' if x['ok'] else 'FAIL'}" for k, x in res["gate_all6_same_lr"].items()))
    if res.get("lm_same_lr"):
        L.append(f"  same-lr LM cost all6@{res['picks']['same_lr']} - none@{res['picks']['same_lr']} (seeds 8-15, in-training): val loss "
                 f"{f1(res['lm_same_lr']['d_val_loss'], 4)}, bpb x1e3 {f1(res['lm_same_lr']['d_bpb_1e3'], 2)}")
    for a, x in res["arms"].items():
        L.append(f"  arm {a:16s} @{x['lr']:6s} LM share a/(a+b) {f0(x['lm_share'], 3)} (min run {f0(x['lm_share_min'], 3)}), b > a "
                 f"{f0(None if x['frac_b_gt_a'] is None else 100 * x['frac_b_gt_a'], 0)}%, clip < 1 {f0(None if x['frac_clip_lt1'] is None else 100 * x['frac_clip_lt1'], 0)}%, "
                 f"clip coef {f0(x['clip_coef'], 3)}; LM cost vs none: val loss {f1(x['d_val_loss'], 4)}, bpb x1e3 {f1(x['d_bpb_1e3'], 2)}"
                 + (" (lr-confounded)" if x["lr_confounded"] else ""))
    L.append("  reading: comparisons and s4 / dim are Tier L (the targets carry the answers); s4 / dim items are single nonce tokens whose "
             "site is the read-out candidate's own (tied) embedding row (AUDIT 'Caveats'); NOT SHOWN / INTERFERENCE at 8 seeds are statements "
             "about this rule, never 'no effect' / 'harm' in general; 'interference not detected' is never 'shown not to interfere' (A2)")
    return "\n".join(L)


def selection_text(R_, dec):
    L = [f"selection (seeds 0-3, mean6 = the mean of the six areas; F87 rule 1 top2 + bpb tie-break; status {dec.get('status')}):"]
    for a in ("all6", "none"):
        u = dec.get("units", {}).get(a, {})
        for lr in u.get("grid", []):
            pm = u["per_m"][lr]
            L.append(f"  {a:5s}@{lr:6s} mean6 {u['mean'][lr]:6.2f} | " + " ".join(f"{m} {pm[m]:5.1f}" for m in MANI)
                     + ("   <- pick" if lr == u.get("pick") else "") + ("  (argmax)" if lr == u.get("argmax") else ""))
        if u.get("excluded"):
            L.append(f"  {a}: gate-excluded {u['excluded']}")
    for lr, g in sorted(dec.get("gate", {}).items(), key=lambda x: float(x[0])):
        L.append(f"  gate all6@{lr}: median LM share {f0(g['median'], 3)} {'ok' if g['ok'] else 'FAIL ' + '; '.join(g['why'])}")
    for n in dec.get("notes", []):
        L.append(f"  NOTE: {n}")
    sl = same_lr_lm(R_)
    if sl:
        L.append("  same-lr LM cost all6 - none (seeds 0-3, in-training): " + "; ".join(
            f"@{lr} val loss {x['d_val_loss']:+.4f}, bpb x1e3 {x['d_bpb_1e3']:+.2f} (n {x['n']})" for lr, x in sl.items()))
    return "\n".join(L)


def final_decision(R_, picks_dir=HERE):
    """The final decision from the picks files (picks_after_L1x.json if present, else picks_after_L1.json), each re-derived and
    compared. -> (decision, problems)."""
    probs = []
    f1_, fx_ = Path(picks_dir) / "picks_after_L1.json", Path(picks_dir) / "picks_after_L1x.json"
    if not f1_.exists():
        return None, ["no picks_after_L1.json"]
    p1 = json.loads(f1_.read_text())
    d1 = decide(R_, "L1")
    if strip(d1) != strip(p1):
        probs.append("picks_after_L1.json differs from its re-derivation")
    if p1.get("stage_written") == "C":
        return p1, probs
    if not fx_.exists():
        return p1, probs + ["the L1x decision is pending (no picks_after_L1x.json)"]
    px = json.loads(fx_.read_text())
    dx = decide(R_, "L1x", p1)
    if strip(dx) != strip(px):
        probs.append("picks_after_L1x.json differs from its re-derivation")
    return px, probs


def readout(root, allow_partial=False, picks_dir=HERE, guard=True):
    R_ = load(root, guard)
    dec, probs = final_decision(R_, picks_dir)
    probs = probs + integrity(R_)
    prov = []
    if dec is None or dec.get("stage_written") != "C" or dec.get("status") != "decided":
        txt = selection_text(R_, decide(R_, "L1") if dec is None else dec) if dec is not None else "no decision yet"
        return {"decision": dec, "problems": probs, "final": False}, txt + "\n(no final decision: no verdicts)\n" + "; ".join(probs)
    res = verdict(R_, dec, allow_partial)
    if res["missing"]:
        prov.append(f"{len(res['missing'])} confirmation runs missing / invalid: {res['missing'][:6]}")
    if probs:
        prov.append(f"{len(probs)} problems")
    if prov and not allow_partial:
        raise SystemExit("REFUSED (strict read-out): " + "; ".join(prov + probs[:10]) + " (--allow_partial prints PROVISIONAL)")
    txt = report(selection_text(R_, dec), res, dec, prov, probs)
    flags = [f"{k[0]}@{k[1]} s{k[2]}: {r['why']}" for k, r in sorted(R_.items()) if r["valid"] and (r.get("dp_flag") or r.get("geo_flag"))]
    if flags:
        txt += "\nFLAGGED (valid data): " + "; ".join(flags)
    return {"decision": dec, "verdict": res, "problems": probs, "provisional": prov, "final": not prov}, txt


# ------------------------------------------------------------------------------------------------------------ self-test
def fake_rec(arm, lr, seed, area, share=0.9, bpb=1.2, vloss=4.0, data="d", lm0=11.6, valid=True, final3=None, floor=None, cka=0.5,
             series=None):
    """A planted record (the loader's fields) for the selection / verdict known answers."""
    steps = list(range(0, 18310, 250)) + [18309]
    ser = series or {m: [area[m] / 100] * len(steps) for m in MANI}
    on = [{"prior_on": 1, "step": s, "grad_lm": 1.0, "grad_prior": (1 - share) / share, "grad_cos": 0.0, "clip_coef": 1.0,
           "grad_lm_share": 1.0 / math.sqrt(1 + ((1 - share) / share) ** 2)} for s in steps if WIN[0] <= s <= WIN[1]] if arm != "none" else []
    return {"arm": arm, "lr": lr, "seed": seed, "valid": valid, "finished": True, "why": "planted", "area": dict(area),
            "final3": dict(final3 or area), "series": ser, "steps": steps, "floor": floor or {m: 5.0 for m in MANI},
            "cka_win": {m: cka for m in MANI}, "bpb_intr": bpb,
            "sec": {"s4_readout": {"area": area["s4"] + 1, "final3": area["s4"]}, "dim_readout": {"area": area["dim"] + 1, "final3": area["dim"]},
                    **{f"{d}_hard": {"area": area[d] - 1, "final3": area[d] - 2} for d in CMPD}}, "val_loss": vloss, "data_sha": [f"{data}{seed}"] * len(steps),
            "lm0": lm0, "code_sha": "x", "grad": RO.window_grad(on) if on else {"n_on": 0, "n_ok": 0, "n_missing": 0, "missing": [],
            "share": None, "share_min": None, "n_b_gt_a": 0, "n_clip_lt1": 0, "dp_maxdiff": None, "clip_coef": None}, "problems": [],
            "earlier_finished": [], "dir": f"pt6comb-B-{arm}-lr{lr}-s{seed}"}


def plant_sel(R_, arm, prof, seeds=SEL, share=0.9, bpb=None, jitter=0.0, rng=None):
    """prof {lr: mean6-level}: every manifold at that level (+ seed jitter)."""
    for lr, v in prof.items():
        for s in seeds:
            j = (rng.normal(0, jitter) if rng is not None else 0.0)
            R_[arm, lr, s] = fake_rec(arm, lr, s, {m: v + j for m in MANI}, share=share if arm == "all6" else 0.9,
                                      bpb=(bpb or {}).get(lr, 1.2))


def plant_conf(R_, P_, N_, g, h, base=50.0, share=0.9, seeds=CONF, g2=None, dn=None):
    """Confirmation cells: per manifold m the per-seed g_m (all6@P - none@N) and h_m (A2: all6@2e-3 - only-m@2e-3). A2's same-lr pair:
    none@2e-3 = none@N + dn_m (default 0) and all6@2e-3 = none@2e-3 + g2_m (default g_m); a pair cell that IS none@N / all6@P (N or P =
    2e-3) is that cell (g2 / dn ignored)."""
    L2 = SINGLE_LR
    for i, s in enumerate(seeds):
        none = {m: base for m in MANI}
        all6 = {m: base + g[m][i] for m in MANI}
        R_["none", N_, s] = fake_rec("none", N_, s, none)
        sh = share if not isinstance(share, dict) else share.get(s, 0.9)
        R_["all6", P_, s] = fake_rec("all6", P_, s, all6, share=sh)
        none2 = none if N_ == L2 else {m: base + (dn[m][i] if dn else 0.0) for m in MANI}
        all62 = all6 if P_ == L2 else {m: none2[m] + (g2[m][i] if g2 else g[m][i]) for m in MANI}
        if N_ != L2:
            R_["none", L2, s] = fake_rec("none", L2, s, none2)
        if P_ != L2:
            R_["all6", L2, s] = fake_rec("all6", L2, s, all62)
        for m in MANI:
            only = dict(none2)
            only[m] = all62[m] - h[m][i]
            R_[f"only-{m}", L2, s] = fake_rec(f"only-{m}", L2, s, only)


def selftest():
    ok_all, n = [], 0

    def ok(name, cond, val=""):
        ok_all.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val != "" else ""))
    rng = np.random.default_rng(0)
    # ---- statistics = scipy / the inherited functions
    from scipy import stats
    g = rng.normal(1, 2, 8)
    ok("t_one = scipy one-sided one-sample t", abs(t_one(g) - stats.ttest_1samp(g, 0, alternative="greater").pvalue) < 1e-12)
    ok("t_one(-h) = scipy H1 mean < 0", abs(t_one(-g) - stats.ttest_1samp(g, 0, alternative="less").pvalue) < 1e-12)
    ok("Holm = the step-down rule (hand case)", RO.holm([0.01, 0.04, 0.03, 0.2, 0.001, 0.5]) == [0.05, 0.12, 0.12, 0.4, 0.006, 0.5])
    # ---- selection known answers
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    plant_sel(R_, "all6", {"1e-3": 55, "2e-3": 60, "4e-3": 52})
    d = decide(R_, "L1")
    ok("S1 interior picks -> final at L1: C = none@4e-3, all6@2e-3, none@2e-3 (A2; all6@2e-3 merged), six only-m@2e-3", d["status"] == "decided"
       and d["stage_written"] == "C" and d["C"] == [["none", "4e-3"], ["all6", "2e-3"], ["none", "2e-3"]] + [[f"only-{m}", "2e-3"] for m in MANI]
       and d["E"] == [], (d["C"][:3], d["E"]))
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 45, "8e-3": 50})
    plant_sel(R_, "all6", {"1e-3": 50, "2e-3": 55, "4e-3": 60})
    d = decide(R_, "L1")
    ok("S2 argmax at the top edges -> L1x asks all6@8e-3 and none@1.6e-2 (once)", d["stage_written"] == "L1x" and d["E"] == [["all6", "8e-3"], ["none", "1.6e-2"]], d["E"])
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 60, "4e-3": 55, "8e-3": 50})
    plant_sel(R_, "all6", {"1e-3": 60, "2e-3": 55, "4e-3": 52})
    d = decide(R_, "L1")
    ok("S3 none argmax at its lowest, all6 pick 1e-3 -> none@1e-3 asked once (down step = fixB lr), the PRIOR AT ITS LOWEST LR note",
       d["E"] == [["none", "1e-3"]] and any("PRIOR AT ITS LOWEST LR" in x for x in d["notes"]), (d["E"], d["notes"]))
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    plant_sel(R_, "all6", {"1e-3": 60, "2e-3": 55, "4e-3": 50})
    d = decide(R_, "L1")
    ok("S4 all6 pick 1e-3 below none's range (none interior) -> fixB none@1e-3", d["E"] == [["none", "1e-3"]], d["E"])
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    plant_sel(R_, "all6", {"1e-3": 57, "2e-3": 58, "4e-3": 50}, bpb={"1e-3": 1.1, "2e-3": 1.2})
    d = decide(R_, "L1")
    ok("S5 top two within 2 points -> the lower bpb (1e-3) is the pick, argmax 2e-3 -> fixB none@1e-3", d["units"]["all6"]["pick"] == "1e-3"
       and d["units"]["all6"]["argmax"] == "2e-3" and d["E"] == [["none", "1e-3"]], (d["units"]["all6"]["pick"], d["E"]))
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    plant_sel(R_, "all6", {"1e-3": 50, "2e-3": 55, "4e-3": 60})
    for s in SEL:
        R_["all6", "4e-3", s] = fake_rec("all6", "4e-3", s, {m: 60 for m in MANI}, share=0.3)
    d = decide(R_, "L1")
    ok("S6 the gate excludes all6@4e-3 (LM share 0.3): pick 2e-3, its top edge closed by the gate (no 8e-3)", d["units"]["all6"]["pick"] == "2e-3"
       and d["E"] == [] and d["units"]["all6"]["excluded"] == ["4e-3"] and d["stage_written"] == "C", (d["units"]["all6"], d["E"]))
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    plant_sel(R_, "all6", {"1e-3": 50, "2e-3": 55, "4e-3": 60}, share=0.4)
    d = decide(R_, "L1")
    ok("S7 every all6 cell fails the gate -> GATE, no confirmation", d["status"] == "GATE" and d["C"] == [], d["status"])
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    plant_sel(R_, "all6", {"1e-3": 50, "2e-3": 55, "4e-3": 60})
    del R_["none", "4e-3", 2]
    ok("S8 a missing L1 cell -> refused (missing)", decide(R_, "L1")["status"] == "missing")
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 45, "8e-3": 50})
    plant_sel(R_, "all6", {"1e-3": 50, "2e-3": 55, "4e-3": 60})
    d1 = decide(R_, "L1")
    plant_sel(R_, "all6", {"8e-3": 57})
    plant_sel(R_, "none", {"1.6e-2": 48})
    dx = decide(R_, "L1x", d1)
    ok("S9 after L1x: both extensions worse -> FINAL at all6@4e-3, none@8e-3", dx["status"] == "decided" and dx["pick_all6"] == "4e-3"
       and dx["pick_none"] == "8e-3" and dx["stage_written"] == "C", (dx["status"], dx["pick_all6"], dx["pick_none"]))
    plant_sel(R_, "all6", {"8e-3": 70})
    dx = decide(R_, "L1x", d1)
    ok("S10 after L1x: all6@8e-3 best -> NOT FINAL (no third round)", dx["status"] == "NOT FINAL" and dx["C"] == [], dx.get("notes"))
    plant_sel(R_, "all6", {"8e-3": 57})
    plant_sel(R_, "none", {"1.6e-2": 70})
    dx = decide(R_, "L1x", d1)
    ok("S11 after L1x: none@1.6e-2 best -> NOT FINAL", dx["status"] == "NOT FINAL", dx.get("notes"))
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 60, "4e-3": 55, "8e-3": 50})
    plant_sel(R_, "all6", {"1e-3": 50, "2e-3": 60, "4e-3": 55})
    d1 = decide(R_, "L1")
    plant_sel(R_, "none", {"1e-3": 62})
    dx = decide(R_, "L1x", d1)
    ok("S12 none's down step 1e-3 best after L1x -> NOT FINAL (either edge for none)", d1["E"] == [["none", "1e-3"]] and dx["status"] == "NOT FINAL",
       (d1["E"], dx["status"]))
    plant_sel(R_, "none", {"1e-3": 50})
    dx = decide(R_, "L1x", d1)
    ok("S13 none's down step worse -> FINAL at none@2e-3, all6@2e-3", dx["status"] == "decided" and dx["pick_none"] == "2e-3", dx["pick_none"])
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    for lr, lv in {"1e-3": 50.0, "2e-3": 60.0, "4e-3": 50.0}.items():   # numbers alone peaks at 4e-3, the mean of six at 2e-3
        for s in SEL:
            a_ = {m: lv for m in MANI}
            a_["numbers"] = {"1e-3": 40.0, "2e-3": 45.0, "4e-3": 80.0}[lr]
            R_["all6", lr, s] = fake_rec("all6", lr, s, a_)
    d = decide(R_, "L1")
    ok("S14 the objective is the mean of the six areas (numbers alone would pick 4e-3): all6 pick 2e-3", d["units"]["all6"]["pick"] == "2e-3"
       and d["E"] == [], (d["units"]["all6"]["pick"], d["units"]["all6"]["mean"]))
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    plant_sel(R_, "all6", {"1e-3": 50, "2e-3": 60, "4e-3": 59}, bpb={"2e-3": 1.3, "4e-3": 1.2})
    d = decide(R_, "L1")
    ok("S15 the PICK at the top edge by the bpb tie-break (argmax inside) also asks all6@8e-3 (ptdom2 A1.2: pick OR argmax)",
       d["units"]["all6"]["pick"] == "4e-3" and d["units"]["all6"]["argmax"] == "2e-3" and d["E"] == [["all6", "8e-3"]], d["E"])
    # ---- verdict known answers
    pos = [3, 4, 5, 2, 6, 4, 3, 5]
    z8 = [0.0] * 8
    g = {"numbers": [10 + x for x in pos], "s4": [x - 4 for x in pos], "dim": [3, 3, 3, 3, 3, -1, -1, -1],
         "cities": [2 + 0.1 * x for x in pos], "countries": [2 + 0.1 * x for x in pos], "people": [1 + 0.1 * x for x in pos]}
    h = {"numbers": z8, "s4": z8, "dim": z8, "cities": [-4 + 0.1 * x for x in pos], "countries": [-1 + 0.1 * x for x in pos],
         "people": [-2.9 + 6 * (x - 4) for x in pos]}
    R_ = {}
    plant_conf(R_, "2e-3", "4e-3", g, h)
    dec = {"pick_all6": "2e-3", "pick_none": "4e-3"}
    r = verdict(R_, dec)
    V = r["m"]
    ok("V1 numbers +14 8/8 -> HELPS + NO INTERFERENCE", V["numbers"]["helps"] and V["numbers"]["noint"], (V["numbers"]["mean_g"], V["numbers"]["p_help"]))
    ok("V2 s4 ~0 -> NOT SHOWN", not V["s4"]["helps"], (V["s4"]["mean_g"], V["s4"]["p_help"]))
    ok("V3 dim +1.5, 5/8 -> NOT SHOWN by the wins condition (p alone < .05 would not do)", not V["dim"]["helps"] and V["dim"]["wins_g"] == 5,
       (V["dim"]["wins_g"], V["dim"]["p_help"]))
    ok("V4 cities h = -3.6 < -0.5 x u (u = +6) and p(h < 0) < .05 -> INTERFERENCE", not V["cities"]["noint"], (V["cities"]["mean_h"], V["cities"]["margin"], V["cities"]["p_int"]))
    ok("V5 countries h = -0.6 >= -0.5 x u (-1.3) -> NO INTERFERENCE by the margin", V["countries"]["noint"] and V["countries"]["noint_by"].startswith("margin"),
       (V["countries"]["mean_h"], V["countries"]["margin"]))
    ok("V6 people h = -2.9 below the margin (-2.15) but noisy (p >= .05) -> NO INTERFERENCE by p", V["people"]["noint"] and V["people"]["noint_by"] == "p",
       (V["people"]["mean_h"], V["people"]["margin"], V["people"]["p_int"]))
    ok("V7 headline = the HELPS and NO INTERFERENCE ones (numbers, countries, people)", r["headline"]["list"] == ["numbers", "countries", "people"],
       r["headline"]["text"])
    ok("V8 Holm reference column (family of six) = readout_ptdom.holm of the six raw p", [V[m]["p_help_holm"] for m in MANI] == RO.holm([V[m]["p_help"] for m in MANI]))
    ok("V9 mean6 gain = the per-seed mean of the six g", abs(r["mean6_gain"]["mean"] - np.mean([np.mean([g[m][i] for m in MANI]) for i in range(8)])) < 1e-9)
    g2 = dict(g, people=[1.2, 0.05, 1.4, -0.6, 1.0, -0.1, 1.1, 0.4])
    R2 = {}
    plant_conf(R2, "2e-3", "4e-3", g2, h)
    r2 = verdict(R2, dec)
    ok("V10 raw p < .05 with 7/8 -> HELPS while its Holm p >= .05 (printed, not used)", r2["m"]["people"]["helps"] and not r2["m"]["people"]["helps_holm"],
       (r2["m"]["people"]["p_help"], r2["m"]["people"]["p_help_holm"]))
    g3 = dict(g, people=[1, 1.2, 0.9, 1.1, 1.3, 1.0, -0.2, -0.1])
    R3 = {}
    plant_conf(R3, "2e-3", "4e-3", g3, h)
    ok("V11 exactly 6/8 positive with p < .05 -> HELPS", verdict(R3, dec)["m"]["people"]["helps"], verdict(R3, dec)["m"]["people"]["wins_g"])
    g20 = dict(g, people=[1.2, 0.2, 1.1, -0.8, 1.0, 0.1, 0.9, -0.5])   # 6/8 positive, t 1.47 (df 7): one-sided p ~ .09
    R20 = {}
    plant_conf(R20, "2e-3", "4e-3", g20, h)
    v20 = verdict(R20, dec)["m"]["people"]
    ok("V20 6/8 positive but the one-sided p in (0.05, 0.10) -> NOT SHOWN", (not v20["helps"]) and 0.05 < v20["p_help"] < 0.10 and v20["wins_g"] == 6,
       (v20["p_help"], v20["wins_g"]))
    R4 = {}
    plant_conf(R4, "2e-3", "4e-3", g, h, share={s: (0.3 if s >= 12 else 0.9) for s in CONF})
    r4 = verdict(R4, dec)
    ok("V12 all6's LM share 0.3 on seeds 12-15 only -> all6 GATE (no verdict word), the 8-15 median would pass", r4["gate_failed"]
       and r4["headline"]["k"] is None and r4["gate_all6"]["8-11"]["ok"], r4["gate_all6"])
    u_neg = {m: z8 for m in MANI}
    gn = dict(g, s4=[-1.5 + 0.01 * x for x in pos])
    hn = dict(h, s4=[-0.5 + 0.01 * x for x in pos])   # u = g - h = -1.0 (only-s4 below none), h = -0.46 significantly < 0
    R5 = {}
    plant_conf(R5, "2e-3", "4e-3", gn, hn)
    v5 = verdict(R5, dec)["m"]["s4"]
    ok("V13 A2 u < 0 (the single prior below none@2e-3) and h significantly < 0 -> INTERFERENCE by the t clause alone (no margin; the note)",
       not v5["noint"] and v5["neg_u_note"] and v5["margin"] is None, (v5["mean_h"], v5["margin"], v5["p_int"]))
    R6 = dict(R_)
    del R6["only-dim", "2e-3", 11]
    ok("V14 a missing confirmation run -> listed (strict read-out refuses)", verdict(R6, dec)["missing"] == ["only-dim@2e-3 s11"])
    ok("V15 the window split: in + out = the area gain (planted constant series: in = 26/75 of it)",
       abs(V["numbers"]["window_g"]["in"] + V["numbers"]["window_g"]["out"] - V["numbers"]["mean_g"]) < 1e-9
       and abs(V["numbers"]["window_g"]["in"] - V["numbers"]["mean_g"] * 26 / 75) < 1e-9 and V["numbers"]["window_g"]["n_in"] == 26,
       V["numbers"]["window_g"])
    steps = list(range(0, 18310, 250)) + [18309]
    sa = [0.5 + (0.1 if WIN[0] <= s <= WIN[1] else 0) for s in steps]
    w = window_series_split(sa, [0.5] * 75, steps)
    ok("V16 a gain only inside the window lands in the window part", abs(w["in"] - 100 * 0.1 * 26 / 75) < 1e-9 and abs(w["out"]) < 1e-9, w)
    R7 = {}
    plant_conf(R7, "2e-3", "4e-3", {m: [0.5] * 8 for m in MANI}, {m: z8 for m in MANI}, base=3.0)
    for k_ in R7:
        R7[k_]["floor"] = {m: 30.0 for m in MANI}
    r7 = verdict(R7, dec)
    ok("V17 FLOOR when both arms sit below the line; CEILING when none >= 95", r7["m"]["s4"]["tags"] == ["FLOOR"], r7["m"]["s4"]["tags"])
    R8 = {}
    plant_conf(R8, "2e-3", "4e-3", {m: [0.5] * 8 for m in MANI}, {m: z8 for m in MANI}, base=96.0)
    ok("V18 CEILING", verdict(R8, dec)["m"]["s4"]["tags"] == ["CEILING"])
    R9 = copy.deepcopy(R_)
    R9["only-s4", "2e-3", 9]["data_sha"] = ["other"] * 75
    ok("V19 data differing between the arms of a seed -> an integrity problem", any("seed 9: data_sha" in p for p in integrity(R9)), integrity(R9)[:1])
    # ---- the A6 rule generalised (multi-prior) on synthetic run JSONs
    def a6_case(arm, low=None, gp0=False, extra=None):
        steps_ = list(range(0, 18310, 250)) + [18309]
        pri = M6.pri_of(arm, COMP)
        c = []
        for s in steps_:
            e = {"step": s, "prior_on": int(WIN[0] <= s <= WIN[1]), "grad_prior": 0.01, "geo": (0.2 if "numbers" in pri else 0)}
            for d_ in pri:
                if d_ != "numbers":
                    e[f"geo_{d_}"] = 0.1; e[f"cka_{d_}"] = 0.9
            e["cka_ops_helix3"] = 0.8
            if e["prior_on"]:
                for d_ in pri:
                    e[f"gnorm_prior_{d_}"] = 0.01
            c.append(e)
        if low:
            d_, geo_, cka_ = low
            gk, kk = ("geo", "cka_ops_helix3") if d_ == "numbers" else (f"geo_{d_}", f"cka_{d_}")
            c[10][gk], c[10][kk] = geo_, cka_
        if gp0:
            c[10]["grad_prior"] = 0.0
        if extra:
            extra(c)
        r = {"curve": c, "domains": {x: {"prior": x in pri} for x in MANI}, "args": {"prior_domains": ",".join(pri)}}
        ck = {"checks": [{"check": p + " (...)", "ok": True} for p in KEEP6 + ((M6.NUM_ON_CHECK,) if "numbers" in pri else (NUM_OFF,))]
              + [{"check": GEO_FLAG6 + " (before a divergence)", "ok": False}]}
        return geo_a6_multi(r, ck, arm)
    ok("A6m-1 all6: s4 overshoot -0.00115 at cka 1.0001 -> applies (valid, flagged)", a6_case("all6", ("s4", -0.00115, 1.0001))["why"] == [])
    ok("A6m-2 all6: cities overshoot at cka 0.95 -> does not apply", a6_case("all6", ("cities", -0.0012, 0.95))["why"] != [])
    ok("A6m-3 all6: numbers geo -0.0012 at cka_ops_helix3 0.8 -> does not apply (no perfect fit)", a6_case("all6", ("numbers", -0.0012, 0.8))["why"] != [])
    ok("A6m-4 all6: geo -0.006 at cka 1.0 -> below the floor: does not apply", a6_case("all6", ("dim", -0.006, 1.0))["why"] != [])
    ok("A6m-5 all6: grad_prior 0 at an on-eval -> does not apply", a6_case("all6", ("s4", -0.00115, 1.0), gp0=True)["why"] != [])
    ok("A6m-6 only-s4: an overshoot + a geo_dim field (inactive) -> does not apply",
       a6_case("only-s4", ("s4", -0.0011, 1.0), extra=lambda c: c[5].update(geo_dim=0.3))["why"] != [])
    ok("A6m-7 only-s4: an unexplained failure (nothing low) -> does not apply", a6_case("only-s4")["why"] != [])
    ok("A6m-8 only-s4: numbers inactive but geo != 0 -> does not apply", a6_case("only-s4", ("s4", -0.0011, 1.0), extra=lambda c: c[3].update(geo=0.1))["why"] != [])
    # ---- A1 (audit I1-I3): the report's added lines
    dec2 = {"pick_all6": "2e-3", "pick_none": "4e-3", "status": "decided", "after": "L1"}
    rr_ = verdict(R_, dec2)
    txt = report("(selection)", rr_, dec2, [], [])
    ok("V21 the report prints per manifold how 'interference not detected' was met, h's 95% interval, the lr-confounded contrasts (g only: "
       "h / u are same-lr) and the secondary read-outs", "interference not detected by margin" in txt and "h 95% t CI [" in txt
       and "lr-confounded contrasts: g" in txt and "lr-confounded contrasts: g, " not in txt and "NO INTERFERENCE" not in txt.split("descriptive (never")[0].split("HEADLINE")[1]
       and "secondary s4 held_fresh_nonid area: g" in txt and "secondary cities hard-subset final3" in txt and "by p alone" in txt, txt[-600:])
    ok("V22 the secondary g = the planted offsets (s4 held_fresh_nonid area = s4 area + 1 in both arms -> the same g)",
       abs(rr_["m"]["s4"]["secondary"]["g"] - rr_["m"]["s4"]["mean_g"]) < 1e-9 and rr_["m"]["numbers"].get("secondary") is None)
    # ---- A2: the same-lr interference contrast, the merged cells, u <= 0
    ok("C1 c_cells: P 2e-3, N 4e-3 -> 9 cells (all6@2e-3 merged), P 4e-3 N 8e-3 -> 10, P = N = 2e-3 -> 8, P 4e-3 N 2e-3 -> 9; no cell twice",
       [len(c_cells(*pn)) for pn in (("2e-3", "4e-3"), ("4e-3", "8e-3"), ("2e-3", "2e-3"), ("4e-3", "2e-3"))] == [9, 10, 8, 9]
       and all(len(set(c_cells(*pn))) == len(c_cells(*pn)) for pn in (("2e-3", "4e-3"), ("4e-3", "8e-3"), ("2e-3", "2e-3"), ("4e-3", "2e-3")))
       and c_cells("4e-3", "8e-3")[:4] == [("none", "8e-3"), ("all6", "4e-3"), ("none", "2e-3"), ("all6", "2e-3")])
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    plant_sel(R_, "all6", {"1e-3": 55, "2e-3": 52, "4e-3": 60}, bpb={"4e-3": 1.0})
    plant_sel(R_, "all6", {"8e-3": 50})
    d1 = dict(decide(R_, "L1"), E=[["all6", "8e-3"]])
    dx = decide(R_, "L1x", d1)
    ok("C2 decide's C list (P 4e-3, N 4e-3) = none@4e-3, all6@4e-3, none@2e-3, all6@2e-3, six only-m@2e-3", dx["C"] ==
       [["none", "4e-3"], ["all6", "4e-3"], ["none", "2e-3"], ["all6", "2e-3"]] + [[f"only-{m}", "2e-3"] for m in MANI], dx["C"][:4])
    # P 4e-3 / N 8e-3: all6@P far above only-m (h at P +5) but all6@2e-3 well below only-m (h -4, u +4 -> margin -2, p < .05): INTERFERENCE
    zz = {m: [0.0] * 8 for m in MANI}
    gA = {m: [5.0 + 0.1 * x for x in pos] for m in MANI}
    hA = {m: [-4.0 + 0.1 * x for x in pos] for m in MANI}
    g2A = {m: [0.0 + 0.1 * x for x in pos] for m in MANI}   # all6@2e-3 - none@2e-3 ~ +0.4: only-m = all6@2e-3 + 4 -> u ~ +4.4
    RA = {}
    plant_conf(RA, "4e-3", "8e-3", gA, hA, g2=g2A)
    vA = verdict(RA, {"pick_all6": "4e-3", "pick_none": "8e-3"})["m"]["cities"]
    ok("A2-1 h is the SAME-lr contrast: all6@2e-3 - only-m = -3.6 (< -0.5 x u = -2.2, p < .05) -> INTERFERENCE, while all6@P - only-m = +1.4 "
       "(descriptive, lr-confounded)", (not vA["noint"]) and abs(vA["mean_h"] + 3.6) < 1e-9 and vA["h_at_P"]["lr_confounded"]
       and vA["h_at_P"]["mean"] > 0, (vA["mean_h"], vA["margin"], vA["h_at_P"]["mean"]))
    # u is against none@2e-3: none@N low (so u against none@N would be large) but none@2e-3 high -> u small, margin small -> INTERFERENCE
    dnB = {m: [3.0] * 8 for m in MANI}                       # none@2e-3 = none@N + 3
    g2B = {m: [1.0] * 8 for m in MANI}                       # all6@2e-3 = none@2e-3 + 1
    hB = {m: [-1.3 + 0.1 * (x - 4) for x in pos] for m in MANI}   # only-m = all6@2e-3 + 1.3 -> u = +2.3 vs none@2e-3 (+5.3 vs none@N)
    RB = {}
    plant_conf(RB, "4e-3", "8e-3", gA, hB, g2=g2B, dn=dnB)
    vB = verdict(RB, {"pick_all6": "4e-3", "pick_none": "8e-3"})["m"]["dim"]
    ok("A2-2 u is against none@2e-3 (u +2.3, margin -1.15; mean h -1.3, p < .05 -> INTERFERENCE); against none@N it would be u +5.3, "
       "margin -2.65 -> not detected", (not vB["noint"]) and abs(vB["mean_u"] - 2.3) < 1e-9, (vB["mean_u"], vB["margin"], vB["mean_h"], vB["p_int"]))
    ok("A2-3 the same-lr gain all6@2e-3 - none@2e-3 is printed per manifold (planted +1.0 in RB) and differs from g", abs(vB["g_same_lr"]["mean"] - 1.0) < 1e-9
       and abs(vB["mean_g"] - vB["g_same_lr"]["mean"]) > 1, (vB["g_same_lr"], vB["mean_g"]))
    # u <= 0 known answers (A2 decision 2)
    hC = {m: [0.3 * (x - 4) - 0.2 for x in pos] for m in MANI}          # mean h -0.2, noisy
    g2C = {m: [2.0] * 8 for m in MANI}
    RC = {}
    plant_conf(RC, "4e-3", "8e-3", gA, {m: [v_ + 2.5 for v_ in hC[m]] for m in MANI}, g2=g2C)   # only-m = all6@2e-3 - h -> u = 2 - (h + 2.5) ~ -0.3
    vC = verdict(RC, {"pick_all6": "4e-3", "pick_none": "8e-3"})["m"]["people"]
    ok("A2-4 u <= 0 with a noisy h -> interference not detected by the t clause ('p'), the margin is None and the note is set",
       vC["noint"] and vC["noint_by"] == "p" and vC["margin"] is None and vC["neg_u_note"] and vC["mean_u"] <= 0, (vC["mean_u"], vC["mean_h"], vC["p_int"]))
    txtC = report("(selection)", verdict(RC, {"pick_all6": "4e-3", "pick_none": "8e-3"}), {"pick_all6": "4e-3", "pick_none": "8e-3"}, [], [])
    ok("A2-5 the report prints, for u <= 0, 'the single prior shows no gain itself ... the t clause alone decides', the 90% intervals of h and u, "
       "and the same-lr gain row", "the single prior shows no gain itself (mean u <= 0)" in txtC and "h 90% (boot)" in txtC and "u 90% (boot)" in txtC
       and "descriptive: same-lr gain all6@2e-3 - none@2e-3" in txtC and "all6@4e-3 - only-m@2e-3" in txtC and "(lr-confounded)" in txtC)
    vD = manifold_verdict([1.0] * 8, [0.5] * 7 + [0.6], [-1.0] * 7 + [-1.2])
    ok("A2-6 u <= 0 and h >= 0: not detected by 'p' only (never by the margin)", vD["noint"] and vD["noint_by"] == "p" and vD["margin"] is None,
       vD["noint_by"])
    # ---- A3: the gate at all6@2e-3, the pair withheld, the words, the boundaries
    decA = {"pick_all6": "4e-3", "pick_none": "8e-3"}
    R3 = {}
    plant_conf(R3, "4e-3", "8e-3", gA, hA, g2=g2A)
    for s_ in CONF:
        if s_ >= 12:
            R3["all6", "2e-3", s_] = fake_rec("all6", "2e-3", s_, R3["all6", "2e-3", s_]["area"], share=0.3)
    r3 = verdict(R3, decA)
    t3 = report("(selection)", r3, decA, [], [])
    rows3 = [l for l in t3.split("HEADLINE")[0].splitlines() if l.startswith(tuple(f"  {m:10s}" for m in MANI))]   # the verdict table
    ok("A3-1 all6@2e-3 fails the gate on seeds 12-15 (P 4e-3): every manifold's interference word = 'AUX GATE FAILED at all6@2e-3', no "
       "'interference not detected' in the table, HELPS still judged, the headline withholds the interference words",
       r3["gate_same_lr_failed"] and not r3["gate_failed"] and all("AUX GATE FAILED at all6@2e-3" in l for l in rows3) and len(rows3) == 6
       and "interference not detected" not in "\n".join(rows3) and r3["headline"]["k"] is None and "HELPS (all6@4e-3 vs none@8e-3)" in r3["headline"]["text"],
       r3["headline"]["text"])
    R3b = {}
    plant_conf(R3b, "4e-3", "8e-3", gA, hA, g2=g2A)
    for s_ in CONF:
        if s_ >= 12:
            R3b["all6", "4e-3", s_] = fake_rec("all6", "4e-3", s_, R3b["all6", "4e-3", s_]["area"], share=0.3)
    r3b = verdict(R3b, decA)
    ok("A3-2 the HELPS gate checks all6@P: all6@4e-3 failing on seeds 12-15 -> all6 GATE, while all6@2e-3's own gate passes", r3b["gate_failed"]
       and r3b["gate_same_lr_failed"] is False and r3b["headline"]["k"] is None, (r3b["gate_all6"], r3b["gate_all6_same_lr"]))
    R3c = {}
    plant_conf(R3c, "4e-3", "8e-3", gA, hA, g2=g2A)
    full = verdict(R3c, decA)
    del R3c["all6", "2e-3", 10]
    R3c["none", "2e-3", 12] = dict(R3c["none", "2e-3", 12], valid=False)
    r3c = verdict(R3c, decA)
    t3c = report("(selection)", r3c, decA, [], [])
    ok("A3-3 the same-lr pair incomplete (all6@2e-3 s10 missing, none@2e-3 s12 invalid): the strict missing list is empty (HELPS not refused), "
       "missing_pair lists both, every interference word withheld, the HELPS numbers = the complete world's",
       r3c["missing"] == [] and sorted(r3c["missing_pair"]) == ["all6@2e-3 s10", "none@2e-3 s12"] and all(r3c["m"][m]["noint"] is None for m in MANI)
       and all(r3c["m"][m]["p_help"] == full["m"][m]["p_help"] and r3c["m"][m]["helps"] == full["m"][m]["helps"] for m in MANI)
       and "WITHHELD (the same-lr pair is incomplete)" in t3c and r3c["headline"]["k"] is None and r3c["lm_same_lr"] is None, r3c["missing_pair"])
    del R3c["all6", "4e-3", 9]
    ok("A3-4 a missing all6@P run is still a strict refusal (in the missing list)", verdict(R3c, decA)["missing"] == ["all6@4e-3 s9"])
    tfull = report("(selection)", full, decA, [], []) + "\n" + selection_text({}, decide(*(lambda R_: (R_, "L1"))(_s3_world())))
    ok("A3-5 no 'NO INTERFERENCE' anywhere in a full report or a selection note; the table words are 'interference not detected' / "
       "'INTERFERENCE'; HELPS / interference judged-at line; the a6@2e-3 / n@2e-3 levels and the same-lr LM line printed",
       "NO INTERFERENCE" not in tfull and "NO INTERFERENCE" not in txt and "+ interference not detected" in txt.split("HEADLINE")[0]
       and "+ INTERFERENCE" in tfull.split("HEADLINE")[0]
       and "HELPS is judged at all6@4e-3 vs none@8e-3; interference at the same lr 2e-3" in tfull and "a6@2e-3" in tfull and "n@2e-3" in tfull
       and "same-lr LM cost all6@2e-3 - none@2e-3" in tfull and "gate per half (A3: the arm of h)" in tfull)
    ok("A3-6 lr labels: h / u never lr-confounded; all6@P - only-m flagged when P != 2e-3 (4e-3) and not when P = 2e-3",
       all(not full["m"][m]["lr_conf"]["h"] and not full["m"][m]["lr_conf"]["u"] and full["m"][m]["lr_conf"]["h_at_P"]
           and full["m"][m]["h_at_P"]["lr_confounded"] for m in MANI) and not V["numbers"]["lr_conf"]["h_at_P"])
    vb1 = manifold_verdict([1.0] * 8, [-1.0] * 8, [2.0] * 8)
    ok("A3-7 boundary: mean h exactly = -0.5 x mean u (-1.0) with p(h < 0) = 0 -> not detected by the margin ('>=')", vb1["noint"]
       and vb1["noint_by"] == "margin", (vb1["mean_h"], vb1["margin"], vb1["p_int"]))
    vb2 = manifold_verdict([1.0] * 8, [0.1, -0.1] * 4, [0.0] * 8)
    ok("A3-8 boundary: mean u exactly 0 -> the margin clause does not apply (margin None, 'p')", vb2["margin"] is None and vb2["noint_by"] == "p"
       and vb2["neg_u_note"])
    _t1 = globals()["t_one"]
    try:
        globals()["t_one"] = lambda x: 0.05
        vb3 = manifold_verdict([1.0] * 8, [-5.0] * 8, [2.0] * 8)
    finally:
        globals()["t_one"] = _t1
    ok("A3-9 boundary: a one-sided p of exactly 0.05 -> not HELPS ('<'), interference not detected by p ('>=')", (not vb3["helps"]) and vb3["noint"]
       and vb3["noint_by"] == "p", (vb3["helps"], vb3["noint_by"]))
    print(f"SELFTEST {'PASS' if all(ok_all) else 'FAIL'} {sum(ok_all)}/{len(ok_all)}")
    return all(ok_all)


def _s3_world():
    """The S3 selection world (none falling with lr, all6 best at 1e-3): its note 'PRIOR AT ITS LOWEST LR' (A3: no 'NO INTERFERENCE')."""
    R_ = {}
    plant_sel(R_, "none", {"2e-3": 60, "4e-3": 55, "8e-3": 50})
    plant_sel(R_, "all6", {"1e-3": 60, "2e-3": 55, "4e-3": 52})
    return R_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / "out"))
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selection", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.selection:
        R_ = load(a.root)
        dec = decide(R_, "L1")
        print(selection_text(R_, dec) if dec.get("status") != "missing" else f"selection: missing {dec['missing'][:10]}")
        return
    o, txt = readout(a.root, a.allow_partial)
    print(txt)
    if a.json:
        Path(a.json).write_text(json.dumps(o, indent=1, default=str))


if __name__ == "__main__":
    main()
