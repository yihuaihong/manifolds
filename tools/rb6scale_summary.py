"""Model-size ladder of the six-prior rainbow pretraining (rb6, F88 at 53M): the descriptive per-size read-out of the Torch
runs (190M: llmadd_torch/runs/rb6_190m, 505M: runs/rb6_505m) next to the 53M Burst cells, PRE-REGISTERED in
llmadd_torch/runs/rb6_190m/AUDIT.md (and runs/rb6_505m/AUDIT.md). A thin wrapper around tools/rainbow6_summary.py: its
metric definitions (A5 READ), integrity check, strict dim set, s4_hard (A8.3), seed-level gate, paired test and BH are
used as they are; rainbow6_summary's own output is unchanged (this file does not edit it).

Cells per size (roles; every cell is labelled none@<lr> etc. by its actual lr, never "none@best": none's lr was chosen on
the numbers domain only, not tuned for any other domain): none_plr = none at the prior arms' lr (the lr-matched baseline),
none_nlr = none at the lr selected on the numbers area of the pt2 single-prior setting, all6 = the six priors (lambda 1
each), only-numbers = the numbers prior alone; seeds 0-3.
  53m   none@1e-3 (rb6pb-53m-2pct-k2-none), none@8e-3 (rb6mb-...-none-lr8e-3), all6-lam1@1e-3 (rb6pb-...-all6-lam1),
        only-numbers@1e-3 (rb6mb-...-only-numbers): the F88 cells (Burst; identity by rainbow6_summary.run_identity)
  190m  none@6e-4, none@4.8e-3, all6-lam1@6e-4, only-numbers@6e-4 (tags rb6tb-190m-2pct-k2-<arm>-lr<lr>; the pt2s190 lrs)
  505m  none@<all6's pick>, none@<none's pick>, all6-lam1@<all6's pick>, only-numbers@<all6's pick> (--picks: the JSON of
        tools/rb6scale_select.py over the 505M sweep)
Output: per size and cell the A5 READ metric per domain (mean over seeds 0-3, per-seed values in seed order, the seed-level
90% t interval and gate verdict; A5's INVALID rule for the none cells: a domain whose seen final3 < 80 in any seed is
INVALID, and an INVALID baseline never makes a test one-sided), s4_hard area, offline bpb x1e3, per-domain seen, the
gradient split of every prior cell (LM share, prior / LM, clip factor, share of on-evals with prior > LM; A5's FLAG when a
seed's median prior / LM > 1; NOT LOGGED when the run has no decomposition); the sizes side by side (no statistic across
sizes: F87); per size the paired families S1 / S1b / SE / S2 / S3 / SH / SL, every line "on the lr-selection seeds 0-3:
descriptive, no confirmatory claim", every non-numbers line with the caveat that none's lr was not tuned for that domain
(and what the 53M S4 / dim / clock findings compared instead); the caveats also in the JSON. A missing / excluded /
unfinished run -> PROVISIONAL (n=k of 4) on every line that needs it; without --allow_partial the read-out refuses. A
diverged Torch run is data (A6 rule).
Build audit 2026-10-22 (notes/batch1_build/audit_rb6_190m_build.md), applied before any 190M result was read: Q4 INVALID,
Q5 labels / caveats, Q7 the SH / S4 / dim / clock wording, Q8 the gradient report, the 53M SE label (not F88's FE).
Amendment S1 (llmadd_torch/runs/rb6_505m/AUDIT.md, 2026-10-25, before any 505M main result was read), 505M only:
  (i) 505M's lr source in its role line and lr-provenance note (the rb6_505m sweep, seeds 0-1; not pt2);
  (ii) the picks file's grid / amendment / bracket and the X1 label are printed next to the 505M cells, on every paired row
      with the numbers-selected none cell and in the JSON ('x1');
  (iii) a picks file with a 'grid' must carry Amendment X1's grids and a none pick of 2.4e-3 or 4.8e-3;
  (iv) --picks needs --x1 <outcome>: a / b (X1 read-outs) need picks_x1.json (with 'grid'); b-pre / c need picks.json
      (without). Without --picks (53M / 190M) the output, text and JSON, is byte-identical to the pre-S1 version.
  S1.1 (after the build review, still before any 505M main result): --x1_json <picks_x1.json path> is required; the outcome
      is derived from it (absent / not picked -> c; picked -> a / b) and must equal --x1 (b-pre: b); a / b read that very
      file. The three sizes must all be present. The picks file's seeds 0-1 area / bpb at each pick must equal the
      loaded runs. Under b-pre every 505M line carries the SECONDARY label; the none cell at all6-lam1's pick says it
      is not a grid point of none.
    python tools/rb6scale_summary.py <dirs with run_*.json ...> [--picks picks.json --x1 {a,b,b-pre,c} --x1_json
        picks_x1.json] [--allow_partial] [--json out.json]
"""
import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rainbow6_summary as R6   # noqa: E402
import make_rb6m_tasks as G      # noqa: E402

SEEDS = (0, 1, 2, 3)
ROLES = ("none_plr", "none_nlr", "all6", "only_numbers")
ROLE_BASE = {"none_plr": "none", "none_nlr": "none", "all6": "all6-lam1", "only_numbers": "only-numbers"}
ROLE_DESC = {"none_plr": "none at the prior arms' lr (the lr-matched baseline)",
             "none_nlr": "none at the lr selected on the numbers area (pt2 single-prior 2%, seeds 0-3)",
             "all6": "the six priors, lambda 1 each", "only_numbers": "the numbers prior alone, lambda 1"}
NONE_ROLES = ("none_plr", "none_nlr")
DOMS = R6.DOM6   # numbers, months, weekdays, clock, s4, dim
DESIGN53 = {"none_plr": ("none", "1e-3"), "none_nlr": ("none-lr8e-3", "8e-3"), "all6": ("all6-lam1", "1e-3"),
            "only_numbers": ("only-numbers", "1e-3")}   # 53M arm names (as in the F88 tags) and lrs
# the Torch sizes: model facts (the generator's, written again here as a second path) and their lrs per role
SIZE_FACTS = {"190m": dict(dims=(768, 12, 12, 3072), n_params=190297344, layers=[4, 8]),
              "505m": dict(dims=(1024, 24, 16, 4096), n_params=505437184, layers=[8, 16])}
TORCH_SHA = {"code_sha": "93e22bf4c4c0", "toy_struct.py": "d182ce548bcc"}
DESIGN = {"190m": {"none_plr": "6e-4", "none_nlr": "4.8e-3", "all6": "6e-4", "only_numbers": "6e-4"}}
TORCH_ARM = re.compile(r"^(none|all6-lam1|only-numbers)-lr(.+)$")
LABEL = "[on the lr-selection seeds 0-3: descriptive, no confirmatory claim]"
FAMILIES = [   # (name, contrast (role a, role b), rows)
    ("S1", ("all6", "none_plr"), ("numbers", "clock", "s4", "dim")),
    ("S1b", ("all6", "none_nlr"), ("numbers", "clock", "s4", "dim")),
    ("SE", None, (("all6", "none_plr", "months"), ("all6", "none_plr", "weekdays"), ("all6", "none_nlr", "months"),
                  ("all6", "none_nlr", "weekdays"))),
    ("S2", ("only_numbers", "none_plr"), DOMS),
    ("S3", ("all6", "only_numbers"), DOMS),
    ("SH", None, (("all6", "none_plr", "s4_hard"), ("all6", "none_nlr", "s4_hard"))),
    ("SL", None, (("all6", "none_plr", "bpb"), ("only_numbers", "none_plr", "bpb"))),
]
FAM_DESC = {   # the family headers (Q7: SH is A8.3's METRIC, never 'the A8.3 comparison'; Q5 / minor: what each row is)
    "S1": "all6 - none at the prior arms' lr (same lr)",
    "S1b": "all6 - none at the numbers-selected lr (none's lr NOT tuned for clock / s4 / dim)",
    "SE": "tier E months / weekdays vs both none cells, seeds 0-3, descriptive",
    "S2": "only-numbers - none at the prior arms' lr",
    "S3": "all6 - only-numbers (the five domain priors on top of the numbers prior)",
    "SH": "s4_hard area vs none@<lr>, seeds 0-3, descriptive (A8.3's METRIC, NOT the A8.3 comparison: A8.3 = all6@1e-3 vs "
          "the S4-tuned none@1.6e-2 on the fresh seeds 8-15)",
    "SL": "offline bpb x1e3, prior cell - none at the prior arms' lr",
}
SE53 = ("53M SE on seeds 0-3 includes the discovery seeds 0, 1 of none / all6-lam1: NOT F88's FE (decided on the fresh seeds 2-7); "
        "descriptive only")
NOT_TUNED = "none's lr not tuned for {d} (chosen on numbers)"
CAVEATS = {
    "lr_provenance": "the Torch sizes' lrs are reused from the pt2 single-prior 2% setting (chosen on the numbers area over seeds "
                     "0-3); none's lr is NOT tuned for S4, dim, clock, months or weekdays: a none cell is 'none at the numbers-"
                     "selected lr' or 'none at the prior's lr', never 'none at its best lr' for a domain",
    "selection_seeds": "seeds 0-3 share the init and the numbers lines with the lr-selection runs (winner's curse on the numbers "
                       "rows); every paired line is descriptive, no confirmatory claim",
    "size_trend": "no size-trend claim (F87): sizes are printed side by side, never tested against each other",
    "s4_53m": "53M S4 / s4_hard rows here are all6@1e-3 vs none@1e-3 / none@8e-3 on seeds 0-3: NOT F88's final S4 comparison "
              "(A8.3: all6@1e-3 vs the S4-tuned none@1.6e-2 on the fresh seeds 8-15, s4_hard area +26.3); F88 A6: 'S4 not "
              "learnable without the prior' NOT ESTABLISHED",
    "s4_torch": "{size} has no none whose lr was tuned on S4: its S4 / s4_hard rows are not comparable with A8.3 and cannot show "
                "'not learnable without the prior'",
    "dim_53m": "53M dim: none@8e-3 reaches a MEASURABLE strict dim (F88 A6: 'dim not learnable without the prior' does not hold)",
    "dim_torch": "{size} has no none whose lr was tuned on dim",
    "clock_53m": "53M clock: the S1 gain comes from the numbers prior and none@1e-3's too-low lr (F88), not the clock prior",
    "clock_torch": "{size}: none's lr not tuned on clock (at 53M the S1 clock gain was an lr effect)",
    "se_53m": SE53,
}
# ---- Amendment S1 (505M only; nothing below changes the 53M / 190M output)
ROLE_DESC_505 = dict(ROLE_DESC, none_nlr="none at the lr selected on the numbers area of the rb6_505m sweep (seeds 0-1)")
_LP = CAVEATS["lr_provenance"]
assert _LP.startswith("the Torch sizes' lrs are reused from the pt2 single-prior 2% setting")
LR_PROV_190 = "190m: " + _LP.replace("the Torch sizes' lrs are", "the lrs are", 1)
LR_PROV_505 = ("505m: the lrs were selected on THIS batch's sweep (tools/rb6scale_select.py over runs/rb6_505m, seeds 0-1, "
               "numbers area): none over {gn}, all6-lam1 over {ga}; none at the prior arms' lr and only-numbers reuse "
               "all6-lam1's pick; the selection seeds 0-1 are seeds of this read-out (winner's curse on the numbers rows, the "
               "numbers-selected none cell in particular); none's lr is NOT tuned for S4, dim, clock, months or weekdays: never "
               "'none at its best lr' for a domain")
PRE_GRID = {"none": ["4.8e-3", "9.6e-3", "1.92e-2"], "all6-lam1": ["3e-4", "6e-4", "1.2e-3"]}   # the sweep (AUDIT.md Stage 1)
X1_GRID = {"none": ["2.4e-3", "4.8e-3", "9.6e-3", "1.92e-2"], "all6-lam1": ["3e-4", "6e-4", "1.2e-3"]}   # Amendment X1
X1_GRIDTXT = "{2.4e-3, 4.8e-3, 9.6e-3, 1.92e-2}"
X1_STATES = {   # Amendment X1 outcomes and X1.1 (#3, #4): the picks file each reads and its pre-registered label
    "a": dict(grid=True, none="4.8e-3", br="closed",
              label=f"X1: picked over {X1_GRIDTXT} on seeds 0-1; bracket closed by X1"),
    "b": dict(grid=True, none="2.4e-3", br="OPEN at the lower edge",
              label=f"X1: none's lr picked over {X1_GRIDTXT} on seeds 0-1 (Amendment X1); lower edge OPEN; PRIMARY read-out"),
    "b-pre": dict(grid=False, none="4.8e-3", br="OPEN at the lower edge",
                  label="pre-X1 pick (none@4.8e-3): SECONDARY to the X1 read-out (Amendment X1.1 #4), never quoted on its own"),
    "c": dict(grid=False, none="4.8e-3", br="OPEN at the lower edge",
              label="none's bracket OPEN at the lower edge; X1 incomplete"),
}


def x1_check(pk, state):
    """Amendment S1 (iii) / (iv): the 505M picks file against the stated X1 outcome -> the X1 info printed / stored."""
    assert state in X1_STATES, (f"a 505M read-out (--picks) needs --x1 one of {sorted(X1_STATES)} (Amendment S1: the "
                                f"Amendment X1 outcome it is for)", state)
    st = X1_STATES[state]
    has = "grid" in pk
    assert has == st["grid"], (f"--x1 {state} reads " + ("picks_x1.json (an X1 read-out: the file must have the 'grid' of "
                               "rb6scale_select.py --grid_none)" if st["grid"] else "the pre-registered picks.json (no 'grid')")
                               + f"; this file {'has' if has else 'has no'} 'grid'")
    if has:   # (iii) Amendment X1's grids; X1 reading 3: the pick is 2.4e-3 or 4.8e-3, any other pick is a bug
        assert pk["grid"] == X1_GRID, ("the picks file's grid is not Amendment X1's", pk["grid"], X1_GRID)
        assert str(pk.get("amendment", "")).startswith("X1"), ("not an Amendment X1 picks file", pk.get("amendment"))
        assert pk["pick"]["none"] in ("2.4e-3", "4.8e-3"), ("X1 reading 3: none's pick is 2.4e-3 or 4.8e-3", pk["pick"]["none"])
    assert pk["pick"]["none"] == st["none"], (f"--x1 {state} needs none's pick {st['none']}", pk["pick"]["none"])
    assert pk["bracket"]["none"].startswith(st["br"]), (f"--x1 {state}: none's bracket", pk["bracket"]["none"], st["br"])
    assert pk["pick"]["all6-lam1"] == "6e-4" and pk["bracket"]["all6-lam1"] == "closed", \
        ("Amendment X1: all6-lam1 untouched (pick 6e-4, closed)", pk["pick"]["all6-lam1"], pk["bracket"]["all6-lam1"])
    per = {a: {k: pk["table"][a][k][pk["pick"][a]] for k in ("area", "bpb")} for a in ("none", "all6-lam1")}
    return {"state": state, "label": st["label"], "grid": pk.get("grid", PRE_GRID),
            "grid_source": "Amendment X1 (--grid_none)" if has else "the pre-registered sweep grid",
            "amendment": pk.get("amendment", "none (the pre-registered selection)"), "bracket": pk["bracket"],
            "pick": pk["pick"], "per_seed": per}


def x1_derive(path):
    """Amendment S1.1 (build review I2): the X1 outcome from the X1 selection's own output (runs/rb6_505m/picks_x1.json):
    absent or status != 'picked' -> 'c' (X1 incomplete; a nonzero exit of the selection writes no JSON); 'picked' -> 'a'
    (none 4.8e-3) or 'b' (none 2.4e-3), the file checked as an X1 picks file. An existing file must have the X1 'grid'
    (else it is not the X1 selection's output). -> (outcome, sha256[:12] or None, status)."""
    p = Path(path)
    if not p.exists():
        return "c", None, "absent"
    raw = p.read_bytes()
    xj = json.loads(raw)
    assert "grid" in xj, (f"--x1_json {path} has no 'grid': not written by rb6scale_select.py --grid_none (the X1 "
                          "selection)")
    if xj.get("status") != "picked":
        return "c", hashlib.sha256(raw).hexdigest()[:12], str(xj.get("status"))
    st = "b" if xj["pick"]["none"] == "2.4e-3" else "a"
    x1_check(xj, st)
    return st, hashlib.sha256(raw).hexdigest()[:12], "picked"


def x1_bind(cells, x1, pk):
    """Amendment S1.1 (build review I4): the picks file's seeds 0-1 table = the loaded runs at each pick (numbers area and
    in-training bpb, the selection's own definitions): the picks file belongs to these runs. -> None (asserts)."""
    for ro, arm in (("none_nlr", "none"), ("all6", "all6-lam1")):
        lr = pk["pick"][arm]
        for i, s in enumerate((0, 1)):
            x = cells["505m"].get(ro, {}).get(s)
            assert x is not None, (f"505m {arm}@{lr} s{s} (a selection run of the picks file) is not in the read-out")
            got = (value(x, "numbers"), R6.final_bpb(x["r"]))
            want = (pk["table"][arm]["area"][lr][i], pk["table"][arm]["bpb"][lr][i])
            same = [(g == w) or (g is not None and w is not None and math.isfinite(g) and math.isfinite(w) and abs(g - w) < 1e-9)
                    or (g is not None and w is None and not math.isfinite(g)) for g, w in zip(got, want)]
            assert all(same), (f"the picks file's table != the loaded run at 505m {arm}@{lr} s{s} [run, file]", got, want)


def role_desc(sz, ro):
    return ROLE_DESC_505[ro] if sz == "505m" else ROLE_DESC[ro]


def x1_cell_note(x1, sz, ro):
    """S1 (ii): the note next to a 505M cell ('' elsewhere): none's bracket + the X1 label at the numbers-selected none
    cell; all6-lam1's bracket at the cells on all6-lam1's pick (none_plr: never a grid point of none, S1.1 / review m1);
    under b-pre every 505M cell carries the SECONDARY label (S1.1 / review I3)."""
    if sz != "505m" or x1 is None:
        return ""
    if ro == "none_nlr":
        return f" [{x1['label']}; none's bracket: {x1['bracket']['none']}]"
    sec = f"; {x1['label']}" if x1["state"] == "b-pre" else ""
    if ro == "none_plr":
        return (f" [none at all6-lam1's pick: not a grid point of none, does not close none's bracket; all6-lam1's bracket: "
                f"{x1['bracket']['all6-lam1']}{sec}]")
    return f" [at all6-lam1's pick; its bracket: {x1['bracket']['all6-lam1']}{sec}]"


def cell_label(design, sz, ro):
    """A cell's label: its arm at its actual lr (Q5: never 'none@best')."""
    lr = DESIGN53[ro][1] if sz == "53m" else design[sz][ro]
    return f"{ROLE_BASE[ro]}@{lr}"


def dom_caveat(sz, d, b=None):
    """The caveat printed next to a domain's verdict / test line (and stored in the JSON): None for numbers (its lr was
    selected on it; the selection-seed caveat is global) and for a test whose baseline is not a none cell."""
    if d in ("numbers", "bpb") or (b is not None and b not in NONE_ROLES):
        return None
    dd = "s4" if d == "s4_hard" else d
    parts = [NOT_TUNED.format(d=dd)]
    if dd in ("s4", "dim", "clock"):
        parts.append(CAVEATS[f"{dd}_53m" if sz == "53m" else f"{dd}_torch"].format(size=sz))
    return "; ".join(parts)


def fmt(v, w=6, p=1):
    return "-".rjust(w) if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:{w}.{p}f}"


def torch_identity(x, size, base, lr):
    """The run JSON of a Torch scale run against its intent (a second path next to the job's CHECK_PY): every trainer
    argument of make_rb6m_tasks.intent(base) with the size's model dims, lr and tag; the run facts; code / toy shas; the
    lambda per domain, the active / shuffled priors, the logged multiplier; check.json PASS. -> list of problems."""
    r, s = x["r"], x["seed"]
    F = SIZE_FACTS[size]
    I = G.intent(base, s)
    want = dict(I["args"], d_model=F["dims"][0], n_layers=F["dims"][1], n_heads=F["dims"][2], d_ff=F["dims"][3],
                lr=float(lr), tag=f"_{x['tag']}")
    bad = []
    dif = {k: [r["args"].get(k), v] for k, v in want.items() if r["args"].get(k) != v}
    if dif:
        bad.append(f"trainer args differ from the intent [run, intent]: {dif}")
    facts = {"code_sha": TORCH_SHA["code_sha"], "steps": I["steps"], "n_params": F["n_params"], "grad_ckpt_on": True,
             "geo_layers": F["layers"], "read_layers": F["layers"], "lm_chunk": 0, "eval_every_steps": I["eval_every_steps"],
             "lam_rule": I["lam_rule"]}
    fd = {k: [r.get(k), v] for k, v in facts.items() if r.get(k) != v}
    if fd:
        bad.append(f"run facts differ [run, intent]: {fd}")
    if r.get("dep_sha", {}).get("toy_struct.py") != TORCH_SHA["toy_struct.py"]:
        bad.append(f"toy_struct dep_sha {r.get('dep_sha')} != {TORCH_SHA['toy_struct.py']}")
    if r.get("lam_domains") != I["lam_domains"]:
        bad.append(f"lam_domains {r.get('lam_domains')} != {I['lam_domains']}")
    act = sorted(d for d in DOMS if r["domains"][d].get("prior"))
    shf = sorted(d for d in DOMS if r["domains"][d].get("shuf"))
    if act != sorted(I["active"]) or shf != sorted(I["shuf"]):
        bad.append(f"active / shuffled priors {act} / {shf} != {sorted(I['active'])} / {sorted(I['shuf'])}")
    on = [e for e in r["curve"] if e.get("prior_on")]
    div = R6.diverged_at(r)
    for d in I["active"]:
        rat = [e[f"gnorm_prior_{d}"] / e[f"gnorm_true_{d}"] for e in on if (div is None or e["step"] < div)
               and e.get(f"gnorm_prior_{d}") is not None and e.get(f"gnorm_true_{d}")]
        if not rat or any(abs(v / I["lam_domains"][d] - 1) >= 1e-3 for v in rat):
            bad.append(f"logged multiplier of {d} != lambda {I['lam_domains'][d]} (or never logged): "
                       f"{[round(min(rat), 5), round(max(rat), 5)] if rat else None}")
    f = Path(x["path"]).parent.parent / "check.json"
    try:
        cj = json.loads(f.read_text())
    except (OSError, ValueError):
        cj = None
        bad.append("no readable check.json next to the run (the job's intent checks did not run)")
    if cj is not None and not cj.get("pass"):
        bad.append(f"its check FAILED: {[c['check'][:80] for c in cj.get('checks', []) if not c.get('ok')]}")
    return bad


def assign(runs, design):
    """-> cells {size: {role: {seed: x}}}, ignored [labels], torch_keys {(size, role, seed)} (Torch runs)."""
    cells = defaultdict(lambda: defaultdict(dict))
    ignored, info = [], {}
    for x in runs:
        size = R6.TAG.match(x["tag"])["size"]
        x["size"] = size
        # the cell is the TAG's (arm, lr); the run's own arguments are checked against it by the identity checks
        # (rainbow6_summary.run_identity / torch_identity): a mislabelled run is EXCLUDED there, not ignored here
        role, base, lr = None, None, None
        if size == "53m":
            role = next((ro for ro, (a, l_) in DESIGN53.items() if a == x["arm"]), None)
            base, lr = (x["arm"], DESIGN53[role][1]) if role else (None, None)
        else:
            m = TORCH_ARM.match(x["arm"])
            if m and size in design:
                base, lr = m[1], m[2]
                role = next((ro for ro in ROLES if ROLE_BASE[ro] == base and float(design[size][ro]) == float(lr)), None)
        lab = f"{x['tag']} s{x['seed']} ({Path(x['path']).parent.parent.name})"
        if role is None or x["seed"] not in SEEDS or x["k"] != 2:
            ignored.append(lab)
            continue
        assert x["seed"] not in cells[size][role], f"two runs of {size} {role} s{x['seed']}"
        cells[size][role][x["seed"]] = x
        info[id(x)] = (size, role, ROLE_BASE[role] if size != "53m" else x["arm"], lr)
    return cells, ignored, info


def value(x, metric):
    """One run's read-out value: an A5 READ domain, 's4_hard' (A8.3 area), 'bpb' (offline x1e3, None if absent) or 'seen'."""
    if metric == "s4_hard":
        return R6.s4_hard_metrics(x["r"])["area"]
    return R6.mval(x, metric)


def main():
    ap = argparse.ArgumentParser(description="rb6 model-size ladder read-out; see the module docstring")
    ap.add_argument("roots", nargs="+", help="directories with the run JSONs (searched recursively): the Torch out/ dirs and "
                    "the 53M Burst results/<task> dirs")
    ap.add_argument("--picks", default=None, help="505M: the JSON of tools/rb6scale_select.py (status 'picked')")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--x1", default=None, help="505M (with --picks), Amendment S1: the Amendment X1 outcome this read-out is "
                    "for: a / b = picks_x1.json (with 'grid'), b-pre = the secondary picks.json read-out under b, c = X1 "
                    "incomplete (picks.json)")
    ap.add_argument("--x1_json", default=None, help="505M (with --picks), Amendment S1.1: the X1 selection's output path "
                    "(runs/rb6_505m/picks_x1.json; may be absent): the outcome is derived from it and must equal --x1")
    opt = ap.parse_args()
    design = dict(DESIGN)
    x1 = None
    if opt.picks:
        raw = Path(opt.picks).read_bytes()
        pk = json.loads(raw)
        assert pk.get("status") == "picked" and pk["size"] == "505m", ("not a completed 505M selection", pk.get("status"))
        x1 = dict(x1_check(pk, opt.x1), picks_file=str(opt.picks), picks_sha=hashlib.sha256(raw).hexdigest()[:12])
        # S1.1 (review I2): the outcome is not self-reported: derived from the X1 selection's output, = --x1
        assert opt.x1_json, "a 505M read-out needs --x1_json <runs/rb6_505m/picks_x1.json> (Amendment S1.1; may be absent)"
        got, xsha, xstat = x1_derive(opt.x1_json)
        want = "b" if opt.x1 == "b-pre" else opt.x1
        assert got == want, (f"--x1 {opt.x1} but the X1 selection's output {opt.x1_json} gives outcome {got} "
                             f"(status {xstat})")
        if opt.x1 in ("a", "b"):   # the X1 read-out reads the X1 selection's own file
            assert xsha == x1["picks_sha"], ("--picks is not the X1 selection's output --x1_json", x1["picks_sha"], xsha)
        x1.update(x1_json=str(opt.x1_json), x1_json_sha=xsha, x1_json_status=xstat)
        a_, n_ = pk["pick"]["all6-lam1"], pk["pick"]["none"]
        design["505m"] = {"none_plr": a_, "none_nlr": n_, "all6": a_, "only_numbers": a_}
    else:
        assert opt.x1 is None and opt.x1_json is None, "--x1 / --x1_json are the 505M read-out's X1 outcome: they need --picks"
    for sz, d in design.items():
        assert float(d["none_plr"]) != float(d["none_nlr"]), (sz, "the two none cells have one lr: one cell, not two")
    runs = R6.load(opt.roots)
    cells, ignored, info = assign(runs, design)
    sizes = [s for s in ("53m", "190m", "505m") if s in cells]
    if x1 is not None:   # S1.1 (review I1): the 505M read-out prints the three sizes side by side; a missing root is an error
        assert sizes == ["53m", "190m", "505m"], (f"505M read-out: sizes {sizes} != 53m, 190m, 505m (a root missing or "
                                                  "mistyped? the 53M Burst dirs / rb6_190m/out / rb6_505m/out)")
    print(f"rb6 scale summary: sizes {sizes}; {sum(len(v) for c in cells.values() for v in c.values())} runs in the design, "
          f"{len(ignored)} ignored; mode {'PARTIAL (--allow_partial: every verdict PROVISIONAL)' if opt.allow_partial else 'strict'}")
    print("  cells per size: " + "; ".join(f"{s} " + ", ".join(cell_label(design, s, ro) for ro in ROLES) for s in ("53m",) + tuple(design)))
    if x1 is None:
        print("  roles: " + "; ".join(f"{ro} = {ROLE_DESC[ro]}" for ro in ROLES))
    else:   # S1 (i): the 505M role line apart (its lr source is the rb6_505m sweep, not pt2)
        print("  roles (53m, 190m): " + "; ".join(f"{ro} = {ROLE_DESC[ro]}" for ro in ROLES))
        print("  roles (505m): " + "; ".join(f"{ro} = {ROLE_DESC_505[ro]}" for ro in ROLES))
    lp505 = None if x1 is None else LR_PROV_505.format(gn="{" + ", ".join(x1["grid"]["none"]) + "}",
                                                      ga="{" + ", ".join(x1["grid"]["all6-lam1"]) + "}")
    for k in ("lr_provenance", "selection_seeds", "size_trend"):
        if k == "lr_provenance" and x1 is not None:   # S1 (i)
            print(f"  NOTE (pre-registered): {LR_PROV_190}")
            print(f"  NOTE (Amendment S1): {lp505}")
            continue
        print(f"  NOTE (pre-registered): {CAVEATS[k]}")
    if x1 is not None:   # S1 (ii): the picks file's grid / amendment / bracket and the X1 label
        pf = lambda a: (f"{a} {x1['pick'][a]} (seeds 0-1 numbers area [{' '.join(fmt(v, 5) for v in x1['per_seed'][a]['area'])}], "
                        f"in-training bpb [{' '.join(fmt(v, 6, 4) for v in x1['per_seed'][a]['bpb'])}])")
        print(f"  505m picks (Amendment S1): {x1['picks_file']} (sha256[:12] {x1['picks_sha']}); X1 outcome {x1['state']}: "
              f"{x1['label']}")
        print(f"    X1 selection output (--x1_json): {x1['x1_json']} (status {x1['x1_json_status']}, sha256[:12] "
              f"{x1['x1_json_sha']}) -> outcome {'b' if x1['state'] == 'b-pre' else x1['state']}")
        print(f"    grid ({x1['grid_source']}): none {' '.join(x1['grid']['none'])}; all6-lam1 {' '.join(x1['grid']['all6-lam1'])}; "
              f"amendment: {x1['amendment']}")
        print(f"    bracket: none {x1['bracket']['none']}; all6-lam1 {x1['bracket']['all6-lam1']}")
        print(f"    picks: {pf('none')}; {pf('all6-lam1')}")
    for lab in ignored:
        print(f"  IGNORED (not a design cell / seed): {lab}")
    out = {"sizes": sizes, "design": {"53m": {k: v[1] for k, v in DESIGN53.items()}, **design}, "roles": ROLE_DESC,
           "labels": {s: {ro: cell_label(design, s, ro) for ro in ROLES} for s in ("53m",) + tuple(design)},
           "caveats": {k: v.format(size="the Torch sizes") for k, v in CAVEATS.items()}, "ignored": ignored,
           "excluded": [], "warnings": [], "cells": {}, "families": {}, "missing": {}, "grad_split": {}}
    if x1 is not None:   # S1: JSON keys only in a 505M read-out
        out["roles"] = {"53m,190m": ROLE_DESC, "505m": ROLE_DESC_505}   # S1.1 (review m2): no unqualified pt2 roles
        out["caveats"]["lr_provenance"] = LR_PROV_190
        out["caveats"]["lr_provenance_505m"] = lp505
        out["x1"] = x1
    # ---- run identity (53M: rainbow6_summary.run_identity, the F88 read-out's; Torch: torch_identity)
    excluded = []
    if "53m" in cells:
        xs = [x for c in cells["53m"].values() for x in c.values()]
        kept, ident = R6.run_identity(xs, arms=("none", "none-lr8e-3", "all6-lam1", "only-numbers"))
        keep = {id(x) for x in kept}
        for lab, why in ident["excluded"]:
            excluded.append(("53m", lab, why))
        for n_ in ident["not_checked"]:
            print("  NOTE", n_)
        for ro in list(cells["53m"]):
            cells["53m"][ro] = {s: x for s, x in cells["53m"][ro].items() if id(x) in keep}
    for sz in sizes:
        if sz == "53m":
            continue
        for ro in list(cells[sz]):
            for s, x in list(cells[sz][ro].items()):
                _, _, base, lr = info[id(x)]
                bad = torch_identity(x, sz, base, lr)
                if bad:
                    excluded.append((sz, f"{cell_label(design, sz, ro)} s{s} ({Path(x['path']).parent.parent.name})", bad))
                    del cells[sz][ro][s]
    for sz, lab, why in excluded:
        print(f"  EXCLUDED run {sz} {lab}: " + "; ".join(why))
        out["excluded"].append([sz, lab, why])
    # ---- integrity per size (rainbow6_summary.check; a Torch run's non-finite values are data: the A6 rule)
    torch_arms = tuple(sorted({x["arm"] for sz in sizes if sz != "53m" for c in cells[sz].values() for x in c.values()}))
    R6.A6_ARMS = R6.A6_ARMS + tuple(a for a in torch_arms if a not in R6.A6_ARMS)
    for sz in sizes:
        xs = [x for c in cells[sz].values() for x in c.values()]
        if xs:
            w = R6.check(xs, opt.allow_partial)
            for m_ in w:
                print(f"  WARNING {sz} {m_}")
                out["warnings"].append(f"{sz} {m_}")
    unfinished = {id(x) for sz in sizes for c in cells[sz].values() for x in c.values()
                  if x["r"]["curve"][-1]["step"] != x["r"]["steps"] - 1}
    for sz in sizes:   # unfinished = missing (A5.1 rule 1)
        for ro in cells[sz]:
            cells[sz][ro] = {s: x for s, x in cells[sz][ro].items() if id(x) not in unfinished}
    # data identity ACROSS sizes: per seed the same data_sha at every eval (the batches are size-independent)
    by_seed = defaultdict(list)
    for sz in sizes:
        for c in cells[sz].values():
            for s, x in c.items():
                by_seed[s].append((sz, x))
    for s, xs in sorted(by_seed.items()):
        ref = [e["data_sha"] for e in xs[0][1]["r"]["curve"]]
        for sz, x in xs[1:]:
            got = [e["data_sha"] for e in x["r"]["curve"]]
            n = min(len(ref), len(got))
            assert got[:n] == ref[:n], (s, sz, x["tag"], "the data differ across sizes (data_sha series)")
    print(f"  data identity: per seed every run of every size has the same data_sha at every eval ({len(by_seed)} seeds)")
    if x1 is not None:   # S1.1 (review I4): the picks file belongs to these runs
        x1_bind(cells, x1, pk)
        print(f"  505m picks bound: the picks file's seeds 0-1 numbers area / in-training bpb = the loaded runs at "
              f"none@{pk['pick']['none']} and all6-lam1@{pk['pick']['all6-lam1']}")
    # ---- missing seeds
    missing = {sz: {ro: [s for s in SEEDS if s not in cells[sz].get(ro, {})] for ro in ROLES} for sz in sizes}
    out["missing"] = missing
    miss_txt = [f"{sz} {cell_label(design, sz, ro)} {v}" for sz in sizes for ro, v in missing[sz].items() if v]
    if miss_txt and not opt.allow_partial:
        print("  missing seeds: " + "; ".join(miss_txt))
        raise AssertionError("rb6 scale read-out REFUSED (strict): missing / excluded / unfinished seeds " + "; ".join(miss_txt)
                             + " (--allow_partial to look anyway: PROVISIONAL)")
    prov = lambda sz, ros: [ro for ro in ros if missing[sz][ro]]

    # ---- per size: the cells (descriptive)
    METRICS = DOMS + ("s4_hard", "bpb", "seen")
    for sz in sizes:
        print(f"\n== {sz}: per cell, the A5 READ metric per domain (numbers area; months / weekdays tier-E final3; clock wrap; "
              "s4 fresh non-id; dim strict), s4_hard area, offline bpb x1e3, seen (mean of the 5 domains); mean over seeds [per "
              "seed, seed order], the seed-level gate (90% t interval vs [chance + 5, 90]) and, for the none cells, A5's INVALID "
              f"rule (a domain's seen final3 < {R6.SEEN_MIN:.0f} in any seed)")
        for ro in ROLES:
            c = cells[sz].get(ro, {})
            seeds = sorted(c)
            lr = DESIGN53[ro][1] if sz == "53m" else design[sz][ro]
            tag = f"PROVISIONAL (n={len(seeds)} of 4)" if missing[sz][ro] else f"n=4"
            print(f"  {cell_label(design, sz, ro):22s} ({role_desc(sz, ro)}) {tag}{x1_cell_note(x1, sz, ro)}")
            for mt in METRICS:
                vals = [value(c[s], mt) for s in seeds]
                ok = [v for v in vals if v is not None and math.isfinite(v)]
                mean = float(np.mean(ok)) if len(ok) == len(vals) and vals else None
                gate, g, seen_d, cav = "", None, None, None
                if mt in R6.EXTRA and seeds:
                    key = R6.READ[mt][0]
                    g = R6.gate_key([c[s]["r"] for s in seeds], mt, key)
                    seen_d = [R6.metrics(c[s]["r"], f"{mt}_seen")["final3"] for s in seeds]
                    if ro in NONE_ROLES and min(seen_d) < R6.SEEN_MIN:   # A5 (rainbow6_summary main_readout / domain_gate)
                        g["verdict"] = f"INVALID (seen {min(seen_d):.0f} < {R6.SEEN_MIN:.0f})"
                    ci = g["seed_ci"]
                    cav = dom_caveat(sz, mt) if ro in NONE_ROLES else None
                    gate = (f" | seed CI [{fmt(ci[0], 5)},{fmt(ci[1], 5)}] vs floor {g['floor']:.1f} -> {g['verdict']}; "
                            f"heuristic {fmt(g['heuristic'], 4) if g.get('heuristic') is not None else 'n/a'}; seen "
                            f"[{' '.join(fmt(v, 5) for v in seen_d)}]" + (f" [{cav}]" if cav else ""))
                print(f"    {mt:8s} {fmt(mean)} [{' '.join(fmt(v, 5) for v in vals)}]{gate}")
                out["cells"][f"{sz}|{ro}|{mt}"] = {"label": cell_label(design, sz, ro), "lr": lr, "seeds": seeds, "values": vals,
                                                    "mean": mean, "seen_per_seed": seen_d, "caveat": cav,
                                                    "gate": g and {k: g[k] for k in ("verdict", "seed_ci", "floor", "chance")},
                                                    "provisional": bool(missing[sz][ro])}
                if sz == "505m" and x1 is not None:   # S1 (ii)
                    out["cells"][f"{sz}|{ro}|{mt}"]["x1_note"] = x1_cell_note(x1, sz, ro).strip(" []")
            if ro in NONE_ROLES:
                continue
            # Q8 (manifold_aux_loss_principle: every prior cell reports the LM share and the clip): window medians at the
            # on-evals of the logged pre-clip decomposition (grad_lm, grad_prior, grad_cos; T = |g_lm + g_prior|, clip =
            # 1 / max(1, T)); rainbow6_summary.grad_split. The trainer's own pre-clip norm (grad_total / clip_coef) is logged
            # only on its Rev D path: printed when present, else NOT LOGGED.
            rows, flag = [], []
            for s in seeds:
                gsp = R6.grad_split(c[s]["r"])
                own = [e["clip_coef"] for e in c[s]["r"]["curve"] if e.get("prior_on") and e.get("clip_coef") is not None]
                if gsp is None:
                    rows.append(f"s{s} NOT LOGGED (no grad_lm / grad_prior at the on-evals)")
                    out["grad_split"][f"{sz}|{ro}|s{s}"] = "NOT LOGGED"
                    continue
                if gsp["prior_over_lm"] > 1:
                    flag.append(s)
                rows.append(f"s{s} LM share {gsp['lm_share']:.2f} prior/LM {gsp['prior_over_lm']:.2f} clip {gsp['clip']:.2f} "
                            f"prior>LM {gsp['frac_prior_gt_lm']:.2f} (n {gsp['n']})"
                            + (f" trainer clip_coef {float(np.median(own)):.2f}" if own else ""))
                out["grad_split"][f"{sz}|{ro}|s{s}"] = dict(gsp, trainer_clip_coef=float(np.median(own)) if own else "NOT LOGGED")
            print("    gradient split (window medians at the on-evals; clip from the logged decomposition, the trainer's own "
                  "grad_total / clip_coef " + ("logged" if any("trainer clip_coef" in r_ for r_ in rows) else "NOT LOGGED")
                  + "): " + "; ".join(rows))
            print("    " + (f"FLAG: median prior / LM > 1 at seeds {flag} (A5: the prior is auxiliary)" if flag else
                            "no seed with a median prior / LM > 1 (A5 FLAG not raised)"))
        if sz != "53m":
            h = {s: R6.s4_hard(x["r"]) for c in cells[sz].values() for s, x in c.items()}
            print("    s4_hard per seed (n / chance / Latin level, the same for every arm of a seed): "
                  + " ".join(f"s{s} {v['n']}/{v['chance']:.1f}/{v['level']:.1f}" for s, v in sorted(h.items())))

    # ---- sizes side by side (descriptive; no statistic across sizes)
    print("\n== sizes side by side (means over seeds 0-3; no statistic across sizes: F87)")
    print("  " + "cell (lr per size)".ljust(34) + "metric".ljust(9) + "".join(s.rjust(9) for s in sizes))
    for ro in ROLES:
        rl = ROLE_BASE[ro] + "@" + "/".join((DESIGN53[ro][1] if sz == "53m" else design[sz][ro]) for sz in sizes)
        for mt in METRICS:
            row = [out["cells"].get(f"{sz}|{ro}|{mt}", {}).get("mean") for sz in sizes]
            pv = ["*" if missing[sz][ro] else " " for sz in sizes]
            print("  " + rl.ljust(34) + mt.ljust(9) + "".join(f"{fmt(v, 8)}{p}" for v, p in zip(row, pv)))
    print("  (* = PROVISIONAL: fewer than 4 seeds. none's lr is not tuned for any non-numbers domain at any size; the 53M S4 "
          "column is NOT F88's A8.3 comparison)")
    if x1 is not None and "505m" in sizes:   # S1 (ii)
        print(f"  (505m: {cell_label(design, '505m', 'none_nlr')} = {ROLE_DESC_505['none_nlr']}: {x1['label']}; none's bracket "
              f"{x1['bracket']['none']}; the other 505m cells are at all6-lam1's pick, bracket {x1['bracket']['all6-lam1']} "
              f"(the none cell there is not a grid point of none and does not close none's bracket))")

    # ---- paired families per size
    for sz in sizes:
        print(f"\n== {sz}: paired by seed, A5 style (paired t df n-1, exact sign-flip p, 95% CI, BH q .05 within the family) {LABEL}")
        s4_floor, s4_gate = {}, {}
        for ro in NONE_ROLES:   # one-sided only against a FLOOR baseline; INVALID / BORDERLINE / ... -> two-sided (Q4)
            gk = out["cells"].get(f"{sz}|{ro}|s4", {}).get("gate")
            s4_gate[ro] = gk["verdict"] if gk else None
            s4_floor[ro] = bool(gk) and gk["verdict"] == "FLOOR"
        for fam, con, rows in FAMILIES:
            items = [(con[0], con[1], d) for d in rows] if con else list(rows)
            res = []
            for a, b, mt in items:
                ca, cb = cells[sz].get(a, {}), cells[sz].get(b, {})
                seeds = [s for s in SEEDS if s in ca and s in cb]
                va = [value(ca[s], mt) for s in seeds]
                vb = [value(cb[s], mt) for s in seeds]
                side = "greater" if (mt in ("s4", "s4_hard") and b in s4_floor and s4_floor[b]) else "two"
                bad = [s for s, p, q in zip(seeds, va, vb) if p is None or q is None or not (math.isfinite(p) and math.isfinite(q))]
                if bad:
                    res.append(dict(a=a, b=b, m=mt, side=side, seeds=seeds, p_t=None,
                                    note=f"NOT TESTED: no finite value at seeds {bad} (missing offline bpb or DIVERGED)"))
                    continue
                pr = R6.paired([p - q for p, q in zip(va, vb)], side)
                res.append(dict(a=a, b=b, m=mt, side=side, seeds=seeds, **pr))
            for r_ in res:
                r_["label"] = f"{cell_label(design, sz, r_['a'])} - {cell_label(design, sz, r_['b'])}"
                cav = dom_caveat(sz, r_["m"], r_["b"])
                if r_["m"] in ("s4", "s4_hard") and r_["b"] in NONE_ROLES and s4_gate.get(r_["b"]) and r_["side"] == "two":
                    cav = (cav + "; " if cav else "") + f"two-sided: the baseline's s4 gate is {s4_gate[r_['b']]}"
                if fam == "SE" and sz == "53m":
                    cav = (cav + "; " if cav else "") + SE53
                # S1 (ii): every none_nlr row; S1.1 (review I3): under b-pre every 505M row (its BH q values differ too)
                if sz == "505m" and x1 is not None and ("none_nlr" in (r_["a"], r_["b"]) or x1["state"] == "b-pre"):
                    cav = (cav + "; " if cav else "") + x1["label"]
                r_["caveat"] = cav
            adj = R6.bh([r_.get("p_t") for r_ in res])
            for r_, q in zip(res, adj):
                r_["p_bh"] = q
            print(f"  -- {fam}: {FAM_DESC[fam]}" + (f" ({cell_label(design, sz, con[0])} - {cell_label(design, sz, con[1])})" if con else "")
                  + (f" [{SE53}]" if fam == "SE" and sz == "53m" else ""))
            for r_ in res:
                pv = prov(sz, (r_["a"], r_["b"]))
                head = f"    {r_['m']:8s} {r_['label']:34s}"
                if r_.get("note"):
                    print(head + " " + r_["note"] + (f" [{r_['caveat']}]" if r_.get("caveat") else ""))
                    continue
                ci = r_.get("ci") or [None, None]
                flag = " LM COST FLAG (mean > +4)" if r_["m"] == "bpb" and r_["mean"] is not None and r_["mean"] > 4 else ""
                pmark = f" PROVISIONAL (n={r_['n']} of 4)" if pv else ""
                sig = (" sig. at BH .05 (descriptive)" if r_["p_bh"] is not None and r_["p_bh"] < 0.05 else "") if not pv else ""
                print(f"{head} {fmt(r_['mean'], 7)} [{' '.join(fmt(v, 6) for v in r_['diffs'])}] wins {r_['wins']}/{r_['n']} "
                      f"t-p {fmt(r_['p_t'], 5, 3)} ({r_['side']}) sf-p {fmt(r_['p_sf'], 5, 3)} BH {fmt(r_['p_bh'], 5, 3)} "
                      f"95% CI [{fmt(ci[0], 6)},{fmt(ci[1], 6)}]{sig}{flag}{pmark}"
                      + (f" [{r_['caveat']}]" if r_.get("caveat") else ""))
            out["families"][f"{sz}|{fam}"] = res
    status = "PROVISIONAL" if (opt.allow_partial or miss_txt or excluded) else "complete"
    print(f"\n== rb6 scale read-out status: {status}" + (f" (missing: {'; '.join(miss_txt)})" if miss_txt else "")
          + f" {LABEL}")
    out["status"] = status
    if x1 is not None:   # S1 (ii)
        print(f"== 505m X1 outcome {x1['state']} ({x1['picks_file']}): {x1['label']}")
    if opt.json:
        Path(opt.json).write_text(json.dumps(out, indent=1, default=lambda o: None))


if __name__ == "__main__":
    main()
