"""Design constants of the arithmetic-share dose batch runs/ptshare (Torch L40S, 53M, 2026-10-26; intent spec and
pre-registered rules: report/audits/ptshare_AUDIT.md = llmadd_torch/runs/ptshare/AUDIT.md). One place for the shares, the
lr grids, the one-step expansions and the tags, imported by the generator (llmadd_torch/make_jobs_ptshare.py, from the
read-out snapshot runs/ptshare/readout), the read-out (tools/ptshare_summary.py) and the confirm / expansion trigger
(tools/ptshare_pick.py).

The share is set by the trainer's existing --arith_lines (tools/pretrain_range.py, ptlr snapshot 64abc19 = code_sha
a3bce3ab63c1, no trainer change): every 512-token row is a FineWeb-Edu chunk followed by k arithmetic lines. The 2% of F82 /
F86 is k = 1 and the 7% of F82 / F87 is k = 4: the same mechanism. The dose doubles k: 1, 2, 4, 8 (nominal 2 / 4 / 7 / 14%;
the measured token shares are tools/arith_share.py's, in ptshare_refs.json). The line-length distribution does not depend on
k, so the expected share is proportional to k and ln(share) = ln(k) + const: slopes are per unit ln(share).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINES = (1, 2, 4, 8)                      # --arith_lines of the four shares
LABEL = {1: "2pct", 2: "4pct", 4: "7pct", 8: "14pct"}   # nominal labels (F82 / F87 names for k = 1, 4)
NEW = (1, 2, 8)                           # new cells (runs/ptshare); k = 4 is reused from runs/ptlr (+ ptgw main-W)
REUSED = 4
ARMS = ("none", "win-ctx")
# 3-lr grids centred on the 7% (F87, ptlr 53M) best lrs: none 8e-3, win-ctx 2e-3 (F86 pt2b 2% picked the same two)
GRID = {"none": ("4e-3", "8e-3", "1.6e-2"), "win-ctx": ("1e-3", "2e-3", "4e-3")}
# one factor-2 step beyond an open edge (readout_rules.bracket 'open'), seeds 0-3; once per (k, arm)
EXPAND = {"none": {"lower": "2e-3", "upper": "3.2e-2"}, "win-ctx": {"lower": "5e-4", "upper": "8e-3"}}
MAIN, CONFIRM = (0, 1, 2, 3), (4, 5, 6, 7)
SIZE, HOURS = "53m", "03:00:00"           # ptlr's 53M time limit (runs take 0.96-0.98 L40S-h)
NICE = 2000                                # after the fixed-tokens-per-parameter batch (nice 20), before rb6_190m_dim (5000)
REFS = HERE / "ptshare_refs.json"         # CPU references: step-0 lm, dry rows, measured shares per (k, seed)
RULE5_READING = "A"   # pre-submission amendment 2026-10-26 (coordinator, user-authorised): P = Delta area for the whole slope,
#                       S = log R_0.5 with a censored R used as its bound (direction agreement only)
MAX_JOBS = 120        # 72 main + 6 arms x 4 expansion + 6 arms x 4 confirm (= 118 L40S-h at 0.98 h): the generator never exceeds it
WINDOW_START_M = 916 * 32 * 512 / 1e6     # the window (0.05, 0.4) opens at step 916 = 15.0M tokens: no prior before it
# the reused 7% cells, pinned by source and tag (a later run with the same args under another tag is left out, not merged)
REUSED_TAGS = {("ptlr", "none", lr): f"53m-lr{lr}-7pct-none" for lr in GRID["none"]}
REUSED_TAGS.update({("ptlr", "win-ctx", lr): f"53m-lr{lr}-7pct-win-ctx" for lr in GRID["win-ctx"]})
REUSED_TAGS[("ptgw", "win-ctx", "1e-3")] = "main-W"


def tag(k, arm, lr):
    assert k in LINES and arm in ARMS and lr in grid(k, arm, expanded=True), (k, arm, lr)
    return f"{SIZE}-lr{lr}-{LABEL[k]}-{arm}"


def grid(k, arm, expanded=False):
    """the pre-registered lrs of (k, arm); expanded=True adds the two possible one-step expansions (k != 4 only)"""
    g = GRID[arm]
    if expanded and k != REUSED:
        g = (EXPAND[arm]["lower"],) + g + (EXPAND[arm]["upper"],)
    return g


def share_label(k, shares=None):
    """the measured share of k lines as printed in every read-out output: the mean over seeds 0-7 of the CPU replay
    (ptshare_refs.json), e.g. '2.12%'; shares = {k: {seed: share}} (default: this file's references); the nominal LABEL
    (a name only, used in tags) when no reference exists"""
    if shares is None:
        r = refs()
        shares = {kk: {s: v["share"] for s, v in d.items()} for kk, d in r["share"].items()} if r else {}
    if k not in shares or not shares[k]:
        return LABEL[k]
    return f"{100 * sum(shares[k].values()) / len(shares[k]):.2f}%"


def refs():
    """{"lm0": {k: {seed: lm}}, "share": {k: ...}, "dry": {...}} with int keys (None if the file does not exist yet)"""
    if not REFS.exists():
        return None
    r = json.loads(REFS.read_text())
    out = {sec: {int(k): {int(s): v for s, v in d.items()} for k, d in r[sec].items()} for sec in ("lm0", "share", "dry")}
    out["meta"] = r.get("meta", {})
    return out
