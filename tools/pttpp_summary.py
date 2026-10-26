"""Pre-registered read-out of the fixed tokens-per-parameter batch runs/pttpp (intent spec and rules:
/scratch/yh6210/llmadd_torch/runs/pttpp/AUDIT.md; written and tested before any pttpp result existed). Definitions (area,
t_theta, censored medians, best lr, bracket, diverged, unstable, bootstrap of a ratio) are tools/readout_rules.py, unchanged.
The design (sizes, groups, grids, steps, eval spacing, window events, references) is imported from the generator
(--gen, default /scratch/yh6210/llmadd_torch/make_jobs_pttpp.py); nothing else of it is used.
Cells = (setting, size, tpp) at each arm's pick, seeds 0-7: the new groups (picks = the C decisions of
runs/pttpp/decisions/<group>.json, recomputed here by the same rule and asserted equal), the reused cells (F87 ptlr 53M /
190M / 505M, ptlr_1b 1.07B, pt2s505), and the hardware bridge (505M@3e8 on B200).
Printed for every cell and every seed set (0-3, 4-7, 0-7): Delta area (prior - none) with a 90% paired seed-bootstrap CI,
R_0.5 and R_0.9 (readout_rules.speedup: median t(none) / median t(prior), censored at the budget; '>=' when none's median
is censored) with readout_rules.boot_ci and the censored counts. Slopes (OLS; x = ln N_nonemb for the size question Q1,
ln T for the token question Q2), every metric and seed set, 90% CI from a joint paired seed bootstrap (one resample of the
seed indices for every cell at once: seed s = the same data stream and held-out split in every cell), 10000 resamples,
numpy default_rng(0). Verdicts (AUDIT.md "Pre-registered rules", user 2026-10-26): primary metric 7% ln R_0.5, 2% Delta
area; "shown (direction)" iff the 8-seed slope CI excludes 0, the confirm-only (seeds 4-7) point estimate has the same sign
and the other metric's 8-seed point estimate has the same sign; CI including 0 -> "not shown"; guard or other metric
opposite -> "direction unstable".
    python tools/pttpp_summary.py [--root <runs dir>] [--gen <make_jobs_pttpp.py>]            full read-out
    python tools/pttpp_summary.py --select <group> --json <file> [--root ...]                 a 7% group's lr selection
"""
import argparse
import collections
import glob
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_rules as R   # noqa: E402

GEN_DEFAULT = "/scratch/yh6210/llmadd_torch/make_jobs_pttpp.py"
ROOT_DEFAULT = "/scratch/yh6210/llmadd_torch/runs"
SNAP7, SNAP2 = "a3bce3ab63c1", "93e22bf4c4c0"
ALL8, MAIN, CONF = tuple(range(8)), R.MAIN, R.CONFIRM
SETS = ((MAIN, "0-3"), (CONF, "4-7"), (ALL8, "0-7"))
N_BOOT = R.N_BOOT
FLOOR_AREA = 5.0                  # floor rule: both arms' 8-seed mean area < chance (0) + 5 points
BRIDGE_TOL = {"win-ctx": 2.0, "none": 8.0, "gap": 8.0, "lnR": 0.20}   # AUDIT.md "Hardware bridge"
LM0_TOL = 1e-3


def load_gen(path):
    spec = importlib.util.spec_from_file_location("pttpp_gen", path)
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    return G


def fmt_lr(lr):
    m, e = f"{lr:.2e}".split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def arm_of(a):
    if a["target"] == "none":
        return "none" if a.get("schedule", "whole") == "whole" else None
    if (a["target"], a.get("schedule"), a.get("window"), a.get("site", "ctx")) == ("helix3", "window", "0.05,0.4", "ctx"):
        return "win-ctx"
    return None


# ---------------------------------------------------------------- collect
class Data:
    """every run of the read-out: runs[(gkey, arm, lr, seed)] = rec; rec = dict(file, r, own, check, off, bad, src)"""

    def __init__(self, G, root):
        self.G, self.root, self.Rd = G, Path(root), Path(root) / "pttpp"
        self.runs, self.notes, self.unfinished = {}, [], []
        self.size_of = {v[:4]: k for k, v in G.SIZES.items()}
        own = set()
        of = self.Rd / "OWN_FAILURES.txt"
        if of.exists():
            for line in of.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    t, s = line.split()
                    own.add((t, int(s)))
        self.own = own
        self._new7()
        self._new2()
        self._reused()
        self._integrity()

    def gkey_new(self, setting, size, S, tokens_arg):
        G = self.G
        if setting == "7" and size == "505m" and S == G.STEPS_3E8 and tokens_arg == 3e8:
            return G.BRIDGE
        for gid, c in G.GROUPS.items():
            if c["setting"] == setting and c["size"] == size and G.steps_of(size, c["line"]) == S:
                return gid
        return None

    def _add(self, key, rec):
        assert key not in self.runs, f"two runs for {key}: {rec['file']} and {self.runs[key]['file']}"
        self.runs[key] = rec

    def _new7(self):
        for f in sorted(glob.glob(str(self.Rd / "out" / "*" / "run_*.json"))):
            r = json.loads(Path(f).read_text())
            a = r["args"]
            size = self.size_of.get((a["d_model"], a["n_layers"], a["n_heads"], a["d_ff"]))
            arm = arm_of(a)
            g = self.gkey_new("7", size, r["steps"], a["tokens"]) if size and a["arith_lines"] == 4 else None
            if g is None or arm is None:
                self.notes.append(f"NOT IN THE DESIGN (ignored): {f}")
                continue
            lr, s = fmt_lr(a["lr"]), a["seed"]
            tag = self.G.tag_of(g, arm, lr)
            chk = Path(f).parent / f"check_s{s}.json"
            rec = dict(file=f, r=r, own=False, check=json.loads(chk.read_text()) if chk.exists() else None, off=None,
                       bad=[], src="pttpp", setting="7", size=size, tag=tag)
            if a["tag"] != f"_{tag}":
                rec["bad"].append(f"tag {a['tag']} != the design's _{tag}")
            self._finish(rec, (g, arm, lr, s))

    def _new2(self):
        for f in sorted(glob.glob(str(self.Rd / "out2" / "*" / "*" / "run_*.json"))):
            r = json.loads(Path(f).read_text())
            a = r["args"]
            size = self.size_of.get((a["d_model"], a["n_layers"], a["n_heads"], a["d_ff"]))
            arm = arm_of(a)
            g = self.gkey_new("2", size, r["steps"], a["tokens"]) if size and a["arith_lines"] == 1 else None
            if g is None or arm is None:
                self.notes.append(f"NOT IN THE DESIGN (ignored): {f}")
                continue
            lr, s = fmt_lr(a["lr"]), a["seed"]
            tag = self.G.tag_of(g, arm, lr)
            task = Path(f).parent.parent
            rec = dict(file=f, r=r, own=False, check=json.loads((task / "check.json").read_text()) if (task / "check.json").exists() else None,
                       off=self._offline(task, tag, s, r), bad=[], src="pttpp", setting="2", size=size, tag=tag)
            if a["tag"] != f"_{tag}" or task.name != f"{tag}-s{s}":
                rec["bad"].append(f"tag / task dir {a['tag']} / {task.name} != the design's _{tag} / {tag}-s{s}")
            self._finish(rec, (g, arm, lr, s))

    def _offline(self, task, tag, s, r):
        f = task / "offline_lm.json"
        if not f.exists():
            return None
        o = [v for v in json.loads(f.read_text()).values() if v.get("tag") == f"_{tag}" and v.get("seed") == s and v.get("max_windows") == 0]
        return R.offline_bpb(o[0], R.diverged_at(r) is not None) if o else None

    def _reused(self):
        for ck, arms in self.G.REUSED.items():
            for arm, (src, tag, lr) in arms.items():
                if ck[0] == "7":
                    fs = sorted(glob.glob(str(self.root / src / tag / "run_*.json")))
                else:
                    fs = sorted(glob.glob(str(self.root / src / f"{tag}-s*" / tag / "run_*.json")))
                for f in fs:
                    r = json.loads(Path(f).read_text())
                    s = r["args"]["seed"]
                    task = Path(f).parent.parent
                    rec = dict(file=f, r=r, own=False, check=None, off=None, bad=[], src=src.split("/")[0], setting=ck[0],
                               size=ck[1], tag=tag)
                    if ck[0] == "2":
                        rec["check"] = json.loads((task / "check.json").read_text()) if (task / "check.json").exists() else None
                        rec["off"] = self._offline(task, tag, s, r)
                    if fmt_lr(r["args"]["lr"]) != lr or arm_of(r["args"]) != arm:
                        rec["bad"].append(f"lr / arm {r['args']['lr']} / {arm_of(r['args'])} != {lr} / {arm}")
                    self._finish(rec, ("R" + "|".join(ck), arm, lr, s))

    def _finish(self, rec, key):
        r = rec["r"]
        tag, s = rec["tag"], key[3]
        if not R.finished(r):
            if (tag, s) in self.own:
                rec["own"] = True
                self.notes.append(f"own failure (counts; missing evals scored 0): {rec['file']}")
            else:
                self.unfinished.append((key, rec["file"], r["curve"][-1]["step"] if r["curve"] else None, r["steps"]))
                return
        self._add(key, rec)

    # expected facts of a run of a key
    def expect(self, key):
        G = self.G
        g, arm, lr, s = key
        if g.startswith("R"):
            st, size, _ = g[1:].split("|")
            S, E = G.STEPS_3E8, G.EVAL_3E8
        elif g == G.BRIDGE:
            st, size, S, E = "7", "505m", G.STEPS_3E8, G.EVAL_3E8
        else:
            c = G.GROUPS[g]
            st, size = c["setting"], c["size"]
            S = G.steps_of(size, c["line"])
            E = G.eval_every_of(S)
        return dict(setting=st, size=size, S=S, E=E, evals=G.n_evals(S, E),
                    events=G.window_events(S)[0] if arm != "none" else [], on=G.window_events(S)[1] if arm != "none" else 0.0,
                    code=SNAP2 if st == "2" else SNAP7)

    def _integrity(self):
        G = self.G
        lm0 = collections.defaultdict(dict)
        for key, rec in self.runs.items():
            r, a, c = rec["r"], rec["r"]["args"], rec["r"]["curve"]
            X = self.expect(key)
            bad = rec["bad"]
            if r.get("code_sha") != X["code"]:
                bad.append(f"code_sha {r.get('code_sha')} (want {X['code']})")
            if r["steps"] != X["S"] or r.get("eval_every_steps", G.EVAL_3E8) != X["E"]:
                bad.append(f"steps / eval_every {r['steps']} / {r.get('eval_every_steps')} (want {X['S']} / {X['E']})")
            if not rec["own"] and [e["step"] for e in c] != sorted(set(range(0, X["S"], X["E"])) | {X["S"] - 1}):
                bad.append(f"{len(c)} evals not at the intended steps (want {X['evals']})")
            d, nl, nh, dff, n_params, nne = G.SIZES[X["size"]]
            if (a["d_model"], a["n_layers"], a["n_heads"], a["d_ff"]) != (d, nl, nh, dff):
                bad.append("model dims")
            if r.get("n_params") is not None and (r["n_params"] != n_params or r["n_params"] - G.V * a["d_model"] != nne):
                bad.append(f"n_params {r.get('n_params')} (want {n_params}, N_nonemb {nne})")
            if r.get("grad_ckpt_on") is not None and r["grad_ckpt_on"] != (n_params > 150e6):
                bad.append(f"grad_ckpt_on {r['grad_ckpt_on']}")
            fixed = {"arith_lines": 4 if X["setting"] == "7" else 1, "lam": 5.0, "bs": 32, "seq": 512, "n_eval": 150,
                     "window": "0.05,0.4", "tokenization": "whole", "geo_layers": None}
            wrong = {k: a.get(k) for k, v in fixed.items() if a.get(k, None if k == "geo_layers" else "?") != v}
            if a.get("untie", False) or a.get("site", "ctx") != "ctx":
                wrong["untie/site"] = (a.get("untie"), a.get("site"))
            if wrong:
                bad.append(f"args {wrong}")
            if not rec["own"]:
                if key[1] == "win-ctx":
                    if r.get("prior_events") != X["events"] or abs(c[-1].get("prior_on_frac", -1) - X["on"]) > 1e-9:
                        bad.append(f"prior_events / on_frac {r.get('prior_events')} / {c[-1].get('prior_on_frac')} (want {X['events']})")
                elif c[-1].get("prior_used_frac", 0) != 0 or c[-1].get("prior_on_frac", 0) != 0:
                    bad.append("none run with the prior on")
            if c and not rec["own"] and (R.t_theta(r, 0.5) == 0 or R.held(c[0]) >= 0.5):
                bad.append("the held-out mean is >= 0.5 at step 0 (an untrained model cannot be)")
            if c:
                ref = G.lm0_ref(X["setting"], X["size"], key[3])
                if ref is None or not (math.isfinite(c[0]["lm"]) and abs(c[0]["lm"] - ref) <= LM0_TOL):
                    bad.append(f"step-0 lm {c[0]['lm']} vs the reference {ref}")
                lm0[(X["setting"], X["size"], key[3])][key] = c[0]["lm"]
            if rec["src"] == "pttpp" or X["setting"] == "2":   # every new run and pt2s505 have their job's check
                if not rec["own"] and not (rec["check"] or {}).get("pass"):
                    bad.append("check json missing or FAILED")
            if X["setting"] == "2" and rec["off"] is None and not rec["own"]:
                bad.append("no valid full-val offline_lm entry (pt2main rule 6)")
        for k, d in lm0.items():   # pairing: every cell of a (setting, size, seed) starts from the same init and batch
            med = float(np.median(list(d.values())))
            for key, v in d.items():
                if abs(v - med) > LM0_TOL:
                    self.runs[key]["bad"].append(f"step-0 lm {v:.6f} is {v - med:+.4f} off the {k} median (pairing broken)")


# ---------------------------------------------------------------- per-run quantities
def area_of(D, key):
    rec = D.runs[key]
    if rec["own"]:
        return 100 * sum(R.held(e) for e in rec["r"]["curve"]) / D.expect(key)["evals"]
    return R.area(rec["r"])


def t_of(D, key, th):
    return R.t_theta(D.runs[key]["r"], th)


def bpb_of(D, key):
    rec = D.runs[key]
    if rec["own"]:
        return math.inf, math.inf
    return rec["off"], rec["r"]["curve"][-1]["val_bits_per_byte"]


def budget(D, key):
    return R.budget_M(D.runs[key]["r"])


# ---------------------------------------------------------------- decisions and selection
def decisions(D, gid):
    f = D.Rd / "decisions" / f"{gid}.json"
    log = json.loads(f.read_text())["log"] if f.exists() else []
    return {(e["arm"], e["stage"]): e for e in log}


def select_arm(D, gid, arm):
    """F87 rule 1 (7%: top-two tie, in-training bpb) / pt2main rule 1 (2%: every lr within 2, offline bpb) over the arm's
    grid (stage A + the X lr) on the group's selection seeds -> dict(status, best, open, ...)"""
    G = D.G
    c = G.GROUPS[gid]
    dec = decisions(D, gid)
    lrs = sorted(set(G.GRID[gid][arm]) | ({dec[(arm, "X")]["lr"]} if (arm, "X") in dec else set()), key=float)
    stray = sorted({k[2] for k in D.runs if k[0] == gid and k[1] == arm and k[3] in c["sel"]} - set(lrs), key=float)
    out = dict(gid=gid, arm=arm, grid=lrs, sel=list(c["sel"]), stray=stray)
    miss = [(lr, s) for lr in lrs for s in c["sel"] if (gid, arm, lr, s) not in D.runs]
    if miss:
        return dict(out, status="incomplete", missing=miss)
    reasons = [f"{arm}@{lr} seed {s}: {w}" for lr in lrs for s in c["sel"] for w in D.runs[(gid, arm, lr, s)]["bad"]]
    if reasons:
        return dict(out, status="withheld", reasons=reasons)
    inp = {lr: {"area": [area_of(D, (gid, arm, lr, s)) for s in c["sel"]],
                "off": [bpb_of(D, (gid, arm, lr, s))[0] for s in c["sel"]],
                "intr": [bpb_of(D, (gid, arm, lr, s))[1] for s in c["sel"]]} for lr in lrs}
    if c["setting"] == "7":
        for v in inp.values():
            v["off"] = [None] * len(v["off"])   # F87 rule 1: the 7% runs have no offline bpb; in-training for all
    s_ = R.best_lr(inp, near_rule="top2" if c["setting"] == "7" else "within")
    br = R.bracket(lrs, s_["mean"], s_["best"])
    return dict(out, status="ok", best=s_["best"], near=s_["near"], offline=s_["offline"], mean=s_["mean"], bpb=s_["bpb"],
                open=br[1] if br and br[0] == "open" else None, note=br[1] if br and br[0] == "note" else None,
                unstable=[f"{lr} s{s}" for lr in lrs for s in c["sel"]
                          if R.unstable(D.runs[(gid, arm, lr, s)]["r"], D.runs.get((gid, "none", G.GRID[gid]["none"][0], s), {}).get("r"))],
                diverged=[f"{lr} s{s}@{R.diverged_at(D.runs[(gid, arm, lr, s)]['r'])}" for lr in lrs for s in c["sel"]
                          if R.diverged_at(D.runs[(gid, arm, lr, s)]["r"]) is not None])


def say_select(sel):
    if sel["status"] != "ok":
        why = sel.get("missing") or sel.get("reasons")
        return f"  {sel['arm']}: {sel['status'].upper()} ({len(why)}: {why[0]}{' ...' if len(why) > 1 else ''})"
    g = ", ".join(f"{lr} {sel['mean'][lr]:.1f}" for lr in sel["grid"])
    tb = "" if len(sel["near"]) == 1 else f"; tied {', '.join(sel['near'])} -> lower {'offline' if sel['offline'] else 'in-training*'} bpb"
    return (f"  {sel['arm']}: best {sel['best']} (grid on seeds {sel['sel'][0]}-{sel['sel'][-1]}: {g}{tb})"
            + (f"  OPEN BRACKET at the {sel['open']} edge" if sel["open"] else "")
            + (f"  (edge pick within 2 points at the {sel['note']} edge: not an open bracket)" if sel["note"] else "")
            + (f"  [diverged {', '.join(sel['diverged'])}]" if sel["diverged"] else "")
            + (f"  [unstable {', '.join(sel['unstable'])}]" if sel["unstable"] else "")
            + (f"  [ignored, not in the design: {sel['stray']}]" if sel["stray"] else ""))


# ---------------------------------------------------------------- cells at the picks
def build_cells(D):
    """ckey -> cell dict; ckey = (setting, size, tpp label)"""
    G = D.G
    cells = {}
    for gid, c in G.GROUPS.items():
        dec = decisions(D, gid)
        cell = dict(gid=gid, setting=c["setting"], size=c["size"], tpp=c["line"], hw=c["hw"], src="pttpp " + gid,
                    S=G.steps_of(c["size"], c["line"]), lr={}, keys={}, status="final", why=[],
                    sel_note="lrs picked on seeds 0-1 (B200)" if c["size"] == "1b" else "")
        for arm in G.ARMS:
            sel = select_arm(D, gid, arm)
            cp = dec.get((arm, "C"))
            if (arm, "STOP") in dec:
                cell["status"], _ = "notfinal", cell["why"].append(f"{arm}: STOP ({dec[(arm, 'STOP')].get('why', '')})")
                continue
            if cp is None:
                cell["status"] = "notfinal" if cell["status"] == "final" else cell["status"]
                cell["why"].append(f"{arm}: no C decision yet (selection {sel['status']})")
                continue
            if sel["status"] == "ok" and (sel["best"] != cp["lr"] or sel["open"]):
                cell["status"] = "withheld"
                cell["why"].append(f"{arm}: the C decision {cp['lr']} != the recomputed pick {sel['best']} / open {sel['open']}")
            elif sel["status"] != "ok":
                cell["status"] = "withheld"
                cell["why"].append(f"{arm}: C decided at {cp['lr']} but the selection now reads {sel['status']}")
            cell["lr"][arm] = cp["lr"]
            cell["keys"][arm] = {s: (gid, arm, cp["lr"], s) for s in ALL8}
        cells[(c["setting"], c["size"], c["line"])] = cell
    for ck, arms in G.REUSED.items():
        cells[ck] = dict(gid=None, setting=ck[0], size=ck[1], tpp=ck[2], hw="b200" if ck[1] == "1b" else ("a100" if ck[0] == "2" else "l40s"),
                         src="reused " + ", ".join(f"{a} {t[0]}/{t[1]}" for a, t in arms.items()), S=G.STEPS_3E8, status="final",
                         why=[], lr={a: t[2] for a, t in arms.items()},
                         keys={a: {s: ("R" + "|".join(ck), a, t[2], s) for s in ALL8} for a, t in arms.items()},
                         sel_note="lrs picked on seeds 0-1 (B200)" if ck[1] == "1b" else "")
    for ck, cell in cells.items():
        if cell["status"] != "final":
            continue
        miss = [k for a in cell["keys"] for k in cell["keys"][a].values() if k not in D.runs]
        if miss:
            cell["status"], _ = "incomplete", cell["why"].append(f"{len(miss)} runs missing / unfinished, e.g. {miss[0]}")
            continue
        bad = [f"{k}: {w}" for a in cell["keys"] for k in cell["keys"][a].values() for w in D.runs[k]["bad"]]
        if bad:
            cell["status"], _ = "withheld", cell["why"].extend(bad[:3] + ([f"(+{len(bad) - 3} more)"] if len(bad) > 3 else []))
            continue
        cell["nne"] = G.SIZES[cell["size"]][5]
        cell["T"] = cell["S"] * G.TOK_STEP
        arr = lambda arm, f: np.array([f(D, cell["keys"][arm][s]) for s in ALL8], float)
        cell["an"], cell["ap"] = arr("none", area_of), arr("win-ctx", area_of)
        for th in (0.5, 0.9):
            cell[f"tn{th}"] = arr("none", lambda D_, k: t_of(D_, k, th))
            cell[f"tp{th}"] = arr("win-ctx", lambda D_, k: t_of(D_, k, th))
        cell["bud"] = budget(D, cell["keys"]["none"][0])
        cell["floor"] = (cell["an"].mean() < FLOOR_AREA and cell["ap"].mean() < FLOOR_AREA) or R.censored_median(cell["tp0.5"])
        cell["grad"] = grad_stats(D, cell)
    return cells


def grad_stats(D, cell):
    """LM vs prior gradient (pre-clip norms logged at the on-evals): share by norm a / (a + b), by squared norm, the share
    of on-evals with prior > LM, the first on-eval's prior / LM, and the clip coefficient derived from the logged norms and
    cosine (total = sqrt(a^2 + b^2 + 2ab cos), clip = min(1, 1 / (total + 1e-6)); the trainer does not log it)"""
    sh, sq, gt, first, clip = [], [], [], [], []
    for s in ALL8:
        on = [e for e in D.runs[cell["keys"]["win-ctx"][s]]["r"]["curve"] if e.get("prior_on") and e.get("grad_lm") is not None]
        if not on:
            continue
        a = np.array([e["grad_lm"] for e in on]); b = np.array([e["grad_prior"] for e in on]); cs = np.array([e["grad_cos"] for e in on])
        sh.append(np.median(a / (a + b))); sq.append(np.median(a ** 2 / (a ** 2 + b ** 2))); gt.append(float(np.mean(b > a)))
        first.append(b[0] / a[0])
        tot = np.sqrt(np.maximum(a ** 2 + b ** 2 + 2 * a * b * cs, 0))
        clip.append(np.minimum(1.0, 1.0 / (tot + 1e-6)))
    if not sh:
        return None
    cl = np.concatenate(clip)
    return dict(share=float(np.median(sh)), share_min=float(np.min(sh)), share_sq=float(np.median(sq)), frac_gt=float(np.mean(gt)),
                first=(float(np.min(first)), float(np.max(first))), clip_med=float(np.median(cl)), clip_min=float(np.min(cl)),
                clip_frac=float(np.mean(cl < 1)), n=len(sh))


# ---------------------------------------------------------------- metrics, bootstrap, slopes
def pos_of(seeds):
    return np.array([ALL8.index(s) for s in seeds])


def metric_vals(cell, metric, pos):
    """pos: (..., n) seed positions -> metric value(s); ln R with censored runs at the budget (F87's convention)"""
    if metric == "area":
        return (cell["ap"][pos] - cell["an"][pos]).mean(-1)
    th = 0.5 if metric == "lnR50" else 0.9
    bud = cell["bud"]
    tn = np.where(np.isfinite(cell[f"tn{th}"]), cell[f"tn{th}"], bud)[pos]
    tp = np.where(np.isfinite(cell[f"tp{th}"]), cell[f"tp{th}"], bud)[pos]
    return np.log(np.median(tn, -1) / np.median(tp, -1))


def boot_idx(seeds):
    p = pos_of(seeds)
    return p[np.random.default_rng(0).integers(0, len(p), size=(N_BOOT, len(p)))]


def cell_line(cell, seeds):
    p = pos_of(seeds)
    d = cell["ap"][p] - cell["an"][p]
    bd = metric_vals(cell, "area", boot_idx(seeds))
    out = {"area": (float(d.mean()), float(np.percentile(bd, 5)), float(np.percentile(bd, 95)), int((d > 0).sum()), len(d)),
           "an": float(cell["an"][p].mean()), "ap": float(cell["ap"][p].mean())}
    for th in (0.5, 0.9):
        tn, tp = cell[f"tn{th}"][p], cell[f"tp{th}"][p]
        sp = R.speedup(tn, tp, cell["bud"])
        ci = R.boot_ci(tn, tp, cell["bud"]) if sp["ratio"] is not None else None
        out[f"R{th}"] = (sp, ci)
    return out


def r_status(cell, seeds, th=0.5):
    p = pos_of(seeds)
    cn, cp = R.censored_median(cell[f"tn{th}"][p]), R.censored_median(cell[f"tp{th}"][p])
    return "point" if not (cn or cp) else ">=" if cn and not cp else "<=" if cp and not cn else "none"


def fit_slope(x, Y, groups=None):
    """OLS slope of Y (..., n cells) on x (n cells); groups: fixed effects (slope within groups)"""
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)
    if groups is None:
        xc = x - x.mean()
        return (Y - Y.mean(-1, keepdims=True)) @ xc / (xc @ xc)
    xd, Yd = np.empty_like(x), np.empty_like(Y)
    for g in set(groups):
        m = np.array([gg == g for gg in groups])
        xd[m] = x[m] - x[m].mean()
        Yd[..., m] = Y[..., m] - Y[..., m].mean(-1, keepdims=True)
    return Yd @ xd / (xd @ xd)


def slope(cells, metric, seeds, xkey, groups=None):
    """-> dict(b, lo, hi, n, cells) or None (fewer than 2 cells / no within-group variation)"""
    if len(cells) < 2 or (groups is not None and not any(groups.count(g) > 1 for g in set(groups))):
        return None
    x = [math.log(c[xkey]) for c in cells]
    if groups is not None:
        keep = [i for i, g in enumerate(groups) if groups.count(g) > 1]
        cells, x, groups = [cells[i] for i in keep], [x[i] for i in keep], [groups[i] for i in keep]
    p = pos_of(seeds)
    y = np.array([metric_vals(c, metric, p) for c in cells])
    b = float(fit_slope(x, y, groups))
    idx = boot_idx(seeds)
    Yb = np.stack([metric_vals(c, metric, idx) for c in cells], -1)
    bb = fit_slope(x, Yb, groups)
    if not (np.isfinite(b) and np.all(np.isfinite(bb))):   # e.g. a zero median t (never in real data: S2 checks step 0)
        return dict(b=float("nan"), lo=float("nan"), hi=float("nan"), n=len(cells), cells=[label(c) for c in cells],
                    bad="non-finite values in the slope or its bootstrap")
    return dict(b=b, lo=float(np.percentile(bb, 5)), hi=float(np.percentile(bb, 95)), n=len(cells),
                cells=[label(c) for c in cells])


def label(c):
    return f"{c['size']}@{c['tpp']}"


def sgn(v):
    return 1 if v > 0 else -1 if v < 0 else 0


def verdict(main, guard, other, dirs):
    """the pre-registered rule (AUDIT.md "Verdicts"); main / guard / other: slope dicts or None; dirs: {+1: text, -1: text}"""
    if main is None or main.get("bad"):
        return ("NOT DETERMINABLE (" + (main["bad"] if main else "fewer than 2 informative cells with this metric") + ")",
                "undeterminable")
    guard = None if guard is None or guard.get("bad") else guard
    other = None if other is None or other.get("bad") else other
    shown = main["lo"] > 0 or main["hi"] < 0
    d = sgn(main["b"])
    if not shown:
        extra = []
        if guard is not None and sgn(guard["b"]) != d:
            extra.append("the confirm-only point estimate has the other sign")
        if other is not None and sgn(other["b"]) != d:
            extra.append("the other metric's point estimate has the other sign")
        return (f"NOT SHOWN (8-seed slope {main['b']:+.3f}, 90% CI {main['lo']:+.3f} to {main['hi']:+.3f} includes 0)"
                + (f"; also: {'; '.join(extra)}" if extra else ""), "not shown")
    if guard is None or other is None:
        miss = [n for n, v in (("confirm-only slope", guard), ("other metric's slope", other)) if v is None]
        return (f"NOT FULLY SHOWN: the 8-seed CI excludes 0 ({dirs.get(d, 'zero slope')}), but the {' and the '.join(miss)} cannot be computed, so "
                f"the pre-registered sign checks cannot be made", "unchecked")
    bad = [n for n, v in (("the confirm-only (seeds 4-7) point estimate", guard), ("the other metric's point estimate", other))
           if sgn(v["b"]) != d]
    if bad:
        return f"DIRECTION UNSTABLE: the 8-seed CI excludes 0 ({dirs.get(d, 'zero slope')}) but {' and '.join(bad)} has the other sign", "unstable"
    return f"SHOWN: {dirs.get(d, 'zero slope')}", "shown"


def fmt_slope(s):
    return ("            -" if s is None else "   non-finite" if s.get("bad") else
            f"{s['b']:+7.3f} [{s['lo']:+7.3f}, {s['hi']:+7.3f}]")


# ---------------------------------------------------------------- printing
def fmt_R(sp, ci):
    if sp["ratio"] is None:
        return f"{'no ratio':>8s} {'':17s}"
    return f"{sp['mark'] + format(sp['ratio'], '.2f'):>8s} [{ci['lo']:5.2f}, {ci['hi']:5.2f}]"


def print_cells(cells, out):
    print("\n## cells at the picks: Delta area = prior - none (area points, mean over the seeds; 90% CI paired seed bootstrap; "
          "wins); R_0.5 / R_0.9 = median t(none) / median t(prior) (readout_rules.speedup, censored at the budget; '>=' = none's "
          "median censored, a lower bound) with readout_rules.boot_ci; cens = runs that never reach theta (none/prior)", file=out)
    hdr = (f"  {'cell':24s} {'seeds':5s} | {'Delta area [90% CI]':26s} {'wins':5s} | {'R_0.5 [90% CI]':26s} {'cens':5s} | "
           f"{'R_0.9 [90% CI]':26s} {'cens':5s} | none / prior area")
    print(hdr, file=out)
    order = sorted(cells, key=lambda k: (k[0], float(k[2]) if k[2] != "bridge" else 99, cells[k].get("nne") or 0, k[1]))
    for ck in order:
        c = cells[ck]
        name = f"{c['setting']}% {c['size']} tpp {c['tpp']} ({c['hw']})"
        if c["status"] != "final":
            print(f"  {name:24s} {c['status'].upper()}: {'; '.join(c['why'][:3])}", file=out)
            continue
        for seeds, lab in SETS:
            L = cell_line(c, seeds)
            a = L["area"]
            r5, r9 = L["R0.5"], L["R0.9"]
            print(f"  {name if lab == '0-3' else '':24s} {lab:5s} | {a[0]:+6.1f} [{a[1]:+6.1f}, {a[2]:+6.1f}]{'':5s} {a[3]}/{a[4]:<3d} | "
                  f"{fmt_R(*r5)}{'':0s} {r5[0]['cen_ref']}/{r5[0]['cen_arm']:<3d} | {fmt_R(*r9)} {r9[0]['cen_ref']}/{r9[0]['cen_arm']:<3d} | "
                  f"{L['an']:5.1f} / {L['ap']:5.1f}", file=out)
        G_lines = 4 if c["setting"] == "7" else 1
        probs = c["S"] * 32 * G_lines
        tp = R.median_at_budget(c["tp0.5"], c["bud"])
        print(f"  {'':24s}       lr none {c['lr']['none']} / prior {c['lr']['win-ctx']}; budget {c['bud']:.1f}M tokens = {c['S']} "
              f"steps; N_nonemb {c['nne'] / 1e6:.1f}M; tpp {c['T'] / c['nne']:.4f}; arithmetic lines seen {probs:,}; prior t50 "
              f"{'>=' if R.censored_median(c['tp0.5']) else ''}{tp:.1f}M tokens = {tp * 1e6 / c['T'] * probs:,.0f} lines"
              + (f"; {c['sel_note']}" if c["sel_note"] else "")
              + (f"  ** FLOOR: uninformative (both areas < {FLOOR_AREA:g} or the prior's t50 median censored), out of every slope"
                 if c["floor"] else "") + f"  [{c['src']}]", file=out)


def print_grad(cells, out):
    print("\n## gradient split of every prior cell at its pick (on-evals inside the window, pre-clip norms; clip derived, "
          "the trainer does not log it)", file=out)
    for ck in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
        c = cells[ck]
        if c["status"] != "final" or not c.get("grad"):
            continue
        g = c["grad"]
        print(f"  {c['setting']}% {c['size']} tpp {c['tpp']}: LM share (norm) median {g['share']:.2f} (min over seeds "
              f"{g['share_min']:.2f}), by squared norm {g['share_sq']:.2f}; on-evals with prior > LM {100 * g['frac_gt']:.0f}%; "
              f"first on-eval prior / LM {g['first'][0]:.2f}-{g['first'][1]:.2f}; clip coefficient median {g['clip_med']:.2f} "
              f"(min {g['clip_min']:.2f}; clipped on {100 * g['clip_frac']:.0f}% of on-evals); n={g['n']} seeds"
              + ("  FLAG: the prior's gradient exceeds the LM's (median share < 0.5)" if g["share"] < 0.5 else ""), file=out)


METRICS = (("area", "Delta area"), ("lnR50", "ln R_0.5"), ("lnR90", "ln R_0.9 (descriptive)"))


def unit_slopes(cs, xkey, groups=None):
    """every metric x seed set -> slope dicts; R metrics over the cells whose 8-seed R is a point value"""
    res = {}
    for m, _ in METRICS:
        th = 0.5 if m == "lnR50" else 0.9
        keep = [i for i, c in enumerate(cs) if m == "area" or r_status(c, ALL8, th) == "point"]
        use = [cs[i] for i in keep]
        gr = None if groups is None else [groups[i] for i in keep]
        for seeds, lab in SETS:
            res[(m, lab)] = slope(use, m, seeds, xkey, gr)
        res[(m, "excluded")] = [f"{label(c)} R {r_status(c, ALL8, th)}" for i, c in enumerate(cs) if i not in keep]
    return res


def print_slopes(title, res, out):
    print(f"  {title}", file=out)
    print(f"    {'metric':24s} {'seeds 0-7 (main)':26s} {'seeds 4-7 (guard)':26s} {'seeds 0-3':26s} cells", file=out)
    for m, name in METRICS:
        s8 = res[(m, "0-7")]
        print(f"    {name:24s} {fmt_slope(s8):26s} {fmt_slope(res[(m, '4-7')]):26s} {fmt_slope(res[(m, '0-3')]):26s} "
              + (", ".join(s8["cells"]) if s8 else "-")
              + (f"  [R not a point value, left out: {', '.join(res[(m, 'excluded')])}]" if res.get((m, "excluded")) else ""), file=out)


def decide(res, primary, dirs):
    other = "area" if primary == "lnR50" else "lnR50"
    return verdict(res[(primary, "0-7")], res[(primary, "4-7")], res[(other, "0-7")], dirs)


Q1_DIRS = {1: "the gain is larger at larger sizes (same tpp)", -1: "the gain is smaller at larger sizes (same tpp)"}
Q2_DIRS = {-1: "the gain is larger with fewer tokens (same size)", 1: "the gain is larger with more tokens (same size)"}


def bridge(D, out):
    G = D.G
    print("\n## hardware bridge: 505M at 3e8 tokens, B200 (pttpp) vs L40S (F87 ptlr), F87's picks, seeds 0-3 (tolerances "
          f"pre-registered: win-ctx area {BRIDGE_TOL['win-ctx']:g}, none area {BRIDGE_TOL['none']:g}, gap {BRIDGE_TOL['gap']:g} "
          f"points and the same sign; |ln R_0.5(B200) - ln R_0.5(L40S)| <= {BRIDGE_TOL['lnR']:g}, both point values)", file=out)
    kb = {a: [(G.BRIDGE, a, lr, s) for s in MAIN] for a, lr in G.BRIDGE_CELLS}
    kl = {a: [("R7|505m|0.745", a, lr, s) for s in MAIN] for a, lr in G.BRIDGE_CELLS}
    miss = [k for v in list(kb.values()) + list(kl.values()) for k in v if k not in D.runs]
    if miss:
        print(f"  INCOMPLETE ({len(miss)} runs missing, e.g. {miss[0]}): 1.07B points are PROVISIONAL in Q1", file=out)
        return "incomplete"
    bad = [f"{k}: {w}" for v in list(kb.values()) + list(kl.values()) for k in v for w in D.runs[k]["bad"]]
    if bad:
        print(f"  WITHHELD: {bad[0]}", file=out)
        return "withheld"
    ar = {(hw, a): np.array([area_of(D, k) for k in ks[a]]) for hw, ks in (("B200", kb), ("L40S", kl)) for a in kb}
    gap = {hw: ar[(hw, "win-ctx")] - ar[(hw, "none")] for hw in ("B200", "L40S")}
    lnr, txt = {}, {}
    for hw, ks in (("B200", kb), ("L40S", kl)):
        tn = np.array([t_of(D, k, 0.5) for k in ks["none"]]); tp = np.array([t_of(D, k, 0.5) for k in ks["win-ctx"]])
        sp = R.speedup(tn, tp, budget(D, ks["none"][0]))
        lnr[hw] = math.log(sp["ratio"]) if sp["ratio"] is not None and not sp["mark"] else None
        txt[hw] = sp["text"]
    checks = {"win-ctx area": abs(ar[("B200", "win-ctx")].mean() - ar[("L40S", "win-ctx")].mean()) <= BRIDGE_TOL["win-ctx"],
              "none area": abs(ar[("B200", "none")].mean() - ar[("L40S", "none")].mean()) <= BRIDGE_TOL["none"],
              "gap": sgn(gap["B200"].mean()) == sgn(gap["L40S"].mean()) and abs(gap["B200"].mean() - gap["L40S"].mean()) <= BRIDGE_TOL["gap"],
              "ln R_0.5": None not in lnr.values() and abs(lnr["B200"] - lnr["L40S"]) <= BRIDGE_TOL["lnR"]}
    for a in ("none", "win-ctx"):
        print(f"  {a}: B200 {ar[('B200', a)].mean():5.1f} vs L40S {ar[('L40S', a)].mean():5.1f} (paired diff "
              f"{(ar[('B200', a)] - ar[('L40S', a)]).mean():+.1f}, per seed {', '.join(f'{v:+.1f}' for v in ar[('B200', a)] - ar[('L40S', a)])})", file=out)
    print(f"  gap win-ctx - none: B200 {gap['B200'].mean():+.1f} vs L40S {gap['L40S'].mean():+.1f}; R_0.5 B200 {txt['B200']} vs "
          f"L40S {txt['L40S']}", file=out)
    v = "PASS" if all(checks.values()) else "FAIL"
    print(f"  -> bridge {v} ({', '.join(f'{k} {chr(10003) if ok else chr(10007)}' for k, ok in checks.items())})", file=out)
    return v.lower()


def full(D, out):
    G = D.G
    for n in D.notes:
        print("NOTE", n, file=out)
    for key, f, last, steps in D.unfinished:
        print(f"UNFINISHED {key}: {f} (last eval {last} of {steps})", file=out)
    for key in sorted(D.runs, key=str):
        for w in D.runs[key]["bad"]:
            print(f"INTEGRITY {key}: {w}", file=out)
    print(f"== {len(D.runs)} runs read ({sum(r['src'] == 'pttpp' for r in D.runs.values())} pttpp, the rest reused)", file=out)
    print("\n## selection per group (F87 rule 1 at 7%, pt2main rule 1 at 2%; decisions = runs/pttpp/decisions/<group>.json)", file=out)
    for gid in G.GROUPS:
        dec = decisions(D, gid)
        print(f"  {gid} (selection seeds {G.GROUPS[gid]['sel'][0]}-{G.GROUPS[gid]['sel'][-1]}): decisions "
              + (", ".join(f"{a} {st} {e.get('lr', '')}" for (a, st), e in sorted(dec.items())) or "none"), file=out)
        for arm in G.ARMS:
            print("  " + say_select(select_arm(D, gid, arm)), file=out)
    cells = build_cells(D)
    print_cells(cells, out)
    bv = bridge(D, out)
    print_grad(cells, out)
    fin = [c for c in cells.values() if c["status"] == "final" and not c["floor"]]
    notfin = [label(c) + f" ({c['setting']}%, {c['status']})" for c in cells.values() if c["status"] != "final"]
    prov = f"  PROVISIONAL: not final: {', '.join(notfin)}" if notfin else ""
    print("\n## Q1 size effect at fixed tpp: slope vs ln N_nonemb per tpp line (primary metric: 7% ln R_0.5, 2% Delta area; "
          "verdict = AUDIT.md rule V1-V4)" + ("\n" + prov if prov else ""), file=out)
    calls = {}
    for st, line in (("7", "0.745"), ("7", "0.318"), ("2", "0.745")):
        cs = sorted([c for c in fin if c["setting"] == st and c["tpp"] == line], key=lambda c: c["nne"])
        prim = "lnR50" if st == "7" else "area"
        variants = [("all sizes", cs)]
        if st == "7":
            variants.append(("L40S only (1.07B left out)", [c for c in cs if c["hw"] != "b200"]))
        for vname, vc in variants:
            res = unit_slopes(vc, "nne")
            text, cat = decide(res, prim, Q1_DIRS)
            use = st == "2" or (vname.startswith("all") and bv == "pass") or (vname.startswith("L40S") and bv in ("fail", "withheld"))
            tag = ("PRIMARY" if use else "descriptive") + ("" if st == "2" or bv in ("pass", "fail", "withheld") else " (bridge incomplete)")
            print_slopes(f"{st}% tpp {line}, {vname} [{tag}]", res, out)
            print(f"    -> {st}% tpp {line} ({vname}): {text}", file=out)
            if use:
                calls[(st, line)] = (cat, text, sgn(res[(prim, "0-7")]["b"]) if res[(prim, "0-7")] else 0)
    c7 = [calls.get(("7", ln)) for ln in ("0.745", "0.318")]
    if None in c7:
        q1 = "Q1 7%: waiting for the bridge (no primary variant yet)"
    else:
        cats = [c[0] for c in c7]
        if all(c == "shown" for c in cats) and c7[0][2] == c7[1][2]:
            q1 = f"Q1 7%: SHOWN on both lines: {Q1_DIRS[c7[0][2]]}"
        elif "unstable" in cats or (cats.count("shown") == 2 and c7[0][2] != c7[1][2]):
            q1 = "Q1 7%: DIRECTION UNSTABLE (a line is unstable, or the two lines show opposite directions)"
        elif cats.count("shown") == 1:
            i = cats.index("shown")
            q1 = f"Q1 7%: shown on the tpp {('0.745', '0.318')[i]} line only ({Q1_DIRS[c7[i][2]]}); the other line: {cats[1 - i]}"
        else:
            q1 = f"Q1 7%: NOT SHOWN (lines: {', '.join(cats)})"
    print(f"  => {q1}" + (" [PROVISIONAL]" if notfin else ""), file=out)
    print("\n## Q2 token effect at fixed size: slope vs ln T (total tokens), 7%; PRIMARY = pooled within size (size fixed "
          "effects, every size with >= 2 informative cells; 1.07B's cells are both B200), per size descriptive"
          + ("\n" + prov if prov else ""), file=out)
    cs = sorted([c for c in fin if c["setting"] == "7"], key=lambda c: (c["nne"], c["T"]))
    res = unit_slopes(cs, "T", groups=[c["size"] for c in cs])
    print_slopes("7% pooled within size [PRIMARY]", res, out)
    text, cat = decide(res, "lnR50", Q2_DIRS)
    print(f"    -> Q2 7% pooled: {text}" + (" [PROVISIONAL]" if notfin else ""), file=out)
    for size in G.SIZES:
        sc = [c for c in cs if c["size"] == size]
        if len(sc) >= 2:
            r2 = unit_slopes(sc, "T")
            print_slopes(f"7% {size} ({', '.join('tpp ' + c['tpp'] for c in sc)}) [descriptive]", r2, out)
            print(f"    -> {size}: {decide(r2, 'lnR50', Q2_DIRS)[0]} [descriptive]", file=out)
    print("\n## descriptive regression (7%, every informative cell): gain ~ a + b ln N_nonemb + c ln T; T is also the number "
          "of arithmetic lines seen (4 x 32 x steps), so c reads equally as 'more problems seen'; on a tpp line ln T = ln N + "
          "const, the 3e8 column and the two lines separate b and c", file=out)
    for m, name in METRICS[:2]:
        use = cs if m == "area" else [c for c in cs if r_status(c, ALL8) == "point"]
        if len(use) < 4:
            print(f"  {name}: fewer than 4 cells", file=out)
            continue
        X = np.array([[1.0, math.log(c["nne"]), math.log(c["T"])] for c in use])
        P = np.linalg.pinv(X)
        y = np.array([metric_vals(c, m, pos_of(ALL8)) for c in use])
        Yb = np.stack([metric_vals(c, m, boot_idx(ALL8)) for c in use], -1)
        bt, bb = P @ y, Yb @ P.T
        ci = lambda j: f"{bt[j]:+.3f} [{np.percentile(bb[:, j], 5):+.3f}, {np.percentile(bb[:, j], 95):+.3f}]"
        print(f"  {name}: b (ln N) {ci(1)}, c (ln T) {ci(2)} (90% CI, joint seed bootstrap; {len(use)} cells: "
              f"{', '.join(label(c) for c in use)})", file=out)
    print("\n(no 2% vs 7% comparison is made: AUDIT.md)", file=out)


def main(argv=None, out=None):
    out = out or sys.stdout
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--gen", default=GEN_DEFAULT)
    ap.add_argument("--select")
    ap.add_argument("--json")
    A = ap.parse_args(argv)
    G = load_gen(A.gen)
    D = Data(G, A.root)
    if A.select:
        gid = A.select
        assert gid in G.GROUPS and G.GROUPS[gid]["setting"] == "7", f"--select is for the 7% groups ({gid}); 2%: pt2main_summary --gen"
        sel = {arm: select_arm(D, gid, arm) for arm in G.ARMS}
        print(f"## selection {gid} (F87 rule 1 on seeds {G.GROUPS[gid]['sel']}; runs read {len(D.runs)})", file=out)
        for key, f, last, steps in D.unfinished:
            if key[0] == gid:
                print(f"  unfinished: {f} (last eval {last} of {steps})", file=out)
        for arm in G.ARMS:
            print(say_select(sel[arm]), file=out)
        if A.json:
            Path(A.json).write_text(json.dumps({"gid": gid, "arms": sel}, indent=1, default=str))
        return sel
    full(D, out)


if __name__ == "__main__":
    main()
