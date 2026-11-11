"""Amendment A1, known answers for the anchor sub-read-outs of readout_ptdom (anchor_meta, split_info, run_subsets):
  1. site tokens shared between items = the build audit's tokenizer count (lasttok.py: stars 30 / 89 in 12 tokens,
     landmarks 16 / 76, languages 2 / 62, colours 0 / 24, cities 5 / 60); the table sha = the CPU reference of every seed
  2. the re-derived split per seed 0-7 = the CPU references (n_train / n_held); seed 0's heuristic lines = the AUDIT table
     (and exactly the smoke run's logged heuristic_ansfreq / heuristic_shorter / n_hard / masks)
  3. the anchor-conditional baseline (A1) on seeds 0 and 1 = the audit's heur2.py output (held and on the hard subset, and
     the hard-subset sizes)
  4. run_subsets on the REAL smoke run JSONs (only-stars, none; Rev E trainer; 11 evals, 40 held questions): the full and
     hard 2-way series recomputed from the per-question hits = the logged ones at every eval, areas = the metrics, no problem
  5. tampers are reported: a flipped hit, a wrong table sha, a wrong split seed
  6. the per-seed table for AUDIT (seeds 0-7): unique-site questions, the baseline on held / hard
"""
import copy
import glob
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

R = Path("/scratch/yh6210/llmadd_torch/runs/ptdom")
sys.path.insert(0, str(R))
import readout_ptdom as RO   # noqa: E402

ok = []


def T(name, cond, val=None):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val is not None else ""))


refs = {int(k): v for k, v in json.loads((R / "ptdom_refs.json").read_text())["seeds"].items()}
meta = RO.anchor_meta()
WANT_SHARED = {"stars": (30, 89, 12), "landmarks": (16, 76, None), "languages": (2, 62, None), "colours": (0, 24, 0), "cities": (5, 60, 1)}
got = {d: (len(m["shared"]), len(m["items"]), len(m["groups"])) for d, m in meta.items()}
T("site tokens shared with another item = the build audit's count (stars 30/89 in 12 tokens, landmarks 16/76, languages 2/62, "
  "colours 0/24, cities 5/60 in 1 token ' City')", all(got[d][:2] == w[:2] and (w[2] is None or got[d][2] == w[2]) for d, w in WANT_SHARED.items())
  and list(meta["cities"]["groups"]) == [" City"]
  and sorted(meta["cities"]["groups"][" City"]) == sorted(["Ho Chi Minh City", "Mexico City", "New York City", "New Taipei City", "Quezon City"]), got)
for d, m in meta.items():
    print(f"    {d:10s} shared groups: {m['groups']}")
T("table sha (trainer's formula on the snapshot) = the CPU reference of every seed 0-7",
  all(refs[s]["table_sha"][d] == meta[d]["sha"] for s in refs for d in RO.ANC), {d: meta[d]["sha"] for d in RO.ANC})
SI = {(s, d): RO.split_info(meta[d], refs[s]["dom"][d]["split_seed"]) for s in range(8) for d in RO.ANC}
T("re-derived split per seed 0-7 = the references (n_train, n_held)",
  all((len(SI[s, d]["train"]), len(SI[s, d]["held"])) == (refs[s]["dom"][d]["n_train"], refs[s]["dom"][d]["n_held"])
      for s in range(8) for d in RO.ANC))
AUD0 = {"stars": (49.3, 49.8), "landmarks": (61.6, 51.1), "languages": (87.2, 62.2), "colours": (55.6, 50.3), "cities": (67.7, 54.9)}
T("seed 0 heuristic lines (answer frequency / shorter option) = the AUDIT table",
  all((round(100 * SI[0, d]["heur_af"], 1), round(100 * SI[0, d]["heur_sh"], 1)) == AUD0[d] for d in RO.ANC),
  {d: (round(100 * SI[0, d]["heur_af"], 1), round(100 * SI[0, d]["heur_sh"], 1)) for d in RO.ANC})
H2 = {0: {"stars": (50.3, 9.1, 428), "landmarks": (63.0, 13.5, 334), "languages": (88.1, 13.4, 134), "colours": (86.0, 71.0, 434), "cities": (70.3, 14.7, 238)},
      1: {"stars": (49.2, 11.1, 468), "landmarks": (64.1, 11.0, 326), "languages": (86.5, 12.8, 156), "colours": (84.3, 69.4, 546), "cities": (71.6, 14.5, 234)}}


def h3(s, d):
    si = SI[s, d]
    v, hd = np.asarray(si["h3"]), np.asarray(si["masks"]["hard"])
    return round(100 * v.mean(), 1), round(100 * v[hd].mean(), 1), int(hd.sum())


T("anchor-conditional baseline (A1) on seeds 0 / 1: held, hard-subset value and hard-subset size = the audit's heur2.py output "
  "(colours 86.0 / 71.0 on 434 and 84.3 / 69.4 on 546)", all(h3(s, d) == H2[s][d] for s in (0, 1) for d in RO.ANC),
  {s: {d: h3(s, d) for d in RO.ANC} for s in (0, 1)})
# 4. the real smoke runs (seed 0) through the loader
S = R / "tests" / "smoke"
with tempfile.TemporaryDirectory() as tmp:
    recs = {}
    for arm, sub in (("only-stars", "stars"), ("none", "none")):
        tag = f"ptdom-53m-k2-{arm}-lr1e-3"
        dd = Path(tmp) / f"{tag}-s0" / tag
        dd.mkdir(parents=True)
        shutil.copy(glob.glob(str(S / sub / "run_*.json"))[0], dd / f"run_whole_none_seed0_{tag}.json")
        recs[arm] = RO.load(tmp)[(arm, "1e-3", 0)]
    J = json.loads(Path(glob.glob(str(S / "stars" / "run_*.json"))[0]).read_text())
    T("smoke (seed 0, full held set for the heuristics): logged heuristic_ansfreq / heuristic_shorter / n_hard = the re-derived",
      all(abs(J["domains"][d]["heuristic_ansfreq"] - SI[0, d]["heur_af"]) < 1e-12 and abs(J["domains"][d]["heuristic_shorter"] - SI[0, d]["heur_sh"]) < 1e-12
          and J["domains"][d]["n_hard"] == sum(SI[0, d]["masks"]["hard"]) for d in RO.ANC))
    allp, res = [], {}
    for arm, rec in recs.items():
        T(f"loader: {arm} smoke run has the per-question data of the 5 anchor domains (40 questions, 11 evals)",
          sorted(rec["anc"]) == sorted(RO.ANC) and all(len(x["q"]) == 40 and len(x["hits"]) == 11 for x in rec["anc"].values()))
        for d in RO.ANC:
            s_, p_ = RO.run_subsets(rec["anc"][d], meta[d])
            allp += [f"{arm} {d}: {p}" for p in p_]
            res[arm, d] = s_
    T("run_subsets on the real smoke JSONs: no problem (table sha, question order, masks, heuristic lines, the full and hard "
      "series = the logged at every eval, areas = the metrics)", not allp, allp[:5])
    T("recomputed areas = the trainer's <d>_readout / <d>_readout_hard metrics (both arms, 5 domains)",
      all(abs(res[a, d]["full"] - recs[a]["area"][d]) < 1e-9 and abs(res[a, d]["hard"] - recs[a]["hard"][d]) < 1e-9
          for a in recs for d in RO.ANC), {d: [round(res["only-stars", d]["full"], 2), round(res["only-stars", d]["hard"], 2)] for d in RO.ANC})
    T("the baseline is arm-independent (same split): only-stars = none", all(
        (res["only-stars", d]["h3_held"], res["only-stars", d]["n_uniq"]) == (res["none", d]["h3_held"], res["none", d]["n_uniq"]) for d in RO.ANC))
    print("    smoke unique-site subset (40 questions):", {d: (res["only-stars", d]["n_uniq"], round(res["only-stars", d]["uniq"], 1)) for d in RO.ANC})
    # 5. tampers
    x = copy.deepcopy(recs["only-stars"]["anc"]["stars"]); h = list(x["hits"][4]); h[0] = "1" if h[0] == "0" else "0"; x["hits"][4] = "".join(h)
    T("tamper: one flipped hit -> 'recomputed all 2-way series != the logged one'", any("all 2-way series" in p for p in RO.run_subsets(x, meta["stars"])[1]))
    x = copy.deepcopy(recs["only-stars"]["anc"]["colours"]); x["table_sha"] = "000000000000"
    T("tamper: a wrong table sha is reported", any("table sha" in p for p in RO.run_subsets(x, meta["colours"])[1]))
    x = copy.deepcopy(recs["none"]["anc"]["cities"]); x["split_seed"] += 1
    T("tamper: a wrong split seed -> the question order / masks differ", any("held order" in p for p in RO.run_subsets(x, meta["cities"])[1]))
# 6. per-seed table for AUDIT
print("per domain over seeds 0-7: held questions, unique-site held questions (all three items' site tokens unique), hard subset, "
      "the anchor-conditional baseline on held / on hard (mean [min, max])")
for d in RO.ANC:
    nh = [len(SI[s, d]["held"]) for s in range(8)]
    nu = [sum(SI[s, d]["uniq"]) for s in range(8)]
    nd = [sum(SI[s, d]["masks"]["hard"]) for s in range(8)]
    bh = [100 * np.mean(SI[s, d]["h3"]) for s in range(8)]
    bd = [100 * np.asarray(SI[s, d]["h3"])[np.asarray(SI[s, d]["masks"]["hard"])].mean() for s in range(8)]
    af = [100 * SI[s, d]["heur_af"] for s in range(8)]
    print(f"  {d:10s} held {np.mean(nh):.0f}; unique-site {np.mean(nu):.0f} ({100 * np.mean(nu) / np.mean(nh):.0f}%, [{min(nu)}, {max(nu)}]); "
          f"hard {np.mean(nd):.0f}; answer-frequency line {np.mean(af):.1f}; anchor-conditional baseline held {np.mean(bh):.1f} "
          f"[{min(bh):.1f}, {max(bh):.1f}], hard {np.mean(bd):.1f} [{min(bd):.1f}, {max(bd):.1f}]")
print("A1 ANCHOR", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)})")
sys.exit(0 if all(ok) else 1)
