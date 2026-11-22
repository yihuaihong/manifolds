#!/usr/bin/env python
"""pt6v2 Amendment V6 (post-hoc interpretability, exploratory, 2026-11-22; exp-builder): driver of the per-checkpoint
read-out probe_v6_lib/probe6_inner.py (A1 shared answer code, A2 answer-position subspace ablation, A3 logit lens + DLA,
A4 weights). No training; CPU. One checkpoint per process; the 12 V5 checkpoints = 4 runs x steps (7323, 12000, 18309).

  python -B runs/pt6v2/probe_v6.py --index I            # I in 0..11: (run, step) of TARGETS (the cpu_short array)
  python -B runs/pt6v2/probe_v6.py --label only-dim_s16 --step 18309 [--smoke]
  python -B runs/pt6v2/probe_v6.py --smoke_c            # code check on the 3-layer smoke config (out_v5/smoke/C)
Output: runs/pt6v2/out_v6/<label>_step<step>[_smoke]/{result.json, report.txt, inner.log, states_ans.npz}
--smoke cuts only the null draws (n_rand 4, n_shuf 3, n_null 50) and says so in the result; rows and analyses are the full ones.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
LIB = R / "probe_v6_lib"
TOOLS = R / "out_v5" / "code" / "tools"
B = Path("/scratch/yh6210/llmadd_torch")
OUT = R / "out_v6"
SIX = {"numbers", "dim", "cities", "countries", "torus", "box5"}
NOT_ARGS = {"out", "probe_wts", "probe6_out", "probe6_cfg"}
STEPS = (7323, 12000, 18309)
RUNS = [(f"{a}_s{s}", R / "out_v5" / f"pt6v2-{a}-lr2e-3-s{s}" / f"pt6v2-{a}-lr2e-3") for s in (16, 17) for a in ("all6", "only-dim")]
TARGETS = [(lab, d, st) for lab, d in RUNS for st in STEPS]
SMOKE_NULLS = dict(n_rand=4, n_shuf=3, n_null=50, n_dose=2)
FULL_NULLS = dict(n_rand=20, n_shuf=10, n_null=200, n_dose=5)


def argv_from(a: dict) -> list:   # = probe_v5.argv_from
    out = []
    for k, v in a.items():
        if k in NOT_ARGS or v is None:
            continue
        if isinstance(v, bool):
            out += [f"--{k}"] if v else []
        else:
            out += [f"--{k}", str(v)]
    return out


def run_files(rdir: Path, step: int):
    rj = sorted(rdir.glob("run_*.json"))
    assert len(rj) == 1, (rdir, rj)
    w = sorted(rdir.glob(f"wts_*_step{step}.pt"))
    assert len(w) == 1, (rdir, step, w)
    return rj[0], w[0]


def arm_of(run_json: dict) -> str:
    pd = run_json["args"].get("prior_domains") or ""
    s = set(pd.split(",")) if pd else set()
    return "all6" if s == SIX else "only-dim" if s == {"dim"} else f"other({pd})"


def run_one(label: str, rdir: Path, step: int, cfg_over: dict, tag: str = "") -> Path:
    sha = subprocess.run([sys.executable, "-B", str(LIB / "make_probe_copy6.py")] + (["--allow_sha"] if tag == "_smokeC" else []),
                         check=True, capture_output=True, text=True).stdout.strip()
    print(sha, flush=True)
    rjp, wts = run_files(rdir, step)
    rj = json.loads(rjp.read_text())
    od = OUT / f"{label}_step{step}{tag}"
    od.mkdir(parents=True, exist_ok=True)
    v5m = R / "out_v5" / "probe_v5" / f"dump_{label}" / "meta.json"
    cfg = dict(FULL_NULLS, run_json=str(rjp), step=step, label=label, arm=arm_of(rj),
               v5_meta=str(v5m) if v5m.exists() else "") | cfg_over
    cmd = [sys.executable, "-B", str(LIB / "pretrain_range_probe6.py")] + argv_from(rj["args"]) + \
          ["--out", str(od / "trainer_out"), "--probe_wts", str(wts), "--probe6_out", str(od), "--probe6_cfg", json.dumps(cfg)]
    env = dict(os.environ, HF_HOME=str(B / "hf_cache"), PYTHONPATH=str(B / "pylib"), PYTHONDONTWRITEBYTECODE="1")
    print(f"probe_v6: {label} step {step} ({cfg['arm']}) -> {od}; torch threads {os.environ.get('OMP_NUM_THREADS', 'default')}", flush=True)
    t0 = time.time()
    with open(od / "inner.log", "w") as f:
        rc = subprocess.run(cmd, cwd=TOOLS, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    wall = round(time.time() - t0, 1)
    if rc:
        print(open(od / "inner.log").read()[-5000:])
        sys.exit(f"probe_v6 failed rc={rc}: {od / 'inner.log'}")
    res = json.loads((od / "result.json").read_text())
    res["wall_s"] = wall; res["label"] = label; res["arm"] = cfg["arm"]; res["smoke_nulls"] = cfg_over.get("n_rand") == SMOKE_NULLS["n_rand"]
    (od / "result.json").write_text(json.dumps(res, indent=1))
    txt = report(res)
    (od / "report.txt").write_text(txt)
    print(txt, flush=True)
    return od


def f3(x):
    return "  nan" if x is None or x != x else f"{x:.3f}"


def f1(x):
    return "nan" if x is None or x != x else f"{x:.1f}"


def g_(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def report(r: dict) -> str:
    L = []
    p = L.append
    p(f"== probe_v6 {r['label']} ({r['arm']}) step {r['step']}: wall {r.get('wall_s')} s; inner timing {json.dumps(r['timing'])}")
    if r.get("smoke_nulls"):
        p("   SMOKE: null draws reduced to " + json.dumps(SMOKE_NULLS) + " (full: " + json.dumps(FULL_NULLS) + ")")
    p(f"   rows {json.dumps(r['rows'])}; dim strict rows {r['n_strict']}")
    p(f"   clean exact match: " + ", ".join(f"{k} {v:.3f}" for k, v in r["acc"].items()) + f"; dim strict {r['dim_strict_acc']:.3f}")
    p(f"   checks: last-layer lens == logits {r['checks']['lens_last_equals_logits']}")
    p(f"           hits vs logged / V5 CPU probe: {json.dumps(r['checks']['hits_vs_logged'])}")
    p("   decoded inputs:")
    for q in r["prompts"]:
        p(f"     {q['set']} r{q['row']} seq {q['seq_len']} ans-pred pos {q['answer_pred_pos']} span {q['answer_tokens']}: {q['text']!r}")
    tp = r["A1"].get("torus_pair")
    if tp:
        p(f"-- A1 torus coordinates (B1): chosen character pair {tp['chosen']} (mean SEEN-row grouped R2 {f3(tp['mean_seen_r2'])} over "
          f"L{tp['sel_layers']}, {tp['n_pairs']} generating pairs) vs the prior's DTAB {tp['dtab_pair']} ({f3(tp['dtab_mean_seen_r2'])}); "
          "top5 " + ", ".join(f"{u}:{f3(v)}" for u, v in tp["top5"]))
        p("   rows 'torus' use the chosen pair, rows 'torus_dtab' the DTAB pair; held R2 is untouched by the selection (seen rows only)")
    p("-- A1 decoders (grouped OOF R2 | held R2 | nearest-lattice | shuffled-label R2 mean / max, z)")
    for k, v in r["A1"]["r2"].items():
        ex = f"  strict R2 {f3(v.get('strict_r2'))} near {f3(v.get('strict_nearest'))}" if "strict_r2" in v else ""
        p(f"   {k:14s} R2 {f3(v['r2'])} held {f3(v['held_r2'])} near {f3(v['nearest'])} | shuf {f3(v['r2_shuf']['mean'])}/"
          f"{f3(v['r2_shuf'].get('max'))} z {f1(v.get('r2_z_shuf'))}{ex}")
    p("-- A1 overlap, mean cos^2. PRIMARY class-mean PCs k3, k6 [row-shuffle null max, z | cov-matched p95]; "
      "secondary decoder column space [random p95, cov-matched p95 | shuffled-label max, z]")
    for k, v in r["A1"]["overlap"].items():
        pcs = " ".join(f"{kk} {f3(v[kk])} [{f3(v[f'{kk}_null_rowshuffle'].get('max'))} z {f1(v.get(f'{kk}_z_rowshuffle'))} | "
                       f"{f3(v[f'{kk}_null_covmatched']['p95'])}]" for kk in ("pc3", "pc6"))
        p(f"   {k:18s} {pcs} || dec {f3(v['dec'])} [{f3(v['dec_null_random']['p95'])} {f3(v['dec_null_covmatched']['p95'])} | "
          f"{f3(v['dec_null_shuffled_labels'].get('max'))} z {f1(v.get('dec_z_shuffled'))}]" +
          (f" dec_dtab {f3(v['dec_dtab'])}" if "dec_dtab" in v else ""))
    p("-- A1 transfer (src decoder -> tgt states -> OOF map; R2 | strict | null shuffled-src mean/max, z | random-proj p95 | own decoder)")
    for k, v in r["A1"]["transfer"].items():
        ns = v["null_shuffled_src"]
        p(f"   {k:22s} R2 {f3(v['r2'])} strict {f3(v['strict_r2'])} | shuf {f3(ns['mean'])}/{f3(ns.get('max'))} z {f1(v.get('z_shuffled_src'))} | "
          f"rand {f3(v['null_random_proj']['p95'])} | own {f3(v['own_decoder_r2'])}")
    p("-- A2 ablation at the answer-prediction position (exact match). Rows: 'held' = all held rows (PRIMARY); for dim_held also "
      f"'strict' (n {r['n_strict']}). Cell = ablated value @ dose (fraction of the evaluated rows' centred state variance removed); "
      "nulls = mean over draws (random = isotropic, cov = from the evaluated set's own covariance, shuf_<d> = class-mean PCs of d with "
      "shuffled labels); exc = drop minus the drop that a line through all null draws predicts at the same dose (z vs its residual sd)")
    for nm, rec in r["A2"].items():
        cl = rec["clean"]
        p(f"   {nm}: clean acc {f3(cl['acc'])} tok1 {f3(cl['tok1'])} tok2(\\n) {f3(cl['tok2'])}" +
          (f" strict {f3(cl['strict'])}" if "strict" in cl else "") + f"; k {rec['k']}")
        for h, hv in rec["conds"].items():
            for kk, cr in hv.items():
                if kk == "k0":
                    continue
                srcs = [s for s in cr if s != "dose_cmp" and not (s == "random" or s.startswith("null_"))]
                nls = [s for s in cr if (s == "random" or s.startswith("null_")) and cr[s]]
                dc = cr.get("dose_cmp", {})
                for lab, m_ in [("held", "acc")] + ([("strict", "strict")] if "strict" in cl else []):
                    cells = " ".join(f"{s} {f3(cr[s][m_])}@{f3(cr[s].get('dose'))}" for s in srcs)
                    nc = " ".join(f"{s.replace('null_', '')} {f3(cr[s][m_]['mean'])}@{f3(g_(cr[s], 'dose', 'mean'))}" for s in nls)
                    ex = " ".join(f"{s} {f3(dc[m_][s]['excess_vs_null_fit'])} (z {f1(dc[m_][s]['excess_z'])})" for s in srcs) if m_ in dc else "n/a"
                    p(f"     {h} {kk:3s} {lab:6s} {cells} | {nc} | exc {ex}")
                if "dose_cmp" in cr:
                    p(f"     {h} {kk:3s} null dose range {[round(x, 3) for x in cr['dose_cmp']['null_dose_range']]} "
                      f"({cr['dose_cmp']['n_null_draws']} draws); tok1 " + " ".join(f"{s} {f3(cr[s]['tok1'])}" for s in srcs))
    p("-- A3 logit lens at the answer-prediction position, dim answer tokens (margin / top-1 per layer)")
    for nm, rec in r["A3"].items():
        subs = list(rec["n"])
        p(f"   {nm} n {json.dumps(rec['n'])}; lens fp32 vs model bf16 max|d| {rec['lens_fp32_vs_model_bf16_maxabs']:.3g}, argmax agree "
          f"{rec['lens_fp32_argmax_eq_model']:.3f}; DLA sum-vs-resid {rec['dla_sum_vs_final_resid_maxabs']:.3g}")
        for l_, lv in rec["layers"].items():
            p(f"     L{l_}: " + "  ".join(f"{s} {f3(lv[s]['margin'])}/{f3(lv[s]['top1'])}" for s in subs if lv[s]['margin'] is not None))
        for s, dv in rec["dla"].items():
            p(f"     DLA {s:11s} " + " ".join(f"{k} {v:+.2f}" for k, v in dv.items()))
    p("-- A4 weights (CKA unembed rows vs coords [perm p95] | class means of final-norm state vs unembed rows [perm p95] | vs coords [perm p95]; "
      "torus coords = chosen pair, DTAB pair in parentheses)")
    for d, v in r["A4"].items():
        dt = (f" (DTAB {f3(v['cka_unembed_vs_coords_dtab'])} / {f3(v['cka_classmean_finalnorm_vs_coords_dtab'])})"
              if "cka_unembed_vs_coords_dtab" in v else "")
        p(f"   {d:6s} {f3(v['cka_unembed_vs_coords'])} [{f3(v['null_perm']['p95'])}] | {f3(v['cka_classmean_finalnorm_vs_unembed'])} "
          f"[{f3(v['null_perm_classmean']['p95'])}] | {f3(v['cka_classmean_finalnorm_vs_coords'])} "
          f"[{f3(g_(v, 'null_perm_classmean_coords', 'p95'))}]  (classes {v['n_classes']}){dt}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    ap.add_argument("--label")
    ap.add_argument("--step", type=int)
    ap.add_argument("--smoke", action="store_true", help="reduced null draws (rows and analyses unchanged)")
    ap.add_argument("--smoke_c", action="store_true", help="code check on out_v5/smoke/C (3 layers, d64)")
    ap.add_argument("--cfg", default="{}", help="JSON overrides of the inner config")
    a = ap.parse_args()
    over = json.loads(a.cfg)
    if a.smoke_c:
        return run_one("smokeC", R / "out_v5" / "smoke" / "C", 19, SMOKE_NULLS | over, tag="_smokeC")
    if a.index is not None:
        lab, d, st = TARGETS[a.index]
    else:
        d = dict(RUNS)[a.label]; lab, st = a.label, a.step
    run_one(lab, d, st, (SMOKE_NULLS if a.smoke else {}) | over, tag="_smoke" if a.smoke else "")


if __name__ == "__main__":
    main()
