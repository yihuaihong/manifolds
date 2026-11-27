"""SAMELR downstream read-out (2026-11-25, exp-builder). Pre-registered rule = F96's (runs/downstream/AUDIT.md, Amendment A4):
the snapshot read-out runs/downstream/code_65475e329342/downstream_readout.py (sha d7bdff1097e6) is imported unchanged and its
paired() / verdict() / tci() / boot2() / check_pair() are applied to new pairs: at each size (190M, 505M, 1.07B)
all6@6e-4 vs none@6e-4, seeds 16, 17, 18, 19 (d_s = all6 - none per seed; mean, SD, 95% t-CI with T975[3]; margin 1.0 point;
verdicts worse / non-inferior / undetermined / at chance; LAMBADA ppl ratio and boot2 descriptive).
Inputs: runs/pt6v2_size/downstream/out_<size>/pt6v2_size/<model stem>.json (downstream_eval.py 5d694e38ab75, one out dir per
size because the model stems are the same at every size).
Stops (exit 2, no verdicts written) on: an arm-identity problem (check_pair, A2), a record whose model is not under
runs/pt6v2_size/<size>/, n_params other than the size's, or an eval code sha other than 5d694e38ab75. Flags only (printed beside
the result): n per benchmark != EXPECT_N, a score outside [chance - 3, 100], a double-path greedy mismatch, a load check
|replica - logged| > 2e-3, fewer than 4 seeds (the rule's verdict is then INCOMPLETE).
usage: samelr_readout.py [--write <prefix>]    |    samelr_readout.py --selftest"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path("/scratch/yh6210/llmadd_torch/runs/downstream/code_65475e329342")
D = Path("/scratch/yh6210/llmadd_torch/runs/pt6v2_size/downstream")
READOUT_SHA, EVAL_SHA = "d7bdff1097e6", "5d694e38ab75"
assert hashlib.sha256((CODE / "downstream_readout.py").read_bytes()).hexdigest()[:12] == READOUT_SHA, "snapshot read-out sha"
_spec = importlib.util.spec_from_file_location("downstream_readout_snapshot", CODE / "downstream_readout.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
SEEDS = (16, 17, 18, 19)
assert len(SEEDS) == len(R.SEEDS)   # the rule's verdict needs len(R.SEEDS) = 4 paired seeds
SIZES = {"190m": 190297344, "505m": 505437184, "1b": 1072259840}
PRIOR, NONE = "pt6v2_size/pt6v2-all6-lr6e-4", "pt6v2_size/pt6v2-none-lr6e-4"


def read_size(z, out_dir, root="/scratch/yh6210/llmadd_torch/runs/pt6v2_size"):
    cells = R.load(out_dir)
    cp = {"id": f"samelr-{z}", "prior": PRIOR, "none": NONE}
    stop = list(R.check_pair(cells, cp))
    flags = []
    for c, v in cells.items():
        for s, r in v.items():
            if c not in (PRIOR, NONE):
                flags.append(f"{c} s{s}: cell not in this comparison (ignored)")
                continue
            if not str(r["model"]).startswith(f"{root}/{z}/"):
                stop.append(f"{c} s{s}: model {r['model']} not under {root}/{z}/")
            if r.get("n_params") != SIZES[z]:
                stop.append(f"{c} s{s}: n_params {r.get('n_params')} != {SIZES[z]}")
            if r.get("eval_code_sha") != EVAL_SHA:
                stop.append(f"{c} s{s}: eval code sha {r.get('eval_code_sha')} != {EVAL_SHA}")
            if s not in SEEDS:
                flags.append(f"{c} s{s}: seed outside {SEEDS} (not paired)")
            for b in R.BENCH:
                sc = r["scores"][b]
                if sc["n"] != R.EXPECT_N[b]:
                    flags.append(f"{c} s{s} {b}: n {sc['n']} != {R.EXPECT_N[b]}")
                ch = sc.get("chance", R.CHANCE.get(b, 0.0))
                for m in ("acc", "acc_norm"):
                    x = sc.get(m)
                    if x is not None and not (ch - 3 <= x <= 100):
                        flags.append(f"{c} s{s} {b} {m} {x:.2f} outside [chance {ch:.1f} - 3, 100]")
                if sc["double_path"].get("greedy_mismatch", 0):
                    flags.append(f"{c} s{s} {b}: double path greedy mismatch {sc['double_path']['greedy_mismatch']}/24")
            lc = r.get("load_check") or {}
            if abs(lc.get("replica_minus_logged", 0.0)) > 2e-3:
                flags.append(f"{c} s{s}: load check |replica - logged| {lc['replica_minus_logged']:+.2e} > 2e-3")
    have = {c: sorted(cells.get(c, {})) for c in (PRIOR, NONE)}
    if any(len(v) < len(SEEDS) for v in have.values()):
        flags.append(f"seeds present {have}: fewer than 4 paired seeds -> verdict INCOMPLETE")
    res = {"size": z, "out_dir": str(out_dir), "seeds_present": have, "stop": stop, "flags": flags}
    if not stop:
        res["stats"] = R.paired(cells, PRIOR, NONE, R.BENCH, seeds=SEEDS, boot=True)
        res["bpb_logged"] = {c: {s: cells[c][s].get("val_bpb_logged_final") for s in have[c]} for c in (PRIOR, NONE)}
    return res


def report(rs):
    L = ["# SAMELR downstream: all6@6e-4 - none@6e-4, paired over seeds 16-19 (F96 rule, Amendment A4)", ""]
    for z, r in rs.items():
        L.append(f"## {z}  seeds {r['seeds_present']}")
        if r["stop"]:
            L += ["STOPPED (no verdicts): " + p for p in r["stop"]]
        else:
            L.append(f"{'bench':10s} {'n':>2s} {'all6':>7s} {'none':>7s} {'diff':>7s} {'95% t-CI':>17s} {'wins':>4s} | verdict | per-seed d")
            for b in R.BENCH:
                x = r["stats"][b]
                if not x["n"]:
                    L.append(f"{b:10s}  0 | INCOMPLETE")
                    continue
                ci = "n/a" if x["lo"] is None else f"[{x['lo']:+.2f}, {x['hi']:+.2f}]"
                L.append(f"{b:10s} {x['n']:2d} {x['prior_mean']:7.2f} {x['none_mean']:7.2f} {x['mean']:+7.2f} {ci:>17s} {x['wins']:4d} | "
                         f"{x['verdict']} | " + " ".join(f"{d:+.2f}" for d in x["d"]))
            pr = r["stats"].get("lambada_ppl_ratio")
            if pr:
                L.append(f"LAMBADA ppl ratio all6/none (descriptive): geo mean {pr['geo_mean']:.3f}, per seed "
                         + " ".join(f"{v:.3f}" for v in pr["per_seed"]))
        L += ["flag: " + f for f in r["flags"]] + [""]
    return "\n".join(L)


def selftest():
    tmp = Path(tempfile.mkdtemp(prefix="samelr_readout_selftest_"))
    try:
        base = {"blimp": 70.0, "lambada": 15.0, "hellaswag": 30.0, "arc_easy": 40.0, "piqa": 60.0}
        root = str(tmp / "runs/pt6v2_size")

        def put(z, arm, s, delta, lr=0.0006, npar=None, sub=None):
            out = tmp / f"out_{z}" / "pt6v2_size"
            out.mkdir(parents=True, exist_ok=True)
            tgt = "none" if arm == "none" else "helix3"
            mp = f"{root}/{sub or z}/{arm}-lr6e-4-s{s}/pt6v2-{arm}-lr6e-4/model_whole_{tgt}_seed{s}_pt6v2-{arm}-lr6e-4.pt"
            r = R._fake_rec(mp, s, lr, 768, {b: v + delta.get(b, 0.0) for b, v in base.items()})
            r.update(n_params=npar or SIZES[z], eval_code_sha=EVAL_SHA)
            (out / (Path(mp).stem + ".json")).write_text(json.dumps(r))

        # 190m: blimp worse by ~3 (all seeds), piqa +0.1 (non-inferior), hellaswag noisy -> undetermined
        noise = {16: -2.5, 17: 1.8, 18: -3.0, 19: 2.2}
        for s in SEEDS:
            put("190m", "none", s, {})
            put("190m", "all6", s, {"blimp": -3.0 - 0.1 * (s - 16), "piqa": 0.1, "hellaswag": noise[s] - 0.4})
        r = read_size("190m", tmp / "out_190m", root)
        assert not r["stop"], r["stop"]
        v = {b: r["stats"][b]["verdict"] for b in R.BENCH}
        assert v["blimp"] == "worse" and v["piqa"].startswith("non-inferior") and v["hellaswag"] == "undetermined", v
        assert v["lambada"].startswith("non-inferior") and abs(r["stats"]["blimp"]["mean"] + 3.15) < 1e-9, r["stats"]["blimp"]
        # 505m: seed 19 missing -> INCOMPLETE + flag
        for s in SEEDS[:3]:
            put("505m", "none", s, {})
            put("505m", "all6", s, {})
        r = read_size("505m", tmp / "out_505m", root)
        assert not r["stop"] and all(r["stats"][b]["verdict"] == "INCOMPLETE" for b in R.BENCH), r
        assert any("fewer than 4" in f for f in r["flags"]), r["flags"]
        # 1b: an lr mismatch (A2) and a model from another size -> stop
        for s in SEEDS:
            put("1b", "none", s, {}, lr=0.002 if s == 17 else 0.0006)
            put("1b", "all6", s, {}, sub="505m" if s == 18 else None)
        r = read_size("1b", tmp / "out_1b", root)
        assert any("lr differs" in p for p in r["stop"]) and any("not under" in p for p in r["stop"]) and "stats" not in r, r["stop"]
        print(report({"190m": read_size("190m", tmp / "out_190m", root)}))
        print("SELFTEST PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None, help="prefix for <prefix>.md / <prefix>.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    rs = {z: read_size(z, D / f"out_{z}") for z in SIZES}
    txt = report(rs)
    print(txt)
    if a.write:
        for ext, body in ((".json", json.dumps(rs, indent=1, default=str)), (".md", txt)):
            t = Path(a.write + ext + ".tmp"); t.write_text(body); t.replace(a.write + ext)
        print("wrote", a.write + ".md", a.write + ".json")
    if any(r["stop"] for r in rs.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
