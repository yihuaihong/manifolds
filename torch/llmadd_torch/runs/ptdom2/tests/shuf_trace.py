"""ptdom2 test: --prior_shuf on every domain ptdom2 shuffles (S: countries, people, cities, months; R: stars, landmarks, compass,
prefixes, ranks, scalar_adj), CPU, tiny model, the frozen trainer (runs/ptdom2/code = 11f61b9, d87a1502cee8).

  run <batch> <arm> <lam> <out>   the trainer's own main() in-process, with the module's centred_gram / gram_raw / cka wrapped
                                  to RECORD every prior computation (nothing is changed: the wrappers call the originals and
                                  return their values): the step, the caller chain (training forward / dom_grads / dom_eval_log /
                                  evaluate), the domain, whether the table passed is DTAB_ACT[d] / DTAB[d], the item ids at the
                                  sites, and at the first on-step the hooked states -> <out>/trace.json + the run JSON
  check <dir>                     the known answers below on every recorded (domain, arm) pair -> PASS / FAIL lines

Tiny setting (the A3 / A4 smoke's model, testing-only eval reductions): d64 x 3 layers, seq 128, lr 1e-3, seed 0; bs 8 (batch A:
bs 24: months has one site per line); 160 planned steps -> the window (0.05, 0.4) = steps 8..63 with evals every 8 steps (bs 24:
every 4), so the FIRST on-step is an eval step and both arms have identical states there (no prior before it, the same data);
--stop_after 24. Known answers (per domain d, true arm only-d vs shuffled arm only-d-shuf, same seed and data):
  T1 target: an INDEPENDENT re-derivation of the permutation (this file's own far_perm / gram code on the trainer's registered
     stream: SeedSequence(4300 + seed).spawn(54)[3 i + 1] for NEWDOM[i], SeedSequence(4200 + seed).spawn(16)[6 + i] for
     ANCHOR[i], SeedSequence(4100 + seed).spawn(16)[6 + i] for EXTRA[i]) gives q with DTAB_ACT[d] == DTAB[d][q] exactly in the
     shuffled arm (DTAB_ACT[d] is DTAB[d] in the true arm), a row permutation that moves items, CKA to the true target = the
     logged shuf_cka_to_true (< 0.15)
  T2 the training-forward prior is computed for d ONLY, on DTAB_ACT[d] (shuffled arm: DTAB[d][q]), at the same steps (inside
     the window only) and the same sites / item ids in both arms (window, site and the 4-site / 3-item minimum identical);
     no other domain has a training prior call, a geo_<x>, a gnorm_prior_<x> or shuf true
  T3 dose: lam_domains / domains.<d>.lam = 1 in both arms; at the first on-step (identical states) lm, grad_lm and
     gnorm_true_<d> are equal in both arms, the true arm's gnorm_prior_<d> = gnorm_true_<d> (lambda 1), and (compass, countries)
     a shuffled run with --lam 2 has gnorm_prior_<d> = 2 x the --lam 1 shuffled run's (the weight multiplies the shuffled term)
  T4 the logged geo_<d> at every on-window eval = 1 - CKA of the recorded states to the SHUFFLED target: recomputed here in fp64
     numpy (|diff| < 0.02) and EXACTLY (<= 1e-6) by the trainer's own bf16-autocast arithmetic applied to the recorded states
     and the independently derived target, which gives another value with the other target (the reverse in the true arm)
  T5 data: data_sha equal at every eval in both arms; the eval-prompt CKA cka_<d> is computed on the TRUE target in both arms
     (every evaluate() call uses DTAB[d])
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

B = Path("/scratch/yh6210/llmadd_torch")
PIN = B / "runs" / "ptdom2" / "code" / "tools"
DATA, TOK = B / "pretrain_data", B / "models/OLMo-2-1124-7B/stage1-step2000-tokens9B"
DOMS = {"A": "months,stars,landmarks,languages,colours,cities", "B1": "countries,animals,country_size,animal_size,people,units",
        "B2": "zodiac,compass,planets,elements,prefixes,ranks", "B3": "chinese_zodiac,kinship,life_stages,geo_periods,greek,scalar_adj"}
CELLS = [("B1", "countries"), ("B1", "people"), ("A", "cities"), ("A", "months"), ("A", "stars"), ("A", "landmarks"),
         ("B2", "compass"), ("B2", "prefixes"), ("B2", "ranks"), ("B3", "scalar_adj")]
LAM2 = ("compass", "countries")
STOP, SEED = 24, 0


def tiny_args(batch, arm, lam, out):
    bs, ev = (24, 3) if batch == "A" else (8, 2)   # eval every round(ev * 32 / bs) = 4 / 8 steps
    a = (f"--data {DATA} --tokenizer {TOK} --tokenization whole --out {out} --arith_lines 1 --domains numbers,{DOMS[batch]} "
         f"--extra_lines 2 --lr 1e-3 --d_model 64 --n_layers 3 --n_heads 4 --d_ff 128 --seq 128 --bs {bs} --tokens {160 * bs * 128} "
         f"--eval_every {ev} --n_eval 20 --dom_eval_rows 20 --dom_seen 20 --anc_eval_max 40 --seed {SEED} --stop_after {STOP}")
    if arm == "none":
        return a + " --target none"
    d = arm[5:].replace("-shuf", "")
    return a + (f" --target none --schedule window --window 0.05,0.4 --site ctx --lam_rule each --prior_domains {d}"
                + (f" --prior_shuf {d}" if arm.endswith("-shuf") else "") + f" --lam {lam}")


# ------------------------------------------------------------------------------------------------ run (instrumented)
def run(batch, arm, lam, out):
    os.environ.pop("SLURM_JOB_ID", None)
    sys.path.insert(0, str(PIN))
    import torch
    torch.set_num_threads(1)
    import pretrain_range as pr
    assert Path(pr.__file__).resolve().parent == PIN.resolve()
    act = arm[5:].replace("-shuf", "")
    S_ON = 8
    trace, tabs, states = [], {}, {}
    o_cg, o_raw, o_cka = pr.centred_gram, pr.gram_raw, pr.cka

    def ctx():
        f, chain = sys._getframe(2), []
        while f is not None:
            chain.append(f)
            if f.f_code.co_name == "main" and "DTAB_ACT" in f.f_locals:
                break
            f = f.f_back
        names = [g.f_code.co_name for g in chain]
        mf = chain[-1] if chain and chain[-1].f_code.co_name == "main" else None
        return chain, names, mf

    def record(F, kind):
        chain, names, mf = ctx()
        if mf is None or "s" not in mf.f_locals or "far_perm" in names:
            return   # build time (far_perm's search, the build asserts): not a prior computation of training
        s = int(mf.f_locals["s"])
        L = mf.f_locals
        if "dom_prior" in names:
            dp = chain[names.index("dom_prior")]
            d, table, m = dp.f_locals["d"], dp.f_locals["table"], dp.f_locals["m"]
            caller = chain[names.index("dom_prior") + 1].f_code.co_name
            if d not in tabs:
                tabs[d] = {"true": np.array(L["DTAB"][d]).tolist(), "act": np.array(L["DTAB_ACT"][d]).tolist()}
            trace.append({"s": s, "where": caller, "d": d, "kind": kind, "is_act": table is L["DTAB_ACT"][d],
                          "is_true": table is L["DTAB"][d], "ids": [int(v) for v in m[:, 2]],
                          "rows_ok": bool(np.array_equal(np.asarray(F), np.asarray(table)[m[:, 2]]))})
        elif "evaluate" in names:
            ev = chain[names.index("evaluate")]
            d = ev.f_locals.get("d")
            v = ev.f_locals.get("v")
            ok_true = v is not None and np.array_equal(np.asarray(F), np.asarray(L["DTAB"][d])[v])
            ok_act = v is not None and np.array_equal(np.asarray(F), np.asarray(L["DTAB_ACT"][d])[v])
            trace.append({"s": s, "where": "evaluate", "d": d, "kind": kind, "rows_true": bool(ok_true), "rows_act": bool(ok_act)})
        else:
            trace.append({"s": s, "where": "/".join(names[:4]), "d": None, "kind": kind})

    def cg(F, dev):
        record(F, "centred_gram")
        return o_cg(F, dev)

    def raw(F):
        record(F, "gram_raw")
        return o_raw(F)

    def ck(H, K):
        chain, names, mf = ctx()
        if mf is not None and "dom_prior" in names and "s" in mf.f_locals:
            dp = chain[names.index("dom_prior")]
            caller = chain[names.index("dom_prior") + 1].f_code.co_name
            if caller == "main" and dp.f_locals["d"] == act:   # the training forward's prior term (every on-step)
                st = states.setdefault(str(int(mf.f_locals["s"])), {"H": [], "ids": None})
                st["H"].append(H.detach().float().cpu().numpy().tolist())
                st["ids"] = [int(v) for v in dp.f_locals["m"][:, 2]]
        return o_cka(H, K)

    pr.centred_gram, pr.gram_raw, pr.cka = cg, raw, ck
    Path(out).mkdir(parents=True, exist_ok=True)
    sys.argv = ["pretrain_range.py"] + tiny_args(batch, arm, lam, out).split()
    pr.main()
    (Path(out) / "trace.json").write_text(json.dumps({"batch": batch, "arm": arm, "lam": lam, "act": act, "s_on": S_ON,
                                                      "trace": trace, "tabs": tabs, "states": states}))
    print(f"traced {batch} {arm} lam {lam}: {len(trace)} prior computations recorded")


# ------------------------------------------------------------------------------------------------ independent re-derivation
def gram_std_mine(F):
    F = np.asarray(F, np.float64)
    F = F[:, F.std(0) > 1e-9]
    F = (F - F.mean(0)) / F.std(0)
    F = F - F.mean(0)
    return F @ F.T


def gram_raw_mine(F):
    F = np.asarray(F, np.float64)
    F = F - F.mean(0)
    return F @ F.T


def far_perm_mine(F, rng, gram, n_try=3000, thr=0.15):
    G0 = gram(F)
    best = (2.0, None)
    for _ in range(n_try):
        q = rng.permutation(len(F))
        G1 = gram(F[q])
        c = float((G0 * G1).sum() / (np.linalg.norm(G0) * np.linalg.norm(G1)))
        if c < best[0]:
            best = (c, q)
        if c < thr:
            break
    return best


NEWDOM = ("countries", "animals", "country_size", "animal_size", "people", "units", "zodiac", "compass", "planets", "elements",
          "prefixes", "ranks", "chinese_zodiac", "kinship", "life_stages", "geo_periods", "greek", "scalar_adj")
ANCHOR = ("stars", "landmarks", "languages", "colours", "cities")
EXTRA = ("months", "weekdays", "clock", "s4", "dim")
RAW = ("languages", "animals", "units", "scalar_adj", "kinship")   # tree / within targets: re-checked against domains.<d>.raw_gram


def stream(d, seed):
    if d in NEWDOM:
        return np.random.SeedSequence(4300 + seed).spawn(3 * len(NEWDOM))[3 * NEWDOM.index(d) + 1]
    if d in ANCHOR:
        return np.random.SeedSequence(4200 + seed).spawn(16)[6 + ANCHOR.index(d)]
    return np.random.SeedSequence(4100 + seed).spawn(16)[6 + EXTRA.index(d)]


def cka_mine(H, F, gram):
    H = np.asarray(H, np.float64)
    H = H - H.mean(0)
    G = H @ H.T
    K = gram(F)
    return float((G * K).sum() / (np.linalg.norm(G) * np.linalg.norm(K) + 1e-8))


def geo_trainer_arith(Hs, F, raw):
    """The trainer's own arithmetic (pretrain_helix.cka / centred_gram of the frozen snapshot; numpy gram_raw for a raw target)
    under CPU bf16 autocast, reduced as dom_prior / prior do (sum over the layers of 1 - CKA, / the number of layers), on the
    RECORDED states and a GIVEN target (the target is the independent input here)."""
    import torch
    sys.path.insert(0, str(PIN))
    from pretrain_helix import cka, centred_gram
    F = np.asarray(F, np.float64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        K = torch.tensor(gram_raw_mine(F), dtype=torch.float32) if raw else centred_gram(F, "cpu")
        v = sum(1 - cka(torch.tensor(np.asarray(H), dtype=torch.float32), K) for H in Hs) / len(Hs)
    return float(v)


# ------------------------------------------------------------------------------------------------ check
def load_dir(d):
    tr = json.loads((d / "trace.json").read_text())
    rj = next(d.glob("run_*.json"))
    return tr, json.loads(rj.read_text())


def check(root):
    root = Path(root)
    res = []
    T = lambda name, ok, info="": (res.append((name, bool(ok))), print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info != "" else "")))
    for batch, d in CELLS:
        tt, rt = load_dir(root / f"{batch}-only-{d}")
        ts, rs = load_dir(root / f"{batch}-only-{d}-shuf")
        S_ON = ts["s_on"]
        raw = bool(rs["domains"][d].get("raw_gram", False)) if d != "months" else False
        gram = gram_raw_mine if raw else gram_std_mine
        F = np.asarray(ts["tabs"][d]["true"], np.float64)
        Fa_s, Fa_t = np.asarray(ts["tabs"][d]["act"], np.float64), np.asarray(tt["tabs"][d]["act"], np.float64)
        c_mine, q = far_perm_mine(F, np.random.default_rng(stream(d, SEED)), gram)
        moved = int((q != np.arange(len(q))).sum())
        T(f"T1 {d}: shuffled table DTAB_ACT == DTAB[q] (q re-derived independently on the registered stream), a row permutation moving "
          f"{moved}/{len(q)} items; true arm DTAB_ACT == DTAB", np.array_equal(Fa_s, F[q]) and np.array_equal(Fa_t, F)
          and np.array_equal(np.asarray(tt["tabs"][d]["true"]), F) and moved > 0
          and np.array_equal(np.sort(Fa_s.round(12), axis=0), np.sort(F.round(12), axis=0)))
        lg = rs["domains"][d].get("shuf_cka_to_true")
        T(f"T1 {d}: shuf_cka_to_true logged {lg:.6f} = the re-derivation {c_mine:.6f} (< 0.15); raw Gram {raw}",
          abs(lg - c_mine) < 1e-9 and c_mine < 0.15 and rs["domains"][d]["shuf"] is True and rt["domains"][d]["shuf"] is False)
        tr_s = [x for x in ts["trace"] if x["where"] == "main"]
        tr_t = [x for x in tt["trace"] if x["where"] == "main"]
        w0, w1 = 8, 64
        T(f"T2 {d}: training-forward prior calls only for {d}, on DTAB_ACT (rows = table[ids] at every call), only inside the window "
          f"({len(tr_s)} / {len(tr_t)} calls in the shuffled / true arm)",
          tr_s and tr_t and all(x["d"] == d and x["is_act"] and x["rows_ok"] and w0 <= x["s"] < w1 for x in tr_s + tr_t)
          and all(not x["is_true"] for x in tr_s) and all(x["is_true"] for x in tr_t))
        key = lambda xs: [(x["s"], tuple(x["ids"])) for x in xs if x["kind"] in ("centred_gram", "gram_raw")]
        T(f"T2 {d}: the same steps and the same site item ids in both arms (window / site / minimum identical)", key(tr_s) == key(tr_t),
          f"{len(key(tr_s))} computations")
        others = sorted({x["d"] for x in tr_s + tr_t if x["d"] != d})
        c_s, c_t = rs["curve"], rt["curve"]
        ina = [x for x in rs["domains"] if x != d]
        T(f"T2 {d}: no other domain is trained, logged or shuffled (no geo_<x> / gnorm_prior_<x> / shuf for {len(ina)} others)",
          not others and not any(e.get(f"geo_{x}") is not None or f"gnorm_prior_{x}" in e for e in c_s + c_t for x in ina if x != "numbers")
          and not any(rs["domains"][x].get("shuf") for x in ina) and all(e["geo"] == 0 for e in c_s), others)
        e_s = next(e for e in c_s if e["step"] == S_ON)
        e_t = next(e for e in c_t if e["step"] == S_ON)
        lam_ok = rs["lam_domains"][d] == 1.0 == rt["lam_domains"][d] and rs["domains"][d]["lam"] == 1.0 == rt["domains"][d]["lam"]
        T(f"T3 {d}: dose at the first on-step {S_ON}: lambda 1 in both; lm / grad_lm / gnorm_true_{d} equal in both arms; true arm "
          f"gnorm_prior = gnorm_true (lambda 1); prior_on in both",
          lam_ok and e_s["prior_on"] == e_t["prior_on"] == 1 and e_s["lm"] == e_t["lm"] and e_s["grad_lm"] == e_t["grad_lm"]
          and abs(e_s[f"gnorm_true_{d}"] - e_t[f"gnorm_true_{d}"]) <= 1e-6 * e_t[f"gnorm_true_{d}"]
          and abs(e_t[f"gnorm_prior_{d}"] / e_t[f"gnorm_true_{d}"] - 1) < 1e-6,
          f"gnorm_prior shuf {e_s[f'gnorm_prior_{d}']:.5f} true {e_t[f'gnorm_prior_{d}']:.5f}; gnorm_true {e_s[f'gnorm_true_{d}']:.5f} / {e_t[f'gnorm_true_{d}']:.5f}")
        if d in LAM2:
            t2, r2 = load_dir(root / f"{batch}-only-{d}-shuf-lam2")
            e2 = next(e for e in r2["curve"] if e["step"] == S_ON)
            T(f"T3 {d}: --lam 2 shuffled run at step {S_ON}: gnorm_prior = 2.000 x the --lam 1 shuffled run's, the same lm / gnorm_true",
              abs(e2[f"gnorm_prior_{d}"] / e_s[f"gnorm_prior_{d}"] - 2) < 1e-5 and e2["lm"] == e_s["lm"] and r2["lam_domains"][d] == 2.0,
              f"ratio {e2[f'gnorm_prior_{d}'] / e_s[f'gnorm_prior_{d}']:.7f}")
        for nm, tr_, run_, Ft, Fo in (("shuffled", ts, rs, F[q], F), ("true", tt, rt, F, F[q])):
            evs = {e["step"]: e for e in run_["curve"] if e.get("prior_on") and e.get(f"geo_{d}") is not None}
            steps = sorted(int(k) for k in tr_["states"] if int(k) in evs)
            rows_, own_ok, disc = [], True, 0
            for st in steps:
                H, ids = tr_["states"][str(st)]["H"], tr_["states"][str(st)]["ids"]
                lgv = evs[st][f"geo_{d}"]
                g_my = float(np.mean([1 - cka_mine(h, Ft[ids], gram) for h in H]))
                b_my, b_other = geo_trainer_arith(H, Ft[ids], raw), geo_trainer_arith(H, Fo[ids], raw)
                own_ok &= abs(lgv - g_my) < 0.02 and abs(lgv - b_my) <= 1e-6
                disc += abs(lgv - b_other) > 1e-6
                rows_.append(f"s{st} logged {lgv:.4f} own {b_my:.4f} (fp64 {g_my:.4f}) other {b_other:.4f}")
            T(f"T4 {d} ({nm} arm): at every on-window eval ({steps}) the logged geo_{d} = 1 - CKA to its OWN target (the trainer's bf16 "
              f"arithmetic on the recorded states with the independently derived target, |diff| <= 1e-6; fp64 within 0.02); with the "
              f"other target it differs at {disc} of them (>= 1 required)", steps and own_ok and disc >= 1, "; ".join(rows_))
        ev_s = [x for x in ts["trace"] if x["where"] == "evaluate" and x["d"] == d]
        ev_t = [x for x in tt["trace"] if x["where"] == "evaluate" and x["d"] == d]
        T(f"T5 {d}: data_sha equal at every eval in both arms; every eval-prompt CKA of {d} on the TRUE target ({len(ev_s)} / {len(ev_t)} calls; "
          f"in the shuffled arm never on its shuffled table)",
          [e["data_sha"] for e in c_s] == [e["data_sha"] for e in c_t] and ev_s and ev_t and all(x["rows_true"] for x in ev_s + ev_t)
          and not any(x["rows_act"] for x in ev_s))
    n_ok = sum(ok for _, ok in res)
    print(f"SHUF TRACE {'PASS' if n_ok == len(res) else 'FAIL'} {n_ok}/{len(res)}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        check(sys.argv[2])
