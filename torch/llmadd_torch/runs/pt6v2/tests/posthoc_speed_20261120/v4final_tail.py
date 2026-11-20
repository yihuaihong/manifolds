
# ---------------------------------------------------------------- V4 final descriptive read (8 seeds; read only)
OUT4 = B / "runs" / "pt6v2" / "out_v4"
NM = {"numbers": "8e-3", "dim": "3.2e-2", "cities": "4e-3", "countries": "4e-3", "torus": "1.6e-2", "box5": "3.2e-2"}


def series4(r):
    c = r["curve"]
    S = {"numbers": np.array([np.mean([e[f"held_{b}"] for b in BKT]) for e in c])}
    D = r["domains"]["dim"]
    keep, _, _ = dim_strict(D)
    H = np.array([[int(x) for x in e["hits_dim_held"]] for e in c], float) / D["eval_held"]["contexts"]
    S["dim"] = H[:, keep].mean(1)
    for d in ("cities", "countries"):
        S[d] = np.array([e[f"{d}_held_2way"] for e in c])
    for d in ("torus", "box5"):
        S[d] = np.array([e[TOY_KEY[d]] for e in c])
    return S, c


def gpu_names(d):
    f = d / "gpu.csv"
    if not f.exists():
        return []
    return sorted({ln.split(",")[1].strip() for ln in f.read_text().splitlines()[1:] if ln.count(",") >= 1})


V = {}
for cell, lr, pd in (("all6full", "1e-3", "numbers,dim,cities,countries,torus,box5"), ("all4", "2e-3", "numbers,dim,cities,countries")):
    for s in SEEDS:
        fs = sorted((OUT4 / f"pt6v2-{cell}-lr{lr}-s{s}").glob("*/run_whole_*.json"))
        assert len(fs) == 1, (cell, s, fs)
        r = json.loads(fs[0].read_text())
        assert r["curve"][-1]["step"] == 18309 and len(r["curve"]) == 75, (cell, s)
        a = r["args"]
        assert a["seed"] == s and abs(a["lr"] - float(lr)) < 1e-15 and a["prior_domains"] == pd, (cell, s)
        if cell == "all6full":
            assert a["window"] == "0.05,1.0"
        chk = json.loads((fs[0].parent.parent / "check.json").read_text())
        S, c = series4(r)
        V[(cell, s)] = {"S": S, "bpb": c[-1]["val_bits_per_byte"], "sha": c[-1].get("data_sha") == DB[("all6", P, s)]["data_sha"],
                        "pass": chk.get("pass"), "gpus": gpu_names(fs[0].parent.parent)}
print("V4 runs:", len(V), " all CHECKS pass:", all(v["pass"] for v in V.values()), " data_sha == pt6v2 all:", all(v["sha"] for v in V.values()))
print("GPUs:", {k: v["gpus"] for k, v in sorted(V.items())})

f3 = lambda v: 100 * float(np.mean(v[-3:]))   # noqa: E731
ar = lambda v: 100 * float(np.mean(v))   # noqa: E731
TT = stats.t.ppf(0.975, len(SEEDS) - 1)


def summ(x):
    x = np.asarray(x, float)
    m, h = x.mean(), TT * x.std(ddof=1) / math.sqrt(len(x))
    return f"{m:+6.1f} [{m - h:+6.1f}, {m + h:+6.1f}] wins {int((x > 0).sum())}/8"


def summ4(x):
    x = np.asarray(x, float)
    m, h = x.mean(), TT * x.std(ddof=1) / math.sqrt(len(x))
    return f"{m:+.4f} [{m - h:+.4f}, {m + h:+.4f}] ({100 * m / 1.21:+.2f}%) lower in {int((x < 0).sum())}/8"


print("\n[A] all6full@1e-3 (priors on 0.05 -> end) vs all6 window@1e-3 (0.05 -> 0.4), same lr, same seed")
for d in MANI:
    full = np.array([f3(V[("all6full", s)]["S"][d]) for s in SEEDS])
    win = np.array([f3(DB[("all6", P, s)]["S"][d]) for s in SEEDS])
    afull = np.array([ar(V[("all6full", s)]["S"][d]) for s in SEEDS])
    awin = np.array([ar(DB[("all6", P, s)]["S"][d]) for s in SEEDS])
    print(f"  {d:9s} final3 full {full.mean():5.1f} win {win.mean():5.1f}  diff {summ(full - win)} | area full {afull.mean():5.1f} win {awin.mean():5.1f} diff {summ(afull - awin)}")

print("\n[B] all6full final3 vs no prior: (i) none@N_m (area-best lr), (ii) the lr with the best 8-seed mean final3, (iii) per-seed oracle max over 6 lrs")
for d in MANI:
    full = np.array([f3(V[("all6full", s)]["S"][d]) for s in SEEDS])
    nm = np.array([f3(DB[("none", NM[d], s)]["S"][d]) for s in SEEDS])
    mf = {lr: np.mean([f3(DB[("none", lr, s)]["S"][d]) for s in SEEDS]) for lr in NONE_LRS}
    blr = max(mf, key=mf.get)
    bl = np.array([f3(DB[("none", blr, s)]["S"][d]) for s in SEEDS])
    orc = np.array([max(f3(DB[("none", lr, s)]["S"][d]) for lr in NONE_LRS) for s in SEEDS])
    print(f"  {d:9s} full {full.mean():5.1f} | (i) none@{NM[d]} {nm.mean():5.1f} diff {summ(full - nm)} | (ii) none@{blr} {bl.mean():5.1f} diff {summ(full - bl)}"
          f" | (iii) oracle {orc.mean():5.1f} diff {summ(full - orc)}")

print("\n[C] bpb (final val bits per byte; paired per seed)")
bf = np.array([V[("all6full", s)]["bpb"] for s in SEEDS])
bw = np.array([DB[("all6", P, s)]["bpb"] for s in SEEDS])
print(f"  all6full {bf.mean():.4f} (median {np.median(bf):.4f}); all6win@1e-3 {bw.mean():.4f}; full - win {summ4(bf - bw)}")
nb = {lr: np.array([DB[("none", lr, s)]["bpb"] for s in SEEDS]) for lr in NONE_LRS}
for lr in NONE_LRS:
    print(f"  none@{lr:6s} {nb[lr].mean():.4f} (median {np.median(nb[lr]):.4f})  full - none {summ4(bf - nb[lr])}  median diff {np.median(bf - nb[lr]):+.4f}")
bo = np.array([min(DB[("none", lr, s)]["bpb"] for lr in NONE_LRS) for s in SEEDS])
print(f"  per-seed best none bpb (oracle) {bo.mean():.4f}; full - oracle {summ4(bf - bo)}")

print("\n[D] all4@2e-3 (priors numbers, dim, cities, countries; window 0.05-0.4) vs all6@2e-3 and only-m@2e-3 (same lr, same seed)")
for d in ("numbers", "dim", "cities", "countries"):
    a4 = np.array([ar(V[("all4", s)]["S"][d]) for s in SEEDS])
    a6 = np.array([ar(DB[("all6", SAME, s)]["S"][d]) for s in SEEDS])
    om = np.array([ar(DB[(f"only-{d}", SAME, s)]["S"][d]) for s in SEEDS])
    g4 = np.array([f3(V[("all4", s)]["S"][d]) for s in SEEDS])
    g6 = np.array([f3(DB[("all6", SAME, s)]["S"][d]) for s in SEEDS])
    go = np.array([f3(DB[(f"only-{d}", SAME, s)]["S"][d]) for s in SEEDS])
    print(f"  {d:9s} AREA all4 {a4.mean():5.1f} all6 {a6.mean():5.1f} only {om.mean():5.1f} | h4 = all4 - only {summ(a4 - om)} | h6 = all6 - only {summ(a6 - om)} | h6 - h4 = all6 - all4 {summ(a6 - a4)}")
    print(f"  {'':9s} FINAL3 all4 {g4.mean():5.1f} all6 {g6.mean():5.1f} only {go.mean():5.1f} | all4 - only {summ(g4 - go)} | all6 - only {summ(g6 - go)} | all6 - all4 {summ(g6 - g4)}")
print("  torus / box5 without their own prior (all4) vs none@2e-3, only-dim@2e-3 and all6@2e-3 (area; final3)")
for d in ("torus", "box5"):
    a4 = np.array([ar(V[("all4", s)]["S"][d]) for s in SEEDS])
    nn = np.array([ar(DB[("none", SAME, s)]["S"][d]) for s in SEEDS])
    od = np.array([ar(DB[("only-dim", SAME, s)]["S"][d]) for s in SEEDS])
    oo = np.array([ar(DB[(f"only-{d}", SAME, s)]["S"][d]) for s in SEEDS])
    a6 = np.array([ar(DB[("all6", SAME, s)]["S"][d]) for s in SEEDS])
    g4 = np.array([f3(V[("all4", s)]["S"][d]) for s in SEEDS])
    gn = np.array([f3(DB[("none", SAME, s)]["S"][d]) for s in SEEDS])
    print(f"  {d:6s} area all4 {a4.mean():5.1f} none {nn.mean():5.1f} only-dim {od.mean():5.1f} only-{d} {oo.mean():5.1f} all6 {a6.mean():5.1f} | all4 - none {summ(a4 - nn)} | only-dim - none {summ(od - nn)}"
          f" | final3 all4 {g4.mean():5.1f} none {gn.mean():5.1f} diff {summ(g4 - gn)}")
b4 = np.array([V[("all4", s)]["bpb"] for s in SEEDS])
b6 = np.array([DB[("all6", SAME, s)]["bpb"] for s in SEEDS])
print(f"  bpb all4 {b4.mean():.4f} (median {np.median(b4):.4f}) all6@2e-3 {b6.mean():.4f} none@2e-3 {nb['2e-3'].mean():.4f}; all4 - all6 {summ4(b4 - b6)}; all4 - none@2e-3 {summ4(b4 - nb['2e-3'])} median {np.median(b4 - nb['2e-3']):+.4f}")
print("  per-seed all4 bpb:", np.round(b4, 4).tolist())
