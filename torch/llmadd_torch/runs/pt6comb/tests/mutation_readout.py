"""pt6comb test: the self-test and the real-data known answers BITE: each mutation of readout_pt6comb.py (a copy in a temp dir, the
module re-pointed by PYTHONPATH order) must make `--selftest` or tests/real_known.py fail; the unmutated copy must pass both.
-> mutation_readout.out"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = (HERE / "readout_pt6comb.py").read_text()
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
MUT = [  # (name, old, new) exact replacements, each must occur once
    ("wins 6 -> 5", "N_EVALS, ALPHA, MIN_WINS, INT_FRAC = 75, 0.05, 6, 0.5", "N_EVALS, ALPHA, MIN_WINS, INT_FRAC = 75, 0.05, 5, 0.5"),
    ("alpha 0.05 -> 0.10", "N_EVALS, ALPHA, MIN_WINS, INT_FRAC = 75, 0.05, 6, 0.5", "N_EVALS, ALPHA, MIN_WINS, INT_FRAC = 75, 0.10, 6, 0.5"),
    ("interference margin 0.5 -> 0", "N_EVALS, ALPHA, MIN_WINS, INT_FRAC = 75, 0.05, 6, 0.5", "N_EVALS, ALPHA, MIN_WINS, INT_FRAC = 75, 0.05, 6, 0.0"),
    ("HELPS two-sided", "    p_help, wins = t_one(g), int((g > 0).sum())", "    p_help, wins = RO.t_test(g)[1], int((g > 0).sum())"),
    ("interference p in the wrong direction", "    p_int = t_one(-h)", "    p_int = t_one(h)"),
    ("NO INTERFERENCE needs both clauses (and)", "    noint = bool(by_m or by_p)", "    noint = bool(by_m and by_p)"),
    ("margin sign flipped", "    margin = -INT_FRAC * float(u.mean()) if u_pos else None", "    margin = INT_FRAC * float(u.mean()) if u_pos else None"),
    ("HELPS without the wins condition", "    helps = bool(p_help < ALPHA and wins >= MIN_WINS)", "    helps = bool(p_help < ALPHA)"),
    ("Holm applied to the verdict", "        res[\"m\"][m][\"helps_holm\"] = bool(ph < ALPHA and res[\"m\"][m][\"wins_g\"] >= MIN_WINS)",
     "        res[\"m\"][m][\"helps_holm\"] = bool(ph < ALPHA and res[\"m\"][m][\"wins_g\"] >= MIN_WINS)\n        res[\"m\"][m][\"helps\"] = res[\"m\"][m][\"helps_holm\"]"),
    ("headline counts HELPS only", "        ok = [m for m in MANI if res[\"m\"][m][\"helps\"] and res[\"m\"][m][\"noint\"]]",
     "        ok = [m for m in MANI if res[\"m\"][m][\"helps\"]]"),
    ("gate on the 8 seeds pooled (no halves)", "    halves = {\"8-11\": [s for s in seeds if s <= 11], \"12-15\": [s for s in seeds if s >= 12]}",
     "    halves = {\"8-15\": list(seeds)}"),
    ("gate ignored at the confirmation", "    res[\"gate_failed\"] = any(not g[\"ok\"] for g in g_h.values())", "    res[\"gate_failed\"] = False"),
    ("window 916 -> 1001", "WIN = (916, 7323)", "WIN = (1001, 7323)"),
    ("selection objective = numbers only", "    return float(np.mean([r[\"area\"][m] for m in MANI]))", "    return float(r[\"area\"][\"numbers\"])"),
    ("extension trigger argmax only (all6)", "        if top_a in (P_, ua[\"argmax\"]) and top_a not in bad:", "        if top_a == ua[\"argmax\"] and top_a not in bad:"),
    ("no fixB", "        if not (min(rng) <= float(P_) <= max(rng)):", "        if False:"),
    ("NOT FINAL ignored", "        if nf:\n            return dict(base, status=\"NOT FINAL\"", "        if False:\n            return dict(base, status=\"NOT FINAL\""),
    ("gate exclusion ignored in the selection", "    ua = select_unit(R_, \"all6\", lrs[\"all6\"], bad)", "    ua = select_unit(R_, \"all6\", lrs[\"all6\"], [])"),
    ("none's lower edge not extended", "        if lo_n in (N_, un[\"argmax\"]):\n            E.append((\"none\", M6.EXT_DOWN[\"none\"]))", "        pass"),
    ("A6 floor dropped", "                if x[\"geo\"] < GEO_FLOOR:", "                if False:"),
    ("A6 inactive-numbers check dropped", "    if \"numbers\" not in pri and (any(e.get(\"geo\") not in (None, 0) for e in c) or any(\"gnorm_prior_numbers\" in e for e in c)):",
     "    if False:"),
    ("s4 metric = held_fresh_nonid (not s4_hard)", "        return [v for _, v in RS.s4_hard_series(r)]",
     "        return [e[\"s4_held_fresh_nonid\"] for e in r[\"curve\"]]"),
    ("dim metric = no_negtwin_nonid (not strict)", "        return [v for _, v in RS.strict_series(r)]",
     "        return [e[\"dim_held_no_negtwin_nonid\"] for e in r[\"curve\"]]"),
    ("numbers metric = b33 only", "        return [float(np.mean([e[f\"held_{b}\"] for b in BKT])) for e in c]", "        return [e[\"held_b33\"] for e in c]"),
    ("A2: h against all6@P (not all6@2e-3)", '        v = manifold_verdict(dif("all6", "none", m), dif("all62", o, m) if pair_ok else None, dif(o, "none2", m) if pair_ok else None)',
     '        v = manifold_verdict(dif("all6", "none", m), dif("all6", o, m) if pair_ok else None, dif(o, "none2", m) if pair_ok else None)'),
    ("A2: u against none@N (not none@2e-3)", '        v = manifold_verdict(dif("all6", "none", m), dif("all62", o, m) if pair_ok else None, dif(o, "none2", m) if pair_ok else None)',
     '        v = manifold_verdict(dif("all6", "none", m), dif("all62", o, m) if pair_ok else None, dif(o, "none", m) if pair_ok else None)'),
    ("A2: the margin clause also when u <= 0", "    u_pos = float(u.mean()) > 0\n", "    u_pos = True\n"),
    ("A2: no same-lr pair in the C list", '    for c in [("none", N_), ("all6", P_), ("none", SINGLE_LR), ("all6", SINGLE_LR)] + [(f"only-{m}", SINGLE_LR) for m in MANI]:',
     '    for c in [("none", N_), ("all6", P_)] + [(f"only-{m}", SINGLE_LR) for m in MANI]:'),
    ("A2: the same-lr pair not merged (duplicates)", "        if c not in out:\n            out.append(c)", "        out.append(c)"),
    ("A2: HELPS on the same-lr gain", '        v = manifold_verdict(dif("all6", "none", m), dif("all62", o, m) if pair_ok else None, dif(o, "none2", m) if pair_ok else None)',
     '        v = manifold_verdict(dif("all62", "none2", m), dif("all62", o, m) if pair_ok else None, dif(o, "none2", m) if pair_ok else None)'),
    ("A3 boundary: margin '>=' -> '>'", "    by_m = u_pos and float(h.mean()) >= margin", "    by_m = u_pos and float(h.mean()) > margin"),
    ("A3 boundary: u > 0 -> u >= 0", "    u_pos = float(u.mean()) > 0\n", "    u_pos = float(u.mean()) >= 0\n"),
    ("A3 boundary: p >= .05 -> > .05", "    by_p = p_int >= ALPHA", "    by_p = p_int > ALPHA"),
    ("A3: the same-lr pair not in the missing list", "    pair_ok = not miss_pair", "    pair_ok = True"),
    ("A3: the pair's missing runs refuse HELPS (strict)", '    miss = [f"{a}@{lr} s{s}" for a, lr in core for s in CONF if not vok((a, lr), s)]',
     '    miss = [f"{a}@{lr} s{s}" for a, lr in core + pair for s in CONF if not vok((a, lr), s)]'),
    ("A3: table word back to NO INTERFERENCE", '("interference not detected" if v["noint"] else "INTERFERENCE"))',
     '("NO INTERFERENCE" if v["noint"] else "INTERFERENCE"))'),
    ("A3: the HELPS gate on all6@2e-3 (not all6@P)", '    g_h = {k: RO.cell_gate([get("all6", s) for s in v]) for k, v in halves.items() if v}',
     '    g_h = {k: RO.cell_gate([get("all62", s) for s in v]) for k, v in halves.items() if v}'),
    ("A3: the all6@2e-3 gate ignored", '        res["gate_same_lr_failed"] = any(not g["ok"] for g in g_2.values())', '        res["gate_same_lr_failed"] = False'),
    ("A3: the stale note back", '"conservative for HELPS; the interference contrast is at 2e-3 whatever P is, A2)")',
     '"conservative for HELPS and for NO INTERFERENCE)")'),
    ("integrity: data identity dropped", "        if len({tuple(r[\"data_sha\"]) for r in rs}) != 1:", "        if False:"),
]


def run(dirp):
    env = dict(os.environ, PYTHONPATH=f"{dirp}:/scratch/yh6210/llmadd_torch/pylib", PYTHONDONTWRITEBYTECODE="1")
    st = subprocess.run([PY, str(Path(dirp) / "readout_pt6comb.py"), "--selftest"], capture_output=True, text=True, env=env)
    # real_known.py imports readout_pt6comb from HERE (its sys.path insert): run a copy that imports the mutated one first
    rk = (HERE / "tests" / "real_known.py").read_text().replace("sys.path.insert(0, str(HERE))", f"sys.path.insert(0, {str(dirp)!r})", 1)
    (Path(dirp) / "real_known_m.py").write_text(rk)
    kn = subprocess.run([PY, str(Path(dirp) / "real_known_m.py")], capture_output=True, text=True, env=env)
    last = lambda o: (o.stdout.strip().splitlines() or [o.stderr.strip()[-200:]])[-1]
    return last(st), last(kn)


def prep(text):
    d = tempfile.mkdtemp()
    (Path(d) / "readout_pt6comb.py").write_text(text)
    # the copy resolves HERE = its own dir: give it the snapshot and the picks / job files it reads (read-only links)
    for x in ("code",):
        os.symlink(HERE / x, Path(d) / x)
    for f in HERE.glob("job_pt6comb-*.sbatch"):
        os.symlink(f, Path(d) / f.name)
    return d


res = []
d0 = prep(SRC)
s0, k0 = run(d0)
ok0 = s0.startswith("SELFTEST PASS") and k0.startswith("REAL KNOWN PASS")
res.append(ok0)
print(f"{'PASS' if ok0 else 'FAIL'} control (unmutated): {s0} | {k0}")
shutil.rmtree(d0)
for name, old, new in MUT:
    assert SRC.count(old) == 1, (name, SRC.count(old))
    d = prep(SRC.replace(old, new))
    s, k = run(d)
    caught = not (s.startswith("SELFTEST PASS") and k.startswith("REAL KNOWN PASS"))
    res.append(caught)
    print(f"{'PASS' if caught else 'FAIL'} mutation '{name}': {'caught' if caught else 'NOT CAUGHT'} ({s} | {k})")
    shutil.rmtree(d)
print(f"MUTATION READOUT {'PASS' if all(res) else 'FAIL'} {sum(res[1:])}/{len(MUT)} mutations caught" + ("" if res[0] else " (control FAILED)"))
