"""V2 build (AUDIT.md Amendment V2): patch make_jobs_pt6v2.py (--extra_none 6.4e-2, CELLS.json), readout_pt6v2.py (none grid from
CELLS.json, top-lr flag on the actual top, Tier L reading line for every ALLOWED domain), submit_pt6v2.sh (--extra_none passthrough,
CELLS.json check), tests/run_refs_pt6v2.sh (placeholder mode for the auto_pt6v2.sh poll-and-grep test). Every replacement asserts its
occurrence count. Run once (refuses when already patched)."""
import hashlib
from pathlib import Path

B = Path("/scratch/yh6210/llmadd_torch")
R = B / "runs/pt6v2"


def sha12(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def rep(t, old, new, n=1):
    assert t.count(old) == n, (t.count(old), old[:120])
    return t.replace(old, new)


# ------------------------------------------------------------------ make_jobs_pt6v2.py
G = B / "make_jobs_pt6v2.py"
assert sha12(G) == "2f10290f95b2", "already patched?"
t = G.read_text()
t = rep(t, 'assert MJ.HW["b200"]["ckpt"] == CKPT and all(lr in LADDER for lr in NONE_LRS + (P_LR, SAME_LR))\n',
        'assert MJ.HW["b200"]["ckpt"] == CKPT and all(lr in LADDER for lr in NONE_LRS + (P_LR, SAME_LR))\n'
        'EXTRA_NONE_OK = ("6.4e-2",)   # AUDIT.md Amendment V2 (conditional; opt-in --extra_none 6.4e-2): none@6.4e-2 for all 8 seeds\n'
        'assert all(x in LADDER and float(x) > float(max(NONE_LRS, key=float)) for x in EXTRA_NONE_OK)\n')
t = rep(t, '''def cells(doms):
    """-> [(arm, lr)] of the 13 cells of a seed (AUDIT.md "Design" + Amendment V1)."""
    c = [("all6", P_LR), ("all6", SAME_LR)] + [("none", lr) for lr in NONE_LRS] + [(f"only-{m}", SAME_LR) for m in manifolds(doms)]
    assert len(c) == 13 and len(set(c)) == 13
    return c''', '''def cells(doms, extra=None):
    """-> [(arm, lr)] of the 13 cells of a seed (AUDIT.md "Design" + Amendment V1); 14 with extra = '6.4e-2' (Amendment V2, opt-in:
    none@6.4e-2 right after the none grid)."""
    assert extra is None or extra in EXTRA_NONE_OK, f"--extra_none {extra}: one of {EXTRA_NONE_OK}"
    c = ([("all6", P_LR), ("all6", SAME_LR)] + [("none", lr) for lr in NONE_LRS + ((extra,) if extra else ())]
         + [(f"only-{m}", SAME_LR) for m in manifolds(doms)])
    nc = 13 + bool(extra)
    assert len(c) == nc and len(set(c)) == nc
    return c''')
t = rep(t, 'COMP_FILE = R / "COMPOSITION.json"\n',
        'COMP_FILE = R / "COMPOSITION.json"\n'
        'CELLS_FILE = R / "CELLS.json"   # AUDIT.md Amendment V2: the cell set of the written batch (read by readout_pt6v2.py / submit_pt6v2.sh)\n')
a, b_ = t.split("def plan(doms, alloc):")
p, m = b_.split("def main():")
p = rep(p, "    nc = len(cells(doms))\n    assert nc == 13\n", "    nc = len(cells(doms, extra))\n    assert nc == 13 + bool(extra)\n")
p = p.replace("cells(doms)", "cells(doms, extra)")
m = rep(m, '''    alloc = parse_alloc(al_w) if al_w else None
    if "--plan" in sys.argv:
        plan(doms, alloc)''', '''    alloc = parse_alloc(al_w) if al_w else None
    extra = opt("--extra_none")
    assert ("--extra_none" in sys.argv) == (extra in EXTRA_NONE_OK), f"--extra_none {extra}: one of {EXTRA_NONE_OK} (AUDIT.md Amendment V2)"
    if extra:
        print(f"extra_none: {extra} (AUDIT.md Amendment V2: none@{extra} for all 8 seeds)")
    if "--plan" in sys.argv:
        plan(doms, alloc, extra)''')
m = rep(m, "        for a, lr in cells(doms):\n            dev = alloc[s]", "        for a, lr in cells(doms, extra):\n            dev = alloc[s]")
m = rep(m, "    assert len(jobs) == 105 and len({j[0] for j in jobs}) == 105   # 13 x 8 + the canary (Amendment V1)\n", '''    nj = 8 * len(cells(doms, extra)) + 1
    assert len(jobs) == nj == (113 if extra else 105) and len({j[0] for j in jobs}) == nj   # 13 (V2: 14) x 8 + the canary
    manifest = {"design": PREFIX, "domains_arg": domains_arg(doms), "none_lrs": list(NONE_LRS) + ([extra] if extra else []),
                "extra_none": extra, "cells": [list(c_) for c_ in cells(doms, extra)], "n_runs": nj - 1, "seeds": list(SEEDS),
                "canary": list(CANARY)}
    subm = sorted(p_.name for p_ in R.glob(f"SUBMITTED_{PREFIX}-*"))
    m_disk = json.loads(CELLS_FILE.read_text()) if CELLS_FILE.exists() else None
    m_state = "missing" if m_disk is None else ("same" if m_disk == manifest else "differ")
    if "--write" in sys.argv and m_state != "same":
        assert not subm, f"{CELLS_FILE} is {m_state} but {len(subm)} jobs were submitted: the cell set is frozen once a job is submitted"
        if CELLS_FILE.exists():
            CELLS_FILE.chmod(0o644)
        CELLS_FILE.write_text(json.dumps(manifest, indent=1) + "\\n")
        CELLS_FILE.chmod(0o444)
        print(f"written {CELLS_FILE}: {len(manifest['cells'])} cells x 8 seeds, none grid {manifest['none_lrs']}")
        m_state = "same"
''')
m = rep(m, '        print(f"disk: unsubmitted job files {st_}")\n',
        '        print(f"disk: unsubmitted job files {st_}")\n'
        '        print(f"cells manifest: {m_state} ({CELLS_FILE.name}; {len(manifest[\'cells\'])} cells x 8 seeds, none grid {manifest[\'none_lrs\']})")\n')
m = rep(m, '        assert RO_FILE.exists(), "--submit: runs/pt6v2/readout_pt6v2.py must exist (the canary calls it)"\n',
        '        assert RO_FILE.exists(), "--submit: runs/pt6v2/readout_pt6v2.py must exist (the canary calls it)"\n'
        '        assert m_state == "same", f"--submit: {CELLS_FILE} is {m_state} (run --check --write with the same --extra_none first)"\n')
t = a + "def plan(doms, alloc, extra=None):" + p + "def main():" + m
G.write_text(t)
gsha = sha12(G)
print("make_jobs_pt6v2.py", gsha)

# ------------------------------------------------------------------ readout_pt6v2.py
F = R / "readout_pt6v2.py"
assert sha12(F) == "f4bbc4de583f", "already patched?"
t = F.read_text()
t = rep(t, 'MJ2_FILE, MJ2_SHA = B / "make_jobs_pt6v2.py", "2f10290f95b2"   # frozen after AUDIT.md Amendment V1 (V1 item 5)',
        f'MJ2_FILE, MJ2_SHA = B / "make_jobs_pt6v2.py", "{gsha}"   # frozen after the V2 build (AUDIT.md "V2 build"; V1: 2f10290f95b2)')
t = rep(t, "assert TOP_LR == max(NONE_LRS, key=float) and BOT_LR == min(NONE_LRS, key=float)\n", '''assert TOP_LR == max(NONE_LRS, key=float) and BOT_LR == min(NONE_LRS, key=float)
EXTRA_NONE = MJ2.EXTRA_NONE_OK
assert EXTRA_NONE == ("6.4e-2",)   # AUDIT.md Amendment V2 (conditional, opt-in)
CELLS_FILE = MJ2.CELLS_FILE


def none_grid():
    """The none lr grid of the written batch: CELLS.json (make_jobs_pt6v2.py --write) or, without it, the default 5-lr grid."""
    if not CELLS_FILE.exists():
        return NONE_LRS
    g = tuple(json.loads(CELLS_FILE.read_text())["none_lrs"])
    assert g in (NONE_LRS,) + tuple(NONE_LRS + (x,) for x in EXTRA_NONE), f"{CELLS_FILE}: none grid {g} is not a registered grid"
    return g


def top_flag(lr):
    return f"none at its top lr {lr}"
''')
t = rep(t, 'LOW_FLAG = "prior at its lowest lr"\n', '''LOW_FLAG = "prior at its lowest lr"
assert TOP_FLAG == top_flag(TOP_LR)
# Tier L (AUDIT.md "V2 build"): the prior's target carries the answers, so a HELPS reads "given the structure, the model can use it".
# Sources: ptdom AUDIT 145-156 (anchors yes, months no), ptdom2 AUDIT 81 (Part R six + countries / people / cities), ptdom3 AUDIT 14-24
# (toy3: all yes) + tests/byteid_23439548.log (caesar), pt6comb AUDIT 232-235 (s4 / dim; the tied-row caveat).
TIER_L = {"s4": True, "dim": True, "cities": True, "countries": True, "people": True, "months": False,
          "stars": True, "landmarks": True, "compass": True, "prefixes": True, "ranks": True, "scalar_adj": True,
          "cyc12": True, "torus": True, "hcube": True, "d6": True, "box5": True, "caesar": True}
NONCE_TIED = ("s4", "dim", "cyc12", "torus", "hcube", "d6", "box5")
assert set(TIER_L) == set(MJ2.ALLOWED) and set(NONCE_TIED) <= set(TIER_L)
''')
t = rep(t, "             \"interfere' (A2)\")   # = runs/pt6comb/readout.txt's reading line, verbatim (V1 item 4)\n",
        "             \"interfere' (A2)\")   # = runs/pt6comb/readout.txt's reading line, verbatim (V1 item 4)\n"
        "Q_TAIL = (\"NOT SHOWN / INTERFERENCE at 8 seeds are statements about this rule, never 'no effect' / 'harm' in general; 'interference \"\n"
        "          \"not detected' is never 'shown not to interfere' (A2)\")\n"
        "assert Q_READING.endswith(\"; \" + Q_TAIL)\n\n\n"
        "def reading_line(mani):\n"
        "    \"\"\"The composition's Tier L reading line (V2 build; pt6comb's line generalised to every ALLOWED domain, same tail).\"\"\"\n"
        "    doms = [d for d in mani if d != \"numbers\"]\n"
        "    tl, nl, nt = ([d for d in doms if TIER_L.get(d)], [d for d in doms if TIER_L.get(d) is False], [d for d in doms if d not in TIER_L])\n"
        "    s = (\"reading: comparisons\" + \"\".join(f\", {d}\" for d in tl) + \" are Tier L (the prior's target carries the answers: a HELPS \"\n"
        "         \"reads 'given the structure, the model can use it')\")\n"
        "    if nl:\n"
        "        s += f\"; {' / '.join(nl)} not Tier L (the target does not carry the answers: a HELPS there is not that reading)\"\n"
        "    if nt:\n"
        "        s += f\"; {' / '.join(nt)}: Tier L not registered (not a pt6v2 candidate)\"\n"
        "    nz = [d for d in doms if d in NONCE_TIED]\n"
        "    if nz:\n"
        "        s += f\"; {' / '.join(nz)} items are single nonce tokens whose site is the read-out candidate's own (tied) embedding row (AUDIT 'Caveats')\"\n"
        "    if \"caesar\" in doms:\n"
        "        s += \"; caesar items are real letter tokens (its own block)\"\n"
        "    return s + \"; \" + Q_TAIL\n")
t = rep(t, "    pick = lambda d_: max(none_lrs, key=lambda lr: (round(d_[lr], 9), float(lr)))   # noqa: E731\n",
        "    pick = lambda d_: max(none_lrs, key=lambda lr: (round(d_[lr], 9), float(lr)))   # noqa: E731\n"
        "    TOP_, BOT_ = (max(none_lrs, key=float), min(none_lrs, key=float)) if len(none_lrs) > 1 else (None, None)   # V2: the grid's edges\n")
t = rep(t, 'v["flags"] = ([TOP_FLAG] if N == TOP_LR else []) + ([BOT_FLAG] if N == BOT_LR else []) + [LOW_FLAG]',
        'v["flags"] = ([top_flag(TOP_)] if N == TOP_ else []) + ([BOT_FLAG] if N == BOT_ else []) + [LOW_FLAG]')
t = rep(t, 'def spec_v2(doms):\n', 'def spec_v2(doms, none_lrs=NONE_LRS):\n')
t = rep(t, '"lrs": set(NONE_LRS) | {P_LR, SAME_LR}, "n_evals"', '"lrs": set(none_lrs) | {P_LR, SAME_LR}, "n_evals"')
t = rep(t, '    L += [f"  {Q_SAME_SEEDS} (none grid {NL})", f"  {Q_READING}"]', '    L += [f"  {Q_SAME_SEEDS} (none grid {NL})", f"  {reading_line(mani)}"]')
t = rep(t, '''    doms, src = composition(comp)
    spec = spec_v2(doms)
    R_ = load_v2(root or (V2 / "out"), spec, guard)
    probs = PC.integrity(R_, set(SEEDS))
    res, txt = readout_from(R_, spec["mani"], src, probs, allow_partial)''', '''    doms, src = composition(comp)
    NL = none_grid()   # V2: CELLS.json's none grid (default 5 lrs; + 6.4e-2 under Amendment V2)
    spec = spec_v2(doms, NL)
    root_ = Path(root or (V2 / "out"))
    R_ = load_v2(root_, spec, guard)
    probs = list(PC.integrity(R_, set(SEEDS)))
    rx_ = re.compile(rf"^{PREFIX}-none-lr([0-9.e-]+)-s\\d+")
    probs += [f"{p_.name}: a none lr outside the CELLS.json grid {list(NL)}" for p_ in sorted(root_.glob(f"{PREFIX}-none-lr*"))
              if rx_.match(p_.name) and rx_.match(p_.name)[1] not in NL]
    res, txt = readout_from(R_, spec["mani"], src, probs, allow_partial, none_lrs=NL)''')
t = rep(t, '    ("the top-lr flag on the wrong lr", "[TOP_FLAG] if N == TOP_LR else []", "[TOP_FLAG] if N == none_lrs[0] else []"),\n',
        '    ("the top-lr flag on the wrong lr", "[top_flag(TOP_)] if N == TOP_ else []", "[top_flag(TOP_)] if N == none_lrs[0] else []"),\n'
        '    ("V2: the top-lr flag fixed at 3.2e-2", "(max(none_lrs, key=float), min(none_lrs, key=float)) if len", "(TOP_LR, min(none_lrs, key=float)) if len"),\n'
        '    ("V2: N_m ignores none@6.4e-2", "pick = lambda d_: max(none_lrs, key=", "pick = lambda d_: max(none_lrs[:5], key="),\n'
        '    ("V2: months read as Tier L", \'"months": False\', \'"months": True\'),\n')
t = rep(t, '    ("V1: the low-edge flag on the top lr", "([BOT_FLAG] if N == BOT_LR else [])", "([BOT_FLAG] if N == TOP_LR else [])"),\n',
        '    ("V1: the low-edge flag on the top lr", "([BOT_FLAG] if N == BOT_ else [])", "([BOT_FLAG] if N == TOP_ else [])"),\n')
anchor = '    chk("headline counts HELPS + no interference: 3 of 6", res["headline"]["k"] == 3 and res["headline"]["list"] == [m0, m4, m5])\n'
t = rep(t, anchor, anchor + '''
    # V2 (Amendment V2, opt-in none@6.4e-2): N_m over every none lr present; the top-lr flag on the grid's actual top
    def mod_v2(L):
        L["none", "6.4e-2"] = {m: [(72 if m == m2 else 40) + e for e in E_] for m in mani}
        L["none", "3.2e-2"][m3] = [71 + e for e in E_]
    rv2, tv2 = readout_from(world(mani, mod_v2), mani, none_lrs=NONE_LRS + EXTRA_NONE)
    chk("V2: N_m over the 6-lr grid (m2 -> 6.4e-2, m3 -> 3.2e-2, m0 -> 1.6e-2)",
        (rv2["m"][m2]["N"], rv2["m"][m3]["N"], rv2["m"][m0]["N"]) == ("6.4e-2", "3.2e-2", "1.6e-2"))
    chk("V2: 'none at its top lr 6.4e-2' iff N_m = 6.4e-2; 3.2e-2 no longer flagged as the top",
        top_flag("6.4e-2") in rv2["m"][m2]["flags"] and all(not any(f.startswith("none at its top lr") for f in rv2["m"][m]["flags"])
                                                             for m in mani if m != m2))
    chk("V2: the report shows the 6-lr none grid", "6.4e-2" in tv2.split("HEADLINE")[0])
    rl = reading_line(mani)
    chk("V2: placeholder reading line: comparisons + the 5 domains Tier L; dim / hcube nonce-tied; pt6comb's tail",
        rl.startswith("reading: comparisons, ") and all(d in rl.split(" are Tier L")[0] for d in mani if d != "numbers")
        and "dim / hcube items are single nonce tokens" in rl and rl.endswith("; " + Q_TAIL) and "not Tier L" not in rl)
    chk("V2: every ALLOWED domain: Tier L iff not months; the line names it",
        all((f", {d} are Tier L" in reading_line(("numbers", d))) == (d != "months") for d in MJ2.ALLOWED)
        and "months not Tier L" in reading_line(("numbers", "months")))
''')
t = rep(t, "0 < txt.index(Q_SAME_SEEDS) < hl and 0 < txt.index(Q_READING) < hl)", "0 < txt.index(Q_SAME_SEEDS) < hl and 0 < txt.index(reading_line(mani)) < hl)")
F.write_text(t)
print("readout_pt6v2.py", sha12(F))

# ------------------------------------------------------------------ submit_pt6v2.sh
S = R / "submit_pt6v2.sh"
assert sha12(S) == "3f07090c34da", "already patched?"
t = S.read_text()
t = rep(t, "#   bash runs/pt6v2/submit_pt6v2.sh [--alloc 'b200:16,17,18 l40s:19,20,21 a100:22,23'] [--canary_dev l40s|a100] [--go]\n",
        "#   bash runs/pt6v2/submit_pt6v2.sh [--alloc 'b200:16,17,18 l40s:19,20,21 a100:22,23'] [--canary_dev l40s|a100] [--extra_none 6.4e-2] [--go]\n"
        "# V2 build: --extra_none 6.4e-2 (AUDIT.md Amendment V2) is passed to make_jobs; refuses unless runs/pt6v2/CELLS.json = this cell set.\n")
t = rep(t, "CDEV=()\n", "CDEV=()\nEXN=()\n")
t = rep(t, '    --canary_dev) CDEV=(--canary_dev "$2"); shift 2;;\n',
        '    --canary_dev) CDEV=(--canary_dev "$2"); shift 2;;\n    --extra_none) EXN=(--extra_none "$2"); shift 2;;\n')
t = rep(t, '"$PY" make_jobs_pt6v2.py --alloc "${AL[@]}" "${CDEV[@]}" --check --print --disk',
        '"$PY" make_jobs_pt6v2.py --alloc "${AL[@]}" "${CDEV[@]}" "${EXN[@]}" --check --print --disk')
t = rep(t, "first=$(grep -m1", "grep -q '^cells manifest: same ' \"$CK\" \\\n  || fail \"runs/pt6v2/CELLS.json missing or differing from this cell set (--write and this call must agree on --extra_none; $CK)\"\nfirst=$(grep -m1")
t = rep(t, 'echo "alloc: $ALLOC"\n', 'echo "alloc: $ALLOC; extra_none: ${EXN[1]:-none}"\n')
t = rep(t, '  "$PY" make_jobs_pt6v2.py --alloc "${AL[@]}" "${CDEV[@]}" --submit', '  "$PY" make_jobs_pt6v2.py --alloc "${AL[@]}" "${CDEV[@]}" "${EXN[@]}" --submit')
S.write_text(t)
print("submit_pt6v2.sh", sha12(S))

# ------------------------------------------------------------------ tests/run_refs_pt6v2.sh
Q = R / "tests/run_refs_pt6v2.sh"
assert sha12(Q) == "c82e56ddfd7d", "already patched?"
t = Q.read_text()
t = rep(t, "else\n  mix=comp; extra=()\nfi\n", "elif [ \"$m\" = placeholder ]; then\n  mix=placeholder; extra=()   # V2 build: the comp job list on the "
        "placeholder (tests auto_pt6v2.sh's poll-and-grep)\nelse\n  mix=comp; extra=()\nfi\n")
Q.write_text(t)
print("run_refs_pt6v2.sh", sha12(Q))
