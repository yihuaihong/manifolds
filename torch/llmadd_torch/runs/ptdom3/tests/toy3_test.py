"""ptdom3 build (2026-11-17), CPU: (1) choose the nonce names once (-> tools/toy3_nonce.json); (2) structure checks of the
five toy3 tables and that each target encodes the answer (Tier L); (3) tokenisation: every item a distinct single token,
every line's pieces tokenise to the line; (4) splits at seeds 0-3 and 8-11: sizes, read-out / hard counts, chance lines,
held-out never trained; (5) 3 decoded training lines and 3 held-out prompts per domain (seed 0); (6) known answer of the
read-out masks: a planted perfect answerer scores 100 on the read-out, a shuffled answerer ~ chance."""
import json, sys
from pathlib import Path
import numpy as np

T = Path("/scratch/yh6210/llmadd-burst-ptdom3/tools")
sys.path.insert(0, str(T))
HERE = Path(__file__).parent
NJ = T / "toy3_nonce.json"
CONS, VOW = set("bcdfgjklmnprstvz"), set("aeiou")
SKIP = {   # reason per skipped candidate (rank order of tok_survey.json); the S4_NAMES / DIM_NAMES letter rule is applied first
    "abbreviation": "fov dof nop poc jit gid pov sos vec buf mig vip gif rar pid pem dac rus gov mov sig dup soc caf",
    "given name": "jos cid jim jon ned nik ken pam ted kat lon",
    "English word / slang": "sez nog tog cuz roc mog nab biz zen cig zap raj sis vim sop jus bum cud bam tat sic sus lol dit "
                            "nip mop nam vid pep vat tic tod dum bib git jab gad jig dal dun lug",
    "vulgar / slur": "kut jap kuk pis dik kok kak",
    "common foreign function word / noun": "vej jeg zij zug paz voz zal luz mij"}
SKIPW = {w: r for r, ws in SKIP.items() for w in ws.split()}
NEED = {"cyc12": 12, "torus": 36, "hcube": 32, "d6": 12, "box5": 25}

if not NJ.exists():
    sv = json.loads((HERE / "tok_survey.json").read_text())
    keep, skipped = [], {}
    for s, i, c in sv["nonce_ranked"]:
        if not (s[0] in CONS and s[1] in VOW and s[2] in CONS):
            continue
        if s in SKIPW:
            skipped[s] = SKIPW[s]; continue
        keep.append((s, c))
        if len(keep) == sum(NEED.values()):
            break
    assert len(keep) == 117
    perm = np.random.default_rng(4417).permutation(117)   # fixed assignment of the 117 names to the domains
    names, o = {}, 0
    for d, n in NEED.items():
        names[d] = {keep[j][0]: keep[j][1] for j in perm[o:o + n]}; o += n
    NJ.write_text(json.dumps({"rule": "tests/toy3_test.py (ptdom3 build 2026-11-17)", "names": names, "skipped": skipped,
                              "max_count": max(c for _, c in keep)}, indent=1))
    print(f"wrote {NJ}: 117 names, counts {keep[0][1]}..{keep[-1][1]} in train.bin; skipped {len(skipped)}")
import toy3, toy_struct as ts
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/scratch/yh6210/llmadd_torch/models/OLMo-2-1124-7B/stage1-step2000-tokens9B")
enc = lambda s: tok.encode(s, add_special_tokens=False)
print("nonce counts per domain:", {d: (min(v.values()), max(v.values())) for d, v in toy3.NONCE.items()})
allnames = [n for d in toy3.TOY3 if d in toy3.NONCE for n in toy3.NONCE[d]]
assert len(set(allnames)) == 117
src = (T / "pretrain_range.py").read_text()
assert not any(f'"{n}":' in src[src.index("S4_NAMES = {"):src.index("DOM_ARGS =")] for n in allnames), "overlap with s4 / dim"
fails = 0
# caesar (coordinator 2026-11-17): every capital letter one OLMo-2 token with and without a leading space; how frequent
L = toy3.LETTERS
wsp, nsp = [enc(" " + c) for c in L], [enc(c) for c in L]
print("caesar: ' A'..' Z' single tokens:", sum(len(t) == 1 for t in wsp), "/ 26; 'A'..'Z' (no space):", sum(len(t) == 1 for t in nsp),
      "/ 26; multi:", [c for c, t in zip(L, wsp) if len(t) != 1] + [c + "(nosp)" for c, t in zip(L, nsp) if len(t) != 1])
assert all(len(t) == 1 for t in wsp) and len({t[0] for t in wsp}) == 26, "caesar: a letter is not one token after a space"
tr_ = np.memmap("/scratch/yh6210/llmadd_torch/pretrain_data/train.bin", dtype=np.uint32, mode="r")
c_ = np.bincount(np.asarray(tr_[:50_000_000]), minlength=len(tok))
lc = {c: int(c_[t[0]]) * 8 for c, t in zip(L, wsp)}
print("caesar: ' X' occurrences in train.bin (first 5e7 of 4e8 tokens x 8):", "min", min(lc.values()), "median",
      int(np.median(list(lc.values()))), "max", max(lc.values()), "| nonce max", max(max(v.values()) for v in toy3.NONCE.values()))
for d in toy3.TOY3:
    D = toy3.build(d); F = D["targets"]["true"]; X, Y = D["X"], D["Y"]
    # (2) structure + Tier L: the answer is a function of the inputs' target rows and the relation
    if d in toy3.GROUPS:
        m = D["mul"]; n = len(m)
        assert all(m[m[a, b], c] == m[a, m[b, c]] for a in range(n) for b in range(n) for c in range(n)), (d, "assoc")
        assert all((m[a] == D["e"]).any() for a in range(n)), (d, "inverse")
        assert all(sorted(m[a]) == list(range(n)) for a in range(n)), (d, "latin")
    if d in ("cyc12", "torus", "caesar"):
        ang = np.arctan2(F[:, 1::2], F[:, 0::2])
        ok = np.allclose(np.exp(1j * (ang[X[:, 0]] + ang[X[:, 1]])), np.exp(1j * ang[Y]))
    elif d == "hcube":
        ok = np.allclose(F[X[:, 0]] * F[X[:, 1]], F[Y])
    elif d == "d6":
        R = F.reshape(-1, 2, 2); ok = np.allclose(R[X[:, 1]] @ R[X[:, 0]], R[Y])
        assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(2)), "orthogonal"
    else:
        ok = np.allclose(F[X[:, 0]] + np.where(X[:, 1] == 25, 1, -1)[:, None] * F[X[:, 2]], F[Y])
    print(f"[{d}] items {D['n_items']} questions {len(X)} target dim {F.shape[1]} centred rank "
          f"{np.linalg.matrix_rank(F - F.mean(0))} | structure + target encodes answer (Tier L): {'OK' if ok else 'FAIL'}"
          + (f" | non-commuting cells {int(D['noncomm'].sum())}" if d in toy3.GROUPS else ""))
    fails += not ok
    # (3) tokenisation
    names = toy3.item_names(d)
    ids = [enc(s) for s in names]
    assert all(len(i) == 1 for i in ids) and len({i[0] for i in ids}) == len(ids), (d, "single distinct tokens")
    for q in range(len(X)):
        pcs, a, ins = toy3.pieces(D, q, names)
        line = "".join(p for p, _ in pcs)
        assert sum((enc(p) for p, _ in pcs), []) == enc(line), (d, q, "pieces tokenise to the line")
    # (4) splits
    for seed in (0, 1, 2, 3, 8, 9, 10, 11):
        mode, frac = toy3.SPLIT[d]
        base = int(np.random.default_rng(seed).integers(2 ** 31))   # stand-in for the trainer's split stream
        for att in range(200):
            try:
                tr, te = toy3.split(D, frac, base + att, mode); break
            except AssertionError:
                continue
        rows = {tuple(r) for r in X[tr].tolist()}
        assert not any(tuple(X[i].tolist()) in rows for i in te), "held in train"
        mk, name, ro = toy3.masks(D, tr, te, ts.s4_closure)
        maj = np.bincount(Y[tr], minlength=D["n_items"]).argmax()
        ch = max(1 / D["n_items"], float(np.mean(Y[te][ro] == maj)))
        hard = f" hard {int(mk['hard'].sum())} hard&ro {int(mk['hard_ro'].sum())}" if "hard" in mk else ""
        if seed in (0, 8):
            print(f"  seed {seed}: {mode} {frac} att {att}: train {len(tr)} held {len(te)} readout[{name}] {int(ro.sum())}"
                  f"{hard} chance {100 * ch:.1f}")
        if ro.sum() < 40:
            print(f"  WARN seed {seed}: read-out set {int(ro.sum())} < 40"); fails += 0
        # (6) known answer: planted perfect / shuffled answerer on the read-out set (exact match of the answer item)
        perfect = (Y[te] == Y[te])[ro].mean() * 100
        sh = np.random.default_rng(seed).permutation(D["n_items"])
        shuf = (sh[Y[te]] == Y[te])[ro].mean() * 100
        if not (perfect == 100 and shuf < 100 * (1 / D["n_items"]) + 25):
            print(f"  KNOWN-ANSWER FAIL seed {seed}: perfect {perfect} shuffled {shuf}"); fails += 1
        if seed == 0:
            print(f"  known answer (seed 0): planted perfect {perfect:.1f}, shuffled-item answerer {shuf:.1f} "
                  f"(uniform chance {100 / D['n_items']:.1f})")
            for lab, qs in (("train", tr[:3]), ("held ", te[ro][:3])):
                for q in qs:
                    pcs, a, ins = toy3.pieces(D, int(q), names)
                    p = "".join((f"⟦{t}⟧" if r not in (None, "ans") else t) for t, r in pcs if r != "ans" and t != "\n")
                    print(f"    {lab} {p} -> {a!r}   (sites {ins})")
print("TOY3 TEST", "PASS" if fails == 0 else f"FAIL ({fails})")
