"""Unit tests of the ptdom task generators (Rev E of tools/pretrain_range.py, on runs/ptdom/code):
  A. the anchor splits re-derived independently (ft36.pair_questions + ft36.split on the trainer's registered stream
     SeedSequence(4200 + seed).spawn(16)[1 + i]) = the trainer's (split_seed, n_held, the held questions of a run JSON);
     leakage by key: no held option pair {b, c} in any training question (either order, any anchor), both orders of every
     held comparison held, no held prompt string = a training prompt; seed-specific splits; months: the rb6 tier-S class
     split (cyclic_split on stream 4100) re-derived = the run JSON's held questions, no held (item, k mod 12) class trained
  B. prior-site decoding on the 53M CPU dry logs: every marked site (⟦ ⟧) of an anchor line is the LAST token of an item
     name of that domain (the text up to and including it ends with the name; the next character is not a letter / digit);
     months sites are month names; every anchor line in a training row has 3 sites, every months line 1
  C. the dry logs: 3 decoded training rows with sites, >= 3 eval rows per domain, the other-option rows, the site-rule lines
    python test_generators.py <run JSON with full eval sets> <run JSON of the smoke> <dry log seed 0> <dry log seed 1>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

PIN = Path("/scratch/yh6210/llmadd_torch/runs/ptdom/code/tools")
sys.path.insert(0, str(PIN))
import ft36                                 # noqa: E402
import pretrain_range as PR                 # noqa: E402  (module-level helpers only: cyclic_split, ANCHOR, constants)
from manifold_domains import domains        # noqa: E402

full_json, smoke_json, dry0, dry1 = sys.argv[1:5]
MD = domains()
ok = []


def T(name, cond, val=None):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val is not None else ""))


def derive(d, seed):
    qs = ft36.pair_questions(d, MD[d])
    keys = [q["key"] for q in qs]
    sr = np.random.default_rng(np.random.SeedSequence(PR.ANC_STREAM + seed).spawn(16)[1 + PR.ANCHOR.index(d)])
    ss = int(sr.integers(2 ** 31))
    hk = ft36.split(keys, PR.ANC_HELD, ss)
    held = [i for i, k in enumerate(keys) if k in hk]
    train = [i for i, k in enumerate(keys) if k not in hk]
    return qs, ss, held, train


# ---------------------------------------------------------------- A. splits and leakage by key
J = json.loads(Path(full_json).read_text())
S = json.loads(Path(smoke_json).read_text())
seed = J["args"]["seed"]
assert S["args"]["seed"] == seed == 0
for d in PR.ANCHOR:
    qs, ss, held, train = derive(d, seed)
    D = J["domains"][d]
    T(f"[{d}] split_seed / n_held / n_train / n_held_keys = the independent derivation",
      (D["split_seed"], D["n_held"], D["n_train"], D["n_held_keys"]) == (ss, len(held), len(train), len({qs[i]["key"] for i in held})),
      (D["split_seed"], D["n_held"], D["n_train"]))
    T(f"[{d}] the run's held eval questions = the derived held set (full run JSON), and its first 40 (smoke, --anc_eval_max 40)",
      D["eval_held"]["q"] == held and S["domains"][d]["eval_held"]["q"] == held[:40])
    tr_pairs = {frozenset(qs[i]["opts"]) for i in train}
    T(f"[{d}] leakage by key: no held option pair in training (either order, any anchor)",
      not any(frozenset(qs[i]["opts"]) in tr_pairs for i in held))
    hs = set(held)
    T(f"[{d}] both orders of every held comparison held (a position bias scores 50% exactly)",
      all((i ^ 1) in hs and qs[i ^ 1]["opts"] == qs[i]["opts"][::-1] for i in held))
    T(f"[{d}] no held prompt string equals a training prompt", not ({qs[i]["prompt"] for i in held} & {qs[i]["prompt"] for i in train}))
    T(f"[{d}] the seen eval questions are training questions", set(D["eval_seen"]["q"]) <= set(train))
    _, ss1, held1, _ = derive(d, 1)
    T(f"[{d}] seed-specific: seed 1's split differs from seed 0's", ss1 != ss and held1 != held)
    T(f"[{d}] the 2-way chance is 50 by construction; heuristics logged", D["chance_readout"] == 0.5 and
      0 <= D["heuristic_ansfreq"] <= 1 and 0 <= D["heuristic_shorter"] <= 1 if "heuristic_shorter" in D else True,
      (D.get("heuristic_ansfreq"), D.get("heuristic_shorter")))
# months: rb6's class split, on its own stream (SeedSequence(4100 + seed) child 1 + EXTRA.index('months'))
sr = np.random.default_rng(np.random.SeedSequence(4100 + seed).spawn(16)[1 + PR.EXTRA.index("months")])
tr_m, held_m, held_e, Scls = PR.cyclic_split(12, sr, 0.4, 0.5)
Dm = J["domains"]["months"]
T("[months] tier S held questions = the rb6 class split re-derived (cyclic_split, stream 4100 + seed)",
  [tuple(q) for q in Dm["eval_held"]["q"]] == held_m and [tuple(q) for q in Dm["eval_heldE"]["q"]] == held_e)
T("[months] leakage by key: no held (item, k mod 12) class has a training question; tier E classes are trained",
  not ({(i, k % 12) for i, k in held_m} & {(i, k % 12) for i, k in tr_m}) and all((i, k % 12) in {(a, b % 12) for a, b in tr_m}
                                                                                  for i, k in held_e))
T("[months] the split = rb6's (seed 0: 118 train / 106 held / 40 heldE, make_rb6m_tasks.DOMS)",
  (Dm["n_train"], Dm["n_held"], Dm["n_heldE"]) == (118, 106, 40))

# ---------------------------------------------------------------- B / C. site decoding and decoded rows (53M dry logs)
ITEMS = {d: MD[d]["items"] for d in PR.ANCHOR}
TPL = {"stars": "Which star is closer", "landmarks": "Which landmark is closer", "languages": "Which language is more closely",
       "colours": "Which colour is closer", "cities": "Which city is closer"}
MONTHS = set(PR.MONTHS)


def check_line(text):
    """One decoded line with ⟦ ⟧ marks -> (domain or None, n sites, problems)."""
    dom = next((d for d, t in TPL.items() if t in text), None)
    probs, n, plain = [], 0, ""
    for part in re.split(r"(⟦[^⟧]*⟧)", text):
        if part.startswith("⟦"):
            tokn = part[1:-1]
            plain += tokn
            n += 1
            rest = text[text.index(part) + len(part):]
            if dom:
                if not any(plain.endswith(" " + it) or plain.endswith(it) and plain[:-len(it)].endswith(" ") for it in ITEMS[dom]):
                    probs.append(f"{dom}: site {tokn!r} does not end an item name: ...{plain[-30:]!r}")
                nxt = re.sub(r"⟦|⟧", "", rest)[:1]
                if nxt.isalnum():
                    probs.append(f"{dom}: the name continues after the site {tokn!r}: {nxt!r}")
            elif re.search(r"months? after", text):
                if tokn.strip() not in MONTHS:
                    probs.append(f"months: site {tokn!r} is not a month")
        else:
            plain += part
    return dom, n, probs


T("the site checker itself catches a site inside a name and a site on a non-item word (known-bad lines)",
  bool(check_line("Q: Which star is closer to Alg⟦ie⟧ba in the night sky, Vega or Rigel? Answer:")[2])
  and bool(check_line("Q: Which star is closer to Algieba in the⟦ night⟧ sky, Vega or Rigel? Answer:")[2])
  and not check_line("Q: Which star is closer to Algie⟦ba⟧ in the night sky,⟦ Vega⟧ or Rig⟦el⟧? Answer:")[2])
for dl in (dry0, dry1):
    txt = Path(dl).read_text()
    probs, n_lines, n_sites, per = [], 0, 0, {}
    for L in txt.splitlines():
        if "⟦" not in L or not L.lstrip().startswith(("train line", "eval ", "train row")):
            continue
        body = L.split(": ", 1)[1] if L.lstrip().startswith(("train line", "eval")) else L.split("): ", 1)[1]
        for seg in body.encode().decode("unicode_escape").encode("latin-1").decode("utf-8").split("\n"):
            if "⟦" not in seg:
                continue
            dom, n, pr = check_line(seg)
            probs += pr
            n_lines += 1; n_sites += n
            if dom:
                per.setdefault(dom, set()).add(n)
            elif re.search(r"months? after", seg) and "Q:" not in seg:
                per.setdefault("months", set()).add(n)
    T(f"{Path(dl).name}: every marked site of {n_lines} decoded lines ({n_sites} sites) decodes to the end of its item name",
      not probs and n_lines > 50, probs[:3])
    T(f"{Path(dl).name}: 3 sites per anchor line, 1 per months line (decoded lines with sites)",
      all(v == {3} for d, v in per.items() if d != "months") and per.get("months") == {1} and set(per) == set(TPL) | {"months"}, per)
    rows = [L for L in txt.splitlines() if re.match(r"^train row \d \(len 512, \d+ prior sites", L)]
    ev = {d: len(re.findall(rf"^\[{d}\] ", txt, re.M)) for d in ("months",) + PR.ANCHOR}
    blocks = re.split(r"^\[(\w+)\] ", txt, flags=re.M)
    nev = {}
    for i in range(1, len(blocks) - 1, 2):
        nev[blocks[i]] = len(re.findall(r"^  eval (held|seen|heldE)(_other)? row", blocks[i + 1], re.M))
    T(f"{Path(dl).name}: 3 decoded training rows (len 512) with sites; >= 3 eval rows + the other-option rows per anchor domain",
      len(rows) == 3 and all(nev.get(d, 0) >= 3 for d in ("months",) + PR.ANCHOR) and all(nev[d] >= 5 for d in PR.ANCHOR), nev)
    T(f"{Path(dl).name}: the site rule asserted for the 5 anchor domains (9000 sites each)",
      len(re.findall(r"^  site rule: .*: 9000 sites in 3000 lines asserted", txt, re.M)) == 5)
print("GENERATORS", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)})")
