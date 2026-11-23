#!/usr/bin/env python3
"""pt6v2 Amendment V7b: the trainer's only change on top of V7, applied to a copy of runs/pt6v2/out_v7/code ->
runs/pt6v2/out_v7b/code (the V7 trainer 8da22c6ea208 stays as it is; the registered snapshot runs/pt6v2/code stays a398fccbfe93).
One flag, --drop_fill {left, inplace} (used with --drop_domains_after):
  left (default) = V7: the dropped torus / box5 lines are left out and the row's text extends to the left (code path unchanged);
  inplace        = each dropped line is replaced, IN ITS SLOT, by a text filler of the same token length: training text from
                   its own stream (default_rng(7100 + seed); every other stream unchanged), its last token(s) = the line's
                   newline (every torus / box5 training line ends with enc("\\n"), asserted), so the next line still starts a
                   new line. The row's left text is the flag-off text (n unchanged), so every kept line (numbers, dim, cities,
                   countries) keeps its row, positions, items and slot index of the flag-off (registered all6) run; the filler
                   tokens count as text, so the text tokens equal V7's.
Default "left" is kept out of the run JSON's args (REVV7B_DEFAULTS, like REVV7_DEFAULTS). Logged when inplace: the parsed value at
start, a line at step 0 and at the drop step, filler lines / tokens per eval (fill) and in the run JSON
(drop_domains_after.fill / fill_stream / fill_counts); the filler stream state is checkpointed. Counted edits: each anchor
must occur exactly once."""
import hashlib
import shutil
from pathlib import Path

W = Path(__file__).resolve().parent
SRC, DST = W.parent / "out_v7" / "code", W / "code"
sha = lambda b: hashlib.sha256(b).hexdigest()[:12]   # noqa: E731
assert sha((SRC / "tools" / "pretrain_range.py").read_bytes()) == "8da22c6ea208", "not the V7 trainer"
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__"))
P = DST / "tools" / "pretrain_range.py"
src = P.read_text()
E = [
    ('REVV7_DEFAULTS = {"drop_domains_after": ""}   # pt6v2 Amendment V7: kept out of the run JSON\'s args at this value\n',
     'REVV7_DEFAULTS = {"drop_domains_after": ""}   # pt6v2 Amendment V7: kept out of the run JSON\'s args at this value\n'
     'REVV7B_DEFAULTS = {"drop_fill": "left"}   # pt6v2 Amendment V7b: kept out of the run JSON\'s args at this value\n'
     'FILL = "_fill"   # V7b: the line type of a filler (text tokens in a dropped line\'s slot)\n'),
    ('                    "of these domains are drawn as before and left out of the row (the row\'s text fills their space)")\n',
     '                    "of these domains are drawn as before and left out of the row (the row\'s text fills their space)")\n'
     '    ap.add_argument("--drop_fill", default="left", choices=("left", "inplace"), help="pt6v2 V7b, with --drop_domains_after: "\n'
     '                    "left = V7 (the row\'s text extends to the left); inplace = each dropped line is replaced in its slot by "\n'
     '                    "a training-text filler of the same token length (own stream 7100 + seed, last token(s) = the line\'s "\n'
     '                    "newline): every kept line keeps its positions and slot")\n'),
    ('              f"{DROP_AT} of {steps} (drawn, then dropped; text fills the space)", flush=True)\n',
     '              f"{DROP_AT} of {steps} (drawn, then dropped; text fills the space)", flush=True)\n'
     '    DROP_FILL = args.drop_fill   # V7b\n'
     '    assert DROP_FILL == "left" or DROP_D, ("--drop_fill inplace needs --drop_domains_after", DROP_FILL)\n'
     '    if DROP_D and DROP_FILL == "inplace":\n'
     '        NLT = list(enc("\\n"))\n'
     '        assert FILL not in EXT and all(list(DLINE[d, q][0][-len(NLT):]) == NLT and len(DLINE[d, q][0]) > len(NLT)\n'
     '                                       for d in DROP_D for q in DPOOL[d]), "a dropped domain\'s training line does not end with the newline"\n'
     '        fill_rng = np.random.default_rng(7100 + args.seed)   # V7b: the fillers\' own stream (every other stream unchanged)\n'
     '        print(f"V7b drop_fill = {DROP_FILL!r}: each dropped line of {list(DROP_D)} from step {DROP_AT} -> a training-text filler "\n'
     '              f"of its token length in its slot (stream 7100 + seed = {7100 + args.seed}; last {len(NLT)} token(s) = the line\'s "\n'
     '              f"newline {NLT}); the kept lines keep their positions and slots, the row\'s text is the flag-off text", flush=True)\n'),
    ('    def dom_row(s, r, parts, rows, ops, ans):\n',
     '    def fill_ids(ids):\n'
     '        """V7b: the filler of a dropped line ids: len(ids) training-text tokens from the fillers\' own stream, the last\n'
     '        len(NLT) of them replaced by the line\'s newline."""\n'
     '        L_ = len(ids)\n'
     '        o_ = int(fill_rng.integers(0, len(train) - L_))\n'
     '        return [int(v) for v in np.asarray(train[o_:o_ + L_ - len(NLT)], dtype=np.int64)] + list(NLT)\n'
     '\n'
     '    def dom_row(s, r, parts, rows, ops, ans):\n'),
    ('            batch.ntext = []; batch.slot = []; batch.dropped = []\n',
     '            batch.ntext = []; batch.slot = []; batch.dropped = []; batch.fill = []\n'),
    ('            ext = [(t, p) for t, p in ext if t not in DROP_D]\n',
     '            if DROP_FILL == "inplace":   # V7b: in its slot, a text filler of the line\'s token length (the row\'s text unchanged)\n'
     '                ext = [(FILL, fill_ids(DLINE[t, p][0])) if t in DROP_D else (t, p) for t, p in ext]\n'
     '            else:\n'
     '                ext = [(t, p) for t, p in ext if t not in DROP_D]\n'),
    ('        n = n0 - sum(len(DLINE[t, p][0]) for t, p in ext)\n',
     '        n = n0 - sum(len(p) if t == FILL else len(DLINE[t, p][0]) for t, p in ext)\n'),
    ('        batch.slot += [(t, j) for j, (t, _) in enumerate(ext)]   # slot j + 1 after the numbers line(s)\n',
     '        batch.slot += [(t, j) for j, (t, _) in enumerate(ext) if t != FILL]   # slot j + 1 after the numbers line(s)\n'),
    ('            else:\n'
     '                ids, sites, _, _ = DLINE[t, p]\n'
     '                batch.dsite[t] += [(r, off + q, v) for q, v in sites]\n'
     '            row += ids; off += len(ids); batch.ntok[t] += len(ids)\n',
     '            elif t == FILL:   # V7b: text tokens in the dropped line\'s slot (no sites; counted as text)\n'
     '                ids = p\n'
     '                batch.fill.append((r, off, len(ids)))\n'
     '            else:\n'
     '                ids, sites, _, _ = DLINE[t, p]\n'
     '                batch.dsite[t] += [(r, off + q, v) for q, v in sites]\n'
     '            row += ids; off += len(ids); batch.ntok["text" if t == FILL else t] += len(ids)\n'),
    ('        dyn["ddrop"] = {d: 0 for d in DROP_D}\n',
     '        dyn["ddrop"] = {d: 0 for d in DROP_D}\n'
     '    if DROP_D and DROP_FILL == "inplace":   # V7b: filler lines / tokens so far (saved with dyn)\n'
     '        dyn["dfill"] = {"lines": 0, "tokens": 0}\n'),
    ('                    assert args.drop_domains_after == ",".join(DROP_D) + f"@{DROP_AT}" and not batch.dropped, (\n'
     '                        args.drop_domains_after, batch.dropped)\n',
     '                    assert args.drop_domains_after == ",".join(DROP_D) + f"@{DROP_AT}" and not batch.dropped, (\n'
     '                        args.drop_domains_after, batch.dropped)\n'
     '                if DROP_FILL == "inplace":   # V7b\n'
     '                    dyn["dfill"]["lines"] += len(batch.fill); dyn["dfill"]["tokens"] += sum(L_ for _, _, L_ in batch.fill)\n'
     '                    if s == 0:   # the flag is parsed as given and there is no filler before DROP_AT\n'
     '                        assert args.drop_fill == "inplace" and not batch.fill, (args.drop_fill, batch.fill)\n'
     '                    if s == 0 or s == DROP_AT:   # logged, not asserted (one filler per dropped line)\n'
     '                        print(f"V7b step {s}: {len(batch.fill)} fillers ({sum(L_ for _, _, L_ in batch.fill)} tokens) for "\n'
     '                              f"{len(batch.dropped)} dropped lines: " + ("ok" if len(batch.fill) == len(batch.dropped)\n'
     '                                                                       else "## FLAG fillers != dropped lines"), flush=True)\n'),
    ('            lg["dropped_lines"] = dict(dyn["ddrop"])\n',
     '            lg["dropped_lines"] = dict(dyn["ddrop"])\n'
     '        if DROP_D and DROP_FILL == "inplace":   # V7b: filler lines / tokens so far\n'
     '            lg["fill"] = dict(dyn["dfill"])\n'),
    ('                             and not (k in REVV7_DEFAULTS and v == REVV7_DEFAULTS[k])}, "steps": steps,\n',
     '                             and not (k in REVV7_DEFAULTS and v == REVV7_DEFAULTS[k])\n'
     '                             and not (k in REVV7B_DEFAULTS and v == REVV7B_DEFAULTS[k])}, "steps": steps,\n'),
    ('                res_["drop_domains_after"] = {"domains": list(DROP_D), "step": DROP_AT, "dropped_lines": dict(dyn["ddrop"])}\n',
     '                res_["drop_domains_after"] = {"domains": list(DROP_D), "step": DROP_AT, "dropped_lines": dict(dyn["ddrop"])}\n'
     '                if DROP_FILL != "left":   # V7b: the fill mode as consumed, its stream, the fillers so far\n'
     '                    res_["drop_domains_after"].update(fill=DROP_FILL, fill_stream=7100 + args.seed, fill_counts=dict(dyn["dfill"]))\n'),
    ('                st_.update(dom_rng=dom_rng.bit_generator.state, order_rng=order_rng.bit_generator.state)\n',
     '                st_.update(dom_rng=dom_rng.bit_generator.state, order_rng=order_rng.bit_generator.state)\n'
     '            if DROP_D and DROP_FILL == "inplace":   # V7b\n'
     '                st_["fill_rng"] = fill_rng.bit_generator.state\n'),
    ('            dom_rng.bit_generator.state = st["dom_rng"]; order_rng.bit_generator.state = st["order_rng"]\n',
     '            dom_rng.bit_generator.state = st["dom_rng"]; order_rng.bit_generator.state = st["order_rng"]\n'
     '        if DROP_D and DROP_FILL == "inplace":   # V7b\n'
     '            fill_rng.bit_generator.state = st["fill_rng"]\n'),
]
for old, new in E:
    assert src.count(old) == 1, ("anchor count", src.count(old), old[:80])
    src = src.replace(old, new)
P.write_text(src)
new = sha(src.encode())
C = DST / "COMMIT"
c = C.read_text()
old_c = "pretrain_range.py 8da22c6ea208 (pt6v2 Amendment V7: a398fccbfe93 + --drop_domains_after, runs/pt6v2/out_v7/patch_trainer_v7.py)"
assert c.count(old_c) == 1
C.write_text(c.replace(old_c, f"pretrain_range.py {new} (pt6v2 Amendment V7b: 8da22c6ea208 + --drop_fill, "
                              "runs/pt6v2/out_v7b/patch_trainer_v7b.py; 8da22c6ea208 = a398fccbfe93 + --drop_domains_after, "
                              "runs/pt6v2/out_v7/patch_trainer_v7.py)"))
print("patched", P, new)
