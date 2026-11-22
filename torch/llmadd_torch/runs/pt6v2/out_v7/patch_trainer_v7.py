#!/usr/bin/env python3
"""pt6v2 Amendment V7: the trainer's only change, applied to the V7 copy runs/pt6v2/out_v7/code/tools/pretrain_range.py
(the registered snapshot runs/pt6v2/code stays a398fccbfe93; the V5 trainer 041b2f4241b9 is not the parent).
One flag, --drop_domains_after "d1,d2@S": from training step S on, the domain lines of d1, d2 are still DRAWN (question
from dom_rng, slot order from order_rng: every RNG stream advances exactly as before) and then left out of the row; the
text offset is the default draw (rng, same bound) and the row's text is the END of the same default chunk, extended to the
left by the dropped lines' length (row length = seq, the numbers line sees the same context, tokens per step unchanged).
Every other line (numbers, the kept domain lines) is unchanged in rate and content. Default "" = the a398fccbfe93 code path
(kept out of the run JSON's args at its default, like the Rev C/D/E flags). Logged when on: the parsed value at start, a
line at step 0 and at the drop step, dropped-line counts per domain in each eval entry (dropped_lines) and the run JSON
(drop_domains_after). Counted edits: each anchor must occur exactly once."""
import hashlib
from pathlib import Path

P = Path(__file__).resolve().parent / "code" / "tools" / "pretrain_range.py"
src = P.read_text()
assert hashlib.sha256(src.encode()).hexdigest()[:12] == "a398fccbfe93", "not the registered trainer"
E = [
    ('REVE_DEFAULTS = {"anc_eval_max": 0}   # Rev E (testing only): kept out of the run JSON\'s args at this value\n',
     'REVE_DEFAULTS = {"anc_eval_max": 0}   # Rev E (testing only): kept out of the run JSON\'s args at this value\n'
     'REVV7_DEFAULTS = {"drop_domains_after": ""}   # pt6v2 Amendment V7: kept out of the run JSON\'s args at this value\n'),
    ('    ap.add_argument("--stop_after", type=int, default=-1, help="testing only: exit after this step (simulated preemption)")\n',
     '    ap.add_argument("--stop_after", type=int, default=-1, help="testing only: exit after this step (simulated preemption)")\n'
     '    ap.add_argument("--drop_domains_after", default="", help="pt6v2 V7: \'d1,d2@S\' = from training step S on, the lines "\n'
     '                    "of these domains are drawn as before and left out of the row (the row\'s text fills their space)")\n'),
    ('                             and not (k in REVE_DEFAULTS and v == REVE_DEFAULTS[k])}, "steps": steps,\n',
     '                             and not (k in REVE_DEFAULTS and v == REVE_DEFAULTS[k])\n'
     '                             and not (k in REVV7_DEFAULTS and v == REVV7_DEFAULTS[k])}, "steps": steps,\n'),
    ('    steps = int(args.tokens // (args.bs * args.seq))\n',
     '    steps = int(args.tokens // (args.bs * args.seq))\n'
     '    DROP_D, DROP_AT = (), -1   # V7: domains whose lines are left out from step DROP_AT on\n'
     '    if args.drop_domains_after:\n'
     '        d_, s_ = args.drop_domains_after.split("@")\n'
     '        DROP_D, DROP_AT = tuple(lst(d_)), int(s_)\n'
     '        assert DOM and DROP_D and len(set(DROP_D)) == len(DROP_D) and set(DROP_D) <= set(EXT) and 0 < DROP_AT < steps, (\n'
     '            "--drop_domains_after", DROP_D, DROP_AT, steps)\n'
     '        print(f"V7 drop_domains_after = {args.drop_domains_after!r}: domain lines {list(DROP_D)} left out from step "\n'
     '              f"{DROP_AT} of {steps} (drawn, then dropped; text fills the space)", flush=True)\n'),
    ('            batch.ntext = []; batch.slot = []\n',
     '            batch.ntext = []; batch.slot = []; batch.dropped = []\n'),
    ('        ext = [ext[i] for i in order_rng.permutation(len(ext))]\n',
     '        ext = [ext[i] for i in order_rng.permutation(len(ext))]\n'
     '        if DROP_D and s >= DROP_AT:   # V7: drawn as before (every stream unchanged), then left out of the row\n'
     '            batch.dropped += [t for t, _ in ext if t in DROP_D]\n'
     '            ext = [(t, p) for t, p in ext if t not in DROP_D]\n'),
    ('                   data_sha="", dslot={d: [0] * args.extra_lines for d in EXT})   # A2: domain lines per slot after numbers\n',
     '                   data_sha="", dslot={d: [0] * args.extra_lines for d in EXT})   # A2: domain lines per slot after numbers\n'
     '    if DROP_D:   # V7: lines drawn and left out, per domain (saved with dyn)\n'
     '        dyn["ddrop"] = {d: 0 for d in DROP_D}\n'),
    ('            for t, j in batch.slot:\n'
     '                dyn["dslot"][t][j] += 1\n',
     '            for t, j in batch.slot:\n'
     '                dyn["dslot"][t][j] += 1\n'
     '            if DROP_D:   # V7\n'
     '                for t in batch.dropped:\n'
     '                    dyn["ddrop"][t] += 1\n'
     '                if s == 0 or s == DROP_AT:\n'
     '                    print(f"V7 step {s}: {len(batch.dropped)} lines dropped in this batch, sites "\n'
     '                          f"{ {d: int(len(batch.dsite[d])) for d in EXT} }", flush=True)\n'
     '                if s == 0:   # the flag is parsed as given and nothing is dropped before DROP_AT\n'
     '                    assert args.drop_domains_after == ",".join(DROP_D) + f"@{DROP_AT}" and not batch.dropped, (\n'
     '                        args.drop_domains_after, batch.dropped)\n'),
    ('        lg["data_sha"] = dyn["data_sha"][:16]\n'
     '        return lg\n',
     '        lg["data_sha"] = dyn["data_sha"][:16]\n'
     '        if DROP_D:   # V7: lines left out so far, per domain\n'
     '            lg["dropped_lines"] = dict(dyn["ddrop"])\n'
     '        return lg\n'),
    ('                    "curve": curve}\n',
     '                    "curve": curve}\n'
     '            if DROP_D:   # V7: the flag as consumed and the lines left out so far\n'
     '                res_["drop_domains_after"] = {"domains": list(DROP_D), "step": DROP_AT, "dropped_lines": dict(dyn["ddrop"])}\n'),
]
for old, new in E:
    assert src.count(old) == 1, ("anchor count", src.count(old), old[:80])
    src = src.replace(old, new)
P.write_text(src)
print("patched", P, hashlib.sha256(src.encode()).hexdigest()[:12])
