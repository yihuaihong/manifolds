"""Amendment A1: after the full re-run of the CPU references (a1_refs_full.sh = every cell of run_refs.sh, logs rewritten in
tests/refs/), every dry / step-0 cell = the stored runs/ptdom/ptdom_refs.json entry of its seed (which is NOT rewritten):
DRY rows [sha, n], the splits, the anchor table shas, the step-0 lm (exact) and the data sha after step 0."""
import json
import sys
from pathlib import Path

T = Path(__file__).resolve().parent
ref = {int(k): v for k, v in json.loads((T.parent / "ptdom_refs.json").read_text())["seeds"].items()}
bad, n = [], 0
for f in sorted((T / "refs").glob("dry_*.json")):
    j = json.loads(f.read_text()); s = j["seed"]; n += 1
    if j["rc"] != 0 or [j["sha"], j["n"]] != ref[s]["dry"] or j["doms"] != ref[s]["dom"] or j["table_sha"] != ref[s]["table_sha"] \
            or j["site_rule_lines"] != 5:
        bad.append(f.name)
    print(f"{'PASS' if f.name not in bad else 'FAIL'} {f.stem}: rc {j['rc']}, DRY {j['sha'][:12] if j['sha'] else None} n {j['n']} "
          f"(ref {ref[s]['dry'][0][:12]}), splits / table shas {'=' if j['doms'] == ref[s]['dom'] and j['table_sha'] == ref[s]['table_sha'] else '!='} ref, {j['sec']} s")
for f in sorted((T / "refs").glob("step0_*.json")):
    j = json.loads(f.read_text()); s = j["seed"]; n += 1
    if j["rc"] != 0 or j.get("lm") != ref[s]["lm0"] or j.get("data_sha") != ref[s]["dsha0"]:
        bad.append(f.name)
    print(f"{'PASS' if f.name not in bad else 'FAIL'} {f.stem}: lm {j.get('lm')} (ref {ref[s]['lm0']}), data sha {j.get('data_sha')} (ref {ref[s]['dsha0']})")
# A2: the label says what this script checks: every STORED cell (refs/*.json, whenever it was run) against ptdom_refs.json;
# which cells were re-run in an invocation is in the log lines above it (a1_refs_full.sh prints one JSON line per cell it ran)
print(f"A1 REFS STORED CELLS = ptdom_refs.json: {'PASS' if not bad else 'FAIL'} ({n - len(bad)}/{n} stored cells; not a statement "
      f"about which were re-run)")
sys.exit(0 if not bad else 1)
