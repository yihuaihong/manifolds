"""Amendment A1: the two NEW none lrs (1e-3, 2e-3) use the CPU references of their seed unchanged. Known answer: the
seed-0 entry of runs/ptdom/ptdom_refs.json (DRY rows sha / count, the splits, the anchor table shas, the step-0 lm, the
data sha after step 0). Checks the new cells refs/dry_none_lr1e-3_s0.json and refs/step0_none_lr2e-3_s0.json against it
(lr independence shown on a cell of the new grid, not only inferred from none@8e-3 = only-stars@1e-3 = only-months@5e-4)."""
import json
import sys
from pathlib import Path

T = Path(__file__).resolve().parent
ref = json.loads((T.parent / "ptdom_refs.json").read_text())["seeds"]["0"]
d = json.loads((T / "refs" / "dry_none_lr1e-3_s0.json").read_text())
s = json.loads((T / "refs" / "step0_none_lr2e-3_s0.json").read_text())
res = []


def C(name, ok, val):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + f"  ({val})")


C("dry none@1e-3 s0 ran (rc 0, 5 site-rule lines)", d["rc"] == 0 and d["site_rule_lines"] == 5, [d["rc"], d["site_rule_lines"], d["sec"]])
C("dry none@1e-3 s0: DRY rows [sha, n] = the seed-0 reference", [d["sha"], d["n"]] == ref["dry"], [d["sha"][:12], d["n"], ref["dry"][0][:12]])
C("dry none@1e-3 s0: splits = the reference", d["doms"] == ref["dom"], "6 domains")
C("dry none@1e-3 s0: anchor table shas = the reference", d["table_sha"] == ref["table_sha"], d["table_sha"])
C("dry none@1e-3 s0: no prior on any domain (lam 0)", all(not v["prior"] and v["lam"] == 0.0 for v in d["prior"].values()), "")
C("step0 none@2e-3 s0 ran", s["rc"] == 0 and "lm" in s, s["rc"])
C("step0 none@2e-3 s0: step-0 lm = the reference (exact)", s.get("lm") == ref["lm0"], [s.get("lm"), ref["lm0"]])
C("step0 none@2e-3 s0: data sha after step 0 = the reference", s.get("data_sha") == ref["dsha0"], [s.get("data_sha"), ref["dsha0"]])
print(f"A1 REFS {'PASS' if all(res) else 'FAIL'} ({sum(res)}/{len(res)})")
sys.exit(0 if all(res) else 1)
