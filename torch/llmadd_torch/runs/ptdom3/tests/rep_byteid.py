"""Byte identity of the 23 screen renders (make_jobs_ptdom3.build) vs runs/ptdom3/jobs/ and the shas in submitted.tsv,
before and after a replication render (make_jobs_ptdom3r patches make_jobs_ptdom3's globals for one call: they must be restored).
Writes nothing. Usage: python rep_byteid.py [rep cell tokens, e.g. none@3.2e-2:0]"""
import json
import sys

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_ptdom3 as M3     # noqa: E402
import make_jobs_ptdom3r as MR    # noqa: E402

P = M3.load_P()
rec = {l.split("\t")[0]: l.split("\t")[3].strip() for l in (M3.R / "submitted.tsv").read_text().splitlines() if l.strip()}


def screen_check(label):
    jobs = M3.build(P)
    bad = []
    for slot, name, tag, text, args, I in jobs:
        f = M3.jobfile(name)
        same = f.exists() and f.read_bytes() == text.encode()
        ij = M3.R / "jobs" / f"{name}.intent.json"
        same_i = ij.exists() and json.loads(ij.read_text()) == json.loads(json.dumps(I))
        same_s = rec.get(name) == M3.sha(f)
        if not (same and same_i and same_s):
            bad.append((name, same, same_i, same_s))
    print(f"[{label}] screen renders: {len(jobs)}; byte-identical job files, equal intent JSON, sha = submitted.tsv: "
          f"{len(jobs) - len(bad)}/{len(jobs)}", "PASS" if not bad else f"FAIL {bad}")
    return not bad


ok = screen_check("before")
toks = sys.argv[1:] or ["none@3.2e-2:0"]
cells = MR.parse_cells(toks)
commit = M3.snapshot()
for i, (a, lr, s) in enumerate(cells):
    name, tag, text, args, I = MR.render_rep(P, a, lr, s, commit, i % MR.N_REP)
    print(f"rep render {name}: slot {i % MR.N_REP}, {len(text)} bytes, seed {I['args']['seed']}, lr {I['args']['lr']}")
ok &= screen_check("after a rep render")
print("BYTEID", "PASS" if ok else "FAIL")
