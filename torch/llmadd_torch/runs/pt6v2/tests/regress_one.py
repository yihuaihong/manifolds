"""pt6v2 test 1 helper: print the normalised diff of one pt6comb C job file (argv[1]) vs its pt6v2 re-rendering; the intent JSON
lines are compared key by key."""
import difflib, json, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
f = Path(sys.argv[1]); sys.argv = sys.argv[:1]
import regress_pt6comb as T  # noqa: E402  (runs the full regression once; its output is printed first)
m = re.fullmatch(r"job_pt6comb-B-(.+)-lr([0-9.e-]+)-s(\d+)\.sbatch", f.name)
old = f.read_text(); slot = int(re.search(r"slot(\d+)", old)[1])
_, _, text, _, I = T.V.render_v2(T.P, T.doms, m[1], m[2], int(m[3]), T.REF[m[3]], T.commit, "b200", slot, seeds=tuple(range(8, 16)))
for ln in difflib.unified_diff(old.splitlines(), T.norm(text).splitlines(), lineterm="", n=0):
    if ln.startswith(("---", "+++", "@@")): continue
    js = re.search(r"(\{\"batch\".*\})", ln)
    print((ln[:1] + " <intent JSON, %d chars>" % len(js[1])) if js else ln[:240])
oj = [json.loads(x) for x in re.findall(r"(\{\"batch\".*\})", old)]; nj = [json.loads(x) for x in re.findall(r"(\{\"batch\".*\})", T.norm(text))]
for a, b in zip(oj, nj):
    print("intent keys differing:", {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k) and k not in ("dep_sha",)},
          "| dep_sha files old-only/new-only/changed:", sorted(set(a["dep_sha"]) - set(b["dep_sha"])), sorted(set(b["dep_sha"]) - set(a["dep_sha"])),
          sorted(k for k in set(a["dep_sha"]) & set(b["dep_sha"]) if a["dep_sha"][k] != b["dep_sha"][k]))
