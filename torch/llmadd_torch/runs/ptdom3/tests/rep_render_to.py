"""Render replication cells into a directory (not runs/ptdom3/jobs; nothing else written), for diffing against the screen's files.
Usage: python rep_render_to.py OUTDIR <cell tokens...>"""
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_ptdom3 as M3     # noqa: E402
import make_jobs_ptdom3r as MR    # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
P = M3.load_P()
commit = M3.snapshot()
for i, (a, lr, s) in enumerate(MR.parse_cells(sys.argv[2:])):
    name, tag, text, args, I = MR.render_rep(P, a, lr, s, commit, i % MR.N_REP)
    (out / f"{name}.sbatch").write_text(text)
    print(out / f"{name}.sbatch")
