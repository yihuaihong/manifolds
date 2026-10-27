"""Plumbing-test stand-in for make_jobs_pttpp.py: the real module with its batch dir R pointed at the fake root
(PTTPP_DIR), every function and constant re-exported; main() is the real one except that --submit never calls sbatch: it
writes SUBMITTED_<run> = 'FAKE' and records the run in SUBMIT_CALLED.txt (--test-only runs the real sbatch --test-only)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import importlib   # noqa: E402
T = importlib.import_module(os.environ.get("PLUMB_GENMOD", "make_jobs_pttpp"))   # T2 tests: the amended generator
if os.environ.get("PLUMB_COST_MAX"):   # cost-guard test: a lower budget
    T.COST_MAX = {"b200": float(os.environ["PLUMB_COST_MAX"].split(",")[0]), "other": float(os.environ["PLUMB_COST_MAX"].split(",")[1])}

REAL_R = T.R
T.R = Path(os.environ["PTTPP_DIR"])


class _B:
    """T.B for the guard in main() ('never another batch's dir name'): the real runs/pttpp is this batch's real twin, so
    its dirs are not 'another batch' here; everything else is the real path"""
    def __init__(self, p):
        self.p = p
    def __str__(self):
        return str(self.p)
    def __fspath__(self):
        return str(self.p)
    def __truediv__(self, o):
        return self.p / o
    def glob(self, pat):
        return [q for q in self.p.glob(pat) if REAL_R not in q.parents]


T.B = _B(T.B)
globals().update({k: v for k, v in T.__dict__.items() if not k.startswith("__")})
R = T.R


def main():
    if "--submit" in sys.argv:
        gid = sys.argv[sys.argv.index("--group") + 1]
        T.snapshots()
        n = 0
        for name, txt, F in T.render(gid):
            job, flag = T.R / f"job_{name}.sbatch", T.R / f"SUBMITTED_{name}"
            if not flag.exists():
                job.write_text(txt)
                flag.write_text("FAKE\n")
                with open(T.R / "SUBMIT_CALLED.txt", "a") as fh:
                    fh.write(f"{gid} {F['stage']} {name}\n")
                n += 1
        print(f"stand-in --submit: {n} fake submissions")
        return
    T.main()


if __name__ == "__main__":
    main()
