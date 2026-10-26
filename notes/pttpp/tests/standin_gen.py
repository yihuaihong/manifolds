"""Plumbing-test stand-in for make_jobs_pttpp.py: the real module with its batch dir R pointed at the fake root
(PTTPP_DIR), every function and constant re-exported; main() is the real one except that --submit never calls sbatch: it
writes SUBMITTED_<run> = 'FAKE' and records the run in SUBMIT_CALLED.txt (--test-only runs the real sbatch --test-only)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

T.R = Path(os.environ["PTTPP_DIR"])
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
