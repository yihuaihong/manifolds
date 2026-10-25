"""Amendment S1: the 505M read-out's command shape (fake 505m runs of ro_s1_test + the real rb6_190m out + the 16 53M F88
Burst dirs, --picks P_b --x1 b) runs in strict mode, and its 53m / 190m sections (per-cell blocks and paired families)
are line-identical to the 190M read-out without --picks (reg/r190_new.txt of reg_s1.sh). No 505M metric exists in it
(the 505m cells are relabelled 53M runs).
    python ro_s1_combined.py"""
import glob
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path("/scratch/yh6210/llmadd-burst-scale")
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
RO = "/scratch/yh6210/llmadd-burst-ro"
pats = ["401-rb6pb-*", "404-rb6pb-*", "430-rb6pb-*", "431-rb6pb-*", "419-rb6pb-*", "415-rb6pb-*", "432-rb6pb-*", "433-rb6pb-*",
        "438-rb6mb-*", "444-rb6mb-*", "450-rb6mb-*", "456-rb6mb-*", "46[6-9]-rb6mb-*"]
b53 = sorted(p for pt in pats for p in glob.glob(f"{RO}/results/{pt}"))
cmd = [PY, "tools/rb6scale_summary.py", str(HERE / "ro_work/fake505"), "/scratch/yh6210/llmadd_torch/runs/rb6_190m/out", *b53,
       "--picks", str(HERE / "ro_work/P_b.json"), "--x1", "b", "--json", str(HERE / "ro_work/out_combined.json")]
o = subprocess.run(cmd, cwd=WT, capture_output=True, text=True, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
t = o.stdout + o.stderr
(HERE / "ro_work/out_combined.txt").write_text(t)
ref = (HERE / "reg/r190_new.txt").read_text()


def blocks(txt, sz):
    return re.findall(rf"^== {sz}: .*?(?=^==)", txt, re.M | re.S)


ok = o.returncode == 0 and "== rb6 scale read-out status: complete" in t and "sizes ['53m', '190m', '505m']" in t
same = all(blocks(t, sz) == blocks(ref, sz) and len(blocks(ref, sz)) == 2 for sz in ("53m", "190m"))
print(("PASS" if ok and same else "FAIL") + f" combined 53m + 190m + fake 505m read-out (--x1 b): rc {o.returncode}, strict complete "
      f"{ok}; the 53m / 190m cell and paired sections line-identical to the 190M read-out without --picks: {same}")
