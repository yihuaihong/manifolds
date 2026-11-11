"""pt6comb test: select -> picks file -> generator (the CLI chain), on PLANTED records (readout_pt6comb.load is replaced in this process
by a planted world; the loader itself is tested by loader_test.py): select_pt6comb.main --after L1 / --after L1x / --verify writes and
checks the picks files in a temp dir; make_jobs_pt6comb.cells renders exactly the asked cells; tampered / GATE / NOT FINAL files are
refused by the generator; nothing is written in runs/pt6comb. -> plumbing.out"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
argv0 = sys.argv[:1]
sys.argv = argv0
import readout_pt6comb as RP   # noqa: E402
import select_pt6comb as SP    # noqa: E402
M6 = RP.M6
res = []
WORLD = {}
RP.load = lambda root, guard=True: WORLD   # the planted world (the loader is tested elsewhere)


def ok(name, cond, val=""):
    res.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({val})" if val != "" else ""))


def select(args):
    sys.argv = ["select_pt6comb.py"] + args
    buf = io.StringIO()
    code = 0
    try:
        with redirect_stdout(buf):
            SP.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        buf.write(str(e.code))
    sys.argv = argv0
    return code, buf.getvalue()


def refused(f):
    try:
        f()
        return False
    except (AssertionError, SystemExit):
        return True


before = sorted(p.name for p in HERE.iterdir())
P = M6.load_P()
RF = M6.refs6("B")
with tempfile.TemporaryDirectory() as tmp:
    WORLD.clear()
    RP.plant_sel(WORLD, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    RP.plant_sel(WORLD, "all6", {"1e-3": 55, "2e-3": 60, "4e-3": 52})
    code, out = select(["--after", "L1", "--picks_dir", tmp])
    f1 = Path(tmp) / "picks_after_L1.json"
    pk = json.loads(f1.read_text())
    ok("P1 --after L1 (interior) writes picks_after_L1.json once, chmod 444, stage C, rc 0", code == 0 and pk["stage_written"] == "C"
       and not (f1.stat().st_mode & 0o222), (code, pk["stage_written"]))
    ok("P2 --verify agrees", select(["--verify", str(f1)])[0] == 0)
    C = M6.cells("C", "B", str(f1))
    want = [(a, lr, s) for s in M6.CONF for a, lr in [("none", "4e-3"), ("all6", "2e-3"), ("none", "2e-3")] + [(f"only-{m}", "2e-3") for m in RP.MANI]]
    ok("P3 the generator's part C = none@4e-3, all6@2e-3, none@2e-3 (A2; all6@2e-3 merged), six only-m@2e-3 x seeds 8-15 (72), seed-major",
       C == want, len(C))
    names = [M6.render6(P, "C", "B", a, lr, s, RF[s], M6.COMMIT, 0, i % 8)[0] for i, (a, lr, s) in enumerate(C)]
    ok("P4 every part-C job renders (72 names, unique)", len(set(names)) == 72, names[:2])
    ok("P5 a second --after L1 is refused (written once)", select(["--after", "L1", "--picks_dir", tmp])[0] != 0)
    ok("P6 part L1x from a stage-C file is refused", refused(lambda: M6.cells("L1x", "B", str(f1))))
with tempfile.TemporaryDirectory() as tmp:
    WORLD.clear()
    RP.plant_sel(WORLD, "none", {"2e-3": 40, "4e-3": 45, "8e-3": 50})
    RP.plant_sel(WORLD, "all6", {"1e-3": 50, "2e-3": 55, "4e-3": 60})
    code, out = select(["--after", "L1", "--picks_dir", tmp])
    f1 = Path(tmp) / "picks_after_L1.json"
    pk = json.loads(f1.read_text())
    E = M6.cells("L1x", "B", str(f1))
    ok("P7 edges -> stage L1x; the generator renders exactly all6@8e-3 + none@1.6e-2 x seeds 0-3", code == 0 and pk["stage_written"] == "L1x"
       and E == [(a, lr, s) for s in M6.SEL for a, lr in (("all6", "8e-3"), ("none", "1.6e-2"))], E[:2])
    ok("P8 --after L1x refused before the asked cells exist", select(["--after", "L1x", "--picks_dir", tmp])[0] != 0)
    RP.plant_sel(WORLD, "all6", {"8e-3": 57})
    RP.plant_sel(WORLD, "none", {"1.6e-2": 48})
    code, out = select(["--after", "L1x", "--picks_dir", tmp])
    fx = Path(tmp) / "picks_after_L1x.json"
    pkx = json.loads(fx.read_text())
    ok("P9 --after L1x -> stage C at all6@4e-3 / none@8e-3", code == 0 and pkx["stage_written"] == "C" and pkx["pick_all6"] == "4e-3"
       and pkx["pick_none"] == "8e-3", (code, pkx["pick_all6"], pkx["pick_none"]))
    ok("P10 --verify of both files agrees", select(["--verify", str(f1)])[0] == 0 and select(["--verify", str(fx)])[0] == 0)
    os.chmod(fx, 0o644)
    ok("P11 a writable picks file is refused by the generator", refused(lambda: M6.cells("C", "B", str(fx))))
    j = json.loads(fx.read_text()); j["pick_none"] = "4e-3"; fx.write_text(json.dumps(j)); os.chmod(fx, 0o444)
    ok("P12 a tampered picks file: --verify DIFFERENT (rc 1) and the generator refuses its C list", select(["--verify", str(fx)])[0] == 1
       and refused(lambda: M6.cells("C", "B", str(fx))))
with tempfile.TemporaryDirectory() as tmp:
    WORLD.clear()
    RP.plant_sel(WORLD, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    RP.plant_sel(WORLD, "all6", {"1e-3": 55, "2e-3": 60, "4e-3": 52}, share=0.3)
    code, out = select(["--after", "L1", "--picks_dir", tmp])
    f1 = Path(tmp) / "picks_after_L1.json"
    ok("P13 every all6 cell fails the gate -> GATE file, rc 2; the generator refuses C and L1x from it", code == 2
       and json.loads(f1.read_text())["status"] == "GATE" and refused(lambda: M6.cells("C", "B", str(f1))) and refused(lambda: M6.cells("L1x", "B", str(f1))))
with tempfile.TemporaryDirectory() as tmp:
    WORLD.clear()
    RP.plant_sel(WORLD, "none", {"2e-3": 40, "4e-3": 45, "8e-3": 50})
    RP.plant_sel(WORLD, "all6", {"1e-3": 50, "2e-3": 55, "4e-3": 60})
    select(["--after", "L1", "--picks_dir", tmp])
    RP.plant_sel(WORLD, "all6", {"8e-3": 70})
    RP.plant_sel(WORLD, "none", {"1.6e-2": 48})
    code, out = select(["--after", "L1x", "--picks_dir", tmp])
    fx = Path(tmp) / "picks_after_L1x.json"
    ok("P14 NOT FINAL after L1x -> rc 2, the file says NOT FINAL, the generator refuses C", code == 2 and json.loads(fx.read_text())["status"] == "NOT FINAL"
       and refused(lambda: M6.cells("C", "B", str(fx))))
with tempfile.TemporaryDirectory() as tmp:
    WORLD.clear()
    RP.plant_sel(WORLD, "none", {"2e-3": 40, "4e-3": 50, "8e-3": 44})
    RP.plant_sel(WORLD, "all6", {"1e-3": 55, "2e-3": 60, "4e-3": 52})
    WORLD["only-s4", "2e-3", 1] = RP.fake_rec("only-s4", "2e-3", 1, {m: 50 for m in RP.MANI}, data="other")
    code, out = select(["--after", "L1", "--picks_dir", tmp])
    ok("P15 data differing between arms at a selection seed -> refused (rc 1), nothing written", code == 1 and not (Path(tmp) / "picks_after_L1.json").exists(), out[-120:])
ok("P16 nothing was written in runs/pt6comb", sorted(p.name for p in HERE.iterdir()) == before)
print(f"PLUMBING {'PASS' if all(res) else 'FAIL'} {sum(res)}/{len(res)}")
