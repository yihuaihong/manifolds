"""Amendment X1: the main read-out tools/rb6scale_summary.py (UNCHANGED, sha256[:12] e807e43be36b) takes none@best from
--picks only; this tests that an X1 picks JSON (pick none 2.4e-3) is accepted as it is, on fake 505M runs (no 505M result
is read): every 505M cell = a 53M F88 Burst run of the same seed with the 505M size facts (data / init are
size-independent; the read-out's identity checks see the 505M intent):
  none@6e-4 <- rb6pb none (401/404/430/431), all6-lam1@6e-4 <- rb6pb all6-lam1 (419/415/432/433),
  only-numbers@6e-4 <- rb6mb only-numbers (438/444/450/456), none@2.4e-3 AND none@4.8e-3 <- rb6mb none-lr8e-3 (466-469).
Known answers: (b) the X1 picks JSON written by rb6scale_select.py --grid_none (sel_work/b_clear_flag.json: none 2.4e-3)
-> none_nlr = none@2.4e-3, the none@4.8e-3 runs IGNORED, strict status complete; (a) picks.json-like (none 4.8e-3) ->
none_nlr = none@4.8e-3, the 2.4e-3 runs IGNORED; every cell value and paired statistic of (a) = (b) (same source runs),
only the labels differ; (b) without none@2.4e-3 s2 / s3 (the X1 follow-up seeds) -> strict REFUSED, --allow_partial ->
PROVISIONAL (n=2 of 4) on the none@2.4e-3 lines.
    python ro_x1_test.py"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path("/scratch/yh6210/llmadd-burst-scale")
RO = Path("/scratch/yh6210/llmadd-burst-ro/results")
PY = "/scratch/yh6210/miniconda3/envs/open_r1/bin/python"
W = HERE / "ro_work"
SRC = {("none", "6e-4"): ["401-rb6pb-53m-2pct-k2-none-s0", "404-rb6pb-53m-2pct-k2-none-s1", "430-rb6pb-53m-2pct-k2-none-s2",
                          "431-rb6pb-53m-2pct-k2-none-s3"],
       ("all6-lam1", "6e-4"): ["419-rb6pb-53m-2pct-k2-all6-lam1-s0", "415-rb6pb-53m-2pct-k2-all6-lam1-s1",
                               "432-rb6pb-53m-2pct-k2-all6-lam1-s2", "433-rb6pb-53m-2pct-k2-all6-lam1-s3"],
       ("only-numbers", "6e-4"): [f"{n}-rb6mb-53m-2pct-k2-only-numbers-s{s}" for s, n in enumerate((438, 444, 450, 456))],
       ("none", "2.4e-3"): [f"{466 + s}-rb6mb-53m-2pct-k2-none-lr8e-3-s{s}" for s in range(4)],
       ("none", "4.8e-3"): [f"{466 + s}-rb6mb-53m-2pct-k2-none-lr8e-3-s{s}" for s in range(4)]}
ok_all = True


def say(cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


def runjson(d):
    return next(Path(d).glob("*/run_*.json"))


def write(root, src_dir, arm, lr, s):
    r = json.loads(runjson(src_dir).read_text())
    assert r["args"]["seed"] == s and r["args"]["d_model"] == 384
    tag = f"rb6tb-505m-2pct-k2-{arm}-lr{lr}"
    tdir = root / f"{tag}-s{s}" / tag
    tdir.mkdir(parents=True)
    r["args"].update(tag=f"_{tag}", lr=float(lr), d_model=1024, n_layers=24, n_heads=16, d_ff=4096, out=str(tdir))
    r.update(n_params=505437184, grad_ckpt_on=True, geo_layers=[8, 16], read_layers=[8, 16], code_sha="93e22bf4c4c0")
    r["dep_sha"] = dict(r.get("dep_sha", {}), **{"toy_struct.py": "d182ce548bcc"})
    out = tdir / f"run_whole_{r['args']['target']}_seed{s}_{tag}.json"
    out.write_text(json.dumps(r))
    off = json.loads((Path(src_dir) / "offline_lm.json").read_text())
    assert len(off) == 1
    v = dict(next(iter(off.values())), tag=f"_{tag}", code_sha=r["code_sha"])
    (tdir.parent / "offline_lm.json").write_text(json.dumps({str(out): v}))
    (tdir.parent / "check.json").write_text(json.dumps({"pass": True, "checks": [], "info": {}}))


def run(root, picks, extra=(), name="x"):
    js = W / f"{name}.json"
    js.unlink(missing_ok=True)
    o = subprocess.run([PY, "tools/rb6scale_summary.py", str(root), "--picks", str(picks), "--json", str(js), *extra],
                       cwd=WT, capture_output=True, text=True)
    (W / f"{name}.txt").write_text(o.stdout + o.stderr)
    return o.returncode, o.stdout + o.stderr, (json.loads(js.read_text()) if js.exists() and o.returncode == 0 else None)


say(hashlib.sha256((WT / "tools/rb6scale_summary.py").read_bytes()).hexdigest()[:12] == "e807e43be36b",
    "tools/rb6scale_summary.py is the pre-registered read-out e807e43be36b (unchanged by X1)")
shutil.rmtree(W, ignore_errors=True)
W.mkdir(parents=True)
root = W / "fake505"
for (arm, lr), dirs in SRC.items():
    for s, d in enumerate(dirs):
        write(root, RO / d, arm, lr, s)
pb = HERE / "sel_work" / "b_clear_flag.json"   # written by rb6scale_select.py --grid_none on the b_clear fake
PB = json.loads(pb.read_text())
assert PB["pick"] == {"none": "2.4e-3", "all6-lam1": "6e-4"} and "grid" in PB
pa = W / "picks_a.json"
pa.write_text(json.dumps(dict(PB, pick={"none": "4.8e-3", "all6-lam1": "6e-4"})))

rc, tb, Jb = run(root, pb, name="b")
ign_b = [l for l in tb.splitlines() if "IGNORED" in l]
say(rc == 0 and Jb["status"] == "complete" and Jb["design"]["505m"]["none_nlr"] == "2.4e-3"
    and Jb["labels"]["505m"]["none_nlr"] == "none@2.4e-3" and len(ign_b) == 4 and all("none-lr4.8e-3" in l for l in ign_b)
    and "505m none@6e-4, none@2.4e-3, all6-lam1@6e-4, only-numbers@6e-4" in tb,
    f"(b) X1 picks JSON (none 2.4e-3): none_nlr = none@2.4e-3, strict status {Jb and Jb['status']}, "
    f"{len(ign_b)} IGNORED lines (all none@4.8e-3)")
s1b = [r_ for r_ in Jb["families"]["505m|S1b"]] if Jb else []
say(bool(s1b) and all(r_["label"] == "all6-lam1@6e-4 - none@2.4e-3" for r_ in s1b),
    "(b) S1b rows labelled 'all6-lam1@6e-4 - none@2.4e-3'")
rc, ta, Ja = run(root, pa, name="a")
ign_a = [l for l in ta.splitlines() if "IGNORED" in l]
say(rc == 0 and Ja["status"] == "complete" and Ja["design"]["505m"]["none_nlr"] == "4.8e-3" and len(ign_a) == 4
    and all("none-lr2.4e-3" in l for l in ign_a), "(a) picks none 4.8e-3: none_nlr = none@4.8e-3, the 4 none@2.4e-3 runs IGNORED")
strip = lambda c: {k: {kk: vv for kk, vv in v.items() if kk not in ("label", "lr")} for k, v in c.items()}
fam = lambda F: {k: [{kk: vv for kk, vv in r_.items() if kk != "label"} for r_ in v] for k, v in F.items()}
say(Ja and strip(Ja["cells"]) == strip(Jb["cells"]) and fam(Ja["families"]) == fam(Jb["families"]),
    f"(a) = (b) in every cell value ({len(Jb['cells'])} cell-metric entries) and paired statistic: only labels / lr differ")
# (b) before the X1 follow-up seeds 2-3 of 2.4e-3 exist
rootp = W / "fake505_partial"
shutil.copytree(root, rootp)
for s in (2, 3):
    shutil.rmtree(rootp / f"rb6tb-505m-2pct-k2-none-lr2.4e-3-s{s}")
rc, tp, _ = run(rootp, pb, name="b_partial_strict")
say(rc != 0 and "REFUSED (strict)" in tp and "505m none@2.4e-3 [2, 3]" in tp,
    "(b) without none@2.4e-3 s2 / s3: strict read-out REFUSED, naming 505m none@2.4e-3 [2, 3]")
rc, tp2, Jp = run(rootp, pb, ("--allow_partial",), name="b_partial")
say(rc == 0 and Jp["status"] == "PROVISIONAL" and "none@2.4e-3" in tp2 and "PROVISIONAL (n=2 of 4)" in tp2
    and Jp["missing"]["505m"]["none_nlr"] == [2, 3], "(b) --allow_partial: status PROVISIONAL, none@2.4e-3 PROVISIONAL (n=2 of 4)")
print("RO X1 TEST", "PASS" if ok_all else "FAIL")
