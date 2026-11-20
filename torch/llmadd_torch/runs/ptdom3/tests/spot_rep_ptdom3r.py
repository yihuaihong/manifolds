"""Hand spot-check: area / final3 straight from the raw run JSON curves (no readout_ptdom3 import), vs the read-out JSON."""
import glob
import json
import sys

OUT = "/scratch/yh6210/llmadd_torch/runs/ptdom3/out"
RO = json.load(open(sys.argv[1]))


def raw(arm, lr, s, key):
    f = glob.glob(f"{OUT}/ptdom3-T3-{arm}-lr{lr}-s{s}/*/run_whole_none_seed{s}_*.json")
    assert len(f) == 1, f
    r = json.load(open(f[0]))
    v = [e[key] for e in r["curve"]]
    return 100 * sum(v) / len(v), 100 * sum(v[-3:]) / 3, len(v), f[0].split("/out/")[1]


def show(name, arm, lr, s, key, d, which, ro_val):
    a1, f1, n1, p1 = raw(arm, lr, s, key)
    a0, f0, n0, p0 = raw("none", NONE, s, key)
    hand = (a1 - a0) if which == "area" else (f1 - f0)
    print(f"{name}: {arm}@{lr} s{s} {which} {(a1 if which == 'area' else f1):.4f} ({n1} evals, {p1}) - none@{NONE} s{s} "
          f"{(a0 if which == 'area' else f0):.4f} ({n0} evals) = {hand:+.4f} | read-out JSON {ro_val:+.4f} | "
          f"{'MATCH' if abs(hand - ro_val) < 1e-9 else 'DIFF'}")


NONE = "1.6e-2"
D = RO["domains"]
show("SC1 torus s2 g", "only-torus", "1e-3", 2, "torus_held_nonid", "torus", "area", D["torus"]["primary"]["g"]["2"])
show("SC2 torus s2 final3 g", "only-torus", "1e-3", 2, "torus_held_nonid", "torus", "final3", D["torus"]["primary"]["final3"]["g"]["2"])
show("SC3 hcube s1 full-set g (secondary)", "only-hcube", "2e-3", 1, "hcube_held_nonid", "hcube", "area", D["hcube"]["secondary"]["g"]["1"])
show("SC4 box5 s0 g", "only-box5", "1e-3", 0, "box5_held_no_negtwin_nonid", "box5", "area", D["box5"]["primary"]["g"]["0"])
a, f, n, p = raw("none", "3.2e-2", 1, "torus_held_nonid")
print(f"SC5 none@3.2e-2 s1 torus area {a:.4f} (hand) vs read-out cell {D['torus']['_cells']['none@3.2e-2']['1']:.4f}")
