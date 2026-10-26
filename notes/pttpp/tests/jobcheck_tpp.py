"""Every pttpp job file vs the nearest job that was already submitted (runs/pttpp/AUDIT.md "Tests"): token by token, only the
items the design lists may differ. Stage A of every group (the files make_jobs_pttpp.py writes) + stage X / C renderings
(rendered in-process from facts(); no decision file is written).
  7% L40S  vs F87 runs/ptlr job_<190m|500m>-lr<pick>-7pct-<arm>-s<seed> (same arm and seed): ARGS may differ only in --out,
           --tag, --lr, the model dims, + --tokens / --eval_every, - --save_model / --ckpt_every (+ --save_model in stage C);
           lines: job name, --nice, --time, comments, the GPU guard, data sha, INTENT, CHECK7, the hook
  B200     vs ptlr_1b job_1b-lr<pick>-7pct-<arm>-s<seed>: ARGS --out, --tag, --lr, (1.07B@0.745) --tokens / --eval_every,
           (bridge) the 505M dims and no --save_model; header: slot name, account in the file, --nice, --time, output
  bridge   also vs F87's 505M job of the arm and seed: ARGS --out, --tag, --ckpt_every 2000 -> 500, no --save_model, nothing else
  2% A100  vs runs/pt2s505 job_pt2s505-lr<lr>-<arm>-s<seed> (same lr when on pt2s505's grid): ARGS --out, --tag, the dims,
           --tokens, --eval_every, no --ckpt_every, --lr; the intent JSON only in the size / tokens / tag fields; the DRY hash,
           the model line, code2, the hook, the model removal (stage A / X), INTENT, RC
Planted errors (each must be caught): an extra --lam, --eval_every 250, --tokens off by one step, a missing hook, --nice
removed, a changed intent field (events), a changed CHECK_PY line. Prints JOBCHECK PASS (n) / FAIL."""
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/yh6210/llmadd_torch")
import make_jobs_pttpp as T   # noqa: E402

RUNS = T.B / "runs"
G2 = T.g2()
fails, n_ok = [], 0


def flags(text):
    a = [l for l in text.splitlines() if l.startswith("ARGS=")]
    assert len(a) == 1, "one ARGS line"
    return T.M1.flags(a[0][len('ARGS="'):-1])


def arg_diff(new, old):
    fn, fo = flags(new), flags(old)
    return {k: (fo.get(k), fn.get(k)) for k in set(fn) | set(fo) if fn.get(k) != fo.get(k)}


def line_diff(new, old):
    d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
         if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    return d


def ok_lines(diff, allowed):
    bad = [l for l in diff if not any(re.match(p, l[1:]) for p in allowed)]
    return bad


def judge(name, cond, why):
    global n_ok
    if cond:
        n_ok += 1
    else:
        fails.append(f"{name}: {why}")
        print("FAIL", name, why)


DIM = {"--d_model", "--n_layers", "--n_heads", "--d_ff"}
L7 = [r"#SBATCH --job-name=", r"#SBATCH --nice=20$", r"#SBATCH --time=", r"#SBATCH --output=", r"# ", r"#$",
      r"if \[ -e /scratch/yh6210/llmadd_torch/runs/(ptlr|ptlr_1b|pttpp)/HOLD \]", r"cd /scratch/yh6210/llmadd_torch/runs/(ptlr|ptlr_1b|pttpp)/code7?/tools$",
      r"echo \"=== ", r"GPU=\"\$\(nvidia-smi", r"nvidia-smi --query-gpu=name --format=csv,noheader$",
      r"case \"\$GPU\" in \*(L40S|B200)\*\) ;; \*\) echo \"not a (L40S|B200): exiting\"; exit 4;; esac$",
      r"\(cd /scratch/yh6210/llmadd_torch/pretrain_data && sha256sum -c -\) <<'SHA' \|\| exit 3$", r"[0-9a-f]{64}  (train|val)\.bin$",
      r"SHA$", r"echo \"INTENT ", r"ARGS=", r"RC=0$", r"/scratch/yh6210/miniconda3/envs/open_r1/bin/python - \"",
      r"bash /scratch/yh6210/llmadd_torch/runs/(ptlr_1b|pttpp)/maybe_select.sh ", r"exit \$RC$", r"PY$"]
CHECK7_LINES = set(T.CHECK7.splitlines())


def check7(name, new, old, allowed_args, extra_lines=()):
    ad = arg_diff(new, old)
    judge(name, set(ad) <= allowed_args, f"ARGS differ beyond the design in {sorted(set(ad) - allowed_args)}: {ad}")
    bad = [l for l in ok_lines(line_diff(new, old), L7 + list(extra_lines)) if l[1:] not in CHECK7_LINES]
    judge(name, not bad, f"lines differ beyond the design: {bad[:4]}")
    judge(name, new.count(T.CHECK7) == 1, "CHECK7 not verbatim once")
    return ad


def nearest7(F):
    lr = {"none": {"190m": "4.8e-3", "500m": "9.6e-3"}, "win-ctx": {"190m": "1.2e-3", "500m": "1.2e-3"}}[F["arm"]]
    sz = "500m" if F["size"] == "505m" else "190m"
    p = RUNS / "ptlr" / f"job_{sz}-lr{lr[sz]}-7pct-{F['arm']}-s{F['seed']}.sbatch"
    return p


def nearest_b200(F):
    lr = {"none": "2.4e-3", "win-ctx": "6e-4"}[F["arm"]]
    return RUNS / "ptlr_1b" / f"job_1b-lr{lr}-7pct-{F['arm']}-s{F['seed']}.sbatch"


def nearest2(F):
    on = {"none": ("4.8e-3", "9.6e-3", "1.92e-2"), "win-ctx": ("3e-4", "6e-4", "1.2e-3")}[F["arm"]]
    lr = F["lr"] if F["lr"] in on else ("4.8e-3" if F["arm"] == "none" else "6e-4")
    s = F["seed"] if F["seed"] < 4 else F["seed"]
    p = RUNS / "pt2s505" / f"job_pt2s505-lr{lr}-{F['arm']}-s{s}.sbatch"
    if not p.exists():   # pt2s505 seeds 4-7 ran only at its picks
        p = RUNS / "pt2s505" / f"job_pt2s505-lr{'4.8e-3' if F['arm'] == 'none' else '6e-4'}-{F['arm']}-s{s}.sbatch"
    return p


def intent_of(text):
    m = re.search(r"\"\$OUT/check.json\" '(\{.*\})' <<'PY'", text)
    return json.loads(m.group(1))


L2 = [r"#SBATCH --job-name=", r"#SBATCH --time=", r"# ", r"R=/scratch/yh6210/llmadd_torch/runs/(pt2s505|pttpp); OUT=", r"cd \"\$R/code2?/tools\"$",
      r"grep -q \"pretrain_range.py 93e22bf4c4c0\" \"\$R/code2?/COMMIT\"", r"echo \"INTENT ", r"ARGS=",
      r"grep -qxF 'model: [\d.]+M parameters; gradient checkpointing on' ", r"\[ \"\$got\" = [0-9a-f]{64} \]", r"echo \"## dry_inputs rows = the reference",
      r"RC=0$", r"\"\$PY\" - \"\$OUT/\$TAG/run_whole_", r"  \"\$OUT/check.json\" '\{", r"echo \"## 5. selection hook",
      r"bash \"\$R/maybe_select.sh\" ", r"echo \"## done ", r"exit \$RC$", r"if \[ \"\$RC\" = 0 \]; then rm -f \"\$OUT/\$TAG/model_whole_",
      r"\"\$PY\" - \"\$OUT/\$TAG/\{runfile\}\"", r"echo \"## done \$\(date -u \+%FT%TZ\)\"$"]
INTENT_OK = {"steps", "n_params", "geo_layers", "read_layers", "eval_every_steps", "evals", "events", "on_frac", "lm0_ref",
             "args.tokens", "args.lr", "args.d_model", "args.n_layers", "args.n_heads", "args.d_ff", "args.tag", "args.ckpt_every",
             "args.eval_every", "args.seed"}


def check2(name, new, old, F):
    ad = arg_diff(new, old)
    allow = {"--out", "--tag", "--tokens", "--eval_every", "--ckpt_every", "--lr"} | DIM
    judge(name, set(ad) <= allow, f"ARGS differ beyond the design in {sorted(set(ad) - allow)}")
    judge(name, "--ckpt_every" not in flags(new) and "--save_model" in flags(new), "2%: no ckpt, model saved (offline_lm)")
    bad = ok_lines(line_diff(new, old), L2)
    judge(name, not bad, f"lines differ beyond the design: {bad[:4]}")
    In, Io = intent_of(new), intent_of(old)
    dk = {k for k in In if k != "args" and In[k] != Io.get(k)} | {f"args.{k}" for k in In["args"] if In["args"][k] != Io["args"].get(k)}
    dk |= {k for k in Io if k not in In} | {f"args.{k}" for k in Io["args"] if k not in In["args"]}
    judge(name, dk <= INTENT_OK, f"intent fields differ beyond the design: {sorted(dk - INTENT_OK)}")
    judge(name, new.count(G2.CHECK_PY) == 1, "CHECK_PY not verbatim once")
    judge(name, In["steps"] == F["S"] and In["evals"] == F["evals"] and In["args"]["tokens"] == float(F["T"])
          and In["args"]["eval_every"] == F["E"] and In["events"] == F["events"], "intent != the design's facts")
    return ad, dk


def all_jobs():
    """stage A of every group (the generator's own rendering) + stage X / C examples (facts() + job7 / job2)"""
    out = []
    for gid in T.all_groups():
        for name, txt, F in T.render(gid):
            out.append((name, txt, F))
    for gid, c in T.GROUPS.items():   # one X (beyond the upper edge) and one C job per arm, seeds as the design
        for a in T.ARMS:
            for st, lr, s in (("X", T.step_out(T.GRID[gid][a][1], "upper"), c["sel"][-1]), ("C", T.GRID[gid][a][0], c["conf"][0])):
                F = T.facts(gid, a, lr, s, st)
                txt = T.job2(G2, F)[1] if c["setting"] == "2" else T.job7(F, slot=0 if c["hw"] == "b200" else None)
                out.append((F["name"] + f" [{st} rendering]", txt, F))
    return out


HEAD = {"l40s": ["#SBATCH --account=torch_pr_625_general", "#SBATCH --partition=l40s_public", "#SBATCH --qos=gpu48"],
        "b200": ["#SBATCH --account=torch_pr_520_tandon_priority", "#SBATCH --constraint=b200", '#SBATCH --comment="preemption=yes"',
                 "#SBATCH --dependency=singleton", "#SBATCH --requeue"],
        "a100": ["#SBATCH --account=torch_pr_520_tandon_priority", "#SBATCH --partition=a100_tandon", "#SBATCH --qos=gpu48"]}


def check_one(name, txt, F):
    lines = txt.splitlines()
    judge(name, all(h in lines for h in HEAD[F["hw"]] + ["#SBATCH --nice=20", "#SBATCH --gres=gpu:1", "#SBATCH --requeue",
                                                         f"#SBATCH --time={F['tlim']}"]), "header: account / partition / nice / time")
    if F["setting"] == "2":
        old = nearest2(F).read_text()
        ad, dk = check2(name, txt, old, F)
        return ad
    if F["hw"] == "l40s":
        old = nearest7(F).read_text()
        allow = {"--out", "--tag", "--lr", "--tokens", "--eval_every", "--save_model", "--ckpt_every"} | DIM
        ad = check7(name, txt, old, allow)
        f = flags(txt)
        judge(name, ("--save_model" in f) == (F["stage"] == "C") and "--ckpt_every" not in f, "save_model only in C, no ckpt")
    else:
        old = nearest_b200(F).read_text()
        allow = {"--out", "--tag", "--lr", "--tokens", "--eval_every", "--save_model"} | (DIM if F["gid"] == T.BRIDGE else set())
        hdr = [r"#SBATCH --account=torch_pr_(520_tandon_priority|625_general)$", r"#SBATCH --dependency=singleton$", r"#SBATCH --cpus-per-task=16$",
               r"#SBATCH --mem=128G$", r"#SBATCH --open-mode=append$"]
        ad = check7(name, txt, old, allow, hdr)
        f = flags(txt)
        judge(name, f.get("--ckpt_every") == "500" and ("--save_model" in f) == (F["stage"] == "C"), "B200: ckpt 500; save_model only in C")
        if F["gid"] == T.BRIDGE:
            o87 = (RUNS / "ptlr" / f"job_500m-lr{F['lr']}-7pct-{F['arm']}-s{F['seed']}.sbatch").read_text()
            d87 = arg_diff(txt, o87)
            judge(name, set(d87) == {"--out", "--tag", "--save_model", "--ckpt_every"} and d87["--ckpt_every"] == ("2000", "500"),
                  f"bridge ARGS vs F87's 505M job: {d87}")
    if F["gid"] != T.BRIDGE:
        judge(name, f"maybe_select.sh {F['gid']} " in txt, "hook with the group id")
        judge(name, flags(txt)["--tokens"] == str(F["T"]) and flags(txt)["--eval_every"] == str(F["E"]), "tokens / eval_every = the design")
    else:
        judge(name, "maybe_select.sh" not in txt.replace("no hook", ""), "the bridge has no hook")
    return ad


jobs = all_jobs()
first = {}
for name, txt, F in jobs:
    ad = check_one(name, txt, F)
    k = (F["setting"], F["hw"], F["gid"] == T.BRIDGE)
    if k not in first:
        first[k] = (name, ad)
for k, (name, ad) in first.items():
    print(f"example {k}: {name}: ARGS differences vs the nearest submitted job: {ad}")
# planted errors: each must make some check fail
name, txt, F = next(j for j in jobs if j[2]["hw"] == "l40s")
name2, txt2, F2 = next(j for j in jobs if j[2]["setting"] == "2")
planted = {"extra --lam": (txt.replace("--seed ", "--lam 1 --seed ", 1), F),
           "--eval_every 250": (re.sub(r"--eval_every \d+", "--eval_every 250", txt, 1), F),
           "--tokens one step short": (txt.replace(f"--tokens {F['T']}", f"--tokens {F['T'] - 16384}", 1), F),
           "hook removed": (txt.replace(f"bash {T.R}/maybe_select.sh {F['gid']}", "true", 1), F),
           "nice removed": (txt.replace("#SBATCH --nice=20\n", "", 1), F),
           "2% intent events changed": (txt2.replace(json.dumps(F2["events"], separators=(",", ":")), "[[1,1],[2,0]]", 1), F2),
           "2% CHECK_PY line changed": (txt2.replace("abs(on_frac - I[\"on_frac\"]) < 1e-9", "abs(on_frac - I[\"on_frac\"]) < 1e-3", 1), F2)}
nf0 = len(fails)
caught = 0
for what, (t, Fp) in planted.items():
    before = len(fails)
    try:
        check_one("PLANTED " + what, t, Fp)
    except Exception as e:   # a parse failure counts as caught
        fails.append(f"PLANTED {what}: {e}")
    hit = len(fails) > before
    caught += hit
    print(("caught " if hit else "MISSED ") + what)
del fails[nf0:]
print(f"planted errors caught: {caught}/{len(planted)}")
ok = not fails and caught == len(planted)
print(f"JOBCHECK {'PASS' if ok else 'FAIL'}: {len(jobs)} job texts, {n_ok} checks passed" + (f"; failures: {fails[:5]}" if fails else ""))
sys.exit(0 if ok else 1)
