"""Burst task files for the rainbow6 MAIN design (rb6m: tools/pretrain_range.py Rev B --domains at PIN c434d29; intent
spec and pre-registered read-out: runs/rb6/AUDIT.md amendment A5; read-out: tools/rainbow6_summary.py).
User decisions 2026-10-19 (after the pilot, tasks 400-419): K = 2 domain lines per row (the pre-registered gate's K*), lr
1e-3, window (0.05, 0.4), ctx site, lam_rule each, lambda 1 per active prior (lambda 5 rejected: the priors made ~2/3 of
every update; "the manifold loss is only an auxiliary loss"), seeds 0-3; one all6 arm at lambda 0.3 per prior; every
arm at its own best lr: none's best lr at the 2% 53M setting is 8e-3 (Burst pt2b bracket) -> a none@8e-3 arm; the prior
arms' bracket is flat at 1e-3 vs 2e-3 (70.2 vs 69.9), so they stay at 1e-3. Arms (K = 2, 4 seeds each):
  none                 no prior (the same data)                                            REUSED pilot cell (seeds 0, 1)
  all6-lam1            helix3 on the operands + the five domain priors, lambda 1 each      REUSED pilot cell (seeds 0, 1)
  all6-lam0.3          as all6-lam1 with lambda 0.3 per prior
  only-<d>             one prior (d = numbers (helix3 on the operands), months, weekdays, clock, s4, dim), lambda 1
  all6-shuf            all six priors on SHUFFLED targets (numbers --target helix3_shuf = the per-seed permutation
                       default_rng(124 + seed) of 0..999; the others --prior_shuf: a far permutation of the items, CKA
                       to the true target < 0.15 except s4), lambda 1, the same window = the same dose
  none-lr8e-3          as none with lr 8e-3
Reuse: the pilot's K2 none (tag rb6pb-53m-2pct-k2-none) and K2 all6-lam1 (rb6pb-53m-2pct-k2-all6-lam1) runs of seeds 0, 1
(tasks 401, 404, 419, 415; same PIN, same arguments) are these cells: only seeds 2, 3 are generated, under the pilot's
TAGS, so seeds 0-3 form one cell in the summary. Every new cell is tagged rb6mb-53m-2pct-k2-<arm> (mb = main on Burst).
40 new runs: 2 + 2 + 4 (lam0.3) + 24 (singles) + 4 (shuf) + 4 (none@8e-3), tasks 430-469 (418-429 = pilot spares),
+ 8 confirm runs 470-477 (seeds 4-7 of none / all6-lam1) + 4 runs 490-493 (none-lr8e-3 seeds 4-7) = 52 tasks.
Numbering = submission order (Burst's tick submits ready tasks in file-name order, MAX_GPU_JOBS = 12):
  430-433  none s2, s3, all6-lam1 s2, s3: completes the core all6 - none contrast at n = 4; every other contrast is paired
           against these cells
  434-437  all6-shuf s0-s3: the geometry-specificity control of any all6 effect (all6 - all6-shuf)
  438-461  the six single-prior arms, seed-major (438-443 = seed 0 of numbers, months, weekdays, clock, s4, dim; then
           seeds 1, 2, 3): after every 6 tasks each single arm has the same seeds, so a partial read-out stays paired and
           balanced across the six (own-domain effect, stacking, cross-domain harm)
  462-465  all6-lam0.3 s0-s3 (the dose question: less harm without losing the gain?)
  466-469  none-lr8e-3 s0-s3 (the baseline at its own best lr: does it change a floor verdict?)
  The first wave (12) is 430-441 if no other GPU task is ready (asserted at render: no task file 430-469 exists).
  470-477  the confirm block (user decision 2026-10-19: ON, CONFIRM = True): seeds 4-7 of the two headline cells (none and
           all6-lam1 under the pilot's tags), seed-major. Seeds 0-3 decide, 4-7 confirm; seeds 2-7 decide the pilot's tier-E
           hypothesis (rainbow6_summary A5).
  478-479, 483-489  reserved for re-runs (480-482 = toyb tasks); --check accepts task files there.
  490-493  none-lr8e-3 seeds 4-7 (coordinator decision 2026-10-19): the fresh seeds 2-7 (n = 6) of F1b / FE against
           none@8e-3, and its seeds not used to select lr 8e-3 (winner's curse); written with the confirm block.
Name NNN-<tag>-s<seed>; `# time:` 02:00:00 as the pilot (pilot K2 tasks: 52-55 min wall incl. dry run, offline LM and
checks; projected_run_h 0.84; planning 0.92 A100-h).
Every task (header '# after: 331-env311 332-data'), as the pilot (make_rb6_tasks.py) with the intent generalised to
any set of active priors, shuffled targets and lr:
1. inputs: data sha256 = Torch's; --dry_inputs through burst/pin_run.sh at PIN: the model line, the prior target (none ->
   no 'target features' line; helix3 / helix3_shuf -> 1000 x 10), and from the 'domains:' line (the trainer's DINFO) for
   EACH of the 6 domains its "prior", "shuf" and "lam" (+ the numbers "target"): active domains prior true / lam lambda,
   inactive prior false / lam 0.0, shuf true exactly for the shuffled arm; the numbers line first; the content scan; the
   sha256 + count of the DRY_GREP lines = Torch's for (K 2, seed) (exit 3 otherwise, before any training);
2. training (--save_model --ckpt_every 2000; a tick retry keeps $OUT and resumes); 3. offline_lm.py on the saved model;
4. intent checks of the run JSON (CHECK_PY below) -> $OUT/check.{json,txt}, exit 4 on a fail: args, shas, facts,
   lambda per domain, data hash / step-0 lm = Torch's, layout, splits, formats; prior arms: the window, each ACTIVE prior
   on exactly in the window and applied on >= 95% of the on-steps, each INACTIVE one never on / never applied / no
   geo_<d> / no weighted gradient (numbers inactive: geo == 0), geo > GEO_TOL and finite gradients, and the CKA to the
   target ACTUALLY USED rises by >= CKA_RISE (true target: item-state cka_<d> / cka_ops_helix3 as the pilot; shuffled
   target: 1 - geo_<d> (resp. 1 - geo) on the training batch, the only logged CKA to the shuffled table) with the CKA to
   the true target logged; the lambda actually used: gnorm_prior_<d> / gnorm_true_<d> = lambda for every unshuffled
   active prior at every on-eval; none arms: no prior ever on. The info line adds the gradient split (window medians of
   grad_lm / T, the clip factor, grad_prior / grad_lm) and the library versions of the env311 python (python, torch,
   transformers, numpy, tokenizers; compared across runs by the summary). nvidia-smi every 60 s -> gpu.csv.
--check also asserts that the pilot generator's renderings are unchanged: its 18 tasks = tasks/400-417 byte for byte
after undoing its one later edit (9cbfca7: geo tolerance, 'future renders only'), and 418 / 419 = 408 / 409.
Idempotent as make_rb6_tasks.py: an existing task file is never overwritten; a name with status/ but no file is refused.
A6 (user decisions 2026-10-19 after the main read-out; amendment A6 of report/audits/rb6_A5_main_design.md, pre-registered
before these runs): 494-501 = all6-lam1 at lr 4e-3 / 8e-3 and 502-509 = none at lr 3e-3 / 1.6e-2, seeds 0-3 (seed-major),
each identical to all6-lam1 / none except --lr and the tag (rb6mb-53m-2pct-k2-<arm>); the same intent checks (lr in the
args); read by the A6 section of tools/rainbow6_summary.py. A diverged run (1.6e-2 may) is data, as in pt2main: its
check passes when nothing else fails.
A7 (user decisions 2026-10-19 after the A6 read-out; amendment A7 of report/audits/rb6_A5_main_design.md, pre-registered
before these runs): 510-521 = per seed s = 0-3 none at lr 3.2e-2 (seed s), all6-lam1 at lr 5e-4 (seed s), none at lr 1.6e-2
(seed s + 4); each identical to none / all6-lam1 except --lr and the tag; read by the A7 section of rainbow6_summary.py.
A8 (amendment A8, 2026-10-20, pre-registered before any A7 result and any A8 run): 522-529 = all6-lam1@1e-3 and
none@1.6e-2 at the fresh seeds 8-11 (per seed the pair), the A5 / A6 cells' arguments with the seed; Torch references of
seeds 8-11 computed with the same method after recomputing seed 4 exactly; read by rainbow6_summary's A8 section.
A8.3 (amendment A8.3, 2026-10-20, user decision before any A8 result was read): 530-537 = all6-lam1@1e-3 and
none@1.6e-2 at the fresh seeds 12-15 (per seed the pair), the A8 tasks' arguments with the seed (asserted against the
seed-8 tasks); Torch references of seeds 12-15 computed with the same method after reproducing seed 8 exactly; with
seeds 8-11 the confirmatory set of rainbow6_summary's A8.3 section.
    python tools/make_rb6m_tasks.py [--check] [--show] [--no_confirm] [--no_a6] [--no_a7] [--no_a8] [--no_a83]
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_rb6_tasks as P   # the pilot generator: constants, flags(), hours(), window_events(), render() (unchanged)

REPO = P.REPO
PIN, PIN_SHA, TOY_SHA = P.PIN, P.PIN_SHA, P.TOY_SHA
K, SEEDS, FIRST = 2, (0, 1, 2, 3), 430
ALL6, EXT, W = P.ALL6, P.EXT, P.W
DOMS6 = ("numbers",) + EXT
# arm -> (numbers --target, active priors, shuffled priors, --lam (None = not passed), --lr as passed)
SPEC = {"none": ("none", (), (), None, "1e-3"),
        "all6-lam1": ("helix3", DOMS6, (), "1", "1e-3"),
        "all6-lam0.3": ("helix3", DOMS6, (), "0.3", "1e-3"),
        **{f"only-{d}": ("helix3" if d == "numbers" else "none", (d,), (), "1", "1e-3") for d in DOMS6},
        "all6-shuf": ("helix3_shuf", DOMS6, DOMS6, "1", "1e-3"),
        "none-lr8e-3": ("none", (), (), None, "8e-3"),
        # A6 (user decisions 2026-10-19 after the main read-out; report/audits/rb6_A5_main_design.md A6): every arm at
        # its own best lr, the prior arm too, and none's lr bracket closed on s4 / dim; = all6-lam1 / none except --lr
        "all6-lam1-lr4e-3": ("helix3", DOMS6, (), "1", "4e-3"),
        "all6-lam1-lr8e-3": ("helix3", DOMS6, (), "1", "8e-3"),
        "none-lr3e-3": ("none", (), (), None, "3e-3"),
        "none-lr1.6e-2": ("none", (), (), None, "1.6e-2"),
        # A7 (user decisions 2026-10-19 after the A6 read-out; report/audits/rb6_A5_main_design.md A7): none's s4 bracket
        # one step up, all6's lr bracket one step down; = none / all6-lam1 except --lr
        "none-lr3.2e-2": ("none", (), (), None, "3.2e-2"),
        "all6-lam1-lr5e-4": ("helix3", DOMS6, (), "1", "5e-4")}
A6_ARMS = ("all6-lam1-lr4e-3", "all6-lam1-lr8e-3", "none-lr3e-3", "none-lr1.6e-2")
A7_ARMS = ("none-lr3.2e-2", "all6-lam1-lr5e-4")   # + none-lr1.6e-2 at the fresh seeds 4-7 (A7_ORDER)
REUSED = {"none": "rb6pb-53m-2pct-k2-none", "all6-lam1": "rb6pb-53m-2pct-k2-all6-lam1"}   # pilot cells: seeds 0, 1 ran
DESC = {"none": "no prior (the same data); REUSED pilot cell rb6pb (seeds 0, 1 = tasks 401, 404)",
        "all6-lam1": "helix3 on the operands + the five domain priors on their item tokens (ctx), one window (0.05, 0.4), "
                     "lambda 1 per active prior (lam_rule each); REUSED pilot cell rb6pb (seeds 0, 1 = tasks 419, 415)",
        "all6-lam0.3": "as all6-lam1 with lambda 0.3 per active prior",
        **{f"only-{d}": f"only the {d} prior ({'helix3 on the operands' if d == 'numbers' else 'on its item tokens, ctx'}), "
                        "window (0.05, 0.4), lambda 1 (lam_rule each); the other five priors off" for d in DOMS6},
        "all6-shuf": "all six priors on SHUFFLED targets (numbers helix3_shuf, the others a far permutation of the items), "
                     "window (0.05, 0.4), lambda 1 each: the same dose as all6-lam1 on the wrong geometry",
        "none-lr8e-3": "as none with lr 8e-3 (none's best lr at the 2% 53M setting, Burst pt2b bracket)",
        "all6-lam1-lr4e-3": "as all6-lam1 with lr 4e-3 (A6: the prior arm at its own best lr; grid 1e-3 / 4e-3 / 8e-3)",
        "all6-lam1-lr8e-3": "as all6-lam1 with lr 8e-3 (A6: the prior arm at its own best lr; same-lr contrast with none@8e-3)",
        "none-lr3e-3": "as none with lr 3e-3 (A6: none's lr bracket on s4 / dim; grid 1e-3 / 3e-3 / 8e-3 / 1.6e-2)",
        "none-lr1.6e-2": "as none with lr 1.6e-2 (A6: none's lr bracket on s4 / dim, the upper edge; may diverge: data)",
        "none-lr3.2e-2": "as none with lr 3.2e-2 (A7: none's s4 bracket one step above 1.6e-2; may diverge: data)",
        "all6-lam1-lr5e-4": "as all6-lam1 with lr 5e-4 (A7: all6's lr bracket, open at 1e-3, one step down)"}
ORDER = ([("none", 2), ("none", 3), ("all6-lam1", 2), ("all6-lam1", 3)] + [("all6-shuf", s) for s in SEEDS]
         + [(f"only-{d}", s) for s in SEEDS for d in DOMS6] + [("all6-lam0.3", s) for s in SEEDS]
         + [("none-lr8e-3", s) for s in SEEDS])
# Confirm block (coordinator 2026-10-19: "the most stable result first"; user decision 2026-10-19: ON): seeds 4-7 of the two
# headline cells (none and all6-lam1, the pilot's tags), tasks 470-477, seed-major so every finished pair is paired. Seeds
# 0-3 decide, 4-7 confirm (readout_rules.MAIN / CONFIRM); seeds 2-7 are the fresh seeds that decide the pilot's tier-E
# hypothesis (rainbow6_summary A5, family FE). CONFIRM = False would render and assert it without writing it.
CONFIRM = True
# Re-run numbers (a task that ran keeps its status/, so a re-run needs a new name): 478, 479 and 483-489 (480-482 are the
# toyb tasks pushed 2026-10-19). A re-run task file there is made by hand from the failed task's rendering (new number,
# same tag / seed / args); the summary refuses two runs of one (arm, K, seed), so the failed run's directory is left out
# of the read-out command.
RERUN_NUMBERS = (478, 479, 483, 484, 485, 486, 487, 488, 489)
CONFIRM_SEEDS = (4, 5, 6, 7)
CONFIRM_ORDER = [(a, s) for s in CONFIRM_SEEDS for a in ("none", "all6-lam1")]
# The second confirm block (coordinator decision 2026-10-19): none-lr8e-3 seeds 4-7 as tasks 490-493 (478-489 stay
# reserved / toyb), so that F1b and FE also have the fresh seeds 2-7 (n = 6) against none@8e-3; their seeds 4-7 were not
# used to select lr 8e-3 (pt2b chose it on seeds 0-3), so they also measure its winner's curse. Written with the first.
FIRST2, CONFIRM2_ORDER = 490, [("none-lr8e-3", s) for s in CONFIRM_SEEDS]
# The A6 block (user decisions 2026-10-19 after the main read-out; pre-registered in amendment A6 of
# report/audits/rb6_A5_main_design.md before these runs): all6-lam1 at lr 4e-3 and 8e-3 (494-501), then none at lr
# 3e-3 and 1.6e-2 (502-509), seeds 0-3, seed-major in each block (every finished pair of lrs is paired). The Torch
# references are the seeds' (lr acts only after the step-0 forward: verified on the dry logs of all 16 and at step 0 of
# two of them). Written with the main tasks (A6 = True).
A6 = True
FIRST6 = 494
A6_ORDER = ([(a, s) for s in SEEDS for a in ("all6-lam1-lr4e-3", "all6-lam1-lr8e-3")]
            + [(a, s) for s in SEEDS for a in ("none-lr3e-3", "none-lr1.6e-2")])
# The A7 block (user decisions 2026-10-19 after the A6 read-out; pre-registered in amendment A7 of
# report/audits/rb6_A5_main_design.md before these runs): 510-521, seed-major, per step (none@3.2e-2 seed s, all6@5e-4
# seed s, none@1.6e-2 seed s + 4) for s = 0-3. The Torch references are the seeds' (A5: seeds 0-3, the confirm block's
# 4-7; lr acts only after the step-0 forward: dry logs of all 12 and step 0 of one cell checked). 483-489 stay for re-runs.
A7 = True
FIRST7 = 510
A7_ORDER = [x for s in SEEDS for x in (("none-lr3.2e-2", s), ("all6-lam1-lr5e-4", s), ("none-lr1.6e-2", s + 4))]
# The A8 block (amendment A8 of report/audits/rb6_A5_main_design.md, 2026-10-20, pre-registered before any A7 result and
# any A8 run): all6-lam1@1e-3 and none@1.6e-2 at the fresh seeds 8-11 (both sides fresh), 522-529, seed-major (each seed's
# pair together). Same arguments as the A5 / A6 cells except the seed; Torch references of seeds 8-11 below.
A8 = True
FIRST8 = 522
A8_SEEDS = (8, 9, 10, 11)
A8_ORDER = [(a, s) for s in A8_SEEDS for a in ("all6-lam1", "none-lr1.6e-2")]
# The A8.3 block (amendment A8.3 of report/audits/rb6_A5_main_design.md, 2026-10-20, user decision before any A8 result
# was read): the confirmatory set is seeds 8-15 of all6-lam1@1e-3 and none@1.6e-2; seeds 12-15 are these 8 new tasks
# 530-537 (the numbers after the highest existing task 529; 483-489 stay reserved for re-runs), seed-major as A8, the
# same arguments as the A8 tasks except the seed (asserted); Torch references of seeds 12-15 below.
A83 = True
FIRST83 = 530
A83_SEEDS = (12, 13, 14, 15)
A83_ORDER = [(a, s) for s in A83_SEEDS for a in ("all6-lam1", "none-lr1.6e-2")]
GEO_TOL = -1e-3   # float CKA exceeds 1 at a perfect fit: pilot 411 geo_weekdays -3.2e-5, 416 -2.4e-4 (rc 4 under the
#                   pilot's > 0); the pilot generator's later -1e-4 would still fail 416
CKA_RISE = P.CKA_RISE   # 0.1 (see make_rb6_tasks.py: a sanity floor against a broken prior path, not a strength test)
# Torch reference values (K 2). Seeds 0, 1 = the pilot's (make_rb6_tasks.DRY / LM0 / DSHA0 / DOMS), reproduced exactly
# (bitwise) with the same method before computing seeds 2-7 (session scratchpad rb6m/refs/: run_refs.py, step0.py,
# out/*.log + *.json; 2026-10-19), all with the PINNED tools (git archive c434d29 tools), the task's own arguments and tag,
# on the Torch data (sha256 = DATA): DRY = sha256 + count of the DRY_GREP lines of --dry_inputs; LM0 = the step-0 LM loss
# on CPU (bf16 autocast; the trainer's own main() up to its step-0 forward, the vocabulary log-softmax in row chunks,
# reduced by the same nll_loss kernel); DSHA0 = data_sha after step 0; DOMS = the split facts. Arm-independent (DRY and
# DOMS identical over all 10-11 arms at seeds 0-3, over none / all6-lam1 / none-lr8e-3 at 4-7; step-0 lm / data_sha over
# none, only-s4, only-numbers, all6-shuf, none-lr8e-3, all6-lam1 where computed), so one value per seed. Seeds 4-7: the
# confirm blocks (none, all6-lam1: 470-477; none-lr8e-3: 490-493; arm-independent references).
DRY = {**{s: P.DRY[(K, s)] for s in (0, 1)},
       2: ('964d1b8b61a15305ed080f7bf994f7792f929018dd0bb539d8543130e43cdfdf', 88),
       3: ('ea21c901acad5eca30cf8477ec32e533475804ee59f58024c2a59e92b2fd015e', 88),
       4: ('1f07da985f7d7acf2186a6e5ff45d37d955105a7abd870f94c9a019121ddb471', 88),
       5: ('b0fe6b14623ddf6345562f17d7e9348902f1f5a1968e9e991fea6ef72b80bd04', 88),
       6: ('7db5d83aefd58e6303eff4207106c81eedc25f9e5a6c7ffab0b76c9e1ae83e18', 88),
       7: ('4465f4f1ae6a72290cab8ac0c414cdd6df0e4788445b5c15600a984f6a492e95', 88)}
LM0 = {**{s: P.LM0[(K, s)] for s in (0, 1)},
       2: 11.570906639099121, 3: 11.604287147521973, 4: 11.58791446685791, 5: 11.569442749023438, 6: 11.584881782531738, 7: 11.593488693237305}
DSHA0 = {**{s: P.DSHA0[(K, s)] for s in (0, 1)},
         2: '36b63d307358a0a4', 3: 'df68c5608eedaa9a', 4: '9451e7743f0607ed', 5: '99801f9e457357aa', 6: '20b49004bf55264c', 7: '83ce379a1c8bd6cd'}
DOMS = {**{s: P.DOMS[s] for s in (0, 1)},
        2: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 476, "n_held": 442, "n_heldE": 144, "n_readout": 330, "n_readoutE": 122, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 194, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 344, "n_held": 342, "n_readout": 142, "readout": "held_no_negtwin_nonid", "n_items": 27}},
        3: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 478, "n_held": 442, "n_heldE": 146, "n_readout": 339, "n_readoutE": 112, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 200, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 344, "n_held": 342, "n_readout": 132, "readout": "held_no_negtwin_nonid", "n_items": 27}},
        4: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 480, "n_held": 442, "n_heldE": 149, "n_readout": 330, "n_readoutE": 108, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 202, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 344, "n_held": 342, "n_readout": 128, "readout": "held_no_negtwin_nonid", "n_items": 27}},
        5: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 480, "n_held": 442, "n_heldE": 148, "n_readout": 338, "n_readoutE": 121, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 197, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 342, "n_held": 344, "n_readout": 156, "readout": "held_no_negtwin_nonid", "n_items": 27}},
        6: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 482, "n_held": 442, "n_heldE": 150, "n_readout": 327, "n_readoutE": 115, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 204, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 344, "n_held": 342, "n_readout": 150, "readout": "held_no_negtwin_nonid", "n_items": 27}},
        7: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 475, "n_held": 442, "n_heldE": 144, "n_readout": 338, "n_readoutE": 109, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 209, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 342, "n_held": 344, "n_readout": 130, "readout": "held_no_negtwin_nonid", "n_items": 27}}}
# A8 (2026-10-20): the fresh seeds 8-11 for all6-lam1 / none-lr1.6e-2, the same method (refs/run_refs_a8.py on the pinned
# tools: --dry_inputs of none and all6-lam1, equal DRY / DOMS; step0.py of none at every seed and all6-lam1 at seed 8,
# equal), after recomputing seed 4 = its recorded DRY / DOMS / LM0 / DSHA0 exactly (a8/refs8.py)
DRY.update({8: ('9cc7c5f0e911d73ba2c5b626b45c633a00d91175c7b65ff44a1db4b48d33d49c', 88), 9: ('61ae59aeb59e9d5eff590f8b097411872f8de47e46931e4bccf298c61768ade7', 88), 10: ('6bb4ed0e31c43f6fc0d8de5e6223f5ad4520842f0ebc0265529d8b0fad64f4eb', 88), 11: ('43d9194d521dacc2e01aa82bbe6abb263352f339e365dab307e0184895581cb1', 88)})
LM0.update({8: 11.61306095123291, 9: 11.606510162353516, 10: 11.57604694366455, 11: 11.581033706665039})
DSHA0.update({8: 'a3181fb4799d4806', 9: 'c38dac2d365eb3ae', 10: '0c4d582a9d024701', 11: 'fd9ea9b83ff3522d'})
DOMS.update({8: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 477, "n_held": 442, "n_heldE": 146, "n_readout": 342, "n_readoutE": 111, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 200, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 342, "n_held": 344, "n_readout": 154, "readout": "held_no_negtwin_nonid", "n_items": 27}},
             9: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 477, "n_held": 442, "n_heldE": 146, "n_readout": 335, "n_readoutE": 113, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 198, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 342, "n_held": 344, "n_readout": 124, "readout": "held_no_negtwin_nonid", "n_items": 27}},
             10: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 480, "n_held": 442, "n_heldE": 149, "n_readout": 334, "n_readoutE": 112, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 195, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 344, "n_held": 342, "n_readout": 138, "readout": "held_no_negtwin_nonid", "n_items": 27}},
             11: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 474, "n_held": 442, "n_heldE": 142, "n_readout": 337, "n_readoutE": 103, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 203, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 342, "n_held": 344, "n_readout": 136, "readout": "held_no_negtwin_nonid", "n_items": 27}}})
# A8.3 (2026-10-20): the fresh seeds 12-15 for all6-lam1 / none-lr1.6e-2, the same method (the session scratchpad's
# rb6m/a83/run_refs_a83.py = refs/run_refs_a8.py on the pinned tools, out_a83: --dry_inputs of both task arms, equal DRY /
# DOMS; step 0 of both arms at every seed, equal), after recomputing seed 8 = its recorded DRY / DOMS / LM0 / DSHA0 exactly
# (a83/refs83.py; step 0 of none / all6-lam1 / none-lr1.6e-2 at seed 8 and none at seed 4, bitwise). The step-0 forward ran
# under torch.no_grad (a83/step0_ng.py: no saved activations, < 8 GB per process; the same forward kernels, so the same
# value bitwise at seeds 8 and 4)
DRY.update({12: ('a1e4d5ab94eea75386a90cd675d9a8e8929c4e43f801b0b4f101db17aa73f94e', 88), 13: ('54117a94f23394c5466da5700163ebce032c0e4e637748c77b6e88db43676724', 88), 14: ('274cfca9662b60b3bbd75649cc53cf0c780495eec6984ced7127125fef93097c', 88), 15: ('4e70d40ee2b2050179992fdaf95fa201ccb15d2e3c24c01c808c47d64d88b89e', 88)})
LM0.update({12: 11.590072631835938, 13: 11.586889266967773, 14: 11.586933135986328, 15: 11.587129592895508})
DSHA0.update({12: '274975fa0ed6a352', 13: '5ca3a1c782d2f1b0', 14: '194f10dd32bff764', 15: '47ab5685b5108572'})
DOMS.update({12: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 472, "n_held": 442, "n_heldE": 141, "n_readout": 342, "n_readoutE": 109, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 201, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 344, "n_held": 342, "n_readout": 146, "readout": "held_no_negtwin_nonid", "n_items": 27}},
             13: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 476, "n_held": 442, "n_heldE": 146, "n_readout": 328, "n_readoutE": 115, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 199, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 344, "n_held": 342, "n_readout": 162, "readout": "held_no_negtwin_nonid", "n_items": 27}},
             14: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 474, "n_held": 442, "n_heldE": 143, "n_readout": 329, "n_readoutE": 111, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 197, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 342, "n_held": 344, "n_readout": 136, "readout": "held_no_negtwin_nonid", "n_items": 27}},
             15: {"months": {"n_train": 118, "n_held": 106, "n_heldE": 40, "n_readout": 106, "n_readoutE": 40, "readout": "held", "n_items": 12}, "weekdays": {"n_train": 38, "n_held": 34, "n_heldE": 12, "n_readout": 34, "n_readoutE": 12, "readout": "held", "n_items": 7}, "clock": {"n_train": 478, "n_held": 442, "n_heldE": 148, "n_readout": 332, "n_readoutE": 114, "readout": "held_wrap", "n_items": 24}, "s4": {"n_train": 345, "n_held": 231, "n_readout": 199, "readout": "held_fresh_nonid", "n_items": 24}, "dim": {"n_train": 342, "n_held": 344, "n_readout": 154, "readout": "held_no_negtwin_nonid", "n_items": 27}}})
assert all(P.DRY[(K, s)][1] == 88 for s in (0, 1))
# the pilot generator's one later edit (9cbfca7, 'future renders only'): its current renderings carry the new geo check
GEO_NEW = ('chk("geo > -1e-4 for every prior (float CKA can exceed 1 by ~3e-5 at a perfect fit; 2026-10-19) and 0 < '
           'grad_prior < inf at every eval where the priors acted (before a divergence)",\n        (bool(on) or diverged) '
           'and all(e["geo"] > -1e-4 and all(e.get(f"geo_{d}") is None or e[f"geo_{d}"] > -1e-4 for d in EXT)')
GEO_OLD = ('chk("geo > 0 for every prior and 0 < grad_prior < inf at every eval where the priors acted (before a '
           'divergence)",\n        (bool(on) or diverged) and all(e["geo"] > 0 and all(e.get(f"geo_{d}") is None or '
           'e[f"geo_{d}"] > 0 for d in EXT)')


def tag_of(arm):
    return REUSED.get(arm, f"rb6mb-53m-2pct-k{K}-{arm}")


def arm_flags(arm):
    """The $ARM part of the trainer arguments (the pilot's string for the reused cells)."""
    tgt, pri, shf, lam, _ = SPEC[arm]
    if not pri:
        return "--target none"
    s = f"--target {tgt} {W} --prior_domains {','.join(pri)}"
    return s + (f" --prior_shuf {','.join(shf)}" if shf else "") + (f" --lam {lam}" if lam is not None else "")


def want_flags(arm):
    """{flag: value} the rendered arguments must contain for this arm (None = absent: the trainer default)."""
    tgt, pri, shf, lam, _ = SPEC[arm]
    if not pri:
        return {"--target": "none", "--schedule": None, "--window": None, "--site": None, "--lam_rule": None,
                "--prior_domains": None, "--prior_shuf": None, "--lam": None}
    return {"--target": tgt, "--schedule": "window", "--window": "0.05,0.4", "--site": "ctx", "--lam_rule": "each",
            "--prior_domains": ",".join(pri), "--prior_shuf": ",".join(shf) if shf else None, "--lam": lam}


def lam_of(arm):
    """The intended lambda of every ACTIVE prior (lam_domains of the run); inactive domains 0.0."""
    tgt, pri, shf, lam, _ = SPEC[arm]
    return float(lam) if pri else 0.0


def number(arm, seed):
    if (arm, seed) in A83_ORDER:
        return FIRST83 + A83_ORDER.index((arm, seed))
    if (arm, seed) in A8_ORDER:
        return FIRST8 + A8_ORDER.index((arm, seed))
    if (arm, seed) in A7_ORDER:
        return FIRST7 + A7_ORDER.index((arm, seed))
    if (arm, seed) in A6_ORDER:
        return FIRST6 + A6_ORDER.index((arm, seed))
    if (arm, seed) in CONFIRM2_ORDER:
        return FIRST2 + CONFIRM2_ORDER.index((arm, seed))
    return FIRST + (ORDER + CONFIRM_ORDER).index((arm, seed))


CHECK_PY = r'''import csv, json, math, statistics, sys
from pathlib import Path
run, offl, gpu, pinned, data, out_json = sys.argv[1:7]
I = json.loads(sys.argv[7])
B = ("b11", "b12", "b13", "b22", "b23", "b33")
EXT = ("months", "weekdays", "clock", "s4", "dim")
ALLD = ("numbers",) + EXT
ACT, SHUF, TOL = I["active"], I["shuf"], I["geo_tol"]
mean = statistics.fmean
checks = []
def chk(name, ok, value):
    checks.append({"check": name, "ok": bool(ok), "value": value})
r = json.loads(Path(run).read_text()); c = r["curve"]; a = r["args"]
want = dict(I["args"], data=data, out=str(Path(run).parent))
diff = {k: [a.get(k), v] for k, v in want.items() if a.get(k) != v}
chk(f"run args = intent ({len(want)} fields)", not diff, diff)
chk("code_sha " + I["code_sha"], r.get("code_sha") == I["code_sha"], r.get("code_sha"))
chk("dep_sha toy_struct.py " + I["toy_sha"], r.get("dep_sha", {}).get("toy_struct.py") == I["toy_sha"], r.get("dep_sha"))
keys = ("steps", "n_params", "grad_ckpt_on", "geo_layers", "read_layers", "lm_chunk", "eval_every_steps", "lam_rule")
got = {k: r.get(k) for k in keys}
chk("run facts = intent " + json.dumps({k: I[k] for k in keys}), all(got[k] == I[k] for k in keys), got)
chk("lambda per domain = intent " + json.dumps(I["lam_domains"]), r.get("lam_domains") == I["lam_domains"], r.get("lam_domains"))
dp = {d: [r["domains"][d].get(k) for k in ("prior", "shuf", "lam")] for d in ALLD}
dw = {d: [d in ACT, d in SHUF, I["lam_domains"][d]] for d in ALLD}
chk("domains.<d> prior / shuf / lam = intent (active " + ",".join(ACT) + "; shuffled " + (",".join(SHUF) or "none") + ")",
    dp == dw and r["domains"]["numbers"].get("target") == I["args"]["target"], dp)
st = [e["step"] for e in c]
chk(f"finished: {I['evals']} evals, increasing, the last at step {I['steps'] - 1}",
    len(c) == I["evals"] and st == sorted(set(st)) and st[-1] == I["steps"] - 1, [len(c), st[-1]])
# a diverged run is data, not a failed check (as pt2main): onset = the first eval with a non-finite LM / val / prior value
fin = lambda v: v is None or math.isfinite(v)
bad = [e["step"] for e in c if not (fin(e["lm"]) and fin(e["val_loss"]) and fin(e["geo"]) and fin(e.get("grad_prior"))
                                     and all(fin(e.get(f"geo_{d}")) for d in EXT))]
diverged = bool(bad)
pre = [e for e in c if not bad or e["step"] < bad[0]]
# data: the same mixture as Torch's dry run (step 0) and as every other arm of this (K, seed)
chk(f"data_sha after step 0 = Torch's {I['data_sha0']}", c[0].get("data_sha") == I["data_sha0"], c[0].get("data_sha"))
chk("step-0 lm within 1e-3 of the Torch CPU reference (same init and first batch; the prior is off at step 0)",
    abs(c[0]["lm"] - I["lm0_ref"]) <= 1e-3, c[0]["lm"] - I["lm0_ref"])
chk("the numbers line is the first line after the text in every row (A2)", abs(c[-1]["first_line_frac"]["numbers"] - 1) < 1e-12,
    c[-1]["first_line_frac"])
ts = c[-1]["token_share"]["numbers"]
chk(f"numbers token share in {I['numbers_share']} (the 2% setting; Torch dry 2.12-2.13%)",
    I["numbers_share"][0] <= ts <= I["numbers_share"][1], ts)
sl = c[-1]["slot_frac"]
chk(f"domain lines per slot after the numbers line: K = {I['K']} slots, every slot within 1/K +- 0.05 of every domain's lines",
    all(len(v) == I["K"] and all(abs(x - 1 / I["K"]) <= 0.05 for x in v) for v in sl.values()), sl)
dm = {d: {k: r["domains"][d].get(k) for k in I["dom"][d]} for d in EXT}
chk("domain splits = Torch's dry run " + json.dumps(I["dom"]), dm == I["dom"], dm)
nf = {d: len(r["domains"][d].get("formats", {})) for d in EXT}
fk = [f"{d}_p{t}_{s}" for d in EXT for t in range(1, I["n_fmt"][d] + 1) for s in ("held", "seen")]
chk(f"held-out-format templates {I['n_fmt']}, scored at every eval ({len(fk)} sets)",
    nf == I["n_fmt"] and all(e.get(k) is not None for e in c for k in fk), nf)
# priors: generalised to any set of active priors (A5)
cka_used = cka_true = None
if I["events"]:
    on_frac = c[-1]["prior_on_frac"]
    chk(f"prior_events == {I['events']}", r.get("prior_events") == I["events"], r.get("prior_events"))
    chk(f"prior_on_frac == {I['on_frac']:.6f}", abs(on_frac - I["on_frac"]) < 1e-9, on_frac)
    chk("prior_on_domains: the active priors (" + ",".join(ACT) + ") on exactly at the evals with prior_on, every other domain never",
        all(len(e["prior_on_domains"]) == 6 and all(e["prior_on_domains"][d] == (e["prior_on"] if d in ACT else 0) for d in ALLD)
            for e in c), {d: sum(e["prior_on_domains"][d] for e in c) for d in ALLD})
    if "numbers" in ACT:
        chk("numbers prior applied on every on-step (ctx: prior_used_frac == prior_used_frac_numbers == prior_on_frac)",
            abs(c[-1]["prior_used_frac"] - on_frac) < 1e-9 and abs(c[-1]["prior_used_frac_numbers"] - on_frac) < 1e-9,
            [c[-1]["prior_used_frac"], c[-1]["prior_used_frac_numbers"], on_frac])
    else:
        chk("numbers prior off: never applied (prior_used_frac_numbers 0), no numbers loss term (geo == 0 at every eval), "
            "no weighted numbers gradient", c[-1]["prior_used_frac_numbers"] == 0 and all(e["geo"] == 0 for e in c)
            and all("gnorm_prior_numbers" not in e for e in c), [c[-1]["prior_used_frac_numbers"], sorted({e["geo"] for e in c})[:3]])
    used = {d: c[-1][f"prior_used_frac_{d}"] for d in EXT}
    chk("every active domain prior applied on >= 95% of the on-steps; every inactive one never",
        all((u >= 0.95 * on_frac - 1e-12) if d in ACT else (u == 0) for d, u in used.items()), used)
    ng = {d: sum(e.get(f"geo_{d}") is not None for e in c) for d in EXT}
    chk("geo_<d> logged at >= 95% of the evals for every active domain prior, never for an inactive one",
        all((n >= 0.95 * len(c)) if d in ACT else (n == 0) for d, n in ng.items()), ng)
    on = [e for e in pre if e["prior_on"]]
    gk = {d: sum(f"gnorm_prior_{d}" in e for e in on) for d in ALLD}
    chk("weighted prior gradients at the on-evals from the active priors only (each at >= 90% of the on-evals)",
        all((n >= 0.9 * len(on)) if d in ACT else (n == 0) for d, n in gk.items()) and all(
            not any(k.startswith("gnorm_prior_") for k in e) for e in c if not e["prior_on"]), gk)
    # the lambda actually used (A5, build audit): an unshuffled active prior's weighted gradient is lambda x its true term's
    # (the trainer scales the same vector: pilot 419 / 415 1.0000, 418 5.0000), at every on-eval that logged both norms
    lr_ = {d: [e[f"gnorm_prior_{d}"] / e[f"gnorm_true_{d}"] for e in on if e.get(f"gnorm_prior_{d}") is not None
               and e.get(f"gnorm_true_{d}")] for d in ACT if d not in SHUF}
    chk("gnorm_prior_<d> / gnorm_true_<d> = lambda (rel. 1e-3) at every on-eval, every unshuffled active prior",
        all(abs(v / I["lam_domains"][d] - 1) < 1e-3 for d, vs in lr_.items() for v in vs),
        {d: [round(min(vs), 5), round(max(vs), 5), len(vs)] if vs else None for d, vs in lr_.items()})
    # geo_<d> is None on a batch with < 4 sites or < 3 distinct items of d (the prior skipped there; counted above)
    chk(f"geo > {TOL} for every active prior (float CKA exceeds 1 at a perfect fit: pilot 411 / 416) and 0 < grad_prior < inf at "
        "every eval where the priors acted (before a divergence)",
        (bool(on) or diverged) and all(("numbers" not in ACT or e["geo"] > TOL) and all(e.get(f"geo_{d}") is None or e[f"geo_{d}"] > TOL for d in EXT)
                                      and 0 < e.get("grad_prior", 0) < math.inf for e in on), len(on))
    if on:
        we = on[-1]
        g0 = lambda e, d: (e["geo"] if d == "numbers" else e.get(f"geo_{d}"))
        ck = lambda e, d: (e["cka_ops_helix3"] if d == "numbers" else e[f"cka_{d}"])
        cka_true = {d: round(ck(we, d) - ck(c[0], d), 4) for d in ACT}
        # the target actually used: the true one -> the item-state CKA (as the pilot); a shuffled one -> 1 - geo on the
        # training batch (the prior's own value; the only logged CKA to the shuffled table; logged at step 0 too)
        cka_used = {d: (None if g0(we, d) is None or g0(c[0], d) is None else round(g0(c[0], d) - g0(we, d), 4)) if d in SHUF
                    else cka_true[d] for d in ACT}
        chk(f"CKA to the target USED rises by >= {I['cka_rise']} from step 0 to the window's last eval (step {we['step']}), every "
            "active prior (true target: cka_<d> / cka_ops_helix3; shuffled: 1 - geo_<d> / 1 - geo); CKA to the true target in the info"
            + ("; shuffled priors: a FAIL of only this check is a finding (a permuted target fitted weakly), the run stays valid"
               " (A5)" if SHUF else ""),
            diverged or all(v is not None and v >= I["cka_rise"] for v in cka_used.values()), {"used": cka_used, "true": cka_true})
else:
    chk("no prior ever on (prior_on_frac 0, no events, no domain prior on or applied, no prior loss term or gradient)",
        c[-1]["prior_on_frac"] == 0 and r.get("prior_events") == [] and all(
            not any(e["prior_on_domains"].values()) for e in c) and all(c[-1][f"prior_used_frac_{d}"] == 0 for d in ALLD)
        and all(e["geo"] == 0 and not any(k.startswith(("geo_", "gnorm_prior_")) for k in e) for e in c),
        c[-1]["prior_on_frac"])
o = json.loads(Path(offl).read_text()).get(str(Path(run).resolve())) if Path(offl).exists() else None
chk("offline_lm on the full val, |orig_minus_logged| < 0.002 (skipped if the run diverged)",
    o is not None and o["max_windows"] == 0 and (diverged or abs(o["orig_minus_logged"]) < 0.002), o and o["orig_minus_logged"])
held = lambda e: mean([e[f"held_{b}"] for b in B])
m = r.get("metrics", {})
rate = [(q["sec"] - p["sec"]) / (q["step"] - p["step"]) for p, q in zip(c, c[1:]) if q["sec"] > p["sec"] and q["step"] > p["step"]]
util, mem = [], []
for row in csv.reader(open(gpu)) if Path(gpu).exists() else []:
    try:
        util.append(int(row[2].split()[0])); mem.append(int(row[4].split()[0]))
    except (IndexError, ValueError):
        pass
f3 = lambda k: m.get(k, {}).get("final3")
gs = []   # the gradient split before the clip at the on-evals (user: the prior is auxiliary)
for e in c:
    if e.get("prior_on") and e.get("grad_prior") is not None and e.get("grad_lm"):
        gl_, gp_, cs_ = e["grad_lm"], e["grad_prior"], e["grad_cos"]
        T_ = math.sqrt(max(gl_ * gl_ + gp_ * gp_ + 2 * gl_ * gp_ * cs_, 0.0))
        gs.append((gl_ / T_ if T_ else None, 1 / max(1.0, T_), gp_ / gl_))
med = lambda i: statistics.median([g[i] for g in gs if g[i] is not None]) if gs else None
import importlib.metadata as ilm, platform
def ver(p):
    try:
        return ilm.version(p)
    except ilm.PackageNotFoundError:
        return None
# this python = the env311 one pin_run trains with (burst/pin_run.sh PY311): the library versions of the run (A5)
versions = {"python": platform.python_version(), **{p: ver(p) for p in ("torch", "transformers", "numpy", "tokenizers")}}
info = {"versions": versions,"tag": a["tag"], "seed": a["seed"], "K": a["extra_lines"], "lr": a["lr"], "active": ACT, "shuf": SHUF,
        "lam_domains": r.get("lam_domains"), "numbers_area": round(100 * mean([held(e) for e in c]), 2),
        "numbers_final3": f3("numbers"), "readout_final3": {d: f3(f"{d}_readout") for d in EXT},
        "readoutE_final3": {d: f3(f"{d}_readoutE") for d in EXT}, "seen_final3": {d: f3(f"{d}_seen") for d in EXT},
        "format_readout_final3": {d: [f3(f"{d}_p{t}_readout") for t in range(1, I["n_fmt"][d] + 1)] for d in EXT},
        "final_cka": {d: c[-1].get(f"cka_{d}") for d in EXT}, "cka_ops_helix3": c[-1].get("cka_ops_helix3"),
        "cka_rise_used": cka_used, "cka_rise_true": cka_true,
        "grad_window_median": {"lm_share_of_T": med(0), "clip_factor": med(1), "prior_over_lm": med(2), "n_on_evals": len(gs),
                               "frac_prior_gt_lm": (sum(g[2] > 1 for g in gs) / len(gs)) if gs else None},
        "seen_all_ge_80": all((f3(f"{d}_seen") or 0) >= 80 for d in EXT),
        "lm0": c[0]["lm"], "final_val_bpb": c[-1]["val_bits_per_byte"], "bpb_full": o and o["bpb_full"], "diverged": diverged,
        "diverged_at_step": bad[0] if bad else None, "sec_per_step_median": statistics.median(rate) if rate else None,
        "projected_run_h": statistics.median(rate) * r["steps"] / 3600 if rate else None,
        "gpu_util_median": statistics.median(util) if util else None, "gpu_mem_max_MiB": max(mem) if mem else None,
        "gpu_samples": len(util), "pinned_calls": len(Path(pinned).read_text().splitlines()) if Path(pinned).exists() else 0}
ok = all(k["ok"] for k in checks)
Path(out_json).write_text(json.dumps({"pass": ok, "checks": checks, "info": info}, indent=1))
for k in checks:
    print(("PASS " if k["ok"] else "FAIL ") + k["check"] + f"  ({k['value']})")
print(json.dumps(info))
print("CHECKS", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 4)
'''


def intent(arm, seed):
    """The intent of one task: everything CHECK_PY asserts on the run JSON (also used by the CPU tests of CHECK_PY)."""
    tgt, pri, shf, lam, lr = SPEC[arm]
    events, on_frac = P.window_events(P.STEPS) if pri else ([], 0.0)
    if pri:
        assert events == [[916, 1], [7324, 0]], events
    lam_rule = "each" if pri else "norm"
    return {"code_sha": PIN_SHA, "toy_sha": TOY_SHA, "steps": P.STEPS, "n_params": P.N_PARAMS, "grad_ckpt_on": False,
            "geo_layers": [2, 4], "read_layers": [2, 4], "lm_chunk": 0, "eval_every_steps": P.EVAL_EVERY,
            "lam_rule": lam_rule, "lam_domains": {d: (lam_of(arm) if d in pri else 0.0) for d in DOMS6}, "K": K,
            "active": list(pri), "shuf": list(shf), "geo_tol": GEO_TOL,
            "evals": len(range(0, P.STEPS, P.EVAL_EVERY)) + ((P.STEPS - 1) % P.EVAL_EVERY != 0), "events": events,
            "on_frac": on_frac, "lm0_ref": LM0.get(seed), "data_sha0": DSHA0.get(seed), "dom": DOMS.get(seed),
            "n_fmt": P.N_FMT, "cka_rise": CKA_RISE, "numbers_share": [0.019, 0.024],
            "args": {"tokenization": "whole", "target": tgt, "site": "ctx", "schedule": "window" if pri else "whole",
                     "window": "0.05,0.4", "arith_lines": 1, "lr": float(lr), "lam": float(lam) if lam is not None else P.LAM_DEFAULT,
                     "tokens": 3e8, "bs": P.BS, "seq": P.SEQ, "d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536,
                     "seed": seed, "tag": f"_{tag_of(arm)}", "save_model": True, "ckpt_every": 2000, "geo_layers": None,
                     "eval_every": P.EVAL_EVERY, "n_eval": 150, "grad_ckpt": "auto", "lm_chunk_tokens": -1, "untie": False,
                     "gate_thr": 0.5, "gate_start": 0.0, "gate_hyst": 0.0, "text_min": 16, "n_prefix": 256, "stop_after": -1,
                     "dry_inputs": False, "domains": ALL6, "extra_lines": K, "prior_domains": ",".join(pri),
                     "prior_shuf": ",".join(shf), "lam_rule": lam_rule, "cyc_held": 0.4, "cyc_heldE": 0.5, "s4_train": 0.6,
                     "dim_train": 0.5, "dom_eval_rows": 200, "dom_seen": 150}}


def trainer_args(arm, seed):
    """The trainer ARGS string of the task (shell variables $DATA $TOK $OUT $TAG $ARM $SEED unexpanded)."""
    return (f"--data $DATA --tokenizer $TOK --tokenization whole --out $OUT/$TAG $ARM --arith_lines 1 --domains {ALL6} "
            f"--extra_lines {K} --lr {SPEC[arm][4]} --seed $SEED --tag _$TAG --save_model --ckpt_every 2000")


def expand(arm, seed, args=None):
    """ARGS with $ARM, $SEED and $TAG expanded ($DATA $TOK $OUT stay)."""
    return (args or trainer_args(arm, seed)).replace("$ARM", arm_flags(arm)).replace("$SEED", str(seed)).replace("$TAG", tag_of(arm))


def dry_block(arm):
    """The input checks of --dry_inputs that depend on the arm: prior target, and per domain prior / shuf / lam."""
    tgt, pri, shf, lam, _ = SPEC[arm]
    if tgt == "none":
        ta = """! grep -q '^target features' "$OUT/dry_inputs.txt" || { echo "## a numbers prior target in a run without one"; exit 3; }"""
    else:
        ta = """grep -qxF 'target features (1000, 10)' "$OUT/dry_inputs.txt" || { echo "## not a helix3 table (1000 x 10)"; exit 3; }"""
    specs = " ".join(f"{d}:{str(d in pri).lower()}:{str(d in shf).lower()}:{json.dumps(lam_of(arm) if d in pri else 0.0)}"
                     for d in DOMS6)
    # one dict per domain in the 'domains:' line (no nested dicts in it: the trainer drops dict values), so
    # '"<d>": {[^}]*}' is that domain's block; "dim" cannot match inside another key (a quote precedes the name). The
    # values are matched as fixed strings followed by ',' or '}' (json.dumps of the trainer's floats: 1.0, 0.3, 0.0)
    lb = ("dl=$(grep -m1 '^domains: ' \"$OUT/dry_inputs.txt\" || true)\n"
          "has() { grep -qF -e \"$1,\" -e \"$1}\" <<< \"$blk\"; }\n"
          f"for sp in {specs}; do\n"
          "  IFS=: read -r d p s l <<< \"$sp\"\n"
          "  blk=$(grep -oE \"\\\"$d\\\": \\{[^}]*\\}\" <<< \"$dl\" || true)\n"
          "  has \"\\\"prior\\\": $p\" && has \"\\\"shuf\\\": $s\" && has \"\\\"lam\\\": $l\" ||\n"
          "    { echo \"## domain $d not prior $p / shuf $s / lam $l in the domains: line\"; exit 3; }\n"
          "done\n"
          f"grep -qF '\"numbers\": {{\"target\": \"{tgt}\", ' <<< \"$dl\" || {{ echo \"## numbers target not {tgt}\"; exit 3; }}")
    return ta, lb, specs


def render(arm, seed):
    """(task name, tag, file text, trainer ARGS with the shell variables) of one (arm, seed)."""
    tgt, pri, shf, lam, lr = SPEC[arm]
    num, tag = number(arm, seed), tag_of(arm)
    name = f"{num}-{tag}-s{seed}"
    args = trainer_args(arm, seed)
    f = P.flags(expand(arm, seed, args))
    want = {"--data": "$DATA", "--tokenizer": "$TOK", "--tokenization": "whole", "--out": f"$OUT/{tag}", "--arith_lines": "1",
            "--domains": ALL6, "--extra_lines": str(K), "--lr": lr, "--seed": str(seed), "--tag": f"_{tag}",
            "--save_model": True, "--ckpt_every": "2000", **want_flags(arm)}
    for kk, v in want.items():
        assert f.get(kk) == v, (name, kk, f.get(kk), v)
    assert set(f) == {kk for kk, v in want.items() if v is not None}, (name, sorted(f))   # every other flag = its default
    assert P.LIM_X * P.PLAN_H[K] <= P.hours(P.TLIM) < P.LIM_X * P.PLAN_H[K] + 0.5, (name, P.TLIM)
    if arm in REUSED:   # a reused pilot cell: the pilot's arguments with another seed, the pilot's tag
        assert arm_flags(arm) == P.ARM[{"none": "none", "all6-lam1": "all6-lam1"}[arm]], arm
        assert tag == f"rb6pb-53m-2pct-k{K}-{arm}" and seed in (2, 3) + CONFIRM_SEEDS + A8_SEEDS + A83_SEEDS, (arm, seed)
    confirm = seed in CONFIRM_SEEDS
    assert not confirm or arm in REUSED or arm == "none-lr8e-3" or (arm, seed) in A7_ORDER, (arm, seed)
    I = intent(arm, seed)
    for key in ("lm0_ref", "data_sha0", "dom"):
        assert I[key] is not None, f"{name}: no Torch reference {key} for (K {K}, seed {seed}); compute it first"
    assert DRY.get(seed) is not None, f"{name}: no DRY hash for (K {K}, seed {seed}); compute it on Torch first"
    dry, n_dry = DRY[seed]
    ij = json.dumps(I, separators=(",", ":"))
    assert "'" not in ij
    target_assert, lam_loop, specs = dry_block(arm)
    runfile = f"run_whole_{tgt}_seed{seed}_{tag}.json"
    nm6 = lambda ds: "all 6" if tuple(ds) == DOMS6 else ",".join(ds)
    lam_txt = f"lambda {lam_of(arm)} on {nm6(pri)}" if pri else "no prior"
    about = textwrap.fill(f"Cell {tag}: {DESC[arm]}; 53M, 3e8 tokens = {P.STEPS} steps of {P.BS} x {P.SEQ} on {P.DATA[0]}, "
                          f"lr {lr}; 2% numbers (1 line per row, first after the text) + {K} domain lines per row (months, "
                          f"weekdays, clock, s4, dim cycled); seed {seed}.", 120, initial_indent="# ", subsequent_indent="# ")
    conf_txt = ("" if not confirm else
                "\n# CONFIRM SEED (seeds 4-7 of the two headline cells; user decision 2026-10-19): seeds 0-3 decide, seeds"
                "\n# 4-7 confirm; seeds 2-7 decide the pilot's tier-E hypothesis (rainbow6_summary A5)." if arm in REUSED else
                "\n# CONFIRM SEED (none@8e-3 seeds 4-7, coordinator decision 2026-10-19): the fresh seeds 2-7 of F1b / FE"
                "\n# against none@8e-3, and seeds not used to select lr 8e-3 (rainbow6_summary A5).")
    if arm in A6_ARMS:
        conf_txt = ("\n# A6 (user decisions 2026-10-19 after the main read-out; amendment A6 of report/audits/rb6_A5_main_design.md,"
                    "\n# pre-registered before these runs): each arm at its own best lr, the prior arm too (all6-lam1 at 4e-3 /"
                    "\n# 8e-3), and none's lr bracket on s4 / dim closed (none at 3e-3 / 1.6e-2); read by rainbow6_summary's A6"
                    "\n# section, never by the A5 rules.")
    if (arm, seed) in A7_ORDER:
        conf_txt = ("\n# A7 (user decisions 2026-10-19 after the A6 read-out; amendment A7 of report/audits/rb6_A5_main_design.md,"
                    "\n# pre-registered before these runs): none's s4 bracket one step up (none at 3.2e-2, seeds 0-3), all6's lr"
                    "\n# bracket one step down (all6-lam1 at 5e-4, seeds 0-3), none@1.6e-2 at the fresh seeds 4-7; read by"
                    "\n# rainbow6_summary's A7 section, never by the A5 / A6 rules.")
    if (arm, seed) in A8_ORDER:
        conf_txt = ("\n# A8 (amendment A8 of report/audits/rb6_A5_main_design.md, 2026-10-20, pre-registered before any A7 result"
                    "\n# and any A8 run): all6-lam1@1e-3 vs none@1.6e-2 at the fresh seeds 8-11, both sides fresh; S4 read on"
                    "\n# s4_hard; read by rainbow6_summary's A8 section, never by the A5 / A6 / A7 rules.")
    if (arm, seed) in A83_ORDER:
        conf_txt = ("\n# A8.3 (amendment A8.3 of report/audits/rb6_A5_main_design.md, 2026-10-20, user decision before any A8"
                    "\n# result was read): all6-lam1@1e-3 vs none@1.6e-2 at the fresh seeds 12-15; with A8's seeds 8-11 the"
                    "\n# confirmatory set (8 pairs); decision = the s4_hard area, one-sided paired t; read by rainbow6_summary's"
                    "\n# A8.3 section, never by the A5 / A6 / A7 / A8 rules.")
    text = f"""# kind: gpu
# time: {P.TLIM}
# cpus: 8
# after: 331-env311 332-data
# rb6 rainbow6 MAIN design (2026-10-19 user decisions after the pilot: K = 2, lr 1e-3, lambda 1 per active prior,
# lam_rule each, window (0.05, 0.4), ctx; arms none / all6 / all6 lambda 0.3 / the six single priors / all6 on shuffled
# targets / none at lr 8e-3 x seeds 0-3; intent spec and pre-registered read-out runs/rb6/AUDIT.md amendment A5).
# Generated by tools/make_rb6m_tasks.py: change the generator, not this file.{conf_txt}
{about}
# Read-out: tools/rainbow6_summary.py over the pilot's reused K2 none / all6-lam1 runs + this batch (tags rb6pb- / rb6mb-).
# Trainer pinned at {PIN} (tools/pretrain_range.py sha256 {PIN_SHA}..., toy_struct.py {TOY_SHA}...) through burst/pin_run.sh.
# 1. inputs: data sha256 = Torch's; --dry_inputs: model line, prior target ({tgt}), per domain prior / shuf / lam
#    ({lam_txt}{'; shuffled: ' + nm6(shf) if shf else ''}), numbers line first, and the sha256 of the selected decoded
#    lines = the value computed on Torch for (K {K}, seed {seed}) (exit 3 otherwise, before any training)
# 2. training with --save_model --ckpt_every 2000 (a retry keeps $OUT and resumes from the checkpoint)
# 3. offline_lm.py on the saved model (full val.bin of {P.DATA[0]})
# 4. intent checks of the run JSON -> $OUT/check.{{json,txt}} (exit 4 on a fail)
# nvidia-smi utilisation / memory every 60 s -> $OUT/gpu.csv (planning: {P.PLAN_H[K]:.2f} A100-h; limit {P.TLIM}).
set -euo pipefail
SEED={seed}; TAG={tag}; ARM="{arm_flags(arm)}"
ROOT=${{LLMADD_ROOT:-/scratch/$USER}}; DATA=$ROOT/llmadd_data/{P.DATA[0]}; TOK=$ROOT/llmadd_data/olmo2_tok; PIN={PIN}
[ -s "$OUT/gpu.csv" ] || echo "timestamp, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]" > "$OUT/gpu.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader -l 60 >> "$OUT/gpu.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT
{{ date -u +%FT%TZ; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "cpus $(nproc)"; }} >> "$OUT/node.txt" 2>&1 || true

echo "## 1. inputs $(date -u +%FT%TZ)"
(cd "$DATA" && sha256sum -c -) <<'SHA' || exit 3
{P.DATA[1]}  train.bin
{P.DATA[2]}  val.bin
SHA
ARGS="{args}"
bash burst/pin_run.sh $PIN pretrain_range.py $ARGS --dry_inputs > "$OUT/dry_inputs.txt" 2>&1 || {{ cat "$OUT/dry_inputs.txt"; exit 3; }}
cat "$OUT/dry_inputs.txt"
grep -qxF '{P.MODEL_LINE}' "$OUT/dry_inputs.txt" || {{ echo "## not the intended model size / gradient checkpointing state"; exit 3; }}
{target_assert}
{lam_loop}
grep -q '^first line after the text (share of rows): {{"numbers": 1.0, ' "$OUT/dry_inputs.txt" || {{ echo "## the numbers line is not first in every row"; exit 3; }}
grep -q '^content scan: 640 training rows, no held-out prompt of ' "$OUT/dry_inputs.txt" || {{ echo "## no content scan"; exit 3; }}
n=$(set +o pipefail; grep -cE "{P.DRY_GREP}" "$OUT/dry_inputs.txt")
got=$(set +o pipefail; grep -E "{P.DRY_GREP}" "$OUT/dry_inputs.txt" | sha256sum | cut -d' ' -f1)
[ "$n" = {n_dry} ] && [ "$got" = {dry} ] || {{ echo "## dry_inputs differ from the rows computed on Torch ($n lines, sha256 $got)"; exit 3; }}
grep -q " {PIN_SHA}[0-9a-f]* tools/pretrain_range.py " "$OUT/PINNED.txt" || {{ echo "## the pinned trainer is not {PIN_SHA}"; exit 3; }}

echo "## 2. training $(date -u +%FT%TZ)"
bash burst/pin_run.sh $PIN pretrain_range.py $ARGS
echo "## 3. offline LM $(date -u +%FT%TZ)"
bash burst/pin_run.sh $PIN offline_lm.py --data "$DATA" --tokenizer "$TOK" --out "$OUT/offline_lm.json" "$OUT/$TAG"
echo "## 4. intent checks $(date -u +%FT%TZ)"
"$ROOT/llmadd_env311/bin/python" - "$OUT/$TAG/{runfile}" "$OUT/offline_lm.json" "$OUT/gpu.csv" "$OUT/PINNED.txt" "$DATA" \\
  "$OUT/check.json" '{ij}' <<'PY' | tee "$OUT/check.txt"
{CHECK_PY}PY
"""
    return name, tag, text, args


def tasks(confirm=True, a6=True, a7=True, a8=True, a83=True):
    """Every (arm, seed) of the main design (+ the confirm blocks, + the A6 block, + the A7 block, + A8, + A8.3) in
    task-number order: [(name, tag, text, args, arm, seed)]."""
    return sorted((render(a, s) + (a, s) for a, s in ORDER + (CONFIRM_ORDER + CONFIRM2_ORDER if confirm else [])
                   + (A6_ORDER if a6 else []) + (A7_ORDER if a7 else []) + (A8_ORDER if a8 else [])
                   + (A83_ORDER if a83 else [])), key=lambda t: t[0])


def pilot_unchanged(repo):
    """The pilot generator's 18 renderings = tasks/400-417 byte for byte after undoing 9cbfca7 (its geo line: 'future renders
    only'); 418 / 419 = 408 / 409 byte for byte (re-runs under spare numbers). -> list of problems (empty = unchanged)."""
    bad = []
    for name, _, text, *_ in P.tasks():
        assert text.count(GEO_NEW) == 1, (name, "the pilot generator's geo line changed again")
        p = repo / "tasks" / f"{name}.sh"
        if not p.exists() or p.read_text() != text.replace(GEO_NEW, GEO_OLD):
            bad.append(name)
    for new, old in (("418-rb6pb-53m-2pct-k2-all6-s0", "408-rb6pb-53m-2pct-k2-all6-s0"),
                     ("419-rb6pb-53m-2pct-k2-all6-lam1-s0", "409-rb6pb-53m-2pct-k2-all6-lam1-s0")):
        pn, po = repo / "tasks" / f"{new}.sh", repo / "tasks" / f"{old}.sh"
        if not (pn.exists() and po.exists() and pn.read_bytes() == po.read_bytes()):
            bad.append(new)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="render and assert only; write nothing")
    ap.add_argument("--show", action="store_true", help="print the trainer arguments of every task")
    ap.add_argument("--no_confirm", action="store_true", help="do not write the confirm blocks 470-477 (seeds 4-7 of none / "
                    "all6-lam1) and 490-493 (none-lr8e-3 seeds 4-7); ON by the decisions of 2026-10-19; --check renders them")
    ap.add_argument("--no_a6", action="store_true", help="do not write the A6 block 494-509 (ON by the user's decisions of "
                    "2026-10-19 after the main read-out); --check renders it either way")
    ap.add_argument("--no_a7", action="store_true", help="do not write the A7 block 510-521 (ON by the user's decisions of "
                    "2026-10-19 after the A6 read-out); --check renders it either way")
    ap.add_argument("--no_a8", action="store_true", help="do not write the A8 block 522-529 (ON by amendment A8, 2026-10-20);"
                    " --check renders it either way")
    ap.add_argument("--no_a83", action="store_true", help="do not write the A8.3 block 530-537 (ON by amendment A8.3, "
                    "2026-10-20); --check renders it either way")
    ap.add_argument("--repo", default=str(REPO))
    a = ap.parse_args()
    repo = Path(a.repo)
    bad = pilot_unchanged(repo)
    assert not bad, f"the pilot generator's renderings differ from tasks/400-419: {bad}"
    print("pilot generator unchanged: its 18 renderings = tasks/400-417 (9cbfca7's geo line undone), 418 / 419 = 408 / 409")
    T = tasks()
    names = [t[0] for t in T]
    assert len(set(names)) == len(names) == 96 and [int(n[:3]) for n in names] == list(range(FIRST, FIRST + 48)) + list(
        range(FIRST2, FIRST2 + 4)) + list(range(FIRST6, FIRST6 + 16)) + list(range(FIRST7, FIRST7 + 12)) + list(
        range(FIRST8, FIRST8 + 8)) + list(range(FIRST83, FIRST83 + 8)), names
    cells = {}
    for t in T:
        cells.setdefault(t[1], set()).add(t[5])
    assert len(cells) == len(SPEC) == 17, sorted(cells)
    assert all(v == ({2, 3, *CONFIRM_SEEDS, *A8_SEEDS, *A83_SEEDS} if tg == tag_of("all6-lam1") else {2, 3, *CONFIRM_SEEDS} if any(
        tag_of(r) == tg for r in REUSED) else {*SEEDS, *CONFIRM_SEEDS, *A8_SEEDS, *A83_SEEDS} if tg == tag_of("none-lr1.6e-2") else
        {*SEEDS, *CONFIRM_SEEDS} if tg == tag_of("none-lr8e-3") else set(SEEDS)) for tg, v in cells.items()), cells
    # A8.3: the 8 tasks 530-537, seed-major; each = the A8 seed-8 task of its arm except the seed and the seed's Torch
    # references (the trainer flags, and the intent except args.seed / lm0_ref / data_sha0 / dom)
    assert [(t[4], t[5]) for t in T[88:]] == A83_ORDER and all("# A8.3 (amendment A8.3" in t[2] for t in T[88:]) and not any(
        "# A8.3 (amendment A8.3" in t[2] for t in T[:88]), "A8.3 block 530-537"
    for t in T[88:]:
        fa, fb = P.flags(expand(t[4], t[5], t[3])), P.flags(expand(t[4], 8))
        assert t[3] == trainer_args(t[4], 8) and {k: v for k, v in fa.items() if k != "--seed"} == {
            k: v for k, v in fb.items() if k != "--seed"} and fa["--seed"] == str(t[5]), (t[0], fa, fb)
        ia, ib = intent(t[4], t[5]), intent(t[4], 8)
        assert {k: v for k, v in ia.items() if k not in ("args", "lm0_ref", "data_sha0", "dom")} == {
            k: v for k, v in ib.items() if k not in ("args", "lm0_ref", "data_sha0", "dom")} and {
            k: v for k, v in ia["args"].items() if k != "seed"} == {k: v for k, v in ib["args"].items() if k != "seed"} and (
            ia["args"]["seed"] == t[5]), (t[0], "intent differs from the A8 seed-8 task beyond the seed / references")
        assert DRY[t[5]] not in [DRY[s] for s in DRY if s != t[5]] and DSHA0[t[5]] not in [DSHA0[s] for s in DSHA0 if s != t[5]], \
            (t[0], "a seed's Torch reference equals another seed's")
    # A8: the 8 tasks 522-529, seed-major; each = the A5 / A6 cell's arguments with the seed
    assert [(t[4], t[5]) for t in T[80:88]] == A8_ORDER and all("# A8 (amendment A8" in t[2] for t in T[80:88]) and not any(
        "# A8 (amendment A8" in t[2] for t in T[:80] + T[88:]), "A8 block 522-529"
    for t in T[80:88]:
        fa, fb = P.flags(expand(t[4], t[5], t[3])), P.flags(expand(t[4], 2 if t[4] == "all6-lam1" else 0))
        assert {k: v for k, v in fa.items() if k != "--seed"} == {k: v for k, v in fb.items() if k != "--seed"} and fa["--seed"] == str(t[5]), (t[0], fa, fb)
    # A7: the 12 tasks 510-521, seed-major; each identical to its 1e-3 arm except --lr and the tag (none-lr1.6e-2 at
    # seeds 4-7 = the A6 cell's tag and arguments with another seed)
    assert [(t[4], t[5]) for t in T[68:80]] == A7_ORDER and all("# A7 (user decisions" in t[2] for t in T[68:80]) and not any(
        "# A7 (user decisions" in t[2] for t in T[:68] + T[80:]), "A7 block 510-521"
    for t in T[68:80]:
        base = "all6-lam1" if t[4].startswith("all6") else "none"
        fa, fb = P.flags(expand(t[4], t[5], t[3])), P.flags(expand(base, t[5]))
        assert {k: v for k, v in fa.items() if k not in ("--lr", "--tag", "--out")} == {
            k: v for k, v in fb.items() if k not in ("--lr", "--tag", "--out")} and fa["--lr"] == SPEC[t[4]][4], (t[0], fa, fb)
    # A6: the 16 tasks 494-509, all6 lr arms first, seed-major; each identical to its 1e-3 arm except --lr and the tag
    assert [(t[4], t[5]) for t in T[52:68]] == A6_ORDER and all("# A6 (user decisions" in t[2] for t in T[52:68]) and not any(
        "# A6 (user decisions" in t[2] for t in T[:52] + T[68:]), "A6 block 494-509"
    for t in T[52:68]:
        base = "all6-lam1" if t[4].startswith("all6") else "none"
        fa, fb = P.flags(expand(t[4], t[5], t[3])), P.flags(expand(base, t[5]))
        assert {k: v for k, v in fa.items() if k not in ("--lr", "--tag", "--out")} == {
            k: v for k, v in fb.items() if k not in ("--lr", "--tag", "--out")} and fa["--lr"] == SPEC[t[4]][4], (t[0], fa, fb)
    # waves (MAX_GPU_JOBS 12): 430-441 = none / all6-lam1 seeds 2, 3 + all6-shuf s0-s3 + seed 0 of four single arms
    assert [(t[4], t[5]) for t in T[:8]] == ORDER[:8] and {t[4] for t in T[:4]} == set(REUSED), T[:8]
    assert [t[5] for t in T[8:32]] == [s for s in SEEDS for _ in DOMS6], "singles seed-major"
    assert [(t[4], t[5]) for t in T[40:48]] == CONFIRM_ORDER and [(t[4], t[5]) for t in T[48:52]] == CONFIRM2_ORDER and all(
        "# CONFIRM SEED" in t[2] for t in T[40:52]) and not any("# CONFIRM SEED" in t[2] for t in T[:40] + T[52:]), \
        "confirm blocks 470-477, 490-493"
    assert not set(RERUN_NUMBERS) & {int(n[:3]) for n in names} and not set(RERUN_NUMBERS) & set(range(480, 483))
    write_confirm = CONFIRM and not a.no_confirm
    write_a6 = A6 and not a.no_a6
    write_a7 = A7 and not a.no_a7
    write_a8 = A8 and not a.no_a8
    write_a83 = A83 and not a.no_a83
    if a.show:
        for name, tag, text, args, arm, seed in T:
            print(f"{name}{'' if seed not in CONFIRM_SEEDS else ' (confirm block: ' + ('written' if write_confirm else 'NOT written') + ')'}"
                  f"\n    {expand(arm, seed, args)}")
    tdir, sdir = repo / "tasks", repo / "status"
    foreign = sorted(p.name for p in list(tdir.glob("4[3-7][0-9]*.sh")) + list(tdir.glob("49[0-9]*.sh"))
                     + list(tdir.glob("50[0-9]*.sh")) + list(tdir.glob("51[0-9]*.sh")) + list(tdir.glob("52[0-9]*.sh"))
                     + list(tdir.glob("53[0-7]*.sh"))
                     if p.stem not in names and int(p.name[:3]) not in RERUN_NUMBERS)
    assert not foreign, f"task files in 430-479 / 490-537 not made by this generator (re-runs: {RERUN_NUMBERS}): {foreign}"
    reruns = sorted(p.name for p in tdir.glob("4[7-8][0-9]*.sh") if int(p.name[:3]) in RERUN_NUMBERS)
    if reruns:
        print(f"re-run task files in the reserved numbers (not made by this generator): {reruns}")
    drop = set()   # A8: by membership (the blocks' positions no longer matter)
    for flag, block, msg in ((write_a83, A83_ORDER, "A8.3 block (530-537) rendered and asserted, NOT written (--no_a83)"),
                             (write_a8, A8_ORDER, "A8 block (522-529) rendered and asserted, NOT written (--no_a8)"),
                             (write_a7, A7_ORDER, "A7 block (510-521) rendered and asserted, NOT written (--no_a7)"),
                             (write_a6, A6_ORDER, "A6 block (494-509) rendered and asserted, NOT written (--no_a6)"),
                             (write_confirm, CONFIRM_ORDER + CONFIRM2_ORDER,
                              "confirm blocks (470-477, 490-493) rendered and asserted, NOT written (--no_confirm)")):
        if not flag:
            drop |= set(block)
            print(msg)
    T = [t for t in T if (t[4], t[5]) not in drop]
    new, same, differ, ran = [], [], [], []
    for name, tag, text, *_ in T:
        p = tdir / f"{name}.sh"
        has_status = (sdir / f"{name}.json").exists()
        if p.exists():
            (same if p.read_text() == text else differ).append(name)
            if has_status:
                ran.append(name)
        elif has_status:
            print(f"refused: {name} has status/{name}.json (it ran) but no task file; a re-run needs a new task name; "
                  "nothing written", file=sys.stderr)
            sys.exit(2)
        else:
            new.append((p, text))
    for name in differ:
        print(f"DIFFERS from the current rendering (kept, not overwritten): tasks/{name}.sh")
    if not a.check:
        for p, text in new:
            p.write_text(text)
    print(f"{len(T)} tasks: {len(new)} {'to write' if a.check else 'written'}, {len(same)} unchanged, "
          f"{len(differ)} differ (kept), {len(ran)} already ran")


if __name__ == "__main__":
    main()
