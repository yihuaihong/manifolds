# torch/ — code run on the NYU Torch cluster

- `dl.sbatch`, `full_ft.sbatch`, `full_ft_fsdp.sbatch`: early Torch jobs (model download, full fine-tuning).
- `llmadd_torch/`: a **snapshot** (2026-11-10) of our own code and written records from the Torch working
  directory `/scratch/yh6210/llmadd_torch`. The jobs ran from that directory, not from this copy. Paths inside the
  scripts (`/scratch/yh6210/llmadd_torch/...`) refer to that directory and the snapshot is not kept in sync.

## What the snapshot contains

| path | what |
|---|---|
| `llmadd_torch/make_jobs_<batch>.py` | job generators: write `runs/<batch>/job_*.sbatch` (account, GPU, trainer flags, seeds) |
| `llmadd_torch/download_xmodels.py`, `make_625_copies.py` | model download for atlas v2; second-account copies of pending jobs |
| `llmadd_torch/runs/<batch>/AUDIT.md` | pre-registration, intent spec, amendments and audit record of the batch, written before its results |
| `llmadd_torch/runs/<batch>/*.py`, `*.sh` | batch-specific read-out, lr selection, submission and check scripts |
| `llmadd_torch/runs/<batch>/tests/` | tests of those scripts, and the audit reports (`audit_*.md`) |

Left out: run outputs (`out/`, logs, `*.json`, checkpoints, `*.pt` / `*.npz`), data, models, caches, the generated
`job_*.sbatch` files, the trainer snapshots `runs/<batch>/code*` (each is a `git archive` of this repository at the
commit named in the batch's AUDIT.md), `superseded_*` / `wip_*` copies, and any file over 1 MB.

The trainers themselves live in `tools/` of this repository (main) or on these branches:
`ptdom` (pretrain_range Rev E–G: ptdom, ptdom2), `downstream` (downstream benchmarks), `fig1-export` (pt2b state export),
`ft999` (manifold_train --add3), `arith-dose` (ptshare read-out), `ladder-1b` (ptlr_1b read-out), `ladder-tpp` (pttpp),
`scale-ladder` (rb6_190m / rb6_505m / rb6_190m_dim read-outs).

## Batches and their FINDINGS entries

F numbers are the entries of `report/FINDINGS.md` that report the batch ("—": no entry yet).

| `runs/` | experiment | FINDINGS |
|---|---|---|
| `atlas2` | manifold atlas v2: 36 domains x three pretraining trajectories + other model families | F74 (v1: F67) |
| `ft36` | fine-tuning prior screen over the 36 atlas domains (OLMo-2-7B, LoRA) | F81 |
| `ft999`, `ft999_dev` | three-digit addition LoRA fine-tuning with the helix3 prior (`ft999_dev`: smoke / regression checks) | F93 |
| `fig1_pt2b` | Figure 1 number-state export of F86's 2% runs | F92 |
| `downstream` | zero-shot BLiMP / LAMBADA / HellaSwag / ARC-E / PIQA of the pretrained models | F96 |
| `ptstart` | in-context eval and window start-time sweep | F65, F68 |
| `pt999` | unified 0..999 setup, main arms | F82 |
| `pt999b` | where and when the prior acts (0..999) | F83 |
| `ptrangeg` | gated / windowed priors (0..999) | F79 |
| `ptsize999` | size ladder in the 0..999 setup (cells reused in F87) | F87 |
| `ptteach` | teacher-derived target geometry | F77 |
| `pttext` | natural-text prior | F78 |
| `ptgw` | 5% window vs 5% self-timed gate | F85 |
| `ptlr` | per-arm best lr, 7% size ladder (53M / 190M / 505M) | F87 |
| `ptlr_1b` | 1B point of the 7% ladder (B200) | F87 |
| `pt2s505` | F86's 2% single-prior setting at 505M | F86 |
| `pt2same` | 2% size comparison on one cluster | F98 |
| `pt2sz` | 93M and 294M added to the 2% size comparison | — |
| `pttpp` | gain at fixed tokens per parameter | F97 |
| `pt300x` | 93M and 294M on the fixed-token 7% ladder | F99 |
| `ptsz` | four more sizes on the fixed-token 7% ladder | — (follow-up of F99) |
| `ptshare` | arithmetic-share dose at 53M | — |
| `ptheld` | wall 1: held-out numbers | F90 |
| `ptheld2` | wall 1 follow-up: held-out large numbers | F91 |
| `rb6` | six manifold priors pretrained together (rainbow6), 53M | F88 |
| `rb6_190m`, `rb6_190m_ext`, `rb6_190m_dim` | rainbow6 at 190M and its follow-ups | F88 addenda |
| `rb6_505m` | rainbow6 at 505M | F88 addendum |
| `pt6comb` | six priors trained together at 53M: does each help, does none interfere? (built; GPU smoke only) | — |
| `ptdom` | single-prior screen over 24 candidate domains (B200) | F100 |
| `ptdom2` | fresh-seed replication with shuffled-target controls | F101 |
| `toystruct` | toy structure priors: S4 and the dimension lattice | F84 |
| `toynone` | toy: is "not learned without the prior" an optimiser artefact? | F84 |
| `toyf84fu` | toy follow-ups of F84 | F84, F89 |
| `toyshape` | wall 2: shape-only prior | F89 |
| `newmani` | new-manifold toy screen, stages 1 and 2 | F94, F95 |
