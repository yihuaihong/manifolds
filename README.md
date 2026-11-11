# llmadd-burst

Runs LLM_addition extension experiments on NYU HPC Burst, driven entirely through git. Only our own code
lives here; the collaborator's repository (firefly's LLM_addition_code) is not copied in, and no file in this
repository is copied from it.

## How it works

A sentinel job on Burst (`llmadd_sentinel`: 1 CPU, no GPU, renews itself every 48 h) syncs this repo every
minute and runs `burst/tick.sh`:

- every `tasks/NNN-name.sh` without a `status/NNN-name.json` whose `after:` tasks completed is submitted
  once, from a frozen snapshot of the repo, to the CPU or GPU partition named in its header;
- when the job ends, `status/<task>.json` and the small files it wrote to `$OUT` (json/csv/png/md/txt,
  < 5 MB each) are pushed to `results/<task>/`, with the last 200 log lines;
- `heartbeat.json` is pushed at least every 10 minutes.

Task header (comment lines at the top of the script):

```
# kind: cpu | gpu        (gpu = 1 x A100 40GB on GPU_PARTITION)
# time: HH:MM:SS         (Slurm limit)
# cpus: N
# gpus: 2                (optional; 2 x A100 on GPU2_PARTITION)
# after: 001-env 002-x   (space separated; waits until those completed)
# after_ended: 006-x     (waits until those ended, in any state)
```

Tasks see `$OUT` (write results here), `$PY` (the env's python), `$MODELS`, `$HF_HOME`. A task runs once;
to re-run, add a new task file. `burst/config.env` holds ENABLED, partitions, the account, the GPU budget. GPU tasks go to the accounts in `GPU_ACCOUNTS` (in order, each with its own hour cap; committed hours = max(Slurm sacct incl. queued/running jobs at their time limit, our status/ records)); an account the submit filter refuses falls through to the next. A retry after an outside kill keeps the run's code snapshot (`commit`) and `out/`, so long tasks should resume from their checkpoints. Set `# time:` to about 1.5x the measured run time: the guard reserves the full limit.

## Scheduling without a resident sentinel

Burst's root cancels low-utilisation jobs about 38 minutes after they start (checks at :11/:31/:51), and
n2c48m24 is exclusive, so a 1-CPU sentinel holds a whole 48-core node. With `SENTINEL=off` in
`burst/config.env`, every running task ticks every 90 s and once more when it ends (publishing its own
results and submitting what it unblocked). If nothing is running and new tasks are pushed, one kick on any
Burst terminal starts them: `bash /scratch/$USER/llmadd/burst/tick_locked.sh`.

## One-time setup on Burst (the only manual step)

```bash
test -f ~/.ssh/id_ed25519_llmadd || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519_llmadd -C burst-llmadd; cat ~/.ssh/id_ed25519_llmadd.pub
# the public key is added to this repo as a deploy key WITH write access, then:
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_llmadd -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  git clone -q git@github.com:yihuaihong/llmadd-burst.git /scratch/$USER/llmadd && bash /scratch/$USER/llmadd/burst/boot.sh
```

Cancel one task from anywhere: add its name to `burst/cancel.list` and push.

Stop everything: `scancel -n llmadd_sentinel` (running tasks: `scancel -n llmadd_<task>`), or set
`ENABLED=0` in `burst/config.env` to stop new submissions.

## Torch-side code and side branches

Results are reported in `report/FINDINGS.md` (F numbers). Not every run went through Burst:

- `torch/llmadd_torch/`: a snapshot of the code and records of the batches run on the NYU Torch cluster (job
  generators, per-batch read-out / selection / submission scripts, pre-registrations `AUDIT.md`, tests and audit
  reports). The jobs ran from `/scratch` on Torch, not from this copy. `torch/README.md` maps each batch to its
  FINDINGS entries.
- Branches kept off `main` (trainer or read-out revisions of individual batches):

| branch | contents | FINDINGS |
|---|---|---|
| `ptdom` | `tools/pretrain_range.py` Rev E–G: anchor-comparison and k-step domains, heuristic-hard splits, k-way read-out | F100, F101 |
| `downstream` | `tools/downstream_eval.py`, `downstream_readout.py`, `make_downstream_jobs.py`, pre-registration | F96 |
| `fig1-export` | `pretrain_range.py --save_at`, `tools/fig1_export.py`, `fig1_logcka.py` | F92 |
| `ft999` | `tools/manifold_train.py --add3`, `tools/ft999_summary.py` | F93 |
| `ladder-1b` | `ptscale_summary.py` read-out for the 1B point (ptlr_1b) | F87 |
| `ladder-tpp` | pttpp read-out and records (`notes/pttpp`; builds on `scale-ladder`) | F97 |
| `scale-ladder` | rainbow6 190M / 505M read-outs and records (`notes/rb6_*`) | F88 |
| `arith-dose` | ptshare (arithmetic-share dose) design and read-out | — |
