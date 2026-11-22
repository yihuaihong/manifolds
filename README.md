<div align="center">

# Manifold Pretraining Engineering

**Right Geometry, Right Time, Right Place**

Yihuai Hong · Hanxi Ma · Chen Zhao

Paper (arXiv, coming soon) · [Project page](https://yihuaihong.github.io/manifolds/)

<img src="https://yihuaihong.github.io/manifolds/media/fig1_manifold.png" width="860" alt="A 53M model pretrained from scratch and OLMo-2-7B fine-tuned, with and without a helix prior on numbers: their number manifolds and their accuracy on unseen addition problems">

</div>

A **manifold prior** is an auxiliary loss that pulls a model's representations of a concept towards a target
manifold, each item towards its own point on it; for numbers, the target is a helix. It turns what interpretability
knows about concept manifolds into a training technique.

<table>
<tr>
<td width="50%" valign="top">
<h3>18% → 70%</h3>
<b>Scarce data, learned.</b> Pretraining a 53M model from scratch with 2% arithmetic tokens, the prior raises
held-out addition accuracy, averaged over training, from 18% to 70%, each model at its own best learning rate
(8 of 8 seeds).
</td>
<td width="50%" valign="top">
<h3>6 of 6</h3>
<b>Priors that help each other.</b> Six priors trained together all gain. At the end of training they lead the
dimensions prior alone by 25.0 points (7 of 8 seeds, one tie), and kept on until the end, all six end above the best model without priors
(8 of 8 seeds).
</td>
</tr>
<tr>
<td valign="top">
<h3>99% vs. ≤ 11%</h3>
<b>What 4× the data does not teach.</b> Three-digit numbers seen only in web text, never in arithmetic: with the
prior, the model learns to add them (99% right on 700–999); without it, every setting ends at about 11% or below,
even with four times the tokens.
</td>
<td valign="top">
<h3>+36.1 points</h3>
<b>Holds up to 1.07B.</b> At 1.07B parameters, the largest model we pretrained (53% → 89%, 8 of 8 seeds); the
advantage holds at every size from 53M.
</td>
</tr>
<tr>
<td valign="top">
<h3>≈ 5× sooner</h3>
<b>Fine-tuning a 7B model.</b> An early OLMo-2-7B checkpoint reaches 90% on two-digit addition about five times
sooner, and ends at 100% against about 95% without the prior.
</td>
<td valign="top">
<h3>+43.5 points</h3>
<b>Even in plain web text.</b> At 190M, from a prior only on numbers in ordinary web text, never on the problems
(8 of 8 seeds).
</td>
</tr>
</table>

It helps with the right geometry, at the right time, in the right place.

## The prior

```math
\mathcal{L} \;=\; \mathcal{L}_{\mathrm{LM}} \;+\; \lambda_t\,\bigl(1-\mathrm{CKA}(H,F)\bigr)
```

- **H**: hidden states of the concept's items at chosen layers. **F**: their target coordinates; for a number x,
  cos and sin of 2πx/T for T = 2, 5, 10, 100 (and 1000 for operands up to 999), plus a linear term.
- **When**: on from about 5% of training; until 40% in most runs, until the end when several priors are combined.
- **Where**: on the concept's own tokens, either the operands in the training text or numbers in ordinary web text.
- **Control**: the same coordinates assigned to shuffled items.

Code: `cka` and `centred_gram` in `tools/pretrain_helix.py`; targets in `features` of `tools/pretrain_range.py`.

## Quick start

Python 3.11 with the versions the runs used (torch 2.10.0 for CUDA 12.8, transformers 4.57.6):

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r results/331-env311/requirements311.txt
```

**Pretrain** a 53M model with the helix prior (300M tokens, 2% arithmetic; under an hour on one A100):

```bash
# FineWeb-Edu sample-10BT, OLMo-2 tokenizer, at the dataset commit the runs used
python tools/pretrain_data.py --tokenizer allenai/OLMo-2-1124-7B --out data/pretrain_data \
  --tokens 400000000 --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9

# helix prior on the operands, from 5% to 40% of training
python tools/pretrain_range.py --data data/pretrain_data --tokenizer allenai/OLMo-2-1124-7B \
  --tokenization whole --out out/helix --arith_lines 1 --lr 2e-3 --seed 0 \
  --target helix3 --schedule window --window 0.05,0.4 --site ctx

# no prior, at its own best learning rate
python tools/pretrain_range.py --data data/pretrain_data --tokenizer allenai/OLMo-2-1124-7B \
  --tokenization whole --out out/none --arith_lines 1 --lr 8e-3 --seed 0 --target none
```

Each run writes `run_whole_<target>_seed<s>.json`: held-out accuracy per operand-length bucket (`held_b11` …
`held_b33`) at every evaluation.

| To change | Flag |
|---|---|
| prior on numbers in web text | `--site text` |
| 7% arithmetic | `--arith_lines 4` |
| model size | `--d_model`, `--n_layers`, `--n_heads`, `--d_ff` |
| training tokens | `--tokens` |

**Fine-tune** OLMo-2-7B with the prior (two-digit addition, 1,500 training pairs, LoRA, prior at layers 4, 8, 12, 16):

```bash
python tools/fetch_olmo.py --revision stage1-step2000-tokens9B --dest models/OLMo-2-7B-9B
python tools/manifold_train.py --model models/OLMo-2-7B-9B --out out/ft --mode lora --lam 20 \
  --geoms none,helix,helix_shuf --seeds 0
```

Each run writes `run_<geometry>_seed<s>.json`; other tasks, targets and splits are in the script header.

**Check a paper number without a GPU**: `python tools/pt2main_summary.py` reads `results/` and prints the
curve averages of Figure 1d over 8 seeds: 18.4 without the prior, 69.9 with it.

## Reproducing the paper

Scripts are in `tools/` unless a path is given; `make_jobs_*.py` are in `torch/llmadd_torch/`, and `runs/` is
`torch/llmadd_torch/runs/`.

| Paper | Code | Runs |
|---|---|---|
| **Fig. 1a–d** · pretraining, 53M | `pretrain_range.py`; tasks from `make_pt2main_tasks.py` (e.g. `tasks/350-pt2b-lr2e-3-win-ctx-s0.sh`); read-out `pt2main_summary.py`; panels b, c: `--save_at` and `fig1_export.py` (branch `fig1-export`) | `results/*-pt2b-*/`, `runs/fig1_pt2b/` |
| **Fig. 1e–h** · fine-tuning OLMo-2-7B | `manifold_train.py --train_pairs 7000 --epochs 10 --fixed_test --curve_every 50`; `make_jobs_broad.py`, `make_jobs_s1.py`, `make_jobs_viz.py`; 3-D axes as in `viz_helix3d.py` | `results/b3-curve-9B-n7000-*-torch/`, `results/s1-curve-9B-n7000-*-torch/`, `runs/viz-9B-n7000-*/` |
| **Fig. 2a** · model size, 2% | as Fig. 1a–d (e.g. `tasks/358-pt2s190-lr6e-4-win-ctx-s0.sh`); 505M: `make_jobs_pt2s505.py` | `results/*-pt2s190-*/`, `runs/pt2s505/` |
| **Fig. 2b** · model size, 7% | `make_jobs_ptlr.py`, `make_jobs_ptlr1b.py`; read-out `ptscale_summary.py` (1.07B: branch `ladder-1b`) | `runs/ptlr/`, `runs/ptlr_1b/` |
| **Fig. 2c** · first set of six priors | 53M: `make_rb6_tasks.py`, `make_rb6m_tasks.py` (e.g. `tasks/419-rb6pb-53m-2pct-k2-all6-lam1-s0.sh`), read-out `rainbow6_summary.py`; 190M, 505M: `make_jobs_rb6scale.py`, read-out `rb6scale_summary.py` (branch `scale-ladder`) | `results/*-rb6pb-*/`, `results/*-rb6mb-*/`, `runs/rb6_190m/`, `runs/rb6_505m/` |
| **Fig. 3** · gain vs. size at fixed data | `make_jobs_pttpp.py` (read-out `pttpp_summary.py`, branch `ladder-tpp`), `make_jobs_pt2same.py`, `make_jobs_pt300x.py` | `runs/pttpp/`, `runs/pt2same/`, `runs/pt300x/` |
| **Fig. 4** · right time and place | `make_jobs_pt999.py`, `make_jobs_pt999b.py`; read-out `pt999_summary.py` | `runs/pt999/`, `runs/pt999b/` |
| **Fig. 5** · the benefit window, OLMo-2 1B–32B | `manifold_train.py`, `concept_train.py`, `fetch_olmo.py` (e.g. `tasks/081-curve-lora-lam20-s1-2000.sh`) | `results/` of the tasks in FINDINGS F14, F23–F27, F32, F34 |
| **Fig. 6** · the window for Pythia | `pythia_train.py` (e.g. `tasks/242-pythia-160m-step1000.sh`) | `results/*-pythia-*/` |
| **Table 1** · concepts helped | rows from Fig. 1, Fig. 2c and Table 2; months, weekdays: `ft36.py`, `make_jobs_ft36.py`, read-out `ft36_summary.py`; vector sums: `toy_struct.py`, `make_jobs_newmani.py` | `results/ft36-torch/`, `runs/newmani/` |
| **Table 2** · six priors together | [below](#six-priors-together-table-2) | `runs/pt6v2/` |
| **Table 3** · parts of the helix; three operations | `manifold_train.py` (e.g. `tasks/288-p4-sep-7b-tokens42B.sh`); b: `--task mixed`, `make_jobs_mixops.py` | `results/28[89]-*/`, `results/290-*/`, `results/mo-*-torch/` |
| **Table 4** · costs | zero-shot benchmarks: `downstream_eval.py`, `downstream_readout.py` (branch `downstream`); the other rows reuse the runs above | `runs/downstream/` |

`results/<task>/` holds run outputs. `runs/<batch>/` holds the records of the runs on our second cluster
(pre-registration `AUDIT.md`, read-out scripts) and their per-seed outputs under `out*/` (JSON and CSV, no weights).

### Six priors together (Table 2)

On branch `ptdom3`: six priors (λ 1 each) at learning rate 1e-3. In the flags, `dim` is dimensions and `box5` is the
grid.

```bash
git checkout ptdom3
python tools/pretrain_data.py --tokenizer allenai/OLMo-2-1124-7B --out pretrain_data --tokens 400000000 \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
cd tools
python pretrain_range.py --data ../pretrain_data --tokenizer allenai/OLMo-2-1124-7B --tokenization whole \
  --out ../out/all6-lr1e-3-s16 --target helix3 --schedule window --window 0.05,0.4 --site ctx --lam_rule each \
  --prior_domains numbers,dim,cities,countries,torus,box5 --lam 1 \
  --arith_lines 1 --domains numbers,dim,cities,countries,torus,box5 --extra_lines 2 \
  --lr 1e-3 --seed 16 --tag _all6-lr1e-3 --ckpt_every 500
```

| Variant | Change |
|---|---|
| priors on until the end | `--window 0.05,1.0` |
| no prior | `--target none`, without `--schedule --window --site --lam_rule --prior_domains --lam`; learning rates 2e-3 to 6.4e-2, doubling |
| one prior alone (Six − one column) | `--lr 2e-3 --prior_domains <one>`, with `--target helix3` for numbers and `--target none` otherwise |

Seeds 16–23; a run takes 0.3–0.75 GPU-hours on one A100 or B200. Cities and countries come from `geonamescache`
(pip); the other domains in `tools/manifold_domains.py` read CSV files from `$MANIFOLD_DATA` (default `data/atlas`)
and are skipped when those files are absent. Every Table 2 run used the commit tagged `pt6v2-run`; `ptdom3` adds
only this data-folder change. Pre-registration and amendments: `runs/pt6v2/AUDIT.md`.

## Repository

```
tools/                 trainers, data and read-outs
  pretrain_range.py      pretraining from scratch, 53M–1.07B, with or without priors
  pretrain_data.py       tokenised FineWeb-Edu for it
  manifold_train.py      LoRA fine-tuning of OLMo-2 on addition (also subtraction and mixed operations)
  concept_train.py       the same for weekdays, months, letters, number words and Roman numerals
  pythia_train.py        the same for Pythia
  ft36.py                the 36-domain screen
  fetch_olmo.py          downloads an intermediate OLMo-2 checkpoint
  *_summary.py           read-outs, with shared definitions in readout_rules.py
tasks/                 one script per run, holding its exact trainer command
results/<task>/        run outputs
torch/llmadd_torch/    job generators, records and per-seed outputs of the runs on our second cluster
report/FINDINGS.md     findings log (in Chinese; entries F1, F2, …, each naming its code and runs)
```

Some trainer versions live on branches: `ptdom3` (Table 2), `fig1-export` (Figure 1 states), `scale-ladder`,
`ladder-1b`, `ladder-tpp` (model size) and `downstream` (zero-shot benchmarks). `torch/llmadd_torch/` is a
snapshot: its generators hard-code the paths of the cluster they ran on, so they record what was run but do not run
as they are; to rerun a cell, use its trainer command. Its read-outs also hard-code the cluster root, and they read
the outputs in `runs/<batch>/out*/` once that root points at `torch/llmadd_torch/`.

## Citation

```bibtex
@article{hong2026manifold,
  title   = {Manifold Pretraining Engineering: Right Geometry, Right Time, Right Place},
  author  = {Hong, Yihuai and Ma, Hanxi and Zhao, Chen},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT, see [`LICENSE`](LICENSE).
