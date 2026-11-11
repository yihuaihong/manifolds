# atlas2 — manifold atlas v2 (36 domains × 3 pretraining trajectories + 5 other families)

Same method as v1 (runs/atlas, F67; tools/manifold_atlas.py docstring). Changes:
- 18 new domains (tools/manifold_domains.py, "atlas v2"), appended after the v1 ones. v1 items, targets and test
  sets are bit-identical to v1 (checked against the v1 module; only the elements / colours domains gained extra
  NON-primary targets). The z-scoring of states runs over all items of all domains, so v1 numbers can shift slightly.
- Layers: 9 evenly spaced layers 0..L for any depth (= 0, 4, …, 32 on OLMo-2-7B, as in v1); states saved at 1/4, 1/2,
  3/4 depth (= 8, 16, 24 on OLMo-2-7B).
- Models: OLMo-2-7B timeline (as v1), OLMo-2-1B timeline (step 0 = real init … 4T, main), Pythia-6.9B timeline
  (4B … 300B), finals: SmolLM2-1.7B, Qwen2.5-7B, Qwen3-8B-Base, OLMo-3-7B, Mistral-7B. Llama / Gemma left out (gated).

## New domains: ground truth, primary target, test
| domain | n | ground truth (source) | primary target | test (3-shot, chance) |
|---|---|---|---|---|
| countries | 70 | capital lat/lon (geonamescache) | spherical harmonics deg ≤ 2 | closer to A: B or C (≥ 25° apart), 1/2 |
| country_size | 70 | log10 population (geonamescache) | line | more people (≥ 3×), 1/2 |
| landmarks | 76 | Wikipedia primary coordinates (tools/atlas_fetch_data.py) | harmonics | closer (≥ 25°), 1/2 |
| us_cities | 45 | largest city per state, lat/lon (Gurnee & Tegmark us_place) | plane (lat, lon) | closer (≥ 5°), 1/2 |
| people | 106 | birth year (Gurnee & Tegmark historical_figure) | cubic curve of year | born earlier (≥ 50 y), 1/2 |
| geo_periods | 12 | order of Phanerozoic periods | curve | k periods after X, 1/12 |
| animals | 65 | class / order (traditional) | tree code (class 1.0, order 0.7) | more closely related to A (tree distance), 1/2 |
| animal_size | 65 | log10 typical adult mass (order-of-magnitude values written in the code) | line | heavier (≥ 10×), 1/2 |
| languages | 62 | family / branch | tree code | more closely related (tree), 1/2 |
| kinship | 21 | generation, sex, lineal | generation line | older generation (≥ 1), 1/2 |
| scalar_adj | 31 | rank within 5 scales | within-scale lines | per-scale "Which is hotter/bigger/…" (≥ 2 ranks), 1/2 |
| ranks / life_stages | 8 / 7 | order | curve | higher / later (≥ 2), 1/2 |
| zodiac / chinese_zodiac | 12 / 12 | order (Z_12) | circle k = 1 | k signs / years after X, 1/12 |
| compass | 8 | angle (Z_8) | circle | closer direction (≥ 45°), 1/2 |
| units | 24 | log10 SI value within time / length / mass | within-dimension lines | lasts longer / longer / heavier (≥ 3×), 1/2 |
| prefixes | 11 | power of ten | curve | larger amount (≥ 2 orders), 1/2 |
Tree targets drop singleton categories (a held-out singleton is unpredictable under LOO); the tree TESTS use all
categories (so two singletons, e.g. Japanese and Korean, are not "related").

## Known confounds to read results against
- Knowledge domains (countries, landmarks, people, sizes): the target encodes facts, not only a symmetry; geometry =
  "the model knows where/when", and the random-token control only rules out tokenisation.
- countries vs country_size share items and the neutral prompt (same states): only the cued prompt differs.
- animals vs animal_size: same (different cue); tree and mass are correlated (insects small, mammals big).
- people: multi-token names, last-token state; fame is uneven across centuries (dense 1400–1960).
- zodiac shares Z_12 with months and its signs map onto month positions; overlap with months is a hypothesis.
- kinship words are gendered pairs: the "plane" target has a sex axis that can be read from the word form alone.

## Expected
- Random-init (OLMo-2-7B rand, OLMo-2-1B step 0): best-layer R² ≈ null for all domains except tokenisation-driven ones.
- Final models: geography (countries, landmarks, us_cities) and time (people) clearly above null (Gurnee & Tegmark).
- Cross-family: if manifolds are universal, the same domains pass the null in all five families.

## Audit 2026-10-13 (fresh adversarial subagent; scripts in scratchpad/audit_atlas2) — fixed before the jobs ran
Verified myself: test-set duplication (life_stages 24 distinct of 60 rows; 9 rows identical to a shot), curve ceiling
for a perfect linear code (0.08 / 0.10 / 0.16 / 0.31 at n = 7 / 8 / 12 / 106), languages tree rank 16 of 25 columns,
people step function (ancient vs modern) LOO R² 0.84.
- B1 test chance ≠ 1/2 (summed log-prob favours short candidates; item priors; answer position 0.40–0.63; nothing
  per-row saved). Fixed: distinct comparisons asked in both orders (answer position exactly 0.50), per-row scores
  saved (log p, content-free log p, n tokens), acc / acc_norm / acc_cal (Zhao et al. 2021 content-free calibration)
  / acc_prior (candidate prior alone). Candidates scored as " c\n" (also fixes " north" being a token prefix of
  " northwest" for Mistral). Headline test metric = acc_cal; random-init models give the per-domain floor.
  Smoke (random-init Pythia-160m): acc 0.61–0.77 on ranks / languages / country_size, prior-only 0.56–0.77,
  acc_cal 0.33–0.50 → raw accuracy is NOT usable for small domains; calibrated is closer to chance but tests with
  < 30 comparisons (planets 15 rows, life_stages 24, geo_periods 27, ranks 36, days 39) stay noisy.
- I1 few-shot leak / duplicates: questions now unique, shots disjoint from the test (offset domains: all valid
  (item, k) drawn without replacement, so small domains have fewer rows). This changes the v1 test sets too: F67's
  capability column must be re-read from this run.
- I2 primary targets: ordered domains → line, places → unit sphere (both were saved before, so v1-style numbers can
  still be computed from the JSON); people → rank of birth year (I5).
- I3 tree: duplicate categories removed (languages 16 columns, rank 16); pooled R² (r2p, weights columns by
  variance) computed for every target and used as the headline for tree domains; bases are rank-revealing (SVD).
- I4 BOS: prompts now use the tokenizer's default special tokens (adds BOS for Mistral only; recorded as bos_added).
- m1 capitals matched accent-insensitively (Brazil, Colombia, Cameroon, Kazakhstan back); countries margin 35°.
- m2 removal null (bases fitted to a's permuted target) saved as removal_null.
- m7 p-values (k+1)/(N+1); args, model class, bos flag saved in the JSON.
- m8 plots: random-init tags detected in v2 names; axis labels per trajectory; calibrated test score.
- Not fixed (documented): us_cities covers 45 states (no AK, HI, ME, VT, DE; ambiguous names like Portland);
  layer spacing uneven for 12/28/36-layer models (compare relative depth only); MLK renamed "Martin Luther King".
- 2026-10-13 (after the living-lab audit): helix-type targets had a sin(pi x) noise column (period-2 term on integers, standardised); _std now drops constant columns (commit 0201fe6). Numbers helix R2 at L16 (v1 states, cued): 9B/21B/main 0.377/0.738/0.841 -> 0.428/0.824/0.948. Snapshot refreshed before any job ran.
