# toy_struct s4, frac 0.6, question split, budget 100000 steps, lam 1.0, cell windows ['always']

chance (max of 1/n_out and the majority-of-train baseline): 0.042; floor line = chance + 0.05 = 0.092

| arm | seeds | t_mem median | t_grok censored median | grokked | AUC median (co-primary) | t_0.5 median | final test mean | final test_fresh mean | CKA own target @2000 / final |
|---|---|---|---|---|---|---|---|---|---|
| R@always | 5 | 100 | 1000 | 5/5 | 0.981 | 500 | 1.000 | 1.000 | 1.00 / 1.00 |
| R_sign@always | 5 | 100 | 1100 | 5/5 | 0.981 | 300 | 0.997 | 0.996 | 1.00 / 1.00 |
| specrand@always | 5 | 100 | > 100000 | 0/5 | 0.024 | > 100000 | 0.033 | 0.022 | 1.00 / 0.99 |

qualifying fraction for rule (a) (none's censored median t_grok >= 4000): False

comparison T(a, b), paired by seed; PASS on either co-primary, each at p < 0.025 (t_grok: a earlier; AUC: a higher by >= 0.05 in the medians); wins need >= ceil(0.8 n)

| test | n | t_grok median a / b | wins | p | AUC median a / b | wins | p | verdict |
|---|---|---|---|---|---|---|---|---|
| R@always < specrand@always | 5 | 1000 / > budget | 5/5 | 0.004 | 0.981 / 0.024 | 5/5 | 0.004 | PASS (t_grok + AUC) |
| R_sign@always < specrand@always | 5 | 1100 / > budget | 5/5 | 0.004 | 0.981 / 0.024 | 5/5 | 0.004 | PASS (t_grok + AUC) |

rule (a): arms missing

rule (a) [@always arms, secondary]: untestable at this fraction (none's censored median t_grok < 4000: none groks early); not a negative

rule (b): coset_share of wrong test answers pooled over evals in [1000, 2000] (uniform-error chance 0.130, approximate for dim: answers are not uniform); per seed (share, n_wrong); a seed counts only with n_wrong >= 20

- R@always: s0 0.00 (17), s1 0.00 (127), s2 0.00 (5), s3 0.00 (140), s4 0.00 (100)
- R_sign@always: s0 0.02 (500), s1 0.00 (71), s2 0.00 (56), s3 0.00 (37), s4 0.00 (89)
- specrand@always: s0 0.21 (2469), s1 0.32 (2411), s2 0.28 (2480), s3 0.26 (2463), s4 0.26 (2490)
rule (c) (read at the lowest fraction of the design only): arms missing
rule (c) [@always] (read at the lowest fraction of the design only): arms missing
