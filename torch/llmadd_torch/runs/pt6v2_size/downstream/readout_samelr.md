# SAMELR downstream: all6@6e-4 - none@6e-4, paired over seeds 16-19 (F96 rule, Amendment A4)

## 190m  seeds {'pt6v2_size/pt6v2-all6-lr6e-4': [16, 17, 18, 19], 'pt6v2_size/pt6v2-none-lr6e-4': [16, 17, 18, 19]}
bench       n    all6    none    diff          95% t-CI wins | verdict | per-seed d
blimp       4   75.25   75.11   +0.14    [-0.61, +0.89]    2 | non-inferior | +0.42 -0.43 -0.04 +0.62
lambada     4   12.93   13.20   -0.27    [-0.76, +0.21]    0 | non-inferior | -0.02 -0.37 -0.04 -0.66
hellaswag   4   27.32   27.56   -0.24    [-0.72, +0.24]    1 | non-inferior | -0.19 -0.19 +0.07 -0.66
arc_easy    4   43.81   43.73   +0.08    [-0.80, +0.97]    2 | non-inferior | +0.00 -0.59 +0.76 +0.17
piqa        4   58.15   58.42   -0.27    [-0.95, +0.40]    1 | non-inferior | -0.54 +0.33 -0.60 -0.27
LAMBADA ppl ratio all6/none (descriptive): geo mean 1.028, per seed 0.992 1.104 1.014 1.006

## 505m  seeds {'pt6v2_size/pt6v2-all6-lr6e-4': [16, 17, 18, 19], 'pt6v2_size/pt6v2-none-lr6e-4': [16, 17, 18, 19]}
bench       n    all6    none    diff          95% t-CI wins | verdict | per-seed d
blimp       4   76.14   76.10   +0.04    [-0.56, +0.63]    2 | non-inferior | +0.36 +0.31 -0.44 -0.09
lambada     4   13.15   13.19   -0.04    [-1.13, +1.05]    2 | undetermined | +0.85 -0.29 +0.04 -0.78
hellaswag   4   28.14   28.30   -0.16    [-0.83, +0.50]    2 | non-inferior | -0.69 -0.31 +0.16 +0.19
arc_easy    4   45.05   44.98   +0.07    [-1.43, +1.58]    3 | undetermined | +0.08 +1.22 +0.08 -1.09
piqa        4   58.77   58.73   +0.04    [-0.33, +0.41]    1 | non-inferior | -0.11 -0.11 +0.00 +0.38
LAMBADA ppl ratio all6/none (descriptive): geo mean 1.052, per seed 1.050 1.131 0.951 1.084

## 1b  seeds {'pt6v2_size/pt6v2-all6-lr6e-4': [16, 17, 18, 19], 'pt6v2_size/pt6v2-none-lr6e-4': [16, 17, 18, 19]}
bench       n    all6    none    diff          95% t-CI wins | verdict | per-seed d
blimp       4   76.36   76.67   -0.31    [-1.88, +1.27]    1 | undetermined | -0.83 +1.11 -1.12 -0.38
lambada     4   12.67   12.91   -0.24    [-1.30, +0.82]    2 | undetermined | -0.27 +0.02 +0.43 -1.14
hellaswag   4   28.91   28.69   +0.22    [-0.12, +0.56]    4 | non-inferior | +0.54 +0.11 +0.11 +0.12
arc_easy    4   46.55   45.99   +0.56    [-2.25, +3.36]    3 | undetermined | -1.98 +2.02 +1.39 +0.80
piqa        4   58.94   59.39   -0.45    [-2.01, +1.11]    1 | undetermined | +0.98 -1.25 -0.82 -0.71
LAMBADA ppl ratio all6/none (descriptive): geo mean 1.077, per seed 1.029 1.204 1.057 1.026
