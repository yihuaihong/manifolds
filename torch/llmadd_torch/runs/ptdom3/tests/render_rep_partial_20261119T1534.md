## [2026-11-19] F104（PROVISIONAL 草稿：输入不全） toy3 复现：torus 的 prior 增益 +12.6 pp（area，replicated at 3 seeds，supports）、hcube 的 prior 增益 +20.9 pp（area，replicated at 3 seeds，supports）

> **PROVISIONAL**：`complete` = False；缺 2 项（见限定 1）。下面的数是部分读数，不是终读。

**设置**
- ptdom3 toy3（53M、75 次 eval、prior window 为 step 916–7323），seed 0–2。复现的 prior 格子：only-torus@1e-3、only-hcube@2e-3、only-box5@1e-3；none 网格：screen 的 2e-3…1.6e-2（s0）、1.6e-2（s0–2）和 3.2e-2（s0–2）。
- 主指标（PRIMARY）：torus area (full set)；hcube twin-free area；box5 area (full set)。hcube / d6 用 twin-free 子集，即 read-out 题目中三元组不在任何训练 fact 里的那部分。
- 增益 g = only-d@P − none@N，按 seed 配对。P 和 N 都按 pick_pt6v2 的规则选：在证据 seed 上，一个 lr 的 seed 覆盖数超过同臂最好覆盖数的一半才算 present；然后取均值最高的 lr，打平取较大的 lr。Cohen d = mean / sd（ddof 1）。boot90：paired bootstrap over seeds: resample the k per-seed g with replacement, percentile 5 / 95 of the resampled means（B 20000）。
- 判级：supports iff k >= 2, every per-seed g has the same sign and |mean g| > max g - min g。判定词只用 E1–E5（pt6v2 pick rule）和标签 preliminary (1 seed) / replicated at k seeds。
- 输入：readout_ptdom3r.py e328078abe3c、readout_ptdom3.py 9abd89ad75a7、pick_pt6v2.py 7a2e73faa8c8；读于 2026-11-19T15:35Z；共 31 个 run。

**结果**（单位：百分点。area = read-out 准确率曲线在全部 75 次 eval 上的均值 ×100，不只是 prior window；final3 = 最后 3 次 eval 的均值；窗口拆分：window 内 = prior window（step 916–7323）内各次 eval 的均值，window 外 = 其余各次 eval 的均值（含 step < 916 的早期 eval），与 readout_ptdom3 的 in / out 同一定义）

| 领域 | 指标 | P / N | 每 seed g（area） | 均值 g [boot90] | wins | d | final3 g（均值） | 标签 | E1–E5 |
|---|---|---|---|---|---|---|---|---|---|
| torus | area (full set) | 1e-3 / 1.6e-2 | s0 +8.5, s1 +14.4, s2 +14.9 | +12.6 [+10.5, +14.8] | 3/3 | 3.50 | s0 -6.8, s1 -16.2, s2 -13.5（-12.2） | replicated at 3 seeds，supports（+） | E1✓ E2✓ E3✓ E4✓ E5✓ |
| hcube | twin-free area | 2e-3 / 1.6e-2 | s0 +14.8, s1 +21.2, s2 +26.8 | +20.9 [+16.9, +24.9] | 3/3 | 3.48 | s0 +6.1, s1 -1.5, s2 +9.3（+4.6） | replicated at 3 seeds，supports（+） | E1✓ E2✓ E3✓ E4✓ E5✓ |
| box5 | area (full set) | 1e-3 / 1.6e-2 | s0 +33.4 | +33.4 | 1/1 | — | s0 +15.3（+15.3） | preliminary (1 seed) | E1✓ E2✗ E3✓ E4✓ E5✗ |
| cyc12（只有 s0） | area (full set) | 2e-3 / 3.2e-2 | s0 -3.0 | -3.0 | 0/1 | — | s0 -18.8（-18.8） | preliminary (1 seed)；FLOOR | E1✗ E2✗ E3✓ E4✗ E5✗ |
| d6（只有 s0） | twin-free area | 1e-3 / 3.2e-2 | s0 -4.1 | -4.1 | 0/1 | — | s0 -7.5（-7.5） | preliminary (1 seed)；FLOOR | E1✗ E2✗ E3✓ E4✗ E5✗ |
| caesar（只有 s0） | area (full set) | 1e-3 / 3.2e-2 | s0 +0.6 | +0.6 | 1/1 | — | s0 -6.9（-6.9） | preliminary (1 seed)；FLOOR | E1✗ E2✗ E3✓ E4✓ E5✗ |

窗口拆分（同一组 P / N 格子、同一主指标；window 内 = step 916–7323）：

| 领域 | window 内 g（均值，wins） | window 外 g（均值，wins） | final3 g（均值，wins） | prior 臂水平 内 / 外 / final3 | none 臂水平 内 / 外 / final3 |
|---|---|---|---|---|---|
| torus | s0 +29.2, s1 +49.7, s2 +52.9（+44.0，3/3） | s0 -2.5, s1 -4.3, s2 -5.3（-4.0，0/3） | s0 -6.8, s1 -16.2, s2 -13.5（-12.2，0/3） | 83.2 / 80.6 / 82.6 | 39.2 / 84.6 / 94.8 |
| hcube | s0 +35.9, s1 +53.3, s2 +53.7（+47.6，3/3） | s0 +3.6, s1 +4.1, s2 +12.6（+6.8，3/3） | s0 +6.1, s1 -1.5, s2 +9.3（+4.6，2/3） | 92.9 / 88.6 / 96.8 | 45.2 / 81.8 / 92.2 |
| box5 | s0 +56.3（+56.3，1/1） | s0 +21.2（+21.2，1/1） | s0 +15.3（+15.3，1/1） | 67.7 / 45.7 / 38.7 | 11.4 / 24.6 / 23.3 |

- **torus**（area (full set)，chance 2.8）：only-torus@1e-3 各 seed area s0 82.9, s1 80.8, s2 80.8；none@1.6e-2 各 seed area s0 74.4, s1 66.4, s2 65.8。配对均值 prior 81.5 / none 68.9。判级依据：same sign, |mean| 12.60 > seed range 6.46。
  - only-torus@1e-3 各 seed 水平（window 内 / window 外 / final3）：s0 83.9 / 82.4 / 87.0, s1 82.4 / 80.0 / 80.8, s2 83.3 / 79.4 / 80.0。
  - none@1.6e-2 各 seed 水平（window 内 / window 外 / final3）：s0 54.6 / 84.9 / 93.8, s1 32.7 / 84.3 / 97.0, s2 30.4 / 84.7 / 93.5。
  - 同 lr 增益（描述性，只有 s0 有 none 同 lr）：2e-3: s0 +63.6；4e-3: s0 +55.7。
  - 次指标 twin-free area（同一组格子）：g s0 +8.6, s1 +14.7, s2 +14.8，均值 +12.7，wins 3/3，d 3.59；final3 均值 -12.6。
  - LM 梯度占比（only-torus@1e-3，prior 开启期间的中位数）：s0 0.99, s1 0.99, s2 0.99。
  - read-out 规模（twin-free / twin / 合计）：s0 450 / 39 / 489, s1 449 / 42 / 491, s2 430 / 52 / 482。
- **hcube**（twin-free area，chance 3.1）：only-hcube@2e-3 各 seed area s0 90.5, s1 89.5, s2 90.3；none@1.6e-2 各 seed area s0 75.7, s1 68.3, s2 63.5。配对均值 prior 90.1 / none 69.1。判级依据：same sign, |mean| 20.92 > seed range 12.03。
  - only-hcube@2e-3 各 seed 水平（window 内 / window 外 / final3）：s0 94.2 / 88.5 / 99.0, s1 92.4 / 87.9 / 92.4, s2 92.1 / 89.3 / 99.1。
  - none@1.6e-2 各 seed 水平（window 内 / window 外 / final3）：s0 58.3 / 84.9 / 92.9, s1 39.0 / 83.8 / 93.9, s2 38.4 / 76.8 / 89.8。
  - 同 lr 增益（描述性，只有 s0 有 none 同 lr）：2e-3: s0 +35.0；4e-3: s0 +24.8。
  - 次指标 area (full set)（同一组格子）：g s0 +11.1, s1 +18.1, s2 +20.6，均值 +16.6，wins 3/3，d 3.35；final3 均值 +0.3。
  - LM 梯度占比（only-hcube@2e-3，prior 开启期间的中位数）：s0 1.00, s1 1.00, s2 1.00。
  - read-out 规模（twin-free / twin / 合计）：s0 66 / 302 / 368, s1 66 / 300 / 366, s2 72 / 298 / 370。
- **box5**（area (full set)，chance 4.0）：only-box5@1e-3 各 seed area s0 53.4；none@1.6e-2 各 seed area s0 20.0, s1 23.6, s2 16.9。配对均值 prior 53.4 / none 20.0。判级依据：1 seed。
  - only-box5@1e-3 各 seed 水平（window 内 / window 外 / final3）：s0 67.7 / 45.7 / 38.7。
  - none@1.6e-2 各 seed 水平（window 内 / window 外 / final3）：s0 11.4 / 24.6 / 23.3。
  - 同 lr 增益（描述性，只有 s0 有 none 同 lr）：2e-3: s0 +33.7；4e-3: s0 +38.0。
  - 次指标 twin-free area（同一组格子）：g s0 +33.4，均值 +33.4，wins 1/1，d —；final3 均值 +15.3。
  - LM 梯度占比（only-box5@1e-3，prior 开启期间的中位数）：s0 0.99。
  - read-out 规模（twin-free / twin / 合计）：s0 150 / 0 / 150, s1 153 / 0 / 153, s2 181 / 0 / 181。
- cyc12 / d6 / caesar 只有 s0：seed 0 上是 FLOOR（标签见表），没有复现。caesar 用的是真实字母 token，单独看。

**限定**
1. PROVISIONAL：缺 replication only-box5@1e-3 s1: MISSING；replication only-box5@1e-3 s2: MISSING。
2. torus：g 的符号不一致（字段 domains.torus.primary.window.in / .out.mean_g、domains.torus.primary.final3.mean_g、domains.torus.primary.mean_g）：prior window 内 +44.0（prior 臂更高，wins 3/3）；window 外 -4.0（none 更高，wins 0/3）；最后 3 次 eval -12.2（none 更高，wins 0/3）；area（全部 75 次 eval） +12.6（prior 臂更高，wins 3/3）。水平（各 seed 均值，window 内 → final3）：only-torus@1e-3 83.2 → 82.6（降 0.6），none@1.6e-2 39.2 → 94.8（升 55.5）。即 window 之后 prior 臂降、none@1.6e-2 升，变化主要来自 none@1.6e-2（|Δ| 55.5 vs 0.6）。
3. torus：k = 3，boot90 只有 10 种不同的重采样均值，区间粗糙（k = 2 时就是 [min g, max g]）。
4. hcube：k = 3，boot90 只有 10 种不同的重采样均值，区间粗糙（k = 2 时就是 [min g, max g]）。
5. box5：只有 1 个配对 seed（preliminary (1 seed)），没有 d 和 boot90。
6. P 和 N 都是在同一批 seed 上按均值挑出来的（两边都有 winner's curse）：P 一侧让 g 偏高，N 一侧让 g 偏低。
7. seed 1–2 没有 auditor 计数，twin-free 掩码只做了能做的断言；各 seed 的 twin-free n 已在上面列出。
8. 与 pick_pt6v2.py --allow_partial 交叉核对：EQUAL（84 个字段）；seed 0 与 screen read-out 一致性：EQUAL（42 行）。

**审查状态**：readout_ptdom3r.py e328078abe3c 待新开代码审查；探索档，没有预登记判定规则。

<!-- rendered from runs/ptdom3/tests/readout_rep_partial_20261119T1534.json (sha 22ec895d1fa3) by render_findings_ptdom3r.py bc6ec787a438 -->
