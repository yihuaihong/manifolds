# pt6v2 结果审查：all6full@1e-3 1×（V4，8 seed）对 none@3.2e-2 4×（V8，s16/17）（2026-11-23，exp-auditor，只读，未重跑）
这次比较的两臂：all6full = 六个先验开到训练结束；none×4 = 不加先验、训练 4 倍长。脚本是自己写的：strict 掩码按定义重写，题表顺序已用 run 自带的 times / nonid / negtwin_trained 掩码逐题核对；torus / box5 从 hits 重算后和日志里的值一致。没有 import 或 exec v8read.py / v4final.py。文件：`/scratch/yh6210/tmp_audit/pt6v2_full_vs_v8/{indep.py,indep_out.txt,sha.py}`。

## (1) 数字核对：一处写错（「best 85.1%」），其余全部复现
- 两臂 10 个数都复现了，按每臂 seed 平均的 final3 计：strict 量纲格 96.52 对 78.12，格（box5）91.43 对 50.75，torus 99.76 对 93.97，城市 96.87 对 92.85（v8read 记 92.8），国家 98.38 对 95.39。
- 运行状态：all6full 8 个 run 完整（75 次 eval，events [[916,1]]，check 全过）；none×4 两个 run 完整（294 次 eval，check 全过）。
- 同一个训练器（a398fccbfe93）。同 seed 两臂的 data_sha 在 74 个共有 eval（≤ 18250 步）上全部相等，所以到 1× 步数为止数据和初始化逐位相同，「跨批」只差 lr 曲线和卡型。
- 「8 个 seed 都高于它的两个 seed」成立：all6full 最低的 s22 是 85.5，none×4 最高的 s17 是 81.0。
- **写错的地方**：「best 85.1%」是两个 seed 各自峰值的平均（s16 84.5 @63000，s17 85.7 @55000）。单次 eval 的最好值是 **85.7%**，应改成 85.7。「任何一次 eval 都没到 96.5」成立（588 次 eval 中 0 次）。

## (2) 证据撑得住的最强说法
**量纲格和格：结论成立，而且可以写得更强。** 3.2e-2 是 none 在这两个概念上的 N_m（面积最优 lr，按同一批 seed 挑出），先验臂的 lr 又是没调过的最低档 1e-3，所以选 lr 这一步偏向 none，这个比较对结论是保守的。
- 量纲格在全部 held 题上也是 96.9 对 87.3，8 个 seed 都高于 none 的两个 seed。F113 需要的「只在 strict 集」限定（那里对比的是 window run，全部 held 题上 2× 就追平）在这里用不上。
- 格：none 训得越久也没起色（1× / 2× / 4× 为 55.3 / 42.9 / 50.8）。
- 「从来没到」可以放宽到所有不加先验的 run：6 个 lr × 8 seed（1×），加上 2× 和 4×，strict 量纲格在任何一次 eval 都没超过 87.9（这个最高值出现在 2×）。拿 none 的单次峰值去比 all6full 的 8-seed final3 均值，对 none 有利，所以这样说是公平的。但不能改成逐 seed 的说法：all6full s22 的 final3 是 85.5，低于 85.7。
- 完全分离：每个概念上 8 个 seed 都高于 2 个 seed。按排列检验，单侧精确 p = 1/C(10,2) = 0.022（每个概念单独算）。按卡型分开看也一样：A100 上 all6full s17–19 对 none s16，H100 上 s21–23 对 none s17，每个概念都完全分开。卡型只记次要。

**重要（范围上的假阳性）：torus、城市、国家这三项，4× 的对比测不到这句话想说的东西。**
- 3.2e-2 不是这三项的最优 lr：N_m 分别是 1.6e-2 / 4e-3 / 4e-3，final3 最好的 lr 三项都是 1.6e-2。
- none@1.6e-2 只训 1× 就已经和 none@3.2e-2 训 4× 一样好或更好：城市 93.3 对 92.8，国家 95.8 对 95.4，torus 93.3（s16/17 为 96.5）对 94.0。所以这三项上「训 4 倍长」什么也没加，none×4 不是最强的无先验基线。
- 这三项的证据应当用 F106 的结果：比每个 seed 最好的 none lr（1×）高 +2.5 / +2.1 / +4.1，8/8。
- 最优 lr 训到 4× 会不会追上，是未知数：3.2e-2 从 1× 到 4× 涨了 +2.8 / +2.6 / +2.5（s16/17），和 F106 的差距同一量级；国家按这个涨幅外推正好打平（95.8 + 2.6 ≈ 98.4）。
- 第二个 lr 的佐证：only-dim@2e-3×4 也没有这三项的先验。国家（95.5 / 96.6）和 torus（90.4 / 98.0）8 个 seed 都高于它；城市均值 94.8 对 96.9，但不是 8/8（all6full s17 是 95.47，only-dim s16 是 95.5）。
- 「every concept」只列了五个概念，数字题（98.3 对 49.5，同样 8/8）没列；按 F113 不得写 1 不列它是对的，但「every」会被读者按六个概念数。
- F113 现有的限定覆盖了 2 个 seed、单一 lr、拉长的 cosine 和卡型，没有覆盖「3.2e-2 对 torus、城市、国家不是最优 lr」（F113 只说了数字题）。所以要加一个范围限定，不需要更多。

**建议写法**（只带一个范围限定）：
"Kept on until the end, the six priors at the original length stay above the model without priors trained four times as long at its best rate for dimensions and the grid: strict dimensions 96.5% vs 78.1% (all held dimension questions 96.9% vs 87.3%) and the grid 91.4% vs 50.8%, each of the 8 seeds above both of its seeds; no run without priors, at any rate or length, reaches 88% on strict dimensions at any evaluation (4× best 85.7%). At that rate they also stay above on the torus (99.8% vs 94.0%), cities (96.9% vs 92.8%) and countries (98.4% vs 95.4%), but there the no-prior optimum is another rate, against which the margin at the original length is 2–4 points (F106)."
